#!/usr/bin/env python3
"""OpenCodeWEB OS — Core Kernel Bootstrapper.

The kernel is the first process of the OpenCodeWEB OS runtime. It:

1. Initializes the OpenCodeWEB OS environment (system root, logs, IPC bus).
2. Probes hardware capabilities (CPU threads, RAM, Vulkan, CUDA).
3. Starts the IPC server (Unix domain socket on POSIX, loopback TCP on Windows).
4. Spawns the AiA Master Intelligence Engine as a background daemon.
5. Exposes a CLI loop accepting prompt triggers and system commands.

Zero-Constraint Policy: this kernel imposes *no* artificial rate limits,
token caps, or hard timeouts on any subsystem it manages.

Ecosystem:
    - Founders:           ABsUP & ABsUPs
    - Edge gateway:       https://opencodeweb.xup.workers.dev
    - Portal:             https://pocwu.pages.dev
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# System root. Override with OPENCODE_ROOT for non-Linux development hosts.
SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))

IPC_SOCK_PATH: Path = Path(os.environ.get("OPENCODE_IPC_SOCK", "/tmp/opencode-ipc.sock"))
# Loopback TCP fallback for platforms without AF_UNIX (e.g. Windows dev hosts).
IPC_TCP_HOST: str = "127.0.0.1"
IPC_TCP_PORT: int = int(os.environ.get("OPENCODE_IPC_PORT", "8790"))

AIA_ENGINE_REL: str = "core/aia/aia_engine.py"
PID_FILE_REL: str = "run/aia.pid"
LOG_DIR_REL: str = "log"

PROTOCOL_VERSION: int = 1
SOCKET_BACKLOG: int = 16
HEARTBEAT_INTERVAL_S: float = 30.0

_logger = logging.getLogger("opencode.kernel")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Snapshot of the host's capabilities, collected at boot time."""

    platform: str = platform.platform()
    python: str = platform.python_version()
    cpu_threads: int = os.cpu_count() or 1
    ram_bytes: int = 0
    vulkan: bool = False
    cuda: bool = False
    disk_free_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Hardware probing
# ---------------------------------------------------------------------------


def _read_proc_meminfo() -> int:
    """Read total RAM in bytes from /proc/meminfo (Linux only)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    except (OSError, ValueError):
        pass
    return 0


def _sysctl_hw_memsize() -> int:
    """Read total RAM on macOS via sysctl."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def _command_exists(name: str) -> bool:
    """Return True if ``name`` is on PATH."""
    return shutil.which(name) is not None


def detect_hardware() -> HardwareProfile:
    """Probe CPU threads, RAM, Vulkan and CUDA availability.

    Every probe is defensive: a missing tool or failed subprocess simply
    reports ``False``/``0`` instead of crashing the kernel.
    """
    profile = HardwareProfile()

    # RAM — per-platform readers, falling back to a ctypes call on Windows.
    if sys.platform.startswith("linux"):
        profile.ram_bytes = _read_proc_meminfo()
    elif sys.platform == "darwin":
        profile.ram_bytes = _sysctl_hw_memsize()
    elif sys.platform == "win32":
        try:  # GlobalMemoryStatusEx via ctypes — no external dependency.
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            profile.ram_bytes = int(stat.ullTotalPhys)
        except (OSError, AttributeError):
            profile.ram_bytes = 0

    # Vulkan — presence of vulkaninfo (or vulkaninfoSDK) is a good proxy.
    profile.vulkan = any(
        _command_exists(cmd) for cmd in ("vulkaninfo", "vulkaninfoSDK")
    )

    # CUDA — nvidia-smi present and responding.
    if _command_exists("nvidia-smi"):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            profile.cuda = probe.returncode == 0 and bool(probe.stdout.strip())
        except subprocess.SubprocessError:
            profile.cuda = False

    # Disk headroom for the dynamic module cache.
    try:
        profile.disk_free_bytes = shutil.disk_usage(SYS_ROOT).free
    except OSError:
        profile.disk_free_bytes = 0

    return profile


# ---------------------------------------------------------------------------
# IPC server (Unix domain socket / TCP fallback)
# ---------------------------------------------------------------------------


def _ipc_family() -> socket.AddressFamily:
    """Choose AF_UNIX where available, otherwise AF_INET."""
    return socket.AF_UNIX if hasattr(socket, "AF_UNIX") else socket.AF_INET


class IpcServer:
    """JSON-line IPC server for OpenCodeWEB OS.

    Handlers are registered per command name and dispatched on a
    per-connection worker thread. The wire format is one JSON object per
    line: ``{"cmd": str, "payload": object, "id": str}`` and the reply is
    ``{"ok": bool, "result": object, "error": str | None, "id": str}``.
    """

    def __init__(self, sock_path: Path = IPC_SOCK_PATH) -> None:
        self.sock_path = Path(sock_path)
        self._server: socket.socket | None = None
        self._handlers: dict[str, object] = {}
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    # -- handler registration ----------------------------------------------

    def register(self, cmd: str, handler: object) -> None:
        """Register a callable ``handler(cmd, payload) -> object``."""
        self._handlers[cmd] = handler

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Bind and listen; spawn the accept loop thread."""
        family = _ipc_family()
        self._server = socket.socket(family, socket.SOCK_STREAM)

        if family == socket.AF_UNIX:
            # Remove a stale socket file left by an unclean shutdown.
            self.sock_path.unlink(missing_ok=True)
            self._server.bind(str(self.sock_path))
        else:  # pragma: no cover - Windows dev fallback
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((IPC_TCP_HOST, IPC_TCP_PORT))

        self._server.listen(SOCKET_BACKLOG)
        self._running.set()
        self._thread = threading.Thread(target=self._accept_loop, name="ipc-accept", daemon=True)
        self._thread.start()
        _logger.info("IPC server listening at %s", self._endpoint())

    def _endpoint(self) -> str:
        if _ipc_family() == socket.AF_UNIX:
            return str(self.sock_path)
        return f"{IPC_TCP_HOST}:{IPC_TCP_PORT}"  # pragma: no cover

    def stop(self) -> None:
        """Stop accepting connections and release the socket."""
        self._running.clear()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if _ipc_family() == socket.AF_UNIX:
            self.sock_path.unlink(missing_ok=True)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- accept / dispatch ---------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._server is not None
        while self._running.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                break
            worker = threading.Thread(
                target=self._handle_connection, args=(conn,), name="ipc-worker", daemon=True
            )
            worker.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Read JSON lines from a connection and reply per command."""
        with conn:
            buffer = ""
            while self._running.is_set():
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    reply = self._dispatch(line)
                    try:
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                    except OSError:
                        return

    def _dispatch(self, line: str) -> dict[str, object]:
        """Parse one request line and route it to a registered handler."""
        try:
            request = json.loads(line)
            cmd = str(request.get("cmd", ""))
            payload = request.get("payload", {})
            req_id = request.get("id")
        except (json.JSONDecodeError, AttributeError):
            return {"ok": False, "result": None, "error": "malformed request", "id": None}

        handler = self._handlers.get(cmd)
        if handler is None:
            return {"ok": False, "result": None, "error": f"unknown command: {cmd}", "id": req_id}

        try:
            result = handler(cmd, payload)  # type: ignore[operator]
            return {"ok": True, "result": result, "error": None, "id": req_id}
        except Exception as exc:  # handler errors must not kill the server
            _logger.exception("handler %s failed", cmd)
            return {"ok": False, "result": None, "error": str(exc), "id": req_id}


# ---------------------------------------------------------------------------
# AiA engine daemon management
# ---------------------------------------------------------------------------


class AiADaemon:
    """Spawn / supervise the AiA Master Intelligence Engine process."""

    def __init__(self, sys_root: Path = SYS_ROOT) -> None:
        self.sys_root = sys_root
        self.pid_file = sys_root / PID_FILE_REL
        self.engine_script = sys_root / AIA_ENGINE_REL

    def is_running(self) -> bool:
        """Return True when the recorded PID is alive."""
        pid = self.read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def read_pid(self) -> int | None:
        """Read the daemon PID file, if present."""
        try:
            return int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def spawn(self) -> int:
        """Launch the AiA engine detached from the kernel process.

        Returns the new process PID. Raises RuntimeError when the engine
        script is missing or the process cannot be started.
        """
        if not self.engine_script.exists():
            raise RuntimeError(f"AiA engine not found: {self.engine_script}")

        self.sys_root.mkdir(parents=True, exist_ok=True)
        (self.sys_root / LOG_DIR_REL).mkdir(parents=True, exist_ok=True)
        log_path = self.sys_root / LOG_DIR_REL / "aia-engine.log"

        env = dict(os.environ)
        env["OPENCODE_ROOT"] = str(self.sys_root)
        env["OPENCODE_IPC_SOCK"] = str(IPC_SOCK_PATH)
        env["PYTHONUNBUFFERED"] = "1"

        log_handle = log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(self.engine_script), "--daemon"],
                cwd=str(self.sys_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # detach from the kernel's process group
            )
        except OSError as exc:
            log_handle.close()
            raise RuntimeError(f"failed to spawn AiA engine: {exc}") from exc

        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(proc.pid))
        return proc.pid

    def stop(self) -> bool:
        """Terminate the daemon (SIGTERM, then SIGKILL after a grace period)."""
        pid = self.read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except OSError:
            return False
        # Zero-Constraint note: the grace period is a *resource* guard, not
        # an artificial work timeout; it only bounds shutdown waiting.
        for _ in range(50):
            if not self.is_running():
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        return True


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class OpenCodeKernel:
    """Top-level kernel: environment, hardware, IPC and AiA supervision."""

    def __init__(self, sys_root: Path = SYS_ROOT) -> None:
        self.sys_root = sys_root
        self.ipc = IpcServer()
        self.aia = AiADaemon(sys_root)
        self.profile = detect_hardware()
        self.booted_at: float = 0.0
        self._shutdown = threading.Event()

    # -- bootstrap ------------------------------------------------------------

    def initialize(self) -> None:
        """Create the OpenCodeWEB OS directory skeleton."""
        for rel in ("bin", "core/aia", "core/runtime", "core/ipc", "lib/modules", "run", "log", "Trash"):
            (self.sys_root / rel).mkdir(parents=True, exist_ok=True)
        _logger.info("OpenCodeWEB OS root ready at %s", self.sys_root)

    def register_handlers(self) -> None:
        """Wire the IPC command table."""
        self.ipc.register("ping", lambda _c, _p: {"pong": True, "protocol": PROTOCOL_VERSION})
        self.ipc.register("status", self._cmd_status)
        self.ipc.register("hardware", self._cmd_hardware)
        self.ipc.register("shutdown", self._cmd_shutdown)
        # Prompt triggers are forwarded to the AiA engine daemon by the CLI
        # client; the kernel itself stays a thin transport.

    def _cmd_status(self, _cmd: str, _payload: object) -> dict[str, object]:
        return {
            "booted_at": self.booted_at,
            "uptime_s": time.time() - self.booted_at,
            "aia_running": self.aia.is_running(),
            "aia_pid": self.aia.read_pid(),
            "ipc_endpoint": self.ipc._endpoint(),  # noqa: SLF001 - internal status
        }

    def _cmd_hardware(self, _cmd: str, _payload: object) -> dict[str, object]:
        return self.profile.to_dict()

    def _cmd_shutdown(self, _cmd: str, _payload: object) -> dict[str, object]:
        self._shutdown.set()
        return {"shutdown": True}

    # -- lifecycle -------------------------------------------------------------

    def start(self, spawn_aia: bool = True) -> None:
        """Boot the kernel: env, hardware, IPC, then the AiA daemon."""
        self.initialize()
        self.profile = detect_hardware()
        _logger.info(
            "Hardware: %d threads, %.1f GiB RAM, Vulkan=%s, CUDA=%s",
            self.profile.cpu_threads,
            self.profile.ram_bytes / (1024**3),
            self.profile.vulkan,
            self.profile.cuda,
        )
        self.register_handlers()
        self.ipc.start()
        self.booted_at = time.time()

        if spawn_aia:
            if self.aia.is_running():
                _logger.info("AiA engine already running (pid %s)", self.aia.read_pid())
            else:
                pid = self.aia.spawn()
                _logger.info("AiA engine spawned (pid %d)", pid)

    def wait(self) -> None:
        """Block until a shutdown is requested."""
        try:
            while not self._shutdown.wait(timeout=1.0):
                pass
        except KeyboardInterrupt:
            _logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful teardown: stop IPC, optionally stop the AiA daemon."""
        self.ipc.stop()
        if self.aia.is_running():
            self.aia.stop()
        _logger.info("OpenCodeWEB OS kernel stopped")


# ---------------------------------------------------------------------------
# CLI client helpers
# ---------------------------------------------------------------------------


def _client_socket() -> socket.socket:
    """Open a connection to the running kernel."""
    family = _ipc_family()
    client = socket.socket(family, socket.SOCK_STREAM)
    if family == socket.AF_UNIX:
        client.connect(str(IPC_SOCK_PATH))
    else:  # pragma: no cover - Windows dev fallback
        client.connect((IPC_TCP_HOST, IPC_TCP_PORT))
    return client


def send_command(cmd: str, payload: object = None, timeout_s: float = 10.0) -> dict[str, object]:
    """Send a JSON-line command to the kernel and await the reply."""
    request = {"cmd": cmd, "payload": payload or {}, "id": f"cli-{time.time_ns()}"}
    with _client_socket() as client:
        client.settimeout(timeout_s)
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        reply_line = client.recv(65536).decode("utf-8", errors="replace")
    if not reply_line.strip():
        raise ConnectionError("empty reply from kernel")
    reply = json.loads(reply_line)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or "kernel returned failure")
    return reply


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------


def interactive_loop() -> None:
    """REPL: prompt triggers are forwarded to the AiA engine; ``/`` commands
    are interpreted by the kernel client."""
    print("OpenCodeWEB OS kernel CLI — type a prompt or a /command. '/help' for help.")
    while True:
        try:
            line = input("opencode> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            if cmd == "/exit" or cmd == "/quit":
                print("bye")
                return
            if cmd == "/help":
                print(
                    "/status  /hardware  /aia <prompt>  /shutdown  /exit\n"
                    "Any other input is treated as a prompt for AiA."
                )
                continue
            if cmd == "/status":
                print(json.dumps(send_command("status"), indent=2))
                continue
            if cmd == "/hardware":
                print(json.dumps(send_command("hardware"), indent=2))
                continue
            if cmd == "/shutdown":
                print(json.dumps(send_command("shutdown"), indent=2))
                return
            if cmd == "/aia":
                line = rest  # fall through to prompt forwarding
            else:
                print(f"unknown command: {cmd}")
                continue

        # Prompt trigger -> forward to the AiA engine through the kernel.
        try:
            reply = send_command("aia.prompt", {"text": line, "author": "ABsUP"})
            print(json.dumps(reply.get("result"), indent=2))
        except (ConnectionError, RuntimeError) as exc:
            print(f"[kernel] {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode-kernel",
        description="OpenCodeWEB OS core kernel bootstrapper (ABsUP & ABsUPs).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("start", help="boot kernel, IPC server and AiA daemon")
    sub.add_parser("status", help="query kernel status")
    sub.add_parser("hardware", help="print hardware capabilities")
    sub.add_parser("shutdown", help="gracefully stop kernel and AiA daemon")
    sub.add_parser("shell", help="interactive CLI loop")

    parser.add_argument("--no-aia", action="store_true", help="start kernel without spawning AiA")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    try:
        if args.action == "start":
            kernel = OpenCodeKernel()
            kernel.start(spawn_aia=not args.no_aia)
            kernel.wait()
            return 0

        if args.action == "status":
            print(json.dumps(send_command("status"), indent=2))
            return 0

        if args.action == "hardware":
            print(json.dumps(detect_hardware().to_dict(), indent=2))
            return 0

        if args.action == "shutdown":
            print(json.dumps(send_command("shutdown"), indent=2))
            return 0

        if args.action == "shell":
            interactive_loop()
            return 0
    except (ConnectionError, RuntimeError) as exc:
        _logger.error("kernel operation failed: %s", exc)
        return 1

    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
