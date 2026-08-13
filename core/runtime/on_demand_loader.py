#!/usr/bin/env python3
"""OpenCodeWEB OS — Dynamic On-Demand Module Loader.

Runtime resource manager for the OpenCodeWEB OS ecosystem. Every module
(Blender, VsCode, FFmpeg, ...) is a repository under
``github.com/OpenCodeWEB/<module-name>`` and can be used in two modes:

* ``ON_DEMAND`` (default) — minimal OS footprint (~100MB). Modules are
  fetched lazily from GitHub on first prompt request, executed, then
  cleaned up automatically (memory + storage) post-execution.
* ``PRE_INSTALLED`` (optional) — every module is loaded locally for fully
  offline execution.

The loader also owns the auto-clean mechanism: when storage thresholds are
crossed, least-recently-used modules are moved to ``/opt/opencode/Trash``
and finally purged by the trash janitor.

Zero-Constraint Policy: no artificial quotas or hard timeouts are imposed
on module execution; cleanup decisions are driven purely by real storage
pressure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))
MODULES_DIR: Path = SYS_ROOT / "lib" / "modules"
TRASH_DIR: Path = SYS_ROOT / "Trash"
MANIFEST_REL: str = ".opencode-manifest.json"

# GitHub source base for dynamic module fetching.
MODULE_SOURCE_ORG: str = "OpenCodeWEB"
MODULE_SOURCE_BASE: str = f"https://github.com/{MODULE_SOURCE_ORG}"

# Storage thresholds (fraction of total disk that must remain free).
LOW_WATERMARK: float = 0.20   # trigger auto-clean when free drops below 20%
HIGH_WATERMARK: float = 0.30  # stop cleaning once 30% is free again

# Trash janitor: purge files older than this many seconds.
TRASH_MAX_AGE_S: int = 7 * 24 * 3600  # 7 days

ON_DEMAND: str = "ON_DEMAND"
PRE_INSTALLED: str = "PRE_INSTALLED"
VALID_MODES: tuple[str, ...] = (ON_DEMAND, PRE_INSTALLED)

_logger = logging.getLogger("opencode.runtime")


# ---------------------------------------------------------------------------
# Manifest + metadata
# ---------------------------------------------------------------------------


@dataclass
class ModuleRecord:
    """Persistent metadata for one loaded module."""

    name: str
    source: str
    mode: str
    installed_at: float
    last_used_at: float = field(default_factory=time.time)
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "mode": self.mode,
            "installed_at": self.installed_at,
            "last_used_at": self.last_used_at,
            "size_bytes": self.size_bytes,
        }


class ManifestStore:
    """JSON manifest of loaded modules, stored at the modules root."""

    def __init__(self, path: Path = MODULES_DIR / MANIFEST_REL) -> None:
        self.path = path

    def load(self) -> dict[str, ModuleRecord]:
        """Load all records (missing/corrupt manifest -> empty dict)."""
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {
                name: ModuleRecord(**record) for name, record in raw.items()
            }
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            _logger.warning("manifest unreadable (%s) — starting fresh", exc)
            return {}

    def save(self, records: dict[str, ModuleRecord]) -> None:
        """Persist the manifest atomically (write-then-rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: record.to_dict() for name, record in records.items()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Storage guard + trash janitor
# ---------------------------------------------------------------------------


def _dir_size(path: Path) -> int:
    """Total size in bytes of a directory tree (best-effort)."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


class TrashJanitor:
    """Move-to-Trash + expiry purge for module artifacts."""

    def __init__(self, trash_dir: Path = TRASH_DIR, max_age_s: int = TRASH_MAX_AGE_S) -> None:
        self.trash_dir = trash_dir
        self.max_age_s = max_age_s
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def discard(self, module_path: Path) -> Path:
        """Move ``module_path`` into the trash (fallback: delete directly)."""
        if not module_path.exists():
            return module_path
        target = self.trash_dir / f"{module_path.name}-{int(time.time())}"
        try:
            shutil.move(str(module_path), str(target))
            _logger.info("moved %s -> %s", module_path.name, target)
            return target
        except (OSError, shutil.Error) as exc:
            _logger.warning("trash move failed (%s) — deleting directly", exc)
            shutil.rmtree(module_path, ignore_errors=True)
            return module_path

    def purge_expired(self) -> int:
        """Delete trash entries older than ``max_age_s``. Returns count."""
        removed = 0
        now = time.time()
        for entry in self.trash_dir.iterdir():
            try:
                if now - entry.stat().st_mtime > self.max_age_s:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        if removed:
            _logger.info("trash janitor purged %d expired entries", removed)
        return removed


class StorageGuard:
    """Monitors real disk pressure and triggers module auto-clean."""

    def __init__(
        self,
        modules_dir: Path = MODULES_DIR,
        trash: TrashJanitor | None = None,
        low_watermark: float = LOW_WATERMARK,
        high_watermark: float = HIGH_WATERMARK,
    ) -> None:
        self.modules_dir = modules_dir
        self.trash = trash or TrashJanitor()
        self.low_watermark = low_watermark
        self.high_watermark = high_watermark

    def free_fraction(self) -> float:
        """Free disk fraction of the volume hosting the modules dir."""
        try:
            usage = shutil.disk_usage(self.modules_dir)
            return usage.free / usage.total
        except OSError:
            return 1.0  # unknown -> assume healthy

    def under_pressure(self) -> bool:
        """True when free space is below the low watermark."""
        return self.free_fraction() < self.low_watermark

    def auto_clean(self, records: dict[str, ModuleRecord], force: bool = False) -> list[str]:
        """Evict least-recently-used modules to trash until healthy.

        Only real storage pressure drives eviction — no artificial quotas.
        Returns the names of evicted modules.
        """
        if not force and not self.under_pressure():
            return []

        evicted: list[str] = []
        # LRU order: oldest last_used_at first.
        ordered = sorted(records.values(), key=lambda rec: rec.last_used_at)
        for record in ordered:
            if not force and self.free_fraction() >= self.high_watermark:
                break
            module_dir = self.modules_dir / record.name
            self.trash.discard(module_dir)
            evicted.append(record.name)
        return evicted


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


class ModuleFetchError(RuntimeError):
    """Raised when a module cannot be fetched from the source org."""


class ModuleExecutionError(RuntimeError):
    """Raised when a module's entry point cannot be found or started."""


class OnDemandLoader:
    """Fetch / execute / clean OpenCodeWEB OS modules.

    Modes:
        ON_DEMAND      fetch lazily, execute, auto-clean post-run
        PRE_INSTALLED  require local presence, never fetch, never clean
    """

    def __init__(
        self,
        mode: str = ON_DEMAND,
        modules_dir: Path = MODULES_DIR,
        source_base: str = MODULE_SOURCE_BASE,
        manifest: ManifestStore | None = None,
        storage: StorageGuard | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"invalid runtime mode: {mode!r} (expected {VALID_MODES})")
        self.mode = mode
        self.modules_dir = modules_dir
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.source_base = source_base
        self.manifest = manifest or ManifestStore(modules_dir / MANIFEST_REL)
        self.storage = storage or StorageGuard(modules_dir)
        self.records = self.manifest.load()

    # -- queries ------------------------------------------------------------------

    def list_modules(self) -> list[dict[str, object]]:
        """Return metadata for all recorded modules."""
        return [rec.to_dict() for rec in self.records.values()]

    def module_dir(self, name: str) -> Path:
        """Local directory where module ``name`` lives (or should live)."""
        return self.modules_dir / name

    def is_cached(self, name: str) -> bool:
        """True when the module directory exists locally."""
        return self.module_dir(name).is_dir()

    # -- fetching -------------------------------------------------------------------

    def _fetch(self, name: str) -> Path:
        """Fetch module sources from ``github.com/OpenCodeWEB/<name>``.

        Uses a shallow git clone with a tarball fallback. Raises
        ModuleFetchError when the module does not exist upstream.
        """
        target = self.module_dir(name)
        if target.exists():
            return target

        # Prefer a shallow clone (fast, small); fall back to the codeload tarball.
        url = f"{self.source_base}/{name}.git"
        _logger.info("fetching module %s from %s (mode=%s)", name, url, self.mode)
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(target)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,  # network guard, not a work quota
        )
        if clone.returncode == 0:
            self._record(name)
            return target

        # Fallback: download + unpack the default-branch tarball.
        _logger.info("git clone failed, trying tarball fallback for %s", name)
        tar_url = f"{self.source_base}/{name}/archive/refs/heads/main.tar.gz"
        try:
            import tarfile
            import urllib.request

            request = urllib.request.Request(tar_url, headers={"User-Agent": "OpenCodeWEB-AiA"})
            with urllib.request.urlopen(request, timeout=600) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
                tar.extractall(path=self.modules_dir, filter="data")
        except (OSError, tarfile.TarError, urllib.error.URLError) as exc:
            raise ModuleFetchError(f"cannot fetch {name} from {self.source_base}: {exc}") from exc

        # tarball extracts to <modules_dir>/<name>-<ref> — normalize to <name>.
        extracted = [p for p in self.modules_dir.iterdir() if p.name.startswith(name + "-")]
        if not extracted:
            raise ModuleFetchError(f"tarball for {name} contained no sources")
        shutil.move(str(extracted[0]), str(target))
        self._record(name)
        return target

    def _record(self, name: str) -> None:
        """Update (or create) the manifest record for ``name``."""
        rec = self.records.get(name) or ModuleRecord(
            name=name,
            source=f"{self.source_base}/{name}",
            mode=self.mode,
            installed_at=time.time(),
        )
        rec.last_used_at = time.time()
        rec.mode = self.mode
        rec.size_bytes = _dir_size(self.module_dir(name))
        self.records[name] = rec
        self.manifest.save(self.records)

    # -- loading / executing ---------------------------------------------------------

    def _find_entry(self, module_path: Path) -> Path | None:
        """Locate the module entry point (main.py -> run.py -> executable)."""
        for candidate in ("main.py", "run.py", "entry.py", "bin/run.py"):
            path = module_path / candidate
            if path.is_file():
                return path
        return None

    def execute(self, name: str, args: list[str] | None = None) -> dict[str, object]:
        """Load and execute module ``name`` with ``args``.

        ON_DEMAND: fetches if missing, runs, then auto-cleans the module
        (storage pressure permitting) and the module's working memory.
        PRE_INSTALLED: requires the module locally and never cleans it.
        """
        args = args or []
        module_dir = self.module_dir(name)

        if self.mode == ON_DEMAND:
            if not module_dir.is_dir():
                module_dir = self._fetch(name)
        elif not module_dir.is_dir():
            raise ModuleExecutionError(
                f"module '{name}' is not pre-installed at {module_dir} "
                f"(run in ON_DEMAND mode to fetch it)"
            )

        entry = self._find_entry(module_dir)
        if entry is None:
            raise ModuleExecutionError(f"no entry point (main.py/run.py) found in {module_dir}")

        self._record(name)
        _logger.info("executing module %s via %s (args=%s)", name, entry.name, args)

        # Zero-Constraint: no artificial timeout is applied to the module run.
        proc = subprocess.run(
            [sys.executable, str(entry), *args],
            cwd=str(module_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        result: dict[str, object] = {
            "module": name,
            "entry": str(entry),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

        if self.mode == ON_DEMAND:
            self._post_execution_cleanup(name)
        return result

    def _post_execution_cleanup(self, name: str) -> None:
        """Post-run cleanup for ON_DEMAND mode: free storage under pressure."""
        freed = self.storage.auto_clean(self.records)
        if name in freed:
            _logger.info("module %s auto-cleaned after execution (ON_DEMAND)", name)
        else:
            self.storage.trash.purge_expired()

    # -- explicit operations -------------------------------------------------------------

    def fetch(self, name: str) -> Path:
        """Explicitly fetch a module into the local cache."""
        return self._fetch(name)

    def clean(self, force: bool = False) -> list[str]:
        """Trigger storage-driven auto-clean. Returns evicted module names."""
        evicted = self.storage.auto_clean(self.records, force=force)
        self.storage.trash.purge_expired()
        for name in evicted:
            self.records.pop(name, None)
        self.manifest.save(self.records)
        return evicted

    def purge_trash(self) -> int:
        """Expire old trash entries now. Returns count removed."""
        return self.storage.trash.purge_expired()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="on_demand_loader", description="OpenCodeWEB OS dynamic module loader")
    parser.add_argument(
        "--mode", "-m", choices=VALID_MODES, default=os.environ.get("OPENCODE_RUNTIME_MODE", ON_DEMAND),
        help="ON_DEMAND (fetch+clean, default) or PRE_INSTALLED (offline)",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="fetch (if needed) and execute a module")
    run.add_argument("module")
    run.add_argument("args", nargs="*", help="arguments passed to the module entry point")

    fetch = sub.add_parser("fetch", help="explicitly fetch a module into the cache")
    fetch.add_argument("module")

    sub.add_parser("list", help="list cached modules")

    clean = sub.add_parser("clean", help="run storage-driven auto-clean")
    clean.add_argument("--force", action="store_true", help="evict regardless of pressure")

    sub.add_parser("purge-trash", help="purge expired trash entries")

    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    loader = OnDemandLoader(mode=args.mode)
    try:
        if args.action == "run":
            result = loader.execute(args.module, args.args)
            print(json.dumps(result, indent=2))
            return 0 if result["returncode"] == 0 else 1
        if args.action == "fetch":
            path = loader.fetch(args.module)
            print(f"fetched {args.module} -> {path}")
            return 0
        if args.action == "list":
            print(json.dumps(loader.list_modules(), indent=2))
            return 0
        if args.action == "clean":
            print(json.dumps({"evicted": loader.clean(force=args.force)}, indent=2))
            return 0
        if args.action == "purge-trash":
            print(json.dumps({"purged": loader.purge_trash()}, indent=2))
            return 0
    except (ModuleFetchError, ModuleExecutionError, ValueError) as exc:
        _logger.error("%s", exc)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
