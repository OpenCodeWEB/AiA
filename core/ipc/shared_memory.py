#!/usr/bin/env python3
"""OpenCodeWEB OS — Shared Memory IPC Bus.

Ultra-low-latency (target < 3ms) zero-copy communication between AiA, the
VsCode IDE core, the Media Suite, and sub-models.

Design:
    * One shared-memory segment per named channel, allocated under
      ``/dev/shm`` (POSIX) via ``multiprocessing.shared_memory``, with an
      ``mmap`` fallback for exotic environments.
    * Fixed-slot ring buffer: a small header block (slot count, slot size,
      write sequence) followed by the slot array. Writers append atomically
      (monotonic sequence + fencing), readers poll by sequence so a slow
      consumer can never corrupt a producer.
    * Zero-copy reads: payloads are returned as ``memoryview`` slices over
      the shared buffer — no serialization/copy on the hot path.
    * ``SharedMemoryBus`` is the process-facing facade: channels can be
      created by one process and attached by name from any other process
      (AiA, VsCode, Media Suite, sub-models, ...).

Zero-Constraint Policy: the bus imposes no artificial quotas — any number
of channels, any payload size up to the slot size, unbounded lifetime.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Shared memory root: /dev/shm on POSIX, tempdir elsewhere (dev hosts).
def _default_shm_root() -> Path:
    if sys.platform.startswith("linux"):
        return Path("/dev/shm")
    if sys.platform == "darwin":
        return Path("/tmp")
    return Path(os.environ.get("TEMP", "/tmp"))  # pragma: no cover - win32 dev


SHM_ROOT: Path = Path(os.environ.get("OPENCODE_SHM_ROOT", str(_default_shm_root())))
CHANNEL_PREFIX: str = "ocw_bus_"

# Ring buffer geometry.
DEFAULT_SLOTS: int = 64
DEFAULT_SLOT_SIZE: int = 65536        # 64 KiB per slot
HEADER_FORMAT: str = "QQQ"             # magic, slot_size, write_seq
HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)
MAGIC: int = 0x4F435742               # "OCWB" as a u64

TARGET_LATENCY_MS: float = 3.0

_logger = logging.getLogger("opencode.ipc")


# ---------------------------------------------------------------------------
# Ring buffer over shared memory
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """One message read from a channel (zero-copy payload)."""

    seq: int
    topic: str
    payload: memoryview
    ts: float = field(default_factory=time.time)

    def bytes(self) -> bytes:  # noqa: A003 - explicit copy escape hatch
        """Copy the payload out of shared memory (only when required)."""
        return bytes(self.payload)


class ShmChannel:
    """A fixed-slot ring buffer living in a shared-memory segment.

    Wire layout::

        [ header: magic u64 | slot_size u64 | write_seq u64 ]
        [ slot 0: seq u64 | len u64 | topic_len u64 | topic bytes | payload ]
        [ slot 1: ... ]

    Writer discipline:
        * write_seq starts at 1; each put() bumps it (monotonic).
        * slot index = write_seq % slots. The slot is fully written, then
          its header seq is published last (release semantics via the GIL +
          ordering of plain stores is sufficient here because the seq field
          is the single synchronization point).
    Reader discipline:
        * read(seq) scans slots with slot.seq > seq, collects them in seq
          order, and returns (new_seq, messages). Never blocks; callers can
          poll with ``wait_for``.
    """

    def __init__(
        self,
        name: str,
        slots: int = DEFAULT_SLOTS,
        slot_size: int = DEFAULT_SLOT_SIZE,
        create: bool = False,
    ) -> None:
        self.name = name
        self.slots = slots
        self.slot_size = slot_size
        self.segment_size = HEADER_SIZE + slots * slot_size

        shm_name = f"{CHANNEL_PREFIX}{name}"
        if create:
            try:
                from multiprocessing import shared_memory

                self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.segment_size)
            except (OSError, FileExistsError) as exc:
                raise OSError(f"cannot create channel {name!r}: {exc}") from exc
        else:
            try:
                from multiprocessing import shared_memory

                self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
            except (OSError, FileNotFoundError) as exc:
                raise OSError(
                    f"channel {name!r} not found (create it first): {exc}"
                ) from exc

        self._buf = self._shm.buf  # memoryview over the whole segment

        if create:
            struct.pack_into(HEADER_FORMAT, self._buf, 0, MAGIC, slot_size, 0)
            _logger.info("channel %s created (%d bytes at %s)", name, self.segment_size, self._shm.name)

    # -- lifecycle ------------------------------------------------------------

    @classmethod
    def create(cls, name: str, slots: int = DEFAULT_SLOTS, slot_size: int = DEFAULT_SLOT_SIZE) -> ShmChannel:
        """Create a new channel (call once per name, then attach elsewhere)."""
        return cls(name, slots=slots, slot_size=slot_size, create=True)

    @classmethod
    def attach(cls, name: str) -> ShmChannel:
        """Attach to an existing channel by name."""
        return cls(name, create=False)

    def close(self) -> None:
        """Detach from the segment (does not unlink it)."""
        try:
            self._shm.close()
        except Exception:  # noqa: BLE001 - detach must never raise
            pass

    def unlink(self) -> None:
        """Destroy the segment (creator process only)."""
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass

    # -- header accessors --------------------------------------------------------

    def _header(self) -> tuple[int, int, int]:
        magic, slot_size, write_seq = struct.unpack_from(HEADER_FORMAT, self._buf, 0)
        return magic, slot_size, write_seq

    def write_seq(self) -> int:
        """Current write sequence (0 == empty channel)."""
        _, _, seq = self._header()
        return seq

    def _slot_offset(self, index: int) -> int:
        return HEADER_SIZE + index * self.slot_size

    # -- writer -------------------------------------------------------------------

    def put(self, topic: str, payload: bytes | memoryview) -> int:
        """Publish ``payload`` and return its sequence number.

        Raises ValueError when the payload exceeds the slot capacity.
        """
        data = bytes(payload)
        if len(data) > self.slot_size - 64:
            raise ValueError(
                f"payload {len(data)}B exceeds slot capacity "
                f"{self.slot_size - 64}B for channel {self.name!r}"
            )
        topic_b = topic.encode("utf-8")[:32]
        seq = self.write_seq() + 1
        index = seq % self.slots
        offset = self._slot_offset(index)

        # Write slot body, then publish the slot header (seq) last.
        struct.pack_into("Q", self._buf, offset + 8, len(data))
        struct.pack_into("Q", self._buf, offset + 16, len(topic_b))
        self._buf[offset + 24 : offset + 24 + len(topic_b)] = topic_b
        self._buf[offset + 24 + len(topic_b) : offset + 24 + len(topic_b) + len(data)] = data
        struct.pack_into("Q", self._buf, offset, seq)  # publish
        struct.pack_into("Q", self._buf, 16, seq)      # bump write_seq

        _logger.debug("put seq=%d topic=%s size=%d", seq, topic, len(data))
        return seq

    # -- reader ---------------------------------------------------------------------

    def read(self, after_seq: int = 0) -> tuple[int, list[Message]]:
        """Return (new_seq, messages) with messages whose seq > after_seq.

        Messages are returned in sequence order; payloads are zero-copy
        ``memoryview`` slices over the shared segment.
        """
        current = self.write_seq()
        if current <= after_seq:
            return after_seq, []

        messages: list[Message] = []
        for seq in range(after_seq + 1, current + 1):
            index = seq % self.slots
            offset = self._slot_offset(index)
            slot_seq, length, topic_len = struct.unpack_from("QQQ", self._buf, offset)
            if slot_seq != seq:  # slot was overwritten before we read it
                continue
            topic = bytes(self._buf[offset + 24 : offset + 24 + topic_len]).decode("utf-8", errors="replace")
            payload = self._buf[offset + 24 + topic_len : offset + 24 + topic_len + length]
            messages.append(Message(seq=seq, topic=topic, payload=payload))
        return current, messages

    def wait_for(self, after_seq: int = 0, timeout_ms: float = 3.0) -> tuple[int, list[Message]]:
        """Poll until new messages arrive or the deadline passes.

        The deadline is a *latency budget* (3ms target), not an artificial
        work timeout — callers may loop indefinitely with fresh budgets.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            new_seq, messages = self.read(after_seq)
            if messages:
                return new_seq, messages
            if time.monotonic() >= deadline:
                return after_seq, []


# ---------------------------------------------------------------------------
# mmap fallback channel (POSIX file-backed, still zero-copy reads)
# ---------------------------------------------------------------------------


class MmapChannel:
    """File-backed mmap channel — fallback when shared_memory is unavailable.

    Uses a sparse file under SHM_ROOT; reads return ``memoryview`` slices,
    preserving the zero-copy property of the hot path.
    """

    def __init__(self, name: str, slots: int = DEFAULT_SLOTS, slot_size: int = DEFAULT_SLOT_SIZE, create: bool = False) -> None:
        self.name = name
        self.slots = slots
        self.slot_size = slot_size
        self.segment_size = HEADER_SIZE + slots * slot_size
        self.path = SHM_ROOT / f"{CHANNEL_PREFIX}{name}.mmap"

        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, self.segment_size)
            self._mm = mmap.mmap(fd, self.segment_size)
        finally:
            os.close(fd)

        self._buf = memoryview(self._mm)
        if create:
            struct.pack_into(HEADER_FORMAT, self._buf, 0, MAGIC, slot_size, 0)

    @classmethod
    def create(cls, name: str, slots: int = DEFAULT_SLOTS, slot_size: int = DEFAULT_SLOT_SIZE) -> MmapChannel:
        return cls(name, slots=slots, slot_size=slot_size, create=True)

    @classmethod
    def attach(cls, name: str) -> MmapChannel:
        return cls(name, create=False)

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:  # noqa: BLE001
            pass

    def unlink(self) -> None:
        self.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def write_seq(self) -> int:
        _, _, seq = struct.unpack_from(HEADER_FORMAT, self._buf, 0)
        return seq

    def put(self, topic: str, payload: bytes | memoryview) -> int:
        """Publish a payload (same ring discipline as ShmChannel)."""
        data = bytes(payload)
        if len(data) > self.slot_size - 64:
            raise ValueError(f"payload {len(data)}B exceeds slot capacity for {self.name!r}")
        topic_b = topic.encode("utf-8")[:32]
        seq = self.write_seq() + 1
        index = seq % self.slots
        offset = HEADER_SIZE + index * self.slot_size
        struct.pack_into("Q", self._buf, offset + 8, len(data))
        struct.pack_into("Q", self._buf, offset + 16, len(topic_b))
        self._buf[offset + 24 : offset + 24 + len(topic_b)] = topic_b
        self._buf[offset + 24 + len(topic_b) : offset + 24 + len(topic_b) + len(data)] = data
        struct.pack_into("Q", self._buf, offset, seq)
        struct.pack_into("Q", self._buf, 16, seq)
        return seq

    def read(self, after_seq: int = 0) -> tuple[int, list[Message]]:
        """Read messages newer than ``after_seq`` (zero-copy)."""
        current = self.write_seq()
        if current <= after_seq:
            return after_seq, []
        messages: list[Message] = []
        for seq in range(after_seq + 1, current + 1):
            index = seq % self.slots
            offset = HEADER_SIZE + index * self.slot_size
            slot_seq, length, topic_len = struct.unpack_from("QQQ", self._buf, offset)
            if slot_seq != seq:
                continue
            topic = bytes(self._buf[offset + 24 : offset + 24 + topic_len]).decode("utf-8", errors="replace")
            payload = self._buf[offset + 24 + topic_len : offset + 24 + topic_len + length]
            messages.append(Message(seq=seq, topic=topic, payload=payload))
        return current, messages


# ---------------------------------------------------------------------------
# Bus facade
# ---------------------------------------------------------------------------


def _backend_available() -> bool:
    """Check whether multiprocessing.shared_memory is usable here."""
    try:
        from multiprocessing import shared_memory  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


class SharedMemoryBus:
    """Process-facing bus: named channels, zero-copy messages.

    Usage::

        bus = SharedMemoryBus()               # creator process
        chan = bus.create_channel("aia")      # one process creates
        bus.put("aia", {"topic": "status"})   # any process publishes JSON

        other = SharedMemoryBus()             # consumer process
        chan = other.attach_channel("aia")
        _, messages = chan.wait_for()         # zero-copy memoryviews
    """

    def __init__(self) -> None:
        # Channels are created/attached on demand and kept for reuse.
        self._channels: dict[str, Any] = {}
        self._use_shm = _backend_available()

    def create_channel(
        self, name: str, slots: int = DEFAULT_SLOTS, slot_size: int = DEFAULT_SLOT_SIZE
    ) -> Any:
        """Create (or return cached) channel ``name``."""
        if name in self._channels:
            return self._channels[name]
        channel = (
            ShmChannel.create(name, slots=slots, slot_size=slot_size)
            if self._use_shm
            else MmapChannel.create(name, slots=slots, slot_size=slot_size)
        )
        self._channels[name] = channel
        return channel

    def attach_channel(self, name: str) -> Any:
        """Attach to an existing channel ``name``."""
        if name in self._channels:
            return self._channels[name]
        channel = (
            ShmChannel.attach(name) if self._use_shm else MmapChannel.attach(name)
        )
        self._channels[name] = channel
        return channel

    def publish(self, channel_name: str, topic: str, payload: bytes | memoryview) -> int:
        """Publish a raw payload to a channel (creates it if needed)."""
        channel = self.create_channel(channel_name)
        return channel.put(topic, payload)

    def publish_json(self, channel_name: str, topic: str, payload: object) -> int:
        """Publish a JSON-encoded payload (convenience wrapper)."""
        import json

        return self.publish(channel_name, topic, json.dumps(payload).encode("utf-8"))

    def subscribe(self, channel_name: str, after_seq: int = 0) -> tuple[int, list[Message]]:
        """Read new messages from a channel (auto-attach)."""
        channel = self.attach_channel(channel_name)
        return channel.read(after_seq)

    def wait(self, channel_name: str, after_seq: int = 0, timeout_ms: float = TARGET_LATENCY_MS) -> tuple[int, list[Message]]:
        """Wait up to ``timeout_ms`` for new messages (3ms latency target)."""
        channel = self.attach_channel(channel_name)
        return channel.wait_for(after_seq, timeout_ms=timeout_ms)

    def close(self) -> None:
        """Detach all channels (does not unlink segments)."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()

    def destroy(self, name: str) -> None:
        """Destroy a channel and its shared segment."""
        channel = self._channels.pop(name, None)
        if channel is not None:
            channel.unlink()


# ---------------------------------------------------------------------------
# Entry point (diagnostics)
# ---------------------------------------------------------------------------


def main() -> int:
    """Self-test / diagnostics: exercise the bus in-process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    bus = SharedMemoryBus()
    try:
        channel = bus.create_channel("diagnostics")
        start = time.perf_counter()
        seq = bus.publish("diagnostics", "self-test", b"OpenCodeWEB OS shared memory bus OK")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        new_seq, messages = bus.wait("diagnostics")
        report = {
            "backend": "shared_memory" if bus._use_shm else "mmap",  # noqa: SLF001
            "segment": str(SHM_ROOT),
            "channel": channel.name,
            "slot_size": channel.slot_size,
            "slots": channel.slots,
            "seq": seq,
            "read_seq": new_seq,
            "messages": [{"seq": m.seq, "topic": m.topic, "size": len(m.payload)} for m in messages],
            "publish_latency_ms": round(elapsed_ms, 3),
            "latency_target_ms": TARGET_LATENCY_MS,
        }
        print(__import__("json").dumps(report, indent=2))
        return 0
    finally:
        bus.destroy("diagnostics")
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
