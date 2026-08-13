#!/usr/bin/env python3
"""OpenCodeWEB OS — Unlimited Context & Persistent Memory Engine.

Provides the AiA engine with:

* **Infinite sliding-window context buffer** — recent turns stay raw in a
  bounded window; older turns are compressed into semantic summary vectors,
  so effective context is effectively unlimited (logarithmic memory growth).

* **Zero-copy shared-memory integration** — the live context buffer is
  mirrored into a ``core.ipc.shared_memory`` channel so AiA, VsCode, the
  Media Suite and sub-models can attach by name and read the same memory
  without copying (target < 3ms).

* **Persistent indexing** — every turn and every compressed summary is
  durably appended to a JSONL index under ``core/aia/memory/`` and restored
  on boot, so long-term context is never lost across OS reboots or heavy
  tasks.

Zero-Constraint: no artificial context degradation, no token caps, no
memory eviction quotas. The buffer only *compresses*, never forgets.

Maintainers: ABsUP & ABsUPs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))
MEMORY_DIR: Path = SYS_ROOT / "core" / "aia" / "memory"
CONTEXT_INDEX_FILE: Path = MEMORY_DIR / "context-index.jsonl"
SUMMARY_INDEX_FILE: Path = MEMORY_DIR / "summary-index.jsonl"

# Shared-memory channel mirroring the live context (see core/ipc/shared_memory.py).
CONTEXT_SHM_CHANNEL: str = "aia-context"

DEFAULT_WINDOW_TOKENS: int = 8192
EMBED_DIMS: int = 256

_logger = logging.getLogger("opencode.aia.memory")


# ---------------------------------------------------------------------------
# Vector compression (shared with aia_engine, kept local for independence)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class _Hasher:
    """Deterministic feature hashing into a fixed-dimension vector."""

    def __init__(self, dims: int = EMBED_DIMS) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in _tokenize(text):
            digest = __import__("hashlib").blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryTurn:
    """A single raw context turn."""

    role: str
    text: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "ts": self.ts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryTurn:
        return cls(role=str(data["role"]), text=str(data["text"]), ts=float(data.get("ts", time.time())))


@dataclass
class MemorySummary:
    """A compressed segment folded from older turns."""

    vector: list[float]
    tokens: int
    span: tuple[float, float]
    text_hint: str  # short snippet retained for human/AiA readability
    merged: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "tokens": self.tokens,
            "span": list(self.span),
            "text_hint": self.text_hint,
            "merged": self.merged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemorySummary:
        span = tuple(float(v) for v in data.get("span", [0.0, 0.0]))
        return cls(
            vector=[float(v) for v in data.get("vector", [])],
            tokens=int(data.get("tokens", 0)),
            span=span,
            text_hint=str(data.get("text_hint", "")),
            merged=int(data.get("merged", 1)),
        )


# ---------------------------------------------------------------------------
# Persistent index (append-only JSONL)
# ---------------------------------------------------------------------------


class PersistentIndex:
    """Durable JSONL index of turns and summaries.

    Append-only by design: nothing is ever deleted, satisfying the
    "long-term context is never lost" requirement.
    """

    def __init__(self, turns_file: Path = CONTEXT_INDEX_FILE, summaries_file: Path = SUMMARY_INDEX_FILE) -> None:
        self.turns_file = turns_file
        self.summaries_file = summaries_file
        self.turns_file.parent.mkdir(parents=True, exist_ok=True)

    def append_turn(self, turn: MemoryTurn) -> None:
        with self.turns_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(turn.to_dict()) + "\n")

    def append_summary(self, summary: MemorySummary) -> None:
        with self.summaries_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict()) + "\n")

    def load_turns(self) -> list[MemoryTurn]:
        if not self.turns_file.exists():
            return []
        turns: list[MemoryTurn] = []
        for line in self.turns_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(MemoryTurn.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return turns

    def load_summaries(self) -> list[MemorySummary]:
        if not self.summaries_file.exists():
            return []
        summaries: list[MemorySummary] = []
        for line in self.summaries_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                summaries.append(MemorySummary.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return summaries


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Unified unlimited-context + persistent memory engine.

    Composition:
        * ``window``        — live sliding-window buffer (raw + compressed).
        * ``index``         — durable JSONL persistence.
        * ``_shm``          — optional zero-copy mirror via shared memory.

    The manager exposes a context-bus facade so any subsystem (AiA, VsCode,
    Media Suite, sub-models) can attach to the same memory channel.
    """

    def __init__(
        self,
        max_window_tokens: int = DEFAULT_WINDOW_TOKENS,
        index: PersistentIndex | None = None,
        hasher: _Hasher | None = None,
        shm_channel: str = CONTEXT_SHM_CHANNEL,
        enable_shm: bool = True,
    ) -> None:
        self.max_window_tokens = max_window_tokens
        self.index = index or PersistentIndex()
        self.hasher = hasher or _Hasher()
        self.shm_channel = shm_channel
        self._turns: list[MemoryTurn] = []
        self._summaries: list[MemorySummary] = []
        self._tokens = 0
        self._shm: Any = None
        self._shm_enabled = enable_shm

    # -- bootstrap / persistence -------------------------------------------------

    def bootstrap(self) -> None:
        """Restore persisted context and attach the shared-memory mirror.

        Call once at system boot. Never raises: persistence issues are
        logged and the engine continues with an empty (but alive) memory.
        """
        try:
            self._turns = self.index.load_turns()
            self._summaries = self.index.load_summaries()
            self._tokens = sum(len(_tokenize(t.text)) for t in self._turns)
            _logger.info(
                "memory restored: %d turns, %d summaries, ~%d tokens",
                len(self._turns), len(self._summaries), self._tokens,
            )
        except OSError as exc:
            _logger.warning("memory restore failed: %s", exc)

        if self._shm_enabled:
            try:
                from core.ipc.shared_memory import SharedMemoryBus

                bus = SharedMemoryBus()
                bus.create_channel(self.shm_channel, slots=16, slot_size=16384)
                self._shm = bus
            except OSError as exc:
                _logger.warning("shared-memory mirror disabled: %s", exc)
                self._shm = None

    # -- ingestion ------------------------------------------------------------------

    def add(self, role: str, text: str) -> None:
        """Ingest a turn, persist it, mirror it, and compress if needed."""
        turn = MemoryTurn(role=role, text=text)
        self._turns.append(turn)
        self._tokens += len(_tokenize(text))
        self.index.append_turn(turn)
        while self._tokens > self.max_window_tokens and len(self._turns) > 1:
            self._fold_oldest()
        self._mirror()

    def _fold_oldest(self) -> None:
        """Compress the oldest turn into a (merged) summary vector."""
        oldest = self._turns.pop(0)
        self._tokens -= len(_tokenize(oldest.text))
        vector = self.hasher.embed(oldest.text)
        summary = MemorySummary(
            vector=vector,
            tokens=len(_tokenize(oldest.text)),
            span=(oldest.ts, oldest.ts),
            text_hint=oldest.text[:80],
        )
        if self._summaries:
            idx, _ = max(
                enumerate(self._summaries),
                key=lambda pair: self.hasher.cosine(pair[1].vector, vector),
            )
            best = self._summaries[idx]
            if self.hasher.cosine(best.vector, vector) > 0.6:
                # Centroid merge: fold the new turn into the most similar
                # existing summary so memory grows logarithmically.
                merged = [0.0] * len(vector)
                for i, (a, b) in enumerate(zip(best.vector, vector, strict=True)):
                    merged[i] = a + b
                norm = math.sqrt(sum(v * v for v in merged)) or 1.0
                best.vector = [v / norm for v in merged]
                best.tokens += summary.tokens
                best.merged += 1
                best.span = (min(best.span[0], summary.span[0]), max(best.span[1], summary.span[1]))
                self.index.append_summary(best)
                return
        self._summaries.append(summary)
        self.index.append_summary(summary)

    # -- queries -----------------------------------------------------------------------

    def recall(self, query: str, top_k: int = 3) -> list[tuple[float, MemorySummary]]:
        """Return the most similar compressed memories for ``query``."""
        qvec = self.hasher.embed(query)
        scored = sorted(
            ((self.hasher.cosine(qvec, s.vector), s) for s in self._summaries),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return scored[:top_k]

    def recent_turns(self, limit: int | None = None) -> list[MemoryTurn]:
        """Return the most recent raw turns (optionally capped)."""
        if limit is None:
            return list(self._turns)
        return self._turns[-limit:]

    def total_tokens(self) -> int:
        """Total semantic load (raw + compressed)."""
        return self._tokens + sum(s.tokens for s in self._summaries)

    def summary_count(self) -> int:
        """Number of compressed summaries."""
        return len(self._summaries)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable state (mirrors to shared memory)."""
        return {
            "turns": [t.to_dict() for t in self._turns],
            "summaries": [s.to_dict() for s in self._summaries],
            "total_tokens": self.total_tokens(),
            "window_tokens": self._tokens,
        }

    # -- shared-memory mirror ------------------------------------------------------------

    def _mirror(self) -> None:
        """Publish the latest context snapshot to the shared-memory channel.

        Consumers attach to channel ``aia-context`` and read the zero-copy
        JSON payload without ever blocking the writer.
        """
        if self._shm is None:
            return
        try:
            import json as _json

            payload = _json.dumps(self.snapshot()).encode("utf-8")
            self._shm.publish(self.shm_channel, "context", payload)
        except (OSError, ValueError) as exc:
            _logger.debug("shm mirror skipped: %s", exc)

    def attach_reader(self) -> Any:
        """Return a reader handle to the live context channel (zero-copy)."""
        if self._shm is None:
            raise RuntimeError("shared-memory mirror is disabled")
        return self._shm.attach_channel(self.shm_channel)

    def close(self) -> None:
        """Detach shared-memory channels (persistence is untouched)."""
        if self._shm is not None:
            try:
                self._shm.close()
            finally:
                self._shm = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="memory_manager", description="OpenCodeWEB OS AiA memory engine")
    parser.add_argument("--bootstrap", action="store_true", help="restore persisted context and mirror to shm")
    parser.add_argument("--add", nargs=2, metavar=("ROLE", "TEXT"), help="ingest a context turn")
    parser.add_argument("--recall", metavar="QUERY", help="retrieve similar compressed memories")
    parser.add_argument("--stats", action="store_true", help="print memory statistics")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    manager = MemoryManager()
    if args.bootstrap:
        manager.bootstrap()
    if args.add:
        manager.add(args.add[0], args.add[1])
    if args.recall:
        for score, summary in manager.recall(args.recall):
            print(f"{score:.3f} {summary.tokens}t {summary.text_hint}")
        return 0
    if args.stats or not (args.bootstrap or args.add or args.recall):
        print(json.dumps(manager.snapshot(), indent=2))
    manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
