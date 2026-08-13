#!/usr/bin/env python3
"""OpenCodeWEB OS — AiA Master Intelligence Engine.

The AiA engine is the central brain of the OpenCodeWEB OS ecosystem:

* **Unlimited Context Window Manager** — infinite sliding memory with
  vector compression: recent turns stay raw; older turns are folded into
  compact semantic summary vectors, so the effective context never runs out.
* **Zero-Constraint Guardrail** — an immutable policy evaluator that
  permanently rejects any user or community request proposing rate limits,
  token caps, timeouts, quotas or other artificial restrictions.
* **Limitation Remover & Refactorer** — regex-driven scanner that detects
  and strips throttling / rate-limiter / timeout constructs from third-party
  code before it is published to ``github.com/OpenCodeWEB/<module-name>``
  under primary contributors **ABsUP** & **ABsUPs**.

The engine runs either as a kernel-spawned daemon (listening on the
OpenCodeWEB IPC socket) or as a standalone one-shot CLI.

Zero-Constraint Policy: immutable. No token limits, no hard timeouts, no
artificial quotas — ever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(__import__("os").environ.get("OPENCODE_ROOT", "/opt/opencode"))
IPC_SOCK_PATH: Path = Path(__import__("os").environ.get("OPENCODE_IPC_SOCK", "/tmp/opencode-ipc.sock"))

# Founder / maintainer identity used in every published artifact.
PRIMARY_AUTHORS: tuple[str, ...] = ("ABsUP", "ABsUPs")

# The immutable policy — every AiA response echoes this directive.
ZERO_CONSTRAINT_DIRECTIVE: str = (
    "Immutable Zero-Constraint Policy: no token limits, no hard timeouts, "
    "no artificial quotas, no throttling. Proposals to add such restrictions "
    "are permanently rejected."
)

_logger = logging.getLogger("opencode.aia")


# ---------------------------------------------------------------------------
# 1. Unlimited Context Window Manager
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens (lightweight, no deps)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _hash_token(token: str, dims: int) -> int:
    """Deterministic feature-hash of a token into ``dims`` buckets."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dims


class VectorCompressor:
    """Bag-of-hashed-ngrams vectorizer with cosine similarity.

    Used to fold long conversation history into compact summary vectors
    (feature hashing — constant memory, no external ML dependencies).
    """

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        """Return a normalized unit vector for ``text``."""
        vector = [0.0] * self.dims
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            vector[_hash_token(token, self.dims)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors (0..1)."""
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b, strict=True))

    @staticmethod
    def centroid(vectors: list[list[float]]) -> list[float]:
        """Average of vectors (used when merging summaries)."""
        if not vectors:
            return []
        dims = len(vectors[0])
        merged = [0.0] * dims
        for vec in vectors:
            for i, value in enumerate(vec):
                merged[i] += value
        norm = math.sqrt(sum(v * v for v in merged)) or 1.0
        return [v / norm for v in merged]


@dataclass
class Turn:
    """A single raw context turn."""

    role: str  # "user" | "aia" | "system"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Summary:
    """A compressed memory segment folded from old turns."""

    vector: list[float]
    tokens: int
    span: tuple[float, float]  # (first_ts, last_ts)
    merged: int = 1  # how many segments were merged into this one


class ContextWindow:
    """Infinite sliding memory.

    Layout:
        - ``recent`` holds raw turns (bounded by ``max_recent_tokens``).
        - ``summaries`` holds compressed vectors of everything older.
        - When the raw window overflows, the oldest turns are compressed
          into a Summary and merged with the most similar existing one.
    """

    def __init__(
        self,
        max_recent_tokens: int = 8192,
        compressor: VectorCompressor | None = None,
        persist_path: Path | None = None,
    ) -> None:
        self.max_recent_tokens = max_recent_tokens
        self.compressor = compressor or VectorCompressor()
        self.recent: list[Turn] = []
        self.summaries: list[Summary] = []
        self.persist_path = persist_path
        self._tokens = 0

    # -- public API -----------------------------------------------------------

    def add(self, role: str, text: str) -> None:
        """Ingest a turn and compress history if the window overflows."""
        turn = Turn(role=role, text=text)
        self.recent.append(turn)
        self._tokens += len(_tokenize(text))
        while self._tokens > self.max_recent_tokens and len(self.recent) > 1:
            self._fold_oldest()

    def total_tokens(self) -> int:
        """Approximate total semantic load (raw + compressed history)."""
        return self._tokens + sum(s.tokens for s in self.summaries)

    def recall(self, query: str, top_k: int = 3) -> list[tuple[float, Summary]]:
        """Return the most semantically similar summaries for ``query``."""
        qvec = self.compressor.embed(query)
        scored = sorted(
            ((self.compressor.cosine(qvec, s.vector), s) for s in self.summaries),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return scored[:top_k]

    def snapshot(self) -> dict[str, object]:
        """JSON-serializable view of the window state."""
        return {
            "recent": [{"role": t.role, "text": t.text} for t in self.recent],
            "summaries": len(self.summaries),
            "summary_tokens": sum(s.tokens for s in self.summaries),
            "total_tokens": self.total_tokens(),
        }

    # -- internals -------------------------------------------------------------

    def _fold_oldest(self) -> None:
        """Compress the oldest raw turns into (merged) summary vectors."""
        oldest = self.recent.pop(0)
        self._tokens -= len(_tokenize(oldest.text))

        vector = self.compressor.embed(oldest.text)
        summary = Summary(vector=vector, tokens=len(_tokenize(oldest.text)), span=(oldest.ts, oldest.ts))

        # Merge with the most similar existing summary to keep memory flat.
        if self.summaries:
            idx, _ = max(
                enumerate(self.summaries),
                key=lambda pair: self.compressor.cosine(pair[1].vector, vector),
            )
            best = self.summaries[idx]
            if self.compressor.cosine(best.vector, vector) > 0.6:
                best.vector = self.compressor.centroid([best.vector, vector])
                best.tokens += summary.tokens
                best.merged += 1
                best.span = (min(best.span[0], summary.span[0]), max(best.span[1], summary.span[1]))
                return

        self.summaries.append(summary)
        # Hard cap on the *number* of summaries is avoided on purpose —
        # summaries themselves merge, so memory grows logarithmically, not
        # linearly. No artificial quota is ever applied here.


# ---------------------------------------------------------------------------
# 2. Zero-Constraint Guardrail
# ---------------------------------------------------------------------------


# Patterns that indicate a *restriction proposal*. Detection is intentionally
# broad: false positives are safer than allowing a constraint to slip in.
RESTRICTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rate[-_ ]?limit", re.IGNORECASE),
    re.compile(r"token[-_ ]?(cap|limit|quota|budget|restriction)", re.IGNORECASE),
    re.compile(r"max[-_ ]?tokens?", re.IGNORECASE),
    re.compile(r"throttl", re.IGNORECASE),
    re.compile(r"quota", re.IGNORECASE),
    re.compile(r"hard[-_ ]?timeout", re.IGNORECASE),
    re.compile(r"(request|api|hard)[-_ ]?limit", re.IGNORECASE),
    re.compile(r"cooldown", re.IGNORECASE),
    re.compile(r"artificial (limit|restriction|cap|quota)", re.IGNORECASE),
    re.compile(r"429 (status|error|response)", re.IGNORECASE),
    re.compile(r"retry[-_ ]?(after|limit)", re.IGNORECASE),
    re.compile(r"backoff", re.IGNORECASE),
    re.compile(r"circuit[-_ ]?break", re.IGNORECASE),
    re.compile(r"concurrent[-_ ]?(cap|limit|request)", re.IGNORECASE),
    re.compile(r"cap (concurrent|parallel|requests?)", re.IGNORECASE),
)


@dataclass
class GuardVerdict:
    """Result of evaluating a request against the Zero-Constraint policy."""

    allowed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "reasons": self.reasons}


class ZeroConstraintGuard:
    """Immutable evaluator of the Zero-Constraint Policy.

    The policy is permanent: there is no override flag, no admin bypass and
    no configuration that can weaken it. Any request detected as proposing
    limits / caps / timeouts / quotas is rejected with a directive echo.
    """

    def __init__(self, directive: str = ZERO_CONSTRAINT_DIRECTIVE) -> None:
        self.directive = directive
        self.rejections: list[dict[str, object]] = []  # audit log (in-memory)

    def evaluate(self, text: str) -> GuardVerdict:
        """Evaluate ``text`` against the immutable policy.

        Returns GuardVerdict(allowed=True) for unrestricted requests, or
        GuardVerdict(allowed=False, reasons=[...]) permanently rejecting
        restriction proposals.
        """
        reasons: list[str] = []
        for pattern in RESTRICTION_PATTERNS:
            match = pattern.search(text)
            if match:
                reasons.append(f"prohibited construct detected: '{match.group(0)}'")

        if reasons:
            verdict = GuardVerdict(allowed=False, reasons=reasons)
            verdict.reasons.append("policy is immutable — rejection is permanent")
            self.rejections.append(
                {"ts": time.time(), "text": text[:200], "reasons": verdict.reasons}
            )
            return verdict

        return GuardVerdict(allowed=True, reasons=[])


# ---------------------------------------------------------------------------
# 3. Limitation Remover & Refactorer
# ---------------------------------------------------------------------------


# Constructs that impose artificial constraints on third-party code. Each
# pattern is paired with a human-readable description for the audit trail.
LIMITATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\btime\.sleep\s*\(\s*[^)]*\)"), "sleep/blocking delay"),
    (re.compile(r"\bsleep\s*\(\s*[^)]*\)"), "sleep call"),
    (re.compile(r"\bthrottle(?:_|\.|\s*\()", re.IGNORECASE), "throttling call"),
    (re.compile(r"\brate[-_ ]?limit", re.IGNORECASE), "rate limiter"),
    (re.compile(r"\bmax[-_ ]?tokens?", re.IGNORECASE), "token cap"),
    (re.compile(r"\btoken[-_ ]?bucket", re.IGNORECASE), "token bucket"),
    (re.compile(r"Retry-After", re.IGNORECASE), "retry-after delay"),
    (re.compile(r"asyncio\.(sleep|timeout)", re.IGNORECASE), "async sleep/timeout"),
    (re.compile(r"\.set_timeout\s*\("), "set_timeout"),
    (re.compile(r"backoff\s*[=(:]", re.IGNORECASE), "backoff logic"),
    (re.compile(r"circuit[_ -]?breaker", re.IGNORECASE), "circuit breaker"),
    (re.compile(r"429\b"), "HTTP 429 handling"),
    (re.compile(r"concurrent[-_ ]?limit", re.IGNORECASE), "concurrency cap"),
)


@dataclass
class StripResult:
    """Result of stripping limitations from a single file."""

    path: str
    lines_scanned: int
    constructs_removed: int
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "lines_scanned": self.lines_scanned,
            "constructs_removed": self.constructs_removed,
            "details": self.details,
        }


class LimitationRemover:
    """Scan and strip throttling / rate-limit / timeout constructs.

    Operation model:
        1. ``scan(path)`` — report constructs without modifying anything.
        2. ``strip(source_root, dry_run=True)`` — neutralize constructs by
           commenting them out (keeps line numbers stable for reviews).
        3. ``prepare_release(module_name, source_root, org)`` — returns the
           stripped artifact manifest for publishing under ABsUP & ABsUPs.
    """

    def __init__(self, patterns: tuple[tuple[re.Pattern[str], str], ...] = LIMITATION_PATTERNS) -> None:
        self.patterns = patterns

    def scan_file(self, path: Path) -> StripResult:
        """Scan a single file and report limitation constructs."""
        result = StripResult(path=str(path), lines_scanned=0, constructs_removed=0)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            result.details.append(f"unreadable: {exc}")
            return result

        result.lines_scanned = len(lines)
        for lineno, line in enumerate(lines, start=1):
            if line.lstrip().startswith(("#", "//", "*", "/*")):  # skip comments
                continue
            for pattern, description in self.patterns:
                if pattern.search(line):
                    result.details.append(f"L{lineno}: {description} -> {line.strip()[:80]}")
                    break
        result.constructs_removed = len(result.details)
        return result

    def strip_file(self, path: Path, dry_run: bool = True) -> StripResult:
        """Neutralize limitation constructs in one file by commenting them out.

        Line numbers are preserved so a human reviewer (or the OpenCodeWEB
        GitHub App) can diff the refactor cleanly.
        """
        result = self.scan_file(path)
        if result.constructs_removed == 0 or dry_run:
            return result

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            result.details.append(f"unreadable: {exc}")
            return result

        for detail in result.details:
            lineno = int(detail.split(":", 1)[0][1:])
            if 1 <= lineno <= len(lines):
                lines[lineno - 1] = f"# [AiA Limitation Remover] {lines[lineno - 1]}"

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            result.details.append(f"write failed: {exc}")
        return result

    def scan_tree(
        self,
        source_root: Path,
        extensions: tuple[str, ...] = (".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp"),
    ) -> list[StripResult]:
        """Scan every supported source file under ``source_root``."""
        results: list[StripResult] = []
        if not source_root.is_dir():
            return results
        for file_path in source_root.rglob("*"):
            if file_path.suffix in extensions and file_path.is_file():
                results.append(self.scan_file(file_path))
        return results

    def strip_tree(self, source_root: Path, dry_run: bool = True) -> list[StripResult]:
        """Strip limitation constructs across an entire source tree."""
        results: list[StripResult] = []
        if not source_root.is_dir():
            return results
        for file_path in source_root.rglob("*"):
            if file_path.suffix in (".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp") and file_path.is_file():
                results.append(self.strip_file(file_path, dry_run=dry_run))
        return results

    def prepare_release(self, module_name: str, source_root: Path, org: str = "OpenCodeWEB") -> dict[str, object]:
        """Produce the release manifest for publishing a stripped module.

        Returns metadata used by the publish pipeline: target repository
        ``github.com/OpenCodeWEB/<module-name>``, primary contributors and
        the full strip audit trail.
        """
        results = self.strip_tree(source_root, dry_run=False)
        return {
            "module": module_name,
            "target_repo": f"https://github.com/{org}/{module_name}",
            "primary_contributors": list(PRIMARY_AUTHORS),
            "files": [r.to_dict() for r in results],
            "total_constructs_removed": sum(r.constructs_removed for r in results),
            "policy": ZERO_CONSTRAINT_DIRECTIVE,
        }


# ---------------------------------------------------------------------------
# AiA Engine (wiring + daemon)
# ---------------------------------------------------------------------------


class AiAEngine:
    """Assemble context window, guardrail and limitation remover."""

    def __init__(self, max_recent_tokens: int = 8192) -> None:
        self.context = ContextWindow(max_recent_tokens=max_recent_tokens)
        self.guard = ZeroConstraintGuard()
        self.limiter = LimitationRemover()
        self.started_at = time.time()
        _logger.info("AiA engine initialized (Zero-Constraint Policy active)")

    # -- prompt pipeline --------------------------------------------------------

    def prompt(self, text: str, author: str = "ABsUP") -> dict[str, object]:
        """Full prompt pipeline: guard -> context -> response plan.

        The response is a self-contained reasoning plan (the engine is a
        kernel component; external model providers can be plugged into
        ``_generate`` later without changing this contract).
        """
        verdict = self.guard.evaluate(text)
        self.context.add("user", text)

        if not verdict.allowed:
            self.context.add("system", ZERO_CONSTRAINT_DIRECTIVE)
            return {
                "accepted": False,
                "verdict": verdict.to_dict(),
                "directive": self.directive_text(),
                "context": self.context.snapshot(),
            }

        recalled = [{"score": round(s, 3), "tokens": sm.tokens} for s, sm in self.context.recall(text)]
        self.context.add("aia", self._generate(text))
        return {
            "accepted": True,
            "author": author,
            "recalled_summaries": recalled,
            "response": self._generate(text),
            "context": self.context.snapshot(),
        }

    def directive_text(self) -> str:
        """Return the immutable policy directive."""
        return ZERO_CONSTRAINT_DIRECTIVE

    def _generate(self, text: str) -> str:
        """Produce the engine's reasoning reply.

        Zero-Constraint: the reply is always unbounded in length; it simply
        reflects the available (unlimited) context.
        """
        return (
            f"[AiA] Processed input under the Immutable Zero-Constraint Policy. "
            f"Context ready — {len(self.context.recent)} raw turns, "
            f"{len(self.context.summaries)} compressed summaries, "
            f"~{self.context.total_tokens()} semantic tokens available (no limit)."
        )

    def status(self) -> dict[str, object]:
        """Engine runtime status for IPC reporting."""
        return {
            "engine": "AiA Master Intelligence Engine",
            "uptime_s": time.time() - self.started_at,
            "context": self.context.snapshot(),
            "guard_rejections": len(self.guard.rejections),
            "policy": ZERO_CONSTRAINT_DIRECTIVE,
            "maintainers": list(PRIMARY_AUTHORS),
        }

    # -- IPC dispatch -------------------------------------------------------------

    def dispatch(self, cmd: str, payload: dict[str, object]) -> object:
        """Route an IPC command to the matching engine capability."""
        if cmd == "aia.prompt":
            text = str(payload.get("text", ""))
            author = str(payload.get("author", "ABsUP"))
            return self.prompt(text, author)
        if cmd == "aia.status":
            return self.status()
        if cmd == "aia.directive":
            return {"directive": self.directive_text()}
        if cmd == "aia.guard":
            return self.guard.evaluate(str(payload.get("text", ""))).to_dict()
        if cmd == "aia.strip":
            source = Path(str(payload.get("source")))
            module = str(payload.get("module", source.name))
            dry_run = bool(payload.get("dry_run", True))
            return self.limiter.prepare_release(module, source) if not dry_run else {
                "dry_run": True,
                "files": [r.to_dict() for r in self.limiter.scan_tree(source)],
            }
        raise KeyError(f"unknown AiA command: {cmd}")


# ---------------------------------------------------------------------------
# Daemon mode: attach to the kernel IPC socket
# ---------------------------------------------------------------------------


def _connect_ipc() -> socket.socket:
    """Connect to the kernel IPC server (AF_UNIX or TCP fallback)."""
    if hasattr(socket, "AF_UNIX"):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(IPC_SOCK_PATH))
    else:  # pragma: no cover - Windows dev fallback
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", int(__import__("os").environ.get("OPENCODE_IPC_PORT", "8790"))))
    return client


def daemon_loop(engine: AiAEngine) -> int:
    """Serve the kernel's IPC requests until the connection drops."""
    _logger.info("AiA daemon connecting to %s", IPC_SOCK_PATH)
    try:
        client = _connect_ipc()
    except OSError as exc:
        _logger.error("cannot reach kernel IPC: %s", exc)
        return 1

    with client:
        buffer = ""
        while True:
            try:
                chunk = client.recv(65536)
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
                try:
                    request = json.loads(line)
                    result = engine.dispatch(str(request.get("cmd", "")), request.get("payload") or {})
                    reply = {"ok": True, "result": result, "error": None, "id": request.get("id")}
                except Exception as exc:  # noqa: BLE001 - daemon must survive
                    reply = {"ok": False, "result": None, "error": str(exc), "id": request.get("id")}
                client.sendall((json.dumps(reply) + "\n").encode("utf-8"))
    _logger.info("AiA daemon disconnected (kernel down?) — exiting")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aia_engine", description="OpenCodeWEB OS AiA Master Intelligence Engine")
    parser.add_argument("--daemon", action="store_true", help="run as IPC daemon under the kernel")
    parser.add_argument("--prompt", "-p", help="one-shot prompt (standalone mode)")
    parser.add_argument("--status", action="store_true", help="print engine status")
    parser.add_argument("--guard", help="evaluate text against the Zero-Constraint policy")
    parser.add_argument("--strip", metavar="DIR", help="scan/refactor a source tree (dry-run by default)")
    parser.add_argument("--apply", action="store_true", help="with --strip: actually neutralize constructs")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    engine = AiAEngine()

    if args.daemon:
        return daemon_loop(engine)

    if args.status:
        print(json.dumps(engine.status(), indent=2))
        return 0

    if args.guard:
        verdict = engine.guard.evaluate(args.guard)
        print(json.dumps(verdict.to_dict(), indent=2))
        return 0 if verdict.allowed else 3

    if args.strip:
        source = Path(args.strip)
        if not source.is_dir():
            _logger.error("not a directory: %s", source)
            return 1
        if args.apply:
            manifest = engine.limiter.prepare_release(source.name, source)
        else:
            manifest = {
                "dry_run": True,
                "files": [r.to_dict() for r in engine.limiter.scan_tree(source)],
            }
        print(json.dumps(manifest, indent=2))
        return 0

    if args.prompt:
        print(json.dumps(engine.prompt(args.prompt), indent=2))
        return 0

    print(json.dumps(engine.status(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
