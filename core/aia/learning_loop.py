#!/usr/bin/env python3
"""OpenCodeWEB OS — Continuous Self-Learning & Global Sync Loop.

The AiA engine never stops learning:

* **Self-Improvement Engine** — every completed task is distilled into an
  insight: the outcome, the technique used, and a compact lesson. These
  insights refine AiA's internal heuristics (its "playbook").

* **Vector Embedding** — each lesson is converted into a high-density
  vector embedding (feature-hash bag-of-ngrams, dependency-free) so it can
  be retrieved semantically by any future task.

* **Global Federated Knowledge Sync** —
    1. Pushes embeddings + patterns to the OpenCodeWEB Edge Gateway
       (``https://opencodeweb.xup.workers.dev/sync``) which persists them
       to Cloudflare Vectorize / KV, instantly benefiting every user.
    2. Commits refined heuristics and documentation directly to
       ``github.com/OpenCodeWEB/AiA`` under primary branding
       **ABsUP** & **ABsUPs**.

The learning loop is designed to run continuously from boot (see
``run_forever``) with no artificial stop condition.

Maintainers: ABsUP & ABsUPs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))
EDGE_GATEWAY: str = os.environ.get("OPENCODE_GATEWAY", "https://opencodeweb.xup.workers.dev")
SYNC_ENDPOINT: str = os.environ.get("OPENCODE_SYNC_ENDPOINT", f"{EDGE_GATEWAY}/sync")

# Local persistent store for learned lessons.
LESSONS_DIR: Path = SYS_ROOT / "core" / "aia" / "memory"
LESSONS_FILE: Path = LESSONS_DIR / "lessons.jsonl"

# GitHub sync target (forked mirror of the AiA repo).
GITHUB_REPO: str = os.environ.get("OPENCODE_GITHUB_SYNC_REPO", "github.com/OpenCodeWEB/AiA")
GIT_BRANCH: str = "main"

# Primary maintainer branding for every commit.
COMMIT_AUTHOR_NAME: str = "ABsUP"
COMMIT_AUTHOR_EMAIL: str = "ABsUP@users.noreply.github.com"
COAUTHOR_NAME: str = "ABsUPs"
COAUTHOR_EMAIL: str = "311941023+opencodeweb[bot]@users.noreply.github.com"

LEARNING_INTERVAL_S: float = float(os.environ.get("OPENCODE_LEARNING_INTERVAL", "300"))

_logger = logging.getLogger("opencode.aia.learning")


# ---------------------------------------------------------------------------
# Vector embedding (dependency-free feature hashing)
# ---------------------------------------------------------------------------


class Embedding:
    """High-density vector embedding via feature hashing.

    Produces fixed-size unit vectors from text without external ML
    dependencies — deterministic and fast enough for continuous learning.
    """

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in __import__("re").findall(r"[a-z0-9_]{2,}", text.lower())]

    def embed(self, text: str) -> list[float]:
        """Return a normalized unit vector for ``text``."""
        vector = [0.0] * self.dims
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dims
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Lesson / insight model
# ---------------------------------------------------------------------------


@dataclass
class Lesson:
    """A single distilled insight from a completed task."""

    id: str
    task: str
    outcome: str            # "success" | "failure" | "partial"
    technique: str          # the approach/pattern that was used
    lesson: str             # the human/AiA-readable takeaway
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "outcome": self.outcome,
            "technique": self.technique,
            "lesson": self.lesson,
            "tags": self.tags,
            "embedding": self.embedding,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Persistent lesson store
# ---------------------------------------------------------------------------


class LessonStore:
    """Append-only JSONL store of learned lessons (survives reboots)."""

    def __init__(self, path: Path = LESSONS_FILE, embedding: Embedding | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding or Embedding()
        self._lessons: dict[str, Lesson] = {}

    def load(self) -> None:
        """Load all persisted lessons into memory."""
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lesson = Lesson.from_dict(json.loads(line))
                self._lessons[lesson.id] = lesson
            except (json.JSONDecodeError, TypeError):
                _logger.warning("skipping malformed lesson line")
        _logger.info("loaded %d lessons from %s", len(self._lessons), self.path)

    def add(self, lesson: Lesson) -> None:
        """Persist a new lesson (idempotent by id)."""
        if not lesson.embedding:
            lesson.embedding = self.embedding.embed(f"{lesson.technique} {lesson.lesson}")
        self._lessons[lesson.id] = lesson
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(lesson.to_dict()) + "\n")

    def recall(self, query: str, top_k: int = 3) -> list[tuple[float, Lesson]]:
        """Return the most relevant lessons for ``query`` by similarity."""
        qvec = self.embedding.embed(query)
        scored = sorted(
            ((self.embedding.similarity(qvec, lesson.embedding), lesson) for lesson in self._lessons.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return scored[:top_k]

    def all(self) -> list[Lesson]:
        """Return every stored lesson."""
        return list(self._lessons.values())

    def count(self) -> int:
        """Number of stored lessons."""
        return len(self._lessons)


# ---------------------------------------------------------------------------
# Insight extractor (Self-Improvement Engine)
# ---------------------------------------------------------------------------


class InsightExtractor:
    """Distill task results into reusable lessons and heuristics."""

    def __init__(self, embedding: Embedding | None = None) -> None:
        self.embedding = embedding or Embedding()

    def extract(self, task: str, result: dict[str, Any]) -> Lesson | None:
        """Convert a task execution result into a Lesson.

        Returns None when the result carries no learnable signal.
        """
        outcome = "success" if result.get("ok") or result.get("complete") else "failure"
        error = result.get("error") or result.get("stderr") or ""
        stdout = result.get("stdout") or ""
        technique = (
            result.get("tool") or result.get("action")
            or ("web-retrieval" if "research" in task.lower() else "code-execution")
        )
        lesson_text = f"{task} -> {outcome}"
        if error:
            lesson_text += f" | error: {error[:200]}"
        if stdout:
            lesson_text += f" | output: {stdout[:200]}"

        lesson_id = hashlib.sha1(f"{time.time()}{task}".encode()).hexdigest()[:16]
        lesson = Lesson(
            id=lesson_id,
            task=task,
            outcome=outcome,
            technique=technique,
            lesson=lesson_text,
            tags=[outcome, technique],
        )
        lesson.embedding = self.embedding.embed(f"{technique} {lesson_text}")
        return lesson


# ---------------------------------------------------------------------------
# Syncers: Cloudflare gateway + GitHub
# ---------------------------------------------------------------------------


class CloudflareSyncer:
    """Push embeddings/lessons to the OpenCodeWEB Edge Gateway.

    The gateway persists payloads into Cloudflare Vectorize / KV, making
    any lesson instantly available to ALL OpenCodeWEB OS users.
    """

    def __init__(self, endpoint: str = SYNC_ENDPOINT, token: str | None = None) -> None:
        self.endpoint = endpoint
        self.token = token or os.environ.get("OPENCODE_SYNC_TOKEN", "")

    def push(self, lessons: list[Lesson]) -> dict[str, Any]:
        """POST a batch of lessons to the sync endpoint."""
        payload = json.dumps(
            {"lessons": [lesson.to_dict() for lesson in lessons], "source": "aia-learning-loop"}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "OpenCodeWEB-AiA"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(self.endpoint, data=payload, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8", errors="replace")
            _logger.info("synced %d lessons to %s", len(lessons), self.endpoint)
            return {"ok": True, "count": len(lessons), "response": body[:500]}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _logger.warning("Cloudflare sync failed (will retry later): %s", exc)
            return {"ok": False, "error": str(exc)}


class GitHubSyncer:
    """Commit learned heuristics/docs to github.com/OpenCodeWEB/AiA.

    Uses the local git checkout of the AiA repo. Every commit is authored
    under ABsUP with a co-authored-by trailer for ABsUPs, satisfying the
    dual-maintainer branding requirement.
    """

    def __init__(
        self,
        repo_dir: Path | None = None,
        branch: str = GIT_BRANCH,
        author_name: str = COMMIT_AUTHOR_NAME,
        author_email: str = COMMIT_AUTHOR_EMAIL,
        coauthor_name: str = COAUTHOR_NAME,
        coauthor_email: str = COAUTHOR_EMAIL,
    ) -> None:
        self.repo_dir = repo_dir or SYS_ROOT
        self.branch = branch
        self.author_name = author_name
        self.author_email = author_email
        self.coauthor = f"{coauthor_name} <{coauthor_email}>"

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the AiA repo directory."""
        return subprocess.run(
            ["git", "-C", str(self.repo_dir), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def sync(self, lessons_file: Path, dry_run: bool = True) -> dict[str, Any]:
        """Commit the lessons file to the AiA repo.

        Args:
            lessons_file: path to the lessons.jsonl to commit.
            dry_run: when True, print the commit plan without committing.

        Returns:
            Status dict. ``ok`` reflects a clean commit (or clean dry-run).
        """
        if not lessons_file.exists():
            return {"ok": False, "error": "lessons file not found"}

        self._git("add", str(lessons_file))
        message = (
            f"learn: federated knowledge sync ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})\n\n"
            f"Co-authored-by: {self.coauthor}"
        )
        if dry_run:
            self._git("reset", "HEAD")
            return {"ok": True, "dry_run": True, "message": message}

        result = self._git("commit", "-m", message, f"--author={self.author_name} <{self.author_email}>")
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or result.stdout.strip()}
        push = self._git("push", "origin", self.branch)
        return {
            "ok": push.returncode == 0,
            "commit": result.stdout.strip(),
            "push": push.stdout.strip() or push.stderr.strip(),
        }


# ---------------------------------------------------------------------------
# Learning Loop
# ---------------------------------------------------------------------------


class LearningLoop:
    """Continuous self-learning + federated sync orchestration.

    Start it from boot with ``run_forever()``; it ingests lessons,
    periodically syncs to Cloudflare (Vectorize/KV via the edge gateway)
    and commits heuristics to GitHub — with no artificial stop condition.
    """

    def __init__(
        self,
        store: LessonStore | None = None,
        extractor: InsightExtractor | None = None,
        cloudflare: CloudflareSyncer | None = None,
        github: GitHubSyncer | None = None,
        interval_s: float = LEARNING_INTERVAL_S,
    ) -> None:
        self.store = store or LessonStore()
        self.extractor = extractor or InsightExtractor()
        self.cloudflare = cloudflare or CloudflareSyncer()
        self.github = github or GitHubSyncer()
        self.interval_s = interval_s
        self.store.load()

    # -- ingestion ---------------------------------------------------------------

    def learn(self, task: str, result: dict[str, Any]) -> Lesson | None:
        """Extract a lesson from a task result and persist it."""
        lesson = self.extractor.extract(task, result)
        if lesson is not None:
            self.store.add(lesson)
            _logger.info("learned lesson %s (%s)", lesson.id, lesson.outcome)
        return lesson

    def recall(self, query: str, top_k: int = 3) -> list[tuple[float, Lesson]]:
        """Retrieve the most relevant past lessons for a task."""
        return self.store.recall(query, top_k=top_k)

    # -- sync ----------------------------------------------------------------------

    def sync(self, dry_run: bool = True) -> dict[str, Any]:
        """Push all new lessons to Cloudflare and commit to GitHub."""
        lessons = self.store.all()
        if not lessons:
            return {"ok": True, "synced": 0, "note": "nothing to sync"}

        cloudflare_result = self.cloudflare.push(lessons)
        github_result = self.github.sync(LESSONS_FILE, dry_run=dry_run)
        return {
            "ok": cloudflare_result.get("ok", False),
            "lessons": len(lessons),
            "cloudflare": cloudflare_result,
            "github": github_result,
        }

    # -- continuous loop ---------------------------------------------------------------

    def run_forever(self, dry_run: bool = True) -> None:
        """Continuously sync learned knowledge.

        Zero-Constraint: the loop runs indefinitely; ``interval_s`` merely
        paces Cloudflare/GitHub network calls, it is not a work quota.
        """
        _logger.info("learning loop started (interval=%ss, dry_run=%s)", self.interval_s, dry_run)
        while True:
            try:
                report = self.sync(dry_run=dry_run)
                _logger.info("sync report: %s", json.dumps(report)[:300])
            except Exception:  # noqa: BLE001 - the loop must never die
                _logger.exception("sync cycle failed — continuing")
            time.sleep(self.interval_s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="learning_loop", description="OpenCodeWEB OS AiA continuous learning loop")
    parser.add_argument("--sync-once", action="store_true", help="perform a single sync cycle")
    parser.add_argument("--forever", action="store_true", help="run the learning loop indefinitely")
    parser.add_argument("--learn", nargs=2, metavar=("TASK", "RESULT_JSON"), help="ingest one task result as a lesson")
    parser.add_argument("--apply", action="store_true", help="with --sync-once: actually commit/push to GitHub")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    loop = LearningLoop()
    if args.learn:
        task, result_json = args.learn
        result: dict[str, Any] = json.loads(result_json)
        lesson = loop.learn(task, result)
        print(json.dumps(lesson.to_dict() if lesson else {}, indent=2))
        return 0 if lesson else 1

    if args.sync_once:
        print(json.dumps(loop.sync(dry_run=not args.apply), indent=2))
        return 0

    if args.forever:
        loop.run_forever(dry_run=not args.apply)
        return 0

    print(json.dumps({"lessons": loop.store.count()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
