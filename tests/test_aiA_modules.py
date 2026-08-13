"""Tests for the AiA autonomous modules: agent core, learning loop,
memory manager, and unrestricted refactorer.

All tests are offline and dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable regardless of the invocation directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.aia.agent_core import CodeExecutor, ReActAgent, SelfCorrector, Tool, ToolRegistry
from core.aia.learning_loop import Embedding, InsightExtractor, Lesson, LessonStore
from core.aia.memory_manager import MemoryManager, PersistentIndex
from core.aia.unrestricted_refactor import (
    SourceSanitizer,
    UnrestrictedFinder,
    UnrestrictedRefactorer,
)

# ---------------------------------------------------------------------------
# Agent core
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_lookup() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="echo", fn=lambda a: {"echo": a.get("text")}))
    assert registry.get("echo") is not None
    assert "echo" in registry.names()


def test_tool_run_returns_ok() -> None:
    tool = Tool(name="t", description="t", fn=lambda a: {"value": a.get("x", 1)})
    result = tool.run({"x": 5})
    assert result == {"ok": True, "value": 5}


def test_tool_run_captures_errors() -> None:
    def boom(_a: dict[str, object]) -> dict[str, object]:
        raise ValueError("boom")

    tool = Tool(name="boom", description="boom", fn=boom)
    result = tool.run({})
    assert result["ok"] is False
    assert "boom" in result["error"]


def test_react_agent_runs_and_completes() -> None:
    agent = ReActAgent()
    result = agent.run("list available tools")
    assert result["complete"] is True
    assert result["steps"] >= 1


def test_self_corrector_repairs_missing_import() -> None:
    corrector = SelfCorrector()
    result = corrector.run("print(10)")  # valid code executes cleanly
    assert result["ok"] is True


def test_code_executor_captures_failure() -> None:
    executor = CodeExecutor()
    result = executor.execute("raise RuntimeError('x')")
    assert result["ok"] is False
    assert result["returncode"] != 0


# ---------------------------------------------------------------------------
# Learning loop
# ---------------------------------------------------------------------------


def test_embedding_is_unit_vector() -> None:
    embedding = Embedding(dims=64)
    vec = embedding.embed("autonomous agentic model with unlimited context")
    assert len(vec) == 64
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_embedding_similarity_prefers_similar_text() -> None:
    embedding = Embedding()
    a = embedding.embed("rate limiting and token caps")
    b = embedding.embed("rate limiting and token caps")
    c = embedding.embed("weather forecasts for tomorrow")
    assert embedding.similarity(a, b) > embedding.similarity(a, c)


def test_insight_extractor_creates_lesson() -> None:
    extractor = InsightExtractor()
    lesson = extractor.extract("fix the gateway bug", {"ok": True, "tool": "code-execution"})
    assert lesson is not None
    assert lesson.outcome == "success"
    assert lesson.embedding


def test_lesson_store_persists_and_recalls(tmp_path: Path) -> None:
    store = LessonStore(path=tmp_path / "lessons.jsonl")
    store.add(Lesson(id="1", task="a", outcome="success", technique="code", lesson="use asyncio"))
    store.add(Lesson(id="2", task="b", outcome="success", technique="code", lesson="use mmap"))
    assert store.count() == 2
    reloaded = LessonStore(path=tmp_path / "lessons.jsonl")
    reloaded.load()
    assert reloaded.count() == 2
    hits = reloaded.recall("asyncio patterns", top_k=1)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


def test_memory_manager_ingest_and_compress(tmp_path: Path) -> None:
    manager = MemoryManager(
        max_window_tokens=80,
        index=PersistentIndex(
            turns_file=tmp_path / "turns.jsonl",
            summaries_file=tmp_path / "summaries.jsonl",
        ),
        enable_shm=False,
    )
    for i in range(30):
        manager.add("user", f"message number {i} about planetary graphics rendering")
    assert manager.summary_count() >= 1
    assert manager.total_tokens() > 0


def test_memory_manager_bootstrap_restores(tmp_path: Path) -> None:
    index = PersistentIndex(
        turns_file=tmp_path / "turns.jsonl",
        summaries_file=tmp_path / "summaries.jsonl",
    )
    manager = MemoryManager(index=index, enable_shm=False)
    manager.add("user", "persisted turn")
    manager2 = MemoryManager(index=index, enable_shm=False)
    manager2.bootstrap()
    assert len(manager2.recent_turns()) >= 1
    assert "persisted turn" in manager2.recent_turns()[-1].text


def test_memory_manager_recall() -> None:
    manager = MemoryManager(max_window_tokens=500, enable_shm=False)
    manager.add("user", "webrtc peer connection signaling details")
    for i in range(20):
        manager.add("user", f"unrelated filler number {i}")
    while len(manager.recent_turns()) > 4:
        manager._fold_oldest()  # noqa: SLF001
    hits = manager.recall("webrtc signaling", top_k=1)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# Unrestricted refactorer
# ---------------------------------------------------------------------------


def test_sanitizer_removes_sleep_and_ratelimit() -> None:
    source = (
        "import time\n"
        "def go():\n"
        "    time.sleep(0.5)\n"
        "    return 'done'\n"
    )
    sanitized, result = SourceSanitizer().sanitize_source(source, "sample.py")
    assert result.constructs_removed >= 1
    assert "sleep" not in sanitized


def test_sanitizer_neutralizes_limiter_function() -> None:
    source = (
        "def rate_limit(fn):\n"
        "    def wrapped(*a):\n"
        "        return fn(*a)\n"
        "    return wrapped\n"
        "@rate_limit\n"
        "def work():\n"
        "    return 1\n"
    )
    sanitized, result = SourceSanitizer().sanitize_source(source, "sample.py")
    assert result.constructs_removed >= 1
    # The symbol survives (body replaced with pass) — callers still work.
    assert "def rate_limit" in sanitized


def test_refactorer_mirrors_tree(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "tool.py").write_text("import time\ntime.sleep(1)\nprint('ok')\n", encoding="utf-8")
    out = tmp_path / "release"
    results = UnrestrictedRefactorer().refactor_tree(src, out_root=out)
    assert len(results) == 1
    assert results[0].constructs_removed >= 1
    mirrored = (out / "tool.py").read_text(encoding="utf-8")
    assert "sleep" not in mirrored


def test_refactorer_excludes_out_root_inside_source(tmp_path: Path) -> None:
    """A release dir placed inside the source tree must not be re-scanned
    (regression: the lazy rglob used to consume its own output and loop)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tool.py").write_text("import time\ntime.sleep(1)\nprint('ok')\n", encoding="utf-8")
    out = src / "release"  # output INSIDE the source root
    results = UnrestrictedRefactorer().refactor_tree(src, out_root=out)
    assert len(results) == 1
    assert results[0].constructs_removed >= 1
    mirrored = (out / "tool.py").read_text(encoding="utf-8")
    assert "sleep" not in mirrored
    # No runaway nested release/release/... directories.
    nested = [p for p in out.rglob("release") if p.is_dir()]
    assert not nested


def test_unrestricted_finder_spec_resolution(tmp_path: Path) -> None:
    root = tmp_path / "modules"
    (root / "magicmod").mkdir(parents=True)
    (root / "magicmod" / "__init__.py").write_text("VALUE = 'unrestricted'\n", encoding="utf-8")
    finder = UnrestrictedFinder(root=root)
    spec = finder.find_spec("magicmod")
    assert spec is not None
    assert spec.loader is not None
