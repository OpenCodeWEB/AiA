"""Tests for the AiA Master Engine core (process_task pipeline)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aia_core_engine import AiAMasterEngine  # noqa: E402
from executors.mock import MockExecutor  # noqa: E402


def _engine(tmp_path, executors=None):
    return AiAMasterEngine(brain_dir=tmp_path / "brain", executors=executors or [MockExecutor()], verbose=False)


def test_unknown_task_delegates_to_mock_and_learns(tmp_path):
    engine = _engine(tmp_path)
    result = engine.process_task("Write a Flutter UI component with custom glassmorphism")
    assert result["ok"] is True
    assert result["mode"] == "delegated"
    assert result["source"] == "mock"
    assert engine.knowledge["learned_from_models"]  # observation recorded
    assert engine.memory.stats()["entries"] == 2  # user + aia turns ingested


def test_learned_skill_promotes_after_threshold_and_runs_natively(tmp_path):
    engine = _engine(tmp_path)
    prompt = "Fix the flutter widget overflow error"
    # 3 successful delegated runs of the same pattern → promotion threshold
    for _ in range(3):
        engine.process_task(prompt)
    promoted = [s for s in engine.knowledge["skills"] if s.get("source", "").startswith("promoted:")]
    assert promoted, "pattern should be promoted to a skill after 3 successes"

    # Now the same pattern is routed natively (skill pattern matches)
    result = engine.process_task(prompt)
    assert result["mode"] == "native"
    assert result["source"] == "aia-native"
    assert promoted[0]["usage_count"] >= 1


def test_dedupe_prevents_duplicate_learned_records(tmp_path):
    engine = _engine(tmp_path)
    engine.process_task("Refactor the api client timeout handling")
    n = len(engine.knowledge["learned_from_models"])
    engine.process_task("Refactor the api client timeout handling")
    assert len(engine.knowledge["learned_from_models"]) == n  # dedupe by sig


def test_knowledge_persists_across_engine_instances(tmp_path):
    brain = tmp_path / "brain"
    engine = AiAMasterEngine(brain_dir=brain, executors=[MockExecutor()], verbose=False)
    engine.process_task("Optimize the database query with an index")
    engine2 = AiAMasterEngine(brain_dir=brain, executors=[MockExecutor()], verbose=False)
    assert engine2.knowledge["learned_from_models"]
    assert engine2.profile["interactions"] >= 1
    assert engine2.memory.stats()["entries"] == engine.memory.stats()["entries"]


def test_status_shape(tmp_path):
    engine = _engine(tmp_path)
    status = engine.status()
    assert status["ok"] is True
    assert "memory" in status and "executors" in status and "swarm" in status
    assert status["swarm"]["endpoint"].endswith("/v1")


def test_prd_demo_runs_and_returns_output(tmp_path):
    engine = _engine(tmp_path)
    result = engine.process_task("Write a Flutter UI component with custom glassmorphism")
    assert "output" in result and result["output"]
    assert json.loads(json.dumps(result))  # JSON-serializable
