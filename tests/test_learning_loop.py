"""Tests for the learning hub (promotion, anti-patterns, user profile)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_loop import LearningHub  # noqa: E402


def test_promotion_after_threshold(tmp_path):
    hub = LearningHub(tmp_path, promotion_threshold=3)
    prompt = "fix the payment webhook retry storm"
    output = "backoff with jitter + idempotency keys"
    assert hub.learn_from_execution("gemini", prompt, output) is None
    assert hub.learn_from_execution("opencode", prompt, output) is None
    skill = hub.learn_from_execution("mock", prompt, output)
    assert skill is not None
    assert skill["source"] == "promoted:mock"
    assert skill["solution"] == output


def test_failure_never_promotes(tmp_path):
    hub = LearningHub(tmp_path, promotion_threshold=2)
    prompt = "deploy the kubernetes manifest"
    for _ in range(4):
        assert hub.learn_from_execution("gemini", prompt, "boom", success=False) is None


def test_anti_pattern_recorded(tmp_path):
    hub = LearningHub(tmp_path)
    rec = hub.anti_pattern("connect to the database", "connection refused", "timeout")
    assert rec["prompt_pattern"] == "connect to the database"
    assert hub.promotions == {}  # failures never count toward promotion


def test_observation_log_rotates(tmp_path):
    hub = LearningHub(tmp_path)
    for i in range(5):
        hub.record_execution("mock", f"task {i}", f"out {i}")
    lines = hub.observation_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    import json

    assert json.loads(lines[0])["source_model"] == "mock"


def test_user_profile_learning(tmp_path):
    hub = LearningHub(tmp_path)
    profile = {"languages": {}, "styles": {}, "interactions": 0}
    hub.update_user_profile(profile, "write a fast python script", "pip install && run")
    assert profile["languages"].get("python", 0) >= 1
    assert profile["styles"].get("performance", 0) >= 1
    assert profile["interactions"] == 1
