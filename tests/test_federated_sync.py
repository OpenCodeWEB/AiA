"""Tests for the federated swarm sync — privacy guarantees are the core assertion."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from federated_learning_sync import AiAFederatedSync, _abstract_solution, _categorize  # noqa: E402


def _setup(tmp_path):
    knowledge = {"learned_from_models": [], "skills": []}
    sync = AiAFederatedSync(tmp_path / "brain", knowledge, endpoint="https://hub.test/v1")
    return sync, knowledge


def test_anonymize_never_leaks_raw_text(tmp_path):
    sync, _ = _setup(tmp_path)
    record = {
        "source_model": "gemini",
        "prompt_pattern": "fix login bug in /Users/abisl/secret-project/api/auth.py",
        "learned_solution": (
            "def fix_login(): token = 'my-private-token-abc123'; "
            "url = 'https://api.example.com/secret'; return token + url"
        ),
    }
    payload = sync.anonymize_pattern(record)
    serialized = json.dumps(payload)
    assert "login" not in serialized
    assert "secret" not in serialized
    assert "my-private-token" not in serialized
    assert "https://" not in serialized
    assert "/Users/" not in serialized
    assert "gemini" not in serialized
    assert isinstance(payload["feature_vector"], list)
    assert all(isinstance(x, float) for x in payload["feature_vector"])
    assert payload["signature"] and len(payload["signature"]) == 32


def test_abstract_solution_strips_identifiers_and_strings():
    text = "def my_function(): secret = 'abc123'; url = 'https://x.com/y'; path = '/tmp/private.txt'; return 42"
    abstract = _abstract_solution(text)
    assert "abc123" not in abstract
    assert "https://" not in abstract
    assert "private" not in abstract
    assert "<str>" in abstract or "<url>" in abstract or "<path>" in abstract
    assert "def" in abstract and "return" in abstract  # structure preserved


def test_categorize():
    assert _categorize("flutter widget glassmorphism") == "flutter_ui"
    assert _categorize("python traceback debugging") == "python_debug"
    assert _categorize("totally unknown topic xyz") == "general_coding"


def test_signature_binds_to_instance_secret(tmp_path):
    sync, _ = _setup(tmp_path)
    p1 = sync.anonymize_pattern({"prompt_pattern": "x", "learned_solution": "same solution text"})
    # new instance (new secret) → different signature for identical content
    sync2 = AiAFederatedSync(tmp_path / "brain2", {"learned_from_models": []}, endpoint="https://hub.test/v1")
    p2 = sync2.anonymize_pattern({"prompt_pattern": "x", "learned_solution": "same solution text"})
    assert p1["signature"] != p2["signature"]
    # same instance → stable signature
    p3 = sync.anonymize_pattern({"prompt_pattern": "x", "learned_solution": "same solution text"})
    assert p1["signature"] == p3["signature"]


def test_push_watermark_advances_only_on_success(tmp_path):
    sync, knowledge = _setup(tmp_path)
    knowledge["learned_from_models"] = [
        {"source_model": "mock", "prompt_pattern": "p1", "learned_solution": "s1", "ts": 1},
        {"source_model": "mock", "prompt_pattern": "p2", "learned_solution": "s2", "ts": 2},
    ]

    def ok_post(url, body):
        assert "pattern" in json.dumps(body)  # never raw text
        assert body["device"]
        return {"ok": True, "received": 2}

    sync._post = ok_post
    result = sync.sync_local_knowledge_to_global_swarm()
    assert result["pushed"] == 2
    assert sync.state["uploaded_watermark"] == 2

    # failure → watermark untouched → retried later
    sync2 = AiAFederatedSync(tmp_path / "brain", knowledge, endpoint="https://hub.test/v1")

    def fail_post(url, body):
        import urllib.error

        raise urllib.error.URLError("offline")

    sync2._post = fail_post
    sync2.state["uploaded_watermark"] = 1  # only 1 pushed so far
    result2 = sync2.sync_local_knowledge_to_global_swarm()
    assert result2["pushed"] == 0
    assert sync2.state["uploaded_watermark"] == 1


def test_patch_download_validates_and_dedupes(tmp_path):
    sync, knowledge = _setup(tmp_path)
    knowledge["skills"] = [{"signature": "dup-sig-12345678"}]

    sync._get = lambda url: {
        "patches": [
            {
                "category": "flutter_ui",
                "pattern": "flutter overflow fix",
                "solution_template": "wrap in SingleChildScrollView",
                "signature": "new-sig-12345678",
                "ts": 5,
            },
            {"category": "flutter_ui", "pattern": "dup", "solution_template": "x", "signature": "dup-sig-12345678", "ts": 6},
            {"category": "x", "pattern": "bad", "solution_template": "y", "signature": "short", "ts": 7},  # invalid signature
            "not-a-dict",
        ],
        "server_ts": 10,
    }
    result = sync.download_global_skill_updates()
    assert result["applied"] == 1  # only the new valid one
    skills = knowledge["skills"]
    assert len(skills) == 2
    swarm_skill = skills[1]
    assert swarm_skill["source"] == "swarm"
    assert swarm_skill["confirmed"] is False
    assert sync.state["last_patch_ts"] == 10
