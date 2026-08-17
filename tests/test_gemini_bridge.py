"""Tests for the Gemini bridge (zero API key, OAuth session)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from gemini_bridge import GeminiBridge, GeminiUnavailableError  # noqa: E402


def test_unavailable_when_cli_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    bridge = GeminiBridge()
    assert bridge.available() is False
    status = bridge.status()
    assert status["connected"] is False
    assert status["api_key"] == "none"
    with pytest.raises(GeminiUnavailableError):
        bridge.ask("hello")


def test_status_connected_when_cli_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)
    bridge = GeminiBridge()
    assert bridge.available() is True
    status = bridge.status()
    assert status["connected"] is True
    assert "gemini-cli" in status["provider"]


def test_ask_runs_command_and_returns_stdout(monkeypatch):
    import subprocess

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gemini")

    class FakeProc:
        returncode = 0
        stdout = "real gemini answer"
        stderr = ""

    def fake_run(cmd, **kw):
        assert "gemini" in cmd and "hello" in " ".join(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    bridge = GeminiBridge()
    out, err = bridge.ask("hello")
    assert out == "real gemini answer"


def test_cookie_bridge_experimental_raises(monkeypatch):
    monkeypatch.setenv("AIA_GEMINI_COOKIE_BRIDGE", "1")
    bridge = GeminiBridge()
    assert bridge.cookie_bridge is True
    with pytest.raises(NotImplementedError):
        bridge.status()


def test_connect_returns_status(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    bridge = GeminiBridge()
    result = bridge.connect(session_cookie="abc")
    assert result["connected"] is False
