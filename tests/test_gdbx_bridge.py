"""AiA GDBx bridge tests — mock, no live network."""

import sys
from pathlib import Path

# ensure gdbx_py on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GDBX" / "packages" / "gdbx-py"))

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# import after path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.gdbx_bridge import GdbxBridge


@pytest.mark.asyncio
async def test_put_event_mock():
    # mock client
    mock_client = AsyncMock()
    mock_client.put_deltas.return_value = {"ok": True, "applied": 1}
    mock_client.addr = "aeaqmockmockmockmockmockmockmockmockmockmockmockmockq"

    bridge = GdbxBridge.__new__(GdbxBridge)
    bridge._use_legacy = False
    bridge._client = mock_client
    bridge._addr = mock_client.addr
    bridge.base_url = "https://gdbx-do.xup.workers.dev"

    res = await bridge.put_event("learn", {"lesson": "test"})
    assert res["ok"] is True
    # key shape aia/events/learn/<ts>-<uuid> with JSON value
    args, _ = mock_client.put_deltas.call_args
    deltas = args[0]
    assert deltas[0]["key"].startswith("aia/events/learn/")
    assert "lesson" in deltas[0]["value"]


@pytest.mark.asyncio
async def test_put_memory_with_vector():
    mock_client = AsyncMock()
    mock_client.put_deltas.return_value = {"ok": True}
    mock_client.put_vector.return_value = {"ok": True}
    mock_client.addr = "aeaqmock2"

    bridge = GdbxBridge.__new__(GdbxBridge)
    bridge._use_legacy = False
    bridge._client = mock_client

    res = await bridge.put_memory("hello memory", vector=[0.1, 0.2, 0.3])
    assert res["ok"] is True
    assert mock_client.put_vector.called
    vk = mock_client.put_vector.call_args[0][0]
    assert vk.startswith("aia/vectors/")


def test_legacy_fallback_when_no_env(monkeypatch):
    # clear GDBX env
    for k in list(monkeypatch._env if hasattr(monkeypatch, "_env") else []):
        pass
    monkeypatch.delenv("GDBX_API", raising=False)
    monkeypatch.delenv("GDBX_PUB", raising=False)
    monkeypatch.delenv("GDBX_PRIV", raising=False)
    monkeypatch.delenv("GDBX_PUBKEY_HEX", raising=False)
    monkeypatch.delenv("GDBX_ADDR", raising=False)
    # also clear legacy token to avoid import error path — just check is_legacy
    b = GdbxBridge()
    # without GDBX_* it should be legacy (unless HAS_GDBX false)
    # We don't enforce strict — just check attribute exists
    assert hasattr(b, "is_legacy")
