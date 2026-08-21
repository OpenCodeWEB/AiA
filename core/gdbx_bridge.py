"""
AiA GDBx Bridge — GDBx as decentralized memory for AiA.

Replaces gun-relay bridge with GDBx self-sovereign mesh.
Falls back to legacy GUN_BRIDGE_TOKEN path if GDBX_* env not set.

Env:
  GDBX_API=https://gdbx-do.xup.workers.dev
  GDBX_ADDR (auto-derived if not set)
  GDBX_PUB (x.y)
  GDBX_PRIV (b64url raw d)
  GDBX_PUBKEY_HEX (04... 130 hex)
"""

import os
import time
import uuid
import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Try GDBx Python SDK (shared from GDBX repo)
try:
    import sys
    from pathlib import Path

    # add GDBX python package to path (relative)
    gdbx_pkg = Path(__file__).parent.parent.parent.parent / "GDBX" / "packages" / "gdbx-py"
    if gdbx_pkg.exists() and str(gdbx_pkg) not in sys.path:
        sys.path.insert(0, str(gdbx_pkg))
    from gdbx_py.crypto import pair as gdbx_pair
    from gdbx_py.codec import make_address
    from gdbx_py.client import GdbxClient

    HAS_GDBX = True
except Exception as e:
    log.warning(f"gdbx_py not available: {e}")
    HAS_GDBX = False
    GdbxClient = None  # type: ignore
    gdbx_pair = None  # type: ignore
    make_address = None  # type: ignore


class GdbxBridge:
    """Thin AiA → GDBx bridge. One bridge per AiA instance (singleton)."""

    def __init__(self, base_url: Optional[str] = None, pair: Optional[dict] = None, pubkey_hex: Optional[str] = None):
        self.base_url = base_url or os.getenv("GDBX_API", "https://gdbx-do.xup.workers.dev")
        self._client: Optional[GdbxClient] = None
        self._pair = pair
        self._pubkey_hex = pubkey_hex
        self._addr: Optional[str] = None
        self._use_legacy = False

        if not HAS_GDBX:
            log.warning("HAS_GDBX false — will use legacy gun bridge")
            self._use_legacy = True
            return

        # resolve identity from env or generate ephemeral
        pub = os.getenv("GDBX_PUB")
        priv = os.getenv("GDBX_PRIV")
        hex_env = os.getenv("GDBX_PUBKEY_HEX")
        addr_env = os.getenv("GDBX_ADDR")

        if pub and priv and hex_env:
            self._pair = {"pub": pub, "priv": priv, "pubkey_hex": hex_env}
            self._pubkey_hex = hex_env
            self._addr = addr_env or make_address(hex_env)
        elif pair is not None:
            self._pair = pair
            self._pubkey_hex = pubkey_hex or pair.get("pubkey_hex")
            self._addr = make_address(self._pubkey_hex)
        else:
            # check if any GDBX env set — if not, fallback
            if not any([base_url, pub, priv, hex_env, addr_env]):
                # no GDBX config — legacy path
                log.info("GDBX_* not set — using legacy gun bridge (set GDBX_* to migrate)")
                self._use_legacy = True
                return
            # otherwise generate ephemeral pair (for dev)
            self._pair = gdbx_pair()
            self._pubkey_hex = self._pair["pubkey_hex"]
            self._addr = make_address(self._pubkey_hex)
            log.info(f"Generated ephemeral GDBx identity {self._addr[:12]}...")

        if not self._use_legacy:
            self._client = GdbxClient(self.base_url, self._pair, self._pubkey_hex)
            self._addr = self._client.addr

    @property
    def addr(self) -> Optional[str]:
        return self._addr

    @property
    def is_legacy(self) -> bool:
        return self._use_legacy

    async def register(self) -> dict:
        if self._use_legacy:
            return {"ok": True, "legacy": True}
        assert self._client is not None
        try:
            return await self._client.register_did()
        except Exception as e:
            # already registered is ok (200)
            if "already" in str(e).lower() or "exists" in str(e).lower():
                return {"ok": True, "already": True}
            raise

    def register_sync(self) -> dict:
        if self._use_legacy:
            return {"ok": True, "legacy": True}
        assert self._client is not None
        return self._client.register_did_sync()

    async def put_event(self, kind: str, payload: Any) -> dict:
        """Publish AiA event -> GDBx delta `aia/events/<kind>/<ts>-<uuid>`."""
        if self._use_legacy:
            # fallback: try legacy gun bridge if available
            try:
                from core.gun_bridge import put as gun_put  # type: ignore

                key = f"os/aia/events/{kind}/{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
                return gun_put(key, payload)  # type: ignore
            except Exception as e:
                log.warning(f"legacy gun put failed: {e}")
                return {"ok": False, "error": str(e), "legacy": True}

        assert self._client is not None
        key = f"aia/events/{kind}/{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        # flatten payload to primitive (GDBx rule: flat values only)
        if isinstance(payload, dict):
            # store as JSON string under single key (flat)
            deltas = [{"key": key, "value": json.dumps(payload, ensure_ascii=False)}]
        else:
            deltas = [{"key": key, "value": str(payload)}]
        return await self._client.put_deltas(deltas)

    def put_event_sync(self, kind: str, payload: Any) -> dict:
        import asyncio

        return asyncio.run(self.put_event(kind, payload))

    async def put_memory(self, text: str, vector: Optional[List[float]] = None, key: Optional[str] = None) -> dict:
        """Store agent memory + optional vector."""
        if self._use_legacy:
            return await self.put_event("memory", {"text": text})

        assert self._client is not None
        k = key or f"aia/memory/{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        deltas = [{"key": k, "value": text}]
        res = await self._client.put_deltas(deltas)
        if vector is not None:
            vk = k.replace("aia/memory/", "aia/vectors/")
            await self._client.put_vector(vk, text, vector)
        return res

    async def recall(self, query_vec: List[float], top_k: int = 5) -> List[dict]:
        if self._use_legacy or self._client is None:
            return []
        return await self._client.search_vector(query_vec, top_k=top_k, prefix="aia/vectors/")

    # sync wrappers for non-async callers (AiA executors often sync)
    def put_memory_sync(self, *a, **kw):
        import asyncio

        return asyncio.run(self.put_memory(*a, **kw))

    def recall_sync(self, *a, **kw):
        import asyncio

        return asyncio.run(self.recall(*a, **kw))


# convenience singleton getter
_bridge: Optional[GdbxBridge] = None


def get_bridge() -> GdbxBridge:
    global _bridge
    if _bridge is None:
        _bridge = GdbxBridge()
    return _bridge


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AiA GDBx Bridge demo")
    ap.add_argument("--check", action="store_true", help="live check against GDBX_API")
    ap.add_argument("--demo", action="store_true", help="write demo event")
    args = ap.parse_args()

    b = GdbxBridge()
    print(f"Bridge addr: {b.addr} legacy={b.is_legacy} api={b.base_url}")

    if args.check or args.demo:
        import asyncio

        async def run():
            print("register...", await b.register())
            print("put demo...", await b.put_event("demo", {"msg": "hello from AiA", "ts": int(time.time()*1000)}))
            print("put_memory...", await b.put_memory("demo memory", vector=[0.1, 0.2, 0.3] * 10))
            if b._client:
                print("get...", await b._client.get_deltas("aia/demo"))

        asyncio.run(run())
