# Track: AiA on GDBx — Decentralized Memory

> Part of **org-gdbx-unification** (GDBX Phase 6)

## Goal
Migrate AiA `core/` from GunX `gun-relay/bridge.js` (via `GUN_BRIDGE_TOKEN`) to GDBx self-sovereign mesh.

## Spec
- `core/gdbx_bridge.py` wraps `gdbx_py` — `GdbxBridge` with `register()`, `put_event()`, `put_memory()`, `recall()`
- Env: `GDBX_API`, `GDBX_PUB`, `GDBX_PRIV`, `GDBX_PUBKEY_HEX`, `GDBX_ADDR` (fallback to legacy gun if missing)
- Brain: `aia/memory/<session>` deltas + `aia/vectors/<id>` vectors (cosine search brute-force via `GdbxClient.search_vector`, future `sqlite-vec`)
- Storage key: `aia/events/<kind>/<ts>-<uuid>` — flat JSON value (GDBx flat-primitive rule)
- Tests: `tests/test_gdbx_bridge.py` (mock, no network)

## Links
- GDBX spec: `GDBX/conductor/tracks/org-gdbx-unification/spec.md`
- SDK: `GDBX/packages/gdbx-py/`
