"""Cluster Heartbeat Election - regional anchor election on the 3D mesh.

Peers within the same Y region (latency cluster) announce themselves with
heartbeats carrying their grid coordinates and evolution stage. The most
stable, most evolved live peer is elected as the Regional Cluster Anchor
(Z=1); the anchor then aggregates cluster knowledge toward the global
consensus layer (Z=2).

Election scoring (local, deterministic, no central authority):
  stage * 1000 + heartbeat_count + freshness(now - ts)

Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import time
from typing import Any


class ClusterHeartbeatElection:
    """Tracks Y-region heartbeats and elects the regional cluster anchor."""

    def __init__(self, region: str = "asia-south", heartbeat_ttl: int = 60, min_quorum: int = 3) -> None:
        self.region = region
        self.heartbeat_ttl = heartbeat_ttl
        self.min_quorum = min_quorum
        self.heartbeats: dict[str, dict[str, Any]] = {}
        self.anchor: str | None = None
        self.elections = 0

    # ------------------------------------------------------------------ #
    # heartbeat + election
    # ------------------------------------------------------------------ #
    def heartbeat(
        self,
        node_id: str,
        coords: dict[str, int],
        stage: int = 0,
        ts: int | None = None,
    ) -> dict[str, Any]:
        """Record one peer heartbeat and re-run the local election."""
        now = ts if ts is not None else int(time.time())
        entry = self.heartbeats.get(node_id)
        if entry is None:
            entry = {"ts": now, "count": 0, "coords": {"x": 0, "y": 0, "z": 0}, "stage": 0}
            self.heartbeats[node_id] = entry
        entry["ts"] = now
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["coords"] = {k: int(coords.get(k, 0)) for k in ("x", "y", "z")}
        entry["stage"] = max(0, min(2, int(stage)))
        self._prune(now)
        changed = self._elect(now)
        return {
            "ok": True,
            "region": self.region,
            "anchor": self.anchor,
            "anchor_changed": changed,
            "live_peers": len(self.heartbeats),
        }

    def _prune(self, now: int) -> None:
        stale = [n for n, e in self.heartbeats.items() if now - int(e["ts"]) > self.heartbeat_ttl]
        for node_id in stale:
            del self.heartbeats[node_id]
        if self.anchor not in self.heartbeats:
            self.anchor = None

    def _elect(self, now: int) -> bool:
        live = {n: e for n, e in self.heartbeats.items() if now - int(e["ts"]) <= self.heartbeat_ttl}
        if not live:
            return False
        winner = max(
            live.items(),
            key=lambda item: int(item[1]["stage"]) * 1000 + int(item[1]["count"]) + 1.0 / (1.0 + (now - int(item[1]["ts"]))),
        )[0]
        if winner != self.anchor:
            self.anchor = winner
            self.elections += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "anchor": self.anchor,
            "live_peers": len(self.heartbeats),
            "elections": self.elections,
            "heartbeat_ttl": self.heartbeat_ttl,
            "min_quorum": self.min_quorum,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "cluster_heartbeat",
            "region": self.region,
            "heartbeat_ttl": self.heartbeat_ttl,
            "min_quorum": self.min_quorum,
            "anchor": self.anchor,
            "elections": self.elections,
            "heartbeats": self.heartbeats,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ClusterHeartbeatElection:
        cluster = cls(
            region=str(data.get("region", "asia-south")),
            heartbeat_ttl=int(data.get("heartbeat_ttl", 60)),
            min_quorum=int(data.get("min_quorum", 3)),
        )
        cluster.anchor = data.get("anchor")
        cluster.elections = int(data.get("elections", 0))
        cluster.heartbeats = dict(data.get("heartbeats", {}))
        return cluster
