"""Spatial Grid Coordinator - maps the local brain onto the NxMxK 3D mesh.

Decentralized global-brain architecture (per the Gemini design):

  X axis  -> Functional Specialization Layer
             X=0 perception (autoencoder, anomaly)
             X=1 sequence  (rnn, attention)
             X=2 decision  (dqn, hopfield)
             X=3 executive (self-evolving, curriculum)
  Y axis  -> Geographic / Network Latency Region
             asia-south, eu-central, us-east, apac, global
  Z axis  -> Temporal Evolution Stage
             Z=0 edge node -> Z=1 regional cluster anchor -> Z=2 global consensus

Each node derives its own (x, y, z) from local competence signals and
publishes a signed, flat-primitives metadata block to the GunX mesh
(gun.get('os').get('neural').get('3d_grid').get(node_id)). The HMAC
signature binds {coords, spatial_layer, region, ts} to the per-install
secret, so peers can verify spatial claims without a central authority.

Privacy: only abstract coordinates + competence-derived layers are shared;
raw samples, private weights and secrets never leave the device.

Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

# X-axis: functional specialization layers (index -> layer id)
X_LAYERS: dict[int, str] = {
    0: "perception",
    1: "sequence",
    2: "decision",
    3: "executive",
}

# X-layer membership: module name -> layer index
_X_MODULES: dict[str, int] = {
    "autoencoder": 0,
    "anomaly": 0,
    "rnn": 1,
    "attention": 1,
    "dqn": 2,
    "hopfield": 2,
    "self_evolving": 3,
    "curriculum": 3,
}

# Y-axis: geographic / network-latency regions
Y_REGIONS: list[str] = ["asia-south", "eu-central", "us-east", "apac", "global"]

# Z-axis: temporal evolution stages
Z_STAGES: dict[int, str] = {
    0: "edge",
    1: "cluster",
    2: "global",
}


def _sign(payload: dict[str, Any], secret: str) -> str:
    """HMAC-SHA256 signature of a flat JSON payload (same convention as federated_sync)."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _verify(payload: dict[str, Any], signature: str, secret: str) -> bool:
    expected = _sign(payload, secret)
    return hmac.compare_digest(expected, signature)


class SpatialGridCoordinator:
    """Computes, signs and tracks spatial identities on the 3D neural mesh."""

    def __init__(
        self,
        node_id: str | None = None,
        secret: str = "aia-spatial-default",
        region: str | None = None,
    ) -> None:
        self.node_id = node_id or f"node-{int(time.time() * 1000) % 100000}"
        self.secret = secret
        self.region = region
        self.stage = 0  # Z coordinate: 0=edge, 1=cluster, 2=global
        self._competence: dict[str, float] = {}
        self.nodes: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # local state
    # ------------------------------------------------------------------ #
    def update_competence(self, module: str, score: float) -> None:
        """Feed a module mastery score (0..1) to steer the X coordinate."""
        if 0.0 <= score <= 1.0:
            self._competence[module] = score

    # ------------------------------------------------------------------ #
    # coordinate derivation
    # ------------------------------------------------------------------ #
    def _x_from_state(self) -> int:
        layer_scores = [0.0, 0.0, 0.0, 0.0]
        for module, score in self._competence.items():
            layer = _X_MODULES.get(module)
            if layer is not None:
                layer_scores[layer] += score
        return max(range(len(layer_scores)), key=lambda i: layer_scores[i])

    def _y_from_state(self) -> int:
        if self.region is not None:
            if self.region in Y_REGIONS:
                return Y_REGIONS.index(self.region)
            return Y_REGIONS.index("global")
        # stable hash of the node id -> latency region index
        digest = hashlib.sha256(self.node_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:2], "big") % len(Y_REGIONS)

    def coords(self) -> dict[str, int]:
        """Current (x, y, z) position of this node on the grid."""
        return {
            "x": self._x_from_state(),
            "y": self._y_from_state(),
            "z": max(0, min(2, self.stage)),
        }

    def spatial_metadata(self) -> dict[str, Any]:
        """Unsigned spatial identity block (flat primitives)."""
        coords = self.coords()
        return {
            "kind": "spatial_node",
            "node_id": self.node_id,
            "coords": coords,
            "spatial_layer": Z_STAGES[coords["z"]],
            "region": Y_REGIONS[coords["y"]],
            "weight_delta": None,  # reserved: published by spatial_temporal_sync
            "last_pulse": None,  # reserved: heartbeat timestamp
        }

    # ------------------------------------------------------------------ #
    # signed registration
    # ------------------------------------------------------------------ #
    def register(self, peer_id: str) -> dict[str, Any]:
        """Register a peer and get its signed spatial metadata block.

        Same peer always maps to the same coordinates (deterministic).
        """
        if peer_id not in self.nodes:
            coords = self.coords()
            block: dict[str, Any] = {
                "kind": "spatial_node",
                "node_id": peer_id,
                "ts": int(time.time()),
                "coords": coords,
                "spatial_layer": Z_STAGES[coords["z"]],
                "region": Y_REGIONS[coords["y"]],
                "weight_delta": None,
                "last_pulse": None,
            }
            block["hmac_signature"] = _sign(block, self.secret)
            self.nodes[peer_id] = block
        return dict(self.nodes[peer_id])

    def verify(self, block: dict[str, Any]) -> bool:
        """Constant-time verification of a signed spatial block."""
        if "hmac_signature" not in block:
            return False
        sig = block.pop("hmac_signature")
        ok = _verify(block, sig, self.secret)
        block["hmac_signature"] = sig
        return ok

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        coords = self.coords()
        return {
            "node_id": self.node_id,
            "coords": coords,
            "spatial_layer": Z_STAGES[coords["z"]],
            "region": Y_REGIONS[coords["y"]],
            "stage": self.stage,
            "competence": self._competence,
            "registered": len(self.nodes),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "spatial_grid",
            "node_id": self.node_id,
            "region": self.region,
            "stage": self.stage,
            "competence": self._competence,
            "nodes": self.nodes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SpatialGridCoordinator:
        coord = cls(
            node_id=str(data.get("node_id", "node-restored")),
            region=data.get("region"),
        )
        coord.stage = int(data.get("stage", 0))
        coord._competence = dict(data.get("competence", {}))
        coord.nodes = dict(data.get("nodes", {}))
        return coord
