"""Federated Neural Weight Sync - FedAvg over the GunX mesh.

Multiple brains (this PC, other users' devices, browser instances) each learn
on their own private data. Instead of sharing raw solutions, each node:

  1. computes its weight DELTA since the last sync (flat primitives),
  2. signs it with HMAC-SHA256 (per-install secret, never transmitted),
  3. publishes {node_id, round, weight_deltas, sample_count} to the mesh,
  4. aggregates all incoming deltas with Federated Averaging (FedAvg),
     weighted by sample_count, and applies the merged update to the brain.

Privacy: raw samples/weights never leave the device - only abstract deltas.
Transport is pluggable: publish() accepts a callback so the GunX bridge or
any flat-primitive channel can be attached; without transport the module
keeps a local round buffer (offline federation).

Pure stdlib (hmac, hashlib, json), flat-primitives JSON only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable
from typing import Any


def _sign(payload: dict[str, Any], secret: str) -> str:
    """HMAC-SHA256 signature of a flat JSON payload."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _verify(payload: dict[str, Any], signature: str, secret: str) -> bool:
    """Constant-time verification of a payload signature."""
    expected = _sign(payload, secret)
    return hmac.compare_digest(expected, signature)


class FederatedSync:
    """Federated averaging coordinator for SelfEvolvingNN-style brains."""

    def __init__(
        self,
        secret: str = "aia-federated-default",
        node_id: str | None = None,
        min_delta_samples: int = 8,
        publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.secret = secret
        self.node_id = node_id or f"node-{int(time.time() * 1000) % 100000}"
        self.min_delta_samples = min_delta_samples
        self.publish = publish
        self.round = 0
        self.last_weights: list[list[list[float]]] | None = None
        self.last_biases: list[list[float]] | None = None
        self.published: int = 0
        self.received: list[dict[str, Any]] = []
        self.aggregations: int = 0

    # ------------------------------------------------------------------ #
    # delta computation & signing
    # ------------------------------------------------------------------ #
    def compute_delta(self, weights: list[list[list[float]]], biases: list[list[float]]) -> dict[str, Any] | None:
        """Delta since the last sync. None when the brain hasn't moved enough."""
        if self.last_weights is None:
            self.last_weights = [list(map(list, w)) for w in weights]
            self.last_biases = [list(b) for b in biases]
            return None
        delta_w = [
            [
                [
                    weights[layer_idx][i][j] - self.last_weights[layer_idx][i][j]
                    for j in range(len(self.last_weights[layer_idx][i]))
                ]
                for i in range(len(self.last_weights[layer_idx]))
            ]
            for layer_idx in range(len(self.last_weights))
        ]
        delta_b = [
            [biases[layer_idx][i] - self.last_biases[layer_idx][i] for i in range(len(self.last_biases[layer_idx]))]
            for layer_idx in range(len(self.last_biases))
        ]
        # snapshot the new baseline
        self.last_weights = [list(map(list, w)) for w in weights]
        self.last_biases = [list(b) for b in biases]

        magnitude = sum(abs(v) for layer in delta_w for row in layer for v in row)
        magnitude += sum(abs(v) for layer in delta_b for v in layer)
        if magnitude < 1e-6:
            return None
        return {"delta_w": delta_w, "delta_b": delta_b}

    def make_update(
        self,
        weights: list[list[list[float]]],
        biases: list[list[float]],
        sample_count: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build a signed federated update (flat primitives)."""
        delta = self.compute_delta(weights, biases)
        if delta is None:
            return None
        self.round += 1
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "round": self.round,
            "ts": int(time.time()),
            "sample_count": sample_count,
            **delta,
        }
        if extra:
            payload.update(extra)
        payload["signature"] = _sign(payload, self.secret)
        if self.publish is not None:
            self.publish(payload)
        self.published += 1
        return payload

    # ------------------------------------------------------------------ #
    # aggregation (FedAvg)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _verify_update(update: dict[str, Any], secret: str) -> bool:
        if "signature" not in update:
            return False
        sig = update.pop("signature")
        ok = _verify(update, sig, secret)
        update["signature"] = sig
        return ok

    def receive(self, update: dict[str, Any]) -> bool:
        """Ingest one signed update from the mesh. Returns accepted flag."""
        if not self._verify_update(update, self.secret):
            return False
        if update.get("node_id") == self.node_id:
            return False
        if not isinstance(update.get("delta_w"), list) or not isinstance(update.get("delta_b"), list):
            return False
        self.received.append(update)
        return True

    def aggregate(
        self,
        weights: list[list[list[float]]],
        biases: list[list[float]],
    ) -> dict[str, Any]:
        """FedAvg: apply all received deltas weighted by sample_count.

        Returns the merge report and CLEARS the pending update queue
        (the merged knowledge is now baked into the local brain).
        """
        if not self.received:
            return {"merged": False, "updates": 0, "note": "no updates queued"}
        total_samples = sum(int(u.get("sample_count", 1)) for u in self.received)
        scale = 1.0 / total_samples if total_samples else 0.0

        # weighted average delta
        avg_w = [
            [
                [
                    sum(u["delta_w"][layer_idx][i][j] * int(u.get("sample_count", 1)) for u in self.received) * scale
                    for j in range(len(self.received[0]["delta_w"][layer_idx][i]))
                ]
                for i in range(len(self.received[0]["delta_w"][layer_idx]))
            ]
            for layer_idx in range(len(self.received[0]["delta_w"]))
        ]
        avg_b = [
            [
                sum(u["delta_b"][layer_idx][i] * int(u.get("sample_count", 1)) for u in self.received) * scale
                for i in range(len(self.received[0]["delta_b"][layer_idx]))
            ]
            for layer_idx in range(len(self.received[0]["delta_b"]))
        ]

        # apply to the local brain
        for layer_idx in range(len(weights)):
            for i in range(len(weights[layer_idx])):
                for j in range(len(weights[layer_idx][i])):
                    weights[layer_idx][i][j] += avg_w[layer_idx][i][j]
        for layer_idx in range(len(biases)):
            for i in range(len(biases[layer_idx])):
                biases[layer_idx][i] += avg_b[layer_idx][i]

        self.aggregations += 1
        report = {
            "merged": True,
            "updates": len(self.received),
            "total_samples": total_samples,
            "rounds_merged": [u.get("round") for u in self.received],
        }
        self.received = []
        return report

    def status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "round": self.round,
            "published": self.published,
            "queued_updates": len(self.received),
            "aggregations": self.aggregations,
            "min_delta_samples": self.min_delta_samples,
            "transport": "gunx-mesh" if self.publish is not None else "local-buffer",
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "federated_sync",
            "node_id": self.node_id,
            "round": self.round,
            "published": self.published,
            "aggregations": self.aggregations,
            "last_weights": self.last_weights,
            "last_biases": self.last_biases,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FederatedSync:
        sync = cls(node_id=str(data.get("node_id", "node-restored")))
        sync.round = int(data.get("round", 0))
        sync.published = int(data.get("published", 0))
        sync.aggregations = int(data.get("aggregations", 0))
        sync.last_weights = data.get("last_weights")
        sync.last_biases = data.get("last_biases")
        return sync
