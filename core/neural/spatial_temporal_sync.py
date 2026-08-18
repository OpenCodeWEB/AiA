"""Spatial Temporal Sync - distance-filtered, distance-weighted mesh sync.

Closer peers on the 3D grid influence the local brain more than distant
ones: signed federated updates are filtered by spatial distance
(Manhattan over x/y/z) and merged with a 1/(1+d) weighting, layered on
top of the standard sample_count weighting of federated_sync.

The update format is the SAME signed flat payload emitted by
FederatedSync.make_update(), so peers, clusters and the global layer all
speak one wire protocol. Only spatial metadata + abstract weight deltas
are shared - never raw data or secrets.

Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import time
from typing import Any

from .federated_sync import _verify


class SpatialTemporalSync:
    """Spatially-aware federated update ingestion and aggregation."""

    def __init__(self, max_distance: int = 6, distance_decay: float = 0.5) -> None:
        self.max_distance = max_distance
        self.distance_decay = distance_decay
        self.coords: dict[str, int] = {"x": 0, "y": 0, "z": 0}
        self.entries: list[dict[str, Any]] = []
        self.pulses = 0
        self.rejected = 0
        self.aggregations = 0

    # ------------------------------------------------------------------ #
    # spatial weighting
    # ------------------------------------------------------------------ #
    def set_coords(self, coords: dict[str, int]) -> None:
        """Anchor the local node on the grid (x/y/z from the coordinator)."""
        self.coords = {k: int(coords.get(k, 0)) for k in ("x", "y", "z")}

    @staticmethod
    def distance(a: dict[str, int], b: dict[str, int]) -> int:
        return sum(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in ("x", "y", "z"))

    def weight_for(self, d: int) -> float:
        """Spatial trust weight: 1 at distance 0, decaying with distance."""
        return 1.0 / (1.0 + float(d)) ** self.distance_decay

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        peer_id: str,
        coords: dict[str, int],
        update: dict[str, Any],
        secret: str = "aia-federated-default",
    ) -> bool:
        """Filter + store one signed federated pulse. Returns accepted flag."""
        if self.distance(self.coords, coords) > self.max_distance:
            self.rejected += 1
            return False
        if not isinstance(update.get("delta_w"), list) or not isinstance(update.get("delta_b"), list):
            self.rejected += 1
            return False
        if not isinstance(update.get("signature"), str) or not _verify(
            {k: v for k, v in update.items() if k != "signature"}, update["signature"], secret
        ):
            self.rejected += 1
            return False
        self.entries.append(
            {
                "peer_id": peer_id,
                "coords": {k: int(coords.get(k, 0)) for k in ("x", "y", "z")},
                "distance": self.distance(self.coords, coords),
                "weight": self.weight_for(self.distance(self.coords, coords)),
                "ts": int(time.time()),
                "update": update,
            }
        )
        self.pulses += 1
        return True

    # ------------------------------------------------------------------ #
    # spatially weighted FedAvg
    # ------------------------------------------------------------------ #
    def aggregate(self, weights: list[list[list[float]]], biases: list[list[float]]) -> dict[str, Any]:
        """Merge queued pulses into the local brain, weighted by space + samples.

        Returns the merge report and CLEARS the pending pulse queue.
        """
        if not self.entries:
            return {"merged": False, "updates": 0, "note": "no spatial pulses queued"}
        total = sum(e["weight"] * int(e["update"].get("sample_count", 1)) for e in self.entries)
        scale = 1.0 / total if total else 0.0

        avg_w = [
            [
                [
                    sum(
                        e["weight"] * int(e["update"].get("sample_count", 1)) * e["update"]["delta_w"][layer_idx][i][j]
                        for e in self.entries
                    )
                    * scale
                    for j in range(len(self.entries[0]["update"]["delta_w"][layer_idx][i]))
                ]
                for i in range(len(self.entries[0]["update"]["delta_w"][layer_idx]))
            ]
            for layer_idx in range(len(self.entries[0]["update"]["delta_w"]))
        ]
        avg_b = [
            [
                sum(
                    e["weight"] * int(e["update"].get("sample_count", 1)) * e["update"]["delta_b"][layer_idx][i]
                    for e in self.entries
                )
                * scale
                for i in range(len(self.entries[0]["update"]["delta_b"][layer_idx]))
            ]
            for layer_idx in range(len(self.entries[0]["update"]["delta_b"]))
        ]

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
            "updates": len(self.entries),
            "total_samples": round(total, 4),
            "distances": [e["distance"] for e in self.entries],
            "weights": [round(e["weight"], 4) for e in self.entries],
        }
        self.entries = []
        return report

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "coords": self.coords,
            "max_distance": self.max_distance,
            "pulses": self.pulses,
            "rejected": self.rejected,
            "queued_pulses": len(self.entries),
            "aggregations": self.aggregations,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "spatial_temporal_sync",
            "coords": self.coords,
            "max_distance": self.max_distance,
            "distance_decay": self.distance_decay,
            "pulses": self.pulses,
            "rejected": self.rejected,
            "aggregations": self.aggregations,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SpatialTemporalSync:
        sync = cls(
            max_distance=int(data.get("max_distance", 6)),
            distance_decay=float(data.get("distance_decay", 0.5)),
        )
        sync.coords = {k: int(data.get("coords", {}).get(k, 0)) for k in ("x", "y", "z")}
        sync.pulses = int(data.get("pulses", 0))
        sync.rejected = int(data.get("rejected", 0))
        sync.aggregations = int(data.get("aggregations", 0))
        return sync
