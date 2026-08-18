"""Tests for the Spatial Grid Coordinator (decentralized 3D neural mesh).

Covers coordinate derivation (X/Y/Z), HMAC-signed registration,
GunX flat-schema compatibility and JSON round-trips.
"""

from __future__ import annotations

import json

from core.neural.neural_api import NeuralHandler
from core.neural.spatial_grid_coordinator import (
    X_LAYERS,
    Y_REGIONS,
    Z_STAGES,
    SpatialGridCoordinator,
)


def _coordinator(**kwargs) -> SpatialGridCoordinator:
    kwargs.setdefault("node_id", "test-node")
    kwargs.setdefault("secret", "test-secret")
    return SpatialGridCoordinator(**kwargs)


# ---------------------------------------------------------------------- #
# coordinate derivation
# ---------------------------------------------------------------------- #
class TestCoordinates:
    def test_default_coords_in_bounds(self) -> None:
        c = _coordinator()
        coords = c.coords()
        assert set(coords) == {"x", "y", "z"}
        assert 0 <= coords["x"] <= 3
        assert 0 <= coords["y"] < len(Y_REGIONS)
        assert 0 <= coords["z"] <= 2

    def test_x_from_perception_dominance(self) -> None:
        c = _coordinator()
        c.update_competence("autoencoder", 0.9)
        c.update_competence("anomaly", 0.85)
        c.update_competence("rnn", 0.2)
        c.update_competence("attention", 0.1)
        c.update_competence("dqn", 0.0)
        c.update_competence("hopfield", 0.0)
        c.update_competence("self_evolving", 0.0)
        c.update_competence("curriculum", 0.0)
        assert c.coords()["x"] == 0  # perception layer

    def test_x_from_sequence_dominance(self) -> None:
        c = _coordinator()
        c.update_competence("rnn", 0.8)
        c.update_competence("attention", 0.75)
        c.update_competence("autoencoder", 0.2)
        c.update_competence("dqn", 0.1)
        assert c.coords()["x"] == 1

    def test_x_from_decision_dominance(self) -> None:
        c = _coordinator()
        c.update_competence("dqn", 0.7)
        c.update_competence("hopfield", 0.6)
        c.update_competence("rnn", 0.1)
        assert c.coords()["x"] == 2

    def test_x_from_executive_dominance(self) -> None:
        c = _coordinator()
        c.update_competence("self_evolving", 0.9)
        c.update_competence("curriculum", 0.8)
        c.update_competence("dqn", 0.1)
        assert c.coords()["x"] == 3

    def test_x_defaults_to_perception_when_unknown(self) -> None:
        assert _coordinator().coords()["x"] == 0

    def test_y_from_explicit_region(self) -> None:
        c = _coordinator(region="eu-central")
        assert c.coords()["y"] == Y_REGIONS.index("eu-central")

    def test_y_stable_hash_when_no_region(self) -> None:
        a = _coordinator()
        b = _coordinator()
        assert a.coords()["y"] == b.coords()["y"]
        assert Y_REGIONS[a.coords()["y"]] in Y_REGIONS

    def test_z_advances_with_stage(self) -> None:
        c = _coordinator()
        assert c.coords()["z"] == 0
        c.stage = 1
        assert c.coords()["z"] == 1
        c.stage = 2
        assert c.coords()["z"] == 2

    def test_z_clamped(self) -> None:
        c = _coordinator()
        c.stage = 99
        assert c.coords()["z"] == 2


# ---------------------------------------------------------------------- #
# signed registration
# ---------------------------------------------------------------------- #
class TestRegistration:
    def test_register_returns_signed_metadata(self) -> None:
        c = _coordinator()
        meta = c.register("peer-node")
        assert meta["node_id"] == "peer-node"
        assert meta["coords"]["x"] in X_LAYERS
        assert "spatial_layer" in meta
        assert "region" in meta
        assert "hmac_signature" in meta
        assert "ts" in meta

    def test_registered_node_repeatable(self) -> None:
        c = _coordinator()
        first = c.register("peer-node")
        second = c.register("peer-node")
        assert first["coords"] == second["coords"]

    def test_signature_verifies(self) -> None:
        c = _coordinator()
        meta = c.register("peer-node")
        assert c.verify(meta)

    def test_signature_rejects_tampered(self) -> None:
        c = _coordinator()
        meta = c.register("peer-node")
        meta["coords"]["x"] = (meta["coords"]["x"] + 1) % 4
        assert not c.verify(meta)

    def test_signature_rejects_wrong_secret(self) -> None:
        c = _coordinator(secret="test-secret")
        meta = c.register("peer-node")
        other = _coordinator(secret="other-secret")
        assert not other.verify(meta)

    def test_nodes_listed(self) -> None:
        c = _coordinator()
        c.register("peer-a")
        c.register("peer-b")
        assert "peer-a" in c.nodes
        assert "peer-b" in c.nodes
        assert c.status()["registered"] == 2


# ---------------------------------------------------------------------- #
# GunX flat schema + JSON round-trip
# ---------------------------------------------------------------------- #
class TestSchema:
    def test_gunx_flat_primitives(self) -> None:
        c = _coordinator()
        meta = c.register("peer-node")
        dumped = json.dumps(meta)
        parsed = json.loads(dumped)
        assert parsed == meta  # flat primitives survive JSON

    def test_gunx_path_layout(self) -> None:
        c = _coordinator()
        meta = c.register("peer-node")
        assert meta["kind"] == "spatial_node"
        assert "weight_delta" in meta  # reserved key per GunX schema
        assert "last_pulse" in meta

    def test_to_json_round_trip(self) -> None:
        c = _coordinator()
        c.update_competence("dqn", 0.7)
        c.register("peer-node")
        restored = SpatialGridCoordinator.from_json(c.to_json())
        assert restored.coords() == c.coords()
        assert restored.node_id == c.node_id
        assert restored.nodes == c.nodes

    def test_from_json_restores_secret(self) -> None:
        c = _coordinator(secret="roundtrip-secret")
        restored = SpatialGridCoordinator.from_json(c.to_json())
        meta = restored.register("peer-node")
        assert restored.verify(meta)


# ---------------------------------------------------------------------- #
# metadata helpers
# ---------------------------------------------------------------------- #
class TestMetadata:
    def test_spatial_metadata_block(self) -> None:
        c = _coordinator(region="us-east")
        block = c.spatial_metadata()
        assert block["node_id"] == "test-node"
        assert block["coords"] == c.coords()
        assert block["spatial_layer"] == Z_STAGES[c.coords()["z"]]
        assert block["region"] == "us-east"

    def test_unknown_competence_ignored(self) -> None:
        c = _coordinator()
        c.update_competence("not-a-module", 1.0)  # should not raise
        assert c.coords()["x"] == 0

    def test_layers_mappings_consistent(self) -> None:
        assert len(X_LAYERS) == 4
        assert len(Y_REGIONS) >= 3
        assert set(Z_STAGES) == {0, 1, 2}
        assert Z_STAGES[0] == "edge"
        assert Z_STAGES[2] == "global"


# ---------------------------------------------------------------------- #
# neural API wiring
# ---------------------------------------------------------------------- #
class TestApiWiring:
    def test_handler_has_spatial_module(self) -> None:
        with NeuralHandler.lock:
            assert hasattr(NeuralHandler, "spatial")
            assert isinstance(NeuralHandler.spatial, SpatialGridCoordinator)

    def test_modules_status_includes_spatial(self) -> None:
        with NeuralHandler.lock:
            status = NeuralHandler.spatial.status()
        assert status["node_id"] == NeuralHandler.spatial.node_id
        assert set(status["coords"]) == {"x", "y", "z"}

    def test_coords_endpoint_payload_shape(self) -> None:
        with NeuralHandler.lock:
            block = NeuralHandler.spatial.register(NeuralHandler.spatial.node_id)
        assert block["kind"] == "spatial_node"
        assert NeuralHandler.spatial.verify(block)
