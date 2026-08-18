"""Tests for SpatialTemporalSync (distance-filtered, distance-weighted sync)."""

from __future__ import annotations

import pytest

from core.neural.federated_sync import FederatedSync
from core.neural.self_evolving_nn import SelfEvolvingNN
from core.neural.spatial_temporal_sync import SpatialTemporalSync


def _sync(**kwargs) -> SpatialTemporalSync:
    kwargs.setdefault("max_distance", 6)
    return SpatialTemporalSync(**kwargs)


def _signed_update(secret: str = "test-secret", sample_count: int = 4) -> dict:
    sync = FederatedSync(secret=secret, node_id="sender")
    nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
    sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
    nn.train([0.5, 0.5], [0.8])
    update = sync.make_update(nn.weights, nn.biases, sample_count)
    assert update is not None
    return update


class TestDistance:
    def test_manhattan_distance(self) -> None:
        s = _sync()
        assert s.distance({"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 2, "z": 3}) == 6
        assert s.distance({"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}) == 0

    def test_weight_decays_with_distance(self) -> None:
        s = _sync()
        assert s.weight_for(0) == 1.0
        assert s.weight_for(1) > s.weight_for(3) > s.weight_for(6)
        assert 0.0 < s.weight_for(6) < 1.0

    def test_set_coords_normalizes(self) -> None:
        s = _sync()
        s.set_coords({"x": 3, "y": 2, "z": 1})
        assert s.coords == {"x": 3, "y": 2, "z": 1}


class TestIngest:
    def test_nearby_pulse_accepted(self) -> None:
        s = _sync(max_distance=6)
        s.set_coords({"x": 0, "y": 0, "z": 0})
        ok = s.ingest("peer", {"x": 1, "y": 1, "z": 1}, _signed_update(), secret="test-secret")
        assert ok is True
        assert s.pulses == 1
        assert s.entries[0]["distance"] == 3
        assert s.entries[0]["weight"] == pytest.approx(s.weight_for(3))

    def test_far_pulse_rejected(self) -> None:
        s = _sync(max_distance=2)
        s.set_coords({"x": 0, "y": 0, "z": 0})
        ok = s.ingest("peer", {"x": 3, "y": 0, "z": 0}, _signed_update(), secret="test-secret")
        assert ok is False
        assert s.rejected == 1
        assert s.entries == []

    def test_bad_signature_rejected(self) -> None:
        s = _sync()
        s.set_coords({"x": 0, "y": 0, "z": 0})
        update = _signed_update(secret="sender-secret")
        ok = s.ingest("peer", {"x": 0, "y": 0, "z": 0}, update, secret="receiver-secret")
        assert ok is False
        assert s.rejected == 1

    def test_tampered_update_rejected(self) -> None:
        s = _sync()
        s.set_coords({"x": 0, "y": 0, "z": 0})
        update = _signed_update()
        update["delta_w"][0][0][0] += 1.0
        assert s.ingest("peer", {"x": 0, "y": 0, "z": 0}, update) is False

    def test_missing_delta_rejected(self) -> None:
        s = _sync()
        s.set_coords({"x": 0, "y": 0, "z": 0})
        assert s.ingest("peer", {"x": 0, "y": 0, "z": 0}, {"signature": "x"}) is False


class TestAggregate:
    def test_empty_aggregate_noop(self) -> None:
        s = _sync()
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        report = s.aggregate(nn.weights, nn.biases)
        assert report["merged"] is False
        assert s.aggregations == 0

    def test_aggregate_applies_weights(self) -> None:
        s = _sync()
        s.set_coords({"x": 0, "y": 0, "z": 0})
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        before = [list(map(list, w)) for w in nn.weights]
        ok = s.ingest("peer", {"x": 0, "y": 0, "z": 0}, _signed_update(sample_count=10), secret="test-secret")
        assert ok is True
        report = s.aggregate(nn.weights, nn.biases)
        assert report["merged"] is True
        assert report["updates"] == 1
        moved = any(
            abs(nn.weights[li][i][j] - before[li][i][j]) > 1e-9
            for li in range(len(nn.weights))
            for i in range(len(nn.weights[li]))
            for j in range(len(nn.weights[li][i]))
        )
        assert moved is True
        assert s.entries == []  # queue cleared after merge

    def test_closer_peer_dominates_merge(self) -> None:
        s = _sync(max_distance=6)
        s.set_coords({"x": 0, "y": 0, "z": 0})
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        # near peer: large sample_count, far peer: small sample_count
        near = _signed_update(sample_count=100)
        far = _signed_update(sample_count=1)
        assert s.ingest("near", {"x": 0, "y": 0, "z": 0}, near, secret="test-secret") is True
        assert s.ingest("far", {"x": 3, "y": 3, "z": 0}, far, secret="test-secret") is True
        report = s.aggregate(nn.weights, nn.biases)
        assert report["merged"] is True
        assert len(report["distances"]) == 2
        assert report["weights"][0] > report["weights"][1]  # near > far


class TestSerialization:
    def test_status_shape(self) -> None:
        s = _sync()
        s.set_coords({"x": 1, "y": 1, "z": 0})
        status = s.status()
        assert status["coords"] == {"x": 1, "y": 1, "z": 0}
        assert status["queued_pulses"] == 0

    def test_to_json_round_trip(self) -> None:
        s = _sync(max_distance=3)
        s.set_coords({"x": 2, "y": 1, "z": 0})
        restored = SpatialTemporalSync.from_json(s.to_json())
        assert restored.max_distance == 3
        assert restored.coords == s.coords
        assert restored.pulses == s.pulses

    def test_from_json_missing_fields_safe(self) -> None:
        restored = SpatialTemporalSync.from_json({})
        assert restored.max_distance == 6
        assert restored.coords == {"x": 0, "y": 0, "z": 0}
