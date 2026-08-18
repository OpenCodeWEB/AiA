"""Tests for ClusterHeartbeatElection (regional anchor election)."""

from __future__ import annotations

from core.neural.cluster_heartbeat_election import ClusterHeartbeatElection


def _cluster(**kwargs) -> ClusterHeartbeatElection:
    kwargs.setdefault("region", "asia-south")
    return ClusterHeartbeatElection(**kwargs)


class TestHeartbeat:
    def test_heartbeat_registers_peer(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        result = c.heartbeat("peer-a", {"x": 0, "y": 0, "z": 0}, stage=0, ts=100)
        assert result["ok"] is True
        assert "peer-a" in c.heartbeats
        assert result["live_peers"] == 1

    def test_first_live_peer_becomes_anchor(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        result = c.heartbeat("peer-a", {"x": 0, "y": 0, "z": 0}, stage=0, ts=100)
        assert result["anchor"] == "peer-a"
        assert result["anchor_changed"] is True
        assert c.elections == 1

    def test_higher_stage_wins_election(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        c.heartbeat("edge-1", {"x": 0, "y": 0, "z": 0}, stage=0, ts=100)
        result = c.heartbeat("evolved-1", {"x": 1, "y": 0, "z": 2}, stage=2, ts=101)
        assert result["anchor"] == "evolved-1"
        assert result["anchor_changed"] is True

    def test_stable_peer_keeps_anchor(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        c.heartbeat("leader", {"x": 0, "y": 0, "z": 1}, stage=1, ts=100)
        result = c.heartbeat("follower", {"x": 0, "y": 0, "z": 0}, stage=0, ts=102)
        assert result["anchor"] == "leader"
        assert result["anchor_changed"] is False

    def test_heartbeat_count_breaks_tie(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        c.heartbeat("a", {"x": 0, "y": 0, "z": 1}, stage=1, ts=100)
        c.heartbeat("a", {"x": 0, "y": 0, "z": 1}, stage=1, ts=101)
        result = c.heartbeat("b", {"x": 1, "y": 0, "z": 1}, stage=1, ts=102)
        assert result["anchor"] == "a"  # more heartbeats -> more stable

    def test_stage_clamped(self) -> None:
        c = _cluster(heartbeat_ttl=1000)
        c.heartbeat("peer", {"x": 0, "y": 0, "z": 0}, stage=99, ts=100)
        assert c.heartbeats["peer"]["stage"] == 2


class TestPruning:
    def test_stale_peer_pruned_and_anchor_lost(self) -> None:
        c = _cluster(heartbeat_ttl=10)
        c.heartbeat("solo", {"x": 0, "y": 0, "z": 0}, stage=0, ts=100)
        assert c.anchor == "solo"
        result = c.heartbeat("new", {"x": 0, "y": 0, "z": 0}, stage=0, ts=200)
        assert "solo" not in c.heartbeats
        assert result["anchor"] == "new"

    def test_anchor_demoted_when_stale(self) -> None:
        c = _cluster(heartbeat_ttl=10)
        c.heartbeat("leader", {"x": 0, "y": 0, "z": 1}, stage=1, ts=100)
        result = c.heartbeat("fresh", {"x": 0, "y": 0, "z": 0}, stage=0, ts=200)
        assert "leader" not in c.heartbeats
        assert result["anchor"] == "fresh"
        assert result["anchor_changed"] is True


class TestSerialization:
    def test_status_shape(self) -> None:
        c = _cluster()
        c.heartbeat("peer", {"x": 0, "y": 0, "z": 0}, stage=1, ts=100)
        status = c.status()
        assert status["region"] == "asia-south"
        assert status["anchor"] == "peer"
        assert status["live_peers"] == 1

    def test_to_json_round_trip(self) -> None:
        c = _cluster()
        c.heartbeat("peer", {"x": 1, "y": 2, "z": 0}, stage=1, ts=100)
        restored = ClusterHeartbeatElection.from_json(c.to_json())
        assert restored.region == c.region
        assert restored.anchor == c.anchor
        assert restored.elections == c.elections
        assert restored.heartbeats == c.heartbeats

    def test_from_json_missing_fields_safe(self) -> None:
        restored = ClusterHeartbeatElection.from_json({})
        assert restored.region == "asia-south"
        assert restored.anchor is None
