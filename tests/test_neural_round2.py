"""Round-2 neural module tests: federated sync, DQN agent, attention layer."""

import json

from core.neural import (
    AttentionLayer,
    DQNAgent,
    FederatedSync,
    SelfEvolvingNN,
)
from core.neural.neural_api import NeuralHandler


# ---------------------------------------------------------------------- #
# FederatedSync
# ---------------------------------------------------------------------- #
class TestFederatedSync:
    def test_delta_and_signature_roundtrip(self):
        sync = FederatedSync(secret="test-secret")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.5, 0.5], [0.8])
        update = sync.make_update(nn.weights, nn.biases, sample_count=10)
        assert update is not None
        assert update["node_id"] == sync.node_id
        assert update["round"] == 1
        assert update["sample_count"] == 10
        assert len(update["signature"]) == 64

    def test_tampered_update_rejected(self):
        sync = FederatedSync(secret="test-secret")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.5, 0.5], [0.8])
        update = sync.make_update(nn.weights, nn.biases, sample_count=10)
        assert update is not None
        update["delta_w"][0][0][0] += 0.5  # tamper
        assert sync.receive(update) is False

    def test_wrong_secret_rejected(self):
        sender = FederatedSync(secret="sender-secret")
        receiver = FederatedSync(secret="receiver-secret")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        sender.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.5, 0.5], [0.8])
        update = sender.make_update(nn.weights, nn.biases, sample_count=10)
        assert update is not None
        assert receiver.receive(update) is False

    def test_own_update_ignored(self):
        sync = FederatedSync(secret="test-secret")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.5, 0.5], [0.8])
        update = sync.make_update(nn.weights, nn.biases, sample_count=10)
        assert update is not None
        assert sync.receive(update) is False  # same node_id -> ignore

    def test_fedavg_merges_updates(self):
        # two peers, both sign with the shared secret
        peer_a = FederatedSync(secret="shared", node_id="peer-a")
        peer_b = FederatedSync(secret="shared", node_id="peer-b")
        host = FederatedSync(secret="shared", node_id="host")

        nn_a = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1, seed=1)
        nn_b = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1, seed=2)
        host_nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1, seed=3)
        before = [list(map(list, w)) for w in host_nn.weights]

        peer_a.make_update(nn_a.weights, nn_a.biases, 1)  # baselines BEFORE training
        peer_b.make_update(nn_b.weights, nn_b.biases, 1)
        host.make_update(host_nn.weights, host_nn.biases, 1)

        for _ in range(20):
            nn_a.train([0.4, 0.6], [0.9])
            nn_b.train([0.6, 0.4], [0.2])

        ua = peer_a.make_update(nn_a.weights, nn_a.biases, sample_count=20)
        ub = peer_b.make_update(nn_b.weights, nn_b.biases, sample_count=20)
        assert ua is not None and ub is not None

        assert host.receive(ua) is True
        assert host.receive(ub) is True
        report = host.aggregate(host_nn.weights, host_nn.biases)
        assert report["merged"] is True
        assert report["updates"] == 2
        assert len(host.received) == 0
        # brain moved toward the peers
        moved = sum(
            abs(host_nn.weights[li][i][j] - before[li][i][j])
            for li in range(len(before))
            for i in range(len(before[li]))
            for j in range(len(before[li][i]))
        )
        assert moved > 0.0

    def test_aggregate_without_updates(self):
        sync = FederatedSync(secret="test-secret")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        report = sync.aggregate(nn.weights, nn.biases)
        assert report["merged"] is False

    def test_serialization_roundtrip(self):
        sync = FederatedSync(secret="s", node_id="n1")
        nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
        sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.3, 0.7], [0.5])
        sync.make_update(nn.weights, nn.biases, sample_count=5)
        restored = FederatedSync.from_json(sync.to_json())
        assert restored.node_id == "n1"
        assert restored.round == 1
        assert restored.last_weights is not None


# ---------------------------------------------------------------------- #
# DQNAgent
# ---------------------------------------------------------------------- #
class TestDQNAgent:
    def test_act_returns_valid_action(self):
        agent = DQNAgent(state_size=2, action_size=3, seed=7)
        result = agent.act([0.1, 0.9])
        assert 0 <= result["action"] < 3
        assert len(result["q_values"]) == 3
        assert 0.0 < result["epsilon"] <= 1.0

    def test_epsilon_decays(self):
        agent = DQNAgent(state_size=2, action_size=2, epsilon=1.0, epsilon_decay=0.9, seed=1)
        agent.act([0.5, 0.5])
        agent.act([0.5, 0.5])
        assert agent.epsilon < 0.9

    def test_step_learns_simple_task(self):
        # goal: state [1,0] -> action 0 good; [0,1] -> action 1 good
        agent = DQNAgent(state_size=2, action_size=2, epsilon=0.9, epsilon_min=0.05, lr=0.1, batch_size=4, seed=42)
        for _ in range(300):
            for s, good in (([1.0, 0.0], 0), ([0.0, 1.0], 1)):
                a = agent.act(s)["action"]
                r = 1.0 if a == good else -0.5
                ns = s
                agent.step(s, a, r, ns, done=True)
        # greedy behavior after training
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0
        q0 = agent.act([1.0, 0.0])
        q1 = agent.act([0.0, 1.0])
        assert q0["action"] == 0
        assert q1["action"] == 1

    def test_serialization_roundtrip(self):
        agent = DQNAgent(state_size=2, action_size=3, seed=5)
        agent.step([0.2, 0.8], 1, 0.5, [0.8, 0.2], done=False)
        restored = DQNAgent.from_json(agent.to_json())
        assert restored.state_size == 2
        assert restored.action_size == 3
        assert restored.steps == agent.steps
        agent.epsilon = 0.0
        restored.epsilon = 0.0
        agent.epsilon_min = 0.0
        restored.epsilon_min = 0.0
        r1 = agent.act([0.5, 0.5])
        r2 = restored.act([0.5, 0.5])
        assert r1["action"] == r2["action"]  # same weights -> same greedy result


# ---------------------------------------------------------------------- #
# AttentionLayer
# ---------------------------------------------------------------------- #
class TestAttentionLayer:
    def test_attention_map_rows_sum_to_one(self):
        layer = AttentionLayer(d_model=4, seed=3)
        result = layer.forward([[0.5, 0.2, 0.8, 0.1], [0.9, 0.3, 0.1, 0.7]])
        attn = result["attention_map"]
        assert len(attn) == 2
        for row in attn:
            assert abs(sum(row) - 1.0) < 1e-6
        assert len(result["context"]) == 2

    def test_attention_focuses_on_similar_tokens(self):
        layer = AttentionLayer(d_model=4, seed=1)
        # orthogonal tokens -> each token attends mostly to itself (diag > off-diag)
        result = layer.forward([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        attn = result["attention_map"]
        assert attn[0][0] > attn[0][1]
        assert attn[1][1] > attn[1][0]
        # identical tokens -> attention splits symmetrically (~0.5 each)
        same = layer.forward([[0.7, 0.7, 0.7, 0.7], [0.7, 0.7, 0.7, 0.7]])
        assert abs(same["attention_map"][0][0] - 0.5) < 0.05

    def test_train_reduces_reconstruction_loss(self):
        layer = AttentionLayer(d_model=4, lr=0.01, seed=9)
        x = [[0.5, 0.1, 0.9, 0.3], [0.2, 0.8, 0.4, 0.6], [0.7, 0.5, 0.2, 0.9]]
        before = layer.forward(x)["context"]
        result = layer.train(x, steps=30)
        after = result["context"]
        err_before = sum(abs(before[i][j] - x[i][j]) for i in range(3) for j in range(4))
        err_after = sum(abs(after[i][j] - x[i][j]) for i in range(3) for j in range(4))
        assert err_after < err_before
        assert result["train_calls"] == 1

    def test_serialization_roundtrip(self):
        layer = AttentionLayer(d_model=4, seed=2)
        layer.train([[0.4, 0.6, 0.2, 0.8]], steps=5)
        restored = AttentionLayer.from_json(layer.to_json())
        assert restored.d_model == 4
        assert restored.train_calls == 1
        a = layer.forward([[0.5, 0.5, 0.5, 0.5]])["attention_map"]
        b = restored.forward([[0.5, 0.5, 0.5, 0.5]])["attention_map"]
        assert a == b


# ---------------------------------------------------------------------- #
# API wiring
# ---------------------------------------------------------------------- #
class TestRound2API:
    def test_modules_include_round2(self):
        with NeuralHandler.lock:
            assert hasattr(NeuralHandler, "federated")
            assert hasattr(NeuralHandler, "dqn")
            assert hasattr(NeuralHandler, "attention")

    def test_federated_flow_end_to_end(self):
        with NeuralHandler.lock:
            NeuralHandler.nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1)
            NeuralHandler.federated = FederatedSync(secret="api-secret", node_id="api-host")
            # simulate a peer
            peer = FederatedSync(secret="api-secret", node_id="peer-x")
            peer_nn = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1, seed=11)
            peer.make_update(peer_nn.weights, peer_nn.biases, 1)  # baseline frame
            for _ in range(15):
                peer_nn.train([0.5, 0.5], [0.9])
            update = peer.make_update(peer_nn.weights, peer_nn.biases, sample_count=15)
            assert update is not None
            assert NeuralHandler.federated.receive(update) is True
            before = [list(map(list, w)) for w in NeuralHandler.nn.weights]
            report = NeuralHandler.federated.aggregate(NeuralHandler.nn.weights, NeuralHandler.nn.biases)
            assert report["merged"] is True
            moved = sum(
                abs(NeuralHandler.nn.weights[li][i][j] - before[li][i][j])
                for li in range(len(before))
                for i in range(len(before[li]))
                for j in range(len(before[li][i]))
            )
            assert moved > 0.0

    def test_dqn_and_attention_objects_live(self):
        with NeuralHandler.lock:
            assert NeuralHandler.dqn.state_size == 2
            assert NeuralHandler.dqn.action_size == 3
            assert NeuralHandler.attention.d_model == 4

    def test_json_schema_flat(self):
        """Round-2 payloads must stay flat-primitives (GunX rule)."""
        sync = FederatedSync(secret="s")
        nn = SelfEvolvingNN(input_size=2, hidden=[4], output_size=1)
        sync.make_update(nn.weights, nn.biases, 1)  # baseline frame
        nn.train([0.1, 0.9], [0.5])
        update = sync.make_update(nn.weights, nn.biases, sample_count=3)
        assert update is not None
        raw = json.dumps(update)
        assert '"delta_w"' in raw and '"signature"' in raw
