"""Tests for the Self-Evolving Neural Network."""

from __future__ import annotations

import json
import random

from core.neural.neural_api import NeuralHandler, SelfEvolvingNN


def _xor(net: SelfEvolvingNN, rounds: int = 1) -> None:
    data = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
    # deterministic shuffle: test outcome must not depend on the global
    # random state left behind by earlier test files
    rng = random.Random(42)
    for _ in range(rounds):
        rng.shuffle(data)
        for x, t in data:
            net.train(x, t)


def _xor_accuracy(net: SelfEvolvingNN) -> float:
    correct = 0
    for x, t in [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]:
        if (net.forward(x)[0] >= 0.5) == (t[0] >= 0.5):
            correct += 1
    return correct / 4


def test_xor_learns_online() -> None:
    """The brain must master XOR through pure online single-sample training."""
    net = SelfEvolvingNN(input_size=2, hidden=[6], output_size=1, lr=0.12, seed=42)
    for _ in range(600):
        _xor(net, rounds=2)
        if _xor_accuracy(net) == 1.0:
            break
    assert _xor_accuracy(net) == 1.0, f"XOR not mastered: {_xor_accuracy(net)}"
    assert net.eval_accuracy is None or net.eval_accuracy > 0.8


def test_evolution_grows_when_stuck() -> None:
    """A too-small brain must evolve its architecture instead of staying stuck."""
    net = SelfEvolvingNN(input_size=2, hidden=[2], output_size=1, lr=0.05, seed=7)
    net.patience = 30
    before = list(net.hidden)
    for _ in range(400):
        _xor(net, rounds=2)
    assert net.generation >= 1, "evolution never triggered"
    assert net.hidden != before or len(net.hidden) > 1, "architecture did not grow"
    assert net.evolutions, "no evolution log entries"


def test_adaptive_learning_rate_changes() -> None:
    net = SelfEvolvingNN(input_size=2, hidden=[4], output_size=1, lr=0.05, seed=1)
    lrs = set()
    for _ in range(120):
        _xor(net, rounds=2)
        lrs.add(round(net.lr, 5))
    assert len(lrs) > 1, "learning rate never adapted"


def test_experience_replay() -> None:
    net = SelfEvolvingNN(input_size=2, hidden=[4], output_size=1, lr=0.05, seed=3)
    _xor(net, rounds=50)
    assert len(net.replay) >= 8, "replay buffer not filling"
    before = net.forward([1.0, 0.0])[0]
    net.replay_pass(batch_size=8)
    assert net.forward([1.0, 0.0])[0] != before or net.samples_seen > 400


def test_save_load_roundtrip() -> None:
    net = SelfEvolvingNN(input_size=2, hidden=[5, 3], output_size=1, lr=0.09, seed=11)
    _xor(net, rounds=30)
    payload = net.to_json()
    clone = SelfEvolvingNN.from_json(payload)
    assert clone.hidden == net.hidden
    assert clone.weights == net.weights
    assert clone.biases == net.biases
    assert clone.generation == net.generation
    for x, _t in [([0.0, 1.0], [1.0]), ([1.0, 1.0], [0.0])]:
        assert clone.forward(x) == net.forward(x)


def test_status_is_flat_primitives() -> None:
    net = SelfEvolvingNN(input_size=2, hidden=[4], output_size=1, seed=5)
    _xor(net, rounds=10)
    status = net.status()
    assert isinstance(status, dict)
    assert json.dumps(status, ensure_ascii=False)  # serializable
    assert "hidden" in status and "generation" in status and "evolutions" in status


def test_api_handler_train_and_predict() -> None:
    handler = NeuralHandler
    old = handler.nn
    try:
        handler.nn = SelfEvolvingNN(input_size=2, hidden=[4], output_size=1, lr=0.1, seed=2)
        handler.nn.train([0.0, 1.0], [1.0])
        status = handler.nn.status()
        assert status["samples_seen"] == 1
        pred = handler.nn.forward([0.0, 1.0])
        assert 0.0 <= pred[0] <= 1.0
    finally:
        handler.nn = old
