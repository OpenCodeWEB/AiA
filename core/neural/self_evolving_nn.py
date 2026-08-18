"""Self-Evolving Neural Network - the continuously improving brain of AiA.

A dependency-free (pure standard library) multilayer perceptron that improves
itself continuously:

- ONLINE LEARNING  : trains sample-by-sample, never stops (Zero-Constraint)
- ADAPTIVE LR      : learning rate grows on progress, shrinks on plateaus
- EXPERIENCE REPLAY: remembers past samples and rehearses them in batches
- SELF-EVALUATION  : measures its own rolling loss/accuracy constantly
- ARCHITECTURE EVOLUTION: grows new neurons/layers when stuck on a plateau,
  prunes redundancy when it becomes over-competent

Everything serializes to JSON (flat primitives) so the network can sync
through the GunX mesh / GitHub without violating the flat-payload rule.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from typing import Any


def _tanh(x: float) -> float:
    return math.tanh(x)


def _d_tanh(y: float) -> float:
    return 1.0 - y * y


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _d_sigmoid(y: float) -> float:
    return y * (1.0 - y)


def _mse(pred: list[float], target: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(pred, target, strict=True)) / len(target)


def _dot(a: list[float], col: list[float]) -> float:
    return sum(x * wi for x, wi in zip(a, col, strict=True))


class SelfEvolvingNN:
    """A small self-improving neural network.

    Architecture starts small and evolves: when the rolling loss plateaus
    for `patience` evaluations, the network grows (neurons or a new layer);
    when it becomes over-competent it may prune a redundant layer.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden: list[int] | None = None,
        output_size: int = 1,
        lr: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden: list[int] = hidden or [6]
        self.lr = lr
        self.base_lr = lr
        self.generation = 0
        self.evolutions: list[str] = []

        # weight matrices: [layer][from][to]
        sizes = [self.input_size, *self.hidden, self.output_size]
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        rng = random.Random(seed)
        for a, b in zip(sizes, sizes[1:]):  # intentionally non-strict: pairs adjacent layer sizes  # noqa: B905
            scale = math.sqrt(2.0 / a)
            self.weights.append([[rng.uniform(-scale, scale) for _ in range(b)] for _ in range(a)])
            self.biases.append([0.0 for _ in range(b)])

        # learning bookkeeping
        self.samples_seen = 0
        self.ema_loss: float | None = None
        self.best_loss: float | None = None
        self.evals_without_improvement = 0
        self.patience = 120
        self.lr_floor = 1e-4
        self.lr_ceiling = 0.5

        # experience replay buffer (flat primitives only)
        self.replay_capacity = 2000
        self.replay: deque[tuple[list[float], list[float]]] = deque(maxlen=self.replay_capacity)

        # evaluation window
        self.eval_window: deque[tuple[list[float], list[float]]] = deque(maxlen=64)
        self.eval_loss: float | None = None
        self.eval_accuracy: float | None = None

    # ------------------------------------------------------------------ #
    # forward / backprop
    # ------------------------------------------------------------------ #
    def forward(self, x: list[float]) -> list[float]:
        """Predict outputs for input vector x."""
        acts = x[:]
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            acts = self._hidden_pass(acts, w, b)
        return self._output_layer(acts)

    def train(self, x: list[float], target: list[float]) -> float:
        """Online single-sample training. Returns the sample loss."""
        # --- forward, keeping every activation layer ---
        acts: list[list[float]] = [x[:]]
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            acts.append(self._hidden_pass(acts[-1], w, b))
        out = self._output_layer(acts[-1])
        acts.append(out)

        loss = _mse(out, target)

        # --- backward ---
        deltas: list[list[float]] = [None] * len(self.weights)  # type: ignore[list-item]
        deltas[-1] = [2.0 * (p - t) * _d_sigmoid(p) / len(target) for p, t in zip(out, target, strict=True)]
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            incoming = deltas[layer_idx + 1]
            w_next = self.weights[layer_idx + 1]
            d = []
            for j in range(len(acts[layer_idx + 1])):
                grad = sum(incoming[k] * w_next[j][k] for k in range(len(incoming)))
                d.append(grad * _d_tanh(acts[layer_idx + 1][j]))
            deltas[layer_idx] = d

        # --- weight update ---
        for layer_idx, w in enumerate(self.weights):
            a_in = acts[layer_idx]
            d_out = deltas[layer_idx]
            for i in range(len(a_in)):
                for j in range(len(d_out)):
                    w[i][j] -= self.lr * a_in[i] * d_out[j]
            for j in range(len(d_out)):
                self.biases[layer_idx][j] -= self.lr * d_out[j]

        # --- bookkeeping ---
        self.samples_seen += 1
        self.replay.append((x[:], target[:]))
        self.eval_window.append((x[:], target[:]))
        self._update_learning(loss)
        return loss

    @staticmethod
    def _hidden_pass(a: list[float], w: list[list[float]], b: list[float]) -> list[float]:
        return [_tanh(_dot(a, col) + bi) for col, bi in zip(zip(*w, strict=True), b, strict=True)]

    def _output_layer(self, a: list[float]) -> list[float]:
        w_out, b_out = self.weights[-1], self.biases[-1]
        return [_sigmoid(_dot(a, col) + bi) for col, bi in zip(zip(*w_out, strict=True), b_out, strict=True)]

    def _update_learning(self, loss: float) -> None:
        """Adaptive learning rate + plateau detection (the self-improvement loop)."""
        if self.ema_loss is None:
            self.ema_loss = loss
            self.best_loss = loss
            return
        self.ema_loss = 0.98 * self.ema_loss + 0.02 * loss
        if self.best_loss is None or self.ema_loss < self.best_loss * 0.999:
            self.best_loss = self.ema_loss
            self.evals_without_improvement = 0
            # progress -> gently accelerate
            self.lr = min(self.lr * 1.01, self.lr_ceiling)
        else:
            self.evals_without_improvement += 1
            # plateau -> slow down (stability)
            self.lr = max(self.lr * 0.995, self.lr_floor)
            if self.evals_without_improvement >= self.patience:
                self.evolve(reason="plateau")

    # ------------------------------------------------------------------ #
    # self-evaluation
    # ------------------------------------------------------------------ #
    def self_evaluate(self) -> dict[str, Any]:
        """Score the network on its recent experience window."""
        if not self.eval_window:
            self.eval_loss = None
            self.eval_accuracy = None
            return {"loss": None, "accuracy": None}
        losses: list[float] = []
        correct = 0
        for x, t in self.eval_window:
            pred = self.forward(x)
            losses.append(_mse(pred, t))
            if all((p >= 0.5) == (ti >= 0.5) for p, ti in zip(pred, t, strict=True)):
                correct += 1
        self.eval_loss = sum(losses) / len(losses)
        self.eval_accuracy = correct / len(self.eval_window)
        return {"loss": self.eval_loss, "accuracy": self.eval_accuracy}

    def replay_pass(self, batch_size: int = 16) -> float:
        """Rehearse a random batch from memory (experience replay)."""
        if len(self.replay) < 2:
            return 0.0
        batch = random.sample(tuple(self.replay), min(batch_size, len(self.replay)))
        total = 0.0
        for x, t in batch:
            total += self.train(x, t)
        return total / len(batch)

    # ------------------------------------------------------------------ #
    # architecture evolution
    # ------------------------------------------------------------------ #
    def evolve(self, reason: str = "plateau", grow_by: int = 2) -> dict[str, Any]:
        """Grow (or occasionally prune) the network. Returns evolution info."""
        self.generation += 1
        event: dict[str, Any] = {"generation": self.generation, "reason": reason, "action": "grow"}

        # grow: add neurons to a rotating hidden layer, or add a new layer
        layer_idx = (self.generation - 1) % len(self.hidden)
        old_size = self.hidden[layer_idx]
        new_size = min(old_size + grow_by, 128)
        if new_size == old_size:
            # cap reached -> add a fresh layer instead
            if len(self.hidden) < 6:
                self.hidden.append(grow_by)
                event["action"] = "add_layer"
                self._extend_weights()
            else:
                event["action"] = "reset_weights_small"
                self._reinitialize_weights()
        else:
            self.hidden[layer_idx] = new_size
            self._grow_layer_weights(layer_idx, grow_by)

        # occasional prune when over-competent (self-optimization)
        if self.eval_accuracy is not None and self.eval_accuracy >= 0.99 and len(self.hidden) > 1:
            if random.random() < 0.25:
                self.hidden.pop()
                self._prune_last_layer()
                event["action"] = "prune"

        self.evals_without_improvement = 0
        self.evolutions.append(f"gen{self.generation}:{event['action']}({event['reason']})")
        # rebirth: restore energy (learning rate) and relax the loss target so
        # the new capacity is allowed to prove itself before the next plateau
        self.lr = self.base_lr
        if self.ema_loss is not None:
            self.best_loss = self.ema_loss * 1.15
        return event

    def _grow_layer_weights(self, layer_idx: int, grow_by: int) -> None:
        """Insert `grow_by` new neurons into hidden layer `layer_idx`."""
        rng = random.Random()
        # incoming weights: layer_idx-th matrix gains grow_by columns
        w_in = self.weights[layer_idx]
        for row in w_in:
            row.extend(rng.uniform(-0.1, 0.1) for _ in range(grow_by))
        self.biases[layer_idx].extend(0.0 for _ in range(grow_by))
        # outgoing weights: next matrix gains grow_by rows
        w_out = self.weights[layer_idx + 1]
        a = len(w_out[0])
        w_out.extend([rng.uniform(-0.1, 0.1) for _ in range(a)] for _ in range(grow_by))

    def _extend_weights(self) -> None:
        """Add a brand-new hidden layer (with random weights) before the output layer."""
        rng = random.Random()
        prev_size = len(self.weights[-1])  # rows of the output matrix = last hidden size
        new_size = self.hidden[-1]
        out_size = len(self.weights[-1][0])  # output width
        scale = math.sqrt(2.0 / prev_size)
        w = [[rng.uniform(-scale, scale) for _ in range(new_size)] for _ in range(prev_size)]
        b = [0.0 for _ in range(new_size)]
        # splice the new layer between the last hidden layer and the output
        self.weights.insert(-1, w)
        self.biases.insert(-1, b)
        # the old output matrix must now connect the NEW layer -> output
        scale2 = math.sqrt(2.0 / new_size)
        self.weights[-1] = [[rng.uniform(-scale2, scale2) for _ in range(out_size)] for _ in range(new_size)]
        self.biases[-1] = [0.0 for _ in range(out_size)]

    def _prune_last_layer(self) -> None:
        """Drop the last hidden layer; re-seed the output matrix from the new last layer."""
        out_w = self.weights.pop()  # old output matrix
        self.biases.pop()
        out_size = len(out_w[0])
        prev_in = len(self.weights[-1])  # rows of last remaining matrix = new last hidden size
        rng = random.Random()
        scale = math.sqrt(2.0 / prev_in)
        self.weights[-1] = [[rng.uniform(-scale, scale) for _ in range(out_size)] for _ in range(prev_in)]
        self.biases[-1] = [0.0 for _ in range(out_size)]

    def _reinitialize_weights(self) -> None:
        """Re-seed weights with a tiny scale to escape a saturated state."""
        rng = random.Random()
        for w in self.weights:
            for row in w:
                for i in range(len(row)):
                    row[i] = rng.uniform(-0.15, 0.15)
        for b in self.biases:
            for i in range(len(b)):
                b[i] = 0.0

    # ------------------------------------------------------------------ #
    # persistence & status (flat primitives only)
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Public, JSON-safe snapshot (no private data)."""
        self.self_evaluate()
        return {
            "name": "SelfEvolvingNN",
            "input_size": self.input_size,
            "hidden": list(self.hidden),
            "output_size": self.output_size,
            "generation": self.generation,
            "samples_seen": self.samples_seen,
            "learning_rate": round(self.lr, 6),
            "ema_loss": None if self.ema_loss is None else round(self.ema_loss, 6),
            "eval_loss": None if self.eval_loss is None else round(self.eval_loss, 6),
            "eval_accuracy": None if self.eval_accuracy is None else round(self.eval_accuracy, 6),
            "replay_size": len(self.replay),
            "replay_capacity": self.replay_capacity,
            "evolutions": list(self.evolutions[-12:]),
            "total_evolutions": len(self.evolutions),
            "evolving": self.evals_without_improvement >= self.patience,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "input_size": self.input_size,
                "hidden": self.hidden,
                "output_size": self.output_size,
                "lr": self.lr,
                "generation": self.generation,
                "weights": self.weights,
                "biases": self.biases,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> SelfEvolvingNN:
        data = json.loads(payload)
        net = cls(input_size=data["input_size"], hidden=data["hidden"], output_size=data["output_size"])
        net.lr = data.get("lr", net.lr)
        net.generation = data.get("generation", 0)
        net.weights = data["weights"]
        net.biases = data["biases"]
        return net
