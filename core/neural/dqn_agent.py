"""Deep Q-Network agent - autonomous decision making with experience replay.

The agent learns Q(s, a) - the expected future reward of taking action `a`
in state `s` - from (state, action, reward, next_state, done) transitions
pushed through the environment loop:

    API: POST /rl/step   {"state": [...], "action": n, "reward": f,
                           "next_state": [...], "done": bool}
         -> {"loss": ..., "epsilon": ..., "q_estimate": ...}
         POST /rl/act    {"state": [...]}
         -> {"action": n, "q_values": [...], "epsilon": ...}

Pure stdlib MLP (tanh hidden layer, linear output head) with:

  * experience replay buffer (ring, maxlen)
  * epsilon-greedy exploration with decay
  * soft-updated target network (tau) for stable TD targets
  * TD-error backprop on minibatches

Flat-primitives JSON only (GunX-compatible).
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class _Network:
    """Tiny 2-layer MLP: state_size -> hidden (tanh) -> action_size (linear)."""

    def __init__(self, state_size: int, action_size: int, hidden: int = 16, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden = hidden
        # weights[l][to][from] (from = fan-in)
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(state_size)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-0.5, 0.5) for _ in range(hidden)] for _ in range(action_size)]
        self.b2 = [0.0] * action_size

    def forward(self, state: list[float]) -> tuple[list[float], list[float]]:
        h = [math.tanh(_dot(self.w1[i], state) + self.b1[i]) for i in range(self.hidden)]
        q = [_dot(self.w2[a], h) + self.b2[a] for a in range(self.action_size)]
        return q, h

    def copy_from(self, other: _Network) -> None:
        self.w1 = [list(row) for row in other.w1]
        self.b1 = list(other.b1)
        self.w2 = [list(row) for row in other.w2]
        self.b2 = list(other.b2)

    def to_json(self) -> dict[str, Any]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2, "hidden": self.hidden}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _Network:
        net = cls(len(data["w1"][0]), len(data["w2"]), int(data.get("hidden", 16)))
        net.w1 = data["w1"]
        net.b1 = data["b1"]
        net.w2 = data["w2"]
        net.b2 = data["b2"]
        return net


class DQNAgent:
    """Deep Q-Network with replay buffer and soft target updates."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden: int = 16,
        gamma: float = 0.9,
        lr: float = 0.05,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        tau: float = 0.05,
        buffer_size: int = 2000,
        batch_size: int = 32,
        seed: int | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.net = _Network(state_size, action_size, hidden, seed=seed)
        self.target = _Network(state_size, action_size, hidden, seed=seed)
        self.target.copy_from(self.net)
        self.memory: deque[tuple[list[float], int, float, list[float], bool]] = deque(maxlen=buffer_size)
        self.steps = 0
        self.train_calls = 0
        self.last_loss: float | None = None

    # ------------------------------------------------------------------ #
    # policy
    # ------------------------------------------------------------------ #
    def act(self, state: list[float]) -> dict[str, Any]:
        """Epsilon-greedy action selection."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.rng.random() < self.epsilon:
            action = self.rng.randrange(self.action_size)
            q_values, _ = self.net.forward(state)
        else:
            q_values, _ = self.net.forward(state)
            action = max(range(self.action_size), key=lambda a: q_values[a])
        return {"action": int(action), "q_values": [round(q, 6) for q in q_values], "epsilon": round(self.epsilon, 6)}

    def remember(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool) -> None:
        self.memory.append((list(state), int(action), float(reward), list(next_state), bool(done)))

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def _soft_update(self) -> None:
        """tau-weighted blend of online weights into the target network."""
        t, n = self.target, self.net
        for i in range(len(t.w1)):
            for j in range(len(t.w1[i])):
                t.w1[i][j] = self.tau * n.w1[i][j] + (1 - self.tau) * t.w1[i][j]
            t.b1[i] = self.tau * n.b1[i] + (1 - self.tau) * t.b1[i]
        for a in range(len(t.w2)):
            for i in range(len(t.w2[a])):
                t.w2[a][i] = self.tau * n.w2[a][i] + (1 - self.tau) * t.w2[a][i]
            t.b2[a] = self.tau * n.b2[a] + (1 - self.tau) * t.b2[a]

    def step(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool) -> dict[str, Any]:
        """One environment step: store the transition and train on a batch."""
        self.remember(state, action, reward, next_state, done)
        self.steps += 1
        self._train_batch()
        self.last_loss = self.last_loss if self.last_loss is not None else 0.0
        return {
            "steps": self.steps,
            "memory": len(self.memory),
            "epsilon": round(self.epsilon, 6),
            "loss": round(self.last_loss, 6),
        }

    def _train_batch(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = self.rng.sample(list(self.memory), self.batch_size)
        total_err = 0.0
        n = self.net
        for state, action, reward, next_state, done in batch:
            q_vals, h = n.forward(state)
            q_sa = q_vals[action]
            if done:
                target_q = reward
            else:
                t_q, _ = self.target.forward(next_state)
                target_q = reward + self.gamma * max(t_q)
            err = target_q - q_sa
            total_err += err * err

            # --- backprop (TD error) ---
            # output layer
            for i in range(len(n.w2[action])):
                n.w2[action][i] += self.lr * err * h[i]
            n.b2[action] += self.lr * err
            # hidden layer
            grad_h = [self.lr * err * n.w2[action][i] for i in range(len(h))]
            for i in range(self.net.hidden):
                dh = grad_h[i] * (1 - h[i] * h[i])
                for j in range(len(state)):
                    n.w1[i][j] += dh * state[j]
                n.b1[i] += dh
        self.last_loss = math.sqrt(total_err / len(batch))
        self.train_calls += 1
        if self.train_calls % 10 == 0:
            self._soft_update()

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "steps": self.steps,
            "memory": len(self.memory),
            "epsilon": round(self.epsilon, 6),
            "train_calls": self.train_calls,
            "last_loss": round(self.last_loss, 6) if self.last_loss is not None else None,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "dqn_agent",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "lr": self.lr,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "tau": self.tau,
            "steps": self.steps,
            "train_calls": self.train_calls,
            "last_loss": self.last_loss,
            "net": self.net.to_json(),
            "target": self.target.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DQNAgent:
        agent = cls(
            state_size=int(data["state_size"]),
            action_size=int(data["action_size"]),
            gamma=float(data.get("gamma", 0.9)),
            lr=float(data.get("lr", 0.05)),
            epsilon=float(data.get("epsilon", 1.0)),
            epsilon_min=float(data.get("epsilon_min", 0.05)),
            epsilon_decay=float(data.get("epsilon_decay", 0.995)),
            tau=float(data.get("tau", 0.05)),
        )
        agent.steps = int(data.get("steps", 0))
        agent.train_calls = int(data.get("train_calls", 0))
        agent.last_loss = data.get("last_loss")
        agent.net = _Network.from_json(data["net"])
        agent.target = _Network.from_json(data["target"])
        return agent
