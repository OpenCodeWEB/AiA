"""Recurrent Experience Memory - Elman-style simple RNN for sequences.

The main brain treats every sample independently. This module gives AiA a
sense of TIME: an Elman recurrent network keeps a hidden state that is fed
back into itself at the next time step, so it can learn and predict
sequences (usage patterns, time series, command flows).

  h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
  y_t = tanh(W_hy * h_t + b_y)

Training uses truncated backpropagation through time over a small lookback
window (pure stdlib, fast enough for online use). Flat JSON serialization.
"""

from __future__ import annotations

import math
import random
from typing import Any


class ElmanRNN:
    """A small online Elman RNN with truncated BPTT."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 8,
        output_size: int = 1,
        lr: float = 0.05,
        lookback: int = 8,
        seed: int | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.lookback = max(2, lookback)
        self.lr_floor = 1e-5

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / input_size)
        self.w_xh: list[list[float]] = [[rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b_h: list[float] = [0.0 for _ in range(hidden_size)]
        scale_h = math.sqrt(2.0 / hidden_size)
        self.w_hh: list[list[float]] = [[rng.uniform(-scale_h, scale_h) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.w_hy: list[list[float]] = [[rng.uniform(-scale, scale) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b_y: list[float] = [0.0 for _ in range(output_size)]

        self.hidden: list[float] = [0.0 for _ in range(hidden_size)]  # current state
        self.samples_seen = 0
        self.sequences_seen = 0
        self.ema_loss: float | None = None

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def step(self, x: list[float]) -> list[float]:
        """One time step: update the hidden state and predict the output."""
        h = [0.0 for _ in range(self.hidden_size)]
        for j in range(self.hidden_size):
            total = self.b_h[j]
            for i in range(self.input_size):
                total += x[i] * self.w_xh[i][j]
            for k in range(self.hidden_size):
                total += self.hidden[k] * self.w_hh[k][j]
            h[j] = math.tanh(total)
        self.hidden = h
        y = []
        for o in range(self.output_size):
            total = self.b_y[o]
            for j in range(self.hidden_size):
                total += h[j] * self.w_hy[j][o]
            y.append(math.tanh(total))
        return y

    def reset_state(self) -> None:
        self.hidden = [0.0 for _ in range(self.hidden_size)]

    # ------------------------------------------------------------------ #
    # learning (truncated BPTT over lookback window)
    # ------------------------------------------------------------------ #
    def train_sequence(self, seq: list[list[float]], targets: list[list[float]]) -> float:
        """Train on a whole sequence with truncated BPTT. Returns avg loss.

        `seq` is a list of input vectors; `targets` parallel output vectors.
        Only the last `lookback` steps update the input/recurrent weights;
        errors from earlier steps still flow through the recurrence chain.
        """
        if len(seq) != len(targets) or not seq:
            raise ValueError("seq and targets must be non-empty and equal length")
        n_steps = len(seq)
        horizon = min(n_steps, self.lookback)
        hid, out, ins = self.hidden_size, self.output_size, self.input_size

        # tanh output saturates at +/-1, so binary targets {0,1} are mapped to
        # {+1,-1}; otherwise the gradient vanishes exactly at y=0 and the
        # network can never learn the SIGN of the prediction.
        targets = [[2.0 * targets[t][o] - 1.0 for o in range(out)] for t in range(n_steps)]

        # forward pass, storing activations
        h_states: list[list[float]] = []
        y_preds: list[list[float]] = []
        self.reset_state()
        for t in range(n_steps):
            y = self.step(seq[t])
            h_states.append(self.hidden[:])
            y_preds.append(y[:])

        losses = [sum((y_preds[t][o] - targets[t][o]) ** 2 for o in range(out)) / out for t in range(n_steps)]
        avg_loss = sum(losses) / n_steps

        # mean-normalized, clipped gradients (stable online training)
        g_scale = 1.0 / n_steps
        d_h_next: list[float] = [0.0 for _ in range(hid)]
        for t in range(n_steps - 1, -1, -1):
            d_y = [
                max(-1.0, min(1.0, (y_preds[t][o] - targets[t][o]) * (1.0 - y_preds[t][o] ** 2) * g_scale))
                for o in range(out)
            ]
            dh = [
                max(
                    -1.0,
                    min(1.0, (1.0 - h_states[t][j] ** 2) * (sum(d_y[o] * self.w_hy[j][o] for o in range(out)) + d_h_next[j])),
                )
                for j in range(hid)
            ]
            # propagate the error one step back through the (pre-update) weights
            propagated = [sum(dh[j] * self.w_hh[k][j] for j in range(hid)) for k in range(hid)]

            # output weights (always updated)
            for j in range(hid):
                for o in range(out):
                    self.w_hy[j][o] -= self.lr * d_y[o] * h_states[t][j]
            for o in range(out):
                self.b_y[o] -= self.lr * d_y[o]

            # input / recurrent weights only inside the truncation horizon
            if t >= n_steps - horizon:
                for i in range(ins):
                    for j in range(hid):
                        self.w_xh[i][j] -= self.lr * dh[j] * seq[t][i]
                for j in range(hid):
                    self.b_h[j] -= self.lr * dh[j]
                if t > 0:
                    for k in range(hid):
                        for j in range(hid):
                            self.w_hh[k][j] -= self.lr * dh[j] * h_states[t - 1][k]
                d_h_next = propagated
            else:
                d_h_next = [0.0 for _ in range(hid)]  # truncated BPTT boundary

        self.sequences_seen += 1
        self.samples_seen += n_steps
        self.ema_loss = avg_loss if self.ema_loss is None else 0.9 * self.ema_loss + 0.1 * avg_loss
        self.lr = max(self.lr * 0.9995, self.lr_floor)
        return avg_loss

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict_sequence(self, seq: list[list[float]], steps_ahead: int = 1) -> list[list[float]]:
        """Forecast `steps_ahead` outputs by continuing the pattern.

        The first prediction is the model's own last output (persistence
        forecast: in an alternating/periodic signal the current target IS the
        next value), then the forecast rolls forward autoregressively by
        feeding each prediction back as the next input.
        """
        self.reset_state()
        y_last: list[float] | None = None
        for x in seq:
            y_last = self.step(x)
        if y_last is None:
            y_last = [0.0 for _ in range(self.output_size)]
        preds = [y_last[:]]
        last = y_last
        for _ in range(max(0, steps_ahead - 1)):
            y = self.step(last)
            preds.append(y[:])
            last = y[: self.input_size] if len(y) >= self.input_size else last
        return preds

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "lookback": self.lookback,
            "sequences_seen": self.sequences_seen,
            "samples_seen": self.samples_seen,
            "ema_loss": round(self.ema_loss, 6) if self.ema_loss is not None else None,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "elman_rnn",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "lr": self.lr,
            "lookback": self.lookback,
            "w_xh": self.w_xh,
            "b_h": self.b_h,
            "w_hh": self.w_hh,
            "w_hy": self.w_hy,
            "b_y": self.b_y,
            "sequences_seen": self.sequences_seen,
            "samples_seen": self.samples_seen,
            "ema_loss": self.ema_loss,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ElmanRNN:
        net = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            lr=float(data.get("lr", 0.05)),
            lookback=int(data.get("lookback", 8)),
        )
        net.w_xh = [list(row) for row in data["w_xh"]]
        net.b_h = list(data["b_h"])
        net.w_hh = [list(row) for row in data["w_hh"]]
        net.w_hy = [list(row) for row in data["w_hy"]]
        net.b_y = list(data["b_y"])
        net.sequences_seen = int(data.get("sequences_seen", 0))
        net.samples_seen = int(data.get("samples_seen", 0))
        net.ema_loss = data.get("ema_loss")
        return net
