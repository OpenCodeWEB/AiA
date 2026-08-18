"""Adaptive Autoencoder - experience compression & vector memory for AiA.

The main brain (SelfEvolvingNN) keeps an experience replay buffer of at most
2000 samples. When the buffer is full, old samples are normally forgotten.
This module solves that: it trains a small bottleneck autoencoder on the
brain's experiences and compresses them into tiny latent vectors, so an
effectively unlimited history can be kept in almost no memory.

  ENCODER: input(n) -> tanh(hidden) -> latent(bottleneck)
  DECODER: latent(bottleneck) -> tanh(hidden) -> sigmoid(output)

The reconstruction error is also the foundation of the anomaly detector
(see anomaly_detector.py). Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from typing import Any


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class AutoencoderCompressor:
    """A small online autoencoder that compresses samples into latent vectors.

    Learns continuously from every sample it sees (reconstruction loss), so
    the latent space stays adapted to the brain's current experience stream.
    """

    def __init__(
        self,
        input_size: int = 2,
        bottleneck: int = 2,
        hidden: int = 8,
        lr: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if bottleneck >= input_size:
            raise ValueError("bottleneck must be smaller than input_size (it is a compressor)")
        self.input_size = input_size
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.lr = lr
        self.lr_floor = 1e-4

        rng = random.Random(seed)
        # encoder: input -> hidden
        self.w_in: list[list[float]] = [[rng.uniform(-1, 1) for _ in range(hidden)] for _ in range(input_size)]
        self.b_in: list[float] = [0.0 for _ in range(hidden)]
        # encoder: hidden -> bottleneck
        self.w_hid: list[list[float]] = [[rng.uniform(-1, 1) for _ in range(bottleneck)] for _ in range(hidden)]
        self.b_hid: list[float] = [0.0 for _ in range(bottleneck)]
        # decoder: bottleneck -> hidden
        self.w_bot: list[list[float]] = [[rng.uniform(-1, 1) for _ in range(hidden)] for _ in range(bottleneck)]
        self.b_bot: list[float] = [0.0 for _ in range(hidden)]
        # decoder: hidden -> output
        self.w_out: list[list[float]] = [[rng.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden)]
        self.b_out: list[float] = [0.0 for _ in range(input_size)]

        self.samples_seen = 0
        self.ema_loss: float | None = None
        self.recent_errors: deque[float] = deque(maxlen=256)

    # ------------------------------------------------------------------ #
    # forward passes
    # ------------------------------------------------------------------ #
    def encode(self, x: list[float]) -> list[float]:
        """Map input to the compressed latent vector."""
        h = [math.tanh(sum(x[i] * self.w_in[i][j] for i in range(self.input_size)) + self.b_in[j]) for j in range(self.hidden)]
        z = [math.tanh(sum(h[j] * self.w_hid[j][k] for j in range(self.hidden)) + self.b_hid[k]) for k in range(self.bottleneck)]
        return z

    def decode(self, z: list[float]) -> list[float]:
        """Rebuild the input from a latent vector."""
        h = [math.tanh(sum(z[k] * self.w_bot[k][j] for k in range(self.bottleneck)) + self.b_bot[j]) for j in range(self.hidden)]
        out = [_sigmoid(sum(h[j] * self.w_out[j][i] for j in range(self.hidden)) + self.b_out[i]) for i in range(self.input_size)]
        return out

    def forward(self, x: list[float]) -> list[float]:
        """Full autoencode: compress then reconstruct."""
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: list[float]) -> float:
        """MSE between the input and its reconstruction (anomaly signal)."""
        out = self.forward(x)
        return sum((a - b) ** 2 for a, b in zip(x, out, strict=True)) / self.input_size

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def train(self, x: list[float]) -> float:
        """One online reconstruction-learning step. Returns the loss."""
        n, hd, b = self.input_size, self.hidden, self.bottleneck
        # --- encoder forward ---
        h1 = [math.tanh(sum(x[i] * self.w_in[i][j] for i in range(n)) + self.b_in[j]) for j in range(hd)]
        z = [math.tanh(sum(h1[j] * self.w_hid[j][k] for j in range(hd)) + self.b_hid[k]) for k in range(b)]
        # --- decoder forward ---
        h2 = [math.tanh(sum(z[k] * self.w_bot[k][j] for k in range(b)) + self.b_bot[j]) for j in range(hd)]
        y = [_sigmoid(sum(h2[j] * self.w_out[j][i] for j in range(hd)) + self.b_out[i]) for i in range(n)]
        # --- reconstruction loss ---
        loss = sum((y[i] - x[i]) ** 2 for i in range(n)) / n

        # --- decoder backprop ---
        d_out = [(y[i] - x[i]) * y[i] * (1.0 - y[i]) for i in range(n)]
        d_h2 = [0.0 for _ in range(hd)]
        for j in range(hd):
            grad = 0.0
            for i in range(n):
                grad += d_out[i] * self.w_out[j][i]
            d_h2[j] = grad * (1.0 - h2[j] * h2[j])
        for j in range(hd):
            for i in range(n):
                self.w_out[j][i] -= self.lr * d_out[i] * h2[j]
        for i in range(n):
            self.b_out[i] -= self.lr * d_out[i]
        for k in range(b):
            for j in range(hd):
                self.w_bot[k][j] -= self.lr * d_h2[j] * z[k]
            self.b_bot[k] -= self.lr * sum(d_h2)

        # --- encoder backprop (through bottleneck + first hidden) ---
        d_z = [0.0 for _ in range(b)]
        for k in range(b):
            grad = 0.0
            for j in range(hd):
                grad += d_h2[j] * self.w_bot[k][j]
            d_z[k] = grad * (1.0 - z[k] * z[k])
        for k in range(b):
            for j in range(hd):
                self.w_hid[j][k] -= self.lr * d_z[k] * h1[j]
            self.b_hid[k] -= self.lr * sum(d_z)
        d_h1 = [0.0 for _ in range(hd)]
        for j in range(hd):
            grad = 0.0
            for k in range(b):
                grad += d_z[k] * self.w_hid[j][k]
            d_h1[j] = grad * (1.0 - h1[j] * h1[j])
        for j in range(hd):
            for i in range(n):
                self.w_in[i][j] -= self.lr * d_h1[j] * x[i]
            self.b_in[j] -= self.lr * sum(d_h1)

        # --- bookkeeping ---
        self.samples_seen += 1
        self.ema_loss = loss if self.ema_loss is None else 0.9 * self.ema_loss + 0.1 * loss
        self.recent_errors.append(loss)
        self.lr = max(self.lr * 0.999, self.lr_floor)
        return loss

    def compress(self, samples: list[list[float]]) -> list[list[float]]:
        """Compress many samples into latent vectors (the vector memory)."""
        return [self.encode(x) for x in samples]

    def train_batch(self, samples: list[list[float]]) -> float:
        """Train on a batch of samples, returns average loss."""
        losses = [self.train(x) for x in samples]
        return sum(losses) / len(losses)

    # ------------------------------------------------------------------ #
    # serialization (flat primitives only)
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "bottleneck": self.bottleneck,
            "hidden": self.hidden,
            "samples_seen": self.samples_seen,
            "ema_reconstruction_loss": round(self.ema_loss, 6) if self.ema_loss is not None else None,
            "compression_ratio": f"{self.input_size // self.bottleneck}x" if self.bottleneck else "0x",
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "autoencoder",
            "input_size": self.input_size,
            "bottleneck": self.bottleneck,
            "hidden": self.hidden,
            "lr": self.lr,
            "w_in": self.w_in, "b_in": self.b_in,
            "w_hid": self.w_hid, "b_hid": self.b_hid,
            "w_bot": self.w_bot, "b_bot": self.b_bot,
            "w_out": self.w_out, "b_out": self.b_out,
            "samples_seen": self.samples_seen,
            "ema_loss": self.ema_loss,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AutoencoderCompressor:
        net = cls(
            input_size=int(data["input_size"]),
            bottleneck=int(data["bottleneck"]),
            hidden=int(data["hidden"]),
            lr=float(data.get("lr", 0.1)),
        )
        net.w_in = [list(row) for row in data["w_in"]]
        net.b_in = list(data["b_in"])
        net.w_hid = [list(row) for row in data["w_hid"]]
        net.b_hid = list(data["b_hid"])
        net.w_bot = [list(row) for row in data["w_bot"]]
        net.b_bot = list(data["b_bot"])
        net.w_out = [list(row) for row in data["w_out"]]
        net.b_out = list(data["b_out"])
        net.samples_seen = int(data.get("samples_seen", 0))
        net.ema_loss = data.get("ema_loss")
        return net

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh)

    @classmethod
    def load(cls, path: str) -> AutoencoderCompressor:
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))
