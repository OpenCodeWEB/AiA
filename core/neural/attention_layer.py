"""Pure-math scaled dot-product self-attention layer.

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The layer projects an input matrix through learnable Wq/Wk/Wv, computes the
attention map (which tokens/features should attend to which), and returns a
context matrix. It learns online with a reconstruction objective: the
attended context is decoded back toward the input, so the projections
converge to meaningful correlation patterns.

    API: POST /attention/weights  {"input_matrix": [[...], ...]}
         -> {"attention_map": [...], "context": [...], "recon_loss": ...}

Flat-primitives JSON only. Pure stdlib.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """a (n x m) * b (m x p) -> (n x p)."""
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def _softmax_rows(m: list[list[float]]) -> list[list[float]]:
    out = []
    for row in m:
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps) or 1.0
        out.append([e / s for e in exps])
    return out


class AttentionLayer:
    """Learnable scaled dot-product self-attention with reconstruction training."""

    def __init__(self, d_model: int = 4, lr: float = 0.01, seed: int | None = None) -> None:
        self.d_model = d_model
        self.lr = lr
        self.rng = random.Random(seed)
        # projections: d x d, near-identity so attention starts sensible
        self.wq = [[(1.0 if i == j else 0.0) + self.rng.uniform(-0.02, 0.02) for j in range(d_model)] for i in range(d_model)]
        self.wk = [[(1.0 if i == j else 0.0) + self.rng.uniform(-0.02, 0.02) for j in range(d_model)] for i in range(d_model)]
        self.wv = [[(1.0 if i == j else 0.0) + self.rng.uniform(-0.02, 0.02) for j in range(d_model)] for i in range(d_model)]
        self.train_calls = 0
        self.last_recon_loss: float | None = None
        self.last_attention_map: list[list[float]] | None = None

    def forward(self, x: list[list[float]]) -> dict[str, Any]:
        """Full forward pass: projections -> attention map -> context."""
        n = len(x)
        d = self.d_model
        q = _matmul(x, self.wq)
        k = _matmul(x, self.wk)
        v = _matmul(x, self.wv)
        scores = _matmul(q, _transpose(k))
        scale = math.sqrt(d) or 1.0
        scores = [[s / scale for s in row] for row in scores]
        attn = _softmax_rows(scores)
        context = _matmul(attn, v)
        self.last_attention_map = attn
        return {
            "attention_map": [[round(a, 6) for a in row] for row in attn],
            "context": [[round(c, 6) for c in row] for row in context],
            "tokens": n,
            "d_model": d,
        }

    def attention_weights(self, x: list[list[float]]) -> list[list[float]]:
        """Just the attention map (per-token softmax focus)."""
        return self.forward(x)["attention_map"]

    def train(self, x: list[list[float]], steps: int = 20) -> dict[str, Any]:
        """Online training: minimize reconstruction error of the context."""
        n = len(x)
        d = self.d_model
        scale = math.sqrt(d)
        last = None
        for _ in range(steps):
            # --- forward ---
            q = _matmul(x, self.wq)
            k = _matmul(x, self.wk)
            v = _matmul(x, self.wv)
            scores = _matmul(q, _transpose(k))
            scores = [[s / scale for s in row] for row in scores]
            attn = _softmax_rows(scores)
            context = _matmul(attn, v)

            # reconstruction error: context should reproduce x
            err = [[context[i][j] - x[i][j] for j in range(d)] for i in range(n)]
            loss = math.sqrt(sum(e * e for row in err for e in row) / (n * d))
            last = loss

            # --- gradient of wv (through context) ---
            grad_wv = [[0.0] * d for _ in range(d)]
            for i in range(n):
                for a in range(n):
                    for j in range(d):
                        for m in range(d):
                            grad_wv[j][m] += 2 * err[i][j] * attn[i][a] * x[a][m] / (n * d)

            # --- gradient of wq/wk (through scores -> softmax -> context) ---
            grad_wq = [[0.0] * d for _ in range(d)]
            grad_wk = [[0.0] * d for _ in range(d)]
            for i in range(n):
                for a in range(n):
                    # d(context[i]) / d(attn[i][a]) = v[a]
                    dv = [v[a][j] for j in range(d)]
                    # d(attn[i][a]) / d(scores[i][b]) = attn[i][a] * (delta_ab - attn[i][b])
                    for b in range(n):
                        d_attn = attn[i][a] * ((1.0 if a == b else 0.0) - attn[i][b]) / scale
                        # propagate through q[i] and k[b]
                        for i_dim in range(d):
                            for m in range(d):
                                grad_wq[i_dim][m] += 2 * _dot(err[i], dv) * d_attn * x[i][m] * k[b][i_dim] / (n * d)
                                grad_wk[i_dim][m] += 2 * _dot(err[i], dv) * d_attn * x[b][m] * q[i][i_dim] / (n * d)

            # --- apply (clipped for stability) ---
            def _apply(w: list[list[float]], g: list[list[float]]) -> None:
                for i in range(d):
                    for j in range(d):
                        w[i][j] -= self.lr * max(-1.0, min(1.0, g[i][j]))

            _apply(self.wq, grad_wq)
            _apply(self.wk, grad_wk)
            _apply(self.wv, grad_wv)

        self.train_calls += 1
        self.last_recon_loss = last
        result = self.forward(x)
        result["recon_loss"] = round(last or 0.0, 6)
        result["train_calls"] = self.train_calls
        return result

    def status(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "train_calls": self.train_calls,
            "last_recon_loss": round(self.last_recon_loss, 6) if self.last_recon_loss is not None else None,
            "last_attention_map": self.last_attention_map,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "attention_layer",
            "d_model": self.d_model,
            "lr": self.lr,
            "train_calls": self.train_calls,
            "last_recon_loss": self.last_recon_loss,
            "wq": self.wq,
            "wk": self.wk,
            "wv": self.wv,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AttentionLayer:
        layer = cls(d_model=int(data.get("d_model", 4)), lr=float(data.get("lr", 0.01)))
        layer.wq = data.get("wq", layer.wq)
        layer.wk = data.get("wk", layer.wk)
        layer.wv = data.get("wv", layer.wv)
        layer.train_calls = int(data.get("train_calls", 0))
        layer.last_recon_loss = data.get("last_recon_loss")
        return layer
