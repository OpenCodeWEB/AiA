"""Anomaly & Outlier Detector - auto-threshold reconstruction error monitor.

Wraps a small autoencoder and streams its reconstruction errors through
Welford statistics (mean mu / standard deviation sigma). Inputs whose error
exceeds mu + k*sigma are flagged as anomalies. The statistics adapt online,
so the detector tracks a slowly changing environment without any config.

  check(input) -> { is_anomaly, anomaly_score, error, threshold }

Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import math
from typing import Any

from .autoencoder_compressor import AutoencoderCompressor


class AnomalyDetector:
    """Streaming anomaly detector built on autoencoder reconstruction error."""

    def __init__(
        self,
        input_size: int = 2,
        bottleneck: int = 1,
        hidden: int = 6,
        k_sigma: float = 3.0,
        seed: int | None = None,
    ) -> None:
        self.autoencoder = AutoencoderCompressor(
            input_size=input_size, bottleneck=bottleneck, hidden=hidden, seed=seed
        )
        self.k_sigma = k_sigma
        self.input_size = input_size

        # Welford streaming statistics over reconstruction errors
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.checked: int = 0
        self.anomalies_found: int = 0

    # ------------------------------------------------------------------ #
    # streaming statistics
    # ------------------------------------------------------------------ #
    def _update_stats(self, err: float) -> None:
        self.n += 1
        delta = err - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (err - self.mean)

    @property
    def sigma(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    @property
    def threshold(self) -> float:
        return self.mean + self.k_sigma * self.sigma

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def train(self, x: list[float]) -> float:
        """Learn the normal shape of the input stream (autoencoder step)."""
        return self.autoencoder.train(x)

    def check(self, x: list[float]) -> dict[str, Any]:
        """Score one input. Returns the anomaly verdict and diagnostics."""
        err = self.autoencoder.reconstruction_error(x)
        # threshold from the PAST error distribution: the current sample must
        # not pollute the very statistics used to judge it
        thr = self.threshold
        is_anomaly = bool(self.n >= 5 and err > thr)
        self._update_stats(err)
        self.checked += 1
        if is_anomaly:
            self.anomalies_found += 1
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(err, 6),
            "error": round(err, 6),
            "threshold": round(thr, 6),
            "mean": round(self.mean, 6),
            "sigma": round(self.sigma, 6),
            "n_samples": self.n,
        }

    def status(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "k_sigma": self.k_sigma,
            "checked": self.checked,
            "anomalies_found": self.anomalies_found,
            "mean_error": round(self.mean, 6),
            "sigma_error": round(self.sigma, 6),
            "threshold": round(self.threshold, 6),
            "autoencoder": self.autoencoder.status(),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "anomaly_detector",
            "k_sigma": self.k_sigma,
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
            "checked": self.checked,
            "anomalies_found": self.anomalies_found,
            "autoencoder": self.autoencoder.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnomalyDetector:
        det = cls(input_size=int(data["autoencoder"]["input_size"]))
        det.autoencoder = AutoencoderCompressor.from_json(data["autoencoder"])
        det.k_sigma = float(data.get("k_sigma", 3.0))
        det.n = int(data.get("n", 0))
        det.mean = float(data.get("mean", 0.0))
        det.m2 = float(data.get("m2", 0.0))
        det.checked = int(data.get("checked", 0))
        det.anomalies_found = int(data.get("anomalies_found", 0))
        return det
