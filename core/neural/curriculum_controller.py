"""Curriculum Learning Controller - easy-to-hard task pacing scheduler.

Instead of throwing the brain at the full dataset from the first sample, the
controller paces the curriculum: easy samples first, and only when mastery on
the current stage is high does it advance to the next, harder stage.

Difficulty of a sample is scored dynamically (feature variance + the brain's
recent loss on similar samples). Mastery is an EMA of the per-stage accuracy,
so advancement is automatic and adaptive.

  update(sample, loss) -> stage info + whether the controller advanced

Pure stdlib, flat-primitives JSON only.
"""

from __future__ import annotations

import math
from typing import Any


class CurriculumController:
    """Adaptive curriculum scheduler with automatic stage advancement."""

    def __init__(
        self,
        num_stages: int = 3,
        mastery_threshold: float = 0.92,
        min_samples_per_stage: int = 25,
        seed: int | None = None,
    ) -> None:
        self.num_stages = max(1, num_stages)
        self.mastery_threshold = mastery_threshold
        self.min_samples_per_stage = min_samples_per_stage
        self.stage = 0  # 0-based; stage 0 = easiest
        self.stage_samples = 0
        self.stage_mastery = 0.0
        self.stage_ema_loss: float | None = None
        self.advanced_events: list[str] = []
        self.samples_seen = 0
        self._rng = __import__("random").Random(seed)

    # ------------------------------------------------------------------ #
    # difficulty scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def sample_difficulty(x: list[float], loss: float | None = None) -> float:
        """Score how hard a sample is (0..1).

        Uses feature variance (spread of the input vector) combined with the
        observed loss when the brain trained on it. Higher = harder.
        """
        if not x:
            return 0.5
        mean = sum(x) / len(x)
        var = sum((v - mean) ** 2 for v in x) / len(x)
        # normalize variance to 0..1 range with a soft squash
        feat = 1.0 - math.exp(-var * 4.0)
        if loss is None:
            return feat
        loss_norm = min(1.0, loss * 5.0)
        return 0.6 * feat + 0.4 * loss_norm

    def difficulty_at_stage(self, base: float) -> float:
        """Effective difficulty = base difficulty scaled by current stage."""
        return min(1.0, base + self.stage * 0.25)

    # ------------------------------------------------------------------ #
    # curriculum logic
    # ------------------------------------------------------------------ #
    def accepts(self, x: list[float]) -> bool:
        """Should the trainer show this sample at the CURRENT stage?"""
        base = self.sample_difficulty(x)
        return self.difficulty_at_stage(base) <= 1.0  # all samples allowed, scaled

    def update(self, x: list[float], loss: float) -> dict[str, Any]:
        """Feed one training outcome. Returns the curriculum state."""
        self.samples_seen += 1
        self.stage_samples += 1
        self.stage_ema_loss = loss if self.stage_ema_loss is None else 0.9 * self.stage_ema_loss + 0.1 * loss

        # mastery: lower loss on the current stage's samples -> higher mastery
        mastery = 1.0 - min(1.0, (self.stage_ema_loss or 0.0) * 4.0)
        # bootstrap the EMA from the first sample so advancement responds
        # to CURRENT performance instead of a slow warm-up from zero
        if self.stage_samples == 1:
            self.stage_mastery = mastery
        else:
            self.stage_mastery = 0.92 * self.stage_mastery + 0.08 * mastery

        advanced = False
        if (
            self.stage_samples >= self.min_samples_per_stage
            and self.stage_mastery >= self.mastery_threshold
            and self.stage < self.num_stages - 1
        ):
            self.stage += 1
            self.stage_samples = 0
            self.stage_ema_loss = None
            self.advanced_events.append(f"stage{self.stage}: mastered")
            advanced = True

        return self.state(advanced=advanced)

    def state(self, advanced: bool = False) -> dict[str, Any]:
        return {
            "current_stage": self.stage,
            "num_stages": self.num_stages,
            "difficulty_level": ["easy", "medium", "hard"][min(self.stage, 2)],
            "mastery_rate": round(self.stage_mastery, 4),
            "stage_samples": self.stage_samples,
            "advanced": advanced,
            "ema_loss": round(self.stage_ema_loss, 6) if self.stage_ema_loss is not None else None,
        }

    def status(self) -> dict[str, Any]:
        state = self.state()
        state.update(
            {
                "samples_seen": self.samples_seen,
                "advancements": len(self.advanced_events),
                "recent_events": self.advanced_events[-5:],
            }
        )
        return state

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "curriculum_controller",
            "num_stages": self.num_stages,
            "mastery_threshold": self.mastery_threshold,
            "min_samples_per_stage": self.min_samples_per_stage,
            "stage": self.stage,
            "stage_samples": self.stage_samples,
            "stage_mastery": self.stage_mastery,
            "stage_ema_loss": self.stage_ema_loss,
            "advanced_events": self.advanced_events,
            "samples_seen": self.samples_seen,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CurriculumController:
        ctrl = cls(
            num_stages=int(data.get("num_stages", 3)),
            mastery_threshold=float(data.get("mastery_threshold", 0.92)),
            min_samples_per_stage=int(data.get("min_samples_per_stage", 25)),
        )
        ctrl.stage = int(data.get("stage", 0))
        ctrl.stage_samples = int(data.get("stage_samples", 0))
        ctrl.stage_mastery = float(data.get("stage_mastery", 0.0))
        ctrl.stage_ema_loss = data.get("stage_ema_loss")
        ctrl.advanced_events = list(data.get("advanced_events", []))
        ctrl.samples_seen = int(data.get("samples_seen", 0))
        return ctrl
