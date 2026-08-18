"""Associative Hopfield Memory - pattern storage & reconstruction.

A classic discrete Hopfield network: patterns are stored with the Hebbian
rule  W = sum(x_i * x_i^T)  (bipolar vectors, zero diagonal). A corrupted or
noisy input is then reconstructed by energy minimization: repeatedly apply
  s_j = sign(sum_i W_ij * s_i)
until the state stops changing (or a step budget is exhausted).

This gives AiA an associative memory: show it a partial / noisy pattern and
it recovers the closest stored original. Pure stdlib, flat-primitives JSON.
"""

from __future__ import annotations

from typing import Any


class HopfieldMemory:
    """Discrete Hopfield network with Hebbian storage and recall."""

    def __init__(self, size: int = 16) -> None:
        self.size = size
        self.weight: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        self.patterns: list[list[int]] = []
        self.recalls = 0
        self.successful_recalls = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bipolar(values: list[float]) -> list[int]:
        """Map 0 -> -1 and 1 -> +1 (any float below 0.5 becomes -1)."""
        return [1 if v >= 0.5 else -1 for v in values]

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #
    def store(self, pattern: list[float]) -> int:
        """Store one pattern (0/1 or float values). Returns pattern index."""
        if len(pattern) != self.size:
            raise ValueError(f"pattern must have {self.size} elements")
        s = self._bipolar(pattern)
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.weight[i][j] += s[i] * s[j]
        self.patterns.append(s)
        return len(self.patterns) - 1

    def clear(self) -> None:
        self.weight = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        self.patterns = []
        self.recalls = 0
        self.successful_recalls = 0

    # ------------------------------------------------------------------ #
    # recall
    # ------------------------------------------------------------------ #
    def recall(
        self,
        corrupted: list[float],
        max_steps: int = 50,
        return_steps: bool = False,
    ) -> dict[str, Any]:
        """Reconstruct the closest stored pattern via energy minimization.

        Returns the recovered pattern as 0/1 values plus diagnostics.
        """
        if len(corrupted) != self.size:
            raise ValueError(f"corrupted pattern must have {self.size} elements")
        s = self._bipolar(corrupted)
        steps = 0
        for _ in range(max_steps):
            updated = False
            for j in range(self.size):
                total = 0.0
                for i in range(self.size):
                    if i != j:
                        total += self.weight[i][j] * s[i]
                new = 1 if total >= 0 else -1
                if new != s[j]:
                    s[j] = new
                    updated = True
            steps += 1
            if not updated:
                break

        recovered = [1 if v == 1 else 0 for v in s]
        self.recalls += 1
        # success = the recall converged to a stored pattern (stored are
        # bipolar +/-1; recovered is 0/1, so compare on the same scale)
        matched = any(
            all(a == (1 if b == 1 else 0) for a, b in zip(recovered, stored, strict=True))
            for stored in self.patterns
        )
        if matched:
            self.successful_recalls += 1

        result: dict[str, Any] = {
            "recovered": recovered,
            "steps": steps,
            "converged": steps < max_steps,
            "matched_stored": matched,
        }
        if return_steps:
            result["energy"] = self._energy(s)
        return result

    def _energy(self, s: list[int]) -> float:
        """Compute the Hopfield energy of a state (lower = more stable)."""
        total = 0.0
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    total += self.weight[i][j] * s[i] * s[j]
        return -0.5 * total

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "patterns_stored": len(self.patterns),
            "capacity_estimate": max(1, int(0.138 * self.size)),
            "recalls": self.recalls,
            "successful_recalls": self.successful_recalls,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "hopfield_memory",
            "size": self.size,
            "weight": self.weight,
            "patterns": self.patterns,
            "recalls": self.recalls,
            "successful_recalls": self.successful_recalls,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HopfieldMemory:
        mem = cls(size=int(data["size"]))
        mem.weight = [[float(v) for v in row] for row in data["weight"]]
        mem.patterns = [[int(v) for v in p] for p in data.get("patterns", [])]
        mem.recalls = int(data.get("recalls", 0))
        mem.successful_recalls = int(data.get("successful_recalls", 0))
        return mem
