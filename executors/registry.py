"""Executor registry — priority order: Gemini → opencode → mock."""

from __future__ import annotations

from typing import Any

from executors.base import Executor
from executors.gemini_executor import GeminiExecutor
from executors.mock import MockExecutor
from executors.opencode_executor import OpenCodeExecutor


def default_executors() -> list[Executor]:
    return [GeminiExecutor(), OpenCodeExecutor(), MockExecutor()]


class ExecutorRegistry:
    def __init__(self, executors: list[Executor] | None = None) -> None:
        self.executors = executors or default_executors()

    def pick(self, prompt: str) -> Executor | None:
        """First available executor in priority order."""
        for ex in self.executors:
            if ex.available():
                return ex
        return None

    def run_with_fallback(self, prompt: str, timeout: int = 120) -> tuple[str, str, list[str]]:
        """Run on the first available executor; on failure fall through to the next.

        Returns (output, executor_name, errors) — never raises for executor
        failures as long as at least one executor produced output.
        """
        errors: list[str] = []
        for ex in self.executors:
            if not ex.available():
                continue
            try:
                output, meta = ex.run(prompt, timeout=timeout)
                return output, ex.name, errors
            except Exception as e:  # noqa: BLE001 — try the next executor
                errors.append(f"{ex.name}: {e}")
        if not errors:
            raise RuntimeError("no executor available")
        raise RuntimeError(f"all executors failed: {'; '.join(errors)}")

    def describe(self) -> list[dict[str, Any]]:
        return [ex.describe() for ex in self.executors]
