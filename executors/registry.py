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

    def describe(self) -> list[dict[str, Any]]:
        return [ex.describe() for ex in self.executors]
