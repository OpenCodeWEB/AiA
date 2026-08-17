"""AiA executor abstraction — a delegated model/CLI that AiA can send work to."""

from __future__ import annotations

from typing import Any


class Executor:
    """Base class for delegated executors.

    Priority order in the registry: Gemini (session) → opencode agents → mock.
    """

    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def run(self, prompt: str, timeout: int = 120) -> tuple[str, dict[str, Any]]:
        """Execute a task, returning (output_text, meta)."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available()}
