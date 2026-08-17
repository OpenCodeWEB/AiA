"""Mock executor — always available dev fallback (tagged as source_model="mock")."""

from __future__ import annotations

import time
from typing import Any

from executors.base import Executor


class MockExecutor(Executor):
    name = "mock"

    def available(self) -> bool:
        return True

    def run(self, prompt: str, timeout: int = 120) -> tuple[str, dict[str, Any]]:
        meta = {"executor": self.name, "ts": time.time()}
        return f"[mock-executor] simulated solution for: {prompt[:120]}", meta
