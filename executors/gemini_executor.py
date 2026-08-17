"""Gemini executor — runs through the gemini-cli OAuth session (zero API key).

The google account session is stored by gemini-cli itself; AiA never sees the
token. Command template overridable via env `AIA_GEMINI_CMD` (default
`gemini --prompt {prompt}` — adjust for your gemini-cli version).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from typing import Any

from executors.base import Executor
from gemini_bridge import GeminiBridge


class GeminiExecutor(Executor):
    name = "gemini"

    def __init__(self) -> None:
        self.bridge = GeminiBridge()

    def available(self) -> bool:
        return self.bridge.available()

    def run(self, prompt: str, timeout: int = 180) -> tuple[str, dict[str, Any]]:
        if not self.bridge.available():
            raise RuntimeError("gemini-cli not available")
        started = time.time()
        output, err = self.bridge.ask(prompt, timeout=timeout)
        if not output and err:
            raise RuntimeError(f"gemini failed: {err[:300]}")
        meta = {"executor": self.name, "duration_ms": int((time.time() - started) * 1000)}
        return output, meta
