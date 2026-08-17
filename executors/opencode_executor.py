"""OpenCode agent executor — delegates to the `opencode run` CLI when available.

Command template overridable via env `AIA_OPENCODE_CMD` (e.g.
`opencode run --model gpt-5 {prompt}`). Default: `opencode run {prompt}`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from typing import Any

from executors.base import Executor


class OpenCodeExecutor(Executor):
    name = "opencode"

    def __init__(self) -> None:
        template = os.environ.get("AIA_OPENCODE_CMD", "opencode run {prompt}")
        self.template = template
        self._cmd = shlex.split(template)[0]

    def available(self) -> bool:
        return shutil.which(self._cmd) is not None

    def run(self, prompt: str, timeout: int = 90) -> tuple[str, dict[str, Any]]:
        cmd = [self._cmd] + shlex.split(self.template)[1:]
        cmd = [c.replace("{prompt}", prompt) for c in cmd]
        started = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
        output = proc.stdout.strip() or proc.stderr.strip() or f"[opencode] exit {proc.returncode}"
        meta = {"executor": self.name, "duration_ms": int((time.time() - started) * 1000), "exit": proc.returncode}
        return output, meta
