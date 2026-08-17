"""Gemini bridge — connect to Gemini via a Google account session, zero API key.

Primary transport: the official `gemini` CLI, which stores the user's Google
account OAuth session locally (no developer API key involved). AiA only invokes
the CLI; it never sees or stores Google credentials.

Experimental (off by default): cookie/session bridge to Gemini web — fragile
and ToS-risky; enabled only via env `AIA_GEMINI_COOKIE_BRIDGE=1`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any

DEFAULT_GEMINI_CMD = "gemini --prompt {prompt}"


class GeminiUnavailableError(RuntimeError):
    pass


class GeminiBridge:
    def __init__(self) -> None:
        self.template = os.environ.get("AIA_GEMINI_CMD", DEFAULT_GEMINI_CMD)
        self._cmd = shlex.split(self.template)[0]
        self.cookie_bridge = os.environ.get("AIA_GEMINI_COOKIE_BRIDGE", "0") == "1"

    def available(self) -> bool:
        return shutil.which(self._cmd) is not None

    def ask(self, prompt: str, timeout: int = 180) -> tuple[str, str]:
        """Run gemini-cli non-interactively. Returns (stdout, stderr)."""
        if not self.available():
            raise GeminiUnavailableError(f"gemini CLI not found on PATH ({self._cmd!r})")
        parts = shlex.split(self.template)
        cmd = [parts[0]] + [p.replace("{prompt}", prompt) for p in parts[1:]]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            raise GeminiUnavailableError("gemini CLI timed out") from e
        return (proc.stdout or "").strip(), (proc.stderr or "").strip()

    def status(self) -> dict[str, Any]:
        if self.cookie_bridge:
            raise NotImplementedError(
                "AIA_GEMINI_COOKIE_BRIDGE is experimental and not implemented — "
                "use the gemini-cli OAuth session instead (no API key needed)."
            )
        if self.available():
            return {"connected": True, "provider": "Google Gemini via gemini-cli OAuth", "api_key": "none"}
        return {
            "connected": False,
            "provider": "Google Gemini via gemini-cli OAuth",
            "api_key": "none",
            "hint": "install the official gemini CLI and sign in with your Google account",
        }

    def connect(self, session_cookie: str | None = None) -> dict[str, Any]:
        """PRD entry point: connect with a Google account (zero API key)."""
        return self.status()
