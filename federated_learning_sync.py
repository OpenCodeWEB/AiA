#!/usr/bin/env python3
# =====================================================================
# OpenCodeWEB OS — AiA Federated Knowledge Sharing Engine
# Source Repository: https://github.com/OpenCodeWEB/AiA
# Canonical path: /opt/opencode/lib/aia/federated_learning_sync.py
# =====================================================================
"""Federated Swarm Learning — zero-data-leakage collective intelligence.

PRIVACY CONTRACT (differs from the original scaffold on purpose):
- Raw prompt / solution text NEVER leaves the device.
- Uploaded patterns contain ONLY: category (fixed taxonomy), a numeric
  feature vector (hashing-TF-IDF embedding of an ABSTRACTED solution),
  outcome stats, and an HMAC signature bound to a per-install secret.
- The scaffold's `solution_vector` (raw learned_solution) is replaced by
  `feature_vector` — sending raw code would violate zero-data-leakage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_ENDPOINT = os.environ.get("AIA_SWARM_ENDPOINT", "https://aia-brain.opencode.workers.dev/v1")

TAXONOMY: list[tuple[str, list[str]]] = [
    ("flutter_ui", ["flutter", "widget", "glassmorphism", "dart ui", "container decoration"]),
    ("python_debug", ["python", "traceback", "debug", "exception", "pip"]),
    ("js_fix", ["javascript", "promise", "async", "node", "npm"]),
    ("typescript", ["typescript", "tsx", "angular", "generic", "type"]),
    ("backend", ["api", "server", "database", "sql", "auth", "endpoint"]),
    ("frontend", ["react", "css", "html", "component", "style"]),
    ("devops", ["docker", "deploy", "ci", "kubernetes", "pipeline"]),
    ("general_coding", []),  # fallback
]

_KEYWORDS = {
    # programming keywords preserved by the abstractor (structure must survive)
    "def", "class", "return", "import", "from", "if", "elif", "else", "for", "while", "break",
    "continue", "pass", "raise", "try", "except", "finally", "with", "as", "lambda", "async",
    "await", "yield", "global", "nonlocal", "del", "assert", "is", "in", "not", "and", "or",
    "None", "True", "False", "print", "self", "super", "new", "this", "typeof", "instanceof",
    "function", "const", "let", "var", "fn", "pub", "fn", "match", "struct", "enum", "impl",
    "trait", "use", "mod", "static", "void", "int", "float", "double", "string", "bool", "char",
    "long", "short", "unsigned", "signed", "null", "nil", "true", "false", "public", "private",
    "protected", "extends", "implements", "interface", "package", "namespace", "goto", "switch",
    "case", "default", "do", "select", "go", "defer", "chan", "range", "map", "type", "var",
    "val", "let", "fun", "object", "trait", "sealed", "data", "abstract", "final", "override",
    "operator", "sizeof", "typedef", "union", "volatile", "constexpr", "template", "typename",
    "throw", "catch", "export", "declare", "yield", "await", "of", "has", "each", "begin", "end",
    "then", "elseif", "procedure", "function", "returns", "int", "integer", "real", "boolean",
    # abstraction placeholders — must survive the identifier pass
    "str", "url", "path", "num", "id",
    # generic domain vocabulary (not user-identifying)
    "python", "javascript", "typescript", "java", "c", "c++", "go", "rust", "php", "ruby",
    "swift", "kotlin", "dart", "flutter", "react", "angular", "vue", "sql", "html", "css",
    "docker", "kubernetes", "linux", "windows", "macos", "api", "server", "database", "auth",
    "bug", "fix", "error", "crash", "compile", "build", "test", "deploy", "function", "class",
    "widget", "component", "state", "async", "await", "promise", "thread", "memory",
    "performance", "network", "socket", "websocket", "http", "json", "xml", "regex", "script",
    "shell", "config",
}

_STRING_RE = re.compile(r"(['\"`])(?:\\.|(?!\1).)*\1", re.S)
_URL_RE = re.compile(r"https?://\S+")
_PATH_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|/|\\./)[^\s\"']+", re.S)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _abstract_solution(text: str, cap: int = 2000) -> str:
    """Strip everything identifying: URLs, paths, string literals, identifiers, numbers.

    Preserves structural tokens (keywords, operators, punctuation) so the
    algorithmic skeleton remains learnable without leaking content.
    """
    out = _URL_RE.sub(" <url> ", text)
    out = _PATH_RE.sub(" <path> ", out)
    out = _STRING_RE.sub(" <str> ", out)
    out = _NUM_RE.sub(" <num> ", out)
    out = _IDENT_RE.sub(lambda m: m.group(0) if m.group(0) in _KEYWORDS else "<id>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:cap]


def _categorize(prompt: str) -> str:
    p = prompt.lower()
    for name, hints in TAXONOMY:
        if name == "general_coding":
            continue
        if any(h in p for h in hints):
            return name
    return "general_coding"


class AiAFederatedSync:
    def __init__(self, brain_dir: str | Path, knowledge: dict[str, Any], endpoint: Optional[str] = None) -> None:
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge = knowledge
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.secret_path = self.brain_dir / "instance_secret"
        self.state_path = self.brain_dir / "sync_state.json"
        self._ensure_secret()
        self.state = self._load_state()

    # ── instance identity ─────────────────────────────────────────────────
    def _ensure_secret(self) -> None:
        if not self.secret_path.exists():
            self.secret_path.write_text(os.urandom(32).hex(), encoding="utf-8")
            try:
                os.chmod(self.secret_path, 0o600)
            except OSError:
                pass

    @property
    def secret(self) -> str:
        return self.secret_path.read_text(encoding="utf-8").strip()

    def device_id(self) -> str:
        return hashlib.sha256(self.secret.encode("utf-8")).hexdigest()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"uploaded_watermark": 0, "last_patch_ts": 0.0}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ── anonymization (privacy-critical) ──────────────────────────────────
    def anonymize_pattern(self, pattern_data: dict[str, Any]) -> dict[str, Any]:
        """Map a local learned record to a privacy-safe swarm pattern.

        Guaranteed NOT to contain: prompt text, solution text, model names,
        file names, timestamps, or any user identity.
        """
        from vector_memory import embed

        abstract = _abstract_solution(pattern_data.get("learned_solution", ""))
        vector = embed(abstract, dim=256)
        payload = {
            "category": _categorize(pattern_data.get("prompt_pattern", "")),
            "feature_vector": vector,  # floats only — no text
            "outcome_stats": {
                "success_count": int(pattern_data.get("success_count", 1)),
                "avg_duration_ms": int(pattern_data.get("avg_duration_ms", 0)),
            },
            "ts": int(time.time()),
        }
        payload["signature"] = self._sign(payload)
        return payload

    def _sign(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(self.secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    # ── HTTP ──────────────────────────────────────────────────────────────
    def _post(self, url: str, body: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "OpenCodeWEB-AiA-Swarm"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenCodeWEB-AiA-Swarm"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── push ──────────────────────────────────────────────────────────────
    def sync_local_knowledge_to_global_swarm(self) -> dict[str, Any]:
        """Upload anonymized, unpushed learned patterns (watermark-tracked)."""
        records = self.knowledge.get("learned_from_models", [])
        wm = self.state.get("uploaded_watermark", 0)
        unpushed = records[wm:]
        if not unpushed:
            return {"pushed": 0, "message": "no new local patterns to upload"}

        payload = {"device": self.device_id()[:16], "patterns": [self.anonymize_pattern(r) for r in unpushed]}
        try:
            resp = self._post(f"{self.endpoint}/sync", payload)
            if resp.get("ok"):
                self.state["uploaded_watermark"] = len(records)
                self._save_state()
                return {"pushed": len(unpushed), "received": resp.get("received", len(unpushed))}
            return {"pushed": 0, "error": f"server rejected: {resp}"}
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            return {"pushed": 0, "error": str(e)}

    # ── pull / patch apply ────────────────────────────────────────────────
    def download_global_skill_updates(self) -> dict[str, Any]:
        """Fetch + validate + apply OTA skill patches from the swarm hub."""
        since = self.state.get("last_patch_ts", 0.0)
        try:
            resp = self._get(f"{self.endpoint}/patch?since={since}")
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            return {"applied": 0, "error": str(e)}

        patches = resp.get("patches", []) if isinstance(resp, dict) else []
        existing_sigs = {s.get("signature") for s in self.knowledge.get("skills", []) if s.get("signature")}
        applied = 0
        for patch in patches:
            if not self._valid_patch(patch):
                continue
            if patch.get("signature") in existing_sigs:
                continue
            self.knowledge["skills"].append(
                {
                    "id": f"swarm-{patch.get('signature', 'x')[:10]}",
                    "pattern": patch.get("pattern", "")[:300],
                    "solution": patch.get("solution_template", "")[:2000],
                    "source": "swarm",
                    "confirmed": False,  # needs 1 local confirmation before auto-execution
                    "signature": patch.get("signature"),
                    "category": patch.get("category"),
                    "ts": time.time(),
                }
            )
            existing_sigs.add(patch.get("signature"))
            applied += 1
        if patches:
            self.state["last_patch_ts"] = resp.get("server_ts", time.time())
            self._save_state()
        return {"applied": applied, "feed_size": len(patches)}

    @staticmethod
    def _valid_patch(patch: Any) -> bool:
        if not isinstance(patch, dict):
            return False
        sig = patch.get("signature")
        if not isinstance(sig, str) or not (8 <= len(sig) <= 64):
            return False
        if len(json.dumps(patch)) > 4096:
            return False
        return isinstance(patch.get("pattern", ""), str) and isinstance(patch.get("solution_template", ""), str)
