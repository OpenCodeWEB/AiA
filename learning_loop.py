"""AiA continuous learning hub — observations, skill promotion, anti-patterns, user profile.

This module records WHAT happened (observations) and computes promotion state;
the master engine owns the knowledge base (`learned_patterns.json`).

- `record_execution`   → JSONL observation log (rotating) + promotion counters
- `learn_from_execution` → dedupe + return a skill record when a pattern succeeds
  ≥ `promotion_threshold` times (3) across models → engine promotes it
- `anti_pattern`       → failed solutions are recorded, never promoted
- `update_user_profile` → per-interaction preference learning
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

ROTATE_BYTES = 5 * 1024 * 1024  # 5 MB

_LANG_HINTS: list[tuple[str, list[str]]] = [
    ("python", ["python", "pip ", "pandas", "fastapi", "django", "flask", ".py"]),
    ("typescript", ["typescript", "tsx", ".ts", "angular", "nestjs"]),
    ("javascript", ["javascript", ".js", "node", "react", "npm ", "yarn "]),
    ("flutter", ["flutter", "dart", "widget", "materialapp", "pubspec"]),
    ("rust", ["rust", "cargo", "trait", "ownership", ".rs"]),
    ("go", ["golang", "goroutine", "go mod", ".go"]),
    ("cpp", ["c++", "cpp", "cmake", "header", "std::"]),
]

_STYLE_HINTS: list[tuple[str, list[str]]] = [
    ("concise", ["concise", "short", "minimal", "one-liner", "tl;dr"]),
    ("detailed", ["detailed", "explain", "step by step", "thorough", "documented"]),
    ("testing", ["test", "tdd", "pytest", "spec", "coverage"]),
    ("performance", ["fast", "optimize", "perf", "efficient", "benchmark"]),
    ("security", ["secure", "auth", "encrypt", "sanitize", "owasp"]),
]


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class LearningHub:
    def __init__(self, brain_dir: str | Path, promotion_threshold: int = 3) -> None:
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.observation_path = self.brain_dir / "observation.log"
        self.promotion_path = self.brain_dir / "skill_promotion.json"
        self.threshold = promotion_threshold
        self.promotions: dict[str, dict[str, Any]] = {}
        self._load_promotions()

    # ── persistence ───────────────────────────────────────────────────────
    def _load_promotions(self) -> None:
        if self.promotion_path.exists():
            try:
                self.promotions = json.loads(self.promotion_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.promotions = {}

    def _save_promotions(self) -> None:
        self.promotion_path.write_text(json.dumps(self.promotions, indent=2), encoding="utf-8")

    def _rotate_observations(self) -> None:
        if self.observation_path.exists() and self.observation_path.stat().st_size > ROTATE_BYTES:
            rotated = self.observation_path.with_suffix(".log.1")
            if rotated.exists():
                rotated.unlink()
            self.observation_path.replace(rotated)

    # ── recording ─────────────────────────────────────────────────────────
    def record_execution(
        self,
        source_model: str,
        prompt: str,
        output: str,
        success: bool = True,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Append an observation line; update promotion counters; return the record."""
        self._rotate_observations()
        record = {
            "ts": time.time(),
            "source_model": source_model,
            "prompt": prompt,
            "output": output,
            "success": success,
            "duration_ms": duration_ms,
            "sig": _hash(prompt + output),
        }
        with open(self.observation_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if success:
            key = _hash(prompt)
            state = self.promotions.get(key, {"count": 0, "prompt": prompt[:300], "models": []})
            state["count"] += 1
            if source_model not in state["models"]:
                state["models"].append(source_model)
            self.promotions[key] = state
            self._save_promotions()
        return record

    # ── learning ──────────────────────────────────────────────────────────
    def learn_from_execution(self, source_model: str, prompt: str, output: str, success: bool = True) -> Optional[dict[str, Any]]:
        """Record an execution; return a promoted skill when threshold is reached."""
        self.record_execution(source_model, prompt, output, success=success)
        if not success:
            return None
        key = _hash(prompt)
        state = self.promotions.get(key)
        if not state or state["count"] < self.threshold:
            return None
        skill = {
            "id": f"skill-{key}",
            "pattern": state["prompt"],
            "solution": output[:2000],
            "source": f"promoted:{source_model}",
            "usage_count": 0,
            "ts": time.time(),
        }
        # reset counter so the same pattern is not promoted repeatedly
        state["count"] = 0
        self._save_promotions()
        return skill

    def anti_pattern(self, prompt: str, output: str, error: str = "") -> dict[str, Any]:
        """Failed executions are recorded as anti-patterns (never promoted)."""
        record = {
            "prompt_pattern": prompt[:300],
            "failed_solution": output[:1000],
            "error": error[:500],
            "ts": time.time(),
            "sig": _hash(prompt + output),
        }
        self.record_execution("anti_pattern", prompt, output, success=False)
        return record

    # ── user preferences ──────────────────────────────────────────────────
    def update_user_profile(self, profile: dict[str, Any], prompt: str, output: str) -> dict[str, Any]:
        """Increment preference counters derived from this interaction."""
        text = f"{prompt} {output}".lower()
        lang = profile.setdefault("languages", {})
        for name, hints in _LANG_HINTS:
            if any(h in text for h in hints):
                lang[name] = lang.get(name, 0) + 1
        style = profile.setdefault("styles", {})
        for name, hints in _STYLE_HINTS:
            if any(h in text for h in hints):
                style[name] = style.get(name, 0) + 1
        profile["interactions"] = profile.get("interactions", 0) + 1
        profile["last_ts"] = time.time()
        return profile
