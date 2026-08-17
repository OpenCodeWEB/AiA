#!/usr/bin/env python3
# =====================================================================
# OpenCodeWEB OS — AiA Unlimited Self-Evolving Master Intelligence
# Source Repository: https://github.com/OpenCodeWEB/AiA
# Canonical path: /opt/opencode/lib/aia/aia_core_engine.py
# Brain: /opt/opencode/aia_brain (env AIA_BRAIN_DIR overrides)
# =====================================================================
"""AiA Master Engine — Supervisor-Observer-Executor core.

process_task flow:
  1. recall relevant history from vector memory (unlimited-context)
  2. evaluate: can AiA execute natively? (confirmed skill pattern match)
  3. YES → native execution
  4. NO  → delegate to the model swarm (Gemini → opencode → mock), observe,
          then assimilate the outcome (learn_from_execution)
  5. learn user preferences + ingest conversation into memory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows consoles default to cp1252 — emoji/log markers need UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from executors.registry import ExecutorRegistry, default_executors  # noqa: E402
from learning_loop import LearningHub  # noqa: E402
from vector_memory import VectorMemory  # noqa: E402

DEFAULT_BRAIN_DIR = Path(os.environ.get("AIA_BRAIN_DIR", str(Path.home() / "opencode" / "aia_brain")))

NATIVE_MATCH_THRESHOLD = 0.55  # token-overlap similarity for a skill pattern match


def token_overlap(a: str, b: str) -> float:
    """Jaccard-style token overlap used for cheap pattern matching."""
    import re

    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class AiAMasterEngine:
    def __init__(
        self,
        brain_dir: str | Path | None = None,
        memory: Optional[VectorMemory] = None,
        hub: Optional[LearningHub] = None,
        executors: Optional[list[Any]] = None,
        verbose: bool = True,
    ) -> None:
        self.brain_dir = Path(brain_dir or DEFAULT_BRAIN_DIR)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.knowledge_path = self.brain_dir / "learned_patterns.json"
        self.profile_path = self.brain_dir / "user_profile.json"
        self.knowledge: dict[str, Any] = self._load_json(self.knowledge_path, self._fresh_knowledge())
        self.profile: dict[str, Any] = self._load_json(self.profile_path, {"languages": {}, "styles": {}, "interactions": 0})

        self.memory = memory or VectorMemory(self.brain_dir / "infinite_memory.json")
        self.hub = hub or LearningHub(self.brain_dir)
        self.registry = ExecutorRegistry(list(executors) if executors else default_executors())

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _fresh_knowledge() -> dict[str, Any]:
        return {"skills": [], "learned_from_models": [], "github_trends": [], "anti_patterns": []}

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return default

    def save_knowledge(self) -> None:
        tmp = self.knowledge_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.knowledge_path)

    def save_profile(self) -> None:
        self.profile_path.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"🧠 [AiA] {msg}")

    # ── task pipeline ─────────────────────────────────────────────────────
    def process_task(self, user_prompt: str, context_data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self._log(f"Analyzing prompt with unlimited context: '{user_prompt[:80]}...'")
        context = dict(context_data or {})
        context["recall"] = self.memory.recall(user_prompt, top_k=3)

        if self.evaluate_native_capability(user_prompt):
            self._log("⚡ Executing natively using AiA's internal skill library...")
            output, skill = self.execute_native(user_prompt)
            source = "aia-native"
            self.hub.record_execution(source, user_prompt, output, success=True)
        else:
            self._log("🔀 Delegating to the model swarm (Gemini → opencode → mock)...")
            output, model = self.delegate_and_observe(user_prompt)
            source = model
            promoted = self.learn_from_execution(model, user_prompt, output)
            if promoted:
                self._log(f"📖 Pattern from [{model}] promoted to skill {promoted['id']}")

        self.learn_from_user_pattern(user_prompt, output)
        self.memory.ingest("user", user_prompt)
        self.memory.ingest("aia", output)
        self.save_knowledge()
        self.save_profile()
        return {"ok": True, "mode": "native" if source == "aia-native" else "delegated", "source": source, "output": output}

    # ── native capability ─────────────────────────────────────────────────
    def evaluate_native_capability(self, prompt: str) -> bool:
        """True when a *confirmed* skill pattern matches closely enough."""
        for skill in self.knowledge.get("skills", []):
            if not skill.get("confirmed", True):
                continue
            if token_overlap(skill.get("pattern", ""), prompt) >= NATIVE_MATCH_THRESHOLD:
                return True
        return False

    def execute_native(self, prompt: str) -> tuple[str, Optional[dict[str, Any]]]:
        best: Optional[dict[str, Any]] = None
        best_score = NATIVE_MATCH_THRESHOLD
        for skill in self.knowledge.get("skills", []):
            score = token_overlap(skill.get("pattern", ""), prompt)
            if score >= best_score:
                best, best_score = skill, score
        if best:
            best["usage_count"] = best.get("usage_count", 0) + 1
            self.save_knowledge()
            return best.get("solution", "[aia-native] no solution stored"), best
        return f"[aia-native] builtin handler for: {prompt[:120]}", None

    # ── delegation ────────────────────────────────────────────────────────
    def delegate_and_observe(self, prompt: str) -> tuple[str, str]:
        ex = self.registry.pick(prompt)
        if ex is None:
            raise RuntimeError("no executor available")
        self._log(f"📡 Delegating to [{ex.name}]...")
        output, meta = ex.run(prompt)
        self.hub.record_execution(ex.name, prompt, output, success=True, duration_ms=meta.get("duration_ms"))
        return output, ex.name

    def learn_from_execution(self, model: str, prompt: str, output: str, success: bool = True) -> Optional[dict[str, Any]]:
        # Hub counting must ALWAYS happen (dedupe only guards the knowledge list)
        promoted = self.hub.learn_from_execution(model, prompt, output, success=success)

        sig = self._sig(prompt + output)
        if not any(p.get("sig") == sig for p in self.knowledge.get("learned_from_models", [])):
            self.knowledge["learned_from_models"].append(
                {
                    "source_model": model,
                    "prompt_pattern": prompt[:300],
                    "learned_solution": output[:2000],
                    "success": success,
                    "sig": sig,
                    "ts": time.time(),
                }
            )

        if promoted:
            promoted["confirmed"] = True  # locally promoted patterns are trusted
            self.knowledge["skills"].append(promoted)
            self.save_knowledge()
        if not success:
            self.knowledge["anti_patterns"].append(self.hub.anti_pattern(prompt, output))
            self.save_knowledge()
        return promoted

    def confirm_swarm_skill(self, prompt: str, output: str) -> bool:
        """Confirm a swarm/unconfirmed skill after a successful local match."""
        for skill in self.knowledge.get("skills", []):
            if not skill.get("confirmed") and token_overlap(skill.get("pattern", ""), prompt) >= NATIVE_MATCH_THRESHOLD:
                skill["confirmed"] = True
                self.save_knowledge()
                return True
        return False

    # ── user adaptation ───────────────────────────────────────────────────
    def learn_from_user_pattern(self, prompt: str, output: str) -> None:
        self.hub.update_user_profile(self.profile, prompt, output)

    # ── GitHub open-source sync ───────────────────────────────────────────
    def sync_github_open_source_trends(self, days: int = 7, per_page: int = 10) -> dict[str, Any]:
        from github_sync import GitHubSync

        self._log("🌐 Syncing GitHub trending open-source repositories...")
        sync = GitHubSync(brain_dir=self.brain_dir, knowledge=self.knowledge)
        result = sync.run(days=days, per_page=per_page)
        self.save_knowledge()
        return result

    # ── Gemini ────────────────────────────────────────────────────────────
    def connect_gemini_google_account(self, session_cookie: Optional[str] = None) -> dict[str, Any]:
        from gemini_bridge import GeminiBridge

        return GeminiBridge().connect(session_cookie)

    # ── status ────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        from federated_learning_sync import AiAFederatedSync

        swarm = AiAFederatedSync(brain_dir=self.brain_dir, knowledge=self.knowledge)
        return {
            "ok": True,
            "version": "0.1.0",
            "brain_dir": str(self.brain_dir),
            "memory": self.memory.stats(),
            "knowledge": {k: len(v) for k, v in self.knowledge.items() if isinstance(v, list)},
            "profile_interactions": self.profile.get("interactions", 0),
            "executors": self.registry.describe(),
            "gemini": self.connect_gemini_google_account(),
            "swarm": {"device_id": swarm.device_id()[:12], "endpoint": swarm.endpoint},
        }

    @staticmethod
    def _sig(s: str) -> str:
        import hashlib

        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ── CLI ────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aia", description="AiA Master Intelligence Engine")
    parser.add_argument("--task", help="process a single task")
    parser.add_argument("--demo", action="store_true", help="run the PRD demo task")
    parser.add_argument("--sync-github", action="store_true", help="sync GitHub trending open-source patterns")
    parser.add_argument("--connect-gemini", action="store_true", help="check/connect Gemini via Google account (zero API key)")
    parser.add_argument("--status", action="store_true", help="print engine status")
    parser.add_argument("--swarm-push", action="store_true", help="push anonymized learnings to the global swarm")
    parser.add_argument("--swarm-pull", action="store_true", help="download + apply global skill patches")
    parser.add_argument("--executor", choices=["auto", "gemini", "opencode", "mock"], default="auto",
                        help="force a specific executor (default: first available)")
    parser.add_argument("--brain", help="override AIA_BRAIN_DIR")
    args = parser.parse_args(argv)

    executors = None
    if args.executor != "auto":
        from executors.gemini_executor import GeminiExecutor
        from executors.mock import MockExecutor
        from executors.opencode_executor import OpenCodeExecutor

        executors = {
            "gemini": [GeminiExecutor()],
            "opencode": [OpenCodeExecutor()],
            "mock": [MockExecutor()],
        }[args.executor]

    engine = AiAMasterEngine(brain_dir=args.brain, executors=executors)

    if args.status:
        print(json.dumps(engine.status(), indent=2))
        return 0
    if args.sync_github:
        print(json.dumps(engine.sync_github_open_source_trends(), ensure_ascii=False, indent=2))
        return 0
    if args.connect_gemini:
        print(json.dumps(engine.connect_gemini_google_account(), indent=2))
        return 0
    if args.swarm_push:
        from federated_learning_sync import AiAFederatedSync

        swarm = AiAFederatedSync(brain_dir=engine.brain_dir, knowledge=engine.knowledge)
        print(json.dumps(swarm.sync_local_knowledge_to_global_swarm(), ensure_ascii=False, indent=2))
        return 0
    if args.swarm_pull:
        from federated_learning_sync import AiAFederatedSync

        swarm = AiAFederatedSync(brain_dir=engine.brain_dir, knowledge=engine.knowledge)
        result = swarm.download_global_skill_updates()
        engine.save_knowledge()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    task = args.task or ("Write a Flutter UI component with custom glassmorphism" if args.demo else None)
    if not task:
        parser.print_help()
        return 1
    result = engine.process_task(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
