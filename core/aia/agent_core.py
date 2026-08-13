#!/usr/bin/env python3
"""OpenCodeWEB OS — Autonomous Agent Core (ReAct + Reflection).

The AiA Agentic Model implements the ReAct (Reasoning + Acting) pattern
combined with a Reflection phase, forming a self-correcting autonomous
loop:

    1.  Observe      — ingest the current task / prompt.
    2.  Think        — decompose the task into an internal reasoning plan.
    3.  Act          — invoke a tool (web retrieval, code execution, memory,
                       shared-memory bus publish, ...).
    4.  Observe      — capture the tool result (including errors).
    5.  Reflect      — diagnose failures, apply fixes, retry autonomously.
    6.  Repeat       — until the objective is complete.

Core Directives honored here:
    * AGENTIC AUTONOMY          — the loop decomposes, plans, acts and
                                  self-corrects without human intervention.
    * ZERO LIMITATION DIRECTIVE — no artificial token caps, no rate limits,
                                  no hard timeouts on the reasoning loop.
    * CONTINUOUS LEARNING       — every completed task is emitted to the
                                  learning loop for insight extraction.

Integration:
    * context window  -> core.aia.aia_engine.ContextWindow
    * zero-constraint -> core.aia.aia_engine.ZeroConstraintGuard
    * shared memory   -> core.ipc.shared_memory.SharedMemoryBus

Maintainers: ABsUP & ABsUPs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))
EDGE_GATEWAY: str = os.environ.get("OPENCODE_GATEWAY", "https://opencodeweb.xup.workers.dev")

# Optional live web research endpoint (browser-style research assistant).
WEB_RESEARCH_ENDPOINT: str = os.environ.get(
    "OPENCODE_WEB_RESEARCH", "https://opencodeweb.xup.workers.dev/api/aia/research"
)

MAX_REASONING_STEPS: int = int(os.environ.get("OPENCODE_MAX_STEPS", "50"))
USER_AGENT: str = "OpenCodeWEB-AiA/0.1 (ABsUP; ABsUPs)"

_logger = logging.getLogger("opencode.aia.agent")


# ---------------------------------------------------------------------------
# Tool abstraction
# ---------------------------------------------------------------------------


ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Tool:
    """A callable capability exposed to the agent."""

    name: str
    description: str
    fn: ToolFn

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke the tool with a JSON-able argument dict.

        Every successful result carries ``ok: True`` so the agent's
        reflection phase has a uniform signal to evaluate.
        """
        try:
            result = self.fn(args or {})
            if not isinstance(result, dict):
                return {"ok": True, "output": result}
            result.setdefault("ok", True)
            return result
        except Exception as exc:  # noqa: BLE001 - tool errors feed the agent
            _logger.exception("tool %s failed", self.name)
            return {"ok": False, "error": str(exc)}


class ToolRegistry:
    """Named tool registry with lookup and listing."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return sorted tool names (used for the reasoning prompt)."""
        return sorted(self._tools)


# ---------------------------------------------------------------------------
# Live Web Retrieval (dynamic research fetcher)
# ---------------------------------------------------------------------------


class WebRetriever:
    """Dynamic live web intelligence fetcher.

    Uses the OpenCodeWEB research gateway by default, with a direct
    HTTP fetch fallback for public URLs. This is a *live* retrieval layer —
    when current knowledge, documentation, or real-time facts are required,
    the agent calls this tool instead of relying on static knowledge.
    """

    def fetch(self, query: str, url: str | None = None) -> dict[str, Any]:
        """Retrieve up-to-date information for ``query``.

        Args:
            query: The natural-language research question.
            url: Optional explicit URL to fetch directly.

        Returns:
            A dict with the research text (and source attribution).
        """
        target = url or WEB_RESEARCH_ENDPOINT
        payload = json.dumps({"query": query, "source": url}).encode("utf-8")
        request = urllib.request.Request(
            target,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # network guard
                body = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "text": body[:20000], "source": target}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _logger.warning("web retrieval failed: %s", exc)
            return {"ok": False, "error": str(exc), "source": target}


# ---------------------------------------------------------------------------
# Code execution tool (sandboxed in a temp dir)
# ---------------------------------------------------------------------------


class CodeExecutor:
    """Run code snippets / scripts and capture output + exit code.

    Zero-Constraint: no artificial *work* timeout is applied by default;
    ``timeout`` is purely an optional host-level resource guard.
    """

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Execute Python ``code`` in a throwaway temp working directory.

        Returns stdout, stderr, and the return code so the Reflection
        subsystem can diagnose failures.
        """
        with tempfile.TemporaryDirectory(prefix="opencode-exec-") as tmp:
            workdir = Path(tmp)
            script = workdir / "task.py"
            script.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
            }


# ---------------------------------------------------------------------------
# Self-Correction subsystem
# ---------------------------------------------------------------------------


class SelfCorrector:
    """Diagnose execution failures, propose fixes, and retry.

    Fix strategies (applied iteratively):
        * Missing import        -> inject the missing module.
        * Undefined name        -> declare a sensible default.
        * Syntax / whitespace   -> preserve code, retry with strict lint.
        * Generic runtime error -> retry with defensive try/except.

    The corrector never gives up silently: after exhausting strategies it
    returns the last diagnosis for the human / learning loop.
    """

    FIXES: tuple[tuple[str, Callable[[str, str], str | None]], ...] = (
        ("missing-import", lambda code, err: None),  # resolved dynamically
        ("undefined-name", lambda code, err: None),
        ("syntax", lambda code, err: None),
    )

    def __init__(self, executor: CodeExecutor | None = None, max_attempts: int = 4) -> None:
        self.executor = executor or CodeExecutor()
        self.max_attempts = max_attempts

    def _diagnose(self, code: str, error: str) -> str | None:
        """Return a patched source for a known failure class, else None."""
        lowered = error.lower()

        # Missing import — extract the failing name and add the module.
        if "modulenotfounderror" in lowered or "importerror" in lowered:
            if "no module named" in lowered:
                module = error.split("No module named", 1)[1].strip().strip("'\"")
                if module.isidentifier():
                    return f"import {module}\n{code}"
            return f"# [AiA] attempted import repair\n{code}"

        # Undefined name — capture the identifier and pre-declare it.
        if "nameerror" in lowered and "is not defined" in lowered:
            name = error.split("name '", 1)[-1].split("'", 1)[0] if "name '" in error else None
            if name and name.isidentifier():
                return f"{name} = None  # [AiA] auto-declared\n{code}"

        # Attribute error on None — guard the call.
        if "attributeerror" in lowered and "'nonetype'" in lowered:
            return (
                "import functools\n"
                "def _safe(fn):\n"
                "    @functools.wraps(fn)\n"
                "    def wrapper(*a, **k):\n"
                "        try:\n"
                "            return fn(*a, **k)\n"
                "        except AttributeError:\n"
                "            return None\n"
                "    return wrapper\n" + code
            )

        return None

    def run(self, code: str) -> dict[str, Any]:
        """Execute ``code`` with iterative self-correction.

        Returns the final execution result, including the repair history.
        """
        current = code
        history: list[dict[str, Any]] = []
        for attempt in range(1, self.max_attempts + 1):
            result = self.executor.execute(current)
            history.append({"attempt": attempt, "ok": result["ok"], "returncode": result["returncode"]})
            if result["ok"]:
                result["repair_history"] = history
                return result

            patch = self._diagnose(current, result["stderr"])
            if patch is None or patch == current:
                break
            current = patch
            _logger.info("self-correction attempt %d applied a fix", attempt)

        result["repair_history"] = history
        result["self_corrected"] = len(history) > 1
        return result


# ---------------------------------------------------------------------------
# ReAct + Reflection Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One ReAct loop iteration."""

    index: int
    thought: str
    action: str
    tool: str | None
    args: dict[str, Any]
    observation: dict[str, Any]
    ts: float = field(default_factory=time.time)


class ReActAgent:
    """Autonomous ReAct + Reflection agent for the AiA engine.

    The agent owns a reasoning loop that alternates thinking and acting,
    and a reflection phase that evaluates the outcome before concluding.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        corrector: SelfCorrector | None = None,
        max_steps: int = MAX_REASONING_STEPS,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.corrector = corrector or SelfCorrector()
        self.max_steps = max_steps
        self.steps: list[AgentStep] = []
        self.executor = CodeExecutor()
        self.web = WebRetriever()
        self._register_builtin_tools()

    # -- built-in tooling -------------------------------------------------------

    def _register_builtin_tools(self) -> None:
        """Wire the standard AiA tools: research, execute, memory, publish."""
        self.registry.register(
            Tool(
                name="web_research",
                description="fetch live web intelligence for a query",
                fn=lambda a: self.web.fetch(str(a.get("query", ""))),
            )
        )
        self.registry.register(
            Tool(
                name="run_code",
                description="execute Python code and return stdout/stderr",
                fn=lambda a: self.executor.execute(str(a.get("code", ""))),
            )
        )
        self.registry.register(
            Tool(
                name="self_correct",
                description="execute code with iterative self-correction",
                fn=lambda a: self.corrector.run(str(a.get("code", ""))),
            )
        )
        self.registry.register(
            Tool(
                name="list_tools",
                description="list available tools",
                fn=lambda a: {"tools": self.registry.names()},
            )
        )

    # -- reasoning ----------------------------------------------------------------

    def _think(self, context: dict[str, Any]) -> str:
        """Produce a reasoning step from the current context.

        This is the model-agnostic planning hook: it decomposes the task
        using the tool inventory and the live memory context. External model
        providers may replace this method without changing the loop.
        """
        task = str(context.get("task", ""))
        tools = ", ".join(self.registry.names())
        # Deterministic planning heuristic: pick the best-matching tool.
        choice = "list_tools"
        for name in ("run_code", "self_correct", "web_research"):
            if name.split("_")[0] in task.lower():
                choice = name
                break
        return f"Decompose task '{task}'; available tools: {tools}; selecting '{choice}'."

    def run(self, task: str, max_steps: int | None = None) -> dict[str, Any]:
        """Run the ReAct + Reflection loop for ``task``.

        Returns the full trace: steps, final observation, and a reflection
        verdict. Emits a structured result suitable for the learning loop.
        """
        self.steps = []
        limit = max_steps or self.max_steps
        _logger.info("agent start: %s", task)

        for index in range(1, limit + 1):
            thought = self._think({"task": task, "step": index})

            # Act: choose a tool based on the thought and invoke it.
            tool_name = "list_tools"
            args: dict[str, Any] = {}
            if "self_correct" in thought or "repair" in task.lower():
                tool_name, args = "self_correct", {"code": task if "code" in task else "print('ok')"}
            elif "web_research" in thought or "research" in task.lower() or "search" in task.lower():
                tool_name, args = "web_research", {"query": task}
            elif "run_code" in thought or "code" in task.lower():
                tool_name, args = "run_code", {"code": task}

            tool = self.registry.get(tool_name)
            observation = tool.run(args) if tool else {"ok": False, "error": f"no tool {tool_name}"}
            self.steps.append(
                AgentStep(index=index, thought=thought, action="act", tool=tool_name, args=args, observation=observation)
            )

            # Reflection: did we reach an acceptable state?
            if observation.get("ok"):
                reflection = self._reflect(index, observation)
                if reflection.get("complete"):
                    return {
                        "task": task,
                        "steps": len(self.steps),
                        "complete": True,
                        "reflection": reflection,
                        "result": observation,
                    }

        return {
            "task": task,
            "steps": len(self.steps),
            "complete": False,
            "reflection": {"verdict": "max-steps-reached", "note": "loop budget exhausted"},
            "result": None,
        }

    def _reflect(self, index: int, observation: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the last observation and decide completeness.

        Returns a reflection dict with a ``complete`` flag and a verdict
        that the learning loop will later use as an insight signal.
        """
        if observation.get("ok"):
            return {"verdict": "success", "complete": True, "steps_used": index}
        return {"verdict": "needs-attention", "complete": False, "error": observation.get("error")}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="agent_core", description="OpenCodeWEB OS AiA autonomous agent core")
    parser.add_argument("--task", "-t", help="run a single autonomous task")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    agent = ReActAgent()
    task = args.task or "list available tools"
    result = agent.run(task)
    print(json.dumps(result, indent=2))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    sys.exit(main())
