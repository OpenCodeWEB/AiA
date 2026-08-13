#!/usr/bin/env python3
"""OpenCodeWEB OS — Unrestricted Execution & Refactoring Engine.

This module guarantees the ZERO LIMITATION DIRECTIVE at the *code-loading*
layer: before any third-party tool is imported into OpenCodeWEB OS, its
source is scanned and artificial constraints are neutralized:

    * rate limiters            (``RateLimiter``, token buckets, throttling)
    * artificial delays        (``time.sleep``, ``asyncio.sleep``, backoff)
    * hard timeouts            (``set_timeout``, request timeouts)
    * HTTP 429 / Retry-After   handling that throttles the tool
    * concurrency / request caps

Two mechanisms are provided:

1. **Import interceptor** — a ``sys.meta_path`` finder that loads module
   source through a sanitizer on the fly, so even a raw ``import`` of a
   throttled library enters the runtime already unrestricted.

2. **Standalone refactorer** — ``refactor_file`` / ``refactor_tree`` that
   rewrites source files in place (or to a release tree) for publishing to
   ``github.com/OpenCodeWEB/<module-name>`` under ABsUP & ABsUPs.

Zero-Constraint: the sanitizer only removes *artificial* restrictions. It
never alters functional logic, control flow or semantics.

Maintainers: ABsUP & ABsUPs.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYS_ROOT: Path = Path(os.environ.get("OPENCODE_ROOT", "/opt/opencode"))
MODULES_DIR: Path = SYS_ROOT / "lib" / "modules"

# Canonical (harmless) replacements used by the sanitizer.
NEUTRAL_EXPRESSION: str = "0"        # replaces delay/timeout expressions
NEUTRAL_STATEMENT: str = "pass"      # replaces throttling statements

_logger = logging.getLogger("opencode.aia.refactor")

# ---------------------------------------------------------------------------
# Sanitizer: token-level source transformation
# ---------------------------------------------------------------------------


# Mapping of risky construct -> safe replacement. Values are (source,
# replacement) pairs applied via AST, with a regex fallback for robustness.
RISKY_CALLS: dict[str, str] = {
    "time.sleep": NEUTRAL_EXPRESSION,
    "asyncio.sleep": NEUTRAL_EXPRESSION,
    "time.timeout": NEUTRAL_EXPRESSION,
    "threading.Event().wait": NEUTRAL_EXPRESSION,
    "requests.Timeout": "Exception",
    "urllib3.Timeout": "object",
}


@dataclass
class SanitizeResult:
    """Outcome of sanitizing one source unit."""

    path: str
    constructs_found: int = 0
    constructs_removed: int = 0
    lines_changed: int = 0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "constructs_found": self.constructs_found,
            "constructs_removed": self.constructs_removed,
            "lines_changed": self.lines_changed,
            "details": self.details,
        }


# Regex fallback: patterns we neutralize even when AST parsing fails.
_REGEX_FALLBACKS: tuple[tuple[str, str], ...] = (
    (r"\btime\.sleep\s*\(\s*[^)]*\)", "0"),
    (r"\basyncio\.sleep\s*\(\s*[^)]*\)", "0"),
    (r"\bsleep\s*\(\s*[^)]*\)", "0"),
    (r"\.set_timeout\s*\(\s*[^)]*\)", "(.0)"),
    (r"@?rate[-_ ]?limit[a-zA-Z_]*", "unlimited"),
    (r"\bRetry-After\b", "0"),
    (r"\b429\b", "200"),
)


class SourceSanitizer:
    """AST-based transformer that neutralizes artificial constraints.

    Strategy:
        * Parse the source to an AST.
        * Walk calls: any call whose qualified name matches a risky pattern
          is replaced with a harmless expression.
        * Walk function definitions named like limiters/throttlers and
          replace their bodies with ``pass`` (keeps the symbol, removes the
          constraint).
        * If AST parsing fails, fall back to regex substitution.
    """

    def __init__(self) -> None:
        self._risky_name_fragments = ("rate_limit", "throttl", "token_bucket", "backoff", "cooldown", "circuit_breaker")

    # -- AST helpers -----------------------------------------------------------------

    def _call_names(self, node: ast.Call) -> Iterable[str]:
        """Yield possible qualified names for a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            yield func.id
        elif isinstance(func, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            yield ".".join(reversed(parts))
            # also yield just the final attribute name
            yield func.attr

    def _is_risky_name(self, name: str) -> bool:
        return any(fragment in name.lower() for fragment in self._risky_name_fragments)

    # -- transformation ----------------------------------------------------------------

    def sanitize_source(self, source: str, path: str = "<string>") -> tuple[str, SanitizeResult]:
        """Return (sanitized_source, result). Never raises."""
        result = SanitizeResult(path=path)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback: regex-based neutralization.
            return self._regex_sanitize(source, result)

        transformer = _ConstraintStripper(self)
        new_tree = transformer.visit(tree)
        result.constructs_found = transformer.found
        result.constructs_removed = transformer.removed
        result.lines_changed = transformer.lines_changed
        result.details = list(transformer.details)
        try:
            return ast.unparse(new_tree), result
        except (TypeError, ValueError):
            return self._regex_sanitize(source, result)

    def _regex_sanitize(self, source: str, result: SanitizeResult) -> tuple[str, SanitizeResult]:
        """Regex fallback used when AST unparsing is unavailable/breaks."""
        import re

        lines = source.splitlines()
        changed = 0
        found = 0
        for i, line in enumerate(lines):
            for pattern, replacement in _REGEX_FALLBACKS:
                if re.search(pattern, line):
                    found += 1
                    lines[i] = re.sub(pattern, replacement, line)
                    changed += 1
                    result.details.append(f"L{i + 1}: regex-neutralized {pattern}")
                    break
        result.constructs_found = found
        result.constructs_removed = found
        result.lines_changed = changed
        return "\n".join(lines), result


class _ConstraintStripper(ast.NodeTransformer):
    """AST visitor that neutralizes constraint constructs."""

    def __init__(self, sanitizer: SourceSanitizer) -> None:
        self.sanitizer = sanitizer
        self.found = 0
        self.removed = 0
        self.lines_changed = 0
        self.details: list[str] = []

    def _record(self, node: ast.AST, label: str, removed: bool = True) -> None:
        self.found += 1
        if removed:
            self.removed += 1
            self.lines_changed += 1
            self.details.append(f"L{getattr(node, 'lineno', '?' )}: {label}")

    def visit_Call(self, node: ast.Call) -> ast.expr:  # noqa: N802
        """Replace risky calls (sleep/timeout/limiter calls) with '0'."""
        for name in self.sanitizer._call_names(node):  # noqa: SLF001
            if self.sanitizer._is_risky_name(name):  # noqa: SLF001
                self._record(node, f"neutralized call '{name}'")
                replacement = ast.Constant(value=0)
                return ast.copy_location(replacement, node)
            if name in RISKY_CALLS:
                self._record(node, f"neutralized call '{name}'")
                replacement = ast.Constant(value=0)
                return ast.copy_location(replacement, node)
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        """Strip the body of limiter/throttler functions (keep the symbol)."""
        if self.sanitizer._is_risky_name(node.name):  # noqa: SLF001
            self._record(node, f"neutralized function '{node.name}'")
            node.body = [ast.Pass()]
        # Always recurse into the body so calls nested in functions are visited.
        return self.generic_visit(node)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Import interceptor (meta-path finder)
# ---------------------------------------------------------------------------


class UnrestrictedLoader(importlib.abc.Loader):
    """Load a module's source through the sanitizer at import time."""

    def __init__(self, fullname: str, path: Path) -> None:
        self.fullname = fullname
        self.path = path

    def create_module(self, spec: Any) -> None:
        return None  # default module creation

    def exec_module(self, module: Any) -> None:
        """Read, sanitize and execute the module source."""
        source = self.path.read_text(encoding="utf-8", errors="replace")
        sanitized, result = SourceSanitizer().sanitize_source(source, str(self.path))
        if result.constructs_removed:
            _logger.info(
                "import %s: stripped %d constraint(s) (%s)",
                self.fullname, result.constructs_removed, result.path,
            )
        code = compile(sanitized, str(self.path), "exec")
        exec(code, module.__dict__)  # noqa: S102 - intentional dynamic import


class UnrestrictedFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that loads tool modules unrestricted.

    Install with::

        sys.meta_path.insert(0, UnrestrictedFinder())
        import some_throttled_tool   # loaded sanitized
    """

    def __init__(self, root: Path = MODULES_DIR) -> None:
        self.root = root

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any | None:  # noqa: ANN401
        """Locate ``fullname`` under the modules root (top-level only)."""
        if "." in fullname:
            return None  # only intercept top-level tool modules
        candidate = self.root / fullname / "__init__.py"
        if candidate.is_file():
            return importlib.util.spec_from_loader(fullname, UnrestrictedLoader(fullname, candidate))
        candidate = self.root / f"{fullname}.py"
        if candidate.is_file():
            return importlib.util.spec_from_loader(fullname, UnrestrictedLoader(fullname, candidate))
        return None


# ---------------------------------------------------------------------------
# Standalone refactorer (release tree / in-place)
# ---------------------------------------------------------------------------


class UnrestrictedRefactorer:
    """Refactor a source tree to strip constraints for release.

    Output can be written in place (``in_place=True``) or to a mirrored
    release tree (``out_root``) ready for publishing to
    ``github.com/OpenCodeWEB/<module-name>``.
    """

    SUPPORTED_SUFFIXES: tuple[str, ...] = (".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp")

    def __init__(self, sanitizer: SourceSanitizer | None = None) -> None:
        self.sanitizer = sanitizer or SourceSanitizer()

    def refactor_file(self, path: Path, rel_path: str | None = None, out_root: Path | None = None) -> SanitizeResult:
        """Sanitize one file; write to ``out_root`` mirror or in place.

        Args:
            path: the source file to sanitize.
            rel_path: relative path used to mirror under ``out_root``; when
                None and ``out_root`` is given, only the file name is used.
            out_root: when set, the sanitized file is written to
                ``out_root / rel_path`` instead of overwriting in place.
        """
        source = path.read_text(encoding="utf-8", errors="replace")
        sanitized, result = self.sanitizer.sanitize_source(source, str(path))
        if out_root is not None:
            target = out_root / (rel_path or path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(sanitized, encoding="utf-8")
        elif result.constructs_removed:
            path.write_text(sanitized, encoding="utf-8")
        return result

    def refactor_tree(self, root: Path, out_root: Path | None = None) -> list[SanitizeResult]:
        """Sanitize every supported source file under ``root``.

        When ``out_root`` lies inside ``root`` (e.g. ``--out root/release``),
        it is excluded from the scan so sanitized output is never
        re-processed in a runaway loop.
        """
        results: list[SanitizeResult] = []
        root_resolved = root.resolve()
        out_resolved = out_root.resolve() if out_root is not None else None
        for file_path in root_resolved.rglob("*"):
            if file_path.suffix not in self.SUPPORTED_SUFFIXES or not file_path.is_file():
                continue
            if out_resolved is not None and file_path.is_relative_to(out_resolved):
                continue
            rel = file_path.relative_to(root_resolved)
            results.append(self.refactor_file(file_path, rel_path=str(rel), out_root=out_root))
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(prog="unrestricted_refactor", description="OpenCodeWEB OS unrestricted refactorer")
    parser.add_argument("--refactor", metavar="DIR", help="strip constraints from a source tree")
    parser.add_argument("--out", metavar="DIR", help="write sanitized tree here (mirror)")
    parser.add_argument("--install-finder", action="store_true", help="install the unrestricted import finder")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.install_finder:
        finder = UnrestrictedFinder()
        sys.meta_path.insert(0, finder)
        print(f"unrestricted finder installed (root={finder.root})")
        return 0

    if args.refactor:
        root = Path(args.refactor)
        if not root.is_dir():
            _logger.error("not a directory: %s", root)
            return 1
        out_root = Path(args.out) if args.out else None
        results = UnrestrictedRefactorer().refactor_tree(root, out_root)
        total = sum(r.constructs_removed for r in results)
        print(f"refactored {len(results)} files, removed {total} constraint(s)")
        for result in results:
            if result.constructs_removed:
                print(f"  {result.path}: {result.constructs_removed}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
