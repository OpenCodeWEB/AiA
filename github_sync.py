"""GitHub open-source learning sync — trending repos → abstracted patterns.

Daily cadence: GitHub search API (`created:>7d`, sorted by stars) → top repos →
README fetch (raw.githubusercontent.com) → fenced code-block pattern extraction.
Resilient: timeouts, graceful degradation, rate-limit safe (unauthenticated
search quota: 10 req/min — one call per run).
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

SEARCH_API = "https://api.github.com/search/repositories"
RAW_API = "https://raw.githubusercontent.com"
MAX_README_BYTES = 512_000
MAX_PATTERNS_PER_REPO = 3


class GitHubSync:
    def __init__(self, brain_dir: str | Path, knowledge: dict[str, Any], token: Optional[str] = None) -> None:
        self.brain_dir = Path(brain_dir)
        self.knowledge = knowledge
        self.token = token

    # ── HTTP (monkeypatchable in tests) ───────────────────────────────────
    def _http_json(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenCodeWEB-AiA"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_text(self, url: str) -> Optional[str]:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenCodeWEB-AiA"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(MAX_README_BYTES + 1)
            return data[:MAX_README_BYTES].decode("utf-8", errors="replace")

    # ── fetching ──────────────────────────────────────────────────────────
    def fetch_trending(self, days: int = 7, per_page: int = 10) -> list[dict[str, Any]]:
        since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        url = f"{SEARCH_API}?q=created:%3E{since}&sort=stars&order=desc&per_page={per_page}"
        data = self._http_json(url)
        repos = []
        for item in data.get("items", []):
            repos.append(
                {
                    "full_name": item.get("full_name", ""),
                    "html_url": item.get("html_url", ""),
                    "stargazers_count": item.get("stargazers_count", 0),
                    "language": item.get("language"),
                    "description": (item.get("description") or "")[:300],
                }
            )
        return repos

    def fetch_readme(self, full_name: str) -> Optional[str]:
        for branch in ("HEAD", "main", "master"):
            try:
                return self._http_text(f"{RAW_API}/{full_name}/{branch}/README.md")
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
        return None

    @staticmethod
    def extract_patterns(readme: str) -> list[str]:
        """Fenced code blocks (``` … ```) — capped count and length."""
        patterns: list[str] = []
        in_block = False
        buf: list[str] = []
        for line in readme.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_block:
                    text = "\n".join(buf).strip()
                    if text and len(text) <= 2000:
                        patterns.append(text)
                    buf = []
                    in_block = False
                    if len(patterns) >= MAX_PATTERNS_PER_REPO:
                        break
                else:
                    in_block = True
                continue
            if in_block:
                buf.append(line)
        return patterns

    # ── run ───────────────────────────────────────────────────────────────
    def run(self, days: int = 7, per_page: int = 10, readme_repos: int = 3) -> dict[str, Any]:
        result: dict[str, Any] = {"synced_at": time.time(), "repos": [], "patterns_added": 0, "errors": []}
        try:
            repos = self.fetch_trending(days=days, per_page=per_page)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            result["errors"].append(f"trending fetch failed: {e}")
            return result

        existing = {r.get("full_name") for r in self.knowledge.get("github_trends", [])}
        for repo in repos[:readme_repos]:
            try:
                readme = self.fetch_readme(repo["full_name"])
                patterns = self.extract_patterns(readme) if readme else []
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                result["errors"].append(f"{repo['full_name']}: {e}")
                patterns = []
            repo["patterns"] = patterns
            result["patterns_added"] += len(patterns)
            if repo["full_name"] not in existing:
                repo["learned"] = time.time()
                self.knowledge["github_trends"].append(repo)
                result["repos"].append(repo["full_name"])
        return result
