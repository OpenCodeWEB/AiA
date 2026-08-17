"""Tests for the GitHub open-source sync (HTTP mocked — no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from github_sync import GitHubSync  # noqa: E402


def _sync(tmp_path):
    knowledge = {"github_trends": []}
    return GitHubSync(tmp_path, knowledge), knowledge


def test_run_with_mocked_http(tmp_path):
    sync, knowledge = _sync(tmp_path)
    sync._http_json = lambda url: {
        "items": [
            {
                "full_name": "opencode/awesome",
                "html_url": "https://github.com/opencode/awesome",
                "stargazers_count": 12345,
                "language": "Python",
                "description": "awesome things",
            }
        ]
    }
    sync.fetch_readme = lambda full_name: "# Awesome\n```python\nprint('hi')\n```\nmore text\n```js\nx = 1\n```"
    result = sync.run(readme_repos=1)
    assert result["repos"] == ["opencode/awesome"]
    assert result["patterns_added"] == 2
    assert knowledge["github_trends"][0]["stargazers_count"] == 12345
    assert knowledge["github_trends"][0]["patterns"] == ["print('hi')", "x = 1"]


def test_run_handles_fetch_failure(tmp_path):
    sync, knowledge = _sync(tmp_path)

    def boom(url):
        import urllib.error

        raise urllib.error.URLError("network down")

    sync._http_json = boom
    result = sync.run()
    assert result["errors"]
    assert result["repos"] == []


def test_extract_patterns_caps_count():
    readme = "\n".join(f"```lang{i}\ncode{i}\n```" for i in range(10))
    patterns = GitHubSync.extract_patterns(readme)
    assert len(patterns) == 3  # MAX_PATTERNS_PER_REPO


def test_dedupe_existing_repo(tmp_path):
    sync, knowledge = _sync(tmp_path)
    knowledge["github_trends"] = [{"full_name": "opencode/awesome"}]
    sync._http_json = lambda url: {"items": [{"full_name": "opencode/awesome", "html_url": "", "stargazers_count": 1}]}
    sync.fetch_readme = lambda full_name: None
    result = sync.run(readme_repos=1)
    assert result["repos"] == []  # already known
    assert len(knowledge["github_trends"]) == 1
