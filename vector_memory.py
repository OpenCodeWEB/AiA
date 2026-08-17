"""AiA vector memory — unlimited-context sliding window with hierarchical compression.

Pure-Python hashing TF-IDF embeddings (zero external dependencies, no Ollama).
Recent turns stay verbatim; older content is compressed into summary vectors and
recalled by cosine similarity. Persisted to `infinite_memory.json` in the brain.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on", "with", "is", "are",
    "was", "were", "be", "been", "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "not", "no", "yes", "so", "if", "then", "else", "than",
    "as", "at", "by", "from", "up", "down", "out", "over", "under", "again", "further", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "just", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "am", "s", "t", "ll", "re", "ve", "use",
    "using", "used", "make", "make", "get", "set", "new", "like", "one", "two", "also", "way",
}


def tokenize(text: str) -> list[str]:
    """Lowercase token stream, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _hash_idx(token: str, dim: int = EMBED_DIM) -> int:
    """Deterministic token → dimension index (md5 hashing trick)."""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big") % dim


def embed(text: str, idf: Optional[dict[str, float]] = None, dim: int = EMBED_DIM) -> list[float]:
    """Hashing TF-IDF embedding of a text chunk (weighted bag of hashed tokens)."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    for t, tf in counts.items():
        w = 1.0 + math.log(tf)
        if idf:
            w *= idf.get(t, 1.0)
        vec[_hash_idx(t, dim)] += w
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorMemory:
    """Sliding window + hierarchical compression memory store."""

    def __init__(
        self,
        path: str | Path,
        window_size: int = 60,
        compress_below: int = 10,
        summary_cap: int = 200,
    ):
        self.path = Path(path)
        self.window_size = window_size
        self.compress_below = compress_below
        self.summary_cap = summary_cap
        self.entries: list[dict[str, Any]] = []      # verbatim recent turns
        self.summaries: list[dict[str, Any]] = []    # compressed chunks
        self.idf: dict[str, float] = {}
        self._load()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", [])
            self.summaries = data.get("summaries", [])
            self.idf = data.get("idf", {})
        except (json.JSONDecodeError, OSError):
            self.entries, self.summaries, self.idf = [], [], {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"entries": self.entries, "summaries": self.summaries, "idf": self.idf},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # ── ingest ────────────────────────────────────────────────────────────
    def ingest(self, role: str, text: str) -> None:
        entry = {"role": role, "text": text, "ts": time.time(), "vec": embed(text, self.idf)}
        self.entries.append(entry)
        self._refresh_idf()
        if len(self.entries) > self.window_size:
            self._compress()
        self.save()

    def _refresh_idf(self) -> None:
        docs = [e["text"] for e in self.entries] + [s["preview"] for s in self.summaries]
        if not docs:
            return
        df: dict[str, int] = {}
        for d in docs:
            for t in set(tokenize(d)):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}
        # re-embed verbatim entries with updated idf (cheap: window is small)
        for e in self.entries:
            e["vec"] = embed(e["text"], self.idf)

    def _compress(self) -> None:
        """Merge the oldest `compress_below` entries into one summary (hierarchical)."""
        old = self.entries[: self.compress_below]
        self.entries = self.entries[self.compress_below :]
        merged = " ".join(f"{e['role']}: {e['text']}" for e in old)
        summary = {
            "preview": merged[:2000],
            "vec": embed(merged, self.idf),
            "ts": old[0]["ts"],
            "n": len(old),
        }
        self.summaries.append(summary)
        if len(self.summaries) > self.summary_cap:
            self._merge_oldest_summaries()

    def _merge_oldest_summaries(self) -> None:
        """Hierarchical: oldest 8 summaries collapse into a super-summary."""
        old = self.summaries[:8]
        self.summaries = self.summaries[8:]
        merged = " ".join(s["preview"] for s in old)
        self.summaries.append(
            {
                "preview": merged[:2000],
                "vec": embed(merged, self.idf),
                "ts": old[0]["ts"],
                "n": sum(s["n"] for s in old),
            }
        )

    # ── recall ────────────────────────────────────────────────────────────
    def recall(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q = embed(query, self.idf)
        scored: list[tuple[float, dict[str, Any]]] = []
        for e in self.entries:
            scored.append((cosine(q, e["vec"]), {"kind": "entry", "role": e["role"], "text": e["text"], "ts": e["ts"]}))
        for s in self.summaries:
            scored.append((cosine(q, s["vec"]), {"kind": "summary", "preview": s["preview"], "ts": s["ts"], "n": s["n"]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(item, score=round(score, 4)) for score, item in scored[:top_k] if score > 0.01]

    def stats(self) -> dict[str, int]:
        return {"entries": len(self.entries), "summaries": len(self.summaries), "idf_terms": len(self.idf)}
