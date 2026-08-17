"""Tests for the vector memory (sliding window + compression + recall)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_memory import VectorMemory, cosine, embed  # noqa: E402


def test_ingest_and_recall_ranks_relevant_higher(tmp_path):
    vm = VectorMemory(tmp_path / "mem.json", window_size=10)
    vm.ingest("user", "flutter glassmorphism container decoration blur")
    vm.ingest("user", "flutter navigation hero animation transitions")
    hits = vm.recall("flutter glassmorphism widget", top_k=2)
    assert len(hits) == 2
    assert hits[0]["kind"] == "entry"
    assert "glassmorphism" in hits[0]["text"]  # relevant entry ranks first
    assert hits[0]["score"] > hits[1]["score"]


def test_compression_after_window_overflow(tmp_path):
    vm = VectorMemory(tmp_path / "mem.json", window_size=10, compress_below=4)
    for i in range(40):
        vm.ingest("user", f"entry number {i} about flutter widgets and state management")
    assert len(vm.entries) <= 10
    assert len(vm.summaries) >= 1
    # compressed summary is still recallable
    hits = vm.recall("flutter state management")
    assert any(h["kind"] == "summary" for h in hits)


def test_hierarchical_summary_cap(tmp_path):
    vm = VectorMemory(tmp_path / "mem.json", window_size=10, compress_below=4, summary_cap=16)
    for i in range(300):
        vm.ingest("user", f"session {i} discusses react hooks useEffect useMemo patterns")
    assert len(vm.summaries) <= 16  # super-summaries keep the cap


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "mem.json"
    vm = VectorMemory(path, window_size=6)
    for i in range(20):
        vm.ingest("user", f"async task {i} with promises and fetch")
    vm2 = VectorMemory(path)
    assert vm2.stats() == vm.stats()
    hits = vm2.recall("promises async fetch")
    assert hits


def test_embed_deterministic_and_normalized():
    a = embed("flutter glassmorphism")
    b = embed("flutter glassmorphism")
    c = embed("rust ownership borrowing")
    assert a == b
    assert a != c
    import math

    assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-3
    assert cosine(a, b) > 0.99
    assert cosine(a, c) < 0.5
