"""Tests for the neural companion modules (Gemini roadmap round 1).

Covers: AutoencoderCompressor, AnomalyDetector, CurriculumController,
HopfieldMemory, ElmanRNN + their neural_api endpoints.
"""

from __future__ import annotations

import json
import random

from core.neural import (
    AnomalyDetector,
    AutoencoderCompressor,
    CurriculumController,
    ElmanRNN,
    HopfieldMemory,
)
from core.neural.neural_api import NeuralHandler


# ---------------------------------------------------------------------- #
# AutoencoderCompressor
# ---------------------------------------------------------------------- #
def test_autoencoder_learns_to_reconstruct() -> None:
    ae = AutoencoderCompressor(input_size=4, bottleneck=2, hidden=10, lr=0.3, seed=11)
    samples = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]
    first = ae.reconstruction_error(samples[0])
    for _ in range(60):
        for s in samples:
            ae.train(s)
    last = ae.reconstruction_error(samples[0])
    assert last < first, f"reconstruction did not improve: {first} -> {last}"
    assert last < 0.2, f"reconstruction too poor: {last}"


def test_autoencoder_compresses_to_latent() -> None:
    ae = AutoencoderCompressor(input_size=4, bottleneck=2, hidden=8, seed=1)
    latent = ae.compress([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    assert all(len(z) == 2 for z in latent)
    assert ae.status()["compression_ratio"] == "2x"


def test_autoencoder_roundtrip() -> None:
    ae = AutoencoderCompressor(input_size=4, bottleneck=2, hidden=8, seed=5)
    ae.train([1.0, 0.0, 1.0, 0.0])
    data = ae.to_json()
    restored = AutoencoderCompressor.from_json(data)
    assert restored.w_in == ae.w_in
    assert restored.w_out == ae.w_out
    assert restored.samples_seen == ae.samples_seen


# ---------------------------------------------------------------------- #
# AnomalyDetector
# ---------------------------------------------------------------------- #
def test_anomaly_detector_flags_outliers() -> None:
    det = AnomalyDetector(input_size=2, bottleneck=1, hidden=6, seed=2)
    normal = [[0.5, 0.5], [0.52, 0.48], [0.48, 0.52], [0.51, 0.49], [0.49, 0.51], [0.5, 0.5]]
    for x in normal:
        det.train(x)
        det.check(x)
    # an outlier must be flagged after enough normal context
    verdict = det.check([9.0, 9.0])
    assert verdict["is_anomaly"] is True, f"outlier not detected: {verdict}"
    # a normal sample must stay calm
    calm = det.check([0.5, 0.5])
    assert calm["is_anomaly"] is False, f"normal sample flagged: {calm}"


def test_anomaly_detector_roundtrip() -> None:
    det = AnomalyDetector(input_size=2, bottleneck=1, hidden=6, seed=3)
    det.train([0.5, 0.5])
    det.check([0.6, 0.4])
    restored = AnomalyDetector.from_json(det.to_json())
    assert restored.mean == det.mean
    assert restored.checked == det.checked


# ---------------------------------------------------------------------- #
# CurriculumController
# ---------------------------------------------------------------------- #
def test_curriculum_advances_stages() -> None:
    ctrl = CurriculumController(num_stages=3, mastery_threshold=0.9, min_samples_per_stage=10)
    for _ in range(40):
        state = ctrl.update([0.5, 0.5], loss=0.01)  # easy, near-perfect
    assert ctrl.stage >= 2, f"curriculum never advanced: stage={ctrl.stage}"
    assert state["difficulty_level"] == "hard"


def test_curriculum_stays_on_easy_when_failing() -> None:
    ctrl = CurriculumController(num_stages=3, mastery_threshold=0.9, min_samples_per_stage=10)
    for _ in range(50):
        ctrl.update([0.5, 0.5], loss=0.9)  # failing hard
    assert ctrl.stage == 0, "curriculum advanced despite failing"


def test_curriculum_difficulty_bounded() -> None:
    for x in ([0.5, 0.5], [0.0, 0.0], [9.0, 9.0], []):
        d = CurriculumController.sample_difficulty(x)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------- #
# HopfieldMemory
# ---------------------------------------------------------------------- #
def _patterns(size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pats = []
    for _ in range(2):
        pat = [1 if rng.random() < 0.5 else 0 for _ in range(size)]
        pats.append(pat)
    return pats


def test_hopfield_recalls_clean_pattern() -> None:
    mem = HopfieldMemory(size=16)
    a, b = _patterns(16, 7)
    mem.store(a)
    mem.store(b)
    result = mem.recall(a, return_steps=False)
    assert result["recovered"] == a, f"clean recall failed: {result}"


def test_hopfield_reconstructs_corrupted() -> None:
    mem = HopfieldMemory(size=16)
    a, b = _patterns(16, 9)
    mem.store(a)
    mem.store(b)
    corrupted = list(a)
    corrupted[0] = 1 - corrupted[0]
    corrupted[5] = 1 - corrupted[5]
    corrupted[11] = 1 - corrupted[11]
    result = mem.recall(corrupted)
    assert result["recovered"] == a, f"corrupted recall failed: {result}"
    assert result["matched_stored"] is True


def test_hopfield_roundtrip() -> None:
    mem = HopfieldMemory(size=16)
    a, b = _patterns(16, 4)
    mem.store(a)
    mem.store(b)
    restored = HopfieldMemory.from_json(mem.to_json())
    assert restored.weight == mem.weight
    assert restored.patterns == mem.patterns


# ---------------------------------------------------------------------- #
# ElmanRNN
# ---------------------------------------------------------------------- #
def _sine_sequence(n: int = 12) -> tuple[list[list[float]], list[list[float]]]:
    """Alternating 0/1 sequence targets the NEXT step (a trivial temporal task)."""
    seq = [[float(i % 2)] for i in range(n)]
    tgt = [[1.0 - seq[i][0]] for i in range(n)]
    return seq, tgt


def test_rnn_learns_sequence() -> None:
    rnn = ElmanRNN(input_size=1, hidden_size=10, output_size=1, lr=0.3, lookback=6, seed=6)
    seq, tgt = _sine_sequence(12)
    first = rnn.train_sequence(seq, tgt)
    for _ in range(80):
        rnn.train_sequence(seq, tgt)
    last = rnn.train_sequence(seq, tgt)
    assert last < first, f"rnn loss did not drop: {first} -> {last}"
    assert last < 0.3, f"rnn loss too high: {last}"


def test_rnn_predicts_pattern() -> None:
    rnn = ElmanRNN(input_size=1, hidden_size=10, output_size=1, lr=0.3, lookback=6, seed=6)
    seq, tgt = _sine_sequence(12)
    for _ in range(80):
        rnn.train_sequence(seq, tgt)
    preds = rnn.predict_sequence(seq[-4:], steps_ahead=3)
    # the alternation pattern predicts: last input is 0? next is 1, then 0, then 1
    expected = [1.0 if seq[-1][0] == 0 else 0.0]
    assert (preds[0][0] > 0) == (expected[0] > 0), f"first prediction wrong: {preds[0]}"
    assert len(preds) == 3


def test_rnn_roundtrip() -> None:
    rnn = ElmanRNN(input_size=1, hidden_size=8, output_size=1, seed=2)
    seq, tgt = _sine_sequence(6)
    rnn.train_sequence(seq, tgt)
    restored = ElmanRNN.from_json(rnn.to_json())
    assert restored.w_hh == rnn.w_hh
    assert restored.w_xh == rnn.w_xh
    assert restored.sequences_seen == rnn.sequences_seen


# ---------------------------------------------------------------------- #
# neural_api wiring
# ---------------------------------------------------------------------- #
def test_api_modules_endpoint_smoke() -> None:
    handler = NeuralHandler
    assert handler.nn is not None
    assert handler.compressor is not None
    assert handler.anomaly is not None
    assert handler.curriculum is not None
    assert handler.memory is not None
    assert handler.rnn is not None
    # every module must serialize to flat primitives without exception
    payload = {
        "brain": handler.nn.status(),
        "compressor": handler.compressor.status(),
        "anomaly": handler.anomaly.status(),
        "curriculum": handler.curriculum.status(),
        "memory": handler.memory.status(),
        "rnn": handler.rnn.status(),
    }
    json.dumps(payload)  # must not raise


def test_api_json_flat_primitives() -> None:
    """to_json payloads must be plain JSON-serializable structures."""
    modules = (
        NeuralHandler.compressor,
        NeuralHandler.anomaly,
        NeuralHandler.curriculum,
        NeuralHandler.memory,
        NeuralHandler.rnn,
    )
    for mod in modules:
        data = mod.to_json()
        json.dumps(data)
        assert data["kind"], "module missing kind tag"
