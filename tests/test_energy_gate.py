"""Tests for src.detect.energy_gate using synthetic signals only.

Per project policy, no real recordings are touched here — every signal is
generated in-process with a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detect.energy_gate import DEFAULT_LOW_HZ, EnergyGate, resolve_band

SAMPLE_RATE = 44100


def _make_signal(
    duration_s: float,
    bursts: list[tuple[float, float]],
    rng: np.random.Generator,
    bg_amp: float = 0.001,
    burst_amp: float = 0.05,
) -> np.ndarray:
    """Quiet background noise with louder bursts inserted at known positions."""
    n = int(duration_s * SAMPLE_RATE)
    signal = rng.normal(0.0, bg_amp, n).astype(np.float32)
    for start_s, dur_s in bursts:
        start = int(start_s * SAMPLE_RATE)
        end = min(int((start_s + dur_s) * SAMPLE_RATE), n)
        signal[start:end] += rng.normal(0.0, burst_amp, end - start).astype(np.float32)
    return signal


def _run_gate(signal: np.ndarray, block_seconds: float = 5.0, **gate_kwargs):
    gate = EnergyGate(sample_rate=SAMPLE_RATE, session_id="test", **gate_kwargs)
    block_len = int(block_seconds * SAMPLE_RATE)
    for start in range(0, signal.size, block_len):
        gate.process_block(signal[start : start + block_len], segment_file="segment_000.wav")
    return gate.finalize()


def test_detects_bursts_at_known_positions():
    rng = np.random.default_rng(42)
    bursts = [(10.0, 0.6), (30.0, 1.0), (50.0, 0.45)]
    signal = _make_signal(70.0, bursts, rng)

    events = _run_gate(signal)

    assert len(events) == len(bursts)
    for event, (expected_start, expected_dur) in zip(events, bursts):
        assert event.start_s == pytest.approx(expected_start, abs=0.15)
        assert event.duration_s == pytest.approx(expected_dur, abs=0.2)
        assert event.segment_file == "segment_000.wav"
        assert 0.0 <= event.band_ratio_high <= 1.0
        assert event.snr_above_floor_db > 0.0


def test_drops_events_shorter_than_min_duration():
    rng = np.random.default_rng(1)
    bursts = [(10.0, 1.0), (30.0, 0.15)]  # second burst is below the 0.4s floor
    signal = _make_signal(45.0, bursts, rng)

    events = _run_gate(signal, min_duration_s=0.4)

    assert len(events) == 1
    assert events[0].start_s == pytest.approx(10.0, abs=0.15)


def test_merges_close_events():
    rng = np.random.default_rng(2)
    bursts = [(10.0, 0.6), (10.9, 0.6)]  # 0.3s apart, gap < the 0.5s merge threshold
    signal = _make_signal(20.0, bursts, rng)

    events = _run_gate(signal)

    assert len(events) == 1
    assert events[0].start_s == pytest.approx(10.0, abs=0.15)
    assert events[0].end_s == pytest.approx(11.5, abs=0.15)


def test_does_not_merge_events_far_apart():
    rng = np.random.default_rng(4)
    bursts = [(10.0, 0.6), (12.0, 0.6)]  # 1.4s apart, should stay separate
    signal = _make_signal(20.0, bursts, rng)

    events = _run_gate(signal)

    assert len(events) == 2


def test_detection_independent_of_caller_block_size():
    """Results must not depend on how the caller chunks process_block calls."""
    rng = np.random.default_rng(3)
    bursts = [(5.0, 0.8), (20.0, 0.6)]
    signal = _make_signal(30.0, bursts, rng)

    events_small_blocks = _run_gate(signal, block_seconds=2.0)
    events_one_block = _run_gate(signal, block_seconds=1000.0)

    assert len(events_small_blocks) == len(events_one_block) == len(bursts)
    for a, b in zip(events_small_blocks, events_one_block):
        assert a.start_s == pytest.approx(b.start_s, abs=0.05)
        assert a.end_s == pytest.approx(b.end_s, abs=0.05)


def test_silence_produces_no_events():
    rng = np.random.default_rng(5)
    signal = _make_signal(60.0, bursts=[], rng=rng)

    events = _run_gate(signal)

    assert events == []


def test_resolve_band_scales_with_sample_rate():
    band_44k = resolve_band(44100)
    assert band_44k.low_hz == DEFAULT_LOW_HZ
    assert band_44k.high_hz == 16000.0
    assert not band_44k.narrow_band_warning

    band_11k = resolve_band(11025)
    assert band_11k.high_hz == pytest.approx(11025 / 2 * 0.95)
    assert band_11k.narrow_band_warning  # high_hz ~5237 Hz < 6000 Hz


def test_resolve_band_rejects_too_low_sample_rate():
    with pytest.raises(ValueError):
        resolve_band(4000)  # Nyquist*0.95 (1900 Hz) doesn't clear the 2000 Hz low edge
