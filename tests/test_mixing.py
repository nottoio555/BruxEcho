"""Tests for mixing reference clips into a recording at a controlled level.

Synthetic signals throughout, for the usual reason: these encode the exact
conditions that produced wrong output on real data (a clip landing in digital
silence, a gain loud enough to clip) and a real night can't be asked to
contain those on demand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.mixing import (
    PEAK_CEILING,
    active_spans,
    apply_fade,
    frame_energy_db,
    gain_for_snr,
    mix_into,
    plan_positions,
    resample_to,
    rms,
    to_mono,
    trim_silence,
)

SR = 16000


def noise(seconds: float, amplitude: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * amplitude).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class TestConversions:
    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert np.allclose(to_mono(stereo), [0.5, 0.5])

    def test_to_mono_passes_mono_through(self):
        mono = np.array([0.1, 0.2], dtype=np.float32)
        assert np.array_equal(to_mono(mono), mono)

    def test_resample_changes_length_proportionally(self):
        x = noise(1.0)
        out = resample_to(x, SR, 8000)
        assert out.size == pytest.approx(x.size // 2, abs=2)

    def test_resample_is_identity_at_same_rate(self):
        x = noise(0.1)
        assert np.array_equal(resample_to(x, SR, SR), x)


class TestTrimSilence:
    def test_removes_leading_and_trailing_quiet(self):
        burst = np.concatenate([silence(0.5), noise(0.3, amplitude=0.5), silence(0.5)])
        trimmed = trim_silence(burst, SR)
        assert trimmed.size < burst.size
        assert trimmed.size == pytest.approx(0.3 * SR, rel=0.25)

    def test_keeps_a_clip_that_is_all_signal(self):
        x = noise(0.4, amplitude=0.5)
        assert trim_silence(x, SR).size == pytest.approx(x.size, rel=0.05)

    def test_all_silence_is_returned_unchanged(self):
        x = silence(0.2)
        assert np.array_equal(trim_silence(x, SR), x)

    def test_empty_input(self):
        assert trim_silence(np.zeros(0, dtype=np.float32), SR).size == 0


class TestFade:
    def test_edges_are_ramped_to_zero(self):
        x = np.ones(SR, dtype=np.float32)
        faded = apply_fade(x, SR, fade_ms=10.0)
        assert faded[0] == pytest.approx(0.0, abs=1e-6)
        assert faded[-1] == pytest.approx(0.0, abs=1e-6)

    def test_middle_is_untouched(self):
        x = np.ones(SR, dtype=np.float32)
        assert apply_fade(x, SR, fade_ms=10.0)[SR // 2] == pytest.approx(1.0)

    def test_clip_shorter_than_two_fades_is_left_alone(self):
        x = np.ones(10, dtype=np.float32)
        assert np.array_equal(apply_fade(x, SR, fade_ms=10.0), x)


class TestGainForSnr:
    def test_gain_puts_clip_at_requested_snr(self):
        clip = noise(0.5, amplitude=0.1, seed=1)
        noise_rms = 0.01
        gain = gain_for_snr(clip, noise_rms, 15.0)
        achieved = 20 * np.log10(rms(clip * gain) / noise_rms)
        assert achieved == pytest.approx(15.0, abs=0.01)

    def test_silent_background_falls_back_to_unity(self):
        """SNR against a zero floor is undefined — callers must not place here."""
        assert gain_for_snr(noise(0.1), 0.0, 15.0) == 1.0

    def test_silent_clip_falls_back_to_unity(self):
        assert gain_for_snr(silence(0.1), 0.01, 15.0) == 1.0


class TestActiveSpans:
    def test_finds_the_audible_stretch_between_silence(self):
        track = np.concatenate([silence(10.0), noise(10.0, amplitude=0.01), silence(10.0)])
        spans = active_spans(track, SR, min_span_s=1.0)
        assert len(spans) == 1
        start, end = spans[0]
        assert start == pytest.approx(10.0, abs=0.2)
        assert end == pytest.approx(20.0, abs=0.2)

    def test_digital_silence_yields_nothing(self):
        assert active_spans(silence(30.0), SR) == []

    def test_spans_shorter_than_the_minimum_are_dropped(self):
        track = np.concatenate([silence(5.0), noise(0.5, amplitude=0.01), silence(5.0)])
        assert active_spans(track, SR, min_span_s=5.0) == []

    def test_frame_energy_is_quiet_for_silence(self):
        assert frame_energy_db(silence(1.0), SR).max() < -100.0


class TestPlanPositions:
    def test_positions_land_inside_active_spans(self):
        spans = [(100.0, 200.0), (400.0, 500.0)]
        positions = plan_positions(spans, 2, [1.0], min_gap_s=10.0)
        assert len(positions) == 2
        for p in positions:
            assert any(start <= p <= end for start, end in spans)

    def test_silence_between_spans_is_never_used(self):
        spans = [(0.0, 50.0), (1000.0, 1050.0)]
        for p in plan_positions(spans, 4, [1.0], min_gap_s=1.0):
            assert not (50.0 < p < 1000.0)

    def test_minimum_gap_is_respected(self):
        positions = plan_positions([(0.0, 600.0)], 10, [1.0], min_gap_s=60.0)
        gaps = np.diff(positions)
        assert all(gap >= 60.0 for gap in gaps)

    def test_returns_fewer_when_the_gap_cannot_be_met(self):
        assert len(plan_positions([(0.0, 100.0)], 10, [1.0], min_gap_s=60.0)) < 10

    def test_no_spans_means_no_positions(self):
        assert plan_positions([], 5, [1.0]) == []

    def test_span_too_short_for_the_clip_is_skipped(self):
        assert plan_positions([(0.0, 3.0)], 1, [10.0]) == []


class TestMixInto:
    def test_clip_is_added_at_the_requested_offset(self):
        track = silence(10.0)
        clip = np.ones(SR, dtype=np.float32) * 0.1
        track, _ = mix_into(track, clip, SR * 2, 1.0)
        assert track[SR * 2 + 100] == pytest.approx(0.1)
        assert track[100] == pytest.approx(0.0)

    def test_gain_is_backed_off_to_avoid_clipping(self):
        track = silence(2.0)
        clip = np.ones(SR, dtype=np.float32)
        track, used = mix_into(track, clip, 0, 10.0)
        assert used < 10.0
        assert float(np.max(np.abs(track))) <= PEAK_CEILING + 1e-6

    def test_gain_is_left_alone_when_there_is_headroom(self):
        track = silence(2.0)
        clip = np.ones(SR, dtype=np.float32) * 0.01
        _, used = mix_into(track, clip, 0, 2.0)
        assert used == pytest.approx(2.0)

    def test_offset_past_the_end_is_a_no_op(self):
        track = silence(1.0)
        before = track.copy()
        track, used = mix_into(track, np.ones(100, dtype=np.float32), track.size + 10, 1.0)
        assert used == 0.0
        assert np.array_equal(track, before)

    def test_clip_running_past_the_end_is_truncated_not_an_error(self):
        track = silence(1.0)
        track, _ = mix_into(track, np.ones(SR, dtype=np.float32) * 0.1, track.size - 100, 1.0)
        assert track.size == SR
        assert track[-1] == pytest.approx(0.1)


class TestSpreadPlacement:
    """`tools.spread_session.place_spans` — stretching a short recording over a night."""

    def _spans(self, n: int, each_s: float = 10.0):
        return [(i * 100.0, i * 100.0 + each_s) for i in range(n)]

    def test_every_span_gets_a_position(self):
        from tools.spread_session import place_spans

        spans = self._spans(5)
        positions = place_spans(spans, 3600.0, np.random.default_rng(0))
        assert len(positions) == len(spans)

    def test_positions_stay_inside_the_night(self):
        from tools.spread_session import place_spans

        spans = self._spans(5)
        night = 3600.0
        positions = place_spans(spans, night, np.random.default_rng(1))
        for start, (span_start, span_end) in zip(positions, spans):
            assert 0.0 <= start
            assert start + (span_end - span_start) <= night

    def test_spans_never_overlap_after_placement(self):
        from tools.spread_session import place_spans

        spans = self._spans(20)
        positions = place_spans(spans, 7200.0, np.random.default_rng(2))
        ends = [p + (e - s) for p, (s, e) in zip(positions, spans)]
        assert all(nxt >= prev for nxt, prev in zip(positions[1:], ends[:-1]))

    def test_chronological_order_is_preserved(self):
        """Shuffling would splice unrelated moments into new, never-recorded sounds."""
        from tools.spread_session import place_spans

        positions = place_spans(self._spans(10), 7200.0, np.random.default_rng(3))
        assert positions == sorted(positions)

    def test_same_seed_gives_the_same_night(self):
        from tools.spread_session import place_spans

        spans = self._spans(8)
        a = place_spans(spans, 5000.0, np.random.default_rng(7))
        b = place_spans(spans, 5000.0, np.random.default_rng(7))
        assert a == b

    def test_night_shorter_than_the_audio_is_refused(self):
        from tools.spread_session import place_spans

        with pytest.raises(SystemExit):
            place_spans(self._spans(10, each_s=100.0), 60.0, np.random.default_rng(0))
