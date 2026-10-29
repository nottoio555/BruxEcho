"""Tests for the make-it-audible chain applied to clips a person listens to.

Every case here is a condition that was measured on real clips and produced
audibly wrong output — a DC offset eating the headroom, a mostly-silent clip
whose average level is meaningless, a transient that dragged a whole clip
back down through the limiter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.listening import (
    active_rms_dbfs,
    audible_rms_dbfs,
    compress,
    enhance_for_listening,
    high_pass,
    limit,
    remove_dc,
)

SR = 16000


def tone(seconds: float, hz: float = 1500.0, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * hz * t) * amplitude).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def db(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.max(np.abs(x))), 1e-12))


class TestRemoveDc:
    def test_offset_is_removed(self):
        x = tone(1.0) + 0.18  # the offset measured on a real night
        assert abs(float(np.mean(remove_dc(x)))) < 1e-6

    def test_headroom_is_recovered(self):
        x = tone(1.0, amplitude=0.1) + 0.18
        assert db(remove_dc(x)) < db(x)

    def test_empty_input(self):
        assert remove_dc(np.zeros(0, dtype=np.float32)).size == 0


class TestHighPass:
    def test_subsonic_rumble_is_removed(self):
        rumble = tone(1.0, hz=8.0, amplitude=0.5)
        assert db(high_pass(rumble, SR, 300.0)) < db(rumble) - 20

    def test_grinding_band_survives(self):
        """Compared on RMS, not peak: sosfiltfilt leaves a small edge transient
        that lifts the peak ~2 dB while the passband itself is untouched."""
        grind = tone(1.0, hz=1500.0, amplitude=0.1)

        def rms_db(x):
            return 20 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))

        assert rms_db(high_pass(grind, SR, 300.0)) == pytest.approx(rms_db(grind), abs=0.5)

    def test_very_short_input_is_passed_through(self):
        x = np.ones(8, dtype=np.float32)
        assert np.array_equal(high_pass(x, SR), x)


class TestActiveLevel:
    def test_silence_around_a_burst_does_not_drag_the_level_down(self):
        """The bug this exists for: whole-clip RMS of a mostly-silent clip
        reads ~60 dB too quiet, and normalizing on it overshoots wildly."""
        clip = np.concatenate([silence(2.0), tone(1.0), silence(2.0)])
        assert active_rms_dbfs(clip, SR) > audible_rms_dbfs(clip, SR) + 5

    def test_a_clip_that_is_all_signal_agrees_with_plain_rms(self):
        clip = tone(3.0)
        assert active_rms_dbfs(clip, SR) == pytest.approx(audible_rms_dbfs(clip, SR), abs=2.0)


class TestLimit:
    def test_peaks_are_brought_under_the_ceiling(self):
        x = tone(1.0, amplitude=0.9)
        assert db(limit(x, SR, -1.0)) <= -1.0 + 0.1

    def test_a_single_transient_does_not_crush_the_whole_clip(self):
        """Scaling globally by peak/ceiling was the first implementation and
        undid the normalization; only the loud moment should come down."""
        x = tone(3.0, amplitude=0.2)
        x[SR] = 4.0  # one runaway sample
        limited = limit(x, SR, -1.0)
        quiet_part = limited[SR * 2 : SR * 3]
        assert float(np.max(np.abs(quiet_part))) > 0.1

    def test_quiet_audio_is_left_alone(self):
        x = tone(1.0, amplitude=0.01)
        assert np.allclose(limit(x, SR, -1.0), x)


class TestCompress:
    def test_quiet_passage_is_lifted_toward_the_loud_one(self):
        clip = np.concatenate([tone(1.0, amplitude=0.3), tone(1.0, amplitude=0.01)])
        out = compress(clip, SR, target_dbfs=-20.0)
        before = float(np.max(np.abs(clip[SR:]))) / float(np.max(np.abs(clip[:SR])))
        after = float(np.max(np.abs(out[SR:]))) / float(np.max(np.abs(out[:SR])))
        assert after > before

    def test_too_short_to_frame_is_returned_unchanged(self):
        x = tone(0.01)
        assert np.array_equal(compress(x, SR), x)


class TestEnhanceForListening:
    def test_a_real_world_shaped_clip_becomes_audible(self):
        """DC offset + subsonic rumble + a quiet burst buried in silence —
        the exact shape that played as a faint thud."""
        clip = np.concatenate([silence(2.0), tone(1.0, amplitude=0.0005), silence(2.0)])
        clip = (clip + tone(5.0, hz=8.0, amplitude=0.3)[: clip.size] + 0.18).astype(np.float32)
        out = enhance_for_listening(clip, SR)
        assert active_rms_dbfs(out, SR) > active_rms_dbfs(clip, SR) + 30

    def test_the_inaudible_rumble_does_not_set_the_level(self):
        """The original bug: a loud subsonic tone ate all the headroom and the
        audible part stayed ~45 dB down."""
        quiet_grind = tone(1.0, hz=1500.0, amplitude=0.001)
        rumble = tone(1.0, hz=8.0, amplitude=0.5)
        out = enhance_for_listening((quiet_grind + rumble).astype(np.float32), SR)
        assert audible_rms_dbfs(out, SR) > -35

    def test_hissy_top_octave_is_attenuated(self):
        """Band-limiting is what actually fixed the "ซ่า" complaint."""
        hiss = tone(1.0, hz=6500.0, amplitude=0.05)
        grind = tone(1.0, hz=1500.0, amplitude=0.05)
        out = enhance_for_listening((hiss + grind).astype(np.float32), SR)
        spectrum = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(out.size, 1 / SR)
        top = spectrum[freqs > 6000].max()
        mid = spectrum[(freqs > 1000) & (freqs < 2000)].max()
        assert top < mid / 4

    def test_output_never_clips(self):
        clip = np.concatenate([silence(1.0), tone(0.5, amplitude=0.4), silence(1.0)])
        out = enhance_for_listening(clip, SR)
        assert float(np.max(np.abs(out))) <= 1.0

    def test_length_and_rate_are_preserved(self):
        clip = np.concatenate([silence(1.0), tone(0.5), silence(1.0)])
        assert enhance_for_listening(clip, SR).size == clip.size

    def test_digital_silence_stays_silent(self):
        """Never turn a silent clip into amplified hiss."""
        out = enhance_for_listening(silence(2.0), SR)
        assert float(np.max(np.abs(out))) == pytest.approx(0.0, abs=1e-6)

    def test_empty_input(self):
        assert enhance_for_listening(np.zeros(0, dtype=np.float32), SR).size == 0
