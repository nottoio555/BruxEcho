"""Tests for the frame-probabilities -> events stage of the detection pipeline.

Synthetic probability sequences rather than real audio, for the same reason
``test_energy_gate.py`` uses synthetic signals: these encode the exact shapes
that caused real bugs (a single-frame true positive being smoothed away, an
event that never exits), and a real recording can't be made to contain a
specific shape on demand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detect.pipeline import (
    DEFAULT_MEDIAN_FRAMES,
    Event,
    Region,
    frames_to_events,
    median_filter_1d,
    merge_events,
    merge_regions,
)

CLASSES = ["bruxism", "snore", "other"]
BRUXISM, SNORE = 0, 1


def probs(bruxism: list[float], snore: list[float] | None = None) -> np.ndarray:
    """(n_frames, 3) probabilities; "other" takes whatever is left."""
    b = np.asarray(bruxism, dtype=np.float32)
    s = np.asarray(snore if snore is not None else [0.0] * len(bruxism), dtype=np.float32)
    return np.stack([b, s, np.clip(1.0 - b - s, 0.0, 1.0)], axis=1)


class TestMedianFilter:
    def test_size_one_is_identity(self):
        values = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        assert np.array_equal(median_filter_1d(values, size=1), values)

    def test_preserves_length_and_alignment(self):
        values = np.linspace(0, 1, 11, dtype=np.float32)
        assert median_filter_1d(values, size=3).shape == values.shape

    def test_removes_isolated_spike(self):
        values = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        assert median_filter_1d(values, size=3)[1] == pytest.approx(0.1)


class TestFramesToEvents:
    def _events(self, p, threshold=0.8, median_frames=DEFAULT_MEDIAN_FRAMES, **kw):
        return frames_to_events(
            p, CLASSES, region_start_s=0.0, threshold=threshold,
            target_indices=[BRUXISM, SNORE], median_frames=median_frames, **kw,
        )

    def test_empty_input(self):
        assert self._events(np.zeros((0, 3), dtype=np.float32)) == []

    def test_single_frame_peak_survives_by_default(self):
        """The regression this default exists for: one frame at 0.806 between
        two quiet frames is a real confirmed grind, not a glitch."""
        events = self._events(probs([0.098, 0.806, 0.166]))
        assert [e.label for e in events] == ["bruxism"]

    def test_median_three_deletes_that_same_peak(self):
        assert self._events(probs([0.098, 0.806, 0.166]), median_frames=3) == []

    def test_hysteresis_keeps_an_event_through_a_dip(self):
        """A dip to 0.6 is above the 0.8*0.6 exit threshold, so one event, not two."""
        events = self._events(probs([0.9, 0.6, 0.9]))
        assert len(events) == 1

    def test_event_ends_when_probability_falls_below_exit(self):
        events = self._events(probs([0.9, 0.9, 0.05, 0.05, 0.9, 0.9]))
        assert len(events) == 2

    def test_classes_are_tracked_independently(self):
        """An ambiguous stretch must be able to surface as both, so a louder
        snore can't silently suppress the bruxism the app exists to count."""
        events = self._events(probs([0.85, 0.85], snore=[0.95, 0.95]))
        assert sorted(e.label for e in events) == ["bruxism", "snore"]

    def test_short_events_are_dropped_by_min_duration(self):
        assert self._events(probs([0.9]), min_event_s=5.0) == []

    def test_confidence_is_the_mean_over_the_event(self):
        events = self._events(probs([0.9, 0.7]))
        assert events[0].confidence == pytest.approx(0.8, abs=1e-6)

    def test_offsets_are_relative_to_the_region_start(self):
        p = probs([0.05, 0.9, 0.05])
        events = frames_to_events(
            p, CLASSES, region_start_s=100.0, threshold=0.8, target_indices=[BRUXISM]
        )
        assert events[0].start_s == pytest.approx(100.48)


class TestMerging:
    def test_same_label_events_within_the_gap_merge(self):
        merged = merge_events(
            [Event(0.0, 1.0, "bruxism", 0.9), Event(1.2, 2.0, "bruxism", 0.7)], gap_s=0.5
        )
        assert len(merged) == 1 and merged[0].end_s == 2.0

    def test_different_labels_never_merge(self):
        merged = merge_events([Event(0.0, 1.0, "bruxism", 0.9), Event(1.1, 2.0, "snore", 0.9)])
        assert len(merged) == 2

    def test_merged_confidence_is_duration_weighted(self):
        """A long confident stretch shouldn't be dragged down by a short
        marginal one merged onto its end."""
        merged = merge_events(
            [Event(0.0, 3.0, "bruxism", 1.0), Event(3.1, 4.1, "bruxism", 0.0)], gap_s=0.5
        )
        assert merged[0].confidence == pytest.approx(0.75)

    def test_output_is_chronological(self):
        merged = merge_events([Event(10.0, 11.0, "snore", 0.9), Event(1.0, 2.0, "bruxism", 0.9)])
        assert [e.start_s for e in merged] == [1.0, 10.0]

    def test_overlapping_regions_merge_so_audio_is_embedded_once(self):
        assert merge_regions([Region(0.0, 5.0), Region(3.0, 8.0)]) == [Region(0.0, 8.0)]

    def test_distant_regions_stay_separate(self):
        assert len(merge_regions([Region(0.0, 1.0), Region(50.0, 51.0)])) == 2

    def test_empty_inputs(self):
        assert merge_events([]) == [] and merge_regions([]) == []
