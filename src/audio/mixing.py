"""Mixing reference clips into a recording at a controlled SNR.

Exists so a night with no confirmed bruxism can be turned into one whose
grinding locations are known exactly. That is only worth anything if the
mixing is honest: the clip has to sit at a believable level against *local*
background noise, not a global average, and the result has to be written to
the audio file so the detector rediscovers it like any other sound. Nothing
here writes detections anywhere — it writes audio.

Pure functions only (no I/O, no CLI); ``tools/inject_clips.py`` is the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

DEFAULT_TRIM_DB = 35.0
DEFAULT_FADE_MS = 10.0
DEFAULT_MIN_GAP_S = 60.0
PEAK_CEILING = 0.99


@dataclass(frozen=True)
class Placement:
    """One planned injection: where it goes and how loud it ended up."""

    start_s: float
    end_s: float
    source_clip: str
    requested_snr_db: float
    achieved_snr_db: float
    gain_db: float
    noise_rms_dbfs: float


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim > 1 else x.astype(np.float32)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample. Reference clips are 44.1 kHz; nights may be 16 kHz.

    Resampling *down* to the session rate (rather than up-sampling the night)
    is deliberate: the injected clip must be band-limited exactly the way a
    real sound reaching that recording would have been, otherwise the mix
    carries high-frequency content the night's own microphone path could
    never have produced, and the detector could key on that instead.
    """
    if sr_in == sr_out:
        return x.astype(np.float32)
    divisor = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // divisor, sr_in // divisor).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, below_peak_db: float = DEFAULT_TRIM_DB,
                 frame_ms: float = 5.0) -> np.ndarray:
    """Drop leading/trailing quiet so the injected span is the sound itself.

    The span written to the ground-truth CSV is only meaningful if it is the
    grinding and not the silence a hand-cut clip happens to carry around it —
    a 0.6s burst inside a 2s clip would otherwise be scored as a 2s event.
    Threshold is relative to the clip's own peak, so it works regardless of
    how loud the clip was recorded.
    """
    if x.size == 0:
        return x
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (x.size // frame) * frame
    if usable < frame:
        return x
    frames = x[:usable].reshape(-1, frame)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(energy.max())
    if peak <= 0.0:
        return x
    loud = np.flatnonzero(energy >= peak * (10.0 ** (-below_peak_db / 20.0)))
    if loud.size == 0:
        return x
    return x[loud[0] * frame : min(x.size, (loud[-1] + 1) * frame)]


def apply_fade(x: np.ndarray, sr: int, fade_ms: float = DEFAULT_FADE_MS) -> np.ndarray:
    """Linear fade in/out so the splice doesn't click.

    A hard cut into a recording is a step discontinuity — broadband energy
    across every frequency at once. That is both audible and exactly the kind
    of artifact a detector could learn to key on instead of the grinding.
    """
    n = int(sr * fade_ms / 1000.0)
    if n <= 0 or x.size < 2 * n:
        return x
    faded = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    faded[:n] *= ramp
    faded[-n:] *= ramp[::-1]
    return faded


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def _db(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-12))


def gain_for_snr(clip: np.ndarray, noise_rms: float, target_snr_db: float) -> float:
    """Linear gain putting ``clip`` ``target_snr_db`` above the local noise."""
    clip_rms = rms(clip)
    if clip_rms <= 0.0 or noise_rms <= 0.0:
        return 1.0
    return float((noise_rms * (10.0 ** (target_snr_db / 20.0))) / clip_rms)


def frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    """Per-frame RMS in dBFS — the basis for deciding where audio actually is."""
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (x.size // frame) * frame
    if usable < frame:
        return np.array([], dtype=np.float64)
    frames = x[:usable].reshape(-1, frame).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), 1e-12))


def active_spans(
    x: np.ndarray,
    sr: int,
    silence_db: float = -90.0,
    frame_ms: float = 50.0,
    min_span_s: float = 5.0,
) -> list[tuple[float, float]]:
    """Stretches that contain actual recorded background, not digital silence.

    This is not a nicety. Measured on the night this was built for, 61% of the
    file is exact digital silence (an import artifact, not a quiet room), and
    evenly spacing injections across wall-clock time dropped four of six clips
    into dead air. That is worthless as a detection test twice over: SNR
    against a zero floor is undefined, so the clip goes in at some arbitrary
    level, and a sound surrounded by pure silence is trivially detectable in a
    way no real night would be.
    """
    energy = frame_energy_db(x, sr, frame_ms)
    if energy.size == 0:
        return []
    frame_s = frame_ms / 1000.0
    loud = energy > silence_db
    if not loud.any():
        return []
    edges = np.diff(loud.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if loud[0]:
        starts.insert(0, 0)
    if loud[-1]:
        ends.append(loud.size)
    spans = [
        (start * frame_s, end * frame_s)
        for start, end in zip(starts, ends)
        if (end - start) * frame_s >= min_span_s
    ]
    return spans


def plan_positions(
    spans: list[tuple[float, float]],
    count: int,
    clip_durations_s: list[float],
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    edge_margin_s: float = 2.0,
) -> list[float]:
    """Start times spread evenly through the *usable* audio, not the timeline.

    Positions are spaced evenly in "active seconds" and then mapped back to
    wall-clock, so a night that is half silence still gets its injections
    distributed across the parts a detector will actually look at. Spacing of
    at least ``min_gap_s`` is enforced afterwards in wall-clock terms, since
    that constraint exists to keep events from merging into one another.

    Returns fewer than ``count`` if the usable audio cannot hold them.
    """
    longest = max(clip_durations_s) if clip_durations_s else 0.0
    usable = [
        (start + edge_margin_s, end - edge_margin_s - longest)
        for start, end in spans
        if (end - start) > 2 * edge_margin_s + longest
    ]
    usable = [(a, b) for a, b in usable if b > a]
    if not usable or count <= 0:
        return []

    lengths = [b - a for a, b in usable]
    total = sum(lengths)
    offsets = [total * (i + 0.5) / count for i in range(count)]

    positions: list[float] = []
    for offset in offsets:
        cursor = offset
        for (start, _end), length in zip(usable, lengths):
            if cursor <= length:
                candidate = start + cursor
                if not positions or candidate - positions[-1] >= min_gap_s:
                    positions.append(float(candidate))
                break
            cursor -= length
    return positions


def mix_into(
    track: np.ndarray, clip: np.ndarray, start_sample: int, gain: float
) -> tuple[np.ndarray, float]:
    """Add ``clip * gain`` into ``track`` in place, backing off any clipping.

    Returns the (possibly reduced) gain actually used, so the caller can
    record the SNR that was really achieved rather than the one requested —
    a ground-truth file that overstates the level would make the detector
    look worse than it is.
    """
    end_sample = min(track.size, start_sample + clip.size)
    span = end_sample - start_sample
    if span <= 0:
        return track, 0.0

    segment = track[start_sample:end_sample]
    scaled = clip[:span] * gain
    peak = float(np.max(np.abs(segment + scaled))) if span else 0.0
    if peak > PEAK_CEILING:
        gain *= PEAK_CEILING / peak
        scaled = clip[:span] * gain
    track[start_sample:end_sample] = segment + scaled
    return track, float(gain)


def gain_db(gain: float) -> float:
    return _db(gain)


def rms_dbfs_of(x: np.ndarray) -> float:
    return _db(rms(x))
