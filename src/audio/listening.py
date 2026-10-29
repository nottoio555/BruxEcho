"""Making a detected clip actually audible to a person on a phone speaker.

This is presentation-only processing. It is applied to the copies in
``analysis_clips/`` that a human listens to, never to anything the detector
reads — the model always works from the original segments.

### Why peak normalization was not enough

The clips were already peak-normalized to -3 dBFS and still played as a faint
thud or nothing at all. Measured on a real night: **95.7% of a clip's energy
sat below 20 Hz**, with a DC offset as large as -0.18. Peak normalization
therefore spent the entire headroom budget scaling an inaudible DC-shifted
rumble, leaving the grinding itself around -45 dBFS in the 300-8000 Hz band
where a person can actually hear it. The "thud" was the sub-bass; the
grinding was 40+ dB underneath it.

So the chain here works on the band a listener hears:

1. remove DC, then high-pass — recovers ~40 dB of headroom immediately
2. compress — a grind is a short burst inside a mostly-quiet window, so a
   single gain for the whole clip leaves the burst quiet; this lifts the
   quiet parts toward the loud ones
3. normalize the **audible-band** RMS to a target, not the full-band peak
4. limit — catch the peaks compression created, without clipping

Pure functions, no I/O.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# 300 Hz, matching the project's long-standing high-pass choice (see README
# "ทำไมต้อง high-pass filter"). Real sleep bruxism keeps 90% of its energy
# inside 683-5104 Hz, so nothing audible is lost — and leaving 120-300 Hz in
# cost most of the headroom, because peaks down there kept triggering the
# limiter and dragging the whole clip back down.
DEFAULT_HIGHPASS_HZ = 300.0
DEFAULT_TARGET_PEAK_DBFS = -1.0
# How much louder than peak-normalized to push, accepting some limiting.
# Chosen from a measured sweep (contrast / level / hiss, 50 clips):
#   +0 dB  -> 12.2 dB contrast, -40.9 dBFS, hiss -26.3  (clean but quiet)
#   +12 dB -> 10.0 dB contrast, -29.0 dBFS, hiss -14.3  (this)
#   +24 dB ->  5.3 dB contrast, -18.5 dBFS, hiss  -3.3  (loud, hissy)
# Each 6 dB of loudness costs 1-2.5 dB of contrast and adds 6 dB of hiss.
DEFAULT_EXTRA_GAIN_DB = 12.0
DEFAULT_TARGET_RMS_DBFS = -20.0  # kept for callers that still pass it
DEFAULT_LIMIT_DBFS = -1.0
DEFAULT_MAX_BOOST_DB = 30.0
AUDIBLE_LOW_HZ = 300.0
AUDIBLE_HIGH_HZ = 7900.0

# What the *output* is band-limited to, narrower than the band used for
# measurement. Measured across 50 clips, grinding energy peaks in 700-3000 Hz
# and is ~9 dB down by 5-8 kHz, while the room's hiss is broadband — so the
# top octave contributes far more "ซ่า" than grinding. Cutting it is the
# cheapest, least destructive way to make a clip easier to listen to.
OUTPUT_LOW_HZ = 400.0
OUTPUT_HIGH_HZ = 5000.0

# Downward expansion: how far below the clip's own active level a frame has to
# sit before it is pushed further down, and how far down it can go.
EXPAND_THRESHOLD_BELOW_DB = 12.0
EXPAND_RATIO = 2.0
EXPAND_MAX_REDUCTION_DB = 18.0
COMPRESS_FRAME_MS = 50.0
COMPRESS_SMOOTH_FRAMES = 3
# Frames quieter than this are left alone: lifting them just turns the room's
# noise floor into loud hiss between the grinds.
COMPRESS_FLOOR_DBFS = -70.0


def _db(value: float) -> float:
    return 20.0 * float(np.log10(max(float(value), 1e-12)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean. A -0.18 DC offset alone costs ~15 dB of headroom."""
    return (x - np.mean(x)).astype(np.float32) if x.size else x


def high_pass(x: np.ndarray, sr: int, cutoff_hz: float = DEFAULT_HIGHPASS_HZ) -> np.ndarray:
    """Zero-phase high-pass. Clips are short and fully in memory, so
    ``sosfiltfilt`` is fine here (unlike the streaming energy gate)."""
    nyquist = sr / 2.0
    normalized = min(cutoff_hz / nyquist, 0.99)
    if x.size < 32 or normalized <= 0.0:
        return x
    sos = butter(4, normalized, btype="high", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def audible_rms_dbfs(
    x: np.ndarray, sr: int, low_hz: float = AUDIBLE_LOW_HZ, high_hz: float = AUDIBLE_HIGH_HZ
) -> float:
    """RMS restricted to the band a person actually hears on a phone speaker.

    The number to optimize. Full-band RMS is dominated by content no speaker
    can reproduce and no ear can hear, which is exactly how clips measuring
    -12 dBFS ended up inaudible.
    """
    nyquist = sr / 2.0
    low = min(low_hz / nyquist, 0.98)
    high = min(high_hz / nyquist, 0.99)
    if x.size < 32 or low >= high:
        return _db(_rms(x))
    sos = butter(4, [low, high], btype="band", output="sos")
    return _db(_rms(sosfiltfilt(sos, x).astype(np.float32)))


def active_rms_dbfs(
    x: np.ndarray,
    sr: int,
    percentile: float = 90.0,
    frame_ms: float = COMPRESS_FRAME_MS,
) -> float:
    """Level of the parts of the clip that actually contain sound.

    Whole-clip RMS is the wrong yardstick here and measurably so. A 5s window
    holding a 1s grind is mostly silence — in the spread nights it is literal
    digital zero — so its average sits near -85 dBFS. Normalizing *that* to a
    -20 dBFS target multiplies by ~60 dB and throws the grind itself to
    +16 dBFS, whereupon the limiter claws back 17 dB and the clip ends up
    quiet again. Taking a high percentile of the frame levels instead measures
    the burst, which is the thing the listener is trying to hear.
    """
    band = _band_filter(x, sr)
    frame = max(1, int(sr * frame_ms / 1000.0))
    if band.size < frame * 2:
        return _db(_rms(band))
    usable = (band.size // frame) * frame
    frames = band[:usable].reshape(-1, frame).astype(np.float64)
    levels = np.sqrt(np.mean(frames**2, axis=1))
    return _db(float(np.percentile(levels, percentile)))


def _band_filter(x: np.ndarray, sr: int) -> np.ndarray:
    nyquist = sr / 2.0
    low = min(AUDIBLE_LOW_HZ / nyquist, 0.98)
    high = min(AUDIBLE_HIGH_HZ / nyquist, 0.99)
    if x.size < 32 or low >= high:
        return x
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def band_limit(
    x: np.ndarray, sr: int, low_hz: float = OUTPUT_LOW_HZ, high_hz: float = OUTPUT_HIGH_HZ
) -> np.ndarray:
    """Keep only the band the grinding actually occupies."""
    nyquist = sr / 2.0
    low = min(low_hz / nyquist, 0.98)
    high = min(high_hz / nyquist, 0.99)
    if x.size < 32 or low >= high:
        return x
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def expand(
    x: np.ndarray,
    sr: int,
    reference_dbfs: float,
    threshold_below_db: float = EXPAND_THRESHOLD_BELOW_DB,
    ratio: float = EXPAND_RATIO,
    max_reduction_db: float = EXPAND_MAX_REDUCTION_DB,
    frame_ms: float = COMPRESS_FRAME_MS,
) -> np.ndarray:
    """Push the gaps between grinds further down — the opposite of [compress].

    This replaces compression as the default for a measured reason: raw clips
    hold about 12.6 dB between the grinding and the room behind it, and
    compressing squeezed that to 4.2 dB, which is why the processed clips came
    back as mostly hiss. Expansion moves the other way, so the grind stands
    out more after processing than before, not less.

    The threshold is relative to the clip's own active level, because the room
    floor differs from night to night and an absolute value would gate half
    the grinding away on a quiet recording.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    if x.size < frame * 2:
        return x

    usable = (x.size // frame) * frame
    frames = x[:usable].reshape(-1, frame).astype(np.float64)
    level_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(frames**2, axis=1)), 1e-12))

    threshold = reference_dbfs - threshold_below_db
    below = np.minimum(level_db - threshold, 0.0)
    gain_db = np.maximum(below * (ratio - 1.0), -max_reduction_db)

    if COMPRESS_SMOOTH_FRAMES > 1 and gain_db.size >= COMPRESS_SMOOTH_FRAMES:
        kernel = np.ones(COMPRESS_SMOOTH_FRAMES) / COMPRESS_SMOOTH_FRAMES
        gain_db = np.convolve(gain_db, kernel, mode="same")

    centers = np.arange(gain_db.size) * frame + frame / 2.0
    sample_gain_db = np.interp(np.arange(x.size), centers, gain_db)
    return (x * (10.0 ** (sample_gain_db / 20.0))).astype(np.float32)


def compress(
    x: np.ndarray,
    sr: int,
    target_dbfs: float = -18.0,
    max_boost_db: float = DEFAULT_MAX_BOOST_DB,
    frame_ms: float = COMPRESS_FRAME_MS,
    floor_dbfs: float = COMPRESS_FLOOR_DBFS,
) -> np.ndarray:
    """Level out a clip so a short quiet burst isn't buried by one loud moment.

    An envelope-following gain rather than a textbook ratio/knee compressor:
    the goal is only "make the grinding audible", and a per-frame gain toward
    a target, smoothed and capped, is easier to reason about and cannot
    produce the pumping a badly-tuned release would.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    if x.size < frame * 2:
        return x

    usable = (x.size // frame) * frame
    frames = x[:usable].reshape(-1, frame).astype(np.float64)
    envelope = np.sqrt(np.mean(frames**2, axis=1))
    envelope_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))

    gain_db = np.clip(target_dbfs - envelope_db, 0.0, max_boost_db)
    gain_db[envelope_db < floor_dbfs] = 0.0

    if COMPRESS_SMOOTH_FRAMES > 1 and gain_db.size >= COMPRESS_SMOOTH_FRAMES:
        kernel = np.ones(COMPRESS_SMOOTH_FRAMES) / COMPRESS_SMOOTH_FRAMES
        gain_db = np.convolve(gain_db, kernel, mode="same")

    # Interpolate per-frame gain to per-sample so there are no steps at frame
    # boundaries, which would be audible as clicks.
    frame_centers = np.arange(gain_db.size) * frame + frame / 2.0
    sample_gain_db = np.interp(np.arange(x.size), frame_centers, gain_db)
    return (x * (10.0 ** (sample_gain_db / 20.0))).astype(np.float32)


def limit(
    x: np.ndarray,
    sr: int,
    ceiling_dbfs: float = DEFAULT_LIMIT_DBFS,
    frame_ms: float = 20.0,
) -> np.ndarray:
    """Pull down only the moments that exceed the ceiling.

    Scaling the whole clip by peak/ceiling — the obvious implementation — was
    measurably wrong here: one transient sets the peak, the entire clip gets
    divided by it, and the careful loudness normalization just above is undone.
    Measured that way the chain gained 10.8 dB; limiting locally instead keeps
    what the normalization achieved.

    Downward-only and frame-local, with the same interpolation as [compress]
    so no step lands mid-waveform as a click.
    """
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    frame = max(1, int(sr * frame_ms / 1000.0))
    if x.size < frame * 2:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        return (x * (ceiling / peak)).astype(np.float32) if peak > ceiling else x

    usable = (x.size // frame) * frame
    frames = np.abs(x[:usable].reshape(-1, frame))
    peaks = frames.max(axis=1)
    gain = np.where(peaks > ceiling, ceiling / np.maximum(peaks, 1e-12), 1.0)

    # Spread each reduction over its neighbours so a limited frame doesn't sit
    # next to an unlimited one at full level.
    gain = np.minimum.reduce([gain, np.roll(gain, 1), np.roll(gain, -1)])

    centers = np.arange(gain.size) * frame + frame / 2.0
    sample_gain = np.interp(np.arange(x.size), centers, gain)
    limited = (x * sample_gain).astype(np.float32)

    # Safety net for anything the frame grid missed.
    peak = float(np.max(np.abs(limited)))
    if peak > ceiling:
        limited = (limited * (ceiling / peak)).astype(np.float32)
    return limited


def enhance_for_listening(
    x: np.ndarray,
    sr: int,
    highpass_hz: float = DEFAULT_HIGHPASS_HZ,
    extra_gain_db: float = DEFAULT_EXTRA_GAIN_DB,
    limit_dbfs: float = DEFAULT_LIMIT_DBFS,
    use_compression: bool = False,
    use_expansion: bool = False,
    band_limit_output: bool = True,
    target_rms_dbfs: float | None = None,
) -> np.ndarray:
    """The full chain. Returns audio at the same sample rate and length.

    **No dynamics processing by default, and that is a measured result rather
    than an omission.** Scored across 50 real clips on grind-to-background
    contrast and the level of the 5-8 kHz hiss:

    | chain                    | contrast | hiss  |
    |--------------------------|----------|-------|
    | level only               | 12.4 dB  | +10.0 |
    | + band-limit 400-5000 Hz | 12.2 dB  |  -5.4 |
    | + expansion              |  8.5 dB  |  -8.1 |
    | + compression            |  3.6 dB  | +11.5 |

    Compression was the first attempt and was the worst option available: it
    lifts the room's hiss to the same level as the grinding and costs ~9 dB of
    contrast. Expansion was the obvious correction and measured *worse than
    doing nothing*, buying 2.7 dB of hiss for 3.7 dB of contrast because its
    gain reduction smears into the grind's own attack and decay.

    What worked was the dull option: keep the band the grinding occupies and
    discard the rest. Grinding peaks at 700-3000 Hz and is ~9 dB down by
    5 kHz, while hiss is broadband, so the top octave is nearly all hiss.

    Level is then set by **peak**, not RMS. Peak normalization failed
    originally only because the peak was a DC-shifted subsonic rumble; with
    that removed the peak *is* the grind transient, which is what should sit
    near full scale. ``extra_gain_db`` pushes past that for audibility and is
    the knob to turn if a recording is still too quiet or too hissy.
    """
    if x.size == 0:
        return x

    y = remove_dc(x)
    y = high_pass(y, sr, highpass_hz)
    if band_limit_output:
        y = band_limit(y, sr)

    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return y
    # Normalize to the ceiling first so everything downstream works in a known
    # range, then add the deliberate extra gain on top.
    y = (y * (10.0 ** (DEFAULT_TARGET_PEAK_DBFS / 20.0) / peak)).astype(np.float32)
    if extra_gain_db:
        y = (y * (10.0 ** (extra_gain_db / 20.0))).astype(np.float32)

    reference = target_rms_dbfs if target_rms_dbfs is not None else DEFAULT_TARGET_RMS_DBFS
    if use_expansion:
        y = expand(y, sr, reference_dbfs=reference)
    if use_compression:
        y = compress(y, sr, target_dbfs=reference)

    return limit(y, sr, limit_dbfs)
