"""Spectral and energy feature computations for audio signals.

Pure library code: no plotting, no CLI, no printing, no side effects.
Every function takes arrays (and plain parameters) and returns values.
"""

from __future__ import annotations

from typing import Sequence

import librosa
import numpy as np
from scipy import signal as sp_signal

# Floor used to avoid log10(0) when a signal is silent.
_EPS = 1e-12

# (label, low_hz, high_hz) — non-overlapping bands that sum to 100% of energy.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-500Hz", 0.0, 500.0),
    ("500-2kHz", 500.0, 2000.0),
    ("2k-4kHz", 2000.0, 4000.0),
    ("4k-8kHz", 4000.0, 8000.0),
    ("8k-16kHz", 8000.0, 16000.0),
    ("16kHz+", 16000.0, float("inf")),
)

# (label, low_hz, high_hz) — bands used for per-band SNR, starting above the
# default high-pass cutoff so low-frequency environmental hum doesn't leak in.
SNR_BANDS: tuple[tuple[str, float, float], ...] = (
    ("300_2k", 300.0, 2000.0),
    ("2k_4k", 2000.0, 4000.0),
    ("4k_8k", 4000.0, 8000.0),
    ("8k_16k", 8000.0, 16000.0),
    ("16k_plus", 16000.0, float("inf")),
)


def rms_dbfs(x: np.ndarray) -> float:
    """Compute RMS level in dBFS (0 dBFS = full-scale amplitude of 1.0).

    Args:
        x: Mono signal, samples nominally in [-1, 1].

    Returns:
        RMS level in dBFS. Silence is floored at ``20*log10(_EPS)`` instead
        of ``-inf``.
    """
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, _EPS))


def peak_dbfs(x: np.ndarray) -> float:
    """Compute peak level in dBFS.

    Args:
        x: Mono signal, samples nominally in [-1, 1].

    Returns:
        Peak absolute sample value expressed in dBFS.
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(max(peak, _EPS))


def snr_db(signal_rms_dbfs: float, noise_rms_dbfs: float) -> float:
    """Signal-to-noise ratio in dB from two RMS levels already in dBFS.

    Args:
        signal_rms_dbfs: RMS level of the signal-of-interest clip.
        noise_rms_dbfs: RMS level of the noise/baseline (e.g. silence) clip.

    Returns:
        ``signal_rms_dbfs - noise_rms_dbfs``, in dB.
    """
    return signal_rms_dbfs - noise_rms_dbfs


def power_spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-signal power spectrum via a single real FFT.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.

    Returns:
        Tuple of ``(freqs_hz, power)``, both 1-D arrays of equal length.
    """
    if x.size == 0:
        return np.array([]), np.array([])
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    power = np.square(np.abs(spectrum))
    return freqs, power


def band_energy_ratios(
    x: np.ndarray,
    sr: int,
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
) -> dict[str, float]:
    """Percentage of total spectral energy falling into each frequency band.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        bands: Sequence of ``(label, low_hz, high_hz)`` ranges.

    Returns:
        Mapping of band label to percentage (0-100) of total energy.
    """
    freqs, power = power_spectrum(x, sr)
    total = float(power.sum()) if power.size else 0.0
    if total <= 0.0:
        return {label: 0.0 for label, _, _ in bands}
    ratios: dict[str, float] = {}
    for label, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        ratios[label] = float(power[mask].sum() / total * 100.0)
    return ratios


def high_freq_ratio(x: np.ndarray, sr: int, cutoff_hz: float = 8000.0) -> float:
    """Fraction of total spectral energy that lies above ``cutoff_hz``.

    This is the single most decision-relevant metric for this project: it
    tells us whether a 16 kHz-native model (YAMNet, which only "sees" up to
    8 kHz) is throwing away meaningful signal.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        cutoff_hz: Frequency threshold in Hz.

    Returns:
        Ratio in [0, 1] — multiply by 100 for a percentage.
    """
    freqs, power = power_spectrum(x, sr)
    total = float(power.sum()) if power.size else 0.0
    if total <= 0.0:
        return 0.0
    return float(power[freqs > cutoff_hz].sum() / total)


def band_power_db_dict(
    x: np.ndarray,
    sr: int,
    bands: Sequence[tuple[str, float, float]] = SNR_BANDS,
) -> dict[str, float]:
    """Absolute power (in dB) contained in each frequency band.

    Unlike ``band_energy_ratios``, this is not normalized against the
    signal's own total energy — it's meant to be compared across two
    different signals (e.g. a clip vs. a baseline) to get a per-band SNR.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        bands: Sequence of ``(label, low_hz, high_hz)`` ranges.

    Returns:
        Mapping of band label to power in dB (10*log10 of summed power,
        floored to avoid log(0)).
    """
    freqs, power = power_spectrum(x, sr)
    result: dict[str, float] = {}
    for label, lo, hi in bands:
        if power.size == 0:
            result[label] = 10.0 * np.log10(_EPS)
            continue
        mask = (freqs >= lo) & (freqs < hi)
        band_power = float(power[mask].sum())
        result[label] = 10.0 * np.log10(max(band_power, _EPS))
    return result


def power_to_db(power: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    """Convert a linear power array to dB, flooring to avoid log(0).

    Args:
        power: Linear power values (e.g. a PSD).
        floor: Minimum value substituted for entries at or below zero.

    Returns:
        ``10 * log10(max(power, floor))``, element-wise.
    """
    return 10.0 * np.log10(np.maximum(power, floor))


def high_pass_filter(
    x: np.ndarray, sr: int, cutoff_hz: float = 300.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter.

    Used to remove low-frequency environmental hum (e.g. HVAC rumble) that
    would otherwise dominate whole-signal energy metrics and mask real
    signal in higher bands.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz. ``<= 0`` disables filtering
            (returns ``x`` unchanged).
        order: Filter order.

    Returns:
        Filtered signal, same shape and dtype family as ``x``. Falls back
        to a causal (non-zero-phase) filter if ``x`` is too short for
        zero-phase filtering's padding requirement.
    """
    if x.size == 0 or cutoff_hz <= 0.0:
        return x
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        return x
    sos = sp_signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    try:
        y = sp_signal.sosfiltfilt(sos, x)
    except ValueError:
        y = sp_signal.sosfilt(sos, x)
    return y.astype(np.float32)


def detect_narrow_band_dips(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    window_hz: float = 200.0,
    drop_threshold_db: float = 10.0,
) -> list[float]:
    """Find frequencies where power drops sharply relative to their neighborhood.

    Flags a narrow local notch rather than a broad rolloff: each bin is
    compared only against nearby bins (within ``window_hz``), not the whole
    spectrum, so a genuine wideband high-frequency rolloff won't trigger it.

    Args:
        freqs: Frequency bins in Hz (e.g. from ``welch_psd``).
        psd_db: PSD in dB, same length as ``freqs``.
        window_hz: Half-width of the neighborhood used as the local reference.
        drop_threshold_db: Minimum drop below the local neighborhood median
            to flag a bin as a dip.

    Returns:
        Frequencies (Hz) flagged as narrow dips.
    """
    dips: list[float] = []
    if freqs.size == 0:
        return dips
    lo_edge, hi_edge = freqs[0] + window_hz, freqs[-1] - window_hz
    for i, f in enumerate(freqs):
        # Skip bins whose neighborhood would be one-sided (near DC or
        # Nyquist) — a truncated window isn't a fair local reference and
        # falsely flags the natural spectral shape at the edges.
        if f < lo_edge or f > hi_edge:
            continue
        mask = (freqs >= f - window_hz) & (freqs <= f + window_hz)
        mask[i] = False
        if mask.sum() < 4:
            continue
        neighborhood_median = float(np.median(psd_db[mask]))
        if neighborhood_median - psd_db[i] > drop_threshold_db:
            dips.append(float(f))
    return dips


def find_common_frequency_dips(
    dip_lists: Sequence[Sequence[float]],
    bin_hz: float = 200.0,
    min_fraction: float = 1.0,
) -> list[float]:
    """Frequencies flagged as a narrow dip in at least ``min_fraction`` of clips.

    Dips are binned to the nearest ``bin_hz`` before comparing across clips,
    since the exact bin of a codec/resample artifact can shift slightly
    between recordings.

    Args:
        dip_lists: One list of dip frequencies (Hz) per clip.
        bin_hz: Bin width used to align dips across clips.
        min_fraction: Minimum fraction of clips a (binned) frequency must
            appear in to be reported.

    Returns:
        Sorted list of binned frequencies (Hz) common across clips.
    """
    if not dip_lists:
        return []
    counts: dict[float, int] = {}
    for dips in dip_lists:
        binned = {round(f / bin_hz) * bin_hz for f in dips}
        for f in binned:
            counts[f] = counts.get(f, 0) + 1
    threshold = max(1, int(np.ceil(min_fraction * len(dip_lists))))
    return sorted(f for f, c in counts.items() if c >= threshold)


def welch_psd(
    x: np.ndarray, sr: int, nperseg: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density estimate via Welch's method.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        nperseg: Segment length for Welch's method; clamped to signal length.

    Returns:
        Tuple of ``(freqs_hz, psd)``.
    """
    if x.size == 0:
        return np.array([]), np.array([])
    seg = min(nperseg, x.size)
    freqs, psd = sp_signal.welch(x, fs=sr, nperseg=seg)
    return freqs, psd


def log_mel_spectrogram(
    x: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-power mel spectrogram.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        n_mels: Number of mel bands.
        n_fft: FFT window size; clamped to signal length.
        hop_length: Hop size in samples between frames.

    Returns:
        Tuple of ``(log_mel_db, frame_times_s, mel_band_freqs_hz)`` where
        ``log_mel_db`` has shape ``(n_mels, n_frames)``.
    """
    n_fft = min(n_fft, x.size) if x.size else n_fft
    mel = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(
        np.arange(log_mel_db.shape[1]), sr=sr, hop_length=hop_length
    )
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    return log_mel_db, times, mel_freqs


def spectral_centroid_stats(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Mean and standard deviation of the frame-wise spectral centroid.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.

    Returns:
        Tuple of ``(mean_hz, std_hz)``.
    """
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    return float(np.mean(centroid)), float(np.std(centroid))


def spectral_rolloff_mean(x: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    """Mean frame-wise spectral rolloff frequency.

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        roll_percent: Fraction of total energy below the rolloff frequency.

    Returns:
        Mean rolloff frequency in Hz.
    """
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=roll_percent)[0]
    return float(np.mean(rolloff))


def spectral_flatness_mean(x: np.ndarray) -> float:
    """Mean frame-wise spectral flatness (0 = tonal, 1 = noise-like).

    Args:
        x: Mono signal.

    Returns:
        Mean spectral flatness.
    """
    flatness = librosa.feature.spectral_flatness(y=x)[0]
    return float(np.mean(flatness))


def zero_crossing_rate_mean(x: np.ndarray) -> float:
    """Mean frame-wise zero-crossing rate.

    Args:
        x: Mono signal.

    Returns:
        Mean zero-crossing rate.
    """
    zcr = librosa.feature.zero_crossing_rate(x)[0]
    return float(np.mean(zcr))


def rms_envelope(
    x: np.ndarray, frame_length: int = 2048, hop_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise RMS envelope, useful for visualizing temporal patterns.

    Args:
        x: Mono signal.
        frame_length: Analysis frame length in samples.
        hop_length: Hop size in samples between frames.

    Returns:
        Tuple of ``(frame_times_s, rms_values)`` — ``frame_times_s`` requires
        the caller to know ``sr``; pass it separately when plotting.
    """
    frame_length = min(frame_length, x.size) if x.size else frame_length
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.arange(rms.size)
    return frames * hop_length, rms
