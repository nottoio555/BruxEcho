"""Audio file I/O helpers.

Pure library code: no plotting, no CLI, no printing. Every function takes
paths/arrays and returns values — callers decide what to do with them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".flac")


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32, preserving its native sample rate.

    Stereo (or multi-channel) files are downmixed to mono by averaging all
    channels. No resampling is performed under any circumstance — the caller
    receives the file's native sample rate as-is.

    Args:
        path: Path to a .wav or .flac file.

    Returns:
        Tuple of ``(signal, sample_rate)`` where ``signal`` is a 1-D float32
        array and ``sample_rate`` is in Hz.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def save_audio(path: Path, x: np.ndarray, sr: int) -> None:
    """Write a signal to disk as a WAV/FLAC file.

    Args:
        path: Output file path. Its parent directory must already exist.
        x: Signal samples (mono or multi-channel).
        sr: Sample rate in Hz.
    """
    sf.write(str(path), x, sr)


def trim_start(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Drop the first ``seconds`` of a signal (e.g. a record-button click).

    Args:
        x: Mono signal.
        sr: Sample rate in Hz.
        seconds: Duration to remove from the start. Clamped to the signal's
            length — a signal shorter than this becomes empty.

    Returns:
        The signal with its start removed. Returns ``x`` unchanged if
        ``seconds <= 0``.
    """
    if seconds <= 0.0:
        return x
    n = min(x.size, int(round(seconds * sr)))
    return x[n:]


def find_audio_files(directory: Path) -> list[Path]:
    """List supported audio files directly inside a directory.

    Args:
        directory: Directory to scan (non-recursive).

    Returns:
        Paths with a ``.wav`` or ``.flac`` suffix, sorted by filename.
    """
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
