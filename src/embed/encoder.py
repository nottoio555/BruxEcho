"""YAMNet audio embeddings, with a content-addressed disk cache.

Pure library code: no plotting, no CLI, no printing, no logging.

## Why YAMNet and not PANNs

Sprint 1 picked PANNs because the 8-16 kHz band carried the signal, and
YAMNet's 16 kHz sample rate can only see up to 8 kHz. Sprint 3.5 measured
*real sleep* bruxism instead of deliberate tooth-rubbing and found the
opposite: 90% of the energy sits inside 683-5,104 Hz, with a median 0.5% of
energy above 8 kHz. Everything that matters is therefore below YAMNet's
ceiling, which makes it the better choice here — ~20x smaller than PANNs and
available as TFLite, so the same embedding can eventually run on-device in
the Android app rather than only in this toolkit.

## Why embeddings at all (instead of training a classifier)

There are ~6 confirmed sleep-bruxism clips. That is far too few to train
anything; it is enough to define a *prototype* to compare against. This
module only produces vectors — the comparison lives in ``prototypes``.

## Pooling and its main limitation

YAMNet emits one 1024-d vector per ~0.48s frame (0.24s hop). ``embed_clip``
mean-pools over time by default, which is standard but **dilutes short
events**: a 5s candidate clip holding a 0.7s grind is ~86% background, and
the mean is dominated by that background. ``pooling="max"`` keeps the
per-dimension peak instead, which is less diluted but noisier. Both poolings
are cached, so switching costs nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import librosa
import numpy as np

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000
EMBEDDING_DIM = 1024

Pooling = Literal["mean", "max"]

# Bumped whenever anything that changes the stored vectors changes (model
# handle, resampling, pooling maths). Part of every cache key, so a stale
# cache can never be silently mixed with fresh vectors.
CACHE_VERSION = "yamnet-v1"


@dataclass(frozen=True)
class ClipEmbedding:
    """Both poolings of one clip, so callers can switch without re-encoding."""

    mean: np.ndarray
    max: np.ndarray
    n_frames: int

    def pooled(self, pooling: Pooling = "mean") -> np.ndarray:
        return self.mean if pooling == "mean" else self.max


def _load_yamnet():
    """Import TensorFlow lazily so merely importing this module stays cheap."""
    import tensorflow_hub as hub  # noqa: PLC0415 — deliberate lazy import

    return hub.load(YAMNET_HANDLE)


def to_model_input(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at YAMNet's 16 kHz, the only input shape the model accepts.

    Args:
        audio: Mono or multi-channel signal.
        sr: Sample rate of ``audio`` in Hz.

    Returns:
        1-D float32 array at 16 kHz.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != YAMNET_SAMPLE_RATE and x.size:
        x = librosa.resample(x, orig_sr=sr, target_sr=YAMNET_SAMPLE_RATE)
    return np.ascontiguousarray(x, dtype=np.float32)


def file_cache_key(path: Path, cache_version: str = CACHE_VERSION) -> str:
    """Content hash of a file, namespaced by encoder version.

    Keyed on bytes rather than path+mtime so re-running detection over
    regenerated clips (identical audio, new timestamps) reuses the cache,
    while genuinely different audio never collides.
    """
    digest = hashlib.sha256()
    digest.update(cache_version.encode("utf-8"))
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YamnetEncoder:
    """Encodes audio into 1024-d YAMNet embeddings, caching results on disk.

    Usage::

        encoder = YamnetEncoder(cache_dir=Path("results/embed_cache"))
        ids, matrix = encoder.embed_batch(sorted(clip_dir.glob("*.wav")))

    The model itself is loaded on first use, not at construction, so building
    an encoder (e.g. to inspect its cache) never pays TensorFlow's startup.
    """

    def __init__(self, cache_dir: Path | None = None, cache_version: str = CACHE_VERSION):
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self._model = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            self._model = _load_yamnet()
        return self._model

    def embed_frames(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Per-frame embeddings, shape ``(n_frames, 1024)``.

        Raises:
            ValueError: if ``audio`` has no samples — an empty clip has no
                meaningful embedding and silently returning zeros would make
                it look maximally dissimilar to everything instead of broken.
        """
        x = to_model_input(audio, sr)
        if x.size == 0:
            raise ValueError("ไม่มีข้อมูลเสียงให้ encode (คลิปว่าง)")
        _scores, embeddings, _spectrogram = self.model(x)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_clip(self, audio: np.ndarray, sr: int, pooling: Pooling = "mean") -> np.ndarray:
        """One 1024-d vector for a clip, pooled over time (see module docstring)."""
        return self.embed_clip_full(audio, sr).pooled(pooling)

    def embed_clip_full(self, audio: np.ndarray, sr: int) -> ClipEmbedding:
        frames = self.embed_frames(audio, sr)
        return ClipEmbedding(
            mean=frames.mean(axis=0).astype(np.float32),
            max=frames.max(axis=0).astype(np.float32),
            n_frames=int(frames.shape[0]),
        )

    def embed_audio(self, audio: np.ndarray, sr: int, cache_key: str | None = None) -> ClipEmbedding:
        """Embed an in-memory signal, optionally cached under an explicit key.

        Lets callers cache derived audio (e.g. a clip cropped to its event
        window) that has no file of its own — the caller owns the key, so it
        can encode whatever transform it applied.
        """
        cache_path = None
        if cache_key is not None and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{cache_key}.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as data:
                        return ClipEmbedding(
                            mean=data["mean"], max=data["max"], n_frames=int(data["n_frames"])
                        )
                except (OSError, KeyError, ValueError):
                    cache_path.unlink(missing_ok=True)

        embedding = self.embed_clip_full(audio, sr)
        if cache_path is not None:
            np.savez(
                cache_path,
                mean=embedding.mean,
                max=embedding.max,
                n_frames=np.int32(embedding.n_frames),
            )
        return embedding

    def cache_key_for(self, path: Path, suffix: str = "") -> str:
        """Cache key for a file, optionally namespaced by a transform description."""
        key = file_cache_key(path, self.cache_version)
        return f"{key}_{suffix}" if suffix else key

    def embed_file(self, path: Path, use_cache: bool = True) -> ClipEmbedding:
        """Embed one audio file, reading from / writing to the disk cache."""
        cache_path = None
        if use_cache and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{file_cache_key(path, self.cache_version)}.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as data:
                        return ClipEmbedding(
                            mean=data["mean"], max=data["max"], n_frames=int(data["n_frames"])
                        )
                except (OSError, KeyError, ValueError):
                    # A truncated/corrupt cache entry (interrupted write) must
                    # not poison the run — recompute and overwrite it.
                    cache_path.unlink(missing_ok=True)

        import soundfile as sf  # noqa: PLC0415 — lazy, keeps import cost off module load

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        embedding = self.embed_clip_full(audio, sr)

        if cache_path is not None:
            np.savez(
                cache_path,
                mean=embedding.mean,
                max=embedding.max,
                n_frames=np.int32(embedding.n_frames),
            )
        return embedding

    def embed_batch(
        self,
        paths: Sequence[Path],
        pooling: Pooling = "mean",
        use_cache: bool = True,
        on_progress: "callable | None" = None,
    ) -> tuple[list[str], np.ndarray]:
        """Embed many files.

        Args:
            paths: Audio files to encode.
            pooling: Which pooling to return in the matrix.
            use_cache: Whether to read/write the disk cache.
            on_progress: Optional ``(done, total, path)`` callback — the only
                way this module reports progress, since it must not print.

        Returns:
            ``(ids, matrix)`` where ``ids`` are file stems and ``matrix`` is
            ``(n_files, 1024)``. Files that fail to decode are skipped, so
            ``ids`` may be shorter than ``paths``.
        """
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        total = len(paths)
        for i, path in enumerate(paths, start=1):
            try:
                embedding = self.embed_file(path, use_cache=use_cache)
            except Exception:  # noqa: BLE001 — one unreadable clip shouldn't abort a 2,500-clip run
                if on_progress is not None:
                    on_progress(i, total, path)
                continue
            ids.append(path.stem)
            vectors.append(embedding.pooled(pooling))
            if on_progress is not None:
                on_progress(i, total, path)
        matrix = (
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        return ids, matrix


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero instead of becoming NaN."""
    array = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, eps)).astype(np.float32)


def iter_audio_files(directory: Path, patterns: Iterable[str] = ("*.wav", "*.flac")) -> list[Path]:
    """Sorted audio files directly inside ``directory`` (non-recursive)."""
    found: list[Path] = []
    for pattern in patterns:
        found.extend(directory.glob(pattern))
    return sorted(set(found))
