"""Frozen YAMNet + a small MLP head — embedding extraction and training.

Pure library code: no plotting, no CLI, no printing, no logging (the CLI in
``tools/train_model.py`` owns all of that).

## Why a frozen backbone and a 2-layer head

YAMNet was trained on AudioSet, so its embeddings already encode "what kind
of sound is this". With 1,886 bruxism events, fine-tuning 3.7M backbone
parameters would fit the four sleepers in this dataset rather than the
sound. Freezing it leaves ~270k trainable parameters in the head, which is
a defensible ratio to the data — and means embeddings can be cached once
and reused across every training run, turning a 17-minute job into seconds.

## Why training samples are frames, not events

The night-time pipeline slides a 0.96s window (0.48s hop) and classifies
each one, so training on whole variable-length events would optimize
something the model is never asked to do at inference. Each YAMNet frame
inside an annotated event therefore becomes one training sample carrying
that event's label. It also multiplies 39,592 events into ~144,000 samples.

The cost is label noise: a frame near the edge of a 0.86s bruxism event
(the median) covers audio that is partly background. Nothing here tries to
correct that — it is the honest consequence of frame-level supervision from
event-level annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.embed.encoder import EMBEDDING_DIM
from src.train.dataset import CLASS_NAMES, Segment, config_fingerprint, read_segment_audio

CACHE_VERSION = "frames-v1"


@dataclass
class FrameDataset:
    """Frame-level embeddings plus everything needed to split and weight them."""

    embeddings: np.ndarray  # (n_frames, 1024)
    labels: np.ndarray  # (n_frames,) int class indices
    sessions: np.ndarray  # (n_frames,) str
    subjects: np.ndarray  # (n_frames,) str
    segment_index: np.ndarray  # (n_frames,) int — which Segment each frame came from
    class_names: tuple[str, ...] = CLASS_NAMES

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def counts(self) -> dict[str, int]:
        return {
            name: int(np.sum(self.labels == i)) for i, name in enumerate(self.class_names)
        }


@dataclass
class TrainingArtifacts:
    """Everything a trained head needs at inference time."""

    model: object  # keras.Model
    mean: np.ndarray
    std: np.ndarray
    class_names: tuple[str, ...]
    history: dict[str, list[float]] = field(default_factory=dict)


def cache_path_for(cache_dir: Path, segmentation_tsv: Path, n_segments: int) -> Path:
    """Cache file keyed by the TSV's identity plus this module's version."""
    stat = segmentation_tsv.stat()
    fingerprint = config_fingerprint(
        CACHE_VERSION, segmentation_tsv.name, stat.st_size, int(stat.st_mtime), n_segments
    )
    return cache_dir / f"frames_{fingerprint}.npz"


def extract_frame_dataset(
    segments: list[Segment],
    audio_dir: Path,
    encoder,
    cache_path: Path | None = None,
    on_progress=None,
) -> FrameDataset:
    """Embed every segment into per-frame YAMNet vectors, with a bulk cache.

    One ``.npz`` for the whole dataset rather than a file per segment: at
    ~144,000 frames the per-file overhead would dominate, and the set is
    only ever consumed whole.

    Args:
        on_progress: Optional ``(done, total)`` callback — this module must
            not print, so progress is reported outward.
    """
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return FrameDataset(
                embeddings=data["embeddings"],
                labels=data["labels"],
                sessions=data["sessions"].astype(str),
                subjects=data["subjects"].astype(str),
                segment_index=data["segment_index"],
            )

    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    blocks: list[np.ndarray] = []
    labels: list[int] = []
    sessions: list[str] = []
    subjects: list[str] = []
    segment_index: list[int] = []

    total = len(segments)
    for i, segment in enumerate(segments):
        try:
            audio, sr = read_segment_audio(audio_dir, segment)
            if audio.size < 160:  # < 10 ms of audio
                continue
            frames = encoder.embed_frames(audio, sr)
        except Exception:  # noqa: BLE001 — one unreadable file must not kill a 17-minute run
            if on_progress is not None:
                on_progress(i + 1, total)
            continue
        n = int(frames.shape[0])
        if n == 0:
            continue
        blocks.append(frames.astype(np.float32))
        labels.extend([class_index[segment.label]] * n)
        sessions.extend([segment.session] * n)
        subjects.extend([segment.subject] * n)
        segment_index.extend([i] * n)
        if on_progress is not None:
            on_progress(i + 1, total)

    dataset = FrameDataset(
        embeddings=np.vstack(blocks).astype(np.float32)
        if blocks
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        sessions=np.asarray(sessions, dtype=object).astype(str),
        subjects=np.asarray(subjects, dtype=object).astype(str),
        segment_index=np.asarray(segment_index, dtype=np.int32),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=dataset.embeddings,
            labels=dataset.labels,
            sessions=dataset.sessions,
            subjects=dataset.subjects,
            segment_index=dataset.segment_index,
        )
    return dataset


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Z-score using training statistics only (never the validation set)."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = [(matrix - mean) / std for matrix in others]
    return ((train - mean) / std).astype(np.float32), mean.astype(np.float32), [
        m.astype(np.float32) for m in scaled
    ]


def class_weights(labels: np.ndarray, n_classes: int = len(CLASS_NAMES)) -> dict[int, float]:
    """Inverse-frequency weights.

    Bruxism is ~2% of frames; without this the cheapest way to a low loss is
    to never predict it.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_head(input_dim: int = EMBEDDING_DIM, n_classes: int = len(CLASS_NAMES), seed: int = 0):
    """Two hidden layers on top of the frozen embedding."""
    import keras  # noqa: PLC0415 — lazy, keeps TensorFlow off module import

    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="bruxism_head",
    )


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    verbose: int = 0,
) -> TrainingArtifacts:
    """Fit the head on standardized embeddings, weighted against imbalance."""
    import keras  # noqa: PLC0415

    x_train_scaled, mean, (x_val_scaled,) = standardize(x_train, x_val)
    model = build_head(input_dim=x_train.shape[1], seed=seed)
    std = np.where(x_train.std(axis=0) < 1e-6, 1.0, x_train.std(axis=0)).astype(np.float32)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=verbose,
    )
    return TrainingArtifacts(
        model=model,
        mean=mean,
        std=std,
        class_names=CLASS_NAMES,
        history={k: [float(v) for v in vs] for k, vs in history.history.items()},
    )


def save_artifacts(artifacts: TrainingArtifacts, model_dir: Path, extra_metadata: dict | None = None) -> None:
    """Write the head, the normalization stats, and the metadata to load them."""
    import json  # noqa: PLC0415

    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.model.save(model_dir / "head.keras")
    np.savez(model_dir / "normalization.npz", mean=artifacts.mean, std=artifacts.std)
    metadata = {
        "class_names": list(artifacts.class_names),
        "embedding": "yamnet",
        "embedding_dim": int(artifacts.mean.shape[0]),
        "cache_version": CACHE_VERSION,
        **(extra_metadata or {}),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (model_dir / "history.json").write_text(
        json.dumps(artifacts.history, indent=2), encoding="utf-8"
    )


def load_artifacts(model_dir: Path) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """Load ``(model, mean, std, metadata)`` saved by :func:`save_artifacts`."""
    import json  # noqa: PLC0415

    import keras  # noqa: PLC0415

    model = keras.models.load_model(model_dir / "head.keras")
    with np.load(model_dir / "normalization.npz") as data:
        mean, std = data["mean"], data["std"]
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, mean, std, metadata


def predict_proba(model, mean: np.ndarray, std: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Class probabilities for raw (un-standardized) YAMNet embeddings."""
    scaled = ((np.atleast_2d(embeddings) - mean) / std).astype(np.float32)
    return np.asarray(model.predict(scaled, verbose=0), dtype=np.float32)
