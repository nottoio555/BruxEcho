"""Nearest-centroid (prototypical) classification over audio embeddings.

Pure library code: no plotting, no CLI, no printing, no logging.

## Why prototypes instead of a trained classifier

There are ~6 confirmed sleep-bruxism clips. Any classifier trained on that
would mostly memorize the recording conditions of one night from one person.
A prototype is just the centroid of the reference embeddings — nothing is
fitted, nothing can overfit in the usual sense, and adding a newly confirmed
clip means recomputing one mean rather than retraining.

The honest limitation is the same small number: a centroid of 6 clips from a
single subject describes *those recordings* as much as it describes
"bruxism". Treat the output as a ranking to listen through, not a verdict —
see ``tools/evaluate_similar.py`` for how well that ranking actually holds up
under leave-one-out.

## Why cosine similarity

YAMNet embeddings are not normalized, and their magnitude tracks loudness,
which is exactly the nuisance variable here: the reference clips and the
night's candidate clips were recorded at wildly different gains (Sprint 3.5
measured ~34 dB between the two clip sets). Cosine similarity compares
direction only, so it ignores that offset. Centroids are therefore built
from L2-normalized vectors and re-normalized afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.embed.encoder import l2_normalize


@dataclass
class Prototype:
    """The centroid of one label's reference embeddings."""

    label: str
    centroid: np.ndarray
    n_samples: int
    source_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Classification:
    """Nearest prototype for one embedding, with a confidence proxy."""

    label: str
    similarity: float
    margin: float  # best minus runner-up; small margin = the two classes look alike
    runner_up: str | None


def build_prototypes(
    embeddings_by_label: dict[str, np.ndarray],
    sources_by_label: dict[str, list[str]] | None = None,
    min_samples: int = 1,
) -> list[Prototype]:
    """Build one centroid per label.

    Args:
        embeddings_by_label: ``{label: (n_samples, dim)}`` raw (un-normalized)
            embeddings.
        sources_by_label: Optional ``{label: [filename, ...]}`` recorded on the
            prototype so a result can always be traced back to what defined it.
        min_samples: Labels with fewer samples than this are skipped — a
            "class" of one clip is a single recording, not a category.

    Returns:
        Prototypes sorted by label. Labels with no usable rows are omitted.
    """
    prototypes: list[Prototype] = []
    for label in sorted(embeddings_by_label):
        matrix = np.atleast_2d(np.asarray(embeddings_by_label[label], dtype=np.float32))
        if matrix.size == 0 or matrix.shape[0] < min_samples:
            continue
        # Normalize first, then average: otherwise a single loud reference clip
        # (larger vector magnitude) would dominate the centroid's direction.
        centroid = l2_normalize(l2_normalize(matrix).mean(axis=0))[0]
        prototypes.append(
            Prototype(
                label=label,
                centroid=centroid.astype(np.float32),
                n_samples=int(matrix.shape[0]),
                source_files=list((sources_by_label or {}).get(label, [])),
            )
        )
    return prototypes


def score(embedding: np.ndarray, prototypes: list[Prototype]) -> dict[str, float]:
    """Cosine similarity between one embedding and every prototype."""
    if not prototypes:
        return {}
    vector = l2_normalize(np.asarray(embedding, dtype=np.float32))[0]
    return {p.label: float(np.dot(vector, p.centroid)) for p in prototypes}


def score_matrix(embeddings: np.ndarray, prototypes: list[Prototype]) -> tuple[list[str], np.ndarray]:
    """Vectorized :func:`score` for many embeddings at once.

    Returns:
        ``(labels, similarities)`` where ``similarities`` is
        ``(n_embeddings, n_prototypes)`` aligned to ``labels``.
    """
    labels = [p.label for p in prototypes]
    if not prototypes:
        return labels, np.zeros((len(np.atleast_2d(embeddings)), 0), dtype=np.float32)
    centroids = np.vstack([p.centroid for p in prototypes]).astype(np.float32)
    return labels, (l2_normalize(embeddings) @ centroids.T).astype(np.float32)


def classify(embedding: np.ndarray, prototypes: list[Prototype]) -> Classification:
    """Nearest prototype plus the gap to the runner-up.

    The margin matters more than the raw similarity here: YAMNet embeddings
    of quiet night-time audio are all fairly close together, so an absolute
    similarity of 0.8 means little, while "0.8 for grind vs 0.79 for snore"
    means the two are effectively indistinguishable to this model.
    """
    similarities = score(embedding, prototypes)
    if not similarities:
        return Classification(label="", similarity=float("nan"), margin=float("nan"), runner_up=None)
    ranked = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_similarity = ranked[0]
    runner_up_label, runner_up_similarity = ranked[1] if len(ranked) > 1 else (None, None)
    margin = best_similarity - runner_up_similarity if runner_up_similarity is not None else float("nan")
    return Classification(
        label=best_label,
        similarity=float(best_similarity),
        margin=float(margin),
        runner_up=runner_up_label,
    )


def rank_by_label(
    similarities: np.ndarray, labels: list[str], target_label: str
) -> np.ndarray:
    """Indices ordered by similarity to ``target_label``, most similar first.

    This is the tool's actual product: a listening order. Raises if the label
    is absent rather than silently ranking by something else.
    """
    if target_label not in labels:
        raise ValueError(f"ไม่มี prototype ชื่อ {target_label!r} (มี: {labels})")
    column = similarities[:, labels.index(target_label)]
    return np.argsort(-column, kind="stable")
