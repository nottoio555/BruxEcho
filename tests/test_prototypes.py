"""Tests for src.embed.prototypes using synthetic vectors only.

Per project policy no real recordings are touched, and nothing here loads
TensorFlow — prototype scoring is plain linear algebra, so it is tested
without the model that normally produces the vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embed.prototypes import (
    build_prototypes,
    classify,
    rank_by_label,
    score,
    score_matrix,
)

DIM = 8


def _unit(*components: float) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for i, value in enumerate(components):
        vector[i] = value
    return vector


def test_centroid_is_unit_length_and_counts_samples():
    embeddings = {"grind": np.stack([_unit(1.0), _unit(0.9, 0.1)])}
    sources = {"grind": ["a.wav", "b.wav"]}

    prototypes = build_prototypes(embeddings, sources)

    assert len(prototypes) == 1
    proto = prototypes[0]
    assert proto.label == "grind"
    assert proto.n_samples == 2
    assert proto.source_files == ["a.wav", "b.wav"]
    assert np.linalg.norm(proto.centroid) == pytest.approx(1.0, abs=1e-5)


def test_centroid_ignores_magnitude_differences():
    """A loud reference must not dominate the centroid's direction.

    Embedding magnitude tracks loudness, and the reference clips were recorded
    ~34 dB louder than the night's clips (Sprint 3.5), so direction is the only
    comparable part.
    """
    quiet = np.stack([_unit(1.0), _unit(0.0, 1.0)])
    one_loud = np.stack([_unit(1.0), _unit(0.0, 50.0)])

    a = build_prototypes({"x": quiet})[0].centroid
    b = build_prototypes({"x": one_loud})[0].centroid

    assert np.allclose(a, b, atol=1e-5)


def test_score_is_cosine_similarity():
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0)])})

    similarities = score(_unit(3.0), prototypes)  # magnitude must not matter

    assert similarities["grind"] == pytest.approx(1.0, abs=1e-5)
    assert similarities["snore"] == pytest.approx(0.0, abs=1e-5)


def test_classify_returns_nearest_label_and_margin():
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0)])})

    result = classify(_unit(1.0, 0.2), prototypes)

    assert result.label == "grind"
    assert result.runner_up == "snore"
    assert result.margin > 0.0
    assert result.margin == pytest.approx(result.similarity - score(_unit(1.0, 0.2), prototypes)["snore"], abs=1e-6)


def test_classify_margin_is_small_for_an_ambiguous_vector():
    """Equidistant input => ~zero margin, which is the point of reporting it."""
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0)])})

    result = classify(_unit(1.0, 1.0), prototypes)

    assert result.margin == pytest.approx(0.0, abs=1e-5)


def test_labels_with_too_few_samples_are_skipped():
    embeddings = {"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0), _unit(0.0, 0.9)])}

    prototypes = build_prototypes(embeddings, min_samples=2)

    assert [p.label for p in prototypes] == ["snore"]


def test_score_matrix_matches_per_row_score():
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0)])})
    rows = np.stack([_unit(1.0), _unit(0.0, 1.0), _unit(1.0, 1.0)])

    labels, matrix = score_matrix(rows, prototypes)

    assert labels == ["grind", "snore"]
    for i in range(rows.shape[0]):
        per_row = score(rows[i], prototypes)
        for j, label in enumerate(labels):
            assert matrix[i, j] == pytest.approx(per_row[label], abs=1e-6)


def test_rank_by_label_orders_most_similar_first():
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)]), "snore": np.stack([_unit(0.0, 1.0)])})
    rows = np.stack([_unit(0.0, 1.0), _unit(1.0), _unit(1.0, 1.0)])

    labels, matrix = score_matrix(rows, prototypes)
    order = rank_by_label(matrix, labels, "grind")

    assert order[0] == 1  # the pure-grind row ranks first
    assert order[-1] == 0  # the pure-snore row ranks last


def test_rank_by_label_rejects_unknown_label():
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)])})
    labels, matrix = score_matrix(np.stack([_unit(1.0)]), prototypes)

    with pytest.raises(ValueError):
        rank_by_label(matrix, labels, "speech")


def test_zero_vector_scores_without_nan():
    """A silent/degenerate clip must score 0, not NaN — NaN would sort unpredictably."""
    prototypes = build_prototypes({"grind": np.stack([_unit(1.0)])})

    similarities = score(np.zeros(DIM, dtype=np.float32), prototypes)

    assert similarities["grind"] == pytest.approx(0.0, abs=1e-6)
