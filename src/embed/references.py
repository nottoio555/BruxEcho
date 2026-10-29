"""Assembling the labelled reference clips that define each prototype.

Pure library code: no plotting, no CLI, no printing. Uses the stdlib ``csv``
module rather than pandas to keep ``src/`` free of the analysis stack.

## Where each class's references come from

- **grind** — ``data/raw/grind_clips/sleep_*.wav``, the clips confirmed by
  ear as real sleep bruxism.
- **snore / other** — clips from a night's ``results/events/<id>/clips/``,
  matched to the human labels in ``labels.csv`` through ``candidates.csv``
  (which is what maps an ``event_id`` to its clip filename).

## Why awake_* clips are excluded from the grind prototype

``data/raw/grind_clips/`` also holds ``awake_*.wav`` — deliberate tooth
rubbing, recorded 30 cm from the mic while awake. Sprint 3.5 measured those
against real sleep bruxism and found them spectrally almost opposite (36.7%
vs 0.5% of energy above 8 kHz; centroid 9,651 vs 3,051 Hz; flatness 0.155 vs
0.0013). Averaging them into the grind centroid would aim it at a sound the
night recordings do not contain. :func:`load_grind_references` therefore
matches ``sleep_*`` only, and :data:`EXCLUDED_REFERENCE_PREFIXES` documents
the exclusion rather than leaving it to a glob nobody questions later.
"""

from __future__ import annotations

import csv
from pathlib import Path

# The review tool (tools/review.py) writes Thai labels; everything downstream
# works in these canonical keys.
THAI_LABEL_MAP: dict[str, str] = {
    "กัดฟัน": "grind",
    "กรน": "snore",
    "อื่นๆ": "other",
    "ไม่แน่ใจ": "unsure",
}

# "unsure" is a human saying they could not tell — averaging those into a
# centroid would define a class out of ambiguity.
EXCLUDED_PROTOTYPE_LABELS: frozenset[str] = frozenset({"unsure"})

# See module docstring.
EXCLUDED_REFERENCE_PREFIXES: tuple[str, ...] = ("awake_",)
GRIND_REFERENCE_GLOB = "sleep_*.wav"


def canonical_label(raw_label: str) -> str:
    """Map a Thai review label to its canonical key (unknown labels pass through)."""
    return THAI_LABEL_MAP.get(raw_label.strip(), raw_label.strip())


def load_grind_references(references_dir: Path) -> list[Path]:
    """Confirmed sleep-bruxism clips, excluding the awake tooth-rubbing set."""
    candidates = sorted(references_dir.glob(GRIND_REFERENCE_GLOB))
    return [p for p in candidates if not p.name.startswith(EXCLUDED_REFERENCE_PREFIXES)]


def load_clip_map(candidates_csv: Path) -> dict[str, str]:
    """``{event_id: clip_file}`` from a candidates.csv."""
    mapping: dict[str, str] = {}
    with open(candidates_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_file = (row.get("clip_file") or "").strip()
            if clip_file:
                mapping[row["event_id"]] = clip_file
    return mapping


def load_labels(labels_csv: Path) -> dict[str, str]:
    """``{event_id: canonical_label}`` from a labels.csv."""
    labels: dict[str, str] = {}
    with open(labels_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = (row.get("event_id") or "").strip()
            raw = (row.get("label") or "").strip()
            if event_id and raw:
                labels[event_id] = canonical_label(raw)
    return labels


def load_labeled_clips(
    labels_csv: Path, candidates_csv: Path, clips_dir: Path
) -> tuple[dict[str, list[Path]], list[str]]:
    """Group a night's labelled clips by canonical label.

    Returns:
        ``(clips_by_label, warnings)``. A label whose clip file is missing on
        disk is reported in ``warnings`` rather than dropped silently — a
        prototype quietly built from 8 of 10 clips is a bug you never see.
    """
    clip_map = load_clip_map(candidates_csv)
    labels = load_labels(labels_csv)

    clips_by_label: dict[str, list[Path]] = {}
    warnings: list[str] = []
    for event_id, label in sorted(labels.items()):
        clip_file = clip_map.get(event_id)
        if not clip_file:
            warnings.append(f"{event_id}: ไม่มี clip_file ใน candidates.csv")
            continue
        path = clips_dir / clip_file
        if not path.exists():
            warnings.append(f"{event_id}: ไม่พบไฟล์คลิป {path.name}")
            continue
        clips_by_label.setdefault(label, []).append(path)
    return clips_by_label, warnings


def build_reference_sets(
    references_dir: Path,
    labels_csv: Path,
    candidates_csv: Path,
    clips_dir: Path,
    include_labeled_grind: bool = False,
) -> tuple[dict[str, list[Path]], list[str]]:
    """All prototype reference clips, keyed by canonical label.

    Args:
        include_labeled_grind: Whether the night's own clips labelled "grind"
            also feed the grind prototype. Off by default: those same events
            are what ``tools/evaluate_similar.py`` measures retrieval against,
            and a prototype built from them would be scoring itself.

    Returns:
        ``(clips_by_label, warnings)`` with ``EXCLUDED_PROTOTYPE_LABELS``
        already removed.
    """
    clips_by_label, warnings = load_labeled_clips(labels_csv, candidates_csv, clips_dir)

    if not include_labeled_grind:
        clips_by_label.pop("grind", None)

    grind_refs = load_grind_references(references_dir)
    if grind_refs:
        clips_by_label.setdefault("grind", []).extend(grind_refs)
    else:
        warnings.append(f"ไม่พบคลิปอ้างอิง {GRIND_REFERENCE_GLOB} ใน {references_dir}")

    for label in EXCLUDED_PROTOTYPE_LABELS:
        clips_by_label.pop(label, None)

    return {k: sorted(set(v)) for k, v in clips_by_label.items() if v}, warnings
