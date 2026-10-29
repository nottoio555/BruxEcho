"""DreamCatcher segment extraction and grouped train/val splitting.

Pure library code: no plotting, no CLI, no printing, no logging.

## The dataset

DreamCatcher (NeurIPS 2024, CC-BY-4.0) — overnight audio annotated with
sleep-sound events. Read in place from the path in ``configs/paths.yaml``;
nothing here copies or moves it (the repo lives in OneDrive, and 17,774 wav
files would sync forever).

``validation.tsv`` has one row per event: ``filename, onset, offset,
event_label``, with onset/offset in seconds **from the start of that file**,
and several events per file. Audio is 16 kHz stereo, which is already
YAMNet's sample rate — only a channel downmix is needed.

## Three classes, not seven

The app has to answer "was that grinding, snoring, or neither", so the seven
annotated labels collapse to ``bruxism`` / ``snore`` / ``other``. Keeping
``breathe`` (27,118 events) as its own class would spend the model's
capacity separating sounds the app never reports.

## Why the split is grouped, never random per clip

Clips from one recording session share a microphone, a room, and a sleeper.
A random per-clip split puts near-duplicate clips on both sides and reports
a score the model cannot reproduce on a new night. :func:`grouped_split`
therefore splits on whole groups.

``group_by="session"`` (the default, e.g. ``04-1-1``) is what the sprint
asked for. Be aware of what it does *not* control: the dataset has only four
subjects (04, 08, 11, 12), so a session-level split still trains and
validates on the same four people — the resulting F1 describes "another
night from someone the model has heard", not "a new user". ``group_by=
"subject"`` measures the stricter thing and is the number to trust for
deployment.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Collapse the annotation vocabulary to what the app reports.
CLASS_BRUXISM = "bruxism"
CLASS_SNORE = "snore"
CLASS_OTHER = "other"
CLASS_NAMES: tuple[str, ...] = (CLASS_BRUXISM, CLASS_SNORE, CLASS_OTHER)

LABEL_MAP: dict[str, str] = {
    "bruxism": CLASS_BRUXISM,
    "snore": CLASS_SNORE,
    "breathe": CLASS_OTHER,
    "movements": CLASS_OTHER,
    "swallow": CLASS_OTHER,
    "somniloquy": CLASS_OTHER,
    "cough": CLASS_OTHER,
}

TARGET_SAMPLE_RATE = 16000
MIN_SEGMENT_SAMPLES = 160  # 10 ms — shorter than this is an annotation artifact, not a sound

# "04-1-1-user_1006.265244_1008.098323.wav" -> session "04-1-1", subject "04"
# The optional ``_tag`` matters: 3,000 DreamCatcher files are named like
# ``04_new-1-1-user_...``. Without it they failed to match, all nine of those
# sessions collapsed into one group named "", and that single group — wholly
# on the validation side — made up 55% of the validation frames and half of
# its bruxism. The tag stays in the *session* (04_new-1-1 is a different
# recording from 04-1-1) but not in the *subject*: ``04_new`` is subject 04,
# and treating it as a separate person would let subject 04 sit on both sides
# of a subject-wise split.
_SESSION_RE = re.compile(r"^(?P<subject>\d+)(?P<tag>_[A-Za-z]+)?-(?P<night>\d+)-(?P<part>\d+)")


@dataclass(frozen=True)
class Segment:
    """One annotated event: where it is, what it is, and who recorded it."""

    filename: str
    onset_s: float
    offset_s: float
    label: str  # already collapsed to CLASS_NAMES
    session: str
    subject: str

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


def parse_session(filename: str) -> tuple[str, str]:
    """``(session, subject)`` from a DreamCatcher filename.

    Returns ``("", "")`` when the name doesn't match, so a stray file is
    grouped separately rather than silently joining another sleeper's group.
    """
    match = _SESSION_RE.match(filename)
    if not match:
        return "", ""
    tag = match["tag"] or ""
    return f"{match['subject']}{tag}-{match['night']}-{match['part']}", match["subject"]


def load_segments(
    segmentation_tsv: Path,
    label_map: dict[str, str] | None = None,
    drop_unmapped: bool = True,
) -> list[Segment]:
    """Read the TSV into :class:`Segment` records with collapsed labels.

    Args:
        segmentation_tsv: Path to ``validation.tsv``.
        label_map: Override for :data:`LABEL_MAP`.
        drop_unmapped: Drop rows whose ``event_label`` isn't in the map. When
            False they raise instead — useful if a new DreamCatcher release
            adds a label and silently dropping it would skew the classes.
    """
    import csv  # noqa: PLC0415 — stdlib, kept local so importing this module is free

    mapping = label_map or LABEL_MAP
    segments: list[Segment] = []
    with open(segmentation_tsv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw_label = (row.get("event_label") or "").strip()
            if raw_label not in mapping:
                if drop_unmapped:
                    continue
                raise ValueError(f"ไม่รู้จัก event_label {raw_label!r} — เพิ่มลง LABEL_MAP ก่อน")
            filename = (row.get("filename") or "").strip()
            onset, offset = float(row["onset"]), float(row["offset"])
            if offset <= onset:
                continue
            session, subject = parse_session(filename)
            segments.append(
                Segment(
                    filename=filename,
                    onset_s=onset,
                    offset_s=offset,
                    label=mapping[raw_label],
                    session=session,
                    subject=subject,
                )
            )
    return segments


def read_segment_audio(audio_dir: Path, segment: Segment) -> tuple[np.ndarray, int]:
    """Mono 16 kHz samples for one annotated event.

    DreamCatcher is already 16 kHz, so this only downmixes channels — no
    resampling, which would be lossy for no reason.
    """
    import soundfile as sf  # noqa: PLC0415 — lazy

    path = audio_dir / segment.filename
    info = sf.info(str(path))
    start = max(0, int(round(segment.onset_s * info.samplerate)))
    stop = min(info.frames, int(round(segment.offset_s * info.samplerate)))
    if stop <= start:
        return np.zeros(0, dtype=np.float32), info.samplerate
    audio, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def grouped_split(
    groups: list[str],
    labels: list[str],
    val_fraction: float = 0.25,
    balance_label: str = CLASS_BRUXISM,
) -> tuple[set[str], set[str]]:
    """Split whole groups into train/val, balancing the rare class.

    A plain random group split is unusable here: bruxism counts per session
    range from 1 to 280, so one unlucky draw puts most of the positives on
    one side. This walks groups from most to fewest ``balance_label`` events
    and sends each to whichever side is furthest below its target share —
    deterministic, no seed, and the rare class ends up split close to
    ``val_fraction``.

    Returns:
        ``(train_groups, val_groups)``.
    """
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for group, label in zip(groups, labels):
        totals[group] = totals.get(group, 0) + 1
        if label == balance_label:
            counts[group] = counts.get(group, 0) + 1

    ordered = sorted(totals, key=lambda g: (-counts.get(g, 0), -totals[g], g))

    train: set[str] = set()
    val: set[str] = set()
    train_score = val_score = 0.0
    for group in ordered:
        # Weight by the rare class, with total size as a tiebreaker so groups
        # holding no bruxism still get spread instead of piling onto one side.
        weight = counts.get(group, 0) + totals[group] * 1e-3
        if val_score <= train_score * (val_fraction / max(1.0 - val_fraction, 1e-9)):
            val.add(group)
            val_score += weight
        else:
            train.add(group)
            train_score += weight
    return train, val


def config_fingerprint(*parts: object) -> str:
    """Short stable hash of whatever defines a cache's contents."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
    return digest.hexdigest()[:16]


def class_distribution(labels: list[str]) -> dict[str, int]:
    return {name: sum(1 for label in labels if label == name) for name in CLASS_NAMES}
