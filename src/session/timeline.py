"""Mapping between audio-relative offsets and wall-clock time.

Pure library code: no plotting, no CLI, no printing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .schema import SessionInfo


def audio_offset_to_wallclock(session: SessionInfo, offset_s: float) -> datetime:
    """Convert an offset into the concatenated recording to a wall-clock time.

    ``offset_s`` is relative to the continuous audio timeline (segment
    files back-to-back, exactly as :mod:`src.detect.energy_gate` reports
    event positions) — it does **not** include the wall-clock duration of
    any interruption gaps, since those gaps have no audio and session.json
    doesn't record which segment-offset each interruption happened at.
    For a session with interruptions, the returned time will read slightly
    early relative to the true wall clock by the amount of gap time elapsed
    before that point. For a session with no interruptions it's exact.

    Args:
        session: Parsed session.json.
        offset_s: Seconds into the concatenated segment audio.

    Returns:
        Naive datetime derived from ``session.started_at`` (its UTC offset
        is dropped — callers that need it should parse ``started_at``
        directly).
    """
    started_at = datetime.fromisoformat(session.started_at)
    return started_at.replace(tzinfo=None) + timedelta(seconds=offset_s)
