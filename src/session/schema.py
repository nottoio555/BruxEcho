"""Shared session.json schema.

Matches the schema the BruxEcho Android app writes (``android/app/.../
SessionStore.kt``) so tools here can read sessions recorded by the app
without any translation layer, plus a few extra fields (``provenance``,
``original_file``, ``original_codec``, ``quality_warnings``) that only
externally-imported recordings populate — see ``tools/import_external.py``.

Pure library code: no plotting, no CLI, no printing. The only I/O is the
explicit read/write functions below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Sessions recorded by the app itself never set `provenance` — this is the
# default assumed when the field is absent from session.json.
PROVENANCE_APP = "bruxecho_app"
PROVENANCE_EXTERNAL = "external"


@dataclass
class SegmentInfo:
    file: str
    duration_s: float


@dataclass
class InterruptionInfo:
    at: str
    reason: str
    gap_s: float


@dataclass
class SessionInfo:
    session_id: str
    started_at: str
    ended_at: str | None = None
    device_model: str | None = None
    android_version: str | None = None
    audio_source_used: str | None = None
    audio_source_fallback_reason: str | None = None
    sample_rate_requested: int | None = None
    sample_rate_actual: int | None = None
    channels: int = 1
    bit_depth: int = 16
    effects_disabled: list[str] = field(default_factory=list)
    effects_unavailable: list[str] = field(default_factory=list)
    segments: list[SegmentInfo] = field(default_factory=list)
    interruptions: list[InterruptionInfo] = field(default_factory=list)
    total_duration_s: float = 0.0
    provenance: str = PROVENANCE_APP
    original_file: str | None = None
    original_codec: str | None = None
    quality_warnings: list[str] = field(default_factory=list)


def read_session_json(session_dir: Path) -> SessionInfo:
    """Read and parse ``session_dir/session.json``.

    Tolerant of missing optional keys (older/foreign session.json files)
    but requires ``session_id`` and ``started_at`` — those two are load-
    bearing for every consumer (segment ordering, wall-clock reconstruction).

    Raises:
        FileNotFoundError: if ``session.json`` doesn't exist.
        ValueError: if the file isn't valid JSON, or is missing a required key.
    """
    path = session_dir / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบ session.json ใน {session_dir}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"session.json ที่ {path} ไม่ใช่ JSON ที่ถูกต้อง: {e}") from e

    for required_key in ("session_id", "started_at"):
        if required_key not in raw:
            raise ValueError(f"session.json ที่ {path} ขาดฟิลด์ที่จำเป็น: {required_key}")

    return SessionInfo(
        session_id=raw["session_id"],
        started_at=raw["started_at"],
        ended_at=raw.get("ended_at"),
        device_model=raw.get("device_model"),
        android_version=raw.get("android_version"),
        audio_source_used=raw.get("audio_source_used"),
        audio_source_fallback_reason=raw.get("audio_source_fallback_reason"),
        sample_rate_requested=raw.get("sample_rate_requested"),
        sample_rate_actual=raw.get("sample_rate_actual"),
        channels=raw.get("channels", 1),
        bit_depth=raw.get("bit_depth", 16),
        effects_disabled=list(raw.get("effects_disabled", [])),
        effects_unavailable=list(raw.get("effects_unavailable", [])),
        segments=[SegmentInfo(**seg) for seg in raw.get("segments", [])],
        interruptions=[InterruptionInfo(**gap) for gap in raw.get("interruptions", [])],
        total_duration_s=raw.get("total_duration_s", 0.0),
        provenance=raw.get("provenance", PROVENANCE_APP),
        original_file=raw.get("original_file"),
        original_codec=raw.get("original_codec"),
        quality_warnings=list(raw.get("quality_warnings", [])),
    )


def write_session_json(session_dir: Path, info: SessionInfo) -> None:
    """Write ``info`` to ``session_dir/session.json``.

    Writes to a temp file and renames over the target, matching the
    Android app's write pattern — a crash mid-write never leaves a
    truncated file behind.
    """
    payload = {
        "session_id": info.session_id,
        "started_at": info.started_at,
        "ended_at": info.ended_at,
        "device_model": info.device_model,
        "android_version": info.android_version,
        "audio_source_used": info.audio_source_used,
        "audio_source_fallback_reason": info.audio_source_fallback_reason,
        "sample_rate_requested": info.sample_rate_requested,
        "sample_rate_actual": info.sample_rate_actual,
        "channels": info.channels,
        "bit_depth": info.bit_depth,
        "effects_disabled": info.effects_disabled,
        "effects_unavailable": info.effects_unavailable,
        "segments": [{"file": s.file, "duration_s": s.duration_s} for s in info.segments],
        "interruptions": [
            {"at": g.at, "reason": g.reason, "gap_s": g.gap_s} for g in info.interruptions
        ],
        "total_duration_s": info.total_duration_s,
    }
    # The provenance block only makes sense for externally-imported audio —
    # a session.json written by the app itself never has these keys at all,
    # so an app-provenance SessionInfo (round-tripped through this writer)
    # stays byte-for-byte schema-compatible with what the app produces.
    if info.provenance != PROVENANCE_APP:
        payload["provenance"] = info.provenance
        payload["original_file"] = info.original_file
        payload["original_codec"] = info.original_codec
        payload["quality_warnings"] = info.quality_warnings

    target = session_dir / "session.json"
    tmp = session_dir / "session.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
