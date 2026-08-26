"""Run the energy gate over one session and produce reviewable candidates.

Streams every segment listed in session.json through
``src.detect.energy_gate.EnergyGate`` (never loading a whole night into
memory), then writes:

- ``candidates.csv`` — one row per detected event
- ``clips/<session_id>_<seq>_<HHMMSS>.wav`` — a ~5s clip per event, for
  ``tools/review.py``
- ``overview.png`` — whole-night RMS envelope + adaptive threshold, with
  detected events marked

Usage:
    python -m tools.detect_events --session data/raw/sleep/ext_20260820
    python -m tools.detect_events --session <path> --k 5 --min-duration 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.detect.energy_gate import DEFAULT_K, DEFAULT_MIN_EVENT_DURATION_S, EnergyGate, EventCandidate
from src.session.schema import SessionInfo, read_session_json
from src.session.timeline import audio_offset_to_wallclock

logger = logging.getLogger("bruxecho.detect_events")

READ_BLOCK_SECONDS = 2.0  # also sets envelope_trace_db's plotting resolution — small enough that brief events aren't smoothed away
CLIP_DURATION_S = 5.0
CLIP_EDGE_PADDING_S = 0.25  # extra padding on each side when an event is already >= CLIP_DURATION_S
TOO_MANY_EVENTS = 2000
TOO_FEW_EVENTS = 20


def resolve_sample_rate(session: SessionInfo, session_dir: Path) -> int:
    if session.sample_rate_actual:
        return session.sample_rate_actual
    if not session.segments:
        raise SystemExit("session.json ไม่มี sample_rate_actual และไม่มี segment ให้ probe แทน")
    first = session_dir / session.segments[0].file
    return sf.info(str(first)).samplerate


def stream_session(
    session: SessionInfo, session_dir: Path, gate: EnergyGate
) -> list[tuple[str, float, float]]:
    """Feed every segment through the gate; return (file, start_s, end_s) boundaries.

    Boundaries are built from samples actually read, not session.json's
    recorded duration_s, so they stay exactly consistent with the sample
    offsets EnergyGate reports for detected events.
    """
    boundaries: list[tuple[str, float, float]] = []
    cursor_s = 0.0
    block_frames = int(READ_BLOCK_SECONDS * gate.sample_rate)

    for seg in session.segments:
        seg_path = session_dir / seg.file
        if not seg_path.exists():
            logger.warning("ข้าม segment ที่หายไป: %s", seg_path)
            continue
        frames_read = 0
        for block in sf.blocks(str(seg_path), blocksize=block_frames, dtype="float32", always_2d=False):
            gate.process_block(np.asarray(block, dtype=np.float32), segment_file=seg.file)
            frames_read += len(block)
        seg_duration_s = frames_read / gate.sample_rate
        boundaries.append((seg.file, cursor_s, cursor_s + seg_duration_s))
        cursor_s += seg_duration_s

    return boundaries


def read_window(
    segment_boundaries: list[tuple[str, float, float]],
    session_dir: Path,
    window_start_s: float,
    window_end_s: float,
) -> tuple[np.ndarray, int]:
    """Read audio for [window_start_s, window_end_s), possibly spanning segments."""
    chunks: list[np.ndarray] = []
    sample_rate = 0
    for seg_file, seg_start, seg_end in segment_boundaries:
        if seg_end <= window_start_s or seg_start >= window_end_s:
            continue
        read_start_s = max(window_start_s, seg_start) - seg_start
        read_end_s = min(window_end_s, seg_end) - seg_start
        path = session_dir / seg_file
        info = sf.info(str(path))
        sample_rate = info.samplerate
        start_frame = int(round(read_start_s * info.samplerate))
        n_frames = int(round((read_end_s - read_start_s) * info.samplerate))
        if n_frames <= 0:
            continue
        data, _ = sf.read(str(path), start=start_frame, frames=n_frames, dtype="float32")
        chunks.append(data)
    if not chunks:
        return np.array([], dtype=np.float32), sample_rate
    return np.concatenate(chunks), sample_rate


def clip_window_for_event(event: EventCandidate, total_duration_s: float) -> tuple[float, float]:
    span = event.end_s - event.start_s
    if span >= CLIP_DURATION_S:
        start = event.start_s - CLIP_EDGE_PADDING_S
        end = event.end_s + CLIP_EDGE_PADDING_S
    else:
        pad = (CLIP_DURATION_S - span) / 2.0
        start = event.start_s - pad
        end = event.end_s + pad
    return max(0.0, start), min(total_duration_s, end)


def write_clips(
    events: list[EventCandidate],
    segment_boundaries: list[tuple[str, float, float]],
    session_dir: Path,
    clips_dir: Path,
    session: SessionInfo,
) -> list[str]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    total_duration_s = segment_boundaries[-1][2] if segment_boundaries else 0.0
    clip_files: list[str] = []
    for i, event in enumerate(events, start=1):
        start_s, end_s = clip_window_for_event(event, total_duration_s)
        audio, sr = read_window(segment_boundaries, session_dir, start_s, end_s)
        wall_clock = audio_offset_to_wallclock(session, event.start_s)
        clip_name = f"{session.session_id}_{i:04d}_{wall_clock.strftime('%H%M%S')}.wav"
        if audio.size and sr:
            sf.write(str(clips_dir / clip_name), audio, sr, subtype="PCM_16")
        else:
            logger.warning("ตัดคลิปไม่ได้สำหรับ %s (ไม่มีข้อมูลเสียงในช่วงนี้)", event.event_id)
            clip_name = ""
        clip_files.append(clip_name)
    return clip_files


def write_candidates_csv(
    events: list[EventCandidate], clip_files: list[str], session: SessionInfo, output_path: Path
) -> pd.DataFrame:
    rows = []
    for event, clip_file in zip(events, clip_files):
        row = asdict(event)
        row["wall_clock_time"] = audio_offset_to_wallclock(session, event.start_s).isoformat()
        row["clip_file"] = clip_file
        # Denormalized from session.json so tools/review.py can work from
        # results/events/ alone — detect_events.py's --session can point
        # anywhere, so that original directory isn't reliably re-locatable
        # from the session_id later.
        row["provenance"] = session.provenance
        rows.append(row)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def plot_overview(
    gate: EnergyGate, events: list[EventCandidate], session: SessionInfo, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    if gate.envelope_trace_db:
        times_s, values_db = zip(*gate.envelope_trace_db)
        wall_times = [audio_offset_to_wallclock(session, t) for t in times_s]
        ax.plot(wall_times, values_db, color="steelblue", linewidth=0.8, label="gate-band RMS")

    if gate.threshold_history_db:
        th_times_s, th_values_db = zip(*gate.threshold_history_db)
        th_wall_times = [audio_offset_to_wallclock(session, t) for t in th_times_s]
        ax.plot(th_wall_times, th_values_db, color="crimson", linewidth=0.8, linestyle="--", label="adaptive threshold")

    if events:
        event_times = [audio_offset_to_wallclock(session, e.start_s) for e in events]
        event_values = [e.peak_rms_db for e in events]
        ax.scatter(event_times, event_values, color="red", s=14, zorder=5, label=f"events ({len(events)})")

    # Plot text is kept in English (unlike this tool's console output) to
    # match every other chart in this project (see tools/explore.py) —
    # the default matplotlib font has no Thai glyphs.
    ax.set_xlabel("time")
    ax.set_ylabel("RMS (dB)")
    ax.set_title(f"Energy gate overview — {session.session_id}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(
    events: list[EventCandidate],
    session: SessionInfo,
    gate: EnergyGate,
    total_duration_s: float,
    k: float,
) -> None:
    print()
    print("═" * 45)
    print(f"สรุปผล — {session.session_id}")
    print("═" * 45)
    print(f"ย่านความถี่ที่ใช้จริง: {gate.band.low_hz:.0f}-{gate.band.high_hz:.0f} Hz")
    if gate.band.narrow_band_warning:
        print("⚠ ย่านความถี่แคบกว่า 6000 Hz — ประสิทธิภาพการกรองจะลดลงมาก")

    n_events = len(events)
    hours = total_duration_s / 3600.0 if total_duration_s > 0 else 0.0
    events_per_hour = n_events / hours if hours > 0 else 0.0
    gated_event_s = sum(e.duration_s for e in events)
    gated_out_pct = (1.0 - gated_event_s / total_duration_s) * 100.0 if total_duration_s > 0 else 100.0

    print(f"จำนวน event: {n_events}")
    print(f"event/ชั่วโมง: {events_per_hour:.1f}")
    print(f"สัดส่วนของคืนที่ถูกคัดออก (ไม่ใช่ event): {gated_out_pct:.2f}%")

    if events:
        buckets: dict[str, int] = {}
        for e in events:
            wall = audio_offset_to_wallclock(session, e.start_s)
            key = wall.strftime("%Y-%m-%d %H:00")
            buckets[key] = buckets.get(key, 0) + 1
        print("การกระจายรายชั่วโมง:")
        for key in sorted(buckets):
            print(f"  {key}: {buckets[key]} event")

    if n_events > TOO_MANY_EVENTS:
        suggested = k * 1.5
        print(f"⚠ event เกิน {TOO_MANY_EVENTS} — ลองเพิ่ม --k เป็นประมาณ {suggested:.1f} แล้วรันใหม่")
    elif n_events < TOO_FEW_EVENTS:
        suggested = max(1.0, k * 0.6)
        print(f"⚠ event น้อยกว่า {TOO_FEW_EVENTS} — ลองลด --k เป็นประมาณ {suggested:.1f} แล้วรันใหม่")
    else:
        print(f"จำนวน event อยู่ในช่วงเป้าหมาย (100-500 ต่อคืนถือว่าฟังไหว, ได้ {n_events})")
    print("═" * 45)


def run(session_dir: Path, output_root: Path, k: float, min_duration_s: float) -> None:
    session = read_session_json(session_dir)
    if not session.segments:
        raise SystemExit(f"session.json ที่ {session_dir} ไม่มี segment ให้ประมวลผล")

    sample_rate = resolve_sample_rate(session, session_dir)
    logger.info(
        "session=%s provenance=%s sample_rate=%d Hz segments=%d",
        session.session_id,
        session.provenance,
        sample_rate,
        len(session.segments),
    )

    gate = EnergyGate(sample_rate=sample_rate, session_id=session.session_id, k=k, min_duration_s=min_duration_s)
    if gate.band.narrow_band_warning:
        logger.warning(
            "ย่านความถี่ที่ใช้จริงแคบกว่า 6000 Hz (%.0f-%.0f Hz) — ประสิทธิภาพการกรองจะลดลงมาก", gate.band.low_hz, gate.band.high_hz
        )

    segment_boundaries = stream_session(session, session_dir, gate)
    if not segment_boundaries:
        raise SystemExit("ไม่มี segment ใดอ่านได้สำเร็จเลย")
    total_duration_s = segment_boundaries[-1][2]

    events = gate.finalize()
    logger.info("พบ %d event ดิบหลังกรอง merge/min-duration", len(events))

    session_output_dir = output_root / session.session_id
    clip_files = write_clips(events, segment_boundaries, session_dir, session_output_dir / "clips", session)
    write_candidates_csv(events, clip_files, session, session_output_dir / "candidates.csv")
    plot_overview(gate, events, session, session_output_dir / "overview.png")

    logger.info("บันทึก %s", session_output_dir / "candidates.csv")
    logger.info("บันทึก %s", session_output_dir / "overview.png")
    logger.info("บันทึกคลิป %d ไฟล์ที่ %s", len([c for c in clip_files if c]), session_output_dir / "clips")

    print_summary(events, session, gate, total_duration_s, k)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="รัน energy gate บน session หนึ่งชุด แล้วสร้างคลิป+สรุปผลสำหรับตรวจทาน")
    parser.add_argument("--session", required=True, type=Path, help="โฟลเดอร์ session (มี session.json อยู่ข้างใน)")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help=f"ตัวคูณ MAD สำหรับ threshold (ค่าเริ่มต้น {DEFAULT_K})")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_EVENT_DURATION_S,
        help=f"ความยาวขั้นต่ำของ event เป็นวินาที (ค่าเริ่มต้น {DEFAULT_MIN_EVENT_DURATION_S})",
    )
    parser.add_argument("--output", type=Path, default=Path("results/events"), help="โฟลเดอร์ปลายทางหลัก")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    if not (args.session / "session.json").exists():
        raise SystemExit(f"ไม่พบ session.json ใน {args.session}")

    run(args.session, args.output, args.k, args.min_duration)


if __name__ == "__main__":
    main()
