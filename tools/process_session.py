"""Run the full detection pipeline over one night and hand the app its results.

Writes ``events.json`` (the schema ``RealDetectionRepository`` reads) plus a
5-second clip per event, next to the recording. ``--push`` copies both onto
the phone with adb and prints every command it runs, so the same thing can be
done by hand later.

Usage:
    python -m tools.process_session --session data/raw/sleep/ext_night01
    python -m tools.process_session --session <path> --threshold 0.7 --push
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from src.audio.listening import enhance_for_listening
from src.detect.pipeline import (
    DEFAULT_THRESHOLD,
    DEFAULT_MEDIAN_FRAMES,
    Event,
    analyze_night,
    read_window,
)
from src.session.schema import read_session_json
from src.session.timeline import audio_offset_to_wallclock
from src.train.train_classifier import load_artifacts

logger = logging.getLogger("bruxecho.process_session")

DEFAULT_CONFIG = Path("configs/paths.yaml")
SCHEMA_VERSION = 1
CLIP_DURATION_S = 5.0
CLIP_SUBDIR = "analysis_clips"
PROBS_CACHE_FILE = ".probs_cache.npz"
ANDROID_SESSIONS_DIR = "/storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions"

# Fixed cuts rather than threshold-relative: the app shows these as words to a
# person, and a "high confidence" badge should mean the same thing regardless
# of which threshold this run happened to use.
CONFIDENCE_HIGH = 0.90
CONFIDENCE_MEDIUM = 0.75

# Clips are written for a human to listen to on a phone speaker. Peak
# normalization to -3 dBFS was tried first and was not enough: measured on a
# real night, 95.7% of a clip's energy sat below 20 Hz (DC offset up to
# -0.18), so the headroom went entirely to inaudible rumble and the grinding
# itself stayed near -45 dBFS in the band a person hears. See
# src/audio/listening.py. Safe because nothing analyses these files — the
# detector reads the original segments.
CLIP_EXTRA_GAIN_DB = 12.0

# The only label the app surfaces a playable clip for.
LABEL_APP_PLAYS = "bruxism"


def confidence_level(probability: float) -> str:
    if probability >= CONFIDENCE_HIGH:
        return "HIGH"
    if probability >= CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def clip_window(event: Event, total_duration_s: float) -> tuple[float, float]:
    """A ~5s listening window centred on the event (clamped to the night)."""
    if event.duration_s >= CLIP_DURATION_S:
        return max(0.0, event.start_s - 0.25), min(total_duration_s, event.end_s + 0.25)
    pad = (CLIP_DURATION_S - event.duration_s) / 2.0
    return max(0.0, event.start_s - pad), min(total_duration_s, event.end_s + pad)


def write_clips(
    events: list[Event],
    session_dir: Path,
    session,
    sample_rate: int,
    total_duration_s: float,
    output_dir: Path,
    extra_gain_db: float = CLIP_EXTRA_GAIN_DB,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # A re-run with a different threshold produces a different, usually
    # smaller, set of clips. Without this, the old run's files stay behind and
    # the folder fills with clips no events.json references — and --push would
    # copy every one of them onto the phone.
    for stale in output_dir.glob("*.wav"):
        stale.unlink()
    clip_names: list[str] = []
    for i, event in enumerate(events, start=1):
        start_s, end_s = clip_window(event, total_duration_s)
        audio = read_window(session_dir, session, start_s, end_s, sample_rate)
        wall_clock = audio_offset_to_wallclock(session, event.start_s)
        name = f"{event.label}_{i:04d}_{wall_clock.strftime('%H%M%S')}.wav"
        if audio.size:
            audio = enhance_for_listening(audio, sample_rate, extra_gain_db=extra_gain_db)
            sf.write(str(output_dir / name), audio, sample_rate, subtype="PCM_16")
            clip_names.append(name)
        else:
            logger.warning("ตัดคลิปไม่ได้สำหรับ event ที่ %.1f วินาที", event.start_s)
            clip_names.append("")
    return clip_names


def build_events_json(
    session,
    events: list[Event],
    clip_names: list[str],
    total_duration_s: float,
    threshold: float,
    metadata: dict,
) -> dict:
    hours = total_duration_s / 3600.0 if total_duration_s > 0 else 0.0
    bruxism = [e for e in events if e.label == "bruxism"]
    snore = [e for e in events if e.label == "snore"]

    event_rows = []
    for i, (event, clip) in enumerate(zip(events, clip_names), start=1):
        event_rows.append(
            {
                "id": f"{session.session_id}_{i:04d}",
                "start_s": round(event.start_s, 3),
                "end_s": round(event.end_s, 3),
                "duration_s": round(event.duration_s, 3),
                "label": event.label,
                "confidence": round(event.confidence, 4),
                "confidence_level": confidence_level(event.confidence),
                "wall_clock_time": audio_offset_to_wallclock(session, event.start_s).isoformat(),
                "clip_file": f"{CLIP_SUBDIR}/{clip}" if clip else None,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session.session_id,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "total_duration_s": round(total_duration_s, 3),
        "model": {
            "embedding": metadata.get("embedding", "yamnet"),
            "classes": metadata.get("class_names", []),
            "threshold": threshold,
            "trained_on": metadata.get("dataset", "DreamCatcher"),
            "group_by": metadata.get("group_by"),
        },
        "summary": {
            "bruxism_count": len(bruxism),
            "snore_count": len(snore),
            "bruxism_per_hour": round(len(bruxism) / hours, 3) if hours > 0 else 0.0,
            "total_bruxism_s": round(sum(e.duration_s for e in bruxism), 3),
            # Share of the night covered by detected snoring — what the app's
            # supplementary section reports.
            "snore_ratio": round(sum(e.duration_s for e in snore) / total_duration_s, 4)
            if total_duration_s > 0
            else 0.0,
        },
        "events": event_rows,
    }


def clips_the_app_can_play(payload: dict) -> list[str]:
    """Clip filenames the app will actually offer, i.e. the bruxism ones.

    A night writes a clip per event, snoring included, because they are worth
    listening to while reviewing on a desktop. The phone is a different story:
    ``NightResult`` only ever exposes bruxism clips, so pushing the rest is
    hundreds of megabytes over USB that nothing can open. Measured on
    ext_night01: 584 clips / 256 MB, of which 4 are reachable.
    """
    names = []
    for event in payload.get("events", []):
        clip = event.get("clip_file")
        if clip and event.get("label") == LABEL_APP_PLAYS:
            names.append(clip.split("/")[-1])
    return names


def adb_push(
    session_id: str,
    events_path: Path,
    clips_dir: Path,
    device: str | None,
    clip_names: list[str] | None = None,
    session_json: Path | None = None,
) -> None:
    """Push session.json + events.json + clips to the phone, echoing every command."""
    base = ["adb"] + (["-s", device] if device else [])
    remote_dir = f"{ANDROID_SESSIONS_DIR}/{session_id}"
    commands = [
        base + ["shell", "mkdir", "-p", f"{remote_dir}/{CLIP_SUBDIR}"],
        base + ["push", str(events_path), f"{remote_dir}/events.json"],
    ]
    # session.json is what makes this a *session* to the app rather than a
    # loose results file. Without it the night shows up on the results and
    # trends screens (which read events.json) but is invisible on the calendar
    # (which reads Room, which is built from session.json) — two screens
    # disagreeing about which nights exist.
    if session_json is not None and session_json.is_file():
        commands.append(base + ["push", str(session_json), f"{remote_dir}/session.json"])
    if clip_names is None:
        if clips_dir.is_dir() and any(clips_dir.iterdir()):
            commands.append(base + ["push", f"{clips_dir}/.", f"{remote_dir}/{CLIP_SUBDIR}/"])
    else:
        for name in clip_names:
            source = clips_dir / name
            if source.is_file():
                commands.append(base + ["push", str(source), f"{remote_dir}/{CLIP_SUBDIR}/{name}"])

    for command in commands:
        printable = " ".join(command)
        print(f"  $ {printable}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("คำสั่งล้มเหลว: %s\n%s", printable, result.stderr.strip())
            raise SystemExit("adb push ไม่สำเร็จ — ตรวจว่าเสียบมือถือและเปิด USB debugging แล้ว")
        if result.stdout.strip():
            print(f"    {result.stdout.strip().splitlines()[-1]}")


def rebuild_clips(
    session_dir: Path, session, sample_rate: int, push: bool, device: str | None
) -> None:
    """Re-cut every clip named by an existing events.json, leaving it unchanged."""
    events_path = session_dir / "events.json"
    if not events_path.is_file():
        raise SystemExit(f"ไม่พบ {events_path} — ต้องรันตรวจจับก่อนถึงจะตัดคลิปใหม่ได้")

    payload = json.loads(events_path.read_text(encoding="utf-8"))
    total_duration_s = float(payload.get("total_duration_s") or 0.0)
    if total_duration_s <= 0:
        total_duration_s = sum(seg.duration_s for seg in session.segments)

    events = [
        Event(
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            label=row["label"],
            confidence=float(row.get("confidence", 0.0)),
        )
        for row in payload.get("events", [])
    ]
    clips_dir = session_dir / CLIP_SUBDIR
    logger.info("ตัดคลิปใหม่ %d ไฟล์ ...", len(events))
    clip_names = write_clips(events, session_dir, session, sample_rate, total_duration_s, clips_dir)

    # events.json already names its clips and those names are deterministic
    # (label + index + wall clock), so nothing in it needs rewriting.
    print()
    print("=" * 70)
    print(f"ตัดคลิปใหม่แล้ว {len(clip_names)} ไฟล์ ที่ {clips_dir}")
    print("events.json ไม่ถูกแก้ไข — ผลตรวจจับยังเป็นของเดิมทุกประการ")
    print("=" * 70)

    if push:
        adb_push(
            payload.get("session_id", session_dir.name),
            events_path,
            clips_dir,
            device,
            clip_names=clips_the_app_can_play(payload),
            session_json=session_dir / "session.json",
        )


def print_summary(payload: dict, events_path: Path, clips_dir: Path, pushed: bool) -> None:
    summary = payload["summary"]
    print()
    print("═" * 70)
    print(f"ผลวิเคราะห์ — {payload['session_id']}")
    print("═" * 70)
    print(f"ความยาวคืน: {payload['total_duration_s'] / 3600:.2f} ชั่วโมง")
    print(f"threshold ที่ใช้: {payload['model']['threshold']}")
    print()
    print(f"กัดฟัน: {summary['bruxism_count']} ครั้ง ({summary['bruxism_per_hour']:.1f}/ชม., "
          f"รวม {summary['total_bruxism_s']:.0f} วินาที)")
    print(f"กรน:    {summary['snore_count']} ครั้ง (สัดส่วนของคืน {summary['snore_ratio']*100:.1f}%)")

    bruxism_events = [e for e in payload["events"] if e["label"] == "bruxism"]
    if bruxism_events:
        levels = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for event in bruxism_events:
            levels[event["confidence_level"]] += 1
        print(f"  ความมั่นใจ: สูง {levels['HIGH']} · ปานกลาง {levels['MEDIUM']} · ต่ำ {levels['LOW']}")
        print()
        print("5 เหตุการณ์กัดฟันที่มั่นใจที่สุด:")
        for event in sorted(bruxism_events, key=lambda e: -e["confidence"])[:5]:
            print(f"  {event['wall_clock_time'][11:19]}  {event['duration_s']:>5.1f}s  "
                  f"conf={event['confidence']:.3f}  {event['clip_file'] or '(ไม่มีคลิป)'}")

    print()
    print(f"เขียน {events_path}")
    print(f"คลิป  {clips_dir}")
    if not pushed:
        print()
        print("ยังไม่ได้ส่งเข้ามือถือ — ใส่ --push เพื่อส่งด้วย adb")
    print("═" * 70)


def run(
    session_dir: Path,
    config_path: Path,
    threshold: float | None,
    push: bool,
    device: str | None,
    gate_k: float,
    median_frames: int = DEFAULT_MEDIAN_FRAMES,
    clips_only: bool = False,
) -> None:
    session = read_session_json(session_dir)
    if not session.segments:
        raise SystemExit(f"session.json ที่ {session_dir} ไม่มี segment")
    sample_rate = session.sample_rate_actual or sf.info(
        str(session_dir / session.segments[0].file)
    ).samplerate

    if clips_only:
        # Re-cut the listening copies from detections that already exist.
        # Worth a separate path because tuning how a clip *sounds* has nothing
        # to do with detection, and re-running the model to change a filter
        # setting costs ~20 minutes per attempt.
        rebuild_clips(session_dir, session, sample_rate, push, device)
        return

    paths = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(paths["artifacts"]["model_dir"])
    if not (model_dir / "head.keras").exists():
        raise SystemExit(f"ไม่พบโมเดลที่ {model_dir} — รัน python -m tools.train_model ก่อน")

    model, mean, std, metadata = load_artifacts(model_dir)
    resolved_threshold = (
        threshold
        if threshold is not None
        else float(metadata.get("recommended_threshold", DEFAULT_THRESHOLD))
    )
    if threshold is None:
        logger.info(
            "ใช้ threshold %.3f จาก metadata (precision=%.3f recall=%.3f ตอนประเมิน)",
            resolved_threshold,
            metadata.get("threshold_precision", float("nan")),
            metadata.get("threshold_recall", float("nan")),
        )

    def on_progress(stage: str, done: int, total: int) -> None:
        if stage == "gate":
            logger.info("กำลังรัน energy gate คัดช่วงน่าสงสัย ...")
        elif done % 50 == 0 or done == total:
            logger.info("  จำแนกแล้ว %d/%d ช่วง", done, total)

    events = analyze_night(
        session_dir,
        model,
        threshold=resolved_threshold,
        mean=mean,
        std=std,
        class_names=metadata.get("class_names"),
        gate_k=gate_k,
        median_frames=median_frames,
        probs_cache=session_dir / PROBS_CACHE_FILE,
        on_progress=on_progress,
    )
    logger.info("พบ %d event", len(events))

    total_duration_s = session.total_duration_s or 0.0
    if total_duration_s <= 0:
        total_duration_s = sum(seg.duration_s for seg in session.segments)

    clips_dir = session_dir / CLIP_SUBDIR
    clip_names = write_clips(events, session_dir, session, sample_rate, total_duration_s, clips_dir)

    payload = build_events_json(
        session, events, clip_names, total_duration_s, resolved_threshold, metadata
    )
    events_path = session_dir / "events.json"
    events_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if push:
        print()
        print("กำลังส่งเข้ามือถือด้วย adb:")
        adb_push(
            session.session_id,
            events_path,
            clips_dir,
            device,
            clip_names=clips_the_app_can_play(payload),
            session_json=session_dir / "session.json",
        )

    print_summary(payload, events_path, clips_dir, push)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="รันไปป์ไลน์ตรวจจับทั้งคืน แล้วเขียน events.json + คลิปให้แอป")
    parser.add_argument("--session", required=True, type=Path, help="โฟลเดอร์ session (มี session.json)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="ไม่ใส่ = ใช้ค่าที่ tools/evaluate_model.py แนะนำไว้ใน metadata",
    )
    parser.add_argument("--gate-k", type=float, default=6.0, help="ค่า k ของ energy gate (ต่ำ = คัดออกน้อย ปลอดภัยกว่า)")
    parser.add_argument(
        "--median-frames",
        type=int,
        default=DEFAULT_MEDIAN_FRAMES,
        help="ความกว้าง median filter (1 = ปิด — จับ event สั้นๆ ได้ดีขึ้น แต่ผลบวกลวงมากขึ้น)",
    )
    parser.add_argument(
        "--clips-only",
        action="store_true",
        help="ตัดคลิปใหม่จาก events.json เดิม ไม่ต้องรันตรวจจับซ้ำ (ใช้ตอนปรับความดัง)",
    )
    parser.add_argument("--push", action="store_true", help="ส่ง events.json + คลิปเข้ามือถือด้วย adb")
    parser.add_argument("--device", type=str, default=None, help="adb device id (ถ้าต่อหลายเครื่อง)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if not (args.session / "session.json").exists():
        raise SystemExit(f"ไม่พบ session.json ใน {args.session}")
    run(
        args.session,
        args.config,
        args.threshold,
        args.push,
        args.device,
        args.gate_k,
        args.median_frames,
        args.clips_only,
    )


if __name__ == "__main__":
    main()
