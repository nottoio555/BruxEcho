"""สุ่มกระจายเสียงของ session ออกไปให้เต็มคืน แล้วตั้งเวลาเริ่ม-จบใหม่

ใช้ตอนที่ไฟล์เสียงสั้นกว่าคืนจริงมาก (เช่น 57 นาที) แต่อยากให้หน้าผลตรวจจับ
กับไทม์ไลน์ดูเป็นคืนเต็มจริงๆ

**กระจาย "เสียง" ไม่ใช่กระจาย "เวลาของ event"** — เครื่องมือนี้สร้างไฟล์เสียง
ใหม่ที่ยาวเท่าคืนจริง แล้วย้ายช่วงที่มีเสียงของไฟล์เดิมไปวางแบบสุ่มทั่วทั้งคืน
ที่เหลือเป็นความเงียบ จากนั้นต้องรัน process_session ใหม่ให้โมเดลตรวจเอง
เวลาของ event ที่ได้จึงเป็นเวลาจริงของเสียงนั้นๆ และคลิปที่ตัดออกมาก็ตรงกับ
เวลาที่แสดง ถ้าไปแก้เวลาใน events.json ตรงๆ คลิปจะไม่ตรงกับเวลาที่โชว์

ถ้ามี ``injected_ground_truth.csv`` อยู่ จะถูกย้ายตำแหน่งตามไปด้วย

วิธีใช้:
    python -m tools.spread_session --session data/raw/sleep/dc_demo_mixed
        --started-at "2026-09-18 23:30" --ended-at "2026-09-19 05:47"
        --output data/raw/sleep/night_18
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.mixing import active_spans, to_mono
from src.session.schema import read_session_json
from tools.set_session_time import parse_timestamp

logger = logging.getLogger("bruxecho.spread_session")

GROUND_TRUTH_FILE = "injected_ground_truth.csv"
PROVENANCE = "synthetic_spread"
WRITE_BLOCK_S = 60.0
# Keep a little room either side of a span so an event never starts on the
# very first sample of the night, where the energy gate has no history yet.
EDGE_MARGIN_S = 30.0


def place_spans(
    spans: list[tuple[float, float]],
    night_s: float,
    rng: np.random.Generator,
    edge_margin_s: float = EDGE_MARGIN_S,
) -> list[float]:
    """New start time for each span, in order, spread randomly across the night.

    Order is preserved and gaps are randomized rather than shuffling the spans
    outright: whatever continuity the original recording has (a bout of
    grinding, a stretch of snoring) stays intact, and only the silence between
    stretches changes length. Shuffling would splice unrelated moments
    together and could manufacture sounds that were never recorded.
    """
    durations = [end - start for start, end in spans]
    total_audio = sum(durations)
    slack = night_s - 2 * edge_margin_s - total_audio
    if slack <= 0:
        raise SystemExit(
            f"คืนยาว {night_s:.0f} วินาที แต่เสียงจริงมี {total_audio:.0f} วินาที — สั้นเกินไป"
        )

    # Random gaps that sum to the slack: a Dirichlet draw gives uniformly
    # distributed spacing instead of the clumping plain uniform offsets cause.
    gaps = rng.dirichlet(np.ones(len(spans) + 1)) * slack

    positions: list[float] = []
    cursor = edge_margin_s
    for gap, duration in zip(gaps[:-1], durations):
        cursor += gap
        positions.append(cursor)
        cursor += duration
    return positions


def write_night(
    source: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    positions: list[float],
    night_s: float,
    out_path: Path,
) -> None:
    """Stream the new night to disk: silence, then each span at its new place.

    Written incrementally because a 6-hour 16 kHz night held as float32 is well
    over a gigabyte in memory for no reason.
    """
    total_frames = int(round(night_s * sr))
    block = np.zeros(int(WRITE_BLOCK_S * sr), dtype=np.float32)
    written = 0

    with sf.SoundFile(str(out_path), "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
        for (span_start, span_end), new_start in zip(spans, positions):
            target = int(round(new_start * sr))
            while written < target:
                chunk = min(block.size, target - written)
                out.write(block[:chunk])
                written += chunk
            audio = source[int(round(span_start * sr)) : int(round(span_end * sr))]
            out.write(audio)
            written += audio.size
        while written < total_frames:
            chunk = min(block.size, total_frames - written)
            out.write(block[:chunk])
            written += chunk


def remap_ground_truth(
    path: Path,
    spans: list[tuple[float, float]],
    positions: list[float],
    out_path: Path,
) -> int:
    """Move each injected marker to wherever its span ended up."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    moved = 0
    for row in rows:
        start = float(row["start_s"])
        end = float(row["end_s"])
        for (span_start, span_end), new_start in zip(spans, positions):
            if span_start <= start < span_end:
                offset = new_start - span_start
                row["start_s"] = f"{start + offset:.3f}"
                row["end_s"] = f"{end + offset:.3f}"
                moved += 1
                break
        else:
            # A marker outside every active span can't be placed; drop it
            # rather than leave a timestamp pointing at the wrong audio.
            row["start_s"] = ""

    kept = [r for r in rows if r["start_s"]]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda r: float(r["start_s"])))
    return moved


def run(
    session_dir: Path,
    started_at_text: str,
    ended_at_text: str,
    output_dir: Path,
    seed: int,
    force: bool,
) -> None:
    session = read_session_json(session_dir)
    if len(session.segments) != 1:
        raise SystemExit(f"รองรับเฉพาะ session ที่มี segment เดียว (นี่มี {len(session.segments)})")

    reference = datetime.fromisoformat(session.started_at) if session.started_at else None
    new_start = parse_timestamp(started_at_text, reference)
    new_end = parse_timestamp(ended_at_text, reference)
    night_s = (new_end - new_start).total_seconds()
    if night_s <= 0:
        raise SystemExit("เวลาจบต้องอยู่หลังเวลาเริ่ม")

    if output_dir.exists():
        if not force:
            raise SystemExit(f"{output_dir} มีอยู่แล้ว — ใช้ --force ถ้าต้องการเขียนทับ")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    segment_name = session.segments[0].file
    audio, sr = sf.read(str(session_dir / segment_name), dtype="float32", always_2d=False)
    track = to_mono(np.asarray(audio, dtype=np.float32))
    spans = active_spans(track, sr, min_span_s=1.0)
    if not spans:
        raise SystemExit("ไม่พบช่วงที่มีเสียงจริงในไฟล์ต้นทาง")

    audio_s = sum(end - start for start, end in spans)
    rng = np.random.default_rng(seed)
    positions = place_spans(spans, night_s, rng)

    logger.info(
        "เสียงจริง %d ช่วง รวม %.0f วินาที -> กระจายลงคืนยาว %.0f วินาที (%.1f ชั่วโมง)",
        len(spans), audio_s, night_s, night_s / 3600.0,
    )
    write_night(track, sr, spans, positions, night_s, output_dir / segment_name)

    data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    data["session_id"] = output_dir.name
    data["started_at"] = new_start.isoformat()
    data["ended_at"] = new_end.isoformat()
    data["total_duration_s"] = night_s
    data["segments"] = [{"file": segment_name, "duration_s": night_s}]
    data["provenance"] = PROVENANCE
    data["spread_source_session"] = session.session_id
    warnings = list(data.get("quality_warnings") or [])
    if "audio_spread_across_night" not in warnings:
        warnings.append("audio_spread_across_night")
    data["quality_warnings"] = warnings
    (output_dir / "session.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    truth = session_dir / GROUND_TRUTH_FILE
    moved = 0
    if truth.is_file():
        moved = remap_ground_truth(truth, spans, positions, output_dir / GROUND_TRUTH_FILE)

    print()
    print("=" * 70)
    print(f"สร้างคืนใหม่ {output_dir.name}")
    print("=" * 70)
    print(f"  เวลา       {new_start.strftime('%d/%m %H:%M')} - {new_end.strftime('%d/%m %H:%M')}"
          f"  ({night_s / 3600.0:.2f} ชั่วโมง)")
    print(f"  เสียงจริง  {audio_s:.0f} วินาที ใน {len(spans)} ช่วง กระจายแบบสุ่ม (seed {seed})")
    print(f"  ที่เหลือ   ความเงียบ {night_s - audio_s:.0f} วินาที")
    if moved:
        print(f"  ground truth ย้ายตำแหน่งตามแล้ว {moved} จุด")
    print()
    print("ขั้นถัดไป — ให้โมเดลตรวจใหม่ (เวลาที่ได้จะเป็นเวลาจริงของเสียง):")
    print(f"  python -m tools.process_session --session {output_dir} --push")
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="กระจายเสียงของ session ออกไปให้เต็มคืน")
    parser.add_argument("--session", required=True, type=Path, help="session ต้นฉบับ (ไม่ถูกแก้ไข)")
    parser.add_argument("--started-at", required=True, help="เวลาเริ่มคืนใหม่")
    parser.add_argument("--ended-at", required=True, help="เวลาจบคืนใหม่")
    parser.add_argument("--output", required=True, type=Path, help="โฟลเดอร์ session ใหม่")
    parser.add_argument("--seed", type=int, default=18, help="seed ของการสุ่มตำแหน่ง")
    parser.add_argument("--force", action="store_true", help="เขียนทับโฟลเดอร์ปลายทางถ้ามีอยู่")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.session, args.started_at, args.ended_at, args.output, args.seed, args.force)


if __name__ == "__main__":
    main()
