"""เทียบผลตรวจจับกับจุดที่แทรกไว้ — ตอบว่า "เจอกี่จุดจากทั้งหมด"

อ่าน ``injected_ground_truth.csv`` (จาก tools/inject_clips.py) กับ
``events.json`` (จาก tools/process_session.py) แล้วจับคู่กันด้วยการซ้อนทับ
ของช่วงเวลา

**ไม่แก้ไข events.json ใดๆ ทั้งสิ้น** — เครื่องมือนี้อ่านอย่างเดียว ผลตรวจจับ
ต้องมาจากโมเดลเท่านั้น

วิธีใช้:
    python -m tools.score_injected --session data/raw/sleep/dc_demo_mixed
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bruxecho.score_injected")

GROUND_TRUTH_FILE = "injected_ground_truth.csv"
EVENTS_FILE = "events.json"
LABEL_BRUXISM = "bruxism"
DEFAULT_TOLERANCE_S = 1.0


@dataclass
class Injected:
    start_s: float
    end_s: float
    source_clip: str
    snr_db: float


def read_ground_truth(path: Path) -> list[Injected]:
    with path.open(encoding="utf-8") as handle:
        return [
            Injected(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                source_clip=row["source_clip"],
                snr_db=float(row["snr_db"]),
            )
            for row in csv.DictReader(handle)
        ]


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float, tolerance_s: float) -> bool:
    """Whether a detection covers an injection, allowing a small timing slack.

    Slack matters because YAMNet frames are 0.96s wide with a 0.48s hop, so a
    detection's boundaries quantize to that grid and can legitimately sit a
    fraction of a second either side of the true span.
    """
    return a_start - tolerance_s < b_end and a_end + tolerance_s > b_start


def run(session_dir: Path, tolerance_s: float, label: str) -> None:
    truth_path = session_dir / GROUND_TRUTH_FILE
    events_path = session_dir / EVENTS_FILE
    if not truth_path.is_file():
        raise SystemExit(f"ไม่พบ {truth_path} — session นี้ไม่ได้ถูกแทรกคลิป")
    if not events_path.is_file():
        raise SystemExit(f"ไม่พบ {events_path} — รัน python -m tools.process_session ก่อน")

    injected = read_ground_truth(truth_path)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    detections = [e for e in payload.get("events", []) if e.get("label") == label]

    matched: list[tuple[Injected, dict | None]] = []
    used: set[int] = set()
    for item in injected:
        hit = None
        for index, event in enumerate(detections):
            if index in used:
                continue
            if overlaps(float(event["start_s"]), float(event["end_s"]), item.start_s, item.end_s, tolerance_s):
                hit = event
                used.add(index)
                break
        matched.append((item, hit))

    found = sum(1 for _, hit in matched if hit is not None)
    threshold = payload.get("model", {}).get("threshold")

    print()
    print("=" * 74)
    print(f"ผลตรวจจับเทียบกับจุดที่แทรกไว้ — {session_dir.name}")
    print("=" * 74)
    print(f"threshold ที่ใช้: {threshold}")
    print()
    print(f"{'#':>3}{'เวลา':>10}{'ยาว':>7}{'SNR':>7}  {'ผล':<8}{'conf':>7}  คลิปต้นทาง")
    for i, (item, hit) in enumerate(matched, start=1):
        status = "เจอ" if hit else "ไม่เจอ"
        confidence = f"{hit['confidence']:.3f}" if hit else "-"
        print(
            f"{i:>3}{item.start_s:>10.1f}{item.end_s - item.start_s:>7.2f}"
            f"{item.snr_db:>7.1f}  {status:<8}{confidence:>7}  {item.source_clip}"
        )
    print()
    print("-" * 74)
    print(f"► ตรวจเจอจุดที่แทรกไว้ {found}/{len(injected)} จุด ({found / max(len(injected), 1):.0%})")
    print(f"  ระบบรายงาน {label} ทั้งคืน {len(detections)} ครั้ง")
    unexplained = len(detections) - found
    print(f"  ในจำนวนนั้น {unexplained} ครั้งไม่ตรงกับจุดที่แทรก")
    print()
    print("  หมายเหตุ: ไฟล์ต้นทางเป็นเสียงนอนจริง จึงอาจมีเสียงกัดฟันของตัวเอง")
    print("  อยู่แล้ว — ตัวเลขนี้จึงบอก recall ของจุดที่แทรกได้ตรงๆ แต่ยังสรุป")
    print("  precision ไม่ได้ ถ้าไม่มีคนมาฟังยืนยันส่วนที่เหลือ")
    print("=" * 74)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="เทียบ events.json กับจุดที่แทรกไว้")
    parser.add_argument("--session", required=True, type=Path, help="โฟลเดอร์ session ที่แทรกคลิปแล้ว")
    parser.add_argument("--tolerance-s", type=float, default=DEFAULT_TOLERANCE_S, help="ความคลาดเคลื่อนเวลาที่ยอมรับ")
    parser.add_argument("--label", default=LABEL_BRUXISM, help="ชนิด event ที่นับ")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.session, args.tolerance_s, args.label)


if __name__ == "__main__":
    main()
