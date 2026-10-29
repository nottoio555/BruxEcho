"""ดูว่า threshold แต่ละค่าจะให้ event กี่ครั้ง — โดยไม่ต้องรันตรวจจับใหม่

อ่านแคชผลจำแนก (`.probs_cache.npz`) ที่ `tools/process_session.py` เขียนไว้
แล้วประกอบ event ใหม่ที่ threshold ต่างๆ ส่วนที่ช้า (energy gate + YAMNet)
ขึ้นกับเสียงกับค่า gate เท่านั้น ไม่ได้ขึ้นกับ threshold จึงข้ามได้ทั้งหมด

**ไม่แก้ events.json** — เป็นเครื่องมือดูอย่างเดียว เลือกค่าได้แล้วค่อยสั่ง
process_session ด้วย --threshold ที่ต้องการ

วิธีใช้:
    python -m tools.tune_threshold --session data/raw/sleep/night_20_real
    python -m tools.tune_threshold --session <path> --target-events 101
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.detect.pipeline import (
    DEFAULT_GATE_HIGH_HZ,
    DEFAULT_GATE_LOW_HZ,
    DEFAULT_GATE_K,
    DEFAULT_MEDIAN_FRAMES,
    DEFAULT_MERGE_GAP_S,
    DEFAULT_MIN_EVENT_S,
    _load_probs_cache,
    frames_to_events,
    merge_events,
)
from src.session.schema import read_session_json

logger = logging.getLogger("bruxecho.tune_threshold")

PROBS_CACHE_FILE = ".probs_cache.npz"
CLASS_NAMES = ["bruxism", "snore", "other"]


def count_at(regions, probs, threshold: float, label: str = "bruxism") -> int:
    target = [CLASS_NAMES.index(label)]
    events = []
    for region, probabilities in zip(regions, probs):
        events.extend(
            frames_to_events(
                probabilities, CLASS_NAMES, region_start_s=region.start_s,
                threshold=threshold, target_indices=target,
                min_event_s=DEFAULT_MIN_EVENT_S, median_frames=DEFAULT_MEDIAN_FRAMES,
            )
        )
    return len(merge_events(events, gap_s=DEFAULT_MERGE_GAP_S))


def run(session_dir: Path, target_events: int | None, label: str) -> None:
    session = read_session_json(session_dir)
    sample_rate = session.sample_rate_actual or 16000
    cached = _load_probs_cache(
        session_dir / PROBS_CACHE_FILE,
        DEFAULT_GATE_K, DEFAULT_GATE_LOW_HZ, DEFAULT_GATE_HIGH_HZ, sample_rate,
    )
    if cached is None:
        raise SystemExit(
            f"ไม่พบแคชที่ {session_dir / PROBS_CACHE_FILE} — "
            "รัน python -m tools.process_session --session <path> หนึ่งครั้งก่อน"
        )
    regions, probs = cached

    grid = np.round(np.arange(0.80, 0.9951, 0.005), 4)
    counts = [(float(t), count_at(regions, probs, float(t), label)) for t in grid]

    print()
    print("=" * 56)
    print(f"{session_dir.name} — จำนวน {label} ที่แต่ละ threshold")
    print("=" * 56)
    for t, n in counts:
        if n == 0 and t > 0.9:
            continue
        print(f"  threshold {t:.3f}  ->  {n:4d} ครั้ง")

    if target_events is not None:
        best_t, best_n = min(counts, key=lambda tn: (abs(tn[1] - target_events), -tn[0]))
        print()
        print(f"► ใกล้เป้า {target_events} ที่สุด: threshold {best_t:.3f} -> {best_n} ครั้ง")
        print()
        print("ใช้จริง:")
        print(f"  python -m tools.process_session --session {session_dir} --threshold {best_t:.3f} --push")
    print("=" * 56)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ดูจำนวน event ที่แต่ละ threshold จากแคช")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--target-events", type=int, default=None, help="จำนวนที่อยากได้")
    parser.add_argument("--label", default="bruxism")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.session, args.target_events, args.label)


if __name__ == "__main__":
    main()
