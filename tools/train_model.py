"""Train the bruxism classifier head on DreamCatcher.

The heavy part is embedding 39,592 annotated segments through YAMNet
(~17 min on CPU); that result is cached, so re-training with different
split/hyperparameters afterwards takes seconds.

Usage:
    python -m tools.train_model                       # session-level split (default)
    python -m tools.train_model --group-by subject    # stricter: unseen sleeper
    python -m tools.train_model --max-other 20000     # subsample the dominant class
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.embed.encoder import YamnetEncoder
from src.train.dataset import (
    CLASS_BRUXISM,
    CLASS_NAMES,
    CLASS_OTHER,
    class_distribution,
    grouped_split,
    load_segments,
)
from src.train.train_classifier import (
    cache_path_for,
    extract_frame_dataset,
    save_artifacts,
    train_head,
)

logger = logging.getLogger("bruxecho.train_model")

DEFAULT_CONFIG = Path("configs/paths.yaml")


def load_paths(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def subsample_other(segments, max_other: int | None, seed: int = 0):
    """Cap the ``other`` class, which is 73% of the dataset (27k are 'breathe').

    Only touches ``other`` — every bruxism and snore event is kept, since
    those are the two the app actually reports.
    """
    if max_other is None:
        return segments
    other_idx = [i for i, s in enumerate(segments) if s.label == CLASS_OTHER]
    if len(other_idx) <= max_other:
        return segments
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(other_idx, size=max_other, replace=False).tolist())
    return [s for i, s in enumerate(segments) if s.label != CLASS_OTHER or i in keep]


def run(
    config_path: Path,
    group_by: str,
    val_fraction: float,
    max_other: int | None,
    epochs: int,
    seed: int,
) -> None:
    paths = load_paths(config_path)
    audio_dir = Path(paths["dreamcatcher"]["audio_dir"])
    tsv = Path(paths["dreamcatcher"]["segmentation_tsv"])
    cache_dir = Path(paths["artifacts"]["embedding_cache"])
    model_dir = Path(paths["artifacts"]["model_dir"])

    if not tsv.exists():
        raise SystemExit(f"ไม่พบ {tsv} — ตรวจ configs/paths.yaml")

    segments = load_segments(tsv)
    logger.info("อ่าน segment ได้ %d รายการ: %s", len(segments), class_distribution([s.label for s in segments]))
    segments = subsample_other(segments, max_other, seed)
    if max_other is not None:
        logger.info("หลังสุ่มลด other: %s", class_distribution([s.label for s in segments]))

    encoder = YamnetEncoder()
    cache_path = cache_path_for(cache_dir, tsv, len(segments))
    if cache_path.exists():
        logger.info("ใช้ embedding cache ที่ %s", cache_path)
    else:
        logger.info("กำลังคำนวณ embedding %d segment (ครั้งแรกใช้เวลา ~17 นาที) ...", len(segments))

    started = time.time()
    last_report = [0]

    def on_progress(done: int, total: int) -> None:
        if done - last_report[0] >= 2000 or done == total:
            last_report[0] = done
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            logger.info("  %d/%d segment (%.0f/s, เหลืออีก ~%.0f วินาที)", done, total, rate, remaining)

    frames = extract_frame_dataset(segments, audio_dir, encoder, cache_path, on_progress)
    logger.info(
        "ได้ %d frame จาก %d segment ใน %.0f วินาที — %s",
        len(frames),
        len(segments),
        time.time() - started,
        frames.counts(),
    )
    if len(frames) == 0:
        raise SystemExit("ไม่ได้ frame เลยสักอัน")

    group_values = frames.sessions if group_by == "session" else frames.subjects
    frame_labels = [CLASS_NAMES[i] for i in frames.labels]
    train_groups, val_groups = grouped_split(
        list(group_values), frame_labels, val_fraction=val_fraction, balance_label=CLASS_BRUXISM
    )
    is_val = np.isin(group_values, list(val_groups))
    x_train, y_train = frames.embeddings[~is_val], frames.labels[~is_val]
    x_val, y_val = frames.embeddings[is_val], frames.labels[is_val]

    def counts(y: np.ndarray) -> dict[str, int]:
        return {name: int(np.sum(y == i)) for i, name in enumerate(CLASS_NAMES)}

    logger.info("แบ่งตาม %s — train %d frame %s", group_by, len(y_train), counts(y_train))
    logger.info("               val   %d frame %s", len(y_val), counts(y_val))
    if len(y_val) == 0 or len(y_train) == 0:
        raise SystemExit("การแบ่ง train/val ได้ฝั่งว่าง — ลองปรับ --val-fraction")

    logger.info("กำลังเทรน head ...")
    train_started = time.time()
    artifacts = train_head(x_train, y_train, x_val, y_val, epochs=epochs, seed=seed, verbose=0)
    logger.info("เทรนเสร็จใน %.0f วินาที", time.time() - train_started)

    save_artifacts(
        artifacts,
        model_dir,
        extra_metadata={
            "dataset": "DreamCatcher validation split (NeurIPS 2024, CC-BY-4.0)",
            "group_by": group_by,
            "val_fraction": val_fraction,
            "max_other": max_other,
            "n_train_frames": int(len(y_train)),
            "n_val_frames": int(len(y_val)),
            "train_groups": sorted(train_groups),
            "val_groups": sorted(val_groups),
            "seed": seed,
        },
    )
    logger.info("บันทึกโมเดลที่ %s", model_dir)

    # Saved next to the model so tools/evaluate_model.py never has to guess
    # which frames were held out.
    split_path = model_dir / "val_split.npz"
    np.savez_compressed(
        split_path,
        x_val=x_val,
        y_val=y_val,
        groups_val=group_values[is_val],
        segment_index_val=frames.segment_index[is_val],
    )
    logger.info("บันทึกชุด validation ที่ %s", split_path)

    print()
    print("═" * 70)
    print(f"เทรนเสร็จ — แบ่งข้อมูลตาม {group_by}")
    print("═" * 70)
    print(f"train: {len(y_train):>7} frame  {counts(y_train)}")
    print(f"val:   {len(y_val):>7} frame  {counts(y_val)}")
    print(f"โมเดล: {model_dir}")
    print()
    print("ขั้นถัดไป: python -m tools.evaluate_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="เทรนตัวจำแนก bruxism/snore/other บน YAMNet embedding")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--group-by",
        choices=("session", "subject"),
        default="session",
        help="แบ่ง train/val ตาม session (ค่าเริ่มต้น) หรือตามคน (เข้มกว่า — วัดการ generalize ข้ามคน)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--max-other",
        type=int,
        default=None,
        help="สุ่มลดคลาส other ให้เหลือไม่เกินกี่ event (ไม่ใส่ = ใช้ทั้งหมด)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.config, args.group_by, args.val_fraction, args.max_other, args.epochs, args.seed)


if __name__ == "__main__":
    main()
