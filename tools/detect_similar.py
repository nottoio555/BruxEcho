"""Rank a night's candidate clips by similarity to confirmed grinding.

The energy gate can cut a night down by ~95%, but Sprint 3.5/4 showed it
cannot tell grinding from talking — every band tried surfaced mostly speech.
This tool re-ranks what the gate produced by comparing YAMNet embeddings
against prototypes built from clips a human already confirmed, so the clips
most likely to be real grinding float to the top of the listening queue.

**This is a ranking tool, not a verdict.** With ~6 confirmed grind clips the
prototype describes those recordings as much as it describes bruxism. The
output to trust is the order of ``ranked_clips/``, not ``predicted_label``.

Usage:
    python -m tools.detect_similar \
        --candidates results/events/ext_night01/candidates.csv \
        --references data/raw/grind_clips \
        --labels results/events/ext_night01/labels.csv \
        --output results/similar/ext_night01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.embed.encoder import YamnetEncoder, iter_audio_files
from src.embed.prototypes import build_prototypes, classify, rank_by_label, score_matrix
from src.embed.references import build_reference_sets

logger = logging.getLogger("bruxecho.detect_similar")

DEFAULT_CACHE_DIR = Path("results/embed_cache")
RANKED_CLIP_COUNT = 40
RANKED_CLIP_GAIN = 10.0  # real grinding is very quiet; review.py amplifies the same way
TARGET_LABEL = "grind"
CROP_EDGE_PAD_S = 0.1  # keep a little context so the event's onset isn't clipped off


def embed_candidates(
    encoder: YamnetEncoder,
    clip_paths: list[Path],
    durations: dict[str, float],
    crop_to_event: bool,
    on_progress,
) -> tuple[list[str], np.ndarray]:
    """Embed candidate clips, optionally cropped to just their event window.

    Why cropping matters: the reference grind clips are tight crops (0.46-2.31s,
    almost entirely event), while a candidate clip is a 5s window holding the
    event plus silence padding — a 0.7s event is ~86% background. Mean-pooled
    embeddings of those two things differ for reasons unrelated to grinding.
    Measured on ext_night01, cropping moved the three human-confirmed grind
    events from median rank 1651 to 449 out of 2536.
    """
    if not crop_to_event:
        return encoder.embed_batch(clip_paths, on_progress=on_progress)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    total = len(clip_paths)
    for i, path in enumerate(clip_paths, start=1):
        duration = durations.get(path.stem)
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if duration is not None and duration > 0:
                # detect_events centres the event in its clip window.
                centre = audio.shape[0] / 2
                half = int((duration / 2 + CROP_EDGE_PAD_S) * sr)
                lo, hi = max(0, int(centre - half)), min(audio.shape[0], int(centre + half))
                cropped = audio[lo:hi]
                if cropped.shape[0] >= int(0.05 * sr):
                    audio = cropped
            key = encoder.cache_key_for(path, f"crop{duration or 0:.2f}")
            embedding = encoder.embed_audio(audio, sr, cache_key=key)
        except Exception:  # noqa: BLE001 — one bad clip shouldn't abort the run
            if on_progress is not None:
                on_progress(i, total, path)
            continue
        ids.append(path.stem)
        vectors.append(embedding.mean)
        if on_progress is not None:
            on_progress(i, total, path)

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 1024), dtype=np.float32)
    return ids, matrix


def build_prototype_set(
    references_dir: Path,
    labels_csv: Path,
    candidates_csv: Path,
    clips_dir: Path,
    encoder: YamnetEncoder,
    include_labeled_grind: bool,
):
    clips_by_label, warnings = build_reference_sets(
        references_dir, labels_csv, candidates_csv, clips_dir, include_labeled_grind
    )
    for warning in warnings:
        logger.warning("อ้างอิง: %s", warning)
    if not clips_by_label:
        raise SystemExit("ไม่มีคลิปอ้างอิงเลย — ตรวจ --references / --labels / --candidates อีกครั้ง")

    embeddings_by_label: dict[str, np.ndarray] = {}
    sources_by_label: dict[str, list[str]] = {}
    for label, paths in sorted(clips_by_label.items()):
        ids, matrix = encoder.embed_batch(paths)
        if matrix.shape[0] == 0:
            logger.warning("label %s: encode ไม่สำเร็จสักไฟล์ ข้ามไป", label)
            continue
        embeddings_by_label[label] = matrix
        sources_by_label[label] = ids
        logger.info("prototype %-6s ← %d คลิป", label, matrix.shape[0])

    prototypes = build_prototypes(embeddings_by_label, sources_by_label)
    if not any(p.label == TARGET_LABEL for p in prototypes):
        raise SystemExit(f"ไม่มี prototype ของ {TARGET_LABEL!r} — จัดอันดับไม่ได้")
    return prototypes


def write_ranked_clips(
    ranked_paths: list[Path], output_dir: Path, scores: list[float], gain: float
) -> int:
    """Copy the top clips out in listening order, amplified.

    Filenames are prefixed with their rank so a file browser plays them in
    score order, and carry the similarity so it is visible while listening.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rank, (path, similarity) in enumerate(zip(ranked_paths, scores), start=1):
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001 — one bad clip shouldn't stop the queue
            logger.warning("อ่านคลิปไม่ได้ %s (%s)", path.name, exc)
            continue
        amplified = np.clip(audio * gain, -1.0, 1.0)
        out_name = f"{rank:03d}_sim{similarity:.3f}_{path.name}"
        sf.write(str(output_dir / out_name), amplified, sr, subtype="PCM_16")
        written += 1
    return written


def print_summary(df: pd.DataFrame, labels: list[str], target_label: str, ranked_dir: Path) -> None:
    print()
    print("═" * 74)
    print(f"ผลการจัดอันดับด้วยความคล้าย — {len(df)} คลิป")
    print("═" * 74)

    print("ทำนายเป็นแต่ละ label (ดูเป็นข้อมูลประกอบ ไม่ใช่คำตัดสิน):")
    counts = df["predicted_label"].value_counts()
    for label in labels:
        print(f"  {label:<8} {int(counts.get(label, 0)):>6} คลิป")

    column = f"sim_{target_label}"
    print()
    print(f"การกระจายของ {column}:")
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  เปอร์เซ็นไทล์ {q*100:>5.1f}: {df[column].quantile(q):.4f}")
    print(f"  ค่าเฉลี่ย: {df[column].mean():.4f}   ส่วนเบี่ยงเบน: {df[column].std():.4f}")

    print()
    print(f"10 อันดับแรกตาม {column}:")
    top = df.nlargest(10, column)
    for i, row in enumerate(top.itertuples(), start=1):
        print(
            f"  {i:>2}. {row.event_id:<20} {column}={getattr(row, column):.4f} "
            f"margin={row.margin:+.4f} ทำนาย={row.predicted_label}"
        )

    print()
    print("═" * 74)
    print(f"ขั้นถัดไป: ฟังไล่ตามลำดับใน {ranked_dir}")
    print("ถ้าเสียงกัดฟันจริงอยู่อันดับต้นๆ แปลว่าวิธีนี้ใช้ได้ — ถ้ากระจายอยู่ท้ายๆ แปลว่าไม่ได้ผล")
    print("หมายเหตุ: prototype สร้างจากคลิปยืนยันแค่ ~6 อัน จึงอธิบาย 'การอัดชุดนั้น' พอๆ กับ 'เสียงกัดฟัน'")


def run(
    candidates_csv: Path,
    references_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    cache_dir: Path,
    clips_dir: Path | None,
    top_n: int,
    gain: float,
    include_labeled_grind: bool,
    crop_to_event: bool,
) -> None:
    candidates = pd.read_csv(candidates_csv)
    resolved_clips_dir = clips_dir or (candidates_csv.parent / "clips")
    if not resolved_clips_dir.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์คลิป {resolved_clips_dir}")

    encoder = YamnetEncoder(cache_dir=cache_dir)
    logger.info("กำลังโหลด YAMNet และสร้าง prototype ...")
    prototypes = build_prototype_set(
        references_dir, labels_csv, candidates_csv, resolved_clips_dir, encoder, include_labeled_grind
    )

    clip_paths = [
        resolved_clips_dir / str(name)
        for name in candidates["clip_file"].fillna("")
        if str(name).strip()
    ]
    clip_paths = [p for p in clip_paths if p.exists()]
    if not clip_paths:
        raise SystemExit(f"ไม่พบไฟล์คลิปที่อ้างถึงใน {candidates_csv}")

    logger.info("กำลังคำนวณ embedding ของ %d คลิป (ใช้ cache ที่ %s) ...", len(clip_paths), cache_dir)

    def on_progress(done: int, total: int, _path: Path) -> None:
        if done % 250 == 0 or done == total:
            logger.info("  embedding %d/%d", done, total)

    durations = {
        Path(str(row.clip_file)).stem: float(row.duration_s)
        for row in candidates.itertuples()
        if isinstance(row.clip_file, str) and row.clip_file.strip()
    }
    ids, matrix = embed_candidates(encoder, clip_paths, durations, crop_to_event, on_progress)
    if matrix.shape[0] == 0:
        raise SystemExit("encode คลิปไม่สำเร็จเลยสักไฟล์")

    labels, similarities = score_matrix(matrix, prototypes)
    stem_to_event = {
        Path(str(row.clip_file)).stem: row.event_id
        for row in candidates.itertuples()
        if isinstance(row.clip_file, str) and row.clip_file.strip()
    }

    rows = []
    for i, clip_id in enumerate(ids):
        result = classify(matrix[i], prototypes)
        row = {
            "event_id": stem_to_event.get(clip_id, clip_id),
            "clip_file": f"{clip_id}.wav",
            "predicted_label": result.label,
            "margin": result.margin,
        }
        for j, label in enumerate(labels):
            row[f"sim_{label}"] = float(similarities[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    ordering = rank_by_label(similarities, labels, TARGET_LABEL)
    df = df.iloc[ordering].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info("บันทึก %s", scored_path)

    prototype_rows = [
        {"label": p.label, "n_samples": p.n_samples, "source_files": ";".join(p.source_files)}
        for p in prototypes
    ]
    pd.DataFrame(prototype_rows).to_csv(output_dir / "prototypes.csv", index=False)

    ranked_dir = output_dir / "ranked_clips"
    top_paths = [resolved_clips_dir / str(name) for name in df["clip_file"].head(top_n)]
    top_scores = df[f"sim_{TARGET_LABEL}"].head(top_n).tolist()
    written = write_ranked_clips(top_paths, ranked_dir, top_scores, gain)
    logger.info("เขียนคลิป %d ไฟล์ (ขยายเสียง %.0f เท่า) ที่ %s", written, gain, ranked_dir)

    print_summary(df, labels, TARGET_LABEL, ranked_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="จัดอันดับคลิป candidate ตามความคล้ายกับเสียงกัดฟันที่ยืนยันแล้ว (few-shot, ไม่มีการเทรน)"
    )
    parser.add_argument("--candidates", required=True, type=Path, help="candidates.csv ของคืนที่จะจัดอันดับ")
    parser.add_argument("--references", required=True, type=Path, help="โฟลเดอร์คลิปอ้างอิง (ใช้เฉพาะ sleep_*.wav)")
    parser.add_argument("--labels", required=True, type=Path, help="labels.csv ที่ติด label ไว้แล้ว")
    parser.add_argument("--output", required=True, type=Path, help="โฟลเดอร์ปลายทาง")
    parser.add_argument(
        "--clips-dir", type=Path, default=None, help="โฟลเดอร์คลิป (ค่าเริ่มต้น: clips/ ข้าง candidates.csv)"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="ที่เก็บ cache ของ embedding")
    parser.add_argument("--top-n", type=int, default=RANKED_CLIP_COUNT, help=f"เขียนคลิปกี่อันดับแรก (ค่าเริ่มต้น {RANKED_CLIP_COUNT})")
    parser.add_argument("--gain", type=float, default=RANKED_CLIP_GAIN, help=f"ขยายเสียงกี่เท่าในคลิปที่เขียนออก (ค่าเริ่มต้น {RANKED_CLIP_GAIN:.0f})")
    parser.add_argument(
        "--crop-to-event",
        action="store_true",
        help="ตัดคลิป candidate ให้เหลือเฉพาะช่วง event ก่อน encode "
        "(คลิปอ้างอิงเป็นช่วง event ล้วน แต่ candidate เป็นหน้าต่าง 5 วินาทีที่มีความเงียบปน "
        "— วัดแล้วช่วยดันอันดับ event กัดฟันจริงจากมัธยฐาน 1651 เป็น 449)",
    )
    parser.add_argument(
        "--include-labeled-grind",
        action="store_true",
        help="ใส่ event ที่ label ว่ากัดฟันของคืนนี้เข้า prototype ด้วย (ค่าเริ่มต้นไม่ใส่ เพราะใช้เป็นชุดวัดผล)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    for path in (args.candidates, args.references, args.labels):
        if not path.exists():
            raise SystemExit(f"ไม่พบ {path}")

    run(
        args.candidates,
        args.references,
        args.labels,
        args.output,
        args.cache_dir,
        args.clips_dir,
        args.top_n,
        args.gain,
        args.include_labeled_grind,
        args.crop_to_event,
    )


if __name__ == "__main__":
    main()
