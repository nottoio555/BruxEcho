"""ประกอบคืนเต็มจากคลิป DreamCatcher เพื่อใช้เป็น session สาธิต

คลิปใน DreamCatcher ถูกตัดมาจากการอัดต่อเนื่องทั้งคืน
และชื่อไฟล์เก็บตำแหน่งเวลาเดิมไว้:

    04-1-1-user_1006.265244_1008.098323.wav
    └─ session ─┘ └─ เริ่ม ─┘ └─ จบ (วินาที) ─┘

สคริปต์นี้นำคลิปทั้งหมดของ session หนึ่งมาวางกลับที่ตำแหน่งเดิม
ช่วงที่ไม่มีคลิปจะเติมความเงียบ ได้ไฟล์เสียงคืนเต็มที่ใช้กับ
import_external + process_session ได้ทันที

วิธีใช้:
    python tools/build_dreamcatcher_night.py 04-1-1-user
    python tools/build_dreamcatcher_night.py 04-1-1-user --max-hours 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DC_ROOT = Path("C:/datasets/dreamcatcher")
CLIP_DIR = DC_ROOT / "data" / "validation"
TSV = DC_ROOT / "segmentation" / "validation.tsv"


def parse_name(name: str) -> tuple[str, float, float]:
    """แยก session / เวลาเริ่ม / เวลาจบ ออกจากชื่อไฟล์"""
    stem = name[:-4] if name.lower().endswith(".wav") else name
    parts = stem.split("_")
    return parts[0], float(parts[1]), float(parts[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="เช่น 04-1-1-user")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="จำกัดความยาว (ชั่วโมง) เพื่อให้ไฟล์เล็กลง")
    ap.add_argument("--out", default="C:/datasets/night_raw",
                    help="โฟลเดอร์ปลายทาง")
    args = ap.parse_args()

    if not TSV.exists():
        sys.exit(f"ไม่พบ {TSV}")

    clips = sorted(CLIP_DIR.glob(f"{args.session}_*.wav"))
    if not clips:
        sys.exit(f"ไม่พบคลิปของ session '{args.session}' ใน {CLIP_DIR}")

    # หาความยาวรวมและ sample rate
    sr = sf.info(str(clips[0])).samplerate
    ends = [parse_name(p.name)[2] for p in clips]
    starts = [parse_name(p.name)[1] for p in clips]
    t0 = min(starts)
    total_s = max(ends) - t0
    if args.max_hours:
        total_s = min(total_s, args.max_hours * 3600)

    print(f"session      : {args.session}")
    print(f"จำนวนคลิป    : {len(clips)}")
    print(f"sample rate  : {sr} Hz")
    print(f"ความยาวรวม   : {total_s / 3600:.2f} ชั่วโมง")

    night = np.zeros(int(total_s * sr) + sr, dtype=np.float32)

    placed = 0
    for p in clips:
        _, start, _ = parse_name(p.name)
        offset = int((start - t0) * sr)
        if offset >= len(night):
            continue
        audio, _ = sf.read(str(p), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        end = min(offset + len(audio), len(night))
        night[offset:end] += audio[: end - offset]
        placed += 1

    np.clip(night, -1.0, 1.0, out=night)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"dc_{args.session.replace('-', '')}.wav"
    sf.write(str(wav_path), night, sr, subtype="PCM_16")

    # สร้าง ground truth จาก TSV (แปลง onset/offset ของคลิป -> เวลาในคืน)
    tsv = pd.read_csv(TSV, sep="\t")
    tsv = tsv[tsv["filename"].str.startswith(f"{args.session}_")]
    rows = []
    for _, r in tsv.iterrows():
        _, cstart, _ = parse_name(str(r["filename"]))
        abs_start = cstart - t0 + float(r["onset"])
        abs_end = cstart - t0 + float(r["offset"])
        if abs_start <= total_s:
            rows.append({
                "start_s": round(abs_start, 3),
                "end_s": round(abs_end, 3),
                "event_label": r["event_label"],
            })
    gt = pd.DataFrame(rows).sort_values("start_s")
    gt_path = out_dir / f"dc_{args.session.replace('-', '')}_ground_truth.csv"
    gt.to_csv(gt_path, index=False)

    print(f"\nวางคลิปแล้ว  : {placed}")
    print(f"ไฟล์เสียง    : {wav_path}")
    print(f"ground truth : {gt_path}")
    print("\nจำนวน event จริงในคืนนี้:")
    print(gt["event_label"].value_counts().to_string())
    print("\nขั้นต่อไป:")
    print(f'  python -m tools.import_external --file "{wav_path}" --session-id dc_demo')
    print("  python -m tools.process_session --session data/raw/sleep/dc_demo --push")


if __name__ == "__main__":
    main()