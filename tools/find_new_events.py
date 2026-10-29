"""หา event ที่ย่านใหม่เจอ แต่ย่านเดิมไม่เคยเจอ

การรัน sweep บอกได้แค่ "ย่านใหม่ทำของเดิมหลุดไหม"
สคริปต์นี้ตอบคำถามที่สำคัญกว่า: "ย่านใหม่เจอของที่ย่านเดิมพลาดไปหรือเปล่า"

วิธีใช้:
    python tools/find_new_events.py results/events_800x4000_k8

จะสร้างโฟลเดอร์ listen_new/ ที่มีเฉพาะคลิปที่ย่านเดิมไม่เคยตรวจเจอ
ขยายเสียงแล้ว เรียงลำดับให้ฟังไล่ได้เลย
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

NEW_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "results/events_800x4000_k8")
OLD_CSV = Path("results/events/ext_night01/candidates.csv")
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
MAX_CLIPS = 40

new_csv = NEW_DIR / "candidates.csv"
if not new_csv.exists():
    sys.exit(f"ไม่พบ {new_csv}")
if not OLD_CSV.exists():
    sys.exit(f"ไม่พบ {OLD_CSV}")

old = pd.read_csv(OLD_CSV)
new = pd.read_csv(new_csv)

old_start = old["start_s"].to_numpy()
old_end = old["end_s"].to_numpy()


def has_overlap(start: float, end: float) -> bool:
    """True ถ้าช่วงเวลานี้ทับกับ event ใดๆ ของย่านเดิม"""
    return bool(np.any((old_start < end) & (old_end > start)))


new["is_new"] = [
    not has_overlap(s, e) for s, e in zip(new["start_s"], new["end_s"])
]

only_new = new[new["is_new"]].copy()

print(f"ย่านเดิม : {len(old)} event")
print(f"ย่านใหม่ : {len(new)} event")
print(f"เฉพาะย่านใหม่ (ย่านเดิมไม่เคยเจอ) : {len(only_new)} event")

if only_new.empty:
    print("\nย่านใหม่ไม่ได้เจออะไรที่ย่านเดิมพลาดเลย")
    print("→ การเปลี่ยนย่านช่วยแค่ 'ลดจำนวนที่ต้องฟัง' ไม่ได้ช่วย 'เจอของใหม่'")
    sys.exit(0)

only_new.to_csv(NEW_DIR / "new_only.csv", index=False)

out_dir = NEW_DIR / "listen_new"
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

clips_dir = NEW_DIR / "clips"
picked = only_new.nlargest(MAX_CLIPS, "peak_rms_db").reset_index(drop=True)

written = 0
for i, row in picked.iterrows():
    matches = sorted(clips_dir.glob(f"{row['event_id']}*.wav"))
    if not matches:
        continue
    audio, sr = sf.read(matches[0])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio * GAIN, -1.0, 1.0)
    name = f"{i + 1:02d}_{row['event_id']}_{row['duration_s']:.1f}s.wav"
    sf.write(out_dir / name, audio, sr)
    written += 1

print(f"\nเขียน {written} คลิป (ขยาย {GAIN:.0f} เท่า) ลงที่:")
print(f"  {out_dir.resolve()}")
print("\nฟังไล่ตามลำดับ — ถ้าเจอเสียงกัดฟันในนี้ คือหลักฐานว่าย่านเดิมพลาดของจริงไป")