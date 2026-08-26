"""ดึงคลิปที่ยาวที่สุดมาขยายเสียง แล้วเรียงลำดับให้ฟังไล่ได้ในโฟลเดอร์เดียว

วิธีใช้:
    python tools/listen.py                    # ext_night01, ขยาย 10 เท่า
    python tools/listen.py ext_night01 20     # ระบุ session และอัตราขยาย

ผลลัพธ์อยู่ที่ results/events/<session>/listen/
เปิดโฟลเดอร์นั้นแล้วเล่นไฟล์ไล่ตามลำดับ 01, 02, 03 ...
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SESSION = sys.argv[1] if len(sys.argv) > 1 else "ext_night01"
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

MIN_DURATION = 3.0    # ตัดเสียงกระแทกสั้นๆ เช่น ปิดประตู
MAX_DURATION = 15.0   # ตัด event ยาวผิดปกติ ซึ่งมักเป็นเสียงต่อเนื่อง ไม่ใช่การกัดฟัน
N_CLIPS = 25

base = Path("results/events") / SESSION
candidates_csv = base / "candidates.csv"
clips_dir = base / "clips"
out_dir = base / "listen"

if not candidates_csv.exists():
    sys.exit(f"ไม่พบ {candidates_csv}")

cands = pd.read_csv(candidates_csv)

selected = cands[
    (cands["duration_s"] >= MIN_DURATION) & (cands["duration_s"] <= MAX_DURATION)
].nlargest(N_CLIPS, "duration_s").reset_index(drop=True)

if selected.empty:
    sys.exit("ไม่มีคลิปที่อยู่ในช่วงระยะเวลาที่กำหนด")

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

written = 0
for i, row in selected.iterrows():
    matches = sorted(clips_dir.glob(f"{row['event_id']}*.wav"))
    if not matches:
        print(f"  ข้าม {row['event_id']} — ไม่พบไฟล์คลิป")
        continue

    audio, sr = sf.read(matches[0])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio * GAIN, -1.0, 1.0)

    name = f"{i + 1:02d}_{row['event_id']}_{row['duration_s']:.1f}s.wav"
    sf.write(out_dir / name, audio, sr)
    written += 1

print(f"\nเขียน {written} คลิป (ขยายเสียง {GAIN:.0f} เท่า) ลงที่:")
print(f"  {out_dir.resolve()}")
print("\nเปิดโฟลเดอร์นี้แล้วเล่นไล่ตามลำดับได้เลย")
print("จดไว้ว่าคลิปไหนเป็นเสียงกัดฟัน แล้วเทียบกับชื่อไฟล์ (มี event_id อยู่ในชื่อ)")