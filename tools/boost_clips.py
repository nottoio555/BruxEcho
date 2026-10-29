"""ตรวจระดับเสียงของคลิป แล้วขยายให้ดังพอฟังในแอป

วิธีใช้:
    python tools/boost_clips.py data/raw/sleep/dc_demo/analysis_clips          # ตรวจอย่างเดียว
    python tools/boost_clips.py data/raw/sleep/dc_demo/analysis_clips --apply  # ขยายจริง (เขียนทับ)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_PEAK_DB = -3.0
MIN_RMS_DB = -28.0


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir")
    ap.add_argument("--apply", action="store_true", help="เขียนทับไฟล์จริง (สำรองไว้ก่อน)")
    args = ap.parse_args()

    d = Path(args.clip_dir)
    files = sorted(d.glob("*.wav"))
    if not files:
        sys.exit(f"ไม่พบไฟล์ .wav ใน {d}")

    peaks, rmss = [], []
    for f in files:
        y, _ = sf.read(str(f), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        peaks.append(db(float(np.max(np.abs(y)))))
        rmss.append(db(float(np.sqrt(np.mean(y**2)))))

    print(f"ไฟล์ทั้งหมด : {len(files)}")
    print(f"peak  เฉลี่ย {np.mean(peaks):6.1f} dBFS   (ต่ำสุด {min(peaks):.1f} / สูงสุด {max(peaks):.1f})")
    print(f"RMS   เฉลี่ย {np.mean(rmss):6.1f} dBFS   (ต่ำสุด {min(rmss):.1f} / สูงสุด {max(rmss):.1f})")

    if np.mean(peaks) > -6:
        print("\n→ ไฟล์ดังพออยู่แล้ว ปัญหาน่าจะอยู่ที่ระดับเสียงมือถือหรือการเล่นในแอป")
    else:
        need = TARGET_PEAK_DB - np.mean(peaks)
        print(f"\n→ เบากว่าที่ควร ต้องขยายราว {need:.0f} dB ({10**(need/20):.0f} เท่า)")

    if not args.apply:
        print("\nใส่ --apply เพื่อขยายจริง (จะสำรองไฟล์เดิมไว้ที่โฟลเดอร์ _original)")
        return

    backup = d.parent / (d.name + "_original")
    if not backup.exists():
        shutil.copytree(d, backup)
        print(f"\nสำรองไฟล์เดิมไว้ที่ {backup}")

    for f in files:
        y, sr = sf.read(str(backup / f.name), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        peak = float(np.max(np.abs(y)))
        if peak > 1e-9:
            y = y / peak * (10 ** (TARGET_PEAK_DB / 20))
        rms = float(np.sqrt(np.mean(y**2)))
        if db(rms) < MIN_RMS_DB:
            y = y * (10 ** ((MIN_RMS_DB - db(rms)) / 20))
        sf.write(str(f), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")

    print(f"ขยายเสียงแล้ว {len(files)} ไฟล์")
    print("\nขั้นต่อไป — push ทับของเดิม:")
    print('  $dst = "/storage/emulated/0/Android/data/com.bruxecho.recorder/files/sessions/dc_demo"')
    print(f'  adb push {d}\\. "$dst/analysis_clips/"')


if __name__ == "__main__":
    main()