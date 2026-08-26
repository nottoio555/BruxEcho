"""Import a single external audio file into the BruxEcho pipeline.

Turns a night recorded by some other app into the same shape the BruxEcho
app itself produces (rotating ~1-hour PCM16 mono WAV segments + a
session.json) so every downstream tool — ``tools/detect_events.py``,
``tools/review.py`` — can treat "recorded by the app" and "imported from
elsewhere" uniformly, while still being able to tell them apart via the
``provenance`` field (see ``src/session/schema.py``).

Reads the source file with soundfile's block-streaming API throughout —
never loads it fully into memory, so an 8-hour file is fine.

Usage:
    python -m tools.import_external --file "path/to/night.wav" --session-id ext_20260820
    python -m tools.import_external --file "path/to/night.wav" --session-id ext_20260820 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from src.session.schema import PROVENANCE_EXTERNAL, SegmentInfo, SessionInfo, write_session_json

logger = logging.getLogger("bruxecho.import_external")

SEGMENT_DURATION_S = 3600.0
READ_BLOCK_SECONDS = 10.0
SAMPLE_RATE_WARNING_HZ = 32_000

# soundfile subtype -> a codec label consistent with the ffprobe-style names
# used elsewhere in this project (e.g. android/README, verify_recording.sh).
_SUBTYPE_TO_CODEC_LABEL: dict[str, str] = {
    "PCM_16": "pcm_s16le",
    "PCM_24": "pcm_s24le",
    "PCM_32": "pcm_s32le",
    "PCM_U8": "pcm_u8",
    "FLOAT": "pcm_f32le",
    "DOUBLE": "pcm_f64le",
}
# Subtypes soundfile can decode losslessly. Anything else (ADPCM, u-law/
# A-law, Vorbis, ...) went through lossy compression before we ever saw it.
_LOSSLESS_SUBTYPES = frozenset(_SUBTYPE_TO_CODEC_LABEL)


def codec_label(subtype: str) -> str:
    return _SUBTYPE_TO_CODEC_LABEL.get(subtype, subtype.lower())


def probe_file(path: Path) -> Any:
    """Read the source file's header via soundfile — no audio data loaded.

    Raises:
        SystemExit: if soundfile can't open the file at all, which usually
            means it's actually a compressed container (AAC/M4A, MP3, ...)
            that was given a misleading .wav extension — exactly the
            Sprint 1 bug this whole project exists to avoid repeating.
    """
    try:
        return sf.info(str(path))
    except Exception as e:
        raise SystemExit(
            f"เปิดไฟล์ {path} ด้วย soundfile ไม่ได้: {e}\n\n"
            "สาเหตุที่พบบ่อยที่สุด: ไฟล์เป็น AAC/M4A (หรือคอนเทนเนอร์บีบอัดอื่น) "
            "ที่ถูกเปลี่ยนนามสกุลเป็น .wav — เหมือนปัญหาที่พบใน Sprint 1 "
            "soundfile/libsndfile อ่านได้เฉพาะ WAV/FLAC/OGG ที่แท้จริงเท่านั้น\n"
            "ลองแปลงเป็น PCM WAV ก่อนด้วย ffmpeg แล้วนำเข้าไฟล์ที่แปลงแล้วแทน:\n"
            f'  ffmpeg -i "{path}" -ar 44100 -ac 1 -c:a pcm_s16le "{path.stem}_pcm.wav"'
        )


def build_quality_warnings(info: Any) -> list[str]:
    warnings: list[str] = []
    if info.subtype not in _LOSSLESS_SUBTYPES:
        warnings.append("non_pcm_source_codec")
        logger.warning(
            "ไฟล์ต้นฉบับไม่ใช่ PCM (subtype=%s) — ข้อมูลความถี่สูงอาจสูญหายไปแล้วจากการบีบอัดก่อนหน้านี้ "
            "ตัวเลขที่วิเคราะห์ได้จากไฟล์นี้จึงใช้อ้างอิงในรายงานไม่ได้ 100%% "
            "(จะดำเนินการนำเข้าต่อไป แต่ผลลัพธ์ต้องตีความด้วยความระมัดระวัง)",
            info.subtype,
        )
    if info.samplerate < SAMPLE_RATE_WARNING_HZ:
        warnings.append("sample_rate_below_32k")
        logger.warning(
            "sample rate %d Hz ต่ำกว่า %d Hz — ย่านความถี่ 8-16 kHz จะวิเคราะห์ไม่ได้อย่างน่าเชื่อถือ "
            "(Nyquist = %.0f Hz)",
            info.samplerate,
            SAMPLE_RATE_WARNING_HZ,
            info.samplerate / 2.0,
        )
    return warnings


def plan_segments(total_frames: int, samplerate: int) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame), ...] for ~1-hour segments."""
    seg_frames = int(SEGMENT_DURATION_S * samplerate)
    plan: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + seg_frames, total_frames)
        plan.append((start, end))
        start = end
    return plan


def import_file(
    source_path: Path,
    session_id: str,
    output_root: Path,
    dry_run: bool,
) -> None:
    info = probe_file(source_path)
    is_lossless = info.subtype in _LOSSLESS_SUBTYPES
    quality_warnings = build_quality_warnings(info)

    logger.info(
        "ไฟล์ต้นฉบับ: %s | codec=%s (%s) | sample_rate=%d Hz | channels=%d | duration=%.1f นาที",
        source_path.name,
        codec_label(info.subtype),
        "lossless" if is_lossless else "lossy/compressed",
        info.samplerate,
        info.channels,
        info.duration / 60.0,
    )

    session_dir = output_root / session_id
    if session_dir.exists() and any(session_dir.iterdir()):
        if dry_run:
            logger.warning("session_id '%s' มีอยู่แล้วที่ %s (จะไม่เขียนทับ — นี่คือ --dry-run)", session_id, session_dir)
        else:
            print(f"โฟลเดอร์ {session_dir} มีอยู่แล้วและไม่ว่างเปล่า")
            answer = input("ลบของเดิมทั้งหมดแล้วนำเข้าใหม่หรือไม่? พิมพ์ 'y' เพื่อยืนยัน: ").strip().lower()
            if answer != "y":
                raise SystemExit("ยกเลิกการนำเข้า — session_id ซ้ำ")
            shutil.rmtree(session_dir)

    segment_plan = plan_segments(info.frames, info.samplerate)
    source_copy_path = session_dir / "source" / source_path.name

    if dry_run:
        print("=== --dry-run: จะไม่มีการเขียน/คัดลอกไฟล์ใดๆ ===")
        print(f"จะคัดลอกต้นฉบับไปที่: {source_copy_path}")
        print(f"จะสร้าง {len(segment_plan)} segment ที่ {session_dir}/:")
        for i, (start, end) in enumerate(segment_plan):
            dur_s = (end - start) / info.samplerate
            print(f"  segment_{i:03d}.wav — {dur_s / 60.0:.1f} นาที")
        print(f"quality_warnings ที่จะบันทึก: {quality_warnings or '(ไม่มี)'}")
        return

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "source").mkdir(parents=True, exist_ok=True)

    logger.info("คัดลอกไฟล์ต้นฉบับ (ไม่แตะ/ย้ายไฟล์เดิม) -> %s", source_copy_path)
    shutil.copy2(source_path, source_copy_path)

    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    segments_meta: list[SegmentInfo] = []

    block_frames = int(READ_BLOCK_SECONDS * info.samplerate)
    reader = sf.blocks(str(source_path), blocksize=block_frames, dtype="float32", always_2d=True)

    seg_index = 0
    writer: sf.SoundFile | None = None
    frames_in_segment = 0
    seg_frame_limit = int(SEGMENT_DURATION_S * info.samplerate)
    total_frames_written = 0

    def open_next_segment() -> sf.SoundFile:
        path = session_dir / f"segment_{seg_index:03d}.wav"
        return sf.SoundFile(str(path), mode="w", samplerate=info.samplerate, channels=1, subtype="PCM_16")

    try:
        for block in reader:
            mono = block.mean(axis=1).astype("float32") if block.shape[1] > 1 else block[:, 0]
            offset = 0
            while offset < mono.shape[0]:
                # Lazily (re)open the writer rather than only right after
                # closing one — a segment can close exactly at a block
                # boundary, in which case the next writer must be opened
                # at the start of the *next* block's iteration, not this one.
                if writer is None:
                    writer = open_next_segment()
                    frames_in_segment = 0

                remaining_in_segment = seg_frame_limit - frames_in_segment
                take = min(remaining_in_segment, mono.shape[0] - offset)
                writer.write(mono[offset : offset + take])
                frames_in_segment += take
                total_frames_written += take
                offset += take

                if frames_in_segment >= seg_frame_limit:
                    writer.close()
                    segments_meta.append(
                        SegmentInfo(file=Path(writer.name).name, duration_s=frames_in_segment / info.samplerate)
                    )
                    seg_index += 1
                    writer = None
        if writer is not None:
            writer.close()
            if frames_in_segment > 0:
                segments_meta.append(
                    SegmentInfo(file=Path(writer.name).name, duration_s=frames_in_segment / info.samplerate)
                )
    except Exception:
        if writer is not None:
            writer.close()
        raise

    total_duration_s = total_frames_written / info.samplerate
    ended_at = datetime.fromisoformat(started_at) + timedelta(seconds=total_duration_s)
    session_info = SessionInfo(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at.isoformat(timespec="seconds"),
        device_model=None,
        android_version=None,
        audio_source_used="unknown",
        sample_rate_requested=None,
        sample_rate_actual=info.samplerate,
        channels=1,
        bit_depth=16,
        segments=segments_meta,
        total_duration_s=total_duration_s,
        provenance=PROVENANCE_EXTERNAL,
        original_file=source_path.name,
        original_codec=codec_label(info.subtype),
        quality_warnings=quality_warnings,
    )
    write_session_json(session_dir, session_info)

    logger.info("นำเข้าเสร็จสิ้น: %d segment, รวม %.1f นาที -> %s", len(segments_meta), total_duration_s / 60.0, session_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="นำเข้าไฟล์เสียงภายนอกเข้าสู่ไปป์ไลน์ BruxEcho")
    parser.add_argument("--file", required=True, type=Path, help="ไฟล์เสียงต้นฉบับ (.wav/.flac/...)")
    parser.add_argument("--session-id", required=True, type=str, help="ชื่อ session (เช่น ext_20260820)")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw/sleep"), help="โฟลเดอร์ปลายทางหลัก (ค่าเริ่มต้น data/raw/sleep)"
    )
    parser.add_argument("--dry-run", action="store_true", help="แสดงแผนการทำงานโดยไม่แตะไฟล์ใดๆ จริง")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    if not args.file.is_file():
        raise SystemExit(f"ไม่พบไฟล์: {args.file}")

    import_file(args.file, args.session_id, args.output_root, args.dry_run)


if __name__ == "__main__":
    main()
