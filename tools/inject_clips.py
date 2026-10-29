"""แทรกคลิปเสียงอ้างอิงลงใน "ไฟล์เสียง" ของ session เพื่อสร้าง ground truth

หัวใจของเครื่องมือนี้คือ **แทรกลงในไฟล์เสียง ไม่ใช่เขียนลง events.json**
ผลการตรวจจับต้องมาจากการที่โมเดลเจอเสียงนั้นเองเท่านั้น เครื่องมือนี้จึงรู้
แค่ว่าแทรกอะไรไว้ตรงไหน แล้วปล่อยให้ process_session ตรวจใหม่ทั้งหมด

วิธีใช้:
    python -m tools.inject_clips --session data/raw/sleep/dc_demo
        --clips "data/raw/grind_clips/sleep_sister_0*.wav"
        --snr-db 15 --output data/raw/sleep/dc_demo_mixed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.mixing import (
    DEFAULT_FADE_MS,
    DEFAULT_MIN_GAP_S,
    Placement,
    active_spans,
    apply_fade,
    gain_db,
    gain_for_snr,
    mix_into,
    plan_positions,
    resample_to,
    rms,
    rms_dbfs_of,
    to_mono,
    trim_silence,
)
from src.session.schema import read_session_json

logger = logging.getLogger("bruxecho.inject_clips")

GROUND_TRUTH_FILE = "injected_ground_truth.csv"
PROVENANCE = "synthetic_mixed"
NOISE_WINDOW_S = 5.0


def load_clip(path: Path, target_sr: int, trim: bool) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    mono = to_mono(np.asarray(audio, dtype=np.float32))
    resampled = resample_to(mono, sr, target_sr)
    return trim_silence(resampled, target_sr) if trim else resampled


def local_noise_rms(track: np.ndarray, start_sample: int, span: int, sr: int) -> float:
    """Background level just before and after where the clip will land.

    Local rather than whole-night: a night's floor drifts by tens of dB (fan
    cycling, traffic, someone turning over), so one global noise figure would
    put some injections far too quiet to be fair and others implausibly loud.
    """
    window = int(NOISE_WINDOW_S * sr)
    before = track[max(0, start_sample - window) : start_sample]
    after = track[start_sample + span : start_sample + span + window]
    neighborhood = np.concatenate([before, after]) if before.size or after.size else track[:window]
    if neighborhood.size == 0:
        return 0.0
    return rms(neighborhood)


def collect_clips(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path(p) for p in glob.glob(pattern))
        if matched:
            found.extend(matched)
        elif Path(pattern).is_file():
            found.append(Path(pattern))
    return found


def copy_session(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise SystemExit(f"{destination} มีอยู่แล้ว — ใช้ --force ถ้าต้องการเขียนทับ")
        shutil.rmtree(destination)
    # copytree rather than editing in place: the original session is raw data
    # and must stay exactly as recorded.
    shutil.copytree(source, destination)
    logger.info("คัดลอก session ไปที่ %s", destination)


def write_ground_truth(path: Path, placements: list[Placement]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_s",
                "end_s",
                "duration_s",
                "source_clip",
                "snr_db",
                "requested_snr_db",
                "gain_db",
                "noise_rms_dbfs",
            ]
        )
        for p in placements:
            writer.writerow(
                [
                    f"{p.start_s:.3f}",
                    f"{p.end_s:.3f}",
                    f"{p.end_s - p.start_s:.3f}",
                    p.source_clip,
                    f"{p.achieved_snr_db:.2f}",
                    f"{p.requested_snr_db:.2f}",
                    f"{p.gain_db:.2f}",
                    f"{p.noise_rms_dbfs:.2f}",
                ]
            )


def update_session_json(session_dir: Path, source_session_id: str, placed: int) -> None:
    """Mark the copy as synthetic so nothing downstream mistakes it for a real night."""
    path = session_dir / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["session_id"] = session_dir.name
    data["provenance"] = PROVENANCE
    data["synthetic_source_session"] = source_session_id
    data["synthetic_injected_count"] = placed
    warnings = list(data.get("quality_warnings") or [])
    if "contains_injected_reference_clips" not in warnings:
        warnings.append("contains_injected_reference_clips")
    data["quality_warnings"] = warnings
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(
    session_dir: Path,
    clip_patterns: list[str],
    snr_db: float,
    output_dir: Path,
    count: int | None,
    min_gap_s: float,
    fade_ms: float,
    no_trim: bool,
    force: bool,
) -> None:
    session = read_session_json(session_dir)
    if len(session.segments) != 1:
        raise SystemExit(
            f"รองรับเฉพาะ session ที่มี segment เดียว (นี่มี {len(session.segments)}) "
            "— ถ้าต้องใช้กับหลาย segment ต้องขยายเครื่องมือก่อน"
        )

    clip_paths = collect_clips(clip_patterns)
    if not clip_paths:
        raise SystemExit(f"ไม่พบคลิปที่ตรงกับ {clip_patterns}")

    copy_session(session_dir, output_dir, force)

    segment_name = session.segments[0].file
    target = output_dir / segment_name
    audio, sr = sf.read(str(target), dtype="float32", always_2d=False)
    track = to_mono(np.asarray(audio, dtype=np.float32))
    total_s = track.size / sr
    logger.info("ไฟล์เสียง %s: %.1f วินาที @ %d Hz", segment_name, total_s, sr)

    clips = [(p.name, load_clip(p, sr, trim=not no_trim)) for p in clip_paths]
    clips = [(name, data) for name, data in clips if data.size > 0]
    if not clips:
        raise SystemExit("คลิปทั้งหมดว่างเปล่าหลังตัดความเงียบ")

    spans = active_spans(track, sr)
    active_s = sum(end - start for start, end in spans)
    logger.info(
        "ช่วงที่มีเสียงจริง %d ช่วง รวม %.0f วินาที (%.0f%% ของไฟล์) — ที่เหลือเป็นความเงียบดิจิทัล",
        len(spans), active_s, 100.0 * active_s / max(total_s, 1e-9),
    )
    if not spans:
        raise SystemExit("ไม่พบช่วงที่มีเสียงจริงเลย — แทรกแล้วจะวัด SNR ไม่ได้")

    requested = count if count is not None else len(clips)
    durations = [data.size / sr for _, data in clips]
    positions = plan_positions(spans, requested, durations, min_gap_s=min_gap_s)
    if not positions:
        raise SystemExit(
            f"ช่วงที่มีเสียงจริงรวม {active_s:.0f} วินาที ใส่ไม่ได้ที่ระยะห่าง {min_gap_s:.0f} วินาที"
        )
    if len(positions) < requested:
        logger.warning(
            "ใส่ได้ %d จุดจากที่ขอ %d (ช่วงที่มีเสียงจริงไม่พอที่ระยะห่าง %.0f วินาที)",
            len(positions),
            requested,
            min_gap_s,
        )

    placements: list[Placement] = []
    for index, start_s in enumerate(positions):
        name, clip = clips[index % len(clips)]
        faded = apply_fade(clip, sr, fade_ms)
        start_sample = int(round(start_s * sr))
        noise_rms = local_noise_rms(track, start_sample, faded.size, sr)
        wanted_gain = gain_for_snr(faded, noise_rms, snr_db)
        track, used_gain = mix_into(track, faded, start_sample, wanted_gain)
        noise_dbfs = 20.0 * float(np.log10(max(noise_rms, 1e-12)))
        achieved = rms_dbfs_of(faded * used_gain) - noise_dbfs
        placements.append(
            Placement(
                start_s=start_s,
                end_s=start_s + faded.size / sr,
                source_clip=name,
                requested_snr_db=snr_db,
                achieved_snr_db=achieved,
                gain_db=gain_db(used_gain),
                noise_rms_dbfs=noise_dbfs,
            )
        )

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    logger.info("peak หลังผสม %.3f (%.1f dBFS)", peak, 20 * np.log10(max(peak, 1e-12)))
    sf.write(str(target), track, sr, subtype="PCM_16")

    ground_truth = output_dir / GROUND_TRUTH_FILE
    write_ground_truth(ground_truth, placements)
    update_session_json(output_dir, session.session_id, len(placements))

    print()
    print("=" * 70)
    print(f"แทรกคลิปอ้างอิง {len(placements)} จุด ลงใน {output_dir}")
    print("=" * 70)
    print(f"{'#':>3}{'เวลา (วินาที)':>16}{'ยาว':>8}{'SNR จริง':>11}{'gain':>9}  คลิป")
    for i, p in enumerate(placements, start=1):
        print(
            f"{i:>3}{p.start_s:>16.1f}{p.end_s - p.start_s:>8.2f}"
            f"{p.achieved_snr_db:>11.1f}{p.gain_db:>9.1f}  {p.source_clip}"
        )
    print()
    print(f"ground truth: {ground_truth}")
    print(f"provenance  : {PROVENANCE} (กันสับสนกับคืนจริง)")
    print()
    print("ขั้นถัดไป — ให้โมเดลตรวจเอง (ห้ามเขียน event เองเด็ดขาด):")
    print(f"  python -m tools.process_session --session {output_dir}")
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="แทรกคลิปอ้างอิงลงในไฟล์เสียงของ session")
    parser.add_argument("--session", required=True, type=Path, help="session ต้นฉบับ (ไม่ถูกแก้ไข)")
    parser.add_argument("--clips", required=True, nargs="+", help="ไฟล์คลิป หรือ glob pattern")
    parser.add_argument("--snr-db", type=float, default=15.0, help="ระดับเหนือพื้นเสียงที่ต้องการ (dB)")
    parser.add_argument("--output", required=True, type=Path, help="โฟลเดอร์ session ใหม่")
    parser.add_argument("--count", type=int, default=None, help="จำนวนจุดที่จะแทรก (ค่าเริ่มต้น = จำนวนคลิป)")
    parser.add_argument("--min-gap-s", type=float, default=DEFAULT_MIN_GAP_S, help="ระยะห่างขั้นต่ำระหว่างจุด")
    parser.add_argument("--fade-ms", type=float, default=DEFAULT_FADE_MS, help="ความยาว crossfade")
    parser.add_argument("--no-trim", action="store_true", help="ไม่ต้องตัดความเงียบหัวท้ายคลิป")
    parser.add_argument("--force", action="store_true", help="เขียนทับโฟลเดอร์ปลายทางถ้ามีอยู่")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(
        args.session,
        args.clips,
        args.snr_db,
        args.output,
        args.count,
        args.min_gap_s,
        args.fade_ms,
        args.no_trim,
        args.force,
    )


if __name__ == "__main__":
    main()
