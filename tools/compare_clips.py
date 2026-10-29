"""Compare spectral features between two groups of grinding clips.

Groups are taken from the filename prefix before the first underscore
(``awake_lateral_01.wav`` → ``awake``, ``sleep_sister_02.wav`` → ``sleep``),
so the awake-vs-asleep question this project ran into can be answered
directly from a folder of clips.

Why this exists: the energy gate's 2000-16000 Hz band came from Sprint 1's
deliberate tooth-rubbing clips recorded ~30 cm from the mic. Clips of real
sleep bruxism look spectrally very different, which would mean the gate is
listening where the signal isn't. This tool quantifies that difference and
reports where each group's energy actually sits, so ``--low-hz/--high-hz``
for ``tools/detect_events.py`` (and ``tools/sweep_band.py``) can be chosen
from data instead of assumption.

Usage:
    python -m tools.compare_clips --dir data/raw/grind_clips
    python -m tools.compare_clips --dir <dir> --highpass 0 --output results/compare_raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.features.spectral as spectral
from src.audio.io import find_audio_files, load_audio

logger = logging.getLogger("bruxecho.compare_clips")

# Finer than spectral.DEFAULT_BANDS specifically in 500-8000 Hz, because
# that's where the decision lives: the sleep clips' centroid sits near
# 3 kHz, and 2k-4kHz-wide buckets are too coarse to place a band edge.
COMPARISON_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-500Hz", 0.0, 500.0),
    ("500-1kHz", 500.0, 1000.0),
    ("1k-2kHz", 1000.0, 2000.0),
    ("2k-3kHz", 2000.0, 3000.0),
    ("3k-4kHz", 3000.0, 4000.0),
    ("4k-6kHz", 4000.0, 6000.0),
    ("6k-8kHz", 6000.0, 8000.0),
    ("8k-12kHz", 8000.0, 12000.0),
    ("12k-16kHz", 12000.0, 16000.0),
    ("16kHz+", 16000.0, float("inf")),
)

# Level-invariant shape features — the only ones comparable across groups
# recorded at different gain/distance (see print_report's caveat).
BOXPLOT_FEATURES: tuple[tuple[str, str], ...] = (
    ("duration_s", "duration (s)"),
    ("rms_dbfs", "RMS (dBFS)"),
    ("energy_above_8k_pct", "energy >8 kHz (%)"),
    ("spectral_centroid_mean_hz", "spectral centroid (Hz)"),
    ("spectral_flatness", "spectral flatness"),
    ("zero_crossing_rate", "zero-crossing rate"),
    ("spectral_rolloff_85_hz", "rolloff 85% (Hz)"),
    ("energy_p95_hz", "95% energy below (Hz)"),
)

ENERGY_PERCENTILES: tuple[float, ...] = (5.0, 50.0, 95.0)


def group_of(path: Path) -> str:
    """Group label = filename prefix before the first underscore."""
    stem = path.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def energy_percentile_hz(x: np.ndarray, sr: int, percentiles: tuple[float, ...]) -> dict[float, float]:
    """Frequency below which each given percentage of total energy lies.

    Complements librosa's rolloff (which is frame-wise then averaged) with a
    whole-clip figure, and gives the low edge too — a rolloff only ever
    answers "where does the energy stop", never "where does it start".
    """
    freqs, power = spectral.power_spectrum(x, sr)
    total = float(power.sum()) if power.size else 0.0
    if total <= 0.0:
        return {p: 0.0 for p in percentiles}
    cumulative = np.cumsum(power) / total
    return {
        p: float(freqs[min(int(np.searchsorted(cumulative, p / 100.0)), freqs.size - 1)])
        for p in percentiles
    }


def compute_clip_rows(clips: list[tuple[Path, np.ndarray, int]]) -> pd.DataFrame:
    """One feature row per clip."""
    rows: list[dict] = []
    for path, signal, sr in clips:
        centroid_mean, centroid_std = spectral.spectral_centroid_stats(signal, sr)
        percentiles = energy_percentile_hz(signal, sr, ENERGY_PERCENTILES)
        row: dict[str, object] = {
            "filename": path.name,
            "group": group_of(path),
            "sr_hz": sr,
            "duration_s": signal.size / sr if sr else 0.0,
            "rms_dbfs": spectral.rms_dbfs(signal),
            "peak_dbfs": spectral.peak_dbfs(signal),
            "energy_above_8k_pct": spectral.high_freq_ratio(signal, sr, 8000.0) * 100.0,
            "spectral_centroid_mean_hz": centroid_mean,
            "spectral_centroid_std_hz": centroid_std,
            "spectral_rolloff_85_hz": spectral.spectral_rolloff_mean(signal, sr, 0.85),
            "spectral_rolloff_95_hz": spectral.spectral_rolloff_mean(signal, sr, 0.95),
            "spectral_flatness": spectral.spectral_flatness_mean(signal),
            "zero_crossing_rate": spectral.zero_crossing_rate_mean(signal),
        }
        for p, hz in percentiles.items():
            row[f"energy_p{p:.0f}_hz"] = hz
        for label, pct in spectral.band_energy_ratios(signal, sr, COMPARISON_BANDS).items():
            row[f"band_{label}_pct"] = pct
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Median/min/max per group for every numeric feature.

    Median rather than mean: these groups have 5-6 clips each, small enough
    that one outlier would drag a mean around.
    """
    numeric = df.select_dtypes(include=[np.number]).columns
    grouped = df.groupby("group")[list(numeric)]
    summary = grouped.agg(["median", "min", "max", "count"])
    return summary.T


def plot_boxplots(df: pd.DataFrame, output_path: Path, groups: list[str]) -> None:
    """Box plot per feature, one box per group.

    Plot text is kept in English (unlike this tool's console output) to match
    every other chart in this project — the default matplotlib font has no
    Thai glyphs.
    """
    features = [(col, label) for col, label in BOXPLOT_FEATURES if col in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, (col, label) in zip(axes, features):
        data = [df.loc[df["group"] == g, col].dropna().to_numpy() for g in groups]
        # Labels are set via set_xticks rather than boxplot's own keyword,
        # which was renamed labels -> tick_labels across matplotlib versions.
        ax.boxplot(data)
        ax.set_xticks(range(1, len(groups) + 1), groups)
        # Overlay the individual clips: with 5-6 per group the box alone
        # hides how much of the "difference" is really just one clip.
        for i, values in enumerate(data, start=1):
            jitter = rng.normal(0.0, 0.04, values.size)
            ax.scatter(np.full(values.size, i) + jitter, values, s=18, alpha=0.7, color="crimson", zorder=3)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    for ax in axes[len(features) :]:
        ax.set_visible(False)

    fig.suptitle("Clip feature comparison by group", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_band_profile(df: pd.DataFrame, output_path: Path, groups: list[str]) -> None:
    """Median per-band energy share for each group — the band-choice chart."""
    band_cols = [f"band_{label}_pct" for label, _, _ in COMPARISON_BANDS if f"band_{label}_pct" in df.columns]
    labels = [c.removeprefix("band_").removesuffix("_pct") for c in band_cols]
    x = np.arange(len(band_cols))
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, group in enumerate(groups):
        medians = [df.loc[df["group"] == group, c].median() for c in band_cols]
        ax.bar(x + i * width - 0.4 + width / 2, medians, width, label=f"{group} (n={(df['group'] == group).sum()})")

    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("median share of total energy (%)")
    ax.set_xlabel("frequency band")
    ax.set_title("Where each group's energy actually sits")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_report(df: pd.DataFrame, groups: list[str], highpass_hz: float) -> None:
    print()
    print("═" * 78)
    print(f"เปรียบเทียบคลิปตามกลุ่ม (high-pass {highpass_hz:.0f} Hz)" if highpass_hz > 0 else "เปรียบเทียบคลิปตามกลุ่ม (ไม่ผ่าน high-pass)")
    print("═" * 78)
    for group in groups:
        sub = df[df["group"] == group]
        print(f"\nกลุ่ม {group} ({len(sub)} คลิป): {', '.join(sub['filename'])}")

    key_features = [
        ("duration_s", "ความยาว (s)", "{:.2f}"),
        ("rms_dbfs", "RMS (dBFS)", "{:.1f}"),
        ("energy_above_8k_pct", "พลังงาน >8 kHz (%)", "{:.2f}"),
        ("spectral_centroid_mean_hz", "spectral centroid (Hz)", "{:.0f}"),
        ("spectral_flatness", "spectral flatness", "{:.5f}"),
        ("zero_crossing_rate", "ZCR", "{:.3f}"),
        ("spectral_rolloff_85_hz", "rolloff 85% (Hz)", "{:.0f}"),
        ("energy_p5_hz", "พลังงาน 5% แรกอยู่ต่ำกว่า (Hz)", "{:.0f}"),
        ("energy_p50_hz", "พลังงานครึ่งหนึ่งอยู่ต่ำกว่า (Hz)", "{:.0f}"),
        ("energy_p95_hz", "พลังงาน 95% อยู่ต่ำกว่า (Hz)", "{:.0f}"),
    ]

    print()
    name_width = 34
    header = f"{'ฟีเจอร์ (median)':<{name_width}}" + "".join(f"{g:>16}" for g in groups)
    print(header)
    print("─" * 78)
    for col, label, fmt in key_features:
        if col not in df.columns:
            continue
        cells = "".join(f"{fmt.format(df.loc[df['group'] == g, col].median()):>16}" for g in groups)
        print(f"{label:<{name_width}}{cells}")

    print()
    print("ส่วนแบ่งพลังงานรายย่าน (median %):")
    band_labels = [label for label, _, _ in COMPARISON_BANDS if f"band_{label}_pct" in df.columns]
    print(f"{'ย่าน':<{name_width}}" + "".join(f"{g:>16}" for g in groups))
    print("─" * 78)
    for label in band_labels:
        col = f"band_{label}_pct"
        cells = "".join(f"{df.loc[df['group'] == g, col].median():>16.2f}" for g in groups)
        print(f"{label:<{name_width}}{cells}")

    print()
    print("═" * 78)
    print("ข้อควรระวังก่อนสรุป:")
    print("  • RMS/peak เทียบข้ามกลุ่มไม่ได้ ถ้าคลิปมาจากคนละไฟล์ต้นทาง/คนละระยะไมค์/คนละ gain")
    print("    ให้ดูเฉพาะฟีเจอร์ที่ไม่ขึ้นกับความดัง (centroid, flatness, ZCR, ส่วนแบ่งพลังงานรายย่าน)")
    print("  • ถ้าความยาวคลิปสองกลุ่มต่างกันมาก คลิปยาวมักมีช่วงเงียบคั่น ทำให้ RMS/ZCR เอนไปทางเงียบ")
    for group in groups:
        sub = df[df["group"] == group]
        p5 = sub["energy_p5_hz"].median()
        p95 = sub["energy_p95_hz"].median()
        print(f"  • กลุ่ม {group}: พลังงาน 90% กลางอยู่ประมาณ {p5:.0f}-{p95:.0f} Hz")
    print("═" * 78)
    print("ขั้นถัดไป: เอาช่วงข้างบนไปตั้ง --low-hz/--high-hz ใน tools/sweep_band.py แล้วเทียบจำนวน event จริง")


def run(input_dir: Path, output_root: Path, highpass_hz: float) -> None:
    files = find_audio_files(input_dir)
    if not files:
        raise SystemExit(f"ไม่พบไฟล์ .wav/.flac ใน {input_dir}")

    clips: list[tuple[Path, np.ndarray, int]] = []
    sample_rates: set[int] = set()
    for path in files:
        signal, sr = load_audio(path)
        if signal.size == 0:
            logger.warning("ข้ามไฟล์ว่าง: %s", path.name)
            continue
        sample_rates.add(sr)
        clips.append((path, spectral.high_pass_filter(signal, sr, highpass_hz), sr))
        logger.info("โหลด %s (กลุ่ม=%s, sr=%d Hz, %.2fs)", path.name, group_of(path), sr, signal.size / sr)

    if not clips:
        raise SystemExit("ไม่มีคลิปที่อ่านได้เลย")
    if len(sample_rates) > 1:
        # Band-share and >8 kHz numbers are only comparable at equal Nyquist:
        # a 16 kHz-sampled clip has no energy above 8 kHz by construction.
        logger.warning(
            "คลิปมี sample rate ไม่เท่ากัน (%s) — ตัวเลขพลังงานย่านสูงเทียบข้ามกลุ่มไม่ได้ตรงๆ",
            ", ".join(f"{sr} Hz" for sr in sorted(sample_rates)),
        )

    df = compute_clip_rows(clips)
    groups = sorted(df["group"].unique())
    if len(groups) < 2:
        logger.warning("พบกลุ่มเดียว (%s) — จะไม่มีอะไรให้เทียบ", groups[0] if groups else "-")

    output_root.mkdir(parents=True, exist_ok=True)
    clip_csv = output_root / "clip_features.csv"
    df.to_csv(clip_csv, index=False)
    summary_csv = output_root / "group_summary.csv"
    summarize_groups(df).to_csv(summary_csv)
    boxplot_path = output_root / "boxplots.png"
    plot_boxplots(df, boxplot_path, groups)
    band_path = output_root / "band_profile.png"
    plot_band_profile(df, band_path, groups)

    logger.info("บันทึก %s", clip_csv)
    logger.info("บันทึก %s", summary_csv)
    logger.info("บันทึก %s", boxplot_path)
    logger.info("บันทึก %s", band_path)

    print_report(df, groups, highpass_hz)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="เทียบฟีเจอร์เสียงของคลิปสองกลุ่ม (แยกกลุ่มด้วย prefix ของชื่อไฟล์ เช่น awake_ / sleep_)"
    )
    parser.add_argument("--dir", required=True, type=Path, help="โฟลเดอร์ที่มีคลิป .wav/.flac")
    parser.add_argument(
        "--highpass",
        type=float,
        default=300.0,
        help="ความถี่ตัดของ high-pass filter (Hz) ใส่ 0 เพื่อปิด (ค่าเริ่มต้น 300 Hz "
        "เหมือน tools/explore.py เพื่อไม่ให้เสียงรบกวนความถี่ต่ำกลบตัวเลขสัดส่วนพลังงาน)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/compare_clips"), help="โฟลเดอร์ปลายทาง"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    if not args.dir.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ {args.dir}")

    run(args.dir, args.output, args.highpass)


if __name__ == "__main__":
    main()
