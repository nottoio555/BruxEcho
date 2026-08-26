"""Signal-exploration CLI for BruxEcho pilot recordings.

Answers three research questions ahead of model selection:

1. Is grinding sound separable from silence (SNR)?
2. Where does the energy live spectrally, and how much sits above 8 kHz —
   the frequency ceiling YAMNet (16 kHz native) can "see"? Above that
   threshold, PANNs (32 kHz native) becomes the better fit despite its
   larger size.
3. How does grinding differ spectrally from snoring/movement/noise?

This module is a CLI: plotting, argument parsing, and printing live here.
All numeric computation is delegated to ``src.audio.io`` and
``src.features.spectral``, which stay side-effect-free.

Usage:
    python -m tools.explore --input data/raw/pilot
    python -m tools.explore --input data/raw/pilot --baseline 01_silence.wav
    python -m tools.explore --input data/raw/pilot --output results/explore/run2
    python -m tools.explore --input data/raw/pilot --highpass 0   # disable the high-pass filter
    python -m tools.explore --input data/raw/pilot --trim-start 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audio.io import find_audio_files, load_audio, trim_start
from src.features import spectral as sf

logger = logging.getLogger("bruxecho.explore")

MIN_SR_FOR_HIGH_FREQ_ANALYSIS = 32_000
YAMNET_LIMIT_HZ = 8_000
BAND_SNR_DECISION_THRESHOLD_DB = 6.0
BAND_SNR_DECISION_BAND = "8k_16k"


def _normalize_peak(x: np.ndarray) -> np.ndarray:
    """Peak-normalize a signal to [-1, 1] for visualization only.

    Never used ahead of RMS/SNR/spectral computations — those must run on
    the untouched signal so absolute levels stay meaningful.
    """
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak <= 0.0:
        return x
    return x / peak


def _is_baseline_name(stem: str) -> bool:
    return "silence" in stem.lower()


def _short_label(path: Path) -> str:
    return path.stem


def discover_clips(
    input_dir: Path, baseline_arg: str | None, trim_start_s: float = 0.0
) -> tuple[list[tuple[Path, np.ndarray, int]], Path | None]:
    """Load every supported audio file in ``input_dir`` and pick a baseline.

    Args:
        input_dir: Directory containing pilot recordings.
        baseline_arg: Filename (as passed via ``--baseline``) to force as
            the baseline clip, or ``None`` to auto-detect by "silence" in
            the filename.
        trim_start_s: Seconds to drop from the start of every clip (e.g. to
            remove a record-button click), before anything else runs.

    Returns:
        Tuple of (loaded clips as ``(path, signal, sr)``, baseline path or
        ``None`` if none could be identified).
    """
    files = find_audio_files(input_dir)
    if not files:
        raise SystemExit(f"ไม่พบไฟล์ .wav/.flac ใน {input_dir}")

    clips: list[tuple[Path, np.ndarray, int]] = []
    for path in files:
        signal, sr = load_audio(path)
        if trim_start_s > 0.0:
            signal = trim_start(signal, sr, trim_start_s)
        if sr < MIN_SR_FOR_HIGH_FREQ_ANALYSIS:
            logger.warning(
                "%s: sample rate %d Hz < %d Hz — ไม่สามารถตอบคำถามเรื่องพลังงาน"
                " เหนือ 8 kHz ได้อย่างน่าเชื่อถือ (Nyquist = %.0f Hz)."
                " แนะนำให้อัดใหม่ที่ 44.1 kHz",
                path.name,
                sr,
                MIN_SR_FOR_HIGH_FREQ_ANALYSIS,
                sr / 2.0,
            )
        clips.append((path, signal, sr))
        logger.info("โหลด %s (sr=%d Hz, duration=%.2fs)", path.name, sr, signal.size / sr)

    baseline_path: Path | None = None
    if baseline_arg:
        matches = [p for p, _, _ in clips if p.name == baseline_arg]
        if not matches:
            logger.warning("ไม่พบไฟล์ baseline ที่ระบุ: %s", baseline_arg)
        else:
            baseline_path = matches[0]
    else:
        matches = [p for p, _, _ in clips if _is_baseline_name(p.stem)]
        if matches:
            baseline_path = matches[0]

    if baseline_path is None:
        logger.warning(
            "ไม่พบไฟล์ baseline (ไม่มีคำว่า 'silence' ในชื่อไฟล์ และไม่ได้ระบุ --baseline)"
            " — จะไม่สามารถคำนวณ SNR ได้"
        )
    else:
        logger.info("ใช้ baseline: %s", baseline_path.name)

    return clips, baseline_path


def compute_summary_rows(
    clips: list[tuple[Path, np.ndarray, int]], baseline_path: Path | None
) -> pd.DataFrame:
    """Compute the full feature set for every clip.

    Args:
        clips: Loaded clips as ``(path, signal, sr)`` — expected to already
            be high-pass filtered (and trimmed) by the caller, since every
            metric here is computed directly from ``signal``.
        baseline_path: Path of the baseline clip, if identified.

    Returns:
        One row per clip with every computed metric, including a per-band
        SNR column (``snr_<band>_db``) for each of ``sf.SNR_BANDS``.
    """
    baseline_rms = None
    baseline_band_db = None
    if baseline_path is not None:
        for path, signal, sr in clips:
            if path == baseline_path:
                baseline_rms = sf.rms_dbfs(signal)
                baseline_band_db = sf.band_power_db_dict(signal, sr, sf.SNR_BANDS)
                break

    rows: list[dict] = []
    for path, signal, sr in clips:
        rms_db = sf.rms_dbfs(signal)
        bands = sf.band_energy_ratios(signal, sr)
        centroid_mean, centroid_std = sf.spectral_centroid_stats(signal, sr)
        band_db = sf.band_power_db_dict(signal, sr, sf.SNR_BANDS)
        row = {
            "filename": path.name,
            "is_baseline": path == baseline_path,
            "sr_hz": sr,
            "duration_s": signal.size / sr if sr else 0.0,
            "rms_dbfs": rms_db,
            "peak_dbfs": sf.peak_dbfs(signal),
            "snr_db": sf.snr_db(rms_db, baseline_rms) if baseline_rms is not None else np.nan,
            "high_freq_ratio_pct": sf.high_freq_ratio(signal, sr, YAMNET_LIMIT_HZ) * 100.0,
            "spectral_centroid_mean_hz": centroid_mean,
            "spectral_centroid_std_hz": centroid_std,
            "spectral_rolloff_85_hz": sf.spectral_rolloff_mean(signal, sr, 0.85),
            "spectral_rolloff_95_hz": sf.spectral_rolloff_mean(signal, sr, 0.95),
            "spectral_flatness": sf.spectral_flatness_mean(signal),
            "zero_crossing_rate": sf.zero_crossing_rate_mean(signal),
        }
        for label, _, _ in sf.SNR_BANDS:
            row[f"snr_{label}_db"] = (
                sf.snr_db(band_db[label], baseline_band_db[label])
                if baseline_band_db is not None
                else np.nan
            )
        for label, pct in bands.items():
            row[f"{label}_pct"] = pct
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a human-readable summary table to the console."""
    display_cols = [
        "filename",
        "sr_hz",
        "duration_s",
        "rms_dbfs",
        "peak_dbfs",
        "snr_db",
        "high_freq_ratio_pct",
        "spectral_centroid_mean_hz",
        "spectral_flatness",
        "zero_crossing_rate",
    ]
    table = df[display_cols].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].round(2)
    print()
    print(table.to_string(index=False))
    print()


def print_decision_summary(df: pd.DataFrame) -> None:
    """Print the final go/no-go conclusion for YAMNet vs. PANNs.

    Decision is based on per-band SNR (clip vs. baseline) rather than the
    share of total energy above 8 kHz: a low-frequency-dominated recording
    can have a tiny ``high_freq_ratio_pct`` even when the high-frequency
    band is clearly separable from the noise floor, and vice versa.
    """
    non_baseline = df[~df["is_baseline"]]
    grind_rows = non_baseline[non_baseline["filename"].str.contains("grind", case=False)]
    if grind_rows.empty:
        logger.warning("ไม่พบไฟล์ที่มีคำว่า 'grind' ในชื่อ — ใช้ไฟล์ทั้งหมด (ยกเว้น baseline) แทน")
        grind_rows = non_baseline

    band_cols = [f"snr_{label}_db" for label, _, _ in sf.SNR_BANDS]

    print("═" * 41)
    print("ข้อสรุปสำหรับการเลือกโมเดล")
    print("═" * 41)

    if grind_rows.empty or grind_rows[band_cols].isna().all().all():
        print("SNR แยกย่านความถี่: ไม่สามารถคำนวณได้ (ไม่มี baseline)")
        print("═" * 41)
        return

    print("SNR เฉลี่ยแยกย่านความถี่ (คลิปถูฟัน เทียบ baseline):")
    band_means: dict[str, float] = {}
    for label, _, _ in sf.SNR_BANDS:
        col = f"snr_{label}_db"
        mean_val = float(grind_rows[col].mean())
        band_means[label] = mean_val
        print(f"  {label:>10s}: {mean_val:6.1f} dB")

    decision_val = band_means.get(BAND_SNR_DECISION_BAND, float("nan"))

    print()
    if pd.notna(decision_val) and decision_val >= BAND_SNR_DECISION_THRESHOLD_DB:
        print(
            f"→ SNR ย่าน {BAND_SNR_DECISION_BAND} = {decision_val:.1f} dB"
            f" (>= {BAND_SNR_DECISION_THRESHOLD_DB:.0f} dB เกณฑ์แยกได้)"
        )
        print("  ย่านเหนือ 8 kHz มีสัญญาณจริง แนะนำใช้ PANNs (32 kHz) แทน YAMNet (16 kHz)")
        print("  หรือเพิ่มฟีเจอร์ย่านความถี่สูงเข้าไปเสริม")
    else:
        print(
            f"→ SNR ย่าน {BAND_SNR_DECISION_BAND} = {decision_val:.1f} dB"
            f" (< {BAND_SNR_DECISION_THRESHOLD_DB:.0f} dB เกณฑ์แยกได้)"
        )
        print("  ย่านเหนือ 8 kHz ไม่มีสัญญาณนัยสำคัญ YAMNet (16 kHz) น่าจะเพียงพอสำหรับงานนี้")
    print("═" * 41)


def plot_spectrograms(
    clips: list[tuple[Path, np.ndarray, int]], output_path: Path, highpass_hz: float
) -> None:
    """Grid of log-mel spectrograms, raw vs. high-pass filtered side by side.

    Spectrograms are the one plot that intentionally shows the *unfiltered*
    signal too (next to the filtered version) — every other number and plot
    in this tool uses the filtered signal, so this is the place to visually
    confirm the filter isn't discarding real high-frequency content.
    """
    n = len(clips)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n), squeeze=False)

    filter_label = f"high-pass {highpass_hz:.0f} Hz" if highpass_hz > 0 else "high-pass (off)"
    for idx, (path, signal, sr) in enumerate(clips):
        variants = (
            ("raw", signal),
            (filter_label, sf.high_pass_filter(signal, sr, highpass_hz)),
        )
        for col, (variant_label, sig) in enumerate(variants):
            ax = axes[idx][col]
            log_mel, times, mel_freqs = sf.log_mel_spectrogram(_normalize_peak(sig), sr)
            mesh = ax.pcolormesh(times, mel_freqs, log_mel, shading="auto", cmap="magma")
            if sr / 2.0 > YAMNET_LIMIT_HZ:
                ax.axhline(YAMNET_LIMIT_HZ, color="red", linestyle="--", linewidth=1.2)
                ax.text(
                    times[-1] if times.size else 0.0,
                    YAMNET_LIMIT_HZ,
                    " YAMNet limit",
                    color="red",
                    fontsize=8,
                    va="bottom",
                    ha="right",
                )
            ax.set_title(f"{_short_label(path)} ({variant_label})", fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(mesh, ax=ax, format="%+2.0f dB")

    fig.suptitle("Log-mel spectrograms: raw vs. high-pass (dashed red = YAMNet 8 kHz ceiling)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("บันทึก %s", output_path)


def plot_psd_comparison(
    clips: list[tuple[Path, np.ndarray, int]], output_path: Path
) -> None:
    """Overlaid Welch PSD of every clip, log-frequency axis, 8 kHz marker."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for path, signal, sr in clips:
        freqs, psd = sf.welch_psd(signal, sr)
        psd_db = sf.power_to_db(psd)
        ax.plot(freqs, psd_db, label=_short_label(path), linewidth=1.3)

    ax.axvline(YAMNET_LIMIT_HZ, color="red", linestyle="--", linewidth=1.2, label="YAMNet limit (8 kHz)")
    ax.set_xscale("log")
    ax.set_xlim(left=20)
    ax.set_xlabel("Frequency (Hz) [log scale]")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.set_title("Power Spectral Density comparison (Welch)")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("บันทึก %s", output_path)


def plot_band_energy(df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of per-band energy percentage across clips."""
    band_cols = [f"{label}_pct" for label, _, _ in sf.DEFAULT_BANDS]
    labels = [_short_label(Path(f)) for f in df["filename"]]

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 5.5))
    bottom = np.zeros(len(df))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(band_cols) - 1)) for i in range(len(band_cols))]

    for col, color in zip(band_cols, colors):
        values = df[col].to_numpy()
        ax.bar(labels, values, bottom=bottom, label=col.replace("_pct", ""), color=color)
        bottom += values

    ax.set_ylabel("Share of total spectral energy (%)")
    ax.set_xlabel("Clip")
    ax.set_title("Per-band energy distribution")
    ax.legend(title="Band", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("บันทึก %s", output_path)


def plot_band_snr(df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart of per-band SNR (dB) vs. baseline, one group per clip.

    Draws a horizontal reference line at 6 dB, the threshold this tool
    treats as "separable from the noise floor" (see ``print_decision_summary``).
    """
    band_labels = [label for label, _, _ in sf.SNR_BANDS]
    band_cols = [f"snr_{label}_db" for label in band_labels]

    plot_df = df[~df["is_baseline"]]
    if plot_df.empty or plot_df[band_cols].isna().all().all():
        logger.warning(
            "ข้ามการวาด %s: ไม่มีข้อมูล SNR แยกย่านความถี่ (ไม่มี baseline)", output_path.name
        )
        return

    labels = [_short_label(Path(f)) for f in plot_df["filename"]]
    x = np.arange(len(band_labels))
    width = 0.8 / max(1, len(plot_df))

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(band_labels)), 5.5))
    cmap = plt.get_cmap("viridis")
    for i, (label, (_, row)) in enumerate(zip(labels, plot_df.iterrows())):
        values = [row[col] for col in band_cols]
        ax.bar(x + i * width, values, width, label=label, color=cmap(i / max(1, len(plot_df) - 1)))

    ax.axhline(6.0, color="red", linestyle="--", linewidth=1.2, label="6 dB (separable)")
    ax.set_xticks(x + width * (len(plot_df) - 1) / 2)
    ax.set_xticklabels(band_labels)
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("SNR vs. baseline (dB)")
    ax.set_title("Per-band SNR")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("บันทึก %s", output_path)


def check_codec_artifacts(clips: list[tuple[Path, np.ndarray, int]]) -> None:
    """Warn if a narrow, severe spectral notch shows up at the same frequency
    in every clip.

    Run on the raw (unfiltered) signal — a dip that's consistent across
    totally different sounds (silence, grinding, tapping) at a fixed
    frequency isn't part of what was actually happening acoustically; it's
    more likely an artifact of compression or resampling somewhere in the
    recording chain, which would make high-frequency numbers computed from
    these files unreliable.
    """
    if len(clips) < 2:
        return

    dip_lists: list[list[float]] = []
    for _, signal, sr in clips:
        freqs, psd = sf.welch_psd(signal, sr)
        if freqs.size == 0:
            dip_lists.append([])
            continue
        psd_db = sf.power_to_db(psd)
        dip_lists.append(sf.detect_narrow_band_dips(freqs, psd_db))

    common = sf.find_common_frequency_dips(dip_lists)
    if common:
        logger.warning(
            "พบการลดพลังงานฉับพลัน (>10 dB ในช่วงแคบ) ที่ความถี่เดิมซ้ำในทุกไฟล์: %s"
            " — อาจเป็นร่องรอยของการบีบอัด/อัปแซมเปิลเสียง (codec/upsample artifact)"
            " ตัวเลขย่านความถี่สูงจากไฟล์เหล่านี้อาจไม่น่าเชื่อถือ",
            ", ".join(f"{f:.0f} Hz" for f in common),
        )


def plot_waveform_envelope(
    clips: list[tuple[Path, np.ndarray, int]], output_path: Path
) -> None:
    """Grid of waveform + RMS envelope per clip, to compare temporal shape."""
    n = len(clips)
    ncols = min(2, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.8 * nrows), squeeze=False)

    for idx, (path, signal, sr) in enumerate(clips):
        ax = axes[idx // ncols][idx % ncols]
        norm_signal = _normalize_peak(signal)
        t = np.arange(norm_signal.size) / sr
        ax.plot(t, norm_signal, color="steelblue", linewidth=0.4, alpha=0.6, label="waveform")

        offsets, rms_vals = sf.rms_envelope(norm_signal)
        ax.plot(offsets / sr, rms_vals, color="crimson", linewidth=1.5, label="RMS envelope")
        ax.plot(offsets / sr, -rms_vals, color="crimson", linewidth=1.5)

        ax.set_title(_short_label(path), fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (normalized)")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Waveform + RMS envelope (amplitude normalized for display only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("บันทึก %s", output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="สำรวจสัญญาณเสียงคลิปทดลอง BruxEcho (SNR, สเปกตรัม, ย่านความถี่)"
    )
    parser.add_argument("--input", required=True, type=Path, help="โฟลเดอร์ที่มีไฟล์เสียง (.wav/.flac)")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="ชื่อไฟล์ baseline (เช่น 01_silence.wav); ถ้าไม่ระบุจะหาไฟล์ที่มีคำว่า 'silence'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/explore"),
        help="โฟลเดอร์ปลายทางสำหรับกราฟและ summary.csv",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=300.0,
        help="ความถี่ตัดของ high-pass filter (Hz) ใส่ 0 เพื่อปิด (ค่าเริ่มต้น 300 Hz)"
        " ใช้กับทุกการคำนวณยกเว้นแผงเทียบ spectrogram (01) ที่วาดทั้งแบบ raw และ filtered",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=0.5,
        help="ตัดต้นคลิปออกกี่วินาที (ค่าเริ่มต้น 0.5s) เพื่อตัดเสียงกดปุ่มอัด/วางมือถือ",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Console output includes Thai text and box-drawing characters; force
    # UTF-8 so this doesn't crash on Windows terminals defaulting to a
    # legacy codepage (e.g. cp874/cp1252) that can't encode them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if not args.input.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ input: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    clips, baseline_path = discover_clips(args.input, args.baseline, args.trim_start)
    check_codec_artifacts(clips)

    filtered_clips = [
        (path, sf.high_pass_filter(signal, sr, args.highpass), sr) for path, signal, sr in clips
    ]

    df = compute_summary_rows(filtered_clips, baseline_path)

    summary_csv = args.output / "summary.csv"
    df.to_csv(summary_csv, index=False)
    logger.info("บันทึก %s", summary_csv)

    print_summary_table(df)

    plot_spectrograms(clips, args.output / "01_spectrograms.png", args.highpass)
    plot_psd_comparison(filtered_clips, args.output / "02_psd_comparison.png")
    plot_band_energy(df, args.output / "03_band_energy.png")
    plot_waveform_envelope(filtered_clips, args.output / "04_waveform_envelope.png")
    plot_band_snr(df, args.output / "05_band_snr.png")

    print_decision_summary(df)


if __name__ == "__main__":
    main()
