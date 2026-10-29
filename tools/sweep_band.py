"""Compare several gate bands × k values over one session, numbers only.

Sprint 1 fixed the gate band at 2000-16000 Hz from tooth-*rubbing* clips
recorded ~30 cm from the mic. Clips of real sleep bruxism have almost no
energy up there (0.1-2.4% above 8 kHz vs. 32-47%), so that band may be
listening where the signal isn't — which would explain a night returning
2,536 mostly-noise events. This tool answers "what would each band have
returned?" without committing to one.

Deliberately does **not** cut clips or draw per-run overview charts: it
runs the same :class:`~src.detect.energy_gate.EnergyGate` that
``tools/detect_events.py`` runs, but keeps only the counts, so a 15-config
sweep over an 8-hour night stays a coffee break rather than an afternoon.
Once a band looks promising, re-run ``tools/detect_events.py`` with
``--low-hz/--high-hz`` (and a fresh ``--output``) to get listenable clips.

Usage:
    python -m tools.sweep_band --session data/raw/sleep/ext_night01
    python -m tools.sweep_band --session <path> --max-hours 0.5   # quick smoke test
    python -m tools.sweep_band --session <path> --bands 1000-6000 800-4000 --k 6 12
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.detect.energy_gate import DEFAULT_MIN_EVENT_DURATION_S, EnergyGate, resolve_band
from src.session.schema import SessionInfo, read_session_json

logger = logging.getLogger("bruxecho.sweep_band")

READ_BLOCK_SECONDS = 2.0

# (low_hz, high_hz) — the current default first so every table has its
# baseline on the top row, then progressively lower/narrower bands aimed at
# where the real sleep-bruxism clips actually put their energy (~3 kHz
# centroid).
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (2000.0, 16000.0),
    (1000.0, 6000.0),
    (1500.0, 8000.0),
    (800.0, 4000.0),
    (2000.0, 8000.0),
)
DEFAULT_K_VALUES: tuple[float, ...] = (6.0, 8.0, 12.0)


@dataclass
class SweepRow:
    band_low_hz: float
    band_high_hz: float
    band_mid_hz: float
    k: float
    n_events: int
    events_per_hour: float
    total_event_s: float
    gated_out_pct: float
    median_event_s: float

    @property
    def band_label(self) -> str:
        return f"{self.band_low_hz:.0f}-{self.band_high_hz:.0f}"


def parse_band(text: str) -> tuple[float, float]:
    """Parse a ``LOW-HIGH`` band string (Hz), e.g. ``"1500-8000"``."""
    parts = text.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"รูปแบบย่านความถี่ไม่ถูกต้อง: {text!r} (ต้องเป็น LOW-HIGH เช่น 1500-8000)")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"รูปแบบย่านความถี่ไม่ถูกต้อง: {text!r} (ต้องเป็นตัวเลข)") from None
    if high <= low:
        raise argparse.ArgumentTypeError(f"ย่านความถี่ไม่ถูกต้อง: {text!r} (ขอบบนต้องมากกว่าขอบล่าง)")
    return low, high


def resolve_sample_rate(session: SessionInfo, session_dir: Path) -> int:
    if session.sample_rate_actual:
        return session.sample_rate_actual
    if not session.segments:
        raise SystemExit("session.json ไม่มี sample_rate_actual และไม่มี segment ให้ probe แทน")
    return sf.info(str(session_dir / session.segments[0].file)).samplerate


def run_band(
    session: SessionInfo,
    session_dir: Path,
    sample_rate: int,
    band: tuple[float, float],
    k_values: list[float],
    min_duration_s: float,
    max_seconds: float | None,
) -> tuple[list[SweepRow], float]:
    """Stream the session once, feeding every k value's gate from the same blocks.

    Only ``k`` varies within a band, and ``k`` affects nothing before the
    threshold comparison — but the gates still each do their own filtering,
    so this shares the file reads (a few % of runtime) and, more usefully,
    guarantees every k in a band saw byte-identical input.
    """
    low_hz, high_hz = band
    gates = [
        EnergyGate(
            sample_rate=sample_rate,
            session_id=session.session_id,
            k=k,
            min_duration_s=min_duration_s,
            band_low_hz=low_hz,
            band_high_hz=high_hz,
        )
        for k in k_values
    ]

    block_frames = int(READ_BLOCK_SECONDS * sample_rate)
    processed_s = 0.0
    for seg in session.segments:
        seg_path = session_dir / seg.file
        if not seg_path.exists():
            logger.warning("ข้าม segment ที่หายไป: %s", seg_path)
            continue
        for block in sf.blocks(str(seg_path), blocksize=block_frames, dtype="float32", always_2d=False):
            block = np.asarray(block, dtype=np.float32)
            if max_seconds is not None and processed_s + block.size / sample_rate > max_seconds:
                keep = int(round((max_seconds - processed_s) * sample_rate))
                block = block[:keep]
            if block.size:
                for gate in gates:
                    gate.process_block(block, segment_file=seg.file)
                processed_s += block.size / sample_rate
            if max_seconds is not None and processed_s >= max_seconds:
                break
        if max_seconds is not None and processed_s >= max_seconds:
            break

    rows: list[SweepRow] = []
    hours = processed_s / 3600.0
    for gate, k in zip(gates, k_values):
        events = gate.finalize()
        durations = [e.duration_s for e in events]
        total_event_s = sum(durations)
        rows.append(
            SweepRow(
                band_low_hz=gate.band.low_hz,
                band_high_hz=gate.band.high_hz,
                band_mid_hz=gate.band.mid_hz,
                k=k,
                n_events=len(events),
                events_per_hour=len(events) / hours if hours > 0 else 0.0,
                total_event_s=total_event_s,
                gated_out_pct=(1.0 - total_event_s / processed_s) * 100.0 if processed_s > 0 else 100.0,
                median_event_s=float(np.median(durations)) if durations else 0.0,
            )
        )
    return rows, processed_s


def plot_heatmap(df: pd.DataFrame, output_path: Path, session_id: str, hours: float) -> None:
    """Event count as a band × k grid.

    Plot text is kept in English (unlike this tool's console output) to match
    every other chart in this project — the default matplotlib font has no
    Thai glyphs.
    """
    # Order rows by the band's actual edges, not the label's string order —
    # otherwise "800-4000" sorts after "2000-16000".
    band_order = (
        df[["band", "band_low_hz", "band_high_hz"]]
        .drop_duplicates()
        .sort_values(["band_low_hz", "band_high_hz"])["band"]
        .tolist()
    )
    pivot = df.pivot(index="band", columns="k", values="n_events").reindex(band_order)
    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 4, 0.8 * len(pivot.index) + 3))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)), [f"k={c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{b} Hz" for b in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.to_numpy()[i, j]
            # Dark text on light cells, white on dark ones, so the count
            # stays readable at both ends of the colour scale.
            normalized = (value - np.nanmin(pivot.to_numpy())) / max(
                np.nanmax(pivot.to_numpy()) - np.nanmin(pivot.to_numpy()), 1e-9
            )
            ax.text(
                j,
                i,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if normalized > 0.6 else "black",
            )

    ax.set_xlabel("threshold multiplier k")
    ax.set_ylabel("gate band")
    ax.set_title(f"Event count by band × k — {session_id} ({hours:.1f}h)")
    fig.colorbar(im, ax=ax, label="events")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_table(rows: list[SweepRow], hours: float, baseline_label: str) -> None:
    print()
    print("═" * 78)
    print(f"ผลการกวาดย่านความถี่ × k — ประมวลผล {hours:.2f} ชั่วโมง")
    print("═" * 78)
    header = f"{'band (Hz)':>14} {'k':>4} {'event':>7} {'event/ชม.':>11} {'คัดออก %':>10} {'median (s)':>11}"
    print(header)
    print("─" * 78)
    for row in sorted(rows, key=lambda r: (r.band_low_hz, r.band_high_hz, r.k)):
        marker = " ←baseline" if row.band_label == baseline_label else ""
        print(
            f"{row.band_label:>14} {row.k:>4g} {row.n_events:>7d} "
            f"{row.events_per_hour:>11.1f} {row.gated_out_pct:>10.2f} {row.median_event_s:>11.2f}{marker}"
        )
    print("═" * 78)
    print("หมายเหตุ: จำนวน event ที่ 'ดูดี' ไม่ได้แปลว่าย่านนั้นถูกต้อง — ต้องเอาไปตัดคลิปแล้วฟังจริงด้วย")
    print("ขั้นถัดไป: python -m tools.detect_events --session <path> --low-hz L --high-hz H --output results/events_LxH")


def run(
    session_dir: Path,
    output_root: Path,
    bands: list[tuple[float, float]],
    k_values: list[float],
    min_duration_s: float,
    max_hours: float | None,
) -> None:
    session = read_session_json(session_dir)
    if not session.segments:
        raise SystemExit(f"session.json ที่ {session_dir} ไม่มี segment ให้ประมวลผล")

    sample_rate = resolve_sample_rate(session, session_dir)
    max_seconds = max_hours * 3600.0 if max_hours else None

    # Validate every band up front so an unusable one fails in a second
    # rather than after ten minutes of streaming.
    for low_hz, high_hz in bands:
        band = resolve_band(sample_rate, low_hz, high_hz)
        if band.high_hz_was_capped:
            logger.warning(
                "ย่าน %.0f-%.0f Hz: ขอบบนถูกลดเหลือ %.0f Hz ตาม sample rate %d Hz",
                low_hz,
                high_hz,
                band.high_hz,
                sample_rate,
            )

    logger.info(
        "session=%s sample_rate=%d Hz — %d ย่าน × %d ค่า k = %d ชุด",
        session.session_id,
        sample_rate,
        len(bands),
        len(k_values),
        len(bands) * len(k_values),
    )

    rows: list[SweepRow] = []
    processed_s = 0.0
    for i, band in enumerate(bands, start=1):
        started = time.time()
        band_rows, processed_s = run_band(
            session, session_dir, sample_rate, band, k_values, min_duration_s, max_seconds
        )
        rows.extend(band_rows)
        logger.info(
            "[%d/%d] ย่าน %.0f-%.0f Hz เสร็จใน %.0f วินาที — event: %s",
            i,
            len(bands),
            band[0],
            band[1],
            time.time() - started,
            ", ".join(f"k={r.k:g}:{r.n_events}" for r in band_rows),
        )

    if not rows:
        raise SystemExit("ไม่มีชุดค่าใดรันสำเร็จเลย")

    df = pd.DataFrame([{**vars(r), "band": r.band_label} for r in rows])
    hours = processed_s / 3600.0

    session_output_dir = output_root / session.session_id
    session_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    heatmap_path = session_output_dir / "heatmap_events.png"
    plot_heatmap(df, heatmap_path, session.session_id, hours)

    logger.info("บันทึก %s", csv_path)
    logger.info("บันทึก %s", heatmap_path)

    baseline_label = f"{DEFAULT_BANDS[0][0]:.0f}-{DEFAULT_BANDS[0][1]:.0f}"
    print_table(rows, hours, baseline_label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="กวาดหลายย่านความถี่ × หลายค่า k บน session เดียว แล้วสรุปเป็นตาราง+heatmap (ไม่ตัดคลิป)"
    )
    parser.add_argument("--session", required=True, type=Path, help="โฟลเดอร์ session (มี session.json อยู่ข้างใน)")
    parser.add_argument(
        "--bands",
        nargs="+",
        type=parse_band,
        default=list(DEFAULT_BANDS),
        metavar="LOW-HIGH",
        help="ย่านความถี่ที่จะทดสอบ เช่น 1000-6000 (ค่าเริ่มต้น: "
        + " ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in DEFAULT_BANDS)
        + ")",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=float,
        default=list(DEFAULT_K_VALUES),
        dest="k_values",
        help="ค่า k ที่จะทดสอบ (ค่าเริ่มต้น: " + " ".join(f"{k:g}" for k in DEFAULT_K_VALUES) + ")",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_EVENT_DURATION_S,
        help=f"ความยาวขั้นต่ำของ event เป็นวินาที (ค่าเริ่มต้น {DEFAULT_MIN_EVENT_DURATION_S})",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="ประมวลผลแค่กี่ชั่วโมงแรกของคืน (ไม่ใส่ = ทั้งคืน) — ใช้ลองให้เร็วก่อนรันจริง",
    )
    parser.add_argument("--output", type=Path, default=Path("results/sweep"), help="โฟลเดอร์ปลายทางหลัก")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    if not (args.session / "session.json").exists():
        raise SystemExit(f"ไม่พบ session.json ใน {args.session}")

    run(args.session, args.output, args.bands, args.k_values, args.min_duration, args.max_hours)


if __name__ == "__main__":
    main()
