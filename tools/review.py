"""Streamlit clip-review / labeling tool for BruxEcho event candidates.

The base of the labeling tool a dentist will use in a later sprint. Reads
whatever ``tools/detect_events.py`` already produced under
``results/events/<session_id>/`` (candidates.csv + clips/) — never touches
raw session audio directly, and never needs the original session directory
(provenance is denormalized into candidates.csv for exactly this reason).

Usage:
    streamlit run tools/review.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.audio.io import load_audio
from src.features.spectral import log_mel_spectrogram

EVENTS_ROOT = Path("results/events")
LABELS: list[str] = ["กัดฟัน", "กรน", "อื่นๆ", "ไม่แน่ใจ"]
GAIN_OPTIONS: dict[str, float] = {"1x": 1.0, "5x": 5.0, "20x": 20.0}
YAMNET_LIMIT_HZ = 8000.0


def discover_sessions() -> list[Path]:
    if not EVENTS_ROOT.exists():
        return []
    return sorted((p.parent for p in EVENTS_ROOT.glob("*/candidates.csv")), key=lambda p: p.name)


def load_candidates(session_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(session_dir / "candidates.csv")
    return df.sort_values("peak_rms_db", ascending=False).reset_index(drop=True)


def labels_path(session_dir: Path) -> Path:
    return session_dir / "labels.csv"


def load_labels(session_dir: Path) -> pd.DataFrame:
    path = labels_path(session_dir)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["event_id", "label", "labeled_at"])


def save_label(session_dir: Path, event_id: str, label: str) -> None:
    df = load_labels(session_dir)
    now = datetime.now().isoformat(timespec="seconds")
    if (df["event_id"] == event_id).any():
        df.loc[df["event_id"] == event_id, ["label", "labeled_at"]] = [label, now]
    else:
        new_row = pd.DataFrame([{"event_id": event_id, "label": label, "labeled_at": now}])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(labels_path(session_dir), index=False)


def render_spectrogram(audio: np.ndarray, sr: int) -> None:
    log_mel_db, times, mel_freqs = log_mel_spectrogram(audio, sr)
    fig, ax = plt.subplots(figsize=(8, 3))
    mesh = ax.pcolormesh(times, mel_freqs, log_mel_db, shading="auto", cmap="magma")
    if sr / 2.0 > YAMNET_LIMIT_HZ:
        ax.axhline(YAMNET_LIMIT_HZ, color="cyan", linestyle="--", linewidth=1.0)
        ax.text(times[-1] if times.size else 0.0, YAMNET_LIMIT_HZ, " 8 kHz", color="cyan", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.colorbar(mesh, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="BruxEcho review", layout="wide")
    st.title("BruxEcho — ตรวจทานคลิปเสียง")

    sessions = discover_sessions()
    if not sessions:
        st.warning("ไม่พบ session ที่ผ่าน tools/detect_events.py แล้ว (ไม่มี results/events/*/candidates.csv)")
        return

    session_options: dict[str, Path] = {}
    for s in sessions:
        preview = pd.read_csv(s / "candidates.csv", nrows=1)
        provenance = preview["provenance"].iloc[0] if not preview.empty and "provenance" in preview else "?"
        session_options[f"{s.name} ({provenance})"] = s

    selected_key = st.sidebar.selectbox("Session", list(session_options.keys()))
    session_dir = session_options[selected_key]

    if st.session_state.get("_session_dir") != str(session_dir):
        st.session_state["_session_dir"] = str(session_dir)
        st.session_state["clip_index"] = 0

    candidates = load_candidates(session_dir)
    if candidates.empty:
        st.warning("session นี้ไม่มี event เลย")
        return

    labels_df = load_labels(session_dir)

    st.sidebar.markdown("### สรุปการติด label")
    if not labels_df.empty:
        counts = labels_df["label"].value_counts()
        for label in LABELS:
            st.sidebar.write(f"{label}: {int(counts.get(label, 0))}")
        st.sidebar.write(f"รวม: {len(labels_df)} / {len(candidates)}")
    else:
        st.sidebar.write("ยังไม่มีการติด label")

    n = len(candidates)
    idx = max(0, min(st.session_state.get("clip_index", 0), n - 1))
    st.session_state["clip_index"] = idx
    row = candidates.iloc[idx]

    st.subheader(f"คลิป {idx + 1} / {n}")
    st.caption(
        f"เวลาที่เกิด: {row['wall_clock_time']} · peak {row['peak_rms_db']:.1f} dB · "
        f"duration {row['duration_s']:.2f}s · segment {row['segment_file']}"
    )

    clip_file = row.get("clip_file")
    if not isinstance(clip_file, str) or not clip_file:
        st.error("ไม่มีไฟล์คลิปสำหรับ event นี้ (ตัดคลิปไม่สำเร็จตอนรัน detect_events.py)")
    else:
        clip_path = session_dir / "clips" / clip_file
        audio, sr = load_audio(clip_path)
        render_spectrogram(audio, sr)

        gain_label = st.radio("ขยายเสียง (เสียงจริงเบามาก)", list(GAIN_OPTIONS.keys()), horizontal=True, index=1)
        amplified = np.clip(audio * GAIN_OPTIONS[gain_label], -1.0, 1.0)
        st.audio(amplified, sample_rate=sr)

    current_label = None
    if not labels_df.empty and (labels_df["event_id"] == row["event_id"]).any():
        current_label = labels_df.loc[labels_df["event_id"] == row["event_id"], "label"].iloc[0]
    if current_label:
        st.info(f"label ปัจจุบัน: {current_label}")

    st.write("ติด label:")
    label_cols = st.columns(len(LABELS))
    for col, label in zip(label_cols, LABELS):
        if col.button(label, use_container_width=True, type="primary" if label == current_label else "secondary"):
            save_label(session_dir, row["event_id"], label)
            if idx < n - 1:
                st.session_state["clip_index"] = idx + 1
            st.rerun()

    nav_cols = st.columns(2)
    if nav_cols[0].button("← ก่อนหน้า", disabled=idx == 0, use_container_width=True):
        st.session_state["clip_index"] = idx - 1
        st.rerun()
    if nav_cols[1].button("ถัดไป →", disabled=idx == n - 1, use_container_width=True):
        st.session_state["clip_index"] = idx + 1
        st.rerun()


if __name__ == "__main__":
    main()
