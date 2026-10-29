"""Does similarity ranking actually surface real grinding? Leave-one-out test.

``tools/detect_similar.py`` produces a listening order. This asks whether
that order is worth anything, using the only metric that matches how the tool
is used: **how many clips must you listen to before hitting a real one.**

Two measurements, because they fail differently:

1. **Leave-one-out over the confirmed reference clips.** Hold out one
   ``sleep_*`` clip, rebuild the grind prototype from the rest, drop the
   held-out clip into the night's candidate pool, and record its rank. This
   asks "if this grind were hiding in the night, how high would it surface?"
   — and because the prototype never sees the held-out clip, it cannot score
   itself.

2. **Rank of the night's own labelled events.** The three events a human
   confirmed as grinding in ``labels.csv`` are already in the pool; where do
   they land? This is the most direct read, but there are only three of them.

Also plots ``sim_grind`` split by the human label, which answers the blunter
question: do grind clips even separate from snore/other at all?

Usage:
    python -m tools.evaluate_similar \
        --candidates results/events/ext_night01/candidates.csv \
        --references data/raw/grind_clips \
        --labels results/events/ext_night01/labels.csv \
        --output results/similar/ext_night01/eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.embed.encoder import YamnetEncoder, l2_normalize
from tools.detect_similar import embed_candidates
from src.embed.prototypes import build_prototypes, score_matrix
from src.embed.references import build_reference_sets, load_labels

logger = logging.getLogger("bruxecho.evaluate_similar")

DEFAULT_CACHE_DIR = Path("results/embed_cache")
RECALL_AT = (10, 20, 50, 100)
TARGET_LABEL = "grind"


def embed_clip_paths(
    encoder: YamnetEncoder,
    paths: list[Path],
    durations: dict[str, float] | None = None,
    crop_to_event: bool = False,
) -> tuple[list[str], np.ndarray]:
    def on_progress(done: int, total: int, _path: Path) -> None:
        if done % 250 == 0 or done == total:
            logger.info("  embedding %d/%d", done, total)

    return embed_candidates(encoder, paths, durations or {}, crop_to_event, on_progress)


def leave_one_out_ranks(
    grind_matrix: np.ndarray,
    other_embeddings: dict[str, np.ndarray],
    other_sources: dict[str, list[str]],
    pool_matrix: np.ndarray,
    grind_ids: list[str],
) -> pd.DataFrame:
    """Rank each held-out grind clip inside the candidate pool.

    The pool is ``pool_matrix`` (the night's clips) plus the single held-out
    clip, so rank 1 means "ranked above all 2,536 candidates".
    """
    rows = []
    for i, held_out_id in enumerate(grind_ids):
        keep = np.delete(np.arange(grind_matrix.shape[0]), i)
        if keep.size == 0:
            continue
        embeddings = dict(other_embeddings)
        embeddings[TARGET_LABEL] = grind_matrix[keep]
        sources = dict(other_sources)
        sources[TARGET_LABEL] = [grind_ids[j] for j in keep]

        prototypes = build_prototypes(embeddings, sources)
        labels, _ = score_matrix(np.zeros((1, grind_matrix.shape[1]), dtype=np.float32), prototypes)
        target_index = labels.index(TARGET_LABEL)

        centroids = np.vstack([p.centroid for p in prototypes]).astype(np.float32)
        held_out_vector = l2_normalize(grind_matrix[i : i + 1])
        pool_similarity = (l2_normalize(pool_matrix) @ centroids.T)[:, target_index]
        held_out_similarity = float((held_out_vector @ centroids.T)[0, target_index])

        # Rank among pool + itself; ties count against it (conservative).
        rank = int(np.sum(pool_similarity >= held_out_similarity)) + 1
        rows.append(
            {
                "held_out": held_out_id,
                "sim_grind": held_out_similarity,
                "rank": rank,
                "pool_size": int(pool_matrix.shape[0]) + 1,
                "percentile": 100.0 * (1.0 - (rank - 1) / (pool_matrix.shape[0] + 1)),
            }
        )
    return pd.DataFrame(rows)


def labeled_event_ranks(
    scored: pd.DataFrame, labels_by_event: dict[str, str], target_label: str
) -> pd.DataFrame:
    """Where the night's own labelled events land in the ranking."""
    column = f"sim_{target_label}"
    ordered = scored.sort_values(column, ascending=False).reset_index(drop=True)
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    ordered["human_label"] = ordered["event_id"].map(labels_by_event)
    return ordered[ordered["human_label"].notna()][
        ["event_id", "human_label", column, "rank"]
    ].sort_values("rank")


def plot_score_distribution(
    scored: pd.DataFrame, labels_by_event: dict[str, str], output_path: Path, target_label: str
) -> None:
    """sim_grind for every clip, with the human-labelled ones marked.

    Plot text is English to match every other chart in this project — the
    default matplotlib font has no Thai glyphs.
    """
    column = f"sim_{target_label}"
    scored = scored.copy()
    scored["human_label"] = scored["event_id"].map(labels_by_event)

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 5))

    ax_hist.hist(scored[column], bins=60, color="steelblue", alpha=0.75, label=f"all clips (n={len(scored)})")
    ax_hist.set_yscale("log")
    colors = {"grind": "crimson", "snore": "darkorange", "other": "gray", "unsure": "purple"}
    for label, color in colors.items():
        subset = scored[scored["human_label"] == label]
        for value in subset[column]:
            ax_hist.axvline(value, color=color, alpha=0.8, linewidth=1.2)
        if len(subset):
            ax_hist.plot([], [], color=color, label=f"{label} (n={len(subset)})")
    ax_hist.set_xlabel(f"{column} (cosine similarity)")
    ax_hist.set_ylabel("clips (log scale)")
    ax_hist.set_title("Similarity to the grind prototype")
    ax_hist.legend(fontsize=8)

    groups = [g for g in ("grind", "snore", "other", "unsure") if (scored["human_label"] == g).any()]
    data = [scored.loc[scored["human_label"] == g, column].to_numpy() for g in groups]
    if data:
        ax_box.boxplot(data)
        ax_box.set_xticks(range(1, len(groups) + 1), groups)
        rng = np.random.default_rng(0)
        for i, values in enumerate(data, start=1):
            ax_box.scatter(
                np.full(values.size, i) + rng.normal(0, 0.05, values.size),
                values,
                s=22,
                alpha=0.8,
                color="crimson",
                zorder=3,
            )
    ax_box.set_ylabel(column)
    ax_box.set_title("By human label — do the classes separate?")
    ax_box.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_report(loo: pd.DataFrame, labeled: pd.DataFrame, pool_size: int) -> None:
    print()
    print("═" * 74)
    print("ประเมินการจัดอันดับ (leave-one-out บนคลิปกัดฟันที่ยืนยันแล้ว)")
    print("═" * 74)

    if loo.empty:
        print("ไม่มีคลิปอ้างอิงพอสำหรับ leave-one-out")
    else:
        print(f"{'คลิปที่กันไว้':<24}{'sim_grind':>12}{'อันดับ':>10}{'เปอร์เซ็นไทล์':>16}")
        print("─" * 74)
        for row in loo.itertuples():
            print(f"{row.held_out:<24}{row.sim_grind:>12.4f}{row.rank:>10d}{row.percentile:>15.2f}%")
        print("─" * 74)
        print(f"อันดับเฉลี่ย: {loo['rank'].mean():.1f}   มัธยฐาน: {loo['rank'].median():.0f}   "
              f"แย่สุด: {loo['rank'].max():.0f}   จากทั้งหมด ~{pool_size}")
        print()
        print("recall@K (กี่คลิปจาก 6 ที่ติดอันดับ K แรก):")
        for k in RECALL_AT:
            hits = int((loo["rank"] <= k).sum())
            print(f"  recall@{k:<4} {hits}/{len(loo)}  ({100.0*hits/len(loo):.0f}%)")

    print()
    print("─" * 74)
    print("อันดับของ event ที่คนติด label ไว้ในคืนนี้:")
    if labeled.empty:
        print("  (ไม่มี)")
    else:
        for label in ("grind", "snore", "other", "unsure"):
            subset = labeled[labeled["human_label"] == label]
            if subset.empty:
                continue
            ranks = subset["rank"].tolist()
            preview = ", ".join(str(r) for r in ranks[:8]) + (" ..." if len(ranks) > 8 else "")
            print(f"  {label:<8} n={len(subset):<3} อันดับมัธยฐาน={int(np.median(ranks)):<6} อันดับ: {preview}")

    print()
    print("═" * 74)
    print("อ่านผลอย่างไร: ถ้าคลิปกัดฟันติด 100 อันดับแรกจาก ~2,500 = ฟังแค่ 4% ของคืนก็เจอ")
    print("ถ้าอันดับกระจายอยู่กลางๆ หรือท้ายๆ แปลว่า embedding ยังแยกเสียงกัดฟันไม่ได้จริง")
    print("⚠ ตัวอย่างยืนยันมีแค่ 6 คลิปจากคนเดียว/คืนเดียว — ตัวเลขนี้บอกแนวโน้ม ไม่ใช่ความแม่นยำที่เชื่อถือได้")


def run(
    candidates_csv: Path,
    references_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    cache_dir: Path,
    clips_dir: Path | None,
    crop_to_event: bool,
) -> None:
    candidates = pd.read_csv(candidates_csv)
    resolved_clips_dir = clips_dir or (candidates_csv.parent / "clips")
    if not resolved_clips_dir.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์คลิป {resolved_clips_dir}")

    encoder = YamnetEncoder(cache_dir=cache_dir)

    sets, warnings = build_reference_sets(references_dir, labels_csv, candidates_csv, resolved_clips_dir)
    for warning in warnings:
        logger.warning("อ้างอิง: %s", warning)
    if TARGET_LABEL not in sets:
        raise SystemExit(f"ไม่มีคลิปอ้างอิงของ {TARGET_LABEL!r}")

    logger.info("กำลัง encode คลิปอ้างอิง ...")
    reference_embeddings: dict[str, np.ndarray] = {}
    reference_sources: dict[str, list[str]] = {}
    for label, paths in sorted(sets.items()):
        ids, matrix = encoder.embed_batch(paths)
        reference_embeddings[label] = matrix
        reference_sources[label] = ids
        logger.info("  %s: %d คลิป", label, matrix.shape[0])

    clip_paths = [
        resolved_clips_dir / str(name)
        for name in candidates["clip_file"].fillna("")
        if str(name).strip()
    ]
    clip_paths = [p for p in clip_paths if p.exists()]
    durations = {
        Path(str(row.clip_file)).stem: float(row.duration_s)
        for row in candidates.itertuples()
        if isinstance(row.clip_file, str) and row.clip_file.strip()
    }
    logger.info(
        "กำลัง encode คลิป candidate %d ไฟล์ (crop_to_event=%s) ...", len(clip_paths), crop_to_event
    )
    pool_ids, pool_matrix = embed_clip_paths(encoder, clip_paths, durations, crop_to_event)
    if pool_matrix.shape[0] == 0:
        raise SystemExit("encode คลิป candidate ไม่สำเร็จเลย")

    grind_matrix = reference_embeddings.pop(TARGET_LABEL)
    grind_ids = reference_sources.pop(TARGET_LABEL)
    loo = leave_one_out_ranks(
        grind_matrix, reference_embeddings, reference_sources, pool_matrix, grind_ids
    )

    # Full-prototype scoring of the pool, for the label-distribution view.
    reference_embeddings[TARGET_LABEL] = grind_matrix
    reference_sources[TARGET_LABEL] = grind_ids
    prototypes = build_prototypes(reference_embeddings, reference_sources)
    labels, similarities = score_matrix(pool_matrix, prototypes)

    stem_to_event = {
        Path(str(row.clip_file)).stem: row.event_id
        for row in candidates.itertuples()
        if isinstance(row.clip_file, str) and row.clip_file.strip()
    }
    scored = pd.DataFrame({"event_id": [stem_to_event.get(i, i) for i in pool_ids]})
    for j, label in enumerate(labels):
        scored[f"sim_{label}"] = similarities[:, j]

    labels_by_event = load_labels(labels_csv)
    labeled = labeled_event_ranks(scored, labels_by_event, TARGET_LABEL)

    output_dir.mkdir(parents=True, exist_ok=True)
    loo.to_csv(output_dir / "leave_one_out.csv", index=False)
    labeled.to_csv(output_dir / "labeled_event_ranks.csv", index=False)
    plot_path = output_dir / "score_distribution.png"
    plot_score_distribution(scored, labels_by_event, plot_path, TARGET_LABEL)
    logger.info("บันทึก %s", output_dir / "leave_one_out.csv")
    logger.info("บันทึก %s", plot_path)

    print_report(loo, labeled, int(pool_matrix.shape[0]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ประเมินว่าการจัดอันดับด้วยความคล้ายใช้ได้จริงไหม (leave-one-out)")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clips-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--crop-to-event",
        action="store_true",
        help="ตัดคลิป candidate ให้เหลือเฉพาะช่วง event ก่อน encode (ให้ตรงกับ tools/detect_similar.py)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    for path in (args.candidates, args.references, args.labels):
        if not path.exists():
            raise SystemExit(f"ไม่พบ {path}")

    run(
        args.candidates,
        args.references,
        args.labels,
        args.output,
        args.cache_dir,
        args.clips_dir,
        args.crop_to_event,
    )


if __name__ == "__main__":
    main()
