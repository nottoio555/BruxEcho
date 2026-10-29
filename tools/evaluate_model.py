"""Per-class metrics, confusion matrix, and a threshold recommendation.

The threshold matters more than the headline F1 here. The app shows the user
a list of clips to listen to; a false positive costs them a wasted listen and
some trust, while a missed event is invisible. So this reports the
precision-recall curve for bruxism and recommends the lowest threshold that
still clears a precision target — "how confident must the model be before we
show it" — rather than the F1-optimal point.

Usage:
    python -m tools.evaluate_model
    python -m tools.evaluate_model --target-precision 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from src.train.dataset import CLASS_BRUXISM, CLASS_NAMES
from src.train.train_classifier import load_artifacts, predict_proba

logger = logging.getLogger("bruxecho.evaluate_model")

DEFAULT_CONFIG = Path("configs/paths.yaml")
DEFAULT_TARGET_PRECISION = 0.80


def recommend_threshold(
    y_true_binary: np.ndarray, scores: np.ndarray, target_precision: float
) -> dict[str, float]:
    """Lowest threshold whose precision still meets the target.

    Lowest rather than highest: among thresholds that clear the precision
    bar, the loosest one keeps the most true events, so the user sees the
    fullest list that is still trustworthy.
    """
    precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
    # precision/recall have one more entry than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= target_precision
    if not np.any(ok):
        best = int(np.argmax(precision))
        return {
            "threshold": float(thresholds[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "met_target": 0.0,
        }
    idx = int(np.argmax(ok))  # thresholds ascend, so the first hit is the loosest
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "met_target": 1.0,
    }


def event_level_metrics(
    y_true: np.ndarray,
    bruxism_scores: np.ndarray,
    segment_index: np.ndarray,
    bruxism_index: int,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    """Per-annotated-event precision/recall, which is what the app reports.

    Frame-level numbers understate this model badly. A bruxism event covers
    several 0.96s frames and the pipeline only needs *one* of them to cross
    the threshold for the event to reach the user's list — so frame recall of
    4% can still be event recall of 40%. Conversely every non-bruxism segment
    with one over-confident frame becomes a false clip the user must listen
    to, which frame precision spreads out and hides.

    An event counts as detected when its loudest frame clears the threshold.
    """
    order = np.argsort(segment_index, kind="stable")
    sorted_segments = segment_index[order]
    sorted_scores = bruxism_scores[order]
    sorted_truth = y_true[order]

    boundaries = np.flatnonzero(np.diff(sorted_segments)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_segments.size]))

    peak_score = np.array([sorted_scores[s:e].max() for s, e in zip(starts, ends)])
    # Every frame of a segment carries that segment's label, so the first is enough.
    is_bruxism = np.array([sorted_truth[s] == bruxism_index for s, e in zip(starts, ends)])

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        fired = peak_score >= threshold
        true_positive = int(np.sum(fired & is_bruxism))
        false_positive = int(np.sum(fired & ~is_bruxism))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(int(np.sum(is_bruxism)), 1)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "events_shown": int(true_positive + false_positive),
                "true_positive": true_positive,
                "n_bruxism_events": int(np.sum(is_bruxism)),
            }
        )
    return rows


def choose_event_threshold(
    sweep: list[dict[str, float]], target_precision: float
) -> dict[str, float]:
    """The threshold the app will actually ship with.

    First choice is the loosest threshold that meets the precision target,
    which keeps the most events while staying trustworthy.

    The fallback matters more here, because on this data nothing reaches 0.80.
    Taking "the highest precision available" is a trap: precision keeps
    creeping up as the threshold approaches 1 while recall collapses, so the
    apparent best point (0.995, precision 0.73) shows 45 events out of 555 —
    the user's nightly count would be roughly a tenth of what happened, and a
    count that wrong is worse than a noisy one for an app whose whole output
    is "how much did you grind last night".

    So the fallback picks the loosest threshold where precision has caught up
    with recall. Since events_shown / events_that_happened == recall /
    precision, that is exactly the tightest-but-not-yet-under-reporting point:
    the count on screen is approximately right, erring slightly low.
    """
    qualifying = [row for row in sweep if row["precision"] >= target_precision]
    if qualifying:
        choice = dict(min(qualifying, key=lambda row: row["threshold"]))
        choice["met_target"] = 1.0
    else:
        ordered = sorted(sweep, key=lambda row: row["threshold"])
        balanced = next(
            (row for row in ordered if row["precision"] >= row["recall"]),
            max(ordered, key=lambda row: row["precision"]),
        )
        choice = dict(balanced)
        choice["met_target"] = 0.0
    choice["target_precision"] = target_precision
    return choice


def plot_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bruxism_scores: np.ndarray,
    y_true_binary: np.ndarray,
    recommendation: dict[str, float],
    output_path: Path,
) -> None:
    """Confusion matrix + bruxism PR curve.

    Plot text is English to match every other chart in this project — the
    default matplotlib font has no Thai glyphs.
    """
    fig, (ax_cm, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    im = ax_cm.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax_cm.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_title("Confusion matrix (row-normalized)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax_cm.text(
                j,
                i,
                f"{normalized[i, j]:.2f}\n({matrix[i, j]})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if normalized[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046)

    precision, recall, _ = precision_recall_curve(y_true_binary, bruxism_scores)
    average_precision = average_precision_score(y_true_binary, bruxism_scores)
    ax_pr.plot(recall, precision, color="crimson", label=f"bruxism (AP={average_precision:.3f})")
    ax_pr.axhline(
        recommendation["target_precision"],
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"target precision {recommendation['target_precision']:.2f}",
    )
    ax_pr.scatter(
        [recommendation["recall"]],
        [recommendation["precision"]],
        color="black",
        zorder=5,
        label=(
            f"recommended thr={recommendation['threshold']:.3f}\n"
            f"P={recommendation['precision']:.3f} R={recommendation['recall']:.3f}"
        ),
    )
    baseline = float(np.mean(y_true_binary))
    ax_pr.axhline(baseline, color="steelblue", linestyle=":", linewidth=1, label=f"chance ({baseline:.3f})")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Bruxism precision-recall (frame level)")
    ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(fontsize=8, loc="best")
    ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recommendation: dict[str, float],
    average_precision: float,
    metadata: dict,
    output_dir: Path,
    sweep: list[dict[str, float]] | None = None,
    event_choice: dict[str, float] | None = None,
) -> None:
    print()
    print("═" * 74)
    print("ผลประเมินโมเดล (ระดับเฟรม 0.96 วินาที)")
    print("═" * 74)
    print(f"แบ่ง train/val ตาม: {metadata.get('group_by', '?')}")
    print(f"val_groups: {', '.join(metadata.get('val_groups', [])[:12])}")
    print()

    report = classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    print(report)

    print("─" * 74)
    print(f"bruxism average precision (AP): {average_precision:.3f}")
    baseline = float(np.mean(y_true == CLASS_NAMES.index(CLASS_BRUXISM)))
    print(f"สัดส่วน bruxism ใน val (ระดับสุ่ม): {baseline:.3f}")
    print()
    print("═" * 74)
    print(f"threshold ที่แนะนำ: {recommendation['threshold']:.3f}")
    print(f"  ที่จุดนี้ precision={recommendation['precision']:.3f}  recall={recommendation['recall']:.3f}")
    if not recommendation["met_target"]:
        print(f"  ⚠ ไม่มี threshold ไหนทำ precision ถึงเป้า {recommendation['target_precision']:.2f} ได้")
        print("    ค่าที่ให้คือจุดที่ precision สูงสุดเท่าที่ทำได้")
    if sweep and event_choice:
        print()
        print("─" * 74)
        print("ระดับ event (หน่วยที่ผู้ใช้เห็นจริง — event ถูกจับได้ถ้ามีเฟรมไหนผ่าน threshold):")
        print(f"{'threshold':>10}{'precision':>12}{'recall':>10}{'event ที่โชว์':>16}")
        for row in sweep:
            # Below 0.3 nothing useful happens; above it, show the coarse grid
            # plus the fine steps near the chosen point.
            chosen = abs(row["threshold"] - event_choice["threshold"]) < 1e-9
            fine = row["threshold"] >= 0.95 and round(row["threshold"] * 200) % 2 == 0
            if not chosen and (row["threshold"] < 0.3 or (row["threshold"] >= 0.95 and not fine)):
                continue
            mark = "  ←เลือก" if chosen else ""
            print(
                f"{row['threshold']:>10.3f}{row['precision']:>12.3f}{row['recall']:>10.3f}"
                f"{row['events_shown']:>16d}{mark}"
            )
        print()
        print(f"► threshold ที่จะใช้จริง: {event_choice['threshold']:.3f}")
        print(
            f"  จับ bruxism ได้ {event_choice['true_positive']}/{event_choice['n_bruxism_events']} event "
            f"(recall {event_choice['recall']:.1%}) จากที่โชว์ทั้งหมด {event_choice['events_shown']} event "
            f"(precision {event_choice['precision']:.1%})"
        )
        if not event_choice.get("met_target"):
            ratio = event_choice["events_shown"] / max(event_choice["n_bruxism_events"], 1)
            print(f"  ⚠ ไม่มี threshold ไหนทำ precision ถึงเป้า {event_choice['target_precision']:.2f} ได้")
            print("    จึงเลือกจุดที่นับเหตุการณ์ได้ใกล้เคียงจริงที่สุด (precision ≥ recall)")
            print(f"    จำนวนที่โชว์ = {ratio:.2f}x ของจำนวนจริง")
            print("    (ถ้าเลือก precision สูงสุดแทน จะได้ precision 0.73 แต่ recall เหลือ 6% — นับผิดไป 10 เท่า)")

    print()
    print("เลือก threshold จาก precision ไม่ใช่ F1 เพราะผู้ใช้ต้องมานั่งฟังคลิปเอง")
    print("— ผลบวกลวงเสียเวลาและความเชื่อถือ ส่วนของที่พลาดไปผู้ใช้ไม่เห็นอยู่แล้ว")
    print("═" * 74)
    print(f"บันทึกกราฟและตัวเลขไว้ที่ {output_dir}")


def run(config_path: Path, output_dir: Path, target_precision: float) -> None:
    paths = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(paths["artifacts"]["model_dir"])
    split_path = model_dir / "val_split.npz"
    if not split_path.exists():
        raise SystemExit(f"ไม่พบ {split_path} — รัน python -m tools.train_model ก่อน")

    model, mean, std, metadata = load_artifacts(model_dir)
    with np.load(split_path, allow_pickle=False) as data:
        x_val, y_val = data["x_val"], data["y_val"]
        segment_index = data["segment_index_val"]

    logger.info("กำลังทำนาย %d frame ...", len(y_val))
    probabilities = predict_proba(model, mean, std, x_val)
    y_pred = probabilities.argmax(axis=1)

    bruxism_index = CLASS_NAMES.index(CLASS_BRUXISM)
    bruxism_scores = probabilities[:, bruxism_index]
    y_true_binary = (y_val == bruxism_index).astype(int)

    recommendation = recommend_threshold(y_true_binary, bruxism_scores, target_precision)
    recommendation["target_precision"] = target_precision
    average_precision = float(average_precision_score(y_true_binary, bruxism_scores))

    # The deployment threshold is chosen on event-level precision, because an
    # "event" is the unit the user is shown and asked to listen to.
    sweep = event_level_metrics(
        y_val,
        bruxism_scores,
        segment_index,
        bruxism_index,
        # Coarse across the range, then fine near the top where precision
        # is still climbing steeply and the deployment point actually sits.
        np.round(
            np.concatenate([np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.9999, 0.005)]),
            4,
        ),
    )
    event_choice = choose_event_threshold(sweep, target_precision)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_evaluation(y_val, y_pred, bruxism_scores, y_true_binary, recommendation, output_dir / "evaluation.png")

    report_dict = classification_report(
        y_val,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "classification_report": report_dict,
        "confusion_matrix": confusion_matrix(y_val, y_pred, labels=range(len(CLASS_NAMES))).tolist(),
        "bruxism_average_precision": average_precision,
        "recommended_threshold_frame_level": recommendation,
        "recommended_threshold_event_level": event_choice,
        "event_level_sweep": sweep,
        "group_by": metadata.get("group_by"),
        "val_groups": metadata.get("val_groups"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # The pipeline reads its default threshold from here, so evaluating and
    # then deploying can't drift apart.
    metadata["recommended_threshold"] = event_choice["threshold"]
    metadata["threshold_precision"] = event_choice["precision"]
    metadata["threshold_recall"] = event_choice["recall"]
    metadata["threshold_basis"] = "event-level precision on the held-out sessions"
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_report(y_val, y_pred, recommendation, average_precision, metadata, output_dir, sweep, event_choice)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ประเมินโมเดลและแนะนำ threshold ที่ precision สูง")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("results/model_eval"))
    parser.add_argument(
        "--target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        help=f"precision ขั้นต่ำที่ยอมรับได้ (ค่าเริ่มต้น {DEFAULT_TARGET_PRECISION})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.config, args.output, args.target_precision)


if __name__ == "__main__":
    main()
