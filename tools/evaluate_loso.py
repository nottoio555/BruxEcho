"""Leave-one-subject-out evaluation — the number to quote for "a new person".

The shipped model was validated with a split by *session*. Every subject in
DreamCatcher has several sessions, so that split puts the same four people on
both sides: the validation score partly measures recognising a voice, a room
and a microphone the model has already trained on. That is exactly the
question a deployed app never gets asked, since every user is a stranger.

Here each subject is held out in turn. The head is trained on the other
subjects only, early-stopped on sessions carved from those same training
subjects (never on the held-out one, which would leak it back in through the
stopping rule), and then scored on the person it has never seen.

Alongside the trained head, two baselines run on identical folds so the
numbers have something to be compared against:

- **majority class** — always answers "other"; the floor any model must beat
- **logistic regression** on the same YAMNet embeddings — tells whether the
  two hidden layers earn their keep over a linear read-out

Nothing here touches ``models/bruxism_classifier/``: every fold's model lives
only in memory and is discarded. Embeddings come from the existing cache, so
no audio is re-encoded.

Usage:
    python -m tools.evaluate_loso
    python -m tools.evaluate_loso --threshold 0.96 --output results/model_eval_loso
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.train.dataset import CLASS_BRUXISM, CLASS_NAMES, grouped_split, load_segments
from src.train.train_classifier import (
    cache_path_for,
    extract_frame_dataset,
    predict_proba,
    standardize,
    train_head,
)
from tools.evaluate_model import event_level_metrics

logger = logging.getLogger("bruxecho.evaluate_loso")

DEFAULT_CONFIG = Path("configs/paths.yaml")
DEFAULT_OUTPUT = Path("results/model_eval_loso")
DEFAULT_THRESHOLD = 0.96
INNER_VAL_FRACTION = 0.15
BRUXISM = CLASS_NAMES.index(CLASS_BRUXISM)


def relabel_groups(dataset, segments) -> tuple[np.ndarray, np.ndarray]:
    """Recompute session/subject per frame from the current filename parser.

    The cache stores the groups it was built with, which predate the fix for
    ``04_new-*`` filenames. ``segment_index`` points back into the segment
    list, so the correct groups can be rebuilt without re-encoding any audio.
    """
    index = dataset.segment_index
    sessions = np.array([segments[i].session for i in index])
    subjects = np.array([segments[i].subject for i in index])
    return sessions, subjects


def scores_for(name: str, y_true: np.ndarray, probs: np.ndarray, segment_index: np.ndarray,
               threshold: float) -> dict:
    """Every metric reported per fold, for one model on one held-out subject."""
    y_pred = probs.argmax(axis=1)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    binary = (y_true == BRUXISM).astype(int)
    has_both = 0 < binary.sum() < len(binary)
    bruxism_scores = probs[:, BRUXISM]

    event = event_level_metrics(y_true, bruxism_scores, segment_index, BRUXISM, np.array([threshold]))[0]
    event_f1 = (
        2 * event["precision"] * event["recall"] / (event["precision"] + event["recall"])
        if event["precision"] + event["recall"] > 0
        else 0.0
    )
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        **{f"{c}_precision": float(p[i]) for i, c in enumerate(CLASS_NAMES)},
        **{f"{c}_recall": float(r[i]) for i, c in enumerate(CLASS_NAMES)},
        **{f"{c}_f1": float(f[i]) for i, c in enumerate(CLASS_NAMES)},
        "bruxism_ap": float(average_precision_score(binary, bruxism_scores)) if has_both else float("nan"),
        "bruxism_roc_auc": float(roc_auc_score(binary, bruxism_scores)) if has_both else float("nan"),
        "bruxism_chance_ap": float(binary.mean()),
        "event_precision": float(event["precision"]),
        "event_recall": float(event["recall"]),
        "event_f1": float(event_f1),
        "events_true": int(event["n_bruxism_events"]),
        "events_found": int(event["true_positive"]),
        "events_shown": int(event["events_shown"]),
        "n_frames": int(len(y_true)),
        "n_bruxism_frames": int(binary.sum()),
    }


def majority_probs(n: int, majority_class: int) -> np.ndarray:
    probs = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
    probs[:, majority_class] = 1.0
    return probs


def run(config_path: Path, output_dir: Path, threshold: float, epochs: int, seed: int) -> None:
    paths = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tsv = Path(paths["dreamcatcher"]["segmentation_tsv"])
    cache_dir = Path(paths["artifacts"]["embedding_cache"])

    segments = load_segments(tsv)
    cache = cache_path_for(cache_dir, tsv, len(segments))
    if not cache.exists():
        raise SystemExit(f"ไม่พบแคช embedding ที่ {cache} — รัน python -m tools.train_model ก่อน")
    dataset = extract_frame_dataset(segments, Path(paths["dreamcatcher"]["audio_dir"]), None, cache)
    sessions, subjects = relabel_groups(dataset, segments)

    x, y = dataset.embeddings, dataset.labels
    label_names = np.array([CLASS_NAMES[i] for i in y])
    subject_ids = sorted(s for s in set(subjects.tolist()) if s)
    unparsed = int((subjects == "").sum())
    logger.info("subject ทั้งหมด %s | เฟรมที่ยังแยก subject ไม่ได้ %d", subject_ids, unparsed)

    fold_rows: list[dict] = []
    for held_out in subject_ids:
        test_mask = subjects == held_out
        train_pool = (~test_mask) & (subjects != "")

        # Early stopping needs its own validation data, and it must come from
        # the training subjects — using the held-out subject here would leak
        # it back into training through the choice of when to stop.
        pool_idx = np.flatnonzero(train_pool)
        inner_train_groups, inner_val_groups = grouped_split(
            sessions[pool_idx].tolist(), label_names[pool_idx].tolist(), val_fraction=INNER_VAL_FRACTION
        )
        inner_val = train_pool & np.isin(sessions, list(inner_val_groups))
        inner_train = train_pool & np.isin(sessions, list(inner_train_groups))

        logger.info(
            "fold %s: train %d เฟรม (bruxism %d) | ทดสอบ %d เฟรม (bruxism %d)",
            held_out, int(inner_train.sum()), int((y[inner_train] == BRUXISM).sum()),
            int(test_mask.sum()), int((y[test_mask] == BRUXISM).sum()),
        )

        artifacts = train_head(
            x[inner_train], y[inner_train], x[inner_val], y[inner_val], epochs=epochs, seed=seed
        )
        head_probs = predict_proba(artifacts.model, artifacts.mean, artifacts.std, x[test_mask])

        x_train_scaled, mean, (x_test_scaled,) = standardize(x[inner_train | inner_val], x[test_mask])
        logistic = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
        logistic.fit(x_train_scaled, y[inner_train | inner_val])
        logistic_probs = logistic.predict_proba(x_test_scaled)

        majority = int(np.bincount(y[inner_train | inner_val]).argmax())
        majority_scores = majority_probs(int(test_mask.sum()), majority)

        seg = dataset.segment_index[test_mask]
        for name, probs in (
            ("MLP head (ใช้จริง)", head_probs),
            ("Logistic regression", logistic_probs),
            ("Majority class", majority_scores),
        ):
            row = scores_for(name, y[test_mask], probs, seg, threshold)
            row["held_out_subject"] = held_out
            fold_rows.append(row)

    write_outputs(fold_rows, subject_ids, threshold, unparsed, output_dir)


def summarize(fold_rows: list[dict]) -> list[dict]:
    keys = [
        "accuracy", "macro_f1", "bruxism_precision", "bruxism_recall", "bruxism_f1",
        "bruxism_ap", "bruxism_roc_auc", "bruxism_chance_ap",
        "event_precision", "event_recall", "event_f1",
    ]
    out = []
    for model in dict.fromkeys(r["model"] for r in fold_rows):
        rows = [r for r in fold_rows if r["model"] == model]
        summary = {"model": model}
        for k in keys:
            values = np.array([r[k] for r in rows], dtype=float)
            values = values[~np.isnan(values)]
            summary[f"{k}_mean"] = float(values.mean()) if values.size else float("nan")
            summary[f"{k}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        summary["events_found"] = int(sum(r["events_found"] for r in rows))
        summary["events_true"] = int(sum(r["events_true"] for r in rows))
        summary["events_shown"] = int(sum(r["events_shown"] for r in rows))
        out.append(summary)
    return out


def write_outputs(fold_rows, subject_ids, threshold, unparsed, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(fold_rows)

    with (output_dir / "loso_per_fold.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    (output_dir / "loso_metrics.json").write_text(
        json.dumps(
            {"threshold": threshold, "subjects": subject_ids, "unparsed_frames": unparsed,
             "per_fold": fold_rows, "summary": summary},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    plot(fold_rows, subject_ids, output_dir / "loso.png")
    print_report(fold_rows, summary, subject_ids, threshold, output_dir)


def plot(fold_rows, subject_ids, path: Path) -> None:
    """Per-subject bars for the deployed head vs the baselines.

    English text: the default matplotlib font has no Thai glyphs.
    """
    models = list(dict.fromkeys(r["model"] for r in fold_rows))
    labels = {"MLP head (ใช้จริง)": "MLP head (deployed)",
              "Logistic regression": "Logistic regression",
              "Majority class": "Majority class"}
    metrics = [("bruxism_ap", "Bruxism average precision"),
               ("event_recall", "Event recall @ threshold"),
               ("macro_f1", "Macro F1 (frame)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    width = 0.8 / len(models)
    xs = np.arange(len(subject_ids))
    for ax, (key, title) in zip(axes, metrics):
        for i, model in enumerate(models):
            values = [next(r[key] for r in fold_rows if r["model"] == model and r["held_out_subject"] == s)
                      for s in subject_ids]
            ax.bar(xs + i * width - 0.4 + width / 2, values, width, label=labels.get(model, model))
        if key == "bruxism_ap":
            chance = [next(r["bruxism_chance_ap"] for r in fold_rows if r["held_out_subject"] == s)
                      for s in subject_ids]
            ax.scatter(xs, chance, marker="_", s=400, color="black", zorder=5, label="chance")
        ax.set_xticks(xs, [f"subject {s}" for s in subject_ids])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Leave-one-subject-out: each bar is a person the model never trained on")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(summary: dict, key: str) -> str:
    m, s = summary[f"{key}_mean"], summary[f"{key}_std"]
    return "   n/a     " if np.isnan(m) else f"{m:.3f}±{s:.3f}"


def print_report(fold_rows, summary, subject_ids, threshold, output_dir: Path) -> None:
    print()
    print("=" * 78)
    print(f"Leave-one-subject-out ({len(subject_ids)} folds) — ทดสอบกับคนที่โมเดลไม่เคยเห็น")
    print("=" * 78)
    print(f"threshold ระดับ event: {threshold}")
    print()
    print(f"{'โมเดล':<22}{'macro F1':>13}{'bruxism AP':>13}{'ROC-AUC':>13}{'event R':>13}")
    for s in summary:
        print(f"{s['model']:<22}{_fmt(s, 'macro_f1'):>13}{_fmt(s, 'bruxism_ap'):>13}"
              f"{_fmt(s, 'bruxism_roc_auc'):>13}{_fmt(s, 'event_recall'):>13}")
    head = next(s for s in summary if s["model"].startswith("MLP"))
    print()
    print(f"ระดับสุ่ม (chance) ของ bruxism AP: {_fmt(head, 'bruxism_chance_ap')}")
    print()
    print("รายคน (MLP head):")
    print(f"  {'subject':<10}{'AP':>8}{'ROC-AUC':>10}{'event P':>10}{'event R':>10}{'เจอ/ทั้งหมด':>16}")
    for r in fold_rows:
        if not r["model"].startswith("MLP"):
            continue
        print(f"  {r['held_out_subject']:<10}{r['bruxism_ap']:>8.3f}{r['bruxism_roc_auc']:>10.3f}"
              f"{r['event_precision']:>10.3f}{r['event_recall']:>10.3f}"
              f"{r['events_found']:>8}/{r['events_true']:<7}")
    print("=" * 78)
    print(f"บันทึกไว้ที่ {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ประเมินแบบ leave-one-subject-out")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    run(args.config, args.output, args.threshold, args.epochs, args.seed)


if __name__ == "__main__":
    main()
