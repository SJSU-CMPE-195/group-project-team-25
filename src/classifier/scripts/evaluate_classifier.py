"""Evaluate a trained human-likelihood classifier against labeled sessions.

Usage (from repo root):
    python scripts/evaluate_classifier.py \
        --model-dir classifier/models/xgb_v1 \
        --data-dir data/

Output:
    - Confusion matrix
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Score distribution histogram (text-based)
    - Per-session scores (useful for debugging edge cases)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from classifier.data_loader import load_from_directory
from classifier.features import SessionFeatureExtractor, FEATURE_NAMES
from classifier.model import HumanLikelihoodClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate human-likelihood classifier")
    p.add_argument(
        "--model-dir", type=str, default="classifier/models/xgb_v1",
        help="Directory containing the saved classifier",
    )
    p.add_argument(
        "--data-dir", type=str, default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for human vs bot (default: 0.5)",
    )
    p.add_argument(
        "--include-augmented", action="store_true",
        help="Include adversarially augmented bot sessions from data/bot_augmented/ "
             "in the evaluation pool",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-session scores",
    )
    return p.parse_args()


def text_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    """Render a simple text-based histogram of score values."""
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_count = max(counts) if counts.max() > 0 else 1
    lines = ["  Score range    | Count | Bar"]
    lines.append("  " + "-" * 50)
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * int(count / max_count * width)
        lines.append(f"  [{lo:.1f} – {hi:.1f})  | {count:5d} | {bar}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    print(f"[evaluate_classifier] Loading model from {model_dir.resolve()} ...")
    clf = HumanLikelihoodClassifier.load(model_dir)

    # ------------------------------------------------------------------
    # 2. Load sessions
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    print(f"[evaluate_classifier] Loading sessions from {data_dir.resolve()} ...")
    sessions = load_from_directory(data_dir, include_augmented=args.include_augmented)
    labeled  = [s for s in sessions if s.label is not None]

    if not labeled:
        print("ERROR: No labeled sessions found.")
        sys.exit(1)

    humans = [s for s in labeled if s.label == 1]
    bots   = [s for s in labeled if s.label == 0]
    print(f"  Human sessions : {len(humans)}")
    print(f"  Bot sessions   : {len(bots)}")
    if args.include_augmented:
        from classifier.data_loader import is_augmented as _is_aug
        n_aug = sum(1 for s in labeled if _is_aug(s))
        print(f"  (of which {n_aug} are pre-generated augmented bot sessions)")

    # ------------------------------------------------------------------
    # 3. Extract features and predict
    # ------------------------------------------------------------------
    extractor = SessionFeatureExtractor()
    X = extractor.extract_many(labeled)
    y_true = np.array([s.label for s in labeled], dtype=int)

    y_score  = clf.human_score(X)
    y_pred   = (y_score >= args.threshold).astype(int)

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
        roc_curve,
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- Results (threshold={args.threshold}) ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of those predicted human, how many were?)")
    print(f"  Recall    : {rec:.4f}   (of all humans, how many did we catch?)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    print("\n--- Confusion Matrix ---")
    print("             Predicted Bot  Predicted Human")
    print(f"  True Bot   {cm[0][0]:13d}  {cm[0][1]:15d}")
    print(f"  True Human {cm[1][0]:13d}  {cm[1][1]:15d}")

    false_positive_rate = cm[1][0] / max(cm[1].sum(), 1)
    false_negative_rate = cm[0][1] / max(cm[0].sum(), 1)
    print(f"\n  False positive rate (humans wrongly blocked): {false_positive_rate:.2%}")
    print(f"  False negative rate (bots wrongly allowed) : {false_negative_rate:.2%}")

    print()
    print(classification_report(y_true, y_pred, target_names=["bot", "human"], zero_division=0))

    # ------------------------------------------------------------------
    # 5. Score distribution
    # ------------------------------------------------------------------
    human_scores = y_score[y_true == 1].tolist()
    bot_scores   = y_score[y_true == 0].tolist()

    if human_scores:
        print("--- Human score distribution ---")
        print(text_histogram(human_scores))

    if bot_scores:
        print("\n--- Bot score distribution ---")
        print(text_histogram(bot_scores))

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 6a. Confusion matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bot", "Human"])
    ax.set_yticklabels(["Bot", "Human"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)

    # 6b. ROC curve
    ax = axes[0, 1]
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "Not enough classes\nfor ROC curve", ha="center", va="center")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    # 6c. Score distribution histogram
    ax = axes[1, 0]
    if human_scores:
        ax.hist(human_scores, bins=20, range=(0, 1), alpha=0.6, label="Human", color="steelblue")
    if bot_scores:
        ax.hist(bot_scores, bins=20, range=(0, 1), alpha=0.6, label="Bot", color="tomato")
    ax.axvline(x=args.threshold, color="black", linestyle="--", lw=1, label=f"Threshold={args.threshold}")
    ax.set_xlabel("Human-Likelihood Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # 6d. Feature importance (top 10)
    ax = axes[1, 1]
    importances = clf.feature_importances(feature_names=FEATURE_NAMES)
    top_names = list(importances.keys())[:10]
    top_scores = [importances[n] for n in top_names]
    top_names.reverse()
    top_scores.reverse()
    ax.barh(top_names, top_scores, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances")

    fig.suptitle(f"Classifier Evaluation  (n={len(labeled)}, threshold={args.threshold})", fontsize=14)
    fig.tight_layout()
    plt.savefig(Path(args.model_dir) / "evaluation_plots.png", dpi=150)
    print(f"\n[evaluate_classifier] Plots saved to {(Path(args.model_dir) / 'evaluation_plots.png').resolve()}")
    plt.show()

    # ------------------------------------------------------------------
    # 7. Per-session breakdown (verbose)
    # ------------------------------------------------------------------
    if args.verbose:
        print("\n--- Per-session scores ---")
        print(f"  {'Session ID':<40s} {'True':>6} {'Score':>7} {'Pred':>6} {'Correct':>8}")
        print("  " + "-" * 75)
        for session, score, pred, true in zip(labeled, y_score, y_pred, y_true):
            label_str = "human" if true == 1 else "bot"
            pred_str  = "human" if pred == 1 else "bot"
            correct   = "OK" if pred == true else "WRONG"
            print(
                f"  {session.session_id:<40s} {label_str:>6} "
                f"{score:>7.4f} {pred_str:>6} {correct:>8}"
            )


if __name__ == "__main__":
    main()
