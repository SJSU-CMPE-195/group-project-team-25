"""Train the hidden-scoring XGBoost classifier on telemetry sessions.

Usage (from repo root):
    python classifier/scripts/train_classifier.py --data-dir data/ --output-dir classifier/models/xgb_v1

The script:
    1. Loads labeled sessions from data/human/ (label=1) and data/bot/ (label=0)
    2. Extracts 22 aggregate features per session
    3. Splits data into train/test sets (80/20 stratified)
    4. Trains on train set with early stopping on test set
    5. Prints accuracy, F1, ROC-AUC on held-out test set
    6. Prints ranked feature importances
    7. Saves the model to --output-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or scripts/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from classifier.data_loader import load_from_directory
from classifier.features import SessionFeatureExtractor, FEATURE_NAMES
from classifier.model import HumanLikelihoodClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train human-likelihood classifier")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="classifier/models/xgb_v1",
        help="Directory to save the trained model",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load sessions
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    print(f"[train_classifier] Loading sessions from {data_dir.resolve()} ...")
    sessions = load_from_directory(data_dir)

    labeled = [s for s in sessions if s.label is not None]
    if not labeled:
        print(
            "ERROR: No labeled sessions found. Check data/human/ and data/bot/ directories."
        )
        sys.exit(1)

    humans = [s for s in labeled if s.label == 1]
    bots = [s for s in labeled if s.label == 0]
    print(f"  Human sessions : {len(humans)}")
    print(f"  Bot sessions   : {len(bots)}")
    print(f"  Total          : {len(labeled)}")

    if len(labeled) < 4:
        print(
            "WARNING: Very few sessions available — classifier will be unreliable.\n"
            "Collect more data before using this model in production."
        )

    # ------------------------------------------------------------------
    # 2. Extract features
    # ------------------------------------------------------------------
    print("[train_classifier] Extracting features ...")
    extractor = SessionFeatureExtractor()
    X = extractor.extract_many(labeled)
    y = np.array([s.label for s in labeled], dtype=int)

    print(f"  Feature matrix shape: {X.shape}  (sessions x features)")

    # ------------------------------------------------------------------
    # 3. Cross-validation for honest generalization estimate
    # ------------------------------------------------------------------
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    n_folds = 5
    print(
        f"\n[train_classifier] Running {n_folds}-fold stratified cross-validation ..."
    )
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=args.random_state
    )

    cv_accs, cv_f1s, cv_aucs = [], [], []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        fold_clf = HumanLikelihoodClassifier()
        fold_clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        y_pred_f = fold_clf.predict(X_va)
        y_score_f = fold_clf.human_score(X_va)

        cv_accs.append(accuracy_score(y_va, y_pred_f))
        cv_f1s.append(f1_score(y_va, y_pred_f, zero_division=0))
        try:
            cv_aucs.append(roc_auc_score(y_va, y_score_f))
        except ValueError:
            cv_aucs.append(float("nan"))

        print(
            f"  Fold {fold_i}: Acc={cv_accs[-1]:.4f}  F1={cv_f1s[-1]:.4f}  AUC={cv_aucs[-1]:.4f}"
        )

    print(f"\n--- Cross-Validation Summary ({n_folds}-fold) ---")
    print(f"  Accuracy : {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}")
    print(f"  F1 Score : {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")
    print(f"  ROC-AUC  : {np.nanmean(cv_aucs):.4f} +/- {np.nanstd(cv_aucs):.4f}")

    # ------------------------------------------------------------------
    # 4. Final model: train/test split for the saved model
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y if len(np.unique(y)) > 1 else None,
        random_state=args.random_state,
    )

    n_human_train = int((y_train == 1).sum())
    n_bot_train = int((y_train == 0).sum())
    n_human_test = int((y_test == 1).sum())
    n_bot_test = int((y_test == 0).sum())
    print(f"\n  Train set: {len(y_train)} ({n_human_train}H / {n_bot_train}B)")
    print(f"  Test set : {len(y_test)} ({n_human_test}H / {n_bot_test}B)")

    print(f"\n[train_classifier] Training final model on {len(y_train)} sessions ...")
    clf = HumanLikelihoodClassifier()
    clf.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # ------------------------------------------------------------------
    # 4b. Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_score = clf.human_score(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = float("nan")

    print(f"\n--- Test Set Results ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    # ------------------------------------------------------------------
    # 5. Feature importances
    # ------------------------------------------------------------------
    importances = clf.feature_importances(feature_names=FEATURE_NAMES)
    print("\n--- Feature Importances (descending) ---")
    for name, score in importances.items():
        bar = "#" * int(score * 40)
        print(f"  {name:<40s} {score:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    clf.save(output_dir)
    print(f"\n[train_classifier] Final model saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
