"""Train the hidden-scoring XGBoost classifier on telemetry sessions.

Usage (from repo root):
    python classifier/scripts/train_classifier.py --data-dir data/ --output-dir classifier/models/xgb_v1

    # With hyperparameter tuning (requires optuna):
    python classifier/scripts/train_classifier.py --data-dir data/ --tune --n-trials 50

    # With pre-generated adversarially augmented bot sessions (run
    # generate_augmented_data.py first):
    python classifier/scripts/train_classifier.py --data-dir data/ --adversarial-augment

The script:
    1. Loads labeled sessions from data/human/ (label=1) and data/bot/ (label=0).
       With --adversarial-augment, also loads pre-generated humanized bot
       sessions from data/bot_augmented/ (added to the train split only).
    2. Splits data into train/test (80/20 stratified)
    3. Extracts 39 aggregate features per session
    4. Optionally tunes hyperparameters with Optuna
    5. Trains on train set
    6. Evaluates on held-out test set
    7. Prints ranked feature importances
    8. Saves the model to --output-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or scripts/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from classifier.data_loader import is_augmented, load_from_directory
from classifier.features import SessionFeatureExtractor, FEATURE_NAMES
from classifier.model import HumanLikelihoodClassifier
from rl_captcha.config import ClassifierConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train human-likelihood classifier")
    p.add_argument(
        "--data-dir", type=str, default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--output-dir", type=str, default="classifier/models/xgb_v1",
        help="Directory to save the trained model",
    )
    p.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    p.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--no-adversarial", action="store_true",
        help="Disable feature-level adversarial augmentation",
    )
    p.add_argument(
        "--adversarial-augment", action="store_true",
        help="Include pre-generated adversarially augmented bot sessions "
             "from data/bot_augmented/ (run generate_augmented_data.py first)",
    )
    p.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)",
    )
    return p.parse_args()


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    random_state: int,
) -> ClassifierConfig:
    """Run Optuna hyperparameter search with cross-validation scoring."""
    try:
        import optuna
    except ImportError:
        print("ERROR: optuna is required for --tune. Install with: pip install optuna")
        sys.exit(1)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        cfg = ClassifierConfig(
            n_estimators=200,
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
            feature_noise_std=trial.suggest_float("feature_noise_std", 0.0, 1.0),
            n_augment_copies=trial.suggest_int("n_augment_copies", 0, 5),
            label_smooth_alpha=trial.suggest_float("label_smooth_alpha", 0.0, 0.15),
            adversarial_augment=trial.suggest_categorical("adversarial_augment", [True, False]),
            n_adversarial_copies=trial.suggest_int("n_adversarial_copies", 1, 4),
            adversarial_blend_range=(
                trial.suggest_float("adv_blend_lo", 0.1, 0.4),
                trial.suggest_float("adv_blend_hi", 0.3, 0.7),
            ),
            adversarial_noise_std=trial.suggest_float("adversarial_noise_std", 0.1, 0.5),
            random_state=random_state,
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = HumanLikelihoodClassifier(config=cfg)
            clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

            try:
                auc = roc_auc_score(y_va, clf.human_score(X_va))
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\n--- Optuna Best Trial (AUC={study.best_value:.4f}) ---")
    for k, v in best.items():
        print(f"  {k}: {v}")

    return ClassifierConfig(
        n_estimators=200,
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_weight=best["min_child_weight"],
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
        gamma=best["gamma"],
        feature_noise_std=best["feature_noise_std"],
        n_augment_copies=best["n_augment_copies"],
        label_smooth_alpha=best["label_smooth_alpha"],
        adversarial_augment=best["adversarial_augment"],
        n_adversarial_copies=best["n_adversarial_copies"],
        adversarial_blend_range=(best["adv_blend_lo"], best["adv_blend_hi"]),
        adversarial_noise_std=best["adversarial_noise_std"],
        random_state=random_state,
    )


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load sessions
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    print(f"[train_classifier] Loading sessions from {data_dir.resolve()} ...")
    sessions = load_from_directory(data_dir, include_augmented=args.adversarial_augment)

    labeled = [s for s in sessions if s.label is not None]
    if not labeled:
        print("ERROR: No labeled sessions found. Check data/human/ and data/bot/ directories.")
        sys.exit(1)

    # Separate originals from any pre-generated augmented copies; the test
    # split must only contain originals so evaluation reflects the real-world
    # distribution and augmented copies don't leak across the split.
    originals = [s for s in labeled if not is_augmented(s)]
    aug_sessions = [s for s in labeled if is_augmented(s)]

    humans = [s for s in originals if s.label == 1]
    bots   = [s for s in originals if s.label == 0]
    print(f"  Human sessions : {len(humans)}")
    print(f"  Bot sessions   : {len(bots)}")
    print(f"  Augmented bots : {len(aug_sessions)}")
    print(f"  Total          : {len(labeled)}")

    if len(originals) < 4:
        print(
            "WARNING: Very few original sessions available — classifier will be unreliable.\n"
            "Collect more data before using this model in production."
        )

    # ------------------------------------------------------------------
    # 2. Train / Test split (originals only; augmented copies join train)
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_labels = np.array([s.label for s in originals], dtype=int)
    train_idx, test_idx = train_test_split(
        np.arange(len(originals)),
        test_size=args.test_size,
        stratify=y_labels if len(np.unique(y_labels)) > 1 else None,
        random_state=args.random_state,
    )

    train_sessions = [originals[i] for i in train_idx]
    test_sessions  = [originals[i] for i in test_idx]

    n_h_train = sum(1 for s in train_sessions if s.label == 1)
    n_b_train = sum(1 for s in train_sessions if s.label == 0)
    n_h_test  = sum(1 for s in test_sessions if s.label == 1)
    n_b_test  = sum(1 for s in test_sessions if s.label == 0)

    print(f"\n  Train set: {len(train_sessions)} ({n_h_train}H / {n_b_train}B)")
    print(f"  Test set : {len(test_sessions)} ({n_h_test}H / {n_b_test}B)")

    # ------------------------------------------------------------------
    # 2b. Add pre-generated humanized bot sessions to the train split only
    # ------------------------------------------------------------------
    if aug_sessions:
        train_sessions = train_sessions + aug_sessions
        print(f"  Train set after augmentation: {len(train_sessions)} "
              f"({n_h_train}H / {n_b_train + len(aug_sessions)}B)")
    elif args.adversarial_augment:
        print("  WARNING: --adversarial-augment is on but no augmented sessions were loaded.")

    # ------------------------------------------------------------------
    # 3. Extract features
    # ------------------------------------------------------------------
    print("[train_classifier] Extracting features ...")
    extractor = SessionFeatureExtractor()
    X_train = extractor.extract_many(train_sessions)
    y_train = np.array([s.label for s in train_sessions], dtype=int)
    X_test  = extractor.extract_many(test_sessions)
    y_test  = np.array([s.label for s in test_sessions], dtype=int)

    print(f"  Train feature matrix: {X_train.shape}")
    print(f"  Test feature matrix : {X_test.shape}")

    # ------------------------------------------------------------------
    # 4. Optional hyperparameter tuning
    # ------------------------------------------------------------------
    config = ClassifierConfig()
    if args.no_adversarial:
        config.adversarial_augment = False
    if args.tune:
        print(f"\n[train_classifier] Running Optuna tuning ({args.n_trials} trials) ...")
        config = tune_hyperparameters(X_train, y_train, args.n_trials, args.random_state)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\n[train_classifier] Training final model on {len(y_train)} sessions ...")
    clf = HumanLikelihoodClassifier(config=config)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate on held-out test set (never seen during training)
    # ------------------------------------------------------------------
    y_pred  = clf.predict(X_test)
    y_score = clf.human_score(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = float("nan")

    print(f"\n--- Test Set Results ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    # ------------------------------------------------------------------
    # 7. Feature importances
    # ------------------------------------------------------------------
    importances = clf.feature_importances(feature_names=FEATURE_NAMES)
    print("\n--- Feature Importances (descending) ---")
    for name, score in importances.items():
        bar = "#" * int(score * 40)
        print(f"  {name:<40s} {score:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    clf.save(output_dir)
    print(f"\n[train_classifier] Final model saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
