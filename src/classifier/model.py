"""Human-likelihood classifier — the Hidden Scoring Module.

Wraps an XGBoost binary classifier trained on session-level telemetry
features. Outputs a score in [0, 1] where:
    1.0 = very likely human
    0.0 = very likely bot

Typical pipeline::

    from classifier.data_loader import load_from_directory
    from classifier.features import SessionFeatureExtractor
    from classifier.model import HumanLikelihoodClassifier

    sessions = load_from_directory("data/")
    extractor = SessionFeatureExtractor()
    X = extractor.extract_many(sessions)
    y = [s.label for s in sessions]

    clf = HumanLikelihoodClassifier()
    clf.fit(X, y)
    clf.save("classifier/models/xgb_v1")

    score = clf.score_session(session, extractor)  # float in [0, 1]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rl_captcha.config import ClassifierConfig

if TYPE_CHECKING:
    from classifier.data_loader import Session
    from classifier.features import SessionFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class HumanLikelihoodClassifier:
    """XGBoost binary classifier producing a human-likelihood score.

    Chosen over alternatives because:
    - The 39 input features are aggregate session statistics (tabular data),
      not a raw sequence — XGBoost consistently outperforms neural nets here.
    - Handles small labeled datasets well via early stopping + boosting.
    - Provides feature importance scores for research interpretability.
    - Sub-millisecond inference (no GPU needed) meets the <2s performance req.
    - XGBoost is already in requirements.txt and ClassifierConfig already
      defines its hyperparameters.

    Parameters
    ----------
    config : ClassifierConfig, optional
        Hyperparameter config. Defaults to values in rl_captcha/config.py.
    """

    MODEL_FILENAME = "xgb_classifier.pkl"
    CONFIG_FILENAME = "classifier_config.pkl"
    SCALER_FILENAME = "feature_scaler.pkl"

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._model = None
        self._scaler = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: list[int] | np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: list[int] | np.ndarray | None = None,
    ) -> "HumanLikelihoodClassifier":
        """Train the XGBoost classifier.

        Applies feature standardization, label smoothing, multi-copy noise
        augmentation, and class balancing from config to improve
        generalization and produce calibrated probabilities.

        Parameters
        ----------
        X : array of shape (N, feature_dim)
        y : binary labels — 1 = human, 0 = bot
        X_val, y_val : optional validation set for early stopping
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost>=2.0"
            )

        cfg = self.config
        y_arr = np.array(y, dtype=int)

        # --- Feature standardization ---
        # Ensures noise augmentation is applied uniformly across features
        # regardless of their original scale.
        if cfg.standardize:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X.astype(np.float64))
        else:
            self._scaler = None
            X_scaled = X.copy().astype(np.float64)

        # --- Adversarial augmentation: humanize bot samples ---
        # Create copies of bot samples with features blended toward the
        # human distribution. This teaches the classifier to detect bots
        # even when they mimic human-like behavior, forcing it to rely on
        # deeper behavioral patterns rather than surface-level differences.
        if cfg.adversarial_augment and cfg.n_adversarial_copies > 0:
            bot_mask = y_arr == 0
            human_mask = y_arr == 1
            if bot_mask.any() and human_mask.any():
                rng_adv = np.random.RandomState(cfg.random_state + 1)
                human_mean = X_scaled[human_mask].mean(axis=0)
                human_std = X_scaled[human_mask].std(axis=0)
                bot_X = X_scaled[bot_mask]

                adv_X_parts = []
                adv_y_parts = []
                blend_lo, blend_hi = cfg.adversarial_blend_range

                for _ in range(cfg.n_adversarial_copies):
                    # Per-sample random blend factor toward human mean
                    blend = rng_adv.uniform(blend_lo, blend_hi, size=(len(bot_X), 1))
                    humanized = bot_X + blend * (human_mean - bot_X)
                    # Add noise scaled to human feature variability
                    noise = (
                        rng_adv.normal(0, 1, size=humanized.shape)
                        * human_std
                        * cfg.adversarial_noise_std
                    )
                    humanized += noise
                    adv_X_parts.append(humanized)
                    adv_y_parts.append(np.zeros(len(bot_X), dtype=int))  # still bots

                X_scaled = np.vstack([X_scaled] + adv_X_parts)
                y_arr = np.concatenate([y_arr] + adv_y_parts)

                n_adv = sum(len(p) for p in adv_X_parts)
                print(
                    f"  [adversarial] Added {n_adv} humanized bot samples "
                    f"(blend={blend_lo:.1f}-{blend_hi:.1f}, noise_std={cfg.adversarial_noise_std})"
                )

        # --- Multi-copy noise augmentation ---
        # Generate multiple noisy copies of the training data so the model
        # sees more variation and learns broader decision boundaries.
        if cfg.feature_noise_std > 0 and cfg.n_augment_copies > 0:
            rng = np.random.RandomState(cfg.random_state)
            feature_stds = np.std(X_scaled, axis=0)
            augmented_X = [X_scaled]
            augmented_y = [y_arr]
            for _ in range(cfg.n_augment_copies):
                noise = (
                    rng.normal(0, 1, size=X_scaled.shape)
                    * feature_stds
                    * cfg.feature_noise_std
                )
                augmented_X.append(X_scaled + noise)
                augmented_y.append(y_arr)
            X_train = np.vstack(augmented_X)
            y_train = np.concatenate(augmented_y)
        else:
            X_train = X_scaled
            y_train = y_arr

        # --- Label smoothing via sample weights ---
        # Reduce confidence by scaling down sample weights by (1 - alpha).
        # This prevents the model from over-committing to any single
        # training example, producing softer probability estimates.
        alpha = cfg.label_smooth_alpha
        sample_weights = np.full(len(y_train), 1.0 - alpha, dtype=np.float64)

        # --- Class imbalance handling ---
        # Compute scale_pos_weight so the minority class gets proportionally
        # higher weight. This prevents bias toward the majority class.
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1) if n_pos != n_neg else 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            gamma=cfg.gamma,
            scale_pos_weight=scale_pos_weight,
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=(
                cfg.early_stopping_rounds if X_val is not None else None
            ),
            random_state=cfg.random_state,
        )

        fit_kwargs: dict = {
            "X": X_train,
            "y": y_train,
            "sample_weight": sample_weights,
        }
        if X_val is not None and y_val is not None:
            X_val_t = self._transform(X_val)
            fit_kwargs["eval_set"] = [(X_val_t, np.array(y_val, dtype=int))]
            fit_kwargs["verbose"] = False

        self._model.fit(**fit_kwargs)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler if standardization is enabled."""
        if self._scaler is not None:
            return self._scaler.transform(X.astype(np.float64))
        return X.astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2): [bot_prob, human_prob]."""
        self._check_fitted()
        return self._model.predict_proba(self._transform(X))

    def human_score(self, X: np.ndarray) -> np.ndarray:
        """Return human-likelihood scores in [0, 1], shape (N,)."""
        return self.predict_proba(X)[:, 1]

    def score_session(
        self,
        session: "Session",
        extractor: "SessionFeatureExtractor",
    ) -> float:
        """Extract features and return the human-likelihood score for one session.

        Returns
        -------
        float in [0, 1] — probability of the session being human.
        """
        vec = extractor.extract(session)
        return float(self.human_score(vec.reshape(1, -1))[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (1=human, 0=bot), shape (N,)."""
        self._check_fitted()
        return self._model.predict(self._transform(X))

    def feature_importances(self, feature_names: list[str] | None = None) -> dict:
        """Return feature importance scores sorted descending.

        Parameters
        ----------
        feature_names : list of str, optional
            Names matching the feature vector order. If omitted, uses f0..fN.

        Returns
        -------
        dict mapping feature name -> importance score
        """
        self._check_fitted()
        importances = self._model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        pairs = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        return {name: round(float(score), 6) for name, score in pairs}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save the trained model to *directory*.

        Creates the directory if it does not exist. Saves the model,
        config, and scaler (if standardization is enabled).
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / self.MODEL_FILENAME, "wb") as f:
            pickle.dump(self._model, f)

        with open(directory / self.CONFIG_FILENAME, "wb") as f:
            pickle.dump(self.config, f)

        if self._scaler is not None:
            with open(directory / self.SCALER_FILENAME, "wb") as f:
                pickle.dump(self._scaler, f)

        print(f"[HumanLikelihoodClassifier] Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "HumanLikelihoodClassifier":
        """Load a saved classifier from *directory*."""
        directory = Path(directory)
        model_path = directory / cls.MODEL_FILENAME
        config_path = directory / cls.CONFIG_FILENAME
        scaler_path = directory / cls.SCALER_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"No model file found at {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        config = ClassifierConfig()
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)

        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        instance = cls(config=config)
        instance._model = model
        instance._scaler = scaler
        instance._is_fitted = True
        print(f"[HumanLikelihoodClassifier] Loaded from {directory}")
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Classifier is not fitted. Call fit() or load() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"HumanLikelihoodClassifier({status}, config={self.config})"
