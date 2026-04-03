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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_captcha.config import ClassifierConfig

if TYPE_CHECKING:
    from classifier.data_loader import Session
    from classifier.features import SessionFeatureExtractor


class HumanLikelihoodClassifier:
    """XGBoost binary classifier producing a human-likelihood score.

    Chosen over alternatives because:
    - The 22 input features are aggregate session statistics (tabular data),
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

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self._model = None
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

        Applies label smoothing and feature noise augmentation from config
        to improve generalization and produce calibrated probabilities.

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

        # --- Feature noise augmentation ---
        # Add Gaussian noise scaled to each feature's std to simulate
        # data variation and prevent the model from memorizing exact values.
        # This forces the model to learn broader decision boundaries.
        X_train = X.copy().astype(np.float64)
        if cfg.feature_noise_std > 0:
            rng = np.random.RandomState(cfg.random_state)
            feature_stds = np.std(X_train, axis=0)
            noise = (
                rng.normal(0, 1, size=X_train.shape)
                * feature_stds
                * cfg.feature_noise_std
            )
            X_train = X_train + noise

        y_train = np.array(y, dtype=int)

        # --- Label smoothing via sample weights ---
        # Reduce sample weights so the model doesn't over-commit to any
        # single training example, producing softer probability estimates.
        alpha = cfg.label_smooth_alpha
        sample_weights = np.full(len(y_train), 1.0 - alpha)

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
            fit_kwargs["eval_set"] = [(X_val, np.array(y_val, dtype=int))]
            fit_kwargs["verbose"] = False

        self._model.fit(**fit_kwargs)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2): [bot_prob, human_prob]."""
        self._check_fitted()
        return self._model.predict_proba(X)

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
        return self._model.predict(X)

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

        Creates the directory if it does not exist. Saves two files:
        ``xgb_classifier.pkl`` (model) and ``classifier_config.pkl`` (config).
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / self.MODEL_FILENAME, "wb") as f:
            pickle.dump(self._model, f)

        with open(directory / self.CONFIG_FILENAME, "wb") as f:
            pickle.dump(self.config, f)

        print(f"[HumanLikelihoodClassifier] Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "HumanLikelihoodClassifier":
        """Load a saved classifier from *directory*."""
        directory = Path(directory)
        model_path = directory / cls.MODEL_FILENAME
        config_path = directory / cls.CONFIG_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"No model file found at {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        config = ClassifierConfig()
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)

        instance = cls(config=config)
        instance._model = model
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
