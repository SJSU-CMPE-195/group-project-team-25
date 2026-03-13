"""Central configuration for the RL CAPTCHA system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from TicketMonarch (reuse the same DB credentials)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / "TicketMonarch" / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


@dataclass
class DBConfig:
    host: str = os.getenv("MYSQL_HOST", "localhost")
    user: str = os.getenv("MYSQL_USER", "root")
    password: str = os.getenv("MYSQL_PASSWORD", "")
    database: str = os.getenv("MYSQL_DATABASE", "ticketmonarch_db")
    port: int = int(os.getenv("MYSQL_PORT", "3306"))


@dataclass
class FeatureConfig:
    """Feature extraction parameters for the session-level classifier."""

    mouse_speed_cap: float = 10_000.0      # px/s — clamp extreme speeds
    jitter_threshold: float = 3.0          # px — movements below this are jitter
    pause_threshold_ms: float = 500.0      # ms — gaps longer than this count as pauses


@dataclass
class ClassifierConfig:
    """XGBoost hyperparameters for the human-likelihood classifier.

    Tuned for generalization on small datasets (~100 sessions).
    Regularization is intentionally strong to avoid memorizing the
    training set and to encourage the model to learn transferable
    behavioral patterns.
    """

    n_estimators: int = 50
    max_depth: int = 2                     # very shallow trees reduce overfitting
    learning_rate: float = 0.05            # slow learning + early stopping
    subsample: float = 0.7                 # row subsampling per tree
    colsample_bytree: float = 0.7          # feature subsampling per tree
    min_child_weight: int = 5              # prevent leaf nodes with few samples
    reg_alpha: float = 0.3                 # L1 regularization
    reg_lambda: float = 2.0                # L2 regularization
    gamma: float = 0.3                     # min loss reduction for a split
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20
    random_state: int = 42

    # Label smoothing: shift hard 0/1 labels toward 0.5 to encourage
    # calibrated probability outputs and reduce overconfidence.
    label_smooth_alpha: float = 0.05       # 0→0.05, 1→0.95

    # Feature noise augmentation: add Gaussian noise to training features
    # to simulate data variation and prevent reliance on exact values.
    feature_noise_std: float = 0.5          # std of noise relative to feature std


@dataclass
class EventEnvConfig:
    """Windowed Gymnasium environment parameters.

    Each timestep = one window of telemetry events. The agent observes
    all windows sequentially through its LSTM, then makes a terminal
    decision on the final window.

    Action masking enforces two phases:
      - Observation phase (non-final windows): only continue (0) and honeypot (1)
      - Decision phase (final window): only terminal actions (2-6)

    Action indices:
        0 = continue         (non-terminal: keep watching)
        1 = deploy_honeypot  (non-terminal: deploy invisible trap)
        2 = easy_puzzle      (terminal: challenge user)
        3 = medium_puzzle    (terminal: challenge user)
        4 = hard_puzzle      (terminal: challenge user)
        5 = allow            (terminal: let user through)
        6 = block            (terminal: deny access)
    """

    # Event encoding
    event_dim: int = 26               # windowed feature vector dimension
    mouse_subsample: int = 5          # keep every Nth mouse event (66Hz → ~13Hz)
    window_size: int = 30             # events per observation window

    # Session limits
    min_events: int = 10              # skip sessions with fewer events

    # Action costs
    action_costs: list[float] = field(
        default_factory=lambda: [
            0.0,    # continue
            0.01,   # deploy_honeypot
            0.1,    # easy_puzzle
            0.3,    # medium_puzzle
            0.5,    # hard_puzzle
            0.0,    # allow
            0.0,    # block
        ]
    )

    # Reward structure
    reward_correct_block: float = 1.0
    reward_correct_allow: float = 0.5
    penalty_false_positive: float = -1.0
    penalty_false_negative: float = -0.8
    continue_penalty: float = 0.001      # tiny per-window time pressure
    honeypot_info_bonus: float = 0.3
    truncation_penalty: float = -0.5
    max_honeypots: int = 2

    # Puzzle pass rates: {action_index: (human_pass, bot_pass)}
    puzzle_pass_rates: dict = field(
        default_factory=lambda: {
            2: (0.95, 0.40),   # easy
            3: (0.85, 0.15),   # medium
            4: (0.70, 0.05),   # hard
        }
    )

    # Honeypot trigger rates
    honeypot_trigger_rate_bot: float = 0.6
    honeypot_trigger_rate_human: float = 0.01

    # Data augmentation (applied to BOT sessions only, per-episode during training)
    augment: bool = True              # enable stochastic augmentation
    augment_prob: float = 0.5         # probability of augmenting each bot episode
    aug_position_noise_std: float = 15.0   # Gaussian noise on x/y coords (px)
    aug_timing_jitter_std: float = 30.0    # Gaussian noise on timestamps (ms)
    aug_speed_warp_range: tuple = (0.7, 1.4)  # random time stretch/compress

    # Normalization constants for event encoding
    max_coord_x: float = 1920.0
    max_coord_y: float = 1080.0
    max_dt_ms: float = 5000.0
    max_speed: float = 5000.0        # px/s
    max_scroll_dy: float = 500.0     # px


@dataclass
class PPOConfig:
    """PPO with LSTM hyperparameters."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # LSTM
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2

    # Rollout
    rollout_steps: int = 2048
    num_epochs: int = 4

    # Training
    total_timesteps: int = 500_000


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    db: DBConfig = field(default_factory=DBConfig)
    event_env: EventEnvConfig = field(default_factory=EventEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    checkpoints_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "rl_captcha" / "agent" / "checkpoints"
    )
