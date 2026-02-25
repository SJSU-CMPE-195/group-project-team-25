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
    """Parameters for feature extraction."""

    # Mouse
    mouse_speed_cap: float = 5000.0  # px/s — clip extreme outliers
    jitter_threshold: float = 3.0  # px — movements below this are jitter

    # Keystroke
    max_hold_duration: float = 2000.0  # ms — cap unreasonable holds
    max_interval: float = 5000.0  # ms — cap unreasonable gaps

    # Scroll
    scroll_speed_cap: float = 10000.0  # px/s


@dataclass
class ClassifierConfig:
    """XGBoost hyperparameters."""

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class EnvConfig:
    """Gymnasium environment parameters.

    Action indices:
        0 = monitor          (non-terminal: keep watching)
        1 = deploy_honeypot  (non-terminal: deploy invisible trap)
        2 = easy_puzzle      (terminal: challenge user)
        3 = medium_puzzle    (terminal: challenge user)
        4 = hard_puzzle      (terminal: challenge user)
        5 = allow            (terminal: let user through)
        6 = block            (terminal: deny access)
    """

    # Multi-step episode settings
    max_steps: int = 6                    # max decisions per episode
    window_duration_ms: float = 5000.0    # telemetry window size (ms)

    # Action costs (UX friction penalty applied per step)
    action_costs: list[float] = field(
        default_factory=lambda: [
            0.0,   # monitor
            0.05,  # deploy_honeypot
            0.1,   # easy_puzzle
            0.3,   # medium_puzzle
            0.5,   # hard_puzzle
            0.0,   # allow
            0.0,   # block
        ]
    )

    # Reward weights
    reward_correct_block: float = 1.0   # correctly block/puzzle a bot
    reward_correct_allow: float = 0.5   # correctly allow a human (high enough to incentivize discrimination)
    penalty_false_positive: float = -1.0  # block/puzzle a human
    penalty_false_negative: float = -0.8  # allow a bot through

    # Multi-step intermediate rewards
    honeypot_info_bonus: float = 0.3      # bonus when honeypot catches bot
    time_pressure_coeff: float = 0.02     # per-step increasing penalty
    truncation_penalty: float = -0.5      # penalty for running out of steps
    max_honeypots: int = 2                # max honeypots per episode

    # Puzzle pass probabilities per difficulty for humans / bots
    # Keys match action indices 2, 3, 4
    puzzle_pass_rates: dict = field(
        default_factory=lambda: {
            2: (0.95, 0.40),  # easy
            3: (0.85, 0.15),  # medium
            4: (0.70, 0.05),  # hard
        }
    )

    # Honeypot: bots have a chance of triggering it
    honeypot_trigger_rate_bot: float = 0.6
    honeypot_trigger_rate_human: float = 0.01


@dataclass
class SACConfig:
    """Soft Actor-Critic hyperparameters."""

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 1e-4               # slower alpha learning prevents divergence
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 0.2
    target_entropy_ratio: float = 0.4    # 40% of max entropy (was 0.98 — caused alpha explosion)
    alpha_max: float = 5.0               # hard cap on alpha
    batch_size: int = 256
    buffer_size: int = 100_000
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    update_after: int = 1000  # steps before first update
    update_every: int = 1  # update frequency


@dataclass
class EventEnvConfig:
    """Event-level Gymnasium environment parameters.

    Each timestep = one telemetry event. The agent processes raw events
    through its LSTM and decides an action at every event.

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
    event_dim: int = 13               # fixed-size encoding per event
    mouse_subsample: int = 5          # keep every Nth mouse event (66Hz → ~13Hz)

    # Episode limits
    max_events: int = 500             # truncate episodes after this many events
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
    continue_penalty: float = -0.001     # tiny per-event time pressure
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
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1

    # Rollout
    rollout_steps: int = 2048
    num_epochs: int = 4

    # Training
    total_timesteps: int = 500_000


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    db: DBConfig = field(default_factory=DBConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    event_env: EventEnvConfig = field(default_factory=EventEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    models_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "rl_captcha" / "classifier" / "models"
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "rl_captcha" / "agent" / "checkpoints"
    )
