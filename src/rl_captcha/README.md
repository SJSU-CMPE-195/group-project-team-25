# RL CAPTCHA System

A reinforcement learning-based bot detection system that processes raw user telemetry using an LSTM agent. Events are grouped into **windows of 30** and encoded as statistical feature vectors (speed variance, timing regularity, path curvature, etc.). The LSTM accumulates evidence across all windows, then makes a terminal decision on the **final window** only.

Three training algorithms are supported:

| Algorithm | Description | Key idea |
|-----------|-------------|----------|
| **PPO** | Proximal Policy Optimization with clipped surrogate | Stable on-policy baseline with fixed entropy bonus |
| **DG** | Delightful Policy Gradients ([arXiv:2603.14608](https://arxiv.org/abs/2603.14608)) | Delight-gated gradients suppress noisy updates from rare bad actions |
| **Soft PPO** | PPO with adaptive entropy temperature (SAC-inspired) | Learnable alpha auto-tunes exploration vs. exploitation |

All three share the same LSTM network, rollout buffer, and environment -- they differ only in how policy gradients are computed during training.

This package is fully standalone and does not import from TicketMonarch.

## Architecture

```
Raw telemetry events (mouse, click, keystroke, scroll)
        |
        v
Windowed Event Encoder (26-dim feature vector per 30-event window)
        |  Features: speed mean/var, path curvature, click/key timing,
        |  spatial diversity, scroll behavior, interaction quality
        |
        v
LSTM (128 hidden, 1 layer) -- accumulates evidence over windows
        |
        |--> Actor head (128 -> 128 -> 64 -> 7 logits) --> action
        +--> Critic head (128 -> 128 -> 64 -> 1 value) --> V(s)
```

### Two-Phase Episode Structure with Action Masking

Episodes have two distinct phases enforced by **action masking** (invalid actions get `-inf` logits):

1. **Observation phase** (all non-final windows): Only actions 0-1 are valid
   - The LSTM processes windows and accumulates evidence
   - Agent can deploy honeypots to gather more information

2. **Decision phase** (final window only): Only actions 2-6 are valid
   - The agent must make a terminal decision based on all accumulated evidence
   - No more observing -- must choose: puzzle, allow, or block

This ensures the agent always processes the **entire session** before deciding, preventing shortcut strategies where it decides on window 1.

### Windowed Observation Encoding (26 dimensions)

| Dims | Feature | Discriminative power |
|------|---------|---------------------|
| 0-3 | Event type ratios (mouse/click/key/scroll) | Bot profiles have different event mixes |
| 4-6 | Mouse speed: mean, variance, acceleration | Bots have low speed variance |
| 7 | Path curvature (path length / displacement) | Bots move in straight lines (~1.0) |
| 8-10 | Inter-event timing: mean, variance, min | Bots have near-zero timing variance |
| 11-12 | Click timing: mean interval, variance | Bots click at regular intervals |
| 13-14 | Keystroke hold: mean duration, variance | Bots have mechanical uniform holds |
| 15-16 | Key-press interval: mean, variance | Typing rhythm regularity |
| 17-18 | Scroll: total distance, direction changes | Bots rarely scroll organically |
| 19-22 | Spatial: unique positions, x/y range | Bots visit fewer screen areas |
| 23 | Interactive click ratio | Bots may click non-interactive areas |
| 24 | Window duration (log-normalized) | Time span of the window |
| 25 | Event count / window size | How full the window is |

### Action Space (7 actions)

| Index | Action | Phase | Description |
|-------|--------|-------|-------------|
| 0 | `continue` | Observation | Keep watching (masked on final window) |
| 1 | `deploy_honeypot` | Observation | Deploy invisible trap (masked on final window) |
| 2 | `easy_puzzle` | Decision | 95% human pass, 40% bot pass (masked on non-final) |
| 3 | `medium_puzzle` | Decision | 85% human pass, 15% bot pass (masked on non-final) |
| 4 | `hard_puzzle` | Decision | 70% human pass, 5% bot pass (masked on non-final) |
| 5 | `allow` | Decision | Let user through (masked on non-final) |
| 6 | `block` | Decision | Block user (masked on non-final) |

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Correctly allow human | +0.5 |
| Correctly block/puzzle bot | +1.0 |
| False positive (challenge human) | -1.0 |
| False negative (allow bot) | -0.8 |
| Honeypot catches bot | +0.3 |
| Per-window continue penalty | -0.001 |

## Project Structure

```
rl_captcha/
├── config.py                    # EventEnvConfig, PPOConfig, DBConfig
├── requirements.txt             # torch, gymnasium, numpy, scikit-learn
│
├── data/
│   └── loader.py                # Session dataclass, load_from_directory()
│                                # Supports: live_confirm, flat JSON
│
├── environment/
│   └── event_env.py             # Windowed Gymnasium env (26-dim obs, 7 actions)
│                                # EventEncoder + action masking + bot data augmentation
│
├── agent/
│   ├── ppo_lstm.py              # PPO algorithm with LSTM recurrence + action masks
│   ├── dg_lstm.py               # Delightful Policy Gradient (extends PPO)
│   ├── soft_ppo_lstm.py         # Soft PPO with adaptive entropy (extends PPO)
│   ├── lstm_networks.py         # LSTMActorCritic (LSTM, 128 hidden)
│   ├── rollout_buffer.py        # On-policy buffer with GAE + mask storage
│   └── checkpoints/
│       ├── ppo_run1/            # PPO trained weights
│       ├── dg_run1/             # DG trained weights
│       └── soft_ppo_run1/       # Soft PPO trained weights
│
└── scripts/
    ├── train_ppo.py             # Train PPO/DG/Soft-PPO agent (--algorithm flag)
    ├── evaluate_ppo.py          # Evaluate one or more agents (multi-agent comparison)
    ├── plot_training.py         # Visualize training.log (auto-detects algorithm)
    ├── plot_comparison.py       # Side-by-side comparison of all algorithms
    ├── plot_eval.py             # Visualize evaluation results
    └── plot_online.py           # Visualize online_training.log
```

## Setup

```bash
pip install -r rl_captcha/requirements.txt
mkdir -p logs figures
```

Dependencies: PyTorch, Gymnasium, NumPy, scikit-learn, matplotlib.

## Data

Training data lives in `data/human/` (label=1) and `data/bot/` (label=0). Sessions are converted into overlapping 30-event windows, then capped to `max_windows` for training, inference, and online learning. Data is split 70/15/15 (train/val/test) with stratified sampling.

**Supported file formats:**
- `session_*.json` -- Live-confirm format from Dev Dashboard: `{ "sessionId": "...", "segments": [{ "mouse": [...], ... }] }`

**Important:** Only include data from the TicketMonarch site (localhost). Data from external sites will pollute the training distribution.

**Current note:** telemetry collection semantics were updated to avoid inflated mouse traces and dropped batches. Recollect fresh human and bot data before training a new checkpoint.

## Training

All commands from the `src/` directory.

### 1. Collect Data

**Human data:** Browse the live site normally. Sessions are auto-saved to `data/human/` when online learning runs via the `/api/agent/confirm` endpoint.

**Bot data:** Run bots against the live site:
```bash
python bots/selenium_bot.py --runs 10 --type scripted
python bots/llm_bot.py --runs 3 --provider anthropic
```

### 2. Train

Use `--algorithm` to select between `ppo`, `dg`, or `soft_ppo`:

```powershell
# Train PPO (default)
python -u -m rl_captcha.scripts.train_ppo --algorithm ppo --data-dir data/ --save-path rl_captcha/agent/checkpoints/ppo_run1 --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/ppo_training.log

# Train DG
python -u -m rl_captcha.scripts.train_ppo --algorithm dg --data-dir data/ --save-path rl_captcha/agent/checkpoints/dg_run1 --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/dg_training.log

# Train Soft PPO
python -u -m rl_captcha.scripts.train_ppo --algorithm soft_ppo --data-dir data/ --save-path rl_captcha/agent/checkpoints/soft_ppo_run1 --total-timesteps 500000 --target-entropy-ratio 0.5 2>&1 | Tee-Object -FilePath logs/soft_ppo_training.log
```

#### Training CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | `ppo` | Training algorithm: `ppo`, `dg`, or `soft_ppo` |
| `--data-dir` | `data/` | Root directory containing `human/` and `bot/` subdirectories |
| `--save-path` | `rl_captcha/agent/checkpoints/ppo_run1` | Where to save model checkpoints |
| `--total-timesteps` | `500000` (from PPOConfig) | Total environment steps to train for |
| `--log-interval` | `1` | Print stats every N rollouts |
| `--save-interval` | `10` | Save checkpoint + run validation every N rollouts |
| `--val-episodes` | `100` | Number of validation episodes per checkpoint |
| `--device` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |
| `--split-seed` | `42` | Random seed for train/val/test split |
| `--dg-temperature` | `1.0` | DG sigmoid temperature (DG only) |
| `--dg-blend` | `0.0` | DG-PPO blend: 0=pure DG, 1=pure PPO (DG only) |
| `--target-entropy-ratio` | `0.5` | Target entropy as fraction of max (Soft PPO only) |
| `--alpha-lr` | `3e-4` | Entropy temperature learning rate (Soft PPO only) |

### 3. Evaluate

Evaluate one or more checkpoints in a single run. Evaluation now includes **per-bot-family** and **per-tier** detection rate breakdowns alongside global metrics.

```powershell
# Single agent — basic eval on test split
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_run1 --episodes 500 --split test

# Single agent — eval on ALL data (no split)
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_run1 --episodes 500 --split all

# All three agents at once (prints comparison table)
python -m rl_captcha.scripts.evaluate_ppo --agent ppo=rl_captcha/agent/checkpoints/ppo_run1 dg=rl_captcha/agent/checkpoints/dg_run1 soft_ppo=rl_captcha/agent/checkpoints/soft_ppo_run1 --episodes 500 --split test 2>&1 | Tee-Object -FilePath logs/eval_all.log

# Held-out generalization: hold out specific bot families from training
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_run1 --episodes 500 --held-out-families stealth replay

# Held-out generalization: hold out entire tiers from training
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_run1 --episodes 500 --held-out-tiers 3 4
```

Outputs per-agent: confusion matrix, accuracy, precision, recall, F1, outcome distribution, action distribution, **per-bot-family detection table**, **per-tier summary**, and a side-by-side comparison table when multiple agents are provided.

#### Tiered Adversarial Evaluation

Bot types are organized into tiers of increasing sophistication:

| Tier | Name | Bot Types | Description |
|------|------|-----------|-------------|
| 1 | Commodity | linear, tabber, speedrun | Simple, fast, easily detectable |
| 2 | Careful Automation | scripted, stealth, slow, erratic, replay | Tries to mimic human timing/patterns |
| 3 | Semi-Automated | semi_auto | Mix of real human + bot actions in one session |
| 4 | Trace-Conditioned | trace_conditioned | Replays perturbed real human traces |

The `--held-out-families` and `--held-out-tiers` flags let you test **generalization**: train on some bot types, then evaluate on unseen ones. Held-out bots are removed from train/val and placed entirely in the test set.

#### Evaluation CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | *(required)* | One or more checkpoint paths (`name=path` pairs for multi-agent) |
| `--data-dir` | `data/` | Root data directory |
| `--episodes` | `500` | Number of episodes to evaluate per agent |
| `--split` | `test` | Data split: `test`, `val`, `train`, or `all` |
| `--split-seed` | `42` | Random seed for split (must match training) |
| `--device` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |
| `--held-out-families` | `None` | Bot families to hold out from train/val (test-only) |
| `--held-out-tiers` | `None` | Bot tiers to hold out from train/val (test-only) |

### 4. Visualize

```powershell
# Individual training plots (auto-detects DG/Soft PPO metrics)
python -m rl_captcha.scripts.plot_training --log logs/ppo_training.log --out figures/ppo
python -m rl_captcha.scripts.plot_training --log logs/dg_training.log --out figures/dg
python -m rl_captcha.scripts.plot_training --log logs/soft_ppo_training.log --out figures/soft_ppo

# Three-way comparison plots
python -m rl_captcha.scripts.plot_comparison --logs ppo=logs/ppo_training.log dg=logs/dg_training.log soft_ppo=logs/soft_ppo_training.log --out figures/comparison

# Evaluation result plots (run after evaluate step)
python -m rl_captcha.scripts.plot_eval --log logs/eval_all.log --out figures/eval

# Plot online learning log
python -m rl_captcha.scripts.plot_online --log online_training.log --out figures/
```

#### Plot CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | `training.log` (plot_training) / *required* (plot_eval) | Path to log file |
| `--logs` | *(required for plot_comparison)* | Training logs as `name=path` pairs |
| `--out` | `figures` | Output directory for figures |
| `--format` | `png` | Output format: `png`, `pdf`, or `svg` |

## Data Augmentation

During training, **bot sessions** are stochastically augmented (50% probability per episode) to prevent the agent from relying on trivially separable features. Human sessions are never augmented. Augmentation is **disabled** for the validation environment so val metrics reflect real data.

| Augmentation | Default | Effect |
|-------------|---------|--------|
| Position noise | std=15px | Gaussian jitter on x/y coordinates |
| Timing jitter | std=30ms | Gaussian noise on event timestamps |
| Speed warp | 0.7x-1.4x | Random time stretch/compress across session |

Configure in `EventEnvConfig`:

```python
augment: bool = True              # enable/disable
augment_prob: float = 0.5         # probability per bot episode
aug_position_noise_std: float = 15.0
aug_timing_jitter_std: float = 30.0
aug_speed_warp_range: tuple = (0.7, 1.4)
```

## Live Integration

The trained agent is loaded by `TicketMonarch/backend/agent_service.py` for real-time use. The algorithm is selected via:

1. The `algorithm` constructor argument to `AgentService`, OR
2. The `RL_ALGORITHM` environment variable (`ppo`, `dg`, or `soft_ppo`), OR
3. Defaults to `ppo`

The correct agent class (PPOLSTM, DGLSTM, or SoftPPOLSTM) is instantiated automatically, and the matching checkpoint subdirectory is loaded.

**Methods:**

- **`evaluate_session()`** -- Full evaluation with action masking: processes all windows through LSTM, applies observation mask on non-final windows and decision mask on the final window, and returns the terminal policy action used by checkout
- **`rolling_evaluate()`** -- Lightweight polling: returns bot probability from the final window's action distribution
- **`online_learn()`** -- Polymorphic update after confirmed human/bot sessions (3 epochs, 60% learning rate). Each algorithm's `update()` method is called automatically (PPO clip, DG delight-gated, Soft PPO adaptive entropy). Logs before/after comparison to `online_training.log`

All methods are thread-safe (wrapped with `threading.Lock`). Online and offline evaluation use identical action masking logic.

## Configuration

All hyperparameters in `config.py` and per-algorithm config classes:

- **EventEnvConfig** -- Window size (30 events), obs dim (26), min events (10), max windows (`256`), reward weights, puzzle pass rates, action masking, normalization constants, data augmentation
- **PPOConfig** -- Learning rate (3e-4), gamma (0.99), GAE lambda (0.95), clip ratio (0.2), entropy coefficient (0.02), LSTM (128 hidden, 1 layer)
- **DGConfig** (extends PPOConfig) -- `dg_temperature` (sigmoid temperature η), `dg_baseline_weight` (PPO blend weight)
- **SoftPPOConfig** (extends PPOConfig) -- `target_entropy_ratio` (fraction of max entropy), `alpha_lr` (temperature learning rate), `init_log_alpha`, `alpha_min`, `alpha_max`
