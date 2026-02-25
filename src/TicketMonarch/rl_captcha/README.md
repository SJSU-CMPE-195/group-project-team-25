# RL CAPTCHA System

A reinforcement learning-based bot detection system that processes raw user telemetry event-by-event using a PPO+LSTM agent. The LSTM learns temporal patterns directly from mouse movements, clicks, keystrokes, and scroll events — no hand-crafted feature extraction needed.

This package is fully standalone and does not import from TicketMonarch.

## Architecture

```
Raw telemetry events (mouse, click, keystroke, scroll)
        |
        v
Event Encoder (13-dim vector per event)
        |
        v
LSTM (128 hidden) -- accumulates evidence over time
        |
        |--> Actor head (128 -> 64 -> 7 logits) --> action
        +--> Critic head (128 -> 64 -> 1 value) --> V(s)
```

### Event Encoding (13 dimensions)

| Dims | Content |
|------|---------|
| 0-4 | Event type one-hot (mouse, click, key_down, key_up, scroll) |
| 5-6 | Normalized x, y coordinates |
| 7 | Log-normalized time delta since last event |
| 8 | Mouse speed (mouse events only) |
| 9 | Scroll delta (scroll events only) |
| 10 | Special key flag (keystroke events only) |
| 11 | Left-click flag (click events only) |
| 12 | Interactive target flag (click events only) |

### Action Space (7 actions)

| Index | Action | Terminal? | Description |
|-------|--------|-----------|-------------|
| 0 | `continue` | No | Keep observing |
| 1 | `deploy_honeypot` | No | Deploy invisible trap fields |
| 2 | `easy_puzzle` | Yes | 95% human pass, 40% bot pass |
| 3 | `medium_puzzle` | Yes | 85% human pass, 15% bot pass |
| 4 | `hard_puzzle` | Yes | 70% human pass, 5% bot pass |
| 5 | `allow` | Yes | Let user through |
| 6 | `block` | Yes | Block user |

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Correctly allow human | +0.5 |
| Correctly block/puzzle bot | +1.0 |
| False positive (challenge human) | -1.0 |
| False negative (allow bot) | -0.8 |
| Honeypot catches bot | +0.3 |
| Per-event continue penalty | -0.001 |
| Truncated (hit max events) | -0.5 |

## Project Structure

```
rl_captcha/
├── config.py                    # EventEnvConfig, PPOConfig (all hyperparameters)
├── requirements.txt             # torch, gymnasium, numpy, scikit-learn
│
├── data/
│   └── loader.py                # Session dataclass, load_from_directory(), load_from_mysql()
│
├── environment/
│   └── event_env.py             # Event-level Gymnasium env (13-dim obs, 7 actions)
│
├── agent/
│   ├── ppo_lstm.py              # PPO algorithm with LSTM recurrence
│   ├── lstm_networks.py         # LSTMActorCritic network
│   ├── rollout_buffer.py        # On-policy buffer with GAE
│   └── checkpoints/
│       └── ppo_run1/            # Trained model weights (gitignored)
│
└── scripts/
    ├── train_ppo.py             # Train PPO+LSTM agent
    └── evaluate_ppo.py          # Evaluate (confusion matrix, F1, action distribution)
```

## Setup

```bash
pip install -r rl_captcha/requirements.txt
```

Dependencies: PyTorch, Gymnasium, NumPy, scikit-learn.

## Training

All commands from the repository root. Training data goes in `data/human/` (label=1) and `data/bot/` (label=0).

### 1. Collect Data

**Human data:** Record real browsing sessions with the Chrome extension. Export JSON to `data/human/`.

**Bot data:** Run bots against the live site with the Chrome extension recording:
```bash
python bots/selenium_bot.py --runs 5 --type scripted
python bots/llm_bot.py --runs 3 --provider anthropic
```
Export telemetry to `data/bot/`.

### 2. Train

```bash
python -m rl_captcha.scripts.train_ppo \
    --data-dir data/ --total-timesteps 500000
```

Saves checkpoint to `rl_captcha/agent/checkpoints/ppo_run1/`.

### 3. Evaluate

```bash
python -m rl_captcha.scripts.evaluate_ppo \
    --agent rl_captcha/agent/checkpoints/ppo_run1 \
    --data-dir data/
```

## Live Integration

The trained agent is loaded by `TicketMonarch/backend/agent_service.py` for real-time use:

- **`evaluate_session()`** — Full evaluation: processes all events in one batched LSTM forward pass, averages the last 10% of action probabilities for the final decision
- **`rolling_evaluate()`** — Lightweight polling: returns bot probability and whether to deploy honeypot, using exponentially-weighted recent events
- **`online_learn()`** — Gentle PPO update after confirmed human/bot sessions (1 epoch, 1/5th learning rate, saves checkpoint)

All methods are thread-safe (wrapped with `threading.Lock`).

## Configuration

All hyperparameters in `config.py`:

- **EventEnvConfig** — Event encoding params, episode limits (max 500 events, min 10), reward weights, puzzle pass rates, normalization constants
- **PPOConfig** — Learning rate (3e-4), gamma (0.99), GAE lambda (0.95), clip ratio (0.2), entropy coefficient (0.01)
