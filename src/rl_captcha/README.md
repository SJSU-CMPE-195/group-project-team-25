# RL CAPTCHA System

Reinforcement learning-based bot detection using an LSTM agent. Processes raw user telemetry (mouse, clicks, keystrokes, scrolls) in windows of 30 events and makes a terminal decision (allow, block, or puzzle).

Three algorithms are supported: **PPO**, **DG** (Delightful Gradients), and **Soft PPO** (adaptive entropy). All share the same LSTM network and environment.

---

## Setup

All commands from the `src/` directory.

```bash
pip install -r rl_captcha/requirements.txt
mkdir -p logs figures
```

---

## Training Data

Training data lives in:
- `data/human/` — human sessions (label=1)
- `data/bot/` — bot sessions (label=0)
- `data/bot_augmented/` — augmented bot sessions (generated from bot data)

**Collect human data:** Browse the live site normally. Sessions auto-save to `data/human/` when confirmed.

**Collect bot data:** Run bots against the live site (see [bots/README.md](../bots/README.md)).

---

## Step 1: Generate Augmented Data

Run this once (or re-run when source data changes):

```powershell
python -u -m rl_captcha.scripts.generate_augmented_data --data-dir data/ --n-copies 2 --seed 42 2>&1 | Tee-Object -FilePath logs/generate_augmented.log
```

---

## Step 2: Train

Each algorithm can be trained with or without adversarial augmentation (6 checkpoints total).

### Without augmentation (baseline)

```powershell
python -u -m rl_captcha.scripts.train_ppo --algorithm ppo --data-dir data/ --save-path rl_captcha/agent/checkpoints/ppo_noaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/ppo_noaug_training.log

python -u -m rl_captcha.scripts.train_ppo --algorithm dg --data-dir data/ --save-path rl_captcha/agent/checkpoints/dg_noaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/dg_noaug_training.log

python -u -m rl_captcha.scripts.train_ppo --algorithm soft_ppo --data-dir data/ --save-path rl_captcha/agent/checkpoints/soft_ppo_noaug --total-timesteps 500000 --target-entropy-ratio 0.5 2>&1 | Tee-Object -FilePath logs/soft_ppo_noaug_training.log
```

### With adversarial augmentation

```powershell
python -u -m rl_captcha.scripts.train_ppo --algorithm ppo --adversarial-augment --data-dir data/ --save-path rl_captcha/agent/checkpoints/ppo_advaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/ppo_advaug_training.log

python -u -m rl_captcha.scripts.train_ppo --algorithm dg --adversarial-augment --data-dir data/ --save-path rl_captcha/agent/checkpoints/dg_advaug --total-timesteps 500000 2>&1 | Tee-Object -FilePath logs/dg_advaug_training.log

python -u -m rl_captcha.scripts.train_ppo --algorithm soft_ppo --adversarial-augment --data-dir data/ --save-path rl_captcha/agent/checkpoints/soft_ppo_advaug --total-timesteps 500000 --target-entropy-ratio 0.5 2>&1 | Tee-Object -FilePath logs/soft_ppo_advaug_training.log
```

### Training Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--algorithm` | `ppo` | Algorithm: `ppo`, `dg`, or `soft_ppo` |
| `--adversarial-augment` | off | Include augmented bot data from `data/bot_augmented/` |
| `--data-dir` | `data/` | Root data directory |
| `--save-path` | `rl_captcha/agent/checkpoints/ppo_noaug` | Where to save checkpoints |
| `--total-timesteps` | `500000` | Total training steps |
| `--val-episodes` | `100` | Validation episodes per checkpoint |
| `--save-interval` | `10` | Save checkpoint every N rollouts |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--split-seed` | `42` | Random seed for data split |
| `--target-entropy-ratio` | `0.5` | Target entropy (Soft PPO only) |

---

## Step 3: Evaluate

### Single agent
```powershell
python -u -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_noaug --data-dir data/ --episodes 500 --split test 2>&1 | Tee-Object -FilePath logs/eval_ppo_noaug.log
```

### All 6 agents at once
```powershell
python -u -m rl_captcha.scripts.evaluate_ppo `
    --agent ppo_noaug=rl_captcha/agent/checkpoints/ppo_noaug `
            ppo_advaug=rl_captcha/agent/checkpoints/ppo_advaug `
            dg_noaug=rl_captcha/agent/checkpoints/dg_noaug `
            dg_advaug=rl_captcha/agent/checkpoints/dg_advaug `
            soft_ppo_noaug=rl_captcha/agent/checkpoints/soft_ppo_noaug `
            soft_ppo_advaug=rl_captcha/agent/checkpoints/soft_ppo_advaug `
    --data-dir data/ --episodes 500 --split test `
    2>&1 | Tee-Object -FilePath logs/eval_all.log
```

### Held-out generalization tests
```powershell
# Hold out specific bot families from training
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_advaug --episodes 500 --held-out-families stealth replay

# Hold out entire tiers
python -m rl_captcha.scripts.evaluate_ppo --agent rl_captcha/agent/checkpoints/ppo_advaug --episodes 500 --held-out-tiers 3 4
```

### Evaluation Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--agent` | *(required)* | Checkpoint path(s). Use `name=path` for multi-agent |
| `--data-dir` | `data/` | Root data directory |
| `--episodes` | `500` | Episodes to evaluate per agent |
| `--split` | `test` | Data split: `test`, `val`, `train`, or `all` |
| `--split-seed` | `42` | Must match training seed |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--held-out-families` | none | Bot families to exclude from train/val |
| `--held-out-tiers` | none | Bot tiers to exclude from train/val |

---

## Step 4: Plot

### Individual training curves
```powershell
python -m rl_captcha.scripts.plot_training --log logs/ppo_noaug_training.log --out figures/ppo_noaug
python -m rl_captcha.scripts.plot_training --log logs/dg_noaug_training.log --out figures/dg_noaug
python -m rl_captcha.scripts.plot_training --log logs/soft_ppo_noaug_training.log --out figures/soft_ppo_noaug
```

### 6-way comparison
```powershell
python -m rl_captcha.scripts.plot_comparison `
    --logs ppo_noaug=logs/ppo_noaug_training.log `
           ppo_advaug=logs/ppo_advaug_training.log `
           dg_noaug=logs/dg_noaug_training.log `
           dg_advaug=logs/dg_advaug_training.log `
           soft_ppo_noaug=logs/soft_ppo_noaug_training.log `
           soft_ppo_advaug=logs/soft_ppo_advaug_training.log `
    --out figures/comparison
```

### Evaluation plots
```powershell
# Without augmented test set
python -m rl_captcha.scripts.plot_eval --log logs/eval_all.log --out figures/eval

# With augmented test set label
python -m rl_captcha.scripts.plot_eval --log logs/eval_all.log --out figures/eval_aug --augmented
```

### Online learning
```powershell
python -m rl_captcha.scripts.plot_online --log online_training.log --out figures/
```

### Plot Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--log` | *(required)* | Path to log file |
| `--logs` | *(required for comparison)* | Training logs as `name=path` pairs |
| `--out` | `figures` | Output directory |
| `--format` | `png` | Output format: `png`, `pdf`, or `svg` |
| `--augmented` | off | Label plots as "Augmented Test Set" (plot_eval only) |

---

## Bot Tier System

| Tier | Name | Bot Types |
|------|------|-----------|
| 1 | Commodity | linear, tabber, speedrun |
| 2 | Careful Automation | scripted, stealth, slow, erratic, replay |
| 3 | Semi-Automated | semi_auto |
| 4 | Trace-Conditioned | trace_conditioned |

---

## Live Integration

The trained agent is loaded by `TicketMonarch/backend/agent_service.py`. Set `RL_ALGORITHM` env var to select the algorithm (`ppo`, `dg`, or `soft_ppo`).
