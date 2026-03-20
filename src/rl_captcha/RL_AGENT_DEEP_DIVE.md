# RL Agent Deep Dive — Complete Technical Reference

This document walks through every component of the RL CAPTCHA agent codebase in detail: what each piece does, why it's designed that way, and how the pieces connect.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Pipeline — `data/loader.py`](#2-data-pipeline)
3. [Environment — `environment/event_env.py`](#3-environment)
4. [Neural Network — `agent/lstm_networks.py`](#4-neural-network)
5. [Rollout Buffer — `agent/rollout_buffer.py`](#5-rollout-buffer)
6. [PPO Agent — `agent/ppo_lstm.py`](#6-ppo-agent)
7. [DG Agent — `agent/dg_lstm.py`](#7-dg-agent)
8. [Training Script — `scripts/train_ppo.py`](#8-training-script)
9. [The Full Training Loop (End-to-End)](#9-the-full-training-loop)
10. [Configuration Reference — `config.py`](#10-configuration-reference)

---

## 1. High-Level Architecture

```
                    ┌──────────────────────────────────┐
                    │         DATA PIPELINE            │
                    │  loader.py: Session dataclass     │
                    │  JSON/CSV/MySQL → unified format  │
                    └──────────┬───────────────────────┘
                               │ list[Session]
                               ▼
                    ┌──────────────────────────────────┐
                    │         ENVIRONMENT              │
                    │  event_env.py: EventEnv           │
                    │  EventEncoder: raw → 26-dim vec   │
                    │  Windowing: 30 events per window  │
                    │  Action masking: 2-phase episodes  │
                    │  Bot data augmentation            │
                    └──────────┬───────────────────────┘
                               │ obs, reward, done, info
                               ▼
           ┌───────────────────────────────────────────────┐
           │              AGENT (PPO or DG)                │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  LSTMActorCritic (lstm_networks.py)     │  │
           │  │  LSTM(26→256, 2 layers) + dropout       │  │
           │  │  Actor: 256→128→64→7 (action logits)    │  │
           │  │  Critic: 256→128→64→1 (state value)     │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  RolloutBuffer (rollout_buffer.py)      │  │
           │  │  Stores obs, actions, rewards, dones     │  │
           │  │  Computes GAE advantages + returns       │  │
           │  │  Tracks LSTM hidden states per episode   │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  PPOLSTM (ppo_lstm.py)                  │  │
           │  │  select_action(): rollout collection     │  │
           │  │  update(): clipped surrogate + clipped V │  │
           │  │  save()/load(): checkpoint management    │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  DGLSTM (dg_lstm.py) — extends PPOLSTM  │  │
           │  │  update(): delight-gated policy loss     │  │
           │  │  Everything else inherited from PPO      │  │
           │  └─────────────────────────────────────────┘  │
           └───────────────────────────────────────────────┘
```

**The core idea:** We treat bot detection as a sequential decision problem. Instead of a classifier that sees the whole session at once, an RL agent watches user behavior unfold window-by-window and decides what to do at the end — allow, block, or challenge with a puzzle. This lets the system adapt its response based on confidence level (easy puzzle for borderline cases, hard puzzle or block for obvious bots).

---

## 2. Data Pipeline

**File:** `data/loader.py`

### Session Dataclass

Every data source (MySQL, Chrome extension, JSON files) gets normalized into one format:

```python
@dataclass
class Session:
    session_id: str
    label: int | None     # 1 = human, 0 = bot, None = unlabeled
    mouse: list[dict]     # [{"x": 100, "y": 200, "t": 1234}, ...]
    clicks: list[dict]    # [{"x": 100, "y": 200, "t": 1234, "target": {...}}, ...]
    keystrokes: list[dict] # [{"type": "down", "key": "a", "t": 1234}, ...]
    scroll: list[dict]    # [{"dy": -100, "t": 1234}, ...]
    metadata: dict
```

### Loading Functions

| Function | Source | Format |
|----------|--------|--------|
| `load_from_directory(path)` | `data/human/` + `data/bot/` | Auto-detect JSON format |
| `load_from_json(path)` | Single JSON file | Chrome extension export |
| `load_from_csv(path)` | CSV export | Webapp tracking CSV |
| `load_from_mysql(config)` | MySQL database | TicketMonarch user_sessions table |

`load_from_directory()` is the primary loader used during training. It scans `data/human/*.json` (labeled as human) and `data/bot/*.json` (labeled as bot), auto-detecting three JSON formats:

1. **Chrome extension format:** `{ "<sessionId>": { "segments": [...] } }` — dict keyed by session IDs, each with a `segments` array of telemetry batches.
2. **Live-confirm format:** `{ "sessionId": "...", "segments": [...] }` — single session with segments array.
3. **Flat array format:** `[{ "session_id": "...", "mouse": [...], ... }]` — list of session objects.

### Stratified Splitting

```python
split_sessions(sessions, train=0.70, val=0.15, test=0.15, seed=42)
```

Splits human and bot sessions independently so each set maintains the same label proportions. Uses a fixed seed for reproducibility — critical so validation/test sets stay the same across training runs.

### Session Slicing

`slice_session(session, t_start, t_end)` extracts events within a time range. Handles keystroke edge cases: if a key-down event is in range but its matching key-up is slightly past `t_end`, the key-up is still included (up to 2000ms grace period) to preserve hold duration features.

---

## 3. Environment

**File:** `environment/event_env.py`

### EventEncoder — Raw Events to Feature Vectors

The encoder converts a window of raw events (mouse moves, clicks, keystrokes, scrolls) into a **26-dimensional** normalized feature vector. This is the observation the agent sees at each timestep.

**Why 26 dimensions?** Each feature captures a behavioral pattern that distinguishes humans from bots:

| Dims | Feature | Why It Matters |
|------|---------|---------------|
| 0-3 | **Event type ratios** (mouse/click/key/scroll) | Bots often have unnatural event mixes (e.g., clicks only, no mouse movement) |
| 4 | **Mean mouse speed** (px/s, normalized by 5000) | Bots may move at unnaturally constant speeds |
| 5 | **Mouse speed variance** | **Key discriminator** — bots have near-zero variance because programmatic mouse moves are perfectly uniform |
| 6 | **Mean mouse acceleration** | Humans have jerky, variable acceleration; bots are smooth |
| 7 | **Path curvature** (path length / displacement) | Humans curve around targets; bots move in straight lines (curvature ≈ 1.0) |
| 8 | **Mean inter-event dt** (log-normalized) | Log normalization handles the heavy-tailed distribution of human timing |
| 9 | **Timing variance** | **Key discriminator** — bots fire events at metronomic intervals |
| 10 | **Minimum dt** | Superhuman reaction times (< 10ms) indicate a bot |
| 11-12 | **Click timing** (mean interval, variance) | Bots click at regular intervals; humans are irregular |
| 13-14 | **Keystroke hold duration** (mean, variance) | Bots press and release keys identically each time |
| 15-16 | **Key-press intervals** (mean, variance) | Typing rhythm — bots type at constant speed |
| 17 | **Total scroll distance** | Bots rarely scroll organically |
| 18 | **Scroll direction changes** | Humans scroll up/down searching; bots scroll in one direction |
| 19-20 | **Unique x/y positions** (binned to 10px) | Bots visit fewer distinct screen locations |
| 21-22 | **x/y range** (max-min spread) | How much of the screen was used |
| 23 | **Interactive click ratio** | Fraction of clicks on `INPUT`, `BUTTON`, `A`, `SELECT`, `TEXTAREA` — bots may click random areas |
| 24 | **Window duration** (log-normalized) | Time span of the window |
| 25 | **Event count / window size** | How full the window is (padding detection) |

**Normalization strategy:**
- Ratios are naturally [0, 1]
- Speeds are clipped to `max_speed` (5000 px/s) and divided
- Timings use `log1p` normalization to handle heavy tails (a 5ms event and a 5000ms event should both be representable)
- Spatial features are normalized by screen resolution (1920×1080)
- Variances are clipped to prevent explosion from outliers

### EventEnv — Gymnasium Environment

The environment wraps sessions into a standard Gymnasium interface.

#### Windowing

Each session's raw events are merged into a single timeline (sorted by timestamp), then split into **overlapping windows** of 30 events each with 50% overlap (stride = 15). This means:

- A session with 90 events → ~5 windows
- Each window overlaps half its events with the next
- Windows with fewer than `min_events` (10) are dropped

#### Two-Phase Action Masking

This is the most important design decision. Episodes have two phases enforced by **action masking** — the agent literally cannot choose invalid actions because their logits are set to `-inf`:

**Observation phase** (all non-final windows):
- Valid: `continue` (0), `deploy_honeypot` (1)
- Invalid: all terminal actions (2-6)

**Decision phase** (final window only):
- Valid: `easy_puzzle` (2), `medium_puzzle` (3), `hard_puzzle` (4), `allow` (5), `block` (6)
- Invalid: `continue` (0), `deploy_honeypot` (1)

**Why?** Without this, the agent learns a shortcut: decide immediately on window 1 without looking at the full session. By forcing observation through all windows before deciding, the LSTM has time to accumulate evidence. The decision quality depends on the full behavioral trajectory, not a single snapshot.

#### Reward Structure

| Outcome | Reward | Rationale |
|---------|--------|-----------|
| Correctly allow human | +0.5 | Good but less rewarding than catching a bot |
| Correctly block/puzzle bot | +1.0 | Primary objective — catching bots |
| False positive (block/puzzle human) | -1.0 | Worst outcome — losing real users |
| False negative (allow bot) | -0.8 | Bad but slightly less catastrophic than FP |
| Bot fails puzzle | +1.0 | Caught the bot via challenge |
| Bot passes puzzle | -0.4 | `penalty_false_negative * 0.5` — bot slipped through but at least we tried |
| Human passes puzzle | -cost | Minor UX friction (0.10/0.30/0.50 for easy/medium/hard) |
| Human fails puzzle | -1.0 | False positive — human blocked by our puzzle |
| Honeypot catches bot | +0.3 | Information bonus (delivered next step) |
| Per-window continue | -0.001 | Tiny time pressure to prevent infinite observation |

**The asymmetry is intentional:** False positives (blocking humans) are penalized more than false negatives (allowing bots) because losing real customers is worse for business than letting an occasional bot through. The puzzle system gives the agent a middle ground — uncertain? Challenge with a puzzle rather than outright blocking.

#### Honeypot Mechanics

Honeypots are invisible traps (hidden form fields, invisible links) that bots may interact with but humans shouldn't. When deployed:
- Bots trigger 60% of the time
- Humans trigger 1% of the time (accidental interaction)
- Max 2 honeypots per session
- If a bot triggers, the agent gets +0.3 bonus on the **next** step (delayed to simulate real-world detection latency)

#### Bot Data Augmentation

During training only, bot sessions are stochastically perturbed (50% chance per episode) to prevent the agent from overfitting to trivially separable features:

1. **Speed warp** (0.7x–1.4x): Stretches/compresses all timestamps relative to the first event. Makes the bot's overall speed variable across episodes.
2. **Timing jitter** (std=30ms): Gaussian noise on each timestamp. Breaks the perfectly regular intervals that make bots obvious.
3. **Position noise** (std=15px): Gaussian noise on x/y coordinates. Prevents memorizing exact mouse paths.

After perturbation, events are re-sorted by timestamp (jitter may have swapped neighbors). Human sessions are **never** augmented — we want the agent to learn real human patterns, not noisy versions.

#### Balanced Sampling

`reset()` samples sessions 50/50 human/bot regardless of dataset class balance. This prevents the agent from learning a degenerate "always allow" policy if humans outnumber bots in training data.

---

## 4. Neural Network

**File:** `agent/lstm_networks.py`

### LSTMActorCritic

```
Input (26-dim) → LSTM(256 hidden, 2 layers, dropout 0.1) → Actor head → 7 logits
                                                          → Critic head → 1 value
```

#### LSTM Layer

```python
nn.LSTM(input_size=26, hidden_size=256, num_layers=2, batch_first=True)
```

- **Why LSTM?** The agent needs to remember patterns across windows. A feedforward network would see each window independently and lose temporal context (e.g., "speed variance was low in windows 1-3 but spiked in window 4" — only an RNN can detect this pattern).
- **Why 2 layers?** Stacked LSTMs learn hierarchical temporal features: layer 1 captures local patterns (within-window dynamics), layer 2 captures global patterns (cross-window trends).
- **`batch_first=True`**: Input shape is `(batch, sequence, features)` rather than `(sequence, batch, features)`.
- **Dropout 0.1**: Applied between LSTM layers (only active during training). Prevents the upper LSTM layer from co-adapting to specific lower-layer representations.

#### Hidden State

```python
def init_hidden(batch_size=1, device="cpu"):
    h = zeros(num_layers, batch_size, hidden_size)  # hidden state
    c = zeros(num_layers, batch_size, hidden_size)  # cell state
    return (h, c)
```

The LSTM maintains two tensors:
- **h (hidden state)**: The "output" memory — what the network currently "thinks"
- **c (cell state)**: The "long-term" memory — information the network has explicitly chosen to remember via its forget/input gates

Both are initialized to zero at the start of each episode. Shape is `(num_layers, batch, hidden)` — each layer has its own hidden state.

#### Actor Head (Policy)

```python
nn.Sequential(
    nn.Linear(256, 128), nn.Tanh(),
    nn.Linear(128, 64),  nn.Tanh(),
    nn.Linear(64, 7),    # raw logits, NOT probabilities
)
```

Outputs **raw logits** for 7 actions. These are converted to probabilities externally via softmax. Using Tanh activations (not ReLU) because RL policy networks benefit from bounded, smooth gradients — ReLU's dead neurons are problematic when the policy needs to explore all actions.

#### Critic Head (Value Function)

```python
nn.Sequential(
    nn.Linear(256, 128), nn.Tanh(),
    nn.Linear(128, 64),  nn.Tanh(),
    nn.Linear(64, 1),    # scalar state value V(s)
)
```

Outputs a single scalar: the estimated total future reward from the current state. This is used to compute advantages (how much better an action was than expected).

#### Action Masking

```python
if action_mask is not None:
    logits = logits.masked_fill(action_mask == 0, float("-inf"))
```

Invalid actions get `-inf` logits. After softmax, `-inf` → 0 probability. This is **hard masking** — the agent literally cannot select invalid actions. This is more reliable than soft penalties and doesn't waste network capacity learning "don't pick action 5 on non-final windows."

#### Forward Pass Shape Handling

The network handles both single-step inference (during rollout) and multi-step sequences (during training):

```python
# Single step: x is (batch, features) → unsqueeze to (batch, 1, features)
# Multi-step: x is (batch, seq_len, features) → pass through directly
```

This avoids needing separate forward methods for rollout vs. training.

---

## 5. Rollout Buffer

**File:** `agent/rollout_buffer.py`

### Purpose

The rollout buffer is a fixed-capacity array that stores transitions collected during one rollout period (4096 steps by default). It's the bridge between environment interaction and policy updates.

### Storage Layout

```
observations:  (4096, 26)  float32  — windowed feature vectors
actions:       (4096,)     int64    — action indices 0-6
action_masks:  (4096, 7)   float32  — valid action masks
rewards:       (4096,)     float32  — immediate rewards
dones:         (4096,)     float32  — 1.0 if episode ended
log_probs:     (4096,)     float32  — log π(action) at collection time
values:        (4096,)     float32  — V(s) estimates at collection time
advantages:    (4096,)     float32  — GAE advantages (computed after rollout)
returns:       (4096,)     float32  — discounted returns (computed after rollout)
```

All arrays are pre-allocated at initialization for efficiency. A pointer `ptr` tracks the next write position.

### LSTM Hidden State Tracking

```python
episode_starts: list[(step_index, h_0, c_0)]
```

Every time an episode starts, the LSTM's hidden state `(h, c)` is snapshot and stored. During PPO update, each episode segment is replayed through the LSTM starting from its stored hidden state. This is necessary because the LSTM hidden state depends on the entire history — you can't just replay from arbitrary points.

### GAE (Generalized Advantage Estimation)

```python
def compute_gae(last_value, gamma=0.99, gae_lambda=0.95):
```

GAE computes advantages that balance bias and variance:

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)     # TD error
A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...     # GAE
```

- **`gamma` (0.99)**: Discount factor. How much the agent cares about future rewards. 0.99 means rewards 100 steps away are worth ~37% of immediate rewards.
- **`gae_lambda` (0.95)**: Bias-variance tradeoff. λ=0 gives pure TD (low variance, high bias). λ=1 gives Monte Carlo returns (high variance, low bias). 0.95 is a standard compromise.
- **`last_value`**: Bootstrap — the value estimate of the final observation in the rollout (for incomplete episodes at the buffer boundary).
- **Episode boundaries**: When `done[t] = True`, both the next_value term and the GAE continuation are zeroed via `next_non_terminal = 0`. This prevents reward signal from leaking across episodes.

After computing advantages, they're **normalized across the entire rollout**:
```python
advantages = (advantages - mean) / (std + 1e-8)
```

This prevents scale issues when advantages vary widely between rollouts. The normalization happens at the buffer level (across all 4096 steps) rather than per-episode, which is important for short episodes that would have meaningless statistics.

### Episode Segments

```python
def get_episode_segments() -> list[dict]:
```

Converts the flat buffer into a list of episode segments, each containing:
- All transitions for one episode (or a partial episode at rollout boundaries)
- The initial LSTM hidden state `(h0, c0)` for that episode
- Old log probs and old values for importance ratio and clipped value loss computation

These segments are processed sequentially through the LSTM during training — you cannot shuffle steps within a segment because the LSTM state depends on order. But segments themselves can be shuffled between epochs.

---

## 6. PPO Agent

**File:** `agent/ppo_lstm.py`

### Proximal Policy Optimization (PPO) — The Algorithm

PPO is a policy gradient algorithm that prevents destructively large policy updates. The core idea: clip the probability ratio between the new and old policy so the update can't change the policy too much in one step.

### Action Selection (Rollout)

```python
def select_action(obs, action_mask=None, deterministic=False):
```

1. Switch network to **eval mode** (disables dropout)
2. Convert numpy observation to tensor, add batch dimension
3. Forward pass through LSTM with current hidden state
4. Apply action mask to logits
5. Softmax → Categorical distribution
6. Sample action (or argmax if deterministic)
7. Return (action, log_prob, value)
8. Switch network back to **train mode**

The `log_prob` and `value` are stored in the buffer for later use in the PPO update. The hidden state is updated in-place (`self._hidden = new_hidden`) so the next call continues from where this one left off.

### PPO Update

```python
def update() -> dict[str, float]:
```

This is where learning happens. For each of `num_epochs` (6) passes over the data:

1. **Shuffle segments** between epochs to reduce correlation
2. For each episode segment:

#### Policy Loss (Clipped Surrogate)

```python
ratio = exp(new_log_prob - old_log_prob)  # π_new(a|s) / π_old(a|s)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2)
```

**What this does:** The ratio measures how much the policy changed for each action. If the ratio is far from 1.0, the policy has shifted significantly. The clamp prevents the ratio from exceeding `[1-ε, 1+ε]` (ε=0.2), which limits the update magnitude.

**Why min?** The min creates a pessimistic bound:
- If advantage > 0 (good action): we want ratio to increase (higher probability), but clip prevents it from going above 1.2
- If advantage < 0 (bad action): we want ratio to decrease, but clip prevents it from going below 0.8

This asymmetric clipping is PPO's key innovation — it gets most of the benefits of trust region methods (like TRPO) without the computational cost of computing KL divergence constraints.

#### Value Loss (Clipped)

```python
v_clipped = old_values + clamp(values - old_values, -ε, +ε)
value_loss = 0.5 * max((values - returns)², (v_clipped - returns)²)
```

Same clipping idea applied to the value function. Prevents the critic from making large jumps that could destabilize advantage estimation in subsequent rollouts.

#### Entropy Bonus

```python
entropy = dist.entropy().mean()
loss -= entropy_coeff * entropy  # subtracted because we minimize loss
```

Entropy measures how "spread out" the action distribution is. Higher entropy = more exploration. The entropy bonus prevents the policy from collapsing to always picking the same action too early. `entropy_coeff=0.005` is small — just enough to maintain exploration without overriding the reward signal.

#### Total Loss

```python
loss = policy_loss + 0.5 * value_loss - 0.005 * entropy
```

A single loss combines all three objectives. The actor and critic share LSTM weights, so updating both through one loss is efficient and allows shared feature learning.

#### Gradient Clipping

```python
clip_grad_norm_(network.parameters(), max_grad_norm=0.5)
```

Clips the total gradient norm to 0.5 to prevent exploding gradients (common with LSTMs). This is different from PPO's ratio clipping — gradient clipping is a numerical stability measure, not a policy constraint.

### Save / Load

Checkpoints store:
- Network state dict (all weights)
- Optimizer state dict (momentum terms, adaptive learning rates)
- Configuration used for training (for reproducibility)

---

## 7. DG Agent

**File:** `agent/dg_lstm.py`

### Delightful Policy Gradients — The Algorithm

DG modifies the policy gradient by gating each gradient term with a sigmoid of "delight" — the product of advantage and surprisal.

### How It Differs from PPO

The only difference is in `update()`. Everything else (network, buffer, rollout collection, value loss, entropy bonus, save/load) is inherited from `PPOLSTM`.

#### Delight Computation (Algorithm 1 from arXiv:2603.14608)

```python
surprisal = -new_log_probs           # ℓ_t = −log π_θ(A_t|H_t) — CURRENT policy
delight = advantages * surprisal     # χ_t = U_t · ℓ_t
gate = sigmoid(delight / temperature) # w_t = σ(χ_t / η)
```

**Critical detail:** The paper defines surprisal using the **current** policy `π_θ`, not the old rollout policy `π_old`. This means the gate adapts as the policy updates — actions that become more likely during training have lower surprisal, naturally reducing their gate weight.

**The four cases:**

| | High probability action | Low probability action |
|---|---|---|
| **Positive advantage** (good) | Low delight → gate ≈ 0.5 | **High delight → gate ≈ 1.0** |
| **Negative advantage** (bad) | Low negative delight → gate ≈ 0.5 | **Very negative delight → gate ≈ 0.0** |

The sigmoid gates:
- **Rare successes get amplified** (gate → 1): The agent tried something unlikely and it worked. This is the most informative learning signal.
- **Rare failures get suppressed** (gate → 0): The agent tried something unlikely and it failed. The policy was already unlikely to pick this — no need to let it dominate the gradient.
- **Expected outcomes pass through** (gate ≈ 0.5): Whether good or bad, expected outcomes get moderate weight.

#### DG Policy Loss

```python
dg_loss = -(gate.detach() * advantages * new_log_probs).mean()
```

This is a weighted REINFORCE-style gradient where each action's contribution is scaled by its delight gate. The gate is **detached** from the computation graph because the paper treats `w_t` as a per-sample scalar weight on the gradient direction, not as part of the objective to differentiate through. Without detach, gradients would flow through the gate (via surprisal → log_prob → θ), optimizing a different objective than what the paper defines.

#### PPO Blending (Optional)

```python
policy_loss = (1 - w) * dg_loss + w * ppo_loss  # w = dg_baseline_weight
```

Per the paper, DG is a **standalone drop-in replacement** for the policy gradient — it does NOT use importance ratios or PPO clipping. The default `dg_baseline_weight=0.0` gives pure DG as described in the paper. However, the blend option exists for practical experimentation — you can set `dg_baseline_weight > 0` to mix DG with PPO for added stability.

#### DG-Specific Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `dg_temperature` | 1.0 | Controls sigmoid sharpness (η in paper). Lower = more aggressive gating. Higher = flatter (closer to uniform weighting). Paper uses 1.0 across all experiments. |
| `dg_baseline_weight` | 0.0 | Blend with PPO. 0 = pure DG (paper default), 1 = pure PPO. Paper doesn't blend — set > 0 only for practical stability experiments. |

#### Additional Metrics

DG logs two extra metrics beyond PPO:
- `delight_mean`: Average delight value. Should be near 0 if the policy is well-calibrated.
- `gate_mean`: Average gate activation. Should be near 0.5 on average — if it's consistently near 0 or 1, the temperature needs adjustment.

### Why DG Suits This Project

1. **Class imbalance across contexts**: Easy human sessions and hard bot sessions. PPO wastes gradient budget on easy sessions the agent already handles. DG reallocates to the hard cases.
2. **Rare but informative actions**: Honeypot deployment (action 1) is rare and carries the most information about bot status. DG amplifies the signal from successful honeypot deployments.
3. **Discrete categorical policy**: DG's surprisal = -log π maps directly to the Categorical distribution.

---

## 8. Training Script

**File:** `scripts/train_ppo.py`

### Data Loading

```python
sessions = load_from_directory(args.data_dir)
train, val, test = split_sessions(sessions, 0.70, 0.15, 0.15, seed=42)
```

Loads all JSON files from `data/human/` and `data/bot/`, then does a stratified 70/15/15 split. The test set is **never** used during training — it's held out for final evaluation only.

### Environment Setup

Two separate environments are created:
- **Training env**: Augmentation ON (bot sessions get stochastically perturbed)
- **Validation env**: Augmentation OFF (evaluate on real data distribution)

### Rollout Collection

```python
def _collect_rollout(env, agent, num_steps=4096):
```

Runs the agent in the environment for `num_steps` transitions:

1. Reset environment, get first observation
2. Agent selects action via `select_action(obs, action_mask)`
3. Environment returns `(next_obs, reward, terminated, truncated, info)`
4. Transition stored in buffer
5. If episode ends, reset and start a new one
6. After `num_steps`, bootstrap the final value for GAE

Returns statistics: episode rewards, lengths, outcome distribution.

### Training Loop

```
while total_steps < total_timesteps:
    1. Collect rollout (4096 steps)
    2. Compute GAE advantages
    3. Run PPO update (6 epochs over the rollout)
    4. Log statistics every `log_interval` rollouts
    5. Save checkpoint + validate every `save_interval` rollouts
```

### Validation

```python
def _quick_validate(env, agent, num_episodes=100):
```

Runs the agent **deterministically** (argmax over action probabilities) on validation sessions. Returns accuracy: fraction of episodes with correct outcomes (`correct_block`, `bot_blocked_puzzle`, `correct_allow`).

---

## 9. The Full Training Loop (End-to-End)

Here's what happens during one complete training iteration, step by step:

### Phase 1: Rollout Collection (4096 steps)

```
for step in range(4096):
    1. Agent sees obs (26-dim vector of current window)
    2. LSTM processes obs with current hidden state
    3. Actor outputs 7 logits → softmax → sample action
    4. If non-final window: action is continue(0) or honeypot(1)
       If final window: action is puzzle/allow/block (2-6)
    5. Environment returns reward and advances to next window
    6. (obs, action, reward, done, log_prob, value, mask) → buffer
    7. If episode ends: reset env, reset LSTM hidden, start new episode
```

### Phase 2: GAE Computation

```
for t in reversed(range(4096)):
    δ_t = reward_t + γ · V(s_{t+1}) · (1-done_t) - V(s_t)
    A_t = δ_t + (γλ) · (1-done_t) · A_{t+1}
    return_t = A_t + V(s_t)

Normalize all 4096 advantages: A = (A - mean) / std
```

### Phase 3: PPO Update (6 epochs)

```
for epoch in range(6):
    shuffle(episode_segments)
    for segment in segments:
        1. Replay obs sequence through LSTM from stored h0, c0
        2. Get new logits, values, entropy
        3. Compute ratio = π_new(a|s) / π_old(a|s)
        4. Clip ratio to [0.8, 1.2]
        5. Policy loss = -min(ratio * A, clip(ratio) * A)
        6. Value loss = max((V - R)², (clip(V) - R)²)
        7. Total loss = policy + 0.5·value - 0.005·entropy
        8. Backward pass + gradient clip (norm ≤ 0.5)
        9. Optimizer step (Adam, lr=3e-4)
```

### Phase 4: Logging + Checkpointing

Every `save_interval` rollouts:
- Save checkpoint (network + optimizer state)
- Run 100 deterministic validation episodes
- Print accuracy

---

## 10. Configuration Reference

**File:** `config.py`

### EventEnvConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `event_dim` | 26 | Feature vector dimension |
| `mouse_subsample` | 5 | Keep every 5th mouse event (66Hz → ~13Hz) |
| `window_size` | 30 | Events per observation window |
| `min_events` | 10 | Skip sessions with fewer events |
| `reward_correct_block` | 1.0 | Reward for correctly blocking a bot |
| `reward_correct_allow` | 0.5 | Reward for correctly allowing a human |
| `penalty_false_positive` | -1.0 | Penalty for blocking/puzzling a human |
| `penalty_false_negative` | -0.8 | Penalty for allowing a bot through |
| `continue_penalty` | 0.001 | Per-window time pressure |
| `honeypot_info_bonus` | 0.3 | Bonus when honeypot catches a bot |
| `truncation_penalty` | -0.5 | Penalty if episode overflows windows |
| `max_honeypots` | 2 | Maximum honeypot deployments per episode |
| `augment` | True | Enable bot data augmentation |
| `augment_prob` | 0.5 | Probability of augmenting each bot episode |

### PPOConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda (bias-variance tradeoff) |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `value_loss_coeff` | 0.5 | Value loss weight in total loss |
| `entropy_coeff` | 0.005 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `lstm_hidden_size` | 256 | LSTM hidden units per layer |
| `lstm_num_layers` | 2 | Number of stacked LSTM layers |
| `rollout_steps` | 4096 | Transitions per rollout |
| `num_epochs` | 6 | PPO update epochs per rollout |
| `total_timesteps` | 500,000 | Total training steps |

### DGConfig (extends PPOConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dg_temperature` | 1.0 | Sigmoid temperature η (lower = sharper gating; paper uses 1.0) |
| `dg_baseline_weight` | 0.0 | PPO blend weight (0 = pure DG per paper, 1 = pure PPO) |
