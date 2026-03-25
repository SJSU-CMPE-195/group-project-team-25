# Ticket Monarch

Developed by Team 25 — San Jose State University

A mock concert ticket-booking web app that uses a PPO+LSTM reinforcement learning agent to detect bots in real time based on raw telemetry (mouse movements, clicks, keystrokes, scrolls).

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Node.js | 18+ |
| MySQL | 8.0+ |

## Project Structure

```
src/
├── TicketMonarch/          # Main web application
│   ├── backend/            # Flask API + agent inference
│   ├── frontend/           # React + Vite SPA
│   └── .env.example        # MySQL connection config template
├── rl_captcha/             # PPO+LSTM agent (training & evaluation)
├── bots/                   # Selenium & LLM bots for data collection
└── data/                   # Training data (human/ and bot/)
```

## Quick Start (Full Setup)

All commands assume you are in the `src/` directory.

### 1. Create and activate a virtual environment

**PowerShell (Windows):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install all Python dependencies

```bash
pip install -r TicketMonarch/backend/requirements.txt
pip install -r rl_captcha/requirements.txt
pip install -r bots/requirements.txt
```

If you plan to use the LLM bot, also run:
```bash
pip install langchain-anthropic
playwright install chromium
```

### 3. Configure the database

```bash
cp TicketMonarch/.env.example TicketMonarch/.env
```

Edit `TicketMonarch/.env` and set your MySQL password:
```
MYSQL_HOST=localhost
MYSQL_DATABASE=ticketmonarch_db
MYSQL_USER=root
MYSQL_PASSWORD=<your_password>
MYSQL_PORT=3306
```

Then create the database and tables:
```bash
python TicketMonarch/backend/setup_mysql.py
```

### 4. Install frontend dependencies

```bash
cd TicketMonarch/frontend
npm install
cd ../..
```

### 5. Run the application

Open **two terminals** (activate the venv in each if running Python):

```bash
# Terminal 1 — Backend (http://localhost:5000)
python TicketMonarch/backend/app.py

# Terminal 2 — Frontend (http://localhost:3000)
cd TicketMonarch/frontend
npm run dev
```

Open **http://localhost:3000** in your browser. Vite proxies `/api/*` requests to Flask automatically.

## User Flow

1. **Home** (`/`) — Browse concerts and select one
2. **Seat Selection** (`/seats/:id`) — Pick seats from an interactive layout
3. **Checkout** (`/checkout`) — Fill the payment form
   - Rolling inference polls every 3 seconds and can request honeypot deployment
   - On "Purchase": telemetry is force-flushed and the agent evaluates the full session through `/api/agent/evaluate`
   - Policy outputs map directly to `allow`, `block`, or `easy/medium/hard` puzzle
   - Honeypot triggered → instant hard puzzle
4. **Confirmation** (`/confirmation`) — Order confirmed, session sent for online RL update

## Dev Dashboard

Open **http://localhost:3000/dev** in a separate tab.

- **Live Monitor:** Auto-detects the active session, polls every 1 second showing real-time event counts and rolling bot probability.
- **Analyze Session:** Full agent analysis on any session — decision banner, action probability bars, per-event timeline, LSTM hidden-state heatmap.

## Running the Bots

Bots drive a real Chrome browser through the full booking flow to generate labeled training data. The frontend's built-in `tracking.js` captures all telemetry automatically, so the bots just need to interact with the site.

### Prerequisites

Before running any bot, make sure:

1. **TicketMonarch is running** — backend (`python app.py`) + frontend (`npm run dev`)
2. **The venv is activated** with bot dependencies installed (see [Quick Start](#quick-start-full-setup))
3. **Chrome is installed** on your machine

### Bot Dependencies (already installed if you followed Quick Start)

```bash
# Core bot packages
pip install -r bots/requirements.txt

# Additional packages for the LLM bot
pip install langchain-anthropic
playwright install chromium
```

---

### Selenium Bot

Three behavior profiles that produce progressively harder-to-detect bot data:

| Profile | Mouse Movement | Typing | Scrolling | Detection Difficulty |
|---------|---------------|--------|-----------|---------------------|
| `linear` | Straight-line + light idle fidgeting | Uniform 20 ms intervals | None | Tier 1 — Easy |
| `scripted` | Bezier curves + light idle fidgeting | Variable timing, burst typing, thinking pauses | Momentum-based random scrolls | Tier 2 — Medium |
| `stealth` | Human-like Bezier with micro-jitter | Lognormal timing, burst typing | Organic momentum scrolls | Tier 2 — Hard |
| `semi_auto` | Mix of real human + bot actions | Strategy-dependent | Strategy-dependent | Tier 3 |
| `trace_conditioned` | Replays perturbed human traces | Human-profiled intervals | Replayed from source | Tier 4 |

All bot types include small idle movements between actions, but these are intentionally limited so bot traces do not dwarf human checkout sessions.

#### Commands

```bash
# Linear bot — robotic straight-line movement, uniform typing
python bots/selenium_bot.py --runs 5 --type linear

# Scripted bot — Bezier curves, varied timing, scrolling
python bots/selenium_bot.py --runs 5 --type scripted

# Mixed mode — weighted random across all bot types
python -m bots.selenium_bot --runs 50 --type mixed --skip-honeypot
```

#### All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | `3` | Number of bot sessions to run |
| `--type` | `scripted` | Bot behavior: `linear`, `scripted`, `stealth`, `slow`, `erratic`, `speedrun`, `tabber`, `replay`, `semi_auto`, `trace_conditioned`, or `mixed` |
| `--replay-source` | — | Path to a human session JSON file (required for `replay` type) |
| `--pause-between` | `2.0` | Seconds to wait between runs |

#### How It Works (Step-by-Step)

1. Opens Chrome (telemetry is captured by the site's built-in `tracking.js`)
2. Navigates through the full booking flow for each run:
   - **Home** → picks a random concert
   - **Seat Selection** → picks a random section, clicks Continue
   - **Checkout** → fills all form fields with fake identity data (8 built-in personas)
   - **Purchase** → submits the order
   - **Challenge handling** → if a CAPTCHA challenge appears, the bot attempts to interact (slider, canvas text, timed click — up to 3 retries)
4. After each run, the bot pulls telemetry from the backend API, saves it to `data/bot/`, and confirms the session as a bot (`true_label=0`) to trigger an online RL update

#### Behavior Details

**Mouse movement (scripted profile):**
- Bezier curves with 10–25 steps per movement
- Ease-in-out timing (slower at start/end, faster in middle)
- Micro-jitter: Gaussian noise of ~1.5 px (x-axis) and ~1.0 px (y-axis)

**Typing (scripted profile):**
- Burst typing: 2–3 chars quickly (30% probability)
- Variable inter-key delays: lognormal distribution, 0.03–0.25 s
- Thinking pauses: 0.3–0.8 s (8% probability)
- 0.1–0.3 s delay before starting to type each field

**Scrolling (scripted profile):**
- Momentum-based: 3–8 steps per gesture with decreasing scroll amounts
- Pauses between gestures: 0.5–1.5 s

**Replay mode:**
- Replays mouse coordinates from a recorded JSON file
- Adds Gaussian noise: ~6 px (x-axis), ~4 px (y-axis)
- Timing jitter applied to deltas
- 10% chance of micro-corrections per point
- Scroll events replayed with 0.8–1.2x variance

---

### LLM Bot

Uses [browser-use](https://github.com/browser-use/browser-use) to give an LLM (Claude or GPT-4o) full autonomous control of a Chrome browser. The LLM reads the page, decides where to click, what to type, and completes the booking flow on its own — producing the most realistic bot behavior.

#### Environment Variables

Set your API key before running:

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# or for OpenAI:
$env:OPENAI_API_KEY = "sk-..."
```

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or for OpenAI:
export OPENAI_API_KEY=sk-...
```

#### Commands

```bash
# Run with Claude (default)
python bots/llm_bot.py --runs 3 --provider anthropic

# Run with GPT-4o
python bots/llm_bot.py --runs 3 --provider openai

# Custom task prompt
python bots/llm_bot.py --task "Go to localhost:3000, browse concerts, pick the cheapest tickets"

# Enable DOM event injection (alternates injection on/off per run for data diversity)
python bots/llm_bot.py --runs 4 --inject-events
```

#### All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | `1` | Number of bot sessions to run |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `--pause-between` | `3.0` | Seconds to wait between runs |
| `--task` | *(full booking flow)* | Custom instruction prompt for the LLM |
| `--inject-events` | off | Enable DOM event injection script (alternates on/off per run) |

#### How It Works (Step-by-Step)

1. Opens a visible Chrome window (not headless) with `disable_security=True`
2. Optionally injects a DOM event-generation script via CDP that patches `element.click()`, `element.focus()`, `scrollTo/scrollBy`, and input events into real browser events — so `tracking.js` captures telemetry even from programmatic actions
3. The LLM autonomously navigates: browses concerts → selects seats → fills checkout → purchases
4. After each run, the script waits 8 seconds for `tracking.js` to flush, then:
   - Reads `tm_session_id` from the browser's sessionStorage via CDP
   - Falls back to diffing recent session IDs from the backend API
   - Pulls raw telemetry from `/api/session/raw/<sid>`
   - Saves the JSON to `data/bot/` with a UTC timestamp filename
   - Confirms the session as a bot (`true_label=0`) via `/api/agent/confirm`
5. When `--inject-events` is enabled, even-numbered runs get injection and odd runs do not, creating data diversity (sparse vs. richer telemetry)

#### LLM Models Used

| Provider | Model |
|----------|-------|
| Anthropic | `claude-sonnet-4-20250514` |
| OpenAI | `gpt-4o` |

---

### Telemetry Data Flow (All Bots)

```
Bot interacts with site
        │
        ▼
tracking.js captures events in browser ──► Backend stores in MySQL
                                                 │
                                                 ▼
                                        Bot pulls from /api/session/raw/<sid>
                                        and saves JSON to data/bot/
                                                 │
                                                 ▼
                                        Bot calls /api/agent/confirm
                                        with true_label=0 (bot)
                                                 │
                                                 ▼
                                        Online RL PPO update triggered
```

### Output Files

Bot telemetry is saved to `data/bot/` as JSON files:
- Selenium bot: `session_<session_id>.json`
- LLM bot: `llm_bot_<utc_timestamp>.json`

Each file contains the raw telemetry arrays (mouse movements, clicks, keystrokes, scroll events) pulled from the backend.

---

## Training the RL Agent

Training data lives in `data/human/` (label=1) and `data/bot/` (label=0).

Because telemetry capture semantics changed, recollect fresh data before training a new model. The previous JSON dataset was intentionally cleared.

### 1. Collect Data

**Human data:** Browse the site normally. Sessions are auto-saved to `data/human/` when confirmed via the `/api/agent/confirm` endpoint.

**Bot data:** Run the bots (see above). They automatically save to `data/bot/` and confirm sessions as bots.

### 2. Train

```bash
python -u -m rl_captcha.scripts.train_ppo \
    --data-dir data/ \
    --save-path rl_captcha/agent/checkpoints/ppo_run1 \
    --total-timesteps 500000 \
    --val-episodes 100 \
    2>&1 | Tee-Object -FilePath training.log
```

Checkpoint saved to `rl_captcha/agent/checkpoints/ppo_run1/`. Data is split 70/15/15 (train/val/test) with stratified sampling. Bot sessions are stochastically augmented during training (see [Data Augmentation](#data-augmentation)).

### 3. Evaluate

```bash
python -m rl_captcha.scripts.evaluate_ppo \
    --agent rl_captcha/agent/checkpoints/ppo_run1 \
    --data-dir data/ \
    --episodes 500 \
    --split test
```

Outputs confusion matrix, accuracy, precision, recall, F1 score, outcome distribution, and action distribution.

### 4. Visualize

```bash
# Plot training curves
python -m rl_captcha.scripts.plot_training --log training.log --out figures/

# Plot evaluation results
python -m rl_captcha.scripts.evaluate_ppo --agent ... --data-dir data/ 2>&1 | Tee-Object -FilePath eval.log
python -m rl_captcha.scripts.plot_eval --log eval.log --out figures/

# Plot online learning log
python -m rl_captcha.scripts.plot_online --log online_training.log --out figures/
```

#### Training CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/` | Root directory containing `human/` and `bot/` subdirectories |
| `--save-path` | `rl_captcha/agent/checkpoints/ppo_run1` | Where to save model checkpoints |
| `--total-timesteps` | `500000` (from PPOConfig) | Total environment steps to train for |
| `--log-interval` | `1` | Print stats every N rollouts |
| `--save-interval` | `10` | Save checkpoint + run validation every N rollouts |
| `--val-episodes` | `100` | Number of validation episodes per checkpoint |
| `--device` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |
| `--split-seed` | `42` | Random seed for train/val/test split |

#### Evaluation CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | *(required)* | Path to checkpoint directory |
| `--data-dir` | `data/` | Root data directory |
| `--episodes` | `500` | Number of episodes to evaluate |
| `--split` | `test` | Data split: `test`, `val`, `train`, or `all` |
| `--split-seed` | `42` | Random seed for split (must match training) |
| `--device` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |

#### Plot CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--log` | `training.log` (plot_training) / *required* (plot_eval) | Path to log file |
| `--out` | `figures` | Output directory for figures |
| `--format` | `png` | Output format: `png`, `pdf`, or `svg` |

---

## Data Augmentation

During training, **bot sessions** are stochastically augmented (50% probability per episode) to prevent the agent from relying on trivially separable features (e.g., zero mouse variance, perfectly regular timing). Human sessions are never augmented. Augmentation is disabled for the validation environment so val metrics reflect real data.

| Augmentation | Default | Effect |
|-------------|---------|--------|
| Position noise | std=15px | Gaussian jitter on x/y coordinates |
| Timing jitter | std=30ms | Gaussian noise on event timestamps |
| Speed warp | 0.7x-1.4x | Random time stretch/compress across session |

Configure in `rl_captcha/config.py` (`EventEnvConfig`):

```python
augment: bool = True              # enable/disable
augment_prob: float = 0.5         # probability per bot episode
aug_position_noise_std: float = 15.0
aug_timing_jitter_std: float = 30.0
aug_speed_warp_range: tuple = (0.7, 1.4)
```

---

## RL System Design: Policy, Actions, and Reward Function

This section documents the complete reinforcement learning system that powers bot detection. The agent is a PPO+LSTM model that processes raw telemetry events one-by-one and decides in real time whether a user is a human or a bot.

### High-Level Architecture

```
Raw telemetry events (mouse, click, keystroke, scroll)
        │
        ▼
Windowed Event Encoder (26-dimensional feature vector per window of 30 events)
        │  Features: speed mean/variance, path curvature, click timing,
        │  keystroke hold patterns, scroll behavior, spatial spread, etc.
        │
        ▼
LSTM (128 hidden, 1 layer) ── accumulates temporal evidence across windows
        │
        ├──► Actor head (128 → 128 → 64 → 7 logits) ──► action (policy)
        └──► Critic head (128 → 128 → 64 → 1 value)  ──► V(s) (value estimate)
```

The LSTM processes **windows of events** sequentially. Each window contains 30 events and is encoded into a 26-dimensional statistical feature vector capturing behavioral patterns (speed variance, timing regularity, path curvature, etc.) that single events cannot represent. At every timestep (window), the agent chooses one of 7 actions. Most of the time it chooses `continue` (keep watching). When it has seen enough evidence, it makes a terminal decision: allow, block, or deploy a puzzle challenge.

---

### State Representation (Observation Space)

Events are grouped into **windows of 30 events** (with 50% overlap) and each window is encoded into a **26-dimensional statistical feature vector**:

| Dimensions | Feature | Why it matters |
|-----------|---------|----------------|
| 0–3 | Event type ratios | Fraction of mouse/click/key/scroll events in the window |
| 4 | Mean mouse speed | Bots tend to have unnaturally constant speed |
| 5 | Mouse speed variance | **Key discriminator** — bots have near-zero variance |
| 6 | Mean mouse acceleration | Direction change rate |
| 7 | Path curvature | Ratio of path length to displacement — bots move in straight lines |
| 8 | Mean inter-event time | Average time between events |
| 9 | Inter-event time variance | **Key discriminator** — bots have uniform timing |
| 10 | Min inter-event time | Fastest event gap (bots can be inhumanly fast) |
| 11 | Mean click interval | Average time between consecutive clicks |
| 12 | Click interval variance | Bots click at regular intervals |
| 13 | Mean keystroke hold | Average key press duration |
| 14 | Keystroke hold variance | Bots have mechanical, uniform key holds |
| 15 | Mean key-press interval | Time between consecutive key presses |
| 16 | Key-press interval variance | Typing rhythm regularity |
| 17 | Total scroll distance | Absolute scroll amount in the window |
| 18 | Scroll direction changes | Number of up/down reversals |
| 19–20 | Unique x/y positions | Spatial diversity (binned to 10px) |
| 21–22 | X/Y range | Spatial spread normalized by screen size |
| 23 | Interactive click ratio | Fraction of clicks on INPUT/BUTTON/A/SELECT/TEXTAREA |
| 24 | Window duration | Time span of the window (log-normalized) |
| 25 | Event count ratio | Actual events / window_size |

**Preprocessing before encoding:**
- Mouse events are subsampled: every 5th event is kept (66 Hz → ~13 Hz)
- All event types are merged into a single timeline and sorted by timestamp
- Timeline is split into overlapping windows (stride = 15 events)
- Episodes are capped at 500 events maximum
- Sessions with fewer than 10 events are skipped

---

### Action Space (Policy)

The agent chooses from **7 discrete actions** at every timestep:

| Index | Action | Terminal? | Description | UX Cost |
|-------|--------|-----------|-------------|---------|
| 0 | `continue` | No | Keep observing, wait for more events | 0.0 |
| 1 | `deploy_honeypot` | No | Place invisible trap fields on the page | 0.01 |
| 2 | `easy_puzzle` | Yes | Show an easy CAPTCHA challenge | 0.1 |
| 3 | `medium_puzzle` | Yes | Show a medium CAPTCHA challenge | 0.3 |
| 4 | `hard_puzzle` | Yes | Show a hard CAPTCHA challenge | 0.5 |
| 5 | `allow` | Yes | Let the user through without challenge | 0.0 |
| 6 | `block` | Yes | Block the user entirely | 0.0 |

**Terminal actions** end the episode immediately. The agent receives a final reward based on whether its decision was correct.

**Non-terminal actions** let the episode continue to the next event. The agent can observe more behavior before committing to a decision.

**Honeypot mechanics:**
- Maximum 2 honeypots can be deployed per session
- Bots have a 60% chance of triggering a honeypot
- Humans have only a 1% chance of triggering (accidental interaction)
- If triggered by a bot, the agent receives a +0.3 information bonus
- Additional honeypots beyond the limit are silently ignored

---

### Reward Function

The reward structure is designed to balance two competing objectives: **catching bots** (security) and **not annoying humans** (user experience).

#### Terminal Rewards

| Scenario | Action | True Label | Reward | Rationale |
|----------|--------|------------|--------|-----------|
| Correctly allow human | `allow` | Human (1) | **+0.5** | Good, but lower than catching a bot — incentivizes discrimination over passivity |
| Correctly block bot | `block` | Bot (0) | **+1.0** | Highest reward — correctly identifying a bot is the primary goal |
| False positive (block human) | `block` | Human (1) | **-1.0** | Harshest penalty — blocking a real user is the worst outcome for UX |
| False negative (allow bot) | `allow` | Bot (0) | **-0.8** | Strong penalty — letting a bot through defeats the system's purpose |

#### Puzzle Rewards (Probabilistic Outcomes)

Puzzles are not deterministic — they have pass rates that depend on difficulty:

| Difficulty | Human Pass Rate | Bot Pass Rate |
|------------|----------------|---------------|
| Easy | 95% | 40% |
| Medium | 85% | 15% |
| Hard | 70% | 5% |

The reward for deploying a puzzle depends on the outcome:

- **Puzzle shown to a human:**
  - Reward = `penalty_false_positive × (1.0 - human_pass_rate)` minus the action cost
  - Easy puzzle on human: `-1.0 × 0.05 - 0.1 = -0.15` (mild penalty, most humans pass)
  - Hard puzzle on human: `-1.0 × 0.30 - 0.5 = -0.80` (steep penalty, many humans fail)

- **Puzzle shown to a bot:**
  - If bot passes (probabilistic): `penalty_false_negative × 0.5` (partial penalty)
  - If bot fails: `reward_correct_block = +1.0` (full reward for catching the bot)

This means harder puzzles are more effective at catching bots but risk more false-positive damage — the agent must learn when the evidence justifies the risk.

#### Non-Terminal Rewards

| Event | Reward | Purpose |
|-------|--------|---------|
| `continue` (each event) | **-0.001** | Tiny time pressure — discourages the agent from watching forever without deciding |
| `deploy_honeypot` | **-0.01** cost | Minimal UX friction for placing invisible traps |
| Honeypot triggered by bot | **+0.3** bonus | Rewards gathering information, deferred to next step |
| Episode truncated (hit 500 events without deciding) | **-0.5** | Penalizes indecision — the agent must commit before the session ends |

#### Reward Design Philosophy

- **Asymmetric penalties:** Blocking a human (-1.0) is penalized more harshly than allowing a bot (-0.8), reflecting that user experience damage is harder to recover from than a single bot slipping through
- **Graduated difficulty:** The agent can choose proportional responses — a slightly suspicious session gets an easy puzzle, not a full block
- **Information gathering:** Honeypots provide a low-cost way to gather evidence before committing to a terminal action
- **Time pressure:** The continue penalty prevents the agent from stalling, but at -0.001 per event it is gentle enough to allow thorough observation when needed

---

### Network Architecture

```
Input: (batch, seq_len, 26)   ← windowed feature vectors
              │
              ▼
     ┌──────────────────┐
     │  LSTM Layer       │
     │  input:  26       │
     │  hidden: 128      │
     │  layers: 1        │
     │  batch_first      │
     └────────┬─────────┘
              │ (batch, seq_len, 128)
              │
       ┌──────┴──────┐
       ▼              ▼
┌───────────────┐ ┌───────────────┐
│  Actor Head   │ │  Critic Head  │
│ Linear(128→128)│ │ Linear(128→128)│
│ Tanh          │ │ Tanh          │
│ Linear(128→64)│ │ Linear(128→64)│
│ Tanh          │ │ Tanh          │
│ Linear(64→7)  │ │ Linear(64→1)  │
└───────────────┘ └───────────────┘
       │              │
       ▼              ▼
  action logits   value V(s)
    (7-dim)        (scalar)
```

- **1-layer LSTM** (128 hidden units) processes windowed features sequentially
- **LSTM hidden state** (h ∈ ℝ¹²⁸) carries temporal context across windows
- **Cell state** (c ∈ ℝ¹²⁸) stores long-range behavioral patterns
- The actor outputs a probability distribution over 7 actions (softmax of logits)
- The critic estimates the expected future reward from the current state
- **Deeper heads** (3 linear layers each) allow more expressive decision boundaries

---

### PPO Training Algorithm

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Learning rate | 3×10⁻⁴ | Adam optimizer step size |
| Discount (γ) | 0.99 | How much future rewards are valued vs. immediate |
| GAE lambda (λ) | 0.95 | Bias-variance tradeoff for advantage estimation |
| PPO clip (ε) | 0.2 | Limits how far the new policy can deviate from the old |
| Value loss coefficient | 0.5 | Weight of critic loss in total loss |
| Entropy coefficient | 0.02 | Encourages exploration by penalizing overly confident policies |
| Max gradient norm | 0.5 | Gradient clipping for training stability |
| Rollout buffer size | 4096 | Steps collected before each PPO update |
| PPO epochs | 4 | Number of passes over the buffer per update |
| Total timesteps | 500,000 | Total training steps |

**Training loop:**

1. **Collect rollout:** Run the agent in the environment for 4096 steps, storing observations, actions, rewards, log-probs, and values. Episodes are sampled 50/50 between human and bot sessions.
2. **Compute advantages:** Use Generalized Advantage Estimation (GAE) to compute per-step advantages and returns. Bootstrap the final value if the episode is not done.
3. **PPO update:** For 4 epochs over the collected buffer:
   - Replay each episode segment through the LSTM from its initial hidden state (preserving temporal continuity)
   - Normalize advantages within each segment
   - Compute the PPO clipped surrogate loss: `L = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)`
   - Compute value loss: `MSE(predicted_value, return)`
   - Compute total loss: `policy_loss + 0.5 × value_loss - 0.01 × entropy`
   - Clip gradients to max norm 0.5 and step the Adam optimizer
4. **Repeat** until 500,000 total steps are reached

**Note:** Episodes are processed sequentially (not shuffled as minibatches) to preserve LSTM hidden-state continuity across timesteps within each episode.

---

### Live Inference (Production)

The trained agent is loaded by `agent_service.py` in the Flask backend. Three inference modes are used in production:

#### 1. Full Session Evaluation (`evaluate_session`)

Called when the user clicks "Purchase" at checkout.

1. Builds a timeline from raw telemetry, splits into overlapping windows of 30 events
2. Encodes each window into the 26-dim statistical feature vector
3. Runs the entire window sequence through the LSTM in a single batched forward pass
4. Takes the **last 30% of windows** (minimum 2) and averages their action probabilities — this gives the LSTM's most informed assessment after processing all evidence
5. Applies a confidence threshold:
   - If `p_suspicious > 0.45` (sum of puzzle + block probabilities):
     - `p_block > 0.4` or `p_hard_puzzle > 0.25` → hard puzzle
     - Otherwise → medium or easy puzzle based on which has higher probability
   - Otherwise → allow

#### 2. Rolling Evaluation (`rolling_evaluate`)

Called every 3 seconds during the checkout flow for live monitoring.

1. Processes all events collected so far, split into windows
2. Takes the **last 30% of windows** (minimum 2)
3. Applies **exponential weighting** — more recent windows count more heavily
4. Returns a `bot_probability` score and whether to `deploy_honeypot`
5. Used by the frontend to show real-time suspicion levels on the Dev Dashboard

#### 3. Online Learning (`online_learn`)

Called after a session is labeled as human or bot (via bot scripts or the `/api/agent/confirm` endpoint).

1. Evaluates the session **before** the update (captures before-decision)
2. Replays all windows through the agent, collecting actions and rewards
3. Final window receives the true-label reward (correct/incorrect allow/block)
4. Runs an **aggressive PPO update**: 3 epochs, learning rate at 60% of training rate (1.8×10⁻⁴)
5. Evaluates the session **after** the update (captures after-decision)
6. Logs before/after comparison to `online_training.log` with IMPROVED/UNCHANGED/REGRESSED status
7. Saves the updated checkpoint immediately
8. This allows the model to continuously improve from real-world sessions

---

### Evaluation Metrics

When running `evaluate_ppo.py`, the following metrics are reported:

**Confusion matrix:**
- **TP (True Positive):** Bot correctly blocked or caught by puzzle
- **TN (True Negative):** Human correctly allowed through
- **FP (False Positive):** Human incorrectly blocked or challenged
- **FN (False Negative):** Bot incorrectly allowed through or passed puzzle

**Derived metrics:**
- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP) — of all users flagged as bots, how many actually were?
- **Recall** = TP / (TP + FN) — of all actual bots, how many were caught?
- **F1 Score** = 2 × Precision × Recall / (Precision + Recall)

**Additional outputs:**
- Outcome distribution (percentage breakdown of all outcomes)
- Final action distribution (which terminal actions the agent prefers)
- Decision timing (average number of events processed before deciding, split by human vs. bot)

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/checkout` | Submit checkout form |
| POST | `/api/tracking/mouse` | Batch mouse samples |
| POST | `/api/tracking/clicks` | Batch click events |
| POST | `/api/tracking/keystrokes` | Batch keystroke timing |
| POST | `/api/tracking/scroll` | Batch scroll events |
| POST | `/api/agent/rolling` | Rolling inference (bot prob + honeypot deploy) |
| POST | `/api/agent/evaluate` | Full agent evaluation at checkout |
| POST | `/api/agent/confirm` | Online learning update (human/bot label) |
| GET | `/api/agent/dashboard/<sid>` | Full agent analysis + LSTM state |
| GET | `/api/agent/live/<sid>` | Live telemetry counts |
| GET | `/api/agent/sessions` | Recent sessions list |
| GET | `/api/agent/session-ids` | All session IDs |
| GET | `/api/session/raw/<sid>` | Raw session telemetry data |
| GET | `/api/export/tracking` | Export telemetry to CSV |
| GET | `/api/export` | Export checkout data to CSV |
| GET | `/api/health` | Health check |

## Tech Stack

- **Frontend:** React 18.2, Vite 5, React Router DOM 6, vanilla CSS
- **Backend:** Python 3.12, Flask 3.0, Flask-CORS, mysql-connector-python
- **Agent:** PyTorch PPO+LSTM (lazy-loaded, thread-safe)
- **Database:** MySQL 8.0+