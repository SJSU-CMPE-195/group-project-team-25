# Bots

Bot implementations for generating labeled training data. Each bot drives a real Chrome browser through the full TicketMonarch booking flow. The site's built-in `tracking.js` captures all telemetry automatically.

## Before You Start

1. **TicketMonarch must be running** — backend + frontend (see below)
2. **Chrome** must be installed on your machine
3. **Virtual environment** must be activated with bot dependencies installed

## Setup

All commands from the `src/` directory.

```bash
pip install -r bots/requirements.txt
```

For the LLM bot, also run:
```bash
pip install browser-use playwright langchain-anthropic
playwright install chromium
```

Start the app (two terminals):

**PowerShell:**
```powershell
# Terminal 1 — Backend
python TicketMonarch/backend/app.py

# Terminal 2 — Frontend
cd TicketMonarch/frontend
npm run dev
```

---

## Selenium Bot

Generates bot sessions with different behavior profiles.

| Type | Behavior | Difficulty |
|------|----------|------------|
| `linear` | Straight-line mouse, uniform typing | Easy |
| `scripted` | Bezier curves, varied timing, scrolling | Medium |
| `stealth` | Human-like Bezier with micro-jitter, lognormal typing | Hard |
| `slow` | Slow, deliberate movements | Medium |
| `erratic` | Irregular, jerky movements | Medium |
| `speedrun` | Extremely fast completion | Easy |
| `tabber` | Keyboard-heavy navigation | Easy |
| `replay` | Replays a recorded human session with noise | Hard |
| `semi_auto` | Mix of real human + bot actions | Very Hard |
| `trace_conditioned` | Replays perturbed human traces | Very Hard |
| `mixed` | Random mix of all types | Mixed |

### Commands

```bash
# 5 runs with scripted behavior (default type)
python bots/selenium_bot.py --runs 5 --type scripted

# 5 linear bot runs
python bots/selenium_bot.py --runs 5 --type linear

# 5 stealth bot runs
python bots/selenium_bot.py --runs 5 --type stealth

# 10 mixed runs (random bot types)
python bots/selenium_bot.py --runs 10 --type mixed

# Replay a recorded human session
python bots/selenium_bot.py --runs 3 --type replay --replay-source data/human/session_example.json

# Skip honeypot fields
python bots/selenium_bot.py --runs 5 --type scripted --skip-honeypot
```

### All Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--runs` | `3` | Number of sessions to run |
| `--type` | `scripted` | Bot behavior type (see table above) |
| `--replay-source` | — | Path to a human session JSON (required for `replay` type) |
| `--pause-between` | `2.0` | Seconds to wait between runs |
| `--skip-honeypot` | off | Skip unknown form fields to avoid honeypot traps |

### What happens each run

1. Opens Chrome and navigates through Home → Seats → Checkout → Purchase
2. If a challenge appears, the bot tries to solve it (up to 3 retries)
3. After each run, saves telemetry to `data/bot/` and confirms as bot via the API

---

## LLM Bot

Uses an LLM (Claude or GPT-4o) to autonomously control Chrome and complete the booking flow. Produces the most human-like bot behavior.

### Set API keys

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
# or for OpenAI:
$env:OPENAI_API_KEY="sk-..."
```

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or for OpenAI:
export OPENAI_API_KEY=sk-...
```

### Important: Prevent data poisoning

LLM bots that pass the RL agent's check get auto-saved as human data, which poisons the training set. To prevent this, set `DISABLE_HUMAN_SAVE=1` on the **backend** before running LLM bots:

**PowerShell (restart backend with this):**
```powershell
$env:DISABLE_HUMAN_SAVE="1"
$env:RL_ALGORITHM="ppo"
python TicketMonarch/backend/app.py
```

**macOS / Linux:**
```bash
DISABLE_HUMAN_SAVE=1 RL_ALGORITHM=ppo python TicketMonarch/backend/app.py
```

This disables human session saving and human-label online learning updates while keeping bot-label saves and updates working normally. **Remove the env var when you go back to collecting human data.**

### Commands

```bash
# 3 runs with Claude
python bots/llm_bot.py --runs 3 --provider anthropic

# 3 runs with GPT-4o
python bots/llm_bot.py --runs 3 --provider openai

# Use DOM interaction mode (more telemetry)
python bots/llm_bot.py --runs 3 --provider anthropic --mode dom

# Skip honeypot fields
python bots/llm_bot.py --runs 3 --provider anthropic --skip-honeypot

# Enable DOM event injection (alternates on/off per run)
python bots/llm_bot.py --runs 4 --provider anthropic --inject-events
```

### All Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--runs` | `1` | Number of sessions to run |
| `--provider` | `anthropic` | LLM provider: `anthropic`, `openai`, or `gemini` |
| `--mode` | `screenshot` | Interaction mode: `screenshot`, `dom`, `accessibility`, or `mixed` |
| `--pause-between` | `3.0` | Seconds between runs |
| `--task` | *(full booking flow)* | Custom instruction for the LLM |
| `--inject-events` | off | Enable DOM event injection (alternates on/off per run) |
| `--skip-honeypot` | off | Skip unknown form fields |

### What happens each run

1. Opens a visible Chrome window
2. The LLM reads the page and autonomously navigates the booking flow
3. After each run, saves telemetry to `data/bot/` and confirms as bot via the API

---

## Output

Bot telemetry is saved as JSON files in `data/bot/`:
- Selenium: `session_<session_id>.json`
- LLM: `llm_bot_<timestamp>.json`
