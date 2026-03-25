# Bots

Bot implementations for generating training data. Each bot drives a real Chrome browser against the live TicketMonarch site. The frontend's built-in `tracking.js` captures all telemetry automatically, and the bot auto-exports session data to `data/bot/` after each run.

## Prerequisites

- TicketMonarch running locally (`python TicketMonarch/backend/app.py` + `cd TicketMonarch/frontend && npm run dev`)
- Chrome installed on your machine

## Setup

```bash
pip install -r bots/requirements.txt

# For LLM bot only:
pip install browser-use playwright langchain-anthropic
playwright install chromium
```

## Selenium Bot

Three behavior profiles for generating diverse bot data:

| Type | Mouse | Typing | Scrolls | Detection Difficulty |
|------|-------|--------|---------|---------------------|
| `linear` | Straight-line + light idle fidgeting | Uniform 20ms intervals | None | Easy |
| `scripted` | Bezier curves + light idle fidgeting | Varied timing, burst typing | Human-like momentum scrolls | Medium |
| `replay` | Replays recorded human mouse data + noise | Uniform-ish | Replayed from source | Hard |

All bot types include small idle movements between actions, but these are intentionally limited so bot sessions stay closer to real checkout flows.

### Commands

```bash
# Scripted bot (default) -- Bezier curves, varied timing, scrolling, fidgeting
python bots/selenium_bot.py --runs 10 --type scripted

# Linear bot -- robotic straight-line movement, uniform typing
python bots/selenium_bot.py --runs 5 --type linear

# Replay bot -- replays real human mouse patterns with noise
python bots/selenium_bot.py --runs 3 --type replay \
    --replay-source data/human/session_example.json
```

### How It Works

1. Opens Chrome (no extension needed -- `tracking.js` captures everything)
2. Sets `tm_is_bot=1` in sessionStorage so the confirmation page won't auto-confirm as human
3. Runs the full booking flow for each run: Home -> Seat Selection -> Checkout -> Purchase
4. Handles challenge modals if they appear (slider, canvas text, timed click -- up to 3 retries)
5. After each run, waits for telemetry flush, then:
   - Pulls raw telemetry from `/api/session/raw/<sid>`
   - Saves JSON to `data/bot/` with a UTC timestamp filename
   - Confirms the session as a bot (`true_label=0`) via `/api/agent/confirm`, triggering an online RL update

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | `3` | Number of bot sessions to run |
| `--type` | `scripted` | Bot behavior: `linear`, `scripted`, `stealth`, `slow`, `erratic`, `speedrun`, `tabber`, `replay`, `semi_auto`, `trace_conditioned`, or `mixed` |
| `--replay-source` | -- | Path to human session JSON (required for `replay`) |
| `--pause-between` | `2.0` | Seconds between runs |

## LLM Bot

Uses [browser-use](https://github.com/browser-use/browser-use) to give an LLM (Claude or GPT-4o) autonomous control of Chrome. The LLM reads the page, decides where to click, what to type, and completes the booking flow on its own.

### Setup

```bash
pip install browser-use playwright langchain-anthropic
playwright install chromium

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# or for OpenAI:
# pip install langchain-openai
# export OPENAI_API_KEY=sk-...
```

### Commands

```bash
# Run with Claude (default)
python bots/llm_bot.py --runs 3 --provider anthropic

# Run with GPT-4o
python bots/llm_bot.py --runs 3 --provider openai

# Custom task prompt
python bots/llm_bot.py --task "Go to localhost:3000, browse concerts, pick the cheapest tickets"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | `1` | Number of bot sessions |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `--pause-between` | `3.0` | Seconds between runs |
| `--task` | *(full booking flow)* | Custom instruction prompt for the LLM |

### How It Works

1. Opens Chrome (visible, not headless)
2. The LLM autonomously navigates: browses concerts, selects seats, fills checkout
3. `tracking.js` captures all telemetry in the background
4. After each run, auto-exports telemetry to `data/bot/` and confirms as bot

## Current Collection Notes

- Selenium bots still generate intentional mouse movement and pauses, but the movement inflation was reduced.
- The optional LLM event injector no longer emits a continuous background mousemove loop; it now adds small movement bursts around focus and click actions only.
- Because telemetry semantics changed, recollect fresh bot data before retraining.
