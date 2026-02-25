# Bots

Bot implementations for generating training data. Each bot drives a real Chrome browser with the telemetry Chrome extension loaded, so all mouse movements, clicks, keystrokes, and scrolls are captured automatically.

## Prerequisites

- TicketMonarch running locally (`python app.py` + `npm run dev`)
- Chrome extension at `chrome-extension/` (loaded automatically by the bots)

## Setup

```bash
cd bots
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# For LLM bot only:
pip install browser-use playwright langchain-anthropic
playwright install chromium
```

## Selenium Bot

Three behavior profiles for generating diverse bot data:

| Type | Mouse | Typing | Scrolls | Detection Difficulty |
|------|-------|--------|---------|---------------------|
| `linear` | Straight-line, instant | Uniform 20ms intervals | None | Easy |
| `scripted` | Bezier curves, slight jitter | Varied timing | Random scrolls | Medium |
| `replay` | Replays recorded human mouse data + noise | Uniform-ish | Replayed from source | Hard |

### Commands

```bash
# Linear bot -- robotic straight-line movement, uniform typing
python selenium_bot.py --runs 5 --type linear

# Scripted bot -- Bezier curves, varied timing, scrolling
python selenium_bot.py --runs 5 --type scripted

# Replay bot -- replays real human mouse patterns with noise
python selenium_bot.py --runs 3 --type replay \
    --replay-source ../data/human/telemetry_export_example.json
```

### How It Works

1. Opens Chrome with the telemetry extension loaded
2. Prompts you to **click "Start Recording"** in the extension popup
3. Runs the full booking flow: Home -> Seat Selection -> Checkout -> Purchase
4. After all runs, prompts you to **click "Export JSON"** in the extension popup
5. Save the exported file to `data/bot/`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | 3 | Number of bot sessions to run |
| `--type` | linear | Bot behavior: `linear`, `scripted`, or `replay` |
| `--replay-source` | -- | Path to human telemetry JSON (required for `replay`) |
| `--pause-between` | 2.0 | Seconds between runs |

## LLM Bot

Uses [browser-use](https://github.com/browser-use/browser-use) to give an LLM (Claude or GPT-4) autonomous control of Chrome. The LLM reads the page, decides where to click, what to type, and completes the booking flow on its own — producing the most realistic bot behavior.

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
python llm_bot.py --runs 3 --provider anthropic

# Run with GPT-4o
python llm_bot.py --runs 3 --provider openai

# Custom task prompt
python llm_bot.py --task "Go to localhost:3000, browse concerts, pick the cheapest tickets"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | 1 | Number of bot sessions |
| `--provider` | anthropic | LLM provider: `anthropic` or `openai` |
| `--pause-between` | 3.0 | Seconds between runs |
| `--task` | (full booking flow) | Custom instruction prompt for the LLM |

### How It Works

1. Opens Chrome with the telemetry extension (must be visible, not headless)
2. The LLM autonomously navigates: browses concerts, selects seats, fills checkout
3. Chrome extension captures all telemetry in the background
4. After runs complete, export from the extension popup and save to `data/bot/`

## After Running Bots

1. Click **"Export JSON"** in the Chrome extension popup
2. Save the file to `data/bot/`
3. The data is auto-labeled as bot (label=0) during training
