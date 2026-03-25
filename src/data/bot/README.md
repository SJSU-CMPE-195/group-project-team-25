# Bot Telemetry Data

Automated bot browsing sessions. Treated as **label=0 (bot)** by the training pipeline.

**Important:** Only include bot data collected against the TicketMonarch site (localhost). Old bot data from other platforms or sites was removed during cleanup.

## How to Collect

### Selenium Bots

```bash
python -m bots.selenium_bot --runs 50 --type mixed --skip-honeypot
```

The bot auto-exports telemetry from the backend API after each run and saves JSON files here automatically. `tracking.js` captures everything in the browser.

### LLM Bot

```bash
python bots/llm_bot.py --runs 3 --provider anthropic
```

Same auto-export behavior — telemetry is pulled from the backend and saved here after each run.

See `bots/README.md` for full setup and options.

## JSON Format

Sessions use the live-confirm format with segments at the top level:

```json
{
  "sessionId": "...",
  "label": 0,
  "bot_type": "scripted",
  "tier": 2,
  "segments": [
    {
      "mouse": [...],
      "clicks": [...],
      "keystrokes": [...],
      "scroll": [...]
    }
  ]
}
```

For training, all segments within a session are merged into flat event lists, then grouped into 30-event windows for the windowed observation encoder and capped by the current `max_windows` setting.

## Current Status

The previous collected dataset was intentionally cleared after telemetry and bot-behavior changes. Recollect fresh bot sessions before retraining.

## Usage

All `.json` files here are automatically loaded by:

```bash
python -m rl_captcha.scripts.train_ppo --data-dir data/
```

Note: JSON files are gitignored. Training data stays local only.
