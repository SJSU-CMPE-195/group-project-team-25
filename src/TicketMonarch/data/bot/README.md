# Bot Telemetry Data

Automated bot browsing sessions. Treated as **label=0 (bot)** by the training pipeline.

## How to Collect

### Selenium Bots

```bash
cd bots
python selenium_bot.py --runs 5 --type scripted
```

The bot opens Chrome with the telemetry extension. After all runs complete, click **"Export JSON"** in the extension popup and save the file here.

### LLM Bot

```bash
cd bots
python llm_bot.py --runs 3 --provider anthropic
```

Same process — export from the Chrome extension after runs complete.

See `bots/README.md` for full setup and options.

## JSON Format

Files from the Chrome extension use the same format as human data — session objects keyed by UUID with segments containing mouse, clicks, keystrokes, and scroll arrays.

```json
{
  "<session_id>": {
    "sessionId": "...",
    "segments": [
      {
        "mouse": [...],
        "clicks": [...],
        "keystrokes": [...],
        "scroll": [...]
      }
    ]
  }
}
```

## Usage

All `.json` files here are automatically loaded by:

```bash
python -m rl_captcha.scripts.train_ppo --data-dir data/
```

Note: JSON files are gitignored. Training data stays local only.
