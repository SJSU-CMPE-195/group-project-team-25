# Human Telemetry Data

Real human browsing sessions. Treated as **label=1 (human)** by the training pipeline.

**Important:** Only include data from the TicketMonarch site (localhost). Data from external sites (Gmail, GitHub, Canvas, etc.) will pollute the training distribution and was removed during cleanup.

## How to Collect

1. Browse the site normally while the backend is running
2. Call `POST /api/agent/confirm` with `{ "session_id": "...", "true_label": 1 }`
3. Sessions are auto-saved here as `session_<uuid>.json`

Bot scripts call this endpoint automatically after each run. Human confirmations save here and the agent does an online update.

## JSON Format

### Live Confirm Format (`session_*.json`)

Single session with segments at the top level:

```json
{
  "sessionId": "abc-123",
  "segments": [
    {
      "mouse": [{ "x": 100, "y": 200, "t": 1234.5 }],
      "clicks": [...],
      "keystrokes": [...],
      "scroll": [...]
    }
  ]
}
```

Segments are split by idle gaps (3+ seconds of inactivity). For training, all segments within a session are merged into flat event lists, grouped into 30-event windows, and capped by the current `max_windows` setting.

## Current Status

The previous collected dataset was intentionally cleared after telemetry-capture changes. Recollect fresh human sessions before retraining.

## Usage

All `.json` files here are automatically loaded by:

```bash
python -m rl_captcha.scripts.train_ppo --data-dir data/
```

Note: JSON files are gitignored. Training data stays local only.
