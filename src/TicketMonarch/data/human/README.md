# Human Telemetry Data

Real human browsing sessions. Treated as **label=1 (human)** by the training pipeline.

## How to Collect

1. Load the Chrome extension from `chrome-extension/` into Chrome
2. Click **"Start Recording"** in the extension popup
3. Browse the TicketMonarch site normally (Home -> Seats -> Checkout -> Purchase)
4. Click **"Export JSON"** in the extension popup
5. Save the file here as `.json`

Sessions are also auto-saved here by the backend when a human completes a purchase and the online learning update runs.

## JSON Format

Each file contains one or more sessions keyed by session UUID:

```json
{
  "<session_id>": {
    "sessionId": "...",
    "startTime": 1234567890,
    "pageMeta": [...],
    "totalSegments": 3,
    "segments": [
      {
        "segmentId": 1,
        "url": "http://localhost:3000/",
        "mouse": [{ "x": 100, "y": 200, "t": 1234.5 }],
        "clicks": [{ "t": 1234.5, "x": 100, "y": 200, "button": "left", "dt_since_last": 500 }],
        "keystrokes": [{ "field": "card_number", "type": "down", "t": 1234.5, "key": null }],
        "scroll": [{ "t": 1234.5, "scrollX": 0, "scrollY": 500, "dy": 100 }]
      }
    ]
  }
}
```

Segments are split by idle gaps (3+ seconds of inactivity). For training, all segments within a session are merged into flat event lists.

## Usage

All `.json` files here are automatically loaded by:

```bash
python -m rl_captcha.scripts.train_ppo --data-dir data/
```

Note: JSON files are gitignored. Training data stays local only.
