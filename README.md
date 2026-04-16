# RL-Based Strategies for Improving Synthetic CAPTCHAs

> A mock concert ticket-booking web app that uses reinforcement learning (PPO, DG, or Soft PPO — all LSTM-based) to detect bots in real time based on raw telemetry (mouse movements, clicks, keystrokes, scrolls).

[![CI](https://github.com/SJSU-CMPE-195/group-project-team-25/actions/workflows/ci.yml/badge.svg)](https://github.com/SJSU-CMPE-195/group-project-team-25/actions/workflows/ci.yml)

[![codecov](https://codecov.io/gh/SJSU-CMPE-195/group-project-team-25/graph/badge.svg?token=K07K6AT9LO)](https://codecov.io/gh/SJSU-CMPE-195/group-project-team-25)

## Team

| Name | GitHub | Email |
|------|--------|-------|
| Meghana Indukuri | [@meghanai28](https://github.com/meghanai28) | meghana.indukuri@sjsu.edu |
| Eman Naseekhan | [@emannk](https://github.com/emannk) | eman.naseerkhan@sjsu.edu |
| Joshua Rose | [@JB-Rose](https://github.com/JB-Rose) | joshua.rose@sjsu.edu |
| Martin Tran | [@martintranthecoder](https://github.com/martintranthecoder) | vietnhatminh.tran@sjsu.edu |

**Advisor:** Dr. Younghee Park

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | React 18.2, Vite 5, React Router DOM 6, vanilla CSS  |
| Backend | Python 3.12, Flask 3.0, Flask-CORS, mysql-connector-python |
| RL Agent | PyTorch, PPO/DG/Soft-PPO + LSTM (algorithm selectable via `RL_ALGORITHM` env var) |
| Database | MySQL 8.0+ |

---

## Demo Notebooks

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- MySQL 8.0+

### Installation

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

### Running Locally

Open **two terminals** (activate the venv in each if running Python):

```bash
# Terminal 1 — Backend (http://localhost:5000)
# Set RL_ALGORITHM to ppo, dg, or soft_ppo (defaults to ppo)
set RL_ALGORITHM=dg          # PowerShell: $env:RL_ALGORITHM="dg"
cd .. #Should be in the project_root/src/ directory
python -m TicketMonarch.backend.app

# Terminal 2 — Frontend (http://localhost:3000)
cd TicketMonarch/frontend
npm run dev
```

---

## Usage
Open **http://localhost:3000** in your browser. Vite proxies `/api/*` requests to Flask automatically.

1. **Home** (`/`) — Browse concerts and select one
2. **Seat Selection** (`/seats/:id`) — Pick seats from an interactive layout
3. **Checkout** (`/checkout`) — Fill the payment form
   - Rolling inference polls every 3 seconds and can request honeypot deployment
   - Telemetry is force-flushed before final verification
   - Final checkout uses the RL policy endpoint (`/api/agent/evaluate`)
   - Policy outputs map directly to `allow`, `block`, or `easy/medium/hard` puzzle
   - Honeypot interactions immediately escalate to a hard puzzle
4. **Confirmation** (`/confirmation`) — Order confirmed, session sent for online RL update

## Dev Dashboard

Open **http://localhost:3000/dev** in a separate tab.

- **Live Monitor:** Auto-detects the active session, polls every 1 second showing real-time event counts and rolling bot probability.
- **Analyze Session:** Full agent analysis on any session — decision banner, action probability bars, per-event timeline, LSTM hidden-state heatmap.

## Data Collection Notes

- Mouse telemetry is now only sampled after real mouse movement; stationary cursors no longer emit endless duplicate points.
- Frontend telemetry batches are re-queued on failed network writes instead of being dropped.
- Bot generators were toned down so bot sessions better resemble real checkout flows.
- Because telemetry semantics changed, the previous JSON training set was cleared. Recollect fresh human and bot data before retraining.
---

## Testing

Unit test cases are under src/TicketMonarch/frontend/tests/unit and src/TicketMonarch/backend/tests/unit
Integration test are src/TicketMonarch/backend/tests/integration

### Running frontend tests
Under src/TicketMonarch/frontend
```
npm test
```

### Running backend tests
```
pip install pytest pytest-cov gymnasium
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Under project root
```
$env:PYTHONPATH = "src"
pytest  --cov=src/TicketMonarch/backend --cov-report=term-missing
```
---

## API Reference

<details>
<summary>Click to expand API endpoints</summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
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

</details>

---

## Project Structure

```
src/
├── TicketMonarch/          # Main web application
│   ├── backend/            # Flask API + agent inference
│   ├── frontend/           # React + Vite SPA
│   └── .env.example        # MySQL connection config template
├── rl_captcha/             # RL agents: PPO, DG, Soft PPO (training & evaluation)
├── bots/                   # Selenium & LLM bots for data collection
├── chrome-extension/       # Telemetry capture extension
└── data/                   # Training data (human/ and bot/)
```

---

## Troubleshooting

### macOS: Port 5000 Conflict (AirPlay Receiver)

On **macOS Monterey and later**, port 5000 is used by the system's **AirPlay Receiver** feature, which will prevent the Flask backend from starting.

**Symptoms:** `Address already in use` error when running `python TicketMonarch/backend/app.py`

**Fix — Disable AirPlay Receiver:**

1. Open **System Settings** (or System Preferences on older macOS)
2. Go to **General** → **AirDrop & Handoff**
3. Turn off **AirPlay Receiver**
4. After disabling it, restart the backend. Port 5000 will now be free.
     
**Alternative — Run the backend on a different port:**

1. If you prefer to keep AirPlay Receiver enabled, you can change the backend port. In `TicketMonarch/backend/app.py`, find the last line and update it:
   ```python
   # Change 5000 to any free port, e.g. 5001
   app.run(debug=True, port=5001)
   ```
3. Then update `TicketMonarch/frontend/vite.config.js` to proxy to the new port:
   ```js
   proxy: {
     '/api': 'http://localhost:5001'
   }
   ```

*CMPE 195A/B - Senior Design Project | San Jose State University | Spring 2026*
