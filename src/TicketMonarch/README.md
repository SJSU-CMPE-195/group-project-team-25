# Ticket Monarch

A mock concert ticket-booking web app with real-time bot detection using reinforcement learning.

## Prerequisites

- Python 3.12+
- Node.js 18+
- MySQL 8.0+

## Setup

All commands from the `src/` directory.

### 1. Database

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

Create the database:
```bash
python TicketMonarch/backend/setup_mysql.py
```

### 2. Backend dependencies

```bash
pip install -r TicketMonarch/backend/requirements.txt
pip install -r rl_captcha/requirements.txt
```

### 3. Frontend dependencies

```bash
cd TicketMonarch/frontend
npm install
cd ../..
```

---

## Running

Open two terminals (activate venv in each).

**PowerShell:**
```powershell
# Terminal 1 — Backend (http://localhost:5000)
$env:RL_ALGORITHM="ppo"    # Options: ppo, dg, soft_ppo (default: ppo)
python TicketMonarch/backend/app.py

# Terminal 2 — Frontend (http://localhost:3000)
cd TicketMonarch/frontend
npm run dev
```

**macOS / Linux:**
```bash
# Terminal 1
RL_ALGORITHM=ppo python TicketMonarch/backend/app.py

# Terminal 2
cd TicketMonarch/frontend && npm run dev
```

Open **http://localhost:3000** in your browser.

### Environment Variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `RL_ALGORITHM` | `ppo` | RL algorithm to use: `ppo`, `dg`, or `soft_ppo` |
| `DISABLE_HUMAN_SAVE` | not set | Set to `1` to disable human session saving and human-label online learning. Use this when running LLM bots to prevent data poisoning. |

**Example — backend for LLM bot data collection (PowerShell):**
```powershell
$env:RL_ALGORITHM="ppo"
$env:DISABLE_HUMAN_SAVE="1"
python TicketMonarch/backend/app.py
```

---

## User Flow

1. **Home** (`/`) — Browse concerts, select one
2. **Seat Selection** (`/seats/:id`) — Pick seats
3. **Checkout** (`/checkout`) — Fill payment form, RL agent evaluates on Purchase
4. **Confirmation** (`/confirmation`) — Order confirmed, session used for online learning

## Dev Dashboard

Open **http://localhost:3000/dev** in a separate tab.

- **Live Monitor** — Real-time event counts and bot probability (polls every 1s)
- **Analyze Session** — Full agent analysis: decision, action probabilities, LSTM heatmap

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/checkout` | Submit checkout form |
| POST | `/api/tracking/mouse` | Batch mouse samples |
| POST | `/api/tracking/clicks` | Batch click events |
| POST | `/api/tracking/keystrokes` | Batch keystroke timing |
| POST | `/api/tracking/scroll` | Batch scroll events |
| POST | `/api/agent/rolling` | Rolling inference (bot prob + honeypot) |
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
