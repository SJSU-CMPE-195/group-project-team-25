# RL-Based Strategies for Improving and Attacking Synthetic CAPTCHAs

> A mock concert ticket-booking web app that uses a PPO+LSTM reinforcement learning agent to detect bots in real time based on raw telemetry (mouse movements, clicks, keystrokes, scrolls).

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
| Database | MySQL 8.0+ |
| Deployment | |

---

## Getting Started

### Prerequisites

- [Python] 3.12+
- [Node.js] 18+
- [MySQL] 8.0+

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
python TicketMonarch/backend/app.py

# Terminal 2 — Frontend (http://localhost:3000)
cd TicketMonarch/frontend
npm run dev
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
├── rl_captcha/             # PPO+LSTM agent (training & evaluation)
├── bots/                   # Selenium & LLM bots for data collection
├── chrome-extension/       # Telemetry capture extension
└── data/                   # Training data (human/ and bot/)
```

---

*CMPE 195A/B - Senior Design Project | San Jose State University | Spring 2026*
