# Ticket Monarch

Developed by Team 25 - San Jose State University 

Ticket Monarch is a mock web application for concert ticket booking.

In this mock environment, users will browse concerts, select their seats, and checkout. During this process, the system tracks mouse moevments, clicks, keystrokes, and scrolls.

At checkout, a PPO + LSTM agent will evaluate the session - resulting in a bot versus human detection output.

## Project Structure

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- MySQL 8.0+

### Database

```bash
cp .env.example .env          # Edit and set MYSQL_PASSWORD
```

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows (source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
pip install -r ../../rl_captcha/requirements.txt   # For agent inference
python setup_mysql.py
```

### Frontend

```bash
cd frontend
npm install
```

### Running

Open two terminals:

```bash
# Terminal 1: Backend
cd TicketMonarch/backend
venv\Scripts\activate
python app.py                  # http://localhost:5000

# Terminal 2: Frontend
cd TicketMonarch/frontend
npm run dev                    # http://localhost:3000
```

Vite proxies `/api/*` to Flask. Access the app at **http://localhost:3000**.

## User Flow

1. **Home** (`/`) — Browse concerts, select one
2. **Seat Selection** (`/seats/:id`) — Pick seats from interactive layout
3. **Checkout** (`/checkout`) — Fill payment form
   - Rolling inference polls every 3s, deploys honeypot if suspicious
   - On "Purchase": telemetry force-flushed, agent evaluates full session
   - Honeypot triggered → instant hard puzzle
   - High bot probability → scaled puzzle (easy/medium/hard based on confidence)
   - Low suspicion → checkout proceeds
4. **Confirmation** (`/confirmation`) — Order confirmed, session sent for online RL update, session resets

## Telemetry

The frontend captures behavioral signals and batches them to the backend every 5 seconds:

| Signal | Rate | Data |
|--------|------|------|
| Mouse movement | ~66Hz (15ms sampling) | x, y, timestamp |
| Clicks | Every click | x, y, button, target element, time delta |
| Keystrokes | Every key in form fields | field ID, down/up, timestamp, special keys only |
| Scroll | Every scroll | scrollX, scrollY, delta, time delta |

## Dev Dashboard

Open **http://localhost:3000/dev** in a separate tab.

**Live Monitor:** Auto-detects the active session from localStorage. Polls every 1s showing real-time counts (mouse, clicks, keystrokes, scrolls) and rolling bot probability.

**Analyze Session:** Full agent analysis on any session: decision banner, action probability bars, per-event timeline, LSTM hidden state heatmap (128 units).

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
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

## Database

MySQL `ticketmonarch_db` with tables:
- **checkouts** — Order form submissions
- **user_sessions** — Telemetry in JSON columns (`mouse_movements`, `click_events`, `keystroke_data`, `scroll_events`).

## Tech Stack

- **Frontend**: React 18.2, Vite 5, React Router DOM 6, vanilla CSS, UUID v4
- **Backend**: Python 3.12, Flask 3.0, Flask-CORS, mysql-connector-python
- **Agent**: PyTorch PPO+LSTM (lazy-loaded, thread-safe with Lock)
- **Database**: MySQL 8.0+