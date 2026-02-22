# TicketMonarch

A full-stack concert ticket booking application with integrated behavioral telemetry and RL-based bot detection. Users browse concerts, select seats, and checkout -- while the system captures mouse movements, clicks, keystrokes, and scrolls. At checkout, a PPO+LSTM agent evaluates the session and decides whether to allow, challenge, or block.

## Project Structure

```
TicketMonarch/
├── backend/
│   ├── app.py              # Flask API server (tracking, checkout, agent endpoints)
│   ├── agent_service.py    # PPO+LSTM inference wrapper (lazy-loaded)
│   ├── database.py         # MySQL queries, telemetry storage (JSON_MERGE_PRESERVE)
│   ├── config.py           # MySQL configuration from .env
│   ├── setup_mysql.py      # Database & table creation script
│   ├── models.py           # SQLAlchemy ORM definitions (legacy, unused)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Router + tracking initialization
│   │   ├── pages/
│   │   │   ├── Home.jsx              # Concert selection
│   │   │   ├── SeatSelection.jsx     # Interactive seat picker
│   │   │   ├── Checkout.jsx          # Payment form + agent evaluation
│   │   │   ├── Confirmation.jsx      # Order confirmation
│   │   │   └── DevDashboard.jsx      # Live telemetry monitor + agent analysis
│   │   ├── components/
│   │   │   └── ChallengeModal.jsx    # Puzzle/block challenge UI
│   │   └── services/
│   │       ├── api.js                # Backend API calls
│   │       └── tracking.js           # Telemetry capture (mouse 66Hz, clicks, keys, scroll)
│   ├── vite.config.js      # Vite config (proxies /api to Flask)
│   └── package.json        # Node.js dependencies
└── README.md
```

## Setup

### Prerequisites
- Python 3.12+, Node.js 18+, MySQL 8.0+

### Database

```bash
cp .env.example .env          # Set MYSQL_PASSWORD
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r ../../rl_captcha/requirements.txt   # For agent inference
python setup_mysql.py
```

### Running

```bash
# Terminal 1: Backend
cd backend
python app.py                  # http://localhost:5000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev                    # http://localhost:3000
```

## User Flow

1. **Home** (`/`) -- Browse concerts, select one
2. **Seat Selection** (`/seats/:id`) -- Pick seats from interactive layout
3. **Checkout** (`/checkout`) -- Fill payment form
   - On submit: telemetry is force-flushed, agent evaluates session
   - Agent allows: checkout proceeds
   - Agent challenges: puzzle modal appears (easy/medium/hard)
   - Solve puzzle: checkout proceeds. Fail: blocked.
4. **Confirmation** (`/confirmation`) -- Order summary

## Telemetry Tracking

The frontend captures behavioral signals and batches them to the backend every 5 seconds:

| Signal | Sampling | Data |
|--------|----------|------|
| Mouse movement | ~66Hz (15ms interval) | x, y, timestamp |
| Clicks | Every click | x, y, button, target element, time delta |
| Keystrokes | Every key in form fields | field ID, down/up, timestamp, special keys |
| Scroll | Every scroll event | scrollX, scrollY, time delta |

Tracking is disabled on the `/dev` dashboard to prevent data pollution.

## Dev Dashboard

Open **http://localhost:3000/dev** in a separate browser tab.

### Live Monitor
- Reads the active session ID from `localStorage` (set by the browsing tab)
- Polls `/api/agent/live/<sid>` every 1 second
- Shows real-time counts: mouse moves, clicks, keystrokes, scrolls
- "Run Agent Analysis" button switches to Analyze mode

### Analyze Session
- Dropdown of recent sessions from the database
- Full agent analysis: decision banner, telemetry summary
- Action probability bars (7 actions)
- Event timeline with per-event agent decisions
- LSTM hidden state heatmap (128 units, color-coded positive/negative)

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/checkout` | Submit checkout form |
| POST | `/api/tracking/mouse` | Batch mouse samples |
| POST | `/api/tracking/clicks` | Batch click events |
| POST | `/api/tracking/keystrokes` | Batch keystroke timing |
| POST | `/api/tracking/scroll` | Batch scroll events |
| POST | `/api/agent/evaluate` | Run RL agent on session |
| GET | `/api/agent/dashboard/<sid>` | Agent analysis + LSTM state |
| GET | `/api/agent/live/<sid>` | Live telemetry counts |
| GET | `/api/agent/sessions` | Recent sessions list |
| GET | `/api/export/tracking` | Export telemetry to CSV |
| GET | `/api/export` | Export checkout data to CSV |
| GET | `/api/health` | Health check |

## Database

MySQL `ticketmonarch_db` with tables:
- **checkouts** -- Order form submissions
- **user_sessions** -- Telemetry data in JSON columns (mouse_movements, click_events, keystroke_data, scroll_events). Uses `JSON_MERGE_PRESERVE` on duplicate key to append data across flush cycles.

## Tech Stack

- **Frontend**: React 18.2, Vite 5, React Router DOM 6, vanilla CSS, UUID v4
- **Backend**: Python 3.12, Flask 3.0, Flask-CORS, mysql-connector-python, Pandas
- **Agent**: PyTorch PPO+LSTM (lazy-loaded to avoid slow startup)
- **Database**: MySQL 8.0+
