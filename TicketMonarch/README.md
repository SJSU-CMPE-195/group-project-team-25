# TicketMonarch - Full-Stack Web Application

A full-stack web application built with Flask (Python) backend, React with Vite frontend, and a MySQL database with CSV import capability.

## Project Structure

```
TicketMonarch/
├── backend/
│   ├── app.py              # Flask API server
│   ├── models.py           # (legacy) SQLAlchemy models (not used by default)
│   ├── database.py         # MySQL-based checkout & tracking storage and CSV export
│   ├── config.py           # MySQL configuration (env-based)
│   ├── setup_mysql.py      # MySQL database & orders table setup script
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── main.jsx        # Vite entry point
│   │   ├── App.css         # Application styles
│   │   └── index.css       # Global styles
│   ├── index.html          # HTML template
│   ├── vite.config.js      # Vite configuration
│   └── package.json        # Node.js dependencies
├── data/
│   └── orders.csv          # (optional) CSV file for legacy order data
└── README.md               # This file
```

## Features

- **Backend API**: Flask REST API with MySQL database
- **Frontend**: React application with Vite for fast development
- **Database**: MySQL with SQLAlchemy ORM and raw connector utilities
- **CSV Import**: Import orders from CSV file
- **CORS Enabled**: Backend configured for frontend communication
- **Modern UI**: Beautiful, responsive design

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Quick Start

Follow these steps to get the application running:

### Step 1: Install Python Dependencies

Navigate to the backend directory and install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

**Note:** It's recommended to use a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Node Dependencies

Navigate to the frontend directory and install the required Node.js packages:

```bash
cd frontend
npm install
```

### Step 3: Set Up MySQL

1. **Install MySQL Community Server** (if you don't already have it):
   - Download and install from the official MySQL website.
   - Make sure the MySQL service is running on your machine.

2. **Create a MySQL user (optional but recommended)**:
   - You can use the default `root` user, or create a dedicated user.
   - Ensure the user has permissions to create databases and tables.

3. **Configure environment variables**:
   - In the project root, copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Update the values (especially `MYSQL_PASSWORD`) to match your local MySQL setup.

4. **Run the MySQL setup script** to create the database and `orders` table:
   ```bash
   cd backend
   python setup_mysql.py
   ```

This will create the `ticketmonarch_db` database (if it doesn't exist) and an `orders` table that mirrors the existing `Order` model in `models.py`.

### Step 4: Run the Flask Backend

From the backend directory, start the Flask server:

```bash
cd backend
python app.py
```

The backend API will be running on `http://localhost:5000`

### Step 5: Run the React Frontend

From the frontend directory, start the development server:

```bash
cd frontend
npm run dev
```

The frontend will be running on `http://localhost:5173` (Vite default port) (May run on other ports, check terminal)

### Step 5: Access the Application

Open your web browser and navigate to:

**http://localhost:5173**

The application should now be running with both frontend and backend connected.

### Database Setup

The MySQL database (`ticketmonarch_db` by default) is created by `backend/setup_mysql.py` and used by both the checkout flow and telemetry tracking (`user_sessions` table).

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/checkout` - Submit checkout form data
- `GET /api/export` - Export checkout data to CSV
- `POST /api/tracking/mouse` - Submit batched mouse movement telemetry
- `POST /api/tracking/clicks` - Submit batched click telemetry
- `POST /api/tracking/keystrokes` - Submit batched keystroke timing telemetry
- `GET /api/export/tracking` - Export user session tracking data to CSV for RL/ML

## Tracking Data Export for RL/ML Training

Tracking data from `user_sessions` is exported as a CSV suitable for RL pipelines (e.g., SAC/PPO agents with PyTorch or OpenAI Gym):

- **Endpoint**: `GET /api/export/tracking`
- **Output file**: `data/tracking_sessions.csv`
- **Columns** (one row per session):
  - `session_id`, `session_start`, `page`
  - `mouse_movements` (JSON array of `{x, y, t}` samples)
  - `click_events` (JSON array of click events with timing and targets)
  - `keystroke_data` (JSON array of timing events for form fields)
  - `scroll_events`, `form_completion_time`, `browser_info`, `session_metadata` (JSON)

In your RL/ML code you can load each row, `json.loads` the JSON columns, and convert them into tensors or time-series features for training SAC/PPO agents.

## Development

### Backend Development

- The Flask server runs in debug mode by default
- Database changes are automatically reflected
- CORS is enabled for frontend communication

### Frontend Development

- Vite provides hot module replacement (HMR) for instant updates
- The frontend proxies API requests to `http://localhost:5000`
- Changes to React components are reflected immediately

## Building for Production

### Frontend

Build the React app for production:
```bash
cd frontend
npm run build
```

The production build will be in the `frontend/dist` directory.

### Backend

For production deployment:
1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Troubleshooting

### Backend Issues

- **Port 5000 already in use**: Change the port in `app.py` or stop the process using port 5000
- **Database errors**: Delete `data/ticketmonarch.db` and restart the server to recreate the database
- **Import errors**: Ensure `data/orders.csv` exists and has the correct format

### Frontend Issues

- **Port 5173 already in use**: Vite will automatically use the next available port
- **API connection errors**: Ensure the Flask server is running on port 5000
- **Module not found**: Run `npm install` again to ensure all dependencies are installed
- **CORS errors**: Make sure the backend CORS is configured for `http://localhost:5173`

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-CORS, Pandas
- **Frontend**: React, Vite
- **Database**: SQLite
- **Styling**: CSS3 with modern design patterns

## License

This project is open source and available for use.

