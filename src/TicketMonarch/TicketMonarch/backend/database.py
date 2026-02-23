import csv
import json
import os
from datetime import datetime
import mysql.connector

from config import get_db_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists (used for CSV exports)
os.makedirs(DATA_DIR, exist_ok=True)


def get_connection():
    """
    Create and return a new MySQL connection using configuration
    from environment variables (see backend/config.py).
    """
    config = get_db_config()
    return mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config["port"],
    )


def init_database():
    """
    Initialize the MySQL database by ensuring required tables exist.

    Note: The MySQL database (`ticketmonarch_db` by default) should already
    exist. Use `backend/setup_mysql.py` to create the database and core tables.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Main checkout storage
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkouts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(100),
                email VARCHAR(100),
                card_number VARCHAR(32),
                card_expiry VARCHAR(10),
                card_cvv VARCHAR(10),
                billing_address VARCHAR(255),
                city VARCHAR(100),
                state VARCHAR(50),
                zip_code VARCHAR(20),
                timestamp DATETIME
            )
            """
        )

        # Telemetry / interaction tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                page VARCHAR(50),
                mouse_movements JSON,
                click_events JSON,
                keystroke_data JSON,
                scroll_events JSON,
                form_completion_time JSON,
                browser_info JSON,
                session_metadata JSON
            )
            """
        )

        conn.commit()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def save_order(order_data):
    """Save submitted form data to the MySQL `checkouts` table."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO checkouts (
                full_name,
                email,
                card_number,
                card_expiry,
                card_cvv,
                billing_address,
                city,
                state,
                zip_code,
                timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            order_data.get("full_name", ""),
            order_data.get("email", ""),
            order_data.get("card_number", ""),
            order_data.get("card_expiry", ""),
            order_data.get("card_cvv", ""),
            order_data.get("billing_address", ""),
            order_data.get("city", ""),
            order_data.get("state", ""),
            order_data.get("zip_code", ""),
            datetime.now(),
        )

        cursor.execute(insert_query, values)
        conn.commit()
        return cursor.lastrowid
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def export_to_csv():
    """
    Export all checkout data from MySQL to a CSV file in the `data/` folder.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM checkouts")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    csv_path = os.path.join(DATA_DIR, "checkouts.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(rows)

    return csv_path


def export_tracking_data_to_csv():
    """
    Export all user session telemetry data to a CSV file in the `data/` folder.

    The CSV is structured for RL/ML pipelines (e.g., PyTorch/OpenAI Gym):
    - One row per session
    - Columns:
        session_id, session_start, page,
        mouse_movements, click_events, keystroke_data,
        scroll_events, form_completion_time,
        browser_info, session_metadata
    - Sequence fields are JSON-encoded arrays that can be parsed into tensors.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                session_id,
                session_start,
                page,
                mouse_movements,
                click_events,
                keystroke_data,
                scroll_events,
                form_completion_time,
                browser_info,
                session_metadata
            FROM user_sessions
            ORDER BY session_start ASC
            """
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # Normalize JSON-like columns to plain JSON strings for CSV
    json_columns = {
        "mouse_movements",
        "click_events",
        "keystroke_data",
        "scroll_events",
        "form_completion_time",
        "browser_info",
        "session_metadata",
    }

    normalized_rows = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        for key in json_columns:
            value = row_dict.get(key)
            if value is None:
                row_dict[key] = ""
            else:
                try:
                    # If already a dict/list, dump directly; otherwise try to parse then dump
                    if isinstance(value, (dict, list)):
                        row_dict[key] = json.dumps(value)
                    else:
                        try:
                            parsed = json.loads(value)
                            row_dict[key] = json.dumps(parsed)
                        except Exception:
                            # Fallback: store as-is
                            row_dict[key] = str(value)
                except Exception:
                    row_dict[key] = str(value)

        normalized_rows.append([row_dict[col] for col in columns])

    csv_path = os.path.join(DATA_DIR, "tracking_sessions.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(normalized_rows)

    return csv_path


def save_user_session(session_id, telemetry):
    """
    Insert or update a user session telemetry snapshot in real-time.

    This function is intended to be called repeatedly as the frontend sends
    updated tracking data for a given session.

    `telemetry` is expected to be a dict that can contain:
    - page
    - mouse_movements
    - click_events
    - keystroke_data
    - scroll_events
    - form_completion_time
    - browser_info
    - session_metadata
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO user_sessions (
                session_id,
                page,
                mouse_movements,
                click_events,
                keystroke_data,
                scroll_events,
                form_completion_time,
                browser_info,
                session_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                page = VALUES(page),
                mouse_movements = CASE
                    WHEN VALUES(mouse_movements) IS NOT NULL THEN
                        JSON_MERGE_PRESERVE(
                            COALESCE(mouse_movements, JSON_ARRAY()),
                            VALUES(mouse_movements)
                        )
                    ELSE mouse_movements END,
                click_events = CASE
                    WHEN VALUES(click_events) IS NOT NULL THEN
                        JSON_MERGE_PRESERVE(
                            COALESCE(click_events, JSON_ARRAY()),
                            VALUES(click_events)
                        )
                    ELSE click_events END,
                keystroke_data = CASE
                    WHEN VALUES(keystroke_data) IS NOT NULL THEN
                        JSON_MERGE_PRESERVE(
                            COALESCE(keystroke_data, JSON_ARRAY()),
                            VALUES(keystroke_data)
                        )
                    ELSE keystroke_data END,
                scroll_events = CASE
                    WHEN VALUES(scroll_events) IS NOT NULL THEN
                        JSON_MERGE_PRESERVE(
                            COALESCE(scroll_events, JSON_ARRAY()),
                            VALUES(scroll_events)
                        )
                    ELSE scroll_events END,
                form_completion_time = VALUES(form_completion_time),
                browser_info = VALUES(browser_info),
                session_metadata = VALUES(session_metadata)
        """

        def _to_json(value):
            if value is None:
                return None
            return json.dumps(value)

        values = (
            session_id,
            telemetry.get("page"),
            _to_json(telemetry.get("mouse_movements")),
            _to_json(telemetry.get("click_events")),
            _to_json(telemetry.get("keystroke_data")),
            _to_json(telemetry.get("scroll_events")),
            _to_json(telemetry.get("form_completion_time")),
            _to_json(telemetry.get("browser_info")),
            _to_json(telemetry.get("session_metadata")),
        )

        cursor.execute(insert_query, values)
        conn.commit()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_user_session(session_id):
    """
    Retrieve a single session's telemetry data as a Python dict.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                session_id,
                session_start,
                page,
                mouse_movements,
                click_events,
                keystroke_data,
                scroll_events,
                form_completion_time,
                browser_info,
                session_metadata
            FROM user_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        row = cursor.fetchone()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if not row:
        return None

    def _parse_json_field(value):
        if value is None:
            return None
        try:
            # MySQL JSON columns may already be returned as dicts;
            # if it's a string, attempt to parse.
            if isinstance(value, (dict, list)):
                return value
            return json.loads(value)
        except Exception:
            return value

    for key in [
        "mouse_movements",
        "click_events",
        "keystroke_data",
        "scroll_events",
        "form_completion_time",
        "browser_info",
        "session_metadata",
    ]:
        row[key] = _parse_json_field(row.get(key))

    return row


def get_user_sessions(page=None, limit=100):
    """
    Retrieve multiple sessions for analysis.

    Args:
        page (str, optional): Filter by page name (e.g., 'home', 'checkout').
        limit (int): Maximum number of sessions to return.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        if page:
            cursor.execute(
                """
                SELECT
                    session_id,
                    session_start,
                    page,
                    mouse_movements,
                    click_events,
                    keystroke_data,
                    scroll_events,
                    form_completion_time,
                    browser_info,
                    session_metadata
                FROM user_sessions
                WHERE page = %s
                ORDER BY session_start DESC
                LIMIT %s
                """,
                (page, limit),
            )
        else:
            cursor.execute(
                """
                SELECT
                    session_id,
                    session_start,
                    page,
                    mouse_movements,
                    click_events,
                    keystroke_data,
                    scroll_events,
                    form_completion_time,
                    browser_info,
                    session_metadata
                FROM user_sessions
                ORDER BY session_start DESC
                LIMIT %s
                """,
                (limit,),
            )

        rows = cursor.fetchall()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _parse_json_field(value):
        if value is None:
            return None
        try:
            if isinstance(value, (dict, list)):
                return value
            return json.loads(value)
        except Exception:
            return value

    for row in rows:
        for key in [
            "mouse_movements",
            "click_events",
            "keystroke_data",
            "scroll_events",
            "form_completion_time",
            "browser_info",
            "session_metadata",
        ]:
            row[key] = _parse_json_field(row.get(key))

    return rows
