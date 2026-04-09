"""
MySQL setup script for TicketMonarch.

This script will:
- Connect to the MySQL server using environment-based configuration
- Create the `ticketmonarch_db` database if it does not exist
- Create the `orders` table with the same structure as the existing SQLite model

Usage (from the `backend/` directory):

    python setup_mysql.py

Ensure that MySQL is running locally and that the configured user has
permissions to create databases and tables.
"""

import re

import mysql.connector

from config import get_db_config


def create_database_if_not_exists():
    config = get_db_config()
    db_name = config["database"]

    # Validate database name to prevent injection via .env
    if not re.match(r"^[A-Za-z0-9_]+$", db_name):
        raise ValueError(f"Invalid database name: {db_name!r}")

    # Connect without specifying the database to create it if needed
    admin_config = {
        "host": config["host"],
        "user": config["user"],
        "password": config["password"],
        "port": config["port"],
    }

    conn = mysql.connector.connect(**admin_config)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4"
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def create_orders_table():
    """
    Create the `orders` table in the configured database with the same schema
    as defined in `models.py` (Order model).
    """
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL,
                product_name VARCHAR(200) NOT NULL,
                quantity INT NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                total DECIMAL(10, 2) NOT NULL,
                order_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def create_user_sessions_table():
    """
    Create the `user_sessions` table used for telemetry / interaction tracking.
    """
    config = get_db_config()
    conn = mysql.connector.connect(**config)
    try:
        cursor = conn.cursor()
        cursor.execute("""
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
            """)

        # Index on session_start for fast ORDER BY DESC LIMIT queries
        try:
            cursor.execute("""
                CREATE INDEX idx_session_start
                ON user_sessions (session_start DESC)
                """)
        except Exception:
            pass  # Index already exists

        conn.commit()
        cursor.close()
    finally:
        conn.close()


def main():
    create_database_if_not_exists()
    create_orders_table()
    create_user_sessions_table()
    print("MySQL database, `orders`, and `user_sessions` tables are ready.")


if __name__ == "__main__":
    main()
