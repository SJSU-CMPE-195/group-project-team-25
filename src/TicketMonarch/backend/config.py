import os

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


def get_db_config():
    """
    Return MySQL database configuration using environment variables.

    Environment variables (with defaults):
    - MYSQL_HOST (default: localhost)
    - MYSQL_DATABASE (default: ticketmonarch_db)
    - MYSQL_USER (default: root)
    - MYSQL_PASSWORD (no default, can be empty)
    - MYSQL_PORT (default: 3306)
    """
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "database": os.getenv("MYSQL_DATABASE", "ticketmonarch_db"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
    }
