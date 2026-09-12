"""Central configuration and database connection manager for F1 Telemetry Platform.

Never commit real credentials.  Every value is read from environment
variables (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT); the defaults
below are structural placeholders only.  Set DB_PASSWORD (and the others
if they differ from the defaults) before running anything.
"""

import os
import mysql.connector

# Placeholder default -- replace it via the DB_PASSWORD environment variable.
# The app refuses to connect while this placeholder is in effect, so a
# forgotten setup fails loudly instead of guessing a password.
_PLACEHOLDER_PASSWORD = 'CHANGE_ME'

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'password'),
    'database': os.environ.get('DB_NAME', 'f1_strategy'),
    'port': int(os.environ.get('DB_PORT', 3306)),
}


def get_db_connection():
    """Return a MySQL database connection using DB_CONFIG.

    Raises RuntimeError while the password is still the placeholder, so
    real credentials can never be silently replaced by a guess.
    """
    if DB_CONFIG['password'] == _PLACEHOLDER_PASSWORD:
        raise RuntimeError(
            "MySQL password is still the placeholder 'CHANGE_ME' -- set the "
            "DB_PASSWORD environment variable before running. "
            "(Other values can be overridden with DB_HOST / DB_USER / "
            "DB_NAME / DB_PORT; see the README.)"
        )
    return mysql.connector.connect(**DB_CONFIG)
