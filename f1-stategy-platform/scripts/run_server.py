"""Production WSGI Server runner for F1 Telemetry Platform."""

import os
import sys
from pathlib import Path

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import app, clickable

def display_url(host, port):
    # 0.0.0.0 / :: are bind addresses, not addresses a browser can open —
    # show localhost so the printed link is actually clickable.
    if host in ('0.0.0.0', '::'):
        host = 'localhost'
    return f"http://{host}:{port}"

def run():
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    url = clickable(display_url(host, port))

    try:
        from waitress import serve
        print("=" * 60)
        print(f"F1 DIGITAL PIT WALL — Production Server (Waitress WSGI)")
        print(f"Listening on {url}")
        print("=" * 60)
        serve(app, host=host, port=port)
    except ImportError:
        print("=" * 60)
        print(f"F1 DIGITAL PIT WALL — Development Server (Flask)")
        print(f"Listening on {url}")
        print("=" * 60)
        app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run()
