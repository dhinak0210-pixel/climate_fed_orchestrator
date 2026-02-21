import os
import sys


# Import Flask app
try:
    from app import app
except ImportError as e:
    print(f"ERROR: Cannot import app from app.py: {e}")
    raise

# Gunicorn looks for 'application' variable by default
application = app

if __name__ == "__main__":
    app.run()
