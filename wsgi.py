import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
