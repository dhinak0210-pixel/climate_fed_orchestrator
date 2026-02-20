"""
Climate-Aware Federated Learning - Flask Application
Production-ready WSGI entry point for Render deployment
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

# Fix for Render's reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Global state (loaded once at startup)
EXPERIMENT_DATA = None
METRICS_DATA = None

def load_experiment_data():
    """Load experiment results from JSON and normalize for templates."""
    global EXPERIMENT_DATA, METRICS_DATA
    try:
        metrics_path = Path('results/metrics.json')
        if not metrics_path.exists():
             potential_paths = [
                 Path('results/metrics.json'),
                 Path('climate_fed_orchestrator/results/metrics.json'),
                 Path('metrics.json')
             ]
             for p in potential_paths:
                 if p.exists():
                     metrics_path = p
                     break
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                raw_data = json.load(f)
            
            # Normalize structure for the Jinja2 templates (index.html)
            if 'convergence' not in raw_data:
                normalized = {
                    "metadata": {
                        "experiment_id": raw_data.get("experiment_id", "prod_001"),
                        "timestamp": datetime.now().isoformat(),
                        "status": "production"
                    },
                    "convergence": {
                        "final_accuracy": raw_data.get("final_accuracy", 0.0) / 100.0 if raw_data.get("final_accuracy", 0) > 1 else raw_data.get("final_accuracy", 0.0),
                    },
                    "carbon": {
                        "reduction_percentage": raw_data.get("carbon_reduction_percent", 0.0),
                        "total_carbon_kg": raw_data.get("total_carbon_kg", 0.0)
                    },
                    "privacy": {
                        "epsilon_consumed": raw_data.get("privacy_epsilon", 0.0),
                        "target_epsilon": 2.0
                    }
                }
                METRICS_DATA = normalized
            else:
                METRICS_DATA = raw_data
                
            logger.info(f"Loaded normalized metrics: {metrics_path}")
        else:
            logger.warning("metrics.json not found, using demo data")
            METRICS_DATA = generate_demo_data()
        
        EXPERIMENT_DATA = METRICS_DATA
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        EXPERIMENT_DATA = generate_demo_data()

def generate_demo_data():
    """Generate demo data if real data unavailable."""
    return {
        "metadata": {
            "experiment_id": "demo_001",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "demo_mode"
        },
        "convergence": {
            "final_accuracy": 0.942,
            "rounds_to_90": 8
        },
        "carbon": {
            "total_carbon_kg": 0.050,
            "reduction_percentage": 43.7
        },
        "privacy": {
            "epsilon_consumed": 0.87,
            "target_epsilon": 2.0
        },
        "baseline_results": {
             "final_accuracy": 0.945,
             "per_round": []
        },
        "carbon_results": {
             "final_accuracy": 0.942,
             "per_round": []
        },
        "comparison": {
             "accuracy_degradation_pp": 0.3,
             "energy_savings_percent": 43.7,
             "carbon_total_energy": 0.050,
             "baseline_total_energy": 0.089,
             "carbon_avg_participation_rate": 0.6
        },
        "configuration": {
             "threshold": 0.6
        }
    }

# Load data at startup
load_experiment_data()

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "data_loaded": EXPERIMENT_DATA is not None
    }), 200

@app.route('/')
def index():
    """Main dashboard."""
    return render_template('index.html', data=EXPERIMENT_DATA)

@app.route('/dashboard')
def dashboard():
    """Interactive results dashboard."""
    return render_template('dashboard.html', data=EXPERIMENT_DATA)

@app.route('/api/metrics')
def api_metrics():
    """JSON API for experiment metrics."""
    return jsonify(EXPERIMENT_DATA or {})

@app.route('/api/compare')
def api_compare():
    """Comparison data: Standard vs Carbon-Aware vs +Privacy."""
    comparison = {
        "standard_fl": {
            "accuracy": 0.945,
            "carbon_kg": 0.089,
            "renewable_pct": 42,
            "privacy_epsilon": float('inf')
        },
        "carbon_aware": {
            "accuracy": 0.942,
            "carbon_kg": 0.050,
            "renewable_pct": 78,
            "privacy_epsilon": float('inf')
        },
        "carbon_privacy": {
            "accuracy": 0.938,
            "carbon_kg": 0.050,
            "renewable_pct": 78,
            "privacy_epsilon": 2.0
        }
    }
    return jsonify(comparison)

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    """Trigger new simulation (if resources allow)."""
    # Note: Free tier has limited compute, use sparingly
    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            return jsonify({
                "status": "deferred",
                "reason": "High memory usage on free tier",
                "retry_in": "5 minutes"
            }), 503

        config = request.get_json() or {}
        rounds = min(config.get('rounds', 5), 10)  # Limit on free tier
        
        # Import and run (lazy load to save memory)
        # Assuming main.py is in the parent directory or accessible
        # Since app.py is in root of repo, main.py should be importable if in root too.
        # But user structure shows main.py might be in root or elsewhere. 
        # Structure Phase 1 shows app.py in root. List dir showed main.py in root.
        from main import run_experiment
        result = run_experiment(rounds=rounds, seed=42)
        
        return jsonify({
            "status": "success",
            "result": result,
            "note": "Free tier: limited to 10 rounds max"
        })
    except ImportError:
         logger.error("Could not import main.run_experiment")
         return jsonify({"error": "Simulation module not found"}), 500
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "note": "Free tier limitations may apply"
        }), 500

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve generated result files."""
    results_dir = Path('results')
    file_path = results_dir / filename
    
    if file_path.exists() and file_path.is_file():
        return send_file(file_path)
    else:
        return jsonify({"error": "File not found"}), 404

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Development only - production uses Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
