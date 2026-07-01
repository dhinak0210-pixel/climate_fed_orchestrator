"""Unit tests for the Flask API routes."""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["status"] == "healthy"
    assert data["live"] is True


def test_root_redirects_to_dashboard(client):
    r = client.get("/")
    assert r.status_code in (301, 302)
    assert "/dashboard" in r.headers.get("Location", "")


def test_api_metrics_returns_json(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.content_type.startswith("application/json")
    data = json.loads(r.data)
    assert "convergence" in data or "carbon" in data


def test_api_compare_returns_live_structure(client):
    r = client.get("/api/compare")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "standard_fl" in data
    assert "carbon_aware" in data
    assert "carbon_reduction_percent" in data
    # Values must be numeric
    assert isinstance(data["standard_fl"]["accuracy"], (int, float))
    assert isinstance(data["carbon_aware"]["carbon_kg"], (int, float))


def test_run_simulation_returns_202_and_job_id(client):
    r = client.post(
        "/api/run_simulation",
        data=json.dumps({"rounds": 2}),
        content_type="application/json",
    )
    assert r.status_code == 202
    data = json.loads(r.data)
    assert data["status"] == "started"
    assert "job_id" in data


def test_job_polling_not_found(client):
    r = client.get("/api/job/nonexistent_job_999")
    assert r.status_code == 404
    data = json.loads(r.data)
    assert data["status"] == "not_found"


def test_404_handler(client):
    r = client.get("/this/does/not/exist")
    assert r.status_code == 404
    data = json.loads(r.data)
    assert "error" in data
