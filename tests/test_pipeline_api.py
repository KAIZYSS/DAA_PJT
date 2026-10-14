"""
tests/test_pipeline_api.py
==========================
Tests for ROUTEWISE Data & ML REST API endpoints:
- GET /api/data/sources
- GET /api/data/status
- GET /api/anomalies
- GET /api/road-risk/<segment_id>
- POST /api/predict-risk
- POST /api/route
- POST /api/reroute
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_data_sources(client):
    """Verify data sources registry endpoint returns provenance and status."""
    resp = client.get("/api/data/sources")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "sources" in data
    assert len(data["sources"]) >= 5

    source_ids = {s["id"] for s in data["sources"]}
    assert "tn_district_accidents" in source_ids
    assert "adas_mobility_tn" in source_ids
    assert "osm_road_network" in source_ids


def test_api_data_status(client):
    """Verify pipeline health and database status endpoint."""
    resp = client.get("/api/data/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["status"] == "OPERATIONAL"
    assert "database" in data
    assert "ml_model" in data
    assert data["database"]["osm_road_segments"] > 0


def test_api_anomalies_filtered(client):
    """Verify anomaly filtering by source and type."""
    resp = client.get("/api/anomalies?limit=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "anomalies" in data


def test_api_road_risk_detail(client):
    """Verify road segment risk decomposition endpoint."""
    resp = client.get("/api/road-risk/road_001")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["segment_id"] == "road_001"
    assert "risk_decomposition" in data
    decomp = data["risk_decomposition"]
    assert "historical_risk" in decomp
    assert "predicted_risk" in decomp
    assert "live_traffic_factor" in decomp
    assert "combined_risk" in decomp
    assert "provenance" in data


def test_api_predict_risk_post(client):
    """Verify ML inference endpoint with feature vector."""
    features = {
        "road_type_enc": 3,
        "length_m": 2500.0,
        "lanes": 6,
        "speed_limit": 60.0,
        "district_baseline_risk": 0.5,
        "district_fatality_rate": 0.22,
        "fcw_count": 2,
        "pcw_count": 3,
        "ldw_count": 8,
        "hmw_count": 15,
        "total_adas_warnings": 28,
        "adas_warning_density": 11.2
    }
    resp = client.post(
        "/api/predict-risk",
        data=json.dumps(features),
        content_type="application/json"
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert 0.0 <= data["prediction"]["predicted_risk"] <= 1.0


def test_api_route_post(client):
    """Verify POST /api/route endpoint with custom weights and lambda."""
    payload = {
        "start": "node_central",
        "destination": "node_vit",
        "mode": "reliable",
        "lambda": 1.2
    }
    resp = client.post(
        "/api/route",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["route"]["nodes"][0] == "node_central"
    assert data["route"]["nodes"][-1] == "node_vit"
    assert "data_provenance" in data["route"]


def test_api_reroute_post(client):
    """Verify POST /api/reroute endpoint with dynamic traffic update."""
    payload = {
        "start": "node_central",
        "destination": "node_vit",
        "current_path": ["node_central", "node_egmore", "node_t_nagar"],
        "mode": "reliable",
        "threshold": 0.15,
        "updated_traffic": [
            {"segment_id": "road_001", "traffic_factor": 2.0}
        ]
    }
    resp = client.post(
        "/api/reroute",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "should_reroute" in data
    assert "cost_difference_pct" in data
    assert "explanation" in data
