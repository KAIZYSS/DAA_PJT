"""
tests/test_api.py
=================
Integration tests for the Flask REST API endpoints.
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_locations(client):
    """Verify predefined locations endpoint returns valid list."""
    res = client.get("/api/locations")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["locations"]) >= 8
    names = [loc["name"] for loc in data["locations"]]
    assert "Chennai Central" in names
    assert "VIT Chennai" in names


def test_api_roads(client):
    """Verify roads endpoint returns complete road network."""
    res = client.get("/api/roads")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total"] >= 20
    first_road = data["roads"][0]
    assert "id" in first_road
    assert "geometry" in first_road
    assert "risk_score" in first_road


def test_api_network_summary(client):
    """Verify network summary stats."""
    res = client.get("/api/network-summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    summary = data["summary"]
    assert summary["total_segments"] >= 20
    assert summary["open_roads"] > 0
    assert "high_risk_roads" in summary


def test_api_route_valid(client):
    """Verify route calculation between Chennai Central and VIT Chennai."""
    res = client.get("/api/route?start=node_central&destination=node_vit&mode=reliable")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    route = data["route"]
    assert route["nodes"][0] == "node_central"
    assert route["nodes"][-1] == "node_vit"
    assert route["distance_km"] > 0
    assert route["travel_time_minutes"] > 0
    assert "comparison" in data
    assert "explanations" in data
    assert len(data["explanations"]) > 0


def test_api_route_missing_params(client):
    """Missing query params should yield 400 Bad Request."""
    res = client.get("/api/route")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "required" in data["error"]


def test_api_route_same_start_destination(client):
    """Identical start and destination should yield 400 Bad Request."""
    res = client.get("/api/route?start=node_central&destination=node_central")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "identical" in data["error"]


def test_api_route_invalid_node(client):
    """Non-existent node should yield 404 Not Found."""
    res = client.get("/api/route?start=node_central&destination=node_unknown_xyz")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert "Invalid destination node ID" in data["error"]


def test_api_health(client):
    """Verify health endpoint reports API and SQLite database linkage with row counts."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "healthy"
    assert data["api"] == "online"
    assert data["database"]["connected"] is True
    assert data["database"]["counts"]["nodes"] >= 19
    assert data["database"]["counts"]["roads"] >= 52
    assert data["database"]["counts"]["locations"] >= 8
    assert data["database"]["counts"]["anomalies"] >= 30


def test_api_cors_headers(client):
    """Verify CORS headers are present on API responses."""
    res = client.get("/api/locations")
    assert res.headers.get("Access-Control-Allow-Origin") == "*"

