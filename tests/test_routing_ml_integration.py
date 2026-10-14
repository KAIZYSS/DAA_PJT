"""
tests/test_routing_ml_integration.py
====================================
Tests for Dijkstra routing integration with ML predicted risk and dynamic rerouting.
Verifies:
- Combined risk formula: w1*hist + w2*pred + w3*live
- Travel time scaling with live_traffic_factor
- Route divergence between Fastest and Reliable modes
- Dynamic rerouting threshold (> 15% required)
"""

import pytest
from algorithms.dijkstra import (
    dijkstra, fastest_weight, reliable_weight, balanced_weight,
    calculate_combined_risk, calculate_travel_time, evaluate_reroute
)


def test_combined_risk_calculation():
    """Verify combined risk calculation weights and bounding."""
    edge = {
        "historical_risk": 0.40,
        "predicted_risk": 0.60,
        "live_incident_risk": 0.50
    }
    # 0.3 * 0.4 + 0.5 * 0.6 + 0.2 * 0.5 = 0.12 + 0.30 + 0.10 = 0.52
    comb = calculate_combined_risk(edge, w1=0.30, w2=0.50, w3=0.20)
    assert comb == pytest.approx(0.52, 0.001)

    # Upper bound check
    edge_max = {"historical_risk": 1.0, "predicted_risk": 1.0, "live_incident_risk": 1.0}
    assert calculate_combined_risk(edge_max) <= 1.0


def test_travel_time_with_live_traffic():
    """Verify live_traffic_factor scales travel time without mutating distance or speed."""
    edge_normal = {"distance_km": 10.0, "speed_kmph": 60.0, "live_traffic_factor": 1.0}
    t_normal = calculate_travel_time(edge_normal)
    assert t_normal == pytest.approx(10.0, 0.01)  # 10 min

    edge_congested = {"distance_km": 10.0, "speed_kmph": 60.0, "live_traffic_factor": 1.8}
    t_congested = calculate_travel_time(edge_congested)
    assert t_congested == pytest.approx(18.0, 0.01)  # 18 min


def test_dijkstra_ml_mode_divergence():
    """
    Verify that Reliable mode avoids a corridor with high ML predicted risk,
    while Fastest mode selects it if transit duration is shorter.
    """
    graph = {
        "A": [
            # High speed (60 kmph), short distance (6 km), but HIGH ML predicted risk (0.85)
            ("B", {
                "distance_km": 6.0, "speed_kmph": 60.0,
                "historical_risk": 0.2, "predicted_risk": 0.85,
                "live_incident_risk": 0.0, "live_traffic_factor": 1.0
            }),
            # Slower (45 kmph), longer distance (7.5 km), but LOW ML predicted risk (0.05)
            ("C", {
                "distance_km": 7.5, "speed_kmph": 45.0,
                "historical_risk": 0.05, "predicted_risk": 0.05,
                "live_incident_risk": 0.0, "live_traffic_factor": 1.0
            })
        ],
        "B": [
            ("D", {
                "distance_km": 6.0, "speed_kmph": 60.0,
                "historical_risk": 0.2, "predicted_risk": 0.85,
                "live_incident_risk": 0.0, "live_traffic_factor": 1.0
            })
        ],
        "C": [
            ("D", {
                "distance_km": 7.5, "speed_kmph": 45.0,
                "historical_risk": 0.05, "predicted_risk": 0.05,
                "live_incident_risk": 0.0, "live_traffic_factor": 1.0
            })
        ],
        "D": []
    }

    # Fastest mode optimizes travel time: Path A-B-D takes 12 min vs A-C-D 20 min
    p_fast, c_fast, _, _ = dijkstra(graph, "A", "D", fastest_weight)
    assert p_fast == ["A", "B", "D"]
    assert c_fast == pytest.approx(12.0, 0.1)

    # Reliable mode penalizes high predicted risk on A-B-D:
    # A-B-D combined risk = 0.3*0.2 + 0.5*0.85 = 0.485 -> Cost = 12 * (1 + 1.5 * 0.485) = 20.73
    # A-C-D combined risk = 0.3*0.05 + 0.5*0.05 = 0.04 -> Cost = 20 * (1 + 1.5 * 0.04) = 21.2
    # With higher lambda (e.g. 2.0), Reliable strictly diverts to A-C-D:
    def high_rel_weight(e):
        return reliable_weight(e, lambda_param=2.5)

    p_rel, c_rel, _, _ = dijkstra(graph, "A", "D", high_rel_weight)
    assert p_rel == ["A", "C", "D"]


def test_evaluate_reroute_threshold():
    """Verify reroute evaluation requires exceeding configured threshold (15%)."""
    # 20% improvement -> should recommend rerouting
    should, pct, _ = evaluate_reroute(current_route_cost=25.0, new_route_cost=20.0, threshold=0.15)
    assert should is True
    assert pct == 20.0

    # 10% improvement -> below 15% threshold -> should NOT recommend rerouting
    should_not, pct_low, _ = evaluate_reroute(current_route_cost=20.0, new_route_cost=18.0, threshold=0.15)
    assert should_not is False
    assert pct_low == 10.0
