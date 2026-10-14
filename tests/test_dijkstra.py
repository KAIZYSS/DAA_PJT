"""
tests/test_dijkstra.py
======================
Unit tests for the manual Dijkstra algorithm implementation.
"""

import pytest
from algorithms.dijkstra import dijkstra, fastest_weight, reliable_weight, balanced_weight


@pytest.fixture
def sample_graph():
    """
    A 4-node controlled diamond graph:
         (A)
       /     \
      v       v
     (B)     (C)
      \       /
       v     v
         (D)

    Path 1 (A -> B -> D): Short distance (10 km), high speed (60 km/h) = 10 min.
                          Heavy risk (risk_score = 0.9).
    Path 2 (A -> C -> D): Longer distance (14 km), lower speed (42 km/h) = 20 min.
                          Low risk (risk_score = 0.05).
    """
    return {
        "A": [
            ("B", {"distance_km": 5.0, "speed_kmph": 60.0, "risk_score": 0.9}),
            ("C", {"distance_km": 7.0, "speed_kmph": 42.0, "risk_score": 0.05})
        ],
        "B": [
            ("D", {"distance_km": 5.0, "speed_kmph": 60.0, "risk_score": 0.9})
        ],
        "C": [
            ("D", {"distance_km": 7.0, "speed_kmph": 42.0, "risk_score": 0.05})
        ],
        "D": []
    }


def test_dijkstra_fastest_mode(sample_graph):
    """Fastest mode should choose Path 1 (A -> B -> D) because nominal travel time is 10 min vs 20 min."""
    path, cost, visited, exec_ms = dijkstra(sample_graph, "A", "D", fastest_weight)
    assert path == ["A", "B", "D"]
    assert cost == pytest.approx(10.0, 0.01)
    assert visited > 0
    assert exec_ms >= 0


def test_dijkstra_reliable_mode(sample_graph):
    """
    Reliable mode penalizes risk:
    Path 1 cost = 10 * (1 + 0.9) = 19.0 min
    Wait, let's check edge by edge:
    A->B: 5/60*60 = 5 min * (1 + 0.9) = 9.5
    B->D: 5 min * (1 + 0.9) = 9.5 -> total = 19.0
    Path 2:
    A->C: 7/42*60 = 10 min * (1 + 0.05) = 10.5
    C->D: 10 min * (1 + 0.05) = 10.5 -> total = 21.0
    Let's adjust risk or time so Reliable strictly prefers Path 2:
    """
    graph = {
        "A": [
            ("B", {"distance_km": 5.0, "speed_kmph": 60.0, "risk_score": 0.9}),
            ("C", {"distance_km": 6.0, "speed_kmph": 50.0, "risk_score": 0.0})
        ],
        "B": [
            ("D", {"distance_km": 5.0, "speed_kmph": 60.0, "risk_score": 0.9})
        ],
        "C": [
            ("D", {"distance_km": 6.0, "speed_kmph": 50.0, "risk_score": 0.0})
        ],
        "D": []
    }
    # Path 1 fastest time: 10 min. Risk 0.9 -> Reliable cost = 10 * 1.9 = 19.0 min.
    # Path 2 fastest time: 2 * (6/50*60) = 14.4 min. Risk 0.0 -> Reliable cost = 14.4 min.
    p_fast, c_fast, _, _ = dijkstra(graph, "A", "D", fastest_weight)
    p_rel, c_rel, _, _ = dijkstra(graph, "A", "D", reliable_weight)

    assert p_fast == ["A", "B", "D"]
    assert c_fast == pytest.approx(10.0, 0.01)
    assert p_rel == ["A", "C", "D"]
    assert c_rel == pytest.approx(14.4, 0.01)


def test_dijkstra_same_start_destination(sample_graph):
    """Start equals destination should return single node path with zero cost."""
    path, cost, visited, exec_ms = dijkstra(sample_graph, "A", "A", fastest_weight)
    assert path == ["A"]
    assert cost == 0.0


def test_dijkstra_disconnected_nodes():
    """Graph with unreachable target should return None path and infinite cost."""
    disconnected_graph = {
        "A": [("B", {"distance_km": 2.0, "speed_kmph": 40.0, "risk_score": 0.1})],
        "B": [],
        "Z": []
    }
    path, cost, visited, exec_ms = dijkstra(disconnected_graph, "A", "Z", fastest_weight)
    assert path is None
    assert cost == float("inf")


def test_dijkstra_unknown_node(sample_graph):
    """Querying nodes not present in graph returns None."""
    path, cost, _, _ = dijkstra(sample_graph, "A", "NON_EXISTENT", fastest_weight)
    assert path is None
    assert cost == float("inf")
