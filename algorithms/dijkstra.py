"""
algorithms/dijkstra.py
======================
Manual Dijkstra's Single-Source Shortest Path Algorithm using a binary min-heap.

Computational Complexity:
-------------------------
Time Complexity:  O((V + E) log V)
    - Each vertex is extracted from the min-heap at most once: O(V log V)
    - Each edge relaxation updates or pushes to the heap: O(E log V)
    - Total: O((V + E) log V) where V = count of nodes, E = count of edges.
Space Complexity: O(V + E) for the adjacency list and distance map.

Architectural Design:
---------------------
Dijkstra is strictly decoupled from domain-specific edge weights. It accepts a
generic `weight_function(edge_data)` callable.

Risk & Routing Formulae:
------------------------
historical_risk: derived from district-level baselines and historical incident rates
predicted_risk:  calibrated ML model probability (0.0 to 1.0)
live_incident_risk: active live incident severity (0.0 to 1.0)
traffic_factor: live traffic delay multiplier (>= 1.0)

combined_risk = w1 * historical_risk + w2 * predicted_risk + w3 * live_incident_risk
travel_time = (distance_km / speed_kmph) * 60.0 * traffic_factor

FASTEST:  cost = travel_time
RELIABLE: cost = travel_time * (1.0 + lambda * combined_risk)
BALANCED: cost = travel_time * (1.0 + 0.5 * lambda * combined_risk)
"""

import heapq
import time
from typing import Dict, List, Tuple, Callable, Any, Optional

DEFAULT_LAMBDA = 1.0
DEFAULT_REROUTE_THRESHOLD = 0.15  # 15% cost improvement required for rerouting


def calculate_combined_risk(
    edge: Dict[str, Any],
    w1: float = 0.30,
    w2: float = 0.50,
    w3: float = 0.20
) -> float:
    """
    Calculate combined risk adhering to:
    combined_risk = w1 * historical_risk + w2 * predicted_risk + w3 * live_incident_risk
    If edge only has 'risk_score' without separate predicted_risk, uses risk_score directly.
    Ensures bounded output in [0.0, 1.0].
    """
    if "predicted_risk" in edge or "live_incident_risk" in edge:
        hist_risk = float(edge.get("historical_risk", edge.get("risk_score", 0.0)))
        pred_risk = float(edge.get("predicted_risk", 0.0))
        live_risk = float(edge.get("live_incident_risk", 0.0))
        combined = w1 * hist_risk + w2 * pred_risk + w3 * live_risk
    else:
        combined = float(edge.get("risk_score", 0.0))

    return round(max(0.0, min(1.0, combined)), 4)


def calculate_travel_time(edge: Dict[str, Any]) -> float:
    """
    Compute effective travel time in minutes considering distance, speed limit,
    and live traffic factor.
    travel_time = (distance_km / speed_kmph) * 60.0 * live_traffic_factor
    """
    distance = float(edge.get("distance_km", 1.0))
    speed = max(float(edge.get("speed_kmph", 40.0)), 5.0)
    traffic_factor = max(1.0, float(edge.get("live_traffic_factor", 1.0)))
    base_time = (distance / speed) * 60.0
    return base_time * traffic_factor


def fastest_weight(edge: Dict[str, Any], lambda_param: float = DEFAULT_LAMBDA) -> float:
    """
    Cost = Effective travel time (minutes).
    Traffic influences travel time; no risk penalty.
    """
    return calculate_travel_time(edge)


def reliable_weight(edge: Dict[str, Any], lambda_param: float = DEFAULT_LAMBDA) -> float:
    """
    Cost = Travel time penalized by combined risk.
    cost = travel_time * (1.0 + lambda * combined_risk)
    """
    travel_time = calculate_travel_time(edge)
    risk = calculate_combined_risk(edge)
    return travel_time * (1.0 + lambda_param * risk)


def balanced_weight(edge: Dict[str, Any], lambda_param: float = DEFAULT_LAMBDA) -> float:
    """
    Cost = Balanced tradeoff between travel time and road stability.
    cost = travel_time * (1.0 + 0.5 * lambda * combined_risk)
    """
    travel_time = calculate_travel_time(edge)
    risk = calculate_combined_risk(edge)
    return travel_time * (1.0 + 0.5 * lambda_param * risk)


WEIGHT_MODES: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "fastest": fastest_weight,
    "reliable": reliable_weight,
    "balanced": balanced_weight
}


def evaluate_reroute(
    current_route_cost: float,
    new_route_cost: float,
    threshold: float = DEFAULT_REROUTE_THRESHOLD
) -> Tuple[bool, float, str]:
    """
    Compare current route cost vs best newly calculated route cost.
    Only recommends rerouting when the configured threshold is exceeded.
    Returns: (should_reroute, cost_difference_pct, explanation)
    """
    if current_route_cost <= 0.0:
        return False, 0.0, "Current route cost is zero or invalid."

    diff = (current_route_cost - new_route_cost) / current_route_cost
    diff_pct = round(diff * 100.0, 2)

    if diff >= threshold:
        return True, diff_pct, (
            f"Rerouting recommended: Alternative path achieves a {diff_pct:.1f}% "
            f"cost reduction (exceeds {threshold*100.0:.0f}% threshold)."
        )
    elif diff > 0:
        return False, diff_pct, (
            f"Rerouting not recommended: Alternative path is only {diff_pct:.1f}% "
            f"faster/safer (below {threshold*100.0:.0f}% threshold)."
        )
    else:
        return False, diff_pct, "Current route remains optimal."


def dijkstra(
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    start_node: str,
    destination_node: str,
    weight_function: Callable[[Dict[str, Any]], float] = fastest_weight
) -> Tuple[Optional[List[str]], float, int, float]:
    """
    Execute manual Dijkstra's algorithm to discover the optimal path from start_node
    to destination_node using a binary min-heap priority queue.

    :param graph: Adjacency list mapping node_id -> list of (neighbor_node_id, edge_data_dict)
    :param start_node: Starting node ID
    :param destination_node: Destination node ID
    :param weight_function: Callable mapping edge_data -> non-negative cost
    :return: (path_node_ids, total_cost, visited_nodes_count, execution_time_ms)
             If no path exists, returns (None, infinity, visited_count, execution_time_ms)
    """
    start_time = time.perf_counter()

    if start_node not in graph or destination_node not in graph:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return None, float("inf"), 0, round(elapsed_ms, 3)

    if start_node == destination_node:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return [start_node], 0.0, 1, round(elapsed_ms, 3)

    pq: List[Tuple[float, str]] = []
    heapq.heappush(pq, (0.0, start_node))

    distances: Dict[str, float] = {start_node: 0.0}
    predecessors: Dict[str, str] = {}
    visited_count = 0
    closed_set = set()

    while pq:
        current_cost, u = heapq.heappop(pq)

        if u in closed_set:
            continue
        closed_set.add(u)
        visited_count += 1

        if u == destination_node:
            break

        if current_cost > distances.get(u, float("inf")):
            continue

        for v, edge_data in graph.get(u, []):
            weight = weight_function(edge_data)
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative edge weights.")

            new_cost = current_cost + weight
            if new_cost < distances.get(v, float("inf")):
                distances[v] = new_cost
                predecessors[v] = u
                heapq.heappush(pq, (new_cost, v))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if destination_node not in distances:
        return None, float("inf"), visited_count, round(elapsed_ms, 3)

    path: List[str] = []
    curr: Optional[str] = destination_node
    while curr is not None:
        path.append(curr)
        curr = predecessors.get(curr)
    path.reverse()

    return path, round(distances[destination_node], 3), visited_count, round(elapsed_ms, 3)
