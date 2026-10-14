"""
app.py
======
ROUTEWISE — Reliable Route Advisor (Review 1 Prototype)
Flask REST API server and routing engine.
"""

import os
import sys
from typing import Dict, List, Any, Tuple
from flask import Flask, jsonify, request, render_template

# Ensure local project modules can be resolved
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.database import (
    get_all_nodes,
    get_all_roads,
    get_road_by_id,
    get_anomalies_for_road,
    get_locations,
    get_network_summary,
    ensure_database_initialized,
    check_database_health
)
from algorithms.dijkstra import (
    dijkstra,
    WEIGHT_MODES,
    fastest_weight,
    reliable_weight,
    balanced_weight,
    calculate_combined_risk,
    calculate_travel_time,
    evaluate_reroute,
    DEFAULT_LAMBDA,
    DEFAULT_REROUTE_THRESHOLD
)
from reliability.scoring import get_risk_tier
from backend.data_pipeline.storage.database import (
    init_pipeline_tables,
    get_all_data_sources,
    get_all_road_segments,
    get_road_segment_by_id as get_pipeline_road_segment,
    get_anomalies_filtered,
    update_segment_live_traffic,
    get_connection as get_pipeline_conn
)
from backend.data_pipeline.models.predict import predict_segment_risk, get_model
from backend.data_pipeline.collectors.google_traffic import GoogleTrafficCollector

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Ensure SQLite database is linked and populated on startup
ensure_database_initialized()
init_pipeline_tables()


@app.after_request
def add_cors_headers(response):
    """Enable CORS so frontend can communicate with API across origins/ports."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def api_options(path):
    """Handle preflight requests for any API endpoint."""
    return "", 204



def build_graph(roads: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Build directed adjacency list graph from road records."""
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for r in roads:
        u = r["from_node"]
        v = r["to_node"]
        if u not in graph:
            graph[u] = []
        graph[u].append((v, r))
    return graph


def get_edge_between(graph: Dict[str, List[Tuple[str, Dict[str, Any]]]], u: str, v: str) -> Dict[str, Any]:
    """Find edge dictionary between two adjacent nodes."""
    for neighbor, edge in graph.get(u, []):
        if neighbor == v:
            return edge
    return {}


def assemble_route_details(
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    path_nodes: List[str],
    mode: str
) -> Dict[str, Any]:
    """
    Given an ordered sequence of node IDs, assemble the path's composite
    geometry, distance, travel time, and risk/reliability statistics.
    Distinguishes historical, ML predicted, live traffic, and combined factors.
    """
    total_dist = 0.0
    base_total_time = 0.0
    effective_total_time = 0.0
    risk_scores = []
    hist_risks = []
    pred_risks = []
    traffic_factors = []
    combined_risks = []
    edges = []
    full_geometry = []
    high_risk_count = 0
    flagged_count = 0

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edge = get_edge_between(graph, u, v)
        if not edge:
            continue

        edges.append(edge)
        dist = edge.get("distance_km", 0.0)
        speed = max(edge.get("speed_kmph", 40.0), 5.0)
        base_time_min = (dist / speed) * 60.0
        tf = max(1.0, float(edge.get("live_traffic_factor", 1.0)))
        eff_time_min = base_time_min * tf

        h_risk = float(edge.get("risk_score", 0.0))
        p_risk = float(edge.get("predicted_risk", 0.0))
        c_risk = calculate_combined_risk(edge)
        status = edge.get("status", "open")

        total_dist += dist
        base_total_time += base_time_min
        effective_total_time += eff_time_min

        risk_scores.append(h_risk)
        hist_risks.append(h_risk)
        pred_risks.append(p_risk)
        traffic_factors.append(tf)
        combined_risks.append(c_risk)

        if c_risk >= 0.50 or h_risk >= 0.67:
            high_risk_count += 1
        if status != "open":
            flagged_count += 1

        geom = edge.get("geometry", [])
        if full_geometry and geom:
            # Skip the first point if it's identical to the previous edge's last point
            full_geometry.extend(geom[1:] if geom[0] == full_geometry[-1] else geom)
        else:
            full_geometry.extend(geom)

    n_edges = max(len(edges), 1)
    avg_hist_risk = sum(hist_risks) / n_edges
    avg_pred_risk = sum(pred_risks) / n_edges
    avg_traffic_factor = sum(traffic_factors) / n_edges
    avg_comb_risk = sum(combined_risks) / n_edges

    reliability = max(0.0, min(1.0, 1.0 - avg_comb_risk))
    reliability_pct = round(reliability * 100.0, 1)

    mode_descriptions = {
        "reliable": "Recommended route prioritizes reliability, penalizing ML predicted risk and incident exposure.",
        "fastest": "Route optimized purely for travel time considering live traffic speeds on major arterials.",
        "balanced": "Route balances travel time with ML-predicted road stability."
    }

    is_live_traffic = GoogleTrafficCollector().is_live_connected

    return {
        "nodes": path_nodes,
        "geometry": full_geometry,
        "distance_km": round(total_dist, 1),
        "travel_time_minutes": round(effective_total_time, 1),
        "base_travel_time_minutes": round(base_total_time, 1),
        "reliability": round(reliability, 3),
        "reliability_pct": reliability_pct,
        "risk": round(avg_comb_risk, 3),
        "historical_risk": round(avg_hist_risk, 3),
        "predicted_risk": round(avg_pred_risk, 3),
        "live_traffic_factor": round(avg_traffic_factor, 2),
        "risk_tier": get_risk_tier(avg_comb_risk).upper(),
        "segments_count": len(edges),
        "high_risk_segments": high_risk_count,
        "flagged_segments": flagged_count,
        "mode": mode,
        "mode_description": mode_descriptions.get(mode, "Custom route optimization."),
        "data_provenance": {
            "historical_accidents": "HISTORICAL (TN District Accidents & OpenCity)",
            "predicted_risk": "PREDICTED (ML Random Forest on ADAS Mobility Signals)",
            "live_traffic": "LIVE (Google Maps Platform)" if is_live_traffic else "DEMO (Time-of-day simulation)",
            "routing_decision": "OUR CUSTOM DIJKSTRA"
        }
    }



def generate_why_this_route(
    selected_route: Dict[str, Any],
    fastest_route: Dict[str, Any]
) -> List[str]:
    """
    Generate dynamic bullet points explaining why the chosen route was recommended
    over the default fastest route (Spec Section 24).
    """
    bullets = []
    sel_high = selected_route.get("high_risk_segments", 0)
    fast_high = fastest_route.get("high_risk_segments", 0)

    # 1. Avoided high-risk segments
    avoided_high = fast_high - sel_high
    if avoided_high > 0:
        bullets.append(f"Avoids {avoided_high} high-risk corridor{'s' if avoided_high > 1 else ''} with severe incident history")
    elif sel_high == 0:
        bullets.append("Zero high-risk road segments encountered along this path")

    # 2. Reliability improvement
    sel_rel = selected_route.get("reliability_pct", 0)
    fast_rel = fastest_route.get("reliability_pct", 0)
    diff_rel = round(sel_rel - fast_rel, 1)
    if diff_rel > 0:
        bullets.append(f"+{diff_rel}% higher reliability rating than nominal fastest corridor")
    else:
        bullets.append(f"High historical route stability ({sel_rel}% reliability score)")

    # 3. Status checks
    sel_flagged = selected_route.get("flagged_segments", 0)
    if sel_flagged == 0:
        bullets.append("No active road closures, construction blockades, or flagged hazards")
    else:
        bullets.append(f"Minimizes active road flags ({sel_flagged} monitored segment{'s' if sel_flagged > 1 else ''})")

    # 4. Fallback/Context note
    if selected_route.get("mode") == "fastest":
        bullets.append("Selects highest-speed arterials to minimize estimated transit duration")
    elif selected_route.get("mode") == "balanced":
        bullets.append("Optimally trades minor detour distance for significant hazard avoidance")

    return bullets[:4]


# -----------------------------------------------------------------------------
# Web Page Route
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main ROUTEWISE dashboard."""
    return render_template("index.html")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/locations", methods=["GET"])
def api_locations():
    """Predefined locations with coordinates for autocomplete and map pins."""
    try:
        locations = get_locations()
        return jsonify({"success": True, "locations": locations})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to fetch locations: {str(e)}"}), 500


@app.route("/api/roads", methods=["GET"])
def api_roads():
    """Full road network layer for map overlays."""
    try:
        roads = get_all_roads()
        return jsonify({"success": True, "roads": roads, "total": len(roads)})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to fetch road network: {str(e)}"}), 500


@app.route("/api/roads/<road_id>", methods=["GET"])
def api_road_detail(road_id: str):
    """Detailed road info, reliability percentage, and anomaly history."""
    try:
        road = get_road_by_id(road_id)
        if not road:
            return jsonify({"success": False, "error": f"Road '{road_id}' not found."}), 404

        anomalies = get_anomalies_for_road(road_id)
        risk = road.get("risk_score", 0.0)
        reliability_pct = round((1.0 - risk) * 100.0, 1)

        return jsonify({
            "success": True,
            "road": {
                **road,
                "reliability_pct": reliability_pct,
                "risk_tier": get_risk_tier(risk).upper(),
                "anomalies": anomalies,
                "anomaly_count": len(anomalies)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to fetch road details: {str(e)}"}), 500


@app.route("/api/route", methods=["GET", "POST"])
def api_route():
    """
    Compute optimal route between start and destination nodes using our custom Dijkstra.
    Accepts GET query params or POST JSON payload:
      start: node_id
      destination: node_id
      mode: fastest | reliable | balanced (default: reliable)
      lambda: risk penalty multiplier (default 1.0)
    """
    if request.method == "POST":
        body = request.get_json() or {}
        start = str(body.get("start", "")).strip()
        dest = str(body.get("destination", "")).strip()
        mode = str(body.get("mode", "reliable")).strip().lower()
        lambda_val = float(body.get("lambda", DEFAULT_LAMBDA))
    else:
        start = request.args.get("start", "").strip()
        dest = request.args.get("destination", "").strip()
        mode = request.args.get("mode", "reliable").strip().lower()
        lambda_val = float(request.args.get("lambda", DEFAULT_LAMBDA))

    if not start or not dest:
        return jsonify({
            "success": False,
            "error": "Both 'start' and 'destination' parameters are required."
        }), 400

    if start == dest:
        return jsonify({
            "success": False,
            "error": "Start and destination locations cannot be identical."
        }), 400

    if mode not in WEIGHT_MODES:
        return jsonify({
            "success": False,
            "error": f"Invalid mode '{mode}'. Choose from 'fastest', 'reliable', or 'balanced'."
        }), 400

    try:
        nodes = {n["id"]: n for n in get_all_nodes()}
        if start not in nodes:
            return jsonify({"success": False, "error": f"Invalid start node ID '{start}'."}), 404
        if dest not in nodes:
            return jsonify({"success": False, "error": f"Invalid destination node ID '{dest}'."}), 404

        roads = get_all_roads()
        graph = build_graph(roads)

        # Mode weight functions with lambda parameter
        mode_weight_fns = {
            "fastest": lambda e: fastest_weight(e, lambda_val),
            "reliable": lambda e: reliable_weight(e, lambda_val),
            "balanced": lambda e: balanced_weight(e, lambda_val)
        }

        # 1. Compute route for the requested mode
        weight_fn = mode_weight_fns[mode]
        path, cost, visited, exec_ms = dijkstra(graph, start, dest, weight_fn)

        if not path:
            return jsonify({
                "success": False,
                "error": "No viable path exists between the specified locations (graph disconnect)."
            }), 404

        selected_details = assemble_route_details(graph, path, mode)
        selected_details["execution_time_ms"] = exec_ms
        selected_details["visited_nodes_count"] = visited
        selected_details["dijkstra_cost"] = round(cost, 2)

        # 2. Compute comparison routes (Fastest and Reliable) for Route Comparison panel
        fastest_path, f_cost, _, _ = dijkstra(graph, start, dest, mode_weight_fns["fastest"])
        reliable_path, r_cost, _, _ = dijkstra(graph, start, dest, mode_weight_fns["reliable"])

        fastest_details = assemble_route_details(graph, fastest_path, "fastest") if fastest_path else selected_details
        reliable_details = assemble_route_details(graph, reliable_path, "reliable") if reliable_path else selected_details

        # Generate dynamic "Why This Route?" explanation
        explanations = generate_why_this_route(selected_details, fastest_details)

        return jsonify({
            "success": True,
            "route": selected_details,
            "comparison": {
                "fastest": {
                    "mode": "fastest",
                    "title": "Fastest Route",
                    "travel_time_minutes": fastest_details["travel_time_minutes"],
                    "reliability_pct": fastest_details["reliability_pct"],
                    "distance_km": fastest_details["distance_km"],
                    "geometry": fastest_details["geometry"],
                    "nodes": fastest_details["nodes"]
                },
                "recommended": {
                    "mode": "reliable",
                    "title": "Recommended Route",
                    "travel_time_minutes": reliable_details["travel_time_minutes"],
                    "reliability_pct": reliable_details["reliability_pct"],
                    "distance_km": reliable_details["distance_km"],
                    "geometry": reliable_details["geometry"],
                    "nodes": reliable_details["nodes"]
                }
            },
            "explanations": explanations
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Internal server routing error: {str(e)}"}), 500


@app.route("/api/reroute", methods=["POST"])
def api_reroute():
    """
    Dynamic Rerouting Evaluation Endpoint.
    Accepts:
      start: str
      destination: str
      current_path: List[str]
      mode: fastest | reliable | balanced
      threshold: float (default 0.15 for 15% cost improvement requirement)
      updated_traffic: List[{"segment_id": str, "traffic_factor": float}]
    """
    try:
        data = request.get_json() or {}
        start = data.get("start")
        dest = data.get("destination")
        current_path = data.get("current_path", [])
        mode = data.get("mode", "reliable").lower()
        threshold = float(data.get("threshold", DEFAULT_REROUTE_THRESHOLD))
        updated_traffic = data.get("updated_traffic", [])

        if not start or not dest:
            return jsonify({"success": False, "error": "Both 'start' and 'destination' are required."}), 400

        # Apply any live traffic updates to the road segments in database
        for item in updated_traffic:
            sid = item.get("segment_id")
            tf = item.get("traffic_factor")
            if sid and tf is not None:
                update_segment_live_traffic(sid, float(tf))

        roads = get_all_roads()
        graph = build_graph(roads)

        weight_fn = WEIGHT_MODES.get(mode, reliable_weight)

        # 1. Compute current path cost under updated conditions
        current_cost = 0.0
        if current_path and len(current_path) >= 2:
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i + 1]
                edge = get_edge_between(graph, u, v)
                if edge:
                    current_cost += weight_fn(edge)
                else:
                    current_cost = float("inf")
                    break

        # 2. Re-run our Dijkstra to find best newly calculated path
        new_path, new_cost, visited, exec_ms = dijkstra(graph, start, dest, weight_fn)

        if not new_path:
            return jsonify({"success": False, "error": "No route exists under current conditions."}), 404

        if not current_path or current_cost == float("inf"):
            current_cost = new_cost
            current_path = new_path

        should_reroute, diff_pct, explanation = evaluate_reroute(current_cost, new_cost, threshold=threshold)

        new_route_details = assemble_route_details(graph, new_path, mode)
        new_route_details["execution_time_ms"] = exec_ms
        new_route_details["visited_nodes_count"] = visited

        return jsonify({
            "success": True,
            "should_reroute": should_reroute,
            "cost_difference_pct": diff_pct,
            "threshold_pct": round(threshold * 100.0, 1),
            "explanation": explanation,
            "current_route": {
                "nodes": current_path,
                "cost": round(current_cost, 2)
            },
            "new_optimal_route": {
                "nodes": new_path,
                "cost": round(new_cost, 2),
                "route_details": new_route_details
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/data/sources", methods=["GET"])
def api_data_sources():
    """List all registered data sources, collection metadata, status, and limitations."""
    try:
        sources = get_all_data_sources()
        return jsonify({
            "success": True,
            "total_sources": len(sources),
            "sources": sources
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/data/status", methods=["GET"])
def api_data_status():
    """Pipeline and data layer operational health status."""
    try:
        with get_pipeline_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM data_sources")
            sources_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM pipeline_anomalies")
            anomalies_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM road_segments")
            segments_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM pipeline_anomalies WHERE road_segment_id IS NOT NULL")
            matched_anomalies = c.fetchone()[0]
            c.execute("SELECT * FROM model_evaluation_metrics ORDER BY id DESC LIMIT 1")
            last_metric = c.fetchone()
            latest_metrics = dict(last_metric) if last_metric else None

        model_payload = get_model()
        model_loaded = model_payload is not None

        return jsonify({
            "success": True,
            "status": "OPERATIONAL",
            "database": {
                "data_sources_registered": sources_count,
                "total_anomalies_ingested": anomalies_count,
                "spatially_matched_anomalies": matched_anomalies,
                "osm_road_segments": segments_count
            },
            "ml_model": {
                "loaded": model_loaded,
                "model_name": model_payload.get("metrics", {}).get("model_name", "RandomForestClassifier") if model_loaded else None,
                "trained_at": model_payload.get("trained_at") if model_loaded else None,
                "latest_evaluation": latest_metrics
            },
            "live_traffic": {
                "source": "Google Maps Platform",
                "connected": GoogleTrafficCollector().is_live_connected,
                "mode": "LIVE" if GoogleTrafficCollector().is_live_connected else "DEMO (Time-of-day simulation)"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/anomalies", methods=["GET"])
def api_anomalies():
    """Query normalized anomalies from pipeline_anomalies."""
    source = request.args.get("source")
    anomaly_type = request.args.get("type")
    road_segment_id = request.args.get("road_segment_id")
    is_live_arg = request.args.get("is_live")
    is_live = None
    if is_live_arg is not None:
        is_live = is_live_arg.lower() in ("true", "1")
    limit = int(request.args.get("limit", 200))

    try:
        anomalies = get_anomalies_filtered(
            source=source,
            anomaly_type=anomaly_type,
            road_segment_id=road_segment_id,
            is_live=is_live,
            limit=limit
        )
        return jsonify({
            "success": True,
            "total": len(anomalies),
            "anomalies": anomalies
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/road-risk/<segment_id>", methods=["GET"])
def api_road_risk(segment_id: str):
    """Detailed multi-factor risk decomposition for a specific OSM road segment."""
    try:
        seg = get_pipeline_road_segment(segment_id)
        if not seg:
            road = get_road_by_id(segment_id)
            if not road:
                return jsonify({"success": False, "error": f"Road segment '{segment_id}' not found."}), 404
            seg = road

        hist_risk = float(seg.get("risk_score", 0.0))
        pred_risk = float(seg.get("predicted_risk", 0.0))
        traffic_factor = float(seg.get("live_traffic_factor", 1.0))
        live_risk = float(seg.get("live_incident_risk", 0.0))

        comb_risk = calculate_combined_risk({
            "historical_risk": hist_risk,
            "predicted_risk": pred_risk,
            "live_incident_risk": live_risk
        })

        pred_info = predict_segment_risk({
            "road_type_enc": 3 if seg.get("road_type") == "trunk" else 2,
            "length_m": seg.get("length_m", 1000.0),
            "lanes": seg.get("lanes", 2),
            "speed_limit": seg.get("speed_limit", 40.0),
            "district_baseline_risk": hist_risk,
            "adas_warning_density": round(seg.get("adas_warning_count", 0) / max(0.1, seg.get("length_m", 1000.0) / 1000.0), 2)
        })

        return jsonify({
            "success": True,
            "segment_id": segment_id,
            "road_name": seg.get("road_name", seg.get("name", "")),
            "district": seg.get("district", "Chennai"),
            "road_type": seg.get("road_type", "primary"),
            "risk_decomposition": {
                "historical_risk": hist_risk,
                "predicted_risk": pred_risk if pred_risk > 0 else pred_info["predicted_risk"],
                "live_traffic_factor": traffic_factor,
                "live_incident_risk": live_risk,
                "combined_risk": comb_risk,
                "risk_tier": pred_info["risk_tier"]
            },
            "provenance": {
                "historical_data": "HISTORICAL (Tamil Nadu District Accidents & OpenCity)",
                "predicted_risk": "PREDICTED (Machine Learning Random Forest on ADAS signals)",
                "live_traffic": "LIVE" if GoogleTrafficCollector().is_live_connected else "DEMO (Time-of-day simulation)",
                "live_incidents": "Real-time monitored flags"
            },
            "contributing_factors": pred_info["top_factors"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-risk", methods=["POST"])
def api_predict_risk():
    """ML model inference endpoint for custom road segment feature vector."""
    try:
        data = request.get_json() or {}
        if "segment_id" in data and len(data) == 1:
            seg = get_pipeline_road_segment(data["segment_id"])
            if not seg:
                return jsonify({"success": False, "error": f"Segment '{data['segment_id']}' not found."}), 404
            features = {
                "road_type_enc": 3 if seg.get("road_type") == "trunk" else 2,
                "length_m": seg.get("length_m", 1000.0),
                "lanes": seg.get("lanes", 2),
                "speed_limit": seg.get("speed_limit", 40.0),
                "district_baseline_risk": seg.get("risk_score", 0.4),
                "district_fatality_rate": 0.2,
                "adas_warning_density": round(seg.get("adas_warning_count", 0) / max(0.1, seg.get("length_m", 1000.0) / 1000.0), 2)
            }
        else:
            features = data

        prediction = predict_segment_risk(features)
        return jsonify({
            "success": True,
            "prediction": prediction,
            "features_evaluated": features
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/network-summary", methods=["GET"])
def api_network_summary():
    """Summary counts of roads, open/flagged statuses, and anomalies."""
    try:
        summary = get_network_summary()
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to compute network summary: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check verifying Flask API and SQLite database linkage."""
    try:
        health = check_database_health()
        return jsonify({
            "success": True,
            "status": "healthy",
            "api": "online",
            "database": health
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "database_error",
            "api": "online",
            "error": str(e)
        }), 500


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_database_initialized()
    port = int(os.environ.get("PORT", 5000))
    print(f"ROUTEWISE server running on http://127.0.0.1:{port} (and http://localhost:{port})")
    app.run(host="0.0.0.0", port=port, debug=False)

