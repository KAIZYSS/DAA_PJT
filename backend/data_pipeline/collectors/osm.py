"""
backend/data_pipeline/collectors/osm.py
======================================
Adapter for SOURCE 7: OpenStreetMap Road Graph.
Extracts road segments as the central spatial entity of ROUTEWISE:
- osm_way_id, geometry, road_name, highway_type, length_m, lanes, speed_limit, district.
CRITICAL: OSM is the road network graph, NOT accident data.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from backend.data_pipeline.storage.database import (
    save_data_source, save_road_segments, DEFAULT_DB_PATH
)

ROADS_JSON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "roads.json")
)
NODES_JSON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "nodes.json")
)


def determine_road_type(road_name: str) -> str:
    """Infer OSM highway classification from road name and hierarchy."""
    r_lower = road_name.lower()
    if any(k in r_lower for k in ["gst", "grand southern", "national highway", "nh45", "expressway", "bypass"]):
        return "trunk"
    elif any(k in r_lower for k in ["anna salai", "mount road", "omr", "old mahabalipuram", "poonamallee", "inner ring"]):
        return "primary"
    elif any(k in r_lower for k in ["velachery", "medavakkam", "saidapet", "kodambakkam", "adyar", "t. nagar"]):
        return "secondary"
    return "tertiary"


def determine_lanes(road_type: str) -> int:
    """Estimate lane count where explicit OSM tag is missing."""
    if road_type == "trunk":
        return 6
    elif road_type == "primary":
        return 4
    elif road_type == "secondary":
        return 2
    return 2


def determine_district(lat: float, lng: float) -> str:
    """Assign administrative district based on latitude."""
    if lat < 12.91:
        return "CHENGALPATTU"  # Tambaram South, Vandalur, Kandigai, Kelambakkam
    elif lng < 80.12:
        return "KANCHEEPURAM"  # Sriperumbudur, West suburban
    return "CHENNAI CITY"     # Central, T Nagar, Guindy, Airport, Adyar


def collect_osm_road_segments(
    roads_path: str = ROADS_JSON,
    nodes_path: str = NODES_JSON,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ingest road segments from OSM graph data and populate the road_segments table.
    """
    if not os.path.exists(roads_path):
        raise FileNotFoundError(f"Roads file not found at: {roads_path}")

    with open(roads_path, "r", encoding="utf-8") as f:
        raw_roads = json.load(f)

    segments = []
    now_str = datetime.now(timezone.utc).isoformat()

    for r in raw_roads:
        geom = r.get("geometry", [])
        if not geom:
            continue

        lats = [pt[0] for pt in geom]
        lngs = [pt[1] for pt in geom]
        centroid_lat = round(sum(lats) / len(lats), 6)
        centroid_lng = round(sum(lngs) / len(lngs), 6)

        road_name = r.get("name", "Unnamed Road")
        road_type = determine_road_type(road_name)
        dist_km = float(r.get("distance_km", 1.0))
        length_m = round(dist_km * 1000.0, 1)
        speed = float(r.get("speed_kmph", 40.0))
        district = determine_district(centroid_lat, centroid_lng)

        seg = {
            "segment_id": r["id"],
            "osm_way_id": f"way_{r['id']}",
            "road_name": road_name,
            "road_type": road_type,
            "length_m": length_m,
            "lanes": determine_lanes(road_type),
            "speed_limit": speed,
            "district": district,
            "latitude": centroid_lat,
            "longitude": centroid_lng,
            "historical_accident_count": 0,
            "historical_fatality_count": 0,
            "adas_warning_count": 0,
            "hotspot_score": 0.0,
            "live_incident_count": 0,
            "live_traffic_factor": 1.0,
            "risk_score": 0.0,
            "predicted_risk": 0.0,
            "geometry": geom,
            "updated_at": now_str
        }
        segments.append(seg)

    # Persist to database
    save_road_segments(segments, db_path=db_path)

    source_record = {
        "id": "osm_road_network",
        "name": "OpenStreetMap Road Graph (Chennai Metropolitan Corridor)",
        "url": "https://www.openstreetmap.org",
        "collection_time": now_str,
        "geographic_coverage": "Greater Chennai & Chengalpattu Highway Corridors (GST, OMR, Anna Salai, Inner Ring)",
        "time_coverage": "2026 Active Road Graph",
        "fields_available": [
            "segment_id", "osm_way_id", "road_name", "road_type",
            "length_m", "lanes", "speed_limit", "district", "geometry"
        ],
        "is_historical": False,
        "is_live": False,
        "is_synthetic": False,
        "accessible": True,
        "status": "ACTIVE",
        "record_count": len(segments),
        "limitations": "Road network graph definition. Does not contain accidents.",
        "licensing_notes": "OpenStreetMap (ODbL)"
    }
    save_data_source(source_record, db_path=db_path)

    summary = {
        "total_segments": len(segments),
        "total_length_km": round(sum(s["length_m"] for s in segments) / 1000.0, 2),
        "road_types": dict(pd.Series([s["road_type"] for s in segments]).value_counts()),
        "districts": dict(pd.Series([s["district"] for s in segments]).value_counts())
    }
    return segments, summary
