"""
backend/data_pipeline/processors/normalizer.py
==============================================
Maps heterogeneous source records into the standardized common anomaly schema.
Enforces strict semantic constraints:
- ADAS alerts mapped to 'ADAS_WARNING' (NEVER 'ACCIDENT')
- Historical aggregates preserved without manufacturing fake point coordinates.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


def normalize_adas_record(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize a raw record from ADAS Mobility Data.
    Raw fields: Alert, Date, Time, Lat, Long, Vehicle, Speed
    Alert types: cas_ldw, cas_hmw, cas_fcw, cas_pcw
    """
    alert_raw = str(raw.get("Alert", "")).strip().lower()
    date_str = str(raw.get("Date", "")).strip()
    time_str = str(raw.get("Time", "")).strip()
    timestamp = f"{date_str} {time_str}".strip() if date_str else datetime.now(timezone.utc).isoformat()

    lat = float(raw["Lat"]) if raw.get("Lat") is not None else None
    lng = float(raw["Long"]) if raw.get("Long") is not None else None
    speed = float(raw["Speed"]) if raw.get("Speed") is not None else None
    vehicle = str(raw.get("Vehicle", ""))

    # Severity weighting based on collision urgency (scale 1.0 - 10.0)
    severity_map = {
        "cas_fcw": 7.5,  # Forward Collision Warning (imminent front impact)
        "cas_pcw": 8.5,  # Pedestrian Collision Warning (pedestrian vulnerable road user)
        "cas_ldw": 4.0,  # Lane Departure Warning (lateral drift / vigilance issue)
        "cas_hmw": 3.0,  # Headway Monitoring Warning (close following / tailgating)
    }
    severity = severity_map.get(alert_raw, 3.0)

    # Human-readable specific alert category
    alert_subtypes = {
        "cas_fcw": "FORWARD_COLLISION_WARNING",
        "cas_pcw": "PEDESTRIAN_COLLISION_WARNING",
        "cas_ldw": "LANE_DEPARTURE_WARNING",
        "cas_hmw": "HEADWAY_MONITORING_WARNING"
    }

    return {
        "id": f"adas_{index:06d}",
        "source": "ADAS_MOBILITY_TN",
        "latitude": lat,
        "longitude": lng,
        "timestamp": timestamp,
        "anomaly_type": "ADAS_WARNING",  # STRICT: never label as ACCIDENT
        "severity": severity,
        "confidence": 0.95,               # Direct CAN/sensor telemetry
        "road_segment_id": None,          # Assigned by spatial matcher
        "district": "Chennai/Chengalpattu",
        "state": "Tamil Nadu",
        "is_historical": True,
        "is_live": False,
        "match_distance_m": None,
        "metadata": {
            "specific_alert": alert_subtypes.get(alert_raw, alert_raw),
            "raw_alert": alert_raw,
            "vehicle_id": vehicle,
            "speed_kmph": speed
        }
    }


def normalize_live_incident_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a live incident report (e.g. reported road work, closure, severe jam)."""
    incident_id = raw.get("id") or f"live_{uuid.uuid4().hex[:8]}"
    anomaly_type = raw.get("anomaly_type", "OTHER").upper()
    valid_types = {
        "ACCIDENT", "ROAD_WORK", "ROAD_CLOSURE",
        "TRAFFIC_JAM", "FLOODING", "ADAS_WARNING", "HOTSPOT", "OTHER"
    }
    if anomaly_type not in valid_types:
        anomaly_type = "OTHER"

    return {
        "id": incident_id,
        "source": raw.get("source", "LIVE_MONITOR"),
        "latitude": float(raw["latitude"]) if raw.get("latitude") is not None else None,
        "longitude": float(raw["longitude"]) if raw.get("longitude") is not None else None,
        "timestamp": raw.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "anomaly_type": anomaly_type,
        "severity": float(raw.get("severity", 5.0)),
        "confidence": float(raw.get("confidence", 0.85)),
        "road_segment_id": raw.get("road_segment_id"),
        "district": raw.get("district", "Chennai"),
        "state": "Tamil Nadu",
        "is_historical": False,
        "is_live": True,
        "match_distance_m": raw.get("match_distance_m"),
        "metadata": raw.get("metadata", {})
    }
