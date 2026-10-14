"""
database/database.py
====================
SQLite connection helpers and query interfaces for the ROUTEWISE prototype.
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_advisor.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Obtain a SQLite connection configured with dict-like row access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_database_initialized(db_path: str = DB_PATH) -> bool:
    """
    Verify that the SQLite database exists, contains all required tables,
    and is populated with seed data. If not, auto-seeds the database.
    """
    required_tables = ["nodes", "roads", "anomalies", "locations"]
    needs_seeding = False

    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        needs_seeding = True
    else:
        try:
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                for t in required_tables:
                    if t not in existing_tables:
                        needs_seeding = True
                        break
                if not needs_seeding:
                    cursor.execute("SELECT COUNT(*) FROM roads")
                    if cursor.fetchone()[0] == 0:
                        needs_seeding = True
        except Exception:
            needs_seeding = True

    if needs_seeding:
        print(f"[Database] Initializing and seeding SQLite database at {db_path}...")
        from database.seed import seed_database
        seed_database(db_path)
        print("[Database] Seeding complete and database linked successfully.")
        return True

    return False


def check_database_health(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Check database connection and table record counts."""
    ensure_database_initialized(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        counts = {}
        for table in ["nodes", "roads", "anomalies", "locations"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        return {
            "connected": True,
            "engine": "SQLite",
            "db_path": os.path.basename(db_path),
            "counts": counts
        }



def get_all_nodes() -> List[Dict[str, Any]]:
    """Fetch all junction nodes in the graph."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, lat, lng FROM nodes ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]


def get_all_roads() -> List[Dict[str, Any]]:
    """Fetch all road segments with decoded coordinate geometry and ML risk attributes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='road_segments'")
        has_pipeline = cursor.fetchone() is not None

        if has_pipeline:
            cursor.execute("""
                SELECT r.id, r.name, r.from_node, r.to_node, r.distance_km, r.speed_kmph,
                       r.status, r.risk_score, r.geometry,
                       COALESCE(rs.predicted_risk, 0.0) AS predicted_risk,
                       COALESCE(rs.live_traffic_factor, 1.0) AS live_traffic_factor,
                       COALESCE(rs.adas_warning_count, 0) AS adas_warning_count,
                       rs.district, rs.road_type, rs.lanes
                FROM roads r
                LEFT JOIN road_segments rs ON r.id = rs.segment_id
                ORDER BY r.id
            """)
        else:
            cursor.execute("""
                SELECT id, name, from_node, to_node, distance_km, speed_kmph,
                       status, risk_score, geometry
                FROM roads
                ORDER BY id
            """)

        roads = []
        for row in cursor.fetchall():
            d = dict(row)
            try:
                d["geometry"] = json.loads(d["geometry"])
            except Exception:
                d["geometry"] = []
            roads.append(d)
        return roads


def get_road_by_id(road_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single road record by ID with ML risk attributes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='road_segments'")
        has_pipeline = cursor.fetchone() is not None

        if has_pipeline:
            cursor.execute("""
                SELECT r.id, r.name, r.from_node, r.to_node, r.distance_km, r.speed_kmph,
                       r.status, r.risk_score, r.geometry,
                       COALESCE(rs.predicted_risk, 0.0) AS predicted_risk,
                       COALESCE(rs.live_traffic_factor, 1.0) AS live_traffic_factor,
                       COALESCE(rs.adas_warning_count, 0) AS adas_warning_count,
                       rs.district, rs.road_type, rs.lanes
                FROM roads r
                LEFT JOIN road_segments rs ON r.id = rs.segment_id
                WHERE r.id = ?
            """, (road_id,))
        else:
            cursor.execute("""
                SELECT id, name, from_node, to_node, distance_km, speed_kmph,
                       status, risk_score, geometry
                FROM roads
                WHERE id = ?
            """, (road_id,))

        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["geometry"] = json.loads(d["geometry"])
        except Exception:
            d["geometry"] = []
        return d



def get_anomalies_for_road(road_id: str) -> List[Dict[str, Any]]:
    """Fetch all historical anomaly records associated with a given road ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, road_id, type, severity, timestamp, duration_minutes, description
            FROM anomalies
            WHERE road_id = ?
            ORDER BY timestamp DESC
        """, (road_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_locations() -> List[Dict[str, Any]]:
    """Fetch all predefined named locations for autocomplete."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, node_id, lat, lng, description FROM locations ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]


def get_network_summary() -> Dict[str, Any]:
    """
    Compute dynamic network statistics for the sidebar summary panel (Spec Section 25).
    - Road segments count
    - Open roads count
    - Flagged roads count
    - High-risk roads count (risk_score >= 0.67)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM roads")
        total_segments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM roads WHERE status = 'open'")
        open_roads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM roads WHERE status != 'open'")
        flagged_roads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM roads WHERE risk_score >= 0.67")
        high_risk_roads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM anomalies")
        total_anomalies = cursor.fetchone()[0]

        return {
            "total_segments": total_segments,
            "open_roads": open_roads,
            "flagged_roads": flagged_roads,
            "high_risk_roads": high_risk_roads,
            "total_anomalies": total_anomalies,
            "is_prototype": True
        }
