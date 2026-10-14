"""
backend/data_pipeline/storage/database.py
=========================================
Storage interface for ROUTEWISE Data + ML Pipeline.
Manages SQLite schemas for data sources, normalized anomalies,
OSM road segments, validation logs, and ML evaluation metrics.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "database", "route_advisor.db")
)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a connection to the SQLite database with dict-like Row factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_pipeline_tables(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Ensure all pipeline tables exist in the database without destroying
    existing core tables.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Data sources provenance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT,
                collection_time TEXT NOT NULL,
                geographic_coverage TEXT,
                time_coverage TEXT,
                fields_available TEXT,
                is_historical INTEGER NOT NULL DEFAULT 1,
                is_live INTEGER NOT NULL DEFAULT 0,
                is_synthetic INTEGER NOT NULL DEFAULT 0,
                accessible INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                limitations TEXT,
                licensing_notes TEXT
            )
        """)

        # Dataset validation logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_validation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                file_path TEXT,
                row_count INTEGER NOT NULL,
                columns TEXT NOT NULL,
                date_range TEXT,
                coord_availability TEXT,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                missing_pct REAL NOT NULL DEFAULT 0.0,
                data_nature TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Normalized anomalies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_anomalies (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                timestamp TEXT,
                anomaly_type TEXT NOT NULL,
                severity REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                road_segment_id TEXT,
                district TEXT,
                state TEXT DEFAULT 'Tamil Nadu',
                is_historical INTEGER NOT NULL DEFAULT 1,
                is_live INTEGER NOT NULL DEFAULT 0,
                match_distance_m REAL,
                metadata TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anom_road_segment ON pipeline_anomalies(road_segment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anom_source ON pipeline_anomalies(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anom_type ON pipeline_anomalies(anomaly_type)")

        # Road segments table (OSM road graph central entity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS road_segments (
                segment_id TEXT PRIMARY KEY,
                osm_way_id TEXT,
                road_name TEXT NOT NULL,
                road_type TEXT,
                length_m REAL NOT NULL,
                lanes INTEGER,
                speed_limit REAL,
                district TEXT,
                latitude REAL,
                longitude REAL,
                historical_accident_count INTEGER DEFAULT 0,
                historical_fatality_count INTEGER DEFAULT 0,
                adas_warning_count INTEGER DEFAULT 0,
                hotspot_score REAL DEFAULT 0.0,
                live_incident_count INTEGER DEFAULT 0,
                live_traffic_factor REAL DEFAULT 1.0,
                risk_score REAL DEFAULT 0.0,
                predicted_risk REAL DEFAULT 0.0,
                geometry TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_segment_name ON road_segments(road_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_segment_district ON road_segments(district)")

        # Model evaluation metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                target_variable TEXT NOT NULL,
                split_type TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                roc_auc REAL,
                pr_auc REAL,
                confusion_matrix TEXT,
                feature_importances TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()


def save_data_source(source_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace record in data_sources."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO data_sources (
                id, name, url, collection_time, geographic_coverage,
                time_coverage, fields_available, is_historical, is_live,
                is_synthetic, accessible, status, record_count,
                limitations, licensing_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_data["id"],
            source_data["name"],
            source_data.get("url"),
            source_data.get("collection_time", datetime.now(timezone.utc).isoformat()),
            source_data.get("geographic_coverage"),
            source_data.get("time_coverage"),
            json.dumps(source_data.get("fields_available", [])),
            1 if source_data.get("is_historical", True) else 0,
            1 if source_data.get("is_live", False) else 0,
            1 if source_data.get("is_synthetic", False) else 0,
            1 if source_data.get("accessible", True) else 0,
            source_data.get("status", "ACTIVE"),
            source_data.get("record_count", 0),
            source_data.get("limitations"),
            source_data.get("licensing_notes")
        ))
        conn.commit()


def save_validation_log(log_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Save dataset validation log."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dataset_validation_logs (
                source_id, file_path, row_count, columns, date_range,
                coord_availability, duplicate_count, missing_pct,
                data_nature, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_data["source_id"],
            log_data.get("file_path"),
            log_data["row_count"],
            json.dumps(log_data["columns"]),
            log_data.get("date_range"),
            log_data.get("coord_availability"),
            log_data.get("duplicate_count", 0),
            log_data.get("missing_pct", 0.0),
            log_data.get("data_nature", "REAL"),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()


def save_anomalies_batch(anomalies: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> int:
    """Batch insert or replace pipeline anomalies."""
    if not anomalies:
        return 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO pipeline_anomalies (
                id, source, latitude, longitude, timestamp,
                anomaly_type, severity, confidence, road_segment_id,
                district, state, is_historical, is_live,
                match_distance_m, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                a["id"],
                a["source"],
                a.get("latitude"),
                a.get("longitude"),
                a.get("timestamp"),
                a["anomaly_type"],
                a.get("severity", 1.0),
                a.get("confidence", 1.0),
                a.get("road_segment_id"),
                a.get("district"),
                a.get("state", "Tamil Nadu"),
                1 if a.get("is_historical", True) else 0,
                1 if a.get("is_live", False) else 0,
                a.get("match_distance_m"),
                json.dumps(a.get("metadata", {}))
            )
            for a in anomalies
        ])
        conn.commit()
    return len(anomalies)


def save_road_segments(segments: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> int:
    """Batch insert or replace road segments."""
    if not segments:
        return 0
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO road_segments (
                segment_id, osm_way_id, road_name, road_type,
                length_m, lanes, speed_limit, district,
                latitude, longitude, historical_accident_count,
                historical_fatality_count, adas_warning_count,
                hotspot_score, live_incident_count, live_traffic_factor,
                risk_score, predicted_risk, geometry, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                s["segment_id"],
                s.get("osm_way_id"),
                s["road_name"],
                s.get("road_type", "primary"),
                s.get("length_m", 1000.0),
                s.get("lanes", 2),
                s.get("speed_limit", 50.0),
                s.get("district", "Chennai"),
                s.get("latitude"),
                s.get("longitude"),
                s.get("historical_accident_count", 0),
                s.get("historical_fatality_count", 0),
                s.get("adas_warning_count", 0),
                s.get("hotspot_score", 0.0),
                s.get("live_incident_count", 0),
                s.get("live_traffic_factor", 1.0),
                s.get("risk_score", 0.0),
                s.get("predicted_risk", 0.0),
                json.dumps(s.get("geometry", [])),
                s.get("updated_at", now_str)
            )
            for s in segments
        ])
        conn.commit()
    return len(segments)


def get_all_road_segments(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all road segments with deserialized geometry."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM road_segments")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("geometry"), str):
                try:
                    d["geometry"] = json.loads(d["geometry"])
                except Exception:
                    d["geometry"] = []
            result.append(d)
        return result


def get_road_segment_by_id(segment_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single road segment by id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM road_segments WHERE segment_id = ?", (segment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("geometry"), str):
            try:
                d["geometry"] = json.loads(d["geometry"])
            except Exception:
                d["geometry"] = []
        return d


def update_segment_live_traffic(segment_id: str, traffic_factor: float, db_path: str = DEFAULT_DB_PATH) -> None:
    """Update live_traffic_factor and updated_at for a road segment."""
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE road_segments
            SET live_traffic_factor = ?, updated_at = ?
            WHERE segment_id = ?
        """, (traffic_factor, now_str, segment_id))
        conn.commit()


def get_all_data_sources(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Return all recorded data sources."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM data_sources ORDER BY id")
        rows = cursor.fetchall()
        sources = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("fields_available"), str):
                try:
                    d["fields_available"] = json.loads(d["fields_available"])
                except Exception:
                    pass
            sources.append(d)
        return sources


def get_anomalies_filtered(
    source: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    road_segment_id: Optional[str] = None,
    is_live: Optional[bool] = None,
    limit: int = 200,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Fetch pipeline anomalies with optional filtering."""
    query = "SELECT * FROM pipeline_anomalies WHERE 1=1"
    params = []
    if source:
        query += " AND source = ?"
        params.append(source)
    if anomaly_type:
        query += " AND anomaly_type = ?"
        params.append(anomaly_type)
    if road_segment_id:
        query += " AND road_segment_id = ?"
        params.append(road_segment_id)
    if is_live is not None:
        query += " AND is_live = ?"
        params.append(1 if is_live else 0)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        anomalies = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("metadata"), str):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except Exception:
                    pass
            anomalies.append(d)
        return anomalies
