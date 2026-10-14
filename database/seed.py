"""
database/seed.py
================
Database seeder script.
Loads prototype JSON datasets, computes dynamic risk scores using
reliability/scoring.py, and populates a fresh SQLite route_advisor.db.
"""

import os
import sys
import json
import sqlite3

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reliability.scoring import compute_all_risk_scores


def seed_database(db_path: str = None):
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_advisor.db")

    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # Load JSON files
    with open(os.path.join(data_dir, "nodes.json"), "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(os.path.join(data_dir, "locations.json"), "r", encoding="utf-8") as f:
        locations = json.load(f)

    with open(os.path.join(data_dir, "roads.json"), "r", encoding="utf-8") as f:
        roads = json.load(f)

    with open(os.path.join(data_dir, "anomalies.json"), "r", encoding="utf-8") as f:
        anomalies = json.load(f)

    # Compute risk scores using the exact mathematical formula
    risk_scores = compute_all_risk_scores(roads, anomalies)

    print(f"Seeding database at: {db_path}")

    # Recreate tables
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    cursor = conn.cursor()

    # Insert Nodes
    for n in nodes:
        cursor.execute(
            "INSERT INTO nodes (id, name, lat, lng) VALUES (?, ?, ?, ?)",
            (n["id"], n["name"], n["lat"], n["lng"])
        )

    # Insert Locations
    for loc in locations:
        cursor.execute(
            "INSERT INTO locations (id, name, node_id, lat, lng, description) VALUES (?, ?, ?, ?, ?, ?)",
            (loc["id"], loc["name"], loc["node_id"], loc["lat"], loc["lng"], loc.get("description", ""))
        )

    # Insert Roads with computed risk score
    for r in roads:
        score = risk_scores.get(r["id"], 0.0)
        geom_json = json.dumps(r["geometry"])
        cursor.execute(
            """
            INSERT INTO roads (id, name, from_node, to_node, distance_km, speed_kmph, status, risk_score, geometry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (r["id"], r["name"], r["from_node"], r["to_node"], r["distance_km"], r["speed_kmph"], r["status"], score, geom_json)
        )

    # Insert Anomalies
    for a in anomalies:
        cursor.execute(
            """
            INSERT INTO anomalies (id, road_id, type, severity, timestamp, duration_minutes, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (a["id"], a["road_id"], a["type"], a["severity"], a["timestamp"], a["duration_minutes"], a.get("description", ""))
        )

    conn.commit()
    conn.close()

    print(f"Successfully seeded:")
    print(f" - {len(nodes)} nodes")
    print(f" - {len(locations)} locations")
    print(f" - {len(roads)} road segments")
    print(f" - {len(anomalies)} anomalies")


if __name__ == "__main__":
    seed_database()
