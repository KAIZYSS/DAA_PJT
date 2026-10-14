-- ROUTEWISE Database Schema
-- Review 1 Prototype

DROP TABLE IF EXISTS anomalies;
DROP TABLE IF EXISTS roads;
DROP TABLE IF EXISTS locations;
DROP TABLE IF EXISTS nodes;

CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL
);

CREATE TABLE locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    description TEXT
);

CREATE TABLE roads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    from_node TEXT NOT NULL REFERENCES nodes(id),
    to_node TEXT NOT NULL REFERENCES nodes(id),
    distance_km REAL NOT NULL,
    speed_kmph REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    risk_score REAL NOT NULL DEFAULT 0.0,
    geometry TEXT NOT NULL
);

CREATE TABLE anomalies (
    id TEXT PRIMARY KEY,
    road_id TEXT NOT NULL REFERENCES roads(id),
    type TEXT NOT NULL,
    severity INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_roads_from ON roads(from_node);
CREATE INDEX IF NOT EXISTS idx_roads_to ON roads(to_node);
CREATE INDEX IF NOT EXISTS idx_anomalies_road ON anomalies(road_id);
