import json
import math

nodes = [
    {"id": "node_central", "name": "Chennai Central", "lat": 13.0827, "lng": 80.2755},
    {"id": "node_egmore", "name": "Egmore Junction", "lat": 13.0784, "lng": 80.2608},
    {"id": "node_t_nagar", "name": "T. Nagar Panagal Park", "lat": 13.0418, "lng": 80.2341},
    {"id": "node_saidapet", "name": "Saidapet Anna Salai", "lat": 13.0213, "lng": 80.2231},
    {"id": "node_guindy", "name": "Guindy Kathipara", "lat": 13.0067, "lng": 80.2025},
    {"id": "node_airport", "name": "Chennai Airport", "lat": 12.9866, "lng": 80.1652},
    {"id": "node_pallavaram", "name": "Pallavaram Junction", "lat": 12.9675, "lng": 80.1491},
    {"id": "node_chromepet", "name": "Chromepet Flyover", "lat": 12.9516, "lng": 80.1408},
    {"id": "node_tambaram", "name": "Tambaram Gateway", "lat": 12.9249, "lng": 80.1000},
    {"id": "node_perungalathur", "name": "Perungalathur Bypass", "lat": 12.9045, "lng": 80.0890},
    {"id": "node_vandalur", "name": "Vandalur Zoo Junction", "lat": 12.8900, "lng": 80.0815},
    {"id": "node_kelambakkam_rd", "name": "Kandigai Junction", "lat": 12.8550, "lng": 80.1200},
    {"id": "node_vit", "name": "VIT Chennai Campus", "lat": 12.8406, "lng": 80.1534},
    {"id": "node_adyar", "name": "Adyar Signal", "lat": 13.0012, "lng": 80.2565},
    {"id": "node_velachery", "name": "Velachery MRTS Bypass", "lat": 12.9759, "lng": 80.2212},
    {"id": "node_medavakkam", "name": "Medavakkam Koot Road", "lat": 12.9180, "lng": 80.1915},
    {"id": "node_sholinganallur", "name": "Sholinganallur ELCOT", "lat": 12.9010, "lng": 80.2279},
    {"id": "node_omr", "name": "OMR Navalur Toll", "lat": 12.8458, "lng": 80.2265},
    {"id": "node_mambakkam", "name": "Mambakkam Signal", "lat": 12.8420, "lng": 80.1900}
]

# Raw segments (bidirectional)
# [from_id, to_id, name, dist_km, speed_kmh, status, intermediate_points]
segments = [
    # GST Corridor (Central -> Egmore -> T.Nagar -> Saidapet -> Guindy -> Airport -> Pallavaram -> Chromepet -> Tambaram -> Perungalathur -> Vandalur -> Kandigai -> VIT)
    ("node_central", "node_egmore", "Poonamallee High Road", 2.2, 45, "open", []),
    ("node_egmore", "node_t_nagar", "Chetpet - GN Chetty Road", 4.6, 50, "flagged", [[13.0600, 80.2450]]),
    ("node_t_nagar", "node_saidapet", "Anna Salai Arterial", 2.5, 48, "open", []),
    ("node_saidapet", "node_guindy", "Maraimalai Adigal Bridge", 2.4, 45, "flagged", []),
    ("node_guindy", "node_airport", "GST Expressway (Airport Link)", 4.8, 55, "flagged", [[12.9950, 80.1830]]),
    ("node_airport", "node_pallavaram", "GST Road (Pallavaram Stretch)", 2.8, 50, "open", []),
    ("node_pallavaram", "node_chromepet", "GST Flyover Corridor", 2.1, 52, "open", []),
    ("node_chromepet", "node_tambaram", "Sanatorium Corridor", 4.4, 52, "flagged", [[12.9380, 80.1200]]),
    ("node_tambaram", "node_perungalathur", "Mudichur Link - GST", 2.9, 48, "open", []),
    ("node_perungalathur", "node_vandalur", "Vandalur Flyover Arterial", 2.3, 50, "flagged", []),
    ("node_vandalur", "node_kelambakkam_rd", "Vandalur-Kelambakkam West Rd", 5.6, 55, "open", [[12.8700, 80.1000]]),
    ("node_kelambakkam_rd", "node_vit", "Kandigai - VIT Access Road", 3.8, 50, "open", []),

    # Coastal / Outer / OMR Corridor (Central -> Adyar -> Velachery -> Medavakkam -> Mambakkam -> VIT)
    ("node_central", "node_adyar", "Kamarajar Promenade & Santhome Rd", 10.2, 42, "open", [[13.0450, 80.2800], [13.0200, 80.2650]]),
    ("node_egmore", "node_adyar", "TTK Road & Chamiers Rd", 9.4, 38, "open", [[13.0400, 80.2500]]),
    ("node_adyar", "node_velachery", "Raj Bhavan & Velachery Main Rd", 4.9, 40, "open", []),
    ("node_guindy", "node_velachery", "Inner Ring Road (Guindy-Velachery)", 3.8, 42, "open", []),
    ("node_velachery", "node_medavakkam", "Velachery-Tambaram Main Road", 7.2, 45, "open", [[12.9450, 80.2050]]),
    ("node_medavakkam", "node_mambakkam", "Medavakkam-Mambakkam Expressway", 8.8, 45, "open", [[12.8800, 80.1900]]),
    ("node_mambakkam", "node_vit", "Mambakkam-VIT Campus Highway", 4.1, 48, "open", []),

    # OMR & Cross links
    ("node_velachery", "node_sholinganallur", "Perungudi OMR Link", 8.5, 42, "open", [[12.9400, 80.2350]]),
    ("node_sholinganallur", "node_omr", "Rajiv Gandhi IT Expressway", 6.8, 52, "open", [[12.8700, 80.2270]]),
    ("node_medavakkam", "node_sholinganallur", "Medavakkam-Sholinganallur Rd", 4.3, 40, "open", []),
    ("node_medavakkam", "node_tambaram", "Tambaram-Velachery Main Road", 8.9, 40, "open", [[12.9200, 80.1450]]),
    ("node_airport", "node_velachery", "200 Feet Radial Road (West)", 6.5, 46, "open", [[12.9800, 80.1900]]),
    ("node_pallavaram", "node_medavakkam", "Pallavaram-Thoraipakkam 200ft Rd", 6.2, 45, "open", [[12.9400, 80.1700]]),
    ("node_omr", "node_mambakkam", "Navalur-Mambakkam Cross Link", 5.2, 44, "open", [[12.8440, 80.2080]])
]

# Create node lookup
node_map = {n["id"]: n for n in nodes}

roads_data = []
road_id_counter = 1

# Anomalies list
anomalies_data = []
anomaly_id_counter = 1

# We will assign specific anomalies to roads
# Roads on GST Road have multiple heavy anomalies to ensure Fastest != Reliable
anomaly_templates = {
    ("node_egmore", "node_t_nagar"): [
        ("congestion", 8, 60, "Metro line tunnel construction causing lane closures and severe peak gridlock"),
        ("repair", 7, 45, "Underground utility repair work near Sterling Road junction")
    ],
    ("node_saidapet", "node_guindy"): [
        ("repair", 7, 50, "Bridge structural maintenance and resurfacing bottleneck on Anna Salai"),
        ("congestion", 6, 35, "Signal timing failure during high peak commuter rush")
    ],
    ("node_guindy", "node_airport"): [
        ("accident", 9, 80, "Multi-vehicle collision near Kathipara ramp; heavy backlog"),
        ("congestion", 8, 55, "Airport access congestion and perimeter construction barrier")
    ],
    ("node_chromepet", "node_tambaram"): [
        ("congestion", 8, 60, "Severe bottleneck at Chromepet MIT bridge intersection"),
        ("pothole", 7, 40, "Extensive monsoon cratering requiring cautionary crawl speed")
    ],
    ("node_perungalathur", "node_vandalur"): [
        ("congestion", 8, 50, "Interstate bus terminal transit bottleneck and flyover merging"),
        ("repair", 7, 45, "Elevated bypass expansion structural work")
    ],
    ("node_airport", "node_pallavaram"): [
        ("pothole", 5, 25, "Localized road surface deterioration on curbside lane")
    ],
    ("node_velachery", "node_medavakkam"): [
        ("pothole", 2, 10, "Minor asphalt wearing near Pallikaranai marshland boundary")
    ],
    ("node_medavakkam", "node_mambakkam"): [
        ("congestion", 3, 15, "Brief evening bazaar clearance delay at Medavakkam junction")
    ],
    ("node_velachery", "node_sholinganallur"): [
        ("repair", 4, 20, "Fiber optic cable trench resurfacing on service lane")
    ],
    ("node_sholinganallur", "node_omr"): [
        ("congestion", 4, 25, "IT shift end peak queue at Navalur toll plaza")
    ]
}

for u, v, name, dist, speed, status, intermediates in segments:
    u_node = node_map[u]
    v_node = node_map[v]

    # Forward geometry
    geom_fwd = [[u_node["lat"], u_node["lng"]]] + intermediates + [[v_node["lat"], v_node["lng"]]]
    geom_rev = [[v_node["lat"], v_node["lng"]]] + list(reversed(intermediates)) + [[u_node["lat"], u_node["lng"]]]

    fwd_id = f"road_{road_id_counter:03d}"
    road_id_counter += 1
    rev_id = f"road_{road_id_counter:03d}"
    road_id_counter += 1

    roads_data.append({
        "id": fwd_id,
        "name": name,
        "from_node": u,
        "to_node": v,
        "distance_km": dist,
        "speed_kmph": speed,
        "status": status,
        "risk_score": 0.0, # will be populated by seed.py
        "geometry": geom_fwd
    })

    roads_data.append({
        "id": rev_id,
        "name": f"{name} (Return)",
        "from_node": v,
        "to_node": u,
        "distance_km": dist,
        "speed_kmph": speed,
        "status": status,
        "risk_score": 0.0,
        "geometry": geom_rev
    })

    # Add anomalies for this segment pair if present
    pair_key = (u, v) if (u, v) in anomaly_templates else ((v, u) if (v, u) in anomaly_templates else None)
    if pair_key:
        for anom_type, sev, duration, desc in anomaly_templates[pair_key]:
            # Assign to fwd road
            anomalies_data.append({
                "id": f"anom_{anomaly_id_counter:03d}",
                "road_id": fwd_id,
                "type": anom_type,
                "severity": sev,
                "timestamp": "2026-08-15 08:30:00",
                "duration_minutes": duration,
                "description": desc
            })
            anomaly_id_counter += 1
            # Also assign to rev road
            anomalies_data.append({
                "id": f"anom_{anomaly_id_counter:03d}",
                "road_id": rev_id,
                "type": anom_type,
                "severity": sev,
                "timestamp": "2026-08-15 08:30:00",
                "duration_minutes": duration,
                "description": desc
            })
            anomaly_id_counter += 1

with open("data/roads.json", "w", encoding="utf-8") as f:
    json.dump(roads_data, f, indent=2)

with open("data/anomalies.json", "w", encoding="utf-8") as f:
    json.dump(anomalies_data, f, indent=2)

print(f"Generated {len(roads_data)} roads and {len(anomalies_data)} anomalies.")
