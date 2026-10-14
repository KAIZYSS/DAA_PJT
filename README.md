# ROUTEWISE — Reliable Route Advisor

> **"Don't just find the fastest route. Find the route you can rely on."**
> DAA (Design and Analysis of Algorithms) Course Project — Review 1 Prototype

---

## 1. Problem Statement

Standard navigation engines and shortest-path algorithms (such as plain Dijkstra or $A^*$) optimize strictly for minimal distance or nominal free-flow travel time. In real-world urban transit networks—such as Chennai—the nominally "fastest" arterial highway often suffers from severe, recurrent disruptions: construction bottlenecks, water-logging, high accident frequency, and sudden lane constrictions. Consequently, drivers face unpredictable travel times, high variance, and frequent gridlock.

**ROUTEWISE** introduces **Anomaly-Aware Rerouting**. By integrating historical road fault incidents, severity metrics, and downtime distributions into edge evaluation functions, ROUTEWISE identifies the most *reliable* corridor rather than merely the nominally fastest one.

---

## 2. Review 1 Scope

### What Is Implemented in Review 1
- **Real OpenStreetMap GIS Foundation**: Interactive Leaflet.js raster map integration with pan, zoom, custom animated SVG pins, and standard OSM attribution.
- **Curated Urban Road Network**: 19 junction nodes and 52 directed road segments across Chennai with authentic geographical coordinates.
- **Multi-Factor Reliability Scoring**: Mathematical formulation incorporating anomaly frequency, mean severity, and historical downtime ($0.40 / 0.35 / 0.25$ formula).
- **Manual Dijkstra Implementation**: Single-source shortest path algorithm using Python's `heapq` with time complexity $\mathcal{O}((V + E) \log V)$.
- **Decoupled Weight Modes**: Standalone cost functions for **Fastest**, **Reliable**, and **Balanced** routing.
- **Demonstrable Divergence**: Chennai Central $\to$ VIT Chennai and Chennai Central $\to$ Tambaram scenarios where Fastest $\neq$ Reliable.
- **Route Comparison Engine**: Runs Dijkstra across candidate modes on the same graph to generate side-by-side trade-offs.
- **Dynamic "Why This Route?" Explanation**: Evaluates avoided high-risk corridors, stability gains, and active flags.
- **Professional GIS Analytics UI**: Dark dashboard aesthetic with rolling number count-ups, progressive polyline draw-on ($800\text{ms} - 1.2\text{s}$), sliding pill segmented controls, road inspection panel, and `prefers-reduced-motion` compliance.

### What Is Deferred to Review 2
- Direct Overpass API / OSM automated network extraction
- Real-time GPS vehicle tracking & map matching
- Dynamic incident simulation engine
- All-Pairs Shortest Path (Floyd-Warshall) precomputation & exact-vs-approximate benchmarking
- Machine learning predictive congestion models

---

## 3. Architecture Overview

```
e:\DAA_PJT\
├── app.py                     # Flask REST API & routing coordinator
├── requirements.txt           # Project dependencies (Flask, pytest)
├── README.md                  # System documentation
├── database/
│   ├── database.py            # SQLite connection & query helpers
│   ├── schema.sql             # SQL schema (nodes, roads, anomalies, locations)
│   ├── seed.py                # Database population & risk calculation script
│   ├── generate_data.py       # Chennai network data generator
│   └── route_advisor.db       # Generated SQLite database
├── algorithms/
│   ├── __init__.py
│   └── dijkstra.py            # Manual binary min-heap Dijkstra algorithm
├── reliability/
│   ├── __init__.py
│   └── scoring.py             # 3-factor normalized risk scoring model
├── data/
│   ├── locations.json         # 8 named transit hubs in Chennai
│   ├── nodes.json             # 19 network junction nodes
│   ├── roads.json             # 52 directed road segments with geometry
│   └── anomalies.json         # 30 historical anomaly records
├── templates/
│   └── index.html             # Dashboard layout & Leaflet container
├── static/
│   ├── css/
│   │   └── style.css          # GIS Dark Theme, keyframes, transitions
│   └── js/
│       ├── api.js             # Client fetch wrapper with caching
│       ├── map.js             # Leaflet layer control & progressive drawing
│       └── app.js             # UI state, rolling numbers, and interactions
└── tests/
    ├── __init__.py
    ├── test_dijkstra.py       # Unit tests for manual Dijkstra
    └── test_api.py            # Integration tests for Flask endpoints
```

---

## 4. Algorithm & Complexity Analysis

### Dijkstra's Algorithm
Manual implementation using Python's standard `heapq` priority queue:
- **Time Complexity**: $\mathcal{O}((V + E) \log V)$
  - Vertex extractions: $V$ extractions taking $\mathcal{O}(\log V)$ each.
  - Edge relaxations: at most $E$ updates pushed to the heap, taking $\mathcal{O}(\log V)$ each.
- **Space Complexity**: $\mathcal{O}(V + E)$ for adjacency list, predecessor mapping, and distance dictionary.

### Edge Weight Formulations
For each edge $e$:
$$\text{Base Travel Time } T(e) = \left(\frac{\text{distance\_km}}{\text{speed\_kmph}}\right) \times 60$$

1. **Fastest Mode**:
   $$\text{Cost}_{\text{fastest}}(e) = T(e)$$
2. **Reliable Mode**:
   $$\text{Cost}_{\text{reliable}}(e) = T(e) \times (1.0 + \text{risk\_score}(e))$$
3. **Balanced Mode**:
   $$\text{Cost}_{\text{balanced}}(e) = T(e) \times (1.0 + 0.5 \times \text{risk\_score}(e))$$

---

## 5. Mathematical Scoring Model

File: `reliability/scoring.py`

For each road segment, three normalized sub-scores in $[0.0, 1.0]$ are calculated:
1. **Frequency Score**: $\frac{\text{anomaly count}}{\max(\text{max count in network}, 1)}$
2. **Severity Score**: $\frac{\text{average severity of anomalies}}{10.0}$
3. **Downtime Score**: $\frac{\text{total anomaly duration}}{\max(\text{max downtime in network}, 1)}$

$$\text{risk\_score} = 0.40 \times \text{Frequency} + 0.35 \times \text{Severity} + 0.25 \times \text{Downtime}$$
$$\text{Reliability \%} = (1.0 - \text{risk\_score}) \times 100$$

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Edge, Firefox, Safari)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Seed Database
```powershell
python database/seed.py
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/ -v
```

### 4. Start the Application
```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 7. Demo Scenario (Chennai Central $\to$ VIT Chennai)

1. Open `http://127.0.0.1:5000`.
2. Notice the initial **App Shell Load Sequence**: navbar slides down, sidebar enters, road network fades in, and `● MAP ONLINE` status dot pulses softly.
3. Start Location defaults to **Chennai Central** and Destination to **VIT Chennai**.
4. With mode set to **Fastest**:
   - Click **"FIND BEST ROUTE"**.
   - Path routes along the inner GST Road arterial (shorter nominal time, but crosses 5 high-risk anomaly zones; Reliability $\approx 62\%$).
5. Switch mode to **Reliable** (sliding pill glides smoothly):
   - Path reroutes along the Outer / Medavakkam / Mambakkam corridor (slightly longer distance, but avoids all major disruptions; Reliability $\approx 88\%$).
6. Observe:
   - **Progressive Polyline Draw**: Route progressively traces itself from A to B.
   - **Rolling Numbers**: Distance, time, and reliability animate smoothly from 0.
   - **Radial Gauge Fill**: Reliability gauge fills synchronously.
   - **Why This Route?**: Explanatory bullets appear with staggered animation.
   - **Route Comparison Rows**: Hover lift and click to cross-fade swap polylines.
   - **Road Inspection**: Click any road segment on the map to slide open its historical incident log.

---

## 8. Academic Disclaimer
*Road-condition indicators, anomaly logs, and reliability scores shown in this Review 1 prototype use simulated data designed for academic demonstration of anomaly-aware pathfinding algorithms.*
