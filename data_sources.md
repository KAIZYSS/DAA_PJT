# ROUTEWISE — Data Sources & Provenance Documentation

This document provides complete, transparent documentation of all data sources, adapters, accessibility statuses, schemas, geographic coverages, limitations, and licensing for the **ROUTEWISE Data + ML Layer**.

---

## Data Architecture Overview

```
                          ┌────────────────────────────────────────────────────────┐
                          │               REAL HISTORICAL DATA                     │
                          │ - TN District Accidents (2021-2023)                    │
                          │ - ADAS Mobility Telemetry (21,325 records, 2022)       │
                          │ - OpenCity Greater Chennai Police (1998-2025)          │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌───────────────────────┐
                                          │     DATA CLEANING     │
                                          │    & NORMALIZATION    │
                                          └───────────┬───────────┘
                                                      │
                                                      ▼
┌─────────────────────────┐               ┌───────────────────────┐
│     OSM ROAD GRAPH      │──────────────▶│   SPATIAL MATCHING    │
│  52 Segments, 270.4 km  │               │ (Point-to-Polyline)   │
└────────────┬────────────┘               └───────────┬───────────┘
             │                                        │
             │                                        ▼
             │                            ┌───────────────────────┐
             │                            │ ROAD-SEGMENT FEATURES │
             │                            │ (Chronological T)     │
             │                            └───────────┬───────────┘
             │                                        │
             │                                        ▼
             │                            ┌───────────────────────┐
             │                            │   ML PREDICTED RISK   │
             │                            │  Random Forest (0-1)  │
             │                            └───────────┬───────────┘
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OSM ROAD-SEGMENT GRAPH                       │
│  Edge Attributes: length, lanes, speed, hist_risk, pred_risk    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▲
 ┌───────────────────────────────┴───────────────────────────────┐
 │               LIVE / DYNAMIC INTEGRATION                      │
 │ - Google Maps Live Traffic (live_traffic_factor)              │
 │ - Real-Time Incidents (live_incident_risk)                    │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │     DYNAMIC EDGE WEIGHTS      │
                 │   Travel Time & Combined Risk │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │     OUR CUSTOM DIJKSTRA       │
                 │   Fastest / Balanced / Safe   │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │       DYNAMIC REROUTING       │
                 │   (Threshold > 15% Evaluated) │
                 └───────────────────────────────┘
```

---

## Detailed Data Source Registry

### SOURCE 1: Tamil Nadu District Road Accident Dataset
- **Identifier:** `tn_district_accidents`
- **File / Source:** `data_raw/tn_district_accidents.csv`
- **Data Nature:** **REAL / HISTORICAL**
- **Row Count:** 38 administrative districts + 1 state summary row
- **Date Range:** 2021 – 2023
- **Geographic Coverage:** All 38 administrative districts of Tamil Nadu
- **Fields Available:** `Sl No`, `District`, `Total Accidents 2021`, `Total Deaths 2021`, `Total Accidents 2022`, `Total Deaths 2022`, `Total Accidents 2023`, `Total Deaths 2023`
- **Derived Metrics:** `accidents_3y`, `fatalities_3y`, `fatality_rate`, `accident_trend`, `district_baseline_risk` (normalized 0.0 to 1.0)
- **Role in Pipeline:** District-level prior for road segments located within each district.
- **Critical Limitations:** **DISTRICT-LEVEL AGGREGATE ONLY.** Does not contain road-level or point coordinates. It is never treated as road-level accident frequency.
- **Licensing / Provenance:** Government of Tamil Nadu, State Police & Road Safety Authority. Public government statistical release.

---

### SOURCE 2: Tamil Nadu Accident Dashboard (TNGIS)
- **Identifier:** `tngis_accident_dashboard`
- **URL:** `https://www.tngis.tn.gov.in/apps/accident-dashboard/`
- **Data Nature:** **OFFICIAL STATE GIS PORTAL (LIVE INVESTIGATION)**
- **Detected Layers:** `district`, `accidents`, `accident_heatmap`, `accident_hotspot`, `ambulance_base_location`, `police_station_location`, `tasmac`
- **Detected Endpoints:**
  - GeoServer WMS: `https://tngis.tn.gov.in/tngismaps/wms`
  - AJAX Query: `POST https://www.tngis.tn.gov.in/apps/accident-dashboard/app/rerender.php`
- **Accessibility Status:** **ACCESSIBLE_BUT_FAILED (db error)**
- **Investigation Finding:** Live calls to `app/rerender.php` with bounding box parameters return `db error` from the government server; WMS capabilities requests timed out after 15 seconds.
- **Pipeline Implementation:** `backend/data_pipeline/collectors/tn_dashboard.py` implements the exact HTTP POST request structure with timeout handling. Zero records fabricated. Status recorded transparently as unavailable.
- **Licensing:** Government of Tamil Nadu Spatial Data Infrastructure (TNGIS).

---

### SOURCE 3: Tamil Nadu iRAD / eDAR
- **Identifier:** `irad_edar_tn`
- **URL:** `https://gisnic.tn.nic.in/irad_new/`
- **Data Nature:** **RESTRICTED POLICE / RTO INTRANET (LIVE INVESTIGATION)**
- **Accessibility Status:** **UNAVAILABLE (HTTP 503 / NIC Intranet Restricted)**
- **Investigation Finding:** Live HTTP connection to `https://gisnic.tn.nic.in/irad_new/` returns HTTP 503 Service Unavailable. The application is hosted on NIC intranet infrastructure and restricted to authorized police and transport department officials.
- **Pipeline Implementation:** `backend/data_pipeline/collectors/irad.py` implements the `IRADAdapter` interface with the complete standard iRAD field specification (`accident_id`, `latitude`, `longitude`, `date_time`, `severity`, `vehicles`, `road_conditions`, `collision_type`). Marked `accessible = False`. Zero records fabricated.
- **Licensing:** Ministry of Road Transport and Highways (MoRTH) / National Informatics Centre (NIC).

---

### SOURCE 4: Tamil Nadu ADAS Mobility Dataset
- **Identifier:** `adas_mobility_tn`
- **File / Source:** `data_raw/ADAS Mobility Data.csv`
- **URL:** `https://www.kaggle.com/datasets/bharathtoutireddy/advanced-driver-assistance-systems-mobility-data`
- **Data Nature:** **REAL / HISTORICAL SENSOR TELEMETRY**
- **Row Count:** 21,325 records (20,936 deduplicated)
- **Date Range:** 2022-06-01 to 2022-08-31 (3 full months)
- **Geographic Coverage:** Lat [12.3387, 13.1828], Lng [79.7741, 80.3161] (Chennai, Tambaram, GST Road, Chengalpattu, OMR corridors)
- **Coordinate Availability:** 100% valid GPS pairs
- **Signals / Alerts:**
  - `cas_ldw` (Lane Departure Warning): 6,431 records
  - `cas_hmw` (Headway Monitoring Warning): 12,328 records
  - `cas_fcw` (Forward Collision Warning): 590 records
  - `cas_pcw` (Pedestrian Collision Warning): 1,976 records
- **Vehicles:** 5 test fleet vehicles (`2846`, `805`, `1995`, `5339`, `3143`)
- **Speed Distribution:** 0 to 65 km/h (Mean: 38.4 km/h)
- **Spatial Matching:** 3,944 alerts matched to 21 active Chennai arterial OSM segments within the 150m/250m threshold (Mean distance: 85.99 m, Median: 70.69 m). 16,992 alerts along outer highway stretches properly rejected.
- **Role in Pipeline:** Sourced as road-risk and mobility anomaly indicators.
- **CRITICAL CONSTRAINT:** **ADAS warnings are NEVER labeled as crashes.**
- **Licensing:** Kaggle Open Dataset (CC BY-SA 4.0).

---

### SOURCE 5: Indian Roads Dataset (Kaggle Benchmark)
- **Identifier:** `indian_roads_benchmark`
- **File / Source:** `data_raw/indian_roads_dataset.csv`
- **Data Nature:** **SYNTHETIC / EXPERIMENTAL BENCHMARK**
- **Row Count:** 20,000 total rows (2,575 rows marked 'Chennai')
- **Analysis & Provenance:** Statistical inspection revealed exact uniform category distributions and synthetic step increments (e.g. risk_scores at 0.10, 0.45, 0.65, 0.85).
- **Pipeline Decision:** **EXCLUDED FROM GROUND-TRUTH TRAINING.** Adhering strictly to User Directive 1: Not used as ground-truth accident data. Retained purely as a labeled synthetic benchmark.

---

### SOURCE 6: Chennai Road Crashes Historical Data (OpenCity)
- **Identifier:** `opencity_chennai_crashes`
- **File / Source:** `data_raw/chennai_road_crashes_opencity.csv`
- **URL:** `https://data.opencity.in/dataset/chennai-road-crashes-data`
- **Data Nature:** **REAL / HISTORICAL AGGREGATE STATISTICS**
- **Row Count:** 28 annual rows
- **Time Coverage:** 1998 to 2025
- **Geographic Coverage:** Greater Chennai Police Jurisdiction
- **Fields Available:** `YEAR`, `Fatal Accidents`, `Fatal Killed`, `Grievous Injury - Accidents`, `Grievous Injurty - Injured`, `Minor Injury - Accidents`, `Minor Injury - Injured`, `Non Injury`, `Total Accidents`, `Total number of persons involved`
- **Role in Pipeline:** Macro-level historical temporal indicators and long-term fatality trends.
- **Critical Limitations:** **ANNUAL AGGREGATE ONLY.** Does not contain road-level coordinates. Not used for spatial matching.
- **Licensing:** Greater Chennai Police / OpenCity Public Domain.

---

### SOURCE 7: OpenStreetMap (OSM) Road Network Graph
- **Identifier:** `osm_road_network`
- **Data Nature:** **REAL ROAD NETWORK GRAPH**
- **Row Count:** 52 directed road segments across 19 major Chennai nodes
- **Total Network Length:** 270.4 km
- **Road Classifications:** Trunk (GST Road, NH45), Primary (Anna Salai, Poonamallee High Road, OMR), Secondary (Velachery Bypass, Medavakkam Main Road), Tertiary
- **Attributes:** `segment_id`, `osm_way_id`, `road_name`, `road_type`, `length_m`, `lanes`, `speed_limit`, `district`, `geometry`
- **Role in Pipeline:** The central spatial entity of ROUTEWISE. All risk metrics, live traffic factors, and Dijkstra edge costs attach to these segments.
- **Licensing:** OpenStreetMap Foundation (ODbL).

---

### SOURCE 8: Google Maps Live Traffic
- **Identifier:** `google_live_traffic`
- **URL:** `https://maps.googleapis.com/maps/api/distancematrix`
- **Data Nature:** **LIVE DYNAMIC SENSOR FEED**
- **Role in Pipeline:** Real-time delay factor (`live_traffic_factor = duration_in_traffic / duration`).
- **CRITICAL CONSTRAINTS:**
  - Google is a **LIVE DATA SOURCE ONLY**.
  - Google Maps does **NOT** return the final route.
  - Our **custom Dijkstra implementation** remains the sole route selection engine.
  - When `GOOGLE_MAPS_API_KEY` is not provided, the adapter operates in a transparent time-of-day model explicitly labeled `DEMO / SIMULATED` (morning peak 1.65x, evening peak 1.80x, off-peak 1.05x). Never falsely claims live connectivity.
- **Licensing:** Google Maps Platform Terms of Service.

---

## Pipeline Summary Matrix

| Source ID | Name | Nature | Records | Coordinates | Status | Pipeline Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tn_district_accidents` | TN District Accidents | Real Historical | 38 districts | No (District) | ACTIVE | Baseline Risk Prior |
| `tngis_accident_dashboard` | TNGIS Dashboard | Government GIS | 0 | No (Server error) | RESTRICTED | Adapter documented |
| `irad_edar_tn` | iRAD / eDAR | Government Intranet | 0 | Restricted | RESTRICTED | Adapter interface |
| `adas_mobility_tn` | ADAS Mobility Data | Real Telemetry | 21,325 | Yes (GPS) | ACTIVE | Spatial Risk & ML Target |
| `opencity_chennai_crashes` | Chennai Crashes (OpenCity) | Real Historical | 28 years | No (Annual) | ACTIVE | Macro Trend Modeling |
| `osm_road_network` | OSM Road Graph | Real Network | 52 segments | Yes (Polylines) | ACTIVE | Central Graph Entity |
| `google_live_traffic` | Google Live Traffic | Live Feed | Dynamic | Segment Match | ACTIVE | Dynamic Edge Weights |
| `indian_roads_benchmark` | Indian Roads Dataset | Synthetic | 20,000 | Yes (Synthetic) | EXPERIMENTAL | Benchmark only (No ML) |
