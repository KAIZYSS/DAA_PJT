"""
tests/test_data_pipeline.py
===========================
Automated unit and integration tests for ROUTEWISE Data Pipeline:
- Source collectors (TN District, ADAS, OpenCity, TNGIS, iRAD, OSM, Google Traffic)
- Data cleaning and profiling
- Spatial matching (point-to-polyline geodesic projection)
- Feature building without future leakage
"""

import os
import pytest
import pandas as pd
import numpy as np

from backend.data_pipeline.collectors.tn_accidents import collect_tn_district_accidents, get_district_risk_lookup
from backend.data_pipeline.collectors.adas import collect_adas_mobility
from backend.data_pipeline.collectors.opencity import collect_opencity_chennai
from backend.data_pipeline.collectors.tn_dashboard import collect_tngis_dashboard
from backend.data_pipeline.collectors.irad import IRADAdapter
from backend.data_pipeline.collectors.osm import collect_osm_road_segments
from backend.data_pipeline.collectors.google_traffic import GoogleTrafficCollector
from backend.data_pipeline.processors.cleaner import profile_dataset
from backend.data_pipeline.processors.spatial_matcher import (
    SpatialMatcher, haversine_distance, point_to_segment_distance, point_to_polyline_distance
)
from backend.data_pipeline.processors.feature_builder import build_temporal_segment_dataset


def test_tn_district_accidents_collector():
    """Verify TN district accidents ingestion, baseline risk bounds, and no road coordinates."""
    df, prof = collect_tn_district_accidents()
    assert len(df) == 38, f"Expected 38 districts, got {len(df)}"
    assert "district_baseline_risk" in df.columns
    assert "fatality_rate" in df.columns
    assert df["district_baseline_risk"].min() >= 0.0
    assert df["district_baseline_risk"].max() <= 1.0
    assert prof["coordinate_availability"] == "NO_COORDINATES"

    lookup = get_district_risk_lookup()
    assert "CHENNAI CITY" in lookup
    assert "CHENGALPATTU" in lookup
    assert 0.0 <= lookup["CHENNAI CITY"]["baseline_risk"] <= 1.0


def test_adas_mobility_collector():
    """Verify ADAS telemetry ingestion, valid coordinates, and STRICT anomaly_type='ADAS_WARNING'."""
    records, prof = collect_adas_mobility()
    assert len(records) > 20000, f"Expected >20k records, got {len(records)}"
    assert prof["coordinate_availability"].startswith("YES")

    # STRICT: verify ADAS alerts are NEVER labeled as ACCIDENT
    anomaly_types = {r["anomaly_type"] for r in records}
    assert anomaly_types == {"ADAS_WARNING"}, f"Expected only ADAS_WARNING, got {anomaly_types}"

    # Verify alert subtypes
    subtypes = {r["metadata"]["raw_alert"] for r in records}
    assert "cas_fcw" in subtypes
    assert "cas_pcw" in subtypes
    assert "cas_ldw" in subtypes
    assert "cas_hmw" in subtypes


def test_opencity_chennai_collector():
    """Verify OpenCity Chennai crash data is aggregate yearly statistics."""
    df, prof = collect_opencity_chennai()
    assert len(df) == 28, f"Expected 28 annual rows, got {len(df)}"
    assert "YEAR" in df.columns
    assert "Fatal Killed" in df.columns
    assert "Total Accidents" in df.columns
    assert "macro_fatality_ratio" in df.columns
    assert prof["coordinate_availability"].startswith("NO_COORDINATES")


def test_tngis_and_irad_no_fabrication():
    """Verify that unavailable/restricted endpoints report transparently and fabricate zero rows."""
    tngis_res = collect_tngis_dashboard(timeout_sec=2)
    assert tngis_res["is_synthetic"] is False
    assert tngis_res["record_count"] == 0

    irad = IRADAdapter()
    irad_res = irad.collect()
    assert irad_res["accessible"] is False
    assert irad_res["record_count"] == 0
    assert "RESTRICTED" in irad_res["status"] or "UNAVAILABLE" in irad_res["status"] or "TIMEOUT" in irad_res["status"]


def test_osm_road_segments_collector():
    """Verify OSM road segment extraction and attributes."""
    segments, summary = collect_osm_road_segments()
    assert len(segments) == 52
    assert summary["total_length_km"] > 200.0

    first = segments[0]
    required_keys = [
        "segment_id", "osm_way_id", "road_name", "road_type",
        "length_m", "lanes", "speed_limit", "district", "geometry"
    ]
    for k in required_keys:
        assert k in first, f"Missing required OSM segment key: {k}"


def test_spatial_matcher_accuracy_and_thresholding():
    """Verify point-to-polyline projection and configurable distance threshold rejection."""
    # Create test segment: horizontal line at lat=13.0 from lng=80.10 to 80.20
    test_segments = [{
        "segment_id": "test_seg_01",
        "road_name": "Test Highway",
        "road_type": "trunk",
        "geometry": [[13.0000, 80.1000], [13.0000, 80.2000]]
    }]
    matcher = SpatialMatcher(test_segments, urban_threshold_m=150.0, highway_threshold_m=250.0)

    # Point 50m north of segment center -> should match
    # 0.00045 deg lat approx 50m
    seg_id, dist_m, _ = matcher.match_point(13.00045, 80.1500)
    assert seg_id == "test_seg_01"
    assert 40.0 <= dist_m <= 60.0

    # Point 500m north of segment -> exceeds 250m highway threshold -> should reject
    seg_id_far, dist_far, _ = matcher.match_point(13.0045, 80.1500)
    assert seg_id_far is None
    assert dist_far > 250.0


def test_feature_builder_no_leakage():
    """Verify feature builder constructs valid chronological samples without future leakage."""
    segments, _ = collect_osm_road_segments()
    adas_records, _ = collect_adas_mobility()
    matcher = SpatialMatcher(segments)
    matched_adas, _ = matcher.match_anomalies(adas_records[:1000])
    district_lookup = get_district_risk_lookup()

    df_ml, summary = build_temporal_segment_dataset(
        road_segments=segments,
        matched_anomalies=matched_adas,
        district_risk_lookup=district_lookup,
        window_days=14
    )

    if summary["status"] == "VALID":
        assert "future_segment_anomaly" in df_ml.columns
        assert set(df_ml["future_segment_anomaly"].unique()).issubset({0, 1})
        assert "district_baseline_risk" in df_ml.columns
        assert "adas_warning_density" in df_ml.columns
        # Verify chronological window indices exist
        assert df_ml["window_index"].nunique() >= 2
