"""
backend/data_pipeline/run_pipeline.py
=====================================
Master Orchestration Script for ROUTEWISE Data + ML Pipeline.
Executes end-to-end:
1. Data Ingestion & Validation for all 8 sources
2. Spatial Matching onto OSM Polylines
3. Feature Engineering with Strict Chronological Guardrails
4. ML Model Training & Evaluation
5. Road Segment Risk Calibration
6. Comprehensive Pipeline Report
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_pipeline.storage.database import (
    init_pipeline_tables, save_anomalies_batch, save_road_segments,
    get_all_road_segments, DEFAULT_DB_PATH
)
from backend.data_pipeline.processors.cleaner import profile_dataset, print_dataset_profile
from backend.data_pipeline.processors.spatial_matcher import SpatialMatcher
from backend.data_pipeline.processors.feature_builder import build_temporal_segment_dataset
from backend.data_pipeline.collectors.tn_accidents import collect_tn_district_accidents, get_district_risk_lookup
from backend.data_pipeline.collectors.adas import collect_adas_mobility
from backend.data_pipeline.collectors.opencity import collect_opencity_chennai
from backend.data_pipeline.collectors.tn_dashboard import collect_tngis_dashboard
from backend.data_pipeline.collectors.irad import IRADAdapter
from backend.data_pipeline.collectors.osm import collect_osm_road_segments
from backend.data_pipeline.collectors.google_traffic import GoogleTrafficCollector
from backend.data_pipeline.models.train import train_risk_model, SAVED_MODEL_PATH
from backend.data_pipeline.models.predict import predict_segment_risk


def run_pipeline(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Execute the full data and machine learning pipeline."""
    print("==================================================")
    print("STARTING ROUTEWISE DATA + ML INGESTION PIPELINE")
    print("==================================================")

    # 1. Initialize Storage
    init_pipeline_tables(db_path=db_path)

    # 2. Ingest & Validate Source 1 (TN District Accidents)
    df_district, prof_district = collect_tn_district_accidents(db_path=db_path)
    district_lookup = get_district_risk_lookup()

    # 3. Ingest & Validate Source 4 (ADAS Mobility Data)
    adas_records, prof_adas = collect_adas_mobility(db_path=db_path)

    # 4. Ingest & Validate Source 6 (OpenCity Chennai Road Crashes)
    df_opencity, prof_opencity = collect_opencity_chennai(db_path=db_path)

    # 5. Ingest & Validate Source 5 (Indian Roads Dataset - Evaluated as Synthetic Benchmark)
    raw_synthetic_path = os.path.join(PROJECT_ROOT, "data_raw", "indian_roads_dataset.csv")
    prof_synthetic = None
    if os.path.exists(raw_synthetic_path):
        df_synth = pd.read_csv(raw_synthetic_path)
        prof_synthetic = profile_dataset(
            df=df_synth,
            source_name="Indian Roads Dataset (Kaggle Benchmark)",
            file_path=raw_synthetic_path,
            data_nature="SYNTHETIC / EXPERIMENTAL BENCHMARK",
            provenance="Kaggle synthetic road accident dataset. Used only for experimental comparative testing.",
            lat_col="latitude",
            lng_col="longitude",
            date_col="date"
        )

    # 6. Probe Source 2 (TNGIS Dashboard)
    source_tngis = collect_tngis_dashboard(timeout_sec=4, db_path=db_path)

    # 7. Probe Source 3 (iRAD / eDAR)
    irad_adapter = IRADAdapter()
    source_irad = irad_adapter.collect(db_path=db_path)

    # 8. Ingest Source 7 (OSM Road Graph)
    road_segments, osm_summary = collect_osm_road_segments(db_path=db_path)

    # 9. Register Source 8 (Google Live Traffic)
    traffic_collector = GoogleTrafficCollector()
    traffic_collector.register_source(db_path=db_path)

    # 10. SPATIAL MATCHING onto OSM Polylines
    matcher = SpatialMatcher(road_segments, urban_threshold_m=150.0, highway_threshold_m=250.0)
    matched_adas, match_stats = matcher.match_anomalies(adas_records)

    # Persist matched anomalies to database
    save_anomalies_batch(matched_adas, db_path=db_path)

    # Update segment warning counts from spatial matches
    segment_warning_counts = {}
    for a in matched_adas:
        sid = a.get("road_segment_id")
        if sid:
            segment_warning_counts[sid] = segment_warning_counts.get(sid, 0) + 1

    for seg in road_segments:
        sid = seg["segment_id"]
        seg["adas_warning_count"] = segment_warning_counts.get(sid, 0)

    # 11. FEATURE ENGINEERING
    df_ml, feature_summary = build_temporal_segment_dataset(
        road_segments=road_segments,
        matched_anomalies=matched_adas,
        district_risk_lookup=district_lookup,
        window_days=7
    )

    # 12. MODEL TRAINING & EVALUATION
    model_result = train_risk_model(df_ml, db_path=db_path)

    # 13. PREDICTED RISK CALIBRATION ON ALL ROAD SEGMENTS
    for seg in road_segments:
        sid = seg["segment_id"]
        w_count = seg["adas_warning_count"]
        length_km = max(0.1, seg["length_m"] / 1000.0)
        density = round(w_count / length_km, 3)
        dist_name = str(seg.get("district", "CHENNAI CITY")).strip().upper()
        d_risk = district_lookup.get(dist_name, {}).get("baseline_risk", 0.4)

        pred_res = predict_segment_risk({
            "road_type_enc": 3 if seg["road_type"] == "trunk" else 2,
            "length_m": seg["length_m"],
            "lanes": seg["lanes"],
            "speed_limit": seg["speed_limit"],
            "district_baseline_risk": d_risk,
            "district_fatality_rate": district_lookup.get(dist_name, {}).get("fatality_rate", 0.2),
            "fcw_count": int(w_count * 0.05),
            "pcw_count": int(w_count * 0.10),
            "ldw_count": int(w_count * 0.35),
            "hmw_count": int(w_count * 0.50),
            "total_adas_warnings": w_count,
            "adas_warning_density": density
        })
        seg["predicted_risk"] = pred_res["predicted_risk"]
        seg["risk_score"] = round(0.4 * d_risk + 0.6 * pred_res["predicted_risk"], 3)

    # Persist updated segments with ML predictions
    save_road_segments(road_segments, db_path=db_path)

    # Count matched segments
    matched_osm_segments = len(set(a["road_segment_id"] for a in matched_adas if a.get("road_segment_id")))

    # 14. GENERATE STRUCTURED REPORT
    print("\n====================================")
    print("ROUTEWISE DATA PIPELINE REPORT")
    print("====================================")

    print("\nSOURCE: TN District Accidents")
    print(f"Records:               {len(df_district)}")
    print(f"Date range:            2021 - 2023")
    print(f"Coverage:              38 Districts of Tamil Nadu (District aggregates)")
    print(f"Status:                ACTIVE (Baseline risk priors extracted)")

    print("\nSOURCE: ADAS")
    print(f"Records:               {prof_adas['row_count']:,}")
    print(f"Coordinates:           {prof_adas['coordinate_availability']}")
    print(f"Date range:            {prof_adas['date_range']}")
    print(f"Matched OSM segments:  {matched_osm_segments} segments ({match_stats['matched_count']:,} alerts)")

    print("\nSOURCE: OpenCity")
    print(f"Records:               {len(df_opencity)} annual rows")
    print(f"Type:                  AGGREGATE ANNUAL CRASH STATISTICS")
    print(f"Date range:            {prof_opencity['date_range']}")

    print("\nSOURCE: TNGIS")
    print(f"Status:                {source_tngis['status']}")
    print(f"Accessible:            {source_tngis['accessible']}")

    print("\nSOURCE: iRAD")
    print(f"Status:                {source_irad['status']}")
    print(f"Accessible:            {source_irad['accessible']}")

    print("\nSOURCE: OSM")
    print(f"Road segments:         {len(road_segments)}")
    print(f"Coverage:              {osm_summary['total_length_km']} km (Chennai & Arterial Corridors)")

    print("\n====================================")
    print("SPATIAL MATCHING")
    print("====================================")
    print(f"Total coordinate records: {match_stats['total_coordinate_records']:,}")
    print(f"Matched:                  {match_stats['matched_count']:,}")
    print(f"Unmatched:                {match_stats['unmatched_count']:,}")
    print(f"Mean match distance:      {match_stats['mean_match_distance_m']} m")
    print(f"Median match distance:    {match_stats['median_match_distance_m']} m")

    print("\n====================================")
    print("ML DATASET")
    print("====================================")
    if model_result["status"] == "SUCCESS":
        print(f"Training samples:         {model_result['train_samples']}")
        print(f"Validation:               {model_result['val_samples']}")
        print(f"Test:                     {model_result['test_samples']}")
        print(f"Positive:                 {model_result['positive_samples']}")
        print(f"Negative:                 {model_result['negative_samples']}")
    else:
        print(f"Status:                   {model_result.get('message')}")

    print("\n====================================")
    print("MODEL")
    print("====================================")
    if model_result["status"] == "SUCCESS":
        rf_m = model_result["rf_metrics"]
        print(f"Model:                    {rf_m['model_name']}")
        print(f"F1:                       {rf_m['f1']}")
        print(f"ROC-AUC:                  {rf_m['roc_auc']}")
        print(f"PR-AUC:                   {rf_m['pr_auc']}")
        print(f"Accuracy:                 {rf_m['accuracy']} (vs Baseline: {model_result['baseline_metrics']['accuracy']})")
    else:
        print(f"Model Status:             {model_result.get('message')}")

    print("\n====================================")
    print("ROUTEWISE READY")
    print("====================================\n")

    return {
        "sources": {
            "tn_district": len(df_district),
            "adas": len(adas_records),
            "opencity": len(df_opencity),
            "tngis": source_tngis["status"],
            "irad": source_irad["status"],
            "osm": len(road_segments)
        },
        "spatial_matching": match_stats,
        "ml_result": model_result
    }


if __name__ == "__main__":
    run_pipeline()
