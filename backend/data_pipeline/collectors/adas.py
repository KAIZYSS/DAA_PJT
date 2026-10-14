"""
backend/data_pipeline/collectors/adas.py
========================================
Adapter for SOURCE 4: Tamil Nadu ADAS Mobility Dataset.
Kaggle dataset: https://www.kaggle.com/datasets/bharathtoutireddy/advanced-driver-assistance-systems-mobility-data
Extracts 21,325 GPS warning observations (cas_ldw, cas_hmw, cas_fcw, cas_pcw)
along Tamil Nadu / Chennai arterial corridors.
CRITICAL: ADAS warnings are mobility/anomaly indicators, NEVER labeled as crashes.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from backend.data_pipeline.processors.cleaner import profile_dataset
from backend.data_pipeline.processors.normalizer import normalize_adas_record
from backend.data_pipeline.storage.database import save_data_source, save_validation_log, DEFAULT_DB_PATH

CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_raw", "ADAS Mobility Data.csv")
)


def collect_adas_mobility(
    csv_path: str = CSV_PATH,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ingest, clean, and normalize the ADAS Mobility Dataset.
    Returns normalized anomaly records (anomaly_type='ADAS_WARNING') and validation profile.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ADAS Mobility CSV not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Produce formal validation profile
    profile = profile_dataset(
        df=df_raw,
        source_name="Tamil Nadu ADAS Mobility Dataset",
        file_path=csv_path,
        data_nature="REAL",
        provenance="Fleet Vehicle ADAS Telemetry, Kaggle (B. Toutireddy)",
        lat_col="Lat",
        lng_col="Long",
        date_col="Date"
    )

    # Clean and deduplicate
    df = df_raw.drop_duplicates(subset=["Alert", "Date", "Time", "Lat", "Long", "Vehicle"]).copy()

    # Filter out invalid coordinates
    df = df.dropna(subset=["Lat", "Long"])
    df = df[(df["Lat"].between(8.0, 14.0)) & (df["Long"].between(76.0, 81.0))]

    # Convert to normalized anomaly records
    normalized_records = []
    for idx, row in df.iterrows():
        record = normalize_adas_record(row.to_dict(), index=idx)
        normalized_records.append(record)

    # Record data source provenance
    source_record = {
        "id": "adas_mobility_tn",
        "name": "Tamil Nadu ADAS Mobility Dataset",
        "url": "https://www.kaggle.com/datasets/bharathtoutireddy/advanced-driver-assistance-systems-mobility-data",
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "geographic_coverage": profile["geographic_coverage"],
        "time_coverage": profile["date_range"],
        "fields_available": list(df_raw.columns),
        "is_historical": True,
        "is_live": False,
        "is_synthetic": False,
        "accessible": True,
        "status": "ACTIVE",
        "record_count": len(normalized_records),
        "limitations": "Sensor warnings from 5 test vehicles over 3 months (Jun-Aug 2022). Not crash records.",
        "licensing_notes": "Kaggle Open Dataset (CC BY-SA 4.0)"
    }
    save_data_source(source_record, db_path=db_path)
    save_validation_log({
        "source_id": "adas_mobility_tn",
        "file_path": csv_path,
        "row_count": len(normalized_records),
        "columns": list(df_raw.columns),
        "date_range": profile["date_range"],
        "coord_availability": profile["coordinate_availability"],
        "duplicate_count": profile["duplicate_count"],
        "missing_pct": profile["missing_value_pct"],
        "data_nature": "REAL"
    }, db_path=db_path)

    return normalized_records, profile
