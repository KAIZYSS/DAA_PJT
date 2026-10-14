"""
backend/data_pipeline/collectors/opencity.py
============================================
Adapter for SOURCE 6: Chennai Road Crashes Historical Data (OpenCity / Greater Chennai Police).
Dataset: https://data.opencity.in/dataset/chennai-road-crashes-data
VERIFIED: This dataset is AGGREGATE YEARLY STATISTICS (1998-2025).
It does NOT contain individual road-level coordinates.
Used strictly for temporal historical trends, NOT spatial point matching.
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime, timezone
from backend.data_pipeline.processors.cleaner import profile_dataset
from backend.data_pipeline.storage.database import save_data_source, save_validation_log, DEFAULT_DB_PATH

CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_raw", "chennai_road_crashes_opencity.csv")
)


def collect_opencity_chennai(
    csv_path: str = CSV_PATH,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingest and validate the OpenCity Chennai Road Crashes Dataset.
    Produces annual temporal historical trend metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OpenCity Chennai crashes CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    profile = profile_dataset(
        df=df,
        source_name="OpenCity Chennai Road Crashes Historical Data",
        file_path=csv_path,
        data_nature="REAL",
        provenance="Greater Chennai Police (GCP) / OpenCity CKAN repository",
        date_col="YEAR"
    )

    # Clean numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute macro temporal indicators
    df["macro_fatality_ratio"] = (df["Fatal Killed"] / df["Total Accidents"].replace(0, 1)).clip(0.0, 1.0)
    df["grievous_injury_ratio"] = (df["Grievous Injurty - Injured"] / df["Total Accidents"].replace(0, 1)).clip(0.0, 1.0)

    # 5-year rolling average of total accidents
    df["accidents_5y_moving_avg"] = df["Total Accidents"].rolling(window=5, min_periods=1).mean().round(1)

    source_record = {
        "id": "opencity_chennai_crashes",
        "name": "Chennai Road Crashes Data (OpenCity)",
        "url": "https://data.opencity.in/dataset/chennai-road-crashes-data",
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "geographic_coverage": "Greater Chennai Police Jurisdiction (City-wide)",
        "time_coverage": f"{int(df['YEAR'].min())}-{int(df['YEAR'].max())}",
        "fields_available": list(df.columns),
        "is_historical": True,
        "is_live": False,
        "is_synthetic": False,
        "accessible": True,
        "status": "ACTIVE",
        "record_count": len(df),
        "limitations": "AGGREGATE ANNUAL STATISTICS ONLY (1998-2025). Contains no individual road coordinates.",
        "licensing_notes": "Public Domain (Open Definition / OpenCity.in)"
    }
    save_data_source(source_record, db_path=db_path)
    save_validation_log({
        "source_id": "opencity_chennai_crashes",
        "file_path": csv_path,
        "row_count": len(df),
        "columns": list(df.columns),
        "date_range": f"{int(df['YEAR'].min())}-{int(df['YEAR'].max())}",
        "coord_availability": "NO_COORDINATES (Annual aggregate statistics)",
        "duplicate_count": int(df.duplicated().sum()),
        "missing_pct": profile["missing_value_pct"],
        "data_nature": "REAL"
    }, db_path=db_path)

    return df, profile
