"""
backend/data_pipeline/collectors/tn_accidents.py
================================================
Adapter for SOURCE 1: Tamil Nadu District Road Accident Dataset (2021-2023).
Extracts district-level accident totals, fatalities, trends, and baseline risk priors.
IMPORTANT: District-level only; does NOT contain individual road accident coordinates.
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime, timezone
from backend.data_pipeline.processors.cleaner import profile_dataset
from backend.data_pipeline.storage.database import save_data_source, save_validation_log, DEFAULT_DB_PATH

CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_raw", "tn_district_accidents.csv")
)


def collect_tn_district_accidents(
    csv_path: str = CSV_PATH,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingest and validate the Tamil Nadu District Road Accident Dataset.
    Produces district baseline risk metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"District accidents CSV not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Validation profile before cleaning
    profile = profile_dataset(
        df=df_raw,
        source_name="Tamil Nadu District Road Accident Dataset",
        file_path=csv_path,
        data_nature="REAL",
        provenance="Government of Tamil Nadu, State Police & Road Safety (2021-2023)"
    )

    # Filter out summary row (e.g. 'Total' row)
    df = df_raw[df_raw["District"].str.strip().str.lower() != "total"].copy()

    # Clean numeric columns
    numeric_cols = [
        "Total Accidents 2021", "Total Deaths 2021",
        "Total Accidents 2022", "Total Deaths 2022",
        "Total Accidents 2023", "Total Deaths 2023"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Compute aggregate district metrics
    df["accidents_3y"] = df["Total Accidents 2021"] + df["Total Accidents 2022"] + df["Total Accidents 2023"]
    df["fatalities_3y"] = df["Total Deaths 2021"] + df["Total Deaths 2022"] + df["Total Deaths 2023"]
    df["fatality_rate"] = (df["fatalities_3y"] / df["accidents_3y"].replace(0, 1)).clip(0.0, 1.0)
    df["accident_trend"] = df["Total Accidents 2023"] - df["Total Accidents 2021"]

    # Normalize district baseline risk index between 0.0 and 1.0
    max_acc = df["accidents_3y"].max()
    min_acc = df["accidents_3y"].min()
    if max_acc > min_acc:
        df["district_baseline_risk"] = ((df["accidents_3y"] - min_acc) / (max_acc - min_acc)).round(4)
    else:
        df["district_baseline_risk"] = 0.5

    # Persist data source provenance & validation log
    source_record = {
        "id": "tn_district_accidents",
        "name": "Tamil Nadu District Road Accident Dataset",
        "url": "Provided official state dataset (data_raw/tn_district_accidents.csv)",
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "geographic_coverage": "38 Districts across Tamil Nadu state",
        "time_coverage": "2021-2023",
        "fields_available": list(df.columns),
        "is_historical": True,
        "is_live": False,
        "is_synthetic": False,
        "accessible": True,
        "status": "ACTIVE",
        "record_count": len(df),
        "limitations": "District-level aggregate only. Does NOT contain road-level coordinates.",
        "licensing_notes": "Official state road safety statistics. Open public reporting."
    }
    save_data_source(source_record, db_path=db_path)
    save_validation_log({
        "source_id": "tn_district_accidents",
        "file_path": csv_path,
        "row_count": len(df),
        "columns": list(df.columns),
        "date_range": "2021-2023",
        "coord_availability": "NO_COORDINATES (District aggregates)",
        "duplicate_count": int(df_raw.duplicated().sum()),
        "missing_pct": profile["missing_value_pct"],
        "data_nature": "REAL"
    }, db_path=db_path)

    return df, profile


def get_district_risk_lookup(csv_path: str = CSV_PATH) -> Dict[str, Dict[str, float]]:
    """Return dictionary mapping normalized district name to baseline risk metrics."""
    df, _ = collect_tn_district_accidents(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        name = str(row["District"]).strip().upper()
        lookup[name] = {
            "baseline_risk": float(row["district_baseline_risk"]),
            "fatality_rate": float(row["fatality_rate"]),
            "accidents_3y": int(row["accidents_3y"]),
            "fatalities_3y": int(row["fatalities_3y"]),
            "accident_trend": float(row["accident_trend"])
        }
    return lookup
