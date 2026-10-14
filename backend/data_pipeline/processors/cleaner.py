"""
backend/data_pipeline/processors/cleaner.py
===========================================
Dataset validation and sanitation utilities.
Inspects datasets and produces formal provenance and quality profiles.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def profile_dataset(
    df: pd.DataFrame,
    source_name: str,
    file_path: Optional[str] = None,
    data_nature: str = "REAL",
    provenance: str = "",
    lat_col: Optional[str] = None,
    lng_col: Optional[str] = None,
    date_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Produce a comprehensive dataset validation profile adhering to
    strict project requirements:
    - row count, columns, date range, geographic coverage,
      coordinate availability, duplicate count, missing-value percentage,
      real/historical/live/synthetic nature, and provenance.
    """
    row_count = int(len(df))
    columns = list(df.columns)
    duplicate_count = int(df.duplicated().sum())
    total_cells = int(df.size)
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0, 2)

    # Date range analysis
    date_range = "N/A"
    if date_col and date_col in df.columns:
        valid_dates = df[date_col].dropna().astype(str)
        if not valid_dates.empty:
            date_range = f"{valid_dates.min()} to {valid_dates.max()}"

    # Coordinate & geographic coverage analysis
    coord_availability = "NO_COORDINATES"
    geographic_coverage = "Not spatially localized"
    if lat_col and lng_col and lat_col in df.columns and lng_col in df.columns:
        valid_coords = df[[lat_col, lng_col]].dropna()
        if not valid_coords.empty:
            min_lat, max_lat = valid_coords[lat_col].min(), valid_coords[lat_col].max()
            min_lng, max_lng = valid_coords[lng_col].min(), valid_coords[lng_col].max()
            coord_availability = f"YES ({len(valid_coords)} valid pairs)"
            geographic_coverage = f"Lat [{min_lat:.4f}, {max_lat:.4f}], Lng [{min_lng:.4f}, {max_lng:.4f}]"

    profile = {
        "source": source_name,
        "file_or_api": file_path or "API / In-memory",
        "row_count": row_count,
        "columns": columns,
        "date_range": date_range,
        "geographic_coverage": geographic_coverage,
        "coordinate_availability": coord_availability,
        "duplicate_count": duplicate_count,
        "missing_value_pct": missing_pct,
        "data_nature": data_nature,
        "provenance": provenance
    }
    return profile


def print_dataset_profile(profile: Dict[str, Any]) -> None:
    """Print formatted inspection profile to terminal."""
    print("--------------------------------------------------")
    print(f"DATASET VALIDATION: {profile['source']}")
    print("--------------------------------------------------")
    print(f"  File/API:             {profile['file_or_api']}")
    print(f"  Nature:               {profile['data_nature']}")
    print(f"  Row Count:            {profile['row_count']:,}")
    print(f"  Columns ({len(profile['columns'])}):         {', '.join(profile['columns'][:6])}{'...' if len(profile['columns']) > 6 else ''}")
    print(f"  Date Range:           {profile['date_range']}")
    print(f"  Coordinate Avail:     {profile['coordinate_availability']}")
    print(f"  Geographic Extent:    {profile['geographic_coverage']}")
    print(f"  Duplicates:           {profile['duplicate_count']:,}")
    print(f"  Missing Value %:      {profile['missing_value_pct']}%")
    print(f"  Provenance:           {profile['provenance']}")
    print("--------------------------------------------------")
