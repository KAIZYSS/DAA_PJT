"""
backend/data_pipeline/processors/feature_builder.py
===================================================
Constructs tabular feature matrices for road segments across chronological time windows.
Strictly avoids data leakage:
- Features at time window T ONLY use information available at or before T.
- Future target variable 'future_segment_anomaly' is evaluated strictly at window T+1.
- Live traffic is NEVER leaked into historical ML training.
- District accidents are explicitly represented as district-level priors.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROAD_TYPE_MAP = {
    "trunk": 3,
    "primary": 2,
    "secondary": 1,
    "tertiary": 0
}


def build_temporal_segment_dataset(
    road_segments: List[Dict[str, Any]],
    matched_anomalies: List[Dict[str, Any]],
    district_risk_lookup: Dict[str, Dict[str, float]],
    window_days: int = 7
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct a time-sliced road segment dataset.
    Divides the anomaly timeline into chronological windows of length 'window_days'.
    At each window T, computes historical features strictly from events <= T.
    Evaluates target 'future_segment_anomaly' strictly in window T+1.
    """
    # Extract timestamped anomalies that have a valid road segment match
    timestamped = []
    for a in matched_anomalies:
        if not a.get("road_segment_id") or not a.get("timestamp"):
            continue
        try:
            ts = pd.to_datetime(a["timestamp"])
            timestamped.append({
                "road_segment_id": a["road_segment_id"],
                "timestamp": ts,
                "anomaly_type": a["anomaly_type"],
                "severity": a.get("severity", 1.0),
                "specific_alert": a.get("metadata", {}).get("raw_alert", "")
            })
        except Exception:
            continue

    if not timestamped:
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA",
            "message": "Insufficient real temporal/location data for supervised prediction."
        }

    df_events = pd.DataFrame(timestamped)
    min_date = df_events["timestamp"].min().floor("D")
    max_date = df_events["timestamp"].max().ceil("D")

    # Generate sequential time windows
    window_boundaries = []
    curr = min_date
    while curr < max_date:
        w_end = curr + timedelta(days=window_days)
        window_boundaries.append((curr, w_end))
        curr = w_end

    if len(window_boundaries) < 2:
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_WINDOWS",
            "message": f"Only {len(window_boundaries)} time window available; minimum 2 required for chronological T -> T+1 prediction."
        }

    rows = []
    # Build observations for each window t (except the last window, which has no future window t+1)
    for w_idx in range(len(window_boundaries) - 1):
        t_start, t_end = window_boundaries[w_idx]
        future_start, future_end = window_boundaries[w_idx + 1]

        # Events strictly before or during window t (NO FUTURE LEAKAGE)
        hist_events = df_events[df_events["timestamp"] < t_end]

        # Future events strictly in window t+1
        future_events = df_events[(df_events["timestamp"] >= future_start) & (df_events["timestamp"] < future_end)]

        for seg in road_segments:
            seg_id = seg["segment_id"]
            seg_hist = hist_events[hist_events["road_segment_id"] == seg_id]
            seg_future = future_events[future_events["road_segment_id"] == seg_id]

            # ADAS warning counts strictly up to t_end
            fcw_count = int((seg_hist["specific_alert"] == "cas_fcw").sum())
            pcw_count = int((seg_hist["specific_alert"] == "cas_pcw").sum())
            ldw_count = int((seg_hist["specific_alert"] == "cas_ldw").sum())
            hmw_count = int((seg_hist["specific_alert"] == "cas_hmw").sum())
            total_warnings = fcw_count + pcw_count + ldw_count + hmw_count

            length_km = max(0.1, seg.get("length_m", 1000.0) / 1000.0)
            warning_density = round(total_warnings / length_km, 3)

            # District prior
            dist_name = str(seg.get("district", "CHENNAI CITY")).strip().upper()
            d_prior = district_risk_lookup.get(dist_name, {
                "baseline_risk": 0.5,
                "fatality_rate": 0.2,
                "accidents_3y": 3000,
                "fatalities_3y": 600,
                "accident_trend": 0
            })

            # Road attributes
            r_type = seg.get("road_type", "primary")
            r_type_enc = ROAD_TYPE_MAP.get(r_type, 1)
            lanes = int(seg.get("lanes", 2))
            speed = float(seg.get("speed_limit", 40.0))

            # Target: 1 if qualifying safety anomaly (FCW or PCW collision threat or >0 incidents) occurs in future window t+1
            # 0 otherwise
            qualifying_future = int(
                ((seg_future["specific_alert"].isin(["cas_fcw", "cas_pcw"])) |
                 (seg_future["severity"] >= 7.0)).sum()
            )
            target = 1 if qualifying_future > 0 else 0

            rows.append({
                "window_index": w_idx,
                "window_start": t_start,
                "window_end": t_end,
                "segment_id": seg_id,
                "road_name": seg.get("road_name", ""),
                "road_type_enc": r_type_enc,
                "length_m": seg.get("length_m", 1000.0),
                "lanes": lanes,
                "speed_limit": speed,
                "district_baseline_risk": d_prior["baseline_risk"],
                "district_fatality_rate": d_prior["fatality_rate"],
                "fcw_count": fcw_count,
                "pcw_count": pcw_count,
                "ldw_count": ldw_count,
                "hmw_count": hmw_count,
                "total_adas_warnings": total_warnings,
                "adas_warning_density": warning_density,
                "future_segment_anomaly": target
            })

    df_ml = pd.DataFrame(rows)
    pos_count = int((df_ml["future_segment_anomaly"] == 1).sum())
    neg_count = int((df_ml["future_segment_anomaly"] == 0).sum())

    summary = {
        "status": "VALID",
        "total_samples": len(df_ml),
        "total_windows": len(window_boundaries) - 1,
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "positive_ratio": round(pos_count / max(1, len(df_ml)), 4),
        "feature_count": 12
    }
    return df_ml, summary
