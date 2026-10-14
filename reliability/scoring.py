"""
reliability/scoring.py
======================
Mathematical scoring models for calculating road reliability and risk scores.

Formulation (Review 1 Spec Section 18):
---------------------------------------
Three normalized sub-scores in [0.0, 1.0] are derived from anomaly records:
1. frequency_score: Normalized count of anomalies on the road relative to
   the highest anomaly count observed in the network:
       frequency_score = count / max(max_count, 1)

2. severity_score: Normalized mean severity of the road's anomalies (severity
   scale is 1 to 10 in the data model):
       severity_score = (sum(severity) / count) / 10.0

3. downtime_score: Normalized total anomaly duration relative to the maximum
   total downtime observed in the network:
       downtime_score = total_duration_minutes / max(max_downtime, 1)

Combined Risk Formula:
----------------------
risk_score = 0.40 * frequency_score + 0.35 * severity_score + 0.25 * downtime_score
Clamped strictly to [0.0, 1.0].

Reliability Percentage:
-----------------------
reliability_pct = (1.0 - risk_score) * 100.0
"""

from typing import List, Dict, Any


def compute_road_risk_score(
    anomalies: List[Dict[str, Any]],
    max_count_across_roads: int = 1,
    max_downtime_across_roads: int = 1
) -> float:
    """
    Compute normalized risk score for a single road based on its historical anomalies.

    :param anomalies: List of anomaly dicts associated with this road.
    :param max_count_across_roads: Normalization baseline for anomaly frequency.
    :param max_downtime_across_roads: Normalization baseline for anomaly downtime.
    :return: Clamped risk score between 0.0 (safest) and 1.0 (most hazardous).
    """
    if not anomalies:
        return 0.0

    count = len(anomalies)
    frequency_score = min(1.0, count / max(max_count_across_roads, 1))

    total_severity = sum(a.get("severity", 1) for a in anomalies)
    avg_severity = total_severity / count
    severity_score = min(1.0, max(0.0, avg_severity / 10.0))

    total_downtime = sum(a.get("duration_minutes", 0) for a in anomalies)
    downtime_score = min(1.0, total_downtime / max(max_downtime_across_roads, 1))

    raw_risk = (0.40 * frequency_score) + (0.35 * severity_score) + (0.25 * downtime_score)
    return round(min(1.0, max(0.0, raw_risk)), 3)


def compute_all_risk_scores(
    roads: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Batch compute risk scores for all roads in the network, deriving the true
    max_count and max_downtime baselines across the dataset.

    :param roads: List of road dicts.
    :param anomalies: List of all anomaly dicts.
    :return: Dictionary mapping road_id -> risk_score.
    """
    # Group anomalies by road_id
    road_anomalies: Dict[str, List[Dict[str, Any]]] = {r["id"]: [] for r in roads}
    for a in anomalies:
        rid = a.get("road_id")
        if rid in road_anomalies:
            road_anomalies[rid].append(a)

    # Compute network-wide maximums
    max_count = max((len(anoms) for anoms in road_anomalies.values()), default=1)
    max_downtime = max(
        (sum(a.get("duration_minutes", 0) for a in anoms) for anoms in road_anomalies.values()),
        default=1
    )

    scores = {}
    for r in roads:
        rid = r["id"]
        anoms = road_anomalies[rid]
        scores[rid] = compute_road_risk_score(anoms, max_count, max_downtime)

    return scores


def get_risk_tier(risk_score: float) -> str:
    """
    Categorize a risk score into one of three visual tiers (Spec Section 12):
    - Low: 0.00 - 0.33
    - Moderate: 0.34 - 0.66
    - High: 0.67 - 1.00
    """
    if risk_score <= 0.33:
        return "low"
    elif risk_score <= 0.66:
        return "moderate"
    else:
        return "high"
