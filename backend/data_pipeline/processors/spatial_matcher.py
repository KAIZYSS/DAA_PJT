"""
backend/data_pipeline/processors/spatial_matcher.py
===================================================
High-accuracy spatial matcher projecting coordinate points onto OSM polyline road geometries.
Calculates perpendicular distance in meters using geodesic projections.
Enforces strict distance thresholds:
- Urban / residential: 150m
- Primary / Trunk / Highway: 250m
Rejects distant points and tracks comprehensive matching statistics.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def point_to_segment_distance(
    p_lat: float, p_lng: float,
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float
) -> Tuple[float, float, float]:
    """
    Compute minimal distance in meters from point P to line segment AB,
    and return (distance_m, nearest_lat, nearest_lng).
    """
    # Deg-to-meter conversion factors around average latitude
    lat_mid = math.radians((a_lat + b_lat) / 2.0)
    cos_lat = math.cos(lat_mid)
    m_per_deg_lat = 111132.95
    m_per_deg_lng = 111412.84 * cos_lat

    # Vector AB in meters
    dx = (b_lng - a_lng) * m_per_deg_lng
    dy = (b_lat - a_lat) * m_per_deg_lat
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        dist = haversine_distance(p_lat, p_lng, a_lat, a_lng)
        return dist, a_lat, a_lng

    # Vector AP in meters
    px = (p_lng - a_lng) * m_per_deg_lng
    py = (p_lat - a_lat) * m_per_deg_lat

    # Projection factor t clamped to [0, 1]
    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))

    nearest_lng = a_lng + t * (b_lng - a_lng)
    nearest_lat = a_lat + t * (b_lat - a_lat)
    dist = haversine_distance(p_lat, p_lng, nearest_lat, nearest_lng)
    return dist, nearest_lat, nearest_lng


def point_to_polyline_distance(
    p_lat: float, p_lng: float,
    polyline: List[List[float]]
) -> Tuple[float, float, float]:
    """
    Compute minimal distance in meters from point P to an ordered polyline [[lat, lng], ...].
    """
    if not polyline:
        return float("inf"), p_lat, p_lng
    if len(polyline) == 1:
        d = haversine_distance(p_lat, p_lng, polyline[0][0], polyline[0][1])
        return d, polyline[0][0], polyline[0][1]

    min_dist = float("inf")
    best_lat = polyline[0][0]
    best_lng = polyline[0][1]

    for i in range(len(polyline) - 1):
        a_lat, a_lng = polyline[i][0], polyline[i][1]
        b_lat, b_lng = polyline[i + 1][0], polyline[i + 1][1]
        dist, n_lat, n_lng = point_to_segment_distance(p_lat, p_lng, a_lat, a_lng, b_lat, b_lng)
        if dist < min_dist:
            min_dist = dist
            best_lat = n_lat
            best_lng = n_lng

    return min_dist, best_lat, best_lng


class SpatialMatcher:
    """
    Spatially indexes road segments and maps arbitrary coordinate points
    to the nearest valid road segment based on road type thresholds.
    """

    def __init__(
        self,
        road_segments: List[Dict[str, Any]],
        urban_threshold_m: float = 150.0,
        highway_threshold_m: float = 250.0
    ):
        self.urban_threshold_m = urban_threshold_m
        self.highway_threshold_m = highway_threshold_m
        self.segments = []

        # Precompute bounding boxes and geometry for each segment
        for seg in road_segments:
            geom = seg.get("geometry", [])
            if not geom or len(geom) < 2:
                continue
            lats = [pt[0] for pt in geom]
            lngs = [pt[1] for pt in geom]
            road_type = str(seg.get("road_type", "primary")).lower()
            is_highway = any(h in road_type for h in ["trunk", "motorway", "highway", "expressway", "primary"])
            threshold = self.highway_threshold_m if is_highway else self.urban_threshold_m

            self.segments.append({
                "segment_id": seg["segment_id"],
                "road_name": seg.get("road_name", ""),
                "road_type": road_type,
                "threshold": threshold,
                "geometry": geom,
                "min_lat": min(lats) - 0.005,  # approx 500m bbox padding
                "max_lat": max(lats) + 0.005,
                "min_lng": min(lngs) - 0.005,
                "max_lng": max(lngs) + 0.005
            })

    def match_point(self, lat: float, lng: float) -> Tuple[Optional[str], float, Optional[Dict[str, Any]]]:
        """
        Find closest road segment within its specific distance threshold.
        Returns: (segment_id or None, match_distance_m, matched_segment_info or None)
        """
        best_segment_id = None
        best_distance = float("inf")
        best_seg_info = None

        for seg in self.segments:
            # Quick bounding box rejection
            if not (seg["min_lat"] <= lat <= seg["max_lat"] and seg["min_lng"] <= lng <= seg["max_lng"]):
                continue

            dist, _, _ = point_to_polyline_distance(lat, lng, seg["geometry"])
            if dist < best_distance:
                best_distance = dist
                best_seg_info = seg

        # Verify acceptance against specific road threshold
        if best_seg_info and best_distance <= best_seg_info["threshold"]:
            return best_seg_info["segment_id"], round(best_distance, 2), best_seg_info

        return None, round(best_distance, 2) if best_distance != float("inf") else 99999.0, None

    def match_anomalies(
        self, anomalies: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Batch match a list of anomalies.
        Attaches 'road_segment_id' and 'match_distance_m' to each anomaly.
        Returns matched anomalies and statistics summary.
        """
        total = len(anomalies)
        matched_count = 0
        unmatched_count = 0
        distances = []

        for anom in anomalies:
            lat = anom.get("latitude")
            lng = anom.get("longitude")
            if lat is None or lng is None:
                unmatched_count += 1
                anom["road_segment_id"] = None
                anom["match_distance_m"] = None
                continue

            seg_id, dist_m, _ = self.match_point(lat, lng)
            anom["match_distance_m"] = dist_m
            if seg_id is not None:
                anom["road_segment_id"] = seg_id
                matched_count += 1
                distances.append(dist_m)
            else:
                anom["road_segment_id"] = None
                unmatched_count += 1

        stats = {
            "total_coordinate_records": total,
            "matched_count": matched_count,
            "unmatched_count": unmatched_count,
            "mean_match_distance_m": round(float(np.mean(distances)), 2) if distances else 0.0,
            "median_match_distance_m": round(float(np.median(distances)), 2) if distances else 0.0
        }
        return anomalies, stats
