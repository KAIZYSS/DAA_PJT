"""
backend/data_pipeline/processors/geocoder.py
============================================
Spatial and bounding box validation for Tamil Nadu and Chennai coordinates.
"""

from typing import Tuple, Optional, Any

# Bounding box coordinates
TN_BOUNDS = {
    "min_lat": 8.0,
    "max_lat": 13.6,
    "min_lng": 76.2,
    "max_lng": 80.5
}

CHENNAI_METRO_BOUNDS = {
    "min_lat": 12.6,
    "max_lat": 13.4,
    "min_lng": 79.8,
    "max_lng": 80.4
}


def is_valid_coordinate(lat: Any, lng: Any) -> bool:
    """Check if lat/lng are valid non-null floats."""
    try:
        f_lat = float(lat)
        f_lng = float(lng)
        return -90.0 <= f_lat <= 90.0 and -180.0 <= f_lng <= 180.0
    except (ValueError, TypeError):
        return False


def is_in_tamil_nadu(lat: float, lng: float) -> bool:
    """Verify coordinate falls within the geographic envelope of Tamil Nadu."""
    if not is_valid_coordinate(lat, lng):
        return False
    return (
        TN_BOUNDS["min_lat"] <= float(lat) <= TN_BOUNDS["max_lat"] and
        TN_BOUNDS["min_lng"] <= float(lng) <= TN_BOUNDS["max_lng"]
    )


def is_in_chennai_region(lat: float, lng: float) -> bool:
    """Verify coordinate falls within Greater Chennai / CMA bounds."""
    if not is_valid_coordinate(lat, lng):
        return False
    return (
        CHENNAI_METRO_BOUNDS["min_lat"] <= float(lat) <= CHENNAI_METRO_BOUNDS["max_lat"] and
        CHENNAI_METRO_BOUNDS["min_lng"] <= float(lng) <= CHENNAI_METRO_BOUNDS["max_lng"]
    )
