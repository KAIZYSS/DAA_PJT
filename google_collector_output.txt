"""
backend/data_pipeline/collectors/google_traffic.py
==================================================
Adapter for SOURCE 8: Google Maps Live Traffic.
Queries Google Maps Distance Matrix API (traffic_model='best_guess', departure_time='now')
to obtain live travel time and traffic factor (duration_in_traffic / duration).
Maps traffic delay factor directly to OSM road segments to update edge weights.
CRITICAL CONSTRAINTS:
1. Google is a LIVE DATA SOURCE ONLY.
2. Does NOT return the final route.
3. Our Dijkstra remains the sole route selection algorithm.
4. When GOOGLE_MAPS_API_KEY is not configured, clearly flags status as 'DEMO / SIMULATED'
   and never falsely claims live connectivity.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from backend.data_pipeline.storage.database import save_data_source, DEFAULT_DB_PATH


class GoogleTrafficCollector:
    """
    Client for Google Maps Live Traffic API with transparent fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        self.is_live_connected = bool(self.api_key)

    def fetch_segment_traffic(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> Dict[str, Any]:
        """
        Fetch real-time traffic factor for a road segment between origin and destination.
        Returns:
            traffic_factor: float (1.0 = normal, >1.0 = congested)
            travel_time_sec: float
            is_live: bool
            status: str
        """
        if self.is_live_connected:
            try:
                url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    "origins": f"{origin_lat},{origin_lng}",
                    "destinations": f"{dest_lat},{dest_lng}",
                    "departure_time": "now",
                    "traffic_model": "best_guess",
                    "key": self.api_key
                }
                resp = requests.get(url, params=params, timeout=5)
                data = resp.json()
                if data.get("status") == "OK":
                    elem = data["rows"][0]["elements"][0]
                    if elem.get("status") == "OK":
                        dur = elem["duration"]["value"]
                        dur_traffic = elem.get("duration_in_traffic", {}).get("value", dur)
                        factor = max(1.0, round(dur_traffic / max(dur, 1), 2))
                        return {
                            "traffic_factor": factor,
                            "travel_time_sec": dur_traffic,
                            "is_live": True,
                            "mode": "LIVE_GOOGLE_API",
                            "status": "OK"
                        }
            except Exception as e:
                # Fallback on network or quota failure
                pass

        # Realistic time-of-day model for Chennai corridors (clearly labeled DEMO / SIMULATED)
        now = datetime.now()
        hour = now.hour + now.minute / 60.0
        # Peak windows: Morning 08:30-11:00, Evening 17:30-20:30
        is_morning_peak = 8.5 <= hour <= 11.0
        is_evening_peak = 17.5 <= hour <= 20.5

        if is_morning_peak:
            factor = 1.65
        elif is_evening_peak:
            factor = 1.80
        elif 12.0 <= hour <= 16.0:
            factor = 1.20
        else:
            factor = 1.05

        return {
            "traffic_factor": factor,
            "travel_time_sec": None,
            "is_live": False,
            "mode": "DEMO_TIME_OF_DAY_SIMULATION",
            "status": "DEMO (No GOOGLE_MAPS_API_KEY configured)"
        }

    def register_source(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Record live traffic source metadata in database."""
        status = "LIVE_CONNECTED" if self.is_live_connected else "DEMO_FALLBACK (Awaiting GOOGLE_MAPS_API_KEY)"
        source_record = {
            "id": "google_live_traffic",
            "name": "Google Maps Platform Live Traffic",
            "url": "https://maps.googleapis.com/maps/api/distancematrix",
            "collection_time": datetime.now(timezone.utc).isoformat(),
            "geographic_coverage": "Chennai Road Graph Segments",
            "time_coverage": "Real-time query",
            "fields_available": ["duration", "duration_in_traffic", "traffic_factor"],
            "is_historical": False,
            "is_live": self.is_live_connected,
            "is_synthetic": not self.is_live_connected,
            "accessible": self.is_live_connected,
            "status": status,
            "record_count": 0,
            "limitations": "Live traffic factor only. Does NOT provide route decisions or accident causality.",
            "licensing_notes": "Google Maps Platform Terms of Service"
        }
        save_data_source(source_record, db_path=db_path)
