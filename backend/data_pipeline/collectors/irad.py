"""
backend/data_pipeline/collectors/irad.py
========================================
Adapter for SOURCE 3: Tamil Nadu iRAD/eDAR (Integrated Road Accident Database).
URL: https://gisnic.tn.nic.in/irad_new/
Interface specification for official NIC iRAD integration.
VERIFIED: Server returns HTTP 503 (NIC intranet / access restricted to police & RTO).
CRITICAL: accessible is marked False. Zero records fabricated.
"""

import requests
import urllib3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from backend.data_pipeline.storage.database import save_data_source, DEFAULT_DB_PATH

urllib3.disable_warnings()

IRAD_URL = "https://gisnic.tn.nic.in/irad_new/"


class IRADAdapter:
    """
    Standard ingestion interface for official Tamil Nadu iRAD / eDAR system.
    Ready for authenticated integration once NIC API credentials or intranet tunnel are supplied.
    """

    EXPECTED_FIELDS = [
        "accident_id", "latitude", "longitude", "date_time",
        "accident_severity", "vehicles_involved", "road_conditions",
        "collision_type", "landmark", "police_station"
    ]

    def __init__(self, base_url: str = IRAD_URL, api_token: Optional[str] = None):
        self.base_url = base_url
        self.api_token = api_token

    def probe_connectivity(self, timeout_sec: int = 5) -> Dict[str, Any]:
        """Probe the official iRAD server to verify availability."""
        status_detail = "UNKNOWN"
        accessible = False

        try:
            r = requests.get(
                self.base_url,
                headers={"User-Agent": "Mozilla/5.0"},
                verify=False,
                timeout=timeout_sec
            )
            if r.status_code == 200:
                status_detail = "ACTIVE"
                accessible = True
            elif r.status_code == 503:
                status_detail = "UNAVAILABLE (HTTP 503 Service Unavailable / NIC Intranet Restricted)"
                accessible = False
            else:
                status_detail = f"RESTRICTED (HTTP {r.status_code})"
                accessible = False
        except requests.exceptions.Timeout:
            status_detail = f"TIMEOUT (No response within {timeout_sec}s)"
            accessible = False
        except Exception as e:
            status_detail = f"CONNECTION_FAILED ({type(e).__name__}: {str(e)[:60]})"
            accessible = False

        return {
            "status": status_detail,
            "accessible": accessible
        }

    def collect(self, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
        """
        Attempt collection and document status in data_sources.
        Does not fabricate any records if inaccessible.
        """
        probe = self.probe_connectivity()
        source_record = {
            "id": "irad_edar_tn",
            "name": "Tamil Nadu iRAD / eDAR (Integrated Road Accident Database)",
            "url": self.base_url,
            "collection_time": datetime.now(timezone.utc).isoformat(),
            "geographic_coverage": "State of Tamil Nadu (Police Station & Landmark Level)",
            "time_coverage": "2021-Present",
            "fields_available": self.EXPECTED_FIELDS,
            "is_historical": True,
            "is_live": True,
            "is_synthetic": False,
            "accessible": probe["accessible"],
            "status": probe["status"],
            "record_count": 0,
            "limitations": f"Access status: {probe['status']}. Public access restricted. Adapter interface ready for authenticated feed.",
            "licensing_notes": "Ministry of Road Transport and Highways (MoRTH) / NIC Government of Tamil Nadu"
        }
        save_data_source(source_record, db_path=db_path)
        return source_record
