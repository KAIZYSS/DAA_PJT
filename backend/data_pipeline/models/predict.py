"""
backend/data_pipeline/models/predict.py
======================================
Inference pipeline for road segment risk prediction.
Loads trained model artifact and outputs calibrated predicted_risk (0.0 to 1.0).
Provides interpretability breakdown of key contributing risk factors.
"""

import os
import joblib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from backend.data_pipeline.models.train import SAVED_MODEL_PATH, FEATURE_COLS

_LOADED_MODEL_PAYLOAD = None


def get_model(model_path: str = SAVED_MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Cached loader for the trained model artifact."""
    global _LOADED_MODEL_PAYLOAD
    if _LOADED_MODEL_PAYLOAD is None and os.path.exists(model_path):
        _LOADED_MODEL_PAYLOAD = joblib.load(model_path)
    return _LOADED_MODEL_PAYLOAD


def predict_segment_risk(
    features: Dict[str, Any],
    model_path: str = SAVED_MODEL_PATH
) -> Dict[str, Any]:
    """
    Predict road segment risk given feature vector.
    Outputs:
        predicted_risk: float (0.0 to 1.0)
        risk_tier: str (LOW, MEDIUM, HIGH, CRITICAL)
        top_factors: List[str]
    """
    payload = get_model(model_path)
    if payload is None or "model" not in payload:
        # Fallback to district baseline or heuristic prior if model not yet trained
        dist_risk = float(features.get("district_baseline_risk", 0.35))
        density = float(features.get("adas_warning_density", 0.0))
        fallback_risk = round(min(1.0, dist_risk * 0.6 + min(density / 10.0, 1.0) * 0.4), 3)
        return {
            "predicted_risk": fallback_risk,
            "risk_tier": "MEDIUM" if fallback_risk > 0.4 else "LOW",
            "is_model_inference": False,
            "top_factors": ["District Baseline Prior (Model artifact not loaded)"]
        }

    model = payload["model"]
    feature_cols = payload.get("feature_cols", FEATURE_COLS)

    # Format input DataFrame with column names
    row_dict = {c: float(features.get(c, 0.0)) for c in feature_cols}
    X = pd.DataFrame([row_dict], columns=feature_cols)

    # Probability of class 1 (future anomaly)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] > 1:
            raw_prob = float(probs[0, 1])
        else:
            raw_prob = float(probs[0, 0])
    else:
        raw_prob = float(model.predict(X)[0])

    predicted_risk = round(max(0.0, min(1.0, raw_prob)), 3)

    if predicted_risk >= 0.70:
        tier = "CRITICAL"
    elif predicted_risk >= 0.45:
        tier = "HIGH"
    elif predicted_risk >= 0.25:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Identify primary contributing factors
    contributing_factors = []
    if features.get("adas_warning_density", 0) > 3.0:
        contributing_factors.append(f"Elevated ADAS warning density ({features['adas_warning_density']:.1f}/km)")
    if features.get("fcw_count", 0) > 0:
        contributing_factors.append(f"Recent Forward Collision Warnings ({features['fcw_count']})")
    if features.get("pcw_count", 0) > 0:
        contributing_factors.append(f"Pedestrian Collision Warnings recorded ({features['pcw_count']})")
    if features.get("district_baseline_risk", 0) > 0.5:
        contributing_factors.append(f"High District Baseline Risk ({features['district_baseline_risk']:.2f})")
    if not contributing_factors:
        contributing_factors.append("Nominal road corridor baseline")

    return {
        "predicted_risk": predicted_risk,
        "risk_tier": tier,
        "is_model_inference": True,
        "top_factors": contributing_factors
    }
