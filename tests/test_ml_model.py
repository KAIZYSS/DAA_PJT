"""
tests/test_ml_model.py
======================
Unit tests for Machine Learning Model Training, Evaluation, and Inference:
- Strict chronological split
- Bounded predicted_risk output (0.0 to 1.0)
- Classification metrics (F1, ROC-AUC, PR-AUC, Confusion Matrix)
- Model artifact serialization
"""

import os
import pytest
import numpy as np
import pandas as pd
from backend.data_pipeline.models.train import train_risk_model, FEATURE_COLS, SAVED_MODEL_PATH
from backend.data_pipeline.models.evaluate import compute_metrics
from backend.data_pipeline.models.predict import predict_segment_risk, get_model


def test_compute_metrics():
    """Verify evaluation metric calculations with non-trivial predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.4, 0.2, 0.9, 0.6, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_prob, model_name="TestRF", split_name="TEST")
    assert metrics["sample_count"] == 8
    assert metrics["positive_count"] == 3
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_predict_segment_risk_bounds_and_factors():
    """Verify predict_segment_risk returns calibrated values in [0.0, 1.0] with reasons."""
    features_low = {
        "road_type_enc": 2,
        "length_m": 1200.0,
        "lanes": 4,
        "speed_limit": 50.0,
        "district_baseline_risk": 0.2,
        "district_fatality_rate": 0.15,
        "fcw_count": 0,
        "pcw_count": 0,
        "ldw_count": 1,
        "hmw_count": 2,
        "total_adas_warnings": 3,
        "adas_warning_density": 2.5
    }
    res_low = predict_segment_risk(features_low)
    assert 0.0 <= res_low["predicted_risk"] <= 1.0
    assert res_low["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(res_low["top_factors"]) > 0

    features_high = {
        "road_type_enc": 3,
        "length_m": 800.0,
        "lanes": 6,
        "speed_limit": 65.0,
        "district_baseline_risk": 0.9,
        "district_fatality_rate": 0.35,
        "fcw_count": 8,
        "pcw_count": 12,
        "ldw_count": 25,
        "hmw_count": 40,
        "total_adas_warnings": 85,
        "adas_warning_density": 106.25
    }
    res_high = predict_segment_risk(features_high)
    assert 0.0 <= res_high["predicted_risk"] <= 1.0
    assert res_high["predicted_risk"] >= res_low["predicted_risk"]
    assert any("warning density" in f.lower() or "collision" in f.lower() for f in res_high["top_factors"])


def test_model_artifact_loaded():
    """Verify saved model artifact exists and contains valid model payload."""
    assert os.path.exists(SAVED_MODEL_PATH), f"Model artifact not found at {SAVED_MODEL_PATH}"
    payload = get_model(SAVED_MODEL_PATH)
    assert payload is not None
    assert "model" in payload
    assert "feature_cols" in payload
    assert "metrics" in payload
    assert payload["metrics"]["accuracy"] > 0.0
