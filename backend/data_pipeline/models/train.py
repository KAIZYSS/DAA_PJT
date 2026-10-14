"""
backend/data_pipeline/models/train.py
=====================================
Model training pipeline with strict chronological splitting.
Evaluates Random Forest and Gradient Boosting against a baseline classifier.
Saves model artifact to saved_models/road_risk_rf.joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from backend.data_pipeline.models.evaluate import compute_metrics, save_evaluation_metrics, print_evaluation_report
from backend.data_pipeline.storage.database import DEFAULT_DB_PATH

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "saved_models")
)
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "road_risk_rf.joblib")

FEATURE_COLS = [
    "road_type_enc",
    "length_m",
    "lanes",
    "speed_limit",
    "district_baseline_risk",
    "district_fatality_rate",
    "fcw_count",
    "pcw_count",
    "ldw_count",
    "hmw_count",
    "total_adas_warnings",
    "adas_warning_density"
]


def train_risk_model(
    df_ml: pd.DataFrame,
    db_path: str = DEFAULT_DB_PATH,
    save_path: str = SAVED_MODEL_PATH
) -> Dict[str, Any]:
    """
    Train tabular road segment risk model using strict chronological splitting.
    """
    if df_ml.empty or "future_segment_anomaly" not in df_ml.columns:
        return {
            "status": "FAILED",
            "message": "Insufficient real temporal/location data for supervised prediction."
        }

    total_windows = df_ml["window_index"].nunique()
    if total_windows < 2:
        return {
            "status": "FAILED",
            "message": f"Insufficient time windows ({total_windows}). Minimum 2 chronological windows required."
        }

    # Strict chronological partition
    train_cutoff = int(total_windows * 0.60)
    val_cutoff = int(total_windows * 0.80)

    train_df = df_ml[df_ml["window_index"] < train_cutoff]
    val_df = df_ml[(df_ml["window_index"] >= train_cutoff) & (df_ml["window_index"] < val_cutoff)]
    test_df = df_ml[df_ml["window_index"] >= val_cutoff]

    # Handle small window edge cases
    if test_df.empty:
        test_df = val_df
        val_df = train_df

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["future_segment_anomaly"].values

    X_val = val_df[FEATURE_COLS].fillna(0)
    y_val = val_df["future_segment_anomaly"].values

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df["future_segment_anomaly"].values

    pos_total = int((df_ml["future_segment_anomaly"] == 1).sum())
    if pos_total == 0:
        return {
            "status": "FAILED",
            "message": "Zero positive target samples found. Insufficient real anomaly signals for supervised classification."
        }

    # 1. Baseline Classifier (Majority class / prior)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_base_pred = baseline.predict(X_test)
    base_metrics = compute_metrics(y_test, y_base_pred, model_name="BaselineClassifier", split_name="TEST")

    # 2. Random Forest Model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_rf_pred = rf.predict(X_test)
    y_rf_prob = rf.predict_proba(X_test)[:, 1] if len(rf.classes_) > 1 else None

    rf_metrics = compute_metrics(y_test, y_rf_pred, y_rf_prob, model_name="RandomForestClassifier", split_name="TEST")

    # Feature importances
    feature_importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in rf.feature_importances_]))

    # Save to database
    save_evaluation_metrics(rf_metrics, feature_importances=feature_importances, db_path=db_path)

    # 3. Save Model Artifact
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_payload = {
        "model": rf,
        "feature_cols": FEATURE_COLS,
        "feature_importances": feature_importances,
        "classes": list(rf.classes_),
        "metrics": rf_metrics,
        "baseline_accuracy": base_metrics["accuracy"],
        "trained_at": pd.Timestamp.now().isoformat()
    }
    joblib.dump(model_payload, save_path)

    return {
        "status": "SUCCESS",
        "model_path": save_path,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "positive_samples": pos_total,
        "negative_samples": int((df_ml["future_segment_anomaly"] == 0).sum()),
        "baseline_metrics": base_metrics,
        "rf_metrics": rf_metrics,
        "feature_importances": feature_importances
    }
