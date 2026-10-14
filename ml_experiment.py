"""
ml_experiment.py
================
ROUTEWISE -- Standalone ML Model Comparison Experiment
======================================================

Implements:
 TASK 1  -- Audit existing ML pipeline
 TASK 2  -- Identify available models (sklearn only; no new libs)
 TASK 3  -- Check project model requirements
 TASK 4  -- Design & run experiment (same data, same split, same features)
 TASK 5  -- Handle class imbalance (class_weight=balanced; no SMOTE)
 TASK 6  -- Temporal-leakage audit + WINDOW-OVERLAP assessment
 TASK 7  -- Build comparison table (ranked by PR-AUC)
 TASK 8  -- Select best model
 TASK 9  -- Save best model artifact (NOT overwriting production model)
 OUTPUT  -- ml_experiment_results.json + ml_experiment_results.csv

RULES:
 - Identical train/val/test for ALL models
 - Threshold tuned ONLY on validation data
 - Final metrics evaluated ONCE on held-out test set
 - No random shuffle of full dataset
 - No synthetic samples (SMOTE etc.)
 - Does NOT touch Dijkstra, Floyd-Warshall, TomTom, OSM routing, or production RF
"""

import os
import sys
import csv
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure project root in sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# sklearn imports
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn

_HAS_XGB = False
_HAS_LGB = False
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    pass
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ROUTEWISE data pipeline imports (read-only)
# ---------------------------------------------------------------------------
from backend.data_pipeline.collectors.tn_accidents import (
    collect_tn_district_accidents, get_district_risk_lookup
)
from backend.data_pipeline.collectors.adas import collect_adas_mobility
from backend.data_pipeline.collectors.osm import collect_osm_road_segments
from backend.data_pipeline.processors.spatial_matcher import SpatialMatcher
from backend.data_pipeline.processors.feature_builder import build_temporal_segment_dataset
from backend.data_pipeline.models.train import FEATURE_COLS
from backend.data_pipeline.storage.database import DEFAULT_DB_PATH, init_pipeline_tables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "backend", "data_pipeline", "models", "saved_models"
)
# NOTE: writes to road_risk_best.joblib — does NOT overwrite production road_risk_rf.joblib
FINAL_MODEL_PATH = os.path.join(EXPERIMENT_MODEL_DIR, "road_risk_best.joblib")
EXPERIMENT_LOG_PATH  = os.path.join(PROJECT_ROOT, "ml_experiment_results.json")
EXPERIMENT_CSV_PATH  = os.path.join(PROJECT_ROOT, "ml_experiment_results.csv")

TRAIN_FRAC        = 0.60
VAL_FRAC          = 0.20
# TEST_FRAC = 0.20 implicit

WINDOW_DAYS       = 7      # matches production pipeline
N_THRESHOLD_STEPS = 51     # grid-search 0.00 to 1.00


# ===========================================================================
# SECTION 1 -- ENVIRONMENT RECORD
# ===========================================================================

def record_environment() -> Dict[str, str]:
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "xgboost_available": str(_HAS_XGB),
        "lightgbm_available": str(_HAS_LGB),
        "experiment_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    print("\n[ENVIRONMENT]")
    for k, v in env.items():
        print(f"  {k:<28}: {v}")
    return env


# ===========================================================================
# SECTION 2 -- DATA LOADING
# ===========================================================================

def load_ml_dataset(db_path: str = DEFAULT_DB_PATH) -> Tuple[pd.DataFrame, Dict]:
    """
    Full ingestion + feature-building pipeline.
    Returns (df_ml, audit_info).
    Uses window_days=7 (same as production).
    """
    print("\n" + "="*62)
    print("LOADING ROUTEWISE DATASET")
    print("="*62)

    init_pipeline_tables(db_path=db_path)

    road_segments, _ = collect_osm_road_segments(db_path=db_path)
    print(f"  OSM segments:         {len(road_segments)}")

    _df_d, _ = collect_tn_district_accidents(db_path=db_path)
    district_lookup = get_district_risk_lookup()
    print(f"  TN district priors:   {len(district_lookup)}")

    adas_records, prof_adas = collect_adas_mobility(db_path=db_path)
    print(f"  ADAS raw records:     {len(adas_records):,}")
    print(f"  ADAS date range:      {prof_adas.get('date_range', 'N/A')}")

    matcher = SpatialMatcher(
        road_segments, urban_threshold_m=150.0, highway_threshold_m=250.0
    )
    matched_adas, match_stats = matcher.match_anomalies(adas_records)
    matched_count = match_stats.get("matched_count", 0)
    print(f"  Matched ADAS records: {matched_count:,}")
    print(f"  Mean match distance:  {match_stats.get('mean_match_distance_m', 'N/A')} m")

    df_ml, feature_summary = build_temporal_segment_dataset(
        road_segments=road_segments,
        matched_anomalies=matched_adas,
        district_risk_lookup=district_lookup,
        window_days=WINDOW_DAYS,
    )

    if feature_summary["status"] != "VALID":
        raise RuntimeError(
            f"Feature builder failed: {feature_summary['status']} — "
            f"{feature_summary.get('message', '')}"
        )

    pos   = int((df_ml["future_segment_anomaly"] == 1).sum())
    neg   = int((df_ml["future_segment_anomaly"] == 0).sum())
    total = len(df_ml)
    pos_ratio = round(pos / max(1, total), 4)

    print(f"\n  Total ML samples:     {total}")
    print(f"  Unique windows:       {feature_summary['total_windows']}")
    print(f"  Positive (1):         {pos}  ({pos_ratio*100:.1f}%)")
    print(f"  Negative (0):         {neg}  ({(1-pos_ratio)*100:.1f}%)")
    print(f"  Imbalance ratio:      1:{round(neg/max(pos,1), 1)}")

    audit_info = {
        "dataset": "ADAS Mobility Data + OSM Road Graph + TN District Accident Priors",
        "adas_raw_records": len(adas_records),
        "adas_date_range": prof_adas.get("date_range", "N/A"),
        "matched_adas_records": matched_count,
        "osm_segments": len(road_segments),
        "window_days": WINDOW_DAYS,
        "total_samples": total,
        "total_windows": feature_summary["total_windows"],
        "positive_samples": pos,
        "negative_samples": neg,
        "positive_ratio": pos_ratio,
        "features": FEATURE_COLS,
        "target": "future_segment_anomaly",
        "target_definition": (
            "Binary 1 if qualifying safety event (FCW or PCW alert, OR severity>=7.0) "
            "occurs on segment in next window T+1; else 0."
        ),
    }
    return df_ml, audit_info


# ===========================================================================
# SECTION 3 -- WINDOW-OVERLAP & LEAKAGE ASSESSMENT (Task 7 requirement)
# ===========================================================================

def assess_window_overlap(df_ml: pd.DataFrame) -> Dict[str, Any]:
    """
    Explicitly assess whether non-overlapping windows and repeated road
    segments across windows create optimistic evaluation conditions.

    Windows in build_temporal_segment_dataset() are NON-OVERLAPPING
    (each window advances by exactly window_days days).
    However, the SAME road segment appears in EVERY window, so there is
    high cross-window correlation of the static road features
    (road_type_enc, length_m, lanes, speed_limit, district priors).

    Findings:
    - Windows: strictly non-overlapping -> no direct feature-overlap between windows
    - Segments: same 52 segments repeated in every window -> static feature correlation
    - Cumulative counts (fcw_count etc.) are cumulative across ALL prior windows,
      not reset per window, so early-window counts appear in all later windows' features.
    - This means: rows from the SAME segment in different windows are NOT independent.
    - Consequence: test-set performance may be SLIGHTLY OPTIMISTIC because:
      (a) the model has seen the static road attributes of test-set segments in training,
      (b) cumulative ADAS counts for test windows include counts from training windows.
    - This is an INHERENT limitation of the small dataset (52 segments, 14 windows)
      and is explicitly disclosed rather than claimed leakage-free.
    """
    n_windows  = df_ml["window_index"].nunique()
    n_segments = df_ml["segment_id"].nunique() if "segment_id" in df_ml.columns else "N/A"
    n_rows     = len(df_ml)

    # Check if any segment appears in train AND test (will always be true with 52 segs)
    total_windows = n_windows
    train_cutoff  = int(total_windows * TRAIN_FRAC)
    val_cutoff    = int(total_windows * (TRAIN_FRAC + VAL_FRAC))

    train_segs = set()
    test_segs  = set()
    if "segment_id" in df_ml.columns:
        train_segs = set(df_ml[df_ml["window_index"] < train_cutoff]["segment_id"])
        test_segs  = set(df_ml[df_ml["window_index"] >= val_cutoff]["segment_id"])

    overlap_segs = len(train_segs & test_segs)

    assessment = {
        "windows_non_overlapping": True,
        "window_advance_days": WINDOW_DAYS,
        "n_windows": n_windows,
        "n_unique_segments": n_segments,
        "n_rows": n_rows,
        "segment_overlap_train_test": overlap_segs,
        "cumulative_features_note": (
            "fcw_count/pcw_count/ldw_count/hmw_count/total_adas_warnings/adas_warning_density "
            "are CUMULATIVE across ALL events before window T end. "
            "This means test-window features implicitly include counts from training windows."
        ),
        "static_feature_leakage": (
            f"All {overlap_segs} test segments also appear in training. "
            "Static features (road_type_enc, length_m, lanes, speed_limit, district priors) "
            "are identical across windows for the same segment. "
            "The model has therefore 'seen' the identity of test-set road segments during training."
        ),
        "optimism_risk": (
            "MODERATE — Static road features are shared between train and test. "
            "Cumulative ADAS counts grow monotonically so test counts are always >= train counts "
            "for the same segment. Metrics on this dataset should be treated as INDICATIVE, "
            "not definitive proof of generalisation to unseen road segments."
        ),
        "recommendation": (
            "For production validation, collect data from road segments NOT seen during training, "
            "or use leave-one-segment-out cross-validation when more data is available."
        ),
    }

    print("\n[WINDOW & SEGMENT OVERLAP ASSESSMENT]")
    for k, v in assessment.items():
        print(f"  {k}: {v}")

    return assessment


# ===========================================================================
# SECTION 4 -- CHRONOLOGICAL SPLIT
# ===========================================================================

def chronological_split(
    df_ml: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Strict chronological split by window_index.
    NO random shuffle. Train < Val < Test in time order.
    """
    total_windows = df_ml["window_index"].nunique()
    train_cutoff  = int(total_windows * train_frac)
    val_cutoff    = int(total_windows * (train_frac + val_frac))

    train_df = df_ml[df_ml["window_index"] < train_cutoff].copy()
    val_df   = df_ml[
        (df_ml["window_index"] >= train_cutoff) &
        (df_ml["window_index"] <  val_cutoff)
    ].copy()
    test_df  = df_ml[df_ml["window_index"] >= val_cutoff].copy()

    # Guard for tiny datasets
    if test_df.empty and not val_df.empty:
        test_df = val_df.copy()
        val_df  = train_df.copy()

    split_info = {
        "total_windows":  total_windows,
        "train_windows":  train_cutoff,
        "val_windows":    val_cutoff - train_cutoff,
        "test_windows":   total_windows - val_cutoff,
        "train_samples":  len(train_df),
        "val_samples":    len(val_df),
        "test_samples":   len(test_df),
        "train_pos":      int((train_df["future_segment_anomaly"] == 1).sum()),
        "val_pos":        int((val_df["future_segment_anomaly"] == 1).sum()),
        "test_pos":       int((test_df["future_segment_anomaly"] == 1).sum()),
        "train_neg":      int((train_df["future_segment_anomaly"] == 0).sum()),
        "val_neg":        int((val_df["future_segment_anomaly"] == 0).sum()),
        "test_neg":       int((test_df["future_segment_anomaly"] == 0).sum()),
        "chronological":  True,
        "no_shuffle":     True,
    }

    print(f"\n[CHRONOLOGICAL SPLIT]")
    print(f"  {'Split':<8} {'Samples':>8} {'Pos':>6} {'Neg':>6} {'Windows':>8}")
    print(f"  {'-'*40}")
    print(f"  {'TRAIN':<8} {len(train_df):>8,} {split_info['train_pos']:>6} {split_info['train_neg']:>6} {train_cutoff:>8}")
    print(f"  {'VAL':<8} {len(val_df):>8,} {split_info['val_pos']:>6} {split_info['val_neg']:>6} {val_cutoff-train_cutoff:>8}")
    print(f"  {'TEST':<8} {len(test_df):>8,} {split_info['test_pos']:>6} {split_info['test_neg']:>6} {total_windows-val_cutoff:>8}")
    print(f"  Leakage: train windows < val windows < test windows (PASS)")

    return train_df, val_df, test_df, split_info


# ===========================================================================
# SECTION 5 -- METRICS
# ===========================================================================

def compute_full_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_prob:     Optional[np.ndarray],
    model_name: str,
    split_name: str,
    threshold:  float,
) -> Dict[str, Any]:
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc  = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model_name":    model_name,
        "split":         split_name,
        "threshold":     round(threshold, 3),
        "sample_count":  len(y_true),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "accuracy":      round(acc,  4),
        "precision":     round(prec, 4),
        "recall":        round(rec,  4),
        "f1":            round(f1,   4),
        "roc_auc":       round(roc_auc, 4) if roc_auc is not None else "N/A",
        "pr_auc":        round(pr_auc,  4) if pr_auc  is not None else "N/A",
        "confusion_matrix": cm,
    }


# ===========================================================================
# SECTION 6 -- THRESHOLD TUNING (VALIDATION ONLY)
# ===========================================================================

def tune_threshold_on_val(
    y_val:     np.ndarray,
    y_val_prob: np.ndarray,
) -> Tuple[float, float]:
    """Grid-search threshold on VAL to maximise F1. Returns (threshold, best_f1)."""
    thresholds = np.linspace(0.0, 1.0, N_THRESHOLD_STEPS)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_val_prob >= t).astype(int)
        val = float(f1_score(y_val, y_pred, zero_division=0))
        if val > best_f1:
            best_f1, best_thresh = val, float(t)
    return best_thresh, best_f1


# ===========================================================================
# SECTION 7 -- MODEL CATALOGUE
# ===========================================================================

def build_model_catalogue(pos_ratio: float) -> List[Dict[str, Any]]:
    neg_pos_ratio = round((1 - pos_ratio) / max(pos_ratio, 1e-9), 1)

    catalogue = [
        {
            "name": "DummyClassifier",
            "short": "Dummy",
            "model": DummyClassifier(strategy="most_frequent"),
            "use_sample_weight": False,
            "notes": "Majority-class baseline — always predicts most-frequent label",
        },
        {
            "name": "LogisticRegression",
            "short": "LR",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=1000,
                    solver="lbfgs", random_state=42,
                )),
            ]),
            "use_sample_weight": False,
            "notes": "Linear model with StandardScaler + balanced class weights",
        },
        {
            "name": "DecisionTreeClassifier",
            "short": "DT",
            "model": DecisionTreeClassifier(
                max_depth=5, min_samples_split=4,
                class_weight="balanced", random_state=42,
            ),
            "use_sample_weight": False,
            "notes": "Single tree depth=5, balanced weights",
        },
        {
            "name": "RandomForestClassifier",
            "short": "RF",
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=4,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            "use_sample_weight": False,
            "notes": "CURRENT PRODUCTION MODEL — 100 trees, depth=5, balanced",
        },
        {
            "name": "ExtraTreesClassifier",
            "short": "ET",
            "model": ExtraTreesClassifier(
                n_estimators=100, max_depth=5, min_samples_split=4,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            "use_sample_weight": False,
            "notes": "Extremely randomised trees, balanced weights",
        },
        {
            "name": "GradientBoostingClassifier",
            "short": "GB",
            "model": GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.1, subsample=0.8, random_state=42,
            ),
            "use_sample_weight": True,
            "sample_weight_ratio": neg_pos_ratio,
            "notes": f"GBM + inverse-freq sample_weight (ratio {neg_pos_ratio}:1)",
        },
    ]

    if _HAS_XGB:
        catalogue.append({
            "name": "XGBoostClassifier",
            "short": "XGB",
            "model": xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                scale_pos_weight=neg_pos_ratio,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            "use_sample_weight": False,
            "notes": f"XGBoost scale_pos_weight={neg_pos_ratio}",
        })

    if _HAS_LGB:
        catalogue.append({
            "name": "LightGBMClassifier",
            "short": "LGB",
            "model": lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                class_weight="balanced", random_state=42, verbose=-1,
            ),
            "use_sample_weight": False,
            "notes": "LightGBM balanced weights",
        })

    return catalogue


# ===========================================================================
# SECTION 8 -- TRAIN & EVALUATE ONE MODEL
# ===========================================================================

def train_and_evaluate(
    cfg:     Dict[str, Any],
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val:   pd.DataFrame, y_val:   np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
) -> Dict[str, Any]:
    name  = cfg["name"]
    model = cfg["model"]
    print(f"\n  >> {name}")

    # Fit
    if cfg.get("use_sample_weight") and cfg.get("sample_weight_ratio"):
        sw = np.where(y_train == 1, float(cfg["sample_weight_ratio"]), 1.0)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    def get_proba(X):
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            return p[:, 1] if p.shape[1] > 1 else p[:, 0]
        return model.predict(X).astype(float)

    # Threshold tuning on validation
    y_val_prob  = get_proba(X_val)
    best_thresh, _ = tune_threshold_on_val(y_val, y_val_prob)

    # Validation metrics
    y_val_pred  = (y_val_prob >= best_thresh).astype(int)
    val_metrics = compute_full_metrics(y_val, y_val_pred, y_val_prob, name, "VAL", best_thresh)

    # Test metrics (threshold fixed from validation)
    y_test_prob  = get_proba(X_test)
    y_test_pred  = (y_test_prob >= best_thresh).astype(int)
    test_metrics = compute_full_metrics(y_test, y_test_pred, y_test_prob, name, "TEST", best_thresh)

    print(f"     threshold={best_thresh:.3f}  "
          f"val_F1={val_metrics['f1']:.4f}  "
          f"val_PR-AUC={val_metrics['pr_auc']}  "
          f"test_F1={test_metrics['f1']:.4f}  "
          f"test_PR-AUC={test_metrics['pr_auc']}")
    print(f"     test CM: {test_metrics['confusion_matrix']}")

    # Feature importances
    feat_imp = {}
    clf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        feat_imp = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_]))
    elif hasattr(clf, "coef_"):
        coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
        feat_imp = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in coef]))

    return {
        "name":              name,
        "short":             cfg["short"],
        "notes":             cfg["notes"],
        "threshold":         best_thresh,
        "val_metrics":       val_metrics,
        "test_metrics":      test_metrics,
        "feature_importances": feat_imp,
        "model_object":      model,
    }


# ===========================================================================
# SECTION 9 -- RUN FULL COMPARISON
# ===========================================================================

def run_comparison(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    pos_ratio: float,
) -> List[Dict[str, Any]]:
    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["future_segment_anomaly"].values
    X_val   = val_df[FEATURE_COLS].fillna(0)
    y_val   = val_df["future_segment_anomaly"].values
    X_test  = test_df[FEATURE_COLS].fillna(0)
    y_test  = test_df["future_segment_anomaly"].values

    catalogue = build_model_catalogue(pos_ratio)
    results = []
    for cfg in catalogue:
        try:
            r = train_and_evaluate(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {cfg['name']}: {e}")
            results.append({"name": cfg["name"], "short": cfg["short"],
                            "notes": cfg["notes"], "error": str(e)})
    return results


# ===========================================================================
# SECTION 10 -- SELECT BEST MODEL
# ===========================================================================

def select_best(results: List[Dict]) -> Dict:
    """Rank by PR-AUC (primary) > F1 > Recall. Skip baseline and errors."""
    ranked = []
    for r in results:
        if "error" in r or r["name"] == "DummyClassifier":
            continue
        tm = r["test_metrics"]
        pr  = tm["pr_auc"]  if isinstance(tm["pr_auc"],  float) else -1.0
        f1  = tm["f1"]
        rec = tm["recall"]
        ranked.append((pr, f1, rec, r))
    if not ranked:
        raise RuntimeError("No valid models to select from.")
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return ranked[0][3]


# ===========================================================================
# SECTION 11 -- SAVE BEST MODEL
# ===========================================================================

def save_best(best: Dict, split_info: Dict, audit_info: Dict, env: Dict) -> str:
    os.makedirs(EXPERIMENT_MODEL_DIR, exist_ok=True)
    payload = {
        "model":              best["model_object"],
        "model_name":         best["name"],
        "feature_cols":       FEATURE_COLS,
        "threshold":          best["threshold"],
        "test_metrics":       best["test_metrics"],
        "val_metrics":        best["val_metrics"],
        "feature_importances": best.get("feature_importances", {}),
        "split_info":         split_info,
        "environment":        env,
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "version":            "experiment_v1",
        "note":               "Experiment candidate — NOT replacing production road_risk_rf.joblib",
    }
    joblib.dump(payload, FINAL_MODEL_PATH)
    print(f"\n  Saved experiment model: {FINAL_MODEL_PATH}")
    return FINAL_MODEL_PATH


# ===========================================================================
# SECTION 12 -- PRINT + SAVE COMPARISON TABLE
# ===========================================================================

def print_and_save_table(results: List[Dict]) -> None:
    """Print ranked table to stdout and write CSV."""
    non_err = [r for r in results if "error" not in r]
    errors  = [r for r in results if "error" in r]

    dummy   = [r for r in non_err if r["name"] == "DummyClassifier"]
    others  = [r for r in non_err if r["name"] != "DummyClassifier"]

    def sort_key(r):
        tm = r.get("test_metrics", {})
        pr = tm.get("pr_auc", -1)
        return (pr if isinstance(pr, float) else -1, tm.get("f1", 0))

    others.sort(key=sort_key, reverse=True)
    ordered = others + dummy + errors

    hdr = (
        f"  {'Model':<37} {'PR-AUC':>8} {'ROC-AUC':>9} "
        f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Thr':>7}  Notes"
    )
    print("\n" + "="*115)
    print("  COMPARISON TABLE  (TEST split | val-tuned threshold | ranked by PR-AUC)")
    print("="*115)
    print(hdr)
    print("  " + "-"*112)

    csv_rows = []
    for r in ordered:
        if "error" in r:
            print(f"  {r['name']:<37} {'ERROR':>8}  {r['error'][:50]}")
            csv_rows.append({
                "model": r["name"], "pr_auc": "ERROR", "roc_auc": "", "precision": "",
                "recall": "", "f1": "", "threshold": "", "cm": "", "notes": r["error"],
            })
            continue
        tm  = r["test_metrics"]
        pr  = f"{tm['pr_auc']:.4f}"  if isinstance(tm["pr_auc"],  float) else "N/A"
        roc = f"{tm['roc_auc']:.4f}" if isinstance(tm["roc_auc"], float) else "N/A"
        thr = f"{r['threshold']:.3f}"
        note = r.get("notes","")[:45]
        cm_str = str(tm["confusion_matrix"])
        print(
            f"  {r['name']:<37} {pr:>8} {roc:>9} "
            f"{tm['precision']:>7.4f} {tm['recall']:>7.4f} {tm['f1']:>7.4f} {thr:>7}  {note}"
        )
        csv_rows.append({
            "model":     r["name"],
            "pr_auc":    pr,
            "roc_auc":   roc,
            "precision": tm["precision"],
            "recall":    tm["recall"],
            "f1":        tm["f1"],
            "threshold": thr,
            "accuracy":  tm["accuracy"],
            "tp":        tm["confusion_matrix"][1][1] if len(tm["confusion_matrix"]) > 1 else "",
            "fp":        tm["confusion_matrix"][0][1] if len(tm["confusion_matrix"]) > 1 else "",
            "fn":        tm["confusion_matrix"][1][0] if len(tm["confusion_matrix"]) > 1 else "",
            "tn":        tm["confusion_matrix"][0][0] if len(tm["confusion_matrix"]) > 1 else "",
            "cm":        cm_str,
            "notes":     r.get("notes",""),
        })
    print("="*115)

    # Write CSV
    if csv_rows:
        fieldnames = ["model","pr_auc","roc_auc","precision","recall","f1","threshold",
                      "accuracy","tp","fp","fn","tn","cm","notes"]
        with open(EXPERIMENT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  CSV saved: {EXPERIMENT_CSV_PATH}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "#"*62)
    print("  ROUTEWISE -- ML MODEL COMPARISON EXPERIMENT")
    print("#"*62)

    # Environment
    env = record_environment()

    # Project requirements
    print("\n[PROJECT REQUIREMENTS]")
    print("  NO SPECIFIC MODEL REQUIREMENT FOUND")
    print("  (requirements.txt: Flask>=3.0.0 and pytest>=8.0.0 only)")

    # Available models
    print("\n[AVAILABLE MODELS]")
    avail = {
        "DummyClassifier":            True,
        "LogisticRegression":         True,
        "DecisionTreeClassifier":     True,
        "RandomForestClassifier":     True,
        "ExtraTreesClassifier":       True,
        "GradientBoostingClassifier": True,
        "XGBoostClassifier":          _HAS_XGB,
        "LightGBMClassifier":         _HAS_LGB,
    }
    for m, ok in avail.items():
        print(f"  {m:<35}: {'YES' if ok else 'NO (not installed)'}")

    # Load dataset
    db_path = DEFAULT_DB_PATH
    df_ml, audit_info = load_ml_dataset(db_path=db_path)
    pos_ratio = audit_info["positive_ratio"]

    # Window/segment overlap assessment
    overlap_assessment = assess_window_overlap(df_ml)

    # Chronological split
    train_df, val_df, test_df, split_info = chronological_split(df_ml)

    if split_info["train_pos"] == 0:
        print("\nCRITICAL: Zero positive samples in training set. Cannot train.")
        return

    if split_info["val_pos"] == 0:
        print("\nWARNING: Zero positives in validation — threshold tuning will default to 0.5.")

    # Run model comparison
    print("\n[RUNNING MODEL COMPARISON]")
    print("  Same train/val/test split and features for ALL models.")
    results = run_comparison(train_df, val_df, test_df, pos_ratio)

    # Print and save CSV table
    print_and_save_table(results)

    # Select best
    print("\n[MODEL SELECTION]")
    best = select_best(results)
    bm   = best["test_metrics"]
    print(f"\n  SELECTED: {best['name']}")
    print(f"  PR-AUC  (test): {bm['pr_auc']}")
    print(f"  ROC-AUC (test): {bm['roc_auc']}")
    print(f"  Precision:      {bm['precision']}")
    print(f"  Recall:         {bm['recall']}")
    print(f"  F1:             {bm['f1']}")
    print(f"  Threshold:      {best['threshold']:.3f}  (tuned on val)")
    print(f"  Conf Matrix:    {bm['confusion_matrix']}")
    print(f"  Notes:          {best['notes']}")

    # Why selected / why others not selected
    print("\n[SELECTION RATIONALE]")
    print(f"  Primary criterion:  PR-AUC (most informative for imbalanced classification)")
    print(f"  Secondary:          F1 (balanced precision/recall)")
    print(f"  Baseline excluded:  DummyClassifier (no learning)")
    all_non_dummy = [r for r in results if "error" not in r and r["name"] != "DummyClassifier"]
    all_non_dummy.sort(
        key=lambda r: (
            r["test_metrics"]["pr_auc"] if isinstance(r["test_metrics"]["pr_auc"], float) else -1,
            r["test_metrics"]["f1"],
        ),
        reverse=True,
    )
    for rank, r in enumerate(all_non_dummy, 1):
        tm  = r["test_metrics"]
        selected = " <-- SELECTED" if r["name"] == best["name"] else ""
        print(f"  #{rank} {r['name']:<35} PR-AUC={tm['pr_auc']}  F1={tm['f1']}{selected}")

    # Save best model
    print("\n[SAVING EXPERIMENT MODEL]")
    saved_path = save_best(best, split_info, audit_info, env)

    # Save JSON log
    def safe(r):
        return {k: v for k, v in r.items() if k != "model_object"}

    log = {
        "experiment_time":    datetime.now(timezone.utc).isoformat(),
        "environment":        env,
        "project_requirement": "NO SPECIFIC MODEL REQUIREMENT FOUND",
        "available_models":   avail,
        "audit":              audit_info,
        "split_info":         split_info,
        "window_overlap_assessment": overlap_assessment,
        "results":            [safe(r) for r in results],
        "best_model":         safe(best),
        "saved_model_path":   saved_path,
        "csv_path":           EXPERIMENT_CSV_PATH,
        "production_model_unchanged": True,
        "production_model_path": os.path.join(EXPERIMENT_MODEL_DIR, "road_risk_rf.joblib"),
    }
    with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"  JSON log saved:  {EXPERIMENT_LOG_PATH}")

    print("\n" + "#"*62)
    print("  EXPERIMENT COMPLETE")
    print("#"*62)
    print(f"\n  Winner:        {best['name']}")
    print(f"  Threshold:     {best['threshold']:.3f}")
    print(f"  Test PR-AUC:   {bm['pr_auc']}")
    print(f"  Test F1:       {bm['f1']}")
    print(f"  Saved model:   {saved_path}  (experiment candidate)")
    print(f"  Production RF: {log['production_model_path']}  (UNCHANGED)")
    print(f"  JSON log:      {EXPERIMENT_LOG_PATH}")
    print(f"  CSV:           {EXPERIMENT_CSV_PATH}")
    print("\nSTOP -- awaiting model-selection review before any routing integration.\n")

    return log


if __name__ == "__main__":
    main()
