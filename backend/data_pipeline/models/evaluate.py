"""
backend/data_pipeline/models/evaluate.py
========================================
Evaluation metrics calculation for tabular road segment risk models.
Produces:
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion Matrix
- Baseline Comparison
- Feature Importances
"""

from typing import Dict, Any, List, Optional
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from backend.data_pipeline.storage.database import get_connection, DEFAULT_DB_PATH
from datetime import datetime, timezone


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "RandomForestClassifier",
    split_name: str = "TEST"
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model_name": model_name,
        "split": split_name,
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A (Single class)",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A (Single class)",
        "confusion_matrix": cm
    }


def save_evaluation_metrics(
    metrics: Dict[str, Any],
    feature_importances: Optional[Dict[str, float]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Save metrics to model_evaluation_metrics table."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_evaluation_metrics (
                model_name, target_variable, split_type,
                accuracy, precision_score, recall_score,
                f1_score, roc_auc, pr_auc, confusion_matrix,
                feature_importances, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics["model_name"],
            "future_segment_anomaly",
            metrics["split"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"] if isinstance(metrics["roc_auc"], (int, float)) else None,
            metrics["pr_auc"] if isinstance(metrics["pr_auc"], (int, float)) else None,
            json.dumps(metrics["confusion_matrix"]),
            json.dumps(feature_importances or {}),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()


def print_evaluation_report(metrics: Dict[str, Any], feature_importances: Optional[Dict[str, float]] = None) -> None:
    """Print structured metrics report."""
    print("--------------------------------------------------")
    print(f"MODEL EVALUATION REPORT: {metrics['model_name']} [{metrics['split']}]")
    print("--------------------------------------------------")
    print(f"  Samples:              {metrics['sample_count']} (Pos: {metrics['positive_count']}, Neg: {metrics['negative_count']})")
    print(f"  Accuracy:             {metrics['accuracy']}")
    print(f"  Precision:            {metrics['precision']}")
    print(f"  Recall:               {metrics['recall']}")
    print(f"  F1-Score:             {metrics['f1']}")
    print(f"  ROC-AUC:              {metrics['roc_auc']}")
    print(f"  PR-AUC:               {metrics['pr_auc']}")
    print(f"  Confusion Matrix:     {metrics['confusion_matrix']}")
    if feature_importances:
        print("  Top Features:")
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_feats[:6]:
            print(f"    - {name:22s}: {imp:.4f}")
    print("--------------------------------------------------")
