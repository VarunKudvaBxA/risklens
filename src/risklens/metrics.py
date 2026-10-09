"""Credit-risk metrics. Accuracy is useless at a ~7% default rate -> use these instead."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)


def ks_statistic(y_true, p) -> float:
    """Max gap between the cumulative distributions of defaulters and non-defaulters (0..1)."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def gini(y_true, p) -> float:
    return float(2 * roc_auc_score(y_true, p) - 1)


def recall_at_precision(y_true, p, min_precision: float = 0.3) -> float:
    """Best recall achievable while keeping precision >= min_precision (a realistic operating point)."""
    precision, recall, _ = precision_recall_curve(y_true, p)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, p, min_precision: float = 0.3) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "ks": ks_statistic(y_true, p),
        "gini": gini(y_true, p),
        f"recall@precision{min_precision:.0%}": recall_at_precision(y_true, p, min_precision),
        "brier": float(brier_score_loss(y_true, p)),
    }
