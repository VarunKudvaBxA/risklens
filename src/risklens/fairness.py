"""Fairness audit with Fairlearn: do approval rates and error rates differ across groups?
Demo uses age bands because the public dataset has no other demographic fields.
NOTE: in real lending, regulations restrict which attributes may be used; this shows the *method*."""
from __future__ import annotations

import numpy as np
import pandas as pd


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 45, 60, 200], labels=["<=30", "31-45", "46-60", "60+"]).astype(str)


def audit(y_true, y_pred, sensitive) -> dict:
    """y_pred = 1 means DECLINED. Returns per-group table + summary gaps."""
    from fairlearn.metrics import (MetricFrame, demographic_parity_difference, equalized_odds_difference,
                                   false_positive_rate, selection_rate, true_positive_rate)
    mf = MetricFrame(
        metrics={"decline_rate": selection_rate, "recall(TPR)": true_positive_rate,
                 "false_positive_rate": false_positive_rate},
        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred), sensitive_features=np.asarray(sensitive),
    )
    table = mf.by_group.round(4)
    table["n"] = pd.Series(np.asarray(sensitive)).value_counts()
    summary = {
        "demographic_parity_difference": float(demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive)),
        "equalized_odds_difference": float(equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive)),
    }
    return {"table": table, "summary": summary}
