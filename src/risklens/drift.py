"""Data-drift monitoring: PSI and KS test, implemented from scratch (transparent, no black box).

PSI rule of thumb used across banking:  < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `actual` relative to `expected` (training) distribution."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant / binary feature
        edges = np.unique(np.r_[expected.min() - 1e-9, np.unique(expected), expected.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "STABLE" if value < 0.10 else "MODERATE" if value < 0.25 else "MAJOR"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    features = features or list(reference.columns)
    rows = []
    for f in features:
        v = psi(reference[f], current[f])
        ks = ks_2samp(reference[f].dropna(), current[f].dropna())
        rows.append({"feature": f, "psi": round(v, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": float(ks.pvalue), "status": status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, severity: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Economic-downturn scenario: incomes fall, utilization and delinquencies rise."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["monthly_income"] = out["monthly_income"] * (1 - 0.25 * severity)
    out["utilization"] = np.clip(out["utilization"] + 0.15 * severity, 0, 3)
    bump = rng.random(len(out)) < 0.08 * severity
    out.loc[bump, "late_30_59"] = out.loc[bump, "late_30_59"].fillna(0) + 1
    return out


def evidently_html_report(reference: pd.DataFrame, current: pd.DataFrame, path: str) -> bool:
    """Optional: HTML report with Evidently (API shown is for evidently 0.4.x). Returns False if unavailable."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(path)
        return True
    except Exception as exc:  # ImportError or API change between versions
        print(f"[drift] Evidently report skipped ({exc.__class__.__name__}). PSI/KS table is still valid.")
        return False
