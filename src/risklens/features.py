"""Feature engineering. ONE function is used in training, API and UI -> no train/serve skew."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = ["utilization", "age", "late_30_59", "debt_ratio", "monthly_income", "open_credit_lines",
                "late_90", "real_estate_loans", "late_60_89", "dependents"]

ENGINEERED = ["total_late", "late_severity", "any_serious_late", "income_missing", "log_income",
              "debt_amount", "income_per_dependent", "high_utilization", "lines_per_year_of_age"]

FEATURES = RAW_FEATURES + ENGINEERED

# Human-readable text for adverse-action "reason codes"
REASON_TEXT = {
    "utilization": "High use of available revolving credit ({value:.0%})",
    "late_30_59": "Payments 30-59 days late ({value:.0f} times)",
    "late_60_89": "Payments 60-89 days late ({value:.0f} times)",
    "late_90": "Payments 90+ days late ({value:.0f} times)",
    "total_late": "Number of past-due events ({value:.0f})",
    "late_severity": "Severity of past delinquencies (score {value:.0f})",
    "any_serious_late": "History of serious (60+ day) delinquency",
    "debt_ratio": "High debt-to-income ratio ({value:.2f})",
    "debt_amount": "High absolute debt level ({value:,.0f})",
    "monthly_income": "Low monthly income ({value:,.0f})",
    "log_income": "Low monthly income",
    "income_per_dependent": "Low income per household member ({value:,.0f})",
    "income_missing": "Income not provided / unverifiable",
    "age": "Age / short credit history ({value:.0f})",
    "open_credit_lines": "Number of open credit lines ({value:.0f})",
    "real_estate_loans": "Number of real-estate loans ({value:.0f})",
    "dependents": "Number of dependents ({value:.0f})",
    "high_utilization": "Credit utilization above 80%",
    "lines_per_year_of_age": "Credit-line density relative to age",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality problems in the public dataset."""
    out = df.copy()
    for col in ("late_30_59", "late_60_89", "late_90"):
        out[col] = out[col].where(out[col] < 90)        # 96/98 are sentinel codes, not real counts
    out["utilization"] = out["utilization"].clip(0, 3)  # some rows have utilization in the thousands
    out["debt_ratio"] = out["debt_ratio"].clip(0, 5)
    out["age"] = out["age"].where(out["age"] >= 18)     # age 0 is invalid
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = clean_raw(df[RAW_FEATURES])
    late_sum = d[["late_30_59", "late_60_89", "late_90"]].sum(axis=1, min_count=1)
    d["total_late"] = late_sum
    d["late_severity"] = d["late_30_59"] + 2 * d["late_60_89"] + 3 * d["late_90"]
    d["any_serious_late"] = ((d["late_60_89"] > 0) | (d["late_90"] > 0)).astype(int)
    d["income_missing"] = d["monthly_income"].isna().astype(int)
    d["log_income"] = np.log1p(d["monthly_income"])
    d["debt_amount"] = d["debt_ratio"] * d["monthly_income"]
    d["income_per_dependent"] = d["monthly_income"] / (d["dependents"].fillna(0) + 1)
    d["high_utilization"] = (d["utilization"] > 0.8).astype(int)
    d["lines_per_year_of_age"] = d["open_credit_lines"] / d["age"]
    d = d.replace([np.inf, -np.inf], np.nan)
    return d[FEATURES]
