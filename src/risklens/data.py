"""Data loading.

Real data: Kaggle 'Give Me Some Credit' (cs-training.csv). Download instructions are in the README.
Fallback: a realistic synthetic generator with the SAME schema so the repo runs out-of-the-box and CI works.
>>> Never quote synthetic-data metrics as real results. <<<
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import SEED, TARGET

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}


def load_real(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    return df.rename(columns=RENAME)


def make_synthetic(n: int = 30_000, seed: int = SEED, default_rate: float = 0.07) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 15, n), 21, 95).round()
    income = rng.lognormal(8.7, 0.6, n).round()
    util = np.clip(rng.beta(0.8, 2.5, n), 0, 1.5)
    debt_ratio = np.clip(rng.beta(1.2, 3, n), 0, 3)
    open_lines = rng.poisson(8, n)
    re_loans = rng.poisson(1.0, n)
    dependents = rng.poisson(0.8, n).astype(float)
    stress = rng.normal(size=n)  # hidden "financial stress" factor linking delinquencies and default
    late30 = rng.poisson(np.exp(-2.2 + 0.5 * stress + 0.8 * util))
    late60 = rng.poisson(np.exp(-3.0 + 0.5 * stress + 0.8 * util))
    late90 = rng.poisson(np.exp(-3.2 + 0.6 * stress + 0.9 * util))

    z = (2.6 * util + 0.9 * late30 + 1.2 * late60 + 1.5 * late90 - 0.05 * (age - 50)
         + 0.6 * debt_ratio - 0.35 * np.log(income / 6000) + 0.5 * stress)
    lo, hi = -15.0, 5.0  # bisection: pick intercept so the overall default rate matches the target
    for _ in range(50):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() < default_rate:
            lo = mid
        else:
            hi = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    y = rng.binomial(1, p)

    df = pd.DataFrame({
        TARGET: y, "utilization": util, "age": age, "late_30_59": late30, "debt_ratio": debt_ratio,
        "monthly_income": income, "open_credit_lines": open_lines, "late_90": late90,
        "real_estate_loans": re_loans, "late_60_89": late60, "dependents": dependents,
    })
    df.loc[rng.random(n) < 0.20, "monthly_income"] = np.nan   # real data has ~20% missing income
    df.loc[rng.random(n) < 0.026, "dependents"] = np.nan
    return df


def load_data(path: Optional[str | Path] = None, n_synth: int = 30_000) -> pd.DataFrame:
    if path and Path(path).exists():
        print(f"[data] loading real dataset: {path}")
        return load_real(path)
    print(f"[data] no dataset found -> generating {n_synth} synthetic rows (DEMO ONLY)")
    return make_synthetic(n_synth)
