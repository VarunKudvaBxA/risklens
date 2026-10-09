"""Pick the decision threshold from BUSINESS COST, not from the default 0.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def cost_curve(y_true, p, cost_fn: float = COST_FALSE_NEGATIVE, cost_fp: float = COST_FALSE_POSITIVE,
               grid: np.ndarray | None = None) -> pd.DataFrame:
    y = np.asarray(y_true)
    p = np.asarray(p)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    rows = []
    for t in grid:
        pred = p >= t                       # predicted "will default" -> decline
        fn = int(((~pred) & (y == 1)).sum())  # approved a defaulter
        fp = int((pred & (y == 0)).sum())     # declined a good customer
        rows.append({"threshold": float(t), "fn": fn, "fp": fp, "cost": fn * cost_fn + fp * cost_fp})
    return pd.DataFrame(rows)


def optimal_threshold(y_true, p, cost_fn: float = COST_FALSE_NEGATIVE,
                      cost_fp: float = COST_FALSE_POSITIVE) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(y_true, p, cost_fn, cost_fp)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"]), curve


def cost_at(y_true, p, threshold: float, cost_fn: float = COST_FALSE_NEGATIVE,
            cost_fp: float = COST_FALSE_POSITIVE) -> float:
    y, pred = np.asarray(y_true), np.asarray(p) >= threshold
    return float(((~pred) & (y == 1)).sum() * cost_fn + (pred & (y == 0)).sum() * cost_fp)
