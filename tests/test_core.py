import numpy as np
import pandas as pd

from risklens.config import TARGET
from risklens.data import make_synthetic
from risklens.drift import drift_report, psi, simulate_drift
from risklens.features import FEATURES, RAW_FEATURES, engineer
from risklens.metrics import evaluate, ks_statistic, recall_at_precision
from risklens.scoring import prob_to_score
from risklens.thresholds import cost_at, optimal_threshold


def test_synthetic_schema_and_rate():
    df = make_synthetic(5000)
    assert set(RAW_FEATURES + [TARGET]) <= set(df.columns)
    assert 0.04 < df[TARGET].mean() < 0.10


def test_engineer_is_deterministic_and_has_all_features():
    df = make_synthetic(500).drop(columns=[TARGET])
    a, b = engineer(df), engineer(df)
    assert list(a.columns) == FEATURES
    pd.testing.assert_frame_equal(a, b)
    assert not np.isinf(a.to_numpy(dtype=float)).any()


def test_sentinel_values_are_cleaned():
    df = make_synthetic(10).drop(columns=[TARGET])
    df.loc[0, "late_90"] = 98
    assert np.isnan(engineer(df).loc[0, "late_90"])


def test_metrics_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks_statistic(y, np.array([.1, .2, .3, .7, .8, .9])) == 1.0
    assert recall_at_precision(y, np.array([.1, .2, .3, .7, .8, .9]), 0.9) == 1.0
    assert evaluate(y, np.array([.1, .2, .3, .7, .8, .9]))["roc_auc"] == 1.0


def test_threshold_moves_with_costs():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p * 0.3).astype(int)
    t_cheap_fn, _ = optimal_threshold(y, p, cost_fn=1, cost_fp=1)
    t_costly_fn, _ = optimal_threshold(y, p, cost_fn=20, cost_fp=1)
    assert t_costly_fn < t_cheap_fn  # when missing a defaulter is expensive, decline more readily
    assert cost_at(y, p, t_costly_fn, 20, 1) <= cost_at(y, p, 0.5, 20, 1)


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20000)
    assert psi(base, rng.normal(0, 1, 20000)) < 0.05
    assert psi(base, rng.normal(1.0, 1, 20000)) > 0.25


def test_drift_report_flags_simulated_downturn():
    X = engineer(make_synthetic(8000).drop(columns=[TARGET]))
    rep = drift_report(X, engineer(simulate_drift(X, 1.5)), FEATURES)
    flagged = set(rep[rep["status"] != "STABLE"]["feature"])
    assert {"utilization", "monthly_income"} <= flagged


def test_score_scaling_monotonic():
    assert prob_to_score(0.01) > prob_to_score(0.1) > prob_to_score(0.5)
    assert prob_to_score(1 / 51) == 600 or abs(prob_to_score(1 / 51) - 600) <= 1
