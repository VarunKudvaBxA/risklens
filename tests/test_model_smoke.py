"""Trains a tiny XGBoost model end-to-end and checks the scorer + SHAP reason codes."""
import joblib
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")

from xgboost import XGBClassifier  # noqa: E402

from risklens.calibration import PlattCalibrator  # noqa: E402
from risklens.config import TARGET  # noqa: E402
from risklens.data import make_synthetic  # noqa: E402
from risklens.features import FEATURES, engineer  # noqa: E402
from risklens.scoring import Scorer  # noqa: E402


def test_scorer_returns_reason_codes(tmp_path):
    df = make_synthetic(6000)
    X, y = engineer(df.drop(columns=[TARGET])), df[TARGET]
    model = XGBClassifier(n_estimators=60, max_depth=3, random_state=0).fit(X, y)
    cal = PlattCalibrator().fit(model.predict_proba(X)[:, 1], y)
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "calibrator": cal, "features": FEATURES, "threshold": 0.2}, path)
    risky = {"utilization": 1.4, "age": 25, "late_30_59": 3, "debt_ratio": 1.5, "monthly_income": 1500,
             "open_credit_lines": 4, "late_90": 3, "real_estate_loans": 0, "late_60_89": 2, "dependents": 3}
    safe = {**risky, "utilization": 0.05, "age": 55, "late_30_59": 0, "late_90": 0, "late_60_89": 0,
            "debt_ratio": 0.1, "monthly_income": 9000}
    out = Scorer(path).score([risky, safe])
    assert out[0]["probability_of_default"] > out[1]["probability_of_default"]
    assert out[0]["decision"] == "DECLINE" and len(out[0]["reason_codes"]) >= 1
    assert out[0]["credit_score"] < out[1]["credit_score"]
