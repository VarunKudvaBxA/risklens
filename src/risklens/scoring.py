"""Inference layer used by the API and the Streamlit app."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import BASE_ODDS, BASE_SCORE, MODEL_PATH, PDO
from .features import RAW_FEATURES, engineer


def prob_to_score(p: float) -> int:
    """Classic scorecard scaling: score = offset + factor * ln(good:bad odds)."""
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return int(np.clip(round(offset + factor * np.log((1 - p) / p)), 300, 900))


class Scorer:
    def __init__(self, path: str | Path = MODEL_PATH, with_explanations: bool = True):
        art = joblib.load(path)
        self.model = art["model"]
        self.features = art["features"]
        self.threshold = art["threshold"]
        self.calibrator = art.get("calibrator")
        self.metrics = art.get("test_metrics", {})
        self.model_name = art.get("model_name", "xgboost")
        self.explainer = None
        if with_explanations:
            from .explain import Explainer
            self.explainer = Explainer(self.model, self.features)

    def to_frame(self, records: list[dict]) -> pd.DataFrame:
        raw = pd.DataFrame(records).reindex(columns=RAW_FEATURES).astype(float)
        return engineer(raw)

    def score(self, records: list[dict], top_k: int = 3) -> list[dict]:
        X = self.to_frame(records)
        probs = self.model.predict_proba(X[self.features])[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
        out = []
        for i, p in enumerate(probs):
            decision = "DECLINE" if p >= self.threshold else "APPROVE"
            item = {"probability_of_default": round(float(p), 4), "credit_score": prob_to_score(p),
                    "decision": decision, "threshold": round(self.threshold, 3)}
            if self.explainer is not None:
                item["reason_codes"] = self.explainer.reason_codes(X.iloc[[i]], top_k) if decision == "DECLINE" \
                    else []
                item["top_risk_drivers"] = self.explainer.reason_codes(X.iloc[[i]], top_k)
            out.append(item)
        return out
