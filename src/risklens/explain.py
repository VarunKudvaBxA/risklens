"""Explainability: SHAP values -> per-applicant 'reason codes' (the way lenders issue adverse-action notices)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import REASON_TEXT


class Explainer:
    def __init__(self, model, features: list[str]):
        self.features = features
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Contribution of each feature to the log-odds of default (positive = pushes toward default)."""
        values = self.explainer.shap_values(X[self.features])
        if isinstance(values, list):  # older shap versions return one array per class
            values = values[1]
        return np.asarray(values)

    def reason_codes(self, X_row: pd.DataFrame, top_k: int = 3) -> list[dict]:
        sv = self.shap_values(X_row)[0]
        order = np.argsort(-sv)  # most risk-increasing first
        codes = []
        for i in order[:top_k]:
            if sv[i] <= 0:
                break
            feat = self.features[i]
            value = X_row.iloc[0][feat]
            template = REASON_TEXT.get(feat, feat.replace("_", " ").capitalize())
            text = template.format(value=value) if "{" in template and pd.notna(value) else template.split(" (")[0]
            codes.append({"feature": feat, "impact": round(float(sv[i]), 3), "reason": text})
        return codes

    def explanation(self, X_row: pd.DataFrame) -> shap.Explanation:
        sv = self.shap_values(X_row)[0]
        base = self.explainer.expected_value
        base = float(np.asarray(base).ravel()[-1])
        return shap.Explanation(values=sv, base_values=base, data=X_row.iloc[0].to_numpy(dtype=float),
                                feature_names=self.features)
