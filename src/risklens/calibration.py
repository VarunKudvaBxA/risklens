"""Probability calibration. Class-weighting / scale_pos_weight improve ranking but distort probabilities
(a model may output 0.45 for a borrower whose true default risk is 8%). Platt scaling fixes this using a
held-out set, so 'probability of default' and the credit score are meaningful numbers."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, raw_probs, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_probs), y)
        return self

    def predict(self, raw_probs):
        return self.lr.predict_proba(_logit(raw_probs))[:, 1]
