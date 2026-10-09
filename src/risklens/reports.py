"""Plots for the README: ROC/PR, calibration, cost curve, SHAP summary."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import REPORT_DIR


def _save(fig, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / name, dpi=140)
    plt.close(fig)


def plot_curves(y, preds: dict):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=name)
        ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=7)
    _save(fig, "roc_pr_curves.png")


def plot_calibration(y, p, name="final model"):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(mean, frac, "o-", label=name)
    ax.plot([0, max(mean.max(), frac.max())], [0, max(mean.max(), frac.max())], "k--", lw=0.8, label="perfect")
    ax.set(title="Calibration", xlabel="Predicted probability", ylabel="Observed default rate")
    ax.legend()
    _save(fig, "calibration.png")


def plot_cost_curve(curve, best_t):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(curve["threshold"], curve["cost"])
    ax.axvline(best_t, color="r", ls="--", label=f"optimal = {best_t:.2f}")
    ax.set(title="Business cost vs decision threshold", xlabel="Threshold", ylabel="Total cost")
    ax.legend()
    _save(fig, "cost_curve.png")


def plot_shap_summary(explainer, X_sample):
    import shap
    values = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(values, X_sample[explainer.features], show=False, max_display=12)
    fig = plt.gcf()
    _save(fig, "shap_summary.png")
