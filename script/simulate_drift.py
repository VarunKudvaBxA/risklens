"""Simulate an economic downturn and show which features drift (PSI / KS).
Usage: python scripts/simulate_drift.py --severity 1.0"""
import argparse

import joblib

from risklens.config import MODEL_PATH
from risklens.drift import drift_report, evidently_html_report, simulate_drift
from risklens.features import FEATURES, engineer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=1.0)
    args = ap.parse_args()
    ref = joblib.load(MODEL_PATH)["reference"]
    # shift the RAW columns, then re-derive every engineered feature from them (as production would)
    current = engineer(simulate_drift(ref, args.severity))
    print(drift_report(ref, current, FEATURES).head(10).to_string(index=False))
    print("\nPSI < 0.10 stable | 0.10-0.25 moderate | > 0.25 major  -> major drift triggers a retraining review")
    evidently_html_report(ref, current, "reports/evidently_drift.html")


if __name__ == "__main__":
    main()
