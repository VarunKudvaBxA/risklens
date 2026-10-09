"""Train, compare, tune, calibrate, threshold, audit, and save.

    python -m risklens.train                      # synthetic demo data
    python -m risklens.train --data data/cs-training.csv --n-trials 40   # real Kaggle data
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import fairness, reports
from .calibration import PlattCalibrator
from .config import (COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, MODEL_DIR, MODEL_PATH, REPORT_DIR, SEED, TARGET)
from .data import load_data
from .explain import Explainer
from .features import FEATURES, engineer
from .metrics import evaluate
from .thresholds import cost_at, optimal_threshold

XGB_BASE = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)


def _mlflow_run(enabled: bool, name: str):
    if not enabled:
        return nullcontext()
    import mlflow
    return mlflow.start_run(run_name=name, nested=True)


def _log(enabled: bool, params: dict | None, metrics: dict):
    if enabled:
        import mlflow
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def tune_xgb(X, y, spw: float, n_trials: int) -> dict:
    """Optuna search maximising 3-fold CV PR-AUC on the TRAIN set only (validation/test stay untouched)."""
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, spw),
        )
        model = XGBClassifier(**{**XGB_BASE, **params})
        return float(cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"[optuna] best CV PR-AUC={study.best_value:.4f}  params={study.best_params}")
    return study.best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to Kaggle cs-training.csv (else synthetic demo data)")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--n-synth", type=int, default=30_000)
    args = ap.parse_args()
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("risklens")

    # ---------- data & split (60/20/20, stratified: default rate stays equal in every split) ----------
    df = load_data(args.data, args.n_synth)
    y = df[TARGET].astype(int)
    X = engineer(df.drop(columns=[TARGET]))
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[data] rows={len(df)}  default rate={y.mean():.2%}  scale_pos_weight={spw:.1f}")

    # ---------- model zoo: compare ways of dealing with class imbalance ----------
    prep = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    candidates = {
        "logreg_balanced": Pipeline(prep + [("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]),
        "logreg_smote": ImbPipeline(prep + [("smote", SMOTE(random_state=SEED)),
                                            ("clf", LogisticRegression(max_iter=2000))]),
        "xgb_unweighted": XGBClassifier(**XGB_BASE),
        "xgb_scale_pos_weight": XGBClassifier(**XGB_BASE, scale_pos_weight=spw),
    }

    parent = mlflow.start_run(run_name="training") if use_mlflow else nullcontext()
    with parent:
        print(f"[optuna] tuning XGBoost ({args.n_trials} trials)...")
        best = tune_xgb(X_train, y_train, spw, args.n_trials)
        candidates["xgb_tuned"] = XGBClassifier(**{**XGB_BASE, **best})

        rows, val_preds = [], {}
        for name, model in candidates.items():
            with _mlflow_run(use_mlflow, name):
                model.fit(X_train, y_train)
                p = model.predict_proba(X_val)[:, 1]
                m = evaluate(y_val, p)
                rows.append({"model": name, **m})
                val_preds[name] = p
                _log(use_mlflow, {"model": name}, {f"val_{k}": v for k, v in m.items()})
        comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
        REPORT_DIR.mkdir(exist_ok=True, parents=True)
        comparison.round(4).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
        print("\n=== Validation comparison ===\n", comparison.round(4).to_string(index=False))

        # ---------- deploy the best TREE model (SHAP needs trees; logistic regression stays as benchmark) ----------
        best_name = comparison[comparison["model"].str.startswith("xgb")].iloc[0]["model"]
        model = candidates[best_name]
        print(f"\n[select] deploying: {best_name}")

        # ---------- calibration + business-cost threshold (both fitted on VALIDATION only) ----------
        calibrator = PlattCalibrator().fit(val_preds[best_name], y_val)
        p_val = calibrator.predict(val_preds[best_name])
        threshold, curve = optimal_threshold(y_val, p_val)

        # ---------- final, untouched TEST evaluation ----------
        p_test = calibrator.predict(model.predict_proba(X_test)[:, 1])
        test_metrics = evaluate(y_test, p_test)
        pred = (p_test >= threshold).astype(int)
        cost_model = cost_at(y_test, p_test, threshold)
        cost_approve_all = float(y_test.sum() * COST_FALSE_NEGATIVE)
        cost_default_05 = cost_at(y_test, p_test, 0.5)
        business = {"cost_optimal_threshold": cost_model, "cost_approve_everyone": cost_approve_all,
                    "cost_threshold_0.5": cost_default_05,
                    "saving_vs_approve_all_pct": 100 * (1 - cost_model / cost_approve_all),
                    "saving_vs_threshold_0.5_pct": 100 * (1 - cost_model / max(cost_default_05, 1))}
        print("\n=== TEST metrics ===", json.dumps({k: round(v, 4) for k, v in test_metrics.items()}, indent=1))
        print(f"threshold={threshold:.2f}  (cost FN={COST_FALSE_NEGATIVE}, FP={COST_FALSE_POSITIVE})")
        print("business:", {k: round(v, 1) for k, v in business.items()})

        # ---------- fairness audit ----------
        audit = fairness.audit(y_test, pred, fairness.age_band(X_test["age"]))
        audit["table"].to_csv(REPORT_DIR / "fairness_by_age.csv")
        print("\n=== Fairness (by age band) ===\n", audit["table"], "\n", audit["summary"])

        # ---------- plots ----------
        reports.plot_curves(y_val, val_preds)
        reports.plot_calibration(y_test, p_test)
        reports.plot_cost_curve(curve, threshold)
        explainer = Explainer(model, FEATURES)
        reports.plot_shap_summary(explainer, X_test.sample(min(2000, len(X_test)), random_state=SEED))

        # ---------- persist ----------
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        joblib.dump({
            "model": model, "calibrator": calibrator, "features": FEATURES, "threshold": threshold,
            "model_name": best_name, "test_metrics": test_metrics, "business": business,
            "reference": X_train.sample(min(5000, len(X_train)), random_state=SEED).reset_index(drop=True),
        }, MODEL_PATH)
        (REPORT_DIR / "test_metrics.json").write_text(json.dumps(
            {"model": best_name, "threshold": threshold, "test": test_metrics, "business": business,
             "fairness": audit["summary"]}, indent=2))
        if use_mlflow:
            import mlflow
            mlflow.log_params({"deployed_model": best_name, "threshold": threshold})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_artifacts(str(REPORT_DIR))
        print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
