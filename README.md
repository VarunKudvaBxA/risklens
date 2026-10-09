# RiskLens — Explainable Credit Risk Scoring with Fairness Audit & Drift Monitoring

Predicts the probability that a borrower will become seriously delinquent, turns it into a **credit score**, picks the
approve/decline cut-off from **business cost**, explains every decision with **SHAP reason codes**, audits it for **fairness**,
and monitors it for **data drift** — packaged as an API, a Streamlit app, a Docker image and a CI pipeline.

```
 data ─► feature engineering ─► model zoo (LogReg, SMOTE, XGBoost, Optuna-tuned) ─► pick best by PR-AUC
                                      │
              Platt calibration ◄─────┘──► cost-based threshold ─► test-set metrics + business impact
                    │                              │
              SHAP reason codes             Fairlearn audit (age bands)
                    │
        FastAPI /predict  ·  Streamlit underwriting UI  ·  PSI/KS drift monitor  ·  MLflow tracking
```

## Why this project
Real lenders don't just need an accurate model. They need to handle **heavily imbalanced data**, **explain** declines to customers/regulators,
choose a threshold from **money** not from 0.5, check **fairness**, and notice when the **population changes**. This repo demonstrates each.

## Data
- **Real (recommended):** Kaggle *Give Me Some Credit* → download `cs-training.csv` from https://www.kaggle.com/c/GiveMeSomeCredit/data into `data/`.
- **Demo:** if no file is given, a synthetic dataset with the same schema is generated so the repo runs anywhere (and in CI).
  **Never report synthetic-data metrics as real results.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m risklens.train --data data/cs-training.csv --n-trials 40    # omit --data for demo data
mlflow ui                                                              # compare runs at localhost:5000
streamlit run app/streamlit_app.py                                     # underwriting UI with SHAP waterfall
uvicorn app.api:app --reload                                           # API docs at /docs
python scripts/simulate_drift.py --severity 1.0                        # downturn drift scenario
pytest -q
```
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
 "utilization":0.9,"age":34,"late_30_59":1,"debt_ratio":0.6,"monthly_income":3200,
 "open_credit_lines":6,"late_90":1,"real_estate_loans":0,"late_60_89":0,"dependents":2}'
```

## What's inside
| Module | What it does |
|---|---|
| `data.py` | Loads Kaggle data or generates schema-identical synthetic data |
| `features.py` | Cleaning (sentinel codes 96/98, outliers) + 9 engineered features; single function reused in training and serving |
| `train.py` | 5-model comparison, Optuna tuning (CV on train only), calibration, cost-based threshold, test evaluation, MLflow logging |
| `metrics.py` | ROC-AUC, PR-AUC, KS, Gini, recall@precision, Brier |
| `thresholds.py` | Total business cost vs threshold; picks the minimum |
| `calibration.py` | Platt scaling so probabilities are real probabilities |
| `explain.py` | SHAP → human-readable reason codes |
| `fairness.py` | Fairlearn audit (decline rate, TPR, FPR by group; demographic-parity and equalized-odds gaps) |
| `drift.py` | PSI + KS per feature, downturn simulator, optional Evidently HTML report |
| `scoring.py` | Inference + scorecard scaling (600 pts = 50:1 odds, 20 pts to double) |

## Results
> Fill in after training on the **real** dataset. Everything below is produced automatically in `reports/`.

| Model (validation) | ROC-AUC | PR-AUC | KS | Gini |
|---|---|---|---|---|
| _see `reports/model_comparison.csv`_ | | | | |

Test set (deployed model): ROC-AUC `__`, PR-AUC `__`, KS `__`, recall@30% precision `__`
Business impact: estimated cost vs "approve everyone" `__%` lower; vs fixed 0.5 threshold `__%` lower (`reports/test_metrics.json`).
Plots: `reports/roc_pr_curves.png`, `calibration.png`, `cost_curve.png`, `shap_summary.png`.
For orientation: tuned gradient-boosting models typically reach roughly 0.85-0.87 ROC-AUC on this public dataset — verify your own numbers.

## Design decisions
1. **PR-AUC/KS/recall@precision, not accuracy** — with ~7% defaulters, "approve everybody" is 93% accurate and useless.
2. **Imbalance strategies are compared, not assumed** (class weights vs SMOTE vs none).
3. **Tune on CV of the train set; choose the threshold on validation; report once on an untouched test set.**
4. **Calibration** because class weighting distorts probabilities; PD and credit score must be meaningful.
5. **Cost-based threshold** (FN=10,000 vs FP=1,000 are editable assumptions in `config.py`).
6. **Logistic regression kept as the scorecard benchmark; trees deployed** for accuracy, explained with SHAP.
7. **PSI/KS implemented from scratch** (transparent); Evidently report optional.

## Limitations & future work
- Random split, not time-based; real systems validate out-of-time.
- Age is used in the demo fairness audit; in real lending, which attributes may be used is regulated — consult compliance.
- Cost numbers are illustrative; reject inference (applicants never seen) is ignored.
- Fraud detection (transaction-level, IEEE-CIS) is a natural extension and is not part of this repo.
- No automated retraining; drift triggers a *review*, not a deployment.
