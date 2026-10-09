"""Interactive underwriting demo.   streamlit run app/streamlit_app.py"""
import matplotlib.pyplot as plt
import shap
import streamlit as st

from risklens.config import MODEL_PATH
from risklens.scoring import Scorer

st.set_page_config(page_title="RiskLens", layout="wide")
st.title("RiskLens: Explainable Credit Risk Scoring")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m risklens.train` first.")
    st.stop()


@st.cache_resource
def load():
    return Scorer(MODEL_PATH)


scorer = load()

with st.sidebar:
    st.header("Applicant")
    rec = {
        "age": st.slider("Age", 18, 90, 40),
        "monthly_income": st.number_input("Monthly income", 0, 100_000, 4500, step=100),
        "utilization": st.slider("Credit utilization (used / limit)", 0.0, 2.0, 0.4, 0.01),
        "debt_ratio": st.slider("Debt ratio (debt payments / income)", 0.0, 3.0, 0.35, 0.01),
        "open_credit_lines": st.slider("Open credit lines", 0, 30, 7),
        "real_estate_loans": st.slider("Real-estate loans", 0, 10, 1),
        "dependents": st.slider("Dependents", 0, 10, 1),
        "late_30_59": st.slider("Times 30-59 days late", 0, 10, 0),
        "late_60_89": st.slider("Times 60-89 days late", 0, 10, 0),
        "late_90": st.slider("Times 90+ days late", 0, 10, 0),
    }

res = scorer.score([rec])[0]
c1, c2, c3 = st.columns(3)
c1.metric("Credit score", res["credit_score"])
c2.metric("Probability of default", f"{res['probability_of_default']:.1%}")
c3.metric("Decision", res["decision"], help=f"Decline if PD >= {res['threshold']} (cost-optimised)")

st.subheader("Why? Top risk drivers (SHAP)")
for r in res["top_risk_drivers"] or [{"reason": "No factor increases risk above baseline", "impact": 0}]:
    st.write(f"- **{r['reason']}**  (impact {r['impact']:+.2f} log-odds)")
if res["decision"] == "DECLINE":
    st.info("Adverse-action reasons: " + "; ".join(r["reason"] for r in res["reason_codes"]))

X = scorer.to_frame([rec])
fig = plt.figure()
shap.plots.waterfall(scorer.explainer.explanation(X), show=False, max_display=10)
st.pyplot(plt.gcf(), clear_figure=True)
with st.expander("Model performance (held-out test set)"):
    st.json(scorer.metrics)
