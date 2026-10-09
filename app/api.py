"""FastAPI scoring service.   uvicorn app.api:app --reload   ->  http://localhost:8000/docs"""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from risklens.config import MODEL_PATH
from risklens.scoring import Scorer

app = FastAPI(title="RiskLens", version="0.1.0", description="Explainable credit-risk scoring")
_scorer: Optional[Scorer] = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m risklens.train")
        _scorer = Scorer(MODEL_PATH)
    return _scorer


class Applicant(BaseModel):
    utilization: float = Field(..., ge=0, le=5, description="Revolving credit used / credit limit")
    age: float = Field(..., ge=18, le=110)
    late_30_59: float = Field(0, ge=0, le=50)
    debt_ratio: float = Field(..., ge=0, le=10, description="Monthly debt payments / monthly income")
    monthly_income: Optional[float] = Field(None, ge=0)
    open_credit_lines: float = Field(..., ge=0, le=80)
    late_90: float = Field(0, ge=0, le=50)
    real_estate_loans: float = Field(0, ge=0, le=50)
    late_60_89: float = Field(0, ge=0, le=50)
    dependents: Optional[float] = Field(None, ge=0, le=20)

    model_config = {"json_schema_extra": {"example": {
        "utilization": 0.9, "age": 34, "late_30_59": 1, "debt_ratio": 0.6, "monthly_income": 3200,
        "open_credit_lines": 6, "late_90": 1, "real_estate_loans": 0, "late_60_89": 0, "dependents": 2}}}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    s = get_scorer()
    return {"model": s.model_name, "threshold": s.threshold, "test_metrics": s.metrics}


@app.post("/predict")
def predict(applicant: Applicant):
    start = time.perf_counter()
    result = get_scorer().score([applicant.model_dump()])[0]
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result
