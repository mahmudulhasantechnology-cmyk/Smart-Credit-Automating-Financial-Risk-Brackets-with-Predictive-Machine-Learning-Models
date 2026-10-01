"""
Credit Score API (FastAPI)

Serves the final model from Model.ipynb: Logistic Regression (C=0.1, balanced),
trained on the full train.csv with label-encoded categoricals and median imputation.

Run:
    uvicorn main:app --reload

Docs:
    http://127.0.0.1:8000/docs

On first start the model is trained from train.csv and saved to model.joblib;
later starts just load it. Set RETRAIN=1 to force retraining.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = Path(os.getenv("TRAIN_PATH", BASE_DIR / "train.csv"))
ARTIFACT_PATH = Path(os.getenv("ARTIFACT_PATH", BASE_DIR / "model.joblib"))

TARGET_COL = "Credit_Score"
DROP_COLS = [TARGET_COL, "ID", "Customer_ID", "Name", "SSN", "Type_of_Loan"]
NUMERIC_CANDIDATES = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]
TARGET_MAP = {"Poor": 0, "Standard": 1, "Good": 2}
LABELS = {v: k for k, v in TARGET_MAP.items()}


# ---------------------------------------------------------------------
# Preprocessing (same code path for training and inference)
# ---------------------------------------------------------------------
def _as_text(v) -> str:
    """Turn a value into the clean string used for categorical encoding."""
    if v is None or pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.map(_as_text).str.strip(" \t_")


def encode_features(df: pd.DataFrame, feature_cols, cat_cols, categories) -> pd.DataFrame:
    """Numeric columns -> to_numeric; categorical columns -> codes using training categories.
    Categories not seen in training get code -1."""
    df = df.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        if col in cat_cols:
            df[col] = pd.Categorical(_clean_text(df[col]), categories=categories[col]).codes
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).astype(float)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_and_save() -> dict:
    if not TRAIN_PATH.exists():
        raise RuntimeError(
            f"No saved model at {ARTIFACT_PATH} and training data not found at {TRAIN_PATH}. "
            "Put train.csv next to main.py (or set TRAIN_PATH)."
        )

    raw = pd.read_csv(TRAIN_PATH, low_memory=False)

    for col in NUMERIC_CANDIDATES:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw[TARGET_COL] = raw[TARGET_COL].astype(str).str.strip(" \t_").map(TARGET_MAP)
    raw = raw.dropna(subset=[TARGET_COL])
    y = raw[TARGET_COL].astype(int)

    feature_cols = [c for c in raw.columns if c not in DROP_COLS]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(raw[c])]
    categories = {c: pd.Categorical(_clean_text(raw[c])).categories.tolist() for c in cat_cols}

    X = encode_features(raw[feature_cols], feature_cols, cat_cols, categories)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_imp, y)

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "categories": categories,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


def load_artifact() -> dict:
    if ARTIFACT_PATH.exists() and os.getenv("RETRAIN") != "1":
        return joblib.load(ARTIFACT_PATH)
    return train_and_save()


# ---------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------
class CreditRecord(BaseModel):
    """One customer-month record. Fields are the model's input features;
    anything omitted is treated as missing and filled with the training median."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "Month": "January",
                "Age": 28,
                "Occupation": "Scientist",
                "Annual_Income": 19114.12,
                "Monthly_Inhand_Salary": 1824.84,
                "Num_Bank_Accounts": 3,
                "Num_Credit_Card": 4,
                "Interest_Rate": 3,
                "Num_of_Loan": 4,
                "Delay_from_due_date": 3,
                "Num_of_Delayed_Payment": 7,
                "Changed_Credit_Limit": 11.27,
                "Num_Credit_Inquiries": 4,
                "Credit_Mix": "Good",
                "Outstanding_Debt": 809.98,
                "Credit_Utilization_Ratio": 26.82,
                "Credit_History_Age": "22 Years and 9 Months",
                "Payment_of_Min_Amount": "No",
                "Total_EMI_per_month": 49.57,
                "Amount_invested_monthly": 80.42,
                "Payment_Behaviour": "High_spent_Small_value_payments",
                "Monthly_Balance": 312.49,
            }
        },
    )

    Month: Optional[str] = None
    Age: Optional[float] = None
    Occupation: Optional[str] = None
    Annual_Income: Optional[float] = None
    Monthly_Inhand_Salary: Optional[float] = None
    Num_Bank_Accounts: Optional[float] = None
    Num_Credit_Card: Optional[float] = None
    Interest_Rate: Optional[float] = None
    Num_of_Loan: Optional[float] = None
    Delay_from_due_date: Optional[float] = None
    Num_of_Delayed_Payment: Optional[float] = None
    Changed_Credit_Limit: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = None
    Credit_Mix: Optional[str] = None
    Outstanding_Debt: Optional[float] = None
    Credit_Utilization_Ratio: Optional[float] = None
    Credit_History_Age: Optional[str] = None
    Payment_of_Min_Amount: Optional[str] = None
    Total_EMI_per_month: Optional[float] = None
    Amount_invested_monthly: Optional[float] = None
    Payment_Behaviour: Optional[str] = None
    Monthly_Balance: Optional[float] = None


class BatchRequest(BaseModel):
    records: list[CreditRecord] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    credit_score: str
    class_id: int
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["artifact"] = load_artifact()
    yield
    state.clear()


app = FastAPI(
    title="Credit Score Classification API",
    description="Predicts a customer's credit score (Poor / Standard / Good).",
    version="1.0.0",
    lifespan=lifespan,
)


def run_predictions(records: list[CreditRecord]) -> list[Prediction]:
    art = state["artifact"]
    df = pd.DataFrame([r.model_dump() for r in records])
    X = encode_features(df, art["feature_cols"], art["cat_cols"], art["categories"])
    X_imp = art["imputer"].transform(X)

    model = art["model"]
    proba = model.predict_proba(X_imp)
    classes = [int(c) for c in model.classes_]

    results = []
    for row in proba:
        best = int(np.argmax(row))
        class_id = classes[best]
        results.append(
            Prediction(
                credit_score=LABELS[class_id],
                class_id=class_id,
                probabilities={LABELS[c]: round(float(p), 4) for c, p in zip(classes, row)},
            )
        )
    return results


@app.get("/")
def root():
    return {"service": "Credit Score Classification API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifact" in state}


@app.post("/predict", response_model=Prediction)
def predict(record: CreditRecord):
    try:
        return run_predictions([record])[0]
    except Exception as exc:  # surface preprocessing/model errors as a clean 500
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    try:
        return BatchResponse(predictions=run_predictions(req.records))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
