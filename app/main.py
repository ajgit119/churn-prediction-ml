# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib, pandas as pd, logging, os
from src.monitor import init_db, log_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn",
    version="1.0.0"
)

MODEL_PATH = "models/churn_model.pkl"
model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")
init_db()

class CustomerFeatures(BaseModel):
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           float
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     float

    @field_validator('tenure', 'MonthlyCharges', 'TotalCharges')
    @classmethod
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError('must be >= 0')
        return v

class PredictionOut(BaseModel):
    churn:       int
    probability: float
    risk_level:  str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict", response_model=PredictionOut)
def predict(data: CustomerFeatures):
    try:
        df   = pd.DataFrame([data.model_dump()])
        pred = int(model.predict(df)[0])
        prob = round(float(model.predict_proba(df)[0][1]), 4)

        risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

        log_prediction(data.model_dump(), pred, prob)
        logger.info(f"Prediction: churn={pred}, prob={prob}, risk={risk}")

        return PredictionOut(churn=pred, probability=prob, risk_level=risk)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/predictions/history")
def history():
    import sqlite3, json
    conn = sqlite3.connect("predictions.db")
    rows = conn.execute(
        "SELECT ts, pred, prob FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [{"timestamp": r[0], "churn": r[1], "probability": r[2]} for r in rows]