# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_valid_response():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in ["low", "medium", "high"]

def test_predict_high_risk_customer():
    high_risk = {**SAMPLE, "tenure": 1, "Contract": "Month-to-month"}
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    assert r.json()["churn"] == 1

def test_invalid_negative_tenure():
    bad = {**SAMPLE, "tenure": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422