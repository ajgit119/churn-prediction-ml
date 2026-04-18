# Customer Churn Prediction System

End-to-end ML pipeline predicting telecom customer churn — from raw data to a deployed REST API.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 78.96% |
| ROC-AUC | 0.8345 |
| Model | Random Forest |

## Project Structure
project/
├── data/          # Raw CSV data
├── src/           # Pipeline modules
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor.py
├── app/           # FastAPI service
│   └── main.py
├── models/        # Saved model (.pkl)
├── tests/         # Pytest suite
├── Dockerfile
└── requirements.txt

## Quickstart
```bash
pip install -r requirements.txt

# Train
python -m src.train

# Evaluate
python -m src.evaluate

# Serve
uvicorn app.main:app --reload
```

## API
`POST /predict` — returns churn prediction, probability, and risk level  
`GET  /health`  — health check  
`GET  /predictions/history` — last 20 predictions  

Interactive docs: http://localhost:8000/docs

## Tech Stack
Python · scikit-learn · XGBoost · FastAPI · MLflow · Docker
Run tests and share the output — then your project is 100% complete and portfolio-ready! 🎉



# 📊 Churn Prediction ML System

End-to-end machine learning project for predicting customer churn using classical ML models, deployed via FastAPI and tracked with MLflow.

---

## 🚀 Features

* 🔄 Data preprocessing pipeline
* 🤖 Model training & selection (LogReg, Random Forest, XGBoost)
* 📈 Cross-validation with ROC-AUC
* 📊 MLflow experiment tracking
* 🌐 FastAPI deployment
* 🧪 Pytest-based testing
* 🐳 Docker-ready

---

## 📂 Project Structure

```
.
├── app/                # FastAPI app
├── src/                # ML pipeline (ingestion, preprocessing, training)
├── tests/              # API tests
├── models/             # Saved models (ignored in git)
├── data/               # Dataset (ignored in git)
├── requirements.txt
├── Dockerfile
```

---

## 📊 Dataset

This project uses the **Telco Customer Churn dataset**.

👉 Download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place it in:

```
data/churn.csv
```

---

## ⚙️ Setup

```bash
python -m pip install -r requirements.txt
```

---

## 🧠 Train Model

```bash
python -m src.train
```

---

## 🌐 Run API

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Run Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

---

## 🐳 Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 📈 Example Output

```json
{
  "churn": 1,
  "probability": 0.51,
  "risk": "medium"
}
```

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* XGBoost
* FastAPI
* MLflow
* Pytest
* Docker

---

## 🎯 Status

✅ End-to-end ML pipeline
✅ API deployment
✅ Testing
🔄 Ready for cloud deployment

---

## 🚀 Future Improvements

* Hyperparameter tuning (Optuna)
* SHAP explainability
* Cloud deployment (AWS / Render)
* Frontend dashboard

---
