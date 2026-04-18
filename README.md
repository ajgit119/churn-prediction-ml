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