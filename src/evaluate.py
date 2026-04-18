# src/evaluate.py
import joblib, logging
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_proba), 4),
        "report"   : classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
        "confusion" : confusion_matrix(y_test, y_pred).tolist(),
    }

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("test_roc_auc",  metrics["roc_auc"])

    return metrics

if __name__ == "__main__":
    from src.ingestion import load_data
    from src.preprocessing import preprocess

    df = load_data("data/churn.csv")
    _, X_test, _, y_test = preprocess(df)

    pipeline = joblib.load("models/churn_model.pkl")
    metrics  = evaluate(pipeline, X_test, y_test)

    cm = metrics["confusion"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    print(f"\n{'='*45}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*45}")
    print(f"  Accuracy  : {metrics['accuracy']:.2%}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"{'='*45}")
    print(f"\nConfusion Matrix:")
    print(f"  True  Negatives (correct no-churn) : {tn}")
    print(f"  False Positives (wrong churn alert) : {fp}")
    print(f"  False Negatives (missed churners)   : {fn}")
    print(f"  True  Positives (caught churners)   : {tp}")
    print(f"\nClassification Report:")
    print(metrics["report"])