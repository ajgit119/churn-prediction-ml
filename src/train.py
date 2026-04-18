# src/train.py
import joblib, logging
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from src.features import build_preprocessor
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest':       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
     'gradient_boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=200, random_state=42),
    'xgboost':             XGBClassifier(n_estimators=200, learning_rate=0.05,
                               eval_metric='logloss', random_state=42),
}

def train_best(X_train, y_train) -> Pipeline:
    os.makedirs("models", exist_ok=True)
    best_score, best_name, best_pipe = 0, '', None

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run(run_name="model_selection"):
        for name, model in MODELS.items():
            logger.info(f"Training {name}...")
            prep = build_preprocessor()
            pipe = Pipeline([('prep', prep), ('model', model)])

            scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            mean_auc = scores.mean()
            std_auc  = scores.std()

            mlflow.log_metric(f"{name}_auc_mean", round(mean_auc, 4))
            mlflow.log_metric(f"{name}_auc_std",  round(std_auc, 4))
            logger.info(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")

            if mean_auc > best_score:
                best_score, best_name, best_pipe = mean_auc, name, pipe

        logger.info(f"\nBest model: {best_name} (AUC={best_score:.4f})")
        best_pipe.fit(X_train, y_train)

        joblib.dump(best_pipe, "models/churn_model.pkl")
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_auc",   round(best_score, 4))
        mlflow.sklearn.log_model(best_pipe, "model")
        logger.info("Model saved to models/churn_model.pkl")

    return best_pipe, best_name

if __name__ == "__main__":
    from src.ingestion import load_data
    from src.preprocessing import preprocess

    df = load_data("data/churn.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    best_pipe, best_name = train_best(X_train, y_train)
    print(f"\nDone! Best model: {best_name}")
    print("Saved: models/churn_model.pkl")