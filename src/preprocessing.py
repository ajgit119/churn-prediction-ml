# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Fix TotalCharges — some rows have blank strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    dropped = df['TotalCharges'].isna().sum()
    if dropped > 0:
        logger.info(f"Dropping {dropped} rows with blank TotalCharges")
    df = df.dropna(subset=['TotalCharges'])

    # Drop non-feature column
    df = df.drop(columns=['customerID'])

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    logger.info(f"Features: {X.shape[1]} | Train/test split: 80/20 stratified")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    from src.ingestion import load_data
    logging.basicConfig(level=logging.INFO)

    df = load_data("data/churn.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    print(f"\nX_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"\nTrain churn rate: {y_train.mean():.2%}")
    print(f"Test  churn rate: {y_test.mean():.2%}")
    print(f"\nSample X_train columns:\n{list(X_train.columns)}")