# src/ingestion.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load CSV (or TSV) and return DataFrame."""
    # Auto-detect separator — handles both comma and tab
    df = pd.read_csv(path, sep=None, engine='python')
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df = load_data("data/churn.csv")
    print(df.head(3))
    print("\nShape:", df.shape)
    print("\nChurn counts:\n", df['Churn'].value_counts())
    print("\nNull values:\n", df.isnull().sum()[df.isnull().sum() > 0])