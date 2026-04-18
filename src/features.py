# src/features.py
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

CATEGORICAL = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
            ('num', StandardScaler(), NUMERICAL),
        ],
        remainder='drop'
    )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from src.ingestion import load_data
    from src.preprocessing import preprocess

    df = load_data("data/churn.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    prep = build_preprocessor()
    X_train_transformed = prep.fit_transform(X_train)
    X_test_transformed  = prep.transform(X_test)

    print(f"X_train after encoding: {X_train_transformed.shape}")
    print(f"X_test  after encoding: {X_test_transformed.shape}")
    print(f"\nCategorical cols : {len(CATEGORICAL)}")
    print(f"Numerical cols   : {len(NUMERICAL)}")
    print(f"Total features   : {X_train_transformed.shape[1]} (after one-hot expansion)")