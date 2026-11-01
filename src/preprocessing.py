"""Reusable preprocessing for the Telco Customer Churn assignment.

The pipeline returned by build_preprocessor() accepts raw feature columns exactly
as they appear in the original CSV or in test_data.csv (project root), so model
training and the Streamlit app share the same transformations.

Running this module creates the stratified train/test split and saves the
held-out test set:  python3 -m src.preprocessing
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

ID_COLUMN = "customerID"
TARGET = "Churn"
TARGET_MAPPING = {"No": 0, "Yes": 1}

# The 19 model inputs, in original CSV order (customerID and Churn excluded).
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is stored as 0/1 but is a yes/no flag, so it is categorical.
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERICAL_FEATURES]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_data(path=RAW_DATA_PATH):
    """Read the original CSV without altering it on disk."""
    return pd.read_csv(path)


def encode_target(churn):
    """Map Churn labels No -> 0 and Yes -> 1; reject any other label."""
    encoded = churn.astype(str).str.strip().map(TARGET_MAPPING)
    if encoded.isna().any():
        unexpected = sorted(churn[encoded.isna()].astype(str).unique())
        raise ValueError(f"Unexpected {TARGET} labels: {unexpected}")
    return encoded.astype(int)


def split_features_target(df):
    """Return the 19 raw input columns and the encoded target.

    Works on the original CSV (customerID is dropped) and on test_data.csv.
    """
    _check_columns(df, FEATURE_COLUMNS + [TARGET])
    return df[FEATURE_COLUMNS].copy(), encode_target(df[TARGET])


def clean_features(X):
    """Stateless cleaning applied before encoding; nothing is learned from data.

    - Keeps only the 19 input columns, so extra columns (customerID, Churn) are ignored.
    - Converts numerical columns with pd.to_numeric(errors="coerce"); the blank
      TotalCharges strings become NaN.
    - Fills missing TotalCharges with 0: in this dataset the blanks occur only
      for tenure == 0 customers, who have not been billed yet.
    - Casts categorical columns to stripped strings so SeniorCitizen 0/1 and
      text categories are encoded the same way at training and app time.
    """
    _check_columns(X, FEATURE_COLUMNS)
    X = X[FEATURE_COLUMNS].copy()
    for col in NUMERICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    return X


def _cleaned_feature_names(transformer, input_features):
    # Module-level (not a lambda) so the fitted pipeline can be pickled with joblib.
    return np.asarray(FEATURE_COLUMNS, dtype=object)


def build_preprocessor():
    """Return an unfitted Pipeline: stateless cleaning -> scaling + one-hot encoding.

    Fit it on training data only (normally as the first step of a model pipeline).
    """
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("clean", FunctionTransformer(clean_features,
                                      feature_names_out=_cleaned_feature_names)),
        ("encode", encoder),
    ])


def split_data(X, y):
    """80/20 stratified split with a fixed seed."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


def load_train_test_split(path=RAW_DATA_PATH):
    """Load the raw CSV and return X_train, X_test, y_train, y_test (raw features)."""
    X, y = split_features_target(load_raw_data(path))
    return split_data(X, y)


def save_test_data(test_index, path=TEST_DATA_PATH, raw_path=RAW_DATA_PATH):
    """Save the held-out rows exactly as they appear in the raw CSV, minus customerID.

    Rows are re-read as text so values keep their original formatting (e.g.
    MonthlyCharges "91" stays "91", blank TotalCharges stay blank); numeric
    conversion happens later in clean_features(), wherever the file is loaded.
    """
    raw_text = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    test_df = raw_text.loc[test_index, FEATURE_COLUMNS + [TARGET]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(path, index=False)
    return test_df


def _check_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main():
    X_train, X_test, y_train, y_test = load_train_test_split()
    save_test_data(X_test.index)
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.4f}  Test churn rate: {y_test.mean():.4f}")
    print(f"Saved test set to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
