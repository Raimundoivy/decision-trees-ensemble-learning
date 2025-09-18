import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.credit_risk.train import feature_engineering # Import the feature engineering function
from src.credit_risk.data_utils import load_and_clean_data

# Sample consistent with API tests
APPLICANT_DATA = {
    "seniority": 9,
    "home": "rent",
    "time": 60,
    "age": 30,
    "marital": "married",
    "records": "no",
    "job": "freelance",
    "expenses": 73,
    "income": 129.0,
    "assets": 0.0,
    "debt": 0.0,
    "amount": 800,
    "price": 846,
}


def test_model_predict_proba_bounds():
    """Tests that the model prediction is a valid probability."""
    artifact = joblib.load("artifacts/model.joblib")
    dv = artifact["vectorizer"]
    model = artifact["model"]
    selector = artifact["feature_selector"]

    # Apply the full transformation pipeline to the single applicant data
    applicant_df = pd.DataFrame([APPLICANT_DATA])
    applicant_engineered_df = feature_engineering(applicant_df)
    applicant_dict_engineered = applicant_engineered_df.to_dict(orient='records')
    X_raw = dv.transform(applicant_dict_engineered)
    X_selected = selector.transform(X_raw)
    proba = float(model.predict_proba(X_selected)[0, 1])
    assert 0.0 <= proba <= 1.0


def test_model_performance_auc():
    """Tests that the model AUC on validation data is above a threshold."""
    # Load the model
    artifact = joblib.load("artifacts/model.joblib")
    dv = artifact["vectorizer"]
    model = artifact["model"]
    selector = artifact["feature_selector"]

    # --- Load and prepare validation data ---
    # Use the centralized data loading and cleaning function
    # Note: We assume tests are run from the project root directory
    df = load_and_clean_data("data/CreditScoring.csv")

    # Split data to get the same validation set as in training
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Create target variable
    y_val = (df_val.status == 'default').astype(int)

    # Fill NaNs and remove target column
    df_val = df_val.fillna(0)
    del df_val["status"]

    # --- Prediction and validation ---
    # Apply the same feature engineering and vectorization as in training
    df_val_engineered = feature_engineering(df_val)
    val_dicts = df_val_engineered.to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    X_val_selected = selector.transform(X_val)

    y_pred = model.predict_proba(X_val_selected)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    # Assert that the AUC is above the minimum threshold
    # NOTE: The AUC will be different now because we are testing the final model artifact
    # which includes feature selection. This is a more accurate regression test.
    assert auc >= 0.80, f"Model AUC score ({auc:.3f}) is below the regression threshold of 0.80"