import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    dv, model = joblib.load("model.joblib")
    X = dv.transform([APPLICANT_DATA])
    proba = float(model.predict_proba(X)[0, 1])
    assert 0.0 <= proba <= 1.0


def test_model_performance_auc():
    """Tests that the model AUC on validation data is above a threshold."""
    # Load the model
    dv, model = joblib.load("model.joblib")

    # --- Load and prepare validation data ---
    df = pd.read_csv("CreditScoring.csv")
    df.columns = df.columns.str.lower()

    # Map categorical features
    status_values = {1: 'ok', 2: 'default', 0: 'unk'}
    df.status = df.status.map(status_values)
    home_values = {1: 'rent', 2: 'owner', 3: 'private', 4: 'ignore', 5: 'parents', 6: 'other', 0: 'unk'}
    df.home = df.home.map(home_values)
    marital_values = {1: 'single', 2: 'married', 3: 'widow', 4: 'separated', 5: 'divorced', 0: 'unk'}
    df.marital = df.marital.map(marital_values)
    records_values = {1: 'no', 2: 'yes', 0: 'unk'}
    df.records = df.records.map(records_values)
    job_values = {1: 'fixed', 2: 'parttime', 3: 'freelance', 4: 'others', 0: 'unk'}
    df.job = df.job.map(job_values)

    # Clean special values
    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    df = df[df.status != 'unk']

    # Split data to get the same validation set as in training
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Create target variable
    y_val = (df_val.status == 'default').astype(int)

    # Fill NaNs and remove target column
    df_val = df_val.fillna(0)
    del df_val["status"]

    # --- Prediction and validation ---
    val_dicts = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    # Assert that the AUC is above the minimum threshold
    assert auc >= 0.80, f"Model AUC score ({auc:.3f}) is below the threshold of 0.80"