import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

# Parameters
output_file = "model.joblib"
metrics_file = "training_metrics.json"
validation_split = 0.2


# This function now only contains the feature generation part
def feature_engineering(df):
    df_with_ratio_features = df.copy()
    # Ratio features
    df_with_ratio_features["debt_to_income_ratio"] = df_with_ratio_features["debt"] / (df_with_ratio_features["income"] + 1e-6)
    df_with_ratio_features["assets_to_debt_ratio"] = df_with_ratio_features["assets"] / (df_with_ratio_features["debt"] + 1e-6)
    df_with_ratio_features["loan_amount_to_income_ratio"] = df_with_ratio_features["amount"] / (df_with_ratio_features["income"] + 1e-6)
    df_with_ratio_features["seniority_to_age_ratio"] = df_with_ratio_features["seniority"] / (df_with_ratio_features["age"] + 1e-6)
    df_with_ratio_features["expenses_to_income_ratio"] = df_with_ratio_features["expenses"] / (df_with_ratio_features["income"] + 1e-6)

    all_numerical_features = [
        "seniority", "time", "age", "expenses", "income", "assets", "debt", "amount", "price",
        "debt_to_income_ratio", "assets_to_debt_ratio", "loan_amount_to_income_ratio",
        "seniority_to_age_ratio", "expenses_to_income_ratio",
    ]
    original_numerical_features = [
        "seniority", "time", "age", "expenses", "income", "assets", "debt", "amount", "price",
    ]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features_array = poly.fit_transform(df_with_ratio_features[all_numerical_features])
    poly_feature_names = poly.get_feature_names_out(all_numerical_features)
    df_poly = pd.DataFrame(
        poly_features_array,
        columns=poly_feature_names,
        index=df_with_ratio_features.index,
    )

    columns_to_drop_from_df_poly = [col for col in all_numerical_features if col in df_poly.columns]
    df_poly_only_new_terms = df_poly.drop(columns=columns_to_drop_from_df_poly)

    columns_to_keep = [col for col in df_with_ratio_features.columns if col not in original_numerical_features]
    df_final = df_with_ratio_features[columns_to_keep]

    df_engineered = pd.concat([df_final, df_poly_only_new_terms], axis=1)
    return df_engineered


def main():
    # Data loading and initial cleaning
    df = pd.read_csv("CreditScoring.csv")
    df.columns = df.columns.str.lower()

    # --- Start: Data Cleaning and Mapping from Notebook ---
    # Map categorical features from numeric codes to strings
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

    # Handle special numeric values and missing data
    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)

    # Remove unknown status
    df = df[df.status != 'unk']
    # --- End: Data Cleaning and Mapping ---

    # Split data
    df_train, df_val = train_test_split(df, test_size=validation_split, random_state=42)

    # Create target variable
    y_train = (df_train.status == 'default').astype(int)
    y_val = (df_val.status == 'default').astype(int)

    # Fill missing values after split
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)

    # Delete status column
    del df_train["status"]
    del df_val["status"]

    # Apply feature engineering
    df_train_engineered = feature_engineering(df_train)
    df_val_engineered = feature_engineering(df_val)

    # Convert to dictionaries for vectorizer
    dict_train = df_train_engineered.to_dict(orient="records")
    dict_val = df_val_engineered.to_dict(orient="records")

    # Training
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)

    # Hyperparameter tuning with GridSearchCV for RandomForest
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [10, 15],
        'min_samples_leaf': [3, 5]
    }

    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Best model
    model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Saving the model
    joblib.dump((dv, model), output_file)
    print(f"The model is saved to {output_file}")

    # Save metrics
    metrics = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()

