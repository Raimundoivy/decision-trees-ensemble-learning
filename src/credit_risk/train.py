import pandas as pd
import numpy as np
import joblib
import json
import os
import yaml
import importlib
from datetime import datetime, UTC
from pathlib import Path

# --- Imports for Model Experimentation, Feature Selection, and Explainability ---
import shap
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# --- Helper function to dynamically load classes from config ---
def get_class(class_path):
    """Dynamically imports a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def feature_engineering(df):
    """Applies feature engineering to the input dataframe."""
    # Create ratio features
    df_with_ratio_features = df.copy()
    df_with_ratio_features["debt_to_income_ratio"] = df_with_ratio_features["debt"] / (df_with_ratio_features["income"] + 1e-6)
    df_with_ratio_features["assets_to_debt_ratio"] = df_with_ratio_features["assets"] / (df_with_ratio_features["debt"] + 1e-6)
    df_with_ratio_features["loan_amount_to_income_ratio"] = df_with_ratio_features["amount"] / (df_with_ratio_features["income"] + 1e-6)
    df_with_ratio_features["seniority_to_age_ratio"] = df_with_ratio_features["seniority"] / (df_with_ratio_features["age"] + 1e-6)
    df_with_ratio_features["expenses_to_income_ratio"] = df_with_ratio_features["expenses"] / (df_with_ratio_features["income"] + 1e-6)
    all_numerical_features = ["seniority", "time", "age", "expenses", "income", "assets", "debt", "amount", "price","debt_to_income_ratio", "assets_to_debt_ratio", "loan_amount_to_income_ratio","seniority_to_age_ratio", "expenses_to_income_ratio",]
    
    # Create polynomial and interaction features
    original_numerical_features = ["seniority", "time", "age", "expenses", "income", "assets", "debt", "amount", "price",]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features_array = poly.fit_transform(df_with_ratio_features[all_numerical_features])
    poly_feature_names = poly.get_feature_names_out(all_numerical_features)
    df_poly = pd.DataFrame(poly_features_array,columns=poly_feature_names,index=df_with_ratio_features.index,)
    
    # Combine engineered features with original categorical features
    columns_to_drop_from_df_poly = [col for col in all_numerical_features if col in df_poly.columns]
    df_poly_only_new_terms = df_poly.drop(columns=columns_to_drop_from_df_poly)
    
    # Keep original categorical features and newly engineered features
    columns_to_keep = [col for col in df_with_ratio_features.columns if col not in original_numerical_features]
    df_final = df_with_ratio_features[columns_to_keep]
    df_engineered = pd.concat([df_final, df_poly_only_new_terms], axis=1)
    
    return df_engineered


def main():
    # --- SDS Recommendation: Load all parameters from config.yaml ---
    print("--- Loading configuration from config.yaml ---")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Data loading and initial cleaning
    df = pd.read_csv(config["data_paths"]["raw_data"])
    df.columns = df.columns.str.lower()
    
    # [Data cleaning logic remains the same...]
    status_values = {1: 'ok', 2: 'default', 0: 'unk'}; df.status = df.status.map(status_values)
    home_values = {1: 'rent', 2: 'owner', 3: 'private', 4: 'ignore', 5: 'parents', 6: 'other', 0: 'unk'}; df.home = df.home.map(home_values)
    marital_values = {1: 'single', 2: 'married', 3: 'widow', 4: 'separated', 5: 'divorced', 0: 'unk'}; df.marital = df.marital.map(marital_values)
    records_values = {1: 'no', 2: 'yes', 0: 'unk'}; df.records = df.records.map(records_values)
    job_values = {1: 'fixed', 2: 'parttime', 3: 'freelance', 4: 'others', 0: 'unk'}; df.job = df.job.map(job_values)
    for c in ['income', 'assets', 'debt']: df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    df = df[df.status != 'unk']
    
    # Data splitting
    df_train, df_val = train_test_split(df, test_size=config["training"]["validation_split_size"], random_state=config["training"]["random_state"])
    y_train = (df_train.status == 'default').astype(int); y_val = (df_val.status == 'default').astype(int)
    df_train = df_train.fillna(0); df_val = df_val.fillna(0)
    del df_train["status"]; del df_val["status"]

    # Feature Engineering and Vectorization
    df_train_engineered = feature_engineering(df_train); df_val_engineered = feature_engineering(df_val)
    dict_train = df_train_engineered.to_dict(orient="records"); dict_val = df_val_engineered.to_dict(orient="records")
    dv = DictVectorizer(sparse=False); X_train = dv.fit_transform(dict_train); X_val = dv.transform(dict_val)

    # Feature Selection Step
    print("\n--- Running Feature Selection ---")
    selection_model_class = get_class("sklearn.ensemble.RandomForestClassifier")
    selection_model = selection_model_class(n_estimators=config["features"]["feature_selection"]["n_estimators"], random_state=config["training"]["random_state"], n_jobs=-1)
    selector = SelectFromModel(selection_model, threshold=config["features"]["feature_selection"]["threshold"], prefit=False)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train); X_val_selected = selector.transform(X_val)
    print(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")

    # Model Experimentation Loop
    all_metrics = {}; best_model_overall = None; best_auc_score = -1; best_model_name = ""

    for model_name, model_info in config["training"]["models_to_run"].items():
        print(f"\n--- Training {model_name} ---")
        EstimatorClass = get_class(model_info["estimator_class"])
        estimator = EstimatorClass(**model_info["static_params"])
        
        grid_search = GridSearchCV(estimator=estimator, param_grid=model_info["param_grid"], cv=config["training"]["cv_folds"], n_jobs=-1, verbose=1, scoring='roc_auc')
        grid_search.fit(X_train_selected, y_train)

        model = grid_search.best_estimator_
        y_pred_val = model.predict_proba(X_val_selected)[:, 1]
        auc_val = roc_auc_score(y_val, y_pred_val)

        print(f"Best CV AUC for {model_name}: {grid_search.best_score_:.4f}"); print(f"Validation AUC for {model_name}: {auc_val:.4f}")

        all_metrics[model_name] = {"best_params": grid_search.best_params_, "best_cv_auc": grid_search.best_score_, "validation_auc": auc_val}
        if auc_val > best_auc_score:
            best_auc_score = auc_val; best_model_overall = model; best_model_name = model_name

    # Create and Save Final Artifact
    print(f"\n--- Best Model: {best_model_name} (Validation AUC: {best_auc_score:.4f}) ---")
    explainer = None
    if 'Logistic' in best_model_name: explainer = shap.LinearExplainer(best_model_overall, X_train_selected)
    else: explainer = shap.TreeExplainer(best_model_overall)

    final_artifact = {"vectorizer": dv, "feature_selector": selector, "model": best_model_overall, "explainer": explainer}

    # --- Create output directory and save artifacts ---
    output_dir = Path(config["data_paths"]["model_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    versioned_filename = f"model-{timestamp}-{best_model_name}-{best_auc_score:.2f}.joblib"
    versioned_filepath = output_dir / versioned_filename
    latest_filepath = output_dir / config["data_paths"]["latest_model_name"]

    joblib.dump(final_artifact, versioned_filepath); joblib.dump(final_artifact, latest_filepath)
    print(f"Versioned artifact saved to: {versioned_filepath}"); print(f"Latest artifact saved to: {latest_filepath}")

    # Save Metrics and Data Profile
    final_metrics_log = {"best_model": best_model_name, "best_validation_auc": best_auc_score, "timestamp": datetime.now(UTC).isoformat(), "experiments": all_metrics}
    with open(config["data_paths"]["metrics_output"], "w") as f: json.dump(final_metrics_log, f, indent=2)
    print(f"Metrics saved to {config['data_paths']['metrics_output']}")
    
    # --- Create a comprehensive data profile for monitoring ---
    # This profile includes stats, value counts, and a data sample for drift detection.
    profile = {
        'numerical_stats': df_train[config["features"]["numerical_features"]].describe().to_dict(),
        'categorical_counts': {
            col: df_train[col].value_counts().to_dict() 
            for col in config["features"]["categorical_features"]
        },
        'sample_numerical': df_train[config["features"]["numerical_features"]]
                            .sample(500, random_state=config["training"]["random_state"])
                            .to_dict('list')
    }
    with open(config["data_paths"]["training_profile"], 'w') as f: json.dump(profile, f, indent=2)
    print(f"Training data profile saved to {config['data_paths']['training_profile']}")

if __name__ == "__main__":
    main()