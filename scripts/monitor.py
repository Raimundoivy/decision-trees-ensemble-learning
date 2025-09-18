import pandas as pd
import numpy as np
import json
import os
from scipy.stats import ks_2samp
from src.credit_risk.data_utils import load_and_clean_data

# --- Configuration for Drift Detection ---
ARTIFACTS_DIR = "artifacts"
PROFILE_PATH = os.path.join(ARTIFACTS_DIR, "training_data_profile.json")
DRIFT_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "drift_report.json")
NUMERICAL_DRIFT_THRESHOLD = 0.05  # Using p-value from KS-test
CATEGORICAL_FEATURES = ['home', 'marital', 'records', 'job']
NUMERICAL_FEATURES = ['seniority', 'time', 'age', 'expenses', 'income', 'assets', 'debt', 'amount', 'price']


def load_data_and_profile():
    """Loads the reference data profile and a batch of new data for comparison."""
    print(f"Loading training data profile from {PROFILE_PATH}...")
    with open(PROFILE_PATH, 'r') as f:
        training_profile = json.load(f)

    print("Loading new data batch for monitoring...")
    # In a real system, this would come from a database of recent predictions.
    # Here, we simulate it by using the last 20% of the original dataset.
    df_full = load_and_clean_data("data/CreditScoring.csv")
    
    # Simulate a new batch of data (e.g., the most recent 800 records)
    live_data = df_full.tail(800).copy()

    # Apply the same basic cleaning as in train.py
    live_data.fillna(0, inplace=True)
    
    return training_profile, live_data


def check_for_drift(training_profile, live_data):
    """Compares the live data against the training profile to detect drift."""
    drift_report = {}

    # 1. Check for numerical feature drift using the Kolmogorov-Smirnov test
    print("\n--- Checking for numerical feature drift ---")
    training_stats = training_profile['numerical_stats']
    training_sample = training_profile['sample_numerical']

    for feature in NUMERICAL_FEATURES:
        training_mean = training_stats[feature]['mean']
        live_mean = live_data[feature].mean()
        
        ks_statistic, p_value = ks_2samp(training_sample[feature], live_data[feature])
        
        drift_detected = p_value < NUMERICAL_DRIFT_THRESHOLD
        drift_report[feature] = {
            'type': 'numerical',
            'training_mean': training_mean,
            'live_mean': live_mean,
            'ks_p_value': p_value,
            'drift_detected': drift_detected
        }
        if drift_detected:
            print(f"Drift DETECTED in '{feature}'. P-value: {p_value:.4f}")
        else:
            print(f"No significant drift in '{feature}'. P-value: {p_value:.4f}")

    # 2. Check for categorical feature drift (new categories or changed distributions)
    print("\n--- Checking for categorical feature drift ---")
    training_counts = training_profile['categorical_counts']
    
    for feature in CATEGORICAL_FEATURES:
        training_categories = set(training_counts[feature].keys())
        live_categories = set(live_data[feature].unique())
        
        new_categories = live_categories - training_categories
        drift_detected = bool(new_categories)
        
        drift_report[feature] = {
            'type': 'categorical',
            'training_categories': list(training_categories),
            'live_categories': list(live_categories),
            'new_categories': list(new_categories),
            'drift_detected': drift_detected
        }
        if drift_detected:
            print(f"Drift DETECTED in '{feature}'. New categories found: {new_categories}")
        else:
            print(f"No new categories found in '{feature}'.")

    return drift_report


def main():
    """Main function to run the monitoring process."""
    # Load data and profile
    training_profile, live_data = load_data_and_profile()
    
    # Generate drift report
    drift_report = check_for_drift(training_profile, live_data)
    
    # Save the report
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(drift_report, f, indent=2)
    print(f"\nDrift report saved to {DRIFT_REPORT_PATH}")
    
    # Check if any drift was detected overall
    if any(details['drift_detected'] for details in drift_report.values()):
        print("\nWARNING: Data drift detected in one or more features. Model retraining may be required.")
    else:
        print("\nSUCCESS: No significant data drift was detected.")


if __name__ == "__main__":
    main()