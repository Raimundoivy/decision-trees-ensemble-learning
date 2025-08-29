import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from train import feature_engineering, DictVectorizer, DecisionTreeClassifier, GridSearchCV, joblib
import numpy as np

# Sample data for testing feature_engineering
@pytest.fixture
def sample_dataframe():
    data = {
        'seniority': [1, 5, 10],
        'home': ['rent', 'owner', 'mortgage'],
        'time': [12, 36, 60],
        'age': [25, 35, 45],
        'marital': ['single', 'married', 'divorced'],
        'records': ['no', 'yes', 'no'],
        'job': ['freelance', 'engineer', 'manager'],
        'expenses': [50, 100, 150],
        'income': [1000, 2000, 3000],
        'assets': [0, 5000, 10000],
        'debt': [100, 200, 300],
        'amount': [500, 1500, 2500],
        'price': [550, 1600, 2600],
        'status': [1, 2, 1]
    }
    return pd.DataFrame(data)

def test_feature_engineering_ratio_features(sample_dataframe):
    df_engineered = feature_engineering(sample_dataframe.copy())

    # Check if ratio features are created
    assert 'debt_to_income_ratio' in df_engineered.columns
    assert 'assets_to_debt_ratio' in df_engineered.columns
    assert 'loan_amount_to_income_ratio' in df_engineered.columns
    assert 'seniority_to_age_ratio' in df_engineered.columns
    assert 'expenses_to_income_ratio' in df_engineered.columns

    # Check some calculated values (using a small epsilon for float comparison)
    assert abs(df_engineered.loc[0, 'debt_to_income_ratio'] - (100 / 1000)) < 1e-6
    assert abs(df_engineered.loc[1, 'assets_to_debt_ratio'] - (5000 / 200)) < 1e-6

def test_feature_engineering_polynomial_features(sample_dataframe):
    df_engineered = feature_engineering(sample_dataframe.copy())

    # List of original numerical features that should be dropped
    original_numerical_features_to_drop = [
        'seniority', 'time', 'age', 'expenses', 'income',
        'assets', 'debt', 'amount', 'price'
    ]
    for feature in original_numerical_features_to_drop:
        assert feature not in df_engineered.columns

    # Check if ratio features are still present
    ratio_features = [
        'debt_to_income_ratio', 'assets_to_debt_ratio',
        'loan_amount_to_income_ratio', 'seniority_to_age_ratio',
        'expenses_to_income_ratio'
    ]
    for feature in ratio_features:
        assert feature in df_engineered.columns

    # Check for presence of some expected polynomial features (e.g., 'seniority^2', 'seniority time')
    # This is a basic check, more robust checks might involve specific feature names
    poly_feature_present = False
    for col in df_engineered.columns:
        if 'seniority' in col and '^2' in col:
            poly_feature_present = True
            break
    assert poly_feature_present

    poly_feature_present = False
    for col in df_engineered.columns:
        if 'seniority' in col and 'time' in col and ' ' in col: # Check for interaction term
            poly_feature_present = True
            break
    assert poly_feature_present

    # Ensure the number of rows remains the same
    assert len(df_engineered) == len(sample_dataframe)

# Mocking the entire training process for an integration-like test
@patch('train.pd.read_csv')
@patch('train.joblib.dump')
@patch('train.DictVectorizer')
@patch('train.DecisionTreeClassifier')
@patch('train.GridSearchCV')
def test_full_training_pipeline(mock_grid_search, mock_dt_classifier, mock_dv, mock_joblib_dump, mock_read_csv, sample_dataframe):
    # Configure mock for pd.read_csv to return our sample_dataframe
    mock_read_csv.return_value = sample_dataframe

    # Mock DictVectorizer
    mock_dv_instance = MagicMock()
    mock_dv.return_value = mock_dv_instance
    mock_dv_instance.fit_transform.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mock_dv_instance.transform.return_value = np.array([[10, 11, 12]])

    # Mock DecisionTreeClassifier
    mock_dt_instance = MagicMock()
    mock_dt_classifier.return_value = mock_dt_instance

    # Mock GridSearchCV
    mock_grid_search_instance = MagicMock()
    mock_grid_search.return_value = mock_grid_search_instance
    mock_grid_search_instance.best_estimator_ = mock_dt_instance # GridSearchCV returns the best estimator
    mock_grid_search_instance.best_params_ = {'max_depth': 10, 'min_samples_leaf': 5}
    mock_grid_search_instance.fit.return_value = None # fit doesn't need to return anything specific for this test

    # Call the main function of train.py
    from train import main
    main()

    # Assert that read_csv was called
    mock_read_csv.assert_called_once_with('CreditScoring.csv')

    # Assert that DictVectorizer was instantiated and fit_transform was called
    mock_dv.assert_called_once()
    mock_dv_instance.fit_transform.assert_called_once()
    mock_dv_instance.transform.assert_called_once()

    # Assert that DecisionTreeClassifier was instantiated
    mock_dt_classifier.assert_called_once()

    # Assert that GridSearchCV was instantiated and fit was called
    mock_grid_search.assert_called_once()
    mock_grid_search_instance.fit.assert_called_once()

    # Assert that joblib.dump was called to save the model
    mock_joblib_dump.assert_called_once_with((mock_dv_instance, mock_dt_instance), 'model.joblib')