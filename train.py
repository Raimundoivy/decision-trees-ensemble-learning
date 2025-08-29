import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures # Import PolynomialFeatures


# Parameters
output_file = 'model.joblib'
validation_split = 0.2


# Advanced Feature Engineering
def feature_engineering(df):
    # Ratio features - these are added to the original df
    df_with_ratio_features = df.copy() # Start with a copy to avoid modifying the original df passed in
    df_with_ratio_features['debt_to_income_ratio'] = df_with_ratio_features['debt'] / (df_with_ratio_features['income'] + 1e-6)
    df_with_ratio_features['assets_to_debt_ratio'] = df_with_ratio_features['assets'] / (df_with_ratio_features['debt'] + 1e-6)
    df_with_ratio_features['loan_amount_to_income_ratio'] = df_with_ratio_features['amount'] / (df_with_ratio_features['income'] + 1e-6)
    df_with_ratio_features['seniority_to_age_ratio'] = df_with_ratio_features['seniority'] / (df_with_ratio_features['age'] + 1e-6)
    df_with_ratio_features['expenses_to_income_ratio'] = df_with_ratio_features['expenses'] / (df_with_ratio_features['income'] + 1e-6)

    # List of all numerical features (original + ratio)
    all_numerical_features = [
        'seniority', 'time', 'age', 'expenses', 'income',
        'assets', 'debt', 'amount', 'price',
        'debt_to_income_ratio', 'assets_to_debt_ratio',
        'loan_amount_to_income_ratio', 'seniority_to_age_ratio',
        'expenses_to_income_ratio'
    ]

    # List of original numerical features
    original_numerical_features = [
        'seniority', 'time', 'age', 'expenses', 'income',
        'assets', 'debt', 'amount', 'price'
    ]

    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features_array = poly.fit_transform(df_with_ratio_features[all_numerical_features])
    poly_feature_names = poly.get_feature_names_out(all_numerical_features)
    df_poly = pd.DataFrame(poly_features_array, columns=poly_feature_names, index=df_with_ratio_features.index)

    # Drop the original numerical features (degree 1 terms) from df_poly
    columns_to_drop_from_df_poly = [col for col in all_numerical_features if col in df_poly.columns]
    df_poly_only_new_terms = df_poly.drop(columns=columns_to_drop_from_df_poly)

    # Now, we need to combine:
    # 1. Original categorical features from df_with_ratio_features
    # 2. Ratio features from df_with_ratio_features (these are already in df_with_ratio_features)
    # 3. New polynomial features from df_poly_only_new_terms

    # Get all columns that are NOT original numerical features
    columns_to_keep = [col for col in df_with_ratio_features.columns if col not in original_numerical_features]
    df_final = df_with_ratio_features[columns_to_keep]

    # Concatenate with the new polynomial terms
    df_engineered = pd.concat([df_final, df_poly_only_new_terms], axis=1)

    return df_engineered


def main():
    # Data preparation
    df = pd.read_csv('CreditScoring.csv')
    df.columns = df.columns.str.lower()

    df = feature_engineering(df)

    df_train, df_val = train_test_split(df, test_size=validation_split, random_state=42)

    y_train = (df_train.status == 2).astype(int)
    y_val = (df_val.status == 2).astype(int)

    del df_train['status']
    del df_val['status']

    dict_train = df_train.to_dict(orient='records')
    dict_val = df_val.to_dict(orient='records')


    # Training
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_leaf': [1, 5, 10, 15],
        'criterion': ['gini', 'entropy']
    }

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model
    dt = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")


    # Saving the model
    joblib.dump((dv, dt), output_file)

    print(f'The model is saved to {output_file}')

if __name__ == "__main__":
    main()