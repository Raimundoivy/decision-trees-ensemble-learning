import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(data_path='CreditScoring.csv', model_output_path='model.joblib'):
    """Trains and saves a model.

    Args:
        data_path (str): Path to the training data.
        model_output_path (str): Path to save the trained model.
    """
    logging.info("Starting model training...")

    # Load data
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Preprocess data
    df.columns = df.columns.str.lower()
    status_values = {
        1: 'ok',
        2: 'default',
        0: 'unk'
    }
    df.status = df.status.map(status_values)

    home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        0: 'unk'
    }
    df.home = df.home.map(home_values)

    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }
    df.marital = df.marital.map(marital_values)

    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }
    df.records = df.records.map(records_values)

    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    }
    df.job = df.job.map(job_values)

    for col in ['income', 'assets', 'debt']:
        df[col] = df[col].replace(to_replace=99999999, value=0)

    df_train, _ = train_test_split(df, test_size=0.2, random_state=1)
    df_train = df_train.reset_index(drop=True)
    y_train = (df_train.status == 'default').astype('int').values

    del df_train['status']

    # Train the DictVectorizer and transform the data
    logging.info("Training DictVectorizer...")
    train_dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    # Train the XGBoost Model
    logging.info("Training XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save the vectorizer and model
    logging.info(f"Saving model and vectorizer to {model_output_path}...")
    joblib.dump((dv, model), model_output_path)
    logging.info("Model training finished.")

if __name__ == "__main__":
    train_model()