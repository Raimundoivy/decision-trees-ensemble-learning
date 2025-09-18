from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings are loaded from environment variables.
    A .env file can be used for local development.
    """
    # Define the location of the .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- Prediction Service Settings ---
    # The default threshold for classifying a loan as 'default'
    PREDICTION_THRESHOLD: float = 0.5

    # --- Model Artifact Settings ---
    # Directory where artifacts (models, profiles) are stored
    ARTIFACTS_DIR: str = "artifacts"
    # The name of the primary model file to be used by the prediction service
    MODEL_FILE: str = "model.joblib"

# Create a single instance of the settings to be used throughout the application
settings = Settings()