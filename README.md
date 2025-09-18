# Credit Risk Prediction API: A Production-Ready Machine Learning Service

This project provides a complete, end-to-end solution for predicting credit default risk. It features a machine learning model trained on a credit scoring dataset, served via a REST API, and supported by a robust MLOps pipeline for continuous integration, deployment, and monitoring.

## Key Features

  - **Automated Model Training & Experimentation:** The training pipeline, orchestrated with Hydra, automates data preprocessing, feature engineering, and hyperparameter tuning for multiple models, including Logistic Regression, Random Forest, and XGBoost.
  - **RESTful API for Predictions:** A Flask-based API serves predictions and includes interactive documentation via Swagger UI, making it easy to integrate with other services.
  - **CI/CD Automation:** GitHub Actions are used to automate testing, code quality checks, Docker image builds, and deployments to Google Cloud Run, ensuring a streamlined and reliable release process.
  - **Scheduled Model Retraining:** A weekly GitHub Actions workflow automatically retrains the model to ensure it remains up-to-date with the latest data patterns.
  - **Data Drift Monitoring:** The project includes a monitoring script that detects data drift by comparing incoming data distributions against a training data profile, helping to identify when model retraining is necessary.
  - **Containerized for Portability:** The entire application is containerized using a multi-stage Dockerfile, making it easy to deploy and run in any environment.

## System Architecture

The project follows a modern MLOps architecture designed for scalability and maintainability:

1.  **Data Ingestion & Cleaning:** Raw data is loaded and cleaned to prepare it for feature engineering and model training.
2.  **Training Pipeline:** The `train.py` script, configured with Hydra, runs experiments to identify the best-performing model. The final artifact, including the model, vectorizer, and feature selector, is saved to the `artifacts/` directory.
3.  **Prediction Service:** The Flask API (`predict.py`) loads the trained model artifact and exposes a `/predict` endpoint to serve real-time predictions.
4.  **CI/CD Workflow:** On every push to the `main` branch, the CI/CD pipeline in `ci.yml` is triggered. It runs tests, builds a new Docker image, and deploys it to staging and production environments on Google Cloud Run.
5.  **Scheduled Retraining:** The `retrain.yml` workflow runs on a weekly schedule, executing the training pipeline to generate a new model artifact.

## API Documentation

The API provides endpoints for health checks, versioning, and predictions.

  - **Interactive Docs (Swagger UI):** Once the API is running, you can access the interactive documentation at `http://localhost:9696/docs`.
  - **OpenAPI Spec:** The raw OpenAPI 3.0 specification is available at `http://localhost:9696/openapi.json`.

### Prediction Endpoint

  - **URL:** `/predict`

  - **Method:** `POST`

  - **Description:** Predicts the probability of credit default for a given loan application.

  - **Example `curl` Request:**

    ```bash
    curl -X POST http://localhost:9696/predict \
      -H "Content-Type: application/json" \
      -d '{
        "seniority": 3, "home": "owner", "time": 60, "age": 30, "marital": "married",
        "records": "no", "job": "freelance", "expenses": 73, "income": 129,
        "assets": 0, "debt": 0, "amount": 800, "price": 1000
      }'
    ```

## Getting Started

### Prerequisites

  - Python 3.12
  - Docker

### Option 1: Run with Docker (Recommended)

This is the simplest way to run the application, as Docker handles all dependencies.

```bash
# 1. Build the Docker image
docker build -t credit-risk-api .

# 2. Run the container
docker run -p 9696:9696 credit-risk-api
```

The API will be available at `http://localhost:9696`.

### Option 2: Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Raimundoivy/decision-trees-ensemble-learning.git
    cd decision-trees-ensemble-learning
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements-dev.txt
    ```
3.  **Set up pre-commit hooks (optional but recommended):**
    ```bash
    pre-commit install
    ```

## Local Usage

### 1\. Train the Model

To run the training pipeline and generate a new model artifact, use the following command:

```bash
python -m src.credit_risk.train
```

### 2\. Run the API Locally

Start the Flask application using Gunicorn:

```bash
gunicorn --bind=0.0.0.0:9696 src.credit_risk.predict:app
```

### 3\. Run the Monitoring Script

To check for data drift, run the monitoring script:

```bash
python scripts/monitor.py
```

## Testing

The project includes both unit and integration tests.

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m "not integration"
```

## CI/CD Pipeline

The CI/CD pipeline, defined in `.github/workflows/ci.yml`, consists of the following jobs:

1.  **`test`:** Installs dependencies, runs linters (`black`, `flake8`), and executes unit tests.
2.  **`build-and-push`:** Builds and pushes a Docker image to GitHub Container Registry.
3.  **`integration`:** Runs integration tests against the newly built Docker container.
4.  **`deploy`:** Deploys the container to Google Cloud Run environments (dev, staging, prod).

## Future Improvements

  - **Advanced Monitoring:** Implement more sophisticated monitoring for concept drift and model performance degradation.
  - **Experiment Tracking:** Integrate an experiment tracking tool like MLflow or Weights & Biases to log and compare model training runs.
  - **Data Validation:** Add a dedicated data validation step to the pipeline using a library like Great Expectations.
