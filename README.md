# Credit Risk Prediction API

**A machine learning project to predict credit default probability, served via a REST API.**

This project trains a Decision Tree classifier on a credit scoring dataset to predict the likelihood of a loan default. The trained model is exposed through a Flask-based REST API, which is containerized using Docker. The entire workflow is supported by a robust CI/CD pipeline using GitHub Actions for testing, building, and deploying the application to Google Cloud Run.

## Features

- **Automated Model Training**: A script (`train.py`) for training the model, including feature engineering and hyperparameter tuning.
- **Prediction API**: A Flask API (`predict.py`) to serve predictions.
- **API Documentation**: Interactive API documentation available via Swagger UI.
- **Containerization**: A `Dockerfile` to package the application for consistent deployment.
- **CI/CD Pipeline**: GitHub Actions (`ci.yml`) for automated testing, linting, building, and deployment.
- **Scheduled Retraining**: A GitHub Actions workflow (`retrain.yml`) to automatically retrain the model weekly.
- **Dependency Management**: Uses `pipenv` for managing Python dependencies.
- **Code Quality**: Pre-commit hooks for automated code formatting and linting.

## API Documentation

The API provides endpoints for health checks, versioning, and predictions.

- **Interactive Docs (Swagger UI)**: Once the API is running, visit `http://localhost:9696/docs`.
- **OpenAPI Spec**: The raw OpenAPI 3.0 specification is available at `http://localhost:9696/openapi.json`.

### Prediction Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Predicts the probability of credit default for a given loan application. See the `LoanApplication` schema in the `openapi.json` file for the full structure.
- **Example `curl` Request**:
  ```bash
  curl -X POST http://localhost:9696/predict \
    -H "Content-Type: application/json" \
    -d \
    '{
      "seniority": 3, "home": "owner", "time": 60, "age": 30, "marital": "married",
      "records": "no", "job": "freelance", "expenses": 73, "income": 129,
      "assets": 0, "debt": 0, "amount": 800, "price": 1000
    }'
  ```

## Getting Started

### Prerequisites

- Python 3.13
- `pipenv`
- Docker

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Raimundoivy/decision-trees-ensemble-learning.git
    cd decision-trees-ensemble-learning
    ```

2.  **Install dependencies:**
    ```bash
    pipenv install --dev
    ```

3.  **Set up pre-commit hooks (optional but recommended):**
    ```bash
    pipenv run pre-commit install
    ```

## Usage

### 1. Train the Model

Run the training script. This will generate the `model.joblib` file.
*Note: This requires the `CreditScoring.csv` dataset in the root directory.*

```bash
pipenv run python train.py
```

### 2. Run the API Locally

Start the Flask application:

```bash
pipenv run python predict.py
```

The API will be available at `http://localhost:9696`.

### 3. Run with Docker

Build and run the Docker container:

```bash
docker build -t credit-risk-api .
docker run -p 9696:9696 credit-risk-api
```

## Testing

Run the test suite using `pytest`. The tests are separated into unit and integration tests.

```bash
# Run all tests
pipenv run pytest

# Run only unit tests
pipenv run pytest -m "not integration"
```

## CI/CD Pipeline

The project uses GitHub Actions for its CI/CD pipeline, defined in `.github/workflows/ci.yml`. The pipeline consists of the following jobs:

1.  **`test`**: Installs dependencies, runs linters (`black`, `flake8`), and executes unit tests on every push and pull request to `main`.
2.  **`build-and-push`**: Builds and pushes a Docker image to GitHub Container Registry (ghcr.io) on pushes to `main`.
3.  **`integration`**: Runs integration tests by spinning up the newly built Docker container and sending test predictions to it.
4.  **`deploy`**: Deploys the container to Google Cloud Run environments (dev, staging, prod) after all previous jobs succeed.

Additionally, the `.github/workflows/retrain.yml` workflow automatically retrains the model every Monday at 03:00 UTC.
