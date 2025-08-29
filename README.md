# Credit Risk Prediction API

[![CI](https://github.com/Raimundoivy/decision-trees-ensemble-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Raimundoivy/decision-trees-ensemble-learning/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com/)

A containerized Flask API that predicts customer credit default risk using a pre-trained XGBoost model.

## Core Problem & Solution

**Problem:** Financial institutions require a fast, consistent, and scalable method to assess the creditworthiness of loan applicants to minimize risk.

**Solution:** This project provides a lightweight, production-ready microservice that exposes a simple API endpoint to score applicants, making it easy to integrate into larger financial applications.

## Key Features

-   📈 **Risk Prediction API:** Accepts applicant data via a JSON payload and returns a default probability score and a final decision.
-   🐳 **Dockerized Environment:** Packaged with a multi-stage `Dockerfile` and Gunicorn for a lean, consistent, and production-grade deployment.
-   🛡️ **Input Validation:** Uses Pydantic to define and enforce a strict schema for incoming request data, preventing errors.
-   🔄 **Reproducible Training:** Includes a `train.py` script to retrain the XGBoost model from the source `CreditScoping.csv` dataset.
-   🧪 **Comprehensive Testing:** Features a full `pytest` suite for unit testing the API and a CI pipeline powered by GitHub Actions to automate testing.

## Visual Demo

[INSERT-DEMO-GIF-HERE]

*A brief animation demonstrating a `curl` request to the API and the corresponding JSON response provides immediate clarity on the project's core functionality.*

## Technology Stack

-   **Backend:** Python, Flask, Gunicorn
-   **Machine Learning:** Scikit-Learn, XGBoost, Pandas
-   **Data Validation:** Pydantic
-   **Containerization:** Docker
-   **CI/CD:** GitHub Actions
-   **Testing:** Pytest

## Getting Started

This guide provides instructions for running the application using Docker (recommended) or locally with a Python virtual environment.

### Prerequisites

-   Git
-   Docker
-   Python 3.12+

### Option 1: Run with Docker (Recommended)

This is the simplest and most reliable way to run the application.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Raimundoivy/decision-trees-ensemble-learning.git](https://github.com/Raimundoivy/decision-trees-ensemble-learning.git)
    cd decision-trees-ensemble-learning
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t credit-risk-api .
    ```

3.  **Run the Docker container:**
    The application will be exposed on port `9696`.
    ```bash
    docker run -p 9696:9696 credit-risk-api
    ```

The API is now available at `http://localhost:9696`.

### Option 2: Run Locally for Development

This method is suitable for actively developing the application code.

1.  **Clone the repository and navigate into it.**

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `Dockerfile` uses `requirements.txt`, which is ideal for reproducing the production environment.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    The application runs in debug mode on port `9696` when executed directly.
    ```bash
    python predict.py
    ```

## API Usage

Interact with the service by sending `POST` requests to the `/predict` endpoint.

-   **Endpoint:** `POST /predict`
-   **Description:** Predicts the probability of credit default.
-   **Request Body:** A JSON object with applicant details.
-   **Success Response:** A `200 OK` with a JSON object containing the `default_probability` and a boolean `default` decision.
-   **Error Response:** A `400 Bad Request` if the input data is invalid or malformed.

### Example Request

You can use the provided `predict-test.py` script or a tool like `curl`.

**Using `curl`:**
```bash
curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
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
           "price": 846
         }'
````

**Example Success Response:**

```json
{
  "default_probability": 0.13998721539974213,
  "default": false
}
```

## Project Structure

A brief overview of the key files in this repository:

  - `predict.py`: The Flask application that serves the model.
  - `train.py`: Script to train the model from source data.
  - `model.joblib`: The pre-trained and serialized model file.
  - `Dockerfile`: Defines the container for deployment.
  - `test_predict.py`: The `pytest` test suite for the API.
  - `.github/workflows/ci.yml`: The GitHub Actions CI configuration.
  - `decision_trees_ensemble_learning.ipynb`: Jupyter notebook with original model development and analysis.

## Running Tests

To ensure the application is working correctly, run the test suite using `pytest`.

1.  **Install development dependencies:**

    ```bash
    pip install pytest requests
    ```

2.  **Run the tests:**

    ```bash
    pytest
    ```

## Model Training

To retrain the model, simply run the `train.py` script. It will overwrite the existing `model.joblib` file.

```bash
python train.py
```
