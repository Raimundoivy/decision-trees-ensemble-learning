# Credit Risk Prediction API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com/)

A containerized Flask API that predicts customer credit default risk using a pre-trained XGBoost model.

## Core Problem & Solution

**Problem:** Financial institutions require a fast, consistent, and scalable method to assess the creditworthiness of loan applicants to minimize risk.

**Solution:** This project provides a lightweight, production-ready microservice that exposes a simple API endpoint to score applicants, making it easy to integrate into larger banking applications.

## Key Features

-   📈 **Risk Prediction:** Accepts applicant data via a JSON payload and returns a default probability score.
-   🐳 **Dockerized:** Packaged with Docker and Gunicorn for consistent, production-grade deployments.
-   ⚡ **Fast API:** Built with Flask for a lightweight and responsive web service.
-   🧪 **Tested & Integrated:** Includes a `pytest` suite and a GitHub Actions CI pipeline to ensure reliability and correctness.
-   🔄 **Reproducible:** Comes with a training script (`train.py`) to regenerate the model from source data.
-   ☁️ **Cloud-Ready:** Includes a deployment guide for AWS Elastic Beanstalk.

## Technology Stack

-   **Backend:** Python, Flask, Gunicorn
-   **Machine Learning:** Scikit-Learn, XGBoost, Pandas
-   **Containerization:** Docker
-   **CI/CD:** GitHub Actions
-   **Testing:** Pytest, Requests

## Getting Started

This guide provides instructions for running the application using Docker (recommended for production/staging) or locally with a Python virtual environment (ideal for development).

### Prerequisites

-   [Git](https://git-scm.com/)
-   [Docker](https://www.docker.com/products/docker-desktop/)
-   [Python 3.12+](https://www.python.org/downloads/)

### Option 1: Run with Docker (Recommended)

This is the simplest and most reliable way to run the application, as it encapsulates all dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t credit-risk-api .
    ```

3.  **Run the Docker container:**
    ```bash
    docker run -p 9696:9696 credit-risk-api
    ```

The API will now be available at `http://localhost:9696`.

### Option 2: Run Locally for Development

This method is suitable for actively developing the application code.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(Note: This project uses Pipfile. A `requirements.txt` is also included for convenience.)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python predict.py
    ```

The API will now be available in debug mode at `http://localhost:9696`.

## API Usage

You can interact with the API by sending `POST` requests to the `/predict` endpoint.

-   **Endpoint:** `POST /predict`
-   **Description:** Predicts the probability of credit default based on applicant data.
-   **Request Body:** A JSON object containing applicant details.
-   **Success Response:** `200 OK` with a JSON object containing the `default_probability` and a boolean `default` decision.
-   **Error Response:** `400 Bad Request` if the input data is invalid or malformed.

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
           "job": "freance",
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
  "default": false,
  "default_probability": 0.13998721539974213
}
```

## Running Tests

To ensure the application is working correctly, you can run the test suite using `pytest`.

1.  **Install development dependencies:**

    ```bash
    pip install pytest requests
    ```

2.  **Run the tests:**

    ```bash
    pytest
    ```

## Model Training

The model can be retrained using the `train.py` script. The script expects the dataset `CreditScoring.csv` to be in the root directory.

1.  **Ensure all dependencies are installed:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the training script:**

    ```bash
    python train.py
    ```

    This will generate a new `model.joblib` file in the project root.

## Model Development Notebook

The Jupyter notebook `decision_trees_ensemble_learning.ipynb` contains the original data exploration, feature engineering, and model training process. For a deeper understanding of how the model was developed and evaluated, please refer to this notebook.

## Deployment to AWS Elastic Beanstalk

The original deployment instructions are provided below.

### Prerequisites

  - An AWS Account.
  - [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) installed and configured.
  - [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html) installed.

### Deployment Steps

1.  **Initialize the Elastic Beanstalk Application:**
    Navigate to the project's root directory and run:

    ```bash
    eb init -p "Docker" --region us-east-1 credit-risk-app
    ```

    This command initializes a new EB application named `credit-risk-app` configured for the Docker platform.

2.  **Create the Environment:**
    Create a new environment for the application.

    ```bash
    eb create credit-risk-env
    ```

    This command deploys your application. The process may take several minutes.

3.  **Verify the Deployment:**
    Once complete, open the application in your browser:

    ```bash
    eb open
    ```

    You should see the welcome message: `{"message":"Welcome to the Credit Risk Prediction API!"}`.

4.  **Clean Up:**
    To avoid incurring charges, terminate your environment when you are done.

    ```bash
    eb terminate credit-risk-env
    ```

```
```
