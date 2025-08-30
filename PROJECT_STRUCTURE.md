# Project Structure Overview

This document provides a developer-focused explanation of each file and directory in the project. It's a reference for understanding the purpose and function of each component.

## Core Application Files

These files are central to the machine learning application's functionality.

- **`train.py`**: 
  - **Purpose**: This script handles the entire model training process.
  - **Function**: It loads the raw data (`CreditScoring.csv`), performs feature engineering, runs a `GridSearchCV` to find the best hyperparameters for a Decision Tree classifier, and then saves the final trained model and the data vectorizer to `model.joblib`.

- **`predict.py`**:
  - **Purpose**: This is the main application file that runs the prediction API.
  - **Function**: It uses Flask to create a web server. It loads the `model.joblib` artifact and exposes several endpoints:
    - `/`: A welcome message.
    - `/ping`: A simple health check.
    - `/version`: Provides metadata about the running service (model hash, git SHA).
    - `/predict`: The core endpoint that accepts loan application data (JSON), validates it using Pydantic, and returns a credit default prediction.

- **`Dockerfile`**:
  - **Purpose**: To containerize the application.
  - **Function**: It defines a multi-stage build process to create a Docker image. It copies the application code, installs dependencies from `requirements.txt` (which is generated from `Pipfile`), and specifies the command to run the Gunicorn server, making the application portable and easy to deploy.

- **`Pipfile` & `Pipfile.lock`**:
  - **Purpose**: Python dependency management.
  - **Function**: `Pipfile` declares the packages needed for the project (e.g., `flask`, `scikit-learn`). `Pipfile.lock` records the exact versions of all dependencies, ensuring that the environment is 100% reproducible across different machines.

## Testing

These files ensure the application is working correctly.

- **`test_*.py`** (e.g., `test_predict.py`, `test_train.py`):
  - **Purpose**: Automated unit tests.
  - **Function**: These files contain tests written for the `pytest` framework. They should verify the logic of individual components in isolation. For example, `test_predict.py` would test the logic of the prediction endpoint without making a real HTTP request.

- **`predict-test.py`**:
  - **Purpose**: A simple integration test script.
  - **Function**: This script sends a real HTTP request to a running instance of the API and checks if the response is valid. It's used in the `ci.yml` pipeline to test the actual Docker container.

- **`pytest.ini`**:
  - **Purpose**: Configuration for `pytest`.
  - **Function**: It allows customization of the test runner, such as defining test markers (`-m`) to selectively run or skip certain tests (e.g., separating `unit` from `integration` tests).

## CI/CD (Continuous Integration/Continuous Deployment)

These files automate the project's lifecycle.

- **`.github/workflows/ci.yml`**:
  - **Purpose**: The main CI/CD pipeline.
  - **Function**: This GitHub Actions workflow automates everything. On a push to `main`, it runs linters and tests, builds the Docker image, pushes it to GitHub Container Registry, runs integration tests against the live container, and finally deploys the new version to Google Cloud Run.

- **`.github/workflows/retrain.yml`**:
  - **Purpose**: Scheduled model retraining.
  - **Function**: This workflow runs automatically on a schedule (e.g., weekly). It checks out the code, installs dependencies, and runs the `train.py` script to create a new `model.joblib` artifact. This new model can then be manually or automatically promoted to production.

## Documentation and Configuration

- **`README.md`**:
  - **Purpose**: The main entry point for project documentation.
  - **Function**: Provides a general overview and instructions for users and developers on how to set up, run, and use the project.

- **`ROADMAP.md`**:
  - **Purpose**: High-level project planning.
  - **Function**: A place for your notes, ideas, and future development checklist.

- **`openapi.json`**:
  - **Purpose**: API Specification.
  - **Function**: A machine-readable JSON file that formally defines your API's endpoints, request/response schemas, etc. It's used by `predict.py` to serve the interactive Swagger UI documentation.

- **`.gitignore`**:
  - **Purpose**: To exclude files from Git tracking.
  - **Function**: Specifies files and directories that Git should ignore. This is critical for keeping the repository clean of temporary files, local configurations (`.idea`), and large data/model files (`CreditScoring.csv`, `model.joblib`).

- **`.pre-commit-config.yaml`**:
  - **Purpose**: To enforce code quality locally.
  - **Function**: Configures pre-commit hooks that automatically run tools like `black` (formatter) and `flake8` (linter) on your code before you make a commit. This helps maintain a consistent code style.

- **`setup.cfg`**:
  - **Purpose**: Configuration for development tools.
  - **Function**: A standard file for configuring tools. In this project, it's likely used to set rules for `flake8`.

## Scripts and Other Directories

- **`scripts/generate_ssh_key.sh`**:
  - **Purpose**: A utility script.
  - **Function**: A simple shell script to generate SSH keys without interactive prompts, which is useful for automation.
