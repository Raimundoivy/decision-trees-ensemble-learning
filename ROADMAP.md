# Project Roadmap — What to do now

This roadmap explains what you can (and should) do next with the Credit Risk Prediction project. It covers local development, testing, model lifecycle, CI/CD and deployment, and a prioritized next-actions checklist.

## 1) Quickstart for developers

- Install dependencies (Pipenv):
  - pip install pipenv
  - pipenv install --dev
- Run tests (unit only):
  - pipenv run pytest -m "not integration" -q
- Lint and format checks (used in CI):
  - pipenv run black --check .
  - pipenv run flake8 .
- Run the API locally (dev server):
  - PREDICTION_THRESHOLD=0.5 pipenv run python predict.py
  - Open http://localhost:9696/ping → {"status": "ok"}

Endpoints:
- GET / → Welcome message
- GET /ping → Health check
- GET /version → Service metadata including model file and threshold
- POST /predict → JSON body with applicant fields; returns default_probability and default

Environment variables:
- PREDICTION_THRESHOLD (float, default 0.5) — decision boundary for classification

## 2) Running in Docker locally

- Ensure requirements.txt exists (CI generates it; locally you can run: pipenv requirements > requirements.txt)
- Build: docker build -t credit-risk:local .
- Run: docker run --rm -p 9696:9696 credit-risk:local
- Health: curl http://localhost:9696/ping
- Predict: python predict-test.py (uses PREDICTION_URL env or defaults to http://localhost:9696/predict)

Notes:
- Container runs as non-root and exposes port 9696; healthcheck is built-in.

## 3) Model lifecycle (training → serving)

- Training entrypoint: train.py
  - Reads CreditScoring.csv
  - Performs feature engineering (ratios + polynomial features), then trains DecisionTree with GridSearchCV
  - Outputs (dv, model) to model.joblib
- Re-train steps:
  1. Update/validate dataset (CreditScoring.csv)
  2. pipenv run python train.py
  3. Commit the updated model.joblib if it improves metrics and remains compatible with predict.py
- Serving expects:
  - model.joblib containing a tuple (DictVectorizer, classifier) — predict.py loads it at startup
  - Input schema validated via Pydantic (LoanApplication in predict.py)

Tips for reproducibility:
- Keep Pipfile.lock up-to-date
- Record hyperparameters and metrics from GridSearchCV outputs (console logs)

## 4) CI/CD and deployments

- GitHub Actions workflow (ci.yml) has four jobs:
  1) test: lint + unit tests (3.13)
  2) build-and-push: builds Docker and pushes to ghcr.io on master
  3) integration: runs container and executes predict-test.py against it
  4) deploy: deploys to Google Cloud Run using secrets

Secrets needed for deploy:
- GCP_SA_KEY — JSON for a GCP service account with Cloud Run deploy permissions
- GCP_PROJECT_ID — target project id

Cloud Run notes:
- Service name: credit-risk-service; region: us-central1
- Env var PREDICTION_THRESHOLD can be configured from the workflow

Elastic Beanstalk (AWS):
- See README.md for EB CLI steps to initialize, create env, open, and terminate.

Non-interactive SSH key generation (automation/CI):
- Use scripts/generate_ssh_key.sh to avoid interactive prompts like "Enter file in which to save the key"
  - Example: ./scripts/generate_ssh_key.sh idk

## 5) Reliability, security, and quality improvements (backlog)

- Observability
  - Add structured logging (JSON) and request IDs
  - Expose metrics (Prometheus) and basic tracing
- API quality
  - Add OpenAPI/Swagger docs and a /docs endpoint
  - Expand Pydantic validation with enums/ranges; better error messages
- Testing
  - Add contract tests for /predict schema
  - Add model regression test (e.g., ensure AUC not below a floor)
- Security
  - Ensure secrets are only from CI/Secrets; no debug mode in prod
  - Review dependency vulnerabilities; pin versions where needed
- Delivery
  - Add pre-commit hooks (black, flake8, isort)
  - Multi-env deploys (dev/staging/prod) with manual approvals
- Model ops
  - Version model artifacts; include model hash in /version
  - Store training metrics as artifacts in CI

## 6) Concrete next actions (suggested order)

1. Add OpenAPI spec and basic docs for the API; publish via /docs.
2. Introduce pre-commit hooks (black, flake8), and enable isort.
3. Add a simple model regression test (e.g., validate predict_proba bounds and a basic sanity dataset).
4. Parameterize PREDICTION_THRESHOLD per environment in CI (dev/staging/prod matrix).
5. Add structured logging and propagate a request_id header through predictions.
6. Pin key dependencies (Flask, scikit-learn, pydantic) and address any warnings.
7. Store and version model artifacts; expose model version/hash in /version.
8. Optional: Add scheduled retraining workflow if data updates are expected.
