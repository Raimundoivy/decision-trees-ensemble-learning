import joblib
from flask import Flask, request, jsonify, send_from_directory, make_response
import logging
import pandas as pd
import os
import sys
import hashlib
import uuid
import numpy as np
from pydantic import BaseModel, ValidationError
from enum import Enum
from .train import feature_engineering 
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_FILE_PATH = os.path.join(settings.ARTIFACTS_DIR, settings.MODEL_FILE)

logging.info(f"Loading artifact from {MODEL_FILE_PATH}...")
try:
    artifact = joblib.load(MODEL_FILE_PATH) 
    dv = artifact["vectorizer"]
    model = artifact["model"]
    selector = artifact["feature_selector"]
    explainer = artifact.get("explainer")
    selected_feature_names = artifact.get("selected_feature_names")
except (FileNotFoundError, KeyError) as e:
    logging.error(f"Model file not found at {MODEL_FILE_PATH} or is invalid. Please run the training script first.")
    logging.error(f"Error loading artifact: {e}")
    dv, model, selector, explainer, selected_feature_names = None, None, None, None, None

if not all([dv, model, selector]):
    logging.critical("Essential model components (dv, model, selector) could not be loaded. Exiting.")
    sys.exit(1)
# Compute a hash of the model artifact for versioning/traceability
try:
    with open(MODEL_FILE_PATH, "rb") as f:
        _data = f.read()
        MODEL_HASH = hashlib.sha256(_data).hexdigest()
except Exception:
    MODEL_HASH = "unknown"
logging.info("Model loaded successfully.")

app = Flask("credit-risk")
app.title = "Credit Risk Prediction API"


# Define Enums for categorical features
class HomeEnum(str, Enum):
    rent = 'rent'
    owner = 'owner'
    private = 'private'
    ignore = 'ignore'
    parents = 'parents'
    other = 'other'
    unk = 'unk'

class MaritalStatusEnum(str, Enum):
    single = 'single'
    married = 'married'
    widow = 'widow'
    separated = 'separated'
    divorced = 'divorced'
    unk = 'unk'

class RecordsStatusEnum(str, Enum):
    no = 'no'
    yes = 'yes'
    unk = 'unk'

class JobStatusEnum(str, Enum):
    fixed = 'fixed'
    parttime = 'parttime'
    freelance = 'freelance'
    others = 'others'
    unk = 'unk'


# Define the data model for input validation using Pydantic
class LoanApplication(BaseModel):
    seniority: int
    home: HomeEnum
    time: int
    age: int
    marital: MaritalStatusEnum
    records: RecordsStatusEnum
    job: JobStatusEnum
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


@app.route("/version", methods=["GET"])
def version():
    return jsonify({
        "service": "credit-risk",
        "git_sha": os.getenv("GIT_SHA", "dev"),
        "model_file": settings.MODEL_FILE,
        "threshold": settings.PREDICTION_THRESHOLD,
        "model_hash": MODEL_HASH,
    })


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with a welcome message."""
    return jsonify({"message": "Welcome to the Credit Risk Prediction API!"})


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


# Serve the OpenAPI spec
@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    return send_from_directory(app.root_path, "openapi.json")


# Minimal Swagger UI page using CDN
@app.route("/docs", methods=["GET"])
def docs():
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset='utf-8' />
        <title>API Docs</title>
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
        <style>body {{ margin:0; }} #swagger-ui {{ height: 100vh; }}</style>
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
          window.onload = () => {{
            window.ui = SwaggerUIBundle({{
              url: '/openapi.json',
              dom_id: '#swagger-ui'
            }});
          }};
        </script>
      </body>
    </html>
    """
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html"
    return resp


@app.route("/predict", methods=["POST"])
def predict():
    """Predict credit risk based on loan application data."""
    # Correlation ID from header or generate a new one
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    try:
        # Get and validate the loan application data from the request's JSON body
        application_data = request.get_json()
        application = LoanApplication(**application_data)
    except ValidationError as e:
        logging.error(f"[{request_id}] Validation error: {e.errors()}")
        resp = jsonify({"error": "Invalid input", "details": e.errors()})
        resp.status_code = 400
        resp.headers["X-Request-ID"] = request_id
        return resp
    except Exception:
        # Catch other potential errors like a non-JSON body
        logging.error(f"[{request_id}] Request body is not valid JSON")
        resp = jsonify({
            "error": "Bad Request",
            "details": "Request body must be valid JSON",
        })
        resp.status_code = 400
        resp.headers["X-Request-ID"] = request_id
        return resp

    # Convert the Pydantic model to a dictionary for the vectorizer
    # .model_dump() will convert Enums to their string values
    application_dict = application.model_dump()

    logging.info(f"[{request_id}] Received prediction request: {application_dict}")

    # --- Full Prediction Pipeline ---
    # 1. Convert to DataFrame and apply feature engineering
    application_df = pd.DataFrame([application_dict])
    application_engineered_df = feature_engineering(application_df)
    application_engineered_dict = application_engineered_df.to_dict(orient='records')

    # 2. Vectorize the engineered features
    X_raw = dv.transform(application_engineered_dict)

    # 3. Apply feature selection
    X_selected = selector.transform(X_raw)

    # 4. Make the prediction
    y_pred = model.predict_proba(X_selected)[0, 1]

    # 5. Get SHAP explanations if the explainer was loaded
    explanation = {}
    if explainer and selected_feature_names:
        try:
            # For TreeExplainer, shap_values returns a list (for multi-class) or a single array.
            # We are interested in the SHAP values for the "default" class (class 1).
            shap_values_output = explainer.shap_values(X_selected)

            if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                # For binary classifiers, shap_values often returns a list of two arrays
                shap_values_for_prediction = shap_values_output[1][0]
            else:
                # For LinearExplainer or single-output TreeExplainer
                shap_values_for_prediction = shap_values_output[0]

            explanation = dict(zip(selected_feature_names, np.round(shap_values_for_prediction, 4)))
        except Exception as e:
            logging.warning(f"[{request_id}] Could not generate SHAP explanation: {e}")

    # Set a decision threshold
    default = y_pred >= settings.PREDICTION_THRESHOLD

    # Prepare the JSON response
    result = {
        "default_probability": float(y_pred),
        "default": bool(default),
        "explanation": explanation,
        "request_id": request_id,
    }

    logging.info(f"[{request_id}] Prediction result: {result}")

    # Return the results as a JSON response with request id header
    resp = jsonify(result)
    resp.headers["X-Request-ID"] = request_id
    return resp


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
