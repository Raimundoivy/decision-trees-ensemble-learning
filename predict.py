import joblib
from flask import Flask, request, jsonify, send_from_directory, make_response
import logging
import os
import hashlib
import uuid
from pydantic import BaseModel, ValidationError

# Name of the model file
model_file = "model.joblib"

# Get prediction threshold from environment variable, with a default of 0.5
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.5))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"Loading model from {model_file}...")
dv, model = joblib.load(model_file)
# Compute a hash of the model artifact for versioning/traceability
try:
    with open(model_file, "rb") as f:
        _data = f.read()
        MODEL_HASH = hashlib.sha256(_data).hexdigest()
except Exception:
    MODEL_HASH = "unknown"
logging.info("Model loaded successfully.")

app = Flask("credit-risk")
app.title = "Credit Risk Prediction API"


# Define the data model for input validation using Pydantic
class LoanApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
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
        "model_file": model_file,
        "threshold": THRESHOLD,
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
    application_dict = application.model_dump()

    logging.info(f"[{request_id}] Received prediction request: {application_dict}")
    # Use the loaded DictVectorizer to convert the application data into a feature matrix
    X = dv.transform([application_dict])

    # Use the loaded model to predict the probability of default
    # The notebook uses predict_proba to get scores for the AUC calculation
    y_pred = model.predict_proba(X)[0, 1]

    # Set a decision threshold
    default = y_pred >= THRESHOLD

    # Prepare the JSON response
    result = {"default_probability": float(y_pred), "default": bool(default), "request_id": request_id}

    logging.info(f"[{request_id}] Prediction result: {result}")

    # Return the results as a JSON response with request id header
    resp = jsonify(result)
    resp.headers["X-Request-ID"] = request_id
    return resp


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)