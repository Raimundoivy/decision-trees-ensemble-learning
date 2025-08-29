import joblib
from flask import Flask, request, jsonify
import logging
import os
from pydantic import BaseModel, ValidationError

# Name of the model file
model_file = 'model.joblib'

# Get prediction threshold from environment variable, with a default of 0.5
THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', 0.5))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Loading model from {model_file}...")
dv, model = joblib.load(model_file)
logging.info("Model loaded successfully.")

app = Flask('credit-risk')
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

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with a welcome message."""
    return jsonify({"message": "Welcome to the Credit Risk Prediction API!"})


@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict credit risk based on loan application data."""
    try:
        # Get and validate the loan application data from the request's JSON body
        application_data = request.get_json()
        application = LoanApplication(**application_data)
    except ValidationError as e:
        logging.error(f"Validation error: {e.errors()}")
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception:
        # Catch other potential errors like a non-JSON body
        logging.error("Request body is not valid JSON")
        return jsonify({"error": "Bad Request", "details": "Request body must be valid JSON"}), 400

    # Convert the Pydantic model to a dictionary for the vectorizer
    application_dict = application.model_dump()

    logging.info(f"Received prediction request: {application_dict}")
    # Use the loaded DictVectorizer to convert the application data into a feature matrix
    X = dv.transform([application_dict])

    # Use the loaded model to predict the probability of default
    # The notebook uses predict_proba to get scores for the AUC calculation
    y_pred = model.predict_proba(X)[0, 1]
    
    # Set a decision threshold
    default = y_pred >= THRESHOLD

    # Prepare the JSON response
    result = {
        'default_probability': float(y_pred),
        'default': bool(default)
    }
    
    logging.info(f"Prediction result: {result}")

    # Return the results as a JSON response
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)