import os
import requests

# Get the URL from the environment variable or use a default
url = os.getenv('PREDICTION_URL', 'http://localhost:9696/predict')

# A sample loan application
applicant = {
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
}

try:
    # Send the POST request and get the response
    response = requests.post(url, json=applicant)
    response.raise_for_status()  # Raise an exception for bad status codes
    result = response.json()

    # Display the results
    print("Credit Risk Prediction:")
    print(result)

except requests.exceptions.RequestException as e:
    print(f"Error connecting to the prediction service: {e}")
