import requests

# The URL of your local server's predict endpoint
url = 'http://localhost:9696/predict'

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

# Send the POST request and get the response
response = requests.post(url, json=applicant)
result = response.json()

# Display the results
print("Credit Risk Prediction:")
print(result)