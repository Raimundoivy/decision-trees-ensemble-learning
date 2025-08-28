import pytest
from predict import app as flask_app

# The sample applicant data from predict-test.py
APPLICANT_DATA = {
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

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    # Set the app to testing mode
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_ping_endpoint(client):
    """Test the /ping health check endpoint."""
    response = client.get('/ping')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_predict_endpoint_success(client):
    """Test a successful prediction from the /predict endpoint."""
    response = client.post('/predict', json=APPLICANT_DATA)
    assert response.status_code == 200
    result = response.json
    assert 'default_probability' in result
    assert 'default' in result
    assert isinstance(result['default_probability'], float)
    assert isinstance(result['default'], bool)

def test_predict_endpoint_invalid_data(client):
    """Test the /predict endpoint with missing data to ensure it returns a 400 error."""
    invalid_data = APPLICANT_DATA.copy()
    del invalid_data['seniority']  # Remove a required field

    response = client.post('/predict', json=invalid_data)
    assert response.status_code == 400
    result = response.json
    assert 'error' in result
    assert result['error'] == 'Invalid input'