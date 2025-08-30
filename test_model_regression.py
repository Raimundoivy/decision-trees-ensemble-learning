import joblib

# Sample consistent with API tests
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
    "price": 846,
}


def test_model_predict_proba_bounds():
    dv, model = joblib.load("model.joblib")
    X = dv.transform([APPLICANT_DATA])
    proba = float(model.predict_proba(X)[0, 1])
    assert 0.0 <= proba <= 1.0
