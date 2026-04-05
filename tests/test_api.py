import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# --- Fixtures ---

@pytest.fixture
def high_risk_customer():
    """Month-to-month, short tenure, fibre, no add-ons — typically high churn risk."""
    return {
        "tenure": 3,
        "gender_male": 1,
        "is_senior": 0,
        "has_partner": 0,
        "has_dependents": 0,
        "contract_type": "Month-to-month",
        "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 85.0,
        "has_phone": 1,
        "multiple_lines": 0,
        "internet_service": "Fiber optic",
        "has_online_security": 0,
        "has_tech_support": 0,
        "has_online_backup": 0,
        "has_device_protection": 0,
        "has_streaming_tv": 0,
        "has_streaming_movies": 0,
        "bundle_depth": 1
    }

@pytest.fixture
def low_risk_customer():
    """Two-year contract, long tenure, DSL, many add-ons — typically low churn risk."""
    return {
        "tenure": 60,
        "gender_male": 0,
        "is_senior": 0,
        "has_partner": 1,
        "has_dependents": 1,
        "contract_type": "Two year",
        "paperless_billing": 0,
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 55.0,
        "has_phone": 1,
        "multiple_lines": 1,
        "internet_service": "DSL",
        "has_online_security": 1,
        "has_tech_support": 1,
        "has_online_backup": 1,
        "has_device_protection": 1,
        "has_streaming_tv": 1,
        "has_streaming_movies": 1,
        "bundle_depth": 7
    }

@pytest.fixture
def day0_customer(high_risk_customer):
    """tenure=0 triggers the day0 segment with its own thresholds."""
    return {**high_risk_customer, "tenure": 0}


# --- /health ---

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_response_shape():
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_version" in body
    assert "metrics" in body

def test_health_status_is_healthy():
    response = client.get("/health")
    assert response.json()["status"] == "healthy"

def test_health_model_version():
    response = client.get("/health")
    assert response.json()["model_version"] == "v1.0"


# --- /predict: response shape & types ---

def test_predict_returns_200(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    assert response.status_code == 200

def test_predict_response_has_required_fields(high_risk_customer):
    body = client.post("/predict", json=high_risk_customer).json()
    assert "churn_propensity_score" in body
    assert "bucket" in body
    assert "tenure_segment" in body
    assert "model_version" in body

def test_predict_score_is_valid_probability(high_risk_customer):
    body = client.post("/predict", json=high_risk_customer).json()
    score = body["churn_propensity_score"]
    assert 0.0 <= score <= 1.0

def test_predict_bucket_is_valid(high_risk_customer):
    body = client.post("/predict", json=high_risk_customer).json()
    assert body["bucket"] in {"low", "medium", "high", "critical"}

def test_predict_model_version(high_risk_customer):
    body = client.post("/predict", json=high_risk_customer).json()
    assert body["model_version"] == "v1.0"


# --- /predict: tenure segments ---

def test_predict_day0_segment(day0_customer):
    body = client.post("/predict", json=day0_customer).json()
    assert body["tenure_segment"] == "day0"

def test_predict_new_segment(high_risk_customer):
    # tenure=3 is <=12, so "new"
    body = client.post("/predict", json=high_risk_customer).json()
    assert body["tenure_segment"] == "new"

def test_predict_old_segment(low_risk_customer):
    # tenure=60 is >12, so "old"
    body = client.post("/predict", json=low_risk_customer).json()
    assert body["tenure_segment"] == "old"

def test_predict_tenure_12_is_new(high_risk_customer):
    customer = {**high_risk_customer, "tenure": 12}
    body = client.post("/predict", json=customer).json()
    assert body["tenure_segment"] == "new"

def test_predict_tenure_13_is_old(high_risk_customer):
    customer = {**high_risk_customer, "tenure": 13}
    body = client.post("/predict", json=customer).json()
    assert body["tenure_segment"] == "old"


# --- /predict: directional sanity ---

def test_high_risk_scores_above_low_risk(high_risk_customer, low_risk_customer):
    high_score = client.post("/predict", json=high_risk_customer).json()["churn_propensity_score"]
    low_score = client.post("/predict", json=low_risk_customer).json()["churn_propensity_score"]
    assert high_score > low_score


# --- /predict: validation errors ---

def test_predict_missing_field_returns_422(high_risk_customer):
    del high_risk_customer["tenure"]
    response = client.post("/predict", json=high_risk_customer)
    assert response.status_code == 422

def test_predict_wrong_type_returns_422(high_risk_customer):
    high_risk_customer["monthly_charges"] = "not-a-number"
    response = client.post("/predict", json=high_risk_customer)
    assert response.status_code == 422

def test_predict_empty_body_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422
