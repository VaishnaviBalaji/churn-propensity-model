from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Propensity API",
    description="Predicts customer churn probability and risk bucket",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vaishnavibalaji.github.io"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# Load model and metadata on startup
MODEL_DIR = Path(__file__).parent.parent / "models"

def load_model():
    model_path = MODEL_DIR / "churn_model_v1.ubj"
    metadata_path = MODEL_DIR / "churn_model_v1_metadata.json"

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model loaded: {metadata['model_version']}")
    logger.info(f"Metrics: {metadata['metrics']}")
    return model, metadata

model, metadata = load_model()

# Input schema — mirrors feature store columns
class CustomerFeatures(BaseModel):
    tenure: int
    gender_male: int
    is_senior: int
    has_partner: int
    has_dependents: int
    contract_type: str
    paperless_billing: int
    payment_method: str
    monthly_charges: float
    has_phone: int
    multiple_lines: int
    internet_service: str
    has_online_security: int
    has_tech_support: int
    has_online_backup: int
    has_device_protection: int
    has_streaming_tv: int
    has_streaming_movies: int
    bundle_depth: int

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 5,
                "gender_male": 1,
                "is_senior": 0,
                "has_partner": 0,
                "has_dependents": 0,
                "contract_type": "Month-to-month",
                "paperless_billing": 1,
                "payment_method": "Electronic check",
                "monthly_charges": 70.5,
                "has_phone": 1,
                "multiple_lines": 0,
                "internet_service": "Fiber optic",
                "has_online_security": 0,
                "has_tech_support": 0,
                "has_online_backup": 0,
                "has_device_protection": 0,
                "has_streaming_tv": 1,
                "has_streaming_movies": 0,
                "bundle_depth": 2
            }
        }

def get_tenure_segment(tenure: int) -> str:
    if tenure == 0:
        return "day0"
    elif tenure <= 12:
        return "new"
    return "old"

def get_bucket(score: float, tenure_segment: str) -> str:
    thresholds = metadata['bucket_thresholds'][tenure_segment]
    if score < thresholds[0]:
        return "low"
    elif score < thresholds[1]:
        return "medium"
    elif score < thresholds[2]:
        return "high"
    return "critical"

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": metadata['model_version'],
        "metrics": metadata['metrics']
    }

@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        # Build feature dataframe
        features = customer.model_dump()
        df = pd.DataFrame([features])
        
        # Apply categorical encoding
        for col in metadata['categorical_features']:
            df[col] = pd.Categorical(df[col])
        
        # Ensure correct feature order
        df = df[metadata['features']]
        
        # Score
        score = float(model.predict_proba(df)[0, 1])
        tenure_segment = get_tenure_segment(customer.tenure)
        bucket = get_bucket(score, tenure_segment)
        
        logger.info(
            f"Prediction: tenure={customer.tenure}, "
            f"segment={tenure_segment}, "
            f"score={score:.4f}, bucket={bucket}"
        )
        
        return {
            "churn_propensity_score": round(score, 4),
            "bucket": bucket,
            "tenure_segment": tenure_segment,
            "model_version": metadata['model_version']
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))