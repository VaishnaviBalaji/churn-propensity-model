# Churn Propensity Model

End-to-end MLOps system for predicting telco customer churn — from feature store to live API.

**Live demo:** https://vaishnavibalaji.github.io/churn-propensity-model  
**API:** https://churn-api-844653534188.europe-west2.run.app/docs

---

## Stack

| Layer | Technology |
|---|---|
| Feature store | BigQuery (partitioned + clustered) |
| EDA + training | Jupyter notebooks |
| Model | XGBoost + MLflow experiment tracking |
| Serving | FastAPI + Docker + Cloud Run |
| Retraining | Prefect DAG + Optuna hyperparameter tuning |
| CI/CD | GitHub Actions |
| Frontend | GitHub Pages |

---

## Model performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.85 |
| PR-AUC | 0.68 |
| Churn recall | 0.79 |
| Top decile churn rate | 0.69 |

Trained on 7,043 telco customers (26.5% churn rate). Primary metric is PR-AUC due to class imbalance.

---

## Project structure

```
├── dags/
│   └── retrain.py          # Prefect retraining pipeline
├── docs/
│   └── index.html          # GitHub Pages frontend
├── notebooks/
│   ├── Eda_churn_propensity.ipynb
│   └── 02_modelling_churn.ipynb
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI app
│   └── models/
│       ├── churn_model_v1.ubj
│       └── churn_model_v1_metadata.json
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml              # CI pipeline
├── Dockerfile
└── requirements.txt
```

---

## API endpoints

### `GET /health`
Returns model status and metrics.

```bash
curl https://churn-api-844653534188.europe-west2.run.app/health
```

### `POST /predict`
Returns churn propensity score, risk bucket, and tenure segment.

```bash
curl -X POST https://churn-api-844653534188.europe-west2.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "churn_propensity_score": 0.7888,
  "bucket": "high",
  "tenure_segment": "new",
  "model_version": "v1.0"
}
```

Risk buckets: `low` / `medium` / `high` / `critical` — thresholds calibrated per tenure segment.

---

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8080

# Run tests
PYTHONPATH=. pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -t churn-propensity:v1 .

# Run
docker run -p 8080:8080 churn-propensity:v1
```

See [docs/gcp-commands.md](docs/gcp-commands.md) for deploying to Cloud Run.

---

## Retraining pipeline

The Prefect DAG in `dags/retrain.py` runs monthly and:

1. Checks if current model PR-AUC has dropped below 0.55 — skips if healthy
2. Fetches fresh data from BigQuery
3. Validates data quality (row count, null rates, churn rate range)
4. Retrains with fixed v1 hyperparameters (fast path)
5. If fast-path model doesn't improve → runs Optuna search (30 trials)
6. If either candidate beats current → promotes model, writes `reports/promotion_report.json`
7. If neither wins → writes `reports/rejection_report.json` for human review

```bash
# Start the scheduler
python dags/retrain.py
```
