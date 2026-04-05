# GCP Deployment Commands

Reference for building, pushing, and deploying the churn API to Google Cloud Run.

---

## Prerequisites

```bash
# Check gcloud is installed
gcloud --version

# Login
gcloud auth login

# Set project
gcloud config set project churn-propensity-model

# Enable required services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Authenticate Docker with GCP
gcloud auth configure-docker europe-west2-docker.pkg.dev
```

---

## Artifact Registry

```bash
# Create the repository (one-time setup)
gcloud artifacts repositories create churn-api \
  --repository-format=docker \
  --location=europe-west2 \
  --description="Churn propensity model API"

# List repositories
gcloud artifacts repositories list
```

---

## Build and push image

```bash
# Build locally
docker build -t churn-propensity:v1 .

# Tag for Artifact Registry
docker tag churn-propensity:v1 \
  europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v1

# Push
docker push \
  europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v1

# List images in registry
gcloud artifacts docker images list \
  europe-west2-docker.pkg.dev/churn-propensity-model/churn-api
```

---

## Deploy to Cloud Run

```bash
# Deploy (or redeploy with new image version)
gcloud run deploy churn-api \
  --image europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v1 \
  --platform managed \
  --region europe-west2 \
  --port 8080 \
  --allow-unauthenticated

# View service details
gcloud run services describe churn-api --region europe-west2

# List all services
gcloud run services list
```

---

## Redeploy after model promotion

When `reports/promotion_report.json` is written by the retraining pipeline:

```bash
# 1. Rebuild with new model
docker build -t churn-propensity:v2 .

# 2. Tag and push
docker tag churn-propensity:v2 \
  europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v2

docker push \
  europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v2

# 3. Deploy new revision
gcloud run deploy churn-api \
  --image europe-west2-docker.pkg.dev/churn-propensity-model/churn-api/churn-propensity:v2 \
  --platform managed \
  --region europe-west2 \
  --port 8080 \
  --allow-unauthenticated
```

---

## Useful commands

```bash
# View live logs
gcloud run services logs read churn-api --region europe-west2

# Stream logs in real time
gcloud beta run services logs tail churn-api --region europe-west2

# View all revisions
gcloud run revisions list --service churn-api --region europe-west2

# Roll back to previous revision
gcloud run services update-traffic churn-api \
  --region europe-west2 \
  --to-revisions REVISION_NAME=100
```
