"""
Churn model retraining pipeline.

Schedule: monthly
Strategy:
  1. Check if current model PR-AUC is below degradation threshold — skip if healthy
  2. Fetch fresh data from BigQuery
  3. Validate data quality
  4. Train with v1 hyperparameters (fast path)
  5. If fast-path model beats current → promote
  6. Else → run Optuna hyperparameter search (30 trials)
  7. If tuned model beats current → promote
  8. Else → reject, write report, flag for human review
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from prefect import flow, task
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
MODEL_DIR     = PROJECT_ROOT / "src" / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CURRENT_MODEL_PATH    = MODEL_DIR / "churn_model_v1.ubj"
CURRENT_METADATA_PATH = MODEL_DIR / "churn_model_v1_metadata.json"

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT     = "churn-propensity-model"
BQ_TABLE        = "churn-propensity-model.churn_ds.feature_store"
DROP_COLS       = ["customerID", "tenure_segment", "churn_label"]
CAT_COLS        = ["contract_type", "payment_method", "internet_service"]
DEGRADATION_THRESHOLD = 0.55   # retrain only if current PR-AUC drops below this
OPTUNA_TRIALS         = 30
RANDOM_STATE          = 42

logger = logging.getLogger(__name__)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="check_model_health")
def check_model_health() -> tuple[bool, float]:
    """Return (should_retrain, current_pr_auc). Skip retraining if model is healthy."""
    with open(CURRENT_METADATA_PATH) as f:
        metadata = json.load(f)
    current_pr_auc = metadata["metrics"]["pr_auc"]
    should_retrain = current_pr_auc < DEGRADATION_THRESHOLD
    if should_retrain:
        logger.info(f"Model PR-AUC {current_pr_auc:.4f} below threshold {DEGRADATION_THRESHOLD} — retraining.")
    else:
        logger.info(f"Model PR-AUC {current_pr_auc:.4f} is healthy — skipping retraining.")
    return should_retrain, current_pr_auc


@task(name="fetch_data")
def fetch_data() -> pd.DataFrame:
    """Pull latest data from BigQuery feature store."""
    client = bigquery.Client(project=GCP_PROJECT)
    df = client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()
    logger.info(f"Fetched {len(df):,} rows — churn rate: {df['churn_label'].mean():.2%}")
    return df


@task(name="validate_data")
def validate_data(df: pd.DataFrame) -> None:
    """Basic data quality checks. Raises if data is unfit for training."""
    assert len(df) >= 1000, f"Too few rows: {len(df)}"

    null_rates = df[["tenure", "monthly_charges", "churn_label"]].isnull().mean()
    for col, rate in null_rates.items():
        assert rate < 0.05, f"Column '{col}' has {rate:.1%} nulls"

    churn_rate = df["churn_label"].mean()
    assert 0.05 <= churn_rate <= 0.60, f"Churn rate {churn_rate:.2%} out of expected range"

    logger.info(f"Data validation passed — {len(df):,} rows, churn rate {churn_rate:.2%}")


@task(name="prepare_features")
def prepare_features(df: pd.DataFrame):
    """Split into train/test and encode categoricals."""
    X = df.drop(columns=DROP_COLS)
    y = df["churn_label"].astype(int)

    for col in CAT_COLS:
        X[col] = pd.Categorical(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


@task(name="train_fixed_params")
def train_fixed_params(X_train, X_test, y_train, y_test) -> tuple[xgb.XGBClassifier, dict]:
    """Train with v1 hyperparameters (fast path)."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc":  round(float(average_precision_score(y_test, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    logger.info(f"Fixed-params model — PR-AUC: {metrics['pr_auc']:.4f}")
    return model, metrics


@task(name="tune_hyperparameters")
def tune_hyperparameters(X_train, X_test, y_train, y_test) -> tuple[xgb.XGBClassifier, dict]:
    """Optuna search — only called if fixed-params model didn't beat current."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 4.0),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "enable_categorical": True,
            "random_state":     RANDOM_STATE,
            "eval_metric":      "aucpr",
            "early_stopping_rounds": 20,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return average_precision_score(y_test, m.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "enable_categorical": True,
        "random_state": RANDOM_STATE,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 20,
    })
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc":  round(float(average_precision_score(y_test, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    logger.info(f"Tuned model — PR-AUC: {metrics['pr_auc']:.4f} (best trial: {study.best_value:.4f})")
    return model, metrics


@task(name="promote_model")
def promote_model(
    model: xgb.XGBClassifier,
    new_metrics: dict,
    current_pr_auc: float,
    strategy: str,
    df: pd.DataFrame,
) -> None:
    """Save new model and write promotion report."""
    run_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Archive current model
    archive_path = MODEL_DIR / f"churn_model_archive_{run_date}.ubj"
    current_model = xgb.XGBClassifier()
    current_model.load_model(CURRENT_MODEL_PATH)
    current_model.save_model(archive_path)

    # Save new model in place
    model.save_model(CURRENT_MODEL_PATH)

    # Update metadata
    with open(CURRENT_METADATA_PATH) as f:
        metadata = json.load(f)

    metadata["metrics"]["pr_auc"]  = new_metrics["pr_auc"]
    metadata["metrics"]["roc_auc"] = new_metrics["roc_auc"]
    metadata["trained_date"]       = datetime.utcnow().strftime("%Y-%m-%d")
    metadata["training_data"]["total_rows"] = len(df)
    metadata["training_data"]["churn_rate"] = round(float(df["churn_label"].mean()), 4)

    with open(CURRENT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write promotion report
    report = {
        "run_date":       run_date,
        "status":         "PROMOTED",
        "strategy":       strategy,
        "previous_pr_auc": current_pr_auc,
        "new_pr_auc":     new_metrics["pr_auc"],
        "improvement":    round(new_metrics["pr_auc"] - current_pr_auc, 4),
        "action_required": "Rebuild and redeploy Docker image to Cloud Run.",
    }
    report_path = REPORTS_DIR / f"promotion_report_{run_date}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Model promoted via {strategy}. PR-AUC: {current_pr_auc:.4f} → {new_metrics['pr_auc']:.4f}")
    logger.info(f"Report saved: {report_path}")
    logger.info("ACTION REQUIRED: Rebuild Docker image and redeploy to Cloud Run.")


@task(name="reject_model")
def reject_model(current_pr_auc: float, fixed_pr_auc: float, tuned_pr_auc: float) -> None:
    """Log rejection report — neither candidate beat the current model."""
    run_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "run_date":         run_date,
        "status":           "REJECTED",
        "current_pr_auc":   current_pr_auc,
        "fixed_params_pr_auc": fixed_pr_auc,
        "tuned_pr_auc":     tuned_pr_auc,
        "action_required":  "Review data quality and feature distributions. Manual investigation needed.",
    }
    report_path = REPORTS_DIR / f"rejection_report_{run_date}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.warning(f"Both candidates rejected. Current: {current_pr_auc:.4f}, Fixed: {fixed_pr_auc:.4f}, Tuned: {tuned_pr_auc:.4f}")
    logger.warning(f"Report saved: {report_path}")


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="churn_model_retraining", log_prints=True)
def retrain_pipeline():
    # 1. Check if retraining is needed
    should_retrain, current_pr_auc = check_model_health()
    if not should_retrain:
        return

    # 2. Fetch and validate data
    df = fetch_data()
    validate_data(df)

    # 3. Prepare features
    X_train, X_test, y_train, y_test = prepare_features(df)

    # 4. Fast path — fixed hyperparameters
    with mlflow.start_run(run_name="fixed_params"):
        fixed_model, fixed_metrics = train_fixed_params(X_train, X_test, y_train, y_test)
        mlflow.log_metrics(fixed_metrics)

    if fixed_metrics["pr_auc"] > current_pr_auc:
        promote_model(fixed_model, fixed_metrics, current_pr_auc, "fixed_params", df)
        return

    # 5. Slow path — Optuna tuning
    logger.info("Fixed-params model didn't improve. Running Optuna search...")
    with mlflow.start_run(run_name="optuna_tuned"):
        tuned_model, tuned_metrics = tune_hyperparameters(X_train, X_test, y_train, y_test)
        mlflow.log_metrics(tuned_metrics)

    if tuned_metrics["pr_auc"] > current_pr_auc:
        promote_model(tuned_model, tuned_metrics, current_pr_auc, "optuna_tuned", df)
        return

    # 6. Both failed — reject
    reject_model(current_pr_auc, fixed_metrics["pr_auc"], tuned_metrics["pr_auc"])


if __name__ == "__main__":
    retrain_pipeline.serve(
        name="churn-retraining-monthly",
        cron="0 6 1 * *",   # 6am UTC on the 1st of every month
    )
