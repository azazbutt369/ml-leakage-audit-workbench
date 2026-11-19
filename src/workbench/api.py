"""
FastAPI Service for the Tabular ML and Leakage-Audit Workbench.
Exposes endpoints for contract inspection, dataset loading, CSV upload validation,
leakage auditing, controlled fixture benchmarks, model training/evaluation,
single/batch inference and report export.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from workbench.config import (
    BRAND_AUTHOR,
    BRAND_ROLE,
    BRAND_SCENARIO_TITLE,
    BRAND_WORKBENCH_NAME,
    CLEAN_DATA_DIR,
    DEFAULT_SEED,
    EXPORTS_DIR,
    FIXTURES_DIR,
    MAX_UPLOAD_BYTES,
    MODELS_DIR,
    REPORTS_DIR,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
    ensure_directories,
)
from workbench.contract import (
    build_default_saas_contract,
    validate_uploaded_dataframe,
)
from workbench.explainability import compute_local_instance_attribution
from workbench.leakage_audit import run_leakage_audit
from workbench.runner import run_complete_workbench_pipeline
from workbench.splits import (
    split_protocol_a_new_account_generalisation,
    split_protocol_b_future_period_prediction,
)
from workbench.storage import (
    load_verified_model_artifact,
    run_batch_prediction,
)

app = FastAPI(
    title=f"{BRAND_AUTHOR} — {BRAND_WORKBENCH_NAME} API",
    description=(
        f"{BRAND_SCENARIO_TITLE}. Provides reproducible tabular ML pipelines, "
        "11-rule leakage auditing, controlled fault benchmarks, calibration diagnostics "
        "and schema-validated batch inference."
    ),
    version="1.0.0",
)


class SinglePredictionRequest(BaseModel):
    protocol_id: Literal["protocol_b_temporal_purged", "protocol_a_group_holdout"] = "protocol_b_temporal_purged"
    model_key: Literal["hist_gradient_boosting", "logistic_regression", "class_prior_baseline"] = (
        "hist_gradient_boosting"
    )
    features: Dict[str, Any]
    custom_threshold: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    protocol_id: Literal["protocol_b_temporal_purged", "protocol_a_group_holdout"] = "protocol_b_temporal_purged"
    model_key: Literal["hist_gradient_boosting", "logistic_regression", "class_prior_baseline"] = (
        "hist_gradient_boosting"
    )
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=10000)
    custom_threshold: Optional[float] = None


class CustomAuditRequest(BaseModel):
    protocol_id: Literal["protocol_b_temporal_purged", "protocol_a_group_holdout"] = "protocol_b_temporal_purged"
    allow_returning_accounts: bool = True
    target_col: str = "churned_within_90d"
    account_col: Optional[str] = "account_id"
    date_col: Optional[str] = "observation_date"
    outcome_end_col: Optional[str] = "outcome_window_end"
    feature_cols: Optional[List[str]] = None
    records: Optional[List[Dict[str, Any]]] = None


@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Return service health, runtime environment and artefact availability."""
    ensure_directories()
    master_exists = (REPORTS_DIR / "master_workbench_summary.json").exists()
    return {
        "status": "ok",
        "author": BRAND_AUTHOR,
        "role": BRAND_ROLE,
        "workbench": BRAND_WORKBENCH_NAME,
        "scenario": BRAND_SCENARIO_TITLE,
        "python_version": sys.version.split()[0],
        "benchmark_artefacts_ready": master_exists,
        "synthetic_disclosure": SYNTHETIC_BENCHMARK_DISCLOSURE,
    }


@app.get("/contract")
def get_prediction_contract() -> Dict[str, Any]:
    """Return the formal B2B SaaS Churn Prediction Contract and Feature Availability Registry."""
    contract = build_default_saas_contract()
    return contract.model_dump()


@app.get("/datasets/demo")
def get_demo_dataset_summary(sample_rows: int = 25) -> Dict[str, Any]:
    """Return the clean synthetic dataset manifest, data dictionary and sample preview rows."""
    manifest_path = CLEAN_DATA_DIR / "generation_manifest.json"
    modelling_csv = CLEAN_DATA_DIR / "saas_churn_snapshots_modelling.csv"
    if not manifest_path.exists() or not modelling_csv.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    df_preview = pd.read_csv(modelling_csv, nrows=max(1, min(sample_rows, 200)))
    return {
        "manifest": manifest,
        "preview_rows": json.loads(df_preview.to_json(orient="records")),
    }


@app.post("/datasets/upload")
async def validate_uploaded_csv(
    request: Request,
    target_col: str = Query("churned_within_90d"),
    account_id_col: Optional[str] = Query(None),
    timestamp_col: Optional[str] = Query(None),
    outcome_window_end_col: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Validate an uploaded CSV payload (raw text/csv body or JSON {'csv_text': ...}) for computational
    bounds, schema validity and target binary domain, returning an explicit Provenance Capability Matrix.
    Never silently declares uploaded data leakage-free.
    """
    raw_bytes = await request.body()
    file_size = len(raw_bytes)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file size ({file_size} bytes) exceeds the {MAX_UPLOAD_BYTES} byte limit.",
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV payload is empty.")

    content_type = request.headers.get("content-type", "").lower()
    try:
        if "application/json" in content_type:
            payload = json.loads(raw_bytes.decode("utf-8"))
            csv_text = payload.get("csv_text", "")
            target_col = payload.get("target_col", target_col)
            account_id_col = payload.get("account_id_col", account_id_col)
            timestamp_col = payload.get("timestamp_col", timestamp_col)
            outcome_window_end_col = payload.get("outcome_window_end_col", outcome_window_end_col)
            df = pd.read_csv(io.StringIO(csv_text))
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV payload: {exc}") from exc

    report = validate_uploaded_dataframe(
        df=df,
        target_col=target_col,
        account_id_col=account_id_col if account_id_col else None,
        timestamp_col=timestamp_col if timestamp_col else None,
        outcome_window_end_col=outcome_window_end_col if outcome_window_end_col else None,
        file_size_bytes=file_size,
        has_feature_availability_metadata=False,
        has_fit_provenance=False,
        has_resampling_provenance=False,
    )
    return report.model_dump()


@app.get("/audit/fixtures")
def get_controlled_fixtures_audit() -> Dict[str, Any]:
    """Return the benchmark audit results across all 7 controlled leakage and negative-control fixtures."""
    summary_path = FIXTURES_DIR / "controlled_fixtures_audit_summary.json"
    if not summary_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.post("/audit/run")
def run_audit_endpoint(req: CustomAuditRequest) -> Dict[str, Any]:
    """Run the 11-rule Leakage-Audit Engine on either the demonstration dataset or custom records."""
    contract = build_default_saas_contract()

    if req.records is not None and len(req.records) > 0:
        df = pd.DataFrame(req.records)
        val_rep = validate_uploaded_dataframe(
            df=df,
            target_col=req.target_col,
            account_id_col=req.account_col,
            timestamp_col=req.date_col,
            outcome_window_end_col=req.outcome_end_col,
            feature_cols=req.feature_cols,
            has_feature_availability_metadata=False,
            has_fit_provenance=False,
            has_resampling_provenance=False,
        )
        if not val_rep.is_valid:
            raise HTTPException(status_code=422, detail={"validation_errors": val_rep.errors})

        feat_cols = req.feature_cols or (val_rep.detected_numerical_cols + val_rep.detected_categorical_cols)
        clean_df = df.dropna(subset=[req.target_col]).copy().reset_index(drop=True)
        clean_df[req.target_col] = pd.to_numeric(clean_df[req.target_col], errors="raise").astype(int)

        if req.protocol_id == "protocol_a_group_holdout" and req.account_col and req.account_col in clean_df.columns:
            bundle = split_protocol_a_new_account_generalisation(
                clean_df,
                account_col=req.account_col,
                date_col=req.date_col or "observation_date",
                target_col=req.target_col,
                save_manifest_dir=None,
            )
            tr_df, vl_df, te_df = bundle.train_df, bundle.val_df, bundle.test_df
        else:
            tr_df, temp_df = train_test_split(
                clean_df, test_size=0.30, stratify=clean_df[req.target_col], random_state=DEFAULT_SEED
            )
            vl_df, te_df = train_test_split(
                temp_df, test_size=0.50, stratify=temp_df[req.target_col], random_state=DEFAULT_SEED
            )
            tr_df, vl_df, te_df = tr_df.reset_index(drop=True), vl_df.reset_index(drop=True), te_df.reset_index(drop=True)

        report = run_leakage_audit(
            train_df=tr_df,
            val_df=vl_df,
            test_df=te_df,
            feature_cols=feat_cols,
            target_col=req.target_col,
            account_col=req.account_col,
            date_col=req.date_col,
            outcome_end_col=req.outcome_end_col,
            protocol_id=req.protocol_id,
            allow_returning_accounts=req.allow_returning_accounts,
            contract=None,
            has_lineage_metadata=False,
            fit_provenance=None,
            resampling_provenance=None,
            dataset_name="Custom_Uploaded_Dataset",
        )
        return report.model_dump()

    # Default: return persisted or freshly computed audit for the requested demo protocol
    audit_path = REPORTS_DIR / f"audit_{req.protocol_id}.json"
    if not audit_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(audit_path.read_text(encoding="utf-8"))


@app.get("/models/summary")
def get_models_summary() -> Dict[str, Any]:
    """Return the master evaluation summary across both protocols and all candidate models."""
    summary_path = REPORTS_DIR / "master_workbench_summary.json"
    if not summary_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.post("/models/train")
def trigger_pipeline_training(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute a fresh deterministic training, auditing and evaluation run."""
    summary = run_complete_workbench_pipeline(seed=seed)
    return {
        "status": "completed",
        "seed": seed,
        "comparison_table": summary["comparison_table"],
    }


@app.post("/predict/single")
def predict_single_instance(req: SinglePredictionRequest) -> Dict[str, Any]:
    """Score a single account snapshot and return probability, decision and local attributions."""
    model_path = MODELS_DIR / f"{req.protocol_id}__{req.model_key}.joblib"
    if not model_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)

    try:
        artifact = load_verified_model_artifact(req.protocol_id, req.model_key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    input_df = pd.DataFrame([req.features])
    try:
        pred_df, diag = run_batch_prediction(
            artifact=artifact,
            input_df=input_df,
            custom_threshold=req.custom_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Compute local feature attributions against clean training reference
    modelling_csv = CLEAN_DATA_DIR / "saas_churn_snapshots_modelling.csv"
    local_attr = []
    if modelling_csv.exists():
        ref_df = pd.read_csv(modelling_csv, nrows=1000)
        local_attr = compute_local_instance_attribution(
            artifact=artifact,
            instance_row=input_df.iloc[0],
            train_reference_df=ref_df,
            top_k=6,
        )

    row_res = json.loads(pred_df.to_json(orient="records"))[0]
    return {
        "prediction": row_res,
        "local_attributions": local_attr,
        "diagnostics": diag,
    }


@app.post("/predict/batch")
def predict_batch_instances(req: BatchPredictionRequest) -> Dict[str, Any]:
    """Score a batch of account snapshots with strict schema validation and row alignment."""
    model_path = MODELS_DIR / f"{req.protocol_id}__{req.model_key}.joblib"
    if not model_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)

    try:
        artifact = load_verified_model_artifact(req.protocol_id, req.model_key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    input_df = pd.DataFrame(req.records)
    try:
        pred_df, diag = run_batch_prediction(
            artifact=artifact,
            input_df=input_df,
            custom_threshold=req.custom_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predictions": json.loads(pred_df.to_json(orient="records")),
        "diagnostics": diag,
    }


@app.get("/export/{protocol_id}")
def export_protocol_bundle(
    protocol_id: Literal["protocol_b_temporal_purged", "protocol_a_group_holdout"],
) -> Dict[str, Any]:
    """Export the complete persisted evaluation, audit and manifest bundle for a protocol."""
    bundle_path = REPORTS_DIR / f"evaluation_bundle_{protocol_id}.json"
    if not bundle_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(bundle_path.read_text(encoding="utf-8"))
