# Troubleshooting and Operational Guide

**Author**: Azaz Butt — AI/ML Engineer & Researcher  
**System**: Tabular ML and Leakage-Audit Workbench (`v1.0.0`)

---

## 1. Environment & Dependency Issues

### Symptom: `ModuleNotFoundError: No module named 'workbench'`
- **Cause**: Running a script or `pytest` outside the package root without `src` on `PYTHONPATH` or editable installation.
- **Resolution**:
  ```powershell
  cd tabular-ml-leakage-workbench
  pip install -e .
  # Or explicitly set PYTHONPATH:
  $env:PYTHONPATH="src"
  python -m pytest tests/test_workbench.py -v
  ```

### Symptom: Missing `python-multipart` when calling `POST /datasets/upload`
- **Cause**: FastAPI requires `python-multipart` to parse `UploadFile` and `Form` fields.
- **Resolution**:
  ```powershell
  pip install python-multipart>=0.0.6
  ```

---

## 2. Model Artefact & Integrity Errors

### Symptom: `ValueError: SHA-256 integrity mismatch for model artefact`
- **Cause**: `load_verified_model_artifact()` computes the SHA-256 digest of the `.joblib` binary before deserialising and compares it against `artifacts/models/<artifact_id>_manifest.json`. If the `.joblib` file was modified outside the pipeline or partially overwritten, loading is blocked as a safety measure.
- **Resolution**: Re-run the deterministic pipeline to regenerate verified model artefacts and sidecar digests:
  ```powershell
  python scripts/run_all.py
  ```

### Symptom: `FileNotFoundError: Model artefact or manifest not found`
- **Cause**: Starting the Streamlit UI or calling prediction endpoints before initial artefact generation.
- **Resolution**: Run `python scripts/run_all.py` once (or click **"Re-run Full Clean Benchmark Pipeline"** in the Streamlit sidebar). Both the FastAPI endpoints and Streamlit views also auto-initialise artefacts on first request if missing.

---

## 3. CSV Upload & Batch Prediction Validation Errors

### Symptom: `HTTP 413: Uploaded file size exceeds the 50 MB limit` or `Row count exceeds 100,000`
- **Cause**: The local workbench enforces explicit computational guardrails (`MAX_UPLOAD_MB = 50`, `MAX_UPLOAD_ROWS = 100,000`) to keep CPU memory and pairwise near-duplicate auditing responsive.
- **Resolution**: Downsample or partition large historical extracts by cohort or observation quarter before uploading to the interactive workbench.

### Symptom: `ValueError: Input data is missing required predictor columns`
- **Cause**: Batch inference (`POST /predict/batch` or `run_batch_prediction`) requires all 21 contract predictor columns (16 numerical, 5 categorical) expected by the trained `ColumnTransformer`.
- **Resolution**: Inspect `GET /contract` to verify the exact list of required predictor names (`mrr_gbp`, `seats_purchased`, `tenure_months`, `discount_pct`, `active_seats_ratio_30d`, `usage_events_30d`, `usage_trend_90d_pct`, `api_error_rate_30d`, `feature_adoption_score`, `days_since_last_admin_login`, `support_tickets_30d`, `urgent_tickets_90d`, `mean_first_response_hours`, `csat_trailing_90d`, `late_payments_12m`, `dunning_events_6m`, `plan_tier`, `contract_type`, `industry_sector`, `company_size_band`, `payment_method`). Legitimate missing values (`NaN` / `null`) inside those columns are handled automatically by the pipeline's trained imputers.

### Symptom: `ValueError: Numerical predictor column '...' contains non-numeric string values`
- **Cause**: Currency symbols (e.g. `£1,200`) or text strings (`"N/A_STRING"`) were passed inside a numerical feature column.
- **Resolution**: Strip currency symbols and format missing values as standard empty CSV cells (`""` / `NaN`).
