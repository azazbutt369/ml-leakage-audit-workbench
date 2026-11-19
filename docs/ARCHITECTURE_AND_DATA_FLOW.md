# Architecture and Data-Flow Explanation

**Author**: Azaz Butt — AI/ML Engineer & Researcher  
**System**: Tabular ML and Leakage-Audit Workbench (`v1.0.0`)

---

## 1. Architectural Philosophy

Most commercial tabular machine-learning prototypes fail between notebook experimentation and production deployment not because the classifier algorithm is weak, but because:

1. **Outcome windows bleed across temporal splits** (training on snapshots whose future 90-day churn label had not yet materialised at the validation cutoff date).
2. **Repeated account snapshots cross partition boundaries** when teams evaluate new-account generalisation using a naive row-level `train_test_split`.
3. **Transformations and resampling cross the training boundary** (`StandardScaler`, `SimpleImputer` or minority-class oversampling fitted on the complete dataset prior to partitioning).
4. **Post-observation proxies enter the predictor matrix** without feature-availability lag auditing.
5. **Decision thresholds and probability calibration are tuned on the final test set**, invalidating out-of-sample uncertainty estimates.

This workbench enforces structural separation across every stage of the tabular ML lifecycle.

---

## 2. End-to-End System Data Flow

```
+-----------------------------------------------------------------------------------+
| 1. PREDICTION CONTRACT & DATA LINEAGE LAYER (src/workbench/contract.py)           |
|    - Unit of observation: (account_id, observation_date) monthly active snapshot  |
|    - Prediction horizon: 90 days (outcome_window_end = observation_date + 90d)    |
|    - Right-censoring filter: snapshots with outcome_window_end > 2025-06-30       |
|      are flagged is_censored_outcome = True and excluded from supervised splits.  |
|    - FeatureSpec registry: records availability_lag_hours <= 0 for all predictors.|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. DUAL EVALUATION PROTOCOL ENGINE (src/workbench/splits.py)                      |
|    - Protocol B (Primary — Future-Period Prediction):                             |
|        * Train: 2024-01-01 .. 2024-08-01 (outcomes complete by 2024-10-30)        |
|        * Purge Buffer 1: 2024-09-01 .. 2024-10-01 (outcomes overlap Val start)    |
|        * Development (Val): 2024-11-01 .. 2024-12-01                              |
|        * Purge Buffer 2: 2025-01-01 .. 2025-02-01 (outcomes overlap Test start)   |
|        * Frozen Test: 2025-03-01 .. 2025-04-01                                    |
|    - Protocol A (Secondary — New-Account Generalisation):                         |
|        * 20-fold StratifiedGroupKFold keyed on account_id (70% / 15% / 15%)       |
|        * Zero account_id overlap across Train, Development and Frozen Test        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. TRAIN-ONLY PIPELINES & FIT PROVENANCE (src/workbench/pipelines.py)             |
|    - AuditedModelArtifact & FitProvenanceRecord record the exact SHA-256          |
|      fitted_row_hashes, account_ids, date bounds and learned medians/means        |
|      for every CV fold and final training fit.                                    |
|    - Out-of-fold Platt sigmoid calibrator fitted strictly on inner-CV OOF         |
|      training predictions (zero validation or test rows used).                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. 11-RULE LEAKAGE-AUDIT ENGINE (src/workbench/leakage_audit.py)                  |
|    - LAK-001: Group/account overlap across partitions                             |
|    - LAK-002: Exact predictor-vector SHA-256 duplicates crossing partitions       |
|    - LAK-003: Bounded standardised L2 near-duplicate search across splits         |
|    - LAK-004: Target or target-derived features in predictors                     |
|    - LAK-005: Point-in-time feature availability lag verification                 |
|    - LAK-006: Temporal ordering & 90-day outcome-window bleed verification        |
|    - LAK-007: Preprocessing fit boundary verification via FitProvenanceRecord     |
|    - LAK-008: Pre-split minority resampling / duplication check                   |
|    - LAK-009: Invalid identifiers, timestamps or lineage metadata in predictors   |
|    - LAK-010: Conflicting labels on (account_id, observation_date) & censoring    |
|    - LAK-011: Covariate & prevalence shift (PSI / TVD), separated from leakage    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. EVALUATION, EXPLAINABILITY & PERSISTENCE (evaluation.py, explainability.py)    |
|    - Decision threshold tau* selected on Development (Val) and frozen before Test |
|    - Account-cluster bootstrap (B=250 resamples of unique account_id clusters)    |
|    - Global permutation importance on Development set; FP/FN inspection on Test   |
|    - Persisted to SQLite (workbench_state.db), Joblib (.joblib + SHA-256) & CSV   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Provenance Verification vs Raw CSV Upload Boundaries

When a user uploads an arbitrary external CSV file via `POST /datasets/upload` or the Streamlit interface:

- **Directly Verifiable from Flat CSV**:
  - `LAK-001` (Account/group overlap, provided an entity ID column is selected)
  - `LAK-002` (Exact duplicates crossing partitions)
  - `LAK-003` (Bounded near-duplicate check)
  - `LAK-009` (High-cardinality IDs/timestamps in predictors)
  - `LAK-010` (Conflicting labels and missing target values)
  - `LAK-011` (Covariate PSI/TVD and target prevalence shift)
- **Requires Lineage or Execution Provenance Metadata (`UNVERIFIABLE_MISSING_PROVENANCE` on raw CSV)**:
  - `LAK-005` (Feature availability lags at prediction time: a flat CSV does not prove when a column was backfilled in the source database)
  - `LAK-006` (Outcome-window overlap if `outcome_window_end` or the horizon definition is omitted)
  - `LAK-007` (Whether an uploaded table had already been imputed or scaled across all rows prior to upload)
  - `LAK-008` (Whether synthetic interpolation such as SMOTE was applied before the CSV was exported)

By surfacing this distinction in the **Provenance Capability Matrix**, the workbench never silently declares an uploaded dataset leakage-free.
