# Model Card and Data Card (Including Methodological Limitations)

**Author**: Azaz Butt — AI/ML Engineer & Researcher  
**Version**: `1.0.0`

---

## Part 1: Data Card (`SYNTH-B2B-SAAS-CHURN-V1`)

### 1.1 Dataset Identity & Provenance
- **Dataset ID**: `SYNTH-B2B-SAAS-CHURN-V1`
- **Generation Mechanism**: Deterministic multi-wave longitudinal survival simulation (`src/workbench/data_generator.py`, `SEED = 20261105`).
- **Licence & Privacy**: 100% synthetic demonstration data; contains zero real customer records, PII, or proprietary commercial data.
- **Fingerprints (SHA-256)**:
  - Full Dataset (`15,259` rows): `353eb22cc4d21dc310726eb0c6334be19f1f5cc8305195e48d29cf64f85a6f33` (stored in `data/clean/generation_manifest.json`)
  - Uncensored Modelling Dataset (`14,198` rows): `5a896989783454d8c2b4e04500b8d6fd537ab3a1087ed89ffc6c25f44e024cf0`

### 1.2 Observation Structure & Censoring Accounting
- **Unit of Observation**: Monthly active B2B SaaS account snapshot `(account_id, observation_date)` recorded at `00:00:00 UTC` on the 1st of each calendar month between `2024-01-01` and `2025-06-01` (18 monthly waves across `1,650` unique customer accounts).
- **Target (`churned_within_90d`)**: Binary indicator (`1` = subscription cancelled or terminated for non-payment within $(t_{\text{obs}}, t_{\text{obs}} + 90\text{ days}]$; `0` = retained past $t_{\text{obs}} + 90\text{ days}$).
- **Right-Censoring Treatment**: Global extraction cutoff is `2025-06-30`. The `1,061` snapshots observed on `2025-05-01` and `2025-06-01` have 90-day outcome windows extending past `2025-06-30`; they are flagged `is_censored_outcome = True` (`churned_within_90d = NaN`) and excluded from supervised partitions prior to modelling.
- **Structural Missingness**:
  - `csat_trailing_90d` (`47.74%` missing): Structurally missing when no CSAT survey was submitted by the account in the trailing 90 days.
  - `mean_first_response_hours` (`18.42%` missing): Structurally missing when the account opened zero support tickets in the trailing 90 days.
  - `usage_trend_90d_pct` (`10.22%` missing): Missing when `tenure_months < 3` or historical telemetry backfill is incomplete.
  - `industry_sector` (`3.30%` missing) and `company_size_band` (`2.70%` missing): Firmographic enrichment missingness on self-serve signups.

---

## Part 2: Model Card

### 2.1 Candidate Models & Preprocessing Pipelines
1. **`class_prior_baseline`**: `DummyClassifier(strategy="prior")` predicting the training base rate.
2. **`logistic_regression`**: `SimpleImputer(strategy="median", add_indicator=True)` + `StandardScaler()` for numericals; `SimpleImputer(fill_value="__MISSING__")` + `OneHotEncoder(handle_unknown="ignore")` for categoricals; `LogisticRegression(max_iter=1000)` tuned over `C in [0.25, 1.0]`, `class_weight in [None, "balanced"]`, and fold-isolated oversampling.
3. **`hist_gradient_boosting`**: `SimpleImputer(strategy="median", add_indicator=True)` for numericals; `SimpleImputer(fill_value="__MISSING__")` + `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)` for categoricals; `HistGradientBoostingClassifier` tuned over tree complexity and regularisation strictly on inner cross-validation folds.

### 2.2 Preserved Unsuccessful Comparisons
In accordance with rigorous evaluation reporting, all unsuccessful hyperparameter and resampling trials are preserved in `artifacts/reports/master_workbench_summary.json`:
- **Fold-Isolated Minority Oversampling (35% ratio)** degraded out-of-fold Brier calibration on Logistic Regression (Brier worsened from `0.0757` to `0.1190`) and reduced inner-CV PR-AUC on `HistGradientBoostingClassifier` (`0.3358` vs `0.3547` under Protocol B; `0.3282` vs `0.3413` under Protocol A). Consequently, the unresampled models with out-of-fold Platt scaling were selected by the inner-CV criterion.
- **High-Capacity Over-Parameterised Trees (`max_leaf_nodes=63, min_samples_leaf=15, l2=0.1`)** overfitted the training folds and underperformed regularized shallow trees (`max_leaf_nodes=15, min_samples_leaf=40, l2=1.5`) on inner-CV PR-AUC (`0.2995` vs `0.3547`).

### 2.3 Intended Use & Legitimate Decision Support
- **Supported**: Ranking active customer accounts by estimated 90-day churn probability at monthly review cadences; triaging proactive Customer Success outreach within fixed team capacity; auditing dataset splits and feature pipelines before production deployment.
- **Not Supported**: Automated punitive contract actions; causal attribution of *why* an individual customer is dissatisfied without human CSM review; extrapolating probabilities to unrepresented product lines or enterprise segments.

---

## Part 3: Explicit Methodological & System Limitations

1. **Same-Author Synthetic Benchmark Limitation**:
   - Because the synthetic generator and evaluation pipelines were written by the same author, the benchmark validates software correctness, split purging, and audit rules—not real-world predictive accuracy on an external business's warehouse.
2. **Audit Engine Coverage Boundaries**:
   - No automated leakage detector can identify every form of subtle domain leakage. On raw uploaded CSVs without `FeatureSpec` lag registries or `FitProvenanceRecord` logs, upstream look-ahead SQL joins (`LAK-005`), pre-upload global scaling/imputation (`LAK-007`), and pre-upload SMOTE interpolation (`LAK-008`) cannot be proven from cell values alone and are explicitly marked `UNVERIFIABLE_MISSING_PROVENANCE`.
3. **Near-Duplicate Search Computational Cap (`LAK-003`)**:
   - Standardised Euclidean pairwise comparison is bounded at `NEAR_DUPLICATE_MAX_ROWS_PER_SPLIT = 1,500` (`2,250,000` pairs) to keep CPU execution fast; exact duplicates (`LAK-002`) scan 100% of rows via $O(N)$ SHA-256 hashing.
4. **Finite Temporal Horizon Uncertainty**:
   - While account-cluster bootstrapping (`B = 250` cluster resamples) accounts for intra-account correlation across repeated monthly snapshots, Protocol B's frozen test set spans two calendar months (`2025-03-01` and `2025-04-01`), which cannot capture multi-year macro-economic seasonality.
5. **Non-Causal Feature Attributions**:
   - Permutation importances and instance-level marginal perturbations describe predictive associations within the fitted model; they do not establish causal treatment effects.
