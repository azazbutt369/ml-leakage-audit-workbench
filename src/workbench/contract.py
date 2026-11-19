"""
Formal Prediction Contract, Feature Availability Specification and Upload Schema Validator.
Defines the exact unit of observation, prediction timestamp, outcome horizon, censoring rules
and provenance requirements before any modelling occurs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from workbench.config import (
    DEFAULT_SEED,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_ROWS,
    MIN_REQUIRED_ROWS,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
)


class FeatureSpec(BaseModel):
    """Metadata and availability contract for a single column."""

    name: str
    dtype_category: Literal["numerical", "categorical", "identifier", "timestamp", "metadata", "target"]
    description: str
    source_system: str
    lookback_window: str
    availability_lag_hours: float = Field(
        description=(
            "Hours relative to observation_date when the feature becomes finalised. "
            "Must be <= 0.0 for valid predictors available at prediction time. "
            "Values > 0.0 indicate future information unavailable at prediction time."
        )
    )
    is_predictor: bool
    allow_missing: bool = True
    missingness_mechanism: str = "None"
    expected_domain: Optional[str] = None


class PredictionContract(BaseModel):
    """Formal prediction contract governing dataset construction, splitting and auditing."""

    contract_id: str = "CONTRACT-B2B-SAAS-CHURN-90D-V1"
    scenario_name: str = "B2B SaaS Customer-Churn Prediction"
    unit_of_observation: str = (
        "Monthly active B2B SaaS account snapshot (account_id, observation_date) "
        "recorded at 00:00:00 UTC on the first calendar day of each month."
    )
    account_id_col: str = "account_id"
    snapshot_id_col: str = "snapshot_id"
    timestamp_col: str = "observation_date"
    outcome_window_end_col: str = "outcome_window_end"
    dataset_cutoff_col: str = "dataset_cutoff_date"
    censoring_flag_col: str = "is_censored_outcome"
    target_col: str = "churned_within_90d"
    prediction_horizon_days: int = 90
    dataset_cutoff_date: str = "2025-06-30"
    target_definition: str = (
        "Binary indicator (1 = churned, 0 = retained). Equals 1 if and only if an account "
        "that is ACTIVE at observation_date experiences voluntary subscription cancellation "
        "or involuntary non-payment termination within (observation_date, observation_date + 90 days]."
    )
    censoring_and_missing_outcome_policy: str = (
        "Snapshots whose 90-day outcome window extends past dataset_cutoff_date (2025-06-30) "
        "have right-censored outcomes (is_censored_outcome = True, churned_within_90d = NaN). "
        "Censored snapshots are preserved in raw lineage storage for auditing and strictly "
        "excluded from supervised train, validation and test partitions prior to modelling."
    )
    intended_deployment_population: str = (
        "Active commercial B2B SaaS customer accounts with at least 30 days of subscription "
        "tenure at the monthly scoring timestamp."
    )
    primary_protocol: str = "Protocol B: Future-Period Prediction (Chronological Split with 90-Day Outcome-Window Purging)"
    secondary_protocol: str = "Protocol A: New-Account Generalisation (Disjoint Account-Group Partitioning)"
    numerical_features: List[str]
    categorical_features: List[str]
    identifier_and_metadata_cols: List[str]
    forbidden_predictor_cols: List[str]
    feature_registry: Dict[str, FeatureSpec]
    synthetic_disclosure: str = SYNTHETIC_BENCHMARK_DISCLOSURE
    seed: int = DEFAULT_SEED

    @property
    def all_predictor_cols(self) -> List[str]:
        return list(self.numerical_features) + list(self.categorical_features)


def build_default_saas_contract() -> PredictionContract:
    """Construct the authoritative B2B SaaS Churn Prediction Contract."""
    specs: List[FeatureSpec] = [
        # Identifiers & Lineage Metadata (Excluded from predictors)
        FeatureSpec(
            name="snapshot_id",
            dtype_category="identifier",
            description="Unique deterministic identifier for the account-month observation row.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="account_id",
            dtype_category="identifier",
            description="Unique customer organisation identifier across longitudinal snapshots.",
            source_system="crm_core",
            lookback_window="Point-in-time",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="observation_date",
            dtype_category="timestamp",
            description="Monthly prediction timestamp (YYYY-MM-01 00:00 UTC).",
            source_system="feature_store_scheduler",
            lookback_window="Point-in-time",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="outcome_window_end",
            dtype_category="timestamp",
            description="End date of the 90-day forward churn observation window (observation_date + 90 days).",
            source_system="label_generation_engine",
            lookback_window="Forward 90d (+2160h)",
            availability_lag_hours=2160.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="dataset_cutoff_date",
            dtype_category="timestamp",
            description="Global data extraction cutoff date (2025-06-30) determining outcome censoring.",
            source_system="label_generation_engine",
            lookback_window="Global constant",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="account_status_at_obs",
            dtype_category="metadata",
            description="Account subscription state at observation_date (must be ACTIVE for eligible cohort).",
            source_system="billing_warehouse",
            lookback_window="Point-in-time",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="is_censored_outcome",
            dtype_category="metadata",
            description="Boolean flag indicating outcome_window_end > dataset_cutoff_date.",
            source_system="label_generation_engine",
            lookback_window="Forward 90d check",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        FeatureSpec(
            name="row_hash",
            dtype_category="metadata",
            description="SHA-256 digest of the canonical snapshot values for provenance tracking.",
            source_system="audit_lineage",
            lookback_window="Point-in-time",
            availability_lag_hours=0.0,
            is_predictor=False,
            allow_missing=False,
        ),
        # Numerical Predictors (16)
        FeatureSpec(
            name="mrr_gbp",
            dtype_category="numerical",
            description="Contracted Monthly Recurring Revenue in GBP at observation_date.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 40.0",
        ),
        FeatureSpec(
            name="seats_purchased",
            dtype_category="numerical",
            description="Number of licensed user seats active on the subscription at observation_date.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 1",
        ),
        FeatureSpec(
            name="tenure_months",
            dtype_category="numerical",
            description="Completed months elapsed since initial paid subscription activation.",
            source_system="billing_warehouse",
            lookback_window="Cumulative up to t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 1",
        ),
        FeatureSpec(
            name="discount_pct",
            dtype_category="numerical",
            description="Commercial discount percentage applied to the active subscription.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="[0.0, 45.0]",
        ),
        FeatureSpec(
            name="active_seats_ratio_30d",
            dtype_category="numerical",
            description="Ratio of distinct active users in [t_obs - 30d, t_obs] to seats_purchased.",
            source_system="product_telemetry",
            lookback_window="Trailing 30d closed at t_obs",
            availability_lag_hours=-2.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="[0.0, 1.0]",
        ),
        FeatureSpec(
            name="usage_events_30d",
            dtype_category="numerical",
            description="Total core workflow events logged by the account in [t_obs - 30d, t_obs].",
            source_system="product_telemetry",
            lookback_window="Trailing 30d closed at t_obs",
            availability_lag_hours=-2.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 0",
        ),
        FeatureSpec(
            name="usage_trend_90d_pct",
            dtype_category="numerical",
            description="Percentage change in 30d usage volume relative to the prior 60-90d baseline.",
            source_system="product_telemetry",
            lookback_window="Trailing 90d closed at t_obs",
            availability_lag_hours=-2.0,
            is_predictor=True,
            allow_missing=True,
            missingness_mechanism="Missing when tenure_months < 3 or telemetry backfill incomplete (~6% missing).",
            expected_domain="[-100.0, 300.0]",
        ),
        FeatureSpec(
            name="api_error_rate_30d",
            dtype_category="numerical",
            description="Fraction of customer API/webhook requests returning 4xx/5xx errors in trailing 30d.",
            source_system="api_gateway_logs",
            lookback_window="Trailing 30d closed at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="[0.0, 1.0]",
        ),
        FeatureSpec(
            name="feature_adoption_score",
            dtype_category="numerical",
            description="Composite adoption score [0, 100] across sticky platform integrations at t_obs.",
            source_system="product_telemetry",
            lookback_window="Trailing 30d closed at t_obs",
            availability_lag_hours=-2.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="[0.0, 100.0]",
        ),
        FeatureSpec(
            name="days_since_last_admin_login",
            dtype_category="numerical",
            description="Days elapsed between the most recent workspace administrator login and t_obs.",
            source_system="auth_service",
            lookback_window="Trailing 90d closed at t_obs",
            availability_lag_hours=-0.5,
            is_predictor=True,
            allow_missing=False,
            expected_domain="[0.0, 90.0]",
        ),
        FeatureSpec(
            name="support_tickets_30d",
            dtype_category="numerical",
            description="Count of customer support tickets opened in [t_obs - 30d, t_obs].",
            source_system="helpdesk_crm",
            lookback_window="Trailing 30d closed at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 0",
        ),
        FeatureSpec(
            name="urgent_tickets_90d",
            dtype_category="numerical",
            description="Count of Severity-1/2 escalation tickets opened in [t_obs - 90d, t_obs].",
            source_system="helpdesk_crm",
            lookback_window="Trailing 90d closed at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 0",
        ),
        FeatureSpec(
            name="mean_first_response_hours",
            dtype_category="numerical",
            description="Average support first-response latency (hours) over trailing 90d.",
            source_system="helpdesk_crm",
            lookback_window="Trailing 90d closed at t_obs",
            availability_lag_hours=-4.0,
            is_predictor=True,
            allow_missing=True,
            missingness_mechanism="Structurally missing when zero support tickets occurred in trailing 90d (~18% missing).",
            expected_domain=">= 0.1",
        ),
        FeatureSpec(
            name="csat_trailing_90d",
            dtype_category="numerical",
            description="Mean customer satisfaction score [1.0, 5.0] from surveys completed on or before t_obs.",
            source_system="helpdesk_crm",
            lookback_window="Trailing 90d closed at t_obs",
            availability_lag_hours=-6.0,
            is_predictor=True,
            allow_missing=True,
            missingness_mechanism="Structurally missing when customer submitted no CSAT survey in trailing 90d (~28% missing).",
            expected_domain="[1.0, 5.0]",
        ),
        FeatureSpec(
            name="late_payments_12m",
            dtype_category="numerical",
            description="Count of invoices settled more than 7 days past due in [t_obs - 365d, t_obs].",
            source_system="billing_warehouse",
            lookback_window="Trailing 12m closed at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 0",
        ),
        FeatureSpec(
            name="dunning_events_6m",
            dtype_category="numerical",
            description="Automated payment retry/dunning sequences triggered in [t_obs - 180d, t_obs].",
            source_system="billing_warehouse",
            lookback_window="Trailing 6m closed at t_obs",
            availability_lag_hours=-1.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain=">= 0",
        ),
        # Categorical Predictors (5)
        FeatureSpec(
            name="plan_tier",
            dtype_category="categorical",
            description="Subscription commercial tier at observation_date.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="Starter | Growth | Enterprise",
        ),
        FeatureSpec(
            name="contract_type",
            dtype_category="categorical",
            description="Billing renewal cycle structure at observation_date.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="Monthly | Annual",
        ),
        FeatureSpec(
            name="industry_sector",
            dtype_category="categorical",
            description="Customer vertical classification in CRM.",
            source_system="crm_core",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=True,
            missingness_mechanism="Occasionally unrecorded for self-serve signups (~3% missing).",
            expected_domain="FinTech | HealthTech | Logistics | ECommerce | DevTools | ProfessionalServices",
        ),
        FeatureSpec(
            name="company_size_band",
            dtype_category="categorical",
            description="Employee headcount bracket recorded at onboarding/enrichment.",
            source_system="crm_core",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=True,
            missingness_mechanism="Missing when firmographic enrichment is unavailable (~2.5% missing).",
            expected_domain="1-50 | 51-250 | 251-1000 | 1000+",
        ),
        FeatureSpec(
            name="payment_method",
            dtype_category="categorical",
            description="Primary payment instrument configured at observation_date.",
            source_system="billing_warehouse",
            lookback_window="Point-in-time at t_obs",
            availability_lag_hours=0.0,
            is_predictor=True,
            allow_missing=False,
            expected_domain="DirectDebit | CreditCard | Invoice",
        ),
        # Target Column
        FeatureSpec(
            name="churned_within_90d",
            dtype_category="target",
            description="Binary outcome: 1 if subscription terminated in (t_obs, t_obs + 90d], 0 otherwise (NaN if right-censored).",
            source_system="label_generation_engine",
            lookback_window="Forward 90d (+2160h)",
            availability_lag_hours=2160.0,
            is_predictor=False,
            allow_missing=True,
            missingness_mechanism="NaN only when outcome_window_end > dataset_cutoff_date (right-censored).",
            expected_domain="{0, 1}",
        ),
    ]

    numerical_cols = [s.name for s in specs if s.dtype_category == "numerical" and s.is_predictor]
    categorical_cols = [s.name for s in specs if s.dtype_category == "categorical" and s.is_predictor]
    id_meta_cols = [
        s.name
        for s in specs
        if s.dtype_category in ("identifier", "timestamp", "metadata")
    ]
    forbidden_cols = id_meta_cols + [
        "churned_within_90d",
        "true_latent_propensity",
        "retention_credits_issued_next_60d",
        "cancellation_survey_completed",
        "days_to_cancellation",
    ]

    return PredictionContract(
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        identifier_and_metadata_cols=id_meta_cols,
        forbidden_predictor_cols=forbidden_cols,
        feature_registry={s.name: s for s in specs},
    )


class UploadValidationReport(BaseModel):
    """Structured validation result and provenance capability assessment for any dataset/CSV."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    file_size_bytes: int = 0
    dataset_sha256: str = ""
    detected_numerical_cols: List[str] = Field(default_factory=list)
    detected_categorical_cols: List[str] = Field(default_factory=list)
    missing_value_summary: Dict[str, float] = Field(default_factory=dict)
    target_distribution: Dict[str, Any] = Field(default_factory=dict)
    provenance_capabilities: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Explicitly distinguishes audit rules supported by the uploaded data vs "
            "rules requiring external lineage, feature-availability lags or pipeline fit logs."
        ),
    )


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint for a DataFrame."""
    hashed_series = pd.util.hash_pandas_object(df, index=True)
    digest = hashlib.sha256()
    digest.update(hashed_series.values.tobytes())
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()


def compute_row_hashes(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.Series:
    """Compute canonical, dtype-invariant SHA-256 hex strings per row over the specified columns."""
    subset = (df[cols] if cols is not None else df).copy()
    for col in subset.columns:
        if pd.api.types.is_bool_dtype(subset[col]):
            subset[col] = subset[col].astype(str)
        elif pd.api.types.is_numeric_dtype(subset[col]):
            subset[col] = pd.to_numeric(subset[col], errors="coerce").round(6).astype(float)
        else:
            subset[col] = subset[col].fillna("__NULL__").astype(str)
    int_hashes = pd.util.hash_pandas_object(subset, index=False)
    return int_hashes.map(lambda x: hashlib.sha256(int(x).to_bytes(8, "little", signed=False)).hexdigest()[:24])


def validate_uploaded_dataframe(
    df: pd.DataFrame,
    target_col: str,
    account_id_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    outcome_window_end_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    file_size_bytes: int = 0,
    has_feature_availability_metadata: bool = False,
    has_fit_provenance: bool = False,
    has_resampling_provenance: bool = False,
) -> UploadValidationReport:
    """
    Validate a DataFrame (from demo, fixture or external CSV upload) against computational limits,
    column typing, target validity and provenance availability.
    Never silently declares an uploaded CSV leakage-free.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if file_size_bytes > MAX_UPLOAD_BYTES:
        errors.append(
            f"File size ({file_size_bytes / (1024 * 1024):.2f} MB) exceeds the {MAX_UPLOAD_MB} MB safety limit."
        )

    n_rows, n_cols = df.shape
    if n_rows > MAX_UPLOAD_ROWS:
        errors.append(f"Row count ({n_rows:,}) exceeds the computational cap of {MAX_UPLOAD_ROWS:,} rows.")
    if n_rows < MIN_REQUIRED_ROWS:
        errors.append(f"Row count ({n_rows}) is below the minimum required {MIN_REQUIRED_ROWS} observations.")

    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset columns.")

    if account_id_col and account_id_col not in df.columns:
        errors.append(f"Account/group identifier column '{account_id_col}' not found in dataset.")

    if timestamp_col and timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found in dataset.")

    reserved = {c for c in [target_col, account_id_col, timestamp_col, outcome_window_end_col] if c}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in reserved]
    else:
        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            errors.append(f"Selected feature columns missing from dataset: {missing_feats}")

    if not feature_cols:
        errors.append("At least one predictor feature column must be selected.")

    # Target validation
    target_dist: Dict[str, Any] = {}
    if target_col in df.columns:
        non_null_target = df[target_col].dropna()
        unique_vals = set(non_null_target.unique().tolist())
        if len(unique_vals) < 2:
            errors.append(
                f"Target column '{target_col}' must contain exactly 2 distinct classes for binary classification; found {unique_vals}."
            )
        elif not unique_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1"}):
            errors.append(
                f"Target column '{target_col}' contains non-binary values: {sorted(map(str, list(unique_vals)[:10]))}. Expected {{0, 1}}."
            )
        else:
            numeric_target = pd.to_numeric(non_null_target, errors="coerce")
            pos_count = int((numeric_target == 1).sum())
            neg_count = int((numeric_target == 0).sum())
            null_count = int(df[target_col].isna().sum())
            target_dist = {
                "positive_count": pos_count,
                "negative_count": neg_count,
                "censored_or_missing_count": null_count,
                "positive_prevalence": round(float(pos_count / max(1, pos_count + neg_count)), 4),
            }
            if null_count > 0:
                warnings.append(
                    f"Target column '{target_col}' contains {null_count} missing/censored rows. "
                    "These will be excluded from supervised splits and audited under LAK-010."
                )

    # Detect numerical vs categorical among selected feature_cols
    valid_feats = [c for c in feature_cols if c in df.columns]
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for col in valid_feats:
        if pd.api.types.is_bool_dtype(df[col]):
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    missing_summary = {
        col: round(float(df[col].isna().mean()), 4)
        for col in df.columns
        if df[col].isna().any()
    }

    # Build the explicit Provenance & Evidence Capability Matrix
    provenance_matrix: Dict[str, Dict[str, str]] = {
        "LAK-001 (Group/Account Overlap)": {
            "status": "SUPPORTED_DIRECTLY" if account_id_col and account_id_col in df.columns else "UNVERIFIABLE_MISSING_PROVENANCE",
            "requirement": (
                f"Verified via '{account_id_col}' across partitions."
                if account_id_col and account_id_col in df.columns
                else "Requires an account/entity identifier column; random row splits cannot verify entity isolation."
            ),
        },
        "LAK-002 (Exact Cross-Split Duplicates)": {
            "status": "SUPPORTED_DIRECTLY",
            "requirement": "Verified directly via SHA-256 row and predictor-vector hashing across partitions.",
        },
        "LAK-003 (Near-Duplicate Cross-Split Check)": {
            "status": "SUPPORTED_DIRECTLY",
            "requirement": "Verified via standardised L2/Hamming nearest-neighbour search within computational pair limits.",
        },
        "LAK-004 (Target/Derived Feature in Predictors)": {
            "status": "SUPPORTED_WITH_PROVENANCE" if has_feature_availability_metadata else "HEURISTIC_ONLY_ON_RAW_CSV",
            "requirement": (
                "Full lineage registry available to confirm target-derived columns."
                if has_feature_availability_metadata
                else "Only statistical separation (AUC/MI) and lexical heuristics available; provenance is required to prove causal derivation."
            ),
        },
        "LAK-005 (Future Information Unavailable at Prediction Time)": {
            "status": "SUPPORTED_WITH_PROVENANCE" if has_feature_availability_metadata else "UNVERIFIABLE_MISSING_PROVENANCE",
            "requirement": (
                "Verified against feature availability lag registry relative to observation_date."
                if has_feature_availability_metadata
                else "Cannot verify when features became available in production from a flat CSV without feature-timestamp lineage."
            ),
        },
        "LAK-006 (Temporal Ordering & Outcome-Window Bleed)": {
            "status": (
                "SUPPORTED_DIRECTLY"
                if (timestamp_col and timestamp_col in df.columns and outcome_window_end_col and outcome_window_end_col in df.columns)
                else ("PARTIAL_TIMESTAMP_ONLY" if timestamp_col and timestamp_col in df.columns else "UNVERIFIABLE_MISSING_PROVENANCE")
            ),
            "requirement": (
                "Verified via observation_date and outcome_window_end across splits."
                if (timestamp_col and outcome_window_end_col and outcome_window_end_col in df.columns)
                else "Requires both prediction timestamp and outcome-window end date (or horizon) to verify zero label bleed."
            ),
        },
        "LAK-007 (Preprocessing Fitted Outside Training Boundary)": {
            "status": "SUPPORTED_WITH_PROVENANCE" if has_fit_provenance else "UNVERIFIABLE_MISSING_PROVENANCE",
            "requirement": (
                "Verified by intersecting FitProvenanceRecord fitted row hashes against Val/Test row hashes."
                if has_fit_provenance
                else "If an uploaded CSV was already pre-scaled/imputed before upload, raw values alone cannot prove train-only fitting."
            ),
        },
        "LAK-008 (Resampling/Balancing Before Partitioning)": {
            "status": "SUPPORTED_WITH_PROVENANCE" if has_resampling_provenance else "HEURISTIC_DUPLICATE_CHECK_ONLY",
            "requirement": (
                "Verified via ResamplingProvenanceRecord and pre/post-split row hash multiplicity."
                if has_resampling_provenance
                else "Can detect exact minority-class duplicates crossing splits, but synthetic SMOTE prior to upload requires lineage logs."
            ),
        },
        "LAK-009 (Invalid Identifiers/Metadata in Predictors)": {
            "status": "SUPPORTED_DIRECTLY",
            "requirement": "Verified via contract metadata registry, cardinality ratios and timestamp/ID pattern checks.",
        },
        "LAK-010 (Conflicting Labels & Data-Quality Issues)": {
            "status": "SUPPORTED_DIRECTLY",
            "requirement": "Verified directly via duplicate key label consistency, censoring checks and missingness diagnostics.",
        },
        "LAK-011 (Covariate & Prevalence Shift)": {
            "status": "SUPPORTED_DIRECTLY",
            "requirement": "Verified directly via Population Stability Index (PSI), KS-test and categorical TVD across splits.",
        },
    }

    if not (has_feature_availability_metadata and has_fit_provenance and has_resampling_provenance):
        warnings.append(
            "External/raw CSV mode: Provenance-dependent checks (feature availability lags, pre-upload scaling/imputation, "
            "or pre-upload synthetic resampling) are marked UNVERIFIABLE_MISSING_PROVENANCE or HEURISTIC_ONLY. "
            "Uploaded data is never silently declared leakage-free."
        )

    sha256_digest = compute_dataframe_sha256(df) if len(errors) == 0 else ""

    return UploadValidationReport(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        row_count=n_rows,
        column_count=n_cols,
        file_size_bytes=file_size_bytes,
        dataset_sha256=sha256_digest,
        detected_numerical_cols=num_cols,
        detected_categorical_cols=cat_cols,
        missing_value_summary=missing_summary,
        target_distribution=target_dist,
        provenance_capabilities=provenance_matrix,
    )
