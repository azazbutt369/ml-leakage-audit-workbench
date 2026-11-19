"""
Explainability, Missing/Unknown-Category Diagnostics and Error Analysis Engine.
Provides permutation importance (computed on development data), linear coefficients,
model-faithful local marginal attributions against a documented training reference summary,
and inspectable False Positive / False Negative customer snapshot records.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from workbench.config import CAUSAL_ATTRIBUTION_DISCLOSURE, DEFAULT_SEED
from workbench.pipelines import AuditedModelArtifact


def compute_global_feature_importance(
    artifact: AuditedModelArtifact,
    val_df: pd.DataFrame,
    target_col: str = "churned_within_90d",
    n_repeats: int = 8,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Compute global feature importance on the Development (val_df) partition using
    permutation importance (scoring = 'average_precision' / PR-AUC).
    For Logistic Regression, also extracts standardised model coefficients.
    """
    X_val = val_df[artifact.feature_columns]
    y_val = val_df[target_col].to_numpy(dtype=int)

    perm_res = permutation_importance(
        artifact.pipeline,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
    )

    rows: List[Dict[str, Any]] = []
    for idx, col in enumerate(artifact.feature_columns):
        rows.append(
            {
                "feature": col,
                "dtype": "numerical" if col in artifact.numerical_features else "categorical",
                "mean_pr_auc_drop": round(float(perm_res.importances_mean[idx]), 5),
                "std_pr_auc_drop": round(float(perm_res.importances_std[idx]), 5),
            }
        )

    rows.sort(key=lambda r: r["mean_pr_auc_drop"], reverse=True)

    # Extract linear coefficients if the underlying classifier is LogisticRegression
    linear_coefficients: List[Dict[str, Any]] = []
    clf = artifact.pipeline.named_steps["classifier"]
    preproc = artifact.pipeline.named_steps["preprocessor"]
    if hasattr(clf, "coef_") and hasattr(preproc, "get_feature_names_out"):
        feat_names_out = [str(f) for f in preproc.get_feature_names_out()]
        coefs = clf.coef_[0]
        for fname, cval in zip(feat_names_out, coefs):
            linear_coefficients.append(
                {
                    "transformed_feature": fname,
                    "coefficient_log_odds": round(float(cval), 5),
                    "abs_coefficient": round(abs(float(cval)), 5),
                }
            )
        linear_coefficients.sort(key=lambda r: r["abs_coefficient"], reverse=True)

    return {
        "method": (
            f"Permutation Importance on Development Partition (Metric: Average Precision / PR-AUC, "
            f"{n_repeats} repeats, seed={seed})"
        ),
        "reference_partition": "val_development (Test set untouched for model revision)",
        "non_causal_disclosure": CAUSAL_ATTRIBUTION_DISCLOSURE,
        "feature_importances": rows,
        "linear_coefficients_top20": linear_coefficients[:20],
    }


def compute_local_instance_attribution(
    artifact: AuditedModelArtifact,
    instance_row: pd.Series,
    train_reference_df: pd.DataFrame,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Generate model-faithful local feature attributions for a single snapshot by replacing each
    feature one at a time with the training-set reference baseline (median for numericals, mode
    for categoricals) and measuring the exact probability change delta_p = p(x) - p(x_{-j, ref}).
    """
    base_df = pd.DataFrame([instance_row[artifact.feature_columns].to_dict()])
    base_prob = float(artifact.predict_calibrated_proba(base_df)[0])

    ref_values: Dict[str, Any] = {}
    for col in artifact.numerical_features:
        ref_values[col] = float(train_reference_df[col].median())
    for col in artifact.categorical_features:
        mode_ser = train_reference_df[col].dropna().mode()
        ref_values[col] = str(mode_ser.iloc[0]) if len(mode_ser) > 0 else "__MISSING__"

    attributions: List[Dict[str, Any]] = []
    for col in artifact.feature_columns:
        perturbed_df = base_df.copy()
        perturbed_df[col] = ref_values[col]
        perturbed_prob = float(artifact.predict_calibrated_proba(perturbed_df)[0])
        delta_p = base_prob - perturbed_prob
        val_raw = instance_row[col]
        if pd.isna(val_raw):
            obs_clean = None
        elif isinstance(val_raw, (int, np.integer)):
            obs_clean = int(val_raw)
        elif isinstance(val_raw, (float, np.floating)):
            obs_clean = round(float(val_raw), 4)
        else:
            obs_clean = str(val_raw)

        attributions.append(
            {
                "feature": col,
                "observed_value": obs_clean,
                "training_reference_value": round(ref_values[col], 4) if isinstance(ref_values[col], float) else ref_values[col],
                "probability_contribution_delta": round(float(delta_p), 4),
                "direction": "increases_churn_risk" if delta_p >= 0 else "decreases_churn_risk",
            }
        )

    attributions.sort(key=lambda r: abs(r["probability_contribution_delta"]), reverse=True)
    return attributions[:top_k]


def inspect_false_positives_and_false_negatives(
    artifact: AuditedModelArtifact,
    eval_df: pd.DataFrame,
    train_reference_df: pd.DataFrame,
    target_col: str = "churned_within_90d",
    max_examples_per_type: int = 10,
) -> Dict[str, Any]:
    """
    Extract inspectable False Positive (FP) and False Negative (FN) customer snapshots
    at the frozen threshold, enriched with local marginal attributions and error taxonomy tags.
    """
    work_df = eval_df.copy().reset_index(drop=True)
    probs = artifact.predict_calibrated_proba(work_df)
    tau = artifact.frozen_threshold
    preds = (probs >= tau).astype(int)
    y_true = work_df[target_col].to_numpy(dtype=int)

    work_df["predicted_prob"] = np.round(probs, 4)
    work_df["predicted_label"] = preds
    work_df["actual_label"] = y_true
    work_df["threshold_margin"] = np.round(probs - tau, 4)

    fp_df = work_df.loc[(work_df["predicted_label"] == 1) & (work_df["actual_label"] == 0)].sort_values(
        by="predicted_prob", ascending=False
    )
    fn_df = work_df.loc[(work_df["predicted_label"] == 0) & (work_df["actual_label"] == 1)].sort_values(
        by="predicted_prob", ascending=True
    )

    def _build_records(sub_df: pd.DataFrame, error_type: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for _, row in sub_df.head(max_examples_per_type).iterrows():
            local_attr = compute_local_instance_attribution(
                artifact, row, train_reference_df=train_reference_df, top_k=5
            )
            records.append(
                {
                    "error_type": error_type,
                    "snapshot_id": str(row.get("snapshot_id", "N/A")),
                    "account_id": str(row.get("account_id", "N/A")),
                    "observation_date": str(row.get("observation_date", "N/A")),
                    "actual_churn_90d": int(row["actual_label"]),
                    "predicted_prob": float(row["predicted_prob"]),
                    "frozen_threshold": round(float(tau), 4),
                    "threshold_margin": float(row["threshold_margin"]),
                    "plan_tier": str(row.get("plan_tier", "N/A")),
                    "contract_type": str(row.get("contract_type", "N/A")),
                    "mrr_gbp": float(row.get("mrr_gbp", 0.0)),
                    "tenure_months": int(row.get("tenure_months", 0)),
                    "active_seats_ratio_30d": float(row.get("active_seats_ratio_30d", 0.0)),
                    "urgent_tickets_90d": int(row.get("urgent_tickets_90d", 0)),
                    "dunning_events_6m": int(row.get("dunning_events_6m", 0)),
                    "top_local_attributions": local_attr,
                }
            )
        return records

    # Compute threshold-sensitive error shifts across 5 representative thresholds
    threshold_error_sensitivity: List[Dict[str, Any]] = []
    for cand_t in [0.20, 0.30, round(float(tau), 3), 0.50, 0.65]:
        c_preds = (probs >= cand_t).astype(int)
        fp_mask = (c_preds == 1) & (y_true == 0)
        fn_mask = (c_preds == 0) & (y_true == 1)
        threshold_error_sensitivity.append(
            {
                "threshold": cand_t,
                "is_frozen_threshold": abs(cand_t - tau) < 1e-4,
                "false_positives_count": int(fp_mask.sum()),
                "false_negatives_count": int(fn_mask.sum()),
                "total_errors": int(fp_mask.sum() + fn_mask.sum()),
                "fp_mean_mrr_gbp": round(float(work_df.loc[fp_mask, "mrr_gbp"].mean()), 2)
                if fp_mask.any() and "mrr_gbp" in work_df.columns
                else 0.0,
                "fn_mean_mrr_gbp": round(float(work_df.loc[fn_mask, "mrr_gbp"].mean()), 2)
                if fn_mask.any() and "mrr_gbp" in work_df.columns
                else 0.0,
            }
        )

    return {
        "frozen_threshold_used": round(float(tau), 4),
        "total_false_positives": int(len(fp_df)),
        "total_false_negatives": int(len(fn_df)),
        "reference_background_documentation": (
            "Instance-level attributions measure marginal probability change Δp when replacing each "
            "feature value with the Training partition baseline (median for numerical predictors, mode "
            "for categorical predictors)."
        ),
        "non_causal_disclosure": CAUSAL_ATTRIBUTION_DISCLOSURE,
        "false_positive_examples": _build_records(fp_df, "FALSE_POSITIVE"),
        "false_negative_examples": _build_records(fn_df, "FALSE_NEGATIVE"),
        "threshold_sensitive_error_table": threshold_error_sensitivity,
    }


def compute_missing_and_unknown_category_diagnostics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    """
    Inspect missing-value rates across Train/Val/Test and identify any unseen categorical levels
    appearing in Validation or Test that were absent from Training.
    """
    missing_rows: List[Dict[str, Any]] = []
    for col in list(numerical_features) + list(categorical_features):
        tr_miss = float(train_df[col].isna().mean()) if col in train_df.columns else 0.0
        vl_miss = float(val_df[col].isna().mean()) if col in val_df.columns and len(val_df) > 0 else 0.0
        te_miss = float(test_df[col].isna().mean()) if col in test_df.columns and len(test_df) > 0 else 0.0
        if tr_miss > 0 or vl_miss > 0 or te_miss > 0:
            missing_rows.append(
                {
                    "feature": col,
                    "dtype": "numerical" if col in numerical_features else "categorical",
                    "train_missing_rate": round(tr_miss, 4),
                    "val_missing_rate": round(vl_miss, 4),
                    "test_missing_rate": round(te_miss, 4),
                    "max_split_gap": round(max(abs(te_miss - tr_miss), abs(vl_miss - tr_miss)), 4),
                }
            )

    unknown_category_rows: List[Dict[str, Any]] = []
    for col in categorical_features:
        if col not in train_df.columns:
            continue
        tr_cats = set(train_df[col].dropna().astype(str).unique())
        vl_cats = set(val_df[col].dropna().astype(str).unique()) if len(val_df) > 0 else set()
        te_cats = set(test_df[col].dropna().astype(str).unique()) if len(test_df) > 0 else set()

        unseen_vl = sorted(list(vl_cats - tr_cats))
        unseen_te = sorted(list(te_cats - tr_cats))
        unseen_te_count = (
            int(test_df[col].dropna().astype(str).isin(unseen_te).sum())
            if unseen_te and len(test_df) > 0
            else 0
        )
        unknown_category_rows.append(
            {
                "categorical_feature": col,
                "train_unique_levels": len(tr_cats),
                "unseen_levels_in_val": unseen_vl,
                "unseen_levels_in_test": unseen_te,
                "test_rows_with_unseen_levels": unseen_te_count,
            }
        )

    return {
        "missing_value_diagnostics": missing_rows,
        "unknown_category_diagnostics": unknown_category_rows,
    }
