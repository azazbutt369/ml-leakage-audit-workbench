"""
Metrics, Structure-Aware Uncertainty, Calibration Diagnostics and Frozen Threshold Selection.
Implements account-cluster bootstrapping, separate pre/post-recalibration Brier/ECE metrics,
development-only threshold optimisation, cohort slice metrics and timed inference latency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from workbench.config import (
    CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_SEED,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
)
from workbench.pipelines import AuditedModelArtifact


PR_AUC_CONVENTION_NOTE: str = (
    "PR-AUC is computed via sklearn.metrics.average_precision_score: weighted mean of precisions "
    "achieved at each threshold, weighted by the increase in recall from the previous threshold "
    "(step-function rectangular integration, avoiding optimistic linear trapezoidal interpolation)."
)


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Brier score, Expected Calibration Error (ECE) and reliability curve bins."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    brier = float(brier_score_loss(y_arr, p_arr))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_arr, bin_edges[1:-1], right=True)

    bins_data: List[Dict[str, Any]] = []
    ece = 0.0
    n_total = max(1, len(y_arr))

    for b_idx in range(n_bins):
        mask = bin_indices == b_idx
        count = int(mask.sum())
        low_edge = float(bin_edges[b_idx])
        high_edge = float(bin_edges[b_idx + 1])
        if count > 0:
            mean_pred = float(p_arr[mask].mean())
            obs_freq = float(y_arr[mask].mean())
            abs_gap = abs(obs_freq - mean_pred)
            ece += (count / n_total) * abs_gap
            bins_data.append(
                {
                    "bin_index": b_idx,
                    "bin_lower": round(low_edge, 2),
                    "bin_upper": round(high_edge, 2),
                    "sample_count": count,
                    "mean_predicted_prob": round(mean_pred, 4),
                    "observed_positive_rate": round(obs_freq, 4),
                    "calibration_gap": round(abs_gap, 4),
                }
            )
        else:
            bins_data.append(
                {
                    "bin_index": b_idx,
                    "bin_lower": round(low_edge, 2),
                    "bin_upper": round(high_edge, 2),
                    "sample_count": 0,
                    "mean_predicted_prob": None,
                    "observed_positive_rate": None,
                    "calibration_gap": None,
                }
            )

    return {
        "brier_score": round(brier, 5),
        "expected_calibration_error_ece": round(float(ece), 5),
        "n_bins": n_bins,
        "reliability_bins": bins_data,
    }


def compute_classification_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    raw_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification metrics at a given probability threshold."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    y_pred = (p_arr >= threshold).astype(int)

    n_samples = int(len(y_arr))
    pos_count = int((y_arr == 1).sum())
    neg_count = int((y_arr == 0).sum())
    prevalence = float(pos_count / max(1, n_samples))

    precision = float(precision_score(y_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_arr, y_pred, zero_division=0))
    f1_pos = float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_arr, y_pred, average="macro", zero_division=0))

    both_classes_present = len(np.unique(y_arr)) == 2
    pr_auc = float(average_precision_score(y_arr, p_arr)) if both_classes_present else None
    roc_auc = float(roc_auc_score(y_arr, p_arr)) if both_classes_present else None

    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    cal_diag = compute_calibration_diagnostics(y_arr, p_arr, n_bins=10)
    raw_cal_diag = (
        compute_calibration_diagnostics(y_arr, raw_prob, n_bins=10)
        if raw_prob is not None
        else cal_diag
    )

    return {
        "sample_count": n_samples,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_prevalence": round(prevalence, 4),
        "threshold_applied": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_pos, 4),
        "macro_f1": round(macro_f1, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "pr_auc_convention": PR_AUC_CONVENTION_NOTE,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": cal_diag["brier_score"],
        "expected_calibration_error_ece": cal_diag["expected_calibration_error_ece"],
        "uncalibrated_brier_score": raw_cal_diag["brier_score"],
        "uncalibrated_ece": raw_cal_diag["expected_calibration_error_ece"],
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_positive_rate": round(float(fp / max(1, fp + tn)), 4),
            "false_negative_rate": round(float(fn / max(1, fn + tp)), 4),
        },
        "calibration_curve": cal_diag["reliability_bins"],
        "uncalibrated_curve": raw_cal_diag["reliability_bins"],
    }


def select_development_threshold_and_build_explorer(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    objective: str = "max_f1",
    hypothetical_intervention_cost_gbp: float = 55.0,
    hypothetical_retained_value_gbp: float = 420.0,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Explore candidate thresholds strictly on Development (Validation) predictions and
    freeze the optimal threshold prior to Test evaluation.
    Business cost figures are explicitly labelled as hypothetical user-configurable assumptions.
    """
    y_arr = np.asarray(y_val, dtype=int)
    p_arr = np.clip(np.asarray(prob_val, dtype=float), 1e-7, 1.0 - 1e-7)

    thresholds = np.round(np.linspace(0.05, 0.90, 35), 3)
    explorer_rows: List[Dict[str, Any]] = []

    best_thresh = 0.50
    best_obj_val = -np.inf

    for t in thresholds:
        preds = (p_arr >= t).astype(int)
        prec = float(precision_score(y_arr, preds, zero_division=0))
        rec = float(recall_score(y_arr, preds, zero_division=0))
        f1_val = float(f1_score(y_arr, preds, pos_label=1, zero_division=0))
        macro_f1 = float(f1_score(y_arr, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Hypothetical net utility relative to "no intervention" baseline:
        # Each flagged account (TP + FP) incurs hypothetical outreach/CSM cost;
        # each true positive receives an assumed expected save value.
        hypothetical_net_utility = float(
            tp * (hypothetical_retained_value_gbp - hypothetical_intervention_cost_gbp)
            - fp * hypothetical_intervention_cost_gbp
        )

        explorer_rows.append(
            {
                "threshold": float(t),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1_val, 4),
                "macro_f1": round(macro_f1, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "flagged_rate": round(float((tp + fp) / max(1, len(y_arr))), 4),
                "hypothetical_net_utility_gbp": round(hypothetical_net_utility, 2),
            }
        )

        score_for_selection = f1_val if objective == "max_f1" else hypothetical_net_utility
        if score_for_selection > best_obj_val:
            best_obj_val = score_for_selection
            best_thresh = float(t)

    selection_summary = {
        "selected_frozen_threshold": round(best_thresh, 4),
        "selection_partition": "development_validation_only",
        "selection_objective": objective,
        "best_objective_value_on_dev": round(float(best_obj_val), 4),
        "frozen_before_test_evaluation": True,
        "hypothetical_cost_assumptions_disclosure": (
            f"Hypothetical Cost Assumptions: Outreach cost = £{hypothetical_intervention_cost_gbp:.2f} per "
            f"flagged account; expected retained value per true positive = £{hypothetical_retained_value_gbp:.2f}. "
            "These are illustrative user-configurable assumptions, not verified commercial financial outcomes."
        ),
    }
    return best_thresh, selection_summary, explorer_rows


def compute_structure_aware_uncertainty(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    account_col: Optional[str] = "account_id",
    protocol_id: str = "protocol_b_temporal_purged",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Compute 95% bootstrap confidence intervals that respect the longitudinal account cluster structure.
    Rather than resampling individual rows independently (which understates variance when accounts
    have repeated snapshots), this resamples unique account_id clusters with replacement.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)

    pr_aucs: List[float] = []
    roc_aucs: List[float] = []
    f1s: List[float] = []
    briers: List[float] = []

    if account_col and account_col in df_eval.columns:
        groups = df_eval[account_col].astype(str).to_numpy()
        unique_accs = np.unique(groups)
        acc_to_indices = {acc: np.where(groups == acc)[0] for acc in unique_accs}
        n_clusters = len(unique_accs)
        resampling_method = (
            f"Account-Cluster Bootstrap ({n_bootstrap} resamples of {n_clusters} unique '{account_col}' "
            "clusters with replacement)"
        )

        for _ in range(n_bootstrap):
            sampled_accs = rng.choice(unique_accs, size=n_clusters, replace=True)
            boot_idx = np.concatenate([acc_to_indices[a] for a in sampled_accs])
            yb = y_arr[boot_idx]
            pb = p_arr[boot_idx]
            if len(np.unique(yb)) < 2:
                continue
            pr_aucs.append(float(average_precision_score(yb, pb)))
            roc_aucs.append(float(roc_auc_score(yb, pb)))
            f1s.append(float(f1_score(yb, (pb >= threshold).astype(int), pos_label=1, zero_division=0)))
            briers.append(float(brier_score_loss(yb, pb)))
    else:
        n_rows = len(y_arr)
        resampling_method = f"Row-Level Bootstrap ({n_bootstrap} resamples; no group column supplied)"
        for _ in range(n_bootstrap):
            boot_idx = rng.integers(0, n_rows, size=n_rows)
            yb = y_arr[boot_idx]
            pb = p_arr[boot_idx]
            if len(np.unique(yb)) < 2:
                continue
            pr_aucs.append(float(average_precision_score(yb, pb)))
            roc_aucs.append(float(roc_auc_score(yb, pb)))
            f1s.append(float(f1_score(yb, (pb >= threshold).astype(int), pos_label=1, zero_division=0)))
            briers.append(float(brier_score_loss(yb, pb)))

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low_q, high_q = alpha * 100.0, (1.0 - alpha) * 100.0

    def _ci_dict(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "std_error": None, "ci_95_lower": None, "ci_95_upper": None}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std_error": round(float(np.std(vals)), 4),
            "ci_95_lower": round(float(np.percentile(vals, low_q)), 4),
            "ci_95_upper": round(float(np.percentile(vals, high_q)), 4),
        }

    temporal_limitation_note = (
        "Dependence & Uncertainty Disclosure: Account-cluster bootstrap accounts for within-account "
        "correlation across repeated monthly snapshots. Under Protocol B (Future-Period Prediction), "
        "only 2 test monthly waves are observed, so macro-temporal regime variance across multi-year "
        "economic cycles cannot be estimated from this finite horizon. Furthermore, confidence intervals "
        "on a synthetic benchmark measure finite-sample simulation variance, not real-world generalisation."
    )

    return {
        "method": resampling_method,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_iterations": len(pr_aucs),
        "metrics_ci": {
            "pr_auc": _ci_dict(pr_aucs),
            "roc_auc": _ci_dict(roc_aucs),
            "f1_at_frozen_threshold": _ci_dict(f1s),
            "brier_score": _ci_dict(briers),
        },
        "methodological_limitation_disclosure": temporal_limitation_note,
    }


def evaluate_cohort_and_temporal_slices(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate classification performance across meaningful business cohorts and observation months."""
    work_df = df_eval.copy().reset_index(drop=True)
    work_df["__y_true"] = np.asarray(y_true, dtype=int)
    work_df["__y_prob"] = np.asarray(y_prob, dtype=float)
    work_df["__y_pred"] = (work_df["__y_prob"] >= threshold).astype(int)

    if "tenure_months" in work_df.columns:
        work_df["tenure_band"] = pd.cut(
            work_df["tenure_months"],
            bins=[-1, 3, 12, 24, 999],
            labels=["1-3m (Onboarding)", "4-12m (Early)", "13-24m (Established)", "25m+ (Mature)"],
        ).astype(str)

    slice_dimensions = [
        c
        for c in ["plan_tier", "contract_type", "tenure_band", "payment_method", "observation_date"]
        if c in work_df.columns
    ]

    slice_results: Dict[str, List[Dict[str, Any]]] = {}
    for col in slice_dimensions:
        rows: List[Dict[str, Any]] = []
        for val, grp in work_df.groupby(col, dropna=False):
            n_g = len(grp)
            if n_g < 10:
                continue
            yt = grp["__y_true"].to_numpy(dtype=int)
            yp = grp["__y_prob"].to_numpy(dtype=float)
            yd = grp["__y_pred"].to_numpy(dtype=int)
            both = len(np.unique(yt)) == 2
            cm = confusion_matrix(yt, yd, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            rows.append(
                {
                    "cohort_dimension": col,
                    "cohort_value": str(val),
                    "sample_count": int(n_g),
                    "unique_accounts": int(grp["account_id"].nunique()) if "account_id" in grp.columns else int(n_g),
                    "positive_prevalence": round(float(yt.mean()), 4),
                    "pr_auc": round(float(average_precision_score(yt, yp)), 4) if both else None,
                    "roc_auc": round(float(roc_auc_score(yt, yp)), 4) if both else None,
                    "precision": round(float(precision_score(yt, yd, zero_division=0)), 4),
                    "recall": round(float(recall_score(yt, yd, zero_division=0)), 4),
                    "f1": round(float(f1_score(yt, yd, pos_label=1, zero_division=0)), 4),
                    "brier_score": round(float(brier_score_loss(yt, yp)), 4),
                    "fp": fp,
                    "fn": fn,
                }
            )
        slice_results[col] = rows

    return slice_results


def measure_prediction_latency(
    artifact: AuditedModelArtifact,
    sample_df: pd.DataFrame,
    n_repeats: int = 25,
) -> Dict[str, Any]:
    """
    Measure CPU inference latency with a documented procedure:
    - 3 warm-up passes (excluded from timing)
    - n_repeats timed batch evaluations (up to 1,000 rows)
    - n_repeats timed single-row evaluations (1-row DataFrame, simulating real-time REST scoring)
    """
    batch_df = sample_df[artifact.feature_columns].iloc[: min(1000, len(sample_df))].copy()
    single_df = batch_df.iloc[:1].copy()
    batch_size = max(1, len(batch_df))

    # Warm-up passes
    for _ in range(3):
        _ = artifact.predict_calibrated_proba(batch_df)
        _ = artifact.predict_calibrated_proba(single_df)

    batch_times_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = artifact.predict_calibrated_proba(batch_df)
        batch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    single_times_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = artifact.predict_calibrated_proba(single_df)
        single_times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_batch_ms = float(np.mean(batch_times_ms))
    p95_batch_ms = float(np.percentile(batch_times_ms, 95))
    ms_per_1k_rows = (mean_batch_ms / batch_size) * 1000.0

    return {
        "batch_size_measured": batch_size,
        "repeats": n_repeats,
        "warmup_iterations": 3,
        "batch_latency_mean_ms": round(mean_batch_ms, 3),
        "batch_latency_p95_ms": round(p95_batch_ms, 3),
        "throughput_ms_per_1000_rows": round(ms_per_1k_rows, 3),
        "single_row_latency_mean_ms": round(float(np.mean(single_times_ms)), 3),
        "single_row_latency_p95_ms": round(float(np.percentile(single_times_ms, 95)), 3),
        "measurement_procedure": (
            f"Timed on local CPU via time.perf_counter() across {n_repeats} iterations following "
            f"3 warm-up passes, including full ColumnTransformer imputation/encoding and probability calibration."
        ),
    }


def evaluate_model_on_protocol(
    artifact: AuditedModelArtifact,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "churned_within_90d",
    account_col: str = "account_id",
    threshold_objective: str = "max_f1",
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Complete end-to-end evaluation of a fitted AuditedModelArtifact:
    1. Predict on Development (val_df).
    2. Select and freeze decision threshold tau* strictly on Development data.
    3. Evaluate on Frozen Test (test_df) using frozen tau* without any further tuning.
    4. Compute cluster-bootstrapped 95% CIs, cohort slices and CPU latency.
    """
    y_tr = train_df[target_col].to_numpy(dtype=int)
    y_vl = val_df[target_col].to_numpy(dtype=int)
    y_te = test_df[target_col].to_numpy(dtype=int)

    raw_tr = artifact.predict_raw_proba(train_df)
    cal_tr = artifact.predict_calibrated_proba(train_df)

    raw_vl = artifact.predict_raw_proba(val_df)
    cal_vl = artifact.predict_calibrated_proba(val_df)

    # Select and freeze decision threshold strictly on Development data
    frozen_tau, thresh_summary, thresh_explorer = select_development_threshold_and_build_explorer(
        y_val=y_vl,
        prob_val=cal_vl,
        objective=threshold_objective,
    )
    artifact.frozen_threshold = frozen_tau
    artifact.threshold_objective = threshold_objective
    artifact.threshold_selection_summary = thresh_summary

    # Now evaluate once on Frozen Test using the locked threshold
    raw_te = artifact.predict_raw_proba(test_df)
    cal_te = artifact.predict_calibrated_proba(test_df)

    train_metrics = compute_classification_metrics_at_threshold(y_tr, cal_tr, threshold=frozen_tau, raw_prob=raw_tr)
    val_metrics = compute_classification_metrics_at_threshold(y_vl, cal_vl, threshold=frozen_tau, raw_prob=raw_vl)
    test_metrics = compute_classification_metrics_at_threshold(y_te, cal_te, threshold=frozen_tau, raw_prob=raw_te)

    train_metrics["unique_accounts"] = int(train_df[account_col].nunique()) if account_col in train_df.columns else len(train_df)
    val_metrics["unique_accounts"] = int(val_df[account_col].nunique()) if account_col in val_df.columns else len(val_df)
    test_metrics["unique_accounts"] = int(test_df[account_col].nunique()) if account_col in test_df.columns else len(test_df)

    test_uncertainty = compute_structure_aware_uncertainty(
        df_eval=test_df,
        y_true=y_te,
        y_prob=cal_te,
        threshold=frozen_tau,
        account_col=account_col,
        protocol_id=artifact.protocol_id,
        seed=seed,
    )
    cohort_slices = evaluate_cohort_and_temporal_slices(
        df_eval=test_df,
        y_true=y_te,
        y_prob=cal_te,
        threshold=frozen_tau,
    )
    latency_report = measure_prediction_latency(artifact, test_df)

    return {
        "model_key": artifact.model_key,
        "model_display_name": artifact.model_display_name,
        "protocol_id": artifact.protocol_id,
        "selected_hyperparameters": artifact.selected_hyperparameters,
        "tuning_trials": artifact.tuning_trials,
        "selection_criterion": artifact.selection_criterion,
        "calibrator_method": artifact.calibrator_method,
        "calibration_provenance": artifact.calibration_provenance,
        "frozen_threshold": frozen_tau,
        "threshold_selection_summary": thresh_summary,
        "dev_threshold_explorer": thresh_explorer,
        "metrics": {
            "train": train_metrics,
            "val_development": val_metrics,
            "test_frozen": test_metrics,
        },
        "test_uncertainty": test_uncertainty,
        "test_cohort_slices": cohort_slices,
        "latency_benchmark": latency_report,
        "synthetic_benchmark_disclosure": SYNTHETIC_BENCHMARK_DISCLOSURE,
    }
