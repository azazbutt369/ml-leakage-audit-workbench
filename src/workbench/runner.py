"""
Master Experiment & Audit Orchestrator.
Executes the full clean benchmark workflow across Protocol B (Primary: Future-Period Purged)
and Protocol A (Secondary: New-Account Group Holdout), runs the Leakage-Audit Engine,
evaluates the 7 controlled leakage fixtures and persists all artefacts to SQLite and JSON/CSV.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from workbench.config import (
    CLEAN_DATA_DIR,
    DEFAULT_SEED,
    EXPORTS_DIR,
    FIXTURES_DIR,
    REPORTS_DIR,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
    ensure_directories,
)
from workbench.contract import build_default_saas_contract
from workbench.data_generator import generate_saas_churn_dataset
from workbench.evaluation import evaluate_model_on_protocol
from workbench.explainability import (
    compute_global_feature_importance,
    compute_missing_and_unknown_category_diagnostics,
    inspect_false_positives_and_false_negatives,
)
from workbench.fixtures import generate_and_audit_controlled_fixtures
from workbench.leakage_audit import run_leakage_audit
from workbench.pipelines import fit_and_tune_candidate_models
from workbench.splits import (
    PartitionBundle,
    build_inner_cv_splits,
    split_protocol_a_new_account_generalisation,
    split_protocol_b_future_period_prediction,
)
from workbench.storage import (
    init_sqlite_db,
    persist_audit_report,
    persist_experiment_bundle,
    save_model_artifact_and_predictions,
)


def execute_protocol_workflow(
    partition_bundle: PartitionBundle,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Run training, leakage audit, evaluation, explainability and persistence for a single protocol."""
    contract = build_default_saas_contract()
    protocol_id = partition_bundle.protocol_id
    run_id = f"RUN-{protocol_id.upper()}-{seed}"

    inner_cv = build_inner_cv_splits(
        train_df=partition_bundle.train_df,
        protocol_id=protocol_id,
        account_col=contract.account_id_col,
        date_col=contract.timestamp_col,
        target_col=contract.target_col,
        n_splits=4,
        seed=seed,
    )

    artifacts = fit_and_tune_candidate_models(
        train_df=partition_bundle.train_df,
        inner_cv_splits=inner_cv,
        numerical_features=contract.numerical_features,
        categorical_features=contract.categorical_features,
        target_col=contract.target_col,
        protocol_id=protocol_id,
        account_col=contract.account_id_col,
        date_col=contract.timestamp_col,
        outcome_end_col=contract.outcome_window_end_col,
        seed=seed,
    )

    # Run the Leakage-Audit Engine using the primary tree model's fit provenance
    primary_art = artifacts["hist_gradient_boosting"]
    audit_report = run_leakage_audit(
        train_df=partition_bundle.train_df,
        val_df=partition_bundle.val_df,
        test_df=partition_bundle.test_df,
        feature_cols=contract.all_predictor_cols,
        target_col=contract.target_col,
        account_col=contract.account_id_col,
        date_col=contract.timestamp_col,
        outcome_end_col=contract.outcome_window_end_col,
        protocol_id=protocol_id,
        allow_returning_accounts=partition_bundle.allow_returning_accounts,
        contract=contract,
        has_lineage_metadata=True,
        fit_provenance=primary_art.fit_provenance,
        cv_fold_provenance=primary_art.cv_fold_provenance,
        resampling_provenance=primary_art.resampling_provenance,
        dataset_name=f"B2B_SaaS_Clean_Benchmark__{protocol_id}",
    )
    persist_audit_report(audit_report)

    missing_and_unknown_diag = compute_missing_and_unknown_category_diagnostics(
        train_df=partition_bundle.train_df,
        val_df=partition_bundle.val_df,
        test_df=partition_bundle.test_df,
        numerical_features=contract.numerical_features,
        categorical_features=contract.categorical_features,
    )

    model_evaluations: Dict[str, Any] = {}
    model_manifests: Dict[str, Any] = {}

    for m_key, art in artifacts.items():
        eval_res = evaluate_model_on_protocol(
            artifact=art,
            train_df=partition_bundle.train_df,
            val_df=partition_bundle.val_df,
            test_df=partition_bundle.test_df,
            target_col=contract.target_col,
            account_col=contract.account_id_col,
            threshold_objective="max_f1",
            seed=seed,
        )
        # Global importance on Development (val) set & error inspection on Test set
        if m_key != "class_prior_baseline":
            eval_res["global_explainability"] = compute_global_feature_importance(
                artifact=art,
                val_df=partition_bundle.val_df,
                target_col=contract.target_col,
                n_repeats=8,
                seed=seed,
            )
            eval_res["error_analysis_test"] = inspect_false_positives_and_false_negatives(
                artifact=art,
                eval_df=partition_bundle.test_df,
                train_reference_df=partition_bundle.train_df,
                target_col=contract.target_col,
                max_examples_per_type=8,
            )
        else:
            eval_res["global_explainability"] = None
            eval_res["error_analysis_test"] = None

        manifest_meta = save_model_artifact_and_predictions(
            artifact=art,
            eval_summary=eval_res,
            test_df=partition_bundle.test_df,
            run_id=run_id,
            target_col=contract.target_col,
        )
        eval_res["artifact_manifest"] = manifest_meta
        eval_res["fit_provenance_summary"] = {
            "stage_name": art.fit_provenance.stage_name,
            "fitted_row_count": art.fit_provenance.fitted_row_count,
            "fitted_min_observation_date": art.fit_provenance.fitted_min_observation_date,
            "fitted_max_observation_date": art.fit_provenance.fitted_max_observation_date,
            "fitted_max_outcome_window_end": art.fit_provenance.fitted_max_outcome_window_end,
            "unknown_category_policy": art.fit_provenance.unknown_category_policy,
            "learned_summary_stats": art.fit_provenance.learned_summary_stats,
        }
        model_evaluations[m_key] = eval_res
        model_manifests[m_key] = manifest_meta

    protocol_bundle = {
        "run_id": run_id,
        "protocol_id": protocol_id,
        "protocol_name": partition_bundle.protocol_name,
        "is_primary_benchmark": partition_bundle.is_primary_benchmark,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest": partition_bundle.manifest,
        "audit_report": audit_report.model_dump(),
        "missing_and_unknown_diagnostics": missing_and_unknown_diag,
        "models": model_evaluations,
        "artifact_manifests": model_manifests,
        "synthetic_benchmark_disclosure": SYNTHETIC_BENCHMARK_DISCLOSURE,
    }

    persist_experiment_bundle(
        run_id=run_id,
        protocol_id=protocol_id,
        dataset_name="SYNTH-B2B-SAAS-CHURN-V1",
        bundle_dict=protocol_bundle,
    )
    return protocol_bundle


def run_complete_workbench_pipeline(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """
    Execute the complete, deterministic build & evaluation pipeline:
      1. Generate synthetic B2B SaaS churn dataset (~15,000 snapshots) and data dictionary.
      2. Execute Primary Benchmark: Protocol B (Future-Period Prediction with 90d Outcome Purging).
      3. Execute Secondary Benchmark: Protocol A (New-Account Generalisation via Disjoint Group Split).
      4. Build and audit all 7 Controlled Leakage & Negative-Control Fixtures (FIX-01 to FIX-07).
      5. Export summary comparison tables to JSON and CSV.
    """
    ensure_directories()
    init_sqlite_db()

    full_df, modelling_df, gen_manifest = generate_saas_churn_dataset(
        seed=seed, n_accounts=1650, save_to_disk=True
    )

    proto_b_splits = split_protocol_b_future_period_prediction(full_df, save_manifest_dir=CLEAN_DATA_DIR)
    proto_a_splits = split_protocol_a_new_account_generalisation(
        full_df, seed=seed, save_manifest_dir=CLEAN_DATA_DIR
    )

    bundle_b = execute_protocol_workflow(proto_b_splits, seed=seed)
    bundle_a = execute_protocol_workflow(proto_a_splits, seed=seed)

    fixtures_summary = generate_and_audit_controlled_fixtures(
        modelling_df=modelling_df, seed=seed, output_dir=FIXTURES_DIR
    )

    # Build a flat comparison table across both protocols and all 3 models for CSV export
    comparison_rows = []
    for bundle in (bundle_b, bundle_a):
        p_id = bundle["protocol_id"]
        p_role = "Primary (Future-Period Purged)" if bundle["is_primary_benchmark"] else "Secondary (New-Account Group)"
        for m_key, m_eval in bundle["models"].items():
            dev_m = m_eval["metrics"]["val_development"]
            test_m = m_eval["metrics"]["test_frozen"]
            ci_pr = m_eval["test_uncertainty"]["metrics_ci"]["pr_auc"]
            ci_roc = m_eval["test_uncertainty"]["metrics_ci"]["roc_auc"]
            lat = m_eval["latency_benchmark"]
            comparison_rows.append(
                {
                    "protocol_id": p_id,
                    "protocol_role": p_role,
                    "model_key": m_key,
                    "model_display_name": m_eval["model_display_name"],
                    "frozen_threshold": m_eval["frozen_threshold"],
                    "dev_pr_auc": dev_m["pr_auc"],
                    "dev_roc_auc": dev_m["roc_auc"],
                    "dev_f1": dev_m["f1"],
                    "dev_brier": dev_m["brier_score"],
                    "test_pr_auc": test_m["pr_auc"],
                    "test_pr_auc_ci95_low": ci_pr["ci_95_lower"],
                    "test_pr_auc_ci95_high": ci_pr["ci_95_upper"],
                    "test_roc_auc": test_m["roc_auc"],
                    "test_roc_auc_ci95_low": ci_roc["ci_95_lower"],
                    "test_roc_auc_ci95_high": ci_roc["ci_95_upper"],
                    "test_precision": test_m["precision"],
                    "test_recall": test_m["recall"],
                    "test_f1": test_m["f1"],
                    "test_macro_f1": test_m["macro_f1"],
                    "test_brier_calibrated": test_m["brier_score"],
                    "test_brier_uncalibrated": test_m["uncalibrated_brier_score"],
                    "test_ece_calibrated": test_m["expected_calibration_error_ece"],
                    "test_tp": test_m["confusion_matrix"]["tp"],
                    "test_fp": test_m["confusion_matrix"]["fp"],
                    "test_tn": test_m["confusion_matrix"]["tn"],
                    "test_fn": test_m["confusion_matrix"]["fn"],
                    "latency_ms_per_1000_rows": lat["throughput_ms_per_1000_rows"],
                    "single_row_latency_ms": lat["single_row_latency_mean_ms"],
                }
            )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(EXPORTS_DIR / "benchmark_model_comparison.csv", index=False)

    master_summary = {
        "workbench_version": "1.0.0",
        "author": "Azaz Butt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_manifest": gen_manifest,
        "primary_protocol_b": bundle_b,
        "secondary_protocol_a": bundle_a,
        "controlled_fixtures_summary": fixtures_summary,
        "comparison_table": comparison_rows,
    }
    (REPORTS_DIR / "master_workbench_summary.json").write_text(
        json.dumps(master_summary, indent=2), encoding="utf-8"
    )
    return master_summary
