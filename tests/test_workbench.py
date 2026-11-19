"""
Comprehensive 13-Family Automated Verification & API Smoke Test Suite
for the Tabular ML and Leakage-Audit Workbench.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from workbench.api import app
from workbench.config import (
    CLEAN_DATA_DIR,
    DB_PATH,
    DEFAULT_SEED,
    EXPORTS_DIR,
    FIXTURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from workbench.contract import (
    build_default_saas_contract,
    validate_uploaded_dataframe,
)
from workbench.evaluation import (
    compute_classification_metrics_at_threshold,
    select_development_threshold_and_build_explorer,
)
from workbench.pipelines import (
    build_preprocessor,
    resample_training_fold_only,
)
from workbench.runner import run_complete_workbench_pipeline
from workbench.splits import (
    split_protocol_a_new_account_generalisation,
    split_protocol_b_future_period_prediction,
)
from workbench.storage import (
    load_verified_model_artifact,
    run_batch_prediction,
)


@pytest.fixture(scope="module")
def master_bundle() -> dict:
    """Ensure the complete benchmark pipeline has executed and return its summary."""
    summary_path = REPORTS_DIR / "master_workbench_summary.json"
    if not summary_path.exists():
        return run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def datasets(master_bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = pd.read_csv(CLEAN_DATA_DIR / "saas_churn_snapshots_full.csv")
    mod_df = pd.read_csv(CLEAN_DATA_DIR / "saas_churn_snapshots_modelling.csv")
    return full_df, mod_df


# -----------------------------------------------------------------------------
# 1. Disjoint Group Partitions (Protocol A)
# -----------------------------------------------------------------------------
def test_01_disjoint_group_partitions(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    full_df, _ = datasets
    bundle_a = split_protocol_a_new_account_generalisation(full_df, seed=DEFAULT_SEED, save_manifest_dir=None)

    tr_accs = set(bundle_a.train_df["account_id"].unique())
    vl_accs = set(bundle_a.val_df["account_id"].unique())
    te_accs = set(bundle_a.test_df["account_id"].unique())

    assert len(tr_accs) > 500
    assert len(vl_accs) > 100
    assert len(te_accs) > 100
    assert tr_accs.isdisjoint(vl_accs), "Train and Val accounts must be strictly disjoint under Protocol A"
    assert tr_accs.isdisjoint(te_accs), "Train and Test accounts must be strictly disjoint under Protocol A"
    assert vl_accs.isdisjoint(te_accs), "Val and Test accounts must be strictly disjoint under Protocol A"


# -----------------------------------------------------------------------------
# 2. Temporal Ordering and 90-Day Outcome-Window Purging (Protocol B)
# -----------------------------------------------------------------------------
def test_02_temporal_ordering_and_outcome_window_purging(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    full_df, _ = datasets
    bundle_b = split_protocol_b_future_period_prediction(full_df, save_manifest_dir=None)

    # Verify right-censored snapshots are excluded from all active splits
    for split_df in (bundle_b.train_df, bundle_b.val_df, bundle_b.test_df):
        assert not split_df["is_censored_outcome"].any()
        assert split_df["churned_within_90d"].notna().all()

    assert len(bundle_b.censored_df) > 500, "Right-censored late waves must be identified and held out"
    assert len(bundle_b.purged_df) > 1000, "Overlap outcome-window snapshots must be purged"

    tr_max_obs = pd.to_datetime(bundle_b.train_df["observation_date"]).max()
    tr_max_out = pd.to_datetime(bundle_b.train_df["outcome_window_end"]).max()
    vl_min_obs = pd.to_datetime(bundle_b.val_df["observation_date"]).min()
    vl_max_out = pd.to_datetime(bundle_b.val_df["outcome_window_end"]).max()
    te_min_obs = pd.to_datetime(bundle_b.test_df["observation_date"]).min()
    te_max_out = pd.to_datetime(bundle_b.test_df["outcome_window_end"]).max()

    assert tr_max_obs < vl_min_obs
    assert tr_max_out <= vl_min_obs, "Training 90d outcome windows must not bleed past Validation start date"
    assert vl_max_out <= te_min_obs, "Validation 90d outcome windows must not bleed past Test start date"
    assert te_max_out <= pd.Timestamp("2025-06-30"), "Test outcomes must finish on or before dataset cutoff"


# -----------------------------------------------------------------------------
# 3. No Target or Forbidden Metadata in Predictors
# -----------------------------------------------------------------------------
def test_03_no_target_or_forbidden_metadata_in_predictors(master_bundle: dict) -> None:
    contract = build_default_saas_contract()
    forbidden = set(contract.forbidden_predictor_cols)
    predictors = set(contract.all_predictor_cols)

    assert predictors.isdisjoint(forbidden)
    for proto_key in ("primary_protocol_b", "secondary_protocol_a"):
        for m_key in ("class_prior_baseline", "logistic_regression", "hist_gradient_boosting"):
            art = load_verified_model_artifact(master_bundle[proto_key]["protocol_id"], m_key)
            assert set(art.feature_columns) == predictors
            assert set(art.feature_columns).isdisjoint(forbidden)


# -----------------------------------------------------------------------------
# 4. Train-Only Preprocessing Verified via FitProvenanceRecord
# -----------------------------------------------------------------------------
def test_04_train_only_preprocessing_provenance(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    full_df, _ = datasets
    bundle_b = split_protocol_b_future_period_prediction(full_df, save_manifest_dir=None)
    art = load_verified_model_artifact("protocol_b_temporal_purged", "hist_gradient_boosting")

    train_hashes = set(bundle_b.train_df["row_hash"].astype(str))
    val_hashes = set(bundle_b.val_df["row_hash"].astype(str))
    test_hashes = set(bundle_b.test_df["row_hash"].astype(str))
    fitted_hashes = set(art.fit_provenance.fitted_row_hashes)

    assert fitted_hashes == train_hashes
    assert fitted_hashes.isdisjoint(val_hashes), "No validation rows may be used during preprocessor.fit()"
    assert fitted_hashes.isdisjoint(test_hashes), "No test rows may be used during preprocessor.fit()"

    # Also verify every inner cross-validation fold fitted exclusively on a subset of train_hashes
    assert len(art.cv_fold_provenance) >= 2
    for fold_prov in art.cv_fold_provenance:
        f_set = set(fold_prov.fitted_row_hashes)
        assert f_set.issubset(train_hashes)
        assert f_set.isdisjoint(val_hashes)
        assert f_set.isdisjoint(test_hashes)


# -----------------------------------------------------------------------------
# 5. Fold-Isolated Resampling
# -----------------------------------------------------------------------------
def test_05_fold_isolated_resampling(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, mod_df = datasets
    contract = build_default_saas_contract()
    fold_train = mod_df.iloc[:400].copy().reset_index(drop=True)
    fold_val = mod_df.iloc[400:600].copy().reset_index(drop=True)

    tr_hashes = fold_train["row_hash"].astype(str).tolist()
    val_hashes = set(fold_val["row_hash"].astype(str).tolist())

    X_res, y_res, res_prov = resample_training_fold_only(
        X_train_df=fold_train[contract.all_predictor_cols + ["row_hash"]],
        y_train=fold_train["churned_within_90d"],
        row_hashes=tr_hashes,
        target_minority_ratio=0.40,
        seed=DEFAULT_SEED,
    )
    assert res_prov.applied is True
    assert res_prov.applied_strictly_inside_train_boundary is True
    assert res_prov.val_or_test_rows_touched == 0
    assert len(X_res) > len(fold_train)
    assert abs(float(y_res.mean()) - 0.40) < 0.02
    assert set(X_res["row_hash"].astype(str)).isdisjoint(val_hashes)


# -----------------------------------------------------------------------------
# 6. Unknown Categorical Values & Missing Input Handling in Preprocessors
# -----------------------------------------------------------------------------
def test_06_unknown_categorical_and_missing_value_handling(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, mod_df = datasets
    contract = build_default_saas_contract()
    sample_train = mod_df.iloc[:500].copy()
    sample_test = mod_df.iloc[500:525].copy()

    # Inject unseen categorical levels and missing numericals into sample_test
    sample_test.loc[sample_test.index[:5], "plan_tier"] = "UnseenQuantumTier"
    sample_test.loc[sample_test.index[5:10], "industry_sector"] = "SpaceMining_Unseen"
    sample_test.loc[sample_test.index[:8], "mrr_gbp"] = np.nan
    sample_test.loc[sample_test.index[:8], "payment_method"] = None

    for m_type in ("linear_or_dummy", "tree"):
        preproc, _ = build_preprocessor(
            contract.numerical_features, contract.categorical_features, model_type=m_type
        )
        preproc.fit(sample_train[contract.all_predictor_cols])
        transformed = preproc.transform(sample_test[contract.all_predictor_cols])
        assert transformed.shape[0] == len(sample_test)
        assert not np.isnan(transformed).any(), f"Transformed matrix for {m_type} must contain zero NaNs"


# -----------------------------------------------------------------------------
# 7. Missing Features and Invalid CSV Inputs Rejection
# -----------------------------------------------------------------------------
def test_07_missing_features_and_invalid_csv_inputs(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, mod_df = datasets
    art = load_verified_model_artifact("protocol_b_temporal_purged", "hist_gradient_boosting")

    # 7a. Missing required predictor column during batch inference
    incomplete_df = mod_df[art.feature_columns[:-2]].head(10).copy()
    with pytest.raises(ValueError, match="missing required predictor columns"):
        run_batch_prediction(art, incomplete_df)

    # 7b. Non-numeric string injected into numerical predictor column
    corrupt_num_df = mod_df[art.feature_columns].head(10).copy()
    corrupt_num_df["mrr_gbp"] = "INVALID_NUMBER"
    with pytest.raises(ValueError, match="non-numeric string"):
        run_batch_prediction(art, corrupt_num_df)

    # 7c. Invalid non-binary target values in uploaded CSV
    bad_target_df = mod_df.head(80).copy()
    bad_target_df["churned_within_90d"] = np.random.choice([0, 1, 2, 5], size=len(bad_target_df))
    rep = validate_uploaded_dataframe(bad_target_df, target_col="churned_within_90d")
    assert rep.is_valid is False
    assert any("non-binary" in e for e in rep.errors)


# -----------------------------------------------------------------------------
# 8. Metric Calculations and Dev-Only Threshold Selection
# -----------------------------------------------------------------------------
def test_08_metric_calculations_and_threshold_selection() -> None:
    y_val = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1] * 20, dtype=int)
    p_val = np.array([0.05, 0.12, 0.18, 0.25, 0.42, 0.31, 0.68, 0.82, 0.15, 0.74] * 20, dtype=float)

    tau, summary, explorer = select_development_threshold_and_build_explorer(
        y_val=y_val, prob_val=p_val, objective="max_f1"
    )
    assert 0.05 <= tau <= 0.90
    assert summary["selection_partition"] == "development_validation_only"
    assert summary["frozen_before_test_evaluation"] is True
    assert len(explorer) == 35

    m = compute_classification_metrics_at_threshold(y_val, p_val, threshold=tau)
    assert m["pr_auc"] == round(float(average_precision_score(y_val, p_val)), 4)
    assert m["roc_auc"] == round(float(roc_auc_score(y_val, p_val)), 4)
    assert m["brier_score"] == round(float(brier_score_loss(y_val, p_val)), 5)
    cm = m["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == len(y_val)


# -----------------------------------------------------------------------------
# 9. Model Saving / Reloading with Matching Predictions & Tamper Detection
# -----------------------------------------------------------------------------
def test_09_model_saving_loading_and_prediction_reproducibility(
    datasets: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, mod_df = datasets
    sample_df = mod_df.tail(150).copy()

    for proto_id in ("protocol_b_temporal_purged", "protocol_a_group_holdout"):
        for m_key in ("class_prior_baseline", "logistic_regression", "hist_gradient_boosting"):
            loaded = load_verified_model_artifact(proto_id, m_key)
            p1 = loaded.predict_calibrated_proba(sample_df)
            p2 = loaded.predict_calibrated_proba(sample_df)
            np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-12)

            # Verify saved test predictions on disk match live model evaluation
            saved_preds = pd.read_csv(EXPORTS_DIR / f"test_predictions__{proto_id}__{m_key}.csv")
            assert len(saved_preds) > 500
            assert set(saved_preds["predicted_label"].unique()).issubset({0, 1})


# -----------------------------------------------------------------------------
# 10. Batch-Prediction Schema and Exact Row Alignment
# -----------------------------------------------------------------------------
def test_10_batch_prediction_schema_and_row_alignment(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, mod_df = datasets
    art = load_verified_model_artifact("protocol_b_temporal_purged", "hist_gradient_boosting")

    # Create a non-contiguous shuffled index DataFrame
    sub = mod_df.sample(n=65, random_state=99)
    out_df, diag = run_batch_prediction(art, sub)

    assert diag["row_alignment_verified"] is True
    assert len(out_df) == 65
    assert list(out_df.index) == list(sub.index)
    assert (out_df["snapshot_id"].values == sub["snapshot_id"].values).all()
    assert (out_df["account_id"].values == sub["account_id"].values).all()


# -----------------------------------------------------------------------------
# 11. Injected Leakage Detection and Clean Negative Controls (FIX-01 to FIX-07)
# -----------------------------------------------------------------------------
def test_11_injected_leakage_detection_and_clean_controls(master_bundle: dict) -> None:
    fx_sum = master_bundle["controlled_fixtures_summary"]
    assert fx_sum["total_fixtures_evaluated"] == 7
    assert fx_sum["contaminated_fixture_count"] == 5
    assert fx_sum["clean_control_fixture_count"] == 2
    assert fx_sum["contaminated_detected_with_provenance"] == 5
    assert fx_sum["false_positive_flags_on_clean_controls"] == 0

    by_id = {f["fixture_id"]: f for f in fx_sum["fixtures"]}

    # Verify each deliberate fault was caught by its expected rule ID
    assert "LAK-001" in by_id["FIX-01"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-004" in by_id["FIX-02"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-005" in by_id["FIX-02"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-002" in by_id["FIX-03"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-003" in by_id["FIX-03"]["full_provenance_mode"]["suspicious_signal_rules"]
    assert "LAK-006" in by_id["FIX-04"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-007" in by_id["FIX-04"]["full_provenance_mode"]["confirmed_leakage_rules"]
    assert "LAK-008" in by_id["FIX-05"]["full_provenance_mode"]["confirmed_leakage_rules"]

    # Verify Raw-CSV mode provenance boundaries (LAK-007 cannot be confirmed without FitProvenanceRecord)
    assert "LAK-007" in by_id["FIX-04"]["raw_csv_mode"]["unverifiable_rules"]
    assert "LAK-007" not in by_id["FIX-04"]["raw_csv_mode"]["confirmed_leakage_rules"]

    # Verify clean negative control (FIX-06) and distribution shift control (FIX-07)
    assert by_id["FIX-06"]["full_provenance_mode"]["confirmed_leakage_rules"] == []
    assert by_id["FIX-07"]["full_provenance_mode"]["confirmed_leakage_rules"] == []
    assert "LAK-011" in by_id["FIX-07"]["full_provenance_mode"]["distribution_shift_rules"]


# -----------------------------------------------------------------------------
# 12. SQLite Persistence and Export Fidelity
# -----------------------------------------------------------------------------
def test_12_persistence_and_export_fidelity(master_bundle: dict) -> None:
    assert DB_PATH.exists()
    with sqlite3.connect(str(DB_PATH)) as conn:
        exp_count = conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0]
        aud_count = conn.execute("SELECT COUNT(*) FROM audit_runs").fetchone()[0]
        mod_count = conn.execute("SELECT COUNT(*) FROM model_artifacts").fetchone()[0]

    assert exp_count >= 2
    assert aud_count >= 2
    assert mod_count >= 6

    # Check that exported CSV metrics match the actual saved test predictions on disk
    comp_df = pd.read_csv(EXPORTS_DIR / "benchmark_model_comparison.csv")
    assert len(comp_df) == 6
    for _, row in comp_df.iterrows():
        pred_file = EXPORTS_DIR / f"test_predictions__{row['protocol_id']}__{row['model_key']}.csv"
        pred_df = pd.read_csv(pred_file)
        recalc_pr_auc = round(
            float(average_precision_score(pred_df["actual_churn_90d"], pred_df["calibrated_probability"])), 4
        )
        recalc_brier = round(
            float(brier_score_loss(pred_df["actual_churn_90d"], pred_df["calibrated_probability"])), 5
        )
        assert abs(recalc_pr_auc - float(row["test_pr_auc"])) <= 1e-4
        assert abs(recalc_brier - float(row["test_brier_calibrated"])) <= 1e-4


# -----------------------------------------------------------------------------
# 13. Separation of Dev Decisions from Test Evaluation + FastAPI Smoke Tests
# -----------------------------------------------------------------------------
def test_13_dev_test_separation_and_fastapi_smoke_suite(datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, mod_df = datasets
    client = TestClient(app)

    # 13a. Health & Contract
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_contract = client.get("/contract")
    assert r_contract.status_code == 200
    assert r_contract.json()["prediction_horizon_days"] == 90

    # 13b. Demo dataset & Controlled fixtures
    r_demo = client.get("/datasets/demo?sample_rows=10")
    assert r_demo.status_code == 200
    assert len(r_demo.json()["preview_rows"]) == 10

    r_fix = client.get("/audit/fixtures")
    assert r_fix.status_code == 200
    assert r_fix.json()["contaminated_detected_with_provenance"] == 5

    # 13c. CSV Upload endpoint (verifying provenance warnings on raw CSV)
    sample_csv_bytes = mod_df.head(120).to_csv(index=False).encode("utf-8")
    r_up = client.post(
        "/datasets/upload?target_col=churned_within_90d&account_id_col=account_id",
        content=sample_csv_bytes,
        headers={"Content-Type": "text/csv"},
    )
    assert r_up.status_code == 200
    up_json = r_up.json()
    assert up_json["is_valid"] is True
    assert up_json["provenance_capabilities"]["LAK-007 (Preprocessing Fitted Outside Training Boundary)"]["status"] == "UNVERIFIABLE_MISSING_PROVENANCE"

    # 13d. Single & Batch Prediction endpoints
    contract = build_default_saas_contract()
    first_record_json_safe = json.loads(
        mod_df[contract.all_predictor_cols].iloc[0:1].to_json(orient="records")
    )[0]
    single_payload = {
        "protocol_id": "protocol_b_temporal_purged",
        "model_key": "hist_gradient_boosting",
        "features": first_record_json_safe,
    }
    r_single = client.post("/predict/single", json=single_payload)
    assert r_single.status_code == 200
    assert "calibrated_churn_probability" in r_single.json()["prediction"]
    assert len(r_single.json()["local_attributions"]) > 0

    batch_payload = {
        "protocol_id": "protocol_b_temporal_purged",
        "model_key": "logistic_regression",
        "records": json.loads(mod_df[contract.all_predictor_cols].head(15).to_json(orient="records")),
    }
    r_batch = client.post("/predict/batch", json=batch_payload)
    assert r_batch.status_code == 200
    assert len(r_batch.json()["predictions"]) == 15
    assert r_batch.json()["diagnostics"]["row_alignment_verified"] is True

    # 13e. Export endpoint
    r_exp = client.get("/export/protocol_b_temporal_purged")
    assert r_exp.status_code == 200
    exp_bundle = r_exp.json()
    hgb_thresh_summary = exp_bundle["models"]["hist_gradient_boosting"]["threshold_selection_summary"]
    assert hgb_thresh_summary["selection_partition"] == "development_validation_only"
    assert hgb_thresh_summary["frozen_before_test_evaluation"] is True
