"""
Local Persistence, Versioned Model Artefact Storage (with SHA-256 Integrity Verification),
Saved Prediction Manifests and Batch Inference Engine.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from workbench.config import (
    ARTIFACTS_DIR,
    DB_PATH,
    EXPORTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_directories,
)
from workbench.leakage_audit import AuditReport
from workbench.pipelines import AuditedModelArtifact


def _compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def init_sqlite_db(db_path: Path = DB_PATH) -> None:
    """Initialise local SQLite tables for experiment metadata, audit reports and model artefacts."""
    ensure_directories()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_runs (
                audit_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                protocol_id TEXT NOT NULL,
                overall_verdict TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                protocol_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_key TEXT NOT NULL,
                protocol_id TEXT NOT NULL,
                filepath TEXT NOT NULL,
                sha256_digest TEXT NOT NULL,
                frozen_threshold REAL NOT NULL,
                test_pr_auc REAL,
                test_roc_auc REAL,
                test_brier REAL,
                timestamp_utc TEXT NOT NULL
            )
            """
        )
        conn.commit()


def persist_audit_report(
    report: AuditReport,
    db_path: Path = DB_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save an AuditReport to SQLite and export a versioned JSON file."""
    init_sqlite_db(db_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report.model_dump(), indent=2)
    out_path = reports_dir / f"audit_{report.protocol_id}.json"
    out_path.write_text(report_json, encoding="utf-8")

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audit_runs
            (audit_id, dataset_name, protocol_id, overall_verdict, timestamp_utc, report_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report.audit_id,
                report.dataset_name,
                report.protocol_id,
                report.overall_verdict,
                report.timestamp_utc,
                report_json,
            ),
        )
        conn.commit()
    return out_path


def save_model_artifact_and_predictions(
    artifact: AuditedModelArtifact,
    eval_summary: Dict[str, Any],
    test_df: pd.DataFrame,
    run_id: str,
    target_col: str = "churned_within_90d",
    models_dir: Path = MODELS_DIR,
    exports_dir: Path = EXPORTS_DIR,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Persist a fitted AuditedModelArtifact to disk using joblib, record its SHA-256 digest in
    a sidecar manifest and SQLite, and save row-level test predictions to CSV so every reported
    metric is directly traceable to saved predictions.
    """
    init_sqlite_db(db_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    artifact_id = f"{artifact.protocol_id}__{artifact.model_key}"
    model_path = models_dir / f"{artifact_id}.joblib"
    joblib.dump(artifact, model_path)
    sha256_digest = _compute_file_sha256(model_path)

    # Save row-level test set predictions
    raw_probs = artifact.predict_raw_proba(test_df)
    cal_probs = artifact.predict_calibrated_proba(test_df)
    binary_preds = (cal_probs >= artifact.frozen_threshold).astype(int)

    pred_df = pd.DataFrame(
        {
            "row_index": np.arange(len(test_df), dtype=int),
            "snapshot_id": test_df["snapshot_id"].astype(str) if "snapshot_id" in test_df.columns else np.arange(len(test_df)).astype(str),
            "account_id": test_df["account_id"].astype(str) if "account_id" in test_df.columns else "N/A",
            "observation_date": test_df["observation_date"].astype(str) if "observation_date" in test_df.columns else "N/A",
            "actual_churn_90d": test_df[target_col].astype(int) if target_col in test_df.columns else np.nan,
            "raw_probability": np.round(raw_probs, 6),
            "calibrated_probability": np.round(cal_probs, 6),
            "frozen_threshold": round(float(artifact.frozen_threshold), 4),
            "predicted_label": binary_preds,
        }
    )
    pred_csv_path = exports_dir / f"test_predictions__{artifact_id}.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    pred_sha256 = _compute_file_sha256(pred_csv_path)

    manifest_meta = {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "model_key": artifact.model_key,
        "model_display_name": artifact.model_display_name,
        "protocol_id": artifact.protocol_id,
        "model_filepath": str(model_path.name),
        "model_sha256": sha256_digest,
        "saved_test_predictions_csv": str(pred_csv_path.name),
        "saved_test_predictions_sha256": pred_sha256,
        "frozen_threshold": artifact.frozen_threshold,
        "selected_hyperparameters": artifact.selected_hyperparameters,
        "feature_columns": artifact.feature_columns,
        "numerical_features": artifact.numerical_features,
        "categorical_features": artifact.categorical_features,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = models_dir / f"{artifact_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest_meta, indent=2), encoding="utf-8")

    test_m = eval_summary["metrics"]["test_frozen"]
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_artifacts
            (artifact_id, run_id, model_key, protocol_id, filepath, sha256_digest,
             frozen_threshold, test_pr_auc, test_roc_auc, test_brier, timestamp_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                artifact.model_key,
                artifact.protocol_id,
                str(model_path),
                sha256_digest,
                float(artifact.frozen_threshold),
                test_m.get("pr_auc"),
                test_m.get("roc_auc"),
                test_m.get("brier_score"),
                manifest_meta["timestamp_utc"],
            ),
        )
        conn.commit()

    return manifest_meta


def load_verified_model_artifact(
    protocol_id: str,
    model_key: str,
    models_dir: Path = MODELS_DIR,
) -> AuditedModelArtifact:
    """
    Load a persisted AuditedModelArtifact after verifying its SHA-256 digest against
    its sidecar manifest. Never loads unverified or tampered model binaries.
    """
    artifact_id = f"{protocol_id}__{model_key}"
    model_path = models_dir / f"{artifact_id}.joblib"
    manifest_path = models_dir / f"{artifact_id}_manifest.json"

    if not model_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Model artefact or manifest not found for '{artifact_id}' in {models_dir}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = _compute_file_sha256(model_path)
    expected_sha256 = manifest.get("model_sha256", "")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 integrity mismatch for model artefact '{artifact_id}': "
            f"expected {expected_sha256}, found {actual_sha256}."
        )

    loaded_obj = joblib.load(model_path)
    if not isinstance(loaded_obj, AuditedModelArtifact):
        raise TypeError(f"Loaded object for '{artifact_id}' is not an AuditedModelArtifact.")
    return loaded_obj


def run_batch_prediction(
    artifact: AuditedModelArtifact,
    input_df: pd.DataFrame,
    custom_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute schema-validated batch or single-instance prediction while preserving exact
    input row ordering and reporting missing/unknown-category diagnostics.
    """
    if len(input_df) == 0:
        raise ValueError("Input DataFrame for batch prediction is empty (0 rows).")

    missing_cols = [c for c in artifact.feature_columns if c not in input_df.columns]
    if missing_cols:
        raise ValueError(
            f"Input data is missing required predictor columns for model '{artifact.model_key}': {missing_cols}"
        )

    # Coerce numerical columns safely and check for unparseable strings
    work_df = input_df.copy()
    for col in artifact.numerical_features:
        coerced = pd.to_numeric(work_df[col], errors="coerce")
        invalid_non_null = int(work_df[col].notna().sum() - coerced.notna().sum())
        if invalid_non_null > 0:
            raise ValueError(
                f"Numerical predictor column '{col}' contains {invalid_non_null} non-numeric string values."
            )
        work_df[col] = coerced

    # Check unseen categories relative to fitted vocabulary
    learned_vocab = artifact.fit_provenance.learned_summary_stats.get("categorical_vocabularies", {})
    unseen_summary: Dict[str, List[str]] = {}
    for col in artifact.categorical_features:
        known_levels = set(learned_vocab.get(col, []))
        if known_levels:
            obs_levels = set(work_df[col].dropna().astype(str).unique())
            unseen = sorted(list(obs_levels - known_levels))
            if unseen:
                unseen_summary[col] = unseen

    raw_probs = artifact.predict_raw_proba(work_df)
    cal_probs = artifact.predict_calibrated_proba(work_df)
    tau = float(custom_threshold) if custom_threshold is not None else float(artifact.frozen_threshold)
    preds = (cal_probs >= tau).astype(int)

    risk_bands = np.where(
        cal_probs >= max(0.60, tau + 0.20),
        "Critical",
        np.where(
            cal_probs >= tau,
            "High (Above Threshold)",
            np.where(cal_probs >= 0.15, "Moderate", "Low"),
        ),
    )

    out_df = pd.DataFrame(index=input_df.index)
    out_df["input_row_position"] = np.arange(len(input_df), dtype=int)
    for id_col in ["snapshot_id", "account_id", "observation_date"]:
        if id_col in input_df.columns:
            out_df[id_col] = input_df[id_col].values

    out_df["raw_churn_probability"] = np.round(raw_probs, 5)
    out_df["calibrated_churn_probability"] = np.round(cal_probs, 5)
    out_df["decision_threshold_applied"] = round(tau, 4)
    out_df["predicted_churn_90d"] = preds
    out_df["risk_band"] = risk_bands

    diagnostics = {
        "model_key": artifact.model_key,
        "protocol_id": artifact.protocol_id,
        "input_row_count": int(len(input_df)),
        "output_row_count": int(len(out_df)),
        "row_alignment_verified": bool(len(input_df) == len(out_df) and (out_df.index == input_df.index).all()),
        "threshold_applied": round(tau, 4),
        "is_frozen_dev_threshold": custom_threshold is None or abs(tau - artifact.frozen_threshold) < 1e-6,
        "predicted_positive_count": int(preds.sum()),
        "predicted_positive_rate": round(float(preds.mean()), 4),
        "mean_calibrated_probability": round(float(cal_probs.mean()), 4),
        "missing_values_handled_by_pipeline": {
            col: int(work_df[col].isna().sum())
            for col in artifact.feature_columns
            if work_df[col].isna().any()
        },
        "unseen_categories_handled_by_pipeline": unseen_summary,
    }
    return out_df, diagnostics


def persist_experiment_bundle(
    run_id: str,
    protocol_id: str,
    dataset_name: str,
    bundle_dict: Dict[str, Any],
    db_path: Path = DB_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save the complete protocol evaluation bundle to SQLite and JSON."""
    init_sqlite_db(db_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    json_str = json.dumps(bundle_dict, indent=2)
    out_path = reports_dir / f"evaluation_bundle_{protocol_id}.json"
    out_path.write_text(json_str, encoding="utf-8")

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiment_runs
            (run_id, protocol_id, dataset_name, timestamp_utc, summary_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, protocol_id, dataset_name, ts, json_str),
        )
        conn.commit()
    return out_path
