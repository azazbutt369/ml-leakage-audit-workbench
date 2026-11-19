"""
Evaluation Protocols & Partitioning Engine.
Implements two distinct, auditable evaluation protocols:
- Protocol A (group_holdout): New-account generalisation with strictly disjoint account_id partitions.
- Protocol B (temporal_purged): Future-period prediction with chronological boundaries and
  90-day outcome-window purging so no future label information leaks across cutoffs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from workbench.config import CLEAN_DATA_DIR, DEFAULT_SEED
from workbench.contract import compute_dataframe_sha256


@dataclass
class PartitionBundle:
    """Container for Train, Development (Validation) and Frozen Test splits plus provenance."""

    protocol_id: Literal["protocol_a_group_holdout", "protocol_b_temporal_purged", "custom_split"]
    protocol_name: str
    is_primary_benchmark: bool
    allow_returning_accounts: bool
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    purged_df: pd.DataFrame
    censored_df: pd.DataFrame
    manifest: Dict[str, Any]


def _summarise_partition(df: pd.DataFrame, account_col: str, date_col: str, target_col: str) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "row_count": 0,
            "unique_accounts": 0,
            "positive_count": 0,
            "positive_prevalence": 0.0,
            "min_observation_date": None,
            "max_observation_date": None,
            "sha256": "",
        }
    valid_targets = pd.to_numeric(df[target_col], errors="coerce").dropna()
    return {
        "row_count": int(len(df)),
        "unique_accounts": int(df[account_col].nunique()) if account_col in df.columns else 0,
        "positive_count": int((valid_targets == 1).sum()),
        "negative_count": int((valid_targets == 0).sum()),
        "positive_prevalence": round(float((valid_targets == 1).mean()), 4) if len(valid_targets) > 0 else 0.0,
        "min_observation_date": str(df[date_col].min()) if date_col in df.columns else None,
        "max_observation_date": str(df[date_col].max()) if date_col in df.columns else None,
        "max_outcome_window_end": (
            str(df["outcome_window_end"].max()) if "outcome_window_end" in df.columns else None
        ),
        "sha256": compute_dataframe_sha256(df),
    }


def split_protocol_a_new_account_generalisation(
    full_df: pd.DataFrame,
    account_col: str = "account_id",
    date_col: str = "observation_date",
    target_col: str = "churned_within_90d",
    censoring_col: str = "is_censored_outcome",
    seed: int = DEFAULT_SEED,
    save_manifest_dir: Path | None = CLEAN_DATA_DIR,
) -> PartitionBundle:
    """
    Protocol A — New-Account Generalisation (Secondary Benchmark).
    Evaluates model generalisation to customer organisations completely absent from training.
    Keeps all longitudinal snapshots of an account strictly within the same partition
    (approx. 70% Train accounts, 15% Development accounts, 15% Frozen Test accounts).
    """
    if censoring_col in full_df.columns:
        censored_df = full_df.loc[full_df[censoring_col] == True].copy().reset_index(drop=True)  # noqa: E712
        eligible_df = full_df.loc[full_df[censoring_col] == False].copy().reset_index(drop=True)  # noqa: E712
    else:
        censored_df = full_df.loc[full_df[target_col].isna()].copy().reset_index(drop=True)
        eligible_df = full_df.loc[full_df[target_col].notna()].copy().reset_index(drop=True)

    eligible_df[target_col] = eligible_df[target_col].astype(int)

    # Use 20-fold StratifiedGroupKFold (each fold = 5% of accounts):
    # Folds 0..13 (14 folds = 70%) -> Train
    # Folds 14..16 (3 folds = 15%) -> Development (Validation)
    # Folds 17..19 (3 folds = 15%) -> Frozen Test
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(eligible_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(eligible_df, eligible_df[target_col], eligible_df[account_col])
    ):
        fold_assignments[val_idx] = fold_idx

    train_df = eligible_df.loc[fold_assignments < 14].copy().reset_index(drop=True)
    val_df = eligible_df.loc[(fold_assignments >= 14) & (fold_assignments < 17)].copy().reset_index(drop=True)
    test_df = eligible_df.loc[fold_assignments >= 17].copy().reset_index(drop=True)
    purged_df = eligible_df.iloc[0:0].copy()

    train_accs = set(train_df[account_col].unique())
    val_accs = set(val_df[account_col].unique())
    test_accs = set(test_df[account_col].unique())

    manifest: Dict[str, Any] = {
        "protocol_id": "protocol_a_group_holdout",
        "protocol_name": "Protocol A: New-Account Generalisation (Disjoint Group Holdout)",
        "is_primary_benchmark": False,
        "deployment_question": (
            "How accurately does the model predict 90-day churn for brand-new customer accounts "
            "whose historical snapshots were never observed during training?"
        ),
        "allow_returning_accounts": False,
        "seed": seed,
        "account_overlap_checks": {
            "train_val_overlap_count": len(train_accs.intersection(val_accs)),
            "train_test_overlap_count": len(train_accs.intersection(test_accs)),
            "val_test_overlap_count": len(val_accs.intersection(test_accs)),
        },
        "partitions": {
            "train": _summarise_partition(train_df, account_col, date_col, target_col),
            "val": _summarise_partition(val_df, account_col, date_col, target_col),
            "test": _summarise_partition(test_df, account_col, date_col, target_col),
            "purged": {"row_count": 0, "reason": "Group-isolated protocol; no temporal cutoff purge required."},
            "censored_excluded": {"row_count": int(len(censored_df))},
        },
    }

    if save_manifest_dir is not None:
        save_manifest_dir.mkdir(parents=True, exist_ok=True)
        (save_manifest_dir / "split_manifest_protocol_a.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return PartitionBundle(
        protocol_id="protocol_a_group_holdout",
        protocol_name=manifest["protocol_name"],
        is_primary_benchmark=False,
        allow_returning_accounts=False,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        purged_df=purged_df,
        censored_df=censored_df,
        manifest=manifest,
    )


def split_protocol_b_future_period_prediction(
    full_df: pd.DataFrame,
    account_col: str = "account_id",
    date_col: str = "observation_date",
    outcome_end_col: str = "outcome_window_end",
    target_col: str = "churned_within_90d",
    censoring_col: str = "is_censored_outcome",
    val_start_date: str = "2024-11-01",
    val_end_date: str = "2024-12-01",
    test_start_date: str = "2025-03-01",
    test_end_date: str = "2025-04-01",
    horizon_days: int = 90,
    save_manifest_dir: Path | None = CLEAN_DATA_DIR,
) -> PartitionBundle:
    """
    Protocol B — Future-Period Prediction with Outcome-Window Purging (Primary Benchmark).
    Evaluates forward-in-time monthly production scoring on active SaaS accounts.
    Returning active accounts are explicitly allowed (as an existing customer base is scored each month),
    AND label availability is strictly enforced at every temporal boundary:
      - Training observations must have outcome_window_end <= val_start_date.
      - Any observation before val_start_date whose outcome_window_end > val_start_date is PURGED.
      - Development observations span [val_start_date, val_end_date] and have outcome_window_end <= test_start_date.
      - Any observation between val_end_date and test_start_date whose outcome_window_end > test_start_date is PURGED.
      - Frozen Test observations span [test_start_date, test_end_date] with outcomes fully realised by dataset_cutoff_date.
    """
    if censoring_col in full_df.columns:
        censored_df = full_df.loc[full_df[censoring_col] == True].copy().reset_index(drop=True)  # noqa: E712
        eligible_df = full_df.loc[full_df[censoring_col] == False].copy().reset_index(drop=True)  # noqa: E712
    else:
        censored_df = full_df.loc[full_df[target_col].isna()].copy().reset_index(drop=True)
        eligible_df = full_df.loc[full_df[target_col].notna()].copy().reset_index(drop=True)

    eligible_df[target_col] = eligible_df[target_col].astype(int)

    obs_ts = pd.to_datetime(eligible_df[date_col])
    if outcome_end_col in eligible_df.columns:
        out_end_ts = pd.to_datetime(eligible_df[outcome_end_col])
    else:
        out_end_ts = obs_ts + pd.Timedelta(days=horizon_days)

    val_start_ts = pd.Timestamp(val_start_date)
    val_end_ts = pd.Timestamp(val_end_date)
    test_start_ts = pd.Timestamp(test_start_date)
    test_end_ts = pd.Timestamp(test_end_date)

    # 1. Train: observed before val_start_ts AND 90d outcome window finished on or before val_start_ts
    train_mask = (obs_ts < val_start_ts) & (out_end_ts <= val_start_ts)

    # 2. Purge Buffer 1: observed before val_start_ts BUT outcome window extends past val_start_ts
    purge_1_mask = (obs_ts < val_start_ts) & (out_end_ts > val_start_ts)

    # 3. Development (Validation): observed in [val_start_ts, val_end_ts] AND outcome window finished <= test_start_ts
    val_mask = (obs_ts >= val_start_ts) & (obs_ts <= val_end_ts) & (out_end_ts <= test_start_ts)

    # 4. Purge Buffer 2: observed before test_start_ts (after val_start_ts) with outcome window > test_start_ts
    purge_2_mask = (obs_ts > val_end_ts) & (obs_ts < test_start_ts) & (out_end_ts > test_start_ts)

    # 5. Frozen Test: observed in [test_start_ts, test_end_ts]
    test_mask = (obs_ts >= test_start_ts) & (obs_ts <= test_end_ts)

    train_df = eligible_df.loc[train_mask].copy().reset_index(drop=True)
    val_df = eligible_df.loc[val_mask].copy().reset_index(drop=True)
    test_df = eligible_df.loc[test_mask].copy().reset_index(drop=True)
    purged_df = eligible_df.loc[purge_1_mask | purge_2_mask].copy().reset_index(drop=True)

    train_accs = set(train_df[account_col].unique()) if account_col in train_df.columns else set()
    val_accs = set(val_df[account_col].unique()) if account_col in val_df.columns else set()
    test_accs = set(test_df[account_col].unique()) if account_col in test_df.columns else set()

    manifest: Dict[str, Any] = {
        "protocol_id": "protocol_b_temporal_purged",
        "protocol_name": "Protocol B: Future-Period Prediction (Chronological with 90-Day Outcome-Window Purging)",
        "is_primary_benchmark": True,
        "deployment_question": (
            "Given historical monthly snapshots whose 90-day outcomes have fully materialised by the "
            "training cutoff timestamp, how accurately can we score active accounts (including returning "
            "customers and newly onboarded accounts) in future calendar months?"
        ),
        "allow_returning_accounts": True,
        "temporal_boundaries": {
            "val_start_date": val_start_date,
            "val_end_date": val_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "prediction_horizon_days": horizon_days,
        },
        "purging_summary": {
            "purged_train_to_val_rows": int(purge_1_mask.sum()),
            "purged_val_to_test_rows": int(purge_2_mask.sum()),
            "total_purged_rows": int(len(purged_df)),
            "purged_observation_dates": sorted(purged_df[date_col].unique().tolist()) if len(purged_df) > 0 else [],
            "censored_late_rows_excluded": int(len(censored_df)),
        },
        "returning_account_diagnostics": {
            "val_returning_accounts_from_train": len(train_accs.intersection(val_accs)),
            "val_new_accounts": len(val_accs - train_accs),
            "test_returning_accounts_from_train": len(train_accs.intersection(test_accs)),
            "test_new_accounts": len(test_accs - train_accs),
        },
        "partitions": {
            "train": _summarise_partition(train_df, account_col, date_col, target_col),
            "val": _summarise_partition(val_df, account_col, date_col, target_col),
            "test": _summarise_partition(test_df, account_col, date_col, target_col),
            "purged": _summarise_partition(purged_df, account_col, date_col, target_col),
            "censored_excluded": {"row_count": int(len(censored_df))},
        },
    }

    if save_manifest_dir is not None:
        save_manifest_dir.mkdir(parents=True, exist_ok=True)
        (save_manifest_dir / "split_manifest_protocol_b.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return PartitionBundle(
        protocol_id="protocol_b_temporal_purged",
        protocol_name=manifest["protocol_name"],
        is_primary_benchmark=True,
        allow_returning_accounts=True,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        purged_df=purged_df,
        censored_df=censored_df,
        manifest=manifest,
    )


def build_inner_cv_splits(
    train_df: pd.DataFrame,
    protocol_id: str,
    account_col: str = "account_id",
    date_col: str = "observation_date",
    target_col: str = "churned_within_90d",
    n_splits: int = 4,
    seed: int = DEFAULT_SEED,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Generate inner cross-validation fold indices strictly inside train_df that respect
    the declared evaluation protocol (group-isolated folds for Protocol A; expanding-window
    purged temporal folds or group-isolated folds for Protocol B).
    """
    if protocol_id == "protocol_b_temporal_purged" and date_col in train_df.columns:
        unique_dates = sorted(train_df[date_col].unique().tolist())
        # If enough monthly waves exist (e.g. 8 waves in Train: Jan..Aug 2024), create expanding
        # temporal folds with a 2-month (60-90d) purge gap where possible, falling back to StratifiedGroupKFold
        if len(unique_dates) >= 6:
            folds: List[tuple[np.ndarray, np.ndarray]] = []
            # E.g. waves [0..2] -> purge [3..4] -> val [5], etc.
            candidate_configs = [
                (unique_dates[:3], unique_dates[5:6]),
                (unique_dates[:4], unique_dates[6:7]),
                (unique_dates[:5], unique_dates[7:8]),
            ]
            for tr_dates, vl_dates in candidate_configs:
                if tr_dates and vl_dates:
                    tr_idx = np.where(train_df[date_col].isin(tr_dates))[0]
                    vl_idx = np.where(train_df[date_col].isin(vl_dates))[0]
                    if len(tr_idx) > 20 and len(vl_idx) > 20 and train_df.iloc[tr_idx][target_col].nunique() == 2:
                        folds.append((tr_idx, vl_idx))
            if len(folds) >= 2:
                return folds

    if account_col in train_df.columns and train_df[account_col].nunique() >= n_splits:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(train_df, train_df[target_col], train_df[account_col]))

    # Fallback stratified split if no group column exists (e.g. external CSV without group ID)
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(train_df, train_df[target_col]))
