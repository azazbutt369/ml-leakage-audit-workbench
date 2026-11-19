"""
CLI Runner for the Tabular ML and Leakage-Audit Workbench.
Generates the deterministic synthetic B2B SaaS dataset, executes Protocol B (Primary) and
Protocol A (Secondary), evaluates all 7 controlled leakage and negative-control fixtures,
persists all artefacts and prints a concise summary of actual measured results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workbench.config import DEFAULT_SEED  # noqa: E402
from workbench.runner import run_complete_workbench_pipeline  # noqa: E402


def main() -> None:
    print("=" * 88)
    print("AZAZ BUTT — TABULAR ML & LEAKAGE-AUDIT WORKBENCH: FULL PIPELINE EXECUTION")
    print("=" * 88)

    master = run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    dm = master["dataset_manifest"]
    print(f"\n[1] Synthetic Dataset Generated (Seed={dm['simulation_seed']}):")
    print(f"    - Total Snapshots (incl. right-censored): {dm['total_snapshots_including_censored']:,}")
    print(f"    - Uncensored Active Snapshots:            {dm['uncensored_modelling_snapshots']:,}")
    print(f"    - Right-Censored Snapshots Excluded:      {dm['right_censored_snapshots_excluded']:,}")
    print(f"    - Unique Accounts (Total / Uncensored):   {dm['unique_accounts_total']:,} / {dm['unique_accounts_uncensored']:,}")
    print(f"    - 90-Day Churn Prevalence (Uncensored):   {dm['target_positive_prevalence'] * 100:.2f}%")
    print(f"    - Modelling Dataset SHA-256:              {dm['modelling_dataset_sha256']}")

    print("\n[2] Clean Benchmark Audit Verdicts:")
    print(
        f"    - Protocol B (Primary, Future-Period Purged): {master['primary_protocol_b']['audit_report']['overall_verdict']} "
        f"(Confirmed Leakage = {master['primary_protocol_b']['audit_report']['summary_counts']['CONFIRMED_LEAKAGE']})"
    )
    print(
        f"    - Protocol A (Secondary, New-Account Group):  {master['secondary_protocol_a']['audit_report']['overall_verdict']} "
        f"(Confirmed Leakage = {master['secondary_protocol_a']['audit_report']['summary_counts']['CONFIRMED_LEAKAGE']})"
    )

    print("\n[3] Model Comparison Table (Dev vs Frozen Test):")
    for row in master["comparison_table"]:
        print(
            f"    * [{row['protocol_id']}] {row['model_key']:<24} | "
            f"Tau*={row['frozen_threshold']:.3f} | "
            f"Dev PR-AUC={row['dev_pr_auc']:.4f} | "
            f"Test PR-AUC={row['test_pr_auc']:.4f} [{row['test_pr_auc_ci95_low']:.4f}, {row['test_pr_auc_ci95_high']:.4f}] | "
            f"Test ROC-AUC={row['test_roc_auc']:.4f} | "
            f"Test F1={row['test_f1']:.4f} | "
            f"Brier={row['test_brier_calibrated']:.4f} (Raw {row['test_brier_uncalibrated']:.4f})"
        )

    fx_sum = master["controlled_fixtures_summary"]
    print("\n[4] Controlled Leakage & Negative-Control Fixtures Summary:")
    print(
        f"    - Contaminated Fixtures Detected (Full Provenance): "
        f"{fx_sum['contaminated_detected_with_provenance']} / {fx_sum['contaminated_fixture_count']}"
    )
    print(
        f"    - Contaminated Fixtures Confirmed (Raw CSV Only):   "
        f"{fx_sum['contaminated_confirmed_from_raw_csv_alone']} / {fx_sum['contaminated_fixture_count']}"
    )
    print(
        f"    - False Positive Leakage Flags on Clean Controls:   "
        f"{fx_sum['false_positive_flags_on_clean_controls']}"
    )
    for fx in fx_sum["fixtures"]:
        print(
            f"      [{fx['fixture_id']}] {fx['name']:<55} | "
            f"Full={fx['full_provenance_mode']['confirmed_leakage_rules']} | "
            f"CSV_Confirmed={fx['raw_csv_mode']['confirmed_leakage_rules']} | "
            f"CSV_Unverifiable={fx['raw_csv_mode']['unverifiable_rules']}"
        )

    print("\n" + "=" * 88)
    print("PIPELINE EXECUTION COMPLETE — ALL ARTEFACTS SAVED TO data/ AND artifacts/")
    print("=" * 88)


if __name__ == "__main__":
    main()
