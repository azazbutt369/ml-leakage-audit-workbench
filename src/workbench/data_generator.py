"""
Deterministic Synthetic B2B SaaS Longitudinal Customer-Churn Dataset Generator.
Generates underlying account trajectories and calculates future 90-day outcomes strictly
from the simulation process, preserving lineage metadata, hashes and data dictionaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from workbench.config import (
    CLEAN_DATA_DIR,
    DEFAULT_SEED,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
    ensure_directories,
)
from workbench.contract import (
    PredictionContract,
    build_default_saas_contract,
    compute_dataframe_sha256,
    compute_row_hashes,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_saas_churn_dataset(
    seed: int = DEFAULT_SEED,
    n_accounts: int = 1650,
    save_to_disk: bool = True,
    output_dir: Path = CLEAN_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate the deterministic B2B SaaS longitudinal dataset (~14,500–16,000 total snapshots).

    Returns:
        full_df: Complete longitudinal snapshot table including right-censored late waves.
        modelling_df: Eligible uncensored active snapshots (is_censored_outcome == False).
        manifest: Generation record with seed, hashes, prevalence and censoring statistics.
    """
    ensure_directories()
    rng = np.random.default_rng(seed)
    contract: PredictionContract = build_default_saas_contract()

    # 18 monthly observation waves: 2024-01-01 through 2025-06-01
    # Cutoff date is 2025-06-30, so 2025-04-01 (+90d = 2025-06-30) is the last uncensored wave.
    # 2025-05-01 and 2025-06-01 have outcome windows extending beyond 2025-06-30 (right-censored).
    observation_dates = pd.date_range("2024-01-01", "2025-06-01", freq="MS")
    cutoff_ts = pd.Timestamp(contract.dataset_cutoff_date)

    industries = [
        "FinTech",
        "HealthTech",
        "Logistics",
        "ECommerce",
        "DevTools",
        "ProfessionalServices",
    ]
    industry_probs = [0.18, 0.14, 0.16, 0.20, 0.17, 0.15]

    size_bands = ["1-50", "51-250", "251-1000", "1000+"]
    size_probs = [0.42, 0.33, 0.17, 0.08]

    plan_tiers = ["Starter", "Growth", "Enterprise"]
    payment_methods = ["DirectDebit", "CreditCard", "Invoice"]

    records = []

    for acc_idx in range(1, n_accounts + 1):
        account_id = f"ACC-{acc_idx:05d}"

        # Account-level baseline characteristics
        size_band = rng.choice(size_bands, p=size_probs)
        if size_band == "1-50":
            plan_tier = rng.choice(plan_tiers, p=[0.65, 0.32, 0.03])
            base_seats = int(rng.integers(3, 28))
        elif size_band == "51-250":
            plan_tier = rng.choice(plan_tiers, p=[0.22, 0.63, 0.15])
            base_seats = int(rng.integers(18, 120))
        elif size_band == "251-1000":
            plan_tier = rng.choice(plan_tiers, p=[0.05, 0.50, 0.45])
            base_seats = int(rng.integers(65, 380))
        else:
            plan_tier = rng.choice(plan_tiers, p=[0.01, 0.24, 0.75])
            base_seats = int(rng.integers(180, 950))

        contract_type = rng.choice(
            ["Monthly", "Annual"],
            p=[0.68, 0.32] if plan_tier == "Starter" else ([0.42, 0.58] if plan_tier == "Growth" else [0.18, 0.82]),
        )
        payment_method = rng.choice(
            payment_methods,
            p=[0.35, 0.55, 0.10] if plan_tier == "Starter" else [0.50, 0.25, 0.25],
        )
        industry = rng.choice(industries, p=industry_probs)

        # Structural missingness on firmographics for a small subset of accounts
        industry_val = None if rng.random() < 0.03 else str(industry)
        size_band_val = None if rng.random() < 0.025 else str(size_band)

        # Entry wave index (0 = active from 2024-01-01, or staggered onboarding)
        entry_wave = 0 if rng.random() < 0.68 else int(rng.integers(1, 11))
        initial_tenure = int(rng.integers(1, 36)) if entry_wave == 0 else 1

        price_per_seat = {"Starter": 24.0, "Growth": 48.0, "Enterprise": 85.0}[plan_tier]
        discount_pct = float(np.round(rng.choice([0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]), 1))

        # Persistent account-level latent traits (drives realistic intra-account correlation)
        account_random_effect = float(rng.normal(0.0, 0.55))
        base_engagement = float(np.clip(rng.beta(5.2, 2.4), 0.12, 0.98))
        base_adoption = float(np.clip(rng.normal(62.0 if plan_tier != "Starter" else 48.0, 16.0), 8.0, 99.0))

        is_active = True
        current_seats = base_seats
        current_engagement = base_engagement
        current_adoption = base_adoption

        for wave_idx, obs_ts in enumerate(observation_dates):
            if wave_idx < entry_wave or not is_active:
                continue

            tenure_months = initial_tenure + (wave_idx - entry_wave)

            # Evolve dynamic usage and support state up to t_obs
            engagement_shock = float(rng.normal(-0.012, 0.075))
            current_engagement = float(np.clip(current_engagement + engagement_shock, 0.03, 1.0))
            current_adoption = float(
                np.clip(current_adoption + rng.normal(0.15, 2.8) + 8.0 * engagement_shock, 5.0, 100.0)
            )

            seat_delta = int(rng.choice([-2, -1, 0, 0, 0, 1, 2], p=[0.04, 0.08, 0.62, 0.10, 0.06, 0.06, 0.04]))
            current_seats = max(1, current_seats + seat_delta)
            mrr_gbp = round(
                float(current_seats * price_per_seat * (1.0 - discount_pct / 100.0)),
                2,
            )

            active_seats_ratio_30d = round(
                float(np.clip(current_engagement + rng.normal(0.0, 0.05), 0.02, 1.0)),
                4,
            )
            usage_events_30d = int(
                max(
                    0,
                    np.round(
                        current_seats
                        * active_seats_ratio_30d
                        * rng.gamma(shape=4.5, scale=18.0)
                    ),
                )
            )

            # 90d usage trend (missing if tenure < 3 months or rare telemetry gap)
            if tenure_months < 3 or rng.random() < 0.02:
                usage_trend_90d_pct = np.nan
                effective_trend = 0.0
            else:
                effective_trend = float(
                    np.clip(
                        (active_seats_ratio_30d - base_engagement) * 145.0 + rng.normal(0.0, 14.0),
                        -95.0,
                        220.0,
                    )
                )
                usage_trend_90d_pct = round(effective_trend, 2)

            api_error_rate_30d = round(
                float(np.clip(rng.beta(1.2, 38.0) + (0.04 if industry == "DevTools" else 0.0), 0.0, 0.45)),
                4,
            )
            feature_adoption_score = round(current_adoption, 2)

            days_since_last_admin_login = round(
                float(
                    np.clip(
                        rng.exponential(scale=4.5 if active_seats_ratio_30d > 0.5 else 18.0),
                        0.1,
                        89.0,
                    )
                ),
                1,
            )

            # Support & billing history up to t_obs
            ticket_rate = max(0.2, 1.4 + 0.015 * current_seats + 6.5 * api_error_rate_30d)
            support_tickets_30d = int(rng.poisson(lam=ticket_rate))
            urgent_tickets_90d = int(
                rng.binomial(
                    n=max(1, support_tickets_30d * 2),
                    p=float(np.clip(0.08 + 1.1 * api_error_rate_30d, 0.02, 0.65)),
                )
            )

            # Mean response hours & CSAT (with realistic structural missingness)
            if support_tickets_30d == 0 and urgent_tickets_90d == 0 and rng.random() < 0.72:
                mean_first_response_hours = np.nan
                effective_resp_hours = 3.5
            else:
                effective_resp_hours = float(
                    np.clip(rng.lognormal(mean=1.35 + 0.14 * urgent_tickets_90d, sigma=0.55), 0.2, 72.0)
                )
                mean_first_response_hours = round(effective_resp_hours, 2)

            if rng.random() < 0.28 or ( np.isnan(mean_first_response_hours) and rng.random() < 0.85 ):
                csat_trailing_90d = np.nan
                effective_csat = 3.8
            else:
                raw_csat = (
                    4.35
                    - 0.28 * urgent_tickets_90d
                    - 0.025 * effective_resp_hours
                    + 0.45 * (active_seats_ratio_30d - 0.5)
                    + rng.normal(0.0, 0.38)
                )
                effective_csat = float(np.clip(raw_csat, 1.0, 5.0))
                csat_trailing_90d = round(effective_csat, 2)

            billing_risk_base = 0.35 if payment_method == "CreditCard" else (0.22 if payment_method == "Invoice" else 0.08)
            late_payments_12m = int(rng.poisson(lam=billing_risk_base * (1.5 if plan_tier == "Starter" else 0.9)))
            dunning_events_6m = int(
                rng.poisson(
                    lam=0.18 * late_payments_12m + (0.30 if payment_method == "CreditCard" else 0.05)
                )
            )

            # Underlying latent 90-day churn hazard (combines moderate baseline linear trend with
            # realistic B2B SaaS non-linear regime interactions and threshold cliff effects)
            linear_logit = (
                -2.65
                - 0.75 * (active_seats_ratio_30d - 0.55)
                - 0.008 * (feature_adoption_score - 55.0)
                - 0.005 * effective_trend
                + 0.012 * (days_since_last_admin_login - 10.0)
                + 0.11 * urgent_tickets_90d
                + 0.16 * dunning_events_6m
                + 0.10 * late_payments_12m
                + 1.10 * (api_error_rate_30d - 0.03)
                - 0.15 * (effective_csat - 3.8)
                + (0.25 if contract_type == "Monthly" else -0.20)
                + (0.15 if plan_tier == "Starter" else (-0.12 if plan_tier == "Enterprise" else 0.0))
                + 0.25 * account_random_effect
            )

            # Strong non-linear interaction regimes (shelf/cliff effects typical of SaaS churn)
            nonlinear_boost = 0.0
            # Regime 1: Shelf/cliff collapse in seat utilisation on Monthly contracts
            if active_seats_ratio_30d < 0.38 and contract_type == "Monthly":
                nonlinear_boost += 1.65
            elif active_seats_ratio_30d < 0.25 and contract_type == "Annual":
                nonlinear_boost += 1.25

            # Regime 2: Support SLA escalation crisis (multiple urgent tickets + slow response)
            if urgent_tickets_90d >= 2 and effective_resp_hours > 6.5:
                nonlinear_boost += 1.55
            elif support_tickets_30d >= 5 and effective_csat < 2.8:
                nonlinear_boost += 1.30

            # Regime 3: Payment failure & dunning cascade on credit cards
            if dunning_events_6m >= 2 and (payment_method == "CreditCard" or late_payments_12m >= 2):
                nonlinear_boost += 1.50

            # Regime 4: Ghost-admin onboarding drop-off (tenure <= 5m, low adoption, admin absent)
            if tenure_months <= 5 and days_since_last_admin_login > 18.0 and feature_adoption_score < 48.0:
                nonlinear_boost += 1.45

            # Regime 5: Deeply embedded healthy enterprise/growth stickiness shield
            if active_seats_ratio_30d > 0.68 and feature_adoption_score > 65.0 and dunning_events_6m == 0:
                nonlinear_boost -= 1.35

            churn_prob_90d = float(_sigmoid(np.array(linear_logit + nonlinear_boost + rng.normal(0.0, 0.22))))

            # Realise forward 90-day outcome from underlying Bernoulli trial
            realised_churn_90d = int(rng.random() < churn_prob_90d)

            outcome_window_end_ts = obs_ts + pd.Timedelta(days=contract.prediction_horizon_days)
            is_censored = bool(outcome_window_end_ts > cutoff_ts)

            obs_date_str = obs_ts.strftime("%Y-%m-%d")
            outcome_end_str = outcome_window_end_ts.strftime("%Y-%m-%d")
            snapshot_id = f"SNAP-{account_id}-{obs_ts.strftime('%Y%m')}"

            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "account_id": account_id,
                    "observation_date": obs_date_str,
                    "outcome_window_end": outcome_end_str,
                    "dataset_cutoff_date": contract.dataset_cutoff_date,
                    "account_status_at_obs": "ACTIVE",
                    "is_censored_outcome": is_censored,
                    "mrr_gbp": mrr_gbp,
                    "seats_purchased": current_seats,
                    "tenure_months": tenure_months,
                    "discount_pct": discount_pct,
                    "active_seats_ratio_30d": active_seats_ratio_30d,
                    "usage_events_30d": usage_events_30d,
                    "usage_trend_90d_pct": usage_trend_90d_pct,
                    "api_error_rate_30d": api_error_rate_30d,
                    "feature_adoption_score": feature_adoption_score,
                    "days_since_last_admin_login": days_since_last_admin_login,
                    "support_tickets_30d": support_tickets_30d,
                    "urgent_tickets_90d": urgent_tickets_90d,
                    "mean_first_response_hours": mean_first_response_hours,
                    "csat_trailing_90d": csat_trailing_90d,
                    "late_payments_12m": late_payments_12m,
                    "dunning_events_6m": dunning_events_6m,
                    "plan_tier": str(plan_tier),
                    "contract_type": str(contract_type),
                    "industry_sector": industry_val,
                    "company_size_band": size_band_val,
                    "payment_method": str(payment_method),
                    # If right-censored at dataset cutoff, target outcome is not yet observable -> NaN
                    "churned_within_90d": np.nan if is_censored else int(realised_churn_90d),
                }
            )

            # If the account churned within this 90d window, it terminates before subsequent monthly waves
            # (with ~55% terminating within the first 30d and ~45% within 31-90d).
            if realised_churn_90d == 1 and rng.random() < 0.72:
                is_active = False

    full_df = pd.DataFrame(records)
    # Compute canonical row hash over predictor + identifier columns
    hash_cols = ["account_id", "observation_date"] + contract.all_predictor_cols
    full_df["row_hash"] = compute_row_hashes(full_df, cols=hash_cols)

    # Order columns cleanly
    ordered_cols = (
        [
            "snapshot_id",
            "account_id",
            "observation_date",
            "outcome_window_end",
            "dataset_cutoff_date",
            "account_status_at_obs",
            "is_censored_outcome",
            "row_hash",
        ]
        + contract.numerical_features
        + contract.categorical_features
        + ["churned_within_90d"]
    )
    full_df = full_df[ordered_cols].reset_index(drop=True)

    # Filter uncensored active observations for supervised modelling
    modelling_df = full_df.loc[~full_df["is_censored_outcome"]].copy().reset_index(drop=True)
    modelling_df["churned_within_90d"] = modelling_df["churned_within_90d"].astype(int)

    full_sha256 = compute_dataframe_sha256(full_df)
    modelling_sha256 = compute_dataframe_sha256(modelling_df)

    manifest: Dict[str, Any] = {
        "dataset_id": "SYNTH-B2B-SAAS-CHURN-V1",
        "contract_id": contract.contract_id,
        "simulation_seed": seed,
        "synthetic_disclosure": SYNTHETIC_BENCHMARK_DISCLOSURE,
        "total_snapshots_including_censored": int(len(full_df)),
        "uncensored_modelling_snapshots": int(len(modelling_df)),
        "right_censored_snapshots_excluded": int(full_df["is_censored_outcome"].sum()),
        "unique_accounts_total": int(full_df["account_id"].nunique()),
        "unique_accounts_uncensored": int(modelling_df["account_id"].nunique()),
        "observation_date_min": str(full_df["observation_date"].min()),
        "observation_date_max_full": str(full_df["observation_date"].max()),
        "observation_date_max_uncensored": str(modelling_df["observation_date"].max()),
        "dataset_cutoff_date": contract.dataset_cutoff_date,
        "prediction_horizon_days": contract.prediction_horizon_days,
        "target_positive_count": int(modelling_df["churned_within_90d"].sum()),
        "target_negative_count": int((modelling_df["churned_within_90d"] == 0).sum()),
        "target_positive_prevalence": round(float(modelling_df["churned_within_90d"].mean()), 4),
        "full_dataset_sha256": full_sha256,
        "modelling_dataset_sha256": modelling_sha256,
        "missing_rates_uncensored": {
            col: round(float(modelling_df[col].isna().mean()), 4)
            for col in contract.all_predictor_cols
            if modelling_df[col].isna().any()
        },
    }

    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(output_dir / "saas_churn_snapshots_full.csv", index=False)
        modelling_df.to_csv(output_dir / "saas_churn_snapshots_modelling.csv", index=False)

        data_dict = {
            "contract": contract.model_dump(),
            "columns": [spec.model_dump() for spec in contract.feature_registry.values()],
        }
        (output_dir / "data_dictionary.json").write_text(
            json.dumps(data_dict, indent=2), encoding="utf-8"
        )
        (output_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return full_df, modelling_df, manifest
