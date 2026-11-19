"""
Streamlit Technical Workbench Interface for Azaz Butt's Tabular ML and Leakage-Audit Workbench.
Provides 8 structured engineering views covering the Prediction Contract, CSV Upload & Provenance
Validator, 11-Rule Leakage-Audit Engine, Controlled Leakage Benchmark Lab, Model Comparison,
Calibration & Frozen Threshold Explorer, Explainability & Error Analysis, and Batch/Single Inference.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from workbench.config import (
    BRAND_AUTHOR,
    BRAND_ROLE,
    BRAND_SCENARIO_TITLE,
    BRAND_WORKBENCH_NAME,
    CAUSAL_ATTRIBUTION_DISCLOSURE,
    CLEAN_DATA_DIR,
    DEFAULT_SEED,
    EXPORTS_DIR,
    FIXTURES_DIR,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_ROWS,
    REPORTS_DIR,
    SYNTHETIC_BENCHMARK_DISCLOSURE,
)
from workbench.contract import (
    build_default_saas_contract,
    validate_uploaded_dataframe,
)
from workbench.evaluation import select_development_threshold_and_build_explorer
from workbench.explainability import compute_local_instance_attribution
from workbench.leakage_audit import run_leakage_audit
from workbench.runner import run_complete_workbench_pipeline
from workbench.splits import split_protocol_a_new_account_generalisation
from workbench.storage import (
    load_verified_model_artifact,
    run_batch_prediction,
)


st.set_page_config(
    page_title=f"{BRAND_AUTHOR} | {BRAND_WORKBENCH_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading verified benchmark artefacts...")
def load_master_summary() -> Dict[str, Any]:
    summary_path = REPORTS_DIR / "master_workbench_summary.json"
    if not summary_path.exists():
        return run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading synthetic B2B SaaS dataset...")
def load_clean_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_path = CLEAN_DATA_DIR / "saas_churn_snapshots_full.csv"
    mod_path = CLEAN_DATA_DIR / "saas_churn_snapshots_modelling.csv"
    if not full_path.exists() or not mod_path.exists():
        run_complete_workbench_pipeline(seed=DEFAULT_SEED)
    return pd.read_csv(full_path), pd.read_csv(mod_path)


def render_header() -> None:
    st.markdown(
        f"""
        <div style="border-bottom: 1px solid #2d3748; padding-bottom: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; font-weight: 600;">
                {BRAND_AUTHOR} &middot; {BRAND_ROLE}
            </span>
            <h1 style="margin: 0.2rem 0 0.3rem 0; font-size: 1.85rem;">{BRAND_WORKBENCH_NAME}</h1>
            <p style="margin: 0; color: #4a5568; font-size: 0.98rem;">
                Primary Scenario: <strong>{BRAND_SCENARIO_TITLE}</strong> &mdash;
                Predictive Modelling, Leakage Prevention, Calibration &amp; Error Diagnostics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(SYNTHETIC_BENCHMARK_DISCLOSURE)


def main() -> None:
    render_header()
    master = load_master_summary()
    full_df, mod_df = load_clean_datasets()
    contract = build_default_saas_contract()

    st.sidebar.markdown(f"### {BRAND_AUTHOR}")
    st.sidebar.caption(f"{BRAND_ROLE}\n\n**{BRAND_WORKBENCH_NAME} v1.0**")
    st.sidebar.divider()

    active_protocol_choice = st.sidebar.radio(
        "Active Benchmark Protocol",
        options=[
            "protocol_b_temporal_purged",
            "protocol_a_group_holdout",
        ],
        format_func=lambda x: (
            "Protocol B (Primary): Future-Period Purged"
            if x == "protocol_b_temporal_purged"
            else "Protocol A (Secondary): New-Account Group Split"
        ),
    )
    active_bundle = (
        master["primary_protocol_b"]
        if active_protocol_choice == "protocol_b_temporal_purged"
        else master["secondary_protocol_a"]
    )

    if st.sidebar.button("Re-run Full Clean Benchmark Pipeline", use_container_width=True):
        with st.spinner("Executing deterministic data generation, splits, audits and model training..."):
            run_complete_workbench_pipeline(seed=DEFAULT_SEED)
            st.cache_data.clear()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Dataset Fingerprint (SHA-256)**\n`{master['dataset_manifest']['modelling_dataset_sha256'][:20]}...`\n\n"
        f"**Simulation Seed**: `{master['seed']}`\n\n"
        f"**Total Snapshots**: `{master['dataset_manifest']['total_snapshots_including_censored']:,}`\n"
        f"**Uncensored Active**: `{master['dataset_manifest']['uncensored_modelling_snapshots']:,}`\n"
        f"**Right-Censored Excluded**: `{master['dataset_manifest']['right_censored_snapshots_excluded']:,}`"
    )

    tabs = st.tabs(
        [
            "1. Contract & Data",
            "2. CSV Upload & Custom Audit",
            "3. Leakage-Audit Engine",
            "4. Controlled Leakage Lab",
            "5. Model Comparison",
            "6. Calibration & Thresholds",
            "7. Explainability & Errors",
            "8. Inference & Exports",
        ]
    )

    # =========================================================================
    # TAB 1: Prediction Contract & Demonstration Dataset
    # =========================================================================
    with tabs[0]:
        st.subheader("1. Formal Prediction Contract & Longitudinal B2B SaaS Dataset")
        col1, col2, col3, col4 = st.columns(4)
        gen_m = master["dataset_manifest"]
        col1.metric("Total Snapshots (Raw)", f"{gen_m['total_snapshots_including_censored']:,}")
        col2.metric("Uncensored Modelling Rows", f"{gen_m['uncensored_modelling_snapshots']:,}")
        col3.metric("Right-Censored Excluded", f"{gen_m['right_censored_snapshots_excluded']:,}")
        col4.metric("90-Day Churn Prevalence", f"{gen_m['target_positive_prevalence'] * 100:.2f}%")

        st.markdown("#### Formal Prediction Contract Specification")
        contract_table = pd.DataFrame(
            [
                {"Contract Attribute": "Unit of Observation", "Specification": contract.unit_of_observation},
                {"Contract Attribute": "Account / Group Identifier", "Specification": f"`{contract.account_id_col}` ({gen_m['unique_accounts_total']:,} unique organisations)"},
                {"Contract Attribute": "Prediction Timestamp", "Specification": f"`{contract.timestamp_col}` (Monthly waves: {gen_m['observation_date_min']} to {gen_m['observation_date_max_full']})"},
                {"Contract Attribute": "Prediction Horizon & Cutoff", "Specification": f"{contract.prediction_horizon_days} days forward; global dataset extraction cutoff `{contract.dataset_cutoff_date}`"},
                {"Contract Attribute": "Target Definition (`churned_within_90d`)", "Specification": contract.target_definition},
                {"Contract Attribute": "Missing & Right-Censored Policy", "Specification": contract.censoring_and_missing_outcome_policy},
                {"Contract Attribute": "Intended Deployment Population", "Specification": contract.intended_deployment_population},
                {"Contract Attribute": "Primary Benchmark Protocol", "Specification": contract.primary_protocol},
                {"Contract Attribute": "Secondary Benchmark Protocol", "Specification": contract.secondary_protocol},
            ]
        )
        st.table(contract_table)

        st.markdown("#### Feature Availability & Provenance Registry")
        reg_df = pd.DataFrame([s.model_dump() for s in contract.feature_registry.values()])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        st.markdown("#### Sample Longitudinal Snapshots (Uncensored Modelling Set)")
        st.dataframe(mod_df.head(30), use_container_width=True)

    # =========================================================================
    # TAB 2: CSV Upload, Column Mapping & Provenance Capability Matrix
    # =========================================================================
    with tabs[1]:
        st.subheader("2. External CSV Upload, Column Selection & Provenance Boundary")
        st.markdown(
            f"Upload an external CSV (computational limits: max **{MAX_UPLOAD_MB} MB**, max **{MAX_UPLOAD_ROWS:,} rows**) "
            "or inspect a sample fixture CSV. The workbench explicitly distinguishes checks supported directly "
            "by the uploaded table from checks that require lineage or transformation fit provenance."
        )

        uploaded_file = st.file_uploader("Upload CSV Dataset for Validation & Auditing", type=["csv"])
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            try:
                user_df = pd.read_csv(io.BytesIO(raw_bytes))
                st.success(f"Loaded `{uploaded_file.name}` ({len(user_df):,} rows, {len(user_df.columns)} columns).")
            except Exception as exc:
                st.error(f"CSV Parsing Error: {exc}")
                user_df = None
        else:
            st.caption("No external file uploaded — previewing column selector on a 500-row sample of the demonstration dataset (simulated as an external raw CSV upload).")
            user_df = mod_df.head(500).copy()
            raw_bytes = b""

        if user_df is not None:
            all_cols = list(user_df.columns)
            c1, c2, c3, c4 = st.columns(4)
            sel_target = c1.selectbox(
                "Target Column",
                options=all_cols,
                index=all_cols.index("churned_within_90d") if "churned_within_90d" in all_cols else len(all_cols) - 1,
            )
            sel_account = c2.selectbox(
                "Account / Group Identifier (Optional)",
                options=["<None>"] + all_cols,
                index=(all_cols.index("account_id") + 1) if "account_id" in all_cols else 0,
            )
            sel_date = c3.selectbox(
                "Prediction Timestamp (Optional)",
                options=["<None>"] + all_cols,
                index=(all_cols.index("observation_date") + 1) if "observation_date" in all_cols else 0,
            )
            sel_out_end = c4.selectbox(
                "Outcome Window End (Optional)",
                options=["<None>"] + all_cols,
                index=(all_cols.index("outcome_window_end") + 1) if "outcome_window_end" in all_cols else 0,
            )

            reserved_cols = {
                sel_target,
                sel_account if sel_account != "<None>" else None,
                sel_date if sel_date != "<None>" else None,
                sel_out_end if sel_out_end != "<None>" else None,
                "snapshot_id",
                "dataset_cutoff_date",
                "account_status_at_obs",
                "is_censored_outcome",
                "row_hash",
            }
            default_feats = [c for c in all_cols if c not in reserved_cols]
            sel_features = st.multiselect(
                "Predictor Feature Columns",
                options=[c for c in all_cols if c != sel_target],
                default=default_feats,
            )
            sel_proto = st.radio(
                "Declared Evaluation Protocol for Uploaded CSV",
                options=["protocol_a_group_holdout", "protocol_b_temporal_purged"],
                format_func=lambda x: (
                    "Protocol A: New-Account Generalisation (Disjoint Group Split)"
                    if x == "protocol_a_group_holdout"
                    else "Protocol B: Future-Period Prediction (Chronological Split)"
                ),
                horizontal=True,
            )

            val_report = validate_uploaded_dataframe(
                df=user_df,
                target_col=sel_target,
                account_id_col=None if sel_account == "<None>" else sel_account,
                timestamp_col=None if sel_date == "<None>" else sel_date,
                outcome_window_end_col=None if sel_out_end == "<None>" else sel_out_end,
                feature_cols=sel_features,
                file_size_bytes=len(raw_bytes),
                has_feature_availability_metadata=False,
                has_fit_provenance=False,
                has_resampling_provenance=False,
            )

            if not val_report.is_valid:
                for err in val_report.errors:
                    st.error(err)
            else:
                for w in val_report.warnings:
                    st.warning(w)

                st.markdown("#### Evidence & Provenance Availability Matrix (Raw CSV Upload)")
                prov_rows = [
                    {"Rule": k, "Availability Status": v["status"], "Evidence Requirement & Boundary": v["requirement"]}
                    for k, v in val_report.provenance_capabilities.items()
                ]
                st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)

                if st.button("Run Leakage Audit on Selected CSV Configuration"):
                    clean_u = user_df.dropna(subset=[sel_target]).copy().reset_index(drop=True)
                    clean_u[sel_target] = pd.to_numeric(clean_u[sel_target], errors="coerce").fillna(0).astype(int)
                    acc_c = None if sel_account == "<None>" else sel_account
                    date_c = None if sel_date == "<None>" else sel_date
                    out_c = None if sel_out_end == "<None>" else sel_out_end

                    if sel_proto == "protocol_a_group_holdout" and acc_c and acc_c in clean_u.columns and clean_u[acc_c].nunique() >= 20:
                        b_u = split_protocol_a_new_account_generalisation(
                            clean_u, account_col=acc_c, date_col=date_c or "observation_date", target_col=sel_target, save_manifest_dir=None
                        )
                        tr_u, vl_u, te_u = b_u.train_df, b_u.val_df, b_u.test_df
                        allow_ret = False
                    else:
                        tr_u, tmp_u = train_test_split(clean_u, test_size=0.30, stratify=clean_u[sel_target], random_state=DEFAULT_SEED)
                        vl_u, te_u = train_test_split(tmp_u, test_size=0.50, stratify=tmp_u[sel_target], random_state=DEFAULT_SEED)
                        tr_u, vl_u, te_u = tr_u.reset_index(drop=True), vl_u.reset_index(drop=True), te_u.reset_index(drop=True)
                        allow_ret = (sel_proto == "protocol_b_temporal_purged")

                    custom_audit = run_leakage_audit(
                        train_df=tr_u,
                        val_df=vl_u,
                        test_df=te_u,
                        feature_cols=sel_features,
                        target_col=sel_target,
                        account_col=acc_c,
                        date_col=date_c,
                        outcome_end_col=out_c,
                        protocol_id=sel_proto,
                        allow_returning_accounts=allow_ret,
                        contract=None,
                        has_lineage_metadata=False,
                        fit_provenance=None,
                        resampling_provenance=None,
                        dataset_name=uploaded_file.name if uploaded_file else "Simulated_Raw_CSV_Sample",
                    )
                    st.markdown(f"**Audit Overall Verdict**: `{custom_audit.overall_verdict}`")
                    st.json(custom_audit.summary_counts)
                    st.dataframe(
                        pd.DataFrame([f.model_dump() for f in custom_audit.findings]),
                        use_container_width=True,
                        hide_index=True,
                    )

    # =========================================================================
    # TAB 3: Leakage-Audit Engine (Clean Benchmark Protocol Inspection)
    # =========================================================================
    with tabs[2]:
        st.subheader(f"3. Leakage-Audit Engine — {active_bundle['protocol_name']}")
        audit_rep = active_bundle["audit_report"]
        st.markdown(
            f"**Audit ID**: `{audit_rep['audit_id']}` &nbsp;|&nbsp; "
            f"**Overall Verdict**: `{audit_rep['overall_verdict']}` &nbsp;|&nbsp; "
            f"**Lineage Metadata**: `{audit_rep['has_lineage_metadata']}` &nbsp;|&nbsp; "
            f"**Fit Provenance Verified**: `{audit_rep['has_fit_provenance']}`"
        )
        st.caption(audit_rep["scope_disclaimer"])

        s_cols = st.columns(6)
        for idx, (cat_k, count_v) in enumerate(audit_rep["summary_counts"].items()):
            s_cols[idx].metric(cat_k, count_v)

        findings_df = pd.DataFrame(audit_rep["findings"])
        st.dataframe(
            findings_df[
                [
                    "rule_id",
                    "rule_title",
                    "category",
                    "severity",
                    "verification_status",
                    "consequence_explanation",
                    "recommended_correction",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Inspect Detailed Rule Evidence & Fit Provenance")
        for f_item in audit_rep["findings"]:
            with st.expander(f"{f_item['rule_id']} — {f_item['rule_title']} [{f_item['category']} | {f_item['severity']}]"):
                st.markdown(f"**Consequence**: {f_item['consequence_explanation']}")
                st.markdown(f"**Recommended Correction**: {f_item['recommended_correction']}")
                st.markdown("**Observed Evidence**:")
                st.json(f_item["observed_evidence"])
                st.markdown("**Affected Scope**:")
                st.json(f_item["affected_scope"])

        st.markdown("#### Partition & Outcome-Window Purge Manifest")
        st.json(active_bundle["split_manifest"])

    # =========================================================================
    # TAB 4: Controlled Leakage Demonstrations (FIX-01 to FIX-07)
    # =========================================================================
    with tabs[3]:
        st.subheader("4. Controlled Leakage Benchmark Lab (Deliberate Faults & Negative Controls)")
        fix_sum = master["controlled_fixtures_summary"]

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        fc1.metric("Total Fixtures", fix_sum["total_fixtures_evaluated"])
        fc2.metric("Contaminated Variants", fix_sum["contaminated_fixture_count"])
        fc3.metric("Detected (With Provenance)", f"{fix_sum['contaminated_detected_with_provenance']} / {fix_sum['contaminated_fixture_count']}")
        fc4.metric("Confirmed from Raw CSV Alone", f"{fix_sum['contaminated_confirmed_from_raw_csv_alone']} / {fix_sum['contaminated_fixture_count']}")
        fc5.metric("False Flags on Clean Controls", fix_sum["false_positive_flags_on_clean_controls"])

        st.info(fix_sum["provenance_dependency_analysis"])

        fix_table_rows = []
        for fx in fix_sum["fixtures"]:
            hgb_contam = fx["contaminated_vs_corrected_metrics"]["fixture_measured"]["hist_gradient_boosting"]
            hgb_clean = fx["contaminated_vs_corrected_metrics"]["clean_protocol_reference"]["hist_gradient_boosting"]
            fix_table_rows.append(
                {
                    "Fixture ID": fx["fixture_id"],
                    "Scenario": fx["name"],
                    "Contaminated?": fx["is_contaminated"],
                    "Requires Provenance?": fx["requires_external_provenance"],
                    "Expected Rules": ", ".join(fx["expected_confirmed_leakage_rules"]) or "None (Control)",
                    "Detected Confirmed (Full Provenance)": ", ".join(fx["full_provenance_mode"]["confirmed_leakage_rules"]) or "None",
                    "Detected Confirmed (Raw CSV Mode)": ", ".join(fx["raw_csv_mode"]["confirmed_leakage_rules"]) or "None",
                    "Raw CSV Suspicious / Unverifiable": ", ".join(
                        fx["raw_csv_mode"]["suspicious_signal_rules"] + fx["raw_csv_mode"]["unverifiable_rules"]
                    ),
                    "Fixture HistGB PR-AUC": hgb_contam["test_pr_auc"],
                    "Clean Ref HistGB PR-AUC": hgb_clean["test_pr_auc"],
                    "Fixture HistGB ROC-AUC": hgb_contam["test_roc_auc"],
                    "Clean Ref HistGB ROC-AUC": hgb_clean["test_roc_auc"],
                }
            )
        st.dataframe(pd.DataFrame(fix_table_rows), use_container_width=True, hide_index=True)

        for fx in fix_sum["fixtures"]:
            with st.expander(f"{fx['fixture_id']}: {fx['name']} — Verdict: {fx['full_provenance_mode']['overall_verdict']}"):
                st.markdown(f"**Injected Mechanism (Evaluator Manifest)**: {fx['injected_fault_description']}")
                st.markdown(f"**Metric Comparison Interpretation**: {fx['contaminated_vs_corrected_metrics']['interpretation']}")
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("**Full-Provenance Audit Output**")
                    st.json(fx["full_provenance_mode"])
                with c_right:
                    st.markdown("**Raw-CSV Upload Mode Audit Output**")
                    st.json(fx["raw_csv_mode"])

    # =========================================================================
    # TAB 5: Baselines, Candidate Models & Protocol Comparison
    # =========================================================================
    with tabs[4]:
        st.subheader("5. Baselines, Candidate Models & Protocol Comparison")
        comp_df = pd.DataFrame(master["comparison_table"])
        st.markdown("#### Cross-Protocol Frozen Test & Development Benchmark Table")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig_comp = px.bar(
            comp_df,
            x="model_display_name",
            y="test_pr_auc",
            color="protocol_role",
            barmode="group",
            title="Frozen Test PR-AUC by Model Family and Evaluation Protocol",
            labels={"model_display_name": "Model", "test_pr_auc": "Test PR-AUC (Average Precision)"},
            text_auto=".3f",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown(f"#### Detailed Inner-CV Tuning & Uncertainty for `{active_bundle['protocol_name']}`")
        for m_key, m_eval in active_bundle["models"].items():
            with st.expander(f"{m_eval['model_display_name']} (`{m_key}`) — Frozen Threshold τ* = {m_eval['frozen_threshold']}"):
                st.markdown(f"**Selected Hyperparameters**: `{json.dumps(m_eval['selected_hyperparameters'])}`")
                st.markdown(f"**Selection Criterion**: {m_eval['selection_criterion']}")
                st.markdown("**Inner Cross-Validation Tuning Trials (Including Fold-Isolated Resampling Comparison)**:")
                st.dataframe(pd.DataFrame(m_eval["tuning_trials"]), use_container_width=True, hide_index=True)
                st.markdown("**Account-Cluster Bootstrapped 95% Confidence Intervals (Frozen Test)**:")
                st.json(m_eval["test_uncertainty"])
                st.markdown("**CPU Inference Latency Benchmark**:")
                st.json(m_eval["latency_benchmark"])

    # =========================================================================
    # TAB 6: Calibration Diagnostics & Decision Threshold Explorer
    # =========================================================================
    with tabs[5]:
        st.subheader(f"6. Probability Calibration & Decision Threshold Explorer ({active_bundle['protocol_id']})")
        sel_m_key = st.selectbox(
            "Select Model for Calibration & Threshold Analysis",
            options=["hist_gradient_boosting", "logistic_regression", "class_prior_baseline"],
            format_func=lambda k: active_bundle["models"][k]["model_display_name"],
        )
        m_data = active_bundle["models"][sel_m_key]
        test_m = m_data["metrics"]["test_frozen"]
        dev_m = m_data["metrics"]["val_development"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frozen Threshold (Selected on Dev)", f"{m_data['frozen_threshold']:.3f}")
        c2.metric("Test Brier (Calibrated)", f"{test_m['brier_score']:.4f}", delta=f"{test_m['brier_score'] - test_m['uncalibrated_brier_score']:.4f} vs Raw", delta_color="inverse")
        c3.metric("Test ECE (Calibrated)", f"{test_m['expected_calibration_error_ece']:.4f}")
        c4.metric("Test F1 at Frozen τ*", f"{test_m['f1']:.4f}")

        st.markdown("#### Calibration Provenance (Measured Separately from Recalibration)")
        st.json(m_data["calibration_provenance"])

        # Reliability Curve Plot (Uncalibrated vs OOF-Recalibrated on Frozen Test)
        cal_bins = pd.DataFrame(test_m["calibration_curve"]).dropna(subset=["mean_predicted_prob"])
        raw_bins = pd.DataFrame(test_m["uncalibrated_curve"]).dropna(subset=["mean_predicted_prob"])

        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly Calibrated", line=dict(dash="dash", color="gray"))
        )
        if len(raw_bins) > 0:
            fig_cal.add_trace(
                go.Scatter(
                    x=raw_bins["mean_predicted_prob"],
                    y=raw_bins["observed_positive_rate"],
                    mode="lines+markers",
                    name=f"Uncalibrated (Brier={test_m['uncalibrated_brier_score']:.4f})",
                )
            )
        if len(cal_bins) > 0:
            fig_cal.add_trace(
                go.Scatter(
                    x=cal_bins["mean_predicted_prob"],
                    y=cal_bins["observed_positive_rate"],
                    mode="lines+markers",
                    name=f"OOF-Calibrated (Brier={test_m['brier_score']:.4f})",
                )
            )
        fig_cal.update_layout(
            title="Reliability Diagram on Frozen Test Partition (10 Quantile/Uniform Bins)",
            xaxis_title="Mean Predicted Churn Probability",
            yaxis_title="Observed 90-Day Churn Frequency",
        )
        st.plotly_chart(fig_cal, use_container_width=True)

        st.divider()
        st.markdown("#### Development Set Threshold Explorer vs Frozen Test Reporting")
        st.warning(
            "Methodological Boundary: The interactive explorer below operates strictly on the **Development (Validation)** "
            "partition. The **Frozen Test** partition is evaluated once at the locked threshold "
            f"`τ* = {m_data['frozen_threshold']:.3f}`. Business cost values are hypothetical user-configurable assumptions."
        )

        exp_df = pd.DataFrame(m_data["dev_threshold_explorer"])
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(x=exp_df["threshold"], y=exp_df["precision"], mode="lines", name="Dev Precision"))
        fig_th.add_trace(go.Scatter(x=exp_df["threshold"], y=exp_df["recall"], mode="lines", name="Dev Recall"))
        fig_th.add_trace(go.Scatter(x=exp_df["threshold"], y=exp_df["f1"], mode="lines", name="Dev F1 Score"))
        fig_th.add_vline(
            x=m_data["frozen_threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Frozen τ* = {m_data['frozen_threshold']:.3f}",
        )
        fig_th.update_layout(
            title="Development Partition Precision, Recall & F1 Trade-off Curve",
            xaxis_title="Candidate Decision Threshold (τ)",
            yaxis_title="Metric Score",
        )
        st.plotly_chart(fig_th, use_container_width=True)
        st.dataframe(exp_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 7: Explainability, Cohort Slices & Error Analysis
    # =========================================================================
    with tabs[6]:
        st.subheader(f"7. Explainability, Cohort Slices & Error Analysis ({active_bundle['protocol_id']})")
        st.info(CAUSAL_ATTRIBUTION_DISCLOSURE)

        err_m_key = st.selectbox(
            "Select Model for Explainability & Error Inspection",
            options=["hist_gradient_boosting", "logistic_regression"],
            format_func=lambda k: active_bundle["models"][k]["model_display_name"],
        )
        err_model = active_bundle["models"][err_m_key]

        glob_exp = err_model.get("global_explainability")
        if glob_exp:
            st.markdown("#### Global Permutation Feature Importance (Development Partition, Metric: PR-AUC)")
            imp_df = pd.DataFrame(glob_exp["feature_importances"])
            fig_imp = px.bar(
                imp_df.head(15).sort_values("mean_pr_auc_drop", ascending=True),
                x="mean_pr_auc_drop",
                y="feature",
                error_x="std_pr_auc_drop",
                orientation="h",
                title="Top 15 Features by Mean PR-AUC Drop Under Permutation (Development Set)",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Performance Across Business Cohorts & Observation Months (Frozen Test)")
        slices = err_model["test_cohort_slices"]
        slice_dim = st.selectbox("Select Cohort Dimension", options=list(slices.keys()))
        st.dataframe(pd.DataFrame(slices[slice_dim]), use_container_width=True, hide_index=True)

        st.markdown("#### Missing-Data & Unknown-Category Diagnostics Across Splits")
        diag_mu = active_bundle["missing_and_unknown_diagnostics"]
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Missing-Value Rates by Partition**")
            st.dataframe(pd.DataFrame(diag_mu["missing_value_diagnostics"]), use_container_width=True, hide_index=True)
        with mc2:
            st.markdown("**Unseen Categorical Levels in Val / Test**")
            st.dataframe(pd.DataFrame(diag_mu["unknown_category_diagnostics"]), use_container_width=True, hide_index=True)

        err_analysis = err_model.get("error_analysis_test")
        if err_analysis:
            st.markdown("#### Threshold-Sensitive Error Shift Table")
            st.dataframe(
                pd.DataFrame(err_analysis["threshold_sensitive_error_table"]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Inspectable False Positive & False Negative Customer Snapshots")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f"**Top False Positives (Flagged >= {err_analysis['frozen_threshold_used']}, Retained)**")
                for fp_ex in err_analysis["false_positive_examples"][:4]:
                    with st.expander(f"FP: {fp_ex['account_id']} ({fp_ex['observation_date']}) — Prob: {fp_ex['predicted_prob']:.3f}"):
                        st.json(fp_ex)
            with ec2:
                st.markdown(f"**Top False Negatives (Predicted < {err_analysis['frozen_threshold_used']}, Churned)**")
                for fn_ex in err_analysis["false_negative_examples"][:4]:
                    with st.expander(f"FN: {fn_ex['account_id']} ({fn_ex['observation_date']}) — Prob: {fn_ex['predicted_prob']:.3f}"):
                        st.json(fn_ex)

    # =========================================================================
    # TAB 8: Single/Batch Inference & Artefact Exports
    # =========================================================================
    with tabs[7]:
        st.subheader("8. Single/Batch Inference & Reproducible Artefact Exports")
        inf_m_key = st.selectbox(
            "Select Verified Model Artefact for Inference",
            options=["hist_gradient_boosting", "logistic_regression", "class_prior_baseline"],
            key="inf_model_sel",
        )
        artifact = load_verified_model_artifact(active_protocol_choice, inf_m_key)

        st.markdown("#### A. Interactive Single-Account Snapshot Scorer")
        sample_row = mod_df.iloc[0]
        with st.form("single_inference_form"):
            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            in_mrr = f_c1.number_input("mrr_gbp", min_value=10.0, max_value=100000.0, value=float(sample_row["mrr_gbp"]))
            in_seats = f_c2.number_input("seats_purchased", min_value=1, max_value=5000, value=int(sample_row["seats_purchased"]))
            in_tenure = f_c3.number_input("tenure_months", min_value=1, max_value=120, value=int(sample_row["tenure_months"]))
            in_active_ratio = f_c4.slider("active_seats_ratio_30d", 0.0, 1.0, float(sample_row["active_seats_ratio_30d"]))

            f_c5, f_c6, f_c7, f_c8 = st.columns(4)
            in_urgent = f_c5.number_input("urgent_tickets_90d", min_value=0, max_value=50, value=int(sample_row["urgent_tickets_90d"]))
            in_dunning = f_c6.number_input("dunning_events_6m", min_value=0, max_value=20, value=int(sample_row["dunning_events_6m"]))
            in_plan = f_c7.selectbox("plan_tier", options=["Starter", "Growth", "Enterprise"], index=0)
            in_contract = f_c8.selectbox("contract_type", options=["Monthly", "Annual"], index=0)

            submitted = st.form_submit_button("Score Account Snapshot")
            if submitted:
                single_dict = sample_row[artifact.feature_columns].to_dict()
                single_dict.update(
                    {
                        "mrr_gbp": in_mrr,
                        "seats_purchased": in_seats,
                        "tenure_months": in_tenure,
                        "active_seats_ratio_30d": in_active_ratio,
                        "urgent_tickets_90d": in_urgent,
                        "dunning_events_6m": in_dunning,
                        "plan_tier": in_plan,
                        "contract_type": in_contract,
                    }
                )
                in_df = pd.DataFrame([single_dict])
                pred_out, pred_diag = run_batch_prediction(artifact, in_df)
                local_attr = compute_local_instance_attribution(artifact, in_df.iloc[0], mod_df.head(1000))
                st.success(
                    f"Calibrated 90-Day Churn Probability: **{pred_out['calibrated_churn_probability'].iloc[0]:.4f}** "
                    f"(Frozen Threshold τ* = `{artifact.frozen_threshold:.3f}` → "
                    f"Predicted Label: **{pred_out['predicted_churn_90d'].iloc[0]}** | "
                    f"Risk Band: **{pred_out['risk_band'].iloc[0]}**)"
                )
                st.markdown("**Instance-Level Feature Attributions (vs Training Reference Distribution)**:")
                st.dataframe(pd.DataFrame(local_attr), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### B. Batch CSV Prediction (Schema & Row-Alignment Verified)")
        if st.button("Run Batch Prediction on 100 Held-Out Snapshots"):
            batch_sample = mod_df.tail(100).copy()
            batch_preds, batch_diag = run_batch_prediction(artifact, batch_sample)
            st.json(batch_diag)
            st.dataframe(batch_preds.head(20), use_container_width=True)
            st.download_button(
                "Download Batch Predictions CSV",
                data=batch_preds.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_predictions_{active_protocol_choice}_{inf_m_key}.csv",
                mime="text/csv",
            )

        st.divider()
        st.markdown("#### C. Export Reproducible Evaluation & Audit Dossiers")
        ex1, ex2, ex3 = st.columns(3)
        ex1.download_button(
            "Download Master Evaluation Summary (JSON)",
            data=json.dumps(master, indent=2).encode("utf-8"),
            file_name="master_workbench_summary.json",
            mime="application/json",
        )
        ex2.download_button(
            "Download Active Protocol Audit Report (JSON)",
            data=json.dumps(active_bundle["audit_report"], indent=2).encode("utf-8"),
            file_name=f"audit_{active_protocol_choice}.json",
            mime="application/json",
        )
        comp_csv_path = EXPORTS_DIR / "benchmark_model_comparison.csv"
        if comp_csv_path.exists():
            ex3.download_button(
                "Download Model Comparison Table (CSV)",
                data=comp_csv_path.read_bytes(),
                file_name="benchmark_model_comparison.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
