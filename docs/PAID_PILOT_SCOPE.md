# Proposed Paid Pilot Scope: Tabular ML & Evaluation-Integrity Audit

**Prepared by**: Azaz Butt — AI/ML Engineer & Researcher  
**Engagement Type**: Fixed-Scope Technical Pilot (2–3 Weeks)  
**Target Use Cases**: B2B SaaS Churn/Retention, Lead Conversion, Subscription Expansion, Credit/Payment Risk, Operational Forecasting

---

## 1. Objective

Deliver an evidence-backed assessment of whether your existing historical data can reliably predict a defined future business outcome at the moment a decision must be taken—eliminating temporal label bleed, entity overlap and preprocessing leakage before engineering resources are committed to production integration.

---

## 2. Required Client Inputs (Prerequisites)

To commence the pilot on Day 1, the client provides:

1. **Historical Data Extract (CSV or Parquet, up to 1M rows)**:
   - Historical entity snapshots or event tables covering at least $2\times$ the target prediction horizon.
   - Explicit entity/account identifier (`account_id` / `user_id` / `organisation_id`).
   - Explicit observation/scoring timestamp (`observation_date`) and outcome event timestamp (`churned_at` / `converted_at`).
2. **Data Dictionary & Source Lineage Notes**:
   - Brief description of each candidate predictor table (CRM, billing, product telemetry, support desk) and whether columns are point-in-time snapshots or mutable overwritten fields (e.g., `updated_at` vs immutable event logs).
3. **Operational Decision Context**:
   - How often predictions will be generated in production (daily, weekly, monthly), who consumes the scores (CSM team, automated workflow, pricing engine), and approximate intervention capacity or unit cost assumptions.

---

## 3. Pilot Deliverables (End of Week 2 / Week 3)

1. **Signed Prediction Contract (`PredictionContract` JSON & Specification)**:
   - Formal definition of observation unit, eligibility cohort, prediction horizon, right-censoring cutoff rules, and feature availability lag registry.
2. **11-Rule Leakage & Data-Quality Audit Dossier (`AuditReport`)**:
   - Empirical verification of entity isolation, exact/near cross-split duplicates, temporal outcome-window bleed, point-in-time feature availability, label conflicts, and covariate shift (`PSI`/`TVD`).
3. **Calibrated Baseline & Candidate Benchmark Suite**:
   - Reproducible scikit-learn pipelines comparing a naive prior baseline, regularised linear baseline, and CPU-efficient gradient-boosted tree model under the deployment-aligned evaluation protocol.
   - Out-of-fold probability calibration diagnostics, account-cluster bootstrap 95% confidence intervals, and frozen decision-threshold trade-off curves.
4. **Cohort & Failure-Mode Error Analysis**:
   - Performance breakdown across key customer segments and time periods, plus inspectable False Positive and False Negative case profiles.
5. **Working Local Workbench, REST Scoring API & Go/No-Go Recommendation**:
   - Complete packaged repository (FastAPI + Streamlit + `pytest` suite + versioned `.joblib` artefacts) and an executive Go/No-Go briefing detailing whether predictive lift justifies production deployment.

---

## 4. Acceptance Criteria

The pilot is deemed complete and accepted when:

- [ ] The formal Prediction Contract and right-censoring rules are documented and executed against the client dataset.
- [ ] All 11 audit rules (`LAK-001` to `LAK-011`) have been executed and documented with inspectable row/column evidence.
- [ ] Baseline and candidate models are trained strictly within training boundaries (verified via `FitProvenanceRecord` hashes) and evaluated once on a held-out test partition using a development-frozen threshold.
- [ ] All automated verification tests (`pytest`) pass in a clean local virtual environment.
- [ ] The client receives the complete source repository, exported evaluation/audit reports, and executive readout session.

---

## 5. Explicit Exclusions (Out of Scope for the Pilot)

To keep the pilot fast, credible and fixed-cost, the following items are **excluded** from the initial pilot scope (and may be scoped separately in a subsequent Phase 2 production engagement):

1. **Live Production Cloud Infrastructure & CI/CD Provisioning**: Setting up production AWS/GCP/Azure Kubernetes clusters, Terraform pipelines, or live production database write-backs.
2. **Real-Time Streaming Feature Stores**: Re-architecting upstream production Kafka/Spark pipelines or fixing broken historical logging in source operational databases.
3. **Causal Uplift Guarantees**: Predictive churn probabilities rank risk under historical patterns; establishing causal treatment lift of a new retention offer requires a prospective randomised controlled trial (A/B test) in production.
4. **Unbounded Model Zoo / Deep Learning Search**: The pilot focuses on transparent, high-leverage tabular baselines and gradient boosting rather than costly neural architecture searches that rarely outperform well-audited tree ensembles on structured business tables.
