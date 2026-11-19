# Portfolio Case Study Description

**Author**: Azaz Butt — AI/ML Engineer & Researcher  
**Project**: Tabular ML and Leakage-Audit Workbench (`tabular-ml-leakage-workbench`)

---

## Short Portfolio Summary (Website / CV / Proposal Snippet)

**Tabular ML & Leakage-Audit Workbench (Python, scikit-learn, FastAPI, Streamlit, SQLite)**  
Designed and implemented an end-to-end tabular machine-learning and evaluation-integrity workbench demonstrated on longitudinal B2B SaaS customer-churn prediction (~15,000 account-month snapshots). Engineered a formal Prediction Contract enforcing point-in-time feature availability and right-censored outcome filtering; implemented dual evaluation protocols comparing **Future-Period Prediction with 90-day outcome-window purging** against **Disjoint Account-Group Generalisation**; and built an **11-rule Leakage-Audit Engine** with SHA-256 `FitProvenanceRecord` row-hash tracking that distinguishes confirmed leakage, suspicious signals, data-quality defects, covariate shift, and unverifiable missing provenance on uploaded CSVs. Validated the detector across 7 controlled leakage and negative-control fixtures (100% recall on injected faults in full-provenance mode, 0 false flags on clean and distribution-shift controls) and packaged calibrated baseline/tree pipelines, account-cluster bootstrap uncertainty, cohort error diagnostics, and schema-validated REST inference into a reproducible FastAPI + Streamlit application.

---

## Key Technical Highlights Demonstrated

1. **Temporal & Group Evaluation Discipline**:
   - Explicitly separates *scoring existing accounts in future periods* (Protocol B, requiring 90-day outcome-window purging at every split cutoff) from *generalising to unseen customer accounts* (Protocol A, requiring strict `account_id` group isolation).
2. **Execution-Backed Leakage Auditing (`LAK-001` – `LAK-011`)**:
   - Records exact SHA-256 row hashes passed into `preprocessor.fit()` across cross-validation folds and final training fits (`FitProvenanceRecord`), catching pre-split imputation/scaling and pre-split resampling from execution evidence rather than relying on pipeline class inspection.
   - Enforces epistemic honesty on external CSV uploads by distinguishing directly verifiable checks from checks requiring upstream lineage metadata.
3. **Calibration, Threshold Freezing & Structure-Aware Uncertainty**:
   - Measures raw calibration (Brier score, 10-bin ECE) separately from out-of-fold Platt recalibration, selects decision thresholds strictly on Development data before evaluating once on the Frozen Test set, and computes 95% confidence intervals via account-cluster bootstrapping.
4. **Production-Ready Engineering & Verification**:
   - Delivered with a FastAPI REST service, an 8-view Streamlit technical workbench, SQLite + SHA-256 model artefact persistence, and a 13-family automated `pytest` verification suite.
