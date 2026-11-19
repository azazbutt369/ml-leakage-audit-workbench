# Tabular ML and Leakage-Audit Workbench

**Author**: Azaz Butt — AI/ML Engineer & Researcher  
**Scenario**: B2B SaaS Customer-Churn Prediction (90-Day Horizon)  
**Version**: `1.0.0`

---

## 1. Overview & Purpose

The **Tabular ML and Leakage-Audit Workbench** is a complete, reproducible machine-learning and evaluation-integrity engineering project built for **Azaz Butt**. It demonstrates how to frame, audit, train, calibrate, and evaluate tabular predictive models without crossing temporal, entity, or preprocessing boundaries.

Using **B2B SaaS customer-churn prediction** as the primary scenario, the application helps engineering and data teams answer five core questions:

1. **Can we predict a defined future churn event (90-day horizon) using only information finalised at prediction time?**
2. **Does the offline evaluation protocol match the intended production deployment scenario?**
3. **Are repeated accounts, duplicate snapshots, overlapping outcome windows, or pre-split transformations contaminating the results?**
4. **Which False Positive and False Negative errors occur across cohorts, and how do frozen decision thresholds affect them?**
5. **What can the model's calibrated probabilities and feature attributions legitimately support?**

> [!IMPORTANT]
> **Same-Author Synthetic Benchmark Disclosure**: The included dataset (`SYNTH-B2B-SAAS-CHURN-V1`, `SEED = 20261105`) is deterministically generated from a documented longitudinal B2B SaaS simulation. Because the data generator and evaluation pipelines share authorship, reported metrics demonstrate pipeline mechanics, boundary enforcement, and audit sensitivity; they do not constitute an independent real-world benchmark or commercial client outcome.

---

## 2. Tested Runtime & Setup Instructions

### Tested Environment
- **OS**: Windows 11 (`win32`)
- **Python**: `3.14.7` (compatible with Python `>=3.10`)
- **Key Libraries**: `pandas 3.0.1`, `numpy 2.4.2`, `scikit-learn 1.8.0`, `scipy 1.17.1`, `fastapi 0.135.1`, `pydantic 2.12.5`, `streamlit 1.55.0`, `plotly 6.6.0`, `joblib 1.5.3`, `pytest 9.0.2`, `httpx 0.28.1`

### 1. Install Dependencies
```powershell
cd tabular-ml-leakage-workbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full Deterministic Pipeline (Data, Splits, Audits, Models & Fixtures)
```powershell
python scripts/run_all.py
```

### 3. Run the 13-Family Automated Verification Suite
```powershell
python -m pytest tests/test_workbench.py -v
```

### 4. Launch the FastAPI Service
```powershell
uvicorn workbench.api:app --host 127.0.0.1 --port 8000
```

### 5. Launch the Streamlit Technical Workbench
```powershell
streamlit run src/workbench/app.py --server.port 8501
```

---

## 3. Actual Measured Benchmark Results

### 3.1 Synthetic Dataset Summary (`SEED = 20261105`)
- **Total Longitudinal Snapshots (`2024-01-01` to `2025-06-01`)**: `15,259`
- **Uncensored Active Snapshots (`2024-01-01` to `2025-04-01`)**: `14,198`
- **Right-Censored Snapshots Excluded (`2025-05-01` to `2025-06-01`, window > `2025-06-30`)**: `1,061`
- **Unique Customer Accounts**: `1,650`
- **90-Day Churn Prevalence (Uncensored)**: `10.66%` (`1,514` positive / `12,684` negative)
- **Modelling Dataset SHA-256**: `5a896989783454d8c2b4e04500b8d6fd537ab3a1087ed89ffc6c25f44e024cf0`

### 3.2 Model Performance Across Protocols (Frozen Test Set)

| Protocol | Model Family | Frozen $\tau^*$ (Dev) | Dev PR-AUC | Test PR-AUC (95% Cluster CI) | Test ROC-AUC (95% Cluster CI) | Test Precision | Test Recall | Test F1 | Test Macro-F1 | Test Brier (Cal / Raw) | Latency (ms / 1k rows) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protocol B (Primary: Future-Period Purged)** | Class-Prior Baseline | `0.050` | `0.1015` | `0.0996` (`[0.0820, 0.1163]`) | `0.5000` (`[0.5000, 0.5000]`) | `0.0996` | `1.0000` | `0.1811` | `0.0906` | `0.0898` / `0.0898` | `14.55` |
| **Protocol B (Primary: Future-Period Purged)** | Logistic Regression (L2) | `0.275` | `0.3927` | `0.3566` (`[0.2700, 0.4509]`) | `0.7738` (`[0.7294, 0.8229]`) | `0.4568` | `0.3008` | `0.3627` | `0.6527` | `0.0777` / `0.0773` | `35.31` |
| **Protocol B (Primary: Future-Period Purged)** | **HistGradientBoosting** | `0.275` | **`0.4108`** | **`0.3597`** (`[0.2789, 0.4551]`) | **`0.7876`** (`[0.7448, 0.8336]`) | `0.3889` | `0.3984` | **`0.3936`** | **`0.6628`** | **`0.0765`** / `0.0759` | `39.52` |
| **Protocol A (Secondary: New-Account Group)** | Class-Prior Baseline | `0.050` | `0.1066` | `0.1067` (`[0.0935, 0.1212]`) | `0.5000` (`[0.5000, 0.5000]`) | `0.1067` | `1.0000` | `0.1929` | `0.0964` | `0.0954` / `0.0954` | `10.96` |
| **Protocol A (Secondary: New-Account Group)** | Logistic Regression (L2) | `0.200` | `0.3463` | `0.2855` (`[0.2377, 0.3420]`) | `0.7434` (`[0.7078, 0.7745]`) | `0.3216` | `0.3596` | `0.3395` | `0.6277` | `0.0866` / `0.0866` | `16.08` |
| **Protocol A (Secondary: New-Account Group)** | **HistGradientBoosting** | `0.175` | **`0.3610`** | **`0.2935`** (`[0.2451, 0.3487]`) | **`0.7515`** (`[0.7153, 0.7837]`) | `0.3345` | `0.4254` | **`0.3745`** | **`0.6441`** | **`0.0857`** / `0.0858` | `35.89` |

### 3.3 Controlled Leakage & Negative-Control Benchmark (`FIX-01` – `FIX-07`)

- **Contaminated Fixtures Detected (Full Provenance Mode)**: **`5 / 5`** (`100%`)
- **Contaminated Fixtures Confirmed from Raw CSV Alone**: **`4 / 5`** (`FIX-02` future feature is flagged as `SUSPICIOUS_SIGNAL` on raw CSV and requires `FeatureSpec` lag provenance to confirm definitively; `FIX-04` pre-split scaling requires `FitProvenanceRecord` to confirm `LAK-007`, while its unpurged temporal bleed is caught directly via `LAK-006`).
- **False Positive Leakage Flags on Clean Controls (`FIX-06`, `FIX-07`)**: **`0`** (`FIX-07` is accurately classified as `DISTRIBUTION_SHIFT` under `LAK-011`).

---

## 4. Key Documentation Links

- [Architecture & Data-Flow Guide](docs/ARCHITECTURE_AND_DATA_FLOW.md)
- [Model Card & Data Card (Including Limitations)](docs/MODEL_AND_DATA_CARDS.md)
- [Troubleshooting & Operational Guide](docs/TROUBLESHOOTING.md)
- [Client Demonstration Script (3–5 Minutes)](docs/CLIENT_DEMO_SCRIPT.md)
- [Portfolio Case Study Description](docs/PORTFOLIO_DESCRIPTION.md)
- [Proposed Paid Pilot Scope](docs/PAID_PILOT_SCOPE.md)
