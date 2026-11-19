# Client Demonstration Script (3–5 Minutes)

**Presenter**: Azaz Butt — AI/ML Engineer & Researcher  
**Target Audience**: CTOs, Heads of Data/AI, Founders at AI Startups, Software Agencies & B2B SaaS Businesses  
**Duration**: 4 Minutes (Live Local Walkthrough)

---

## 0:00 – 0:45 | Framing the Commercial Problem (Tab 1: Contract & Data)

> *"When B2B SaaS teams build customer-churn or expansion models, the hardest problem isn't choosing between XGBoost, LightGBM or Logistic Regression—it's making sure the offline evaluation actually matches how the model will be used in production.*
>
> *In this workbench, we model 90-day subscription churn across 15,000 longitudinal monthly account snapshots. Before touching a model, Tab 1 locks a formal **Prediction Contract**: every observation is an active account on the 1st of a month, every feature has a documented availability lag on or before prediction time, and snapshots from May and June 2025—whose 90-day outcome windows haven't finished by our June 30th cutoff—are explicitly flagged as right-censored and excluded from training."*

---

## 0:45 – 1:45 | Choosing the Right Evaluation Protocol & Auditing for Leakage (Tab 3 & Tab 4)

> *"Next, in the sidebar and Tab 3, we distinguish two completely different deployment questions:*
> - ***Protocol B (our Primary Benchmark)*** *answers: 'Can we score our active customer base forward in time next month?' Crucially, notice the **90-day outcome-window purge buffer**: training stops in August 2024 because September and October snapshots have 90-day churn outcomes that extend into November and December. If you don't purge those overlapping windows, your training set uses future labels that wouldn't exist yet on the cutoff date.*
> - ***Protocol A (our Secondary Benchmark)*** *answers: 'How well do we generalise to brand-new customer organisations?' by keeping all snapshots of an `account_id` inside a single partition.*
>
> *In Tab 4 (**Controlled Leakage Lab**), we test our 11-rule Leakage-Audit Engine against 5 deliberate faults and 2 clean controls. It catches repeated accounts crossing splits (`LAK-001`), post-observation retention credits (`LAK-004`/`LAK-005`), cross-split duplicates (`LAK-002`/`LAK-003`), pre-split scaling via SHA-256 fit-provenance hashes (`LAK-007`), and pre-split minority oversampling (`LAK-008`)—while producing zero false flags on our clean negative control and correctly classifying macro cohort drift (`FIX-07`) as **Distribution Shift** rather than leakage."*

---

## 1:45 – 2:45 | What Happens When a Client Uploads a Raw CSV? (Tab 2)

> *"In Tab 2, when you upload an external CSV, the workbench validates schema and target integrity, and immediately displays the **Provenance Capability Matrix**. Rather than giving a false green tick, it shows which rules can be verified directly from the flat table—like group overlap and duplicate rows—and which rules are marked `UNVERIFIABLE_MISSING_PROVENANCE` because a flat CSV doesn't carry feature-store event timestamps or transformation fit logs."*

---

## 2:45 – 3:45 | Baselines, Calibration & Frozen Thresholds (Tab 5 & Tab 6)

> *"In Tab 5 and Tab 6, we compare a Class-Prior baseline, L2 Logistic Regression, and CPU-friendly Histogram Gradient Boosting. Every transformation is fitted strictly inside the training folds, uncertainty is estimated via **account-cluster bootstrapping** (respecting repeated snapshots per customer), and probability calibration is fitted strictly on out-of-fold training predictions.*
>
> *In Tab 6, we explore precision/recall trade-offs and hypothetical outreach costs on the **Development** partition, freeze the chosen decision threshold $\tau^*$, and report a single unmanipulated evaluation on the **Frozen Test** partition."*

---

## 3:45 – 4:30 | Error Analysis, Production Scoring & Paid Pilot Scope (Tab 7 & Tab 8)

> *"Finally, Tab 7 lets your team inspect actual False Positives and False Negatives by plan tier, contract cycle and tenure band—with explicit non-causal attribution boundaries—and Tab 8 runs schema-validated single and batch inference backed by FastAPI and SHA-256 verified model artefacts.*
>
> *For client engagements, I use this exact workbench structure to run a focused 2-to-3-week **Tabular ML & Leakage-Audit Pilot** on your historical warehouse extract—delivering a signed prediction contract, a full leakage audit report, calibrated baselines, and a deployment readiness recommendation."*
