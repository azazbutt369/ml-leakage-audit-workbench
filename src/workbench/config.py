"""
Centralised configuration, constants, computational boundaries and UK English branding
for the Tabular ML and Leakage-Audit Workbench.
"""

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CLEAN_DATA_DIR: Path = DATA_DIR / "clean"
FIXTURES_DIR: Path = DATA_DIR / "fixtures"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
EXPORTS_DIR: Path = ARTIFACTS_DIR / "exports"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
DB_PATH: Path = ARTIFACTS_DIR / "workbench_state.db"

DEFAULT_SEED: int = 20261105

# Computational & Upload Validation Guardrails
MAX_UPLOAD_MB: int = 50
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
MAX_UPLOAD_ROWS: int = 100_000
MIN_REQUIRED_ROWS: int = 50

# Near-Duplicate Search Guardrails (Explicit Computational Bound)
NEAR_DUPLICATE_MAX_PAIR_COMPARISONS: int = 2_500_000
NEAR_DUPLICATE_L2_THRESHOLD: float = 0.06
NEAR_DUPLICATE_MAX_ROWS_PER_SPLIT: int = 1_500

# Bootstrap & Uncertainty Configuration
DEFAULT_BOOTSTRAP_ITERATIONS: int = 250
CONFIDENCE_LEVEL: float = 0.95

# Branding & Epistemic Disclosures (UK English)
BRAND_AUTHOR: str = "Azaz Butt"
BRAND_ROLE: str = "AI/ML Engineer & Researcher"
BRAND_WORKBENCH_NAME: str = "Tabular ML & Leakage-Audit Workbench"
BRAND_SCENARIO_TITLE: str = "B2B SaaS Customer-Churn Prediction (90-Day Horizon)"

SYNTHETIC_BENCHMARK_DISCLOSURE: str = (
    "Synthetic Benchmark Disclosure: This dataset is deterministically generated from a "
    "documented longitudinal B2B SaaS simulation (SEED=20261105). Because the synthetic "
    "generator and evaluation pipelines share authorship, reported metrics demonstrate "
    "pipeline mechanics, temporal/group boundary enforcement and leakage-audit sensitivity. "
    "They do not constitute an independent real-world benchmark or commercial client outcome."
)

CAUSAL_ATTRIBUTION_DISCLOSURE: str = (
    "Non-Causal Attribution Notice: Global permutation importances, linear coefficients and "
    "instance-level marginal attributions describe associations within the fitted predictive "
    "model relative to the training reference distribution. They do not establish causal "
    "effects or guarantee that intervening on a feature will alter customer churn."
)


def ensure_directories() -> None:
    """Ensure all required local data and artefact directories exist."""
    for directory in (
        DATA_DIR,
        CLEAN_DATA_DIR,
        FIXTURES_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        EXPORTS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
