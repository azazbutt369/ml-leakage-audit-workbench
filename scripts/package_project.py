"""
End-to-End Verification and Clean Project Packaging Script.
Verifies Streamlit module compilation, runs a fresh batch prediction check against persisted
SHA-256 verified models, and creates the clean downloadable ZIP archive:
c:\\Users\\azazb\\gemini_playground\\Azaz_Butt_Tabular_ML_Leakage_Workbench_v1.zip
excluding caches, virtual environments, credentials and temporary binaries.
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from workbench.config import CLEAN_DATA_DIR, EXPORTS_DIR  # noqa: E402
from workbench.storage import load_verified_model_artifact, run_batch_prediction  # noqa: E402


EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git", ".mypy_cache", ".ruff_cache"}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".env", ".pem", ".key"}


def verify_fresh_prediction_and_ui_compilation() -> None:
    # 1. Verify Streamlit app.py AST & syntax compilation
    app_path = SRC_DIR / "workbench" / "app.py"
    source_code = app_path.read_text(encoding="utf-8")
    ast.parse(source_code, filename=str(app_path))
    print(f"[OK] Streamlit UI ({app_path.name}) AST & syntax check passed.")

    # 2. Run a fresh prediction verification pass
    mod_df = pd.read_csv(CLEAN_DATA_DIR / "saas_churn_snapshots_modelling.csv")
    fresh_batch = mod_df.tail(50).copy()
    art = load_verified_model_artifact("protocol_b_temporal_purged", "hist_gradient_boosting")
    preds_df, diag = run_batch_prediction(art, fresh_batch)

    assert diag["row_alignment_verified"] is True
    assert len(preds_df) == 50
    fresh_out = EXPORTS_DIR / "fresh_verification_predictions_50rows.csv"
    preds_df.to_csv(fresh_out, index=False)
    print(
        f"[OK] Fresh prediction run verified: {len(preds_df)} rows scored | "
        f"Mean calibrated prob = {diag['mean_calibrated_probability']:.4f} | "
        f"Flagged at frozen tau* ({diag['threshold_applied']}) = {diag['predicted_positive_count']}"
    )


def build_clean_zip() -> Path:
    zip_path = WORKSPACE_ROOT / "Azaz_Butt_Tabular_ML_Leakage_Workbench_v1.zip"
    if zip_path.exists():
        zip_path.unlink()

    included_files = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(PROJECT_ROOT.rglob("*")):
            if file_path.is_dir():
                continue
            rel_parts = set(file_path.relative_to(PROJECT_ROOT).parts)
            if rel_parts.intersection(EXCLUDED_DIRS):
                continue
            if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue
            arcname = str(Path("tabular-ml-leakage-workbench") / file_path.relative_to(PROJECT_ROOT))
            zf.write(file_path, arcname=arcname)
            included_files.append(arcname)

    size_kb = zip_path.stat().st_size / 1024.0
    print(f"[OK] Packaged {len(included_files)} files into {zip_path} ({size_kb:.1f} KB)")
    for item in included_files:
        print(f"     - {item}")
    return zip_path


if __name__ == "__main__":
    verify_fresh_prediction_and_ui_compilation()
    build_clean_zip()
