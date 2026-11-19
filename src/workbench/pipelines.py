"""
Auditable Preprocessing, Model Pipelines, Fold-Isolated Resampling and Out-of-Fold Calibration.
Records exact row hashes, account IDs and timestamps used when fitting every transformation
so the Leakage-Audit Engine can empirically verify train-only fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from workbench.config import DEFAULT_SEED
from workbench.contract import compute_row_hashes


class FitProvenanceRecord(BaseModel):
    """Empirical record of the exact observations used to fit a preprocessing/model pipeline."""

    stage_name: str
    fitted_row_count: int
    fitted_row_hashes: List[str]
    fitted_account_ids: List[str] = Field(default_factory=list)
    fitted_min_observation_date: Optional[str] = None
    fitted_max_observation_date: Optional[str] = None
    fitted_max_outcome_window_end: Optional[str] = None
    feature_columns: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    unknown_category_policy: str
    learned_summary_stats: Dict[str, Any] = Field(default_factory=dict)


class ResamplingProvenanceRecord(BaseModel):
    """Empirical record of where and when class resampling/balancing was executed."""

    applied: bool
    method: str = "none"
    applied_strictly_inside_train_boundary: bool = True
    pre_Split_resampled: bool = False
    original_train_row_count: int = 0
    resampled_train_row_count: int = 0
    pre_resample_positive_rate: float = 0.0
    post_resample_positive_rate: float = 0.0
    source_row_hashes_used: List[str] = Field(default_factory=list)
    val_or_test_rows_touched: int = 0


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    model_type: Literal["linear_or_dummy", "tree"] = "linear_or_dummy",
) -> Tuple[ColumnTransformer, str]:
    """
    Construct an explicit scikit-learn ColumnTransformer with documented missing-value
    and unknown-category behaviour.
    """
    if model_type == "linear_or_dummy":
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        unknown_policy = (
            "OneHotEncoder(handle_unknown='ignore'): unseen categories at inference map to an "
            "all-zero indicator vector without raising runtime errors; missing categorical values "
            "are mapped to explicit '__MISSING__' level."
        )
    else:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-2,
                    ),
                ),
            ]
        )
        unknown_policy = (
            "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2): "
            "unseen categories at inference receive dedicated integer code -1; missing values map to "
            "'__MISSING__' prior to encoding."
        )

    transformers = []
    if numerical_features:
        transformers.append(("num", num_pipe, list(numerical_features)))
    if categorical_features:
        transformers.append(("cat", cat_pipe, list(categorical_features)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, unknown_policy


def extract_preprocessor_learned_stats(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    """Extract inspectable learned parameters (medians, means, category vocabularies) from fitted preprocessor."""
    stats: Dict[str, Any] = {"numerical_medians": {}, "numerical_means": {}, "categorical_vocabularies": {}}
    if "num" in preprocessor.named_transformers_ and numerical_features:
        num_pipe: Pipeline = preprocessor.named_transformers_["num"]
        imputer: SimpleImputer = num_pipe.named_steps["imputer"]
        if hasattr(imputer, "statistics_"):
            for col, val in zip(numerical_features, imputer.statistics_[: len(numerical_features)]):
                stats["numerical_medians"][col] = round(float(val), 6) if not np.isnan(val) else None
        if "scaler" in num_pipe.named_steps:
            scaler: StandardScaler = num_pipe.named_steps["scaler"]
            if hasattr(scaler, "mean_"):
                for col, val in zip(numerical_features, scaler.mean_[: len(numerical_features)]):
                    stats["numerical_means"][col] = round(float(val), 6)

    if "cat" in preprocessor.named_transformers_ and categorical_features:
        cat_pipe: Pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipe.named_steps["encoder"]
        if hasattr(encoder, "categories_"):
            for col, cats in zip(categorical_features, encoder.categories_):
                stats["categorical_vocabularies"][col] = [str(c) for c in cats[:25]]

    return stats


def resample_training_fold_only(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    row_hashes: List[str],
    target_minority_ratio: float = 0.35,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.Series, ResamplingProvenanceRecord]:
    """
    Perform random minority oversampling strictly within the provided training partition/fold.
    Never touches validation or test observations.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_train, dtype=int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    n_orig = len(y_arr)
    orig_pos_rate = float(len(pos_idx) / max(1, n_orig))

    if len(pos_idx) == 0 or orig_pos_rate >= target_minority_ratio:
        return (
            X_train_df.copy(),
            y_train.copy(),
            ResamplingProvenanceRecord(
                applied=False,
                method="none",
                applied_strictly_inside_train_boundary=True,
                pre_Split_resampled=False,
                original_train_row_count=n_orig,
                resampled_train_row_count=n_orig,
                pre_resample_positive_rate=round(orig_pos_rate, 4),
                post_resample_positive_rate=round(orig_pos_rate, 4),
                source_row_hashes_used=list(row_hashes),
                val_or_test_rows_touched=0,
            ),
        )

    # Calculate required positive samples so pos_new / (neg + pos_new) = target_minority_ratio
    desired_pos = int(np.ceil((target_minority_ratio * len(neg_idx)) / (1.0 - target_minority_ratio)))
    extra_pos_needed = max(0, desired_pos - len(pos_idx))
    sampled_pos_idx = rng.choice(pos_idx, size=extra_pos_needed, replace=True)

    combined_idx = np.concatenate([np.arange(n_orig), sampled_pos_idx])
    rng.shuffle(combined_idx)

    X_res = X_train_df.iloc[combined_idx].reset_index(drop=True)
    y_res = y_train.iloc[combined_idx].reset_index(drop=True)
    new_pos_rate = float((y_res == 1).mean())

    return (
        X_res,
        y_res,
        ResamplingProvenanceRecord(
            applied=True,
            method=f"train_fold_minority_oversample(target_ratio={target_minority_ratio})",
            applied_strictly_inside_train_boundary=True,
            pre_Split_resampled=False,
            original_train_row_count=n_orig,
            resampled_train_row_count=int(len(X_res)),
            pre_resample_positive_rate=round(orig_pos_rate, 4),
            post_resample_positive_rate=round(new_pos_rate, 4),
            source_row_hashes_used=list(row_hashes),
            val_or_test_rows_touched=0,
        ),
    )


@dataclass
class AuditedModelArtifact:
    """Complete trained model bundle with fit provenance, CV history, calibrator and frozen threshold."""

    model_key: str
    model_display_name: str
    protocol_id: str
    pipeline: Pipeline
    numerical_features: List[str]
    categorical_features: List[str]
    feature_columns: List[str]
    fit_provenance: FitProvenanceRecord
    cv_fold_provenance: List[FitProvenanceRecord]
    resampling_provenance: ResamplingProvenanceRecord
    tuning_trials: List[Dict[str, Any]]
    selected_hyperparameters: Dict[str, Any]
    selection_criterion: str
    calibrator_method: Literal["none", "sigmoid", "isotonic"] = "none"
    calibrator_model: Optional[Any] = None
    calibration_provenance: Dict[str, Any] = field(default_factory=dict)
    frozen_threshold: float = 0.50
    threshold_objective: str = "max_f1_on_dev"
    threshold_selection_summary: Dict[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SEED

    def predict_raw_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated positive-class probabilities from the base pipeline."""
        X_sub = X_df[self.feature_columns]
        probs = self.pipeline.predict_proba(X_sub)[:, 1]
        return np.clip(np.asarray(probs, dtype=float), 1e-7, 1.0 - 1e-7)

    def predict_calibrated_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Return calibrated positive-class probabilities (or raw if calibrator_method == 'none')."""
        raw_p = self.predict_raw_proba(X_df)
        if self.calibrator_model is None or self.calibrator_method == "none":
            return raw_p
        if self.calibrator_method == "sigmoid":
            logit = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
            cal_p = self.calibrator_model.predict_proba(logit)[:, 1]
            return np.clip(np.asarray(cal_p, dtype=float), 1e-7, 1.0 - 1e-7)
        if self.calibrator_method == "isotonic":
            cal_p = self.calibrator_model.predict(raw_p)
            return np.clip(np.asarray(cal_p, dtype=float), 1e-7, 1.0 - 1e-7)
        return raw_p

    def predict_With_frozen_threshold(self, X_df: pd.DataFrame, use_calibrated: bool = True) -> np.ndarray:
        """Produce binary predictions using the development-frozen decision threshold."""
        probs = self.predict_calibrated_proba(X_df) if use_calibrated else self.predict_raw_proba(X_df)
        return (probs >= self.frozen_threshold).astype(int)


def _create_provenance_record(
    df_used: pd.DataFrame,
    stage_name: str,
    numerical_features: List[str],
    categorical_features: List[str],
    unknown_policy: str,
    preprocessor: ColumnTransformer,
    account_col: str = "account_id",
    date_col: str = "observation_date",
    outcome_end_col: str = "outcome_window_end",
) -> FitProvenanceRecord:
    feature_cols = list(numerical_features) + list(categorical_features)
    if "row_hash" in df_used.columns:
        row_hashes = df_used["row_hash"].astype(str).tolist()
    else:
        hash_cols = ([account_col] if account_col in df_used.columns else []) + (
            [date_col] if date_col in df_used.columns else []
        ) + feature_cols
        row_hashes = compute_row_hashes(df_used, cols=hash_cols).tolist()

    acc_ids = (
        sorted(df_used[account_col].astype(str).unique().tolist())
        if account_col in df_used.columns
        else []
    )
    min_date = str(df_used[date_col].min()) if date_col in df_used.columns and len(df_used) > 0 else None
    max_date = str(df_used[date_col].max()) if date_col in df_used.columns and len(df_used) > 0 else None
    max_out_end = (
        str(df_used[outcome_end_col].max())
        if outcome_end_col in df_used.columns and len(df_used) > 0
        else None
    )

    learned_stats = extract_preprocessor_learned_stats(
        preprocessor, numerical_features, categorical_features
    )

    return FitProvenanceRecord(
        stage_name=stage_name,
        fitted_row_count=int(len(df_used)),
        fitted_row_hashes=row_hashes,
        fitted_account_ids=acc_ids,
        fitted_min_observation_date=min_date,
        fitted_max_observation_date=max_date,
        fitted_max_outcome_window_end=max_out_end,
        feature_columns=feature_cols,
        numerical_features=list(numerical_features),
        categorical_features=list(categorical_features),
        unknown_category_policy=unknown_policy,
        learned_summary_stats=learned_stats,
    )


def fit_and_tune_candidate_models(
    train_df: pd.DataFrame,
    inner_cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    numerical_features: List[str],
    categorical_features: List[str],
    target_col: str = "churned_within_90d",
    protocol_id: str = "protocol_b_temporal_purged",
    account_col: str = "account_id",
    date_col: str = "observation_date",
    outcome_end_col: str = "outcome_window_end",
    seed: int = DEFAULT_SEED,
) -> Dict[str, AuditedModelArtifact]:
    """
    Train and tune the 3 disciplined model candidates strictly on train_df:
      1. class_prior_baseline (DummyClassifier)
      2. logistic_regression (L2 LogisticRegression with small 4-config tuning budget)
      3. hist_gradient_boosting (HistGradientBoostingClassifier with small 4-config tuning budget)

    Also evaluates fold-isolated resampling within the tuning grid to preserve honest comparison
    records (including cases where resampling hurts calibration/PR-AUC).
    Fits out-of-fold sigmoid calibration strictly on inner-CV out-of-fold training predictions.
    """
    feature_cols = list(numerical_features) + list(categorical_features)
    y_train_all = pd.to_numeric(train_df[target_col], errors="raise").astype(int)

    model_specs: List[Dict[str, Any]] = [
        {
            "model_key": "class_prior_baseline",
            "display_name": "Class-Prior Baseline (DummyClassifier)",
            "model_type": "linear_or_dummy",
            "candidates": [
                {
                    "params": {"strategy": "prior"},
                    "resample": False,
                    "estimator_fn": lambda p: DummyClassifier(strategy=p["strategy"], random_state=seed),
                }
            ],
            "calibrate": False,
        },
        {
            "model_key": "logistic_regression",
            "display_name": "Logistic Regression (L2 Regularised)",
            "model_type": "linear_or_dummy",
            "candidates": [
                {
                    "params": {"C": 0.25, "class_weight": None},
                    "resample": False,
                    "estimator_fn": lambda p: LogisticRegression(
                        C=p["C"], class_weight=p["class_weight"], max_iter=1000, random_state=seed
                    ),
                },
                {
                    "params": {"C": 1.0, "class_weight": None},
                    "resample": False,
                    "estimator_fn": lambda p: LogisticRegression(
                        C=p["C"], class_weight=p["class_weight"], max_iter=1000, random_state=seed
                    ),
                },
                {
                    "params": {"C": 1.0, "class_weight": "balanced"},
                    "resample": False,
                    "estimator_fn": lambda p: LogisticRegression(
                        C=p["C"], class_weight=p["class_weight"], max_iter=1000, random_state=seed
                    ),
                },
                {
                    "params": {"C": 1.0, "class_weight": None, "fold_oversample_ratio": 0.35},
                    "resample": True,
                    "estimator_fn": lambda p: LogisticRegression(
                        C=p["C"], class_weight=p["class_weight"], max_iter=1000, random_state=seed
                    ),
                },
            ],
            "calibrate": True,
        },
        {
            "model_key": "hist_gradient_boosting",
            "display_name": "Histogram Gradient Boosting (CPU Tree Ensemble)",
            "model_type": "tree",
            "candidates": [
                {
                    "params": {
                        "max_iter": 140,
                        "learning_rate": 0.05,
                        "max_leaf_nodes": 15,
                        "min_samples_leaf": 40,
                        "l2_regularization": 1.5,
                    },
                    "resample": False,
                    "estimator_fn": lambda p: HistGradientBoostingClassifier(
                        max_iter=p["max_iter"],
                        learning_rate=p["learning_rate"],
                        max_leaf_nodes=p["max_leaf_nodes"],
                        min_samples_leaf=p["min_samples_leaf"],
                        l2_regularization=p["l2_regularization"],
                        random_state=seed,
                    ),
                },
                {
                    "params": {
                        "max_iter": 160,
                        "learning_rate": 0.06,
                        "max_leaf_nodes": 31,
                        "min_samples_leaf": 30,
                        "l2_regularization": 1.0,
                    },
                    "resample": False,
                    "estimator_fn": lambda p: HistGradientBoostingClassifier(
                        max_iter=p["max_iter"],
                        learning_rate=p["learning_rate"],
                        max_leaf_nodes=p["max_leaf_nodes"],
                        min_samples_leaf=p["min_samples_leaf"],
                        l2_regularization=p["l2_regularization"],
                        random_state=seed,
                    ),
                },
                {
                    "params": {
                        "max_iter": 200,
                        "learning_rate": 0.08,
                        "max_leaf_nodes": 63,
                        "min_samples_leaf": 15,
                        "l2_regularization": 0.1,
                    },
                    "resample": False,
                    "estimator_fn": lambda p: HistGradientBoostingClassifier(
                        max_iter=p["max_iter"],
                        learning_rate=p["learning_rate"],
                        max_leaf_nodes=p["max_leaf_nodes"],
                        min_samples_leaf=p["min_samples_leaf"],
                        l2_regularization=p["l2_regularization"],
                        random_state=seed,
                    ),
                },
                {
                    "params": {
                        "max_iter": 140,
                        "learning_rate": 0.05,
                        "max_leaf_nodes": 15,
                        "min_samples_leaf": 40,
                        "l2_regularization": 1.5,
                        "fold_oversample_ratio": 0.35,
                    },
                    "resample": True,
                    "estimator_fn": lambda p: HistGradientBoostingClassifier(
                        max_iter=p["max_iter"],
                        learning_rate=p["learning_rate"],
                        max_leaf_nodes=p["max_leaf_nodes"],
                        min_samples_leaf=p["min_samples_leaf"],
                        l2_regularization=p["l2_regularization"],
                        random_state=seed,
                    ),
                },
            ],
            "calibrate": True,
        },
    ]

    trained_artifacts: Dict[str, AuditedModelArtifact] = {}

    for spec in model_specs:
        m_key = spec["model_key"]
        m_type = spec["model_type"]
        trials_log: List[Dict[str, Any]] = []
        best_score = -1.0
        best_candidate = spec["candidates"][0]
        best_fold_provenance: List[FitProvenanceRecord] = []
        best_oof_preds = np.full(len(train_df), np.nan, dtype=float)
        best_oof_mask = np.zeros(len(train_df), dtype=bool)

        for cand_idx, cand in enumerate(spec["candidates"]):
            fold_pr_aucs: List[float] = []
            fold_briers: List[float] = []
            fold_prov_records: List[FitProvenanceRecord] = []
            oof_preds = np.full(len(train_df), np.nan, dtype=float)
            oof_mask = np.zeros(len(train_df), dtype=bool)

            for fold_i, (tr_idx, vl_idx) in enumerate(inner_cv_splits):
                fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
                fold_val_df = train_df.iloc[vl_idx].reset_index(drop=True)
                y_fold_tr = y_train_all.iloc[tr_idx].reset_index(drop=True)
                y_fold_vl = y_train_all.iloc[vl_idx].reset_index(drop=True)

                fold_hashes = (
                    fold_train_df["row_hash"].astype(str).tolist()
                    if "row_hash" in fold_train_df.columns
                    else compute_row_hashes(fold_train_df, cols=feature_cols).tolist()
                )

                if cand["resample"]:
                    X_fit_df, y_fit_ser, _ = resample_training_fold_only(
                        fold_train_df,
                        y_fold_tr,
                        row_hashes=fold_hashes,
                        target_minority_ratio=0.35,
                        seed=seed + fold_i,
                    )
                else:
                    X_fit_df = fold_train_df
                    y_fit_ser = y_fold_tr

                preproc, unknown_policy = build_preprocessor(
                    numerical_features, categorical_features, model_type=m_type
                )
                clf = cand["estimator_fn"](cand["params"])
                pipe = Pipeline([("preprocessor", preproc), ("classifier", clf)])
                pipe.fit(X_fit_df[feature_cols], y_fit_ser)

                # Record exact fold training provenance
                fold_prov_records.append(
                    _create_provenance_record(
                        df_used=fold_train_df,
                        stage_name=f"{m_key}_cv_fold_{fold_i}",
                        numerical_features=numerical_features,
                        categorical_features=categorical_features,
                        unknown_policy=unknown_policy,
                        preprocessor=pipe.named_steps["preprocessor"],
                        account_col=account_col,
                        date_col=date_col,
                        outcome_end_col=outcome_end_col,
                    )
                )

                val_probs = pipe.predict_proba(fold_val_df[feature_cols])[:, 1]
                oof_preds[vl_idx] = val_probs
                oof_mask[vl_idx] = True

                fold_pr_aucs.append(float(average_precision_score(y_fold_vl, val_probs)))
                fold_briers.append(float(brier_score_loss(y_fold_vl, val_probs)))

            mean_pr_auc = float(np.mean(fold_pr_aucs))
            std_pr_auc = float(np.std(fold_pr_aucs))
            mean_brier = float(np.mean(fold_briers))

            trials_log.append(
                {
                    "trial_index": cand_idx,
                    "params": cand["params"],
                    "fold_isolated_resampling": bool(cand["resample"]),
                    "mean_cv_pr_auc": round(mean_pr_auc, 5),
                    "std_cv_pr_auc": round(std_pr_auc, 5),
                    "mean_cv_brier": round(mean_brier, 5),
                    "selected": False,
                }
            )

            # Selection rule: maximise mean inner-CV PR-AUC, breaking ties with lower Brier score
            if (mean_pr_auc - best_score) > 1e-4 or (
                abs(mean_pr_auc - best_score) <= 1e-4
                and not cand["resample"]
            ):
                best_score = mean_pr_auc
                best_candidate = cand
                best_fold_provenance = fold_prov_records
                best_oof_preds = oof_preds
                best_oof_mask = oof_mask

        # Mark selected trial
        for t in trials_log:
            if t["params"] == best_candidate["params"] and t["fold_isolated_resampling"] == bool(
                best_candidate["resample"]
            ):
                t["selected"] = True
                break

        # Fit final pipeline on full train_df (strictly inside train boundary)
        train_hashes = (
            train_df["row_hash"].astype(str).tolist()
            if "row_hash" in train_df.columns
            else compute_row_hashes(train_df, cols=feature_cols).tolist()
        )
        if best_candidate["resample"]:
            X_final_fit, y_final_fit, res_prov = resample_training_fold_only(
                train_df,
                y_train_all,
                row_hashes=train_hashes,
                target_minority_ratio=0.35,
                seed=seed,
            )
        else:
            X_final_fit = train_df
            y_final_fit = y_train_all
            res_prov = ResamplingProvenanceRecord(
                applied=False,
                method="none",
                applied_strictly_inside_train_boundary=True,
                pre_Split_resampled=False,
                original_train_row_count=int(len(train_df)),
                resampled_train_row_count=int(len(train_df)),
                pre_resample_positive_rate=round(float(y_train_all.mean()), 4),
                post_resample_positive_rate=round(float(y_train_all.mean()), 4),
                source_row_hashes_used=train_hashes,
                val_or_test_rows_touched=0,
            )

        final_preproc, unknown_policy = build_preprocessor(
            numerical_features, categorical_features, model_type=m_type
        )
        final_clf = best_candidate["estimator_fn"](best_candidate["params"])
        final_pipe = Pipeline([("preprocessor", final_preproc), ("classifier", final_clf)])
        final_pipe.fit(X_final_fit[feature_cols], y_final_fit)

        fit_prov = _create_provenance_record(
            df_used=train_df,
            stage_name=f"{m_key}_full_train_fit",
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            unknown_policy=unknown_policy,
            preprocessor=final_pipe.named_steps["preprocessor"],
            account_col=account_col,
            date_col=date_col,
            outcome_end_col=outcome_end_col,
        )

        # Fit out-of-fold probability calibrator strictly on inner-CV out-of-fold predictions
        cal_method: Literal["none", "sigmoid", "isotonic"] = "none"
        cal_model: Optional[Any] = None
        cal_prov: Dict[str, Any] = {
            "fitted_on": "none",
            "oof_sample_count": 0,
            "oof_brier_before": None,
            "oof_brier_after": None,
        }

        if spec["calibrate"] and int(best_oof_mask.sum()) >= 100:
            valid_oof_p = np.clip(best_oof_preds[best_oof_mask], 1e-6, 1.0 - 1e-6)
            valid_oof_y = y_train_all.iloc[ np.where(best_oof_mask)[0] ].to_numpy(dtype=int)
            if len(np.unique(valid_oof_y)) == 2:
                brier_before = float(brier_score_loss(valid_oof_y, valid_oof_p))
                # Fit Platt scaling (LogisticRegression on OOF logit)
                oof_logits = np.log(valid_oof_p / (1.0 - valid_oof_p)).reshape(-1, 1)
                platt = LogisticRegression(C=1e4, solver="lbfgs", random_state=seed)
                platt.fit(oof_logits, valid_oof_y)
                platt_p = platt.predict_proba(oof_logits)[:, 1]
                brier_platt = float(brier_score_loss(valid_oof_y, platt_p))

                cal_method = "sigmoid"
                cal_model = platt
                cal_prov = {
                    "fitted_on": "inner_cv_out_of_fold_train_predictions_only",
                    "calibrator_type": "Platt Sigmoid (LogisticRegression on OOF Logits)",
                    "oof_sample_count": int(best_oof_mask.sum()),
                    "oof_brier_before_recalibration": round(brier_before, 5),
                    "oof_brier_after_recalibration": round(brier_platt, 5),
                    "val_or_test_rows_used": 0,
                }

        trained_artifacts[m_key] = AuditedModelArtifact(
            model_key=m_key,
            model_display_name=spec["display_name"],
            protocol_id=protocol_id,
            pipeline=final_pipe,
            numerical_features=list(numerical_features),
            categorical_features=list(categorical_features),
            feature_columns=feature_cols,
            fit_provenance=fit_prov,
            cv_fold_provenance=best_fold_provenance,
            resampling_provenance=res_prov,
            tuning_trials=trials_log,
            selected_hyperparameters=best_candidate["params"],
            selection_criterion="Max Inner-CV PR-AUC (Average Precision) strictly on Training folds",
            calibrator_method=cal_method,
            calibrator_model=cal_model,
            calibration_provenance=cal_prov,
            seed=seed,
        )

    return trained_artifacts
