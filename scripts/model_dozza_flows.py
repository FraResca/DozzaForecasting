#!/usr/bin/env python3
"""Feature selection e impianto predittivo per ingressi/uscite Dozza.

Lo script parte dal dataset unito prodotto da `analyze_dozza_datasets.py`
e valuta modelli di regressione con split temporale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, PoissonRegressor, Ridge, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dozza_modeling.interpretability import (
    feature_group_table,
    group_ablation_study,
    intrinsic_importance_for_best_models,
    save_explanatory_figures,
    save_importance_plots,
    shap_for_best_models,
    single_feature_ablation_study,
)
from dozza_modeling.plotting import configure_paper_plots, save_figure
from dozza_modeling.targets import (
    CALENDAR_FEATURE_COLUMNS,
    FLOW_TARGETS,
    FEATURE_SCOPE_CHOICES,
    TargetSet,
    candidate_feature_allowed,
    feature_group,
    historical_source_columns,
    resolve_target_set,
    resolved_feature_scope,
    target_set_choices,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_paper_plots()


TARGET_COLUMNS = FLOW_TARGETS
CYCLICAL_TIME_FEATURE_COLUMNS = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"}
TIME_SERIES_BASELINE_MODELS = {
    "last_hour",
    "same_hour_previous_day",
    "same_hour_previous_week",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "sarimax",
    "prophet",
}
FEATURE_BASELINE_MODELS = {"dummy_mean", "dummy_median", "log1p_ridge", "poisson", "tweedie"}
TWO_STAGE_MODELS = {"two_stage", "two_stage_ridge"}
DEFAULT_MASE_SEASONALITY = 24
DEFAULT_PEAK_QUANTILE = 0.90

GENERATED_OUTPUT_FILES = {
    "ablation_study.csv",
    "best_feature_model_metrics.csv",
    "best_model_metrics.csv",
    "bootstrap_metric_intervals.csv",
    "feature_groups.csv",
    "feature_selection_dropped.csv",
    "feature_selection_ranking.csv",
    "model_feature_importance.csv",
    "model_metrics.csv",
    "modeling_metadata.json",
    "modeling_report.md",
    "permutation_importance.csv",
    "residuals_by_hour.csv",
    "rolling_validation_metrics.csv",
    "rolling_validation_rank_stability.csv",
    "rolling_validation_selected_features.csv",
    "rolling_validation_summary.csv",
    "selected_feature_correlation_matrix.csv",
    "selected_feature_correlation_matrix.png",
    "selected_feature_correlation_matrix.pdf",
    "selected_feature_target_correlations.csv",
    "selected_feature_target_correlations.png",
    "selected_feature_target_correlations.pdf",
    "selected_features.csv",
    "shap_importance.csv",
    "shap_status.json",
    "single_feature_ablation.csv",
    "target_distribution_train_test.png",
    "target_distribution_train_test.pdf",
    "target_timeseries_train_test.png",
    "target_timeseries_train_test.pdf",
    "test_predictions.csv",
    "top_k_auto_selection.csv",
    "top_k_validation_best.csv",
    "top_k_validation_metrics.csv",
    "top_k_validation_report.md",
    "top_k_validation_rolling_metrics.csv",
    "top_k_validation_rolling_summary.csv",
    "top_k_validation_selected_features.csv",
    "top_k_validation_summary.csv",
    "top_k_validation_tuning_results.csv",
    "tuning_results.csv",
}
GENERATED_OUTPUT_GLOBS = (
    "ablation_*.png",
    "ablation_*.pdf",
    "model_feature_importance_*.png",
    "model_feature_importance_*.pdf",
    "model_metric_*.png",
    "model_metric_*.pdf",
    "permutation_importance_*.png",
    "permutation_importance_*.pdf",
    "plot_*.png",
    "plot_*.pdf",
    "selected_feature_groups.png",
    "selected_feature_groups.pdf",
    "shap_importance_*.png",
    "shap_importance_*.pdf",
)


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    ranking: pd.DataFrame
    dropped: pd.DataFrame
    filtered_features: list[str]


@dataclass
class ModelingDataset:
    frame: pd.DataFrame
    observed_targets: pd.DataFrame


def positive_int_list(value: str) -> list[int]:
    if not value:
        return []
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("I valori devono essere interi > 0.")
        items.append(parsed)
    return sorted(set(items))


def clean_generated_output_dir(output_dir: Path) -> list[Path]:
    """Rimuove artefatti generati da run precedenti nella cartella output."""
    removed: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if model_dir.exists():
        shutil.rmtree(model_dir)
        removed.append(model_dir)

    candidates = {output_dir / name for name in GENERATED_OUTPUT_FILES}
    for pattern in GENERATED_OUTPUT_GLOBS:
        candidates.update(output_dir.glob(pattern))

    for path in sorted(candidates):
        if not path.exists() or path.is_dir():
            continue
        path.unlink()
        removed.append(path)
    return removed


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    values = np.zeros_like(denom, dtype=float)
    np.divide(
        2.0 * np.abs(y_pred - y_true),
        denom,
        out=values,
        where=denom != 0,
    )
    return float(np.mean(values) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_pred - y_true)) / denom * 100)


def mase_denominator(y_train: pd.Series, seasonality: int) -> float:
    values = y_train.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return np.nan
    lag = seasonality if len(values) > seasonality else 1
    diffs = np.abs(values[lag:] - values[:-lag])
    denom = float(np.mean(diffs)) if len(diffs) else np.nan
    return denom if denom > 0 else np.nan


def peak_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    actual_peak = y_true >= threshold
    pred_peak = y_pred >= threshold
    tp = int(np.logical_and(actual_peak, pred_peak).sum())
    fp = int(np.logical_and(~actual_peak, pred_peak).sum())
    fn = int(np.logical_and(actual_peak, ~pred_peak).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
        else np.nan
    )
    return {
        "peak_threshold": float(threshold),
        "peak_support": int(actual_peak.sum()),
        "peak_predicted": int(pred_peak.sum()),
        "peak_precision": float(precision) if not np.isnan(precision) else np.nan,
        "peak_recall": float(recall) if not np.isnan(recall) else np.nan,
        "peak_f1": float(f1) if not np.isnan(f1) else np.nan,
    }


def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "year" not in out:
        out["year"] = out["timestamp"].dt.year
    if "month" not in out:
        out["month"] = out["timestamp"].dt.month
    if "day" not in out:
        out["day"] = out["timestamp"].dt.day
    if "hour" not in out:
        out["hour"] = out["timestamp"].dt.hour
    if "dayofweek" not in out:
        out["dayofweek"] = out["timestamp"].dt.dayofweek
    if "is_weekend" not in out:
        out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def parse_lags(value: str) -> list[int]:
    if not value:
        return []
    lags = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        lag = int(raw)
        if lag <= 0:
            raise argparse.ArgumentTypeError("I lag devono essere interi > 0.")
        lags.append(lag)
    return sorted(set(lags))


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("Specificare almeno un modello.")
    return models


def create_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    targets: list[str],
    include_targets: bool,
) -> pd.DataFrame:
    if not windows:
        return df

    out = df.sort_values("timestamp").copy()
    source_cols = historical_source_columns(out.columns, targets)
    if include_targets:
        source_cols.extend([target for target in targets if target in out.columns])
    source_cols = sorted(set(source_cols))
    if not source_cols:
        return out

    indexed = out.set_index("timestamp")
    rolling_parts = []
    for window in windows:
        rolled = indexed[source_cols].rolling(f"{window}h", closed="left", min_periods=1).mean()
        rolled = rolled.rename(columns={col: f"{col}_roll{window}h_mean" for col in source_cols})
        rolling_parts.append(rolled)

    rolling_df = pd.concat(rolling_parts, axis=1).reset_index()
    return out.merge(rolling_df, on="timestamp", how="left", validate="one_to_one")


def create_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    targets: list[str],
    include_targets: bool,
) -> pd.DataFrame:
    if not lags:
        return df

    out = df.copy()
    source_cols = historical_source_columns(out.columns, targets)
    if include_targets:
        source_cols.extend([target for target in targets if target in out.columns])
    source_cols = sorted(set(source_cols))

    for lag in lags:
        lagged = out[["timestamp", *source_cols]].copy()
        lagged["timestamp"] = lagged["timestamp"] + pd.to_timedelta(lag, unit="h")
        lagged = lagged.rename(columns={col: f"{col}_lag{lag}h" for col in source_cols})
        out = out.merge(lagged, on="timestamp", how="left", validate="one_to_one")
    return out


def shift_targets_for_horizon(df: pd.DataFrame, targets: list[str], horizon_hours: int) -> pd.DataFrame:
    if horizon_hours == 0:
        return df
    present_targets = [target for target in targets if target in df.columns]
    future = df[["timestamp", *present_targets]].copy()
    future["timestamp"] = future["timestamp"] - pd.to_timedelta(horizon_hours, unit="h")
    future = future.rename(columns={target: f"{target}_future" for target in present_targets})
    without_targets = df.drop(columns=present_targets, errors="ignore")
    merged = without_targets.merge(future, on="timestamp", how="inner", validate="one_to_one")
    rename_back = {f"{target}_future": target for target in present_targets}
    return merged.rename(columns=rename_back)


def load_modeling_dataset(
    input_csv: Path,
    targets: list[str],
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    include_target_lags: bool,
    include_target_rolling: bool,
    max_rows: int | None,
) -> ModelingDataset:
    df = pd.read_csv(input_csv)
    df = ensure_time_features(df)
    observed_targets = df[["timestamp", *[target for target in targets if target in df.columns]]].copy()
    df = create_rolling_features(
        df,
        windows=rolling_windows,
        targets=targets,
        include_targets=include_target_rolling,
    )
    df = create_lag_features(
        df,
        lags=lags,
        targets=targets,
        include_targets=include_target_lags,
    )
    df = shift_targets_for_horizon(df, targets=targets, horizon_hours=horizon_hours)
    df = df.dropna(subset=[target for target in targets if target in df]).reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return ModelingDataset(frame=df, observed_targets=observed_targets)


def candidate_feature_columns(
    df: pd.DataFrame,
    mode: str,
    targets: list[str],
    target_set: TargetSet,
    feature_scope: str,
    allow_target_history: bool,
) -> list[str]:
    features = []
    for col in df.columns:
        if col == "timestamp":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if candidate_feature_allowed(
            feature=col,
            targets=targets,
            target_set=target_set,
            mode=mode,
            feature_scope=feature_scope,
            allow_target_history=allow_target_history,
        ):
            features.append(col)
    return features


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    embargo_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size deve essere tra 0 e 1.")
    if embargo_hours < 0:
        raise ValueError("--embargo-hours deve essere >= 0.")
    split_idx = max(1, int(len(df) * (1 - test_size)))
    split_idx = min(split_idx, len(df) - 1)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if embargo_hours > 0:
        test_start = pd.to_datetime(test_df["timestamp"]).min()
        train_cutoff = test_start - pd.to_timedelta(embargo_hours, unit="h")
        train_df = train_df[pd.to_datetime(train_df["timestamp"]) < train_cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporale vuoto dopo embargo. Ridurre --test-size o --embargo-hours."
        )
    return train_df, test_df


def select_features(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    targets: list[str],
    missing_threshold: float,
    collinearity_threshold: float,
    top_k: int | None,
    random_state: int,
) -> FeatureSelectionResult:
    print(f"[FEATURE] Valuto {len(candidate_features)} feature candidate", flush=True)
    dropped_rows: list[dict[str, Any]] = []
    filtered = []
    total = len(train_df)

    for feature in candidate_features:
        missing_pct = float(train_df[feature].isna().sum() / total) if total else 1.0
        nunique = int(train_df[feature].nunique(dropna=True))
        if missing_pct > missing_threshold:
            dropped_rows.append(
                {"feature": feature, "reason": "too_many_missing", "value": missing_pct}
            )
            continue
        if nunique <= 1:
            dropped_rows.append({"feature": feature, "reason": "constant", "value": nunique})
            continue
        filtered.append(feature)

    if not filtered:
        raise ValueError("Nessuna feature disponibile dopo i filtri iniziali.")

    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(train_df[filtered]),
        columns=filtered,
        index=train_df.index,
    )
    target_corr = {}
    for feature in filtered:
        values = []
        for target in targets:
            pair = pd.concat([imputed[feature], train_df[target]], axis=1).dropna()
            if len(pair) < 3 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
                values.append(0.0)
            else:
                values.append(abs(pair[feature].corr(pair[target], method="pearson")))
        target_corr[feature] = max(values) if values else 0.0

    kept = list(filtered)
    if len(kept) >= 2:
        corr = imputed[kept].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in list(upper.columns):
            high_pairs = upper.index[upper[col] > collinearity_threshold].tolist()
            for row in high_pairs:
                if row not in kept or col not in kept:
                    continue
                drop_col = row if target_corr.get(row, 0.0) < target_corr.get(col, 0.0) else col
                keep_col = col if drop_col == row else row
                kept.remove(drop_col)
                dropped_rows.append(
                    {
                        "feature": drop_col,
                        "reason": "high_collinearity",
                        "value": float(corr.loc[row, col]),
                        "kept_instead": keep_col,
                    }
                )

    imputed_kept = imputed[kept]
    ranking_rows = []
    for feature in kept:
        pearson_scores = []
        mi_scores = []
        for target in targets:
            pair = pd.concat([imputed_kept[feature], train_df[target]], axis=1).dropna()
            if len(pair) < 3 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
                pearson = 0.0
                mi = 0.0
            else:
                pearson = abs(pair[feature].corr(pair[target], method="pearson"))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi = mutual_info_regression(
                        pair[[feature]],
                        pair[target],
                        random_state=random_state,
                        discrete_features=False,
                    )[0]
            pearson_scores.append(float(pearson))
            mi_scores.append(float(mi))
        ranking_rows.append(
            {
                "feature": feature,
                "max_abs_pearson_to_target": max(pearson_scores) if pearson_scores else 0.0,
                "mean_abs_pearson_to_targets": float(np.mean(pearson_scores)) if pearson_scores else 0.0,
                "mean_mutual_info_to_targets": float(np.mean(mi_scores)) if mi_scores else 0.0,
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise ValueError("Ranking feature vuoto.")

    for score_col in ["mean_abs_pearson_to_targets", "mean_mutual_info_to_targets"]:
        max_value = ranking[score_col].max()
        ranking[f"{score_col}_norm"] = ranking[score_col] / max_value if max_value > 0 else 0.0
    ranking["selection_score"] = (
        0.55 * ranking["mean_abs_pearson_to_targets_norm"]
        + 0.45 * ranking["mean_mutual_info_to_targets_norm"]
    )
    ranking = ranking.sort_values("selection_score", ascending=False).reset_index(drop=True)

    selected = ranking["feature"].tolist()
    if top_k is not None:
        selected = selected[: max(1, top_k)]
    print(
        f"[FEATURE] Selezionate {len(selected)} feature "
        f"({len(dropped_rows)} scartate)",
        flush=True,
    )

    dropped = pd.DataFrame(dropped_rows)
    if dropped.empty:
        dropped = pd.DataFrame(columns=["feature", "reason", "value", "kept_instead"])
    return FeatureSelectionResult(
        selected_features=selected,
        ranking=ranking,
        dropped=dropped,
        filtered_features=kept,
    )


class TwoStageRidgeRegressor(BaseEstimator, RegressorMixin):
    """Modello a due stadi con classificatore e regressori per valori bassi/alti."""

    def __init__(
        self,
        threshold_quantile: float = 0.75,
        alpha: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.threshold_quantile = threshold_quantile
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "TwoStageRidgeRegressor":
        y_values = np.asarray(y, dtype=float)
        self.threshold_ = float(np.quantile(y_values, self.threshold_quantile))
        high_mask = y_values >= self.threshold_
        self.fallback_ = float(np.mean(y_values)) if len(y_values) else 0.0
        self.has_two_classes_ = bool(high_mask.any() and (~high_mask).any())

        self.classifier_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.low_regressor_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.high_regressor_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.global_regressor_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.global_regressor_.fit(X, y_values)
        if self.has_two_classes_:
            self.classifier_.fit(X, high_mask.astype(int))
            self.low_regressor_.fit(X.loc[~high_mask], y_values[~high_mask])
            self.high_regressor_.fit(X.loc[high_mask], y_values[high_mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_regressor_"):
            raise ValueError("Il modello two-stage non e' stato addestrato.")
        if not getattr(self, "has_two_classes_", False):
            return np.clip(self.global_regressor_.predict(X), a_min=0, a_max=None)
        high_probability = self.classifier_.predict_proba(X)[:, 1]
        low_pred = self.low_regressor_.predict(X)
        high_pred = self.high_regressor_.predict(X)
        pred = (1.0 - high_probability) * low_pred + high_probability * high_pred
        return np.clip(pred, a_min=0, a_max=None)


def build_model(
    model_name: str,
    random_state: int,
    quick: bool,
    params: dict[str, Any] | None = None,
) -> Any:
    params = params or {}
    if model_name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if model_name == "dummy_median":
        return DummyRegressor(strategy="median")
    if model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params.get("alpha", 10.0))),
            ]
        )
    if model_name == "log1p_ridge":
        base = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params.get("alpha", 10.0))),
            ]
        )
        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    if model_name in TWO_STAGE_MODELS:
        return TwoStageRidgeRegressor(
            threshold_quantile=params.get("threshold_quantile", 0.75),
            alpha=params.get("alpha", 10.0),
            random_state=random_state,
        )
    if model_name == "elasticnet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20_000, random_state=random_state)),
            ]
        )
    if model_name == "poisson":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=params.get("alpha", 0.1), max_iter=2000)),
            ]
        )
    if model_name == "tweedie":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    TweedieRegressor(
                        power=params.get("power", 1.3),
                        alpha=params.get("alpha", 0.1),
                        max_iter=2000,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60 if quick else 400,
                        min_samples_leaf=params.get("min_samples_leaf", 3),
                        max_features="sqrt",
                        max_depth=params.get("max_depth"),
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=80 if quick else 500,
                        min_samples_leaf=params.get("min_samples_leaf", 3),
                        max_features=params.get("max_features", "sqrt"),
                        max_depth=params.get("max_depth"),
                        bootstrap=params.get("bootstrap", False),
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=80 if quick else 350,
            learning_rate=params.get("learning_rate", 0.05),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            l2_regularization=params.get("l2_regularization", 0.05),
            random_state=random_state,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost non installato.") from exc
        return XGBRegressor(
            n_estimators=80 if quick else 500,
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.85),
            colsample_bytree=params.get("colsample_bytree", 0.85),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm non installato.") from exc
        return LGBMRegressor(
            n_estimators=80 if quick else 500,
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 10),
            subsample=params.get("subsample", 0.85),
            colsample_bytree=params.get("colsample_bytree", 0.85),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Modello non riconosciuto: {model_name}")


def tuning_grid(model_name: str, quick: bool) -> list[dict[str, Any]]:
    if model_name in {"dummy_mean", "dummy_median"}:
        return [{}]
    if model_name == "ridge":
        return [{"alpha": value} for value in ([1.0, 10.0] if quick else [0.1, 1.0, 10.0, 50.0])]
    if model_name == "log1p_ridge":
        return [{"alpha": value} for value in ([1.0, 10.0] if quick else [0.1, 1.0, 10.0, 50.0])]
    if model_name in TWO_STAGE_MODELS:
        quantiles = [0.75] if quick else [0.70, 0.75, 0.85]
        alphas = [10.0] if quick else [1.0, 10.0]
        return [{"threshold_quantile": q, "alpha": alpha} for q, alpha in product(quantiles, alphas)]
    if model_name == "poisson":
        return [{"alpha": value} for value in ([0.05, 0.5] if quick else [0.01, 0.05, 0.1, 0.5])]
    if model_name == "tweedie":
        powers = [1.2, 1.5] if quick else [1.1, 1.3, 1.5]
        alphas = [0.05, 0.5] if quick else [0.01, 0.1, 0.5]
        return [{"power": power, "alpha": alpha} for power, alpha in product(powers, alphas)]
    if model_name == "hist_gradient_boosting":
        learning_rates = [0.05, 0.08] if quick else [0.03, 0.05, 0.08]
        leaves = [31] if quick else [15, 31]
        return [
            {"learning_rate": lr, "max_leaf_nodes": leaf, "l2_regularization": 0.05}
            for lr, leaf in product(learning_rates, leaves)
        ]
    if model_name == "lightgbm":
        learning_rates = [0.05, 0.08] if quick else [0.03, 0.05, 0.08]
        leaves = [31] if quick else [15, 31]
        return [
            {"learning_rate": lr, "num_leaves": leaf, "min_child_samples": 10}
            for lr, leaf in product(learning_rates, leaves)
        ]
    if model_name == "xgboost":
        learning_rates = [0.05, 0.08] if quick else [0.03, 0.05, 0.08]
        depths = [3] if quick else [2, 3, 4]
        return [{"learning_rate": lr, "max_depth": depth} for lr, depth in product(learning_rates, depths)]
    if model_name == "random_forest":
        leaves = [3] if quick else [1, 3, 8]
        return [{"min_samples_leaf": leaf, "max_depth": None} for leaf in leaves]
    if model_name == "extra_trees":
        leaves = [3] if quick else [1, 3, 8]
        max_features = ["sqrt"] if quick else ["sqrt", 0.7]
        return [
            {"min_samples_leaf": leaf, "max_depth": None, "max_features": mf}
            for leaf, mf in product(leaves, max_features)
        ]
    if model_name == "elasticnet":
        return [{}]
    return [{}]


def tune_model_params(
    train_df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_names: list[str],
    random_state: int,
    quick: bool,
    validation_size: float,
    embargo_hours: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if not 0 < validation_size < 1:
        raise ValueError("--tune-validation-size deve essere tra 0 e 1.")
    if len(train_df) < 50:
        return {}, pd.DataFrame(columns=["model", "params", "mean_mae"])

    try:
        subtrain, validation = temporal_train_test_split(
            train_df,
            test_size=validation_size,
            embargo_hours=embargo_hours,
        )
    except ValueError:
        return {}, pd.DataFrame(columns=["model", "params", "mean_mae"])
    X_train = subtrain[features]
    X_valid = validation[features]
    rows = []
    best_params: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        grid = tuning_grid(model_name, quick=quick)
        if len(grid) <= 1:
            best_params[model_name] = grid[0] if grid else {}
            continue
        print(f"[TUNE] {model_name}: {len(grid)} combinazioni", flush=True)
        model_best_params = grid[0]
        model_best_mae = np.inf
        for params in grid:
            maes = []
            for target in targets:
                model = build_model(model_name, random_state=random_state, quick=quick, params=params)
                model.fit(X_train, subtrain[target])
                pred = np.clip(model.predict(X_valid), a_min=0, a_max=None)
                maes.append(mean_absolute_error(validation[target], pred))
            mean_mae = float(np.mean(maes))
            rows.append({"model": model_name, "params": json.dumps(params), "mean_mae": mean_mae})
            if mean_mae < model_best_mae:
                model_best_mae = mean_mae
                model_best_params = params
        best_params[model_name] = model_best_params
        print(f"[TUNE] {model_name}: best_mae={model_best_mae:.3f}, params={model_best_params}", flush=True)

    return best_params, pd.DataFrame(rows).sort_values(["model", "mean_mae"]).reset_index(drop=True)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_train: pd.Series | None = None,
    mase_seasonality: int = DEFAULT_MASE_SEASONALITY,
    peak_threshold: float | None = None,
) -> dict[str, float]:
    y_true_values = y_true.to_numpy(dtype=float)
    y_pred_values = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_target = float(np.mean(y_true))
    denom = mase_denominator(y_train, mase_seasonality) if y_train is not None else np.nan
    peak_threshold = (
        float(peak_threshold)
        if peak_threshold is not None and not np.isnan(peak_threshold)
        else float(np.quantile(y_true_values, DEFAULT_PEAK_QUANTILE))
    )
    scores = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": smape(y_true_values, y_pred_values),
        "wape_pct": wape(y_true_values, y_pred_values),
        "mase": float(mae / denom) if denom and not np.isnan(denom) else np.nan,
        "nmae_vs_mean_target": float(mae / mean_target) if mean_target else np.nan,
    }
    peak_mask = y_true_values >= peak_threshold
    scores["peak_mae"] = (
        float(mean_absolute_error(y_true_values[peak_mask], y_pred_values[peak_mask]))
        if peak_mask.any()
        else np.nan
    )
    scores.update(peak_classification_metrics(y_true_values, y_pred_values, peak_threshold))
    return scores


def is_time_series_baseline(model_name: str) -> bool:
    return model_name in TIME_SERIES_BASELINE_MODELS


def is_feature_model(model_name: str) -> bool:
    return not is_time_series_baseline(model_name)


def time_series_baseline_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observed_targets: pd.DataFrame,
    target: str,
    model_name: str,
    horizon_hours: int,
) -> np.ndarray:
    train_series = train_df.set_index("timestamp")[target].astype(float).sort_index()
    test_index = pd.DatetimeIndex(pd.to_datetime(test_df["timestamp"]))
    fallback = float(train_series.mean()) if len(train_series) else 0.0

    if model_name == "last_hour":
        observed = observed_targets.set_index("timestamp")[target].astype(float).sort_index()
        source_index = test_index - pd.to_timedelta(1, unit="h")
        pred = observed.reindex(source_index)
        pred.index = test_index
    elif model_name == "same_hour_previous_day":
        observed = observed_targets.set_index("timestamp")[target].astype(float).sort_index()
        source_index = test_index - pd.to_timedelta(24, unit="h")
        pred = observed.reindex(source_index)
        pred.index = test_index
    elif model_name == "same_hour_previous_week":
        observed = observed_targets.set_index("timestamp")[target].astype(float).sort_index()
        source_index = test_index - pd.to_timedelta(168, unit="h")
        pred = observed.reindex(source_index)
        pred.index = test_index
    elif model_name == "rolling_mean_24h":
        observed = observed_targets.set_index("timestamp")[target].astype(float).sort_index()
        pred = observed.rolling("24h", closed="left", min_periods=1).mean().reindex(test_index)
    elif model_name == "rolling_mean_168h":
        observed = observed_targets.set_index("timestamp")[target].astype(float).sort_index()
        pred = observed.rolling("168h", closed="left", min_periods=1).mean().reindex(test_index)
    elif model_name == "sarimax":
        pred = sarimax_prediction(train_series, len(test_df))
    elif model_name == "prophet":
        pred = prophet_prediction(train_df, test_df, target)
    else:
        raise ValueError(f"Baseline temporale non riconosciuta: {model_name}")

    pred = pd.Series(pred, index=test_index, dtype=float)
    pred = pred.ffill().fillna(fallback).to_numpy(dtype=float)
    return np.clip(pred, a_min=0, a_max=None)


def sarimax_prediction(train_series: pd.Series, steps: int) -> np.ndarray:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError("statsmodels non installato, impossibile usare sarimax.") from exc
    if len(train_series) < 48:
        fallback = float(train_series.mean()) if len(train_series) else 0.0
        return np.full(steps, fallback)
    model = SARIMAX(
        train_series.to_numpy(dtype=float),
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=80)
    return np.asarray(fitted.forecast(steps=steps), dtype=float)


def prophet_prediction(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("prophet non installato, impossibile usare prophet.") from exc
    frame = train_df[["timestamp", target]].rename(columns={"timestamp": "ds", target: "y"}).copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    future = test_df[["timestamp"]].rename(columns={"timestamp": "ds"}).copy()
    future["ds"] = pd.to_datetime(future["ds"])
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="additive",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame)
        forecast = model.predict(future)
    return forecast["yhat"].to_numpy(dtype=float)


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observed_targets: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_names: list[str],
    random_state: int,
    quick: bool,
    horizon_hours: int,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any], pd.DataFrame]:
    X_train = train_df[features]
    X_test = test_df[features]
    metrics_rows = []
    fitted: dict[tuple[str, str], Any] = {}
    prediction_columns: dict[str, Any] = {"timestamp": test_df["timestamp"].to_numpy()}

    for model_name in model_names:
        if is_time_series_baseline(model_name):
            print(f"[BASELINE] Valuto {model_name}", flush=True)
            base_model = None
        else:
            print(f"[MODEL] Fit {model_name}", flush=True)
            base_model = build_model(
                model_name,
                random_state=random_state,
                quick=quick,
                params=(model_params or {}).get(model_name, {}),
            )
        for target in targets:
            print(f"[MODEL]   target={target}", flush=True)
            if base_model is None:
                baseline_start = time.perf_counter()
                pred = time_series_baseline_prediction(
                    train_df=train_df,
                    test_df=test_df,
                    observed_targets=observed_targets,
                    target=target,
                    model_name=model_name,
                    horizon_hours=horizon_hours,
                )
                baseline_seconds = time.perf_counter() - baseline_start
                fit_seconds = baseline_seconds if model_name in {"sarimax", "prophet"} else 0.0
                predict_seconds = 0.0 if model_name in {"sarimax", "prophet"} else baseline_seconds
            else:
                model = clone(base_model)
                fit_start = time.perf_counter()
                model.fit(X_train, train_df[target])
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                pred = np.clip(model.predict(X_test), a_min=0, a_max=None)
                predict_seconds = time.perf_counter() - predict_start
                fitted[(model_name, target)] = model
            prediction_columns[f"{target}__{model_name}"] = pred
            peak_threshold = float(train_df[target].quantile(DEFAULT_PEAK_QUANTILE))
            scores = evaluate_predictions(
                test_df[target],
                pred,
                y_train=train_df[target],
                peak_threshold=peak_threshold,
            )
            metrics_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "model_family": "time_series_baseline" if base_model is None else "feature_model",
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "total_seconds": float(fit_seconds + predict_seconds),
                    "inference_ms_per_row": (
                        float(predict_seconds / len(test_df) * 1000) if len(test_df) else np.nan
                    ),
                    **scores,
                }
            )
    metrics = pd.DataFrame(metrics_rows).sort_values(["target", "mae"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_columns)
    return metrics, fitted, predictions


def permutation_importance_for_best_models(
    metrics: pd.DataFrame,
    fitted: dict[tuple[str, str], Any],
    test_df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    if repeats <= 0:
        return pd.DataFrame(columns=["target", "model", "feature", "importance_mean", "importance_std"])
    rows = []
    X_test = test_df[features]
    rng = np.random.default_rng(random_state)
    for target in targets:
        best_row = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best_row.empty:
            continue
        model_name = str(best_row.iloc[0]["model"])
        print(f"[IMPORTANCE] target={target}, modello={model_name}", flush=True)
        model = fitted[(model_name, target)]
        baseline_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)
        baseline_mae = mean_absolute_error(test_df[target], baseline_pred)
        feature_rows = []
        for feature in features:
            importances = []
            for _ in range(repeats):
                permuted = X_test.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                pred = np.clip(model.predict(permuted), a_min=0, a_max=None)
                shuffled_mae = mean_absolute_error(test_df[target], pred)
                importances.append(float(shuffled_mae - baseline_mae))
            feature_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature": feature,
                    "importance_mean": float(np.mean(importances)),
                    "importance_std": float(np.std(importances, ddof=0)),
                    "baseline_mae": float(baseline_mae),
                }
            )
        for row in sorted(feature_rows, key=lambda item: item["importance_mean"], reverse=True):
            rows.append(
                row
            )
    return pd.DataFrame(rows)


def rolling_month_folds(
    df: pd.DataFrame,
    min_train_periods: int,
    max_folds: int | None,
    embargo_hours: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    periods = sorted(df["timestamp"].dt.to_period("M").unique())
    folds = []
    for idx in range(min_train_periods, len(periods)):
        train_periods = periods[:idx]
        test_period = periods[idx]
        train_fold = df[df["timestamp"].dt.to_period("M").isin(train_periods)].copy()
        test_fold = df[df["timestamp"].dt.to_period("M").eq(test_period)].copy()
        if embargo_hours > 0 and not test_fold.empty:
            test_start = pd.to_datetime(test_fold["timestamp"]).min()
            train_cutoff = test_start - pd.to_timedelta(embargo_hours, unit="h")
            train_fold = train_fold[pd.to_datetime(train_fold["timestamp"]) < train_cutoff].copy()
        if len(train_fold) < 30 or len(test_fold) < 10:
            continue
        folds.append((str(test_period), train_fold, test_fold))
    if max_folds is not None:
        folds = folds[:max_folds]
    return folds


def evaluate_rolling_validation(
    df: pd.DataFrame,
    observed_targets: pd.DataFrame,
    candidate_features: list[str],
    targets: list[str],
    model_names: list[str],
    missing_threshold: float,
    collinearity_threshold: float,
    top_k: int | None,
    random_state: int,
    quick: bool,
    min_train_periods: int,
    max_folds: int | None,
    tune: bool,
    tune_validation_size: float,
    horizon_hours: int,
    embargo_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = rolling_month_folds(
        df,
        min_train_periods=min_train_periods,
        max_folds=max_folds,
        embargo_hours=embargo_hours,
    )
    metric_parts = []
    selected_parts = []
    for fold_name, train_fold, test_fold in folds:
        print(
            f"[ROLLING] Fold test={fold_name}, train={len(train_fold)}, test={len(test_fold)}",
            flush=True,
        )
        selection = select_features(
            train_df=train_fold,
            candidate_features=candidate_features,
            targets=targets,
            missing_threshold=missing_threshold,
            collinearity_threshold=collinearity_threshold,
            top_k=top_k,
            random_state=random_state,
        )
        model_params, _ = ({}, pd.DataFrame())
        if tune:
            model_params, _ = tune_model_params(
                train_df=train_fold,
                features=selection.selected_features,
                targets=targets,
                model_names=model_names,
                random_state=random_state,
                quick=quick,
                validation_size=tune_validation_size,
                embargo_hours=embargo_hours,
            )
        metrics, _, _ = evaluate_models(
            train_df=train_fold,
            test_df=test_fold,
            observed_targets=observed_targets,
            features=selection.selected_features,
            targets=targets,
            model_names=model_names,
            random_state=random_state,
            quick=quick,
            horizon_hours=horizon_hours,
            model_params=model_params,
        )
        metrics.insert(0, "fold", fold_name)
        metrics.insert(1, "train_start", train_fold["timestamp"].min())
        metrics.insert(2, "train_end", train_fold["timestamp"].max())
        metrics.insert(3, "test_start", test_fold["timestamp"].min())
        metrics.insert(4, "test_end", test_fold["timestamp"].max())
        metric_parts.append(metrics)
        selected_parts.append(
            pd.DataFrame(
                {
                    "fold": fold_name,
                    "feature": selection.selected_features,
                    "rank": np.arange(1, len(selection.selected_features) + 1),
                }
            )
        )

    if metric_parts:
        metrics = pd.concat(metric_parts, ignore_index=True)
    else:
        metrics = pd.DataFrame()
    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = pd.DataFrame(columns=["fold", "feature", "rank"])
    return metrics, selected


def summarize_rolling_metrics(rolling_metrics: pd.DataFrame) -> pd.DataFrame:
    if rolling_metrics.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "target",
                "folds",
                "fit_seconds_mean",
                "predict_seconds_mean",
                "inference_ms_per_row_mean",
                "mae_mean",
                "mae_std",
                "mae_median",
                "mae_q75",
                "mae_max",
                "rmse_mean",
                "r2_mean",
                "r2_std",
                "smape_pct_mean",
                "wape_pct_mean",
                "mase_mean",
                "peak_mae_mean",
                "peak_f1_mean",
            ]
        )
    group_cols = ["model", "target"]
    sort_cols = ["target", "mae_mean"]
    if "top_k" in rolling_metrics.columns:
        group_cols = ["top_k", *group_cols]
        sort_cols = ["target", "top_k", "mae_mean"]
    return (
        rolling_metrics.groupby(group_cols, as_index=False)
        .agg(
            folds=("fold", "nunique"),
            fit_seconds_mean=("fit_seconds", "mean"),
            predict_seconds_mean=("predict_seconds", "mean"),
            inference_ms_per_row_mean=("inference_ms_per_row", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            mae_q75=("mae", lambda value: float(value.quantile(0.75))),
            mae_max=("mae", "max"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            smape_pct_mean=("smape_pct", "mean"),
            wape_pct_mean=("wape_pct", "mean"),
            mase_mean=("mase", "mean"),
            peak_mae_mean=("peak_mae", "mean"),
            peak_f1_mean=("peak_f1", "mean"),
        )
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )


def rolling_rank_stability(rolling_metrics: pd.DataFrame) -> pd.DataFrame:
    if rolling_metrics.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "model",
                "best_fold_count",
                "best_fold_mae_mean",
                "best_fold_smape_mean",
            ]
        )
    best = (
        rolling_metrics.sort_values("mae")
        .groupby(["fold", "target"], as_index=False)
        .head(1)
        .copy()
    )
    return (
        best.groupby(["target", "model"], as_index=False)
        .agg(
            best_fold_count=("fold", "nunique"),
            best_fold_mae_mean=("mae", "mean"),
            best_fold_smape_mean=("smape_pct", "mean"),
        )
        .sort_values(["target", "best_fold_count", "best_fold_mae_mean"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def bootstrap_metric_intervals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: list[str],
    samples: int,
    random_state: int,
) -> pd.DataFrame:
    if samples <= 0 or metrics.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    rows = []
    metric_cols = ["mae", "rmse", "r2", "smape_pct", "wape_pct", "mase", "peak_mae", "peak_f1"]
    for _, row in metrics.iterrows():
        target = str(row["target"])
        model_name = str(row["model"])
        pred_col = f"{target}__{model_name}"
        if target not in targets or pred_col not in predictions:
            continue
        y_true = test_df[target].reset_index(drop=True)
        y_pred = pd.Series(predictions[pred_col].to_numpy(dtype=float))
        peak_threshold = float(train_df[target].quantile(DEFAULT_PEAK_QUANTILE))
        boot_rows = []
        for _ in range(samples):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            scores = evaluate_predictions(
                y_true.iloc[idx].reset_index(drop=True),
                y_pred.iloc[idx].to_numpy(dtype=float),
                y_train=train_df[target],
                peak_threshold=peak_threshold,
            )
            boot_rows.append(scores)
        boot = pd.DataFrame(boot_rows)
        for metric in metric_cols:
            if metric not in boot:
                continue
            values = boot[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "metric": metric,
                    "estimate": float(row.get(metric, np.nan)),
                    "bootstrap_mean": float(values.mean()),
                    "ci95_low": float(values.quantile(0.025)),
                    "ci95_high": float(values.quantile(0.975)),
                    "bootstrap_samples": samples,
                }
            )
    return pd.DataFrame(rows)


def summarize_top_k_metrics(top_k_metrics: pd.DataFrame) -> pd.DataFrame:
    if top_k_metrics.empty:
        return pd.DataFrame()
    return (
        top_k_metrics.groupby(["top_k", "model", "target"], as_index=False)
        .agg(
            fit_seconds_mean=("fit_seconds", "mean"),
            predict_seconds_mean=("predict_seconds", "mean"),
            inference_ms_per_row_mean=("inference_ms_per_row", "mean"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            smape_pct_mean=("smape_pct", "mean"),
            wape_pct_mean=("wape_pct", "mean"),
            mase_mean=("mase", "mean"),
            peak_mae_mean=("peak_mae", "mean"),
            peak_f1_mean=("peak_f1", "mean"),
        )
        .sort_values(["target", "top_k", "mae_mean"])
        .reset_index(drop=True)
    )


def best_top_k_rows(
    top_k_metrics: pd.DataFrame,
    top_k_rolling_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not top_k_metrics.empty:
        best_final = (
            top_k_metrics.sort_values("mae")
            .groupby(["top_k", "target"], as_index=False)
            .head(1)
            .copy()
        )
        best_final["validation_scope"] = "final_split"
        best_final = best_final.rename(columns={"mae": "mae_mean", "rmse": "rmse_mean", "r2": "r2_mean"})
        rows.append(
            best_final[
                ["validation_scope", "top_k", "target", "model", "mae_mean", "rmse_mean", "r2_mean", "smape_pct"]
            ].rename(columns={"smape_pct": "smape_pct_mean"})
        )
    if not top_k_rolling_summary.empty:
        best_rolling = (
            top_k_rolling_summary.sort_values("mae_mean")
            .groupby(["top_k", "target"], as_index=False)
            .head(1)
            .copy()
        )
        best_rolling["validation_scope"] = "rolling"
        rows.append(
            best_rolling[
                [
                    "validation_scope",
                    "top_k",
                    "target",
                    "model",
                    "mae_mean",
                    "rmse_mean",
                    "r2_mean",
                    "smape_pct_mean",
                ]
            ]
        )
    if rows:
        return pd.concat(rows, ignore_index=True).sort_values(
            ["validation_scope", "target", "mae_mean"]
        ).reset_index(drop=True)
    return pd.DataFrame(
        columns=[
            "validation_scope",
            "top_k",
            "target",
            "model",
            "mae_mean",
            "rmse_mean",
            "r2_mean",
            "smape_pct_mean",
        ]
    )


def select_top_k_automatically(
    top_k_metrics: pd.DataFrame,
    top_k_rolling_summary: pd.DataFrame,
    allow_final_split: bool = False,
) -> tuple[int | None, str, pd.DataFrame]:
    if not top_k_rolling_summary.empty:
        scope = "rolling"
        best_per_target = (
            top_k_rolling_summary.sort_values("mae_mean")
            .groupby(["top_k", "target"], as_index=False)
            .head(1)
            .copy()
        )
    elif not allow_final_split:
        return (
            None,
            "manual_no_rolling",
            pd.DataFrame(
                columns=[
                    "validation_scope",
                    "top_k",
                    "targets",
                    "mean_best_mae",
                    "max_best_mae",
                    "mean_best_rmse",
                    "mean_best_r2",
                    "mean_best_smape_pct",
                    "max_best_smape_pct",
                    "selected",
                ]
            ),
        )
    elif not top_k_metrics.empty:
        scope = "final_split"
        best_per_target = (
            top_k_metrics.sort_values("mae")
            .groupby(["top_k", "target"], as_index=False)
            .head(1)
            .rename(
                columns={
                    "mae": "mae_mean",
                    "rmse": "rmse_mean",
                    "r2": "r2_mean",
                    "smape_pct": "smape_pct_mean",
                }
            )
            .copy()
        )
    else:
        raise ValueError("Nessuna metrica disponibile per selezionare automaticamente top-k.")

    if "smape_pct_mean" not in best_per_target:
        best_per_target["smape_pct_mean"] = np.nan

    selection = (
        best_per_target.groupby("top_k", as_index=False)
        .agg(
            validation_scope=("top_k", lambda _: scope),
            targets=("target", "nunique"),
            mean_best_mae=("mae_mean", "mean"),
            max_best_mae=("mae_mean", "max"),
            mean_best_rmse=("rmse_mean", "mean"),
            mean_best_r2=("r2_mean", "mean"),
            mean_best_smape_pct=("smape_pct_mean", "mean"),
            max_best_smape_pct=("smape_pct_mean", "max"),
        )
    )
    max_targets = selection["targets"].max()
    selection = selection[selection["targets"].eq(max_targets)].copy()
    selection = selection.sort_values(["mean_best_mae", "max_best_mae", "top_k"]).reset_index(drop=True)
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    return int(selection.loc[0, "top_k"]), scope, selection


def best_model_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.sort_values(["target", "mae"])
        .groupby("target", as_index=False)
        .head(1)
        .sort_values("target")
        .reset_index(drop=True)
    )


def dataframe_records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, np.integer):
                clean_row[key] = int(value)
            elif isinstance(value, np.floating):
                clean_row[key] = float(value)
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records


def evaluate_top_k_grid(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    observed_targets: pd.DataFrame,
    candidate_features: list[str],
    targets: list[str],
    top_k_values: list[int],
    model_names: list[str],
    missing_threshold: float,
    collinearity_threshold: float,
    random_state: int,
    quick: bool,
    tune: bool,
    tune_validation_size: float,
    rolling_validation: bool,
    rolling_min_train_periods: int,
    rolling_max_folds: int | None,
    horizon_hours: int,
    embargo_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts = []
    rolling_parts = []
    selected_parts = []
    tuning_parts = []

    for top_k in top_k_values:
        print(f"[TOPK] Valido top_k={top_k}", flush=True)
        selection = select_features(
            train_df=train_df,
            candidate_features=candidate_features,
            targets=targets,
            missing_threshold=missing_threshold,
            collinearity_threshold=collinearity_threshold,
            top_k=top_k,
            random_state=random_state,
        )
        selected_parts.append(
            pd.DataFrame(
                {
                    "validation_scope": "final_split",
                    "top_k": top_k,
                    "feature": selection.selected_features,
                    "rank": np.arange(1, len(selection.selected_features) + 1),
                }
            )
        )

        if tune:
            model_params, tuning_results = tune_model_params(
                train_df=train_df,
                features=selection.selected_features,
                targets=targets,
                model_names=model_names,
                random_state=random_state,
                quick=quick,
                validation_size=tune_validation_size,
                embargo_hours=embargo_hours,
            )
            if not tuning_results.empty:
                tuning_results = tuning_results.copy()
                tuning_results.insert(0, "top_k", top_k)
                tuning_parts.append(tuning_results)
        else:
            model_params = {}

        metrics, _, _ = evaluate_models(
            train_df=train_df,
            test_df=test_df,
            observed_targets=observed_targets,
            features=selection.selected_features,
            targets=targets,
            model_names=model_names,
            random_state=random_state,
            quick=quick,
            horizon_hours=horizon_hours,
            model_params=model_params,
        )
        metrics.insert(0, "top_k", top_k)
        metrics.insert(1, "validation_scope", "final_split")
        metric_parts.append(metrics)

        if rolling_validation:
            rolling_metrics, rolling_selected = evaluate_rolling_validation(
                df=df,
                observed_targets=observed_targets,
                candidate_features=candidate_features,
                targets=targets,
                model_names=model_names,
                missing_threshold=missing_threshold,
                collinearity_threshold=collinearity_threshold,
                top_k=top_k,
                random_state=random_state,
                quick=quick,
                min_train_periods=rolling_min_train_periods,
                max_folds=rolling_max_folds,
                tune=tune,
                tune_validation_size=tune_validation_size,
                horizon_hours=horizon_hours,
                embargo_hours=embargo_hours,
            )
            if not rolling_metrics.empty:
                rolling_metrics = rolling_metrics.copy()
                rolling_metrics.insert(0, "top_k", top_k)
                rolling_parts.append(rolling_metrics)
            if not rolling_selected.empty:
                rolling_selected = rolling_selected.copy()
                rolling_selected.insert(0, "validation_scope", "rolling")
                rolling_selected.insert(1, "top_k", top_k)
                selected_parts.append(rolling_selected)

    top_k_metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    top_k_rolling_metrics = (
        pd.concat(rolling_parts, ignore_index=True) if rolling_parts else pd.DataFrame()
    )
    top_k_selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    top_k_tuning = pd.concat(tuning_parts, ignore_index=True) if tuning_parts else pd.DataFrame()
    return top_k_metrics, top_k_rolling_metrics, top_k_selected, top_k_tuning


def write_top_k_validation_report(
    output_dir: Path,
    top_k_values: list[int],
    top_k_metrics: pd.DataFrame,
    top_k_summary: pd.DataFrame,
    top_k_rolling_summary: pd.DataFrame,
    top_k_best: pd.DataFrame,
    top_k_selected: pd.DataFrame,
    top_k_auto_selection: pd.DataFrame,
    auto_selected_top_k: int | None,
    auto_selection_scope: str | None,
) -> None:
    feature_counts = (
        top_k_selected.groupby(["validation_scope", "top_k"], as_index=False)
        .agg(selected_features=("feature", "nunique"))
        .sort_values(["validation_scope", "top_k"])
        if not top_k_selected.empty
        else pd.DataFrame()
    )
    report = f"""# Validazione numero feature

## Griglia

- Top-k testati: {", ".join(map(str, top_k_values))}

## Migliori configurazioni

{markdown_table(top_k_best, max_rows=80)}

## Scelta automatica

- Top-k scelto: {auto_selected_top_k if auto_selected_top_k is not None else "non disponibile"}
- Criterio usato: {auto_selection_scope if auto_selection_scope else "non disponibile"}

{markdown_table(top_k_auto_selection, max_rows=80)}

## Sintesi final split

{markdown_table(top_k_summary, max_rows=120)}

## Sintesi rolling validation

{markdown_table(top_k_rolling_summary, max_rows=120)}

## Feature selezionate per top-k

{markdown_table(feature_counts, max_rows=80)}

## Output

- `top_k_validation_metrics.csv`
- `top_k_validation_summary.csv`
- `top_k_validation_rolling_metrics.csv`
- `top_k_validation_rolling_summary.csv`
- `top_k_validation_best.csv`
- `top_k_auto_selection.csv`
- `top_k_validation_selected_features.csv`
- `top_k_validation_tuning_results.csv`
"""
    (output_dir / "top_k_validation_report.md").write_text(report, encoding="utf-8")


def save_prediction_plots(
    output_dir: Path,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    rows = []
    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        pred_col = f"{target}__{model_name}"
        if pred_col not in predictions:
            continue
        plot_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(predictions["timestamp"]),
                "observed": test_df[target].to_numpy(),
                "predicted": predictions[pred_col].to_numpy(),
            }
        )
        plot_df["residual"] = plot_df["predicted"] - plot_df["observed"]
        plot_df["hour"] = plot_df["timestamp"].dt.hour
        by_hour = (
            plot_df.groupby("hour", as_index=False)
            .agg(
                residual_mean=("residual", "mean"),
                residual_mae=("residual", lambda value: float(np.mean(np.abs(value)))),
                n=("residual", "size"),
            )
            .assign(target=target, model=model_name)
        )
        rows.append(by_hour)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        axes[0, 0].plot(plot_df["timestamp"], plot_df["observed"], label="osservato", linewidth=1.5)
        axes[0, 0].plot(plot_df["timestamp"], plot_df["predicted"], label="predetto", linewidth=1.2)
        axes[0, 0].set_title(f"{target}: osservato vs predetto ({model_name})")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=35)

        axes[0, 1].scatter(plot_df["observed"], plot_df["predicted"], alpha=0.6, s=18)
        max_value = max(plot_df["observed"].max(), plot_df["predicted"].max())
        axes[0, 1].plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
        axes[0, 1].set_title("Predetto vs osservato")
        axes[0, 1].set_xlabel("osservato")
        axes[0, 1].set_ylabel("predetto")

        axes[1, 0].plot(plot_df["timestamp"], plot_df["residual"], linewidth=1.0)
        axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_title("Residui nel tempo")
        axes[1, 0].tick_params(axis="x", rotation=35)

        axes[1, 1].bar(by_hour["hour"], by_hour["residual_mae"])
        axes[1, 1].set_title("MAE residui per ora")
        axes[1, 1].set_xlabel("ora")
        axes[1, 1].set_ylabel("MAE")

        save_figure(fig, output_dir / f"plot_{target}_{model_name}.png", dpi=300)
        plt.close(fig)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["hour", "residual_mean", "residual_mae", "n", "target", "model"])


def save_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    width_per_col: float = 0.45,
    height_per_row: float = 0.45,
) -> None:
    if matrix.empty:
        return

    rows, cols = matrix.shape
    fig_width = min(max(7.0, cols * width_per_col), 24.0)
    fig_height = min(max(5.0, rows * height_per_row), 24.0)
    label_size = 8 if max(rows, cols) <= 35 else 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=label_size)
    ax.set_yticklabels(matrix.index, fontsize=label_size)
    fig.colorbar(image, ax=ax, label="correlazione Pearson")
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)


def save_selected_feature_correlations(
    output_dir: Path,
    train_df: pd.DataFrame,
    selected_features: list[str],
    targets: list[str],
) -> dict[str, str]:
    numeric_features = [
        feature
        for feature in selected_features
        if feature in train_df.columns and pd.api.types.is_numeric_dtype(train_df[feature])
    ]
    outputs: dict[str, str] = {}
    if numeric_features:
        feature_matrix = train_df[numeric_features].corr(method="pearson", numeric_only=True)
        feature_csv = output_dir / "selected_feature_correlation_matrix.csv"
        feature_png = output_dir / "selected_feature_correlation_matrix.png"
        feature_matrix.to_csv(feature_csv)
        save_correlation_heatmap(
            feature_matrix,
            feature_png,
            "Correlazione Pearson - feature selezionate",
        )
        outputs["selected_feature_correlation_matrix_csv"] = str(feature_csv)
        outputs["selected_feature_correlation_matrix_png"] = str(feature_png)
        outputs["selected_feature_correlation_matrix_pdf"] = str(feature_png.with_suffix(".pdf"))

    target_cols = [target for target in targets if target in train_df.columns]
    if numeric_features and target_cols:
        target_feature_matrix = train_df[[*target_cols, *numeric_features]].corr(
            method="pearson",
            numeric_only=True,
        ).loc[target_cols, numeric_features]
        target_csv = output_dir / "selected_feature_target_correlations.csv"
        target_png = output_dir / "selected_feature_target_correlations.png"
        target_feature_matrix.to_csv(target_csv)
        save_correlation_heatmap(
            target_feature_matrix,
            target_png,
            "Correlazione Pearson - target vs feature selezionate",
            width_per_col=0.5,
            height_per_row=0.9,
        )
        outputs["selected_feature_target_correlations_csv"] = str(target_csv)
        outputs["selected_feature_target_correlations_png"] = str(target_png)
        outputs["selected_feature_target_correlations_pdf"] = str(target_png.with_suffix(".pdf"))

    return outputs


def save_final_models(
    output_dir: Path,
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    metrics: pd.DataFrame,
    random_state: int,
    quick: bool,
    model_params: dict[str, dict[str, Any]],
) -> dict[str, str]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    X = df[features]
    saved = {}
    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        model = build_model(
            model_name,
            random_state=random_state,
            quick=quick,
            params=model_params.get(model_name, {}),
        )
        model.fit(X, df[target])
        path = model_dir / f"{target}__{model_name}.joblib"
        dump(model, path)
        saved[target] = str(path)
    return saved


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_Nessuna riga._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Mostrate {max_rows} di {len(df)} righe._")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    input_csv: Path,
    target_set: TargetSet,
    feature_scope: str,
    effective_feature_scope: str,
    mode: str,
    horizon_hours: int,
    embargo_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    top_k: int | None,
    top_k_auto_selection_scope: str | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selection: FeatureSelectionResult,
    metrics: pd.DataFrame,
    intrinsic_importance: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
    single_feature_ablation: pd.DataFrame,
    shap_importance: pd.DataFrame,
    explanatory_figures: dict[str, str],
    tuning_results: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    rolling_rank: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    residuals_by_hour: pd.DataFrame,
    best_metrics: pd.DataFrame,
    best_feature_metrics: pd.DataFrame,
    saved_models: dict[str, str],
    selected_correlation_outputs: dict[str, str],
    model_names: list[str],
) -> None:
    selected_preview = pd.DataFrame({"feature": selection.selected_features})
    selected_correlation_table = pd.DataFrame(
        [
            {"file": value, "contenuto": key}
            for key, value in selected_correlation_outputs.items()
        ]
    )
    explanatory_figures_table = pd.DataFrame(
        [{"file": value, "contenuto": key} for key, value in explanatory_figures.items()]
    )
    report = f"""# Modellazione predittiva Dozza

## Input

- Dataset: `{input_csv}`
- Analisi target: `{target_set.name}` - {target_set.title}
- Target: {", ".join(target_set.targets)}
- Feature scope richiesto: `{feature_scope}`
- Feature scope effettivo: `{effective_feature_scope}`
- Modalita': `{mode}`
- Orizzonte predittivo: {horizon_hours} ore
- Embargo temporale train/test: {embargo_hours} ore
- Lag usati: {", ".join(map(str, lags)) if lags else "nessuno"}
- Rolling windows usate: {", ".join(map(str, rolling_windows)) if rolling_windows else "nessuna"}
- Top-k feature usato: {top_k if top_k is not None else "tutte"}
- Fonte scelta top-k: {top_k_auto_selection_scope if top_k_auto_selection_scope else "manuale/default"}
- Modelli valutati: {", ".join(model_names)}

## Split temporale

- Train: {len(train_df)} righe, {train_df["timestamp"].min()} - {train_df["timestamp"].max()}
- Test: {len(test_df)} righe, {test_df["timestamp"].min()} - {test_df["timestamp"].max()}
- Gap applicato: {embargo_hours} ore tra ultimo origin del train e primo origin del test.

## Selezione feature

- Feature candidate dopo esclusione leakage: {len(selection.filtered_features) + len(selection.dropped)}
- Feature rimaste dopo filtri e collinearita': {len(selection.filtered_features)}
- Feature selezionate per i modelli: {len(selection.selected_features)}

### Feature selezionate

{markdown_table(selected_preview, max_rows=80)}

### Ranking feature

{markdown_table(selection.ranking, max_rows=40)}

### Feature scartate

{markdown_table(selection.dropped, max_rows=40)}

### Correlazioni feature selezionate

Le matrici sono calcolate sul train split, dopo la selezione feature.

{markdown_table(selected_correlation_table, max_rows=20)}

## Performance

Le metriche sono sul test temporale finale, quindi sulle date successive al train.
Le baseline naive e rolling sono calcolate sulla serie osservata all'origine
della previsione, non sulla serie target gia' traslata all'orizzonte futuro.
Per coerenza con i lag target dei modelli, `last_hour` usa il valore osservato
a `t-1` per prevedere `t+h`; le rolling terminano prima di `t`.

{markdown_table(metrics, max_rows=80)}

### Migliori modelli per target

Questa tabella e' salvata anche in `best_model_metrics.csv` e include `smape_pct`.

{markdown_table(best_metrics, max_rows=20)}

### Tempi modello

`fit_seconds` misura il tempo di addestramento per target/modello.
`predict_seconds` misura il tempo di inferenza sul test split.
`inference_ms_per_row` normalizza l'inferenza rispetto al numero di righe test.

### Figure esplicative

{markdown_table(explanatory_figures_table, max_rows=80)}

## Tuning leggero

{markdown_table(tuning_results, max_rows=60)}

## Validazione temporale rolling

La rolling validation usa fold mensili espandendo il train e testando sul mese successivo.
Lo stesso embargo temporale del final split viene applicato tra train fold e test fold.

### Sintesi rolling

{markdown_table(rolling_summary, max_rows=80)}

### Stabilita' ranking rolling

Conta in quanti fold mensili ogni modello risulta il migliore per target.

{markdown_table(rolling_rank, max_rows=80)}

### Fold rolling

{markdown_table(rolling_metrics, max_rows=120)}

## Intervalli Bootstrap

Intervalli empirici al 95% sul test temporale finale, calcolati ricampionando
le ore del test split. Servono per leggere l'incertezza delle metriche senza
assumere normalita' degli errori.

{markdown_table(bootstrap_intervals, max_rows=120)}

## Importanza Feature

### Importanza feature intrinseca

Importanza nativa del miglior modello per ciascun target quando disponibile
(`feature_importances_` per alberi/boosting, coefficienti assoluti per modelli
lineari).

{markdown_table(intrinsic_importance, max_rows=80)}

### Importanza permutation

Permutation importance calcolata sul miglior modello per ciascun target usando MAE come metrica.

{markdown_table(permutation, max_rows=80)}

## Studio Ablativo

Include due viste: leave-one-group-out e scenari cumulativi paper-ready
(`calendar`, `calendar+meteo`, `calendar+tim`, `calendar+pedoni`, full model).
`delta_mae > 0` indica che la variante peggiora rispetto al full model.

{markdown_table(ablation, max_rows=100)}

### Ablazione singola feature

Leave-one-feature-out calcolata solo sulle feature selezionate finali e sul
miglior modello feature-based per ciascun target. Non ritocca gli iperparametri:
misura il contributo marginale della singola feature dentro il modello gia'
selezionato.

{markdown_table(single_feature_ablation, max_rows=120)}

## SHAP

SHAP viene calcolato solo se `--shap-samples` e' maggiore di zero. In locale
conviene usare pochi campioni; sul cluster si puo' aumentare il numero.

{markdown_table(shap_importance, max_rows=100)}

## Residui

### Residui per ora

{markdown_table(residuals_by_hour, max_rows=80)}

Grafici salvati:

- `plot_<target>_<modello>.png/.pdf`
- `model_feature_importance_<target>.png/.pdf`
- `permutation_importance_<target>.png/.pdf`
- `shap_importance_<target>.png/.pdf`, se SHAP e' attivo
- `target_timeseries_train_test.png/.pdf`
- `target_distribution_train_test.png/.pdf`
- `model_metric_mae.png/.pdf`, `model_metric_r2.png/.pdf`, `model_metric_smape_pct.png/.pdf`
- `model_metric_wape_pct.png/.pdf`, `model_metric_mase.png/.pdf`, `model_metric_peak_f1.png/.pdf`
- `model_metric_fit_seconds.png/.pdf`, `model_metric_inference_ms_per_row.png/.pdf`
- `selected_feature_groups.png/.pdf`
- `ablation_delta_mae.png/.pdf`, `ablation_group_set_mae.png/.pdf`
- `ablation_single_feature_delta_mae.png/.pdf`

## Artefatti modello

I modelli salvati sono scelti tra i modelli basati su feature. Le baseline
temporali restano nel confronto metriche, ma non generano artefatti modello.

### Migliori modelli feature-based

{markdown_table(best_feature_metrics, max_rows=20)}

{markdown_table(pd.DataFrame([{"target": key, "model_path": value} for key, value in saved_models.items()]), max_rows=20)}

## Modelli consigliati

- Baseline `dummy_mean`: serve solo per controllare che i modelli veri battano una media storica.
- Baseline temporali `last_hour`, `same_hour_previous_day`, `same_hour_previous_week`,
  `rolling_mean_24h`, `rolling_mean_168h`: sono confronti obbligatori per un paper.
- `sarimax` / `prophet`: baseline time-series piu' pesanti e opzionali; usarle se
  l'ambiente cluster ha le dipendenze e se il budget tempo e' sufficiente.
- `ridge` / `elasticnet`: utili come baseline interpretabili e stabili, soprattutto con feature molto correlate.
- `log1p_ridge`: variante robusta per conteggi positivi e code lunghe.
- `poisson` / `tweedie`: modelli lineari per conteggi non negativi; utili se vuoi previsioni conservative e interpretabili.
- `two_stage_ridge`: modello a due stadi per separare ore ordinarie e ore ad alta affluenza.
- `hist_gradient_boosting`: buon candidato principale per dataset tabellari piccoli/medi.
- `random_forest`: robusto, ma spesso meno preciso dei boosting e meno adatto a extrapolare.
- `extra_trees`: simile a random forest ma piu' randomizzato; utile come confronto robusto e spesso veloce.
- `xgboost` / `lightgbm`: da provare nella run completa; spesso migliori su relazioni non lineari, ma richiedono tuning.

Nota operativa: con `mode=nowcast` lo script usa TIM dello stesso orario del target. Questo e' corretto se TIM e' disponibile in tempo reale o quasi reale. Per previsione anticipata usare `mode=forecast`, `--horizon-hours` e lag/meteo disponibili prima dell'ora da prevedere.
"""
    (output_dir / "modeling_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection e modelli predittivi Dozza")
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/dozza_analysis/dozza_joined_hourly_inner.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dozza_modeling"))
    parser.add_argument(
        "--target-set",
        choices=target_set_choices(),
        default="flow",
        help=(
            "Analisi target: flow=ingressi/uscite, nationality=italiani/stranieri, "
            "age=fasce eta'. Alias accettati: dual, six."
        ),
    )
    parser.add_argument(
        "--feature-scope",
        choices=FEATURE_SCOPE_CHOICES,
        default="auto",
        help=(
            "Gruppi feature ammessi. auto usa TIM+meteo per flow e pedoni+meteo "
            "per target TIM, evitando leakage dalla stessa famiglia TIM."
        ),
    )
    parser.add_argument("--mode", choices=["nowcast", "forecast"], default="nowcast")
    parser.add_argument("--horizon-hours", type=int, default=0)
    parser.add_argument("--lags", type=parse_lags, default=parse_lags(""))
    parser.add_argument("--rolling-windows", type=positive_int_list, default=positive_int_list("3,6,24"))
    parser.add_argument("--include-target-lags", action="store_true")
    parser.add_argument("--include-target-rolling", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--embargo-hours",
        type=int,
        help=(
            "Gap temporale tra train e test. Default: uguale a --horizon-hours in forecast, "
            "0 in nowcast."
        ),
    )
    parser.add_argument("--missing-threshold", type=float, default=0.2)
    parser.add_argument("--collinearity-threshold", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--top-k-grid",
        type=positive_int_list,
        help="Valori top-k da validare, es. 15,20,30,40,60.",
    )
    parser.add_argument(
        "--top-k-validation-only",
        action="store_true",
        help="Esegue solo la validazione top-k e non allena/salva il modello finale.",
    )
    parser.add_argument(
        "--allow-final-split-top-k-selection",
        action="store_true",
        help=(
            "Permette di scegliere automaticamente top-k usando il test finale se manca "
            "la rolling validation. Sconsigliato per metriche paper-ready perche' introduce "
            "selection leakage sul final split."
        ),
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=parse_models(
            "dummy_mean,dummy_median,last_hour,same_hour_previous_day,"
            "same_hour_previous_week,rolling_mean_24h,rolling_mean_168h,"
            "ridge,log1p_ridge,poisson,tweedie,random_forest,extra_trees,"
            "hist_gradient_boosting,xgboost,lightgbm,two_stage_ridge"
        ),
    )
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Disattiva lo studio ablativo per gruppi feature.",
    )
    parser.add_argument(
        "--no-single-feature-ablation",
        action="store_true",
        help="Disattiva la leave-one-feature-out ablation sulle feature selezionate.",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=0,
        help="Numero righe test per SHAP. 0 disattiva SHAP; usare valori piccoli in locale.",
    )
    parser.add_argument("--shap-background-size", type=int, default=80)
    parser.add_argument(
        "--shap-max-evals",
        type=int,
        help="Budget max_evals per SHAP permutation. Default: max(2*n_feature+1, 50).",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-validation-size", type=float, default=0.2)
    parser.add_argument("--rolling-validation", action="store_true")
    parser.add_argument("--rolling-min-train-periods", type=int, default=1)
    parser.add_argument("--rolling-max-folds", type=int)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Numero ricampionamenti bootstrap per intervalli sulle metriche finali. 0 disattiva.",
    )
    parser.add_argument("--no-save-final-models", action="store_true")
    parser.add_argument(
        "--no-clean-output-dir",
        action="store_true",
        help="Non rimuove gli artefatti generati da run precedenti nella output-dir.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean_output_dir:
        removed_outputs = clean_generated_output_dir(args.output_dir)
        if removed_outputs:
            print(
                f"[CLEAN] Rimossi {len(removed_outputs)} artefatti precedenti da {args.output_dir}",
                flush=True,
            )
    target_set = resolve_target_set(args.target_set)
    targets = target_set.targets
    effective_feature_scope = resolved_feature_scope(target_set, args.feature_scope)
    allow_target_history = args.include_target_lags or args.include_target_rolling
    embargo_hours = (
        args.embargo_hours
        if args.embargo_hours is not None
        else (args.horizon_hours if args.mode == "forecast" else 0)
    )
    if embargo_hours < 0:
        raise ValueError("--embargo-hours deve essere >= 0.")

    print(f"[DATA] Leggo {args.input_csv}", flush=True)
    modeling_dataset = load_modeling_dataset(
        input_csv=args.input_csv,
        targets=targets,
        horizon_hours=args.horizon_hours,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
        include_target_lags=args.include_target_lags,
        include_target_rolling=args.include_target_rolling,
        max_rows=args.max_rows,
    )
    df = modeling_dataset.frame
    observed_targets = modeling_dataset.observed_targets
    present_targets = [target for target in targets if target in df.columns]
    if len(present_targets) != len(targets):
        raise ValueError(f"Target mancanti nel dataset: {set(targets) - set(present_targets)}")
    targets = present_targets
    if len(df) < 30:
        raise ValueError("Dataset troppo piccolo per split temporale e valutazione.")

    candidate_features = candidate_feature_columns(
        df,
        mode=args.mode,
        targets=targets,
        target_set=target_set,
        feature_scope=args.feature_scope,
        allow_target_history=allow_target_history,
    )
    train_df, test_df = temporal_train_test_split(
        df,
        test_size=args.test_size,
        embargo_hours=embargo_hours,
    )
    print(
        f"[DATA] Righe={len(df)}, train={len(train_df)}, test={len(test_df)}, "
        f"embargo={embargo_hours}h, "
        f"target_set={target_set.name}, feature_scope={effective_feature_scope}, "
        f"candidate_features={len(candidate_features)}",
        flush=True,
    )
    top_k_auto_selection = pd.DataFrame()
    auto_selected_top_k = None
    auto_selection_scope = None
    if args.top_k_validation_only and not args.top_k_grid:
        raise ValueError("--top-k-validation-only richiede --top-k-grid.")
    if args.top_k_grid:
        top_k_metrics, top_k_rolling_metrics, top_k_selected, top_k_tuning = evaluate_top_k_grid(
            df=df,
            train_df=train_df,
            test_df=test_df,
            observed_targets=observed_targets,
            candidate_features=candidate_features,
            targets=targets,
            top_k_values=args.top_k_grid,
            model_names=args.models,
            missing_threshold=args.missing_threshold,
            collinearity_threshold=args.collinearity_threshold,
            random_state=args.random_state,
            quick=args.quick,
            tune=args.tune,
            tune_validation_size=args.tune_validation_size,
            rolling_validation=args.rolling_validation,
            rolling_min_train_periods=args.rolling_min_train_periods,
            rolling_max_folds=args.rolling_max_folds,
            horizon_hours=args.horizon_hours,
            embargo_hours=embargo_hours,
        )
        top_k_summary = summarize_top_k_metrics(top_k_metrics)
        top_k_rolling_summary = summarize_rolling_metrics(top_k_rolling_metrics)
        if not top_k_rolling_summary.empty:
            top_k_rolling_summary = top_k_rolling_summary.sort_values(
                ["target", "top_k", "mae_mean"]
            ).reset_index(drop=True)
        top_k_best = best_top_k_rows(top_k_metrics, top_k_rolling_summary)
        auto_selected_top_k, auto_selection_scope, top_k_auto_selection = select_top_k_automatically(
            top_k_metrics=top_k_metrics,
            top_k_rolling_summary=top_k_rolling_summary,
            allow_final_split=args.allow_final_split_top_k_selection,
        )

        top_k_metrics.to_csv(args.output_dir / "top_k_validation_metrics.csv", index=False)
        top_k_summary.to_csv(args.output_dir / "top_k_validation_summary.csv", index=False)
        top_k_rolling_metrics.to_csv(
            args.output_dir / "top_k_validation_rolling_metrics.csv",
            index=False,
        )
        top_k_rolling_summary.to_csv(
            args.output_dir / "top_k_validation_rolling_summary.csv",
            index=False,
        )
        top_k_best.to_csv(args.output_dir / "top_k_validation_best.csv", index=False)
        top_k_selected.to_csv(
            args.output_dir / "top_k_validation_selected_features.csv",
            index=False,
        )
        top_k_tuning.to_csv(
            args.output_dir / "top_k_validation_tuning_results.csv",
            index=False,
        )
        top_k_auto_selection.to_csv(args.output_dir / "top_k_auto_selection.csv", index=False)
        write_top_k_validation_report(
            output_dir=args.output_dir,
            top_k_values=args.top_k_grid,
            top_k_metrics=top_k_metrics,
            top_k_summary=top_k_summary,
            top_k_rolling_summary=top_k_rolling_summary,
            top_k_best=top_k_best,
            top_k_selected=top_k_selected,
            top_k_auto_selection=top_k_auto_selection,
            auto_selected_top_k=auto_selected_top_k,
            auto_selection_scope=auto_selection_scope,
        )
        print(f"[TOPK] Report scritto in {args.output_dir / 'top_k_validation_report.md'}", flush=True)
        if args.top_k_validation_only:
            print("[OK] Validazione top-k completata.", flush=True)
            return
        if auto_selected_top_k is not None:
            args.top_k = auto_selected_top_k
            print(
                f"[TOPK] Uso automaticamente top_k={args.top_k} "
                f"(criterio={auto_selection_scope}) per la run finale.",
                flush=True,
            )
        else:
            print(
                f"[TOPK] Nessuna scelta automatica da rolling validation: uso --top-k={args.top_k}. "
                "Per scegliere da final split usare --allow-final-split-top-k-selection.",
                flush=True,
            )

    selection = select_features(
        train_df=train_df,
        candidate_features=candidate_features,
        targets=targets,
        missing_threshold=args.missing_threshold,
        collinearity_threshold=args.collinearity_threshold,
        top_k=args.top_k,
        random_state=args.random_state,
    )
    selected_correlation_outputs = save_selected_feature_correlations(
        output_dir=args.output_dir,
        train_df=train_df,
        selected_features=selection.selected_features,
        targets=targets,
    )
    if args.tune:
        model_params, tuning_results = tune_model_params(
            train_df=train_df,
            features=selection.selected_features,
            targets=targets,
            model_names=args.models,
            random_state=args.random_state,
            quick=args.quick,
            validation_size=args.tune_validation_size,
            embargo_hours=embargo_hours,
        )
    else:
        model_params = {}
        tuning_results = pd.DataFrame(columns=["model", "params", "mean_mae"])
    metrics, fitted, predictions = evaluate_models(
        train_df=train_df,
        test_df=test_df,
        observed_targets=observed_targets,
        features=selection.selected_features,
        targets=targets,
        model_names=args.models,
        random_state=args.random_state,
        quick=args.quick,
        horizon_hours=args.horizon_hours,
        model_params=model_params,
    )
    feature_model_names = sorted({model_name for model_name, _target in fitted})
    feature_model_metrics = metrics[metrics["model"].isin(feature_model_names)].copy()
    if args.rolling_validation:
        rolling_metrics, rolling_selected = evaluate_rolling_validation(
            df=df,
            observed_targets=observed_targets,
            candidate_features=candidate_features,
            targets=targets,
            model_names=args.models,
            missing_threshold=args.missing_threshold,
            collinearity_threshold=args.collinearity_threshold,
            top_k=args.top_k,
            random_state=args.random_state,
            quick=args.quick,
            min_train_periods=args.rolling_min_train_periods,
            max_folds=args.rolling_max_folds,
            tune=args.tune,
            tune_validation_size=args.tune_validation_size,
            horizon_hours=args.horizon_hours,
            embargo_hours=embargo_hours,
        )
    else:
        rolling_metrics = pd.DataFrame()
        rolling_selected = pd.DataFrame(columns=["fold", "feature", "rank"])
    rolling_summary = summarize_rolling_metrics(rolling_metrics)
    rolling_rank = rolling_rank_stability(rolling_metrics)
    bootstrap_intervals = bootstrap_metric_intervals(
        train_df=train_df,
        test_df=test_df,
        predictions=predictions,
        metrics=metrics,
        targets=targets,
        samples=args.bootstrap_samples,
        random_state=args.random_state,
    )
    permutation = permutation_importance_for_best_models(
        metrics=feature_model_metrics,
        fitted=fitted,
        test_df=test_df,
        features=selection.selected_features,
        targets=targets,
        repeats=args.permutation_repeats,
        random_state=args.random_state,
    )
    intrinsic_importance = intrinsic_importance_for_best_models(
        metrics=feature_model_metrics,
        fitted=fitted,
        features=selection.selected_features,
        targets=targets,
    )
    save_importance_plots(
        args.output_dir,
        intrinsic_importance,
        "model_feature_importance",
        "importance_normalized",
    )
    feature_groups = feature_group_table(selection.selected_features, targets)
    if args.no_ablation:
        ablation = pd.DataFrame()
    else:
        ablation = group_ablation_study(
            train_df=train_df,
            test_df=test_df,
            features=selection.selected_features,
            targets=targets,
            metrics=feature_model_metrics,
            model_factory=build_model,
            random_state=args.random_state,
            quick=args.quick,
            model_params=model_params,
        )
    if args.no_ablation or args.no_single_feature_ablation:
        single_feature_ablation = pd.DataFrame()
    else:
        single_feature_ablation = single_feature_ablation_study(
            train_df=train_df,
            test_df=test_df,
            features=selection.selected_features,
            targets=targets,
            metrics=feature_model_metrics,
            model_factory=build_model,
            random_state=args.random_state,
            quick=args.quick,
            model_params=model_params,
        )
    shap_importance = shap_for_best_models(
        output_dir=args.output_dir,
        metrics=feature_model_metrics,
        fitted=fitted,
        test_df=test_df,
        features=selection.selected_features,
        targets=targets,
        samples=args.shap_samples,
        background_size=args.shap_background_size,
        max_evals=args.shap_max_evals,
        random_state=args.random_state,
    )
    explanatory_figures = save_explanatory_figures(
        output_dir=args.output_dir,
        df=df,
        train_df=train_df,
        test_df=test_df,
        targets=targets,
        feature_groups=feature_groups,
        metrics=metrics,
        permutation=permutation,
        ablation=ablation,
        single_feature_ablation=single_feature_ablation,
    )
    residuals_by_hour = save_prediction_plots(
        output_dir=args.output_dir,
        test_df=test_df,
        predictions=predictions,
        metrics=metrics,
        targets=targets,
    )
    best_metrics_df = best_model_metrics(metrics)
    best_feature_metrics_df = best_model_metrics(feature_model_metrics)
    if args.no_save_final_models:
        saved_models = {}
    else:
        saved_models = save_final_models(
            output_dir=args.output_dir,
            df=df,
            features=selection.selected_features,
            targets=targets,
            metrics=feature_model_metrics,
            random_state=args.random_state,
            quick=args.quick,
            model_params=model_params,
        )

    selection.ranking.to_csv(args.output_dir / "feature_selection_ranking.csv", index=False)
    selection.dropped.to_csv(args.output_dir / "feature_selection_dropped.csv", index=False)
    pd.DataFrame({"feature": selection.selected_features}).to_csv(
        args.output_dir / "selected_features.csv",
        index=False,
    )
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    best_metrics_df.to_csv(args.output_dir / "best_model_metrics.csv", index=False)
    best_feature_metrics_df.to_csv(args.output_dir / "best_feature_model_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    feature_groups.to_csv(args.output_dir / "feature_groups.csv", index=False)
    intrinsic_importance.to_csv(args.output_dir / "model_feature_importance.csv", index=False)
    permutation.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    ablation.to_csv(args.output_dir / "ablation_study.csv", index=False)
    single_feature_ablation.to_csv(args.output_dir / "single_feature_ablation.csv", index=False)
    shap_importance.to_csv(args.output_dir / "shap_importance.csv", index=False)
    tuning_results.to_csv(args.output_dir / "tuning_results.csv", index=False)
    rolling_metrics.to_csv(args.output_dir / "rolling_validation_metrics.csv", index=False)
    rolling_summary.to_csv(args.output_dir / "rolling_validation_summary.csv", index=False)
    rolling_rank.to_csv(args.output_dir / "rolling_validation_rank_stability.csv", index=False)
    rolling_selected.to_csv(args.output_dir / "rolling_validation_selected_features.csv", index=False)
    bootstrap_intervals.to_csv(args.output_dir / "bootstrap_metric_intervals.csv", index=False)
    residuals_by_hour.to_csv(args.output_dir / "residuals_by_hour.csv", index=False)
    metadata = {
        "input_csv": str(args.input_csv),
        "target_set": target_set.name,
        "target_set_title": target_set.title,
        "targets": targets,
        "feature_scope": args.feature_scope,
        "effective_feature_scope": effective_feature_scope,
        "mode": args.mode,
        "horizon_hours": args.horizon_hours,
        "lags": args.lags,
        "rolling_windows": args.rolling_windows,
        "include_target_lags": args.include_target_lags,
        "include_target_rolling": args.include_target_rolling,
        "test_size": args.test_size,
        "embargo_hours": embargo_hours,
        "baseline_definition": (
            "Le baseline naive e rolling sono calcolate sui target osservati all'origine "
            "della previsione. last_hour usa y(t-1) per predire y(t+h)."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "top_k": args.top_k,
        "top_k_grid": args.top_k_grid or [],
        "top_k_auto_selected": auto_selected_top_k,
        "top_k_auto_selection_scope": auto_selection_scope,
        "allow_final_split_top_k_selection": args.allow_final_split_top_k_selection,
        "models": args.models,
        "ablation_enabled": not args.no_ablation,
        "single_feature_ablation_enabled": not args.no_ablation and not args.no_single_feature_ablation,
        "shap_samples": args.shap_samples,
        "shap_background_size": args.shap_background_size,
        "shap_max_evals": args.shap_max_evals,
        "selected_features": selection.selected_features,
        "selected_correlation_outputs": selected_correlation_outputs,
        "model_params": model_params,
        "saved_models": saved_models,
        "best_model_metrics": dataframe_records_for_json(best_metrics_df),
        "best_feature_model_metrics": dataframe_records_for_json(best_feature_metrics_df),
        "explanatory_figures": explanatory_figures,
    }
    (args.output_dir / "modeling_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        output_dir=args.output_dir,
        input_csv=args.input_csv,
        target_set=target_set,
        feature_scope=args.feature_scope,
        effective_feature_scope=effective_feature_scope,
        mode=args.mode,
        horizon_hours=args.horizon_hours,
        embargo_hours=embargo_hours,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
        top_k=args.top_k,
        top_k_auto_selection_scope=auto_selection_scope,
        train_df=train_df,
        test_df=test_df,
        selection=selection,
        metrics=metrics,
        intrinsic_importance=intrinsic_importance,
        permutation=permutation,
        ablation=ablation,
        single_feature_ablation=single_feature_ablation,
        shap_importance=shap_importance,
        explanatory_figures=explanatory_figures,
        tuning_results=tuning_results,
        rolling_metrics=rolling_metrics,
        rolling_summary=rolling_summary,
        rolling_rank=rolling_rank,
        bootstrap_intervals=bootstrap_intervals,
        residuals_by_hour=residuals_by_hour,
        best_metrics=best_metrics_df,
        best_feature_metrics=best_feature_metrics_df,
        saved_models=saved_models,
        selected_correlation_outputs=selected_correlation_outputs,
        model_names=args.models,
    )
    print(metrics.to_string(index=False))
    print(f"\n[OK] Report scritto in {args.output_dir / 'modeling_report.md'}")


if __name__ == "__main__":
    main()
