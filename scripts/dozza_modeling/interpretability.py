"""Utility di interpretabilita' condivise dalle analisi Dozza."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .targets import feature_group

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import configure_paper_plots, save_figure

configure_paper_plots()


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    values = np.zeros_like(denom, dtype=float)
    np.divide(
        2.0 * np.abs(y_pred - y_true),
        denom,
        out=values,
        where=denom != 0,
    )
    return float(np.mean(values) * 100)


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_true_values = y_true.to_numpy(dtype=float)
    y_pred_values = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true_values)))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": _smape(y_true_values, y_pred_values),
        "wape_pct": float(np.sum(np.abs(y_pred_values - y_true_values)) / denom * 100) if denom else np.nan,
    }


def _final_estimator(model: Any) -> Any:
    if hasattr(model, "regressor_"):
        return _final_estimator(model.regressor_)
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _importance_values(estimator: Any) -> tuple[str, np.ndarray] | None:
    if hasattr(estimator, "feature_importances_"):
        return "feature_importances_", np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        return "abs_coef", np.abs(coef).ravel()
    return None


def intrinsic_importance_for_best_models(
    metrics: pd.DataFrame,
    fitted: dict[tuple[str, str], Any],
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    rows = []
    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        model = fitted.get((model_name, target))
        if model is None:
            continue
        values = _importance_values(_final_estimator(model))
        if values is None:
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature": None,
                    "importance_type": "not_available",
                    "importance": np.nan,
                    "importance_normalized": np.nan,
                }
            )
            continue
        importance_type, importance = values
        if len(importance) != len(features):
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature": None,
                    "importance_type": "shape_mismatch",
                    "importance": np.nan,
                    "importance_normalized": np.nan,
                }
            )
            continue
        denom = float(np.sum(np.abs(importance)))
        normalized = importance / denom if denom > 0 else np.zeros_like(importance)
        for feature, value, norm_value in zip(features, importance, normalized):
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature": feature,
                    "feature_group": feature_group(feature, targets),
                    "importance_type": importance_type,
                    "importance": float(value),
                    "importance_normalized": float(norm_value),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "target",
                "model",
                "feature",
                "feature_group",
                "importance_type",
                "importance",
                "importance_normalized",
            ]
        )
    return pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])


def save_importance_plots(output_dir: Path, importance: pd.DataFrame, stem: str, value_col: str) -> None:
    if importance.empty or value_col not in importance:
        return
    for target, target_df in importance.dropna(subset=["feature"]).groupby("target"):
        view = target_df.sort_values(value_col, ascending=False).head(25)
        if view.empty:
            continue
        fig_height = max(5.0, min(12.0, 0.32 * len(view) + 2.0))
        fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
        ordered = view.iloc[::-1]
        ax.barh(ordered["feature"], ordered[value_col])
        ax.set_title(f"{stem} - {target}")
        ax.set_xlabel(value_col)
        save_figure(fig, output_dir / f"{stem}_{target}.png", dpi=300)
        plt.close(fig)


def save_target_overview_plot(
    output_dir: Path,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str],
) -> str | None:
    if df.empty or not targets:
        return None
    plot_df = df[["timestamp", *targets]].copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    split_time = pd.to_datetime(test_df["timestamp"].min()) if not test_df.empty else None
    n_targets = len(targets)
    fig_height = min(max(3.0, 2.1 * n_targets), 16.0)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, fig_height), sharex=True, constrained_layout=True)
    if n_targets == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        ax.plot(plot_df["timestamp"], plot_df[target], linewidth=1.1)
        if split_time is not None:
            ax.axvline(split_time, color="black", linestyle="--", linewidth=1, label="inizio test")
        ax.set_title(target)
        ax.set_ylabel("valore")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestamp")
    path = output_dir / "target_timeseries_train_test.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_target_distribution_plot(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str],
) -> str | None:
    if not targets:
        return None
    data = []
    labels = []
    for target in targets:
        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            values = split_df[target].dropna().to_numpy(dtype=float)
            if len(values):
                data.append(values)
                labels.append(f"{target}\n{split_name}")
    if not data:
        return None
    fig_width = min(max(10.0, 0.75 * len(data)), 22.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Distribuzione target: train vs test")
    ax.set_ylabel("valore")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "target_distribution_train_test.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def _metric_pivot(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    return metrics.pivot_table(index="target", columns="model", values=metric, aggfunc="mean")


def save_metric_heatmap(
    output_dir: Path,
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    cmap: str,
) -> str | None:
    if metrics.empty or metric not in metrics:
        return None
    matrix = _metric_pivot(metrics, metric)
    if matrix.empty:
        return None
    fig_width = min(max(7.0, 1.15 * len(matrix.columns) + 3.0), 18.0)
    fig_height = min(max(4.0, 0.65 * len(matrix.index) + 2.0), 12.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    path = output_dir / f"model_metric_{metric}.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_feature_group_counts_plot(output_dir: Path, feature_groups: pd.DataFrame) -> str | None:
    if feature_groups.empty or "feature_group" not in feature_groups:
        return None
    counts = feature_groups["feature_group"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(counts.index, counts.values)
    ax.set_title("Feature selezionate per gruppo")
    ax.set_ylabel("numero feature")
    ax.tick_params(axis="x", rotation=30)
    path = output_dir / "selected_feature_groups.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_ablation_delta_plot(output_dir: Path, ablation: pd.DataFrame) -> str | None:
    if ablation.empty or "delta_mae" not in ablation:
        return None
    view = ablation
    if "ablation_type" in view:
        filtered = view[view["ablation_type"].eq("leave_one_group_out")]
        if not filtered.empty:
            view = filtered
    matrix = view.pivot_table(index="target", columns="ablated_group", values="delta_mae", aggfunc="mean")
    if matrix.empty:
        return None
    fig_width = min(max(7.0, 1.1 * len(matrix.columns) + 3.0), 16.0)
    fig_height = min(max(4.0, 0.65 * len(matrix.index) + 2.0), 12.0)
    vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))) if matrix.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Studio ablativo: delta MAE per gruppo rimosso")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="delta MAE")
    path = output_dir / "ablation_delta_mae.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_ablation_group_set_plot(output_dir: Path, ablation: pd.DataFrame) -> str | None:
    if ablation.empty or "ablation_type" not in ablation or "mae_ablated" not in ablation:
        return None
    view = ablation[ablation["ablation_type"].eq("group_set")].copy()
    if view.empty:
        return None
    view["scenario"] = view["kept_groups"].fillna(view["ablated_group"])
    matrix = view.pivot_table(index="target", columns="scenario", values="mae_ablated", aggfunc="mean")
    if matrix.empty:
        return None
    fig_width = min(max(8.0, 1.05 * len(matrix.columns) + 3.0), 20.0)
    fig_height = min(max(4.0, 0.65 * len(matrix.index) + 2.0), 12.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    ax.set_title("Studio ablativo: MAE per scenario di gruppi feature")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="MAE")
    path = output_dir / "ablation_group_set_mae.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_single_feature_ablation_plot(output_dir: Path, single_feature_ablation: pd.DataFrame) -> str | None:
    if single_feature_ablation.empty or "delta_mae" not in single_feature_ablation:
        return None
    top_rows = []
    for _, target_frame in single_feature_ablation.groupby("target", sort=True):
        top_rows.append(target_frame.sort_values("delta_mae", ascending=False).head(8))
    view = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    if view.empty:
        return None
    view = view.sort_values(["target", "delta_mae"], ascending=[True, True]).reset_index(drop=True)
    labels = [f"{row.target}: {row.ablated_feature}" for row in view.itertuples()]
    fig_height = min(max(4.5, 0.28 * len(view) + 1.8), 14.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    colors = ["#b14a4a" if value >= 0 else "#4f7db8" for value in view["delta_mae"]]
    ax.barh(np.arange(len(view)), view["delta_mae"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(view)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("delta MAE")
    ax.set_title("Single-feature ablation: top MAE changes after feature removal")
    path = output_dir / "ablation_single_feature_delta_mae.png"
    save_figure(fig, path, dpi=300)
    plt.close(fig)
    return str(path)


def save_explanatory_figures(
    output_dir: Path,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str],
    feature_groups: pd.DataFrame,
    metrics: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
    single_feature_ablation: pd.DataFrame | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    paths = {
        "target_timeseries_train_test": save_target_overview_plot(output_dir, df, train_df, test_df, targets),
        "target_distribution_train_test": save_target_distribution_plot(output_dir, train_df, test_df, targets),
        "model_metric_mae": save_metric_heatmap(output_dir, metrics, "mae", "MAE per target e modello", "viridis_r"),
        "model_metric_r2": save_metric_heatmap(output_dir, metrics, "r2", "R2 per target e modello", "viridis"),
        "model_metric_smape_pct": save_metric_heatmap(
            output_dir,
            metrics,
            "smape_pct",
            "sMAPE percentuale per target e modello",
            "viridis_r",
        ),
        "model_metric_wape_pct": save_metric_heatmap(
            output_dir,
            metrics,
            "wape_pct",
            "WAPE percentuale per target e modello",
            "viridis_r",
        ),
        "model_metric_mase": save_metric_heatmap(
            output_dir,
            metrics,
            "mase",
            "MASE per target e modello",
            "viridis_r",
        ),
        "model_metric_peak_f1": save_metric_heatmap(
            output_dir,
            metrics,
            "peak_f1",
            "F1 ore ad alta affluenza per target e modello",
            "viridis",
        ),
        "model_metric_fit_seconds": save_metric_heatmap(
            output_dir,
            metrics,
            "fit_seconds",
            "Tempo training in secondi per target e modello",
            "viridis_r",
        ),
        "model_metric_inference_ms_per_row": save_metric_heatmap(
            output_dir,
            metrics,
            "inference_ms_per_row",
            "Tempo inferenza per riga in millisecondi",
            "viridis_r",
        ),
        "selected_feature_groups": save_feature_group_counts_plot(output_dir, feature_groups),
        "ablation_delta_mae": save_ablation_delta_plot(output_dir, ablation),
        "ablation_group_set_mae": save_ablation_group_set_plot(output_dir, ablation),
        "ablation_single_feature_delta_mae": save_single_feature_ablation_plot(
            output_dir,
            single_feature_ablation if single_feature_ablation is not None else pd.DataFrame(),
        ),
    }
    save_importance_plots(output_dir, permutation, "permutation_importance", "importance_mean")
    if not permutation.empty:
        for target in sorted(permutation["target"].dropna().unique()):
            path = output_dir / f"permutation_importance_{target}.png"
            if path.exists():
                paths[f"permutation_importance_{target}"] = str(path)
    for key, value in paths.items():
        if value:
            outputs[key] = value
    return outputs


def feature_group_table(features: list[str], targets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "feature_group": [feature_group(feature, targets) for feature in features],
        }
    )


def _group_scenarios(available_groups: set[str]) -> list[tuple[str, set[str]]]:
    base_scenarios: list[tuple[str, set[str]]] = [
        ("calendar", {"calendar"}),
        ("calendar+meteo", {"calendar", "meteo"}),
        ("calendar+events", {"calendar", "events"}),
        ("calendar+target_history", {"calendar", "target_history"}),
        ("calendar+tim", {"calendar", "tim"}),
        ("calendar+pedoni", {"calendar", "pedoni"}),
        ("calendar+meteo+events", {"calendar", "meteo", "events"}),
        ("calendar+meteo+target_history", {"calendar", "meteo", "target_history"}),
        ("calendar+meteo+tim", {"calendar", "meteo", "tim"}),
        ("calendar+meteo+pedoni", {"calendar", "meteo", "pedoni"}),
        ("calendar+meteo+events+tim", {"calendar", "meteo", "events", "tim"}),
        ("calendar+meteo+events+pedoni", {"calendar", "meteo", "events", "pedoni"}),
        ("calendar+meteo+tim+pedoni", {"calendar", "meteo", "tim", "pedoni"}),
        ("calendar+meteo+events+tim+pedoni", {"calendar", "meteo", "events", "tim", "pedoni"}),
        ("full", set(available_groups)),
    ]
    scenarios = []
    seen = set()
    for label, groups in base_scenarios:
        kept = groups.intersection(available_groups)
        if not kept:
            continue
        key = tuple(sorted(kept))
        if key in seen:
            continue
        seen.add(key)
        scenarios.append((label if label != "full" else "+".join(sorted(kept)), kept))
    return scenarios


def group_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    metrics: pd.DataFrame,
    model_factory: Callable[[str, int, bool, dict[str, Any] | None], Any],
    random_state: int,
    quick: bool,
    model_params: dict[str, dict[str, Any]] | None,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    groups = feature_group_table(features, targets)
    available_groups = set(groups["feature_group"].unique())
    rows = []
    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        full_scores = {
            "mae": float(best.iloc[0]["mae"]),
            "rmse": float(best.iloc[0]["rmse"]),
            "r2": float(best.iloc[0]["r2"]),
            "smape_pct": float(best.iloc[0]["smape_pct"]),
            "wape_pct": float(best.iloc[0]["wape_pct"]) if "wape_pct" in best else np.nan,
        }
        for group_name in sorted(groups["feature_group"].unique()):
            removed = groups.loc[groups["feature_group"].eq(group_name), "feature"].tolist()
            kept = [feature for feature in features if feature not in removed]
            if not removed or not kept:
                continue
            model = model_factory(
                model_name,
                random_state,
                quick,
                (model_params or {}).get(model_name, {}),
            )
            model = clone(model)
            model.fit(train_df[kept], train_df[target])
            pred = np.clip(model.predict(test_df[kept]), a_min=0, a_max=None)
            scores = _scores(test_df[target], pred)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "ablation_type": "leave_one_group_out",
                    "ablated_group": group_name,
                    "kept_groups": "+".join(sorted(available_groups - {group_name})),
                    "removed_groups": group_name,
                    "n_features_full": len(features),
                    "n_features_removed": len(removed),
                    "n_features_kept": len(kept),
                    "mae_full": full_scores["mae"],
                    "mae_ablated": scores["mae"],
                    "delta_mae": scores["mae"] - full_scores["mae"],
                    "rmse_full": full_scores["rmse"],
                    "rmse_ablated": scores["rmse"],
                    "delta_rmse": scores["rmse"] - full_scores["rmse"],
                    "r2_full": full_scores["r2"],
                    "r2_ablated": scores["r2"],
                    "delta_r2": scores["r2"] - full_scores["r2"],
                    "smape_pct_full": full_scores["smape_pct"],
                    "smape_pct_ablated": scores["smape_pct"],
                    "delta_smape_pct": scores["smape_pct"] - full_scores["smape_pct"],
                    "wape_pct_full": full_scores["wape_pct"],
                    "wape_pct_ablated": scores.get("wape_pct", np.nan),
                    "delta_wape_pct": scores.get("wape_pct", np.nan) - full_scores["wape_pct"],
                }
            )
        for scenario_name, kept_groups in _group_scenarios(available_groups):
            kept = groups.loc[groups["feature_group"].isin(kept_groups), "feature"].tolist()
            removed = [feature for feature in features if feature not in kept]
            if not kept:
                continue
            model = model_factory(
                model_name,
                random_state,
                quick,
                (model_params or {}).get(model_name, {}),
            )
            model = clone(model)
            model.fit(train_df[kept], train_df[target])
            pred = np.clip(model.predict(test_df[kept]), a_min=0, a_max=None)
            scores = _scores(test_df[target], pred)
            kept_label = "+".join(sorted(kept_groups))
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "ablation_type": "group_set",
                    "ablated_group": f"keep:{scenario_name}",
                    "kept_groups": kept_label,
                    "removed_groups": "+".join(sorted(available_groups - kept_groups)),
                    "n_features_full": len(features),
                    "n_features_removed": len(removed),
                    "n_features_kept": len(kept),
                    "mae_full": full_scores["mae"],
                    "mae_ablated": scores["mae"],
                    "delta_mae": scores["mae"] - full_scores["mae"],
                    "rmse_full": full_scores["rmse"],
                    "rmse_ablated": scores["rmse"],
                    "delta_rmse": scores["rmse"] - full_scores["rmse"],
                    "r2_full": full_scores["r2"],
                    "r2_ablated": scores["r2"],
                    "delta_r2": scores["r2"] - full_scores["r2"],
                    "smape_pct_full": full_scores["smape_pct"],
                    "smape_pct_ablated": scores["smape_pct"],
                    "delta_smape_pct": scores["smape_pct"] - full_scores["smape_pct"],
                    "wape_pct_full": full_scores["wape_pct"],
                    "wape_pct_ablated": scores.get("wape_pct", np.nan),
                    "delta_wape_pct": scores.get("wape_pct", np.nan) - full_scores["wape_pct"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target", "ablation_type", "delta_mae"],
        ascending=[True, True, False],
    )


def single_feature_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    metrics: pd.DataFrame,
    model_factory: Callable[[str, int, bool, dict[str, Any] | None], Any],
    random_state: int,
    quick: bool,
    model_params: dict[str, dict[str, Any]] | None,
) -> pd.DataFrame:
    if len(features) <= 1:
        return pd.DataFrame()
    groups = feature_group_table(features, targets)
    feature_to_group = dict(zip(groups["feature"], groups["feature_group"]))
    rows = []
    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        full_scores = {
            "mae": float(best.iloc[0]["mae"]),
            "rmse": float(best.iloc[0]["rmse"]),
            "r2": float(best.iloc[0]["r2"]),
            "smape_pct": float(best.iloc[0]["smape_pct"]),
            "wape_pct": float(best.iloc[0]["wape_pct"]) if "wape_pct" in best else np.nan,
        }
        for removed_feature in features:
            kept = [feature for feature in features if feature != removed_feature]
            if not kept:
                continue
            model = model_factory(
                model_name,
                random_state,
                quick,
                (model_params or {}).get(model_name, {}),
            )
            model = clone(model)
            model.fit(train_df[kept], train_df[target])
            pred = np.clip(model.predict(test_df[kept]), a_min=0, a_max=None)
            scores = _scores(test_df[target], pred)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "ablation_type": "leave_one_feature_out",
                    "ablated_feature": removed_feature,
                    "feature_group": feature_to_group.get(removed_feature, "other"),
                    "n_features_full": len(features),
                    "n_features_removed": 1,
                    "n_features_kept": len(kept),
                    "mae_full": full_scores["mae"],
                    "mae_ablated": scores["mae"],
                    "delta_mae": scores["mae"] - full_scores["mae"],
                    "rmse_full": full_scores["rmse"],
                    "rmse_ablated": scores["rmse"],
                    "delta_rmse": scores["rmse"] - full_scores["rmse"],
                    "r2_full": full_scores["r2"],
                    "r2_ablated": scores["r2"],
                    "delta_r2": scores["r2"] - full_scores["r2"],
                    "smape_pct_full": full_scores["smape_pct"],
                    "smape_pct_ablated": scores["smape_pct"],
                    "delta_smape_pct": scores["smape_pct"] - full_scores["smape_pct"],
                    "wape_pct_full": full_scores["wape_pct"],
                    "wape_pct_ablated": scores.get("wape_pct", np.nan),
                    "delta_wape_pct": scores.get("wape_pct", np.nan) - full_scores["wape_pct"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target", "delta_mae", "ablated_feature"],
        ascending=[True, False, True],
    )


def shap_for_best_models(
    output_dir: Path,
    metrics: pd.DataFrame,
    fitted: dict[tuple[str, str], Any],
    test_df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    samples: int,
    background_size: int,
    max_evals: int | None,
    random_state: int,
) -> pd.DataFrame:
    status_path = output_dir / "shap_status.json"
    if samples <= 0:
        status_path.write_text(json.dumps({"status": "disabled"}, indent=2), encoding="utf-8")
        return pd.DataFrame()
    try:
        import shap  # type: ignore
    except ImportError as exc:
        status_path.write_text(
            json.dumps({"status": "skipped", "reason": f"shap non installato: {exc}"}, indent=2),
            encoding="utf-8",
        )
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    X = test_df[features]
    if len(X) == 0:
        return pd.DataFrame()
    sample_n = min(samples, len(X))
    background_n = min(background_size, len(X))
    sample_idx = np.sort(rng.choice(len(X), size=sample_n, replace=False))
    background_idx = np.sort(rng.choice(len(X), size=background_n, replace=False))
    X_sample = X.iloc[sample_idx].copy()
    X_background = X.iloc[background_idx].copy()
    effective_max_evals = max_evals or max(2 * len(features) + 1, 50)
    rows = []

    for target in targets:
        best = metrics[metrics["target"].eq(target)].sort_values("mae").head(1)
        if best.empty:
            continue
        model_name = str(best.iloc[0]["model"])
        model = fitted.get((model_name, target))
        if model is None:
            continue

        def predict_fn(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(values, columns=features)
            return np.clip(model.predict(frame), a_min=0, a_max=None)

        print(
            f"[SHAP] target={target}, modello={model_name}, samples={sample_n}, "
            f"background={background_n}, max_evals={effective_max_evals}",
            flush=True,
        )
        explainer = shap.PermutationExplainer(predict_fn, X_background.to_numpy())
        explanation = explainer(X_sample.to_numpy(), max_evals=effective_max_evals)
        values = np.asarray(explanation.values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, 0]
        mean_abs = np.mean(np.abs(values), axis=0)
        for feature, value in zip(features, mean_abs):
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature": feature,
                    "feature_group": feature_group(feature, targets),
                    "mean_abs_shap": float(value),
                    "n_samples": sample_n,
                    "background_size": background_n,
                    "max_evals": effective_max_evals,
                }
            )

    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "samples": sample_n,
                "background_size": background_n,
                "max_evals": effective_max_evals,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["target", "mean_abs_shap"], ascending=[True, False])
    save_importance_plots(output_dir, out, "shap_importance", "mean_abs_shap")
    return out
