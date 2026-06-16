#!/usr/bin/env python3
"""Riassume gli output di modellazione Dozza sugli orizzonti di forecast."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dozza_modeling.plotting import configure_paper_plots, save_figure

configure_paper_plots()


ANALYSES = ("flow", "nationality", "age")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_analysis_dirs(root_output_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    horizon_dirs = [
        horizon_dir
        for horizon_dir in sorted(root_output_dir.glob("h*h"))
        if horizon_dir.is_dir() and horizon_dir.name[1:-1].isdigit()
    ]
    if not horizon_dirs:
        for analysis in ANALYSES:
            path = root_output_dir / analysis
            if (path / "model_metrics.csv").exists():
                dirs.append(path)
    for horizon_dir in horizon_dirs:
        if not horizon_dir.is_dir():
            continue
        for analysis in ANALYSES:
            path = horizon_dir / analysis
            if (path / "model_metrics.csv").exists():
                dirs.append(path)
    return sorted(set(dirs))


def frame_with_metadata(path: Path, file_name: str) -> pd.DataFrame:
    csv_path = path / file_name
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    metadata = read_json(path / "modeling_metadata.json")
    target_set = metadata.get("target_set") or path.name
    horizon_hours = metadata.get("horizon_hours")
    if horizon_hours is None:
        parent = path.parent.name
        if parent.startswith("h") and parent.endswith("h"):
            try:
                horizon_hours = int(parent[1:-1])
            except ValueError:
                horizon_hours = np.nan
    frame = frame.drop(columns=[col for col in ["analysis_dir", "target_set", "horizon_hours"] if col in frame])
    frame.insert(0, "analysis_dir", str(path))
    frame.insert(0, "target_set", target_set)
    frame.insert(0, "horizon_hours", horizon_hours)
    return frame


def collect(root_output_dir: Path, file_name: str) -> pd.DataFrame:
    parts = [frame_with_metadata(path, file_name) for path in discover_analysis_dirs(root_output_dir)]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "horizon_hours" in out:
        out["horizon_hours"] = pd.to_numeric(out["horizon_hours"], errors="coerce")
    return out.sort_values([col for col in ["horizon_hours", "target_set", "target", "model"] if col in out])


def persistence_comparison(model_metrics: pd.DataFrame, best_feature: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty or best_feature.empty:
        return pd.DataFrame()
    baseline = model_metrics[model_metrics["model"].eq("last_hour")].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline_cols = [
        "horizon_hours",
        "target_set",
        "target",
        "mae",
        "rmse",
        "r2",
        "smape_pct",
        "wape_pct",
        "mase",
    ]
    baseline = baseline[[col for col in baseline_cols if col in baseline]].rename(
        columns={col: f"last_hour_{col}" for col in baseline_cols if col not in {"horizon_hours", "target_set", "target"}}
    )
    feature_cols = [
        "horizon_hours",
        "target_set",
        "target",
        "model",
        "mae",
        "rmse",
        "r2",
        "smape_pct",
        "wape_pct",
        "mase",
    ]
    feature = best_feature[[col for col in feature_cols if col in best_feature]].rename(
        columns={
            "model": "best_feature_model",
            "mae": "best_feature_mae",
            "rmse": "best_feature_rmse",
            "r2": "best_feature_r2",
            "smape_pct": "best_feature_smape_pct",
            "wape_pct": "best_feature_wape_pct",
            "mase": "best_feature_mase",
        }
    )
    merged = baseline.merge(feature, on=["horizon_hours", "target_set", "target"], how="inner")
    if "last_hour_mae" in merged and "best_feature_mae" in merged:
        merged["feature_mae_reduction_pct_vs_last_hour"] = (
            (merged["last_hour_mae"] - merged["best_feature_mae"]) / merged["last_hour_mae"] * 100
        )
    if "last_hour_r2" in merged and "best_feature_r2" in merged:
        merged["feature_r2_delta_vs_last_hour"] = merged["best_feature_r2"] - merged["last_hour_r2"]
    return merged.sort_values(["horizon_hours", "target_set", "target"])


def plot_metric_by_horizon(
    output_dir: Path,
    frame: pd.DataFrame,
    metric: str,
    file_stem: str,
    title: str,
    ylabel: str,
) -> str | None:
    if frame.empty or metric not in frame:
        return None
    view = frame.dropna(subset=["horizon_hours", metric]).copy()
    if view.empty:
        return None
    for target_set, group in view.groupby("target_set"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        for target, target_df in group.groupby("target"):
            target_df = target_df.sort_values("horizon_hours")
            ax.plot(
                target_df["horizon_hours"],
                target_df[metric],
                marker="o",
                linewidth=1.6,
                label=str(target),
            )
        ax.set_title(f"{title} - {target_set}")
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        save_figure(fig, output_dir / f"{file_stem}_{target_set}.png", dpi=300)
        plt.close(fig)
    return str(output_dir / f"{file_stem}_<target_set>.png")


def write_report(
    output_dir: Path,
    best_overall: pd.DataFrame,
    best_feature: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Sintesi orizzonti Dozza",
        "",
        "Questa cartella riassume le run di modellazione completate sui diversi orizzonti di forecast.",
        "",
        "## File",
        "",
        "- `model_metrics_by_horizon.csv`: metriche di tutti i modelli.",
        "- `best_model_metrics_by_horizon.csv`: miglior modello per target e orizzonte.",
        "- `best_feature_model_metrics_by_horizon.csv`: miglior modello non-baseline per target e orizzonte.",
        "- `persistence_comparison_by_horizon.csv`: baseline last-hour contro miglior modello feature-based.",
        "- `*_by_horizon_*.png` e `*_by_horizon_*.pdf`: figure di confronto pronte per il paper.",
        "",
    ]
    if not best_overall.empty:
        lines.extend(
            [
                "## Copertura",
                "",
                f"- Orizzonti: {', '.join(str(int(h)) for h in sorted(best_overall['horizon_hours'].dropna().unique()))}",
                f"- Analisi: {', '.join(sorted(best_overall['target_set'].dropna().astype(str).unique()))}",
                "",
            ]
        )
    if not comparison.empty and "feature_mae_reduction_pct_vs_last_hour" in comparison:
        top = comparison.sort_values("feature_mae_reduction_pct_vs_last_hour", ascending=False).head(10)
        lines.extend(["## Maggiori guadagni dei modelli feature-based rispetto a last-hour", ""])
        for _, row in top.iterrows():
            lines.append(
                "- h={h:g}, {target}: {model}, riduzione MAE {gain:.2f}%".format(
                    h=row["horizon_hours"],
                    target=row["target"],
                    model=row["best_feature_model"],
                    gain=row["feature_mae_reduction_pct_vs_last_hour"],
                )
            )
        lines.append("")
    output_dir.joinpath("horizon_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Dozza outputs across forecast horizons.")
    parser.add_argument("--root-output-dir", type=Path, default=Path("outputs/dozza_three_analyses"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dozza_three_analyses/horizon_summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = collect(args.root_output_dir, "model_metrics.csv")
    best_overall = collect(args.root_output_dir, "best_model_metrics.csv")
    best_feature = collect(args.root_output_dir, "best_feature_model_metrics.csv")
    rolling = collect(args.root_output_dir, "rolling_validation_summary.csv")
    comparison = persistence_comparison(model_metrics, best_feature)

    model_metrics.to_csv(args.output_dir / "model_metrics_by_horizon.csv", index=False)
    best_overall.to_csv(args.output_dir / "best_model_metrics_by_horizon.csv", index=False)
    best_feature.to_csv(args.output_dir / "best_feature_model_metrics_by_horizon.csv", index=False)
    rolling.to_csv(args.output_dir / "rolling_validation_summary_by_horizon.csv", index=False)
    comparison.to_csv(args.output_dir / "persistence_comparison_by_horizon.csv", index=False)

    plot_metric_by_horizon(
        args.output_dir,
        best_overall,
        "mae",
        "best_model_mae_by_horizon",
        "Best model MAE by horizon",
        "MAE",
    )
    plot_metric_by_horizon(
        args.output_dir,
        best_overall,
        "r2",
        "best_model_r2_by_horizon",
        "Best model R2 by horizon",
        "R2",
    )
    plot_metric_by_horizon(
        args.output_dir,
        best_feature,
        "mae",
        "best_feature_model_mae_by_horizon",
        "Best feature-model MAE by horizon",
        "MAE",
    )
    plot_metric_by_horizon(
        args.output_dir,
        comparison,
        "feature_mae_reduction_pct_vs_last_hour",
        "feature_gain_vs_last_hour_by_horizon",
        "Feature-model gain over last-hour by horizon",
        "MAE reduction vs last-hour (%)",
    )
    write_report(args.output_dir, best_overall, best_feature, comparison)
    print(f"[OK] Wrote horizon summary to {args.output_dir}")


if __name__ == "__main__":
    main()
