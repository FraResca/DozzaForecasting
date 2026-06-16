#!/usr/bin/env python3
"""Genera tabelle CSV e figure per il manoscritto LNCS su Dozza."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dozza_lncs_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "dozza_lncs_paper"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
OUTPUTS = ROOT / "outputs"
PREPROCESS = OUTPUTS / "slurm_dozza_preprocess"
ANALYSES = OUTPUTS / "slurm_dozza_three_analyses"
SUMMARY = ANALYSES / "horizon_summary"

TRACKS = ["flow", "nationality", "age"]
HORIZONS = [1, 3, 6, 12, 24]
PRIMARY_HORIZON = 1


def save_csv(name: str, frame_or_rows: pd.DataFrame | list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame = frame_or_rows if isinstance(frame_or_rows, pd.DataFrame) else pd.DataFrame(frame_or_rows)
    frame.to_csv(TABLE_DIR / name, index=False)


def round_numeric(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def latex_path(value: object) -> str:
    text = str(value).replace("|", "/")
    return rf"\path|{text}|"


def latex_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def compact_label(value: object, mapping: dict[str, str]) -> str:
    return mapping.get(str(value), str(value).replace("_", "\\_"))


def analysis_dir(track: str, horizon: int = PRIMARY_HORIZON) -> Path:
    return ANALYSES / f"h{horizon}h" / track


def metric_row(row: pd.Series, track: str, horizon: int = PRIMARY_HORIZON) -> dict[str, object]:
    return {
        "horizon_hours": horizon,
        "track": track,
        "target": row["target"],
        "model": row["model"],
        "model_family": row["model_family"],
        "mae": round(float(row["mae"]), 4),
        "rmse": round(float(row["rmse"]), 4),
        "r2": round(float(row["r2"]), 4),
        "smape_pct": round(float(row["smape_pct"]), 4),
        "wape_pct": round(float(row["wape_pct"]), 4),
        "mase": round(float(row["mase"]), 4),
        "peak_f1": round(float(row["peak_f1"]), 4) if pd.notna(row.get("peak_f1")) else "",
        "fit_seconds": round(float(row["fit_seconds"]), 4),
        "inference_ms_per_row": round(float(row["inference_ms_per_row"]), 4),
    }


def build_dataset_summary() -> None:
    metadata_path = PREPROCESS / "tim_aggregation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    joined_path = PREPROCESS / "dozza_joined_hourly_inner_with_events.csv"
    if not joined_path.exists():
        joined_path = PREPROCESS / "dozza_joined_hourly_inner.csv"
    joined_header = pd.read_csv(joined_path, nrows=0)
    event_cols = [col for col in joined_header.columns if col.startswith("event_")]
    joined_timestamps = pd.read_csv(joined_path, usecols=["timestamp"])
    weather = pd.read_csv(PREPROCESS / "weather_hourly_features.csv", usecols=["timestamp"])
    event_active_rows = 0
    if event_cols:
        event_values = pd.read_csv(joined_path, usecols=event_cols)
        event_active_rows = int(event_values.fillna(0).abs().sum(axis=1).gt(0).sum())

    save_csv(
        "dataset_summary.csv",
        [
            {
                "component": "Pedestrian raw sheet",
                "description": "Hourly camera rows in the pedestrian Excel source",
                "quantity": 77769,
            },
            {
                "component": "Aggregated pedestrian timeline",
                "description": "Unique hourly pedestrian timestamps after aggregation",
                "quantity": 26744,
            },
            {
                "component": "Complete pedestrian targets",
                "description": "Hours with complete entrance and exit targets",
                "quantity": 25036,
            },
            {
                "component": "TIM raw inputs",
                "description": "15-minute TIM rows read from compressed CSV files",
                "quantity": metadata["raw_rows"],
            },
            {
                "component": "TIM files",
                "description": "Compressed TIM CSV files processed",
                "quantity": metadata["files_processed"],
            },
            {
                "component": "TIM hourly timeline",
                "description": "Hourly TIM timestamps after 15-minute aggregation",
                "quantity": 2883,
            },
            {
                "component": "Weather hourly timeline",
                "description": "ERG5 hourly weather rows joined from two grid cells",
                "quantity": len(weather),
            },
            {
                "component": "Aligned event-aware modeling dataset",
                "description": "Inner joined hourly rows used by the multi-horizon modeling outputs",
                "quantity": len(joined_timestamps),
            },
            {
                "component": "Event feature columns",
                "description": "Hourly event-intensity, distance, city, category, and lag features",
                "quantity": len(event_cols),
            },
            {
                "component": "Rows with non-zero event signal",
                "description": "Modeling rows where at least one event feature is active",
                "quantity": event_active_rows,
            },
        ],
    )


def build_dataset_profile() -> None:
    joined_path = PREPROCESS / "dozza_joined_hourly_inner_with_events.csv"
    if not joined_path.exists():
        joined_path = PREPROCESS / "dozza_joined_hourly_inner.csv"
    frame = pd.read_csv(joined_path, parse_dates=["timestamp"])

    span_hours = int(((frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() // 3600) + 1)
    coverage_pct = 100.0 * len(frame) / span_hours
    weekend_pct = 100.0 * frame["is_weekend"].mean()
    foreign_share_pct = 100.0 * (frame["tim_Ns_mean15"] / frame["tim_P_mean15"]).mean()
    age_51_60_share_pct = 100.0 * (frame["tim_F5_mean15"] / frame["tim_Tc_mean15"]).mean()
    age_61_share_pct = 100.0 * (frame["tim_F6_mean15"] / frame["tim_Tc_mean15"]).mean()
    rainy_hours_pct = 100.0 * frame["meteo_erg5_prec_is_positive"].mean()
    event_active_pct = 100.0 * frame["event_active_any"].mean() if "event_active_any" in frame else 0.0

    core_cols = [
        "ingressi_borgo",
        "uscite_borgo",
        "tim_P_mean15",
        "tim_Ni_mean15",
        "tim_Ns_mean15",
        "meteo_erg5_tavg_mean",
        "meteo_erg5_prec_mean",
    ]
    if "event_active_any" in frame:
        core_cols.append("event_active_any")
    core_missing_pct = 100.0 * frame[core_cols].isna().mean().mean()

    save_csv(
        "dataset_profile.csv",
        [
            {
                "aspect": "Aligned temporal window",
                "value": f"{frame['timestamp'].min():%Y-%m-%d %H:%M} to {frame['timestamp'].max():%Y-%m-%d %H:%M}",
                "diagnostic": f"{len(frame):,} observed hours over {span_hours:,} calendar hours ({coverage_pct:.1f}% coverage)",
            },
            {
                "aspect": "Weekend representation",
                "value": f"{weekend_pct:.1f}% of aligned rows",
                "diagnostic": "Weekend hours are present but weekdays dominate the overlapping TIM period",
            },
            {
                "aspect": "Entrance-count scale",
                "value": f"median {frame['ingressi_borgo'].median():.0f}, p90 {frame['ingressi_borgo'].quantile(0.90):.0f}, max {frame['ingressi_borgo'].max():.0f}",
                "diagnostic": "Hourly pedestrian counts are right-tailed, with occasional large peaks",
            },
            {
                "aspect": "Exit-count scale",
                "value": f"median {frame['uscite_borgo'].median():.0f}, p90 {frame['uscite_borgo'].quantile(0.90):.0f}, max {frame['uscite_borgo'].max():.0f}",
                "diagnostic": "Exit counts have a similar skewed scale to entrance counts",
            },
            {
                "aspect": "TIM total-presence scale",
                "value": f"mean {frame['tim_P_mean15'].mean():,.1f}, p90 {frame['tim_P_mean15'].quantile(0.90):,.1f}",
                "diagnostic": "TIM presences are a much smoother and larger aggregate than camera counts",
            },
            {
                "aspect": "TIM nationality mix",
                "value": f"foreign share {foreign_share_pct:.1f}% on average",
                "diagnostic": "Foreign presences are a small-scale target relative to Italian presences",
            },
            {
                "aspect": "TIM age composition",
                "value": f"51-60 share {age_51_60_share_pct:.1f}%, 61+ share {age_61_share_pct:.1f}%",
                "diagnostic": "Older age bands form the largest part of the Tc demographic aggregate",
            },
            {
                "aspect": "Weather conditions",
                "value": f"mean temperature {frame['meteo_erg5_tavg_mean'].mean():.1f} C; rainy hours {rainy_hours_pct:.1f}%",
                "diagnostic": "The overlap period is mostly spring-summer and contains relatively few rainy hours",
            },
            {
                "aspect": "Event-signal density",
                "value": f"{event_active_pct:.1f}% of aligned rows have at least one event feature active",
                "diagnostic": "Event variables are frequent but vary by distance, city, category, and intensity",
            },
            {
                "aspect": "Core-field missingness",
                "value": f"{core_missing_pct:.1f}% average missingness",
                "diagnostic": "The inner join produces complete core target, TIM, weather, and event fields",
            },
        ],
    )


def build_tim_dictionary() -> None:
    save_csv(
        "tim_dictionary.csv",
        [
            {
                "fields": "event_time; cod_regione; cod_provincia; cod_comune; cod_ace; x; y",
                "meaning": "15-minute timestamp and spatial identifiers for region, province, municipality, ACE area, and grid tile.",
            },
            {"fields": "P, Ni, Ns", "meaning": "Total presences, Italian presences, and foreign presences; Ni + Ns = P in the raw data."},
            {"fields": "Tc, Tb", "meaning": "Italian consumer users with demographic detail and Italian business/non-consumer users; Tc + Tb approximates Ni."},
            {"fields": "Gm, Gf", "meaning": "Male and female aggregates within Tc; small differences may reflect rounding or unknown categories."},
            {"fields": "F1-F6", "meaning": "Age bands within Tc: <18, 18-30, 31-40, 41-50, 51-60, and 61+; F1+...+F6 = Tc."},
            {"fields": "Vp, Vr, Vi, Ve", "meaning": "Italian mobility categories: commuters, residents, intraregional visitors, and extraregional visitors; their sum approximates Ni."},
            {"fields": "mean15, sum15, max15", "meaning": "Hourly mean, sum, and maximum over 15-minute observations; sum15 is a person-quarter-hour aggregate."},
        ],
    )


def build_experimental_setup() -> None:
    save_csv(
        "experimental_setup.csv",
        [
            {"item": "Forecast mode", "value": "forecast"},
            {"item": "Prediction horizons", "value": "1, 3, 6, 12, and 24 hours"},
            {"item": "Primary reported horizon", "value": "1 hour, with multi-horizon robustness tables"},
            {"item": "Train split", "value": "2,164 rows, 2025-03-04 18:00:00 to 2025-09-05 07:00:00"},
            {"item": "Test split", "value": "541 rows, 2025-09-05 08:00:00 to 2025-09-30 22:00:00"},
            {"item": "Lags", "value": "1, 2, 24, and 168 hours"},
            {"item": "Rolling windows", "value": "3, 6, and 24 hours"},
            {
                "item": "Execution nodes",
                "value": "CPU jobs ran on cnode04.hpc.prv and cnode06.hpc.prv, both in the cnode0[3-6] class: 2 x AMD EPYC 9454 48-Core CPUs and 512 GB DDR5 RAM per node",
            },
            {
                "item": "Slurm resources",
                "value": "Model jobs requested 4 CPU threads; 16 GB RAM for flow/nationality jobs and 24 GB RAM for age jobs",
            },
            {"item": "Top-k feature selection", "value": "Rolling validation over k in {15, 25, 30, 40, 60} for every track and horizon"},
            {"item": "Models", "value": "Persistence baselines, Ridge/log-Ridge, Poisson, Tweedie, Random Forest, Extra Trees, HGB, XGBoost, LightGBM, and two-stage Ridge"},
            {
                "item": "Interpretability",
                "value": "Permutation importance, SHAP, leave-one-group-out ablation, leave-one-feature-out ablation, and event-feature ablation summaries",
            },
        ],
    )


def build_metric_tables() -> None:
    all_best: list[dict[str, object]] = []
    feature_best: list[dict[str, object]] = []
    persistence: list[dict[str, object]] = []
    rolling_best: list[dict[str, object]] = []

    for track in TRACKS:
        base = analysis_dir(track)
        best = pd.read_csv(base / "best_model_metrics.csv")
        best_feature = pd.read_csv(base / "best_feature_model_metrics.csv")
        metrics = pd.read_csv(base / "model_metrics.csv")
        rolling = pd.read_csv(base / "rolling_validation_summary.csv")

        all_best.extend(metric_row(row, track) for _, row in best.iterrows())
        feature_best.extend(metric_row(row, track) for _, row in best_feature.iterrows())

        for target, group in metrics.groupby("target"):
            last_hour = group[group["model"].eq("last_hour")]
            best_row = group.loc[group["mae"].idxmin()]
            feature_row = group[group["model_family"].eq("feature_model")].sort_values("mae").iloc[0]
            if not last_hour.empty:
                last = last_hour.iloc[0]
                persistence.append(
                    {
                        "horizon_hours": PRIMARY_HORIZON,
                        "track": track,
                        "target": target,
                        "best_overall_model": best_row["model"],
                        "best_overall_mae": round(float(best_row["mae"]), 4),
                        "last_hour_mae": round(float(last["mae"]), 4),
                        "last_hour_r2": round(float(last["r2"]), 4),
                        "best_feature_model": feature_row["model"],
                        "best_feature_mae": round(float(feature_row["mae"]), 4),
                        "best_feature_r2": round(float(feature_row["r2"]), 4),
                        "mae_improvement_vs_last_hour_pct": round(
                            100.0 * (float(last["mae"]) - float(best_row["mae"])) / float(last["mae"]), 4
                        ),
                    }
                )

        for target, group in rolling.groupby("target"):
            row = group.sort_values("mae_mean").iloc[0]
            rolling_best.append(
                {
                    "horizon_hours": PRIMARY_HORIZON,
                    "track": track,
                    "target": target,
                    "model": row["model"],
                    "folds": int(row["folds"]),
                    "mae_mean": round(float(row["mae_mean"]), 4),
                    "mae_std": round(float(row["mae_std"]), 4),
                    "r2_mean": round(float(row["r2_mean"]), 4),
                    "wape_pct_mean": round(float(row["wape_pct_mean"]), 4),
                    "mase_mean": round(float(row["mase_mean"]), 4),
                }
            )

    save_csv("best_overall_metrics.csv", all_best)
    save_csv("best_feature_model_metrics.csv", feature_best)
    save_csv("persistence_comparison.csv", persistence)
    save_csv("rolling_validation_best.csv", rolling_best)

    for source, destination in [
        ("best_model_metrics_by_horizon.csv", "best_model_metrics_by_horizon.csv"),
        ("best_feature_model_metrics_by_horizon.csv", "best_feature_model_metrics_by_horizon.csv"),
        ("persistence_comparison_by_horizon.csv", "persistence_comparison_by_horizon.csv"),
        ("rolling_validation_summary_by_horizon.csv", "rolling_validation_summary_by_horizon.csv"),
    ]:
        path = SUMMARY / source
        if path.exists():
            save_csv(destination, round_numeric(pd.read_csv(path)))


def build_best_model_by_target_horizon_table() -> None:
    stale_outputs = [
        TABLE_DIR / "all_model_target_horizon_metrics.csv",
        TABLE_DIR / "all_model_target_horizon_metrics.tex",
    ]
    for path in stale_outputs:
        path.unlink(missing_ok=True)

    rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        for track in TRACKS:
            path = analysis_dir(track, horizon) / "best_model_metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "horizon_hours", horizon)
            frame.insert(1, "track", track)
            rows.append(frame)

    if not rows:
        return

    table = pd.concat(rows, ignore_index=True)
    columns = [
        "horizon_hours",
        "track",
        "target",
        "model",
        "model_family",
        "n_train",
        "n_test",
        "mae",
        "rmse",
        "r2",
        "smape_pct",
        "wape_pct",
        "mase",
        "fit_seconds",
        "inference_ms_per_row",
    ]
    table = table[columns].sort_values(["horizon_hours", "track", "target"]).reset_index(drop=True)
    save_csv("best_model_by_target_horizon_metrics.csv", round_numeric(table, digits=4))

    track_labels = {"flow": "Flow", "nationality": "Nat.", "age": "Age"}
    target_labels = {
        "ingressi_borgo": "Entr.",
        "uscite_borgo": "Exit",
        "tim_Ni_mean15": "Ita.",
        "tim_Ns_mean15": "For.",
        "tim_F1_mean15": "F1",
        "tim_F2_mean15": "F2",
        "tim_F3_mean15": "F3",
        "tim_F4_mean15": "F4",
        "tim_F5_mean15": "F5",
        "tim_F6_mean15": "F6",
    }
    model_labels = {
        "dummy_median": "Med.",
        "last_hour": "LH",
        "rolling_mean_168h": "RM168",
        "ridge": "Ridge",
        "log1p_ridge": "log-Ridge",
        "two_stage_ridge": "2S-Ridge",
        "poisson": "Pois.",
        "hist_gradient_boosting": "HGB",
        "random_forest": "RF",
        "extra_trees": "ET",
        "xgboost": "XGB",
        "lightgbm": "LGBM",
    }

    display_rows = []
    for _, row in table.iterrows():
        display_rows.append(
            [
                f"{int(row['horizon_hours'])}h",
                compact_label(row["track"], track_labels),
                compact_label(row["target"], target_labels),
                compact_label(row["model"], model_labels),
                latex_num(row["mae"], 2),
                latex_num(row["r2"], 2),
            ]
        )

    split = (len(display_rows) + 1) // 2
    left_rows = display_rows[:split]
    right_rows = display_rows[split:]
    empty = ["", "", "", "", "", ""]

    tex_path = TABLE_DIR / "best_model_by_target_horizon_metrics.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[!t]\n")
        handle.write("\\setlength{\\abovecaptionskip}{1pt}\n")
        handle.write("\\setlength{\\belowcaptionskip}{2pt}\n")
        handle.write(
            "\\caption{Lowest-MAE final-split model for each target and forecast horizon. "
            "Abbrev.: F1--F6 age bands; LH last hour; RM168 weekly rolling mean; ET/RF/XGB/HGB/LGBM are tree and boosting models.}\n"
        )
        handle.write("\\label{tab:all-best-models}\n")
        handle.write("\\centering\n")
        handle.write("\\tiny\n")
        handle.write("\\setlength{\\tabcolsep}{1.7pt}\n")
        handle.write("\\renewcommand{\\arraystretch}{0.72}\n")
        handle.write("\\begin{tabular}{@{}rlllrr@{\\hspace{0.8em}}rlllrr@{}}\n")
        handle.write("\\toprule\n")
        handle.write(
            "H & Track & Target & Model & MAE & $R^2$ & H & Track & Target & Model & MAE & $R^2$ \\\\\n"
        )
        handle.write("\\midrule\n")
        for index in range(split):
            left = left_rows[index]
            right = right_rows[index] if index < len(right_rows) else empty
            handle.write(" & ".join(left + right) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\vspace{-1.1em}\n")
        handle.write("\\end{table}\n")


def build_top_k_table() -> None:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for track in TRACKS:
            path = analysis_dir(track, horizon) / "top_k_auto_selection.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            selected = frame[frame["selected"].astype(bool)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            rows.append(
                {
                    "horizon_hours": horizon,
                    "track": track,
                    "selected_top_k": int(row["top_k"]),
                    "validation_scope": row["validation_scope"],
                    "targets": int(row["targets"]),
                    "mean_best_mae": round(float(row["mean_best_mae"]), 4),
                    "mean_best_r2": round(float(row["mean_best_r2"]), 4),
                }
            )
    save_csv("top_k_selection_by_horizon.csv", rows)


def build_primary_model_leaderboard() -> None:
    rows: list[dict[str, object]] = []
    model_order = [
        "extra_trees",
        "random_forest",
        "xgboost",
        "lightgbm",
        "hist_gradient_boosting",
        "last_hour",
    ]
    flow_metrics = pd.read_csv(analysis_dir("flow") / "model_metrics.csv")
    flow_metrics = flow_metrics[flow_metrics["model"].isin(model_order)].copy()
    order_map = {model: rank for rank, model in enumerate(model_order)}
    flow_metrics["model_order"] = flow_metrics["model"].map(order_map)
    for target, group in flow_metrics.groupby("target"):
        group = group.sort_values(["mae", "model_order"])
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            rows.append(
                {
                    "horizon_hours": PRIMARY_HORIZON,
                    "track": "flow",
                    "target": target,
                    "rank_by_mae": rank,
                    "model": row["model"],
                    "model_family": row["model_family"],
                    "mae": round(float(row["mae"]), 4),
                    "r2": round(float(row["r2"]), 4),
                    "wape_pct": round(float(row["wape_pct"]), 4),
                    "fit_seconds": round(float(row["fit_seconds"]), 4),
                    "inference_ms_per_row": round(float(row["inference_ms_per_row"]), 4),
                }
            )
    save_csv("primary_flow_model_leaderboard.csv", rows)


def build_ablation_tables() -> None:
    rows: list[dict[str, object]] = []
    group_counts: list[dict[str, object]] = []
    group_count_pivot: list[dict[str, object]] = []
    top_importance: list[dict[str, object]] = []

    for track in TRACKS:
        base = analysis_dir(track)
        ablation = pd.read_csv(base / "ablation_study.csv")
        loo = ablation[ablation["ablation_type"].eq("leave_one_group_out")].copy()
        for _, row in loo.iterrows():
            rows.append(
                {
                    "horizon_hours": PRIMARY_HORIZON,
                    "track": track,
                    "target": row["target"],
                    "model": row["model"],
                    "removed_group": row["ablated_group"],
                    "n_features_removed": int(row["n_features_removed"]),
                    "mae_full": round(float(row["mae_full"]), 4),
                    "mae_ablated": round(float(row["mae_ablated"]), 4),
                    "delta_mae": round(float(row["delta_mae"]), 4),
                    "delta_wape_pct": round(float(row["delta_wape_pct"]), 4),
                }
            )

        feature_groups = pd.read_csv(base / "feature_groups.csv")
        counts = feature_groups["feature_group"].value_counts().reset_index()
        counts.columns = ["feature_group", "selected_features"]
        count_map = counts.set_index("feature_group")["selected_features"].to_dict()
        group_count_pivot.append(
            {
                "horizon_hours": PRIMARY_HORIZON,
                "track": track,
                "calendar": int(count_map.get("calendar", 0)),
                "events": int(count_map.get("events", 0)),
                "meteo": int(count_map.get("meteo", 0)),
                "pedoni": int(count_map.get("pedoni", 0)),
                "target_history": int(count_map.get("target_history", 0)),
                "tim": int(count_map.get("tim", 0)),
                "total_selected": int(len(feature_groups)),
            }
        )
        for _, row in counts.iterrows():
            group_counts.append(
                {
                    "horizon_hours": PRIMARY_HORIZON,
                    "track": track,
                    "feature_group": row["feature_group"],
                    "selected_features": int(row["selected_features"]),
                }
            )

        shap = pd.read_csv(base / "shap_importance.csv")
        for _, row in shap.sort_values("mean_abs_shap", ascending=False).head(10).iterrows():
            top_importance.append(
                {
                    "horizon_hours": PRIMARY_HORIZON,
                    "track": track,
                    "target": row["target"],
                    "model": row["model"],
                    "feature": row["feature"],
                    "feature_group": row["feature_group"],
                    "mean_abs_shap": round(float(row["mean_abs_shap"]), 4),
                }
            )

    save_csv("ablation_leave_one_group_out.csv", rows)
    save_csv("selected_feature_group_counts.csv", group_counts)
    save_csv("selected_feature_groups_one_hour.csv", group_count_pivot)
    save_csv("top_shap_features.csv", top_importance)


def build_single_feature_ablation_table() -> None:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    one_hour_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for track in TRACKS:
            path = analysis_dir(track, horizon) / "single_feature_ablation.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            strongest = frame.sort_values("delta_mae", ascending=False).iloc[0]
            summary_rows.append(
                {
                    "horizon_hours": horizon,
                    "track": track,
                    "target": strongest["target"],
                    "model": strongest["model"],
                    "ablated_feature": strongest["ablated_feature"],
                    "feature_group": strongest["feature_group"],
                    "delta_mae": round(float(strongest["delta_mae"]), 4),
                    "delta_wape_pct": round(float(strongest["delta_wape_pct"]), 4),
                }
            )
            for target, target_frame in frame.groupby("target"):
                ranked = target_frame.sort_values("delta_mae", ascending=False).head(5)
                for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                    rows.append(
                        {
                            "horizon_hours": horizon,
                            "track": track,
                            "target": target,
                            "rank": rank,
                            "model": row["model"],
                            "ablated_feature": row["ablated_feature"],
                            "feature_group": row["feature_group"],
                            "delta_mae": round(float(row["delta_mae"]), 4),
                            "delta_wape_pct": round(float(row["delta_wape_pct"]), 4),
                        }
                    )
                    if horizon == PRIMARY_HORIZON and rank == 1:
                        one_hour_rows.append(
                            {
                                "horizon_hours": horizon,
                                "track": track,
                                "target": target,
                                "model": row["model"],
                                "ablated_feature": row["ablated_feature"],
                                "feature_group": row["feature_group"],
                                "delta_mae": round(float(row["delta_mae"]), 4),
                                "delta_wape_pct": round(float(row["delta_wape_pct"]), 4),
                            }
                        )
    save_csv(
        "single_feature_ablation_top.csv",
        rows
        if rows
        else [
            {
                "horizon_hours": "",
                "track": "",
                "target": "",
                "rank": "",
                "model": "",
                "ablated_feature": "",
                "feature_group": "",
                "delta_mae": "",
                "delta_wape_pct": "",
            }
        ],
    )
    save_csv("single_feature_ablation_summary.csv", summary_rows)
    save_csv("single_feature_ablation_one_hour.csv", one_hour_rows)


def build_event_tables() -> None:
    selected_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    shap_rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        for track in TRACKS:
            base = analysis_dir(track, horizon)
            feature_groups = pd.read_csv(base / "feature_groups.csv")
            event_features = int(feature_groups["feature_group"].eq("events").sum())
            selected_rows.append(
                {
                    "horizon_hours": horizon,
                    "track": track,
                    "selected_features": len(feature_groups),
                    "event_features": event_features,
                    "event_share_pct": round(100.0 * event_features / len(feature_groups), 4),
                }
            )

            ablation = pd.read_csv(base / "ablation_study.csv")
            loo = ablation[ablation["ablation_type"].eq("leave_one_group_out")].copy()
            if not loo.empty:
                grouped = (
                    loo.groupby("ablated_group")
                    .agg(
                        mean_delta_mae=("delta_mae", "mean"),
                        median_delta_mae=("delta_mae", "median"),
                        targets=("target", "nunique"),
                        rows=("target", "size"),
                    )
                    .reset_index()
                )
                if "events" in set(grouped["ablated_group"]):
                    grouped["rank_mean_delta"] = grouped["mean_delta_mae"].rank(method="min", ascending=False).astype(int)
                    event_row = grouped[grouped["ablated_group"].eq("events")].iloc[0]
                    ablation_rows.append(
                        {
                            "horizon_hours": horizon,
                            "track": track,
                            "targets": int(event_row["targets"]),
                            "event_rows": int(event_row["rows"]),
                            "mean_delta_mae": round(float(event_row["mean_delta_mae"]), 4),
                            "median_delta_mae": round(float(event_row["median_delta_mae"]), 4),
                            "rank_mean_delta": int(event_row["rank_mean_delta"]),
                            "n_groups": len(grouped),
                        }
                    )

            shap_path = base / "shap_importance.csv"
            if shap_path.exists():
                shap = pd.read_csv(shap_path)
                for target, target_frame in shap.groupby("target"):
                    ranked = target_frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                    ranked["rank"] = range(1, len(ranked) + 1)
                    events = ranked[ranked["feature_group"].eq("events")]
                    if events.empty:
                        continue
                    top_event = events.iloc[0]
                    top = ranked.iloc[0]
                    shap_rows.append(
                        {
                            "horizon_hours": horizon,
                            "track": track,
                            "target": target,
                            "best_event_feature": top_event["feature"],
                            "best_event_rank": int(top_event["rank"]),
                            "best_event_mean_abs_shap": round(float(top_event["mean_abs_shap"]), 4),
                            "top_feature": top["feature"],
                            "top_feature_group": top["feature_group"],
                        }
                    )

    save_csv("event_selected_feature_summary.csv", selected_rows)
    save_csv("event_ablation_summary.csv", ablation_rows)
    save_csv("event_top_shap_features.csv", shap_rows)


def copy_figure(source_without_suffix: Path, destination_stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        source = source_without_suffix.with_suffix(suffix)
        if source.exists():
            shutil.copy2(source, FIGURE_DIR / f"{destination_stem}{suffix}")


def save_plot(fig: plt.Figure, destination_stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(FIGURE_DIR / f"{destination_stem}{suffix}", bbox_inches="tight", pad_inches=0.04, dpi=300)


def build_paper_ready_figures() -> None:
    joined_path = PREPROCESS / "dozza_joined_hourly_inner_with_events.csv"
    if not joined_path.exists():
        joined_path = PREPROCESS / "dozza_joined_hourly_inner.csv"
    test_predictions = pd.read_csv(analysis_dir("flow") / "test_predictions.csv", parse_dates=["timestamp"])
    test_start = test_predictions["timestamp"].min()
    frame = pd.read_csv(
        joined_path,
        usecols=["timestamp", "ingressi_borgo", "uscite_borgo"],
        parse_dates=["timestamp"],
    )
    train = frame[frame["timestamp"].lt(test_start)]
    test = frame[frame["timestamp"].ge(test_start)]
    data = [
        train["ingressi_borgo"].dropna(),
        test["ingressi_borgo"].dropna(),
        train["uscite_borgo"].dropna(),
        test["uscite_borgo"].dropna(),
    ]

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    box = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    colors = ["#8fb7d9", "#276fbf", "#e8b47a", "#c95f1a"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Entrances\ntrain", "Entrances\ntest", "Exits\ntrain", "Exits\ntest"])
    ax.set_ylabel("Hourly count")
    ax.set_title("Pedestrian-flow target distributions")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "flow_target_distribution_paper")
    plt.close(fig)


def build_figures() -> None:
    for track in TRACKS:
        copy_figure(analysis_dir(track) / "model_metric_r2", f"{track}_model_metric_r2")
    copy_figure(analysis_dir("flow") / "selected_feature_groups", "flow_selected_feature_groups")
    copy_figure(analysis_dir("flow") / "ablation_delta_mae", "flow_ablation_delta_mae")
    copy_figure(SUMMARY / "best_model_r2_by_horizon_flow", "best_model_r2_by_horizon_flow")
    copy_figure(SUMMARY / "feature_gain_vs_last_hour_by_horizon_flow", "feature_gain_vs_last_hour_by_horizon_flow")
    copy_figure(SUMMARY / "best_feature_model_mae_by_horizon_nationality", "best_feature_model_mae_by_horizon_nationality")
    copy_figure(SUMMARY / "best_feature_model_mae_by_horizon_age", "best_feature_model_mae_by_horizon_age")


def main() -> None:
    build_dataset_summary()
    build_dataset_profile()
    build_tim_dictionary()
    build_experimental_setup()
    build_metric_tables()
    build_best_model_by_target_horizon_table()
    build_top_k_table()
    build_primary_model_leaderboard()
    build_ablation_tables()
    build_single_feature_ablation_table()
    build_event_tables()
    build_paper_ready_figures()
    build_figures()
    print(f"[OK] Wrote paper tables to {TABLE_DIR}")
    print(f"[OK] Copied paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
