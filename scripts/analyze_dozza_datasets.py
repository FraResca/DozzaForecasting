#!/usr/bin/env python3
"""Analisi e join dei dati pedonali Dozza con le presenze TIM.

Lo script:
- legge il foglio orario dei contatori pedonali;
- aggrega i CSV TIM da 15 minuti a ora senza caricare tutto in memoria;
- crea il join orario su timestamp;
- produce report e CSV utili per la fase successiva di modellazione.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dozza_modeling.plotting import configure_paper_plots, save_figure

configure_paper_plots()


TIM_NUMERIC_COLUMNS = [
    "P",
    "Ni",
    "Ns",
    "Tc",
    "Tb",
    "Gm",
    "Gf",
    "Vp",
    "Vr",
    "Vi",
    "Ve",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
]

ENTRY_CAMERAS = ["Arcoribellino", "Piazzarocca Ingresso"]
EXIT_CAMERAS = ["Arcoribellino", "Piazzarocca Uscita"]
TARGET_COLUMNS = ["ingressi_borgo", "uscite_borgo"]
CALENDAR_FEATURE_COLUMNS = ["year", "month", "day", "hour", "dayofweek", "is_weekend"]


def slug(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def fmt_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_Nessuna riga._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    view = view.reset_index(drop=True)
    columns = [str(col) for col in view.columns]
    rows = [[fmt_value(value) for value in row] for row in view.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    suffix = ""
    if max_rows is not None and len(df) > max_rows:
        suffix = f"\n\n_Mostrate {max_rows} di {len(df)} righe._"
    return "\n".join([header, sep, *body]) + suffix


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "colonna": col,
                "dtype": str(df[col].dtype),
                "mancanti": missing,
                "mancanti_pct": (missing / total * 100) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["mancanti", "colonna"], ascending=[False, True])


def time_value_to_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M:%S")
    if isinstance(value, dt.datetime):
        return value.time().strftime("%H:%M:%S")
    if isinstance(value, dt.time):
        return value.strftime("%H:%M:%S")
    text = str(value).strip()
    if text.startswith("0 days "):
        text = text.replace("0 days ", "", 1)
    return text


def combine_date_and_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    times = pd.to_timedelta(time_series.map(time_value_to_string), errors="coerce")
    return (dates + times).dt.floor("h")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    return out


def read_pedestrians(excel_path: Path, sheet_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)
    required = {"data", "ora", "telecamera", "entra", "uscita"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Colonne mancanti nel foglio pedoni: {missing}")

    raw = raw.copy()
    raw["timestamp"] = combine_date_and_time(raw["data"], raw["ora"])
    raw["telecamera_slug"] = raw["telecamera"].map(slug)
    raw["entra"] = pd.to_numeric(raw["entra"], errors="coerce")
    raw["uscita"] = pd.to_numeric(raw["uscita"], errors="coerce")

    invalid_timestamp = int(raw["timestamp"].isna().sum())
    valid = raw.dropna(subset=["timestamp"]).copy()

    values = valid.pivot_table(
        index="timestamp",
        columns="telecamera_slug",
        values=["entra", "uscita"],
        aggfunc="sum",
        fill_value=0,
    )
    values.columns = [f"{measure}_{camera}" for measure, camera in values.columns]

    observed = valid.assign(observed=1).pivot_table(
        index="timestamp",
        columns="telecamera_slug",
        values="observed",
        aggfunc="max",
        fill_value=0,
    )
    observed.columns = [f"obs_{camera}" for camera in observed.columns]

    hourly = pd.concat([values, observed], axis=1).reset_index()
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)

    entry_slugs = [slug(camera) for camera in ENTRY_CAMERAS]
    exit_slugs = [slug(camera) for camera in EXIT_CAMERAS]
    entry_cols = [f"entra_{camera}" for camera in entry_slugs if f"entra_{camera}" in hourly]
    exit_cols = [f"uscita_{camera}" for camera in exit_slugs if f"uscita_{camera}" in hourly]
    all_entry_cols = [col for col in hourly.columns if col.startswith("entra_")]
    all_exit_cols = [col for col in hourly.columns if col.startswith("uscita_")]

    hourly["ingressi_borgo"] = hourly[entry_cols].sum(axis=1) if entry_cols else np.nan
    hourly["uscite_borgo"] = hourly[exit_cols].sum(axis=1) if exit_cols else np.nan
    hourly["entra_all_cameras"] = hourly[all_entry_cols].sum(axis=1) if all_entry_cols else np.nan
    hourly["uscita_all_cameras"] = hourly[all_exit_cols].sum(axis=1) if all_exit_cols else np.nan
    hourly["saldo_ingressi_uscite"] = hourly["ingressi_borgo"] - hourly["uscite_borgo"]

    target_ingressi_complete = pd.Series(True, index=hourly.index)
    target_uscite_complete = pd.Series(True, index=hourly.index)
    for camera in entry_slugs:
        col = f"obs_{camera}"
        target_ingressi_complete &= hourly[col].gt(0) if col in hourly else False
    for camera in exit_slugs:
        col = f"obs_{camera}"
        target_uscite_complete &= hourly[col].gt(0) if col in hourly else False
    hourly["target_ingressi_complete"] = target_ingressi_complete.astype(int)
    hourly["target_uscite_complete"] = target_uscite_complete.astype(int)
    hourly["target_complete"] = (
        hourly["target_ingressi_complete"].eq(1) & hourly["target_uscite_complete"].eq(1)
    ).astype(int)

    hourly = add_calendar_features(hourly)

    camera_stats = (
        valid.groupby("telecamera")
        .agg(
            righe=("telecamera", "size"),
            ore_uniche=("timestamp", "nunique"),
            data_min=("timestamp", "min"),
            data_max=("timestamp", "max"),
            entra_totale=("entra", "sum"),
            uscita_totale=("uscita", "sum"),
        )
        .reset_index()
    )

    full_hours = pd.date_range(hourly["timestamp"].min(), hourly["timestamp"].max(), freq="h")
    missing_hour_slots = full_hours.difference(pd.DatetimeIndex(hourly["timestamp"]))

    stats = {
        "raw_rows": int(len(raw)),
        "valid_timestamp_rows": int(len(valid)),
        "invalid_timestamp_rows": invalid_timestamp,
        "hourly_rows": int(len(hourly)),
        "timestamp_min": str(hourly["timestamp"].min()),
        "timestamp_max": str(hourly["timestamp"].max()),
        "full_hour_slots_between_min_max": int(len(full_hours)),
        "missing_hour_slots_between_min_max": int(len(missing_hour_slots)),
        "target_complete_rows": int(hourly["target_complete"].sum()),
        "target_incomplete_rows": int(len(hourly) - hourly["target_complete"].sum()),
        "camera_stats": camera_stats,
        "raw_missing": missing_summary(raw.drop(columns=["telecamera_slug"], errors="ignore")),
        "hourly_missing": missing_summary(hourly),
    }
    return hourly, stats


def discover_excel(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Nessun file .xlsx trovato in {data_dir}")
    if len(files) > 1:
        print(f"[INFO] Trovati piu' Excel, uso il primo: {files[0]}", flush=True)
    return files[0]


def convert_tim_time_to_analysis_tz(
    event_time: pd.Series,
    source_tz: str | None,
    analysis_tz: str | None,
) -> pd.Series:
    parsed = pd.to_datetime(event_time, errors="coerce")
    if not source_tz or source_tz.lower() in {"none", "naive"}:
        return parsed
    if not analysis_tz:
        analysis_tz = source_tz
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(source_tz)
    else:
        localized = parsed
    return localized.dt.tz_convert(analysis_tz).dt.tz_localize(None)


def convert_weather_time_to_analysis_tz(
    timestamp: pd.Series,
    source_tz: str | None,
    analysis_tz: str | None,
) -> pd.Series:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if not source_tz or source_tz.lower() in {"none", "naive", "local"}:
        return parsed
    if not analysis_tz:
        analysis_tz = source_tz
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(source_tz)
    else:
        localized = parsed
    return localized.dt.tz_convert(analysis_tz).dt.tz_localize(None)


def to_numeric_weather(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def aggregate_tim(
    tim_dir: Path,
    output_dir: Path,
    chunk_size: int,
    limit_files: int | None,
    rebuild: bool,
    tim_source_tz: str | None,
    analysis_tz: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    csv_files = sorted(tim_dir.glob("*/*.csv.gz"))
    if limit_files is not None:
        csv_files = csv_files[:limit_files]
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV TIM .csv.gz trovato in {tim_dir}")

    cache_15min = output_dir / "tim_15min_area_totals.csv"
    cache_meta = output_dir / "tim_aggregation_metadata.json"
    if cache_15min.exists() and cache_meta.exists() and not rebuild and limit_files is None:
        with cache_meta.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        cache_matches_timezone = (
            metadata.get("tim_source_timezone") == tim_source_tz
            and metadata.get("analysis_timezone") == analysis_tz
        )
        if cache_matches_timezone:
            print(f"[TIM] Uso cache esistente: {cache_15min}", flush=True)
            tim_15min = pd.read_csv(cache_15min, parse_dates=["timestamp_15min"])
            return tim_15min, tim_15min_to_hourly(tim_15min), metadata
        print("[TIM] Cache ignorata: timezone richieste diverse dalla cache.", flush=True)

    selected_columns = ["event_time", *TIM_NUMERIC_COLUMNS]
    dtype = {col: "float64" for col in TIM_NUMERIC_COLUMNS}
    partials: list[pd.DataFrame] = []
    missing_counts = {col: 0 for col in selected_columns}
    raw_rows = 0
    invalid_event_time = 0
    file_stats = []

    for idx, path in enumerate(csv_files, start=1):
        print(f"[TIM] {idx}/{len(csv_files)} {path}", flush=True)
        file_rows = 0
        file_invalid_time = 0
        file_partials = 0
        for chunk in pd.read_csv(
            path,
            sep=";",
            compression="gzip",
            usecols=selected_columns,
            dtype=dtype,
            chunksize=chunk_size,
        ):
            raw_rows += len(chunk)
            file_rows += len(chunk)
            chunk_missing = chunk.isna().sum()
            for col in selected_columns:
                missing_counts[col] += int(chunk_missing.get(col, 0))

            event_time = convert_tim_time_to_analysis_tz(
                chunk["event_time"],
                source_tz=tim_source_tz,
                analysis_tz=analysis_tz,
            )
            valid_mask = event_time.notna()
            invalid_count = int((~valid_mask).sum())
            invalid_event_time += invalid_count
            file_invalid_time += invalid_count
            if not valid_mask.any():
                continue

            numeric = chunk.loc[valid_mask, TIM_NUMERIC_COLUMNS].copy()
            numeric["timestamp_15min"] = event_time.loc[valid_mask].dt.floor("15min").to_numpy()
            numeric["tile_rows"] = 1.0
            grouped = numeric.groupby("timestamp_15min", sort=False)[
                [*TIM_NUMERIC_COLUMNS, "tile_rows"]
            ].sum()
            partials.append(grouped)
            file_partials += len(grouped)

        file_stats.append(
            {
                "file": str(path),
                "rows": int(file_rows),
                "invalid_event_time_rows": int(file_invalid_time),
                "partial_15min_groups": int(file_partials),
            }
        )

    if not partials:
        raise ValueError("Nessun dato TIM valido dopo il parsing di event_time.")

    tim_15min = pd.concat(partials).groupby(level=0).sum().sort_index().reset_index()
    tim_15min = tim_15min.rename(columns={"index": "timestamp_15min"})
    tim_15min.to_csv(cache_15min, index=False)

    metadata = {
        "files_processed": len(csv_files),
        "source_months": sorted({path.parent.name for path in csv_files}),
        "raw_rows": int(raw_rows),
        "invalid_event_time_rows": int(invalid_event_time),
        "missing_counts_selected_columns": missing_counts,
        "timestamp_15min_min": str(tim_15min["timestamp_15min"].min()),
        "timestamp_15min_max": str(tim_15min["timestamp_15min"].max()),
        "tim_source_timezone": tim_source_tz,
        "analysis_timezone": analysis_tz,
        "unique_15min_timestamps": int(len(tim_15min)),
        "file_stats": file_stats,
    }
    with cache_meta.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return tim_15min, tim_15min_to_hourly(tim_15min), metadata


def tim_15min_to_hourly(tim_15min: pd.DataFrame) -> pd.DataFrame:
    work = tim_15min.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp_15min"]).dt.floor("h")
    grouped = work.groupby("timestamp", sort=True)

    mean_features = grouped[TIM_NUMERIC_COLUMNS].mean().add_prefix("tim_").add_suffix("_mean15")
    sum_features = grouped[TIM_NUMERIC_COLUMNS].sum().add_prefix("tim_").add_suffix("_sum15")
    max_features = grouped[TIM_NUMERIC_COLUMNS].max().add_prefix("tim_").add_suffix("_max15")
    coverage = grouped.agg(
        tim_15min_obs=("timestamp_15min", "size"),
        tim_tile_rows_sum=("tile_rows", "sum"),
        tim_tile_rows_mean15=("tile_rows", "mean"),
    )
    hourly = pd.concat([coverage, mean_features, sum_features, max_features], axis=1)
    return hourly.reset_index()


def load_weather(weather_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(weather_csv)
    lower = {col.lower(): col for col in raw.columns}
    timestamp_col = None
    for candidate in ["timestamp", "datetime", "date_time", "data_ora", "dataora"]:
        if candidate in lower:
            timestamp_col = lower[candidate]
            break

    work = raw.copy()
    if timestamp_col:
        work["timestamp"] = pd.to_datetime(work[timestamp_col], errors="coerce").dt.floor("h")
    elif "data" in lower and "ora" in lower:
        work["timestamp"] = combine_date_and_time(work[lower["data"]], work[lower["ora"]])
    else:
        raise ValueError(
            "CSV meteo non riconosciuto: serve una colonna timestamp/datetime/data_ora "
            "oppure la coppia data + ora."
        )

    numeric_cols = [
        col
        for col in work.select_dtypes(include="number").columns
        if col != "timestamp" and col not in {"year", "month", "day", "hour"}
    ]
    if not numeric_cols:
        raise ValueError("CSV meteo senza colonne numeriche utilizzabili.")

    hourly = work.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False)[numeric_cols].mean()
    rename = {col: col if col.startswith("meteo_") else f"meteo_{slug(col)}" for col in numeric_cols}
    hourly = hourly.rename(columns=rename)
    stats = {
        "raw_rows": int(len(raw)),
        "hourly_rows": int(len(hourly)),
        "timestamp_min": str(hourly["timestamp"].min()),
        "timestamp_max": str(hourly["timestamp"].max()),
        "raw_missing": missing_summary(raw),
        "columns": list(hourly.columns),
    }
    return hourly, stats


def discover_local_erg5_hourly_files(weather_dir: Path) -> list[Path]:
    return sorted(weather_dir.glob("**/*_h.csv"))


def cell_code_from_path(path: Path) -> str:
    for value in [path.stem, path.parent.name]:
        match = re.search(r"\d{5}", value)
        if match:
            return match.group(0)
    return slug(path.parent.name or path.stem)


def load_local_erg5_weather(
    weather_dir: Path,
    source_tz: str | None,
    analysis_tz: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = discover_local_erg5_hourly_files(weather_dir)
    if not files:
        raise FileNotFoundError(f"Nessun file ERG5 orario *_h.csv trovato in {weather_dir}")

    frames = []
    file_stats = []
    variables_by_cell: dict[str, list[str]] = {}
    raw_missing_parts = []

    for path in files:
        raw = pd.read_csv(path)
        lower = {col.lower(): col for col in raw.columns}
        timestamp_col = lower.get("pragatime") or lower.get("timestamp") or lower.get("datetime")
        if not timestamp_col:
            raise ValueError(f"File meteo ERG5 senza colonna PragaTime/timestamp: {path}")

        cell_code = cell_code_from_path(path)
        work = raw.copy()
        work["timestamp"] = convert_weather_time_to_analysis_tz(
            work[timestamp_col],
            source_tz=source_tz,
            analysis_tz=analysis_tz,
        ).dt.floor("h")

        numeric_cols = []
        for col in work.columns:
            if col in {timestamp_col, "timestamp"}:
                continue
            numeric = to_numeric_weather(work[col])
            if numeric.notna().any():
                work[col] = numeric
                numeric_cols.append(col)
        if not numeric_cols:
            raise ValueError(f"File meteo ERG5 senza colonne numeriche utilizzabili: {path}")

        variables_by_cell[cell_code] = [slug(col) for col in numeric_cols]
        hourly = work.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False)[numeric_cols].mean()
        hourly = hourly.rename(
            columns={
                col: f"meteo_erg5_cell{cell_code}_{slug(col)}"
                for col in numeric_cols
            }
        )
        frames.append(hourly)

        file_stats.append(
            {
                "file": str(path),
                "cell_code": cell_code,
                "raw_rows": int(len(raw)),
                "hourly_rows": int(len(hourly)),
                "timestamp_min": str(hourly["timestamp"].min()),
                "timestamp_max": str(hourly["timestamp"].max()),
                "invalid_timestamp_rows": int(work["timestamp"].isna().sum()),
                "duplicate_hour_rows": int(work["timestamp"].duplicated().sum()),
            }
        )
        missing = missing_summary(raw)
        missing["file"] = str(path)
        missing["cell_code"] = cell_code
        raw_missing_parts.append(missing)

    hourly = frames[0]
    for frame in frames[1:]:
        hourly = hourly.merge(frame, on="timestamp", how="outer", validate="one_to_one")
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)

    variable_names = sorted({var for variables in variables_by_cell.values() for var in variables})
    cell_codes = sorted(variables_by_cell)
    for variable in variable_names:
        cols = [
            f"meteo_erg5_cell{cell_code}_{variable}"
            for cell_code in cell_codes
            if f"meteo_erg5_cell{cell_code}_{variable}" in hourly.columns
        ]
        if not cols:
            continue
        values = hourly[cols]
        hourly[f"meteo_erg5_{variable}_mean"] = values.mean(axis=1)
        if len(cols) > 1:
            hourly[f"meteo_erg5_{variable}_min"] = values.min(axis=1)
            hourly[f"meteo_erg5_{variable}_max"] = values.max(axis=1)
            hourly[f"meteo_erg5_{variable}_std"] = values.std(axis=1, ddof=0)
        if "prec" in variable or "piogg" in variable or "rain" in variable:
            hourly[f"meteo_erg5_{variable}_is_positive"] = values.fillna(0).max(axis=1).gt(0).astype(int)

    stats = {
        "source": "local_erg5_dir",
        "weather_dir": str(weather_dir),
        "raw_rows": int(sum(item["raw_rows"] for item in file_stats)),
        "hourly_rows": int(len(hourly)),
        "timestamp_min": str(hourly["timestamp"].min()),
        "timestamp_max": str(hourly["timestamp"].max()),
        "source_timezone": source_tz,
        "analysis_timezone": analysis_tz,
        "cells": cell_codes,
        "files": file_stats,
        "raw_missing": pd.concat(raw_missing_parts, ignore_index=True)
        if raw_missing_parts
        else pd.DataFrame(),
        "columns": list(hourly.columns),
    }
    return hourly, stats


def build_join(
    pedestrians: pd.DataFrame,
    tim_hourly: pd.DataFrame,
    weather_hourly: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pedestrian_model = pedestrians[pedestrians["target_complete"].eq(1)].copy()
    left = pedestrian_model.merge(tim_hourly, on="timestamp", how="left", validate="one_to_one")
    inner = pedestrian_model.merge(tim_hourly, on="timestamp", how="inner", validate="one_to_one")

    if weather_hourly is not None:
        left = left.merge(weather_hourly, on="timestamp", how="left", validate="one_to_one")
        inner = inner.merge(weather_hourly, on="timestamp", how="left", validate="one_to_one")

    stats = {
        "pedestrian_hourly_rows": int(len(pedestrians)),
        "pedestrian_target_complete_rows": int(len(pedestrian_model)),
        "tim_hourly_rows": int(len(tim_hourly)),
        "joined_inner_rows": int(len(inner)),
        "joined_left_rows": int(len(left)),
        "pedestrian_rows_without_tim": int(left["tim_15min_obs"].isna().sum()),
        "tim_hours_not_in_complete_pedestrians": int(
            len(
                pd.DatetimeIndex(tim_hourly["timestamp"]).difference(
                    pd.DatetimeIndex(pedestrian_model["timestamp"])
                )
            )
        ),
    }
    return inner, left, stats


def model_feature_columns(joined: pd.DataFrame) -> list[str]:
    return [
        col
        for col in joined.columns
        if col.startswith("tim_")
        or col.startswith("meteo_")
        or col in CALENDAR_FEATURE_COLUMNS
    ]


def save_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    width_per_col: float = 0.32,
    height_per_row: float = 0.32,
) -> None:
    if matrix.empty:
        return

    n_rows, n_cols = matrix.shape
    fig_width = min(max(8.0, n_cols * width_per_col), 28.0)
    fig_height = min(max(5.0, n_rows * height_per_row), 28.0)
    label_size = 8 if max(n_rows, n_cols) <= 30 else 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        linewidths=0.0,
        cbar_kws={"label": "correlazione"},
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=label_size)
    ax.tick_params(axis="y", labelrotation=0, labelsize=label_size)
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)


def save_correlation_images(
    output_dir: Path,
    full_matrix_pearson: pd.DataFrame,
    full_matrix_spearman: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    targets: list[str],
    features: list[str],
) -> None:
    save_correlation_heatmap(
        full_matrix_pearson,
        output_dir / "correlation_matrix_pearson.png",
        "Matrice di correlazione Pearson - target + feature",
    )
    save_correlation_heatmap(
        full_matrix_spearman,
        output_dir / "correlation_matrix_spearman.png",
        "Matrice di correlazione Spearman - target + feature",
    )
    save_correlation_heatmap(
        feature_matrix,
        output_dir / "feature_correlation_matrix.png",
        "Matrice di correlazione Pearson - feature",
    )

    ordered_features = [feature for feature in features if feature in full_matrix_pearson.columns]
    ordered_targets = [target for target in targets if target in full_matrix_pearson.index]
    if ordered_targets and ordered_features:
        save_correlation_heatmap(
            full_matrix_pearson.loc[ordered_targets, ordered_features],
            output_dir / "target_feature_correlation_heatmap_pearson.png",
            "Correlazione Pearson target vs feature",
            width_per_col=0.34,
            height_per_row=0.7,
        )
    ordered_features = [feature for feature in features if feature in full_matrix_spearman.columns]
    ordered_targets = [target for target in targets if target in full_matrix_spearman.index]
    if ordered_targets and ordered_features:
        save_correlation_heatmap(
            full_matrix_spearman.loc[ordered_targets, ordered_features],
            output_dir / "target_feature_correlation_heatmap_spearman.png",
            "Correlazione Spearman target vs feature",
            width_per_col=0.34,
            height_per_row=0.7,
        )


def correlation_rows(
    df: pd.DataFrame,
    targets: list[str],
    features: list[str],
    method: str,
    lag_hours: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    available_cols = [col for col in [*targets, *features] if col in df.columns]
    numeric = df[available_cols].select_dtypes(include="number")
    available_targets = [target for target in targets if target in numeric.columns]
    available_features = [feature for feature in features if feature in numeric.columns]
    if numeric.empty or not available_targets or not available_features:
        return rows

    for target in targets:
        if target not in numeric:
            continue
        for feature in available_features:
            pair = numeric[[target, feature]].dropna()
            n = int(len(pair))
            if n < 3 or pair[target].nunique(dropna=True) < 2 or pair[feature].nunique(dropna=True) < 2:
                continue
            value = pair[target].corr(pair[feature], method=method)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "method": method,
                    "lag_hours": lag_hours,
                    "n": n,
                    "corr": float(value),
                    "abs_corr": float(abs(value)),
                }
            )
    return rows


def top_feature_pair_correlations(
    joined: pd.DataFrame,
    features: list[str],
    method: str,
    threshold: float,
) -> pd.DataFrame:
    numeric = joined[[col for col in features if col in joined.columns]].select_dtypes(include="number")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "method", "corr", "abs_corr"])

    corr = numeric.corr(method=method, numeric_only=True)
    rows = []
    columns = list(corr.columns)
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            if pd.isna(value) or abs(value) < threshold:
                continue
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "method": method,
                    "corr": float(value),
                    "abs_corr": float(abs(value)),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def analyze_correlations(
    joined: pd.DataFrame,
    output_dir: Path,
    lags: list[int],
    top_n: int,
    feature_pair_threshold: float,
) -> dict[str, pd.DataFrame]:
    targets = [target for target in TARGET_COLUMNS if target in joined.columns]
    features = model_feature_columns(joined)

    overall_rows = []
    for method in ["pearson", "spearman"]:
        overall_rows.extend(correlation_rows(joined, targets, features, method=method))
    overall = pd.DataFrame(overall_rows)
    if not overall.empty:
        overall = overall.sort_values(["target", "method", "abs_corr"], ascending=[True, True, False])

    laggable_features = [
        feature
        for feature in features
        if feature.startswith("tim_") or feature.startswith("meteo_")
    ]
    ordered = joined.sort_values("timestamp").copy()
    lag_rows = []
    for lag in sorted(set(lags)):
        lagged_features = ordered[["timestamp", *laggable_features]].copy()
        lagged_features["timestamp"] = lagged_features["timestamp"] + pd.to_timedelta(lag, unit="h")
        lagged = ordered[["timestamp", *targets]].merge(
            lagged_features,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        for method in ["pearson", "spearman"]:
            lag_rows.extend(correlation_rows(lagged, targets, laggable_features, method=method, lag_hours=lag))
    lagged_correlations = pd.DataFrame(lag_rows)
    if not lagged_correlations.empty:
        lagged_correlations = lagged_correlations.sort_values(
            ["target", "method", "abs_corr"],
            ascending=[True, True, False],
        )

    target_intercorrelation = pd.DataFrame()
    if len(targets) >= 2:
        target_intercorrelation = joined[targets].corr(method="pearson", numeric_only=True)

    full_matrix_cols = [col for col in [*targets, *features] if col in joined.columns]
    full_matrix_numeric = joined[full_matrix_cols].select_dtypes(include="number")
    full_matrix_pearson = full_matrix_numeric.corr(method="pearson", numeric_only=True)
    full_matrix_spearman = full_matrix_numeric.corr(method="spearman", numeric_only=True)

    feature_matrix_cols = [col for col in features if col in joined.columns]
    feature_matrix = joined[feature_matrix_cols].select_dtypes(include="number").corr(
        method="pearson",
        numeric_only=True,
    )
    feature_pairs = top_feature_pair_correlations(
        joined,
        features=features,
        method="pearson",
        threshold=feature_pair_threshold,
    )

    best_lag = pd.DataFrame()
    if not lagged_correlations.empty:
        best_lag = (
            lagged_correlations.sort_values("abs_corr", ascending=False)
            .groupby(["target", "feature", "method"], as_index=False)
            .head(1)
            .sort_values(["target", "method", "abs_corr"], ascending=[True, True, False])
        )

    top_overall = (
        overall.sort_values("abs_corr", ascending=False)
        .groupby(["target", "method"], as_index=False)
        .head(top_n)
        .sort_values(["target", "method", "abs_corr"], ascending=[True, True, False])
        if not overall.empty
        else overall
    )
    top_lagged = (
        lagged_correlations.sort_values("abs_corr", ascending=False)
        .groupby(["target", "method"], as_index=False)
        .head(top_n)
        .sort_values(["target", "method", "abs_corr"], ascending=[True, True, False])
        if not lagged_correlations.empty
        else lagged_correlations
    )

    outputs = {
        "overall": overall,
        "top_overall": top_overall,
        "lagged": lagged_correlations,
        "top_lagged": top_lagged,
        "best_lag": best_lag,
        "target_intercorrelation": target_intercorrelation,
        "full_matrix_pearson": full_matrix_pearson,
        "full_matrix_spearman": full_matrix_spearman,
        "feature_matrix": feature_matrix,
        "feature_pairs": feature_pairs,
    }
    outputs["overall"].to_csv(output_dir / "target_feature_correlations.csv", index=False)
    outputs["top_overall"].to_csv(output_dir / "target_feature_correlations_top.csv", index=False)
    outputs["lagged"].to_csv(output_dir / "target_feature_lag_correlations.csv", index=False)
    outputs["top_lagged"].to_csv(output_dir / "target_feature_lag_correlations_top.csv", index=False)
    outputs["best_lag"].to_csv(output_dir / "target_feature_best_lag_correlations.csv", index=False)
    outputs["target_intercorrelation"].to_csv(output_dir / "target_intercorrelation.csv")
    outputs["full_matrix_pearson"].to_csv(output_dir / "correlation_matrix_pearson.csv")
    outputs["full_matrix_spearman"].to_csv(output_dir / "correlation_matrix_spearman.csv")
    outputs["feature_matrix"].to_csv(output_dir / "feature_correlation_matrix.csv")
    outputs["feature_pairs"].to_csv(output_dir / "feature_pair_high_correlations.csv", index=False)
    save_correlation_images(
        output_dir=output_dir,
        full_matrix_pearson=outputs["full_matrix_pearson"],
        full_matrix_spearman=outputs["full_matrix_spearman"],
        feature_matrix=outputs["feature_matrix"],
        targets=targets,
        features=features,
    )
    return outputs


def write_report(
    output_dir: Path,
    excel_path: Path,
    tim_dir: Path,
    pedestrian_stats: dict[str, Any],
    tim_15min: pd.DataFrame,
    tim_hourly: pd.DataFrame,
    tim_metadata: dict[str, Any],
    join_stats: dict[str, Any],
    joined_inner: pd.DataFrame,
    joined_left: pd.DataFrame,
    correlations: dict[str, pd.DataFrame],
    correlation_lags: list[int],
    correlation_top_n: int,
    feature_pair_threshold: float,
    weather_stats: dict[str, Any] | None,
) -> str:
    tim_missing = pd.DataFrame(
        [
            {
                "colonna": col,
                "mancanti": value,
                "mancanti_pct": value / tim_metadata["raw_rows"] * 100
                if tim_metadata.get("raw_rows")
                else 0,
            }
            for col, value in tim_metadata.get("missing_counts_selected_columns", {}).items()
        ]
    ).sort_values(["mancanti", "colonna"], ascending=[False, True])

    hourly_missing_inner = missing_summary(joined_inner)
    hourly_missing_left = missing_summary(joined_left)
    output_paths = pd.DataFrame(
        [
            {"file": str(output_dir / "pedoni_hourly_targets.csv"), "contenuto": "Target pedonali orari"},
            {"file": str(output_dir / "tim_15min_area_totals.csv"), "contenuto": "TIM aggregato per quarto d'ora"},
            {"file": str(output_dir / "tim_hourly_features.csv"), "contenuto": "Feature TIM orarie"},
            {"file": str(output_dir / "weather_hourly_features.csv"), "contenuto": "Feature meteo orarie, se caricate"},
            {"file": str(output_dir / "dozza_joined_hourly_inner.csv"), "contenuto": "Join pedoni + TIM"},
            {"file": str(output_dir / "dozza_joined_hourly_left.csv"), "contenuto": "Pedoni completi con TIM opzionale"},
            {"file": str(output_dir / "missing_join_inner.csv"), "contenuto": "Mancanti nel join inner"},
            {"file": str(output_dir / "missing_join_left.csv"), "contenuto": "Mancanti nel join left"},
            {"file": str(output_dir / "target_feature_correlations.csv"), "contenuto": "Correlazioni target-feature complete"},
            {"file": str(output_dir / "target_feature_correlations_top.csv"), "contenuto": "Top correlazioni contemporanee"},
            {"file": str(output_dir / "target_feature_lag_correlations.csv"), "contenuto": "Correlazioni target-feature con lag"},
            {"file": str(output_dir / "target_feature_lag_correlations_top.csv"), "contenuto": "Top correlazioni con lag"},
            {"file": str(output_dir / "target_feature_best_lag_correlations.csv"), "contenuto": "Miglior lag per feature"},
            {"file": str(output_dir / "target_intercorrelation.csv"), "contenuto": "Correlazione tra target"},
            {"file": str(output_dir / "correlation_matrix_pearson.csv"), "contenuto": "Matrice Pearson completa target + feature"},
            {"file": str(output_dir / "correlation_matrix_spearman.csv"), "contenuto": "Matrice Spearman completa target + feature"},
            {"file": str(output_dir / "feature_correlation_matrix.csv"), "contenuto": "Matrice correlazione feature"},
            {"file": str(output_dir / "correlation_matrix_pearson.png"), "contenuto": "Immagine matrice Pearson completa"},
            {"file": str(output_dir / "correlation_matrix_spearman.png"), "contenuto": "Immagine matrice Spearman completa"},
            {"file": str(output_dir / "feature_correlation_matrix.png"), "contenuto": "Immagine matrice Pearson feature"},
            {"file": str(output_dir / "target_feature_correlation_heatmap_pearson.png"), "contenuto": "Immagine Pearson target vs feature"},
            {"file": str(output_dir / "target_feature_correlation_heatmap_spearman.png"), "contenuto": "Immagine Spearman target vs feature"},
            {"file": str(output_dir / "feature_pair_high_correlations.csv"), "contenuto": "Coppie feature molto correlate"},
        ]
    )
    top_overall = correlations["top_overall"]
    top_lagged = correlations["top_lagged"]
    best_lag = correlations["best_lag"]
    target_intercorrelation = correlations["target_intercorrelation"].reset_index()
    if not target_intercorrelation.empty:
        first_col = target_intercorrelation.columns[0]
        target_intercorrelation = target_intercorrelation.rename(columns={first_col: "target"})
    feature_pairs = correlations["feature_pairs"]

    report = f"""# Analisi dataset Dozza

## Input

- Excel pedoni: `{excel_path}`
- Directory TIM: `{tim_dir}`
- Mesi TIM elaborati: {", ".join(tim_metadata.get("source_months", []))}
- Timezone TIM sorgente: {tim_metadata.get("tim_source_timezone", "non specificata")}
- Timezone analisi/join: {tim_metadata.get("analysis_timezone", "non specificata")}
- Meteo: {"presente" if weather_stats else "non fornito nello script run corrente"}

## Pedoni orari

- Righe raw sheet orario: {pedestrian_stats["raw_rows"]}
- Righe con timestamp valido: {pedestrian_stats["valid_timestamp_rows"]}
- Timestamp orari aggregati: {pedestrian_stats["hourly_rows"]}
- Intervallo: {pedestrian_stats["timestamp_min"]} - {pedestrian_stats["timestamp_max"]}
- Slot orari teorici tra min e max: {pedestrian_stats["full_hour_slots_between_min_max"]}
- Slot orari mancanti tra min e max: {pedestrian_stats["missing_hour_slots_between_min_max"]}
- Righe con target ingressi+uscite completi: {pedestrian_stats["target_complete_rows"]}
- Righe con target incompleti: {pedestrian_stats["target_incomplete_rows"]}

Definizione target proposta:

- `ingressi_borgo` = `entra` di Arcoribellino + `entra` di Piazzarocca Ingresso
- `uscite_borgo` = `uscita` di Arcoribellino + `uscita` di Piazzarocca Uscita
- I target vengono marcati completi solo se le telecamere richieste sono presenti nello stesso timestamp orario.

### Copertura per telecamera

{markdown_table(pedestrian_stats["camera_stats"])}

### Mancanti pedoni raw

{markdown_table(pedestrian_stats["raw_missing"])}

## TIM

- File CSV elaborati: {tim_metadata["files_processed"]}
- Righe raw TIM lette: {tim_metadata["raw_rows"]}
- Righe con `event_time` non valido: {tim_metadata["invalid_event_time_rows"]}
- Timestamp TIM a 15 minuti: {len(tim_15min)}
- Timestamp TIM orari: {len(tim_hourly)}
- Intervallo TIM 15 minuti: {tim_metadata["timestamp_15min_min"]} - {tim_metadata["timestamp_15min_max"]}
- Ore TIM con meno di 4 campioni da 15 minuti: {int(tim_hourly["tim_15min_obs"].lt(4).sum())}

### Mancanti TIM nelle colonne lette

{markdown_table(tim_missing)}

## Join proposto

Granularita': oraria.

1. Pedoni: costruire `timestamp = data + ora`, normalizzato con `floor('h')`; aggregare per `timestamp`.
2. TIM: sommare le tile per ogni `event_time` da 15 minuti; poi aggregare a ora calcolando `mean15`, `sum15`, `max15` per ciascuna variabile TIM.
3. Join: `pedoni_hourly.target_complete == 1` INNER JOIN `tim_hourly_features` su `timestamp`.
4. Meteo: se disponibile un CSV meteo orario, join LEFT su `timestamp` dopo TIM.

## Dataset unito

- Esempi disponibili per training con TIM (`inner join`): {join_stats["joined_inner_rows"]}
- Righe pedonali complete considerate: {join_stats["pedestrian_target_complete_rows"]}
- Righe pedonali complete senza TIM (`left join`): {join_stats["pedestrian_rows_without_tim"]}
- Ore TIM non presenti nei pedoni completi: {join_stats["tim_hours_not_in_complete_pedestrians"]}
- Colonne del dataset inner: {len(joined_inner.columns)}

### Prime righe dataset unito

{markdown_table(joined_inner.head(10))}

### Mancanti nel join inner

{markdown_table(hourly_missing_inner[hourly_missing_inner["mancanti"].gt(0)].head(30))}

### Mancanti nel join left, colonne piu' critiche

{markdown_table(hourly_missing_left[hourly_missing_left["mancanti"].gt(0)].head(30))}

## Correlazioni esplorative

Le correlazioni sono solo diagnostica iniziale, non una valutazione predittiva. Pearson misura associazioni lineari; Spearman misura associazioni monotone ed e' meno sensibile alla scala. I lag sono calcolati come `target[t]` contro `feature[t-lag]` tramite join temporale.

Matrici complete salvate:

- `correlation_matrix_pearson.csv`: target + feature, correlazione Pearson.
- `correlation_matrix_spearman.csv`: target + feature, correlazione Spearman.
- `correlation_matrix_pearson.png/.pdf`: immagine della matrice Pearson completa.
- `correlation_matrix_spearman.png/.pdf`: immagine della matrice Spearman completa.
- `target_feature_correlation_heatmap_pearson.png/.pdf`: immagine piu' leggibile con target sulle righe e feature sulle colonne.
- `target_feature_correlation_heatmap_spearman.png/.pdf`: equivalente Spearman target vs feature.

### Target tra loro

{markdown_table(target_intercorrelation)}

### Top correlazioni contemporanee

Top {correlation_top_n} per target e metodo su calendario, TIM e meteo.

{markdown_table(top_overall, max_rows=80)}

### Top correlazioni con lag

Lag testati: {", ".join(str(lag) for lag in sorted(set(correlation_lags)))} ore.

{markdown_table(top_lagged, max_rows=80)}

### Miglior lag per feature

{markdown_table(best_lag, max_rows=80)}

### Feature molto correlate tra loro

Soglia assoluta Pearson: {feature_pair_threshold}.

{markdown_table(feature_pairs, max_rows=40)}

## Meteo

"""
    if weather_stats:
        report += f"""- Righe raw meteo: {weather_stats["raw_rows"]}
- Righe meteo orarie: {weather_stats["hourly_rows"]}
- Intervallo meteo: {weather_stats["timestamp_min"]} - {weather_stats["timestamp_max"]}
- Fonte meteo: {weather_stats.get("source", "csv")}
- Timezone meteo sorgente: {weather_stats.get("source_timezone", "non specificata")}
- Celle ERG5: {", ".join(weather_stats.get("cells", [])) if weather_stats.get("cells") else "non applicabile"}
- Colonne meteo nel join: {", ".join(weather_stats["columns"])}

### File meteo locali

{markdown_table(pd.DataFrame(weather_stats.get("files", [])), max_rows=20)}

"""
    else:
        report += (
            "Nessun meteo e' stato caricato. Per usare i file locali ERG5, fornire "
            "`--weather-dir Data/Meteo`; per un CSV gia' normalizzato usare `--weather-csv`.\n\n"
        )

    report += f"""## Output prodotti

{markdown_table(output_paths)}
"""

    report_path = output_dir / "report_dozza_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    hourly_missing_inner.to_csv(output_dir / "missing_join_inner.csv", index=False)
    hourly_missing_left.to_csv(output_dir / "missing_join_left.csv", index=False)
    return report


def parse_lags(value: str) -> list[int]:
    lags = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        lag = int(raw)
        if lag < 0:
            raise argparse.ArgumentTypeError("I lag devono essere >= 0.")
        lags.append(lag)
    if not lags:
        raise argparse.ArgumentTypeError("Specificare almeno un lag, es. 0,1,24.")
    return sorted(set(lags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisi Dozza pedoni + TIM")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--excel-path", type=Path)
    parser.add_argument("--pedestrian-sheet", default="dati pedoni orari")
    parser.add_argument("--tim-dir", type=Path, default=Path("Data/TIM_ENEA"))
    parser.add_argument("--weather-csv", type=Path)
    parser.add_argument("--weather-dir", type=Path, default=Path("Data/Meteo"))
    parser.add_argument(
        "--weather-source-tz",
        default="none",
        help="Timezone dei timestamp meteo locali. Usa 'none' se sono gia' allineati ai pedoni.",
    )
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dozza_analysis"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--tim-limit-files", type=int)
    parser.add_argument("--rebuild-tim", action="store_true")
    parser.add_argument(
        "--tim-source-tz",
        default="UTC",
        help="Timezone dei timestamp TIM prima del join. Usa 'none' se sono gia' locali.",
    )
    parser.add_argument(
        "--analysis-tz",
        default="Europe/Rome",
        help="Timezone locale usata per allineare TIM a pedoni/meteo.",
    )
    parser.add_argument(
        "--correlation-lags",
        type=parse_lags,
        default=parse_lags("0,1,2,3,6,12,24"),
        help="Lag in ore da testare per correlare target[t] con feature[t-lag], es. 0,1,24.",
    )
    parser.add_argument(
        "--correlation-top-n",
        type=int,
        default=20,
        help="Numero di correlazioni principali da riportare per target e metodo.",
    )
    parser.add_argument(
        "--feature-correlation-threshold",
        type=float,
        default=0.95,
        help="Soglia assoluta Pearson per segnalare feature molto correlate tra loro.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = args.excel_path or discover_excel(args.data_dir)
    print(f"[PEDONI] Leggo {excel_path}", flush=True)
    pedestrians, pedestrian_stats = read_pedestrians(excel_path, args.pedestrian_sheet)
    pedestrians.to_csv(args.output_dir / "pedoni_hourly_targets.csv", index=False)

    print(f"[TIM] Aggrego CSV in {args.tim_dir}", flush=True)
    tim_15min, tim_hourly, tim_metadata = aggregate_tim(
        args.tim_dir,
        args.output_dir,
        args.chunk_size,
        args.tim_limit_files,
        args.rebuild_tim,
        args.tim_source_tz,
        args.analysis_tz,
    )
    tim_hourly.to_csv(args.output_dir / "tim_hourly_features.csv", index=False)

    weather_hourly = None
    weather_stats = None
    if args.no_weather:
        print("[METEO] Meteo disattivato da --no-weather", flush=True)
    elif args.weather_csv:
        print(f"[METEO] Leggo {args.weather_csv}", flush=True)
        weather_hourly, weather_stats = load_weather(args.weather_csv)
        weather_hourly.to_csv(args.output_dir / "weather_hourly_features.csv", index=False)
    elif args.weather_dir and args.weather_dir.exists():
        print(f"[METEO] Leggo ERG5 locale da {args.weather_dir}", flush=True)
        weather_hourly, weather_stats = load_local_erg5_weather(
            weather_dir=args.weather_dir,
            source_tz=args.weather_source_tz,
            analysis_tz=args.analysis_tz,
        )
        weather_hourly.to_csv(args.output_dir / "weather_hourly_features.csv", index=False)

    joined_inner, joined_left, join_stats = build_join(pedestrians, tim_hourly, weather_hourly)
    joined_inner.to_csv(args.output_dir / "dozza_joined_hourly_inner.csv", index=False)
    joined_left.to_csv(args.output_dir / "dozza_joined_hourly_left.csv", index=False)

    correlations = analyze_correlations(
        joined_inner,
        output_dir=args.output_dir,
        lags=args.correlation_lags,
        top_n=args.correlation_top_n,
        feature_pair_threshold=args.feature_correlation_threshold,
    )

    report = write_report(
        output_dir=args.output_dir,
        excel_path=excel_path,
        tim_dir=args.tim_dir,
        pedestrian_stats=pedestrian_stats,
        tim_15min=tim_15min,
        tim_hourly=tim_hourly,
        tim_metadata=tim_metadata,
        join_stats=join_stats,
        joined_inner=joined_inner,
        joined_left=joined_left,
        correlations=correlations,
        correlation_lags=args.correlation_lags,
        correlation_top_n=args.correlation_top_n,
        feature_pair_threshold=args.feature_correlation_threshold,
        weather_stats=weather_stats,
    )
    print(report)
    print(f"\n[OK] Report scritto in {args.output_dir / 'report_dozza_analysis.md'}", flush=True)


if __name__ == "__main__":
    main()
