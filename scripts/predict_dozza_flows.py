#!/usr/bin/env python3
"""Predizione ingressi/uscite usando gli artefatti di `model_dozza_flows.py`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_dozza_flows import (  # noqa: E402
    TARGET_COLUMNS,
    create_lag_features,
    create_rolling_features,
    ensure_time_features,
)


def resolve_model_path(model_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = model_dir / "models" / path.name
    if candidate.exists():
        return candidate
    candidate = model_dir / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Modello non trovato: {raw_path}")


def load_prediction_frame(input_csv: Path, metadata: dict, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = ensure_time_features(df)
    df = create_rolling_features(
        df,
        windows=metadata.get("rolling_windows", []),
        include_targets=metadata.get("include_target_rolling", False),
    )
    df = create_lag_features(
        df,
        lags=metadata.get("lags", []),
        include_targets=metadata.get("include_target_lags", False),
    )
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predici ingressi/uscite Dozza da artefatti modello")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.model_dir / "modeling_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_features = metadata["selected_features"]
    saved_models = metadata.get("saved_models", {})
    if not saved_models:
        raise ValueError("Nessun modello salvato nei metadata. Rilancia senza --no-save-final-models.")

    df = load_prediction_frame(args.input_csv, metadata=metadata, max_rows=args.max_rows)
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = np.nan
    X = df[selected_features]

    output = pd.DataFrame({"timestamp": df["timestamp"]})
    horizon = int(metadata.get("horizon_hours", 0))
    output["target_timestamp"] = output["timestamp"] + pd.to_timedelta(horizon, unit="h")

    for target in TARGET_COLUMNS:
        if target in df.columns:
            output[f"{target}_observed"] = df[target].to_numpy()
        model_path_raw = saved_models.get(target)
        if not model_path_raw:
            continue
        model_path = resolve_model_path(args.model_dir, model_path_raw)
        model = load(model_path)
        output[f"{target}_predicted"] = np.clip(model.predict(X), a_min=0, a_max=None)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"[OK] Predizioni scritte in {args.output_csv}")


if __name__ == "__main__":
    main()
