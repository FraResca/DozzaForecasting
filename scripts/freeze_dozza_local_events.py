#!/usr/bin/env python3
"""Congela gli eventi Dozza scaricati localmente in un CSV usabile su Slurm.

Il preprocessing sul cluster non deve dipendere dall'accesso a Internet. Lo
script prende l'output locale di build_dozza_event_features.py e scrive in
Data/Eventi solo gli eventi automatici scaricati, sincronizzati poi da
send_this.sh. Gli eventi manuali e curati sono esclusi di default per evitare
duplicati con manual_events.csv e major_events_2025.csv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/dozza_events_local/events_raw.csv")
DEFAULT_OUTPUT = Path("Data/Eventi/downloaded_events_local.csv")
CURATED_SOURCE_MARKERS = ("manual_official", "curated_official_2025")
KEEP_COLUMNS = [
    "event_name",
    "city",
    "lat",
    "lon",
    "start_datetime",
    "end_datetime",
    "category",
    "scale",
    "scale_reason",
    "source",
    "source_url",
    "confidence",
    "is_manual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Congela gli eventi scaricati localmente per l'uso su Slurm.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-curated",
        action="store_true",
        help="Include anche eventi manuali/curati. Di default vengono esclusi per evitare duplicati.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Path opzionale per il riepilogo JSON. Default: stesso path dell'output con suffisso .metadata.json.",
    )
    return parser.parse_args()


def source_is_curated(source: object) -> bool:
    text = str(source or "")
    return any(marker in text for marker in CURATED_SOURCE_MARKERS)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input eventi non trovato: {args.input_csv}")

    raw = pd.read_csv(args.input_csv)
    missing_required = {"event_name", "start_datetime"} - set(raw.columns)
    if missing_required:
        raise ValueError(f"Colonne obbligatorie mancanti in {args.input_csv}: {sorted(missing_required)}")

    frame = raw.copy()
    if not args.include_curated and "source" in frame:
        frame = frame.loc[~frame["source"].map(source_is_curated)].copy()

    for column in KEEP_COLUMNS:
        if column not in frame:
            frame[column] = ""
    if "is_manual" in frame:
        frame["is_manual"] = pd.to_numeric(frame["is_manual"], errors="coerce").fillna(0).astype(int)
    if "confidence" in frame:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.6)
    if "scale" in frame:
        frame["scale"] = pd.to_numeric(frame["scale"], errors="coerce")

    frame = frame[KEEP_COLUMNS].copy()
    frame = frame.dropna(subset=["event_name", "start_datetime"])
    frame = frame[frame["event_name"].astype(str).str.strip().ne("")]
    frame = frame.sort_values(["start_datetime", "city", "source", "event_name"], na_position="last")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    metadata_path = args.metadata_json or args.output_csv.with_suffix(".metadata.json")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": args.input_csv.as_posix(),
        "output_csv": args.output_csv.as_posix(),
        "include_curated": args.include_curated,
        "input_rows": int(len(raw)),
        "output_rows": int(len(frame)),
        "source_counts": frame["source"].fillna("").value_counts().to_dict() if "source" in frame else {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] Eventi congelati: {args.output_csv} ({len(frame)} righe)")
    print(f"[OK] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
