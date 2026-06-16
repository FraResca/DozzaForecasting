#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

REFERENCE_CSV="${REFERENCE_CSV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dozza_events_local}"
MANUAL_EVENTS="${MANUAL_EVENTS:-Data/Eventi/manual_events.csv}"
ADDITIONAL_EVENTS="${ADDITIONAL_EVENTS:-Data/Eventi/major_events_2025.csv}"
CITY_LOCATIONS="${CITY_LOCATIONS:-Data/Eventi/city_locations.csv}"
SOURCE_CONFIG="${SOURCE_CONFIG:-Data/Eventi/source_config_auto.csv}"
FREEZE_LOCAL_EVENTS="${FREEZE_LOCAL_EVENTS:-1}"
FROZEN_EVENTS="${FROZEN_EVENTS:-Data/Eventi/downloaded_events_local.csv}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-30}"
DOZZA_MIN_SCALE="${DOZZA_MIN_SCALE:-2}"
IMOLA_MIN_SCALE="${IMOLA_MIN_SCALE:-3}"
NEARBY_MIN_SCALE="${NEARBY_MIN_SCALE:-3}"
NEARBY_RADIUS_KM="${NEARBY_RADIUS_KM:-15}"
OTHER_MIN_SCALE="${OTHER_MIN_SCALE:-4}"
WRITE_LEFT_JOIN="${WRITE_LEFT_JOIN:-1}"
LEFT_REFERENCE_CSV="${LEFT_REFERENCE_CSV:-}"

if [[ -z "${REFERENCE_CSV}" ]]; then
  for candidate in \
    outputs/dozza_preprocess/dozza_joined_hourly_inner.csv \
    outputs/dozza_analysis_local_meteo/dozza_joined_hourly_inner.csv \
    outputs/dozza_analysis/dozza_joined_hourly_inner.csv
  do
    if [[ -s "${candidate}" ]]; then
      REFERENCE_CSV="${candidate}"
      break
    fi
  done
fi

if [[ -z "${REFERENCE_CSV}" || ! -s "${REFERENCE_CSV}" ]]; then
  echo "[ERROR] Nessun reference CSV trovato. Imposta REFERENCE_CSV=path/al/dozza_joined_hourly_inner.csv" >&2
  exit 1
fi

if [[ -z "${LEFT_REFERENCE_CSV}" ]]; then
  candidate_from_reference="${REFERENCE_CSV/_inner.csv/_left.csv}"
  if [[ "${candidate_from_reference}" != "${REFERENCE_CSV}" && -s "${candidate_from_reference}" ]]; then
    LEFT_REFERENCE_CSV="${candidate_from_reference}"
  else
    for candidate in \
      outputs/dozza_preprocess/dozza_joined_hourly_left.csv \
      outputs/dozza_analysis_local_meteo/dozza_joined_hourly_left.csv \
      outputs/dozza_analysis/dozza_joined_hourly_left.csv
    do
      if [[ -s "${candidate}" ]]; then
        LEFT_REFERENCE_CSV="${candidate}"
        break
      fi
    done
  fi
fi

mkdir -p "${OUTPUT_DIR}"
JOINED_OUTPUT="${OUTPUT_DIR}/dozza_joined_hourly_inner_with_events.csv"
LEFT_JOINED_OUTPUT="${OUTPUT_DIR}/dozza_joined_hourly_left_with_events.csv"

cmd=(
  python
  scripts/build_dozza_event_features.py
  --manual-events "${MANUAL_EVENTS}"
  --city-locations "${CITY_LOCATIONS}"
  --reference-csv "${REFERENCE_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --joined-output-csv "${JOINED_OUTPUT}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --dozza-min-scale "${DOZZA_MIN_SCALE}"
  --imola-min-scale "${IMOLA_MIN_SCALE}"
  --nearby-min-scale "${NEARBY_MIN_SCALE}"
  --nearby-radius-km "${NEARBY_RADIUS_KM}"
  --other-min-scale "${OTHER_MIN_SCALE}"
)

if [[ -s "${ADDITIONAL_EVENTS}" ]]; then
  cmd+=(--additional-events-csv "${ADDITIONAL_EVENTS}")
fi

if [[ -s "${SOURCE_CONFIG}" ]]; then
  cmd+=(--source-config-csv "${SOURCE_CONFIG}")
fi

if [[ "${WRITE_LEFT_JOIN}" == "1" && -n "${LEFT_REFERENCE_CSV}" && -s "${LEFT_REFERENCE_CSV}" && "${LEFT_REFERENCE_CSV}" != "${REFERENCE_CSV}" ]]; then
  cmd+=(--extra-joined-reference-csv "${LEFT_REFERENCE_CSV}")
  cmd+=(--extra-joined-output-csv "${LEFT_JOINED_OUTPUT}")
fi

echo "[INFO] Reference CSV: ${REFERENCE_CSV}"
echo "[INFO] Left reference CSV: ${LEFT_REFERENCE_CSV:-}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Source config: ${SOURCE_CONFIG}"
echo "[INFO] Additional events: ${ADDITIONAL_EVENTS}"
"${cmd[@]}"

if [[ "${FREEZE_LOCAL_EVENTS}" == "1" ]]; then
  python scripts/freeze_dozza_local_events.py \
    --input-csv "${OUTPUT_DIR}/events_raw.csv" \
    --output-csv "${FROZEN_EVENTS}"
fi

python - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

out = Path(sys.argv[1])

def safe_read(name: str) -> pd.DataFrame:
    path = out / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

status = safe_read("event_source_status.csv")
raw = safe_read("events_raw.csv")
clean = safe_read("events_clean.csv")
hourly = safe_read("event_hourly_features.csv")

print("\n[SUMMARY]")
print(f"events_raw: {len(raw)}")
print(f"events_clean: {len(clean)}")
if not hourly.empty and "event_active_any" in hourly:
    print(f"active_hours: {int(hourly['event_active_any'].sum())}")
if not clean.empty:
    print("\n[EVENTS BY SOURCE]")
    print(clean.groupby("source", dropna=False).size().sort_values(ascending=False).to_string())
    print("\n[EVENTS BY CITY]")
    print(clean.groupby("city", dropna=False).size().sort_values(ascending=False).to_string())
if not status.empty:
    print("\n[SOURCE STATUS]")
    print(status[["source", "status", "rows", "error"]].to_string(index=False))
print(f"\nReport: {out / 'event_report.md'}")
print(f"Joined CSV: {out / 'dozza_joined_hourly_inner_with_events.csv'}")
left_joined = out / "dozza_joined_hourly_left_with_events.csv"
if left_joined.exists():
    left = pd.read_csv(left_joined, usecols=["timestamp", "event_active_any"])
    print(f"Left joined CSV: {left_joined}")
    print(f"left_rows: {len(left)}")
    print(f"left_active_rows: {int(left['event_active_any'].sum())}")
PY
