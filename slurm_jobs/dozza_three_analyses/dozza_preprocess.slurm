#!/bin/bash
#SBATCH --job-name=dozza_preprocess
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_dozza/%x_%j.out
#SBATCH --error=logs/slurm_dozza/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "logs/slurm_dozza" "outputs/slurm_dozza_preprocess"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-cache"
export JOBLIB_TEMP_FOLDER="${TMPDIR:-/tmp}/joblib"
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
mkdir -p "$MPLCONFIGDIR" "$JOBLIB_TEMP_FOLDER"


module load miniconda3/ 2>/dev/null || module load conda 2>/dev/null || true

# Inizializza conda.
eval "$(conda shell.bash hook)" 2>/dev/null || true
echo "[INFO] Preprocessing condiviso Dozza"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

conda \
  run \
  -n \
  s4c \
  python \
  scripts/analyze_dozza_datasets.py \
  --output-dir \
  outputs/slurm_dozza_preprocess \
  --weather-dir \
  Data/Meteo \
  --chunk-size \
  500000 \
  --tim-source-tz \
  UTC \
  --analysis-tz \
  Europe/Rome \
  --weather-source-tz \
  none \
  --pedestrian-sheet \
  'dati pedoni orari' \
  --tim-dir \
  Data/TIM_ENEA

test -s "outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner.csv"

echo "[INFO] Costruzione feature eventi Dozza"

for required_event_file in \
  "Data/Eventi/manual_events.csv" \
  "Data/Eventi/city_locations.csv" \
  "Data/Eventi/major_events_2025.csv" \
  "Data/Eventi/downloaded_events_local.csv" \
  ; do
  if [[ ! -s "${required_event_file}" ]]; then
    echo "[ERROR] File eventi richiesto mancante o vuoto: ${required_event_file}" >&2
    exit 1
  fi
done

conda \
  run \
  -n \
  s4c \
  python \
  scripts/build_dozza_event_features.py \
  --manual-events \
  Data/Eventi/manual_events.csv \
  --city-locations \
  Data/Eventi/city_locations.csv \
  --reference-csv \
  outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner.csv \
  --output-dir \
  outputs/slurm_dozza_preprocess \
  --joined-output-csv \
  outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv \
  --extra-joined-reference-csv \
  outputs/slurm_dozza_preprocess/dozza_joined_hourly_left.csv \
  --extra-joined-output-csv \
  outputs/slurm_dozza_preprocess/dozza_joined_hourly_left_with_events.csv \
  --additional-events-csv \
  Data/Eventi/major_events_2025.csv \
  --additional-events-csv \
  Data/Eventi/downloaded_events_local.csv

test -s "outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv"


echo "[INFO] End: $(date -Is)"
