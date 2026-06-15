#!/bin/bash
#SBATCH --job-name=dozza_nationality_h12h
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_dozza/%x_%j.out
#SBATCH --error=logs/slurm_dozza/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "logs/slurm_dozza" "outputs/slurm_dozza_three_analyses" "outputs/slurm_dozza_three_analyses/h12h/nationality"

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

echo "[INFO] Analisi: Italiani e stranieri TIM"
echo "[INFO] Orizzonte: 12 ore"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

conda \
  run \
  -n \
  s4c \
  python \
  scripts/model_dozza_flows.py \
  --input-csv \
  outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv \
  --output-dir \
  outputs/slurm_dozza_three_analyses/h12h/nationality \
  --target-set \
  nationality \
  --feature-scope \
  auto \
  --mode \
  forecast \
  --horizon-hours \
  12 \
  --embargo-hours \
  12 \
  --lags \
  1,2,24,168 \
  --rolling-windows \
  3,6,24 \
  --include-target-lags \
  --models \
  dummy_mean,dummy_median,last_hour,same_hour_previous_day,same_hour_previous_week,rolling_mean_24h,rolling_mean_168h,ridge,log1p_ridge,poisson,tweedie,random_forest,extra_trees,hist_gradient_boosting,xgboost,lightgbm,two_stage_ridge \
  --top-k-grid \
  15,25,30,40,60 \
  --permutation-repeats \
  20 \
  --shap-samples \
  300 \
  --shap-background-size \
  100 \
  --bootstrap-samples \
  300 \
  --tune \
  --rolling-validation \
  --rolling-min-train-periods \
  1 \
  --rolling-max-folds \
  8

echo "[INFO] End: $(date -Is)"
