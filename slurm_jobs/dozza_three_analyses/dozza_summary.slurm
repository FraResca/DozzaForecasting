#!/bin/bash
#SBATCH --job-name=dozza_summary
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_dozza/%x_%j.out
#SBATCH --error=logs/slurm_dozza/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "logs/slurm_dozza" "outputs/slurm_dozza_three_analyses" "outputs/slurm_dozza_three_analyses/horizon_summary"

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

echo "[INFO] Sintesi multi-orizzonte Dozza"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

conda \
  run \
  -n \
  s4c \
  python \
  scripts/summarize_dozza_horizon_results.py \
  --root-output-dir \
  outputs/slurm_dozza_three_analyses \
  --output-dir \
  outputs/slurm_dozza_three_analyses/horizon_summary

echo "[INFO] End: $(date -Is)"
