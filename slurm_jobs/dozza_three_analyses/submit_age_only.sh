#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PREPROCESS_JOB_ID="$(sbatch --parsable slurm_jobs/dozza_three_analyses/dozza_preprocess.slurm)"
echo "Submitted preprocessing: ${PREPROCESS_JOB_ID}"

DOZZA_AGE_H1H_JOB_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} slurm_jobs/dozza_three_analyses/dozza_age_h1h.slurm)"
echo "Submitted dozza_age_h1h: ${DOZZA_AGE_H1H_JOB_ID} afterok:${PREPROCESS_JOB_ID}"
DOZZA_AGE_H3H_JOB_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} slurm_jobs/dozza_three_analyses/dozza_age_h3h.slurm)"
echo "Submitted dozza_age_h3h: ${DOZZA_AGE_H3H_JOB_ID} afterok:${PREPROCESS_JOB_ID}"
DOZZA_AGE_H6H_JOB_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} slurm_jobs/dozza_three_analyses/dozza_age_h6h.slurm)"
echo "Submitted dozza_age_h6h: ${DOZZA_AGE_H6H_JOB_ID} afterok:${PREPROCESS_JOB_ID}"
DOZZA_AGE_H12H_JOB_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} slurm_jobs/dozza_three_analyses/dozza_age_h12h.slurm)"
echo "Submitted dozza_age_h12h: ${DOZZA_AGE_H12H_JOB_ID} afterok:${PREPROCESS_JOB_ID}"
DOZZA_AGE_H24H_JOB_ID="$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB_ID} slurm_jobs/dozza_three_analyses/dozza_age_h24h.slurm)"
echo "Submitted dozza_age_h24h: ${DOZZA_AGE_H24H_JOB_ID} afterok:${PREPROCESS_JOB_ID}"
