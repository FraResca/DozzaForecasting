#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash slurm_jobs/dozza_three_analyses/submit_all.sh "$@"
