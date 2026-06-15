#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-fresca}"
REMOTE_HOST="${REMOTE_HOST:-copernico.endif.man}"
REMOTE_BASE="${REMOTE_BASE:-/hpc/home/${REMOTE_USER}}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_BASE}/RetryDozza}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"

# Push sincronizza codice, dati e job lasciando invariati output e log remoti.
# Usa RSYNC_DELETE=0 per non cancellare file di codice remoti obsoleti.
RSYNC_DELETE="${RSYNC_DELETE:-1}"
# Pull-results allinea gli output remoti e rimuove artefatti locali non piu' presenti.
# La pulizia remota degli output viene gestita da model_dozza_flows.py.
RSYNC_RESULTS_DELETE="${RSYNC_RESULTS_DELETE:-1}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

RSYNC_CODE=(
  -avz
  --progress
  --human-readable
  --itemize-changes
  --exclude ".git/"
  --exclude ".agents/"
  --exclude ".codex/"
  --exclude ".conda/"
  --exclude ".conda-env/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude ".ipynb_checkpoints/"
  --exclude "logs/"
  --exclude "outputs/"
)

DELETE_OPTS=()
if [[ "${RSYNC_DELETE}" == "1" ]]; then
  DELETE_OPTS=(--delete)
fi

RESULT_DELETE_OPTS=()
if [[ "${RSYNC_RESULTS_DELETE}" == "1" ]]; then
  RESULT_DELETE_OPTS=(--delete)
fi

usage() {
  cat <<EOF
Usage:
  bash send_this.sh push
  bash send_this.sh verify-remote
  bash send_this.sh pull
  bash send_this.sh pull-results
  bash send_this.sh dry-clean-remote
  bash send_this.sh clean-remote
  bash send_this.sh dry-push
  bash send_this.sh dry-pull

Environment overrides:
  REMOTE_USER=fresca
  REMOTE_HOST=copernico.unife.it
  REMOTE_BASE=/hpc/home/fresca
  REMOTE_DIR=/hpc/home/fresca/RetryDozza
  RSYNC_DELETE=1         # push predefinito: allinea codice, dati e job
  RSYNC_DELETE=0         # disabilita cancellazioni remote
  RSYNC_RESULTS_DELETE=1 # pull-results predefinito: allinea output remoti
  RSYNC_RESULTS_DELETE=0 # conserva output locali non piu' presenti sul remoto

Examples:
  bash send_this.sh dry-push
  bash send_this.sh push
  bash send_this.sh verify-remote
  bash send_this.sh pull-results
  bash send_this.sh dry-clean-remote
  bash send_this.sh clean-remote
  RSYNC_DELETE=0 bash send_this.sh push

Remote path:
  ${REMOTE}:${REMOTE_DIR}

Notes:
  push preserves remote outputs/ and logs/, but deletes obsolete remote code/job files.
  pull-results mirrors outputs/ by default and downloads logs/.
EOF
}

ensure_remote_dir() {
  ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"
}

push_repo() {
  ensure_remote_dir
  echo "[PUSH CODE/DATA] ${LOCAL_DIR}/ -> ${REMOTE}:${REMOTE_DIR}/"
  echo "[PUSH CODE/DATA] delete obsolete remote code files: ${RSYNC_DELETE}"
  rsync "${RSYNC_CODE[@]}" "${DELETE_OPTS[@]}" \
    "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
}

pull_repo() {
  mkdir -p "${LOCAL_DIR}"
  echo "[PULL CODE/DATA] ${REMOTE}:${REMOTE_DIR}/ -> ${LOCAL_DIR}/"
  rsync "${RSYNC_CODE[@]}" \
    "${REMOTE}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
}

pull_results() {
  echo "[PULL RESULTS] ${REMOTE}:${REMOTE_DIR}/outputs/ -> ${LOCAL_DIR}/outputs/"
  mkdir -p "${LOCAL_DIR}/outputs" "${LOCAL_DIR}/logs"
  rsync -avz --progress --human-readable "${RESULT_DELETE_OPTS[@]}" \
    "${REMOTE}:${REMOTE_DIR}/outputs/" "${LOCAL_DIR}/outputs/" || true
  rsync -avz --progress --human-readable \
    "${REMOTE}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/logs/" || true
}

dry_push() {
  ensure_remote_dir
  echo "[DRY PUSH CODE/DATA] ${LOCAL_DIR}/ -> ${REMOTE}:${REMOTE_DIR}/"
  echo "[DRY PUSH CODE/DATA] delete obsolete remote code files: ${RSYNC_DELETE}"
  rsync --dry-run "${RSYNC_CODE[@]}" "${DELETE_OPTS[@]}" \
    "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
}

dry_pull() {
  echo "[DRY PULL CODE/DATA] ${REMOTE}:${REMOTE_DIR}/ -> ${LOCAL_DIR}/"
  rsync --dry-run "${RSYNC_CODE[@]}" \
    "${REMOTE}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
}

verify_remote() {
  echo "[VERIFY REMOTE] ${REMOTE}:${REMOTE_DIR}"
  local remote_python
  remote_python="$(mktemp)"
  cat > "${remote_python}" <<'PY'
from pathlib import Path

model_path = Path("scripts/model_dozza_flows.py")
events_script_path = Path("scripts/build_dozza_event_features.py")
event_source_config_path = Path("Data/Eventi/source_config_auto.csv")
event_major_events_path = Path("Data/Eventi/major_events_2025.csv")
event_downloaded_events_path = Path("Data/Eventi/downloaded_events_local.csv")
expected_horizons = [1, 3, 6, 12, 24]
age_job_paths = [Path(f"slurm_jobs/dozza_three_analyses/dozza_age_h{h}h.slurm") for h in expected_horizons]
flow_job_paths = [Path(f"slurm_jobs/dozza_three_analyses/dozza_flow_h{h}h.slurm") for h in expected_horizons]
nationality_job_paths = [
    Path(f"slurm_jobs/dozza_three_analyses/dozza_nationality_h{h}h.slurm") for h in expected_horizons
]
preprocess_job_path = Path("slurm_jobs/dozza_three_analyses/dozza_preprocess.slurm")
summary_job_path = Path("slurm_jobs/dozza_three_analyses/dozza_summary.slurm")
submit_all_path = Path("slurm_jobs/dozza_three_analyses/submit_all.sh")
submit_age_path = Path("slurm_jobs/dozza_three_analyses/submit_age_only.sh")
old_paths = [
    Path("slurm_jobs/dozza_three_analyses/dozza_flow.slurm"),
    Path("slurm_jobs/dozza_three_analyses/dozza_nationality.slurm"),
    Path("slurm_jobs/dozza_three_analyses/dozza_age.slurm"),
    Path("slurm_jobs/dozza_three_analyses/dozza_dual.slurm"),
    Path("slurm_jobs/dozza_three_analyses/dozza_six.slurm"),
    Path("scripts/slurm_jobs"),
]

text = model_path.read_text(encoding="utf-8") if model_path.exists() else ""
events_text = events_script_path.read_text(encoding="utf-8") if events_script_path.exists() else ""
age_job_texts = [path.read_text(encoding="utf-8") if path.exists() else "" for path in age_job_paths]
flow_job_texts = [path.read_text(encoding="utf-8") if path.exists() else "" for path in flow_job_paths]
nationality_job_texts = [path.read_text(encoding="utf-8") if path.exists() else "" for path in nationality_job_paths]
age_job_text = "\n".join(age_job_texts)
flow_job_text = "\n".join(flow_job_texts)
nationality_job_text = "\n".join(nationality_job_texts)
preprocess_job_text = preprocess_job_path.read_text(encoding="utf-8") if preprocess_job_path.exists() else ""
summary_job_text = summary_job_path.read_text(encoding="utf-8") if summary_job_path.exists() else ""
submit_all_text = submit_all_path.read_text(encoding="utf-8") if submit_all_path.exists() else ""
submit_age_text = submit_age_path.read_text(encoding="utf-8") if submit_age_path.exists() else ""

checks = {
    "model_script_exists": model_path.exists(),
    "events_script_exists": events_script_path.exists(),
    "event_source_config_exists": event_source_config_path.exists(),
    "event_major_events_exists": event_major_events_path.exists(),
    "event_downloaded_events_exists": event_downloaded_events_path.exists(),
    "all_age_horizon_jobs_exist": all(path.exists() for path in age_job_paths),
    "all_flow_horizon_jobs_exist": all(path.exists() for path in flow_job_paths),
    "all_nationality_horizon_jobs_exist": all(path.exists() for path in nationality_job_paths),
    "preprocess_job_exists": preprocess_job_path.exists(),
    "summary_job_exists": summary_job_path.exists(),
    "no_sklearn_permutation_importance_import": "sklearn.inspection" not in text,
    "no_sklearn_permutation_importance_call": "result = permutation_importance(" not in text,
    "manual_permutation_baseline_mae": "baseline_mae" in text,
    "timing_metrics_present": "fit_seconds" in text and "inference_ms_per_row" in text,
    "causal_baseline_present": "observed_targets" in text and "source_index = test_index - pd.to_timedelta(1" in text and 'closed="left"' in text,
    "forecast_embargo_default": "else (args.horizon_hours if args.mode == \"forecast\" else 0)" in text,
    "two_stage_present": "TwoStageRidgeRegressor" in text,
    "output_cleanup_present": "clean_generated_output_dir" in text and "--no-clean-output-dir" in text,
    "paper_ready_pdf_figures": "save_figure" in text and ".pdf" in text,
    "pull_results_delete_present": "RSYNC_RESULTS_DELETE" in Path("send_this.sh").read_text(encoding="utf-8"),
    "preprocess_builds_events": "build_dozza_event_features.py" in preprocess_job_text and "dozza_joined_hourly_inner_with_events.csv" in preprocess_job_text,
    "event_builder_no_zip_strict": "strict=" not in events_text,
    "preprocess_skips_cluster_event_download": "--source-config-csv" not in preprocess_job_text,
    "preprocess_uses_major_events": "Data/Eventi/major_events_2025.csv" in preprocess_job_text,
    "preprocess_uses_downloaded_events": "Data/Eventi/downloaded_events_local.csv" in preprocess_job_text,
    "preprocess_builds_left_events": "dozza_joined_hourly_left_with_events.csv" in preprocess_job_text,
    "flow_job_uses_events": "dozza_joined_hourly_inner_with_events.csv" in flow_job_text,
    "nationality_job_uses_events": "dozza_joined_hourly_inner_with_events.csv" in nationality_job_text,
    "age_job_uses_events": "dozza_joined_hourly_inner_with_events.csv" in age_job_text,
    "age_job_validates_top_k": "--top-k-grid" in age_job_text and "--top-k 30" not in age_job_text,
    "all_model_jobs_pass_embargo": all("--embargo-hours" in job_text for job_text in [*age_job_texts, *flow_job_texts, *nationality_job_texts]),
    "age_job_time_4h": "#SBATCH --time=04:00:00" in age_job_text,
    "age_job_mem_24g": "#SBATCH --mem=24G" in age_job_text,
    "flow_job_time_2h": "#SBATCH --time=02:00:00" in flow_job_text,
    "flow_job_mem_16g": "#SBATCH --mem=16G" in flow_job_text,
    "nationality_job_time_2h": "#SBATCH --time=02:00:00" in nationality_job_text,
    "nationality_job_mem_16g": "#SBATCH --mem=16G" in nationality_job_text,
    "preprocess_job_mem_12g": "#SBATCH --mem=12G" in preprocess_job_text,
    "joblib_tmp_configured": "JOBLIB_TEMP_FOLDER" in age_job_text and "JOBLIB_TEMP_FOLDER" in flow_job_text,
    "submit_all_afterok": "--dependency=afterok:${PREPROCESS_JOB_ID}" in submit_all_text,
    "submit_all_summary_afterany": "dozza_summary.slurm" in submit_all_text and "--dependency=afterany:" in submit_all_text,
    "summary_script_present": "summarize_dozza_horizon_results.py" in summary_job_text,
    "submit_age_after_preprocess": "afterok:${PREPROCESS_JOB_ID}" in submit_age_text,
}

failed = False
for name, ok in checks.items():
    print(f"{name}: {ok}")
    failed = failed or not ok

for path in old_paths:
    absent = not path.exists()
    print(f"obsolete_absent {path}: {absent}")
    failed = failed or not absent

if failed:
    raise SystemExit(1)
PY
  scp "${remote_python}" "${REMOTE}:/tmp/retrydozza_verify_remote.py" >/dev/null
  rm -f "${remote_python}"
  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && python /tmp/retrydozza_verify_remote.py"
}

dry_clean_remote() {
  echo "[DRY CLEAN REMOTE] ${REMOTE}:${REMOTE_DIR}"
  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && python scripts/clean_old_analyses.py --include-legacy-runs --include-logs"
}

clean_remote() {
  echo "[CLEAN REMOTE] ${REMOTE}:${REMOTE_DIR}"
  ssh "${REMOTE}" "cd '${REMOTE_DIR}' && python scripts/clean_old_analyses.py --include-legacy-runs --include-logs --execute"
}

case "${ACTION}" in
  push)
    push_repo
    ;;
  verify-remote)
    verify_remote
    ;;
  pull)
    pull_repo
    ;;
  pull-results)
    pull_results
    ;;
  dry-clean-remote)
    dry_clean_remote
    ;;
  clean-remote)
    clean_remote
    ;;
  dry-push)
    dry_push
    ;;
  dry-pull)
    dry_pull
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 1
    ;;
esac
