#!/usr/bin/env python3
"""Genera i job Slurm per le tre analisi Dozza."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ANALYSES = {
    "flow": {
        "job": "dozza_flow",
        "description": "Ingressi e uscite pedonali",
        "feature_scope": "auto",
    },
    "nationality": {
        "job": "dozza_nationality",
        "description": "Italiani e stranieri TIM",
        "feature_scope": "auto",
    },
    "age": {
        "job": "dozza_age",
        "description": "Sei fasce eta' TIM",
        "feature_scope": "auto",
    },
}

ANALYSIS_ALIASES = {
    "dual": "nationality",
    "italiani_stranieri": "nationality",
    "nazionalita": "nationality",
    "six": "age",
    "eta": "age",
    "fasce_eta": "age",
}

OBSOLETE_JOB_FILES = {
    "dozza_dual.slurm",
    "dozza_six.slurm",
}


def comma_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_ints(value: str, allow_zero: bool = False) -> list[int]:
    parsed: list[int] = []
    for item in comma_items(value):
        integer = int(item)
        if integer < 0 or (integer == 0 and not allow_zero):
            minimum = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"Gli orizzonti devono essere interi {minimum}: {item}")
        parsed.append(integer)
    return sorted(set(parsed))


def canonical_analysis(name: str) -> str:
    return ANALYSIS_ALIASES.get(name, name)


def slurm_header(job_name: str, args: argparse.Namespace, mem: str, time: str) -> str:
    qos_line = f"#SBATCH --qos={args.qos}\n" if args.qos else ""
    account_line = f"#SBATCH --account={args.account}\n" if args.account else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
{qos_line}{account_line}#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={args.log_dir}/%x_%j.out
#SBATCH --error={args.log_dir}/%x_%j.err
"""


def runtime_prelude(args: argparse.Namespace, extra_dirs: list[str] | None = None) -> str:
    module_lines = "\n".join(f"module load {module}" for module in comma_items(args.modules))
    if not module_lines and args.conda_env:
        # Se manca un modulo esplicito, prova a caricare un ambiente conda.
        module_lines = "module load miniconda3/ 2>/dev/null || module load conda 2>/dev/null || true"
    if module_lines:
        module_lines = f"\n{module_lines}\n"
    
    conda_init = ""
    if args.conda_env:
        conda_init = """
# Inizializza conda.
eval "$(conda shell.bash hook)" 2>/dev/null || true
"""
    
    dirs = [args.log_dir, *list(extra_dirs or [])]
    mkdir_args = " ".join(f'"{path}"' for path in dirs)
    return f"""
set -euo pipefail

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p {mkdir_args}

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-cache"
export JOBLIB_TEMP_FOLDER="${{TMPDIR:-/tmp}}/joblib"
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
mkdir -p "$MPLCONFIGDIR" "$JOBLIB_TEMP_FOLDER"

{module_lines}{conda_init}"""


def rendered_command(command: list[str], conda_env: str | None = None) -> str:
    if conda_env:
        # Esegue il comando dentro l'ambiente conda indicato.
        command = ["conda", "run", "-n", conda_env] + command
    return " \\\n  ".join(shlex.quote(part) for part in command)


def project_path(path: str | Path) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else PROJECT_ROOT / parsed


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def model_input_csv(args: argparse.Namespace) -> str:
    if args.no_preprocess:
        return args.input_csv
    if args.include_events:
        return f"{args.preprocess_output_dir}/dozza_joined_hourly_inner_with_events.csv"
    return f"{args.preprocess_output_dir}/dozza_joined_hourly_inner.csv"


def model_top_k_args(target_set: str, args: argparse.Namespace) -> list[str]:
    if target_set == "age" and args.age_top_k and args.age_top_k > 0:
        return ["--top-k", str(args.age_top_k)]
    return ["--top-k-grid", args.top_k_grid]


def model_time(target_set: str, args: argparse.Namespace) -> str:
    if target_set == "flow" and args.flow_time:
        return args.flow_time
    if target_set == "nationality" and args.nationality_time:
        return args.nationality_time
    if target_set == "age" and args.age_time:
        return args.age_time
    return args.time


def model_mem(target_set: str, args: argparse.Namespace) -> str:
    if target_set == "flow" and args.flow_mem:
        return args.flow_mem
    if target_set == "nationality" and args.nationality_mem:
        return args.nationality_mem
    if target_set == "age" and args.age_mem:
        return args.age_mem
    return args.mem


def render_preprocess_job(args: argparse.Namespace) -> str:
    command = [
        "python",
        "scripts/analyze_dozza_datasets.py",
        "--output-dir",
        args.preprocess_output_dir,
        "--weather-dir",
        args.weather_dir,
        "--chunk-size",
        str(args.chunk_size),
        "--tim-source-tz",
        args.tim_source_tz,
        "--analysis-tz",
        args.analysis_tz,
        "--weather-source-tz",
        args.weather_source_tz,
    ]
    if args.excel_path:
        command.extend(["--excel-path", args.excel_path])
    if args.pedestrian_sheet:
        command.extend(["--pedestrian-sheet", args.pedestrian_sheet])
    if args.tim_dir:
        command.extend(["--tim-dir", args.tim_dir])
    if args.tim_limit_files is not None:
        command.extend(["--tim-limit-files", str(args.tim_limit_files)])
    if args.rebuild_tim:
        command.append("--rebuild-tim")
    if args.no_weather:
        command.append("--no-weather")

    event_command = []
    event_guard = ""
    event_preflight = ""
    if args.include_events:
        event_command = [
            "python",
            "scripts/build_dozza_event_features.py",
            "--manual-events",
            args.event_manual_events,
            "--city-locations",
            args.event_city_locations,
            "--reference-csv",
            f"{args.preprocess_output_dir}/dozza_joined_hourly_inner.csv",
            "--output-dir",
            args.preprocess_output_dir,
            "--joined-output-csv",
            f"{args.preprocess_output_dir}/dozza_joined_hourly_inner_with_events.csv",
            "--extra-joined-reference-csv",
            f"{args.preprocess_output_dir}/dozza_joined_hourly_left.csv",
            "--extra-joined-output-csv",
            f"{args.preprocess_output_dir}/dozza_joined_hourly_left_with_events.csv",
        ]
        if args.event_additional_events_csv:
            event_command.extend(["--additional-events-csv", args.event_additional_events_csv])
        if args.event_downloaded_events_csv:
            event_command.extend(["--additional-events-csv", args.event_downloaded_events_csv])
        if args.download_events_on_cluster and args.event_source_config_csv:
            event_command.extend(["--source-config-csv", args.event_source_config_csv])
        required_event_inputs = [
            args.event_manual_events,
            args.event_city_locations,
        ]
        if args.event_additional_events_csv:
            required_event_inputs.append(args.event_additional_events_csv)
        if args.event_downloaded_events_csv:
            required_event_inputs.append(args.event_downloaded_events_csv)
        event_preflight = "\n".join(
            [
                'for required_event_file in \\',
                *[f'  "{path}" \\' for path in required_event_inputs],
                "  ; do",
                '  if [[ ! -s "${required_event_file}" ]]; then',
                '    echo "[ERROR] File eventi richiesto mancante o vuoto: ${required_event_file}" >&2',
                "    exit 1",
                "  fi",
                "done",
            ]
        )
        event_guard = f'\ntest -s "{args.preprocess_output_dir}/dozza_joined_hourly_inner_with_events.csv"\n'

    event_block = ""
    if event_command:
        event_block = f"""
echo "[INFO] Costruzione feature eventi Dozza"

{event_preflight}

{rendered_command(event_command, args.conda_env)}
"""

    return (
        slurm_header(f"{args.job_prefix}_preprocess", args, args.preprocess_mem, args.preprocess_time)
        + runtime_prelude(args, [args.preprocess_output_dir])
        + f"""echo "[INFO] Preprocessing condiviso Dozza"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

{rendered_command(command, args.conda_env)}

test -s "{args.preprocess_output_dir}/dozza_joined_hourly_inner.csv"
{event_block}{event_guard}

echo "[INFO] End: $(date -Is)"
"""
    )


def render_job(
    target_set: str,
    args: argparse.Namespace,
    horizon_hours: int,
    multi_horizon: bool,
) -> str:
    analysis = ANALYSES[target_set]
    output_dir = (
        f"{args.run_output_dir}/h{horizon_hours}h/{target_set}"
        if multi_horizon
        else f"{args.run_output_dir}/{target_set}"
    )
    command = [
        "python",
        "scripts/model_dozza_flows.py",
        "--input-csv",
        model_input_csv(args),
        "--output-dir",
        output_dir,
        "--target-set",
        target_set,
        "--feature-scope",
        analysis["feature_scope"],
        "--mode",
        args.mode,
        "--horizon-hours",
        str(horizon_hours),
        "--embargo-hours",
        str(horizon_hours if args.mode == "forecast" else 0),
        "--lags",
        args.lags,
        "--rolling-windows",
        args.rolling_windows,
        "--include-target-lags",
        "--models",
        args.models,
        *model_top_k_args(target_set, args),
        "--permutation-repeats",
        str(args.permutation_repeats),
        "--shap-samples",
        str(args.shap_samples),
        "--shap-background-size",
        str(args.shap_background_size),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--tune",
        "--rolling-validation",
        "--rolling-min-train-periods",
        str(args.rolling_min_train_periods),
    ]
    if args.rolling_max_folds is not None:
        command.extend(["--rolling-max-folds", str(args.rolling_max_folds)])
    if args.no_save_final_models:
        command.append("--no-save-final-models")

    return (
        slurm_header(
            f"{args.job_prefix}_{target_set}_h{horizon_hours}h" if multi_horizon else f"{args.job_prefix}_{target_set}",
            args,
            model_mem(target_set, args),
            model_time(target_set, args),
        )
        + runtime_prelude(args, [args.run_output_dir, output_dir])
        + f"""
echo "[INFO] Analisi: {analysis['description']}"
echo "[INFO] Orizzonte: {horizon_hours} ore"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

{rendered_command(command, args.conda_env)}

echo "[INFO] End: $(date -Is)"
"""
    )


def render_summary_job(args: argparse.Namespace) -> str:
    output_dir = f"{args.run_output_dir}/horizon_summary"
    command = [
        "python",
        "scripts/summarize_dozza_horizon_results.py",
        "--root-output-dir",
        args.run_output_dir,
        "--output-dir",
        output_dir,
    ]
    return (
        slurm_header(f"{args.job_prefix}_summary", args, args.summary_mem, args.summary_time)
        + runtime_prelude(args, [args.run_output_dir, output_dir])
        + f"""
echo "[INFO] Sintesi multi-orizzonte Dozza"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date -Is)"

{rendered_command(command, args.conda_env)}

echo "[INFO] End: $(date -Is)"
"""
    )


def render_submit_script(
    preprocess_path: Path | None,
    job_paths: list[Path],
    summary_path: Path | None = None,
    summary_dependency: str = "afterany",
) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        "",
    ]
    submitted_vars: list[str] = []
    if preprocess_path is not None:
        lines.append(f'PREPROCESS_JOB_ID="$(sbatch --parsable {display_path(preprocess_path)})"')
        lines.append('echo "Submitted preprocessing: ${PREPROCESS_JOB_ID}"')
        lines.append("")
        for path in job_paths:
            var_name = path.stem.upper().replace("-", "_") + "_JOB_ID"
            lines.append(
                f'{var_name}="$(sbatch --parsable --dependency=afterok:${{PREPROCESS_JOB_ID}} {display_path(path)})"'
            )
            lines.append(f'echo "Submitted {path.stem}: ${{{var_name}}} afterok:${{PREPROCESS_JOB_ID}}"')
            submitted_vars.append(var_name)
    else:
        for path in job_paths:
            var_name = path.stem.upper().replace("-", "_") + "_JOB_ID"
            lines.append(f'{var_name}="$(sbatch --parsable {display_path(path)})"')
            lines.append(f'echo "Submitted {path.stem}: ${{{var_name}}}"')
            submitted_vars.append(var_name)
    if summary_path is not None and submitted_vars:
        dependency = ":".join(f"${{{var}}}" for var in submitted_vars)
        lines.append("")
        lines.append(
            f'SUMMARY_JOB_ID="$(sbatch --parsable --dependency={summary_dependency}:{dependency} '
            f'{display_path(summary_path)})"'
        )
        lines.append(f'echo "Submitted {summary_path.stem}: ${{SUMMARY_JOB_ID}} {summary_dependency}:{dependency}"')
    lines.append("")
    return "\n".join(lines)


def render_submit_single_script(job_path: Path, preprocess_path: Path | None, preprocess_csv: str) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        "",
    ]
    if preprocess_path is None:
        lines.append(f'JOB_ID="$(sbatch --parsable {display_path(job_path)})"')
        lines.append('echo "Submitted job: ${JOB_ID}"')
    else:
        lines.extend(
            [
                f'if [[ -s "{preprocess_csv}" ]]; then',
                f'  JOB_ID="$(sbatch --parsable {display_path(job_path)})"',
                '  echo "Submitted job: ${JOB_ID}"',
                "else",
                f'  PREPROCESS_JOB_ID="$(sbatch --parsable {display_path(preprocess_path)})"',
                '  echo "Submitted preprocessing: ${PREPROCESS_JOB_ID}"',
                f'  JOB_ID="$(sbatch --parsable --dependency=afterok:${{PREPROCESS_JOB_ID}} {display_path(job_path)})"',
                '  echo "Submitted job: ${JOB_ID} afterok:${PREPROCESS_JOB_ID}"',
                "fi",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Slurm jobs for Dozza analyses.")
    parser.add_argument("--input-csv", default="outputs/dozza_analysis_local_meteo/dozza_joined_hourly_inner.csv")
    parser.add_argument("--output-dir", default="slurm_jobs/dozza_three_analyses")
    parser.add_argument("--run-output-dir", default="outputs/slurm_dozza_three_analyses")
    parser.add_argument("--preprocess-output-dir", default="outputs/slurm_dozza_preprocess")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--analyses", default="flow,nationality,age")
    parser.add_argument("--job-prefix", default="dozza")
    parser.add_argument("--mode", choices=["nowcast", "forecast"], default="forecast")
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument(
        "--horizons",
        default="1,3,6,12,24",
        help=(
            "Orizzonti forecast in ore, separati da virgola. Default: 1,3,6,12,24. "
            "Usare stringa vuota per mantenere solo --horizon-hours."
        ),
    )
    parser.add_argument("--lags", default="1,2,24,168")
    parser.add_argument("--rolling-windows", default="3,6,24")
    parser.add_argument(
        "--models",
        default=(
            "dummy_mean,dummy_median,last_hour,same_hour_previous_day,"
            "same_hour_previous_week,rolling_mean_24h,rolling_mean_168h,"
            "ridge,log1p_ridge,poisson,tweedie,random_forest,extra_trees,"
            "hist_gradient_boosting,xgboost,lightgbm,two_stage_ridge"
        ),
    )
    parser.add_argument("--top-k-grid", default="15,25,30,40,60")
    parser.add_argument(
        "--age-top-k",
        type=int,
        default=0,
        help=(
            "Top-k fisso opzionale per l'analisi age. Default 0: valida anche age con --top-k-grid. "
            "Usare un valore > 0 solo per una run age piu' leggera."
        ),
    )
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--shap-samples", type=int, default=300)
    parser.add_argument("--shap-background-size", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--rolling-min-train-periods", type=int, default=1)
    parser.add_argument("--rolling-max-folds", type=int, default=8)
    parser.add_argument("--no-save-final-models", action="store_true")
    parser.add_argument("--excel-path", default="")
    parser.add_argument("--pedestrian-sheet", default="dati pedoni orari")
    parser.add_argument("--tim-dir", default="Data/TIM_ENEA")
    parser.add_argument("--weather-dir", default="Data/Meteo")
    parser.add_argument("--no-weather", action="store_true")
    event_group = parser.add_mutually_exclusive_group()
    event_group.add_argument(
        "--include-events",
        dest="include_events",
        action="store_true",
        default=True,
        help="Nel pre-job genera dozza_joined_hourly_inner_with_events.csv e usa quel CSV nei modelli. Default.",
    )
    event_group.add_argument(
        "--no-events",
        dest="include_events",
        action="store_false",
        help="Disattiva la costruzione delle feature evento nel pre-job.",
    )
    parser.add_argument("--event-manual-events", default="Data/Eventi/manual_events.csv")
    parser.add_argument("--event-additional-events-csv", default="Data/Eventi/major_events_2025.csv")
    parser.add_argument(
        "--event-downloaded-events-csv",
        default="Data/Eventi/downloaded_events_local.csv",
        help=(
            "CSV di eventi automatici gia' scaricati localmente e sincronizzati nel repository. "
            "Usato di default dal cluster al posto del download da internet."
        ),
    )
    parser.add_argument("--event-city-locations", default="Data/Eventi/city_locations.csv")
    parser.add_argument("--event-source-config-csv", default="Data/Eventi/source_config_auto.csv")
    parser.add_argument(
        "--download-events-on-cluster",
        action="store_true",
        help="Abilita il download eventi nel pre-job Slurm tramite --event-source-config-csv.",
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--tim-limit-files", type=int)
    parser.add_argument("--rebuild-tim", action="store_true")
    parser.add_argument("--tim-source-tz", default="UTC")
    parser.add_argument("--analysis-tz", default="Europe/Rome")
    parser.add_argument("--weather-source-tz", default="none")

    parser.add_argument("--partition", default="compute")
    parser.add_argument("--qos", default="")
    parser.add_argument("--account", default="")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--flow-mem", default="16G")
    parser.add_argument("--nationality-mem", default="16G")
    parser.add_argument("--age-mem", default="24G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--flow-time", default="02:00:00")
    parser.add_argument("--nationality-time", default="02:00:00")
    parser.add_argument(
        "--age-time",
        default="04:00:00",
        help="Time limit dedicato all'analisi age, piu' lunga perche' ha sei target e valida top-k.",
    )
    parser.add_argument("--preprocess-mem", default="12G")
    parser.add_argument("--preprocess-time", default="00:30:00")
    parser.add_argument("--summary-mem", default="4G")
    parser.add_argument("--summary-time", default="00:30:00")
    parser.add_argument(
        "--summary-dependency",
        choices=["afterany", "afterok"],
        default="afterany",
        help=(
            "Dipendenza del job di sintesi dai job modello. Default afterany: la sintesi parte anche "
            "se un modello fallisce e riepiloga gli output disponibili."
        ),
    )
    parser.add_argument("--no-summary-job", action="store_true")
    parser.add_argument("--log-dir", default="logs/slurm_dozza")
    parser.add_argument("--modules", default="")
    parser.add_argument("--conda-env", default="s4c")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in OBSOLETE_JOB_FILES:
        obsolete_path = output_dir / obsolete
        if obsolete_path.exists():
            obsolete_path.unlink()
            print(f"Removed obsolete: {display_path(obsolete_path)}")
    requested = [canonical_analysis(item) for item in comma_items(args.analyses)]
    requested = list(dict.fromkeys(requested))
    unknown = sorted(set(requested) - set(ANALYSES))
    if unknown:
        raise ValueError(f"Analisi non riconosciute: {', '.join(unknown)}")
    horizons = comma_ints(args.horizons, allow_zero=args.mode == "nowcast") if args.horizons else [args.horizon_hours]
    if args.mode == "forecast" and any(horizon <= 0 for horizon in horizons):
        raise ValueError("In forecast gli orizzonti devono essere > 0.")
    multi_horizon = len(horizons) > 1 or bool(args.horizons)

    preprocess_path = None
    if not args.no_preprocess:
        preprocess_path = output_dir / f"{args.job_prefix}_preprocess.slurm"
        preprocess_path.write_text(render_preprocess_job(args), encoding="utf-8")
        print(f"Saved: {display_path(preprocess_path)}")

    job_paths = []
    if multi_horizon:
        for target_set in ANALYSES:
            stale_single = output_dir / f"{args.job_prefix}_{target_set}.slurm"
            if stale_single.exists():
                stale_single.unlink()
                print(f"Removed stale single-horizon job: {display_path(stale_single)}")
    for horizon in horizons:
        for target_set in requested:
            suffix = f"{target_set}_h{horizon}h" if multi_horizon else target_set
            path = output_dir / f"{args.job_prefix}_{suffix}.slurm"
            path.write_text(render_job(target_set, args, horizon, multi_horizon), encoding="utf-8")
            job_paths.append(path)
            print(f"Saved: {display_path(path)}")

    summary_path = None
    if not args.no_summary_job:
        summary_path = output_dir / f"{args.job_prefix}_summary.slurm"
        summary_path.write_text(render_summary_job(args), encoding="utf-8")
        print(f"Saved: {display_path(summary_path)}")

    submit_path = output_dir / "submit_all.sh"
    submit_path.write_text(
        render_submit_script(preprocess_path, job_paths, summary_path, args.summary_dependency),
        encoding="utf-8",
    )
    submit_path.chmod(0o755)
    print(f"Saved: {display_path(submit_path)}")

    if "age" in requested:
        submit_age_path = output_dir / "submit_age_only.sh"
        age_paths = [path for path in job_paths if f"{args.job_prefix}_age" in path.stem]
        if multi_horizon:
            submit_age_path.write_text(
                render_submit_script(preprocess_path, age_paths, None),
                encoding="utf-8",
            )
        else:
            age_path = output_dir / f"{args.job_prefix}_age.slurm"
            submit_age_path.write_text(
                render_submit_single_script(
                    age_path,
                    preprocess_path,
                    model_input_csv(args),
                ),
                encoding="utf-8",
            )
        submit_age_path.chmod(0o755)
        print(f"Saved: {display_path(submit_age_path)}")


if __name__ == "__main__":
    main()
