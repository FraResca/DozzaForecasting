#!/usr/bin/env python3
"""Pulisce artefatti obsoleti delle analisi Dozza.

La modalita' predefinita e' dry-run. Usa --execute per rimuovere davvero file e
cartelle.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SAFE_OUTPUT_DIRS = {
    "dozza_analysis_corr_test",
    "dozza_analysis_erg5",
    "dozza_analysis_image_test",
    "dozza_analysis_local_meteo_smoketest",
    "dozza_analysis_matrix_test",
    "dozza_analysis_test",
    "dozza_analysis_tz_test",
    "dozza_auto_topk_smoketest",
    "dozza_modeling_smoketest_counts",
    "dozza_rename_figures_smoketest",
    "dozza_three_analyses_smoketest",
}

LEGACY_OUTPUT_DIRS = {
    "dozza_analysis",
    "dozza_auto_topk_tree_models",
    "dozza_modeling_forecast_1h",
    "dozza_modeling_forecast_1h_erg5",
    "dozza_modeling_forecast_1h_local_meteo",
    "dozza_modeling_forecast_1h_v2",
    "dozza_modeling_nowcast",
}

PROTECTED_OUTPUT_DIRS = {
    "dozza_preprocess",
    "dozza_three_analyses",
    "dozza_analysis_local_meteo",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def matches_any(path: Path, patterns: list[str]) -> bool:
    label = rel(path)
    return any(fnmatch.fnmatch(label, pattern) for pattern in patterns)


def add_candidate(candidates: list[Candidate], path: Path, reason: str) -> None:
    if path.exists():
        candidates.append(Candidate(path=path, reason=reason))


def stale_output_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    outputs = PROJECT_ROOT / "outputs"
    if not outputs.exists():
        return candidates

    for child in sorted(outputs.iterdir()):
        if not child.is_dir():
            continue
        if child.name in PROTECTED_OUTPUT_DIRS:
            continue
        if child.name in SAFE_OUTPUT_DIRS or child.name.endswith("_smoketest") or child.name.endswith("_test"):
            add_candidate(candidates, child, "smoke/test output")
        elif child.name in LEGACY_OUTPUT_DIRS and args.include_legacy_runs:
            add_candidate(candidates, child, "legacy modeling output")

    current = outputs / "dozza_three_analyses"
    for obsolete_name in ["dual", "six"]:
        add_candidate(candidates, current / obsolete_name, "obsolete target-set output")

    if args.include_failed_current:
        for child in sorted(current.glob("*")):
            if not child.is_dir():
                continue
            if child.name not in {"flow", "nationality", "age"}:
                continue
            has_report = (child / "modeling_report.md").exists()
            has_metadata = (child / "modeling_metadata.json").exists()
            if not (has_report and has_metadata):
                add_candidate(candidates, child, "partial/failed current output")

    return candidates


def stale_job_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in [PROJECT_ROOT / "job_scripts", PROJECT_ROOT / "batch_jobs", PROJECT_ROOT / "scheduler_jobs"]:
        add_candidate(candidates, path, "generated execution job directory")
    return candidates


def stale_cache_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    if args.keep_pycache:
        return candidates
    for path in sorted(PROJECT_ROOT.rglob("__pycache__")):
        add_candidate(candidates, path, "python bytecode cache")
    for cache_name in [".pytest_cache", ".mypy_cache", ".ruff_cache"]:
        add_candidate(candidates, PROJECT_ROOT / cache_name, "tool cache")
    return candidates


def stale_log_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not args.include_logs:
        return candidates
    log_dir = PROJECT_ROOT / "logs" / "dozza"
    if not log_dir.exists():
        return candidates

    keep_latest_per_prefix: set[Path] = set()
    if args.keep_latest_logs > 0:
        groups: dict[str, list[Path]] = {}
        for path in log_dir.glob("*.out"):
            prefix = path.name.rsplit("_", 1)[0]
            groups.setdefault(prefix, []).append(path)
        for paths in groups.values():
            keep_latest_per_prefix.update(sorted(paths, key=lambda p: p.stat().st_mtime)[-args.keep_latest_logs :])
            for out_path in list(keep_latest_per_prefix):
                err_path = out_path.with_suffix(".err")
                if err_path.exists():
                    keep_latest_per_prefix.add(err_path)

    for path in sorted(log_dir.glob("*")):
        if path.is_file() and path not in keep_latest_per_prefix:
            add_candidate(candidates, path, "old execution log")
    return candidates


def collect_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = [
        *stale_output_candidates(args),
        *stale_job_candidates(args),
        *stale_cache_candidates(args),
        *stale_log_candidates(args),
    ]

    filtered: list[Candidate] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if matches_any(candidate.path, args.keep):
            continue
        filtered.append(candidate)
    return sorted(filtered, key=lambda item: rel(item.path))


def remove_candidate(candidate: Candidate) -> None:
    path = candidate.path
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean stale Dozza analysis artifacts.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove candidates. Without this flag the script only prints a dry-run plan.",
    )
    parser.add_argument(
        "--include-legacy-runs",
        action="store_true",
        help="Also remove older non-smoketest modeling runs such as dozza_modeling_forecast_1h*.",
    )
    parser.add_argument(
        "--include-failed-current",
        action="store_true",
        help="Also remove incomplete current outputs under outputs/dozza_three_analyses.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Also clean logs/dozza. Keeps latest logs per job prefix by default.",
    )
    parser.add_argument(
        "--keep-pycache",
        action="store_true",
        help="Keep __pycache__ and local tool cache directories.",
    )
    parser.add_argument(
        "--keep-latest-logs",
        type=int,
        default=1,
        help="When --include-logs is used, keep this many latest .out/.err pairs per job prefix.",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Glob relative to repo root to keep. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = collect_candidates(args)
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] {len(candidates)} candidate(s)")
    for candidate in candidates:
        path_type = "dir " if candidate.path.is_dir() else "file"
        print(f"{path_type}  {rel(candidate.path):<75}  # {candidate.reason}")

    if not args.execute:
        print("\nNo files removed. Re-run with --execute to delete these candidates.")
        return

    for candidate in candidates:
        remove_candidate(candidate)
    print(f"\nRemoved {len(candidates)} candidate(s).")


if __name__ == "__main__":
    main()
