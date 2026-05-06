import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from secore.ac_se import ACStateEstimator


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ("ieee300", "ieee3k", "ieee1w", "ieee3w")
_CASE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _case_paths(case_name: str):
    if not _CASE_RE.match(case_name):
        raise ValueError(f"Invalid case name: {case_name!r}")
    base = ROOT_DIR / "data" / "ac"
    return base / f"{case_name}.e", base / f"{case_name}.meas"


def run_case(
    case_name: str,
    repeats: int,
    parameter_file: Path,
    flat_start: bool = True,
    skip_bad_data: bool = True,
    profile: bool = False,
):
    e_file, meas_file = _case_paths(case_name)
    runs = []
    profile_totals = {}
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        estimator = ACStateEstimator(
            e_file=e_file,
            meas_file=meas_file,
            parameter_file=parameter_file,
            flat_start=flat_start,
            profile=profile,
        )
        result = estimator.estimate(verbose=False, final_diagnostics=not skip_bad_data)
        bad_count = None
        max_normalized = None
        if not skip_bad_data:
            bad_items, normalized = estimator.identify_bad_data(result)
            bad_count = len(bad_items)
            max_normalized = float(normalized.max()) if normalized.size else 0.0
        elapsed = time.perf_counter() - start
        runs.append(elapsed)
        if profile:
            for name, value in estimator.profile_times.items():
                profile_totals[name] = profile_totals.get(name, 0.0) + float(value)
        last = {
            "case": case_name,
            "observable": result.observability.observable,
            "rank": result.observability.rank,
            "state_count": result.observability.state_count,
            "measurements": result.observability.measurement_count,
            "converged": result.converged,
            "iterations": result.iterations,
            "objective": result.objective,
            "max_dx": result.max_correction,
            "norm_res": result.residual_inf,
            "bad_count": bad_count,
            "max_normalized": max_normalized,
        }
    last["times"] = runs
    last["avg"] = sum(runs) / len(runs)
    last["min"] = min(runs)
    if profile:
        last["profile"] = {name: value / len(runs) for name, value in sorted(profile_totals.items())}
    return last


def run_cases(cases: Iterable[str], repeats: int, parameter_file: Path, flat_start: bool, skip_bad_data: bool, profile: bool):
    return [run_case(case, repeats, parameter_file, flat_start, skip_bad_data, profile) for case in cases]


def run_case_cold_process(
    case_name: str,
    repeats: int,
    parameter_file: Path,
    flat_start: bool = True,
    skip_bad_data: bool = True,
    profile: bool = False,
):
    runs = []
    profile_totals = {}
    last = None
    for _ in range(repeats):
        cmd = [
            sys.executable,
            "-m",
            "secore.ac_se_benchmark",
            case_name,
            "--repeats",
            "1",
            "--para",
            str(parameter_file),
            "--json",
        ]
        if not flat_start:
            cmd.append("--file-start")
        if not skip_bad_data:
            cmd.append("--with-bad-data")
        if profile:
            cmd.append("--profile")
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            check=True,
            capture_output=True,
            text=True,
        )
        child_result = json.loads(completed.stdout)
        runs.extend(float(item) for item in child_result["times"])
        if profile:
            for name, value in child_result.get("profile", {}).items():
                profile_totals[name] = profile_totals.get(name, 0.0) + float(value)
        last = child_result
    last["times"] = runs
    last["avg"] = sum(runs) / len(runs)
    last["min"] = min(runs)
    if profile:
        last["profile"] = {name: value / len(runs) for name, value in sorted(profile_totals.items())}
    return last


def run_cases_cold_process(
    cases: Iterable[str],
    repeats: int,
    parameter_file: Path,
    flat_start: bool,
    skip_bad_data: bool,
    profile: bool,
):
    return [
        run_case_cold_process(case, repeats, parameter_file, flat_start, skip_bad_data, profile)
        for case in cases
    ]


def _print_result(result) -> None:
    times = " ".join(f"{item:.3f}" for item in result["times"])
    print(
        f"{result['case']}: avg={result['avg']:.3f}s min={result['min']:.3f}s "
        f"runs=[{times}] converged={result['converged']} iter={result['iterations']} "
        f"observable={result['observable']} rank={result['rank']}/{result['state_count']} "
        f"measurements={result['measurements']} objective={result['objective']:.6e} "
        f"max_dx={result['max_dx']:.3e} norm_res={result['norm_res']:.3e}"
    )
    if result["bad_count"] is not None:
        print(
            f"  bad_data_count={result['bad_count']} "
            f"max_normalized_residual={result['max_normalized']:.3e}"
        )
    if "profile" in result:
        print("  profile:")
        for name, value in result["profile"].items():
            print(f"    {name}={value:.6f}s")


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Run AC SE benchmarks in one Python process.")
    parser.add_argument("cases", nargs="*", default=list(DEFAULT_CASES), help="Case basenames under data/ac.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per case.")
    parser.add_argument("--para", default=str(ROOT_DIR / "se.para"), help="State-estimation parameter file.")
    parser.add_argument("--file-start", action="store_true", help="Use E-file voltage/angle start instead of flat start.")
    parser.add_argument("--with-bad-data", action="store_true", help="Include post-estimation bad-data analysis.")
    parser.add_argument("--profile", action="store_true", help="Print averaged initialization profile timings.")
    parser.add_argument("--cold-process", action="store_true", help="Run each repeat in a fresh Python process.")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    runner = run_cases_cold_process if args.cold_process else run_cases
    results = runner(
        args.cases,
        max(1, args.repeats),
        Path(args.para),
        flat_start=not args.file_start,
        skip_bad_data=not args.with_bad_data,
        profile=args.profile,
    )
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, sort_keys=True))
        return 0 if all(item["converged"] and item["observable"] for item in results) else 1
    for result in results:
        _print_result(result)
    return 0 if all(item["converged"] and item["observable"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
