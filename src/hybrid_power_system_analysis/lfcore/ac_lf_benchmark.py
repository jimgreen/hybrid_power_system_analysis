import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT_DIR = _find_project_root()
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import build_ac_ppc_from_e_file  # noqa: E402
from ac_lf import ACPowerFlowCalc  # noqa: E402
from paths import model_file  # noqa: E402


DEFAULT_CASES = ("ieee300", "ieee3k", "ieee1w", "ieee3w")
_CASE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _case_path(case_name: str) -> Path:
    if not _CASE_RE.match(case_name):
        raise ValueError(f"Invalid case name: {case_name!r}")
    return model_file("ac", f"{case_name}.e")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def run_case(case_name: str, repeats: int, profile: bool = False, algorithm: str = "nr", linear_solver: str = "scipy"):
    e_file = _case_path(case_name)
    if not e_file.exists():
        raise FileNotFoundError(e_file)
    runs = []
    profile_totals = {}
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        stage_start = start
        ppc = build_ac_ppc_from_e_file(e_file)
        load_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        calc = ACPowerFlowCalc(
            ppc,
            tol=1e-8,
            max_iter=300 if algorithm == "pq" else 50,
            algorithm=algorithm,
            keep_node_objects=False,
            linear_solver=linear_solver,
        )
        calc_init_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        _silent(calc.prepare)
        prepare_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        rc = _silent(calc.run)
        solve_s = time.perf_counter() - stage_start

        elapsed = time.perf_counter() - start
        runs.append(elapsed)
        if profile:
            profile_totals["load_network"] = profile_totals.get("load_network", 0.0) + load_s
            profile_totals["calc_init"] = profile_totals.get("calc_init", 0.0) + calc_init_s
            profile_totals["prepare"] = profile_totals.get("prepare", 0.0) + prepare_s
            profile_totals["solve"] = profile_totals.get("solve", 0.0) + solve_s
        last = {
            "case": case_name,
            "converged": calc.converged,
            "rc": int(rc),
            "iterations": calc.iterations,
            "norm": float(calc.normF),
            "nodes": int(ppc["bus"].shape[0]),
            "states": int(calc.total_vars),
            "algorithm": calc.used_algorithm,
            "linear_solver": calc.linear_solver,
        }
    last["times"] = runs
    last["avg"] = sum(runs) / len(runs)
    last["min"] = min(runs)
    if profile:
        last["profile"] = {name: value / len(runs) for name, value in sorted(profile_totals.items())}
    return last


def run_cases(cases: Iterable[str], repeats: int, profile: bool, algorithm: str = "nr", linear_solver: str = "scipy"):
    return [run_case(case, repeats, profile, algorithm, linear_solver) for case in cases]


def run_case_cold_process(
    case_name: str,
    repeats: int,
    profile: bool = False,
    algorithm: str = "nr",
    linear_solver: str = "scipy",
):
    runs = []
    profile_totals = {}
    last = None
    for _ in range(repeats):
        cmd = [
            sys.executable,
            "-m",
            "lfcore.ac_lf_benchmark",
            case_name,
            "--repeats",
            "1",
            "--json",
            "--algorithm",
            algorithm,
            "--linear-solver",
            linear_solver,
        ]
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


def run_cases_cold_process(cases: Iterable[str], repeats: int, profile: bool, algorithm: str = "nr", linear_solver: str = "scipy"):
    return [run_case_cold_process(case, repeats, profile, algorithm, linear_solver) for case in cases]


def _print_result(result) -> None:
    times = " ".join(f"{item:.3f}" for item in result["times"])
    print(
        f"{result['case']} [{result.get('algorithm', 'nr')}/{result.get('linear_solver', 'scipy')}]: "
        f"avg={result['avg']:.3f}s min={result['min']:.3f}s "
        f"runs=[{times}] converged={result['converged']} rc={result['rc']} "
        f"iter={result['iterations']} norm={result['norm']:.3e} "
        f"nodes={result['nodes']} states={result['states']}"
    )
    if "profile" in result:
        print("  profile:")
        for name, value in result["profile"].items():
            print(f"    {name}={value:.6f}s")


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Run AC LF cold-start benchmarks.")
    parser.add_argument("cases", nargs="*", default=list(DEFAULT_CASES), help="Case basenames under data/ac.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per case.")
    parser.add_argument("--profile", action="store_true", help="Print averaged stage timings.")
    parser.add_argument("--cold-process", action="store_true", help="Run each repeat in a fresh Python process.")
    parser.add_argument("--algorithm", choices=("nr", "pq"), default="nr", help="Power-flow algorithm.")
    parser.add_argument("--linear-solver", choices=("scipy", "auto", "pypardiso", "umfpack", "klu"), default="scipy", help="Sparse linear solver for NR.")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    runner = run_cases_cold_process if args.cold_process else run_cases
    results = runner(args.cases, max(1, args.repeats), args.profile, args.algorithm, args.linear_solver)
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, sort_keys=True))
        return 0 if all(item["converged"] and item["rc"] == 0 for item in results) else 1
    for result in results:
        _print_result(result)
    return 0 if all(item["converged"] and item["rc"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
