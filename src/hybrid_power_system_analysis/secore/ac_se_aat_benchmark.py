import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE
from paths import measurement_file, model_file
from secore.ac_se import ACStateEstimator
from secore.se_math import (
    CHOLMOD_ANALYZE_AAT,
    CHOLMOD_CHOLESKY_AAT,
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    LowerNormalEquationCscPlan,
    NormalEquationSolver,
)


DEFAULT_CASES = ("ieee3k", "ieee3w")
_CASE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _case_paths(case_name: str):
    if not _CASE_RE.match(case_name):
        raise ValueError(f"Invalid case name: {case_name!r}")
    return model_file("ac", f"{case_name}.e"), measurement_file("ac", f"{case_name}.meas")


def _initial_linearization(case_name: str, parameter_file: Path, flat_start: bool):
    e_file, meas_file = _case_paths(case_name)
    estimator = ACStateEstimator(
        e_file=e_file,
        meas_file=meas_file,
        parameter_file=parameter_file,
        flat_start=flat_start,
        auto_prepare=False,
    )
    estimator.prepare()
    measurement_plan_tables = estimator._active_measurement_plan_tables_ref()
    x = estimator.initial_state()
    z, weight = estimator._measurement_vectors(measurement_plan_tables)
    z_est = np.empty_like(z)
    residual = np.empty_like(z)
    estimator.evaluate(x, measurement_plan_tables, out=z_est)
    estimator._measurement_residual(z, z_est, measurement_plan_tables, out=residual)
    H = estimator.jacobian_sparse(x, measurement_plan_tables)
    return {
        "H": H,
        "residual": residual,
        "weight": weight,
        "uniform_weight": estimator.active_uniform_weight,
        "weights_are_uniform": estimator.active_weights_are_uniform,
        "state_count": int(estimator.n_state),
        "measurement_count": int(z.size),
    }


def _summary(values):
    values = [float(item) for item in values]
    return {
        "total": float(sum(values)),
        "avg": float(sum(values) / len(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
        "runs": values,
    }


def _benchmark_lower(data, repeats: int):
    H = data["H"]
    residual = data["residual"]
    weight = data["weight"]
    start = time.perf_counter()
    plan = LowerNormalEquationCscPlan.from_jacobian(H)
    plan_build = time.perf_counter() - start
    solver = NormalEquationSolver(assume_fixed_pattern=True)
    assemble_times = []
    solve_times = []
    dx = None
    for _ in range(repeats):
        start = time.perf_counter()
        gain, rhs = plan.assemble(
            H,
            residual,
            weight,
            uniform_weight=data["uniform_weight"],
            weights_are_uniform=data["weights_are_uniform"],
            dense_gain_limit=0,
            assume_fixed_weights=True,
            copy_rhs=False,
        )
        assemble_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dx, _ = solver.solve(gain, rhs, return_factor_diag=False)
        solve_times.append(time.perf_counter() - start)
    return {
        "plan_build": float(plan_build),
        "assemble": _summary(assemble_times),
        "solve": _summary(solve_times),
        "total": float(plan_build + sum(assemble_times) + sum(solve_times)),
        "dx": dx,
    }


def _benchmark_aat(data, repeats: int):
    H = data["H"]
    residual = data["residual"]
    weight = data["weight"]
    start = time.perf_counter()
    plan = CholmodAAtNormalEquationPlan.from_jacobian(H)
    plan_build = time.perf_counter() - start
    solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True)
    assemble_times = []
    solve_times = []
    dx = None
    for _ in range(repeats):
        start = time.perf_counter()
        A, rhs = plan.assemble(
            H,
            residual,
            weight,
            uniform_weight=data["uniform_weight"],
            weights_are_uniform=data["weights_are_uniform"],
            assume_fixed_weights=True,
            copy_rhs=False,
        )
        assemble_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dx, _ = solver.solve(A, rhs, return_factor_diag=False)
        solve_times.append(time.perf_counter() - start)
    return {
        "plan_build": float(plan_build),
        "assemble": _summary(assemble_times),
        "solve": _summary(solve_times),
        "total": float(plan_build + sum(assemble_times) + sum(solve_times)),
        "dx": dx,
        "cholmod_aat_available": bool(CHOLMOD_ANALYZE_AAT is not None or CHOLMOD_CHOLESKY_AAT is not None),
        "cholmod_aat_disabled": bool(getattr(solver, "_cholmod_disabled", False)),
    }


def run_case(case_name: str, repeats: int, parameter_file: Path, flat_start: bool = True):
    load_start = time.perf_counter()
    data = _initial_linearization(case_name, parameter_file, flat_start)
    load_time = time.perf_counter() - load_start
    lower = _benchmark_lower(data, repeats)
    aat = _benchmark_aat(data, repeats)
    dx_diff = np.asarray(aat["dx"], dtype=np.float64) - np.asarray(lower["dx"], dtype=np.float64)
    lower.pop("dx", None)
    aat.pop("dx", None)
    return {
        "case": case_name,
        "state_count": data["state_count"],
        "measurement_count": data["measurement_count"],
        "jacobian_nnz": int(data["H"].nnz),
        "linearization_load": float(load_time),
        "repeats": int(repeats),
        "lower": lower,
        "aat": aat,
        "dx_max_abs_diff": float(np.max(np.abs(dx_diff))) if dx_diff.size else 0.0,
        "dx_l2_diff": float(np.linalg.norm(dx_diff)) if dx_diff.size else 0.0,
    }


def run_cases(cases: Iterable[str], repeats: int, parameter_file: Path, flat_start: bool):
    return [run_case(case, repeats, parameter_file, flat_start=flat_start) for case in cases]


def _print_result(result) -> None:
    print(
        f"{result['case']}: states={result['state_count']} meas={result['measurement_count']} "
        f"H.nnz={result['jacobian_nnz']} linearization_load={result['linearization_load']:.3f}s "
        f"dx_max_abs_diff={result['dx_max_abs_diff']:.3e}"
    )
    for name in ("lower", "aat"):
        item = result[name]
        print(
            f"  {name}: total={item['total']:.6f}s plan={item['plan_build']:.6f}s "
            f"assemble_avg={item['assemble']['avg']:.6f}s solve_avg={item['solve']['avg']:.6f}s "
            f"solve_runs=[{' '.join(f'{value:.6f}' for value in item['solve']['runs'])}]"
        )
        if name == "aat":
            print(
                f"    cholmod_aat_available={item['cholmod_aat_available']} "
                f"cholmod_aat_disabled={item['cholmod_aat_disabled']}"
            )


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare AC SE lower normal CSC and CHOLMOD AAt prototypes.")
    parser.add_argument("cases", nargs="*", default=list(DEFAULT_CASES), help="Case basenames under data/model/ac.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeated kernel solves for the same linearization.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation parameter file.")
    parser.add_argument("--file-start", action="store_true", help="Use E-file voltage/angle start instead of flat start.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args(argv)

    results = run_cases(
        args.cases,
        max(1, int(args.repeats)),
        Path(args.para),
        flat_start=not args.file_start,
    )
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, sort_keys=True))
    else:
        for result in results:
            _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
