#!/usr/bin/env python3
"""Validate and benchmark the rich 5000-node multi-energy LF/SE case."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from model.meas_type import DEVICE_TYPE_CODES
from scripts.check_hybrid_converter_all_modes_1k import _decode_control_modes
from scripts.generate_hybrid_multi_energy_rich_5k import (
    COUPLING_TYPES,
    COUPLINGS_PER_TYPE,
    DEFAULT_CASE,
    DEFAULT_MEASUREMENTS,
    NODE_COUNTS,
    STORAGE_COUNTS,
)
from secore.hybrid_se import HybridStateEstimator


def _maximum(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _model_statistics(case: Path) -> dict[str, object]:
    book = EBook(case)
    node_blocks = {
        "ac": "ACNode",
        "dc": "DCNode",
        "heat": "HeatNode",
        "gas": "GasNode",
        "hydro": "HydroNode",
        "steam": "SteamNode",
    }
    return {
        "nodes": {
            domain: len(book.data[block].data)
            for domain, block in node_blocks.items()
        },
        "couplings": {
            table: len(book.data[table].data)
            for table in COUPLING_TYPES
        },
        "converters": {
            "ACACConverter": len(book.data["ACACConverter"].data),
            "DCDCConverter": len(book.data["DCDCConverter"].data),
            "DCACConverter": len(book.data["DCACConverter"].data),
        },
        "dcac_device_types": dict(
            sorted(
                Counter(
                    str(row["dev_type"])
                    for row in book.data["DCACConverter"].data
                ).items()
            )
        ),
        "storages": {
            table: len(book.data[table].data)
            for table in STORAGE_COUNTS
        },
    }


def benchmark_lf(case: Path) -> dict[str, object]:
    internal_start = time.perf_counter()
    stage_start = time.perf_counter()
    network = _read_lf_network_from_file(case)
    load_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    calc = HybridPowerFlowCalc(
        network,
        result_mode="full",
        linear_solver="scipy",
        tol=1.0e-8,
        max_iter=50,
        verbose=False,
    )
    construct_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    calc.prepare()
    prepare_time = time.perf_counter() - stage_start

    writeback_time = 0.0
    original_writeback = calc._write_back

    def timed_writeback():
        nonlocal writeback_time
        start = time.perf_counter()
        value = original_writeback()
        writeback_time += time.perf_counter() - start
        return value

    calc._write_back = timed_writeback
    stage_start = time.perf_counter()
    rc = calc.run()
    run_time = time.perf_counter() - stage_start
    residual = calc.get_f(calc.x)
    jacobian = calc.get_jacobi(calc.x)
    coupling_rows = np.asarray(
        [plan.eq_row for plan in calc.energy_coupling_plans],
        dtype=np.int64,
    )
    storage = {}
    for domain, fluid_calc in calc.fluid_calcs.items():
        flow = np.asarray(fluid_calc.lf_result.arrays["storage_flow"], dtype=np.float64)
        storage[domain] = {
            "count": int(flow.size),
            "discharging": int(np.count_nonzero(flow > 0.0)),
            "charging": int(np.count_nonzero(flow < 0.0)),
            "max_abs_flow": _maximum(flow),
        }
        if domain == "heat":
            storage[domain]["max_abs_heat_power"] = _maximum(
                fluid_calc.lf_result.arrays["storage_heat_power"]
            )
    return {
        "algorithm": "Hybrid LF",
        "converged": bool(rc == 0 and calc.converged),
        "iterations": int(calc.iterations),
        "phases": {
            "load": load_time,
            "construct": construct_time,
            "prepare": prepare_time,
            "newton": max(0.0, run_time - writeback_time),
            "writeback": writeback_time,
        },
        "accuracy": {
            "maximum_residual": _maximum(residual),
            "maximum_coupling_residual": _maximum(residual[coupling_rows]),
        },
        "statistics": {
            **_model_statistics(case),
            "variables": int(calc.total_vars),
            "equations": int(calc.total_eq),
            "jacobian_shape": list(jacobian.shape),
            "jacobian_nnz": int(jacobian.nnz),
            "active_couplings": len(calc.energy_coupling_plans),
            "converter_modes": _decode_control_modes(calc),
            "storage_results": storage,
        },
        "internal_total_s": time.perf_counter() - internal_start,
    }


def benchmark_se(case: Path, measurements: Path) -> dict[str, object]:
    internal_start = time.perf_counter()
    stage_start = time.perf_counter()
    estimator = HybridStateEstimator(
        case,
        measurements,
        flat_start=True,
        max_iter=50,
        auto_prepare=False,
    )
    construct_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    estimator.prepare()
    prepare_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    observability = estimator.observability_analysis()
    observability_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    result = estimator.estimate(
        observability=observability,
        verbose=False,
        final_diagnostics=True,
    )
    estimate_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    estimator._commit_multi_energy_results(
        result,
        result_mode="array",
        skip_bad_data=True,
        threshold=estimator.params.bad_threshold,
    )
    writeback_time = time.perf_counter() - stage_start

    table = result.measurement_table
    storage = {}
    for device_type in STORAGE_COUNTS:
        rows = np.flatnonzero(
            np.asarray(table.device_type_code, dtype=np.int64)
            == DEVICE_TYPE_CODES[device_type]
        )
        storage[device_type] = {
            "measurements": int(rows.size),
            "maximum_residual": _maximum(result.residual[rows]),
        }
    coupling_rows = estimator.multi_energy_coupling_measurement_slice
    return {
        "algorithm": "Hybrid SE",
        "converged": bool(observability.observable and result.converged),
        "iterations": int(result.iterations),
        "phases": {
            "construct": construct_time,
            "prepare": prepare_time,
            "observability": observability_time,
            "wls": estimate_time,
            "writeback": writeback_time,
        },
        "accuracy": {
            "observable": bool(observability.observable),
            "rank": int(observability.rank),
            "state_count": int(observability.state_count),
            "objective": float(result.objective),
            "maximum_residual": float(result.residual_inf),
            "maximum_coupling_residual": _maximum(result.residual[coupling_rows]),
            "storage_measurements": storage,
        },
        "statistics": {
            **_model_statistics(case),
            "states": int(estimator.multi_energy_state_count),
            "measurements": int(estimator.multi_energy_measurement_count),
            "jacobian_shape": list(result.H.shape),
            "jacobian_nnz": int(result.H.nnz),
            "active_couplings": len(estimator.multi_energy_se_coupling_plans),
        },
        "internal_total_s": time.perf_counter() - internal_start,
    }


def _run_worker(kind: str, case: Path, measurements: Path) -> int:
    result = benchmark_lf(case) if kind == "lf" else benchmark_se(case, measurements)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def _run_process(kind: str, case: Path, measurements: Path) -> dict[str, object]:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--case",
        str(case),
        "--measurements",
        str(measurements),
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(
            f"{kind} worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    payload["end_to_end_s"] = elapsed
    payload["startup_import_s"] = max(
        0.0,
        elapsed - float(payload["internal_total_s"]),
    )
    return payload


def _summary(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "average": statistics.fmean(data),
        "minimum": min(data),
        "maximum": max(data),
    }


def _aggregate(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "runs": len(runs),
        "converged": sum(bool(run["converged"]) for run in runs),
        "iterations": _summary(run["iterations"] for run in runs),
        "end_to_end_s": _summary(run["end_to_end_s"] for run in runs),
        "startup_import_s": _summary(run["startup_import_s"] for run in runs),
        "phase_average_s": {
            phase: statistics.fmean(float(run["phases"][phase]) for run in runs)
            for phase in runs[0]["phases"]
        },
    }


def _print_report(results: dict[str, list[dict[str, object]]]) -> None:
    model = results["lf"][0]["statistics"]
    print("\nModel")
    print(f"  nodes={model['nodes']} total={sum(model['nodes'].values())}")
    print(
        f"  couplings={sum(model['couplings'].values())} "
        f"({len(model['couplings'])} types x {COUPLINGS_PER_TYPE})"
    )
    print(
        f"  converters={model['converters']}, "
        f"DCAC device types={model['dcac_device_types']}"
    )
    print(f"  storages={model['storages']}")

    print("\nIndependent-process timing")
    print("  Algorithm | Runs | Converged | Iter avg | End-to-end avg | Min | Max")
    print("  ----------|------|-----------|----------|----------------|-----|----")
    for kind in ("lf", "se"):
        summary = _aggregate(results[kind])
        timing = summary["end_to_end_s"]
        print(
            f"  {results[kind][0]['algorithm']} | {summary['runs']} | "
            f"{summary['converged']}/{summary['runs']} | "
            f"{summary['iterations']['average']:.1f} | "
            f"{timing['average']:.6f}s | {timing['minimum']:.6f}s | "
            f"{timing['maximum']:.6f}s"
        )
        phases = ", ".join(
            f"{name}={value:.6f}s"
            for name, value in summary["phase_average_s"].items()
        )
        print(
            f"    {phases}, startup/import="
            f"{summary['startup_import_s']['average']:.6f}s"
        )

    lf = results["lf"][0]
    se = results["se"][0]
    print("\nAccuracy")
    print(
        f"  LF variables/equations={lf['statistics']['variables']}/"
        f"{lf['statistics']['equations']}, Jacobian="
        f"{lf['statistics']['jacobian_shape']} nnz={lf['statistics']['jacobian_nnz']}, "
        f"residual={lf['accuracy']['maximum_residual']:.3e}, coupling="
        f"{lf['accuracy']['maximum_coupling_residual']:.3e}"
    )
    print(
        f"  SE states/measurements={se['statistics']['states']}/"
        f"{se['statistics']['measurements']}, rank="
        f"{se['accuracy']['rank']}/{se['accuracy']['state_count']}, Jacobian="
        f"{se['statistics']['jacobian_shape']} nnz={se['statistics']['jacobian_nnz']}, "
        f"residual={se['accuracy']['maximum_residual']:.3e}, coupling="
        f"{se['accuracy']['maximum_coupling_residual']:.3e}"
    )
    print(f"  LF storage results={lf['statistics']['storage_results']}")
    print(f"  SE storage residuals={se['accuracy']['storage_measurements']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=("lf", "se"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    case = args.case.resolve()
    measurements = args.measurements.resolve()
    if args.worker:
        return _run_worker(args.worker, case, measurements)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not case.exists() or not measurements.exists():
        parser.error("generate the rich 5k model and measurement files first")

    order = [kind for _ in range(args.runs) for kind in ("lf", "se")]
    random.Random(args.seed).shuffle(order)
    results: dict[str, list[dict[str, object]]] = {"lf": [], "se": []}
    for sequence, kind in enumerate(order, start=1):
        run = _run_process(kind, case, measurements)
        results[kind].append(run)
        print(
            f"[{sequence}/{len(order)}] {run['algorithm']}: "
            f"converged={run['converged']}, iterations={run['iterations']}, "
            f"end-to-end={run['end_to_end_s']:.6f}s"
        )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    _print_report(results)
    return 0 if all(run["converged"] for group in results.values() for run in group) else 1


if __name__ == "__main__":
    raise SystemExit(main())
