#!/usr/bin/env python3
"""Benchmark the 1040-node hybrid converter mode-coverage case.

The controller starts every LF/SE repetition in a fresh Python process.  The
worker reports calculation-stage timings while the controller adds process
startup and module-import overhead to obtain an end-to-end time.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from model.ac_array_model import (
    ACAC_COLS,
    ACAC_LEGACY_CONTROL_LABEL,
    ACAC_LEGACY_TO_PAIR,
    ACAC_SIDE_CONTROL_LABEL,
    SWITCH_COLS as AC_SWITCH_COLS,
)
from model.dc_array_model import (
    DCDC_COLS,
    DCDC_SIDE_CONTROL_LABEL,
    SWITCH_COLS as DC_SWITCH_COLS,
)
from model.hybrid_array_model import (
    DCAC_AC_CONTROL_LABEL,
    DCAC_COLS,
    DCAC_DC_CONTROL_LABEL,
    DCAC_DEVICE_TYPE_LABEL,
    DCAC_SUPPORTED_CONTROL_PAIRS,
)
from model.meas_type import (
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_ACBranch,
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACNode,
    DEVICE_TYPE_ACThreeWindingTransformer,
    DEVICE_TYPE_ACTransformer,
    DEVICE_TYPE_ACZeroBranch,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCZeroBranch,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_THIRD,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_THIRD,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_V,
)
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_all_modes_1k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_converter_all_modes_1k.meas"

EXPECTED_ACAC_MODES = frozenset(
    {
        ("PQ", "PQ"),
        ("PV", "PQ"),
        ("PQ", "PV"),
        ("PV", "PV"),
    }
)
EXPECTED_DCDC_MODES = frozenset(
    {
        ("P", "NONE"),
        ("NONE", "P"),
        ("V", "NONE"),
        ("NONE", "V"),
        ("I", "NONE"),
        ("NONE", "I"),
    }
)
EXPECTED_DCAC_MODES = frozenset(DCAC_SUPPORTED_CONTROL_PAIRS)


def _maximum(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _mode_text(mode: Sequence[str]) -> str:
    return "/".join(str(item) for item in mode)


def _decode_control_modes(calc: HybridPowerFlowCalc) -> dict[str, list[list[str]]]:
    acac = np.asarray(calc.ac_calc.ppc.get("acac", ()), dtype=np.float64)
    acac_modes = {
        (
            ACAC_SIDE_CONTROL_LABEL[int(row[ACAC_COLS["i_control_type"]])],
            ACAC_SIDE_CONTROL_LABEL[int(row[ACAC_COLS["j_control_type"]])],
        )
        for row in acac
        if int(row[ACAC_COLS["run_stat"]]) == 1
    }

    dcdc = np.asarray(calc.dc_calc.ppc.get("dcdc", ()), dtype=np.float64)
    dcdc_modes = {
        (
            DCDC_SIDE_CONTROL_LABEL[int(row[DCDC_COLS["i_control_type"]])],
            DCDC_SIDE_CONTROL_LABEL[int(row[DCDC_COLS["j_control_type"]])],
        )
        for row in dcdc
        if int(row[DCDC_COLS["run_stat"]]) == 1
    }

    dcac = np.asarray(calc.network.ppc.get("dcac", ()), dtype=np.float64)
    dcac_modes = {
        (
            DCAC_AC_CONTROL_LABEL[int(row[DCAC_COLS["ac_control_type"]])],
            DCAC_DC_CONTROL_LABEL[int(row[DCAC_COLS["dc_control_type"]])],
        )
        for row in dcac
        if int(row[DCAC_COLS["run_stat"]]) == 1
    }
    return {
        "acac": [list(mode) for mode in sorted(acac_modes)],
        "dcdc": [list(mode) for mode in sorted(dcdc_modes)],
        "dcac": [list(mode) for mode in sorted(dcac_modes)],
    }


def _validate_control_modes(modes: dict[str, list[list[str]]]) -> None:
    actual = {
        key: frozenset(tuple(mode) for mode in value)
        for key, value in modes.items()
    }
    expected = {
        "acac": EXPECTED_ACAC_MODES,
        "dcdc": EXPECTED_DCDC_MODES,
        "dcac": EXPECTED_DCAC_MODES,
    }
    if actual != expected:
        raise RuntimeError(f"converter control coverage mismatch: actual={actual}, expected={expected}")


def _lf_control_errors(calc: HybridPowerFlowCalc, residual: np.ndarray) -> dict[str, float]:
    errors: dict[str, float] = {}
    ac_calc = calc.ac_calc
    for pos in range(ac_calc.N_acac):
        legacy = ACAC_LEGACY_CONTROL_LABEL[int(ac_calc.acac_ctrl_code[pos])]
        mode = ACAC_LEGACY_TO_PAIR[legacy]
        rows = np.asarray(
            (
                ac_calc.acac_eq_loss[pos],
                ac_calc.acac_eq_ctrl_1[pos],
                ac_calc.acac_eq_ctrl_2[pos],
                ac_calc.acac_eq_ctrl_3[pos],
            ),
            dtype=np.int32,
        )
        errors[f"ACAC:{_mode_text(mode)}"] = _maximum(residual[rows])

    dc_calc = calc.dc_calc
    for pos in range(dc_calc.N_dcdc):
        mode = (
            DCDC_SIDE_CONTROL_LABEL[int(dc_calc.dcdc_i_ctrl_code[pos])],
            DCDC_SIDE_CONTROL_LABEL[int(dc_calc.dcdc_j_ctrl_code[pos])],
        )
        rows = [calc.ac_eq + int(dc_calc.dcdc_eq_ctrl[pos])]
        if dc_calc.dcdc_eq_loss.size:
            rows.append(calc.ac_eq + int(dc_calc.dcdc_eq_loss[pos]))
        errors[f"DCDC:{_mode_text(mode)}"] = _maximum(residual[np.asarray(rows, dtype=np.int32)])

    for pos in range(calc.N_dcac):
        mode = (
            DCAC_AC_CONTROL_LABEL[int(calc.dcac_ac_control_code[pos])],
            DCAC_DC_CONTROL_LABEL[int(calc.dcac_dc_control_code[pos])],
        )
        rows = np.asarray(
            (
                calc.dcac_eq_loss[pos],
                calc.dcac_eq_ctrl_1[pos],
                calc.dcac_eq_ctrl_2[pos],
            ),
            dtype=np.int32,
        )
        errors[f"DCAC:{_mode_text(mode)}"] = _maximum(residual[rows])
    return errors


def _switch_state_counts(table: np.ndarray, columns: dict[str, int]) -> dict[str, int]:
    table = np.asarray(table, dtype=np.float64)
    if table.size == 0:
        return {"total": 0, "closed": 0, "open": 0, "out_of_service": 0}
    online = table[:, columns["run_stat"]].astype(np.int8, copy=False) == 1
    closed = online & (table[:, columns["status"]].astype(np.int8, copy=False) == 1)
    return {
        "total": int(table.shape[0]),
        "closed": int(np.count_nonzero(closed)),
        "open": int(np.count_nonzero(online & ~closed)),
        "out_of_service": int(np.count_nonzero(~online)),
    }


def _network_statistics(calc: HybridPowerFlowCalc) -> dict[str, object]:
    ac_ppc = calc.ac_calc.ppc
    dc_ppc = calc.dc_calc.ppc
    dcac = np.asarray(calc.network.ppc.get("dcac", ()), dtype=np.float64)
    modes = _decode_control_modes(calc)
    _validate_control_modes(modes)
    dcac_device_types: dict[str, int] = {}
    for row in dcac:
        if int(row[DCAC_COLS["run_stat"]]) != 1:
            continue
        label = DCAC_DEVICE_TYPE_LABEL[int(row[DCAC_COLS["dev_type"]])]
        dcac_device_types[label] = dcac_device_types.get(label, 0) + 1
    return {
        "raw_nodes": int(ac_ppc["bus"].shape[0] + dc_ppc["bus"].shape[0]),
        "ac_nodes": int(ac_ppc["bus"].shape[0]),
        "ac_buses": int(calc.ac_calc.N),
        "ac_branches": int(ac_ppc["branch"].shape[0]),
        "ac_transformers": int(ac_ppc["transformer"].shape[0]),
        "ac_three_winding_transformers": int(ac_ppc["three_winding_transformer"].shape[0]),
        "ac_zero_branches": int(ac_ppc["zero_branch"].shape[0]),
        "ac_switches": _switch_state_counts(ac_ppc["switch"], AC_SWITCH_COLS),
        "ac_breakers": _switch_state_counts(ac_ppc["break"], AC_SWITCH_COLS),
        "ac_shunts": int(ac_ppc["shunt"].shape[0]),
        "ac_generators": int(ac_ppc["gen"].shape[0]),
        "ac_loads": int(ac_ppc["load"].shape[0]),
        "dc_nodes": int(dc_ppc["bus"].shape[0]),
        "dc_buses": int(calc.dc_calc.N),
        "dc_branches": int(dc_ppc["branch"].shape[0]),
        "dc_zero_branches": int(dc_ppc["zero_branch"].shape[0]),
        "dc_switches": _switch_state_counts(dc_ppc["switch"], DC_SWITCH_COLS),
        "dc_breakers": _switch_state_counts(dc_ppc["break"], DC_SWITCH_COLS),
        "dc_generators": int(dc_ppc["gen"].shape[0]),
        "dc_loads": int(dc_ppc["load"].shape[0]),
        "acac_count": int(calc.ac_calc.N_acac),
        "dcdc_count": int(calc.dc_calc.N_dcdc),
        "dcac_count": int(calc.N_dcac),
        "dcac_device_types": dcac_device_types,
        "converter_modes": modes,
        "lf_variables": int(calc.total_vars),
        "lf_equations": int(calc.total_eq),
        "linear_solver": str(calc._linear_solver_resolved),
    }


def benchmark_lf(case: Path) -> dict[str, object]:
    internal_start = time.perf_counter()
    stage_start = time.perf_counter()
    network = _read_lf_network_from_file(case)
    load_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    calc = HybridPowerFlowCalc(network, result_mode="full", verbose=False)
    construct_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prepare()
    prepare_time = time.perf_counter() - stage_start

    writeback_time = 0.0
    original_write_back = calc._write_back

    def timed_write_back():
        nonlocal writeback_time
        writeback_start = time.perf_counter()
        result = original_write_back()
        writeback_time += time.perf_counter() - writeback_start
        return result

    calc._write_back = timed_write_back
    stage_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        rc = calc.run()
    run_time = time.perf_counter() - stage_start
    solve_time = max(0.0, run_time - writeback_time)

    residual = calc.get_f(calc.x).copy()
    ac_x, dc_x, _dcac_x = calc._split_x(calc.x)
    _theta, ac_voltage, _phi_re, _phi_im = calc.ac_calc._extract_state_vars(ac_x)
    dc_voltage = dc_x[: calc.dc_calc.N]
    control_errors = _lf_control_errors(calc, residual)
    statistics = _network_statistics(calc)
    return {
        "algorithm": "Hybrid LF",
        "converged": bool(rc == 0 and calc.converged),
        "iterations": int(calc.iterations),
        "accuracy": {
            "newton_residual": _maximum(residual),
            "ac_subblock_residual": _maximum(residual[: calc.ac_eq]),
            "dc_subblock_residual": _maximum(residual[calc.ac_eq : calc.dcac_eq_start]),
            "dcac_coupling_residual": _maximum(residual[calc.dcac_eq_start :]),
            "control_equation_errors": control_errors,
            "max_control_equation_error": max(control_errors.values(), default=0.0),
            "ac_voltage_min": float(np.min(ac_voltage)),
            "ac_voltage_max": float(np.max(ac_voltage)),
            "dc_voltage_min": float(np.min(dc_voltage)),
            "dc_voltage_max": float(np.max(dc_voltage)),
            "dcdc_loss_infeasible_count": int(
                np.count_nonzero(calc.dc_calc.last_dcdc_loss_infeasible_mask)
            ),
        },
        "phases": {
            "load_network": load_time,
            "construct": construct_time,
            "prepare": prepare_time,
            "newton_solve": solve_time,
            "writeback": writeback_time,
        },
        "internal_total_s": time.perf_counter() - internal_start,
        "statistics": statistics,
    }


def _measurement_residual_metrics(result) -> dict[str, float | int]:
    table = result.measurement_table
    residual = np.abs(np.asarray(result.residual, dtype=np.float64))
    device_type = np.asarray(table.device_type_code, dtype=np.int16)
    measurement_type = np.asarray(table.meas_type_code, dtype=np.int64)
    # Generated LF measurements use weight 1.0.  Automatically added pseudo
    # rows use the configured lower-confidence value and are reported apart.
    original = np.isclose(np.asarray(table.weight, dtype=np.float64), 1.0)

    groups = {
        "original_measurements": original,
        "node_voltage": original
        & np.isin(device_type, (DEVICE_TYPE_ACNode, DEVICE_TYPE_DCNode))
        & (measurement_type == MEAS_TYPE_V),
        "ac_branch_flow": original
        & np.isin(
            device_type,
            (
                DEVICE_TYPE_ACBranch,
                DEVICE_TYPE_ACTransformer,
                DEVICE_TYPE_ACThreeWindingTransformer,
            ),
        )
        & np.isin(
            measurement_type,
            (
                MEAS_TYPE_P_FROM,
                MEAS_TYPE_Q_FROM,
                MEAS_TYPE_P_TO,
                MEAS_TYPE_Q_TO,
                MEAS_TYPE_P_THIRD,
                MEAS_TYPE_Q_THIRD,
            ),
        ),
        "dc_branch_flow": original
        & (device_type == DEVICE_TYPE_DCBranch)
        & np.isin(measurement_type, (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO)),
        "ac_ideal_branch": original
        & np.isin(device_type, (DEVICE_TYPE_ACZeroBranch, DEVICE_TYPE_ACBreak)),
        "dc_ideal_branch": original
        & np.isin(device_type, (DEVICE_TYPE_DCZeroBranch, DEVICE_TYPE_DCBreak)),
        "acac_converter": original & (device_type == DEVICE_TYPE_ACACConverter),
        "dcdc_converter": original & (device_type == DEVICE_TYPE_DCDCConverter),
        "dcac_converter": original & (device_type == DEVICE_TYPE_DCACConverter),
        "pseudo_measurements": ~original,
    }
    metrics: dict[str, float | int] = {}
    for name, mask in groups.items():
        metrics[f"{name}_count"] = int(np.count_nonzero(mask))
        metrics[f"{name}_max_residual"] = _maximum(residual[mask])
    return metrics


def benchmark_se(case: Path, measurements: Path) -> dict[str, object]:
    internal_start = time.perf_counter()
    stage_start = time.perf_counter()
    estimator = HybridStateEstimator(
        case,
        measurements,
        flat_start=True,
        max_iter=50,
        profile=True,
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
        final_diagnostics=False,
    )
    estimate_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    estimator.write_back(result, result_mode="summary")
    diagnostics_time = time.perf_counter() - stage_start
    residual_metrics = _measurement_residual_metrics(result)
    return {
        "algorithm": "Hybrid SE",
        "converged": bool(observability.observable and result.converged),
        "iterations": int(result.iterations),
        "accuracy": {
            "observable": bool(observability.observable),
            "rank": int(observability.rank),
            "state_count": int(observability.state_count),
            "objective": float(result.objective),
            "residual_inf": float(result.residual_inf),
            "bad_data_count": int(len(estimator.bad_items)),
            **residual_metrics,
        },
        "phases": {
            "construct": construct_time,
            "prepare": prepare_time,
            "observability": observability_time,
            "wls_estimate": estimate_time,
            "diagnostics_result": diagnostics_time,
        },
        "internal_total_s": time.perf_counter() - internal_start,
        "statistics": {
            "se_states": int(observability.state_count),
            "se_active_measurements": int(len(estimator.active_measurements)),
            "se_original_measurements": int(residual_metrics["original_measurements_count"]),
            "se_pseudo_measurements": int(residual_metrics["pseudo_measurements_count"]),
            "se_rank": int(observability.rank),
            "profile_times": {key: float(value) for key, value in estimator.profile_times.items()},
        },
    }


def _worker(kind: str, case: Path, measurements: Path) -> int:
    if kind == "lf":
        payload = benchmark_lf(case)
    elif kind == "se":
        payload = benchmark_se(case, measurements)
    else:
        raise ValueError(f"unknown worker kind: {kind}")
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"{kind} worker failed with exit code {completed.returncode}:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError(f"{kind} worker returned no JSON output")
    payload = json.loads(output_lines[-1])
    payload["end_to_end_s"] = elapsed
    payload["startup_import_overhead_s"] = max(0.0, elapsed - float(payload["internal_total_s"]))
    return payload


def _time_summary(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "average": statistics.fmean(data),
        "minimum": min(data),
        "maximum": max(data),
    }


def _aggregate(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "runs": len(runs),
        "converged_runs": sum(bool(run["converged"]) for run in runs),
        "iterations": _time_summary(float(run["iterations"]) for run in runs),
        "end_to_end_s": _time_summary(float(run["end_to_end_s"]) for run in runs),
        "internal_total_s": _time_summary(float(run["internal_total_s"]) for run in runs),
        "startup_import_overhead_s": _time_summary(
            float(run["startup_import_overhead_s"]) for run in runs
        ),
        "phase_average_s": {
            phase: statistics.fmean(float(run["phases"][phase]) for run in runs)
            for phase in runs[0]["phases"]
        },
    }


def _print_report(results: dict[str, list[dict[str, object]]]) -> None:
    first_lf = results["lf"][0]
    first_se = results["se"][0]
    stats = first_lf["statistics"]
    modes = stats["converter_modes"]

    print("\nNetwork statistics")
    print(
        f"  raw nodes={stats['raw_nodes']} | "
        f"AC nodes/buses/branches={stats['ac_nodes']}/{stats['ac_buses']}/{stats['ac_branches']} | "
        f"DC nodes/buses/branches={stats['dc_nodes']}/{stats['dc_buses']}/{stats['dc_branches']}"
    )
    print(
        f"  converters: ACAC={stats['acac_count']} {[_mode_text(x) for x in modes['acac']]}; "
        f"DCDC={stats['dcdc_count']} {[_mode_text(x) for x in modes['dcdc']]}; "
        f"DCAC/ACDC={stats['dcac_count']} {stats['dcac_device_types']} "
        f"{[_mode_text(x) for x in modes['dcac']]}"
    )
    print(
        "  AC elements: "
        f"zero={stats['ac_zero_branches']}, switch={stats['ac_switches']}, "
        f"breaker={stats['ac_breakers']}, shunt={stats['ac_shunts']}, "
        f"2-winding={stats['ac_transformers']}, "
        f"3-winding={stats['ac_three_winding_transformers']}, "
        f"generator/load={stats['ac_generators']}/{stats['ac_loads']}"
    )
    print(
        "  DC elements: "
        f"zero={stats['dc_zero_branches']}, switch={stats['dc_switches']}, "
        f"breaker={stats['dc_breakers']}, "
        f"generator/load={stats['dc_generators']}/{stats['dc_loads']}"
    )
    print(
        f"  LF variables/equations={stats['lf_variables']}/{stats['lf_equations']} "
        f"({stats['linear_solver']}); SE states/measurements/rank="
        f"{first_se['statistics']['se_states']}/"
        f"{first_se['statistics']['se_active_measurements']}/"
        f"{first_se['statistics']['se_rank']}"
    )

    print("\nIndependent-process timing")
    print("  Algorithm | Runs | Converged | Iter avg | End-to-end avg | Min | Max")
    print("  ----------|------|-----------|----------|----------------|-----|----")
    for kind in ("lf", "se"):
        summary = _aggregate(results[kind])
        name = results[kind][0]["algorithm"]
        end = summary["end_to_end_s"]
        print(
            f"  {name} | {summary['runs']} | {summary['converged_runs']}/{summary['runs']} | "
            f"{summary['iterations']['average']:.1f} | {end['average']:.6f}s | "
            f"{end['minimum']:.6f}s | {end['maximum']:.6f}s"
        )

    print("\nAverage stage timing")
    for kind in ("lf", "se"):
        summary = _aggregate(results[kind])
        phases = ", ".join(
            f"{name}={value:.6f}s" for name, value in summary["phase_average_s"].items()
        )
        print(
            f"  {results[kind][0]['algorithm']}: {phases}, "
            f"startup/import={summary['startup_import_overhead_s']['average']:.6f}s"
        )

    lf_accuracy = first_lf["accuracy"]
    se_accuracy = first_se["accuracy"]
    print("\nAccuracy")
    print(
        "  Hybrid LF: "
        f"Newton={lf_accuracy['newton_residual']:.3e}, "
        f"AC={lf_accuracy['ac_subblock_residual']:.3e}, "
        f"DC={lf_accuracy['dc_subblock_residual']:.3e}, "
        f"DCAC={lf_accuracy['dcac_coupling_residual']:.3e}, "
        f"max control error={lf_accuracy['max_control_equation_error']:.3e}"
    )
    print(
        "  Hybrid SE: "
        f"observable={se_accuracy['observable']} ({se_accuracy['rank']}/{se_accuracy['state_count']}), "
        f"objective={se_accuracy['objective']:.3e}, residual_inf={se_accuracy['residual_inf']:.3e}, "
        f"bad data={se_accuracy['bad_data_count']}"
    )
    print(
        "  SE original-measurement max residuals: "
        f"node V={se_accuracy['node_voltage_max_residual']:.3e}, "
        f"AC branch flow={se_accuracy['ac_branch_flow_max_residual']:.3e}, "
        f"DC branch flow={se_accuracy['dc_branch_flow_max_residual']:.3e}, "
        f"AC zero/break={se_accuracy['ac_ideal_branch_max_residual']:.3e}, "
        f"DC zero/break={se_accuracy['dc_ideal_branch_max_residual']:.3e}, "
        f"ACAC={se_accuracy['acac_converter_max_residual']:.3e}, "
        f"DCDC={se_accuracy['dcdc_converter_max_residual']:.3e}, "
        f"DCAC={se_accuracy['dcac_converter_max_residual']:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=("lf", "se"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    case = args.case.resolve()
    measurements = args.measurements.resolve()
    if args.worker:
        return _worker(args.worker, case, measurements)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not case.exists() or not measurements.exists():
        parser.error("generate the benchmark E/meas files before running the benchmark")

    order = [kind for _ in range(args.runs) for kind in ("lf", "se")]
    random.Random(args.seed).shuffle(order)
    results: dict[str, list[dict[str, object]]] = {"lf": [], "se": []}
    for sequence, kind in enumerate(order, start=1):
        run = _run_process(kind, case, measurements)
        results[kind].append(run)
        print(
            f"[{sequence}/{len(order)}] {run['algorithm']}: "
            f"converged={run['converged']}, iter={run['iterations']}, "
            f"end-to-end={run['end_to_end_s']:.6f}s"
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    _print_report(results)
    return 0 if all(run["converged"] for runs in results.values() for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
