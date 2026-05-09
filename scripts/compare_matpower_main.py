import argparse
import contextlib
import io
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_brch import BR_STATUS, F_BUS, PF, PT, QF, QT, T_BUS
from pypower.idx_bus import BS, BUS_I, GS, PD, QD, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS


ROOT_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ac_net_flow import ACPowerFlowCalc
from ac_net_model import ACPowerNetwork
from hybrid_net_flow import run_hybrid_power_flow


CASES = (
    ("ieee9", "case9", ROOT_DIR / "data" / "model" / "ac" / "ieee9.e"),
    ("ieee14", "case14", ROOT_DIR / "data" / "model" / "ac" / "ieee14.e"),
    ("ieee24", "case24_ieee_rts", ROOT_DIR / "data" / "model" / "ac" / "ieee24.e"),
    ("ieee30", "case30", ROOT_DIR / "data" / "model" / "ac" / "ieee30.e"),
    ("ieee39", "case39", ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"),
    ("ieee57", "case57", ROOT_DIR / "data" / "model" / "ac" / "ieee57.e"),
    ("ieee118", "case118", ROOT_DIR / "data" / "model" / "ac" / "ieee118.e"),
    ("ieee300", "case300", ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"),
)


def _quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _bus_id(name):
    if not str(name).startswith("bus_"):
        raise ValueError(f"Cannot map node name to MATPOWER bus id: {name}")
    return int(str(name).split("_", 1)[1])


def _angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _case_func(case_name):
    module = __import__(f"pypower.{case_name}", fromlist=[case_name])
    return getattr(module, case_name)


def _run_matpower_reference(case_name):
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-8, PF_MAX_IT=50, ENFORCE_Q_LIMS=False)
    result, success = _quiet_call(runpf, _case_func(case_name)(), options)
    if not success:
        raise RuntimeError("PYPOWER/MATPOWER-compatible runpf did not converge")
    return result


def _run_ac_flow(e_file):
    net = ACPowerNetwork()
    net.read_from_file(e_file)
    net.topo()
    calc = ACPowerFlowCalc(net, tol=1e-8, max_iter=50)
    _quiet_call(calc.prepare)
    rc = _quiet_call(calc.run)
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"ac_flow.py did not converge for {e_file}")
    return net, calc


def _run_hybrid_flow(e_file):
    result = run_hybrid_power_flow(e_file, tol=1e-8, max_iter=50, verbose=False)
    if not result.converged:
        raise RuntimeError(
            f"hybrid_flow.py did not converge for {e_file}: "
            f"ac_errors={result.ac_errors}, dc_errors={result.dc_errors}"
        )
    return result.ac_network, result.ac, result


def _extract_ref(mp_result):
    base_mva = float(mp_result["baseMVA"])
    bus = {int(row[BUS_I]): row for row in mp_result["bus"]}

    active_gen = mp_result["gen"][mp_result["gen"][:, GEN_STATUS] > 0]
    gen_buses = {int(row[GEN_BUS]) for row in active_gen}
    branch_p = defaultdict(float)
    branch_q = defaultdict(float)
    active_branch = mp_result["branch"][mp_result["branch"][:, BR_STATUS] > 0]
    for row in active_branch:
        f_bus = int(row[F_BUS])
        t_bus = int(row[T_BUS])
        branch_p[f_bus] += row[PF] / base_mva
        branch_q[f_bus] += row[QF] / base_mva
        branch_p[t_bus] += row[PT] / base_mva
        branch_q[t_bus] += row[QT] / base_mva

    pg = {}
    qg = {}
    for bus_id in gen_buses:
        row = bus[bus_id]
        vm = row[VM]
        pg[bus_id] = branch_p[bus_id] + row[PD] / base_mva + row[GS] / base_mva * vm * vm
        qg[bus_id] = branch_q[bus_id] + row[QD] / base_mva - row[BS] / base_mva * vm * vm

    for row in active_gen:
        bus_id = int(row[GEN_BUS])
        pg.setdefault(bus_id, 0.0)
        qg.setdefault(bus_id, 0.0)

    p_loss = float(np.sum(active_branch[:, PF] + active_branch[:, PT]) / base_mva)
    q_loss = float(np.sum(active_branch[:, QF] + active_branch[:, QT]) / base_mva)
    return bus, pg, qg, p_loss, q_loss


def _accuracy_against_ref(network, calc, mp_result):
    mp_bus, mp_pg, mp_qg, mp_p_loss, mp_q_loss = _extract_ref(mp_result)

    vm_diff = []
    va_diff = []
    for node in calc.node_list:
        bus_id = _bus_id(node.name)
        row = mp_bus[bus_id]
        vm_diff.append(node.voltage - row[VM])
        va_diff.append(_angle_diff_deg(node.angle * 180.0 / math.pi, row[VA]))

    node_name = {node.idx: node.name for node in network.nodes}
    main_pg = defaultdict(float)
    main_qg = defaultdict(float)
    for gen in network.generators:
        if getattr(gen, "is_alive", False):
            bus_id = _bus_id(node_name[gen.node])
            main_pg[bus_id] += gen.p
            main_qg[bus_id] += gen.q

    gen_buses = sorted(set(mp_pg) | set(main_pg))
    pg_diff = [main_pg[bus] - mp_pg[bus] for bus in gen_buses]
    qg_diff = [main_qg[bus] - mp_qg[bus] for bus in gen_buses]

    main_p_loss = sum(br.i_p + br.j_p for br in network.branches if getattr(br, "is_alive", False))
    main_q_loss = sum(br.i_q + br.j_q for br in network.branches if getattr(br, "is_alive", False))
    main_p_loss += sum(tr.i_p + tr.j_p for tr in network.transformers if getattr(tr, "is_alive", False))
    main_q_loss += sum(tr.i_q + tr.j_q for tr in network.transformers if getattr(tr, "is_alive", False))

    return {
        "vm_max": _max_abs(vm_diff),
        "va_max": _max_abs(va_diff),
        "pg_max": _max_abs(pg_diff),
        "qg_max": _max_abs(qg_diff),
        "p_loss_diff": abs(main_p_loss - mp_p_loss),
        "q_loss_diff": abs(main_q_loss - mp_q_loss),
        "iterations": calc.iterations,
        "normF": calc.normF,
    }


def _time_runner(func, repeat):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "min_ms": min(times),
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "max_ms": max(times),
    }


def _bench_case(name, case_name, e_file, repeat):
    mp_result = _run_matpower_reference(case_name)
    ac_network, ac_calc = _run_ac_flow(e_file)
    hybrid_network, hybrid_calc, hybrid_result = _run_hybrid_flow(e_file)

    matpower_time = _time_runner(lambda: _run_matpower_reference(case_name), repeat)
    ac_time = _time_runner(lambda: _run_ac_flow(e_file), repeat)
    hybrid_time = _time_runner(lambda: _run_hybrid_flow(e_file), repeat)

    return {
        "case": name,
        "matpower_time": matpower_time,
        "ac_time": ac_time,
        "hybrid_time": hybrid_time,
        "ac_accuracy": _accuracy_against_ref(ac_network, ac_calc, mp_result),
        "hybrid_accuracy": _accuracy_against_ref(hybrid_network, hybrid_calc, mp_result),
        "hybrid_shape": hybrid_result.global_jacobian_shape,
    }


def _fmt(value, digits=3):
    return f"{value:.{digits}e}"


def _print_table(title, rows, headers):
    print(title)
    widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row)))
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark IEEE AC E files against PYPOWER/MATPOWER-compatible runpf.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing repeats per solver and case.")
    args = parser.parse_args()

    results = [_bench_case(*case, repeat=args.repeat) for case in CASES]

    time_rows = []
    for result in results:
        time_rows.append(
            [
                result["case"],
                f'{result["matpower_time"]["median_ms"]:.3f}',
                f'{result["ac_time"]["median_ms"]:.3f}',
                f'{result["hybrid_time"]["median_ms"]:.3f}',
            ]
        )
    _print_table("median runtime, ms", time_rows, ["case", "matpower", "ac_flow.py", "hybrid_flow.py"])

    detail_time_rows = []
    for result in results:
        for solver_key, label in (
            ("matpower_time", "matpower"),
            ("ac_time", "ac_flow.py"),
            ("hybrid_time", "hybrid_flow.py"),
        ):
            timing = result[solver_key]
            detail_time_rows.append(
                [
                    result["case"],
                    label,
                    f'{timing["min_ms"]:.3f}',
                    f'{timing["median_ms"]:.3f}',
                    f'{timing["mean_ms"]:.3f}',
                    f'{timing["max_ms"]:.3f}',
                ]
            )
    _print_table("runtime detail, ms", detail_time_rows, ["case", "solver", "min", "median", "mean", "max"])

    accuracy_rows = []
    for result in results:
        for accuracy_key, label in (
            ("ac_accuracy", "ac_flow.py"),
            ("hybrid_accuracy", "hybrid_flow.py"),
        ):
            acc = result[accuracy_key]
            accuracy_rows.append(
                [
                    result["case"],
                    label,
                    acc["iterations"],
                    _fmt(acc["normF"]),
                    _fmt(acc["vm_max"]),
                    _fmt(acc["va_max"]),
                    _fmt(acc["pg_max"]),
                    _fmt(acc["qg_max"]),
                    _fmt(acc["p_loss_diff"]),
                    _fmt(acc["q_loss_diff"]),
                ]
            )
    _print_table(
        "max absolute error vs matpower reference",
        accuracy_rows,
        ["case", "solver", "iter", "normF", "Vm(pu)", "Va(deg)", "Pg(pu)", "Qg(pu)", "Ploss(pu)", "Qloss(pu)"],
    )

    shape_rows = [[result["case"], str(result["hybrid_shape"])] for result in results]
    _print_table("hybrid global Jacobian shape", shape_rows, ["case", "shape"])


if __name__ == "__main__":
    main()
