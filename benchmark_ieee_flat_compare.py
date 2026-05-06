import argparse
import contextlib
import io
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    T_BUS,
    TAP,
)
from pypower.idx_bus import BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE, GS, PD, QD, VA, VM, VMAX, VMIN, ZONE
from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG


ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import (  # noqa: E402
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    SHUNT_COLS,
    SWITCH_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
    build_ac_ppc_from_e_file,
)
from ac_lf import ACPowerFlowCalc  # noqa: E402


PQ = 1
PV = 2
REF = 3


class DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _run_silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _build_matpower_ppc(acppc):
    """Convert the array E-file model into a PYPOWER/MATPOWER ppc.

    Ideal zero-impedance branches and closed switches are represented by
    collapsing their terminal buses. That gives PYPOWER the same electrical
    network without stamping singular zero-impedance admittances.
    """
    base_mva = float(acppc["base"][0])
    bus0 = np.asarray(acppc["bus"])
    branch0 = np.asarray(acppc["branch"])
    transformer0 = np.asarray(acppc["transformer"])
    gen0 = np.asarray(acppc["gen"])
    load0 = np.asarray(acppc["load"])
    shunt0 = np.asarray(acppc["shunt"])
    zero0 = np.asarray(acppc["zero_branch"])
    switch0 = np.asarray(acppc["switch"])

    node_ids = bus0[:, BUS_COLS["idx"]].astype(np.int64)
    row_by_node = {int(node): row for row, node in enumerate(node_ids)}
    active_bus = bus0[:, BUS_COLS["run_stat"]] == 1

    dsu = DSU(bus0.shape[0])
    if zero0.size:
        live = zero0[:, ZERO_BRANCH_COLS["run_stat"]] == 1
        for row in zero0[live]:
            left = row_by_node.get(int(row[ZERO_BRANCH_COLS["i_node"]]))
            right = row_by_node.get(int(row[ZERO_BRANCH_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)
    if switch0.size:
        live = (switch0[:, SWITCH_COLS["run_stat"]] == 1) & (switch0[:, SWITCH_COLS["status"]] == 1)
        for row in switch0[live]:
            left = row_by_node.get(int(row[SWITCH_COLS["i_node"]]))
            right = row_by_node.get(int(row[SWITCH_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)

    active_rows = np.flatnonzero(active_bus)
    root_to_comp = {}
    comp_rows = []
    for row in active_rows:
        root = dsu.find(int(row))
        if root not in root_to_comp:
            root_to_comp[root] = len(comp_rows)
            comp_rows.append([])
        comp_rows[root_to_comp[root]].append(int(row))

    row_to_comp = np.full(bus0.shape[0], -1, dtype=np.int64)
    for comp, rows in enumerate(comp_rows):
        row_to_comp[rows] = comp
    comp_to_bus_id = np.arange(1, len(comp_rows) + 1, dtype=np.int64)
    row_to_bus_id = np.where(row_to_comp >= 0, comp_to_bus_id[np.maximum(row_to_comp, 0)], -1)

    pd = np.zeros(len(comp_rows), dtype=float)
    qd = np.zeros(len(comp_rows), dtype=float)
    gs = np.zeros(len(comp_rows), dtype=float)
    bs = np.zeros(len(comp_rows), dtype=float)
    bus_type = np.full(len(comp_rows), PQ, dtype=float)
    base_kv = np.zeros(len(comp_rows), dtype=float)
    for comp, rows in enumerate(comp_rows):
        base_kv[comp] = bus0[rows[0], BUS_COLS["vbase"]]

    if load0.size:
        for row in load0[load0[:, LOAD_COLS["run_stat"]] == 1]:
            bus_row = row_by_node.get(int(row[LOAD_COLS["node"]]))
            if bus_row is None or row_to_comp[bus_row] < 0:
                continue
            comp = row_to_comp[bus_row]
            pd[comp] += (row[LOAD_COLS["pv0"]] + row[LOAD_COLS["pv1"]] + row[LOAD_COLS["pv2"]]) * base_mva
            qd[comp] += (row[LOAD_COLS["qv0"]] + row[LOAD_COLS["qv1"]] + row[LOAD_COLS["qv2"]]) * base_mva

    if shunt0.size:
        for row in shunt0[shunt0[:, SHUNT_COLS["run_stat"]] == 1]:
            bus_row = row_by_node.get(int(row[SHUNT_COLS["node"]]))
            if bus_row is None or row_to_comp[bus_row] < 0:
                continue
            comp = row_to_comp[bus_row]
            gs[comp] += row[SHUNT_COLS["g_set"]] * base_mva
            bs[comp] += row[SHUNT_COLS["b_set"]] * base_mva

    if gen0.size:
        for row in gen0[gen0[:, GEN_COLS["run_stat"]] == 1]:
            bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
            if bus_row is None or row_to_comp[bus_row] < 0:
                continue
            comp = row_to_comp[bus_row]
            control = int(row[GEN_COLS["control_type"]])
            if control == CTRL_SLACK:
                bus_type[comp] = REF
            elif bus_type[comp] != REF and control in (CTRL_PV, CTRL_P):
                bus_type[comp] = PV

    bus = np.zeros((len(comp_rows), 13), dtype=float)
    bus[:, BUS_I] = comp_to_bus_id
    bus[:, BUS_TYPE] = bus_type
    bus[:, PD] = pd
    bus[:, QD] = qd
    bus[:, GS] = gs
    bus[:, BS] = bs
    bus[:, BUS_AREA] = 1
    bus[:, VM] = 1.0
    bus[:, VA] = 0.0
    bus[:, BASE_KV] = base_kv
    bus[:, ZONE] = 1
    bus[:, VMAX] = 1.2
    bus[:, VMIN] = 0.8

    gen_rows = []
    active_gen_rows = []
    if gen0.size:
        for idx, row in enumerate(gen0):
            if row[GEN_COLS["run_stat"]] != 1:
                continue
            bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
            if bus_row is None or row_to_comp[bus_row] < 0:
                continue
            gen_row = np.zeros(21, dtype=float)
            gen_row[GEN_BUS] = row_to_bus_id[bus_row]
            gen_row[PG] = row[GEN_COLS["p_set"]] * base_mva
            gen_row[QG] = row[GEN_COLS["q_set"]] * base_mva
            gen_row[QMAX] = 1e9
            gen_row[QMIN] = -1e9
            gen_row[VG] = row[GEN_COLS["v_set"]]
            gen_row[MBASE] = base_mva
            gen_row[GEN_STATUS] = 1
            gen_row[PMAX] = 1e9
            gen_row[PMIN] = -1e9
            gen_rows.append(gen_row)
            active_gen_rows.append(idx)
    gen = np.vstack(gen_rows) if gen_rows else np.zeros((0, 21), dtype=float)

    branch_rows = []
    branch_map = []

    def add_devices(devices, cols, kind):
        if not devices.size:
            return
        for idx, row in enumerate(devices):
            if row[cols["run_stat"]] != 1:
                continue
            i_row = row_by_node.get(int(row[cols["i_node"]]))
            j_row = row_by_node.get(int(row[cols["j_node"]]))
            if i_row is None or j_row is None or row_to_comp[i_row] < 0 or row_to_comp[j_row] < 0:
                continue
            i_bus = row_to_bus_id[i_row]
            j_bus = row_to_bus_id[j_row]
            if i_bus == j_bus:
                continue
            branch_row = np.zeros(13, dtype=float)
            branch_row[F_BUS] = i_bus
            branch_row[T_BUS] = j_bus
            branch_row[BR_R] = row[cols["r"]]
            branch_row[BR_X] = row[cols["x"]]
            branch_row[BR_B] = row[cols["b"]]
            branch_row[RATE_A] = 0.0
            branch_row[RATE_B] = 0.0
            branch_row[RATE_C] = 0.0
            if kind == "transformer":
                tap = row[cols["tap"]]
                branch_row[TAP] = 0.0 if abs(tap - 1.0) < 1e-12 else tap
                branch_row[SHIFT] = row[cols["shift"]]
            branch_row[BR_STATUS] = 1
            branch_row[ANGMIN] = -360.0
            branch_row[ANGMAX] = 360.0
            branch_rows.append(branch_row)
            branch_map.append((kind, idx))

    add_devices(branch0, BRANCH_COLS, "branch")
    add_devices(transformer0, TRANSFORMER_COLS, "transformer")
    branch = np.vstack(branch_rows) if branch_rows else np.zeros((0, 13), dtype=float)
    ppc = {"version": "2", "baseMVA": base_mva, "bus": bus, "gen": gen, "branch": branch}
    return ppc, row_to_bus_id, branch_map, np.asarray(active_gen_rows, dtype=np.int64)


def _run_matpower(ppc):
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-8, PF_MAX_IT=50, ENFORCE_Q_LIMS=False)
    start = time.perf_counter()
    result, success = _run_silent(runpf, ppc, options)
    return result, bool(success), time.perf_counter() - start


def _run_ours(acppc):
    calc = ACPowerFlowCalc.from_ppc(acppc, tol=1e-8, max_iter=50)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prepare()
        rc = calc.run()
    return calc, rc, time.perf_counter() - start


def _angle_diff_deg(left, right):
    return (left - right + 180.0) % 360.0 - 180.0


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _accuracy(acppc, calc, mp_result, row_to_bus_id, branch_map, active_gen_rows):
    base_mva = float(acppc["base"][0])
    mp_bus_by_id = {int(row[BUS_I]): row for row in mp_result["bus"]}
    our_bus = calc.result["bus"]
    active_bus = acppc["bus"][:, BUS_COLS["run_stat"]] == 1

    vm_err = []
    va_err = []
    for row in np.flatnonzero(active_bus):
        bus_id = int(row_to_bus_id[row])
        if bus_id <= 0:
            continue
        mp_bus = mp_bus_by_id[bus_id]
        vm_err.append(our_bus[row, BUS_COLS["voltage"]] - mp_bus[VM])
        va_err.append(_angle_diff_deg(math.degrees(our_bus[row, BUS_COLS["angle"]]), mp_bus[VA]))

    flow_p_err = []
    flow_q_err = []
    our_branch = calc.result["branch"]
    our_transformer = calc.result["transformer"]
    for mp_row, (kind, idx) in zip(mp_result["branch"], branch_map):
        if kind == "branch":
            row = our_branch[idx]
            values = (
                row[BRANCH_COLS["i_p"]],
                row[BRANCH_COLS["i_q"]],
                row[BRANCH_COLS["j_p"]],
                row[BRANCH_COLS["j_q"]],
            )
        else:
            row = our_transformer[idx]
            values = (
                row[TRANSFORMER_COLS["i_p"]],
                row[TRANSFORMER_COLS["i_q"]],
                row[TRANSFORMER_COLS["j_p"]],
                row[TRANSFORMER_COLS["j_q"]],
            )
        flow_p_err.extend([values[0] - mp_row[PF] / base_mva, values[2] - mp_row[PT] / base_mva])
        flow_q_err.extend([values[1] - mp_row[QF] / base_mva, values[3] - mp_row[QT] / base_mva])

    mp_pg = defaultdict(float)
    mp_qg = defaultdict(float)
    for row in mp_result["gen"]:
        if row[GEN_STATUS] > 0:
            mp_pg[int(row[GEN_BUS])] += row[PG] / base_mva
            mp_qg[int(row[GEN_BUS])] += row[QG] / base_mva

    node_to_row = {int(node): row for row, node in enumerate(acppc["bus"][:, BUS_COLS["idx"]].astype(np.int64))}
    our_pg = defaultdict(float)
    our_qg = defaultdict(float)
    our_gen = calc.result["gen"]
    for idx in active_gen_rows:
        node = int(acppc["gen"][idx, GEN_COLS["node"]])
        bus_row = node_to_row[node]
        bus_id = int(row_to_bus_id[bus_row])
        our_pg[bus_id] += our_gen[idx, GEN_COLS["p"]]
        our_qg[bus_id] += our_gen[idx, GEN_COLS["q"]]

    gen_buses = sorted(set(mp_pg) | set(our_pg))
    pg_err = [our_pg[bus] - mp_pg[bus] for bus in gen_buses]
    qg_err = [our_qg[bus] - mp_qg[bus] for bus in gen_buses]

    return {
        "vm": _max_abs(vm_err),
        "va": _max_abs(va_err),
        "flow_p": _max_abs(flow_p_err),
        "flow_q": _max_abs(flow_q_err),
        "pg": _max_abs(pg_err),
        "qg": _max_abs(qg_err),
    }


def _bench_case(case_name: str):
    e_file = ROOT_DIR / "data" / "ac" / f"{case_name}.e"
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    start = time.perf_counter()
    acppc = build_ac_ppc_from_e_file(e_file, use_cache=True, copy_arrays=False)
    e_build_s = time.perf_counter() - start

    start = time.perf_counter()
    matpower_ppc, row_to_bus_id, branch_map, active_gen_rows = _build_matpower_ppc(acppc)
    matpower_build_s = time.perf_counter() - start

    matpower_result, matpower_success, matpower_s = _run_matpower(matpower_ppc)
    calc, rc, ours_s = _run_ours(acppc)
    accuracy = _accuracy(acppc, calc, matpower_result, row_to_bus_id, branch_map, active_gen_rows)
    return {
        "case": case_name,
        "e_buses": int(acppc["bus"].shape[0]),
        "mp_buses": int(matpower_ppc["bus"].shape[0]),
        "mp_branches": int(matpower_ppc["branch"].shape[0]),
        "gens": int(matpower_ppc["gen"].shape[0]),
        "matpower_success": matpower_success,
        "matpower_s": matpower_s,
        "ours_rc": int(rc),
        "ours_converged": bool(calc.converged),
        "ours_iter": int(calc.iterations),
        "ours_normF": float(calc.normF),
        "ours_s": ours_s,
        "e_build_s": e_build_s,
        "matpower_build_s": matpower_build_s,
        **accuracy,
    }


def _print_rows(results):
    headers = [
        "case",
        "E buses",
        "MP buses",
        "MP s",
        "ours s",
        "speedup",
        "iter",
        "normF",
        "Vm err",
        "Va err(deg)",
        "Flow P err",
        "Flow Q err",
        "Pg err",
        "Qg err",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["case"],
                result["e_buses"],
                result["mp_buses"],
                f'{result["matpower_s"]:.6f}',
                f'{result["ours_s"]:.6f}',
                f'{result["matpower_s"] / result["ours_s"]:.2f}x' if result["ours_s"] > 0 else "inf",
                result["ours_iter"],
                f'{result["ours_normF"]:.3e}',
                f'{result["vm"]:.3e}',
                f'{result["va"]:.3e}',
                f'{result["flow_p"]:.3e}',
                f'{result["flow_q"]:.3e}',
                f'{result["pg"]:.3e}',
                f'{result["qg"]:.3e}',
            ]
        )
    widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    print(" | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(row)))


def main():
    parser = argparse.ArgumentParser(description="Flat-start IEEE AC benchmark against PYPOWER/MATPOWER runpf.")
    parser.add_argument("cases", nargs="*", default=["ieee300", "ieee3k", "ieee1w", "ieee3w"])
    args = parser.parse_args()

    results = []
    for case_name in args.cases:
        result = _bench_case(case_name)
        results.append(result)
        print(
            "CASE "
            f"{case_name}: matpower_success={result['matpower_success']} "
            f"ours_converged={result['ours_converged']} "
            f"matpower_s={result['matpower_s']:.6f} ours_s={result['ours_s']:.6f} "
            f"Vm_err={result['vm']:.3e} Va_err={result['va']:.3e}",
            flush=True,
        )
    print()
    _print_rows(results)


if __name__ == "__main__":
    main()
