import contextlib
import io
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_bus import BUS_I, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_ieee_flat_compare import (  # noqa: E402
    BUS_COLS,
    GEN_COLS,
    _accuracy,
    _build_matpower_ppc,
    _run_ours,
    build_ac_ppc_from_e_file,
)
from hybrid_lf import HybridPowerFlowCalc, HybridPowerFlowResult, _read_lf_network_from_file  # noqa: E402


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _run_matpower(ppc):
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-8, PF_MAX_IT=50, ENFORCE_Q_LIMS=False)
    start = time.perf_counter()
    result, success = _silent(runpf, ppc, options)
    return result, bool(success), time.perf_counter() - start


def _angle_diff_deg(left, right):
    return (left - right + 180.0) % 360.0 - 180.0


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _accuracy_hybrid(acppc, hybrid_result, mp_result, row_to_bus_id, branch_map, active_gen_rows):
    base_mva = float(acppc["base"][0])
    mp_bus_by_id = {int(row[BUS_I]): row for row in mp_result["bus"]}
    vm_err = []
    va_err = []
    node_by_idx = {int(node.idx): node for node in hybrid_result.ac_network.nodes}
    active_bus = acppc["bus"][:, BUS_COLS["run_stat"]] == 1
    for row in np.flatnonzero(active_bus):
        node_id = int(acppc["bus"][row, BUS_COLS["idx"]])
        bus_id = int(row_to_bus_id[row])
        if bus_id <= 0 or node_id not in node_by_idx:
            continue
        node = node_by_idx[node_id]
        mp_bus = mp_bus_by_id[bus_id]
        vm_err.append(float(node.voltage) - mp_bus[VM])
        va_err.append(_angle_diff_deg(math.degrees(float(node.angle)), mp_bus[VA]))

    mp_pg = defaultdict(float)
    mp_qg = defaultdict(float)
    for row in mp_result["gen"]:
        if row[GEN_STATUS] > 0:
            mp_pg[int(row[GEN_BUS])] += row[PG] / base_mva
            mp_qg[int(row[GEN_BUS])] += row[QG] / base_mva

    node_to_row = {int(node): row for row, node in enumerate(acppc["bus"][:, BUS_COLS["idx"]].astype(np.int64))}
    our_pg = defaultdict(float)
    our_qg = defaultdict(float)
    gen_by_idx = {int(gen.idx): gen for gen in hybrid_result.ac_network.generators}
    for idx in active_gen_rows:
        node = int(acppc["gen"][idx, GEN_COLS["node"]])
        bus_row = node_to_row[node]
        bus_id = int(row_to_bus_id[bus_row])
        gen = gen_by_idx.get(int(acppc["gen"][idx, GEN_COLS["idx"]]))
        if gen is None:
            continue
        our_pg[bus_id] += float(getattr(gen, "p", 0.0) or 0.0)
        our_qg[bus_id] += float(getattr(gen, "q", 0.0) or 0.0)

    gen_buses = sorted(set(mp_pg) | set(our_pg))
    pg_err = [our_pg[bus] - mp_pg[bus] for bus in gen_buses]
    qg_err = [our_qg[bus] - mp_qg[bus] for bus in gen_buses]

    # For pure AC hybrid runs, branch arrays are copied back to the AC facade.
    branch_by_idx = {int(br.idx): br for br in hybrid_result.ac_network.branches}
    tr_by_idx = {int(tr.idx): tr for tr in hybrid_result.ac_network.transformers}
    flow_p_err = []
    flow_q_err = []
    from benchmark_ieee_flat_compare import BRANCH_COLS, TRANSFORMER_COLS
    for mp_row, (kind, idx) in zip(mp_result["branch"], branch_map):
        if kind == "branch":
            dev = branch_by_idx.get(int(acppc["branch"][idx, BRANCH_COLS["idx"]]))
        else:
            dev = tr_by_idx.get(int(acppc["transformer"][idx, TRANSFORMER_COLS["idx"]]))
        if dev is None:
            continue
        flow_p_err.extend([
            float(getattr(dev, "i_p", 0.0) or 0.0) - mp_row[PF] / base_mva,
            float(getattr(dev, "j_p", 0.0) or 0.0) - mp_row[PT] / base_mva,
        ])
        flow_q_err.extend([
            float(getattr(dev, "i_q", 0.0) or 0.0) - mp_row[QF] / base_mva,
            float(getattr(dev, "j_q", 0.0) or 0.0) - mp_row[QT] / base_mva,
        ])

    return {
        "vm": _max_abs(vm_err),
        "va": _max_abs(va_err),
        "flow_p": _max_abs(flow_p_err),
        "flow_q": _max_abs(flow_q_err),
        "pg": _max_abs(pg_err),
        "qg": _max_abs(qg_err),
    }


def _run_hybrid(e_file):
    start = time.perf_counter()
    network = _read_lf_network_from_file(e_file)
    load_s = time.perf_counter() - start
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)
    start = time.perf_counter()
    _silent(calc.prepare)
    rc = _silent(calc.run)
    solve_s = time.perf_counter() - start
    result = HybridPowerFlowResult(
        network=network,
        ac_network=network.ac,
        dc_network=network.dc,
        calc=calc,
        ac=calc.ac_calc,
        dc=calc.dc_calc,
        rc=rc,
        ac_warnings=[],
        ac_errors=[],
        dc_warnings=[],
        dc_errors=[],
        lf_result=getattr(calc, "lf_result", None),
    )
    return result, load_s, solve_s


def bench(case):
    e_file = ROOT_DIR / "data" / "ac" / f"{case}.e"
    t0 = time.perf_counter()
    acppc = build_ac_ppc_from_e_file(e_file, use_cache=True, copy_arrays=False)
    parse_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    mp_ppc, row_to_bus_id, branch_map, active_gen_rows = _build_matpower_ppc(acppc)
    mp_build_s = time.perf_counter() - t0
    mp_result, mp_ok, mp_s = _run_matpower(mp_ppc)
    ac_calc, ac_rc, ac_s = _run_ours(acppc)
    hybrid_result, hybrid_load_s, hybrid_s = _run_hybrid(e_file)
    return {
        "case": case,
        "parse_s": parse_s,
        "mp_build_s": mp_build_s,
        "e_buses": int(acppc["bus"].shape[0]),
        "mp_buses": int(mp_ppc["bus"].shape[0]),
        "mp_ok": mp_ok,
        "mp_s": mp_s,
        "ac_ok": ac_rc == 0 and ac_calc.converged,
        "ac_s": ac_s,
        "ac_iter": int(ac_calc.iterations),
        "ac_norm": float(ac_calc.normF),
        "hybrid_ok": hybrid_result.converged,
        "hybrid_load_s": hybrid_load_s,
        "hybrid_s": hybrid_s,
        "hybrid_iter": int(hybrid_result.calc.iterations),
        "hybrid_norm": float(hybrid_result.calc.normF),
        "ac_acc": _accuracy(acppc, ac_calc, mp_result, row_to_bus_id, branch_map, active_gen_rows),
        "hybrid_acc": _accuracy_hybrid(acppc, hybrid_result, mp_result, row_to_bus_id, branch_map, active_gen_rows),
    }


def main():
    results = []
    for case in ("ieee3k", "ieee1w", "ieee3w"):
        result = bench(case)
        results.append(result)
        print(f"DONE {case}", flush=True)
    print("RESULT_START")
    for r in results:
        print(
            r["case"],
            r["e_buses"],
            r["mp_buses"],
            r["parse_s"],
            r["mp_build_s"],
            r["mp_ok"],
            r["mp_s"],
            r["ac_ok"],
            r["ac_s"],
            r["ac_iter"],
            r["ac_norm"],
            r["hybrid_ok"],
            r["hybrid_load_s"],
            r["hybrid_s"],
            r["hybrid_iter"],
            r["hybrid_norm"],
            r["ac_acc"],
            r["hybrid_acc"],
            sep="\t",
        )


if __name__ == "__main__":
    main()
