import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_lf import HybridPowerFlowCalc, HybridPowerFlowResult, _read_lf_network_from_file  # noqa: E402


CASES = ("qinling_100", "qinling_1000")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _count_live(items):
    return sum(1 for item in items if getattr(item, "is_alive", False))


def _volt_range(nodes):
    values = np.asarray(
        [float(getattr(node, "voltage", 0.0) or 0.0) for node in nodes if getattr(node, "is_alive", False)],
        dtype=float,
    )
    return {
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def _abs_max(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _finite_sum(values):
    return float(sum(float(value or 0.0) for value in values))


def _reset_flat_start(network):
    for node in getattr(network.ac, "nodes", []):
        node.voltage = 1.0
        node.angle = 0.0
    for node in getattr(network.dc, "nodes", []):
        node.voltage = 1.0


def _summary(network):
    ac = network.ac
    dc = network.dc
    ac_angles = np.asarray(
        [float(getattr(node, "angle", 0.0) or 0.0) for node in getattr(ac, "nodes", []) if getattr(node, "is_alive", False)],
        dtype=float,
    )
    ac_gens = [gen for gen in getattr(ac, "generators", []) if getattr(gen, "is_alive", False)]
    ac_loads = [load for load in getattr(ac, "loads", []) if getattr(load, "is_alive", False)]
    dc_gens = [gen for gen in getattr(dc, "generators", []) if getattr(gen, "is_alive", False)]
    dc_loads = [load for load in getattr(dc, "loads", []) if getattr(load, "is_alive", False)]
    ac_branches = [br for br in getattr(ac, "branches", []) if getattr(br, "is_alive", False)]
    dc_branches = [br for br in getattr(dc, "branches", []) if getattr(br, "is_alive", False)]
    dcdc = [conv for conv in getattr(dc, "dcdc_converters", []) if getattr(conv, "is_alive", False)]
    dcac = [conv for conv in getattr(network, "dcac_converters", []) if getattr(conv, "is_alive", False)]
    return {
        "ac_nodes": _count_live(getattr(ac, "nodes", [])),
        "dc_nodes": _count_live(getattr(dc, "nodes", [])),
        "ac_branches": _count_live(getattr(ac, "branches", [])),
        "ac_transformers": _count_live(getattr(ac, "transformers", [])),
        "ac_zero_branches": _count_live(getattr(ac, "zero_branches", [])),
        "ac_breakers": _count_live(getattr(ac, "breakers", [])),
        "dc_branches": _count_live(getattr(dc, "branches", [])),
        "dc_zero_branches": _count_live(getattr(dc, "zero_branches", [])),
        "dc_breakers": _count_live(getattr(dc, "breakers", [])),
        "dcdc": _count_live(getattr(dc, "dcdc_converters", [])),
        "dcac": _count_live(getattr(network, "dcac_converters", [])),
        "acac": _count_live(getattr(network, "acac_converters", [])),
        "ac_v": _volt_range(getattr(ac, "nodes", [])),
        "dc_v": _volt_range(getattr(dc, "nodes", [])),
        "ac_angle_deg": {
            "min": float(np.degrees(ac_angles.min())) if ac_angles.size else 0.0,
            "max": float(np.degrees(ac_angles.max())) if ac_angles.size else 0.0,
        },
        "ac_gen_p_sum": _finite_sum(getattr(gen, "p", 0.0) for gen in ac_gens),
        "ac_gen_q_sum": _finite_sum(getattr(gen, "q", 0.0) for gen in ac_gens),
        "ac_load_p_sum": _finite_sum(getattr(load, "p", 0.0) for load in ac_loads),
        "ac_load_q_sum": _finite_sum(getattr(load, "q", 0.0) for load in ac_loads),
        "dc_gen_p_sum": _finite_sum(getattr(gen, "p", 0.0) for gen in dc_gens),
        "dc_load_p_sum": _finite_sum(getattr(load, "p", 0.0) for load in dc_loads),
        "ac_branch_abs_p_max": _abs_max(value for br in ac_branches for value in (getattr(br, "i_p", 0.0), getattr(br, "j_p", 0.0))),
        "ac_branch_abs_q_max": _abs_max(value for br in ac_branches for value in (getattr(br, "i_q", 0.0), getattr(br, "j_q", 0.0))),
        "dc_branch_abs_p_max": _abs_max(value for br in dc_branches for value in (getattr(br, "i_p", 0.0), getattr(br, "j_p", 0.0))),
        "dcdc_abs_p_max": _abs_max(value for conv in dcdc for value in (getattr(conv, "i_p", 0.0), getattr(conv, "j_p", 0.0))),
        "dcac_dc_p_sum": _finite_sum(getattr(conv, "dc_p", 0.0) for conv in dcac),
        "dcac_ac_p_sum": _finite_sum(getattr(conv, "ac_p", 0.0) for conv in dcac),
        "dcac_ac_q_sum": _finite_sum(getattr(conv, "ac_q", 0.0) for conv in dcac),
        "dcac_abs_p_max": _abs_max(value for conv in dcac for value in (getattr(conv, "dc_p", 0.0), getattr(conv, "ac_p", 0.0))),
    }


def bench(case):
    e_file = ROOT_DIR / "data" / "hybrid" / f"{case}.e"
    start = time.perf_counter()
    network = _read_lf_network_from_file(e_file)
    load_s = time.perf_counter() - start
    _reset_flat_start(network)

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
    ac_calc = calc.ac_calc
    dc_calc = calc.dc_calc
    return {
        "case": case,
        "load_s": load_s,
        "solve_s": solve_s,
        "total_s": load_s + solve_s,
        "rc": int(rc),
        "converged": bool(result.converged),
        "hybrid_iterations": int(getattr(calc, "iterations", 0)),
        "hybrid_normF": float(getattr(calc, "normF", 0.0) or 0.0),
        "ac_converged": bool(getattr(ac_calc, "converged", False)) if ac_calc is not None else None,
        "ac_iterations": int(getattr(ac_calc, "iterations", 0)) if ac_calc is not None else 0,
        "ac_normF": float(getattr(ac_calc, "normF", 0.0) or 0.0) if ac_calc is not None else 0.0,
        "dc_converged": bool(getattr(dc_calc, "converged", False)) if dc_calc is not None else None,
        "dc_iterations": int(getattr(dc_calc, "iterations", 0)) if dc_calc is not None else 0,
        "dc_normF": float(getattr(dc_calc, "normF", 0.0) or 0.0) if dc_calc is not None else 0.0,
        "summary": _summary(network),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASES))
    args = parser.parse_args()
    results = []
    for case in args.cases:
        result = bench(case)
        results.append(result)
        print(f"DONE {case}", flush=True)
    print("RESULT_START")
    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
