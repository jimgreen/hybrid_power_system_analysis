import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dc_array_model import DCPowerNetwork  # noqa: E402
from dc_lf import DCPowerFlowCalc  # noqa: E402
from hybrid_lf import HybridPowerFlowCalc, HybridPowerFlowResult, _read_lf_network_from_file  # noqa: E402


CASES = ("dc_net_3000", "dc_net_1w", "dc_net_3w")
ALGOS = ("dc_lf", "hybrid_lf")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _max_abs(values):
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _device_key(device):
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def _dc_state(network):
    return {
        node.name: float(getattr(node, "voltage", 0.0) or 0.0)
        for node in getattr(network, "nodes", [])
        if getattr(node, "is_alive", False)
    }


def _device_values(devices, fields):
    values = {}
    for dev in devices:
        if not getattr(dev, "is_alive", True):
            continue
        values[_device_key(dev)] = tuple(float(getattr(dev, field, 0.0) or 0.0) for field in fields)
    return values


def _compare_maps(left, right):
    common = sorted(set(left) & set(right))
    if not common:
        return {"common": 0, "max": 0.0}
    diffs = []
    for key in common:
        diffs.extend(np.asarray(right[key], dtype=float) - np.asarray(left[key], dtype=float))
    return {"common": len(common), "max": _max_abs(diffs)}


def _network_summary(network):
    nodes = [node for node in getattr(network, "nodes", []) if getattr(node, "is_alive", False)]
    branches = [br for br in getattr(network, "branches", []) if getattr(br, "is_alive", False)]
    zbr = [br for br in getattr(network, "zero_branches", []) if getattr(br, "is_alive", False)]
    breakers = [br for br in getattr(network, "breakers", []) if getattr(br, "is_alive", False)]
    dcdc = [conv for conv in getattr(network, "dcdc_converters", []) if getattr(conv, "is_alive", False)]
    gens = [gen for gen in getattr(network, "generators", []) if getattr(gen, "is_alive", False)]
    loads = [load for load in getattr(network, "loads", []) if getattr(load, "is_alive", False)]
    volt = np.asarray([float(getattr(node, "voltage", 0.0) or 0.0) for node in nodes], dtype=float)
    return {
        "nodes": len(nodes),
        "branches": len(branches),
        "zero_branches": len(zbr),
        "breakers": len(breakers),
        "dcdc": len(dcdc),
        "generators": len(gens),
        "loads": len(loads),
        "volt_min": float(volt.min()) if volt.size else 0.0,
        "volt_max": float(volt.max()) if volt.size else 0.0,
        "gen_p_sum": float(sum(float(getattr(gen, "p", 0.0) or 0.0) for gen in gens)),
        "load_p_sum": float(sum(float(getattr(load, "p", 0.0) or 0.0) for load in loads)),
    }


def _run_dc_lf(e_file):
    start = time.perf_counter()
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    load_s = time.perf_counter() - start

    calc = DCPowerFlowCalc(network)
    start = time.perf_counter()
    rc = _silent(calc.run, tol=1e-8, max_iter=50, verbose=False)
    solve_s = time.perf_counter() - start
    return {
        "network": network,
        "calc": calc,
        "rc": int(rc),
        "load_s": load_s,
        "solve_s": solve_s,
        "converged": bool(calc.converged),
        "iterations": int(getattr(calc, "iterations", 0)),
        "normF": float(getattr(calc, "normF", 0.0) or 0.0),
    }


def _run_hybrid_lf(e_file):
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
    return {
        "network": network.dc,
        "calc": calc.dc_calc,
        "hybrid_calc": calc,
        "result": result,
        "rc": int(rc),
        "load_s": load_s,
        "solve_s": solve_s,
        "converged": bool(result.converged),
        "iterations": int(getattr(calc.dc_calc, "iterations", 0) if calc.dc_calc is not None else getattr(calc, "iterations", 0)),
        "normF": float(getattr(calc.dc_calc, "normF", 0.0) if calc.dc_calc is not None else getattr(calc, "normF", 0.0) or 0.0),
    }


def _run_once(case):
    e_file = ROOT_DIR / "data" / "dc" / f"{case}.e"
    dc = _run_dc_lf(e_file)
    hybrid = _run_hybrid_lf(e_file)
    dc_net = dc["network"]
    hy_net = hybrid["network"]
    dc_volt = _dc_state(dc_net)
    hy_volt = _dc_state(hy_net)
    common_nodes = sorted(set(dc_volt) & set(hy_volt))
    volt_err = [hy_volt[name] - dc_volt[name] for name in common_nodes]
    comparisons = {
        "volt": {"common": len(common_nodes), "max": _max_abs(volt_err)},
        "branch_pci": _compare_maps(
            _device_values(dc_net.branches, ("i_p", "i_c", "j_p", "j_c")),
            _device_values(hy_net.branches, ("i_p", "i_c", "j_p", "j_c")),
        ),
        "zero_branch_pci": _compare_maps(
            _device_values(dc_net.zero_branches, ("i_p", "i_c")),
            _device_values(hy_net.zero_branches, ("i_p", "i_c")),
        ),
        "break_pci": _compare_maps(
            _device_values(getattr(dc_net, "breakers", []), ("i_p", "i_c")),
            _device_values(getattr(hy_net, "breakers", []), ("i_p", "i_c")),
        ),
        "dcdc_pci": _compare_maps(
            _device_values(dc_net.dcdc_converters, ("i_p", "i_c", "j_p", "j_c")),
            _device_values(hy_net.dcdc_converters, ("i_p", "i_c", "j_p", "j_c")),
        ),
        "gen_pci": _compare_maps(
            _device_values(dc_net.generators, ("p", "current")),
            _device_values(hy_net.generators, ("p", "current")),
        ),
        "load_pci": _compare_maps(
            _device_values(dc_net.loads, ("p", "current")),
            _device_values(hy_net.loads, ("p", "current")),
        ),
    }
    return {
        "case": case,
        "dc_lf": {key: dc[key] for key in ("load_s", "solve_s", "converged", "iterations", "normF", "rc")},
        "hybrid_lf": {key: hybrid[key] for key in ("load_s", "solve_s", "converged", "iterations", "normF", "rc")},
        "dc_summary": _network_summary(dc_net),
        "hybrid_summary": _network_summary(hy_net),
        "diff": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASES))
    args = parser.parse_args()
    results = []
    for case in args.cases:
        result = _run_once(case)
        results.append(result)
        print(f"DONE {case}", flush=True)
    print("RESULT_START")
    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
