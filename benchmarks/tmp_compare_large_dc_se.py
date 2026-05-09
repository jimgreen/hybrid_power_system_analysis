import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from secore.dc_se import DCStateEstimator  # noqa: E402
from secore.hybrid_se import HybridStateEstimator  # noqa: E402


CASES = ("dc_net_3000", "dc_net_1w", "dc_net_3w")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _device_key(device):
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def _nodes(estimator, kind):
    return estimator.nodes if kind == "dc_se" else estimator.dc_nodes


def _network(estimator, kind):
    return estimator.network if kind == "dc_se" else estimator.network.dc


def _node_voltage_map(estimator, kind):
    values = {}
    for node in _nodes(estimator, kind):
        members = getattr(node, "nodes", None) or [node]
        for member in members:
            values[member.name] = float(getattr(member, "voltage", getattr(node, "voltage", 0.0)) or 0.0)
        values.setdefault(node.name, float(getattr(node, "voltage", 0.0) or 0.0))
    return values


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


def _summary(estimator, kind):
    network = _network(estimator, kind)
    volt = np.asarray(list(_node_voltage_map(estimator, kind).values()), dtype=float)
    gens = [gen for gen in getattr(network, "generators", []) if getattr(gen, "is_alive", False)]
    loads = [load for load in getattr(network, "loads", []) if getattr(load, "is_alive", False)]
    return {
        "nodes": int(len(volt)),
        "branches": sum(1 for br in getattr(network, "branches", []) if getattr(br, "is_alive", False)),
        "zero_branches": sum(1 for br in getattr(network, "zero_branches", []) if getattr(br, "is_alive", False)),
        "breakers": sum(1 for br in getattr(network, "breakers", []) if getattr(br, "is_alive", False)),
        "dcdc": sum(1 for conv in getattr(network, "dcdc_converters", []) if getattr(conv, "is_alive", False)),
        "generators": len(gens),
        "loads": len(loads),
        "volt_min": float(volt.min()) if volt.size else 0.0,
        "volt_max": float(volt.max()) if volt.size else 0.0,
        "gen_p_sum": float(sum(float(getattr(gen, "p", 0.0) or 0.0) for gen in gens)),
        "load_p_sum": float(sum(float(getattr(load, "p", 0.0) or 0.0) for load in loads)),
    }


def _run_estimator(estimator_cls, e_file, meas_file, kind):
    start = time.perf_counter()
    estimator = estimator_cls(e_file=e_file, meas_file=meas_file)
    init_s = time.perf_counter() - start

    start = time.perf_counter()
    observability = _silent(estimator.observability_analysis)
    obs_s = time.perf_counter() - start

    start = time.perf_counter()
    result = estimator.estimate(verbose=False)
    wls_s = time.perf_counter() - start

    start = time.perf_counter()
    bad_items, normalized = estimator.identify_bad_data(result)
    bad_s = time.perf_counter() - start

    return estimator, {
        "init_s": init_s,
        "obs_s": obs_s,
        "wls_s": wls_s,
        "bad_s": bad_s,
        "total_s": init_s + obs_s + wls_s + bad_s,
        "observable": bool(observability.observable),
        "rank": int(observability.rank),
        "state_count": int(observability.state_count),
        "measurement_count": int(observability.measurement_count),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "objective": float(result.objective),
        "max_dx": float(result.max_correction) if np.isfinite(result.max_correction) else None,
        "residual_inf": float(result.residual_inf),
        "bad_count": int(len(bad_items)),
        "max_norm": float(np.max(normalized)) if normalized.size else 0.0,
        "summary": _summary(estimator, kind),
    }


def _diff(dc_estimator, hybrid_estimator):
    dc_network = _network(dc_estimator, "dc_se")
    hybrid_network = _network(hybrid_estimator, "hybrid_se")
    dc_v = _node_voltage_map(dc_estimator, "dc_se")
    hy_v = _node_voltage_map(hybrid_estimator, "hybrid_se")
    common_nodes = sorted(set(dc_v) & set(hy_v))
    volt_err = [hy_v[name] - dc_v[name] for name in common_nodes]
    return {
        "volt": {"common": len(common_nodes), "max": _max_abs(volt_err)},
        "branch_pci": _compare_maps(
            _device_values(dc_network.branches, ("i_p", "i_c", "j_p", "j_c")),
            _device_values(hybrid_network.branches, ("i_p", "i_c", "j_p", "j_c")),
        ),
        "zero_branch_pci": _compare_maps(
            _device_values(dc_network.zero_branches, ("i_p", "i_c")),
            _device_values(hybrid_network.zero_branches, ("i_p", "i_c")),
        ),
        "break_pci": _compare_maps(
            _device_values(getattr(dc_network, "breakers", []), ("i_p", "i_c")),
            _device_values(getattr(hybrid_network, "breakers", []), ("i_p", "i_c")),
        ),
        "dcdc_pci": _compare_maps(
            _device_values(dc_network.dcdc_converters, ("i_p", "i_c", "j_p", "j_c")),
            _device_values(hybrid_network.dcdc_converters, ("i_p", "i_c", "j_p", "j_c")),
        ),
        "gen_pci": _compare_maps(
            _device_values(dc_network.generators, ("p", "current")),
            _device_values(hybrid_network.generators, ("p", "current")),
        ),
        "load_pci": _compare_maps(
            _device_values(dc_network.loads, ("p", "current")),
            _device_values(hybrid_network.loads, ("p", "current")),
        ),
    }


def bench(case):
    e_file = ROOT_DIR / "data" / "dc" / f"{case}.e"
    meas_file = ROOT_DIR / "data" / "dc" / f"{case}.meas"
    dc_estimator, dc = _run_estimator(DCStateEstimator, e_file, meas_file, "dc_se")
    hybrid_estimator, hybrid = _run_estimator(HybridStateEstimator, e_file, meas_file, "hybrid_se")
    return {
        "case": case,
        "dc_se": dc,
        "hybrid_se": hybrid,
        "diff": _diff(dc_estimator, hybrid_estimator),
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
