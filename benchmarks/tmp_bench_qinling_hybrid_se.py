import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from secore.hybrid_se import HybridStateEstimator  # noqa: E402


CASES = ("qinling_100", "qinling_1000")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _count_live(items):
    return sum(1 for item in items if getattr(item, "is_alive", False))


def _network_summary(estimator):
    network = estimator.network
    ac = network.ac
    dc = network.dc
    ac_v = np.asarray(
        [float(getattr(node, "voltage", 0.0) or 0.0) for node in estimator.ac_nodes],
        dtype=float,
    )
    dc_v = np.asarray(
        [float(getattr(node, "voltage", 0.0) or 0.0) for node in estimator.dc_nodes],
        dtype=float,
    )
    return {
        "ac_nodes": len(estimator.ac_nodes),
        "dc_nodes": len(estimator.dc_nodes),
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
        "ac_v_min": float(ac_v.min()) if ac_v.size else 0.0,
        "ac_v_max": float(ac_v.max()) if ac_v.size else 0.0,
        "dc_v_min": float(dc_v.min()) if dc_v.size else 0.0,
        "dc_v_max": float(dc_v.max()) if dc_v.size else 0.0,
    }


def bench(case):
    e_file = ROOT_DIR / "data" / "hybrid" / f"{case}.e"
    meas_file = ROOT_DIR / "data" / "hybrid" / f"{case}.meas"

    start = time.perf_counter()
    estimator = HybridStateEstimator(e_file=e_file, meas_file=meas_file)
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

    return {
        "case": case,
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
        "summary": _network_summary(estimator),
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
