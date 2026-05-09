import argparse
import contextlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from secore.ac_se import ACStateEstimator  # noqa: E402
from secore.hybrid_se import HybridStateEstimator  # noqa: E402


CASES = ("ieee3k", "ieee1w", "ieee3w")
ALGOS = ("ac_se", "hybrid_se")


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _angle_diff_rad(left, right):
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _node_state(estimator, algo):
    if algo == "ac_se":
        nodes = estimator.nodes
    else:
        nodes = estimator.network.ac.nodes
    return {
        node.name: (float(node.voltage), float(node.angle))
        for node in nodes
        if getattr(node, "is_alive", False)
    }


def _state_stats(estimator, algo):
    values = list(_node_state(estimator, algo).values())
    vm = np.asarray([item[0] for item in values], dtype=float)
    va = np.asarray([item[1] for item in values], dtype=float)
    return {
        "node_count": int(len(values)),
        "vm_min": float(vm.min()) if vm.size else 0.0,
        "vm_max": float(vm.max()) if vm.size else 0.0,
        "va_deg_min": math.degrees(float(va.min())) if va.size else 0.0,
        "va_deg_max": math.degrees(float(va.max())) if va.size else 0.0,
    }


def _bench_once(case, algo):
    e_file = ROOT_DIR / "data" / "ac" / f"{case}.e"
    meas_file = ROOT_DIR / "data" / "ac" / f"{case}.meas"
    estimator_cls = ACStateEstimator if algo == "ac_se" else HybridStateEstimator

    start = time.perf_counter()
    estimator = estimator_cls(e_file=e_file, meas_file=meas_file, flat_start=True)
    init_s = time.perf_counter() - start

    start = time.perf_counter()
    observability = _silent(estimator.observability_analysis)
    obs_s = time.perf_counter() - start

    start = time.perf_counter()
    try:
        result = estimator.estimate(verbose=False, final_diagnostics=False)
    except TypeError:
        result = estimator.estimate(verbose=False)
    wls_s = time.perf_counter() - start

    start = time.perf_counter()
    bad_items, normalized = estimator.identify_bad_data(result)
    bad_s = time.perf_counter() - start

    return {
        "case": case,
        "algo": algo,
        "init_s": init_s,
        "obs_s": obs_s,
        "wls_s": wls_s,
        "bad_s": bad_s,
        "internal_total_s": init_s + obs_s + wls_s + bad_s,
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
        "state_stats": _state_stats(estimator, algo),
    }


def _run_child(case, algo):
    print(json.dumps(_bench_once(case, algo), sort_keys=True), flush=True)


def _run_one_process(case, algo, repeat_idx):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--case", case, "--algo", algo]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    process_wall_s = time.perf_counter() - start
    if completed.returncode != 0:
        return {
            "case": case,
            "algo": algo,
            "repeat": repeat_idx,
            "returncode": completed.returncode,
            "process_wall_s": process_wall_s,
            "error": completed.stderr[-2000:],
        }
    data = json.loads(completed.stdout.strip().splitlines()[-1])
    data["repeat"] = repeat_idx
    data["returncode"] = completed.returncode
    data["process_wall_s"] = process_wall_s
    return data


def _summary(values):
    arr = np.asarray(values, dtype=float)
    return {
        "avg": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _summarize_group(runs):
    ok_runs = [run for run in runs if run.get("returncode") == 0]
    first = ok_runs[-1] if ok_runs else runs[-1]
    summary = {
        "case": first["case"],
        "algo": first["algo"],
        "runs": len(runs),
        "ok_runs": len(ok_runs),
    }
    if not ok_runs:
        summary["error"] = first.get("error", "")
        return summary
    for key in ("process_wall_s", "internal_total_s", "init_s", "obs_s", "wls_s", "bad_s"):
        summary[key] = _summary([run[key] for run in ok_runs])
    for key in (
        "observable",
        "rank",
        "state_count",
        "measurement_count",
        "converged",
        "iterations",
        "objective",
        "max_dx",
        "residual_inf",
        "bad_count",
        "max_norm",
        "state_stats",
    ):
        summary[key] = first[key]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--algo", choices=ALGOS)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.child:
        _run_child(args.case, args.algo)
        return

    all_runs = []
    for case in CASES:
        for algo in ALGOS:
            for repeat_idx in range(1, args.repeats + 1):
                run = _run_one_process(case, algo, repeat_idx)
                all_runs.append(run)
                print(f"DONE {case} {algo} repeat={repeat_idx}", flush=True)

    print("RESULT_START")
    print(json.dumps({"runs": all_runs}, sort_keys=True))
    print("SUMMARY_START")
    summaries = []
    for case in CASES:
        for algo in ALGOS:
            group = [run for run in all_runs if run["case"] == case and run["algo"] == algo]
            summaries.append(_summarize_group(group))
    print(json.dumps({"summaries": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
