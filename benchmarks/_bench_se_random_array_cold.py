"""Single-process random-order cold-start SE benchmark in array mode.

Mirrors benchmarks/_bench_lf_random_array_cold.py but for state estimation.

Requirements:
- result_mode='array'
- flat start
- single process per launched benchmark script
- random call order per round
- end-to-end timing (estimator construction through run() finish)
- cold start: fresh estimator each call, no warmup, no cross-process cache
- report min / max / average
"""

import argparse
import contextlib
import gc
import io
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_ROOT = SRC_DIR / "hybrid_power_system_analysis"
MODEL_DIR = PKG_ROOT / "model"
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_power_system_analysis.secore.ac_se import ACStateEstimator  # noqa: E402
from hybrid_power_system_analysis.secore.dc_se import DCStateEstimator  # noqa: E402
from hybrid_power_system_analysis.secore.hybrid_se import HybridStateEstimator  # noqa: E402

DATA = PROJECT_ROOT / "data"
CASE_MAP = {
    "ieee300":       ("ac", DATA / "model/ac/ieee300.e", DATA / "meas/ac/ieee300.meas"),
    "ieee3k":        ("ac", DATA / "model/ac/ieee3k.e", DATA / "meas/ac/ieee3k.meas"),
    "ieee1w":        ("ac", DATA / "model/ac/ieee1w.e", DATA / "meas/ac/ieee1w.meas"),
    "ieee3w":        ("ac", DATA / "model/ac/ieee3w.e", DATA / "meas/ac/ieee3w.meas"),
    "dc_net_1000":   ("dc", DATA / "model/dc/dc_net_1000.e", DATA / "meas/dc/dc_net_1000.meas"),
    "dc_net_3000":   ("dc", DATA / "model/dc/dc_net_3000.e", DATA / "meas/dc/dc_net_3000.meas"),
    "dc_net_3w":     ("dc", DATA / "model/dc/dc_net_3w.e", DATA / "meas/dc/dc_net_3w.meas"),
    "qinling_100":   ("hybrid", DATA / "model/hybrid/qinling_100.e", DATA / "meas/hybrid/qinling_100.meas"),
    "qinling_1000":  ("hybrid", DATA / "model/hybrid/qinling_1000.e", DATA / "meas/hybrid/qinling_1000.meas"),
    "hybrid_net_4k": ("hybrid", DATA / "model/hybrid/hybrid_net_4k.e", DATA / "meas/hybrid/hybrid_net_4k.meas"),
    "hybrid_net_4w": ("hybrid", DATA / "model/hybrid/hybrid_net_4w.e", DATA / "meas/hybrid/hybrid_net_4w.meas"),
}

EST_CLS = {"ac": ACStateEstimator, "dc": DCStateEstimator, "hybrid": HybridStateEstimator}


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def run_case(case_name: str):
    kind, e_file, meas_file = CASE_MAP[case_name]
    est_cls = EST_CLS[kind]
    t0 = time.perf_counter()
    est = est_cls(e_file=e_file, meas_file=meas_file, flat_start=True)
    _silent(est.run, result_mode="array", verbose=False)
    elapsed = time.perf_counter() - t0
    result = est.estimate_result
    obs = est.observability_result
    return {
        "case": case_name,
        "kind": kind,
        "elapsed_s": elapsed,
        "iter": int(result.iterations),
        "resid": float(result.residual_inf),
        "converged": bool(result.converged),
        "observable": bool(obs.observable) if obs is not None else None,
        "state_count": int(obs.state_count) if obs is not None else None,
        "measurement_count": int(obs.measurement_count) if obs is not None else None,
    }


def summarize(all_runs, ordered_cases):
    summary = {}
    for case in ordered_cases:
        rows = [r for r in all_runs if r["case"] == case]
        vals = [r["elapsed_s"] for r in rows]
        summary[case] = {
            "kind": rows[0]["kind"],
            "min_s": min(vals),
            "max_s": max(vals),
            "avg_s": sum(vals) / len(vals),
            "iter": rows[0]["iter"],
            "resid": rows[0]["resid"],
            "converged": rows[0]["converged"],
            "observable": rows[0]["observable"],
            "state_count": rows[0]["state_count"],
            "measurement_count": rows[0]["measurement_count"],
            "runs": len(rows),
        }
    return summary


def print_table(summary, ordered_cases):
    print(f"{'case':<14} {'kind':<8} {'min_s':<10} {'max_s':<10} {'avg_s':<10} {'iter':<6} {'resid':<12} {'obs':<4} {'ok':<4}")
    print("-" * 100)
    for case in ordered_cases:
        s = summary[case]
        ok = "Y" if s["converged"] else "N"
        obs = "Y" if s["observable"] else "N"
        print(f"{case:<14} {s['kind']:<8} {s['min_s']:<10.4f} {s['max_s']:<10.4f} {s['avg_s']:<10.4f} {s['iter']:<6} {s['resid']:<12.3e} {obs:<4} {ok:<4}")


def main():
    parser = argparse.ArgumentParser(description="Single-process random-order cold-start SE benchmark in array mode.")
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ordered_cases = [c for c in args.cases if c in CASE_MAP]
    rng = random.Random(args.seed)

    print(f"Measure: {args.repeat} randomized rounds (cold-start, array mode, no warmup)...")
    all_runs = []
    for round_idx in range(args.repeat):
        round_cases = ordered_cases[:]
        rng.shuffle(round_cases)
        print(f"round {round_idx + 1}: {' -> '.join(round_cases)}")
        for case in round_cases:
            all_runs.append(run_case(case))
            gc.collect()

    summary = summarize(all_runs, ordered_cases)
    print()
    print_table(summary, ordered_cases)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "seed": args.seed,
                "repeat": args.repeat,
                "summary": summary,
                "runs": all_runs,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
