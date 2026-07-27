"""Single-process random-order LF benchmark in array mode.

Requirements:
- result_mode='array'
- flat start
- single process
- random call order per round
- end-to-end timing
- 5 runs per case (or caller-specified repeat)
- report min / max / average
"""

import argparse
import gc
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

from hybrid_power_system_analysis.lfcore.ac_lf import ACPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc  # noqa: E402

CASE_MAP = {
    "ieee300":       ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee300.e"),
    "ieee3k":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3k.e"),
    "ieee1w":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee1w.e"),
    "ieee3w":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3w.e"),
    "dc_net_1000":   ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_1000.e"),
    "dc_net_3000":   ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_3000.e"),
    "dc_net_3w":     ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_3w.e"),
    "qinling_100":   ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_100.e"),
    "qinling_1000":  ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_1000.e"),
    "hybrid_net_4k": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "hybrid_net_4k.e"),
    "hybrid_net_4w": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "hybrid_net_4w.e"),
}


def run_case(case_name: str):
    kind, path = CASE_MAP[case_name]
    t0 = time.perf_counter()
    if kind == "ac":
        calc = ACPowerFlowCalc.from_file_fast(path, tol=1e-8, max_iter=50, verbose=False, result_mode="array")
    elif kind == "dc":
        calc = DCPowerFlowCalc.from_file_fast(path, tol=1e-8, max_iter=50, verbose=False, result_mode="array")
    else:
        calc = HybridPowerFlowCalc.from_file_fast(path, tol=1e-8, max_iter=50, verbose=False, result_mode="array")
    calc.prepare()
    rc = calc.run()
    elapsed = time.perf_counter() - t0
    return {
        "case": case_name,
        "kind": kind,
        "elapsed_s": elapsed,
        "iter": int(calc.iterations),
        "normF": float(calc.normF),
        "converged": bool(calc.converged and rc == 0),
        "rc": int(rc),
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
            "normF": rows[0]["normF"],
            "converged": rows[0]["converged"],
            "runs": len(rows),
        }
    return summary


def print_table(summary, ordered_cases):
    print(f"{'case':<14} {'kind':<8} {'min_s':<10} {'max_s':<10} {'avg_s':<10} {'iter':<6} {'normF':<12} {'ok':<4}")
    print('-' * 86)
    for case in ordered_cases:
        s = summary[case]
        ok = 'Y' if s['converged'] else 'N'
        print(f"{case:<14} {s['kind']:<8} {s['min_s']:<10.4f} {s['max_s']:<10.4f} {s['avg_s']:<10.4f} {s['iter']:<6} {s['normF']:<12.3e} {ok:<4}")


def main():
    parser = argparse.ArgumentParser(description="Single-process random-order end-to-end LF benchmark in array mode.")
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ordered_cases = [c for c in args.cases if c in CASE_MAP]
    rng = random.Random(args.seed)

    print(f"Warmup: {args.warmup} randomized rounds...")
    for _ in range(args.warmup):
        round_cases = ordered_cases[:]
        rng.shuffle(round_cases)
        for case in round_cases:
            run_case(case)
            gc.collect()

    print(f"Measure: {args.repeat} randomized rounds...")
    all_runs = []
    for round_idx in range(args.repeat):
        round_cases = ordered_cases[:]
        rng.shuffle(round_cases)
        print(f"round {round_idx + 1}: {' -> '.join(round_cases)}")
        for case in round_cases:
            result = run_case(case)
            all_runs.append(result)
            gc.collect()

    summary = summarize(all_runs, ordered_cases)
    print()
    print_table(summary, ordered_cases)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "seed": args.seed,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "summary": summary,
                "runs": all_runs,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
