"""LF benchmark using cProfile — gives accurate function-level timings.

Usage: python benchmarks/_bench_lf_cprofile.py ieee300 ieee3k ... --repeat 2
"""

import argparse
import contextlib
import gc
import io
import json
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

import numpy as np  # noqa: E402
import cProfile  # noqa: E402
import pstats  # noqa: E402

from hybrid_power_system_analysis.model.ac_array_model import build_ac_ppc_from_e_file  # noqa: E402
from hybrid_power_system_analysis.lfcore.ac_lf import ACPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork  # noqa: E402
from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork  # noqa: E402


CASE_MAP = {
    "ieee300":      ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee300.e"),
    "ieee3k":       ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3k.e"),
    "ieee1w":       ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee1w.e"),
    "ieee3w":       ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3w.e"),
    "dc_net_1000":  ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_1000.e"),
    "dc_net_3000":  ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_3000.e"),
    "qinling_100":  ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_100.e"),
    "qinling_1000": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_1000.e"),
}


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _run_ac(e_file, profiler):
    ppc = build_ac_ppc_from_e_file(e_file)
    calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
    profiler.enable()
    rc = calc.run()
    profiler.disable()
    bus = ppc["bus"]
    V = bus[:, 7] if bus.shape[1] > 7 else np.zeros(1)
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": int(bus.shape[0]),
        "V_head": V[:8].tolist(), "V_norm": float(np.linalg.norm(V)),
    }


def _run_dc(e_file, profiler):
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    calc = DCPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)
    profiler.enable()
    rc = calc.run()
    profiler.disable()
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": len(network.nodes),
    }


def _run_hybrid(e_file, profiler):
    network = HybridPowerNetwork.read_from_file(e_file)
    with contextlib.redirect_stdout(io.StringIO()):
        network.prepare(False)
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)

    def solve():
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            return calc.run()

    profiler.enable()
    rc = solve()
    profiler.disable()
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": network.total_nodes,
    }


def _time_call(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def _hot_keys(kind):
    """Symbols we want timings for, by kind."""
    if kind == "ac":
        return [
            "_get_jacobi_from_precomputed_pattern",
            "_fill_standard_jacobian_data",
            "_fill_zero_top_jacobian_data",
            "_get_standard_jacobi_direct",
            "_calc_power_balance",
            "_calc_zero_branch_power",
            "_factor_jacobian",
            "_run_newton_raphson",
            "spsolve",
        ]
    if kind == "dc":
        return [
            "_get_jacobi_from_precomputed_pattern",
            "_get_jacobi_from_terms",
            "_factor_jacobian",
            "_run_newton_raphson",
            "_write_back_ppc",
            "spsolve",
        ]
    return [
        "_assemble_jacobian",
        "_assemble_jacobian_from_precomputed_pattern",
        "_build_newton_system",
        "_run_newton_raphson",
        "_factor_jacobian",
        "spsolve",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    all_results = []
    for case_name in args.cases:
        if case_name not in CASE_MAP:
            continue
        kind, e_file = CASE_MAP[case_name]
        if not e_file.exists():
            print(f"!! Missing: {e_file}")
            continue
        runs = []
        for _ in range(args.repeat):
            profiler = cProfile.Profile()
            try:
                if kind == "ac":
                    info, total_s = _time_call(lambda: _run_ac(e_file, profiler))
                elif kind == "dc":
                    info, total_s = _time_call(lambda: _run_dc(e_file, profiler))
                else:
                    info, total_s = _time_call(lambda: _run_hybrid(e_file, profiler))
            except Exception as e:
                info, total_s = {"error": f"{type(e).__name__}: {e}"}, 0.0
                runs.append({"info": info, "total_s": total_s, "hot": {}})
                gc.collect()
                continue

            stats = pstats.Stats(profiler)
            hot = {}
            for key in _hot_keys(kind):
                t = 0.0
                for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
                    fname, lineno, fname_short = func_key
                    if fname_short == key:
                        t += tt
                hot[key] = t
            runs.append({"info": info, "total_s": total_s, "hot": hot})
            gc.collect()

        runs_clean = [r for r in runs if "error" not in r["info"]]
        if not runs_clean:
            print(f"[{case_name}] ERROR: {runs[0]['info']['error']}")
            all_results.append({"case": case_name, "kind": kind, "error": runs[0]["info"]["error"]})
            continue
        runs_clean.sort(key=lambda r: r["total_s"])
        med = runs_clean[len(runs_clean) // 2]
        info = med["info"]
        agg = {
            "case": case_name, "kind": kind, "nodes": info.get("nodes"),
            "iter": info["iter"], "converged": info["converged"], "rc": info["rc"],
            "normF": info["normF"], "total_s": med["total_s"],
            "hot": {k: v for k, v in med["hot"].items()},
            "V_head": info.get("V_head"), "V_norm": info.get("V_norm"),
        }
        all_results.append(agg)
        hot_str = " ".join(f"{k.split('_')[-1]}={v*1000:.1f}ms" for k, v in med["hot"].items() if v > 0.001)
        print(
            f"[{case_name}] median total={med['total_s']:.4f}s iter={info['iter']} "
            f"norm={info['normF']:.2e} ok={info['converged']} | {hot_str}",
            flush=True,
        )

    print()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
