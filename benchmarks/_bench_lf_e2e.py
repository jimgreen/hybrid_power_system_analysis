"""严格端到端 LF 对比:模型文件读取 → 网络构造 → 潮流求解, 单进程、平启动。

Usage:
  python benchmarks/_bench_lf_e2e.py [case ...] --repeat N --out file.json
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
    "dc_net_3w":    ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_3w.e"),
    "qinling_100":  ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_100.e"),
    "qinling_1000": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_1000.e"),
    "hybrid_net_4k": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "hybrid_net_4k.e"),
    "hybrid_net_4w": ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "hybrid_net_4w.e"),
}


def _silent_run(func, *args, **kwargs):
    """Run and capture stdout, return result."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


# --- 端到端流程: 从 .e 文件到求解完成, 完全新建对象, 无复用 ---

def run_ac_e2e(e_file: Path, result_mode: str = "full") -> dict:
    """AC LF 端到端: 读 .e → build_ac_ppc → ACPowerFlowCalc(flat start) → run()."""
    # 1. 加载 PPC
    ppc = build_ac_ppc_from_e_file(e_file)
    # 2. flat start 是默认行为 (V=1.0, theta=0)
    calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode=result_mode)
    # 3. 求解
    rc = calc.run()
    bus = ppc["bus"]
    V = bus[:, 7] if bus.shape[1] > 7 else np.zeros(1)
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": int(bus.shape[0]),
        "V_head": V[:8].tolist(), "V_norm": float(np.linalg.norm(V)),
    }


def run_dc_e2e(e_file: Path, result_mode: str = "full") -> dict:
    """DC LF 端到端: 读 .e → DCPowerNetwork → DCPowerFlowCalc(flat start) → run()."""
    # 1. 构建网络
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    # 2. flat start 默认；调用方无需先 topo()
    calc = DCPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    # 3. prepare() / run() 内会确保拓扑/PPC 就绪
    rc = calc.run()
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": len(network.nodes),
    }


def run_hybrid_e2e(e_file: Path, result_mode: str = "full", fast_hybrid: bool = False) -> dict:
    """Hybrid LF 端到端: 读 .e → network/ppc → HybridPowerFlowCalc(flat start) → run()."""
    from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork
    if fast_hybrid:
        calc = HybridPowerFlowCalc.from_file_fast(
            e_file,
            tol=1e-8,
            max_iter=50,
            verbose=False,
            result_mode=result_mode,
        )
        calc.prepare()
        rc = calc.run()
        return {
            "rc": rc, "converged": calc.converged, "iter": calc.iterations,
            "normF": float(calc.normF), "nodes": calc.network.total_nodes,
            "fast_hybrid": True,
        }

    network = HybridPowerNetwork.read_from_file(e_file)
    # 调用方无需先 network.prepare(False)；calc.prepare()/run() 会直接准备子求解器
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    rc = calc.run()
    return {
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), "nodes": network.total_nodes,
        "fast_hybrid": False,
    }


RUNNERS = {"ac": run_ac_e2e, "dc": run_dc_e2e, "hybrid": run_hybrid_e2e}


def time_run(case_name: str, result_mode: str = "full", fast_hybrid: bool = False) -> dict:
    kind, e_file = CASE_MAP[case_name]
    gc.collect()
    t0 = time.perf_counter()
    if kind == "hybrid":
        info = run_hybrid_e2e(e_file, result_mode, fast_hybrid=fast_hybrid)
    else:
        runner = RUNNERS[kind]
        info = runner(e_file, result_mode)
    elapsed = time.perf_counter() - t0
    return {"case": case_name, "kind": kind, "elapsed_s": elapsed, "result_mode": result_mode,
            "fast_hybrid": fast_hybrid, **info}


def summarize(runs: list, label: str) -> dict:
    """Aggregate runs into a per-case summary with min/median."""
    by_case = {}
    for r in runs:
        by_case.setdefault(r["case"], []).append(r)
    out = {"label": label, "cases": {}}
    for case, rs in by_case.items():
        if not rs:
            continue
        rs = sorted(rs, key=lambda r: r["elapsed_s"])
        med_idx = len(rs) // 2
        med = rs[med_idx]
        out["cases"][case] = {
            "kind": med["kind"],
            "nodes": med.get("nodes"),
            "min_s": rs[0]["elapsed_s"],
            "median_s": med["elapsed_s"],
            "max_s": rs[-1]["elapsed_s"],
            "iter": med["iter"],
            "converged": med["converged"],
            "rc": med["rc"],
            "normF": med["normF"],
            "V_head": med.get("V_head"),
            "V_norm": med.get("V_norm"),
            "n_runs": len(rs),
        }
    return out


def print_table(label: str, summary: dict):
    print(f"\n=== {label} ===")
    print(f"{'case':<14} {'kind':<7} {'nodes':<7} {'min_s':<9} {'med_s':<9} "
          f"{'iter':<5} {'normF':<11} {'ok':<5}")
    print("-" * 70)
    for case, c in summary["cases"].items():
        ok = "Y" if (c["converged"] and c["rc"] == 0) else "N"
        print(f"{case:<14} {c['kind']:<7} {c.get('nodes','-'):<7} "
              f"{c['min_s']:<9.4f} {c['median_s']:<9.4f} "
              f"{c['iter']:<5} {c['normF']:<11.3e} {ok:<5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--result-mode", dest="result_mode", type=str, default="full",
                        choices=["full", "array", "summary", "none"])
    parser.add_argument("--fast-hybrid", dest="fast_hybrid", action="store_true",
                        help="Use HybridPowerFlowCalc.from_file_fast() for hybrid cases")
    args = parser.parse_args()

    cases = [c for c in args.cases if c in CASE_MAP]

    # Warmup
    print(f"Warmup: {args.warmup} rounds × {len(cases)} cases (result_mode={args.result_mode}, fast_hybrid={args.fast_hybrid})...")
    for _ in range(args.warmup):
        for c in cases:
            try:
                time_run(c, args.result_mode, fast_hybrid=args.fast_hybrid)
            except Exception as e:
                print(f"  warmup error {c}: {e}")
            gc.collect()

    # Real measurement
    print(f"Measuring: {args.repeat} rounds × {len(cases)} cases...")
    runs = []
    for r_idx in range(args.repeat):
        for c in cases:
            try:
                runs.append(time_run(c, args.result_mode, fast_hybrid=args.fast_hybrid))
            except Exception as e:
                print(f"  ERROR {c} run {r_idx}: {type(e).__name__}: {e}")
            gc.collect()

    summary = summarize(runs, args.label)
    print_table(args.label, summary)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.write(f"# result_mode: {args.result_mode}\n")
            f.write(f"# fast_hybrid: {args.fast_hybrid}\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
