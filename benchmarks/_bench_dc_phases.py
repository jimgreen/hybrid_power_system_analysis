"""分项计时:把 DC 大算例端到端时间拆成 3-4 个阶段。

DC LF 的热点: 读 E 文件 + 构建 DCPowerNetwork + topo() + Newton 求解。
"""

import argparse
import contextlib
import gc
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_ROOT = SRC_DIR / "hybrid_power_system_analysis"
MODEL_DIR = PKG_ROOT / "model"
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork  # noqa: E402


def time_run_phases(e_file: Path, result_mode: str = "array") -> Dict:
    """分 4 个阶段计时:read_from_file / topo / construct_calc / newton。"""
    gc.collect()
    phases = {}

    # 阶段 1: 读 E 文件
    t0 = time.perf_counter()
    network = DCPowerNetwork()
    network.read_from_file(e_file)
    phases["read_file_s"] = time.perf_counter() - t0

    # 阶段 2: topo() — 拓扑分析
    t0 = time.perf_counter()
    network.topo()
    phases["topo_s"] = time.perf_counter() - t0

    # 阶段 3: 构造 DCPowerFlowCalc (含 build_jacobian_template 等)
    t0 = time.perf_counter()
    calc = DCPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    phases["construct_calc_s"] = time.perf_counter() - t0

    # 阶段 4: Newton 求解
    t0 = time.perf_counter()
    rc = calc.run()
    phases["newton_solve_s"] = time.perf_counter() - t0

    # 阶段 5: 端到端重新测一次(不展开, 取 wall-clock 总时间)
    gc.collect()
    t0 = time.perf_counter()
    n2 = DCPowerNetwork()
    n2.read_from_file(e_file)
    n2.topo()
    c2 = DCPowerFlowCalc(n2, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    c2.run()
    phases["total_s"] = time.perf_counter() - t0

    # 元数据
    phases["nodes"] = len(network.nodes)
    phases["branches"] = len(network.branches)
    phases["converged"] = calc.converged
    phases["iter"] = calc.iterations
    phases["normF"] = float(calc.normF)
    phases["rc"] = rc
    phases["result_mode"] = result_mode
    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--e-file", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--result-mode", dest="result_mode", type=str, default="array",
                        choices=["full", "array", "summary", "none"])
    args = parser.parse_args()

    e_file = Path(args.e_file)
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    print("Warmup 1 round...")
    time_run_phases(e_file, args.result_mode)
    gc.collect()

    print(f"Measuring {args.repeat} rounds (result_mode={args.result_mode})...")
    runs: List[Dict] = []
    for _ in range(args.repeat):
        runs.append(time_run_phases(e_file, args.result_mode))
        gc.collect()

    keys = ["read_file_s", "topo_s", "construct_calc_s", "newton_solve_s", "total_s"]
    agg = {"case": args.case, "e_file": str(e_file), "n_runs": len(runs),
           "result_mode": args.result_mode}
    for k in keys:
        vals = sorted([r[k] for r in runs])
        agg[k] = {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}
    agg["nodes"] = runs[0]["nodes"]
    agg["branches"] = runs[0]["branches"]
    agg["iter"] = runs[0]["iter"]
    agg["converged"] = runs[0]["converged"]
    agg["normF"] = runs[0]["normF"]

    print()
    print(f"=== {args.case} ({agg['nodes']} nodes, {agg['branches']} branches, "
          f"{agg['iter']} iter, norm={agg['normF']:.2e}, mode={args.result_mode}) ===")
    print(f"{'phase':<22} {'min':<10} {'median':<10} {'% of total (med)':<18}")
    print("-" * 60)
    total_med = agg["total_s"]["median"]
    for k in keys[:-1]:
        v = agg[k]["median"]
        pct = v / total_med * 100 if total_med > 0 else 0
        print(f"{k:<22} {agg[k]['min']:<10.4f} {v:<10.4f} {pct:<18.1f}")
    print("-" * 60)
    print(f"{'total_s':<22} {agg['total_s']['min']:<10.4f} {total_med:<10.4f} 100.0")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
