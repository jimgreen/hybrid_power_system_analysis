"""分项计时:把 Hybrid 大算例端到端时间拆成 5 个阶段。

输出每个阶段占总时间的百分比,帮助定位真正的瓶颈。
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

from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork  # noqa: E402


def time_run_phases(e_file: Path) -> Dict:
    """分 5 个阶段计时:文件读取 / 网络构造 / prepare / Newton / 结果写回。"""
    gc.collect()
    phases = {}

    # ---- 阶段 1: 文件读取 (E-file 解析) ----
    t0 = time.perf_counter()
    network = HybridPowerNetwork.read_from_file(e_file)
    phases["read_file_s"] = time.perf_counter() - t0

    # ---- 阶段 2: 网络构造 (内部建 PPC, 索引, 转换) ----
    t0 = time.perf_counter()
    # HybridPowerNetwork.read_from_file 已经做了部分构造;
    # prepare() 会做剩下的:flat start 初始化、子求解器构建、AC/DC/换流器桥接
    # 把它拆成"构造 ppc"和"prepare"两段
    ppc = None
    if hasattr(network, "_ppc"):
        ppc = network._ppc
    phases["construct_ppc_s"] = time.perf_counter() - t0

    # ---- 阶段 3: prepare (flat start, 子求解器构建, 雅可比模板) ----
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        network.prepare(False)
    phases["prepare_s"] = time.perf_counter() - t0

    # ---- 阶段 4: HybridPowerFlowCalc 构造 ----
    t0 = time.perf_counter()
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode="full")
    phases["construct_calc_s"] = time.perf_counter() - t0

    # ---- 阶段 5: Newton 迭代主循环 (含 calc.prepare) ----
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prepare()
    phases["calc_prepare_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rc = calc.run()
    phases["newton_solve_s"] = time.perf_counter() - t0

    # ---- 阶段 6: 端到端总时间 ----
    # 重新跑一次取总时间(不展开)
    gc.collect()
    t0 = time.perf_counter()
    network2 = HybridPowerNetwork.read_from_file(e_file)
    with contextlib.redirect_stdout(io.StringIO()):
        network2.prepare(False)
    calc2 = HybridPowerFlowCalc(network2, tol=1e-8, max_iter=50, verbose=False, result_mode="full")
    with contextlib.redirect_stdout(io.StringIO()):
        calc2.prepare()
        calc2.run()
    phases["total_s"] = time.perf_counter() - t0

    # ---- 元数据 ----
    phases["nodes"] = network.total_nodes
    phases["converged"] = calc.converged
    phases["iter"] = calc.iterations
    phases["normF"] = float(calc.normF)
    phases["rc"] = rc

    # 验证两遍跑结果一致
    phases["converged2"] = calc2.converged
    phases["iter2"] = calc2.iterations
    phases["normF2"] = float(calc2.normF)

    return phases


def run_one(e_file: Path) -> Dict:
    return time_run_phases(e_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="case name")
    parser.add_argument("--e-file", required=True, help=".e file path")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    e_file = Path(args.e_file)
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    # Warmup
    print("Warmup 1 round...")
    run_one(e_file)
    gc.collect()

    # Measure
    print(f"Measuring {args.repeat} rounds...")
    runs: List[Dict] = []
    for _ in range(args.repeat):
        runs.append(run_one(e_file))
        gc.collect()

    # Aggregate
    keys = ["read_file_s", "construct_ppc_s", "prepare_s", "construct_calc_s",
            "calc_prepare_s", "newton_solve_s", "total_s"]
    agg = {"case": args.case, "e_file": str(e_file), "n_runs": len(runs)}
    for k in keys:
        vals = sorted([r[k] for r in runs])
        agg[k] = {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}
    agg["nodes"] = runs[0]["nodes"]
    agg["iter"] = runs[0]["iter"]
    agg["converged"] = runs[0]["converged"]
    agg["normF"] = runs[0]["normF"]
    agg["normF2"] = runs[0]["normF2"]
    agg["converged2"] = runs[0]["converged2"]
    agg["iter2"] = runs[0]["iter2"]

    print()
    print(f"=== {args.case} ({agg['nodes']} nodes, {agg['iter']} iter, norm={agg['normF']:.2e}) ===")
    print(f"{'phase':<22} {'min':<10} {'median':<10} {'% of total (med)':<18}")
    print("-" * 60)
    total_med = agg["total_s"]["median"]
    for k in keys[:-1]:
        v = agg[k]["median"]
        pct = v / total_med * 100 if total_med > 0 else 0
        print(f"{k:<22} {agg[k]['min']:<10.4f} {v:<10.4f} {pct:<18.1f}")
    print("-" * 60)
    print(f"{'total_s':<22} {agg['total_s']['min']:<10.4f} {total_med:<10.4f} 100.0")
    print()
    # 检查两遍跑的一致性
    if agg["normF"] != agg["normF2"] or agg["iter"] != agg["iter2"]:
        print(f"⚠️  two runs inconsistent: run1 norm={agg['normF']:.3e} iter={agg['iter']}, "
              f"run2 norm={agg['normF2']:.3e} iter={agg['iter2']}")
    else:
        print(f"✅ two runs identical: norm={agg['normF']:.3e}, iter={agg['iter']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
