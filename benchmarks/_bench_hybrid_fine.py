"""细粒度: 用 cProfile + 子函数级计时 + 显式 wall-clock 包围,精准定位 Hybrid 大算例瓶颈。

不做 monkey-patch 失败也无所谓,主要靠显式计时。
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


def profile_hybrid(e_file: Path, result_mode: str = "array") -> Dict:
    """分 7+ 个显式 wall-clock 段。"""
    from hybrid_power_system_analysis.efile_read import _read_efile_rows
    from hybrid_power_system_analysis.model.ac_array_model import build_ac_ppc_from_efile_rows
    from hybrid_power_system_analysis.model.dc_array_model import build_dc_ppc_from_efile_rows
    from hybrid_power_system_analysis.model.hybrid_array_model import (
        build_hybrid_ppc_from_efile_rows,
        build_hybrid_ppc_from_e_file,
        _build_dcac_from_rows,
        _build_ac_network_from_ppc,
    )
    from hybrid_power_system_analysis.model.ac_model import ACPowerNetwork
    from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork
    from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc

    timings: Dict[str, float] = {}

    # 1. 读 E 文件(纯文本 → row dicts)
    t0 = time.perf_counter()
    rows = _read_efile_rows(e_file)
    timings["read_efile_rows_s"] = time.perf_counter() - t0

    # 2. AC PPC 构建
    t0 = time.perf_counter()
    ac_ppc = build_ac_ppc_from_efile_rows(e_file, rows)
    timings["build_ac_ppc_s"] = time.perf_counter() - t0

    # 3. DC PPC 构建
    t0 = time.perf_counter()
    dc_ppc = build_dc_ppc_from_efile_rows(e_file, rows)
    timings["build_dc_ppc_s"] = time.perf_counter() - t0

    # 4. DCAC 换流器构建
    t0 = time.perf_counter()
    vbase_maps = sys.modules["hybrid_power_system_analysis.model.hybrid_array_model"]._raw_vbase_maps(ac_ppc, dc_ppc)
    dcac, dcac_name, _ = _build_dcac_from_rows(rows, ac_ppc, dc_ppc, build_objects=False, vbase_maps=vbase_maps)
    timings["build_dcac_s"] = time.perf_counter() - t0

    # 5. AC 网络对象构建
    t0 = time.perf_counter()
    ac_network = _build_ac_network_from_ppc(ac_ppc)
    timings["build_ac_network_s"] = time.perf_counter() - t0

    # 6. HybridPowerNetwork 构造(从这里走 read_from_file 类似路径)
    t0 = time.perf_counter()
    from hybrid_power_system_analysis.model.hybrid_model import HybridPowerNetwork
    # 用 build_hybrid_ppc_from_efile_rows 走另一条路
    ppc_full = build_hybrid_ppc_from_efile_rows(e_file, rows)
    timings["build_hybrid_ppc_full_s"] = time.perf_counter() - t0

    # 7. HybridPowerNetwork 真正构造(走 read_from_file 路径)
    t0 = time.perf_counter()
    network = HybridPowerNetwork.read_from_file(e_file)
    timings["hybrid_read_from_file_s"] = time.perf_counter() - t0

    # 8. AC topo
    t0 = time.perf_counter()
    network.ac.topo()
    timings["ac_topo_s"] = time.perf_counter() - t0

    # 9. DC topo
    t0 = time.perf_counter()
    network.dc.topo()
    timings["dc_topo_s"] = time.perf_counter() - t0

    # 10. _build_hybrid_topo
    t0 = time.perf_counter()
    network._build_hybrid_topo()
    timings["build_hybrid_topo_s"] = time.perf_counter() - t0

    # 11. construct calc
    t0 = time.perf_counter()
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    timings["construct_calc_s"] = time.perf_counter() - t0

    # 12. calc.prepare
    t0 = time.perf_counter()
    calc.prepare()
    timings["calc_prepare_s"] = time.perf_counter() - t0

    # 13. Newton solve
    t0 = time.perf_counter()
    rc = calc.run()
    timings["newton_solve_s"] = time.perf_counter() - t0

    # 14. End-to-end (不展开)
    gc.collect()
    t0 = time.perf_counter()
    n2 = HybridPowerNetwork.read_from_file(e_file)
    n2.prepare(False)
    c2 = HybridPowerFlowCalc(n2, tol=1e-8, max_iter=50, verbose=False, result_mode=result_mode)
    c2.prepare()
    c2.run()
    timings["total_e2e_s"] = time.perf_counter() - t0

    timings["nodes"] = network.total_nodes
    timings["ac_nodes"] = len(network.ac.nodes)
    timings["dc_nodes"] = len(network.dc.nodes)
    timings["converged"] = calc.converged
    timings["iter"] = calc.iterations
    timings["normF"] = float(calc.normF)
    return timings


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
    profile_hybrid(e_file, args.result_mode)
    gc.collect()

    print(f"Measuring {args.repeat} rounds (result_mode={args.result_mode})...")
    runs: List[Dict] = []
    for _ in range(args.repeat):
        runs.append(profile_hybrid(e_file, args.result_mode))
        gc.collect()

    keys = [
        "read_efile_rows_s", "build_ac_ppc_s", "build_dc_ppc_s", "build_dcac_s",
        "build_ac_network_s", "build_hybrid_ppc_full_s", "hybrid_read_from_file_s",
        "ac_topo_s", "dc_topo_s", "build_hybrid_topo_s",
        "construct_calc_s", "calc_prepare_s", "newton_solve_s", "total_e2e_s",
    ]
    agg = {"case": args.case, "e_file": str(e_file), "n_runs": len(runs),
           "result_mode": args.result_mode}
    for k in keys:
        vals = sorted([r[k] for r in runs])
        agg[k] = {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}
    agg["nodes"] = runs[0]["nodes"]
    agg["ac_nodes"] = runs[0]["ac_nodes"]
    agg["dc_nodes"] = runs[0]["dc_nodes"]
    agg["iter"] = runs[0]["iter"]
    agg["converged"] = runs[0]["converged"]
    agg["normF"] = runs[0]["normF"]

    print()
    print(f"=== {args.case} ({agg['ac_nodes']} AC + {agg['dc_nodes']} DC = {agg['nodes']} nodes, "
          f"{agg['iter']} iter, norm={agg['normF']:.2e}, mode={args.result_mode}) ===")
    print(f"{'phase':<32} {'min':<10} {'median':<10} {'% of total':<12}")
    print("-" * 65)
    total_med = agg["total_e2e_s"]["median"]
    for k in keys[:-1]:
        v = agg[k]["median"]
        pct = v / total_med * 100 if total_med > 0 else 0
        print(f"{k:<32} {agg[k]['min']:<10.4f} {v:<10.4f} {pct:<12.1f}")
    print("-" * 65)
    print(f"{'total_e2e_s':<32} {agg['total_e2e_s']['min']:<10.4f} {total_med:<10.4f} 100.0")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
