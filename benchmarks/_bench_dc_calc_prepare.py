"""精确定位 dc_calc.prepare 内部:哪个步骤慢。"""

import argparse
import contextlib
import gc
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


def profile_dc_calc_prepare(e_file: Path):
    from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork
    from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc

    net = DCPowerNetwork()
    net.read_from_file(e_file)
    net.topo()
    calc = DCPowerFlowCalc(net, tol=1e-8, max_iter=50, verbose=False, result_mode="full")

    timings = {}
    orig_prepare = DCPowerFlowCalc.prepare

    # 1. ensure_dc_ppc_topology
    from hybrid_power_system_analysis.model.ppc_topology import ensure_dc_ppc_topology
    orig_ensure = ensure_dc_ppc_topology
    def t_ensure(ppc, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_ensure(ppc, *a, **kw)
        finally:
            timings["ensure_dc_ppc_topology_s"] = timings.get("ensure_dc_ppc_topology_s", 0) + (time.perf_counter() - t0)
    ensure_dc_ppc_topology = t_ensure
    import hybrid_power_system_analysis.lfcore.dc_lf as dcm
    dcm.ensure_dc_ppc_topology = t_ensure
    from hybrid_power_system_analysis.model import ppc_topology as pt
    pt.ensure_dc_ppc_topology = t_ensure

    # 2. _prepare_direct_ppc_topology
    orig_prep_dir = calc._prepare_direct_ppc_topology
    def t_prep_dir():
        t0 = time.perf_counter()
        try:
            return orig_prep_dir()
        finally:
            timings["_prepare_direct_ppc_topology_s"] = timings.get("_prepare_direct_ppc_topology_s", 0) + (time.perf_counter() - t0)
    calc._prepare_direct_ppc_topology = t_prep_dir

    # 3. _prepare_from_ppc 总
    orig_prep_from = calc._prepare_from_ppc
    def t_prep_from():
        t0 = time.perf_counter()
        try:
            return orig_prep_from()
        finally:
            timings["_prepare_from_ppc_total_s"] = timings.get("_prepare_from_ppc_total_s", 0) + (time.perf_counter() - t0)
    calc._prepare_from_ppc = t_prep_from

    t0 = time.perf_counter()
    calc.prepare()
    timings["calc_prepare_total_s"] = time.perf_counter() - t0
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--e-file", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    e_file = Path(args.e_file)
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    print("Warmup 1 round...")
    profile_dc_calc_prepare(e_file)
    gc.collect()

    print(f"Measuring {args.repeat} rounds...")
    runs = []
    for _ in range(args.repeat):
        runs.append(profile_dc_calc_prepare(e_file))
        gc.collect()

    keys = ["ensure_dc_ppc_topology_s", "_prepare_direct_ppc_topology_s",
            "_prepare_from_ppc_total_s", "calc_prepare_total_s"]
    print()
    print(f"=== {args.case} dc_calc.prepare 内部分项 ===")
    print(f"{'phase':<35} {'min':<10} {'median':<10} {'% of total':<12}")
    print("-" * 70)
    total_med = sorted([r["calc_prepare_total_s"] for r in runs])[len(runs) // 2]
    for k in keys[:-1]:
        vals = sorted([r[k] for r in runs])
        med = vals[len(vals) // 2]
        pct = med / total_med * 100 if total_med > 0 else 0
        print(f"{k:<35} {vals[0]:<10.4f} {med:<10.4f} {pct:<12.1f}")
    print("-" * 70)
    print(f"{'calc_prepare_total_s':<35} {sorted([r['calc_prepare_total_s'] for r in runs])[0]:<10.4f} "
          f"{total_med:<10.4f} 100.0")


if __name__ == "__main__":
    main()
