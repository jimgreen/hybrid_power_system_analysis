"""精确定位 HybridPowerFlowCalc.prepare 内部: ac_calc.prepare vs dc_calc.prepare vs _prepare_dcac/acac。"""

import argparse
import contextlib
import gc
import io
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


def profile_calc_prepare(e_file: Path):
    from hybrid_power_system_analysis.model.hybrid_model import HybridPowerNetwork
    from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc

    network = HybridPowerNetwork.read_from_file(e_file)
    network.prepare(False)
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode="full")

    timings = {}
    # Wrap ac_calc.prepare
    orig_ac = calc.ac_calc.prepare
    orig_dc = calc.dc_calc.prepare
    orig_dcac = calc._prepare_dcac_converters
    orig_acac = calc._prepare_acac_converters

    def t_ac(self):
        t0 = time.perf_counter()
        try:
            return orig_ac()
        finally:
            timings["ac_calc_prepare_s"] = timings.get("ac_calc_prepare_s", 0) + (time.perf_counter() - t0)

    def t_dc(self):
        t0 = time.perf_counter()
        try:
            return orig_dc()
        finally:
            timings["dc_calc_prepare_s"] = timings.get("dc_calc_prepare_s", 0) + (time.perf_counter() - t0)

    def t_dcac(self):
        t0 = time.perf_counter()
        try:
            return orig_dcac()
        finally:
            timings["prepare_dcac_s"] = timings.get("prepare_dcac_s", 0) + (time.perf_counter() - t0)

    def t_acac(self):
        t0 = time.perf_counter()
        try:
            return orig_acac()
        finally:
            timings["prepare_acac_s"] = timings.get("prepare_acac_s", 0) + (time.perf_counter() - t0)

    calc.ac_calc.prepare = t_ac.__get__(calc.ac_calc)
    calc.dc_calc.prepare = t_dc.__get__(calc.dc_calc)
    calc._prepare_dcac_converters = t_dcac.__get__(calc)
    calc._prepare_acac_converters = t_acac.__get__(calc)

    t0 = time.perf_counter()
    calc.prepare()
    total = time.perf_counter() - t0
    timings["total_calc_prepare_s"] = total
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
    profile_calc_prepare(e_file)
    gc.collect()

    print(f"Measuring {args.repeat} rounds...")
    runs = []
    for _ in range(args.repeat):
        runs.append(profile_calc_prepare(e_file))
        gc.collect()

    keys = ["ac_calc_prepare_s", "dc_calc_prepare_s", "prepare_dcac_s",
            "prepare_acac_s", "total_calc_prepare_s"]
    print()
    print(f"=== {args.case} calc.prepare 内部分项 ===")
    print(f"{'phase':<28} {'min':<10} {'median':<10} {'% of total':<12}")
    print("-" * 60)
    total_med = sorted([r["total_calc_prepare_s"] for r in runs])[len(runs) // 2]
    for k in keys[:-1]:
        vals = sorted([r[k] for r in runs])
        med = vals[len(vals) // 2]
        pct = med / total_med * 100 if total_med > 0 else 0
        print(f"{k:<28} {vals[0]:<10.4f} {med:<10.4f} {pct:<12.1f}")
    print("-" * 60)
    print(f"{'total_calc_prepare_s':<28} {sorted([r['total_calc_prepare_s'] for r in runs])[0]:<10.4f} "
          f"{total_med:<10.4f} 100.0")


if __name__ == "__main__":
    main()
