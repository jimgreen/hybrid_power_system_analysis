"""Compare old HybridPowerNetwork path vs from_file_fast PPC-only path.

Checks:
- rc / converged / iter / normF
- AC voltage head (first 10)
- DC voltage head (first 10)
- L2 norm difference of bus voltage arrays when available
"""

import argparse
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
from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork  # noqa: E402


def run_old(path: Path):
    t0 = time.perf_counter()
    net = HybridPowerNetwork.read_from_file(path)
    calc = HybridPowerFlowCalc(net, tol=1e-8, max_iter=50, verbose=False, result_mode="array")
    calc.prepare()
    rc = calc.run()
    dt = time.perf_counter() - t0
    return calc, rc, dt


def run_new(path: Path):
    t0 = time.perf_counter()
    calc = HybridPowerFlowCalc.from_file_fast(path, tol=1e-8, max_iter=50, verbose=False, result_mode="array")
    calc.prepare()
    rc = calc.run()
    dt = time.perf_counter() - t0
    return calc, rc, dt


def extract_voltages(calc):
    ac_v = None
    dc_v = None
    if calc.ac_calc is not None and isinstance(getattr(calc.ac_calc, "result", None), dict):
        bus = calc.ac_calc.result.get("bus")
        if bus is not None and bus.shape[1] > 7:
            ac_v = bus[:, 7].copy()
    if calc.dc_calc is not None and isinstance(getattr(calc.dc_calc, "result", None), dict):
        bus = calc.dc_calc.result.get("bus")
        if bus is not None and bus.shape[1] > 1:
            dc_v = bus[:, 1].copy() if bus.shape[1] > 1 else bus[:, 0].copy()
    return ac_v, dc_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-file", required=True)
    args = parser.parse_args()

    e_file = Path(args.e_file)
    old, rc_old, dt_old = run_old(e_file)
    new, rc_new, dt_new = run_new(e_file)

    ac_old, dc_old = extract_voltages(old)
    ac_new, dc_new = extract_voltages(new)

    report = {
        "file": str(e_file),
        "old": {"rc": rc_old, "converged": old.converged, "iter": old.iterations, "normF": old.normF, "dt_s": dt_old},
        "new": {"rc": rc_new, "converged": new.converged, "iter": new.iterations, "normF": new.normF, "dt_s": dt_new},
    }

    print(f"file: {e_file}")
    print(f"old: rc={rc_old}, ok={old.converged}, iter={old.iterations}, norm={old.normF:.3e}, dt={dt_old:.4f}s")
    print(f"new: rc={rc_new}, ok={new.converged}, iter={new.iterations}, norm={new.normF:.3e}, dt={dt_new:.4f}s")

    if ac_old is not None and ac_new is not None:
        diff = float(np.linalg.norm(ac_old - ac_new))
        print(f"AC voltage L2 diff: {diff:.3e}")
        print(f"AC head old: {ac_old[:8].tolist()}")
        print(f"AC head new: {ac_new[:8].tolist()}")
        report["ac_l2_diff"] = diff
    if dc_old is not None and dc_new is not None:
        diff = float(np.linalg.norm(dc_old - dc_new))
        print(f"DC voltage L2 diff: {diff:.3e}")
        print(f"DC head old: {dc_old[:8].tolist()}")
        print(f"DC head new: {dc_new[:8].tolist()}")
        report["dc_l2_diff"] = diff

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
