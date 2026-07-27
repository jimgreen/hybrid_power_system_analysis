"""cProfile only HybridStateEstimator.__init__ for cold-start hotspot discovery."""

import argparse
import cProfile
import gc
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_ROOT = SRC_DIR / "hybrid_power_system_analysis"
MODEL_DIR = PKG_ROOT / "model"
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_power_system_analysis.secore.hybrid_se import HybridStateEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-file", required=True)
    ap.add_argument("--meas-file", required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    e_file = Path(args.e_file)
    meas_file = Path(args.meas_file)

    print("Warmup 1 round...")
    HybridStateEstimator(e_file=e_file, meas_file=meas_file, flat_start=True)
    gc.collect()

    print("Profiling HybridStateEstimator.__init__ ...")
    prof = cProfile.Profile()
    prof.enable()
    HybridStateEstimator(e_file=e_file, meas_file=meas_file, flat_start=True)
    prof.disable()

    print(f"\n=== top {args.top} by cumulative time ===")
    st = pstats.Stats(prof)
    st.sort_stats("cumulative")
    st.print_stats(args.top)

    print(f"\n=== top {args.top} by self time ===")
    st.sort_stats("tottime")
    st.print_stats(args.top)


if __name__ == '__main__':
    main()
