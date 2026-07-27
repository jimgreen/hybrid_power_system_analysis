"""cProfile 整个端到端流程,按累计时间和单次时间排序,精确到函数名。"""

import argparse
import cProfile
import contextlib
import gc
import io
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


def run_full(e_file: Path):
    """模拟用户的端到端使用。"""
    from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork
    network = HybridPowerNetwork.read_from_file(e_file)
    network.prepare(False)
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False, result_mode="full")
    calc.prepare()
    calc.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--e-file", required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    e_file = Path(args.e_file)
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    # Warmup
    print("Warmup 1 round...")
    run_full(e_file)
    gc.collect()

    # Profile
    print("Profiling 1 round...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_full(e_file)
    profiler.disable()

    print(f"\n=== {args.case} cProfile top {args.top} by cumulative time ===")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top)

    print(f"\n=== {args.case} cProfile top {args.top} by tottime (self time) ===")
    stats.sort_stats("tottime")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
