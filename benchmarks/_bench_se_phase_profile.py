"""Cold-start array-mode SE phase + cProfile breakdown for a single case."""

import argparse
import cProfile
import contextlib
import io
import pstats
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

from hybrid_power_system_analysis.secore.ac_se import ACStateEstimator  # noqa: E402
from hybrid_power_system_analysis.secore.dc_se import DCStateEstimator  # noqa: E402
from hybrid_power_system_analysis.secore.hybrid_se import HybridStateEstimator  # noqa: E402

DATA = PROJECT_ROOT / "data"
CASE_MAP = {
    "ieee300":       ("ac", DATA / "model/ac/ieee300.e", DATA / "meas/ac/ieee300.meas"),
    "ieee3k":        ("ac", DATA / "model/ac/ieee3k.e", DATA / "meas/ac/ieee3k.meas"),
    "ieee1w":        ("ac", DATA / "model/ac/ieee1w.e", DATA / "meas/ac/ieee1w.meas"),
    "ieee3w":        ("ac", DATA / "model/ac/ieee3w.e", DATA / "meas/ac/ieee3w.meas"),
    "dc_net_1000":   ("dc", DATA / "model/dc/dc_net_1000.e", DATA / "meas/dc/dc_net_1000.meas"),
    "dc_net_3000":   ("dc", DATA / "model/dc/dc_net_3000.e", DATA / "meas/dc/dc_net_3000.meas"),
    "dc_net_3w":     ("dc", DATA / "model/dc/dc_net_3w.e", DATA / "meas/dc/dc_net_3w.meas"),
    "qinling_100":   ("hybrid", DATA / "model/hybrid/qinling_100.e", DATA / "meas/hybrid/qinling_100.meas"),
    "qinling_1000":  ("hybrid", DATA / "model/hybrid/qinling_1000.e", DATA / "meas/hybrid/qinling_1000.meas"),
    "hybrid_net_4k": ("hybrid", DATA / "model/hybrid/hybrid_net_4k.e", DATA / "meas/hybrid/hybrid_net_4k.meas"),
    "hybrid_net_4w": ("hybrid", DATA / "model/hybrid/hybrid_net_4w.e", DATA / "meas/hybrid/hybrid_net_4w.meas"),
}
EST_CLS = {"ac": ACStateEstimator, "dc": DCStateEstimator, "hybrid": HybridStateEstimator}


def run_once(case_name, *, profile=False):
    kind, e_file, meas_file = CASE_MAP[case_name]
    est_cls = EST_CLS[kind]
    est = est_cls(e_file=e_file, meas_file=meas_file, flat_start=True, profile=True)
    with contextlib.redirect_stdout(io.StringIO()):
        est.run(result_mode="array", verbose=False)
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs before profiling (strip import cost)")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    # Warmup to strip one-time import/JIT cost so the phase breakdown reflects steady cold compute.
    for _ in range(args.warmup):
        run_once(args.case)

    t0 = time.perf_counter()
    est = run_once(args.case)
    total = time.perf_counter() - t0

    print(f"\n=== {args.case} phase breakdown (total run={total:.4f}s) ===")
    times = est.profile_times
    rows = sorted(times.items(), key=lambda kv: kv[1], reverse=True)
    for name, val in rows:
        if val >= 5e-4:
            print(f"  {name:<48} {val:.4f}s  ({100*val/total:5.1f}%)")

    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        run_once(args.case)
        pr.disable()
        st = pstats.Stats(pr)
        st.sort_stats("cumulative")
        print(f"\n=== cProfile cumulative top {args.topn} ===")
        st.print_stats(args.topn)
        st.sort_stats("tottime")
        print(f"\n=== cProfile tottime top {args.topn} ===")
        st.print_stats(args.topn)


if __name__ == "__main__":
    main()
