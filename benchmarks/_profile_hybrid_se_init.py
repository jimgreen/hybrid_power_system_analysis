"""Fine-grained constructor profiling for HybridStateEstimator cold start."""

import argparse
import gc
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


def profile_hybrid_se_init(e_file: Path, meas_file: Path):
    import hybrid_power_system_analysis.secore.hybrid_se as hy
    timings = {}

    # Wrap selected methods on the class during one constructor call.
    targets = [
        "_load_network",
        "_build_estimators",
        "_init_measurements",
        "_init_state_views",
        "_init_targeted_observability",
        "_init_result_cache",
    ]
    originals = {}

    def wrap_method(name, fn):
        def wrapped(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(self, *a, **kw)
            finally:
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)
        return wrapped

    for name in targets:
        fn = getattr(hy.HybridStateEstimator, name, None)
        if fn is not None:
            originals[name] = fn
            setattr(hy.HybridStateEstimator, name, wrap_method(name, fn))

    try:
        t0 = time.perf_counter()
        est = hy.HybridStateEstimator(e_file=e_file, meas_file=meas_file, flat_start=True)
        timings["total_init_s"] = time.perf_counter() - t0
        return timings, est
    finally:
        for name, fn in originals.items():
            setattr(hy.HybridStateEstimator, name, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-file", required=True)
    ap.add_argument("--meas-file", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    e_file = Path(args.e_file)
    meas_file = Path(args.meas_file)
    print("Measuring HybridStateEstimator.__init__ ...")
    runs = []
    for _ in range(args.repeat):
        gc.collect()
        timings, est = profile_hybrid_se_init(e_file, meas_file)
        runs.append(timings)

    keys = sorted({k for r in runs for k in r.keys()})
    agg = {}
    for k in keys:
        vals = sorted(r.get(k, 0.0) for r in runs)
        agg[k] = {
            "min": vals[0],
            "median": vals[len(vals)//2],
            "max": vals[-1],
        }

    print()
    print(f"{'phase':<32} {'min':<10} {'median':<10} {'max':<10}")
    print('-' * 70)
    for k in keys:
        print(f"{k:<32} {agg[k]['min']:<10.4f} {agg[k]['median']:<10.4f} {agg[k]['max']:<10.4f}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
