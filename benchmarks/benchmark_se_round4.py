"""Serial paired SE timings: fresh networks, no cross-instance timing caches."""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/hybrid_power_system_analysis")]
ROUND = ROOT / "output/se_optimization_20261116/round4"
CASES = [("ac", "ieee3k"), ("dc", "dc_net_3000"),
         ("hybrid", "hybrid_net_400"), ("hybrid", "qinling_100"),
         ("hybrid", "qinling_1000")]
MODULES = ("ac_se", "dc_se", "hybrid_se")


def run_variant(kind, case, variant, sources, repeat):
    import secore.ac_se as ac
    import secore.dc_se as dc
    import secore.hybrid_se as hybrid
    import model.meas_array_model as meas
    import secore.se_math as math
    for module, name in zip((ac, dc, hybrid), MODULES):
        exec(sources[variant][name], module.__dict__)
    classes = {"ac": ac.ACStateEstimator, "dc": dc.DCStateEstimator,
               "hybrid": hybrid.HybridStateEstimator}
    meas._MEAS_PPC_FILE_CACHE.clear()
    for name in ("_NORMAL_EQUATION_PATTERN_CACHE", "_SPARSE_PATTERN_EXPANSION_CACHE",
                 "_SPARSE_PATTERN_LINEAR_INDEX_CACHE", "_SPARSE_CSR_ROW_INDEX_CACHE"):
        getattr(math, name).clear()
    gc.collect()
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        estimator = classes[kind](e_file=ROOT / f"data/model/{kind}/{case}.e",
                                  meas_file=ROOT / f"data/meas/{kind}/{case}.meas",
                                  flat_start=True, max_iter=30, auto_prepare=False)
        estimator.prepare()
        prep = time.perf_counter() - start
        start = time.perf_counter()
        estimator.run(result_mode="array", skip_bad_data=True,
                      final_diagnostics=False, verbose=False)
        solve = time.perf_counter() - start
    result = estimator.estimate_result
    return dict(case=case, kind=kind, variant=variant, repeat=repeat,
                prepare_s=prep, run_s=solve, total_s=prep + solve,
                converged=bool(result.converged), iterations=int(result.iterations),
                states=int(result.x.size), measurements=int(result.residual.size),
                objective=float(result.objective), residual_inf=float(result.residual_inf),
                x=result.x.copy(), residual=result.residual.copy())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=ROUND / "baseline_src")
    parser.add_argument("--output", type=Path, default=ROUND / "paired")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sources, hashes = {}, {}
    for variant, folder in (("baseline", args.baseline_dir),
                            ("optimized", ROOT / "src/hybrid_power_system_analysis/secore")):
        sources[variant], hashes[variant] = {}, {}
        for name in MODULES:
            path = folder / (name + ".py")
            data = path.read_bytes()
            sources[variant][name] = compile(data, str(path), "exec")
            hashes[variant][name] = hashlib.sha256(data).hexdigest()
    cases = [(kind, case) for kind, case in CASES if not args.cases or case in args.cases]
    if not cases:
        parser.error("No matching cases")
    rows, arrays = [], {}
    for kind, case in cases:
        for repeat in range(args.repeats):
            order = ["baseline", "optimized"] if repeat % 2 == 0 else ["optimized", "baseline"]
            for variant in order:
                row = run_variant(kind, case, variant, sources, repeat)
                for field in ("x", "residual"):
                    arrays[f"{case}_{variant}_{repeat}_{field}"] = row.pop(field)
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.output / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    np.savez_compressed(args.output / "comparison.npz", **arrays)
    summaries = []
    for kind, case in cases:
        before = [r for r in rows if r["case"] == case and r["variant"] == "baseline"]
        after = [r for r in rows if r["case"] == case and r["variant"] == "optimized"]
        for b, a in zip(before, after):
            assert b["converged"] and a["converged"], case
            for key in ("iterations", "states", "measurements", "objective", "residual_inf"):
                assert b[key] == a[key], (case, key, b[key], a[key])
            for field in ("x", "residual"):
                old = arrays[f"{case}_baseline_{b['repeat']}_{field}"]
                new = arrays[f"{case}_optimized_{a['repeat']}_{field}"]
                assert np.isfinite(old).all() and np.isfinite(new).all()
                np.testing.assert_array_equal(old, new)
        summary = dict(case=case, kind=kind, samples=args.repeats,
                       states=before[0]["states"], measurements=before[0]["measurements"],
                       iterations=before[0]["iterations"], exact_equal=True)
        for phase in ("prepare", "run", "total"):
            old = [r[phase + "_s"] for r in before]
            new = [r[phase + "_s"] for r in after]
            summary[phase] = dict(before_s=statistics.median(old), after_s=statistics.median(new),
                                  speedup=statistics.median(old) / statistics.median(new),
                                  paired_speedup=statistics.median(b / a for b, a in zip(old, new)),
                                  before_range_s=[min(old), max(old)], after_range_s=[min(new), max(new)])
        summaries.append(summary)
    input_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for kind, case in cases
                    for p in (ROOT / f"data/model/{kind}/{case}.e", ROOT / f"data/meas/{kind}/{case}.meas")}
    payload = dict(baseline=str(args.baseline_dir), source_sha256=hashes, input_sha256=input_hashes,
                   python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                   repeats=args.repeats, cases=summaries, rows=rows,
                   cache_policy="fresh instances; measurement and sparse-math caches cleared per sample; no network/result cache; OS cache unchanged",
                   options=dict(flat_start=True, max_iter=30, result_mode="array", skip_bad_data=True,
                                final_diagnostics=False, blas_threads=1))
    (args.output / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
