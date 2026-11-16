"""Compare SE revisions with identical inputs; serial timing, independent snapshots.

Example: python benchmarks/benchmark_se_optimization.py --baseline-ref 9ff31a1 --output output/se_before
         python benchmarks/benchmark_se_optimization.py --output output/se_after
Every sample builds a new estimator, rereads the model, rebuilds topology, and
clears the existing measurement-file cache. OS file caches are not flushed.
Only the three SE modules change for the baseline; all other dependencies and
working-tree modifications stay identical. Do not run benchmarks concurrently.
"""
import argparse
import contextlib
import gc
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/hybrid_power_system_analysis')]
import numpy as np
import scipy
import secore.ac_se as ac
import secore.dc_se as dc
import secore.hybrid_se as hybrid
import model.meas_array_model as meas

CASES = [('ac', 'ieee3k'), ('dc', 'dc_net_3000'),
         ('hybrid', 'hybrid_net_400'), ('hybrid', 'qinling_100'), ('hybrid', 'qinling_1000')]


def main():
    parser = argparse.ArgumentParser(__doc__)
    revisions = parser.add_mutually_exclusive_group()
    revisions.add_argument('--baseline-ref')
    revisions.add_argument('--compare-ref', help='Alternate baseline/working-tree order within each repetition')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--profile', action='store_true', help='Separate stage profiling run; adds instrumentation overhead')
    args = parser.parse_args()
    modules = (ac, dc, hybrid)
    revision = args.baseline_ref or args.compare_ref
    sources = {'working-tree': [compile(Path(m.__file__).read_bytes(), m.__file__, 'exec') for m in modules]}
    if revision:
        sources[revision] = []
        for module in modules:
            path = Path(module.__file__).relative_to(ROOT).as_posix()
            source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)
            sources[revision].append(compile(source, f'{revision}/{path}', 'exec'))
    rows, arrays = [], {}
    for side, case in CASES:
        if args.cases and case not in args.cases:
            continue
        for repeat in range(args.repeats):
            variants = [args.baseline_ref or 'working-tree']
            if args.compare_ref:
                variants = [args.compare_ref, 'working-tree']
                if repeat % 2:
                    variants.reverse()
            for variant in variants:
                # Restore all three classes together, outside the timed block.
                # Each sample still parses fresh input and builds fresh topology.
                for module, source in zip(modules, sources[variant]):
                    exec(source, module.__dict__)
                classes = {'ac': ac.ACStateEstimator, 'dc': dc.DCStateEstimator, 'hybrid': hybrid.HybridStateEstimator}
                meas._MEAS_PPC_FILE_CACHE.clear()
                gc.collect()
                with contextlib.redirect_stdout(io.StringIO()):
                    start = time.perf_counter()
                    est = classes[side](e_file=ROOT / f'data/model/{side}/{case}.e',
                                        meas_file=ROOT / f'data/meas/{side}/{case}.meas',
                                        flat_start=True, max_iter=30, auto_prepare=False, profile=args.profile)
                    est.prepare()
                    prep = time.perf_counter() - start
                    start = time.perf_counter()
                    est.run(result_mode='array', skip_bad_data=True, final_diagnostics=False, verbose=False)
                    solve = time.perf_counter() - start
                r = est.estimate_result
                key = f'{case}_{variant}_{repeat}'
                arrays[key + '_x'] = r.x.copy()
                arrays[key + '_residual'] = r.residual.copy()
                rows.append(dict(case=case, revision=variant, temperature='uncached', repeat=repeat,
                                 prepare_s=prep, run_s=solve, total_s=prep+solve,
                                 converged=bool(r.converged), iterations=int(r.iterations),
                                 states=int(r.x.size), measurements=int(r.residual.size),
                                 objective=float(r.objective), residual_inf=float(r.residual_inf)))
                if args.profile:
                    rows[-1]['profile_times'] = dict(est.profile_times)
                print(json.dumps(rows[-1]), flush=True)
                del est, r
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        revision=args.compare_ref or args.baseline_ref or 'working-tree', paired=bool(args.compare_ref), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        cache_policy='No network cache. Measurement-file cache cleared before every sample. OS file cache unchanged.',
        options=dict(flat_start=True, max_iter=30, result_mode='array', skip_bad_data=True, final_diagnostics=False),
        runs=rows), indent=2), encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)


if __name__ == '__main__':
    main()
