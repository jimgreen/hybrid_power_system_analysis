"""Serial A/B benchmark of independent, freshly loaded LF calculations.

Baseline is a source snapshot of the pre-change worktree, including local edits.
No network/result cache or previous solved state is reused between samples.
"""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import scipy

    root = Path(args.repo)
    source = Path(args.source)
    package = source / 'hybrid_power_system_analysis'
    sys.path[:0] = [str(source), str(package), str(package / 'model'), str(package / 'lfcore')]
    from lfcore.ac_lf import ACPowerFlowCalc
    from lfcore.dc_lf import DCPowerFlowCalc
    from lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork
    from model.dc_model import DCPowerNetwork
    from model.ac_model import ACPowerNetwork

    kind, name, path_kind = args.case.split(':')
    cls = dict(ac=ACPowerFlowCalc, dc=DCPowerFlowCalc, hybrid=HybridPowerFlowCalc)[kind]
    path = root / 'data/model' / kind / (name + '.e')
    options = dict(tol=1e-8, max_iter=50, linear_solver=args.solver,
                   result_mode=args.mode, verbose=False, parameter_file=root / 'data/lf/lf.para')
    snapshots = {}

    def capture(prefix, obj):
        if isinstance(obj, np.ndarray) and obj.dtype.kind in 'biufcUS':
            snapshots[prefix] = obj.copy()
        elif isinstance(obj, dict):
            for key, value in obj.items():
                capture(prefix + '/' + key, value)
        elif isinstance(obj, (bool, int, float, np.number)):
            snapshots[prefix] = np.asarray(obj)

    timings = []
    for iteration in range(args.repeat + 1):
        gc.collect()
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            if path_kind == 'fast':
                calc = cls.from_file_fast(path, **options)
            else:
                if kind == 'hybrid':
                    network = HybridPowerNetwork.read_from_file(path)
                elif kind == 'dc':
                    network = DCPowerNetwork()
                    network.read_from_file(path)
                else:
                    network = ACPowerNetwork()
                    network.read_from_file(path)
                calc = cls(network, **options)
            t1 = time.perf_counter()
            calc.prepare()
            t2 = time.perf_counter()
            rc = calc.run()
            t3 = time.perf_counter()
        if rc != 0 or not calc.converged:
            raise RuntimeError(f'{args.case} {args.mode}: rc={rc}, normF={calc.normF}')
        if iteration:
            timings.append(dict(load_ms=(t1-t0)*1000, prepare_ms=(t2-t1)*1000,
                                run_ms=(t3-t2)*1000, total_ms=(t3-t0)*1000))
        if iteration == 1:
            capture('x', calc.x)
            jacobian = calc.get_jacobi(calc.x).tocsc()
            capture('jacobian/indptr', jacobian.indptr)
            capture('jacobian/indices', jacobian.indices)
            capture('jacobian/data', jacobian.data)
            capture('result', calc.result)
            for side in ('ac', 'dc'):
                sub = getattr(calc, side + '_calc', None)
                if sub is not None:
                    capture(side + '_result', sub.result)
            # Compare every numeric full-object field as well as array tables.
            full = getattr(calc, 'lf_result', None)
            full_sides = [('full', full)]
            if kind == 'hybrid' and full is not None:
                full_sides += [(s, getattr(full, s, None)) for s in ('ac', 'dc', 'dcac')]
                full_sides += [('fluid_' + key, value) for key, value in getattr(full, 'fluid', {}).items()]
            if path_kind == 'object':
                full_sides += [('network', network)]
                if kind == 'hybrid':
                    full_sides += [('network_ac', network.ac), ('network_dc', network.dc)]
            for label, result in full_sides:
                if result is None:
                    continue
                capture(label + '/arrays', getattr(result, 'arrays', {}))
                for collection in ('nodes', 'generators', 'loads', 'branches', 'transformers',
                                   'three_winding_transformers', 'zero_branches', 'breakers', 'acac_converters',
                                   'dcdc_converters', 'dcac_converters', 'sources', 'storages', 'edges'):
                    devices = getattr(result, collection, {})
                    if isinstance(devices, dict):
                        capture(label + '/' + collection + '/keys', np.asarray(sorted(devices), dtype=str))
                        devices = [devices[key] for key in sorted(devices)]
                    if not devices:
                        continue
                    capture(label + '/' + collection + '/fields', np.asarray(
                        [repr(sorted(vars(device))) for device in devices], dtype=str))
                    fields = sorted(set.intersection(*(set(vars(device)) for device in devices)))
                    for field in fields:
                        values = [getattr(device, field) for device in devices]
                        if all(isinstance(v, (bool, int, float, np.number)) for v in values):
                            capture(label + '/' + collection + '/' + field, np.asarray(values))
        status = dict(rc=int(rc), converged=bool(calc.converged), iterations=int(calc.iterations),
                      normF=float(calc.normF), resolved_solver=calc._linear_solver_resolved)
        del calc
    result = dict(case=args.case, mode=args.mode, status=status, samples=timings,
                  median={k:statistics.median(row[k] for row in timings) for k in timings[0]},
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                  threads={key:os.environ.get(key) for key in
                           ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out.with_suffix('.npz'), **snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--baseline-src')
    parser.add_argument('--source')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--solver', default='pyklu')
    parser.add_argument('--case')
    parser.add_argument('--mode')
    parser.add_argument('--cases', nargs='+', default=[
        'ac:ieee3k:fast', 'ac:ieee3w:fast', 'dc:dc_net_3000:fast',
        'dc:dc_net_3000:object', 'hybrid:qinling_100:fast',
        'hybrid:qinling_100:object', 'hybrid:qinling_1000:fast'])
    parser.add_argument('--modes', nargs='+', default=['full','array','summary','none'])
    args = parser.parse_args()
    if args.source:
        worker(args)
        return
    import numpy as np
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for case_no, case in enumerate(args.cases):
        for mode_no, mode in enumerate(args.modes):
            results = {}
            versions = [('baseline', Path(args.baseline_src)), ('optimized', Path(args.repo)/'src')]
            if (case_no + mode_no) % 2:
                versions.reverse()
            for label, source in versions:
                dest = output / (case.replace(':','_') + '_' + mode + '_' + label + '.json')
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                command = [sys.executable, '-B', __file__, '--repo', args.repo, '--source', str(source),
                           '--case', case, '--mode', mode, '--repeat', str(args.repeat),
                           '--solver', args.solver, '--out', str(dest)]
                subprocess.run(command, check=True, env=env)
                results[label] = json.loads(dest.read_text(encoding='utf-8'))
            basefile = output / (case.replace(':','_') + '_' + mode + '_baseline.npz')
            newfile = output / (case.replace(':','_') + '_' + mode + '_optimized.npz')
            max_error = 0.
            with np.load(basefile) as before, np.load(newfile) as after:
                assert set(before.files) == set(after.files)
                for key in before.files:
                    a, b = before[key], after[key]
                    assert a.shape == b.shape and a.dtype == b.dtype, key
                    if a.dtype.kind in 'US':
                        np.testing.assert_array_equal(a, b, err_msg=key)
                        continue
                    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10, err_msg=key)
                    if a.size and a.dtype.kind != 'b':
                        finite = np.isfinite(a) & np.isfinite(b)
                        if np.any(finite):
                            max_error = max(max_error, float(np.max(np.abs(a[finite]-b[finite]))))
            before, after = results['baseline'], results['optimized']
            assert before['input_sha256'] == after['input_sha256']
            assert before['status'] == after['status']
            row = dict(case=case, mode=mode, baseline=before['median'], optimized=after['median'],
                       speedup=before['median']['total_ms']/after['median']['total_ms'],
                       max_abs_error=max_error, status=after['status'], samples=args.repeat)
            comparisons.append(row)
            print(f'{case:32} {mode:7} {row["baseline"]["total_ms"]:8.2f} -> '
                  f'{row["optimized"]["total_ms"]:8.2f} ms  {row["speedup"]:.2f}x  error={max_error:.2e}', flush=True)
            (output/'comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
