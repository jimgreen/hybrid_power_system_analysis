"""A/B benchmark for AC/DC reusable factorizer.

Compares `make_reusable_factorizer` enabled vs disabled under the same solver.
"""

import argparse
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


def bench(calc_cls, module, path, solver, use_reuse: bool, repeat: int = 5):
    orig = module._make_reusable_factorizer
    if not use_reuse:
        module._make_reusable_factorizer = lambda matrix, resolved_name: None
    vals = []
    try:
        for _ in range(repeat):
            gc.collect()
            t0 = time.perf_counter()
            c = calc_cls.from_file_fast(path, tol=1e-8, max_iter=50, verbose=False, result_mode='array', linear_solver=solver)
            c.prepare(); c.run()
            vals.append(time.perf_counter() - t0)
        vals.sort()
        return vals[0], vals[len(vals)//2], vals[-1]
    finally:
        module._make_reusable_factorizer = orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()

    import hybrid_power_system_analysis.lfcore.ac_lf as ac_mod
    import hybrid_power_system_analysis.lfcore.dc_lf as dc_mod
    from hybrid_power_system_analysis.lfcore.ac_lf import ACPowerFlowCalc
    from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc

    tests = [
        ('ac', 'data/model/ac/ieee3w.e', ACPowerFlowCalc, ac_mod, 'umfpack'),
        ('dc', 'data/model/dc/dc_net_3w.e', DCPowerFlowCalc, dc_mod, 'umfpack'),
    ]
    for kind, path, cls, mod, solver in tests:
        off = bench(cls, mod, path, solver, use_reuse=False, repeat=args.repeat)
        on = bench(cls, mod, path, solver, use_reuse=True, repeat=args.repeat)
        print(f'{kind} {solver}')
        print(f'  reuse off: min={off[0]:.4f}s med={off[1]:.4f}s max={off[2]:.4f}s')
        print(f'  reuse on : min={on[0]:.4f}s med={on[1]:.4f}s max={on[2]:.4f}s')
        print(f'  delta min={(on[0]-off[0])/off[0]*100:+.1f}% delta med={(on[1]-off[1])/off[1]*100:+.1f}%')


if __name__ == '__main__':
    main()
