"""Quick SE split: constructor vs estimate only."""

import argparse, contextlib, io, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
PKG_ROOT = SRC_DIR / 'hybrid_power_system_analysis'
MODEL_DIR = PKG_ROOT / 'model'
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from hybrid_power_system_analysis.secore.ac_se import ACStateEstimator
from hybrid_power_system_analysis.secore.dc_se import DCStateEstimator
from hybrid_power_system_analysis.secore.hybrid_se import HybridStateEstimator

MAP = {
 'ieee3k': ('ac', PROJECT_ROOT/'data/model/ac/ieee3k.e', PROJECT_ROOT/'data/meas/ac/ieee3k.meas'),
 'dc_net_3000': ('dc', PROJECT_ROOT/'data/model/dc/dc_net_3000.e', PROJECT_ROOT/'data/meas/dc/dc_net_3000.meas'),
 'qinling_100': ('hybrid', PROJECT_ROOT/'data/model/hybrid/qinling_100.e', PROJECT_ROOT/'data/meas/hybrid/qinling_100.meas'),
}
CLS = {'ac': ACStateEstimator, 'dc': DCStateEstimator, 'hybrid': HybridStateEstimator}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('case'); args=ap.parse_args()
    kind,e,m = MAP[args.case]; cls = CLS[kind]
    t0=time.perf_counter(); est=cls(e_file=e, meas_file=m, flat_start=True); t1=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        res=est.estimate(verbose=False)
    t2=time.perf_counter()
    print(f'case={args.case} kind={kind}')
    print(f'init_s={t1-t0:.4f}')
    print(f'estimate_s={t2-t1:.4f}')
    print(f'total_s={t2-t0:.4f}')
    print(f'iter={res.iterations} resid={res.residual_inf:.3e} obs={res.observability.observable} ok={res.converged}')
if __name__=='__main__':
    main()
