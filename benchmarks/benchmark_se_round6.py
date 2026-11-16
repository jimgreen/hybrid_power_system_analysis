"""Alternating, fresh-network A/B runs against the saved working-tree baseline."""
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

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/se_optimization_20261116/round6'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/hybrid_power_system_analysis')]
import numpy as np
import scipy
import model.meas_array_model as meas
import secore.se_result as result_module
import secore.ac_se as ac
import secore.dc_se as dc
import secore.hybrid_se as hybrid
import secore.se_math as math

MODULES=[(meas,'model/meas_array_model.py'),(result_module,'secore/se_result.py'),
         (ac,'secore/ac_se.py'),(dc,'secore/dc_se.py'),(hybrid,'secore/hybrid_se.py')]
CASES=[('ac','ieee3k'),('dc','dc_net_3000'),('hybrid','hybrid_net_400'),
       ('hybrid','qinling_100'),('hybrid','qinling_1000')]

def clear():
    meas._MEAS_PPC_FILE_CACHE.clear()
    for name in ('_NORMAL_EQUATION_PATTERN_CACHE','_SPARSE_PATTERN_EXPANSION_CACHE',
                 '_SPARSE_PATTERN_LINEAR_INDEX_CACHE','_SPARSE_CSR_ROW_INDEX_CACHE'):
        getattr(math,name).clear()
    gc.collect()

def full_digest(result):
    if result is None:
        return None
    h=hashlib.sha256(repr(vars(result.statistics)).encode())
    for key in ('normal_measurements','pseudo_measurements','prefiltered_measurements','bad_data'):
        h.update(key.encode())
        for row in getattr(result,key):
            h.update(repr(vars(row)).encode())
    return h.hexdigest()

def run(side,case,mode,variant,compiled,repeat):
    for module,path in MODULES:
        exec(compiled[variant][path],module.__dict__)
    clear()
    cls={'ac':ac.ACStateEstimator,'dc':dc.DCStateEstimator,'hybrid':hybrid.HybridStateEstimator}[side]
    with contextlib.redirect_stdout(io.StringIO()):
        start=time.perf_counter()
        est=cls(e_file=ROOT/f'data/model/{side}/{case}.e',meas_file=ROOT/f'data/meas/{side}/{case}.meas',
                flat_start=True,max_iter=30,auto_prepare=False)
        est.prepare()
        prep=time.perf_counter()-start
        original=est.build_se_result
        result_time=[]
        def timed_result(*args,**kwargs):
            start=time.perf_counter()
            value=original(*args,**kwargs)
            result_time.append(time.perf_counter()-start)
            return value
        est.build_se_result=timed_result
        start=time.perf_counter()
        if mode=='array':
            output=est.run(result_mode='array',skip_bad_data=True,final_diagnostics=False)
        else:
            output=est.run(verbose=False)
        run_time=time.perf_counter()-start
    r=est.estimate_result
    return dict(case=case,side=side,mode=mode,variant=variant,repeat=repeat,
        prepare_s=prep,run_s=run_time,total_s=prep+run_time,result_build_s=sum(result_time),
        converged=bool(r.converged),iterations=int(r.iterations),states=len(r.x),measurements=len(r.residual),
        objective=float(r.objective),residual_inf=float(r.residual_inf),
        x_sha256=hashlib.sha256(r.x.tobytes()).hexdigest(),residual_sha256=hashlib.sha256(r.residual.tobytes()).hexdigest(),
        full_result_sha256=full_digest(output),bad_count=len(est.bad_items),removed_count=len(est.removed_bad_data))

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--cases',nargs='*')
    ap.add_argument('--output',type=Path,default=OUT/'paired')
    args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    roots={'baseline':OUT/'baseline_src/hybrid_power_system_analysis','optimized':ROOT/'src/hybrid_power_system_analysis'}
    compiled={};hashes={}
    for variant,folder in roots.items():
        compiled[variant]={};hashes[variant]={}
        for _,path in MODULES:
            data=(folder/path).read_bytes()
            compiled[variant][path]=compile(data,str(folder/path),'exec')
            hashes[variant][path]=hashlib.sha256(data).hexdigest()
    rows=[];summaries=[]
    for side,case in CASES:
        if args.cases and case not in args.cases:
            continue
        for mode in ('array','full') if case!='qinling_1000' else ('array',):
            for repeat in range(args.repeats):
                pair={}
                for variant in ('baseline','optimized') if repeat%2==0 else ('optimized','baseline'):
                    r=run(side,case,mode,variant,compiled,repeat)
                    pair[variant]=r;rows.append(r)
                    (args.output/'samples.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
                for key in ('converged','iterations','states','measurements','objective','residual_inf',
                            'x_sha256','residual_sha256','full_result_sha256','bad_count','removed_count'):
                    assert pair['baseline'][key]==pair['optimized'][key],(case,mode,key)
                assert pair['optimized']['converged']
                print(case,mode,repeat,'B/A ms',round(pair['baseline']['total_s']*1000,2),round(pair['optimized']['total_s']*1000,2),flush=True)
            b=[r for r in rows if r['case']==case and r['mode']==mode and r['variant']=='baseline']
            a=[r for r in rows if r['case']==case and r['mode']==mode and r['variant']=='optimized']
            item=dict(case=case,side=side,mode=mode,exact_equal=True,states=a[0]['states'],measurements=a[0]['measurements'])
            for phase in ('prepare','run','total','result_build'):
                before=[r[phase+'_s'] for r in b];after=[r[phase+'_s'] for r in a]
                item[phase]=dict(before_s=statistics.median(before),after_s=statistics.median(after),
                    speedup=statistics.median(before)/statistics.median(after),
                    paired_speedup=statistics.median(x/y for x,y in zip(before,after)),
                    before_range_s=[min(before),max(before)],after_range_s=[min(after),max(after)])
            summaries.append(item)
    data=dict(cases=summaries,rows=rows,source_sha256=hashes,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),repeats=args.repeats,source_unchanged_during_run=all(
            hashlib.sha256((folder/path).read_bytes()).hexdigest()==hashes[v][path] for v,folder in roots.items() for _,path in MODULES),
        cache_policy='fresh estimators and network rebuilds; global measurement/sparse math caches cleared every sample; OS cache unchanged',
        options=dict(flat_start=True,max_iter=30,blas_threads=1),
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for side,case in CASES
            for p in (ROOT/f'data/model/{side}/{case}.e',ROOT/f'data/meas/{side}/{case}.meas')})
    assert data['source_unchanged_during_run']
    (args.output/'comparison.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
