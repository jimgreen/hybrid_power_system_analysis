"""Fresh-file parser timings and isolated-process SE peak working sets."""
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

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/se_optimization_20261116/round6'
ROOTS={'baseline':OUT/'baseline_src','optimized':ROOT/'src'}
ap=argparse.ArgumentParser(__doc__)
ap.add_argument('--memory-worker',choices=ROOTS)
ap.add_argument('--case',default='ac/ieee3k')
ap.add_argument('--mode',choices=['array','full'],default='full')
args=ap.parse_args()
src=ROOTS.get(args.memory_worker,ROOT/'src')
sys.path[:0]=[str(src),str(src/'hybrid_power_system_analysis')]
import numpy as np
import psutil
import model.meas_array_model as meas

if args.memory_worker:
    from secore.ac_se import ACStateEstimator
    from secore.dc_se import DCStateEstimator
    from secore.hybrid_se import HybridStateEstimator
    side,_=args.case.split('/')
    cls={'ac':ACStateEstimator,'dc':DCStateEstimator,'hybrid':HybridStateEstimator}[side]
    proc=psutil.Process()
    initial=proc.memory_info().peak_wset
    with contextlib.redirect_stdout(io.StringIO()):
        est=cls(e_file=ROOT/f'data/model/{args.case}.e',meas_file=ROOT/f'data/meas/{args.case}.meas',flat_start=True,max_iter=30)
        if args.mode=='array':
            est.run(result_mode='array',skip_bad_data=True,final_diagnostics=False)
        else:
            est.run(verbose=False)
    final=proc.memory_info().peak_wset
    print(json.dumps(dict(variant=args.memory_worker,case=args.case,mode=args.mode,
          peak_working_set_mib=final/2**20,added_peak_mib=(final-initial)/2**20,converged=bool(est.estimate_result.converged))))
    raise SystemExit

functions={}
for variant,root in ROOTS.items():
    file=root/'hybrid_power_system_analysis/model/meas_array_model.py'
    namespace=dict(meas.__dict__)
    exec(compile(file.read_bytes(),str(file),'exec'),namespace)
    functions[variant]=namespace['_build_meas_ppc_from_measurement_file_array_only']
rows=[]
for case in ('ac/ieee3k','dc/dc_net_3000','hybrid/qinling_1000'):
    samples={v:[] for v in ROOTS}
    for repeat in range(9):
        outputs={}
        for variant in ('baseline','optimized') if repeat%2==0 else ('optimized','baseline'):
            gc.collect()
            start=time.perf_counter()
            result=functions[variant](ROOT/f'data/meas/{case}.meas',include_matrix=False)
            samples[variant].append(time.perf_counter()-start)
            outputs[variant]=result
        for key,value in outputs['baseline'].items():
            if isinstance(value,np.ndarray):
                assert value.dtype==outputs['optimized'][key].dtype
                np.testing.assert_array_equal(value,outputs['optimized'][key])
        assert outputs['baseline']['device_name_id_by_name']==outputs['optimized']['device_name_id_by_name']
        del outputs,result
    row=dict(case=case,samples=samples,exact_equal=True,
        paired_speedup=statistics.median(b/a for b,a in zip(samples['baseline'],samples['optimized'])),
        speedup=statistics.median(samples['baseline'])/statistics.median(samples['optimized']))
    rows.append(row)
    print(json.dumps(row),flush=True)
memory=[]
for case,mode in [('ac/ieee3k','full'),('dc/dc_net_3000','full'),('hybrid/qinling_100','full'),('hybrid/qinling_1000','array')]:
    for variant in ROOTS:
        cmd=[sys.executable,__file__,'--memory-worker',variant,'--case',case,'--mode',mode]
        row=json.loads(subprocess.check_output(cmd,text=True))
        assert row['converged']
        memory.append(row)
        print(json.dumps(row),flush=True)
(OUT/'components.json').write_text(json.dumps(dict(parser=rows,memory=memory,
    memory_note='Each variant in a fresh process; native peak working set includes imports; one sample each'),indent=2),encoding='utf8')
