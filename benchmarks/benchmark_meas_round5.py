import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import psutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/hybrid_power_system_analysis')]
import model.meas_array_model as m
OUT = ROOT/'output/se_optimization_20261116/round5'
PATHS = {'baseline': OUT/'baseline_src/meas_array_model.py',
         'optimized': ROOT/'src/hybrid_power_system_analysis/model/meas_array_model.py'}
CASES = ['ac/ieee3k', 'dc/dc_net_3000', 'hybrid/qinling_1000']

def load(variant):
    namespace = dict(m.__dict__)
    path = PATHS[variant]
    exec(compile(path.read_bytes(),str(path),'exec'),namespace)
    return namespace['_build_meas_ppc_from_measurement_file_array_only']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',choices=list(PATHS))
    ap.add_argument('--case')
    args=ap.parse_args()
    if args.worker:
        f=load(args.worker)
        proc=psutil.Process()
        initial=proc.memory_info().peak_wset
        r=f(ROOT/'data/meas'/f'{args.case}.meas',include_matrix=False)
        peak=proc.memory_info().peak_wset
        print(json.dumps(dict(peak_working_set_mb=peak/2**20, added_peak_mb=(peak-initial)/2**20, rows=len(r['idx_array']))))
        return
    functions={v:load(v) for v in PATHS}
    rows=[]
    for case in CASES:
        timings={v:[] for v in PATHS}
        arrays={}
        for repeat in range(5):
            for variant in (list(PATHS) if repeat%2==0 else list(PATHS)[::-1]):
                gc.collect()
                start=time.perf_counter()
                r=functions[variant](ROOT/'data/meas'/f'{case}.meas',include_matrix=False)
                elapsed=time.perf_counter()-start
                timings[variant].append(elapsed)
                arrays[variant]={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
                del r
            for key,value in arrays['baseline'].items():
                np.testing.assert_array_equal(value,arrays['optimized'][key])
        memory={v:json.loads(subprocess.check_output([sys.executable,__file__,'--worker',v,'--case',case],text=True)) for v in PATHS}
        row=dict(case=case,timings=timings,memory=memory,exact_equal=True,
                 speedup=statistics.median(timings['baseline'])/statistics.median(timings['optimized']))
        rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'parser_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
