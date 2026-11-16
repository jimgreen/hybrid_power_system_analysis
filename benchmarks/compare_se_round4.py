"""Paired SE benchmark against the round-4 pre-change source snapshot.

Each sample builds a fresh estimator, rereads the input, clears the existing
measurement-file cache, and compares state/residual arrays exactly.  No
network or solved-state cache is used.
"""
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "output/se_optimization_20261116/round4"
BASELINE = ROUND / "baseline_src"
OUT = ROUND / "paired"
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
CASES = [("ac", "ieee3k"), ("dc", "dc_net_3000"),
         ("hybrid", "hybrid_net_400"), ("hybrid", "qinling_100"),
         ("hybrid", "qinling_1000")]
MODULES = ("secore.ac_se", "secore.dc_se", "secore.hybrid_se")


def run_variant(kind, case, variant, source_map, repeat):
    import secore.ac_se as ac
    import secore.dc_se as dc
    import secore.hybrid_se as hybrid
    import model.meas_array_model as meas

    modules = (ac, dc, hybrid)
    for module, name in zip(modules, MODULES):
        exec(source_map[variant][name], module.__dict__)
    classes = {"ac": ac.ACStateEstimator, "dc": dc.DCStateEstimator,
               "hybrid": hybrid.HybridStateEstimator}
    e_file = ROOT / f"data/model/{kind}/{case}.e"
    meas_file = ROOT / f"data/meas/{kind}/{case}.meas"
    meas._MEAS_PPC_FILE_CACHE.clear()
    gc.collect()
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        estimator = classes[kind](e_file=e_file, meas_file=meas_file,
                                  flat_start=True, max_iter=30,
                                  auto_prepare=False)
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
                x=np.asarray(result.x, dtype=np.float64).copy(),
                residual=np.asarray(result.residual, dtype=np.float64).copy())


def main():
    source_map = {"baseline": {}, "optimized": {}}
    for variant, root in (("baseline", BASELINE), ("optimized", ROOT / "src")):
        for name in MODULES:
            path = root / "hybrid_power_system_analysis" / (name.replace("secore.", "secore/") + ".py")
            source_map[variant][name] = compile(path.read_bytes(), str(path), "exec")
    rows = []
    arrays = {}
    repeats = 3
    for kind, case in CASES:
        for repeat in range(repeats):
            order = [("baseline", "baseline"), ("optimized", "optimized")]
            if repeat % 2:
                order.reverse()
            for label, variant in order:
                row = run_variant(kind, case, variant, source_map, repeat)
                arrays[f"{case}_{variant}_{repeat}_x"] = row.pop("x")
                arrays[f"{case}_{variant}_{repeat}_residual"] = row.pop("residual")
                rows.append(row)
                print(json.dumps(row), flush=True)
    summaries = []
    for kind, case in CASES:
        before = [r for r in rows if r["case"] == case and r["variant"] == "baseline"]
        after = [r for r in rows if r["case"] == case and r["variant"] == "optimized"]
        before.sort(key=lambda r: r["repeat"])
        after.sort(key=lambda r: r["repeat"])
        for b, a in zip(before, after):
            assert b["iterations"] == a["iterations"]
            assert b["states"] == a["states"]
            assert b["measurements"] == a["measurements"]
            assert b["objective"] == a["objective"]
            assert b["residual_inf"] == a["residual_inf"]
            np.testing.assert_array_equal(arrays[f"{case}_baseline_{b['repeat']}_x"],
                                          arrays[f"{case}_optimized_{a['repeat']}_x"])
            np.testing.assert_array_equal(arrays[f"{case}_baseline_{b['repeat']}_residual"],
                                          arrays[f"{case}_optimized_{a['repeat']}_residual"])
        total_before = [r["total_s"] for r in before]
        total_after = [r["total_s"] for r in after]
        block_speedups = [b / a for b, a in zip(total_before, total_after)]
        summaries.append(dict(case=case, kind=kind, samples=repeats,
                              baseline={key: statistics.median(r[key] for r in before)
                                        for key in ("prepare_s", "run_s", "total_s")},
                              optimized={key: statistics.median(r[key] for r in after)
                                         for key in ("prepare_s", "run_s", "total_s")},
                              speedup=statistics.median(total_before) / statistics.median(total_after),
                              paired_speedup=statistics.median(block_speedups),
                              ranges={"baseline": [min(total_before), max(total_before)],
                                      "optimized": [min(total_after), max(total_after)]},
                              exact_equal=True))
    payload = dict(baseline=str(BASELINE), optimized=str(ROOT / "src"),
                   repeats=repeats, cache_policy="measurement cache cleared per sample; no network/result cache",
                   cases=summaries, rows=rows)
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    np.savez_compressed(OUT / "comparison.npz", **arrays)


if __name__ == "__main__":
    main()
