"""LF benchmark with per-phase timing and result snapshot."""

import argparse
import contextlib
import gc
import io
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_ROOT = SRC_DIR / "hybrid_power_system_analysis"
MODEL_DIR = PKG_ROOT / "model"
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from hybrid_power_system_analysis.model.ac_array_model import build_ac_ppc_from_e_file  # noqa: E402
from hybrid_power_system_analysis.lfcore.ac_lf import ACPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc  # noqa: E402
from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork  # noqa: E402
from hybrid_power_system_analysis.lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork  # noqa: E402


CASE_MAP = {
    "ieee300":       ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee300.e"),
    "ieee3k":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3k.e"),
    "ieee1w":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee1w.e"),
    "ieee3w":        ("ac", PROJECT_ROOT / "data" / "model" / "ac" / "ieee3w.e"),
    "dc_net_1000":   ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_1000.e"),
    "dc_net_3000":   ("dc", PROJECT_ROOT / "data" / "model" / "dc" / "dc_net_3000.e"),
    "qinling_100":   ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_100.e"),
    "qinling_1000":  ("hybrid", PROJECT_ROOT / "data" / "model" / "hybrid" / "qinling_1000.e"),
}


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _time_call(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def _wrap(calc, attr_name, state, kind):
    attr = getattr(calc, attr_name, None)
    if attr is None or getattr(attr, "_lf_wrapped", False):
        return
    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        try:
            return attr(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            if kind == "jac":
                state["jac_total_s"] += dt
                state["jac_count"] += 1
            else:
                state["solve_total_s"] += dt
                state["solve_count"] += 1
    wrapped._lf_wrapped = True
    setattr(calc, attr_name, wrapped)


def _install_ac_profile(calc: ACPowerFlowCalc):
    state = {"jac_total_s": 0.0, "jac_count": 0,
             "solve_total_s": 0.0, "solve_count": 0}
    for n in ("_get_jacobi_from_precomputed_pattern",):
        _wrap(calc, n, state, "jac")
    return state


def _install_dc_profile(calc: DCPowerFlowCalc):
    state = {"jac_total_s": 0.0, "jac_count": 0,
             "solve_total_s": 0.0, "solve_count": 0}
    for n in ("_get_jacobi_from_precomputed_pattern", "_get_jacobi_from_terms"):
        _wrap(calc, n, state, "jac")
    return state


def _install_hybrid_profile(calc: HybridPowerFlowCalc):
    state = {"jac_total_s": 0.0, "jac_count": 0,
             "solve_total_s": 0.0, "solve_count": 0}
    for n in ("_assemble_jacobian", "_assemble_jacobian_from_precomputed_pattern",
              "_build_newton_system"):
        _wrap(calc, n, state, "jac")
    return state


def _patch_factor_jacobian():
    """Module-level patch: wrap solver_common.factor_jacobian to count + time calls.

    factor_jacobian is imported as `_factor_jacobian` in lfcore/*.py, so we patch
    the source module so all callers see the wrapped version.
    """
    from hybrid_power_system_analysis.lfcore import solver_common
    if getattr(solver_common.factor_jacobian, "_lf_wrapped", False):
        return solver_common._bench_state
    orig = solver_common.factor_jacobian
    state = solver_common._bench_state = {
        "solve_total_s": 0.0, "solve_count": 0}

    def wrapped(matrix, resolved_name, solver_fn):
        t0 = time.perf_counter()
        try:
            return orig(matrix, resolved_name, solver_fn)
        finally:
            state["solve_total_s"] += time.perf_counter() - t0
            state["solve_count"] += 1
    wrapped._lf_wrapped = True
    solver_common.factor_jacobian = wrapped
    return state


def _get_solve_state():
    from hybrid_power_system_analysis.lfcore import solver_common
    return getattr(solver_common, "_bench_state", {"solve_total_s": 0.0, "solve_count": 0})


def _snapshot_ac(calc: ACPowerFlowCalc):
    ppc = calc.ppc
    bus = ppc.get("bus")
    if bus is not None and bus.shape[1] > 7:
        V = bus[:, 7]
        return {"V_head": V[:8].tolist(), "V_norm": float(np.linalg.norm(V))}
    return {"V_head": [], "V_norm": 0.0}


def _snapshot_dc(calc: DCPowerFlowCalc):
    ppc = getattr(calc, "ppc", None) or getattr(calc, "ac_ppc", None)
    if ppc is None or not isinstance(ppc, dict):
        return {"V_head": [], "V_norm": 0.0}
    bus = ppc.get("bus")
    if bus is None:
        return {"V_head": [], "V_norm": 0.0}
    return {"V_head": bus[:8, 0].tolist(), "V_norm": float(np.linalg.norm(bus[:, 0]))}


def _snapshot_hybrid(calc: HybridPowerFlowCalc):
    V_ac = None
    if calc.ac_calc is not None:
        ac_ppc = getattr(calc.ac_calc, "ppc", None)
        if isinstance(ac_ppc, dict):
            bus = ac_ppc.get("bus")
            if bus is not None and bus.shape[1] > 7:
                V_ac = bus[:, 7]
    return {
        "V_head": V_ac[:8].tolist() if V_ac is not None else [],
        "V_norm": float(np.linalg.norm(V_ac)) if V_ac is not None else 0.0,
    }


def _bench_ac_lf(e_file: Path) -> Dict:
    ppc, init_s = _time_call(lambda: build_ac_ppc_from_e_file(e_file))
    calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
    solve_state = _patch_factor_jacobian()
    profile = _install_ac_profile(calc)
    rc, solve_s = _time_call(lambda: _silent(calc.run))
    snap = _snapshot_ac(calc)
    nodes = int(ppc["bus"].shape[0])
    return {
        "case": e_file.stem, "kind": "ac", "nodes": nodes,
        "lf_init_s": init_s, "lf_solve_s": solve_s,
        "jac_total_s": profile["jac_total_s"], "jac_count": profile["jac_count"],
        "solve_total_s": solve_state["solve_total_s"], "solve_count": solve_state["solve_count"],
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), **snap,
    }


def _bench_dc_lf(e_file: Path) -> Dict:
    def init():
        network = DCPowerNetwork()
        network.read_from_file(e_file)
        network.topo()
        return network
    network, init_s = _time_call(init)
    solve_state = _patch_factor_jacobian()
    calc = DCPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)
    profile = _install_dc_profile(calc)
    rc, solve_s = _time_call(lambda: _silent(calc.run))
    snap = _snapshot_dc(calc)
    return {
        "case": e_file.stem, "kind": "dc", "nodes": len(network.nodes),
        "lf_init_s": init_s, "lf_solve_s": solve_s,
        "jac_total_s": profile["jac_total_s"], "jac_count": profile["jac_count"],
        "solve_total_s": solve_state["solve_total_s"], "solve_count": solve_state["solve_count"],
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), **snap,
    }


def _bench_hybrid_lf(e_file: Path) -> Dict:
    def init():
        network = HybridPowerNetwork.read_from_file(e_file)
        ac_warnings, ac_errors, dc_warnings, dc_errors = _silent(network.prepare, False)
        errors = [*ac_errors, *dc_errors]
        if errors:
            raise RuntimeError(f"{e_file} topology errors: {errors[:5]}")
        return network
    network, init_s = _time_call(init)
    solve_state = _patch_factor_jacobian()
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)
    profile = _install_hybrid_profile(calc)

    def solve():
        _silent(calc.prepare)
        return _silent(calc.run)

    rc, solve_s = _time_call(solve)
    snap = _snapshot_hybrid(calc)
    return {
        "case": e_file.stem, "kind": "hybrid", "nodes": network.total_nodes,
        "lf_init_s": init_s, "lf_solve_s": solve_s,
        "jac_total_s": profile["jac_total_s"], "jac_count": profile["jac_count"],
        "solve_total_s": solve_state["solve_total_s"], "solve_count": solve_state["solve_count"],
        "rc": rc, "converged": calc.converged, "iter": calc.iterations,
        "normF": float(calc.normF), **snap,
    }


def bench_case(case_name: str) -> Dict:
    kind, e_file = CASE_MAP[case_name]
    if not e_file.exists():
        raise FileNotFoundError(e_file)
    try:
        if kind == "ac":
            return _bench_ac_lf(e_file)
        if kind == "dc":
            return _bench_dc_lf(e_file)
        return _bench_hybrid_lf(e_file)
    except Exception as e:
        import traceback
        return {"case": case_name, "kind": kind, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().splitlines()[-5:]}


def print_table(results: List[Dict]):
    headers = ["case", "kind", "nodes", "init_s", "solve_s", "iter",
               "jac_s", "jac#", "solve_s2", "solve#", "normF", "ok"]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["case"], r.get("kind", "?"), "-",
                        "-", "-", "-", "-", "-", "-", "-", "-", f"ERR:{r['error'][:20]}"])
            continue
        rows.append([
            r["case"], r["kind"], r.get("nodes", "-"),
            f"{r.get('lf_init_s',0):.4f}",
            f"{r.get('lf_solve_s',0):.4f}",
            r.get("iter", "-"),
            f"{r.get('jac_total_s',0):.4f}",
            r.get("jac_count", 0),
            f"{r.get('solve_total_s',0):.4f}",
            r.get("solve_count", 0),
            f"{r.get('normF',float('nan')):.2e}",
            "Y" if (r.get("converged") and r.get("rc") == 0) else "N",
        ])
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description="LF benchmark with per-phase profile.")
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=str, default=None, help="Write JSON results to path.")
    args = parser.parse_args()

    all_results = []
    for case_name in args.cases:
        if case_name not in CASE_MAP:
            print(f"!! Unknown case: {case_name}")
            continue
        runs = []
        for _ in range(args.repeat):
            try:
                r = bench_case(case_name)
            except Exception as e:
                r = {"case": case_name, "error": f"{type(e).__name__}: {e}"}
            runs.append(r)
            gc.collect()
        def _agg(key):
            vals = [run[key] for run in runs if key in run and isinstance(run[key], (int, float))]
            if not vals:
                return None
            vals.sort()
            return vals[len(vals) // 2]

        def _agg_min(key):
            vals = [run[key] for run in runs if key in run and isinstance(run[key], (int, float))]
            if not vals:
                return None
            return min(vals)

        agg = {"case": case_name, "kind": runs[0].get("kind"),
               "nodes": runs[0].get("nodes"), "runs": len(runs)}
        for k in ("lf_init_s", "lf_solve_s", "jac_total_s", "jac_count",
                  "solve_total_s", "solve_count", "iter", "normF"):
            v = _agg_min(k) if k.endswith("_s") else _agg(k)
            if v is not None:
                agg[k] = v
        for run in runs[::-1]:
            if "error" in run:
                agg["error"] = run["error"]
                agg["traceback"] = run.get("traceback", [])
                break
            if "V_head" in run:
                agg["V_head"] = run["V_head"]
                agg["V_norm"] = run["V_norm"]
                agg["converged"] = run.get("converged")
                agg["rc"] = run.get("rc")
                break
        all_results.append(agg)
        if "error" in agg:
            print(f"[{case_name}] ERROR: {agg['error']}", flush=True)
            for ln in agg.get("traceback", []):
                print(f"   {ln}")
            continue
        print(
            f"[{case_name}] median over {len(runs)} runs: "
            f"init={agg.get('lf_init_s',0):.4f}s solve={agg.get('lf_solve_s',0):.4f}s "
            f"jac_total={agg.get('jac_total_s',0):.4f}s({agg.get('jac_count',0)}x) "
            f"solve_total={agg.get('solve_total_s',0):.4f}s({agg.get('solve_count',0)}x) "
            f"iter={agg.get('iter','?')} norm={agg.get('normF',float('nan')):.2e} "
            f"ok={agg.get('converged',False)}",
            flush=True,
        )

    print()
    print_table(all_results)
    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
