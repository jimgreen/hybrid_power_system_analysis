import argparse
import contextlib
import gc
import io
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import build_ac_ppc_from_e_file  # noqa: E402
from ac_lf import ACPowerFlowCalc  # noqa: E402
from ac_se import ACStateEstimator  # noqa: E402
from dc_lf import DCPowerFlowCalc  # noqa: E402
from dc_array_model import DCPowerNetwork  # noqa: E402
from dc_se import DCStateEstimator  # noqa: E402
from hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork  # noqa: E402
from hybrid_se import HybridStateEstimator  # noqa: E402


CASE_MAP = {
    "ieee300": ("ac", ROOT_DIR / "data" / "model" / "ac" / "ieee300.e", ROOT_DIR / "data" / "meas" / "ac" / "ieee300.meas"),
    "ieee3k": ("ac", ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e", ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas"),
    "ieee1w": ("ac", ROOT_DIR / "data" / "model" / "ac" / "ieee1w.e", ROOT_DIR / "data" / "meas" / "ac" / "ieee1w.meas"),
    "ieee3w": ("ac", ROOT_DIR / "data" / "model" / "ac" / "ieee3w.e", ROOT_DIR / "data" / "meas" / "ac" / "ieee3w.meas"),
    "dc_net_3000": (
        "dc",
        ROOT_DIR / "data" / "model" / "dc" / "dc_net_3000.e",
        ROOT_DIR / "data" / "meas" / "dc" / "dc_net_3000.meas",
    ),
    "qingling_100": (
        "hybrid",
        ROOT_DIR / "data" / "model" / "hybrid" / "qingling_100.e",
        ROOT_DIR / "data" / "meas" / "hybrid" / "qingling_100.meas",
    ),
    "qingling_1000": (
        "hybrid",
        ROOT_DIR / "data" / "model" / "hybrid" / "qingling_1000.e",
        ROOT_DIR / "data" / "meas" / "hybrid" / "qingling_1000.meas",
    ),
}


def _silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _time_call(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def _bench_ac_lf(e_file: Path):
    ppc, init_s = _time_call(lambda: build_ac_ppc_from_e_file(e_file, use_cache=True, copy_arrays=False))
    calc = ACPowerFlowCalc.from_ppc(ppc, tol=1e-8, max_iter=50)

    def solve():
        _silent(calc.prepare)
        return _silent(calc.run)

    rc, solve_s = _time_call(solve)
    return {
        "lf_init_s": init_s,
        "lf_solve_s": solve_s,
        "lf_rc": rc,
        "lf_converged": calc.converged,
        "lf_iter": calc.iterations,
        "lf_norm": calc.normF,
        "nodes": int(ppc["bus"].shape[0]),
        "lf_states": int(calc.total_vars),
    }


def _bench_dc_lf(e_file: Path):
    def init():
        network = DCPowerNetwork()
        network.read_from_file(e_file)
        network.topo()
        return network

    network, init_s = _time_call(init)
    calc = DCPowerFlowCalc(network)
    rc, solve_s = _time_call(lambda: _silent(calc.run, tol=1e-8, max_iter=50, verbose=False))
    return {
        "lf_init_s": init_s,
        "lf_solve_s": solve_s,
        "lf_rc": rc,
        "lf_converged": calc.converged,
        "lf_iter": calc.iterations,
        "lf_norm": calc.normF,
        "nodes": len(network.nodes),
        "lf_states": int(getattr(calc, "total_vars", 0)),
    }


def _bench_hybrid_lf(e_file: Path):
    def init():
        network = HybridPowerNetwork.read_from_file(e_file)
        ac_warnings, ac_errors, dc_warnings, dc_errors = _silent(network.prepare, False)
        errors = [*ac_errors, *dc_errors]
        if errors:
            raise RuntimeError(f"{e_file} topology errors: {errors[:5]}")
        return network

    network, init_s = _time_call(init)
    calc = HybridPowerFlowCalc(network, tol=1e-8, max_iter=50, verbose=False)

    def solve():
        _silent(calc.prepare)
        return _silent(calc.run)

    rc, solve_s = _time_call(solve)
    return {
        "lf_init_s": init_s,
        "lf_solve_s": solve_s,
        "lf_rc": rc,
        "lf_converged": calc.converged,
        "lf_iter": calc.iterations,
        "lf_norm": calc.normF,
        "nodes": network.total_nodes,
        "lf_states": int(calc.total_vars),
    }


def _bench_se(kind: str, e_file: Path, meas_file: Path):
    estimator_class = {
        "ac": ACStateEstimator,
        "dc": DCStateEstimator,
        "hybrid": HybridStateEstimator,
    }[kind]

    estimator, init_s = _time_call(lambda: estimator_class(e_file=e_file, meas_file=meas_file, flat_start=True))
    result, solve_s = _time_call(lambda: _silent(estimator.estimate, verbose=False))
    data = {
        "se_init_s": init_s,
        "se_solve_s": solve_s,
        "se_converged": result.converged,
        "se_iter": result.iterations,
        "se_resid": result.residual_inf,
        "se_observable": result.observability.observable,
        "se_states": result.observability.state_count,
        "se_meas": result.observability.measurement_count,
    }
    del result
    del estimator
    gc.collect()
    return data


def bench_case(case_name: str):
    kind, e_file, meas_file = CASE_MAP[case_name]
    if not e_file.exists():
        raise FileNotFoundError(e_file)
    if not meas_file.exists():
        raise FileNotFoundError(meas_file)

    if kind == "ac":
        lf = _bench_ac_lf(e_file)
    elif kind == "dc":
        lf = _bench_dc_lf(e_file)
    else:
        lf = _bench_hybrid_lf(e_file)
    se = _bench_se(kind, e_file, meas_file)
    return {"case": case_name, "kind": kind, **lf, **se}


def _fmt_bool(value):
    return "Y" if value else "N"


def print_table(results):
    headers = [
        "case",
        "kind",
        "nodes",
        "LF init s",
        "LF solve s",
        "LF iter",
        "LF norm",
        "LF ok",
        "SE init s",
        "SE solve s",
        "SE iter",
        "SE resid",
        "SE ok",
        "obs",
        "SE meas",
        "SE states",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["case"],
                result["kind"],
                result["nodes"],
                f'{result["lf_init_s"]:.6f}',
                f'{result["lf_solve_s"]:.6f}',
                result["lf_iter"],
                f'{result["lf_norm"]:.3e}',
                _fmt_bool(result["lf_converged"] and result["lf_rc"] == 0),
                f'{result["se_init_s"]:.6f}',
                f'{result["se_solve_s"]:.6f}',
                result["se_iter"],
                f'{result["se_resid"]:.3e}',
                _fmt_bool(result["se_converged"]),
                _fmt_bool(result["se_observable"]),
                result["se_meas"],
                result["se_states"],
            ]
        )
    widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    print(" | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(item).ljust(widths[idx]) for idx, item in enumerate(row)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark flat-start LF and SE runtime for local E/meas cases.")
    parser.add_argument("cases", nargs="*", default=list(CASE_MAP))
    args = parser.parse_args()

    results = []
    for case_name in args.cases:
        if case_name not in CASE_MAP:
            raise KeyError(f"Unknown case: {case_name}")
        result = bench_case(case_name)
        results.append(result)
        print(
            f"CASE {case_name}: "
            f"LF={result['lf_solve_s']:.6f}s/{result['lf_iter']}it ok={result['lf_converged']} "
            f"SE={result['se_solve_s']:.6f}s/{result['se_iter']}it ok={result['se_converged']} "
            f"obs={result['se_observable']}",
            flush=True,
        )
    print()
    print_table(results)


if __name__ == "__main__":
    main()
