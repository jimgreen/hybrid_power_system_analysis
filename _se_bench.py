import sys, os, time, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "src", "hybrid_power_system_analysis")
for p in [ROOT,
          os.path.join(ROOT, "model"),
          os.path.join(ROOT, "lfcore"),
          os.path.join(ROOT, "secore"),
          os.path.join(HERE, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.sparse.linalg import use_solver
use_solver(useUmfpack=False)


def bench_ac(case, repeats=3):
    from secore.ac_se import ACStateEstimator
    from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE
    from paths import measurement_file, model_file
    e = model_file("ac", f"{case}.e")
    m = measurement_file("ac", f"{case}.meas")
    times = []
    info = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        est = ACStateEstimator(
            e_file=e, meas_file=m, parameter_file=DEFAULT_SE_PARAMETER_FILE,
            flat_start=True, profile=False, auto_prepare=False,
        )
        est.prepare()
        est.run(result_mode="none", skip_bad_data=True, verbose=False, final_diagnostics=False)
        r = est.estimate_result
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        info = (r.converged, r.iterations, float(r.residual_inf), bool(r.observability.observable))
    return min(times), sum(times) / len(times), info


def bench_dc(case, repeats=3):
    from secore.dc_se import DCStateEstimator
    from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE
    from paths import measurement_file, model_file
    e = model_file("dc", f"{case}.e")
    m = measurement_file("dc", f"{case}.meas")
    times = []
    info = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            est = DCStateEstimator(
                e_file=e, meas_file=m, parameter_file=DEFAULT_SE_PARAMETER_FILE,
                flat_start=True, profile=False, auto_prepare=False,
            )
            est.prepare()
            with contextlib.redirect_stdout(io.StringIO()):
                est.run(result_mode="none", skip_bad_data=True, verbose=False)
            r = est.estimate_result
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            info = (r.converged, r.iterations, float(getattr(r, 'residual_inf', 0.0)), True)
        except Exception as e:
            print(f"DC {case} FAILED: {type(e).__name__}: {e}")
            return None, None, None
    return min(times), sum(times) / len(times), info


mode = sys.argv[1] if len(sys.argv) > 1 else "ac"
cases = sys.argv[2].split(",") if len(sys.argv) > 2 else (
    ["ieee300", "ieee3k", "ieee1w", "ieee3w"] if mode == "ac"
    else ["dc_net_1000", "dc_net_3000", "dc_net_1w", "dc_net_3w"]
)

print(f"=== {mode.upper()} SE benchmark ===")
print(f"{'case':<14} {'min':>10}  {'avg':>10}  iter  conv  obs  norm")
print("-" * 75)
for case in cases:
    try:
        result = bench_ac(case) if mode == "ac" else bench_dc(case)
        if result[0] is None:
            continue
        mn, avg, (conv, iters, norm, obs) = result
        flag = ' ' if conv else '!'
        print(f"{case:<14} {mn*1000:8.2f}ms  {avg*1000:8.2f}ms  {iters:>4}   {flag}{conv!s:<5} {obs!s:<5} {norm:.1e}")
    except Exception as e:
        print(f"{case}: ERROR - {type(e).__name__}: {e}")
