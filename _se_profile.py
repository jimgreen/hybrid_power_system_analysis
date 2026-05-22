import sys, os, cProfile, pstats, io, contextlib
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

mode = sys.argv[1] if len(sys.argv) > 1 else "ac"
case = sys.argv[2] if len(sys.argv) > 2 else ("ieee3w" if mode == "ac" else "dc_net_3w")
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

if mode == "ac":
    from secore.ac_se import ACStateEstimator as Estimator
else:
    from secore.dc_se import DCStateEstimator as Estimator
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE
from paths import measurement_file, model_file

e_file = model_file(mode, f"{case}.e")
m_file = measurement_file(mode, f"{case}.meas")


def solve_once():
    est = Estimator(e_file=e_file, meas_file=m_file, parameter_file=DEFAULT_SE_PARAMETER_FILE,
                    flat_start=True, profile=False, auto_prepare=False)
    est.prepare()
    with contextlib.redirect_stdout(io.StringIO()):
        if mode == "ac":
            est.run(result_mode="none", skip_bad_data=True, verbose=False, final_diagnostics=False)
        else:
            est.run(result_mode="none", skip_bad_data=True, verbose=False)
    return est.estimate_result


# warmup
for _ in range(2):
    solve_once()

pr = cProfile.Profile()
pr.enable()
for _ in range(repeats):
    r = solve_once()
pr.disable()

print(f"=== {mode.upper()} SE {case}: iter={r.iterations} conv={r.converged} ({repeats} repeats) ===")
buf = io.StringIO()
stats = pstats.Stats(pr, stream=buf)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(40)
stats.sort_stats("tottime")
stats.print_stats(25)
print(buf.getvalue())
