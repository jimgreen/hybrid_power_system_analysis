import statistics
import subprocess
import time

commands = {
    "python-pyklu": [
        "python", "-m", "lfcore.ac_lf", "data/model/ac/ieee3w.e",
        "--linear-solver", "pyklu",
        "--result-mode", "none",
        "--quiet",
    ],
    "python-superlu": [
        "python", "-m", "lfcore.ac_lf", "data/model/ac/ieee3w.e",
        "--linear-solver", "superlu",
        "--result-mode", "none",
        "--quiet",
    ],
    "python-umfpack": [
        "python", "-m", "lfcore.ac_lf", "data/model/ac/ieee3w.e",
        "--linear-solver", "umfpack",
        "--result-mode", "none",
        "--quiet",
    ],
    "nuitka-pyklu": [
        "build_nuitka/ac_lf.dist/ac_lf_app.exe", "data/model/ac/ieee3w.e",
        "--linear-solver", "pyklu",
        "--result-mode", "none",
        "--quiet",
    ],
    "nuitka-superlu": [
        "build_nuitka/ac_lf.dist/ac_lf_app.exe", "data/model/ac/ieee3w.e",
        "--linear-solver", "superlu",
        "--result-mode", "none",
        "--quiet",
    ],
    "nuitka-umfpack": [
        "build_nuitka/ac_lf.dist/ac_lf_app.exe", "data/model/ac/ieee3w.e",
        "--linear-solver", "umfpack",
        "--result-mode", "none",
        "--quiet",
    ],
}

for name, cmd in commands.items():
    times = []
    for _ in range(10):
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        times.append(time.perf_counter() - start)
    print(name, "runs=", times, "median=", statistics.median(times))
