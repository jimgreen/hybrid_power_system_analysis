"""District-heating hydraulic and thermal state estimation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.heat_model import load_heat_network_from_e_file
from paths import measurement_file, model_file
from secore.fluid_se import FluidStateEstimator, print_fluid_se_result


DEFAULT_CASE = model_file("heat", "heat_net_3.e")
DEFAULT_MEAS = measurement_file("heat", "heat_net_3.meas")


class HeatStateEstimator(FluidStateEstimator):
    pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="District-heating state estimation")
    parser.add_argument("case", nargs="?", default=str(DEFAULT_CASE))
    parser.add_argument("--meas", default=str(DEFAULT_MEAS))
    parser.add_argument("--flat-start", action="store_true")
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--result-e")
    args = parser.parse_args(argv)
    estimator = HeatStateEstimator(
        load_heat_network_from_e_file(args.case),
        args.meas,
        flat_start=args.flat_start,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )
    rc = estimator.run()
    print_fluid_se_result(estimator, rc)
    if args.result_e:
        estimator.se_result.write_e_file(args.result_e)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
