"""Steady-state hydrogen-network load flow (historical module name: hydro)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.fluid_lf import FluidLFResult, FluidPowerFlowCalc, print_fluid_result
from model.hydro_model import load_hydro_network_from_e_file
from paths import model_file


DEFAULT_CASE = model_file("hydro", "hydro_net_3.e")


class HydroLFResult(FluidLFResult):
    pass


class HydroPowerFlowCalc(FluidPowerFlowCalc):
    result_class = HydroLFResult


def print_hydro_result(calc: HydroPowerFlowCalc, rc: int) -> None:
    print_fluid_result(calc, rc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Steady-state hydrogen-network load flow")
    parser.add_argument("case", nargs="?", default=str(DEFAULT_CASE))
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--result-mode", default="full", choices=("full", "summary", "array", "none"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    calc = HydroPowerFlowCalc(
        load_hydro_network_from_e_file(args.case),
        tol=args.tol,
        max_iter=args.max_iter,
        result_mode=args.result_mode,
        verbose=args.verbose,
    )
    rc = calc.run()
    print_hydro_result(calc, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
