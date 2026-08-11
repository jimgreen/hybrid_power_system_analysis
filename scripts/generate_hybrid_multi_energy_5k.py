#!/usr/bin/env python3
"""Generate a 5000-node jointly solved electric/fluid multi-energy case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.generate_hybrid_multi_energy_1k import (
    COUPLING_TYPES,
    generate_case as generate_scaled_case,
)


DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_multi_energy_5k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_multi_energy_5k.meas"

NODE_COUNTS = {
    "ac": 1250,
    "dc": 750,
    "heat": 750,
    "gas": 750,
    "hydro": 750,
    "steam": 750,
}
COUPLINGS_PER_TYPE = 20


def generate_case(
    model_path: Path = DEFAULT_CASE,
    measurement_path: Path = DEFAULT_MEASUREMENTS,
    *,
    solve_measurements: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    return generate_scaled_case(
        model_path,
        measurement_path,
        solve_measurements=solve_measurements,
        node_counts=NODE_COUNTS,
        couplings_per_type=COUPLINGS_PER_TYPE,
        measurement_prefix="multi_energy_5k",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args(argv)
    model_path, measurement_path, details = generate_case(
        args.model,
        args.measurements,
        solve_measurements=not args.template_only,
    )
    print(f"model={model_path}")
    print(f"measurements={measurement_path}")
    print(f"nodes={sum(NODE_COUNTS.values())} {NODE_COUNTS}")
    print("converter_counts=ACAC:4 DCDC:6 DCAC/ACDC:4")
    print(f"coupling_types={len(COUPLING_TYPES)}, each={COUPLINGS_PER_TYPE}")
    print(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
