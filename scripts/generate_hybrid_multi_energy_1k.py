#!/usr/bin/env python3
"""Generate a 1000-node jointly solved electric/fluid multi-energy case."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from model.meas_model import Measurement
from scripts.check_fluid_scale_lf_se import (
    _generate_compressible_case,
    _generate_heat_case,
    build_measurements_from_lf,
    write_measurement_file,
)
from scripts.generate_hybrid_converter_all_modes_1k import (
    _append_block_rows,
    _extended_case_text,
    _measurement_template,
    _row,
)
from scripts.update_meas_from_lf import Snapshot, rewrite_measurements


DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_multi_energy_1k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_multi_energy_1k.meas"

NODE_COUNTS = {
    "ac": 250,
    "dc": 150,
    "heat": 150,
    "gas": 150,
    "hydro": 150,
    "steam": 150,
}
COUPLINGS_PER_TYPE = 5
COUPLING_TYPES = (
    "DcE2Heat",
    "Gas2DcE",
    "DcE2Hydro",
    "Steam2AcE",
    "Gas2AcE",
    "Hydro2AcE",
    "Hydro2DcE",
    "AcE2Heat",
    "AcE2Hydro",
    "Steam2DcE",
    "Gas2Heat",
)


def _block(name: str, header: str, rows: Iterable[str]) -> str:
    return "\n".join((f"<{name}>", "@ " + header, *rows, f"</{name}>", ""))


def _add_coupling_electric_devices(
    text: str,
    couplings_per_type: int,
    *,
    include_heat2: bool = False,
) -> str:
    count = int(couplings_per_type)
    electric_load_groups = 3 if include_heat2 else 2
    ac_load_rows = [
        _row(
            (
                8 + pos,
                f"coupled_ac_load_{8 + pos}",
                40 + pos,
                0.02 if pos < count else 0.001,
                100,
                0,
                0,
                0.0002,
                100,
                0,
                0,
                1,
            )
        )
        for pos in range(electric_load_groups * count)
    ]
    text = _append_block_rows(text, "ACLoad", ac_load_rows)

    ac_generator_rows = [
        _row(
            (
                5 + pos,
                f"coupled_ac_gen_{5 + pos}",
                60 + pos,
                "PQ",
                0.04,
                0.005,
                0.0,
                1.0,
                1,
            )
        )
        for pos in range(3 * count)
    ]
    text = _append_block_rows(text, "ACGenerator", ac_generator_rows)

    dc_load_rows = [
        _row(
            (
                17 + pos,
                f"coupled_dc_load_{17 + pos}",
                70 + pos,
                0.02 if pos < count else 0.001,
                100,
                0,
                0,
                1,
            )
        )
        for pos in range(electric_load_groups * count)
    ]
    text = _append_block_rows(text, "DCLoad", dc_load_rows)

    dc_generator_rows = [
        _row(
            (
                15 + pos,
                f"coupled_dc_gen_{15 + pos}",
                90 + pos,
                "P",
                100.0,
                0.04,
                0.0,
                1,
            )
        )
        for pos in range(3 * count)
    ]
    return _append_block_rows(text, "DCGenerator", dc_generator_rows)


def _add_coupling_fluid_sources(
    text: str,
    node_counts: dict[str, int],
    couplings_per_type: int,
    *,
    include_heat2: bool = False,
) -> str:
    count = int(couplings_per_type)
    heat_node_start = 2 * max(2, int(node_counts["heat"]) // 4) + 6
    heat_source_rows = []

    def add_heat_source(
        idx: int,
        *,
        node: object = "-",
        supply_node: object = "-",
        return_node: object = "-",
        flow: float = 0.0001,
    ) -> None:
        heat_source_rows.append(
            _row(
                (
                    idx,
                    f"coupled_heat_source_{idx}",
                    node,
                    supply_node,
                    return_node,
                    "FLOW",
                    0.0,
                    flow,
                    1.0,
                    0.0,
                    0.01,
                    85.0,
                    1,
                )
            )
        )

    if include_heat2:
        supply_count = max(2, int(node_counts["heat"]) // 4)
        explicit_supply_start = supply_count // 2 + 1
        explicit_return_start = supply_count + explicit_supply_start
        explicit_leaf_count = supply_count - supply_count // 2
        explicit_source_count = 2 * count
        # Span the explicit supply/return tree instead of creating a symmetric
        # cluster of temperature controls on one local branch.
        for group in range(5):
            for pos in range(count):
                idx = 2 + group * count + pos
                if group in {1, 3}:
                    source_pos = (group // 2) * count + pos
                    explicit_pos = source_pos * explicit_leaf_count // explicit_source_count
                    add_heat_source(
                        idx,
                        supply_node=explicit_supply_start + explicit_pos,
                        return_node=explicit_return_start + explicit_pos,
                    )
                else:
                    implicit_group = group // 2
                    add_heat_source(
                        idx,
                        node=heat_node_start + implicit_group * count + pos,
                        flow=0.00012 if group == 4 else 0.0001,
                    )
    else:
        for pos in range(3 * count):
            idx = pos + 2
            add_heat_source(
                idx,
                node=heat_node_start + pos,
                flow=0.00012 if pos >= 2 * count else 0.0001,
            )
    text = _append_block_rows(text, "HeatSource", heat_source_rows)

    hydro_node_start = int(node_counts["hydro"]) // 2 + 25
    hydro_source_rows = [
        _row(
            (
                idx,
                f"coupled_hydro_source_{idx}",
                hydro_node_start + pos,
                "FLOW",
                0.0,
                0.0001,
                1.0,
                0.0,
                0.01,
                1,
            )
        )
        for pos, idx in enumerate(range(2, 2 + 2 * count))
    ]
    return _append_block_rows(text, "HydroSource", hydro_source_rows)


def _scale_coupled_fluid_loads(text: str, couplings_per_type: int) -> str:
    count = int(couplings_per_type)
    limits = {"GasLoad": 3 * count, "HydroLoad": 2 * count, "SteamLoad": 2 * count}
    active_block = ""
    output = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("<") and line.endswith(">") and not line.startswith("</"):
            active_block = line[1:-1]
        elif line.startswith("</"):
            active_block = ""
        if active_block in limits and line.startswith("#"):
            fields = raw_line.split()
            if len(fields) > 4 and int(fields[1]) <= limits[active_block]:
                fields[4] = "0.00013333333333333334"
                raw_line = " ".join(fields)
        output.append(raw_line)
    return "\n".join(output)


def _coupling_blocks(
    couplings_per_type: int,
    *,
    include_heat2: bool = False,
) -> str:
    count = int(couplings_per_type)

    def indices(start: int, group: int) -> range:
        first = int(start) + int(group) * count
        return range(first, first + count)

    dc_hydro_group = 2 if include_heat2 else 1
    ac_hydro_group = 2 if include_heat2 else 1
    ac_heat_group = 2 if include_heat2 else 1
    gas_heat_group = 4 if include_heat2 else 2
    specifications = [
        ("DcE2Heat", "idx_dc_load_t1 idx_heat_unit_t2", indices(17, 0), indices(2, 0), 0.95, 950.0),
    ]
    if include_heat2:
        specifications.append(
            (
                "DcE2Heat2",
                "idx_dc_load_t1 idx_heat_unit_t2",
                indices(17, 1),
                indices(2, 1),
                0.95,
                950.0,
            )
        )
    specifications.extend(
        (
        ("Gas2DcE", "idx_gas_load_t1 idx_dc_unit_t2", indices(1, 0), indices(15, 0), 0.40, 750.0),
        ("DcE2Hydro", "idx_dc_load_t1 idx_h2_unit_t2", indices(17, dc_hydro_group), indices(2, 0), 0.75, 750.0),
        ("Steam2AcE", "idx_steam_load_t1 idx_ac_unit_t2", indices(1, 0), indices(5, 0), 0.35, 857.1428571428571),
        ("Gas2AcE", "idx_gas_load_t1 idx_ac_unit_t2", indices(1, 1), indices(5, 1), 0.40, 750.0),
        ("Hydro2AcE", "idx_h2_load_t1 idx_ac_unit_t2", indices(1, 0), indices(5, 2), 0.55, 545.4545454545455),
        ("Hydro2DcE", "idx_h2_load_t1 idx_dc_unit_t2", indices(1, 1), indices(15, 1), 0.55, 545.4545454545455),
        ("AcE2Heat", "idx_ac_load_t1 idx_heat_unit_t2", indices(8, 0), indices(2, ac_heat_group), 0.94, 940.0),
        )
    )
    if include_heat2:
        specifications.append(
            (
                "AcE2Heat2",
                "idx_ac_load_t1 idx_heat_unit_t2",
                indices(8, 1),
                indices(2, 3),
                0.94,
                940.0,
            )
        )
    specifications.extend(
        (
        ("AcE2Hydro", "idx_ac_load_t1 idx_h2_unit_t2", indices(8, ac_hydro_group), indices(2, 1), 0.72, 720.0),
        ("Steam2DcE", "idx_steam_load_t1 idx_dc_unit_t2", indices(1, 1), indices(15, 2), 0.35, 857.1428571428571),
        ("Gas2Heat", "idx_gas_load_t1 idx_heat_unit_t2", indices(1, 2), indices(2, gas_heat_group), 0.90, 750.0),
        )
    )
    blocks = []
    for table_name, endpoint_header, t1_indices, t2_indices, efficiency, factor in specifications:
        if table_name.startswith(("AcE2Heat", "DcE2Heat")):
            coefficient_field = "e2h_coeff"
            default_control = "P"
        elif table_name in {"AcE2Hydro", "DcE2Hydro"}:
            coefficient_field = "e2h_coeff"
            default_control = "FLOW"
        elif table_name in {"Hydro2AcE", "Hydro2DcE"}:
            coefficient_field = "h2e_coeff"
            default_control = "P"
        elif table_name in {"Gas2AcE", "Gas2DcE"}:
            coefficient_field = "g2e_coeff"
            default_control = "P"
            efficiency = 300.0
        elif table_name in {"Steam2AcE", "Steam2DcE"}:
            coefficient_field = "s2e_coeff"
            default_control = "P"
            efficiency = 0.35
        elif table_name == "Gas2Heat":
            coefficient_field = "g2h_coeff"
            default_control = "FLOW"
            efficiency = 86.64
        else:
            coefficient_field = ""
            default_control = ""
        if coefficient_field:
            alternate_control = (
                "T_OUT"
                if table_name.startswith(("AcE2Heat", "DcE2Heat")) or table_name == "Gas2Heat"
                else "P"
                if default_control == "FLOW"
                else "FLOW"
            )
            rows = [
                _row(
                    (
                        pos,
                        f"{table_name.lower()}_{pos}",
                        1,
                        default_control if pos % 2 else alternate_control,
                        t1_idx,
                        t2_idx,
                        efficiency,
                    )
                )
                for pos, (t1_idx, t2_idx) in enumerate(
                    zip(t1_indices, t2_indices),
                    start=1,
                )
            ]
            header = (
                f"idx name run_stat control_type {endpoint_header} {coefficient_field}"
            )
        else:
            rows = [
                _row(
                    (
                        pos,
                        f"{table_name.lower()}_{pos}",
                        1,
                        t1_idx,
                        t2_idx,
                        efficiency,
                        factor,
                    )
                )
                for pos, (t1_idx, t2_idx) in enumerate(
                    zip(t1_indices, t2_indices),
                    start=1,
                )
            ]
            header = f"idx name run_stat {endpoint_header} efficiency energy_factor"
        blocks.append(
            _block(
                table_name,
                header,
                rows,
            )
        )
    return "\n".join(blocks)


def _validated_node_counts(node_counts: dict[str, int]) -> dict[str, int]:
    required = {"ac", "dc", "heat", "gas", "hydro", "steam"}
    counts = {name: int(value) for name, value in node_counts.items()}
    if set(counts) != required:
        raise ValueError(f"node_counts must contain exactly {sorted(required)}")
    if any(value < 10 for value in counts.values()):
        raise ValueError("every energy domain must contain at least 10 nodes")
    return counts


def _validate_case_capacity(
    node_counts: dict[str, int],
    couplings_per_type: int,
    *,
    include_heat2: bool = False,
) -> None:
    count = int(couplings_per_type)
    if count < 1:
        raise ValueError("couplings_per_type must be positive")
    electric_load_groups = 3 if include_heat2 else 2
    minimums = {
        "ac": max(39 + electric_load_groups * count, 59 + 3 * count),
        "dc": max(69 + electric_load_groups * count, 89 + 3 * count),
        "heat": 2 * max(2, node_counts["heat"] // 4) + 5 + 3 * count,
        "gas": 6 * count,
        "hydro": max(4 * count, node_counts["hydro"] // 2 + 24 + 2 * count),
        "steam": 4 * count,
    }
    insufficient = {
        name: (node_counts[name], minimum)
        for name, minimum in minimums.items()
        if node_counts[name] < minimum
    }
    if insufficient:
        raise ValueError(f"node counts are too small for coupling endpoints: {insufficient}")
    if include_heat2:
        supply_count = max(2, node_counts["heat"] // 4)
        explicit_leaf_count = supply_count - supply_count // 2
        if explicit_leaf_count < 2 * count:
            raise ValueError(
                "insufficient explicit heat supply/return leaf pairs for Heat2 sources: "
                f"need {2 * count}, got {explicit_leaf_count}"
            )


def build_case_text(
    node_counts: dict[str, int] | None = None,
    couplings_per_type: int = COUPLINGS_PER_TYPE,
    *,
    include_heat2: bool = False,
) -> str:
    counts = _validated_node_counts(NODE_COUNTS if node_counts is None else node_counts)
    _validate_case_capacity(
        counts,
        couplings_per_type,
        include_heat2=include_heat2,
    )
    electric = _extended_case_text(counts["ac"], counts["dc"])
    electric = _add_coupling_electric_devices(
        electric,
        couplings_per_type,
        include_heat2=include_heat2,
    )
    fluid = (
        _generate_heat_case(counts["heat"]),
        _generate_compressible_case("gas", counts["gas"]),
        _generate_compressible_case("hydro", counts["hydro"]),
        _generate_compressible_case("steam", counts["steam"]),
    )
    model = "\n\n".join((electric, *fluid))
    model = _scale_coupled_fluid_loads(model, couplings_per_type)
    model = _add_coupling_fluid_sources(
        model,
        counts,
        couplings_per_type,
        include_heat2=include_heat2,
    )
    return "\n\n".join(
        (
            model,
            _coupling_blocks(
                couplings_per_type,
                include_heat2=include_heat2,
            ),
        )
    ).rstrip() + "\n"


def _populate_measurements(
    model_path: Path,
    measurement_path: Path,
    measurement_prefix: str,
) -> dict[str, object]:
    measurement_path.write_text(_measurement_template(model_path), encoding="utf-8", newline="\n")
    network = _read_lf_network_from_file(model_path)
    calc = HybridPowerFlowCalc(
        network,
        tol=1e-8,
        max_iter=100,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(
            f"Hybrid LF failed: rc={rc}, iterations={calc.iterations}, residual={calc.normF:.6e}"
        )

    snapshot = Snapshot(
        network,
        ac_grid=network.ac,
        dc_grid=network.dc,
        dcac_converters=network.dcac_converters,
        acac_converters=network.acac_converters,
    )
    updated, missing = rewrite_measurements(measurement_path, snapshot)
    if missing:
        raise RuntimeError(f"failed to populate {missing} electric measurement rows")

    measurements = list(Measurement.read_from_file(measurement_path))
    for fluid_calc in calc.fluid_calcs.values():
        measurements.extend(build_measurements_from_lf(fluid_calc.network, fluid_calc))
    for idx, measurement in enumerate(measurements, start=1):
        measurement.idx = idx
        measurement.name = f"{measurement_prefix}_m{idx}"
    write_measurement_file(measurements, measurement_path)
    EBook(measurement_path)

    coupling_counts = Counter(item.table_name for item in calc.multi_energy.couplings)
    return {
        "lf_iterations": int(calc.iterations),
        "lf_residual": float(calc.normF),
        "lf_variables": int(calc.total_vars),
        "lf_equations": int(calc.total_eq),
        "jacobian_nnz": int(calc.get_jacobi(calc.x).nnz),
        "measurements": len(measurements),
        "electric_measurements_updated": int(updated),
        "couplings": dict(sorted(coupling_counts.items())),
    }


def generate_case(
    model_path: Path = DEFAULT_CASE,
    measurement_path: Path = DEFAULT_MEASUREMENTS,
    *,
    solve_measurements: bool = True,
    node_counts: dict[str, int] | None = None,
    couplings_per_type: int = COUPLINGS_PER_TYPE,
    measurement_prefix: str = "multi_energy_1k",
    include_heat2: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    model_path = Path(model_path)
    measurement_path = Path(measurement_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        build_case_text(
            node_counts,
            couplings_per_type,
            include_heat2=include_heat2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    EBook(model_path)

    if solve_measurements:
        details = _populate_measurements(model_path, measurement_path, measurement_prefix)
    else:
        measurement_path.write_text(_measurement_template(model_path), encoding="utf-8", newline="\n")
        details = {"measurements": "template-only"}
    return model_path, measurement_path, details


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
    print(f"converter_counts=ACAC:4 DCDC:6 DCAC/ACDC:4")
    print(f"coupling_types={len(COUPLING_TYPES)}, each={COUPLINGS_PER_TYPE}")
    print(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
