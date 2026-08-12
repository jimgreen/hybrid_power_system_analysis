#!/usr/bin/env python3
"""Generate a 5000-node multi-energy case with dense device-mode coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from scripts.generate_hybrid_converter_all_modes_1k import _replace_block, _row
from scripts.generate_hybrid_multi_energy_1k import (
    _block,
    _measurement_template,
    _populate_measurements,
    build_case_text,
)


DEFAULT_CASE = (
    ROOT / "data" / "model" / "hybrid" / "hybrid_multi_energy_rich_5k.e"
)
DEFAULT_MEASUREMENTS = (
    ROOT / "data" / "meas" / "hybrid" / "hybrid_multi_energy_rich_5k.meas"
)

NODE_COUNTS = {
    "ac": 1250,
    "dc": 750,
    "heat": 750,
    "gas": 750,
    "hydro": 750,
    "steam": 750,
}
COUPLINGS_PER_TYPE = 20
COUPLING_TYPES = (
    "DcE2Heat",
    "DcE2Heat2",
    "Gas2DcE",
    "DcE2Hydro",
    "Steam2AcE",
    "Gas2AcE",
    "Hydro2AcE",
    "Hydro2DcE",
    "AcE2Heat",
    "AcE2Heat2",
    "AcE2Hydro",
    "Steam2DcE",
    "Gas2Heat",
)
STORAGE_COUNTS = {
    "HeatStorage": 60,
    "GasStorage": 20,
    "HydroStorage": 20,
    "SteamStorage": 20,
}

def _normalize_dc_voltage_controls(text: str) -> str:
    """Keep every DC voltage controller consistent with the flat start."""
    match = re.search(r"<DCGenerator>.*?</DCGenerator>", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("missing E block: DCGenerator")
    lines = match.group(0).splitlines()
    header = lines[1].removeprefix("@").split()
    control_col = header.index("control_type") + 1
    voltage_col = header.index("v_set") + 1
    for pos in range(2, len(lines) - 1):
        fields = lines[pos].split()
        if fields and fields[0] == "#" and fields[control_col] == "V":
            fields[voltage_col] = "100.0"
            lines[pos] = " ".join(fields)
    replacement = "\n".join(lines)
    return text[: match.start()] + replacement + text[match.end() :]


def _converter_blocks(text: str) -> str:
    acac_modes = (("PQ", "PQ"), ("PV", "PQ"), ("PQ", "PV"), ("PV", "PV"))
    acac_rows = []
    for pos in range(40):
        i_control, j_control = acac_modes[pos % len(acac_modes)]
        i_node = 101 + 2 * pos
        j_node = i_node + 1
        acac_rows.append(
            _row(
                (
                    pos + 1,
                    f"rich_acac_{pos + 1}",
                    i_node,
                    j_node,
                    0.002,
                    0.002,
                    i_control,
                    j_control,
                    0.0,
                    0.0,
                    0.0,
                    1.0 if i_control == "PV" else 0.0,
                    1.0 if j_control == "PV" else 0.0,
                    1,
                )
            )
        )
    text = _replace_block(
        text,
        "ACACConverter",
        (
            "idx name i_node j_node r1 r2 i_control_type j_control_type "
            "p_set i_q_set j_q_set i_v_set j_v_set run_stat"
        ),
        acac_rows,
    )

    dcdc_modes = (
        ("P", "NONE"),
        ("NONE", "P"),
        ("V", "NONE"),
        ("NONE", "V"),
        ("I", "NONE"),
        ("NONE", "I"),
    )
    dcdc_rows = []
    for pos in range(42):
        i_control, j_control = dcdc_modes[pos % len(dcdc_modes)]
        # The two terminals use different nodes on repeated, electrically
        # equivalent feeders.  Their uncoupled voltages are identical, so all
        # six controls have a feasible zero-transfer reference solution.
        i_node = 401 + pos
        j_node = i_node + 120
        dcdc_rows.append(
            _row(
                (
                    pos + 1,
                    f"rich_dcdc_{pos + 1}",
                    i_node,
                    j_node,
                    0.0,
                    0.0,
                    i_control,
                    j_control,
                    0.0,
                    0.0,
                    100.0 if "V" in (i_control, j_control) else 0.0,
                    1,
                )
            )
        )
    text = _replace_block(
        text,
        "DCDCConverter",
        (
            "idx name i_node j_node r1 r2 i_control_type j_control_type "
            "p_set i_set v_set run_stat"
        ),
        dcdc_rows,
    )

    dcac_modes = (("PQ", "V"), ("PH", "NONE"), ("PQ", "NONE"), ("NONE", "P"))
    dcac_rows = []
    for pos in range(80):
        ac_control, dc_control = dcac_modes[pos % len(dcac_modes)]
        device_type = "DCACConverter" if pos < 40 else "ACDCConverter"
        dcac_rows.append(
            _row(
                (
                    pos + 1,
                    f"rich_{device_type.lower()}_{pos + 1}",
                    301 + pos,
                    301 + pos,
                    0.002,
                    0.002,
                    ac_control,
                    dc_control,
                    0.0,
                    0.0,
                    0.0,
                    1.0 if ac_control == "PH" else 0.0,
                    100.0 if dc_control == "V" else 0.0,
                    1,
                    device_type,
                )
            )
        )
    return _replace_block(
        text,
        "DCACConverter",
        (
            "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type "
            "p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat dev_type"
        ),
        dcac_rows,
    )


def _alternating_flow(pos: int, magnitude: float = 1.0e-4) -> float:
    return magnitude if pos % 2 else -magnitude


def _storage_blocks() -> str:
    heat_rows = []
    for pos in range(1, STORAGE_COUNTS["HeatStorage"] + 1):
        heat_rows.append(
            _row(
                (
                    pos,
                    f"rich_heat_storage_{pos}",
                    500 + pos,
                    "FLOW",
                    0.0,
                    _alternating_flow(pos),
                    1.0,
                    -0.01,
                    0.01,
                    85.0,
                    65.0,
                    1,
                )
            )
        )

    def compressible_rows(prefix: str, count: int, *, steam: bool = False) -> list[str]:
        rows = []
        for pos in range(1, count + 1):
            values: tuple[object, ...] = (
                pos,
                f"rich_{prefix.lower()}_storage_{pos}",
                500 + pos,
                "FLOW",
                0.0,
                _alternating_flow(pos),
                1.0,
                -0.01,
                0.01,
            )
            if steam:
                values += (3000.0,)
            rows.append(_row((*values, 1)))
        return rows

    return "\n".join(
        (
            _block(
                "HeatStorage",
                (
                    "idx name node control_type pressure_set flow_set alpha flow_min "
                    "flow_max supply_temperature_set return_temperature_set run_stat"
                ),
                heat_rows,
            ),
            _block(
                "GasStorage",
                "idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat",
                compressible_rows("gas", STORAGE_COUNTS["GasStorage"]),
            ),
            _block(
                "HydroStorage",
                "idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat",
                compressible_rows("hydro", STORAGE_COUNTS["HydroStorage"]),
            ),
            _block(
                "SteamStorage",
                (
                    "idx name node control_type pressure_set flow_set alpha flow_min "
                    "flow_max enthalpy_set run_stat"
                ),
                compressible_rows(
                    "steam",
                    STORAGE_COUNTS["SteamStorage"],
                    steam=True,
                ),
            ),
        )
    )


def build_rich_case_text() -> str:
    text = build_case_text(
        NODE_COUNTS,
        COUPLINGS_PER_TYPE,
        include_heat2=True,
    )
    text = _normalize_dc_voltage_controls(_converter_blocks(text))
    return "\n\n".join((text.rstrip(), _storage_blocks().rstrip())) + "\n"


def _structure_details(model_path: Path) -> dict[str, object]:
    book = EBook(model_path)
    network = _read_lf_network_from_file(model_path)
    calc = HybridPowerFlowCalc(
        network,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )
    calc.prepare()
    return {
        "nodes": {
            domain: len(book.data[block].data)
            for domain, block in (
                ("ac", "ACNode"),
                ("dc", "DCNode"),
                ("heat", "HeatNode"),
                ("gas", "GasNode"),
                ("hydro", "HydroNode"),
                ("steam", "SteamNode"),
            )
        },
        "converters": {
            "ACACConverter": len(book.data["ACACConverter"].data),
            "DCDCConverter": len(book.data["DCDCConverter"].data),
            "DCACConverter": len(book.data["DCACConverter"].data),
        },
        "dcac_device_types": dict(
            sorted(
                Counter(
                    str(row["dev_type"])
                    for row in book.data["DCACConverter"].data
                ).items()
            )
        ),
        "couplings": {
            table_name: len(book.data[table_name].data)
            for table_name in COUPLING_TYPES
        },
        "storages": {
            table_name: len(book.data[table_name].data)
            for table_name in STORAGE_COUNTS
        },
        "lf_variables": int(calc.total_vars),
        "lf_equations": int(calc.total_eq),
        "active_couplings": len(calc.energy_coupling_plans),
        "warnings": list(calc.multi_energy.warnings),
    }


def generate_case(
    model_path: Path = DEFAULT_CASE,
    measurement_path: Path = DEFAULT_MEASUREMENTS,
    *,
    solve_measurements: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    model_path = Path(model_path)
    measurement_path = Path(measurement_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(build_rich_case_text(), encoding="utf-8", newline="\n")
    EBook(model_path)

    if solve_measurements:
        details = _populate_measurements(
            model_path,
            measurement_path,
            "multi_energy_rich_5k",
        )
    else:
        measurement_path.write_text(
            _measurement_template(model_path),
            encoding="utf-8",
            newline="\n",
        )
        EBook(measurement_path)
        details = {"measurements": "template-only"}
    details["structure"] = _structure_details(model_path)
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
    print(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
