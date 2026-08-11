#!/usr/bin/env python3
"""Generate a 1040-node hybrid benchmark covering every supported converter mode.

The benchmark deliberately keeps converter terminals on distinct topology buses:

* ACAC: PQ/PQ, PV/PQ, PQ/PV, PV/PV
* DCAC: PQ/V, PH/NONE, PQ/NONE, NONE/P
* DCDC: P, V, I control on either the i or j terminal

The DCAC enums also contain AC=PV and DC=I, but those values are not members of
``DCAC_SUPPORTED_CONTROL_PAIRS`` and therefore are not valid solver modes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from scripts.update_meas_from_lf import rewrite_measurements, solve_hybrid


BASE_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_all_modes_1k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_converter_all_modes_1k.meas"

AC_NODE_COUNT = 520
DC_NODE_COUNT = 520


def _row(values: Iterable[object]) -> str:
    return "# " + " ".join(str(value) for value in values)


def _append_block_rows(text: str, block_name: str, rows: Sequence[str]) -> str:
    if not rows:
        return text
    closing = f"</{block_name}>"
    if closing not in text:
        raise ValueError(f"missing E block: {block_name}")
    return text.replace(closing, "\n".join(rows) + "\n" + closing, 1)


def _replace_block(text: str, block_name: str, header: str, rows: Sequence[str]) -> str:
    pattern = re.compile(
        rf"<{re.escape(block_name)}>.*?</{re.escape(block_name)}>",
        flags=re.DOTALL,
    )
    replacement = "\n".join(
        (
            f"<{block_name}>",
            "@ " + header,
            *rows,
            f"</{block_name}>",
        )
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"missing or repeated E block: {block_name}")
    return updated


def _extended_case_text(ac_node_count: int, dc_node_count: int) -> str:
    if ac_node_count < 22:
        raise ValueError("AC node count must be at least 22")
    if dc_node_count < 46:
        raise ValueError("DC node count must be at least 46")
    text = BASE_CASE.read_text(encoding="utf-8")

    ac_nodes = [
        _row((idx, f"acx_{idx}", 1.0, 1.0, 0.0, 0, 1))
        for idx in range(11, ac_node_count + 1)
    ]
    text = _append_block_rows(text, "ACNode", ac_nodes)

    ac_branches = []
    branch_idx = 15
    for node_idx in range(11, ac_node_count + 1):
        parent = 1 + ((node_idx - 11) % 10)
        ac_branches.append(
            _row((branch_idx, f"acx_line_{parent}_{node_idx}", parent, node_idx, 0.02, 0.08, 0.0, 1))
        )
        branch_idx += 1
    text = _append_block_rows(text, "ACBranch", ac_branches)

    dc_nodes = [
        _row((idx, f"dcx_{idx}", 100.0, 100.0, 0, 1))
        for idx in range(31, dc_node_count + 1)
    ]
    text = _append_block_rows(text, "DCNode", dc_nodes)

    dc_branches = []
    branch_idx = 38
    for node_idx in range(31, dc_node_count + 1):
        parent = 1 + ((node_idx - 31) % 30)
        dc_branches.append(
            _row((branch_idx, f"dcx_line_{parent}_{node_idx}", parent, node_idx, 0.05, 1))
        )
        branch_idx += 1
    text = _append_block_rows(text, "DCBranch", dc_branches)

    acac_rows = (
        _row((1, "acac_pq_pq", 11, 12, 0.002, 0.002, "PQ", "PQ", 1.0, 0.0, 0.0, 0.0, 0.0, 1)),
        _row((2, "acac_pv_pq", 13, 14, 0.002, 0.002, "PV", "PQ", 1.0, 0.0, 0.0, 1.0, 0.0, 1)),
        _row((3, "acac_pq_pv", 15, 16, 0.002, 0.002, "PQ", "PV", 1.0, 0.0, 0.0, 0.0, 1.0, 1)),
        _row((4, "acac_pv_pv", 17, 18, 0.002, 0.002, "PV", "PV", 1.0, 0.0, 0.0, 1.0, 1.0, 1)),
    )
    text = _replace_block(
        text,
        "ACACConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
        acac_rows,
    )

    dcdc_rows = (
        _row((1, "dcdc_i_p", 31, 32, 0.001, 0.001, "P", "NONE", 1.0, 0.0, 0.0, 1)),
        _row((2, "dcdc_j_p", 33, 34, 0.001, 0.001, "NONE", "P", 1.0, 0.0, 0.0, 1)),
        _row((3, "dcdc_i_v", 35, 36, 0.001, 0.001, "V", "NONE", 0.0, 0.0, 100.0, 1)),
        _row((4, "dcdc_j_v", 37, 38, 0.001, 0.001, "NONE", "V", 0.0, 0.0, 100.0, 1)),
        _row((5, "dcdc_i_i", 39, 40, 0.001, 0.001, "I", "NONE", 0.0, 0.0001, 0.0, 1)),
        _row((6, "dcdc_j_i", 41, 42, 0.001, 0.001, "NONE", "I", 0.0, 0.0001, 0.0, 1)),
    )
    text = _replace_block(
        text,
        "DCDCConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat",
        dcdc_rows,
    )

    dcac_rows = (
        # DC node 43 sits near 1.592 pu in the uncoupled network.  Keep the
        # PQ/V target near that feasible operating point so the mode-coverage
        # case tests the control equation instead of an impossible transfer.
        _row((1, "dcac_pq_v", 19, 43, 0.002, 0.002, "PQ", "V", 0.0, 0.0, 0.0, 0.0, 159.2, 1, "DCACConverter")),
        _row((2, "dcac_ph_none", 20, 44, 0.002, 0.002, "PH", "NONE", 0.0, 0.0, 0.0, 1.0, 0.0, 1, "DCACConverter")),
        _row((3, "acdc_pq_none", 21, 45, 0.002, 0.002, "PQ", "NONE", 1.0, 0.0, 0.0, 0.0, 0.0, 1, "ACDCConverter")),
        _row((4, "dcac_none_p", 22, 46, 0.002, 0.002, "NONE", "P", 0.0, 1.0, 0.0, 0.0, 0.0, 1, "DCACConverter")),
    )
    text = _replace_block(
        text,
        "DCACConverter",
        "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat dev_type",
        dcac_rows,
    )
    return text


MEASUREMENT_TYPES = {
    "ACNode": ("V",),
    "ACBranch": ("P_FROM", "Q_FROM", "V_FROM", "P_TO", "Q_TO", "V_TO"),
    "ACTransformer": ("P_FROM", "Q_FROM", "V_FROM", "P_TO", "Q_TO", "V_TO"),
    "ACThreeWindingTransformer": (
        "P_FROM", "Q_FROM", "V_FROM", "I_FROM",
        "P_TO", "Q_TO", "V_TO", "I_TO",
        "P_THIRD", "Q_THIRD", "V_THIRD", "I_THIRD",
    ),
    "ACLoad": ("P_LOAD", "Q_LOAD", "V_LOAD"),
    "ACGenerator": ("P_GEN", "Q_GEN", "V_GEN"),
    "ACZeroBranch": ("P_FROM", "Q_FROM", "V_FROM", "I_FROM", "P_TO", "Q_TO", "V_TO", "I_TO"),
    "ACBreak": ("P_FROM", "Q_FROM", "V_FROM", "I_FROM", "P_TO", "Q_TO", "V_TO", "I_TO"),
    "ACACConverter": ("P_FROM", "Q_FROM", "V_FROM", "I_FROM", "P_TO", "Q_TO", "V_TO", "I_TO"),
    "DCNode": ("V",),
    "DCBranch": ("P_FROM", "V_FROM", "P_TO", "V_TO"),
    "DCLoad": ("P_LOAD", "V_LOAD"),
    "DCGenerator": ("P_GEN", "V_GEN"),
    "DCZeroBranch": ("P_FROM", "V_FROM", "I_FROM", "P_TO", "V_TO", "I_TO"),
    "DCBreak": ("P_FROM", "V_FROM", "I_FROM", "P_TO", "V_TO", "I_TO"),
    "DCDCConverter": ("P_FROM", "V_FROM", "I_FROM", "P_TO", "V_TO", "I_TO"),
    "DCACConverter": ("P_DC", "P_AC", "Q_AC", "V_DC", "I_DC", "V_AC", "I_AC"),
}


def _measurement_template(model_path: Path) -> str:
    book = EBook(model_path)
    lines = [
        "<Measurement>",
        "@ idx name dev_type dev_name meas_type weight valid value",
    ]
    idx = 0
    for device_type, measurement_types in MEASUREMENT_TYPES.items():
        block = book.data.get(device_type)
        if block is None:
            continue
        for device in block.data:
            device_name = str(device.get("name", ""))
            if not device_name:
                continue
            for meas_type in measurement_types:
                meas_name = f"m_{idx}_{device_type}_{device_name}_{meas_type}".lower()
                lines.append(_row((idx, meas_name, device_type, device_name, meas_type, 1.0, 1, 0.0)))
                idx += 1
    lines.append("</Measurement>")
    return "\n".join(lines) + "\n"


def generate_case(
    model_path: Path = DEFAULT_CASE,
    measurement_path: Path = DEFAULT_MEASUREMENTS,
    *,
    ac_node_count: int = AC_NODE_COUNT,
    dc_node_count: int = DC_NODE_COUNT,
    solve_measurements: bool = True,
) -> tuple[Path, Path, str]:
    model_path = Path(model_path)
    measurement_path = Path(measurement_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(_extended_case_text(ac_node_count, dc_node_count), encoding="utf-8")
    EBook(model_path)
    measurement_path.write_text(_measurement_template(model_path), encoding="utf-8")
    EBook(measurement_path)

    detail = "measurements not solved"
    if solve_measurements:
        snapshot, detail = solve_hybrid(model_path)
        updated, missing = rewrite_measurements(measurement_path, snapshot)
        if missing:
            raise RuntimeError(f"failed to populate {missing} measurement rows")
        detail = f"updated={updated}, missing={missing}, {detail}"
        EBook(measurement_path)
    return model_path, measurement_path, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--ac-nodes", type=int, default=AC_NODE_COUNT)
    parser.add_argument("--dc-nodes", type=int, default=DC_NODE_COUNT)
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args()
    model_path, measurement_path, detail = generate_case(
        args.model,
        args.measurements,
        ac_node_count=args.ac_nodes,
        dc_node_count=args.dc_nodes,
        solve_measurements=not args.template_only,
    )
    print(f"model={model_path}")
    print(f"measurements={measurement_path}")
    print(f"nodes={args.ac_nodes + args.dc_nodes} (AC={args.ac_nodes}, DC={args.dc_nodes})")
    print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
