#!/usr/bin/env python3
"""Generate a 5000-node meshed hybrid case with all supported element classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from scripts.generate_hybrid_converter_all_modes_1k import (
    _append_block_rows,
    _measurement_template,
    _replace_block,
    _row,
)
from scripts.update_meas_from_lf import rewrite_measurements, solve_hybrid


BASE_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_all_elements_5k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_all_elements_5k.meas"

AC_NODE_COUNT = 2500
DC_NODE_COUNT = 2500
AC_FEEDERS = 40
DC_FEEDERS = 50


def _insert_block_before(text: str, marker: str, block_name: str, header: str, rows: Sequence[str]) -> str:
    if marker not in text:
        raise ValueError(f"missing insertion marker: {marker}")
    block = "\n".join((f"<{block_name}>", "@ " + header, *rows, f"</{block_name}>", ""))
    return text.replace(marker, block + "\n" + marker, 1)


def _ac_extension(text: str, node_count: int) -> str:
    nodes = [
        _row((idx, f"ac5k_{idx}", 1.0, 1.0, 0.0, 0, 1))
        for idx in range(11, node_count + 1)
    ]
    text = _append_block_rows(text, "ACNode", nodes)

    branches = []
    branch_idx = 15
    for node in range(11, node_count + 1):
        offset = node - 11
        feeder = offset % AC_FEEDERS
        level = offset // AC_FEEDERS
        parent = 1 + feeder % 10 if level == 0 else node - AC_FEEDERS
        r = 0.0018 + 0.00015 * (feeder % 5)
        x = 0.0070 + 0.0004 * (feeder % 7)
        b = 0.0004 + 0.00005 * (level % 3)
        branches.append(_row((branch_idx, f"ac5k_chain_{parent}_{node}", parent, node, r, x, b, 1)))
        branch_idx += 1

    max_level = (node_count - 11) // AC_FEEDERS
    for level in range(7, max_level + 1, 8):
        level_start = 11 + level * AC_FEEDERS
        for feeder in range(0, AC_FEEDERS - 1, 2):
            i_node = level_start + feeder
            j_node = i_node + 1
            if j_node > node_count:
                continue
            branches.append(
                _row((branch_idx, f"ac5k_tie_{i_node}_{j_node}", i_node, j_node, 0.0035, 0.014, 0.0002, 1))
            )
            branch_idx += 1
    text = _append_block_rows(text, "ACBranch", branches)

    loads = []
    load_idx = 8
    for node in range(60, node_count + 1, 5):
        loads.append(
            _row((load_idx, f"ac5k_load_{node}", node, 0.001, 55, 25, 20, 0.0005, 50, 30, 20, 1))
        )
        load_idx += 1
    text = _append_block_rows(text, "ACLoad", loads)

    text = _append_block_rows(
        text,
        "ACGenerator",
        (
            _row((5, "ac5k_gen_p", 27, "P", 0.20, 0.05, 0.0, 1.0, 1)),
            _row((6, "ac5k_gen_ph", 1, "PH", 0.0, 0.0, 1.04, 1.5, 1)),
        ),
    )
    text = _append_block_rows(
        text,
        "ACShuntCompensator",
        (_row((4, "ac5k_shunt_b", 28, "B", 0.0, 0.0, -0.002, 0.0, 1)),),
    )
    text = _append_block_rows(
        text,
        "ACZeroBranch",
        (_row((2, "ac5k_zero_2410_2411", 2410, 2411, 1)),),
    )
    text = _append_block_rows(
        text,
        "ACSwitch",
        (
            _row((1, "ac5k_disconnector_closed", 2420, 2421, 1, 1)),
            _row((2, "ac5k_disconnector_open", 2422, 2423, 0, 1)),
            _row((3, "ac5k_switch_closed", 2424, 2425, 1, 1)),
            _row((4, "ac5k_switch_open", 2426, 2427, 0, 1)),
        ),
    )
    text = _append_block_rows(
        text,
        "ACBreak",
        (
            _row((2, "ac5k_breaker_closed", 2430, 2431, 1, 1)),
            _row((3, "ac5k_breaker_open", 2432, 2433, 0, 1)),
        ),
    )
    text = _append_block_rows(
        text,
        "ACTransformer",
        (_row((2, "ac5k_transformer_2440_2441", 2440, 2441, 0.002, 0.025, 0.0001, -0.0005, 1.01, 0.2, 1)),),
    )
    return _insert_block_before(
        text,
        "<DCNode>",
        "ACThreeWindingTransformer",
        (
            "idx name i_node j_node k_node i_r i_x j_r j_x k_r k_x gt bt "
            "i_tap i_shift j_tap j_shift k_tap k_shift run_stat"
        ),
        (
            _row(
                (
                    1,
                    "ac5k_three_winding_2450_2451_2452",
                    2450,
                    2451,
                    2452,
                    0.003,
                    0.035,
                    0.0035,
                    0.040,
                    0.004,
                    0.045,
                    0.0001,
                    -0.0005,
                    1.01,
                    0.2,
                    1.0,
                    -0.1,
                    0.99,
                    0.0,
                    1,
                )
            ),
        ),
    )


def _dc_extension(text: str, node_count: int) -> str:
    nodes = [
        _row((idx, f"dc5k_{idx}", 100.0, 100.0, 0, 1))
        for idx in range(31, node_count + 1)
    ]
    text = _append_block_rows(text, "DCNode", nodes)

    branches = []
    branch_idx = 38
    for node in range(31, node_count + 1):
        offset = node - 31
        feeder = offset % DC_FEEDERS
        level = offset // DC_FEEDERS
        parent = 1 + feeder % 30 if level == 0 else node - DC_FEEDERS
        r = 0.0030 + 0.00025 * (feeder % 6)
        # Keep the DCAC terminals electrically distinct from the stiff base-grid
        # voltage controls so the coupled flat-start Newton step stays physical.
        if 43 <= node <= 46:
            r = 0.05
        branches.append(_row((branch_idx, f"dc5k_chain_{parent}_{node}", parent, node, r, 1)))
        branch_idx += 1

    max_level = (node_count - 31) // DC_FEEDERS
    for level in range(9, max_level + 1, 10):
        level_start = 31 + level * DC_FEEDERS
        for feeder in range(0, DC_FEEDERS - 1, 2):
            i_node = level_start + feeder
            j_node = i_node + 1
            if j_node > node_count:
                continue
            branches.append(_row((branch_idx, f"dc5k_tie_{i_node}_{j_node}", i_node, j_node, 0.006, 1)))
            branch_idx += 1
    text = _append_block_rows(text, "DCBranch", branches)

    loads = []
    load_idx = 17
    for node in range(60, node_count + 1, 5):
        loads.append(_row((load_idx, f"dc5k_load_{node}", node, 0.0008, 60, 25, 15, 1)))
        load_idx += 1
    text = _append_block_rows(text, "DCLoad", loads)

    text = _append_block_rows(
        text,
        "DCZeroBranch",
        (_row((9, "dc5k_zero_2410_2411", 2410, 2411, 1)),),
    )
    text = _append_block_rows(
        text,
        "DCSwitch",
        (
            _row((5, "dc5k_disconnector_closed", 2420, 2421, 1, 1)),
            _row((6, "dc5k_disconnector_open", 2422, 2423, 0, 1)),
            _row((7, "dc5k_switch_closed", 2424, 2425, 1, 1)),
            _row((8, "dc5k_switch_open", 2426, 2427, 0, 1)),
        ),
    )
    return _append_block_rows(
        text,
        "DCBreak",
        (
            _row((6, "dc5k_breaker_closed", 2430, 2431, 1, 1)),
            _row((7, "dc5k_breaker_open", 2432, 2433, 0, 1)),
        ),
    )


def _converter_blocks(text: str) -> str:
    acac_rows = (
        _row((1, "ac5k_acac_pq_pq", 11, 12, 0.002, 0.002, "PQ", "PQ", 0.4, 0.0, 0.0, 0.0, 0.0, 1)),
        _row((2, "ac5k_acac_pv_pq", 13, 14, 0.002, 0.002, "PV", "PQ", 0.4, 0.0, 0.0, 1.01, 0.0, 1)),
        _row((3, "ac5k_acac_pq_pv", 15, 16, 0.002, 0.002, "PQ", "PV", 0.4, 0.0, 0.0, 0.0, 1.0, 1)),
        _row((4, "ac5k_acac_pv_pv", 17, 18, 0.002, 0.002, "PV", "PV", 0.4, 0.0, 0.0, 1.0, 1.0, 1)),
    )
    text = _replace_block(
        text,
        "ACACConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
        acac_rows,
    )

    dcdc_rows = (
        _row((1, "dc5k_dcdc_i_p", 31, 32, 0.001, 0.001, "P", "NONE", 0.4, 0.0, 0.0, 1)),
        _row((2, "dc5k_dcdc_j_p", 33, 34, 0.001, 0.001, "NONE", "P", 0.4, 0.0, 0.0, 1)),
        _row((3, "dc5k_dcdc_i_v", 35, 36, 0.001, 0.001, "V", "NONE", 0.0, 0.0, 160.0, 1)),
        _row((4, "dc5k_dcdc_j_v", 37, 38, 0.001, 0.001, "NONE", "V", 0.0, 0.0, 157.5, 1)),
        _row((5, "dc5k_dcdc_i_i", 39, 40, 0.001, 0.001, "I", "NONE", 0.0, 0.00002, 0.0, 1)),
        _row((6, "dc5k_dcdc_j_i", 41, 42, 0.001, 0.001, "NONE", "I", 0.0, 0.00002, 0.0, 1)),
    )
    text = _replace_block(
        text,
        "DCDCConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat",
        dcdc_rows,
    )

    dcac_rows = (
        _row((1, "dc5k_dcac_pq_v", 19, 43, 0.002, 0.002, "PQ", "V", 0.0, 0.0, 0.0, 0.0, 159.2, 1, "DCACConverter")),
        _row((2, "dc5k_dcac_ph_none", 20, 44, 0.002, 0.002, "PH", "NONE", 0.0, 0.0, 0.0, 1.0, 0.0, 1, "DCACConverter")),
        _row((3, "dc5k_acdc_pq_none", 21, 45, 0.002, 0.002, "PQ", "NONE", 0.4, 0.0, 0.0, 0.0, 0.0, 1, "ACDCConverter")),
        _row((4, "dc5k_dcac_none_p", 22, 46, 0.002, 0.002, "NONE", "P", 0.0, 0.4, 0.0, 0.0, 0.0, 1, "DCACConverter")),
    )
    return _replace_block(
        text,
        "DCACConverter",
        "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat dev_type",
        dcac_rows,
    )


def build_case_text(ac_node_count: int, dc_node_count: int) -> str:
    if ac_node_count != AC_NODE_COUNT or dc_node_count != DC_NODE_COUNT:
        raise ValueError("this benchmark is fixed at 2500 AC plus 2500 DC nodes")
    text = BASE_CASE.read_text(encoding="utf-8")
    text = _ac_extension(text, ac_node_count)
    text = _dc_extension(text, dc_node_count)
    return _converter_blocks(text)


def generate_case(
    model_path: Path = DEFAULT_CASE,
    measurement_path: Path = DEFAULT_MEASUREMENTS,
    *,
    solve_measurements: bool = True,
) -> tuple[Path, Path, str]:
    model_path = Path(model_path)
    measurement_path = Path(measurement_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(build_case_text(AC_NODE_COUNT, DC_NODE_COUNT), encoding="utf-8")
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
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args()
    model, measurements, detail = generate_case(
        args.model,
        args.measurements,
        solve_measurements=not args.template_only,
    )
    print(f"model={model}")
    print(f"measurements={measurements}")
    print(f"nodes={AC_NODE_COUNT + DC_NODE_COUNT} (AC={AC_NODE_COUNT}, DC={DC_NODE_COUNT})")
    print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
