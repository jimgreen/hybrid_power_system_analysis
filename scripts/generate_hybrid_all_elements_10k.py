#!/usr/bin/env python3
"""Generate a 10000-node meshed hybrid benchmark with dense device coverage."""

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
from scripts.generate_hybrid_all_elements_5k import _insert_block_before
from scripts.generate_hybrid_converter_all_modes_1k import (
    _append_block_rows,
    _measurement_template,
    _replace_block,
    _row,
)
from scripts.update_meas_from_lf import rewrite_measurements, solve_hybrid


BASE_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_all_elements_10k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_all_elements_10k.meas"

AC_NODE_COUNT = 5000
DC_NODE_COUNT = 5000
ZONE_COUNT = 50
ZONE_COLUMNS = 10


def _partition_nodes(first_node: int, last_node: int, zone_count: int = ZONE_COUNT) -> list[list[int]]:
    nodes = list(range(first_node, last_node + 1))
    zone_size, extra = divmod(len(nodes), zone_count)
    zones = []
    start = 0
    for zone_idx in range(zone_count):
        size = zone_size + (1 if zone_idx < extra else 0)
        zones.append(nodes[start : start + size])
        start += size
    return zones


def _zone_node(zones: Sequence[Sequence[int]], zone_idx: int, offset: int) -> int:
    zone = zones[zone_idx % len(zones)]
    if offset >= len(zone):
        raise ValueError(f"zone {zone_idx} has no node at offset {offset}")
    return int(zone[offset])


def _append_ac_grid(text: str, zones: Sequence[Sequence[int]]) -> str:
    nodes = [
        _row((node, f"ac10k_{node}", 1.0, 1.0, 0.0, 0, 1))
        for zone in zones
        for node in zone
    ]
    text = _append_block_rows(text, "ACNode", nodes)

    branches = []
    branch_idx = 15
    zone_roots = []
    for zone_idx, zone in enumerate(zones):
        root = int(zone[0])
        zone_roots.append(root)
        base_parent = 1 + zone_idx % 10
        branches.append(
            _row((branch_idx, f"ac10k_root_{base_parent}_{root}", base_parent, root, 0.020, 0.080, 0.0, 1))
        )
        branch_idx += 1
        for local_idx in range(1, len(zone)):
            node = int(zone[local_idx])
            row, column = divmod(local_idx, ZONE_COLUMNS)
            parent = int(zone[local_idx - 1] if column else zone[local_idx - ZONE_COLUMNS])
            r = 0.010 + 0.001 * ((zone_idx + column) % 4)
            x = 0.040 + 0.004 * ((zone_idx + row) % 5)
            b = 0.0002 * ((row + column) % 3)
            branches.append(_row((branch_idx, f"ac10k_grid_{parent}_{node}", parent, node, r, x, b, 1)))
            branch_idx += 1
            if row > 0 and column in {3, 7}:
                vertical_parent = int(zone[local_idx - ZONE_COLUMNS])
                branches.append(
                    _row(
                        (
                            branch_idx,
                            f"ac10k_mesh_{vertical_parent}_{node}",
                            vertical_parent,
                            node,
                            0.016,
                            0.060,
                            0.0001,
                            1,
                        )
                    )
                )
                branch_idx += 1
    for zone_idx in range(4, len(zone_roots), 4):
        i_node = zone_roots[zone_idx - 1]
        j_node = zone_roots[zone_idx]
        branches.append(
            _row((branch_idx, f"ac10k_zone_tie_{i_node}_{j_node}", i_node, j_node, 0.030, 0.120, 0.0, 1))
        )
        branch_idx += 1
    text = _append_block_rows(text, "ACBranch", branches)

    loads = []
    load_idx = 8
    for zone_idx, zone in enumerate(zones):
        for local_idx in range(12 + zone_idx % 3, len(zone), 6):
            node = int(zone[local_idx])
            pbase = 0.008 + 0.001 * (zone_idx % 5)
            qbase = 0.004 + 0.0005 * (local_idx % 5)
            loads.append(
                _row((load_idx, f"ac10k_load_{node}", node, pbase, 55, 25, 20, qbase, 50, 30, 20, 1))
            )
            load_idx += 1
    text = _append_block_rows(text, "ACLoad", loads)

    generator_rows = []
    generator_idx = 5
    for offset in range(10):
        node = _zone_node(zones, offset, 45)
        generator_rows.append(_row((generator_idx, f"ac10k_gen_pq_{node}", node, "PQ", 0.08, 0.02, 0.0, 1.0, 1)))
        generator_idx += 1
    for offset in range(10, 20):
        node = _zone_node(zones, offset, 46)
        generator_rows.append(_row((generator_idx, f"ac10k_gen_p_{node}", node, "P", 0.08, 0.0, 0.0, 1.0, 1)))
        generator_idx += 1
    ph_node = _zone_node(zones, 20, 2)
    generator_rows.append(_row((generator_idx, "ac10k_gen_ph", ph_node, "PH", 0.0, 0.0, 1.05, 1.0, 1)))
    text = _append_block_rows(text, "ACGenerator", generator_rows)

    shunt_rows = []
    shunt_idx = 4
    for offset in range(21):
        node = _zone_node(zones, offset, 55)
        mode = ("Q", "Z", "B")[offset % 3]
        q_set = 0.15 if mode == "Q" else 0.0
        b_set = -0.006 if mode == "Z" else (0.004 if mode == "B" else 0.0)
        shunt_rows.append(
            _row((shunt_idx, f"ac10k_shunt_{mode.lower()}_{node}", node, mode, q_set, 0.0, b_set, 0.0, 1))
        )
        shunt_idx += 1
    text = _append_block_rows(text, "ACShuntCompensator", shunt_rows)

    text = _append_block_rows(
        text,
        "ACZeroBranch",
        tuple(
            _row((idx + 2, f"ac10k_zero_{idx}", _zone_node(zones, idx, 90), _zone_node(zones, idx, 98), 1))
            for idx in range(23)
        ),
    )
    text = _append_block_rows(
        text,
        "ACSwitch",
        tuple(
            _row(
                (
                    idx + 1,
                    f"ac10k_switch_{idx}",
                    _zone_node(zones, idx, 88),
                    _zone_node(zones, idx, 97),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(24)
        ),
    )
    text = _append_block_rows(
        text,
        "ACBreak",
        tuple(
            _row(
                (
                    idx + 2,
                    f"ac10k_break_{idx}",
                    _zone_node(zones, idx + 24, 86),
                    _zone_node(zones, idx + 24, 96),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(23)
        ),
    )
    text = _append_block_rows(
        text,
        "ACTransformer",
        tuple(
            _row(
                (
                    idx + 2,
                    f"ac10k_transformer_{idx}",
                    _zone_node(zones, idx, 80 + idx % 8),
                    _zone_node(zones, idx + 17, 82 + idx % 8),
                    0.012 + 0.001 * (idx % 4),
                    0.090 + 0.005 * (idx % 5),
                    0.0,
                    -0.0002,
                    0.998 + 0.002 * (idx % 3),
                    0.002 * ((idx % 5) - 2),
                    1,
                )
            )
            for idx in range(23)
        ),
    )

    three_winding_rows = tuple(
        _row(
            (
                idx + 1,
                f"ac10k_three_winding_{idx}",
                _zone_node(zones, idx, 70 + idx % 8),
                _zone_node(zones, idx + 13, 72 + idx % 8),
                _zone_node(zones, idx + 31, 74 + idx % 8),
                0.015,
                0.110 + 0.004 * (idx % 4),
                0.016,
                0.115 + 0.004 * (idx % 4),
                0.017,
                0.120 + 0.004 * (idx % 4),
                0.0,
                -0.0002,
                0.999,
                0.0,
                1.0,
                0.0,
                1.001,
                0.0,
                1,
            )
        )
        for idx in range(20)
    )
    return _insert_block_before(
        text,
        "<DCNode>",
        "ACThreeWindingTransformer",
        (
            "idx name i_node j_node k_node i_r i_x j_r j_x k_r k_x gt bt "
            "i_tap i_shift j_tap j_shift k_tap k_shift run_stat"
        ),
        three_winding_rows,
    )


def _append_dc_grid(text: str, zones: Sequence[Sequence[int]]) -> str:
    nodes = [
        _row((node, f"dc10k_{node}", 100.0, 100.0, 0, 1))
        for zone in zones
        for node in zone
    ]
    text = _append_block_rows(text, "DCNode", nodes)

    branches = []
    branch_idx = 38
    zone_roots = []
    for zone_idx, zone in enumerate(zones):
        root = int(zone[0])
        zone_roots.append(root)
        base_parent = 1 + zone_idx % 30
        branches.append(_row((branch_idx, f"dc10k_root_{base_parent}_{root}", base_parent, root, 0.050, 1)))
        branch_idx += 1
        for local_idx in range(1, len(zone)):
            node = int(zone[local_idx])
            row, column = divmod(local_idx, ZONE_COLUMNS)
            parent = int(zone[local_idx - 1] if column else zone[local_idx - ZONE_COLUMNS])
            r = 0.020 + 0.002 * ((zone_idx + row + column) % 5)
            branches.append(_row((branch_idx, f"dc10k_grid_{parent}_{node}", parent, node, r, 1)))
            branch_idx += 1
            if row > 0 and column in {2, 6}:
                vertical_parent = int(zone[local_idx - ZONE_COLUMNS])
                branches.append(_row((branch_idx, f"dc10k_mesh_{vertical_parent}_{node}", vertical_parent, node, 0.032, 1)))
                branch_idx += 1
    for zone_idx in range(5, len(zone_roots), 5):
        i_node = zone_roots[zone_idx - 1]
        j_node = zone_roots[zone_idx]
        branches.append(_row((branch_idx, f"dc10k_zone_tie_{i_node}_{j_node}", i_node, j_node, 0.075, 1)))
        branch_idx += 1
    text = _append_block_rows(text, "DCBranch", branches)

    loads = []
    load_idx = 17
    for zone_idx, zone in enumerate(zones):
        for local_idx in range(11 + zone_idx % 4, len(zone), 6):
            node = int(zone[local_idx])
            pbase = 0.008 + 0.001 * (zone_idx % 6)
            loads.append(_row((load_idx, f"dc10k_load_{node}", node, pbase, 60, 25, 15, 1)))
            load_idx += 1
    text = _append_block_rows(text, "DCLoad", loads)

    text = _append_block_rows(
        text,
        "DCGenerator",
        tuple(
            _row(
                (
                    idx + 15,
                    f"dc10k_gen_p_{idx}",
                    _zone_node(zones, idx, 45),
                    "P",
                    100.0,
                    0.08,
                    0.0,
                    1,
                )
            )
            for idx in range(10)
        ),
    )
    text = _append_block_rows(
        text,
        "DCZeroBranch",
        tuple(
            _row((idx + 9, f"dc10k_zero_{idx}", _zone_node(zones, idx, 90), _zone_node(zones, idx, 98), 1))
            for idx in range(16)
        ),
    )
    text = _append_block_rows(
        text,
        "DCSwitch",
        tuple(
            _row(
                (
                    idx + 5,
                    f"dc10k_switch_{idx}",
                    _zone_node(zones, idx + 16, 88),
                    _zone_node(zones, idx + 16, 97),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(20)
        ),
    )
    return _append_block_rows(
        text,
        "DCBreak",
        tuple(
            _row(
                (
                    idx + 6,
                    f"dc10k_break_{idx}",
                    _zone_node(zones, idx + 30, 86),
                    _zone_node(zones, idx + 30, 96),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(19)
        ),
    )


def _dc_voltage_candidates(
    zones: Sequence[Sequence[int]],
    offsets: Sequence[int],
    *,
    zone_pair: tuple[int, int] = (3, 33),
) -> list[tuple[int, float]]:
    candidates = []
    split = (len(offsets) + 1) // 2
    for offset_idx, offset in enumerate(offsets):
        zone_idx = zone_pair[0] if offset_idx < split else zone_pair[1]
        candidates.append((_zone_node(zones, zone_idx, offset), 160.0))
    return candidates


def _replace_converter_blocks(text: str, ac_zones: Sequence[Sequence[int]], dc_zones: Sequence[Sequence[int]]) -> str:
    acac_modes = (("PQ", "PQ"), ("PV", "PQ"), ("PQ", "PV"), ("PV", "PV"))
    acac_rows = []
    for idx in range(20):
        i_mode, j_mode = acac_modes[idx % len(acac_modes)]
        i_node = _zone_node(ac_zones, idx, 10 + idx % 5)
        j_node = _zone_node(ac_zones, idx + 25, 15 + idx % 5)
        acac_rows.append(
            _row(
                (
                    idx + 1,
                    f"ac10k_acac_{i_mode.lower()}_{j_mode.lower()}_{idx}",
                    i_node,
                    j_node,
                    0.004,
                    0.004,
                    i_mode,
                    j_mode,
                    0.20,
                    0.01 if i_mode == "PQ" else 0.0,
                    -0.01 if j_mode == "PQ" else 0.0,
                    1.005 if i_mode == "PV" else 0.0,
                    0.995 if j_mode == "PV" else 0.0,
                    1,
                )
            )
        )
    text = _replace_block(
        text,
        "ACACConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
        acac_rows,
    )

    dcdc_modes = (("P", "NONE"), ("NONE", "P"), ("V", "NONE"), ("NONE", "V"), ("I", "NONE"), ("NONE", "I"))
    dcdc_voltage_candidates = _dc_voltage_candidates(dc_zones, (2, 3, 4, 5, 2, 3, 4, 5))
    dcdc_voltage_pos = 0
    dcdc_rows = []
    for idx in range(24):
        i_mode, j_mode = dcdc_modes[idx % len(dcdc_modes)]
        i_node = _zone_node(dc_zones, idx, 60 + idx % 12)
        j_node = _zone_node(dc_zones, idx + 25, 62 + idx % 12)
        v_set = 0.0
        if i_mode == "V":
            i_node, v_set = dcdc_voltage_candidates[dcdc_voltage_pos]
            dcdc_voltage_pos += 1
        elif j_mode == "V":
            j_node, v_set = dcdc_voltage_candidates[dcdc_voltage_pos]
            dcdc_voltage_pos += 1
        dcdc_rows.append(
            _row(
                (
                    idx + 1,
                    f"dc10k_dcdc_{i_mode.lower()}_{j_mode.lower()}_{idx}",
                    i_node,
                    j_node,
                    0.004,
                    0.004,
                    i_mode,
                    j_mode,
                    0.20 if "P" in {i_mode, j_mode} else 0.0,
                    0.00001 if "I" in {i_mode, j_mode} else 0.0,
                    v_set,
                    1,
                )
            )
        )
    text = _replace_block(
        text,
        "DCDCConverter",
        "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat",
        dcdc_rows,
    )

    dcac_modes = [("PQ", "V")] * 10 + [("PH", "NONE")] * 2 + [("PQ", "NONE")] * 14 + [("NONE", "P")] * 14
    dcac_voltage_candidates = _dc_voltage_candidates(
        dc_zones,
        (10, 11, 12, 13, 14, 10, 11, 12, 13, 14),
        zone_pair=(4, 34),
    )
    dcac_voltage_pos = 0
    dcac_rows = []
    for idx, (ac_mode, dc_mode) in enumerate(dcac_modes):
        if ac_mode == "PH":
            ac_node = _zone_node(ac_zones, 10 * (idx - 9), 4)
        else:
            ac_node = _zone_node(ac_zones, idx * 3, 20 + idx % 15)
        dc_node = _zone_node(dc_zones, idx * 7, 30 + idx % 20)
        v_dc_set = 0.0
        if dc_mode == "V":
            dc_node, v_dc_set = dcac_voltage_candidates[dcac_voltage_pos]
            dcac_voltage_pos += 1
        direction = 1.0 if idx % 2 == 0 else -1.0
        p_ac_set = direction * 0.20 if (ac_mode, dc_mode) == ("PQ", "NONE") else 0.0
        p_dc_set = direction * 0.20 if (ac_mode, dc_mode) == ("NONE", "P") else 0.0
        q_ac_set = direction * 0.01 if ac_mode != "PH" else 0.0
        v_ac_set = 1.05 if ac_mode == "PH" else 0.0
        dev_type = "ACDCConverter" if idx % 2 == 0 else "DCACConverter"
        dcac_rows.append(
            _row(
                (
                    idx + 1,
                    f"hy10k_{dev_type.lower()}_{ac_mode.lower()}_{dc_mode.lower()}_{idx}",
                    ac_node,
                    dc_node,
                    0.004,
                    0.004,
                    ac_mode,
                    dc_mode,
                    p_ac_set,
                    p_dc_set,
                    q_ac_set,
                    v_ac_set,
                    v_dc_set,
                    1,
                    dev_type,
                )
            )
        )
    return _replace_block(
        text,
        "DCACConverter",
        "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat dev_type",
        dcac_rows,
    )


def build_case_text(ac_node_count: int = AC_NODE_COUNT, dc_node_count: int = DC_NODE_COUNT) -> str:
    if ac_node_count != AC_NODE_COUNT or dc_node_count != DC_NODE_COUNT:
        raise ValueError("this benchmark is fixed at 5000 AC plus 5000 DC nodes")
    ac_zones = _partition_nodes(11, ac_node_count)
    dc_zones = _partition_nodes(31, dc_node_count)
    text = BASE_CASE.read_text(encoding="utf-8")
    text = _append_ac_grid(text, ac_zones)
    text = _append_dc_grid(text, dc_zones)
    return _replace_converter_blocks(text, ac_zones, dc_zones)


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
    model_path.write_text(build_case_text(), encoding="utf-8")
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
