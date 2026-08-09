#!/usr/bin/env python3
"""Generate a 20000-node hybrid benchmark with all supported device families."""

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
from scripts.generate_hybrid_all_elements_10k import _partition_nodes, _zone_node
from scripts.generate_hybrid_converter_all_modes_1k import (
    _append_block_rows,
    _measurement_template,
    _replace_block,
    _row,
)
from scripts.update_meas_from_lf import rewrite_measurements, solve_hybrid


BASE_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
DEFAULT_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_all_elements_20k.e"
DEFAULT_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_all_elements_20k.meas"

AC_NODE_COUNT = 10000
DC_NODE_COUNT = 10000
ZONE_COUNT = 80
AC_SPOKES = 8
DC_SPOKES = 10


def _spoke_node(zone: Sequence[int], spoke_count: int, spoke: int, depth: int) -> int | None:
    """Return a node in a one-based spoke depth, or None past the last layer."""
    local_idx = 1 + (depth - 1) * spoke_count + spoke
    if local_idx >= len(zone):
        return None
    return int(zone[local_idx])


def _append_ac_network(text: str, zones: Sequence[Sequence[int]]) -> str:
    nodes = [
        _row((node, f"ac20k_{node}", 1.0, 1.0, 0.0, 0, 1))
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
            _row(
                (
                    branch_idx,
                    f"ac20k_root_{base_parent}_{root}",
                    base_parent,
                    root,
                    0.014 + 0.001 * (zone_idx % 4),
                    0.055 + 0.004 * (zone_idx % 5),
                    0.0,
                    1,
                )
            )
        )
        branch_idx += 1

        for local_idx in range(1, len(zone)):
            node = int(zone[local_idx])
            spoke = (local_idx - 1) % AC_SPOKES
            depth = (local_idx - 1) // AC_SPOKES + 1
            parent = root if depth == 1 else int(zone[local_idx - AC_SPOKES])
            r = 0.006 + 0.0005 * ((zone_idx + spoke) % 5)
            x = 0.024 + 0.002 * ((zone_idx + depth) % 6)
            b = 0.0001 * ((spoke + depth) % 3)
            branches.append(_row((branch_idx, f"ac20k_spoke_{parent}_{node}", parent, node, r, x, b, 1)))
            branch_idx += 1

        max_depth = (len(zone) - 2) // AC_SPOKES + 1
        for depth in range(3, max_depth + 1, 3):
            layer = [
                node
                for spoke in range(AC_SPOKES)
                if (node := _spoke_node(zone, AC_SPOKES, spoke, depth)) is not None
            ]
            for pos in range(len(layer) - 1):
                i_node = layer[pos]
                j_node = layer[pos + 1]
                branches.append(
                    _row((branch_idx, f"ac20k_ring_{i_node}_{j_node}", i_node, j_node, 0.012, 0.050, 0.0, 1))
                )
                branch_idx += 1
            if len(layer) == AC_SPOKES:
                branches.append(
                    _row((branch_idx, f"ac20k_ring_{layer[-1]}_{layer[0]}", layer[-1], layer[0], 0.013, 0.052, 0.0, 1))
                )
                branch_idx += 1

    for zone_idx in range(0, len(zone_roots), 8):
        group = zone_roots[zone_idx : zone_idx + 8]
        for pos in range(len(group) - 1):
            branches.append(
                _row((branch_idx, f"ac20k_zone_tie_{group[pos]}_{group[pos + 1]}", group[pos], group[pos + 1], 0.025, 0.100, 0.0, 1))
            )
            branch_idx += 1
    text = _append_block_rows(text, "ACBranch", branches)

    loads = []
    load_idx = 8
    for zone_idx, zone in enumerate(zones):
        for local_idx in range(9 + zone_idx % 5, len(zone), 7):
            node = int(zone[local_idx])
            pbase = 0.006 + 0.0005 * (zone_idx % 7)
            qbase = 0.0025 + 0.0004 * (local_idx % 6)
            loads.append(_row((load_idx, f"ac20k_load_{node}", node, pbase, 50, 30, 20, qbase, 45, 35, 20, 1)))
            load_idx += 1
    text = _append_block_rows(text, "ACLoad", loads)

    generator_rows = []
    generator_idx = 5
    generator_modes = (["PQ"] * 18) + (["P"] * 15) + (["PV"] * 10) + (["V"] * 6) + ["PH"]
    for offset, mode in enumerate(generator_modes):
        if mode == "PH":
            zone_idx = 0
            local_offset = 2
        elif mode == "V":
            zone_idx = 10 * (offset - 43)
            local_offset = 3
        else:
            zone_idx = (offset * 11 + 7) % ZONE_COUNT
            local_offset = 34 + offset % 18
        node = _zone_node(zones, zone_idx, local_offset)
        p_set = 0.060 if mode in {"PQ", "P", "PV"} else 0.0
        q_set = 0.012 if mode == "PQ" else 0.0
        v_set = 1.0 if mode in {"PV", "V"} else (1.05 if mode == "PH" else 0.0)
        generator_rows.append(
            _row((generator_idx, f"ac20k_gen_{mode.lower()}_{node}", node, mode, p_set, q_set, v_set, 1.0, 1))
        )
        generator_idx += 1
    text = _append_block_rows(text, "ACGenerator", generator_rows)

    shunt_rows = []
    shunt_idx = 4
    shunt_modes = ("Q", "Z", "B")
    for offset in range(48):
        node = _zone_node(zones, offset, 66 + offset % 12)
        mode = shunt_modes[offset % len(shunt_modes)]
        q_set = 0.08 if mode == "Q" else 0.0
        b_set = -0.003 if mode == "Z" else (0.002 if mode == "B" else 0.0)
        shunt_rows.append(
            _row((shunt_idx, f"ac20k_shunt_{mode.lower()}_{node}", node, mode, q_set, 0.0, b_set, 0.0, 1))
        )
        shunt_idx += 1
    text = _append_block_rows(text, "ACShuntCompensator", shunt_rows)

    text = _append_block_rows(
        text,
        "ACZeroBranch",
        tuple(
            _row((idx + 2, f"ac20k_zero_{idx}", _zone_node(zones, idx, 92), _zone_node(zones, idx, 108), 1))
            for idx in range(48)
        ),
    )
    text = _append_block_rows(
        text,
        "ACSwitch",
        tuple(
            _row(
                (
                    idx + 1,
                    f"ac20k_switch_{idx}",
                    _zone_node(zones, idx + 8, 89),
                    _zone_node(zones, idx + 8, 105),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(48)
        ),
    )
    text = _append_block_rows(
        text,
        "ACBreak",
        tuple(
            _row(
                (
                    idx + 2,
                    f"ac20k_break_{idx}",
                    _zone_node(zones, idx + 32, 87),
                    _zone_node(zones, idx + 32, 111),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(48)
        ),
    )
    text = _append_block_rows(
        text,
        "ACTransformer",
        tuple(
            _row(
                (
                    idx + 2,
                    f"ac20k_transformer_{idx}",
                    _zone_node(zones, idx, 78 + idx % 10),
                    _zone_node(zones, idx + 19, 80 + idx % 10),
                    0.010 + 0.001 * (idx % 4),
                    0.075 + 0.004 * (idx % 6),
                    0.0,
                    -0.0001,
                    0.998 + 0.002 * (idx % 3),
                    0.0015 * ((idx % 5) - 2),
                    1,
                )
            )
            for idx in range(48)
        ),
    )

    three_winding_rows = tuple(
        _row(
            (
                idx + 1,
                f"ac20k_three_winding_{idx}",
                _zone_node(zones, idx, 54 + idx % 8),
                _zone_node(zones, idx + 21, 58 + idx % 8),
                _zone_node(zones, idx + 43, 62 + idx % 8),
                0.013,
                0.095 + 0.003 * (idx % 5),
                0.014,
                0.100 + 0.003 * (idx % 5),
                0.015,
                0.105 + 0.003 * (idx % 5),
                0.0,
                -0.0001,
                0.999,
                0.0,
                1.0,
                0.0,
                1.001,
                0.0,
                1,
            )
        )
        for idx in range(40)
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


def _append_dc_network(text: str, zones: Sequence[Sequence[int]]) -> str:
    nodes = [
        _row((node, f"dc20k_{node}", 100.0, 100.0, 0, 1))
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
        branches.append(
            _row((branch_idx, f"dc20k_root_{base_parent}_{root}", base_parent, root, 0.035 + 0.003 * (zone_idx % 5), 1))
        )
        branch_idx += 1

        for local_idx in range(1, len(zone)):
            node = int(zone[local_idx])
            spoke = (local_idx - 1) % DC_SPOKES
            depth = (local_idx - 1) // DC_SPOKES + 1
            parent = root if depth == 1 else int(zone[local_idx - DC_SPOKES])
            r = 0.014 + 0.0015 * ((zone_idx + spoke + depth) % 6)
            branches.append(_row((branch_idx, f"dc20k_spoke_{parent}_{node}", parent, node, r, 1)))
            branch_idx += 1

        max_depth = (len(zone) - 2) // DC_SPOKES + 1
        for depth in range(3, max_depth + 1, 3):
            layer = [
                node
                for spoke in range(DC_SPOKES)
                if (node := _spoke_node(zone, DC_SPOKES, spoke, depth)) is not None
            ]
            for pos in range(len(layer) - 1):
                i_node = layer[pos]
                j_node = layer[pos + 1]
                branches.append(_row((branch_idx, f"dc20k_ring_{i_node}_{j_node}", i_node, j_node, 0.030, 1)))
                branch_idx += 1
            if len(layer) == DC_SPOKES:
                branches.append(_row((branch_idx, f"dc20k_ring_{layer[-1]}_{layer[0]}", layer[-1], layer[0], 0.032, 1)))
                branch_idx += 1

    for zone_idx in range(0, len(zone_roots), 10):
        group = zone_roots[zone_idx : zone_idx + 10]
        for pos in range(len(group) - 1):
            branches.append(
                _row((branch_idx, f"dc20k_zone_tie_{group[pos]}_{group[pos + 1]}", group[pos], group[pos + 1], 0.060, 1))
            )
            branch_idx += 1
    text = _append_block_rows(text, "DCBranch", branches)

    loads = []
    load_idx = 17
    for zone_idx, zone in enumerate(zones):
        for local_idx in range(8 + zone_idx % 6, len(zone), 7):
            node = int(zone[local_idx])
            pbase = 0.006 + 0.0006 * (zone_idx % 7)
            loads.append(_row((load_idx, f"dc20k_load_{node}", node, pbase, 55, 30, 15, 1)))
            load_idx += 1
    text = _append_block_rows(text, "DCLoad", loads)

    text = _append_block_rows(
        text,
        "DCGenerator",
        tuple(
            _row(
                (
                    idx + 15,
                    f"dc20k_gen_p_{idx}",
                    _zone_node(zones, idx * 13, 38 + idx % 20),
                    "P",
                    100.0,
                    0.060,
                    0.0,
                    1,
                )
            )
            for idx in range(48)
        ),
    )
    text = _append_block_rows(
        text,
        "DCZeroBranch",
        tuple(
            _row((idx + 9, f"dc20k_zero_{idx}", _zone_node(zones, idx, 91), _zone_node(zones, idx, 111), 1))
            for idx in range(48)
        ),
    )
    text = _append_block_rows(
        text,
        "DCSwitch",
        tuple(
            _row(
                (
                    idx + 5,
                    f"dc20k_switch_{idx}",
                    _zone_node(zones, idx + 12, 88),
                    _zone_node(zones, idx + 12, 108),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(48)
        ),
    )
    return _append_block_rows(
        text,
        "DCBreak",
        tuple(
            _row(
                (
                    idx + 6,
                    f"dc20k_break_{idx}",
                    _zone_node(zones, idx + 36, 86),
                    _zone_node(zones, idx + 36, 116),
                    1 if idx % 2 == 0 else 0,
                    1,
                )
            )
            for idx in range(48)
        ),
    )


def _voltage_candidates(
    zones: Sequence[Sequence[int]],
    count: int,
    *,
    zone_indices: Sequence[int],
    first_offset: int,
) -> list[tuple[int, float]]:
    return [
        (
            _zone_node(zones, zone_indices[pos % len(zone_indices)], first_offset + pos // len(zone_indices)),
            160.0,
        )
        for pos in range(count)
    ]


def _replace_converter_blocks(text: str, ac_zones: Sequence[Sequence[int]], dc_zones: Sequence[Sequence[int]]) -> str:
    acac_modes = (("PQ", "PQ"), ("PV", "PQ"), ("PQ", "PV"), ("PV", "PV"))
    acac_rows = []
    for idx in range(40):
        i_mode, j_mode = acac_modes[idx % len(acac_modes)]
        i_node = _zone_node(ac_zones, idx * 3, 18 + idx % 12)
        j_node = _zone_node(ac_zones, idx * 3 + 37, 24 + idx % 12)
        acac_rows.append(
            _row(
                (
                    idx + 1,
                    f"ac20k_acac_{i_mode.lower()}_{j_mode.lower()}_{idx}",
                    i_node,
                    j_node,
                    0.003,
                    0.003,
                    i_mode,
                    j_mode,
                    0.15 if idx % 2 == 0 else -0.15,
                    0.008 if i_mode == "PQ" else 0.0,
                    -0.008 if j_mode == "PQ" else 0.0,
                    1.003 if i_mode == "PV" else 0.0,
                    0.997 if j_mode == "PV" else 0.0,
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
    dcdc_voltage_candidates = _voltage_candidates(
        dc_zones,
        16,
        zone_indices=(3, 33, 63),
        first_offset=3,
    )
    dcdc_voltage_pos = 0
    dcdc_rows = []
    for idx in range(48):
        i_mode, j_mode = dcdc_modes[idx % len(dcdc_modes)]
        i_node = _zone_node(dc_zones, idx * 5, 48 + idx % 16)
        j_node = _zone_node(dc_zones, idx * 5 + 29, 52 + idx % 16)
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
                    f"dc20k_dcdc_{i_mode.lower()}_{j_mode.lower()}_{idx}",
                    i_node,
                    j_node,
                    0.003,
                    0.003,
                    i_mode,
                    j_mode,
                    0.15 if "P" in {i_mode, j_mode} else 0.0,
                    0.000008 if "I" in {i_mode, j_mode} else 0.0,
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

    dcac_modes = ([('PQ', 'V')] * 20) + ([('PH', 'NONE')] * 4) + ([('PQ', 'NONE')] * 28) + ([('NONE', 'P')] * 28)
    dcac_voltage_candidates = _voltage_candidates(
        dc_zones,
        20,
        zone_indices=(4, 34, 64),
        first_offset=12,
    )
    dcac_voltage_pos = 0
    dcac_rows = []
    ph_zone_indices = (0, 20, 40, 60)
    for idx, (ac_mode, dc_mode) in enumerate(dcac_modes):
        if ac_mode == "PH":
            ac_node = _zone_node(ac_zones, ph_zone_indices[idx - 20], 4)
        else:
            ac_node = _zone_node(ac_zones, idx * 7, 28 + idx % 20)
        dc_node = _zone_node(dc_zones, idx * 11, 30 + idx % 22)
        v_dc_set = 0.0
        if dc_mode == "V":
            dc_node, v_dc_set = dcac_voltage_candidates[dcac_voltage_pos]
            dcac_voltage_pos += 1
        direction = 1.0 if idx % 2 == 0 else -1.0
        p_ac_set = direction * 0.15 if (ac_mode, dc_mode) == ("PQ", "NONE") else 0.0
        p_dc_set = direction * 0.15 if (ac_mode, dc_mode) == ("NONE", "P") else 0.0
        q_ac_set = direction * 0.008 if ac_mode != "PH" else 0.0
        v_ac_set = 1.05 if ac_mode == "PH" else 0.0
        dev_type = "ACDCConverter" if idx % 2 == 0 else "DCACConverter"
        dcac_rows.append(
            _row(
                (
                    idx + 1,
                    f"hy20k_converter_{idx}",
                    ac_node,
                    dc_node,
                    0.003,
                    0.003,
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
        raise ValueError("this benchmark is fixed at 10000 AC plus 10000 DC nodes")
    ac_zones = _partition_nodes(11, ac_node_count, ZONE_COUNT)
    dc_zones = _partition_nodes(31, dc_node_count, ZONE_COUNT)
    text = BASE_CASE.read_text(encoding="utf-8")
    text = _append_ac_network(text, ac_zones)
    text = _append_dc_network(text, dc_zones)
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
