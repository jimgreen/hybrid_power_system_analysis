import importlib
import math
from pathlib import Path

import numpy as np
from pypower.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from pypower.idx_bus import BASE_KV, BS, BUS_I, BUS_TYPE, GS, PD, QD, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG


CASES = {
    "ieee9": "case9",
    "ieee14": "case14",
    "ieee24": "case24_ieee_rts",
    "ieee30": "case30",
    "ieee39": "case39",
    "ieee57": "case57",
    "ieee118": "case118",
    "ieee300": "case300",
}


def _fmt(value):
    value = float(value)
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.10g}"


def _load_case(case_name):
    module = importlib.import_module(f"pypower.{case_name}")
    return getattr(module, case_name)()


def _align_rows(header, rows):
    widths = [len(item) for item in header]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "@ " + " ".join(value.ljust(widths[idx]) for idx, value in enumerate(header)).rstrip()
    row_lines = [
        "# " + " ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)).rstrip()
        for row in rows
    ]
    return header_line, row_lines


def _write_block(lines, name, header, rows):
    lines.append(f"<{name}>")
    header_line, row_lines = _align_rows(header, rows)
    lines.append(header_line)
    lines.extend(row_lines)
    lines.append(f"</{name}>")
    lines.append("")


def _row_status(row, col, default=1):
    return int(row[col]) if row.shape[0] > col else default


def convert_case(case_name, out_path):
    mpc = _load_case(case_name)
    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)

    bus_index = {int(row[BUS_I]): idx for idx, row in enumerate(bus)}
    bus_type = {int(row[BUS_I]): int(row[BUS_TYPE]) for row in bus}
    bus_vm = {int(row[BUS_I]): row[VM] for row in bus}
    fallback_kv = base_mva / math.sqrt(3.0)
    bus_kv = {
        int(row[BUS_I]): (row[BASE_KV] if abs(row[BASE_KV]) > 1e-12 else fallback_kv)
        for row in bus
    }

    lines = []
    _write_block(
        lines,
        "PowerBase",
        ["p_base", "u_scale", "p_scale", "i_scale"],
        [[_fmt(base_mva), "1.0", "0.001", "1.0"]],
    )

    node_rows = []
    for row in bus:
        bus_id = int(row[BUS_I])
        run_stat = 0 if int(row[BUS_TYPE]) == 4 else 1
        node_rows.append(
            [
                str(bus_index[bus_id]),
                f"bus_{bus_id}",
                _fmt(bus_kv[bus_id]),
                _fmt(row[VM] * bus_kv[bus_id]),
                _fmt(row[VA]),
                "0",
                str(run_stat),
            ]
        )
    _write_block(
        lines,
        "ACNode",
        ["idx", "name", "vbase", "voltage", "angle", "isl", "run_stat"],
        node_rows,
    )

    branch_rows = []
    transformer_rows = []
    branch_idx = 0
    transformer_idx = 0
    for row in branch:
        run_stat = _row_status(row, BR_STATUS)
        f_bus = int(row[F_BUS])
        t_bus = int(row[T_BUS])
        i_node = bus_index[f_bus]
        j_node = bus_index[t_bus]
        tap = row[TAP] if row.shape[0] > TAP and row[TAP] != 0 else 1.0
        shift = row[SHIFT] if row.shape[0] > SHIFT else 0.0
        is_transformer = (
            abs(tap - 1.0) > 1e-12
            or abs(shift) > 1e-12
            or abs(bus_kv[f_bus] - bus_kv[t_bus]) > 1e-9
        )
        if is_transformer:
            transformer_rows.append(
                [
                    str(transformer_idx),
                    f"tr_{f_bus}_{t_bus}",
                    str(i_node),
                    str(j_node),
                    _fmt(row[BR_R]),
                    _fmt(row[BR_X]),
                    "0.0",
                    _fmt(row[BR_B] / 2.0),
                    _fmt(tap),
                    _fmt(shift),
                    str(run_stat),
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ]
            )
            transformer_idx += 1
        else:
            branch_rows.append(
                [
                    str(branch_idx),
                    f"line_{f_bus}_{t_bus}",
                    str(i_node),
                    str(j_node),
                    _fmt(row[BR_R]),
                    _fmt(row[BR_X]),
                    _fmt(row[BR_B]),
                    str(run_stat),
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ]
            )
            branch_idx += 1
    _write_block(
        lines,
        "ACBranch",
        ["idx", "name", "i_node", "j_node", "r", "x", "b", "run_stat", "i_p", "i_q", "i_c", "j_p", "j_q", "j_c"],
        branch_rows,
    )

    load_rows = []
    load_idx = 0
    for row in bus:
        bus_id = int(row[BUS_I])
        pd = row[PD]
        qd = row[QD]
        if abs(pd) < 1e-12 and abs(qd) < 1e-12:
            continue
        run_stat = 0 if int(row[BUS_TYPE]) == 4 else 1
        load_rows.append(
            [
                str(load_idx),
                f"load_{bus_id}",
                str(bus_index[bus_id]),
                "1.0",
                _fmt(pd),
                "0.0",
                "0.0",
                "1.0",
                _fmt(qd),
                "0.0",
                "0.0",
                str(run_stat),
                "0.0",
                "0.0",
                "0.0",
            ]
        )
        load_idx += 1
    _write_block(
        lines,
        "ACLoad",
        ["idx", "name", "node", "pbase", "pv0", "pv1", "pv2", "qbase", "qv0", "qv1", "qv2", "run_stat", "p", "q", "current"],
        load_rows,
    )

    gen_rows = []
    for gen_idx, row in enumerate(gen):
        bus_id = int(row[GEN_BUS])
        run_stat = _row_status(row, GEN_STATUS)
        btype = bus_type.get(bus_id, 1)
        control_type = "V" if btype == 3 else "PV"
        gen_rows.append(
            [
                str(gen_idx),
                f"gen_{bus_id}_{gen_idx}",
                str(bus_index[bus_id]),
                control_type,
                _fmt(row[PG]),
                _fmt(row[QG]),
                _fmt((row[VG] if row.shape[0] > VG else bus_vm[bus_id]) * bus_kv[bus_id]),
                "1.0",
                str(run_stat),
                "0.0",
                "0.0",
                "0.0",
            ]
        )
    _write_block(
        lines,
        "ACGenerator",
        ["idx", "name", "node", "control_type", "p_set", "q_set", "v_set", "alpha", "run_stat", "p", "q", "current"],
        gen_rows,
    )

    shunt_rows = []
    shunt_idx = 0
    for row in bus:
        bus_id = int(row[BUS_I])
        gs = row[GS] / base_mva
        bs = row[BS] / base_mva
        if abs(gs) < 1e-12 and abs(bs) < 1e-12:
            continue
        run_stat = 0 if int(row[BUS_TYPE]) == 4 else 1
        shunt_rows.append(
            [
                str(shunt_idx),
                f"shunt_{bus_id}",
                str(bus_index[bus_id]),
                "B",
                "0.0",
                _fmt(gs),
                _fmt(bs),
                "0.0",
                str(run_stat),
                "0.0",
                "0.0",
                "0.0",
            ]
        )
        shunt_idx += 1
    _write_block(
        lines,
        "ACShuntCompensator",
        ["idx", "name", "node", "control_type", "q_set", "g_set", "b_set", "v_set", "run_stat", "p", "q", "current"],
        shunt_rows,
    )

    _write_block(lines, "ACZeroBranch", ["idx", "name", "i_node", "j_node", "run_stat", "p", "q", "current"], [])
    _write_block(lines, "ACSwitch", ["idx", "name", "i_node", "j_node", "status", "run_stat", "p", "q", "current"], [])
    _write_block(
        lines,
        "ACTransformer",
        ["idx", "name", "i_node", "j_node", "r", "x", "gt", "bt", "tap", "shift", "run_stat", "i_p", "i_q", "i_c", "j_p", "j_q", "j_c"],
        transformer_rows,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "case": case_name,
        "out": str(out_path),
        "nodes": len(bus),
        "branches": len(branch_rows),
        "transformers": len(transformer_rows),
        "loads": len(load_rows),
        "generators": len(gen_rows),
        "shunts": len(shunt_rows),
    }


def main():
    data_dir = Path("data") / "model" / "ac"
    summaries = []
    for out_name, case_name in CASES.items():
        summaries.append(convert_case(case_name, data_dir / f"{out_name}.e"))
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
