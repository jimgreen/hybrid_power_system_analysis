"""Generate MATPOWER ``.m`` cases from local IEEE AC E files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pypower.idx_brch import ANGMAX, ANGMIN, BR_B, BR_R, BR_STATUS, BR_X, F_BUS, PF, PT, QF, QT, RATE_A, RATE_B, RATE_C, SHIFT, T_BUS, TAP
from pypower.idx_bus import BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE, GS, PD, QD, VA, VM, VMAX, VMIN, ZONE
from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
for path in (SRC_DIR, ROOT_DIR / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.ac_array_model import (  # noqa: E402
    BREAK_COLS,
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    SHUNT_COLS,
    SWITCH_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
    build_ac_ppc_from_e_file,
)


PQ = 1
PV = 2
REF = 3


class DSU:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, value: int) -> int:
        value = int(value)
        while int(self.parent[value]) != value:
            self.parent[value] = self.parent[int(self.parent[value])]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _active_rows(rows: np.ndarray, run_col: int) -> np.ndarray:
    if rows.size == 0:
        return rows
    return rows[rows[:, run_col] == 1]


def _build_components(acppc: dict) -> Tuple[np.ndarray, List[List[int]], dict, np.ndarray]:
    bus = np.asarray(acppc["bus"])
    zero = np.asarray(acppc["zero_branch"])
    switch = np.asarray(acppc["switch"])
    breaker = np.asarray(acppc.get("break", np.zeros((0, len(BREAK_COLS)), dtype=np.float64)))

    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    row_by_node = {int(node): row for row, node in enumerate(node_ids)}
    active_bus = bus[:, BUS_COLS["run_stat"]] == 1
    dsu = DSU(bus.shape[0])

    for row in _active_rows(zero, ZERO_BRANCH_COLS["run_stat"]):
        left = row_by_node.get(int(row[ZERO_BRANCH_COLS["i_node"]]))
        right = row_by_node.get(int(row[ZERO_BRANCH_COLS["j_node"]]))
        if left is not None and right is not None and active_bus[left] and active_bus[right]:
            dsu.union(left, right)

    if switch.size:
        live = (switch[:, SWITCH_COLS["run_stat"]] == 1) & (switch[:, SWITCH_COLS["status"]] == 1)
        for row in switch[live]:
            left = row_by_node.get(int(row[SWITCH_COLS["i_node"]]))
            right = row_by_node.get(int(row[SWITCH_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)

    if breaker.size:
        live = (breaker[:, BREAK_COLS["run_stat"]] == 1) & (breaker[:, BREAK_COLS["status"]] == 1)
        for row in breaker[live]:
            left = row_by_node.get(int(row[BREAK_COLS["i_node"]]))
            right = row_by_node.get(int(row[BREAK_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)

    root_to_comp = {}
    comp_rows: List[List[int]] = []
    for row in np.flatnonzero(active_bus):
        root = dsu.find(int(row))
        if root not in root_to_comp:
            root_to_comp[root] = len(comp_rows)
            comp_rows.append([])
        comp_rows[root_to_comp[root]].append(int(row))

    row_to_comp = np.full(bus.shape[0], -1, dtype=np.int64)
    for comp, rows in enumerate(comp_rows):
        row_to_comp[rows] = comp
    comp_to_bus_id = np.arange(1, len(comp_rows) + 1, dtype=np.int64)
    row_to_bus_id = np.where(row_to_comp >= 0, comp_to_bus_id[np.maximum(row_to_comp, 0)], -1)
    return row_to_comp, comp_rows, row_by_node, row_to_bus_id


def build_matpower_projection_from_acppc(acppc: dict) -> Dict[str, object]:
    """Build a MATPOWER ppc plus projection metadata from an AC array ppc.

    Zero branches, closed switches, and closed breakers are collapsed into one
    MATPOWER bus. ZIP loads are exported as static P/Q at V=1. Transformer
    ``gt/bt`` grounding admittance is converted to an equivalent i-side bus
    shunt referred through the tap magnitude.
    """

    base_mva = float(acppc["base"][0])
    bus0 = np.asarray(acppc["bus"])
    branch0 = np.asarray(acppc["branch"])
    transformer0 = np.asarray(acppc["transformer"])
    gen0 = np.asarray(acppc["gen"])
    load0 = np.asarray(acppc["load"])
    shunt0 = np.asarray(acppc["shunt"])

    row_to_comp, comp_rows, row_by_node, row_to_bus_id = _build_components(acppc)
    comp_count = len(comp_rows)
    comp_to_bus_id = np.arange(1, comp_count + 1, dtype=np.int64)

    pd = np.zeros(comp_count, dtype=np.float64)
    qd = np.zeros(comp_count, dtype=np.float64)
    gs = np.zeros(comp_count, dtype=np.float64)
    bs = np.zeros(comp_count, dtype=np.float64)
    bus_type = np.full(comp_count, PQ, dtype=np.float64)
    base_kv = np.zeros(comp_count, dtype=np.float64)
    vm0 = np.ones(comp_count, dtype=np.float64)
    va0 = np.zeros(comp_count, dtype=np.float64)

    for comp, rows in enumerate(comp_rows):
        first = rows[0]
        base_kv[comp] = bus0[first, BUS_COLS["vbase"]]
        vm0[comp] = bus0[first, BUS_COLS["voltage"]]
        va0[comp] = np.degrees(bus0[first, BUS_COLS["angle"]])

    for row in _active_rows(load0, LOAD_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[LOAD_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        pd[comp] += row[LOAD_COLS["pbase"]] * (row[LOAD_COLS["pv0"]] + row[LOAD_COLS["pv1"]] + row[LOAD_COLS["pv2"]]) * base_mva
        qd[comp] += row[LOAD_COLS["qbase"]] * (row[LOAD_COLS["qv0"]] + row[LOAD_COLS["qv1"]] + row[LOAD_COLS["qv2"]]) * base_mva

    for row in _active_rows(shunt0, SHUNT_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[SHUNT_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        gs[comp] += row[SHUNT_COLS["g_set"]] * base_mva
        bs[comp] += row[SHUNT_COLS["b_set"]] * base_mva

    for row in _active_rows(transformer0, TRANSFORMER_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[TRANSFORMER_COLS["i_node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        tap = row[TRANSFORMER_COLS["tap"]]
        tap_mag = tap if abs(tap) > 1e-12 else 1.0
        scale = 1.0 / (tap_mag * tap_mag)
        gs[comp] += row[TRANSFORMER_COLS["gt"]] * scale * base_mva
        bs[comp] += row[TRANSFORMER_COLS["bt"]] * scale * base_mva

    for row in _active_rows(gen0, GEN_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        control = int(row[GEN_COLS["control_type"]])
        if control == CTRL_SLACK:
            bus_type[comp] = REF
        elif bus_type[comp] != REF and control in (CTRL_PV, CTRL_P):
            bus_type[comp] = PV

    bus = np.zeros((comp_count, 13), dtype=np.float64)
    bus[:, BUS_I] = comp_to_bus_id
    bus[:, BUS_TYPE] = bus_type
    bus[:, PD] = pd
    bus[:, QD] = qd
    bus[:, GS] = gs
    bus[:, BS] = bs
    bus[:, BUS_AREA] = 1
    bus[:, VM] = vm0
    bus[:, VA] = va0
    bus[:, BASE_KV] = base_kv
    bus[:, ZONE] = 1
    bus[:, VMAX] = 1.2
    bus[:, VMIN] = 0.8

    gen_rows = []
    for row in _active_rows(gen0, GEN_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        gen_row = np.zeros(21, dtype=np.float64)
        gen_row[GEN_BUS] = row_to_bus_id[bus_row]
        gen_row[PG] = row[GEN_COLS["p_set"]] * base_mva
        gen_row[QG] = row[GEN_COLS["q_set"]] * base_mva
        gen_row[QMAX] = 1e9
        gen_row[QMIN] = -1e9
        gen_row[VG] = row[GEN_COLS["v_set"]]
        gen_row[MBASE] = base_mva
        gen_row[GEN_STATUS] = 1
        gen_row[PMAX] = 1e9
        gen_row[PMIN] = -1e9
        gen_rows.append(gen_row)
    gen = np.vstack(gen_rows) if gen_rows else np.zeros((0, 21), dtype=np.float64)

    branch_rows = []
    branch_map: List[Tuple[str, int]] = []
    transformer_shunt_meta = []

    def add_branch_devices(devices: np.ndarray, cols: dict, is_transformer: bool) -> None:
        for pos, row in enumerate(devices):
            if int(row[cols["run_stat"]]) != 1:
                continue
            i_row = row_by_node.get(int(row[cols["i_node"]]))
            j_row = row_by_node.get(int(row[cols["j_node"]]))
            if i_row is None or j_row is None or row_to_comp[i_row] < 0 or row_to_comp[j_row] < 0:
                continue
            i_bus = row_to_bus_id[i_row]
            j_bus = row_to_bus_id[j_row]
            if i_bus == j_bus:
                continue
            branch_row = np.zeros(13, dtype=np.float64)
            branch_row[F_BUS] = i_bus
            branch_row[T_BUS] = j_bus
            branch_row[BR_R] = row[cols["r"]]
            branch_row[BR_X] = row[cols["x"]]
            branch_row[BR_B] = 0.0 if is_transformer else row[cols["b"]]
            branch_row[RATE_A] = 0.0
            branch_row[RATE_B] = 0.0
            branch_row[RATE_C] = 0.0
            if is_transformer:
                tap = row[cols["tap"]]
                branch_row[TAP] = 0.0 if abs(tap - 1.0) < 1e-12 else tap
                branch_row[SHIFT] = row[cols["shift"]]
            branch_row[BR_STATUS] = 1
            branch_row[ANGMIN] = -360.0
            branch_row[ANGMAX] = 360.0
            mp_branch_idx = len(branch_rows)
            branch_rows.append(branch_row)
            kind = "transformer" if is_transformer else "branch"
            branch_map.append((kind, pos))
            if is_transformer:
                tap = row[cols["tap"]]
                tap_mag = tap if abs(tap) > 1e-12 else 1.0
                scale = 1.0 / (tap_mag * tap_mag)
                transformer_shunt_meta.append(
                    {
                        "mp_branch_idx": mp_branch_idx,
                        "transformer_row": pos,
                        "transformer_idx": int(row[cols["idx"]]),
                        "i_bus": int(i_bus),
                        "j_bus": int(j_bus),
                        "g_pu": float(row[cols["gt"]] * scale),
                        "b_pu": float(row[cols["bt"]] * scale),
                    }
                )

    add_branch_devices(branch0, BRANCH_COLS, False)
    add_branch_devices(transformer0, TRANSFORMER_COLS, True)
    branch = np.vstack(branch_rows) if branch_rows else np.zeros((0, 13), dtype=np.float64)
    return {
        "ppc": {"version": "2", "baseMVA": base_mva, "bus": bus, "gen": gen, "branch": branch},
        "row_to_comp": row_to_comp,
        "comp_rows": comp_rows,
        "row_by_node": row_by_node,
        "row_to_bus_id": row_to_bus_id,
        "branch_map": branch_map,
        "transformer_shunt_meta": transformer_shunt_meta,
    }


def build_matpower_ppc_from_acppc(acppc: dict) -> dict:
    """Convert project AC array ppc into a MATPOWER v2 ppc."""
    return build_matpower_projection_from_acppc(acppc)["ppc"]


def extract_matpower_device_losses(mp_result: dict, acppc: dict) -> Dict[str, np.ndarray]:
    """Return MATPOWER branch/transformer losses in project device indexing.

    Transformer totals include the projected i-side ``gt/bt`` bus-shunt term,
    so the returned values are directly comparable with project-side
    ``i_p+j_p`` / ``i_q+j_q`` terminal totals.
    """

    projection = build_matpower_projection_from_acppc(acppc)
    base_mva = float(mp_result["baseMVA"])
    mp_branch = np.asarray(mp_result["branch"], dtype=np.float64)
    mp_bus = {int(row[BUS_I]): row for row in np.asarray(mp_result["bus"], dtype=np.float64)}

    branch_terminal = np.zeros((acppc["branch"].shape[0], 4), dtype=np.float64)
    transformer_terminal = np.zeros((acppc["transformer"].shape[0], 4), dtype=np.float64)
    transformer_projected_shunt = np.zeros((acppc["transformer"].shape[0], 2), dtype=np.float64)

    for mp_idx, (kind, device_idx) in enumerate(projection["branch_map"]):
        row = mp_branch[mp_idx]
        if int(row[BR_STATUS]) != 1:
            continue
        p_i = float(row[PF] / base_mva)
        q_i = float(row[QF] / base_mva)
        p_j = float(row[PT] / base_mva)
        q_j = float(row[QT] / base_mva)
        if kind == "branch":
            branch_terminal[device_idx, :] = (p_i, q_i, p_j, q_j)
        else:
            transformer_terminal[device_idx, :] = (p_i, q_i, p_j, q_j)

    for meta in projection["transformer_shunt_meta"]:
        device_idx = int(meta["transformer_row"])
        i_bus = int(meta["i_bus"])
        vm = float(mp_bus[i_bus][VM])
        p_sh = float(meta["g_pu"] * vm * vm)
        q_sh = float(-meta["b_pu"] * vm * vm)
        transformer_projected_shunt[device_idx, :] = (p_sh, q_sh)

    transformer_total = transformer_terminal.copy()
    transformer_total[:, 0] += transformer_projected_shunt[:, 0]
    transformer_total[:, 1] += transformer_projected_shunt[:, 1]

    return {
        "branch_terminal": branch_terminal,
        "transformer_terminal": transformer_terminal,
        "transformer_projected_shunt": transformer_projected_shunt,
        "transformer_total": transformer_total,
    }


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError(f"Cannot write non-finite MATPOWER value: {value}")
    if abs(value) < 5e-15:
        value = 0.0
    return f"{float(value):.16g}"


def _matrix_block(name: str, matrix: np.ndarray) -> str:
    lines = [f"mpc.{name} = ["]
    for row in np.asarray(matrix, dtype=np.float64):
        lines.append("\t" + "\t".join(_fmt(value) for value in row) + ";")
    lines.append("];")
    return "\n".join(lines)


def write_matpower_case(ppc: dict, output_file: Path, source_file: Path) -> None:
    function_name = output_file.stem
    output_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(
        [
            f"function mpc = {function_name}",
            f"%{function_name} MATPOWER case generated from {source_file.as_posix()}",
            "% Auto-generated by scripts/generate_matpower_cases.py.",
            "% ZIP loads are exported as static P/Q at V=1; zero-impedance ties are bus-collapsed.",
            "mpc.version = '2';",
            f"mpc.baseMVA = {_fmt(ppc['baseMVA'])};",
            _matrix_block("bus", ppc["bus"]),
            _matrix_block("gen", ppc["gen"]),
            _matrix_block("branch", ppc["branch"]),
        ]
    )
    output_file.write_text(text + "\n", encoding="utf-8")


def generate_case(e_file: Path, output_dir: Path) -> Path:
    acppc = build_ac_ppc_from_e_file(e_file)
    matpower_ppc = build_matpower_ppc_from_acppc(acppc)
    output_file = output_dir / f"{e_file.stem}.m"
    write_matpower_case(matpower_ppc, output_file, e_file)
    return output_file


def _case_files(names: Sequence[str], input_dir: Path) -> List[Path]:
    if names:
        files = []
        for name in names:
            file_name = name if name.endswith(".e") else f"{name}.e"
            path = input_dir / file_name
            if not path.exists():
                raise FileNotFoundError(path)
            files.append(path)
        return files
    return sorted(input_dir.glob("ieee*.e"))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate MATPOWER .m cases from IEEE AC E files.")
    parser.add_argument("cases", nargs="*", help="Optional case names, e.g. ieee300 ieee3k. Defaults to all ieee*.e.")
    parser.add_argument("--input-dir", default=str(ROOT_DIR / "data" / "model" / "ac"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data" / "mat"))
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = _case_files(args.cases, input_dir)
    if not files:
        raise RuntimeError(f"No ieee*.e files found in {input_dir}")

    for e_file in files:
        output_file = generate_case(e_file, output_dir)
        print(f"{e_file.name} -> {output_file.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
