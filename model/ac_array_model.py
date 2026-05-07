import math
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from efile_read import read_efile_rows_cached


CTRL_PQ = 0
CTRL_P = 1
CTRL_PV = 2
CTRL_SLACK = 3
CTRL_CODE = {
    "PQ": CTRL_PQ,
    "P": CTRL_P,
    "PV": CTRL_PV,
    "V": CTRL_SLACK,
    "SLACK": CTRL_SLACK,
    "PH": CTRL_SLACK,
}

SHUNT_Q = 0
SHUNT_V = 1
SHUNT_B = 2
SHUNT_Z = 3
SHUNT_CODE = {
    "Q": SHUNT_Q,
    "V": SHUNT_V,
    "B": SHUNT_B,
    "Z": SHUNT_Z,
}

BUS_COLS = {
    "idx": 0,
    "vbase": 1,
    "voltage": 2,
    "angle": 3,
    "isl": 4,
    "run_stat": 5,
}
BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "b": 5,
    "run_stat": 6,
    "i_p": 7,
    "i_q": 8,
    "i_c": 9,
    "j_p": 10,
    "j_q": 11,
    "j_c": 12,
}
TRANSFORMER_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "b": 5,
    "tap": 6,
    "shift": 7,
    "run_stat": 8,
    "i_p": 9,
    "i_q": 10,
    "i_c": 11,
    "j_p": 12,
    "j_q": 13,
    "j_c": 14,
}
GEN_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "p_set": 3,
    "q_set": 4,
    "v_set": 5,
    "alpha": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
}
LOAD_COLS = {
    "idx": 0,
    "node": 1,
    "pbase": 2,
    "pv0": 3,
    "pv1": 4,
    "pv2": 5,
    "qbase": 6,
    "qv0": 7,
    "qv1": 8,
    "qv2": 9,
    "run_stat": 10,
    "p": 11,
    "q": 12,
    "current": 13,
}
SHUNT_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "q_set": 3,
    "g_set": 4,
    "b_set": 5,
    "v_set": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
}
ZERO_BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "run_stat": 3,
    "p": 4,
    "q": 5,
    "current": 6,
}
SWITCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "status": 3,
    "run_stat": 4,
    "p": 5,
    "q": 6,
    "current": 7,
}
BREAK_COLS = SWITCH_COLS

_AC_PPC_CACHE = {}
_AC_PPC_CACHE_LOCK = threading.Lock()


def _table(raw: Dict, block: str) -> Dict:
    return raw.get(block, {"header_list": [], "rows": []})


def _rows(raw: Dict, block: str) -> List[List[str]]:
    return _table(raw, block)["rows"]


def _columns(raw: Dict, block: str) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(_table(raw, block)["header_list"])}


def _float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value in (None, "") else float(value)


def _int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    return default if value in (None, "") else int(float(value))


def _cell(row: List[str], cols: Dict[str, int], key: str, default=""):
    idx = cols.get(key)
    if idx is None or idx >= len(row):
        return default
    value = row[idx]
    return default if value == "" else value


def _float_cell(row: List[str], cols: Dict[str, int], key: str, default: float = 0.0) -> float:
    value = _cell(row, cols, key, default)
    return default if value in (None, "") else float(value)


def _float_column(rows: List[List[str]], col: int, default: float = 0.0) -> np.ndarray:
    return np.fromiter(
        (float(row[col]) if row[col] else default for row in rows),
        dtype=np.float64,
        count=len(rows),
    )


def _row_array(rows: List[List[str]]) -> np.ndarray:
    return np.asarray(rows, dtype=object) if rows else np.empty((0, 0), dtype=object)


def _numeric_columns(row_array: np.ndarray, cols: Iterable[int], defaults=None) -> np.ndarray:
    values = row_array[:, tuple(cols)]
    if defaults is not None:
        values = np.where(values == "", np.asarray(defaults, dtype=object), values)
    return values.astype(np.float64, copy=False)


def _int_cell(row: List[str], cols: Dict[str, int], key: str, default: int = 0) -> int:
    value = _cell(row, cols, key, default)
    return default if value in (None, "") else int(float(value))


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = Path(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def clear_ac_ppc_cache(file_path=None) -> None:
    with _AC_PPC_CACHE_LOCK:
        if file_path is None:
            _AC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _AC_PPC_CACHE.pop((path, True), None)
            _AC_PPC_CACHE.pop((path, False), None)


def _node_row_maps(bus: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    return {int(node_id): pos for pos, node_id in enumerate(node_ids)}, node_ids


def _node_voltage_base(raw_vbase_by_idx: Dict[int, float], node_idx: int) -> float:
    return raw_vbase_by_idx[int(node_idx)]


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def build_ac_ppc_from_e_file(
    file_path,
    use_cache: bool = True,
    copy_arrays: bool = False,
    include_device_names: bool = True,
) -> Dict:
    """Build a MATPOWER-like NumPy dictionary for AC power-flow fast paths.

    The returned arrays are in pu/radians and should be treated as read-only by
    callers. Set copy_arrays=True when a caller needs to mutate the input model.
    """
    file_key = _file_cache_key(file_path)
    if use_cache:
        with _AC_PPC_CACHE_LOCK:
            cached = _AC_PPC_CACHE.get((file_key[0], include_device_names))
            if cached is not None and cached[0] == file_key:
                return _copy_ppc(cached[1]) if copy_arrays else cached[1]

    raw = read_efile_rows_cached(file_key[0], use_cache=use_cache)
    base_cols = _columns(raw, "PowerBase")
    base_row = _rows(raw, "PowerBase")[0]
    p_base = _float_cell(base_row, base_cols, "p_base")
    u_scale = _float_cell(base_row, base_cols, "u_scale")
    p_scale = _float_cell(base_row, base_cols, "p_scale")
    i_scale = _float_cell(base_row, base_cols, "i_scale")

    node_rows = _rows(raw, "ACNode")
    node_cols = _columns(raw, "ACNode")
    node_idx_col = node_cols["idx"]
    node_name_col = node_cols.get("name")
    node_vbase_col = node_cols["vbase"]
    node_voltage_col = node_cols["voltage"]
    node_angle_col = node_cols["angle"]
    node_isl_col = node_cols.get("isl")
    node_run_col = node_cols["run_stat"]
    bus = np.zeros((len(node_rows), len(BUS_COLS)), dtype=np.float64)
    if node_rows:
        node_table = _row_array(node_rows)
        node_idx_values = node_table[:, node_idx_col].astype(np.float64, copy=False)
        raw_vbase_values = node_table[:, node_vbase_col].astype(np.float64, copy=False)
        voltage_values = np.where(
            node_table[:, node_voltage_col] == "",
            raw_vbase_values,
            node_table[:, node_voltage_col],
        ).astype(np.float64, copy=False)
        angle_values = np.where(node_table[:, node_angle_col] == "", 0.0, node_table[:, node_angle_col]).astype(
            np.float64,
            copy=False,
        )
        bus[:, BUS_COLS["idx"]] = node_idx_values
        bus[:, BUS_COLS["vbase"]] = raw_vbase_values / u_scale
        bus[:, BUS_COLS["voltage"]] = voltage_values / raw_vbase_values
        bus[:, BUS_COLS["angle"]] = np.deg2rad(angle_values)
        if node_isl_col is not None:
            bus[:, BUS_COLS["isl"]] = np.where(node_table[:, node_isl_col] == "", 0.0, node_table[:, node_isl_col]).astype(
                np.float64,
                copy=False,
            )
        bus[:, BUS_COLS["run_stat"]] = np.where(node_table[:, node_run_col] == "", 1.0, node_table[:, node_run_col]).astype(
            np.float64,
            copy=False,
        )
        raw_vbase_by_idx: Dict[int, float] = {
            int(idx): float(vbase) for idx, vbase in zip(node_idx_values, raw_vbase_values)
        }
    else:
        node_idx_values = np.array([], dtype=np.float64)
        raw_vbase_by_idx = {}
    bus_names = np.asarray(
        [
            row[node_name_col] if node_name_col is not None and row[node_name_col] else f"bus_{int(node_idx_values[pos])}"
            for pos, row in enumerate(node_rows)
        ],
        dtype=object,
    )

    branch_rows = _rows(raw, "ACBranch")
    branch_cols = _columns(raw, "ACBranch")
    branch_name_col = branch_cols.get("name")
    branch = np.zeros((len(branch_rows), len(BRANCH_COLS)), dtype=np.float64)
    if branch_rows:
        branch_idx_col = branch_cols["idx"]
        branch_i_col = branch_cols["i_node"]
        branch_j_col = branch_cols["j_node"]
        branch_r_col = branch_cols["r"]
        branch_x_col = branch_cols["x"]
        branch_b_col = branch_cols["b"]
        branch_run_col = branch_cols["run_stat"]
        branch_table = _row_array(branch_rows)
        branch_numeric = _numeric_columns(
            branch_table,
            (branch_idx_col, branch_i_col, branch_j_col, branch_r_col, branch_x_col, branch_b_col, branch_run_col),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        branch[:, BRANCH_COLS["idx"]] = branch_numeric[:, 0]
        branch[:, BRANCH_COLS["i_node"]] = branch_numeric[:, 1]
        branch[:, BRANCH_COLS["j_node"]] = branch_numeric[:, 2]
        branch[:, BRANCH_COLS["r"]] = branch_numeric[:, 3]
        branch[:, BRANCH_COLS["x"]] = branch_numeric[:, 4]
        branch[:, BRANCH_COLS["b"]] = branch_numeric[:, 5]
        branch[:, BRANCH_COLS["run_stat"]] = branch_numeric[:, 6]
    branch_names = (
        np.asarray(
            [
                row[branch_name_col] if branch_name_col is not None and row[branch_name_col] else f"branch_{pos}"
                for pos, row in enumerate(branch_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    transformer_rows = _rows(raw, "ACTransformer")
    transformer_cols = _columns(raw, "ACTransformer")
    transformer_name_col = transformer_cols.get("name")
    transformer = np.zeros((len(transformer_rows), len(TRANSFORMER_COLS)), dtype=np.float64)
    if transformer_rows:
        transformer_idx_col = transformer_cols["idx"]
        transformer_i_col = transformer_cols["i_node"]
        transformer_j_col = transformer_cols["j_node"]
        transformer_r_col = transformer_cols["r"]
        transformer_x_col = transformer_cols["x"]
        transformer_b_col = transformer_cols["b"]
        transformer_tap_col = transformer_cols["tap"]
        transformer_shift_col = transformer_cols["shift"]
        transformer_run_col = transformer_cols["run_stat"]
        transformer_table = _row_array(transformer_rows)
        transformer_numeric = _numeric_columns(
            transformer_table,
            (
                transformer_idx_col,
                transformer_i_col,
                transformer_j_col,
                transformer_r_col,
                transformer_x_col,
                transformer_b_col,
                transformer_tap_col,
                transformer_shift_col,
                transformer_run_col,
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        )
        transformer[:, TRANSFORMER_COLS["idx"]] = transformer_numeric[:, 0]
        transformer[:, TRANSFORMER_COLS["i_node"]] = transformer_numeric[:, 1]
        transformer[:, TRANSFORMER_COLS["j_node"]] = transformer_numeric[:, 2]
        transformer[:, TRANSFORMER_COLS["r"]] = transformer_numeric[:, 3]
        transformer[:, TRANSFORMER_COLS["x"]] = transformer_numeric[:, 4]
        transformer[:, TRANSFORMER_COLS["b"]] = transformer_numeric[:, 5]
        transformer[:, TRANSFORMER_COLS["tap"]] = transformer_numeric[:, 6]
        transformer[:, TRANSFORMER_COLS["shift"]] = transformer_numeric[:, 7]
        transformer[:, TRANSFORMER_COLS["run_stat"]] = transformer_numeric[:, 8]
    transformer_names = (
        np.asarray(
            [
                row[transformer_name_col] if transformer_name_col is not None and row[transformer_name_col] else f"transformer_{pos}"
                for pos, row in enumerate(transformer_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    gen_rows = _rows(raw, "ACGenerator")
    gen_cols = _columns(raw, "ACGenerator")
    gen_name_col = gen_cols.get("name")
    gen_alpha_col = gen_cols.get("alpha")
    gen = np.zeros((len(gen_rows), len(GEN_COLS)), dtype=np.float64)
    if gen_rows:
        gen_idx_col = gen_cols["idx"]
        gen_node_col = gen_cols["node"]
        gen_control_col = gen_cols["control_type"]
        gen_p_col = gen_cols["p_set"]
        gen_q_col = gen_cols["q_set"]
        gen_v_col = gen_cols["v_set"]
        gen_run_col = gen_cols["run_stat"]
        gen_table = _row_array(gen_rows)
        gen_nodes = gen_table[:, gen_node_col].astype(np.float64, copy=False)
        gen_raw_vbase = np.fromiter(
            (_node_voltage_base(raw_vbase_by_idx, int(node)) for node in gen_nodes),
            dtype=np.float64,
            count=len(gen_rows),
        )
        gen_v_set_raw = np.where(gen_table[:, gen_v_col] == "", gen_raw_vbase, gen_table[:, gen_v_col]).astype(
            np.float64,
            copy=False,
        )
        inv_p_base = 1.0 / p_base
        gen[:, GEN_COLS["idx"]] = gen_table[:, gen_idx_col].astype(np.float64, copy=False)
        gen[:, GEN_COLS["node"]] = gen_nodes
        gen[:, GEN_COLS["control_type"]] = np.fromiter(
            (CTRL_CODE.get(row[gen_control_col].upper() if row[gen_control_col] else "PQ", CTRL_PQ) for row in gen_rows),
            dtype=np.float64,
            count=len(gen_rows),
        )
        gen[:, GEN_COLS["p_set"]] = gen_table[:, gen_p_col].astype(np.float64, copy=False) * inv_p_base
        gen[:, GEN_COLS["q_set"]] = gen_table[:, gen_q_col].astype(np.float64, copy=False) * inv_p_base
        gen[:, GEN_COLS["v_set"]] = gen_v_set_raw / gen_raw_vbase
        if gen_alpha_col is not None:
            gen[:, GEN_COLS["alpha"]] = np.where(gen_table[:, gen_alpha_col] == "", 1.0, gen_table[:, gen_alpha_col]).astype(
                np.float64,
                copy=False,
            )
        else:
            gen[:, GEN_COLS["alpha"]] = 1.0
        gen[:, GEN_COLS["run_stat"]] = np.where(gen_table[:, gen_run_col] == "", 1.0, gen_table[:, gen_run_col]).astype(
            np.float64,
            copy=False,
        )
    gen_names = (
        np.asarray(
            [
                row[gen_name_col] if gen_name_col is not None and row[gen_name_col] else f"gen_{pos}"
                for pos, row in enumerate(gen_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    load_rows = _rows(raw, "ACLoad")
    load_cols = _columns(raw, "ACLoad")
    load_name_col = load_cols.get("name")
    load = np.zeros((len(load_rows), len(LOAD_COLS)), dtype=np.float64)
    if load_rows:
        load_idx_col = load_cols["idx"]
        load_node_col = load_cols["node"]
        load_run_col = load_cols["run_stat"]
        load_value_cols = tuple(load_cols.get(key) for key in ("pbase", "pv0", "pv1", "pv2", "qbase", "qv0", "qv1", "qv2"))
        pbase_col, pv0_col, pv1_col, pv2_col, qbase_col, qv0_col, qv1_col, qv2_col = load_value_cols
        load[:, LOAD_COLS["idx"]] = _float_column(load_rows, load_idx_col)
        load[:, LOAD_COLS["node"]] = _float_column(load_rows, load_node_col)
        inv_p_base = 1.0 / p_base
        load[:, LOAD_COLS["pbase"]] = (
            _float_column(load_rows, pbase_col, 1.0) if pbase_col is not None else np.ones(len(load_rows))
        ) * inv_p_base
        load[:, LOAD_COLS["pv0"]] = _float_column(load_rows, pv0_col) if pv0_col is not None else 0.0
        load[:, LOAD_COLS["pv1"]] = _float_column(load_rows, pv1_col) if pv1_col is not None else 0.0
        load[:, LOAD_COLS["pv2"]] = _float_column(load_rows, pv2_col) if pv2_col is not None else 0.0
        load[:, LOAD_COLS["qbase"]] = (
            _float_column(load_rows, qbase_col, 1.0) if qbase_col is not None else np.ones(len(load_rows))
        ) * inv_p_base
        load[:, LOAD_COLS["qv0"]] = _float_column(load_rows, qv0_col) if qv0_col is not None else 0.0
        load[:, LOAD_COLS["qv1"]] = _float_column(load_rows, qv1_col) if qv1_col is not None else 0.0
        load[:, LOAD_COLS["qv2"]] = _float_column(load_rows, qv2_col) if qv2_col is not None else 0.0
        load[:, LOAD_COLS["run_stat"]] = _float_column(load_rows, load_run_col, 1.0)
    load_names = (
        np.asarray(
            [
                row[load_name_col] if load_name_col is not None and row[load_name_col] else f"load_{pos}"
                for pos, row in enumerate(load_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    shunt_rows = _rows(raw, "ACShuntCompensator")
    shunt_cols = _columns(raw, "ACShuntCompensator")
    shunt_name_col = shunt_cols.get("name")
    shunt = np.zeros((len(shunt_rows), len(SHUNT_COLS)), dtype=np.float64)
    if shunt_rows:
        shunt_idx_col = shunt_cols["idx"]
        shunt_node_col = shunt_cols["node"]
        shunt_control_col = shunt_cols["control_type"]
        shunt_q_col = shunt_cols["q_set"]
        shunt_g_col = shunt_cols["g_set"]
        shunt_b_col = shunt_cols["b_set"]
        shunt_v_col = shunt_cols["v_set"]
        shunt_run_col = shunt_cols["run_stat"]
        shunt_table = _row_array(shunt_rows)
        shunt_nodes = shunt_table[:, shunt_node_col].astype(np.float64, copy=False)
        shunt_raw_vbase = np.fromiter(
            (_node_voltage_base(raw_vbase_by_idx, int(node)) for node in shunt_nodes),
            dtype=np.float64,
            count=len(shunt_rows),
        )
        shunt_v_set_raw = np.where(shunt_table[:, shunt_v_col] == "", shunt_raw_vbase, shunt_table[:, shunt_v_col]).astype(
            dtype=np.float64,
            copy=False,
        )
        shunt[:, SHUNT_COLS["idx"]] = shunt_table[:, shunt_idx_col].astype(np.float64, copy=False)
        shunt[:, SHUNT_COLS["node"]] = shunt_nodes
        shunt[:, SHUNT_COLS["control_type"]] = np.fromiter(
            (SHUNT_CODE.get(row[shunt_control_col].upper() if row[shunt_control_col] else "Q", SHUNT_Q) for row in shunt_rows),
            dtype=np.float64,
            count=len(shunt_rows),
        )
        shunt[:, SHUNT_COLS["q_set"]] = shunt_table[:, shunt_q_col].astype(np.float64, copy=False) / p_base
        shunt[:, SHUNT_COLS["g_set"]] = shunt_table[:, shunt_g_col].astype(np.float64, copy=False)
        shunt[:, SHUNT_COLS["b_set"]] = shunt_table[:, shunt_b_col].astype(np.float64, copy=False)
        shunt[:, SHUNT_COLS["v_set"]] = shunt_v_set_raw / shunt_raw_vbase
        shunt[:, SHUNT_COLS["run_stat"]] = np.where(shunt_table[:, shunt_run_col] == "", 1.0, shunt_table[:, shunt_run_col]).astype(
            np.float64,
            copy=False,
        )
    shunt_names = (
        np.asarray(
            [
                row[shunt_name_col] if shunt_name_col is not None and row[shunt_name_col] else f"shunt_{pos}"
                for pos, row in enumerate(shunt_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    zero_rows = _rows(raw, "ACZeroBranch")
    zero_cols = _columns(raw, "ACZeroBranch")
    zero_name_col = zero_cols.get("name")
    zero_branch = np.zeros((len(zero_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    if zero_rows:
        zero_idx_col = zero_cols["idx"]
        zero_i_col = zero_cols["i_node"]
        zero_j_col = zero_cols["j_node"]
        zero_run_col = zero_cols["run_stat"]
        zero_table = _row_array(zero_rows)
        zero_numeric = _numeric_columns(
            zero_table,
            (zero_idx_col, zero_i_col, zero_j_col, zero_run_col),
            (0.0, 0.0, 0.0, 1.0),
        )
        zero_branch[:, ZERO_BRANCH_COLS["idx"]] = zero_numeric[:, 0]
        zero_branch[:, ZERO_BRANCH_COLS["i_node"]] = zero_numeric[:, 1]
        zero_branch[:, ZERO_BRANCH_COLS["j_node"]] = zero_numeric[:, 2]
        zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] = zero_numeric[:, 3]
    zero_branch_names = (
        np.asarray(
            [
                row[zero_name_col] if zero_name_col is not None and row[zero_name_col] else f"zero_branch_{pos}"
                for pos, row in enumerate(zero_rows)
            ],
            dtype=object,
        )
        if include_device_names
        else None
    )

    def build_switch_like(block: str):
        rows = _rows(raw, block)
        cols = _columns(raw, block)
        name_col = cols.get("name")
        array = np.zeros((len(rows), len(SWITCH_COLS)), dtype=np.float64)
        if rows:
            table = _row_array(rows)
            numeric = _numeric_columns(
                table,
                (cols["idx"], cols["i_node"], cols["j_node"], cols["status"], cols["run_stat"]),
                (0.0, 0.0, 0.0, 1.0, 1.0),
            )
            array[:, SWITCH_COLS["idx"]] = numeric[:, 0]
            array[:, SWITCH_COLS["i_node"]] = numeric[:, 1]
            array[:, SWITCH_COLS["j_node"]] = numeric[:, 2]
            array[:, SWITCH_COLS["status"]] = numeric[:, 3]
            array[:, SWITCH_COLS["run_stat"]] = numeric[:, 4]
        names = (
            np.asarray(
                [
                    row[name_col] if name_col is not None and row[name_col] else f"{block.lower()}_{pos}"
                    for pos, row in enumerate(rows)
                ],
                dtype=object,
            )
            if include_device_names
            else None
        )
        return rows, cols, array, names

    switch_rows, switch_cols, switch, switch_names = build_switch_like("ACSwitch")
    break_rows, break_cols, break_array, break_names = build_switch_like("ACBreak")

    ppc = {
        "format": "ac_ppc_v1",
        "source": str(file_key[0]),
        "base": np.asarray([p_base, u_scale, p_scale, i_scale, p_base / p_scale], dtype=np.float64),
        "bus": bus,
        "branch": branch,
        "transformer": transformer,
        "gen": gen,
        "load": load,
        "shunt": shunt,
        "zero_branch": zero_branch,
        "switch": switch,
        "break": break_array,
        "bus_name": bus_names,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
    }
    if include_device_names:
        ppc.update(
            {
                "branch_name": branch_names,
                "transformer_name": transformer_names,
                "gen_name": gen_names,
                "load_name": load_names,
                "shunt_name": shunt_names,
                "zero_branch_name": zero_branch_names,
                "switch_name": switch_names,
                "break_name": break_names,
            }
        )
    if use_cache:
        with _AC_PPC_CACHE_LOCK:
            _AC_PPC_CACHE[(file_key[0], include_device_names)] = (file_key, ppc)
    return _copy_ppc(ppc) if copy_arrays else ppc


def _copy_ppc(ppc: Dict) -> Dict:
    copied = {}
    for key, value in ppc.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied
