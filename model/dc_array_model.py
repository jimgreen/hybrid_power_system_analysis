import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from efile_read import read_efile_rows_cached
from unit_system import dc_current_base_ka


CTRL_P = 0
CTRL_V = 1
CTRL_I = 2
CTRL_CODE = {
    "P": CTRL_P,
    "V": CTRL_V,
    "I": CTRL_I,
}

BUS_COLS = {
    "idx": 0,
    "vbase": 1,
    "voltage": 2,
    "isl": 3,
    "run_stat": 4,
}
BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "run_stat": 4,
    "i_p": 5,
    "j_p": 6,
    "current": 7,
}
LOAD_COLS = {
    "idx": 0,
    "node": 1,
    "pv0": 2,
    "pv1": 3,
    "pv2": 4,
    "run_stat": 5,
    "p": 6,
    "current": 7,
}
GEN_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "p_set": 3,
    "v_set": 4,
    "i_set": 5,
    "run_stat": 6,
    "p": 7,
    "current": 8,
}
ZERO_BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "run_stat": 3,
    "p": 4,
    "current": 5,
}
SWITCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "status": 3,
    "run_stat": 4,
    "p": 5,
    "current": 6,
}
DCDC_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_set": 6,
    "i_set": 7,
    "v_set": 8,
    "run_stat": 9,
    "i_p": 10,
    "j_p": 11,
    "i_c": 12,
    "j_c": 13,
}

_DC_PPC_CACHE = {}
_DC_PPC_CACHE_LOCK = threading.Lock()


def _table(raw: Dict, block: str) -> Dict:
    return raw.get(block, {"header_list": [], "rows": []})


def _rows(raw: Dict, block: str) -> List[List[str]]:
    return _table(raw, block)["rows"]


def _columns(raw: Dict, block: str) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(_table(raw, block)["header_list"])}


def _cell(row: List[str], cols: Dict[str, int], key: str, default=""):
    idx = cols.get(key)
    if idx is None or idx >= len(row):
        return default
    value = row[idx]
    return default if value == "" else value


def _float_cell(row: List[str], cols: Dict[str, int], key: str, default: float = 0.0) -> float:
    value = _cell(row, cols, key, default)
    return default if value in (None, "") else float(value)


def _int_cell(row: List[str], cols: Dict[str, int], key: str, default: int = 0) -> int:
    value = _cell(row, cols, key, default)
    return default if value in (None, "") else int(float(value))


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = Path(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def clear_dc_ppc_cache(file_path=None) -> None:
    with _DC_PPC_CACHE_LOCK:
        if file_path is None:
            _DC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _DC_PPC_CACHE.pop((path, True), None)
            _DC_PPC_CACHE.pop((path, False), None)


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _copy_ppc(ppc: Dict) -> Dict:
    copied = {}
    for key, value in ppc.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _node_maps(bus: np.ndarray):
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    return {int(node_id): pos for pos, node_id in enumerate(node_ids)}


def _current_scale(power_base_kw: float, i_scale: float, raw_vbase_by_idx: Dict[int, float], node_idx: int) -> float:
    return i_scale * dc_current_base_ka(power_base_kw, raw_vbase_by_idx[int(node_idx)])


def build_dc_ppc_from_e_file(
    file_path,
    use_cache: bool = True,
    copy_arrays: bool = False,
    include_device_names: bool = True,
) -> Dict:
    """Build a NumPy dictionary for DC power-flow fast paths.

    The returned arrays use the same normalized units as DCPowerNetwork after
    normalize_model_named_units(). Treat cached arrays as read-only unless
    copy_arrays=True is requested.
    """
    file_key = _file_cache_key(file_path)
    if use_cache:
        with _DC_PPC_CACHE_LOCK:
            cached = _DC_PPC_CACHE.get((file_key[0], include_device_names))
            if cached is not None and cached[0] == file_key:
                return _copy_ppc(cached[1]) if copy_arrays else cached[1]

    raw = read_efile_rows_cached(file_key[0], use_cache=use_cache)
    base_cols = _columns(raw, "PowerBase")
    base_row = _rows(raw, "PowerBase")[0]
    p_base = _float_cell(base_row, base_cols, "p_base")
    u_scale = _float_cell(base_row, base_cols, "u_scale")
    p_scale = _float_cell(base_row, base_cols, "p_scale")
    i_scale = _float_cell(base_row, base_cols, "i_scale")
    power_base_kw = p_base / p_scale

    node_rows = _rows(raw, "DCNode")
    node_cols = _columns(raw, "DCNode")
    bus = np.zeros((len(node_rows), len(BUS_COLS)), dtype=np.float64)
    raw_vbase_by_idx = {}
    if node_rows:
        for out_row, row in enumerate(node_rows):
            idx = _int_cell(row, node_cols, "idx")
            raw_vbase = _float_cell(row, node_cols, "vbase")
            raw_voltage = _float_cell(row, node_cols, "voltage", raw_vbase)
            bus[out_row, BUS_COLS["idx"]] = idx
            bus[out_row, BUS_COLS["vbase"]] = raw_vbase / u_scale
            bus[out_row, BUS_COLS["voltage"]] = raw_voltage / (u_scale * raw_vbase)
            bus[out_row, BUS_COLS["isl"]] = _float_cell(row, node_cols, "isl", 0.0)
            bus[out_row, BUS_COLS["run_stat"]] = _float_cell(row, node_cols, "run_stat", 1.0)
            raw_vbase_by_idx[idx] = raw_vbase
    node_names = np.asarray(
        [
            _cell(row, node_cols, "name", f"bus_{int(bus[pos, BUS_COLS['idx']])}")
            for pos, row in enumerate(node_rows)
        ],
        dtype=object,
    )
    node_pos = _node_maps(bus) if len(bus) else {}

    branch_rows = _rows(raw, "DCBranch")
    branch_cols = _columns(raw, "DCBranch")
    branch = np.zeros((len(branch_rows), len(BRANCH_COLS)), dtype=np.float64)
    for out_row, row in enumerate(branch_rows):
        i_node = _int_cell(row, branch_cols, "i_node")
        branch[out_row, BRANCH_COLS["idx"]] = _int_cell(row, branch_cols, "idx")
        branch[out_row, BRANCH_COLS["i_node"]] = i_node
        branch[out_row, BRANCH_COLS["j_node"]] = _int_cell(row, branch_cols, "j_node")
        branch[out_row, BRANCH_COLS["r"]] = _float_cell(row, branch_cols, "r")
        branch[out_row, BRANCH_COLS["run_stat"]] = _float_cell(row, branch_cols, "run_stat", 1.0)
        branch[out_row, BRANCH_COLS["i_p"]] = _float_cell(row, branch_cols, "i_p") / p_base
        branch[out_row, BRANCH_COLS["j_p"]] = _float_cell(row, branch_cols, "j_p") / p_base
        branch[out_row, BRANCH_COLS["current"]] = _float_cell(row, branch_cols, "current") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, i_node
        )

    load_rows = _rows(raw, "DCLoad")
    load_cols = _columns(raw, "DCLoad")
    load = np.zeros((len(load_rows), len(LOAD_COLS)), dtype=np.float64)
    for out_row, row in enumerate(load_rows):
        node = _int_cell(row, load_cols, "node")
        load[out_row, LOAD_COLS["idx"]] = _int_cell(row, load_cols, "idx")
        load[out_row, LOAD_COLS["node"]] = node
        load[out_row, LOAD_COLS["pv0"]] = _float_cell(row, load_cols, "pv0") / p_base
        load[out_row, LOAD_COLS["pv1"]] = _float_cell(row, load_cols, "pv1") / p_base
        load[out_row, LOAD_COLS["pv2"]] = _float_cell(row, load_cols, "pv2") / p_base
        load[out_row, LOAD_COLS["run_stat"]] = _float_cell(row, load_cols, "run_stat", 1.0)
        load[out_row, LOAD_COLS["p"]] = _float_cell(row, load_cols, "p") / p_base
        load[out_row, LOAD_COLS["current"]] = _float_cell(row, load_cols, "current") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, node
        )

    gen_rows = _rows(raw, "DCGenerator")
    gen_cols = _columns(raw, "DCGenerator")
    gen = np.zeros((len(gen_rows), len(GEN_COLS)), dtype=np.float64)
    for out_row, row in enumerate(gen_rows):
        node = _int_cell(row, gen_cols, "node")
        gen[out_row, GEN_COLS["idx"]] = _int_cell(row, gen_cols, "idx")
        gen[out_row, GEN_COLS["node"]] = node
        gen[out_row, GEN_COLS["control_type"]] = CTRL_CODE[str(_cell(row, gen_cols, "control_type", "P"))]
        gen[out_row, GEN_COLS["p_set"]] = _float_cell(row, gen_cols, "p_set") / p_base
        gen[out_row, GEN_COLS["v_set"]] = _float_cell(row, gen_cols, "v_set") / (
            u_scale * raw_vbase_by_idx[node]
        )
        gen[out_row, GEN_COLS["i_set"]] = _float_cell(row, gen_cols, "i_set") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, node
        )
        gen[out_row, GEN_COLS["run_stat"]] = _float_cell(row, gen_cols, "run_stat", 1.0)
        gen[out_row, GEN_COLS["p"]] = _float_cell(row, gen_cols, "p") / p_base
        gen[out_row, GEN_COLS["current"]] = _float_cell(row, gen_cols, "current") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, node
        )

    zero_rows = _rows(raw, "DCZeroBranch")
    zero_cols = _columns(raw, "DCZeroBranch")
    zero_branch = np.zeros((len(zero_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    for out_row, row in enumerate(zero_rows):
        i_node = _int_cell(row, zero_cols, "i_node")
        zero_branch[out_row, ZERO_BRANCH_COLS["idx"]] = _int_cell(row, zero_cols, "idx")
        zero_branch[out_row, ZERO_BRANCH_COLS["i_node"]] = i_node
        zero_branch[out_row, ZERO_BRANCH_COLS["j_node"]] = _int_cell(row, zero_cols, "j_node")
        zero_branch[out_row, ZERO_BRANCH_COLS["run_stat"]] = _float_cell(row, zero_cols, "run_stat", 1.0)
        zero_branch[out_row, ZERO_BRANCH_COLS["p"]] = _float_cell(row, zero_cols, "p") / p_base
        zero_branch[out_row, ZERO_BRANCH_COLS["current"]] = _float_cell(row, zero_cols, "current") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, i_node
        )

    switch_rows = _rows(raw, "DCSwitch")
    switch_cols = _columns(raw, "DCSwitch")
    switch = np.zeros((len(switch_rows), len(SWITCH_COLS)), dtype=np.float64)
    for out_row, row in enumerate(switch_rows):
        i_node = _int_cell(row, switch_cols, "i_node")
        switch[out_row, SWITCH_COLS["idx"]] = _int_cell(row, switch_cols, "idx")
        switch[out_row, SWITCH_COLS["i_node"]] = i_node
        switch[out_row, SWITCH_COLS["j_node"]] = _int_cell(row, switch_cols, "j_node")
        switch[out_row, SWITCH_COLS["status"]] = _float_cell(row, switch_cols, "status", 1.0)
        switch[out_row, SWITCH_COLS["run_stat"]] = _float_cell(row, switch_cols, "run_stat", 1.0)
        switch[out_row, SWITCH_COLS["p"]] = _float_cell(row, switch_cols, "p") / p_base
        switch[out_row, SWITCH_COLS["current"]] = _float_cell(row, switch_cols, "current") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, i_node
        )

    dcdc_rows = _rows(raw, "DCDCConverter")
    dcdc_cols = _columns(raw, "DCDCConverter")
    dcdc = np.zeros((len(dcdc_rows), len(DCDC_COLS)), dtype=np.float64)
    for out_row, row in enumerate(dcdc_rows):
        i_node = _int_cell(row, dcdc_cols, "i_node")
        j_node = _int_cell(row, dcdc_cols, "j_node")
        dcdc[out_row, DCDC_COLS["idx"]] = _int_cell(row, dcdc_cols, "idx")
        dcdc[out_row, DCDC_COLS["i_node"]] = i_node
        dcdc[out_row, DCDC_COLS["j_node"]] = j_node
        dcdc[out_row, DCDC_COLS["r1"]] = _float_cell(row, dcdc_cols, "r1")
        dcdc[out_row, DCDC_COLS["r2"]] = _float_cell(row, dcdc_cols, "r2")
        dcdc[out_row, DCDC_COLS["control_type"]] = CTRL_CODE[str(_cell(row, dcdc_cols, "control_type", "P"))]
        dcdc[out_row, DCDC_COLS["p_set"]] = _float_cell(row, dcdc_cols, "p_set") / p_base
        dcdc[out_row, DCDC_COLS["i_set"]] = _float_cell(row, dcdc_cols, "i_set") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, i_node
        )
        dcdc[out_row, DCDC_COLS["v_set"]] = _float_cell(row, dcdc_cols, "v_set") / (
            u_scale * raw_vbase_by_idx[i_node]
        )
        dcdc[out_row, DCDC_COLS["run_stat"]] = _float_cell(row, dcdc_cols, "run_stat", 1.0)
        dcdc[out_row, DCDC_COLS["i_p"]] = _float_cell(row, dcdc_cols, "i_p") / p_base
        dcdc[out_row, DCDC_COLS["j_p"]] = _float_cell(row, dcdc_cols, "j_p") / p_base
        dcdc[out_row, DCDC_COLS["i_c"]] = _float_cell(row, dcdc_cols, "i_c") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, i_node
        )
        dcdc[out_row, DCDC_COLS["j_c"]] = _float_cell(row, dcdc_cols, "j_c") / _current_scale(
            power_base_kw, i_scale, raw_vbase_by_idx, j_node
        )

    ppc = {
        "format": "dc_ppc_v1",
        "base": {
            "p_base": p_base,
            "u_scale": u_scale,
            "p_scale": p_scale,
            "i_scale": i_scale,
            "p_base_kW": power_base_kw,
        },
        "bus": bus,
        "branch": branch if len(branch_rows) else _empty(len(BRANCH_COLS)),
        "load": load if len(load_rows) else _empty(len(LOAD_COLS)),
        "gen": gen if len(gen_rows) else _empty(len(GEN_COLS)),
        "zero_branch": zero_branch if len(zero_rows) else _empty(len(ZERO_BRANCH_COLS)),
        "switch": switch if len(switch_rows) else _empty(len(SWITCH_COLS)),
        "dcdc": dcdc if len(dcdc_rows) else _empty(len(DCDC_COLS)),
        "node_pos": node_pos,
    }
    if include_device_names:
        ppc.update(
            bus_name=node_names,
            branch_name=np.asarray([_cell(row, branch_cols, "name", "") for row in branch_rows], dtype=object),
            load_name=np.asarray([_cell(row, load_cols, "name", "") for row in load_rows], dtype=object),
            gen_name=np.asarray([_cell(row, gen_cols, "name", "") for row in gen_rows], dtype=object),
            zero_branch_name=np.asarray([_cell(row, zero_cols, "name", "") for row in zero_rows], dtype=object),
            switch_name=np.asarray([_cell(row, switch_cols, "name", "") for row in switch_rows], dtype=object),
            dcdc_name=np.asarray([_cell(row, dcdc_cols, "name", "") for row in dcdc_rows], dtype=object),
        )

    if use_cache:
        with _DC_PPC_CACHE_LOCK:
            _DC_PPC_CACHE[(file_key[0], include_device_names)] = (file_key, ppc)
    return _copy_ppc(ppc) if copy_arrays else ppc
