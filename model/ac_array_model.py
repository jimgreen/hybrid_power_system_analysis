import math
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from efile_read import read_efile_dict_cached


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
    "pv0": 2,
    "pv1": 3,
    "pv2": 4,
    "qv0": 5,
    "qv1": 6,
    "qv2": 7,
    "run_stat": 8,
    "p": 9,
    "q": 10,
    "current": 11,
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

_AC_PPC_CACHE = {}
_AC_PPC_CACHE_LOCK = threading.Lock()


def _rows(raw: Dict, block: str) -> List[Dict]:
    return raw.get(block, {"data": []})["data"]


def _float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value in (None, "") else float(value)


def _int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
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
            _AC_PPC_CACHE.pop(Path(file_path).resolve(), None)


def _node_row_maps(bus: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    return {int(node_id): pos for pos, node_id in enumerate(node_ids)}, node_ids


def _node_voltage_base(raw_vbase_by_idx: Dict[int, float], node_idx: int) -> float:
    return raw_vbase_by_idx[int(node_idx)]


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def build_ac_ppc_from_e_file(file_path, use_cache: bool = True, copy_arrays: bool = False) -> Dict:
    """Build a MATPOWER-like NumPy dictionary for AC power-flow fast paths.

    The returned arrays are in pu/radians and should be treated as read-only by
    callers. Set copy_arrays=True when a caller needs to mutate the input model.
    """
    file_key = _file_cache_key(file_path)
    if use_cache:
        with _AC_PPC_CACHE_LOCK:
            cached = _AC_PPC_CACHE.get(file_key[0])
            if cached is not None and cached[0] == file_key:
                return _copy_ppc(cached[1]) if copy_arrays else cached[1]

    raw = read_efile_dict_cached(file_key[0], use_cache=use_cache)
    base_row = _rows(raw, "PowerBase")[0]
    p_base = _float(base_row, "p_base")
    u_scale = _float(base_row, "u_scale")
    p_scale = _float(base_row, "p_scale")
    i_scale = _float(base_row, "i_scale")

    node_rows = _rows(raw, "ACNode")
    bus = np.zeros((len(node_rows), len(BUS_COLS)), dtype=np.float64)
    bus_names = np.empty(len(node_rows), dtype=object)
    raw_vbase_by_idx: Dict[int, float] = {}
    for pos, row in enumerate(node_rows):
        idx = _int(row, "idx")
        raw_vbase = _float(row, "vbase")
        raw_vbase_by_idx[idx] = raw_vbase
        bus[pos, BUS_COLS["idx"]] = idx
        bus[pos, BUS_COLS["vbase"]] = raw_vbase / u_scale
        bus[pos, BUS_COLS["voltage"]] = _float(row, "voltage", raw_vbase) / raw_vbase
        bus[pos, BUS_COLS["angle"]] = math.radians(_float(row, "angle", 0.0))
        bus[pos, BUS_COLS["isl"]] = _float(row, "isl", 0.0)
        bus[pos, BUS_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        bus_names[pos] = row.get("name", f"bus_{idx}")

    def node_vset(row: Dict, key: str, node_key: str = "node") -> float:
        node_idx = _int(row, node_key)
        raw_vbase = _node_voltage_base(raw_vbase_by_idx, node_idx)
        return _float(row, key, raw_vbase) / raw_vbase

    branch_rows = _rows(raw, "ACBranch")
    branch = np.zeros((len(branch_rows), len(BRANCH_COLS)), dtype=np.float64)
    branch_names = np.empty(len(branch_rows), dtype=object)
    for pos, row in enumerate(branch_rows):
        branch[pos, BRANCH_COLS["idx"]] = _int(row, "idx")
        branch[pos, BRANCH_COLS["i_node"]] = _int(row, "i_node")
        branch[pos, BRANCH_COLS["j_node"]] = _int(row, "j_node")
        branch[pos, BRANCH_COLS["r"]] = _float(row, "r")
        branch[pos, BRANCH_COLS["x"]] = _float(row, "x")
        branch[pos, BRANCH_COLS["b"]] = _float(row, "b")
        branch[pos, BRANCH_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        branch_names[pos] = row.get("name", f"branch_{pos}")

    transformer_rows = _rows(raw, "ACTransformer")
    transformer = np.zeros((len(transformer_rows), len(TRANSFORMER_COLS)), dtype=np.float64)
    transformer_names = np.empty(len(transformer_rows), dtype=object)
    for pos, row in enumerate(transformer_rows):
        transformer[pos, TRANSFORMER_COLS["idx"]] = _int(row, "idx")
        transformer[pos, TRANSFORMER_COLS["i_node"]] = _int(row, "i_node")
        transformer[pos, TRANSFORMER_COLS["j_node"]] = _int(row, "j_node")
        transformer[pos, TRANSFORMER_COLS["r"]] = _float(row, "r")
        transformer[pos, TRANSFORMER_COLS["x"]] = _float(row, "x")
        transformer[pos, TRANSFORMER_COLS["b"]] = _float(row, "b")
        transformer[pos, TRANSFORMER_COLS["tap"]] = _float(row, "tap", 1.0)
        transformer[pos, TRANSFORMER_COLS["shift"]] = _float(row, "shift", 0.0)
        transformer[pos, TRANSFORMER_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        transformer_names[pos] = row.get("name", f"transformer_{pos}")

    gen_rows = _rows(raw, "ACGenerator")
    gen = np.zeros((len(gen_rows), len(GEN_COLS)), dtype=np.float64)
    gen_names = np.empty(len(gen_rows), dtype=object)
    for pos, row in enumerate(gen_rows):
        gen[pos, GEN_COLS["idx"]] = _int(row, "idx")
        gen[pos, GEN_COLS["node"]] = _int(row, "node")
        gen[pos, GEN_COLS["control_type"]] = CTRL_CODE.get(str(row.get("control_type", "PQ")).upper(), CTRL_PQ)
        gen[pos, GEN_COLS["p_set"]] = _float(row, "p_set") / p_base
        gen[pos, GEN_COLS["q_set"]] = _float(row, "q_set") / p_base
        gen[pos, GEN_COLS["v_set"]] = node_vset(row, "v_set")
        gen[pos, GEN_COLS["alpha"]] = _float(row, "alpha", 1.0)
        gen[pos, GEN_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        gen_names[pos] = row.get("name", f"gen_{pos}")

    load_rows = _rows(raw, "ACLoad")
    load = np.zeros((len(load_rows), len(LOAD_COLS)), dtype=np.float64)
    load_names = np.empty(len(load_rows), dtype=object)
    for pos, row in enumerate(load_rows):
        load[pos, LOAD_COLS["idx"]] = _int(row, "idx")
        load[pos, LOAD_COLS["node"]] = _int(row, "node")
        for key in ("pv0", "pv1", "pv2", "qv0", "qv1", "qv2"):
            load[pos, LOAD_COLS[key]] = _float(row, key) / p_base
        load[pos, LOAD_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        load_names[pos] = row.get("name", f"load_{pos}")

    shunt_rows = _rows(raw, "ACShuntCompensator")
    shunt = np.zeros((len(shunt_rows), len(SHUNT_COLS)), dtype=np.float64)
    shunt_names = np.empty(len(shunt_rows), dtype=object)
    for pos, row in enumerate(shunt_rows):
        shunt[pos, SHUNT_COLS["idx"]] = _int(row, "idx")
        shunt[pos, SHUNT_COLS["node"]] = _int(row, "node")
        shunt[pos, SHUNT_COLS["control_type"]] = SHUNT_CODE.get(str(row.get("control_type", "Q")).upper(), SHUNT_Q)
        shunt[pos, SHUNT_COLS["q_set"]] = _float(row, "q_set") / p_base
        shunt[pos, SHUNT_COLS["g_set"]] = _float(row, "g_set")
        shunt[pos, SHUNT_COLS["b_set"]] = _float(row, "b_set")
        shunt[pos, SHUNT_COLS["v_set"]] = node_vset(row, "v_set")
        shunt[pos, SHUNT_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        shunt_names[pos] = row.get("name", f"shunt_{pos}")

    zero_rows = _rows(raw, "ACZeroBranch")
    zero_branch = np.zeros((len(zero_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    zero_branch_names = np.empty(len(zero_rows), dtype=object)
    for pos, row in enumerate(zero_rows):
        zero_branch[pos, ZERO_BRANCH_COLS["idx"]] = _int(row, "idx")
        zero_branch[pos, ZERO_BRANCH_COLS["i_node"]] = _int(row, "i_node")
        zero_branch[pos, ZERO_BRANCH_COLS["j_node"]] = _int(row, "j_node")
        zero_branch[pos, ZERO_BRANCH_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        zero_branch_names[pos] = row.get("name", f"zero_branch_{pos}")

    switch_rows = _rows(raw, "ACSwitch")
    switch = np.zeros((len(switch_rows), len(SWITCH_COLS)), dtype=np.float64)
    switch_names = np.empty(len(switch_rows), dtype=object)
    for pos, row in enumerate(switch_rows):
        switch[pos, SWITCH_COLS["idx"]] = _int(row, "idx")
        switch[pos, SWITCH_COLS["i_node"]] = _int(row, "i_node")
        switch[pos, SWITCH_COLS["j_node"]] = _int(row, "j_node")
        switch[pos, SWITCH_COLS["status"]] = _float(row, "status", 1.0)
        switch[pos, SWITCH_COLS["run_stat"]] = _float(row, "run_stat", 1.0)
        switch_names[pos] = row.get("name", f"switch_{pos}")

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
        "bus_name": bus_names,
        "branch_name": branch_names,
        "transformer_name": transformer_names,
        "gen_name": gen_names,
        "load_name": load_names,
        "shunt_name": shunt_names,
        "zero_branch_name": zero_branch_names,
        "switch_name": switch_names,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
    }
    if use_cache:
        with _AC_PPC_CACHE_LOCK:
            _AC_PPC_CACHE[file_key[0]] = (file_key, ppc)
    return _copy_ppc(ppc) if copy_arrays else ppc


def _copy_ppc(ppc: Dict) -> Dict:
    copied = {}
    for key, value in ppc.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied
