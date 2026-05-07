import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

try:
    from .ac_array_model import (
        BRANCH_COLS as AC_BRANCH_COLS,
        BUS_COLS as AC_BUS_COLS,
        GEN_COLS as AC_GEN_COLS,
        LOAD_COLS as AC_LOAD_COLS,
        BREAK_COLS as AC_BREAK_COLS,
        SHUNT_COLS as AC_SHUNT_COLS,
        SWITCH_COLS as AC_SWITCH_COLS,
        TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
        ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
        CTRL_P,
        CTRL_PQ,
        CTRL_PV,
        CTRL_SLACK,
        SHUNT_B,
        SHUNT_Q,
        SHUNT_V,
        SHUNT_Z,
        build_ac_ppc_from_e_file,
    )
    from .ac_model import (
        ACPowerNetwork,
        ACBranch,
        ACGenerator,
        ACLoad,
        ACNode,
        ACShuntCompensator,
        ACSwitch,
        ACBreak,
        ACTransformer,
        ACZeroBranch,
    )
    from .dc_array_model import (
        BRANCH_COLS as DC_BRANCH_COLS,
        BUS_COLS as DC_BUS_COLS,
        DCDC_COLS as DC_DCDC_COLS,
        GEN_COLS as DC_GEN_COLS,
        LOAD_COLS as DC_LOAD_COLS,
        BREAK_COLS as DC_BREAK_COLS,
        SWITCH_COLS as DC_SWITCH_COLS,
        ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
        DCPowerNetwork,
        build_dc_ppc_from_e_file,
    )
    from .efile_read import read_efile_rows_cached
    from .unit_system import ac_current_base_ka, dc_current_base_ka
except ImportError:
    from ac_array_model import (
        BRANCH_COLS as AC_BRANCH_COLS,
        BUS_COLS as AC_BUS_COLS,
        GEN_COLS as AC_GEN_COLS,
        LOAD_COLS as AC_LOAD_COLS,
        BREAK_COLS as AC_BREAK_COLS,
        SHUNT_COLS as AC_SHUNT_COLS,
        SWITCH_COLS as AC_SWITCH_COLS,
        TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
        ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
        CTRL_P,
        CTRL_PQ,
        CTRL_PV,
        CTRL_SLACK,
        SHUNT_B,
        SHUNT_Q,
        SHUNT_V,
        SHUNT_Z,
        build_ac_ppc_from_e_file,
    )
    from ac_model import (
        ACPowerNetwork,
        ACBranch,
        ACGenerator,
        ACLoad,
        ACNode,
        ACShuntCompensator,
        ACSwitch,
        ACBreak,
        ACTransformer,
        ACZeroBranch,
    )
    from dc_array_model import (
        BRANCH_COLS as DC_BRANCH_COLS,
        BUS_COLS as DC_BUS_COLS,
        DCDC_COLS as DC_DCDC_COLS,
        GEN_COLS as DC_GEN_COLS,
        LOAD_COLS as DC_LOAD_COLS,
        BREAK_COLS as DC_BREAK_COLS,
        SWITCH_COLS as DC_SWITCH_COLS,
        ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
        DCPowerNetwork,
        build_dc_ppc_from_e_file,
    )
    from efile_read import read_efile_rows_cached
    from unit_system import ac_current_base_ka, dc_current_base_ka


DCAC_CONTROL_CODE = {"DCV": 0, "ACV": 1, "ACP": 2}
DCAC_CONTROL_LABEL = {value: key for key, value in DCAC_CONTROL_CODE.items()}
ACAC_CONTROL_CODE = {"PQQ": 0, "PVQ": 1, "PQV": 2, "PVV": 3}
ACAC_CONTROL_LABEL = {value: key for key, value in ACAC_CONTROL_CODE.items()}

DCAC_COLS = {
    "idx": 0,
    "ac_node": 1,
    "dc_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_ac_set": 6,
    "q_ac_set": 7,
    "v_ac_set": 8,
    "v_dc_set": 9,
    "run_stat": 10,
    "dc_p": 11,
    "ac_p": 12,
    "ac_q": 13,
    "dc_i": 14,
    "ac_i": 15,
}
ACAC_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_set": 6,
    "i_q_set": 7,
    "j_q_set": 8,
    "i_v_set": 9,
    "j_v_set": 10,
    "run_stat": 11,
    "i_p": 12,
    "i_q": 13,
    "j_p": 14,
    "j_q": 15,
    "i_i": 16,
    "j_i": 17,
}

_HYBRID_PPC_CACHE = {}
_HYBRID_PPC_CACHE_LOCK = threading.Lock()


class _ArrayDevice(SimpleNamespace):
    __hash__ = object.__hash__


def _device(idx, name, **values):
    obj = _ArrayDevice(idx=int(idx), name=str(name), **values)
    obj.is_alive = False
    return obj


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


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _copy_ppc(ppc: Dict) -> Dict:
    copied = {}
    for key, value in ppc.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = _copy_ppc(value)
        else:
            copied[key] = value
    return copied


def clear_hybrid_ppc_cache(file_path=None) -> None:
    with _HYBRID_PPC_CACHE_LOCK:
        if file_path is None:
            _HYBRID_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _HYBRID_PPC_CACHE.pop(path, None)


def _names(raw: Dict, block: str, count: int, fallback_prefix: str) -> np.ndarray:
    rows = _rows(raw, block)
    cols = _columns(raw, block)
    name_col = cols.get("name")
    return np.asarray(
        [
            row[name_col] if name_col is not None and name_col < len(row) and row[name_col] else f"{fallback_prefix}_{idx}"
            for idx, row in enumerate(rows[:count])
        ],
        dtype=object,
    )


def _build_dcac(raw: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rows = _rows(raw, "DCACConverter")
    cols = _columns(raw, "DCACConverter")
    if not rows:
        return _empty(len(DCAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(rows), len(DCAC_COLS)), dtype=np.float64)
    names = _names(raw, "DCACConverter", len(rows), "dcac")
    for pos, row in enumerate(rows):
        out[pos, DCAC_COLS["idx"]] = _int_cell(row, cols, "idx", pos)
        out[pos, DCAC_COLS["ac_node"]] = _int_cell(row, cols, "ac_node")
        out[pos, DCAC_COLS["dc_node"]] = _int_cell(row, cols, "dc_node")
        out[pos, DCAC_COLS["r1"]] = _float_cell(row, cols, "r1")
        out[pos, DCAC_COLS["r2"]] = _float_cell(row, cols, "r2")
        out[pos, DCAC_COLS["control_type"]] = DCAC_CONTROL_CODE.get(
            str(_cell(row, cols, "control_type", "DCV")).upper(),
            0,
        )
        out[pos, DCAC_COLS["p_ac_set"]] = _float_cell(row, cols, "p_ac_set")
        out[pos, DCAC_COLS["q_ac_set"]] = _float_cell(row, cols, "q_ac_set")
        out[pos, DCAC_COLS["v_ac_set"]] = _float_cell(row, cols, "v_ac_set")
        out[pos, DCAC_COLS["v_dc_set"]] = _float_cell(row, cols, "v_dc_set")
        out[pos, DCAC_COLS["run_stat"]] = _float_cell(row, cols, "run_stat", 1.0)
    return out, names


def _build_acac(raw: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rows = _rows(raw, "ACACConverter")
    cols = _columns(raw, "ACACConverter")
    if not rows:
        return _empty(len(ACAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(rows), len(ACAC_COLS)), dtype=np.float64)
    names = _names(raw, "ACACConverter", len(rows), "acac")
    for pos, row in enumerate(rows):
        out[pos, ACAC_COLS["idx"]] = _int_cell(row, cols, "idx", pos)
        out[pos, ACAC_COLS["i_node"]] = _int_cell(row, cols, "i_node")
        out[pos, ACAC_COLS["j_node"]] = _int_cell(row, cols, "j_node")
        out[pos, ACAC_COLS["r1"]] = _float_cell(row, cols, "r1")
        out[pos, ACAC_COLS["r2"]] = _float_cell(row, cols, "r2")
        out[pos, ACAC_COLS["control_type"]] = ACAC_CONTROL_CODE.get(
            str(_cell(row, cols, "control_type", "PQQ")).upper(),
            0,
        )
        out[pos, ACAC_COLS["p_set"]] = _float_cell(row, cols, "p_set")
        out[pos, ACAC_COLS["i_q_set"]] = _float_cell(row, cols, "i_q_set")
        out[pos, ACAC_COLS["j_q_set"]] = _float_cell(row, cols, "j_q_set")
        out[pos, ACAC_COLS["i_v_set"]] = _float_cell(row, cols, "i_v_set")
        out[pos, ACAC_COLS["j_v_set"]] = _float_cell(row, cols, "j_v_set")
        out[pos, ACAC_COLS["run_stat"]] = _float_cell(row, cols, "run_stat", 1.0)
    return out, names


def _base_values(raw: Dict) -> Tuple[float, float, float, float, float]:
    base_cols = _columns(raw, "PowerBase")
    base_rows = _rows(raw, "PowerBase")
    if not base_rows:
        raise RuntimeError("E file must define <PowerBase> with p_base, u_scale, p_scale, and i_scale")
    base_row = base_rows[0]
    p_base = _float_cell(base_row, base_cols, "p_base")
    u_scale = _float_cell(base_row, base_cols, "u_scale")
    p_scale = _float_cell(base_row, base_cols, "p_scale")
    i_scale = _float_cell(base_row, base_cols, "i_scale")
    return p_base, u_scale, p_scale, i_scale, p_base / p_scale


def _empty_ac_ppc(raw: Dict, source: Path) -> Dict:
    p_base, u_scale, p_scale, i_scale, p_base_kW = _base_values(raw)
    return {
        "format": "ac_ppc_v1",
        "source": str(source),
        "base": np.asarray([p_base, u_scale, p_scale, i_scale, p_base_kW], dtype=np.float64),
        "bus": _empty(len(AC_BUS_COLS)),
        "branch": _empty(len(AC_BRANCH_COLS)),
        "transformer": _empty(len(AC_TRANSFORMER_COLS)),
        "gen": _empty(len(AC_GEN_COLS)),
        "load": _empty(len(AC_LOAD_COLS)),
        "shunt": _empty(len(AC_SHUNT_COLS)),
        "zero_branch": _empty(len(AC_ZERO_BRANCH_COLS)),
        "switch": _empty(len(AC_SWITCH_COLS)),
        "break": _empty(len(AC_BREAK_COLS)),
        "bus_name": np.asarray([], dtype=object),
        "branch_name": np.asarray([], dtype=object),
        "transformer_name": np.asarray([], dtype=object),
        "gen_name": np.asarray([], dtype=object),
        "load_name": np.asarray([], dtype=object),
        "shunt_name": np.asarray([], dtype=object),
        "zero_branch_name": np.asarray([], dtype=object),
        "switch_name": np.asarray([], dtype=object),
        "break_name": np.asarray([], dtype=object),
    }


def _empty_dc_ppc(raw: Dict) -> Dict:
    p_base, u_scale, p_scale, i_scale, p_base_kW = _base_values(raw)
    return {
        "format": "dc_ppc_v1",
        "base": {
            "p_base": p_base,
            "u_scale": u_scale,
            "p_scale": p_scale,
            "i_scale": i_scale,
            "p_base_kW": p_base_kW,
        },
        "bus": _empty(len(DC_BUS_COLS)),
        "branch": _empty(len(DC_BRANCH_COLS)),
        "load": _empty(len(DC_LOAD_COLS)),
        "gen": _empty(len(DC_GEN_COLS)),
        "zero_branch": _empty(len(DC_ZERO_BRANCH_COLS)),
        "switch": _empty(len(DC_SWITCH_COLS)),
        "break": _empty(len(DC_BREAK_COLS)),
        "dcdc": _empty(len(DC_DCDC_COLS)),
        "node_pos": {},
        "bus_name": np.asarray([], dtype=object),
        "branch_name": np.asarray([], dtype=object),
        "load_name": np.asarray([], dtype=object),
        "gen_name": np.asarray([], dtype=object),
        "zero_branch_name": np.asarray([], dtype=object),
        "switch_name": np.asarray([], dtype=object),
        "break_name": np.asarray([], dtype=object),
        "dcdc_name": np.asarray([], dtype=object),
    }


def _node_vbase_by_idx(ppc: Dict, bus_key: str, cols: Dict[str, int]) -> Dict[int, float]:
    return {int(row[cols["idx"]]): float(row[cols["vbase"]]) for row in ppc[bus_key]}


def _scale_voltage(value: float, node_idx: int, vbase_by_idx: Dict[int, float], u_scale: float) -> float:
    vbase = vbase_by_idx.get(int(node_idx))
    if vbase is None:
        return value
    return value / (u_scale * vbase)


def _scale_ac_current(value: float, node_idx: int, ac_vbase_by_idx: Dict[int, float], p_base_kW: float, i_scale: float) -> float:
    vbase = ac_vbase_by_idx.get(int(node_idx))
    if vbase is None:
        return value
    return value / (i_scale * ac_current_base_ka(p_base_kW, vbase))


def _scale_dc_current(value: float, node_idx: int, dc_vbase_by_idx: Dict[int, float], p_base_kW: float, i_scale: float) -> float:
    vbase = dc_vbase_by_idx.get(int(node_idx))
    if vbase is None:
        return value
    return value / (i_scale * dc_current_base_ka(p_base_kW, vbase))


def _normalize_converter_arrays(ppc: Dict) -> None:
    p_base = float(ppc["base"][0])
    u_scale = float(ppc["base"][1])
    i_scale = float(ppc["base"][3])
    p_base_kW = float(ppc["base"][4])
    ac_vbase_by_idx = _node_vbase_by_idx(ppc["ac"], "bus", AC_BUS_COLS)
    dc_vbase_by_idx = _node_vbase_by_idx(ppc["dc"], "bus", DC_BUS_COLS)

    dcac = ppc["dcac"]
    for row in dcac:
        ac_node = int(row[DCAC_COLS["ac_node"]])
        dc_node = int(row[DCAC_COLS["dc_node"]])
        for col in ("p_ac_set", "q_ac_set", "dc_p", "ac_p", "ac_q"):
            row[DCAC_COLS[col]] /= p_base
        row[DCAC_COLS["v_ac_set"]] = _scale_voltage(row[DCAC_COLS["v_ac_set"]], ac_node, ac_vbase_by_idx, u_scale)
        row[DCAC_COLS["v_dc_set"]] = _scale_voltage(row[DCAC_COLS["v_dc_set"]], dc_node, dc_vbase_by_idx, u_scale)
        row[DCAC_COLS["dc_i"]] = _scale_dc_current(row[DCAC_COLS["dc_i"]], dc_node, dc_vbase_by_idx, p_base_kW, i_scale)
        row[DCAC_COLS["ac_i"]] = _scale_ac_current(row[DCAC_COLS["ac_i"]], ac_node, ac_vbase_by_idx, p_base_kW, i_scale)

    acac = ppc["acac"]
    for row in acac:
        i_node = int(row[ACAC_COLS["i_node"]])
        j_node = int(row[ACAC_COLS["j_node"]])
        for col in ("p_set", "i_q_set", "j_q_set", "i_p", "i_q", "j_p", "j_q"):
            row[ACAC_COLS[col]] /= p_base
        row[ACAC_COLS["i_v_set"]] = _scale_voltage(row[ACAC_COLS["i_v_set"]], i_node, ac_vbase_by_idx, u_scale)
        row[ACAC_COLS["j_v_set"]] = _scale_voltage(row[ACAC_COLS["j_v_set"]], j_node, ac_vbase_by_idx, u_scale)
        row[ACAC_COLS["i_i"]] = _scale_ac_current(row[ACAC_COLS["i_i"]], i_node, ac_vbase_by_idx, p_base_kW, i_scale)
        row[ACAC_COLS["j_i"]] = _scale_ac_current(row[ACAC_COLS["j_i"]], j_node, ac_vbase_by_idx, p_base_kW, i_scale)


def build_hybrid_ppc_from_e_file(file_path, use_cache: bool = True, copy_arrays: bool = False) -> Dict:
    file_key = _file_cache_key(file_path)
    if use_cache:
        with _HYBRID_PPC_CACHE_LOCK:
            cached = _HYBRID_PPC_CACHE.get(file_key[0])
            if cached is not None and cached[0] == file_key:
                return _copy_ppc(cached[1]) if copy_arrays else cached[1]

    raw = read_efile_rows_cached(file_key[0], use_cache=use_cache)
    ac_ppc = (
        build_ac_ppc_from_e_file(file_key[0], use_cache=use_cache, copy_arrays=copy_arrays)
        if _rows(raw, "ACNode")
        else _empty_ac_ppc(raw, file_key[0])
    )
    dc_ppc = (
        build_dc_ppc_from_e_file(file_key[0], use_cache=use_cache, copy_arrays=copy_arrays)
        if _rows(raw, "DCNode")
        else _empty_dc_ppc(raw)
    )
    dcac, dcac_name = _build_dcac(raw)
    acac, acac_name = _build_acac(raw)
    ppc = {
        "format": "hybrid_ppc_v1",
        "source": str(file_key[0]),
        "base": ac_ppc["base"],
        "ac": ac_ppc,
        "dc": dc_ppc,
        "dcac": dcac,
        "acac": acac,
        "dcac_name": dcac_name,
        "acac_name": acac_name,
        "dcac_cols": DCAC_COLS,
        "acac_cols": ACAC_COLS,
    }
    _normalize_converter_arrays(ppc)
    if use_cache:
        with _HYBRID_PPC_CACHE_LOCK:
            _HYBRID_PPC_CACHE[file_key[0]] = (file_key, _copy_ppc(ppc))
    return _copy_ppc(ppc) if copy_arrays else ppc


def _list_names(ppc, key, prefix, count):
    values = ppc.get(key)
    if values is None:
        return [f"{prefix}_{idx}" for idx in range(count)]
    return [str(value) for value in values]


def _list_ac_names(ppc: Dict, key: str, prefix: str, count: int) -> List[str]:
    values = ppc.get(key)
    if values is None:
        return [f"{prefix}_{idx}" for idx in range(count)]
    return [str(value) for value in values]


def _build_ac_network(ppc: Dict) -> ACPowerNetwork:
    ctrl_name = {CTRL_PQ: "PQ", CTRL_P: "P", CTRL_PV: "PV", CTRL_SLACK: "V"}
    shunt_ctrl_name = {SHUNT_Q: "Q", SHUNT_V: "V", SHUNT_B: "B", SHUNT_Z: "Z"}
    bus_names = _list_ac_names(ppc, "bus_name", "bus", ppc["bus"].shape[0])
    branch_names = _list_ac_names(ppc, "branch_name", "branch", ppc["branch"].shape[0])
    transformer_names = _list_ac_names(ppc, "transformer_name", "transformer", ppc["transformer"].shape[0])
    gen_names = _list_ac_names(ppc, "gen_name", "gen", ppc["gen"].shape[0])
    load_names = _list_ac_names(ppc, "load_name", "load", ppc["load"].shape[0])
    shunt_names = _list_ac_names(ppc, "shunt_name", "shunt", ppc["shunt"].shape[0])
    zero_branch_names = _list_ac_names(ppc, "zero_branch_name", "zero_branch", ppc["zero_branch"].shape[0])
    switch_names = _list_ac_names(ppc, "switch_name", "switch", ppc["switch"].shape[0])
    break_names = _list_ac_names(ppc, "break_name", "break", ppc.get("break", _empty(len(AC_BREAK_COLS))).shape[0])

    network = ACPowerNetwork()
    base = ppc["base"]
    network.ppc = ppc
    network.p_base = float(base[0])
    network.u_scale = float(base[1])
    network.p_scale = float(base[2])
    network.i_scale = float(base[3])
    network.p_base_kW = float(base[4])

    network.nodes = [
        ACNode(
            int(row[AC_BUS_COLS["idx"]]),
            float(row[AC_BUS_COLS["vbase"]]),
            float(row[AC_BUS_COLS["voltage"]]),
            float(row[AC_BUS_COLS["angle"]]),
            int(row[AC_BUS_COLS["run_stat"]]),
        )
        for row in ppc["bus"]
    ]
    for node in network.nodes:
        node.isl = None
        node.isl_obj = None
        node.v_gens = []
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.zero_branches = []
        node.transformers = []
        node.shunt_compensators = []

    network.branches = [
        ACBranch(
            int(row[AC_BRANCH_COLS["idx"]]),
            int(row[AC_BRANCH_COLS["i_node"]]),
            int(row[AC_BRANCH_COLS["j_node"]]),
            float(row[AC_BRANCH_COLS["r"]]),
            float(row[AC_BRANCH_COLS["x"]]),
            float(row[AC_BRANCH_COLS["b"]]),
            int(row[AC_BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["branch"]
    ]
    network.transformers = [
        ACTransformer(
            int(row[AC_TRANSFORMER_COLS["idx"]]),
            int(row[AC_TRANSFORMER_COLS["i_node"]]),
            int(row[AC_TRANSFORMER_COLS["j_node"]]),
            float(row[AC_TRANSFORMER_COLS["r"]]),
            float(row[AC_TRANSFORMER_COLS["x"]]),
            float(row[AC_TRANSFORMER_COLS["tap"]]),
            float(row[AC_TRANSFORMER_COLS["shift"]]),
            float(row[AC_TRANSFORMER_COLS["b"]]),
            int(row[AC_TRANSFORMER_COLS["run_stat"]]),
        )
        for row in ppc["transformer"]
    ]
    network.generators = [
        ACGenerator(
            int(row[AC_GEN_COLS["idx"]]),
            int(row[AC_GEN_COLS["node"]]),
            ctrl_name.get(int(row[AC_GEN_COLS["control_type"]]), "PQ"),
            float(row[AC_GEN_COLS["p_set"]]),
            float(row[AC_GEN_COLS["q_set"]]),
            float(row[AC_GEN_COLS["v_set"]]),
            float(row[AC_GEN_COLS["alpha"]]),
            int(row[AC_GEN_COLS["run_stat"]]),
        )
        for row in ppc["gen"]
    ]
    network.loads = [
        ACLoad(
            int(row[AC_LOAD_COLS["idx"]]),
            int(row[AC_LOAD_COLS["node"]]),
            float(row[AC_LOAD_COLS["pbase"]]),
            float(row[AC_LOAD_COLS["pv0"]]),
            float(row[AC_LOAD_COLS["pv1"]]),
            float(row[AC_LOAD_COLS["pv2"]]),
            float(row[AC_LOAD_COLS["qbase"]]),
            float(row[AC_LOAD_COLS["qv0"]]),
            float(row[AC_LOAD_COLS["qv1"]]),
            float(row[AC_LOAD_COLS["qv2"]]),
            int(row[AC_LOAD_COLS["run_stat"]]),
        )
        for row in ppc["load"]
    ]
    network.shunt_compensators = [
        ACShuntCompensator(
            int(row[AC_SHUNT_COLS["idx"]]),
            int(row[AC_SHUNT_COLS["node"]]),
            shunt_ctrl_name.get(int(row[AC_SHUNT_COLS["control_type"]]), "Q"),
            float(row[AC_SHUNT_COLS["q_set"]]),
            float(row[AC_SHUNT_COLS["g_set"]]),
            float(row[AC_SHUNT_COLS["b_set"]]),
            float(row[AC_SHUNT_COLS["v_set"]]),
            int(row[AC_SHUNT_COLS["run_stat"]]),
        )
        for row in ppc["shunt"]
    ]
    network.zero_branches = [
        ACZeroBranch(
            int(row[AC_ZERO_BRANCH_COLS["idx"]]),
            int(row[AC_ZERO_BRANCH_COLS["i_node"]]),
            int(row[AC_ZERO_BRANCH_COLS["j_node"]]),
            int(row[AC_ZERO_BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["zero_branch"]
    ]
    network.switches = [
        ACSwitch(
            int(row[AC_SWITCH_COLS["idx"]]),
            int(row[AC_SWITCH_COLS["i_node"]]),
            int(row[AC_SWITCH_COLS["j_node"]]),
            int(row[AC_SWITCH_COLS["status"]]),
            int(row[AC_SWITCH_COLS["run_stat"]]),
        )
        for row in ppc["switch"]
    ]
    network.breakers = [
        ACBreak(
            int(row[AC_BREAK_COLS["idx"]]),
            int(row[AC_BREAK_COLS["i_node"]]),
            int(row[AC_BREAK_COLS["j_node"]]),
            int(row[AC_BREAK_COLS["status"]]),
            int(row[AC_BREAK_COLS["run_stat"]]),
        )
        for row in ppc.get("break", _empty(len(AC_BREAK_COLS)))
    ]
    network.node_dict = {}
    network.switch_dict = {}
    network.break_dict = {}
    network.load_dict = {}
    network.generator_dict = {}
    network.zero_branch_dict = {}
    network.branch_dict = {}
    network.transformer_dict = {}
    network.shunt_compensator_dict = {}
    network.islands = []
    for obj, name in zip(network.nodes, bus_names):
        obj.name = name
        obj.is_alive = False
    for obj, row, name in zip(network.branches, ppc["branch"], branch_names):
        obj.name = name
        obj.i_p = float(row[AC_BRANCH_COLS["i_p"]])
        obj.i_q = float(row[AC_BRANCH_COLS["i_q"]])
        obj.i_c = float(row[AC_BRANCH_COLS["i_c"]])
        obj.j_p = float(row[AC_BRANCH_COLS["j_p"]])
        obj.j_q = float(row[AC_BRANCH_COLS["j_q"]])
        obj.j_c = float(row[AC_BRANCH_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(network.transformers, ppc["transformer"], transformer_names):
        obj.name = name
        obj.i_p = float(row[AC_TRANSFORMER_COLS["i_p"]])
        obj.i_q = float(row[AC_TRANSFORMER_COLS["i_q"]])
        obj.i_c = float(row[AC_TRANSFORMER_COLS["i_c"]])
        obj.j_p = float(row[AC_TRANSFORMER_COLS["j_p"]])
        obj.j_q = float(row[AC_TRANSFORMER_COLS["j_q"]])
        obj.j_c = float(row[AC_TRANSFORMER_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(network.generators, ppc["gen"], gen_names):
        obj.name = name
        obj.p = float(row[AC_GEN_COLS["p"]])
        obj.q = float(row[AC_GEN_COLS["q"]])
        obj.current = float(row[AC_GEN_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.loads, ppc["load"], load_names):
        obj.name = name
        obj.p = float(row[AC_LOAD_COLS["p"]])
        obj.q = float(row[AC_LOAD_COLS["q"]])
        obj.current = float(row[AC_LOAD_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.shunt_compensators, ppc["shunt"], shunt_names):
        obj.name = name
        obj.p = float(row[AC_SHUNT_COLS["p"]])
        obj.q = float(row[AC_SHUNT_COLS["q"]])
        obj.current = float(row[AC_SHUNT_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.zero_branches, ppc["zero_branch"], zero_branch_names):
        obj.name = name
        obj.p = float(row[AC_ZERO_BRANCH_COLS["p"]])
        obj.q = float(row[AC_ZERO_BRANCH_COLS["q"]])
        obj.current = float(row[AC_ZERO_BRANCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.switches, ppc["switch"], switch_names):
        obj.name = name
        obj.p = float(row[AC_SWITCH_COLS["p"]])
        obj.q = float(row[AC_SWITCH_COLS["q"]])
        obj.current = float(row[AC_SWITCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.breakers, ppc.get("break", _empty(len(AC_BREAK_COLS))), break_names):
        obj.name = name
        obj.p = float(row[AC_BREAK_COLS["p"]])
        obj.q = float(row[AC_BREAK_COLS["q"]])
        obj.current = float(row[AC_BREAK_COLS["current"]])
        obj.is_alive = False
    return network


def _build_dc_network(ppc: Dict) -> DCPowerNetwork:
    network = DCPowerNetwork()
    base = ppc["base"]
    network.ppc = ppc
    network.p_base = float(base["p_base"])
    network.p_base_kW = float(base["p_base_kW"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network._load_objects_from_ppc(ppc)
    return network


def build_hybrid_model_from_ppc(ppc: Dict):
    ac_network = _build_ac_network(ppc["ac"])
    dc_network = _build_dc_network(ppc["dc"])
    dcac = [
        _device(
            row[DCAC_COLS["idx"]],
            ppc["dcac_name"][pos],
            ac_node=int(row[DCAC_COLS["ac_node"]]),
            dc_node=int(row[DCAC_COLS["dc_node"]]),
            r1=float(row[DCAC_COLS["r1"]]),
            r2=float(row[DCAC_COLS["r2"]]),
            control_type=DCAC_CONTROL_LABEL.get(int(row[DCAC_COLS["control_type"]]), "DCV"),
            p_ac_set=float(row[DCAC_COLS["p_ac_set"]]),
            q_ac_set=float(row[DCAC_COLS["q_ac_set"]]),
            v_ac_set=float(row[DCAC_COLS["v_ac_set"]]),
            v_dc_set=float(row[DCAC_COLS["v_dc_set"]]),
            run_stat=int(row[DCAC_COLS["run_stat"]]),
            dc_p=float(row[DCAC_COLS["dc_p"]]),
            ac_p=float(row[DCAC_COLS["ac_p"]]),
            ac_q=float(row[DCAC_COLS["ac_q"]]),
            dc_i=float(row[DCAC_COLS["dc_i"]]),
            ac_i=float(row[DCAC_COLS["ac_i"]]),
            ac_node_obj=None,
            dc_node_obj=None,
        )
        for pos, row in enumerate(ppc["dcac"])
    ]
    acac = [
        _device(
            row[ACAC_COLS["idx"]],
            ppc["acac_name"][pos],
            i_node=int(row[ACAC_COLS["i_node"]]),
            j_node=int(row[ACAC_COLS["j_node"]]),
            r1=float(row[ACAC_COLS["r1"]]),
            r2=float(row[ACAC_COLS["r2"]]),
            control_type=ACAC_CONTROL_LABEL.get(int(row[ACAC_COLS["control_type"]]), "PQQ"),
            p_set=float(row[ACAC_COLS["p_set"]]),
            i_q_set=float(row[ACAC_COLS["i_q_set"]]),
            j_q_set=float(row[ACAC_COLS["j_q_set"]]),
            i_v_set=float(row[ACAC_COLS["i_v_set"]]),
            j_v_set=float(row[ACAC_COLS["j_v_set"]]),
            run_stat=int(row[ACAC_COLS["run_stat"]]),
            i_p=float(row[ACAC_COLS["i_p"]]),
            i_q=float(row[ACAC_COLS["i_q"]]),
            j_p=float(row[ACAC_COLS["j_p"]]),
            j_q=float(row[ACAC_COLS["j_q"]]),
            i_i=float(row[ACAC_COLS["i_i"]]),
            j_i=float(row[ACAC_COLS["j_i"]]),
            i_node_obj=None,
            j_node_obj=None,
        )
        for pos, row in enumerate(ppc["acac"])
    ]
    model = SimpleNamespace(
        ac=ac_network,
        dc=dc_network,
        DCNode=dc_network.nodes,
        DCBranch=dc_network.branches,
        DCLoad=dc_network.loads,
        DCGenerator=dc_network.generators,
        DCZeroBranch=dc_network.zero_branches,
        DCSwitch=dc_network.switches,
        DCBreak=getattr(dc_network, "breakers", []),
        DCDCConverter=dc_network.dcdc_converters,
        DCShuntCompensator=[],
        DCACConverter=dcac,
        ACACConverter=acac,
    )
    base = ppc["base"]
    model.p_base = float(base[0])
    model.u_scale = float(base[1])
    model.p_scale = float(base[2])
    model.i_scale = float(base[3])
    model.p_base_kW = float(base[4])
    model.ac.ppc = ppc["ac"]
    model.dc.ppc = ppc["dc"]
    return model
