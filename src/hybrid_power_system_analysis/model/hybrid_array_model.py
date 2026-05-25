import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
for path in (MODEL_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import (
    BUS_COLS as AC_BUS_COLS,
    _build_ac_ppc_from_model,
    build_ac_network_from_ppc as _build_ac_network_from_ppc,
    build_ac_ppc_from_efile_rows,
)
from dc_array_model import (
    BUS_COLS as DC_BUS_COLS,
    _build_dc_ppc_from_model,
    build_dc_network_from_ppc as _build_dc_network_from_ppc,
    build_dc_ppc_from_efile_rows,
)
from efile_read import _read_efile_rows
from unit_system import normalize_model_named_units
from unit_system import ac_current_base_ka, dc_current_base_ka


DCAC_CONTROL_CODE = {"DCV": 0, "ACV": 1, "ACP": 2}
# DCAC 的正式控制码只保留 DCV/ACV/ACP；PH/PQ 是旧 E 文件里的输入别名。
DCAC_CONTROL_PARSE_CODE = {
    **DCAC_CONTROL_CODE,
    "PH": DCAC_CONTROL_CODE["ACV"],
    "PQ": DCAC_CONTROL_CODE["ACP"],
}
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

def _attr(obj, attr: str, default=""):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_attr(obj, attr: str, default: float = 0.0) -> float:
    value = _attr(obj, attr, default)
    return default if value in (None, "") else float(value)


def _int_attr(obj, attr: str, default: int = 0) -> int:
    value = _attr(obj, attr, default)
    return default if value in (None, "") else int(float(value))


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _rows_for(data: Dict, table_name: str):
    table = data.get(table_name)
    if not table:
        return {}, []
    return {str(name): pos for pos, name in enumerate(table.get("header_list", []))}, table.get("rows", [])


def _cell(row, col, default=""):
    if col is None or col >= len(row):
        return default
    value = row[col]
    return default if value in (None, "") else value


def _float_cell(row, col, default: float = 0.0) -> float:
    return float(_cell(row, col, default))


def _int_cell(row, col, default: int = 0) -> int:
    return int(float(_cell(row, col, default)))


def _float_column(table_rows, columns, attr: str, default: float = 0.0) -> np.ndarray:
    col = columns.get(attr)
    n = len(table_rows)
    if col is None or n == 0:
        return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
    values = [None] * n
    for i in range(n):
        row = table_rows[i]
        if col >= len(row):
            values[i] = default
        else:
            v = row[col]
            values[i] = default if v in (None, "") else float(v)
    return np.asarray(values, dtype=np.float64)


def _int_column(table_rows, columns, attr: str, default: int = 0) -> np.ndarray:
    col = columns.get(attr)
    n = len(table_rows)
    if col is None or n == 0:
        return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
    values = [None] * n
    for i in range(n):
        row = table_rows[i]
        if col >= len(row):
            values[i] = default
        else:
            v = row[col]
            values[i] = default if v in (None, "") else int(float(v))
    return np.asarray(values, dtype=np.float64)


def _code_column(table_rows, columns, attr: str, mapping: Dict[str, int], default_label: str) -> np.ndarray:
    col = columns.get(attr)
    default = mapping[default_label]
    if col is None:
        return np.full(len(table_rows), float(default), dtype=np.float64)
    return np.asarray(
        [mapping.get(str(_cell(row, col, default_label)).upper(), default) for row in table_rows],
        dtype=np.float64,
    )


def _names_from_rows(table_rows, columns, prefix: str, idx_values: np.ndarray) -> np.ndarray:
    name_col = columns.get("name")
    if name_col is None:
        return np.asarray([f"{prefix}_{int(idx)}" for idx in idx_values], dtype=object)
    return np.asarray(
        [
            str(_cell(row, name_col, "") or f"{prefix}_{int(idx_values[pos])}")
            for pos, row in enumerate(table_rows)
        ],
        dtype=object,
    )


def _raw_vbase_maps(ac_ppc: Dict, dc_ppc: Dict) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    ac_base = ac_ppc["base"]
    p_base_kW = float(ac_base["p_base_kW"])
    ac_u_scale = float(ac_base["u_scale"])
    ac_i_scale = float(ac_base["i_scale"])
    dc_base = dc_ppc["base"]
    dc_u_scale = float(dc_base["u_scale"])
    dc_i_scale = float(dc_base["i_scale"])
    ac_raw_vbase = {
        int(row[AC_BUS_COLS["idx"]]): float(row[AC_BUS_COLS["vbase"]]) * ac_u_scale
        for row in ac_ppc["bus"]
    }
    dc_raw_vbase = {
        int(row[DC_BUS_COLS["idx"]]): float(row[DC_BUS_COLS["vbase"]]) * dc_u_scale
        for row in dc_ppc["bus"]
    }
    ac_current = {
        int(row[AC_BUS_COLS["idx"]]): ac_i_scale * ac_current_base_ka(p_base_kW, float(row[AC_BUS_COLS["vbase"]]))
        for row in ac_ppc["bus"]
    }
    dc_current = {
        int(row[DC_BUS_COLS["idx"]]): dc_i_scale * dc_current_base_ka(p_base_kW, float(row[DC_BUS_COLS["vbase"]]))
        for row in dc_ppc["bus"]
    }
    return ac_raw_vbase, dc_raw_vbase, ac_current, dc_current


def _scale_by_node(node_values: np.ndarray, scales_by_idx: Dict[int, float]) -> np.ndarray:
    return np.asarray([scales_by_idx.get(int(node), 1.0) for node in node_values], dtype=np.float64)


def _raw_vbase_by_node(node_values: np.ndarray, raw_vbase_by_idx: Dict[int, float]) -> np.ndarray:
    return np.asarray([raw_vbase_by_idx.get(int(node), 0.0) for node in node_values], dtype=np.float64)


def _ppc_bus_value_by_node(ppc: Dict, node_values: np.ndarray, col: int, default: float = 0.0) -> np.ndarray:
    bus = np.asarray(ppc.get("bus", np.zeros((0, 0))), dtype=np.float64)
    if bus.size == 0:
        return np.full(node_values.shape, default, dtype=np.float64)
    idx = bus[:, 0].astype(np.int64, copy=False)
    values = bus[:, col].astype(np.float64, copy=False)
    value_by_idx = {int(node_id): float(value) for node_id, value in zip(idx, values)}
    return np.asarray([value_by_idx.get(int(node), default) for node in node_values], dtype=np.float64)


def _voltage_set_column(
    table_rows,
    columns,
    attr: str,
    node_values: np.ndarray,
    raw_vbase_by_idx: Dict[int, float],
    *,
    default: float = 1.0,
) -> np.ndarray:
    if attr not in columns:
        return np.full(len(table_rows), default, dtype=np.float64)
    raw = _float_column(table_rows, columns, attr, default)
    raw_vbase = _raw_vbase_by_node(node_values.astype(np.int64, copy=False), raw_vbase_by_idx)
    return np.divide(raw, raw_vbase, out=np.full_like(raw, default), where=np.abs(raw_vbase) > 1e-12)


def _power_column(table_rows, columns, attr: str, p_base: float) -> np.ndarray:
    return _float_column(table_rows, columns, attr) / p_base


def _current_column(
    table_rows,
    columns,
    attr: str,
    node_values: np.ndarray,
    current_scale_by_node: Dict[int, float],
) -> np.ndarray:
    raw = _float_column(table_rows, columns, attr)
    scales = _scale_by_node(node_values.astype(np.int64, copy=False), current_scale_by_node)
    return np.divide(raw, scales, out=np.zeros_like(raw), where=np.abs(scales) > 1e-12)


def _names(devices, count: int, fallback_prefix: str) -> np.ndarray:
    return np.asarray(
        [
            str(_attr(dev, "name", f"{fallback_prefix}_{idx}") or f"{fallback_prefix}_{idx}")
            for idx, dev in enumerate(devices[:count])
        ],
        dtype=object,
    )


def _build_dcac(model) -> Tuple[np.ndarray, np.ndarray]:
    converters = list(getattr(model, "DCACConverter", []))
    if not converters:
        return _empty(len(DCAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(converters), len(DCAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "dcac")
    for pos, conv in enumerate(converters):
        out[pos, DCAC_COLS["idx"]] = _int_attr(conv, "idx", pos)
        out[pos, DCAC_COLS["ac_node"]] = _int_attr(conv, "ac_node")
        out[pos, DCAC_COLS["dc_node"]] = _int_attr(conv, "dc_node")
        out[pos, DCAC_COLS["r1"]] = _float_attr(conv, "r1")
        out[pos, DCAC_COLS["r2"]] = _float_attr(conv, "r2")
        out[pos, DCAC_COLS["control_type"]] = DCAC_CONTROL_PARSE_CODE.get(
            str(_attr(conv, "control_type", "DCV")).upper(),
            0,
        )
        out[pos, DCAC_COLS["p_ac_set"]] = _float_attr(conv, "p_ac_set")
        out[pos, DCAC_COLS["q_ac_set"]] = _float_attr(conv, "q_ac_set")
        out[pos, DCAC_COLS["v_ac_set"]] = _float_attr(conv, "v_ac_set")
        out[pos, DCAC_COLS["v_dc_set"]] = _float_attr(conv, "v_dc_set")
        out[pos, DCAC_COLS["run_stat"]] = _float_attr(conv, "run_stat", 1.0)
        out[pos, DCAC_COLS["dc_p"]] = _float_attr(conv, "dc_p")
        out[pos, DCAC_COLS["ac_p"]] = _float_attr(conv, "ac_p")
        out[pos, DCAC_COLS["ac_q"]] = _float_attr(conv, "ac_q")
        out[pos, DCAC_COLS["dc_i"]] = _float_attr(conv, "dc_i")
        out[pos, DCAC_COLS["ac_i"]] = _float_attr(conv, "ac_i")
    return out, names


def _build_acac(model) -> Tuple[np.ndarray, np.ndarray]:
    converters = list(getattr(model, "ACACConverter", []))
    if not converters:
        return _empty(len(ACAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(converters), len(ACAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "acac")
    for pos, conv in enumerate(converters):
        out[pos, ACAC_COLS["idx"]] = _int_attr(conv, "idx", pos)
        out[pos, ACAC_COLS["i_node"]] = _int_attr(conv, "i_node")
        out[pos, ACAC_COLS["j_node"]] = _int_attr(conv, "j_node")
        out[pos, ACAC_COLS["r1"]] = _float_attr(conv, "r1")
        out[pos, ACAC_COLS["r2"]] = _float_attr(conv, "r2")
        out[pos, ACAC_COLS["control_type"]] = ACAC_CONTROL_CODE.get(
            str(_attr(conv, "control_type", "PQQ")).upper(),
            0,
        )
        out[pos, ACAC_COLS["p_set"]] = _float_attr(conv, "p_set")
        out[pos, ACAC_COLS["i_q_set"]] = _float_attr(conv, "i_q_set")
        out[pos, ACAC_COLS["j_q_set"]] = _float_attr(conv, "j_q_set")
        out[pos, ACAC_COLS["i_v_set"]] = _float_attr(conv, "i_v_set")
        out[pos, ACAC_COLS["j_v_set"]] = _float_attr(conv, "j_v_set")
        out[pos, ACAC_COLS["run_stat"]] = _float_attr(conv, "run_stat", 1.0)
        out[pos, ACAC_COLS["i_p"]] = _float_attr(conv, "i_p")
        out[pos, ACAC_COLS["i_q"]] = _float_attr(conv, "i_q")
        out[pos, ACAC_COLS["j_p"]] = _float_attr(conv, "j_p")
        out[pos, ACAC_COLS["j_q"]] = _float_attr(conv, "j_q")
        out[pos, ACAC_COLS["i_i"]] = _float_attr(conv, "i_i")
        out[pos, ACAC_COLS["j_i"]] = _float_attr(conv, "j_i")
    return out, names


def _converter_classes():
    if __package__ == "model":
        from model.hybrid_model import ACACConverter, DCACConverter
    else:
        from hybrid_model import ACACConverter, DCACConverter

    return DCACConverter, ACACConverter


def _build_dcac_with_objects(model) -> Tuple[np.ndarray, np.ndarray, list]:
    converters = list(getattr(model, "DCACConverter", []))
    if not converters:
        return _empty(len(DCAC_COLS)), np.asarray([], dtype=object), []
    DCACConverter, _ACACConverter = _converter_classes()
    out = np.zeros((len(converters), len(DCAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "dcac")
    objects = []
    for pos, conv in enumerate(converters):
        idx = _int_attr(conv, "idx", pos)
        ac_node = _int_attr(conv, "ac_node")
        dc_node = _int_attr(conv, "dc_node")
        r1 = _float_attr(conv, "r1")
        r2 = _float_attr(conv, "r2")
        control_type = str(_attr(conv, "control_type", "DCV")).upper()
        control_code = DCAC_CONTROL_PARSE_CODE.get(control_type, 0)
        p_ac_set = _float_attr(conv, "p_ac_set")
        q_ac_set = _float_attr(conv, "q_ac_set")
        v_ac_set = _float_attr(conv, "v_ac_set")
        v_dc_set = _float_attr(conv, "v_dc_set")
        run_stat = _int_attr(conv, "run_stat", 1)
        dc_p = _float_attr(conv, "dc_p")
        ac_p = _float_attr(conv, "ac_p")
        ac_q = _float_attr(conv, "ac_q")
        dc_i = _float_attr(conv, "dc_i")
        ac_i = _float_attr(conv, "ac_i")
        out[pos, DCAC_COLS["idx"]] = idx
        out[pos, DCAC_COLS["ac_node"]] = ac_node
        out[pos, DCAC_COLS["dc_node"]] = dc_node
        out[pos, DCAC_COLS["r1"]] = r1
        out[pos, DCAC_COLS["r2"]] = r2
        out[pos, DCAC_COLS["control_type"]] = control_code
        out[pos, DCAC_COLS["p_ac_set"]] = p_ac_set
        out[pos, DCAC_COLS["q_ac_set"]] = q_ac_set
        out[pos, DCAC_COLS["v_ac_set"]] = v_ac_set
        out[pos, DCAC_COLS["v_dc_set"]] = v_dc_set
        out[pos, DCAC_COLS["run_stat"]] = run_stat
        out[pos, DCAC_COLS["dc_p"]] = dc_p
        out[pos, DCAC_COLS["ac_p"]] = ac_p
        out[pos, DCAC_COLS["ac_q"]] = ac_q
        out[pos, DCAC_COLS["dc_i"]] = dc_i
        out[pos, DCAC_COLS["ac_i"]] = ac_i
        obj = DCACConverter(
            idx,
            ac_node,
            dc_node,
            r1,
            r2,
            DCAC_CONTROL_LABEL.get(control_code, "DCV"),
            p_ac_set,
            q_ac_set,
            v_ac_set,
            v_dc_set,
            run_stat,
        )
        obj.name = str(names[pos])
        obj.dc_p = dc_p
        obj.ac_p = ac_p
        obj.ac_q = ac_q
        obj.dc_i = dc_i
        obj.ac_i = ac_i
        obj.is_alive = False
        objects.append(obj)
    return out, names, objects


def _build_acac_with_objects(model) -> Tuple[np.ndarray, np.ndarray, list]:
    converters = list(getattr(model, "ACACConverter", []))
    if not converters:
        return _empty(len(ACAC_COLS)), np.asarray([], dtype=object), []
    _DCACConverter, ACACConverter = _converter_classes()
    out = np.zeros((len(converters), len(ACAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "acac")
    objects = []
    for pos, conv in enumerate(converters):
        idx = _int_attr(conv, "idx", pos)
        i_node = _int_attr(conv, "i_node")
        j_node = _int_attr(conv, "j_node")
        r1 = _float_attr(conv, "r1")
        r2 = _float_attr(conv, "r2")
        control_type = str(_attr(conv, "control_type", "PQQ")).upper()
        control_code = ACAC_CONTROL_CODE.get(control_type, 0)
        p_set = _float_attr(conv, "p_set")
        i_q_set = _float_attr(conv, "i_q_set")
        j_q_set = _float_attr(conv, "j_q_set")
        i_v_set = _float_attr(conv, "i_v_set")
        j_v_set = _float_attr(conv, "j_v_set")
        run_stat = _int_attr(conv, "run_stat", 1)
        i_p = _float_attr(conv, "i_p")
        i_q = _float_attr(conv, "i_q")
        j_p = _float_attr(conv, "j_p")
        j_q = _float_attr(conv, "j_q")
        i_i = _float_attr(conv, "i_i")
        j_i = _float_attr(conv, "j_i")
        out[pos, ACAC_COLS["idx"]] = idx
        out[pos, ACAC_COLS["i_node"]] = i_node
        out[pos, ACAC_COLS["j_node"]] = j_node
        out[pos, ACAC_COLS["r1"]] = r1
        out[pos, ACAC_COLS["r2"]] = r2
        out[pos, ACAC_COLS["control_type"]] = control_code
        out[pos, ACAC_COLS["p_set"]] = p_set
        out[pos, ACAC_COLS["i_q_set"]] = i_q_set
        out[pos, ACAC_COLS["j_q_set"]] = j_q_set
        out[pos, ACAC_COLS["i_v_set"]] = i_v_set
        out[pos, ACAC_COLS["j_v_set"]] = j_v_set
        out[pos, ACAC_COLS["run_stat"]] = run_stat
        out[pos, ACAC_COLS["i_p"]] = i_p
        out[pos, ACAC_COLS["i_q"]] = i_q
        out[pos, ACAC_COLS["j_p"]] = j_p
        out[pos, ACAC_COLS["j_q"]] = j_q
        out[pos, ACAC_COLS["i_i"]] = i_i
        out[pos, ACAC_COLS["j_i"]] = j_i
        obj = ACACConverter(
            idx,
            i_node,
            j_node,
            r1,
            r2,
            ACAC_CONTROL_LABEL.get(control_code, "PQQ"),
            p_set,
            i_q_set,
            j_q_set,
            i_v_set,
            j_v_set,
            run_stat,
        )
        obj.name = str(names[pos])
        obj.i_p = i_p
        obj.i_q = i_q
        obj.j_p = j_p
        obj.j_q = j_q
        obj.i_i = i_i
        obj.j_i = j_i
        obj.is_alive = False
        objects.append(obj)
    return out, names, objects


def _build_dcac_from_rows(
    rows: Dict,
    ac_ppc: Dict,
    dc_ppc: Dict,
    *,
    build_objects: bool = True,
    vbase_maps: Optional[Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    columns, table_rows = _rows_for(rows, "DCACConverter")
    if not table_rows:
        return _empty(len(DCAC_COLS)), np.asarray([], dtype=object), []
    p_base = float(ac_ppc["base"]["p_base"])
    ac_raw_vbase, dc_raw_vbase, ac_current, dc_current = (
        _raw_vbase_maps(ac_ppc, dc_ppc) if vbase_maps is None else vbase_maps
    )
    out = np.zeros((len(table_rows), len(DCAC_COLS)), dtype=np.float64)
    out[:, DCAC_COLS["idx"]] = _int_column(table_rows, columns, "idx")
    out[:, DCAC_COLS["ac_node"]] = _int_column(table_rows, columns, "ac_node")
    out[:, DCAC_COLS["dc_node"]] = _int_column(table_rows, columns, "dc_node")
    out[:, DCAC_COLS["r1"]] = _float_column(table_rows, columns, "r1")
    out[:, DCAC_COLS["r2"]] = _float_column(table_rows, columns, "r2")
    out[:, DCAC_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", DCAC_CONTROL_PARSE_CODE, "DCV")
    out[:, DCAC_COLS["p_ac_set"]] = _power_column(table_rows, columns, "p_ac_set", p_base)
    out[:, DCAC_COLS["q_ac_set"]] = _power_column(table_rows, columns, "q_ac_set", p_base)
    out[:, DCAC_COLS["v_ac_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "v_ac_set",
        out[:, DCAC_COLS["ac_node"]],
        ac_raw_vbase,
        default=0.0,
    )
    out[:, DCAC_COLS["v_dc_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "v_dc_set",
        out[:, DCAC_COLS["dc_node"]],
        dc_raw_vbase,
        default=0.0,
    )
    ctrl = out[:, DCAC_COLS["control_type"]].astype(np.int64, copy=False)
    ac_voltage_ctrl = ctrl == DCAC_CONTROL_CODE["ACV"]
    if np.any(ac_voltage_ctrl):
        fallback = _ppc_bus_value_by_node(ac_ppc, out[:, DCAC_COLS["ac_node"]], 2, default=1.0)
        missing = ac_voltage_ctrl & (np.abs(out[:, DCAC_COLS["v_ac_set"]]) <= 1e-12)
        out[missing, DCAC_COLS["v_ac_set"]] = fallback[missing]
    dc_voltage_ctrl = ctrl == DCAC_CONTROL_CODE["DCV"]
    if np.any(dc_voltage_ctrl):
        fallback = _ppc_bus_value_by_node(dc_ppc, out[:, DCAC_COLS["dc_node"]], 2, default=1.0)
        missing = dc_voltage_ctrl & (np.abs(out[:, DCAC_COLS["v_dc_set"]]) <= 1e-12)
        out[missing, DCAC_COLS["v_dc_set"]] = fallback[missing]
    out[:, DCAC_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    if "dc_p" in columns:
        out[:, DCAC_COLS["dc_p"]] = _power_column(table_rows, columns, "dc_p", p_base)
    if "ac_p" in columns:
        out[:, DCAC_COLS["ac_p"]] = _power_column(table_rows, columns, "ac_p", p_base)
    if "ac_q" in columns:
        out[:, DCAC_COLS["ac_q"]] = _power_column(table_rows, columns, "ac_q", p_base)
    if "dc_i" in columns:
        out[:, DCAC_COLS["dc_i"]] = _current_column(table_rows, columns, "dc_i", out[:, DCAC_COLS["dc_node"]], dc_current)
    if "ac_i" in columns:
        out[:, DCAC_COLS["ac_i"]] = _current_column(table_rows, columns, "ac_i", out[:, DCAC_COLS["ac_node"]], ac_current)

    names = _names_from_rows(table_rows, columns, "dcac", out[:, DCAC_COLS["idx"]])
    if not build_objects:
        return out, names, []
    DCACConverter, _ACACConverter = _converter_classes()
    objects = []
    for pos, row in enumerate(out):
        control_code = int(row[DCAC_COLS["control_type"]])
        obj = DCACConverter(
            int(row[DCAC_COLS["idx"]]),
            int(row[DCAC_COLS["ac_node"]]),
            int(row[DCAC_COLS["dc_node"]]),
            float(row[DCAC_COLS["r1"]]),
            float(row[DCAC_COLS["r2"]]),
            DCAC_CONTROL_LABEL.get(control_code, "DCV"),
            float(row[DCAC_COLS["p_ac_set"]]),
            float(row[DCAC_COLS["q_ac_set"]]),
            float(row[DCAC_COLS["v_ac_set"]]),
            float(row[DCAC_COLS["v_dc_set"]]),
            int(row[DCAC_COLS["run_stat"]]),
        )
        obj.name = str(names[pos])
        obj.dc_p = float(row[DCAC_COLS["dc_p"]])
        obj.ac_p = float(row[DCAC_COLS["ac_p"]])
        obj.ac_q = float(row[DCAC_COLS["ac_q"]])
        obj.dc_i = float(row[DCAC_COLS["dc_i"]])
        obj.ac_i = float(row[DCAC_COLS["ac_i"]])
        obj.is_alive = False
        objects.append(obj)
    return out, names, objects


def _build_acac_from_rows(
    rows: Dict,
    ac_ppc: Dict,
    dc_ppc: Dict,
    *,
    build_objects: bool = True,
    vbase_maps: Optional[Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    columns, table_rows = _rows_for(rows, "ACACConverter")
    if not table_rows:
        return _empty(len(ACAC_COLS)), np.asarray([], dtype=object), []
    p_base = float(ac_ppc["base"]["p_base"])
    ac_raw_vbase, _dc_raw_vbase, ac_current, _dc_current = (
        _raw_vbase_maps(ac_ppc, dc_ppc) if vbase_maps is None else vbase_maps
    )
    out = np.zeros((len(table_rows), len(ACAC_COLS)), dtype=np.float64)
    out[:, ACAC_COLS["idx"]] = _int_column(table_rows, columns, "idx")
    out[:, ACAC_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
    out[:, ACAC_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
    out[:, ACAC_COLS["r1"]] = _float_column(table_rows, columns, "r1")
    out[:, ACAC_COLS["r2"]] = _float_column(table_rows, columns, "r2")
    out[:, ACAC_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", ACAC_CONTROL_CODE, "PQQ")
    out[:, ACAC_COLS["p_set"]] = _power_column(table_rows, columns, "p_set", p_base)
    out[:, ACAC_COLS["i_q_set"]] = _power_column(table_rows, columns, "i_q_set", p_base)
    out[:, ACAC_COLS["j_q_set"]] = _power_column(table_rows, columns, "j_q_set", p_base)
    out[:, ACAC_COLS["i_v_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "i_v_set",
        out[:, ACAC_COLS["i_node"]],
        ac_raw_vbase,
        default=0.0,
    )
    out[:, ACAC_COLS["j_v_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "j_v_set",
        out[:, ACAC_COLS["j_node"]],
        ac_raw_vbase,
        default=0.0,
    )
    out[:, ACAC_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    if "i_p" in columns:
        out[:, ACAC_COLS["i_p"]] = _power_column(table_rows, columns, "i_p", p_base)
    if "i_q" in columns:
        out[:, ACAC_COLS["i_q"]] = _power_column(table_rows, columns, "i_q", p_base)
    if "j_p" in columns:
        out[:, ACAC_COLS["j_p"]] = _power_column(table_rows, columns, "j_p", p_base)
    if "j_q" in columns:
        out[:, ACAC_COLS["j_q"]] = _power_column(table_rows, columns, "j_q", p_base)
    if "i_i" in columns:
        out[:, ACAC_COLS["i_i"]] = _current_column(table_rows, columns, "i_i", out[:, ACAC_COLS["i_node"]], ac_current)
    if "j_i" in columns:
        out[:, ACAC_COLS["j_i"]] = _current_column(table_rows, columns, "j_i", out[:, ACAC_COLS["j_node"]], ac_current)

    names = _names_from_rows(table_rows, columns, "acac", out[:, ACAC_COLS["idx"]])
    if not build_objects:
        return out, names, []
    _DCACConverter, ACACConverter = _converter_classes()
    objects = []
    for pos, row in enumerate(out):
        control_code = int(row[ACAC_COLS["control_type"]])
        obj = ACACConverter(
            int(row[ACAC_COLS["idx"]]),
            int(row[ACAC_COLS["i_node"]]),
            int(row[ACAC_COLS["j_node"]]),
            float(row[ACAC_COLS["r1"]]),
            float(row[ACAC_COLS["r2"]]),
            ACAC_CONTROL_LABEL.get(control_code, "PQQ"),
            float(row[ACAC_COLS["p_set"]]),
            float(row[ACAC_COLS["i_q_set"]]),
            float(row[ACAC_COLS["j_q_set"]]),
            float(row[ACAC_COLS["i_v_set"]]),
            float(row[ACAC_COLS["j_v_set"]]),
            int(row[ACAC_COLS["run_stat"]]),
        )
        obj.name = str(names[pos])
        obj.i_p = float(row[ACAC_COLS["i_p"]])
        obj.i_q = float(row[ACAC_COLS["i_q"]])
        obj.j_p = float(row[ACAC_COLS["j_p"]])
        obj.j_q = float(row[ACAC_COLS["j_q"]])
        obj.i_i = float(row[ACAC_COLS["i_i"]])
        obj.j_i = float(row[ACAC_COLS["j_i"]])
        obj.is_alive = False
        objects.append(obj)
    return out, names, objects


def build_hybrid_ppc_from_e_file(file_path):
    rows = _read_efile_rows(file_path)
    return build_hybrid_ppc_from_efile_rows(file_path, rows)


def build_hybrid_ppc_only_from_efile_rows(file_path, rows):
    ac_ppc = build_ac_ppc_from_efile_rows(file_path, rows)
    dc_ppc = build_dc_ppc_from_efile_rows(file_path, rows)
    ac_ppc["source"] = str(file_path)
    dc_ppc["source"] = str(file_path)
    vbase_maps = _raw_vbase_maps(ac_ppc, dc_ppc)
    dcac, dcac_name, _dcac_objects = _build_dcac_from_rows(
        rows, ac_ppc, dc_ppc, build_objects=False, vbase_maps=vbase_maps
    )
    acac, acac_name, _acac_objects = _build_acac_from_rows(
        rows, ac_ppc, dc_ppc, build_objects=False, vbase_maps=vbase_maps
    )
    return {
        "format": "hybrid_ppc_v1",
        "source": str(file_path),
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


def build_hybrid_ppc_from_efile_rows(file_path, rows):
    ac_ppc = build_ac_ppc_from_efile_rows(file_path, rows)
    dc_ppc = build_dc_ppc_from_efile_rows(file_path, rows)
    ac_network = _build_ac_network_from_ppc(ac_ppc)
    dc_network = _build_dc_network_from_ppc(dc_ppc)
    ac_ppc["source"] = str(file_path)
    dc_ppc["source"] = str(file_path)
    ac_network.ppc = ac_ppc
    dc_network.ppc = dc_ppc
    vbase_maps = _raw_vbase_maps(ac_ppc, dc_ppc)
    dcac, dcac_name, dcac_objects = _build_dcac_from_rows(rows, ac_ppc, dc_ppc, vbase_maps=vbase_maps)
    acac, acac_name, acac_objects = _build_acac_from_rows(rows, ac_ppc, dc_ppc, vbase_maps=vbase_maps)
    ppc = {
        "format": "hybrid_ppc_v1",
        "source": str(file_path),
        "base": ac_ppc["base"],
        "ac": ac_ppc,
        "dc": dc_ppc,
        "ac_network": ac_network,
        "dc_network": dc_network,
        "dcac": dcac,
        "acac": acac,
        "dcac_name": dcac_name,
        "acac_name": acac_name,
        "dcac_objects": dcac_objects,
        "acac_objects": acac_objects,
        "dcac_cols": DCAC_COLS,
        "acac_cols": ACAC_COLS,
    }
    network = build_hybrid_model_from_ppc(ppc)
    return network, ppc


def build_hybrid_ppc_from_model(file_path, model):
    normalize_model_named_units(model)
    ac_network, ac_ppc = _build_ac_ppc_from_model(model, units_already_normalized=True)
    ac_ppc["source"] = str(file_path)
    ac_network.ppc = ac_ppc
    dc_network, dc_ppc = _build_dc_ppc_from_model(model, units_already_normalized=True)
    dc_network.ppc = dc_ppc
    dcac, dcac_name, dcac_objects = _build_dcac_with_objects(model)
    acac, acac_name, acac_objects = _build_acac_with_objects(model)
    ppc = {
        "format": "hybrid_ppc_v1",
        "source": str(file_path),
        "base": ac_ppc["base"],
        "ac": ac_ppc,
        "dc": dc_ppc,
        "ac_network": ac_network,
        "dc_network": dc_network,
        "dcac": dcac,
        "acac": acac,
        "dcac_name": dcac_name,
        "acac_name": acac_name,
        "dcac_objects": dcac_objects,
        "acac_objects": acac_objects,
        "dcac_cols": DCAC_COLS,
        "acac_cols": ACAC_COLS,
    }
    network = build_hybrid_model_from_ppc(ppc)
    return network, ppc



def build_hybrid_model_from_ppc(ppc: Dict):
    if __package__ == "model":
        from model.hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork
    else:
        from hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork

    ac_network = ppc["ac_network"]
    dc_network = ppc["dc_network"]
    dcac = list(ppc.get("dcac_objects") or ())
    if not dcac:
        dcac = [
            DCACConverter(
                int(row[DCAC_COLS["idx"]]),
                int(row[DCAC_COLS["ac_node"]]),
                int(row[DCAC_COLS["dc_node"]]),
                float(row[DCAC_COLS["r1"]]),
                float(row[DCAC_COLS["r2"]]),
                DCAC_CONTROL_LABEL.get(int(row[DCAC_COLS["control_type"]]), "DCV"),
                float(row[DCAC_COLS["p_ac_set"]]),
                float(row[DCAC_COLS["q_ac_set"]]),
                float(row[DCAC_COLS["v_ac_set"]]),
                float(row[DCAC_COLS["v_dc_set"]]),
                int(row[DCAC_COLS["run_stat"]]),
            )
            for pos, row in enumerate(ppc["dcac"])
        ]
        for pos, conv in enumerate(dcac):
            conv.name = str(ppc["dcac_name"][pos])
            row = ppc["dcac"][pos]
            conv.dc_p = float(row[DCAC_COLS["dc_p"]])
            conv.ac_p = float(row[DCAC_COLS["ac_p"]])
            conv.ac_q = float(row[DCAC_COLS["ac_q"]])
            conv.dc_i = float(row[DCAC_COLS["dc_i"]])
            conv.ac_i = float(row[DCAC_COLS["ac_i"]])
            conv.is_alive = False
    acac = list(ppc.get("acac_objects") or ())
    if not acac:
        acac = [
            ACACConverter(
                int(row[ACAC_COLS["idx"]]),
                int(row[ACAC_COLS["i_node"]]),
                int(row[ACAC_COLS["j_node"]]),
                float(row[ACAC_COLS["r1"]]),
                float(row[ACAC_COLS["r2"]]),
                ACAC_CONTROL_LABEL.get(int(row[ACAC_COLS["control_type"]]), "PQQ"),
                float(row[ACAC_COLS["p_set"]]),
                float(row[ACAC_COLS["i_q_set"]]),
                float(row[ACAC_COLS["j_q_set"]]),
                float(row[ACAC_COLS["i_v_set"]]),
                float(row[ACAC_COLS["j_v_set"]]),
                int(row[ACAC_COLS["run_stat"]]),
            )
            for pos, row in enumerate(ppc["acac"])
        ]
        for pos, conv in enumerate(acac):
            conv.name = str(ppc["acac_name"][pos])
            row = ppc["acac"][pos]
            conv.i_p = float(row[ACAC_COLS["i_p"]])
            conv.i_q = float(row[ACAC_COLS["i_q"]])
            conv.j_p = float(row[ACAC_COLS["j_p"]])
            conv.j_q = float(row[ACAC_COLS["j_q"]])
            conv.i_i = float(row[ACAC_COLS["i_i"]])
            conv.j_i = float(row[ACAC_COLS["j_i"]])
            conv.is_alive = False

    model = HybridPowerNetwork(ac=ac_network, dc=dc_network, dcac_converters=dcac, acac_converters=acac)
    base = ppc["base"]
    model.p_base = float(base["p_base"])
    model.u_scale = float(base["u_scale"])
    model.p_scale = float(base["p_scale"])
    model.i_scale = float(base["i_scale"])
    model.p_base_kW = float(base["p_base_kW"])
    model.ac.ppc = ppc["ac"]
    model.dc.ppc = ppc["dc"]
    model.ppc = ppc
    model._ac_ppc = ppc["ac"]
    model._dc_ppc = ppc["dc"]
    model.ACNode = ac_network.nodes
    model.ACBranch = ac_network.branches
    model.ACLoad = ac_network.loads
    model.ACGenerator = ac_network.generators
    model.ACZeroBranch = ac_network.zero_branches
    model.ACSwitch = ac_network.switches
    model.ACBreak = getattr(ac_network, "breakers", [])
    model.ACTransformer = ac_network.transformers
    model.ACShuntCompensator = ac_network.shunt_compensators
    model.DCNode = dc_network.nodes
    model.DCBranch = dc_network.branches
    model.DCLoad = dc_network.loads
    model.DCGenerator = dc_network.generators
    model.DCZeroBranch = dc_network.zero_branches
    model.DCSwitch = dc_network.switches
    model.DCBreak = getattr(dc_network, "breakers", [])
    model.DCDCConverter = dc_network.dcdc_converters
    model.DCShuntCompensator = []
    model.DCACConverter = dcac
    model.ACACConverter = acac
    return model
