import threading
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

from efile_read import _read_efile_rows, efile_factory_from_file, efile_factory_from_rows
from paths import resolve_project_file
from unit_system import dc_current_base_ka

MODEL_DIR = Path(__file__).resolve().parent
for path in (MODEL_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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
    "pbase": 2,
    "pv0": 3,
    "pv1": 4,
    "pv2": 5,
    "run_stat": 6,
    "p": 7,
    "current": 8,
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
BREAK_COLS = SWITCH_COLS
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


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def clear_dc_ppc_cache(file_path=None) -> None:
    with _DC_PPC_CACHE_LOCK:
        if file_path is None:
            _DC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _DC_PPC_CACHE.pop(path, None)

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
    # Inline the cell extraction to skip the `_float_cell`/`_cell` call
    # overhead — this loop runs hundreds of thousands of times during prepare
    # for large grids.
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
        [_code_value(_cell(row, col, default_label), mapping, default_label) for row in table_rows],
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


def _base_from_rows(data: Dict) -> Tuple[float, float, float, float, float]:
    columns, table_rows = _rows_for(data, "PowerBase")
    if not table_rows:
        raise RuntimeError("E file must define <PowerBase> with p_base, u_scale, p_scale, and i_scale")
    row = table_rows[0]
    required = {}
    for attr in ("p_base", "u_scale", "p_scale", "i_scale"):
        if attr not in columns:
            raise RuntimeError("E file <PowerBase> must define p_base, u_scale, p_scale, and i_scale")
        value = float(_cell(row, columns[attr], 0.0))
        if value <= 0.0:
            raise RuntimeError(f"Invalid {attr} in <PowerBase>: {value}")
        required[attr] = value
    p_base = required["p_base"]
    p_scale = required["p_scale"]
    return p_base, required["u_scale"], p_scale, required["i_scale"], p_base / p_scale


def _scale_by_node(node_values: np.ndarray, scales_by_idx: Dict[int, float]) -> np.ndarray:
    return np.asarray([scales_by_idx.get(int(node), 1.0) for node in node_values], dtype=np.float64)


def _raw_vbase_by_node(node_values: np.ndarray, raw_vbase_by_idx: Dict[int, float]) -> np.ndarray:
    return np.asarray([raw_vbase_by_idx.get(int(node), 0.0) for node in node_values], dtype=np.float64)


def _assign_power_if_present(out: np.ndarray, col: int, table_rows, columns, attr: str, p_base: float) -> None:
    if attr in columns:
        out[:, col] = _float_column(table_rows, columns, attr) / p_base


def _assign_current_if_present(
    out: np.ndarray,
    col: int,
    table_rows,
    columns,
    attr: str,
    node_values: np.ndarray,
    current_scale_by_node: Dict[int, float],
) -> None:
    if attr not in columns:
        return
    scales = _scale_by_node(node_values.astype(np.int64, copy=False), current_scale_by_node)
    raw = _float_column(table_rows, columns, attr)
    out[:, col] = np.divide(raw, scales, out=np.zeros_like(raw), where=np.abs(scales) > 1e-12)


def _voltage_set_column(table_rows, columns, attr: str, node_values: np.ndarray, raw_vbase_by_idx: Dict[int, float]) -> np.ndarray:
    if attr not in columns:
        return np.ones(len(table_rows), dtype=np.float64)
    raw = _float_column(table_rows, columns, attr, 1.0)
    raw_vbase = _raw_vbase_by_node(node_values.astype(np.int64, copy=False), raw_vbase_by_idx)
    return np.divide(raw, raw_vbase, out=np.ones_like(raw), where=np.abs(raw_vbase) > 1e-12)


def _build_switch_like_from_rows(
    table_rows,
    columns,
    current_scale_by_node: Dict[int, float],
    *,
    prefix: str,
    scale_optional_power: bool = True,
    p_base: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(table_rows), len(SWITCH_COLS)), dtype=np.float64)
    if not table_rows:
        return out, np.asarray([], dtype=object)
    out[:, SWITCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
    out[:, SWITCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
    out[:, SWITCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
    out[:, SWITCH_COLS["status"]] = _float_column(table_rows, columns, "status", 1.0)
    out[:, SWITCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    if scale_optional_power:
        _assign_power_if_present(out, SWITCH_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(
            out,
            SWITCH_COLS["current"],
            table_rows,
            columns,
            "current",
            out[:, SWITCH_COLS["i_node"]],
            current_scale_by_node,
        )
    else:
        if "p" in columns:
            out[:, SWITCH_COLS["p"]] = _float_column(table_rows, columns, "p")
        if "current" in columns:
            out[:, SWITCH_COLS["current"]] = _float_column(table_rows, columns, "current")
    return out, _names_from_rows(table_rows, columns, prefix, out[:, SWITCH_COLS["idx"]])


def _build_dc_ppc_from_rows_dict(rows: Dict, source) -> Dict:
    p_base, u_scale, p_scale, i_scale, p_base_kW = _base_from_rows(rows)

    columns, table_rows = _rows_for(rows, "DCNode")
    bus = np.zeros((len(table_rows), len(BUS_COLS)), dtype=np.float64)
    if table_rows:
        raw_vbase = _float_column(table_rows, columns, "vbase")
        bus[:, BUS_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        bus[:, BUS_COLS["vbase"]] = raw_vbase / u_scale
        raw_voltage = _float_column(table_rows, columns, "voltage", 1.0)
        bus[:, BUS_COLS["voltage"]] = np.divide(
            raw_voltage,
            raw_vbase,
            out=np.ones_like(raw_voltage),
            where=np.abs(raw_vbase) > 1e-12,
        )
        bus[:, BUS_COLS["isl"]] = _float_column(table_rows, columns, "isl", 0.0)
        bus[:, BUS_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    bus_names = _names_from_rows(table_rows, columns, "bus", bus[:, BUS_COLS["idx"]])
    raw_vbase_by_idx = {
        int(idx): float(vbase)
        for idx, vbase in zip(bus[:, BUS_COLS["idx"]], raw_vbase if table_rows else [])
    }
    current_scale_by_node = {
        int(row[BUS_COLS["idx"]]): i_scale * dc_current_base_ka(p_base_kW, float(row[BUS_COLS["vbase"]]))
        for row in bus
    }

    columns, table_rows = _rows_for(rows, "DCBranch")
    branch = np.zeros((len(table_rows), len(BRANCH_COLS)), dtype=np.float64)
    if table_rows:
        branch[:, BRANCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        branch[:, BRANCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        branch[:, BRANCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        branch[:, BRANCH_COLS["r"]] = _float_column(table_rows, columns, "r")
        branch[:, BRANCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(branch, BRANCH_COLS["i_p"], table_rows, columns, "i_p", p_base)
        _assign_power_if_present(branch, BRANCH_COLS["j_p"], table_rows, columns, "j_p", p_base)
        _assign_current_if_present(branch, BRANCH_COLS["current"], table_rows, columns, "current", branch[:, BRANCH_COLS["i_node"]], current_scale_by_node)
    branch_names = _names_from_rows(table_rows, columns, "branch", branch[:, BRANCH_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "DCLoad")
    load = np.zeros((len(table_rows), len(LOAD_COLS)), dtype=np.float64)
    if table_rows:
        load[:, LOAD_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        load[:, LOAD_COLS["node"]] = _int_column(table_rows, columns, "node")
        load[:, LOAD_COLS["pbase"]] = _float_column(table_rows, columns, "pbase", 1.0) / p_base
        load[:, LOAD_COLS["pv0"]] = _float_column(table_rows, columns, "pv0")
        load[:, LOAD_COLS["pv1"]] = _float_column(table_rows, columns, "pv1")
        load[:, LOAD_COLS["pv2"]] = _float_column(table_rows, columns, "pv2")
        load[:, LOAD_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(load, LOAD_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(load, LOAD_COLS["current"], table_rows, columns, "current", load[:, LOAD_COLS["node"]], current_scale_by_node)
    load_names = _names_from_rows(table_rows, columns, "load", load[:, LOAD_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "DCGenerator")
    gen = np.zeros((len(table_rows), len(GEN_COLS)), dtype=np.float64)
    if table_rows:
        gen[:, GEN_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        gen[:, GEN_COLS["node"]] = _int_column(table_rows, columns, "node")
        gen[:, GEN_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", CTRL_CODE, "P")
        gen[:, GEN_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
        gen[:, GEN_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", gen[:, GEN_COLS["node"]], raw_vbase_by_idx)
        _assign_current_if_present(gen, GEN_COLS["i_set"], table_rows, columns, "i_set", gen[:, GEN_COLS["node"]], current_scale_by_node)
        gen[:, GEN_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(gen, GEN_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(gen, GEN_COLS["current"], table_rows, columns, "current", gen[:, GEN_COLS["node"]], current_scale_by_node)
    gen_names = _names_from_rows(table_rows, columns, "gen", gen[:, GEN_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "DCZeroBranch")
    zero_branch = np.zeros((len(table_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    if table_rows:
        zero_branch[:, ZERO_BRANCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        zero_branch[:, ZERO_BRANCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        zero_branch[:, ZERO_BRANCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(zero_branch, ZERO_BRANCH_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(zero_branch, ZERO_BRANCH_COLS["current"], table_rows, columns, "current", zero_branch[:, ZERO_BRANCH_COLS["i_node"]], current_scale_by_node)
    zero_branch_names = _names_from_rows(table_rows, columns, "zero_branch", zero_branch[:, ZERO_BRANCH_COLS["idx"]])

    sw_columns, sw_rows = _rows_for(rows, "DCSwitch")
    switch, switch_names = _build_switch_like_from_rows(sw_rows, sw_columns, current_scale_by_node, prefix="switch", p_base=p_base)
    br_columns, br_rows = _rows_for(rows, "DCBreak")
    breaker, breaker_names = _build_switch_like_from_rows(
        br_rows,
        br_columns,
        current_scale_by_node,
        prefix="break",
        scale_optional_power=False,
        p_base=p_base,
    )

    columns, table_rows = _rows_for(rows, "DCDCConverter")
    dcdc = np.zeros((len(table_rows), len(DCDC_COLS)), dtype=np.float64)
    if table_rows:
        dcdc[:, DCDC_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        dcdc[:, DCDC_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        dcdc[:, DCDC_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        dcdc[:, DCDC_COLS["r1"]] = _float_column(table_rows, columns, "r1")
        dcdc[:, DCDC_COLS["r2"]] = _float_column(table_rows, columns, "r2")
        dcdc[:, DCDC_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", CTRL_CODE, "P")
        dcdc[:, DCDC_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
        _assign_current_if_present(dcdc, DCDC_COLS["i_set"], table_rows, columns, "i_set", dcdc[:, DCDC_COLS["i_node"]], current_scale_by_node)
        dcdc[:, DCDC_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", dcdc[:, DCDC_COLS["i_node"]], raw_vbase_by_idx)
        dcdc[:, DCDC_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(dcdc, DCDC_COLS["i_p"], table_rows, columns, "i_p", p_base)
        _assign_power_if_present(dcdc, DCDC_COLS["j_p"], table_rows, columns, "j_p", p_base)
        _assign_current_if_present(dcdc, DCDC_COLS["i_c"], table_rows, columns, "i_c", dcdc[:, DCDC_COLS["i_node"]], current_scale_by_node)
        _assign_current_if_present(dcdc, DCDC_COLS["j_c"], table_rows, columns, "j_c", dcdc[:, DCDC_COLS["j_node"]], current_scale_by_node)
    dcdc_names = _names_from_rows(table_rows, columns, "dcdc", dcdc[:, DCDC_COLS["idx"]])

    ppc = {
        "format": "dc_ppc_v1",
        "source": str(source),
        "base": {
            "p_base": p_base,
            "u_scale": u_scale,
            "p_scale": p_scale,
            "i_scale": i_scale,
            "p_base_kW": p_base_kW,
        },
        "bus": bus,
        "branch": branch,
        "load": load,
        "gen": gen,
        "zero_branch": zero_branch,
        "switch": switch,
        "break": breaker,
        "dcdc": dcdc,
        "node_pos": _node_maps(bus) if len(bus) else {},
        "bus_name": bus_names,
        "branch_name": branch_names,
        "load_name": load_names,
        "gen_name": gen_names,
        "zero_branch_name": zero_branch_names,
        "switch_name": switch_names,
        "break_name": breaker_names,
        "dcdc_name": dcdc_names,
    }
    return ppc


class _ArrayDevice(SimpleNamespace):
    __hash__ = object.__hash__


def _device(
    idx,
    name,
    **values,
):
    obj = _ArrayDevice(idx=int(idx), name=str(name), **values)
    obj.is_alive = False
    return obj


def _node_maps(bus: np.ndarray):
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    return {int(node_id): pos for pos, node_id in enumerate(node_ids)}


def build_dc_ppc_from_e_file(file_path) -> Dict:
    """Build a DC ppc from an E file through the shared model factory."""
    file_key = _file_cache_key(file_path)
    with _DC_PPC_CACHE_LOCK:
        cached = _DC_PPC_CACHE.get(file_key[0])
        if cached is not None and cached[0] == file_key:
            return cached[1]

    ppc = _build_dc_ppc_from_rows_dict(_read_efile_rows(file_key[0]), file_key[0])
    ppc["source"] = str(file_key[0])
    with _DC_PPC_CACHE_LOCK:
        _DC_PPC_CACHE[file_key[0]] = (file_key, ppc)
    return ppc


def build_dc_ppc_from_efile_rows(file_path, rows) -> Dict:
    """Build DC ppc from E rows that are already loaded in memory."""
    path = resolve_project_file(file_path).resolve()
    return _build_dc_ppc_from_rows_dict(rows, path)


def _value(obj, attr: str, default=0.0):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_value(obj, attr: str, default: float = 0.0) -> float:
    return float(_value(obj, attr, default))


def _int_value(obj, attr: str, default: int = 0) -> int:
    return int(float(_value(obj, attr, default)))


def _name_array(devices, prefix: str) -> np.ndarray:
    return np.asarray(
        [str(getattr(dev, "name", "") or f"{prefix}_{_int_value(dev, 'idx', pos)}") for pos, dev in enumerate(devices)],
        dtype=object,
    )


def _code_value(value, mapping: Dict[str, int], default_label: str) -> int:
    if value in (None, ""):
        return mapping[default_label]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return mapping.get(str(value).upper(), mapping[default_label])


_VALUE_FLOAT = 0
_VALUE_INT = 1
_VALUE_CTRL = 2


def _has_value(devices, attr: str) -> bool:
    return any(getattr(dev, attr, None) not in (None, "") for dev in devices)


def _device_array(devices, width: int, specs, optional_specs=()) -> np.ndarray:
    out = np.zeros((len(devices), width), dtype=np.float64)
    for row, dev in enumerate(devices):
        values = getattr(dev, "__dict__", {})
        for col, attr, default, kind in specs:
            value = values[attr] if attr in values else getattr(dev, attr, default)
            if value is None or value == "":
                value = default
            if kind == _VALUE_INT:
                out[row, col] = int(float(value))
            elif kind == _VALUE_CTRL:
                out[row, col] = _code_value(value, CTRL_CODE, default)
            else:
                out[row, col] = float(value)
    for col, attr, default, kind in optional_specs:
        if not _has_value(devices, attr):
            continue
        for row, dev in enumerate(devices):
            values = getattr(dev, "__dict__", {})
            value = values[attr] if attr in values else getattr(dev, attr, default)
            if value is None or value == "":
                value = default
            if kind == _VALUE_INT:
                out[row, col] = int(float(value))
            elif kind == _VALUE_CTRL:
                out[row, col] = _code_value(value, CTRL_CODE, default)
            else:
                out[row, col] = float(value)
    return out


def build_dc_ppc_from_network(network) -> Dict:
    """Build a DC ppc dictionary from an already loaded DCPowerNetwork."""
    nodes = list(getattr(network, "nodes", []))
    branches = list(getattr(network, "branches", []))
    loads = list(getattr(network, "loads", []))
    generators = list(getattr(network, "generators", []))
    zero_branches = list(getattr(network, "zero_branches", []))
    switches = list(getattr(network, "switches", []))
    breakers = list(getattr(network, "breakers", []))
    dcdcs = list(getattr(network, "dcdc_converters", []))

    p_base = float(getattr(network, "p_base", 1.0))
    u_scale = float(getattr(network, "u_scale", 1.0))
    p_scale = float(getattr(network, "p_scale", 1.0))
    i_scale = float(getattr(network, "i_scale", 1.0))
    p_base_kw = float(getattr(network, "p_base_kW", p_base / p_scale))

    bus = _device_array(
        nodes,
        len(BUS_COLS),
        (
            (BUS_COLS["idx"], "idx", 0, _VALUE_INT),
            (BUS_COLS["vbase"], "vbase", 0.0, _VALUE_FLOAT),
            (BUS_COLS["voltage"], "voltage", 1.0, _VALUE_FLOAT),
            (BUS_COLS["isl"], "isl", 0.0, _VALUE_FLOAT),
            (BUS_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
    )

    branch = _device_array(
        branches,
        len(BRANCH_COLS),
        (
            (BRANCH_COLS["idx"], "idx", 0, _VALUE_INT),
            (BRANCH_COLS["i_node"], "i_node", 0, _VALUE_INT),
            (BRANCH_COLS["j_node"], "j_node", 0, _VALUE_INT),
            (BRANCH_COLS["r"], "r", 0.0, _VALUE_FLOAT),
            (BRANCH_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
        (
            (BRANCH_COLS["i_p"], "i_p", 0.0, _VALUE_FLOAT),
            (BRANCH_COLS["j_p"], "j_p", 0.0, _VALUE_FLOAT),
            (BRANCH_COLS["current"], "current", 0.0, _VALUE_FLOAT),
        ),
    )

    load = _device_array(
        loads,
        len(LOAD_COLS),
        (
            (LOAD_COLS["idx"], "idx", 0, _VALUE_INT),
            (LOAD_COLS["node"], "node", 0, _VALUE_INT),
            (LOAD_COLS["pbase"], "pbase", 1.0, _VALUE_FLOAT),
            (LOAD_COLS["pv0"], "pv0", 0.0, _VALUE_FLOAT),
            (LOAD_COLS["pv1"], "pv1", 0.0, _VALUE_FLOAT),
            (LOAD_COLS["pv2"], "pv2", 0.0, _VALUE_FLOAT),
            (LOAD_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
        (
            (LOAD_COLS["p"], "p", 0.0, _VALUE_FLOAT),
            (LOAD_COLS["current"], "current", 0.0, _VALUE_FLOAT),
        ),
    )

    gen = _device_array(
        generators,
        len(GEN_COLS),
        (
            (GEN_COLS["idx"], "idx", 0, _VALUE_INT),
            (GEN_COLS["node"], "node", 0, _VALUE_INT),
            (GEN_COLS["control_type"], "control_type", "P", _VALUE_CTRL),
            (GEN_COLS["p_set"], "p_set", 0.0, _VALUE_FLOAT),
            (GEN_COLS["v_set"], "v_set", 1.0, _VALUE_FLOAT),
            (GEN_COLS["i_set"], "i_set", 0.0, _VALUE_FLOAT),
            (GEN_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
        (
            (GEN_COLS["p"], "p", 0.0, _VALUE_FLOAT),
            (GEN_COLS["current"], "current", 0.0, _VALUE_FLOAT),
        ),
    )

    zero_branch = _device_array(
        zero_branches,
        len(ZERO_BRANCH_COLS),
        (
            (ZERO_BRANCH_COLS["idx"], "idx", 0, _VALUE_INT),
            (ZERO_BRANCH_COLS["i_node"], "i_node", 0, _VALUE_INT),
            (ZERO_BRANCH_COLS["j_node"], "j_node", 0, _VALUE_INT),
            (ZERO_BRANCH_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
        (
            (ZERO_BRANCH_COLS["p"], "p", 0.0, _VALUE_FLOAT),
            (ZERO_BRANCH_COLS["current"], "current", 0.0, _VALUE_FLOAT),
        ),
    )

    def build_switch_like(devices):
        return _device_array(
            devices,
            len(SWITCH_COLS),
            (
                (SWITCH_COLS["idx"], "idx", 0, _VALUE_INT),
                (SWITCH_COLS["i_node"], "i_node", 0, _VALUE_INT),
                (SWITCH_COLS["j_node"], "j_node", 0, _VALUE_INT),
                (SWITCH_COLS["status"], "status", 1.0, _VALUE_FLOAT),
                (SWITCH_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
            ),
            (
                (SWITCH_COLS["p"], "p", 0.0, _VALUE_FLOAT),
                (SWITCH_COLS["current"], "current", 0.0, _VALUE_FLOAT),
            ),
        )

    dcdc = _device_array(
        dcdcs,
        len(DCDC_COLS),
        (
            (DCDC_COLS["idx"], "idx", 0, _VALUE_INT),
            (DCDC_COLS["i_node"], "i_node", 0, _VALUE_INT),
            (DCDC_COLS["j_node"], "j_node", 0, _VALUE_INT),
            (DCDC_COLS["r1"], "r1", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["r2"], "r2", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["control_type"], "control_type", "P", _VALUE_CTRL),
            (DCDC_COLS["p_set"], "p_set", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["i_set"], "i_set", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["v_set"], "v_set", 1.0, _VALUE_FLOAT),
            (DCDC_COLS["run_stat"], "run_stat", 1.0, _VALUE_FLOAT),
        ),
        (
            (DCDC_COLS["i_p"], "i_p", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["j_p"], "j_p", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["i_c"], "i_c", 0.0, _VALUE_FLOAT),
            (DCDC_COLS["j_c"], "j_c", 0.0, _VALUE_FLOAT),
        ),
    )

    ppc = {
        "format": "dc_ppc_v1",
        "base": {
            "p_base": p_base,
            "u_scale": u_scale,
            "p_scale": p_scale,
            "i_scale": i_scale,
            "p_base_kW": p_base_kw,
        },
        "bus": bus,
        "branch": branch,
        "load": load,
        "gen": gen,
        "zero_branch": zero_branch,
        "switch": build_switch_like(switches),
        "break": build_switch_like(breakers),
        "dcdc": dcdc,
        "node_pos": _node_maps(bus) if len(bus) else {},
    }
    ppc.update(
        bus_name=_name_array(nodes, "bus"),
        branch_name=_name_array(branches, "branch"),
        load_name=_name_array(loads, "load"),
        gen_name=_name_array(generators, "gen"),
        zero_branch_name=_name_array(zero_branches, "zero_branch"),
        switch_name=_name_array(switches, "switch"),
        break_name=_name_array(breakers, "break"),
        dcdc_name=_name_array(dcdcs, "dcdc"),
    )
    return ppc


def _build_dc_ppc_from_model(model, *, units_already_normalized: bool = False):
    from dc_model import DCPowerNetwork as ObjectDCPowerNetwork

    network = ObjectDCPowerNetwork()
    network.model = model
    network._load_from_model(units_already_normalized=units_already_normalized)
    ppc = build_dc_ppc_from_network(network)
    network.ppc = ppc
    return network, ppc


def build_dc_ppc_from_model(model):
    return _build_dc_ppc_from_model(model)




def build_dc_network_from_ppc(ppc: Dict):
    network = DCPowerNetwork()
    network.ppc = ppc
    base = ppc["base"]
    network.p_base = float(base["p_base"])
    network.p_base_kW = float(base["p_base_kW"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network._load_objects_from_ppc(ppc)
    return network


class DCIsl:
    def __init__(self, idx, is_alive):
        self.idx = idx
        self.is_alive = is_alive
        self.buses = []
        self.gens = []
        self.loads = []
        self.branches = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.slack_nodes = []
        self.v_gens = []
        self.v_dcdcs = []


class DCPowerNetwork:
    """Object-compatible DC network facade backed by dc_ppc_v1 arrays."""

    def __init__(self):
        self.ppc = None
        self.nodes = []
        self.branches = []
        self.loads = []
        self.generators = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.buses = []
        self.islands = []
        self.node_dict = {}
        self.bus_dict = {}
        self.node_to_bus = {}
        self.switch_dict = {}
        self.break_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branche_dict = {}
        self.branche_dict = {}
        self.dcdc_converter_dict = {}

    def add_node(self, idx, vbase, voltage=1.0, run_stat=1):
        node = _device(idx, f"nd_{idx}", vbase=float(vbase), voltage=float(voltage), run_stat=int(run_stat))
        node.isl = None
        node.isl_obj = None
        node.v_set = 1.0
        node.v_gens = []
        node.v_dcdcs = []
        node.is_slack = False
        node.bus = None
        node.bus_obj = None
        self.nodes.append(node)
        return node

    def add_branch(self, idx, i_node, j_node, r, run_stat=1):
        br = _device(
            idx,
            f"br_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            r=float(r),
            run_stat=int(run_stat),
            current=None,
            i_p=None,
            j_p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.branches.append(br)
        return br

    def add_load(self, idx, node, pbase, pv0, pv1, pv2, run_stat=1):
        ld = _device(
            idx,
            f"load_{idx}",
            node=int(node),
            pbase=float(pbase),
            pv0=float(pv0),
            pv1=float(pv1),
            pv2=float(pv2),
            run_stat=int(run_stat),
            p=None,
            current=None,
            node_obj=None,
        )
        self.loads.append(ld)
        return ld

    def add_generator(self, idx, node, control_type, p_set, v_set, i_set, run_stat=1):
        gen = _device(
            idx,
            f"gen_{idx}",
            node=int(node),
            control_type=str(control_type),
            p_set=float(p_set),
            v_set=float(v_set),
            i_set=float(i_set),
            run_stat=int(run_stat),
            p=None,
            current=None,
            node_obj=None,
        )
        self.generators.append(gen)
        return gen

    def add_zero_branch(self, idx, i_node, j_node, run_stat=1):
        zbr = _device(
            idx,
            f"zbr_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.zero_branches.append(zbr)
        return zbr

    def add_switch(self, idx, i_node, j_node, status, run_stat=1):
        sw = _device(
            idx,
            f"sw_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            status=int(status),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.switches.append(sw)
        return sw

    def add_break(self, idx, i_node, j_node, status, run_stat=1):
        brk = _device(
            idx,
            f"brk_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            status=int(status),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.breakers.append(brk)
        return brk

    def add_dcdc_converter(self, idx, i_node, j_node, r1, r2, control_type, p_set, i_set, v_set, run_stat=1):
        conv = _device(
            idx,
            f"dcdc_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            r1=float(r1),
            r2=float(r2),
            control_type=str(control_type),
            p_set=float(p_set),
            i_set=float(i_set),
            v_set=float(v_set),
            run_stat=int(run_stat),
            i_p=None,
            j_p=None,
            i_c=None,
            j_c=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.dcdc_converters.append(conv)
        return conv

    def read_from_file(self, file_name):
        self.ppc = build_dc_ppc_from_e_file(file_name)
        base = self.ppc["base"]
        self.p_base = float(base["p_base"])
        self.p_base_kW = float(base["p_base_kW"])
        self.u_scale = float(base["u_scale"])
        self.p_scale = float(base["p_scale"])
        self.i_scale = float(base["i_scale"])
        self._load_objects_from_ppc(self.ppc)

    def _load_objects_from_ppc(self, ppc):
        ctrl_label = {CTRL_P: "P", CTRL_V: "V", CTRL_I: "I"}
        self.nodes = []
        for row, name in zip(ppc["bus"], ppc.get("bus_name", [])):
            node = _device(
                row[BUS_COLS["idx"]],
                name,
                vbase=float(row[BUS_COLS["vbase"]]),
                voltage=float(row[BUS_COLS["voltage"]]),
                run_stat=int(row[BUS_COLS["run_stat"]]),
            )
            node.isl = None
            node.isl_obj = None
            node.v_set = 1.0
            node.v_gens = []
            node.v_dcdcs = []
            node.is_slack = False
            node.bus = None
            node.bus_obj = None
            self.nodes.append(node)

        self.branches = []
        for row, name in zip(ppc["branch"], ppc.get("branch_name", [])):
            self.branches.append(
                _device(
                    row[BRANCH_COLS["idx"]],
                    name,
                    i_node=int(row[BRANCH_COLS["i_node"]]),
                    j_node=int(row[BRANCH_COLS["j_node"]]),
                    r=float(row[BRANCH_COLS["r"]]),
                    run_stat=int(row[BRANCH_COLS["run_stat"]]),
                    i_p=float(row[BRANCH_COLS["i_p"]]),
                    j_p=float(row[BRANCH_COLS["j_p"]]),
                    current=float(row[BRANCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.loads = []
        for row, name in zip(ppc["load"], ppc.get("load_name", [])):
            self.loads.append(
                _device(
                    row[LOAD_COLS["idx"]],
                    name,
                    node=int(row[LOAD_COLS["node"]]),
                    pbase=float(row[LOAD_COLS["pbase"]]),
                    pv0=float(row[LOAD_COLS["pv0"]]),
                    pv1=float(row[LOAD_COLS["pv1"]]),
                    pv2=float(row[LOAD_COLS["pv2"]]),
                    run_stat=int(row[LOAD_COLS["run_stat"]]),
                    p=float(row[LOAD_COLS["p"]]),
                    current=float(row[LOAD_COLS["current"]]),
                    node_obj=None,
                )
            )

        self.generators = []
        for row, name in zip(ppc["gen"], ppc.get("gen_name", [])):
            self.generators.append(
                _device(
                    row[GEN_COLS["idx"]],
                    name,
                    node=int(row[GEN_COLS["node"]]),
                    control_type=ctrl_label[int(row[GEN_COLS["control_type"]])],
                    p_set=float(row[GEN_COLS["p_set"]]),
                    v_set=float(row[GEN_COLS["v_set"]]),
                    i_set=float(row[GEN_COLS["i_set"]]),
                    run_stat=int(row[GEN_COLS["run_stat"]]),
                    p=float(row[GEN_COLS["p"]]),
                    current=float(row[GEN_COLS["current"]]),
                    node_obj=None,
                )
            )

        self.zero_branches = []
        for row, name in zip(ppc["zero_branch"], ppc.get("zero_branch_name", [])):
            self.zero_branches.append(
                _device(
                    row[ZERO_BRANCH_COLS["idx"]],
                    name,
                    i_node=int(row[ZERO_BRANCH_COLS["i_node"]]),
                    j_node=int(row[ZERO_BRANCH_COLS["j_node"]]),
                    run_stat=int(row[ZERO_BRANCH_COLS["run_stat"]]),
                    p=float(row[ZERO_BRANCH_COLS["p"]]),
                    current=float(row[ZERO_BRANCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.switches = []
        for row, name in zip(ppc["switch"], ppc.get("switch_name", [])):
            self.switches.append(
                _device(
                    row[SWITCH_COLS["idx"]],
                    name,
                    i_node=int(row[SWITCH_COLS["i_node"]]),
                    j_node=int(row[SWITCH_COLS["j_node"]]),
                    status=int(row[SWITCH_COLS["status"]]),
                    run_stat=int(row[SWITCH_COLS["run_stat"]]),
                    p=float(row[SWITCH_COLS["p"]]),
                    current=float(row[SWITCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.breakers = []
        for row, name in zip(ppc.get("break", _empty(len(BREAK_COLS))), ppc.get("break_name", [])):
            self.breakers.append(
                _device(
                    row[BREAK_COLS["idx"]],
                    name,
                    i_node=int(row[BREAK_COLS["i_node"]]),
                    j_node=int(row[BREAK_COLS["j_node"]]),
                    status=int(row[BREAK_COLS["status"]]),
                    run_stat=int(row[BREAK_COLS["run_stat"]]),
                    p=float(row[BREAK_COLS["p"]]),
                    current=float(row[BREAK_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.dcdc_converters = []
        for row, name in zip(ppc["dcdc"], ppc.get("dcdc_name", [])):
            self.dcdc_converters.append(
                _device(
                    row[DCDC_COLS["idx"]],
                    name,
                    i_node=int(row[DCDC_COLS["i_node"]]),
                    j_node=int(row[DCDC_COLS["j_node"]]),
                    r1=float(row[DCDC_COLS["r1"]]),
                    r2=float(row[DCDC_COLS["r2"]]),
                    control_type=ctrl_label[int(row[DCDC_COLS["control_type"]])],
                    p_set=float(row[DCDC_COLS["p_set"]]),
                    i_set=float(row[DCDC_COLS["i_set"]]),
                    v_set=float(row[DCDC_COLS["v_set"]]),
                    run_stat=int(row[DCDC_COLS["run_stat"]]),
                    i_p=float(row[DCDC_COLS["i_p"]]),
                    j_p=float(row[DCDC_COLS["j_p"]]),
                    i_c=float(row[DCDC_COLS["i_c"]]),
                    j_c=float(row[DCDC_COLS["j_c"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

    def format_assoc(self):
        self.node_dict = {node.idx: node for node in self.nodes}
        self.switch_dict = {sw.idx: sw for sw in self.switches}
        self.break_dict = {brk.idx: brk for brk in self.breakers}
        self.load_dict = {ld.idx: ld for ld in self.loads}
        self.generator_dict = {gen.idx: gen for gen in self.generators}
        self.zero_branche_dict = {zbr.idx: zbr for zbr in self.zero_branches}
        self.branche_dict = {br.idx: br for br in self.branches}
        self.dcdc_converter_dict = {conv.idx: conv for conv in self.dcdc_converters}

        for node in self.nodes:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.dcdc_converters = []
            node.zero_branches = []
            node.is_alive = False
            node.bus = None
            node.bus_obj = None

        for gen in self.generators:
            gen.node_obj = self.node_dict.get(gen.node, None)
            if gen.node_obj:
                gen.node_obj.generators.append(gen)
        for ld in self.loads:
            ld.node_obj = self.node_dict.get(ld.node, None)
            if ld.node_obj:
                ld.node_obj.loads.append(ld)
        for br in self.branches:
            br.i_node_obj = self.node_dict.get(br.i_node, None)
            br.j_node_obj = self.node_dict.get(br.j_node, None)
            if br.i_node_obj:
                br.i_node_obj.branches.append(br)
            if br.j_node_obj:
                br.j_node_obj.branches.append(br)
        for sw in self.switches:
            sw.i_node_obj = self.node_dict.get(sw.i_node, None)
            sw.j_node_obj = self.node_dict.get(sw.j_node, None)
            if sw.i_node_obj:
                sw.i_node_obj.switches.append(sw)
            if sw.j_node_obj:
                sw.j_node_obj.switches.append(sw)
        for brk in self.breakers:
            brk.i_node_obj = self.node_dict.get(brk.i_node, None)
            brk.j_node_obj = self.node_dict.get(brk.j_node, None)
            if brk.i_node_obj:
                brk.i_node_obj.breakers.append(brk)
            if brk.j_node_obj:
                brk.j_node_obj.breakers.append(brk)
        for conv in self.dcdc_converters:
            conv.i_node_obj = self.node_dict.get(conv.i_node, None)
            conv.j_node_obj = self.node_dict.get(conv.j_node, None)
            if conv.i_node_obj:
                conv.i_node_obj.dcdc_converters.append(conv)
            if conv.j_node_obj:
                conv.j_node_obj.dcdc_converters.append(conv)
        for zbr in self.zero_branches:
            zbr.i_node_obj = self.node_dict.get(zbr.i_node, None)
            zbr.j_node_obj = self.node_dict.get(zbr.j_node, None)
            if zbr.i_node_obj:
                zbr.i_node_obj.zero_branches.append(zbr)
            if zbr.j_node_obj:
                zbr.j_node_obj.zero_branches.append(zbr)

    def topo(self):
        from model import topology as network_topology

        network_topology.prepare_dc_topology(self)

    def det_isl_alive_stat(self):
        for isl in self.islands:
            isl.is_alive = False
            isl.slack_nodes = []
            isl.v_gens = []
            isl.v_dcdcs = []
            isl.buses = []
            isl.gens = []
            isl.loads = []
            isl.branches = []
            isl.dcdc_converters = []
            isl.zero_branches = []
            isl.switches = []
            isl.breakers = []

        for node in self.nodes:
            node.v_gens = []
            node.v_dcdcs = []
            node.v_set = 0.0
            node.is_slack = False
        for bus in self.buses:
            bus.v_gens = []
            bus.v_dcdcs = []
            bus.v_set = 0.0
            bus.is_slack = False
            bus.generators = []
            bus.loads = []
            bus.branches = []
            bus.switches = []
            bus.breakers = []
            bus.dcdc_converters = []
            bus.zero_branches = []

        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            node = gen.node_obj
            if node is None or node.isl_obj is None:
                continue
            node.isl_obj.gens.append(gen)
            if gen.control_type == "V":
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)

        for dcdc in self.dcdc_converters:
            if dcdc.run_stat == 0:
                continue
            if dcdc.i_node_obj is None or dcdc.j_node_obj is None:
                continue
            if dcdc.i_node_obj.isl_obj is None or dcdc.j_node_obj.isl_obj is None:
                continue
            node = dcdc.i_node_obj
            dcdc.i_node_obj.isl_obj.dcdc_converters.append(dcdc)
            dcdc.j_node_obj.isl_obj.dcdc_converters.append(dcdc)
            if dcdc.control_type == "V":
                node.v_dcdcs.append(dcdc)
                if node.bus_obj is not None:
                    node.bus_obj.v_dcdcs.append(dcdc)
                node.isl_obj.v_dcdcs.append(dcdc)

        for load in self.loads:
            if load.run_stat == 0:
                continue
            if load.node_obj is None or load.node_obj.isl_obj is None:
                continue
            load.node_obj.isl_obj.loads.append(load)
        for switch in self.switches:
            if switch.i_node_obj is None or switch.j_node_obj is None:
                continue
            if switch.run_stat == 0 or switch.status == 0:
                continue
            if switch.i_node_obj.isl_obj and switch.j_node_obj.isl_obj and switch.i_node_obj.isl_obj == switch.j_node_obj.isl_obj:
                switch.i_node_obj.isl_obj.switches.append(switch)
        for br in self.branches:
            if br.run_stat == 0:
                continue
            if br.i_node_obj is None or br.j_node_obj is None:
                continue
            if br.i_node_obj.isl_obj and br.j_node_obj.isl_obj and br.i_node_obj.isl_obj == br.j_node_obj.isl_obj:
                br.i_node_obj.isl_obj.branches.append(br)
        for zbr in self.zero_branches:
            if zbr.run_stat == 0:
                continue
            if zbr.i_node_obj is None or zbr.j_node_obj is None:
                continue
            if zbr.i_node_obj.isl_obj and zbr.j_node_obj.isl_obj and zbr.i_node_obj.isl_obj == zbr.j_node_obj.isl_obj:
                zbr.i_node_obj.isl_obj.zero_branches.append(zbr)
        for brk in self.breakers:
            if brk.run_stat == 0 or brk.status == 0:
                continue
            if brk.i_node_obj is None or brk.j_node_obj is None:
                continue
            if brk.i_node_obj.isl_obj and brk.j_node_obj.isl_obj and brk.i_node_obj.isl_obj == brk.j_node_obj.isl_obj:
                brk.i_node_obj.isl_obj.breakers.append(brk)

        for bus in self.buses:
            if bus.isl_obj is None:
                continue
            bus.isl_obj.buses.append(bus)
            if len(bus.v_gens) + len(bus.v_dcdcs) > 0:
                bus.isl_obj.slack_nodes.append(bus)

        for isl in self.islands:
            if len(isl.slack_nodes) + len(isl.v_dcdcs) >= 1:
                isl.is_alive = True

        for bus in self.buses:
            bus.is_alive = bus.run_stat == 1 and bus.isl_obj is not None and bus.isl_obj.is_alive

        for node in self.nodes:
            node.is_alive = node.run_stat == 1 and node.isl_obj is not None and node.isl_obj.is_alive
        self.alive_buses = [bus for bus in self.buses if bus.is_alive]
        for load in self.loads:
            node = load.node_obj
            load.is_alive = node is not None and node.isl_obj is not None and load.run_stat == 1 and node.isl_obj.is_alive
        for gen in self.generators:
            node = gen.node_obj
            gen.is_alive = node is not None and node.isl_obj is not None and gen.run_stat == 1 and node.isl_obj.is_alive
        for br in self.branches:
            br.is_alive = (
                br.i_node_obj is not None
                and br.j_node_obj is not None
                and br.run_stat == 1
                and br.i_node_obj.is_alive
                and br.j_node_obj.is_alive
            )
        for zbr in self.zero_branches:
            zbr.is_alive = (
                zbr.i_node_obj is not None
                and zbr.j_node_obj is not None
                and zbr.run_stat == 1
                and zbr.i_node_obj.is_alive
                and zbr.j_node_obj.is_alive
            )
        for brk in self.breakers:
            brk.is_alive = (
                brk.i_node_obj is not None
                and brk.j_node_obj is not None
                and brk.run_stat == 1
                and brk.status == 1
                and brk.i_node_obj.is_alive
                and brk.j_node_obj.is_alive
            )
        for sw in self.switches:
            sw.is_alive = (
                sw.i_node_obj is not None
                and sw.j_node_obj is not None
                and sw.status == 1
                and sw.run_stat == 1
                and sw.i_node_obj.is_alive
                and sw.j_node_obj.is_alive
            )
        for conv in self.dcdc_converters:
            conv.is_alive = (
                conv.i_node_obj is not None
                and conv.j_node_obj is not None
                and conv.run_stat == 1
                and conv.i_node_obj.is_alive
                and conv.j_node_obj.is_alive
            )

    def print_isl_info(self):
        for isl in self.islands:
            print(f"isl {isl.idx} is_alive = {isl.is_alive}")
            print(f"    buses = {len(isl.buses)}:")
            for node in isl.buses:
                print(f"        {node.idx} {node.name} vbase: {node.vbase}")
            print(f"    gens = {len(isl.gens)}:")
            for gen in isl.gens:
                print(f"        {gen.idx} {gen.name} node = {gen.node} control_type = {gen.control_type}")
            print(f"    loads = {len(isl.loads)}:")
            for load in isl.loads:
                print(f"        {load.idx} {load.name} node = {load.node}")
            print(f"    branches = {len(isl.branches)}:")
            for br in isl.branches:
                print(f"        {br.idx} {br.name} i_node = {br.i_node} j_node = {br.j_node} r = {br.r}")
            print(f"    switches = {len(isl.switches)}:")
            for sw in isl.switches:
                print(f"        {sw.idx} {sw.name} i_node = {sw.i_node} j_node = {sw.j_node} status = {sw.status}")
            print(f"    zero_branches = {len(isl.zero_branches)}:")
            for zbr in isl.zero_branches:
                print(f"        {zbr.idx} {zbr.name} i_node = {zbr.i_node} j_node = {zbr.j_node}")
            print(f"    breakers = {len(getattr(isl, 'breakers', []))}:")
            for brk in getattr(isl, "breakers", []):
                print(f"        {brk.idx} {brk.name} i_node = {brk.i_node} j_node = {brk.j_node} status = {brk.status}")
            print(f"    dcdc_converters = {len(isl.dcdc_converters)}:")
            for dcc in isl.dcdc_converters:
                print(f"        {dcc.idx} {dcc.name} i_node = {dcc.i_node} j_node = {dcc.j_node} r1 = {dcc.r1} r2 = {dcc.r2} control_type = {dcc.control_type}")

    def check_topo(self):
        errors = []
        warns = []
        if len(self.islands) == 0:
            self.topo()

        node_ref_count = {node.idx: 0 for node in self.nodes}

        def check_node(node_idx, dev_type, dev):
            if node_idx not in self.node_dict:
                errors.append(f"设备 {dev_type}[{dev.idx}] {dev.name} 引用的节点 {node_idx} 不存在")
            elif self.node_dict[node_idx].run_stat == 1:
                node_ref_count[node_idx] += 1

        for br in self.branches:
            if br.run_stat:
                check_node(br.i_node, "Branch", br)
                check_node(br.j_node, "Branch", br)
        for zbr in self.zero_branches:
            if zbr.run_stat:
                check_node(zbr.i_node, "ZeroBranch", zbr)
                check_node(zbr.j_node, "ZeroBranch", zbr)
        for sw in self.switches:
            if sw.run_stat:
                check_node(sw.i_node, "Switch", sw)
                check_node(sw.j_node, "Switch", sw)
        for brk in self.breakers:
            if brk.run_stat and brk.status:
                check_node(brk.i_node, "Break", brk)
                check_node(brk.j_node, "Break", brk)
        for ld in self.loads:
            if ld.run_stat:
                check_node(ld.node, "Load", ld)
        for gen in self.generators:
            if gen.run_stat:
                check_node(gen.node, "Generator", gen)
        for dcdc in self.dcdc_converters:
            if dcdc.run_stat:
                check_node(dcdc.i_node, "DCDCConverter", dcdc)
                check_node(dcdc.j_node, "DCDCConverter", dcdc)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {node.name} 未关联任何设备")
            if node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {node.name} 单端悬空，请检查！")

        for isl in self.islands:
            vbase_set = {int(bus.vbase * 1000) for bus in isl.buses}
            if len(vbase_set) > 1:
                str_info = f"岛屿 {isl.idx} 内节点电压基值不一致:"
                for vbase in vbase_set:
                    str_info += f" {vbase / 1000.0 :.2f}"
                errors.append(str_info)
            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V节点:"
                for node in isl.slack_nodes:
                    str_info += f" {node.name}"
                warns.append(str_info)
            if len(isl.v_dcdcs) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V变流器:"
                for dcdc in isl.v_dcdcs:
                    str_info += f" {dcdc.name}"
                warns.append(str_info)
            if len(isl.slack_nodes) + len(isl.v_dcdcs) == 0:
                errors.append(f"岛屿 {isl.idx} , 内无电压控制源（定V节点或定V变流器）")
            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} , 内有多个电压控制源（定V节点或定V变流器）:"
                for node in isl.slack_nodes:
                    str_info += f" node-{node.name}"
                errors.append(str_info)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if len(node.v_gens) + len(node.v_dcdcs) <= 1:
                continue
            if len(node.v_gens) + len(node.v_dcdcs) >= 2:
                errors.append(f"松弛节点 {node.idx} 上的定V发电机与定V变流器数量之和超过1，请检查拓扑！")
            node.v_set = 0.0
            if len(node.v_gens) >= 1:
                node.v_set = node.v_gens[0].v_set
            if len(node.v_dcdcs) > 1:
                node.v_set = node.v_dcdcs[0].v_set
            node.is_slack = True

        return warns, errors
