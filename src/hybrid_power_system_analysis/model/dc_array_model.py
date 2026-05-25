import threading
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from efile_read import _read_efile_rows, efile_factory_from_file, efile_factory_from_rows
from paths import resolve_project_file
from unit_system import dc_current_base_ka

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
for path in (MODEL_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from topology import build_dc_topology_input_ppc
from model.array_common import (
    _assign_current_if_present,
    _assign_power_if_present,
    _base_from_rows,
    _cell,
    _code_column,
    _code_value,
    _empty,
    _float_cell,
    _float_column,
    _float_value,
    _has_value,
    _int_cell,
    _int_column,
    _int_value,
    _name_array,
    _names_from_rows,
    _raw_vbase_by_node,
    _rows_for,
    _scale_by_node,
    _value,
    _voltage_set_column,
    file_cache_key as _file_cache_key,
)


CTRL_P = 0
CTRL_V = 1
CTRL_I = 2
CTRL_SLACK = 3
CTRL_CODE = {
    "P": CTRL_P,
    "CTRL_P": CTRL_P,
    "V": CTRL_V,
    "CTRL_V": CTRL_V,
    "I": CTRL_I,
    "CTRL_I": CTRL_I,
    "SLACK": CTRL_SLACK,
    "CTRL_SLACK": CTRL_SLACK,
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
    "i_control_type": 5,
    "j_control_type": 6,
    "p_set": 7,
    "i_set": 8,
    "v_set": 9,
    "run_stat": 10,
    "i_p": 11,
    "j_p": 12,
    "i_c": 13,
    "j_c": 14,
}

_DC_PPC_CACHE = {}
_DC_PPC_CACHE_LOCK = threading.Lock()


def clear_dc_ppc_cache(file_path=None) -> None:
    with _DC_PPC_CACHE_LOCK:
        if file_path is None:
            _DC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _DC_PPC_CACHE.pop(path, None)

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


def _validate_dcdc_dual_control(i_control: np.ndarray, j_control: np.ndarray) -> None:
    valid_codes = np.asarray([CTRL_P, CTRL_V, CTRL_I, CTRL_SLACK], dtype=np.int64)
    bad = ~np.isin(i_control, valid_codes) | ~np.isin(j_control, valid_codes)
    i_active = i_control != CTRL_SLACK
    j_active = j_control != CTRL_SLACK
    bad |= i_active == j_active
    if np.any(bad):
        pos = int(np.flatnonzero(bad)[0])
        raise ValueError(
            "DCDCConverter 控制类型必须且只能一端为 CTRL_P/CTRL_V/CTRL_I，另一端为 SLACK；"
            f"第 {pos + 1} 行为 i_control_type={int(i_control[pos])}, j_control_type={int(j_control[pos])}"
        )


def _dcdc_control_columns_from_rows(table_rows, columns) -> Tuple[np.ndarray, np.ndarray]:
    if "i_control_type" in columns or "j_control_type" in columns:
        i_control = _code_column(table_rows, columns, "i_control_type", CTRL_CODE, "P").astype(np.int64, copy=False)
        j_control = _code_column(table_rows, columns, "j_control_type", CTRL_CODE, "SLACK").astype(np.int64, copy=False)
    else:
        i_control = _code_column(table_rows, columns, "control_type", CTRL_CODE, "P").astype(np.int64, copy=False)
        j_control = np.full(len(table_rows), CTRL_SLACK, dtype=np.int64)
    _validate_dcdc_dual_control(i_control, j_control)
    return i_control.astype(np.float64, copy=False), j_control.astype(np.float64, copy=False)


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
    current_scale_by_node = None

    def get_current_scale_by_node():
        nonlocal current_scale_by_node
        if current_scale_by_node is None:
            current_scale_by_node = {
                int(row[BUS_COLS["idx"]]): i_scale * dc_current_base_ka(p_base_kW, float(row[BUS_COLS["vbase"]]))
                for row in bus
            }
        return current_scale_by_node

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
        _assign_current_if_present(branch, BRANCH_COLS["current"], table_rows, columns, "current", branch[:, BRANCH_COLS["i_node"]], get_current_scale_by_node)
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
        _assign_current_if_present(load, LOAD_COLS["current"], table_rows, columns, "current", load[:, LOAD_COLS["node"]], get_current_scale_by_node)
    load_names = _names_from_rows(table_rows, columns, "load", load[:, LOAD_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "DCGenerator")
    gen = np.zeros((len(table_rows), len(GEN_COLS)), dtype=np.float64)
    if table_rows:
        gen[:, GEN_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        gen[:, GEN_COLS["node"]] = _int_column(table_rows, columns, "node")
        gen[:, GEN_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", CTRL_CODE, "P")
        gen[:, GEN_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
        gen[:, GEN_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", gen[:, GEN_COLS["node"]], raw_vbase_by_idx)
        _assign_current_if_present(gen, GEN_COLS["i_set"], table_rows, columns, "i_set", gen[:, GEN_COLS["node"]], get_current_scale_by_node)
        gen[:, GEN_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(gen, GEN_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(gen, GEN_COLS["current"], table_rows, columns, "current", gen[:, GEN_COLS["node"]], get_current_scale_by_node)
    gen_names = _names_from_rows(table_rows, columns, "gen", gen[:, GEN_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "DCZeroBranch")
    zero_branch = np.zeros((len(table_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    if table_rows:
        zero_branch[:, ZERO_BRANCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        zero_branch[:, ZERO_BRANCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        zero_branch[:, ZERO_BRANCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(zero_branch, ZERO_BRANCH_COLS["p"], table_rows, columns, "p", p_base)
        _assign_current_if_present(zero_branch, ZERO_BRANCH_COLS["current"], table_rows, columns, "current", zero_branch[:, ZERO_BRANCH_COLS["i_node"]], get_current_scale_by_node)
    zero_branch_names = _names_from_rows(table_rows, columns, "zero_branch", zero_branch[:, ZERO_BRANCH_COLS["idx"]])

    sw_columns, sw_rows = _rows_for(rows, "DCSwitch")
    switch, switch_names = _build_switch_like_from_rows(sw_rows, sw_columns, get_current_scale_by_node, prefix="switch", p_base=p_base)
    br_columns, br_rows = _rows_for(rows, "DCBreak")
    breaker, breaker_names = _build_switch_like_from_rows(
        br_rows,
        br_columns,
        get_current_scale_by_node,
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
        i_control, j_control = _dcdc_control_columns_from_rows(table_rows, columns)
        dcdc[:, DCDC_COLS["i_control_type"]] = i_control
        dcdc[:, DCDC_COLS["j_control_type"]] = j_control
        dcdc[:, DCDC_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
        _assign_current_if_present(dcdc, DCDC_COLS["i_set"], table_rows, columns, "i_set", dcdc[:, DCDC_COLS["i_node"]], get_current_scale_by_node)
        v_control_nodes = np.where(
            j_control.astype(np.int64, copy=False) == CTRL_V,
            dcdc[:, DCDC_COLS["j_node"]],
            dcdc[:, DCDC_COLS["i_node"]],
        )
        dcdc[:, DCDC_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", v_control_nodes, raw_vbase_by_idx)
        dcdc[:, DCDC_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(dcdc, DCDC_COLS["i_p"], table_rows, columns, "i_p", p_base)
        _assign_power_if_present(dcdc, DCDC_COLS["j_p"], table_rows, columns, "j_p", p_base)
        _assign_current_if_present(dcdc, DCDC_COLS["i_c"], table_rows, columns, "i_c", dcdc[:, DCDC_COLS["i_node"]], get_current_scale_by_node)
        _assign_current_if_present(dcdc, DCDC_COLS["j_c"], table_rows, columns, "j_c", dcdc[:, DCDC_COLS["j_node"]], get_current_scale_by_node)
    dcdc_names = _names_from_rows(table_rows, columns, "dcdc", dcdc[:, DCDC_COLS["idx"]])

    ppc = {
        "format": "dc_ppc_v1",
        "source": str(source),
        "base": {
            "p_base": float(p_base),
            "u_scale": float(u_scale),
            "p_scale": float(p_scale),
            "i_scale": float(i_scale),
            "p_base_kW": float(p_base_kW),
        },
        "bus": bus,
        "branch": branch,
        "load": load,
        "gen": gen,
        "zero_branch": zero_branch,
        "switch": switch,
        "break": breaker,
        "dcdc": dcdc,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "load_cols": LOAD_COLS,
        "gen_cols": GEN_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "dcdc_cols": DCDC_COLS,
        "ctrl": {"P": CTRL_P, "V": CTRL_V, "I": CTRL_I, "SLACK": CTRL_SLACK},
        "bus_name": bus_names,
        "branch_name": branch_names,
        "load_name": load_names,
        "gen_name": gen_names,
        "zero_branch_name": zero_branch_names,
        "switch_name": switch_names,
        "break_name": breaker_names,
        "dcdc_name": dcdc_names,
    }
    ppc["_topology_input"] = build_dc_topology_input_ppc(ppc)
    return ppc


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


_VALUE_FLOAT = 0
_VALUE_INT = 1
_VALUE_CTRL = 2


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
    for conv in dcdcs:
        if not hasattr(conv, "i_control_type"):
            setattr(conv, "i_control_type", getattr(conv, "control_type", "P"))
        if not hasattr(conv, "j_control_type"):
            setattr(conv, "j_control_type", "SLACK")

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
            (DCDC_COLS["i_control_type"], "i_control_type", "P", _VALUE_CTRL),
            (DCDC_COLS["j_control_type"], "j_control_type", "SLACK", _VALUE_CTRL),
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
    if dcdc.size:
        _validate_dcdc_dual_control(
            dcdc[:, DCDC_COLS["i_control_type"]].astype(np.int64, copy=False),
            dcdc[:, DCDC_COLS["j_control_type"]].astype(np.int64, copy=False),
        )

    ppc = {
        "format": "dc_ppc_v1",
        "source": str(getattr(network, "source", getattr(network, "file_name", "<network>"))),
        "base": {
            "p_base": float(p_base),
            "u_scale": float(u_scale),
            "p_scale": float(p_scale),
            "i_scale": float(i_scale),
            "p_base_kW": float(p_base_kw),
        },
        "bus": bus,
        "branch": branch,
        "load": load,
        "gen": gen,
        "zero_branch": zero_branch,
        "switch": build_switch_like(switches),
        "break": build_switch_like(breakers),
        "dcdc": dcdc,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "load_cols": LOAD_COLS,
        "gen_cols": GEN_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "dcdc_cols": DCDC_COLS,
        "ctrl": {"P": CTRL_P, "V": CTRL_V, "I": CTRL_I, "SLACK": CTRL_SLACK},
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
    ppc["_topology_input"] = build_dc_topology_input_ppc(ppc)
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
    from dc_model import (
        DCBreak,
        DCBranch,
        DCDCConverter,
        DCGenerator,
        DCLoad,
        DCNode,
        DCPowerNetwork,
        DCSwitch,
        DCZeroBranch,
    )

    def names(key: str, prefix: str, count: int):
        values = ppc.get(key)
        if values is None:
            return [f"{prefix}_{idx}" for idx in range(count)]
        return [str(value) for value in values]

    ctrl_name = {CTRL_P: "P", CTRL_V: "V", CTRL_I: "I", CTRL_SLACK: "SLACK"}
    dcdc_ctrl_name = {CTRL_P: "CTRL_P", CTRL_V: "CTRL_V", CTRL_I: "CTRL_I", CTRL_SLACK: "SLACK"}
    bus_names = names("bus_name", "bus", ppc["bus"].shape[0])
    branch_names = names("branch_name", "branch", ppc["branch"].shape[0])
    load_names = names("load_name", "load", ppc["load"].shape[0])
    gen_names = names("gen_name", "gen", ppc["gen"].shape[0])
    zero_branch_names = names("zero_branch_name", "zero_branch", ppc["zero_branch"].shape[0])
    switch_names = names("switch_name", "switch", ppc["switch"].shape[0])
    break_names = names("break_name", "break", ppc.get("break", _empty(len(BREAK_COLS))).shape[0])
    dcdc_names = names("dcdc_name", "dcdc", ppc["dcdc"].shape[0])

    network = DCPowerNetwork()
    base = ppc["base"]
    network.ppc = ppc
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    network.nodes = [
        DCNode(
            int(row[BUS_COLS["idx"]]),
            float(row[BUS_COLS["vbase"]]),
            float(row[BUS_COLS["voltage"]]),
            int(row[BUS_COLS["run_stat"]]),
        )
        for row in ppc["bus"]
    ]
    for node in network.nodes:
        node.isl = None
        node.isl_obj = None
        node.v_set = 1.0
        node.v_gens = []
        node.v_dcdcs = []
        node.is_slack = False
        node.bus = None
        node.bus_obj = None
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.dcdc_converters = []
        node.zero_branches = []

    network.branches = [
        DCBranch(
            int(row[BRANCH_COLS["idx"]]),
            int(row[BRANCH_COLS["i_node"]]),
            int(row[BRANCH_COLS["j_node"]]),
            float(row[BRANCH_COLS["r"]]),
            int(row[BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["branch"]
    ]

    network.loads = [
        DCLoad(
            int(row[LOAD_COLS["idx"]]),
            int(row[LOAD_COLS["node"]]),
            float(row[LOAD_COLS["pbase"]]),
            float(row[LOAD_COLS["pv0"]]),
            float(row[LOAD_COLS["pv1"]]),
            float(row[LOAD_COLS["pv2"]]),
            int(row[LOAD_COLS["run_stat"]]),
        )
        for row in ppc["load"]
    ]

    network.generators = [
        DCGenerator(
            int(row[GEN_COLS["idx"]]),
            int(row[GEN_COLS["node"]]),
            ctrl_name.get(int(row[GEN_COLS["control_type"]]), "P"),
            float(row[GEN_COLS["p_set"]]),
            float(row[GEN_COLS["v_set"]]),
            float(row[GEN_COLS["i_set"]]),
            int(row[GEN_COLS["run_stat"]]),
        )
        for row in ppc["gen"]
    ]

    network.zero_branches = [
        DCZeroBranch(
            int(row[ZERO_BRANCH_COLS["idx"]]),
            int(row[ZERO_BRANCH_COLS["i_node"]]),
            int(row[ZERO_BRANCH_COLS["j_node"]]),
            int(row[ZERO_BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["zero_branch"]
    ]

    network.switches = [
        DCSwitch(
            int(row[SWITCH_COLS["idx"]]),
            int(row[SWITCH_COLS["i_node"]]),
            int(row[SWITCH_COLS["j_node"]]),
            int(row[SWITCH_COLS["status"]]),
            int(row[SWITCH_COLS["run_stat"]]),
        )
        for row in ppc["switch"]
    ]

    network.breakers = [
        DCBreak(
            int(row[BREAK_COLS["idx"]]),
            int(row[BREAK_COLS["i_node"]]),
            int(row[BREAK_COLS["j_node"]]),
            int(row[BREAK_COLS["status"]]),
            int(row[BREAK_COLS["run_stat"]]),
        )
        for row in ppc.get("break", _empty(len(BREAK_COLS)))
    ]

    network.dcdc_converters = [
        DCDCConverter(
            int(row[DCDC_COLS["idx"]]),
            int(row[DCDC_COLS["i_node"]]),
            int(row[DCDC_COLS["j_node"]]),
            float(row[DCDC_COLS["r1"]]),
            float(row[DCDC_COLS["r2"]]),
            dcdc_ctrl_name.get(int(row[DCDC_COLS["i_control_type"]]), "CTRL_P"),
            float(row[DCDC_COLS["p_set"]]),
            float(row[DCDC_COLS["i_set"]]),
            float(row[DCDC_COLS["v_set"]]),
            int(row[DCDC_COLS["run_stat"]]),
            i_control_type=dcdc_ctrl_name.get(int(row[DCDC_COLS["i_control_type"]]), "CTRL_P"),
            j_control_type=dcdc_ctrl_name.get(int(row[DCDC_COLS["j_control_type"]]), "SLACK"),
        )
        for row in ppc["dcdc"]
    ]

    network.node_dict = {}
    network.switch_dict = {}
    network.break_dict = {}
    network.load_dict = {}
    network.generator_dict = {}
    network.zero_branch_dict = {}
    network.branch_dict = {}
    network.zero_branche_dict = network.zero_branch_dict
    network.branche_dict = network.branch_dict
    network.dcdc_converter_dict = {}
    network.islands = []

    for obj, name in zip(network.nodes, bus_names):
        obj.name = name
        obj.is_alive = False
    for obj, row, name in zip(network.branches, ppc["branch"], branch_names):
        obj.name = name
        obj.i_p = float(row[BRANCH_COLS["i_p"]])
        obj.j_p = float(row[BRANCH_COLS["j_p"]])
        obj.current = float(row[BRANCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.loads, ppc["load"], load_names):
        obj.name = name
        obj.p = float(row[LOAD_COLS["p"]])
        obj.current = float(row[LOAD_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.generators, ppc["gen"], gen_names):
        obj.name = name
        obj.p = float(row[GEN_COLS["p"]])
        obj.current = float(row[GEN_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.zero_branches, ppc["zero_branch"], zero_branch_names):
        obj.name = name
        obj.p = float(row[ZERO_BRANCH_COLS["p"]])
        obj.current = float(row[ZERO_BRANCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.switches, ppc["switch"], switch_names):
        obj.name = name
        obj.p = float(row[SWITCH_COLS["p"]])
        obj.current = float(row[SWITCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.breakers, ppc.get("break", _empty(len(BREAK_COLS))), break_names):
        obj.name = name
        obj.p = float(row[BREAK_COLS["p"]])
        obj.current = float(row[BREAK_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.dcdc_converters, ppc["dcdc"], dcdc_names):
        obj.name = name
        obj.i_p = float(row[DCDC_COLS["i_p"]])
        obj.j_p = float(row[DCDC_COLS["j_p"]])
        obj.i_c = float(row[DCDC_COLS["i_c"]])
        obj.j_c = float(row[DCDC_COLS["j_c"]])
        obj.is_alive = False
    return network
