import argparse
import contextlib
import io
import math
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse.csgraph import maximum_bipartite_matching as sp_maximum_bipartite_matching


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_lf import (
    ACPowerFlowCalc,
    matpower_branch_stamp,
    matpower_branch_stamp_vectorized,
    matpower_transformer_stamp_vectorized,
)
from ac_model import ACPowerNetwork
from efile_read import EBook
from model import topology as network_topology
from ac_array_model import (
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PQ,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    BREAK_COLS,
    SHUNT_B,
    SHUNT_COLS,
    SHUNT_Q,
    SHUNT_V,
    SHUNT_Z,
    SWITCH_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
    build_ac_ppc_from_e_file,
)
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    EstimateResult,
    Measurement,
    MeasurementList,
    MeasurementTable,
    ObservabilityResult,
    is_pseudo_measurement,
    mark_measurement_invalid,
    mark_measurement_pseudo,
    measurement_status_is_active,
    measurement_table_from_measurements,
    measurement_table_status_code,
    normalize_measurement_status,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    SparseJacobianBuilder,
    angle_residual_mask,
    build_normal_equations,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    measurement_residual as build_measurement_residual,
    NormalEquationSolver,
    _normal_equation_structural_pattern,
    observability_rank_details,
    observability_weak_direction,
    sparse_structural_rank,
    targeted_redundancy_count,
    unanchored_angle_state_indices,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata, state_meta_at
from secore.se_array_plan import (
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    rows_by_device_type_code,
    take_measurement_view,
)
from secore.se_result import SEResult, build_seresult_summary, normalize_seresult_return_mode
from unit_system import ac_current_base_ka


DEFAULT_CASE = model_file("ac", "ieee39.e")
DEFAULT_MEAS = measurement_file("ac", "ieee39.meas")


_DEVICE_TYPE_CODES = DEVICE_TYPE_CODES

_TERMINAL_POWER_MEASUREMENT_TYPES = frozenset(("P_FROM", "Q_FROM", "P_TO", "Q_TO"))
_VOLTAGE_MEASUREMENT_TYPES = frozenset(("V", "V_FROM", "V_TO", "V_GEN", "V_LOAD"))
_VOLTAGE_MEASUREMENT_TYPE_TUPLE = tuple(_VOLTAGE_MEASUREMENT_TYPES)
_AC_TERMINAL_MEASUREMENT_KIND = {
    "P_FROM": 0,
    "Q_FROM": 1,
    "V_FROM": 2,
    "I_FROM": 3,
    "P_TO": 4,
    "Q_TO": 5,
    "V_TO": 6,
    "I_TO": 7,
}
_AC_ZERO_MEASUREMENT_KIND = {
    **_AC_TERMINAL_MEASUREMENT_KIND,
    "V_DIFF": 8,
    "ANGLE_DIFF": 9,
    "THETA_DIFF": 9,
}
_AC_NODE_MEASUREMENT_KIND = {"V": 0, "ANGLE": 1, "THETA": 1}
_AC_LOAD_MEASUREMENT_KIND = {"P_LOAD": 0, "Q_LOAD": 1, "I_LOAD": 2, "V_LOAD": 3}
_AC_GENERATOR_POWER_MEASUREMENT_KIND = {"P_GEN": 0, "Q_GEN": 1, "I_GEN": 2}
_AC_GENERATOR_SIMPLE_MEASUREMENT_KIND = {"V_GEN": 0}
_AC_BALANCE_MEASUREMENT_KIND = {"P_BALANCE": 0, "Q_BALANCE": 1}
_AC_CONSTRAINT_MEASUREMENT_KIND = {"V_DIFF": 0, "ANGLE_DIFF": 1, "THETA_DIFF": 1}
_PSEUDO_DEVICE_SUMMARY_TYPES = frozenset(("ACGenerator", "ACLoad"))
_PSEUDO_MEASUREMENT_SUMMARY_TYPES = {
    "ACGenerator": frozenset(("P_GEN", "Q_GEN")),
    "ACLoad": frozenset(("P_LOAD", "Q_LOAD")),
    "ACZeroBranch": frozenset(("P_FROM", "Q_FROM", "V_FROM", "I_FROM")),
    "ACBreak": frozenset(("P_FROM", "Q_FROM", "V_FROM", "I_FROM")),
}

_OBSERVABILITY_RESULT_CACHE = {}
_CTRL_NAME_BY_CODE = {
    CTRL_PQ: "PQ",
    CTRL_P: "P",
    CTRL_PV: "PV",
    CTRL_SLACK: "SLACK",
}
_SHUNT_NAME_BY_CODE = {
    SHUNT_Q: "Q",
    SHUNT_V: "V",
    SHUNT_B: "B",
    SHUNT_Z: "Z",
}


class _ACArrayObject:
    __slots__ = (
        "idx",
        "name",
        "vbase",
        "voltage",
        "angle",
        "run_stat",
        "is_alive",
        "isl",
        "isl_obj",
        "bus",
        "bus_obj",
        "generators",
        "loads",
        "branches",
        "switches",
        "breakers",
        "zero_branches",
        "transformers",
        "shunt_compensators",
        "v_gens",
        "node",
        "node_obj",
        "i_node",
        "j_node",
        "i_node_obj",
        "j_node_obj",
        "r",
        "x",
        "b",
        "gt",
        "bt",
        "tap",
        "shift",
        "status",
        "control_type",
        "p_set",
        "q_set",
        "v_set",
        "alpha",
        "pbase",
        "pv0",
        "pv1",
        "pv2",
        "qbase",
        "qv0",
        "qv1",
        "qv2",
        "g_set",
        "b_set",
        "p",
        "q",
        "current",
        "i_p",
        "i_q",
        "i_c",
        "j_p",
        "j_q",
        "j_c",
    )


def _file_cache_key(file_name: Path) -> Tuple[Path, int, int]:
    path = Path(file_name).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _read_standard_measurement_lines(file_name: Path, data_lines: Sequence[str], measurement_cls):
    count = len(data_lines)
    measurements = [None] * count
    idx_values = np.empty(count, dtype=np.int64)
    name_values = np.empty(count, dtype=object)
    device_type_values = np.empty(count, dtype=object)
    device_name_values = np.empty(count, dtype=object)
    meas_type_values = np.empty(count, dtype=object)
    weight_values = np.empty(count, dtype=np.float64)
    valid_values = np.empty(count, dtype=bool)
    value_values = np.empty(count, dtype=np.float64)
    device_type_code_values = np.empty(count, dtype=np.int16)
    angle_mask_values = np.empty(count, dtype=bool)
    status_values = np.empty(count, dtype=np.int16)

    new_measurement = measurement_cls.__new__
    device_type_code_get = _DEVICE_TYPE_CODES.get
    angle_types = ANGLE_MEASUREMENT_TYPES
    intern = sys.intern
    device_type_cache = {}
    measurement_type_cache = {}
    rows_by_code_values = {}

    for row_pos, raw_line in enumerate(data_lines):
        row = raw_line[1:].split()
        if len(row) < 8:
            raise RuntimeError(f"Malformed Measurement row at line {row_pos + 1} in {file_name}")
        idx = int(row[0])
        name = row[1]
        raw_device_type = row[2]
        device_type_entry = device_type_cache.get(raw_device_type)
        if device_type_entry is None:
            device_type = intern(raw_device_type)
            device_type_code = int(device_type_code_get(device_type, 0))
            code_rows = rows_by_code_values.get(device_type_code)
            if code_rows is None:
                code_rows = []
                rows_by_code_values[device_type_code] = code_rows
            device_type_entry = (device_type, device_type_code, code_rows)
            device_type_cache[raw_device_type] = device_type_entry
        device_type, device_type_code, code_rows = device_type_entry

        raw_meas_type = row[4]
        meas_type_entry = measurement_type_cache.get(raw_meas_type)
        if meas_type_entry is None:
            if raw_meas_type and "a" <= raw_meas_type[0] <= "z":
                meas_type = intern(raw_meas_type.upper())
            else:
                meas_type = intern(raw_meas_type)
            meas_type_entry = (meas_type, meas_type in angle_types)
            measurement_type_cache[raw_meas_type] = meas_type_entry
        meas_type, is_angle_measurement = meas_type_entry

        weight = float(row[5])
        valid = row[6] == "1"
        value = float(row[7])
        status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
        meas = new_measurement(measurement_cls)
        meas.idx = idx
        meas.name = name
        meas.device_type = device_type
        meas.device_name = row[3]
        meas.meas_type = meas_type
        meas.weight = weight
        meas.valid = valid
        meas.value = value
        meas.status = status

        measurements[row_pos] = meas
        idx_values[row_pos] = idx
        name_values[row_pos] = name
        device_type_values[row_pos] = device_type
        device_name_values[row_pos] = meas.device_name
        meas_type_values[row_pos] = meas_type
        weight_values[row_pos] = weight
        valid_values[row_pos] = valid
        value_values[row_pos] = value
        device_type_code_values[row_pos] = device_type_code
        angle_mask_values[row_pos] = is_angle_measurement
        status_values[row_pos] = status
        code_rows.append(row_pos)

    rows_by_code = {
        int(code): np.asarray(rows, dtype=np.int64)
        for code, rows in rows_by_code_values.items()
    }
    table = MeasurementTable(
        idx=idx_values,
        name=name_values,
        device_type=device_type_values,
        device_name=device_name_values,
        meas_type=meas_type_values,
        weight=weight_values,
        valid=valid_values,
        value=value_values,
        device_type_code=device_type_code_values,
        angle_mask=angle_mask_values,
        status_code=status_values,
        rows_by_device_type_code=rows_by_code,
    )
    return MeasurementList(measurements, table, normalized=False)


def _read_measurements_direct(file_name: Path, measurement_cls, scale_context=None):
    required_columns = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
    header = None
    column_index = None
    header_len = 0
    standard_header = False
    idx_col = name_col = dev_type_col = dev_name_col = meas_type_col = weight_col = valid_col = value_col = status_col = -1
    measurements = []
    append_measurement = measurements.append
    idx_values = []
    name_values = []
    device_type_values = []
    device_name_values = []
    meas_type_values = []
    weight_values = []
    valid_values = []
    value_values = []
    device_type_code_values = []
    angle_mask_values = []
    status_values = []
    rows_by_code_values = {}
    append_idx = idx_values.append
    append_name = name_values.append
    append_device_type = device_type_values.append
    append_device_name = device_name_values.append
    append_meas_type = meas_type_values.append
    append_weight = weight_values.append
    append_valid = valid_values.append
    append_value = value_values.append
    append_device_type_code = device_type_code_values.append
    append_angle_mask = angle_mask_values.append
    append_status = status_values.append
    new_measurement = measurement_cls.__new__
    int_cell = int
    float_cell = float
    device_type_code_get = _DEVICE_TYPE_CODES.get
    angle_types = ANGLE_MEASUREMENT_TYPES
    intern = sys.intern
    device_type_cache = {}
    measurement_type_cache = {}
    in_measurement = False
    row_pos = 0
    with open(file_name, mode="rt", encoding="utf8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            first = raw_line[0] if raw_line else ""
            if not in_measurement:
                if first == "<" and raw_line.strip() == "<Measurement>":
                    in_measurement = True
                continue
            if first == "@":
                header = raw_line[1:].split()
                standard_header = tuple(header) == required_columns
                header_len = len(header)
                if standard_header:
                    column_index = {}
                    idx_col, name_col, dev_type_col, dev_name_col = 0, 1, 2, 3
                    meas_type_col, weight_col, valid_col, value_col = 4, 5, 6, 7
                else:
                    column_index = {name: idx for idx, name in enumerate(header)}
                    missing = [name for name in required_columns if name not in column_index]
                    if missing:
                        raise RuntimeError(f"{file_name} Measurement header is missing columns: {missing}")
                    idx_col = column_index["idx"]
                    name_col = column_index["name"]
                    dev_type_col = column_index["dev_type"]
                    dev_name_col = column_index["dev_name"]
                    meas_type_col = column_index["meas_type"]
                    weight_col = column_index["weight"]
                    valid_col = column_index["valid"]
                    value_col = column_index["value"]
                    status_col = column_index.get("status", -1)
                continue
            if first == "#":
                if header is None or column_index is None:
                    raise RuntimeError(f"{file_name} Measurement data appears before the header")
                if standard_header and scale_context is None:
                    data_lines = [raw_line]
                    for raw_line in fp:
                        first = raw_line[0] if raw_line else ""
                        if first == "#":
                            data_lines.append(raw_line)
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line == "</Measurement>":
                            return _read_standard_measurement_lines(file_name, data_lines, measurement_cls)
                        raise SyntaxError(f"Invalid Measurement row at line {line_no} in {file_name}")
                    return _read_standard_measurement_lines(file_name, data_lines, measurement_cls)
                if standard_header:
                    row = raw_line[1:].split()
                    if len(row) < header_len:
                        raise RuntimeError(f"Malformed Measurement row at line {line_no} in {file_name}")
                    idx_text = row[0]
                    name = row[1]
                    raw_device_type = row[2]
                    device_type_entry = device_type_cache.get(raw_device_type)
                    if device_type_entry is None:
                        device_type = intern(raw_device_type)
                        device_type_code = int(device_type_code_get(device_type, 0))
                        code_rows = rows_by_code_values.get(device_type_code)
                        if code_rows is None:
                            code_rows = []
                            rows_by_code_values[device_type_code] = code_rows
                        device_type_entry = (device_type, device_type_code, code_rows)
                        device_type_cache[raw_device_type] = device_type_entry
                    device_type, device_type_code, code_rows = device_type_entry
                    device_name = row[3]
                    raw_meas_type = row[4]
                    meas_type_entry = measurement_type_cache.get(raw_meas_type)
                    if meas_type_entry is None:
                        if raw_meas_type and "a" <= raw_meas_type[0] <= "z":
                            meas_type = intern(raw_meas_type.upper())
                        else:
                            meas_type = intern(raw_meas_type)
                        meas_type_entry = (meas_type, meas_type in angle_types)
                        measurement_type_cache[raw_meas_type] = meas_type_entry
                    meas_type, is_angle_measurement = meas_type_entry
                    weight_text = row[5]
                    valid_text = row[6]
                    value_text = row[7]
                    idx = int_cell(idx_text)
                    weight = float_cell(weight_text)
                    valid = valid_text == "1"
                    value = float_cell(value_text)
                    status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                else:
                    row = raw_line[1:].split(maxsplit=header_len - 1)
                    if len(row) < header_len:
                        raise RuntimeError(f"Malformed Measurement row at line {line_no} in {file_name}")
                    raw_device_type = row[dev_type_col]
                    device_type_entry = device_type_cache.get(raw_device_type)
                    if device_type_entry is None:
                        device_type = intern(raw_device_type)
                        device_type_code = int(device_type_code_get(device_type, 0))
                        code_rows = rows_by_code_values.get(device_type_code)
                        if code_rows is None:
                            code_rows = []
                            rows_by_code_values[device_type_code] = code_rows
                        device_type_entry = (device_type, device_type_code, code_rows)
                        device_type_cache[raw_device_type] = device_type_entry
                    device_type, device_type_code, code_rows = device_type_entry
                    raw_meas_type = row[meas_type_col]
                    meas_type_entry = measurement_type_cache.get(raw_meas_type)
                    if meas_type_entry is None:
                        meas_type = intern(raw_meas_type.upper())
                        meas_type_entry = (meas_type, meas_type in angle_types)
                        measurement_type_cache[raw_meas_type] = meas_type_entry
                    meas_type, is_angle_measurement = meas_type_entry
                    idx = int_cell(row[idx_col])
                    name = row[name_col]
                    device_name = row[dev_name_col]
                    weight = float_cell(row[weight_col])
                    valid = row[valid_col] == "1"
                    value = float_cell(row[value_col])
                    status = (
                        normalize_measurement_status(row[status_col], valid=valid)
                        if status_col >= 0
                        else MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                    )
                    if not measurement_status_is_active(status):
                        valid = False
                meas = new_measurement(measurement_cls)
                meas.idx = idx
                meas.name = name
                meas.device_type = device_type
                meas.device_name = device_name
                meas.meas_type = meas_type
                meas.weight = weight
                meas.valid = valid
                meas.value = value
                meas.status = status
                if scale_context is not None:
                    if is_angle_measurement:
                        mark_measurement_invalid(meas)
                        meas.value = 0.0 if getattr(scale_context, "flat_start", False) else math.radians(float(meas.value))
                    else:
                        normalized_value = _normalize_measurement_value(scale_context, meas)
                        if normalized_value is None:
                            mark_measurement_invalid(meas)
                        else:
                            meas.value = normalized_value
                append_measurement(meas)
                append_idx(idx)
                append_name(name)
                append_device_type(device_type)
                append_device_name(device_name)
                append_meas_type(meas_type)
                append_weight(weight)
                append_valid(meas.valid)
                append_value(meas.value if scale_context is not None else value)
                append_device_type_code(device_type_code)
                append_angle_mask(is_angle_measurement)
                append_status(meas.status)
                code_rows.append(row_pos)
                row_pos += 1
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line == "</Measurement>":
                break
            if first != "#":
                raise SyntaxError(f"Invalid Measurement row at line {line_no} in {file_name}")
    if not in_measurement:
        raise RuntimeError(f"{file_name} does not contain a <Measurement> block")
    if header is None:
        raise RuntimeError(f"{file_name} Measurement block does not contain a header")
    device_type_code_array = np.asarray(device_type_code_values, dtype=np.int16)
    rows_by_code = {
        int(code): np.asarray(rows, dtype=np.int64)
        for code, rows in rows_by_code_values.items()
    }
    table = MeasurementTable(
        idx=np.asarray(idx_values, dtype=np.int64),
        name=np.asarray(name_values, dtype=object),
        device_type=np.asarray(device_type_values, dtype=object),
        device_name=np.asarray(device_name_values, dtype=object),
        meas_type=np.asarray(meas_type_values, dtype=object),
        weight=np.asarray(weight_values, dtype=np.float64),
        valid=np.asarray(valid_values, dtype=bool),
        value=np.asarray(value_values, dtype=np.float64),
        device_type_code=device_type_code_array,
        angle_mask=np.asarray(angle_mask_values, dtype=bool),
        status_code=np.asarray(status_values, dtype=np.int16),
        rows_by_device_type_code=rows_by_code,
    )
    return MeasurementList(measurements, table, normalized=scale_context is not None)


def _measurement_table_from_measurements(measurements: Sequence["Measurement"]) -> MeasurementTable:
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=_DEVICE_TYPE_CODES,
        angle_measurement_types=ANGLE_MEASUREMENT_TYPES,
    )


def _measurement_scale_lookup(
    scale_context,
    device_type: str,
    device_name: str,
    meas_type: str,
) -> Optional[float]:
    if scale_context is None:
        return None
    if meas_type in ANGLE_MEASUREMENT_TYPES:
        return math.pi / 180.0
    power_scale = float(getattr(scale_context, "p_base", 1.0))
    network = getattr(scale_context, "network", None)
    node_by_name = getattr(scale_context, "node_by_name", {})
    branch_by_name = getattr(scale_context, "branch_by_name", {})
    transformer_by_name = getattr(scale_context, "transformer_by_name", {})
    zero_branch_by_name = getattr(scale_context, "zero_branch_by_name", {})
    generator_by_name = getattr(scale_context, "generator_by_name", {})
    load_by_name = getattr(scale_context, "load_by_name", {})

    def node_scale(node_idx: int) -> float:
        if network is None:
            return 1.0
        node = network.node_dict[node_idx]
        return float(getattr(scale_context, "u_scale", 1.0)) * float(node.vbase)

    def current_scale(node_idx: int) -> float:
        if network is None:
            return 1.0
        node = network.node_dict[node_idx]
        return float(getattr(scale_context, "i_scale", 1.0)) * ac_current_base_ka(
            float(getattr(scale_context, "p_base_kW", power_scale)),
            float(node.vbase),
        )

    if device_type == "ACNode":
        if meas_type == "V":
            node = node_by_name.get(device_name)
            return node_scale(node.idx) if node is not None else None
        return 1.0
    if device_type in {"ACBranch", "ACTransformer", "ACZeroBranch", "ACBreak"}:
        device = (
            branch_by_name.get(device_name)
            if device_type == "ACBranch"
            else transformer_by_name.get(device_name)
            if device_type == "ACTransformer"
            else zero_branch_by_name.get(device_name)
            if device_type == "ACZeroBranch"
            else getattr(scale_context, "break_by_name", {}).get(device_name)
            if device_type == "ACBreak"
            else None
        )
        if device is None:
            return None
        if meas_type.startswith(("P_", "Q_")):
            return power_scale
        if meas_type.endswith("_FROM"):
            node_idx = device.i_node
        elif meas_type.endswith("_TO"):
            node_idx = device.j_node
        else:
            return 1.0
        if meas_type.startswith("V_"):
            return node_scale(node_idx)
        if meas_type.startswith("I_"):
            return current_scale(node_idx)
        return 1.0
    if device_type == "ACGenerator":
        gen = generator_by_name.get(device_name)
        if gen is None:
            return None
        if meas_type in ("P_GEN", "Q_GEN"):
            return power_scale
        if meas_type == "V_GEN":
            return node_scale(gen.node)
        if meas_type == "I_GEN":
            return current_scale(gen.node)
        return 1.0
    if device_type == "ACLoad":
        load = load_by_name.get(device_name)
        if load is None:
            return None
        if meas_type in ("P_LOAD", "Q_LOAD"):
            return power_scale
        if meas_type == "V_LOAD":
            return node_scale(load.node)
        if meas_type == "I_LOAD":
            return current_scale(load.node)
        return 1.0
    return None


def _normalize_measurement_value(scale_context, meas: "Measurement") -> Optional[float]:
    scale = _measurement_scale_lookup(scale_context, meas.device_type, meas.device_name, meas.meas_type)
    if scale is None:
        return None
    if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
        return math.radians(float(meas.value))
    return float(meas.value) / scale


def _build_ac_se_network_from_ppc(e_file: Path) -> ACPowerNetwork:
    source = Path(e_file).resolve()
    ppc = build_ac_ppc_from_e_file(source)
    ppc["source"] = str(source)
    topology_arrays = ppc.get("_topology_arrays")
    if topology_arrays is None:
        topology_arrays = network_topology.prepare_ac_topology_ppc(ppc)
        ppc["_topology_arrays"] = topology_arrays
    network = ACPowerNetwork()
    base = ppc["base"]
    network.p_base = float(base[0])
    network.u_scale = float(base[1])
    network.p_scale = float(base[2])
    network.i_scale = float(base[3])
    network.p_base_kW = float(base[4])
    network.model = SimpleNamespace(_named_units_normalized=True)

    bus = ppc["bus"]
    network.nodes = []
    append_node = network.nodes.append
    bus_names = ppc["bus_name"]
    for pos, row in enumerate(bus):
        node = _ACArrayObject()
        node.idx = int(row[BUS_COLS["idx"]])
        node.name = str(bus_names[pos])
        node.vbase = float(row[BUS_COLS["vbase"]])
        node.voltage = float(row[BUS_COLS["voltage"]])
        node.angle = float(row[BUS_COLS["angle"]])
        node.run_stat = int(row[BUS_COLS["run_stat"]])
        node.is_alive = False
        append_node(node)

    def make_branch_like(array, names, cols, extra=None):
        devices = []
        append_device = devices.append
        idx_col = cols["idx"]
        i_node_col = cols["i_node"]
        j_node_col = cols["j_node"]
        run_stat_col = cols["run_stat"]
        r_col = extra.get("r") if extra else None
        x_col = extra.get("x") if extra else None
        b_col = extra.get("b") if extra else None
        gt_col = extra.get("gt") if extra else None
        bt_col = extra.get("bt") if extra else None
        tap_col = extra.get("tap") if extra else None
        shift_col = extra.get("shift") if extra else None
        status_col = extra.get("status") if extra else None
        for pos, row in enumerate(array):
            dev = _ACArrayObject()
            dev.idx = int(row[idx_col])
            dev.name = str(names[pos])
            dev.i_node = int(row[i_node_col])
            dev.j_node = int(row[j_node_col])
            dev.run_stat = int(row[run_stat_col])
            dev.p = None
            dev.q = None
            dev.current = None
            dev.i_p = None
            dev.i_q = None
            dev.i_c = None
            dev.j_p = None
            dev.j_q = None
            dev.j_c = None
            if r_col is not None:
                dev.r = float(row[r_col])
            if x_col is not None:
                dev.x = float(row[x_col])
            if b_col is not None:
                dev.b = float(row[b_col])
            if gt_col is not None:
                dev.gt = float(row[gt_col])
            if bt_col is not None:
                dev.bt = float(row[bt_col])
            if tap_col is not None:
                dev.tap = float(row[tap_col])
            if shift_col is not None:
                dev.shift = float(row[shift_col])
            if status_col is not None:
                dev.status = int(row[status_col])
            append_device(dev)
        return devices

    network.branches = make_branch_like(
        ppc["branch"],
        ppc["branch_name"],
        BRANCH_COLS,
        {"r": BRANCH_COLS["r"], "x": BRANCH_COLS["x"], "b": BRANCH_COLS["b"]},
    )
    network.transformers = make_branch_like(
        ppc["transformer"],
        ppc["transformer_name"],
        TRANSFORMER_COLS,
        {
            "r": TRANSFORMER_COLS["r"],
            "x": TRANSFORMER_COLS["x"],
            "gt": TRANSFORMER_COLS["gt"],
            "bt": TRANSFORMER_COLS["bt"],
            "tap": TRANSFORMER_COLS["tap"],
            "shift": TRANSFORMER_COLS["shift"],
        },
    )
    network.zero_branches = make_branch_like(ppc["zero_branch"], ppc["zero_branch_name"], ZERO_BRANCH_COLS)
    network.switches = make_branch_like(
        ppc["switch"],
        ppc["switch_name"],
        SWITCH_COLS,
        {"status": SWITCH_COLS["status"]},
    )
    network.breakers = make_branch_like(
        ppc.get("break", np.zeros((0, len(BREAK_COLS)))),
        ppc.get("break_name", np.asarray([], dtype=object)),
        BREAK_COLS,
        {"status": BREAK_COLS["status"]},
    )
    network.generators = []
    append_gen = network.generators.append
    gen_names = ppc["gen_name"]
    for pos, row in enumerate(ppc["gen"]):
        gen = _ACArrayObject()
        gen.idx = int(row[GEN_COLS["idx"]])
        gen.name = str(gen_names[pos])
        gen.node = int(row[GEN_COLS["node"]])
        gen.control_type = _CTRL_NAME_BY_CODE.get(int(row[GEN_COLS["control_type"]]), "PQ")
        gen.p_set = float(row[GEN_COLS["p_set"]])
        gen.q_set = float(row[GEN_COLS["q_set"]])
        gen.v_set = float(row[GEN_COLS["v_set"]])
        gen.alpha = float(row[GEN_COLS["alpha"]])
        gen.run_stat = int(row[GEN_COLS["run_stat"]])
        gen.p = None
        gen.q = None
        gen.current = None
        append_gen(gen)
    network.loads = []
    append_load = network.loads.append
    load_names = ppc["load_name"]
    for pos, row in enumerate(ppc["load"]):
        load = _ACArrayObject()
        load.idx = int(row[LOAD_COLS["idx"]])
        load.name = str(load_names[pos])
        load.node = int(row[LOAD_COLS["node"]])
        load.pbase = float(row[LOAD_COLS["pbase"]])
        load.pv0 = float(row[LOAD_COLS["pv0"]])
        load.pv1 = float(row[LOAD_COLS["pv1"]])
        load.pv2 = float(row[LOAD_COLS["pv2"]])
        load.qbase = float(row[LOAD_COLS["qbase"]])
        load.qv0 = float(row[LOAD_COLS["qv0"]])
        load.qv1 = float(row[LOAD_COLS["qv1"]])
        load.qv2 = float(row[LOAD_COLS["qv2"]])
        load.run_stat = int(row[LOAD_COLS["run_stat"]])
        load.p = None
        load.q = None
        load.current = None
        append_load(load)
    network.shunt_compensators = []
    append_shunt = network.shunt_compensators.append
    shunt_names = ppc["shunt_name"]
    for pos, row in enumerate(ppc["shunt"]):
        shunt = _ACArrayObject()
        shunt.idx = int(row[SHUNT_COLS["idx"]])
        shunt.name = str(shunt_names[pos])
        shunt.node = int(row[SHUNT_COLS["node"]])
        shunt.control_type = _SHUNT_NAME_BY_CODE.get(int(row[SHUNT_COLS["control_type"]]), "Q")
        shunt.q_set = float(row[SHUNT_COLS["q_set"]])
        shunt.g_set = float(row[SHUNT_COLS["g_set"]])
        shunt.b_set = float(row[SHUNT_COLS["b_set"]])
        shunt.v_set = float(row[SHUNT_COLS["v_set"]])
        shunt.run_stat = int(row[SHUNT_COLS["run_stat"]])
        shunt.p = None
        shunt.q = None
        shunt.current = None
        append_shunt(shunt)
    network._array_model = ppc
    network._topology_arrays = topology_arrays
    network_topology.apply_ac_topology_arrays(network, topology_arrays, compact=True, build_alive_maps=False)
    return network


class ACStateEstimator:
    def __init__(
        self,
        e_file: Path = DEFAULT_CASE,
        meas_file: Path = DEFAULT_MEAS,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        diff_step: Optional[float] = None,
        flat_start: Optional[bool] = None,
        parameter_file: Path = DEFAULT_SE_PARAMETER_FILE,
        parameters: Optional[StateEstimationParameters] = None,
        profile: bool = False,
        network: Optional[ACPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
        auto_prepare: bool = True,
    ):
        self.profile_enabled = bool(profile)
        self.profile_times: Dict[str, float] = {}
        self.params = (parameters or load_se_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            diff_step=diff_step,
            flat_start=flat_start,
        )
        self.e_file = Path(e_file)
        self.meas_file = Path(meas_file)
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.diff_step = self.params.diff_step
        self.flat_start = self.params.flat_start
        self.pseudo_measurement_weight = self.params.pseudo_measurement_weight
        self.targeted_pseudo_measurement_max = self.params.targeted_pseudo_measurement_max
        self.targeted_pseudo_measurement_redundancy_ratio = (
            self.params.targeted_pseudo_measurement_redundancy_ratio
        )
        self.targeted_pseudo_measurement_step = self.params.targeted_pseudo_measurement_step
        self.voltage_floor = self.params.voltage_floor
        self.min_current_voltage = self.params.min_current_voltage

        self._prepared = False
        self._prepare_network = network
        self._prepare_measurements = measurements
        self._prepare_active_measurements = bool(prepare_active_measurements)
        self._prepare_defer_finalize = bool(defer_prepare_finalize)
        self.observability_result = None
        self.estimate_result = None
        self.removed_bad_data: List[BadDataItem] = []
        self.bad_items: List[BadDataItem] = []
        self.normalized_residual = np.array([], dtype=np.float64)
        self.se_result = None
        if auto_prepare:
            self.prepare(
                network=network,
                measurements=measurements,
                prepare_active_measurements=prepare_active_measurements,
            )

    def prepare(
        self,
        *,
        network: Optional[ACPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: Optional[bool] = None,
        defer_prepare_finalize: Optional[bool] = None,
    ) -> "ACStateEstimator":
        if (
            self._prepared
            and network is None
            and measurements is None
            and prepare_active_measurements is None
            and defer_prepare_finalize is None
        ):
            return self
        if network is None:
            network = self._prepare_network
        else:
            self._prepare_network = network
        if measurements is None:
            measurements = self._prepare_measurements
        else:
            self._prepare_measurements = measurements
        if prepare_active_measurements is None:
            prepare_active_measurements = self._prepare_active_measurements
        else:
            self._prepare_active_measurements = bool(prepare_active_measurements)
        if defer_prepare_finalize is None:
            defer_prepare_finalize = self._prepare_defer_finalize
        else:
            self._prepare_defer_finalize = bool(defer_prepare_finalize)
        prepare_active_measurements = bool(prepare_active_measurements)
        defer_prepare_finalize = bool(defer_prepare_finalize)
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        self.network = network if network is not None else self._load_network(self.e_file)
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            self.measurements = self._load_measurements(self.meas_file)
        elif isinstance(measurements, MeasurementList):
            self.measurements = measurements
        else:
            self.measurements = list(measurements)
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)

        topology_arrays = getattr(self.network, "_topology_arrays", None)
        if topology_arrays is None:
            source_ppc = getattr(self.network, "_array_model", None)
            topology_arrays = source_ppc.get("_topology_arrays") if isinstance(source_ppc, dict) else None
        use_topology_arrays = (
            topology_arrays is not None
            and len(getattr(self.network, "buses", ())) == len(getattr(topology_arrays, "bus_ids", ()))
            and len(getattr(self.network, "nodes", ())) == len(getattr(topology_arrays, "node_ids", ()))
        )
        if use_topology_arrays:
            active_bus_pos = np.flatnonzero(topology_arrays.bus_alive_mask).astype(np.int32, copy=False)
            self.nodes = [self.network.buses[int(pos)] for pos in active_bus_pos]
        else:
            self.nodes = (
                getattr(self.network, "alive_nodes", None)
                or getattr(self.network, "alive_buses", None)
                or [bus for bus in getattr(self.network, "buses", []) if getattr(bus, "is_alive", False)]
                or [node for node in self.network.nodes if getattr(node, "is_alive", False)]
            )
            self.nodes = sorted(self.nodes, key=lambda item: item.idx)
        if not self.nodes:
            raise RuntimeError("No alive AC nodes are available for state estimation")
        self.file_theta = np.asarray(
            [float(getattr(node, "angle", 0.0) or 0.0) for node in self.nodes],
            dtype=np.float64,
        )
        self.file_voltage = np.asarray(
            [float(getattr(node, "voltage", 1.0) or 1.0) for node in self.nodes],
            dtype=np.float64,
        )

        self.node_pos = {}
        self.node_by_name = {}
        self.node_by_idx = {}
        if use_topology_arrays:
            bus_solver_pos = np.full(len(topology_arrays.bus_ids), -1, dtype=np.int32)
            bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
            for node_row, node in enumerate(self.network.nodes):
                bus_pos = int(topology_arrays.node_to_bus_pos[node_row])
                if bus_pos < 0:
                    continue
                solver_pos = int(bus_solver_pos[bus_pos])
                if solver_pos < 0:
                    continue
                bus = self.nodes[solver_pos]
                self.node_pos[int(node.idx)] = solver_pos
                self.node_by_name[str(node.name)] = bus
                self.node_by_idx[int(node.idx)] = bus
            for pos, bus in enumerate(self.nodes):
                self.node_pos.setdefault(int(bus.idx), pos)
                self.node_by_name.setdefault(str(bus.name), bus)
                self.node_by_idx.setdefault(int(bus.idx), bus)
        else:
            for pos, bus in enumerate(self.nodes):
                members = getattr(bus, "nodes", None) or [bus]
                for node in members:
                    self.node_pos[node.idx] = pos
                    self.node_by_name[node.name] = bus
                    self.node_by_idx[node.idx] = bus
                self.node_pos.setdefault(bus.idx, pos)
                self.node_by_name.setdefault(bus.name, bus)
                self.node_by_idx.setdefault(bus.idx, bus)
        if use_topology_arrays:
            devices = topology_arrays.devices

            def alive_by_name(items, key):
                mask = devices[key].alive_mask
                return {dev.name: dev for dev, is_alive in zip(items, mask) if bool(is_alive)}

            self.branch_by_name = alive_by_name(self.network.branches, "branch")
            self.transformer_by_name = alive_by_name(self.network.transformers, "transformer")
            self.generator_by_name = alive_by_name(self.network.generators, "gen")
            self.load_by_name = alive_by_name(self.network.loads, "load")
            self.zero_branch_by_name = alive_by_name(self.network.zero_branches, "zero_branch")
            self.switch_by_name = alive_by_name(self.network.switches, "switch")
            self.break_by_name = alive_by_name(getattr(self.network, "breakers", []), "break")
            self.zero_branches = sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
            self.switches = sorted(self.switch_by_name.values(), key=lambda item: item.idx)
            self.breakers = sorted(self.break_by_name.values(), key=lambda item: item.idx)
            self.generator_order = sorted(self.generator_by_name.values(), key=lambda item: item.idx)
            self.load_order = sorted(self.load_by_name.values(), key=lambda item: item.idx)
        elif hasattr(self.network, "alive_branch_by_name"):
            self.branch_by_name = self.network.alive_branch_by_name
            self.transformer_by_name = self.network.alive_transformer_by_name
            self.generator_by_name = self.network.alive_generator_by_name
            self.load_by_name = self.network.alive_load_by_name
            self.zero_branch_by_name = self.network.alive_zero_branch_by_name
            self.switch_by_name = self.network.alive_switch_by_name
            self.break_by_name = self.network.alive_break_by_name
            self.zero_branches = self.network.alive_zero_branches
            self.switches = self.network.alive_switches
            self.breakers = self.network.alive_breakers
        else:
            self.branch_by_name = {br.name: br for br in self.network.branches if getattr(br, "is_alive", False)}
            self.transformer_by_name = {tr.name: tr for tr in self.network.transformers if getattr(tr, "is_alive", False)}
            self.generator_by_name = {gen.name: gen for gen in self.network.generators if getattr(gen, "is_alive", False)}
            self.load_by_name = {load.name: load for load in self.network.loads if getattr(load, "is_alive", False)}
            self.zero_branch_by_name = {
                zbr.name: zbr
                for zbr in self.network.zero_branches
                if getattr(zbr, "is_alive", False)
            }
            self.switch_by_name = {sw.name: sw for sw in self.network.switches if getattr(sw, "is_alive", False)}
            self.break_by_name = {brk.name: brk for brk in getattr(self.network, "breakers", []) if getattr(brk, "is_alive", False)}
            self.zero_branches = sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
            self.switches = sorted(self.switch_by_name.values(), key=lambda item: item.idx)
            self.breakers = sorted(self.break_by_name.values(), key=lambda item: item.idx)
        if use_topology_arrays:
            pass
        elif hasattr(self.network, "alive_generator_order"):
            self.generator_order = self.network.alive_generator_order
            self.load_order = self.network.alive_load_order
        else:
            self.generator_order = sorted(self.generator_by_name.values(), key=lambda item: item.idx)
            self.load_order = sorted(self.load_by_name.values(), key=lambda item: item.idx)
        self.generator_pos_array = np.asarray([self.node_pos[gen.node] for gen in self.generator_order], dtype=np.int32)
        self.load_pos_array = np.asarray([self.node_pos[load.node] for load in self.load_order], dtype=np.int32)
        self.n_nodes = len(self.nodes)
        self._defer_prepare_finalize_pending = bool(defer_prepare_finalize)
        if defer_prepare_finalize:
            self.power_flow_seed_converged = False
            self.targeted_observability_pseudo_count = 0
            return self
        # Reference-bus voltage values must be known before the compact state layout
        # is built, so real file measurements are normalized after the estimator maps exist.
        self.finalize_prepare(prepare_active_measurements=prepare_active_measurements)
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        self._prepared = True
        return self

    def _require_prepared(self, action: str) -> None:
        if not self._prepared:
            raise RuntimeError(f"Call prepare() before {action}.")

    def finalize_prepare(
        self,
        *,
        prepare_active_measurements: bool = True,
        measurements_already_normalized: bool = False,
    ) -> "ACStateEstimator":
        self._defer_prepare_finalize_pending = False
        if not measurements_already_normalized:
            stage_start = time.perf_counter()
            self._convert_measurements_to_pu()
            self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
        self.power_flow_seed_converged = False
        if not self.flat_start:
            seed_start = time.perf_counter()
            stage_start = time.perf_counter()
            self._apply_measurement_seed_to_network()
            self._record_profile_time("seed.apply_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self.power_flow_seed_converged = bool(self._run_power_flow_seed(self.network, self.params, self.e_file))
            run_seed = getattr(type(self), "_run_power_flow_seed", None)
            original_run_seed = globals().get("_ORIGINAL_AC_RUN_POWER_FLOW_SEED")
            if (
                not self.power_flow_seed_converged
                and original_run_seed is not None
                and run_seed is original_run_seed
            ):
                self._apply_measurement_seed_to_network(force_object=True)
            self._record_profile_time("seed.lf", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_file_state_from_network()
            self._record_profile_time("seed.refresh_file_state", time.perf_counter() - stage_start)
            self._record_profile_time("seed.total", time.perf_counter() - seed_start)
        stage_start = time.perf_counter()
        self._refresh_load_parameter_arrays()
        self.node_voltage_measurements = self._node_voltage_measurements()
        self.node_degrees = self._node_incident_degrees()
        self.references = self._select_reference_nodes()
        self.ref_idx = {self.node_pos[node.idx] for node in self.references}
        self.reference_voltage_by_pos = {
            self.node_pos[node.idx]: self.node_voltage_measurements[node.idx]
            for node in self.references
            if node.idx in self.node_voltage_measurements and node.idx in self.node_pos
        }
        self.reference_angle_by_pos = self._reference_angle_offsets()
        self._rebase_angle_measurements()
        self._build_zero_tie_state_layout()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        self.zero_current_devices = (
            [("Z", zbr) for zbr in self.zero_branches]
            + [("B", brk) for brk in self.breakers]
        )
        self.zero_current_pos = {(kind, dev.name): pos for pos, (kind, dev) in enumerate(self.zero_current_devices)}
        self.zero_branch_pos = {zbr.name: self.zero_current_pos[("Z", zbr.name)] for zbr in self.zero_branches}
        self.switch_pos = {}
        self.break_pos = {brk.name: self.zero_current_pos[("B", brk.name)] for brk in self.breakers}
        self.zero_current_by_node = {pos: [] for pos in range(len(self.nodes))}
        for idx, (_, dev) in enumerate(self.zero_current_devices):
            if dev.i_node in self.node_pos:
                self.zero_current_by_node[self.node_pos[dev.i_node]].append((idx, True))
            if dev.j_node in self.node_pos:
                self.zero_current_by_node[self.node_pos[dev.j_node]].append((idx, False))
        self.zero_current_i = np.asarray(
            [self.node_pos[dev.i_node] for _, dev in self.zero_current_devices],
            dtype=np.int32,
        )
        self.zero_current_j = np.asarray(
            [self.node_pos[dev.j_node] for _, dev in self.zero_current_devices],
            dtype=np.int32,
        )
        gen_seed = [self._generator_pseudo_power(gen) for gen in self.generator_order]
        load_seed = [self._load_pseudo_power(load) for load in self.load_order]
        self.initial_gen_p_array = np.asarray([p for p, _q in gen_seed], dtype=np.float64)
        self.initial_gen_q_array = np.asarray([q for _p, q in gen_seed], dtype=np.float64)
        self.initial_load_p_array = np.asarray([p for p, _q in load_seed], dtype=np.float64)
        self.initial_load_q_array = np.asarray([q for _p, q in load_seed], dtype=np.float64)
        self.n_generator_power = len(self.generator_order)
        self.n_load_power = len(self.load_order)
        self.n_switch_current = len(self.zero_current_devices)
        self.switch_current_idx_array = np.arange(self.n_switch_current, dtype=np.int32)
        self.switch_balance_end_pos = np.concatenate((self.zero_current_i, self.zero_current_j)).astype(
            np.int64,
            copy=False,
        )
        self.switch_balance_end_current_idx = np.concatenate(
            (self.switch_current_idx_array, self.switch_current_idx_array)
        )
        self.switch_balance_sign = np.concatenate(
            (
                np.ones(self.n_switch_current, dtype=np.float64),
                -np.ones(self.n_switch_current, dtype=np.float64),
            )
        )
        self.generator_power_idx_array = np.arange(self.n_generator_power, dtype=np.int32)
        self.load_power_idx_array = np.arange(self.n_load_power, dtype=np.int32)
        self.voltage_control_shunt_order = self._voltage_control_shunts()
        self.n_shunt_q = len(self.voltage_control_shunt_order)
        self.shunt_q_pos_array = np.asarray(
            [self.node_pos[shunt.node] for shunt in self.voltage_control_shunt_order],
            dtype=np.int32,
        )
        self.shunt_q_idx_array = np.arange(self.n_shunt_q, dtype=np.int32)
        self.generator_balance_minus_ones = -np.ones(self.n_generator_power, dtype=np.float64)
        self.load_balance_ones = np.ones(self.n_load_power, dtype=np.float64)
        self.shunt_balance_minus_ones = -np.ones(self.n_shunt_q, dtype=np.float64)
        self.base_switch_re = self.n_angle + self.n_voltage
        self.base_switch_im = self.base_switch_re + self.n_switch_current
        self.base_gen_p = self.base_switch_im + self.n_switch_current
        self.base_gen_q = self.base_gen_p + self.n_generator_power
        self.base_load_p = self.base_gen_q + self.n_generator_power
        self.base_load_q = self.base_load_p + self.n_load_power
        self.base_shunt_q = self.base_load_q + self.n_load_power
        self.n_state = self.base_shunt_q + self.n_shunt_q
        self.gen_p_col_by_name = {
            gen.name: self.base_gen_p + idx for idx, gen in enumerate(self.generator_order)
        }
        self.gen_q_col_by_name = {
            gen.name: self.base_gen_q + idx for idx, gen in enumerate(self.generator_order)
        }
        self.generator_state_index_by_name = {
            gen.name: idx for idx, gen in enumerate(self.generator_order)
        }
        self.load_p_col_by_name = {
            load.name: self.base_load_p + idx for idx, load in enumerate(self.load_order)
        }
        self.load_q_col_by_name = {
            load.name: self.base_load_q + idx for idx, load in enumerate(self.load_order)
        }
        self.load_state_index_by_name = {
            load.name: idx for idx, load in enumerate(self.load_order)
        }
        self.shunt_q_col_by_name = {
            shunt.name: self.base_shunt_q + idx for idx, shunt in enumerate(self.voltage_control_shunt_order)
        }
        self.shunt_q_state_index_by_name = {
            shunt.name: idx for idx, shunt in enumerate(self.voltage_control_shunt_order)
        }
        self.initial_shunt_q_array = np.asarray(
            [self._initial_voltage_control_shunt_q(shunt) for shunt in self.voltage_control_shunt_order],
            dtype=np.float64,
        )
        state_meta: List[StateMeta] = []
        state_meta.extend(
            StateMeta("ac", "angle", "ACNode", self.nodes[pos].name, component="theta", legacy_label=f"theta:{self.nodes[pos].name}")
            for pos in self.angle_state_pos
        )
        state_meta.extend(
            StateMeta("ac", "voltage", "ACNode", self.nodes[pos].name, component="magnitude", legacy_label=f"V:{self.nodes[pos].name}")
            for pos in self.voltage_state_pos
        )
        for kind, dev in self.zero_current_devices:
            device_type = "ACZeroBranch" if kind == "Z" else "ACBreak"
            state_kind = "zero_current" if kind == "Z" else "break_current"
            state_meta.append(
                StateMeta("ac", state_kind, device_type, dev.name, component="re", legacy_label=f"I_{kind}_RE:{dev.name}")
            )
        for kind, dev in self.zero_current_devices:
            device_type = "ACZeroBranch" if kind == "Z" else "ACBreak"
            state_kind = "zero_current" if kind == "Z" else "break_current"
            state_meta.append(
                StateMeta("ac", state_kind, device_type, dev.name, component="im", legacy_label=f"I_{kind}_IM:{dev.name}")
            )
        state_meta.extend(
            StateMeta("ac", "generator_p", "ACGenerator", gen.name, component="p", legacy_label=f"P_GEN:{gen.name}")
            for gen in self.generator_order
        )
        state_meta.extend(
            StateMeta("ac", "generator_q", "ACGenerator", gen.name, component="q", legacy_label=f"Q_GEN:{gen.name}")
            for gen in self.generator_order
        )
        state_meta.extend(
            StateMeta("ac", "load_p", "ACLoad", load.name, component="p", legacy_label=f"P_LOAD:{load.name}")
            for load in self.load_order
        )
        state_meta.extend(
            StateMeta("ac", "load_q", "ACLoad", load.name, component="q", legacy_label=f"Q_LOAD:{load.name}")
            for load in self.load_order
        )
        state_meta.extend(
            StateMeta("ac", "shunt_q", "ACShuntCompensator", shunt.name, component="q", legacy_label=f"Q_SHUNT:{shunt.name}")
            for shunt in self.voltage_control_shunt_order
        )
        self.state_meta = state_meta
        labels = [meta.legacy_label for meta in state_meta]
        self.state_labels = labels if all(labels) else state_labels_from_metadata(self.state_meta)

        self.branch_stamp_by_name = self._build_branch_stamp_map(list(self.branch_by_name.values()), False)
        self.transformer_stamp_by_name = self._build_branch_stamp_map(list(self.transformer_by_name.values()), True)
        self._build_measurement_plan_lookup_arrays()

        stage_start = time.perf_counter()
        self.Y = self._build_y_matrix()
        self._prepare_y_row_cache()
        self.loads_at_pos = self._group_loads()
        self.generators_at_pos = self._group_generators()
        self.generator_share_by_name = self._generator_shares()
        self._record_profile_time("init.network_matrices", time.perf_counter() - stage_start)
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        stage_start = time.perf_counter()
        self._seed_power_state_arrays_from_measurements()
        self._record_profile_time("init.seed_power_states", time.perf_counter() - stage_start)
        self.targeted_observability_pseudo_count = 0
        if prepare_active_measurements:
            stage_start = time.perf_counter()
            self._add_pseudo_power_measurements()
            self._add_power_balance_constraint_measurements()
            self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_active_measurement_indexes()
            self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
            self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        else:
            self.active_measurements = MeasurementList(
                [],
                normalized=getattr(self.measurements, "normalized", False),
            )
            self.active_measurement_rows = np.array([], dtype=np.int64)
            self.active_z = np.array([], dtype=np.float64)
            self.active_weight = np.array([], dtype=np.float64)
            self.active_angle_residual_mask = np.array([], dtype=bool)
            self.active_has_angle_residuals = False
            self.active_uniform_weight = None
            self.active_weights_are_uniform = False
            self._active_rows_by_device_type_code = {}
            self._branch_transformer_vector_plan_cache = {}
            self._simple_jacobian_plan_cache = {}
            self._zero_current_vector_plan_cache = {}
            self._generator_measurement_plan_cache = {}
            self._balance_measurement_plan_cache = {}
            self._active_branch_transformer_vector_plan = None
            self._active_simple_jacobian_plan = None
            self._active_zero_current_vector_plan = None
            self._active_generator_measurement_plan = None
            self._active_balance_measurement_plan = None
            self._active_normal_pattern = None
            self.active_measurements_are_vectorized = False
        self._prepared = True
        return self

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    def _default_observability_cache_key(self) -> Tuple[object, ...]:
        return (
            _file_cache_key(self.e_file),
            _file_cache_key(self.meas_file),
            bool(self.flat_start),
            int(self.targeted_pseudo_measurement_max),
            float(self.targeted_pseudo_measurement_redundancy_ratio),
            int(self.targeted_pseudo_measurement_step),
            int(len(self.active_measurements)),
            int(self._max_measurement_idx),
            int(self.n_state),
        )

    def _cache_observability_matrix(
        self,
        result: ObservabilityResult,
        x: np.ndarray,
        measurements: Sequence[Measurement],
        H,
    ) -> None:
        self._observability_matrix_cache = {
            "result": result,
            "measurements": measurements,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "H": H,
            "normal_pattern": None,
        }

    def _observability_matrix_cache_for(
        self,
        result: Optional[ObservabilityResult],
        measurements: Sequence[Measurement],
        x: np.ndarray,
    ):
        cache = getattr(self, "_observability_matrix_cache", None)
        if cache is None or cache.get("result") is not result:
            return None
        if cache.get("measurements") is not measurements:
            return None
        cached_x = cache.get("x")
        x_array = np.asarray(x, dtype=np.float64)
        if cached_x is None or cached_x.shape != x_array.shape or not np.array_equal(cached_x, x_array):
            return None
        return cache

    def _refresh_active_measurement_indexes(self) -> None:
        """Rebuild active measurement arrays and vectorized measurement plans."""
        self._initial_observability_cache = None
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_measurements,
        )
        table = active_view.source_table
        active_table = active_view.table
        self.measurement_table = table
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_angle_residual_mask = active_view.angle_mask
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        self._active_branch_transformer_vector_plan = None
        self._active_branch_transformer_vector_plan = self._branch_transformer_vector_plan(self.active_measurements)
        self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_branch_transformer_vector_plan,
        )
        self._active_simple_jacobian_plan = None
        self._active_simple_jacobian_plan = self._simple_jacobian_plan(self.active_measurements)
        self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_simple_jacobian_plan,
        )
        self._active_zero_current_vector_plan = None
        self._active_zero_current_vector_plan = self._zero_current_vector_plan(self.active_measurements)
        self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_zero_current_vector_plan,
        )
        self._active_generator_measurement_plan = None
        self._active_generator_measurement_plan = self._generator_measurement_plan(self.active_measurements)
        self._generator_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        self._active_balance_measurement_plan = None
        self._active_balance_measurement_plan = self._balance_measurement_plan(self.active_measurements)
        self._balance_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_balance_measurement_plan,
        )
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._observability_matrix_cache = None
        self.active_measurements_are_vectorized = bool(
            np.all(
                self._active_branch_transformer_vector_plan["handled_mask"]
                | self._active_simple_jacobian_plan["handled_mask"]
                | self._active_zero_current_vector_plan["handled_mask"]
                | self._active_generator_measurement_plan["handled_mask"]
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )

    def _incremental_update_active_measurement_indexes(self, appended_measurements: Sequence[Measurement]) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            return False
        if any((not meas.valid) or float(meas.weight) <= 0.0 for meas in appended_measurements):
            return False
        master_table = getattr(self, "measurement_table", getattr(self.measurements, "table", None))
        active_table = getattr(self, "active_measurement_table", getattr(self.active_measurements, "table", None))
        if master_table is None or active_table is None:
            return False
        if len(master_table.idx) != len(self.measurements) - len(appended_measurements):
            return False
        if len(active_table.idx) != len(self.active_measurements):
            return False

        appended_list = MeasurementList(
            list(appended_measurements),
            _measurement_table_from_measurements(appended_measurements),
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurement_table = concat_measurement_tables(master_table, appended_list.table)
        self.measurements.table = self.measurement_table
        active_view = append_active_measurement_view(
            build_active_measurement_view(self.active_measurements, table_builder=_measurement_table_from_measurements),
            appended_list,
            source_row_start=len(master_table.idx),
            table_builder=_measurement_table_from_measurements,
        )
        self._initial_observability_cache = None
        self._max_measurement_idx = int(self.measurement_table.idx.max()) if self.measurement_table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_view.table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_angle_residual_mask = active_view.angle_mask
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        row_offset = len(active_table.idx)
        previous_branch_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
        previous_simple_plan = getattr(self, "_active_simple_jacobian_plan", None)
        previous_zero_plan = getattr(self, "_active_zero_current_vector_plan", None)
        previous_generator_plan = getattr(self, "_active_generator_measurement_plan", None)
        previous_balance_plan = getattr(self, "_active_balance_measurement_plan", None)
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        if previous_branch_plan is None:
            self._active_branch_transformer_vector_plan = self._branch_transformer_vector_plan(self.active_measurements)
        else:
            self._active_branch_transformer_vector_plan = self._merge_active_plan_dict(
                previous_branch_plan,
                self._build_branch_transformer_vector_plan(appended_list),
                row_offset=row_offset,
                row_keys=("voltage_rows", "power_rows", "current_rows"),
            )
        self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_branch_transformer_vector_plan,
        )
        if previous_simple_plan is None:
            self._active_simple_jacobian_plan = self._simple_jacobian_plan(self.active_measurements)
        else:
            self._active_simple_jacobian_plan = self._merge_active_plan_dict(
                previous_simple_plan,
                self._build_simple_jacobian_plan(appended_list),
                row_offset=row_offset,
                row_keys=(
                    "scalar_rows",
                    "value_voltage_rows",
                    "value_angle_rows",
                    "value_voltage_diff_rows",
                    "value_angle_diff_rows",
                    "load_rows",
                ),
            )
        self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_simple_jacobian_plan,
        )
        if previous_zero_plan is None:
            self._active_zero_current_vector_plan = self._zero_current_vector_plan(self.active_measurements)
        else:
            self._active_zero_current_vector_plan = self._merge_active_plan_dict(
                previous_zero_plan,
                self._build_zero_current_vector_plan(appended_list),
                row_offset=row_offset,
                row_keys=(
                    "scalar_rows",
                    "voltage_rows",
                    "angle_diff_rows",
                    "voltage_diff_rows",
                    "power_rows",
                    "current_rows",
                ),
            )
        self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_zero_current_vector_plan,
        )
        if previous_generator_plan is None:
            self._active_generator_measurement_plan = self._generator_measurement_plan(self.active_measurements)
        else:
            self._active_generator_measurement_plan = self._merge_active_plan_dict(
                previous_generator_plan,
                self._build_generator_measurement_plan(appended_list),
                row_offset=row_offset,
                row_keys=("value_rows",),
            )
        self._generator_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        if previous_balance_plan is None:
            self._active_balance_measurement_plan = self._balance_measurement_plan(self.active_measurements)
        else:
            self._active_balance_measurement_plan = self._merge_active_plan_dict(
                previous_balance_plan,
                self._build_balance_measurement_plan(appended_list),
                row_offset=row_offset,
                row_keys=("rows",),
                mapped_row_keys=("p_row_by_pos", "q_row_by_pos"),
            )
        self._balance_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_balance_measurement_plan,
        )
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._observability_matrix_cache = None
        self.active_measurements_are_vectorized = bool(
            np.all(
                self._active_branch_transformer_vector_plan["handled_mask"]
                | self._active_simple_jacobian_plan["handled_mask"]
                | self._active_zero_current_vector_plan["handled_mask"]
                | self._active_generator_measurement_plan["handled_mask"]
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )
        return True

    @staticmethod
    def _merge_active_plan_dict(
        head: Dict[str, object],
        tail: Dict[str, object],
        *,
        row_offset: int,
        row_keys: Sequence[str],
        mapped_row_keys: Sequence[str] = (),
    ) -> Dict[str, object]:
        row_key_set = set(row_keys)
        mapped_key_set = set(mapped_row_keys)
        merged: Dict[str, object] = {}
        for key, head_value in head.items():
            tail_value = tail[key]
            if key in row_key_set:
                merged[key] = np.concatenate(
                    (
                        np.asarray(head_value, dtype=np.int64),
                        np.asarray(tail_value, dtype=np.int64) + int(row_offset),
                    )
                ).astype(np.int64, copy=False)
                continue
            if key in mapped_key_set:
                mapped = np.asarray(head_value, dtype=np.int32).copy()
                tail_rows = np.asarray(tail_value, dtype=np.int32)
                valid = tail_rows >= 0
                mapped[valid] = tail_rows[valid] + int(row_offset)
                merged[key] = mapped
                continue
            merged[key] = np.concatenate((np.asarray(head_value), np.asarray(tail_value)))
        return merged

    @staticmethod
    def _shrink_plan_rows(
        rows: Sequence[int],
        removed_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        row_array = np.asarray(rows, dtype=np.int64)
        keep = row_array != int(removed_pos)
        kept_rows = row_array[keep]
        kept_rows = kept_rows - (kept_rows > int(removed_pos))
        return kept_rows.astype(np.int64, copy=False), keep

    def _shrink_active_plan_group(
        self,
        plan: Dict[str, object],
        *,
        removed_pos: int,
        groups: Sequence[Tuple[str, Sequence[str]]],
        passthrough_keys: Sequence[str] = (),
    ) -> Dict[str, object]:
        shrunk: Dict[str, object] = {
            "handled_mask": np.delete(np.asarray(plan["handled_mask"], dtype=bool), int(removed_pos))
        }
        used_keys = {"handled_mask"}
        for row_key, value_keys in groups:
            shrunk_rows, keep = self._shrink_plan_rows(plan[row_key], removed_pos)
            shrunk[row_key] = shrunk_rows
            used_keys.add(row_key)
            for value_key in value_keys:
                shrunk[value_key] = np.asarray(plan[value_key])[keep]
                used_keys.add(value_key)
        for key in passthrough_keys:
            shrunk[key] = np.asarray(plan[key]).copy()
            used_keys.add(key)
        for key, value in plan.items():
            if key not in used_keys:
                shrunk[key] = np.asarray(value).copy()
        return shrunk

    def _shrink_balance_measurement_plan(
        self,
        plan: Dict[str, object],
        removed_pos: int,
    ) -> Dict[str, object]:
        shrunk_rows, keep = self._shrink_plan_rows(plan["rows"], removed_pos)
        shrunk_pos = np.asarray(plan["pos"], dtype=np.int64)[keep]
        shrunk_kind = np.asarray(plan["kind"], dtype=np.int64)[keep]
        p_row_by_pos = np.asarray(plan["p_row_by_pos"], dtype=np.int32).copy()
        q_row_by_pos = np.asarray(plan["q_row_by_pos"], dtype=np.int32).copy()
        for mapped in (p_row_by_pos, q_row_by_pos):
            removed_mask = mapped == int(removed_pos)
            mapped[removed_mask] = -1
            shift_mask = mapped > int(removed_pos)
            mapped[shift_mask] -= 1
        y_balance, y_nodes, y_nodes_i64, y_conj = self._balance_y_arrays(shrunk_pos)
        return {
            "handled_mask": np.delete(np.asarray(plan["handled_mask"], dtype=bool), int(removed_pos)),
            "rows": shrunk_rows,
            "pos": shrunk_pos.astype(np.int64, copy=False),
            "pos_i64": np.asarray(shrunk_pos, dtype=np.int64),
            "kind": shrunk_kind.astype(np.int64, copy=False),
            "p_row_by_pos": p_row_by_pos,
            "q_row_by_pos": q_row_by_pos,
            "y_balance": y_balance,
            "y_nodes": y_nodes,
            "y_nodes_i64": y_nodes_i64,
            "y_conj": y_conj,
        }

    def _shrink_active_measurement_indexes(self, removed_pos: int) -> MeasurementList:
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        self.active_measurements = take_measurement_view(self.active_measurements, keep_rows)
        self.active_measurement_table = self.active_measurements.table
        self.measurement_table = self.active_measurement_table
        self.active_measurement_rows = np.arange(len(self.active_measurements), dtype=np.int64)
        self.active_z = np.asarray(self.active_measurement_table.value, dtype=np.float64)
        self.active_weight = np.asarray(self.active_measurement_table.weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(self.active_measurement_table.angle_mask, dtype=bool)
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = rows_by_device_type_code(self.active_measurement_table)
        if not hasattr(self, "n_state"):
            return self.active_measurements
        self._initial_observability_cache = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._observability_matrix_cache = None
        if hasattr(self, "_active_branch_transformer_vector_plan"):
            self._active_branch_transformer_vector_plan = self._shrink_active_plan_group(
                self._active_branch_transformer_vector_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("voltage_rows", ("voltage_pos", "voltage_cols")),
                    ("power_rows", ("power_is_p", "power_own", "power_other", "power_y_self", "power_y_mutual")),
                    ("current_rows", ("current_own", "current_other", "current_y_self", "current_y_mutual")),
                ),
            )
        if hasattr(self, "_active_simple_jacobian_plan"):
            self._active_simple_jacobian_plan = self._shrink_active_plan_group(
                self._active_simple_jacobian_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("value_voltage_rows", ("value_voltage_pos",)),
                    ("value_angle_rows", ("value_angle_pos",)),
                    ("value_voltage_diff_rows", ("value_voltage_diff_i", "value_voltage_diff_j")),
                    ("value_angle_diff_rows", ("value_angle_diff_i", "value_angle_diff_j")),
                    ("load_rows", ("load_pos", "load_index", "load_kind")),
                    ("scalar_rows", ("scalar_cols", "scalar_values")),
                ),
            )
        if hasattr(self, "_active_zero_current_vector_plan"):
            self._active_zero_current_vector_plan = self._shrink_active_plan_group(
                self._active_zero_current_vector_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("voltage_rows", ("voltage_pos",)),
                    ("angle_diff_rows", ("angle_diff_i", "angle_diff_j")),
                    ("voltage_diff_rows", ("voltage_diff_i", "voltage_diff_j")),
                    ("power_rows", ("power_is_p", "power_pos", "power_current_idx", "power_sign")),
                    ("current_rows", ("current_idx",)),
                    ("scalar_rows", ("scalar_cols", "scalar_values")),
                ),
            )
        if hasattr(self, "_active_generator_measurement_plan"):
            self._active_generator_measurement_plan = self._shrink_active_plan_group(
                self._active_generator_measurement_plan,
                removed_pos=int(removed_pos),
                groups=(("value_rows", ("value_kind", "value_pos", "value_index")),),
            )
        if hasattr(self, "_active_balance_measurement_plan"):
            self._active_balance_measurement_plan = self._shrink_balance_measurement_plan(
                self._active_balance_measurement_plan,
                int(removed_pos),
            )
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        branch_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
        simple_plan = getattr(self, "_active_simple_jacobian_plan", None)
        zero_plan = getattr(self, "_active_zero_current_vector_plan", None)
        generator_plan = getattr(self, "_active_generator_measurement_plan", None)
        balance_plan = getattr(self, "_active_balance_measurement_plan", None)
        if branch_plan is not None:
            self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                branch_plan,
            )
        if simple_plan is not None:
            self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                simple_plan,
            )
        if zero_plan is not None:
            self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                zero_plan,
            )
        if generator_plan is not None:
            self._generator_measurement_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                generator_plan,
            )
        if balance_plan is not None:
            self._balance_measurement_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                balance_plan,
            )
        if None not in (branch_plan, simple_plan, zero_plan, generator_plan, balance_plan):
            self.active_measurements_are_vectorized = bool(
                np.all(
                    branch_plan["handled_mask"]
                    | simple_plan["handled_mask"]
                    | zero_plan["handled_mask"]
                    | generator_plan["handled_mask"]
                    | balance_plan["handled_mask"]
                )
            )
        return self.active_measurements

    def _measurement_rows_for_types(
        self,
        measurements: Sequence[Measurement],
        device_types: Tuple[str, ...],
    ):
        """Yield candidate measurement rows, using the active type index when available."""
        if measurements is self.active_measurements:
            rows = self._active_measurement_rows_for_types(device_types)
            active_measurements = self.active_measurements
            if active_measurements is not None:
                for row in rows:
                    yield row, active_measurements[int(row)]
                return
        row_indexes = self._measurement_row_indexes_for_types(measurements, device_types)
        if row_indexes is not None:
            for row in row_indexes:
                yield int(row), measurements[int(row)]
            return
        device_type_set = set(device_types)
        for row, meas in enumerate(measurements):
            if meas.device_type in device_type_set:
                yield row, meas

    def _measurement_row_indexes_for_types(
        self,
        measurements: Sequence[Measurement],
        device_types: Tuple[str, ...],
    ) -> Optional[np.ndarray]:
        if measurements is self.active_measurements:
            return self._active_measurement_rows_for_types(device_types)
        table = getattr(measurements, "table", None)
        if table is None or len(table.device_type_code) != len(measurements):
            return None
        codes = [int(_DEVICE_TYPE_CODES.get(device_type, 0)) for device_type in device_types]
        codes = [code for code in codes if code > 0]
        if not codes:
            return np.empty(0, dtype=np.int64)
        if len(codes) == 1:
            return np.flatnonzero(np.asarray(table.device_type_code, dtype=np.int16) == codes[0]).astype(
                np.int64,
                copy=False,
            )
        code_array = np.asarray(table.device_type_code, dtype=np.int16)
        return np.flatnonzero(np.isin(code_array, np.asarray(codes, dtype=np.int16))).astype(np.int64, copy=False)

    def _active_measurement_rows_for_types(self, device_types: Tuple[str, ...]) -> np.ndarray:
        rows_by_device_type_code = getattr(self, "_active_rows_by_device_type_code", None)
        if rows_by_device_type_code is None:
            return np.empty(0, dtype=np.int64)
        chunks = [rows_by_device_type_code.get(_DEVICE_TYPE_CODES.get(device_type, 0), ()) for device_type in device_types]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        non_empty = [chunk for chunk in chunks if len(chunk)]
        if not non_empty:
            return np.empty(0, dtype=np.int64)
        if len(non_empty) == 1:
            return np.asarray(non_empty[0], dtype=np.int64)
        return np.concatenate(non_empty)

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        if isinstance(measurements, list):
            return measurements
        return list(measurements)

    @staticmethod
    def _measurement_row_count(measurements: Sequence[Measurement]) -> int:
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return len(table.idx)
        return len(measurements)

    @staticmethod
    def _measurement_sequence_arrays(
        measurements: Sequence[Measurement],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        table = getattr(measurements, "table", None)
        if table is None or len(table.idx) != len(measurements):
            return None
        return table.device_type, table.device_name, table.meas_type

    def _measurement_vectors(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
        table = getattr(measurements, "table", None)
        if table is not None and len(table.value) == len(measurements):
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        return (
            np.asarray([meas.value for meas in measurements], dtype=np.float64),
            np.asarray([meas.weight for meas in measurements], dtype=np.float64),
        )

    @staticmethod
    def _uniform_weight(weight: np.ndarray) -> Optional[float]:
        if weight.size == 0:
            return None
        first_weight = float(weight[0])
        return first_weight if bool(np.all(weight == first_weight)) else None

    def _angle_residual_mask(self, measurements: Sequence[Measurement]) -> np.ndarray:
        if measurements is self.active_measurements:
            return self.active_angle_residual_mask
        table = getattr(measurements, "table", None)
        if table is not None and len(table.angle_mask) == len(measurements):
            return np.asarray(table.angle_mask, dtype=bool)
        return angle_residual_mask(measurements)

    def _has_angle_residuals(self, measurements: Sequence[Measurement], angle_mask: np.ndarray) -> bool:
        if measurements is self.active_measurements:
            return self.active_has_angle_residuals
        return bool(np.any(angle_mask))

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
    ) -> np.ndarray:
        angle_mask = self._angle_residual_mask(measurements)
        return build_measurement_residual(
            z,
            z_est,
            angle_mask,
            has_angle_residuals=self._has_angle_residuals(measurements, angle_mask),
        )

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _disable_angle_measurements(self) -> None:
        """Keep all phase-angle measurement rows out of WLS; flat starts store them as zero."""
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.meas_type) == len(self.measurements):
            angle_mask = np.asarray(table.angle_mask, dtype=bool)
            status_code = measurement_table_status_code(table)
            for row in np.flatnonzero(angle_mask):
                mark_measurement_invalid(self.measurements[int(row)])
                table.valid[int(row)] = False
                status_code[int(row)] = MEAS_STATUS_INVALID
                if self.flat_start:
                    self.measurements[int(row)].value = 0.0
                    table.value[int(row)] = 0.0
            return
        for meas in self.measurements:
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                mark_measurement_invalid(meas)
                if self.flat_start:
                    meas.value = 0.0

    def _disable_unavailable_measurements(self) -> None:
        """Keep invalid/off-topology measurement rows out of unit conversion and WLS."""
        device_maps = {
            "ACNode": self.node_by_name,
            "ACBranch": self.branch_by_name,
            "ACTransformer": self.transformer_by_name,
            "ACBreak": self.break_by_name,
            "ACZeroBranch": self.zero_branch_by_name,
            "ACGenerator": self.generator_by_name,
            "ACLoad": self.load_by_name,
        }
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.device_type) == len(self.measurements):
            status_code = measurement_table_status_code(table)
            for row in np.flatnonzero(np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)):
                meas = self.measurements[int(row)]
                if meas.device_type in ("ACSwitch", "ACSwitchConstraint"):
                    mark_measurement_invalid(meas)
                    table.valid[int(row)] = False
                    status_code[int(row)] = MEAS_STATUS_INVALID
                    continue
                if meas.device_type in ("ACZeroBranchConstraint", "ACBreakConstraint"):
                    mark_measurement_invalid(meas)
                    table.valid[int(row)] = False
                    status_code[int(row)] = MEAS_STATUS_INVALID
                    continue
                devices = device_maps.get(meas.device_type)
                if devices is None or meas.device_name not in devices:
                    mark_measurement_invalid(meas)
                    table.valid[int(row)] = False
                    status_code[int(row)] = MEAS_STATUS_INVALID
            return
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            if meas.device_type in ("ACSwitch", "ACSwitchConstraint"):
                mark_measurement_invalid(meas)
                continue
            if meas.device_type in ("ACZeroBranchConstraint", "ACBreakConstraint"):
                mark_measurement_invalid(meas)
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                mark_measurement_invalid(meas)

    def _load_network(self, e_file: Path) -> ACPowerNetwork:
        """Read the AC case and build topology references used by measurements."""
        try:
            return _build_ac_se_network_from_ppc(e_file)
        except (KeyError, ValueError, RuntimeError):
            network = ACPowerNetwork()
            network.read_from_file(e_file)
            network_topology.prepare_ac_topology(network)
            return network

    @staticmethod
    def _run_power_flow_seed(network: ACPowerNetwork, params: StateEstimationParameters, e_file: Path) -> bool:
        """Run one AC load-flow solve so non-flat SE starts from a measured operating point."""
        seed_tol = max(float(params.power_flow_tol), 1e-6)
        ppc = ACStateEstimator._power_flow_seed_ppc_from_network(network)
        if ppc is not None:
            calc = ACPowerFlowCalc(
                ppc,
                tol=seed_tol,
                max_iter=params.power_flow_max_iter,
                min_voltage=params.power_flow_min_voltage,
            )
            calc.skip_lf_result = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()):
                        calc.prepare()
                        try:
                            rc = calc.run(result_mode="none")
                        except TypeError as exc:
                            if "result_mode" not in str(exc):
                                raise
                            rc = calc.run()
                if rc != 0 or not calc.converged:
                    return False
                ACStateEstimator._apply_power_flow_seed_calc_state_to_network(network, ppc, calc)
                return True
            except Exception:
                return False

        snapshot = ACStateEstimator._capture_power_flow_seed_snapshot(network)
        calc_model = ppc if ppc is not None else network
        calc = ACPowerFlowCalc(
            calc_model,
            tol=seed_tol,
            max_iter=params.power_flow_max_iter,
            min_voltage=params.power_flow_min_voltage,
        )
        calc.skip_lf_result = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    calc.prepare()
                    rc = calc.run()
        except Exception:
            ACStateEstimator._restore_power_flow_seed_snapshot(snapshot)
            return False
        ok = bool(rc == 0 and calc.converged)
        if not ok:
            ACStateEstimator._restore_power_flow_seed_snapshot(snapshot)
            return False
        return ok

    @staticmethod
    def _copy_ppc_for_power_flow_seed(ppc):
        copied = dict(ppc)
        for key in (
            "bus",
            "gen",
            "load",
            "shunt",
            "branch",
            "transformer",
            "zero_branch",
            "switch",
            "break",
        ):
            value = ppc.get(key)
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
        for key, value in ppc.items():
            if isinstance(value, dict):
                copied[key] = dict(value)
        return copied

    @staticmethod
    def _power_flow_seed_ppc_from_network(network):
        source = getattr(network, "_array_model", None)
        if not (isinstance(source, dict) and source.get("format") == "ac_ppc_v1"):
            source = getattr(network, "ppc", None)
        if not (isinstance(source, dict) and source.get("format") == "ac_ppc_v1"):
            return None
        ppc = ACStateEstimator._copy_ppc_for_power_flow_seed(source)
        seed_rows = getattr(network, "_se_power_flow_seed_rows", None)
        if seed_rows is None:
            ACStateEstimator._sync_ac_network_to_ppc(network, ppc)
        else:
            ACStateEstimator._apply_power_flow_seed_rows_to_ppc(ppc, seed_rows)
        return ppc

    @staticmethod
    def _row_by_idx(array: np.ndarray, idx_col: int) -> Dict[int, int]:
        if array is None or array.size == 0:
            return {}
        return {int(row[idx_col]): pos for pos, row in enumerate(array)}

    @staticmethod
    def _row_by_name(names) -> Dict[str, int]:
        if names is None:
            return {}
        return {str(name): pos for pos, name in enumerate(names)}

    @staticmethod
    def _apply_power_flow_seed_rows_to_ppc(ppc, seed_rows) -> None:
        bus = ppc.get("bus")
        if bus is None:
            return
        bus_by_name = ACStateEstimator._row_by_name(ppc.get("bus_name", ()))
        bus_by_idx = ACStateEstimator._row_by_idx(bus, BUS_COLS["idx"])
        gen = ppc.get("gen")
        gen_by_name = ACStateEstimator._row_by_name(ppc.get("gen_name", ()))
        load = ppc.get("load")
        load_by_name = ACStateEstimator._row_by_name(ppc.get("load_name", ()))

        def set_bus_voltage_by_idx(node_idx, value):
            row = bus_by_idx.get(int(node_idx))
            if row is not None:
                bus[row, BUS_COLS["voltage"]] = max(float(value), 0.0)

        for device_type, device_name, meas_type, value in seed_rows:
            value = float(value)
            if device_type == "ACNode":
                if meas_type == "V":
                    row = bus_by_name.get(str(device_name))
                    if row is not None:
                        bus[row, BUS_COLS["voltage"]] = max(value, 0.0)
                continue
            if device_type == "ACGenerator" and gen is not None:
                row = gen_by_name.get(str(device_name))
                if row is None:
                    continue
                if meas_type == "P_GEN":
                    gen[row, GEN_COLS["p_set"]] = value
                    gen[row, GEN_COLS["p"]] = value
                elif meas_type == "Q_GEN":
                    gen[row, GEN_COLS["q_set"]] = value
                    gen[row, GEN_COLS["q"]] = value
                elif meas_type == "V_GEN":
                    voltage = max(value, 0.0)
                    gen[row, GEN_COLS["v_set"]] = voltage
                    set_bus_voltage_by_idx(gen[row, GEN_COLS["node"]], voltage)
                elif meas_type == "I_GEN":
                    gen[row, GEN_COLS["current"]] = value
                continue
            if device_type == "ACLoad" and load is not None:
                row = load_by_name.get(str(device_name))
                if row is None:
                    continue
                if meas_type == "P_LOAD":
                    load[row, LOAD_COLS["pbase"]] = 1.0
                    load[row, LOAD_COLS["pv0"]] = value
                    load[row, LOAD_COLS["pv1"]] = 0.0
                    load[row, LOAD_COLS["pv2"]] = 0.0
                    load[row, LOAD_COLS["p"]] = value
                elif meas_type == "Q_LOAD":
                    load[row, LOAD_COLS["qbase"]] = 1.0
                    load[row, LOAD_COLS["qv0"]] = value
                    load[row, LOAD_COLS["qv1"]] = 0.0
                    load[row, LOAD_COLS["qv2"]] = 0.0
                    load[row, LOAD_COLS["q"]] = value
                elif meas_type == "V_LOAD":
                    set_bus_voltage_by_idx(load[row, LOAD_COLS["node"]], value)
                elif meas_type == "I_LOAD":
                    load[row, LOAD_COLS["current"]] = value

    @staticmethod
    def _sync_ac_network_to_ppc(network, ppc) -> None:
        row_by_idx = ACStateEstimator._row_by_idx
        bus = ppc["bus"]
        bus_rows = row_by_idx(bus, BUS_COLS["idx"])
        for node in getattr(network, "nodes", []) or []:
            row = bus_rows.get(int(getattr(node, "idx", -1)))
            if row is None:
                continue
            bus[row, BUS_COLS["voltage"]] = float(getattr(node, "voltage", 1.0) or 1.0)
            bus[row, BUS_COLS["angle"]] = float(getattr(node, "angle", 0.0) or 0.0)
            bus[row, BUS_COLS["run_stat"]] = int(getattr(node, "run_stat", 1))

        def sync_devices(devices, key, cols, attrs):
            array = ppc.get(key)
            if array is None or array.size == 0:
                return
            rows = row_by_idx(array, cols["idx"])
            for dev in devices or []:
                row = rows.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    continue
                for attr, col_name in attrs:
                    if col_name in cols and hasattr(dev, attr):
                        value = getattr(dev, attr)
                        if value is not None:
                            array[row, cols[col_name]] = value

        terminal_attrs = (
            ("run_stat", "run_stat"),
            ("i_p", "i_p"),
            ("i_q", "i_q"),
            ("i_c", "i_c"),
            ("j_p", "j_p"),
            ("j_q", "j_q"),
            ("j_c", "j_c"),
        )
        sync_devices(getattr(network, "branches", []), "branch", BRANCH_COLS, terminal_attrs)
        sync_devices(getattr(network, "transformers", []), "transformer", TRANSFORMER_COLS, terminal_attrs)
        sync_devices(
            getattr(network, "generators", []),
            "gen",
            GEN_COLS,
            (
                ("p_set", "p_set"),
                ("q_set", "q_set"),
                ("v_set", "v_set"),
                ("run_stat", "run_stat"),
                ("p", "p"),
                ("q", "q"),
                ("current", "current"),
            ),
        )
        sync_devices(
            getattr(network, "loads", []),
            "load",
            LOAD_COLS,
            (
                ("pbase", "pbase"),
                ("pv0", "pv0"),
                ("pv1", "pv1"),
                ("pv2", "pv2"),
                ("qbase", "qbase"),
                ("qv0", "qv0"),
                ("qv1", "qv1"),
                ("qv2", "qv2"),
                ("run_stat", "run_stat"),
                ("p", "p"),
                ("q", "q"),
                ("current", "current"),
            ),
        )
        sync_devices(
            getattr(network, "shunt_compensators", []),
            "shunt",
            SHUNT_COLS,
            (
                ("q_set", "q_set"),
                ("g_set", "g_set"),
                ("b_set", "b_set"),
                ("v_set", "v_set"),
                ("run_stat", "run_stat"),
                ("p", "p"),
                ("q", "q"),
                ("current", "current"),
            ),
        )
        zero_attrs = (("run_stat", "run_stat"), ("p", "p"), ("q", "q"), ("current", "current"))
        sync_devices(getattr(network, "zero_branches", []), "zero_branch", ZERO_BRANCH_COLS, zero_attrs)
        switch_attrs = (
            ("status", "status"),
            ("run_stat", "run_stat"),
            ("p", "p"),
            ("q", "q"),
            ("current", "current"),
        )
        sync_devices(getattr(network, "switches", []), "switch", SWITCH_COLS, switch_attrs)
        sync_devices(getattr(network, "breakers", []), "break", BREAK_COLS, switch_attrs)

    @staticmethod
    def _overlay_power_flow_seed_result_ppc(seed_ppc, result):
        if not isinstance(result, dict):
            return seed_ppc
        for key in (
            "bus",
            "gen",
            "load",
            "shunt",
            "branch",
            "transformer",
            "zero_branch",
            "switch",
            "break",
        ):
            value = result.get(key)
            if isinstance(value, np.ndarray):
                seed_ppc[key] = value
        return seed_ppc

    @staticmethod
    def _apply_power_flow_seed_ppc_to_network(network, ppc) -> None:
        row_by_idx = ACStateEstimator._row_by_idx
        bus = ppc["bus"]
        bus_rows = row_by_idx(bus, BUS_COLS["idx"])
        for node in getattr(network, "nodes", []) or []:
            row = bus_rows.get(int(getattr(node, "idx", -1)))
            if row is None:
                continue
            node.voltage = float(bus[row, BUS_COLS["voltage"]])
            node.angle = float(bus[row, BUS_COLS["angle"]])
        for bus_obj in getattr(network, "buses", []) or []:
            members = getattr(bus_obj, "nodes", ()) or ()
            if members:
                ref = members[0]
                bus_obj.voltage = float(getattr(ref, "voltage", 1.0) or 1.0)
                bus_obj.angle = float(getattr(ref, "angle", 0.0) or 0.0)

        def apply_devices(devices, key, cols, attrs):
            array = ppc.get(key)
            if array is None or array.size == 0:
                return
            rows = row_by_idx(array, cols["idx"])
            for dev in devices or []:
                row = rows.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    continue
                for attr, col_name in attrs:
                    if col_name in cols and hasattr(dev, attr):
                        setattr(dev, attr, float(array[row, cols[col_name]]))

        terminal_attrs = (
            ("i_p", "i_p"),
            ("i_q", "i_q"),
            ("i_c", "i_c"),
            ("j_p", "j_p"),
            ("j_q", "j_q"),
            ("j_c", "j_c"),
        )
        apply_devices(getattr(network, "branches", []), "branch", BRANCH_COLS, terminal_attrs)
        apply_devices(getattr(network, "transformers", []), "transformer", TRANSFORMER_COLS, terminal_attrs)
        apply_devices(getattr(network, "generators", []), "gen", GEN_COLS, (("p", "p"), ("q", "q"), ("current", "current")))
        apply_devices(getattr(network, "loads", []), "load", LOAD_COLS, (("p", "p"), ("q", "q"), ("current", "current")))
        apply_devices(
            getattr(network, "shunt_compensators", []),
            "shunt",
            SHUNT_COLS,
            (("p", "p"), ("q", "q"), ("current", "current")),
        )
        zero_attrs = (("p", "p"), ("q", "q"), ("current", "current"))
        apply_devices(getattr(network, "zero_branches", []), "zero_branch", ZERO_BRANCH_COLS, zero_attrs)
        apply_devices(getattr(network, "switches", []), "switch", SWITCH_COLS, zero_attrs)
        apply_devices(getattr(network, "breakers", []), "break", BREAK_COLS, zero_attrs)
        network.ppc = ppc
        if hasattr(network, "_array_model"):
            network._array_model = ppc

    @staticmethod
    def _apply_power_flow_seed_calc_state_to_network(network, ppc, calc) -> None:
        if hasattr(calc, "x") and hasattr(calc, "_extract_state_vars"):
            theta, voltage, _phi_re, _phi_im = calc._extract_state_vars(calc.x, update_cache=False)
            bus = ppc.get("bus")
            if isinstance(bus, np.ndarray) and bus.size:
                bus[:, BUS_COLS["voltage"]] = np.asarray(voltage, dtype=np.float64)
                bus[:, BUS_COLS["angle"]] = np.asarray(theta, dtype=np.float64)
            ACStateEstimator._apply_power_flow_seed_ppc_to_network(network, ppc)
            return
        result_ppc = ACStateEstimator._overlay_power_flow_seed_result_ppc(ppc, getattr(calc, "result", None))
        ACStateEstimator._apply_power_flow_seed_ppc_to_network(network, result_ppc)

    @staticmethod
    def _capture_power_flow_seed_snapshot(network):
        attrs = (
            "voltage",
            "angle",
            "p",
            "q",
            "current",
            "i_p",
            "i_q",
            "i_c",
            "j_p",
            "j_q",
            "j_c",
            "p_set",
            "q_set",
            "v_set",
            "pbase",
            "pv0",
            "pv1",
            "pv2",
            "qbase",
            "qv0",
            "qv1",
            "qv2",
        )
        snapshot = []
        collections = (
            "nodes",
            "buses",
            "branches",
            "transformers",
            "zero_branches",
            "switches",
            "breakers",
            "generators",
            "loads",
            "shunt_compensators",
        )
        seen = set()
        for collection in collections:
            for obj in getattr(network, collection, []) or []:
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                state = {attr: getattr(obj, attr) for attr in attrs if hasattr(obj, attr)}
                if state:
                    snapshot.append((obj, state))
        return snapshot

    @staticmethod
    def _restore_power_flow_seed_snapshot(snapshot) -> None:
        for obj, state in snapshot:
            for attr, value in state.items():
                setattr(obj, attr, value)

    def _refresh_load_parameter_arrays(self) -> None:
        self.load_pv0_array = np.asarray(
            [getattr(load, "pbase", 1.0) * load.pv0 for load in self.load_order],
            dtype=np.float64,
        )
        self.load_pv1_array = np.asarray(
            [getattr(load, "pbase", 1.0) * load.pv1 for load in self.load_order],
            dtype=np.float64,
        )
        self.load_pv2_array = np.asarray(
            [getattr(load, "pbase", 1.0) * load.pv2 for load in self.load_order],
            dtype=np.float64,
        )
        self.load_qv0_array = np.asarray(
            [getattr(load, "qbase", 1.0) * load.qv0 for load in self.load_order],
            dtype=np.float64,
        )
        self.load_qv1_array = np.asarray(
            [getattr(load, "qbase", 1.0) * load.qv1 for load in self.load_order],
            dtype=np.float64,
        )
        self.load_qv2_array = np.asarray(
            [getattr(load, "qbase", 1.0) * load.qv2 for load in self.load_order],
            dtype=np.float64,
        )

    @staticmethod
    def _set_existing_attr(obj, attr: str, value) -> None:
        if hasattr(obj, attr):
            setattr(obj, attr, value)

    def _set_node_voltage_object(self, node, value: float) -> None:
        voltage = max(float(value), self.voltage_floor)
        if node is None:
            return
        if hasattr(node, "voltage"):
            node.voltage = voltage
        for member in getattr(node, "nodes", ()) or ():
            if hasattr(member, "voltage"):
                member.voltage = voltage

    def _set_node_voltage_by_idx(self, node_idx: int, value: float) -> None:
        targets = []
        bus = self.node_by_idx.get(int(node_idx))
        if bus is not None:
            targets.append(bus)
        raw = getattr(self.network, "node_dict", {}).get(int(node_idx))
        if raw is not None:
            targets.append(raw)
        seen = set()
        for target in targets:
            if id(target) in seen:
                continue
            seen.add(id(target))
            self._set_node_voltage_object(target, value)

    def _set_node_voltage_by_name(self, node_name: str, value: float) -> None:
        bus = self.node_by_name.get(node_name)
        if bus is not None:
            self._set_node_voltage_object(bus, value)
            return
        for node in getattr(self.network, "nodes", []):
            if getattr(node, "name", None) == node_name:
                self._set_node_voltage_object(node, value)
                bus_obj = getattr(node, "bus_obj", None)
                if bus_obj is not None:
                    self._set_node_voltage_object(bus_obj, value)
                return

    def _sync_bus_state_from_members(self) -> None:
        for bus in self.nodes:
            members = getattr(bus, "nodes", ()) or ()
            if not members:
                continue
            member = next((node for node in members if getattr(node, "voltage", None) is not None), None)
            if member is None:
                continue
            if hasattr(bus, "voltage"):
                bus.voltage = float(getattr(member, "voltage", 1.0) or 1.0)
            if hasattr(bus, "angle") and hasattr(member, "angle"):
                bus.angle = float(getattr(member, "angle", 0.0) or 0.0)

    def _refresh_file_state_from_network(self) -> None:
        self._sync_bus_state_from_members()
        self.file_theta = np.asarray(
            [float(getattr(node, "angle", 0.0) or 0.0) for node in self.nodes],
            dtype=np.float64,
        )
        self.file_voltage = np.asarray(
            [max(float(getattr(node, "voltage", 1.0) or 1.0), self.voltage_floor) for node in self.nodes],
            dtype=np.float64,
        )

    def _apply_measurement_seed_to_network(self, *, force_object: bool = False) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is not None:
            seed_rows = tuple(seed_rows)
            setattr(self.network, "_se_power_flow_seed_rows", seed_rows)
            run_seed = getattr(type(self), "_run_power_flow_seed", None)
            original_run_seed = globals().get("_ORIGINAL_AC_RUN_POWER_FLOW_SEED")
            if original_run_seed is not None and run_seed is not original_run_seed:
                force_object = True
            source = getattr(self.network, "_array_model", None)
            if not (isinstance(source, dict) and source.get("format") == "ac_ppc_v1"):
                source = getattr(self.network, "ppc", None)
            if not force_object and isinstance(source, dict) and source.get("format") == "ac_ppc_v1":
                return
            for device_type, device_name, meas_type, value in seed_rows:
                self._apply_power_flow_seed_row(device_type, device_name, meas_type, float(value))
            return
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            self._apply_power_flow_seed_row(
                meas.device_type,
                meas.device_name,
                meas.meas_type,
                float(meas.value),
            )

    def _apply_power_flow_seed_row(self, device_type: str, device_name: str, meas_type: str, value: float) -> None:
        if device_type == "ACNode":
            if meas_type == "V":
                self._set_node_voltage_by_name(device_name, value)
            return
        if device_type == "ACGenerator":
            gen = self.generator_by_name.get(device_name)
            if gen is None:
                return
            if meas_type == "P_GEN":
                self._set_existing_attr(gen, "p_set", value)
                self._set_existing_attr(gen, "p", value)
            elif meas_type == "Q_GEN":
                self._set_existing_attr(gen, "q_set", value)
                self._set_existing_attr(gen, "q", value)
            elif meas_type == "V_GEN":
                voltage = max(value, self.voltage_floor)
                self._set_existing_attr(gen, "v_set", voltage)
                self._set_node_voltage_by_idx(gen.node, voltage)
            elif meas_type == "I_GEN":
                self._set_existing_attr(gen, "i_set", value)
                self._set_existing_attr(gen, "current", value)
            return
        if device_type == "ACLoad":
            load = self.load_by_name.get(device_name)
            if load is None:
                return
            if meas_type == "P_LOAD":
                self._set_existing_attr(load, "pbase", 1.0)
                self._set_existing_attr(load, "pv0", value)
                self._set_existing_attr(load, "pv1", 0.0)
                self._set_existing_attr(load, "pv2", 0.0)
                self._set_existing_attr(load, "p", value)
            elif meas_type == "Q_LOAD":
                self._set_existing_attr(load, "qbase", 1.0)
                self._set_existing_attr(load, "qv0", value)
                self._set_existing_attr(load, "qv1", 0.0)
                self._set_existing_attr(load, "qv2", 0.0)
                self._set_existing_attr(load, "q", value)
            elif meas_type == "V_LOAD":
                self._set_node_voltage_by_idx(load.node, value)
            elif meas_type == "I_LOAD":
                self._set_existing_attr(load, "current", value)

    @staticmethod
    def _load_measurements(meas_file: Path, scale_context=None) -> List[Measurement]:
        return Measurement.read_from_file(meas_file, scale_context=scale_context)

    def _node_current_base(self, node_idx: int) -> float:
        return self.i_scale * ac_current_base_ka(self.p_base_kW, self._voltage_base(node_idx))

    def _voltage_base(self, node_idx: int) -> float:
        return float(self.network.node_dict[node_idx].vbase)

    def _voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._voltage_base(node_idx)

    def _power_file_base(self) -> float:
        return self.p_base

    def _terminal_measurement_scale(self, meas: Measurement, device) -> float:
        if meas.meas_type.startswith(("P_", "Q_")):
            return self._power_file_base()
        if meas.meas_type.endswith("_FROM"):
            node_idx = device.i_node
        elif meas.meas_type.endswith("_TO"):
            node_idx = device.j_node
        else:
            return 1.0
        if meas.meas_type.startswith("V_"):
            return self._voltage_file_base(node_idx)
        if meas.meas_type.startswith("I_"):
            return self._node_current_base(node_idx)
        return 1.0

    def _measurement_scale(self, meas: Measurement) -> float:
        if meas.device_type == "ACNode":
            if meas.meas_type == "V":
                return self._voltage_file_base(self.node_by_name[meas.device_name].idx)
            return 1.0
        if meas.device_type == "ACBranch":
            return self._terminal_measurement_scale(meas, self.branch_by_name[meas.device_name])
        if meas.device_type == "ACTransformer":
            return self._terminal_measurement_scale(meas, self.transformer_by_name[meas.device_name])
        if meas.device_type == "ACZeroBranch":
            return self._terminal_measurement_scale(meas, self.zero_branch_by_name[meas.device_name])
        if meas.device_type == "ACBreak":
            return self._terminal_measurement_scale(meas, self.break_by_name[meas.device_name])
        if meas.device_type == "ACGenerator":
            gen = self.generator_by_name[meas.device_name]
            if meas.meas_type in ("P_GEN", "Q_GEN"):
                return self._power_file_base()
            if meas.meas_type == "V_GEN":
                return self._voltage_file_base(gen.node)
            if meas.meas_type == "I_GEN":
                return self._node_current_base(gen.node)
        if meas.device_type == "ACLoad":
            load = self.load_by_name[meas.device_name]
            if meas.meas_type in ("P_LOAD", "Q_LOAD"):
                return self._power_file_base()
            if meas.meas_type == "V_LOAD":
                return self._voltage_file_base(load.node)
            if meas.meas_type == "I_LOAD":
                return self._node_current_base(load.node)
        return 1.0

    def _measurement_value_to_internal_units(self, meas: Measurement) -> float:
        if meas.meas_type in ("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"):
            return math.radians(float(meas.value))
        return float(meas.value) / self._measurement_scale(meas)

    def _convert_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal state-estimation units."""
        table = _measurement_table_from_measurements(self.measurements)
        if getattr(self.measurements, "normalized", False):
            self.measurement_table = table
            self._refresh_measurement_summary_cache()
            self._node_voltage_measurement_cache = {}
            self._real_power_measurement_seed_cache = {}
            self._has_valid_angle_measurements = bool(np.any(table.valid & table.angle_mask))
            seed_rows = []
            for pos, meas in enumerate(self.measurements):
                if meas.valid and meas.weight > 0.0:
                    if meas.device_type == "ACNode" and meas.meas_type == "V":
                        node = self.node_by_name.get(meas.device_name)
                        if node is not None and not is_pseudo_measurement(meas):
                            self._node_voltage_measurement_cache[node.idx] = float(meas.value)
                        seed_rows.append((meas.device_type, meas.device_name, meas.meas_type, float(meas.value)))
                    elif (
                        (meas.device_type == "ACGenerator" and meas.meas_type in ("P_GEN", "Q_GEN"))
                        or (meas.device_type == "ACLoad" and meas.meas_type in ("P_LOAD", "Q_LOAD"))
                    ):
                        self._real_power_measurement_seed_cache[(meas.device_type, meas.device_name, meas.meas_type)] = (
                            float(meas.weight),
                            float(meas.value),
                        )
                        seed_rows.append((meas.device_type, meas.device_name, meas.meas_type, float(meas.value)))
                    elif (
                        (meas.device_type == "ACGenerator" and meas.meas_type in ("V_GEN", "I_GEN"))
                        or (meas.device_type == "ACLoad" and meas.meas_type in ("V_LOAD", "I_LOAD"))
                    ):
                        seed_rows.append((meas.device_type, meas.device_name, meas.meas_type, float(meas.value)))
            self._power_flow_seed_rows = seed_rows
            return
        idx_array = table.idx
        device_type_array = table.device_type
        device_name_array = table.device_name
        meas_type_array = table.meas_type
        weight_array = table.weight
        valid_array = table.valid
        value_array = table.value
        status_array = measurement_table_status_code(table)
        power_scale = self.p_base
        voltage_scale_by_node = {
            node_idx: self.u_scale * float(node.vbase)
            for node_idx, node in self.network.node_dict.items()
        }
        current_scale_by_node = {
            node_idx: self.i_scale * ac_current_base_ka(self.p_base_kW, float(node.vbase))
            for node_idx, node in self.network.node_dict.items()
        }
        node_voltage_scale_by_name = {
            name: voltage_scale_by_node[node.idx]
            for name, node in self.node_by_name.items()
        }
        branch_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.branch_by_name.items()
        }
        transformer_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.transformer_by_name.items()
        }
        zero_branch_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.zero_branch_by_name.items()
        }
        break_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in getattr(self, "break_by_name", {}).items()
        }
        generator_node_scale_by_name = {
            name: (voltage_scale_by_node[gen.node], current_scale_by_node[gen.node])
            for name, gen in self.generator_by_name.items()
        }
        load_node_scale_by_name = {
            name: (voltage_scale_by_node[load.node], current_scale_by_node[load.node])
            for name, load in self.load_by_name.items()
        }
        max_idx = 0
        terminal_maps = {
            "ACBranch": branch_terminal_scale_by_name,
            "ACTransformer": transformer_terminal_scale_by_name,
            "ACZeroBranch": zero_branch_terminal_scale_by_name,
            "ACBreak": break_terminal_scale_by_name,
        }
        active_device_keys = set()
        active_measurement_keys = set()
        add_active_device_key = active_device_keys.add
        add_active_measurement_key = active_measurement_keys.add
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        power_seed_best: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        power_flow_seed_rows = []
        has_valid_angle_measurements = False
        node_pos = getattr(self, "node_pos", {})

        scale_array = np.ones(value_array.size, dtype=np.float64)
        unavailable_mask = np.ones(value_array.size, dtype=bool)

        def rows_for_type(device_type: str) -> np.ndarray:
            return np.flatnonzero(device_type_array == device_type)

        def mark_named_rows(rows: np.ndarray, mapping: Dict[str, object]) -> np.ndarray:
            if rows.size == 0:
                return rows
            found = np.asarray([str(device_name_array[int(row)]) in mapping for row in rows], dtype=bool)
            available = rows[found]
            unavailable_mask[available] = False
            return available

        node_rows = mark_named_rows(rows_for_type("ACNode"), node_voltage_scale_by_name)
        node_v_rows = node_rows[meas_type_array[node_rows] == "V"]
        if node_v_rows.size:
            scale_array[node_v_rows] = np.asarray(
                [float(node_voltage_scale_by_name[str(device_name_array[int(row)])]) for row in node_v_rows],
                dtype=np.float64,
            )

        for terminal_type, scale_by_name in terminal_maps.items():
            terminal_rows = mark_named_rows(rows_for_type(terminal_type), scale_by_name)
            for row in terminal_rows:
                mtype = str(meas_type_array[int(row)])
                terminal_scales = scale_by_name[str(device_name_array[int(row)])]
                if mtype in _TERMINAL_POWER_MEASUREMENT_TYPES:
                    scale_array[int(row)] = power_scale
                elif mtype == "V_FROM":
                    scale_array[int(row)] = terminal_scales[0]
                elif mtype == "I_FROM":
                    scale_array[int(row)] = terminal_scales[1]
                elif mtype == "V_TO":
                    scale_array[int(row)] = terminal_scales[2]
                elif mtype == "I_TO":
                    scale_array[int(row)] = terminal_scales[3]

        gen_rows = mark_named_rows(rows_for_type("ACGenerator"), generator_node_scale_by_name)
        for row in gen_rows:
            mtype = str(meas_type_array[int(row)])
            node_scales = generator_node_scale_by_name[str(device_name_array[int(row)])]
            if mtype in ("P_GEN", "Q_GEN"):
                scale_array[int(row)] = power_scale
            elif mtype == "V_GEN":
                scale_array[int(row)] = node_scales[0]
            elif mtype == "I_GEN":
                scale_array[int(row)] = node_scales[1]

        load_rows = mark_named_rows(rows_for_type("ACLoad"), load_node_scale_by_name)
        for row in load_rows:
            mtype = str(meas_type_array[int(row)])
            node_scales = load_node_scale_by_name[str(device_name_array[int(row)])]
            if mtype in ("P_LOAD", "Q_LOAD"):
                scale_array[int(row)] = power_scale
            elif mtype == "V_LOAD":
                scale_array[int(row)] = node_scales[0]
            elif mtype == "I_LOAD":
                scale_array[int(row)] = node_scales[1]

        def invalidate_measurement(pos: int, meas: Measurement) -> None:
            valid_array[pos] = False
            status_array[pos] = MEAS_STATUS_INVALID
            mark_measurement_invalid(meas)

        for pos, meas in enumerate(self.measurements):
            meas_idx = int(idx_array[pos])
            mtype = meas_type_array[pos]
            device_type = device_type_array[pos]
            device_name = device_name_array[pos]
            weight = float(weight_array[pos])
            if meas_idx > max_idx:
                max_idx = meas_idx
            if mtype in ANGLE_MEASUREMENT_TYPES:
                if self.flat_start:
                    value_array[pos] = 0.0
                    meas.value = 0.0
                invalidate_measurement(pos, meas)
                continue
            if not bool(valid_array[pos]) or weight <= 0.0:
                continue
            if bool(unavailable_mask[pos]):
                invalidate_measurement(pos, meas)
                continue
            scale = float(scale_array[pos])
            converted_value = float(value_array[pos]) / scale
            value_array[pos] = converted_value
            meas.value = converted_value
            is_real_measurement = int(status_array[pos]) != MEAS_STATUS_PSEUDO
            if is_real_measurement and mtype in _VOLTAGE_MEASUREMENT_TYPES:
                node_idx = self._voltage_measurement_node_idx(device_type, device_name, mtype)
                if node_idx is not None and node_idx in node_pos:
                    current = real_voltage_best.get(node_idx)
                    if current is None or weight > current[0]:
                        real_voltage_best[node_idx] = (weight, converted_value)
            if device_type == "ACNode" and mtype == "V":
                if is_real_measurement:
                    node_idx = self.node_by_name[device_name].idx
                    current = node_voltage_best.get(node_idx)
                    if current is None or weight > current[0]:
                        node_voltage_best[node_idx] = (weight, converted_value)
                power_flow_seed_rows.append((device_type, device_name, mtype, converted_value))
            elif (
                (device_type == "ACGenerator" and mtype in ("P_GEN", "Q_GEN"))
                or (device_type == "ACLoad" and mtype in ("P_LOAD", "Q_LOAD"))
            ):
                key = (device_type, device_name, mtype)
                current = power_seed_best.get(key)
                if current is None or weight > current[0]:
                    power_seed_best[key] = (weight, converted_value)
                power_flow_seed_rows.append((device_type, device_name, mtype, converted_value))
            elif (
                (device_type == "ACGenerator" and mtype in ("V_GEN", "I_GEN"))
                or (device_type == "ACLoad" and mtype in ("V_LOAD", "I_LOAD"))
            ):
                power_flow_seed_rows.append((device_type, device_name, mtype, converted_value))
            if device_type in _PSEUDO_DEVICE_SUMMARY_TYPES:
                add_active_device_key((device_type, device_name))
            if mtype in _PSEUDO_MEASUREMENT_SUMMARY_TYPES.get(device_type, ()):
                add_active_measurement_key((device_type, device_name, mtype))
        self.measurement_table = table
        self._active_device_key_cache = active_device_keys
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }
        self._real_power_measurement_seed_cache = power_seed_best
        self._power_flow_seed_rows = power_flow_seed_rows
        self._has_valid_angle_measurements = has_valid_angle_measurements

    def _active_device_keys(self) -> set:
        """Return devices that already have at least one usable measurement."""
        if not hasattr(self, "_active_device_key_cache"):
            self._refresh_measurement_summary_cache()
        return set(self._active_device_key_cache)

    def _active_measurement_keys(self) -> set:
        """Return usable measurement keys at device and measurement-type granularity."""
        if not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        return set(self._active_measurement_key_cache)

    def _next_measurement_idx(self) -> int:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        return int(self._max_measurement_idx) + 1

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        active_device_keys = set()
        active_measurement_keys = set()
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        node_pos = getattr(self, "node_pos", {})
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            max_idx = int(table.idx.max()) if table.idx.size else 0
            active = table.valid & (table.weight > 0.0)
            device_type_array = np.asarray(table.device_type, dtype=object)
            device_name_array = np.asarray(table.device_name, dtype=object)
            meas_type_array = np.asarray(table.meas_type, dtype=object)
            active_device_keys = set(zip(device_type_array[active].tolist(), device_name_array[active].tolist()))
            active_measurement_keys = set(
                zip(
                    device_type_array[active].tolist(),
                    device_name_array[active].tolist(),
                    meas_type_array[active].tolist(),
                )
            )
            status_code = measurement_table_status_code(table)
            voltage_mask = (
                active
                & (status_code != MEAS_STATUS_PSEUDO)
                & np.isin(meas_type_array, _VOLTAGE_MEASUREMENT_TYPE_TUPLE)
            )
            for row, device_type_value, device_name_value, meas_type_value in zip(
                np.flatnonzero(voltage_mask),
                device_type_array[voltage_mask],
                device_name_array[voltage_mask],
                meas_type_array[voltage_mask],
            ):
                device_type = str(device_type_value)
                device_name = str(device_name_value)
                meas_type = str(meas_type_value)
                weight = float(table.weight[row])
                value = float(table.value[row])
                if device_type == "ACNode" and meas_type == "V" and device_name in self.node_by_name:
                    node_idx = int(self.node_by_name[device_name].idx)
                    current = node_voltage_best.get(node_idx)
                    if current is None or weight > current[0]:
                        node_voltage_best[node_idx] = (weight, value)
                node_idx = self._voltage_measurement_node_idx(device_type, device_name, meas_type)
                if node_idx is not None and node_idx in node_pos:
                    current = real_voltage_best.get(node_idx)
                    if current is None or weight > current[0]:
                        real_voltage_best[node_idx] = (weight, value)
        else:
            max_idx = 0
            for meas in self.measurements:
                if meas.idx > max_idx:
                    max_idx = int(meas.idx)
                if meas.valid and meas.weight > 0.0:
                    active_device_keys.add((meas.device_type, meas.device_name))
                    active_measurement_keys.add((meas.device_type, meas.device_name, meas.meas_type))
                    if is_pseudo_measurement(meas):
                        continue
                    if str(meas.meas_type).upper() not in _VOLTAGE_MEASUREMENT_TYPES:
                        continue
                    weight = float(meas.weight)
                    value = float(meas.value)
                    if meas.device_type == "ACNode" and meas.meas_type == "V" and meas.device_name in self.node_by_name:
                        node_idx = int(self.node_by_name[meas.device_name].idx)
                        current = node_voltage_best.get(node_idx)
                        if current is None or weight > current[0]:
                            node_voltage_best[node_idx] = (weight, value)
                    node_idx = self._voltage_measurement_node_idx(meas.device_type, meas.device_name, meas.meas_type)
                    if node_idx is not None and node_idx in node_pos:
                        current = real_voltage_best.get(node_idx)
                        if current is None or weight > current[0]:
                            real_voltage_best[node_idx] = (weight, value)
        self._active_device_key_cache = active_device_keys
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }

    def _record_measurement_summary(self, meas: Measurement) -> None:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        if meas.idx > self._max_measurement_idx:
            self._max_measurement_idx = int(meas.idx)
        if meas.valid and meas.weight > 0.0:
            self._active_device_key_cache.add((meas.device_type, meas.device_name))
            self._active_measurement_key_cache.add((meas.device_type, meas.device_name, meas.meas_type))

    def _append_pseudo_measurement(
        self,
        next_idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        value: float,
        *,
        record_summary: bool = True,
    ) -> int:
        measurement = Measurement.__new__(Measurement)
        measurement.idx = next_idx
        measurement.name = name
        measurement.device_type = device_type
        measurement.device_name = device_name
        measurement.meas_type = meas_type
        measurement.weight = self.pseudo_measurement_weight
        measurement.valid = True
        measurement.value = float(value)
        mark_measurement_pseudo(measurement)
        self.measurements.append(measurement)
        if record_summary:
            self._record_measurement_summary(measurement)
        elif next_idx > getattr(self, "_max_measurement_idx", 0):
            self._max_measurement_idx = int(next_idx)
        return next_idx + 1

    @staticmethod
    def _generator_pseudo_power(gen) -> Tuple[float, float]:
        # Prefer solved output when available; otherwise use configured setpoints.
        p = getattr(gen, "p", None)
        q = getattr(gen, "q", None)
        if p is not None and q is not None and (abs(float(p)) > 1e-12 or abs(float(q)) > 1e-12):
            return float(p), float(q)
        return float(getattr(gen, "p_set", 0.0) or 0.0), float(getattr(gen, "q_set", 0.0) or 0.0)

    @staticmethod
    def _load_pseudo_power(load) -> Tuple[float, float]:
        # Loads may be voltage dependent, so evaluate the ZIP model at the current seed voltage.
        node = getattr(load, "node_obj", None)
        voltage = float(getattr(node, "voltage", 1.0) or 1.0)
        p = getattr(load, "pbase", 1.0) * (load.pv0 + load.pv1 * voltage + load.pv2 * voltage * voltage)
        q = getattr(load, "qbase", 1.0) * (load.qv0 + load.qv1 * voltage + load.qv2 * voltage * voltage)
        return float(p), float(q)

    def _active_angle_measurement_counts(self) -> Dict[int, int]:
        """Count usable local P/Q or direct angle measurements per AC node."""
        counts: Dict[int, int] = {}

        def add(pos: int, amount: int = 1) -> None:
            counts[pos] = counts.get(pos, 0) + amount

        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                if mtype in ("ANGLE", "THETA") and meas.device_name in self.node_by_name:
                    add(self.node_pos[self.node_by_name[meas.device_name].idx], 2)
                continue
            if not mtype.startswith(("P", "Q")):
                continue
            if meas.device_type == "ACBranch":
                dev = self.branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACTransformer":
                dev = self.transformer_by_name.get(meas.device_name)
            elif meas.device_type == "ACZeroBranch":
                dev = self.zero_branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACBreak":
                dev = self.break_by_name.get(meas.device_name)
            else:
                dev = None
            if dev is not None:
                if dev.i_node in self.node_pos:
                    add(self.node_pos[dev.i_node])
                if dev.j_node in self.node_pos:
                    add(self.node_pos[dev.j_node])
                continue
            if meas.device_type == "ACGenerator":
                dev = self.generator_by_name.get(meas.device_name)
                if dev is not None and dev.node in self.node_pos:
                    add(self.node_pos[dev.node])
        return counts

    def _voltage_measurement_node_idx(
        self,
        device_type: str,
        device_name: str,
        meas_type: str,
    ) -> Optional[int]:
        """Return the AC node associated with a voltage measurement row."""
        if device_type == "ACNode":
            if meas_type == "V" and device_name in self.node_by_name:
                return int(self.node_by_name[device_name].idx)
            return None
        if device_type == "ACGenerator":
            gen = self.generator_by_name.get(device_name)
            if meas_type == "V_GEN" and gen is not None:
                return int(gen.node)
            return None
        if device_type == "ACLoad":
            load = self.load_by_name.get(device_name)
            if meas_type == "V_LOAD" and load is not None:
                return int(load.node)
            return None
        if device_type in ("ACBranch", "ACTransformer", "ACZeroBranch", "ACBreak"):
            if device_type == "ACBranch":
                dev = self.branch_by_name.get(device_name)
            elif device_type == "ACTransformer":
                dev = self.transformer_by_name.get(device_name)
            elif device_type == "ACZeroBranch":
                dev = self.zero_branch_by_name.get(device_name)
            else:
                dev = self.break_by_name.get(device_name)
            if dev is None:
                return None
            if meas_type == "V_FROM":
                return int(dev.i_node)
            if meas_type == "V_TO":
                return int(dev.j_node)
        return None

    def _real_voltage_observation_nodes(self) -> Dict[int, float]:
        """Return nodes covered by real usable voltage measurements on any AC device."""
        cache = getattr(self, "_real_voltage_observation_node_cache", None)
        if cache is not None:
            return cache
        best: Dict[int, Tuple[float, float]] = {}
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            meas_type_array = np.asarray(table.meas_type, dtype=object)
            status_code = measurement_table_status_code(table)
            voltage_mask = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
                & np.isin(meas_type_array, _VOLTAGE_MEASUREMENT_TYPE_TUPLE)
            )
            device_type_array = np.asarray(table.device_type, dtype=object)
            device_name_array = np.asarray(table.device_name, dtype=object)
            for row, device_type_value, device_name_value, meas_type_value in zip(
                np.flatnonzero(voltage_mask),
                device_type_array[voltage_mask],
                device_name_array[voltage_mask],
                meas_type_array[voltage_mask],
            ):
                node_idx = self._voltage_measurement_node_idx(
                    str(device_type_value),
                    str(device_name_value),
                    str(meas_type_value),
                )
                if node_idx is None or node_idx not in self.node_pos:
                    continue
                weight = float(table.weight[row])
                current = best.get(int(node_idx))
                if current is None or weight > current[0]:
                    best[int(node_idx)] = (weight, float(table.value[row]))
        else:
            for meas in self.measurements:
                if (
                    not meas.valid
                    or meas.weight <= 0.0
                    or is_pseudo_measurement(meas)
                    or str(meas.meas_type).upper() not in _VOLTAGE_MEASUREMENT_TYPES
                ):
                    continue
                node_idx = self._voltage_measurement_node_idx(meas.device_type, meas.device_name, meas.meas_type)
                if node_idx is None or node_idx not in self.node_pos:
                    continue
                current = best.get(node_idx)
                if current is None or float(meas.weight) > current[0]:
                    best[node_idx] = (float(meas.weight), float(meas.value))
        cache = {node_idx: value for node_idx, (_weight, value) in best.items()}
        self._real_voltage_observation_node_cache = cache
        return cache

    def _real_voltage_observation_value_for_node(self, node_idx: Optional[int]) -> Optional[float]:
        """Return a real voltage value on the node or its compressed zero-tie component."""
        if node_idx is None:
            return None
        observed = self._real_voltage_observation_nodes()
        if node_idx in observed:
            return float(observed[node_idx])
        pos = self.node_pos.get(node_idx)
        component_by_pos = getattr(self, "zero_tie_component_by_pos", None)
        components = getattr(self, "zero_tie_components", None)
        if pos is None or component_by_pos is None or not components:
            return None
        component = components[int(component_by_pos[int(pos)])]
        for member_pos in component:
            member_idx = self.nodes[int(member_pos)].idx
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_pseudo_is_covered(self, device_type: str, device_name: str, meas_type: str) -> bool:
        """Check whether a voltage pseudo row is redundant because the node already has real V data."""
        node_idx = self._voltage_measurement_node_idx(device_type, device_name, meas_type)
        return self._real_voltage_observation_value_for_node(node_idx) is not None

    def _topology_voltage_pseudo_seed(self, dev) -> float:
        """Pick a voltage seed for zero-impedance topology pseudo measurements."""
        for node_idx in (getattr(dev, "i_node", None), getattr(dev, "j_node", None)):
            measured = self._real_voltage_observation_value_for_node(node_idx)
            if measured is not None:
                return max(float(measured), self.voltage_floor)

        return max(float(getattr(getattr(dev, "i_node_obj", None), "voltage", 1.0) or 1.0), self.voltage_floor)

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak P/Q/V priors for unmeasured AC topology-device states."""
        measured_keys = self._active_measurement_key_cache
        added_keys = set()
        topology_weight = float(self.pseudo_measurement_weight) * 1e-4

        for device_type, devices in (
            ("ACZeroBranch", self.zero_branches),
            ("ACBreak", self.breakers),
        ):
            for dev in devices:
                has_terminal_current = any(
                    (device_type, dev.name, meas_type) in measured_keys
                    for meas_type in ("I_FROM", "I_TO")
                )
                values = []
                if not has_terminal_current and not any(
                    (device_type, dev.name, meas_type) in measured_keys
                    for meas_type in ("P_FROM", "P_TO")
                ):
                    values.append(("P_FROM", float(getattr(dev, "p", 0.0) or 0.0)))
                if not has_terminal_current and not any(
                    (device_type, dev.name, meas_type) in measured_keys
                    for meas_type in ("Q_FROM", "Q_TO")
                ):
                    values.append(("Q_FROM", float(getattr(dev, "q", 0.0) or 0.0)))
                if not self._voltage_pseudo_is_covered(device_type, dev.name, "V_FROM"):
                    values.append(("V_FROM", self._topology_voltage_pseudo_seed(dev)))
                for meas_type, value in values:
                    key = (device_type, dev.name, meas_type)
                    if key in measured_keys or key in added_keys:
                        continue
                    next_idx = self._append_pseudo_measurement(
                        next_idx,
                        f"pseudo_{meas_type.lower()}_{dev.name}",
                        device_type,
                        dev.name,
                        meas_type,
                        value,
                        record_summary=False,
                    )
                    self.measurements[-1].weight = topology_weight
                    added_keys.add(key)
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        if not hasattr(self, "_active_device_key_cache") or not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        measured_devices = self._active_device_key_cache
        measured_keys = self._active_measurement_key_cache
        added_keys = set()
        next_idx = self._next_measurement_idx()
        next_idx, topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        added_keys.update(topology_added_keys)

        for gen in sorted(self.generator_by_name.values(), key=lambda item: item.idx):
            p, q = self._generator_pseudo_power(gen)
            voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
            key = ("ACGenerator", gen.name, "P_GEN")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_p_{gen.name}",
                    "ACGenerator",
                    gen.name,
                    "P_GEN",
                    p,
                    record_summary=False,
                )
                added_keys.add(key)
            key = ("ACGenerator", gen.name, "Q_GEN")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_q_{gen.name}",
                    "ACGenerator",
                    gen.name,
                    "Q_GEN",
                    q,
                    record_summary=False,
                )
                added_keys.add(key)
            if ("ACGenerator", gen.name) not in measured_devices:
                key = ("ACGenerator", gen.name, "V_GEN")
                if (
                    key not in measured_keys
                    and key not in added_keys
                    and not self._voltage_pseudo_is_covered("ACGenerator", gen.name, "V_GEN")
                ):
                    next_idx = self._append_pseudo_measurement(
                        next_idx,
                        f"pseudo_v_{gen.name}",
                        "ACGenerator",
                        gen.name,
                        "V_GEN",
                        voltage,
                        record_summary=False,
                    )
                    added_keys.add(key)

        for load in getattr(self, "load_order", None) or sorted(self.load_by_name.values(), key=lambda item: item.idx):
            unmetered_load = ("ACLoad", load.name) not in measured_devices
            p, q = self._load_pseudo_power(load)
            voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
            key = ("ACLoad", load.name, "P_LOAD")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_p_{load.name}",
                    "ACLoad",
                    load.name,
                    "P_LOAD",
                    p,
                    record_summary=False,
                )
                added_keys.add(key)
            key = ("ACLoad", load.name, "Q_LOAD")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_q_{load.name}",
                    "ACLoad",
                    load.name,
                    "Q_LOAD",
                    q,
                    record_summary=False,
                )
                added_keys.add(key)
            if unmetered_load:
                key = ("ACLoad", load.name, "V_LOAD")
                if (
                    key not in measured_keys
                    and key not in added_keys
                    and not self._voltage_pseudo_is_covered("ACLoad", load.name, "V_LOAD")
                ):
                    next_idx = self._append_pseudo_measurement(
                        next_idx,
                        f"pseudo_v_{load.name}",
                        "ACLoad",
                        load.name,
                        "V_LOAD",
                        voltage,
                        record_summary=False,
                    )
                    added_keys.add(key)
        if added_keys:
            measured_keys.update(added_keys)
            measured_devices.update((device_type, device_name) for device_type, device_name, _meas_type in added_keys)

    def _seed_power_state_arrays_from_measurements(self) -> None:
        """Use the best available P/Q rows as initial values for explicit power states."""
        best = getattr(self, "_real_power_measurement_seed_cache", None)
        if best is None:
            best: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
            for meas in self.measurements:
                if not meas.valid or meas.weight <= 0.0:
                    continue
                if meas.device_type == "ACGenerator" and meas.meas_type in ("P_GEN", "Q_GEN"):
                    if meas.device_name not in self.generator_state_index_by_name:
                        continue
                elif meas.device_type == "ACLoad" and meas.meas_type in ("P_LOAD", "Q_LOAD"):
                    if meas.device_name not in self.load_state_index_by_name:
                        continue
                else:
                    continue
                key = (meas.device_type, meas.device_name, meas.meas_type)
                current = best.get(key)
                if current is None or meas.weight > current[0]:
                    best[key] = (float(meas.weight), float(meas.value))

        for (_device_type, name, meas_type), (_weight, value) in best.items():
            if meas_type == "P_GEN":
                self.initial_gen_p_array[self.generator_state_index_by_name[name]] = value
            elif meas_type == "Q_GEN":
                self.initial_gen_q_array[self.generator_state_index_by_name[name]] = value
            elif meas_type == "P_LOAD":
                self.initial_load_p_array[self.load_state_index_by_name[name]] = value
            elif meas_type == "Q_LOAD":
                self.initial_load_q_array[self.load_state_index_by_name[name]] = value

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch AC rank deficiencies and optional post-observability redundancy."""
        total_added = 0
        max_count = max(0, int(self.targeted_pseudo_measurement_max))
        step = max(1, int(getattr(self, "targeted_pseudo_measurement_step", 10)))
        redundancy_target = targeted_redundancy_count(
            getattr(self, "n_state", 0),
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        )
        observability = None
        while total_added < max_count:
            observability = self.observability_analysis()
            batch_limit = min(max_count - total_added, step)
            if observability.observable:
                redundancy = max(0, int(observability.measurement_count) - int(observability.state_count))
                missing = redundancy_target - redundancy
                if missing <= 0:
                    break
                added = self._add_weak_direction_observability_pseudo_measurements(
                    observability,
                    min(batch_limit, missing),
                )
                if added == 0:
                    break
                total_added += added
                observability = None
                continue
            next_idx = self._next_measurement_idx()
            existing_keys = self._active_measurement_keys()
            existing_names = {meas.name for meas in self.measurements}
            added = 0
            refreshed = False
            measurement_count_before = len(self.measurements)
            remaining = batch_limit
            for state_idx, _score in observability.weak_states:
                if added >= remaining:
                    break
                next_idx, added_count = self._append_targeted_observability_pseudo(
                    next_idx,
                    state_idx,
                    existing_keys,
                    existing_names,
                    remaining - added,
                )
                added += added_count
            if added == 0:
                added = self._add_weak_direction_observability_pseudo_measurements(observability, remaining)
                refreshed = added > 0
            if added == 0:
                added = self._add_structural_rank_restoring_pseudo_measurements(remaining)
                refreshed = added > 0
            if added == 0:
                break
            total_added += added
            if not refreshed:
                refreshed = self._incremental_update_active_measurement_indexes(
                    self.measurements[measurement_count_before:]
                )
            if not refreshed:
                self._refresh_active_measurement_indexes()
            observability = None
        if observability is None and total_added < max_count:
            observability = self.observability_analysis()
        if observability is not None:
            self._initial_observability_cache = observability
        return total_added

    def _add_weak_direction_observability_pseudo_measurements(
        self,
        observability: ObservabilityResult,
        max_add: int,
        refresh: bool = True,
    ) -> int:
        """Add the candidate pseudo rows that best observe the current weak direction."""
        if max_add <= 0:
            return 0
        candidates = self._observability_pseudo_candidate_measurements()
        if not candidates:
            return 0
        selected = self._select_weak_direction_pseudo_candidates(observability, candidates, max_add)
        if not selected:
            return 0
        next_idx = self._next_measurement_idx()
        measurement_count_before = len(self.measurements)
        for candidate in selected:
            candidate.idx = next_idx
            next_idx += 1
            self.measurements.append(candidate)
        if refresh:
            refreshed = self._incremental_update_active_measurement_indexes(
                self.measurements[measurement_count_before:]
            )
            if not refreshed:
                self._refresh_active_measurement_indexes()
        return len(selected)

    def _add_redundant_observability_pseudo_measurements(self, max_add: int, refresh: bool = True) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add, refresh)

    def _observability_pseudo_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        existing_keys = self._active_measurement_keys()
        existing_names = {meas.name for meas in self.measurements}
        candidates: List[Measurement] = []

        def add(device_type: str, device_name: str, meas_type: str, value: float) -> None:
            key = (device_type, device_name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if self._voltage_pseudo_is_covered(device_type, device_name, meas_type):
                return
            if key in existing_keys or pseudo_name in existing_names:
                return
            candidates.append(
                Measurement(
                    0,
                    pseudo_name,
                    device_type,
                    device_name,
                    meas_type,
                    self.pseudo_measurement_weight,
                    True,
                    float(value),
                    MEAS_STATUS_PSEUDO,
                )
            )
            existing_keys.add(key)
            existing_names.add(pseudo_name)

        for node in sorted(self.node_by_name.values(), key=lambda item: item.idx):
            add("ACNode", node.name, "V", float(getattr(node, "voltage", 1.0) or 1.0))

        for load in sorted(self.load_by_name.values(), key=lambda item: item.idx):
            p, q = self._load_pseudo_power(load)
            voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
            add("ACLoad", load.name, "P_LOAD", p)
            add("ACLoad", load.name, "Q_LOAD", q)
            add("ACLoad", load.name, "V_LOAD", voltage)

        for gen in getattr(self, "generator_order", None) or sorted(self.generator_by_name.values(), key=lambda item: item.idx):
            p, q = self._generator_pseudo_power(gen)
            voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
            add("ACGenerator", gen.name, "P_GEN", p)
            add("ACGenerator", gen.name, "Q_GEN", q)
            add("ACGenerator", gen.name, "V_GEN", voltage)

        for device_type, devices in (
            ("ACBranch", self.branch_by_name),
            ("ACTransformer", self.transformer_by_name),
        ):
            for dev in sorted(devices.values(), key=lambda item: item.idx):
                add(device_type, dev.name, "P_FROM", float(getattr(dev, "i_p", 0.0) or 0.0))
                add(device_type, dev.name, "Q_FROM", float(getattr(dev, "i_q", 0.0) or 0.0))
                add(device_type, dev.name, "P_TO", float(getattr(dev, "j_p", 0.0) or 0.0))
                add(device_type, dev.name, "Q_TO", float(getattr(dev, "j_q", 0.0) or 0.0))

        return candidates

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: Sequence[Measurement],
        max_add: int,
    ) -> List[Measurement]:
        if max_add <= 0 or not candidates:
            return []
        x = self.initial_state()
        cache = self._observability_matrix_cache_for(observability, self.active_measurements, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, self.active_measurements)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return list(candidates[:max_add])
        candidate_h = self.jacobian_sparse(x, candidates)
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != len(candidates) or not np.any(scores > 0.0):
            return list(candidates[:max_add])
        order = np.argsort(-scores, kind="stable")
        selected = [candidates[int(pos)] for pos in order[:max_add] if scores[int(pos)] > 0.0]
        return selected or list(candidates[:max_add])

    def _rank_restoring_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidates from invalid real device rows, excluding node V and angles."""
        device_maps = {
            "ACBranch": self.branch_by_name,
            "ACTransformer": self.transformer_by_name,
            "ACBreak": self.break_by_name,
            "ACZeroBranch": self.zero_branch_by_name,
            "ACGenerator": self.generator_by_name,
            "ACLoad": self.load_by_name,
        }
        existing_keys = self._active_measurement_keys()
        existing_names = {meas.name for meas in self.measurements}
        seen_keys = set()
        candidates: List[Measurement] = []
        next_idx = self._next_measurement_idx()
        for meas in self.measurements:
            if meas.valid or meas.weight <= 0.0:
                continue
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                continue
            if meas.device_type == "ACNode" and meas.meas_type == "V":
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                continue
            key = (meas.device_type, meas.device_name, meas.meas_type)
            if key in existing_keys or key in seen_keys:
                continue
            pseudo_name = f"pseudo_rank_{meas.name}"
            if pseudo_name in existing_names:
                continue
            seen_keys.add(key)
            candidates.append(
                Measurement(
                    idx=next_idx + len(candidates),
                    name=pseudo_name,
                    device_type=meas.device_type,
                    device_name=meas.device_name,
                    meas_type=meas.meas_type,
                    weight=self.pseudo_measurement_weight,
                    valid=True,
                    value=self._measurement_value_to_internal_units(meas),
                    status=MEAS_STATUS_PSEUDO,
                )
            )
        return candidates

    def _rank_restoring_candidate_indices(self, candidates: Sequence[Measurement], max_add: int) -> List[int]:
        """Select candidate rows that participate in a higher structural-rank matching."""
        if max_add <= 0 or not candidates:
            return []
        base_measurements = self.active_measurements
        x = self.initial_state()
        base_h = self.jacobian_sparse(x, base_measurements)
        base_rank = sparse_structural_rank(base_h)
        if base_rank is None or base_rank >= self.n_state:
            return []

        combined_measurements = list(base_measurements) + list(candidates)
        combined_h = self.jacobian_sparse(x, combined_measurements)
        combined_rank = sparse_structural_rank(combined_h)
        if combined_rank is None or combined_rank <= base_rank:
            return []

        matching = sp_maximum_bipartite_matching(combined_h, perm_type="row")
        base_rows = len(base_measurements)
        selected = sorted({int(row) - base_rows for row in matching if int(row) >= base_rows})
        selected = selected[:max_add]
        remaining = max_add - len(selected)
        if remaining <= 0:
            return selected

        selected_set = set(selected)
        selected_measurements = [candidates[idx] for idx in selected]
        remaining_candidates = [candidate for idx, candidate in enumerate(candidates) if idx not in selected_set]
        if not remaining_candidates:
            return selected

        current_measurements = list(base_measurements) + selected_measurements
        current_h = self.jacobian_sparse(x, current_measurements)
        candidate_h = self.jacobian_sparse(x, remaining_candidates)
        anchor_local_indices = self._angle_anchor_candidate_indices(current_h, candidate_h, remaining)
        if not anchor_local_indices:
            return selected

        original_by_identity = {id(candidate): idx for idx, candidate in enumerate(candidates)}
        selected.extend(original_by_identity[id(remaining_candidates[idx])] for idx in anchor_local_indices)
        return sorted(set(selected))

    def _angle_anchor_candidate_indices(self, current_h, candidate_h, max_add: int) -> List[int]:
        """Select non-angle rows that connect unanchored angle components to an anchored component."""
        if max_add <= 0 or self.n_angle <= 0 or candidate_h.shape[0] == 0:
            return []
        tol = 1e-12
        current_angle = current_h[:, : self.n_angle].tocsr() if issparse(current_h) else csr_matrix(current_h[:, : self.n_angle])
        n_angle = self.n_angle
        parent = np.arange(n_angle, dtype=np.int32)
        anchored = np.zeros(n_angle, dtype=bool)

        def find(pos: int) -> int:
            while int(parent[pos]) != pos:
                parent[pos] = parent[int(parent[pos])]
                pos = int(parent[pos])
            return int(pos)

        def union(left: int, right: int) -> int:
            root_l = find(left)
            root_r = find(right)
            if root_l != root_r:
                parent[root_r] = root_l
                anchored[root_l] = anchored[root_l] or anchored[root_r]
            return root_l

        for row in range(current_angle.shape[0]):
            start = int(current_angle.indptr[row])
            end = int(current_angle.indptr[row + 1])
            if start == end:
                continue
            cols = current_angle.indices[start:end]
            vals = current_angle.data[start:end]
            keep = np.abs(vals) > tol
            if not np.any(keep):
                continue
            cols = cols[keep]
            vals = vals[keep]
            root = int(cols[0])
            for col in cols[1:]:
                root = union(root, int(col))
            root = find(root)
            if abs(float(np.sum(vals))) > tol:
                anchored[root] = True

        roots = {find(col) for col in range(n_angle)}
        anchored_roots = {find(col) for col in range(n_angle) if anchored[find(col)]}
        if roots and roots.issubset(anchored_roots):
            return []

        candidate_angle = candidate_h[:, : self.n_angle].tocsr() if issparse(candidate_h) else csr_matrix(candidate_h[:, : self.n_angle])
        row_roots: List[Tuple[int, Tuple[int, ...], bool]] = []
        incident: Dict[int, List[int]] = {}
        anchor_rows: List[int] = []
        for row in range(candidate_angle.shape[0]):
            start = int(candidate_angle.indptr[row])
            end = int(candidate_angle.indptr[row + 1])
            if start == end:
                continue
            cols = candidate_angle.indices[start:end]
            vals = candidate_angle.data[start:end]
            keep = np.abs(vals) > tol
            if not np.any(keep):
                continue
            vals = vals[keep]
            row_component_roots = tuple(sorted({find(int(col)) for col in cols[keep]}))
            if not row_component_roots:
                continue
            row_id = len(row_roots)
            is_anchor_row = abs(float(np.sum(vals))) > tol
            row_roots.append((row, row_component_roots, is_anchor_row))
            if is_anchor_row:
                anchor_rows.append(row_id)
            for root in row_component_roots:
                incident.setdefault(root, []).append(row_id)

        selected: List[int] = []
        used_rows = set()
        visited = set(anchored_roots)
        queue = list(anchored_roots)

        def activate(row_id: int) -> None:
            if len(selected) >= max_add or row_id in used_rows:
                return
            candidate_row, component_roots, _is_anchor_row = row_roots[row_id]
            new_roots = [root for root in component_roots if root not in visited]
            if not new_roots:
                return
            used_rows.add(row_id)
            selected.append(candidate_row)
            for root in new_roots:
                visited.add(root)
                queue.append(root)

        for row_id in anchor_rows:
            activate(row_id)
            if len(selected) >= max_add:
                break

        cursor = 0
        while cursor < len(queue) and len(selected) < max_add:
            root = queue[cursor]
            cursor += 1
            for row_id in incident.get(root, ()):
                activate(row_id)
                if len(selected) >= max_add:
                    break
        return selected

    def _add_structural_rank_restoring_pseudo_measurements(self, max_add: int) -> int:
        """Add only invalid real-measurement candidates that improve structural observability."""
        candidates = self._rank_restoring_candidate_measurements()
        selected_indices = self._rank_restoring_candidate_indices(candidates, max_add)
        if not selected_indices:
            return 0
        next_idx = self._next_measurement_idx()
        measurement_count_before = len(self.measurements)
        for local_idx in selected_indices:
            candidate = candidates[int(local_idx)]
            candidate.idx = next_idx
            next_idx += 1
            self.measurements.append(candidate)
        refreshed = self._incremental_update_active_measurement_indexes(
            self.measurements[measurement_count_before:]
        )
        if not refreshed:
            self._refresh_active_measurement_indexes()
        return len(selected_indices)

    def _unanchored_angle_state_indices(self) -> List[int]:
        """Return one AC angle state per structurally unanchored angle component."""
        H = self.jacobian_sparse(self.initial_state())
        return unanchored_angle_state_indices(H, self.angle_col[self.angle_col >= 0])

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        """Translate a weak compact AC state into the smallest useful pseudo measurement."""
        meta = state_meta_at(self.state_meta, state_idx)
        if meta is None:
            return next_idx, 0
        name = meta.device_name
        added_total = 0

        def add(device_type: str, device_name: str, meas_type: str, value: float) -> Tuple[int, int]:
            nonlocal added_total
            if added_total >= max_add:
                return next_idx, 0
            key = (device_type, device_name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if self._voltage_pseudo_is_covered(device_type, device_name, meas_type):
                return next_idx, 0
            if key in existing_keys or pseudo_name in existing_names:
                return next_idx, 0
            new_idx = self._append_pseudo_measurement(
                next_idx,
                pseudo_name,
                device_type,
                device_name,
                meas_type,
                value,
            )
            existing_keys.add(key)
            existing_names.add(pseudo_name)
            added_total += 1
            return new_idx, 1

        if meta.kind == "angle" and name in self.node_by_name:
            return next_idx, 0
        if meta.kind == "voltage" and meta.device_type == "ACNode" and name in self.node_by_name:
            node = self.node_by_name[name]
            return add("ACNode", name, "V", float(getattr(node, "voltage", 1.0) or 1.0))
        if meta.kind == "zero_current" and meta.device_type == "ACZeroBranch" and name in self.zero_branch_by_name:
            dev = self.zero_branch_by_name[name]
            next_idx, added_p = add("ACZeroBranch", name, "P_FROM", float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q = add("ACZeroBranch", name, "Q_FROM", float(getattr(dev, "q", 0.0) or 0.0))
            next_idx, added_p_to = add("ACZeroBranch", name, "P_TO", -float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q_to = add("ACZeroBranch", name, "Q_TO", -float(getattr(dev, "q", 0.0) or 0.0))
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta.kind == "break_current" and meta.device_type == "ACBreak" and name in self.break_by_name:
            dev = self.break_by_name[name]
            next_idx, added_p = add("ACBreak", name, "P_FROM", float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q = add("ACBreak", name, "Q_FROM", float(getattr(dev, "q", 0.0) or 0.0))
            next_idx, added_p_to = add("ACBreak", name, "P_TO", -float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q_to = add("ACBreak", name, "Q_TO", -float(getattr(dev, "q", 0.0) or 0.0))
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta.kind in ("generator_p", "generator_q") and name in self.generator_by_name:
            p, q = self._generator_pseudo_power(self.generator_by_name[name])
            meas_type = "P_GEN" if meta.kind == "generator_p" else "Q_GEN"
            return add("ACGenerator", name, meas_type, p if meas_type == "P_GEN" else q)
        if meta.kind in ("load_p", "load_q") and name in self.load_by_name:
            p, q = self._load_pseudo_power(self.load_by_name[name])
            meas_type = "P_LOAD" if meta.kind == "load_p" else "Q_LOAD"
            return add("ACLoad", name, meas_type, p if meas_type == "P_LOAD" else q)
        return next_idx, 0

    def _add_zero_branch_constraint_measurements(self) -> None:
        """Inject ideal zero-impedance voltage equality constraints."""
        existing = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in self.measurements
            if meas.valid
            and meas.weight > 0.0
            and meas.device_type in ("ACZeroBranchConstraint", "ACBreakConstraint")
        }
        next_idx = self._next_measurement_idx()
        weight = 10.0
        ideal_devices = [
            ("ACZeroBranchConstraint", zbr)
            for zbr in self.zero_branches
        ]
        ideal_devices.extend(
            ("ACBreakConstraint", brk)
            for brk in self.breakers
        )
        for device_type, dev in ideal_devices:
            for meas_type in ("V_DIFF",):
                if (device_type, dev.name, meas_type) in existing:
                    continue
                self.measurements.append(
                    Measurement(
                        idx=next_idx,
                        name=f"constraint_{meas_type.lower()}_{dev.name}",
                        device_type=device_type,
                        device_name=dev.name,
                        meas_type=meas_type,
                        weight=weight,
                        valid=True,
                        value=0.0,
                    )
                )
                next_idx += 1

    def _add_power_balance_constraint_measurements(self) -> None:
        """Add nodal AC power-balance equations that tie P/Q states to the grid."""
        next_idx = self._next_measurement_idx()
        weight = 10.0
        new_measurement = Measurement.__new__
        append = self.measurements.append
        for node in self.nodes:
            for meas_type in ("P_BALANCE", "Q_BALANCE"):
                measurement = new_measurement(Measurement)
                measurement.idx = next_idx
                measurement.name = f"constraint_{meas_type.lower()}_{node.name}"
                measurement.device_type = "ACPowerBalance"
                measurement.device_name = node.name
                measurement.meas_type = meas_type
                measurement.weight = weight
                measurement.valid = True
                measurement.value = 0.0
                measurement.status = normalize_measurement_status(None, valid=True)
                append(measurement)
                next_idx += 1
        self._max_measurement_idx = next_idx - 1 if next_idx > 0 else self._max_measurement_idx

    def _voltage_control_shunts(self) -> List[object]:
        """Return live V-control shunts whose reactive output is estimated."""
        shunts = []
        for shunt in getattr(self.network, "shunt_compensators", []) or []:
            if not getattr(shunt, "is_alive", False):
                continue
            if int(getattr(shunt, "node", -1)) not in self.node_pos:
                continue
            if str(getattr(shunt, "control_type", "")).upper() == "V":
                shunts.append(shunt)
        return sorted(shunts, key=lambda item: item.idx)

    @staticmethod
    def _initial_voltage_control_shunt_q(shunt) -> float:
        """Use any already-computed shunt Q as a seed; otherwise start at zero."""
        value = getattr(shunt, "q", None)
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real ACNode voltage measurements keyed by node index."""
        cached = getattr(self, "_node_voltage_measurement_cache", None)
        if cached is not None:
            return dict(cached)
        best: Dict[int, Tuple[float, float]] = {}
        for meas in self.measurements:
            if (
                not meas.valid
                or meas.weight <= 0.0
                or meas.device_type != "ACNode"
                or meas.meas_type != "V"
                or is_pseudo_measurement(meas)
                or meas.device_name not in self.node_by_name
            ):
                continue
            node_idx = self.node_by_name[meas.device_name].idx
            current = best.get(node_idx)
            if current is None or meas.weight > current[0]:
                best[node_idx] = (float(meas.weight), float(meas.value))
        return {node_idx: value for node_idx, (_weight, value) in best.items()}

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live incident AC branches, transformers, switches and zero branches."""
        degrees = {node.idx: 0 for node in self.nodes}
        device_groups = (
            self.branch_by_name.values(),
            self.transformer_by_name.values(),
            self.zero_branch_by_name.values(),
            self.switch_by_name.values(),
            self.break_by_name.values(),
        )
        for devices in device_groups:
            for dev in devices:
                if dev.i_node in self.node_pos:
                    degrees[self.nodes[self.node_pos[dev.i_node]].idx] += 1
                if dev.j_node in self.node_pos:
                    degrees[self.nodes[self.node_pos[dev.j_node]].idx] += 1
        return degrees

    def _select_reference_nodes(self):
        """Choose a measured high-degree V/angle reference for every live AC island."""
        references = []
        for island in self.network.islands:
            if not island.is_alive:
                continue
            candidates = [
                node
                for node in island.buses
                if node.idx in self.node_pos and node.idx in self.node_voltage_measurements
            ]
            if candidates:
                references.append(
                    max(
                        candidates,
                        key=lambda node: (self.node_degrees.get(node.idx, 0), -int(node.idx)),
                    )
                )
            elif island.slack_nodes:
                references.append(sorted(island.slack_nodes, key=lambda item: item.idx)[0])
            elif island.buses:
                references.append(sorted(island.buses, key=lambda item: item.idx)[0])
        return references

    def _reference_angle_offsets(self) -> Dict[int, float]:
        """Map each node position to the original island reference angle."""
        ref_by_island = {getattr(node, "isl", None): node for node in self.references}
        offsets: Dict[int, float] = {}
        for island in self.network.islands:
            if not getattr(island, "is_alive", False):
                continue
            ref = ref_by_island.get(getattr(island, "idx", None))
            if ref is None:
                continue
            offset = float(getattr(ref, "angle", 0.0) or 0.0)
            for node in island.buses:
                if node.idx in self.node_pos:
                    offsets[self.node_pos[node.idx]] = offset
        return offsets

    def _angle_reference_for_node(self, node_idx: int) -> float:
        pos = self.node_pos.get(node_idx)
        if pos is None:
            return 0.0
        return float(self.reference_angle_by_pos.get(pos, 0.0))

    def _node_angle_in_reference_frame(self, node) -> float:
        return float(getattr(node, "angle", 0.0) or 0.0) - self._angle_reference_for_node(node.idx)

    def _rebase_angle_measurements(self) -> None:
        """Convert absolute node angle measurements to the estimator reference frame."""
        if not getattr(self, "_has_valid_angle_measurements", True):
            return
        for meas in self.measurements:
            if (
                meas.valid
                and meas.weight > 0.0
                and meas.device_type == "ACNode"
                and meas.meas_type in ("ANGLE", "THETA")
                and meas.device_name in self.node_by_name
            ):
                node = self.node_by_name[meas.device_name]
                meas.value -= self._angle_reference_for_node(node.idx)

    def _build_zero_tie_state_layout(self) -> None:
        """Compress AC voltage/angle states across ideal switches and zero branches."""
        n = len(self.nodes)
        parent = np.arange(n, dtype=np.int32)

        def find(pos: int) -> int:
            while int(parent[pos]) != pos:
                parent[pos] = parent[int(parent[pos])]
                pos = int(parent[pos])
            return int(pos)

        def union(left: int, right: int) -> None:
            root_l = find(left)
            root_r = find(right)
            if root_l != root_r:
                parent[root_r] = root_l

        node_pos = self.node_pos
        for devices in (self.zero_branches, self.switches, self.breakers):
            for dev in devices:
                i = node_pos.get(dev.i_node)
                j = node_pos.get(dev.j_node)
                if i is not None and j is not None:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for pos in range(n):
            groups.setdefault(find(pos), []).append(pos)
        components = list(groups.values())
        components.sort(key=lambda group: group[0])
        self.zero_tie_components = components
        zero_tie_component_by_pos = np.empty(n, dtype=np.int32)
        for component_idx, component in enumerate(components):
            for pos in component:
                zero_tie_component_by_pos[pos] = component_idx
        self.zero_tie_component_by_pos = zero_tie_component_by_pos

        self.angle_col = np.full(n, -1, dtype=np.int32)
        self.voltage_col = np.full(n, -1, dtype=np.int32)
        self.angle_state_pos: List[int] = []
        self.angle_state_nodes: List[Sequence[int]] = []
        self.voltage_state_pos: List[int] = []
        self.voltage_state_nodes: List[Sequence[int]] = []
        self.ref_angles: Dict[int, float] = {}
        self.ref_voltages: Dict[int, float] = {}
        angle_unpack_nodes = []
        angle_unpack_cols = []
        voltage_unpack_nodes = []
        voltage_unpack_cols = []

        for component in components:
            ref_positions = [pos for pos in component if pos in self.ref_idx]
            if ref_positions:
                ref_pos = min(ref_positions, key=lambda pos: self.nodes[pos].idx)
                for pos in component:
                    self.ref_angles[pos] = 0.0
            else:
                col = len(self.angle_state_pos)
                rep_pos = component[0]
                self.angle_state_pos.append(rep_pos)
                self.angle_state_nodes.append(component)
                angle_unpack_nodes.extend(component)
                angle_unpack_cols.extend([col] * len(component))
                for pos in component:
                    self.angle_col[pos] = col

            voltage_ref_positions = [pos for pos in component if pos in self.reference_voltage_by_pos]
            if voltage_ref_positions:
                ref_pos = min(voltage_ref_positions, key=lambda pos: self.nodes[pos].idx)
                ref_voltage = max(float(self.reference_voltage_by_pos[ref_pos]), self.voltage_floor)
                for pos in component:
                    self.ref_voltages[pos] = ref_voltage
            else:
                v_col = len(self.voltage_state_pos)
                rep_pos = component[0]
                self.voltage_state_pos.append(rep_pos)
                self.voltage_state_nodes.append(component)
                voltage_unpack_nodes.extend(component)
                voltage_unpack_cols.extend([v_col] * len(component))
                for pos in component:
                    self.voltage_col[pos] = v_col

        self.angle_state_pos = np.asarray(self.angle_state_pos, dtype=np.int32)
        self.voltage_state_pos = np.asarray(self.voltage_state_pos, dtype=np.int32)
        self.n_angle = int(self.angle_state_pos.size)
        self.n_voltage = int(self.voltage_state_pos.size)
        self._angle_ref_nodes = np.fromiter(self.ref_angles.keys(), dtype=np.int32, count=len(self.ref_angles))
        self._angle_ref_values = np.fromiter(self.ref_angles.values(), dtype=np.float64, count=len(self.ref_angles))
        self._voltage_ref_nodes = np.fromiter(self.ref_voltages.keys(), dtype=np.int32, count=len(self.ref_voltages))
        self._voltage_ref_values = np.fromiter(
            self.ref_voltages.values(),
            dtype=np.float64,
            count=len(self.ref_voltages),
        )
        self._angle_unpack_nodes = np.asarray(angle_unpack_nodes, dtype=np.int32)
        self._angle_unpack_cols = np.asarray(angle_unpack_cols, dtype=np.int32)
        self._voltage_unpack_nodes = np.asarray(voltage_unpack_nodes, dtype=np.int32)
        self._voltage_unpack_cols = np.asarray(voltage_unpack_cols, dtype=np.int32)
        voltage_mask = self.voltage_col >= 0
        self.voltage_col[voltage_mask] = self.n_angle + self.voltage_col[voltage_mask]

    def _build_y_matrix(self) -> np.ndarray:
        """Build the estimator admittance matrix with the same stamps as load flow."""
        n = len(self.nodes)
        rows = []
        cols = []
        data = []

        for br in self.branch_by_name.values():
            i = self.node_pos[br.i_node]
            j = self.node_pos[br.j_node]
            yff, yft, ytf, ytt = self.branch_stamp_by_name[br.name]
            rows.extend((i, i, j, j))
            cols.extend((i, j, i, j))
            data.extend((yff, yft, ytf, ytt))

        for tr in self.transformer_by_name.values():
            i = self.node_pos[tr.i_node]
            j = self.node_pos[tr.j_node]
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[tr.name]
            rows.extend((i, i, j, j))
            cols.extend((i, j, i, j))
            data.extend((yff, yft, ytf, ytt))

        for sc in self.network.shunt_compensators:
            if not getattr(sc, "is_alive", False):
                continue
            if sc.node not in self.node_pos:
                continue
            if sc.control_type in ("B", "Z") or sc.g_set != 0.0:
                y_sh = complex(sc.g_set, sc.b_set)
                if y_sh != 0.0:
                    pos = self.node_pos[sc.node]
                    rows.append(pos)
                    cols.append(pos)
                    data.append(y_sh)
        Y = coo_matrix((np.asarray(data, dtype=np.complex128), (np.asarray(rows), np.asarray(cols))), shape=(n, n)).tocsr()
        Y.sum_duplicates()
        return Y

    def _prepare_y_row_cache(self) -> None:
        """Cache sparse Y-row topology used repeatedly by generator Jacobian rows."""
        n = len(self.nodes)
        self._y_row_nodes = []
        self._y_row_y_conj = []
        self._y_row_off_mask = [None] * n
        self._y_row_off_nodes = [None] * n
        self._y_row_diag_conj = np.conj(self.Y.diagonal()).astype(np.complex128, copy=False)
        y_indices = self.Y.indices if issparse(self.Y) else None
        y_data_conj = np.conj(self.Y.data) if issparse(self.Y) else None

        for pos in range(n):
            if issparse(self.Y):
                start, end = self.Y.indptr[pos], self.Y.indptr[pos + 1]
                nodes = y_indices[start:end]
                y_conj = y_data_conj[start:end]
            else:
                row = self.Y[pos, :]
                nodes = np.nonzero(row)[0].astype(np.int32)
                y_values = np.asarray(row[nodes], dtype=np.complex128)
                y_conj = np.conj(y_values)

            self._y_row_nodes.append(nodes)
            self._y_row_y_conj.append(y_conj)

    def _group_loads(self) -> Dict[int, List]:
        grouped: Dict[int, List] = {}
        for load in self.load_by_name.values():
            grouped.setdefault(self.node_pos[load.node], []).append(load)
        return grouped

    def _group_generators(self) -> Dict[int, List]:
        grouped: Dict[int, List] = {}
        for gen in self.generator_by_name.values():
            grouped.setdefault(self.node_pos[gen.node], []).append(gen)
        return grouped

    def _generator_shares(self) -> Dict[str, float]:
        """Split a node-level solved injection among co-located generators."""
        shares = {}
        for gens in self.generators_at_pos.values():
            alphas = [float(gen.alpha) for gen in gens if getattr(gen, "alpha", None) is not None]
            total_alpha = sum(alphas)
            for gen in gens:
                if total_alpha > 0.0 and getattr(gen, "alpha", None) is not None:
                    shares[gen.name] = float(gen.alpha) / total_alpha
                else:
                    shares[gen.name] = 1.0 / len(gens)
        return shares

    def initial_state(self) -> np.ndarray:
        if self.flat_start:
            theta = np.zeros(len(self.nodes), dtype=np.float64)
            voltage = np.ones(len(self.nodes), dtype=np.float64)
            return self._pack_state(theta, voltage, rebase_angles=False)
        return self._file_state()

    def _file_state(self) -> np.ndarray:
        """Pack the E-file voltage/angle snapshot in the selected reference frame."""
        return self._pack_state(self.file_theta, self.file_voltage, rebase_angles=True)

    def state_layout(self) -> Dict[str, object]:
        """Expose the canonical AC state layout for reuse by hybrid orchestration."""
        return {
            "state_labels": self.state_labels,
            "state_meta": self.state_meta,
            "angle_col": self.angle_col,
            "voltage_col": self.voltage_col,
            "n_state": self.n_state,
            "references": self.references,
        }

    def state_cols_for_nodes(self, nodes) -> Tuple[np.ndarray, np.ndarray]:
        """Return theta/voltage state columns for the given AC nodes."""
        theta_cols = np.full(len(nodes), -1, dtype=np.int32)
        voltage_cols = np.full(len(nodes), -1, dtype=np.int32)
        for idx, node in enumerate(nodes):
            pos = self.node_pos.get(node.idx)
            if pos is None:
                continue
            theta_cols[idx] = int(self.angle_col[int(pos)])
            voltage_cols[idx] = int(self.voltage_col[int(pos)])
        return theta_cols, voltage_cols

    def _pack_state(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
        switch_current: Optional[np.ndarray] = None,
        gen_p: Optional[np.ndarray] = None,
        gen_q: Optional[np.ndarray] = None,
        load_p: Optional[np.ndarray] = None,
        load_q: Optional[np.ndarray] = None,
        shunt_q: Optional[np.ndarray] = None,
        rebase_angles: bool = True,
    ) -> np.ndarray:
        """Pack full node values into the WLS state vector, excluding reference angles."""
        x = np.zeros(self.n_state, dtype=np.float64)
        if self.n_angle:
            angle_values = np.asarray(theta[self.angle_state_pos], dtype=np.float64).copy()
            if rebase_angles:
                offsets = np.fromiter(
                    (self.reference_angle_by_pos.get(int(pos), 0.0) for pos in self.angle_state_pos),
                    dtype=np.float64,
                    count=self.n_angle,
                )
                angle_values -= offsets
            x[: self.n_angle] = angle_values
        if self.n_voltage:
            x[self.n_angle : self.base_switch_re] = voltage[self.voltage_state_pos]
        if switch_current is not None and self.n_switch_current:
            x[self.base_switch_re : self.base_switch_im] = switch_current.real
            x[self.base_switch_im : self.base_gen_p] = switch_current.imag
        if self.n_generator_power:
            x[self.base_gen_p : self.base_gen_q] = (
                self.initial_gen_p_array if gen_p is None else np.asarray(gen_p, dtype=np.float64)
            )
            x[self.base_gen_q : self.base_load_p] = (
                self.initial_gen_q_array if gen_q is None else np.asarray(gen_q, dtype=np.float64)
            )
        if self.n_load_power:
            x[self.base_load_p : self.base_load_q] = (
                self.initial_load_p_array if load_p is None else np.asarray(load_p, dtype=np.float64)
            )
            x[self.base_load_q : self.base_shunt_q] = (
                self.initial_load_q_array if load_q is None else np.asarray(load_q, dtype=np.float64)
            )
        if self.n_shunt_q:
            x[self.base_shunt_q : self.n_state] = (
                self.initial_shunt_q_array if shunt_q is None else np.asarray(shunt_q, dtype=np.float64)
            )
        return x

    def _unpack_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Restore full theta/V arrays from the compact WLS state vector."""
        n = len(self.nodes)
        theta = np.zeros(n, dtype=np.float64)
        if self._angle_ref_nodes.size:
            theta[self._angle_ref_nodes] = self._angle_ref_values
        if self._angle_unpack_nodes.size:
            theta[self._angle_unpack_nodes] = x[self._angle_unpack_cols]

        state_voltage = np.asarray(x[self.n_angle : self.base_switch_re], dtype=np.float64).copy()
        state_voltage[state_voltage < self.voltage_floor] = self.voltage_floor
        voltage = np.ones(n, dtype=np.float64)
        if self._voltage_ref_nodes.size:
            voltage[self._voltage_ref_nodes] = self._voltage_ref_values
        if self._voltage_unpack_nodes.size:
            voltage[self._voltage_unpack_nodes] = state_voltage[self._voltage_unpack_cols]
        voltage[voltage < self.voltage_floor] = self.voltage_floor
        return theta, voltage

    def _switch_current_from_state(self, x: np.ndarray) -> np.ndarray:
        if self.n_switch_current == 0:
            return np.array([], dtype=np.complex128)
        return (
            np.asarray(x[self.base_switch_re : self.base_switch_im], dtype=np.float64)
            + 1j * np.asarray(x[self.base_switch_im : self.base_gen_p], dtype=np.float64)
        )

    @staticmethod
    def _complex_voltage(theta: np.ndarray, voltage: np.ndarray) -> np.ndarray:
        return voltage * np.exp(1j * theta)

    def _load_power(self, voltage: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        p_values, q_values, p_load, q_load = self._load_power_arrays(voltage)
        load_power = {
            load.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, load in enumerate(self.load_order)
        }
        return load_power, p_load, q_load

    def _load_power_arrays(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-load ZIP powers plus node-level load totals."""
        p_load = np.zeros(len(self.nodes), dtype=np.float64)
        q_load = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), p_load, q_load
        vm = voltage[self.load_pos_array]
        p_values = self.load_pv0_array + self.load_pv1_array * vm + self.load_pv2_array * vm * vm
        q_values = self.load_qv0_array + self.load_qv1_array * vm + self.load_qv2_array * vm * vm
        np.add.at(p_load, self.load_pos_array, p_values)
        np.add.at(q_load, self.load_pos_array, q_values)
        return p_values, q_values, p_load, q_load

    def _load_power_totals(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute only node-level load totals for generator balance equations."""
        _, _, p_load, q_load = self._load_power_arrays(voltage)
        return p_load, q_load

    def _load_power_from_state(self, x: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        """Return load P/Q states and node totals used by balance equations."""
        p_values = np.asarray(x[self.base_load_p : self.base_load_q], dtype=np.float64)
        q_values = np.asarray(x[self.base_load_q : self.base_shunt_q], dtype=np.float64)
        p_load = np.zeros(len(self.nodes), dtype=np.float64)
        q_load = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size:
            p_load = np.bincount(self.load_pos_array, weights=p_values, minlength=self.n_nodes)
            q_load = np.bincount(self.load_pos_array, weights=q_values, minlength=self.n_nodes)
        load_power = {
            load.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, load in enumerate(self.load_order)
        }
        return load_power, p_load, q_load

    def _load_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level load totals from explicit P_LOAD/Q_LOAD states."""
        if not self.load_pos_array.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(self.load_pos_array, weights=x[self.base_load_p : self.base_load_q], minlength=self.n_nodes),
            np.bincount(self.load_pos_array, weights=x[self.base_load_q : self.base_shunt_q], minlength=self.n_nodes),
        )

    def _shunt_q_injections_from_state(self, x: np.ndarray) -> np.ndarray:
        """Return node-level reactive injections from V-control shunt Q states."""
        if not self.n_shunt_q:
            return np.zeros(self.n_nodes, dtype=np.float64)
        return np.bincount(
            self.shunt_q_pos_array,
            weights=x[self.base_shunt_q : self.n_state],
            minlength=self.n_nodes,
        )

    def _generator_power_from_state(self, x: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        """Return generator P/Q states and node totals used by balance equations."""
        p_values = np.asarray(x[self.base_gen_p : self.base_gen_q], dtype=np.float64)
        q_values = np.asarray(x[self.base_gen_q : self.base_load_p], dtype=np.float64)
        p_gen = np.zeros(len(self.nodes), dtype=np.float64)
        q_gen = np.zeros(len(self.nodes), dtype=np.float64)
        if self.generator_pos_array.size:
            p_gen = np.bincount(self.generator_pos_array, weights=p_values, minlength=self.n_nodes)
            q_gen = np.bincount(self.generator_pos_array, weights=q_values, minlength=self.n_nodes)
        gen_power = {
            gen.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, gen in enumerate(self.generator_order)
        }
        return gen_power, p_gen, q_gen

    def _generator_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level generator totals from explicit P_GEN/Q_GEN states."""
        if not self.generator_pos_array.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(self.generator_pos_array, weights=x[self.base_gen_p : self.base_gen_q], minlength=self.n_nodes),
            np.bincount(self.generator_pos_array, weights=x[self.base_gen_q : self.base_load_p], minlength=self.n_nodes),
        )

    def _switch_power_injections(
        self,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node injections from explicit zero-branch/switch current states."""
        if self.n_switch_current == 0:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        end_pos = self.switch_balance_end_pos
        current = switch_current[self.switch_balance_end_current_idx] * self.switch_balance_sign
        s = voltage_complex[end_pos] * np.conj(current)
        return (
            np.bincount(end_pos, weights=s.real, minlength=self.n_nodes),
            np.bincount(end_pos, weights=s.imag, minlength=self.n_nodes),
        )

    def _power_balance_totals(
        self,
        x: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate nodal mismatch S_net + S_switch + S_load - S_gen - Q_shunt."""
        s_network = voltage_complex * np.conj(self.Y.dot(voltage_complex))
        p_switch, q_switch = self._switch_power_injections(voltage_complex, switch_current)
        p_load, q_load = self._load_power_totals_from_state(x)
        p_gen, q_gen = self._generator_power_totals_from_state(x)
        q_shunt = self._shunt_q_injections_from_state(x)
        return (
            s_network.real + p_switch + p_load - p_gen,
            s_network.imag + q_switch + q_load - q_gen - q_shunt,
        )

    def _generator_power(
        self,
        voltage_complex: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        switch_current: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """Infer generator injections from network balance and local load demand."""
        p_gen_total, q_gen_total = self._generator_power_totals(
            voltage_complex,
            p_load,
            q_load,
            switch_current,
        )
        gen_power = {}
        for pos, gens in self.generators_at_pos.items():
            for gen in gens:
                share = self.generator_share_by_name[gen.name]
                gen_power[gen.name] = (share * p_gen_total[pos], share * q_gen_total[pos])
        return gen_power

    def _generator_power_totals(
        self,
        voltage_complex: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer node-level generator injections from network, load and zero-current balances."""
        s_network = voltage_complex * np.conj(self.Y.dot(voltage_complex))
        if self.n_switch_current:
            p_switch = np.zeros(len(self.nodes), dtype=np.float64)
            q_switch = np.zeros(len(self.nodes), dtype=np.float64)
            s_from = voltage_complex[self.zero_current_i] * np.conj(switch_current)
            s_to = voltage_complex[self.zero_current_j] * np.conj(-switch_current)
            np.add.at(p_switch, self.zero_current_i, s_from.real)
            np.add.at(q_switch, self.zero_current_i, s_from.imag)
            np.add.at(p_switch, self.zero_current_j, s_to.real)
            np.add.at(q_switch, self.zero_current_j, s_to.imag)
        else:
            p_switch = q_switch = 0.0
        p_gen_total = s_network.real + p_load
        q_gen_total = s_network.imag + q_load
        p_gen_total = p_gen_total + p_switch
        q_gen_total = q_gen_total + q_switch
        return p_gen_total, q_gen_total

    def _branch_power(self, device, is_transformer: bool, voltage_complex: np.ndarray) -> Tuple[complex, complex]:
        i = self.node_pos[device.i_node]
        j = self.node_pos[device.j_node]
        if is_transformer:
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[device.name]
        else:
            yff, yft, ytf, ytt = self.branch_stamp_by_name[device.name]
        vi = voltage_complex[i]
        vj = voltage_complex[j]
        sij = vi * np.conj(yff * vi + yft * vj)
        sji = vj * np.conj(ytf * vi + ytt * vj)
        return sij, sji

    def _branch_current(self, device, is_transformer: bool, voltage_complex: np.ndarray) -> Tuple[complex, complex]:
        i = self.node_pos[device.i_node]
        j = self.node_pos[device.j_node]
        if is_transformer:
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[device.name]
        else:
            yff, yft, ytf, ytt = self.branch_stamp_by_name[device.name]
        vi = voltage_complex[i]
        vj = voltage_complex[j]
        return yff * vi + yft * vj, ytf * vi + ytt * vj

    def _power_current(self, p: float, q: float, voltage: float) -> float:
        if abs(voltage) <= self.min_current_voltage:
            return 0.0
        return float(np.hypot(p, q) / voltage)

    def _calc_quantities(self, x: np.ndarray):
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        load_power, _, _ = self._load_power_from_state(x)
        gen_power, _, _ = self._generator_power_from_state(x)
        return theta, voltage, voltage_complex, load_power, gen_power, switch_current

    @staticmethod
    def _int_array(values: Sequence[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _bool_array(values: Sequence[bool]) -> np.ndarray:
        return np.asarray(values, dtype=bool)

    @staticmethod
    def _complex_array(values: Sequence[complex]) -> np.ndarray:
        return np.asarray(values, dtype=np.complex128)

    @staticmethod
    def _concat_plan_arrays(chunks: Sequence[np.ndarray], dtype) -> np.ndarray:
        non_empty = [np.asarray(chunk, dtype=dtype) for chunk in chunks if len(chunk)]
        if not non_empty:
            return np.asarray([], dtype=dtype)
        return np.concatenate(non_empty).astype(dtype, copy=False)

    @staticmethod
    def _build_branch_stamp_map(devices: Sequence[object], with_tap: bool) -> Dict[str, Tuple[complex, complex, complex, complex]]:
        """Build MATPOWER branch stamps in one vectorized pass, then attach them by device name."""
        if not devices:
            return {}
        if with_tap:
            yff, yft, ytf, ytt = matpower_transformer_stamp_vectorized(
                [dev.r for dev in devices],
                [dev.x for dev in devices],
                [getattr(dev, "gt", 0.0) for dev in devices],
                [getattr(dev, "bt", getattr(dev, "b", 0.0) / 2.0) for dev in devices],
                [dev.tap for dev in devices],
                [dev.shift for dev in devices],
            )
        else:
            yff, yft, ytf, ytt = matpower_branch_stamp_vectorized(
                [dev.r for dev in devices],
                [dev.x for dev in devices],
                [dev.b for dev in devices],
            )
        return {
            dev.name: (complex(yff[idx]), complex(yft[idx]), complex(ytf[idx]), complex(ytt[idx]))
            for idx, dev in enumerate(devices)
        }

    def _build_measurement_plan_lookup_arrays(self) -> None:
        def safe_col(values, pos: int) -> int:
            if values is None or pos < 0 or pos >= len(values):
                return -1
            return int(values[pos])

        voltage_col = getattr(self, "voltage_col", None)
        angle_col = getattr(self, "angle_col", None)
        node_pos_map = getattr(self, "node_pos", {})
        node_items = []
        for name, node in getattr(self, "node_by_name", {}).items():
            pos = node_pos_map.get(node.idx)
            if pos is not None:
                node_items.append((name, int(pos)))
        self._ac_node_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(node_items)}
        self._ac_node_plan_pos = self._int_array([pos for _, pos in node_items])
        self._ac_node_plan_voltage_col = self._int_array(
            [safe_col(voltage_col, pos) for _, pos in node_items]
        )
        self._ac_node_plan_angle_col = self._int_array(
            [safe_col(angle_col, pos) for _, pos in node_items]
        )

        def branch_items(devices, stamps):
            items = []
            for name, dev in devices.items():
                if dev.i_node not in node_pos_map or dev.j_node not in node_pos_map or name not in stamps:
                    continue
                yff, yft, ytf, ytt = stamps[name]
                items.append((name, node_pos_map[dev.i_node], node_pos_map[dev.j_node], yff, yft, ytf, ytt))
            return items

        branch_plan_items = branch_items(getattr(self, "branch_by_name", {}), getattr(self, "branch_stamp_by_name", {}))
        self._ac_branch_plan_name_to_pos = {name: pos for pos, (name, *_rest) in enumerate(branch_plan_items)}
        self._ac_branch_plan_i = self._int_array([item[1] for item in branch_plan_items])
        self._ac_branch_plan_j = self._int_array([item[2] for item in branch_plan_items])
        self._ac_branch_plan_yff = self._complex_array([item[3] for item in branch_plan_items])
        self._ac_branch_plan_yft = self._complex_array([item[4] for item in branch_plan_items])
        self._ac_branch_plan_ytf = self._complex_array([item[5] for item in branch_plan_items])
        self._ac_branch_plan_ytt = self._complex_array([item[6] for item in branch_plan_items])

        transformer_plan_items = branch_items(
            getattr(self, "transformer_by_name", {}),
            getattr(self, "transformer_stamp_by_name", {}),
        )
        self._ac_transformer_plan_name_to_pos = {
            name: pos for pos, (name, *_rest) in enumerate(transformer_plan_items)
        }
        self._ac_transformer_plan_i = self._int_array([item[1] for item in transformer_plan_items])
        self._ac_transformer_plan_j = self._int_array([item[2] for item in transformer_plan_items])
        self._ac_transformer_plan_yff = self._complex_array([item[3] for item in transformer_plan_items])
        self._ac_transformer_plan_yft = self._complex_array([item[4] for item in transformer_plan_items])
        self._ac_transformer_plan_ytf = self._complex_array([item[5] for item in transformer_plan_items])
        self._ac_transformer_plan_ytt = self._complex_array([item[6] for item in transformer_plan_items])

        def zero_items(devices, current_pos_by_name):
            items = []
            for name, dev in devices.items():
                if dev.i_node not in node_pos_map or dev.j_node not in node_pos_map:
                    continue
                current_pos = current_pos_by_name.get(name)
                if current_pos is None:
                    continue
                items.append((name, node_pos_map[dev.i_node], node_pos_map[dev.j_node], int(current_pos)))
            return items

        zero_plan_items = zero_items(getattr(self, "zero_branch_by_name", {}), getattr(self, "zero_branch_pos", {}))
        self._ac_zero_branch_plan_name_to_pos = {
            name: pos for pos, (name, *_rest) in enumerate(zero_plan_items)
        }
        self._ac_zero_branch_plan_i = self._int_array([item[1] for item in zero_plan_items])
        self._ac_zero_branch_plan_j = self._int_array([item[2] for item in zero_plan_items])
        self._ac_zero_branch_plan_current_pos = self._int_array([item[3] for item in zero_plan_items])

        break_plan_items = zero_items(getattr(self, "break_by_name", {}), getattr(self, "break_pos", {}))
        self._ac_break_plan_name_to_pos = {name: pos for pos, (name, *_rest) in enumerate(break_plan_items)}
        self._ac_break_plan_i = self._int_array([item[1] for item in break_plan_items])
        self._ac_break_plan_j = self._int_array([item[2] for item in break_plan_items])
        self._ac_break_plan_current_pos = self._int_array([item[3] for item in break_plan_items])

        gen_items = []
        for name, gen in getattr(self, "generator_by_name", {}).items():
            index = getattr(self, "generator_state_index_by_name", {}).get(name)
            if index is None or gen.node not in node_pos_map:
                continue
            gen_items.append((name, node_pos_map[gen.node], int(index)))
        self._ac_generator_plan_name_to_pos = {name: pos for pos, (name, *_rest) in enumerate(gen_items)}
        self._ac_generator_plan_node_pos = self._int_array([item[1] for item in gen_items])
        self._ac_generator_plan_voltage_col = self._int_array(
            [safe_col(voltage_col, item[1]) for item in gen_items]
        )
        self._ac_generator_plan_index = self._int_array([item[2] for item in gen_items])

        load_items = []
        for name, load in getattr(self, "load_by_name", {}).items():
            index = getattr(self, "load_state_index_by_name", {}).get(name)
            if index is None or load.node not in node_pos_map:
                continue
            load_items.append((name, node_pos_map[load.node], int(index)))
        self._ac_load_plan_name_to_pos = {name: pos for pos, (name, *_rest) in enumerate(load_items)}
        self._ac_load_plan_node_pos = self._int_array([item[1] for item in load_items])
        self._ac_load_plan_voltage_col = self._int_array(
            [safe_col(voltage_col, item[1]) for item in load_items]
        )
        self._ac_load_plan_index = self._int_array([item[2] for item in load_items])

        self._ac_measurement_plan_device_pos_by_type_code = {
            _DEVICE_TYPE_CODES["ACNode"]: self._ac_node_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACBranch"]: self._ac_branch_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACTransformer"]: self._ac_transformer_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACLoad"]: self._ac_load_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACGenerator"]: self._ac_generator_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACZeroBranch"]: self._ac_zero_branch_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACBreak"]: self._ac_break_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACZeroBranchConstraint"]: self._ac_zero_branch_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACBreakConstraint"]: self._ac_break_plan_name_to_pos,
            _DEVICE_TYPE_CODES["ACPowerBalance"]: self._ac_node_plan_name_to_pos,
        }
        self._ac_branch_transformer_plan_kind_by_type_code = {
            _DEVICE_TYPE_CODES["ACBranch"]: _AC_TERMINAL_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACTransformer"]: _AC_TERMINAL_MEASUREMENT_KIND,
        }
        self._ac_zero_current_plan_kind_by_type_code = {
            _DEVICE_TYPE_CODES["ACZeroBranch"]: _AC_ZERO_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACBreak"]: _AC_TERMINAL_MEASUREMENT_KIND,
        }
        self._ac_simple_plan_kind_by_type_code = {
            _DEVICE_TYPE_CODES["ACNode"]: _AC_NODE_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACGenerator"]: _AC_GENERATOR_SIMPLE_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACLoad"]: _AC_LOAD_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACZeroBranchConstraint"]: _AC_CONSTRAINT_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["ACBreakConstraint"]: _AC_CONSTRAINT_MEASUREMENT_KIND,
        }
        self._ac_generator_plan_kind_by_type_code = {
            _DEVICE_TYPE_CODES["ACGenerator"]: _AC_GENERATOR_POWER_MEASUREMENT_KIND,
        }
        self._ac_balance_plan_kind_by_type_code = {
            _DEVICE_TYPE_CODES["ACPowerBalance"]: _AC_BALANCE_MEASUREMENT_KIND,
        }

    def _ensure_measurement_plan_lookup_arrays(self) -> None:
        if not hasattr(self, "_ac_measurement_plan_device_pos_by_type_code"):
            self._build_measurement_plan_lookup_arrays()

    def _measurement_plan_table(
        self,
        measurements: Sequence[Measurement],
        meas_kind_by_type_code: Dict[int, Dict[str, int]],
        device_pos_by_type_code: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        self._ensure_measurement_plan_lookup_arrays()
        return build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=device_pos_by_type_code or self._ac_measurement_plan_device_pos_by_type_code,
            meas_kind_by_type_code=meas_kind_by_type_code,
            table_builder=_measurement_table_from_measurements,
        )

    def _branch_transformer_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_branch_transformer_vector_plan(measurements)
            self._active_branch_transformer_vector_plan = plan
            return plan
        key = id(measurements)
        cached = self._branch_transformer_vector_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]
        plan = self._build_branch_transformer_vector_plan(measurements)
        if len(self._branch_transformer_vector_plan_cache) > 16:
            self._branch_transformer_vector_plan_cache.clear()
        self._branch_transformer_vector_plan_cache[key] = (measurements, plan)
        return plan

    def _build_branch_transformer_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Precompute ACBranch/ACTransformer row metadata for repeated h/H calls."""
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_branch_transformer_plan_kind_by_type_code,
        )
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        handled_mask = np.asarray(plan_table.handled, dtype=bool).copy()

        def build_device_rows(device_code, device_pos, i_array, j_array, yff, yft, ytf, ytt):
            rows = row[(code == device_code) & handled_mask]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["V_FROM"]
            v_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["V_TO"]
            p_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["P_FROM"]
            q_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["Q_FROM"]
            p_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["P_TO"]
            q_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["Q_TO"]
            i_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["I_FROM"]
            i_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["I_TO"]
            from_power = p_from | q_from
            to_power = p_to | q_to
            return {
                "voltage_rows": self._concat_plan_arrays((rows[v_from], rows[v_to]), np.int64),
                "voltage_pos": self._concat_plan_arrays((i[v_from], j[v_to]), np.int64),
                "power_rows": self._concat_plan_arrays((rows[from_power], rows[to_power]), np.int64),
                "power_is_p": self._concat_plan_arrays((p_from[from_power], p_to[to_power]), bool),
                "power_own": self._concat_plan_arrays((i[from_power], j[to_power]), np.int64),
                "power_other": self._concat_plan_arrays((j[from_power], i[to_power]), np.int64),
                "power_y_self": self._concat_plan_arrays((yff[pos[from_power]], ytt[pos[to_power]]), np.complex128),
                "power_y_mutual": self._concat_plan_arrays((yft[pos[from_power]], ytf[pos[to_power]]), np.complex128),
                "current_rows": self._concat_plan_arrays((rows[i_from], rows[i_to]), np.int64),
                "current_own": self._concat_plan_arrays((i[i_from], j[i_to]), np.int64),
                "current_other": self._concat_plan_arrays((j[i_from], i[i_to]), np.int64),
                "current_y_self": self._concat_plan_arrays((yff[pos[i_from]], ytt[pos[i_to]]), np.complex128),
                "current_y_mutual": self._concat_plan_arrays((yft[pos[i_from]], ytf[pos[i_to]]), np.complex128),
            }

        branch_plan = build_device_rows(
            _DEVICE_TYPE_CODES["ACBranch"],
            plan_table.device_pos,
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
            self._ac_branch_plan_yff,
            self._ac_branch_plan_yft,
            self._ac_branch_plan_ytf,
            self._ac_branch_plan_ytt,
        )
        transformer_plan = build_device_rows(
            _DEVICE_TYPE_CODES["ACTransformer"],
            plan_table.device_pos,
            self._ac_transformer_plan_i,
            self._ac_transformer_plan_j,
            self._ac_transformer_plan_yff,
            self._ac_transformer_plan_yft,
            self._ac_transformer_plan_ytf,
            self._ac_transformer_plan_ytt,
        )
        voltage_rows = self._concat_plan_arrays((branch_plan["voltage_rows"], transformer_plan["voltage_rows"]), np.int64)
        voltage_pos = self._concat_plan_arrays((branch_plan["voltage_pos"], transformer_plan["voltage_pos"]), np.int64)
        power_rows = self._concat_plan_arrays((branch_plan["power_rows"], transformer_plan["power_rows"]), np.int64)
        power_is_p = self._concat_plan_arrays((branch_plan["power_is_p"], transformer_plan["power_is_p"]), bool)
        power_own = self._concat_plan_arrays((branch_plan["power_own"], transformer_plan["power_own"]), np.int64)
        power_other = self._concat_plan_arrays((branch_plan["power_other"], transformer_plan["power_other"]), np.int64)
        power_y_self = self._concat_plan_arrays((branch_plan["power_y_self"], transformer_plan["power_y_self"]), np.complex128)
        power_y_mutual = self._concat_plan_arrays(
            (branch_plan["power_y_mutual"], transformer_plan["power_y_mutual"]),
            np.complex128,
        )
        current_rows = self._concat_plan_arrays((branch_plan["current_rows"], transformer_plan["current_rows"]), np.int64)
        current_own = self._concat_plan_arrays((branch_plan["current_own"], transformer_plan["current_own"]), np.int64)
        current_other = self._concat_plan_arrays((branch_plan["current_other"], transformer_plan["current_other"]), np.int64)
        current_y_self = self._concat_plan_arrays(
            (branch_plan["current_y_self"], transformer_plan["current_y_self"]),
            np.complex128,
        )
        current_y_mutual = self._concat_plan_arrays(
            (branch_plan["current_y_mutual"], transformer_plan["current_y_mutual"]),
            np.complex128,
        )
        return {
            "handled_mask": handled_mask,
            "voltage_rows": voltage_rows,
            "voltage_pos": voltage_pos,
            "voltage_cols": self.voltage_col[voltage_pos].astype(np.int64, copy=False),
            "power_rows": power_rows,
            "power_is_p": power_is_p,
            "power_own": power_own,
            "power_other": power_other,
            "power_y_self": power_y_self,
            "power_y_mutual": power_y_mutual,
            "current_rows": current_rows,
            "current_own": current_own,
            "current_other": current_other,
            "current_y_self": current_y_self,
            "current_y_mutual": current_y_mutual,
        }

    def _fill_branch_transformer_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACBranch/ACTransformer terminal P/Q/V/I measurements."""
        plan = self._branch_transformer_vector_plan(measurements)
        voltage_rows = plan["voltage_rows"]
        if voltage_rows.size:
            values[voltage_rows] = voltage[plan["voltage_pos"]]

        rows = plan["power_rows"]
        if rows.size:
            own = plan["power_own"]
            other = plan["power_other"]
            y_self = plan["power_y_self"]
            y_mutual = plan["power_y_mutual"]
            power = voltage_complex[own] * np.conj(y_self * voltage_complex[own] + y_mutual * voltage_complex[other])
            values[rows] = np.where(plan["power_is_p"], power.real, power.imag)

        rows = plan["current_rows"]
        if rows.size:
            own = plan["current_own"]
            other = plan["current_other"]
            y_self = plan["current_y_self"]
            y_mutual = plan["current_y_mutual"]
            current = y_self * voltage_complex[own] + y_mutual * voltage_complex[other]
            values[rows] = np.abs(current)

        return plan["handled_mask"]

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        measurements = self._normalize_measurements(measurements)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        load_power = gen_power = None
        branch_flow_cache = {}
        values = np.zeros(len(measurements), dtype=np.float64)
        vectorized_branch_rows = self._fill_branch_transformer_values_vectorized(
            values,
            measurements,
            voltage,
            voltage_complex,
        )
        vectorized_zero_rows = self._fill_zero_current_values_vectorized(
            values,
            measurements,
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_simple_rows = self._fill_simple_values_vectorized(
            values,
            measurements,
            x,
            theta,
            voltage,
        )
        vectorized_generator_rows = self._fill_generator_values_vectorized(
            values,
            measurements,
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_values_vectorized(
            values,
            measurements,
            x,
            voltage_complex,
            switch_current,
        )
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_zero_rows
            | vectorized_simple_rows
            | vectorized_generator_rows
            | vectorized_balance_rows
        )
        if (measurements is self.active_measurements and self.active_measurements_are_vectorized) or np.all(vectorized_rows):
            return values

        row_meta = self._measurement_sequence_arrays(measurements)
        if row_meta is not None:
            device_types, device_names, meas_types = row_meta
            row_iter = zip(range(len(measurements)), device_types, device_names, meas_types)
        else:
            row_iter = ((row, meas.device_type, meas.device_name, meas.meas_type) for row, meas in enumerate(measurements))

        for row, device_type, device_name, mtype in row_iter:
            if vectorized_rows[row]:
                continue
            if device_type == "ACNode":
                node = self.node_by_name[device_name]
                pos = self.node_pos[node.idx]
                if mtype == "V":
                    values[row] = voltage[pos]
                elif mtype in ("ANGLE", "THETA"):
                    values[row] = theta[pos]
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")
            elif device_type == "ACBranch":
                br = self.branch_by_name[device_name]
                if mtype == "V_FROM":
                    values[row] = voltage[self.node_pos[br.i_node]]
                elif mtype == "V_TO":
                    values[row] = voltage[self.node_pos[br.j_node]]
                else:
                    cache_key = ("ACBranch", br.name)
                    if cache_key not in branch_flow_cache:
                        sij, sji = self._branch_power(br, False, voltage_complex)
                        iij, iji = self._branch_current(br, False, voltage_complex)
                        branch_flow_cache[cache_key] = (sij, sji, iij, iji)
                    sij, sji, iij, iji = branch_flow_cache[cache_key]
                    values[row] = self._flow_value(mtype, sij, sji, iij, iji)
            elif device_type == "ACTransformer":
                tr = self.transformer_by_name[device_name]
                if mtype == "V_FROM":
                    values[row] = voltage[self.node_pos[tr.i_node]]
                elif mtype == "V_TO":
                    values[row] = voltage[self.node_pos[tr.j_node]]
                else:
                    cache_key = ("ACTransformer", tr.name)
                    if cache_key not in branch_flow_cache:
                        sij, sji = self._branch_power(tr, True, voltage_complex)
                        iij, iji = self._branch_current(tr, True, voltage_complex)
                        branch_flow_cache[cache_key] = (sij, sji, iij, iji)
                    sij, sji, iij, iji = branch_flow_cache[cache_key]
                    values[row] = self._flow_value(mtype, sij, sji, iij, iji)
            elif device_type == "ACGenerator":
                gen = self.generator_by_name[device_name]
                node = self.node_by_name[gen.node_obj.name]
                if gen_power is None:
                    gen_power, _, _ = self._generator_power_from_state(x)
                p, q = gen_power[gen.name]
                if mtype == "P_GEN":
                    values[row] = p
                elif mtype == "Q_GEN":
                    values[row] = q
                elif mtype == "V_GEN":
                    values[row] = voltage[self.node_pos[node.idx]]
                elif mtype == "I_GEN":
                    values[row] = self._power_current(p, q, voltage[self.node_pos[node.idx]])
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")
            elif device_type == "ACLoad":
                load = self.load_by_name[device_name]
                node = self.node_by_name[load.node_obj.name]
                if load_power is None:
                    load_power, _, _ = self._load_power_from_state(x)
                p, q = load_power[load.name]
                if mtype == "P_LOAD":
                    values[row] = p
                elif mtype == "Q_LOAD":
                    values[row] = q
                elif mtype == "V_LOAD":
                    values[row] = voltage[self.node_pos[node.idx]]
                elif mtype == "I_LOAD":
                    values[row] = self._power_current(p, q, voltage[self.node_pos[node.idx]])
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")
            elif device_type == "ACZeroBranch":
                zbr = self.zero_branch_by_name[device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[zbr.i_node]] - voltage[self.node_pos[zbr.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[zbr.i_node]] - theta[self.node_pos[zbr.j_node]]
                else:
                    current = switch_current[self.zero_branch_pos[zbr.name]]
                    values[row] = self._zero_current_measurement_value(zbr, current, mtype, voltage, voltage_complex)
            elif device_type == "ACZeroBranchConstraint":
                zbr = self.zero_branch_by_name[device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[zbr.i_node]] - voltage[self.node_pos[zbr.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[zbr.i_node]] - theta[self.node_pos[zbr.j_node]]
                else:
                    raise RuntimeError(f"Unsupported ACZeroBranchConstraint measurement type: {mtype}")
            elif device_type == "ACBreakConstraint":
                brk = self.break_by_name[device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[brk.i_node]] - voltage[self.node_pos[brk.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[brk.i_node]] - theta[self.node_pos[brk.j_node]]
                else:
                    raise RuntimeError(f"Unsupported ACBreakConstraint measurement type: {mtype}")
            elif device_type == "ACBreak":
                brk = self.break_by_name[device_name]
                current = switch_current[self.break_pos[brk.name]]
                values[row] = self._zero_current_measurement_value(brk, current, mtype, voltage, voltage_complex)
            elif device_type == "ACPowerBalance":
                pos = self.node_pos[self.node_by_name[device_name].idx]
                p_balance, q_balance = self._power_balance_totals(x, voltage_complex, switch_current)
                if mtype == "P_BALANCE":
                    values[row] = p_balance[pos]
                elif mtype == "Q_BALANCE":
                    values[row] = q_balance[pos]
                else:
                    raise RuntimeError(f"Unsupported ACPowerBalance measurement type: {mtype}")
            else:
                raise RuntimeError(f"Unsupported measurement device type: {device_type}")
        return values

    @staticmethod
    def _flow_value(meas_type: str, s_from: complex, s_to: complex, i_from: complex = 0.0, i_to: complex = 0.0) -> float:
        if meas_type == "P_FROM":
            return float(s_from.real)
        if meas_type == "Q_FROM":
            return float(s_from.imag)
        if meas_type == "P_TO":
            return float(s_to.real)
        if meas_type == "Q_TO":
            return float(s_to.imag)
        if meas_type == "I_FROM":
            return float(abs(i_from))
        if meas_type == "I_TO":
            return float(abs(i_to))
        raise RuntimeError(f"Unsupported flow measurement type: {meas_type}")

    def _zero_current_measurement_value(
        self,
        device,
        current: complex,
        meas_type: str,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> float:
        if meas_type.endswith("_FROM"):
            pos = self.node_pos[device.i_node]
            signed_current = current
        elif meas_type.endswith("_TO"):
            pos = self.node_pos[device.j_node]
            signed_current = -current
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

        if meas_type.startswith("P") or meas_type.startswith("Q"):
            power = voltage_complex[pos] * np.conj(signed_current)
            return float(power.real if meas_type.startswith("P") else power.imag)
        if meas_type.startswith("I"):
            return float(abs(current))
        if meas_type.startswith("V"):
            return float(voltage[pos])
        raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

    def _zero_current_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_zero_current_vector_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_zero_current_vector_plan(measurements)
            self._active_zero_current_vector_plan = plan
            return plan
        key = id(measurements)
        cached = self._zero_current_vector_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]
        plan = self._build_zero_current_vector_plan(measurements)
        if len(self._zero_current_vector_plan_cache) > 16:
            self._zero_current_vector_plan_cache.clear()
        self._zero_current_vector_plan_cache[key] = (measurements, plan)
        return plan

    def _build_zero_current_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Precompute ACSwitch/ACZeroBranch row metadata for explicit-current states."""
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_zero_current_plan_kind_by_type_code,
        )
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        handled_mask = np.asarray(plan_table.handled, dtype=bool).copy()

        def build_device_rows(device_code, i_array, j_array, current_pos_array):
            rows = row[(code == device_code) & handled_mask]
            pos = plan_table.device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            current_pos = current_pos_array[pos]
            v_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["V_FROM"]
            v_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["V_TO"]
            p_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["P_FROM"]
            q_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["Q_FROM"]
            p_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["P_TO"]
            q_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["Q_TO"]
            i_from = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["I_FROM"]
            i_to = row_kind == _AC_TERMINAL_MEASUREMENT_KIND["I_TO"]
            from_power = p_from | q_from
            to_power = p_to | q_to
            return {
                "voltage_rows": self._concat_plan_arrays((rows[v_from], rows[v_to]), np.int64),
                "voltage_pos": self._concat_plan_arrays((i[v_from], j[v_to]), np.int64),
                "power_rows": self._concat_plan_arrays((rows[from_power], rows[to_power]), np.int64),
                "power_is_p": self._concat_plan_arrays((p_from[from_power], p_to[to_power]), bool),
                "power_pos": self._concat_plan_arrays((i[from_power], j[to_power]), np.int64),
                "power_current_idx": self._concat_plan_arrays(
                    (current_pos[from_power], current_pos[to_power]),
                    np.int64,
                ),
                "power_sign": self._concat_plan_arrays(
                    (
                        np.ones(np.count_nonzero(from_power), dtype=np.float64),
                        -np.ones(np.count_nonzero(to_power), dtype=np.float64),
                    ),
                    np.float64,
                ),
                "current_rows": self._concat_plan_arrays((rows[i_from], rows[i_to]), np.int64),
                "current_idx": self._concat_plan_arrays((current_pos[i_from], current_pos[i_to]), np.int64),
                "v_diff_rows": rows[row_kind == _AC_ZERO_MEASUREMENT_KIND["V_DIFF"]],
                "angle_diff_rows": rows[row_kind == _AC_ZERO_MEASUREMENT_KIND["ANGLE_DIFF"]],
                "i": i,
                "j": j,
                "kind": row_kind,
            }

        zero_plan = build_device_rows(
            _DEVICE_TYPE_CODES["ACZeroBranch"],
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
            self._ac_zero_branch_plan_current_pos,
        )
        break_plan = build_device_rows(
            _DEVICE_TYPE_CODES["ACBreak"],
            self._ac_break_plan_i,
            self._ac_break_plan_j,
            self._ac_break_plan_current_pos,
        )
        zero_rows = row[(code == _DEVICE_TYPE_CODES["ACZeroBranch"]) & handled_mask]
        zero_pos = plan_table.device_pos[zero_rows]
        zero_kind = kind[zero_rows]
        zero_i = self._ac_zero_branch_plan_i[zero_pos]
        zero_j = self._ac_zero_branch_plan_j[zero_pos]
        v_diff = zero_kind == _AC_ZERO_MEASUREMENT_KIND["V_DIFF"]
        angle_diff = zero_kind == _AC_ZERO_MEASUREMENT_KIND["ANGLE_DIFF"]
        voltage_diff_rows = zero_rows[v_diff]
        voltage_diff_i = zero_i[v_diff]
        voltage_diff_j = zero_j[v_diff]
        angle_diff_rows = zero_rows[angle_diff]
        angle_diff_i = zero_i[angle_diff]
        angle_diff_j = zero_j[angle_diff]
        scalar_rows = self._concat_plan_arrays(
            (
                np.repeat(voltage_diff_rows, 2),
                np.repeat(angle_diff_rows, 2),
            ),
            np.int64,
        )
        scalar_cols = self._concat_plan_arrays(
            (
                np.column_stack((self.voltage_col[voltage_diff_i], self.voltage_col[voltage_diff_j])).ravel()
                if voltage_diff_rows.size
                else np.array([], dtype=np.int64),
                np.column_stack((self.angle_col[angle_diff_i], self.angle_col[angle_diff_j])).ravel()
                if angle_diff_rows.size
                else np.array([], dtype=np.int64),
            ),
            np.int64,
        )
        scalar_values = self._concat_plan_arrays(
            (
                np.tile(np.array([1.0, -1.0], dtype=np.float64), voltage_diff_rows.size),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), angle_diff_rows.size),
            ),
            np.float64,
        )
        return {
            "handled_mask": handled_mask,
            "scalar_rows": scalar_rows,
            "scalar_cols": scalar_cols,
            "scalar_values": scalar_values,
            "voltage_rows": self._concat_plan_arrays((zero_plan["voltage_rows"], break_plan["voltage_rows"]), np.int64),
            "voltage_pos": self._concat_plan_arrays((zero_plan["voltage_pos"], break_plan["voltage_pos"]), np.int64),
            "angle_diff_rows": self._int_array(angle_diff_rows),
            "angle_diff_i": self._int_array(angle_diff_i),
            "angle_diff_j": self._int_array(angle_diff_j),
            "voltage_diff_rows": self._int_array(voltage_diff_rows),
            "voltage_diff_i": self._int_array(voltage_diff_i),
            "voltage_diff_j": self._int_array(voltage_diff_j),
            "power_rows": self._concat_plan_arrays((zero_plan["power_rows"], break_plan["power_rows"]), np.int64),
            "power_is_p": self._concat_plan_arrays((zero_plan["power_is_p"], break_plan["power_is_p"]), bool),
            "power_pos": self._concat_plan_arrays((zero_plan["power_pos"], break_plan["power_pos"]), np.int64),
            "power_current_idx": self._concat_plan_arrays(
                (zero_plan["power_current_idx"], break_plan["power_current_idx"]),
                np.int64,
            ),
            "power_sign": self._concat_plan_arrays((zero_plan["power_sign"], break_plan["power_sign"]), np.float64),
            "current_rows": self._concat_plan_arrays((zero_plan["current_rows"], break_plan["current_rows"]), np.int64),
            "current_idx": self._concat_plan_arrays((zero_plan["current_idx"], break_plan["current_idx"]), np.int64),
        }

    def _fill_zero_current_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACSwitch/ACZeroBranch P/Q/V/I measurements."""
        plan = self._zero_current_vector_plan(measurements)

        rows = plan["voltage_rows"]
        if rows.size:
            values[rows] = voltage[plan["voltage_pos"]]

        rows = plan["voltage_diff_rows"]
        if rows.size:
            values[rows] = voltage[plan["voltage_diff_i"]] - voltage[plan["voltage_diff_j"]]

        rows = plan["angle_diff_rows"]
        if rows.size:
            values[rows] = theta[plan["angle_diff_i"]] - theta[plan["angle_diff_j"]]

        rows = plan["power_rows"]
        if rows.size:
            pos = plan["power_pos"]
            current = plan["power_sign"] * switch_current[plan["power_current_idx"]]
            power = voltage_complex[pos] * np.conj(current)
            values[rows] = np.where(plan["power_is_p"], power.real, power.imag)

        rows = plan["current_rows"]
        if rows.size:
            values[rows] = np.abs(switch_current[plan["current_idx"]])

        return plan["handled_mask"]

    def _fill_zero_current_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACSwitch/ACZeroBranch Jacobian rows from explicit current states."""
        plan = self._zero_current_vector_plan(measurements)

        if plan["scalar_rows"].size:
            self._add_indexed_values(H, plan["scalar_rows"], plan["scalar_cols"], plan["scalar_values"])

        rows = plan["voltage_rows"]
        if rows.size:
            self._add_indexed_values(
                H,
                rows,
                self.voltage_col[plan["voltage_pos"]],
                np.ones(rows.size, dtype=np.float64),
            )

        rows = plan["power_rows"]
        if rows.size:
            pos = plan["power_pos"]
            current_idx = plan["power_current_idx"]
            sign = plan["power_sign"]
            current = switch_current[current_idx]
            power = voltage_complex[pos] * np.conj(sign * current)
            is_p = plan["power_is_p"]

            dtheta = 1j * power
            dvoltage = np.zeros(rows.size, dtype=np.complex128)
            valid_voltage = np.abs(voltage[pos]) > 1e-12
            dvoltage[valid_voltage] = power[valid_voltage] / voltage[pos][valid_voltage]
            dcurrent_re = sign * voltage_complex[pos]
            dcurrent_im = -1j * sign * voltage_complex[pos]

            self._add_indexed_values(H, rows, self.angle_col[pos], np.where(is_p, dtheta.real, dtheta.imag))
            self._add_indexed_values(H, rows, self.voltage_col[pos], np.where(is_p, dvoltage.real, dvoltage.imag))
            self._add_indexed_values(
                H,
                rows,
                self.base_switch_re + current_idx,
                np.where(is_p, dcurrent_re.real, dcurrent_re.imag),
            )
            self._add_indexed_values(
                H,
                rows,
                self.base_switch_im + current_idx,
                np.where(is_p, dcurrent_im.real, dcurrent_im.imag),
            )

        rows = plan["current_rows"]
        if rows.size:
            current = switch_current[plan["current_idx"]]
            current_abs = np.abs(current)
            valid_current = current_abs > 1e-12
            dcurrent_re = np.zeros(rows.size, dtype=np.float64)
            dcurrent_im = np.zeros(rows.size, dtype=np.float64)
            dcurrent_re[valid_current] = current.real[valid_current] / current_abs[valid_current]
            dcurrent_im[valid_current] = current.imag[valid_current] / current_abs[valid_current]
            current_idx = plan["current_idx"]
            self._add_indexed_values(H, rows, self.base_switch_re + current_idx, dcurrent_re)
            self._add_indexed_values(H, rows, self.base_switch_im + current_idx, dcurrent_im)

        return plan["handled_mask"]

    def _branch_power_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex]:
        """Analytical derivatives for S_own = V_own * conj(y_self*V_own + y_mutual*V_other)."""
        angle = theta[own] - theta[other]
        exp_angle = np.exp(1j * angle)
        y_self_conj = np.conj(y_self)
        y_mutual_conj = np.conj(y_mutual)
        off = y_mutual_conj * voltage[own] * voltage[other] * exp_angle
        dtheta_own = 1j * off
        dtheta_other = -1j * off
        dvoltage_own = 2.0 * y_self_conj * voltage[own] + y_mutual_conj * voltage[other] * exp_angle
        dvoltage_other = y_mutual_conj * voltage[own] * exp_angle
        return dtheta_own, dtheta_other, dvoltage_own, dvoltage_other

    def _branch_current_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex, complex]:
        """Analytical derivatives of branch terminal current in polar-voltage variables."""
        v_own = voltage[own] * np.exp(1j * theta[own])
        v_other = voltage[other] * np.exp(1j * theta[other])
        current = y_self * v_own + y_mutual * v_other
        dtheta_own = 1j * y_self * v_own
        dtheta_other = 1j * y_mutual * v_other
        dvoltage_own = y_self * np.exp(1j * theta[own])
        dvoltage_other = y_mutual * np.exp(1j * theta[other])
        return current, dtheta_own, dtheta_other, dvoltage_own, dvoltage_other

    def _add_current_magnitude_derivatives(
        self,
        H: np.ndarray,
        row: int,
        own: int,
        other: int,
        current: complex,
        dtheta_own: complex,
        dtheta_other: complex,
        dvoltage_own: complex,
        dvoltage_other: complex,
    ) -> None:
        current_abs = abs(current)
        if current_abs <= 1e-12:
            return

        def pick(value: complex) -> float:
            # d|I| = real(conj(I) * dI) / |I| projects complex sensitivities to magnitude.
            return float(np.real(np.conj(current) * value) / current_abs)

        own_angle_col = self.angle_col[own]
        if own_angle_col >= 0:
            self._add_scalar_value(H, row, own_angle_col, pick(dtheta_own))
        other_angle_col = self.angle_col[other]
        if other_angle_col >= 0:
            self._add_scalar_value(H, row, other_angle_col, pick(dtheta_other))
        self._add_scalar_value(H, row, self.voltage_col[own], pick(dvoltage_own))
        self._add_scalar_value(H, row, self.voltage_col[other], pick(dvoltage_other))

    def _add_power_derivatives(
        self,
        H: np.ndarray,
        row: int,
        meas_type: str,
        own: int,
        other: int,
        dtheta_own: complex,
        dtheta_other: complex,
        dvoltage_own: complex,
        dvoltage_other: complex,
    ) -> None:
        if meas_type.startswith("P"):
            pick = np.real
        elif meas_type.startswith("Q"):
            pick = np.imag
        else:
            raise RuntimeError(f"Unsupported power measurement type: {meas_type}")

        own_angle_col = self.angle_col[own]
        if own_angle_col >= 0:
            self._add_scalar_value(H, row, own_angle_col, float(pick(dtheta_own)))
        other_angle_col = self.angle_col[other]
        if other_angle_col >= 0:
            self._add_scalar_value(H, row, other_angle_col, float(pick(dtheta_other)))
        self._add_scalar_value(H, row, self.voltage_col[own], float(pick(dvoltage_own)))
        self._add_scalar_value(H, row, self.voltage_col[other], float(pick(dvoltage_other)))

    def _add_indexed_values(
        self,
        H: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        if hasattr(H, "add_many"):
            row_col_mask = (rows >= 0) & (cols >= 0)
            mask = row_col_mask if mask is None else (np.asarray(mask, dtype=bool) & row_col_mask)
            H.add_many(rows, cols, values, mask)
            return
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = (rows >= 0) & (cols >= 0)
        else:
            mask = np.asarray(mask, dtype=bool) & (rows >= 0) & (cols >= 0)
        if np.any(mask):
            np.add.at(H, (rows[mask], cols[mask]), values[mask])

    def _add_indexed_value_blocks(self, H: np.ndarray, blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """Append several vectorized Jacobian write blocks with one sparse/dense scatter."""
        if not blocks:
            return
        if hasattr(H, "add_many"):
            for rows, cols, values in blocks:
                self._add_indexed_values(H, rows, cols, values)
            return
        row_parts = []
        col_parts = []
        value_parts = []
        for rows, cols, values in blocks:
            rows = np.asarray(rows)
            if rows.size == 0:
                continue
            row_parts.append(rows)
            col_parts.append(np.asarray(cols))
            value_parts.append(np.asarray(values))
        if not row_parts:
            return
        if len(row_parts) == 1:
            self._add_indexed_values(H, row_parts[0], col_parts[0], value_parts[0])
            return
        self._add_indexed_values(
            H,
            np.concatenate(row_parts),
            np.concatenate(col_parts),
            np.concatenate(value_parts),
        )

    @staticmethod
    def _complex_bincount(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
        if index.size == 0:
            return np.zeros(size, dtype=np.complex128)
        return (
            np.bincount(index, weights=values.real, minlength=size)
            + 1j * np.bincount(index, weights=values.imag, minlength=size)
        )

    @staticmethod
    def _add_scalar_value(H, row: int, col: int, value: float) -> None:
        if col < 0:
            return
        if hasattr(H, "add"):
            H.add(row, col, value)
        else:
            H[row, col] += value

    def _fill_branch_transformer_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACBranch/ACTransformer P/Q/V/I rows to reduce Python call overhead."""
        plan = self._branch_transformer_vector_plan(measurements)
        voltage_rows = plan["voltage_rows"]
        if voltage_rows.size:
            self._add_indexed_values(
                H,
                voltage_rows,
                plan["voltage_cols"],
                np.ones(voltage_rows.size, dtype=np.float64),
            )

        rows = plan["power_rows"]
        if rows.size:
            own = plan["power_own"]
            other = plan["power_other"]
            y_self_conj = np.conj(plan["power_y_self"])
            y_mutual_conj = np.conj(plan["power_y_mutual"])
            off = y_mutual_conj * voltage_complex[own] * np.conj(voltage_complex[other])
            dtheta_own = 1j * off
            dtheta_other = -1j * off
            dvoltage_own = 2.0 * y_self_conj * voltage[own] + off / voltage[own]
            dvoltage_other = off / voltage[other]
            is_p = plan["power_is_p"]

            own_angle_cols = self.angle_col[own]
            other_angle_cols = self.angle_col[other]
            own_voltage_cols = self.voltage_col[own]
            other_voltage_cols = self.voltage_col[other]
            values = (
                np.where(is_p, dtheta_own.real, dtheta_own.imag),
                np.where(is_p, dtheta_other.real, dtheta_other.imag),
                np.where(is_p, dvoltage_own.real, dvoltage_own.imag),
                np.where(is_p, dvoltage_other.real, dvoltage_other.imag),
            )
            if hasattr(H, "add_many"):
                H.add_many(rows, own_angle_cols, values[0])
                H.add_many(rows, other_angle_cols, values[1])
                H.add_many(rows, own_voltage_cols, values[2])
                H.add_many(rows, other_voltage_cols, values[3])
            else:
                self._add_indexed_values(H, rows, own_angle_cols, values[0])
                self._add_indexed_values(H, rows, other_angle_cols, values[1])
                self._add_indexed_values(H, rows, own_voltage_cols, values[2])
                self._add_indexed_values(H, rows, other_voltage_cols, values[3])

        rows = plan["current_rows"]
        if rows.size:
            own = plan["current_own"]
            other = plan["current_other"]
            y_self = plan["current_y_self"]
            y_mutual = plan["current_y_mutual"]
            v_own = voltage_complex[own]
            v_other = voltage_complex[other]
            exp_own = v_own / voltage[own]
            exp_other = v_other / voltage[other]
            current = y_self * v_own + y_mutual * v_other
            current_abs = np.abs(current)
            valid = current_abs > 1e-12
            current_conj = np.conj(current)

            def project(value: np.ndarray) -> np.ndarray:
                out = np.zeros_like(current_abs, dtype=np.float64)
                out[valid] = (current_conj[valid] * value[valid]).real / current_abs[valid]
                return out

            dtheta_own = 1j * y_self * v_own
            dtheta_other = 1j * y_mutual * v_other
            dvoltage_own = y_self * exp_own
            dvoltage_other = y_mutual * exp_other
            own_angle_cols = self.angle_col[own]
            other_angle_cols = self.angle_col[other]
            own_voltage_cols = self.voltage_col[own]
            other_voltage_cols = self.voltage_col[other]
            values = (
                project(dtheta_own),
                project(dtheta_other),
                project(dvoltage_own),
                project(dvoltage_other),
            )
            if hasattr(H, "add_many"):
                H.add_many(rows, own_angle_cols, values[0])
                H.add_many(rows, other_angle_cols, values[1])
                H.add_many(rows, own_voltage_cols, values[2])
                H.add_many(rows, other_voltage_cols, values[3])
            else:
                self._add_indexed_values(H, rows, own_angle_cols, values[0])
                self._add_indexed_values(H, rows, other_angle_cols, values[1])
                self._add_indexed_values(H, rows, own_voltage_cols, values[2])
                self._add_indexed_values(H, rows, other_voltage_cols, values[3])

        return plan["handled_mask"]

    def _simple_jacobian_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_simple_jacobian_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_simple_jacobian_plan(measurements)
            self._active_simple_jacobian_plan = plan
            return plan
        key = id(measurements)
        cached = self._simple_jacobian_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_simple_jacobian_plan(measurements)
        if len(self._simple_jacobian_plan_cache) > 16:
            self._simple_jacobian_plan_cache.clear()
        self._simple_jacobian_plan_cache[key] = (measurements, plan)
        return plan

    def _build_simple_jacobian_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_simple_plan_kind_by_type_code,
        )
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        device_pos = plan_table.device_pos
        handled = np.asarray(plan_table.handled, dtype=bool).copy()

        node_rows = row[(code == _DEVICE_TYPE_CODES["ACNode"]) & handled]
        node_pos = self._ac_node_plan_pos[device_pos[node_rows]]
        node_kind = kind[node_rows]
        node_v = node_kind == _AC_NODE_MEASUREMENT_KIND["V"]
        node_angle = node_kind == _AC_NODE_MEASUREMENT_KIND["ANGLE"]

        gen_rows = row[(code == _DEVICE_TYPE_CODES["ACGenerator"]) & handled]
        gen_pos = self._ac_generator_plan_node_pos[device_pos[gen_rows]]

        load_rows_all = row[(code == _DEVICE_TYPE_CODES["ACLoad"]) & handled]
        load_plan_pos = device_pos[load_rows_all]
        load_node_pos = self._ac_load_plan_node_pos[load_plan_pos]
        load_kind_all = kind[load_rows_all]
        load_v = load_kind_all == _AC_LOAD_MEASUREMENT_KIND["V_LOAD"]
        load_state = load_kind_all != _AC_LOAD_MEASUREMENT_KIND["V_LOAD"]

        def constraint_rows_for(device_code, i_array, j_array):
            rows = row[(code == device_code) & handled]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_diff = row_kind == _AC_CONSTRAINT_MEASUREMENT_KIND["V_DIFF"]
            angle_diff = row_kind == _AC_CONSTRAINT_MEASUREMENT_KIND["ANGLE_DIFF"]
            return rows[v_diff], i[v_diff], j[v_diff], rows[angle_diff], i[angle_diff], j[angle_diff]

        zero_v_rows, zero_v_i, zero_v_j, zero_a_rows, zero_a_i, zero_a_j = constraint_rows_for(
            _DEVICE_TYPE_CODES["ACZeroBranchConstraint"],
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
        )
        break_v_rows, break_v_i, break_v_j, break_a_rows, break_a_i, break_a_j = constraint_rows_for(
            _DEVICE_TYPE_CODES["ACBreakConstraint"],
            self._ac_break_plan_i,
            self._ac_break_plan_j,
        )
        value_voltage_rows = self._concat_plan_arrays((node_rows[node_v], gen_rows, load_rows_all[load_v]), np.int64)
        value_voltage_pos = self._concat_plan_arrays((node_pos[node_v], gen_pos, load_node_pos[load_v]), np.int64)
        value_angle_rows = self._int_array(node_rows[node_angle])
        value_angle_pos = self._int_array(node_pos[node_angle])
        value_voltage_diff_rows = self._concat_plan_arrays((zero_v_rows, break_v_rows), np.int64)
        value_voltage_diff_i = self._concat_plan_arrays((zero_v_i, break_v_i), np.int64)
        value_voltage_diff_j = self._concat_plan_arrays((zero_v_j, break_v_j), np.int64)
        value_angle_diff_rows = self._concat_plan_arrays((zero_a_rows, break_a_rows), np.int64)
        value_angle_diff_i = self._concat_plan_arrays((zero_a_i, break_a_i), np.int64)
        value_angle_diff_j = self._concat_plan_arrays((zero_a_j, break_a_j), np.int64)

        scalar_rows = self._concat_plan_arrays(
            (
                value_voltage_rows,
                value_angle_rows,
                np.repeat(value_voltage_diff_rows, 2),
                np.repeat(value_angle_diff_rows, 2),
            ),
            np.int64,
        )
        scalar_cols = self._concat_plan_arrays(
            (
                self.voltage_col[value_voltage_pos],
                self.angle_col[value_angle_pos],
                np.column_stack((self.voltage_col[value_voltage_diff_i], self.voltage_col[value_voltage_diff_j])).ravel()
                if value_voltage_diff_rows.size
                else np.array([], dtype=np.int64),
                np.column_stack((self.angle_col[value_angle_diff_i], self.angle_col[value_angle_diff_j])).ravel()
                if value_angle_diff_rows.size
                else np.array([], dtype=np.int64),
            ),
            np.int64,
        )
        scalar_values = self._concat_plan_arrays(
            (
                np.ones(value_voltage_rows.size, dtype=np.float64),
                np.ones(value_angle_rows.size, dtype=np.float64),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), value_voltage_diff_rows.size),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), value_angle_diff_rows.size),
            ),
            np.float64,
        )
        load_rows = load_rows_all[load_state]
        load_device_pos = load_plan_pos[load_state]
        return {
            "handled_mask": handled,
            "scalar_rows": scalar_rows,
            "scalar_cols": scalar_cols,
            "scalar_values": scalar_values,
            "value_voltage_rows": value_voltage_rows,
            "value_voltage_pos": value_voltage_pos,
            "value_angle_rows": value_angle_rows,
            "value_angle_pos": value_angle_pos,
            "value_voltage_diff_rows": value_voltage_diff_rows,
            "value_voltage_diff_i": value_voltage_diff_i,
            "value_voltage_diff_j": value_voltage_diff_j,
            "value_angle_diff_rows": value_angle_diff_rows,
            "value_angle_diff_i": value_angle_diff_i,
            "value_angle_diff_j": value_angle_diff_j,
            "load_rows": self._int_array(load_rows),
            "load_pos": self._ac_load_plan_node_pos[load_device_pos],
            "load_index": self._ac_load_plan_index[load_device_pos],
            "load_kind": self._int_array(load_kind_all[load_state]),
        }

    def _fill_simple_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate node, load, generator-voltage and pseudo-constraint rows."""
        plan = self._simple_jacobian_plan(measurements)

        rows = plan["value_voltage_rows"]
        if rows.size:
            values[rows] = voltage[plan["value_voltage_pos"]]

        rows = plan["value_angle_rows"]
        if rows.size:
            values[rows] = theta[plan["value_angle_pos"]]

        rows = plan["value_voltage_diff_rows"]
        if rows.size:
            values[rows] = voltage[plan["value_voltage_diff_i"]] - voltage[plan["value_voltage_diff_j"]]

        rows = plan["value_angle_diff_rows"]
        if rows.size:
            values[rows] = theta[plan["value_angle_diff_i"]] - theta[plan["value_angle_diff_j"]]

        rows = plan["load_rows"]
        if rows.size:
            pos = plan["load_pos"]
            load_idx = plan["load_index"]
            vm = voltage[pos]
            kind = plan["load_kind"]
            p = np.asarray(x[self.base_load_p + load_idx], dtype=np.float64)
            q = np.asarray(x[self.base_load_q + load_idx], dtype=np.float64)
            row_values = np.zeros(rows.size, dtype=np.float64)
            p_mask = kind == 0
            q_mask = kind == 1
            i_mask = kind == 2
            row_values[p_mask] = p[p_mask]
            row_values[q_mask] = q[q_mask]
            if np.any(i_mask):
                i_idx = np.flatnonzero(i_mask)
                vm_i = vm[i_idx]
                valid = np.abs(vm_i) > self.min_current_voltage
                s_abs = np.hypot(p[i_idx], q[i_idx])
                row_values[i_idx[valid]] = s_abs[valid] / vm_i[valid]
            values[rows] = row_values

        return plan["handled_mask"]

    def _fill_simple_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill node/load/generator-voltage and pseudo constraint Jacobian rows."""
        plan = self._simple_jacobian_plan(measurements)
        if plan["scalar_rows"].size:
            self._add_indexed_values(H, plan["scalar_rows"], plan["scalar_cols"], plan["scalar_values"])

        rows = plan["load_rows"]
        if rows.size:
            pos = plan["load_pos"]
            load_idx = plan["load_index"]
            vm = voltage[pos]
            kind = plan["load_kind"]
            p = np.asarray(x[self.base_load_p + load_idx], dtype=np.float64)
            q = np.asarray(x[self.base_load_q + load_idx], dtype=np.float64)
            p_mask = kind == 0
            q_mask = kind == 1
            i_mask = kind == 2
            if np.any(p_mask):
                self._add_indexed_values(
                    H,
                    rows[p_mask],
                    self.base_load_p + load_idx[p_mask],
                    np.ones(np.count_nonzero(p_mask), dtype=np.float64),
                )
            if np.any(q_mask):
                self._add_indexed_values(
                    H,
                    rows[q_mask],
                    self.base_load_q + load_idx[q_mask],
                    np.ones(np.count_nonzero(q_mask), dtype=np.float64),
                )
            if np.any(i_mask):
                i_rows = rows[i_mask]
                i_idx = load_idx[i_mask]
                p_i = p[i_mask]
                q_i = q[i_mask]
                vm_i = vm[i_mask]
                s_abs = np.hypot(p_i, q_i)
                valid = (s_abs > 1e-12) & (np.abs(vm_i) > self.min_current_voltage)
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.base_load_p + i_idx,
                    np.divide(p_i, s_abs * vm_i, out=np.zeros_like(p_i), where=valid),
                    valid,
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.base_load_q + i_idx,
                    np.divide(q_i, s_abs * vm_i, out=np.zeros_like(q_i), where=valid),
                    valid,
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.voltage_col[pos[i_mask]],
                    np.divide(-s_abs, vm_i * vm_i, out=np.zeros_like(s_abs), where=valid),
                    valid,
                )

        return plan["handled_mask"]

    def _balance_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_balance_measurement_plan", None)
            if active_plan is not None:
                return active_plan
            return self._build_balance_measurement_plan(self.active_measurements)
        key = id(measurements)
        cached = self._balance_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_balance_measurement_plan(measurements)
        if len(self._balance_measurement_plan_cache) > 16:
            self._balance_measurement_plan_cache.clear()
        self._balance_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _balance_y_arrays(self, pos: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_array = np.asarray(pos, dtype=np.int64)
        if pos_array.size == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.complex128),
            )
        counts = np.fromiter(
            (int(self._y_row_nodes[int(node_pos)].size) for node_pos in pos_array),
            dtype=np.int64,
            count=pos_array.size,
        )
        total = int(counts.sum())
        if total == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.complex128),
            )
        y_balance = np.repeat(np.arange(pos_array.size, dtype=np.int32), counts)
        y_nodes = np.empty(total, dtype=np.int32)
        y_conj = np.empty(total, dtype=np.complex128)
        offset = 0
        for node_pos, count in zip(pos_array, counts):
            count = int(count)
            if count == 0:
                continue
            end = offset + count
            y_nodes[offset:end] = self._y_row_nodes[int(node_pos)].astype(np.int32, copy=False)
            y_conj[offset:end] = self._y_row_y_conj[int(node_pos)].astype(np.complex128, copy=False)
            offset = end
        return y_balance, y_nodes, y_nodes.astype(np.int64, copy=False), y_conj

    def _build_balance_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Build the balance plan from a measurement sequence or table-backed active view."""
        self._ensure_measurement_plan_lookup_arrays()
        plan_table = self._measurement_plan_table(
            measurements,
            getattr(
                self,
                "_ac_balance_plan_kind_by_type_code",
                {_DEVICE_TYPE_CODES["ACPowerBalance"]: _AC_BALANCE_MEASUREMENT_KIND},
            ),
            device_pos_by_type_code={
                _DEVICE_TYPE_CODES["ACPowerBalance"]: getattr(self, "_ac_node_plan_name_to_pos", {}),
            },
        )
        rows = plan_table.row[
            (plan_table.device_type_code == _DEVICE_TYPE_CODES["ACPowerBalance"]) & plan_table.handled
        ]
        device_pos = plan_table.device_pos[rows]
        pos = self._ac_node_plan_pos[device_pos]
        kind = plan_table.meas_kind[rows]
        p_row_by_pos = np.empty(len(self.nodes), dtype=np.int32)
        q_row_by_pos = np.empty(len(self.nodes), dtype=np.int32)
        p_row_by_pos.fill(-1)
        q_row_by_pos.fill(-1)
        if rows.size:
            p_mask = kind == _AC_BALANCE_MEASUREMENT_KIND["P_BALANCE"]
            q_mask = kind == _AC_BALANCE_MEASUREMENT_KIND["Q_BALANCE"]
            p_row_by_pos[pos[p_mask]] = rows[p_mask].astype(np.int32, copy=False)
            q_row_by_pos[pos[q_mask]] = rows[q_mask].astype(np.int32, copy=False)

        y_balance, y_nodes, y_nodes_i64, y_conj = self._balance_y_arrays(pos)
        plan = {
            "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
            "rows": self._int_array(rows),
            "pos": self._int_array(pos),
            "pos_i64": np.asarray(pos, dtype=np.int64),
            "kind": self._int_array(kind),
            "p_row_by_pos": p_row_by_pos,
            "q_row_by_pos": q_row_by_pos,
            "y_balance": y_balance,
            "y_nodes": y_nodes,
            "y_nodes_i64": y_nodes_i64,
            "y_conj": y_conj,
        }
        if measurements is self.active_measurements:
            self._active_balance_measurement_plan = plan
        return plan

    def _fill_balance_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate nodal P/Q power-balance mismatch rows."""
        plan = self._balance_measurement_plan(measurements)
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]
        p_balance, q_balance = self._power_balance_totals(x, voltage_complex, switch_current)
        pos = plan["pos"]
        values[rows] = np.where(plan["kind"] == 0, p_balance[pos], q_balance[pos])
        return plan["handled_mask"]

    def _fill_balance_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse nodal balance derivatives for network, switch and P/Q states."""
        plan = self._balance_measurement_plan(measurements)
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        pos = plan["pos_i64"]
        kind = plan["kind"]

        y_balance = plan["y_balance"]
        if y_balance.size:
            y_nodes = plan["y_nodes_i64"]
            y_conj = plan["y_conj"]
            y_pos = pos[y_balance]
            exp_delta = np.exp(1j * (theta[y_pos] - theta[y_nodes]))
            term = y_conj * voltage[y_pos] * voltage[y_nodes] * exp_delta

            off_mask = y_nodes != y_pos
            off_sum = np.zeros(pos.size, dtype=np.complex128)
            if np.any(off_mask):
                off_balance = y_balance[off_mask]
                off_nodes = y_nodes[off_mask]
                off_term = term[off_mask]
                off_sum = self._complex_bincount(off_balance, off_term, pos.size)

                theta_values = -1j * off_term
                theta_rows = rows[off_balance]
                theta_cols = self.angle_col[off_nodes]
                theta_kind = kind[off_balance]

                voltage_values = y_conj[off_mask] * voltage[y_pos[off_mask]] * exp_delta[off_mask]
                voltage_cols = self.voltage_col[off_nodes]
                self._add_indexed_value_blocks(
                    H,
                    (
                        (
                            theta_rows,
                            theta_cols,
                            np.where(theta_kind == 0, theta_values.real, theta_values.imag),
                        ),
                        (
                            theta_rows,
                            voltage_cols,
                            np.where(theta_kind == 0, voltage_values.real, voltage_values.imag),
                        ),
                    ),
                )

            sum_all = self._complex_bincount(y_balance, y_conj * voltage[y_nodes] * exp_delta, pos.size)
        else:
            off_sum = np.zeros(pos.size, dtype=np.complex128)
            sum_all = np.zeros(pos.size, dtype=np.complex128)

        own_theta_values = 1j * off_sum
        own_voltage_values = self._y_row_diag_conj[pos] * voltage[pos] + sum_all
        self._add_indexed_value_blocks(
            H,
            (
                (
                    rows,
                    self.angle_col[pos],
                    np.where(kind == 0, own_theta_values.real, own_theta_values.imag),
                ),
                (
                    rows,
                    self.voltage_col[pos],
                    np.where(kind == 0, own_voltage_values.real, own_voltage_values.imag),
                ),
            ),
        )

        p_row_by_pos = plan["p_row_by_pos"]
        q_row_by_pos = plan["q_row_by_pos"]
        if self.n_switch_current:
            end_pos = self.switch_balance_end_pos
            end_current_idx = self.switch_balance_end_current_idx
            sign = self.switch_balance_sign
            current = switch_current[end_current_idx] * sign
            s = voltage_complex[end_pos] * np.conj(current)
            dS_dtheta = 1j * s
            dS_dV = np.divide(s, voltage[end_pos], out=np.zeros_like(s), where=np.abs(voltage[end_pos]) > 1e-12)
            dS_dIr = sign * voltage_complex[end_pos]
            dS_dIi = -1j * sign * voltage_complex[end_pos]
            p_rows = p_row_by_pos[end_pos]
            q_rows = q_row_by_pos[end_pos]

            self._add_indexed_value_blocks(
                H,
                (
                    (p_rows, self.angle_col[end_pos], dS_dtheta.real),
                    (q_rows, self.angle_col[end_pos], dS_dtheta.imag),
                    (p_rows, self.voltage_col[end_pos], dS_dV.real),
                    (q_rows, self.voltage_col[end_pos], dS_dV.imag),
                    (p_rows, self.base_switch_re + end_current_idx, dS_dIr.real),
                    (q_rows, self.base_switch_re + end_current_idx, dS_dIr.imag),
                    (p_rows, self.base_switch_im + end_current_idx, dS_dIi.real),
                    (q_rows, self.base_switch_im + end_current_idx, dS_dIi.imag),
                ),
            )

        if self.n_generator_power:
            gen_idx = self.generator_power_idx_array
            gen_p_rows = p_row_by_pos[self.generator_pos_array]
            gen_q_rows = q_row_by_pos[self.generator_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (gen_p_rows, self.base_gen_p + gen_idx, self.generator_balance_minus_ones),
                    (gen_q_rows, self.base_gen_q + gen_idx, self.generator_balance_minus_ones),
                ),
            )

        if self.n_load_power:
            load_idx = self.load_power_idx_array
            load_p_rows = p_row_by_pos[self.load_pos_array]
            load_q_rows = q_row_by_pos[self.load_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (load_p_rows, self.base_load_p + load_idx, self.load_balance_ones),
                    (load_q_rows, self.base_load_q + load_idx, self.load_balance_ones),
                ),
            )

        if self.n_shunt_q:
            shunt_q_rows = q_row_by_pos[self.shunt_q_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (
                        shunt_q_rows,
                        self.base_shunt_q + self.shunt_q_idx_array,
                        self.shunt_balance_minus_ones,
                    ),
                ),
            )

        return plan["handled_mask"]

    def _add_zero_current_measurement_derivatives(
        self,
        H: np.ndarray,
        row: int,
        meas_type: str,
        device,
        current: complex,
        current_idx: int,
        voltage_complex: np.ndarray,
        voltage: np.ndarray,
    ) -> None:
        """Fill H entries for switches/zero branches whose current is an explicit state."""
        if meas_type.endswith("_FROM"):
            pos = self.node_pos[device.i_node]
            sign = 1.0
        elif meas_type.endswith("_TO"):
            pos = self.node_pos[device.j_node]
            sign = -1.0
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

        if meas_type.startswith("P") or meas_type.startswith("Q"):
            signed_current = sign * current
            power = voltage_complex[pos] * np.conj(signed_current)
            pick = np.real if meas_type.startswith("P") else np.imag
            angle_col = self.angle_col[pos]
            dS_dtheta = 1j * power
            dS_dV = power / voltage[pos] if abs(voltage[pos]) > 1e-12 else 0.0
            dS_dIr = sign * voltage_complex[pos]
            dS_dIi = -1j * sign * voltage_complex[pos]
            if angle_col >= 0:
                self._add_scalar_value(H, row, angle_col, float(pick(dS_dtheta)))
            self._add_scalar_value(H, row, self.voltage_col[pos], float(pick(dS_dV)))
            self._add_scalar_value(H, row, self.base_switch_re + current_idx, float(pick(dS_dIr)))
            self._add_scalar_value(H, row, self.base_switch_im + current_idx, float(pick(dS_dIi)))
        elif meas_type.startswith("I"):
            current_abs = abs(current)
            if current_abs > 1e-12:
                self._add_scalar_value(H, row, self.base_switch_re + current_idx, current.real / current_abs)
                self._add_scalar_value(H, row, self.base_switch_im + current_idx, current.imag / current_abs)
        elif meas_type.startswith("V"):
            self._add_scalar_value(H, row, self.voltage_col[pos], 1.0)
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

    def _network_power_derivatives(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized derivatives of node network injection S = V * conj(YV)."""
        y_matrix = self.Y.toarray() if issparse(self.Y) else self.Y
        y_conj = np.conj(y_matrix)
        exp_delta = np.exp(1j * (theta[:, None] - theta[None, :]))
        term = y_conj * (voltage[:, None] * voltage[None, :]) * exp_delta
        off = term.copy()
        np.fill_diagonal(off, 0.0)

        dS_dtheta = -1j * off
        np.fill_diagonal(dS_dtheta, 1j * np.sum(off, axis=1))

        dS_dvoltage = y_conj * voltage[:, None] * exp_delta
        y_diag = np.diag(y_conj)
        d_self_from_off = np.sum(y_conj * voltage[None, :] * exp_delta, axis=1) - y_diag * voltage
        np.fill_diagonal(dS_dvoltage, 2.0 * y_diag * voltage + d_self_from_off)
        return dS_dtheta, dS_dvoltage

    def _network_power_derivative_rows(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
        rows: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute network power derivative rows needed by generator measurements."""
        rows = np.asarray(rows, dtype=np.int64)
        dS_dtheta = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.complex128)
        dS_dvoltage = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.complex128)
        if rows.size == 0:
            return dS_dtheta, dS_dvoltage

        y_rows = self.Y[rows, :].toarray() if issparse(self.Y) else self.Y[rows, :]
        y_conj_rows = np.conj(y_rows)
        exp_delta = np.exp(1j * (theta[rows, None] - theta[None, :]))
        term = y_conj_rows * (voltage[rows, None] * voltage[None, :]) * exp_delta
        off = term.copy()
        local = np.arange(rows.size)
        off[local, rows] = 0.0

        theta_rows = -1j * off
        theta_rows[local, rows] = 1j * np.sum(off, axis=1)

        voltage_rows = y_conj_rows * voltage[rows, None] * exp_delta
        y_diag_rows = np.conj(self.Y.diagonal()[rows] if issparse(self.Y) else self.Y[rows, rows])
        d_self_from_off = np.sum(y_conj_rows * voltage[None, :] * exp_delta, axis=1) - y_diag_rows * voltage[rows]
        voltage_rows[local, rows] = 2.0 * y_diag_rows * voltage[rows] + d_self_from_off

        dS_dtheta[rows, :] = theta_rows
        dS_dvoltage[rows, :] = voltage_rows
        return dS_dtheta, dS_dvoltage

    def _load_power_derivatives(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_derivative = np.zeros(len(self.nodes), dtype=np.float64)
        q_derivative = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size == 0:
            return p_derivative, q_derivative
        vm = voltage[self.load_pos_array]
        np.add.at(p_derivative, self.load_pos_array, self.load_pv1_array + 2.0 * self.load_pv2_array * vm)
        np.add.at(q_derivative, self.load_pos_array, self.load_qv1_array + 2.0 * self.load_qv2_array * vm)
        return p_derivative, q_derivative

    def _add_switch_injection_derivatives(
        self,
        dP: np.ndarray,
        dQ: np.ndarray,
        node_pos: int,
        switch_current: np.ndarray,
        voltage_complex: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
        share: float,
    ) -> None:
        """Add zero-branch current sensitivities into a generator injection derivative."""
        for idx, (_, dev) in enumerate(self.zero_current_devices):
            i = self.node_pos[dev.i_node]
            j = self.node_pos[dev.j_node]
            if node_pos != i and node_pos != j:
                continue

            current = switch_current[idx]
            if node_pos == i:
                s = voltage_complex[i] * np.conj(current)
                dS_dIr = voltage_complex[i]
                dS_dIi = -1j * voltage_complex[i]
            else:
                s = voltage_complex[j] * np.conj(-current)
                dS_dIr = -voltage_complex[j]
                dS_dIi = 1j * voltage_complex[j]

            angle_col = self.angle_col[node_pos]
            dS_dtheta = 1j * s
            dS_dV = s / voltage[node_pos] if abs(voltage[node_pos]) > 1e-12 else 0.0
            if angle_col >= 0:
                dP[angle_col] += share * dS_dtheta.real
                dQ[angle_col] += share * dS_dtheta.imag
            voltage_col = int(self.voltage_col[node_pos])
            if voltage_col >= 0:
                dP[voltage_col] += share * np.real(dS_dV)
                dQ[voltage_col] += share * np.imag(dS_dV)
            dP[self.base_switch_re + idx] += share * dS_dIr.real
            dQ[self.base_switch_re + idx] += share * dS_dIr.imag
            dP[self.base_switch_im + idx] += share * dS_dIi.real
            dQ[self.base_switch_im + idx] += share * dS_dIi.imag

    def _current_from_power_derivatives(
        self,
        p: float,
        q: float,
        voltage: float,
        dP: np.ndarray,
        dQ: np.ndarray,
        dV_col: int,
    ) -> np.ndarray:
        dI = np.zeros_like(dP)
        s_abs = float(np.hypot(p, q))
        if abs(voltage) <= self.min_current_voltage or s_abs <= 1e-12:
            return dI
        dI += (p * dP + q * dQ) / (s_abs * voltage)
        if dV_col >= 0:
            dI[dV_col] -= s_abs / (voltage * voltage)
        return dI

    def _network_power_derivative_entries(
        self,
        pos: int,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse dS/dtheta and dS/dV entries for one network injection row."""
        if issparse(self.Y):
            start, end = self.Y.indptr[pos], self.Y.indptr[pos + 1]
            node_cols = self.Y.indices[start:end]
            y_values = self.Y.data[start:end]
        else:
            row = self.Y[pos, :]
            node_cols = np.nonzero(row)[0]
            y_values = row[node_cols]

        if node_cols.size:
            y_conj = np.conj(y_values)
            exp_delta = np.exp(1j * (theta[pos] - theta[node_cols]))
            term = y_conj * voltage[pos] * voltage[node_cols] * exp_delta
            off_mask = node_cols != pos
            off_nodes = node_cols[off_mask]
            off_term = term[off_mask]
            theta_nodes = np.concatenate((off_nodes, np.asarray([pos], dtype=np.int32)))
            theta_values = np.concatenate((-1j * off_term, np.asarray([1j * np.sum(off_term)], dtype=np.complex128)))

            y_diag_values = y_values[node_cols == pos]
            y_diag_conj = np.conj(y_diag_values[0]) if y_diag_values.size else 0.0
            sum_all = np.sum(y_conj * voltage[node_cols] * exp_delta)
            off_voltage_values = y_conj[off_mask] * voltage[pos] * exp_delta[off_mask]
            diag_voltage = y_diag_conj * voltage[pos] + sum_all
            voltage_nodes = np.concatenate((off_nodes, np.asarray([pos], dtype=np.int32)))
            voltage_values = np.concatenate((off_voltage_values, np.asarray([diag_voltage], dtype=np.complex128)))
        else:
            theta_nodes = np.asarray([pos], dtype=np.int32)
            theta_values = np.asarray([0.0j], dtype=np.complex128)
            voltage_nodes = np.asarray([pos], dtype=np.int32)
            voltage_values = np.asarray([0.0j], dtype=np.complex128)

        return theta_nodes, theta_values, voltage_nodes, voltage_values

    def _cached_network_power_derivative_entries(
        self,
        pos: int,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return dS/dtheta and dS/dV entries using cached Y-row sparsity."""
        node_cols = self._y_row_nodes[pos]
        if node_cols.size == 0:
            theta_nodes = np.asarray([pos], dtype=np.int32)
            zero = np.asarray([0.0j], dtype=np.complex128)
            return theta_nodes, zero, theta_nodes, zero

        y_conj = self._y_row_y_conj[pos]
        off_mask = self._y_row_off_mask[pos]
        off_nodes = self._y_row_off_nodes[pos]
        if off_mask is None:
            off_mask = node_cols != pos
            off_nodes = node_cols[off_mask]
            self._y_row_off_mask[pos] = off_mask
            self._y_row_off_nodes[pos] = off_nodes
        exp_delta = np.exp(1j * (theta[pos] - theta[node_cols]))
        term = y_conj * voltage[pos] * voltage[node_cols] * exp_delta
        off_term = term[off_mask]

        theta_nodes = np.empty(off_nodes.size + 1, dtype=np.int32)
        theta_nodes[:-1] = off_nodes
        theta_nodes[-1] = pos
        theta_values = np.empty(theta_nodes.size, dtype=np.complex128)
        theta_values[:-1] = -1j * off_term
        theta_values[-1] = 1j * np.sum(off_term)

        voltage_nodes = theta_nodes.copy()
        voltage_values = np.empty(voltage_nodes.size, dtype=np.complex128)
        voltage_values[:-1] = y_conj[off_mask] * voltage[pos] * exp_delta[off_mask]
        voltage_values[-1] = self._y_row_diag_conj[pos] * voltage[pos] + np.sum(
            y_conj * voltage[node_cols] * exp_delta
        )
        return theta_nodes, theta_values, voltage_nodes, voltage_values

    def _generator_derivative_entries(
        self,
        gen,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sparse dP/dx and dQ/dx entries for generator power measurements."""
        pos = self.node_pos[gen.node]
        share = self.generator_share_by_name[gen.name]
        cols = []
        dP = []
        dQ = []

        theta_nodes, theta_values, voltage_nodes, voltage_values = self._cached_network_power_derivative_entries(
            pos,
            theta,
            voltage,
        )
        angle_cols = self.angle_col[theta_nodes]
        angle_mask = angle_cols >= 0
        if np.any(angle_mask):
            cols.extend(angle_cols[angle_mask].tolist())
            dP.extend((share * theta_values[angle_mask].real).tolist())
            dQ.extend((share * theta_values[angle_mask].imag).tolist())

        voltage_cols = self.voltage_col[voltage_nodes]
        voltage_mask = voltage_cols >= 0
        if np.any(voltage_mask):
            cols.extend(voltage_cols[voltage_mask].tolist())
            dP.extend((share * voltage_values[voltage_mask].real).tolist())
            dQ.extend((share * voltage_values[voltage_mask].imag).tolist())

        own_voltage_col = int(self.voltage_col[pos])
        if own_voltage_col >= 0:
            cols.append(own_voltage_col)
            dP.append(float(share * p_load_dv[pos]))
            dQ.append(float(share * q_load_dv[pos]))

        for idx, from_i in self.zero_current_by_node.get(pos, []):
            current = switch_current[idx]
            if from_i:
                s = voltage_complex[pos] * np.conj(current)
                dS_dIr = voltage_complex[pos]
                dS_dIi = -1j * voltage_complex[pos]
            else:
                s = voltage_complex[pos] * np.conj(-current)
                dS_dIr = -voltage_complex[pos]
                dS_dIi = 1j * voltage_complex[pos]
            dS_dtheta = 1j * s
            dS_dV = s / voltage[pos] if abs(voltage[pos]) > 1e-12 else 0.0
            angle_col = int(self.angle_col[pos])
            if angle_col >= 0:
                cols.append(angle_col)
                dP.append(float(share * dS_dtheta.real))
                dQ.append(float(share * dS_dtheta.imag))
            if own_voltage_col >= 0:
                cols.append(own_voltage_col)
                dP.append(float(share * np.real(dS_dV)))
                dQ.append(float(share * np.imag(dS_dV)))
            cols.extend((self.base_switch_re + idx, self.base_switch_im + idx))
            dP.extend((float(share * dS_dIr.real), float(share * dS_dIi.real)))
            dQ.extend((float(share * dS_dIr.imag), float(share * dS_dIi.imag)))

        return (
            np.asarray(cols, dtype=np.int32),
            np.asarray(dP, dtype=np.float64),
            np.asarray(dQ, dtype=np.float64),
        )

    @staticmethod
    def _add_sparse_generator_row(H, row: int, cols: np.ndarray, values: np.ndarray) -> None:
        if cols.size == 0:
            return
        rows = np.full(cols.size, int(row), dtype=np.int32)
        H.add_many(rows, cols, values)

    @staticmethod
    def _add_sparse_repeated_rows(H, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        if rows.size == 0 or cols.size == 0:
            return
        if rows.size == 1:
            H.add_many(np.full(cols.size, int(rows[0]), dtype=np.int32), cols, values)
            return
        H.add_many(
            np.repeat(rows.astype(np.int32, copy=False), cols.size),
            np.tile(cols.astype(np.int32, copy=False), rows.size),
            np.tile(values.astype(np.float64, copy=False), rows.size),
        )

    def _generator_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, object]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_generator_measurement_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_generator_measurement_plan(measurements)
            self._active_generator_measurement_plan = plan
            return plan
        key = id(measurements)
        cached = self._generator_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_generator_measurement_plan(measurements)
        if len(self._generator_measurement_plan_cache) > 16:
            self._generator_measurement_plan_cache.clear()
        self._generator_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _build_generator_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, object]:
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_generator_plan_kind_by_type_code,
        )
        rows = plan_table.row[
            (plan_table.device_type_code == _DEVICE_TYPE_CODES["ACGenerator"]) & plan_table.handled
        ]
        device_pos = plan_table.device_pos[rows]
        return {
            "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
            "value_rows": self._int_array(rows),
            "value_kind": self._int_array(plan_table.meas_kind[rows]),
            "value_pos": self._ac_generator_plan_node_pos[device_pos],
            "value_index": self._ac_generator_plan_index[device_pos],
        }

    def _fill_generator_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACGenerator P/Q/I measurements from explicit P/Q states."""
        plan = self._generator_measurement_plan(measurements)
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        kind = plan["value_kind"]
        row_values = np.zeros(rows.size, dtype=np.float64)
        p_mask = kind == 0
        q_mask = kind == 1
        i_mask = kind == 2
        row_values[p_mask] = p[p_mask]
        row_values[q_mask] = q[q_mask]
        if np.any(i_mask):
            vm = voltage[pos[i_mask]]
            valid = np.abs(vm) > self.min_current_voltage
            current_values = np.zeros(np.count_nonzero(i_mask), dtype=np.float64)
            current_values[valid] = np.hypot(p[i_mask][valid], q[i_mask][valid]) / vm[valid]
            row_values[i_mask] = current_values
        values[rows] = row_values
        return plan["handled_mask"]

    def _fill_generator_jacobian_sparse(
        self,
        H,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse generator P/Q/I rows from explicit P/Q states."""
        plan = self._generator_measurement_plan(measurements)
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        kind = plan["value_kind"]
        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        p_mask = kind == 0
        q_mask = kind == 1
        i_mask = kind == 2
        if np.any(p_mask):
            self._add_indexed_values(
                H,
                rows[p_mask],
                self.base_gen_p + gen_idx[p_mask],
                np.ones(np.count_nonzero(p_mask), dtype=np.float64),
            )
        if np.any(q_mask):
            self._add_indexed_values(
                H,
                rows[q_mask],
                self.base_gen_q + gen_idx[q_mask],
                np.ones(np.count_nonzero(q_mask), dtype=np.float64),
            )
        if np.any(i_mask):
            i_rows = rows[i_mask]
            i_idx = gen_idx[i_mask]
            vm = voltage[pos[i_mask]]
            p_i = p[i_mask]
            q_i = q[i_mask]
            s_abs = np.hypot(p_i, q_i)
            valid = (np.abs(vm) > self.min_current_voltage) & (s_abs > 1e-12)
            self._add_indexed_values(
                H,
                i_rows,
                self.base_gen_p + i_idx,
                np.divide(p_i, s_abs * vm, out=np.zeros_like(p_i), where=valid),
                valid,
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.base_gen_q + i_idx,
                np.divide(q_i, s_abs * vm, out=np.zeros_like(q_i), where=valid),
                valid,
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.voltage_col[pos[i_mask]],
                np.divide(-s_abs, vm * vm, out=np.zeros_like(s_abs), where=valid),
                valid,
            )

        return plan["handled_mask"]

    def _generator_derivative_vectors(
        self,
        gen,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
        network_dtheta: np.ndarray,
        network_dvoltage: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build dP/dx and dQ/dx for a generator represented by node power balance."""
        pos = self.node_pos[gen.node]
        share = self.generator_share_by_name[gen.name]
        dP = np.zeros(self.n_state, dtype=np.float64)
        dQ = np.zeros(self.n_state, dtype=np.float64)
        angle_mask = self.angle_col >= 0
        if np.any(angle_mask):
            np.add.at(dP, self.angle_col[angle_mask], share * network_dtheta[pos, angle_mask].real)
            np.add.at(dQ, self.angle_col[angle_mask], share * network_dtheta[pos, angle_mask].imag)
        voltage_mask = self.voltage_col >= 0
        if np.any(voltage_mask):
            np.add.at(dP, self.voltage_col[voltage_mask], share * network_dvoltage[pos, voltage_mask].real)
            np.add.at(dQ, self.voltage_col[voltage_mask], share * network_dvoltage[pos, voltage_mask].imag)
        own_voltage_col = int(self.voltage_col[pos])
        if own_voltage_col >= 0:
            dP[own_voltage_col] += share * p_load_dv[pos]
            dQ[own_voltage_col] += share * q_load_dv[pos]
        self._add_switch_injection_derivatives(
            dP,
            dQ,
            pos,
            switch_current,
            voltage_complex,
            theta,
            voltage,
            share,
        )
        return dP, dQ

    def _assemble_jacobian(
        self,
        x: np.ndarray,
        measurements: Optional[Sequence[Measurement]] = None,
        sparse: bool = False,
    ):
        """Assemble the WLS measurement Jacobian H = dh(x)/dx."""
        measurements = self._normalize_measurements(measurements)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        if sparse and measurements is self.active_measurements:
            H = self._jacobian_builder
            H.shape = (len(measurements), self.n_state)
            H.size = H.shape[0] * H.shape[1]
            H.reset()
        else:
            H = SparseJacobianBuilder((len(measurements), self.n_state)) if sparse else np.zeros((len(measurements), self.n_state), dtype=np.float64)
        branch_power_derivative_cache = {}
        branch_current_derivative_cache = {}
        vectorized_branch_rows = self._fill_branch_transformer_jacobian_vectorized(
            H,
            measurements,
            theta,
            voltage,
            voltage_complex,
        )
        vectorized_simple_rows = self._fill_simple_jacobian_vectorized(
            H,
            measurements,
            x,
            voltage,
        )
        vectorized_zero_rows = self._fill_zero_current_jacobian_vectorized(
            H,
            measurements,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_generator_rows = self._fill_generator_jacobian_sparse(
            H,
            measurements,
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_jacobian_vectorized(
            H,
            measurements,
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_simple_rows
            | vectorized_zero_rows
            | vectorized_generator_rows
            | vectorized_balance_rows
        )
        if sparse and (
            (measurements is self.active_measurements and self.active_measurements_are_vectorized)
            or np.all(vectorized_rows)
        ):
            return H.to_csr()

        row_meta = self._measurement_sequence_arrays(measurements)
        if row_meta is not None:
            device_types, device_names, meas_types = row_meta
            row_iter = zip(range(len(measurements)), device_types, device_names, meas_types)
        else:
            row_iter = ((row, meas.device_type, meas.device_name, meas.meas_type) for row, meas in enumerate(measurements))

        for row, device_type, device_name, mtype in row_iter:
            if vectorized_rows[row]:
                continue
            if device_type == "ACNode":
                node = self.node_by_name[device_name]
                pos = self.node_pos[node.idx]
                if mtype == "V":
                    H[row, self.voltage_col[pos]] = 1.0
                elif mtype in ("ANGLE", "THETA"):
                    angle_col = self.angle_col[pos]
                    if angle_col >= 0:
                        H[row, angle_col] = 1.0
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")

            elif device_type == "ACBranch":
                br = self.branch_by_name[device_name]
                i = self.node_pos[br.i_node]
                j = self.node_pos[br.j_node]
                yff, yft, ytf, ytt = self.branch_stamp_by_name[br.name]
                if mtype in ("P_FROM", "Q_FROM"):
                    cache_key = (device_type, br.name, "from")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, i, j, *derivatives)
                elif mtype == "V_FROM":
                    H[row, self.voltage_col[i]] = 1.0
                elif mtype == "I_FROM":
                    cache_key = (device_type, br.name, "from")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, i, j, *derivatives)
                elif mtype in ("P_TO", "Q_TO"):
                    cache_key = (device_type, br.name, "to")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, j, i, *derivatives)
                elif mtype == "V_TO":
                    H[row, self.voltage_col[j]] = 1.0
                elif mtype == "I_TO":
                    cache_key = (device_type, br.name, "to")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, j, i, *derivatives)
                else:
                    raise RuntimeError(f"Unsupported ACBranch measurement type: {mtype}")

            elif device_type == "ACTransformer":
                tr = self.transformer_by_name[device_name]
                i = self.node_pos[tr.i_node]
                j = self.node_pos[tr.j_node]
                yff, yft, ytf, ytt = self.transformer_stamp_by_name[tr.name]
                if mtype in ("P_FROM", "Q_FROM"):
                    cache_key = (device_type, tr.name, "from")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, i, j, *derivatives)
                elif mtype == "V_FROM":
                    H[row, self.voltage_col[i]] = 1.0
                elif mtype == "I_FROM":
                    cache_key = (device_type, tr.name, "from")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, i, j, *derivatives)
                elif mtype in ("P_TO", "Q_TO"):
                    cache_key = (device_type, tr.name, "to")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, j, i, *derivatives)
                elif mtype == "V_TO":
                    H[row, self.voltage_col[j]] = 1.0
                elif mtype == "I_TO":
                    cache_key = (device_type, tr.name, "to")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, j, i, *derivatives)
                else:
                    raise RuntimeError(f"Unsupported ACTransformer measurement type: {mtype}")

            elif device_type == "ACLoad":
                load = self.load_by_name[device_name]
                node = self.node_by_name[load.node_obj.name]
                pos = self.node_pos[node.idx]
                if mtype == "P_LOAD":
                    H[row, self.load_p_col_by_name[load.name]] = 1.0
                elif mtype == "Q_LOAD":
                    H[row, self.load_q_col_by_name[load.name]] = 1.0
                elif mtype == "V_LOAD":
                    H[row, self.voltage_col[pos]] = 1.0
                elif mtype == "I_LOAD":
                    vm = voltage[pos]
                    load_idx = self.load_state_index_by_name[load.name]
                    p = float(x[self.base_load_p + load_idx])
                    q = float(x[self.base_load_q + load_idx])
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(vm) > self.min_current_voltage:
                        H[row, self.base_load_p + load_idx] = p / (s_abs * vm)
                        H[row, self.base_load_q + load_idx] = q / (s_abs * vm)
                        H[row, self.voltage_col[pos]] = -s_abs / (vm * vm)
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")

            elif device_type == "ACGenerator":
                gen = self.generator_by_name[device_name]
                node = self.node_by_name[gen.node_obj.name]
                pos = self.node_pos[node.idx]
                if mtype == "V_GEN":
                    H[row, self.voltage_col[pos]] = 1.0
                    continue

                if mtype == "P_GEN":
                    H[row, self.gen_p_col_by_name[gen.name]] = 1.0
                elif mtype == "Q_GEN":
                    H[row, self.gen_q_col_by_name[gen.name]] = 1.0
                elif mtype == "I_GEN":
                    gen_idx = self.generator_state_index_by_name[gen.name]
                    p = float(x[self.base_gen_p + gen_idx])
                    q = float(x[self.base_gen_q + gen_idx])
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(voltage[pos]) > self.min_current_voltage:
                        H[row, self.base_gen_p + gen_idx] = p / (s_abs * voltage[pos])
                        H[row, self.base_gen_q + gen_idx] = q / (s_abs * voltage[pos])
                        H[row, self.voltage_col[pos]] = -s_abs / (voltage[pos] * voltage[pos])
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")

            elif device_type == "ACZeroBranch":
                zbr = self.zero_branch_by_name[device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[zbr.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[zbr.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[zbr.i_node]]
                    j_col = self.angle_col[self.node_pos[zbr.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    zbr_idx = self.zero_branch_pos[zbr.name]
                    self._add_zero_current_measurement_derivatives(
                        H,
                        row,
                        mtype,
                        zbr,
                        switch_current[zbr_idx],
                        zbr_idx,
                        voltage_complex,
                        voltage,
                    )

            elif device_type == "ACZeroBranchConstraint":
                zbr = self.zero_branch_by_name[device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[zbr.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[zbr.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[zbr.i_node]]
                    j_col = self.angle_col[self.node_pos[zbr.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    raise RuntimeError(f"Unsupported ACZeroBranchConstraint measurement type: {mtype}")

            elif device_type == "ACBreakConstraint":
                brk = self.break_by_name[device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[brk.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[brk.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[brk.i_node]]
                    j_col = self.angle_col[self.node_pos[brk.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    raise RuntimeError(f"Unsupported ACBreakConstraint measurement type: {mtype}")

            elif device_type == "ACBreak":
                brk = self.break_by_name[device_name]
                brk_idx = self.break_pos[brk.name]
                self._add_zero_current_measurement_derivatives(
                    H,
                    row,
                    mtype,
                    brk,
                    switch_current[brk_idx],
                    brk_idx,
                    voltage_complex,
                    voltage,
                )

            else:
                raise RuntimeError(f"Unsupported measurement device type: {device_type}")
        return H.to_csr() if sparse else H

    def jacobian(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Assemble the WLS measurement Jacobian as a dense array for diagnostics/tests."""
        return self._assemble_jacobian(x, measurements, sparse=False)

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None):
        """Assemble the WLS measurement Jacobian directly as sparse CSR triplets."""
        return self._assemble_jacobian(x, measurements, sparse=True)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        """Use the rank of H to identify whether all state variables are observable."""
        if (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
            and self._initial_observability_cache is not None
        ):
            return self._initial_observability_cache
        use_default_cache = (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
        )
        if use_default_cache:
            cached = _OBSERVABILITY_RESULT_CACHE.get(self._default_observability_cache_key())
            if cached is not None:
                self._initial_observability_cache = cached
                return cached
        x = self.initial_state() if x is None else x
        measurements = self._normalize_measurements(measurements)
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            return ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])

        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.n_state,
            normal_matrix=normal_matrix,
            normal_factor_diag=normal_factor_diag,
        )
        if deficiency > 0 and self._has_structural_observability_certificate(H):
            rank = self.n_state
            deficiency = 0
            weak_states = []
        result = ObservabilityResult(
            observable=rank == self.n_state,
            rank=rank,
            state_count=self.n_state,
            measurement_count=len(measurements),
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )
        if use_default_cache:
            if len(_OBSERVABILITY_RESULT_CACHE) > 32:
                _OBSERVABILITY_RESULT_CACHE.clear()
            _OBSERVABILITY_RESULT_CACHE[self._default_observability_cache_key()] = result
            self._initial_observability_cache = result
        self._cache_observability_matrix(result, x, measurements, H)
        return result

    def _has_structural_observability_certificate(self, H) -> bool:
        """Certify large sparse AC cases when numeric LU is conservative but structure is anchored."""
        rank = sparse_structural_rank(H)
        if rank != self.n_state:
            return False
        return not unanchored_angle_state_indices(H, self.angle_col[self.angle_col >= 0])

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        """Run weighted least squares with simple damping to avoid voltage divergence."""
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        measurements = self._normalize_measurements(measurements)
        if len(measurements) < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {len(measurements)} < {self.n_state}")

        x = self.initial_state() if x0 is None else x0.copy()
        if observability is None:
            if measurements is self.active_measurements and x0 is None:
                observability = self.observability_analysis()
            else:
                observability = self.observability_analysis(x, measurements)
        observability_cache = self._observability_matrix_cache_for(observability, measurements, x)
        cached_initial_H = observability_cache.get("H") if observability_cache is not None else None
        z, weight = self._measurement_vectors(measurements)
        uniform_weight = self.active_uniform_weight if measurements is self.active_measurements else self._uniform_weight(weight)
        weights_are_uniform = self.active_weights_are_uniform if measurements is self.active_measurements else uniform_weight is not None
        weighted_residual = None if weights_are_uniform else np.empty_like(weight)
        converged = False
        max_correction = np.inf
        objective = np.inf
        iteration = 0
        H = None
        gain = np.zeros((self.n_state, self.n_state), dtype=np.float64)
        final_quantities_current = False
        cached_z_est = None
        cached_residual = None
        cached_objective = None
        normal_solver = NormalEquationSolver(assume_fixed_pattern=measurements is self.active_measurements)
        normal_pattern = self._active_normal_pattern if measurements is self.active_measurements else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]

        if verbose:
            _print_iteration_header()

        flat_restart_enabled = False
        iteration_limit = self.max_iter

        for iteration in range(1, iteration_limit + 1):
            if cached_z_est is None:
                z_est = self.evaluate(x, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                residual = self._measurement_residual(z, z_est, measurements)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.residual_objective", time.perf_counter() - start)
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
                cached_z_est = cached_residual = cached_objective = None
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if iteration == 1 and cached_initial_H is not None:
                H = cached_initial_H
                cached_initial_H = None
            else:
                H = self.jacobian_sparse(x, measurements)
            if normal_pattern is None and issparse(H):
                normal_pattern = _normal_equation_structural_pattern(H)
                if measurements is self.active_measurements:
                    self._active_normal_pattern = normal_pattern
                if observability_cache is not None:
                    observability_cache["normal_pattern"] = normal_pattern
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            gain, rhs = build_normal_equations(
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=normal_pattern,
                assume_normal_pattern_matches=measurements is self.active_measurements,
            )
            dx, _ = normal_solver.solve(
                gain,
                rhs,
                return_factor_diag=False,
            )

            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            if max_correction < self.tol:
                converged = True
                final_quantities_current = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break

            accepted = False
            step_scale = 1.0
            accepted_step = None
            nonfinite_candidates = 0
            previous_objective = objective
            # Backtracking keeps the weighted objective non-increasing on difficult cases.
            for _ in range(20):
                candidate = x + step_scale * dx
                candidate[self.n_angle : self.base_switch_re] = np.maximum(
                    candidate[self.n_angle : self.base_switch_re],
                    self.voltage_floor,
                )
                candidate_z_est = self.evaluate(candidate, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                candidate_residual = self._measurement_residual(z, candidate_z_est, measurements)
                candidate_objective = self._weighted_objective(weight, candidate_residual)
                if start is not None:
                    self._record_profile_time("solve.line_search_residual_objective", time.perf_counter() - start)
                finite_candidate = np.isfinite(candidate_objective)
                if not finite_candidate:
                    nonfinite_candidates += 1
                    if nonfinite_candidates >= 8:
                        break
                    step_scale *= 0.5
                    continue
                nonfinite_candidates = 0
                objective_tol = max(1e-12, 1e-10 * (abs(objective) + 1.0))
                if finite_candidate and candidate_objective <= objective + objective_tol:
                    x = candidate
                    cached_z_est = candidate_z_est
                    cached_residual = candidate_residual
                    cached_objective = candidate_objective
                    objective = candidate_objective
                    accepted_step = step_scale
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                final_quantities_current = True
                break

            objective_change = abs(previous_objective - objective)
            stagnation_tol = max(1e-14, 1e-10 * (abs(previous_objective) + 1.0))
            practically_converged = (
                max_correction < 10.0 * self.tol
                and objective_change <= stagnation_tol
            )

            if verbose:
                if cached_residual is None:
                    cached_z_est = self.evaluate(x, measurements)
                    cached_residual = self._measurement_residual(z, cached_z_est, measurements)
                    cached_objective = self._weighted_objective(weight, cached_residual)
                updated_residual = cached_residual
                updated_residual_inf = float(np.linalg.norm(updated_residual, np.inf)) if updated_residual.size else 0.0
                _print_iteration(
                    iteration,
                    cached_objective,
                    updated_residual_inf,
                    max_correction,
                    accepted_step,
                    False,
                )

            if max_correction < self.tol or practically_converged:
                converged = True
                break

        if not converged and flat_restart_enabled:
            restart_result = self.estimate(
                measurements,
                x0=self._file_state(),
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
            restart_result.iterations += iteration
            return restart_result

        if not final_quantities_current:
            if cached_z_est is None:
                z_est = self.evaluate(x, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                residual = self._measurement_residual(z, z_est, measurements)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.final_residual_objective", time.perf_counter() - start)
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
        if final_diagnostics and not final_quantities_current:
            H = self.jacobian_sparse(x, measurements)
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            gain, _ = build_normal_equations(
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=normal_pattern,
            )
        elif not final_diagnostics:
            H = None
            gain = None
        self.apply_state(x)
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        return EstimateResult(
            converged=converged,
            iterations=iteration,
            objective=objective,
            max_correction=max_correction,
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=x,
            z_est=z_est,
            residual=residual,
            H=H,
            gain=gain,
            measurements=measurements,
            observability=observability,
        )

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Return measurements whose normalized residual exceeds the bad-data threshold."""
        if result.H is None or result.gain is None:
            result.H = self.jacobian_sparse(result.x, result.measurements)
            result.gain, _ = build_normal_equations(
                result.H,
                result.residual,
                np.asarray([meas.weight for meas in result.measurements], dtype=np.float64),
            )
        threshold = self.params.bad_threshold if threshold is None else threshold
        weights = np.asarray([meas.weight for meas in result.measurements], dtype=np.float64)
        R_diag = 1.0 / weights
        gain_inv = inverse_gain_for_bad_data(result.gain)
        if gain_inv is None:
            leverage = np.zeros_like(R_diag)
        else:
            leverage = measurement_leverage(result.H, gain_inv)
        omega_diag = np.maximum(R_diag - leverage, 1e-12)
        normalized = np.abs(result.residual) / np.sqrt(omega_diag)

        bad_items = []
        for idx in np.where(normalized > threshold)[0]:
            meas = result.measurements[int(idx)]
            bad_items.append(
                BadDataItem(
                    measurement=meas,
                    residual=float(result.residual[idx]),
                    normalized_residual=float(normalized[idx]),
                    estimated_value=float(result.z_est[idx]),
                    measured_value=float(meas.value),
                )
            )
        bad_items.sort(key=lambda item: item.normalized_residual, reverse=True)
        return bad_items, normalized

    def build_se_result(
        self,
        result: EstimateResult,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        return_mode: str = "full",
    ) -> Optional[SEResult]:
        """Build the structured state-estimation result snapshot after WLS."""
        mode = normalize_seresult_return_mode(return_mode)
        if mode in ("none", "array"):
            self.se_result = None
            return None
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        if mode == "summary":
            self.se_result = build_seresult_summary(
                result,
                bad_items=bad_items,
                all_measurements=self.measurements,
            )
            return self.se_result
        self.se_result = SEResult.from_estimate_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurements=self.measurements,
        )
        return self.se_result

    def run(
        self,
        *,
        return_mode: str = "full",
        remove_bad_data: bool = False,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        skip_bad_data: bool = False,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> Optional[SEResult]:
        self._require_prepared("run()")
        mode = normalize_seresult_return_mode(return_mode)
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        array_only = mode == "array"
        needs_bad_data = (not skip_bad_data) and not array_only
        if observability is None:
            observability = self.observability_analysis()
        self.observability_result = observability
        removed: List[BadDataItem] = []
        if remove_bad_data:
            result, removed = self.estimate_with_bad_data_removal(
                threshold,
                max_remove=max_remove,
                verbose=verbose,
            )
        else:
            result = self.estimate(
                verbose=verbose,
                final_diagnostics=final_diagnostics and needs_bad_data,
                observability=observability,
            )
        self.estimate_result = result
        self.removed_bad_data = removed
        if skip_bad_data or array_only:
            bad_items = []
            normalized = np.array([], dtype=np.float64)
        else:
            bad_items, normalized = self.identify_bad_data(result, threshold)
        self.bad_items = bad_items
        self.normalized_residual = normalized
        if mode in ("none", "array"):
            self.se_result = None
            return None
        return self.build_se_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized,
            return_mode=mode,
        )

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        threshold = self.params.bad_threshold if threshold is None else threshold
        max_remove = self.params.max_remove if max_remove is None else max_remove
        measurements = self.active_measurements
        removed: List[BadDataItem] = []
        x0 = self.initial_state()
        for round_idx in range(max_remove + 1):
            if verbose:
                print(f"Bad-data removal round {round_idx + 1}: measurements={len(measurements)}")
            result = self.estimate(measurements, x0=x0, verbose=verbose)
            bad_items, _ = self.identify_bad_data(result, threshold)
            if not bad_items:
                return result, removed
            worst = bad_items[0]
            removed.append(worst)
            remove_pos = result.measurements.index(worst.measurement)
            keep_rows = np.concatenate(
                (
                    np.arange(remove_pos, dtype=np.int64),
                    np.arange(remove_pos + 1, len(result.measurements), dtype=np.int64),
                )
            )
            if measurements is self.active_measurements and result.measurements is self.active_measurements:
                measurements = self._shrink_active_measurement_indexes(remove_pos)
            else:
                measurements = take_measurement_view(result.measurements, keep_rows)
            x0 = result.x
        return result, removed

    def apply_state(self, x: np.ndarray) -> None:
        theta, voltage = self._unpack_state(x)
        for pos, node in enumerate(self.nodes):
            node.angle = float(theta[pos])
            node.voltage = float(voltage[pos])
            for member in getattr(node, "nodes", ()):
                member.angle = node.angle
                member.voltage = node.voltage
        for idx, gen in enumerate(self.generator_order):
            gen.p = float(x[self.base_gen_p + idx])
            gen.q = float(x[self.base_gen_q + idx])
        for idx, load in enumerate(self.load_order):
            load.p = float(x[self.base_load_p + idx])
            load.q = float(x[self.base_load_q + idx])
        for idx, shunt in enumerate(self.voltage_control_shunt_order):
            shunt.p = 0.0
            shunt.q = float(x[self.base_shunt_q + idx])
            voltage_value = float(getattr(getattr(shunt, "node_obj", None), "voltage", 1.0) or 1.0)
            shunt.current = abs(shunt.q) / voltage_value if abs(voltage_value) > self.min_current_voltage else 0.0

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        theta, voltage = self._unpack_state(x)
        print("Estimated states:")
        for pos, node in enumerate(self.nodes[:limit]):
            print(f"  {node.name:10s} V={voltage[pos]:.9f} theta={theta[pos]:.9f} rad")
        if len(self.nodes) > limit:
            print(f"  ... {len(self.nodes) - limit} more nodes")


_ORIGINAL_AC_RUN_POWER_FLOW_SEED = ACStateEstimator._run_power_flow_seed


def _print_observability(result: ObservabilityResult) -> None:
    print(
        "Observability: "
        f"observable={result.observable}, "
        f"rank={result.rank}/{result.state_count}, "
        f"measurements={result.measurement_count}, "
        f"deficiency={result.deficiency}"
    )
    if result.weak_states:
        print("Weak/unobservable state candidates:")
        for label, score in result.weak_states:
            print(f"  {label}: {score:.3e}")


def _print_bad_data(items: Sequence[BadDataItem], normalized: np.ndarray, threshold: float, top: int = 10) -> None:
    max_norm = float(np.max(normalized)) if normalized.size else 0.0
    print(f"Bad data: threshold={threshold:.3f}, max_normalized_residual={max_norm:.3e}, count={len(items)}")
    for item in list(items)[:top]:
        meas = item.measurement
        print(
            f"  idx={meas.idx} name={meas.name} type={meas.meas_type} device={meas.device} "
            f"z={item.measured_value:.9g} h={item.estimated_value:.9g} "
            f"res={item.residual:.3e} rn={item.normalized_residual:.3e}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AC weighted least-squares state estimation.")
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="AC network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat voltage start instead of E-file voltage/angle.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--skip-bad-data", action="store_true", help="Skip post-estimation bad-data analysis.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated node states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    parser.add_argument("--profile", action="store_true", help="Print initialization profile timings.")
    parser.add_argument("--return-mode", default="full", help="SEResult payload mode: full, summary, array, or none.")
    parser.add_argument("--se-result", default=None, help="Write SEResult blocks to a new E file.")
    args = parser.parse_args(argv)

    estimator = ACStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        parameter_file=Path(args.para),
        flat_start=args.flat_start,
        profile=args.profile,
        auto_prepare=False,
    )
    estimator.prepare()

    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    se_result = estimator.run(
        return_mode=args.return_mode if args.se_result else "none",
        remove_bad_data=args.remove_bad_data,
        bad_threshold=bad_threshold,
        max_remove=args.max_remove,
        skip_bad_data=args.skip_bad_data,
        verbose=not args.quiet,
        final_diagnostics=not args.skip_bad_data,
    )
    _print_observability(estimator.observability_result)
    result = estimator.estimate_result
    removed = estimator.removed_bad_data
    if removed:
        print("Removed bad data:")
        for item in removed:
            print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")

    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
    )
    if not args.skip_bad_data:
        bad_items = estimator.bad_items
        normalized = estimator.normalized_residual
        _print_bad_data(bad_items, normalized, bad_threshold)
    else:
        bad_items = []
        normalized = np.array([], dtype=np.float64)

    if args.se_result and se_result is not None:
        se_result.write_e_file(Path(args.se_result))

    if args.print_state:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in sorted(estimator.profile_times.items()):
            print(f"  {name}={value:.6f}s")

    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
