import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from efile_read import _read_efile_rows
from paths import resolve_project_file
from model.meas_model import (
    DEVICE_TYPE_CODES,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_REMOVED,
    MeasurementTable,
    TableBackedMeasurementList,
    normalize_measurement_status,
    measurement_status_is_active,
)


MEAS_COLS = {
    "idx": 0,
    "device_type_code": 1,
    "device_name_id": 2,
    "meas_type_code": 3,
    "weight": 4,
    "valid": 5,
    "value": 6,
    "status": 7,
    "angle_mask": 8,
    "source_row": 9,
}

MEAS_TYPE_CODES = {
    "UNKNOWN": 0,
    "V": 1,
    "ANGLE": 2,
    "THETA": 3,
    "P_FROM": 4,
    "Q_FROM": 5,
    "V_FROM": 6,
    "I_FROM": 7,
    "P_TO": 8,
    "Q_TO": 9,
    "V_TO": 10,
    "I_TO": 11,
    "P_LOAD": 12,
    "Q_LOAD": 13,
    "V_LOAD": 14,
    "I_LOAD": 15,
    "P_GEN": 16,
    "Q_GEN": 17,
    "V_GEN": 18,
    "I_GEN": 19,
    "P_BALANCE": 20,
    "Q_BALANCE": 21,
    "V_DIFF": 22,
    "ANGLE_DIFF": 23,
    "THETA_DIFF": 24,
    "P_DC": 25,
    "V_DC": 26,
    "I_DC": 27,
    "P_AC": 28,
    "Q_AC": 29,
    "V_AC": 30,
    "I_AC": 31,
    "P_IN": 32,
    "P_OUT": 33,
    "I_OUT": 34,
}
MEAS_TYPE_NAMES = {code: name for name, code in MEAS_TYPE_CODES.items()}

ANGLE_MEASUREMENT_TYPE_CODES = np.asarray(
    [
        MEAS_TYPE_CODES["ANGLE"],
        MEAS_TYPE_CODES["THETA"],
        MEAS_TYPE_CODES["ANGLE_DIFF"],
        MEAS_TYPE_CODES["THETA_DIFF"],
    ],
    dtype=np.int16,
)
ANGLE_MEASUREMENT_TYPES = frozenset(MEAS_TYPE_NAMES[int(code)] for code in ANGLE_MEASUREMENT_TYPE_CODES)

_MEAS_PPC_CACHE = {}
_MEAS_PPC_CACHE_LOCK = threading.Lock()
_REQUIRED_COLUMNS = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")


def clear_meas_ppc_cache(file_path=None) -> None:
    with _MEAS_PPC_CACHE_LOCK:
        if file_path is None:
            _MEAS_PPC_CACHE.clear()
            return
        path = resolve_project_file(file_path).resolve()
        _MEAS_PPC_CACHE.pop(path, None)


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _device_type_codes_from_names(device_type_values: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    device_type_code_values = np.zeros(device_type_values.shape[0], dtype=np.int16)
    rows_by_code: Dict[int, np.ndarray] = {}
    for device_type, code in DEVICE_TYPE_CODES.items():
        rows = np.flatnonzero(device_type_values == device_type)
        if rows.size:
            code_int = int(code)
            device_type_code_values[rows] = code_int
            rows_by_code[code_int] = rows.astype(np.int64, copy=False)
    unknown_rows = np.flatnonzero(device_type_code_values == 0)
    if unknown_rows.size:
        rows_by_code[0] = unknown_rows.astype(np.int64, copy=False)
    return device_type_code_values, rows_by_code


def _meas_type_codes_from_names(meas_type_values: np.ndarray) -> np.ndarray:
    if meas_type_values.size == 0:
        return np.empty(0, dtype=np.int16)
    code_get = MEAS_TYPE_CODES.get
    return np.asarray([code_get(str(name).upper(), 0) for name in meas_type_values.tolist()], dtype=np.int16)


def _encode_device_names(device_name_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if device_name_values.size == 0:
        return np.empty(0, dtype=np.int32), np.asarray([], dtype=object)
    ids = np.empty(int(device_name_values.size), dtype=np.int32)
    names = []
    lookup = {}
    for pos, name in enumerate(device_name_values.astype(object, copy=False)):
        name_id = lookup.get(name)
        if name_id is None:
            name_id = len(names)
            lookup[name] = name_id
            names.append(name)
        ids[pos] = name_id
    return ids, np.asarray(names, dtype=object)


def _build_ppc(
    *,
    source: Path,
    idx: np.ndarray,
    name: np.ndarray,
    device_type: np.ndarray,
    device_name: np.ndarray,
    meas_type: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    value: np.ndarray,
    status: np.ndarray,
    rows_by_device_type_code: Optional[Dict[int, np.ndarray]] = None,
    device_type_code: Optional[np.ndarray] = None,
    meas_type_code: Optional[np.ndarray] = None,
    device_name_id: Optional[np.ndarray] = None,
    device_names: Optional[np.ndarray] = None,
) -> Dict:
    count = int(idx.size)
    if device_type_code is None:
        device_type_code, computed_rows_by_code = _device_type_codes_from_names(device_type)
    else:
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        computed_rows_by_code = None
    if rows_by_device_type_code is None:
        if computed_rows_by_code is None:
            rows_by_device_type_code = {
                int(code): np.flatnonzero(device_type_code == int(code)).astype(np.int64, copy=False)
                for code in np.unique(device_type_code)
            }
        else:
            rows_by_device_type_code = computed_rows_by_code
    if meas_type_code is None:
        meas_type_code = _meas_type_codes_from_names(meas_type)
    else:
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
    if device_name_id is None or device_names is None:
        device_name_id, device_names = _encode_device_names(device_name)
    else:
        device_name_id = np.asarray(device_name_id, dtype=np.int32)
        device_names = np.asarray(device_names, dtype=object)
    if device_name_id.size == count and device_names.size:
        device_name = device_names[device_name_id.astype(np.intp, copy=False)]
    angle_mask = np.isin(meas_type_code, ANGLE_MEASUREMENT_TYPE_CODES)
    meas = np.zeros((count, len(MEAS_COLS)), dtype=np.float64)
    if count:
        meas[:, MEAS_COLS["idx"]] = idx.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["device_type_code"]] = device_type_code.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["device_name_id"]] = device_name_id.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["meas_type_code"]] = meas_type_code.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["weight"]] = weight.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["valid"]] = valid.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["value"]] = value.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["status"]] = status.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["angle_mask"]] = angle_mask.astype(np.float64, copy=False)
        meas[:, MEAS_COLS["source_row"]] = np.arange(count, dtype=np.float64)
    return {
        "format": "meas_ppc_v1",
        "source": str(source),
        "meas": meas,
        "meas_cols": MEAS_COLS,
        "meas_type_codes": MEAS_TYPE_CODES,
        "meas_type_names": MEAS_TYPE_NAMES,
        "device_type_codes": DEVICE_TYPE_CODES,
        "device_type_names": {code: name for name, code in DEVICE_TYPE_CODES.items()},
        "name": name,
        "device_type": device_type,
        "device_name": device_name,
        "device_names": device_names,
        "device_name_id_by_name": {
            name: int(pos)
            for pos, name in enumerate(device_names.astype(object, copy=False).tolist())
        },
        "meas_type": meas_type,
        "rows_by_device_type_code": rows_by_device_type_code,
        "normalized": False,
    }


def _empty_meas_ppc(source: Path) -> Dict:
    return _build_ppc(
        source=source,
        idx=np.asarray([], dtype=np.int64),
        name=np.asarray([], dtype=object),
        device_type=np.asarray([], dtype=object),
        device_name=np.asarray([], dtype=object),
        meas_type=np.asarray([], dtype=object),
        weight=np.asarray([], dtype=np.float64),
        valid=np.asarray([], dtype=bool),
        value=np.asarray([], dtype=np.float64),
        status=np.asarray([], dtype=np.int16),
        rows_by_device_type_code={},
        device_type_code=np.asarray([], dtype=np.int16),
        meas_type_code=np.asarray([], dtype=np.int16),
        device_name_id=np.asarray([], dtype=np.int32),
        device_names=np.asarray([], dtype=object),
    )


def _build_meas_ppc_from_rows_dict(rows_dict: Dict, source: Path) -> Dict:
    block = rows_dict.get("Measurement") if isinstance(rows_dict, dict) else None
    if block is None:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    header = tuple(block.get("header_list") or ())
    if not header:
        raise RuntimeError(f"{source} Measurement block does not contain a header")
    header_index = {name: idx for idx, name in enumerate(header)}
    missing = [name for name in _REQUIRED_COLUMNS if name not in header_index]
    if missing:
        raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
    rows = block.get("rows") or ()
    count = len(rows)
    if count == 0:
        return _empty_meas_ppc(source)
    idx_col = header_index["idx"]
    name_col = header_index["name"]
    device_type_col = header_index["dev_type"]
    device_name_col = header_index["dev_name"]
    meas_type_col = header_index["meas_type"]
    weight_col = header_index["weight"]
    valid_col = header_index["valid"]
    value_col = header_index["value"]
    status_col = header_index.get("status", -1)
    intern = sys.intern
    try:
        idx_values = np.fromiter((int(fields[idx_col]) for fields in rows), dtype=np.int64, count=count)
        name_values = np.asarray([fields[name_col] for fields in rows], dtype=object)
        device_type_list = [intern(fields[device_type_col]) for fields in rows]
        device_type_values = np.asarray(device_type_list, dtype=object)
        device_name_values = np.asarray([fields[device_name_col] for fields in rows], dtype=object)
        meas_type_list = [intern(fields[meas_type_col].upper()) for fields in rows]
        meas_type_values = np.asarray(meas_type_list, dtype=object)
        weight_values = np.fromiter((float(fields[weight_col]) for fields in rows), dtype=np.float64, count=count)
        valid_values = np.asarray(
            [
                fields[valid_col] == "1" or fields[valid_col] == 1 or fields[valid_col] is True
                for fields in rows
            ],
            dtype=bool,
        )
        value_values = np.fromiter((float(fields[value_col]) for fields in rows), dtype=np.float64, count=count)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc
    if status_col >= 0:
        status_values = np.empty(count, dtype=np.int16)
        for row_pos, fields in enumerate(rows):
            status = (
                normalize_measurement_status(fields[status_col], valid=bool(valid_values[row_pos]))
                if len(fields) > status_col
                else MEAS_STATUS_NORMAL if valid_values[row_pos] else MEAS_STATUS_INVALID
            )
            if not measurement_status_is_active(status):
                valid_values[row_pos] = False
            status_values[row_pos] = status
    else:
        status_values = np.where(valid_values, MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID).astype(np.int16, copy=False)
    device_type_code_values = np.asarray(
        [DEVICE_TYPE_CODES.get(device_type, 0) for device_type in device_type_list],
        dtype=np.int16,
    )
    meas_type_code_values = np.asarray(
        [MEAS_TYPE_CODES.get(meas_type, 0) for meas_type in meas_type_list],
        dtype=np.int16,
    )
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_values == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_values)
    }
    device_name_id, device_names = _encode_device_names(device_name_values)
    return _build_ppc(
        source=source,
        idx=idx_values,
        name=name_values,
        device_type=device_type_values,
        device_name=device_name_values,
        meas_type=meas_type_values,
        weight=weight_values,
        valid=valid_values,
        value=value_values,
        status=status_values,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_values,
        meas_type_code=meas_type_code_values,
        device_name_id=device_name_id,
        device_names=device_names,
    )


def build_meas_ppc_from_efile_rows(file_path, rows) -> Dict:
    """Build measurement ppc from E rows that are already loaded in memory."""
    path = resolve_project_file(file_path).resolve()
    return _build_meas_ppc_from_rows_dict(rows, path)


def build_meas_ppc_from_e_file(file_path) -> Dict:
    """Read a measurement file directly into a PPC-style NumPy dictionary."""
    file_key = _file_cache_key(file_path)
    with _MEAS_PPC_CACHE_LOCK:
        cached = _MEAS_PPC_CACHE.get(file_key[0])
        if cached is not None and cached[0] == file_key:
            return cached[1]

    ppc = _build_meas_ppc_from_rows_dict(_read_efile_rows(file_key[0]), file_key[0])
    with _MEAS_PPC_CACHE_LOCK:
        _MEAS_PPC_CACHE[file_key[0]] = (file_key, ppc)
    return ppc


def copy_meas_ppc(ppc: Dict) -> Dict:
    """Return a shallow PPC copy with mutable measurement arrays copied."""
    copied = dict(ppc)
    for key in ("meas", "name", "device_type", "device_name", "device_names", "meas_type"):
        value = ppc.get(key)
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
    if isinstance(ppc.get("device_name_id_by_name"), dict):
        copied["device_name_id_by_name"] = dict(ppc["device_name_id_by_name"])
    rows = ppc.get("rows_by_device_type_code")
    if isinstance(rows, dict):
        copied["rows_by_device_type_code"] = {
            int(code): np.asarray(values, dtype=np.int64).copy()
            for code, values in rows.items()
        }
    return copied


def measurement_table_from_meas_ppc(ppc: Dict) -> MeasurementTable:
    meas = ppc["meas"]
    cols = ppc.get("meas_cols", MEAS_COLS)
    table = MeasurementTable(
        idx=meas[:, cols["idx"]].astype(np.int64, copy=False),
        name=np.asarray(ppc.get("name", ()), dtype=object),
        device_type=np.asarray(ppc.get("device_type", ()), dtype=object),
        device_name=np.asarray(ppc.get("device_name", ()), dtype=object),
        meas_type=np.asarray(ppc.get("meas_type", ()), dtype=object),
        weight=meas[:, cols["weight"]],
        valid=meas[:, cols["valid"]].astype(bool, copy=False),
        value=meas[:, cols["value"]],
        device_type_code=meas[:, cols["device_type_code"]].astype(np.int16, copy=False),
        angle_mask=meas[:, cols["angle_mask"]].astype(bool, copy=False),
        status_code=meas[:, cols["status"]].astype(np.int16, copy=False),
        rows_by_device_type_code=ppc.get("rows_by_device_type_code"),
        device_name_id=meas[:, cols["device_name_id"]].astype(np.int64, copy=False),
        meas_type_code=meas[:, cols["meas_type_code"]].astype(np.int16, copy=False),
    )
    return table


def measurement_list_from_meas_ppc(ppc: Dict) -> TableBackedMeasurementList:
    return TableBackedMeasurementList(
        measurement_table_from_meas_ppc(ppc),
        normalized=bool(ppc.get("normalized", False)),
    )


def sync_meas_ppc_from_measurement_table(ppc: Dict, table: MeasurementTable) -> None:
    meas = ppc["meas"]
    cols = ppc.get("meas_cols", MEAS_COLS)
    if meas.shape[0] != table.idx.size:
        return
    meas[:, cols["weight"]] = np.asarray(table.weight, dtype=np.float64)
    meas[:, cols["valid"]] = np.asarray(table.valid, dtype=np.float64)
    meas[:, cols["value"]] = np.asarray(table.value, dtype=np.float64)
    meas[:, cols["status"]] = np.asarray(table.status_code, dtype=np.float64)
    meas[:, cols["angle_mask"]] = np.asarray(table.angle_mask, dtype=np.float64)
    ppc["normalized"] = bool(getattr(table, "normalized", ppc.get("normalized", False)))


def meas_ppc_active_mask(ppc: Dict) -> np.ndarray:
    meas = ppc["meas"]
    return (
        (meas[:, MEAS_COLS["valid"]] != 0.0)
        & (meas[:, MEAS_COLS["weight"]] > 0.0)
        & ~np.isin(
            meas[:, MEAS_COLS["status"]].astype(np.int16, copy=False),
            np.asarray([MEAS_STATUS_INVALID, MEAS_STATUS_REMOVED], dtype=np.int16),
        )
    )
