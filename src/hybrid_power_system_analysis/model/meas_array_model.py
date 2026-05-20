import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

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
_STANDARD_HEADER = _REQUIRED_COLUMNS


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
    unique, inverse = np.unique(device_name_values.astype(object, copy=False), return_inverse=True)
    return inverse.astype(np.int32, copy=False), unique.astype(object, copy=False)


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
) -> Dict:
    count = int(idx.size)
    device_type_code, computed_rows_by_code = _device_type_codes_from_names(device_type)
    if rows_by_device_type_code is None:
        rows_by_device_type_code = computed_rows_by_code
    meas_type_code = _meas_type_codes_from_names(meas_type)
    device_name_id, device_names = _encode_device_names(device_name)
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
        "meas_type": meas_type,
        "rows_by_device_type_code": rows_by_device_type_code,
        "normalized": False,
    }


def _find_measurement_block(raw: bytes, source: Path) -> bytes:
    block_start = raw.find(b"<Measurement>")
    if block_start < 0:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    header_start = raw.find(b"@", block_start)
    if header_start < 0:
        raise RuntimeError(f"{source} Measurement block does not contain a header")
    data_start = raw.find(b"\n", header_start)
    if data_start < 0:
        raise RuntimeError(f"{source} Measurement block does not contain data rows")
    data_start += 1
    block_end = raw.find(b"</Measurement>", data_start)
    if block_end < 0:
        block_end = len(raw)
    return raw[data_start:block_end]


def _parse_standard_block_lines(block: bytes, source: Path) -> Dict:
    count = int(block.count(b"\n#")) + (1 if block.startswith(b"#") else 0)
    idx_list = [0] * count
    name_list = [None] * count
    device_type_list = [None] * count
    device_name_list = [None] * count
    meas_type_list = [None] * count
    weight_list = [0.0] * count
    valid_list = [False] * count
    value_list = [0.0] * count
    status_list = [MEAS_STATUS_INVALID] * count
    rows_by_code_values = {}
    decode = bytes.decode
    intern = sys.intern
    device_type_cache = {}
    device_name_cache = {}
    meas_type_cache = {}
    row_pos = 0
    for raw_line in block.splitlines():
        if not raw_line:
            continue
        if raw_line[0] != 35:
            stripped = raw_line.strip()
            if stripped:
                raise SyntaxError(f"Invalid Measurement row in {source}: {decode(stripped, 'utf8')}")
            continue
        row = raw_line[1:].split()
        if len(row) < 8:
            raise RuntimeError(f"Malformed Measurement row at row {row_pos + 1} in {source}")
        idx = int(row[0])
        name = decode(row[1], "utf8")
        raw_device_name = row[3]
        device_name = device_name_cache.get(raw_device_name)
        if device_name is None:
            device_name = decode(raw_device_name, "utf8")
            device_name_cache[raw_device_name] = device_name
        raw_device_type = row[2]
        device_type_entry = device_type_cache.get(raw_device_type)
        if device_type_entry is None:
            device_type = intern(decode(raw_device_type, "utf8"))
            device_type_code = int(DEVICE_TYPE_CODES.get(device_type, 0))
            code_rows = rows_by_code_values.setdefault(device_type_code, [])
            device_type_entry = (device_type, code_rows)
            device_type_cache[raw_device_type] = device_type_entry
        device_type, code_rows = device_type_entry
        raw_meas_type = row[4]
        meas_type_entry = meas_type_cache.get(raw_meas_type)
        if meas_type_entry is None:
            meas_type = intern(decode(raw_meas_type.upper(), "utf8"))
            meas_type_cache[raw_meas_type] = meas_type
        else:
            meas_type = meas_type_entry
        valid = row[6] == b"1"
        status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
        idx_list[row_pos] = idx
        name_list[row_pos] = name
        device_type_list[row_pos] = device_type
        device_name_list[row_pos] = device_name
        meas_type_list[row_pos] = meas_type
        weight_list[row_pos] = float(row[5])
        valid_list[row_pos] = valid
        value_list[row_pos] = float(row[7])
        status_list[row_pos] = status
        code_rows.append(row_pos)
        row_pos += 1
    if row_pos != count:
        raise RuntimeError(f"{source} Measurement parser read {row_pos} rows, expected {count}")
    rows_by_code = {int(code): np.asarray(rows, dtype=np.int64) for code, rows in rows_by_code_values.items()}
    return _build_ppc(
        source=source,
        idx=np.asarray(idx_list, dtype=np.int64),
        name=np.asarray(name_list, dtype=object),
        device_type=np.asarray(device_type_list, dtype=object),
        device_name=np.asarray(device_name_list, dtype=object),
        meas_type=np.asarray(meas_type_list, dtype=object),
        weight=np.asarray(weight_list, dtype=np.float64),
        valid=np.asarray(valid_list, dtype=bool),
        value=np.asarray(value_list, dtype=np.float64),
        status=np.asarray(status_list, dtype=np.int16),
        rows_by_device_type_code=rows_by_code,
    )


def _parse_general_measurement_text(raw: str, source: Path) -> Dict:
    header = None
    header_index = None
    in_measurement = False
    rows = []
    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        first = raw_line[0] if raw_line else ""
        if not in_measurement:
            if first == "<" and raw_line.strip() == "<Measurement>":
                in_measurement = True
            continue
        if first == "@":
            header = raw_line[1:].split()
            header_index = {name: idx for idx, name in enumerate(header)}
            missing = [name for name in _REQUIRED_COLUMNS if name not in header_index]
            if missing:
                raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
            continue
        if first == "#":
            if header is None or header_index is None:
                raise RuntimeError(f"{source} Measurement data appears before the header")
            fields = raw_line[1:].split(maxsplit=len(header) - 1)
            if len(fields) < len(header):
                raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
            rows.append(fields)
            continue
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "</Measurement>":
            break
        raise SyntaxError(f"Invalid Measurement row at line {line_no} in {source}")
    if not in_measurement:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    if header is None or header_index is None:
        raise RuntimeError(f"{source} Measurement block does not contain a header")
    count = len(rows)
    idx_values = np.empty(count, dtype=np.int64)
    name_values = np.empty(count, dtype=object)
    device_type_values = np.empty(count, dtype=object)
    device_name_values = np.empty(count, dtype=object)
    meas_type_values = np.empty(count, dtype=object)
    weight_values = np.empty(count, dtype=np.float64)
    valid_values = np.empty(count, dtype=bool)
    value_values = np.empty(count, dtype=np.float64)
    status_values = np.empty(count, dtype=np.int16)
    rows_by_code_values = {}
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
    for row_pos, fields in enumerate(rows):
        device_type = intern(fields[device_type_col])
        meas_type = intern(fields[meas_type_col].upper())
        valid = fields[valid_col] == "1"
        status = (
            normalize_measurement_status(fields[status_col], valid=valid)
            if status_col >= 0
            else MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
        )
        if not measurement_status_is_active(status):
            valid = False
        idx_values[row_pos] = int(fields[idx_col])
        name_values[row_pos] = fields[name_col]
        device_type_values[row_pos] = device_type
        device_name_values[row_pos] = fields[device_name_col]
        meas_type_values[row_pos] = meas_type
        weight_values[row_pos] = float(fields[weight_col])
        valid_values[row_pos] = valid
        value_values[row_pos] = float(fields[value_col])
        status_values[row_pos] = status
        rows_by_code_values.setdefault(int(DEVICE_TYPE_CODES.get(device_type, 0)), []).append(row_pos)
    rows_by_code = {int(code): np.asarray(rows, dtype=np.int64) for code, rows in rows_by_code_values.items()}
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
    )


def _build_meas_ppc_from_bytes(raw: bytes, source: Path) -> Dict:
    header_start = raw.find(b"@")
    if header_start < 0:
        raise RuntimeError(f"{source} Measurement block does not contain a header")
    header_end = raw.find(b"\n", header_start)
    header_line = raw[header_start + 1 : header_end if header_end >= 0 else len(raw)].decode("utf8").split()
    if tuple(header_line) == _STANDARD_HEADER:
        block = _find_measurement_block(raw, source)
        return _parse_standard_block_lines(block, source)
    return _parse_general_measurement_text(raw.decode("utf8"), source)


def build_meas_ppc_from_e_file(file_path) -> Dict:
    """Read a measurement file directly into a PPC-style NumPy dictionary."""
    file_key = _file_cache_key(file_path)
    with _MEAS_PPC_CACHE_LOCK:
        cached = _MEAS_PPC_CACHE.get(file_key[0])
        if cached is not None and cached[0] == file_key:
            return cached[1]

    ppc = _build_meas_ppc_from_bytes(file_key[0].read_bytes(), file_key[0])
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
