import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from efile_read import _split_data_row
from paths import resolve_project_file
from model.meas_type import DEVICE_TYPE_CODES, MEAS_TYPE_CODES, MEAS_TYPE_NAMES
from model.meas_model import (
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
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

_DEVICE_TYPE_CODES_BYTES = {name.encode("utf8"): int(code) for name, code in DEVICE_TYPE_CODES.items()}
_MEAS_TYPE_CODES_BYTES = {name.encode("utf8"): int(code) for name, code in MEAS_TYPE_CODES.items()}
_MEAS_PPC_CACHE = {}
_MEAS_PPC_CACHE_LOCK = threading.Lock()
_REQUIRED_COLUMNS = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
_REQUIRED_COLUMNS_BYTES = tuple(name.encode("utf8") for name in _REQUIRED_COLUMNS)


def clear_meas_ppc_cache(file_path=None) -> None:
    with _MEAS_PPC_CACHE_LOCK:
        if file_path is None:
            _MEAS_PPC_CACHE.clear()
            return
        path = resolve_project_file(file_path).resolve()
        for key in list(_MEAS_PPC_CACHE):
            key_path = key[0] if isinstance(key, tuple) and key else key
            if key_path == path:
                _MEAS_PPC_CACHE.pop(key, None)


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
    name: Optional[np.ndarray],
    device_type: Optional[np.ndarray],
    device_name: Optional[np.ndarray],
    meas_type: Optional[np.ndarray],
    weight: np.ndarray,
    valid: np.ndarray,
    value: np.ndarray,
    status: np.ndarray,
    include_strings: bool = True,
    rows_by_device_type_code: Optional[Dict[int, np.ndarray]] = None,
    device_type_code: Optional[np.ndarray] = None,
    meas_type_code: Optional[np.ndarray] = None,
    device_name_id: Optional[np.ndarray] = None,
    device_names: Optional[np.ndarray] = None,
    device_name_id_by_name: Optional[Dict[object, int]] = None,
    include_matrix: bool = True,
) -> Dict:
    count = int(idx.size)
    if device_type_code is None:
        if device_type is None:
            raise RuntimeError("device_type_code is required when device_type strings are omitted")
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
        if meas_type is None:
            raise RuntimeError("meas_type_code is required when meas_type strings are omitted")
        meas_type_code = _meas_type_codes_from_names(meas_type)
    else:
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
    if device_name_id is None or device_names is None:
        if device_name is None:
            raise RuntimeError("device_name_id and device_names are required when device_name strings are omitted")
        device_name_id, device_names = _encode_device_names(device_name)
    else:
        device_name_id = np.asarray(device_name_id, dtype=np.int32)
        device_names = np.asarray(device_names, dtype=object)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    value = np.asarray(value, dtype=np.float64)
    status = np.asarray(status, dtype=np.int16)
    if include_strings:
        name = np.asarray(name if name is not None else (), dtype=object)
        device_type = np.asarray(device_type if device_type is not None else (), dtype=object)
        device_name = np.asarray(device_name if device_name is not None else (), dtype=object)
        meas_type = np.asarray(meas_type if meas_type is not None else (), dtype=object)
        if device_name.size != count and device_name_id.size == count and device_names.size:
            device_name = device_names[device_name_id.astype(np.intp, copy=False)]
    else:
        name = np.asarray([], dtype=object)
        device_type = np.asarray([], dtype=object)
        device_name = np.asarray([], dtype=object)
        meas_type = np.asarray([], dtype=object)
    if device_name_id_by_name is None:
        device_name_id_by_name = dict(
            zip(device_names.astype(object, copy=False).tolist(), range(int(device_names.size)))
        )
    angle_mask = np.isin(meas_type_code, ANGLE_MEASUREMENT_TYPE_CODES)
    if include_matrix:
        meas = np.zeros((count, len(MEAS_COLS)), dtype=np.float64)
        if count:
            meas[:, MEAS_COLS["idx"]] = idx.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["device_type_code"]] = device_type_code.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["device_name_id"]] = device_name_id.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["meas_type_code"]] = meas_type_code.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["weight"]] = weight
            meas[:, MEAS_COLS["valid"]] = valid.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["value"]] = value
            meas[:, MEAS_COLS["status"]] = status.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["angle_mask"]] = angle_mask.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["source_row"]] = np.arange(count, dtype=np.float64)
    else:
        meas = None
    return {
        "format": "meas_ppc_v1",
        "source": str(source),
        "meas": meas,
        "meas_cols": MEAS_COLS,
        "meas_type_codes": MEAS_TYPE_CODES,
        "meas_type_names": MEAS_TYPE_NAMES,
        "device_type_codes": DEVICE_TYPE_CODES,
        "device_type_names": {code: name for name, code in DEVICE_TYPE_CODES.items()},
        "idx_array": idx,
        "weight_array": weight,
        "valid_array": valid,
        "value_array": value,
        "status_array": status,
        "device_type_code_array": device_type_code,
        "device_name_id_array": device_name_id,
        "meas_type_code_array": meas_type_code,
        "angle_mask_array": angle_mask,
        "name": name,
        "device_type": device_type,
        "device_name": device_name,
        "device_names": device_names,
        "device_name_id_by_name": device_name_id_by_name,
        "meas_type": meas_type,
        "rows_by_device_type_code": rows_by_device_type_code,
        "normalized": False,
    }


def _empty_meas_ppc(source: Path, *, include_matrix: bool = True) -> Dict:
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
        include_matrix=include_matrix,
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


def _normalize_status_text(status_text, valid: bool) -> int:
    if isinstance(status_text, str):
        text = status_text.strip().upper()
        if not text:
            return MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
        if text == "0":
            return MEAS_STATUS_NORMAL
        if text == "1":
            return MEAS_STATUS_INVALID
        if text == "2":
            return MEAS_STATUS_PSEUDO
        if text == "3":
            return MEAS_STATUS_REMOVED
    return normalize_measurement_status(status_text, valid=valid)


def _normalize_status_bytes(status_text: bytes, valid: bool) -> int:
    text = status_text.strip().upper()
    if not text:
        return MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
    if text == b"0":
        return MEAS_STATUS_NORMAL
    if text == b"1":
        return MEAS_STATUS_INVALID
    if text == b"2":
        return MEAS_STATUS_PSEUDO
    if text == b"3":
        return MEAS_STATUS_REMOVED
    return _normalize_status_text(text.decode("utf8"), valid)


def _measurement_block_name(line: str) -> Optional[str]:
    if not line.startswith("<") or not line.endswith(">"):
        return None
    name = line[1:-1].strip()
    if name.startswith("/"):
        name = name[1:].strip()
    if " lv " in name:
        name = name.split(" lv ", 1)[0].strip()
    return name


def _measurement_block_name_bytes(line: bytes) -> Optional[bytes]:
    if not line.startswith(b"<") or not line.endswith(b">"):
        return None
    name = line[1:-1].strip()
    if name.startswith(b"/"):
        name = name[1:].strip()
    marker = b" lv "
    if marker in name:
        name = name.split(marker, 1)[0].strip()
    return name


def _initial_measurement_capacity(source: Path) -> int:
    """Estimate a growable row capacity without scanning the measurement file twice."""
    try:
        return max(1024, int(source.stat().st_size // 96) + 128)
    except OSError:
        return 1024


def _grow_measurement_arrays(arrays: Tuple[np.ndarray, ...], new_size: int) -> Tuple[np.ndarray, ...]:
    grown = []
    for array in arrays:
        array.resize(new_size, refcheck=False)
        grown.append(array)
    return tuple(grown)


def _build_meas_ppc_from_measurement_file_array_only(
    source: Path,
    *,
    include_matrix: bool = True,
) -> Dict:
    """Read measurement rows into numeric arrays using one variable-width byte scan."""
    header = ()
    idx_col = device_type_col = device_name_col = meas_type_col = -1
    weight_col = valid_col = value_col = -1
    status_col = -1
    required_max_col = -1
    split_limit = -1
    raw_split_limit = -1
    raw_idx_col = raw_device_type_col = raw_device_name_col = raw_meas_type_col = -1
    raw_weight_col = raw_valid_col = raw_value_col = raw_status_col = -1
    standard_raw_no_status = False
    has_status = False
    capacity = _initial_measurement_capacity(source)
    idx_values = np.empty(capacity, dtype=np.int64)
    weight_values = np.empty(capacity, dtype=np.float64)
    valid_values = np.empty(capacity, dtype=bool)
    value_values = np.empty(capacity, dtype=np.float64)
    status_values = np.empty(capacity, dtype=np.int16)
    device_type_code_values = np.empty(capacity, dtype=np.int16)
    meas_type_code_values = np.empty(capacity, dtype=np.int16)
    device_name_id = np.empty(capacity, dtype=np.int32)
    row_count = 0
    device_names_list = []
    device_name_lookup = {}

    intern = sys.intern
    to_int = int
    to_float = float
    device_code_get = _DEVICE_TYPE_CODES_BYTES.get
    meas_code_get = _MEAS_TYPE_CODES_BYTES.get
    device_name_lookup_get = device_name_lookup.get
    device_names_list_append = device_names_list.append
    status_is_active = measurement_status_is_active
    last_device_type = None
    last_device_type_code = 0
    last_device_name = None
    last_device_name_id = -1
    last_meas_type = None
    last_meas_type_code = 0
    in_measurement = False

    try:
        with open(source, mode="rb") as fp:
            for line_no, raw_line in enumerate(fp, start=1):
                first = raw_line[:1]
                if first == b"#" and in_measurement:
                    text = None
                else:
                    line = raw_line.strip()
                    if not line:
                        continue
                    first = line[:1]
                    if first == b"<":
                        block_name = _measurement_block_name_bytes(line)
                        if not line.startswith(b"</") and block_name == b"Measurement":
                            in_measurement = True
                            header = ()
                        elif line.startswith(b"</") and block_name == b"Measurement" and in_measurement:
                            break
                        elif in_measurement and line.startswith(b"</"):
                            break
                        continue
                    if first == b"@" and in_measurement:
                        header = tuple(line[1:].split())
                        header_index = {name: idx for idx, name in enumerate(header)}
                        missing = [name.decode("utf8") for name in _REQUIRED_COLUMNS_BYTES if name not in header_index]
                        if missing:
                            raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
                        idx_col = header_index[b"idx"]
                        device_type_col = header_index[b"dev_type"]
                        device_name_col = header_index[b"dev_name"]
                        meas_type_col = header_index[b"meas_type"]
                        weight_col = header_index[b"weight"]
                        valid_col = header_index[b"valid"]
                        value_col = header_index[b"value"]
                        status_col = header_index.get(b"status", -1)
                        has_status = status_col >= 0
                        required_max_col = max(
                            idx_col,
                            device_type_col,
                            device_name_col,
                            meas_type_col,
                            weight_col,
                            valid_col,
                            value_col,
                        )
                        split_limit = status_col + 1 if has_status else required_max_col
                        raw_split_limit = split_limit + 1
                        raw_idx_col = idx_col + 1
                        raw_device_type_col = device_type_col + 1
                        raw_device_name_col = device_name_col + 1
                        raw_meas_type_col = meas_type_col + 1
                        raw_weight_col = weight_col + 1
                        raw_valid_col = valid_col + 1
                        raw_value_col = value_col + 1
                        raw_status_col = status_col + 1
                        standard_raw_no_status = (
                            not has_status
                            and idx_col == 0
                            and device_type_col == 2
                            and device_name_col == 3
                            and meas_type_col == 4
                            and weight_col == 5
                            and valid_col == 6
                            and value_col == 7
                        )
                        continue
                    if first != b"#" or not in_measurement:
                        continue
                    text = line[1:]

                if not header:
                    raise RuntimeError(f"{source} Measurement data appears before header at line {line_no}")
                if standard_raw_no_status and first == b"#" and raw_line[:1] == b"#":
                    fields = raw_line.split(None, 8)
                    if len(fields) <= 8:
                        raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                    if row_count >= capacity:
                        capacity *= 2
                        grow_arrays = (
                            idx_values,
                            weight_values,
                            valid_values,
                            value_values,
                            status_values,
                            device_type_code_values,
                            meas_type_code_values,
                            device_name_id,
                        )
                        grown = _grow_measurement_arrays(grow_arrays, capacity)
                        (
                            idx_values,
                            weight_values,
                            valid_values,
                            value_values,
                            status_values,
                            device_type_code_values,
                            meas_type_code_values,
                            device_name_id,
                        ) = grown[:8]

                    idx_values[row_count] = to_int(fields[1])
                    device_type = fields[3]
                    if device_type == last_device_type:
                        device_code = last_device_type_code
                    else:
                        device_code = device_code_get(device_type, 0)
                        last_device_type = device_type
                        last_device_type_code = device_code
                    device_type_code_values[row_count] = device_code

                    device_name = fields[4]
                    if device_name == last_device_name:
                        name_id = last_device_name_id
                    else:
                        name_id = device_name_lookup_get(device_name)
                        if name_id is None:
                            device_name_text = intern(device_name.decode("utf8"))
                            name_id = len(device_names_list)
                            device_name_lookup[device_name] = name_id
                            device_names_list_append(device_name_text)
                        last_device_name = device_name
                        last_device_name_id = name_id
                    device_name_id[row_count] = name_id

                    meas_type = fields[5]
                    if meas_type == last_meas_type:
                        meas_code = last_meas_type_code
                    else:
                        meas_code = meas_code_get(meas_type, 0)
                        if meas_code == 0:
                            meas_code = meas_code_get(meas_type.upper(), 0)
                        last_meas_type = meas_type
                        last_meas_type_code = meas_code
                    meas_type_code_values[row_count] = meas_code

                    weight_field = fields[6]
                    weight_values[row_count] = 1.0 if weight_field == b"1.0" or weight_field == b"1" else to_float(weight_field)
                    valid = fields[7] == b"1"
                    value_values[row_count] = to_float(fields[8])
                    valid_values[row_count] = valid
                    status_values[row_count] = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                    row_count += 1
                    continue
                if first == b"#" and raw_line[:1] == b"#":
                    fields = raw_line.split(None, raw_split_limit)
                    idx_pos = raw_idx_col
                    device_type_pos = raw_device_type_col
                    device_name_pos = raw_device_name_col
                    meas_type_pos = raw_meas_type_col
                    weight_pos = raw_weight_col
                    valid_pos = raw_valid_col
                    value_pos = raw_value_col
                    status_pos = raw_status_col
                    required_len = required_max_col + 1
                else:
                    if text is None:
                        text = raw_line[1:]
                    fields = text.split(None, split_limit)
                    idx_pos = idx_col
                    device_type_pos = device_type_col
                    device_name_pos = device_name_col
                    meas_type_pos = meas_type_col
                    weight_pos = weight_col
                    valid_pos = valid_col
                    value_pos = value_col
                    status_pos = status_col
                    required_len = required_max_col
                if len(fields) <= required_len:
                    raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                if row_count >= capacity:
                    capacity *= 2
                    grow_arrays = (
                        idx_values,
                        weight_values,
                        valid_values,
                        value_values,
                        status_values,
                        device_type_code_values,
                        meas_type_code_values,
                        device_name_id,
                    )
                    grown = _grow_measurement_arrays(grow_arrays, capacity)
                    (
                        idx_values,
                        weight_values,
                        valid_values,
                        value_values,
                        status_values,
                        device_type_code_values,
                        meas_type_code_values,
                        device_name_id,
                    ) = grown[:8]

                idx_values[row_count] = to_int(fields[idx_pos])
                device_type = fields[device_type_pos]
                if device_type == last_device_type:
                    device_code = last_device_type_code
                else:
                    device_code = device_code_get(device_type, 0)
                    last_device_type = device_type
                    last_device_type_code = device_code
                device_type_code_values[row_count] = device_code
                device_name = fields[device_name_pos]
                if device_name == last_device_name:
                    name_id = last_device_name_id
                else:
                    name_id = device_name_lookup_get(device_name)
                    if name_id is None:
                        device_name_text = intern(device_name.decode("utf8"))
                        name_id = len(device_names_list)
                        device_name_lookup[device_name] = name_id
                        device_names_list_append(device_name_text)
                    last_device_name = device_name
                    last_device_name_id = name_id
                device_name_id[row_count] = name_id
                meas_type = fields[meas_type_pos]
                if meas_type == last_meas_type:
                    meas_code = last_meas_type_code
                else:
                    meas_code = meas_code_get(meas_type, 0)
                    if meas_code == 0:
                        meas_code = meas_code_get(meas_type.upper(), 0)
                    last_meas_type = meas_type
                    last_meas_type_code = meas_code
                meas_type_code_values[row_count] = meas_code
                weight_field = fields[weight_pos]
                weight_values[row_count] = 1.0 if weight_field == b"1.0" or weight_field == b"1" else to_float(weight_field)
                valid = fields[valid_pos] == b"1"
                value_values[row_count] = to_float(fields[value_pos])
                if has_status and len(fields) > status_pos:
                    status = _normalize_status_bytes(fields[status_pos], valid)
                    if not status_is_active(status):
                        valid = False
                else:
                    status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                valid_values[row_count] = valid
                status_values[row_count] = status
                row_count += 1
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc

    if not header:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    if row_count == 0:
        return _empty_meas_ppc(source, include_matrix=include_matrix)
    idx_array = idx_values[:row_count]
    device_type_code_array = device_type_code_values[:row_count]
    meas_type_code_array = meas_type_code_values[:row_count]
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_array == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_array)
    }
    device_name_id_by_name = dict(zip(device_names_list, range(len(device_names_list))))
    return _build_ppc(
        source=source,
        idx=idx_array,
        name=None,
        device_type=None,
        device_name=None,
        meas_type=None,
        weight=weight_values[:row_count],
        valid=valid_values[:row_count],
        value=value_values[:row_count],
        status=status_values[:row_count],
        include_strings=False,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_array,
        meas_type_code=meas_type_code_array,
        device_name_id=device_name_id[:row_count],
        device_names=np.asarray(device_names_list, dtype=object),
        device_name_id_by_name=device_name_id_by_name,
        include_matrix=include_matrix,
    )


def _build_meas_ppc_from_measurement_file(
    source: Path,
    *,
    include_strings: bool = True,
    include_matrix: bool = True,
) -> Dict:
    if not include_strings:
        return _build_meas_ppc_from_measurement_file_array_only(source, include_matrix=include_matrix)
    header = ()
    idx_col = name_col = device_type_col = device_name_col = meas_type_col = -1
    weight_col = valid_col = value_col = -1
    status_col = -1
    required_max_col = -1
    idx_values = []
    name_values = [] if include_strings else None
    device_type_values = [] if include_strings else None
    device_name_values = [] if include_strings else None
    meas_type_values = [] if include_strings else None
    weight_values = []
    valid_values = []
    value_values = []
    status_values = []
    device_type_code_values = []
    meas_type_code_values = []
    device_name_id = []
    device_names_list = []
    device_name_lookup = {}

    intern = sys.intern
    to_int = int
    to_float = float
    split_data_row = _split_data_row
    device_code_get = DEVICE_TYPE_CODES.get
    meas_code_get = MEAS_TYPE_CODES.get
    idx_append = idx_values.append
    name_append = name_values.append if include_strings else None
    device_type_append = device_type_values.append if include_strings else None
    device_name_append = device_name_values.append if include_strings else None
    meas_type_append = meas_type_values.append if include_strings else None
    weight_append = weight_values.append
    valid_append = valid_values.append
    value_append = value_values.append
    status_append = status_values.append
    device_type_code_append = device_type_code_values.append
    meas_type_code_append = meas_type_code_values.append
    device_name_id_append = device_name_id.append
    device_name_lookup_get = device_name_lookup.get
    device_names_list_append = device_names_list.append
    status_is_active = measurement_status_is_active
    in_measurement = False

    try:
        with open(source, mode="rt", encoding="utf8") as fp:
            for line_no, raw_line in enumerate(fp, start=1):
                first = raw_line[0] if raw_line else ""
                if first == "#" and in_measurement:
                    text = raw_line[1:]
                else:
                    line = raw_line.strip()
                    if not line:
                        continue
                    first = line[0]
                    if first == "<":
                        block_name = _measurement_block_name(line)
                        if not line.startswith("</") and block_name == "Measurement":
                            in_measurement = True
                            header = ()
                        elif line.startswith("</") and block_name == "Measurement" and in_measurement:
                            break
                        elif in_measurement and line.startswith("</"):
                            break
                        continue
                    if first == "@" and in_measurement:
                        header = tuple(line[1:].split())
                        header_index = {name: idx for idx, name in enumerate(header)}
                        missing = [name for name in _REQUIRED_COLUMNS if name not in header_index]
                        if missing:
                            raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
                        idx_col = header_index["idx"]
                        name_col = header_index["name"]
                        device_type_col = header_index["dev_type"]
                        device_name_col = header_index["dev_name"]
                        meas_type_col = header_index["meas_type"]
                        weight_col = header_index["weight"]
                        valid_col = header_index["valid"]
                        value_col = header_index["value"]
                        status_col = header_index.get("status", -1)
                        required_max_col = max(
                            idx_col,
                            name_col,
                            device_type_col,
                            device_name_col,
                            meas_type_col,
                            weight_col,
                            valid_col,
                            value_col,
                        )
                        continue
                    if first != "#" or not in_measurement:
                        continue
                    text = line[1:]

                fields = text.split() if "'" not in text else split_data_row(text)
                if not header:
                    raise RuntimeError(f"{source} Measurement data appears before header at line {line_no}")
                if len(fields) <= required_max_col:
                    raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                idx_append(to_int(fields[idx_col]))
                if include_strings:
                    name_append(fields[name_col])
                device_type = fields[device_type_col]
                device_code = device_code_get(device_type, 0)
                if include_strings:
                    device_type_append(intern(device_type))
                device_name = fields[device_name_col]
                name_id = device_name_lookup_get(device_name)
                if name_id is None:
                    device_name = intern(device_name)
                    name_id = len(device_names_list)
                    device_name_lookup[device_name] = name_id
                    device_names_list_append(device_name)
                else:
                    device_name = device_names_list[name_id]
                if include_strings:
                    device_name_append(device_name)
                meas_type = fields[meas_type_col]
                meas_code = meas_code_get(meas_type, 0)
                if meas_code == 0:
                    meas_type = meas_type.upper()
                    meas_code = meas_code_get(meas_type, 0)
                if include_strings:
                    meas_type_append(intern(meas_type))
                weight_append(to_float(fields[weight_col]))
                valid = fields[valid_col] == "1"
                value_append(to_float(fields[value_col]))
                if status_col >= 0 and len(fields) > status_col:
                    status = _normalize_status_text(fields[status_col], valid)
                    if not status_is_active(status):
                        valid = False
                else:
                    status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                valid_append(valid)
                status_append(status)
                device_type_code_append(device_code)
                meas_type_code_append(meas_code)
                device_name_id_append(name_id)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc

    if not header:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    if not idx_values:
        return _empty_meas_ppc(source, include_matrix=include_matrix)
    idx_array = np.asarray(idx_values, dtype=np.int64)
    device_type_code_array = np.asarray(device_type_code_values, dtype=np.int16)
    meas_type_code_array = np.asarray(meas_type_code_values, dtype=np.int16)
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_array == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_array)
    }
    return _build_ppc(
        source=source,
        idx=idx_array,
        name=np.asarray(name_values, dtype=object) if include_strings else None,
        device_type=np.asarray(device_type_values, dtype=object) if include_strings else None,
        device_name=np.asarray(device_name_values, dtype=object) if include_strings else None,
        meas_type=np.asarray(meas_type_values, dtype=object) if include_strings else None,
        weight=np.asarray(weight_values, dtype=np.float64),
        valid=np.asarray(valid_values, dtype=bool),
        value=np.asarray(value_values, dtype=np.float64),
        status=np.asarray(status_values, dtype=np.int16),
        include_strings=include_strings,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_array,
        meas_type_code=meas_type_code_array,
        device_name_id=np.asarray(device_name_id, dtype=np.int32),
        device_names=np.asarray(device_names_list, dtype=object),
        device_name_id_by_name=device_name_lookup,
        include_matrix=include_matrix,
    )


def build_meas_ppc_from_efile_rows(file_path, rows) -> Dict:
    """Build measurement ppc from E rows that are already loaded in memory."""
    path = resolve_project_file(file_path).resolve()
    return _build_meas_ppc_from_rows_dict(rows, path)


def build_meas_ppc_from_e_file(
    file_path,
    *,
    include_strings: bool = True,
    use_cache: bool = True,
    include_matrix: bool = True,
) -> Dict:
    """Read a measurement file directly into a PPC-style NumPy dictionary."""
    file_key = _file_cache_key(file_path)
    if not use_cache:
        return _build_meas_ppc_from_measurement_file(
            file_key[0],
            include_strings=bool(include_strings),
            include_matrix=bool(include_matrix),
        )
    cache_key = (file_key[0], bool(include_strings), bool(include_matrix))
    with _MEAS_PPC_CACHE_LOCK:
        cached = _MEAS_PPC_CACHE.get(cache_key)
        if cached is not None and cached[0] == file_key:
            return cached[1]

    ppc = _build_meas_ppc_from_measurement_file(
        file_key[0],
        include_strings=bool(include_strings),
        include_matrix=bool(include_matrix),
    )
    with _MEAS_PPC_CACHE_LOCK:
        _MEAS_PPC_CACHE[cache_key] = (file_key, ppc)
    return ppc


def copy_meas_ppc(ppc: Dict) -> Dict:
    """Return a shallow PPC copy with mutable measurement arrays copied."""
    copied = dict(ppc)
    for key in (
        "meas",
        "idx_array",
        "weight_array",
        "valid_array",
        "value_array",
        "status_array",
        "device_type_code_array",
        "device_name_id_array",
        "meas_type_code_array",
        "angle_mask_array",
        "device_pos",
        "scale",
        "from_pos",
        "to_pos",
        "available",
    ):
        value = ppc.get(key)
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
    return copied


def measurement_table_from_meas_ppc(ppc: Dict, *, include_strings: bool = True) -> MeasurementTable:
    meas = ppc.get("meas")
    cols = ppc.get("meas_cols", MEAS_COLS)
    has_meas = isinstance(meas, np.ndarray) and meas.ndim == 2
    idx_array = ppc.get("idx_array")
    if has_meas:
        row_count = int(meas.shape[0])
        if not isinstance(idx_array, np.ndarray) or int(idx_array.size) != row_count:
            idx_array = meas[:, cols["idx"]].astype(np.int64, copy=False)
    else:
        if not isinstance(idx_array, np.ndarray):
            raise RuntimeError("meas PPC requires idx_array when meas matrix is omitted")
        row_count = int(idx_array.size)
    device_type_code = ppc.get("device_type_code_array")
    if not isinstance(device_type_code, np.ndarray) or int(device_type_code.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires device_type_code_array when meas matrix is omitted")
        device_type_code = meas[:, cols["device_type_code"]].astype(np.int16, copy=False)
    angle_mask = ppc.get("angle_mask_array")
    if not isinstance(angle_mask, np.ndarray) or int(angle_mask.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires angle_mask_array when meas matrix is omitted")
        angle_mask = meas[:, cols["angle_mask"]].astype(bool, copy=False)
    device_name_id = ppc.get("device_name_id_array")
    if not isinstance(device_name_id, np.ndarray) or int(device_name_id.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires device_name_id_array when meas matrix is omitted")
        device_name_id = meas[:, cols["device_name_id"]].astype(np.int64, copy=False)
    meas_type_code = ppc.get("meas_type_code_array")
    if not isinstance(meas_type_code, np.ndarray) or int(meas_type_code.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires meas_type_code_array when meas matrix is omitted")
        meas_type_code = meas[:, cols["meas_type_code"]].astype(np.int16, copy=False)
    weight = ppc.get("weight_array")
    if not isinstance(weight, np.ndarray) or int(weight.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires weight_array when meas matrix is omitted")
        weight = meas[:, cols["weight"]]
    value = ppc.get("value_array")
    if not isinstance(value, np.ndarray) or int(value.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires value_array when meas matrix is omitted")
        value = meas[:, cols["value"]]
    valid = ppc.get("valid_array")
    if not isinstance(valid, np.ndarray) or int(valid.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires valid_array when meas matrix is omitted")
        valid = meas[:, cols["valid"]].astype(bool, copy=False)
    status = ppc.get("status_array")
    if not isinstance(status, np.ndarray) or int(status.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires status_array when meas matrix is omitted")
        status = meas[:, cols["status"]].astype(np.int16, copy=False)
    device_pos = ppc.get("device_pos")
    if isinstance(device_pos, np.ndarray) and int(device_pos.size) == row_count:
        device_pos = device_pos.astype(np.int64, copy=False)
    else:
        device_pos = None
    if include_strings:
        name = np.asarray(ppc.get("name", ()), dtype=object)
        device_type = np.asarray(ppc.get("device_type", ()), dtype=object)
        device_name = np.asarray(ppc.get("device_name", ()), dtype=object)
        meas_type = np.asarray(ppc.get("meas_type", ()), dtype=object)
    else:
        name = np.asarray([], dtype=object)
        device_type = np.asarray([], dtype=object)
        device_name = np.asarray([], dtype=object)
        meas_type = np.asarray([], dtype=object)
    table = MeasurementTable(
        idx=idx_array,
        name=name,
        device_type=device_type,
        device_name=device_name,
        meas_type=meas_type,
        weight=np.asarray(weight, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        value=np.asarray(value, dtype=np.float64),
        device_type_code=device_type_code.astype(np.int16, copy=False),
        angle_mask=angle_mask.astype(bool, copy=False),
        status_code=np.asarray(status, dtype=np.int16),
        rows_by_device_type_code=ppc.get("rows_by_device_type_code"),
        device_name_id=device_name_id.astype(np.int64, copy=False),
        meas_type_code=meas_type_code.astype(np.int16, copy=False),
        device_pos=device_pos,
    )
    for key in ("scale", "from_pos", "to_pos", "available"):
        value = ppc.get(key)
        if isinstance(value, np.ndarray) and int(value.size) == row_count:
            setattr(table, key, value)
    return table


def measurement_list_from_meas_ppc(ppc: Dict) -> TableBackedMeasurementList:
    return TableBackedMeasurementList(
        measurement_table_from_meas_ppc(ppc),
        normalized=bool(ppc.get("normalized", False)),
    )


def sync_meas_ppc_from_measurement_table(ppc: Dict, table: MeasurementTable) -> None:
    meas = ppc.get("meas")
    cols = ppc.get("meas_cols", MEAS_COLS)
    if isinstance(meas, np.ndarray) and meas.ndim == 2 and meas.shape[0] == table.idx.size:
        meas[:, cols["weight"]] = np.asarray(table.weight, dtype=np.float64)
        meas[:, cols["valid"]] = np.asarray(table.valid, dtype=np.float64)
        meas[:, cols["value"]] = np.asarray(table.value, dtype=np.float64)
        meas[:, cols["status"]] = np.asarray(table.status_code, dtype=np.float64)
        meas[:, cols["angle_mask"]] = np.asarray(table.angle_mask, dtype=np.float64)
    ppc["weight_array"] = np.asarray(table.weight, dtype=np.float64)
    ppc["valid_array"] = np.asarray(table.valid, dtype=bool)
    ppc["value_array"] = np.asarray(table.value, dtype=np.float64)
    ppc["status_array"] = np.asarray(table.status_code, dtype=np.int16)
    ppc["normalized"] = bool(getattr(table, "normalized", ppc.get("normalized", False)))


def meas_ppc_active_mask(ppc: Dict) -> np.ndarray:
    meas = ppc.get("meas")
    if isinstance(meas, np.ndarray) and meas.ndim == 2:
        valid = meas[:, MEAS_COLS["valid"]] != 0.0
        weight = meas[:, MEAS_COLS["weight"]]
        status = meas[:, MEAS_COLS["status"]].astype(np.int16, copy=False)
    else:
        valid = np.asarray(ppc.get("valid_array", ()), dtype=bool)
        weight = np.asarray(ppc.get("weight_array", ()), dtype=np.float64)
        status = np.asarray(ppc.get("status_array", ()), dtype=np.int16)
    return (
        valid
        & (weight > 0.0)
        & ~np.isin(status, np.asarray([MEAS_STATUS_INVALID, MEAS_STATUS_REMOVED], dtype=np.int16))
    )
