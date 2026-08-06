from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from paths import resolve_project_file

_P_UNIT_TO_SCALE = {"W": 1000.0, "kW": 1.0, "MW": 0.001}
_U_UNIT_TO_SCALE = {"mV": 1_000_000.0, "V": 1000.0, "kV": 1.0}
_I_UNIT_TO_SCALE = {"mA": 1_000_000.0, "A": 1000.0, "kA": 1.0}


def file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _optional_ppc_vector(ppc, key: str, size: int, default=np.nan) -> np.ndarray:
    """Return an optional PPC vector padded to the requested device count."""
    values = np.asarray((ppc or {}).get(key, ()), dtype=np.float64).reshape(-1)
    result = np.full(int(size), float(default), dtype=np.float64)
    if values.size:
        result[: min(result.size, values.size)] = values[: result.size]
    return result


class EFileTableRows:
    """List-compatible E table rows with cached column access."""

    __slots__ = ("rows", "columns", "_matrix", "_raw_cache")

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self._matrix = None
        self._raw_cache = {}

    def __len__(self):
        return len(self.rows)

    def __bool__(self):
        return bool(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, item):
        return self.rows[item]

    def _as_matrix(self):
        if self._matrix is not None:
            return self._matrix
        if not self.rows:
            self._matrix = np.empty((0, 0), dtype=object)
            return self._matrix
        matrix = np.asarray(self.rows, dtype=object)
        self._matrix = matrix if matrix.ndim == 2 else False
        return self._matrix

    def raw_column(self, col, default="") -> np.ndarray:
        n = len(self.rows)
        if col is None or n == 0:
            return np.full(n, default, dtype=object) if n else np.empty(0, dtype=object)
        key = (int(col), default)
        cached = self._raw_cache.get(key)
        if cached is not None:
            return cached

        matrix = self._as_matrix()
        if isinstance(matrix, np.ndarray) and col < matrix.shape[1]:
            raw = matrix[:, int(col)]
            missing = (raw == "") | (raw == None)
            if np.any(missing):
                values = raw.copy()
                values[missing] = default
            else:
                values = raw
        else:
            values = np.empty(n, dtype=object)
            for i, row in enumerate(self.rows):
                if col >= len(row):
                    values[i] = default
                else:
                    value = row[col]
                    values[i] = default if value in (None, "") else value
        self._raw_cache[key] = values
        return values


class LazyNameArray:
    """Sequence-compatible object name array that materializes on first use."""

    __slots__ = ("_raw_names", "_idx_values", "_prefix", "_array")

    def __init__(self, raw_names, idx_values: np.ndarray, prefix: str):
        self._raw_names = None if raw_names is None else np.asarray(raw_names, dtype=object).copy()
        self._idx_values = np.asarray(idx_values)
        self._prefix = str(prefix)
        self._array = None

    def __len__(self):
        return int(self._idx_values.shape[0])

    def _materialize(self) -> np.ndarray:
        if self._array is not None:
            return self._array
        if self._raw_names is None:
            names = np.asarray([f"{self._prefix}_{int(idx)}" for idx in self._idx_values], dtype=object)
        else:
            raw = self._raw_names
            names = raw.astype(str).astype(object, copy=False)
            missing = (raw == "") | (raw == None)
            if np.any(missing):
                fallback = np.asarray([f"{self._prefix}_{int(idx)}" for idx in self._idx_values], dtype=object)
                names = names.copy()
                names[missing] = fallback[missing]
        self._array = names
        return names

    def __getitem__(self, item):
        return self._materialize()[item]

    def __iter__(self):
        return iter(self._materialize())

    def __array__(self, dtype=None, copy=None):
        array = self._materialize()
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy) if copy is not None else False)
        return array.copy() if copy else array

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        return np.dtype(object)

    def astype(self, dtype, copy=True):
        return self._materialize().astype(dtype, copy=copy)

    def tolist(self):
        return self._materialize().tolist()


def _rows_for(data: Dict, table_name: str):
    table = data.get(table_name)
    if not table:
        return {}, []
    columns = {str(name): pos for pos, name in enumerate(table.get("header_list", []))}
    return columns, EFileTableRows(table.get("rows", []), columns)


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
    if hasattr(table_rows, "raw_column"):
        return table_rows.raw_column(col, default).astype(np.float64, copy=False)
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
    if hasattr(table_rows, "raw_column"):
        return (
            table_rows.raw_column(col, default)
            .astype(np.float64, copy=False)
            .astype(np.int64, copy=False)
            .astype(np.float64)
        )
    values = [None] * n
    for i in range(n):
        row = table_rows[i]
        if col >= len(row):
            values[i] = default
        else:
            v = row[col]
            values[i] = default if v in (None, "") else int(float(v))
    return np.asarray(values, dtype=np.float64)


def _code_value(value, mapping: Dict[str, int], default_label: str) -> int:
    if value in (None, ""):
        return mapping[default_label]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return mapping.get(str(value).upper(), mapping[default_label])


def _code_column(table_rows, columns, attr: str, mapping: Dict[str, int], default_label: str) -> np.ndarray:
    col = columns.get(attr)
    default = mapping[default_label]
    if col is None:
        return np.full(len(table_rows), float(default), dtype=np.float64)
    if hasattr(table_rows, "raw_column"):
        raw = table_rows.raw_column(col, default_label)
        return np.asarray([_code_value(value, mapping, default_label) for value in raw], dtype=np.float64)
    return np.asarray(
        [_code_value(_cell(row, col, default_label), mapping, default_label) for row in table_rows],
        dtype=np.float64,
    )


def _names_from_rows(table_rows, columns, prefix: str, idx_values: np.ndarray) -> np.ndarray:
    name_col = columns.get("name")
    if name_col is None:
        if hasattr(table_rows, "raw_column"):
            return LazyNameArray(None, idx_values, prefix)
        return np.asarray([f"{prefix}_{int(idx)}" for idx in idx_values], dtype=object)
    if hasattr(table_rows, "raw_column"):
        raw = table_rows.raw_column(name_col, "")
        return LazyNameArray(raw, idx_values, prefix)
    return np.asarray(
        [
            str(_cell(row, name_col, "") or f"{prefix}_{int(idx_values[pos])}")
            for pos, row in enumerate(table_rows)
        ],
        dtype=object,
    )


def _base_from_rows(data: Dict) -> Tuple[float, float, float, float, float]:
    block_name = "Model"
    columns, table_rows = _rows_for(data, block_name)
    if not table_rows:
        block_name = "PowerBase"
        columns, table_rows = _rows_for(data, block_name)
    if not table_rows:
        raise RuntimeError("E file must define <Model> with p_base, u_unit, p_unit, and i_unit")
    row = table_rows[0]
    if "p_base" not in columns:
        raise RuntimeError(f"E file <{block_name}> must define p_base")
    p_base = float(_cell(row, columns["p_base"], 0.0))
    if p_base <= 0.0:
        raise RuntimeError(f"Invalid p_base in <{block_name}>: {p_base}")

    def unit_or_legacy_scale(unit_attr: str, scale_attr: str, mapping, default_unit: str) -> float:
        if unit_attr in columns:
            unit = str(_cell(row, columns[unit_attr], default_unit)).strip()
            if unit not in mapping:
                allowed = "/".join(mapping.keys())
                raise RuntimeError(f"Invalid {unit_attr} in <{block_name}>: {unit!r}; expected one of {allowed}")
            return float(mapping[unit])
        if scale_attr not in columns:
            raise RuntimeError(f"E file <{block_name}> must define p_base, u_unit, p_unit, and i_unit")
        value = float(_cell(row, columns[scale_attr], 0.0))
        if value <= 0.0:
            raise RuntimeError(f"Invalid {scale_attr} in <{block_name}>: {value}")
        return value

    u_scale = unit_or_legacy_scale("u_unit", "u_scale", _U_UNIT_TO_SCALE, "kV")
    p_scale = unit_or_legacy_scale("p_unit", "p_scale", _P_UNIT_TO_SCALE, "kW")
    i_scale = unit_or_legacy_scale("i_unit", "i_scale", _I_UNIT_TO_SCALE, "kA")
    return p_base, u_scale, p_scale, i_scale, p_base / p_scale


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
    if callable(current_scale_by_node):
        current_scale_by_node = current_scale_by_node()
    scales = _scale_by_node(node_values.astype(np.int64, copy=False), current_scale_by_node)
    raw = _float_column(table_rows, columns, attr)
    out[:, col] = np.divide(raw, scales, out=np.zeros_like(raw), where=np.abs(scales) > 1e-12)


def _voltage_set_column(table_rows, columns, attr: str, node_values: np.ndarray, raw_vbase_by_idx: Dict[int, float]) -> np.ndarray:
    if attr not in columns:
        return np.ones(len(table_rows), dtype=np.float64)
    raw = _float_column(table_rows, columns, attr, 1.0)
    raw_vbase = _raw_vbase_by_node(node_values.astype(np.int64, copy=False), raw_vbase_by_idx)
    return np.divide(raw, raw_vbase, out=np.ones_like(raw), where=np.abs(raw_vbase) > 1e-12)


def _value(obj, attr: str, default=0.0):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_value(obj, attr: str, default: float = 0.0) -> float:
    return float(_value(obj, attr, default))


def _int_value(obj, attr: str, default: int = 0) -> int:
    return int(float(_value(obj, attr, default)))


def _has_value(devices, attr: str) -> bool:
    return any(getattr(dev, attr, None) not in (None, "") for dev in devices)


def _fill_float_column_if_present(out: np.ndarray, devices, col: int, attr: str) -> None:
    if not _has_value(devices, attr):
        return
    for row, dev in enumerate(devices):
        out[row, col] = _float_value(dev, attr)


def _name_array(devices, prefix: str) -> np.ndarray:
    return np.asarray(
        [str(getattr(dev, "name", "") or f"{prefix}_{_int_value(dev, 'idx', pos)}") for pos, dev in enumerate(devices)],
        dtype=object,
    )
