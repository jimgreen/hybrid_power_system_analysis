"""Shared switching-device boundary and result field handling."""

from collections.abc import Mapping

import numpy as np


_MISSING = (None, "", "-")


def _present(value) -> bool:
    return value not in _MISSING


def _value(source, name, default=None):
    if isinstance(source, Mapping):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    return value if _present(value) else default


def normalize_switching_device_fields(device, source=None) -> None:
    """Populate canonical setpoint/result fields while retaining legacy status."""
    source = device if source is None else source
    status = _value(source, "status", None)
    closed_status = _value(source, "closed_status", status)
    closed_status_set = _value(source, "closed_status_set", None)
    if closed_status_set is None:
        closed_status_set = _value(source, "status_set", None)
    if closed_status_set is None:
        closed_status_set = status if status is not None else closed_status
    if closed_status is None:
        closed_status = closed_status_set
    if status is None:
        status = closed_status

    device.status = int(float(1 if status is None else status))
    device.closed_status_set = int(float(1 if closed_status_set is None else closed_status_set))
    device.closed_status = int(float(1 if closed_status is None else closed_status))


def switching_status_columns(table_rows, columns):
    """Return legacy, boundary, and result status vectors for one E table."""
    count = len(table_rows)
    legacy = np.ones(count, dtype=np.float64)

    def overlay(base, names):
        values = np.asarray(base, dtype=np.float64).copy()
        for name in reversed(tuple(names)):
            col = columns.get(name)
            if col is None:
                continue
            if hasattr(table_rows, "raw_column"):
                raw = np.asarray(table_rows.raw_column(col, None), dtype=object)
            else:
                raw = np.asarray(
                    [row[col] if col < len(row) else None for row in table_rows],
                    dtype=object,
                )
            present = np.asarray([_present(value) for value in raw], dtype=bool)
            if np.any(present):
                values[present] = raw[present].astype(np.float64)
        return values

    legacy = overlay(legacy, ("status",))
    closed_status = overlay(legacy, ("closed_status",))
    closed_status_set = overlay(legacy, ("closed_status_set", "status_set"))
    return legacy, closed_status_set, closed_status


def ensure_switching_status_array(rows, columns):
    """Upgrade an older PPC switching table to the canonical column width."""
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        array = np.zeros((0, max(columns.values()) + 1), dtype=np.float64)
    required_width = max(columns.values()) + 1
    original_width = array.shape[1]
    if original_width >= required_width:
        return array

    upgraded = np.zeros((array.shape[0], required_width), dtype=np.float64)
    upgraded[:, :original_width] = array
    legacy = (
        upgraded[:, columns["status"]]
        if columns["status"] < original_width
        else np.ones(array.shape[0], dtype=np.float64)
    )
    for field in ("closed_status_set", "closed_status"):
        if columns[field] >= original_width:
            upgraded[:, columns[field]] = legacy
    return upgraded


def apply_lf_closed_status_boundaries(ppc, columns, table_keys=("switch", "break")) -> None:
    """Project switching setpoints onto the LF topology and result boundary."""
    for key in table_keys:
        rows = ensure_switching_status_array(
            ppc.get(key, np.zeros((0, max(columns.values()) + 1), dtype=np.float64)),
            columns,
        )
        if rows.size:
            commanded = rows[:, columns["closed_status_set"]]
            rows[:, columns["closed_status"]] = commanded
        ppc[key] = rows


def finalize_lf_closed_status_results(rows, columns) -> np.ndarray:
    """Write the topology state used by LF to the canonical result column."""
    result = ensure_switching_status_array(rows, columns)
    return result
