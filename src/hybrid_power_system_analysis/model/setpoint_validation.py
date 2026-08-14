"""Shared validation for fixed boundary setpoints used by steady-state solvers."""

from __future__ import annotations

from typing import Optional

import numpy as np


DEFAULT_VOLTAGE_SETPOINT = 1.0
MIN_VOLTAGE_SETPOINT = 0.5
MAX_VOLTAGE_SETPOINT = 1.5


def sanitize_fixed_setpoints(
    values,
    active_mask,
    *,
    fallback=DEFAULT_VOLTAGE_SETPOINT,
    minimum=MIN_VOLTAGE_SETPOINT,
    maximum=MAX_VOLTAGE_SETPOINT,
    default_fallback=DEFAULT_VOLTAGE_SETPOINT,
    device_type: str,
    field: str,
    indices=None,
    side: Optional[str] = None,
) -> tuple[np.ndarray, list[dict]]:
    """Replace invalid active fixed-boundary values and describe each correction.

    ``active_mask`` is mandatory so zero placeholders on non-controlling devices
    remain untouched. ``fallback`` may be a scalar or an array matching values.
    """

    array = np.asarray(values, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if array.shape != active.shape:
        raise ValueError(
            f"fixed setpoint values/mask shape mismatch: {array.shape} != {active.shape}"
        )
    if array.size == 0:
        return array, []

    fallback_values = np.broadcast_to(
        np.asarray(fallback, dtype=np.float64), array.shape
    )
    minimum_values = np.broadcast_to(np.asarray(minimum, dtype=np.float64), array.shape)
    maximum_values = np.broadcast_to(np.asarray(maximum, dtype=np.float64), array.shape)
    invalid = active & (
        ~np.isfinite(array)
        | (array < minimum_values)
        | (array > maximum_values)
    )
    if not np.any(invalid):
        return array, []

    default_values = np.broadcast_to(
        np.asarray(default_fallback, dtype=np.float64), array.shape
    )
    usable_fallback = (
        np.isfinite(fallback_values)
        & (fallback_values >= minimum_values)
        & (fallback_values <= maximum_values)
    )
    usable_default = (
        np.isfinite(default_values)
        & (default_values >= minimum_values)
        & (default_values <= maximum_values)
    )
    replacement = np.where(
        usable_fallback,
        fallback_values,
        np.where(usable_default, default_values, DEFAULT_VOLTAGE_SETPOINT),
    )
    positions = np.flatnonzero(invalid)
    original = array[positions].copy()
    array[positions] = replacement[positions]

    if indices is None:
        index_values = positions
    else:
        index_values = np.asarray(indices)[positions]
    corrections = []
    for pos, idx, old in zip(positions.tolist(), index_values.tolist(), original.tolist()):
        item = {
            "device_type": str(device_type),
            "index": int(idx),
            "row": int(pos),
            "field": str(field),
            "original": float(old),
            "replacement": float(array[pos]),
        }
        if side is not None:
            item["side"] = str(side)
        corrections.append(item)
    return array, corrections


def append_setpoint_corrections(target: dict, corrections) -> None:
    if not corrections:
        return
    existing = target.setdefault("_setpoint_corrections", [])
    known = {
        (
            item.get("device_type"),
            item.get("index"),
            item.get("field"),
            item.get("side"),
            item.get("original"),
            item.get("replacement"),
        )
        for item in existing
    }
    for item in corrections:
        key = (
            item.get("device_type"),
            item.get("index"),
            item.get("field"),
            item.get("side"),
            item.get("original"),
            item.get("replacement"),
        )
        if key not in known:
            existing.append(item)
            known.add(key)
