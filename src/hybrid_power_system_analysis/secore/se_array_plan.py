import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from model.meas_model import (
    Measurement,
    MeasurementList,
    MeasurementTable,
    MeasurementTableView,
    TableBackedMeasurementList,
    MeasurementView,
    measurement_table_from_measurements,
    measurement_table_status_code,
)


@dataclass(frozen=True)
class ActiveMeasurementView:
    source_table: MeasurementTable
    measurements: object
    table: MeasurementTable
    source_rows: np.ndarray
    z: np.ndarray
    weight: np.ndarray
    angle_mask: np.ndarray
    all_active: bool
    rows_by_device_type_code: Dict[int, np.ndarray]


@dataclass(frozen=True)
class MeasurementPartitions:
    measurements: Dict[str, MeasurementList]
    rows: Dict[str, np.ndarray]


@dataclass(frozen=True)
class MeasurementPlanTable:
    table: MeasurementTable
    row: np.ndarray
    device_type_code: np.ndarray
    meas_kind: np.ndarray
    device_pos: np.ndarray
    handled: np.ndarray


def concat_measurement_tables(head: MeasurementTable, tail: MeasurementTable) -> MeasurementTable:
    return MeasurementTable(
        idx=np.concatenate((head.idx, tail.idx)),
        name=np.concatenate((head.name, tail.name)),
        device_type=np.concatenate((head.device_type, tail.device_type)),
        device_name=np.concatenate((head.device_name, tail.device_name)),
        meas_type=np.concatenate((head.meas_type, tail.meas_type)),
        weight=np.concatenate((head.weight, tail.weight)),
        valid=np.concatenate((head.valid, tail.valid)),
        value=np.concatenate((head.value, tail.value)),
        device_type_code=np.concatenate((head.device_type_code, tail.device_type_code)),
        angle_mask=np.concatenate((head.angle_mask, tail.angle_mask)),
        status_code=np.concatenate((measurement_table_status_code(head), measurement_table_status_code(tail))),
        rows_by_device_type_code=None,
        device_name_id=_concat_optional_int_field(head, tail, "device_name_id"),
        meas_type_code=_concat_optional_int_field(head, tail, "meas_type_code"),
        device_pos=_concat_optional_int_field(head, tail, "device_pos"),
    )


def _concat_optional_int_field(head: MeasurementTable, tail: MeasurementTable, field_name: str):
    head_values = getattr(head, field_name, None)
    tail_values = getattr(tail, field_name, None)
    if head_values is None and tail_values is None:
        return None
    if head_values is None:
        head_array = np.full(head.idx.size, -1, dtype=np.int64)
    else:
        head_array = np.asarray(head_values, dtype=np.int64)
    if tail_values is None:
        tail_array = np.full(tail.idx.size, -1, dtype=np.int64)
    else:
        tail_array = np.asarray(tail_values, dtype=np.int64)
    return np.concatenate((head_array, tail_array))


def _take_optional_int_field(table: MeasurementTable, row_idx: np.ndarray, field_name: str):
    values = getattr(table, field_name, None)
    if values is None:
        return None
    values = np.asarray(values)
    if values.size != table.idx.size:
        return None
    return values[row_idx]


def _slice_cached_rows_by_device_type_code(
    cached_rows: Optional[Mapping[int, Sequence[int]]],
    row_idx: np.ndarray,
) -> Optional[Dict[int, np.ndarray]]:
    if cached_rows is None:
        return None
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if row_idx.size == 0:
        return {}
    max_row = int(np.max(row_idx))
    if max_row <= max(1024, row_idx.size * 4):
        row_to_pos = np.full(max_row + 1, -1, dtype=np.int64)
        row_to_pos[row_idx] = np.arange(row_idx.size, dtype=np.int64)
        result: Dict[int, np.ndarray] = {}
        for code, source_rows in cached_rows.items():
            source = np.asarray(source_rows, dtype=np.int64)
            if source.size == 0:
                continue
            source = source[source <= max_row]
            if source.size == 0:
                continue
            positions = row_to_pos[source]
            positions = positions[positions >= 0]
            if positions.size:
                positions.sort()
                result[int(code)] = positions
        return result
    row_to_pos = {int(row): pos for pos, row in enumerate(row_idx)}
    result = {}
    for code, source_rows in cached_rows.items():
        positions = [row_to_pos[int(row)] for row in source_rows if int(row) in row_to_pos]
        if positions:
            positions.sort()
            result[int(code)] = np.asarray(positions, dtype=np.int64)
    return result


def measurement_table_take(
    table: MeasurementTable,
    rows: Sequence[int],
    rows_by_device_type_code: Optional[Mapping[int, Sequence[int]]] = None,
) -> MeasurementTable:
    row_idx = np.asarray(rows, dtype=np.int64)
    code_rows = (
        {int(code): np.asarray(values, dtype=np.int64) for code, values in rows_by_device_type_code.items()}
        if rows_by_device_type_code is not None
        else _slice_cached_rows_by_device_type_code(
            getattr(table, "rows_by_device_type_code", None),
            row_idx,
        )
    )
    return MeasurementTable(
        idx=table.idx[row_idx],
        name=table.name[row_idx],
        device_type=table.device_type[row_idx],
        device_name=table.device_name[row_idx],
        meas_type=table.meas_type[row_idx],
        weight=table.weight[row_idx],
        valid=table.valid[row_idx],
        value=table.value[row_idx],
        device_type_code=table.device_type_code[row_idx],
        angle_mask=table.angle_mask[row_idx],
        status_code=measurement_table_status_code(table)[row_idx],
        rows_by_device_type_code=code_rows,
        device_name_id=_take_optional_int_field(table, row_idx, "device_name_id"),
        meas_type_code=_take_optional_int_field(table, row_idx, "meas_type_code"),
        device_pos=_take_optional_int_field(table, row_idx, "device_pos"),
    )


def measurement_table_for(
    measurements: Sequence[Measurement],
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
) -> MeasurementTable:
    if table_builder is not None:
        return table_builder(measurements)
    return measurement_table_from_measurements(measurements)


def copy_measurement_view(measurements: Sequence[Measurement]) -> MeasurementList:
    table = getattr(measurements, "table", None)
    normalized = getattr(measurements, "normalized", False)
    if table is not None and len(table.idx) == len(measurements):
        return TableBackedMeasurementList(table, normalized=normalized)
    return MeasurementList(list(measurements), normalized=normalized)


def take_measurement_view(
    measurements: Sequence[Measurement],
    rows: Sequence[int],
    rows_by_device_type_code: Optional[Mapping[int, Sequence[int]]] = None,
) -> MeasurementList:
    table = measurement_table_for(measurements)
    row_array = np.asarray(rows, dtype=np.int64)
    if isinstance(measurements, MeasurementView):
        source = measurements.source
        source_rows = measurements.rows[row_array]
    else:
        source = measurements
        source_rows = row_array
    return MeasurementView(
        source,
        source_rows,
        measurement_table_take(table, row_array, rows_by_device_type_code=rows_by_device_type_code),
        normalized=getattr(measurements, "normalized", False),
    )


def rows_by_device_type_code(table: MeasurementTable) -> Dict[int, np.ndarray]:
    cached = getattr(table, "rows_by_device_type_code", None)
    if cached is not None:
        return cached
    codes = np.asarray(table.device_type_code, dtype=np.int16)
    result: Dict[int, np.ndarray] = {}
    for code in np.unique(codes):
        result[int(code)] = np.flatnonzero(codes == code)
    table.rows_by_device_type_code = result
    return result


def build_measurement_plan_table(
    measurements: Sequence[Measurement],
    *,
    device_pos_by_type_code: Mapping[int, Mapping[str, int]],
    meas_kind_by_type_code: Mapping[int, Mapping[str, int]],
    device_pos_by_type_code_id: Optional[Mapping[int, np.ndarray]] = None,
    meas_kind_code_by_type_code: Optional[Mapping[int, np.ndarray]] = None,
    require_index_arrays: bool = False,
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
) -> MeasurementPlanTable:
    table = measurement_table_for(measurements, table_builder)
    try:
        measurements.table = table
    except AttributeError:
        pass
    row = np.arange(len(table.idx), dtype=np.int64)
    device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
    meas_kind = _cached_measurement_kind(
        table,
        meas_kind_by_type_code,
        meas_kind_code_by_type_code=meas_kind_code_by_type_code,
        require_index_arrays=require_index_arrays,
    )
    device_pos = _cached_measurement_device_pos(
        table,
        device_pos_by_type_code,
        device_pos_by_type_code_id=device_pos_by_type_code_id,
        require_index_arrays=require_index_arrays,
    )

    return MeasurementPlanTable(
        table=table,
        row=row,
        device_type_code=device_type_code,
        meas_kind=meas_kind,
        device_pos=device_pos,
        handled=(meas_kind >= 0) & (device_pos >= 0),
    )


def _mapping_identity_key(mapping: Mapping[int, Mapping[str, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(code), id(values)) for code, values in mapping.items()))


def _cached_measurement_device_pos(
    table: MeasurementTable,
    device_pos_by_type_code: Mapping[int, Mapping[str, int]],
    *,
    device_pos_by_type_code_id: Optional[Mapping[int, np.ndarray]] = None,
    require_index_arrays: bool = False,
) -> np.ndarray:
    id_key = (
        ()
        if device_pos_by_type_code_id is None
        else tuple(sorted((int(code), id(values)) for code, values in device_pos_by_type_code_id.items()))
    )
    key = (_mapping_identity_key(device_pos_by_type_code), id_key, bool(require_index_arrays))
    precomputed = getattr(table, "device_pos", None)
    if precomputed is not None:
        precomputed = np.asarray(precomputed, dtype=np.int64)
        if precomputed.size == table.idx.size:
            return precomputed
    cache = getattr(table, "_device_pos_plan_cache", None)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None and cached.size == table.idx.size:
            return cached
    device_pos = np.empty(table.idx.size, dtype=np.int64)
    device_pos.fill(-1)
    device_name_id = getattr(table, "device_name_id", None)
    if device_name_id is not None:
        device_name_id = np.asarray(device_name_id, dtype=np.int64)
        if device_name_id.size != table.idx.size:
            device_name_id = None
    if require_index_arrays:
        if device_name_id is None or not device_pos_by_type_code_id:
            warnings.warn(
                "Measurement device position lookup requires device_name_id and indexed device maps; "
                "string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            for code_int, code_rows in rows_by_device_type_code(table).items():
                rows = np.asarray(code_rows, dtype=np.int64)
                if rows.size == 0:
                    continue
                lookup = device_pos_by_type_code_id.get(int(code_int))
                if lookup is None or lookup.size == 0:
                    continue
                ids = device_name_id[rows]
                values = np.empty(rows.size, dtype=np.int64)
                values.fill(-1)
                in_range = (ids >= 0) & (ids < lookup.size)
                if np.any(in_range):
                    values[in_range] = lookup[ids[in_range].astype(np.intp, copy=False)]
                device_pos[rows] = values
        if cache is None:
            cache = {}
            setattr(table, "_device_pos_plan_cache", cache)
        if len(cache) > 16:
            cache.clear()
        cache[key] = device_pos
        return device_pos
    for code_int, code_rows in rows_by_device_type_code(table).items():
        rows = np.asarray(code_rows, dtype=np.int64)
        if rows.size == 0:
            continue
        pos_map = device_pos_by_type_code.get(code_int)
        if pos_map:
            # `np.fromiter` with a generator was the dominant cost here for
            # large measurement files. `tolist()` + list comprehension +
            # `np.array(..., dtype=...)` is markedly faster because it avoids
            # per-element Python <-> numpy boxing.
            names = table.device_name[rows].tolist()
            pos_get = pos_map.get
            device_pos[rows] = np.array(
                [pos_get(name, -1) for name in names],
                dtype=np.int64,
            )
    if cache is None:
        cache = {}
        setattr(table, "_device_pos_plan_cache", cache)
    if len(cache) > 16:
        cache.clear()
    cache[key] = device_pos
    return device_pos


def _cached_measurement_kind(
    table: MeasurementTable,
    meas_kind_by_type_code: Mapping[int, Mapping[str, int]],
    *,
    meas_kind_code_by_type_code: Optional[Mapping[int, np.ndarray]] = None,
    require_index_arrays: bool = False,
) -> np.ndarray:
    code_key = (
        ()
        if meas_kind_code_by_type_code is None
        else tuple(sorted((int(code), id(values)) for code, values in meas_kind_code_by_type_code.items()))
    )
    key = (_mapping_identity_key(meas_kind_by_type_code), code_key, bool(require_index_arrays))
    cache = getattr(table, "_meas_kind_plan_cache", None)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None and cached.size == table.idx.size:
            return cached
    meas_kind = np.empty(table.idx.size, dtype=np.int16)
    meas_kind.fill(-1)
    meas_type_code = getattr(table, "meas_type_code", None)
    if meas_type_code is not None:
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        if meas_type_code.size != table.idx.size:
            meas_type_code = None
    if require_index_arrays:
        if meas_type_code is None or not meas_kind_code_by_type_code:
            warnings.warn(
                "Measurement kind lookup requires meas_type_code and indexed kind maps; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            for code_int, code_rows in rows_by_device_type_code(table).items():
                rows = np.asarray(code_rows, dtype=np.int64)
                if rows.size == 0:
                    continue
                lookup = meas_kind_code_by_type_code.get(int(code_int))
                if lookup is None or lookup.size == 0:
                    continue
                codes = meas_type_code[rows].astype(np.int64, copy=False)
                values = np.empty(rows.size, dtype=np.int16)
                values.fill(-1)
                in_range = (codes >= 0) & (codes < lookup.size)
                if np.any(in_range):
                    values[in_range] = lookup[codes[in_range].astype(np.intp, copy=False)]
                meas_kind[rows] = values
        if cache is None:
            cache = {}
            setattr(table, "_meas_kind_plan_cache", cache)
        if len(cache) > 16:
            cache.clear()
        cache[key] = meas_kind
        return meas_kind
    for code_int, code_rows in rows_by_device_type_code(table).items():
        rows = np.asarray(code_rows, dtype=np.int64)
        if rows.size == 0:
            continue
        kind_map = meas_kind_by_type_code.get(code_int)
        if kind_map:
            names = table.meas_type[rows].tolist()
            kind_get = kind_map.get
            meas_kind[rows] = np.array(
                [kind_get(name, -1) for name in names],
                dtype=np.int16,
            )
    if cache is None:
        cache = {}
        setattr(table, "_meas_kind_plan_cache", cache)
    if len(cache) > 16:
        cache.clear()
    cache[key] = meas_kind
    return meas_kind


def build_active_measurement_view(
    measurements: Sequence[Measurement],
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
    *,
    materialize_measurements: bool = True,
) -> ActiveMeasurementView:
    table = measurement_table_for(measurements, table_builder)
    try:
        measurements.table = table
    except AttributeError:
        pass
    active_mask = np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)
    all_active = bool(active_mask.size == len(measurements) and np.all(active_mask))
    if all_active:
        source_rows = np.arange(active_mask.size, dtype=np.int64)
        active_table = table
        if not materialize_measurements:
            active_measurements = MeasurementTableView(
                active_table,
                normalized=getattr(measurements, "normalized", False),
            )
        elif isinstance(measurements, MeasurementList):
            active_measurements = measurements
            active_measurements.table = active_table
        else:
            source_table = getattr(measurements, "table", None)
            if source_table is not None and len(source_table.idx) == len(measurements):
                active_measurements = TableBackedMeasurementList(
                    active_table,
                    normalized=getattr(measurements, "normalized", False),
                )
            else:
                active_measurements = MeasurementList(
                    list(measurements),
                    active_table,
                    normalized=getattr(measurements, "normalized", False),
                )
    else:
        source_rows = np.flatnonzero(active_mask)
        if materialize_measurements:
            active_measurements = take_measurement_view(measurements, source_rows)
            active_table = active_measurements.table
        else:
            rows_by_code = _slice_cached_rows_by_device_type_code(
                getattr(table, "rows_by_device_type_code", None),
                source_rows,
            )
            active_table = measurement_table_take(table, source_rows, rows_by_device_type_code=rows_by_code)
            active_measurements = MeasurementTableView(
                active_table,
                normalized=getattr(measurements, "normalized", False),
            )
    return ActiveMeasurementView(
        source_table=table,
        measurements=active_measurements,
        table=active_table,
        source_rows=np.asarray(source_rows, dtype=np.int64),
        z=np.asarray(active_table.value, dtype=np.float64),
        weight=np.asarray(active_table.weight, dtype=np.float64),
        angle_mask=np.asarray(active_table.angle_mask, dtype=bool),
        all_active=all_active,
        rows_by_device_type_code=rows_by_device_type_code(active_table),
    )


def append_active_measurement_view(
    view: ActiveMeasurementView,
    additions: Sequence[Measurement],
    *,
    source_row_start: Optional[int] = None,
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
    materialize_measurements: bool = True,
) -> ActiveMeasurementView:
    if not additions:
        return view
    additions_table = measurement_table_for(additions, table_builder)
    try:
        additions.table = additions_table
    except AttributeError:
        pass
    source_start = int(len(view.source_table.idx) if source_row_start is None else source_row_start)
    active_mask = np.asarray(additions_table.valid, dtype=bool) & (np.asarray(additions_table.weight, dtype=np.float64) > 0.0)
    active_rows_in_additions = np.flatnonzero(active_mask)
    source_table = concat_measurement_tables(view.source_table, additions_table)
    if active_rows_in_additions.size == 0:
        if materialize_measurements:
            measurements = TableBackedMeasurementList(
                view.table,
                normalized=getattr(view.measurements, "normalized", False),
            )
        else:
            measurements = MeasurementTableView(
                view.table,
                normalized=getattr(view.measurements, "normalized", False),
            )
        return ActiveMeasurementView(
            source_table=source_table,
            measurements=measurements,
            table=view.table,
            source_rows=view.source_rows,
            z=view.z,
            weight=view.weight,
            angle_mask=view.angle_mask,
            all_active=False,
            rows_by_device_type_code=view.rows_by_device_type_code,
        )
    active_additions_table = measurement_table_take(additions_table, active_rows_in_additions)
    active_table = concat_measurement_tables(view.table, active_additions_table)
    if materialize_measurements:
        measurements = TableBackedMeasurementList(
            active_table,
            normalized=getattr(view.measurements, "normalized", False),
        )
    else:
        measurements = MeasurementTableView(
            active_table,
            normalized=getattr(view.measurements, "normalized", False),
        )
    source_rows = np.concatenate((view.source_rows, source_start + active_rows_in_additions.astype(np.int64, copy=False)))
    return ActiveMeasurementView(
        source_table=source_table,
        measurements=measurements,
        table=active_table,
        source_rows=source_rows.astype(np.int64, copy=False),
        z=np.asarray(active_table.value, dtype=np.float64),
        weight=np.asarray(active_table.weight, dtype=np.float64),
        angle_mask=np.asarray(active_table.angle_mask, dtype=bool),
        all_active=bool(source_rows.size == source_table.idx.size and np.all(source_rows == np.arange(source_rows.size))),
        rows_by_device_type_code=rows_by_device_type_code(active_table),
    )


def partition_measurements_by_code(
    measurements: Sequence[Measurement],
    side_by_device_type_code: Mapping[int, str],
    *,
    side_by_device_type: Optional[Mapping[str, str]] = None,
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
    rows_by_device_type_code: Optional[Mapping[int, Sequence[int]]] = None,
    as_view: bool = False,
    sides: Tuple[str, ...],
) -> MeasurementPartitions:
    table = measurement_table_for(measurements, table_builder)
    try:
        measurements.table = table
    except AttributeError:
        pass
    row_chunks: Dict[str, list] = {side: [] for side in sides}
    code_row_chunks: Dict[str, Dict[int, list]] = {side: {} for side in sides}
    fallback = side_by_device_type or {}
    code_rows = (
        rows_by_device_type_code
        if rows_by_device_type_code is not None
        else globals()["rows_by_device_type_code"](table)
    )
    for code, code_row_values in code_rows.items():
        code_row_array = np.asarray(code_row_values, dtype=np.int64)
        if code_row_array.size == 0:
            continue
        side = side_by_device_type_code.get(int(code))
        if side in row_chunks:
            row_chunks[side].append(code_row_array)
            code_row_chunks[side].setdefault(int(code), []).append(code_row_array)
            continue
        if not fallback:
            continue
        device_types = np.asarray(table.device_type, dtype=object)[code_row_array]
        for device_type in np.unique(device_types):
            fallback_side = fallback.get(str(device_type))
            if fallback_side in row_chunks:
                fallback_rows = code_row_array[device_types == device_type]
                row_chunks[fallback_side].append(fallback_rows)
                code_row_chunks[fallback_side].setdefault(int(code), []).append(fallback_rows)
    normalized = getattr(measurements, "normalized", False)
    measurement_views: Dict[str, MeasurementList] = {}
    row_arrays: Dict[str, np.ndarray] = {}
    for side in sides:
        if row_chunks[side]:
            row_array = np.concatenate(row_chunks[side]).astype(np.int64, copy=False)
            row_array.sort()
        else:
            row_array = np.array([], dtype=np.int64)
        row_arrays[side] = row_array
        local_rows_by_code: Dict[int, np.ndarray] = {}
        if row_array.size:
            for code, chunks in code_row_chunks[side].items():
                if not chunks:
                    continue
                source_rows = (
                    np.concatenate(chunks).astype(np.int64, copy=False)
                    if len(chunks) > 1
                    else np.asarray(chunks[0], dtype=np.int64)
                )
                if source_rows.size == 0:
                    continue
                local_pos = np.searchsorted(row_array, source_rows)
                in_range = local_pos < row_array.size
                valid = np.zeros(local_pos.shape, dtype=bool)
                if np.any(in_range):
                    valid[in_range] = row_array[local_pos[in_range]] == source_rows[in_range]
                local_pos = local_pos[valid].astype(np.int64, copy=False)
                if local_pos.size:
                    local_pos.sort()
                    local_rows_by_code[int(code)] = local_pos
        if as_view:
            measurement_views[side] = take_measurement_view(
                measurements,
                row_array,
                rows_by_device_type_code=local_rows_by_code,
            )
        else:
            measurement_views[side] = TableBackedMeasurementList(
                measurement_table_take(table, row_array, rows_by_device_type_code=local_rows_by_code),
                normalized=normalized,
            )
    return MeasurementPartitions(measurements=measurement_views, rows=row_arrays)


def extend_measurement_partitions(
    partitions: MeasurementPartitions,
    additions: Sequence[Measurement],
    side_by_device_type_code: Mapping[int, str],
    *,
    row_offset: Optional[int] = None,
    side_by_device_type: Optional[Mapping[str, str]] = None,
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
    sides: Tuple[str, ...],
) -> MeasurementPartitions:
    if not additions:
        return partitions
    additions_table = measurement_table_for(additions, table_builder)
    try:
        additions.table = additions_table
    except AttributeError:
        pass
    rows: Dict[str, list] = {side: list(np.asarray(partitions.rows[side], dtype=np.int64).tolist()) for side in sides}
    if row_offset is None:
        existing_sizes = [max(row_values) + 1 for row_values in rows.values() if row_values]
        offset = int(max(existing_sizes) if existing_sizes else 0)
    else:
        offset = int(row_offset)
    fallback = side_by_device_type or {}
    side_additions: Dict[str, list] = {side: [] for side in sides}
    for local_row, code in enumerate(np.asarray(additions_table.device_type_code, dtype=np.int16)):
        side = side_by_device_type_code.get(int(code))
        if side is None and fallback:
            side = fallback.get(str(additions_table.device_type[local_row]))
        if side in rows:
            rows[side].append(offset + local_row)
            side_additions[side].append(additions[local_row])
    normalized = getattr(additions, "normalized", False)
    measurement_views: Dict[str, MeasurementList] = {}
    row_arrays: Dict[str, np.ndarray] = {}
    for side in sides:
        row_arrays[side] = np.asarray(rows[side], dtype=np.int64)
        current = partitions.measurements[side]
        if side_additions[side]:
            addition_table = measurement_table_for(side_additions[side], table_builder)
            measurement_views[side] = TableBackedMeasurementList(
                concat_measurement_tables(current.table, addition_table),
                normalized=getattr(current, "normalized", normalized),
            )
        else:
            measurement_views[side] = TableBackedMeasurementList(
                current.table,
                normalized=getattr(current, "normalized", normalized),
            )
    return MeasurementPartitions(measurements=measurement_views, rows=row_arrays)
