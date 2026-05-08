from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from model.meas_model import Measurement, MeasurementList, MeasurementTable, measurement_table_from_measurements


@dataclass(frozen=True)
class ActiveMeasurementView:
    source_table: MeasurementTable
    measurements: MeasurementList
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
    )


def measurement_table_take(table: MeasurementTable, rows: Sequence[int]) -> MeasurementTable:
    row_idx = np.asarray(rows, dtype=np.int64)
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
        return MeasurementList(list(measurements), table, normalized=normalized)
    return MeasurementList(list(measurements), normalized=normalized)


def rows_by_device_type_code(table: MeasurementTable) -> Dict[int, np.ndarray]:
    codes = np.asarray(table.device_type_code, dtype=np.int16)
    result: Dict[int, np.ndarray] = {}
    for code in np.unique(codes):
        result[int(code)] = np.flatnonzero(codes == code)
    return result


def build_measurement_plan_table(
    measurements: Sequence[Measurement],
    *,
    device_pos_by_type_code: Mapping[int, Mapping[str, int]],
    meas_kind_by_type_code: Mapping[int, Mapping[str, int]],
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
) -> MeasurementPlanTable:
    table = measurement_table_for(measurements, table_builder)
    try:
        measurements.table = table
    except AttributeError:
        pass
    row = np.arange(len(table.idx), dtype=np.int64)
    device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
    meas_kind = np.empty(row.size, dtype=np.int16)
    meas_kind.fill(-1)
    device_pos = np.empty(row.size, dtype=np.int64)
    device_pos.fill(-1)

    for code in np.unique(device_type_code):
        code_int = int(code)
        rows = np.flatnonzero(device_type_code == code)
        kind_map = meas_kind_by_type_code.get(code_int)
        if kind_map:
            meas_kind[rows] = np.fromiter(
                (kind_map.get(str(meas_type), -1) for meas_type in table.meas_type[rows]),
                dtype=np.int16,
                count=rows.size,
            )
        pos_map = device_pos_by_type_code.get(code_int)
        if pos_map:
            device_pos[rows] = np.fromiter(
                (pos_map.get(str(device_name), -1) for device_name in table.device_name[rows]),
                dtype=np.int64,
                count=rows.size,
            )

    return MeasurementPlanTable(
        table=table,
        row=row,
        device_type_code=device_type_code,
        meas_kind=meas_kind,
        device_pos=device_pos,
        handled=(meas_kind >= 0) & (device_pos >= 0),
    )


def build_active_measurement_view(
    measurements: Sequence[Measurement],
    table_builder: Optional[Callable[[Sequence[Measurement]], MeasurementTable]] = None,
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
        if isinstance(measurements, MeasurementList):
            active_measurements = measurements
            active_measurements.table = active_table
        else:
            active_measurements = MeasurementList(
                list(measurements),
                active_table,
                normalized=getattr(measurements, "normalized", False),
            )
    else:
        source_rows = np.flatnonzero(active_mask)
        active_table = measurement_table_take(table, source_rows)
        active_measurements = MeasurementList(
            [measurements[int(row)] for row in source_rows],
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
        measurements = MeasurementList(
            list(view.measurements),
            view.table,
            normalized=getattr(view.measurements, "normalized", False),
        )
        measurements.table = view.table
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
    active_additions = MeasurementList(
        [additions[int(row)] for row in active_rows_in_additions],
        measurement_table_take(additions_table, active_rows_in_additions),
        normalized=getattr(view.measurements, "normalized", False),
    )
    measurements = MeasurementList(
        list(view.measurements) + list(active_additions),
        normalized=getattr(view.measurements, "normalized", False),
    )
    active_table = concat_measurement_tables(view.table, active_additions.table)
    measurements.table = active_table
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
    sides: Tuple[str, ...],
) -> MeasurementPartitions:
    table = measurement_table_for(measurements, table_builder)
    try:
        measurements.table = table
    except AttributeError:
        pass
    rows: Dict[str, list] = {side: [] for side in sides}
    fallback = side_by_device_type or {}
    if len(table.idx) == len(measurements):
        for row, code in enumerate(table.device_type_code):
            side = side_by_device_type_code.get(int(code))
            if side is None and fallback:
                side = fallback.get(str(table.device_type[row]))
            if side in rows:
                rows[side].append(row)
    else:
        for row, meas in enumerate(measurements):
            side = side_by_device_type_code.get(int(table.device_type_code[row]))
            if side is None and fallback:
                side = fallback.get(meas.device_type)
            if side in rows:
                rows[side].append(row)
    normalized = getattr(measurements, "normalized", False)
    measurement_views: Dict[str, MeasurementList] = {}
    row_arrays: Dict[str, np.ndarray] = {}
    for side in sides:
        row_array = np.asarray(rows[side], dtype=np.int64)
        row_arrays[side] = row_array
        measurement_views[side] = MeasurementList(
            [measurements[int(row)] for row in row_array],
            measurement_table_take(table, row_array),
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
            addition_list = MeasurementList(
                list(side_additions[side]),
                measurement_table_for(side_additions[side], table_builder),
                normalized=getattr(current, "normalized", normalized),
            )
            merged = MeasurementList(
                list(current) + list(addition_list),
                normalized=getattr(current, "normalized", normalized),
            )
            merged.table = concat_measurement_tables(current.table, addition_list.table)
            measurement_views[side] = merged
        else:
            measurement_views[side] = MeasurementList(
                list(current),
                current.table,
                normalized=getattr(current, "normalized", normalized),
            )
            measurement_views[side].table = current.table
    return MeasurementPartitions(measurements=measurement_views, rows=row_arrays)
