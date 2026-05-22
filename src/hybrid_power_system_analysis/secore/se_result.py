from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from model.meas_model import (
    MEAS_STATUS_PSEUDO,
    BadDataItem,
    EstimateResult,
    Measurement,
    MeasurementTable,
    is_pseudo_measurement,
    measurement_table_status_code,
)
from model.meas_type import DEVICE_TYPE_NAMES, MEAS_TYPE_NAMES


def normalize_seresult_result_mode(result_mode: str) -> str:
    mode = str(result_mode or "full").strip().lower()
    aliases = {
        "all": "full",
        "full": "full",
        "complete": "full",
        "array": "array",
        "arrays": "array",
        "ppc": "array",
        "summary": "summary",
        "brief": "summary",
        "minimal": "summary",
        "none": "none",
        "skip": "none",
        "raw": "none",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported SEResult result_mode: {result_mode!r}")
    return aliases[mode]


def build_seresult_summary(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    all_measurements: Optional[Iterable[Measurement]] = None,
) -> "SEResult":
    if not result.measurements and result.measurement_table is not None:
        all_table = getattr(all_measurements, "table", None)
        return build_seresult_summary_from_table(
            result,
            bad_items=bad_items,
            all_measurement_table=all_table,
        )
    se_result = SEResult()
    bad_items = list(bad_items or ())
    active_ids = {id(meas) for meas in result.measurements}
    prefiltered_count = 0
    if all_measurements is not None:
        for measurement in all_measurements:
            if id(measurement) not in active_ids and SEResult._prefiltered_reason(measurement):
                prefiltered_count += 1
    pseudo_count = sum(1 for measurement in result.measurements if SEResult._is_pseudo_measurement(measurement))
    bad_count = len(bad_items)
    normal_count = max(0, len(result.measurements) - pseudo_count - bad_count)
    obs = result.observability
    se_result.statistics = SEResult.StatisticsTable(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        objective=float(result.objective),
        max_correction=float(result.max_correction),
        residual_inf=float(result.residual_inf),
        observable=bool(obs.observable),
        rank=int(obs.rank),
        state_count=int(obs.state_count),
        measurement_count=int(obs.measurement_count),
        deficiency=int(obs.deficiency),
        prefiltered_measurement_count=prefiltered_count,
        pseudo_measurement_count=pseudo_count,
        bad_data_count=bad_count,
        normal_measurement_count=normal_count,
    )
    return se_result


class _MeasurementResultTable:
    __slots__ = ("rows",)

    def __init__(self, rows: Optional[Iterable["SEResult.MeasurementRow"]] = None) -> None:
        self.rows = list(rows or ())

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]

    def append(
        self,
        measurement: Measurement,
        *,
        estimated_value: Optional[float] = None,
        residual: Optional[float] = None,
        normalized_residual: Optional[float] = None,
        reason: str = "",
        source: str = "",
    ) -> None:
        self.rows.append(
            SEResult.MeasurementRow.from_measurement(
                measurement,
                estimated_value=estimated_value,
                residual=residual,
                normalized_residual=normalized_residual,
                reason=reason,
                source=source,
            )
        )

    def append_row(self, row: "SEResult.MeasurementRow") -> None:
        self.rows.append(row)

    def extend(self, rows: Iterable["SEResult.MeasurementRow"]) -> None:
        self.rows.extend(rows)

    def to_dicts(self) -> List[dict]:
        return [asdict(row) for row in self.rows]


@dataclass
class SEResult:
    @dataclass
    class StatisticsTable:
        converged: bool = False
        iterations: int = 0
        objective: float = 0.0
        max_correction: float = 0.0
        residual_inf: float = 0.0
        observable: bool = False
        rank: int = 0
        state_count: int = 0
        measurement_count: int = 0
        deficiency: int = 0
        prefiltered_measurement_count: int = 0
        pseudo_measurement_count: int = 0
        bad_data_count: int = 0
        normal_measurement_count: int = 0

    @dataclass
    class MeasurementRow:
        idx: int
        name: str
        device_type: str
        device_name: str
        meas_type: str
        weight: float
        valid: bool
        value: float
        estimated_value: Optional[float] = None
        residual: Optional[float] = None
        normalized_residual: Optional[float] = None
        reason: str = ""
        source: str = ""

        @classmethod
        def from_measurement(
            cls,
            measurement: Measurement,
            *,
            estimated_value: Optional[float] = None,
            residual: Optional[float] = None,
            normalized_residual: Optional[float] = None,
            reason: str = "",
            source: str = "",
        ) -> "SEResult.MeasurementRow":
            return cls(
                idx=int(measurement.idx),
                name=str(measurement.name),
                device_type=str(measurement.device_type),
                device_name=str(measurement.device_name),
                meas_type=str(measurement.meas_type),
                weight=float(measurement.weight),
                valid=bool(measurement.valid),
                value=float(measurement.value),
                estimated_value=None if estimated_value is None else float(estimated_value),
                residual=None if residual is None else float(residual),
                normalized_residual=None if normalized_residual is None else float(normalized_residual),
                reason=str(reason),
                source=str(source),
            )

        @classmethod
        def from_table_row(
            cls,
            table: MeasurementTable,
            row: int,
            *,
            estimated_value: Optional[float] = None,
            residual: Optional[float] = None,
            normalized_residual: Optional[float] = None,
            reason: str = "",
            source: str = "",
        ) -> "SEResult.MeasurementRow":
            pos = int(row)
            idx = int(table.idx[pos])
            name = str(table.name[pos]) if int(table.name.size) > pos else f"m{idx}"
            if int(table.device_type.size) > pos:
                device_type = str(table.device_type[pos])
            else:
                device_type = DEVICE_TYPE_NAMES.get(int(table.device_type_code[pos]), "")
            if int(table.device_name.size) > pos:
                device_name = str(table.device_name[pos])
            else:
                device_pos = getattr(table, "device_pos", None)
                device_pos = None if device_pos is None else np.asarray(device_pos)
                device_name = (
                    f"pos:{int(device_pos[pos])}"
                    if device_pos is not None and int(device_pos.size) > pos and int(device_pos[pos]) >= 0
                    else ""
                )
            if int(table.meas_type.size) > pos:
                meas_type = str(table.meas_type[pos])
            else:
                meas_type_code = getattr(table, "meas_type_code", None)
                meas_type_code = None if meas_type_code is None else np.asarray(meas_type_code)
                meas_type = (
                    MEAS_TYPE_NAMES.get(int(meas_type_code[pos]), "")
                    if meas_type_code is not None and int(meas_type_code.size) > pos
                    else ""
                )
            return cls(
                idx=idx,
                name=name,
                device_type=device_type,
                device_name=device_name,
                meas_type=meas_type,
                weight=float(table.weight[pos]),
                valid=bool(table.valid[pos]),
                value=float(table.value[pos]),
                estimated_value=None if estimated_value is None else float(estimated_value),
                residual=None if residual is None else float(residual),
                normalized_residual=None if normalized_residual is None else float(normalized_residual),
                reason=str(reason),
                source=str(source),
            )

    class PrefilteredMeasurementTable(_MeasurementResultTable):
        pass

    class PseudoMeasurementTable(_MeasurementResultTable):
        pass

    class BadDataTable(_MeasurementResultTable):
        def append_bad_data_item(self, item: BadDataItem) -> None:
            self.append(
                item.measurement,
                estimated_value=item.estimated_value,
                residual=item.residual,
                normalized_residual=item.normalized_residual,
                source="bad_data",
            )

    class NormalMeasurementTable(_MeasurementResultTable):
        pass

    statistics: StatisticsTable = field(default_factory=lambda: SEResult.StatisticsTable())
    prefiltered_measurements: PrefilteredMeasurementTable = field(
        default_factory=lambda: SEResult.PrefilteredMeasurementTable()
    )
    pseudo_measurements: PseudoMeasurementTable = field(default_factory=lambda: SEResult.PseudoMeasurementTable())
    bad_data: BadDataTable = field(default_factory=lambda: SEResult.BadDataTable())
    normal_measurements: NormalMeasurementTable = field(default_factory=lambda: SEResult.NormalMeasurementTable())

    _STATISTICS_COLUMNS = (
        "converged",
        "iterations",
        "objective",
        "max_correction",
        "residual_inf",
        "observable",
        "rank",
        "state_count",
        "measurement_count",
        "deficiency",
        "prefiltered_measurement_count",
        "pseudo_measurement_count",
        "bad_data_count",
        "normal_measurement_count",
    )
    _MEASUREMENT_COLUMNS = (
        "idx",
        "name",
        "device_type",
        "device_name",
        "meas_type",
        "weight",
        "valid",
        "value",
        "estimated_value",
        "residual",
        "normalized_residual",
        "reason",
        "source",
    )

    @staticmethod
    def _is_pseudo_measurement(measurement: Measurement) -> bool:
        return is_pseudo_measurement(measurement)

    @staticmethod
    def _prefiltered_reason(measurement: Measurement) -> str:
        if not bool(measurement.valid):
            return "invalid"
        if float(measurement.weight) <= 0.0:
            return "zero weight"
        return ""

    @classmethod
    def from_estimate_result(
        cls,
        result: EstimateResult,
        *,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        prefiltered_measurements: Optional[Iterable[object]] = None,
        all_measurements: Optional[Iterable[Measurement]] = None,
    ) -> "SEResult":
        se_result = cls()
        bad_items = list(bad_items or ())
        normalized = np.asarray(normalized_residual, dtype=np.float64) if normalized_residual is not None else None
        measurement_ids = {id(meas) for meas in result.measurements}
        bad_ids = {id(item.measurement) for item in bad_items}
        bad_indexes = {int(item.measurement.idx) for item in bad_items}

        prefiltered_rows = cls._normalize_prefiltered_rows(
            prefiltered_measurements,
            all_measurements,
            measurement_ids,
        )
        for measurement, reason in prefiltered_rows:
            se_result.prefiltered_measurements.append(measurement, reason=reason, source="prefiltered")

        for pos, measurement in enumerate(result.measurements):
            estimated = cls._array_value(result.z_est, pos)
            residual = cls._array_value(result.residual, pos)
            normalized_value = cls._array_value(normalized, pos)
            if cls._is_pseudo_measurement(measurement):
                se_result.pseudo_measurements.append(
                    measurement,
                    estimated_value=estimated,
                    residual=residual,
                    normalized_residual=normalized_value,
                    source="pseudo",
                )
                continue
            if id(measurement) in bad_ids or int(measurement.idx) in bad_indexes:
                continue
            se_result.normal_measurements.append(
                measurement,
                estimated_value=estimated,
                residual=residual,
                normalized_residual=normalized_value,
                source="normal",
            )

        for item in bad_items:
            se_result.bad_data.append_bad_data_item(item)

        obs = result.observability
        se_result.statistics = cls.StatisticsTable(
            converged=bool(result.converged),
            iterations=int(result.iterations),
            objective=float(result.objective),
            max_correction=float(result.max_correction),
            residual_inf=float(result.residual_inf),
            observable=bool(obs.observable),
            rank=int(obs.rank),
            state_count=int(obs.state_count),
            measurement_count=int(obs.measurement_count),
            deficiency=int(obs.deficiency),
            prefiltered_measurement_count=len(se_result.prefiltered_measurements),
            pseudo_measurement_count=len(se_result.pseudo_measurements),
            bad_data_count=len(se_result.bad_data),
            normal_measurement_count=len(se_result.normal_measurements),
        )
        return se_result

    @classmethod
    def _normalize_prefiltered_rows(
        cls,
        prefiltered_measurements: Optional[Iterable[object]],
        all_measurements: Optional[Iterable[Measurement]],
        active_measurement_ids: set,
    ) -> List[Tuple[Measurement, str]]:
        if prefiltered_measurements is not None:
            rows = []
            for item in prefiltered_measurements:
                if isinstance(item, tuple):
                    measurement, reason = item
                else:
                    measurement = item
                    reason = cls._prefiltered_reason(measurement)
                rows.append((measurement, str(reason)))
            return rows

        if all_measurements is None:
            return []
        rows = []
        for measurement in all_measurements:
            if id(measurement) in active_measurement_ids:
                continue
            reason = cls._prefiltered_reason(measurement)
            if reason:
                rows.append((measurement, reason))
        return rows

    @staticmethod
    def _array_value(values, pos: int) -> Optional[float]:
        if values is None:
            return None
        if pos >= len(values):
            return None
        return float(values[pos])

    def to_e_text(self) -> str:
        blocks = [
            self._statistics_block_to_e(),
            self._measurement_block_to_e("SEResultPrefilteredMeasurement", self.prefiltered_measurements),
            self._measurement_block_to_e("SEResultPseudoMeasurement", self.pseudo_measurements),
            self._measurement_block_to_e("SEResultBadData", self.bad_data),
            self._measurement_block_to_e("SEResultNormalMeasurement", self.normal_measurements),
        ]
        return "\n\n".join(blocks) + "\n"

    def write_e_file(self, file_path: Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_e_text(), encoding="utf-8")

    def _statistics_block_to_e(self) -> str:
        columns = self._STATISTICS_COLUMNS
        values = [getattr(self.statistics, name) for name in columns]
        return self._block_to_e("SEResultStatistics", columns, [values])

    def _measurement_block_to_e(self, block_name: str, table: _MeasurementResultTable) -> str:
        rows = [
            [getattr(row, name) for name in self._MEASUREMENT_COLUMNS]
            for row in table
        ]
        return self._block_to_e(block_name, self._MEASUREMENT_COLUMNS, rows)

    @classmethod
    def _block_to_e(cls, block_name: str, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
        lines = [f"<{block_name}>", "@ " + " ".join(columns)]
        for row in rows:
            lines.append("# " + " ".join(cls._e_cell(value) for value in row))
        lines.append(f"</{block_name}>")
        return "\n".join(lines)

    @staticmethod
    def _e_cell(value) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (np.bool_,)):
            return "1" if bool(value) else "0"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return format(float(value), ".12g")
        text = str(value)
        if not text:
            return "-"
        return "_".join(text.split())


def _table_prefiltered_reason(table: MeasurementTable, row: int) -> str:
    pos = int(row)
    if not bool(table.valid[pos]):
        return "invalid"
    if float(table.weight[pos]) <= 0.0:
        return "zero weight"
    return ""


def _active_source_mask(table: MeasurementTable, active_source_rows: Optional[Sequence[int]]) -> np.ndarray:
    active = np.zeros(int(table.idx.size), dtype=bool)
    if active_source_rows is None:
        return active
    rows = np.asarray(active_source_rows, dtype=np.int64)
    if rows.size == 0:
        return active
    rows = rows[(rows >= 0) & (rows < active.size)]
    if rows.size:
        active[rows.astype(np.intp, copy=False)] = True
    return active


def _bad_row_positions(bad_items: Optional[Sequence[BadDataItem]], row_count: int) -> set:
    positions = set()
    for item in bad_items or ():
        row_pos = int(getattr(item, "row_pos", -1))
        if 0 <= row_pos < row_count:
            positions.add(row_pos)
    return positions


def _seresult_statistics_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]],
    all_measurement_table: Optional[MeasurementTable],
    active_source_rows: Optional[Sequence[int]],
) -> "SEResult.StatisticsTable":
    table = result.measurement_table
    status_code = measurement_table_status_code(table)
    pseudo_count = int(np.count_nonzero(status_code == MEAS_STATUS_PSEUDO))
    bad_count = len(bad_items or ())
    normal_count = max(0, int(table.idx.size) - pseudo_count - bad_count)
    prefiltered_count = 0
    if all_measurement_table is not None:
        if active_source_rows is None and int(all_measurement_table.idx.size) == int(table.idx.size):
            active_mask = np.ones(int(all_measurement_table.idx.size), dtype=bool)
        else:
            active_mask = _active_source_mask(all_measurement_table, active_source_rows)
        prefiltered = (
            (~active_mask)
            & (
                (~np.asarray(all_measurement_table.valid, dtype=bool))
                | (np.asarray(all_measurement_table.weight, dtype=np.float64) <= 0.0)
            )
        )
        prefiltered_count = int(np.count_nonzero(prefiltered))
    obs = result.observability
    return SEResult.StatisticsTable(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        objective=float(result.objective),
        max_correction=float(result.max_correction),
        residual_inf=float(result.residual_inf),
        observable=bool(obs.observable),
        rank=int(obs.rank),
        state_count=int(obs.state_count),
        measurement_count=int(obs.measurement_count),
        deficiency=int(obs.deficiency),
        prefiltered_measurement_count=prefiltered_count,
        pseudo_measurement_count=pseudo_count,
        bad_data_count=bad_count,
        normal_measurement_count=normal_count,
    )


def build_seresult_summary_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    all_measurement_table: Optional[MeasurementTable] = None,
    active_source_rows: Optional[Sequence[int]] = None,
) -> "SEResult":
    """Build a summary result directly from measurement arrays."""
    if result.measurement_table is None:
        raise RuntimeError("SEResult summary requires result.measurement_table")
    se_result = SEResult()
    se_result.statistics = _seresult_statistics_from_table(
        result,
        bad_items=bad_items,
        all_measurement_table=all_measurement_table,
        active_source_rows=active_source_rows,
    )
    return se_result


def build_seresult_full_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    normalized_residual: Optional[Sequence[float]] = None,
    all_measurement_table: Optional[MeasurementTable] = None,
    active_source_rows: Optional[Sequence[int]] = None,
) -> "SEResult":
    """Build a full SEResult from array-backed measurement tables."""
    table = result.measurement_table
    if table is None:
        raise RuntimeError("SEResult full output requires result.measurement_table")
    se_result = SEResult()
    bad_items = list(bad_items or ())
    normalized = np.asarray(normalized_residual, dtype=np.float64) if normalized_residual is not None else None
    bad_positions = _bad_row_positions(bad_items, int(table.idx.size))
    status_code = measurement_table_status_code(table)

    if all_measurement_table is not None:
        if active_source_rows is None and int(all_measurement_table.idx.size) == int(table.idx.size):
            active_mask = np.ones(int(all_measurement_table.idx.size), dtype=bool)
        else:
            active_mask = _active_source_mask(all_measurement_table, active_source_rows)
        prefiltered = np.flatnonzero(
            (~active_mask)
            & (
                (~np.asarray(all_measurement_table.valid, dtype=bool))
                | (np.asarray(all_measurement_table.weight, dtype=np.float64) <= 0.0)
            )
        )
        for row in prefiltered.tolist():
            reason = _table_prefiltered_reason(all_measurement_table, row)
            if reason:
                se_result.prefiltered_measurements.append_row(
                    SEResult.MeasurementRow.from_table_row(
                        all_measurement_table,
                        row,
                        reason=reason,
                        source="prefiltered",
                    )
                )

    for pos in range(int(table.idx.size)):
        estimated = SEResult._array_value(result.z_est, pos)
        residual = SEResult._array_value(result.residual, pos)
        normalized_value = SEResult._array_value(normalized, pos)
        row = SEResult.MeasurementRow.from_table_row(
            table,
            pos,
            estimated_value=estimated,
            residual=residual,
            normalized_residual=normalized_value,
            source="pseudo" if int(status_code[pos]) == MEAS_STATUS_PSEUDO else "normal",
        )
        if int(status_code[pos]) == MEAS_STATUS_PSEUDO:
            se_result.pseudo_measurements.append_row(row)
            continue
        if pos in bad_positions:
            continue
        se_result.normal_measurements.append_row(row)

    for item in bad_items:
        row_pos = int(getattr(item, "row_pos", -1))
        if 0 <= row_pos < int(table.idx.size):
            row = SEResult.MeasurementRow.from_table_row(
                table,
                row_pos,
                estimated_value=getattr(item, "estimated_value", None),
                residual=getattr(item, "residual", None),
                normalized_residual=getattr(item, "normalized_residual", None),
                source="bad_data",
            )
            se_result.bad_data.append_row(row)
        elif getattr(item, "measurement", None) is not None:
            se_result.bad_data.append_bad_data_item(item)

    se_result.statistics = _seresult_statistics_from_table(
        result,
        bad_items=bad_items,
        all_measurement_table=all_measurement_table,
        active_source_rows=active_source_rows,
    )
    return se_result
