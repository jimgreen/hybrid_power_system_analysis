from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEVICE_TYPE_CODES = {
    "ACNode": 1,
    "ACBranch": 2,
    "ACTransformer": 3,
    "ACLoad": 4,
    "ACGenerator": 5,
    "ACZeroBranch": 6,
    "ACZeroBranchConstraint": 7,
    "ACSwitchConstraint": 8,
    "ACSwitch": 9,
    "ACPowerBalance": 10,
    "DCNode": 11,
    "DCBranch": 12,
    "DCSwitch": 13,
    "DCZeroBranch": 14,
    "DCZeroBranchConstraint": 15,
    "DCSwitchConstraint": 16,
    "DCGenerator": 17,
    "DCLoad": 18,
    "DCDCConverter": 19,
    "DCACConverter": 20,
    "ACACConverter": 21,
    "ACBreak": 22,
    "ACBreakConstraint": 23,
    "DCBreak": 24,
    "DCBreakConstraint": 25,
}

MEAS_STATUS_NORMAL = 0
MEAS_STATUS_INVALID = 1
MEAS_STATUS_PSEUDO = 2
MEAS_STATUS_REMOVED = 3
MEAS_STATUS_INACTIVE = frozenset((MEAS_STATUS_INVALID, MEAS_STATUS_REMOVED))
MEAS_STATUS_BY_NAME = {
    "NORMAL": MEAS_STATUS_NORMAL,
    "REAL": MEAS_STATUS_NORMAL,
    "VALID": MEAS_STATUS_NORMAL,
    "INVALID": MEAS_STATUS_INVALID,
    "DISABLED": MEAS_STATUS_INVALID,
    "PSEUDO": MEAS_STATUS_PSEUDO,
    "BAD": MEAS_STATUS_REMOVED,
    "REMOVED": MEAS_STATUS_REMOVED,
}

TERMINAL_MEASUREMENT_KIND = {"P_FROM": 0, "V_FROM": 1, "I_FROM": 2, "P_TO": 3, "V_TO": 4, "I_TO": 5}
LOAD_MEASUREMENT_KIND = {"P_LOAD": 0, "V_LOAD": 1, "I_LOAD": 2}
GEN_MEASUREMENT_KIND = {"P_GEN": 0, "V_GEN": 1, "I_GEN": 2}
GEN_CONTROL_KIND = {"V": 0, "P": 1, "I": 2}


def measurement_status_from_valid(valid: bool) -> int:
    return MEAS_STATUS_NORMAL if bool(valid) else MEAS_STATUS_INVALID


def normalize_measurement_status(status=None, *, valid: bool = True) -> int:
    if status is None:
        return measurement_status_from_valid(valid)
    if isinstance(status, str):
        text = status.strip().upper()
        if not text:
            return measurement_status_from_valid(valid)
        if text in MEAS_STATUS_BY_NAME:
            return MEAS_STATUS_BY_NAME[text]
        return int(text)
    return int(status)


def measurement_status_is_active(status: int) -> bool:
    return int(status) not in MEAS_STATUS_INACTIVE


def is_pseudo_measurement(measurement) -> bool:
    status = getattr(measurement, "status", measurement_status_from_valid(getattr(measurement, "valid", True)))
    return int(status) == MEAS_STATUS_PSEUDO


def mark_measurement_invalid(measurement) -> None:
    measurement.valid = False
    measurement.status = MEAS_STATUS_INVALID


def mark_measurement_pseudo(measurement) -> None:
    measurement.status = MEAS_STATUS_PSEUDO


@dataclass(init=False, slots=True)
class Measurement:
    idx: int
    name: str
    device_type: str
    device_name: str
    meas_type: str
    weight: float
    valid: bool
    value: float
    status: int

    def __init__(
        self,
        idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        weight: float,
        valid: bool,
        value: float,
        status=None,
    ) -> None:
        self.idx = idx
        self.name = name
        self.device_type = device_type
        self.device_name = device_name
        self.meas_type = meas_type
        self.weight = weight
        self.valid = valid
        self.value = value
        self.status = normalize_measurement_status(status, valid=valid)
        if not measurement_status_is_active(self.status):
            self.valid = False

    @property
    def device(self) -> str:
        return f"{self.device_type}:{self.device_name}"

    @classmethod
    def read_from_file(cls, file_name: Path, scale_context=None):
        # Import lazily to keep the model layer independent from estimator-specific scaling code.
        from secore.ac_se import _read_measurements_direct

        if scale_context is None:
            return _read_measurements_direct(file_name, cls)
        return _read_measurements_direct(file_name, cls, scale_context)


@dataclass
class MeasurementTable:
    idx: np.ndarray
    name: np.ndarray
    device_type: np.ndarray
    device_name: np.ndarray
    meas_type: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    value: np.ndarray
    device_type_code: np.ndarray
    angle_mask: np.ndarray
    status_code: Optional[np.ndarray] = None
    rows_by_device_type_code: Optional[Dict[int, np.ndarray]] = None


class MeasurementList(list):
    __slots__ = ("table", "normalized")

    def __init__(self, measurements=(), table: Optional[MeasurementTable] = None, normalized: bool = False):
        super().__init__(measurements)
        self.table = table
        self.normalized = bool(normalized)


class MeasurementView(MeasurementList):
    """A table-backed row view over an existing measurement sequence."""

    __slots__ = ("source", "rows")

    def __init__(
        self,
        source: Sequence[Measurement],
        rows: Sequence[int],
        table: Optional[MeasurementTable] = None,
        normalized: bool = False,
    ):
        list.__init__(self)
        self.source = source
        self.rows = np.asarray(rows, dtype=np.int64)
        self.table = table
        self.normalized = bool(normalized)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __iter__(self):
        source = self.source
        for row in self.rows:
            yield source[int(row)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.source[int(row)] for row in self.rows[index]]
        return self.source[int(self.rows[index])]

    def __contains__(self, item) -> bool:
        return any(measurement is item or measurement == item for measurement in self)

    def index(self, item, start=0, stop=None) -> int:
        size = len(self)
        start = max(0, int(start))
        stop = size if stop is None else min(size, int(stop))
        for pos in range(start, stop):
            measurement = self[pos]
            if measurement is item or measurement == item:
                return pos
        raise ValueError(f"{item!r} is not in measurement view")


def measurement_status_from_measurement(measurement: Measurement) -> int:
    return normalize_measurement_status(getattr(measurement, "status", None), valid=getattr(measurement, "valid", True))


def measurement_table_status_code(table: MeasurementTable) -> np.ndarray:
    if table.status_code is None:
        table.status_code = np.where(np.asarray(table.valid, dtype=bool), MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID).astype(
            np.int16,
            copy=False,
        )
    return table.status_code


def measurement_table_from_measurements(
    measurements: Sequence[Measurement],
    *,
    device_type_codes=DEVICE_TYPE_CODES,
    angle_measurement_types=frozenset(),
) -> MeasurementTable:
    table = getattr(measurements, "table", None)
    if table is not None:
        table_size = int(table.idx.size)
        measurement_count = len(measurements)
        if measurement_count == table_size:
            measurement_table_status_code(table)
            return table
        if table_size < measurement_count:
            tail_table = measurement_table_from_measurements(
                tuple(measurements[table_size:]),
                device_type_codes=device_type_codes,
                angle_measurement_types=angle_measurement_types,
            )
            table = MeasurementTable(
                idx=np.concatenate((table.idx, tail_table.idx)),
                name=np.concatenate((table.name, tail_table.name)),
                device_type=np.concatenate((table.device_type, tail_table.device_type)),
                device_name=np.concatenate((table.device_name, tail_table.device_name)),
                meas_type=np.concatenate((table.meas_type, tail_table.meas_type)),
                weight=np.concatenate((table.weight, tail_table.weight)),
                valid=np.concatenate((table.valid, tail_table.valid)),
                value=np.concatenate((table.value, tail_table.value)),
                device_type_code=np.concatenate((table.device_type_code, tail_table.device_type_code)),
                angle_mask=np.concatenate((table.angle_mask, tail_table.angle_mask)),
                status_code=np.concatenate((measurement_table_status_code(table), measurement_table_status_code(tail_table))),
                rows_by_device_type_code=None,
            )
            try:
                measurements.table = table
            except AttributeError:
                pass
            return table

    count = len(measurements)
    idx = np.empty(count, dtype=np.int64)
    name = np.empty(count, dtype=object)
    device_type = np.empty(count, dtype=object)
    device_name = np.empty(count, dtype=object)
    meas_type = np.empty(count, dtype=object)
    weight = np.empty(count, dtype=np.float64)
    valid = np.empty(count, dtype=bool)
    value = np.empty(count, dtype=np.float64)
    device_type_code = np.empty(count, dtype=np.int16)
    angle_mask = np.empty(count, dtype=bool)
    status_code = np.empty(count, dtype=np.int16)
    for pos, meas in enumerate(measurements):
        idx[pos] = int(meas.idx)
        name[pos] = meas.name
        device_type[pos] = meas.device_type
        device_name[pos] = meas.device_name
        meas_type[pos] = meas.meas_type
        weight[pos] = float(meas.weight)
        valid[pos] = bool(meas.valid)
        value[pos] = float(meas.value)
        device_type_code[pos] = device_type_codes.get(meas.device_type, 0)
        angle_mask[pos] = meas.meas_type in angle_measurement_types
        status_code[pos] = measurement_status_from_measurement(meas)
    return MeasurementTable(
        idx=idx,
        name=name,
        device_type=device_type,
        device_name=device_name,
        meas_type=meas_type,
        weight=weight,
        valid=valid,
        value=value,
        device_type_code=device_type_code,
        angle_mask=angle_mask,
        status_code=status_code,
    )


@dataclass
class ObservabilityResult:
    observable: bool
    rank: int
    state_count: int
    measurement_count: int
    deficiency: int
    singular_values: np.ndarray
    weak_states: List[Tuple[int, float]]


@dataclass
class EstimateResult:
    converged: bool
    iterations: int
    objective: float
    max_correction: float
    residual_inf: float
    x: np.ndarray
    z_est: np.ndarray
    residual: np.ndarray
    H: Optional[np.ndarray]
    gain: Optional[np.ndarray]
    measurements: List[Measurement]
    observability: ObservabilityResult


@dataclass
class BadDataItem:
    measurement: Measurement
    residual: float
    normalized_residual: float
    estimated_value: float
    measured_value: float


def print_iteration_header() -> None:
    print("Iteration process:")
    print("  iter objective      max_dx      norm_res    step   status")


def print_iteration(
    iteration: int,
    objective: float,
    residual_inf: float,
    max_correction: float,
    step_scale: Optional[float],
    converged: bool,
) -> None:
    step = "-" if step_scale is None else f"{step_scale:.3f}"
    status = "converged" if converged else ""
    print(
        f"  {iteration:4d} "
        f"{objective:12.6e} "
        f"{max_correction:10.3e} "
        f"{residual_inf:10.3e} "
        f"{step:>6s} "
        f"{status}"
    )
