from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
}

TERMINAL_MEASUREMENT_KIND = {"P_FROM": 0, "V_FROM": 1, "I_FROM": 2, "P_TO": 3, "V_TO": 4, "I_TO": 5}
LOAD_MEASUREMENT_KIND = {"P_LOAD": 0, "V_LOAD": 1, "I_LOAD": 2}
GEN_MEASUREMENT_KIND = {"P_GEN": 0, "V_GEN": 1, "I_GEN": 2}
GEN_CONTROL_KIND = {"V": 0, "P": 1, "I": 2}


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
    ) -> None:
        self.idx = idx
        self.name = name
        self.device_type = device_type
        self.device_name = device_name
        self.meas_type = meas_type
        self.weight = weight
        self.valid = valid
        self.value = value

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


class MeasurementList(list):
    __slots__ = ("table", "normalized")

    def __init__(self, measurements=(), table: Optional[MeasurementTable] = None, normalized: bool = False):
        super().__init__(measurements)
        self.table = table
        self.normalized = bool(normalized)


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
    )


@dataclass
class ObservabilityResult:
    observable: bool
    rank: int
    state_count: int
    measurement_count: int
    deficiency: int
    singular_values: np.ndarray
    weak_states: List[Tuple[str, float]]


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
