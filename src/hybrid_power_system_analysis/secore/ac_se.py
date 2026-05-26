import argparse
import contextlib
import io
import math
import sys
import time
import warnings
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse.csgraph import connected_components as sp_connected_components
from scipy.sparse.csgraph import maximum_bipartite_matching as sp_maximum_bipartite_matching


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_lf import (
    ACPowerFlowCalc,
    matpower_branch_stamp_vectorized,
    matpower_transformer_stamp_vectorized,
)
from ac_model import ACPowerNetwork
from ac_array_model import (
    ACAC_COLS,
    BRANCH_COLS,
    BUS_COLS,
    GEN_COLS,
    LOAD_COLS,
    BREAK_COLS,
    SHUNT_B,
    SHUNT_COLS,
    SHUNT_V,
    SHUNT_Z,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
)
from model.ppc_topology import build_ac_ppc_with_topology_from_e_file, ensure_ac_ppc_topology
from model.meas_array_model import (
    MEAS_COLS,
    attach_device_pos_from_name_arrays,
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_table_from_meas_ppc,
)
from model.meas_type import (
    DEVICE_TYPE_CODES,
    DEVICE_TYPE_ACNode,
    DEVICE_TYPE_ACBranch,
    DEVICE_TYPE_ACTransformer,
    DEVICE_TYPE_ACLoad,
    DEVICE_TYPE_ACGenerator,
    DEVICE_TYPE_ACZeroBranch,
    DEVICE_TYPE_ACZeroBranchConstraint,
    DEVICE_TYPE_ACPowerBalance,
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACBreakConstraint,
    DEVICE_TYPE_ACACConverter,
    MEAS_TYPE_CODES,
    MEAS_TYPE_V,
    MEAS_TYPE_ANGLE,
    MEAS_TYPE_THETA,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_V_TO,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_P_LOAD,
    MEAS_TYPE_Q_LOAD,
    MEAS_TYPE_V_LOAD,
    MEAS_TYPE_I_LOAD,
    MEAS_TYPE_P_GEN,
    MEAS_TYPE_Q_GEN,
    MEAS_TYPE_V_GEN,
    MEAS_TYPE_I_GEN,
    MEAS_TYPE_P_BALANCE,
    MEAS_TYPE_Q_BALANCE,
    MEAS_TYPE_V_DIFF,
    MEAS_TYPE_ANGLE_DIFF,
    MEAS_TYPE_THETA_DIFF,
)
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from model.meas_model import (
    BadDataItem,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    EstimateResult,
    Measurement,
    MeasurementList,
    MeasurementTable,
    MeasurementTableView,
    ObservabilityResult,
    measurement_from_table_row,
    measurement_table_from_measurements,
    measurement_table_status_code,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    CHOLMOD_ANALYZE_AAT,
    CHOLMOD_CHOLESKY_AAT,
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    SparseJacobianBuilder,
    angle_residual_mask,
    build_normal_equations,
    full_normal_equation_from_lower,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    LowerNormalEquationCscPlan,
    NormalEquationAssemblyPlan,
    NormalEquationSolver,
    _normal_equation_structural_pattern,
    observability_rank_details,
    observability_weak_direction,
    sparse_structural_rank,
    targeted_redundancy_count,
    unanchored_angle_state_indices,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata
from secore.se_array_plan import (
    MeasurementPlanTable,
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    measurement_table_take,
    rows_by_device_type_code,
    take_measurement_view,
)
from secore.se_result import (
    SEResult,
    build_seresult_full_from_table,
    build_seresult_summary_from_table,
    normalize_seresult_result_mode,
)


DEFAULT_CASE = model_file("ac", "ieee39.e")
DEFAULT_MEAS = measurement_file("ac", "ieee39.meas")


_DEVICE_TYPE_CODES = DEVICE_TYPE_CODES

_AAT_NORMAL_SOLVER_MIN_STATE = 2000
_ACTIVE_DEVICE_KEY_POS_BITS = 40
_ACTIVE_MEASUREMENT_KEY_MEAS_BITS = 16

_TERMINAL_POWER_MEASUREMENT_TYPES = frozenset(
    (
        MEAS_TYPE_P_FROM,
        MEAS_TYPE_Q_FROM,
        MEAS_TYPE_P_TO,
        MEAS_TYPE_Q_TO,
    )
)
_VOLTAGE_MEASUREMENT_TYPES = frozenset(
    (
        MEAS_TYPE_V,
        MEAS_TYPE_V_FROM,
        MEAS_TYPE_V_TO,
        MEAS_TYPE_V_GEN,
        MEAS_TYPE_V_LOAD,
    )
)
_ANGLE_MEASUREMENT_CODE_SET = frozenset(
    (
        MEAS_TYPE_ANGLE,
        MEAS_TYPE_THETA,
        MEAS_TYPE_ANGLE_DIFF,
        MEAS_TYPE_THETA_DIFF,
    )
)
_PSEUDO_DEVICE_SUMMARY_TYPES = frozenset(
    (
        DEVICE_TYPE_ACGenerator,
        DEVICE_TYPE_ACLoad,
    )
)
_PSEUDO_MEASUREMENT_SUMMARY_TYPES = {
    DEVICE_TYPE_ACGenerator: frozenset((MEAS_TYPE_P_GEN, MEAS_TYPE_Q_GEN)),
    DEVICE_TYPE_ACLoad: frozenset((MEAS_TYPE_P_LOAD, MEAS_TYPE_Q_LOAD)),
    DEVICE_TYPE_ACZeroBranch: frozenset(
        (MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM, MEAS_TYPE_V_FROM, MEAS_TYPE_I_FROM)
    ),
    DEVICE_TYPE_ACBreak: frozenset(
        (MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM, MEAS_TYPE_V_FROM, MEAS_TYPE_I_FROM)
    ),
}

_OBSERVABILITY_RESULT_CACHE = {}

_MAX_MEAS_TYPE_CODE = max(MEAS_TYPE_CODES.values())
_TERMINAL_POWER_MEASUREMENT_TYPE_CODES = np.asarray(tuple(_TERMINAL_POWER_MEASUREMENT_TYPES), dtype=np.int16)
_VOLTAGE_MEASUREMENT_TYPE_CODES = np.asarray(tuple(_VOLTAGE_MEASUREMENT_TYPES), dtype=np.int16)
_ANGLE_MEASUREMENT_TYPE_CODES = np.asarray(tuple(_ANGLE_MEASUREMENT_CODE_SET), dtype=np.int16)


def _measurement_type_code_lookup_from_codes(codes: Sequence[int]) -> np.ndarray:
    lookup = np.empty(_MAX_MEAS_TYPE_CODE + 1, dtype=np.int16)
    lookup.fill(-1)
    for meas_code in codes:
        code = int(meas_code)
        if 0 <= code <= _MAX_MEAS_TYPE_CODE:
            lookup[code] = code
    return lookup


_AC_TERMINAL_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_P_FROM,
        MEAS_TYPE_Q_FROM,
        MEAS_TYPE_V_FROM,
        MEAS_TYPE_I_FROM,
        MEAS_TYPE_P_TO,
        MEAS_TYPE_Q_TO,
        MEAS_TYPE_V_TO,
        MEAS_TYPE_I_TO,
    )
)
_AC_ZERO_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_P_FROM,
        MEAS_TYPE_Q_FROM,
        MEAS_TYPE_V_FROM,
        MEAS_TYPE_I_FROM,
        MEAS_TYPE_P_TO,
        MEAS_TYPE_Q_TO,
        MEAS_TYPE_V_TO,
        MEAS_TYPE_I_TO,
        MEAS_TYPE_V_DIFF,
        MEAS_TYPE_ANGLE_DIFF,
        MEAS_TYPE_THETA_DIFF,
    )
)
_AC_NODE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_V,
        MEAS_TYPE_ANGLE,
        MEAS_TYPE_THETA,
    )
)
_AC_LOAD_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_P_LOAD,
        MEAS_TYPE_Q_LOAD,
        MEAS_TYPE_I_LOAD,
        MEAS_TYPE_V_LOAD,
    )
)
_AC_GENERATOR_POWER_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_P_GEN,
        MEAS_TYPE_Q_GEN,
        MEAS_TYPE_I_GEN,
    )
)
_AC_GENERATOR_SIMPLE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (MEAS_TYPE_V_GEN,)
)
_AC_BALANCE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_P_BALANCE,
        MEAS_TYPE_Q_BALANCE,
    )
)
_AC_CONSTRAINT_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_V_DIFF,
        MEAS_TYPE_ANGLE_DIFF,
        MEAS_TYPE_THETA_DIFF,
    )
)


def _meas_type_code_array(meas_type_values) -> np.ndarray:
    values = np.asarray(meas_type_values, dtype=object).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int16)
    warnings.warn(
        "AC SE requires meas_type_code arrays; string meas_type conversion is disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.zeros(int(values.size), dtype=np.int16)


def _measurement_type_code_lookup(kind_map) -> np.ndarray:
    if isinstance(kind_map, np.ndarray):
        return kind_map
    warnings.warn(
        "AC SE measurement plan expects MEAS_TYPE lookup arrays; string kind maps are disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _measurement_type_code_lookup_from_codes(())


class _PackedMeasurementKeyCache:
    """Set-like packed-key cache that avoids materializing Python ints until required."""

    __slots__ = ("estimator", "_keys")

    def __init__(self, estimator: "ACStateEstimator") -> None:
        self.estimator = estimator
        self._keys = None

    def _key_array(self) -> np.ndarray:
        return np.asarray(getattr(self.estimator, "_active_measurement_key_array_cache", ()), dtype=np.int64).reshape(-1)

    def _materialize(self) -> set:
        keys = self._keys
        if keys is None:
            keys = set(self._key_array().astype(object, copy=False))
            self._keys = keys
        return keys

    def _update_materialized(self, key_array: np.ndarray) -> None:
        if self._keys is not None:
            self._keys.update(np.asarray(key_array, dtype=np.int64).astype(object, copy=False))

    def extend_array(self, keys) -> None:
        self.estimator._append_active_measurement_key_array_cache(keys)

    def __contains__(self, key) -> bool:
        keys = self._keys
        if keys is not None:
            return int(key) in keys
        probe = np.asarray([int(key)], dtype=np.int64)
        return bool(self.estimator._active_measurement_key_membership(probe)[0])

    def __bool__(self) -> bool:
        keys = self._keys
        return bool(keys) if keys is not None else bool(self._key_array().size)

    def __len__(self) -> int:
        keys = self._keys
        return len(keys) if keys is not None else int(self._key_array().size)

    def __iter__(self):
        return iter(self._materialize())

    def __eq__(self, other) -> bool:
        return self._materialize() == other

    def add(self, key) -> None:
        self.extend_array(np.asarray([int(key)], dtype=np.int64))

    def update(self, keys) -> None:
        self.extend_array(keys)


class _PseudoMeasurementBuffer:
    """Preallocated row buffer for pseudo-measurement table construction."""

    __slots__ = (
        "store_strings",
        "name",
        "device_type",
        "device_type_code",
        "device_name",
        "device_pos",
        "meas_type",
        "meas_type_code",
        "value",
        "weight",
        "count",
    )

    def __init__(self, capacity: int, *, with_weights: bool = False, store_strings: bool = True) -> None:
        size = max(0, int(capacity))
        self.store_strings = bool(store_strings)
        self.name = np.empty(size, dtype=object) if self.store_strings else np.asarray([], dtype=object)
        self.device_type = np.empty(size, dtype=object) if self.store_strings else np.asarray([], dtype=object)
        self.device_type_code = np.empty(size, dtype=np.int16)
        self.device_name = np.empty(size, dtype=object) if self.store_strings else np.asarray([], dtype=object)
        self.device_pos = np.empty(size, dtype=np.int64)
        self.meas_type = np.empty(size, dtype=object) if self.store_strings else np.asarray([], dtype=object)
        self.meas_type_code = np.empty(size, dtype=np.int16)
        self.value = np.empty(size, dtype=np.float64)
        self.weight = np.empty(size, dtype=np.float64) if with_weights else None
        self.count = 0

    def __len__(self) -> int:
        return int(self.count)

    def _resize_array(self, array: np.ndarray, size: int) -> np.ndarray:
        grown = np.empty(int(size), dtype=array.dtype)
        if self.count:
            grown[: self.count] = array[: self.count]
        return grown

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.value.size:
            return
        size = max(int(required), max(8, int(self.value.size) * 2))
        if self.store_strings:
            self.name = self._resize_array(self.name, size)
            self.device_type = self._resize_array(self.device_type, size)
        self.device_type_code = self._resize_array(self.device_type_code, size)
        if self.store_strings:
            self.device_name = self._resize_array(self.device_name, size)
        self.device_pos = self._resize_array(self.device_pos, size)
        if self.store_strings:
            self.meas_type = self._resize_array(self.meas_type, size)
        self.meas_type_code = self._resize_array(self.meas_type_code, size)
        self.value = self._resize_array(self.value, size)
        if self.weight is not None:
            self.weight = self._resize_array(self.weight, size)

    def add(
        self,
        name: str,
        device_type: str,
        device_type_code: int,
        device_name: str,
        device_pos: int,
        meas_type: str,
        meas_type_code: int,
        value: float,
        *,
        weight: Optional[float] = None,
    ) -> None:
        pos = int(self.count)
        self._ensure_capacity(pos + 1)
        if self.store_strings:
            self.name[pos] = name
            self.device_type[pos] = device_type
        self.device_type_code[pos] = int(device_type_code)
        if self.store_strings:
            self.device_name[pos] = device_name
        self.device_pos[pos] = int(device_pos)
        if self.store_strings:
            self.meas_type[pos] = meas_type
        self.meas_type_code[pos] = int(meas_type_code)
        self.value[pos] = float(value)
        if self.weight is not None:
            self.weight[pos] = float(weight)
        self.count = pos + 1

    def add_many(
        self,
        device_type_code: int,
        device_pos: np.ndarray,
        meas_type_code: int,
        value: np.ndarray,
        *,
        weight: Optional[float] = None,
    ) -> None:
        device_pos = np.asarray(device_pos, dtype=np.int64)
        value = np.asarray(value, dtype=np.float64)
        count = int(device_pos.size)
        if count == 0:
            return
        if value.size != count:
            raise ValueError("pseudo measurement value array size does not match device_pos")
        start = int(self.count)
        end = start + count
        self._ensure_capacity(end)
        if self.store_strings:
            self.name[start:end] = ""
            self.device_type[start:end] = ""
            self.device_name[start:end] = ""
            self.meas_type[start:end] = ""
        self.device_type_code[start:end] = int(device_type_code)
        self.device_pos[start:end] = device_pos
        self.meas_type_code[start:end] = int(meas_type_code)
        self.value[start:end] = value
        if self.weight is not None:
            self.weight[start:end] = float(weight)
        self.count = end

    @property
    def names(self) -> np.ndarray:
        if not self.store_strings:
            return np.asarray([], dtype=object)
        return self.name[: self.count]

    @property
    def device_types(self) -> np.ndarray:
        if not self.store_strings:
            return np.asarray([], dtype=object)
        return self.device_type[: self.count]

    @property
    def device_type_codes(self) -> np.ndarray:
        return self.device_type_code[: self.count]

    @property
    def device_names(self) -> np.ndarray:
        if not self.store_strings:
            return np.asarray([], dtype=object)
        return self.device_name[: self.count]

    @property
    def device_positions(self) -> np.ndarray:
        return self.device_pos[: self.count]

    @property
    def meas_types(self) -> np.ndarray:
        if not self.store_strings:
            return np.asarray([], dtype=object)
        return self.meas_type[: self.count]

    @property
    def meas_type_codes(self) -> np.ndarray:
        return self.meas_type_code[: self.count]

    @property
    def values(self) -> np.ndarray:
        return self.value[: self.count]

    @property
    def weights(self) -> Optional[np.ndarray]:
        return None if self.weight is None else self.weight[: self.count]


def _file_cache_key(file_name: Path) -> Tuple[Path, int, int]:
    path = Path(file_name).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _measurement_table_from_measurements(measurements: Sequence["Measurement"]) -> MeasurementTable:
    table = measurement_table_from_measurements(
        measurements,
        device_type_codes=_DEVICE_TYPE_CODES,
        angle_measurement_types=ANGLE_MEASUREMENT_TYPES,
    )
    meas_type_code = getattr(table, "meas_type_code", None)
    if meas_type_code is None or np.asarray(meas_type_code).size != table.idx.size:
        warnings.warn(
            "AC SE MeasurementTable is missing meas_type_code; string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        table.meas_type_code = _meas_type_code_array(table.meas_type)
    return table


def _build_ac_se_ppc_namespace(ppc: Dict, source: Optional[Path] = None):
    """Create the AC SE network holder without materializing ACPowerNetwork devices."""
    if source is not None:
        ppc["source"] = str(Path(source).resolve())
    ensure_ac_ppc_topology(ppc)
    base = ppc["base"]
    return SimpleNamespace(ppc=ppc, base=base, topology=ppc["_topology_arrays"])


class ACStateEstimator:
    def __init__(
        self,
        e_file: Path = DEFAULT_CASE,
        meas_file: Path = DEFAULT_MEAS,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        diff_step: Optional[float] = None,
        flat_start: Optional[bool] = None,
        parameter_file: Path = DEFAULT_SE_PARAMETER_FILE,
        parameters: Optional[StateEstimationParameters] = None,
        profile: bool = False,
        network: Optional[ACPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
        auto_prepare: bool = True,
        matrix_dump_dir: Optional[Path] = None,
        power_flow_linear_solver: Optional[str] = None,
    ):
        self.profile_enabled = bool(profile)
        self.profile_times: Dict[str, float] = {}
        self.params = (parameters or load_se_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            diff_step=diff_step,
            flat_start=flat_start,
        )
        self.e_file = Path(e_file)
        self.meas_file = Path(meas_file)
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.flat_start = self.params.flat_start
        self.pseudo_measurement_weight = self.params.pseudo_measurement_weight
        self.targeted_pseudo_measurement_max = self.params.targeted_pseudo_measurement_max
        self.targeted_pseudo_measurement_redundancy_ratio = (
            self.params.targeted_pseudo_measurement_redundancy_ratio
        )
        self.targeted_pseudo_measurement_step = self.params.targeted_pseudo_measurement_step
        self.voltage_floor = self.params.voltage_floor
        self.min_current_voltage = self.params.min_current_voltage

        self._prepared = False
        self._prepare_network = network
        self._prepare_measurements = measurements
        self._prepare_active_measurements = bool(prepare_active_measurements)
        self._prepare_defer_finalize = bool(defer_prepare_finalize)
        self._array_only_runtime = True
        self._array_only_estimate_result = True
        self.matrix_dump_dir = Path(matrix_dump_dir) if matrix_dump_dir is not None else None
        solver_name = str(power_flow_linear_solver).strip().lower() if power_flow_linear_solver is not None else ""
        self.power_flow_linear_solver = solver_name or None
        self.observability_result = None
        self.estimate_result = None
        self.removed_bad_data: List[BadDataItem] = []
        self.bad_items: List[BadDataItem] = []
        self.normalized_residual = np.array([], dtype=np.float64)
        self.se_result = None
        if auto_prepare:
            self.prepare(
                network=network,
                measurements=measurements,
                prepare_active_measurements=prepare_active_measurements,
            )

    def _measurement_sequence_from_table(self, table: MeasurementTable, *, normalized: bool = False):
        return MeasurementTableView(table, normalized=normalized)

    @staticmethod
    def _clear_meas_ppc_runtime_arrays(meas_ppc: Dict) -> None:
        """Drop estimator-local measurement runtime arrays copied from a shared file cache."""
        for key in (
            "device_pos",
            "scale",
            "from_pos",
            "to_pos",
            "available",
            "_ac_se_runtime_cache_key",
            "_mutable_runtime_arrays",
        ):
            meas_ppc.pop(key, None)

    def _ppc_runtime_node_and_device_context(self, ppc: Dict, topology_arrays) -> None:
        """Initialize SE runtime arrays from PPC/topology without network device objects."""
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        bus_names = np.asarray(ppc["bus_name"], dtype=object)
        active_bus_pos = np.flatnonzero(topology_arrays.bus_alive_mask).astype(np.int32, copy=False)
        bus_solver_pos = np.full(len(topology_arrays.bus_ids), -1, dtype=np.int32)
        if active_bus_pos.size:
            bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)

        self._ac_island_alive_mask = np.asarray(topology_arrays.island_alive_mask, dtype=bool)
        node_ids = np.asarray(topology_arrays.bus_ids, dtype=np.int64)[active_bus_pos.astype(np.intp, copy=False)]
        node_names = np.empty(active_bus_pos.size, dtype=object)
        node_vbase = np.ones(active_bus_pos.size, dtype=np.float64)
        node_voltage = np.ones(active_bus_pos.size, dtype=np.float64)
        node_angle = np.zeros(active_bus_pos.size, dtype=np.float64)
        node_island_pos = np.full(active_bus_pos.size, -1, dtype=np.int32)
        first_node_row_by_solver_pos = np.full(active_bus_pos.size, -1, dtype=np.int32)
        if active_bus_pos.size:
            offsets = np.asarray(topology_arrays.bus_node_offsets, dtype=np.int64)
            starts = offsets[active_bus_pos.astype(np.intp, copy=False)]
            ends = offsets[active_bus_pos.astype(np.intp, copy=False) + 1]
            has_node = ends > starts
            if np.any(has_node):
                valid_bus_pos = active_bus_pos[has_node].astype(np.intp, copy=False)
                solver_pos = bus_solver_pos[valid_bus_pos].astype(np.intp, copy=False)
                node_rows = np.asarray(topology_arrays.bus_node_indices, dtype=np.int64)[starts[has_node]].astype(
                    np.intp,
                    copy=False,
                )
                row_values = bus[node_rows]
                first_node_row_by_solver_pos[solver_pos] = node_rows.astype(np.int32, copy=False)
                node_names[solver_pos] = bus_names[node_rows].astype(str, copy=False)
                node_vbase[solver_pos] = row_values[:, BUS_COLS["vbase"]]
                node_voltage[solver_pos] = row_values[:, BUS_COLS["voltage"]]
                node_angle[solver_pos] = row_values[:, BUS_COLS["angle"]]
                node_island_pos[solver_pos] = np.asarray(topology_arrays.bus_to_island_pos, dtype=np.int32)[
                    valid_bus_pos
                ]
        missing_names = first_node_row_by_solver_pos < 0
        if np.any(missing_names):
            node_names[missing_names] = [
                f"bus_{int(node_id)}" for node_id in node_ids[missing_names]
            ]
        self._ac_node_ids = node_ids.astype(np.int64, copy=False)
        if self._ac_node_ids.size:
            node_id_order = np.argsort(self._ac_node_ids, kind="stable")
            self._ac_node_id_lookup_ids = self._ac_node_ids[node_id_order].astype(np.int64, copy=False)
            self._ac_node_id_lookup_pos = node_id_order.astype(np.int64, copy=False)
        else:
            self._ac_node_id_lookup_ids = np.asarray([], dtype=np.int64)
            self._ac_node_id_lookup_pos = np.asarray([], dtype=np.int64)
        self._ac_node_names = node_names
        self._ac_node_vbase_by_pos = node_vbase
        self.file_theta = node_angle
        self.file_voltage = node_voltage
        self._ac_node_island_pos = node_island_pos
        self._ac_first_node_row_by_solver_pos = first_node_row_by_solver_pos
        self._ac_island_reference_solver_pos = np.full(
            len(topology_arrays.island_ids),
            -1,
            dtype=np.int32,
        )
        ref_bus_pos = np.asarray(topology_arrays.island_reference_bus_pos, dtype=np.int32)
        valid_ref = (ref_bus_pos >= 0) & (ref_bus_pos < bus_solver_pos.size)
        if np.any(valid_ref):
            self._ac_island_reference_solver_pos[np.flatnonzero(valid_ref).astype(np.intp, copy=False)] = (
                bus_solver_pos[ref_bus_pos[valid_ref].astype(np.intp, copy=False)]
            )

        node_bus_pos = np.asarray(topology_arrays.node_to_bus_pos, dtype=np.int32)
        node_solver_pos = np.full(bus.shape[0], -1, dtype=np.int32)
        node_valid = (
            np.asarray(topology_arrays.node_alive_mask, dtype=bool)
            & (node_bus_pos >= 0)
            & (node_bus_pos < bus_solver_pos.size)
        )
        if np.any(node_valid):
            node_solver_pos[node_valid] = bus_solver_pos[node_bus_pos[node_valid].astype(np.intp, copy=False)]
        self._ac_node_solver_pos_by_ppc_row = node_solver_pos
        self._build_ppc_state_index_arrays()

    def prepare(
        self,
        *,
        network: Optional[ACPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: Optional[bool] = None,
        defer_prepare_finalize: Optional[bool] = None,
    ) -> "ACStateEstimator":
        if (
            self._prepared
            and network is None
            and measurements is None
            and prepare_active_measurements is None
            and defer_prepare_finalize is None
        ):
            return self
        if network is None:
            network = self._prepare_network
        else:
            self._prepare_network = network
        if measurements is None:
            measurements = self._prepare_measurements
        else:
            self._prepare_measurements = measurements
        if prepare_active_measurements is None:
            prepare_active_measurements = self._prepare_active_measurements
        else:
            self._prepare_active_measurements = bool(prepare_active_measurements)
        if defer_prepare_finalize is None:
            defer_prepare_finalize = self._prepare_defer_finalize
        else:
            self._prepare_defer_finalize = bool(defer_prepare_finalize)
        prepare_active_measurements = bool(prepare_active_measurements)
        defer_prepare_finalize = bool(defer_prepare_finalize)
        self._array_only_runtime = True
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        self.network = network if network is not None else self._load_network(self.e_file)
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            self.meas_ppc = build_meas_ppc_from_e_file(
                self.meas_file,
                include_strings=False,
                use_cache=False,
                include_matrix=False,
            )
            self.meas_ppc["_mutable_runtime_arrays"] = True
            self.measurements = self._measurement_sequence_from_table(
                measurement_table_from_meas_ppc(self.meas_ppc, include_strings=False),
                normalized=bool(self.meas_ppc.get("normalized", False)),
            )
        elif isinstance(measurements, dict):
            self.meas_ppc = copy_meas_ppc(measurements)
            self.meas_ppc["_mutable_runtime_arrays"] = True
            self.measurements = self._measurement_sequence_from_table(
                measurement_table_from_meas_ppc(self.meas_ppc, include_strings=False),
                normalized=bool(self.meas_ppc.get("normalized", False)),
            )
        elif isinstance(measurements, MeasurementList):
            table = _measurement_table_from_measurements(measurements)
            self.meas_ppc = self._measurement_table_to_meas_ppc(table)
            self.measurements = self._measurement_sequence_from_table(
                table,
                normalized=getattr(measurements, "normalized", False),
            )
        else:
            table = _measurement_table_from_measurements(list(measurements))
            self.meas_ppc = self._measurement_table_to_meas_ppc(table)
            self.measurements = self._measurement_sequence_from_table(table, normalized=False)
        elapsed = time.perf_counter() - stage_start
        self._record_profile_time("init.load_measurement_parse", elapsed)
        self._record_profile_time("init.load_measurements", elapsed)
        ppc = self._ac_ppc_dict()
        base = ppc.get("base")
        if base is None:
            self._warn_required_runtime_missing("PPC base", "prepare")
        self.p_base = float(base["p_base"])
        self.p_base_kW = float(base["p_base_kW"])
        self.u_scale = float(base["u_scale"])
        self.i_scale = float(base["i_scale"])

        topology_arrays = ppc.get("_topology_arrays")
        if topology_arrays is None:
            self._warn_required_runtime_missing("PPC topology arrays", "prepare")
        stage_start = time.perf_counter()
        self._ppc_runtime_node_and_device_context(ppc, topology_arrays)
        self._record_profile_time("init.ppc_runtime_context", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._attach_meas_ppc_device_pos()
        self._record_profile_time("init.measurement_device_pos", time.perf_counter() - stage_start)
        if int(getattr(self, "_ac_node_ids", np.asarray([], dtype=np.int64)).size) == 0:
            raise RuntimeError("No alive AC nodes are available for state estimation")
        self.n_nodes = int(self._ac_node_ids.size)
        if defer_prepare_finalize:
            self.power_flow_seed_converged = False
            return self
        self.finalize_prepare(prepare_active_measurements=prepare_active_measurements)
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        self._prepared = True
        return self

    def _build_state_meta_arrays(self) -> Dict[str, np.ndarray]:
        n_state = int(getattr(self, "n_state", 0))
        side = np.full(n_state, "ac", dtype=object)
        kind = np.empty(n_state, dtype=object)
        device_type = np.empty(n_state, dtype=object)
        device_name = np.empty(n_state, dtype=object)
        terminal = np.full(n_state, "", dtype=object)
        component = np.empty(n_state, dtype=object)
        legacy_label = np.empty(n_state, dtype=object)
        device_pos = np.full(n_state, -1, dtype=np.int64)
        device_type_code = np.zeros(n_state, dtype=np.int16)
        meas_type_code = np.zeros(n_state, dtype=np.int16)
        cursor = 0

        def inverse_plan_pos(plan_values, size: int) -> np.ndarray:
            values = np.asarray(plan_values, dtype=np.int64)
            out = np.full(max(int(size), 0), -1, dtype=np.int64)
            if values.size == 0 or out.size == 0:
                return out
            valid = (values >= 0) & (values < out.size)
            if np.any(valid):
                out[values[valid].astype(np.intp, copy=False)] = np.flatnonzero(valid).astype(np.int64, copy=False)
            return out

        def plan_pos_for_states(inverse_values: np.ndarray, state_pos: np.ndarray) -> np.ndarray:
            pos = np.asarray(state_pos, dtype=np.int64)
            out = np.full(pos.size, -1, dtype=np.int64)
            valid = (pos >= 0) & (pos < inverse_values.size)
            if np.any(valid):
                out[valid] = inverse_values[pos[valid].astype(np.intp, copy=False)]
            return out

        node_plan_by_solver_pos = inverse_plan_pos(
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_nodes", 0)),
        )
        zero_plan_by_current_pos = inverse_plan_pos(
            getattr(self, "_ac_zero_branch_plan_current_pos", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_switch_current", 0)),
        )
        break_plan_by_current_pos = inverse_plan_pos(
            getattr(self, "_ac_break_plan_current_pos", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_switch_current", 0)),
        )
        gen_plan_by_state_idx = inverse_plan_pos(
            getattr(self, "_ac_generator_plan_index", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_generator_power", 0)),
        )
        load_plan_by_state_idx = inverse_plan_pos(
            getattr(self, "_ac_load_plan_index", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_load_power", 0)),
        )
        acac_plan_by_state_idx = inverse_plan_pos(
            getattr(self, "_ac_acac_plan_index", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_acac_power", 0)),
        )

        angle_state_pos = np.asarray(getattr(self, "angle_state_pos", ()), dtype=np.int64)
        count = int(angle_state_pos.size)
        if count:
            rows = slice(cursor, cursor + count)
            names = np.asarray(self._ac_node_names, dtype=object)[angle_state_pos.astype(np.intp, copy=False)]
            kind[rows] = "angle"
            device_type[rows] = "ACNode"
            device_name[rows] = names
            component[rows] = "theta"
            legacy_label[rows] = np.char.add("theta:", names.astype(str)).astype(object)
            device_pos[rows] = plan_pos_for_states(node_plan_by_solver_pos, angle_state_pos)
            device_type_code[rows] = DEVICE_TYPE_ACNode
            meas_type_code[rows] = MEAS_TYPE_ANGLE
            cursor += count

        voltage_state_pos = np.asarray(getattr(self, "voltage_state_pos", ()), dtype=np.int64)
        count = int(voltage_state_pos.size)
        if count:
            rows = slice(cursor, cursor + count)
            names = np.asarray(self._ac_node_names, dtype=object)[voltage_state_pos.astype(np.intp, copy=False)]
            kind[rows] = "voltage"
            device_type[rows] = "ACNode"
            device_name[rows] = names
            component[rows] = "magnitude"
            legacy_label[rows] = np.char.add("V:", names.astype(str)).astype(object)
            device_pos[rows] = plan_pos_for_states(node_plan_by_solver_pos, voltage_state_pos)
            device_type_code[rows] = DEVICE_TYPE_ACNode
            meas_type_code[rows] = MEAS_TYPE_V
            cursor += count

        zero_current_names = np.asarray(getattr(self, "_ac_zero_current_names", ()), dtype=object)
        zero_current_kind = np.asarray(getattr(self, "_ac_zero_current_kind_code", ()), dtype=np.int8)
        current_count = int(zero_current_names.size)
        if current_count:
            current_pos = np.arange(current_count, dtype=np.int64)
            is_zero = zero_current_kind.astype(np.int8, copy=False) == 0
            zero_pos = plan_pos_for_states(zero_plan_by_current_pos, current_pos)
            break_pos = plan_pos_for_states(break_plan_by_current_pos, current_pos)
            current_device_pos = np.where(is_zero, zero_pos, break_pos).astype(np.int64, copy=False)
            current_device_type = np.where(is_zero, "ACZeroBranch", "ACBreak").astype(object)
            current_kind = np.where(is_zero, "zero_current", "break_current").astype(object)
            current_device_code = np.where(
                is_zero,
                DEVICE_TYPE_ACZeroBranch,
                DEVICE_TYPE_ACBreak,
            ).astype(np.int16, copy=False)
            prefix = np.where(is_zero, "I_Z_", "I_B_").astype(str)
            for component_name, suffix in (("re", "RE:"), ("im", "IM:")):
                rows = slice(cursor, cursor + current_count)
                kind[rows] = current_kind
                device_type[rows] = current_device_type
                device_name[rows] = zero_current_names
                component[rows] = component_name
                legacy_label[rows] = np.char.add(
                    np.char.add(prefix, suffix),
                    zero_current_names.astype(str),
                ).astype(object)
                device_pos[rows] = current_device_pos
                device_type_code[rows] = current_device_code
                meas_type_code[rows] = MEAS_TYPE_I_FROM
                cursor += current_count

        gen_names = np.asarray(getattr(self, "_ac_generator_power_names", ()), dtype=object)
        gen_count = int(gen_names.size)
        if gen_count:
            gen_state_pos = np.arange(gen_count, dtype=np.int64)
            gen_device_pos = plan_pos_for_states(gen_plan_by_state_idx, gen_state_pos)
            for state_kind, component_name, label_prefix, meas_code in (
                ("generator_p", "p", "P_GEN:", MEAS_TYPE_P_GEN),
                ("generator_q", "q", "Q_GEN:", MEAS_TYPE_Q_GEN),
            ):
                rows = slice(cursor, cursor + gen_count)
                kind[rows] = state_kind
                device_type[rows] = "ACGenerator"
                device_name[rows] = gen_names
                component[rows] = component_name
                legacy_label[rows] = np.char.add(label_prefix, gen_names.astype(str)).astype(object)
                device_pos[rows] = gen_device_pos
                device_type_code[rows] = DEVICE_TYPE_ACGenerator
                meas_type_code[rows] = meas_code
                cursor += gen_count

        load_names = np.asarray(getattr(self, "_ac_load_power_names", ()), dtype=object)
        load_count = int(load_names.size)
        if load_count:
            load_state_pos = np.arange(load_count, dtype=np.int64)
            load_device_pos = plan_pos_for_states(load_plan_by_state_idx, load_state_pos)
            for state_kind, component_name, label_prefix, meas_code in (
                ("load_p", "p", "P_LOAD:", MEAS_TYPE_P_LOAD),
                ("load_q", "q", "Q_LOAD:", MEAS_TYPE_Q_LOAD),
            ):
                rows = slice(cursor, cursor + load_count)
                kind[rows] = state_kind
                device_type[rows] = "ACLoad"
                device_name[rows] = load_names
                component[rows] = component_name
                legacy_label[rows] = np.char.add(label_prefix, load_names.astype(str)).astype(object)
                device_pos[rows] = load_device_pos
                device_type_code[rows] = DEVICE_TYPE_ACLoad
                meas_type_code[rows] = meas_code
                cursor += load_count

        shunt_q_names = np.asarray(getattr(self, "_ac_voltage_control_shunt_names", ()), dtype=object)
        shunt_count = int(shunt_q_names.size)
        if shunt_count:
            rows = slice(cursor, cursor + shunt_count)
            kind[rows] = "shunt_q"
            device_type[rows] = "ACShuntCompensator"
            device_name[rows] = shunt_q_names
            component[rows] = "q"
            legacy_label[rows] = np.char.add("Q_SHUNT:", shunt_q_names.astype(str)).astype(object)
            device_pos[rows] = np.arange(shunt_count, dtype=np.int64)
            cursor += shunt_count

        acac_names = np.asarray(getattr(self, "_ac_acac_power_names", ()), dtype=object)
        acac_count = int(acac_names.size)
        if acac_count:
            acac_state_pos = np.arange(acac_count, dtype=np.int64)
            acac_device_pos = plan_pos_for_states(acac_plan_by_state_idx, acac_state_pos)
            for state_kind, terminal_name, component_name, label_prefix, meas_code in (
                ("acac_p_from", "from", "p", "ACAC_P_FROM:", MEAS_TYPE_P_FROM),
                ("acac_q_from", "from", "q", "ACAC_Q_FROM:", MEAS_TYPE_Q_FROM),
                ("acac_p_to", "to", "p", "ACAC_P_TO:", MEAS_TYPE_P_TO),
                ("acac_q_to", "to", "q", "ACAC_Q_TO:", MEAS_TYPE_Q_TO),
            ):
                rows = slice(cursor, cursor + acac_count)
                kind[rows] = state_kind
                device_type[rows] = "ACACConverter"
                device_name[rows] = acac_names
                terminal[rows] = terminal_name
                component[rows] = component_name
                legacy_label[rows] = np.char.add(label_prefix, acac_names.astype(str)).astype(object)
                device_pos[rows] = acac_device_pos
                device_type_code[rows] = DEVICE_TYPE_ACACConverter
                meas_type_code[rows] = meas_code
                cursor += acac_count

        if cursor != n_state:
            warnings.warn(
                f"AC SE state metadata array size mismatch: built {cursor}, expected {n_state}.",
                RuntimeWarning,
                stacklevel=2,
            )
            side = side[:cursor]
            kind = kind[:cursor]
            device_type = device_type[:cursor]
            device_name = device_name[:cursor]
            terminal = terminal[:cursor]
            component = component[:cursor]
            legacy_label = legacy_label[:cursor]
            device_pos = device_pos[:cursor]
            device_type_code = device_type_code[:cursor]
            meas_type_code = meas_type_code[:cursor]
        return {
            "side": side,
            "kind": kind,
            "device_type": device_type,
            "device_name": device_name,
            "terminal": terminal,
            "component": component,
            "legacy_label": legacy_label,
            "device_pos": device_pos,
            "device_type_code": device_type_code,
            "meas_type_code": meas_type_code,
        }

    def _state_meta_arrays_ref(self) -> Dict[str, np.ndarray]:
        arrays = getattr(self, "_state_meta_arrays", None)
        if arrays is None:
            arrays = self._build_state_meta_arrays()
            self._state_meta_arrays = arrays
        return arrays

    def _build_state_meta(self) -> List[StateMeta]:
        arrays = self._state_meta_arrays_ref()
        row_count = int(np.asarray(arrays["kind"], dtype=object).size)
        state_meta = [None] * row_count
        for idx in range(row_count):
            state_meta[idx] = StateMeta(
                str(arrays["side"][idx]),
                str(arrays["kind"][idx]),
                str(arrays["device_type"][idx]),
                str(arrays["device_name"][idx]),
                terminal=str(arrays["terminal"][idx]),
                component=str(arrays["component"][idx]),
                legacy_label=str(arrays["legacy_label"][idx]),
                device_pos=int(arrays["device_pos"][idx]),
                device_type_code=int(arrays["device_type_code"][idx]),
                meas_type_code=int(arrays["meas_type_code"][idx]),
            )
        return state_meta

    @property
    def state_meta(self) -> List[StateMeta]:
        cache = getattr(self, "_state_meta_cache", None)
        if cache is None:
            cache = self._build_state_meta()
            self._state_meta_cache = cache
        return cache

    @state_meta.setter
    def state_meta(self, value) -> None:
        self._state_meta_cache = value
        self._state_labels_cache = None

    @property
    def state_labels(self) -> List[str]:
        cache = getattr(self, "_state_labels_cache", None)
        if cache is None:
            meta_cache = getattr(self, "_state_meta_cache", None)
            if meta_cache is None:
                labels = [str(label) for label in np.asarray(self._state_meta_arrays_ref()["legacy_label"], dtype=object)]
                cache = labels if all(labels) else state_labels_from_metadata(self.state_meta)
            else:
                labels = [meta.legacy_label for meta in meta_cache]
                cache = labels if all(labels) else state_labels_from_metadata(meta_cache)
            self._state_labels_cache = cache
        return cache

    @state_labels.setter
    def state_labels(self, value) -> None:
        self._state_labels_cache = value

    def finalize_prepare(
        self,
        *,
        prepare_active_measurements: bool = True,
        measurements_already_normalized: bool = False,
    ) -> "ACStateEstimator":
        if measurements_already_normalized:
            table = _measurement_table_from_measurements(self.measurements)
            self.measurement_table = table
            self._record_profile_time("init.measurement_device_indexes", 0.0)
        else:
            self._record_profile_time("init.measurement_device_indexes", 0.0)
            stage_start = time.perf_counter()
            self._convert_measurements_to_pu()
            self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
            table = getattr(self, "measurement_table", None)
        device_pos = getattr(table, "device_pos", None) if table is not None else None
        if (
            table is None
            or device_pos is None
            or np.asarray(device_pos).size != int(getattr(table, "idx", np.asarray([])).size)
        ):
            warnings.warn(
                "AC SE measurement table is missing device_pos after normalization; "
                "name-based index rebuild is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.power_flow_seed_converged = False
        if not self.flat_start:
            seed_start = time.perf_counter()
            stage_start = time.perf_counter()
            self._apply_measurement_seed_to_network()
            self._record_profile_time("seed.apply_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            if self.power_flow_linear_solver:
                setattr(self.network, "_se_power_flow_linear_solver", self.power_flow_linear_solver)
            self.power_flow_seed_converged = bool(self._run_power_flow_seed(self.network, self.params, self.e_file))
            self._record_profile_time("seed.lf", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_file_state_from_network()
            self._record_profile_time("seed.refresh_file_state", time.perf_counter() - stage_start)
            self._record_profile_time("seed.total", time.perf_counter() - seed_start)
        stage_start = time.perf_counter()
        self.node_voltage_measurements = self._node_voltage_measurements()
        self.node_degrees = self._node_incident_degrees()
        self.reference_pos = self._select_reference_positions()
        self.references = self.reference_pos
        self.ref_idx = {int(pos) for pos in self.reference_pos}
        self.reference_voltage_by_pos = {
            int(pos): self.node_voltage_measurements[int(self._ac_node_ids[int(pos)])]
            for pos in self.reference_pos
            if 0 <= int(pos) < self._ac_node_ids.size
            and int(self._ac_node_ids[int(pos)]) in self.node_voltage_measurements
        }
        self.reference_angle_by_pos = (
            self._reference_angle_offsets()
            if bool(getattr(self, "_has_valid_angle_measurements", False))
            else {}
        )
        self._rebase_angle_measurements()
        self._build_zero_tie_state_layout()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        ppc = self._ac_ppc_dict()
        gen_rows = self._required_array_attr("_ac_generator_power_rows", dtype=np.int64, context="finalize_prepare")
        load_rows = self._required_array_attr("_ac_load_power_rows", dtype=np.int64, context="finalize_prepare")
        self.initial_gen_p_array, self.initial_gen_q_array = self._ppc_generator_pseudo_power_arrays(
            ppc,
            gen_rows,
        )
        load_node_pos = self._required_array_attr("_ac_load_power_node_pos", dtype=np.int64, context="finalize_prepare")
        load_voltage = (
            self.file_voltage[load_node_pos.astype(np.intp, copy=False)]
            if load_node_pos.size
            else np.asarray([], dtype=np.float64)
        )
        self.initial_load_p_array, self.initial_load_q_array = self._ppc_load_pseudo_power_arrays(
            ppc,
            load_rows,
            load_voltage,
        )
        self.n_generator_power = int(gen_rows.size)
        self.n_load_power = int(load_rows.size)
        self.zero_current_i = self._required_array_attr("_ac_zero_current_i", dtype=np.int32, context="finalize_prepare")
        self.zero_current_j = self._required_array_attr("_ac_zero_current_j", dtype=np.int32, context="finalize_prepare")
        self.n_switch_current = int(self.zero_current_i.size)
        self.switch_current_idx_array = np.arange(self.n_switch_current, dtype=np.int32)
        self.switch_balance_end_pos = np.concatenate((self.zero_current_i, self.zero_current_j)).astype(
            np.int64,
            copy=False,
        )
        self.switch_balance_end_current_idx = np.concatenate(
            (self.switch_current_idx_array, self.switch_current_idx_array)
        )
        self.switch_balance_sign = np.concatenate(
            (
                np.ones(self.n_switch_current, dtype=np.float64),
                -np.ones(self.n_switch_current, dtype=np.float64),
            )
        )
        self.generator_power_idx_array = np.arange(self.n_generator_power, dtype=np.int32)
        self.load_power_idx_array = np.arange(self.n_load_power, dtype=np.int32)
        shunt_rows = self._required_array_attr("_ac_shunt_device_rows", dtype=np.int64, context="finalize_prepare")
        if shunt_rows.size:
            shunt_table = np.asarray(ppc["shunt"], dtype=np.float64)
            control_type = shunt_table[shunt_rows.astype(np.intp, copy=False), SHUNT_COLS["control_type"]].astype(
                np.int64,
                copy=False,
            )
            v_mask = control_type == SHUNT_V
            voltage_control_rows = shunt_rows[v_mask]
            voltage_control_names = np.asarray(getattr(self, "_ac_shunt_device_names", ()), dtype=object)[v_mask]
            shunt_node_pos = self._required_array_attr("_ac_shunt_device_node_pos", dtype=np.int32, context="finalize_prepare")
            self.shunt_q_pos_array = shunt_node_pos[v_mask].astype(np.int32, copy=False)
            self.initial_shunt_q_array = shunt_table[
                voltage_control_rows.astype(np.intp, copy=False),
                SHUNT_COLS["q"],
            ].astype(np.float64, copy=False)
        else:
            voltage_control_rows = np.asarray([], dtype=np.int64)
            voltage_control_names = np.asarray([], dtype=object)
            self.shunt_q_pos_array = np.asarray([], dtype=np.int32)
            self.initial_shunt_q_array = np.asarray([], dtype=np.float64)
        self._ac_voltage_control_shunt_names = voltage_control_names
        self.n_shunt_q = int(voltage_control_rows.size)
        self.shunt_q_idx_array = np.arange(self.n_shunt_q, dtype=np.int32)
        acac_rows = self._required_array_attr("_ac_acac_power_rows", dtype=np.int64, context="finalize_prepare")
        if acac_rows.size:
            acac_table = np.asarray(ppc.get("acac", np.zeros((0, len(ACAC_COLS)))), dtype=np.float64)[
                acac_rows.astype(np.intp, copy=False)
            ]
            acac_p_from = acac_table[:, ACAC_COLS["i_p"]].astype(np.float64, copy=True)
            acac_q_from = acac_table[:, ACAC_COLS["i_q"]].astype(np.float64, copy=True)
            acac_p_to = acac_table[:, ACAC_COLS["j_p"]].astype(np.float64, copy=True)
            acac_q_to = acac_table[:, ACAC_COLS["j_q"]].astype(np.float64, copy=True)
            p_set = acac_table[:, ACAC_COLS["p_set"]]
            i_q_set = acac_table[:, ACAC_COLS["i_q_set"]]
            j_q_set = acac_table[:, ACAC_COLS["j_q_set"]]
            missing_power = (np.abs(acac_p_from) <= 1e-12) & (np.abs(acac_p_to) <= 1e-12)
            if np.any(missing_power):
                acac_p_from[missing_power] = p_set[missing_power]
                acac_p_to[missing_power] = -p_set[missing_power]
            missing_q_from = np.abs(acac_q_from) <= 1e-12
            missing_q_to = np.abs(acac_q_to) <= 1e-12
            if np.any(missing_q_from):
                acac_q_from[missing_q_from] = i_q_set[missing_q_from]
            if np.any(missing_q_to):
                acac_q_to[missing_q_to] = j_q_set[missing_q_to]
            self.initial_acac_p_from_array = acac_p_from
            self.initial_acac_q_from_array = acac_q_from
            self.initial_acac_p_to_array = acac_p_to
            self.initial_acac_q_to_array = acac_q_to
        else:
            self.initial_acac_p_from_array = np.asarray([], dtype=np.float64)
            self.initial_acac_q_from_array = np.asarray([], dtype=np.float64)
            self.initial_acac_p_to_array = np.asarray([], dtype=np.float64)
            self.initial_acac_q_to_array = np.asarray([], dtype=np.float64)
        self.n_acac_power = int(acac_rows.size)
        self.acac_power_idx_array = np.arange(self.n_acac_power, dtype=np.int32)
        self.generator_balance_minus_ones = -np.ones(self.n_generator_power, dtype=np.float64)
        self.load_balance_ones = np.ones(self.n_load_power, dtype=np.float64)
        self.shunt_balance_minus_ones = -np.ones(self.n_shunt_q, dtype=np.float64)
        self.acac_balance_ones = np.ones(self.n_acac_power, dtype=np.float64)
        self.base_switch_re = self.n_angle + self.n_voltage
        self.base_switch_im = self.base_switch_re + self.n_switch_current
        self.base_gen_p = self.base_switch_im + self.n_switch_current
        self.base_gen_q = self.base_gen_p + self.n_generator_power
        self.base_load_p = self.base_gen_q + self.n_generator_power
        self.base_load_q = self.base_load_p + self.n_load_power
        self.base_shunt_q = self.base_load_q + self.n_load_power
        self.base_acac_p_from = self.base_shunt_q + self.n_shunt_q
        self.base_acac_q_from = self.base_acac_p_from + self.n_acac_power
        self.base_acac_p_to = self.base_acac_q_from + self.n_acac_power
        self.base_acac_q_to = self.base_acac_p_to + self.n_acac_power
        self.n_state = self.base_acac_q_to + self.n_acac_power
        self._state_meta_arrays = None
        self._state_meta_cache = None
        self._state_labels_cache = None

        stage_start = time.perf_counter()
        if not hasattr(self, "_ac_measurement_plan_pos_by_ppc_row"):
            self._build_measurement_plan_lookup_arrays()
        self._record_profile_time("init.measurement_plan_lookup", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        if bool(getattr(self, "_array_only_runtime", False)):
            self._state_meta_arrays = None
        else:
            self._state_meta_arrays = self._build_state_meta_arrays()
        self._state_meta_cache = None
        self._state_labels_cache = None
        self._record_profile_time("init.state_meta_arrays", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        self.Y = self._build_y_matrix()
        self._prepare_y_row_cache()
        self._record_profile_time("init.network_matrices", time.perf_counter() - stage_start)
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        stage_start = time.perf_counter()
        self._seed_power_state_arrays_from_measurements()
        self._record_profile_time("init.seed_power_states", time.perf_counter() - stage_start)
        if prepare_active_measurements:
            stage_start = time.perf_counter()
            self._add_pseudo_power_measurements()
            self._add_power_balance_constraint_measurements()
            self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_active_measurement_indexes()
            self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            fast_cache_args = None
            fast_observability = self._fast_active_observability_certificate()
            if fast_observability is None:
                self._add_targeted_observability_pseudo_measurements()
            else:
                self._initial_observability_cache = fast_observability
                initial_state = self.initial_state()
                active_plan_tables = self._active_measurement_plan_tables_ref()
                initial_h = self.jacobian_sparse(initial_state, active_plan_tables)
                self._cache_observability_matrix(
                    fast_observability,
                    initial_state,
                    active_plan_tables,
                    initial_h,
                    cache_lower_normal_plan=False,
                )
                fast_cache_args = (fast_observability, initial_state, active_plan_tables, initial_h)
            self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
            if fast_cache_args is not None:
                self._cache_observability_matrix(*fast_cache_args)
        else:
            base_table = _measurement_table_from_measurements(self.measurements)
            empty_rows = np.asarray([], dtype=np.int64)
            empty_table = measurement_table_take(base_table, empty_rows, rows_by_device_type_code={})
            self.active_measurements = self._measurement_sequence_from_table(
                empty_table,
                normalized=getattr(self.measurements, "normalized", False),
            )
            self.active_measurement_table = empty_table
            self.active_measurement_rows = np.array([], dtype=np.int64)
            self.active_z = np.array([], dtype=np.float64)
            self.active_weight = np.array([], dtype=np.float64)
            self.active_angle_residual_mask = np.array([], dtype=bool)
            self.active_has_angle_residuals = False
            self.active_uniform_weight = None
            self.active_weights_are_uniform = False
            self._active_rows_by_device_type_code = {}
            self._branch_transformer_vector_plan_cache = {}
            self._simple_jacobian_plan_cache = {}
            self._zero_current_vector_plan_cache = {}
            self._generator_measurement_plan_cache = {}
            self._acac_measurement_plan_cache = {}
            self._balance_measurement_plan_cache = {}
            self._active_branch_transformer_vector_plan = None
            self._active_simple_jacobian_plan = None
            self._active_zero_current_vector_plan = None
            self._active_generator_measurement_plan = None
            self._active_acac_measurement_plan = None
            self._active_balance_measurement_plan = None
            self._active_normal_pattern = None
            self._active_normal_assembly_plan = None
            self._active_lower_normal_plan = None
            self.active_measurements_are_vectorized = False
        self._prepared = True
        return self

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    @staticmethod
    def _write_sparse_triplet_file(
        path: Path,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int],
        label: str,
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# sparse_triplet row col value\n")
            handle.write(f"# shape {int(shape[0])} {int(shape[1])} nnz {int(values.size)}\n")
            handle.write(f"# matrix {label}\n")
            if values.size == 0:
                return
            rows = rows + 1
            cols = cols + 1
            chunk_size = 200_000
            for start in range(0, int(values.size), chunk_size):
                end = min(start + chunk_size, int(values.size))
                block = np.column_stack((rows[start:end], cols[start:end], values[start:end]))
                np.savetxt(handle, block, fmt=("%d", "%d", "%.17e"))

    @staticmethod
    def _sparse_triplets_from_matrix(matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        if issparse(matrix):
            coo = matrix.tocoo(copy=False)
            return coo.row, coo.col, coo.data, tuple(int(item) for item in matrix.shape)
        arr = np.asarray(matrix, dtype=np.float64)
        rows, cols = np.nonzero(arr)
        return rows, cols, arr[rows, cols], tuple(int(item) for item in arr.shape)

    @staticmethod
    def _sparse_matrix_stores_lower_triangle(matrix) -> bool:
        if not issparse(matrix) or matrix.shape[0] != matrix.shape[1]:
            return False
        csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        counts = np.diff(csc.indptr).astype(np.int64, copy=False)
        cols = np.repeat(np.arange(int(csc.shape[1]), dtype=np.int64), counts)
        return bool(cols.size == 0 or np.all(csc.indices.astype(np.int64, copy=False) >= cols))

    def _dump_iteration_matrices(self, iteration: int, H, weight: np.ndarray, gain) -> None:
        dump_dir = self.matrix_dump_dir
        if dump_dir is None:
            return
        dump_dir = Path(dump_dir)
        rows, cols, values, shape = self._sparse_triplets_from_matrix(H)
        self._write_sparse_triplet_file(dump_dir / f"j{int(iteration)}.txt", rows, cols, values, shape, "jacobian")

        weight = np.asarray(weight, dtype=np.float64)
        diag = np.arange(int(weight.size), dtype=np.int64)
        self._write_sparse_triplet_file(
            dump_dir / f"d{int(iteration)}.txt",
            diag,
            diag,
            weight,
            (int(weight.size), int(weight.size)),
            "weight_diagonal",
        )

        info_matrix = full_normal_equation_from_lower(gain) if self._sparse_matrix_stores_lower_triangle(gain) else gain
        rows, cols, values, shape = self._sparse_triplets_from_matrix(info_matrix)
        self._write_sparse_triplet_file(
            dump_dir / f"h{int(iteration)}.txt",
            rows,
            cols,
            values,
            shape,
            "information_matrix",
        )

    def _warn_required_runtime_missing(self, name: str, context: str) -> None:
        message = (
            f"AC SE requires {name} during {context}; object-network/no-PPC fallback is disabled."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise RuntimeError(message)

    def _required_array_attr(self, name: str, *, dtype=None, context: str = "runtime") -> np.ndarray:
        if not hasattr(self, name):
            self._warn_required_runtime_missing(name, context)
        value = getattr(self, name)
        if value is None:
            self._warn_required_runtime_missing(name, context)
        return np.asarray(value, dtype=dtype)

    def _warn_missing_required_active_measurements(self, context: str) -> None:
        warnings.warn(
            f"AC SE requires active_measurements during {context}; temporary rebuild fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _active_measurement_table_ref(self) -> MeasurementTable:
        table = getattr(self, "active_measurement_table", None)
        if table is None:
            active = getattr(self, "active_measurements", None)
            table = getattr(active, "table", None)
        if table is None:
            self._warn_missing_required_active_measurements("active measurement table access")
            raise RuntimeError("AC SE requires active_measurement_table for active array execution")
        return table

    def _active_measurement_count(self) -> int:
        return int(self._active_measurement_table_ref().idx.size)

    def _measurement_count(self, measurements_or_plan_tables=None) -> int:
        if measurements_or_plan_tables is None:
            return self._active_measurement_count()
        if isinstance(measurements_or_plan_tables, dict) or isinstance(measurements_or_plan_tables, MeasurementPlanTable):
            return int(self._common_measurement_plan_table(measurements_or_plan_tables).row.size)
        return len(measurements_or_plan_tables)

    def _active_measurement_plan_table(self, plan_name: str) -> MeasurementPlanTable:
        return self._active_measurement_plan_tables_ref()[plan_name]

    def _active_measurement_plan_tables_ref(self) -> Dict[str, MeasurementPlanTable]:
        plan_tables = getattr(self, "_active_measurement_plan_tables_cache", None)
        if plan_tables is None:
            plan_tables = self._active_measurement_plan_tables(self._active_measurement_table_ref())
            self._active_measurement_plan_tables_cache = plan_tables
        return plan_tables

    def _measurement_plan_tables_are_active(self, plan_tables: Dict[str, MeasurementPlanTable]) -> bool:
        return plan_tables is getattr(self, "_active_measurement_plan_tables_cache", None)

    @staticmethod
    def _common_measurement_plan_table(plan_tables) -> MeasurementPlanTable:
        if isinstance(plan_tables, MeasurementPlanTable):
            return plan_tables
        if not isinstance(plan_tables, dict) or not plan_tables:
            raise TypeError("AC SE measurement runtime requires MeasurementPlanTable objects")
        table = plan_tables.get("simple")
        if table is not None:
            return table
        return next(iter(plan_tables.values()))

    def _measurement_plan_tables_for(self, measurements_or_plan_tables=None) -> Dict[str, MeasurementPlanTable]:
        if isinstance(measurements_or_plan_tables, dict):
            return measurements_or_plan_tables
        if isinstance(measurements_or_plan_tables, MeasurementPlanTable):
            return {"simple": measurements_or_plan_tables}
        if measurements_or_plan_tables is None:
            return self._active_measurement_plan_tables_ref()
        measurements = self._normalize_measurements(measurements_or_plan_tables)
        if measurements is None:
            return self._active_measurement_plan_tables_ref()
        table = self._measurement_table_for_indexed_plan(measurements)
        return self._active_measurement_plan_tables(table)

    def _vector_plans_for_measurement_plan_tables(
        self,
        plan_tables: Dict[str, MeasurementPlanTable],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if self._measurement_plan_tables_are_active(plan_tables):
            if getattr(self, "_active_branch_transformer_vector_plan", None) is None:
                self._active_branch_transformer_vector_plan = self._build_branch_transformer_vector_plan_from_table(
                    plan_tables["branch_transformer"]
                )
            if getattr(self, "_active_zero_current_vector_plan", None) is None:
                self._active_zero_current_vector_plan = self._build_zero_current_vector_plan_from_table(
                    plan_tables["zero_current"]
                )
            if getattr(self, "_active_simple_jacobian_plan", None) is None:
                self._active_simple_jacobian_plan = self._build_simple_jacobian_plan_from_table(
                    plan_tables["simple"]
                )
            if getattr(self, "_active_balance_measurement_plan", None) is None:
                self._active_balance_measurement_plan = self._build_balance_measurement_plan_from_table(
                    plan_tables["balance"]
                )
            if getattr(self, "_active_generator_measurement_plan", None) is None:
                self._active_generator_measurement_plan = self._build_generator_measurement_plan_from_table(
                    plan_tables["generator"]
                )
            if getattr(self, "_active_acac_measurement_plan", None) is None:
                self._active_acac_measurement_plan = self._build_acac_measurement_plan_from_table(
                    plan_tables["acac"]
                )
            return {
                "branch_transformer": self._active_branch_transformer_vector_plan,
                "zero_current": self._active_zero_current_vector_plan,
                "simple": self._active_simple_jacobian_plan,
                "balance": self._active_balance_measurement_plan,
                "generator": self._active_generator_measurement_plan,
                "acac": self._active_acac_measurement_plan,
            }
        return {
            "branch_transformer": self._build_branch_transformer_vector_plan_from_table(
                plan_tables["branch_transformer"]
            ),
            "zero_current": self._build_zero_current_vector_plan_from_table(plan_tables["zero_current"]),
            "simple": self._build_simple_jacobian_plan_from_table(plan_tables["simple"]),
            "balance": self._build_balance_measurement_plan_from_table(plan_tables["balance"]),
            "generator": self._build_generator_measurement_plan_from_table(plan_tables["generator"]),
            "acac": self._build_acac_measurement_plan_from_table(plan_tables["acac"]),
        }

    def _default_observability_cache_key(self) -> Tuple[object, ...]:
        return (
            _file_cache_key(self.e_file),
            _file_cache_key(self.meas_file),
            bool(self.flat_start),
            int(self.targeted_pseudo_measurement_max),
            float(self.targeted_pseudo_measurement_redundancy_ratio),
            int(self.targeted_pseudo_measurement_step),
            int(self._active_measurement_count()),
            int(self._max_measurement_idx),
            int(self.n_state),
        )

    def _observability_cache_allowed(self) -> bool:
        return "jacobian_sparse" not in getattr(self, "__dict__", {})

    def _aat_normal_solver_enabled(self) -> bool:
        if CHOLMOD_ANALYZE_AAT is None and CHOLMOD_CHOLESKY_AAT is None:
            return False
        if self.matrix_dump_dir is not None:
            return False
        if int(getattr(self, "n_state", 0)) < _AAT_NORMAL_SOLVER_MIN_STATE:
            return False
        return bool(
            getattr(self, "_array_only_runtime", False)
            or getattr(self, "_array_only_estimate_result", False)
        )

    def _cache_observability_matrix(
        self,
        result: ObservabilityResult,
        x: np.ndarray,
        measurements: Sequence[Measurement],
        H,
        *,
        cache_lower_normal_plan: bool = True,
    ) -> None:
        lower_normal_plan = None
        if (
            cache_lower_normal_plan
            and self._measurement_plan_tables_are_active(measurements)
            and issparse(H)
            and not self._aat_normal_solver_enabled()
        ):
            lower_normal_plan = getattr(self, "_active_lower_normal_plan", None)
            if lower_normal_plan is not None and not lower_normal_plan.matches(H):
                lower_normal_plan = None
            if lower_normal_plan is None:
                start = time.perf_counter() if self.profile_enabled else None
                lower_normal_plan = LowerNormalEquationCscPlan.from_jacobian(H)
                if start is not None:
                    profile_key = (
                        "solve.lower_normal_plan_build"
                        if bool(getattr(self, "_prepared", False))
                        else "init.lower_normal_plan_build"
                    )
                    self._record_profile_time(profile_key, time.perf_counter() - start)
            active_weight = getattr(self, "active_weight", None)
            if active_weight is not None and np.asarray(active_weight).size == int(H.shape[0]):
                lower_normal_plan.prepare_fixed_weights(np.asarray(active_weight, dtype=np.float64))
            self._active_lower_normal_plan = lower_normal_plan
        self._observability_matrix_cache = {
            "result": result,
            "measurements": measurements,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "H": H,
            "normal_pattern": None,
            "normal_assembly_plan": None,
            "lower_normal_plan": lower_normal_plan,
        }

    def _observability_matrix_cache_for(
        self,
        result: Optional[ObservabilityResult],
        measurements: Sequence[Measurement],
        x: np.ndarray,
    ):
        cache = getattr(self, "_observability_matrix_cache", None)
        if cache is None or cache.get("result") is not result:
            return None
        if cache.get("measurements") is not measurements:
            return None
        cached_x = cache.get("x")
        x_array = np.asarray(x, dtype=np.float64)
        if cached_x is None or cached_x.shape != x_array.shape or not np.array_equal(cached_x, x_array):
            return None
        return cache

    def _restore_active_lower_normal_plan_from_observability_cache(
        self,
        result: Optional[ObservabilityResult],
    ) -> None:
        cache = getattr(self, "_observability_matrix_cache", None)
        if cache is None or cache.get("result") is not result:
            return
        lower_normal_plan = cache.get("lower_normal_plan")
        if lower_normal_plan is not None:
            self._active_lower_normal_plan = lower_normal_plan

    def _refresh_active_measurement_indexes(self) -> None:
        """Rebuild active measurement arrays and vectorized measurement plans."""
        self._initial_observability_cache = None
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=False,
        )
        table = active_view.source_table
        active_table = active_view.table
        self.measurement_table = table
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_angle_residual_mask = active_view.angle_mask
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._acac_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        active_plan_tables = self._active_measurement_plan_tables(active_table)
        self._active_measurement_plan_tables_cache = active_plan_tables
        self._active_branch_transformer_vector_plan = self._build_branch_transformer_vector_plan_from_table(
            active_plan_tables["branch_transformer"]
        )
        self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_branch_transformer_vector_plan,
        )
        self._active_simple_jacobian_plan = self._build_simple_jacobian_plan_from_table(
            active_plan_tables["simple"]
        )
        self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_simple_jacobian_plan,
        )
        self._active_zero_current_vector_plan = self._build_zero_current_vector_plan_from_table(
            active_plan_tables["zero_current"]
        )
        self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_zero_current_vector_plan,
        )
        self._active_generator_measurement_plan = self._build_generator_measurement_plan_from_table(
            active_plan_tables["generator"]
        )
        self._generator_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        self._active_acac_measurement_plan = self._build_acac_measurement_plan_from_table(
            active_plan_tables["acac"]
        )
        self._acac_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_acac_measurement_plan,
        )
        self._active_balance_measurement_plan = self._build_balance_measurement_plan_from_table(
            active_plan_tables["balance"]
        )
        self._balance_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_balance_measurement_plan,
        )
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        self._observability_matrix_cache = None
        self.active_measurements_are_vectorized = bool(
            np.all(
                self._active_branch_transformer_vector_plan["handled_mask"]
                | self._active_simple_jacobian_plan["handled_mask"]
                | self._active_zero_current_vector_plan["handled_mask"]
                | self._active_generator_measurement_plan["handled_mask"]
                | self._active_acac_measurement_plan["handled_mask"]
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )

    def _incremental_update_active_measurement_indexes(
        self,
        appended_measurements: Sequence[Measurement],
        *,
        source_row_start: Optional[int] = None,
        master_table: Optional[MeasurementTable] = None,
    ) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            self._warn_missing_required_active_measurements("incremental active measurement update")
            return False
        appended_table = _measurement_table_from_measurements(appended_measurements)
        if np.any(
            (~np.asarray(appended_table.valid, dtype=bool))
            | (np.asarray(appended_table.weight, dtype=np.float64) <= 0.0)
        ):
            return False
        appended_device_pos = getattr(appended_table, "device_pos", None)
        if appended_device_pos is None or np.asarray(appended_device_pos).size != int(appended_table.idx.size):
            warnings.warn(
                "AC SE incremental active update requires appended measurement device_pos; "
                "name-based index rebuild is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        master_table = master_table if master_table is not None else getattr(
            self,
            "measurement_table",
            getattr(self.measurements, "table", None),
        )
        active_table = getattr(self, "active_measurement_table", getattr(self.active_measurements, "table", None))
        if master_table is None or active_table is None:
            return False
        source_row_start = int(len(master_table.idx) if source_row_start is None else source_row_start)
        if len(master_table.idx) != source_row_start:
            return False
        if len(active_table.idx) != len(self.active_measurements):
            return False

        appended_view = MeasurementTableView(
            appended_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurement_table = concat_measurement_tables(master_table, appended_table)
        try:
            self.measurements.table = self.measurement_table
        except AttributeError:
            pass
        active_view = append_active_measurement_view(
            build_active_measurement_view(
                self.active_measurements,
                table_builder=_measurement_table_from_measurements,
                materialize_measurements=False,
            ),
            appended_view,
            source_row_start=source_row_start,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=False,
        )
        self._initial_observability_cache = None
        self._max_measurement_idx = int(self.measurement_table.idx.max()) if self.measurement_table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_view.table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_angle_residual_mask = active_view.angle_mask
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        self._active_measurement_plan_tables_cache = None
        row_offset = len(active_table.idx)
        previous_branch_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
        previous_simple_plan = getattr(self, "_active_simple_jacobian_plan", None)
        previous_zero_plan = getattr(self, "_active_zero_current_vector_plan", None)
        previous_generator_plan = getattr(self, "_active_generator_measurement_plan", None)
        previous_acac_plan = getattr(self, "_active_acac_measurement_plan", None)
        previous_balance_plan = getattr(self, "_active_balance_measurement_plan", None)
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._acac_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        if previous_branch_plan is None:
            self._active_branch_transformer_vector_plan = self._branch_transformer_vector_plan(None)
        else:
            self._active_branch_transformer_vector_plan = self._merge_active_plan_dict(
                previous_branch_plan,
                self._build_branch_transformer_vector_plan(appended_view),
                row_offset=row_offset,
                row_keys=("voltage_rows", "power_rows", "current_rows"),
            )
        self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_branch_transformer_vector_plan,
        )
        if previous_simple_plan is None:
            self._active_simple_jacobian_plan = self._simple_jacobian_plan(None)
        else:
            self._active_simple_jacobian_plan = self._merge_active_plan_dict(
                previous_simple_plan,
                self._build_simple_jacobian_plan(appended_view),
                row_offset=row_offset,
                row_keys=(
                    "scalar_rows",
                    "value_voltage_rows",
                    "value_angle_rows",
                    "value_voltage_diff_rows",
                    "value_angle_diff_rows",
                    "load_rows",
                ),
            )
        self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_simple_jacobian_plan,
        )
        if previous_zero_plan is None:
            self._active_zero_current_vector_plan = self._zero_current_vector_plan(None)
        else:
            self._active_zero_current_vector_plan = self._merge_active_plan_dict(
                previous_zero_plan,
                self._build_zero_current_vector_plan(appended_view),
                row_offset=row_offset,
                row_keys=(
                    "scalar_rows",
                    "voltage_rows",
                    "angle_diff_rows",
                    "voltage_diff_rows",
                    "power_rows",
                    "current_rows",
                ),
            )
        self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_zero_current_vector_plan,
        )
        if previous_generator_plan is None:
            self._active_generator_measurement_plan = self._generator_measurement_plan(None)
        else:
            self._active_generator_measurement_plan = self._merge_active_plan_dict(
                previous_generator_plan,
                self._build_generator_measurement_plan(appended_view),
                row_offset=row_offset,
                row_keys=("value_rows",),
            )
        self._generator_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        if previous_acac_plan is None:
            self._active_acac_measurement_plan = self._acac_measurement_plan(None)
        else:
            self._active_acac_measurement_plan = self._merge_active_plan_dict(
                previous_acac_plan,
                self._build_acac_measurement_plan(appended_view),
                row_offset=row_offset,
                row_keys=("value_rows",),
            )
        self._acac_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_acac_measurement_plan,
        )
        if previous_balance_plan is None:
            self._active_balance_measurement_plan = self._balance_measurement_plan(None)
        else:
            self._active_balance_measurement_plan = self._merge_active_plan_dict(
                previous_balance_plan,
                self._build_balance_measurement_plan(appended_view),
                row_offset=row_offset,
                row_keys=("rows",),
                mapped_row_keys=("p_row_by_pos", "q_row_by_pos"),
            )
        self._balance_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_balance_measurement_plan,
        )
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        self._observability_matrix_cache = None
        self.active_measurements_are_vectorized = bool(
            np.all(
                self._active_branch_transformer_vector_plan["handled_mask"]
                | self._active_simple_jacobian_plan["handled_mask"]
                | self._active_zero_current_vector_plan["handled_mask"]
                | self._active_generator_measurement_plan["handled_mask"]
                | self._active_acac_measurement_plan["handled_mask"]
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )
        return True

    @staticmethod
    def _merge_active_plan_dict(
        head: Dict[str, object],
        tail: Dict[str, object],
        *,
        row_offset: int,
        row_keys: Sequence[str],
        mapped_row_keys: Sequence[str] = (),
    ) -> Dict[str, object]:
        row_key_tuple = tuple(row_keys)
        mapped_key_tuple = tuple(mapped_row_keys)
        merged: Dict[str, object] = {}
        for key, head_value in head.items():
            tail_value = tail[key]
            if key in row_key_tuple:
                merged[key] = np.concatenate(
                    (
                        np.asarray(head_value, dtype=np.int64),
                        np.asarray(tail_value, dtype=np.int64) + int(row_offset),
                    )
                ).astype(np.int64, copy=False)
                continue
            if key in mapped_key_tuple:
                mapped = np.asarray(head_value, dtype=np.int32).copy()
                tail_rows = np.asarray(tail_value, dtype=np.int32)
                valid = tail_rows >= 0
                mapped[valid] = tail_rows[valid] + int(row_offset)
                merged[key] = mapped
                continue
            merged[key] = np.concatenate((np.asarray(head_value), np.asarray(tail_value)))
        return merged

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            self._active_measurement_table_ref()
            return None
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return measurements
        if isinstance(measurements, list):
            active = getattr(self, "active_measurements", None)
            active_table = getattr(active, "table", None)
            if (
                isinstance(active, MeasurementList)
                and active_table is not None
                and len(active_table.idx) == len(active)
            ):
                cache = getattr(self, "_active_measurement_object_row_cache", None)
                cache_key = (id(active), len(active))
                if cache is None or cache[0] != cache_key:
                    cache = (cache_key, {id(meas): row for row, meas in enumerate(active)})
                    self._active_measurement_object_row_cache = cache
                id_to_row = cache[1]
                rows = np.asarray([id_to_row.get(id(meas), -1) for meas in measurements], dtype=np.int64)
                if rows.size and np.all(rows >= 0):
                    return take_measurement_view(active, rows)
                if rows.size == 0:
                    return take_measurement_view(active, rows)
            return measurements
        return list(measurements)

    def _measurement_vectors(self, measurements) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable):
            table = self._common_measurement_plan_table(measurements).table
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        if measurements is None:
            table = self._active_measurement_table_ref()
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        table = getattr(measurements, "table", None)
        if table is not None and len(table.value) == len(measurements):
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        return (
            np.asarray([meas.value for meas in measurements], dtype=np.float64),
            np.asarray([meas.weight for meas in measurements], dtype=np.float64),
        )

    @staticmethod
    def _uniform_weight(weight: np.ndarray) -> Optional[float]:
        if weight.size == 0:
            return None
        first_weight = float(weight[0])
        return first_weight if bool(np.all(weight == first_weight)) else None

    def _angle_residual_mask(self, measurements) -> np.ndarray:
        if isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable):
            return np.asarray(self._common_measurement_plan_table(measurements).table.angle_mask, dtype=bool)
        if measurements is None:
            return np.asarray(self._active_measurement_table_ref().angle_mask, dtype=bool)
        table = getattr(measurements, "table", None)
        if table is not None and len(table.angle_mask) == len(measurements):
            return np.asarray(table.angle_mask, dtype=bool)
        return angle_residual_mask(measurements)

    def _has_angle_residuals(self, measurements, angle_mask: np.ndarray) -> bool:
        if isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable):
            return bool(np.any(angle_mask))
        if measurements is None:
            return self.active_has_angle_residuals
        return bool(np.any(angle_mask))

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        angle_mask = self._angle_residual_mask(measurements)
        has_angle = self._has_angle_residuals(measurements, angle_mask)
        if out is None:
            residual = np.subtract(z, z_est, dtype=np.float64)
        else:
            np.subtract(z, z_est, out=out)
            residual = out
        if has_angle:
            residual[angle_mask] = (residual[angle_mask] + np.pi) % (2.0 * np.pi) - np.pi
        return residual

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _load_network(self, e_file: Path) -> SimpleNamespace:
        """Read the AC case into PPC/base/topology only."""
        source = Path(e_file).resolve()
        ppc = build_ac_ppc_with_topology_from_e_file(source)
        return _build_ac_se_ppc_namespace(ppc, source)

    @staticmethod
    def _run_power_flow_seed(network, params: StateEstimationParameters, e_file: Path) -> bool:
        """Run one AC load-flow solve so non-flat SE starts from a measured operating point."""
        seed_tol = max(float(params.power_flow_tol), 1e-6)
        try:
            ppc = ACStateEstimator._power_flow_seed_ppc_from_network(network)
        except RuntimeError:
            return False
        calc = ACPowerFlowCalc(
            ppc,
            tol=seed_tol,
            max_iter=params.power_flow_max_iter,
            min_voltage=params.power_flow_min_voltage,
            linear_solver=getattr(network, "_se_power_flow_linear_solver", None),
        )
        calc.skip_lf_result = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        rc = calc.run(result_mode="none")
                    except TypeError as exc:
                        if "result_mode" not in str(exc):
                            raise
                        rc = calc.run()
            if rc != 0 or not calc.converged:
                return False
            ACStateEstimator._apply_power_flow_seed_calc_state_to_network(network, ppc, calc)
            return True
        except Exception:
            return False

    @staticmethod
    def _copy_ppc_for_power_flow_seed(ppc):
        copied = dict(ppc)
        for key in ("bus", "gen", "load"):
            value = ppc.get(key)
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
        for key, value in ppc.items():
            if isinstance(value, dict):
                copied[key] = dict(value)
        return copied

    @staticmethod
    def _power_flow_seed_ppc_from_network(network):
        source = getattr(network, "ppc", None)
        if not (isinstance(source, dict) and source.get("format") == "ac_ppc_v1"):
            message = "AC SE power-flow seeding requires a PPC-backed network; object-network fallback is disabled."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            raise RuntimeError(message)
        ppc = ACStateEstimator._copy_ppc_for_power_flow_seed(source)
        seed_rows = getattr(network, "_se_power_flow_seed_rows", None)
        if seed_rows is None:
            warnings.warn(
                "AC SE power-flow seeding requires cached PPC seed rows; measurement-object scan is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            ACStateEstimator._apply_power_flow_seed_rows_to_ppc(ppc, seed_rows)
        return ppc

    @staticmethod
    def _row_by_idx(array: np.ndarray, idx_col: int) -> Dict[int, int]:
        if array is None or array.size == 0:
            return {}
        return {int(row[idx_col]): pos for pos, row in enumerate(array)}

    @staticmethod
    def _apply_power_flow_seed_rows_to_ppc(ppc, seed_rows) -> None:
        bus = ppc.get("bus")
        if bus is None:
            return
        if not isinstance(seed_rows, dict):
            warnings.warn(
                "AC SE power-flow seed requires packed key arrays; row tuple fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        measurement_key = np.asarray(seed_rows.get("measurement_key", ()), dtype=np.int64)
        device_row = np.asarray(seed_rows.get("ppc_row", ()), dtype=np.int64)
        values = np.asarray(seed_rows.get("value", ()), dtype=np.float64)
        if measurement_key.size == 0:
            return
        if device_row.size != measurement_key.size or values.size != measurement_key.size:
            warnings.warn(
                "AC SE power-flow seed requires same-sized measurement_key, ppc_row and value arrays.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        meas_mask = (1 << _ACTIVE_MEASUREMENT_KEY_MEAS_BITS) - 1
        device_key = measurement_key >> _ACTIVE_MEASUREMENT_KEY_MEAS_BITS
        meas_type = (measurement_key & meas_mask).astype(np.int16, copy=False)
        device_type = (device_key >> _ACTIVE_DEVICE_KEY_POS_BITS).astype(np.int16, copy=False)

        gen = ppc.get("gen")
        load = ppc.get("load")

        def valid_rows(mask: np.ndarray, row_count: int) -> Tuple[np.ndarray, np.ndarray]:
            rows = device_row[mask]
            valid = (rows >= 0) & (rows < row_count)
            return rows[valid].astype(np.intp, copy=False), values[mask][valid]

        bus_idx = None
        bus_order = None
        sorted_bus_idx = None

        def set_bus_voltage_by_idx(node_idx_values: np.ndarray, voltage_values: np.ndarray) -> None:
            nonlocal bus_idx, bus_order, sorted_bus_idx
            if node_idx_values.size == 0:
                return
            if bus_idx is None:
                bus_idx = bus[:, BUS_COLS["idx"]].astype(np.int64, copy=False)
                bus_order = np.argsort(bus_idx, kind="stable")
                sorted_bus_idx = bus_idx[bus_order]
            node_idx_values = np.asarray(node_idx_values, dtype=np.int64)
            pos = np.searchsorted(sorted_bus_idx, node_idx_values)
            in_range = pos < sorted_bus_idx.size
            if not np.any(in_range):
                return
            valid_pos = pos[in_range]
            valid_nodes = node_idx_values[in_range]
            hit = sorted_bus_idx[valid_pos] == valid_nodes
            if not np.any(hit):
                return
            bus_rows = bus_order[valid_pos[hit]].astype(np.intp, copy=False)
            bus[bus_rows, BUS_COLS["voltage"]] = np.maximum(voltage_values[in_range][hit], 0.0)

        mask = (device_type == DEVICE_TYPE_ACNode) & (meas_type == MEAS_TYPE_V)
        rows, row_values = valid_rows(mask, bus.shape[0])
        if rows.size:
            bus[rows, BUS_COLS["voltage"]] = np.maximum(row_values, 0.0)

        if isinstance(gen, np.ndarray) and gen.size:
            gen_device = device_type == DEVICE_TYPE_ACGenerator
            mask = gen_device & (meas_type == MEAS_TYPE_P_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                gen[rows, GEN_COLS["p_set"]] = row_values
                gen[rows, GEN_COLS["p"]] = row_values
            mask = gen_device & (meas_type == MEAS_TYPE_Q_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                gen[rows, GEN_COLS["q_set"]] = row_values
                gen[rows, GEN_COLS["q"]] = row_values
            mask = gen_device & (meas_type == MEAS_TYPE_V_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                voltage = np.maximum(row_values, 0.0)
                gen[rows, GEN_COLS["v_set"]] = voltage
                set_bus_voltage_by_idx(gen[rows, GEN_COLS["node"]], voltage)
            mask = gen_device & (meas_type == MEAS_TYPE_I_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                gen[rows, GEN_COLS["current"]] = row_values

        if isinstance(load, np.ndarray) and load.size:
            load_device = device_type == DEVICE_TYPE_ACLoad
            mask = load_device & (meas_type == MEAS_TYPE_P_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                load[rows, LOAD_COLS["pbase"]] = 1.0
                load[rows, LOAD_COLS["pv0"]] = row_values
                load[rows, LOAD_COLS["pv1"]] = 0.0
                load[rows, LOAD_COLS["pv2"]] = 0.0
                load[rows, LOAD_COLS["p"]] = row_values
            mask = load_device & (meas_type == MEAS_TYPE_Q_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                load[rows, LOAD_COLS["qbase"]] = 1.0
                load[rows, LOAD_COLS["qv0"]] = row_values
                load[rows, LOAD_COLS["qv1"]] = 0.0
                load[rows, LOAD_COLS["qv2"]] = 0.0
                load[rows, LOAD_COLS["q"]] = row_values
            mask = load_device & (meas_type == MEAS_TYPE_V_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                set_bus_voltage_by_idx(load[rows, LOAD_COLS["node"]], row_values)
            mask = load_device & (meas_type == MEAS_TYPE_I_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                load[rows, LOAD_COLS["current"]] = row_values

    @staticmethod
    def _overlay_power_flow_seed_result_ppc(seed_ppc, result):
        if not isinstance(result, dict):
            return seed_ppc
        for key in (
            "bus",
            "gen",
            "load",
            "shunt",
            "branch",
            "transformer",
            "zero_branch",
            "switch",
            "break",
        ):
            value = result.get(key)
            if isinstance(value, np.ndarray):
                seed_ppc[key] = value
        return seed_ppc

    @staticmethod
    def _apply_power_flow_seed_ppc_to_network(network, ppc) -> None:
        network.ppc = ppc
        if hasattr(network, "topology"):
            network.topology = ppc.get("_topology_arrays", network.topology)

    @staticmethod
    def _apply_power_flow_seed_calc_state_to_network(network, ppc, calc) -> None:
        if hasattr(calc, "x") and hasattr(calc, "_extract_state_vars"):
            theta, voltage, _phi_re, _phi_im = calc._extract_state_vars(calc.x, update_cache=False)
            bus = ppc.get("bus")
            if isinstance(bus, np.ndarray) and bus.size:
                bus[:, BUS_COLS["voltage"]] = np.asarray(voltage, dtype=np.float64)
                bus[:, BUS_COLS["angle"]] = np.asarray(theta, dtype=np.float64)
            ACStateEstimator._apply_power_flow_seed_ppc_to_network(network, ppc)
            return
        result_ppc = ACStateEstimator._overlay_power_flow_seed_result_ppc(ppc, getattr(calc, "result", None))
        ACStateEstimator._apply_power_flow_seed_ppc_to_network(network, result_ppc)

    def _refresh_file_state_from_network(self) -> None:
        ppc = self._ac_ppc_dict()
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "file-state refresh")
        bus_solver_pos = self._topology_bus_solver_pos(topology)
        file_theta = np.zeros(self.n_nodes, dtype=np.float64)
        file_voltage = np.ones(self.n_nodes, dtype=np.float64)
        bus_solver_pos = np.asarray(bus_solver_pos, dtype=np.int32)
        offsets = np.asarray(topology.bus_node_offsets, dtype=np.int64)
        starts = offsets[:-1]
        valid = (
            (bus_solver_pos >= 0)
            & (bus_solver_pos < self.n_nodes)
            & (starts < offsets[1:])
        )
        if np.any(valid):
            solver_pos = bus_solver_pos[valid].astype(np.intp, copy=False)
            node_rows = np.asarray(topology.bus_node_indices, dtype=np.int64)[starts[valid]].astype(np.intp, copy=False)
            voltage = bus[node_rows, BUS_COLS["voltage"]]
            voltage = np.maximum(np.where(voltage != 0.0, voltage, 1.0), self.voltage_floor)
            file_voltage[solver_pos] = voltage
            file_theta[solver_pos] = bus[node_rows, BUS_COLS["angle"]]
        self.file_theta = file_theta
        self.file_voltage = file_voltage

    def _apply_measurement_seed_to_network(self) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is not None:
            setattr(self.network, "_se_power_flow_seed_rows", seed_rows)
            self._ac_ppc_dict()
            return
        warnings.warn(
            "AC SE power-flow seeding requires cached PPC seed rows; measurement-object scan is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _measurement_table_to_meas_ppc(self, table: MeasurementTable) -> Dict:
        """Create measurement PPC arrays for explicit measurement sequences."""
        row_count = int(table.idx.size)
        device_name_array = np.asarray(table.device_name, dtype=object)
        if row_count and int(device_name_array.size) == row_count:
            device_names, device_name_id = np.unique(device_name_array, return_inverse=True)
            device_name_id = device_name_id.astype(np.int64, copy=False)
        elif row_count:
            device_name_id = getattr(table, "device_name_id", None)
            if device_name_id is None or np.asarray(device_name_id).size != row_count:
                device_name_id = np.full(row_count, -1, dtype=np.int64)
            else:
                device_name_id = np.asarray(device_name_id, dtype=np.int64)
            device_names = np.asarray([], dtype=object)
        else:
            device_names = np.asarray([], dtype=object)
            device_name_id = np.asarray([], dtype=np.int64)
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is None or np.asarray(meas_type_code).size != row_count:
            warnings.warn(
                "AC SE measurement PPC export requires meas_type_code; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            meas_type_code = np.zeros(row_count, dtype=np.int16)
            table.meas_type_code = meas_type_code
        status_code = measurement_table_status_code(table)
        meas = np.zeros((row_count, len(MEAS_COLS)), dtype=np.float64)
        if row_count:
            meas[:, MEAS_COLS["idx"]] = np.asarray(table.idx, dtype=np.float64)
            meas[:, MEAS_COLS["device_type_code"]] = np.asarray(table.device_type_code, dtype=np.float64)
            meas[:, MEAS_COLS["device_name_id"]] = device_name_id.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["meas_type_code"]] = np.asarray(meas_type_code, dtype=np.float64)
            meas[:, MEAS_COLS["weight"]] = np.asarray(table.weight, dtype=np.float64)
            meas[:, MEAS_COLS["valid"]] = np.asarray(table.valid, dtype=np.float64)
            meas[:, MEAS_COLS["value"]] = np.asarray(table.value, dtype=np.float64)
            meas[:, MEAS_COLS["status"]] = np.asarray(status_code, dtype=np.float64)
            meas[:, MEAS_COLS["angle_mask"]] = np.asarray(table.angle_mask, dtype=np.float64)
            meas[:, MEAS_COLS["source_row"]] = np.arange(row_count, dtype=np.float64)
        table.device_name_id = device_name_id
        return {
            "format": "meas_ppc_v1",
            "source": str(Path(getattr(self, "meas_file", "")).resolve()) if getattr(self, "meas_file", None) else "",
            "meas": meas,
            "meas_cols": MEAS_COLS,
            "meas_type_codes": MEAS_TYPE_CODES,
            "idx_array": np.asarray(table.idx, dtype=np.int64),
            "weight_array": np.asarray(table.weight, dtype=np.float64),
            "valid_array": np.asarray(table.valid, dtype=bool),
            "value_array": np.asarray(table.value, dtype=np.float64),
            "status_array": np.asarray(status_code, dtype=np.int16),
            "device_type_code_array": np.asarray(table.device_type_code, dtype=np.int16),
            "device_name_id_array": device_name_id,
            "meas_type_code_array": np.asarray(meas_type_code, dtype=np.int16),
            "angle_mask_array": np.asarray(table.angle_mask, dtype=bool),
            "name": np.asarray(table.name, dtype=object),
            "device_type": np.asarray(table.device_type, dtype=object),
            "device_name": device_name_array,
            "device_names": np.asarray(device_names, dtype=object),
            "meas_type": np.asarray(table.meas_type, dtype=object),
            "rows_by_device_type_code": rows_by_device_type_code(table),
            "normalized": bool(getattr(getattr(self, "measurements", None), "normalized", False)),
        }

    def _attach_meas_ppc_device_pos(self) -> None:
        meas_ppc = getattr(self, "meas_ppc", None)
        if not isinstance(meas_ppc, dict):
            return
        self._ensure_measurement_plan_lookup_arrays()
        rows_by_code = getattr(self, "_ac_measurement_plan_rows_by_type_code", None)
        if not rows_by_code:
            return
        ppc = self._ac_ppc_dict()
        name_key_by_code = {
            DEVICE_TYPE_ACNode: "bus_name",
            DEVICE_TYPE_ACBranch: "branch_name",
            DEVICE_TYPE_ACTransformer: "transformer_name",
            DEVICE_TYPE_ACLoad: "load_name",
            DEVICE_TYPE_ACGenerator: "gen_name",
            DEVICE_TYPE_ACZeroBranch: "zero_branch_name",
            DEVICE_TYPE_ACBreak: "break_name",
            DEVICE_TYPE_ACACConverter: "acac_name",
            DEVICE_TYPE_ACZeroBranchConstraint: "zero_branch_name",
            DEVICE_TYPE_ACBreakConstraint: "break_name",
            DEVICE_TYPE_ACPowerBalance: "bus_name",
        }
        name_arrays_by_code = {}
        for code, rows in rows_by_code.items():
            name_key = name_key_by_code.get(int(code))
            if name_key is None:
                continue
            name_arrays_by_code[int(code)] = self._ppc_names_for_rows(
                ppc.get(name_key, np.asarray([], dtype=object)),
                np.asarray(rows, dtype=np.int64),
            )
        device_pos = attach_device_pos_from_name_arrays(meas_ppc, name_arrays_by_code)
        table = getattr(getattr(self, "measurements", None), "table", None)
        if table is not None and int(getattr(table, "idx", np.asarray([])).size) == int(device_pos.size):
            table.device_pos = device_pos

    def _measurement_device_pos_array(
        self,
        table: MeasurementTable,
        device_pos_by_type_code: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> np.ndarray:
        """Resolve each measurement row to the compact device position used by plan arrays."""
        precomputed = getattr(table, "device_pos", None)
        if precomputed is not None:
            precomputed = np.asarray(precomputed, dtype=np.int64)
            if precomputed.size == int(table.idx.size):
                return precomputed
        meas_ppc = getattr(self, "meas_ppc", None)
        if isinstance(meas_ppc, dict):
            ppc_device_pos = meas_ppc.get("device_pos")
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == int(table.idx.size):
                table.device_pos = np.asarray(ppc_device_pos, dtype=np.int64)
                return table.device_pos
        n_rows = int(table.idx.size)
        device_pos = np.empty(n_rows, dtype=np.int64)
        device_pos.fill(-1)
        if n_rows:
            warnings.warn(
                "AC SE measurement device_pos is missing; name-id/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        table.device_pos = device_pos
        return device_pos

    def _node_scale_arrays_by_pos(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (
            int(getattr(self, "n_nodes", 0)),
            id(getattr(self, "_ac_node_vbase_by_pos", None)),
            float(self.p_base_kW),
            float(self.u_scale),
            float(self.i_scale),
        )
        cached = getattr(self, "_ac_node_scale_by_pos_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        n_nodes = int(getattr(self, "n_nodes", 0))
        node_idx_by_pos = np.empty(n_nodes, dtype=np.int64)
        node_idx_by_pos.fill(-1)
        voltage_scale = np.empty(n_nodes, dtype=np.float64)
        voltage_scale.fill(np.nan)
        current_scale = np.empty(n_nodes, dtype=np.float64)
        current_scale.fill(np.nan)
        valid = np.zeros(n_nodes, dtype=bool)
        node_ids = np.asarray(getattr(self, "_ac_node_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        node_vbase = np.asarray(getattr(self, "_ac_node_vbase_by_pos", np.asarray([], dtype=np.float64)), dtype=np.float64)
        count = min(n_nodes, node_ids.size, node_vbase.size)
        if count:
            vbase_values = node_vbase[:count].astype(np.float64, copy=False)
            valid[:count] = np.isfinite(vbase_values) & (vbase_values > 0.0)
            node_idx_by_pos[:count] = node_ids[:count]
            voltage_scale[:count] = self.u_scale * vbase_values
            current_base = np.ones(count, dtype=np.float64)
            positive = np.abs(vbase_values) > 1e-12
            if np.any(positive):
                current_base[positive] = self.p_base_kW / (
                    1000.0 * math.sqrt(3.0) * np.abs(vbase_values[positive])
                )
            current_scale[:count] = self.i_scale * current_base
            current_scale[:count][~valid[:count]] = np.nan
            voltage_scale[:count][~valid[:count]] = np.nan
        result = (valid, voltage_scale, current_scale, node_idx_by_pos)
        self._ac_node_scale_by_pos_cache = (cache_key, result)
        return result

    @staticmethod
    def _values_for_plan_pos(values: np.ndarray, plan_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(plan_pos, dtype=np.int64)
        out = np.empty(pos.size, dtype=np.int64)
        out.fill(-1)
        valid = (pos >= 0) & (pos < int(values.size))
        if np.any(valid):
            out[valid] = values[pos[valid].astype(np.intp, copy=False)]
        return valid, out

    def _solver_pos_for_node_idx(self, node_idx: int) -> int:
        lookup_ids = np.asarray(getattr(self, "_ac_node_id_lookup_ids", ()), dtype=np.int64)
        lookup_pos = np.asarray(getattr(self, "_ac_node_id_lookup_pos", ()), dtype=np.int64)
        if lookup_ids.size == 0 or lookup_pos.size == 0:
            return -1
        pos = int(np.searchsorted(lookup_ids, int(node_idx)))
        if pos < lookup_ids.size and int(lookup_ids[pos]) == int(node_idx):
            return int(lookup_pos[pos])
        return -1

    def _measurement_scale_for_codes(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        rows_by_code: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return row availability, scale, from-node-pos and to-node-pos using integer metadata only."""
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        n_rows = int(device_type_code.size)
        available = np.zeros(n_rows, dtype=bool)
        scale = np.ones(n_rows, dtype=np.float64)
        from_pos = np.empty(n_rows, dtype=np.int64)
        to_pos = np.empty(n_rows, dtype=np.int64)
        from_pos.fill(-1)
        to_pos.fill(-1)
        node_valid, node_voltage_scale, node_current_scale, _node_idx_by_pos = self._node_scale_arrays_by_pos()

        def rows_for(device_code: int) -> np.ndarray:
            if rows_by_code is not None:
                rows = rows_by_code.get(int(device_code))
                if rows is None:
                    return np.asarray([], dtype=np.int64)
                return np.asarray(rows, dtype=np.int64)
            return np.flatnonzero(device_type_code == int(device_code)).astype(np.int64, copy=False)

        def apply_node(rows: np.ndarray, node_plan_pos: np.ndarray, voltage_code: int) -> None:
            if rows.size == 0:
                return
            valid_plan, node_pos = self._values_for_plan_pos(node_plan_pos, device_pos[rows])
            valid_node = valid_plan & (node_pos >= 0) & (node_pos < node_valid.size)
            if np.any(valid_node):
                valid_node[valid_node] &= node_valid[node_pos[valid_node].astype(np.intp, copy=False)]
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            pos_valid = node_pos[valid_node].astype(np.intp, copy=False)
            from_pos[rows_valid] = pos_valid
            available[rows_valid] = True
            voltage_mask = meas_type_code[rows_valid] == int(voltage_code)
            if np.any(voltage_mask):
                scale[rows_valid[voltage_mask]] = node_voltage_scale[pos_valid[voltage_mask]]

        def apply_terminal(rows: np.ndarray, i_plan_pos: np.ndarray, j_plan_pos: np.ndarray) -> None:
            if rows.size == 0:
                return
            i_valid_plan, i_pos = self._values_for_plan_pos(i_plan_pos, device_pos[rows])
            j_valid_plan, j_pos = self._values_for_plan_pos(j_plan_pos, device_pos[rows])
            valid_node = (
                i_valid_plan
                & j_valid_plan
                & (i_pos >= 0)
                & (j_pos >= 0)
                & (i_pos < node_valid.size)
                & (j_pos < node_valid.size)
            )
            if np.any(valid_node):
                idx = np.flatnonzero(valid_node)
                valid_node[idx] = (
                    node_valid[i_pos[idx].astype(np.intp, copy=False)]
                    & node_valid[j_pos[idx].astype(np.intp, copy=False)]
                )
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            i_valid = i_pos[valid_node].astype(np.intp, copy=False)
            j_valid = j_pos[valid_node].astype(np.intp, copy=False)
            from_pos[rows_valid] = i_valid
            to_pos[rows_valid] = j_valid
            available[rows_valid] = True
            mtypes = meas_type_code[rows_valid]
            power_mask = (
                (mtypes == MEAS_TYPE_P_FROM)
                | (mtypes == MEAS_TYPE_Q_FROM)
                | (mtypes == MEAS_TYPE_P_TO)
                | (mtypes == MEAS_TYPE_Q_TO)
            )
            scale[rows_valid[power_mask]] = self.p_base
            mask = mtypes == MEAS_TYPE_V_FROM
            scale[rows_valid[mask]] = node_voltage_scale[i_valid[mask]]
            mask = mtypes == MEAS_TYPE_I_FROM
            scale[rows_valid[mask]] = node_current_scale[i_valid[mask]]
            mask = mtypes == MEAS_TYPE_V_TO
            scale[rows_valid[mask]] = node_voltage_scale[j_valid[mask]]
            mask = mtypes == MEAS_TYPE_I_TO
            scale[rows_valid[mask]] = node_current_scale[j_valid[mask]]

        def apply_single(rows: np.ndarray, node_plan_pos: np.ndarray, power_codes: Tuple[int, int], voltage_code: int, current_code: int) -> None:
            if rows.size == 0:
                return
            valid_plan, node_pos = self._values_for_plan_pos(node_plan_pos, device_pos[rows])
            valid_node = valid_plan & (node_pos >= 0) & (node_pos < node_valid.size)
            if np.any(valid_node):
                valid_node[valid_node] &= node_valid[node_pos[valid_node].astype(np.intp, copy=False)]
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            pos_valid = node_pos[valid_node].astype(np.intp, copy=False)
            from_pos[rows_valid] = pos_valid
            available[rows_valid] = True
            mtypes = meas_type_code[rows_valid]
            power_mask = (mtypes == int(power_codes[0])) | (mtypes == int(power_codes[1]))
            scale[rows_valid[power_mask]] = self.p_base
            mask = mtypes == int(voltage_code)
            scale[rows_valid[mask]] = node_voltage_scale[pos_valid[mask]]
            mask = mtypes == int(current_code)
            scale[rows_valid[mask]] = node_current_scale[pos_valid[mask]]

        apply_node(
            rows_for(DEVICE_TYPE_ACNode),
            self._ac_node_plan_pos,
            MEAS_TYPE_V,
        )
        apply_terminal(
            rows_for(DEVICE_TYPE_ACBranch),
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
        )
        apply_terminal(
            rows_for(DEVICE_TYPE_ACTransformer),
            self._ac_transformer_plan_i,
            self._ac_transformer_plan_j,
        )
        apply_terminal(
            rows_for(DEVICE_TYPE_ACZeroBranch),
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
        )
        apply_terminal(
            rows_for(DEVICE_TYPE_ACBreak),
            self._ac_break_plan_i,
            self._ac_break_plan_j,
        )
        apply_terminal(
            rows_for(DEVICE_TYPE_ACACConverter),
            self._ac_acac_plan_i,
            self._ac_acac_plan_j,
        )
        apply_single(
            rows_for(DEVICE_TYPE_ACGenerator),
            self._ac_generator_plan_node_pos,
            (MEAS_TYPE_P_GEN, MEAS_TYPE_Q_GEN),
            MEAS_TYPE_V_GEN,
            MEAS_TYPE_I_GEN,
        )
        apply_single(
            rows_for(DEVICE_TYPE_ACLoad),
            self._ac_load_plan_node_pos,
            (MEAS_TYPE_P_LOAD, MEAS_TYPE_Q_LOAD),
            MEAS_TYPE_V_LOAD,
            MEAS_TYPE_I_LOAD,
        )
        return available, scale, from_pos, to_pos

    def _measurement_voltage_node_positions_from_codes(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map voltage measurement rows to from/to solver node positions with integer metadata only."""
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        n_rows = int(device_type_code.size)
        from_pos = np.full(n_rows, -1, dtype=np.int64)
        to_pos = np.full(n_rows, -1, dtype=np.int64)

        def assign_single(rows: np.ndarray, plan_values) -> None:
            if rows.size == 0:
                return
            valid, values = self._values_for_plan_pos(np.asarray(plan_values, dtype=np.int64), device_pos[rows])
            if np.any(valid):
                from_pos[rows[valid]] = values[valid]

        def assign_terminal(device_code: int, i_values, j_values) -> None:
            rows_from = np.flatnonzero(
                (device_type_code == int(device_code)) & (meas_type_code == MEAS_TYPE_V_FROM)
            ).astype(np.int64, copy=False)
            if rows_from.size:
                valid, values = self._values_for_plan_pos(np.asarray(i_values, dtype=np.int64), device_pos[rows_from])
                if np.any(valid):
                    from_pos[rows_from[valid]] = values[valid]
            rows_to = np.flatnonzero(
                (device_type_code == int(device_code)) & (meas_type_code == MEAS_TYPE_V_TO)
            ).astype(np.int64, copy=False)
            if rows_to.size:
                valid, values = self._values_for_plan_pos(np.asarray(j_values, dtype=np.int64), device_pos[rows_to])
                if np.any(valid):
                    to_pos[rows_to[valid]] = values[valid]

        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_ACNode) & (meas_type_code == MEAS_TYPE_V)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
        )
        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_ACGenerator) & (meas_type_code == MEAS_TYPE_V_GEN)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_generator_plan_node_pos", np.asarray([], dtype=np.int64)),
        )
        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_ACLoad) & (meas_type_code == MEAS_TYPE_V_LOAD)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_load_plan_node_pos", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_ACBranch,
            getattr(self, "_ac_branch_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_branch_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_ACTransformer,
            getattr(self, "_ac_transformer_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_transformer_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_ACZeroBranch,
            getattr(self, "_ac_zero_branch_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_zero_branch_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_ACBreak,
            getattr(self, "_ac_break_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_break_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_ACACConverter,
            getattr(self, "_ac_acac_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_acac_plan_j", np.asarray([], dtype=np.int64)),
        )
        return from_pos, to_pos

    def _measurement_ppc_rows_for_device_positions(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> np.ndarray:
        """Return PPC table rows for measurement device positions using prepared plan row arrays."""
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        out = np.full(device_type_code.size, -1, dtype=np.int64)
        rows_by_code = getattr(self, "_ac_measurement_plan_rows_by_type_code", None)
        if not rows_by_code:
            return out
        for code_int, rows in rows_by_code.items():
            rows = np.asarray(rows, dtype=np.int64)
            if rows.size == 0:
                continue
            target = np.flatnonzero(device_type_code == int(code_int)).astype(np.int64, copy=False)
            if target.size == 0:
                continue
            pos = device_pos[target]
            valid = (pos >= 0) & (pos < rows.size)
            if np.any(valid):
                out[target[valid]] = rows[pos[valid].astype(np.intp, copy=False)]
        return out

    def _plan_pos_for_ppc_rows(self, device_type_code: int, rows: np.ndarray, table_size: int) -> np.ndarray:
        """Map PPC table rows back to SE measurement plan positions for numeric key checks."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.size, -1, dtype=np.int64)
        plan_pos_by_row = getattr(self, "_ac_measurement_plan_pos_by_ppc_row", None)
        if plan_pos_by_row:
            lookup = plan_pos_by_row.get(int(device_type_code))
            if lookup is not None and rows.size:
                in_range = (rows >= 0) & (rows < lookup.size)
                if np.any(in_range):
                    out[in_range] = lookup[rows[in_range].astype(np.intp, copy=False)]
                return out
        rows_by_code = getattr(self, "_ac_measurement_plan_rows_by_type_code", None)
        if not rows_by_code or rows.size == 0:
            return out
        plan_rows = rows_by_code.get(int(device_type_code))
        if plan_rows is None:
            return out
        plan_rows = np.asarray(plan_rows, dtype=np.int64)
        if plan_rows.size == 0:
            return out
        lookup_size = max(int(table_size), int(np.max(plan_rows)) + 1, int(np.max(rows)) + 1)
        lookup = np.full(lookup_size, -1, dtype=np.int64)
        lookup[plan_rows.astype(np.intp, copy=False)] = np.arange(plan_rows.size, dtype=np.int64)
        in_range = (rows >= 0) & (rows < lookup.size)
        if np.any(in_range):
            out[in_range] = lookup[rows[in_range].astype(np.intp, copy=False)]
        return out

    @staticmethod
    def _measurement_key_array_from_cache(measurement_keys) -> np.ndarray:
        if isinstance(measurement_keys, np.ndarray):
            keys = np.asarray(measurement_keys, dtype=np.int64).reshape(-1)
        elif isinstance(measurement_keys, _PackedMeasurementKeyCache):
            return measurement_keys._key_array()
        elif not measurement_keys:
            return np.empty(0, dtype=np.int64)
        else:
            keys = np.fromiter(
                (int(key) for key in measurement_keys),
                dtype=np.int64,
                count=len(measurement_keys),
            )
        if keys.size == 0:
            return keys.astype(np.int64, copy=False)
        return keys.astype(np.int64, copy=False)

    def _set_active_key_caches(self, measurement_keys: set) -> None:
        key_array = self._measurement_key_array_from_cache(measurement_keys)
        key_set = measurement_keys if isinstance(measurement_keys, set) else set(key_array.astype(object, copy=False))
        self._set_active_key_caches_from_array(key_array, key_set)

    def _set_active_key_caches_from_array(self, key_array: np.ndarray, key_set: Optional[set] = None) -> None:
        key_array = np.asarray(key_array, dtype=np.int64).reshape(-1)
        if key_set is None:
            key_set = _PackedMeasurementKeyCache(self)
            self._active_measurement_keys = key_set
            self._active_measurement_code_pos_cache = key_set
        else:
            self._active_measurement_keys = key_set
        self._active_measurement_code_pos_cache = key_set
        self._active_measurement_key_array_cache = key_array
        self._active_measurement_key_sorted_array_cache = None

    def _append_active_measurement_key_array_cache(self, measurement_keys) -> None:
        key_array = np.asarray(measurement_keys, dtype=np.int64).reshape(-1)
        if key_array.size == 0:
            return
        key_array = key_array.astype(np.int64, copy=False)
        current = getattr(self, "_active_measurement_key_array_cache", None)
        if current is None or np.asarray(current).size == 0:
            self._active_measurement_key_array_cache = key_array.copy()
        else:
            self._active_measurement_key_array_cache = np.concatenate(
                (
                    np.asarray(current, dtype=np.int64),
                    key_array,
                )
            )
        self._active_measurement_key_sorted_array_cache = None
        seen_cache_ids = set()
        for attr in (
            "_active_measurement_keys",
            "_active_measurement_code_pos_cache",
        ):
            cache = getattr(self, attr, None)
            cache_id = id(cache)
            if cache is None or cache_id in seen_cache_ids:
                continue
            seen_cache_ids.add(cache_id)
            if isinstance(cache, _PackedMeasurementKeyCache):
                cache._update_materialized(key_array)
            elif isinstance(cache, set):
                cache.update(key_array.astype(object, copy=False))

    @staticmethod
    def _active_measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
        return (
            (int(device_type_code) << (_ACTIVE_DEVICE_KEY_POS_BITS + _ACTIVE_MEASUREMENT_KEY_MEAS_BITS))
            | (int(device_pos) << _ACTIVE_MEASUREMENT_KEY_MEAS_BITS)
            | int(meas_type_code)
        )

    @staticmethod
    def _active_measurement_key_array(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> np.ndarray:
        type_values = np.asarray(device_type_code, dtype=np.int64)
        pos_values = np.asarray(device_pos, dtype=np.int64)
        meas_values = np.asarray(meas_type_code, dtype=np.int64)
        return (
            (type_values << (_ACTIVE_DEVICE_KEY_POS_BITS + _ACTIVE_MEASUREMENT_KEY_MEAS_BITS))
            | (pos_values << _ACTIVE_MEASUREMENT_KEY_MEAS_BITS)
            | meas_values
        )

    @staticmethod
    def _active_measurement_key_array_for_type(
        device_type_code: int,
        device_pos: np.ndarray,
        meas_type_code: int,
    ) -> np.ndarray:
        pos_values = np.asarray(device_pos, dtype=np.int64)
        return (
            (np.int64(int(device_type_code)) << (_ACTIVE_DEVICE_KEY_POS_BITS + _ACTIVE_MEASUREMENT_KEY_MEAS_BITS))
            | (pos_values << _ACTIVE_MEASUREMENT_KEY_MEAS_BITS)
            | np.int64(int(meas_type_code))
        )

    @staticmethod
    def _active_key_cache_from_arrays(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        active_mask: np.ndarray,
    ) -> set:
        keys = ACStateEstimator._active_key_array_from_arrays(
            device_type_code,
            device_pos,
            meas_type_code,
            active_mask,
        )
        return set(keys.astype(object, copy=False))

    @staticmethod
    def _active_key_array_from_arrays(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        active_values = np.asarray(active_mask, dtype=bool)
        pos_all = np.asarray(device_pos, dtype=np.int64)
        if active_values.size == pos_all.size and active_values.size and bool(np.all(active_values)):
            valid_pos = pos_all >= 0
            if bool(np.all(valid_pos)):
                return ACStateEstimator._active_measurement_key_array(
                    device_type_code,
                    pos_all,
                    meas_type_code,
                ).astype(np.int64, copy=False)
        rows = np.flatnonzero(active_values).astype(np.int64, copy=False)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        pos_values = pos_all[rows]
        valid_pos = pos_values >= 0
        if not np.any(valid_pos):
            return np.empty(0, dtype=np.int64)
        return ACStateEstimator._active_measurement_key_array(
            np.asarray(device_type_code, dtype=np.int16)[rows][valid_pos],
            pos_values[valid_pos],
            np.asarray(meas_type_code, dtype=np.int16)[rows][valid_pos],
        ).astype(np.int64, copy=False)

    def _voltage_best_from_arrays(
        self,
        table: MeasurementTable,
        real_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        _node_valid, _node_voltage_scale, _node_current_scale, node_idx_by_pos = self._node_scale_arrays_by_pos()
        row_index = np.arange(int(table.idx.size), dtype=np.int64)
        node_best_weight = np.full(int(node_idx_by_pos.size), -np.inf, dtype=np.float64)
        node_best_value = np.zeros(int(node_idx_by_pos.size), dtype=np.float64)
        real_best_weight = np.full(int(node_idx_by_pos.size), -np.inf, dtype=np.float64)
        real_best_value = np.zeros(int(node_idx_by_pos.size), dtype=np.float64)
        table_weight = np.asarray(table.weight, dtype=np.float64)
        table_value = np.asarray(table.value, dtype=np.float64)

        def update_best_arrays(rows: np.ndarray, pos_values: np.ndarray, best_weight: np.ndarray, best_value: np.ndarray) -> None:
            if rows.size == 0:
                return
            pos_values = np.asarray(pos_values, dtype=np.int64)
            in_range = (pos_values >= 0) & (pos_values < node_idx_by_pos.size)
            if not np.any(in_range):
                return
            rows_valid = rows[in_range]
            pos_valid = pos_values[in_range].astype(np.intp, copy=False)
            node_idx_values = node_idx_by_pos[pos_valid]
            valid_nodes = node_idx_values >= 0
            if not np.any(valid_nodes):
                return
            pos_valid = pos_valid[valid_nodes]
            rows_valid = rows_valid[valid_nodes]
            weights = table_weight[rows_valid]
            values = table_value[rows_valid]
            np.maximum.at(best_weight, pos_valid, weights)
            keep = weights >= best_weight[pos_valid]
            if np.any(keep):
                best_value[pos_valid[keep]] = values[keep]

        def best_dict(best_weight: np.ndarray, best_value: np.ndarray) -> Dict[int, float]:
            valid = np.flatnonzero(np.isfinite(best_weight) & (node_idx_by_pos >= 0)).astype(np.int64, copy=False)
            return {
                int(node_idx): float(value)
                for node_idx, value in zip(node_idx_by_pos[valid], best_value[valid])
            }

        node_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_ACNode)
            & (meas_type_code == MEAS_TYPE_V)
        ]
        update_best_arrays(node_v_rows, from_pos[node_v_rows], node_best_weight, node_best_value)
        update_best_arrays(node_v_rows, from_pos[node_v_rows], real_best_weight, real_best_value)

        gen_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_ACGenerator)
            & (meas_type_code == MEAS_TYPE_V_GEN)
        ]
        load_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_ACLoad)
            & (meas_type_code == MEAS_TYPE_V_LOAD)
        ]
        update_best_arrays(gen_v_rows, from_pos[gen_v_rows], real_best_weight, real_best_value)
        update_best_arrays(load_v_rows, from_pos[load_v_rows], real_best_weight, real_best_value)

        terminal_device_mask = (
            (device_type_code == DEVICE_TYPE_ACBranch)
            | (device_type_code == DEVICE_TYPE_ACTransformer)
            | (device_type_code == DEVICE_TYPE_ACZeroBranch)
            | (device_type_code == DEVICE_TYPE_ACBreak)
        )
        terminal_v_from_rows = row_index[real_mask & terminal_device_mask & (meas_type_code == MEAS_TYPE_V_FROM)]
        terminal_v_to_rows = row_index[real_mask & terminal_device_mask & (meas_type_code == MEAS_TYPE_V_TO)]
        update_best_arrays(terminal_v_from_rows, from_pos[terminal_v_from_rows], real_best_weight, real_best_value)
        update_best_arrays(terminal_v_to_rows, to_pos[terminal_v_to_rows], real_best_weight, real_best_value)
        node_valid_pos = node_idx_by_pos >= 0
        self._node_voltage_measurement_pos_mask_cache = np.isfinite(node_best_weight) & node_valid_pos
        self._real_voltage_observation_pos_mask_cache = np.isfinite(real_best_weight) & node_valid_pos
        self._real_voltage_observed_solver_pos_mask_cache = None
        return best_dict(node_best_weight, node_best_value), best_dict(real_best_weight, real_best_value)

    def _power_seed_best_from_arrays(
        self,
        table: MeasurementTable,
        processable_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        def empty_seed() -> Dict[str, np.ndarray]:
            return {
                "measurement_key": np.empty(0, dtype=np.int64),
                "weight": np.empty(0, dtype=np.float64),
                "value": np.empty(0, dtype=np.float64),
            }

        row_index = np.arange(int(table.idx.size), dtype=np.int64)
        gen_power_rows = row_index[
            processable_mask
            & (device_type_code == DEVICE_TYPE_ACGenerator)
            & ((meas_type_code == MEAS_TYPE_P_GEN) | (meas_type_code == MEAS_TYPE_Q_GEN))
            & (device_pos >= 0)
        ]
        load_power_rows = row_index[
            processable_mask
            & (device_type_code == DEVICE_TYPE_ACLoad)
            & ((meas_type_code == MEAS_TYPE_P_LOAD) | (meas_type_code == MEAS_TYPE_Q_LOAD))
            & (device_pos >= 0)
        ]
        power_rows = np.concatenate((gen_power_rows, load_power_rows))
        if power_rows.size == 0:
            return empty_seed()
        key_type = device_type_code[power_rows].astype(np.int64, copy=False)
        key_pos = device_pos[power_rows].astype(np.int64, copy=False)
        key_meas = meas_type_code[power_rows].astype(np.int64, copy=False)
        keys = self._active_measurement_key_array(key_type, key_pos, key_meas)
        weight_values = np.asarray(table.weight, dtype=np.float64)[power_rows]
        value_values = np.asarray(table.value, dtype=np.float64)[power_rows]
        order = np.lexsort((-weight_values, keys))
        sorted_keys = keys[order]
        keep = np.empty(sorted_keys.size, dtype=bool)
        keep[0] = True
        if sorted_keys.size > 1:
            keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        return {
            "measurement_key": sorted_keys[keep].astype(np.int64, copy=False),
            "weight": weight_values[order][keep].astype(np.float64, copy=False),
            "value": value_values[order][keep].astype(np.float64, copy=False),
        }

    def _power_flow_seed_rows_from_arrays(
        self,
        table: MeasurementTable,
        processable_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        def empty_seed() -> Dict[str, np.ndarray]:
            return {
                "measurement_key": np.empty(0, dtype=np.int64),
                "ppc_row": np.empty(0, dtype=np.int64),
                "value": np.empty(0, dtype=np.float64),
            }

        if getattr(self, "flat_start", True):
            return empty_seed()
        seed_rows_mask = (
            (device_type_code == DEVICE_TYPE_ACNode)
            & (meas_type_code == MEAS_TYPE_V)
        ) | (
            (device_type_code == DEVICE_TYPE_ACGenerator)
            & np.isin(
                meas_type_code,
                (
                    MEAS_TYPE_P_GEN,
                    MEAS_TYPE_Q_GEN,
                    MEAS_TYPE_V_GEN,
                    MEAS_TYPE_I_GEN,
                ),
            )
        ) | (
            (device_type_code == DEVICE_TYPE_ACLoad)
            & np.isin(
                meas_type_code,
                (
                    MEAS_TYPE_P_LOAD,
                    MEAS_TYPE_Q_LOAD,
                    MEAS_TYPE_V_LOAD,
                    MEAS_TYPE_I_LOAD,
                ),
            )
        )
        seed_rows = np.flatnonzero(processable_mask & seed_rows_mask & (device_pos >= 0)).astype(np.int64, copy=False)
        if seed_rows.size == 0:
            return empty_seed()
        ppc_rows = self._measurement_ppc_rows_for_device_positions(device_type_code[seed_rows], device_pos[seed_rows])
        valid = ppc_rows >= 0
        if not np.any(valid):
            return empty_seed()
        rows = seed_rows[valid]
        return {
            "measurement_key": self._active_measurement_key_array(
                device_type_code[rows],
                device_pos[rows],
                meas_type_code[rows],
            ).astype(np.int64, copy=False),
            "ppc_row": ppc_rows[valid].astype(np.int64, copy=False),
            "value": np.asarray(table.value, dtype=np.float64)[rows],
        }

    def _measurement_runtime_array_cache_key(self) -> Tuple[object, ...]:
        return (
            id(getattr(self, "_ac_node_plan_pos", None)),
            id(getattr(self, "_ac_branch_plan_i", None)),
            id(getattr(self, "_ac_branch_plan_j", None)),
            id(getattr(self, "_ac_transformer_plan_i", None)),
            id(getattr(self, "_ac_transformer_plan_j", None)),
            id(getattr(self, "_ac_zero_branch_plan_i", None)),
            id(getattr(self, "_ac_zero_branch_plan_j", None)),
            id(getattr(self, "_ac_break_plan_i", None)),
            id(getattr(self, "_ac_break_plan_j", None)),
            id(getattr(self, "_ac_generator_plan_node_pos", None)),
            id(getattr(self, "_ac_load_plan_node_pos", None)),
            len(getattr(self, "nodes", ())),
            float(getattr(self, "p_base", 0.0)),
            float(getattr(self, "u_scale", 0.0)),
            float(getattr(self, "i_scale", 0.0)),
        )

    @staticmethod
    def _cached_measurement_runtime_arrays(
        meas_ppc: Dict,
        n_rows: int,
        cache_key: Tuple[object, ...],
    ):
        if meas_ppc.get("_ac_se_runtime_cache_key") != cache_key:
            return None
        keys = ("device_pos", "available", "scale", "from_pos", "to_pos")
        arrays = []
        for key in keys:
            value = meas_ppc.get(key)
            if not isinstance(value, np.ndarray) or int(value.size) != int(n_rows):
                return None
            arrays.append(value)
        return arrays

    def _normalize_measurements_to_pu_from_meas_ppc(self, table: MeasurementTable, meas_ppc: Dict) -> bool:
        """Normalize file-backed measurement PPC rows using numeric codes and plan positions."""
        meas = meas_ppc.get("meas")
        has_meas = isinstance(meas, np.ndarray) and meas.ndim == 2 and meas.shape[0] == table.idx.size
        cols = meas_ppc.get("meas_cols", MEAS_COLS)
        value_array = table.value
        weight_array = table.weight
        valid_array = table.valid
        status_array = measurement_table_status_code(table)
        idx_array = table.idx
        n_rows = int(value_array.size)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is not None:
            device_pos = np.asarray(device_pos, dtype=np.int64)
            if int(device_pos.size) != n_rows:
                device_pos = None
        if device_pos is None:
            ppc_device_pos = meas_ppc.get("device_pos")
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == n_rows:
                device_pos = np.asarray(ppc_device_pos, dtype=np.int64)
        device_type_code_array = meas_ppc.get("device_type_code_array")
        meas_type_code_array = meas_ppc.get("meas_type_code_array")
        if not isinstance(device_type_code_array, np.ndarray) and has_meas:
            device_type_code_array = meas[:, cols["device_type_code"]].astype(np.int16, copy=False)
        if not isinstance(meas_type_code_array, np.ndarray) and has_meas:
            meas_type_code_array = meas[:, cols["meas_type_code"]].astype(np.int16, copy=False)
        if isinstance(device_type_code_array, np.ndarray):
            device_type_code_array = np.asarray(device_type_code_array, dtype=np.int16)
        if isinstance(meas_type_code_array, np.ndarray):
            meas_type_code_array = np.asarray(meas_type_code_array, dtype=np.int16)
        if (
            device_pos is None
            or not isinstance(device_type_code_array, np.ndarray)
            or not isinstance(meas_type_code_array, np.ndarray)
        ):
            return False
        if device_pos.size != n_rows or device_type_code_array.size != n_rows or meas_type_code_array.size != n_rows:
            return False

        table.device_pos = device_pos
        table.meas_type_code = meas_type_code_array
        table.device_type_code = device_type_code_array
        self._ensure_measurement_plan_lookup_arrays()
        runtime_cache_key = self._measurement_runtime_array_cache_key()
        cached_runtime = self._cached_measurement_runtime_arrays(meas_ppc, n_rows, runtime_cache_key)
        if cached_runtime is None:
            available, scale_array, from_pos, to_pos = self._measurement_scale_for_codes(
                device_type_code_array,
                device_pos,
                meas_type_code_array,
                rows_by_code=rows_by_device_type_code(table),
            )
            runtime_copy = not bool(meas_ppc.get("_mutable_runtime_arrays", False))
            meas_ppc["device_pos"] = device_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["available"] = available.astype(bool, copy=runtime_copy)
            meas_ppc["scale"] = scale_array.astype(np.float64, copy=runtime_copy)
            meas_ppc["from_pos"] = from_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["to_pos"] = to_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["_ac_se_runtime_cache_key"] = runtime_cache_key
        else:
            device_pos, available, scale_array, from_pos, to_pos = cached_runtime
            device_pos = np.asarray(device_pos, dtype=np.int64)
            available = np.asarray(available, dtype=bool)
            scale_array = np.asarray(scale_array, dtype=np.float64)
            from_pos = np.asarray(from_pos, dtype=np.int64)
            to_pos = np.asarray(to_pos, dtype=np.int64)
        table.device_pos = device_pos
        table.available = available
        table.scale = scale_array
        table.from_pos = from_pos
        table.to_pos = to_pos
        unavailable_mask = ~available

        max_idx = int(idx_array.max()) if n_rows else 0
        angle_mask = np.asarray(table.angle_mask, dtype=bool)
        invalidated_mask = np.zeros(n_rows, dtype=bool)
        if angle_mask.any():
            if self.flat_start:
                value_array[angle_mask] = 0.0
            valid_array[angle_mask] = False
            status_array[angle_mask] = MEAS_STATUS_INVALID
            invalidated_mask[angle_mask] = True

        candidate_mask = valid_array & (weight_array > 0.0) & (~angle_mask)
        flagged_unavailable = candidate_mask & unavailable_mask
        if flagged_unavailable.any():
            valid_array[flagged_unavailable] = False
            status_array[flagged_unavailable] = MEAS_STATUS_INVALID
            invalidated_mask[flagged_unavailable] = True
        processable_mask = candidate_mask & (~unavailable_mask)
        if processable_mask.any():
            value_array[processable_mask] = value_array[processable_mask] / scale_array[processable_mask]

        is_pseudo_array = status_array == MEAS_STATUS_PSEUDO
        real_mask = processable_mask & (~is_pseudo_array)
        active_measurement_key_array = self._active_key_array_from_arrays(
            device_type_code_array,
            device_pos,
            meas_type_code_array,
            processable_mask,
        )
        node_voltage_best, real_voltage_best = self._voltage_best_from_arrays(
            table,
            real_mask,
            device_type_code_array,
            meas_type_code_array,
            from_pos,
            to_pos,
        )
        power_seed_best = self._power_seed_best_from_arrays(
            table,
            processable_mask,
            device_type_code_array,
            meas_type_code_array,
            device_pos,
        )
        power_flow_seed_rows = self._power_flow_seed_rows_from_arrays(
            table,
            processable_mask,
            device_type_code_array,
            meas_type_code_array,
            device_pos,
        )

        self.measurement_table = table
        self._set_active_key_caches_from_array(active_measurement_key_array)
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = node_voltage_best
        self._real_voltage_observation_node_cache = real_voltage_best
        self._real_power_measurement_seed_cache = power_seed_best
        self._power_flow_seed_rows = power_flow_seed_rows
        self._has_valid_angle_measurements = False
        try:
            self.measurements.normalized = True
        except AttributeError:
            pass
        meas_ppc["value_array"] = value_array
        meas_ppc["valid_array"] = valid_array
        meas_ppc["status_array"] = status_array
        if has_meas:
            meas[:, cols["value"]] = value_array.astype(np.float64, copy=False)
            meas[:, cols["valid"]] = valid_array.astype(np.float64, copy=False)
            meas[:, cols["status"]] = status_array.astype(np.float64, copy=False)
            if invalidated_mask.any():
                invalid_rows = np.flatnonzero(invalidated_mask)
                meas[invalid_rows, cols["valid"]] = 0.0
                meas[invalid_rows, cols["status"]] = float(MEAS_STATUS_INVALID)
        meas_ppc["normalized"] = True
        object_count = list.__len__(self.measurements) if isinstance(self.measurements, list) else 0
        if object_count == n_rows and object_count > 0:
            for pos, meas_obj in enumerate(list.__iter__(self.measurements)):
                meas_obj.valid = bool(valid_array[pos])
                meas_obj.value = float(value_array[pos])
                meas_obj.status = int(status_array[pos])
        return True

    def _normalize_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal state-estimation units."""
        table = _measurement_table_from_measurements(self.measurements)
        if getattr(self.measurements, "normalized", False):
            self.measurement_table = table
            self._refresh_measurement_summary_cache()
            self._has_valid_angle_measurements = bool(np.any(table.valid & table.angle_mask))
            self.meas_ppc["normalized"] = True
            return
        if self._normalize_measurements_to_pu_from_meas_ppc(table, self.meas_ppc):
            return
        self.measurement_table = table
        self._set_active_key_caches(set())
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self._node_voltage_measurement_cache = {}
        self._real_voltage_observation_node_cache = {}
        self._node_voltage_measurement_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
        self._real_voltage_observation_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
        self._real_voltage_observed_solver_pos_mask_cache = None
        self._real_power_measurement_seed_cache = {
            "measurement_key": np.empty(0, dtype=np.int64),
            "weight": np.empty(0, dtype=np.float64),
            "value": np.empty(0, dtype=np.float64),
        }
        self._power_flow_seed_rows = {
            "measurement_key": np.empty(0, dtype=np.int64),
            "ppc_row": np.empty(0, dtype=np.int64),
            "value": np.empty(0, dtype=np.float64),
        }
        self._has_valid_angle_measurements = bool(np.any(table.valid & table.angle_mask))
        warnings.warn(
            "AC SE measurement normalization requires measurement PPC arrays; name-based fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _convert_measurements_to_pu(self) -> None:
        self._normalize_measurements_to_pu()

    def _active_measurement_keys_ref(self) -> set:
        """Return active measurements as packed integer (device_type_code, device_pos, meas_type_code) keys."""
        if hasattr(self, "_active_measurement_code_pos_cache"):
            self._active_measurement_keys = self._active_measurement_code_pos_cache
            return self._active_measurement_code_pos_cache
        keys = getattr(self, "_active_measurement_keys", None)
        if keys is not None:
            return keys
        key_array = getattr(self, "_active_measurement_key_array_cache", None)
        if key_array is None:
            self._refresh_measurement_summary_cache()
            key_array = getattr(self, "_active_measurement_key_array_cache", None)
        key_array = np.asarray(key_array, dtype=np.int64).reshape(-1)
        keys = set(key_array.astype(object, copy=False))
        self._active_measurement_keys = keys
        self._active_measurement_code_pos_cache = keys
        return keys

    def _active_measurement_key_array_ref(self) -> np.ndarray:
        """Return active measurement keys as an int64 array for vectorized membership checks."""
        code_cache = getattr(self, "_active_measurement_code_pos_cache", None)
        if code_cache is not None and code_cache is not getattr(self, "_active_measurement_keys", None):
            key_array = self._measurement_key_array_from_cache(code_cache)
            self._active_measurement_key_array_cache = key_array
            return key_array
        key_array = getattr(self, "_active_measurement_key_array_cache", None)
        if key_array is not None:
            return np.asarray(key_array, dtype=np.int64)
        keys = self._active_measurement_keys_ref()
        key_array = self._measurement_key_array_from_cache(keys)
        self._active_measurement_key_array_cache = key_array
        return key_array

    def _active_measurement_key_sorted_array_ref(self) -> np.ndarray:
        """Return active measurement keys sorted once for repeated vector membership probes."""
        sorted_cache = getattr(self, "_active_measurement_key_sorted_array_cache", None)
        key_array = self._active_measurement_key_array_ref()
        cache_key = (id(key_array), int(np.asarray(key_array).size))
        if isinstance(sorted_cache, tuple) and len(sorted_cache) == 2 and sorted_cache[0] == cache_key:
            return sorted_cache[1]
        keys = np.asarray(key_array, dtype=np.int64).reshape(-1)
        sorted_keys = np.sort(keys) if keys.size else np.empty(0, dtype=np.int64)
        self._active_measurement_key_sorted_array_cache = (cache_key, sorted_keys)
        return sorted_keys

    def _active_measurement_key_membership(self, keys: np.ndarray) -> np.ndarray:
        """Vectorized membership against packed active measurement keys without np.isin uniquing."""
        probe = np.asarray(keys, dtype=np.int64).reshape(-1)
        if probe.size == 0:
            return np.zeros(0, dtype=bool)
        sorted_keys = self._active_measurement_key_sorted_array_ref()
        if sorted_keys.size == 0:
            return np.zeros(probe.size, dtype=bool)
        pos = np.searchsorted(sorted_keys, probe)
        in_range = pos < sorted_keys.size
        out = np.zeros(probe.size, dtype=bool)
        if np.any(in_range):
            idx = np.flatnonzero(in_range)
            out[idx] = sorted_keys[pos[idx].astype(np.intp, copy=False)] == probe[idx]
        return out

    def _next_measurement_idx(self) -> int:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        return int(self._max_measurement_idx) + 1

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        active_measurement_key_array = np.empty(0, dtype=np.int64)
        node_voltage_best: Dict[int, float] = {}
        real_voltage_best: Dict[int, float] = {}
        power_seed_best: Dict[str, np.ndarray] = {
            "measurement_key": np.empty(0, dtype=np.int64),
            "weight": np.empty(0, dtype=np.float64),
            "value": np.empty(0, dtype=np.float64),
        }
        power_flow_seed_rows: Dict[str, np.ndarray] = {
            "measurement_key": np.empty(0, dtype=np.int64),
            "ppc_row": np.empty(0, dtype=np.int64),
            "value": np.empty(0, dtype=np.float64),
        }
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            max_idx = int(table.idx.max()) if table.idx.size else 0
            active = table.valid & (table.weight > 0.0)
            status_code = measurement_table_status_code(table)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_type_code = getattr(table, "meas_type_code", None)
            if meas_type_code is not None:
                meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
                if meas_type_code.size != table.idx.size:
                    meas_type_code = None
            if meas_type_code is None:
                warnings.warn(
                    "AC SE measurement summary requires meas_type_code; string fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                meas_type_code = np.zeros(int(table.idx.size), dtype=np.int16)
                table.meas_type_code = meas_type_code
            device_pos = getattr(table, "device_pos", None)
            if device_pos is None or np.asarray(device_pos).size != int(table.idx.size):
                device_pos = self._measurement_device_pos_array(table)
                table.device_pos = device_pos
            else:
                device_pos = np.asarray(device_pos, dtype=np.int64)
            active_measurement_key_array = self._active_key_array_from_arrays(
                device_type_code,
                device_pos,
                meas_type_code,
                active,
            )
            real_mask = active & (status_code != MEAS_STATUS_PSEUDO)
            from_pos, to_pos = self._measurement_voltage_node_positions_from_codes(
                device_type_code,
                device_pos,
                meas_type_code,
            )
            node_voltage_best, real_voltage_best = self._voltage_best_from_arrays(
                table,
                real_mask,
                device_type_code,
                meas_type_code,
                from_pos,
                to_pos,
            )
            power_seed_best = self._power_seed_best_from_arrays(
                table,
                active,
                device_type_code,
                meas_type_code,
                device_pos,
            )
            power_flow_seed_rows = self._power_flow_seed_rows_from_arrays(
                table,
                active,
                device_type_code,
                meas_type_code,
                device_pos,
            )
        else:
            max_idx = 0
            if self.measurements:
                warnings.warn(
                    "AC SE measurement summary requires a PPC-backed measurement table; string fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._set_active_key_caches_from_array(
            active_measurement_key_array,
        )
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = node_voltage_best
        self._real_voltage_observation_node_cache = real_voltage_best
        self._real_power_measurement_seed_cache = power_seed_best
        self._power_flow_seed_rows = power_flow_seed_rows
        if table is None or len(table.idx) != len(self.measurements):
            self._node_voltage_measurement_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
            self._real_voltage_observation_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
            self._real_voltage_observed_solver_pos_mask_cache = None

    @staticmethod
    def _pseudo_measurement_table(
        names: Sequence[str],
        device_types: Sequence[str],
        device_names: Sequence[str],
        meas_types: Sequence[str],
        values: Sequence[float],
        weight: float,
        *,
        idx_start: int = 0,
        device_type_codes: Optional[Sequence[int]] = None,
        meas_type_codes: Optional[Sequence[int]] = None,
        device_positions: Optional[Sequence[int]] = None,
    ) -> MeasurementTable:
        row_count = max(
            len(values),
            len(names),
            0 if device_type_codes is None else len(device_type_codes),
            0 if meas_type_codes is None else len(meas_type_codes),
            0 if device_positions is None else len(device_positions),
        )
        if row_count == 0:
            empty_i64 = np.asarray([], dtype=np.int64)
            empty_obj = np.asarray([], dtype=object)
            empty_f = np.asarray([], dtype=np.float64)
            empty_b = np.asarray([], dtype=bool)
            empty_i16 = np.asarray([], dtype=np.int16)
            table = MeasurementTable(
                idx=empty_i64,
                name=empty_obj,
                device_type=empty_obj,
                device_name=empty_obj,
                meas_type=empty_obj,
                weight=empty_f,
                valid=empty_b,
                value=empty_f,
                device_type_code=empty_i16,
                angle_mask=empty_b,
                status_code=empty_i16,
                rows_by_device_type_code={},
                meas_type_code=empty_i16,
                device_pos=empty_i64,
            )
            return table

        def object_array_or_empty(values_array, field_name: str) -> np.ndarray:
            array = np.asarray(values_array, dtype=object)
            if array.size == row_count:
                return array
            if array.size == 0:
                return np.asarray([], dtype=object)
            raise ValueError(f"pseudo measurement {field_name} array size does not match row count")

        name_array = object_array_or_empty(names, "name")
        device_type_array = object_array_or_empty(device_types, "device_type")
        device_name_array = object_array_or_empty(device_names, "device_name")
        meas_type_array = object_array_or_empty(meas_types, "meas_type")
        if device_type_codes is None:
            raise ValueError("pseudo measurement device_type_code array is required; string fallback is disabled")
        else:
            device_type_code = np.asarray(device_type_codes, dtype=np.int16)
            if device_type_code.size != row_count:
                raise ValueError("pseudo measurement device_type_code array size does not match row count")
        if meas_type_codes is None:
            raise ValueError("pseudo measurement meas_type_code array is required; string fallback is disabled")
        else:
            meas_type_code = np.asarray(meas_type_codes, dtype=np.int16)
            if meas_type_code.size != row_count:
                raise ValueError("pseudo measurement meas_type_code array size does not match row count")
        if device_positions is None:
            device_pos = None
        else:
            device_pos = np.asarray(device_positions, dtype=np.int64)
            if device_pos.size != row_count:
                raise ValueError("pseudo measurement device_pos array size does not match row count")
        weight_array = (
            np.full(row_count, float(weight), dtype=np.float64)
            if np.isscalar(weight)
            else np.asarray(weight, dtype=np.float64)
        )
        if weight_array.size != row_count:
            raise ValueError("pseudo measurement weight array size does not match row count")
        table = MeasurementTable(
            idx=np.arange(int(idx_start), int(idx_start) + row_count, dtype=np.int64),
            name=name_array,
            device_type=device_type_array,
            device_name=device_name_array,
            meas_type=meas_type_array,
            weight=weight_array,
            valid=np.ones(row_count, dtype=bool),
            value=np.asarray(values, dtype=np.float64),
            device_type_code=device_type_code,
            angle_mask=np.zeros(row_count, dtype=bool),
            status_code=np.full(row_count, MEAS_STATUS_PSEUDO, dtype=np.int16),
            rows_by_device_type_code={
                int(code): np.flatnonzero(device_type_code == code).astype(np.int64, copy=False)
                for code in np.unique(device_type_code)
            },
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )
        return table

    def _append_pseudo_measurement_rows(
        self,
        next_idx: int,
        names: Sequence[str],
        device_types: Sequence[str],
        device_names: Sequence[str],
        meas_types: Sequence[str],
        values: Sequence[float],
        *,
        weights=None,
        record_summary: bool = True,
        device_type_codes: Optional[Sequence[int]] = None,
        meas_type_codes: Optional[Sequence[int]] = None,
        device_positions: Optional[Sequence[int]] = None,
    ) -> int:
        row_count = max(
            len(values),
            len(names),
            0 if device_type_codes is None else len(device_type_codes),
            0 if meas_type_codes is None else len(meas_type_codes),
            0 if device_positions is None else len(device_positions),
        )
        if row_count == 0:
            return next_idx
        weight_values = self.pseudo_measurement_weight if weights is None else weights
        appended_table = self._pseudo_measurement_table(
            names,
            device_types,
            device_names,
            meas_types,
            values,
            weight_values,
            idx_start=next_idx,
            device_type_codes=device_type_codes,
            meas_type_codes=meas_type_codes,
            device_positions=device_positions,
        )
        self._append_pseudo_measurement_table(appended_table, record_summary=record_summary)
        return int(next_idx) + row_count

    def _append_pseudo_measurement_table(
        self,
        appended_table: MeasurementTable,
        *,
        record_summary: bool = True,
    ) -> MeasurementTable:
        row_count = int(appended_table.idx.size)
        if row_count == 0:
            return _measurement_table_from_measurements(self.measurements)
        base_table = _measurement_table_from_measurements(self.measurements)
        base_count = int(base_table.idx.size)
        total_count = base_count + row_count
        base_device_pos = getattr(base_table, "device_pos", None)
        if base_device_pos is not None:
            base_device_pos = np.asarray(base_device_pos, dtype=np.int64)
        appended_device_pos = getattr(appended_table, "device_pos", None)
        if appended_device_pos is not None:
            appended_device_pos = np.asarray(appended_device_pos, dtype=np.int64)
        combined_table = concat_measurement_tables(base_table, appended_table)
        rows_by_code = {
            int(code): np.asarray(rows, dtype=np.int64).copy()
            for code, rows in rows_by_device_type_code(base_table).items()
        }
        for code, rows in rows_by_device_type_code(appended_table).items():
            tail_rows = np.asarray(rows, dtype=np.int64) + base_count
            if int(code) in rows_by_code:
                rows_by_code[int(code)] = np.concatenate((rows_by_code[int(code)], tail_rows))
            else:
                rows_by_code[int(code)] = tail_rows
        combined_table.rows_by_device_type_code = rows_by_code
        combined_device_pos = getattr(combined_table, "device_pos", None)
        if combined_device_pos is None or np.asarray(combined_device_pos).size != total_count:
            combined_device_pos = np.empty(total_count, dtype=np.int64)
            combined_device_pos.fill(-1)
            if base_device_pos is not None and base_device_pos.size == base_count:
                combined_device_pos[:base_count] = base_device_pos
            if appended_device_pos is not None and appended_device_pos.size == row_count:
                combined_device_pos[base_count:total_count] = appended_device_pos
        else:
            combined_device_pos = np.asarray(combined_device_pos, dtype=np.int64)
        combined_table.device_pos = combined_device_pos
        invalid_device_pos_count = 0
        if base_count and (base_device_pos is None or base_device_pos.size != base_count):
            invalid_device_pos_count += base_count
        if appended_device_pos is None or appended_device_pos.size != row_count:
            invalid_device_pos_count += row_count
        else:
            invalid_device_pos_count += int(np.count_nonzero(appended_device_pos < 0))
        if invalid_device_pos_count:
            warnings.warn(
                (
                    "AC SE pseudo measurement append found "
                    f"{invalid_device_pos_count} rows without valid device_pos; "
                    "name-based device lookup is disabled."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        self.measurements = self._measurement_sequence_from_table(
            combined_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurement_table = combined_table
        if row_count:
            self._max_measurement_idx = int(np.max(appended_table.idx))
        if record_summary:
            tail_type = np.asarray(appended_table.device_type_code, dtype=np.int16)
            tail_meas = np.asarray(appended_table.meas_type_code, dtype=np.int16)
            tail_pos = (
                appended_device_pos
                if appended_device_pos is not None and appended_device_pos.size == row_count
                else np.full(row_count, -1, dtype=np.int64)
            )
            valid_tail = tail_pos >= 0
            if np.any(valid_tail):
                valid_type = tail_type[valid_tail].astype(np.int64, copy=False)
                valid_pos = tail_pos[valid_tail].astype(np.int64, copy=False)
                valid_meas = tail_meas[valid_tail].astype(np.int64, copy=False)
                measurement_keys = self._active_measurement_key_array(valid_type, valid_pos, valid_meas)
                self._append_active_measurement_key_array_cache(measurement_keys)
        return base_table

    def _real_voltage_observation_nodes(self) -> Dict[int, float]:
        """Return nodes covered by real usable voltage measurements on any AC device."""
        cache = getattr(self, "_real_voltage_observation_node_cache", None)
        if cache is not None:
            return cache
        self._refresh_measurement_summary_cache()
        return getattr(self, "_real_voltage_observation_node_cache", {})

    def _real_voltage_observation_value_for_node(self, node_idx: Optional[int]) -> Optional[float]:
        """Return a real voltage value on the node or its compressed zero-tie component."""
        if node_idx is None:
            return None
        observed = self._real_voltage_observation_nodes()
        if node_idx in observed:
            return float(observed[node_idx])
        pos = self._solver_pos_for_node_idx(int(node_idx))
        component_by_pos = getattr(self, "zero_tie_component_by_pos", None)
        components = getattr(self, "zero_tie_components", None)
        if pos < 0 or component_by_pos is None or not components:
            return None
        component_idx = int(component_by_pos[int(pos)])
        if isinstance(components, dict):
            component = components.get(component_idx)
            if component is None:
                return None
        else:
            component = components[component_idx]
        for member_pos in component:
            member_idx = int(self._ac_node_ids[int(member_pos)])
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _real_voltage_observed_solver_pos_mask(self) -> np.ndarray:
        """Return solver-node positions covered by real voltage measurements, including zero-tie peers."""
        observed_pos_mask = getattr(self, "_real_voltage_observation_pos_mask_cache", None)
        cache_key = (
            id(observed_pos_mask),
            int(getattr(self, "n_nodes", 0)),
            id(getattr(self, "zero_tie_component_by_pos", None)),
            id(getattr(self, "zero_tie_component_offsets", None)),
        )
        cache = getattr(self, "_real_voltage_observed_solver_pos_mask_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == cache_key:
            return cache[1]
        n_nodes = int(getattr(self, "n_nodes", 0))
        if isinstance(observed_pos_mask, np.ndarray) and int(observed_pos_mask.size) == n_nodes:
            covered_voltage_pos = np.asarray(observed_pos_mask, dtype=bool).copy()
        else:
            observed_voltage = self._real_voltage_observation_nodes()
            covered_voltage_pos = np.zeros(n_nodes, dtype=bool)
            if observed_voltage:
                observed_node_idx = np.fromiter(
                    (int(node_idx) for node_idx in observed_voltage.keys()),
                    dtype=np.int64,
                    count=len(observed_voltage),
                )
                lookup_ids = np.asarray(getattr(self, "_ac_node_id_lookup_ids", ()), dtype=np.int64)
                lookup_pos = np.asarray(getattr(self, "_ac_node_id_lookup_pos", ()), dtype=np.int64)
                if lookup_ids.size and lookup_pos.size:
                    pos = np.searchsorted(lookup_ids, observed_node_idx)
                    in_range = pos < lookup_ids.size
                    if np.any(in_range):
                        idx = np.flatnonzero(in_range)
                        matched = lookup_ids[pos[idx].astype(np.intp, copy=False)] == observed_node_idx[idx]
                        if np.any(matched):
                            solver_pos = lookup_pos[pos[idx[matched]].astype(np.intp, copy=False)]
                            valid_solver = (solver_pos >= 0) & (solver_pos < covered_voltage_pos.size)
                            if np.any(valid_solver):
                                covered_voltage_pos[solver_pos[valid_solver].astype(np.intp, copy=False)] = True
        if not np.any(covered_voltage_pos):
            self._real_voltage_observed_solver_pos_mask_cache = (cache_key, covered_voltage_pos)
            return covered_voltage_pos
        component_by_pos = getattr(self, "zero_tie_component_by_pos", None)
        component_offsets = getattr(self, "zero_tie_component_offsets", None)
        if component_by_pos is not None and component_offsets is not None and covered_voltage_pos.size:
            component_by_pos = np.asarray(component_by_pos, dtype=np.int64)
            component_count = max(0, int(np.asarray(component_offsets, dtype=np.int64).size) - 1)
            valid_component = (
                component_count > 0
                and component_by_pos.size == covered_voltage_pos.size
                and np.all((component_by_pos >= 0) & (component_by_pos < component_count))
            )
            if valid_component:
                component_observed = np.zeros(component_count, dtype=bool)
                observed_pos = np.flatnonzero(covered_voltage_pos).astype(np.int64, copy=False)
                if observed_pos.size:
                    component_observed[component_by_pos[observed_pos.astype(np.intp, copy=False)].astype(np.intp, copy=False)] = True
                    covered_voltage_pos = component_observed[component_by_pos.astype(np.intp, copy=False)]
        self._real_voltage_observed_solver_pos_mask_cache = (cache_key, covered_voltage_pos)
        return covered_voltage_pos

    def _voltage_measurement_node_pos_from_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> int:
        pos = int(device_pos)
        if pos < 0:
            return -1
        code = int(device_type_code)
        meas_code = int(meas_type_code)
        if code == DEVICE_TYPE_ACNode:
            values = getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64))
            valid_meas = meas_code == MEAS_TYPE_V
        elif code == DEVICE_TYPE_ACGenerator:
            values = getattr(self, "_ac_generator_plan_node_pos", np.asarray([], dtype=np.int64))
            valid_meas = meas_code == MEAS_TYPE_V_GEN
        elif code == DEVICE_TYPE_ACLoad:
            values = getattr(self, "_ac_load_plan_node_pos", np.asarray([], dtype=np.int64))
            valid_meas = meas_code == MEAS_TYPE_V_LOAD
        elif code == DEVICE_TYPE_ACBranch:
            values = (
                getattr(self, "_ac_branch_plan_i", np.asarray([], dtype=np.int64))
                if meas_code == MEAS_TYPE_V_FROM
                else getattr(self, "_ac_branch_plan_j", np.asarray([], dtype=np.int64))
            )
            valid_meas = meas_code in (MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO)
        elif code == DEVICE_TYPE_ACTransformer:
            values = (
                getattr(self, "_ac_transformer_plan_i", np.asarray([], dtype=np.int64))
                if meas_code == MEAS_TYPE_V_FROM
                else getattr(self, "_ac_transformer_plan_j", np.asarray([], dtype=np.int64))
            )
            valid_meas = meas_code in (MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO)
        elif code == DEVICE_TYPE_ACZeroBranch:
            values = (
                getattr(self, "_ac_zero_branch_plan_i", np.asarray([], dtype=np.int64))
                if meas_code == MEAS_TYPE_V_FROM
                else getattr(self, "_ac_zero_branch_plan_j", np.asarray([], dtype=np.int64))
            )
            valid_meas = meas_code in (MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO)
        elif code == DEVICE_TYPE_ACBreak:
            values = (
                getattr(self, "_ac_break_plan_i", np.asarray([], dtype=np.int64))
                if meas_code == MEAS_TYPE_V_FROM
                else getattr(self, "_ac_break_plan_j", np.asarray([], dtype=np.int64))
            )
            valid_meas = meas_code in (MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO)
        else:
            return -1
        if not valid_meas:
            return -1
        values = np.asarray(values, dtype=np.int64)
        if pos >= values.size:
            return -1
        return int(values[pos])

    def _voltage_pseudo_is_covered_by_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> bool:
        node_pos = self._voltage_measurement_node_pos_from_code(device_type_code, device_pos, meas_type_code)
        if node_pos < 0:
            return False
        covered = self._real_voltage_observed_solver_pos_mask()
        return bool(node_pos < covered.size and covered[int(node_pos)])

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak P/Q/V priors for unmeasured AC topology-device states."""
        added_keys = set()
        topology_weight = float(self.pseudo_measurement_weight) * 1e-4
        ppc = self._ac_ppc_dict()
        store_strings = False
        key_probe = np.empty(1, dtype=np.int64)
        pseudo_rows = _PseudoMeasurementBuffer(
            3
            * (
                int(np.asarray(ppc.get("zero_branch", np.zeros((0, len(ZERO_BRANCH_COLS)))), dtype=np.float64).shape[0])
                + int(np.asarray(ppc.get("break", np.zeros((0, len(BREAK_COLS)))), dtype=np.float64).shape[0])
            ),
            with_weights=True,
            store_strings=store_strings,
        )

        def key_is_active(key: int) -> bool:
            key_probe[0] = int(key)
            return bool(self._active_measurement_key_membership(key_probe)[0])

        def queue_topology_pseudo(
            device_type_code: int,
            device_type: str,
            device_name: str,
            device_pos: int,
            meas_type: str,
            meas_type_code: int,
            value: float,
        ) -> None:
            key = self._active_measurement_key(device_type_code, int(device_pos), meas_type_code)
            if key in added_keys or key_is_active(key):
                return
            pseudo_name = f"pseudo_{meas_type.lower()}_{device_name}" if store_strings else ""
            pseudo_rows.add(
                pseudo_name,
                device_type,
                int(device_type_code),
                device_name,
                int(device_pos),
                meas_type,
                int(meas_type_code),
                float(value),
                weight=topology_weight,
            )
            added_keys.add(key)

        for device_type_code, device_type, table_name, name_key, device_key, cols in (
            (DEVICE_TYPE_ACZeroBranch, "ACZeroBranch", "zero_branch", "zero_branch_name", "zero_branch", ZERO_BRANCH_COLS),
            (DEVICE_TYPE_ACBreak, "ACBreak", "break", "break_name", "break", BREAK_COLS),
        ):
            table, device_topology, rows = self._ppc_sorted_alive_rows_for(ppc, table_name, device_key, cols["idx"])
            if rows.size == 0 or device_topology is None:
                continue
            names = (
                self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
                if store_strings
                else repeat("", int(rows.size))
            )
            plan_pos = self._plan_pos_for_ppc_rows(int(device_type_code), rows, table.shape[0])
            table_rows = table[rows.astype(np.intp, copy=False)]
            for row, name, pos, device_values in zip(rows, names, plan_pos, table_rows):
                if int(pos) < 0:
                    continue
                device_name = str(name) if store_strings else ""
                has_terminal_current = any(
                    key_is_active(self._active_measurement_key(device_type_code, int(pos), meas_type_code))
                    for meas_type_code in (MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO)
                )
                if not has_terminal_current and not any(
                    key_is_active(self._active_measurement_key(device_type_code, int(pos), meas_type_code))
                    for meas_type_code in (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO)
                ):
                    queue_topology_pseudo(
                        device_type_code,
                        device_type,
                        device_name,
                        int(pos),
                        "P_FROM",
                        MEAS_TYPE_P_FROM,
                        float(device_values[cols["p"]] or 0.0),
                    )
                if not has_terminal_current and not any(
                    key_is_active(self._active_measurement_key(device_type_code, int(pos), meas_type_code))
                    for meas_type_code in (MEAS_TYPE_Q_FROM, MEAS_TYPE_Q_TO)
                ):
                    queue_topology_pseudo(
                        device_type_code,
                        device_type,
                        device_name,
                        int(pos),
                        "Q_FROM",
                        MEAS_TYPE_Q_FROM,
                        float(device_values[cols["q"]] or 0.0),
                    )
                i_node = int(device_values[cols["i_node"]])
                if self._real_voltage_observation_value_for_node(i_node) is None:
                    queue_topology_pseudo(
                        device_type_code,
                        device_type,
                        device_name,
                        int(pos),
                        "V_FROM",
                        MEAS_TYPE_V_FROM,
                        self._ppc_terminal_voltage_pseudo_seed(ppc, device_topology, int(row)),
                    )
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_rows.names,
            pseudo_rows.device_types,
            pseudo_rows.device_names,
            pseudo_rows.meas_types,
            pseudo_rows.values,
            weights=pseudo_rows.weights,
            record_summary=False,
            device_type_codes=pseudo_rows.device_type_codes,
            meas_type_codes=pseudo_rows.meas_type_codes,
            device_positions=pseudo_rows.device_positions,
        )
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        next_idx = self._next_measurement_idx()
        next_idx, _topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        ppc = self._ac_ppc_dict()
        store_strings = False
        pseudo_rows = _PseudoMeasurementBuffer(
            3
            * (
                int(np.asarray(ppc.get("gen", np.zeros((0, len(GEN_COLS)))), dtype=np.float64).shape[0])
                + int(np.asarray(ppc.get("load", np.zeros((0, len(LOAD_COLS)))), dtype=np.float64).shape[0])
            ),
            store_strings=store_strings,
        )

        covered_voltage_pos = self._real_voltage_observed_solver_pos_mask()

        def append_missing(
            device_type_code: int,
            device_pos: np.ndarray,
            meas_type_code: int,
            values: np.ndarray,
            extra_mask: Optional[np.ndarray] = None,
        ) -> None:
            pos = np.asarray(device_pos, dtype=np.int64)
            if pos.size == 0:
                return
            missing = pos >= 0
            if not np.any(missing):
                return
            keys = self._active_measurement_key_array_for_type(int(device_type_code), pos, int(meas_type_code))
            missing &= ~self._active_measurement_key_membership(keys)
            if extra_mask is not None:
                missing &= np.asarray(extra_mask, dtype=bool)
            if not np.any(missing):
                return
            rows = np.flatnonzero(missing).astype(np.intp, copy=False)
            pseudo_rows.add_many(
                int(device_type_code),
                pos[rows],
                int(meas_type_code),
                np.asarray(values, dtype=np.float64)[rows],
            )

        def voltage_missing_mask(node_pos: np.ndarray) -> np.ndarray:
            node_pos = np.asarray(node_pos, dtype=np.int64)
            out = np.zeros(node_pos.size, dtype=bool)
            valid = (node_pos >= 0) & (node_pos < covered_voltage_pos.size)
            if np.any(valid):
                out[valid] = ~covered_voltage_pos[node_pos[valid].astype(np.intp, copy=False)]
            return out

        gen, gen_topology, gen_rows = self._ppc_sorted_alive_rows_for(ppc, "gen", "gen", GEN_COLS["idx"])
        gen_voltage = self._ppc_device_node_voltage_values(ppc, gen_topology, gen_rows)
        gen_p, gen_q = self._ppc_generator_pseudo_power_arrays(ppc, gen_rows)
        gen_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_ACGenerator, gen_rows, gen.shape[0])
        gen_node_pos = (
            np.asarray(gen_topology.node_pos, dtype=np.int32)[gen_rows.astype(np.intp, copy=False)]
            if gen_topology is not None and gen_rows.size
            else np.asarray([], dtype=np.int32)
        )
        append_missing(DEVICE_TYPE_ACGenerator, gen_plan_pos, MEAS_TYPE_P_GEN, gen_p)
        append_missing(DEVICE_TYPE_ACGenerator, gen_plan_pos, MEAS_TYPE_Q_GEN, gen_q)
        append_missing(
            DEVICE_TYPE_ACGenerator,
            gen_plan_pos,
            MEAS_TYPE_V_GEN,
            np.maximum(gen_voltage, self.voltage_floor),
            voltage_missing_mask(gen_node_pos),
        )

        load, load_topology, load_rows = self._ppc_sorted_alive_rows_for(ppc, "load", "load", LOAD_COLS["idx"])
        load_voltage = self._ppc_device_node_voltage_values(ppc, load_topology, load_rows)
        load_p, load_q = self._ppc_load_pseudo_power_arrays(ppc, load_rows, load_voltage)
        load_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_ACLoad, load_rows, load.shape[0])
        load_node_pos = (
            np.asarray(load_topology.node_pos, dtype=np.int32)[load_rows.astype(np.intp, copy=False)]
            if load_topology is not None and load_rows.size
            else np.asarray([], dtype=np.int32)
        )
        append_missing(DEVICE_TYPE_ACLoad, load_plan_pos, MEAS_TYPE_P_LOAD, load_p)
        append_missing(DEVICE_TYPE_ACLoad, load_plan_pos, MEAS_TYPE_Q_LOAD, load_q)
        append_missing(
            DEVICE_TYPE_ACLoad,
            load_plan_pos,
            MEAS_TYPE_V_LOAD,
            np.maximum(load_voltage, self.voltage_floor),
            voltage_missing_mask(load_node_pos),
        )
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_rows.names,
            pseudo_rows.device_types,
            pseudo_rows.device_names,
            pseudo_rows.meas_types,
            pseudo_rows.values,
            record_summary=False,
            device_type_codes=pseudo_rows.device_type_codes,
            meas_type_codes=pseudo_rows.meas_type_codes,
            device_positions=pseudo_rows.device_positions,
        )

    def _seed_power_state_arrays_from_measurements(self) -> None:
        """Use the best available P/Q rows as initial values for explicit power states."""
        seed = getattr(self, "_real_power_measurement_seed_cache", None)
        if seed is None:
            self._refresh_measurement_summary_cache()
            seed = getattr(self, "_real_power_measurement_seed_cache", {})
        if not isinstance(seed, dict):
            warnings.warn(
                "AC SE power state seeding requires packed key arrays; skipped power seed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        keys = np.asarray(seed.get("measurement_key", ()), dtype=np.int64)
        values = np.asarray(seed.get("value", ()), dtype=np.float64)
        if keys.size == 0:
            return
        if values.size != keys.size:
            warnings.warn(
                "AC SE power state seeding requires same-sized measurement_key and value arrays.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        gen_index = np.asarray(getattr(self, "_ac_generator_plan_index", ()), dtype=np.int64)
        load_index = np.asarray(getattr(self, "_ac_load_plan_index", ()), dtype=np.int64)
        meas_mask = (1 << _ACTIVE_MEASUREMENT_KEY_MEAS_BITS) - 1
        pos_mask = (1 << _ACTIVE_DEVICE_KEY_POS_BITS) - 1
        meas_type_code = keys & meas_mask
        device_pos = (keys >> _ACTIVE_MEASUREMENT_KEY_MEAS_BITS) & pos_mask
        device_type_code = keys >> (_ACTIVE_DEVICE_KEY_POS_BITS + _ACTIVE_MEASUREMENT_KEY_MEAS_BITS)

        def assign(
            device_code: int,
            meas_code: int,
            state_index: np.ndarray,
            target: np.ndarray,
        ) -> None:
            if state_index.size == 0:
                return
            mask = (device_type_code == int(device_code)) & (meas_type_code == int(meas_code))
            if not np.any(mask):
                return
            rows = np.flatnonzero(mask).astype(np.intp, copy=False)
            pos = device_pos[rows]
            valid_pos = (pos >= 0) & (pos < state_index.size)
            if not np.any(valid_pos):
                return
            state_idx = state_index[pos[valid_pos].astype(np.intp, copy=False)]
            valid_state = state_idx >= 0
            if np.any(valid_state):
                source_rows = rows[valid_pos][valid_state]
                target[state_idx[valid_state].astype(np.intp, copy=False)] = values[source_rows]

        assign(DEVICE_TYPE_ACGenerator, MEAS_TYPE_P_GEN, gen_index, self.initial_gen_p_array)
        assign(DEVICE_TYPE_ACGenerator, MEAS_TYPE_Q_GEN, gen_index, self.initial_gen_q_array)
        assign(DEVICE_TYPE_ACLoad, MEAS_TYPE_P_LOAD, load_index, self.initial_load_p_array)
        assign(DEVICE_TYPE_ACLoad, MEAS_TYPE_Q_LOAD, load_index, self.initial_load_q_array)

    @staticmethod
    def _covers_all_state_indices(indices: np.ndarray, count: int) -> bool:
        count = int(count)
        if count <= 0:
            return True
        values = np.asarray(indices, dtype=np.int64)
        if values.size < count:
            return False
        valid = (values >= 0) & (values < count)
        if not np.any(valid):
            return False
        seen = np.zeros(count, dtype=bool)
        seen[values[valid].astype(np.intp, copy=False)] = True
        return bool(np.all(seen))

    def _fast_active_observability_certificate(self) -> Optional[ObservabilityResult]:
        """Certify the generated flat-start AC vectorized active-measurement path."""
        if not bool(getattr(self, "flat_start", False)) and not bool(
            getattr(self, "_allow_nonflat_fast_observability_certificate", False)
        ):
            return None
        if not bool(getattr(self, "active_measurements_are_vectorized", False)):
            return None
        n_state = int(getattr(self, "n_state", 0))
        if n_state <= 0:
            return None
        measurement_count = self._active_measurement_count()
        if measurement_count < n_state:
            return None
        redundancy_target = targeted_redundancy_count(
            n_state,
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        )
        if redundancy_target > 0:
            return None
        plan_tables = getattr(self, "_active_measurement_plan_tables_cache", None)
        if not isinstance(plan_tables, dict) or not plan_tables:
            return None
        vector_plans = (
            getattr(self, "_active_branch_transformer_vector_plan", None),
            getattr(self, "_active_simple_jacobian_plan", None),
            getattr(self, "_active_zero_current_vector_plan", None),
            getattr(self, "_active_generator_measurement_plan", None),
            getattr(self, "_active_balance_measurement_plan", None),
        )
        if any(plan is None for plan in vector_plans):
            return None
        handled = np.zeros(measurement_count, dtype=bool)
        for plan in vector_plans:
            plan_handled = np.asarray(plan.get("handled_mask", ()), dtype=bool)
            if plan_handled.size != measurement_count:
                return None
            handled |= plan_handled
        if not np.all(handled):
            return None

        island_alive = np.asarray(getattr(self, "_ac_island_alive_mask", ()), dtype=bool)
        alive_island_count = int(np.count_nonzero(island_alive))
        reference_pos = np.asarray(getattr(self, "reference_pos", ()), dtype=np.int64)
        if alive_island_count and reference_pos.size < alive_island_count:
            return None
        if int(getattr(self, "n_voltage", 0)) > 0:
            reference_voltage = getattr(self, "reference_voltage_by_pos", {})
            if not reference_voltage:
                return None
            if alive_island_count > 1:
                node_island_pos = np.asarray(getattr(self, "_ac_node_island_pos", ()), dtype=np.int32)
                voltage_ref_pos = np.fromiter((int(pos) for pos in reference_voltage.keys()), dtype=np.int64)
                valid_ref = (voltage_ref_pos >= 0) & (voltage_ref_pos < node_island_pos.size)
                if not np.any(valid_ref):
                    return None
                covered_islands = np.unique(node_island_pos[voltage_ref_pos[valid_ref].astype(np.intp, copy=False)])
                alive_islands = np.flatnonzero(island_alive).astype(np.int32, copy=False)
                if not np.all(np.isin(alive_islands, covered_islands)):
                    return None

        balance_plan = self._active_balance_measurement_plan
        p_row_by_pos = np.asarray(balance_plan.get("p_row_by_pos", ()), dtype=np.int32)
        q_row_by_pos = np.asarray(balance_plan.get("q_row_by_pos", ()), dtype=np.int32)
        n_nodes = int(getattr(self, "n_nodes", 0))
        if p_row_by_pos.size != n_nodes or q_row_by_pos.size != n_nodes:
            return None
        if np.any(p_row_by_pos < 0) or np.any(q_row_by_pos < 0):
            return None
        required_balance_arrays = (
            "gen_p_rows",
            "gen_q_rows",
            "load_p_rows",
            "load_q_rows",
            "switch_p_rows",
            "switch_q_rows",
            "shunt_q_rows",
        )
        for key in required_balance_arrays:
            values = np.asarray(balance_plan.get(key, ()), dtype=np.int64)
            if values.size and np.any(values < 0):
                return None
        if int(getattr(self, "n_switch_current", 0)) > 0:
            expected = 2 * int(self.n_switch_current)
            if (
                np.asarray(balance_plan.get("switch_p_rows", ())).size != expected
                or np.asarray(balance_plan.get("switch_q_rows", ())).size != expected
            ):
                return None

        generator_plan = self._active_generator_measurement_plan
        gen_kind = np.asarray(generator_plan.get("value_kind", ()), dtype=np.int16)
        gen_index = np.asarray(generator_plan.get("value_index", ()), dtype=np.int64)
        n_generator_power = int(getattr(self, "n_generator_power", 0))
        if n_generator_power:
            if not self._covers_all_state_indices(gen_index[gen_kind == MEAS_TYPE_P_GEN], n_generator_power):
                return None
            if not self._covers_all_state_indices(gen_index[gen_kind == MEAS_TYPE_Q_GEN], n_generator_power):
                return None

        simple_plan = self._active_simple_jacobian_plan
        load_kind = np.asarray(simple_plan.get("load_kind", ()), dtype=np.int16)
        load_index = np.asarray(simple_plan.get("load_index", ()), dtype=np.int64)
        n_load_power = int(getattr(self, "n_load_power", 0))
        if n_load_power:
            if not self._covers_all_state_indices(load_index[load_kind == MEAS_TYPE_P_LOAD], n_load_power):
                return None
            if not self._covers_all_state_indices(load_index[load_kind == MEAS_TYPE_Q_LOAD], n_load_power):
                return None

        return ObservabilityResult(
            observable=True,
            rank=n_state,
            state_count=n_state,
            measurement_count=measurement_count,
            deficiency=0,
            singular_values=np.array([], dtype=np.float64),
            weak_states=[],
        )

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch AC rank deficiencies and optional post-observability redundancy."""
        total_added = 0
        max_count = max(0, int(self.targeted_pseudo_measurement_max))
        step = max(1, int(getattr(self, "targeted_pseudo_measurement_step", 10)))
        redundancy_target = targeted_redundancy_count(
            getattr(self, "n_state", 0),
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        )
        observability = None
        while total_added < max_count:
            observability = self.observability_analysis()
            batch_limit = min(max_count - total_added, step)
            if observability.observable:
                redundancy = max(0, int(observability.measurement_count) - int(observability.state_count))
                missing = redundancy_target - redundancy
                if missing <= 0:
                    break
                added = self._add_weak_direction_observability_pseudo_measurements(
                    observability,
                    min(batch_limit, missing),
                )
                if added == 0:
                    break
                total_added += added
                observability = None
                continue
            next_idx = self._next_measurement_idx()
            existing_keys = self._active_measurement_keys_ref()
            added = 0
            refreshed = False
            measurement_count_before = len(self.measurements)
            base_table_before = _measurement_table_from_measurements(self.measurements)
            remaining = batch_limit
            for state_idx, _score in observability.weak_states:
                if added >= remaining:
                    break
                next_idx, added_count = self._append_targeted_observability_pseudo(
                    next_idx,
                    state_idx,
                    existing_keys,
                    remaining - added,
                )
                added += added_count
            if added == 0:
                added = self._add_weak_direction_observability_pseudo_measurements(observability, remaining)
                refreshed = added > 0
            if added == 0:
                added = self._add_structural_rank_restoring_pseudo_measurements(remaining)
                refreshed = added > 0
            if added == 0:
                break
            total_added += added
            if not refreshed:
                current_table = _measurement_table_from_measurements(self.measurements)
                current_count = int(current_table.idx.size)
                if current_count > measurement_count_before:
                    tail_rows = np.arange(measurement_count_before, current_count, dtype=np.int64)
                    tail_table = measurement_table_take(current_table, tail_rows)
                    refreshed = self._incremental_update_active_measurement_indexes(
                        MeasurementTableView(tail_table, normalized=getattr(self.measurements, "normalized", False)),
                        source_row_start=measurement_count_before,
                        master_table=base_table_before,
                    )
            if not refreshed:
                self._refresh_active_measurement_indexes()
            observability = None
        if observability is None and total_added < max_count:
            observability = self.observability_analysis()
        if observability is not None:
            self._initial_observability_cache = observability
        return total_added

    def _add_weak_direction_observability_pseudo_measurements(
        self,
        observability: ObservabilityResult,
        max_add: int,
        refresh: bool = True,
    ) -> int:
        """Add the candidate pseudo rows that best observe the current weak direction."""
        if max_add <= 0:
            return 0
        candidates = self._observability_pseudo_candidate_measurements()
        if not candidates:
            return 0
        selected_rows = self._select_weak_direction_pseudo_candidates(observability, candidates, max_add)
        if selected_rows.size == 0:
            return 0
        next_idx = self._next_measurement_idx()
        base_table_before = _measurement_table_from_measurements(self.measurements)
        measurement_count_before = int(base_table_before.idx.size)
        candidate_table = getattr(candidates, "table", None)
        if candidate_table is None:
            return 0
        selected_table = measurement_table_take(candidate_table, selected_rows)
        selected_count = int(selected_table.idx.size)
        selected_table.idx = np.arange(next_idx, next_idx + selected_count, dtype=np.int64)
        self._append_pseudo_measurement_table(selected_table)
        if refresh:
            refreshed = self._incremental_update_active_measurement_indexes(
                MeasurementTableView(selected_table, normalized=True),
                source_row_start=measurement_count_before,
                master_table=base_table_before,
            )
            if not refreshed:
                self._refresh_active_measurement_indexes()
        return selected_count

    def _observability_pseudo_candidate_measurements(self) -> Sequence[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        existing_keys = self._active_measurement_keys_ref()
        candidate_keys = set()
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "observability pseudo candidate build")
        store_strings = False
        candidate_rows = _PseudoMeasurementBuffer(
            int(np.asarray(ppc.get("bus", np.zeros((0, len(BUS_COLS)))), dtype=np.float64).shape[0])
            + 3 * int(np.asarray(ppc.get("load", np.zeros((0, len(LOAD_COLS)))), dtype=np.float64).shape[0])
            + 3 * int(np.asarray(ppc.get("gen", np.zeros((0, len(GEN_COLS)))), dtype=np.float64).shape[0])
            + 4 * int(np.asarray(ppc.get("branch", np.zeros((0, len(BRANCH_COLS)))), dtype=np.float64).shape[0])
            + 4
            * int(
                np.asarray(
                    ppc.get("transformer", np.zeros((0, len(TRANSFORMER_COLS)))),
                    dtype=np.float64,
                ).shape[0]
            ),
            store_strings=store_strings,
        )

        def add(
            device_type_code: int,
            device_type: str,
            device_name: str,
            device_pos: int,
            meas_type_code: int,
            meas_type: str,
            value: float,
        ) -> None:
            if int(device_pos) < 0:
                return
            key = self._active_measurement_key(device_type_code, int(device_pos), meas_type_code)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}" if store_strings else ""
            if (
                int(meas_type_code) in _VOLTAGE_MEASUREMENT_TYPES
                and self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code)
            ):
                return
            if key in existing_keys or key in candidate_keys:
                return
            candidate_rows.add(
                pseudo_name,
                device_type,
                int(device_type_code),
                device_name,
                int(device_pos),
                meas_type,
                int(meas_type_code),
                float(value),
            )
            candidate_keys.add(key)

        bus = np.asarray(ppc["bus"], dtype=np.float64)
        node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", None)
        if node_solver_pos is None:
            self._build_ppc_state_index_arrays()
            node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", np.asarray([], dtype=np.int32))
        node_rows = np.flatnonzero(np.asarray(node_solver_pos, dtype=np.int32) >= 0).astype(np.int64, copy=False)
        if node_rows.size > 1:
            node_rows = node_rows[np.argsort(bus[node_rows.astype(np.intp, copy=False), BUS_COLS["idx"]], kind="stable")]
        node_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_ACNode, node_rows, bus.shape[0])
        node_names = (
            self._ppc_names_for_rows(ppc["bus_name"], node_rows)
            if store_strings
            else repeat("", int(node_rows.size))
        )
        for row, pos, name in zip(node_rows, node_plan_pos, node_names):
            add(
                DEVICE_TYPE_ACNode,
                "ACNode",
                str(name) if store_strings else "",
                int(pos),
                MEAS_TYPE_V,
                "V",
                float(bus[int(row), BUS_COLS["voltage"]] or 1.0),
            )

        def node_voltage_for_rows(device_topology, rows: np.ndarray) -> np.ndarray:
            if rows.size == 0:
                return np.asarray([], dtype=np.float64)
            node_pos = np.asarray(device_topology.node_pos, dtype=np.int32)[rows.astype(np.intp, copy=False)]
            voltage = np.ones(rows.size, dtype=np.float64)
            valid = (node_pos >= 0) & (node_pos < bus.shape[0])
            if np.any(valid):
                voltage[valid] = bus[node_pos[valid].astype(np.intp, copy=False), BUS_COLS["voltage"]]
            return voltage

        load = np.asarray(ppc["load"], dtype=np.float64)
        load_topology = topology.devices.get("load")
        load_rows = (
            self._sorted_alive_ppc_rows(load, LOAD_COLS["idx"], load_topology.alive_mask)
            if load_topology is not None
            else np.asarray([], dtype=np.int64)
        )
        load_voltage = node_voltage_for_rows(load_topology, load_rows) if load_topology is not None else np.asarray([], dtype=np.float64)
        if load_rows.size:
            load_table = load[load_rows.astype(np.intp, copy=False)]
            p_values = load_table[:, LOAD_COLS["pbase"]] * (
                load_table[:, LOAD_COLS["pv0"]]
                + load_table[:, LOAD_COLS["pv1"]] * load_voltage
                + load_table[:, LOAD_COLS["pv2"]] * load_voltage * load_voltage
            )
            q_values = load_table[:, LOAD_COLS["qbase"]] * (
                load_table[:, LOAD_COLS["qv0"]]
                + load_table[:, LOAD_COLS["qv1"]] * load_voltage
                + load_table[:, LOAD_COLS["qv2"]] * load_voltage * load_voltage
            )
            load_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_ACLoad, load_rows, load.shape[0])
            load_names = (
                self._ppc_names_for_rows(ppc["load_name"], load_rows)
                if store_strings
                else repeat("", int(load_rows.size))
            )
            for name, pos, p_value, q_value, voltage_value in zip(
                load_names,
                load_plan_pos,
                p_values,
                q_values,
                load_voltage,
            ):
                device_name = str(name) if store_strings else ""
                add(DEVICE_TYPE_ACLoad, "ACLoad", device_name, int(pos), MEAS_TYPE_P_LOAD, "P_LOAD", float(p_value))
                add(DEVICE_TYPE_ACLoad, "ACLoad", device_name, int(pos), MEAS_TYPE_Q_LOAD, "Q_LOAD", float(q_value))
                add(DEVICE_TYPE_ACLoad, "ACLoad", device_name, int(pos), MEAS_TYPE_V_LOAD, "V_LOAD", float(voltage_value or 1.0))

        gen = np.asarray(ppc["gen"], dtype=np.float64)
        gen_topology = topology.devices.get("gen")
        gen_rows = (
            self._sorted_alive_ppc_rows(gen, GEN_COLS["idx"], gen_topology.alive_mask)
            if gen_topology is not None
            else np.asarray([], dtype=np.int64)
        )
        gen_voltage = node_voltage_for_rows(gen_topology, gen_rows) if gen_topology is not None else np.asarray([], dtype=np.float64)
        if gen_rows.size:
            gen_table = gen[gen_rows.astype(np.intp, copy=False)]
            p_values = gen_table[:, GEN_COLS["p"]]
            q_values = gen_table[:, GEN_COLS["q"]]
            setpoint_mask = (np.abs(p_values) <= 1e-12) & (np.abs(q_values) <= 1e-12)
            if np.any(setpoint_mask):
                p_values = p_values.copy()
                q_values = q_values.copy()
                p_values[setpoint_mask] = gen_table[setpoint_mask, GEN_COLS["p_set"]]
                q_values[setpoint_mask] = gen_table[setpoint_mask, GEN_COLS["q_set"]]
            gen_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_ACGenerator, gen_rows, gen.shape[0])
            gen_names = (
                self._ppc_names_for_rows(ppc["gen_name"], gen_rows)
                if store_strings
                else repeat("", int(gen_rows.size))
            )
            for name, pos, p_value, q_value, voltage_value in zip(
                gen_names,
                gen_plan_pos,
                p_values,
                q_values,
                gen_voltage,
            ):
                device_name = str(name) if store_strings else ""
                add(DEVICE_TYPE_ACGenerator, "ACGenerator", device_name, int(pos), MEAS_TYPE_P_GEN, "P_GEN", float(p_value))
                add(DEVICE_TYPE_ACGenerator, "ACGenerator", device_name, int(pos), MEAS_TYPE_Q_GEN, "Q_GEN", float(q_value))
                add(DEVICE_TYPE_ACGenerator, "ACGenerator", device_name, int(pos), MEAS_TYPE_V_GEN, "V_GEN", float(voltage_value or 1.0))

        def add_terminal_candidates(
            device_type_code: int,
            device_type: str,
            table_name: str,
            name_key: str,
            device_key: str,
            cols: Dict[str, int],
        ) -> None:
            table = np.asarray(ppc[table_name], dtype=np.float64)
            device_topology = topology.devices.get(device_key)
            rows = (
                self._sorted_alive_ppc_rows(table, cols["idx"], device_topology.alive_mask)
                if device_topology is not None
                else np.asarray([], dtype=np.int64)
            )
            if rows.size == 0:
                return
            table_rows = table[rows.astype(np.intp, copy=False)]
            plan_pos = self._plan_pos_for_ppc_rows(int(device_type_code), rows, table.shape[0])
            names = (
                self._ppc_names_for_rows(ppc[name_key], rows)
                if store_strings
                else repeat("", int(rows.size))
            )
            for name, pos, values in zip(names, plan_pos, table_rows):
                device_name = str(name) if store_strings else ""
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_P_FROM, "P_FROM", float(values[cols["i_p"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_Q_FROM, "Q_FROM", float(values[cols["i_q"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_P_TO, "P_TO", float(values[cols["j_p"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_Q_TO, "Q_TO", float(values[cols["j_q"]] or 0.0))

        add_terminal_candidates(DEVICE_TYPE_ACBranch, "ACBranch", "branch", "branch_name", "branch", BRANCH_COLS)
        add_terminal_candidates(DEVICE_TYPE_ACTransformer, "ACTransformer", "transformer", "transformer_name", "transformer", TRANSFORMER_COLS)

        candidate_table = self._pseudo_measurement_table(
            candidate_rows.names,
            candidate_rows.device_types,
            candidate_rows.device_names,
            candidate_rows.meas_types,
            candidate_rows.values,
            self.pseudo_measurement_weight,
            device_type_codes=candidate_rows.device_type_codes,
            meas_type_codes=candidate_rows.meas_type_codes,
            device_positions=candidate_rows.device_positions,
        )
        return MeasurementTableView(candidate_table, normalized=True)

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: Sequence[Measurement],
        max_add: int,
    ) -> np.ndarray:
        if max_add <= 0 or not candidates:
            return np.asarray([], dtype=np.int64)
        candidate_count = int(len(candidates))
        x = self.initial_state()
        cache = self._observability_matrix_cache_for(observability, None, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, None)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return np.arange(min(int(max_add), candidate_count), dtype=np.int64)
        cache = getattr(self, "_weak_direction_candidate_jacobian_cache", None)
        table = getattr(candidates, "table", None)
        cache_key = (id(candidates), id(table), len(candidates), self.n_state)
        if cache is not None and cache.get("key") == cache_key and cache.get("candidates") is candidates:
            candidate_h = cache["H"]
        else:
            start = time.perf_counter() if self.profile_enabled else None
            candidate_h = self.jacobian_sparse(x, candidates)
            if start is not None:
                self._record_profile_time("init.weak_direction_candidate_jacobian", time.perf_counter() - start)
            self._weak_direction_candidate_jacobian_cache = {
                "key": cache_key,
                "candidates": candidates,
                "H": candidate_h,
            }
        start = time.perf_counter() if self.profile_enabled else None
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if start is not None:
            self._record_profile_time("init.weak_direction_candidate_scores", time.perf_counter() - start)
        if scores.size != len(candidates) or not np.any(scores > 0.0):
            return np.arange(min(int(max_add), candidate_count), dtype=np.int64)
        positive = np.flatnonzero(scores > 0.0)
        if positive.size > max_add:
            top = positive[np.argpartition(-scores[positive], max_add - 1)[:max_add]]
            order = top[np.argsort(-scores[top], kind="stable")]
        else:
            order = positive[np.argsort(-scores[positive], kind="stable")]
        selected = np.asarray(order[:max_add], dtype=np.int64)
        if selected.size:
            return selected
        return np.arange(min(int(max_add), candidate_count), dtype=np.int64)

    def _rank_restoring_candidate_measurements(self) -> MeasurementTableView:
        """Build low-weight candidates from invalid real device rows, excluding node V and angles."""
        table = _measurement_table_from_measurements(self.measurements)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is None or np.asarray(meas_type_code).size != int(table.idx.size):
            warnings.warn(
                "AC SE rank-restoring candidates require meas_type_code; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            meas_type_code = np.zeros(int(table.idx.size), dtype=np.int16)
            table.meas_type_code = meas_type_code
        else:
            meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is None or np.asarray(device_pos).size != int(table.idx.size):
            warnings.warn(
                "AC SE rank-restoring candidates require measurement device_pos; "
                "name-based index rebuild is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            empty_table = self._pseudo_measurement_table(
                (),
                (),
                (),
                (),
                (),
                self.pseudo_measurement_weight,
            )
            return MeasurementTableView(empty_table, normalized=True)
        else:
            device_pos = np.asarray(device_pos, dtype=np.int64)
        existing_key_array = self._active_measurement_key_array_ref()
        next_idx = self._next_measurement_idx()
        invalid_rows = np.flatnonzero(
            (~np.asarray(table.valid, dtype=bool))
            & (np.asarray(table.weight, dtype=np.float64) > 0.0)
        )
        if invalid_rows.size == 0:
            empty_table = self._pseudo_measurement_table(
                (),
                (),
                (),
                (),
                (),
                self.pseudo_measurement_weight,
            )
            return MeasurementTableView(empty_table, normalized=True)
        available, scale, _from_pos, _to_pos = self._measurement_scale_for_codes(
            device_type_code[invalid_rows],
            device_pos[invalid_rows],
            meas_type_code[invalid_rows],
        )
        candidate_mask = (device_pos[invalid_rows] >= 0) & np.asarray(available, dtype=bool)
        candidate_mask &= ~np.isin(meas_type_code[invalid_rows], _ANGLE_MEASUREMENT_TYPE_CODES)
        candidate_mask &= ~(
            (device_type_code[invalid_rows] == DEVICE_TYPE_ACNode)
            & (meas_type_code[invalid_rows] == MEAS_TYPE_V)
        )
        candidate_rows = invalid_rows[candidate_mask]
        if candidate_rows.size == 0:
            empty_table = self._pseudo_measurement_table(
                (),
                (),
                (),
                (),
                (),
                self.pseudo_measurement_weight,
            )
            return MeasurementTableView(empty_table, normalized=True)
        keys = self._active_measurement_key_array(
            device_type_code[candidate_rows].astype(np.int64, copy=False),
            device_pos[candidate_rows].astype(np.int64, copy=False),
            meas_type_code[candidate_rows].astype(np.int64, copy=False),
        )
        keep = np.ones(candidate_rows.size, dtype=bool)
        if existing_key_array.size:
            keep &= ~self._active_measurement_key_membership(keys)
        if keys.size:
            _unique_keys, first_pos = np.unique(keys, return_index=True)
            first_mask = np.zeros(keys.size, dtype=bool)
            first_mask[first_pos.astype(np.intp, copy=False)] = True
            keep &= first_mask
        rows = candidate_rows[keep].astype(np.int64, copy=False)
        if rows.size == 0:
            empty_table = self._pseudo_measurement_table(
                (),
                (),
                (),
                (),
                (),
                self.pseudo_measurement_weight,
            )
            return MeasurementTableView(empty_table, normalized=True)
        scale_rows = scale[candidate_mask][keep]
        values = np.divide(
            np.asarray(table.value, dtype=np.float64)[rows],
            scale_rows,
            out=np.zeros(rows.size, dtype=np.float64),
            where=np.abs(scale_rows) > 1e-12,
        )
        names = np.asarray([], dtype=object)
        device_types = np.asarray([], dtype=object)
        device_names = np.asarray([], dtype=object)
        meas_types = np.asarray([], dtype=object)
        candidate_table = self._pseudo_measurement_table(
            names,
            device_types,
            device_names,
            meas_types,
            values,
            self.pseudo_measurement_weight,
            idx_start=next_idx,
            device_type_codes=device_type_code[rows],
            meas_type_codes=meas_type_code[rows],
            device_positions=device_pos[rows],
        )
        return MeasurementTableView(candidate_table, normalized=True)

    def _rank_restoring_candidate_indices(self, candidates: Sequence[Measurement], max_add: int) -> np.ndarray:
        """Select candidate rows that participate in a higher structural-rank matching."""
        if max_add <= 0 or not candidates:
            return np.asarray([], dtype=np.int64)
        base_measurements = self.active_measurements
        x = self.initial_state()
        base_h = self.jacobian_sparse(x, base_measurements)
        base_rank = sparse_structural_rank(base_h)
        if base_rank is None or base_rank >= self.n_state:
            return np.asarray([], dtype=np.int64)

        base_table = _measurement_table_from_measurements(base_measurements)
        candidate_table = _measurement_table_from_measurements(candidates)
        combined_measurements = MeasurementTableView(
            concat_measurement_tables(base_table, candidate_table),
            normalized=getattr(base_measurements, "normalized", False),
        )
        combined_h = self.jacobian_sparse(x, combined_measurements)
        combined_rank = sparse_structural_rank(combined_h)
        if combined_rank is None or combined_rank <= base_rank:
            return np.asarray([], dtype=np.int64)

        matching = sp_maximum_bipartite_matching(combined_h, perm_type="row")
        base_rows = len(base_measurements)
        selected = np.asarray(matching, dtype=np.int64)
        selected = selected[selected >= base_rows] - int(base_rows)
        if selected.size:
            selected = np.unique(selected)[:max_add].astype(np.int64, copy=False)
        remaining = max_add - int(selected.size)
        if remaining <= 0:
            return selected

        selected_rows = selected.astype(np.int64, copy=False)
        all_candidate_rows = np.arange(len(candidates), dtype=np.int64)
        remaining_mask = np.ones(all_candidate_rows.size, dtype=bool)
        if selected_rows.size:
            remaining_mask[selected_rows.astype(np.intp, copy=False)] = False
        remaining_rows = all_candidate_rows[remaining_mask]
        if remaining_rows.size == 0:
            return selected

        selected_table = measurement_table_take(candidate_table, selected_rows)
        remaining_table = measurement_table_take(candidate_table, remaining_rows)
        current_measurements = MeasurementTableView(
            concat_measurement_tables(base_table, selected_table),
            normalized=getattr(base_measurements, "normalized", False),
        )
        current_h = self.jacobian_sparse(x, current_measurements)
        remaining_candidates = MeasurementTableView(remaining_table, normalized=True)
        candidate_h = self.jacobian_sparse(x, remaining_candidates)
        anchor_local_indices = self._angle_anchor_candidate_indices(current_h, candidate_h, remaining)
        anchor_local_indices = np.asarray(anchor_local_indices, dtype=np.int64)
        if anchor_local_indices.size == 0:
            return selected

        valid_anchor = (anchor_local_indices >= 0) & (anchor_local_indices < remaining_rows.size)
        if not np.any(valid_anchor):
            return selected
        anchor_rows = remaining_rows[anchor_local_indices[valid_anchor].astype(np.intp, copy=False)]
        return np.unique(np.concatenate((selected, anchor_rows.astype(np.int64, copy=False))))[:max_add]

    def _angle_anchor_candidate_indices(self, current_h, candidate_h, max_add: int) -> np.ndarray:
        """Select non-angle rows that connect unanchored angle components to an anchored component."""
        if max_add <= 0 or self.n_angle <= 0 or candidate_h.shape[0] == 0:
            return np.asarray([], dtype=np.int64)
        tol = 1e-12
        current_angle = current_h[:, : self.n_angle].tocsr() if issparse(current_h) else csr_matrix(current_h[:, : self.n_angle])
        n_angle = self.n_angle
        parent = np.arange(n_angle, dtype=np.int32)
        anchored = np.zeros(n_angle, dtype=bool)

        def find(pos: int) -> int:
            while int(parent[pos]) != pos:
                parent[pos] = parent[int(parent[pos])]
                pos = int(parent[pos])
            return int(pos)

        def union(left: int, right: int) -> int:
            root_l = find(left)
            root_r = find(right)
            if root_l != root_r:
                parent[root_r] = root_l
                anchored[root_l] = anchored[root_l] or anchored[root_r]
            return root_l

        for row in range(current_angle.shape[0]):
            start = int(current_angle.indptr[row])
            end = int(current_angle.indptr[row + 1])
            if start == end:
                continue
            cols = current_angle.indices[start:end]
            vals = current_angle.data[start:end]
            keep = np.abs(vals) > tol
            if not np.any(keep):
                continue
            cols = cols[keep]
            vals = vals[keep]
            root = int(cols[0])
            for col in cols[1:]:
                root = union(root, int(col))
            root = find(root)
            if abs(float(np.sum(vals))) > tol:
                anchored[root] = True

        roots = {find(col) for col in range(n_angle)}
        anchored_roots = {find(col) for col in range(n_angle) if anchored[find(col)]}
        if roots and roots.issubset(anchored_roots):
            return np.asarray([], dtype=np.int64)

        candidate_angle = candidate_h[:, : self.n_angle].tocsr() if issparse(candidate_h) else csr_matrix(candidate_h[:, : self.n_angle])
        row_roots: List[Tuple[int, Tuple[int, ...], bool]] = []
        incident: Dict[int, List[int]] = {}
        anchor_rows: List[int] = []
        for row in range(candidate_angle.shape[0]):
            start = int(candidate_angle.indptr[row])
            end = int(candidate_angle.indptr[row + 1])
            if start == end:
                continue
            cols = candidate_angle.indices[start:end]
            vals = candidate_angle.data[start:end]
            keep = np.abs(vals) > tol
            if not np.any(keep):
                continue
            vals = vals[keep]
            row_component_roots = tuple(sorted({find(int(col)) for col in cols[keep]}))
            if not row_component_roots:
                continue
            row_id = len(row_roots)
            is_anchor_row = abs(float(np.sum(vals))) > tol
            row_roots.append((row, row_component_roots, is_anchor_row))
            if is_anchor_row:
                anchor_rows.append(row_id)
            for root in row_component_roots:
                incident.setdefault(root, []).append(row_id)

        selected = np.empty(min(int(max_add), int(candidate_angle.shape[0])), dtype=np.int64)
        selected_count = 0
        used_rows = set()
        visited = set(anchored_roots)
        queue = list(anchored_roots)

        def activate(row_id: int) -> None:
            nonlocal selected_count
            if selected_count >= max_add or row_id in used_rows:
                return
            candidate_row, component_roots, _is_anchor_row = row_roots[row_id]
            new_roots = [root for root in component_roots if root not in visited]
            if not new_roots:
                return
            used_rows.add(row_id)
            selected[selected_count] = int(candidate_row)
            selected_count += 1
            for root in new_roots:
                visited.add(root)
                queue.append(root)

        for row_id in anchor_rows:
            activate(row_id)
            if selected_count >= max_add:
                break

        cursor = 0
        while cursor < len(queue) and selected_count < max_add:
            root = queue[cursor]
            cursor += 1
            for row_id in incident.get(root, ()):
                activate(row_id)
                if selected_count >= max_add:
                    break
        return selected[:selected_count].astype(np.int64, copy=False)

    def _add_structural_rank_restoring_pseudo_measurements(self, max_add: int) -> int:
        """Add only invalid real-measurement candidates that improve structural observability."""
        candidates = self._rank_restoring_candidate_measurements()
        selected_indices = self._rank_restoring_candidate_indices(candidates, max_add)
        if selected_indices.size == 0:
            return 0
        next_idx = self._next_measurement_idx()
        base_table_before = _measurement_table_from_measurements(self.measurements)
        measurement_count_before = int(base_table_before.idx.size)
        candidate_table = getattr(candidates, "table", None)
        if candidate_table is None:
            return 0
        selected_rows = np.asarray(selected_indices, dtype=np.int64)
        selected_table = measurement_table_take(candidate_table, selected_rows)
        selected_count = int(selected_table.idx.size)
        selected_table.idx = np.arange(next_idx, next_idx + selected_count, dtype=np.int64)
        self._append_pseudo_measurement_table(selected_table)
        refreshed = self._incremental_update_active_measurement_indexes(
            MeasurementTableView(selected_table, normalized=True),
            source_row_start=measurement_count_before,
            master_table=base_table_before,
        )
        if not refreshed:
            self._refresh_active_measurement_indexes()
        return selected_count

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
        max_add_or_unused,
        max_add: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Translate a weak compact AC state into the smallest useful pseudo measurement."""
        if max_add is None:
            max_add = int(max_add_or_unused)
        else:
            max_add = int(max_add)
        arrays = self._state_meta_arrays_ref()
        state_pos = int(state_idx)
        if state_pos < 0 or state_pos >= int(np.asarray(arrays["kind"], dtype=object).size):
            return next_idx, 0
        meta_kind = str(arrays["kind"][state_pos])
        name = str(arrays["device_name"][state_pos])
        device_type_code = int(arrays["device_type_code"][state_pos])
        device_pos = int(arrays["device_pos"][state_pos])
        added_total = 0

        def add(
            target_device_type_code: int,
            device_type: str,
            device_name: str,
            target_device_pos: int,
            target_meas_type_code: int,
            meas_type: str,
            value: float,
        ) -> Tuple[int, int]:
            nonlocal added_total
            if added_total >= max_add:
                return next_idx, 0
            target_device_type_code = int(target_device_type_code)
            target_device_pos = int(target_device_pos)
            target_meas_type_code = int(target_meas_type_code)
            if target_device_type_code <= 0 or target_device_pos < 0 or target_meas_type_code <= 0:
                return next_idx, 0
            key = self._active_measurement_key(target_device_type_code, target_device_pos, target_meas_type_code)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if (
                target_meas_type_code in _VOLTAGE_MEASUREMENT_TYPES
                and self._voltage_pseudo_is_covered_by_code(
                    target_device_type_code,
                    target_device_pos,
                    target_meas_type_code,
                )
            ):
                return next_idx, 0
            if key in existing_keys:
                return next_idx, 0
            new_idx = self._append_pseudo_measurement_rows(
                next_idx,
                [pseudo_name],
                [device_type],
                [device_name],
                [meas_type],
                [value],
                device_type_codes=[target_device_type_code],
                meas_type_codes=[target_meas_type_code],
                device_positions=[target_device_pos],
            )
            existing_keys.add(key)
            added_total += 1
            return new_idx, 1

        def plan_ppc_row(target_device_type_code: int, target_device_pos: int) -> int:
            rows = getattr(self, "_ac_measurement_plan_rows_by_type_code", {}).get(int(target_device_type_code))
            if rows is None:
                return -1
            rows = np.asarray(rows, dtype=np.int64)
            if target_device_pos < 0 or target_device_pos >= rows.size:
                return -1
            return int(rows[target_device_pos])

        def node_voltage_seed(target_device_pos: int) -> float:
            plan_pos = int(target_device_pos)
            node_plan_pos = np.asarray(getattr(self, "_ac_node_plan_pos", ()), dtype=np.int64)
            if 0 <= plan_pos < node_plan_pos.size:
                node_pos = int(node_plan_pos[plan_pos])
                if 0 <= node_pos < len(getattr(self, "file_voltage", ())):
                    return float(self.file_voltage[node_pos] or 1.0)
            return 1.0

        def terminal_pq(table_name: str, cols: Dict[str, int], target_device_type_code: int, target_device_pos: int) -> Tuple[float, float]:
            ppc = self._ac_ppc_dict()
            row = plan_ppc_row(target_device_type_code, target_device_pos)
            table = np.asarray(ppc.get(table_name, np.zeros((0, len(cols)))), dtype=np.float64)
            if row < 0 or row >= table.shape[0]:
                return 0.0, 0.0
            values = table[row]
            return float(values[cols["p"]] or 0.0), float(values[cols["q"]] or 0.0)

        def generator_pq(target_device_pos: int) -> Tuple[float, float]:
            plan_pos = int(target_device_pos)
            state_index = np.asarray(getattr(self, "_ac_generator_plan_index", ()), dtype=np.int64)
            if plan_pos < 0 or plan_pos >= state_index.size:
                return 0.0, 0.0
            idx = int(state_index[plan_pos])
            if idx < 0:
                return 0.0, 0.0
            return float(self.initial_gen_p_array[idx]), float(self.initial_gen_q_array[idx])

        def load_pq(target_device_pos: int) -> Tuple[float, float]:
            plan_pos = int(target_device_pos)
            state_index = np.asarray(getattr(self, "_ac_load_plan_index", ()), dtype=np.int64)
            if plan_pos < 0 or plan_pos >= state_index.size:
                return 0.0, 0.0
            idx = int(state_index[plan_pos])
            if idx < 0:
                return 0.0, 0.0
            return float(self.initial_load_p_array[idx]), float(self.initial_load_q_array[idx])

        if meta_kind == "angle":
            return next_idx, 0
        if meta_kind == "voltage" and device_type_code == DEVICE_TYPE_ACNode:
            return add(
                DEVICE_TYPE_ACNode,
                "ACNode",
                name,
                device_pos,
                MEAS_TYPE_V,
                "V",
                node_voltage_seed(device_pos),
            )
        if meta_kind == "zero_current" and device_type_code == DEVICE_TYPE_ACZeroBranch:
            p, q = terminal_pq("zero_branch", ZERO_BRANCH_COLS, device_type_code, device_pos)
            next_idx, added_p = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_P_FROM, "P_FROM", p)
            next_idx, added_q = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_Q_FROM, "Q_FROM", q)
            next_idx, added_p_to = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_P_TO, "P_TO", -p)
            next_idx, added_q_to = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_Q_TO, "Q_TO", -q)
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta_kind == "break_current" and device_type_code == DEVICE_TYPE_ACBreak:
            p, q = terminal_pq("break", BREAK_COLS, device_type_code, device_pos)
            next_idx, added_p = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_P_FROM, "P_FROM", p)
            next_idx, added_q = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_Q_FROM, "Q_FROM", q)
            next_idx, added_p_to = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_P_TO, "P_TO", -p)
            next_idx, added_q_to = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_Q_TO, "Q_TO", -q)
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta_kind in ("generator_p", "generator_q") and device_type_code == DEVICE_TYPE_ACGenerator:
            p, q = generator_pq(device_pos)
            meas_type_code = MEAS_TYPE_P_GEN if meta_kind == "generator_p" else MEAS_TYPE_Q_GEN
            meas_type = "P_GEN" if meta_kind == "generator_p" else "Q_GEN"
            return add(device_type_code, "ACGenerator", name, device_pos, meas_type_code, meas_type, p if meta_kind == "generator_p" else q)
        if meta_kind in ("load_p", "load_q") and device_type_code == DEVICE_TYPE_ACLoad:
            p, q = load_pq(device_pos)
            meas_type_code = MEAS_TYPE_P_LOAD if meta_kind == "load_p" else MEAS_TYPE_Q_LOAD
            meas_type = "P_LOAD" if meta_kind == "load_p" else "Q_LOAD"
            return add(device_type_code, "ACLoad", name, device_pos, meas_type_code, meas_type, p if meta_kind == "load_p" else q)
        return next_idx, 0

    def _add_power_balance_constraint_measurements(self) -> None:
        """Add nodal AC power-balance equations that tie P/Q states to the grid."""
        next_idx = self._next_measurement_idx()
        node_count = int(getattr(self, "n_nodes", 0))
        if node_count == 0:
            return
        row_count = 2 * node_count
        weight = 10.0
        store_strings = False
        if store_strings:
            node_names = np.asarray(getattr(self, "_ac_node_names", np.asarray([], dtype=object)), dtype=object)
            names = np.empty(row_count, dtype=object)
            names[0::2] = [f"constraint_p_balance_{name}" for name in node_names]
            names[1::2] = [f"constraint_q_balance_{name}" for name in node_names]
            meas_type = np.empty(row_count, dtype=object)
            meas_type[0::2] = "P_BALANCE"
            meas_type[1::2] = "Q_BALANCE"
            device_type = np.full(row_count, "ACPowerBalance", dtype=object)
            device_name = np.repeat(node_names, 2)
        else:
            names = np.asarray([], dtype=object)
            meas_type = np.asarray([], dtype=object)
            device_type = np.asarray([], dtype=object)
            device_name = np.asarray([], dtype=object)
        balance_code = DEVICE_TYPE_ACPowerBalance
        meas_type_code = np.empty(row_count, dtype=np.int16)
        meas_type_code[0::2] = MEAS_TYPE_P_BALANCE
        meas_type_code[1::2] = MEAS_TYPE_Q_BALANCE
        node_plan_pos = np.asarray(getattr(self, "_ac_node_plan_pos", ()), dtype=np.int64)
        solver_to_plan_pos = np.empty(node_count, dtype=np.int64)
        solver_to_plan_pos.fill(-1)
        valid_plan = (node_plan_pos >= 0) & (node_plan_pos < node_count)
        if np.any(valid_plan):
            solver_to_plan_pos[node_plan_pos[valid_plan].astype(np.intp, copy=False)] = np.nonzero(valid_plan)[0]
        device_pos = np.repeat(solver_to_plan_pos, 2)
        balance_table = MeasurementTable(
            idx=np.arange(next_idx, next_idx + row_count, dtype=np.int64),
            name=names,
            device_type=device_type,
            device_name=device_name,
            meas_type=meas_type,
            weight=np.full(row_count, weight, dtype=np.float64),
            valid=np.ones(row_count, dtype=bool),
            value=np.zeros(row_count, dtype=np.float64),
            device_type_code=np.full(row_count, balance_code, dtype=np.int16),
            angle_mask=np.zeros(row_count, dtype=bool),
            status_code=np.full(row_count, MEAS_STATUS_NORMAL, dtype=np.int16),
            rows_by_device_type_code={balance_code: np.arange(row_count, dtype=np.int64)},
            device_name_id=np.full(row_count, -1, dtype=np.int64),
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )
        base_table = _measurement_table_from_measurements(self.measurements)
        base_count = int(base_table.idx.size)
        combined_table = concat_measurement_tables(base_table, balance_table)
        rows_by_code = {
            int(code): np.asarray(rows, dtype=np.int64).copy()
            for code, rows in rows_by_device_type_code(base_table).items()
        }
        balance_rows = np.arange(base_count, base_count + row_count, dtype=np.int64)
        if balance_code in rows_by_code:
            rows_by_code[balance_code] = np.concatenate((rows_by_code[balance_code], balance_rows))
        else:
            rows_by_code[balance_code] = balance_rows
        combined_table.rows_by_device_type_code = rows_by_code
        combined_device_pos = getattr(combined_table, "device_pos", None)
        if combined_device_pos is None or np.asarray(combined_device_pos).size != int(combined_table.idx.size):
            combined_device_pos = np.empty(int(combined_table.idx.size), dtype=np.int64)
            combined_device_pos.fill(-1)
            base_device_pos = getattr(base_table, "device_pos", None)
            if base_device_pos is not None and np.asarray(base_device_pos).size == base_count:
                combined_device_pos[:base_count] = np.asarray(base_device_pos, dtype=np.int64)
            combined_device_pos[base_count:base_count + row_count] = device_pos
            combined_table.device_pos = combined_device_pos
        self.measurements = self._measurement_sequence_from_table(
            combined_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurement_table = combined_table
        self._max_measurement_idx = next_idx + row_count - 1

    def _node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real ACNode voltage measurements keyed by node index."""
        cached = getattr(self, "_node_voltage_measurement_cache", None)
        if cached is not None:
            return cached
        self._refresh_measurement_summary_cache()
        return getattr(self, "_node_voltage_measurement_cache", {})

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live incident AC branches, transformers, switches and zero branches."""
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "node incident degree build")
        bus_solver_pos = self._topology_bus_solver_pos(topology)
        node_ids = np.asarray(getattr(self, "_ac_node_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        degree_array = np.zeros(node_ids.size, dtype=np.int32)
        for device_key in ("branch", "transformer", "zero_branch", "switch", "break"):
            device_topology = topology.devices.get(device_key)
            if device_topology is None:
                continue
            if not hasattr(device_topology, "i_bus_pos") or not hasattr(device_topology, "j_bus_pos"):
                continue
            alive = np.asarray(device_topology.alive_mask, dtype=bool)
            i_bus = np.asarray(device_topology.i_bus_pos, dtype=np.int32)
            j_bus = np.asarray(device_topology.j_bus_pos, dtype=np.int32)
            i_valid = alive & (i_bus >= 0) & (i_bus < bus_solver_pos.size)
            j_valid = alive & (j_bus >= 0) & (j_bus < bus_solver_pos.size)
            if np.any(i_valid):
                i_pos = bus_solver_pos[i_bus[i_valid].astype(np.intp, copy=False)]
                i_pos = i_pos[i_pos >= 0]
                if i_pos.size:
                    i_pos = i_pos[i_pos < degree_array.size]
                    if i_pos.size:
                        np.add.at(degree_array, i_pos.astype(np.intp, copy=False), 1)
            if np.any(j_valid):
                j_pos = bus_solver_pos[j_bus[j_valid].astype(np.intp, copy=False)]
                j_pos = j_pos[j_pos >= 0]
                if j_pos.size:
                    j_pos = j_pos[j_pos < degree_array.size]
                    if j_pos.size:
                        np.add.at(degree_array, j_pos.astype(np.intp, copy=False), 1)
        self.node_degree_array = degree_array
        return {int(node_ids[pos]): int(degree_array[pos]) for pos in range(int(node_ids.size))}

    def _select_reference_positions(self) -> np.ndarray:
        """Choose one reference solver position per live AC island using topology arrays."""
        node_ids = np.asarray(getattr(self, "_ac_node_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        node_island_pos = np.asarray(getattr(self, "_ac_node_island_pos", np.asarray([], dtype=np.int32)), dtype=np.int32)
        island_alive = np.asarray(getattr(self, "_ac_island_alive_mask", np.asarray([], dtype=bool)), dtype=bool)
        island_ref_pos = np.asarray(getattr(self, "_ac_island_reference_solver_pos", np.asarray([], dtype=np.int32)), dtype=np.int32)
        if node_ids.size == 0:
            return np.asarray([], dtype=np.int32)
        cached_measured_by_pos = getattr(self, "_node_voltage_measurement_pos_mask_cache", None)
        if isinstance(cached_measured_by_pos, np.ndarray) and int(cached_measured_by_pos.size) == int(node_ids.size):
            measured_by_pos = np.asarray(cached_measured_by_pos, dtype=bool)
        else:
            voltage_measurements = getattr(self, "node_voltage_measurements", {})
            measured_by_pos = np.zeros(node_ids.size, dtype=bool)
            if voltage_measurements:
                measured_ids = np.fromiter((int(key) for key in voltage_measurements.keys()), dtype=np.int64)
                if measured_ids.size:
                    measured_by_pos = np.isin(node_ids, measured_ids, assume_unique=False)
        degree_array = np.asarray(getattr(self, "node_degree_array", np.asarray([], dtype=np.int32)), dtype=np.int32)
        if degree_array.size != node_ids.size:
            degree_array = np.asarray(
                [self.node_degrees.get(int(node_id), 0) for node_id in node_ids],
                dtype=np.int32,
            )
        references = []
        for island_pos in range(int(island_alive.size)):
            if not bool(island_alive[island_pos]):
                continue
            candidates = np.flatnonzero(node_island_pos == island_pos).astype(np.int32, copy=False)
            if candidates.size == 0:
                continue
            measured_mask = measured_by_pos[candidates.astype(np.intp, copy=False)]
            measured = candidates[measured_mask]
            if measured.size:
                measured_pos = measured.astype(np.intp, copy=False)
                order = np.lexsort((node_ids[measured_pos], -degree_array[measured_pos]))
                references.append(int(measured[int(order[0])]))
                continue
            ref_pos = int(island_ref_pos[island_pos]) if island_pos < island_ref_pos.size else -1
            if 0 <= ref_pos < node_ids.size:
                references.append(ref_pos)
                continue
            references.append(int(candidates[np.argmin(node_ids[candidates.astype(np.intp, copy=False)])]))
        return np.asarray(references, dtype=np.int32)

    def _reference_angle_offsets(self) -> Dict[int, float]:
        """Map each node position to the original island reference angle."""
        reference_pos = np.asarray(getattr(self, "reference_pos", np.asarray([], dtype=np.int32)), dtype=np.int32)
        node_island_pos = np.asarray(getattr(self, "_ac_node_island_pos", np.asarray([], dtype=np.int32)), dtype=np.int32)
        file_theta = np.asarray(getattr(self, "file_theta", np.asarray([], dtype=np.float64)), dtype=np.float64)
        offsets: Dict[int, float] = {}
        for ref_pos in reference_pos:
            ref_pos_int = int(ref_pos)
            if ref_pos_int < 0 or ref_pos_int >= node_island_pos.size:
                continue
            island_pos = int(node_island_pos[ref_pos_int])
            if island_pos < 0:
                continue
            offset = float(file_theta[ref_pos_int]) if ref_pos_int < file_theta.size else 0.0
            member_pos = np.flatnonzero(node_island_pos == island_pos)
            for pos in member_pos:
                offsets[int(pos)] = offset
        return offsets

    def _rebase_angle_measurements(self) -> None:
        """Convert absolute node angle measurements to the estimator reference frame."""
        if not getattr(self, "_has_valid_angle_measurements", True):
            return
        table = _measurement_table_from_measurements(self.measurements)
        row_count = int(table.idx.size)
        if row_count == 0:
            return
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = getattr(table, "meas_type_code", None)
        device_pos = getattr(table, "device_pos", None)
        if (
            meas_type_code is None
            or device_pos is None
            or np.asarray(meas_type_code).size != row_count
            or np.asarray(device_pos).size != row_count
        ):
            warnings.warn(
                "AC SE angle rebasing requires meas_type_code and device_pos; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        candidate_rows = np.flatnonzero(
            np.asarray(table.valid, dtype=bool)
            & (np.asarray(table.weight, dtype=np.float64) > 0.0)
            & (device_type_code == DEVICE_TYPE_ACNode)
            & ((meas_type_code == MEAS_TYPE_ANGLE) | (meas_type_code == MEAS_TYPE_THETA))
            & (device_pos >= 0)
        ).astype(np.int64, copy=False)
        if candidate_rows.size == 0:
            return
        reference_angle_by_pos = getattr(self, "reference_angle_by_pos", {})
        if not reference_angle_by_pos:
            return
        node_plan_pos = np.asarray(getattr(self, "_ac_node_plan_pos", ()), dtype=np.int64)
        valid_plan, node_pos = self._values_for_plan_pos(node_plan_pos, device_pos[candidate_rows])
        if not np.any(valid_plan):
            return
        ref_keys = np.fromiter((int(pos) for pos in reference_angle_by_pos.keys()), dtype=np.int64)
        if ref_keys.size == 0:
            return
        valid_keys = ref_keys >= 0
        if not np.any(valid_keys):
            return
        ref_keys = ref_keys[valid_keys]
        ref_values_input = np.fromiter(
            (float(value) for pos, value in reference_angle_by_pos.items() if int(pos) >= 0),
            dtype=np.float64,
            count=int(ref_keys.size),
        )
        max_pos = int(max(int(ref_keys.max()), int(node_pos[valid_plan].max())))
        ref_values = np.zeros(max_pos + 1, dtype=np.float64)
        ref_values[ref_keys.astype(np.intp, copy=False)] = ref_values_input
        valid_node = valid_plan & (node_pos >= 0) & (node_pos < ref_values.size)
        if np.any(valid_node):
            rows = candidate_rows[valid_node]
            table.value[rows] = np.asarray(table.value, dtype=np.float64)[rows] - ref_values[
                node_pos[valid_node].astype(np.intp, copy=False)
            ]
            self.measurement_table = table

    def _zero_tie_solver_edges_from_ppc(self) -> Tuple[np.ndarray, np.ndarray]:
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "zero-tie state layout")
        bus_solver_pos = self._topology_bus_solver_pos(topology)
        left_chunks = []
        right_chunks = []
        for device_key in ("zero_branch", "switch", "break"):
            device_topology = topology.devices.get(device_key)
            if device_topology is None:
                continue
            i_bus = np.asarray(device_topology.i_bus_pos, dtype=np.int32)
            j_bus = np.asarray(device_topology.j_bus_pos, dtype=np.int32)
            alive = np.asarray(device_topology.alive_mask, dtype=bool)
            valid = (
                alive
                & (i_bus >= 0)
                & (j_bus >= 0)
                & (i_bus < bus_solver_pos.size)
                & (j_bus < bus_solver_pos.size)
            )
            if not np.any(valid):
                continue
            i_pos = bus_solver_pos[i_bus[valid].astype(np.intp, copy=False)]
            j_pos = bus_solver_pos[j_bus[valid].astype(np.intp, copy=False)]
            valid_pos = (i_pos >= 0) & (j_pos >= 0)
            if np.any(valid_pos):
                left_chunks.append(i_pos[valid_pos].astype(np.int32, copy=False))
                right_chunks.append(j_pos[valid_pos].astype(np.int32, copy=False))
        if not left_chunks:
            empty = np.asarray([], dtype=np.int32)
            return empty, empty
        return np.concatenate(left_chunks), np.concatenate(right_chunks)

    def _build_zero_tie_state_layout(self) -> None:
        """Compress AC voltage/angle states across ideal switches and zero branches."""
        if bool(getattr(self, "_array_only_runtime", False)):
            self._build_zero_tie_state_layout_array_only()
            return
        n = int(getattr(self, "n_nodes", 0))
        parent = np.arange(n, dtype=np.int32)

        def find(pos: int) -> int:
            while int(parent[pos]) != pos:
                parent[pos] = parent[int(parent[pos])]
                pos = int(parent[pos])
            return int(pos)

        def union(left: int, right: int) -> None:
            root_l = find(left)
            root_r = find(right)
            if root_l != root_r:
                parent[root_r] = root_l

        ppc_edges = self._zero_tie_solver_edges_from_ppc()
        for i, j in zip(ppc_edges[0], ppc_edges[1]):
            union(int(i), int(j))

        groups: Dict[int, List[int]] = {}
        for pos in range(n):
            groups.setdefault(find(pos), []).append(pos)
        components = list(groups.values())
        components.sort(key=lambda group: group[0])
        self.zero_tie_components = components
        zero_tie_component_by_pos = np.empty(n, dtype=np.int32)
        for component_idx, component in enumerate(components):
            for pos in component:
                zero_tie_component_by_pos[pos] = component_idx
        self.zero_tie_component_by_pos = zero_tie_component_by_pos

        self.angle_col = np.full(n, -1, dtype=np.int32)
        self.voltage_col = np.full(n, -1, dtype=np.int32)
        self.angle_state_pos: List[int] = []
        self.angle_state_nodes: List[Sequence[int]] = []
        self.voltage_state_pos: List[int] = []
        self.voltage_state_nodes: List[Sequence[int]] = []
        self.ref_angles: Dict[int, float] = {}
        self.ref_voltages: Dict[int, float] = {}
        angle_unpack_nodes = []
        angle_unpack_cols = []
        voltage_unpack_nodes = []
        voltage_unpack_cols = []

        for component in components:
            ref_positions = [pos for pos in component if pos in self.ref_idx]
            if ref_positions:
                for pos in component:
                    self.ref_angles[pos] = 0.0
            else:
                col = len(self.angle_state_pos)
                rep_pos = component[0]
                self.angle_state_pos.append(rep_pos)
                self.angle_state_nodes.append(component)
                angle_unpack_nodes.extend(component)
                angle_unpack_cols.extend([col] * len(component))
                for pos in component:
                    self.angle_col[pos] = col

            voltage_ref_positions = [pos for pos in component if pos in self.reference_voltage_by_pos]
            if voltage_ref_positions:
                node_ids = np.asarray(getattr(self, "_ac_node_ids", np.arange(n)), dtype=np.int64)
                ref_pos = min(voltage_ref_positions, key=lambda pos: int(node_ids[int(pos)]))
                ref_voltage = max(float(self.reference_voltage_by_pos[ref_pos]), self.voltage_floor)
                for pos in component:
                    self.ref_voltages[pos] = ref_voltage
            else:
                v_col = len(self.voltage_state_pos)
                rep_pos = component[0]
                self.voltage_state_pos.append(rep_pos)
                self.voltage_state_nodes.append(component)
                voltage_unpack_nodes.extend(component)
                voltage_unpack_cols.extend([v_col] * len(component))
                for pos in component:
                    self.voltage_col[pos] = v_col

        self.angle_state_pos = np.asarray(self.angle_state_pos, dtype=np.int32)
        self.voltage_state_pos = np.asarray(self.voltage_state_pos, dtype=np.int32)
        self.n_angle = int(self.angle_state_pos.size)
        self.n_voltage = int(self.voltage_state_pos.size)
        self._angle_ref_nodes = np.fromiter(self.ref_angles.keys(), dtype=np.int32, count=len(self.ref_angles))
        self._angle_ref_values = np.fromiter(self.ref_angles.values(), dtype=np.float64, count=len(self.ref_angles))
        self._voltage_ref_nodes = np.fromiter(self.ref_voltages.keys(), dtype=np.int32, count=len(self.ref_voltages))
        self._voltage_ref_values = np.fromiter(
            self.ref_voltages.values(),
            dtype=np.float64,
            count=len(self.ref_voltages),
        )
        self._angle_unpack_nodes = np.asarray(angle_unpack_nodes, dtype=np.int32)
        self._angle_unpack_cols = np.asarray(angle_unpack_cols, dtype=np.int32)
        self._voltage_unpack_nodes = np.asarray(voltage_unpack_nodes, dtype=np.int32)
        self._voltage_unpack_cols = np.asarray(voltage_unpack_cols, dtype=np.int32)
        voltage_mask = self.voltage_col >= 0
        self.voltage_col[voltage_mask] = self.n_angle + self.voltage_col[voltage_mask]

    def _build_zero_tie_state_layout_array_only(self) -> None:
        """Array-only zero-tie layout without building per-node Python component lists."""
        n = int(getattr(self, "n_nodes", 0))
        left, right = self._zero_tie_solver_edges_from_ppc()
        if n <= 0:
            empty_i32 = np.asarray([], dtype=np.int32)
            empty_i64 = np.asarray([], dtype=np.int64)
            self.zero_tie_components = {}
            self.zero_tie_component_by_pos = empty_i32
            self.zero_tie_component_offsets = empty_i64
            self.zero_tie_component_indices = empty_i32
            self.angle_col = empty_i32.copy()
            self.voltage_col = empty_i32.copy()
            self.angle_state_pos = empty_i32.copy()
            self.voltage_state_pos = empty_i32.copy()
            self.angle_state_nodes = []
            self.voltage_state_nodes = []
            self.ref_angles = {}
            self.ref_voltages = {}
            self.n_angle = 0
            self.n_voltage = 0
            self._angle_ref_nodes = empty_i32.copy()
            self._angle_ref_values = np.asarray([], dtype=np.float64)
            self._voltage_ref_nodes = empty_i32.copy()
            self._voltage_ref_values = np.asarray([], dtype=np.float64)
            self._angle_unpack_nodes = empty_i32.copy()
            self._angle_unpack_cols = empty_i32.copy()
            self._voltage_unpack_nodes = empty_i32.copy()
            self._voltage_unpack_cols = empty_i32.copy()
            return

        if left.size and int(left.size) <= max(4096, n // 4):
            parent = np.arange(n, dtype=np.int32)
            rank = np.zeros(n, dtype=np.int8)

            def find_root(pos: int) -> int:
                while int(parent[pos]) != pos:
                    parent[pos] = parent[int(parent[pos])]
                    pos = int(parent[pos])
                return int(pos)

            for left_pos, right_pos in zip(left, right):
                root_l = find_root(int(left_pos))
                root_r = find_root(int(right_pos))
                if root_l == root_r:
                    continue
                rank_l = int(rank[root_l])
                rank_r = int(rank[root_r])
                if rank_l < rank_r:
                    parent[root_l] = root_r
                elif rank_l > rank_r:
                    parent[root_r] = root_l
                else:
                    parent[root_r] = root_l
                    rank[root_l] = rank_l + 1
            for pos in range(n):
                parent[pos] = find_root(pos)
            _roots, labels = np.unique(parent, return_inverse=True)
            labels = np.asarray(labels, dtype=np.int32)
            n_components = int(_roots.size)
        elif left.size:
            graph_rows = np.concatenate((left, right)).astype(np.int32, copy=False)
            graph_cols = np.concatenate((right, left)).astype(np.int32, copy=False)
            graph_data = np.ones(graph_rows.size, dtype=np.int8)
            graph = coo_matrix((graph_data, (graph_rows, graph_cols)), shape=(n, n)).tocsr()
            n_components, labels = sp_connected_components(graph, directed=False, return_labels=True)
            labels = np.asarray(labels, dtype=np.int32)
            n_components = int(n_components)
        else:
            labels = np.arange(n, dtype=np.int32)
            n_components = n

        node_order = np.arange(n, dtype=np.int32)
        first_pos_by_label = np.empty(n_components, dtype=np.int32)
        first_pos_by_label.fill(n)
        np.minimum.at(first_pos_by_label, labels.astype(np.intp, copy=False), node_order)
        label_order = np.argsort(first_pos_by_label, kind="stable").astype(np.int32, copy=False)
        label_to_component = np.empty(n_components, dtype=np.int32)
        label_to_component[label_order.astype(np.intp, copy=False)] = np.arange(n_components, dtype=np.int32)
        component_by_pos = label_to_component[labels.astype(np.intp, copy=False)]
        component_first_pos = first_pos_by_label[label_order.astype(np.intp, copy=False)]
        self.zero_tie_component_by_pos = component_by_pos.astype(np.int32, copy=False)

        component_counts = np.bincount(component_by_pos, minlength=n_components)
        sorted_nodes = np.argsort(component_by_pos, kind="stable").astype(np.int32, copy=False)
        component_offsets = np.empty(n_components + 1, dtype=np.int64)
        component_offsets[0] = 0
        if n_components:
            component_offsets[1:] = np.cumsum(component_counts, dtype=np.int64)
        self.zero_tie_component_offsets = component_offsets
        self.zero_tie_component_indices = sorted_nodes
        multi_components = np.flatnonzero(component_counts > 1).astype(np.int32, copy=False)
        if multi_components.size:
            sorted_components = component_by_pos[sorted_nodes.astype(np.intp, copy=False)]
            split_points = np.flatnonzero(sorted_components[1:] != sorted_components[:-1]) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), split_points.astype(np.int64, copy=False)))
            ends = np.concatenate((split_points.astype(np.int64, copy=False), np.asarray([sorted_nodes.size], dtype=np.int64)))
            component_members = {}
            for start, end in zip(starts, ends):
                comp_idx = int(sorted_components[int(start)])
                if component_counts[comp_idx] > 1:
                    component_members[comp_idx] = sorted_nodes[int(start) : int(end)]
            self.zero_tie_components = component_members
        else:
            self.zero_tie_components = {}

        ref_idx_array = np.fromiter(self.ref_idx, dtype=np.int32, count=len(self.ref_idx))
        ref_idx_array = ref_idx_array[(ref_idx_array >= 0) & (ref_idx_array < n)]
        angle_ref_component = np.zeros(n_components, dtype=bool)
        if ref_idx_array.size:
            angle_ref_component[component_by_pos[ref_idx_array.astype(np.intp, copy=False)].astype(np.intp, copy=False)] = True
        angle_component = ~angle_ref_component
        angle_col_by_component = np.empty(n_components, dtype=np.int32)
        angle_col_by_component.fill(-1)
        angle_components = np.flatnonzero(angle_component).astype(np.int32, copy=False)
        if angle_components.size:
            angle_col_by_component[angle_components.astype(np.intp, copy=False)] = np.arange(angle_components.size, dtype=np.int32)
        self.angle_col = angle_col_by_component[component_by_pos.astype(np.intp, copy=False)].astype(np.int32, copy=False)
        self.angle_state_pos = component_first_pos[angle_components.astype(np.intp, copy=False)].astype(np.int32, copy=False)
        self.angle_state_nodes = []
        self.n_angle = int(self.angle_state_pos.size)
        self._angle_ref_nodes = np.flatnonzero(angle_ref_component[component_by_pos.astype(np.intp, copy=False)]).astype(np.int32, copy=False)
        self._angle_ref_values = np.zeros(self._angle_ref_nodes.size, dtype=np.float64)
        self.ref_angles = {}

        voltage_ref_component = np.zeros(n_components, dtype=bool)
        voltage_ref_value_by_component = np.zeros(n_components, dtype=np.float64)
        voltage_ref_node_id_by_component = np.full(n_components, np.iinfo(np.int64).max, dtype=np.int64)
        node_ids = np.asarray(getattr(self, "_ac_node_ids", np.arange(n)), dtype=np.int64)
        for pos_raw, value_raw in self.reference_voltage_by_pos.items():
            pos = int(pos_raw)
            if pos < 0 or pos >= n:
                continue
            comp_idx = int(component_by_pos[pos])
            node_id = int(node_ids[pos]) if pos < node_ids.size else pos
            if (not voltage_ref_component[comp_idx]) or node_id < int(voltage_ref_node_id_by_component[comp_idx]):
                voltage_ref_component[comp_idx] = True
                voltage_ref_node_id_by_component[comp_idx] = node_id
                voltage_ref_value_by_component[comp_idx] = max(float(value_raw), self.voltage_floor)
        voltage_component = ~voltage_ref_component
        voltage_col_by_component = np.empty(n_components, dtype=np.int32)
        voltage_col_by_component.fill(-1)
        voltage_components = np.flatnonzero(voltage_component).astype(np.int32, copy=False)
        if voltage_components.size:
            voltage_col_by_component[voltage_components.astype(np.intp, copy=False)] = np.arange(
                voltage_components.size,
                dtype=np.int32,
            )
        voltage_col = voltage_col_by_component[component_by_pos.astype(np.intp, copy=False)].astype(np.int32, copy=False)
        self.voltage_state_pos = component_first_pos[voltage_components.astype(np.intp, copy=False)].astype(np.int32, copy=False)
        self.voltage_state_nodes = []
        self.n_voltage = int(self.voltage_state_pos.size)
        voltage_ref_mask = voltage_ref_component[component_by_pos.astype(np.intp, copy=False)]
        self._voltage_ref_nodes = np.flatnonzero(voltage_ref_mask).astype(np.int32, copy=False)
        self._voltage_ref_values = voltage_ref_value_by_component[
            component_by_pos[self._voltage_ref_nodes.astype(np.intp, copy=False)].astype(np.intp, copy=False)
        ].astype(np.float64, copy=False)
        self.ref_voltages = {
            int(pos): float(value)
            for pos, value in zip(self._voltage_ref_nodes.tolist(), self._voltage_ref_values.tolist())
        }

        self._angle_unpack_nodes = np.flatnonzero(self.angle_col >= 0).astype(np.int32, copy=False)
        self._angle_unpack_cols = self.angle_col[self._angle_unpack_nodes.astype(np.intp, copy=False)].astype(
            np.int32,
            copy=False,
        )
        self._voltage_unpack_nodes = np.flatnonzero(voltage_col >= 0).astype(np.int32, copy=False)
        self._voltage_unpack_cols = voltage_col[self._voltage_unpack_nodes.astype(np.intp, copy=False)].astype(
            np.int32,
            copy=False,
        )
        self.voltage_col = voltage_col
        voltage_mask = self.voltage_col >= 0
        self.voltage_col[voltage_mask] = self.n_angle + self.voltage_col[voltage_mask]

    def _build_y_matrix(self) -> np.ndarray:
        """Build the estimator admittance matrix with the same stamps as load flow."""
        n = int(getattr(self, "n_nodes", 0))
        row_chunks = []
        col_chunks = []
        data_chunks = []

        def add_terminal_stamps(i_pos, j_pos, yff, yft, ytf, ytt) -> None:
            i_pos = np.asarray(i_pos, dtype=np.int64)
            if i_pos.size == 0:
                return
            j_pos = np.asarray(j_pos, dtype=np.int64)
            row_chunks.append(np.concatenate((i_pos, i_pos, j_pos, j_pos)).astype(np.int64, copy=False))
            col_chunks.append(np.concatenate((i_pos, j_pos, i_pos, j_pos)).astype(np.int64, copy=False))
            data_chunks.append(
                np.concatenate(
                    (
                        np.asarray(yff, dtype=np.complex128),
                        np.asarray(yft, dtype=np.complex128),
                        np.asarray(ytf, dtype=np.complex128),
                        np.asarray(ytt, dtype=np.complex128),
                    )
                )
            )

        required_plan_attrs = (
            "_ac_branch_plan_i",
            "_ac_branch_plan_j",
            "_ac_branch_plan_yff",
            "_ac_branch_plan_yft",
            "_ac_branch_plan_ytf",
            "_ac_branch_plan_ytt",
            "_ac_transformer_plan_i",
            "_ac_transformer_plan_j",
            "_ac_transformer_plan_yff",
            "_ac_transformer_plan_yft",
            "_ac_transformer_plan_ytf",
            "_ac_transformer_plan_ytt",
        )
        missing_plan_attrs = [name for name in required_plan_attrs if not hasattr(self, name)]
        if missing_plan_attrs:
            self._warn_required_runtime_missing(", ".join(missing_plan_attrs), "Y matrix build")
        add_terminal_stamps(
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
            self._ac_branch_plan_yff,
            self._ac_branch_plan_yft,
            self._ac_branch_plan_ytf,
            self._ac_branch_plan_ytt,
        )
        add_terminal_stamps(
            self._ac_transformer_plan_i,
            self._ac_transformer_plan_j,
            self._ac_transformer_plan_yff,
            self._ac_transformer_plan_yft,
            self._ac_transformer_plan_ytf,
            self._ac_transformer_plan_ytt,
        )

        ppc = self._ac_ppc_dict()
        shunt_rows = self._required_array_attr("_ac_shunt_device_rows", dtype=np.int64, context="Y matrix build")
        if shunt_rows.size:
            shunt = np.asarray(ppc["shunt"], dtype=np.float64)[shunt_rows.astype(np.intp, copy=False)]
            control_type = shunt[:, SHUNT_COLS["control_type"]].astype(np.int64, copy=False)
            g_set = shunt[:, SHUNT_COLS["g_set"]]
            b_set = shunt[:, SHUNT_COLS["b_set"]]
            stamp_mask = (
                (control_type == SHUNT_B)
                | (control_type == SHUNT_Z)
                | (np.abs(g_set) > 0.0)
            )
            y_values = g_set[stamp_mask] + 1j * b_set[stamp_mask]
            nonzero = np.abs(y_values) > 0.0
            if np.any(nonzero):
                shunt_rows_array = np.asarray(self._ac_shunt_device_node_pos, dtype=np.int64)[stamp_mask][nonzero]
                row_chunks.append(shunt_rows_array)
                col_chunks.append(shunt_rows_array)
                data_chunks.append(np.asarray(y_values[nonzero], dtype=np.complex128))
        if row_chunks:
            rows = np.concatenate(row_chunks)
            cols = np.concatenate(col_chunks)
            data = np.concatenate(data_chunks)
        else:
            rows = cols = np.asarray([], dtype=np.int64)
            data = np.asarray([], dtype=np.complex128)
        Y = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        Y.sum_duplicates()
        return Y

    def _prepare_y_row_cache(self) -> None:
        """Cache sparse Y-row topology used repeatedly by generator Jacobian rows."""
        n = int(getattr(self, "n_nodes", 0))
        self._y_row_nodes = []
        self._y_row_y_conj = []
        self._y_row_diag_conj = np.conj(self.Y.diagonal()).astype(np.complex128, copy=False)
        y_indices = self.Y.indices if issparse(self.Y) else None
        y_data_conj = np.conj(self.Y.data) if issparse(self.Y) else None

        for pos in range(n):
            if issparse(self.Y):
                start, end = self.Y.indptr[pos], self.Y.indptr[pos + 1]
                nodes = y_indices[start:end]
                y_conj = y_data_conj[start:end]
            else:
                row = self.Y[pos, :]
                nodes = np.nonzero(row)[0].astype(np.int32)
                y_values = np.asarray(row[nodes], dtype=np.complex128)
                y_conj = np.conj(y_values)

            self._y_row_nodes.append(nodes)
            self._y_row_y_conj.append(y_conj)

    def initial_state(self) -> np.ndarray:
        if self.flat_start:
            theta = np.zeros(self.n_nodes, dtype=np.float64)
            voltage = np.ones(self.n_nodes, dtype=np.float64)
            return self._pack_state(theta, voltage, rebase_angles=False)
        return self._file_state()

    def _file_state(self) -> np.ndarray:
        """Pack the E-file voltage/angle snapshot in the selected reference frame."""
        return self._pack_state(self.file_theta, self.file_voltage, rebase_angles=True)

    def state_layout(self) -> Dict[str, object]:
        """Expose the canonical AC state layout for reuse by hybrid orchestration."""
        return {
            "state_labels": self.state_labels,
            "state_meta": self.state_meta,
            "angle_col": self.angle_col,
            "voltage_col": self.voltage_col,
            "n_state": self.n_state,
            "references": self.references,
        }

    def _pack_state(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
        switch_current: Optional[np.ndarray] = None,
        gen_p: Optional[np.ndarray] = None,
        gen_q: Optional[np.ndarray] = None,
        load_p: Optional[np.ndarray] = None,
        load_q: Optional[np.ndarray] = None,
        shunt_q: Optional[np.ndarray] = None,
        acac_p_from: Optional[np.ndarray] = None,
        acac_q_from: Optional[np.ndarray] = None,
        acac_p_to: Optional[np.ndarray] = None,
        acac_q_to: Optional[np.ndarray] = None,
        rebase_angles: bool = True,
    ) -> np.ndarray:
        """Pack full node values into the WLS state vector, excluding reference angles."""
        x = np.zeros(self.n_state, dtype=np.float64)
        if self.n_angle:
            angle_values = np.asarray(theta[self.angle_state_pos], dtype=np.float64).copy()
            if rebase_angles:
                offsets = np.fromiter(
                    (self.reference_angle_by_pos.get(int(pos), 0.0) for pos in self.angle_state_pos),
                    dtype=np.float64,
                    count=self.n_angle,
                )
                angle_values -= offsets
            x[: self.n_angle] = angle_values
        if self.n_voltage:
            x[self.n_angle : self.base_switch_re] = voltage[self.voltage_state_pos]
        if switch_current is not None and self.n_switch_current:
            x[self.base_switch_re : self.base_switch_im] = switch_current.real
            x[self.base_switch_im : self.base_gen_p] = switch_current.imag
        if self.n_generator_power:
            x[self.base_gen_p : self.base_gen_q] = (
                self.initial_gen_p_array if gen_p is None else np.asarray(gen_p, dtype=np.float64)
            )
            x[self.base_gen_q : self.base_load_p] = (
                self.initial_gen_q_array if gen_q is None else np.asarray(gen_q, dtype=np.float64)
            )
        if self.n_load_power:
            x[self.base_load_p : self.base_load_q] = (
                self.initial_load_p_array if load_p is None else np.asarray(load_p, dtype=np.float64)
            )
            x[self.base_load_q : self.base_shunt_q] = (
                self.initial_load_q_array if load_q is None else np.asarray(load_q, dtype=np.float64)
            )
        if self.n_shunt_q:
            x[self.base_shunt_q : self.base_acac_p_from] = (
                self.initial_shunt_q_array if shunt_q is None else np.asarray(shunt_q, dtype=np.float64)
            )
        if self.n_acac_power:
            x[self.base_acac_p_from : self.base_acac_q_from] = (
                self.initial_acac_p_from_array if acac_p_from is None else np.asarray(acac_p_from, dtype=np.float64)
            )
            x[self.base_acac_q_from : self.base_acac_p_to] = (
                self.initial_acac_q_from_array if acac_q_from is None else np.asarray(acac_q_from, dtype=np.float64)
            )
            x[self.base_acac_p_to : self.base_acac_q_to] = (
                self.initial_acac_p_to_array if acac_p_to is None else np.asarray(acac_p_to, dtype=np.float64)
            )
            x[self.base_acac_q_to : self.n_state] = (
                self.initial_acac_q_to_array if acac_q_to is None else np.asarray(acac_q_to, dtype=np.float64)
            )
        return x

    def _unpack_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Restore full theta/V arrays from the compact WLS state vector."""
        n = int(getattr(self, "n_nodes", 0))
        theta = np.zeros(n, dtype=np.float64)
        if self._angle_ref_nodes.size:
            theta[self._angle_ref_nodes] = self._angle_ref_values
        if self._angle_unpack_nodes.size:
            theta[self._angle_unpack_nodes] = x[self._angle_unpack_cols]

        state_voltage = np.asarray(x[self.n_angle : self.base_switch_re], dtype=np.float64).copy()
        state_voltage[state_voltage < self.voltage_floor] = self.voltage_floor
        voltage = np.ones(n, dtype=np.float64)
        if self._voltage_ref_nodes.size:
            voltage[self._voltage_ref_nodes] = self._voltage_ref_values
        if self._voltage_unpack_nodes.size:
            voltage[self._voltage_unpack_nodes] = state_voltage[self._voltage_unpack_cols]
        voltage[voltage < self.voltage_floor] = self.voltage_floor
        return theta, voltage

    def _switch_current_from_state(self, x: np.ndarray) -> np.ndarray:
        if self.n_switch_current == 0:
            return np.array([], dtype=np.complex128)
        return (
            np.asarray(x[self.base_switch_re : self.base_switch_im], dtype=np.float64)
            + 1j * np.asarray(x[self.base_switch_im : self.base_gen_p], dtype=np.float64)
        )

    @staticmethod
    def _complex_voltage(theta: np.ndarray, voltage: np.ndarray) -> np.ndarray:
        return voltage * np.exp(1j * theta)

    def _load_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level load totals from explicit P_LOAD/Q_LOAD states."""
        load_node_pos = np.asarray(getattr(self, "_ac_load_power_node_pos", ()), dtype=np.int64)
        if not load_node_pos.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(load_node_pos, weights=x[self.base_load_p : self.base_load_q], minlength=self.n_nodes),
            np.bincount(load_node_pos, weights=x[self.base_load_q : self.base_shunt_q], minlength=self.n_nodes),
        )

    def _shunt_q_injections_from_state(self, x: np.ndarray) -> np.ndarray:
        """Return node-level reactive injections from V-control shunt Q states."""
        if not self.n_shunt_q:
            return np.zeros(self.n_nodes, dtype=np.float64)
        return np.bincount(
            self.shunt_q_pos_array,
            weights=x[self.base_shunt_q : self.base_acac_p_from],
            minlength=self.n_nodes,
        )

    def _generator_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level generator totals from explicit P_GEN/Q_GEN states."""
        gen_node_pos = np.asarray(getattr(self, "_ac_generator_power_node_pos", ()), dtype=np.int64)
        if not gen_node_pos.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(gen_node_pos, weights=x[self.base_gen_p : self.base_gen_q], minlength=self.n_nodes),
            np.bincount(gen_node_pos, weights=x[self.base_gen_q : self.base_load_p], minlength=self.n_nodes),
        )

    def _acac_power_injections_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ACAC converter terminal powers injected into their AC nodes."""
        if not int(getattr(self, "n_acac_power", 0)):
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        i_pos = np.asarray(getattr(self, "_ac_acac_power_i_pos", ()), dtype=np.int64)
        j_pos = np.asarray(getattr(self, "_ac_acac_power_j_pos", ()), dtype=np.int64)
        if i_pos.size == 0 and j_pos.size == 0:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(i_pos, weights=x[self.base_acac_p_from : self.base_acac_q_from], minlength=self.n_nodes)
            + np.bincount(j_pos, weights=x[self.base_acac_p_to : self.base_acac_q_to], minlength=self.n_nodes),
            np.bincount(i_pos, weights=x[self.base_acac_q_from : self.base_acac_p_to], minlength=self.n_nodes)
            + np.bincount(j_pos, weights=x[self.base_acac_q_to : self.n_state], minlength=self.n_nodes),
        )

    def _switch_power_injections(
        self,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node injections from explicit zero-branch/switch current states."""
        if self.n_switch_current == 0:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        end_pos = self.switch_balance_end_pos
        current = switch_current[self.switch_balance_end_current_idx] * self.switch_balance_sign
        s = voltage_complex[end_pos] * np.conj(current)
        return (
            np.bincount(end_pos, weights=s.real, minlength=self.n_nodes),
            np.bincount(end_pos, weights=s.imag, minlength=self.n_nodes),
        )

    def _power_balance_totals(
        self,
        x: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate nodal mismatch S_net + S_switch + S_load - S_gen - Q_shunt."""
        s_network = voltage_complex * np.conj(self.Y.dot(voltage_complex))
        p_switch, q_switch = self._switch_power_injections(voltage_complex, switch_current)
        p_acac, q_acac = self._acac_power_injections_from_state(x)
        p_load, q_load = self._load_power_totals_from_state(x)
        p_gen, q_gen = self._generator_power_totals_from_state(x)
        q_shunt = self._shunt_q_injections_from_state(x)
        return (
            s_network.real + p_switch + p_acac + p_load - p_gen,
            s_network.imag + q_switch + q_acac + q_load - q_gen - q_shunt,
        )

    @staticmethod
    def _int_array(values: Sequence[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _concat_plan_arrays(chunks: Sequence[np.ndarray], dtype) -> np.ndarray:
        non_empty = [np.asarray(chunk, dtype=dtype) for chunk in chunks if len(chunk)]
        if not non_empty:
            return np.asarray([], dtype=dtype)
        return np.concatenate(non_empty).astype(dtype, copy=False)

    def _ac_ppc_dict(self) -> Dict:
        """Return the AC PPC model used by the estimator."""
        network = getattr(self, "network", None)
        if network is None:
            self._warn_required_runtime_missing("PPC-backed network", "AC PPC access")
        ppc = getattr(network, "ppc", None)
        if not (isinstance(ppc, dict) and ppc.get("format") == "ac_ppc_v1"):
            self._warn_required_runtime_missing("PPC-backed network", "AC PPC access")
        ensure_ac_ppc_topology(ppc)
        return ppc

    @staticmethod
    def _ppc_names_for_rows(names, rows: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.asarray([], dtype=object)
        name_array = np.asarray(names, dtype=object)
        return name_array[rows.astype(np.intp, copy=False)]

    @staticmethod
    def _sorted_alive_ppc_rows(table: np.ndarray, idx_col: int, alive_mask: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(np.asarray(alive_mask, dtype=bool)).astype(np.int64, copy=False)
        if rows.size <= 1:
            return rows
        order = np.argsort(np.asarray(table)[rows, idx_col], kind="stable")
        return rows[order].astype(np.int64, copy=False)

    @staticmethod
    def _topology_bus_solver_pos(topology) -> np.ndarray:
        bus_solver_pos = np.full(len(topology.bus_ids), -1, dtype=np.int32)
        active_bus_pos = np.flatnonzero(np.asarray(topology.bus_alive_mask, dtype=bool)).astype(np.intp, copy=False)
        if active_bus_pos.size:
            bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
        return bus_solver_pos

    def _build_ppc_state_index_arrays(self) -> None:
        """Build state-index lookups keyed by PPC row instead of device name."""
        profile_start = time.perf_counter() if self.profile_enabled else None
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "state index build")

        bus_solver_pos = self._topology_bus_solver_pos(topology)

        node_solver_pos = np.full(len(topology.node_ids), -1, dtype=np.int32)
        node_bus_pos = np.asarray(topology.node_to_bus_pos, dtype=np.int32)
        node_valid = (
            np.asarray(topology.node_alive_mask, dtype=bool)
            & (node_bus_pos >= 0)
            & (node_bus_pos < bus_solver_pos.size)
        )
        if np.any(node_valid):
            node_solver_pos[node_valid] = bus_solver_pos[node_bus_pos[node_valid].astype(np.intp, copy=False)]
        self._ac_node_solver_pos_by_ppc_row = node_solver_pos

        def single_device_state_arrays(table_name: str, name_key: str, device_key: str, idx_col: int):
            table = np.asarray(ppc.get(table_name, np.zeros((0, idx_col + 1))), dtype=np.float64)
            state = np.full(table.shape[0], -1, dtype=np.int32)
            device_topology = topology.devices.get(device_key)
            if table.shape[0] == 0 or device_topology is None:
                empty_rows = np.asarray([], dtype=np.int64)
                empty_pos = np.asarray([], dtype=np.int32)
                empty_names = np.asarray([], dtype=object)
                return state, empty_rows, empty_pos, empty_names
            rows = self._sorted_alive_ppc_rows(table, idx_col, device_topology.alive_mask)
            node_pos_by_row = np.full(table.shape[0], -1, dtype=np.int32)
            bus_pos = np.asarray(device_topology.bus_pos, dtype=np.int32)
            valid_bus = (bus_pos >= 0) & (bus_pos < bus_solver_pos.size)
            if np.any(valid_bus):
                node_pos_by_row[valid_bus] = bus_solver_pos[bus_pos[valid_bus].astype(np.intp, copy=False)]
            if rows.size:
                node_values = node_pos_by_row[rows.astype(np.intp, copy=False)]
                valid_rows = node_values >= 0
                rows = rows[valid_rows]
                node_values = node_values[valid_rows].astype(np.int32, copy=False)
                state[rows.astype(np.intp, copy=False)] = np.arange(rows.size, dtype=np.int32)
            else:
                node_values = np.asarray([], dtype=np.int32)
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            return state, rows.astype(np.int64, copy=False), node_values, names.astype(object, copy=False)

        (
            self._ac_generator_state_index_by_ppc_row,
            self._ac_generator_power_rows,
            self._ac_generator_power_node_pos,
            self._ac_generator_power_names,
        ) = single_device_state_arrays("gen", "gen_name", "gen", GEN_COLS["idx"])
        (
            self._ac_load_state_index_by_ppc_row,
            self._ac_load_power_rows,
            self._ac_load_power_node_pos,
            self._ac_load_power_names,
        ) = single_device_state_arrays("load", "load_name", "load", LOAD_COLS["idx"])

        _shunt_state, self._ac_shunt_device_rows, self._ac_shunt_device_node_pos, self._ac_shunt_device_names = (
            single_device_state_arrays("shunt", "shunt_name", "shunt", SHUNT_COLS["idx"])
        )

        def terminal_current_arrays(table_name: str, name_key: str, device_key: str, idx_col: int):
            table = np.asarray(ppc.get(table_name, np.zeros((0, idx_col + 1))), dtype=np.float64)
            device_topology = topology.devices.get(device_key)
            if table.shape[0] == 0 or device_topology is None:
                empty_rows = np.asarray([], dtype=np.int64)
                empty_pos = np.asarray([], dtype=np.int32)
                empty_names = np.asarray([], dtype=object)
                return empty_rows, empty_pos, empty_pos, empty_names
            rows = self._sorted_alive_ppc_rows(table, idx_col, device_topology.alive_mask)
            i_bus = np.asarray(device_topology.i_bus_pos, dtype=np.int32)
            j_bus = np.asarray(device_topology.j_bus_pos, dtype=np.int32)
            i_pos_by_row = np.full(table.shape[0], -1, dtype=np.int32)
            j_pos_by_row = np.full(table.shape[0], -1, dtype=np.int32)
            i_valid = (i_bus >= 0) & (i_bus < bus_solver_pos.size)
            j_valid = (j_bus >= 0) & (j_bus < bus_solver_pos.size)
            if np.any(i_valid):
                i_pos_by_row[i_valid] = bus_solver_pos[i_bus[i_valid].astype(np.intp, copy=False)]
            if np.any(j_valid):
                j_pos_by_row[j_valid] = bus_solver_pos[j_bus[j_valid].astype(np.intp, copy=False)]
            if rows.size:
                i_pos = i_pos_by_row[rows.astype(np.intp, copy=False)]
                j_pos = j_pos_by_row[rows.astype(np.intp, copy=False)]
                valid_rows = (i_pos >= 0) & (j_pos >= 0)
                rows = rows[valid_rows]
                i_pos = i_pos[valid_rows].astype(np.int32, copy=False)
                j_pos = j_pos[valid_rows].astype(np.int32, copy=False)
            else:
                i_pos = j_pos = np.asarray([], dtype=np.int32)
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            return rows.astype(np.int64, copy=False), i_pos, j_pos, names.astype(object, copy=False)

        zero_branch = np.asarray(ppc.get("zero_branch", np.zeros((0, len(ZERO_BRANCH_COLS)))), dtype=np.float64)
        breaker = np.asarray(ppc.get("break", np.zeros((0, len(BREAK_COLS)))), dtype=np.float64)
        zero_current_pos = np.full(zero_branch.shape[0], -1, dtype=np.int32)
        break_current_pos = np.full(breaker.shape[0], -1, dtype=np.int32)
        zero_rows, zero_i, zero_j, zero_names = terminal_current_arrays(
            "zero_branch",
            "zero_branch_name",
            "zero_branch",
            ZERO_BRANCH_COLS["idx"],
        )
        break_rows, break_i, break_j, break_names = terminal_current_arrays(
            "break",
            "break_name",
            "break",
            BREAK_COLS["idx"],
        )
        if zero_rows.size:
            zero_current_pos[zero_rows.astype(np.intp, copy=False)] = np.arange(zero_rows.size, dtype=np.int32)
        if break_rows.size:
            break_current_pos[break_rows.astype(np.intp, copy=False)] = (
                zero_rows.size + np.arange(break_rows.size, dtype=np.int32)
            )
        self._ac_zero_branch_current_pos_by_ppc_row = zero_current_pos
        self._ac_break_current_pos_by_ppc_row = break_current_pos
        self._ac_zero_current_names = np.concatenate((zero_names, break_names)).astype(object, copy=False)
        self._ac_zero_current_kind_code = np.concatenate(
            (
                np.zeros(zero_rows.size, dtype=np.int8),
                np.ones(break_rows.size, dtype=np.int8),
            )
        )
        self._ac_zero_current_i = np.concatenate((zero_i, break_i)).astype(np.int32, copy=False)
        self._ac_zero_current_j = np.concatenate((zero_j, break_j)).astype(np.int32, copy=False)

        acac = np.asarray(ppc.get("acac", np.zeros((0, len(ACAC_COLS)))), dtype=np.float64)
        acac_state_pos = np.full(acac.shape[0], -1, dtype=np.int32)
        acac_rows, acac_i, acac_j, acac_names = terminal_current_arrays(
            "acac",
            "acac_name",
            "acac",
            ACAC_COLS["idx"],
        )
        if acac_rows.size:
            acac_state_pos[acac_rows.astype(np.intp, copy=False)] = np.arange(acac_rows.size, dtype=np.int32)
        self._ac_acac_state_index_by_ppc_row = acac_state_pos
        self._ac_acac_power_rows = acac_rows
        self._ac_acac_power_i_pos = acac_i
        self._ac_acac_power_j_pos = acac_j
        self._ac_acac_power_names = acac_names
        if profile_start is not None:
            self._record_profile_time("init.state_index_arrays", time.perf_counter() - profile_start)

    def _ppc_sorted_alive_rows_for(self, ppc: Dict, table_name: str, device_key: str, idx_col: int):
        topology = ppc.get("_topology_arrays")
        table = np.asarray(ppc.get(table_name, np.zeros((0, idx_col + 1))), dtype=np.float64)
        device_topology = topology.devices.get(device_key) if topology is not None else None
        rows = (
            self._sorted_alive_ppc_rows(table, idx_col, device_topology.alive_mask)
            if device_topology is not None
            else np.asarray([], dtype=np.int64)
        )
        return table, device_topology, rows

    @staticmethod
    def _ppc_device_node_voltage_values(ppc: Dict, device_topology, rows: np.ndarray) -> np.ndarray:
        if rows.size == 0 or device_topology is None:
            return np.asarray([], dtype=np.float64)
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        node_pos = np.asarray(device_topology.node_pos, dtype=np.int32)[rows.astype(np.intp, copy=False)]
        voltage = np.ones(rows.size, dtype=np.float64)
        valid = (node_pos >= 0) & (node_pos < bus.shape[0])
        if np.any(valid):
            voltage[valid] = bus[node_pos[valid].astype(np.intp, copy=False), BUS_COLS["voltage"]]
        return voltage

    @staticmethod
    def _ppc_load_pseudo_power_arrays(ppc: Dict, rows: np.ndarray, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if rows.size == 0:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty
        load = np.asarray(ppc["load"], dtype=np.float64)[rows.astype(np.intp, copy=False)]
        p = load[:, LOAD_COLS["pbase"]] * (
            load[:, LOAD_COLS["pv0"]]
            + load[:, LOAD_COLS["pv1"]] * voltage
            + load[:, LOAD_COLS["pv2"]] * voltage * voltage
        )
        q = load[:, LOAD_COLS["qbase"]] * (
            load[:, LOAD_COLS["qv0"]]
            + load[:, LOAD_COLS["qv1"]] * voltage
            + load[:, LOAD_COLS["qv2"]] * voltage * voltage
        )
        return p.astype(np.float64, copy=False), q.astype(np.float64, copy=False)

    @staticmethod
    def _ppc_generator_pseudo_power_arrays(ppc: Dict, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if rows.size == 0:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty
        gen = np.asarray(ppc["gen"], dtype=np.float64)[rows.astype(np.intp, copy=False)]
        p = gen[:, GEN_COLS["p"]]
        q = gen[:, GEN_COLS["q"]]
        setpoint_mask = (np.abs(p) <= 1e-12) & (np.abs(q) <= 1e-12)
        if np.any(setpoint_mask):
            p = p.copy()
            q = q.copy()
            p[setpoint_mask] = gen[setpoint_mask, GEN_COLS["p_set"]]
            q[setpoint_mask] = gen[setpoint_mask, GEN_COLS["q_set"]]
        return p.astype(np.float64, copy=False), q.astype(np.float64, copy=False)

    def _ppc_terminal_voltage_pseudo_seed(self, ppc: Dict, device_topology, row: int) -> float:
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        topology = ppc.get("_topology_arrays")
        node_rows = (
            int(device_topology.i_node_pos[int(row)]),
            int(device_topology.j_node_pos[int(row)]),
        )
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "terminal voltage pseudo seed")
        for node_row in node_rows:
            if node_row < 0 or node_row >= len(topology.node_ids):
                continue
            measured = self._real_voltage_observation_value_for_node(int(topology.node_ids[node_row]))
            if measured is not None:
                return max(float(measured), self.voltage_floor)
        first_node_row = node_rows[0]
        if 0 <= first_node_row < bus.shape[0]:
            return max(float(bus[first_node_row, BUS_COLS["voltage"]] or 1.0), self.voltage_floor)
        return 1.0

    def _build_measurement_plan_lookup_arrays(self) -> None:
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "measurement plan lookup build")
        if not hasattr(self, "_ac_node_solver_pos_by_ppc_row"):
            self._build_ppc_state_index_arrays()

        bus_solver_pos = self._topology_bus_solver_pos(topology)
        node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", None)
        if node_solver_pos is None:
            raise RuntimeError("AC PPC solver-position arrays are not available")

        node_rows = np.flatnonzero(np.asarray(node_solver_pos, dtype=np.int32) >= 0).astype(np.int64, copy=False)
        node_plan_pos = node_solver_pos[node_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._ac_node_plan_pos = node_plan_pos

        def terminal_solver_positions(device_topology):
            i_bus = np.asarray(device_topology.i_bus_pos, dtype=np.int32)
            j_bus = np.asarray(device_topology.j_bus_pos, dtype=np.int32)
            i_pos = np.full(i_bus.shape, -1, dtype=np.int32)
            j_pos = np.full(j_bus.shape, -1, dtype=np.int32)
            i_valid = (i_bus >= 0) & (i_bus < bus_solver_pos.size)
            j_valid = (j_bus >= 0) & (j_bus < bus_solver_pos.size)
            if np.any(i_valid):
                i_pos[i_valid] = bus_solver_pos[i_bus[i_valid].astype(np.intp, copy=False)]
            if np.any(j_valid):
                j_pos[j_valid] = bus_solver_pos[j_bus[j_valid].astype(np.intp, copy=False)]
            return i_pos, j_pos

        def terminal_alive_rows(device_key: str):
            device_topology = topology.devices[device_key]
            i_pos, j_pos = terminal_solver_positions(device_topology)
            rows = np.flatnonzero(device_topology.alive_mask & (i_pos >= 0) & (j_pos >= 0)).astype(np.int64, copy=False)
            return rows, i_pos, j_pos

        branch = np.asarray(ppc["branch"], dtype=np.float64)
        branch_rows, branch_i_pos, branch_j_pos = terminal_alive_rows("branch")
        branch_table = branch[branch_rows.astype(np.intp, copy=False)] if branch_rows.size else branch[:0]
        if branch_rows.size:
            yff, yft, ytf, ytt = matpower_branch_stamp_vectorized(
                branch_table[:, BRANCH_COLS["r"]],
                branch_table[:, BRANCH_COLS["x"]],
                branch_table[:, BRANCH_COLS["b"]],
            )
        else:
            yff = yft = ytf = ytt = np.asarray([], dtype=np.complex128)
        self._ac_branch_plan_i = branch_i_pos[branch_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._ac_branch_plan_j = branch_j_pos[branch_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._ac_branch_plan_yff = np.asarray(yff, dtype=np.complex128)
        self._ac_branch_plan_yft = np.asarray(yft, dtype=np.complex128)
        self._ac_branch_plan_ytf = np.asarray(ytf, dtype=np.complex128)
        self._ac_branch_plan_ytt = np.asarray(ytt, dtype=np.complex128)

        transformer = np.asarray(ppc["transformer"], dtype=np.float64)
        transformer_rows, transformer_i_pos, transformer_j_pos = terminal_alive_rows("transformer")
        transformer_table = transformer[transformer_rows.astype(np.intp, copy=False)] if transformer_rows.size else transformer[:0]
        if transformer_rows.size:
            yff, yft, ytf, ytt = matpower_transformer_stamp_vectorized(
                transformer_table[:, TRANSFORMER_COLS["r"]],
                transformer_table[:, TRANSFORMER_COLS["x"]],
                transformer_table[:, TRANSFORMER_COLS["gt"]],
                transformer_table[:, TRANSFORMER_COLS["bt"]],
                transformer_table[:, TRANSFORMER_COLS["tap"]],
                transformer_table[:, TRANSFORMER_COLS["shift"]],
            )
        else:
            yff = yft = ytf = ytt = np.asarray([], dtype=np.complex128)
        self._ac_transformer_plan_i = transformer_i_pos[transformer_rows.astype(np.intp, copy=False)].astype(
            np.int64,
            copy=False,
        )
        self._ac_transformer_plan_j = transformer_j_pos[transformer_rows.astype(np.intp, copy=False)].astype(
            np.int64,
            copy=False,
        )
        self._ac_transformer_plan_yff = np.asarray(yff, dtype=np.complex128)
        self._ac_transformer_plan_yft = np.asarray(yft, dtype=np.complex128)
        self._ac_transformer_plan_ytf = np.asarray(ytf, dtype=np.complex128)
        self._ac_transformer_plan_ytt = np.asarray(ytt, dtype=np.complex128)

        def zero_current_plan(device_key: str, name_key: str, current_pos_by_row: np.ndarray):
            rows, i_pos, j_pos = terminal_alive_rows(device_key)
            if rows.size:
                current_pos = np.asarray(current_pos_by_row, dtype=np.int32)[rows.astype(np.intp, copy=False)]
                valid = current_pos >= 0
                rows = rows[valid]
                i_values = i_pos[rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
                j_values = j_pos[rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
                current_pos = current_pos[valid].astype(np.int64, copy=False)
            else:
                i_values = j_values = current_pos = np.asarray([], dtype=np.int64)
            return rows, i_values, j_values, current_pos

        zero_rows, zero_i, zero_j, zero_current = zero_current_plan(
            "zero_branch",
            "zero_branch_name",
            getattr(self, "_ac_zero_branch_current_pos_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_zero_branch_plan_i = zero_i
        self._ac_zero_branch_plan_j = zero_j
        self._ac_zero_branch_plan_current_pos = zero_current

        break_rows, break_i, break_j, break_current = zero_current_plan(
            "break",
            "break_name",
            getattr(self, "_ac_break_current_pos_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_break_plan_i = break_i
        self._ac_break_plan_j = break_j
        self._ac_break_plan_current_pos = break_current

        acac_rows, acac_i_pos, acac_j_pos = terminal_alive_rows("acac")
        if acac_rows.size:
            acac_state_index = np.asarray(
                getattr(self, "_ac_acac_state_index_by_ppc_row", np.asarray([], dtype=np.int32)),
                dtype=np.int32,
            )
            in_range = acac_rows < acac_state_index.size
            if np.any(in_range):
                acac_rows = acac_rows[in_range]
                state_values = acac_state_index[acac_rows.astype(np.intp, copy=False)]
                valid = state_values >= 0
                acac_rows = acac_rows[valid]
                acac_i_values = acac_i_pos[acac_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
                acac_j_values = acac_j_pos[acac_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
                acac_state_values = state_values[valid].astype(np.int64, copy=False)
            else:
                acac_rows = acac_i_values = acac_j_values = acac_state_values = np.asarray([], dtype=np.int64)
        else:
            acac_i_values = acac_j_values = acac_state_values = np.asarray([], dtype=np.int64)
        self._ac_acac_plan_i = acac_i_values
        self._ac_acac_plan_j = acac_j_values
        self._ac_acac_plan_index = acac_state_values

        def single_solver_positions(device_topology):
            bus_pos = np.asarray(device_topology.bus_pos, dtype=np.int32)
            out = np.full(bus_pos.shape, -1, dtype=np.int32)
            valid = (bus_pos >= 0) & (bus_pos < bus_solver_pos.size)
            if np.any(valid):
                out[valid] = bus_solver_pos[bus_pos[valid].astype(np.intp, copy=False)]
            return out

        def single_plan(device_key: str, name_key: str, state_index_by_row: np.ndarray):
            device_topology = topology.devices[device_key]
            node_pos = single_solver_positions(device_topology)
            state_index = np.asarray(state_index_by_row, dtype=np.int32)
            rows = np.flatnonzero(
                device_topology.alive_mask
                & (node_pos >= 0)
                & (np.arange(node_pos.size, dtype=np.int64) < state_index.size)
            ).astype(np.int64, copy=False)
            if rows.size:
                state_values = state_index[rows.astype(np.intp, copy=False)]
                valid = state_values >= 0
                rows = rows[valid]
                state_values = state_values[valid].astype(np.int64, copy=False)
                node_values = node_pos[rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
            else:
                node_values = state_values = np.asarray([], dtype=np.int64)
            return rows, node_values, state_values

        gen_rows, gen_node_pos, gen_state_index = single_plan(
            "gen",
            "gen_name",
            getattr(self, "_ac_generator_state_index_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_generator_plan_node_pos = gen_node_pos
        self._ac_generator_plan_index = gen_state_index

        load_rows, load_node_pos, load_state_index = single_plan(
            "load",
            "load_name",
            getattr(self, "_ac_load_state_index_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_load_plan_node_pos = load_node_pos
        self._ac_load_plan_index = load_state_index

        self._ac_measurement_plan_rows_by_type_code = {
            DEVICE_TYPE_ACNode: node_rows,
            DEVICE_TYPE_ACBranch: branch_rows,
            DEVICE_TYPE_ACTransformer: transformer_rows,
            DEVICE_TYPE_ACLoad: load_rows,
            DEVICE_TYPE_ACGenerator: gen_rows,
            DEVICE_TYPE_ACZeroBranch: zero_rows,
            DEVICE_TYPE_ACBreak: break_rows,
            DEVICE_TYPE_ACACConverter: acac_rows,
            DEVICE_TYPE_ACZeroBranchConstraint: zero_rows,
            DEVICE_TYPE_ACBreakConstraint: break_rows,
            DEVICE_TYPE_ACPowerBalance: node_rows,
        }
        plan_pos_by_row = {}
        for code, rows in self._ac_measurement_plan_rows_by_type_code.items():
            rows = np.asarray(rows, dtype=np.int64)
            table_size = 0
            if rows.size:
                table_size = max(table_size, int(rows.max()) + 1)
            lookup = np.full(table_size, -1, dtype=np.int64)
            if rows.size:
                lookup[rows.astype(np.intp, copy=False)] = np.arange(rows.size, dtype=np.int64)
            plan_pos_by_row[int(code)] = lookup
        self._ac_measurement_plan_pos_by_ppc_row = plan_pos_by_row
        table = getattr(getattr(self, "measurements", None), "table", None)
        present_device_codes = None
        if table is not None and getattr(table, "device_type_code", None) is not None:
            rows_by_code = getattr(table, "rows_by_device_type_code", None)
            if isinstance(rows_by_code, dict) and rows_by_code:
                present_device_codes = {int(code) for code in rows_by_code.keys()}
            else:
                present_device_codes = set(np.unique(np.asarray(table.device_type_code, dtype=np.int16)).astype(object, copy=False))
        self._ac_measurement_present_device_codes = present_device_codes
        self._ac_branch_transformer_plan_kind_by_type_code = {
            DEVICE_TYPE_ACBranch: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACTransformer: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
        }
        self._ac_zero_current_plan_kind_by_type_code = {
            DEVICE_TYPE_ACZeroBranch: _AC_ZERO_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACBreak: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
        }
        self._ac_simple_plan_kind_by_type_code = {
            DEVICE_TYPE_ACNode: _AC_NODE_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACGenerator: _AC_GENERATOR_SIMPLE_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACLoad: _AC_LOAD_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACZeroBranchConstraint: _AC_CONSTRAINT_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_ACBreakConstraint: _AC_CONSTRAINT_MEAS_TYPE_LOOKUP,
        }
        self._ac_generator_plan_kind_by_type_code = {
            DEVICE_TYPE_ACGenerator: _AC_GENERATOR_POWER_MEAS_TYPE_LOOKUP,
        }
        self._ac_acac_plan_kind_by_type_code = {
            DEVICE_TYPE_ACACConverter: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
        }
        self._ac_balance_plan_kind_by_type_code = {
            DEVICE_TYPE_ACPowerBalance: _AC_BALANCE_MEAS_TYPE_LOOKUP,
        }
        self._ac_measurement_plan_kind_codes = {
            "branch_transformer": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_branch_transformer_plan_kind_by_type_code.items()
            },
            "zero_current": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_zero_current_plan_kind_by_type_code.items()
            },
            "simple": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_simple_plan_kind_by_type_code.items()
            },
            "generator": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_generator_plan_kind_by_type_code.items()
            },
            "acac": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_acac_plan_kind_by_type_code.items()
            },
            "balance": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_balance_plan_kind_by_type_code.items()
            },
        }

    def _measurement_kind_code_maps_for(
        self,
        meas_kind_by_type_code: Dict[int, Dict[str, int]],
    ) -> Dict[int, np.ndarray]:
        key = tuple(sorted((int(code), id(values)) for code, values in meas_kind_by_type_code.items()))
        cache = getattr(self, "_ac_measurement_custom_kind_code_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        result = {
            int(code): _measurement_type_code_lookup(kind_map)
            for code, kind_map in meas_kind_by_type_code.items()
        }
        self._ac_measurement_custom_kind_code_cache = (key, result)
        return result

    def _ensure_measurement_sequence_indexes(self, measurements: Sequence[Measurement]) -> MeasurementTable:
        table = _measurement_table_from_measurements(measurements)
        meas_type_code = getattr(table, "meas_type_code", None)
        device_pos = getattr(table, "device_pos", None)
        if (
            meas_type_code is not None
            and np.asarray(meas_type_code).size == int(table.idx.size)
            and device_pos is not None
            and np.asarray(device_pos).size == int(table.idx.size)
        ):
            return table
        device_pos = self._measurement_device_pos_array(table)
        table.device_pos = device_pos
        cache = getattr(table, "_device_pos_plan_cache", None)
        if cache is not None:
            cache.clear()
        cache = getattr(table, "_meas_kind_plan_cache", None)
        if cache is not None:
            cache.clear()
        try:
            measurements.table = table
        except AttributeError:
            pass
        key = (id(measurements), len(measurements))
        self._external_measurement_table_cache = (key, measurements, table)
        return table

    def _measurement_table_for_indexed_plan(self, measurements: Sequence[Measurement]) -> MeasurementTable:
        key = (id(measurements), len(measurements))
        cache = getattr(self, "_external_measurement_table_cache", None)
        if cache is not None and cache[0] == key and cache[1] is measurements:
            return cache[2]
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return table
        return _measurement_table_from_measurements(measurements)

    def _ensure_measurement_plan_lookup_arrays(self) -> None:
        if not hasattr(self, "_ac_measurement_plan_pos_by_ppc_row"):
            self._build_measurement_plan_lookup_arrays()

    def _measurement_plan_table(
        self,
        measurements,
        meas_kind_by_type_code: Dict[int, Dict[str, int]],
        device_pos_by_type_code: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        self._ensure_measurement_plan_lookup_arrays()
        return build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={},
            device_pos_by_type_code_id={},
            meas_kind_by_type_code=meas_kind_by_type_code,
            meas_kind_code_by_type_code=self._measurement_kind_code_maps_for(meas_kind_by_type_code),
            require_index_arrays=True,
            table_builder=self._measurement_table_for_indexed_plan,
        )

    def _active_measurement_plan_tables(self, active_table: MeasurementTable) -> Dict[str, MeasurementPlanTable]:
        """Build all active measurement plan tables from one shared device-position pass."""
        self._ensure_measurement_plan_lookup_arrays()
        n_rows = int(active_table.idx.size)
        row = np.arange(n_rows, dtype=np.int64)
        device_type_code = np.asarray(active_table.device_type_code, dtype=np.int16)
        rows_by_code = rows_by_device_type_code(active_table)
        device_pos = self._measurement_device_pos_array(active_table)

        kind_maps = {
            "branch_transformer": self._ac_branch_transformer_plan_kind_by_type_code,
            "zero_current": self._ac_zero_current_plan_kind_by_type_code,
            "simple": self._ac_simple_plan_kind_by_type_code,
            "generator": self._ac_generator_plan_kind_by_type_code,
            "acac": self._ac_acac_plan_kind_by_type_code,
            "balance": self._ac_balance_plan_kind_by_type_code,
        }
        kind_arrays = {
            name: np.full(n_rows, -1, dtype=np.int16)
            for name in kind_maps
        }
        meas_type_code = getattr(active_table, "meas_type_code", None)
        if meas_type_code is not None:
            meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
            if meas_type_code.size != n_rows:
                meas_type_code = None
        if meas_type_code is None:
            warnings.warn(
                "AC SE active measurement plan requires meas_type_code; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        kind_code_maps = getattr(self, "_ac_measurement_plan_kind_codes", {})
        for code_int, code_rows in rows_by_code.items():
            rows = np.asarray(code_rows, dtype=np.int64)
            if rows.size == 0:
                continue
            for plan_name, _maps_by_code in kind_maps.items():
                if meas_type_code is not None:
                    code_lookup = kind_code_maps.get(plan_name, {}).get(int(code_int))
                    if code_lookup is not None and code_lookup.size:
                        codes = meas_type_code[rows].astype(np.int64, copy=False)
                        values = np.full(rows.size, -1, dtype=np.int16)
                        in_range = (codes >= 0) & (codes < code_lookup.size)
                        if np.any(in_range):
                            values[in_range] = code_lookup[codes[in_range].astype(np.intp, copy=False)]
                        kind_arrays[plan_name][rows] = values

        result: Dict[str, MeasurementPlanTable] = {}
        for plan_name, meas_kind in kind_arrays.items():
            result[plan_name] = MeasurementPlanTable(
                table=active_table,
                row=row,
                device_type_code=device_type_code,
                meas_kind=meas_kind,
                device_pos=device_pos,
                handled=(meas_kind >= 0) & (device_pos >= 0),
            )
        return result

    def _shrink_measurement_plan_tables(
        self,
        plan_tables: Dict[str, MeasurementPlanTable],
        removed_pos: int,
    ) -> Dict[str, MeasurementPlanTable]:
        base_plan = self._common_measurement_plan_table(plan_tables)
        n_rows = int(base_plan.row.size)
        pos = int(removed_pos)
        if pos < 0 or pos >= n_rows:
            raise IndexError("bad-data removal row position is out of range")
        keep_rows = np.concatenate(
            (
                np.arange(pos, dtype=np.int64),
                np.arange(pos + 1, n_rows, dtype=np.int64),
            )
        )
        shrunk_table = measurement_table_take(base_plan.table, keep_rows)
        row = np.arange(keep_rows.size, dtype=np.int64)
        shrunk: Dict[str, MeasurementPlanTable] = {}
        for name, plan in plan_tables.items():
            shrunk[name] = MeasurementPlanTable(
                table=shrunk_table,
                row=row,
                device_type_code=np.asarray(plan.device_type_code, dtype=np.int16)[keep_rows],
                meas_kind=np.asarray(plan.meas_kind, dtype=np.int16)[keep_rows],
                device_pos=np.asarray(plan.device_pos, dtype=np.int64)[keep_rows],
                handled=np.asarray(plan.handled, dtype=bool)[keep_rows],
            )
        return shrunk

    def _branch_transformer_vector_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, np.ndarray]:
        if measurements is None:
            active_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_branch_transformer_vector_plan_from_table(
                self._active_measurement_plan_table("branch_transformer")
            )
            self._active_branch_transformer_vector_plan = plan
            return plan
        key = id(measurements)
        cached = self._branch_transformer_vector_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]
        plan = self._build_branch_transformer_vector_plan(measurements)
        if len(self._branch_transformer_vector_plan_cache) > 16:
            self._branch_transformer_vector_plan_cache.clear()
        self._branch_transformer_vector_plan_cache[key] = (measurements, plan)
        return plan

    def _build_branch_transformer_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Precompute ACBranch/ACTransformer row metadata for repeated h/H calls."""
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_branch_transformer_plan_kind_by_type_code,
        )
        return self._build_branch_transformer_vector_plan_from_table(plan_table)

    def _build_branch_transformer_vector_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        handled_mask = np.asarray(plan_table.handled, dtype=bool).copy()
        if not np.any(handled_mask):
            empty_i = np.array([], dtype=np.int64)
            empty_b = np.array([], dtype=bool)
            empty_c = np.array([], dtype=np.complex128)
            return {
                "handled_mask": handled_mask,
                "voltage_rows": empty_i,
                "voltage_pos": empty_i,
                "voltage_cols": empty_i,
                "voltage_values": np.array([], dtype=np.float64),
                "power_rows": empty_i,
                "power_is_p": empty_b,
                "power_own": empty_i,
                "power_other": empty_i,
                "power_own_angle_cols": empty_i,
                "power_other_angle_cols": empty_i,
                "power_own_voltage_cols": empty_i,
                "power_other_voltage_cols": empty_i,
                "power_y_self": empty_c,
                "power_y_mutual": empty_c,
                "power_y_self_conj": empty_c,
                "power_y_mutual_conj": empty_c,
                "current_rows": empty_i,
                "current_own": empty_i,
                "current_other": empty_i,
                "current_own_angle_cols": empty_i,
                "current_other_angle_cols": empty_i,
                "current_own_voltage_cols": empty_i,
                "current_other_voltage_cols": empty_i,
                "current_y_self": empty_c,
                "current_y_mutual": empty_c,
            }

        def build_device_rows(device_code, device_pos, i_array, j_array, yff, yft, ytf, ytt):
            rows = row[(code == device_code) & handled_mask]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_from = row_kind == MEAS_TYPE_V_FROM
            v_to = row_kind == MEAS_TYPE_V_TO
            p_from = row_kind == MEAS_TYPE_P_FROM
            q_from = row_kind == MEAS_TYPE_Q_FROM
            p_to = row_kind == MEAS_TYPE_P_TO
            q_to = row_kind == MEAS_TYPE_Q_TO
            i_from = row_kind == MEAS_TYPE_I_FROM
            i_to = row_kind == MEAS_TYPE_I_TO
            from_power = p_from | q_from
            to_power = p_to | q_to
            return {
                "voltage_rows": self._concat_plan_arrays((rows[v_from], rows[v_to]), np.int64),
                "voltage_pos": self._concat_plan_arrays((i[v_from], j[v_to]), np.int64),
                "power_rows": self._concat_plan_arrays((rows[from_power], rows[to_power]), np.int64),
                "power_is_p": self._concat_plan_arrays((p_from[from_power], p_to[to_power]), bool),
                "power_own": self._concat_plan_arrays((i[from_power], j[to_power]), np.int64),
                "power_other": self._concat_plan_arrays((j[from_power], i[to_power]), np.int64),
                "power_y_self": self._concat_plan_arrays((yff[pos[from_power]], ytt[pos[to_power]]), np.complex128),
                "power_y_mutual": self._concat_plan_arrays((yft[pos[from_power]], ytf[pos[to_power]]), np.complex128),
                "current_rows": self._concat_plan_arrays((rows[i_from], rows[i_to]), np.int64),
                "current_own": self._concat_plan_arrays((i[i_from], j[i_to]), np.int64),
                "current_other": self._concat_plan_arrays((j[i_from], i[i_to]), np.int64),
                "current_y_self": self._concat_plan_arrays((yff[pos[i_from]], ytt[pos[i_to]]), np.complex128),
                "current_y_mutual": self._concat_plan_arrays((yft[pos[i_from]], ytf[pos[i_to]]), np.complex128),
            }

        branch_plan = build_device_rows(
            DEVICE_TYPE_ACBranch,
            plan_table.device_pos,
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
            self._ac_branch_plan_yff,
            self._ac_branch_plan_yft,
            self._ac_branch_plan_ytf,
            self._ac_branch_plan_ytt,
        )
        transformer_plan = build_device_rows(
            DEVICE_TYPE_ACTransformer,
            plan_table.device_pos,
            self._ac_transformer_plan_i,
            self._ac_transformer_plan_j,
            self._ac_transformer_plan_yff,
            self._ac_transformer_plan_yft,
            self._ac_transformer_plan_ytf,
            self._ac_transformer_plan_ytt,
        )
        voltage_rows = self._concat_plan_arrays((branch_plan["voltage_rows"], transformer_plan["voltage_rows"]), np.int64)
        voltage_pos = self._concat_plan_arrays((branch_plan["voltage_pos"], transformer_plan["voltage_pos"]), np.int64)
        power_rows = self._concat_plan_arrays((branch_plan["power_rows"], transformer_plan["power_rows"]), np.int64)
        power_is_p = self._concat_plan_arrays((branch_plan["power_is_p"], transformer_plan["power_is_p"]), bool)
        power_own = self._concat_plan_arrays((branch_plan["power_own"], transformer_plan["power_own"]), np.int64)
        power_other = self._concat_plan_arrays((branch_plan["power_other"], transformer_plan["power_other"]), np.int64)
        power_y_self = self._concat_plan_arrays((branch_plan["power_y_self"], transformer_plan["power_y_self"]), np.complex128)
        power_y_mutual = self._concat_plan_arrays(
            (branch_plan["power_y_mutual"], transformer_plan["power_y_mutual"]),
            np.complex128,
        )
        current_rows = self._concat_plan_arrays((branch_plan["current_rows"], transformer_plan["current_rows"]), np.int64)
        current_own = self._concat_plan_arrays((branch_plan["current_own"], transformer_plan["current_own"]), np.int64)
        current_other = self._concat_plan_arrays((branch_plan["current_other"], transformer_plan["current_other"]), np.int64)
        current_y_self = self._concat_plan_arrays(
            (branch_plan["current_y_self"], transformer_plan["current_y_self"]),
            np.complex128,
        )
        current_y_mutual = self._concat_plan_arrays(
            (branch_plan["current_y_mutual"], transformer_plan["current_y_mutual"]),
            np.complex128,
        )
        return {
            "handled_mask": handled_mask,
            "voltage_rows": voltage_rows,
            "voltage_pos": voltage_pos,
            "voltage_cols": self.voltage_col[voltage_pos].astype(np.int64, copy=False),
            "voltage_values": np.ones(voltage_rows.size, dtype=np.float64),
            "power_rows": power_rows,
            "power_is_p": power_is_p,
            "power_own": power_own,
            "power_other": power_other,
            "power_own_angle_cols": self.angle_col[power_own].astype(np.int64, copy=False),
            "power_other_angle_cols": self.angle_col[power_other].astype(np.int64, copy=False),
            "power_own_voltage_cols": self.voltage_col[power_own].astype(np.int64, copy=False),
            "power_other_voltage_cols": self.voltage_col[power_other].astype(np.int64, copy=False),
            "power_y_self": power_y_self,
            "power_y_mutual": power_y_mutual,
            "power_y_self_conj": np.conj(power_y_self),
            "power_y_mutual_conj": np.conj(power_y_mutual),
            "current_rows": current_rows,
            "current_own": current_own,
            "current_other": current_other,
            "current_own_angle_cols": self.angle_col[current_own].astype(np.int64, copy=False),
            "current_other_angle_cols": self.angle_col[current_other].astype(np.int64, copy=False),
            "current_own_voltage_cols": self.voltage_col[current_own].astype(np.int64, copy=False),
            "current_other_voltage_cols": self.voltage_col[current_other].astype(np.int64, copy=False),
            "current_y_self": current_y_self,
            "current_y_mutual": current_y_mutual,
        }

    def _fill_branch_transformer_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACBranch/ACTransformer terminal P/Q/V/I measurements."""
        voltage_rows = plan["voltage_rows"]
        if voltage_rows.size:
            values[voltage_rows] = voltage[plan["voltage_pos"]]

        rows = plan["power_rows"]
        if rows.size:
            own = plan["power_own"]
            other = plan["power_other"]
            y_self = plan["power_y_self"]
            y_mutual = plan["power_y_mutual"]
            power = voltage_complex[own] * np.conj(y_self * voltage_complex[own] + y_mutual * voltage_complex[other])
            values[rows] = np.where(plan["power_is_p"], power.real, power.imag)

        rows = plan["current_rows"]
        if rows.size:
            own = plan["current_own"]
            other = plan["current_other"]
            y_self = plan["current_y_self"]
            y_mutual = plan["current_y_mutual"]
            current = y_self * voltage_complex[own] + y_mutual * voltage_complex[other]
            values[rows] = np.abs(current)

        return plan["handled_mask"]

    def evaluate(
        self,
        x: np.ndarray,
        measurement_plan_tables=None,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        measurement_plan_tables = self._measurement_plan_tables_for(measurement_plan_tables)
        vector_plans = self._vector_plans_for_measurement_plan_tables(measurement_plan_tables)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        n_meas = self._measurement_count(measurement_plan_tables)
        active_vectorized = (
            self._measurement_plan_tables_are_active(measurement_plan_tables)
            and bool(getattr(self, "active_measurements_are_vectorized", False))
        )
        if out is None:
            values = np.empty(n_meas, dtype=np.float64) if active_vectorized else np.zeros(n_meas, dtype=np.float64)
        else:
            values = out
            if not active_vectorized:
                values.fill(0.0)
        vectorized_branch_rows = self._fill_branch_transformer_values_vectorized(
            values,
            vector_plans["branch_transformer"],
            voltage,
            voltage_complex,
        )
        vectorized_zero_rows = self._fill_zero_current_values_vectorized(
            values,
            vector_plans["zero_current"],
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_simple_rows = self._fill_simple_values_vectorized(
            values,
            vector_plans["simple"],
            x,
            theta,
            voltage,
        )
        vectorized_generator_rows = self._fill_generator_values_vectorized(
            values,
            vector_plans["generator"],
            x,
            voltage,
        )
        vectorized_acac_rows = self._fill_acac_values_vectorized(
            values,
            vector_plans["acac"],
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_values_vectorized(
            values,
            vector_plans["balance"],
            x,
            voltage_complex,
            switch_current,
        )
        if active_vectorized:
            return values
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_zero_rows
            | vectorized_simple_rows
            | vectorized_generator_rows
            | vectorized_acac_rows
            | vectorized_balance_rows
        )
        if vectorized_rows.all():
            return values
        unhandled = np.flatnonzero(~vectorized_rows)
        warnings.warn(
            f"Skipped {int(unhandled.size)} unhandled AC SE evaluate rows; string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return values

    def _zero_current_vector_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, np.ndarray]:
        if measurements is None:
            active_plan = getattr(self, "_active_zero_current_vector_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_zero_current_vector_plan_from_table(
                self._active_measurement_plan_table("zero_current")
            )
            self._active_zero_current_vector_plan = plan
            return plan
        key = id(measurements)
        cached = self._zero_current_vector_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]
        plan = self._build_zero_current_vector_plan(measurements)
        if len(self._zero_current_vector_plan_cache) > 16:
            self._zero_current_vector_plan_cache.clear()
        self._zero_current_vector_plan_cache[key] = (measurements, plan)
        return plan

    def _build_zero_current_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Precompute ACSwitch/ACZeroBranch row metadata for explicit-current states."""
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_zero_current_plan_kind_by_type_code,
        )
        return self._build_zero_current_vector_plan_from_table(plan_table)

    def _build_zero_current_vector_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        handled_mask = np.asarray(plan_table.handled, dtype=bool).copy()
        if not np.any(handled_mask):
            empty_i = np.array([], dtype=np.int64)
            empty_b = np.array([], dtype=bool)
            empty_f = np.array([], dtype=np.float64)
            return {
                "handled_mask": handled_mask,
                "scalar_rows": empty_i,
                "scalar_cols": empty_i,
                "scalar_values": empty_f,
                "voltage_rows": empty_i,
                "voltage_pos": empty_i,
                "angle_diff_rows": empty_i,
                "angle_diff_i": empty_i,
                "angle_diff_j": empty_i,
                "voltage_diff_rows": empty_i,
                "voltage_diff_i": empty_i,
                "voltage_diff_j": empty_i,
                "power_rows": empty_i,
                "power_is_p": empty_b,
                "power_pos": empty_i,
                "power_current_idx": empty_i,
                "power_sign": empty_f,
                "current_rows": empty_i,
                "current_idx": empty_i,
            }

        def build_device_rows(device_code, i_array, j_array, current_pos_array):
            rows = row[(code == device_code) & handled_mask]
            pos = plan_table.device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            current_pos = current_pos_array[pos]
            v_from = row_kind == MEAS_TYPE_V_FROM
            v_to = row_kind == MEAS_TYPE_V_TO
            p_from = row_kind == MEAS_TYPE_P_FROM
            q_from = row_kind == MEAS_TYPE_Q_FROM
            p_to = row_kind == MEAS_TYPE_P_TO
            q_to = row_kind == MEAS_TYPE_Q_TO
            i_from = row_kind == MEAS_TYPE_I_FROM
            i_to = row_kind == MEAS_TYPE_I_TO
            from_power = p_from | q_from
            to_power = p_to | q_to
            return {
                "voltage_rows": self._concat_plan_arrays((rows[v_from], rows[v_to]), np.int64),
                "voltage_pos": self._concat_plan_arrays((i[v_from], j[v_to]), np.int64),
                "power_rows": self._concat_plan_arrays((rows[from_power], rows[to_power]), np.int64),
                "power_is_p": self._concat_plan_arrays((p_from[from_power], p_to[to_power]), bool),
                "power_pos": self._concat_plan_arrays((i[from_power], j[to_power]), np.int64),
                "power_current_idx": self._concat_plan_arrays(
                    (current_pos[from_power], current_pos[to_power]),
                    np.int64,
                ),
                "power_sign": self._concat_plan_arrays(
                    (
                        np.ones(np.count_nonzero(from_power), dtype=np.float64),
                        -np.ones(np.count_nonzero(to_power), dtype=np.float64),
                    ),
                    np.float64,
                ),
                "current_rows": self._concat_plan_arrays((rows[i_from], rows[i_to]), np.int64),
                "current_idx": self._concat_plan_arrays((current_pos[i_from], current_pos[i_to]), np.int64),
                "v_diff_rows": rows[row_kind == MEAS_TYPE_V_DIFF],
                "angle_diff_rows": rows[
                    (row_kind == MEAS_TYPE_ANGLE_DIFF) | (row_kind == MEAS_TYPE_THETA_DIFF)
                ],
                "i": i,
                "j": j,
                "kind": row_kind,
            }

        zero_plan = build_device_rows(
            DEVICE_TYPE_ACZeroBranch,
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
            self._ac_zero_branch_plan_current_pos,
        )
        break_plan = build_device_rows(
            DEVICE_TYPE_ACBreak,
            self._ac_break_plan_i,
            self._ac_break_plan_j,
            self._ac_break_plan_current_pos,
        )
        zero_rows = row[(code == DEVICE_TYPE_ACZeroBranch) & handled_mask]
        zero_pos = plan_table.device_pos[zero_rows]
        zero_kind = kind[zero_rows]
        zero_i = self._ac_zero_branch_plan_i[zero_pos]
        zero_j = self._ac_zero_branch_plan_j[zero_pos]
        v_diff = zero_kind == MEAS_TYPE_V_DIFF
        angle_diff = (zero_kind == MEAS_TYPE_ANGLE_DIFF) | (zero_kind == MEAS_TYPE_THETA_DIFF)
        voltage_diff_rows = zero_rows[v_diff]
        voltage_diff_i = zero_i[v_diff]
        voltage_diff_j = zero_j[v_diff]
        angle_diff_rows = zero_rows[angle_diff]
        angle_diff_i = zero_i[angle_diff]
        angle_diff_j = zero_j[angle_diff]
        scalar_rows = self._concat_plan_arrays(
            (
                np.repeat(voltage_diff_rows, 2),
                np.repeat(angle_diff_rows, 2),
            ),
            np.int64,
        )
        scalar_cols = self._concat_plan_arrays(
            (
                np.column_stack((self.voltage_col[voltage_diff_i], self.voltage_col[voltage_diff_j])).ravel()
                if voltage_diff_rows.size
                else np.array([], dtype=np.int64),
                np.column_stack((self.angle_col[angle_diff_i], self.angle_col[angle_diff_j])).ravel()
                if angle_diff_rows.size
                else np.array([], dtype=np.int64),
            ),
            np.int64,
        )
        scalar_values = self._concat_plan_arrays(
            (
                np.tile(np.array([1.0, -1.0], dtype=np.float64), voltage_diff_rows.size),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), angle_diff_rows.size),
            ),
            np.float64,
        )
        return {
            "handled_mask": handled_mask,
            "scalar_rows": scalar_rows,
            "scalar_cols": scalar_cols,
            "scalar_values": scalar_values,
            "voltage_rows": self._concat_plan_arrays((zero_plan["voltage_rows"], break_plan["voltage_rows"]), np.int64),
            "voltage_pos": self._concat_plan_arrays((zero_plan["voltage_pos"], break_plan["voltage_pos"]), np.int64),
            "angle_diff_rows": self._int_array(angle_diff_rows),
            "angle_diff_i": self._int_array(angle_diff_i),
            "angle_diff_j": self._int_array(angle_diff_j),
            "voltage_diff_rows": self._int_array(voltage_diff_rows),
            "voltage_diff_i": self._int_array(voltage_diff_i),
            "voltage_diff_j": self._int_array(voltage_diff_j),
            "power_rows": self._concat_plan_arrays((zero_plan["power_rows"], break_plan["power_rows"]), np.int64),
            "power_is_p": self._concat_plan_arrays((zero_plan["power_is_p"], break_plan["power_is_p"]), bool),
            "power_pos": self._concat_plan_arrays((zero_plan["power_pos"], break_plan["power_pos"]), np.int64),
            "power_current_idx": self._concat_plan_arrays(
                (zero_plan["power_current_idx"], break_plan["power_current_idx"]),
                np.int64,
            ),
            "power_sign": self._concat_plan_arrays((zero_plan["power_sign"], break_plan["power_sign"]), np.float64),
            "current_rows": self._concat_plan_arrays((zero_plan["current_rows"], break_plan["current_rows"]), np.int64),
            "current_idx": self._concat_plan_arrays((zero_plan["current_idx"], break_plan["current_idx"]), np.int64),
        }

    def _fill_zero_current_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACSwitch/ACZeroBranch P/Q/V/I measurements."""
        rows = plan["voltage_rows"]
        if rows.size:
            values[rows] = voltage[plan["voltage_pos"]]

        rows = plan["voltage_diff_rows"]
        if rows.size:
            values[rows] = voltage[plan["voltage_diff_i"]] - voltage[plan["voltage_diff_j"]]

        rows = plan["angle_diff_rows"]
        if rows.size:
            values[rows] = theta[plan["angle_diff_i"]] - theta[plan["angle_diff_j"]]

        rows = plan["power_rows"]
        if rows.size:
            pos = plan["power_pos"]
            current = plan["power_sign"] * switch_current[plan["power_current_idx"]]
            power = voltage_complex[pos] * np.conj(current)
            values[rows] = np.where(plan["power_is_p"], power.real, power.imag)

        rows = plan["current_rows"]
        if rows.size:
            values[rows] = np.abs(switch_current[plan["current_idx"]])

        return plan["handled_mask"]

    def _fill_zero_current_jacobian_vectorized(
        self,
        H: np.ndarray,
        plan: Dict[str, np.ndarray],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACSwitch/ACZeroBranch Jacobian rows from explicit current states."""
        if plan["scalar_rows"].size:
            self._add_indexed_values(H, plan["scalar_rows"], plan["scalar_cols"], plan["scalar_values"])

        rows = plan["voltage_rows"]
        if rows.size:
            self._add_indexed_values(
                H,
                rows,
                self.voltage_col[plan["voltage_pos"]],
                np.ones(rows.size, dtype=np.float64),
            )

        rows = plan["power_rows"]
        if rows.size:
            pos = plan["power_pos"]
            current_idx = plan["power_current_idx"]
            sign = plan["power_sign"]
            current = switch_current[current_idx]
            power = voltage_complex[pos] * np.conj(sign * current)
            is_p = plan["power_is_p"]

            dtheta = 1j * power
            dvoltage = np.zeros(rows.size, dtype=np.complex128)
            valid_voltage = np.abs(voltage[pos]) > 1e-12
            dvoltage[valid_voltage] = power[valid_voltage] / voltage[pos][valid_voltage]
            dcurrent_re = sign * voltage_complex[pos]
            dcurrent_im = -1j * sign * voltage_complex[pos]

            self._add_indexed_values(H, rows, self.angle_col[pos], np.where(is_p, dtheta.real, dtheta.imag))
            self._add_indexed_values(H, rows, self.voltage_col[pos], np.where(is_p, dvoltage.real, dvoltage.imag))
            self._add_indexed_values(
                H,
                rows,
                self.base_switch_re + current_idx,
                np.where(is_p, dcurrent_re.real, dcurrent_re.imag),
            )
            self._add_indexed_values(
                H,
                rows,
                self.base_switch_im + current_idx,
                np.where(is_p, dcurrent_im.real, dcurrent_im.imag),
            )

        rows = plan["current_rows"]
        if rows.size:
            current = switch_current[plan["current_idx"]]
            current_abs = np.abs(current)
            valid_current = current_abs > 1e-12
            dcurrent_re = np.zeros(rows.size, dtype=np.float64)
            dcurrent_im = np.zeros(rows.size, dtype=np.float64)
            dcurrent_re[valid_current] = current.real[valid_current] / current_abs[valid_current]
            dcurrent_im[valid_current] = current.imag[valid_current] / current_abs[valid_current]
            current_idx = plan["current_idx"]
            self._add_indexed_values(H, rows, self.base_switch_re + current_idx, dcurrent_re)
            self._add_indexed_values(H, rows, self.base_switch_im + current_idx, dcurrent_im)

        return plan["handled_mask"]

    def _add_indexed_values(
        self,
        H: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        # SparseJacobianBuilder.add_many handles dtype coercion and cols >= 0
        # filtering. The rows>=0 case happens only for a handful of pseudo
        # rows, so we still gate on that explicitly to keep behaviour identical.
        if hasattr(H, "add_many"):
            if mask is None:
                row_mask = rows >= 0
                if row_mask.all():
                    H.add_many(rows, cols, values)
                else:
                    H.add_many(rows, cols, values, row_mask)
            else:
                row_mask = (rows >= 0) & np.asarray(mask, dtype=bool)
                H.add_many(rows, cols, values, row_mask)
            return
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = (rows >= 0) & (cols >= 0)
        else:
            mask = np.asarray(mask, dtype=bool) & (rows >= 0) & (cols >= 0)
        if np.any(mask):
            np.add.at(H, (rows[mask], cols[mask]), values[mask])

    def _add_indexed_value_blocks(self, H: np.ndarray, blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """Append several vectorized Jacobian write blocks with one sparse/dense scatter."""
        if not blocks:
            return
        if hasattr(H, "add_many"):
            for rows, cols, values in blocks:
                rows_array = np.asarray(rows)
                if not rows_array.size:
                    continue
                row_mask = rows_array >= 0
                if np.all(row_mask):
                    H.add_many(rows, cols, values)
                else:
                    H.add_many(rows, cols, values, row_mask)
            return
        row_parts = []
        col_parts = []
        value_parts = []
        for rows, cols, values in blocks:
            rows = np.asarray(rows)
            if rows.size == 0:
                continue
            row_parts.append(rows)
            col_parts.append(np.asarray(cols))
            value_parts.append(np.asarray(values))
        if not row_parts:
            return
        if len(row_parts) == 1:
            self._add_indexed_values(H, row_parts[0], col_parts[0], value_parts[0])
            return
        self._add_indexed_values(
            H,
            np.concatenate(row_parts),
            np.concatenate(col_parts),
            np.concatenate(value_parts),
        )

    @staticmethod
    def _complex_bincount(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
        if index.size == 0:
            return np.zeros(size, dtype=np.complex128)
        return (
            np.bincount(index, weights=values.real, minlength=size)
            + 1j * np.bincount(index, weights=values.imag, minlength=size)
        )

    def _fill_branch_transformer_jacobian_vectorized(
        self,
        H: np.ndarray,
        plan: Dict[str, np.ndarray],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACBranch/ACTransformer P/Q/V/I rows to reduce Python call overhead."""
        voltage_rows = plan["voltage_rows"]
        if voltage_rows.size:
            self._add_indexed_values(
                H,
                voltage_rows,
                plan["voltage_cols"],
                plan["voltage_values"],
            )

        rows = plan["power_rows"]
        if rows.size:
            own = plan["power_own"]
            other = plan["power_other"]
            y_self_conj = plan["power_y_self_conj"]
            y_mutual_conj = plan["power_y_mutual_conj"]
            off = y_mutual_conj * voltage_complex[own] * np.conj(voltage_complex[other])
            dtheta_own = 1j * off
            dtheta_other = -1j * off
            dvoltage_own = 2.0 * y_self_conj * voltage[own] + off / voltage[own]
            dvoltage_other = off / voltage[other]
            is_p = plan["power_is_p"]

            own_angle_cols = plan["power_own_angle_cols"]
            other_angle_cols = plan["power_other_angle_cols"]
            own_voltage_cols = plan["power_own_voltage_cols"]
            other_voltage_cols = plan["power_other_voltage_cols"]
            values = (
                np.where(is_p, dtheta_own.real, dtheta_own.imag),
                np.where(is_p, dtheta_other.real, dtheta_other.imag),
                np.where(is_p, dvoltage_own.real, dvoltage_own.imag),
                np.where(is_p, dvoltage_other.real, dvoltage_other.imag),
            )
            if hasattr(H, "add_many"):
                H.add_many(rows, own_angle_cols, values[0])
                H.add_many(rows, other_angle_cols, values[1])
                H.add_many(rows, own_voltage_cols, values[2])
                H.add_many(rows, other_voltage_cols, values[3])
            else:
                self._add_indexed_values(H, rows, own_angle_cols, values[0])
                self._add_indexed_values(H, rows, other_angle_cols, values[1])
                self._add_indexed_values(H, rows, own_voltage_cols, values[2])
                self._add_indexed_values(H, rows, other_voltage_cols, values[3])

        rows = plan["current_rows"]
        if rows.size:
            own = plan["current_own"]
            other = plan["current_other"]
            y_self = plan["current_y_self"]
            y_mutual = plan["current_y_mutual"]
            v_own = voltage_complex[own]
            v_other = voltage_complex[other]
            exp_own = v_own / voltage[own]
            exp_other = v_other / voltage[other]
            current = y_self * v_own + y_mutual * v_other
            current_abs = np.abs(current)
            valid = current_abs > 1e-12
            current_conj = np.conj(current)

            def project(value: np.ndarray) -> np.ndarray:
                out = np.zeros_like(current_abs, dtype=np.float64)
                out[valid] = (current_conj[valid] * value[valid]).real / current_abs[valid]
                return out

            dtheta_own = 1j * y_self * v_own
            dtheta_other = 1j * y_mutual * v_other
            dvoltage_own = y_self * exp_own
            dvoltage_other = y_mutual * exp_other
            own_angle_cols = plan["current_own_angle_cols"]
            other_angle_cols = plan["current_other_angle_cols"]
            own_voltage_cols = plan["current_own_voltage_cols"]
            other_voltage_cols = plan["current_other_voltage_cols"]
            values = (
                project(dtheta_own),
                project(dtheta_other),
                project(dvoltage_own),
                project(dvoltage_other),
            )
            if hasattr(H, "add_many"):
                H.add_many(rows, own_angle_cols, values[0])
                H.add_many(rows, other_angle_cols, values[1])
                H.add_many(rows, own_voltage_cols, values[2])
                H.add_many(rows, other_voltage_cols, values[3])
            else:
                self._add_indexed_values(H, rows, own_angle_cols, values[0])
                self._add_indexed_values(H, rows, other_angle_cols, values[1])
                self._add_indexed_values(H, rows, own_voltage_cols, values[2])
                self._add_indexed_values(H, rows, other_voltage_cols, values[3])

        return plan["handled_mask"]

    def _simple_jacobian_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, np.ndarray]:
        if measurements is None:
            active_plan = getattr(self, "_active_simple_jacobian_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_simple_jacobian_plan_from_table(
                self._active_measurement_plan_table("simple")
            )
            self._active_simple_jacobian_plan = plan
            return plan
        key = id(measurements)
        cached = self._simple_jacobian_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_simple_jacobian_plan(measurements)
        if len(self._simple_jacobian_plan_cache) > 16:
            self._simple_jacobian_plan_cache.clear()
        self._simple_jacobian_plan_cache[key] = (measurements, plan)
        return plan

    def _build_simple_jacobian_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_simple_plan_kind_by_type_code,
        )
        return self._build_simple_jacobian_plan_from_table(plan_table)

    def _build_simple_jacobian_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        row = plan_table.row
        code = plan_table.device_type_code
        kind = plan_table.meas_kind
        device_pos = plan_table.device_pos
        handled = np.asarray(plan_table.handled, dtype=bool).copy()
        if not np.any(handled):
            empty_i = np.array([], dtype=np.int64)
            empty_f = np.array([], dtype=np.float64)
            return {
                "handled_mask": handled,
                "scalar_rows": empty_i,
                "scalar_cols": empty_i,
                "scalar_values": empty_f,
                "value_voltage_rows": empty_i,
                "value_voltage_pos": empty_i,
                "value_angle_rows": empty_i,
                "value_angle_pos": empty_i,
                "value_voltage_diff_rows": empty_i,
                "value_voltage_diff_i": empty_i,
                "value_voltage_diff_j": empty_i,
                "value_angle_diff_rows": empty_i,
                "value_angle_diff_i": empty_i,
                "value_angle_diff_j": empty_i,
                "load_rows": empty_i,
                "load_pos": empty_i,
                "load_index": empty_i,
                "load_kind": empty_i,
            }

        node_rows = row[(code == DEVICE_TYPE_ACNode) & handled]
        node_pos = self._ac_node_plan_pos[device_pos[node_rows]]
        node_kind = kind[node_rows]
        node_v = node_kind == MEAS_TYPE_V
        node_angle = (node_kind == MEAS_TYPE_ANGLE) | (node_kind == MEAS_TYPE_THETA)

        gen_rows = row[(code == DEVICE_TYPE_ACGenerator) & handled]
        gen_pos = self._ac_generator_plan_node_pos[device_pos[gen_rows]]

        load_rows_all = row[(code == DEVICE_TYPE_ACLoad) & handled]
        load_plan_pos = device_pos[load_rows_all]
        load_node_pos = self._ac_load_plan_node_pos[load_plan_pos]
        load_kind_all = kind[load_rows_all]
        load_v = load_kind_all == MEAS_TYPE_V_LOAD
        load_state = load_kind_all != MEAS_TYPE_V_LOAD

        def constraint_rows_for(device_code, i_array, j_array):
            rows = row[(code == device_code) & handled]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_diff = row_kind == MEAS_TYPE_V_DIFF
            angle_diff = (row_kind == MEAS_TYPE_ANGLE_DIFF) | (row_kind == MEAS_TYPE_THETA_DIFF)
            return rows[v_diff], i[v_diff], j[v_diff], rows[angle_diff], i[angle_diff], j[angle_diff]

        zero_v_rows, zero_v_i, zero_v_j, zero_a_rows, zero_a_i, zero_a_j = constraint_rows_for(
            DEVICE_TYPE_ACZeroBranchConstraint,
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
        )
        break_v_rows, break_v_i, break_v_j, break_a_rows, break_a_i, break_a_j = constraint_rows_for(
            DEVICE_TYPE_ACBreakConstraint,
            self._ac_break_plan_i,
            self._ac_break_plan_j,
        )
        value_voltage_rows = self._concat_plan_arrays((node_rows[node_v], gen_rows, load_rows_all[load_v]), np.int64)
        value_voltage_pos = self._concat_plan_arrays((node_pos[node_v], gen_pos, load_node_pos[load_v]), np.int64)
        value_angle_rows = self._int_array(node_rows[node_angle])
        value_angle_pos = self._int_array(node_pos[node_angle])
        value_voltage_diff_rows = self._concat_plan_arrays((zero_v_rows, break_v_rows), np.int64)
        value_voltage_diff_i = self._concat_plan_arrays((zero_v_i, break_v_i), np.int64)
        value_voltage_diff_j = self._concat_plan_arrays((zero_v_j, break_v_j), np.int64)
        value_angle_diff_rows = self._concat_plan_arrays((zero_a_rows, break_a_rows), np.int64)
        value_angle_diff_i = self._concat_plan_arrays((zero_a_i, break_a_i), np.int64)
        value_angle_diff_j = self._concat_plan_arrays((zero_a_j, break_a_j), np.int64)

        scalar_rows = self._concat_plan_arrays(
            (
                value_voltage_rows,
                value_angle_rows,
                np.repeat(value_voltage_diff_rows, 2),
                np.repeat(value_angle_diff_rows, 2),
            ),
            np.int64,
        )
        scalar_cols = self._concat_plan_arrays(
            (
                self.voltage_col[value_voltage_pos],
                self.angle_col[value_angle_pos],
                np.column_stack((self.voltage_col[value_voltage_diff_i], self.voltage_col[value_voltage_diff_j])).ravel()
                if value_voltage_diff_rows.size
                else np.array([], dtype=np.int64),
                np.column_stack((self.angle_col[value_angle_diff_i], self.angle_col[value_angle_diff_j])).ravel()
                if value_angle_diff_rows.size
                else np.array([], dtype=np.int64),
            ),
            np.int64,
        )
        scalar_values = self._concat_plan_arrays(
            (
                np.ones(value_voltage_rows.size, dtype=np.float64),
                np.ones(value_angle_rows.size, dtype=np.float64),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), value_voltage_diff_rows.size),
                np.tile(np.array([1.0, -1.0], dtype=np.float64), value_angle_diff_rows.size),
            ),
            np.float64,
        )
        load_rows = load_rows_all[load_state]
        load_device_pos = load_plan_pos[load_state]
        return {
            "handled_mask": handled,
            "scalar_rows": scalar_rows,
            "scalar_cols": scalar_cols,
            "scalar_values": scalar_values,
            "value_voltage_rows": value_voltage_rows,
            "value_voltage_pos": value_voltage_pos,
            "value_angle_rows": value_angle_rows,
            "value_angle_pos": value_angle_pos,
            "value_voltage_diff_rows": value_voltage_diff_rows,
            "value_voltage_diff_i": value_voltage_diff_i,
            "value_voltage_diff_j": value_voltage_diff_j,
            "value_angle_diff_rows": value_angle_diff_rows,
            "value_angle_diff_i": value_angle_diff_i,
            "value_angle_diff_j": value_angle_diff_j,
            "load_rows": self._int_array(load_rows),
            "load_pos": self._ac_load_plan_node_pos[load_device_pos],
            "load_index": self._ac_load_plan_index[load_device_pos],
            "load_kind": self._int_array(load_kind_all[load_state]),
        }

    def _fill_simple_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate node, load, generator-voltage and pseudo-constraint rows."""
        rows = plan["value_voltage_rows"]
        if rows.size:
            values[rows] = voltage[plan["value_voltage_pos"]]

        rows = plan["value_angle_rows"]
        if rows.size:
            values[rows] = theta[plan["value_angle_pos"]]

        rows = plan["value_voltage_diff_rows"]
        if rows.size:
            values[rows] = voltage[plan["value_voltage_diff_i"]] - voltage[plan["value_voltage_diff_j"]]

        rows = plan["value_angle_diff_rows"]
        if rows.size:
            values[rows] = theta[plan["value_angle_diff_i"]] - theta[plan["value_angle_diff_j"]]

        rows = plan["load_rows"]
        if rows.size:
            pos = plan["load_pos"]
            load_idx = plan["load_index"]
            vm = voltage[pos]
            kind = plan["load_kind"]
            p = np.asarray(x[self.base_load_p + load_idx], dtype=np.float64)
            q = np.asarray(x[self.base_load_q + load_idx], dtype=np.float64)
            row_values = np.zeros(rows.size, dtype=np.float64)
            p_mask = kind == MEAS_TYPE_P_LOAD
            q_mask = kind == MEAS_TYPE_Q_LOAD
            i_mask = kind == MEAS_TYPE_I_LOAD
            row_values[p_mask] = p[p_mask]
            row_values[q_mask] = q[q_mask]
            if np.any(i_mask):
                i_idx = np.flatnonzero(i_mask)
                vm_i = vm[i_idx]
                valid = np.abs(vm_i) > self.min_current_voltage
                s_abs = np.hypot(p[i_idx], q[i_idx])
                row_values[i_idx[valid]] = s_abs[valid] / vm_i[valid]
            values[rows] = row_values

        return plan["handled_mask"]

    def _fill_simple_jacobian_vectorized(
        self,
        H: np.ndarray,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill node/load/generator-voltage and pseudo constraint Jacobian rows."""
        if plan["scalar_rows"].size:
            self._add_indexed_values(H, plan["scalar_rows"], plan["scalar_cols"], plan["scalar_values"])

        rows = plan["load_rows"]
        if rows.size:
            pos = plan["load_pos"]
            load_idx = plan["load_index"]
            vm = voltage[pos]
            kind = plan["load_kind"]
            p = np.asarray(x[self.base_load_p + load_idx], dtype=np.float64)
            q = np.asarray(x[self.base_load_q + load_idx], dtype=np.float64)
            p_mask = kind == MEAS_TYPE_P_LOAD
            q_mask = kind == MEAS_TYPE_Q_LOAD
            i_mask = kind == MEAS_TYPE_I_LOAD
            if np.any(p_mask):
                self._add_indexed_values(
                    H,
                    rows[p_mask],
                    self.base_load_p + load_idx[p_mask],
                    np.ones(np.count_nonzero(p_mask), dtype=np.float64),
                )
            if np.any(q_mask):
                self._add_indexed_values(
                    H,
                    rows[q_mask],
                    self.base_load_q + load_idx[q_mask],
                    np.ones(np.count_nonzero(q_mask), dtype=np.float64),
                )
            if np.any(i_mask):
                i_rows = rows[i_mask]
                i_idx = load_idx[i_mask]
                p_i = p[i_mask]
                q_i = q[i_mask]
                vm_i = vm[i_mask]
                s_abs = np.hypot(p_i, q_i)
                valid = (s_abs > 1e-12) & (np.abs(vm_i) > self.min_current_voltage)
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.base_load_p + i_idx,
                    np.divide(p_i, s_abs * vm_i, out=np.zeros_like(p_i), where=valid),
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.base_load_q + i_idx,
                    np.divide(q_i, s_abs * vm_i, out=np.zeros_like(q_i), where=valid),
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.voltage_col[pos[i_mask]],
                    np.divide(-s_abs, vm_i * vm_i, out=np.zeros_like(s_abs), where=valid),
                )

        return plan["handled_mask"]

    def _balance_measurement_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, np.ndarray]:
        if measurements is None:
            active_plan = getattr(self, "_active_balance_measurement_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_balance_measurement_plan_from_table(
                self._active_measurement_plan_table("balance")
            )
            self._active_balance_measurement_plan = plan
            return plan
        key = id(measurements)
        cached = self._balance_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_balance_measurement_plan(measurements)
        if len(self._balance_measurement_plan_cache) > 16:
            self._balance_measurement_plan_cache.clear()
        self._balance_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _balance_y_arrays(self, pos: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_array = np.asarray(pos, dtype=np.int64)
        if pos_array.size == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.complex128),
            )
        if issparse(getattr(self, "Y", None)):
            y_csr = self.Y.tocsr()
            indptr = y_csr.indptr
            if (
                pos_array[0] >= 0
                and pos_array[-1] + 1 < indptr.size
                and (pos_array.size == 1 or np.all(pos_array[1:] == pos_array[:-1] + 1))
            ):
                first = int(pos_array[0])
                last = int(pos_array[-1])
                row_indptr = indptr[first : last + 2]
                counts = np.diff(row_indptr).astype(np.int64, copy=False)
                total = int(counts.sum())
                if total == 0:
                    return (
                        np.array([], dtype=np.int32),
                        np.array([], dtype=np.int32),
                        np.array([], dtype=np.int64),
                        np.array([], dtype=np.complex128),
                    )
                start = int(row_indptr[0])
                end = int(row_indptr[-1])
                y_balance = np.repeat(np.arange(pos_array.size, dtype=np.int32), counts)
                y_nodes = y_csr.indices[start:end].astype(np.int32, copy=False)
                return (
                    y_balance,
                    y_nodes,
                    y_nodes.astype(np.int64, copy=False),
                    np.conj(y_csr.data[start:end]).astype(np.complex128, copy=False),
                )
        counts = np.fromiter(
            (int(self._y_row_nodes[int(node_pos)].size) for node_pos in pos_array),
            dtype=np.int64,
            count=pos_array.size,
        )
        total = int(counts.sum())
        if total == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.complex128),
            )
        y_balance = np.repeat(np.arange(pos_array.size, dtype=np.int32), counts)
        y_nodes = np.empty(total, dtype=np.int32)
        y_conj = np.empty(total, dtype=np.complex128)
        offset = 0
        for node_pos, count in zip(pos_array, counts):
            count = int(count)
            if count == 0:
                continue
            end = offset + count
            y_nodes[offset:end] = self._y_row_nodes[int(node_pos)].astype(np.int32, copy=False)
            y_conj[offset:end] = self._y_row_y_conj[int(node_pos)].astype(np.complex128, copy=False)
            offset = end
        return y_balance, y_nodes, y_nodes.astype(np.int64, copy=False), y_conj

    def _build_balance_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        """Build the balance plan from a measurement sequence or table-backed active view."""
        self._ensure_measurement_plan_lookup_arrays()
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            getattr(
                self,
                "_ac_balance_plan_kind_by_type_code",
                {DEVICE_TYPE_ACPowerBalance: _AC_BALANCE_MEAS_TYPE_LOOKUP},
            ),
        )
        plan = self._build_balance_measurement_plan_from_table(plan_table)
        if measurements is None:
            self._active_balance_measurement_plan = plan
        return plan

    def _build_balance_measurement_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        """Build the balance plan from a precomputed measurement plan table."""
        rows = plan_table.row[
            (plan_table.device_type_code == DEVICE_TYPE_ACPowerBalance) & plan_table.handled
        ]
        if rows.size == 0:
            empty_i = np.array([], dtype=np.int64)
            empty_i32 = np.array([], dtype=np.int32)
            empty_c = np.array([], dtype=np.complex128)
            p_row_by_pos = np.empty(len(getattr(self, "nodes", ())), dtype=np.int32)
            q_row_by_pos = np.empty(len(getattr(self, "nodes", ())), dtype=np.int32)
            p_row_by_pos.fill(-1)
            q_row_by_pos.fill(-1)
            return {
                "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
                "rows": empty_i,
                "pos": empty_i,
                "pos_i64": empty_i,
                "kind": empty_i,
                "p_row_by_pos": p_row_by_pos,
                "q_row_by_pos": q_row_by_pos,
                "balance_pos": empty_i,
                "balance_pos_i64": empty_i,
                "y_balance": empty_i32,
                "y_nodes": empty_i32,
                "y_nodes_i64": empty_i,
                "y_conj": empty_c,
                "y_pos": empty_i,
                "p_rows_for_balance": empty_i32,
                "q_rows_for_balance": empty_i32,
                "balance_angle_cols": empty_i,
                "balance_voltage_cols": empty_i,
                "off_mask": np.array([], dtype=bool),
                "off_balance": empty_i32,
                "off_nodes_i64": empty_i,
                "off_y_pos": empty_i,
                "off_y_conj": empty_c,
                "off_theta_cols": empty_i,
                "off_voltage_cols": empty_i,
                "off_theta_p_rows": empty_i32,
                "off_theta_q_rows": empty_i32,
                "switch_p_rows": empty_i32,
                "switch_q_rows": empty_i32,
                "switch_angle_cols": empty_i,
                "switch_voltage_cols": empty_i,
                "switch_re_cols": empty_i,
                "switch_im_cols": empty_i,
                "gen_p_rows": empty_i32,
                "gen_q_rows": empty_i32,
                "gen_p_cols": empty_i,
                "gen_q_cols": empty_i,
                "load_p_rows": empty_i32,
                "load_q_rows": empty_i32,
                "load_p_cols": empty_i,
                "load_q_cols": empty_i,
                "shunt_q_rows": empty_i32,
                "shunt_q_cols": empty_i,
                "acac_p_from_rows": empty_i32,
                "acac_q_from_rows": empty_i32,
                "acac_p_to_rows": empty_i32,
                "acac_q_to_rows": empty_i32,
                "acac_p_from_cols": empty_i,
                "acac_q_from_cols": empty_i,
                "acac_p_to_cols": empty_i,
                "acac_q_to_cols": empty_i,
            }
        device_pos = plan_table.device_pos[rows]
        pos = self._ac_node_plan_pos[device_pos]
        kind = plan_table.meas_kind[rows]
        p_row_by_pos = np.empty(self.n_nodes, dtype=np.int32)
        q_row_by_pos = np.empty(self.n_nodes, dtype=np.int32)
        p_row_by_pos.fill(-1)
        q_row_by_pos.fill(-1)
        if rows.size:
            p_mask = kind == MEAS_TYPE_P_BALANCE
            q_mask = kind == MEAS_TYPE_Q_BALANCE
            p_row_by_pos[pos[p_mask]] = rows[p_mask].astype(np.int32, copy=False)
            q_row_by_pos[pos[q_mask]] = rows[q_mask].astype(np.int32, copy=False)

        balance_pos = np.flatnonzero((p_row_by_pos >= 0) | (q_row_by_pos >= 0)).astype(np.int64, copy=False)
        y_balance, y_nodes, y_nodes_i64, y_conj = self._balance_y_arrays(balance_pos)
        p_rows_for_balance = p_row_by_pos[balance_pos] if balance_pos.size else np.asarray([], dtype=np.int32)
        q_rows_for_balance = q_row_by_pos[balance_pos] if balance_pos.size else np.asarray([], dtype=np.int32)
        y_pos = balance_pos[y_balance] if y_balance.size else np.asarray([], dtype=np.int64)
        off_mask = y_nodes_i64 != y_pos if y_balance.size else np.asarray([], dtype=bool)
        off_balance = y_balance[off_mask] if off_mask.size else np.asarray([], dtype=np.int32)
        off_nodes_i64 = y_nodes_i64[off_mask] if off_mask.size else np.asarray([], dtype=np.int64)
        off_y_pos = y_pos[off_mask] if off_mask.size else np.asarray([], dtype=np.int64)
        off_y_conj = y_conj[off_mask] if off_mask.size else np.asarray([], dtype=np.complex128)
        angle_col = np.asarray(
            getattr(self, "angle_col", np.arange(int(getattr(self, "n_nodes", 0)), dtype=np.int64)),
            dtype=np.int64,
        )
        voltage_col = np.asarray(
            getattr(self, "voltage_col", np.arange(int(getattr(self, "n_nodes", 0)), dtype=np.int64)),
            dtype=np.int64,
        )
        def state_cols_for(cols: np.ndarray, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions, dtype=np.int64)
            out = np.empty(positions.size, dtype=np.int64)
            out.fill(-1)
            valid = (positions >= 0) & (positions < int(cols.size))
            if np.any(valid):
                out[valid] = cols[positions[valid].astype(np.intp, copy=False)]
            return out

        if int(getattr(self, "n_switch_current", 0)):
            switch_end_pos = self.switch_balance_end_pos
            switch_end_current_idx = self.switch_balance_end_current_idx
            switch_p_rows = p_row_by_pos[switch_end_pos]
            switch_q_rows = q_row_by_pos[switch_end_pos]
            switch_angle_cols = angle_col[switch_end_pos].astype(np.int64, copy=False)
            switch_voltage_cols = voltage_col[switch_end_pos].astype(np.int64, copy=False)
            switch_re_cols = (self.base_switch_re + switch_end_current_idx).astype(np.int64, copy=False)
            switch_im_cols = (self.base_switch_im + switch_end_current_idx).astype(np.int64, copy=False)
        else:
            switch_p_rows = switch_q_rows = np.asarray([], dtype=np.int32)
            switch_angle_cols = switch_voltage_cols = switch_re_cols = switch_im_cols = np.asarray([], dtype=np.int64)

        if int(getattr(self, "n_generator_power", 0)):
            gen_idx = self.generator_power_idx_array
            gen_node_pos = np.asarray(self._ac_generator_power_node_pos, dtype=np.int64)
            gen_p_rows = p_row_by_pos[gen_node_pos]
            gen_q_rows = q_row_by_pos[gen_node_pos]
            gen_p_cols = (self.base_gen_p + gen_idx).astype(np.int64, copy=False)
            gen_q_cols = (self.base_gen_q + gen_idx).astype(np.int64, copy=False)
        else:
            gen_p_rows = gen_q_rows = np.asarray([], dtype=np.int32)
            gen_p_cols = gen_q_cols = np.asarray([], dtype=np.int64)

        if int(getattr(self, "n_load_power", 0)):
            load_idx = self.load_power_idx_array
            load_node_pos = np.asarray(self._ac_load_power_node_pos, dtype=np.int64)
            load_p_rows = p_row_by_pos[load_node_pos]
            load_q_rows = q_row_by_pos[load_node_pos]
            load_p_cols = (self.base_load_p + load_idx).astype(np.int64, copy=False)
            load_q_cols = (self.base_load_q + load_idx).astype(np.int64, copy=False)
        else:
            load_p_rows = load_q_rows = np.asarray([], dtype=np.int32)
            load_p_cols = load_q_cols = np.asarray([], dtype=np.int64)

        if int(getattr(self, "n_shunt_q", 0)):
            shunt_q_rows = q_row_by_pos[self.shunt_q_pos_array]
            shunt_q_cols = (self.base_shunt_q + self.shunt_q_idx_array).astype(np.int64, copy=False)
        else:
            shunt_q_rows = np.asarray([], dtype=np.int32)
            shunt_q_cols = np.asarray([], dtype=np.int64)

        if int(getattr(self, "n_acac_power", 0)):
            acac_idx = self.acac_power_idx_array
            acac_i_pos = np.asarray(self._ac_acac_power_i_pos, dtype=np.int64)
            acac_j_pos = np.asarray(self._ac_acac_power_j_pos, dtype=np.int64)
            acac_p_from_rows = p_row_by_pos[acac_i_pos]
            acac_q_from_rows = q_row_by_pos[acac_i_pos]
            acac_p_to_rows = p_row_by_pos[acac_j_pos]
            acac_q_to_rows = q_row_by_pos[acac_j_pos]
            acac_p_from_cols = (self.base_acac_p_from + acac_idx).astype(np.int64, copy=False)
            acac_q_from_cols = (self.base_acac_q_from + acac_idx).astype(np.int64, copy=False)
            acac_p_to_cols = (self.base_acac_p_to + acac_idx).astype(np.int64, copy=False)
            acac_q_to_cols = (self.base_acac_q_to + acac_idx).astype(np.int64, copy=False)
        else:
            acac_p_from_rows = acac_q_from_rows = acac_p_to_rows = acac_q_to_rows = np.asarray([], dtype=np.int32)
            acac_p_from_cols = acac_q_from_cols = acac_p_to_cols = acac_q_to_cols = np.asarray([], dtype=np.int64)

        plan = {
            "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
            "rows": self._int_array(rows),
            "pos": self._int_array(pos),
            "pos_i64": np.asarray(pos, dtype=np.int64),
            "kind": self._int_array(kind),
            "p_row_by_pos": p_row_by_pos,
            "q_row_by_pos": q_row_by_pos,
            "balance_pos": balance_pos,
            "balance_pos_i64": np.asarray(balance_pos, dtype=np.int64),
            "y_balance": y_balance,
            "y_nodes": y_nodes,
            "y_nodes_i64": y_nodes_i64,
            "y_conj": y_conj,
            "y_pos": y_pos,
            "p_rows_for_balance": p_rows_for_balance,
            "q_rows_for_balance": q_rows_for_balance,
            "balance_angle_cols": state_cols_for(angle_col, balance_pos),
            "balance_voltage_cols": state_cols_for(voltage_col, balance_pos),
            "off_mask": off_mask,
            "off_balance": off_balance,
            "off_nodes_i64": off_nodes_i64,
            "off_y_pos": off_y_pos,
            "off_y_conj": off_y_conj,
            "off_theta_cols": state_cols_for(angle_col, off_nodes_i64),
            "off_voltage_cols": state_cols_for(voltage_col, off_nodes_i64),
            "off_theta_p_rows": p_rows_for_balance[off_balance] if off_balance.size else np.asarray([], dtype=np.int32),
            "off_theta_q_rows": q_rows_for_balance[off_balance] if off_balance.size else np.asarray([], dtype=np.int32),
            "switch_p_rows": switch_p_rows,
            "switch_q_rows": switch_q_rows,
            "switch_angle_cols": switch_angle_cols,
            "switch_voltage_cols": switch_voltage_cols,
            "switch_re_cols": switch_re_cols,
            "switch_im_cols": switch_im_cols,
            "gen_p_rows": gen_p_rows,
            "gen_q_rows": gen_q_rows,
            "gen_p_cols": gen_p_cols,
            "gen_q_cols": gen_q_cols,
            "load_p_rows": load_p_rows,
            "load_q_rows": load_q_rows,
            "load_p_cols": load_p_cols,
            "load_q_cols": load_q_cols,
            "shunt_q_rows": shunt_q_rows,
            "shunt_q_cols": shunt_q_cols,
            "acac_p_from_rows": acac_p_from_rows,
            "acac_q_from_rows": acac_q_from_rows,
            "acac_p_to_rows": acac_p_to_rows,
            "acac_q_to_rows": acac_q_to_rows,
            "acac_p_from_cols": acac_p_from_cols,
            "acac_q_from_cols": acac_q_from_cols,
            "acac_p_to_cols": acac_p_to_cols,
            "acac_q_to_cols": acac_q_to_cols,
        }
        return plan

    def _fill_balance_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate nodal P/Q power-balance mismatch rows."""
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]
        p_balance, q_balance = self._power_balance_totals(x, voltage_complex, switch_current)
        pos = plan["pos"]
        values[rows] = np.where(plan["kind"] == MEAS_TYPE_P_BALANCE, p_balance[pos], q_balance[pos])
        return plan["handled_mask"]

    def _fill_balance_jacobian_vectorized(
        self,
        H: np.ndarray,
        plan: Dict[str, np.ndarray],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse nodal balance derivatives for network, switch and P/Q states."""
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        pos = plan["pos_i64"]
        balance_pos = np.asarray(plan.get("balance_pos_i64", pos), dtype=np.int64)
        p_rows_for_balance = plan["p_rows_for_balance"]
        q_rows_for_balance = plan["q_rows_for_balance"]

        y_balance = plan["y_balance"]
        if y_balance.size:
            y_nodes = plan["y_nodes_i64"]
            y_conj = plan["y_conj"]
            y_pos = plan["y_pos"]
            exp_delta = np.exp(1j * (theta[y_pos] - theta[y_nodes]))
            y_voltage_term = y_conj * voltage[y_nodes] * exp_delta
            term = y_voltage_term * voltage[y_pos]

            off_sum = np.zeros(balance_pos.size, dtype=np.complex128)
            off_mask = plan["off_mask"]
            if off_mask.size and np.any(off_mask):
                off_balance = plan["off_balance"]
                off_term = term[off_mask]
                off_sum = self._complex_bincount(off_balance, off_term, balance_pos.size)

                theta_values = -1j * off_term
                theta_cols = plan["off_theta_cols"]
                theta_p_rows = plan["off_theta_p_rows"]
                theta_q_rows = plan["off_theta_q_rows"]

                voltage_values = plan["off_y_conj"] * voltage[plan["off_y_pos"]] * exp_delta[off_mask]
                voltage_cols = plan["off_voltage_cols"]
                self._add_indexed_value_blocks(
                    H,
                    (
                        (
                            theta_p_rows,
                            theta_cols,
                            theta_values.real,
                        ),
                        (
                            theta_q_rows,
                            theta_cols,
                            theta_values.imag,
                        ),
                        (
                            theta_p_rows,
                            voltage_cols,
                            voltage_values.real,
                        ),
                        (
                            theta_q_rows,
                            voltage_cols,
                            voltage_values.imag,
                        ),
                    ),
                )

            sum_all = self._complex_bincount(y_balance, y_voltage_term, balance_pos.size)
        else:
            off_sum = np.zeros(balance_pos.size, dtype=np.complex128)
            sum_all = np.zeros(balance_pos.size, dtype=np.complex128)

        own_theta_values = 1j * off_sum
        own_voltage_values = self._y_row_diag_conj[balance_pos] * voltage[balance_pos] + sum_all
        self._add_indexed_value_blocks(
            H,
            (
                (
                    p_rows_for_balance,
                    plan["balance_angle_cols"],
                    own_theta_values.real,
                ),
                (
                    q_rows_for_balance,
                    plan["balance_angle_cols"],
                    own_theta_values.imag,
                ),
                (
                    p_rows_for_balance,
                    plan["balance_voltage_cols"],
                    own_voltage_values.real,
                ),
                (
                    q_rows_for_balance,
                    plan["balance_voltage_cols"],
                    own_voltage_values.imag,
                ),
            ),
        )

        if int(getattr(self, "n_switch_current", 0)):
            end_pos = self.switch_balance_end_pos
            end_current_idx = self.switch_balance_end_current_idx
            sign = self.switch_balance_sign
            current = switch_current[end_current_idx] * sign
            s = voltage_complex[end_pos] * np.conj(current)
            dS_dtheta = 1j * s
            dS_dV = np.divide(s, voltage[end_pos], out=np.zeros_like(s), where=np.abs(voltage[end_pos]) > 1e-12)
            dS_dIr = sign * voltage_complex[end_pos]
            dS_dIi = -1j * sign * voltage_complex[end_pos]
            p_rows = plan["switch_p_rows"]
            q_rows = plan["switch_q_rows"]

            self._add_indexed_value_blocks(
                H,
                (
                    (p_rows, plan["switch_angle_cols"], dS_dtheta.real),
                    (q_rows, plan["switch_angle_cols"], dS_dtheta.imag),
                    (p_rows, plan["switch_voltage_cols"], dS_dV.real),
                    (q_rows, plan["switch_voltage_cols"], dS_dV.imag),
                    (p_rows, plan["switch_re_cols"], dS_dIr.real),
                    (q_rows, plan["switch_re_cols"], dS_dIr.imag),
                    (p_rows, plan["switch_im_cols"], dS_dIi.real),
                    (q_rows, plan["switch_im_cols"], dS_dIi.imag),
                ),
            )

        if int(getattr(self, "n_generator_power", 0)):
            self._add_indexed_value_blocks(
                H,
                (
                    (plan["gen_p_rows"], plan["gen_p_cols"], self.generator_balance_minus_ones),
                    (plan["gen_q_rows"], plan["gen_q_cols"], self.generator_balance_minus_ones),
                ),
            )

        if int(getattr(self, "n_load_power", 0)):
            self._add_indexed_value_blocks(
                H,
                (
                    (plan["load_p_rows"], plan["load_p_cols"], self.load_balance_ones),
                    (plan["load_q_rows"], plan["load_q_cols"], self.load_balance_ones),
                ),
            )

        if int(getattr(self, "n_shunt_q", 0)):
            self._add_indexed_value_blocks(
                H,
                (
                    (
                        plan["shunt_q_rows"],
                        plan["shunt_q_cols"],
                        self.shunt_balance_minus_ones,
                    ),
                ),
            )

        if int(getattr(self, "n_acac_power", 0)):
            self._add_indexed_value_blocks(
                H,
                (
                    (plan["acac_p_from_rows"], plan["acac_p_from_cols"], self.acac_balance_ones),
                    (plan["acac_q_from_rows"], plan["acac_q_from_cols"], self.acac_balance_ones),
                    (plan["acac_p_to_rows"], plan["acac_p_to_cols"], self.acac_balance_ones),
                    (plan["acac_q_to_rows"], plan["acac_q_to_cols"], self.acac_balance_ones),
                ),
            )

        return plan["handled_mask"]

    def _generator_measurement_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, object]:
        if measurements is None:
            active_plan = getattr(self, "_active_generator_measurement_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_generator_measurement_plan_from_table(
                self._active_measurement_plan_table("generator")
            )
            self._active_generator_measurement_plan = plan
            return plan
        key = id(measurements)
        cached = self._generator_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        plan = self._build_generator_measurement_plan(measurements)
        if len(self._generator_measurement_plan_cache) > 16:
            self._generator_measurement_plan_cache.clear()
        self._generator_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _build_generator_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, object]:
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_generator_plan_kind_by_type_code,
        )
        return self._build_generator_measurement_plan_from_table(plan_table)

    def _build_generator_measurement_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, object]:
        rows = plan_table.row[
            (plan_table.device_type_code == DEVICE_TYPE_ACGenerator) & plan_table.handled
        ]
        if rows.size == 0:
            empty_i = np.array([], dtype=np.int64)
            return {
                "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
                "value_rows": empty_i,
                "value_kind": empty_i,
                "value_pos": empty_i,
                "value_index": empty_i,
            }
        device_pos = plan_table.device_pos[rows]
        return {
            "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
            "value_rows": self._int_array(rows),
            "value_kind": self._int_array(plan_table.meas_kind[rows]),
            "value_pos": self._ac_generator_plan_node_pos[device_pos],
            "value_index": self._ac_generator_plan_index[device_pos],
        }

    def _fill_generator_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACGenerator P/Q/I measurements from explicit P/Q states."""
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        kind = plan["value_kind"]
        row_values = np.zeros(rows.size, dtype=np.float64)
        p_mask = kind == MEAS_TYPE_P_GEN
        q_mask = kind == MEAS_TYPE_Q_GEN
        i_mask = kind == MEAS_TYPE_I_GEN
        row_values[p_mask] = p[p_mask]
        row_values[q_mask] = q[q_mask]
        if np.any(i_mask):
            vm = voltage[pos[i_mask]]
            valid = np.abs(vm) > self.min_current_voltage
            current_values = np.zeros(np.count_nonzero(i_mask), dtype=np.float64)
            current_values[valid] = np.hypot(p[i_mask][valid], q[i_mask][valid]) / vm[valid]
            row_values[i_mask] = current_values
        values[rows] = row_values
        return plan["handled_mask"]

    def _fill_generator_jacobian_sparse(
        self,
        H,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse generator P/Q/I rows from explicit P/Q states."""
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        kind = plan["value_kind"]
        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        p_mask = kind == MEAS_TYPE_P_GEN
        q_mask = kind == MEAS_TYPE_Q_GEN
        i_mask = kind == MEAS_TYPE_I_GEN
        if np.any(p_mask):
            self._add_indexed_values(
                H,
                rows[p_mask],
                self.base_gen_p + gen_idx[p_mask],
                np.ones(np.count_nonzero(p_mask), dtype=np.float64),
            )
        if np.any(q_mask):
            self._add_indexed_values(
                H,
                rows[q_mask],
                self.base_gen_q + gen_idx[q_mask],
                np.ones(np.count_nonzero(q_mask), dtype=np.float64),
            )
        if np.any(i_mask):
            i_rows = rows[i_mask]
            i_idx = gen_idx[i_mask]
            vm = voltage[pos[i_mask]]
            p_i = p[i_mask]
            q_i = q[i_mask]
            s_abs = np.hypot(p_i, q_i)
            valid = (np.abs(vm) > self.min_current_voltage) & (s_abs > 1e-12)
            self._add_indexed_values(
                H,
                i_rows,
                self.base_gen_p + i_idx,
                np.divide(p_i, s_abs * vm, out=np.zeros_like(p_i), where=valid),
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.base_gen_q + i_idx,
                np.divide(q_i, s_abs * vm, out=np.zeros_like(q_i), where=valid),
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.voltage_col[pos[i_mask]],
                np.divide(-s_abs, vm * vm, out=np.zeros_like(s_abs), where=valid),
            )

        return plan["handled_mask"]

    def _acac_measurement_plan(self, measurements: Optional[Sequence[Measurement]]) -> Dict[str, np.ndarray]:
        if measurements is None:
            active_plan = getattr(self, "_active_acac_measurement_plan", None)
            if active_plan is not None:
                return active_plan
            plan = self._build_acac_measurement_plan_from_table(
                self._active_measurement_plan_table("acac")
            )
            self._active_acac_measurement_plan = plan
            return plan
        key = id(measurements)
        cache = getattr(self, "_acac_measurement_plan_cache", {})
        cached = cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]
        plan = self._build_acac_measurement_plan(measurements)
        if len(cache) > 16:
            cache.clear()
        cache[key] = (measurements, plan)
        self._acac_measurement_plan_cache = cache
        return plan

    def _build_acac_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        self._ensure_measurement_sequence_indexes(measurements)
        plan_table = self._measurement_plan_table(
            measurements,
            self._ac_acac_plan_kind_by_type_code,
        )
        return self._build_acac_measurement_plan_from_table(plan_table)

    def _build_acac_measurement_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        rows = plan_table.row[
            (plan_table.device_type_code == DEVICE_TYPE_ACACConverter) & plan_table.handled
        ]
        if rows.size == 0:
            empty_i = np.array([], dtype=np.int64)
            return {
                "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
                "value_rows": empty_i,
                "value_kind": empty_i,
                "from_pos": empty_i,
                "to_pos": empty_i,
                "value_index": empty_i,
            }
        device_pos = plan_table.device_pos[rows]
        return {
            "handled_mask": np.asarray(plan_table.handled, dtype=bool).copy(),
            "value_rows": self._int_array(rows),
            "value_kind": self._int_array(plan_table.meas_kind[rows]),
            "from_pos": self._ac_acac_plan_i[device_pos],
            "to_pos": self._ac_acac_plan_j[device_pos],
            "value_index": self._ac_acac_plan_index[device_pos],
        }

    def _fill_acac_values_vectorized(
        self,
        values: np.ndarray,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACACConverter terminal P/Q/V/I measurements."""
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        idx = plan["value_index"]
        from_pos = plan["from_pos"]
        to_pos = plan["to_pos"]
        kind = plan["value_kind"]
        p_from = np.asarray(x[self.base_acac_p_from + idx], dtype=np.float64)
        q_from = np.asarray(x[self.base_acac_q_from + idx], dtype=np.float64)
        p_to = np.asarray(x[self.base_acac_p_to + idx], dtype=np.float64)
        q_to = np.asarray(x[self.base_acac_q_to + idx], dtype=np.float64)
        row_values = np.zeros(rows.size, dtype=np.float64)
        row_values[kind == MEAS_TYPE_P_FROM] = p_from[kind == MEAS_TYPE_P_FROM]
        row_values[kind == MEAS_TYPE_Q_FROM] = q_from[kind == MEAS_TYPE_Q_FROM]
        row_values[kind == MEAS_TYPE_P_TO] = p_to[kind == MEAS_TYPE_P_TO]
        row_values[kind == MEAS_TYPE_Q_TO] = q_to[kind == MEAS_TYPE_Q_TO]
        from_voltage_mask = kind == MEAS_TYPE_V_FROM
        to_voltage_mask = kind == MEAS_TYPE_V_TO
        if np.any(from_voltage_mask):
            row_values[from_voltage_mask] = voltage[from_pos[from_voltage_mask]]
        if np.any(to_voltage_mask):
            row_values[to_voltage_mask] = voltage[to_pos[to_voltage_mask]]
        i_from_mask = kind == MEAS_TYPE_I_FROM
        if np.any(i_from_mask):
            vm = voltage[from_pos[i_from_mask]]
            valid = np.abs(vm) > self.min_current_voltage
            current_values = np.zeros(np.count_nonzero(i_from_mask), dtype=np.float64)
            current_values[valid] = np.hypot(p_from[i_from_mask][valid], q_from[i_from_mask][valid]) / vm[valid]
            row_values[i_from_mask] = current_values
        i_to_mask = kind == MEAS_TYPE_I_TO
        if np.any(i_to_mask):
            vm = voltage[to_pos[i_to_mask]]
            valid = np.abs(vm) > self.min_current_voltage
            current_values = np.zeros(np.count_nonzero(i_to_mask), dtype=np.float64)
            current_values[valid] = np.hypot(p_to[i_to_mask][valid], q_to[i_to_mask][valid]) / vm[valid]
            row_values[i_to_mask] = current_values
        values[rows] = row_values
        return plan["handled_mask"]

    def _fill_acac_jacobian_sparse(
        self,
        H,
        plan: Dict[str, np.ndarray],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse ACACConverter terminal P/Q/V/I measurement rows."""
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        idx = plan["value_index"]
        from_pos = plan["from_pos"]
        to_pos = plan["to_pos"]
        kind = plan["value_kind"]
        p_from = np.asarray(x[self.base_acac_p_from + idx], dtype=np.float64)
        q_from = np.asarray(x[self.base_acac_q_from + idx], dtype=np.float64)
        p_to = np.asarray(x[self.base_acac_p_to + idx], dtype=np.float64)
        q_to = np.asarray(x[self.base_acac_q_to + idx], dtype=np.float64)

        for mask, col_base in (
            (kind == MEAS_TYPE_P_FROM, self.base_acac_p_from),
            (kind == MEAS_TYPE_Q_FROM, self.base_acac_q_from),
            (kind == MEAS_TYPE_P_TO, self.base_acac_p_to),
            (kind == MEAS_TYPE_Q_TO, self.base_acac_q_to),
        ):
            if np.any(mask):
                self._add_indexed_values(
                    H,
                    rows[mask],
                    col_base + idx[mask],
                    np.ones(np.count_nonzero(mask), dtype=np.float64),
                )

        v_from_mask = kind == MEAS_TYPE_V_FROM
        if np.any(v_from_mask):
            self._add_indexed_values(
                H,
                rows[v_from_mask],
                self.voltage_col[from_pos[v_from_mask]],
                np.ones(np.count_nonzero(v_from_mask), dtype=np.float64),
            )
        v_to_mask = kind == MEAS_TYPE_V_TO
        if np.any(v_to_mask):
            self._add_indexed_values(
                H,
                rows[v_to_mask],
                self.voltage_col[to_pos[v_to_mask]],
                np.ones(np.count_nonzero(v_to_mask), dtype=np.float64),
            )

        def add_current(mask: np.ndarray, p: np.ndarray, q: np.ndarray, node_pos: np.ndarray, p_col: int, q_col: int) -> None:
            if not np.any(mask):
                return
            i_rows = rows[mask]
            i_idx = idx[mask]
            vm = voltage[node_pos[mask]]
            p_i = p[mask]
            q_i = q[mask]
            s_abs = np.hypot(p_i, q_i)
            valid = (np.abs(vm) > self.min_current_voltage) & (s_abs > 1e-12)
            self._add_indexed_values(
                H,
                i_rows,
                p_col + i_idx,
                np.divide(p_i, s_abs * vm, out=np.zeros_like(p_i), where=valid),
            )
            self._add_indexed_values(
                H,
                i_rows,
                q_col + i_idx,
                np.divide(q_i, s_abs * vm, out=np.zeros_like(q_i), where=valid),
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.voltage_col[node_pos[mask]],
                np.divide(-s_abs, vm * vm, out=np.zeros_like(s_abs), where=valid),
            )

        add_current(
            kind == MEAS_TYPE_I_FROM,
            p_from,
            q_from,
            from_pos,
            self.base_acac_p_from,
            self.base_acac_q_from,
        )
        add_current(
            kind == MEAS_TYPE_I_TO,
            p_to,
            q_to,
            to_pos,
            self.base_acac_p_to,
            self.base_acac_q_to,
        )
        return plan["handled_mask"]

    def _assemble_jacobian(
        self,
        x: np.ndarray,
        measurement_plan_tables=None,
        sparse: bool = False,
    ):
        """Assemble the WLS measurement Jacobian H = dh(x)/dx."""
        measurement_plan_tables = self._measurement_plan_tables_for(measurement_plan_tables)
        vector_plans = self._vector_plans_for_measurement_plan_tables(measurement_plan_tables)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        n_meas = self._measurement_count(measurement_plan_tables)
        active_plan_run = self._measurement_plan_tables_are_active(measurement_plan_tables)
        active_vectorized = active_plan_run and bool(getattr(self, "active_measurements_are_vectorized", False))
        if sparse and active_plan_run:
            H = self._jacobian_builder
            H.shape = (n_meas, self.n_state)
            H.size = H.shape[0] * H.shape[1]
            H.reset()
        elif sparse:
            # Cache a fixed-pattern builder per `id(measurements)` so repeated
            # calls with the same measurements list (e.g. from a hybrid parent)
            # reuse the cached CSR pattern instead of rebuilding it each call.
            cache = getattr(self, "_external_jacobian_builder_cache", None)
            if cache is None:
                cache = {}
                self._external_jacobian_builder_cache = cache
            key = id(measurement_plan_tables)
            cached = cache.get(key)
            if cached is not None and cached[0] is measurement_plan_tables:
                H = cached[1]
                H.shape = (n_meas, self.n_state)
                H.size = H.shape[0] * H.shape[1]
                H.reset()
            else:
                H = SparseJacobianBuilder((n_meas, self.n_state))
                H._assume_fixed_pattern = True
                if len(cache) > 4:
                    cache.clear()
                cache[key] = (measurement_plan_tables, H)
        else:
            H = np.zeros((n_meas, self.n_state), dtype=np.float64)
        vectorized_branch_rows = self._fill_branch_transformer_jacobian_vectorized(
            H,
            vector_plans["branch_transformer"],
            theta,
            voltage,
            voltage_complex,
        )
        vectorized_simple_rows = self._fill_simple_jacobian_vectorized(
            H,
            vector_plans["simple"],
            x,
            voltage,
        )
        vectorized_zero_rows = self._fill_zero_current_jacobian_vectorized(
            H,
            vector_plans["zero_current"],
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_generator_rows = self._fill_generator_jacobian_sparse(
            H,
            vector_plans["generator"],
            x,
            voltage,
        )
        vectorized_acac_rows = self._fill_acac_jacobian_sparse(
            H,
            vector_plans["acac"],
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_jacobian_vectorized(
            H,
            vector_plans["balance"],
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        if active_vectorized:
            return H.to_csr() if sparse else H
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_simple_rows
            | vectorized_zero_rows
            | vectorized_generator_rows
            | vectorized_acac_rows
            | vectorized_balance_rows
        )
        if vectorized_rows.all():
            return H.to_csr() if sparse else H
        unhandled = np.flatnonzero(~vectorized_rows)
        warnings.warn(
            f"Skipped {int(unhandled.size)} unhandled AC SE Jacobian rows; string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return H.to_csr() if sparse else H

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None):
        """Assemble the WLS measurement Jacobian directly as sparse CSR triplets."""
        return self._assemble_jacobian(x, measurements, sparse=True)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        """Use the rank of H to identify whether all state variables are observable."""
        if (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
            and self._initial_observability_cache is not None
        ):
            cache = getattr(self, "_observability_matrix_cache", None)
            if cache is not None and cache.get("result") is self._initial_observability_cache:
                self._restore_active_lower_normal_plan_from_observability_cache(self._initial_observability_cache)
                return self._initial_observability_cache
        use_default_cache = (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
        )
        if use_default_cache:
            cached = _OBSERVABILITY_RESULT_CACHE.get(self._default_observability_cache_key())
            if cached is not None:
                cache = getattr(self, "_observability_matrix_cache", None)
                if cache is not None and cache.get("result") is cached:
                    self._initial_observability_cache = cached
                    self._restore_active_lower_normal_plan_from_observability_cache(cached)
                    return cached
        x = self.initial_state() if x is None else x
        measurement_plan_tables = self._measurement_plan_tables_for(measurements)
        H = self.jacobian_sparse(x, measurement_plan_tables) if H is None else H
        measurement_count = self._measurement_count(measurement_plan_tables)
        if matrix_is_empty(H):
            return ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])

        if (
            normal_matrix is None
            and normal_factor_diag is None
            and issparse(H)
            and int(H.shape[0]) >= int(self.n_state)
        ):
            structural_rank = sparse_structural_rank(H)
            if structural_rank == self.n_state:
                unanchored_angles = unanchored_angle_state_indices(H, self.angle_col[self.angle_col >= 0])
                if not unanchored_angles:
                    result = ObservabilityResult(
                        observable=True,
                        rank=self.n_state,
                        state_count=self.n_state,
                        measurement_count=measurement_count,
                        deficiency=0,
                        singular_values=np.array([], dtype=np.float64),
                        weak_states=[],
                    )
                    if use_default_cache:
                        if len(_OBSERVABILITY_RESULT_CACHE) > 32:
                            _OBSERVABILITY_RESULT_CACHE.clear()
                        _OBSERVABILITY_RESULT_CACHE[self._default_observability_cache_key()] = result
                        self._initial_observability_cache = result
                    self._cache_observability_matrix(result, x, measurement_plan_tables, H)
                    return result

        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.n_state,
            normal_matrix=normal_matrix,
            normal_factor_diag=normal_factor_diag,
        )
        if deficiency > 0 and self._has_structural_observability_certificate(H):
            rank = self.n_state
            deficiency = 0
            weak_states = []
        result = ObservabilityResult(
            observable=rank == self.n_state,
            rank=rank,
            state_count=self.n_state,
            measurement_count=measurement_count,
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )
        if use_default_cache:
            if len(_OBSERVABILITY_RESULT_CACHE) > 32:
                _OBSERVABILITY_RESULT_CACHE.clear()
            _OBSERVABILITY_RESULT_CACHE[self._default_observability_cache_key()] = result
            self._initial_observability_cache = result
        self._cache_observability_matrix(result, x, measurement_plan_tables, H)
        return result

    def _has_structural_observability_certificate(self, H) -> bool:
        """Certify large sparse AC cases when numeric LU is conservative but structure is anchored."""
        rank = sparse_structural_rank(H)
        if rank != self.n_state:
            return False
        return not unanchored_angle_state_indices(H, self.angle_col[self.angle_col >= 0])

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        """Run weighted least squares with simple damping to avoid voltage divergence."""
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        source_measurements = None if measurements is None or isinstance(measurements, dict) else self._normalize_measurements(measurements)
        measurement_plan_tables = self._measurement_plan_tables_for(
            measurements if source_measurements is None else source_measurements
        )
        n_meas = self._measurement_count(measurement_plan_tables)
        if n_meas < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {n_meas} < {self.n_state}")

        x = self.initial_state() if x0 is None else x0.copy()
        if observability is None:
            start = time.perf_counter() if self.profile_enabled else None
            if (
                self._measurement_plan_tables_are_active(measurement_plan_tables)
                and x0 is None
                and self._observability_cache_allowed()
            ):
                observability = self.observability_analysis()
            else:
                observability = self.observability_analysis(x, measurement_plan_tables)
            if start is not None:
                self._record_profile_time("solve.observability", time.perf_counter() - start)
        observability_cache = self._observability_matrix_cache_for(observability, measurement_plan_tables, x)
        cached_initial_H = observability_cache.get("H") if observability_cache is not None else None
        z, weight = self._measurement_vectors(measurement_plan_tables)
        active_measurement_run = self._measurement_plan_tables_are_active(measurement_plan_tables)
        uniform_weight = self.active_uniform_weight if active_measurement_run else self._uniform_weight(weight)
        weights_are_uniform = self.active_weights_are_uniform if active_measurement_run else uniform_weight is not None
        weighted_residual = None if (weights_are_uniform or active_measurement_run) else np.empty_like(weight)
        converged = False
        max_correction = np.inf
        objective = np.inf
        iteration = 0
        H = None
        gain = None
        final_quantities_current = False
        normal_solver = NormalEquationSolver(assume_fixed_pattern=active_measurement_run)
        lower_normal_plan = self._active_lower_normal_plan if active_measurement_run else None
        if (
            active_measurement_run
            and observability_cache is not None
            and observability_cache.get("lower_normal_plan") is not None
        ):
            lower_normal_plan = observability_cache["lower_normal_plan"]
        if (
            self.profile_enabled
            and active_measurement_run
            and lower_normal_plan is not None
            and "solve.lower_normal_plan_build" not in self.profile_times
        ):
            self._record_profile_time("solve.lower_normal_plan_build", 0.0)
        normal_pattern = self._active_normal_pattern if active_measurement_run else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]
        normal_assembly_plan = None if active_measurement_run else self._active_normal_assembly_plan
        if (
            not active_measurement_run
            and observability_cache is not None
            and observability_cache.get("normal_assembly_plan") is not None
        ):
            normal_assembly_plan = observability_cache["normal_assembly_plan"]
        normal_assembly_plan_disabled = active_measurement_run
        use_aat_normal_solver = active_measurement_run and self._aat_normal_solver_enabled()
        aat_plan = None
        aat_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True) if use_aat_normal_solver else None

        if verbose:
            _print_iteration_header()

        flat_restart_enabled = False
        iteration_limit = self.max_iter

        # Pre-allocated buffers reused across iterations and line-search trials.
        # ``z_est_dirty`` tracks whether the main buffer already reflects the
        # current ``x`` (set True after an accepted line-search step swap).
        z_est = np.empty(n_meas, dtype=np.float64)
        residual = np.empty(n_meas, dtype=np.float64)
        cand_z_est = np.empty(n_meas, dtype=np.float64)
        cand_residual = np.empty(n_meas, dtype=np.float64)
        z_est_dirty = True  # True => z_est does not yet reflect current x

        for iteration in range(1, iteration_limit + 1):
            if z_est_dirty:
                start = time.perf_counter() if self.profile_enabled else None
                self.evaluate(x, measurement_plan_tables, out=z_est)
                if start is not None:
                    self._record_profile_time("solve.evaluate", time.perf_counter() - start)
                start = time.perf_counter() if self.profile_enabled else None
                self._measurement_residual(z, z_est, measurement_plan_tables, out=residual)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.residual_objective", time.perf_counter() - start)
                z_est_dirty = False
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if iteration == 1 and cached_initial_H is not None:
                H = cached_initial_H
                cached_initial_H = None
            else:
                start = time.perf_counter() if self.profile_enabled else None
                H = self.jacobian_sparse(x, measurement_plan_tables)
                if start is not None:
                    self._record_profile_time("solve.jacobian", time.perf_counter() - start)
            if normal_assembly_plan is not None and not normal_assembly_plan.matches(H):
                normal_assembly_plan = None
                if active_measurement_run:
                    self._active_normal_assembly_plan = None
                if observability_cache is not None:
                    observability_cache["normal_assembly_plan"] = None
            if lower_normal_plan is not None and (
                not issparse(H)
                or tuple(H.shape) != lower_normal_plan.shape
                or int(H.nnz) != int(lower_normal_plan.h_indices.size)
            ):
                lower_normal_plan = None
                if active_measurement_run:
                    self._active_lower_normal_plan = None
                if observability_cache is not None:
                    observability_cache["lower_normal_plan"] = None
            if (
                active_measurement_run
                and not use_aat_normal_solver
                and lower_normal_plan is None
                and issparse(H)
            ):
                start = time.perf_counter() if self.profile_enabled else None
                lower_normal_plan = LowerNormalEquationCscPlan.from_jacobian(H)
                if start is not None:
                    self._record_profile_time("solve.lower_normal_plan_build", time.perf_counter() - start)
                self._active_lower_normal_plan = lower_normal_plan
                if observability_cache is not None:
                    observability_cache["lower_normal_plan"] = lower_normal_plan
            if normal_assembly_plan is None and issparse(H) and not normal_assembly_plan_disabled:
                if not NormalEquationAssemblyPlan.direct_assembly_is_reasonable(H):
                    normal_assembly_plan_disabled = True
                else:
                    start = time.perf_counter() if self.profile_enabled else None
                    normal_assembly_plan = NormalEquationAssemblyPlan.from_jacobian(H)
                    if start is not None:
                        self._record_profile_time("solve.normal_assembly_plan_build", time.perf_counter() - start)
                    if active_measurement_run:
                        self._active_normal_assembly_plan = normal_assembly_plan
                    if observability_cache is not None:
                        observability_cache["normal_assembly_plan"] = normal_assembly_plan
            if (
                not use_aat_normal_solver
                and lower_normal_plan is None
                and normal_pattern is None
                and issparse(H)
            ):
                start = time.perf_counter() if self.profile_enabled else None
                normal_pattern = _normal_equation_structural_pattern(H)
                if start is not None:
                    self._record_profile_time("solve.normal_pattern", time.perf_counter() - start)
                if active_measurement_run:
                    self._active_normal_pattern = normal_pattern
                if observability_cache is not None:
                    observability_cache["normal_pattern"] = normal_pattern
            normal_plan_can_weight_rhs = (
                (lower_normal_plan is not None and active_measurement_run)
                or use_aat_normal_solver
            )
            if weighted_residual is not None and not normal_plan_can_weight_rhs:
                np.multiply(weight, residual, out=weighted_residual)
            start = time.perf_counter() if self.profile_enabled else None
            normal_operand = None
            if use_aat_normal_solver and aat_solver is not None:
                if (
                    aat_plan is not None
                    and not active_measurement_run
                    and not aat_plan.matches(H)
                ):
                    aat_plan = None
                if (
                    aat_plan is not None
                    and active_measurement_run
                    and (
                        not issparse(H)
                        or tuple(H.shape) != aat_plan.shape
                        or int(H.nnz) != int(aat_plan.h_indices.size)
                    )
                ):
                    aat_plan = None
                if aat_plan is None:
                    plan_start = time.perf_counter() if self.profile_enabled else None
                    aat_plan = CholmodAAtNormalEquationPlan.from_jacobian(H)
                    aat_plan.prepare_fixed_weights(weight)
                    if plan_start is not None:
                        self._record_profile_time("solve.aat_plan_build", time.perf_counter() - plan_start)
                normal_operand, rhs = aat_plan.assemble(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=None,
                    assume_fixed_weights=active_measurement_run,
                    copy_rhs=False,
                    assume_pattern_matches=active_measurement_run,
                )
                gain = None
            elif lower_normal_plan is not None:
                gain, rhs = lower_normal_plan.assemble(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=None if normal_plan_can_weight_rhs else weighted_residual,
                    dense_gain_limit=0,
                    assume_fixed_weights=active_measurement_run,
                    copy_rhs=False,
                )
            else:
                gain, rhs = build_normal_equations(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    normal_pattern=normal_pattern,
                    assume_normal_pattern_matches=False,
                    normal_assembly_plan=normal_assembly_plan,
                )
            if start is not None:
                self._record_profile_time("solve.normal_equations", time.perf_counter() - start)
            if self.matrix_dump_dir is not None:
                start = time.perf_counter() if self.profile_enabled else None
                self._dump_iteration_matrices(iteration, H, weight, gain)
                if start is not None:
                    self._record_profile_time("solve.matrix_dump", time.perf_counter() - start)
            start = time.perf_counter() if self.profile_enabled else None
            if normal_operand is not None and aat_solver is not None:
                dx, _ = aat_solver.solve(
                    normal_operand,
                    rhs,
                    return_factor_diag=False,
                )
            else:
                dx, _ = normal_solver.solve(
                    gain,
                    rhs,
                    return_factor_diag=False,
                )
            if start is not None:
                self._record_profile_time("solve.linear_solve", time.perf_counter() - start)

            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            if max_correction < self.tol:
                converged = True
                final_quantities_current = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break

            accepted = False
            step_scale = 1.0
            accepted_step = None
            nonfinite_candidates = 0
            previous_objective = objective
            # Backtracking keeps the weighted objective non-increasing on difficult cases.
            for _ in range(20):
                candidate = x + step_scale * dx
                candidate[self.n_angle : self.base_switch_re] = np.maximum(
                    candidate[self.n_angle : self.base_switch_re],
                    self.voltage_floor,
                )
                start = time.perf_counter() if self.profile_enabled else None
                self.evaluate(candidate, measurement_plan_tables, out=cand_z_est)
                if start is not None:
                    self._record_profile_time("solve.line_search_evaluate", time.perf_counter() - start)
                start = time.perf_counter() if self.profile_enabled else None
                self._measurement_residual(z, cand_z_est, measurement_plan_tables, out=cand_residual)
                candidate_objective = self._weighted_objective(weight, cand_residual)
                if start is not None:
                    self._record_profile_time("solve.line_search_residual_objective", time.perf_counter() - start)
                finite_candidate = np.isfinite(candidate_objective)
                if not finite_candidate:
                    nonfinite_candidates += 1
                    if nonfinite_candidates >= 8:
                        break
                    step_scale *= 0.5
                    continue
                nonfinite_candidates = 0
                objective_tol = max(1e-12, 1e-10 * (abs(objective) + 1.0))
                if finite_candidate and candidate_objective <= objective + objective_tol:
                    x = candidate
                    # Swap so the accepted candidate becomes the main buffer.
                    z_est, cand_z_est = cand_z_est, z_est
                    residual, cand_residual = cand_residual, residual
                    objective = candidate_objective
                    accepted_step = step_scale
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                final_quantities_current = True
                break

            objective_change = abs(previous_objective - objective)
            stagnation_tol = max(1e-14, 1e-10 * (abs(previous_objective) + 1.0))
            practically_converged = (
                max_correction < 10.0 * self.tol
                and objective_change <= stagnation_tol
            )

            if verbose:
                # `z_est`/`residual` already reflect the accepted candidate via swap.
                updated_residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                _print_iteration(
                    iteration,
                    objective,
                    updated_residual_inf,
                    max_correction,
                    accepted_step,
                    False,
                )

            if max_correction < self.tol or practically_converged:
                converged = True
                break

        if not converged and flat_restart_enabled:
            restart_result = self.estimate(
                measurements,
                x0=self._file_state(),
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
            restart_result.iterations += iteration
            return restart_result

        if not final_quantities_current and z_est_dirty:
            start = time.perf_counter() if self.profile_enabled else None
            self.evaluate(x, measurement_plan_tables, out=z_est)
            if start is not None:
                self._record_profile_time("solve.evaluate", time.perf_counter() - start)
            start = time.perf_counter() if self.profile_enabled else None
            self._measurement_residual(z, z_est, measurement_plan_tables, out=residual)
            objective = self._weighted_objective(weight, residual)
            if start is not None:
                self._record_profile_time("solve.final_residual_objective", time.perf_counter() - start)
        if final_diagnostics and not final_quantities_current:
            start = time.perf_counter() if self.profile_enabled else None
            H = self.jacobian_sparse(x, measurement_plan_tables)
            if start is not None:
                self._record_profile_time("solve.jacobian", time.perf_counter() - start)
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            start = time.perf_counter() if self.profile_enabled else None
            if lower_normal_plan is not None:
                gain, _ = lower_normal_plan.assemble(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    dense_gain_limit=0,
                    assume_fixed_weights=active_measurement_run,
                    copy_rhs=False,
                )
            else:
                gain, _ = build_normal_equations(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    normal_pattern=normal_pattern,
                    normal_assembly_plan=normal_assembly_plan,
                )
            if start is not None:
                self._record_profile_time("solve.normal_equations", time.perf_counter() - start)
        elif not final_diagnostics:
            H = None
            gain = None
        elif lower_normal_plan is not None and gain is not None:
            start = time.perf_counter() if self.profile_enabled else None
            gain = full_normal_equation_from_lower(gain)
            if start is not None:
                self._record_profile_time("solve.full_gain_from_lower", time.perf_counter() - start)
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        result_table = self._common_measurement_plan_table(measurement_plan_tables).table
        result_measurements = []
        return EstimateResult(
            converged=converged,
            iterations=iteration,
            objective=objective,
            max_correction=max_correction,
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=x,
            z_est=z_est.copy(),
            residual=residual.copy(),
            H=H,
            gain=gain,
            measurements=result_measurements,
            observability=observability,
            measurement_plan_tables=measurement_plan_tables,
            measurement_table=result_table,
        )

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Return measurements whose normalized residual exceeds the bad-data threshold."""
        plan_tables = getattr(result, "measurement_plan_tables", None)
        if plan_tables is None:
            self._warn_required_runtime_missing(
                "measurement_plan_tables",
                "bad-data identification",
            )
        plan_table = self._common_measurement_plan_table(plan_tables)
        measurement_table = plan_table.table
        if measurement_table is None:
            self._warn_required_runtime_missing(
                "measurement_table",
                "bad-data identification",
            )
        if result.H is None or result.gain is None:
            result.H = self.jacobian_sparse(result.x, plan_tables)
            weights_for_gain = np.asarray(measurement_table.weight, dtype=np.float64)
            result.gain, _ = build_normal_equations(
                result.H,
                result.residual,
                weights_for_gain,
            )
        threshold = self.params.bad_threshold if threshold is None else threshold
        weights = np.asarray(measurement_table.weight, dtype=np.float64)
        R_diag = 1.0 / weights
        gain_inv = inverse_gain_for_bad_data(result.gain)
        if gain_inv is None:
            leverage = np.zeros_like(R_diag)
        else:
            leverage = measurement_leverage(result.H, gain_inv)
        omega_diag = np.maximum(R_diag - leverage, 1e-12)
        normalized = np.abs(result.residual) / np.sqrt(omega_diag)

        bad_items = []
        for idx in np.where(normalized > threshold)[0]:
            row_pos = int(idx)
            meas = measurement_from_table_row(measurement_table, row_pos)
            bad_items.append(
                BadDataItem(
                    measurement=meas,
                    residual=float(result.residual[row_pos]),
                    normalized_residual=float(normalized[row_pos]),
                    estimated_value=float(result.z_est[row_pos]),
                    measured_value=float(measurement_table.value[row_pos]),
                    row_pos=row_pos,
                )
            )
        bad_items.sort(key=lambda item: item.normalized_residual, reverse=True)
        return bad_items, normalized

    def build_se_result(
        self,
        result: EstimateResult,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        result_mode: str = "full",
    ) -> Optional[SEResult]:
        """Build the structured state-estimation result snapshot after WLS."""
        mode = normalize_seresult_result_mode(result_mode)
        if mode in ("none", "array"):
            self.se_result = None
            return None
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        if mode == "summary":
            self.se_result = build_seresult_summary_from_table(
                result,
                bad_items=bad_items,
                all_measurement_table=getattr(self.measurements, "table", None),
                active_source_rows=getattr(self, "active_measurement_rows", None),
            )
            return self.se_result
        self.se_result = build_seresult_full_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurement_table=getattr(self.measurements, "table", None),
            active_source_rows=getattr(self, "active_measurement_rows", None),
        )
        return self.se_result

    def run(
        self,
        *,
        result_mode: str = "full",
        remove_bad_data: bool = False,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        skip_bad_data: bool = False,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> Optional[SEResult]:
        mode = normalize_seresult_result_mode(result_mode)
        array_only = mode == "array"
        if array_only and remove_bad_data:
            raise ValueError("result_mode='array' cannot be combined with remove_bad_data=True")
        self._array_only_runtime = True
        if not self._prepared:
            self.prepare()
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        needs_bad_data = not skip_bad_data
        if observability is None:
            observability = self.observability_analysis()
        self.observability_result = observability
        removed: List[BadDataItem] = []
        previous_array_only = bool(getattr(self, "_array_only_estimate_result", False))
        self._array_only_estimate_result = True
        try:
            if remove_bad_data:
                result, removed = self.estimate_with_bad_data_removal(
                    threshold,
                    max_remove=max_remove,
                    verbose=verbose,
                )
            else:
                result = self.estimate(
                    verbose=verbose,
                    final_diagnostics=final_diagnostics and needs_bad_data,
                    observability=observability,
                )
        finally:
            self._array_only_estimate_result = previous_array_only
            self._array_only_runtime = True
        self.estimate_result = result
        self.removed_bad_data = removed
        if skip_bad_data:
            bad_items = []
            normalized = np.array([], dtype=np.float64)
        else:
            bad_items, normalized = self.identify_bad_data(result, threshold)
        self.bad_items = bad_items
        self.normalized_residual = normalized
        if mode in ("none", "array"):
            self.se_result = None
            return None
        return self.build_se_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized,
            result_mode=mode,
        )

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        threshold = self.params.bad_threshold if threshold is None else threshold
        max_remove = self.params.max_remove if max_remove is None else max_remove
        measurement_plan_tables = self._active_measurement_plan_tables_ref()
        removed: List[BadDataItem] = []
        x0 = self.initial_state()
        for round_idx in range(max_remove + 1):
            if verbose:
                print(
                    f"Bad-data removal round {round_idx + 1}: "
                    f"measurements={self._measurement_count(measurement_plan_tables)}"
                )
            result = self.estimate(measurement_plan_tables, x0=x0, verbose=verbose)
            bad_items, _ = self.identify_bad_data(result, threshold)
            if not bad_items:
                return result, removed
            worst = bad_items[0]
            removed.append(worst)
            remove_pos = int(getattr(worst, "row_pos", -1))
            if remove_pos < 0:
                warnings.warn(
                    "AC SE bad-data removal requires BadDataItem.row_pos; object-index fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            measurement_plan_tables = self._shrink_measurement_plan_tables(
                measurement_plan_tables,
                remove_pos,
            )
            x0 = result.x
        return result, removed

    def apply_state(self, x: np.ndarray) -> None:
        theta, voltage = self._unpack_state(x)
        ppc = self._ac_ppc_dict()
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        topology = ppc.get("_topology_arrays")
        first_rows = np.asarray(getattr(self, "_ac_first_node_row_by_solver_pos", ()), dtype=np.int32)
        for solver_pos, row in enumerate(first_rows):
            row_int = int(row)
            if 0 <= row_int < bus.shape[0]:
                bus[row_int, BUS_COLS["angle"]] = float(theta[int(solver_pos)])
                bus[row_int, BUS_COLS["voltage"]] = float(voltage[int(solver_pos)])
        if topology is not None:
            node_solver_pos = np.asarray(getattr(self, "_ac_node_solver_pos_by_ppc_row", ()), dtype=np.int32)
            rows = np.flatnonzero((node_solver_pos >= 0) & (node_solver_pos < voltage.size)).astype(np.int64, copy=False)
            if rows.size:
                solver = node_solver_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)
                bus[rows.astype(np.intp, copy=False), BUS_COLS["angle"]] = theta[solver]
                bus[rows.astype(np.intp, copy=False), BUS_COLS["voltage"]] = voltage[solver]
        gen_rows = np.asarray(getattr(self, "_ac_generator_power_rows", ()), dtype=np.int64)
        if gen_rows.size:
            gen = np.asarray(ppc["gen"], dtype=np.float64)
            gen[gen_rows.astype(np.intp, copy=False), GEN_COLS["p"]] = x[self.base_gen_p : self.base_gen_q]
            gen[gen_rows.astype(np.intp, copy=False), GEN_COLS["q"]] = x[self.base_gen_q : self.base_load_p]
        load_rows = np.asarray(getattr(self, "_ac_load_power_rows", ()), dtype=np.int64)
        if load_rows.size:
            load = np.asarray(ppc["load"], dtype=np.float64)
            load[load_rows.astype(np.intp, copy=False), LOAD_COLS["p"]] = x[self.base_load_p : self.base_load_q]
            load[load_rows.astype(np.intp, copy=False), LOAD_COLS["q"]] = x[self.base_load_q : self.base_shunt_q]
        shunt_rows = np.asarray(getattr(self, "_ac_shunt_device_rows", ()), dtype=np.int64)
        if shunt_rows.size and self.n_shunt_q:
            shunt = np.asarray(ppc["shunt"], dtype=np.float64)
            control_type = shunt[shunt_rows.astype(np.intp, copy=False), SHUNT_COLS["control_type"]].astype(
                np.int64,
                copy=False,
            )
            q_rows = shunt_rows[control_type == SHUNT_V]
            shunt[q_rows.astype(np.intp, copy=False), SHUNT_COLS["q"]] = x[
                self.base_shunt_q : self.base_acac_p_from
            ]
        acac_rows = np.asarray(getattr(self, "_ac_acac_power_rows", ()), dtype=np.int64)
        if acac_rows.size and self.n_acac_power:
            acac = np.asarray(ppc.get("acac", np.zeros((0, len(ACAC_COLS)))), dtype=np.float64)
            acac[acac_rows.astype(np.intp, copy=False), ACAC_COLS["i_p"]] = x[
                self.base_acac_p_from : self.base_acac_q_from
            ]
            acac[acac_rows.astype(np.intp, copy=False), ACAC_COLS["i_q"]] = x[
                self.base_acac_q_from : self.base_acac_p_to
            ]
            acac[acac_rows.astype(np.intp, copy=False), ACAC_COLS["j_p"]] = x[
                self.base_acac_p_to : self.base_acac_q_to
            ]
            acac[acac_rows.astype(np.intp, copy=False), ACAC_COLS["j_q"]] = x[
                self.base_acac_q_to : self.n_state
            ]

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        theta, voltage = self._unpack_state(x)
        node_names = np.asarray(getattr(self, "_ac_node_names", np.asarray([], dtype=object)), dtype=object)
        print("Estimated states:")
        count = min(int(limit), int(self.n_nodes))
        for pos in range(count):
            name = str(node_names[pos]) if pos < node_names.size else f"bus_{pos + 1}"
            print(f"  {name:10s} V={voltage[pos]:.9f} theta={theta[pos]:.9f} rad")
        if self.n_nodes > limit:
            print(f"  ... {self.n_nodes - limit} more nodes")


_ORIGINAL_AC_RUN_POWER_FLOW_SEED = ACStateEstimator._run_power_flow_seed


def _print_observability(result: ObservabilityResult) -> None:
    print(
        "Observability: "
        f"observable={result.observable}, "
        f"rank={result.rank}/{result.state_count}, "
        f"measurements={result.measurement_count}, "
        f"deficiency={result.deficiency}"
    )
    if result.weak_states:
        print("Weak/unobservable state candidates:")
        for label, score in result.weak_states:
            print(f"  {label}: {score:.3e}")


def _print_bad_data(items: Sequence[BadDataItem], normalized: np.ndarray, threshold: float, top: int = 10) -> None:
    max_norm = float(np.max(normalized)) if normalized.size else 0.0
    print(f"Bad data: threshold={threshold:.3f}, max_normalized_residual={max_norm:.3e}, count={len(items)}")
    for item in list(items)[:top]:
        meas = item.measurement
        print(
            f"  idx={meas.idx} name={meas.name} type={meas.meas_type} device={meas.device} "
            f"z={item.measured_value:.9g} h={item.estimated_value:.9g} "
            f"res={item.residual:.3e} rn={item.normalized_residual:.3e}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AC weighted least-squares state estimation.")
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="AC network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat voltage start instead of E-file voltage/angle.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--skip-bad-data", action="store_true", help="Skip post-estimation bad-data analysis.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated node states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    parser.add_argument("--profile", action="store_true", help="Print initialization profile timings.")
    parser.add_argument(
        "--matrix-dump-dir",
        default=None,
        help="Directory for per-iteration sparse triplet dumps: jN.txt, dN.txt, hN.txt.",
    )
    parser.add_argument(
        "--result-mode",
        default=None,
        choices=("full", "summary", "array", "none"),
        help="SEResult payload mode: full, summary, array, or none.",
    )
    parser.add_argument("--se-result", default=None, help="Write SEResult blocks to a new E file.")
    args = parser.parse_args(argv)

    estimator = ACStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        parameter_file=Path(args.para),
        flat_start=args.flat_start,
        profile=args.profile,
        auto_prepare=False,
        matrix_dump_dir=Path(args.matrix_dump_dir) if args.matrix_dump_dir else None,
    )

    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    result_mode = args.result_mode if args.result_mode is not None else ("full" if args.se_result else "none")
    se_result = estimator.run(
        result_mode=result_mode,
        remove_bad_data=args.remove_bad_data,
        bad_threshold=bad_threshold,
        max_remove=args.max_remove,
        skip_bad_data=args.skip_bad_data,
        verbose=not args.quiet,
        final_diagnostics=not args.skip_bad_data,
    )
    _print_observability(estimator.observability_result)
    result = estimator.estimate_result
    removed = estimator.removed_bad_data
    if removed:
        print("Removed bad data:")
        for item in removed:
            print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")

    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
    )
    if not args.skip_bad_data:
        bad_items = estimator.bad_items
        normalized = estimator.normalized_residual
        _print_bad_data(bad_items, normalized, bad_threshold)
    else:
        bad_items = []
        normalized = np.array([], dtype=np.float64)

    if args.se_result and se_result is not None:
        se_result.write_e_file(Path(args.se_result))

    if args.print_state:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in sorted(estimator.profile_times.items()):
            print(f"  {name}={value:.6f}s")

    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
