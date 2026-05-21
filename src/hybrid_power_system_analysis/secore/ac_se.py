import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, issparse
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
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PQ,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    BREAK_COLS,
    SHUNT_B,
    SHUNT_COLS,
    SHUNT_Q,
    SHUNT_V,
    SHUNT_Z,
    SWITCH_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
)
from model.ppc_topology import build_ac_ppc_with_topology_from_e_file, ensure_ac_ppc_topology
from model.meas_array_model import (
    MEAS_COLS,
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_list_from_meas_ppc,
    measurement_table_from_meas_ppc,
    sync_meas_ppc_from_measurement_table,
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
    DEVICE_TYPE_ACSwitchConstraint,
    DEVICE_TYPE_ACSwitch,
    DEVICE_TYPE_ACPowerBalance,
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACBreakConstraint,
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
    TableBackedMeasurementList,
    measurement_status_is_active,
    measurement_from_table_row,
    measurement_table_from_measurements,
    measurement_table_status_code,
    normalize_measurement_status,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    SparseJacobianBuilder,
    angle_residual_mask,
    build_normal_equations,
    full_normal_equation_from_lower,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    measurement_residual as build_measurement_residual,
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
from secore.state_metadata import StateMeta, state_labels_from_metadata, state_meta_at
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
from secore.se_result import SEResult, build_seresult_summary, normalize_seresult_result_mode
from unit_system import ac_current_base_ka


DEFAULT_CASE = model_file("ac", "ieee39.e")
DEFAULT_MEAS = measurement_file("ac", "ieee39.meas")


_DEVICE_TYPE_CODES = DEVICE_TYPE_CODES

DEVICE_TYPE_CODES_ACNODE = DEVICE_TYPE_ACNode
DEVICE_TYPE_CODES_ACBRANCH = DEVICE_TYPE_ACBranch
DEVICE_TYPE_CODES_ACTRANSFORMER = DEVICE_TYPE_ACTransformer
DEVICE_TYPE_CODES_ACLOAD = DEVICE_TYPE_ACLoad
DEVICE_TYPE_CODES_ACGENERATOR = DEVICE_TYPE_ACGenerator
DEVICE_TYPE_CODES_ACZEROBRANCH = DEVICE_TYPE_ACZeroBranch
DEVICE_TYPE_CODES_ACZEROBRANCHCONSTRAINT = DEVICE_TYPE_ACZeroBranchConstraint
DEVICE_TYPE_CODES_ACSWITCHCONSTRAINT = DEVICE_TYPE_ACSwitchConstraint
DEVICE_TYPE_CODES_ACSWITCH = DEVICE_TYPE_ACSwitch
DEVICE_TYPE_CODES_ACPOWERBALANCE = DEVICE_TYPE_ACPowerBalance
DEVICE_TYPE_CODES_ACBREAK = DEVICE_TYPE_ACBreak
DEVICE_TYPE_CODES_ACBREAKCONSTRAINT = DEVICE_TYPE_ACBreakConstraint

MEAS_TYPE_CODES_V = MEAS_TYPE_V
MEAS_TYPE_CODES_ANGLE = MEAS_TYPE_ANGLE
MEAS_TYPE_CODES_THETA = MEAS_TYPE_THETA
MEAS_TYPE_CODES_P_FROM = MEAS_TYPE_P_FROM
MEAS_TYPE_CODES_Q_FROM = MEAS_TYPE_Q_FROM
MEAS_TYPE_CODES_V_FROM = MEAS_TYPE_V_FROM
MEAS_TYPE_CODES_I_FROM = MEAS_TYPE_I_FROM
MEAS_TYPE_CODES_P_TO = MEAS_TYPE_P_TO
MEAS_TYPE_CODES_Q_TO = MEAS_TYPE_Q_TO
MEAS_TYPE_CODES_V_TO = MEAS_TYPE_V_TO
MEAS_TYPE_CODES_I_TO = MEAS_TYPE_I_TO
MEAS_TYPE_CODES_P_LOAD = MEAS_TYPE_P_LOAD
MEAS_TYPE_CODES_Q_LOAD = MEAS_TYPE_Q_LOAD
MEAS_TYPE_CODES_V_LOAD = MEAS_TYPE_V_LOAD
MEAS_TYPE_CODES_I_LOAD = MEAS_TYPE_I_LOAD
MEAS_TYPE_CODES_P_GEN = MEAS_TYPE_P_GEN
MEAS_TYPE_CODES_Q_GEN = MEAS_TYPE_Q_GEN
MEAS_TYPE_CODES_V_GEN = MEAS_TYPE_V_GEN
MEAS_TYPE_CODES_I_GEN = MEAS_TYPE_I_GEN
MEAS_TYPE_CODES_P_BALANCE = MEAS_TYPE_P_BALANCE
MEAS_TYPE_CODES_Q_BALANCE = MEAS_TYPE_Q_BALANCE
MEAS_TYPE_CODES_V_DIFF = MEAS_TYPE_V_DIFF
MEAS_TYPE_CODES_ANGLE_DIFF = MEAS_TYPE_ANGLE_DIFF
MEAS_TYPE_CODES_THETA_DIFF = MEAS_TYPE_THETA_DIFF

_TERMINAL_POWER_MEASUREMENT_TYPES = frozenset(
    (
        MEAS_TYPE_CODES_P_FROM,
        MEAS_TYPE_CODES_Q_FROM,
        MEAS_TYPE_CODES_P_TO,
        MEAS_TYPE_CODES_Q_TO,
    )
)
_VOLTAGE_MEASUREMENT_TYPES = frozenset(
    (
        MEAS_TYPE_CODES_V,
        MEAS_TYPE_CODES_V_FROM,
        MEAS_TYPE_CODES_V_TO,
        MEAS_TYPE_CODES_V_GEN,
        MEAS_TYPE_CODES_V_LOAD,
    )
)
_ANGLE_MEASUREMENT_CODE_SET = frozenset(
    (
        MEAS_TYPE_CODES_ANGLE,
        MEAS_TYPE_CODES_THETA,
        MEAS_TYPE_CODES_ANGLE_DIFF,
        MEAS_TYPE_CODES_THETA_DIFF,
    )
)
_PSEUDO_DEVICE_SUMMARY_TYPES = frozenset(
    (
        DEVICE_TYPE_CODES_ACGENERATOR,
        DEVICE_TYPE_CODES_ACLOAD,
    )
)
_PSEUDO_MEASUREMENT_SUMMARY_TYPES = {
    DEVICE_TYPE_CODES_ACGENERATOR: frozenset((MEAS_TYPE_CODES_P_GEN, MEAS_TYPE_CODES_Q_GEN)),
    DEVICE_TYPE_CODES_ACLOAD: frozenset((MEAS_TYPE_CODES_P_LOAD, MEAS_TYPE_CODES_Q_LOAD)),
    DEVICE_TYPE_CODES_ACZEROBRANCH: frozenset(
        (MEAS_TYPE_CODES_P_FROM, MEAS_TYPE_CODES_Q_FROM, MEAS_TYPE_CODES_V_FROM, MEAS_TYPE_CODES_I_FROM)
    ),
    DEVICE_TYPE_CODES_ACBREAK: frozenset(
        (MEAS_TYPE_CODES_P_FROM, MEAS_TYPE_CODES_Q_FROM, MEAS_TYPE_CODES_V_FROM, MEAS_TYPE_CODES_I_FROM)
    ),
}

_OBSERVABILITY_RESULT_CACHE = {}
_CTRL_NAME_BY_CODE = ("PQ", "P", "PV", "SLACK")
_SHUNT_NAME_BY_CODE = ("Q", "V", "B", "Z")

_MAX_MEAS_TYPE_CODE = max(MEAS_TYPE_CODES.values())
_TERMINAL_POWER_MEASUREMENT_TYPE_CODES = np.asarray(tuple(_TERMINAL_POWER_MEASUREMENT_TYPES), dtype=np.int16)
_VOLTAGE_MEASUREMENT_TYPE_CODES = np.asarray(tuple(_VOLTAGE_MEASUREMENT_TYPES), dtype=np.int16)


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
        MEAS_TYPE_CODES_P_FROM,
        MEAS_TYPE_CODES_Q_FROM,
        MEAS_TYPE_CODES_V_FROM,
        MEAS_TYPE_CODES_I_FROM,
        MEAS_TYPE_CODES_P_TO,
        MEAS_TYPE_CODES_Q_TO,
        MEAS_TYPE_CODES_V_TO,
        MEAS_TYPE_CODES_I_TO,
    )
)
_AC_ZERO_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_P_FROM,
        MEAS_TYPE_CODES_Q_FROM,
        MEAS_TYPE_CODES_V_FROM,
        MEAS_TYPE_CODES_I_FROM,
        MEAS_TYPE_CODES_P_TO,
        MEAS_TYPE_CODES_Q_TO,
        MEAS_TYPE_CODES_V_TO,
        MEAS_TYPE_CODES_I_TO,
        MEAS_TYPE_CODES_V_DIFF,
        MEAS_TYPE_CODES_ANGLE_DIFF,
        MEAS_TYPE_CODES_THETA_DIFF,
    )
)
_AC_NODE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_V,
        MEAS_TYPE_CODES_ANGLE,
        MEAS_TYPE_CODES_THETA,
    )
)
_AC_LOAD_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_P_LOAD,
        MEAS_TYPE_CODES_Q_LOAD,
        MEAS_TYPE_CODES_I_LOAD,
        MEAS_TYPE_CODES_V_LOAD,
    )
)
_AC_GENERATOR_POWER_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_P_GEN,
        MEAS_TYPE_CODES_Q_GEN,
        MEAS_TYPE_CODES_I_GEN,
    )
)
_AC_GENERATOR_SIMPLE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (MEAS_TYPE_CODES_V_GEN,)
)
_AC_BALANCE_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_P_BALANCE,
        MEAS_TYPE_CODES_Q_BALANCE,
    )
)
_AC_CONSTRAINT_MEAS_TYPE_LOOKUP = _measurement_type_code_lookup_from_codes(
    (
        MEAS_TYPE_CODES_V_DIFF,
        MEAS_TYPE_CODES_ANGLE_DIFF,
        MEAS_TYPE_CODES_THETA_DIFF,
    )
)


def _meas_type_code_array(meas_type_values) -> np.ndarray:
    values = np.asarray(meas_type_values, dtype=object)
    if values.size == 0:
        return np.empty(0, dtype=np.int16)
    code_get = MEAS_TYPE_CODES.get
    return np.asarray([code_get(str(name).upper(), 0) for name in values.tolist()], dtype=np.int16)


def _measurement_type_code_lookup(kind_map) -> np.ndarray:
    if isinstance(kind_map, np.ndarray):
        return kind_map
    warnings.warn(
        "AC SE measurement plan expects MEAS_TYPE lookup arrays; string kind maps are disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _measurement_type_code_lookup_from_codes(())


class _ACArrayObject:
    __slots__ = (
        "idx",
        "name",
        "vbase",
        "voltage",
        "angle",
        "run_stat",
        "is_alive",
        "isl",
        "isl_obj",
        "bus",
        "bus_obj",
        "generators",
        "loads",
        "branches",
        "switches",
        "breakers",
        "zero_branches",
        "transformers",
        "shunt_compensators",
        "v_gens",
        "node",
        "node_obj",
        "ppc_row",
        "plan_pos",
        "i_node",
        "j_node",
        "i_node_obj",
        "j_node_obj",
        "r",
        "x",
        "b",
        "gt",
        "bt",
        "tap",
        "shift",
        "status",
        "control_type",
        "p_set",
        "q_set",
        "v_set",
        "alpha",
        "pbase",
        "pv0",
        "pv1",
        "pv2",
        "qbase",
        "qv0",
        "qv1",
        "qv2",
        "g_set",
        "b_set",
        "p",
        "q",
        "current",
        "i_p",
        "i_q",
        "i_c",
        "j_p",
        "j_q",
        "j_c",
)


class _LazyTerminalDeviceMap:
    """Name-keyed terminal device map that creates device objects only on demand."""

    __slots__ = (
        "table",
        "names",
        "rows",
        "cols",
        "extra_cols",
        "node_by_idx",
        "_name_to_row",
        "_name_to_pos",
        "_cache",
    )

    def __init__(
        self,
        table: np.ndarray,
        names: Sequence[str],
        rows: np.ndarray,
        cols: Dict[str, int],
        extra_cols: Sequence[Tuple[str, int]],
        node_by_idx: Dict[int, object],
    ) -> None:
        self.table = table
        self.names = tuple(str(name) for name in names)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = cols
        self.extra_cols = tuple(extra_cols)
        self.node_by_idx = node_by_idx
        self._name_to_row = {name: int(row) for name, row in zip(self.names, self.rows)}
        self._name_to_pos = {name: int(pos) for pos, name in enumerate(self.names)}
        self._cache: Dict[str, object] = {}

    @property
    def materialized_count(self) -> int:
        return len(self._cache)

    def __bool__(self) -> bool:
        return bool(self._name_to_row)

    def __len__(self) -> int:
        return len(self._name_to_row)

    def __iter__(self):
        return iter(self._name_to_row)

    def __contains__(self, key: str) -> bool:
        return key in self._name_to_row

    def _device_from_row(self, name: str, row_pos: int):
        values = self.table[int(row_pos)]
        cols = self.cols
        dev = _ACArrayObject()
        dev.idx = int(values[cols["idx"]])
        dev.name = str(name)
        dev.ppc_row = int(row_pos)
        dev.plan_pos = int(self._name_to_pos.get(str(name), -1))
        dev.i_node = int(values[cols["i_node"]])
        dev.j_node = int(values[cols["j_node"]])
        dev.run_stat = int(values[cols["run_stat"]])
        dev.is_alive = True
        dev.i_node_obj = self.node_by_idx.get(dev.i_node)
        dev.j_node_obj = self.node_by_idx.get(dev.j_node)
        dev.p = float(values[cols["p"]]) if "p" in cols else 0.0
        dev.q = float(values[cols["q"]]) if "q" in cols else 0.0
        dev.current = float(values[cols["current"]]) if "current" in cols else 0.0
        dev.i_p = float(values[cols["i_p"]]) if "i_p" in cols else 0.0
        dev.i_q = float(values[cols["i_q"]]) if "i_q" in cols else 0.0
        dev.i_c = float(values[cols["i_c"]]) if "i_c" in cols else 0.0
        dev.j_p = float(values[cols["j_p"]]) if "j_p" in cols else 0.0
        dev.j_q = float(values[cols["j_q"]]) if "j_q" in cols else 0.0
        dev.j_c = float(values[cols["j_c"]]) if "j_c" in cols else 0.0
        for attr, col in self.extra_cols:
            setattr(dev, attr, float(values[col]))
        if "status" in cols:
            dev.status = int(values[cols["status"]])
        return dev

    def __getitem__(self, key: str):
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        row_pos = self._name_to_row[key]
        device = self._device_from_row(key, row_pos)
        self._cache[key] = device
        return device

    def get(self, key: str, default=None):
        return self[key] if key in self._name_to_row else default

    def keys(self):
        return self._name_to_row.keys()

    def values(self):
        return [self[name] for name in self._name_to_row]

    def items(self):
        return [(name, self[name]) for name in self._name_to_row]


class _LazySingleDeviceOrder:
    """Index-ordered single-terminal device view that materializes objects on access."""

    __slots__ = ("devices", "names")

    def __init__(self, devices, names: Optional[Sequence[str]] = None) -> None:
        self.devices = devices
        self.names = tuple(str(name) for name in (devices.names if names is None else names))

    @property
    def materialized_count(self) -> int:
        return self.devices.materialized_count

    def __bool__(self) -> bool:
        return bool(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        for name in self.names:
            yield self.devices[name]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.devices[name] for name in self.names[index]]
        return self.devices[self.names[index]]


class _LazySingleDeviceMap:
    """Name-keyed generator/load/shunt map backed by PPC rows."""

    __slots__ = ("table", "names", "rows", "cols", "device_kind", "node_by_idx", "_name_to_pos", "_cache")

    def __init__(
        self,
        table: np.ndarray,
        names: Sequence[str],
        rows: np.ndarray,
        cols: Dict[str, int],
        device_kind: str,
        node_by_idx: Dict[int, object],
    ) -> None:
        self.table = table
        self.names = tuple(str(name) for name in names)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = cols
        self.device_kind = str(device_kind)
        self.node_by_idx = node_by_idx
        self._name_to_pos = {name: int(pos) for pos, name in enumerate(self.names)}
        self._cache: Dict[str, object] = {}

    @property
    def materialized_count(self) -> int:
        return len(self._cache)

    def order(self, names: Optional[Sequence[str]] = None) -> _LazySingleDeviceOrder:
        return _LazySingleDeviceOrder(self, names)

    def __bool__(self) -> bool:
        return bool(self._name_to_pos)

    def __len__(self) -> int:
        return len(self._name_to_pos)

    def __iter__(self):
        return iter(self._name_to_pos)

    def __contains__(self, key: str) -> bool:
        return key in self._name_to_pos

    def _device_from_position(self, pos: int):
        values = self.table[int(self.rows[int(pos)])]
        cols = self.cols
        name = self.names[int(pos)]
        dev = _ACArrayObject()
        dev.idx = int(values[cols["idx"]])
        dev.name = str(name)
        dev.run_stat = int(values[cols["run_stat"]])
        dev.is_alive = True
        dev.node = int(values[cols["node"]])
        dev.node_obj = self.node_by_idx.get(dev.node)
        dev.p = float(values[cols["p"]]) if "p" in cols else 0.0
        dev.q = float(values[cols["q"]]) if "q" in cols else 0.0
        dev.current = float(values[cols["current"]]) if "current" in cols else 0.0
        if self.device_kind == "gen":
            ctrl_code = int(values[GEN_COLS["control_type"]])
            dev.control_type = _CTRL_NAME_BY_CODE[ctrl_code] if 0 <= ctrl_code < len(_CTRL_NAME_BY_CODE) else "PQ"
            dev.p_set = float(values[GEN_COLS["p_set"]])
            dev.q_set = float(values[GEN_COLS["q_set"]])
            dev.v_set = float(values[GEN_COLS["v_set"]])
            dev.alpha = float(values[GEN_COLS["alpha"]])
        elif self.device_kind == "load":
            dev.pbase = float(values[LOAD_COLS["pbase"]])
            dev.pv0 = float(values[LOAD_COLS["pv0"]])
            dev.pv1 = float(values[LOAD_COLS["pv1"]])
            dev.pv2 = float(values[LOAD_COLS["pv2"]])
            dev.qbase = float(values[LOAD_COLS["qbase"]])
            dev.qv0 = float(values[LOAD_COLS["qv0"]])
            dev.qv1 = float(values[LOAD_COLS["qv1"]])
            dev.qv2 = float(values[LOAD_COLS["qv2"]])
        elif self.device_kind == "shunt":
            shunt_code = int(values[SHUNT_COLS["control_type"]])
            dev.control_type = _SHUNT_NAME_BY_CODE[shunt_code] if 0 <= shunt_code < len(_SHUNT_NAME_BY_CODE) else "Q"
            dev.q_set = float(values[SHUNT_COLS["q_set"]])
            dev.g_set = float(values[SHUNT_COLS["g_set"]])
            dev.b_set = float(values[SHUNT_COLS["b_set"]])
            dev.v_set = float(values[SHUNT_COLS["v_set"]])
        return dev

    def __getitem__(self, key: str):
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pos = self._name_to_pos[key]
        device = self._device_from_position(pos)
        self._cache[key] = device
        return device

    def get(self, key: str, default=None):
        return self[key] if key in self._name_to_pos else default

    def keys(self):
        return self._name_to_pos.keys()

    def values(self):
        return [self[name] for name in self._name_to_pos]

    def items(self):
        return [(name, self[name]) for name in self._name_to_pos]


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
        self.diff_step = self.params.diff_step
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
        self._array_only_runtime = False
        self.matrix_dump_dir = Path(matrix_dump_dir) if matrix_dump_dir is not None else None
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
        if bool(getattr(self, "_array_only_runtime", False)):
            return MeasurementTableView(table, normalized=normalized)
        return TableBackedMeasurementList(table, normalized=normalized)

    def _ppc_runtime_node_and_device_context(self, ppc: Dict, topology_arrays) -> None:
        """Initialize SE runtime arrays from PPC/topology without network device objects."""
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        bus_names = np.asarray(ppc["bus_name"], dtype=object)
        active_bus_pos = np.flatnonzero(topology_arrays.bus_alive_mask).astype(np.int32, copy=False)
        bus_solver_pos = np.full(len(topology_arrays.bus_ids), -1, dtype=np.int32)
        if active_bus_pos.size:
            bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
        self._ac_bus_solver_pos_by_topology_bus = bus_solver_pos

        islands = []
        for island_idx, is_alive in zip(topology_arrays.island_ids, topology_arrays.island_alive_mask):
            islands.append(
                SimpleNamespace(
                    idx=int(island_idx),
                    is_alive=bool(is_alive),
                    buses=[],
                    slack_nodes=[],
                )
            )
        self._ac_islands = islands

        self.nodes = []
        bus_obj_by_topology_pos: Dict[int, object] = {}
        for bus_pos in active_bus_pos:
            bus_pos_int = int(bus_pos)
            start = int(topology_arrays.bus_node_offsets[bus_pos_int])
            end = int(topology_arrays.bus_node_offsets[bus_pos_int + 1])
            if end <= start:
                continue
            node_row = int(topology_arrays.bus_node_indices[start])
            row = bus[node_row]
            obj = _ACArrayObject()
            obj.idx = int(topology_arrays.bus_ids[bus_pos_int])
            obj.name = str(bus_names[node_row])
            obj.vbase = float(row[BUS_COLS["vbase"]])
            obj.voltage = float(row[BUS_COLS["voltage"]])
            obj.angle = float(row[BUS_COLS["angle"]])
            obj.run_stat = 1
            obj.is_alive = True
            island_pos = int(topology_arrays.bus_to_island_pos[bus_pos_int])
            obj.isl = int(topology_arrays.island_ids[island_pos]) if island_pos >= 0 else 0
            obj.isl_obj = islands[island_pos] if 0 <= island_pos < len(islands) else None
            obj.bus = obj.idx
            obj.bus_obj = obj
            bus_obj_by_topology_pos[bus_pos_int] = obj
            self.nodes.append(obj)
            if obj.isl_obj is not None:
                obj.isl_obj.buses.append(obj)

        for island_pos, ref_bus_pos in enumerate(np.asarray(topology_arrays.island_reference_bus_pos, dtype=np.int32)):
            bus_obj = bus_obj_by_topology_pos.get(int(ref_bus_pos))
            if bus_obj is not None and 0 <= island_pos < len(islands):
                islands[island_pos].slack_nodes.append(bus_obj)

        self.node_pos = {}
        self.node_by_name = {}
        self.node_by_idx = {}
        self._node_vbase_by_idx: Dict[int, float] = {}
        self._node_voltage_by_idx: Dict[int, float] = {}
        self._node_angle_by_idx: Dict[int, float] = {}
        for node_row, row in enumerate(bus):
            bus_pos = int(topology_arrays.node_to_bus_pos[node_row])
            if bus_pos < 0 or bus_pos >= bus_solver_pos.size:
                continue
            solver_pos = int(bus_solver_pos[bus_pos])
            if solver_pos < 0:
                continue
            node_idx = int(row[BUS_COLS["idx"]])
            node_name = str(bus_names[node_row])
            bus_obj = self.nodes[solver_pos]
            self.node_pos[node_idx] = solver_pos
            self.node_by_idx[node_idx] = bus_obj
            self.node_by_name[node_name] = bus_obj
            self.node_by_name.setdefault(str(bus_obj.name), bus_obj)
            self._node_vbase_by_idx[node_idx] = float(row[BUS_COLS["vbase"]])
            self._node_voltage_by_idx[node_idx] = float(row[BUS_COLS["voltage"]])
            self._node_angle_by_idx[node_idx] = float(row[BUS_COLS["angle"]])

        self.file_theta = np.asarray([float(node.angle) for node in self.nodes], dtype=np.float64)
        self.file_voltage = np.asarray([float(node.voltage) for node in self.nodes], dtype=np.float64)

        def terminal_devices(table_name: str, name_key: str, device_key: str, cols: Dict[str, int], extra_cols=()):
            table, _device_topology, rows = self._ppc_sorted_alive_rows_for(ppc, table_name, device_key, cols["idx"])
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            devices = []
            for row_pos, name in zip(rows, names):
                values = table[int(row_pos)]
                dev = _ACArrayObject()
                dev.idx = int(values[cols["idx"]])
                dev.name = str(name)
                dev.i_node = int(values[cols["i_node"]])
                dev.j_node = int(values[cols["j_node"]])
                dev.run_stat = int(values[cols["run_stat"]])
                dev.is_alive = True
                dev.i_node_obj = self.node_by_idx.get(dev.i_node)
                dev.j_node_obj = self.node_by_idx.get(dev.j_node)
                dev.p = float(values[cols["p"]]) if "p" in cols else 0.0
                dev.q = float(values[cols["q"]]) if "q" in cols else 0.0
                dev.current = float(values[cols["current"]]) if "current" in cols else 0.0
                dev.i_p = float(values[cols["i_p"]]) if "i_p" in cols else 0.0
                dev.i_q = float(values[cols["i_q"]]) if "i_q" in cols else 0.0
                dev.i_c = float(values[cols["i_c"]]) if "i_c" in cols else 0.0
                dev.j_p = float(values[cols["j_p"]]) if "j_p" in cols else 0.0
                dev.j_q = float(values[cols["j_q"]]) if "j_q" in cols else 0.0
                dev.j_c = float(values[cols["j_c"]]) if "j_c" in cols else 0.0
                for attr, col in extra_cols:
                    setattr(dev, attr, float(values[col]))
                if "status" in cols:
                    dev.status = int(values[cols["status"]])
                devices.append(dev)
            return devices

        def terminal_device_map(table_name: str, name_key: str, device_key: str, cols: Dict[str, int], extra_cols=()):
            table, _device_topology, rows = self._ppc_sorted_alive_rows_for(ppc, table_name, device_key, cols["idx"])
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            return _LazyTerminalDeviceMap(table, names, rows, cols, extra_cols, self.node_by_idx)

        self.branch_by_name = terminal_device_map(
            "branch",
            "branch_name",
            "branch",
            BRANCH_COLS,
            (("r", BRANCH_COLS["r"]), ("x", BRANCH_COLS["x"]), ("b", BRANCH_COLS["b"])),
        )
        self.transformer_by_name = terminal_device_map(
            "transformer",
            "transformer_name",
            "transformer",
            TRANSFORMER_COLS,
            (
                ("r", TRANSFORMER_COLS["r"]),
                ("x", TRANSFORMER_COLS["x"]),
                ("gt", TRANSFORMER_COLS["gt"]),
                ("bt", TRANSFORMER_COLS["bt"]),
                ("tap", TRANSFORMER_COLS["tap"]),
                ("shift", TRANSFORMER_COLS["shift"]),
            ),
        )
        self.zero_branch_by_name = {
            dev.name: dev
            for dev in terminal_devices("zero_branch", "zero_branch_name", "zero_branch", ZERO_BRANCH_COLS)
        }
        self.switch_by_name = {
            dev.name: dev
            for dev in terminal_devices("switch", "switch_name", "switch", SWITCH_COLS)
        }
        self.break_by_name = {
            dev.name: dev
            for dev in terminal_devices("break", "break_name", "break", BREAK_COLS)
        }
        self.zero_branches = sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
        self.switches = sorted(self.switch_by_name.values(), key=lambda item: item.idx)
        self.breakers = sorted(self.break_by_name.values(), key=lambda item: item.idx)

        def single_device_arrays(table_name: str, name_key: str, device_key: str, cols: Dict[str, int]):
            table, _device_topology, rows = self._ppc_sorted_alive_rows_for(ppc, table_name, device_key, cols["idx"])
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            if rows.size:
                nodes = table[rows.astype(np.intp, copy=False), cols["node"]].astype(np.int64, copy=False)
                valid = np.fromiter((int(node) in self.node_pos for node in nodes), dtype=bool, count=nodes.size)
                if not np.all(valid):
                    rows = rows[valid]
                    names = names[valid]
                    nodes = nodes[valid]
                pos = np.asarray([self.node_pos[int(node)] for node in nodes], dtype=np.int32)
            else:
                pos = np.asarray([], dtype=np.int32)
            return table, rows.astype(np.int64, copy=False), names.astype(object, copy=False), pos

        gen_table, gen_rows, gen_names, gen_pos = single_device_arrays(
            "gen",
            "gen_name",
            "gen",
            GEN_COLS,
        )
        load_table, load_rows, load_names, load_pos = single_device_arrays(
            "load",
            "load_name",
            "load",
            LOAD_COLS,
        )
        shunt_table, shunt_rows, shunt_names, shunt_pos = single_device_arrays(
            "shunt",
            "shunt_name",
            "shunt",
            SHUNT_COLS,
        )
        self._ac_generator_rows = gen_rows
        self._ac_load_rows = load_rows
        self._ac_shunt_rows = shunt_rows
        self.generator_name_array = gen_names
        self.load_name_array = load_names
        self.shunt_name_array = shunt_names
        self.generator_pos_array = gen_pos
        self.load_pos_array = load_pos
        self.shunt_pos_array = shunt_pos
        self.generator_by_name = _LazySingleDeviceMap(gen_table, gen_names, gen_rows, GEN_COLS, "gen", self.node_by_idx)
        self.load_by_name = _LazySingleDeviceMap(load_table, load_names, load_rows, LOAD_COLS, "load", self.node_by_idx)
        self.shunt_by_name = _LazySingleDeviceMap(shunt_table, shunt_names, shunt_rows, SHUNT_COLS, "shunt", self.node_by_idx)
        self.generator_order = self.generator_by_name.order()
        self.load_order = self.load_by_name.order()
        self.shunt_compensators = self.shunt_by_name.order()
        self.voltage_control_shunt_order = self.shunt_by_name.order(())

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
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        self.network = network if network is not None else self._load_network(self.e_file)
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            self.meas_ppc = copy_meas_ppc(build_meas_ppc_from_e_file(self.meas_file))
            if bool(getattr(self, "_array_only_runtime", False)):
                self.measurements = self._measurement_sequence_from_table(
                    measurement_table_from_meas_ppc(self.meas_ppc),
                    normalized=bool(self.meas_ppc.get("normalized", False)),
                )
            else:
                self.measurements = measurement_list_from_meas_ppc(self.meas_ppc)
        elif isinstance(measurements, dict):
            self.meas_ppc = copy_meas_ppc(measurements)
            if bool(getattr(self, "_array_only_runtime", False)):
                self.measurements = self._measurement_sequence_from_table(
                    measurement_table_from_meas_ppc(self.meas_ppc),
                    normalized=bool(self.meas_ppc.get("normalized", False)),
                )
            else:
                self.measurements = measurement_list_from_meas_ppc(self.meas_ppc)
        elif isinstance(measurements, MeasurementList):
            self.measurements = measurements
            self.meas_ppc = self._measurement_table_to_meas_ppc(_measurement_table_from_measurements(measurements))
        else:
            self.measurements = list(measurements)
            self.meas_ppc = self._measurement_table_to_meas_ppc(_measurement_table_from_measurements(self.measurements))
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
        self.p_scale = float(base["p_scale"])
        self.i_scale = float(base["i_scale"])

        topology_arrays = ppc.get("_topology_arrays")
        if topology_arrays is None:
            self._warn_required_runtime_missing("PPC topology arrays", "prepare")
        stage_start = time.perf_counter()
        self._ppc_runtime_node_and_device_context(ppc, topology_arrays)
        self._record_profile_time("init.ppc_runtime_context", time.perf_counter() - stage_start)
        if not self.nodes:
            raise RuntimeError("No alive AC nodes are available for state estimation")
        self.n_nodes = len(self.nodes)
        self._defer_prepare_finalize_pending = bool(defer_prepare_finalize)
        if defer_prepare_finalize:
            self.power_flow_seed_converged = False
            self.targeted_observability_pseudo_count = 0
            return self
        self.finalize_prepare(prepare_active_measurements=prepare_active_measurements)
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        self._prepared = True
        return self

    def _build_state_meta(self) -> List[StateMeta]:
        state_meta: List[StateMeta] = []
        def inverse_plan_pos(plan_values, size: int) -> np.ndarray:
            values = np.asarray(plan_values, dtype=np.int64)
            out = np.full(max(int(size), 0), -1, dtype=np.int64)
            if values.size == 0 or out.size == 0:
                return out
            valid = (values >= 0) & (values < out.size)
            if np.any(valid):
                out[values[valid].astype(np.intp, copy=False)] = np.flatnonzero(valid).astype(np.int64, copy=False)
            return out

        def plan_pos_for_state(inverse_values: np.ndarray, state_pos: int) -> int:
            pos = int(state_pos)
            if pos < 0 or pos >= inverse_values.size:
                return -1
            return int(inverse_values[pos])

        node_plan_by_solver_pos = inverse_plan_pos(
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
            len(getattr(self, "nodes", ())),
        )
        zero_plan_by_current_pos = inverse_plan_pos(
            getattr(self, "_ac_zero_branch_plan_current_pos", np.asarray([], dtype=np.int64)),
            len(getattr(self, "zero_current_devices", ())),
        )
        break_plan_by_current_pos = inverse_plan_pos(
            getattr(self, "_ac_break_plan_current_pos", np.asarray([], dtype=np.int64)),
            len(getattr(self, "zero_current_devices", ())),
        )
        gen_plan_by_state_idx = inverse_plan_pos(
            getattr(self, "_ac_generator_plan_index", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_generator_power", 0)),
        )
        load_plan_by_state_idx = inverse_plan_pos(
            getattr(self, "_ac_load_plan_index", np.asarray([], dtype=np.int64)),
            int(getattr(self, "n_load_power", 0)),
        )

        for pos in self.angle_state_pos:
            node = self.nodes[int(pos)]
            state_meta.append(
                StateMeta(
                    "ac",
                    "angle",
                    "ACNode",
                    node.name,
                    component="theta",
                    legacy_label=f"theta:{node.name}",
                    device_pos=plan_pos_for_state(node_plan_by_solver_pos, int(pos)),
                    device_type_code=DEVICE_TYPE_CODES_ACNODE,
                    meas_type_code=MEAS_TYPE_CODES_ANGLE,
                )
            )
        for pos in self.voltage_state_pos:
            node = self.nodes[int(pos)]
            state_meta.append(
                StateMeta(
                    "ac",
                    "voltage",
                    "ACNode",
                    node.name,
                    component="magnitude",
                    legacy_label=f"V:{node.name}",
                    device_pos=plan_pos_for_state(node_plan_by_solver_pos, int(pos)),
                    device_type_code=DEVICE_TYPE_CODES_ACNODE,
                    meas_type_code=MEAS_TYPE_CODES_V,
                )
            )
        for current_pos, (kind, dev) in enumerate(self.zero_current_devices):
            device_type = "ACZeroBranch" if kind == "Z" else "ACBreak"
            device_type_code = DEVICE_TYPE_CODES_ACZEROBRANCH if kind == "Z" else DEVICE_TYPE_CODES_ACBREAK
            device_pos = (
                plan_pos_for_state(zero_plan_by_current_pos, current_pos)
                if kind == "Z"
                else plan_pos_for_state(break_plan_by_current_pos, current_pos)
            )
            state_kind = "zero_current" if kind == "Z" else "break_current"
            state_meta.append(
                StateMeta(
                    "ac",
                    state_kind,
                    device_type,
                    dev.name,
                    component="re",
                    legacy_label=f"I_{kind}_RE:{dev.name}",
                    device_pos=device_pos,
                    device_type_code=device_type_code,
                    meas_type_code=MEAS_TYPE_CODES_I_FROM,
                )
            )
        for current_pos, (kind, dev) in enumerate(self.zero_current_devices):
            device_type = "ACZeroBranch" if kind == "Z" else "ACBreak"
            device_type_code = DEVICE_TYPE_CODES_ACZEROBRANCH if kind == "Z" else DEVICE_TYPE_CODES_ACBREAK
            device_pos = (
                plan_pos_for_state(zero_plan_by_current_pos, current_pos)
                if kind == "Z"
                else plan_pos_for_state(break_plan_by_current_pos, current_pos)
            )
            state_kind = "zero_current" if kind == "Z" else "break_current"
            state_meta.append(
                StateMeta(
                    "ac",
                    state_kind,
                    device_type,
                    dev.name,
                    component="im",
                    legacy_label=f"I_{kind}_IM:{dev.name}",
                    device_pos=device_pos,
                    device_type_code=device_type_code,
                    meas_type_code=MEAS_TYPE_CODES_I_FROM,
                )
            )
        gen_names = tuple(str(name) for name in getattr(self, "generator_name_array", ()))
        load_names = tuple(str(name) for name in getattr(self, "load_name_array", ()))
        shunt_q_names = tuple(str(name) for name in getattr(self, "voltage_control_shunt_name_array", ()))
        if not gen_names:
            gen_names = tuple(gen.name for gen in self.generator_order)
        if not load_names:
            load_names = tuple(load.name for load in self.load_order)
        if not shunt_q_names:
            shunt_q_names = tuple(shunt.name for shunt in self.voltage_control_shunt_order)
        for idx, name in enumerate(gen_names):
            device_pos = plan_pos_for_state(gen_plan_by_state_idx, idx)
            state_meta.append(
                StateMeta(
                    "ac",
                    "generator_p",
                    "ACGenerator",
                    name,
                    component="p",
                    legacy_label=f"P_GEN:{name}",
                    device_pos=device_pos,
                    device_type_code=DEVICE_TYPE_CODES_ACGENERATOR,
                    meas_type_code=MEAS_TYPE_CODES_P_GEN,
                )
            )
        for idx, name in enumerate(gen_names):
            device_pos = plan_pos_for_state(gen_plan_by_state_idx, idx)
            state_meta.append(
                StateMeta(
                    "ac",
                    "generator_q",
                    "ACGenerator",
                    name,
                    component="q",
                    legacy_label=f"Q_GEN:{name}",
                    device_pos=device_pos,
                    device_type_code=DEVICE_TYPE_CODES_ACGENERATOR,
                    meas_type_code=MEAS_TYPE_CODES_Q_GEN,
                )
            )
        for idx, name in enumerate(load_names):
            device_pos = plan_pos_for_state(load_plan_by_state_idx, idx)
            state_meta.append(
                StateMeta(
                    "ac",
                    "load_p",
                    "ACLoad",
                    name,
                    component="p",
                    legacy_label=f"P_LOAD:{name}",
                    device_pos=device_pos,
                    device_type_code=DEVICE_TYPE_CODES_ACLOAD,
                    meas_type_code=MEAS_TYPE_CODES_P_LOAD,
                )
            )
        for idx, name in enumerate(load_names):
            device_pos = plan_pos_for_state(load_plan_by_state_idx, idx)
            state_meta.append(
                StateMeta(
                    "ac",
                    "load_q",
                    "ACLoad",
                    name,
                    component="q",
                    legacy_label=f"Q_LOAD:{name}",
                    device_pos=device_pos,
                    device_type_code=DEVICE_TYPE_CODES_ACLOAD,
                    meas_type_code=MEAS_TYPE_CODES_Q_LOAD,
                )
            )
        for idx, name in enumerate(shunt_q_names):
            state_meta.append(
                StateMeta(
                    "ac",
                    "shunt_q",
                    "ACShuntCompensator",
                    name,
                    component="q",
                    legacy_label=f"Q_SHUNT:{name}",
                    device_pos=int(idx),
                )
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
            labels = [meta.legacy_label for meta in self.state_meta]
            cache = labels if all(labels) else state_labels_from_metadata(self.state_meta)
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
        self._defer_prepare_finalize_pending = False
        stage_start = time.perf_counter()
        self._build_measurement_device_indexes()
        self._record_profile_time("init.measurement_device_indexes", time.perf_counter() - stage_start)
        if not measurements_already_normalized:
            stage_start = time.perf_counter()
            self._convert_measurements_to_pu()
            self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
        self.power_flow_seed_converged = False
        if not self.flat_start:
            seed_start = time.perf_counter()
            stage_start = time.perf_counter()
            self._apply_measurement_seed_to_network()
            self._record_profile_time("seed.apply_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self.power_flow_seed_converged = bool(self._run_power_flow_seed(self.network, self.params, self.e_file))
            self._record_profile_time("seed.lf", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_file_state_from_network()
            self._record_profile_time("seed.refresh_file_state", time.perf_counter() - stage_start)
            self._record_profile_time("seed.total", time.perf_counter() - seed_start)
        stage_start = time.perf_counter()
        self.node_voltage_measurements = self._node_voltage_measurements()
        self.node_degrees = self._node_incident_degrees()
        self.references = self._select_reference_nodes()
        self.ref_idx = {self.node_pos[node.idx] for node in self.references}
        self.reference_voltage_by_pos = {
            self.node_pos[node.idx]: self.node_voltage_measurements[node.idx]
            for node in self.references
            if node.idx in self.node_voltage_measurements and node.idx in self.node_pos
        }
        self.reference_angle_by_pos = self._reference_angle_offsets()
        self._rebase_angle_measurements()
        self._build_zero_tie_state_layout()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        self.zero_current_devices = (
            [("Z", zbr) for zbr in self.zero_branches]
            + [("B", brk) for brk in self.breakers]
        )
        self.zero_current_pos = {(kind, dev.name): pos for pos, (kind, dev) in enumerate(self.zero_current_devices)}
        self.zero_branch_pos = {zbr.name: self.zero_current_pos[("Z", zbr.name)] for zbr in self.zero_branches}
        self.switch_pos = {}
        self.break_pos = {brk.name: self.zero_current_pos[("B", brk.name)] for brk in self.breakers}
        self.zero_current_by_node = {pos: [] for pos in range(len(self.nodes))}
        for idx, (_, dev) in enumerate(self.zero_current_devices):
            if dev.i_node in self.node_pos:
                self.zero_current_by_node[self.node_pos[dev.i_node]].append((idx, True))
            if dev.j_node in self.node_pos:
                self.zero_current_by_node[self.node_pos[dev.j_node]].append((idx, False))
        self.zero_current_i = np.asarray(
            [self.node_pos[dev.i_node] for _, dev in self.zero_current_devices],
            dtype=np.int32,
        )
        self.zero_current_j = np.asarray(
            [self.node_pos[dev.j_node] for _, dev in self.zero_current_devices],
            dtype=np.int32,
        )
        ppc = self._ac_ppc_dict()
        gen_rows = self._required_array_attr("_ac_generator_rows", dtype=np.int64, context="finalize_prepare")
        load_rows = self._required_array_attr("_ac_load_rows", dtype=np.int64, context="finalize_prepare")
        self.initial_gen_p_array, self.initial_gen_q_array = self._ppc_generator_pseudo_power_arrays(
            ppc,
            gen_rows,
        )
        load_voltage = (
            self.file_voltage[self.load_pos_array.astype(np.intp, copy=False)]
            if self.load_pos_array.size
            else np.asarray([], dtype=np.float64)
        )
        self.initial_load_p_array, self.initial_load_q_array = self._ppc_load_pseudo_power_arrays(
            ppc,
            load_rows,
            load_voltage,
        )
        self.n_generator_power = int(gen_rows.size)
        self.n_load_power = int(load_rows.size)
        self.n_switch_current = len(self.zero_current_devices)
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
        shunt_rows = self._required_array_attr("_ac_shunt_rows", dtype=np.int64, context="finalize_prepare")
        if shunt_rows.size:
            shunt_table = np.asarray(ppc["shunt"], dtype=np.float64)
            control_type = shunt_table[shunt_rows.astype(np.intp, copy=False), SHUNT_COLS["control_type"]].astype(
                np.int64,
                copy=False,
            )
            v_mask = control_type == SHUNT_V
            voltage_control_rows = shunt_rows[v_mask]
            voltage_control_names = np.asarray(self.shunt_name_array, dtype=object)[v_mask]
            self.shunt_q_pos_array = np.asarray(self.shunt_pos_array, dtype=np.int32)[v_mask]
            self.initial_shunt_q_array = shunt_table[
                voltage_control_rows.astype(np.intp, copy=False),
                SHUNT_COLS["q"],
            ].astype(np.float64, copy=False)
        else:
            voltage_control_rows = np.asarray([], dtype=np.int64)
            voltage_control_names = np.asarray([], dtype=object)
            self.shunt_q_pos_array = np.asarray([], dtype=np.int32)
            self.initial_shunt_q_array = np.asarray([], dtype=np.float64)
        self.voltage_control_shunt_name_array = voltage_control_names
        self.voltage_control_shunt_order = self.shunt_by_name.order(voltage_control_names)
        self.n_shunt_q = int(voltage_control_rows.size)
        self.shunt_q_idx_array = np.arange(self.n_shunt_q, dtype=np.int32)
        self.generator_balance_minus_ones = -np.ones(self.n_generator_power, dtype=np.float64)
        self.load_balance_ones = np.ones(self.n_load_power, dtype=np.float64)
        self.shunt_balance_minus_ones = -np.ones(self.n_shunt_q, dtype=np.float64)
        self.base_switch_re = self.n_angle + self.n_voltage
        self.base_switch_im = self.base_switch_re + self.n_switch_current
        self.base_gen_p = self.base_switch_im + self.n_switch_current
        self.base_gen_q = self.base_gen_p + self.n_generator_power
        self.base_load_p = self.base_gen_q + self.n_generator_power
        self.base_load_q = self.base_load_p + self.n_load_power
        self.base_shunt_q = self.base_load_q + self.n_load_power
        self.n_state = self.base_shunt_q + self.n_shunt_q
        gen_names = tuple(str(name) for name in getattr(self, "generator_name_array", ()))
        load_names = tuple(str(name) for name in getattr(self, "load_name_array", ()))
        shunt_q_names = tuple(str(name) for name in getattr(self, "voltage_control_shunt_name_array", ()))
        if not gen_names:
            gen_names = tuple(gen.name for gen in self.generator_order)
        if not load_names:
            load_names = tuple(load.name for load in self.load_order)
        if not shunt_q_names:
            shunt_q_names = tuple(shunt.name for shunt in self.voltage_control_shunt_order)
        self.gen_p_col_by_name = {name: self.base_gen_p + idx for idx, name in enumerate(gen_names)}
        self.gen_q_col_by_name = {name: self.base_gen_q + idx for idx, name in enumerate(gen_names)}
        self.load_p_col_by_name = {name: self.base_load_p + idx for idx, name in enumerate(load_names)}
        self.load_q_col_by_name = {name: self.base_load_q + idx for idx, name in enumerate(load_names)}
        self._state_meta_cache = None
        self._state_labels_cache = None

        stage_start = time.perf_counter()
        if hasattr(self, "_ac_measurement_plan_device_pos_by_type_code"):
            self._refresh_measurement_plan_state_columns()
        else:
            self._build_measurement_plan_lookup_arrays()
        self._record_profile_time("init.measurement_plan_lookup", time.perf_counter() - stage_start)

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
        self.targeted_observability_pseudo_count = 0
        if prepare_active_measurements:
            stage_start = time.perf_counter()
            self._add_pseudo_power_measurements()
            self._add_power_balance_constraint_measurements()
            self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_active_measurement_indexes()
            self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
            self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        else:
            self.active_measurements = MeasurementList(
                [],
                normalized=getattr(self.measurements, "normalized", False),
            )
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
            self._balance_measurement_plan_cache = {}
            self._active_branch_transformer_vector_plan = None
            self._active_simple_jacobian_plan = None
            self._active_zero_current_vector_plan = None
            self._active_generator_measurement_plan = None
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
            return {
                "branch_transformer": self._active_branch_transformer_vector_plan,
                "zero_current": self._active_zero_current_vector_plan,
                "simple": self._active_simple_jacobian_plan,
                "balance": self._active_balance_measurement_plan,
                "generator": self._active_generator_measurement_plan,
            }
        return {
            "branch_transformer": self._build_branch_transformer_vector_plan_from_table(
                plan_tables["branch_transformer"]
            ),
            "zero_current": self._build_zero_current_vector_plan_from_table(plan_tables["zero_current"]),
            "simple": self._build_simple_jacobian_plan_from_table(plan_tables["simple"]),
            "balance": self._build_balance_measurement_plan_from_table(plan_tables["balance"]),
            "generator": self._build_generator_measurement_plan_from_table(plan_tables["generator"]),
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

    def _cache_observability_matrix(
        self,
        result: ObservabilityResult,
        x: np.ndarray,
        measurements: Sequence[Measurement],
        H,
    ) -> None:
        lower_normal_plan = None
        if self._measurement_plan_tables_are_active(measurements) and issparse(H):
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
            materialize_measurements=not bool(getattr(self, "_array_only_runtime", False)),
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
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )

    def _incremental_update_active_measurement_indexes(self, appended_measurements: Sequence[Measurement]) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            self._warn_missing_required_active_measurements("incremental active measurement update")
            return False
        if any((not meas.valid) or float(meas.weight) <= 0.0 for meas in appended_measurements):
            return False
        master_table = getattr(self, "measurement_table", getattr(self.measurements, "table", None))
        active_table = getattr(self, "active_measurement_table", getattr(self.active_measurements, "table", None))
        if master_table is None or active_table is None:
            return False
        if len(master_table.idx) != len(self.measurements) - len(appended_measurements):
            return False
        if len(active_table.idx) != len(self.active_measurements):
            return False

        appended_list = MeasurementList(
            list(appended_measurements),
            _measurement_table_from_measurements(appended_measurements),
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._build_measurement_device_indexes(appended_list)
        self.measurement_table = concat_measurement_tables(master_table, appended_list.table)
        self.measurements.table = self.measurement_table
        self._build_measurement_device_indexes(self.measurements)
        active_view = append_active_measurement_view(
            build_active_measurement_view(
                self.active_measurements,
                table_builder=_measurement_table_from_measurements,
                materialize_measurements=not bool(getattr(self, "_array_only_runtime", False)),
            ),
            appended_list,
            source_row_start=len(master_table.idx),
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=not bool(getattr(self, "_array_only_runtime", False)),
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
        previous_balance_plan = getattr(self, "_active_balance_measurement_plan", None)
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        if previous_branch_plan is None:
            self._active_branch_transformer_vector_plan = self._branch_transformer_vector_plan(None)
        else:
            self._active_branch_transformer_vector_plan = self._merge_active_plan_dict(
                previous_branch_plan,
                self._build_branch_transformer_vector_plan(appended_list),
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
                self._build_simple_jacobian_plan(appended_list),
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
                self._build_zero_current_vector_plan(appended_list),
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
                self._build_generator_measurement_plan(appended_list),
                row_offset=row_offset,
                row_keys=("value_rows",),
            )
        self._generator_measurement_plan_cache[id(self.active_measurements)] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        if previous_balance_plan is None:
            self._active_balance_measurement_plan = self._balance_measurement_plan(None)
        else:
            self._active_balance_measurement_plan = self._merge_active_plan_dict(
                previous_balance_plan,
                self._build_balance_measurement_plan(appended_list),
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

    @staticmethod
    def _shrink_plan_rows(
        rows: Sequence[int],
        removed_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        row_array = np.asarray(rows, dtype=np.int64)
        keep = row_array != int(removed_pos)
        kept_rows = row_array[keep]
        kept_rows = kept_rows - (kept_rows > int(removed_pos))
        return kept_rows.astype(np.int64, copy=False), keep

    def _shrink_active_plan_group(
        self,
        plan: Dict[str, object],
        *,
        removed_pos: int,
        groups: Sequence[Tuple[str, Sequence[str]]],
        passthrough_keys: Sequence[str] = (),
    ) -> Dict[str, object]:
        shrunk: Dict[str, object] = {
            "handled_mask": np.delete(np.asarray(plan["handled_mask"], dtype=bool), int(removed_pos))
        }
        used_keys = ["handled_mask"]
        for row_key, value_keys in groups:
            shrunk_rows, keep = self._shrink_plan_rows(plan[row_key], removed_pos)
            shrunk[row_key] = shrunk_rows
            used_keys.append(row_key)
            for value_key in value_keys:
                shrunk[value_key] = np.asarray(plan[value_key])[keep]
                used_keys.append(value_key)
        for key in passthrough_keys:
            shrunk[key] = np.asarray(plan[key]).copy()
            used_keys.append(key)
        for key, value in plan.items():
            if key not in used_keys:
                shrunk[key] = np.asarray(value).copy()
        return shrunk

    def _shrink_balance_measurement_plan(
        self,
        plan: Dict[str, object],
        removed_pos: int,
    ) -> Dict[str, object]:
        shrunk_rows, keep = self._shrink_plan_rows(plan["rows"], removed_pos)
        shrunk_pos = np.asarray(plan["pos"], dtype=np.int64)[keep]
        shrunk_kind = np.asarray(plan["kind"], dtype=np.int64)[keep]
        p_row_by_pos = np.asarray(plan["p_row_by_pos"], dtype=np.int32).copy()
        q_row_by_pos = np.asarray(plan["q_row_by_pos"], dtype=np.int32).copy()
        for mapped in (p_row_by_pos, q_row_by_pos):
            removed_mask = mapped == int(removed_pos)
            mapped[removed_mask] = -1
            shift_mask = mapped > int(removed_pos)
            mapped[shift_mask] -= 1
        balance_pos = np.flatnonzero((p_row_by_pos >= 0) | (q_row_by_pos >= 0)).astype(np.int64, copy=False)
        y_balance, y_nodes, y_nodes_i64, y_conj = self._balance_y_arrays(balance_pos)
        return {
            "handled_mask": np.delete(np.asarray(plan["handled_mask"], dtype=bool), int(removed_pos)),
            "rows": shrunk_rows,
            "pos": shrunk_pos.astype(np.int64, copy=False),
            "pos_i64": np.asarray(shrunk_pos, dtype=np.int64),
            "kind": shrunk_kind.astype(np.int64, copy=False),
            "p_row_by_pos": p_row_by_pos,
            "q_row_by_pos": q_row_by_pos,
            "balance_pos": balance_pos,
            "balance_pos_i64": np.asarray(balance_pos, dtype=np.int64),
            "y_balance": y_balance,
            "y_nodes": y_nodes,
            "y_nodes_i64": y_nodes_i64,
            "y_conj": y_conj,
        }

    def _shrink_active_measurement_indexes(self, removed_pos: int) -> MeasurementList:
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        self.active_measurements = take_measurement_view(self.active_measurements, keep_rows)
        self.active_measurement_table = self.active_measurements.table
        self.measurement_table = self.active_measurement_table
        self.active_measurement_rows = np.arange(len(self.active_measurements), dtype=np.int64)
        self.active_z = np.asarray(self.active_measurement_table.value, dtype=np.float64)
        self.active_weight = np.asarray(self.active_measurement_table.weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(self.active_measurement_table.angle_mask, dtype=bool)
        self.active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = rows_by_device_type_code(self.active_measurement_table)
        self._active_measurement_plan_tables_cache = None
        if not hasattr(self, "n_state"):
            return self.active_measurements
        self._initial_observability_cache = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        self._observability_matrix_cache = None
        if hasattr(self, "_active_branch_transformer_vector_plan"):
            self._active_branch_transformer_vector_plan = self._shrink_active_plan_group(
                self._active_branch_transformer_vector_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("voltage_rows", ("voltage_pos", "voltage_cols")),
                    ("power_rows", ("power_is_p", "power_own", "power_other", "power_y_self", "power_y_mutual")),
                    ("current_rows", ("current_own", "current_other", "current_y_self", "current_y_mutual")),
                ),
            )
        if hasattr(self, "_active_simple_jacobian_plan"):
            self._active_simple_jacobian_plan = self._shrink_active_plan_group(
                self._active_simple_jacobian_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("value_voltage_rows", ("value_voltage_pos",)),
                    ("value_angle_rows", ("value_angle_pos",)),
                    ("value_voltage_diff_rows", ("value_voltage_diff_i", "value_voltage_diff_j")),
                    ("value_angle_diff_rows", ("value_angle_diff_i", "value_angle_diff_j")),
                    ("load_rows", ("load_pos", "load_index", "load_kind")),
                    ("scalar_rows", ("scalar_cols", "scalar_values")),
                ),
            )
        if hasattr(self, "_active_zero_current_vector_plan"):
            self._active_zero_current_vector_plan = self._shrink_active_plan_group(
                self._active_zero_current_vector_plan,
                removed_pos=int(removed_pos),
                groups=(
                    ("voltage_rows", ("voltage_pos",)),
                    ("angle_diff_rows", ("angle_diff_i", "angle_diff_j")),
                    ("voltage_diff_rows", ("voltage_diff_i", "voltage_diff_j")),
                    ("power_rows", ("power_is_p", "power_pos", "power_current_idx", "power_sign")),
                    ("current_rows", ("current_idx",)),
                    ("scalar_rows", ("scalar_cols", "scalar_values")),
                ),
            )
        if hasattr(self, "_active_generator_measurement_plan"):
            self._active_generator_measurement_plan = self._shrink_active_plan_group(
                self._active_generator_measurement_plan,
                removed_pos=int(removed_pos),
                groups=(("value_rows", ("value_kind", "value_pos", "value_index")),),
            )
        if hasattr(self, "_active_balance_measurement_plan"):
            self._active_balance_measurement_plan = self._shrink_balance_measurement_plan(
                self._active_balance_measurement_plan,
                int(removed_pos),
            )
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        branch_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
        simple_plan = getattr(self, "_active_simple_jacobian_plan", None)
        zero_plan = getattr(self, "_active_zero_current_vector_plan", None)
        generator_plan = getattr(self, "_active_generator_measurement_plan", None)
        balance_plan = getattr(self, "_active_balance_measurement_plan", None)
        if branch_plan is not None:
            self._branch_transformer_vector_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                branch_plan,
            )
        if simple_plan is not None:
            self._simple_jacobian_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                simple_plan,
            )
        if zero_plan is not None:
            self._zero_current_vector_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                zero_plan,
            )
        if generator_plan is not None:
            self._generator_measurement_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                generator_plan,
            )
        if balance_plan is not None:
            self._balance_measurement_plan_cache[id(self.active_measurements)] = (
                self.active_measurements,
                balance_plan,
            )
        if None not in (branch_plan, simple_plan, zero_plan, generator_plan, balance_plan):
            self.active_measurements_are_vectorized = bool(
                np.all(
                    branch_plan["handled_mask"]
                    | simple_plan["handled_mask"]
                    | zero_plan["handled_mask"]
                    | generator_plan["handled_mask"]
                    | balance_plan["handled_mask"]
                )
            )
        return self.active_measurements

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
        bus_by_idx = ACStateEstimator._row_by_idx(bus, BUS_COLS["idx"])
        gen = ppc.get("gen")
        load = ppc.get("load")

        def set_bus_voltage_by_idx(node_idx, value):
            row = bus_by_idx.get(int(node_idx))
            if row is not None:
                bus[row, BUS_COLS["voltage"]] = max(float(value), 0.0)

        for device_type_code, device_row, meas_type_code, value in seed_rows:
            device_type_code = int(device_type_code)
            device_row = int(device_row)
            meas_type_code = int(meas_type_code)
            value = float(value)
            if device_type_code == DEVICE_TYPE_CODES_ACNODE:
                if meas_type_code == MEAS_TYPE_CODES_V and 0 <= device_row < bus.shape[0]:
                    bus[device_row, BUS_COLS["voltage"]] = max(value, 0.0)
                continue
            if device_type_code == DEVICE_TYPE_CODES_ACGENERATOR and gen is not None:
                if device_row < 0 or device_row >= gen.shape[0]:
                    continue
                if meas_type_code == MEAS_TYPE_CODES_P_GEN:
                    gen[device_row, GEN_COLS["p_set"]] = value
                    gen[device_row, GEN_COLS["p"]] = value
                elif meas_type_code == MEAS_TYPE_CODES_Q_GEN:
                    gen[device_row, GEN_COLS["q_set"]] = value
                    gen[device_row, GEN_COLS["q"]] = value
                elif meas_type_code == MEAS_TYPE_CODES_V_GEN:
                    voltage = max(value, 0.0)
                    gen[device_row, GEN_COLS["v_set"]] = voltage
                    set_bus_voltage_by_idx(gen[device_row, GEN_COLS["node"]], voltage)
                elif meas_type_code == MEAS_TYPE_CODES_I_GEN:
                    gen[device_row, GEN_COLS["current"]] = value
                continue
            if device_type_code == DEVICE_TYPE_CODES_ACLOAD and load is not None:
                if device_row < 0 or device_row >= load.shape[0]:
                    continue
                if meas_type_code == MEAS_TYPE_CODES_P_LOAD:
                    load[device_row, LOAD_COLS["pbase"]] = 1.0
                    load[device_row, LOAD_COLS["pv0"]] = value
                    load[device_row, LOAD_COLS["pv1"]] = 0.0
                    load[device_row, LOAD_COLS["pv2"]] = 0.0
                    load[device_row, LOAD_COLS["p"]] = value
                elif meas_type_code == MEAS_TYPE_CODES_Q_LOAD:
                    load[device_row, LOAD_COLS["qbase"]] = 1.0
                    load[device_row, LOAD_COLS["qv0"]] = value
                    load[device_row, LOAD_COLS["qv1"]] = 0.0
                    load[device_row, LOAD_COLS["qv2"]] = 0.0
                    load[device_row, LOAD_COLS["q"]] = value
                elif meas_type_code == MEAS_TYPE_CODES_V_LOAD:
                    set_bus_voltage_by_idx(load[device_row, LOAD_COLS["node"]], value)
                elif meas_type_code == MEAS_TYPE_CODES_I_LOAD:
                    load[device_row, LOAD_COLS["current"]] = value

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
        bus_solver_pos = getattr(self, "_ac_bus_solver_pos_by_topology_bus", None)
        if bus_solver_pos is None:
            self._warn_required_runtime_missing("_ac_bus_solver_pos_by_topology_bus", "file-state refresh")
        file_theta = np.zeros(len(self.nodes), dtype=np.float64)
        file_voltage = np.ones(len(self.nodes), dtype=np.float64)
        for bus_pos, solver_pos in enumerate(np.asarray(bus_solver_pos, dtype=np.int32)):
            solver_pos_int = int(solver_pos)
            if solver_pos_int < 0 or solver_pos_int >= len(self.nodes):
                continue
            start = int(topology.bus_node_offsets[bus_pos])
            if start >= int(topology.bus_node_offsets[bus_pos + 1]):
                continue
            row = int(topology.bus_node_indices[start])
            voltage = max(float(bus[row, BUS_COLS["voltage"]] or 1.0), self.voltage_floor)
            angle = float(bus[row, BUS_COLS["angle"]] or 0.0)
            node = self.nodes[solver_pos_int]
            node.voltage = voltage
            node.angle = angle
            file_voltage[solver_pos_int] = voltage
            file_theta[solver_pos_int] = angle
        self.file_theta = file_theta
        self.file_voltage = file_voltage

    def _apply_measurement_seed_to_network(self) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is not None:
            seed_rows = tuple(seed_rows)
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
        if row_count:
            device_names, device_name_id = np.unique(device_name_array, return_inverse=True)
            device_name_id = device_name_id.astype(np.int64, copy=False)
        else:
            device_names = np.asarray([], dtype=object)
            device_name_id = np.asarray([], dtype=np.int64)
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is None or np.asarray(meas_type_code).size != row_count:
            meas_type_code = _meas_type_code_array(table.meas_type)
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
            "name": np.asarray(table.name, dtype=object),
            "device_type": np.asarray(table.device_type, dtype=object),
            "device_name": device_name_array,
            "device_names": np.asarray(device_names, dtype=object),
            "device_name_id_by_name": {
                name: int(pos)
                for pos, name in enumerate(np.asarray(device_names, dtype=object).tolist())
            },
            "meas_type": np.asarray(table.meas_type, dtype=object),
            "rows_by_device_type_code": rows_by_device_type_code(table),
            "normalized": bool(getattr(getattr(self, "measurements", None), "normalized", False)),
        }

    def _ensure_measurement_table_device_name_ids(self, table: MeasurementTable) -> bool:
        n_rows = int(table.idx.size)
        device_name_id = getattr(table, "device_name_id", None)
        if device_name_id is not None:
            device_name_id = np.asarray(device_name_id, dtype=np.int64)
            if device_name_id.size == n_rows:
                table.device_name_id = device_name_id
                return True
        meas_ppc = getattr(self, "meas_ppc", None)
        if not isinstance(meas_ppc, dict):
            warnings.warn(
                "AC SE measurement device index lookup requires meas_ppc; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        device_name = getattr(table, "device_name", None)
        if device_name is None:
            return False
        device_name_array = np.asarray(device_name, dtype=object)
        if device_name_array.size != n_rows:
            return False
        device_names = np.asarray(meas_ppc.get("device_names", ()), dtype=object)
        if device_name_array.size:
            unique_names = np.unique(device_name_array)
            missing_names = (
                np.setdiff1d(unique_names, device_names, assume_unique=False)
                if device_names.size
                else unique_names
            )
            if missing_names.size:
                meas_ppc["device_names"] = np.concatenate((device_names, missing_names.astype(object, copy=False)))
                id_by_name = meas_ppc.get("device_name_id_by_name")
                if not isinstance(id_by_name, dict):
                    id_by_name = {
                        name: int(pos)
                        for pos, name in enumerate(device_names.astype(object, copy=False).tolist())
                    }
                start_pos = int(device_names.size)
                for offset, name in enumerate(missing_names.astype(object, copy=False).tolist()):
                    id_by_name[name] = start_pos + int(offset)
                meas_ppc["device_name_id_by_name"] = id_by_name
                if hasattr(self, "_meas_device_name_sorted_id_cache"):
                    del self._meas_device_name_sorted_id_cache
                if hasattr(self, "_ac_measurement_plan_device_pos_by_type_code"):
                    self._ac_measurement_plan_device_pos_by_type_code_id = self._measurement_plan_device_id_lookup_arrays()
        table.device_name_id = self._meas_device_name_ids_for_ppc_names(meas_ppc, device_name_array)
        return True

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
        self._ensure_measurement_plan_lookup_arrays()
        n_rows = int(table.idx.size)
        device_pos = np.empty(n_rows, dtype=np.int64)
        device_pos.fill(-1)
        if n_rows == 0:
            return device_pos

        device_name_id = getattr(table, "device_name_id", None)
        if device_name_id is not None:
            device_name_id = np.asarray(device_name_id, dtype=np.int64)
            if device_name_id.size != n_rows:
                device_name_id = None
        if device_name_id is None:
            if self._ensure_measurement_table_device_name_ids(table):
                device_name_id = np.asarray(table.device_name_id, dtype=np.int64)
            else:
                warnings.warn(
                    "AC SE measurement device index lookup requires device_name_id; string fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return device_pos
        if device_name_id is None:
            warnings.warn(
                "AC SE measurement device index lookup requires device_name_id; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return device_pos

        device_maps = device_pos_by_type_code or self._ac_measurement_plan_device_pos_by_type_code
        if device_pos_by_type_code is None:
            device_pos_by_id = getattr(self, "_ac_measurement_plan_device_pos_by_type_code_id", {})
        else:
            device_pos_by_id = self._measurement_device_id_maps_for(device_maps)
        if not device_pos_by_id:
            warnings.warn(
                "AC SE measurement device index lookup requires indexed device maps; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return device_pos

        for code_int, code_rows in rows_by_device_type_code(table).items():
            rows = np.asarray(code_rows, dtype=np.int64)
            if rows.size == 0:
                continue
            lookup = device_pos_by_id.get(int(code_int))
            if lookup is None or lookup.size == 0:
                continue
            ids = device_name_id[rows]
            values = np.empty(rows.size, dtype=np.int64)
            values.fill(-1)
            in_range = (ids >= 0) & (ids < lookup.size)
            if np.any(in_range):
                values[in_range] = lookup[ids[in_range].astype(np.intp, copy=False)]
            device_pos[rows] = values
        return device_pos

    def _build_measurement_device_indexes(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
    ) -> np.ndarray:
        """Attach measurement-to-device compact indexes once after measurement loading."""
        measurements = self.measurements if measurements is None else measurements
        table = _measurement_table_from_measurements(measurements)
        table.device_pos = None
        device_pos = self._measurement_device_pos_array(table)
        table.device_pos = device_pos
        cache = getattr(table, "_device_pos_plan_cache", None)
        if cache is not None:
            cache.clear()
        try:
            measurements.table = table
        except AttributeError:
            pass
        if measurements is self.measurements:
            self.measurement_table = table
        return device_pos

    def _node_scale_arrays_by_pos(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (
            len(getattr(self, "nodes", ())),
            id(getattr(self, "node_pos", None)),
            id(getattr(self, "_node_vbase_by_idx", None)),
            float(self.p_base_kW),
            float(self.u_scale),
            float(self.i_scale),
        )
        cached = getattr(self, "_ac_node_scale_by_pos_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        n_nodes = len(getattr(self, "nodes", ()))
        node_idx_by_pos = np.empty(n_nodes, dtype=np.int64)
        node_idx_by_pos.fill(-1)
        voltage_scale = np.empty(n_nodes, dtype=np.float64)
        voltage_scale.fill(np.nan)
        current_scale = np.empty(n_nodes, dtype=np.float64)
        current_scale.fill(np.nan)
        valid = np.zeros(n_nodes, dtype=bool)
        node_vbase = getattr(self, "_node_vbase_by_idx", {})
        for node_idx_raw, pos_raw in getattr(self, "node_pos", {}).items():
            pos = int(pos_raw)
            if pos < 0 or pos >= n_nodes:
                continue
            node_idx = int(node_idx_raw)
            vbase = node_vbase.get(node_idx)
            if vbase is None and pos < len(self.nodes):
                vbase = getattr(self.nodes[pos], "vbase", None)
            if vbase is None:
                continue
            vbase = float(vbase)
            node_idx_by_pos[pos] = node_idx
            voltage_scale[pos] = self.u_scale * vbase
            current_scale[pos] = self.i_scale * ac_current_base_ka(self.p_base_kW, vbase)
            valid[pos] = True
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

    def _measurement_scale_for_codes(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
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
                (mtypes == MEAS_TYPE_CODES_P_FROM)
                | (mtypes == MEAS_TYPE_CODES_Q_FROM)
                | (mtypes == MEAS_TYPE_CODES_P_TO)
                | (mtypes == MEAS_TYPE_CODES_Q_TO)
            )
            scale[rows_valid[power_mask]] = self.p_base
            mask = mtypes == MEAS_TYPE_CODES_V_FROM
            scale[rows_valid[mask]] = node_voltage_scale[i_valid[mask]]
            mask = mtypes == MEAS_TYPE_CODES_I_FROM
            scale[rows_valid[mask]] = node_current_scale[i_valid[mask]]
            mask = mtypes == MEAS_TYPE_CODES_V_TO
            scale[rows_valid[mask]] = node_voltage_scale[j_valid[mask]]
            mask = mtypes == MEAS_TYPE_CODES_I_TO
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
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACNODE).astype(np.int64, copy=False),
            self._ac_node_plan_pos,
            MEAS_TYPE_CODES_V,
        )
        apply_terminal(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACBRANCH).astype(np.int64, copy=False),
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
        )
        apply_terminal(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACTRANSFORMER).astype(np.int64, copy=False),
            self._ac_transformer_plan_i,
            self._ac_transformer_plan_j,
        )
        apply_terminal(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACZEROBRANCH).astype(np.int64, copy=False),
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
        )
        apply_terminal(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACBREAK).astype(np.int64, copy=False),
            self._ac_break_plan_i,
            self._ac_break_plan_j,
        )
        apply_single(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACGENERATOR).astype(np.int64, copy=False),
            self._ac_generator_plan_node_pos,
            (MEAS_TYPE_CODES_P_GEN, MEAS_TYPE_CODES_Q_GEN),
            MEAS_TYPE_CODES_V_GEN,
            MEAS_TYPE_CODES_I_GEN,
        )
        apply_single(
            np.flatnonzero(device_type_code == DEVICE_TYPE_CODES_ACLOAD).astype(np.int64, copy=False),
            self._ac_load_plan_node_pos,
            (MEAS_TYPE_CODES_P_LOAD, MEAS_TYPE_CODES_Q_LOAD),
            MEAS_TYPE_CODES_V_LOAD,
            MEAS_TYPE_CODES_I_LOAD,
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
                (device_type_code == int(device_code)) & (meas_type_code == MEAS_TYPE_CODES_V_FROM)
            ).astype(np.int64, copy=False)
            if rows_from.size:
                valid, values = self._values_for_plan_pos(np.asarray(i_values, dtype=np.int64), device_pos[rows_from])
                if np.any(valid):
                    from_pos[rows_from[valid]] = values[valid]
            rows_to = np.flatnonzero(
                (device_type_code == int(device_code)) & (meas_type_code == MEAS_TYPE_CODES_V_TO)
            ).astype(np.int64, copy=False)
            if rows_to.size:
                valid, values = self._values_for_plan_pos(np.asarray(j_values, dtype=np.int64), device_pos[rows_to])
                if np.any(valid):
                    to_pos[rows_to[valid]] = values[valid]

        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_CODES_ACNODE) & (meas_type_code == MEAS_TYPE_CODES_V)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
        )
        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_CODES_ACGENERATOR) & (meas_type_code == MEAS_TYPE_CODES_V_GEN)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_generator_plan_node_pos", np.asarray([], dtype=np.int64)),
        )
        assign_single(
            np.flatnonzero(
                (device_type_code == DEVICE_TYPE_CODES_ACLOAD) & (meas_type_code == MEAS_TYPE_CODES_V_LOAD)
            ).astype(np.int64, copy=False),
            getattr(self, "_ac_load_plan_node_pos", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_CODES_ACBRANCH,
            getattr(self, "_ac_branch_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_branch_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_CODES_ACTRANSFORMER,
            getattr(self, "_ac_transformer_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_transformer_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_CODES_ACZEROBRANCH,
            getattr(self, "_ac_zero_branch_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_zero_branch_plan_j", np.asarray([], dtype=np.int64)),
        )
        assign_terminal(
            DEVICE_TYPE_CODES_ACBREAK,
            getattr(self, "_ac_break_plan_i", np.asarray([], dtype=np.int64)),
            getattr(self, "_ac_break_plan_j", np.asarray([], dtype=np.int64)),
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
        plan_name_rows = getattr(self, "_ac_measurement_plan_name_rows_by_type_code", None)
        if not plan_name_rows:
            return out
        for code_int, name_rows in plan_name_rows.items():
            rows = np.asarray(name_rows[1], dtype=np.int64)
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
        plan_name_rows = getattr(self, "_ac_measurement_plan_name_rows_by_type_code", None)
        if not plan_name_rows or rows.size == 0:
            return out
        name_rows = plan_name_rows.get(int(device_type_code))
        if name_rows is None:
            return out
        plan_rows = np.asarray(name_rows[1], dtype=np.int64)
        if plan_rows.size == 0:
            return out
        lookup_size = max(int(table_size), int(np.max(plan_rows)) + 1, int(np.max(rows)) + 1)
        lookup = np.full(lookup_size, -1, dtype=np.int64)
        lookup[plan_rows.astype(np.intp, copy=False)] = np.arange(plan_rows.size, dtype=np.int64)
        in_range = (rows >= 0) & (rows < lookup.size)
        if np.any(in_range):
            out[in_range] = lookup[rows[in_range].astype(np.intp, copy=False)]
        return out

    def _set_active_key_caches(self, device_keys: set, measurement_keys: set) -> None:
        self._active_device_keys = device_keys
        self._active_measurement_keys = measurement_keys
        self._active_device_code_pos_cache = device_keys
        self._active_measurement_code_pos_cache = measurement_keys

    @staticmethod
    def _active_device_key(device_type_code: int, device_pos: int) -> Tuple[int, int]:
        return int(device_type_code), int(device_pos)

    @staticmethod
    def _active_measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> Tuple[int, int, int]:
        return int(device_type_code), int(device_pos), int(meas_type_code)

    @staticmethod
    def _active_key_cache_from_arrays(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        active_mask: np.ndarray,
    ) -> Tuple[set, set]:
        valid = np.asarray(active_mask, dtype=bool)
        rows = np.flatnonzero(valid).astype(np.int64, copy=False)
        if rows.size == 0:
            return set(), set()
        type_values = np.asarray(device_type_code, dtype=np.int16)[rows].astype(np.int64, copy=False)
        pos_values = np.asarray(device_pos, dtype=np.int64)[rows]
        meas_values = np.asarray(meas_type_code, dtype=np.int16)[rows].astype(np.int64, copy=False)
        valid_pos = pos_values >= 0
        if not np.any(valid_pos):
            return set(), set()
        type_list = type_values[valid_pos].tolist()
        pos_list = pos_values[valid_pos].tolist()
        meas_list = meas_values[valid_pos].tolist()
        device_cache = set(zip(type_list, pos_list))
        measurement_cache = set(zip(type_list, pos_list, meas_list))
        return device_cache, measurement_cache

    def _voltage_best_from_arrays(
        self,
        table: MeasurementTable,
        real_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        _node_valid, _node_voltage_scale, _node_current_scale, node_idx_by_pos = self._node_scale_arrays_by_pos()
        node_pos_map = getattr(self, "node_pos", {})
        row_index = np.arange(int(table.idx.size), dtype=np.int64)

        def update_best(rows: np.ndarray, pos_values: np.ndarray, target: Dict[int, Tuple[float, float]]) -> None:
            if rows.size == 0:
                return
            pos_values = np.asarray(pos_values, dtype=np.int64)
            in_range = (pos_values >= 0) & (pos_values < node_idx_by_pos.size)
            if not np.any(in_range):
                return
            rows_valid = rows[in_range]
            node_idx_values = node_idx_by_pos[pos_values[in_range].astype(np.intp, copy=False)]
            valid_nodes = node_idx_values >= 0
            if not np.any(valid_nodes):
                return
            for node_idx_raw, weight, value in zip(
                node_idx_values[valid_nodes].tolist(),
                np.asarray(table.weight, dtype=np.float64)[rows_valid[valid_nodes]].tolist(),
                np.asarray(table.value, dtype=np.float64)[rows_valid[valid_nodes]].tolist(),
            ):
                node_idx = int(node_idx_raw)
                if node_idx not in node_pos_map:
                    continue
                current = target.get(node_idx)
                if current is None or float(weight) > current[0]:
                    target[node_idx] = (float(weight), float(value))

        node_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_CODES_ACNODE)
            & (meas_type_code == MEAS_TYPE_CODES_V)
        ]
        update_best(node_v_rows, from_pos[node_v_rows], node_voltage_best)
        update_best(node_v_rows, from_pos[node_v_rows], real_voltage_best)

        gen_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_CODES_ACGENERATOR)
            & (meas_type_code == MEAS_TYPE_CODES_V_GEN)
        ]
        load_v_rows = row_index[
            real_mask
            & (device_type_code == DEVICE_TYPE_CODES_ACLOAD)
            & (meas_type_code == MEAS_TYPE_CODES_V_LOAD)
        ]
        update_best(gen_v_rows, from_pos[gen_v_rows], real_voltage_best)
        update_best(load_v_rows, from_pos[load_v_rows], real_voltage_best)

        terminal_device_mask = (
            (device_type_code == DEVICE_TYPE_CODES_ACBRANCH)
            | (device_type_code == DEVICE_TYPE_CODES_ACTRANSFORMER)
            | (device_type_code == DEVICE_TYPE_CODES_ACZEROBRANCH)
            | (device_type_code == DEVICE_TYPE_CODES_ACBREAK)
        )
        terminal_v_from_rows = row_index[real_mask & terminal_device_mask & (meas_type_code == MEAS_TYPE_CODES_V_FROM)]
        terminal_v_to_rows = row_index[real_mask & terminal_device_mask & (meas_type_code == MEAS_TYPE_CODES_V_TO)]
        update_best(terminal_v_from_rows, from_pos[terminal_v_from_rows], real_voltage_best)
        update_best(terminal_v_to_rows, to_pos[terminal_v_to_rows], real_voltage_best)
        return node_voltage_best, real_voltage_best

    def _power_seed_best_from_arrays(
        self,
        table: MeasurementTable,
        processable_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
        best: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        row_index = np.arange(int(table.idx.size), dtype=np.int64)
        gen_power_rows = row_index[
            processable_mask
            & (device_type_code == DEVICE_TYPE_CODES_ACGENERATOR)
            & ((meas_type_code == MEAS_TYPE_CODES_P_GEN) | (meas_type_code == MEAS_TYPE_CODES_Q_GEN))
            & (device_pos >= 0)
        ]
        load_power_rows = row_index[
            processable_mask
            & (device_type_code == DEVICE_TYPE_CODES_ACLOAD)
            & ((meas_type_code == MEAS_TYPE_CODES_P_LOAD) | (meas_type_code == MEAS_TYPE_CODES_Q_LOAD))
            & (device_pos >= 0)
        ]
        for row in np.concatenate((gen_power_rows, load_power_rows)).tolist():
            key = (int(device_type_code[row]), int(device_pos[row]), int(meas_type_code[row]))
            weight = float(table.weight[row])
            current = best.get(key)
            if current is None or weight > current[0]:
                best[key] = (weight, float(table.value[row]))
        return best

    def _power_flow_seed_rows_from_arrays(
        self,
        table: MeasurementTable,
        processable_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> List[Tuple[int, int, int, float]]:
        if getattr(self, "flat_start", True):
            return []
        seed_rows_mask = (
            (device_type_code == DEVICE_TYPE_CODES_ACNODE)
            & (meas_type_code == MEAS_TYPE_CODES_V)
        ) | (
            (device_type_code == DEVICE_TYPE_CODES_ACGENERATOR)
            & np.isin(
                meas_type_code,
                (
                    MEAS_TYPE_CODES_P_GEN,
                    MEAS_TYPE_CODES_Q_GEN,
                    MEAS_TYPE_CODES_V_GEN,
                    MEAS_TYPE_CODES_I_GEN,
                ),
            )
        ) | (
            (device_type_code == DEVICE_TYPE_CODES_ACLOAD)
            & np.isin(
                meas_type_code,
                (
                    MEAS_TYPE_CODES_P_LOAD,
                    MEAS_TYPE_CODES_Q_LOAD,
                    MEAS_TYPE_CODES_V_LOAD,
                    MEAS_TYPE_CODES_I_LOAD,
                ),
            )
        )
        seed_rows = np.flatnonzero(processable_mask & seed_rows_mask & (device_pos >= 0)).astype(np.int64, copy=False)
        if seed_rows.size == 0:
            return []
        ppc_rows = self._measurement_ppc_rows_for_device_positions(device_type_code[seed_rows], device_pos[seed_rows])
        valid = ppc_rows >= 0
        if not np.any(valid):
            return []
        rows = seed_rows[valid]
        return list(
            zip(
                device_type_code[rows].astype(np.int64, copy=False).tolist(),
                ppc_rows[valid].astype(np.int64, copy=False).tolist(),
                meas_type_code[rows].astype(np.int64, copy=False).tolist(),
                np.asarray(table.value, dtype=np.float64)[rows].tolist(),
            )
        )

    def _meas_device_name_ids_for_ppc_names(self, meas_ppc: Dict, names: np.ndarray) -> np.ndarray:
        names = np.asarray(names, dtype=object)
        if names.size == 0:
            return np.empty(0, dtype=np.int64)
        device_names = np.asarray(meas_ppc.get("device_names", ()), dtype=object)
        if device_names.size == 0:
            out = np.empty(names.size, dtype=np.int64)
            out.fill(-1)
            return out
        id_by_name = meas_ppc.get("device_name_id_by_name")
        if isinstance(id_by_name, dict):
            return np.fromiter((int(id_by_name.get(name, -1)) for name in names.tolist()), dtype=np.int64, count=names.size)
        cache = getattr(self, "_meas_device_name_sorted_id_cache", None)
        cache_key = (id(device_names), int(device_names.size))
        if cache is None or cache[0] != cache_key:
            order = np.argsort(device_names, kind="stable")
            cache = (cache_key, order, device_names[order])
            self._meas_device_name_sorted_id_cache = cache
        order = cache[1]
        sorted_names = cache[2]
        pos = np.searchsorted(sorted_names, names)
        out = np.empty(names.size, dtype=np.int64)
        out.fill(-1)
        in_range = pos < sorted_names.size
        if np.any(in_range):
            idx = np.flatnonzero(in_range)
            matched = sorted_names[pos[idx]] == names[idx]
            if np.any(matched):
                out[idx[matched]] = order[pos[idx[matched]]].astype(np.int64, copy=False)
        return out

    def _measurement_runtime_array_cache_key(self) -> Tuple[object, ...]:
        return (
            id(getattr(self, "_ac_measurement_plan_device_pos_by_type_code_id", None)),
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
        if not isinstance(meas, np.ndarray) or meas.shape[0] != table.idx.size:
            return False
        cols = meas_ppc.get("meas_cols", MEAS_COLS)
        value_array = table.value
        weight_array = table.weight
        valid_array = table.valid
        status_array = measurement_table_status_code(table)
        idx_array = table.idx
        n_rows = int(value_array.size)
        device_name_id = meas[:, cols["device_name_id"]].astype(np.int64, copy=False)
        device_type_code_array = meas[:, cols["device_type_code"]].astype(np.int16, copy=False)
        meas_type_code_array = meas[:, cols["meas_type_code"]].astype(np.int16, copy=False)
        if device_name_id.size != n_rows or device_type_code_array.size != n_rows or meas_type_code_array.size != n_rows:
            return False

        table.device_name_id = device_name_id
        table.meas_type_code = meas_type_code_array
        table.device_type_code = device_type_code_array
        runtime_cache_key = self._measurement_runtime_array_cache_key()
        cached_runtime = self._cached_measurement_runtime_arrays(meas_ppc, n_rows, runtime_cache_key)
        if cached_runtime is None:
            device_pos = getattr(table, "device_pos", None)
            if device_pos is not None:
                device_pos = np.asarray(device_pos, dtype=np.int64)
                if int(device_pos.size) != n_rows:
                    device_pos = None
            if device_pos is None:
                device_pos = self._measurement_device_pos_array(table)
            available, scale_array, from_pos, to_pos = self._measurement_scale_for_codes(
                device_type_code_array,
                device_pos,
                meas_type_code_array,
            )
            meas_ppc["device_pos"] = device_pos.astype(np.int64, copy=True)
            meas_ppc["available"] = available.astype(bool, copy=True)
            meas_ppc["scale"] = scale_array.astype(np.float64, copy=True)
            meas_ppc["from_pos"] = from_pos.astype(np.int64, copy=True)
            meas_ppc["to_pos"] = to_pos.astype(np.int64, copy=True)
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
        active_device_keys, active_measurement_keys = self._active_key_cache_from_arrays(
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
        self._set_active_key_caches(active_device_keys, active_measurement_keys)
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }
        self._real_power_measurement_seed_cache = power_seed_best
        self._power_flow_seed_rows = power_flow_seed_rows
        self._has_valid_angle_measurements = False
        try:
            self.measurements.normalized = True
        except AttributeError:
            pass
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
        self._set_active_key_caches(set(), set())
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self._node_voltage_measurement_cache = {}
        self._real_voltage_observation_node_cache = {}
        self._real_power_measurement_seed_cache = {}
        self._power_flow_seed_rows = []
        self._has_valid_angle_measurements = bool(np.any(table.valid & table.angle_mask))
        warnings.warn(
            "AC SE measurement normalization requires measurement PPC arrays; name-based fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _convert_measurements_to_pu(self) -> None:
        self._normalize_measurements_to_pu()

    def _active_device_keys_ref(self) -> set:
        """Return active devices as (device_type_code, device_pos) keys."""
        if hasattr(self, "_active_device_code_pos_cache"):
            self._active_device_keys = self._active_device_code_pos_cache
            return self._active_device_code_pos_cache
        if not hasattr(self, "_active_device_keys"):
            self._refresh_measurement_summary_cache()
        return self._active_device_keys

    def _active_measurement_keys_ref(self) -> set:
        """Return active measurements as (device_type_code, device_pos, meas_type_code) keys."""
        if hasattr(self, "_active_measurement_code_pos_cache"):
            self._active_measurement_keys = self._active_measurement_code_pos_cache
            return self._active_measurement_code_pos_cache
        if not hasattr(self, "_active_measurement_keys"):
            self._refresh_measurement_summary_cache()
        return self._active_measurement_keys

    def _active_device_code_pos_cache_ref(self) -> set:
        return self._active_device_keys_ref()

    def _active_measurement_code_pos_cache_ref(self) -> set:
        return self._active_measurement_keys_ref()

    def _next_measurement_idx(self) -> int:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        return int(self._max_measurement_idx) + 1

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        active_device_keys = set()
        active_measurement_keys = set()
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        power_seed_best: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        power_flow_seed_rows: List[Tuple[int, int, int, float]] = []
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
                meas_type_code = _meas_type_code_array(table.meas_type)
                table.meas_type_code = meas_type_code
            device_pos = getattr(table, "device_pos", None)
            if device_pos is None or np.asarray(device_pos).size != int(table.idx.size):
                device_pos = self._measurement_device_pos_array(table)
                table.device_pos = device_pos
            else:
                device_pos = np.asarray(device_pos, dtype=np.int64)
            active_device_keys, active_measurement_keys = self._active_key_cache_from_arrays(
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
        self._set_active_key_caches(active_device_keys, active_measurement_keys)
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }
        self._real_power_measurement_seed_cache = power_seed_best
        self._power_flow_seed_rows = power_flow_seed_rows

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
    ) -> TableBackedMeasurementList:
        row_count = len(names)
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
        device_type_array = np.asarray(device_types, dtype=object)
        if device_type_codes is None:
            device_type_code = np.asarray(
                [_DEVICE_TYPE_CODES.get(str(device_type), 0) for device_type in device_type_array.tolist()],
                dtype=np.int16,
            )
        else:
            device_type_code = np.asarray(device_type_codes, dtype=np.int16)
            if device_type_code.size != row_count:
                raise ValueError("pseudo measurement device_type_code array size does not match row count")
        if meas_type_codes is None:
            meas_type_code = _meas_type_code_array(meas_types)
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
            name=np.asarray(names, dtype=object),
            device_type=device_type_array,
            device_name=np.asarray(device_names, dtype=object),
            meas_type=np.asarray(meas_types, dtype=object),
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

    @staticmethod
    def _table_backed_pseudo_measurements(
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
    ) -> TableBackedMeasurementList:
        return TableBackedMeasurementList(
            ACStateEstimator._pseudo_measurement_table(
                names,
                device_types,
                device_names,
                meas_types,
                values,
                weight,
                idx_start=idx_start,
                device_type_codes=device_type_codes,
                meas_type_codes=meas_type_codes,
                device_positions=device_positions,
            ),
            normalized=True,
        )

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
        row_count = len(names)
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
        self._max_measurement_idx = int(next_idx) + row_count - 1
        if record_summary:
            device_key_cache = self._active_device_keys_ref()
            measurement_key_cache = self._active_measurement_keys_ref()
            tail_type = np.asarray(appended_table.device_type_code, dtype=np.int16)
            tail_meas = np.asarray(appended_table.meas_type_code, dtype=np.int16)
            tail_pos = (
                appended_device_pos
                if appended_device_pos is not None and appended_device_pos.size == row_count
                else np.full(row_count, -1, dtype=np.int64)
            )
            valid_tail = tail_pos >= 0
            for code, pos in zip(tail_type[valid_tail].tolist(), tail_pos[valid_tail].tolist()):
                device_key_cache.add(self._active_device_key(code, pos))
            for code, pos, meas_code in zip(
                tail_type[valid_tail].tolist(),
                tail_pos[valid_tail].tolist(),
                tail_meas[valid_tail].tolist(),
            ):
                measurement_key_cache.add(self._active_measurement_key(code, pos, meas_code))
        return int(next_idx) + row_count

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
        pos = self.node_pos.get(node_idx)
        component_by_pos = getattr(self, "zero_tie_component_by_pos", None)
        components = getattr(self, "zero_tie_components", None)
        if pos is None or component_by_pos is None or not components:
            return None
        component = components[int(component_by_pos[int(pos)])]
        for member_pos in component:
            member_idx = self.nodes[int(member_pos)].idx
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_pseudo_is_covered_by_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> bool:
        from_pos, to_pos = self._measurement_voltage_node_positions_from_codes(
            np.asarray([int(device_type_code)], dtype=np.int16),
            np.asarray([int(device_pos)], dtype=np.int64),
            np.asarray([int(meas_type_code)], dtype=np.int16),
        )
        node_pos = int(to_pos[0]) if int(meas_type_code) == MEAS_TYPE_CODES_V_TO else int(from_pos[0])
        if node_pos < 0:
            return False
        _node_valid, _node_voltage_scale, _node_current_scale, node_idx_by_pos = self._node_scale_arrays_by_pos()
        if node_pos >= int(node_idx_by_pos.size):
            return False
        node_idx = int(node_idx_by_pos[node_pos])
        if node_idx < 0:
            return False
        return self._real_voltage_observation_value_for_node(node_idx) is not None

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak P/Q/V priors for unmeasured AC topology-device states."""
        measured_keys = self._active_measurement_keys_ref()
        added_keys = set()
        pseudo_names: List[str] = []
        pseudo_device_types: List[str] = []
        pseudo_device_type_codes: List[int] = []
        pseudo_device_names: List[str] = []
        pseudo_device_positions: List[int] = []
        pseudo_meas_types: List[str] = []
        pseudo_meas_type_codes: List[int] = []
        pseudo_row_values: List[float] = []
        pseudo_weights: List[float] = []
        topology_weight = float(self.pseudo_measurement_weight) * 1e-4
        ppc = self._ac_ppc_dict()
        for device_type_code, device_type, table_name, name_key, device_key, cols in (
            (DEVICE_TYPE_CODES_ACZEROBRANCH, "ACZeroBranch", "zero_branch", "zero_branch_name", "zero_branch", ZERO_BRANCH_COLS),
            (DEVICE_TYPE_CODES_ACBREAK, "ACBreak", "break", "break_name", "break", BREAK_COLS),
        ):
            table, device_topology, rows = self._ppc_sorted_alive_rows_for(ppc, table_name, device_key, cols["idx"])
            if rows.size == 0 or device_topology is None:
                continue
            names = self._ppc_names_for_rows(ppc.get(name_key, np.asarray([], dtype=object)), rows)
            plan_pos = self._plan_pos_for_ppc_rows(int(device_type_code), rows, table.shape[0])
            table_rows = table[rows.astype(np.intp, copy=False)]
            for row, name, pos, device_values in zip(rows, names, plan_pos, table_rows):
                if int(pos) < 0:
                    continue
                device_name = str(name)
                has_terminal_current = any(
                    self._active_measurement_key(device_type_code, int(pos), meas_type_code) in measured_keys
                    for meas_type_code in (MEAS_TYPE_CODES_I_FROM, MEAS_TYPE_CODES_I_TO)
                )
                topology_pseudo_values = []
                if not has_terminal_current and not any(
                    self._active_measurement_key(device_type_code, int(pos), meas_type_code) in measured_keys
                    for meas_type_code in (MEAS_TYPE_CODES_P_FROM, MEAS_TYPE_CODES_P_TO)
                ):
                    topology_pseudo_values.append(("P_FROM", MEAS_TYPE_CODES_P_FROM, float(device_values[cols["p"]] or 0.0)))
                if not has_terminal_current and not any(
                    self._active_measurement_key(device_type_code, int(pos), meas_type_code) in measured_keys
                    for meas_type_code in (MEAS_TYPE_CODES_Q_FROM, MEAS_TYPE_CODES_Q_TO)
                ):
                    topology_pseudo_values.append(("Q_FROM", MEAS_TYPE_CODES_Q_FROM, float(device_values[cols["q"]] or 0.0)))
                i_node = int(device_values[cols["i_node"]])
                if self._real_voltage_observation_value_for_node(i_node) is None:
                    topology_pseudo_values.append(("V_FROM", MEAS_TYPE_CODES_V_FROM, self._ppc_terminal_voltage_pseudo_seed(ppc, device_topology, int(row))))
                for meas_type, meas_type_code, value in topology_pseudo_values:
                    key = self._active_measurement_key(device_type_code, int(pos), meas_type_code)
                    if key in measured_keys or key in added_keys:
                        continue
                    pseudo_names.append(f"pseudo_{meas_type.lower()}_{device_name}")
                    pseudo_device_types.append(device_type)
                    pseudo_device_type_codes.append(int(device_type_code))
                    pseudo_device_names.append(device_name)
                    pseudo_device_positions.append(int(pos))
                    pseudo_meas_types.append(meas_type)
                    pseudo_meas_type_codes.append(int(meas_type_code))
                    pseudo_row_values.append(float(value))
                    pseudo_weights.append(topology_weight)
                    added_keys.add(key)
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names,
            pseudo_device_types,
            pseudo_device_names,
            pseudo_meas_types,
            pseudo_row_values,
            weights=pseudo_weights,
            record_summary=False,
            device_type_codes=pseudo_device_type_codes,
            meas_type_codes=pseudo_meas_type_codes,
            device_positions=pseudo_device_positions,
        )
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        if not hasattr(self, "_active_device_keys") or not hasattr(self, "_active_measurement_keys"):
            self._refresh_measurement_summary_cache()
        measured_devices = self._active_device_keys_ref()
        measured_keys = self._active_measurement_keys_ref()
        added_keys = set()
        pseudo_names: List[str] = []
        pseudo_device_types: List[str] = []
        pseudo_device_type_codes: List[int] = []
        pseudo_device_names: List[str] = []
        pseudo_device_positions: List[int] = []
        pseudo_meas_types: List[str] = []
        pseudo_meas_type_codes: List[int] = []
        pseudo_values: List[float] = []

        def queue_pseudo(
            name: str,
            device_type: str,
            device_type_code: int,
            device_name: str,
            device_pos: int,
            meas_type: str,
            meas_type_code: int,
            value: float,
        ) -> None:
            pseudo_names.append(name)
            pseudo_device_types.append(device_type)
            pseudo_device_type_codes.append(int(device_type_code))
            pseudo_device_names.append(device_name)
            pseudo_device_positions.append(int(device_pos))
            pseudo_meas_types.append(meas_type)
            pseudo_meas_type_codes.append(int(meas_type_code))
            pseudo_values.append(float(value))

        next_idx = self._next_measurement_idx()
        next_idx, topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        added_keys.update(topology_added_keys)
        ppc = self._ac_ppc_dict()
        gen, gen_topology, gen_rows = self._ppc_sorted_alive_rows_for(ppc, "gen", "gen", GEN_COLS["idx"])
        gen_voltage = self._ppc_device_node_voltage_values(ppc, gen_topology, gen_rows)
        gen_p, gen_q = self._ppc_generator_pseudo_power_arrays(ppc, gen_rows)
        gen_nodes = gen[gen_rows.astype(np.intp, copy=False), GEN_COLS["node"]].astype(np.int64, copy=False)
        gen_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_CODES_ACGENERATOR, gen_rows, gen.shape[0])
        for name, pos, p, q, voltage, node_idx in zip(self._ppc_names_for_rows(ppc["gen_name"], gen_rows), gen_plan_pos, gen_p, gen_q, gen_voltage, gen_nodes):
            if int(pos) < 0:
                continue
            device_name = str(name)
            key = self._active_measurement_key(DEVICE_TYPE_CODES_ACGENERATOR, int(pos), MEAS_TYPE_CODES_P_GEN)
            if key not in measured_keys and key not in added_keys:
                queue_pseudo(
                    f"pseudo_p_{device_name}",
                    "ACGenerator",
                    DEVICE_TYPE_CODES_ACGENERATOR,
                    device_name,
                    int(pos),
                    "P_GEN",
                    MEAS_TYPE_CODES_P_GEN,
                    p,
                )
                added_keys.add(key)
            key = self._active_measurement_key(DEVICE_TYPE_CODES_ACGENERATOR, int(pos), MEAS_TYPE_CODES_Q_GEN)
            if key not in measured_keys and key not in added_keys:
                queue_pseudo(
                    f"pseudo_q_{device_name}",
                    "ACGenerator",
                    DEVICE_TYPE_CODES_ACGENERATOR,
                    device_name,
                    int(pos),
                    "Q_GEN",
                    MEAS_TYPE_CODES_Q_GEN,
                    q,
                )
                added_keys.add(key)
            if self._active_device_key(DEVICE_TYPE_CODES_ACGENERATOR, int(pos)) not in measured_devices:
                key = self._active_measurement_key(DEVICE_TYPE_CODES_ACGENERATOR, int(pos), MEAS_TYPE_CODES_V_GEN)
                if (
                    key not in measured_keys
                    and key not in added_keys
                    and self._real_voltage_observation_value_for_node(int(node_idx)) is None
                ):
                    queue_pseudo(
                        f"pseudo_v_{device_name}",
                        "ACGenerator",
                        DEVICE_TYPE_CODES_ACGENERATOR,
                        device_name,
                        int(pos),
                        "V_GEN",
                        MEAS_TYPE_CODES_V_GEN,
                        float(voltage or 1.0),
                    )
                    added_keys.add(key)

        load, load_topology, load_rows = self._ppc_sorted_alive_rows_for(ppc, "load", "load", LOAD_COLS["idx"])
        load_voltage = self._ppc_device_node_voltage_values(ppc, load_topology, load_rows)
        load_p, load_q = self._ppc_load_pseudo_power_arrays(ppc, load_rows, load_voltage)
        load_nodes = load[load_rows.astype(np.intp, copy=False), LOAD_COLS["node"]].astype(np.int64, copy=False)
        load_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_CODES_ACLOAD, load_rows, load.shape[0])
        for name, pos, p, q, voltage, node_idx in zip(self._ppc_names_for_rows(ppc["load_name"], load_rows), load_plan_pos, load_p, load_q, load_voltage, load_nodes):
            if int(pos) < 0:
                continue
            device_name = str(name)
            unmetered_load = self._active_device_key(DEVICE_TYPE_CODES_ACLOAD, int(pos)) not in measured_devices
            key = self._active_measurement_key(DEVICE_TYPE_CODES_ACLOAD, int(pos), MEAS_TYPE_CODES_P_LOAD)
            if key not in measured_keys and key not in added_keys:
                queue_pseudo(
                    f"pseudo_p_{device_name}",
                    "ACLoad",
                    DEVICE_TYPE_CODES_ACLOAD,
                    device_name,
                    int(pos),
                    "P_LOAD",
                    MEAS_TYPE_CODES_P_LOAD,
                    p,
                )
                added_keys.add(key)
            key = self._active_measurement_key(DEVICE_TYPE_CODES_ACLOAD, int(pos), MEAS_TYPE_CODES_Q_LOAD)
            if key not in measured_keys and key not in added_keys:
                queue_pseudo(
                    f"pseudo_q_{device_name}",
                    "ACLoad",
                    DEVICE_TYPE_CODES_ACLOAD,
                    device_name,
                    int(pos),
                    "Q_LOAD",
                    MEAS_TYPE_CODES_Q_LOAD,
                    q,
                )
                added_keys.add(key)
            if unmetered_load:
                key = self._active_measurement_key(DEVICE_TYPE_CODES_ACLOAD, int(pos), MEAS_TYPE_CODES_V_LOAD)
                if (
                    key not in measured_keys
                    and key not in added_keys
                    and self._real_voltage_observation_value_for_node(int(node_idx)) is None
                ):
                    queue_pseudo(
                        f"pseudo_v_{device_name}",
                        "ACLoad",
                        DEVICE_TYPE_CODES_ACLOAD,
                        device_name,
                        int(pos),
                        "V_LOAD",
                        MEAS_TYPE_CODES_V_LOAD,
                        float(voltage or 1.0),
                    )
                    added_keys.add(key)
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names,
            pseudo_device_types,
            pseudo_device_names,
            pseudo_meas_types,
            pseudo_values,
            record_summary=False,
            device_type_codes=pseudo_device_type_codes,
            meas_type_codes=pseudo_meas_type_codes,
            device_positions=pseudo_device_positions,
        )
        if added_keys:
            measured_keys.update(added_keys)
            measured_devices.update(
                self._active_device_key(device_type_code, device_pos)
                for device_type_code, device_pos, _meas_type_code in added_keys
            )

    def _seed_power_state_arrays_from_measurements(self) -> None:
        """Use the best available P/Q rows as initial values for explicit power states."""
        best = getattr(self, "_real_power_measurement_seed_cache", None)
        if best is None:
            self._refresh_measurement_summary_cache()
            best = getattr(self, "_real_power_measurement_seed_cache", {})

        gen_index = np.asarray(getattr(self, "_ac_generator_plan_index", ()), dtype=np.int64)
        load_index = np.asarray(getattr(self, "_ac_load_plan_index", ()), dtype=np.int64)
        for (device_type_code, device_pos, meas_type_code), (_weight, value) in best.items():
            device_type_code = int(device_type_code)
            device_pos = int(device_pos)
            meas_type_code = int(meas_type_code)
            if device_type_code == DEVICE_TYPE_CODES_ACGENERATOR:
                if device_pos < 0 or device_pos >= gen_index.size:
                    continue
                state_idx = int(gen_index[device_pos])
                if state_idx < 0:
                    continue
                if meas_type_code == MEAS_TYPE_CODES_P_GEN:
                    self.initial_gen_p_array[state_idx] = value
                elif meas_type_code == MEAS_TYPE_CODES_Q_GEN:
                    self.initial_gen_q_array[state_idx] = value
            elif device_type_code == DEVICE_TYPE_CODES_ACLOAD:
                if device_pos < 0 or device_pos >= load_index.size:
                    continue
                state_idx = int(load_index[device_pos])
                if state_idx < 0:
                    continue
                if meas_type_code == MEAS_TYPE_CODES_P_LOAD:
                    self.initial_load_p_array[state_idx] = value
                elif meas_type_code == MEAS_TYPE_CODES_Q_LOAD:
                    self.initial_load_q_array[state_idx] = value

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
                refreshed = self._incremental_update_active_measurement_indexes(
                    self.measurements[measurement_count_before:]
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
        selected = self._select_weak_direction_pseudo_candidates(observability, candidates, max_add)
        if not selected:
            return 0
        next_idx = self._next_measurement_idx()
        measurement_count_before = len(self.measurements)
        for candidate in selected:
            candidate.idx = next_idx
            next_idx += 1
            self.measurements.append(candidate)
        if refresh:
            refreshed = self._incremental_update_active_measurement_indexes(
                self.measurements[measurement_count_before:]
            )
            if not refreshed:
                self._refresh_active_measurement_indexes()
        return len(selected)

    def _observability_pseudo_candidate_measurements(self) -> Sequence[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        existing_keys = self._active_measurement_keys_ref()
        candidate_keys = set()
        candidate_names: List[str] = []
        candidate_device_types: List[str] = []
        candidate_device_type_codes: List[int] = []
        candidate_device_names: List[str] = []
        candidate_device_positions: List[int] = []
        candidate_meas_types: List[str] = []
        candidate_meas_type_codes: List[int] = []
        candidate_values: List[float] = []
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "observability pseudo candidate build")

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
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if (
                int(meas_type_code) in _VOLTAGE_MEASUREMENT_TYPES
                and self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code)
            ):
                return
            if key in existing_keys or key in candidate_keys:
                return
            candidate_names.append(pseudo_name)
            candidate_device_types.append(device_type)
            candidate_device_type_codes.append(int(device_type_code))
            candidate_device_names.append(device_name)
            candidate_device_positions.append(int(device_pos))
            candidate_meas_types.append(meas_type)
            candidate_meas_type_codes.append(int(meas_type_code))
            candidate_values.append(float(value))
            candidate_keys.add(key)

        bus = np.asarray(ppc["bus"], dtype=np.float64)
        node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", None)
        if node_solver_pos is None:
            self._build_ppc_state_index_arrays()
            node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", np.asarray([], dtype=np.int32))
        node_rows = np.flatnonzero(np.asarray(node_solver_pos, dtype=np.int32) >= 0).astype(np.int64, copy=False)
        if node_rows.size > 1:
            node_rows = node_rows[np.argsort(bus[node_rows.astype(np.intp, copy=False), BUS_COLS["idx"]], kind="stable")]
        node_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_CODES_ACNODE, node_rows, bus.shape[0])
        for row, pos, name in zip(node_rows, node_plan_pos, self._ppc_names_for_rows(ppc["bus_name"], node_rows)):
            add(
                DEVICE_TYPE_CODES_ACNODE,
                "ACNode",
                str(name),
                int(pos),
                MEAS_TYPE_CODES_V,
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
            load_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_CODES_ACLOAD, load_rows, load.shape[0])
            for name, pos, p_value, q_value, voltage_value in zip(
                self._ppc_names_for_rows(ppc["load_name"], load_rows),
                load_plan_pos,
                p_values,
                q_values,
                load_voltage,
            ):
                device_name = str(name)
                add(DEVICE_TYPE_CODES_ACLOAD, "ACLoad", device_name, int(pos), MEAS_TYPE_CODES_P_LOAD, "P_LOAD", float(p_value))
                add(DEVICE_TYPE_CODES_ACLOAD, "ACLoad", device_name, int(pos), MEAS_TYPE_CODES_Q_LOAD, "Q_LOAD", float(q_value))
                add(DEVICE_TYPE_CODES_ACLOAD, "ACLoad", device_name, int(pos), MEAS_TYPE_CODES_V_LOAD, "V_LOAD", float(voltage_value or 1.0))

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
            gen_plan_pos = self._plan_pos_for_ppc_rows(DEVICE_TYPE_CODES_ACGENERATOR, gen_rows, gen.shape[0])
            for name, pos, p_value, q_value, voltage_value in zip(
                self._ppc_names_for_rows(ppc["gen_name"], gen_rows),
                gen_plan_pos,
                p_values,
                q_values,
                gen_voltage,
            ):
                device_name = str(name)
                add(DEVICE_TYPE_CODES_ACGENERATOR, "ACGenerator", device_name, int(pos), MEAS_TYPE_CODES_P_GEN, "P_GEN", float(p_value))
                add(DEVICE_TYPE_CODES_ACGENERATOR, "ACGenerator", device_name, int(pos), MEAS_TYPE_CODES_Q_GEN, "Q_GEN", float(q_value))
                add(DEVICE_TYPE_CODES_ACGENERATOR, "ACGenerator", device_name, int(pos), MEAS_TYPE_CODES_V_GEN, "V_GEN", float(voltage_value or 1.0))

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
            for name, pos, values in zip(self._ppc_names_for_rows(ppc[name_key], rows), plan_pos, table_rows):
                device_name = str(name)
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_CODES_P_FROM, "P_FROM", float(values[cols["i_p"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_CODES_Q_FROM, "Q_FROM", float(values[cols["i_q"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_CODES_P_TO, "P_TO", float(values[cols["j_p"]] or 0.0))
                add(device_type_code, device_type, device_name, int(pos), MEAS_TYPE_CODES_Q_TO, "Q_TO", float(values[cols["j_q"]] or 0.0))

        add_terminal_candidates(DEVICE_TYPE_CODES_ACBRANCH, "ACBranch", "branch", "branch_name", "branch", BRANCH_COLS)
        add_terminal_candidates(DEVICE_TYPE_CODES_ACTRANSFORMER, "ACTransformer", "transformer", "transformer_name", "transformer", TRANSFORMER_COLS)

        return self._table_backed_pseudo_measurements(
            candidate_names,
            candidate_device_types,
            candidate_device_names,
            candidate_meas_types,
            candidate_values,
            self.pseudo_measurement_weight,
            device_type_codes=candidate_device_type_codes,
            meas_type_codes=candidate_meas_type_codes,
            device_positions=candidate_device_positions,
        )

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: Sequence[Measurement],
        max_add: int,
    ) -> List[Measurement]:
        if max_add <= 0 or not candidates:
            return []
        x = self.initial_state()
        cache = self._observability_matrix_cache_for(observability, None, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, None)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return list(candidates[:max_add])
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
            return list(candidates[:max_add])
        positive = np.flatnonzero(scores > 0.0)
        if positive.size > max_add:
            top = positive[np.argpartition(-scores[positive], max_add - 1)[:max_add]]
            order = top[np.argsort(-scores[top], kind="stable")]
        else:
            order = positive[np.argsort(-scores[positive], kind="stable")]
        selected = [candidates[int(pos)] for pos in order[:max_add]]
        return selected or list(candidates[:max_add])

    def _rank_restoring_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidates from invalid real device rows, excluding node V and angles."""
        table = _measurement_table_from_measurements(self.measurements)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is None or np.asarray(meas_type_code).size != int(table.idx.size):
            meas_type_code = _meas_type_code_array(table.meas_type)
            table.meas_type_code = meas_type_code
        else:
            meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is None or np.asarray(device_pos).size != int(table.idx.size):
            device_pos = self._measurement_device_pos_array(table)
            table.device_pos = device_pos
        else:
            device_pos = np.asarray(device_pos, dtype=np.int64)
        existing_keys = self._active_measurement_keys_ref()
        seen_keys = set()
        candidates: List[Measurement] = []
        next_idx = self._next_measurement_idx()
        invalid_rows = np.flatnonzero((~np.asarray(table.valid, dtype=bool)) & (np.asarray(table.weight, dtype=np.float64) > 0.0))
        if invalid_rows.size == 0:
            return candidates
        available, scale, _from_pos, _to_pos = self._measurement_scale_for_codes(
            device_type_code[invalid_rows],
            device_pos[invalid_rows],
            meas_type_code[invalid_rows],
        )
        for local_idx, row in enumerate(invalid_rows.tolist()):
            row = int(row)
            if int(device_pos[row]) < 0 or not bool(available[local_idx]):
                continue
            if int(meas_type_code[row]) in _ANGLE_MEASUREMENT_CODE_SET:
                continue
            if int(device_type_code[row]) == DEVICE_TYPE_CODES_ACNODE and int(meas_type_code[row]) == MEAS_TYPE_CODES_V:
                continue
            key = self._active_measurement_key(device_type_code[row], device_pos[row], meas_type_code[row])
            if key in existing_keys or key in seen_keys:
                continue
            pseudo_name = f"pseudo_rank_{table.name[row]}"
            seen_keys.add(key)
            candidates.append(
                Measurement(
                    idx=next_idx + len(candidates),
                    name=pseudo_name,
                    device_type=str(table.device_type[row]),
                    device_name=str(table.device_name[row]),
                    meas_type=str(table.meas_type[row]),
                    weight=self.pseudo_measurement_weight,
                    valid=True,
                    value=float(table.value[row]) / float(scale[local_idx]),
                    status=MEAS_STATUS_PSEUDO,
                )
            )
        return candidates

    def _rank_restoring_candidate_indices(self, candidates: Sequence[Measurement], max_add: int) -> List[int]:
        """Select candidate rows that participate in a higher structural-rank matching."""
        if max_add <= 0 or not candidates:
            return []
        base_measurements = self.active_measurements
        x = self.initial_state()
        base_h = self.jacobian_sparse(x, base_measurements)
        base_rank = sparse_structural_rank(base_h)
        if base_rank is None or base_rank >= self.n_state:
            return []

        combined_measurements = list(base_measurements) + list(candidates)
        combined_h = self.jacobian_sparse(x, combined_measurements)
        combined_rank = sparse_structural_rank(combined_h)
        if combined_rank is None or combined_rank <= base_rank:
            return []

        matching = sp_maximum_bipartite_matching(combined_h, perm_type="row")
        base_rows = len(base_measurements)
        selected = sorted({int(row) - base_rows for row in matching if int(row) >= base_rows})
        selected = selected[:max_add]
        remaining = max_add - len(selected)
        if remaining <= 0:
            return selected

        selected_set = set(selected)
        selected_measurements = [candidates[idx] for idx in selected]
        remaining_candidates = [candidate for idx, candidate in enumerate(candidates) if idx not in selected_set]
        if not remaining_candidates:
            return selected

        current_measurements = list(base_measurements) + selected_measurements
        current_h = self.jacobian_sparse(x, current_measurements)
        candidate_h = self.jacobian_sparse(x, remaining_candidates)
        anchor_local_indices = self._angle_anchor_candidate_indices(current_h, candidate_h, remaining)
        if not anchor_local_indices:
            return selected

        original_by_identity = {id(candidate): idx for idx, candidate in enumerate(candidates)}
        selected.extend(original_by_identity[id(remaining_candidates[idx])] for idx in anchor_local_indices)
        return sorted(set(selected))

    def _angle_anchor_candidate_indices(self, current_h, candidate_h, max_add: int) -> List[int]:
        """Select non-angle rows that connect unanchored angle components to an anchored component."""
        if max_add <= 0 or self.n_angle <= 0 or candidate_h.shape[0] == 0:
            return []
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
            return []

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

        selected: List[int] = []
        used_rows = set()
        visited = set(anchored_roots)
        queue = list(anchored_roots)

        def activate(row_id: int) -> None:
            if len(selected) >= max_add or row_id in used_rows:
                return
            candidate_row, component_roots, _is_anchor_row = row_roots[row_id]
            new_roots = [root for root in component_roots if root not in visited]
            if not new_roots:
                return
            used_rows.add(row_id)
            selected.append(candidate_row)
            for root in new_roots:
                visited.add(root)
                queue.append(root)

        for row_id in anchor_rows:
            activate(row_id)
            if len(selected) >= max_add:
                break

        cursor = 0
        while cursor < len(queue) and len(selected) < max_add:
            root = queue[cursor]
            cursor += 1
            for row_id in incident.get(root, ()):
                activate(row_id)
                if len(selected) >= max_add:
                    break
        return selected

    def _add_structural_rank_restoring_pseudo_measurements(self, max_add: int) -> int:
        """Add only invalid real-measurement candidates that improve structural observability."""
        candidates = self._rank_restoring_candidate_measurements()
        selected_indices = self._rank_restoring_candidate_indices(candidates, max_add)
        if not selected_indices:
            return 0
        next_idx = self._next_measurement_idx()
        measurement_count_before = len(self.measurements)
        for local_idx in selected_indices:
            candidate = candidates[int(local_idx)]
            candidate.idx = next_idx
            next_idx += 1
            self.measurements.append(candidate)
        refreshed = self._incremental_update_active_measurement_indexes(
            self.measurements[measurement_count_before:]
        )
        if not refreshed:
            self._refresh_active_measurement_indexes()
        return len(selected_indices)

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
        meta = state_meta_at(self.state_meta, state_idx)
        if meta is None:
            return next_idx, 0
        name = meta.device_name
        device_type_code = int(getattr(meta, "device_type_code", 0))
        device_pos = int(getattr(meta, "device_pos", -1))
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
            name_rows = getattr(self, "_ac_measurement_plan_name_rows_by_type_code", {}).get(int(target_device_type_code))
            if name_rows is None:
                return -1
            rows = np.asarray(name_rows[1], dtype=np.int64)
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

        if meta.kind == "angle":
            return next_idx, 0
        if meta.kind == "voltage" and device_type_code == DEVICE_TYPE_CODES_ACNODE:
            return add(
                DEVICE_TYPE_CODES_ACNODE,
                "ACNode",
                name,
                device_pos,
                MEAS_TYPE_CODES_V,
                "V",
                node_voltage_seed(device_pos),
            )
        if meta.kind == "zero_current" and device_type_code == DEVICE_TYPE_CODES_ACZEROBRANCH:
            p, q = terminal_pq("zero_branch", ZERO_BRANCH_COLS, device_type_code, device_pos)
            next_idx, added_p = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_CODES_P_FROM, "P_FROM", p)
            next_idx, added_q = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_CODES_Q_FROM, "Q_FROM", q)
            next_idx, added_p_to = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_CODES_P_TO, "P_TO", -p)
            next_idx, added_q_to = add(device_type_code, "ACZeroBranch", name, device_pos, MEAS_TYPE_CODES_Q_TO, "Q_TO", -q)
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta.kind == "break_current" and device_type_code == DEVICE_TYPE_CODES_ACBREAK:
            p, q = terminal_pq("break", BREAK_COLS, device_type_code, device_pos)
            next_idx, added_p = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_CODES_P_FROM, "P_FROM", p)
            next_idx, added_q = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_CODES_Q_FROM, "Q_FROM", q)
            next_idx, added_p_to = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_CODES_P_TO, "P_TO", -p)
            next_idx, added_q_to = add(device_type_code, "ACBreak", name, device_pos, MEAS_TYPE_CODES_Q_TO, "Q_TO", -q)
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if meta.kind in ("generator_p", "generator_q") and device_type_code == DEVICE_TYPE_CODES_ACGENERATOR:
            p, q = generator_pq(device_pos)
            meas_type_code = MEAS_TYPE_CODES_P_GEN if meta.kind == "generator_p" else MEAS_TYPE_CODES_Q_GEN
            meas_type = "P_GEN" if meta.kind == "generator_p" else "Q_GEN"
            return add(device_type_code, "ACGenerator", name, device_pos, meas_type_code, meas_type, p if meta.kind == "generator_p" else q)
        if meta.kind in ("load_p", "load_q") and device_type_code == DEVICE_TYPE_CODES_ACLOAD:
            p, q = load_pq(device_pos)
            meas_type_code = MEAS_TYPE_CODES_P_LOAD if meta.kind == "load_p" else MEAS_TYPE_CODES_Q_LOAD
            meas_type = "P_LOAD" if meta.kind == "load_p" else "Q_LOAD"
            return add(device_type_code, "ACLoad", name, device_pos, meas_type_code, meas_type, p if meta.kind == "load_p" else q)
        return next_idx, 0

    def _add_power_balance_constraint_measurements(self) -> None:
        """Add nodal AC power-balance equations that tie P/Q states to the grid."""
        next_idx = self._next_measurement_idx()
        node_count = len(self.nodes)
        if node_count == 0:
            return
        row_count = 2 * node_count
        weight = 10.0
        node_names = np.asarray([node.name for node in self.nodes], dtype=object)
        names = np.empty(row_count, dtype=object)
        names[0::2] = [f"constraint_p_balance_{name}" for name in node_names]
        names[1::2] = [f"constraint_q_balance_{name}" for name in node_names]
        meas_type = np.empty(row_count, dtype=object)
        meas_type[0::2] = "P_BALANCE"
        meas_type[1::2] = "Q_BALANCE"
        balance_code = DEVICE_TYPE_CODES_ACPOWERBALANCE
        meas_type_code = np.empty(row_count, dtype=np.int16)
        meas_type_code[0::2] = MEAS_TYPE_CODES_P_BALANCE
        meas_type_code[1::2] = MEAS_TYPE_CODES_Q_BALANCE
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
            device_type=np.full(row_count, "ACPowerBalance", dtype=object),
            device_name=np.repeat(node_names, 2),
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
            return dict(cached)
        self._refresh_measurement_summary_cache()
        return dict(getattr(self, "_node_voltage_measurement_cache", {}))

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live incident AC branches, transformers, switches and zero branches."""
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        bus_solver_pos = getattr(self, "_ac_bus_solver_pos_by_topology_bus", None)
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "node incident degree build")
        if bus_solver_pos is None:
            self._warn_required_runtime_missing("_ac_bus_solver_pos_by_topology_bus", "node incident degree build")
        degree_array = np.zeros(len(self.nodes), dtype=np.int32)
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
                    np.add.at(degree_array, i_pos.astype(np.intp, copy=False), 1)
            if np.any(j_valid):
                j_pos = bus_solver_pos[j_bus[j_valid].astype(np.intp, copy=False)]
                j_pos = j_pos[j_pos >= 0]
                if j_pos.size:
                    np.add.at(degree_array, j_pos.astype(np.intp, copy=False), 1)
        return {node.idx: int(degree_array[pos]) for pos, node in enumerate(self.nodes)}

    def _select_reference_nodes(self):
        """Choose a measured high-degree V/angle reference for every live AC island."""
        references = []
        for island in getattr(self, "_ac_islands", getattr(self.network, "islands", [])):
            if not island.is_alive:
                continue
            candidates = [
                node
                for node in island.buses
                if node.idx in self.node_pos and node.idx in self.node_voltage_measurements
            ]
            if candidates:
                references.append(
                    max(
                        candidates,
                        key=lambda node: (self.node_degrees.get(node.idx, 0), -int(node.idx)),
                    )
                )
            elif island.slack_nodes:
                references.append(sorted(island.slack_nodes, key=lambda item: item.idx)[0])
            elif island.buses:
                references.append(sorted(island.buses, key=lambda item: item.idx)[0])
        return references

    def _reference_angle_offsets(self) -> Dict[int, float]:
        """Map each node position to the original island reference angle."""
        ref_by_island = {getattr(node, "isl", None): node for node in self.references}
        offsets: Dict[int, float] = {}
        for island in getattr(self, "_ac_islands", getattr(self.network, "islands", [])):
            if not getattr(island, "is_alive", False):
                continue
            ref = ref_by_island.get(getattr(island, "idx", None))
            if ref is None:
                continue
            offset = float(getattr(ref, "angle", 0.0) or 0.0)
            for node in island.buses:
                if node.idx in self.node_pos:
                    offsets[self.node_pos[node.idx]] = offset
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
            & (device_type_code == DEVICE_TYPE_CODES_ACNODE)
            & ((meas_type_code == MEAS_TYPE_CODES_ANGLE) | (meas_type_code == MEAS_TYPE_CODES_THETA))
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
        bus_solver_pos = getattr(self, "_ac_bus_solver_pos_by_topology_bus", None)
        if bus_solver_pos is None:
            bus_solver_pos = np.full(len(topology.bus_ids), -1, dtype=np.int32)
            active_bus_pos = np.flatnonzero(topology.bus_alive_mask).astype(np.intp, copy=False)
            if active_bus_pos.size:
                bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
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
        n = len(self.nodes)
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
                ref_pos = min(voltage_ref_positions, key=lambda pos: self.nodes[pos].idx)
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

    def _build_y_matrix(self) -> np.ndarray:
        """Build the estimator admittance matrix with the same stamps as load flow."""
        n = len(self.nodes)
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
        shunt_rows = self._required_array_attr("_ac_shunt_rows", dtype=np.int64, context="Y matrix build")
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
                shunt_rows_array = np.asarray(self.shunt_pos_array, dtype=np.int64)[stamp_mask][nonzero]
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
        n = len(self.nodes)
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
            theta = np.zeros(len(self.nodes), dtype=np.float64)
            voltage = np.ones(len(self.nodes), dtype=np.float64)
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

    def state_cols_for_nodes(self, nodes) -> Tuple[np.ndarray, np.ndarray]:
        """Return theta/voltage state columns for the given AC nodes."""
        theta_cols = np.full(len(nodes), -1, dtype=np.int32)
        voltage_cols = np.full(len(nodes), -1, dtype=np.int32)
        for idx, node in enumerate(nodes):
            pos = self.node_pos.get(node.idx)
            if pos is None:
                continue
            theta_cols[idx] = int(self.angle_col[int(pos)])
            voltage_cols[idx] = int(self.voltage_col[int(pos)])
        return theta_cols, voltage_cols

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
            x[self.base_shunt_q : self.n_state] = (
                self.initial_shunt_q_array if shunt_q is None else np.asarray(shunt_q, dtype=np.float64)
            )
        return x

    def _unpack_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Restore full theta/V arrays from the compact WLS state vector."""
        n = len(self.nodes)
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
        if not self.load_pos_array.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(self.load_pos_array, weights=x[self.base_load_p : self.base_load_q], minlength=self.n_nodes),
            np.bincount(self.load_pos_array, weights=x[self.base_load_q : self.base_shunt_q], minlength=self.n_nodes),
        )

    def _shunt_q_injections_from_state(self, x: np.ndarray) -> np.ndarray:
        """Return node-level reactive injections from V-control shunt Q states."""
        if not self.n_shunt_q:
            return np.zeros(self.n_nodes, dtype=np.float64)
        return np.bincount(
            self.shunt_q_pos_array,
            weights=x[self.base_shunt_q : self.n_state],
            minlength=self.n_nodes,
        )

    def _generator_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level generator totals from explicit P_GEN/Q_GEN states."""
        if not self.generator_pos_array.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(self.generator_pos_array, weights=x[self.base_gen_p : self.base_gen_q], minlength=self.n_nodes),
            np.bincount(self.generator_pos_array, weights=x[self.base_gen_q : self.base_load_p], minlength=self.n_nodes),
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
        p_load, q_load = self._load_power_totals_from_state(x)
        p_gen, q_gen = self._generator_power_totals_from_state(x)
        q_shunt = self._shunt_q_injections_from_state(x)
        return (
            s_network.real + p_switch + p_load - p_gen,
            s_network.imag + q_switch + q_load - q_gen - q_shunt,
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
        return name_array[rows.astype(np.intp, copy=False)].astype(str, copy=False)

    @staticmethod
    def _ppc_name_to_plan_pos(names, rows: np.ndarray) -> Dict[str, int]:
        row_names = ACStateEstimator._ppc_names_for_rows(names, rows)
        return {str(name): int(pos) for pos, name in enumerate(row_names)}

    @staticmethod
    def _sorted_alive_ppc_rows(table: np.ndarray, idx_col: int, alive_mask: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(np.asarray(alive_mask, dtype=bool)).astype(np.int64, copy=False)
        if rows.size <= 1:
            return rows
        order = np.argsort(np.asarray(table)[rows, idx_col], kind="stable")
        return rows[order].astype(np.int64, copy=False)

    def _build_ppc_state_index_arrays(self) -> None:
        """Build state-index lookups keyed by PPC row instead of device name."""
        profile_start = time.perf_counter() if self.profile_enabled else None
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "state index build")

        bus_solver_pos = np.full(len(topology.bus_ids), -1, dtype=np.int32)
        active_bus_pos = np.flatnonzero(topology.bus_alive_mask).astype(np.intp, copy=False)
        if active_bus_pos.size:
            bus_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
        self._ac_bus_solver_pos_by_topology_bus = bus_solver_pos

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

        def state_index_by_row(table_name: str, device_key: str, idx_col: int) -> np.ndarray:
            table = np.asarray(ppc.get(table_name, np.zeros((0, idx_col + 1))), dtype=np.float64)
            state = np.full(table.shape[0], -1, dtype=np.int32)
            device_topology = topology.devices.get(device_key)
            if table.shape[0] == 0 or device_topology is None:
                return state
            rows = self._sorted_alive_ppc_rows(table, idx_col, device_topology.alive_mask)
            if rows.size:
                state[rows.astype(np.intp, copy=False)] = np.arange(rows.size, dtype=np.int32)
            return state

        self._ac_generator_state_index_by_ppc_row = state_index_by_row("gen", "gen", GEN_COLS["idx"])
        self._ac_load_state_index_by_ppc_row = state_index_by_row("load", "load", LOAD_COLS["idx"])

        zero_branch = np.asarray(ppc.get("zero_branch", np.zeros((0, len(ZERO_BRANCH_COLS)))), dtype=np.float64)
        breaker = np.asarray(ppc.get("break", np.zeros((0, len(BREAK_COLS)))), dtype=np.float64)
        zero_current_pos = np.full(zero_branch.shape[0], -1, dtype=np.int32)
        break_current_pos = np.full(breaker.shape[0], -1, dtype=np.int32)
        zero_topology = topology.devices.get("zero_branch")
        break_topology = topology.devices.get("break")
        zero_rows = (
            self._sorted_alive_ppc_rows(zero_branch, ZERO_BRANCH_COLS["idx"], zero_topology.alive_mask)
            if zero_topology is not None
            else np.asarray([], dtype=np.int64)
        )
        break_rows = (
            self._sorted_alive_ppc_rows(breaker, BREAK_COLS["idx"], break_topology.alive_mask)
            if break_topology is not None
            else np.asarray([], dtype=np.int64)
        )
        if zero_rows.size:
            zero_current_pos[zero_rows.astype(np.intp, copy=False)] = np.arange(zero_rows.size, dtype=np.int32)
        if break_rows.size:
            break_current_pos[break_rows.astype(np.intp, copy=False)] = (
                zero_rows.size + np.arange(break_rows.size, dtype=np.int32)
            )
        self._ac_zero_branch_current_pos_by_ppc_row = zero_current_pos
        self._ac_break_current_pos_by_ppc_row = break_current_pos
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
        def state_cols_for_pos(values, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions, dtype=np.int64)
            if values is None or positions.size == 0:
                return np.full(positions.size, -1, dtype=np.int64)
            values_array = np.asarray(values, dtype=np.int32)
            out = np.full(positions.size, -1, dtype=np.int64)
            valid = (positions >= 0) & (positions < values_array.size)
            if np.any(valid):
                out[valid] = values_array[positions[valid].astype(np.intp, copy=False)]
            return out

        voltage_col = getattr(self, "voltage_col", None)
        angle_col = getattr(self, "angle_col", None)
        ppc = self._ac_ppc_dict()
        topology = ppc.get("_topology_arrays")
        if topology is None:
            self._warn_required_runtime_missing("PPC topology arrays", "measurement plan lookup build")
        if not hasattr(self, "_ac_node_solver_pos_by_ppc_row"):
            self._build_ppc_state_index_arrays()

        bus_solver_pos = getattr(self, "_ac_bus_solver_pos_by_topology_bus", None)
        node_solver_pos = getattr(self, "_ac_node_solver_pos_by_ppc_row", None)
        if bus_solver_pos is None or node_solver_pos is None:
            raise RuntimeError("AC PPC solver-position arrays are not available")

        node_rows = np.flatnonzero(np.asarray(node_solver_pos, dtype=np.int32) >= 0).astype(np.int64, copy=False)
        node_plan_pos = node_solver_pos[node_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._ac_node_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["bus_name"], node_rows)
        self._ac_node_plan_pos = node_plan_pos
        self._ac_node_plan_voltage_col = state_cols_for_pos(voltage_col, node_plan_pos)
        self._ac_node_plan_angle_col = state_cols_for_pos(angle_col, node_plan_pos)

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
        self._ac_branch_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["branch_name"], branch_rows)
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
        self._ac_transformer_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["transformer_name"], transformer_rows)
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
        self._ac_zero_branch_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["zero_branch_name"], zero_rows)
        self._ac_zero_branch_plan_i = zero_i
        self._ac_zero_branch_plan_j = zero_j
        self._ac_zero_branch_plan_current_pos = zero_current

        break_rows, break_i, break_j, break_current = zero_current_plan(
            "break",
            "break_name",
            getattr(self, "_ac_break_current_pos_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_break_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc.get("break_name", np.asarray([], dtype=object)), break_rows)
        self._ac_break_plan_i = break_i
        self._ac_break_plan_j = break_j
        self._ac_break_plan_current_pos = break_current

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
        self._ac_generator_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["gen_name"], gen_rows)
        self._ac_generator_plan_node_pos = gen_node_pos
        self._ac_generator_plan_voltage_col = state_cols_for_pos(voltage_col, gen_node_pos)
        self._ac_generator_plan_index = gen_state_index

        load_rows, load_node_pos, load_state_index = single_plan(
            "load",
            "load_name",
            getattr(self, "_ac_load_state_index_by_ppc_row", np.asarray([], dtype=np.int32)),
        )
        self._ac_load_plan_name_to_pos = self._ppc_name_to_plan_pos(ppc["load_name"], load_rows)
        self._ac_load_plan_node_pos = load_node_pos
        self._ac_load_plan_voltage_col = state_cols_for_pos(voltage_col, load_node_pos)
        self._ac_load_plan_index = load_state_index

        self._ac_measurement_plan_device_pos_by_type_code = {
            DEVICE_TYPE_CODES_ACNODE: self._ac_node_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACBRANCH: self._ac_branch_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACTRANSFORMER: self._ac_transformer_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACLOAD: self._ac_load_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACGENERATOR: self._ac_generator_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACZEROBRANCH: self._ac_zero_branch_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACBREAK: self._ac_break_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACZEROBRANCHCONSTRAINT: self._ac_zero_branch_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACBREAKCONSTRAINT: self._ac_break_plan_name_to_pos,
            DEVICE_TYPE_CODES_ACPOWERBALANCE: self._ac_node_plan_name_to_pos,
        }
        self._ac_measurement_plan_name_rows_by_type_code = {
            DEVICE_TYPE_CODES_ACNODE: (ppc["bus_name"], node_rows),
            DEVICE_TYPE_CODES_ACBRANCH: (ppc["branch_name"], branch_rows),
            DEVICE_TYPE_CODES_ACTRANSFORMER: (ppc["transformer_name"], transformer_rows),
            DEVICE_TYPE_CODES_ACLOAD: (ppc["load_name"], load_rows),
            DEVICE_TYPE_CODES_ACGENERATOR: (ppc["gen_name"], gen_rows),
            DEVICE_TYPE_CODES_ACZEROBRANCH: (ppc["zero_branch_name"], zero_rows),
            DEVICE_TYPE_CODES_ACBREAK: (ppc.get("break_name", np.asarray([], dtype=object)), break_rows),
            DEVICE_TYPE_CODES_ACZEROBRANCHCONSTRAINT: (ppc["zero_branch_name"], zero_rows),
            DEVICE_TYPE_CODES_ACBREAKCONSTRAINT: (ppc.get("break_name", np.asarray([], dtype=object)), break_rows),
            DEVICE_TYPE_CODES_ACPOWERBALANCE: (ppc["bus_name"], node_rows),
        }
        self._ac_measurement_plan_device_pos_by_type_code_id = self._measurement_plan_device_id_lookup_arrays()
        self._ac_branch_transformer_plan_kind_by_type_code = {
            DEVICE_TYPE_CODES_ACBRANCH: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACTRANSFORMER: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
        }
        self._ac_zero_current_plan_kind_by_type_code = {
            DEVICE_TYPE_CODES_ACZEROBRANCH: _AC_ZERO_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACBREAK: _AC_TERMINAL_MEAS_TYPE_LOOKUP,
        }
        self._ac_simple_plan_kind_by_type_code = {
            DEVICE_TYPE_CODES_ACNODE: _AC_NODE_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACGENERATOR: _AC_GENERATOR_SIMPLE_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACLOAD: _AC_LOAD_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACZEROBRANCHCONSTRAINT: _AC_CONSTRAINT_MEAS_TYPE_LOOKUP,
            DEVICE_TYPE_CODES_ACBREAKCONSTRAINT: _AC_CONSTRAINT_MEAS_TYPE_LOOKUP,
        }
        self._ac_generator_plan_kind_by_type_code = {
            DEVICE_TYPE_CODES_ACGENERATOR: _AC_GENERATOR_POWER_MEAS_TYPE_LOOKUP,
        }
        self._ac_balance_plan_kind_by_type_code = {
            DEVICE_TYPE_CODES_ACPOWERBALANCE: _AC_BALANCE_MEAS_TYPE_LOOKUP,
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
            "balance": {
                code: _measurement_type_code_lookup(kind_map)
                for code, kind_map in self._ac_balance_plan_kind_by_type_code.items()
            },
        }

    def _refresh_measurement_plan_state_columns(self) -> None:
        """Refresh only state-column arrays after the compact state layout is known."""
        def state_cols_for_pos(values, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions, dtype=np.int64)
            if values is None or positions.size == 0:
                return np.full(positions.size, -1, dtype=np.int64)
            values_array = np.asarray(values, dtype=np.int32)
            out = np.full(positions.size, -1, dtype=np.int64)
            valid = (positions >= 0) & (positions < values_array.size)
            if np.any(valid):
                out[valid] = values_array[positions[valid].astype(np.intp, copy=False)]
            return out

        self._ensure_measurement_plan_lookup_arrays()
        voltage_col = getattr(self, "voltage_col", None)
        angle_col = getattr(self, "angle_col", None)
        self._ac_node_plan_voltage_col = state_cols_for_pos(
            voltage_col,
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
        )
        self._ac_node_plan_angle_col = state_cols_for_pos(
            angle_col,
            getattr(self, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)),
        )
        self._ac_generator_plan_voltage_col = state_cols_for_pos(
            voltage_col,
            getattr(self, "_ac_generator_plan_node_pos", np.asarray([], dtype=np.int64)),
        )
        self._ac_load_plan_voltage_col = state_cols_for_pos(
            voltage_col,
            getattr(self, "_ac_load_plan_node_pos", np.asarray([], dtype=np.int64)),
        )

    def _measurement_plan_device_id_lookup_arrays(self) -> Dict[int, np.ndarray]:
        meas_ppc = self.meas_ppc
        device_names = np.asarray(meas_ppc.get("device_names", ()), dtype=object)
        if device_names.size == 0:
            return {}
        plan_name_rows = getattr(self, "_ac_measurement_plan_name_rows_by_type_code", None)
        if not plan_name_rows:
            warnings.warn(
                "AC SE measurement plan lookup requires PPC name-row arrays; string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        result: Dict[int, np.ndarray] = {}
        for code, name_rows in plan_name_rows.items():
            names, rows = name_rows
            row_names = self._ppc_names_for_rows(names, np.asarray(rows, dtype=np.int64))
            name_ids = self._meas_device_name_ids_for_ppc_names(meas_ppc, row_names)
            lookup = np.empty(device_names.size, dtype=np.int64)
            lookup.fill(-1)
            valid = name_ids >= 0
            if np.any(valid):
                plan_pos = np.arange(name_ids.size, dtype=np.int64)
                lookup[name_ids[valid].astype(np.intp, copy=False)] = plan_pos[valid]
            result[int(code)] = lookup
        return result

    def _measurement_device_id_maps_for(
        self,
        device_pos_by_type_code: Dict[int, Dict[str, int]],
    ) -> Dict[int, np.ndarray]:
        warnings.warn(
            "AC SE custom measurement device maps require indexed PPC maps; string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

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
        if not hasattr(self, "_ac_measurement_plan_device_pos_by_type_code"):
            self._build_measurement_plan_lookup_arrays()

    def _measurement_plan_table(
        self,
        measurements,
        meas_kind_by_type_code: Dict[int, Dict[str, int]],
        device_pos_by_type_code: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        self._ensure_measurement_plan_lookup_arrays()
        device_maps = device_pos_by_type_code or self._ac_measurement_plan_device_pos_by_type_code
        device_id_maps = (
            getattr(self, "_ac_measurement_plan_device_pos_by_type_code_id", {})
            if device_pos_by_type_code is None
            else self._measurement_device_id_maps_for(device_maps)
        )
        return build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=device_maps,
            device_pos_by_type_code_id=device_id_maps,
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
                "power_rows": empty_i,
                "power_is_p": empty_b,
                "power_own": empty_i,
                "power_other": empty_i,
                "power_y_self": empty_c,
                "power_y_mutual": empty_c,
                "current_rows": empty_i,
                "current_own": empty_i,
                "current_other": empty_i,
                "current_y_self": empty_c,
                "current_y_mutual": empty_c,
            }

        def build_device_rows(device_code, device_pos, i_array, j_array, yff, yft, ytf, ytt):
            rows = row[(code == device_code) & handled_mask]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_from = row_kind == MEAS_TYPE_CODES_V_FROM
            v_to = row_kind == MEAS_TYPE_CODES_V_TO
            p_from = row_kind == MEAS_TYPE_CODES_P_FROM
            q_from = row_kind == MEAS_TYPE_CODES_Q_FROM
            p_to = row_kind == MEAS_TYPE_CODES_P_TO
            q_to = row_kind == MEAS_TYPE_CODES_Q_TO
            i_from = row_kind == MEAS_TYPE_CODES_I_FROM
            i_to = row_kind == MEAS_TYPE_CODES_I_TO
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
            DEVICE_TYPE_CODES_ACBRANCH,
            plan_table.device_pos,
            self._ac_branch_plan_i,
            self._ac_branch_plan_j,
            self._ac_branch_plan_yff,
            self._ac_branch_plan_yft,
            self._ac_branch_plan_ytf,
            self._ac_branch_plan_ytt,
        )
        transformer_plan = build_device_rows(
            DEVICE_TYPE_CODES_ACTRANSFORMER,
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
            "power_rows": power_rows,
            "power_is_p": power_is_p,
            "power_own": power_own,
            "power_other": power_other,
            "power_y_self": power_y_self,
            "power_y_mutual": power_y_mutual,
            "current_rows": current_rows,
            "current_own": current_own,
            "current_other": current_other,
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
        if out is None:
            values = np.zeros(n_meas, dtype=np.float64)
        else:
            values = out
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
        vectorized_balance_rows = self._fill_balance_values_vectorized(
            values,
            vector_plans["balance"],
            x,
            voltage_complex,
            switch_current,
        )
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_zero_rows
            | vectorized_simple_rows
            | vectorized_generator_rows
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
            v_from = row_kind == MEAS_TYPE_CODES_V_FROM
            v_to = row_kind == MEAS_TYPE_CODES_V_TO
            p_from = row_kind == MEAS_TYPE_CODES_P_FROM
            q_from = row_kind == MEAS_TYPE_CODES_Q_FROM
            p_to = row_kind == MEAS_TYPE_CODES_P_TO
            q_to = row_kind == MEAS_TYPE_CODES_Q_TO
            i_from = row_kind == MEAS_TYPE_CODES_I_FROM
            i_to = row_kind == MEAS_TYPE_CODES_I_TO
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
                "v_diff_rows": rows[row_kind == MEAS_TYPE_CODES_V_DIFF],
                "angle_diff_rows": rows[
                    (row_kind == MEAS_TYPE_CODES_ANGLE_DIFF) | (row_kind == MEAS_TYPE_CODES_THETA_DIFF)
                ],
                "i": i,
                "j": j,
                "kind": row_kind,
            }

        zero_plan = build_device_rows(
            DEVICE_TYPE_CODES_ACZEROBRANCH,
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
            self._ac_zero_branch_plan_current_pos,
        )
        break_plan = build_device_rows(
            DEVICE_TYPE_CODES_ACBREAK,
            self._ac_break_plan_i,
            self._ac_break_plan_j,
            self._ac_break_plan_current_pos,
        )
        zero_rows = row[(code == DEVICE_TYPE_CODES_ACZEROBRANCH) & handled_mask]
        zero_pos = plan_table.device_pos[zero_rows]
        zero_kind = kind[zero_rows]
        zero_i = self._ac_zero_branch_plan_i[zero_pos]
        zero_j = self._ac_zero_branch_plan_j[zero_pos]
        v_diff = zero_kind == MEAS_TYPE_CODES_V_DIFF
        angle_diff = (zero_kind == MEAS_TYPE_CODES_ANGLE_DIFF) | (zero_kind == MEAS_TYPE_CODES_THETA_DIFF)
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
                self._add_indexed_values(H, rows, cols, values)
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
                np.ones(voltage_rows.size, dtype=np.float64),
            )

        rows = plan["power_rows"]
        if rows.size:
            own = plan["power_own"]
            other = plan["power_other"]
            y_self_conj = np.conj(plan["power_y_self"])
            y_mutual_conj = np.conj(plan["power_y_mutual"])
            off = y_mutual_conj * voltage_complex[own] * np.conj(voltage_complex[other])
            dtheta_own = 1j * off
            dtheta_other = -1j * off
            dvoltage_own = 2.0 * y_self_conj * voltage[own] + off / voltage[own]
            dvoltage_other = off / voltage[other]
            is_p = plan["power_is_p"]

            own_angle_cols = self.angle_col[own]
            other_angle_cols = self.angle_col[other]
            own_voltage_cols = self.voltage_col[own]
            other_voltage_cols = self.voltage_col[other]
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
            own_angle_cols = self.angle_col[own]
            other_angle_cols = self.angle_col[other]
            own_voltage_cols = self.voltage_col[own]
            other_voltage_cols = self.voltage_col[other]
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

        node_rows = row[(code == DEVICE_TYPE_CODES_ACNODE) & handled]
        node_pos = self._ac_node_plan_pos[device_pos[node_rows]]
        node_kind = kind[node_rows]
        node_v = node_kind == MEAS_TYPE_CODES_V
        node_angle = (node_kind == MEAS_TYPE_CODES_ANGLE) | (node_kind == MEAS_TYPE_CODES_THETA)

        gen_rows = row[(code == DEVICE_TYPE_CODES_ACGENERATOR) & handled]
        gen_pos = self._ac_generator_plan_node_pos[device_pos[gen_rows]]

        load_rows_all = row[(code == DEVICE_TYPE_CODES_ACLOAD) & handled]
        load_plan_pos = device_pos[load_rows_all]
        load_node_pos = self._ac_load_plan_node_pos[load_plan_pos]
        load_kind_all = kind[load_rows_all]
        load_v = load_kind_all == MEAS_TYPE_CODES_V_LOAD
        load_state = load_kind_all != MEAS_TYPE_CODES_V_LOAD

        def constraint_rows_for(device_code, i_array, j_array):
            rows = row[(code == device_code) & handled]
            pos = device_pos[rows]
            row_kind = kind[rows]
            i = i_array[pos]
            j = j_array[pos]
            v_diff = row_kind == MEAS_TYPE_CODES_V_DIFF
            angle_diff = (row_kind == MEAS_TYPE_CODES_ANGLE_DIFF) | (row_kind == MEAS_TYPE_CODES_THETA_DIFF)
            return rows[v_diff], i[v_diff], j[v_diff], rows[angle_diff], i[angle_diff], j[angle_diff]

        zero_v_rows, zero_v_i, zero_v_j, zero_a_rows, zero_a_i, zero_a_j = constraint_rows_for(
            DEVICE_TYPE_CODES_ACZEROBRANCHCONSTRAINT,
            self._ac_zero_branch_plan_i,
            self._ac_zero_branch_plan_j,
        )
        break_v_rows, break_v_i, break_v_j, break_a_rows, break_a_i, break_a_j = constraint_rows_for(
            DEVICE_TYPE_CODES_ACBREAKCONSTRAINT,
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
            p_mask = kind == MEAS_TYPE_CODES_P_LOAD
            q_mask = kind == MEAS_TYPE_CODES_Q_LOAD
            i_mask = kind == MEAS_TYPE_CODES_I_LOAD
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
            p_mask = kind == MEAS_TYPE_CODES_P_LOAD
            q_mask = kind == MEAS_TYPE_CODES_Q_LOAD
            i_mask = kind == MEAS_TYPE_CODES_I_LOAD
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
                    valid,
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.base_load_q + i_idx,
                    np.divide(q_i, s_abs * vm_i, out=np.zeros_like(q_i), where=valid),
                    valid,
                )
                self._add_indexed_values(
                    H,
                    i_rows,
                    self.voltage_col[pos[i_mask]],
                    np.divide(-s_abs, vm_i * vm_i, out=np.zeros_like(s_abs), where=valid),
                    valid,
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
                {DEVICE_TYPE_CODES_ACPOWERBALANCE: _AC_BALANCE_MEAS_TYPE_LOOKUP},
            ),
        )
        plan = self._build_balance_measurement_plan_from_table(plan_table)
        if measurements is None:
            self._active_balance_measurement_plan = plan
        return plan

    def _build_balance_measurement_plan_from_table(self, plan_table: MeasurementPlanTable) -> Dict[str, np.ndarray]:
        """Build the balance plan from a precomputed measurement plan table."""
        rows = plan_table.row[
            (plan_table.device_type_code == DEVICE_TYPE_CODES_ACPOWERBALANCE) & plan_table.handled
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
            }
        device_pos = plan_table.device_pos[rows]
        pos = self._ac_node_plan_pos[device_pos]
        kind = plan_table.meas_kind[rows]
        p_row_by_pos = np.empty(len(self.nodes), dtype=np.int32)
        q_row_by_pos = np.empty(len(self.nodes), dtype=np.int32)
        p_row_by_pos.fill(-1)
        q_row_by_pos.fill(-1)
        if rows.size:
            p_mask = kind == MEAS_TYPE_CODES_P_BALANCE
            q_mask = kind == MEAS_TYPE_CODES_Q_BALANCE
            p_row_by_pos[pos[p_mask]] = rows[p_mask].astype(np.int32, copy=False)
            q_row_by_pos[pos[q_mask]] = rows[q_mask].astype(np.int32, copy=False)

        balance_pos = np.flatnonzero((p_row_by_pos >= 0) | (q_row_by_pos >= 0)).astype(np.int64, copy=False)
        y_balance, y_nodes, y_nodes_i64, y_conj = self._balance_y_arrays(balance_pos)
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
        values[rows] = np.where(plan["kind"] == MEAS_TYPE_CODES_P_BALANCE, p_balance[pos], q_balance[pos])
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
        p_row_by_pos = plan["p_row_by_pos"]
        q_row_by_pos = plan["q_row_by_pos"]
        balance_pos = np.asarray(plan.get("balance_pos_i64", pos), dtype=np.int64)
        p_rows_for_balance = p_row_by_pos[balance_pos] if balance_pos.size else np.asarray([], dtype=np.int32)
        q_rows_for_balance = q_row_by_pos[balance_pos] if balance_pos.size else np.asarray([], dtype=np.int32)

        y_balance = plan["y_balance"]
        if y_balance.size:
            y_nodes = plan["y_nodes_i64"]
            y_conj = plan["y_conj"]
            y_pos = balance_pos[y_balance]
            exp_delta = np.exp(1j * (theta[y_pos] - theta[y_nodes]))
            term = y_conj * voltage[y_pos] * voltage[y_nodes] * exp_delta

            off_mask = y_nodes != y_pos
            off_sum = np.zeros(balance_pos.size, dtype=np.complex128)
            if np.any(off_mask):
                off_balance = y_balance[off_mask]
                off_nodes = y_nodes[off_mask]
                off_term = term[off_mask]
                off_sum = self._complex_bincount(off_balance, off_term, balance_pos.size)

                theta_values = -1j * off_term
                theta_cols = self.angle_col[off_nodes]
                theta_p_rows = p_rows_for_balance[off_balance]
                theta_q_rows = q_rows_for_balance[off_balance]

                voltage_values = y_conj[off_mask] * voltage[y_pos[off_mask]] * exp_delta[off_mask]
                voltage_cols = self.voltage_col[off_nodes]
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

            sum_all = self._complex_bincount(y_balance, y_conj * voltage[y_nodes] * exp_delta, balance_pos.size)
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
                    self.angle_col[balance_pos],
                    own_theta_values.real,
                ),
                (
                    q_rows_for_balance,
                    self.angle_col[balance_pos],
                    own_theta_values.imag,
                ),
                (
                    p_rows_for_balance,
                    self.voltage_col[balance_pos],
                    own_voltage_values.real,
                ),
                (
                    q_rows_for_balance,
                    self.voltage_col[balance_pos],
                    own_voltage_values.imag,
                ),
            ),
        )

        if self.n_switch_current:
            end_pos = self.switch_balance_end_pos
            end_current_idx = self.switch_balance_end_current_idx
            sign = self.switch_balance_sign
            current = switch_current[end_current_idx] * sign
            s = voltage_complex[end_pos] * np.conj(current)
            dS_dtheta = 1j * s
            dS_dV = np.divide(s, voltage[end_pos], out=np.zeros_like(s), where=np.abs(voltage[end_pos]) > 1e-12)
            dS_dIr = sign * voltage_complex[end_pos]
            dS_dIi = -1j * sign * voltage_complex[end_pos]
            p_rows = p_row_by_pos[end_pos]
            q_rows = q_row_by_pos[end_pos]

            self._add_indexed_value_blocks(
                H,
                (
                    (p_rows, self.angle_col[end_pos], dS_dtheta.real),
                    (q_rows, self.angle_col[end_pos], dS_dtheta.imag),
                    (p_rows, self.voltage_col[end_pos], dS_dV.real),
                    (q_rows, self.voltage_col[end_pos], dS_dV.imag),
                    (p_rows, self.base_switch_re + end_current_idx, dS_dIr.real),
                    (q_rows, self.base_switch_re + end_current_idx, dS_dIr.imag),
                    (p_rows, self.base_switch_im + end_current_idx, dS_dIi.real),
                    (q_rows, self.base_switch_im + end_current_idx, dS_dIi.imag),
                ),
            )

        if self.n_generator_power:
            gen_idx = self.generator_power_idx_array
            gen_p_rows = p_row_by_pos[self.generator_pos_array]
            gen_q_rows = q_row_by_pos[self.generator_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (gen_p_rows, self.base_gen_p + gen_idx, self.generator_balance_minus_ones),
                    (gen_q_rows, self.base_gen_q + gen_idx, self.generator_balance_minus_ones),
                ),
            )

        if self.n_load_power:
            load_idx = self.load_power_idx_array
            load_p_rows = p_row_by_pos[self.load_pos_array]
            load_q_rows = q_row_by_pos[self.load_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (load_p_rows, self.base_load_p + load_idx, self.load_balance_ones),
                    (load_q_rows, self.base_load_q + load_idx, self.load_balance_ones),
                ),
            )

        if self.n_shunt_q:
            shunt_q_rows = q_row_by_pos[self.shunt_q_pos_array]
            self._add_indexed_value_blocks(
                H,
                (
                    (
                        shunt_q_rows,
                        self.base_shunt_q + self.shunt_q_idx_array,
                        self.shunt_balance_minus_ones,
                    ),
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
            (plan_table.device_type_code == DEVICE_TYPE_CODES_ACGENERATOR) & plan_table.handled
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
        p_mask = kind == MEAS_TYPE_CODES_P_GEN
        q_mask = kind == MEAS_TYPE_CODES_Q_GEN
        i_mask = kind == MEAS_TYPE_CODES_I_GEN
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
        p_mask = kind == MEAS_TYPE_CODES_P_GEN
        q_mask = kind == MEAS_TYPE_CODES_Q_GEN
        i_mask = kind == MEAS_TYPE_CODES_I_GEN
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
                valid,
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.base_gen_q + i_idx,
                np.divide(q_i, s_abs * vm, out=np.zeros_like(q_i), where=valid),
                valid,
            )
            self._add_indexed_values(
                H,
                i_rows,
                self.voltage_col[pos[i_mask]],
                np.divide(-s_abs, vm * vm, out=np.zeros_like(s_abs), where=valid),
                valid,
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
        vectorized_balance_rows = self._fill_balance_jacobian_vectorized(
            H,
            vector_plans["balance"],
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_rows = (
            vectorized_branch_rows
            | vectorized_simple_rows
            | vectorized_zero_rows
            | vectorized_generator_rows
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

    def jacobian(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Assemble the WLS measurement Jacobian as a dense array for diagnostics/tests."""
        return self._assemble_jacobian(x, measurements, sparse=False)

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
            if active_measurement_run and lower_normal_plan is None and issparse(H):
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
            if lower_normal_plan is None and normal_pattern is None and issparse(H):
                start = time.perf_counter() if self.profile_enabled else None
                normal_pattern = _normal_equation_structural_pattern(H)
                if start is not None:
                    self._record_profile_time("solve.normal_pattern", time.perf_counter() - start)
                if active_measurement_run:
                    self._active_normal_pattern = normal_pattern
                if observability_cache is not None:
                    observability_cache["normal_pattern"] = normal_pattern
            lower_plan_can_weight_rhs = lower_normal_plan is not None and active_measurement_run
            if weighted_residual is not None and not lower_plan_can_weight_rhs:
                np.multiply(weight, residual, out=weighted_residual)
            start = time.perf_counter() if self.profile_enabled else None
            if lower_normal_plan is not None:
                gain, rhs = lower_normal_plan.assemble(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=None if lower_plan_can_weight_rhs else weighted_residual,
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
        array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
        if not array_only_result:
            self.apply_state(x)
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        result_table = self._common_measurement_plan_table(measurement_plan_tables).table
        if array_only_result:
            result_measurements = []
        elif source_measurements is not None:
            result_measurements = source_measurements
        elif active_measurement_run:
            result_measurements = self.active_measurements
        else:
            result_measurements = TableBackedMeasurementList(
                result_table,
                normalized=getattr(self.measurements, "normalized", True),
            )
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
            self.se_result = build_seresult_summary(
                result,
                bad_items=bad_items,
                all_measurements=self.measurements,
            )
            return self.se_result
        self.se_result = SEResult.from_estimate_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurements=self.measurements,
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
        previous_runtime_mode = bool(getattr(self, "_array_only_runtime", False))
        self._array_only_runtime = array_only
        if not self._prepared:
            try:
                self.prepare()
            except Exception:
                self._array_only_runtime = previous_runtime_mode
                raise
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        needs_bad_data = (not skip_bad_data) and not array_only
        if observability is None:
            observability = self.observability_analysis()
        self.observability_result = observability
        removed: List[BadDataItem] = []
        previous_array_only = bool(getattr(self, "_array_only_estimate_result", False))
        self._array_only_estimate_result = array_only
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
            self._array_only_runtime = previous_runtime_mode
        self.estimate_result = result
        self.removed_bad_data = removed
        if skip_bad_data or array_only:
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
        for pos, node in enumerate(self.nodes):
            node.angle = float(theta[pos])
            node.voltage = float(voltage[pos])
            for member in getattr(node, "nodes", ()):
                member.angle = node.angle
                member.voltage = node.voltage
        for idx, gen in enumerate(self.generator_order):
            gen.p = float(x[self.base_gen_p + idx])
            gen.q = float(x[self.base_gen_q + idx])
        for idx, load in enumerate(self.load_order):
            load.p = float(x[self.base_load_p + idx])
            load.q = float(x[self.base_load_q + idx])
        for idx, shunt in enumerate(self.voltage_control_shunt_order):
            shunt.p = 0.0
            shunt.q = float(x[self.base_shunt_q + idx])
            voltage_value = float(getattr(getattr(shunt, "node_obj", None), "voltage", 1.0) or 1.0)
            shunt.current = abs(shunt.q) / voltage_value if abs(voltage_value) > self.min_current_voltage else 0.0

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        theta, voltage = self._unpack_state(x)
        print("Estimated states:")
        for pos, node in enumerate(self.nodes[:limit]):
            print(f"  {node.name:10s} V={voltage[pos]:.9f} theta={theta[pos]:.9f} rad")
        if len(self.nodes) > limit:
            print(f"  ... {len(self.nodes) - limit} more nodes")


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
