import argparse
import contextlib
import io
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import maximum_bipartite_matching as sp_maximum_bipartite_matching


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.dc_lf import DCPowerFlowCalc
from model import topology as network_topology
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    BUS_COLS as DC_BUS_COLS,
    CTRL_I as DC_CTRL_I,
    CTRL_P as DC_CTRL_P,
    CTRL_V as DC_CTRL_V,
    DCDC_COLS as DC_DCDC_COLS,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    sanitize_dc_voltage_setpoints,
)
from model.ppc_topology import build_dc_ppc_with_topology_from_e_file, ensure_dc_ppc_topology
from model.meas_array_model import (
    MEAS_COLS,
    attach_device_pos_from_name_arrays,
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_table_from_meas_ppc,
)
from model.meas_type import (
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCBreakConstraint,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCGenerator,
    DEVICE_TYPE_DCLoad,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCZeroBranch,
    DEVICE_TYPE_DCZeroBranchConstraint,
    MEAS_TYPE_CODES,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_I_LOAD,
    MEAS_TYPE_I_GEN,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_GEN,
    MEAS_TYPE_P_LOAD,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_V,
    MEAS_TYPE_V_DIFF,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_V_GEN,
    MEAS_TYPE_V_LOAD,
    MEAS_TYPE_V_TO,
)
from model.meas_model import (
    BadDataItem,
    EstimateResult,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_PSEUDO,
    MeasurementTable,
    MeasurementTableView,
    ObservabilityResult,
    measurement_from_table_row,
    measurement_table_status_code,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from secore.se_math import (
    CHOLMOD_ANALYZE_AAT,
    CHOLMOD_CHOLESKY_AAT,
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    LowerNormalEquationCscPlan,
    NormalEquationAssemblyPlan,
    NormalEquationSolver,
    SparseJacobianBuilder,
    _normal_equation_structural_pattern,
    build_normal_equations,
    full_normal_equation_from_lower,
    is_sparse_matrix,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    observability_rank_details,
    observability_weak_direction,
    sparse_structural_rank,
    targeted_redundancy_count,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata
from secore.se_array_plan import (
    MeasurementPlanTable,
    append_active_measurement_view,
    build_active_measurement_view,
    concat_measurement_tables,
    measurement_table_take,
    rows_by_device_type_code,
)
from secore.se_result import (
    SEResult,
    build_seresult_full_from_table,
    build_seresult_summary_from_table,
    normalize_seresult_result_mode,
)

DEFAULT_CASE = model_file("dc", "dc_net_30.e")
DEFAULT_MEAS = measurement_file("dc", "dc_net_30.meas")

_DC_TERMINAL_MEAS_TYPE_CODES = np.asarray(
    [MEAS_TYPE_P_FROM, MEAS_TYPE_V_FROM, MEAS_TYPE_I_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_V_TO, MEAS_TYPE_I_TO],
    dtype=np.int16,
)
_DC_LOAD_MEAS_TYPE_CODES = np.asarray([MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD], dtype=np.int16)
_DC_GEN_MEAS_TYPE_CODES = np.asarray([MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN], dtype=np.int16)
_DC_NODE_MEAS_TYPE_CODES = np.asarray([MEAS_TYPE_V], dtype=np.int16)
_DC_CONSTRAINT_MEAS_TYPE_CODES = np.asarray([MEAS_TYPE_V_DIFF], dtype=np.int16)
_MAX_MEAS_TYPE_CODE = max(MEAS_TYPE_CODES.values())
_VOLTAGE_MEASUREMENT_TYPE_CODES = np.asarray(
    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD],
    dtype=np.int16,
)
_OBSERVABILITY_RESULT_CACHE = {}
_AAT_NORMAL_SOLVER_MIN_STATE = 2000
_ACTIVE_DEVICE_KEY_POS_BITS = 40
_ACTIVE_MEASUREMENT_KEY_MEAS_BITS = 16


def _file_cache_key(file_name: Path) -> Tuple[Path, int, int]:
    path = Path(file_name).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _measurement_type_code_lookup(meas_type_codes: Sequence[int]) -> np.ndarray:
    lookup = np.full(_MAX_MEAS_TYPE_CODE + 1, -1, dtype=np.int16)
    for value in np.asarray(meas_type_codes, dtype=np.int16):
        code = int(value)
        if 0 <= int(code) < lookup.size:
            lookup[int(code)] = int(value)
    return lookup


def _is_voltage_measurement_type_code(meas_type_code: int) -> bool:
    code = int(meas_type_code)
    return (
        code == MEAS_TYPE_V
        or code == MEAS_TYPE_V_FROM
        or code == MEAS_TYPE_V_TO
        or code == MEAS_TYPE_V_GEN
        or code == MEAS_TYPE_V_LOAD
    )


class _PackedMeasurementKeyCache:
    """Set-like packed-key cache that keeps the common path on int64 arrays."""

    __slots__ = ("estimator", "_keys")

    def __init__(self, estimator: "DCStateEstimator") -> None:
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

    def add(self, key) -> None:
        self.extend_array(np.asarray([int(key)], dtype=np.int64))

    def update(self, keys) -> None:
        self.extend_array(keys)


class _NodeIndexedValueMap(Mapping):
    """Array-backed mapping from node idx to a cached scalar value."""

    __slots__ = ("_node_ids", "_values", "_dict")

    def __init__(self, node_idx_by_pos: np.ndarray, values_by_pos: np.ndarray, valid_mask: np.ndarray) -> None:
        node_idx_by_pos = np.asarray(node_idx_by_pos, dtype=np.int64)
        values_by_pos = np.asarray(values_by_pos)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        valid = np.flatnonzero(valid_mask).astype(np.int64, copy=False)
        if valid.size:
            valid = valid[valid < min(node_idx_by_pos.size, values_by_pos.size)]
            self._node_ids = node_idx_by_pos[valid.astype(np.intp, copy=False)].astype(np.int64, copy=False)
            self._values = values_by_pos[valid.astype(np.intp, copy=False)]
        else:
            self._node_ids = np.empty(0, dtype=np.int64)
            self._values = np.empty(0, dtype=values_by_pos.dtype)
        self._dict = None

    @staticmethod
    def _scalar(value):
        return value.item() if hasattr(value, "item") else value

    def _materialize(self) -> Dict[int, object]:
        cache = self._dict
        if cache is None:
            cache = {
                int(node_idx): self._scalar(value)
                for node_idx, value in zip(self._node_ids, self._values)
            }
            self._dict = cache
        return cache

    def __getitem__(self, key: int):
        node_id = int(key)
        pos = int(np.searchsorted(self._node_ids, node_id))
        if pos < self._node_ids.size and int(self._node_ids[pos]) == node_id:
            return self._scalar(self._values[pos])
        raise KeyError(key)

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self) -> int:
        return int(self._node_ids.size)

    def __bool__(self) -> bool:
        return bool(self._node_ids.size)

    def __contains__(self, key) -> bool:
        node_id = int(key)
        pos = int(np.searchsorted(self._node_ids, node_id))
        return bool(pos < self._node_ids.size and int(self._node_ids[pos]) == node_id)

    def get(self, key, default=None):
        node_id = int(key)
        pos = int(np.searchsorted(self._node_ids, node_id))
        if pos < self._node_ids.size and int(self._node_ids[pos]) == node_id:
            return self._scalar(self._values[pos])
        return default

    def __eq__(self, other) -> bool:
        return self._materialize() == other


def _measurement_table_from_array_view(measurements) -> MeasurementTable:
    if isinstance(measurements, MeasurementTable):
        return measurements
    if isinstance(measurements, dict) and measurements.get("format") == "meas_ppc_v1":
        return measurement_table_from_meas_ppc(measurements, include_strings=False)
    table = getattr(measurements, "table", None)
    if isinstance(table, MeasurementTable):
        return table
    raise RuntimeError(
        "DC SE requires PPC-backed MeasurementTable data; Measurement object parsing is disabled."
    )

def load_dc_ppc_from_e_file(file_name) -> Dict:
    """Read a DC E file into PPC with topology arrays attached."""
    return build_dc_ppc_with_topology_from_e_file(file_name)


class _LazyDCSENetwork(SimpleNamespace):
    """DC SE network namespace that builds its device/object graph lazily.

    Array-mode SE reads only the scalar scale factors (``p_base`` etc.) and the
    ppc/topology arrays, which are populated eagerly. The node and per-device
    ``SimpleNamespace`` graph plus the topology-derived dicts/alive-maps (used
    only by full-result write-back and external consumers) are built on first
    access of any of those attributes, so cold-start array runs avoid
    constructing ~10^5 throwaway objects.
    """

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails. Skip dunder probes so
        # copy/pickle/introspection never force materialization.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        state = self.__dict__
        materialize = state.get("_materialize")
        if materialize is None or state.get("_materialized"):
            raise AttributeError(name)
        state["_materialized"] = True
        materialize(self)
        try:
            return state[name]
        except KeyError:
            raise AttributeError(name)


def _build_dc_se_ppc_namespace(ppc: Dict) -> SimpleNamespace:
    """Create the DC SE PPC namespace, deferring the device/object graph."""
    ensure_dc_ppc_topology(ppc)
    topology_arrays = ppc["_topology_arrays"]
    base = ppc["base"]

    def _materialize(network) -> None:
        node_ids = np.asarray(topology_arrays.node_ids, dtype=np.int64)
        bus_names = np.asarray(ppc.get("bus_name", ()), dtype=object)
        network.nodes = [
            SimpleNamespace(idx=int(node_id), name=str(bus_names[pos]) if pos < bus_names.size else f"bus_{int(node_id)}")
            for pos, node_id in enumerate(node_ids)
        ]

        def device_objects(table_key: str, name_key: str, cols: Dict[str, int]):
            table = np.asarray(ppc.get(table_key, np.zeros((0, len(cols)))), dtype=np.float64)
            names = np.asarray(ppc.get(name_key, ()), dtype=object)
            idx_col = int(cols.get("idx", 0))
            return [
                SimpleNamespace(
                    idx=int(table[row, idx_col]) if table.size else int(row),
                    name=str(names[row]) if row < names.size else f"{table_key}_{int(table[row, idx_col]) if table.size else int(row)}",
                )
                for row in range(int(table.shape[0]) if table.ndim == 2 else 0)
            ]

        network.branches = device_objects("branch", "branch_name", DC_BRANCH_COLS)
        network.generators = device_objects("gen", "gen_name", DC_GEN_COLS)
        network.loads = device_objects("load", "load_name", DC_LOAD_COLS)
        network.zero_branches = device_objects("zero_branch", "zero_branch_name", DC_ZERO_BRANCH_COLS)
        network.switches = device_objects("switch", "switch_name", DC_SWITCH_COLS)
        network.breakers = device_objects("break", "break_name", DC_SWITCH_COLS)
        network.dcdc_converters = device_objects("dcdc", "dcdc_name", DC_DCDC_COLS)
        network_topology.apply_dc_topology_arrays(network, topology_arrays, compact=True, populate_device_links=False)

    network = _LazyDCSENetwork(
        _se_lightweight=True,
        ppc=ppc,
        _array_model=ppc,
        base=base,
        topology=topology_arrays,
        _topology_arrays=topology_arrays,
        p_base=float(base["p_base"]),
        p_base_kW=float(base["p_base_kW"]),
        u_scale=float(base["u_scale"]),
        p_scale=float(base["p_scale"]),
        i_scale=float(base["i_scale"]),
    )
    network.__dict__["_materialize"] = _materialize
    network.__dict__["_materialized"] = False
    return network


class DCStateEstimator:
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
        network: Optional[object] = None,
        measurements: Optional[object] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
        auto_prepare: bool = True,
        matrix_dump_dir: Optional[Path] = None,
        power_flow_linear_solver: Optional[str] = None,
    ):
        self.profile_enabled = bool(profile)
        self.profile_times: Dict[str, float] = {}
        self.matrix_dump_dir = None if matrix_dump_dir is None else Path(matrix_dump_dir)
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
        self._array_only_runtime = True
        self._array_only_estimate_result = True
        solver_name = str(power_flow_linear_solver).strip().lower() if power_flow_linear_solver is not None else ""
        self.power_flow_linear_solver = solver_name or None
        self.observability_result = None
        self._initial_observability_cache = None
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

    def _dc_ppc_dict(self) -> Dict:
        ppc = getattr(self.network, "ppc", None)
        if not isinstance(ppc, dict):
            ppc = getattr(self.network, "_array_model", None)
        if not isinstance(ppc, dict):
            raise RuntimeError("DC SE requires a PPC-backed DC network")
        sanitize_dc_voltage_setpoints(ppc)
        return ppc

    @staticmethod
    def _ppc_names_for_rows(names, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        names_array = np.asarray(names if names is not None else (), dtype=object)
        if names_array.size and rows.size and int(np.max(rows)) < names_array.size:
            return names_array[rows.astype(np.intp, copy=False)].astype(object, copy=False)
        return np.asarray([], dtype=object)

    @staticmethod
    def _ppc_names_for_rows_or_idx(names, rows: np.ndarray, prefix: str, idx_values: np.ndarray) -> np.ndarray:
        row_names = DCStateEstimator._ppc_names_for_rows(names, rows)
        if row_names.size:
            return row_names
        return np.asarray([f"{prefix}_{int(idx)}" for idx in np.asarray(idx_values, dtype=np.int64)], dtype=object)

    @staticmethod
    def _row_order_by_idx(table: np.ndarray, rows: np.ndarray, idx_col: int) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return rows
        order = np.argsort(table[rows.astype(np.intp, copy=False), int(idx_col)].astype(np.int64, copy=False), kind="stable")
        return rows[order].astype(np.int64, copy=False)

    def _build_array_device_context(self) -> None:
        """Build DC SE runtime arrays directly from PPC/topology without device objects."""
        ppc = self._dc_ppc_dict()
        topology_arrays = ppc.get("_topology_arrays", getattr(self.network, "topology", None))
        if topology_arrays is None:
            ensure_dc_ppc_topology(ppc)
            topology_arrays = ppc["_topology_arrays"]
        self._dc_ppc = ppc
        self._dc_topology_arrays = topology_arrays
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        bus_names = np.asarray(ppc.get("bus_name", ()), dtype=object)
        active_bus_pos = np.flatnonzero(np.asarray(topology_arrays.bus_alive_mask, dtype=bool)).astype(np.int64, copy=False)
        if active_bus_pos.size == 0:
            raise RuntimeError("No alive DC nodes are available for state estimation")
        bus_solver_pos = np.full(int(topology_arrays.bus_ids.size), -1, dtype=np.int64)
        bus_solver_pos[active_bus_pos.astype(np.intp, copy=False)] = np.arange(active_bus_pos.size, dtype=np.int64)
        self._bus_solver_pos = bus_solver_pos
        self._topology_bus_pos_by_solver_pos = active_bus_pos
        first_node_offsets = topology_arrays.bus_node_offsets[active_bus_pos.astype(np.intp, copy=False)]
        first_node_rows = topology_arrays.bus_node_indices[first_node_offsets.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self.n_nodes = int(active_bus_pos.size)
        self._node_idx_by_pos = topology_arrays.bus_ids[active_bus_pos.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._node_name_by_pos = self._ppc_names_for_rows_or_idx(bus_names, first_node_rows, "bus", self._node_idx_by_pos)
        self._node_vbase_by_pos = bus[first_node_rows.astype(np.intp, copy=False), DC_BUS_COLS["vbase"]].astype(np.float64, copy=True)
        self._node_voltage_by_pos = bus[first_node_rows.astype(np.intp, copy=False), DC_BUS_COLS["voltage"]].astype(np.float64, copy=True)
        self._node_voltage_file_base_by_pos = self.u_scale * self._node_vbase_by_pos
        abs_vbase = np.abs(self._node_vbase_by_pos)
        current_base = np.ones(abs_vbase.size, dtype=np.float64)
        positive_vbase = abs_vbase > 1e-12
        if np.any(positive_vbase):
            current_base[positive_vbase] = self.p_base_kW / (1000.0 * abs_vbase[positive_vbase])
        self._node_current_file_base_by_pos = self.i_scale * current_base

        node_rows = np.arange(int(topology_arrays.node_ids.size), dtype=np.int64)
        raw_bus_pos = topology_arrays.node_to_bus_pos.astype(np.int64, copy=False)
        raw_solver_pos = np.full(raw_bus_pos.size, -1, dtype=np.int64)
        valid_raw_bus = (raw_bus_pos >= 0) & (raw_bus_pos < bus_solver_pos.size)
        raw_solver_pos[valid_raw_bus] = bus_solver_pos[raw_bus_pos[valid_raw_bus].astype(np.intp, copy=False)]
        alive_node_rows = node_rows[raw_solver_pos >= 0]
        raw_node_names = self._ppc_names_for_rows_or_idx(
            bus_names,
            alive_node_rows,
            "bus",
            topology_arrays.node_ids[alive_node_rows.astype(np.intp, copy=False)],
        )
        self._raw_node_rows_alive = alive_node_rows
        self._raw_node_names_alive = raw_node_names
        self._raw_node_idx_alive = topology_arrays.node_ids[alive_node_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._raw_node_solver_pos_alive = raw_solver_pos[alive_node_rows.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._dc_node_device_pos_by_solver_pos = np.full(self.n_nodes, -1, dtype=np.int64)
        valid_node_pos = (self._raw_node_solver_pos_alive >= 0) & (self._raw_node_solver_pos_alive < self.n_nodes)
        if np.any(valid_node_pos):
            self._dc_node_device_pos_by_solver_pos[
                self._raw_node_solver_pos_alive[valid_node_pos].astype(np.intp, copy=False)
            ] = np.flatnonzero(valid_node_pos).astype(np.int64, copy=False)
        node_lookup_order = np.argsort(self._raw_node_idx_alive, kind="stable")
        self._node_idx_lookup_ids = self._raw_node_idx_alive[node_lookup_order].astype(np.int64, copy=False)
        self._node_idx_lookup_pos = self._raw_node_solver_pos_alive[node_lookup_order].astype(np.int64, copy=False)

        def terminal_context(table_key: str, name_key: str, cols: Dict[str, int], prefix: str):
            table = np.asarray(ppc.get(table_key, np.zeros((0, len(cols)))), dtype=np.float64)
            topo = topology_arrays.devices.get(table_key)
            if table.size == 0 or topo is None:
                empty_i = np.asarray([], dtype=np.int64)
                empty_f = np.asarray([], dtype=np.float64)
                empty_o = np.asarray([], dtype=object)
                return empty_i, empty_o, empty_i, empty_i, empty_i, empty_i, empty_f
            alive_rows = np.flatnonzero(np.asarray(topo.alive_mask, dtype=bool)).astype(np.int64, copy=False)
            rows = self._row_order_by_idx(table, alive_rows, cols["idx"])
            if rows.size == 0:
                empty_i = np.asarray([], dtype=np.int64)
                empty_f = np.asarray([], dtype=np.float64)
                empty_o = np.asarray([], dtype=object)
                return empty_i, empty_o, empty_i, empty_i, empty_i, empty_i, empty_f
            row_pos = rows.astype(np.intp, copy=False)
            names = self._ppc_names_for_rows_or_idx(ppc.get(name_key), rows, prefix, table[row_pos, cols["idx"]])
            i_node = table[row_pos, cols["i_node"]].astype(np.int64, copy=False)
            j_node = table[row_pos, cols["j_node"]].astype(np.int64, copy=False)
            i_bus = topo.i_bus_pos[row_pos].astype(np.int64, copy=False)
            j_bus = topo.j_bus_pos[row_pos].astype(np.int64, copy=False)
            i_pos = bus_solver_pos[i_bus.astype(np.intp, copy=False)]
            j_pos = bus_solver_pos[j_bus.astype(np.intp, copy=False)]
            value = table[row_pos, cols["r"]].astype(np.float64, copy=False) if "r" in cols else np.zeros(rows.size)
            return rows, names, i_node, j_node, i_pos, j_pos, value

        (
            self._branch_rows,
            self._branch_names,
            self._branch_i_node,
            self._branch_j_node,
            self._branch_i_pos,
            self._branch_j_pos,
            self._branch_r,
        ) = terminal_context("branch", "branch_name", DC_BRANCH_COLS, "branch")
        (
            self._zero_branch_rows,
            self._zero_branch_names,
            self._zero_branch_i_node,
            self._zero_branch_j_node,
            self._zero_branch_i_pos,
            self._zero_branch_j_pos,
            _zero_unused,
        ) = terminal_context("zero_branch", "zero_branch_name", DC_ZERO_BRANCH_COLS, "zero_branch")
        (
            self._break_rows,
            self._break_names,
            self._break_i_node,
            self._break_j_node,
            self._break_i_pos,
            self._break_j_pos,
            _break_unused,
        ) = terminal_context("break", "break_name", DC_BREAK_COLS, "break")

        def single_context(table_key: str, name_key: str, cols: Dict[str, int], prefix: str):
            table = np.asarray(ppc.get(table_key, np.zeros((0, len(cols)))), dtype=np.float64)
            topo = topology_arrays.devices.get(table_key)
            if table.size == 0 or topo is None:
                empty_i = np.asarray([], dtype=np.int64)
                empty_o = np.asarray([], dtype=object)
                return table, empty_i, empty_o, empty_i, empty_i
            rows = self._row_order_by_idx(
                table,
                np.flatnonzero(np.asarray(topo.alive_mask, dtype=bool)).astype(np.int64, copy=False),
                cols["idx"],
            )
            if rows.size == 0:
                empty_i = np.asarray([], dtype=np.int64)
                empty_o = np.asarray([], dtype=object)
                return table, empty_i, empty_o, empty_i, empty_i
            row_pos = rows.astype(np.intp, copy=False)
            names = self._ppc_names_for_rows_or_idx(ppc.get(name_key), rows, prefix, table[row_pos, cols["idx"]])
            node_idx = table[row_pos, cols["node"]].astype(np.int64, copy=False)
            bus_pos = topo.bus_pos[row_pos].astype(np.int64, copy=False)
            solver_pos = bus_solver_pos[bus_pos.astype(np.intp, copy=False)]
            return table, rows, names, node_idx, solver_pos

        self._load_table, self._load_rows, self._load_names, self._load_node, self._load_pos = single_context(
            "load", "load_name", DC_LOAD_COLS, "load"
        )
        self._gen_table, self._generator_rows, self._generator_names, self._generator_node, self._generator_pos = single_context(
            "gen", "gen_name", DC_GEN_COLS, "gen"
        )

        dcdc = np.asarray(ppc.get("dcdc", np.zeros((0, len(DC_DCDC_COLS)))), dtype=np.float64)
        dcdc_topo = topology_arrays.devices.get("dcdc")
        if dcdc.size and dcdc_topo is not None:
            self._dcdc_rows = self._row_order_by_idx(
                dcdc,
                np.flatnonzero(np.asarray(dcdc_topo.alive_mask, dtype=bool)).astype(np.int64, copy=False),
                DC_DCDC_COLS["idx"],
            )
            row_pos = self._dcdc_rows.astype(np.intp, copy=False)
            self._dcdc_names = self._ppc_names_for_rows_or_idx(ppc.get("dcdc_name"), self._dcdc_rows, "dcdc", dcdc[row_pos, DC_DCDC_COLS["idx"]])
            self._dcdc_i_node = dcdc[row_pos, DC_DCDC_COLS["i_node"]].astype(np.int64, copy=False)
            self._dcdc_j_node = dcdc[row_pos, DC_DCDC_COLS["j_node"]].astype(np.int64, copy=False)
            self._dcdc_i_pos = bus_solver_pos[dcdc_topo.i_bus_pos[row_pos].astype(np.intp, copy=False)]
            self._dcdc_j_pos = bus_solver_pos[dcdc_topo.j_bus_pos[row_pos].astype(np.intp, copy=False)]
        else:
            self._dcdc_rows = np.asarray([], dtype=np.int64)
            self._dcdc_names = np.asarray([], dtype=object)
            self._dcdc_i_node = np.asarray([], dtype=np.int64)
            self._dcdc_j_node = np.asarray([], dtype=np.int64)
            self._dcdc_i_pos = np.asarray([], dtype=np.int64)
            self._dcdc_j_pos = np.asarray([], dtype=np.int64)

    @staticmethod
    def _clear_meas_ppc_runtime_arrays(meas_ppc: Dict) -> None:
        """Drop estimator-local measurement runtime arrays copied from a shared file cache."""
        for key in (
            "device_pos",
            "scale",
            "from_pos",
            "to_pos",
            "available",
            "_dc_se_runtime_cache_key",
            "_mutable_runtime_arrays",
        ):
            meas_ppc.pop(key, None)

    def prepare(
        self,
        *,
        network: Optional[object] = None,
        measurements: Optional[object] = None,
        prepare_active_measurements: Optional[bool] = None,
        defer_prepare_finalize: Optional[bool] = None,
    ) -> "DCStateEstimator":
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
        if network is None:
            self.network = self._load_network(self.e_file)
        else:
            self.network = network
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            self.meas_ppc = build_meas_ppc_from_e_file(
                self.meas_file,
                include_strings=False,
                include_matrix=False,
            )
            self.meas_ppc["_mutable_runtime_arrays"] = True
            self.measurements = self._measurement_sequence_from_table(
                measurement_table_from_meas_ppc(self.meas_ppc, include_strings=False),
                normalized=bool(self.meas_ppc.get("normalized", False)),
            )
        elif isinstance(measurements, dict) and measurements.get("format") == "meas_ppc_v1":
            self.meas_ppc = measurements if measurements.get("_mutable_runtime_arrays") else copy_meas_ppc(measurements)
            self.meas_ppc["_mutable_runtime_arrays"] = True
            self.measurements = self._measurement_sequence_from_table(
                measurement_table_from_meas_ppc(self.meas_ppc, include_strings=False),
                normalized=bool(self.meas_ppc.get("normalized", False)),
            )
        elif isinstance(measurements, MeasurementTable):
            self.meas_ppc = None
            self.measurements = self._measurement_sequence_from_table(measurements, normalized=False)
        else:
            table = getattr(measurements, "table", None)
            if isinstance(table, MeasurementTable):
                self.meas_ppc = None
                self.measurements = self._measurement_sequence_from_table(
                    table,
                    normalized=getattr(measurements, "normalized", False),
                )
            else:
                raise TypeError(
                    "DC SE measurements must be a meas PPC dict or MeasurementTable; "
                    "Measurement object parsing is disabled."
                )
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)
        stage_start = time.perf_counter()
        self._build_array_device_context()
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._attach_meas_ppc_device_pos()
        self._record_profile_time("init.measurement_device_pos", time.perf_counter() - stage_start)
        self._defer_prepare_finalize_pending = bool(defer_prepare_finalize)
        if defer_prepare_finalize:
            self.power_flow_seed_converged = False
            self.targeted_observability_pseudo_count = 0
            return self
        # Reference selection must see only active, unit-normalized real measurements.
        self.finalize_prepare(prepare_active_measurements=prepare_active_measurements)
        self._prepared = True
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

    def _require_prepared(self, action: str) -> None:
        if not self._prepared:
            raise RuntimeError(f"Call prepare() before {action}.")

    def _build_state_meta_arrays(self) -> Dict[str, np.ndarray]:
        n_state = int(getattr(self, "n_state", 0))
        side = np.full(n_state, "dc", dtype=object)
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

        voltage_state_pos = np.asarray(getattr(self, "voltage_state_pos", ()), dtype=np.int64)
        count = int(voltage_state_pos.size)
        if count:
            rows = slice(cursor, cursor + count)
            solver_pos = voltage_state_pos.astype(np.intp, copy=False)
            names = np.asarray(self._node_name_by_pos, dtype=object)[solver_pos]
            kind[rows] = "voltage"
            device_type[rows] = "DCNode"
            device_name[rows] = names
            component[rows] = "magnitude"
            legacy_label[rows] = np.char.add("V:", names.astype(str)).astype(object)
            local_device_pos = np.full(count, -1, dtype=np.int64)
            valid = voltage_state_pos < self._dc_node_device_pos_by_solver_pos.size
            if np.any(valid):
                local_device_pos[valid] = self._dc_node_device_pos_by_solver_pos[voltage_state_pos[valid].astype(np.intp, copy=False)]
            device_pos[rows] = local_device_pos
            device_type_code[rows] = DEVICE_TYPE_DCNode
            meas_type_code[rows] = MEAS_TYPE_V
            cursor += count

        zero_names = np.asarray(getattr(self, "_zero_branch_names", ()), dtype=object)
        count = int(zero_names.size)
        if count:
            rows = slice(cursor, cursor + count)
            kind[rows] = "zero_current"
            device_type[rows] = "DCZeroBranch"
            device_name[rows] = zero_names
            component[rows] = "current"
            legacy_label[rows] = np.char.add("I_ZERO:", zero_names.astype(str)).astype(object)
            device_pos[rows] = np.arange(count, dtype=np.int64)
            device_type_code[rows] = DEVICE_TYPE_DCZeroBranch
            meas_type_code[rows] = MEAS_TYPE_I_FROM
            cursor += count

        break_names = np.asarray(getattr(self, "_break_names", ()), dtype=object)
        count = int(break_names.size)
        if count:
            rows = slice(cursor, cursor + count)
            kind[rows] = "break_current"
            device_type[rows] = "DCBreak"
            device_name[rows] = break_names
            component[rows] = "current"
            legacy_label[rows] = np.char.add("I_BREAK:", break_names.astype(str)).astype(object)
            device_pos[rows] = np.arange(count, dtype=np.int64)
            device_type_code[rows] = DEVICE_TYPE_DCBreak
            meas_type_code[rows] = MEAS_TYPE_I_FROM
            cursor += count

        dcdc_names = np.asarray(getattr(self, "_dcdc_names", ()), dtype=object)
        count = int(dcdc_names.size)
        if count:
            rows = slice(cursor, cursor + 2 * count)
            repeated_names = np.repeat(dcdc_names, 2)
            kind[rows] = np.tile(np.asarray(["dcdc_p_from", "dcdc_p_to"], dtype=object), count)
            device_type[rows] = "DCDCConverter"
            device_name[rows] = repeated_names
            terminal[rows] = np.tile(np.asarray(["from", "to"], dtype=object), count)
            component[rows] = "p"
            label_prefix = np.tile(np.asarray(["P_DCDC_FROM:", "P_DCDC_TO:"], dtype=object), count)
            legacy_label[rows] = np.char.add(label_prefix.astype(str), repeated_names.astype(str)).astype(object)
            device_pos[rows] = np.repeat(np.arange(count, dtype=np.int64), 2)
            device_type_code[rows] = DEVICE_TYPE_DCDCConverter
            meas_type_code[rows] = np.tile(np.asarray([MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO], dtype=np.int16), count)
            cursor += 2 * count

        vgen_names = np.asarray(getattr(self, "_v_generator_names", ()), dtype=object)
        vgen_device_pos = np.asarray(getattr(self, "_v_generator_local_pos", ()), dtype=np.int64)
        count = int(vgen_names.size)
        if count:
            rows = slice(cursor, cursor + count)
            kind[rows] = "v_generator_p"
            device_type[rows] = "DCGenerator"
            device_name[rows] = vgen_names
            component[rows] = "p"
            legacy_label[rows] = np.char.add("P_VGEN:", vgen_names.astype(str)).astype(object)
            device_pos[rows] = vgen_device_pos
            device_type_code[rows] = DEVICE_TYPE_DCGenerator
            meas_type_code[rows] = MEAS_TYPE_P_GEN
            cursor += count

        if cursor != n_state:
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

    @property
    def zero_tie_components(self) -> Tuple[np.ndarray, ...]:
        cache = getattr(self, "_zero_tie_components_cache", None)
        if cache is None:
            offsets = np.asarray(getattr(self, "zero_tie_component_offsets", ()), dtype=np.int64)
            indices = np.asarray(getattr(self, "zero_tie_component_indices", ()), dtype=np.int32)
            if offsets.size <= 1:
                cache = ()
            else:
                cache = tuple(
                    indices[int(offsets[idx]) : int(offsets[idx + 1])]
                    for idx in range(int(offsets.size) - 1)
                )
            self._zero_tie_components_cache = cache
        return cache

    @zero_tie_components.setter
    def zero_tie_components(self, value) -> None:
        self._zero_tie_components_cache = value

    @property
    def voltage_state_nodes(self) -> Tuple[np.ndarray, ...]:
        cache = getattr(self, "_voltage_state_nodes_cache", None)
        if cache is None:
            offsets = np.asarray(getattr(self, "voltage_state_node_offsets", ()), dtype=np.int64)
            indices = np.asarray(getattr(self, "voltage_state_node_indices", ()), dtype=np.int32)
            if offsets.size <= 1:
                cache = ()
            else:
                cache = tuple(
                    indices[int(offsets[idx]) : int(offsets[idx + 1])]
                    for idx in range(int(offsets.size) - 1)
                )
            self._voltage_state_nodes_cache = cache
        return cache

    @voltage_state_nodes.setter
    def voltage_state_nodes(self, value) -> None:
        self._voltage_state_nodes_cache = value

    def finalize_prepare(
        self,
        *,
        prepare_active_measurements: bool = True,
        measurements_already_normalized: bool = False,
    ) -> "DCStateEstimator":
        self._defer_prepare_finalize_pending = False
        self._disable_unavailable_measurements()
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
            if self.power_flow_linear_solver:
                setattr(self.network, "_se_power_flow_linear_solver", self.power_flow_linear_solver)
            self.power_flow_seed_converged = bool(self._run_power_flow_seed(self.network, self.params, self.e_file))
            self._record_profile_time("seed.lf", time.perf_counter() - stage_start)
            self._record_profile_time("seed.total", time.perf_counter() - seed_start)
        stage_start = time.perf_counter()
        self.node_voltage_measurements = self._node_voltage_measurements()
        self.node_degrees = self._node_incident_degrees()
        self.references = self._select_reference_nodes()
        self._build_zero_tie_voltage_layout()
        gen_ctrl = self._gen_table[self._generator_rows.astype(np.intp, copy=False), DC_GEN_COLS["control_type"]].astype(np.int64, copy=False)
        self._v_generator_local_pos = np.flatnonzero(gen_ctrl == DC_CTRL_V).astype(np.int64, copy=False)
        self._v_generator_rows = self._generator_rows[self._v_generator_local_pos.astype(np.intp, copy=False)]
        self._v_generator_names = self._generator_names[self._v_generator_local_pos.astype(np.intp, copy=False)]
        self._generator_v_state_pos = np.full(self._generator_rows.size, -1, dtype=np.int64)
        if self._v_generator_local_pos.size:
            self._generator_v_state_pos[self._v_generator_local_pos.astype(np.intp, copy=False)] = np.arange(
                self._v_generator_local_pos.size,
                dtype=np.int64,
            )

        self.n_switch = int(self._zero_branch_names.size + self._break_names.size)
        self.n_dcdc_power = 2 * int(self._dcdc_names.size)
        self.n_v_generator = int(self._v_generator_names.size)
        self.switch_start = self.n_voltage
        self.dcdc_start = self.switch_start + self.n_switch
        self.v_generator_start = self.dcdc_start + self.n_dcdc_power
        self.n_state = self.v_generator_start + self.n_v_generator
        self._state_meta_arrays = None
        self._state_meta_cache = None
        self._state_labels_cache = None
        self._build_apply_state_index()
        self._build_initial_state_seed_arrays()
        self._build_measurement_plan_device_cache()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        self.targeted_observability_pseudo_count = 0
        self._observability_matrix_cache = None
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        if prepare_active_measurements:
            stage_start = time.perf_counter()
            self._add_pseudo_power_measurements()
            self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._refresh_active_measurement_indexes()
            self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
            self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        else:
            empty_table = MeasurementTable(
                idx=np.asarray([], dtype=np.int64),
                name=np.asarray([], dtype=object),
                device_type=np.asarray([], dtype=object),
                device_name=np.asarray([], dtype=object),
                meas_type=np.asarray([], dtype=object),
                weight=np.asarray([], dtype=np.float64),
                valid=np.asarray([], dtype=bool),
                value=np.asarray([], dtype=np.float64),
                device_type_code=np.asarray([], dtype=np.int16),
                angle_mask=np.asarray([], dtype=bool),
                status_code=np.asarray([], dtype=np.int16),
                rows_by_device_type_code={},
                device_name_id=np.asarray([], dtype=np.int64),
                meas_type_code=np.asarray([], dtype=np.int16),
                device_pos=np.asarray([], dtype=np.int64),
            )
            self.active_measurements = self._measurement_sequence_from_table(
                empty_table,
                normalized=getattr(self.measurements, "normalized", False),
            )
            self.active_measurement_table = empty_table
            self.active_measurement_rows = np.array([], dtype=np.int64)
            self.active_z = np.array([], dtype=np.float64)
            self.active_weight = np.array([], dtype=np.float64)
            self.active_uniform_weight = None
            self.active_weights_are_uniform = False
            self._active_normal_pattern = None
            self._active_normal_assembly_plan = None
            self._active_lower_normal_plan = None
            self._jacobian_builder = SparseJacobianBuilder((0, self.n_state))
            self._jacobian_builder._assume_fixed_pattern = True
            self._measurement_plan_cache = {}
            self._active_measurement_plan_tables_cache = self._active_measurement_plan_tables(empty_table)
            self._active_measurement_plan = self._measurement_plan(self._active_measurement_plan_tables_cache)
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
        message = f"DC SE requires {name} during {context}; object-network/no-PPC fallback is disabled."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise RuntimeError(message)

    def _refresh_active_measurement_indexes(self) -> None:
        """Rebuild active measurement arrays and the vectorized measurement plan."""
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_array_view,
            materialize_measurements=False,
        )
        self.measurement_table = active_view.source_table
        self._max_measurement_idx = int(self.measurement_table.idx.max()) if self.measurement_table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_view.table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._measurement_plan_cache = {}
        self._active_measurement_plan_tables_cache = self._active_measurement_plan_tables(
            self.active_measurement_table
        )
        self._active_measurement_plan = self._measurement_plan(self._active_measurement_plan_tables_cache)
        handled_mask = self._active_measurement_plan.get("handled_mask")
        self.active_measurements_are_vectorized = bool(np.all(handled_mask)) if handled_mask is not None else False

    def _incremental_update_active_measurement_indexes(
        self,
        appended_measurements: object,
        *,
        source_row_start: Optional[int] = None,
        master_table: Optional[MeasurementTable] = None,
    ) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            warnings.warn(
                "DC SE incremental active update requires active_measurements; full rebuild is required.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        appended_table = _measurement_table_from_array_view(appended_measurements)
        if np.any(
            (~np.asarray(appended_table.valid, dtype=bool))
            | (np.asarray(appended_table.weight, dtype=np.float64) <= 0.0)
        ):
            return False
        appended_device_pos = getattr(appended_table, "device_pos", None)
        if appended_device_pos is None or np.asarray(appended_device_pos).size != int(appended_table.idx.size):
            warnings.warn(
                "DC SE incremental active update requires appended measurement device_pos; "
                "name-based index rebuild is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        appended_device_pos = np.asarray(appended_device_pos, dtype=np.int64)
        if np.any(appended_device_pos < 0):
            warnings.warn(
                "DC SE incremental active update found appended rows without valid device_pos; "
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
                table_builder=_measurement_table_from_array_view,
                materialize_measurements=False,
            ),
            appended_view,
            source_row_start=source_row_start,
            table_builder=_measurement_table_from_array_view,
            materialize_measurements=False,
        )
        self._initial_observability_cache = None
        self._max_measurement_idx = int(self.measurement_table.idx.max()) if self.measurement_table.idx.size else 0
        self.active_measurements = active_view.measurements
        self.active_measurement_rows = active_view.source_rows
        self.active_measurement_table = active_view.table
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_rows_by_device_type_code = active_view.rows_by_device_type_code
        self._measurement_plan_cache = {}
        self._active_measurement_plan_tables_cache = self._active_measurement_plan_tables(
            self.active_measurement_table
        )
        self._active_measurement_plan = self._measurement_plan(self._active_measurement_plan_tables_cache)
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._active_normal_assembly_plan = None
        self._active_lower_normal_plan = None
        self._observability_matrix_cache = None
        self._weak_direction_candidate_jacobian_cache = None
        handled_mask = self._active_measurement_plan.get("handled_mask")
        self.active_measurements_are_vectorized = bool(np.all(handled_mask)) if handled_mask is not None else False
        return True

    @staticmethod
    def _merge_active_plan_dict(
        head: Dict[str, object],
        tail: Dict[str, object],
        *,
        row_offset: int,
        row_keys: Sequence[str],
        mapped_row_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        row_key_tuple = tuple(row_keys)
        merged: Dict[str, object] = {}
        for key, head_value in head.items():
            if key.endswith("_masks"):
                continue
            tail_value = tail[key]
            if key in row_key_tuple:
                merged[key] = np.concatenate(
                    (
                        np.asarray(head_value, dtype=np.int64),
                        np.asarray(tail_value, dtype=np.int64) + int(row_offset),
                    )
                ).astype(np.int64, copy=False)
                continue
            merged[key] = np.concatenate((np.asarray(head_value), np.asarray(tail_value)))
        DCStateEstimator._populate_kind_masks(merged)
        return merged

    @staticmethod
    def _populate_kind_masks(plan: Dict[str, np.ndarray]) -> None:
        """Fill in per-kind bool masks that the fill_* hot loops would otherwise
        recompute every iteration. Tuples of length max_kind+1; entry k is the
        ``section_kind == k`` mask. Called after every plan build / merge /
        shrink so the masks always reflect the current kind arrays."""
        for key, max_kind in (
            ("branch_kind", MEAS_TYPE_I_TO),
            ("load_kind", MEAS_TYPE_I_LOAD),
            ("gen_kind", MEAS_TYPE_I_GEN),
            ("gen_ctrl", 2),
            ("switch_kind", MEAS_TYPE_I_TO),
            ("dcdc_kind", MEAS_TYPE_I_TO),
        ):
            kind = plan.get(key)
            if kind is None:
                continue
            mask_key = key.rsplit("_", 1)[0] + ("_ctrl_masks" if key.endswith("_ctrl") else "_kind_masks")
            plan[mask_key] = tuple((kind == k) for k in range(max_kind + 1))

    def _normalize_measurements(self, measurements: Optional[object]):
        if measurements is None:
            return self.active_measurements
        if isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable):
            return measurements
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return measurements
        raise TypeError(
            "DC SE measurement runtime requires MeasurementPlanTable or MeasurementTableView; "
            "Measurement object iteration is disabled."
        )

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

    def _measurement_plan_tables_are_active(self, plan_tables) -> bool:
        return plan_tables is getattr(self, "_active_measurement_plan_tables_cache", None)

    @staticmethod
    def _common_measurement_plan_table(plan_tables) -> MeasurementPlanTable:
        if isinstance(plan_tables, MeasurementPlanTable):
            return plan_tables
        if not isinstance(plan_tables, dict) or not plan_tables:
            raise TypeError("DC SE measurement runtime requires MeasurementPlanTable objects")
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

    def install_measurement_runtime(self, measurements) -> Dict[str, MeasurementPlanTable]:
        """Install a delegated DC measurement block as the active array runtime."""
        plan_tables = self._measurement_plan_tables_for(measurements)
        common_table = self._common_measurement_plan_table(plan_tables).table
        self._jacobian_builder = SparseJacobianBuilder((len(measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_measurement_plan_tables_cache = plan_tables
        self.active_measurements = measurements
        self.active_measurement_table = common_table
        self.active_z = np.asarray(common_table.value, dtype=np.float64)
        self.active_weight = np.asarray(common_table.weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(common_table.angle_mask, dtype=bool)
        self._allow_nonflat_fast_observability_certificate = True
        self._active_measurement_plan = None
        vector_plans = self._vector_plans_for_measurement_plan_tables(plan_tables)
        handled = np.zeros(len(measurements), dtype=bool)
        handled_valid = True
        for plan in vector_plans.values():
            plan_handled = np.asarray(plan.get("handled_mask", ()), dtype=bool)
            if plan_handled.size != handled.size:
                handled_valid = False
                break
            handled |= plan_handled
        self.active_measurements_are_vectorized = bool(handled_valid and np.all(handled))
        return plan_tables

    def measurement_device_counts(self) -> Dict[int, int]:
        """Return DC measurement-device counts in DC plan order."""
        def size(name: str) -> int:
            return int(np.asarray(getattr(self, name, ())).size)

        zero_count = size("_zero_branch_names")
        break_count = size("_break_names")
        constraint_count = zero_count + break_count
        return {
            DEVICE_TYPE_DCNode: size("_raw_node_names_alive"),
            DEVICE_TYPE_DCBranch: size("_branch_names"),
            DEVICE_TYPE_DCBreak: break_count,
            DEVICE_TYPE_DCZeroBranch: zero_count,
            DEVICE_TYPE_DCGenerator: size("_generator_names"),
            DEVICE_TYPE_DCLoad: size("_load_names"),
            DEVICE_TYPE_DCDCConverter: size("_dcdc_names"),
            DEVICE_TYPE_DCZeroBranchConstraint: constraint_count,
            DEVICE_TYPE_DCBreakConstraint: constraint_count,
        }

    def measurement_device_names(self, requested_codes=None) -> Dict[int, np.ndarray]:
        """Return DC measurement-device names in the same plan order as device_pos."""
        requested = None if requested_codes is None else {int(code) for code in requested_codes}
        constraint_names = np.concatenate(
            (
                np.asarray(getattr(self, "_zero_branch_names", ()), dtype=object),
                np.asarray(getattr(self, "_break_names", ()), dtype=object),
            )
        )
        names = {
            DEVICE_TYPE_DCNode: np.asarray(getattr(self, "_raw_node_names_alive", ()), dtype=object),
            DEVICE_TYPE_DCBranch: np.asarray(getattr(self, "_branch_names", ()), dtype=object),
            DEVICE_TYPE_DCBreak: np.asarray(getattr(self, "_break_names", ()), dtype=object),
            DEVICE_TYPE_DCZeroBranch: np.asarray(getattr(self, "_zero_branch_names", ()), dtype=object),
            DEVICE_TYPE_DCGenerator: np.asarray(getattr(self, "_generator_names", ()), dtype=object),
            DEVICE_TYPE_DCLoad: np.asarray(getattr(self, "_load_names", ()), dtype=object),
            DEVICE_TYPE_DCDCConverter: np.asarray(getattr(self, "_dcdc_names", ()), dtype=object),
            DEVICE_TYPE_DCZeroBranchConstraint: constraint_names,
            DEVICE_TYPE_DCBreakConstraint: constraint_names,
        }
        if requested is None:
            return names
        return {int(code): value for code, value in names.items() if int(code) in requested}

    def _vector_plans_for_measurement_plan_tables(self, plan_tables) -> Dict[str, Dict[str, np.ndarray]]:
        if self._measurement_plan_tables_are_active(plan_tables):
            plan = getattr(self, "_active_measurement_plan", None)
            if plan is None:
                plan = self._measurement_plan(plan_tables)
                self._active_measurement_plan = plan
            return {"simple": plan}
        return {"simple": self._measurement_plan(plan_tables)}

    def _node_pos_from_idx(self, node_idx: int) -> int:
        lookup_ids = getattr(self, "_node_idx_lookup_ids", None)
        lookup_pos = getattr(self, "_node_idx_lookup_pos", None)
        if lookup_ids is None or lookup_pos is None:
            return -1
        ids = np.asarray(lookup_ids, dtype=np.int64)
        if ids.size == 0:
            return -1
        pos = int(np.searchsorted(ids, int(node_idx)))
        if pos < ids.size and int(ids[pos]) == int(node_idx):
            return int(np.asarray(lookup_pos, dtype=np.int64)[pos])
        return -1

    def _measurement_vectors(self, measurements) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is None:
            table = self._active_measurement_table_ref()
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        if isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable):
            table = self._common_measurement_plan_table(measurements).table
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
        table = getattr(measurements, "table", None)
        if table is not None and len(table.value) == len(measurements):
            return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)
        raise TypeError(
            "DC SE measurement vectors require table-backed arrays; Measurement object iteration is disabled."
        )

    @staticmethod
    def _uniform_weight(weight: np.ndarray) -> Optional[float]:
        if weight.size == 0:
            return None
        first_weight = float(weight[0])
        return first_weight if bool(np.all(weight == first_weight)) else None

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: object,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if out is None:
            return np.subtract(z, z_est, dtype=np.float64)
        np.subtract(z, z_est, out=out)
        return out

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _warn_missing_required_active_measurements(self, context: str) -> None:
        warnings.warn(
            f"DC SE requires active_measurements during {context}; temporary rebuild fallback is disabled.",
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
            raise RuntimeError("DC SE requires active_measurement_table for active array execution")
        return table

    def _active_measurement_count(self) -> int:
        return int(self._active_measurement_table_ref().idx.size)

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
        measurements: object,
        H,
        *,
        cache_lower_normal_plan: bool = True,
    ) -> None:
        lower_normal_plan = None
        if (
            cache_lower_normal_plan
            and self._measurement_plan_tables_are_active(measurements)
            and is_sparse_matrix(H)
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
        measurements: object,
        x: np.ndarray,
    ):
        cache = getattr(self, "_observability_matrix_cache", None)
        if cache is None or cache.get("result") is not result:
            return None
        if cache.get("measurements") is not measurements:
            return None
        cached_x = cache.get("x")
        if cached_x is None or cached_x.shape != np.asarray(x).shape or not np.array_equal(cached_x, x):
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

    def _disable_unavailable_measurements(self) -> None:
        """Keep invalid/off-topology measurement rows out of unit conversion and WLS."""
        table = getattr(self.measurements, "table", None)
        table_valid = table.valid if table is not None and len(table.valid) == len(self.measurements) else None
        table_status = measurement_table_status_code(table) if table_valid is not None else None
        if table_valid is not None:
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            precomputed_device_pos = getattr(table, "device_pos", None)
            if precomputed_device_pos is not None and np.asarray(precomputed_device_pos).size == len(table_valid):
                device_pos = np.asarray(precomputed_device_pos, dtype=np.int64)
            else:
                device_pos = self._measurement_device_pos_array(table)
            active = table_valid & (np.asarray(table.weight, dtype=np.float64) > 0.0)
            supported = np.isin(
                device_type_code,
                np.asarray(
                    [
                        DEVICE_TYPE_DCNode,
                        DEVICE_TYPE_DCBranch,
                        DEVICE_TYPE_DCBreak,
                        DEVICE_TYPE_DCZeroBranch,
                        DEVICE_TYPE_DCGenerator,
                        DEVICE_TYPE_DCLoad,
                        DEVICE_TYPE_DCDCConverter,
                    ],
                    dtype=np.int16,
                ),
            )
            invalid_rows = np.flatnonzero(active & ((device_pos < 0) | ~supported))
            if invalid_rows.size:
                table_valid[invalid_rows] = False
                if table_status is not None:
                    table_status[invalid_rows] = MEAS_STATUS_INVALID
            return
        warnings.warn(
            "DC SE measurement availability requires a MeasurementTable with device_pos; object fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real DCNode voltage measurements keyed by DC node index."""
        cached = getattr(self, "_node_voltage_measurement_cache", None)
        if cached is not None:
            return cached
        self._refresh_measurement_summary_cache()
        return getattr(self, "_node_voltage_measurement_cache", {})

    def _voltage_measurement_node_idx_from_pos(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> Optional[int]:
        """Return the DC node associated with a voltage measurement row."""
        pos = int(device_pos)
        if pos < 0:
            return None
        code = int(device_type_code)
        meas_code = int(meas_type_code)
        if code == DEVICE_TYPE_DCNode:
            if meas_code == MEAS_TYPE_V and pos < self._raw_node_idx_alive.size:
                return int(self._raw_node_idx_alive[pos])
            return None
        if code == DEVICE_TYPE_DCGenerator:
            if meas_code == MEAS_TYPE_V_GEN and pos < self._generator_node.size:
                return int(self._generator_node[pos])
            return None
        if code == DEVICE_TYPE_DCLoad:
            if meas_code == MEAS_TYPE_V_LOAD and pos < self._load_node.size:
                return int(self._load_node[pos])
            return None
        if code == DEVICE_TYPE_DCBranch:
            i_node, j_node = self._branch_i_node, self._branch_j_node
        elif code == DEVICE_TYPE_DCZeroBranch:
            i_node, j_node = self._zero_branch_i_node, self._zero_branch_j_node
        elif code == DEVICE_TYPE_DCBreak:
            i_node, j_node = self._break_i_node, self._break_j_node
        elif code == DEVICE_TYPE_DCDCConverter:
            i_node, j_node = self._dcdc_i_node, self._dcdc_j_node
        else:
            return None
        if pos >= i_node.size:
            return None
        if meas_code == MEAS_TYPE_V_FROM:
            return int(i_node[pos])
        if meas_code == MEAS_TYPE_V_TO:
            return int(j_node[pos])
        return None

    def voltage_measurement_node_idx(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> Optional[int]:
        """Return the DC node index associated with a voltage measurement row."""
        return self._voltage_measurement_node_idx_from_pos(
            device_type_code,
            device_pos,
            meas_type_code,
        )

    def _real_voltage_observation_nodes(self) -> Dict[int, float]:
        """Return nodes covered by real usable voltage measurements on any DC device."""
        cache = getattr(self, "_real_voltage_observation_node_cache", None)
        if cache is not None:
            return cache
        best: Dict[int, Tuple[float, float]] = {}
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            meas_type_code = self._ensure_table_meas_type_codes(table)
            device_pos = self._measurement_device_pos_array(table)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            status_code = measurement_table_status_code(table)
            voltage_mask = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
                & np.isin(meas_type_code, _VOLTAGE_MEASUREMENT_TYPE_CODES)
                & (device_pos >= 0)
            )
            for row in np.flatnonzero(voltage_mask):
                node_idx = self._voltage_measurement_node_idx_from_pos(
                    int(device_type_code[row]),
                    int(device_pos[row]),
                    int(meas_type_code[row]),
                )
                if node_idx is None or self._node_pos_from_idx(node_idx) < 0:
                    continue
                weight = float(table.weight[row])
                current = best.get(int(node_idx))
                if current is None or weight > current[0]:
                    best[int(node_idx)] = (weight, float(table.value[row]))
        else:
            warnings.warn(
                "DC SE voltage observation scan requires a MeasurementTable with device_pos; object fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        cache = {node_idx: value for node_idx, (_weight, value) in best.items()}
        self._real_voltage_observation_node_cache = cache
        return cache

    def _real_voltage_observation_value_for_node(self, node_idx: Optional[int]) -> Optional[float]:
        """Return a real voltage value on the node or its compressed zero-tie component."""
        if node_idx is None:
            return None
        observed = self._real_voltage_observation_nodes()
        if node_idx in observed:
            return float(observed[node_idx])
        pos = self._node_pos_from_idx(node_idx)
        component_by_pos = getattr(self, "zero_tie_component_by_pos", None)
        component_offsets = getattr(self, "zero_tie_component_offsets", None)
        component_indices = getattr(self, "zero_tie_component_indices", None)
        if pos < 0 or component_by_pos is None or component_offsets is None or component_indices is None:
            return None
        component_idx = int(component_by_pos[int(pos)])
        if component_idx < 0 or component_idx + 1 >= int(np.asarray(component_offsets).size):
            return None
        start = int(component_offsets[component_idx])
        end = int(component_offsets[component_idx + 1])
        for member_pos in np.asarray(component_indices, dtype=np.int64)[start:end]:
            member_idx = int(self._node_idx_by_pos[int(member_pos)])
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_pseudo_is_covered_by_pos(self, device_type_code: int, device_pos: int, meas_type_code: int) -> bool:
        """Check whether a voltage pseudo row is redundant because the node already has real V data."""
        node_idx = self._voltage_measurement_node_idx_from_pos(device_type_code, device_pos, meas_type_code)
        if node_idx is None:
            return False
        pos = self._node_pos_from_idx(int(node_idx))
        if pos < 0:
            return False
        covered = self._real_voltage_observed_solver_pos_mask()
        return bool(pos < covered.size and covered[int(pos)])

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live DC topology terminals used when choosing island voltage references."""
        degrees = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.int64)
        for left, right in (
            (self._branch_i_pos, self._branch_j_pos),
            (self._zero_branch_i_pos, self._zero_branch_j_pos),
            (self._break_i_pos, self._break_j_pos),
        ):
            left = np.asarray(left, dtype=np.int64)
            right = np.asarray(right, dtype=np.int64)
            if left.size:
                valid = (left >= 0) & (left < degrees.size)
                if np.any(valid):
                    np.add.at(degrees, left[valid].astype(np.intp, copy=False), 1)
            if right.size:
                valid = (right >= 0) & (right < degrees.size)
                if np.any(valid):
                    np.add.at(degrees, right[valid].astype(np.intp, copy=False), 1)
        self._node_degree_by_pos = degrees
        return _NodeIndexedValueMap(self._node_idx_by_pos, degrees, np.ones(degrees.size, dtype=bool))

    def _select_reference_nodes(self) -> np.ndarray:
        """Choose one measured high-degree DC voltage reference per live DC topology island."""
        measured_solver_mask = np.asarray(
            getattr(self, "_node_voltage_measurement_pos_mask_cache", np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)),
            dtype=bool,
        )
        if measured_solver_mask.size != int(getattr(self, "n_nodes", 0)):
            voltage_measurements = self.node_voltage_measurements
            measured_node_idx = (
                np.fromiter(
                    (int(node_idx) for node_idx in voltage_measurements.keys()),
                    dtype=np.int64,
                    count=len(voltage_measurements),
                )
                if voltage_measurements
                else np.asarray([], dtype=np.int64)
            )
            measured_solver_mask = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
            if measured_node_idx.size:
                measured_solver_mask = np.isin(
                    np.asarray(self._node_idx_by_pos, dtype=np.int64),
                    measured_node_idx,
                    assume_unique=False,
                )
        degrees = getattr(self, "_node_degree_by_pos", np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.int64))
        topology_arrays = self._dc_topology_arrays
        bus_solver_pos = self._bus_solver_pos
        island_alive = np.asarray(topology_arrays.island_alive_mask, dtype=bool)
        references = np.full(int(np.count_nonzero(island_alive)), -1, dtype=np.int64)
        ref_count = 0
        has_measured_voltage = bool(np.any(measured_solver_mask))
        for island_pos, alive in enumerate(np.asarray(topology_arrays.island_alive_mask, dtype=bool)):
            if not bool(alive):
                continue
            start = int(topology_arrays.island_bus_offsets[island_pos])
            end = int(topology_arrays.island_bus_offsets[island_pos + 1])
            bus_pos = topology_arrays.island_bus_indices[start:end].astype(np.int64, copy=False)
            solver_pos = bus_solver_pos[bus_pos.astype(np.intp, copy=False)]
            solver_pos = solver_pos[solver_pos >= 0]
            if solver_pos.size == 0:
                continue
            if has_measured_voltage:
                measured_pos = solver_pos[measured_solver_mask[solver_pos.astype(np.intp, copy=False)]]
            else:
                measured_pos = np.asarray([], dtype=np.int64)
            if measured_pos.size:
                measured_pos = measured_pos.astype(np.intp, copy=False)
                measured_degrees = degrees[measured_pos]
                measured_candidate_node_idx = self._node_idx_by_pos[measured_pos]
                best_order = np.lexsort((measured_candidate_node_idx, -measured_degrees))
                references[ref_count] = int(measured_pos[int(best_order[0])])
                ref_count += 1
                continue
            ref_bus = int(topology_arrays.island_reference_bus_pos[island_pos])
            ref_solver = int(bus_solver_pos[ref_bus]) if 0 <= ref_bus < bus_solver_pos.size else -1
            references[ref_count] = ref_solver if ref_solver >= 0 else int(solver_pos[0])
            ref_count += 1
        return references[:ref_count].astype(np.int64, copy=False)

    def _build_zero_tie_voltage_layout(self) -> None:
        """Compress DC voltage states across explicit zero branches and breakers.

        Closed switches have already been contracted by PPC topology and do not
        appear as independent SE devices or constraints.
        """
        n = int(self.n_nodes)
        left_chunks = []
        right_chunks = []
        for left, right in (
            (self._zero_branch_i_pos, self._zero_branch_j_pos),
            (self._break_i_pos, self._break_j_pos),
        ):
            left = np.asarray(left, dtype=np.int64)
            right = np.asarray(right, dtype=np.int64)
            if left.size == 0:
                continue
            valid = (left >= 0) & (left < n) & (right >= 0) & (right < n)
            if np.any(valid):
                left_chunks.append(left[valid].astype(np.int32, copy=False))
                right_chunks.append(right[valid].astype(np.int32, copy=False))

        if left_chunks:
            graph_left = np.concatenate(left_chunks).astype(np.int32, copy=False)
            graph_right = np.concatenate(right_chunks).astype(np.int32, copy=False)
            graph = coo_matrix(
                (np.ones(graph_left.size, dtype=np.int8), (graph_left, graph_right)),
                shape=(n, n),
            )
            _component_count, roots = connected_components(graph, directed=False, return_labels=True)
            roots = roots.astype(np.int32, copy=False)
        else:
            roots = np.arange(n, dtype=np.int32)

        component_count = int(roots.max()) + 1 if roots.size else 0
        component_sizes = np.bincount(roots, minlength=component_count).astype(np.int64, copy=False)
        component_first = np.full(component_count, n, dtype=np.int64)
        if roots.size:
            np.minimum.at(component_first, roots.astype(np.intp, copy=False), np.arange(n, dtype=np.int64))
        component_order = np.argsort(component_first, kind="stable")
        if component_count and not np.array_equal(component_order, np.arange(component_count, dtype=component_order.dtype)):
            remap = np.empty(component_count, dtype=np.int32)
            remap[component_order.astype(np.intp, copy=False)] = np.arange(component_count, dtype=np.int32)
            roots = remap[roots.astype(np.intp, copy=False)]
            component_sizes = component_sizes[component_order.astype(np.intp, copy=False)]

        component_offsets = np.empty(component_count + 1, dtype=np.int64)
        component_offsets[0] = 0
        if component_count:
            component_offsets[1:] = np.cumsum(component_sizes, dtype=np.int64)
        component_indices = np.argsort(roots, kind="stable").astype(np.int32, copy=False)
        zero_tie_component_by_pos = roots.astype(np.int32, copy=False)
        offset = int(component_offsets[-1]) if component_offsets.size else 0
        self.zero_tie_component_offsets = component_offsets
        self.zero_tie_component_indices = component_indices[:offset]
        self._zero_tie_components_cache = None
        self.zero_tie_component_by_pos = zero_tie_component_by_pos

        reference_voltage_by_pos = np.full(n, np.nan, dtype=np.float64)
        references = np.asarray(getattr(self, "references", ()), dtype=np.int64)
        ref_valid = (references >= 0) & (references < self._node_idx_by_pos.size)
        if np.any(ref_valid):
            ref_pos = references[ref_valid].astype(np.intp, copy=False)
            node_voltage_values = getattr(self, "_node_voltage_measurement_value_by_pos", None)
            node_voltage_mask = getattr(self, "_node_voltage_measurement_pos_mask_cache", None)
            if (
                isinstance(node_voltage_values, np.ndarray)
                and isinstance(node_voltage_mask, np.ndarray)
                and node_voltage_values.size == n
                and node_voltage_mask.size == n
            ):
                ref_values = self._node_voltage_by_pos[ref_pos].astype(np.float64, copy=True)
                measured_ref = node_voltage_mask[ref_pos]
                if np.any(measured_ref):
                    ref_values[measured_ref] = node_voltage_values[ref_pos[measured_ref]]
            else:
                ref_values = np.fromiter(
                    (
                        float(
                            self.node_voltage_measurements.get(
                                int(self._node_idx_by_pos[int(pos)]),
                                self._node_voltage_by_pos[int(pos)],
                            )
                        )
                        for pos in ref_pos
                    ),
                    dtype=np.float64,
                    count=int(ref_pos.size),
                )
            reference_voltage_by_pos[ref_pos] = ref_values

        component_sizes = np.diff(component_offsets).astype(np.int64, copy=False)
        component_first = component_indices[component_offsets[:-1].astype(np.intp, copy=False)].astype(np.int32, copy=False)
        component_for_pos = zero_tie_component_by_pos.astype(np.int64, copy=False)
        fixed_voltage_by_component = np.full(component_count, np.nan, dtype=np.float64)
        ref_pos = np.flatnonzero(np.isfinite(reference_voltage_by_pos)).astype(np.int64, copy=False)
        if ref_pos.size:
            ref_component = component_for_pos[ref_pos.astype(np.intp, copy=False)]
            order = np.lexsort((ref_pos, ref_component))
            sorted_component = ref_component[order]
            unique_component, first = np.unique(sorted_component, return_index=True)
            fixed_voltage_by_component[unique_component.astype(np.intp, copy=False)] = np.maximum(
                reference_voltage_by_pos[ref_pos[order][first].astype(np.intp, copy=False)],
                self.voltage_floor,
            )
        fixed_component_mask = np.isfinite(fixed_voltage_by_component)
        state_components = np.flatnonzero(~fixed_component_mask).astype(np.int64, copy=False)
        state_index_by_component = np.full(component_count, -1, dtype=np.int32)
        if state_components.size:
            state_index_by_component[state_components.astype(np.intp, copy=False)] = np.arange(
                state_components.size,
                dtype=np.int32,
            )
        node_voltage_state = state_index_by_component[component_for_pos.astype(np.intp, copy=False)].astype(np.int32, copy=False)

        self.node_voltage_state = node_voltage_state
        self.voltage_state_pos = component_first[state_components.astype(np.intp, copy=False)].astype(np.int32, copy=False)
        component_for_ordered_nodes = component_for_pos[component_indices[:offset].astype(np.intp, copy=False)]
        variable_node_mask = ~fixed_component_mask[component_for_ordered_nodes.astype(np.intp, copy=False)]
        self.voltage_state_node_indices = component_indices[:offset][variable_node_mask].astype(np.int32, copy=False)
        variable_counts = component_sizes[state_components.astype(np.intp, copy=False)]
        self.voltage_state_node_offsets = np.empty(state_components.size + 1, dtype=np.int64)
        self.voltage_state_node_offsets[0] = 0
        if variable_counts.size:
            self.voltage_state_node_offsets[1:] = np.cumsum(variable_counts, dtype=np.int64)
        else:
            self.voltage_state_node_offsets = np.asarray([0], dtype=np.int64)
        self._voltage_state_nodes_cache = None
        self.voltage_col = node_voltage_state.copy()
        self.n_voltage = int(self.voltage_state_pos.size)
        self._voltage_expand_pos = self.voltage_state_node_indices.astype(np.int64, copy=False)
        self._voltage_expand_col = node_voltage_state[self._voltage_expand_pos.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        fixed_node_mask = fixed_component_mask[component_for_pos.astype(np.intp, copy=False)]
        if np.any(fixed_node_mask):
            self._ref_voltage_pos = np.flatnonzero(fixed_node_mask).astype(np.int64, copy=False)
            self._ref_voltage_value = fixed_voltage_by_component[
                component_for_pos[self._ref_voltage_pos.astype(np.intp, copy=False)].astype(np.intp, copy=False)
            ].astype(np.float64, copy=False)
            self.ref_voltages = {
                int(pos): float(value)
                for pos, value in zip(self._ref_voltage_pos, self._ref_voltage_value)
            }
        else:
            self.ref_voltages = {}
            self._ref_voltage_pos = np.array([], dtype=np.int64)
            self._ref_voltage_value = np.array([], dtype=np.float64)

    @staticmethod
    def _int_array(values: Sequence[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _bool_array(values: Sequence[bool]) -> np.ndarray:
        return np.asarray(values, dtype=bool)

    def _build_apply_state_index(self) -> None:
        """Cache device-to-state positions used when writing estimated values back to the model."""
        self._apply_branch_i = self._branch_i_pos.astype(np.int64, copy=False)
        self._apply_branch_j = self._branch_j_pos.astype(np.int64, copy=False)
        self._apply_branch_inv_r = np.divide(
            1.0,
            self._branch_r,
            out=np.zeros_like(self._branch_r, dtype=np.float64),
            where=np.abs(self._branch_r) > 1e-12,
        )
        self._apply_load_pos = self._load_pos.astype(np.int64, copy=False)
        load_rows = self._load_rows.astype(np.intp, copy=False)
        self._apply_load_pv0 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv0"]]
        self._apply_load_pv1 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv1"]]
        self._apply_load_pv2 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv2"]]
        gen_rows = self._generator_rows.astype(np.intp, copy=False)
        self._apply_generator_pos = self._generator_pos.astype(np.int64, copy=False)
        self._apply_generator_ctrl = self._gen_table[gen_rows, DC_GEN_COLS["control_type"]].astype(np.int64, copy=False)
        self._apply_generator_v_pos = self._generator_v_state_pos.astype(np.int64, copy=False)
        self._apply_generator_p_set = self._gen_table[gen_rows, DC_GEN_COLS["p_set"]].astype(np.float64, copy=False)
        self._apply_generator_i_set = self._gen_table[gen_rows, DC_GEN_COLS["i_set"]].astype(np.float64, copy=False)
        self._apply_break_i = self._break_i_pos.astype(np.int64, copy=False)
        self._apply_break_j = self._break_j_pos.astype(np.int64, copy=False)
        self._apply_break_pos = np.arange(
            self._zero_branch_names.size,
            self._zero_branch_names.size + self._break_names.size,
            dtype=np.int64,
        )
        self._apply_zero_branch_i = self._zero_branch_i_pos.astype(np.int64, copy=False)
        self._apply_zero_branch_j = self._zero_branch_j_pos.astype(np.int64, copy=False)
        self._apply_zero_branch_pos = np.arange(self._zero_branch_names.size, dtype=np.int64)
        self._apply_dcdc_i = self._dcdc_i_pos.astype(np.int64, copy=False)
        self._apply_dcdc_j = self._dcdc_j_pos.astype(np.int64, copy=False)
        self._apply_dcdc_pos = np.arange(self._dcdc_names.size, dtype=np.int64)

    def _build_measurement_plan_device_cache(self) -> None:
        """Cache per-device row-plan metadata shared by all measurement types."""
        node_solver_pos = self._raw_node_solver_pos_alive.astype(np.int64, copy=False)
        self._node_plan_node_pos = node_solver_pos
        self._node_plan_col = self.voltage_col[node_solver_pos.astype(np.intp, copy=False)].astype(np.int64, copy=False)

        self._branch_plan_i = self._branch_i_pos.astype(np.int64, copy=False)
        self._branch_plan_j = self._branch_j_pos.astype(np.int64, copy=False)
        self._branch_plan_i_col = self.voltage_col[self._branch_plan_i.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._branch_plan_j_col = self.voltage_col[self._branch_plan_j.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._branch_plan_inv_r = np.divide(1.0, self._branch_r, out=np.zeros_like(self._branch_r), where=np.abs(self._branch_r) > 1e-12)

        load_rows = self._load_rows.astype(np.intp, copy=False)
        self._load_plan_pos = self._load_pos.astype(np.int64, copy=False)
        self._load_plan_col = self.voltage_col[self._load_plan_pos.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._load_plan_pv0 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv0"]]
        self._load_plan_pv1 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv1"]]
        self._load_plan_pv2 = self._load_table[load_rows, DC_LOAD_COLS["pbase"]] * self._load_table[load_rows, DC_LOAD_COLS["pv2"]]

        gen_rows = self._generator_rows.astype(np.intp, copy=False)
        self._generator_plan_ctrl = self._gen_table[gen_rows, DC_GEN_COLS["control_type"]].astype(np.int64, copy=False)
        self._generator_plan_pos = self._generator_pos.astype(np.int64, copy=False)
        self._generator_plan_col = self.voltage_col[self._generator_plan_pos.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._generator_plan_vgen_pos = self._generator_v_state_pos.astype(np.int64, copy=False)
        self._generator_plan_p_col = np.where(
            self._generator_plan_vgen_pos >= 0,
            self.v_generator_start + self._generator_plan_vgen_pos,
            -1,
        ).astype(np.int64, copy=False)
        self._generator_plan_p_set = self._gen_table[gen_rows, DC_GEN_COLS["p_set"]].astype(np.float64, copy=False)
        self._generator_plan_i_set = self._gen_table[gen_rows, DC_GEN_COLS["i_set"]].astype(np.float64, copy=False)

        self._break_plan_i = self._break_i_pos.astype(np.int64, copy=False)
        self._break_plan_j = self._break_j_pos.astype(np.int64, copy=False)
        self._break_plan_i_col = self.voltage_col[self._break_plan_i.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._break_plan_j_col = self.voltage_col[self._break_plan_j.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._break_plan_current_pos = np.arange(self._zero_branch_names.size, self._zero_branch_names.size + self._break_names.size, dtype=np.int64)
        self._break_plan_current_col = self.switch_start + self._break_plan_current_pos

        self._zero_branch_plan_i = self._zero_branch_i_pos.astype(np.int64, copy=False)
        self._zero_branch_plan_j = self._zero_branch_j_pos.astype(np.int64, copy=False)
        self._zero_branch_plan_i_col = self.voltage_col[self._zero_branch_plan_i.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._zero_branch_plan_j_col = self.voltage_col[self._zero_branch_plan_j.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._zero_branch_plan_current_pos = np.arange(self._zero_branch_names.size, dtype=np.int64)
        self._zero_branch_plan_current_col = self.switch_start + self._zero_branch_plan_current_pos

        constraint_i = np.concatenate((self._zero_branch_i_pos, self._break_i_pos)).astype(np.int64, copy=False)
        constraint_j = np.concatenate((self._zero_branch_j_pos, self._break_j_pos)).astype(np.int64, copy=False)
        self._constraint_plan_i = constraint_i
        self._constraint_plan_j = constraint_j
        self._constraint_plan_i_col = self.voltage_col[constraint_i.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._constraint_plan_j_col = self.voltage_col[constraint_j.astype(np.intp, copy=False)].astype(np.int64, copy=False)

        self._dcdc_plan_i = self._dcdc_i_pos.astype(np.int64, copy=False)
        self._dcdc_plan_j = self._dcdc_j_pos.astype(np.int64, copy=False)
        self._dcdc_plan_i_col = self.voltage_col[self._dcdc_plan_i.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._dcdc_plan_j_col = self.voltage_col[self._dcdc_plan_j.astype(np.intp, copy=False)].astype(np.int64, copy=False)
        self._dcdc_plan_pos = np.arange(self._dcdc_names.size, dtype=np.int64)
        self._dcdc_plan_p_col = self.dcdc_start + 2 * self._dcdc_plan_pos
        self._dcdc_plan_q_col = self._dcdc_plan_p_col + 1
        self._build_measurement_plan_lookup_arrays()

    def _build_measurement_plan_lookup_arrays(self) -> None:
        cache_key = self._measurement_plan_lookup_cache_signature()
        if getattr(self, "_measurement_plan_lookup_cache_key", None) == cache_key:
            return
        meas_kind_source = {
            DEVICE_TYPE_DCNode: _DC_NODE_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCBranch: _DC_TERMINAL_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCLoad: _DC_LOAD_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCGenerator: _DC_GEN_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCZeroBranch: _DC_TERMINAL_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCBreak: _DC_TERMINAL_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCZeroBranchConstraint: _DC_CONSTRAINT_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCBreakConstraint: _DC_CONSTRAINT_MEAS_TYPE_CODES,
            DEVICE_TYPE_DCDCConverter: _DC_TERMINAL_MEAS_TYPE_CODES,
        }
        self._measurement_plan_meas_kind_code_by_type_code = {
            int(code): _measurement_type_code_lookup(kind_map)
            for code, kind_map in meas_kind_source.items()
        }
        self._measurement_plan_lookup_cache_key = cache_key

    def _measurement_plan_lookup_cache_signature(self) -> Tuple[object, ...]:
        return ("dc_numeric_measurement_plan",)

    def _ensure_measurement_plan_lookup_arrays(self) -> None:
        if not hasattr(self, "_measurement_plan_meas_kind_code_by_type_code"):
            self._build_measurement_plan_lookup_arrays()

    def _attach_meas_ppc_device_pos(self) -> None:
        meas_ppc = getattr(self, "meas_ppc", None)
        if not isinstance(meas_ppc, dict):
            return
        constraint_names = np.concatenate(
            (
                np.asarray(getattr(self, "_zero_branch_names", ()), dtype=object),
                np.asarray(getattr(self, "_break_names", ()), dtype=object),
            )
        )
        name_arrays_by_code = {
            DEVICE_TYPE_DCNode: np.asarray(getattr(self, "_raw_node_names_alive", ()), dtype=object),
            DEVICE_TYPE_DCBranch: np.asarray(getattr(self, "_branch_names", ()), dtype=object),
            DEVICE_TYPE_DCBreak: np.asarray(getattr(self, "_break_names", ()), dtype=object),
            DEVICE_TYPE_DCZeroBranch: np.asarray(getattr(self, "_zero_branch_names", ()), dtype=object),
            DEVICE_TYPE_DCGenerator: np.asarray(getattr(self, "_generator_names", ()), dtype=object),
            DEVICE_TYPE_DCLoad: np.asarray(getattr(self, "_load_names", ()), dtype=object),
            DEVICE_TYPE_DCDCConverter: np.asarray(getattr(self, "_dcdc_names", ()), dtype=object),
            DEVICE_TYPE_DCZeroBranchConstraint: constraint_names,
            DEVICE_TYPE_DCBreakConstraint: constraint_names,
        }
        device_pos = attach_device_pos_from_name_arrays(meas_ppc, name_arrays_by_code)
        table = getattr(getattr(self, "measurements", None), "table", None)
        if table is not None and int(getattr(table, "idx", np.asarray([])).size) == int(device_pos.size):
            table.device_pos = device_pos

    def _measurement_device_pos_array(
        self,
        table: MeasurementTable,
        device_pos_by_type_code: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        n_rows = int(table.idx.size)
        precomputed = getattr(table, "device_pos", None)
        if precomputed is not None and np.asarray(precomputed).size == n_rows:
            device_pos = np.asarray(precomputed, dtype=np.int64).copy()
        else:
            meas_ppc = getattr(self, "meas_ppc", None)
            ppc_device_pos = meas_ppc.get("device_pos") if isinstance(meas_ppc, dict) else None
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == n_rows:
                device_pos = np.asarray(ppc_device_pos, dtype=np.int64).copy()
            else:
                device_pos = np.empty(n_rows, dtype=np.int64)
                device_pos.fill(-1)
        if n_rows == 0 or np.all(device_pos >= 0):
            return device_pos
        if np.any(device_pos < 0):
            warnings.warn(
                "DC SE measurement device_pos contains unresolved rows; name-id/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        table.device_pos = device_pos
        return device_pos

    @staticmethod
    def _ensure_table_meas_type_codes(table: MeasurementTable) -> np.ndarray:
        n_rows = int(table.idx.size)
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is not None:
            meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
            if meas_type_code.size == n_rows:
                if np.any(meas_type_code < 0):
                    warnings.warn(
                        "DC SE measurement meas_type_code contains unresolved rows; string fallback is disabled.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                table.meas_type_code = meas_type_code
                return meas_type_code
        warnings.warn(
            "DC SE measurement table is missing meas_type_code; string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        meas_type_code = np.full(n_rows, -1, dtype=np.int16)
        table.meas_type_code = meas_type_code
        return meas_type_code

    def _measurement_table_for_indexed_plan(self, measurements: object) -> MeasurementTable:
        table = _measurement_table_from_array_view(measurements)
        self._ensure_table_meas_type_codes(table)
        table.device_pos = self._measurement_device_pos_array(table)
        return table

    def _active_measurement_plan_tables(self, active_table: MeasurementTable) -> Dict[str, MeasurementPlanTable]:
        """Build the DC active measurement plan table from one array pass."""
        self._ensure_measurement_plan_lookup_arrays()
        n_rows = int(active_table.idx.size)
        row = np.arange(n_rows, dtype=np.int64)
        device_type_code = np.asarray(active_table.device_type_code, dtype=np.int16)
        device_pos = self._measurement_device_pos_array(active_table)
        meas_type_code = self._ensure_table_meas_type_codes(active_table)
        meas_kind = np.full(n_rows, -1, dtype=np.int16)
        kind_code_maps = getattr(self, "_measurement_plan_meas_kind_code_by_type_code", {})
        for code_int, code_rows in rows_by_device_type_code(active_table).items():
            rows = np.asarray(code_rows, dtype=np.int64)
            if rows.size == 0:
                continue
            code_lookup = kind_code_maps.get(int(code_int))
            if code_lookup is None or code_lookup.size == 0:
                continue
            codes = meas_type_code[rows].astype(np.int64, copy=False)
            values = np.full(rows.size, -1, dtype=np.int16)
            in_range = (codes >= 0) & (codes < code_lookup.size)
            if np.any(in_range):
                values[in_range] = code_lookup[codes[in_range].astype(np.intp, copy=False)]
            meas_kind[rows] = values
        return {
            "simple": MeasurementPlanTable(
                table=active_table,
                row=row,
                device_type_code=device_type_code,
                meas_kind=meas_kind,
                device_pos=device_pos,
                handled=(meas_kind >= 0) & (device_pos >= 0),
            )
        }

    def _shrink_measurement_plan_tables(
        self,
        plan_tables: Dict[str, MeasurementPlanTable],
        removed_pos: int,
    ) -> Dict[str, MeasurementPlanTable]:
        base_plan = self._common_measurement_plan_table(plan_tables)
        row_count = int(base_plan.row.size)
        pos = int(removed_pos)
        if pos < 0 or pos >= row_count:
            raise IndexError("bad-data removal row position is out of range")
        keep_rows = np.concatenate(
            (
                np.arange(pos, dtype=np.int64),
                np.arange(pos + 1, row_count, dtype=np.int64),
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

    @staticmethod
    def _load_network(e_file: Path) -> SimpleNamespace:
        """Read the DC case and build topology references used by measurements."""
        ppc = load_dc_ppc_from_e_file(e_file)
        return _build_dc_se_ppc_namespace(ppc)

    @staticmethod
    def _run_power_flow_seed(network: object, params: StateEstimationParameters, e_file: Path) -> bool:
        seed_tol = max(float(params.power_flow_tol), 1e-6)
        ppc = DCStateEstimator._power_flow_seed_ppc_from_network(network)
        if ppc is None:
            return False
        calc = DCPowerFlowCalc(
            ppc,
            tol=seed_tol,
            max_iter=params.power_flow_max_iter,
            min_voltage=params.power_flow_min_voltage,
            result_mode="none",
            linear_solver=getattr(network, "_se_power_flow_linear_solver", None),
        )
        calc.skip_lf_result = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = calc.run()
        except Exception:
            return False
        if rc != 0 or not calc.converged:
            return False
        DCStateEstimator._apply_power_flow_seed_calc_state_to_network(network, ppc, calc)
        return True

    @staticmethod
    def _copy_ppc_for_power_flow_seed(ppc):
        copied = dict(ppc)
        for key in (
            "bus",
            "branch",
            "load",
            "gen",
            "zero_branch",
            "switch",
            "break",
            "dcdc",
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
        if not (isinstance(source, dict) and source.get("format") == "dc_ppc_v1"):
            source = getattr(network, "_array_model", None)
        if not (isinstance(source, dict) and source.get("format") == "dc_ppc_v1"):
            return None
        ppc = DCStateEstimator._copy_ppc_for_power_flow_seed(source)
        seed_rows = getattr(network, "_se_power_flow_seed_rows", None)
        if seed_rows is not None:
            DCStateEstimator._apply_power_flow_seed_rows_to_ppc(ppc, seed_rows)
        return ppc

    @staticmethod
    def _apply_power_flow_seed_rows_to_ppc(ppc, seed_rows) -> None:
        bus = ppc.get("bus")
        if bus is None:
            return
        if not isinstance(seed_rows, dict):
            warnings.warn(
                "DC SE power-flow seed requires packed key arrays; row tuple fallback is disabled.",
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
                "DC SE power-flow seed requires same-sized measurement_key, ppc_row and value arrays.",
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
        dcdc = ppc.get("dcdc")

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
                bus_idx = bus[:, DC_BUS_COLS["idx"]].astype(np.int64, copy=False)
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
            bus[bus_rows, DC_BUS_COLS["voltage"]] = np.maximum(voltage_values[in_range][hit], 0.0)

        mask = (device_type == DEVICE_TYPE_DCNode) & (meas_type == MEAS_TYPE_V)
        rows, row_values = valid_rows(mask, bus.shape[0])
        if rows.size:
            bus[rows, DC_BUS_COLS["voltage"]] = np.maximum(row_values, 0.0)

        if isinstance(gen, np.ndarray) and gen.size:
            gen_device = device_type == DEVICE_TYPE_DCGenerator
            mask = gen_device & (meas_type == MEAS_TYPE_P_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                gen[rows, DC_GEN_COLS["p_set"]] = row_values
                gen[rows, DC_GEN_COLS["p"]] = row_values
            mask = gen_device & (meas_type == MEAS_TYPE_V_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                voltage = np.maximum(row_values, 0.0)
                gen[rows, DC_GEN_COLS["v_set"]] = voltage
                set_bus_voltage_by_idx(gen[rows, DC_GEN_COLS["node"]], voltage)
            mask = gen_device & (meas_type == MEAS_TYPE_I_GEN)
            rows, row_values = valid_rows(mask, gen.shape[0])
            if rows.size:
                gen[rows, DC_GEN_COLS["i_set"]] = row_values
                gen[rows, DC_GEN_COLS["current"]] = row_values

        if isinstance(load, np.ndarray) and load.size:
            load_device = device_type == DEVICE_TYPE_DCLoad
            mask = load_device & (meas_type == MEAS_TYPE_P_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                load[rows, DC_LOAD_COLS["pbase"]] = 1.0
                load[rows, DC_LOAD_COLS["pv0"]] = row_values
                load[rows, DC_LOAD_COLS["pv1"]] = 0.0
                load[rows, DC_LOAD_COLS["pv2"]] = 0.0
                load[rows, DC_LOAD_COLS["p"]] = row_values
            mask = load_device & (meas_type == MEAS_TYPE_V_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                set_bus_voltage_by_idx(load[rows, DC_LOAD_COLS["node"]], row_values)
            mask = load_device & (meas_type == MEAS_TYPE_I_LOAD)
            rows, row_values = valid_rows(mask, load.shape[0])
            if rows.size:
                load[rows, DC_LOAD_COLS["current"]] = row_values

        if isinstance(dcdc, np.ndarray) and dcdc.size:
            dcdc_device = device_type == DEVICE_TYPE_DCDCConverter
            mask = dcdc_device & (meas_type == MEAS_TYPE_P_FROM)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                dcdc[rows, DC_DCDC_COLS["p_set"]] = row_values
                dcdc[rows, DC_DCDC_COLS["i_p"]] = row_values
            mask = dcdc_device & (meas_type == MEAS_TYPE_P_TO)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                dcdc[rows, DC_DCDC_COLS["j_p"]] = row_values
            mask = dcdc_device & (meas_type == MEAS_TYPE_V_FROM)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                set_bus_voltage_by_idx(dcdc[rows, DC_DCDC_COLS["i_node"]], row_values)
            mask = dcdc_device & (meas_type == MEAS_TYPE_V_TO)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                set_bus_voltage_by_idx(dcdc[rows, DC_DCDC_COLS["j_node"]], row_values)
            mask = dcdc_device & (meas_type == MEAS_TYPE_I_FROM)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                dcdc[rows, DC_DCDC_COLS["i_set"]] = row_values
                dcdc[rows, DC_DCDC_COLS["i_c"]] = row_values
            mask = dcdc_device & (meas_type == MEAS_TYPE_I_TO)
            rows, row_values = valid_rows(mask, dcdc.shape[0])
            if rows.size:
                dcdc[rows, DC_DCDC_COLS["j_c"]] = row_values

    @staticmethod
    def _overlay_power_flow_seed_result_ppc(seed_ppc, result):
        if not isinstance(result, dict):
            return seed_ppc
        for key in (
            "bus",
            "branch",
            "load",
            "gen",
            "zero_branch",
            "switch",
            "break",
            "dcdc",
        ):
            value = result.get(key)
            if isinstance(value, np.ndarray):
                seed_ppc[key] = value
        return seed_ppc

    @staticmethod
    def _apply_power_flow_seed_ppc_to_network(network, ppc) -> None:
        network.ppc = ppc
        if hasattr(network, "_array_model"):
            network._array_model = ppc
        if hasattr(network, "topology"):
            network.topology = ppc.get("_topology_arrays", network.topology)

    @staticmethod
    def _apply_power_flow_seed_calc_state_to_network(network, ppc, calc) -> None:
        bus = ppc.get("bus")
        if hasattr(calc, "x") and hasattr(calc, "N"):
            voltage = np.asarray(calc.x[: calc.N], dtype=np.float64)
            if isinstance(bus, np.ndarray) and bus.size:
                active_bus_pos = getattr(calc, "_active_bus_pos", None)
                if isinstance(active_bus_pos, np.ndarray) and active_bus_pos.size == voltage.size:
                    rows = active_bus_pos.astype(np.intp, copy=False)
                    valid = (rows >= 0) & (rows < bus.shape[0])
                    if np.any(valid):
                        bus[rows[valid], DC_BUS_COLS["voltage"]] = voltage[valid]
                else:
                    node_pos = getattr(calc, "alive_node_dict", None)
                    if node_pos is not None and hasattr(node_pos, "get"):
                        bus[:, DC_BUS_COLS["voltage"]] = 0.0
                        for row, node_idx in enumerate(bus[:, DC_BUS_COLS["idx"]].astype(np.int64, copy=False)):
                            pos = node_pos.get(int(node_idx), -1)
                            if 0 <= int(pos) < voltage.size:
                                bus[row, DC_BUS_COLS["voltage"]] = voltage[int(pos)]
                    elif voltage.size == bus.shape[0]:
                        bus[:, DC_BUS_COLS["voltage"]] = voltage
            DCStateEstimator._apply_power_flow_seed_ppc_to_network(network, ppc)
            return
        result_ppc = DCStateEstimator._overlay_power_flow_seed_result_ppc(ppc, getattr(calc, "result", None))
        DCStateEstimator._apply_power_flow_seed_ppc_to_network(network, result_ppc)

    def _apply_measurement_seed_to_network(self) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is None:
            seed_rows = {
                "measurement_key": np.empty(0, dtype=np.int64),
                "ppc_row": np.empty(0, dtype=np.int64),
                "value": np.empty(0, dtype=np.float64),
            }
        setattr(self.network, "_se_power_flow_seed_rows", seed_rows)

    @staticmethod
    def _active_measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
        if int(device_pos) < 0:
            return -1
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
        self._set_active_key_caches_from_array(
            key_array,
            measurement_keys if isinstance(measurement_keys, set) else None,
        )

    def _set_active_key_caches_from_array(self, key_array: np.ndarray, key_set: Optional[set] = None) -> None:
        key_array = np.asarray(key_array, dtype=np.int64).reshape(-1)
        if key_set is None:
            key_set = _PackedMeasurementKeyCache(self)
        self._active_measurement_keys = key_set
        self._active_measurement_key_cache = key_set
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
            "_active_measurement_key_cache",
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

    def _measurement_ppc_rows_for_device_positions(self, device_type_code: np.ndarray, device_pos: np.ndarray) -> np.ndarray:
        rows = np.full(np.asarray(device_pos).size, -1, dtype=np.int64)
        codes = np.asarray(device_type_code, dtype=np.int16)
        pos = np.asarray(device_pos, dtype=np.int64)

        def assign(code_value: int, source_rows: np.ndarray) -> None:
            mask = codes == int(code_value)
            if not np.any(mask):
                return
            local_pos = pos[mask]
            valid = (local_pos >= 0) & (local_pos < source_rows.size)
            selected = np.flatnonzero(mask)
            if np.any(valid):
                rows[selected[valid]] = source_rows[local_pos[valid].astype(np.intp, copy=False)]

        assign(DEVICE_TYPE_DCNode, self._raw_node_rows_alive)
        assign(DEVICE_TYPE_DCGenerator, self._generator_rows)
        assign(DEVICE_TYPE_DCLoad, self._load_rows)
        assign(DEVICE_TYPE_DCDCConverter, self._dcdc_rows)
        return rows

    @staticmethod
    def _values_for_plan_pos(plan_values: np.ndarray, device_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(plan_values, dtype=np.int64)
        pos = np.asarray(device_pos, dtype=np.int64)
        out = np.full(pos.size, -1, dtype=np.int64)
        valid = (pos >= 0) & (pos < values.size)
        if np.any(valid):
            out[valid] = values[pos[valid].astype(np.intp, copy=False)]
        return valid, out

    def _measurement_scale_for_codes(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        rows_by_code: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        n_rows = int(device_type_code.size)
        available = np.zeros(n_rows, dtype=bool)
        scale = np.ones(n_rows, dtype=np.float64)
        from_pos = np.full(n_rows, -1, dtype=np.int64)
        to_pos = np.full(n_rows, -1, dtype=np.int64)
        n_nodes = int(self._node_voltage_file_base_by_pos.size)

        def rows_for(device_code: int) -> np.ndarray:
            if rows_by_code is not None:
                rows = rows_by_code.get(int(device_code))
                if rows is None:
                    return np.asarray([], dtype=np.int64)
                return np.asarray(rows, dtype=np.int64)
            return np.flatnonzero(device_type_code == int(device_code)).astype(np.int64, copy=False)

        def apply_node(rows: np.ndarray, node_pos_array: np.ndarray, voltage_code: int) -> None:
            if rows.size == 0:
                return
            valid_plan, node_pos = self._values_for_plan_pos(node_pos_array, device_pos[rows])
            valid_node = valid_plan & (node_pos >= 0) & (node_pos < n_nodes)
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            pos_valid = node_pos[valid_node].astype(np.intp, copy=False)
            kind = meas_type_code[rows_valid]
            voltage_mask = kind == int(voltage_code)
            if not np.any(voltage_mask):
                return
            selected = rows_valid[voltage_mask]
            from_pos[selected] = pos_valid[voltage_mask]
            available[selected] = True
            scale[selected] = self._node_voltage_file_base_by_pos[pos_valid[voltage_mask]]

        def apply_single(rows: np.ndarray, node_pos_array: np.ndarray, p_code: int, v_code: int, i_code: int) -> None:
            if rows.size == 0:
                return
            valid_plan, node_pos = self._values_for_plan_pos(node_pos_array, device_pos[rows])
            valid_node = valid_plan & (node_pos >= 0) & (node_pos < n_nodes)
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            pos_valid = node_pos[valid_node].astype(np.intp, copy=False)
            kind = meas_type_code[rows_valid]
            supported = (kind == int(p_code)) | (kind == int(v_code)) | (kind == int(i_code))
            if not np.any(supported):
                return
            selected = rows_valid[supported]
            selected_pos = pos_valid[supported]
            selected_kind = kind[supported]
            from_pos[selected] = selected_pos
            available[selected] = True
            scale[selected[selected_kind == int(p_code)]] = self.p_base
            voltage_mask = selected_kind == int(v_code)
            if np.any(voltage_mask):
                scale[selected[voltage_mask]] = self._node_voltage_file_base_by_pos[selected_pos[voltage_mask]]
            current_mask = selected_kind == int(i_code)
            if np.any(current_mask):
                scale[selected[current_mask]] = self._node_current_file_base_by_pos[selected_pos[current_mask]]

        def apply_terminal(rows: np.ndarray, i_pos_array: np.ndarray, j_pos_array: np.ndarray) -> None:
            if rows.size == 0:
                return
            i_valid_plan, i_pos = self._values_for_plan_pos(i_pos_array, device_pos[rows])
            j_valid_plan, j_pos = self._values_for_plan_pos(j_pos_array, device_pos[rows])
            valid_node = i_valid_plan & j_valid_plan & (i_pos >= 0) & (j_pos >= 0) & (i_pos < n_nodes) & (j_pos < n_nodes)
            rows_valid = rows[valid_node]
            if rows_valid.size == 0:
                return
            i_valid = i_pos[valid_node].astype(np.intp, copy=False)
            j_valid = j_pos[valid_node].astype(np.intp, copy=False)
            kind = meas_type_code[rows_valid]
            supported = (
                (kind == MEAS_TYPE_P_FROM)
                | (kind == MEAS_TYPE_P_TO)
                | (kind == MEAS_TYPE_V_FROM)
                | (kind == MEAS_TYPE_V_TO)
                | (kind == MEAS_TYPE_I_FROM)
                | (kind == MEAS_TYPE_I_TO)
            )
            if not np.any(supported):
                return
            selected = rows_valid[supported]
            selected_i = i_valid[supported]
            selected_j = j_valid[supported]
            selected_kind = kind[supported]
            from_pos[selected] = selected_i
            to_pos[selected] = selected_j
            available[selected] = True
            power_mask = (selected_kind == MEAS_TYPE_P_FROM) | (selected_kind == MEAS_TYPE_P_TO)
            scale[selected[power_mask]] = self.p_base
            mask = selected_kind == MEAS_TYPE_V_FROM
            if np.any(mask):
                scale[selected[mask]] = self._node_voltage_file_base_by_pos[selected_i[mask]]
            mask = selected_kind == MEAS_TYPE_I_FROM
            if np.any(mask):
                scale[selected[mask]] = self._node_current_file_base_by_pos[selected_i[mask]]
            mask = selected_kind == MEAS_TYPE_V_TO
            if np.any(mask):
                scale[selected[mask]] = self._node_voltage_file_base_by_pos[selected_j[mask]]
            mask = selected_kind == MEAS_TYPE_I_TO
            if np.any(mask):
                scale[selected[mask]] = self._node_current_file_base_by_pos[selected_j[mask]]

        def apply_constraint(rows: np.ndarray) -> None:
            if rows.size == 0:
                return
            constraint_i_pos = np.concatenate((self._zero_branch_i_pos, self._break_i_pos)).astype(
                np.int64,
                copy=False,
            )
            constraint_j_pos = np.concatenate((self._zero_branch_j_pos, self._break_j_pos)).astype(
                np.int64,
                copy=False,
            )
            apply_terminal(rows, constraint_i_pos, constraint_j_pos)
            diff_rows = rows[meas_type_code[rows] == MEAS_TYPE_V_DIFF]
            if diff_rows.size == 0:
                return
            valid_plan, i_pos = self._values_for_plan_pos(constraint_i_pos, device_pos[diff_rows])
            valid_node = valid_plan & (i_pos >= 0) & (i_pos < n_nodes)
            if np.any(valid_node):
                selected = diff_rows[valid_node]
                from_pos[selected] = i_pos[valid_node]
                available[selected] = True
                scale[selected] = self._node_voltage_file_base_by_pos[i_pos[valid_node].astype(np.intp, copy=False)]

        apply_node(rows_for(DEVICE_TYPE_DCNode), self._raw_node_solver_pos_alive, MEAS_TYPE_V)
        apply_terminal(rows_for(DEVICE_TYPE_DCBranch), self._branch_i_pos, self._branch_j_pos)
        apply_terminal(rows_for(DEVICE_TYPE_DCBreak), self._break_i_pos, self._break_j_pos)
        apply_terminal(rows_for(DEVICE_TYPE_DCZeroBranch), self._zero_branch_i_pos, self._zero_branch_j_pos)
        apply_terminal(rows_for(DEVICE_TYPE_DCDCConverter), self._dcdc_i_pos, self._dcdc_j_pos)
        apply_single(rows_for(DEVICE_TYPE_DCGenerator), self._generator_pos, MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN)
        apply_single(rows_for(DEVICE_TYPE_DCLoad), self._load_pos, MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD)
        apply_constraint(
            np.concatenate(
                (
                    rows_for(DEVICE_TYPE_DCZeroBranchConstraint),
                    rows_for(DEVICE_TYPE_DCBreakConstraint),
                )
            ).astype(np.int64, copy=False)
        )
        return available, scale, from_pos, to_pos

    def _measurement_voltage_node_positions_from_codes(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _available, _scale, from_pos, to_pos = self._measurement_scale_for_codes(
            device_type_code,
            device_pos,
            meas_type_code,
        )
        return from_pos, to_pos

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
                return DCStateEstimator._active_measurement_key_array(
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
        return DCStateEstimator._active_measurement_key_array(
            np.asarray(device_type_code, dtype=np.int16)[rows][valid_pos],
            pos_values[valid_pos],
            np.asarray(meas_type_code, dtype=np.int16)[rows][valid_pos],
        ).astype(np.int64, copy=False)

    @staticmethod
    def _active_key_cache_from_arrays(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
        active_mask: np.ndarray,
    ) -> set:
        keys = DCStateEstimator._active_key_array_from_arrays(
            device_type_code,
            device_pos,
            meas_type_code,
            active_mask,
        )
        return set(keys.astype(object, copy=False))

    def _voltage_best_from_arrays(
        self,
        table: MeasurementTable,
        real_mask: np.ndarray,
        device_type_code: np.ndarray,
        meas_type_code: np.ndarray,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
    ) -> Tuple[Mapping[int, float], Mapping[int, float]]:
        n_nodes = int(self._node_idx_by_pos.size)
        node_best_weight = np.full(n_nodes, -np.inf, dtype=np.float64)
        node_best_value = np.zeros(n_nodes, dtype=np.float64)
        real_best_weight = np.full(n_nodes, -np.inf, dtype=np.float64)
        real_best_value = np.zeros(n_nodes, dtype=np.float64)
        row_index = np.arange(int(table.idx.size), dtype=np.int64)
        weight = np.asarray(table.weight, dtype=np.float64)
        value = np.asarray(table.value, dtype=np.float64)

        def update(rows: np.ndarray, pos_values: np.ndarray, best_weight: np.ndarray, best_value: np.ndarray) -> None:
            if rows.size == 0:
                return
            pos_values = np.asarray(pos_values, dtype=np.int64)
            in_range = (pos_values >= 0) & (pos_values < n_nodes)
            if not np.any(in_range):
                return
            rows_valid = rows[in_range]
            pos_valid = pos_values[in_range].astype(np.intp, copy=False)
            weights = weight[rows_valid]
            values = value[rows_valid]
            np.maximum.at(best_weight, pos_valid, weights)
            keep = weights >= best_weight[pos_valid]
            if np.any(keep):
                best_value[pos_valid[keep]] = values[keep]

        node_v_rows = row_index[real_mask & (device_type_code == DEVICE_TYPE_DCNode) & (meas_type_code == MEAS_TYPE_V)]
        update(node_v_rows, from_pos[node_v_rows], node_best_weight, node_best_value)
        update(node_v_rows, from_pos[node_v_rows], real_best_weight, real_best_value)
        gen_v_rows = row_index[real_mask & (device_type_code == DEVICE_TYPE_DCGenerator) & (meas_type_code == MEAS_TYPE_V_GEN)]
        load_v_rows = row_index[real_mask & (device_type_code == DEVICE_TYPE_DCLoad) & (meas_type_code == MEAS_TYPE_V_LOAD)]
        update(gen_v_rows, from_pos[gen_v_rows], real_best_weight, real_best_value)
        update(load_v_rows, from_pos[load_v_rows], real_best_weight, real_best_value)
        terminal_mask = (
            (device_type_code == DEVICE_TYPE_DCBranch)
            | (device_type_code == DEVICE_TYPE_DCZeroBranch)
            | (device_type_code == DEVICE_TYPE_DCBreak)
            | (device_type_code == DEVICE_TYPE_DCDCConverter)
        )
        v_from_rows = row_index[real_mask & terminal_mask & (meas_type_code == MEAS_TYPE_V_FROM)]
        v_to_rows = row_index[real_mask & terminal_mask & (meas_type_code == MEAS_TYPE_V_TO)]
        update(v_from_rows, from_pos[v_from_rows], real_best_weight, real_best_value)
        update(v_to_rows, to_pos[v_to_rows], real_best_weight, real_best_value)
        self._node_voltage_measurement_pos_mask_cache = np.isfinite(node_best_weight)
        self._real_voltage_observation_pos_mask_cache = np.isfinite(real_best_weight)
        self._node_voltage_measurement_value_by_pos = node_best_value
        self._real_voltage_observation_value_by_pos = real_best_value
        self._real_voltage_observed_solver_pos_mask_cache = None
        return (
            _NodeIndexedValueMap(self._node_idx_by_pos, node_best_value, self._node_voltage_measurement_pos_mask_cache),
            _NodeIndexedValueMap(self._node_idx_by_pos, real_best_value, self._real_voltage_observation_pos_mask_cache),
        )

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
            (device_type_code == DEVICE_TYPE_DCNode)
            & (meas_type_code == MEAS_TYPE_V)
        ) | (
            (device_type_code == DEVICE_TYPE_DCGenerator)
            & np.isin(meas_type_code, (MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN))
        ) | (
            (device_type_code == DEVICE_TYPE_DCLoad)
            & np.isin(meas_type_code, (MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD))
        ) | (
            (device_type_code == DEVICE_TYPE_DCDCConverter)
            & np.isin(
                meas_type_code,
                (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO),
            )
        )
        rows = np.flatnonzero(processable_mask & seed_rows_mask & (device_pos >= 0)).astype(np.int64, copy=False)
        if rows.size == 0:
            return empty_seed()
        ppc_rows = self._measurement_ppc_rows_for_device_positions(device_type_code[rows], device_pos[rows])
        valid = ppc_rows >= 0
        if not np.any(valid):
            return empty_seed()
        rows = rows[valid]
        ppc_rows = ppc_rows[valid]
        return {
            "measurement_key": self._active_measurement_key_array(
                device_type_code[rows],
                device_pos[rows],
                meas_type_code[rows],
            ).astype(np.int64, copy=False),
            "ppc_row": ppc_rows.astype(np.int64, copy=False),
            "value": np.asarray(table.value, dtype=np.float64)[rows].astype(np.float64, copy=False),
        }

    def _measurement_runtime_array_cache_key(self) -> Tuple[object, ...]:
        return (
            id(getattr(self, "_raw_node_solver_pos_alive", None)),
            id(getattr(self, "_branch_i_pos", None)),
            id(getattr(self, "_branch_j_pos", None)),
            id(getattr(self, "_zero_branch_i_pos", None)),
            id(getattr(self, "_zero_branch_j_pos", None)),
            id(getattr(self, "_break_i_pos", None)),
            id(getattr(self, "_break_j_pos", None)),
            id(getattr(self, "_generator_pos", None)),
            id(getattr(self, "_load_pos", None)),
            id(getattr(self, "_dcdc_i_pos", None)),
            id(getattr(self, "_dcdc_j_pos", None)),
            float(getattr(self, "p_base", 0.0)),
            float(getattr(self, "u_scale", 0.0)),
            float(getattr(self, "i_scale", 0.0)),
        )

    @staticmethod
    def _cached_measurement_runtime_arrays(meas_ppc: Dict, n_rows: int, cache_key: Tuple[object, ...]):
        if meas_ppc.get("_dc_se_runtime_cache_key") != cache_key:
            return None
        arrays = tuple(meas_ppc.get(key) for key in ("device_pos", "available", "scale", "from_pos", "to_pos"))
        if any((not isinstance(value, np.ndarray)) or int(value.size) != int(n_rows) for value in arrays):
            return None
        return arrays

    def _normalize_measurements_to_pu_from_meas_ppc(self, table: MeasurementTable, meas_ppc: Dict) -> bool:
        meas = meas_ppc.get("meas")
        has_meas = isinstance(meas, np.ndarray) and meas.ndim == 2 and meas.shape[0] == table.idx.size
        cols = meas_ppc.get("meas_cols", MEAS_COLS)
        value = table.value
        valid = table.valid
        weight = table.weight
        status = measurement_table_status_code(table)
        n_rows = int(value.size)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is not None:
            device_pos = np.asarray(device_pos, dtype=np.int64)
            if int(device_pos.size) != n_rows:
                device_pos = None
        if device_pos is None:
            ppc_device_pos = meas_ppc.get("device_pos")
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == n_rows:
                device_pos = np.asarray(ppc_device_pos, dtype=np.int64)
        device_type_code = meas_ppc.get("device_type_code_array")
        meas_type_code = meas_ppc.get("meas_type_code_array")
        if not isinstance(device_type_code, np.ndarray) and has_meas:
            device_type_code = meas[:, cols["device_type_code"]].astype(np.int16, copy=False)
        if not isinstance(meas_type_code, np.ndarray) and has_meas:
            meas_type_code = meas[:, cols["meas_type_code"]].astype(np.int16, copy=False)
        if isinstance(device_type_code, np.ndarray):
            device_type_code = np.asarray(device_type_code, dtype=np.int16)
        if isinstance(meas_type_code, np.ndarray):
            meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        if device_pos is None or not isinstance(device_type_code, np.ndarray) or not isinstance(meas_type_code, np.ndarray):
            return False
        if device_pos.size != n_rows or device_type_code.size != n_rows or meas_type_code.size != n_rows:
            return False

        table.device_pos = device_pos
        table.device_type_code = device_type_code
        table.meas_type_code = meas_type_code
        self._ensure_measurement_plan_lookup_arrays()
        runtime_cache_key = self._measurement_runtime_array_cache_key()
        cached_runtime = self._cached_measurement_runtime_arrays(meas_ppc, n_rows, runtime_cache_key)
        if cached_runtime is None:
            available, scale, from_pos, to_pos = self._measurement_scale_for_codes(
                device_type_code,
                device_pos,
                meas_type_code,
                rows_by_code=rows_by_device_type_code(table),
            )
            runtime_copy = not bool(meas_ppc.get("_mutable_runtime_arrays", False))
            meas_ppc["device_pos"] = device_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["available"] = available.astype(bool, copy=runtime_copy)
            meas_ppc["scale"] = scale.astype(np.float64, copy=runtime_copy)
            meas_ppc["from_pos"] = from_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["to_pos"] = to_pos.astype(np.int64, copy=runtime_copy)
            meas_ppc["_dc_se_runtime_cache_key"] = runtime_cache_key
        else:
            device_pos, available, scale, from_pos, to_pos = cached_runtime
            device_pos = np.asarray(device_pos, dtype=np.int64)
            available = np.asarray(available, dtype=bool)
            scale = np.asarray(scale, dtype=np.float64)
            from_pos = np.asarray(from_pos, dtype=np.int64)
            to_pos = np.asarray(to_pos, dtype=np.int64)
        table.device_pos = device_pos
        table.available = available
        table.scale = scale
        table.from_pos = from_pos
        table.to_pos = to_pos

        candidate = valid & (weight > 0.0)
        unavailable = candidate & (~available)
        if np.any(unavailable):
            valid[unavailable] = False
            status[unavailable] = MEAS_STATUS_INVALID
        processable = candidate & available
        if np.any(processable):
            value[processable] = np.divide(
                value[processable],
                scale[processable],
                out=value[processable].copy(),
                where=np.abs(scale[processable]) > 1e-12,
            )
        real_mask = processable & (status != MEAS_STATUS_PSEUDO)
        self.measurement_table = table
        active_key_array = self._active_key_array_from_arrays(
            device_type_code,
            device_pos,
            meas_type_code,
            processable,
        )
        self._set_active_key_caches_from_array(active_key_array)
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self._node_voltage_measurement_cache, self._real_voltage_observation_node_cache = self._voltage_best_from_arrays(
            table,
            real_mask,
            device_type_code,
            meas_type_code,
            from_pos,
            to_pos,
        )
        self._power_flow_seed_rows = self._power_flow_seed_rows_from_arrays(
            table,
            processable,
            device_type_code,
            meas_type_code,
            device_pos,
        )
        try:
            self.measurements.normalized = True
        except AttributeError:
            pass
        meas_ppc["value_array"] = value
        meas_ppc["valid_array"] = valid
        meas_ppc["status_array"] = status
        if has_meas:
            meas[:, cols["value"]] = value.astype(np.float64, copy=False)
            meas[:, cols["valid"]] = valid.astype(np.float64, copy=False)
            meas[:, cols["status"]] = status.astype(np.float64, copy=False)
        meas_ppc["normalized"] = True
        return True

    def _normalize_measurements_to_pu(self) -> None:
        table = _measurement_table_from_array_view(self.measurements)
        if getattr(self.measurements, "normalized", False):
            self.measurement_table = table
            self._refresh_measurement_summary_cache()
            if isinstance(getattr(self, "meas_ppc", None), dict):
                self.meas_ppc["normalized"] = True
            return
        if isinstance(getattr(self, "meas_ppc", None), dict) and self._normalize_measurements_to_pu_from_meas_ppc(table, self.meas_ppc):
            return
        self.measurement_table = table
        self._set_active_key_caches(set())
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self._node_voltage_measurement_cache = {}
        self._real_voltage_observation_node_cache = {}
        self._node_voltage_measurement_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
        self._real_voltage_observation_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
        self._node_voltage_measurement_value_by_pos = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.float64)
        self._real_voltage_observation_value_by_pos = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.float64)
        self._real_voltage_observed_solver_pos_mask_cache = None
        self._power_flow_seed_rows = {
            "measurement_key": np.empty(0, dtype=np.int64),
            "ppc_row": np.empty(0, dtype=np.int64),
            "value": np.empty(0, dtype=np.float64),
        }
        warnings.warn(
            "DC SE measurement normalization requires measurement PPC arrays; object/string fallback is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _convert_measurements_to_pu(self) -> None:
        self._normalize_measurements_to_pu()

    def _active_measurement_keys_ref(self) -> set:
        """Return active measurements as packed integer (device_type_code, device_pos, meas_type_code) keys."""
        if hasattr(self, "_active_measurement_code_pos_cache"):
            self._active_measurement_keys = self._active_measurement_code_pos_cache
            self._active_measurement_key_cache = self._active_measurement_code_pos_cache
            return self._active_measurement_code_pos_cache
        if not hasattr(self, "_active_measurement_keys"):
            self._refresh_measurement_summary_cache()
        self._active_measurement_key_cache = self._active_measurement_keys
        return self._active_measurement_keys

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
        """Return active measurement keys sorted for repeated vector membership probes."""
        sorted_key_array = getattr(self, "_active_measurement_key_sorted_array_cache", None)
        key_array = self._active_measurement_key_array_ref()
        cache_key = (id(key_array), int(np.asarray(key_array).size))
        if (
            isinstance(sorted_key_array, tuple)
            and len(sorted_key_array) == 2
            and sorted_key_array[0] == cache_key
        ):
            return sorted_key_array[1]
        keys = np.asarray(key_array, dtype=np.int64).reshape(-1)
        sorted_keys = np.sort(keys) if keys.size else np.empty(0, dtype=np.int64)
        self._active_measurement_key_sorted_array_cache = (cache_key, sorted_keys)
        return sorted_keys

    def _active_measurement_key_membership(self, keys: np.ndarray) -> np.ndarray:
        """Vectorized membership against active packed measurement keys without np.isin uniquing."""
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
            return MeasurementTable(
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
                device_name_id=empty_i64,
                meas_type_code=empty_i16,
                device_pos=empty_i64,
            )

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
            raise ValueError("pseudo measurement device_type_code is required; string fallback is disabled")
        else:
            device_type_code = np.asarray(device_type_codes, dtype=np.int16)
            if device_type_code.size != row_count:
                raise ValueError("pseudo measurement device_type_code array size does not match row count")
        if meas_type_codes is None:
            raise ValueError("pseudo measurement meas_type_code is required; string fallback is disabled")
        else:
            meas_type_code = np.asarray(meas_type_codes, dtype=np.int16)
            if meas_type_code.size != row_count:
                raise ValueError("pseudo measurement meas_type_code array size does not match row count")
        if device_positions is None:
            device_pos = np.full(row_count, -1, dtype=np.int64)
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
        return MeasurementTable(
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
            device_name_id=np.full(row_count, -1, dtype=np.int64),
            meas_type_code=meas_type_code,
            device_pos=device_pos,
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
        row_count = max(
            len(values),
            len(names),
            0 if device_type_codes is None else len(device_type_codes),
            0 if meas_type_codes is None else len(meas_type_codes),
            0 if device_positions is None else len(device_positions),
        )
        if row_count == 0:
            return int(next_idx)
        appended_table = self._pseudo_measurement_table(
            names,
            device_types,
            device_names,
            meas_types,
            values,
            self.pseudo_measurement_weight if weights is None else weights,
            idx_start=int(next_idx),
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
            return _measurement_table_from_array_view(self.measurements)
        base_table = getattr(self.measurements, "table", None)
        if base_table is None:
            warnings.warn(
                "DC SE pseudo measurement append requires a PPC-backed measurement table; skipped table append.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _measurement_table_from_array_view(self.measurements)
        base_table = _measurement_table_from_array_view(self.measurements)
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
                    "DC SE pseudo measurement append found "
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
        self._max_measurement_idx = int(np.max(appended_table.idx))
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._weak_direction_candidate_jacobian_cache = None
        self._active_measurement_plan_tables_cache = None
        self._active_measurement_plan = None
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
                measurement_keys = self._active_measurement_key_array(
                    tail_type[valid_tail],
                    tail_pos[valid_tail],
                    tail_meas[valid_tail],
                )
                self._append_active_measurement_key_array_cache(measurement_keys)
        return base_table

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        node_voltage_best: Dict[int, float] = {}
        real_voltage_best: Dict[int, float] = {}
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            max_idx = int(table.idx.max()) if table.idx.size else 0
            active = np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_type_code = self._ensure_table_meas_type_codes(table)
            device_pos = self._measurement_device_pos_array(table)
            status_code = measurement_table_status_code(table)
            active_key_array = self._active_key_array_from_arrays(
                device_type_code,
                device_pos,
                meas_type_code,
                active,
            )
            from_pos, to_pos = self._measurement_voltage_node_positions_from_codes(
                device_type_code,
                device_pos,
                meas_type_code,
            )
            node_voltage_best, real_voltage_best = self._voltage_best_from_arrays(
                table,
                active & (status_code != MEAS_STATUS_PSEUDO),
                device_type_code,
                meas_type_code,
                from_pos,
                to_pos,
            )
        else:
            max_idx = 0
            warnings.warn(
                "DC SE measurement summary requires a MeasurementTable with device_pos; object fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        if table is not None and len(table.idx) == len(self.measurements):
            self._set_active_key_caches_from_array(active_key_array)
        else:
            self._set_active_key_caches(set())
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = node_voltage_best
        self._real_voltage_observation_node_cache = real_voltage_best
        if table is None or len(table.idx) != len(self.measurements):
            self._node_voltage_measurement_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
            self._real_voltage_observation_pos_mask_cache = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=bool)
            self._real_voltage_observed_solver_pos_mask_cache = None

    def _topology_voltage_pseudo_seed(self, i_node: int, j_node: int, i_pos: int) -> float:
        """Pick a voltage seed for DC zero-impedance topology pseudo measurements."""
        for node_idx in (i_node, j_node):
            measured = self._real_voltage_observation_value_for_node(node_idx)
            if measured is not None:
                return max(float(measured), self.voltage_floor)
        if 0 <= int(i_pos) < self._node_voltage_by_pos.size:
            return max(float(self._node_voltage_by_pos[int(i_pos)]), self.voltage_floor)
        return 1.0

    def _real_voltage_observed_solver_pos_mask(self) -> np.ndarray:
        """Return solver-node positions covered by real voltage measurements."""
        observed_pos_mask = getattr(self, "_real_voltage_observation_pos_mask_cache", None)
        cache_key = (
            id(observed_pos_mask),
            int(self.n_nodes),
            id(getattr(self, "zero_tie_component_by_pos", None)),
            id(getattr(self, "zero_tie_component_offsets", None)),
        )
        cache = getattr(self, "_real_voltage_observed_solver_pos_mask_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == cache_key:
            return cache[1]
        if isinstance(observed_pos_mask, np.ndarray) and int(observed_pos_mask.size) == int(self.n_nodes):
            covered_voltage_pos = np.asarray(observed_pos_mask, dtype=bool).copy()
        else:
            observed_voltage = self._real_voltage_observation_nodes()
            covered_voltage_pos = np.zeros(int(self.n_nodes), dtype=bool)
            if observed_voltage:
                observed_node_idx = np.fromiter(
                    (int(node_idx) for node_idx in observed_voltage.keys()),
                    dtype=np.int64,
                    count=len(observed_voltage),
                )
                lookup_ids = np.asarray(getattr(self, "_node_idx_lookup_ids", ()), dtype=np.int64)
                lookup_pos = np.asarray(getattr(self, "_node_idx_lookup_pos", ()), dtype=np.int64)
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
            if component_count:
                component_observed = np.zeros(component_count, dtype=bool)
                observed_pos = np.flatnonzero(covered_voltage_pos).astype(np.int64, copy=False)
                if observed_pos.size:
                    component_observed[component_by_pos[observed_pos.astype(np.intp, copy=False)].astype(np.intp, copy=False)] = True
                    covered_voltage_pos = component_observed[component_by_pos.astype(np.intp, copy=False)]
        self._real_voltage_observed_solver_pos_mask_cache = (cache_key, covered_voltage_pos)
        return covered_voltage_pos

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak P/V priors for unmeasured DC topology-device states."""
        topology_weight = float(self.pseudo_measurement_weight) * 1e-4
        store_strings = False
        capacity = 2 * (self._zero_branch_names.size + self._break_names.size)
        pseudo_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_meas_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_device_positions = np.empty(capacity, dtype=np.int64)
        pseudo_meas_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_values = np.empty(capacity, dtype=np.float64)
        pseudo_weights = np.empty(capacity, dtype=np.float64)
        pseudo_count = 0
        added_key_chunks = []
        covered_voltage_pos = self._real_voltage_observed_solver_pos_mask()

        def append_rows(device_type_code: int, device_pos: np.ndarray, meas_type_code: int, values: np.ndarray) -> None:
            nonlocal pseudo_count
            count = int(device_pos.size)
            if count == 0:
                return
            end = pseudo_count + count
            pseudo_device_type_codes[pseudo_count:end] = int(device_type_code)
            pseudo_device_positions[pseudo_count:end] = device_pos.astype(np.int64, copy=False)
            pseudo_meas_type_codes[pseudo_count:end] = int(meas_type_code)
            pseudo_values[pseudo_count:end] = values.astype(np.float64, copy=False)
            pseudo_weights[pseudo_count:end] = topology_weight
            if store_strings:
                pseudo_names[pseudo_count:end] = ""
                pseudo_device_types[pseudo_count:end] = ""
                pseudo_device_names[pseudo_count:end] = ""
                pseudo_meas_types[pseudo_count:end] = ""
            added_key_chunks.append(
                self._active_measurement_key_array_for_type(
                    int(device_type_code),
                    device_pos.astype(np.int64, copy=False),
                    int(meas_type_code),
                )
            )
            pseudo_count = end

        for device_type_code, device_type, names, i_node, j_node, i_pos, p_values in (
            (
                DEVICE_TYPE_DCZeroBranch,
                "DCZeroBranch",
                self._zero_branch_names,
                self._zero_branch_i_node,
                self._zero_branch_j_node,
                self._zero_branch_i_pos,
                self._dc_ppc["zero_branch"][self._zero_branch_rows.astype(np.intp, copy=False), DC_ZERO_BRANCH_COLS["p"]]
                if self._zero_branch_rows.size and "p" in DC_ZERO_BRANCH_COLS
                else np.zeros(self._zero_branch_names.size, dtype=np.float64),
            ),
            (
                DEVICE_TYPE_DCBreak,
                "DCBreak",
                self._break_names,
                self._break_i_node,
                self._break_j_node,
                self._break_i_pos,
                self._dc_ppc["break"][self._break_rows.astype(np.intp, copy=False), DC_BREAK_COLS["p"]]
                if self._break_rows.size and "p" in DC_BREAK_COLS
                else np.zeros(self._break_names.size, dtype=np.float64),
            ),
        ):
            p_values = np.asarray(p_values, dtype=np.float64)
            device_count = int(names.size)
            if device_count == 0:
                continue
            device_pos = np.arange(device_count, dtype=np.int64)
            terminal_measured = np.zeros(device_count, dtype=bool)
            for meas_code in (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO):
                keys = self._active_measurement_key_array_for_type(
                    int(device_type_code),
                    device_pos,
                    int(meas_code),
                )
                terminal_measured |= self._active_measurement_key_membership(keys)
            p_rows = device_pos[~terminal_measured]
            if p_rows.size:
                append_rows(int(device_type_code), p_rows, MEAS_TYPE_P_FROM, p_values[p_rows.astype(np.intp, copy=False)])

            i_pos_values = np.asarray(i_pos, dtype=np.int64)
            valid_i = (i_pos_values >= 0) & (i_pos_values < covered_voltage_pos.size)
            voltage_missing = np.ones(device_count, dtype=bool)
            if np.any(valid_i):
                voltage_missing[valid_i] = ~covered_voltage_pos[i_pos_values[valid_i].astype(np.intp, copy=False)]
            v_rows = device_pos[valid_i & voltage_missing]
            if v_rows.size:
                seed_pos = i_pos_values[v_rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)
                seed_values = np.maximum(self._node_voltage_by_pos[seed_pos], self.voltage_floor)
                append_rows(int(device_type_code), v_rows, MEAS_TYPE_V_FROM, seed_values)
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_meas_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_values[:pseudo_count],
            weights=pseudo_weights[:pseudo_count],
            device_type_codes=pseudo_device_type_codes[:pseudo_count],
            meas_type_codes=pseudo_meas_type_codes[:pseudo_count],
            device_positions=pseudo_device_positions[:pseudo_count],
        )
        added_keys = (
            set(np.concatenate(added_key_chunks).astype(object, copy=False))
            if added_key_chunks
            else set()
        )
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        next_idx = self._next_measurement_idx()
        next_idx, topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        del topology_added_keys
        store_strings = False
        capacity = 2 * int(self._generator_names.size) + 2 * int(self._load_names.size) + 4 * int(self._dcdc_names.size)
        pseudo_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_meas_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_device_positions = np.empty(capacity, dtype=np.int64)
        pseudo_meas_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_values = np.empty(capacity, dtype=np.float64)
        pseudo_count = 0

        def missing_rows(device_type_code: int, device_pos: np.ndarray, meas_type_code: int) -> np.ndarray:
            device_pos = np.asarray(device_pos, dtype=np.int64)
            if device_pos.size == 0:
                return np.empty(0, dtype=np.int64)
            keys = self._active_measurement_key_array_for_type(
                int(device_type_code),
                device_pos,
                int(meas_type_code),
            )
            return device_pos[~self._active_measurement_key_membership(keys)]

        def append_rows(device_type_code: int, device_pos: np.ndarray, meas_type_code: int, values: np.ndarray) -> None:
            nonlocal pseudo_count
            count = int(device_pos.size)
            if count == 0:
                return
            end = pseudo_count + count
            pseudo_device_type_codes[pseudo_count:end] = int(device_type_code)
            pseudo_device_positions[pseudo_count:end] = device_pos.astype(np.int64, copy=False)
            pseudo_meas_type_codes[pseudo_count:end] = int(meas_type_code)
            pseudo_values[pseudo_count:end] = values.astype(np.float64, copy=False)
            if store_strings:
                pseudo_names[pseudo_count:end] = ""
                pseudo_device_types[pseudo_count:end] = ""
                pseudo_device_names[pseudo_count:end] = ""
                pseudo_meas_types[pseudo_count:end] = ""
            pseudo_count = end

        covered_voltage_pos = self._real_voltage_observed_solver_pos_mask()

        gen_rows = self._generator_rows.astype(np.intp, copy=False)
        gen_table = self._gen_table
        gen_pos = np.arange(int(self._generator_names.size), dtype=np.int64)
        if gen_pos.size:
            gen_node_pos = self._generator_pos.astype(np.int64, copy=False)
            gen_p_values = gen_table[gen_rows, DC_GEN_COLS["p"]].astype(np.float64, copy=True)
            zero_p = gen_p_values == 0.0
            if np.any(zero_p):
                ctrl = gen_table[gen_rows, DC_GEN_COLS["control_type"]].astype(np.int64, copy=False)
                valid_v = (gen_node_pos >= 0) & (gen_node_pos < self._node_voltage_by_pos.size)
                current_ctrl = zero_p & (ctrl == DC_CTRL_I) & valid_v
                if np.any(current_ctrl):
                    gen_p_values[current_ctrl] = (
                        gen_table[gen_rows[current_ctrl], DC_GEN_COLS["i_set"]]
                        * self._node_voltage_by_pos[gen_node_pos[current_ctrl].astype(np.intp, copy=False)]
                    )
                set_ctrl = zero_p & ~current_ctrl
                if np.any(set_ctrl):
                    gen_p_values[set_ctrl] = gen_table[gen_rows[set_ctrl], DC_GEN_COLS["p_set"]]
            rows = missing_rows(DEVICE_TYPE_DCGenerator, gen_pos, MEAS_TYPE_P_GEN)
            append_rows(DEVICE_TYPE_DCGenerator, rows, MEAS_TYPE_P_GEN, gen_p_values[rows.astype(np.intp, copy=False)])
            valid_v = (gen_node_pos >= 0) & (gen_node_pos < covered_voltage_pos.size)
            rows = missing_rows(DEVICE_TYPE_DCGenerator, gen_pos, MEAS_TYPE_V_GEN)
            rows = rows[valid_v[rows.astype(np.intp, copy=False)] & ~covered_voltage_pos[gen_node_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)]]
            if rows.size:
                row_pos = gen_node_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)
                append_rows(DEVICE_TYPE_DCGenerator, rows, MEAS_TYPE_V_GEN, self._node_voltage_by_pos[row_pos])

        load_rows = self._load_rows.astype(np.intp, copy=False)
        load_table = self._load_table
        load_pos = np.arange(int(self._load_names.size), dtype=np.int64)
        if load_pos.size:
            load_node_pos = self._load_pos.astype(np.int64, copy=False)
            valid_v = (load_node_pos >= 0) & (load_node_pos < self._node_voltage_by_pos.size)
            load_voltage = np.ones(load_pos.size, dtype=np.float64)
            if np.any(valid_v):
                load_voltage[valid_v] = self._node_voltage_by_pos[load_node_pos[valid_v].astype(np.intp, copy=False)]
            load_p_values = load_table[load_rows, DC_LOAD_COLS["p"]].astype(np.float64, copy=True)
            zero_p = load_p_values == 0.0
            if np.any(zero_p):
                load_p_values[zero_p] = load_table[load_rows[zero_p], DC_LOAD_COLS["pbase"]] * (
                    load_table[load_rows[zero_p], DC_LOAD_COLS["pv0"]]
                    + load_table[load_rows[zero_p], DC_LOAD_COLS["pv1"]] * load_voltage[zero_p]
                    + load_table[load_rows[zero_p], DC_LOAD_COLS["pv2"]] * load_voltage[zero_p] * load_voltage[zero_p]
                )
            rows = missing_rows(DEVICE_TYPE_DCLoad, load_pos, MEAS_TYPE_P_LOAD)
            append_rows(DEVICE_TYPE_DCLoad, rows, MEAS_TYPE_P_LOAD, load_p_values[rows.astype(np.intp, copy=False)])
            rows = missing_rows(DEVICE_TYPE_DCLoad, load_pos, MEAS_TYPE_V_LOAD)
            rows = rows[valid_v[rows.astype(np.intp, copy=False)] & ~covered_voltage_pos[load_node_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)]]
            append_rows(DEVICE_TYPE_DCLoad, rows, MEAS_TYPE_V_LOAD, load_voltage[rows.astype(np.intp, copy=False)])

        dcdc_table = self._dc_ppc["dcdc"]
        dcdc_rows = self._dcdc_rows.astype(np.intp, copy=False)
        dcdc_pos = np.arange(int(self._dcdc_names.size), dtype=np.int64)
        if dcdc_pos.size:
            rows = missing_rows(DEVICE_TYPE_DCDCConverter, dcdc_pos, MEAS_TYPE_P_FROM)
            append_rows(DEVICE_TYPE_DCDCConverter, rows, MEAS_TYPE_P_FROM, dcdc_table[dcdc_rows[rows.astype(np.intp, copy=False)], DC_DCDC_COLS["i_p"]])
            rows = missing_rows(DEVICE_TYPE_DCDCConverter, dcdc_pos, MEAS_TYPE_P_TO)
            append_rows(DEVICE_TYPE_DCDCConverter, rows, MEAS_TYPE_P_TO, dcdc_table[dcdc_rows[rows.astype(np.intp, copy=False)], DC_DCDC_COLS["j_p"]])
            dcdc_i_pos = self._dcdc_i_pos.astype(np.int64, copy=False)
            dcdc_j_pos = self._dcdc_j_pos.astype(np.int64, copy=False)
            valid_i = (dcdc_i_pos >= 0) & (dcdc_i_pos < covered_voltage_pos.size)
            valid_j = (dcdc_j_pos >= 0) & (dcdc_j_pos < covered_voltage_pos.size)
            rows = missing_rows(DEVICE_TYPE_DCDCConverter, dcdc_pos, MEAS_TYPE_V_FROM)
            rows = rows[valid_i[rows.astype(np.intp, copy=False)] & ~covered_voltage_pos[dcdc_i_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)]]
            if rows.size:
                append_rows(
                    DEVICE_TYPE_DCDCConverter,
                    rows,
                    MEAS_TYPE_V_FROM,
                    self._node_voltage_by_pos[dcdc_i_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)],
                )
            rows = missing_rows(DEVICE_TYPE_DCDCConverter, dcdc_pos, MEAS_TYPE_V_TO)
            rows = rows[valid_j[rows.astype(np.intp, copy=False)] & ~covered_voltage_pos[dcdc_j_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)]]
            if rows.size:
                append_rows(
                    DEVICE_TYPE_DCDCConverter,
                    rows,
                    MEAS_TYPE_V_TO,
                    self._node_voltage_by_pos[dcdc_j_pos[rows.astype(np.intp, copy=False)].astype(np.intp, copy=False)],
                )
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_meas_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_values[:pseudo_count],
            device_type_codes=pseudo_device_type_codes[:pseudo_count],
            meas_type_codes=pseudo_meas_type_codes[:pseudo_count],
            device_positions=pseudo_device_positions[:pseudo_count],
        )

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch DC rank deficiencies and optional post-observability redundancy."""
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
            base_table_before = _measurement_table_from_array_view(self.measurements)
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
                current_table = _measurement_table_from_array_view(self.measurements)
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
        base_table_before = _measurement_table_from_array_view(self.measurements)
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

    def _add_redundant_observability_pseudo_measurements(self, max_add: int, refresh: bool = True) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add, refresh)

    def _observability_pseudo_candidate_measurements(self) -> MeasurementTableView:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        existing_keys = self._active_measurement_keys_ref()
        candidate_keys = set()
        store_strings = False
        capacity = (
            int(self._raw_node_names_alive.size)
            + 2 * int(self._load_names.size)
            + 2 * int(self._generator_names.size)
            + 2 * int(self._branch_names.size)
            + 2 * int(self._dcdc_names.size)
        )
        names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        device_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        device_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        meas_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        device_type_codes = np.empty(capacity, dtype=np.int16)
        device_positions = np.empty(capacity, dtype=np.int64)
        meas_type_codes = np.empty(capacity, dtype=np.int16)
        values = np.empty(capacity, dtype=np.float64)
        count = 0

        def add(
            device_type_code: int,
            device_pos: int,
            meas_type_code: int,
            value: float,
        ) -> None:
            nonlocal count
            device_type_code = int(device_type_code)
            device_pos = int(device_pos)
            meas_type_code = int(meas_type_code)
            if device_type_code <= 0 or device_pos < 0 or meas_type_code <= 0:
                return
            key = self._active_measurement_key(device_type_code, device_pos, meas_type_code)
            if (
                _is_voltage_measurement_type_code(meas_type_code)
                and self._voltage_pseudo_is_covered_by_pos(device_type_code, device_pos, meas_type_code)
            ):
                return
            if key in existing_keys or key in candidate_keys:
                return
            row = count
            if store_strings:
                names[row] = ""
                device_types[row] = ""
                device_names[row] = ""
                meas_types[row] = ""
            device_type_codes[row] = device_type_code
            device_positions[row] = device_pos
            meas_type_codes[row] = meas_type_code
            values[row] = float(value)
            count = row + 1
            candidate_keys.add(key)

        for device_pos in range(int(self._raw_node_names_alive.size)):
            node_pos = self._raw_node_solver_pos_alive[device_pos]
            node_pos = int(node_pos)
            voltage = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            add(DEVICE_TYPE_DCNode, int(device_pos), MEAS_TYPE_V, voltage)

        load_rows = self._load_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._load_names.size)):
            node_pos = int(self._load_pos[device_pos])
            voltage = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            row = int(load_rows[device_pos])
            table_row = self._load_table[row]
            p_value = float(table_row[DC_LOAD_COLS["p"]])
            if p_value == 0.0:
                p_value = float(table_row[DC_LOAD_COLS["pbase"]]) * (
                    float(table_row[DC_LOAD_COLS["pv0"]])
                    + float(table_row[DC_LOAD_COLS["pv1"]]) * voltage
                    + float(table_row[DC_LOAD_COLS["pv2"]]) * voltage * voltage
                )
            add(DEVICE_TYPE_DCLoad, device_pos, MEAS_TYPE_P_LOAD, p_value)
            add(DEVICE_TYPE_DCLoad, device_pos, MEAS_TYPE_V_LOAD, voltage)

        gen_rows = self._generator_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._generator_names.size)):
            node_pos = int(self._generator_pos[device_pos])
            voltage = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            row = int(gen_rows[device_pos])
            table_row = self._gen_table[row]
            p_value = float(table_row[DC_GEN_COLS["p"]])
            if p_value == 0.0:
                ctrl = int(table_row[DC_GEN_COLS["control_type"]])
                p_value = (
                    float(table_row[DC_GEN_COLS["i_set"]]) * voltage
                    if ctrl == DC_CTRL_I
                    else float(table_row[DC_GEN_COLS["p_set"]])
                )
            add(DEVICE_TYPE_DCGenerator, device_pos, MEAS_TYPE_P_GEN, p_value)
            add(DEVICE_TYPE_DCGenerator, device_pos, MEAS_TYPE_V_GEN, voltage)

        branch_table = np.asarray(self._dc_ppc.get("branch", np.zeros((0, len(DC_BRANCH_COLS)))), dtype=np.float64)
        branch_rows = self._branch_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._branch_names.size)):
            row = int(branch_rows[device_pos])
            if 0 <= row < branch_table.shape[0]:
                add(DEVICE_TYPE_DCBranch, device_pos, MEAS_TYPE_P_FROM, float(branch_table[row, DC_BRANCH_COLS["i_p"]]))
                add(DEVICE_TYPE_DCBranch, device_pos, MEAS_TYPE_P_TO, float(branch_table[row, DC_BRANCH_COLS["j_p"]]))

        dcdc_table = np.asarray(self._dc_ppc.get("dcdc", np.zeros((0, len(DC_DCDC_COLS)))), dtype=np.float64)
        dcdc_rows = self._dcdc_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._dcdc_names.size)):
            row = int(dcdc_rows[device_pos])
            if 0 <= row < dcdc_table.shape[0]:
                add(DEVICE_TYPE_DCDCConverter, device_pos, MEAS_TYPE_P_FROM, float(dcdc_table[row, DC_DCDC_COLS["i_p"]]))
                add(DEVICE_TYPE_DCDCConverter, device_pos, MEAS_TYPE_P_TO, float(dcdc_table[row, DC_DCDC_COLS["j_p"]]))

        candidate_table = self._pseudo_measurement_table(
            names[:count] if store_strings else np.asarray([], dtype=object),
            device_types[:count] if store_strings else np.asarray([], dtype=object),
            device_names[:count] if store_strings else np.asarray([], dtype=object),
            meas_types[:count] if store_strings else np.asarray([], dtype=object),
            values[:count],
            self.pseudo_measurement_weight,
            device_type_codes=device_type_codes[:count],
            meas_type_codes=meas_type_codes[:count],
            device_positions=device_positions[:count],
        )
        return MeasurementTableView(candidate_table, normalized=True)

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: object,
        max_add: int,
    ) -> np.ndarray:
        if max_add <= 0 or not candidates:
            return np.asarray([], dtype=np.int64)
        candidate_count = int(len(candidates))
        x = self.initial_state()
        active_plan_tables = self._active_measurement_plan_tables_ref()
        cache = self._observability_matrix_cache_for(observability, active_plan_tables, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, active_plan_tables)
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
        """Build low-weight candidates from invalid real device rows."""
        table = _measurement_table_from_array_view(self.measurements)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = self._ensure_table_meas_type_codes(table)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is None or np.asarray(device_pos).size != int(table.idx.size):
            warnings.warn(
                "DC SE rank-restoring candidates require measurement device_pos; "
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
        candidate_mask &= ~(
            (device_type_code[invalid_rows] == DEVICE_TYPE_DCNode)
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
        store_strings = False
        if store_strings:
            names = np.asarray(table.name, dtype=object)
            names = names[rows] if names.size == int(table.idx.size) else np.asarray([f"rank_{int(table.idx[row])}" for row in rows], dtype=object)
            names = np.asarray([f"pseudo_rank_{name}" for name in names], dtype=object)
            device_types = np.asarray(table.device_type, dtype=object)
            device_types = device_types[rows] if device_types.size == int(table.idx.size) else np.asarray([""] * rows.size, dtype=object)
            device_names = np.asarray(table.device_name, dtype=object)
            device_names = device_names[rows] if device_names.size == int(table.idx.size) else np.asarray([""] * rows.size, dtype=object)
            meas_types = np.asarray(table.meas_type, dtype=object)
            meas_types = meas_types[rows] if meas_types.size == int(table.idx.size) else np.asarray([""] * rows.size, dtype=object)
        else:
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

    def _rank_restoring_candidate_indices(self, candidates: object, max_add: int) -> np.ndarray:
        """Select candidate rows that participate in a higher structural-rank matching."""
        if max_add <= 0 or not candidates:
            return np.asarray([], dtype=np.int64)
        base_measurements = self.active_measurements
        x = self.initial_state()
        base_h = self.jacobian_sparse(x, base_measurements)
        base_rank = sparse_structural_rank(base_h)
        if base_rank is None or base_rank >= self.n_state:
            return np.asarray([], dtype=np.int64)

        base_table = _measurement_table_from_array_view(base_measurements)
        candidate_table = _measurement_table_from_array_view(candidates)
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
        if selected.size == 0:
            return np.asarray([], dtype=np.int64)
        selected = np.unique(selected)
        return selected[:max_add].astype(np.int64, copy=False)

    def _add_structural_rank_restoring_pseudo_measurements(self, max_add: int) -> int:
        """Add invalid real-measurement candidates that improve structural observability."""
        candidates = self._rank_restoring_candidate_measurements()
        selected_indices = self._rank_restoring_candidate_indices(candidates, max_add)
        if selected_indices.size == 0:
            return 0
        next_idx = self._next_measurement_idx()
        base_table_before = _measurement_table_from_array_view(self.measurements)
        measurement_count_before = int(base_table_before.idx.size)
        candidate_table = getattr(candidates, "table", None)
        if candidate_table is None:
            return 0
        selected_rows = selected_indices.astype(np.int64, copy=False)
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
        """Translate a weak compact DC state into the smallest useful pseudo measurement."""
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
            if (
                _is_voltage_measurement_type_code(target_meas_type_code)
                and self._voltage_pseudo_is_covered_by_pos(
                    target_device_type_code,
                    target_device_pos,
                    target_meas_type_code,
                )
            ):
                return next_idx, 0
            if key in existing_keys:
                return next_idx, 0
            store_strings = False
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}" if store_strings else ""
            new_idx = self._append_pseudo_measurement_rows(
                next_idx,
                np.asarray([pseudo_name], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([device_type], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([device_name], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([meas_type], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([float(value)], dtype=np.float64),
                device_type_codes=np.asarray([target_device_type_code], dtype=np.int16),
                meas_type_codes=np.asarray([target_meas_type_code], dtype=np.int16),
                device_positions=np.asarray([target_device_pos], dtype=np.int64),
            )
            existing_keys.add(key)
            added_total += 1
            return new_idx, 1

        if meta_kind == "voltage" and device_type_code == DEVICE_TYPE_DCNode:
            node_pos = int(self._node_plan_node_pos[device_pos]) if 0 <= device_pos < self._node_plan_node_pos.size else -1
            value = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            return add(DEVICE_TYPE_DCNode, "DCNode", name, device_pos, MEAS_TYPE_V, "V", value)
        if meta_kind == "zero_current" and device_type_code == DEVICE_TYPE_DCZeroBranch:
            value = 0.0
            if self._zero_branch_rows.size and "current" in DC_ZERO_BRANCH_COLS:
                value = float(self._dc_ppc["zero_branch"][int(self._zero_branch_rows[device_pos]), DC_ZERO_BRANCH_COLS["current"]])
            return add(DEVICE_TYPE_DCZeroBranch, "DCZeroBranch", name, device_pos, MEAS_TYPE_I_FROM, "I_FROM", value)
        if meta_kind == "break_current" and device_type_code == DEVICE_TYPE_DCBreak:
            value = 0.0
            if self._break_rows.size and "current" in DC_BREAK_COLS:
                value = float(self._dc_ppc["break"][int(self._break_rows[device_pos]), DC_BREAK_COLS["current"]])
            return add(DEVICE_TYPE_DCBreak, "DCBreak", name, device_pos, MEAS_TYPE_I_FROM, "I_FROM", value)
        if meta_kind == "dcdc_p_from" and device_type_code == DEVICE_TYPE_DCDCConverter:
            value = float(self._dc_ppc["dcdc"][int(self._dcdc_rows[device_pos]), DC_DCDC_COLS["i_p"]])
            return add(DEVICE_TYPE_DCDCConverter, "DCDCConverter", name, device_pos, MEAS_TYPE_P_FROM, "P_FROM", value)
        if meta_kind == "dcdc_p_to" and device_type_code == DEVICE_TYPE_DCDCConverter:
            value = float(self._dc_ppc["dcdc"][int(self._dcdc_rows[device_pos]), DC_DCDC_COLS["j_p"]])
            return add(DEVICE_TYPE_DCDCConverter, "DCDCConverter", name, device_pos, MEAS_TYPE_P_TO, "P_TO", value)
        if meta_kind == "v_generator_p" and device_type_code == DEVICE_TYPE_DCGenerator:
            value = float(self._gen_table[int(self._generator_rows[device_pos]), DC_GEN_COLS["p_set"]])
            return add(DEVICE_TYPE_DCGenerator, "DCGenerator", name, device_pos, MEAS_TYPE_P_GEN, "P_GEN", value)
        return next_idx, 0

    def initial_state(self) -> np.ndarray:
        seed = self._initial_state_flat if self.flat_start else self._initial_state_nonflat
        return seed.copy()

    def state_layout(self) -> Dict[str, object]:
        """Expose the canonical DC state layout for reuse by hybrid orchestration."""
        return {
            "state_labels": self.state_labels,
            "state_meta": self.state_meta,
            "voltage_col": self.voltage_col,
            "n_state": self.n_state,
            "references": self.references,
            "voltage_state_pos": self.voltage_state_pos,
            "zero_tie_component_by_pos": self.zero_tie_component_by_pos,
            "zero_tie_component_offsets": self.zero_tie_component_offsets,
            "zero_tie_component_indices": self.zero_tie_component_indices,
            "zero_tie_components": self.zero_tie_components,
        }

    def _build_initial_state_seed_arrays(self) -> None:
        flat = np.zeros(self.n_state, dtype=np.float64)
        flat[: self.n_voltage] = 1.0
        if self.n_v_generator:
            rows = self._v_generator_rows.astype(np.intp, copy=False)
            flat[self.v_generator_start :] = self._gen_table[rows, DC_GEN_COLS["p_set"]]

        nonflat = np.zeros(self.n_state, dtype=np.float64)
        if self.n_voltage:
            nonflat[: self.n_voltage] = self._node_voltage_by_pos[self.voltage_state_pos.astype(np.intp, copy=False)]
        if self.n_switch:
            zero_current = np.zeros(self.n_switch, dtype=np.float64)
            if self._zero_branch_rows.size and "current" in DC_ZERO_BRANCH_COLS:
                zero_current[: self._zero_branch_rows.size] = self._dc_ppc["zero_branch"][
                    self._zero_branch_rows.astype(np.intp, copy=False),
                    DC_ZERO_BRANCH_COLS["current"],
                ]
            if self._break_rows.size and "current" in DC_BREAK_COLS:
                offset = self._zero_branch_rows.size
                zero_current[offset : offset + self._break_rows.size] = self._dc_ppc["break"][
                    self._break_rows.astype(np.intp, copy=False),
                    DC_BREAK_COLS["current"],
                ]
            nonflat[self.switch_start : self.dcdc_start] = zero_current
        if self.n_dcdc_power:
            rows = self._dcdc_rows.astype(np.intp, copy=False)
            nonflat[self.dcdc_start : self.v_generator_start : 2] = self._dc_ppc["dcdc"][rows, DC_DCDC_COLS["i_p"]]
            nonflat[self.dcdc_start + 1 : self.v_generator_start : 2] = self._dc_ppc["dcdc"][rows, DC_DCDC_COLS["j_p"]]
        if self.n_v_generator:
            rows = self._v_generator_rows.astype(np.intp, copy=False)
            nonflat[self.v_generator_start :] = self._gen_table[rows, DC_GEN_COLS["p"]]
        self._initial_state_flat = flat
        self._initial_state_nonflat = nonflat

    def _unpack_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split WLS state into node voltages, switch currents, converter powers and V-gen powers."""
        state_voltage = np.asarray(x[: self.n_voltage], dtype=np.float64).copy()
        state_voltage[state_voltage < self.voltage_floor] = self.voltage_floor
        voltage = np.ones(int(self.n_nodes), dtype=np.float64)
        if self._voltage_expand_pos.size:
            voltage[self._voltage_expand_pos] = state_voltage[self._voltage_expand_col]
        if self._ref_voltage_pos.size:
            voltage[self._ref_voltage_pos] = self._ref_voltage_value
        switch_current = np.asarray(x[self.switch_start : self.dcdc_start], dtype=np.float64)
        dcdc_power = np.asarray(x[self.dcdc_start : self.v_generator_start], dtype=np.float64)
        v_generator_power = np.asarray(x[self.v_generator_start :], dtype=np.float64)
        return voltage, switch_current, dcdc_power, v_generator_power

    @staticmethod
    def _plan_rows_for_device_code(
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
        device_type_code: int,
    ) -> np.ndarray:
        rows = plan_table.row[
            (np.asarray(plan_table.device_type_code, dtype=np.int16) == int(device_type_code))
            & np.asarray(handled, dtype=bool)
        ]
        return np.asarray(rows, dtype=np.int64)

    def _build_node_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rows = self._plan_rows_for_device_code(plan_table, handled, DEVICE_TYPE_DCNode)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)[rows]
        return {
            "node_rows": self._int_array(rows),
            "node_pos": self._int_array(self._node_plan_node_pos[device_pos]),
            "node_col": self._int_array(self._node_plan_col[device_pos]),
        }

    def _build_branch_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rows = self._plan_rows_for_device_code(plan_table, handled, DEVICE_TYPE_DCBranch)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)[rows]
        return {
            "branch_rows": self._int_array(rows),
            "branch_kind": self._int_array(np.asarray(plan_table.meas_kind, dtype=np.int16)[rows]),
            "branch_i": self._int_array(self._branch_plan_i[device_pos]),
            "branch_j": self._int_array(self._branch_plan_j[device_pos]),
            "branch_i_col": self._int_array(self._branch_plan_i_col[device_pos]),
            "branch_j_col": self._int_array(self._branch_plan_j_col[device_pos]),
            "branch_inv_r": np.asarray(self._branch_plan_inv_r[device_pos], dtype=np.float64),
        }

    def _build_load_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rows = self._plan_rows_for_device_code(plan_table, handled, DEVICE_TYPE_DCLoad)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)[rows]
        return {
            "load_rows": self._int_array(rows),
            "load_kind": self._int_array(np.asarray(plan_table.meas_kind, dtype=np.int16)[rows]),
            "load_pos": self._int_array(self._load_plan_pos[device_pos]),
            "load_col": self._int_array(self._load_plan_col[device_pos]),
            "load_pv0": np.asarray(self._load_plan_pv0[device_pos], dtype=np.float64),
            "load_pv1": np.asarray(self._load_plan_pv1[device_pos], dtype=np.float64),
            "load_pv2": np.asarray(self._load_plan_pv2[device_pos], dtype=np.float64),
        }

    def _build_generator_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rows = self._plan_rows_for_device_code(plan_table, handled, DEVICE_TYPE_DCGenerator)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)[rows]
        return {
            "gen_rows": self._int_array(rows),
            "gen_kind": self._int_array(np.asarray(plan_table.meas_kind, dtype=np.int16)[rows]),
            "gen_ctrl": self._int_array(self._generator_plan_ctrl[device_pos]),
            "gen_pos": self._int_array(self._generator_plan_pos[device_pos]),
            "gen_col": self._int_array(self._generator_plan_col[device_pos]),
            "gen_p_col": self._int_array(self._generator_plan_p_col[device_pos]),
            "gen_vgen_pos": self._int_array(self._generator_plan_vgen_pos[device_pos]),
            "gen_p_set": np.asarray(self._generator_plan_p_set[device_pos], dtype=np.float64),
            "gen_i_set": np.asarray(self._generator_plan_i_set[device_pos], dtype=np.float64),
        }

    def _build_switch_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        row = np.asarray(plan_table.row, dtype=np.int64)
        device_type_code = np.asarray(plan_table.device_type_code, dtype=np.int16)
        meas_kind = np.asarray(plan_table.meas_kind, dtype=np.int16)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)

        def switch_plan_for_code(
            code: int,
            plan_i: np.ndarray,
            plan_j: np.ndarray,
            plan_i_col: np.ndarray,
            plan_j_col: np.ndarray,
            plan_current_col: np.ndarray,
            plan_current_pos: np.ndarray,
        ) -> Tuple[np.ndarray, ...]:
            rows = row[(device_type_code == int(code)) & handled]
            positions = device_pos[rows]
            kind = meas_kind[rows]
            current_pos = plan_current_pos[positions]
            keep = (current_pos >= 0) | (kind == MEAS_TYPE_V_FROM) | (kind == MEAS_TYPE_V_TO)
            if keep.size and not np.all(keep):
                handled[rows[~keep]] = False
            rows = rows[keep]
            positions = positions[keep]
            return (
                rows,
                kind[keep],
                plan_i[positions],
                plan_j[positions],
                plan_i_col[positions],
                plan_j_col[positions],
                plan_current_col[positions],
                current_pos[keep],
            )

        zero_switch = switch_plan_for_code(
            DEVICE_TYPE_DCZeroBranch,
            self._zero_branch_plan_i,
            self._zero_branch_plan_j,
            self._zero_branch_plan_i_col,
            self._zero_branch_plan_j_col,
            self._zero_branch_plan_current_col,
            self._zero_branch_plan_current_pos,
        )
        break_switch = switch_plan_for_code(
            DEVICE_TYPE_DCBreak,
            self._break_plan_i,
            self._break_plan_j,
            self._break_plan_i_col,
            self._break_plan_j_col,
            self._break_plan_current_col,
            self._break_plan_current_pos,
        )
        switch_rows = np.concatenate((zero_switch[0], break_switch[0])).astype(np.int64, copy=False)
        switch_kind = np.concatenate((zero_switch[1], break_switch[1])).astype(np.int64, copy=False)
        switch_i = np.concatenate((zero_switch[2], break_switch[2])).astype(np.int64, copy=False)
        switch_j = np.concatenate((zero_switch[3], break_switch[3])).astype(np.int64, copy=False)
        switch_i_col = np.concatenate((zero_switch[4], break_switch[4])).astype(np.int64, copy=False)
        switch_j_col = np.concatenate((zero_switch[5], break_switch[5])).astype(np.int64, copy=False)
        switch_col = np.concatenate((zero_switch[6], break_switch[6])).astype(np.int64, copy=False)
        switch_pos = np.concatenate((zero_switch[7], break_switch[7])).astype(np.int64, copy=False)
        if switch_rows.size:
            order = np.argsort(switch_rows)
            switch_rows = switch_rows[order]
            switch_kind = switch_kind[order]
            switch_i = switch_i[order]
            switch_j = switch_j[order]
            switch_i_col = switch_i_col[order]
            switch_j_col = switch_j_col[order]
            switch_col = switch_col[order]
            switch_pos = switch_pos[order]
        return {
            "switch_rows": self._int_array(switch_rows),
            "switch_kind": self._int_array(switch_kind),
            "switch_i": self._int_array(switch_i),
            "switch_j": self._int_array(switch_j),
            "switch_i_col": self._int_array(switch_i_col),
            "switch_j_col": self._int_array(switch_j_col),
            "switch_col": self._int_array(switch_col),
            "switch_pos": self._int_array(switch_pos),
        }

    def _build_constraint_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        row = np.asarray(plan_table.row, dtype=np.int64)
        device_type_code = np.asarray(plan_table.device_type_code, dtype=np.int16)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)
        rows = np.concatenate(
            (
                row[(device_type_code == DEVICE_TYPE_DCZeroBranchConstraint) & handled],
                row[(device_type_code == DEVICE_TYPE_DCBreakConstraint) & handled],
            )
        ).astype(np.int64, copy=False)
        if rows.size:
            rows = rows[np.argsort(rows)]
        positions = device_pos[rows]
        return {
            "constraint_rows": self._int_array(rows),
            "constraint_i": self._int_array(self._constraint_plan_i[positions]),
            "constraint_j": self._int_array(self._constraint_plan_j[positions]),
            "constraint_i_col": self._int_array(self._constraint_plan_i_col[positions]),
            "constraint_j_col": self._int_array(self._constraint_plan_j_col[positions]),
        }

    def _build_dcdc_vector_plan_from_table(
        self,
        plan_table: MeasurementPlanTable,
        handled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rows = self._plan_rows_for_device_code(plan_table, handled, DEVICE_TYPE_DCDCConverter)
        device_pos = np.asarray(plan_table.device_pos, dtype=np.int64)[rows]
        return {
            "dcdc_rows": self._int_array(rows),
            "dcdc_kind": self._int_array(np.asarray(plan_table.meas_kind, dtype=np.int16)[rows]),
            "dcdc_i": self._int_array(self._dcdc_plan_i[device_pos]),
            "dcdc_j": self._int_array(self._dcdc_plan_j[device_pos]),
            "dcdc_i_col": self._int_array(self._dcdc_plan_i_col[device_pos]),
            "dcdc_j_col": self._int_array(self._dcdc_plan_j_col[device_pos]),
            "dcdc_p_col": self._int_array(self._dcdc_plan_p_col[device_pos]),
            "dcdc_q_col": self._int_array(self._dcdc_plan_q_col[device_pos]),
            "dcdc_pos": self._int_array(self._dcdc_plan_pos[device_pos]),
        }

    def _measurement_plan(self, measurements_or_plan_tables) -> Dict[str, np.ndarray]:
        if isinstance(measurements_or_plan_tables, dict) or isinstance(measurements_or_plan_tables, MeasurementPlanTable):
            plan_table = self._common_measurement_plan_table(measurements_or_plan_tables)
            cache_ref = plan_table
        else:
            plan_table = None
            cache_ref = measurements_or_plan_tables
        key = id(cache_ref)
        cached = self._measurement_plan_cache.get(key)
        if cached is not None and cached[0] is cache_ref:
            return cached[1]

        if plan_table is None:
            indexed_table = self._measurement_table_for_indexed_plan(measurements_or_plan_tables)
            plan_table = self._active_measurement_plan_tables(indexed_table)["simple"]
        handled = np.asarray(plan_table.handled, dtype=bool).copy()
        plan = {"handled_mask": handled}
        plan.update(self._build_node_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_branch_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_load_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_generator_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_switch_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_constraint_vector_plan_from_table(plan_table, handled))
        plan.update(self._build_dcdc_vector_plan_from_table(plan_table, handled))
        # Precompute per-kind bool masks once. The fill_* helpers branch on
        # `kind == k` for each k in the relevant range; without these caches
        # every iteration re-evaluates the comparison even though `kind` is
        # immutable. Each mask is len(section_rows) so storage is trivial.
        self._populate_kind_masks(plan)
        if len(self._measurement_plan_cache) > 16:
            self._measurement_plan_cache.clear()
        self._measurement_plan_cache[key] = (cache_ref, plan)
        if self._measurement_plan_tables_are_active(measurements_or_plan_tables):
            self._active_measurement_plan = plan
        return plan

    def _fill_measurement_values_vectorized(
        self,
        values: np.ndarray,
        measurements: object,
        voltage: np.ndarray,
        switch_current: np.ndarray,
        dcdc_power: np.ndarray,
        v_generator_power: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate common DC measurement rows without per-device Python calls."""
        plan = measurements if isinstance(measurements, dict) and "handled_mask" in measurements else self._measurement_plan(measurements)

        rows = plan["node_rows"]
        if rows.size:
            values[rows] = voltage[plan["node_pos"]]

        rows = plan["branch_rows"]
        if rows.size:
            kind_masks = plan["branch_kind_masks"]
            i = plan["branch_i"]
            j = plan["branch_j"]
            vi = voltage[i]
            vj = voltage[j]
            inv_r = plan["branch_inv_r"]
            current = (vi - vj) * inv_r
            out = np.empty(rows.size, dtype=np.float64)
            m = kind_masks[MEAS_TYPE_P_FROM]
            if m.any():
                out[m] = vi[m] * current[m]
            m = kind_masks[MEAS_TYPE_V_FROM]
            if m.any():
                out[m] = vi[m]
            m = kind_masks[MEAS_TYPE_I_FROM]
            if m.any():
                out[m] = current[m]
            m = kind_masks[MEAS_TYPE_P_TO]
            if m.any():
                out[m] = -vj[m] * current[m]
            m = kind_masks[MEAS_TYPE_V_TO]
            if m.any():
                out[m] = vj[m]
            m = kind_masks[MEAS_TYPE_I_TO]
            if m.any():
                out[m] = -current[m]
            values[rows] = out

        rows = plan["load_rows"]
        if rows.size:
            kind_masks = plan["load_kind_masks"]
            v = voltage[plan["load_pos"]]
            p = plan["load_pv0"] + plan["load_pv1"] * v + plan["load_pv2"] * v * v
            out = np.empty(rows.size, dtype=np.float64)
            m = kind_masks[MEAS_TYPE_P_LOAD]
            if m.any():
                out[m] = p[m]
            m = kind_masks[MEAS_TYPE_V_LOAD]
            if m.any():
                out[m] = v[m]
            i_mask = kind_masks[MEAS_TYPE_I_LOAD]
            if i_mask.any():
                out[i_mask] = 0.0
                v_i = v[i_mask]
                p_i = p[i_mask]
                valid = np.abs(v_i) > self.min_current_voltage
                if valid.any():
                    idx = np.flatnonzero(i_mask)
                    out[idx[valid]] = p_i[valid] / v_i[valid]
            values[rows] = out

        rows = plan["gen_rows"]
        if rows.size:
            kind_masks = plan["gen_kind_masks"]
            ctrl_masks = plan["gen_ctrl_masks"]
            v = voltage[plan["gen_pos"]]
            v_ctrl = ctrl_masks[DC_CTRL_V]
            p_ctrl = ctrl_masks[DC_CTRL_P]
            i_ctrl = ctrl_masks[DC_CTRL_I]
            p = np.zeros(rows.size, dtype=np.float64)
            if v_ctrl.any():
                p[v_ctrl] = v_generator_power[plan["gen_vgen_pos"][v_ctrl]]
            if p_ctrl.any():
                p[p_ctrl] = plan["gen_p_set"][p_ctrl]
            if i_ctrl.any():
                p[i_ctrl] = plan["gen_i_set"][i_ctrl] * v[i_ctrl]
            current = np.zeros(rows.size, dtype=np.float64)
            if i_ctrl.any():
                current[i_ctrl] = plan["gen_i_set"][i_ctrl]
            derived_current = ~i_ctrl
            if derived_current.any():
                v_d = v[derived_current]
                p_d = p[derived_current]
                valid = np.abs(v_d) > self.min_current_voltage
                if valid.any():
                    idx = np.flatnonzero(derived_current)
                    current[idx[valid]] = p_d[valid] / v_d[valid]
            out = np.empty(rows.size, dtype=np.float64)
            m = kind_masks[MEAS_TYPE_P_GEN]
            if m.any():
                out[m] = p[m]
            m = kind_masks[MEAS_TYPE_V_GEN]
            if m.any():
                out[m] = v[m]
            m = kind_masks[MEAS_TYPE_I_GEN]
            if m.any():
                out[m] = current[m]
            values[rows] = out

        rows = plan["switch_rows"]
        if rows.size:
            kind_masks = plan["switch_kind_masks"]
            vi = voltage[plan["switch_i"]]
            vj = voltage[plan["switch_j"]]
            current = np.zeros(rows.size, dtype=np.float64)
            current_mask = plan["switch_pos"] >= 0
            if current_mask.any():
                current[current_mask] = switch_current[plan["switch_pos"][current_mask]]
            out = np.empty(rows.size, dtype=np.float64)
            m = kind_masks[MEAS_TYPE_P_FROM]
            if m.any():
                out[m] = vi[m] * current[m]
            m = kind_masks[MEAS_TYPE_V_FROM]
            if m.any():
                out[m] = vi[m]
            m = kind_masks[MEAS_TYPE_I_FROM]
            if m.any():
                out[m] = current[m]
            m = kind_masks[MEAS_TYPE_P_TO]
            if m.any():
                out[m] = -vj[m] * current[m]
            m = kind_masks[MEAS_TYPE_V_TO]
            if m.any():
                out[m] = vj[m]
            m = kind_masks[MEAS_TYPE_I_TO]
            if m.any():
                out[m] = -current[m]
            values[rows] = out

        rows = plan["constraint_rows"]
        if rows.size:
            values[rows] = voltage[plan["constraint_i"]] - voltage[plan["constraint_j"]]

        rows = plan["dcdc_rows"]
        if rows.size:
            kind_masks = plan["dcdc_kind_masks"]
            conv_pos = plan["dcdc_pos"]
            p_from = dcdc_power[2 * conv_pos]
            p_to = dcdc_power[2 * conv_pos + 1]
            v_from = voltage[plan["dcdc_i"]]
            v_to = voltage[plan["dcdc_j"]]
            out = np.empty(rows.size, dtype=np.float64)
            m = kind_masks[MEAS_TYPE_P_FROM]
            if m.any():
                out[m] = p_from[m]
            m = kind_masks[MEAS_TYPE_V_FROM]
            if m.any():
                out[m] = v_from[m]
            m = kind_masks[MEAS_TYPE_I_FROM]
            if m.any():
                v_f = v_from[m]
                valid = np.abs(v_f) > self.min_current_voltage
                idx = np.flatnonzero(m)
                tmp = np.zeros(idx.size, dtype=np.float64)
                if valid.any():
                    tmp[valid] = p_from[m][valid] / v_f[valid]
                out[idx] = tmp
            m = kind_masks[MEAS_TYPE_P_TO]
            if m.any():
                out[m] = p_to[m]
            m = kind_masks[MEAS_TYPE_V_TO]
            if m.any():
                out[m] = v_to[m]
            m = kind_masks[MEAS_TYPE_I_TO]
            if m.any():
                v_t = v_to[m]
                valid = np.abs(v_t) > self.min_current_voltage
                idx = np.flatnonzero(m)
                tmp = np.zeros(idx.size, dtype=np.float64)
                if valid.any():
                    tmp[valid] = p_to[m][valid] / v_t[valid]
                out[idx] = tmp
            values[rows] = out

        return plan["handled_mask"]

    def evaluate(
        self,
        x: np.ndarray,
        measurement_plan_tables=None,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate h(x): estimated values for each active DC measurement."""
        measurement_plan_tables = self._measurement_plan_tables_for(measurement_plan_tables)
        vector_plans = self._vector_plans_for_measurement_plan_tables(measurement_plan_tables)
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
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
        vectorized_rows = self._fill_measurement_values_vectorized(
            values,
            vector_plans["simple"],
            voltage,
            switch_current,
            dcdc_power,
            v_generator_power,
        )
        if active_vectorized:
            return values
        if not np.all(vectorized_rows):
            missing = int(vectorized_rows.size - np.count_nonzero(vectorized_rows))
            warnings.warn(
                f"DC SE evaluate skipped {missing} non-vectorized measurement rows; object/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        return values

    def _add_indexed_values(
        self,
        H,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        # SparseJacobianBuilder.add_many already handles dtype coercion and
        # cols >= 0 filtering; passing rows/cols straight through avoids the
        # int64/int32 round-trip that used to dominate this function's cost.
        if hasattr(H, "add_many"):
            if mask is None:
                H.add_many(rows, cols, values)
            else:
                H.add_many(rows, cols, values, mask)
            return
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = cols >= 0
        else:
            mask = np.asarray(mask, dtype=bool) & (cols >= 0)
        if np.any(mask):
            np.add.at(H, (rows[mask], cols[mask]), values[mask])

    def _fill_jacobian_vectorized(
        self,
        H,
        measurements: object,
        voltage: np.ndarray,
        switch_current: np.ndarray,
        dcdc_power: np.ndarray,
        v_generator_power: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse/dense DC measurement Jacobian rows."""
        plan = measurements if isinstance(measurements, dict) and "handled_mask" in measurements else self._measurement_plan(measurements)

        rows = plan["node_rows"]
        if rows.size:
            self._add_indexed_values(H, rows, plan["node_col"], np.ones(rows.size, dtype=np.float64))

        rows = plan["branch_rows"]
        if rows.size:
            kind_masks = plan["branch_kind_masks"]
            i = plan["branch_i"]
            j = plan["branch_j"]
            i_col = plan["branch_i_col"]
            j_col = plan["branch_j_col"]
            vi = voltage[i]
            vj = voltage[j]
            inv_r = plan["branch_inv_r"]

            mask = kind_masks[MEAS_TYPE_P_FROM]
            self._add_indexed_values(H, rows, i_col, (2.0 * vi - vj) * inv_r, mask)
            self._add_indexed_values(H, rows, j_col, -vi * inv_r, mask)
            mask = kind_masks[MEAS_TYPE_V_FROM]
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind_masks[MEAS_TYPE_I_FROM]
            self._add_indexed_values(H, rows, i_col, inv_r, mask)
            self._add_indexed_values(H, rows, j_col, -inv_r, mask)
            mask = kind_masks[MEAS_TYPE_P_TO]
            self._add_indexed_values(H, rows, i_col, -vj * inv_r, mask)
            self._add_indexed_values(H, rows, j_col, (-vi + 2.0 * vj) * inv_r, mask)
            mask = kind_masks[MEAS_TYPE_V_TO]
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind_masks[MEAS_TYPE_I_TO]
            self._add_indexed_values(H, rows, i_col, -inv_r, mask)
            self._add_indexed_values(H, rows, j_col, inv_r, mask)

        rows = plan["load_rows"]
        if rows.size:
            kind_masks = plan["load_kind_masks"]
            pos = plan["load_pos"]
            col = plan["load_col"]
            v = voltage[pos]
            self._add_indexed_values(
                H, rows, col, plan["load_pv1"] + 2.0 * plan["load_pv2"] * v, kind_masks[MEAS_TYPE_P_LOAD]
            )
            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_V_LOAD])
            self._add_indexed_values(
                H, rows, col, plan["load_pv2"] - plan["load_pv0"] / (v * v), kind_masks[MEAS_TYPE_I_LOAD]
            )

        rows = plan["gen_rows"]
        if rows.size:
            kind_masks = plan["gen_kind_masks"]
            ctrl_masks = plan["gen_ctrl_masks"]
            pos = plan["gen_pos"]
            col = plan["gen_col"]
            v = voltage[pos]
            p_col = plan["gen_p_col"]
            v_ctrl = ctrl_masks[DC_CTRL_V]
            p_ctrl = ctrl_masks[DC_CTRL_P]
            i_ctrl = ctrl_masks[DC_CTRL_I]
            p = np.zeros(rows.size, dtype=np.float64)
            if v_ctrl.any():
                p[v_ctrl] = v_generator_power[plan["gen_vgen_pos"][v_ctrl]]
            if p_ctrl.any():
                p[p_ctrl] = plan["gen_p_set"][p_ctrl]
            if i_ctrl.any():
                p[i_ctrl] = plan["gen_i_set"][i_ctrl] * v[i_ctrl]

            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_V_GEN])
            self._add_indexed_values(
                H, rows, p_col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_P_GEN] & v_ctrl
            )
            self._add_indexed_values(H, rows, col, plan["gen_i_set"], kind_masks[MEAS_TYPE_P_GEN] & i_ctrl)
            self._add_indexed_values(H, rows, p_col, 1.0 / v, kind_masks[MEAS_TYPE_I_GEN] & v_ctrl)
            self._add_indexed_values(H, rows, col, -p / (v * v), kind_masks[MEAS_TYPE_I_GEN] & v_ctrl)
            self._add_indexed_values(
                H, rows, col, -plan["gen_p_set"] / (v * v), kind_masks[MEAS_TYPE_I_GEN] & p_ctrl
            )

        rows = plan["switch_rows"]
        if rows.size:
            kind_masks = plan["switch_kind_masks"]
            i = plan["switch_i"]
            j = plan["switch_j"]
            i_col = plan["switch_i_col"]
            j_col = plan["switch_j_col"]
            col = plan["switch_col"]
            vi = voltage[i]
            vj = voltage[j]
            current = np.zeros(rows.size, dtype=np.float64)
            current_mask = plan["switch_pos"] >= 0
            if current_mask.any():
                current[current_mask] = switch_current[plan["switch_pos"][current_mask]]

            mask = kind_masks[MEAS_TYPE_P_FROM]
            self._add_indexed_values(H, rows, i_col, current, mask)
            self._add_indexed_values(H, rows, col, vi, mask)
            mask = kind_masks[MEAS_TYPE_V_FROM]
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind_masks[MEAS_TYPE_I_FROM]
            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind_masks[MEAS_TYPE_P_TO]
            self._add_indexed_values(H, rows, j_col, -current, mask)
            self._add_indexed_values(H, rows, col, -vj, mask)
            mask = kind_masks[MEAS_TYPE_V_TO]
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind_masks[MEAS_TYPE_I_TO]
            self._add_indexed_values(H, rows, col, -np.ones(rows.size, dtype=np.float64), mask)

        rows = plan["constraint_rows"]
        if rows.size:
            self._add_indexed_values(H, rows, plan["constraint_i_col"], np.ones(rows.size, dtype=np.float64))
            self._add_indexed_values(H, rows, plan["constraint_j_col"], -np.ones(rows.size, dtype=np.float64))

        rows = plan["dcdc_rows"]
        if rows.size:
            kind_masks = plan["dcdc_kind_masks"]
            i = plan["dcdc_i"]
            j = plan["dcdc_j"]
            i_col = plan["dcdc_i_col"]
            j_col = plan["dcdc_j_col"]
            p_col = plan["dcdc_p_col"]
            q_col = plan["dcdc_q_col"]
            conv_pos = plan["dcdc_pos"]
            p_from = dcdc_power[2 * conv_pos]
            p_to = dcdc_power[2 * conv_pos + 1]
            v_from = voltage[i]
            v_to = voltage[j]

            self._add_indexed_values(H, rows, p_col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_P_FROM])
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_V_FROM])
            mask = kind_masks[MEAS_TYPE_I_FROM]
            self._add_indexed_values(H, rows, p_col, 1.0 / v_from, mask)
            self._add_indexed_values(H, rows, i_col, -p_from / (v_from * v_from), mask)
            self._add_indexed_values(H, rows, q_col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_P_TO])
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), kind_masks[MEAS_TYPE_V_TO])
            mask = kind_masks[MEAS_TYPE_I_TO]
            self._add_indexed_values(H, rows, q_col, 1.0 / v_to, mask)
            self._add_indexed_values(H, rows, j_col, -p_to / (v_to * v_to), mask)

        return plan["handled_mask"]

    def _assemble_jacobian(
        self,
        x: np.ndarray,
        measurement_plan_tables=None,
        sparse: bool = False,
    ):
        """Assemble analytical DC measurement sensitivities for WLS."""
        measurement_plan_tables = self._measurement_plan_tables_for(measurement_plan_tables)
        vector_plans = self._vector_plans_for_measurement_plan_tables(measurement_plan_tables)
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
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
            # calls (e.g. from a hybrid parent) reuse the CSR pattern instead
            # of rebuilding it each call.
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
        vectorized_rows = self._fill_jacobian_vectorized(
            H,
            vector_plans["simple"],
            voltage,
            switch_current,
            dcdc_power,
            v_generator_power,
        )
        if active_vectorized:
            return H.to_csr() if sparse else H
        if not np.all(vectorized_rows):
            missing = int(vectorized_rows.size - np.count_nonzero(vectorized_rows))
            warnings.warn(
                f"DC SE jacobian skipped {missing} non-vectorized measurement rows; object/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        return H.to_csr() if sparse else H

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[object] = None):
        """Assemble analytical DC measurement sensitivities directly as sparse CSR."""
        return self._assemble_jacobian(x, measurements, sparse=True)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[object] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        """Use singular values of H to locate unobservable DC state combinations."""
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
            and is_sparse_matrix(H)
            and int(H.shape[0]) >= int(self.n_state)
        ):
            structural_rank = sparse_structural_rank(H)
            if structural_rank == self.n_state:
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
        """Certify DC observability when sparse structure covers every state."""
        rank = sparse_structural_rank(H)
        return rank == self.n_state

    def estimate(
        self,
        measurements: Optional[object] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        """Solve the weighted least-squares DC state estimate with damped Newton steps."""
        if not self._prepared:
            self.prepare()
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        source_measurements = (
            None
            if measurements is None or isinstance(measurements, dict) or isinstance(measurements, MeasurementPlanTable)
            else self._normalize_measurements(measurements)
        )
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
        normal_assembly_plan = None if active_measurement_run else getattr(self, "_active_normal_assembly_plan", None)
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

        # Pre-allocated buffers reused across iterations and line-search trials.
        # ``z_est_dirty`` tracks whether the main buffer already reflects the
        # current ``x`` after an accepted step swap.
        z_est = np.empty(n_meas, dtype=np.float64)
        residual = np.empty(n_meas, dtype=np.float64)
        cand_z_est = np.empty(n_meas, dtype=np.float64)
        cand_residual = np.empty(n_meas, dtype=np.float64)
        z_est_dirty = True

        for iteration in range(1, self.max_iter + 1):
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
                not is_sparse_matrix(H)
                or tuple(H.shape) != lower_normal_plan.shape
                or int(H.nnz) != int(lower_normal_plan.h_indices.size)
            ):
                lower_normal_plan = None
                if active_measurement_run:
                    self._active_lower_normal_plan = None
                if observability_cache is not None:
                    observability_cache["lower_normal_plan"] = None
            if active_measurement_run and not use_aat_normal_solver and lower_normal_plan is None and is_sparse_matrix(H):
                start = time.perf_counter() if self.profile_enabled else None
                lower_normal_plan = LowerNormalEquationCscPlan.from_jacobian(H)
                if start is not None:
                    self._record_profile_time("solve.lower_normal_plan_build", time.perf_counter() - start)
                self._active_lower_normal_plan = lower_normal_plan
                if observability_cache is not None:
                    observability_cache["lower_normal_plan"] = lower_normal_plan
            if normal_assembly_plan is None and is_sparse_matrix(H) and not normal_assembly_plan_disabled:
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
                and is_sparse_matrix(H)
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
                if aat_plan is not None and not active_measurement_run and not aat_plan.matches(H):
                    aat_plan = None
                if (
                    aat_plan is not None
                    and active_measurement_run
                    and (
                        not is_sparse_matrix(H)
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
                dx, _ = normal_solver.solve(gain, rhs, return_factor_diag=False)
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
                candidate[: self.n_voltage] = np.maximum(candidate[: self.n_voltage], self.voltage_floor)
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
                if candidate_objective <= objective + objective_tol:
                    x = candidate
                    # Swap so the accepted candidate becomes the "main" buffer.
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
            measurements=(),
            observability=observability,
            measurement_plan_tables=measurement_plan_tables,
            measurement_table=result_table,
        )

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Compute largest normalized residuals after accounting for measurement leverage."""
        profile_start = time.perf_counter() if self.profile_enabled else None
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
        weights = np.asarray(measurement_table.weight, dtype=np.float64)
        threshold = self.params.bad_threshold if threshold is None else threshold
        if result.residual.size and threshold > 0.0:
            normalized_upper_bound = np.abs(result.residual) / np.sqrt(1e-12)
            if float(normalized_upper_bound.max()) <= float(threshold):
                self._record_profile_time("bad_data.fast_residual_bound", 0.0)
                if profile_start is not None:
                    self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
                return [], normalized_upper_bound
        if result.H is None or result.gain is None:
            result.H = self.jacobian_sparse(result.x, plan_tables)
            result.gain, _ = build_normal_equations(
                result.H,
                result.residual,
                weights,
            )
        R_diag = 1.0 / weights
        if self.profile_enabled:
            gain_inverse_start = time.perf_counter()
        else:
            gain_inverse_start = None
        gain_inv = inverse_gain_for_bad_data(result.gain)
        if gain_inverse_start is not None:
            self._record_profile_time("bad_data.gain_inverse", time.perf_counter() - gain_inverse_start)
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
            measured_value = float(measurement_table.value[row_pos])
            bad_items.append(
                BadDataItem(
                    measurement=meas,
                    residual=float(result.residual[idx]),
                    normalized_residual=float(normalized[idx]),
                    estimated_value=float(result.z_est[idx]),
                    measured_value=measured_value,
                    row_pos=row_pos,
                )
            )
        bad_items.sort(key=lambda item: item.normalized_residual, reverse=True)
        if profile_start is not None:
            self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
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

    def write_back(
        self,
        result: EstimateResult,
        *,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        result_mode: str = "full",
    ) -> Optional[SEResult]:
        """Store estimate_result / bad-data / se_result after run()."""
        self.estimate_result = result
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        self.bad_items = list(bad_items)
        self.normalized_residual = np.asarray(normalized_residual, dtype=np.float64)
        return self.build_se_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            threshold=threshold,
            result_mode=result_mode,
        )

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
        self.removed_bad_data = removed
        if skip_bad_data:
            bad_items = []
            normalized = np.array([], dtype=np.float64)
        else:
            bad_items, normalized = self.identify_bad_data(result, threshold)
        return self.write_back(
            result,
            bad_items=bad_items,
            normalized_residual=normalized,
            threshold=threshold,
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
                    "DC SE bad-data removal requires BadDataItem.row_pos; object-index fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            measurement_plan_tables = self._shrink_measurement_plan_tables(measurement_plan_tables, remove_pos)
            x0 = result.x
        return result, removed

    def apply_state(self, x: np.ndarray) -> None:
        """Write estimated voltages, currents and powers back to PPC arrays."""
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        ppc = self._dc_ppc
        bus = ppc.get("bus")
        if isinstance(bus, np.ndarray) and bus.size:
            raw_rows = self._raw_node_rows_alive.astype(np.intp, copy=False)
            raw_pos = self._raw_node_solver_pos_alive.astype(np.intp, copy=False)
            bus[raw_rows, DC_BUS_COLS["voltage"]] = voltage[raw_pos]

        if self._apply_branch_i.size:
            vi = voltage[self._apply_branch_i]
            vj = voltage[self._apply_branch_j]
            current = (vi - vj) * self._apply_branch_inv_r
            p_from = vi * current
            p_to = -vj * current
            rows = self._branch_rows.astype(np.intp, copy=False)
            branch = ppc.get("branch")
            if isinstance(branch, np.ndarray) and branch.size:
                branch[rows, DC_BRANCH_COLS["i_p"]] = p_from
                branch[rows, DC_BRANCH_COLS["j_p"]] = p_to
                branch[rows, DC_BRANCH_COLS["current"]] = current

        if self._apply_load_pos.size:
            v = voltage[self._apply_load_pos]
            p = self._apply_load_pv0 + self._apply_load_pv1 * v + self._apply_load_pv2 * v * v
            current = np.zeros_like(p)
            np.divide(p, v, out=current, where=np.abs(v) > self.min_current_voltage)
            rows = self._load_rows.astype(np.intp, copy=False)
            load = ppc.get("load")
            if isinstance(load, np.ndarray) and load.size:
                load[rows, DC_LOAD_COLS["p"]] = p
                load[rows, DC_LOAD_COLS["current"]] = current

        if self._apply_generator_pos.size:
            v = voltage[self._apply_generator_pos]
            p = np.zeros(self._apply_generator_pos.size, dtype=np.float64)
            ctrl = self._apply_generator_ctrl
            v_mask = ctrl == DC_CTRL_V
            p_mask = ctrl == DC_CTRL_P
            i_mask = ctrl == DC_CTRL_I
            if np.any(v_mask):
                p[v_mask] = v_generator_power[self._apply_generator_v_pos[v_mask]]
            p[p_mask] = self._apply_generator_p_set[p_mask]
            p[i_mask] = self._apply_generator_i_set[i_mask] * v[i_mask]
            current = np.zeros_like(p)
            current[i_mask] = self._apply_generator_i_set[i_mask]
            derived = ~i_mask
            np.divide(
                p,
                v,
                out=current,
                where=derived & (np.abs(v) > self.min_current_voltage),
            )
            rows = self._generator_rows.astype(np.intp, copy=False)
            gen = ppc.get("gen")
            if isinstance(gen, np.ndarray) and gen.size:
                gen[rows, DC_GEN_COLS["p"]] = p
                gen[rows, DC_GEN_COLS["current"]] = current

        if self._apply_break_i.size:
            vi = voltage[self._apply_break_i]
            current = switch_current[self._apply_break_pos]
            p_from = vi * current
            rows = self._break_rows.astype(np.intp, copy=False)
            brk = ppc.get("break")
            if isinstance(brk, np.ndarray) and brk.size:
                brk[rows, DC_BREAK_COLS["p"]] = p_from
                brk[rows, DC_BREAK_COLS["current"]] = current

        if self._apply_zero_branch_i.size:
            vi = voltage[self._apply_zero_branch_i]
            current = switch_current[self._apply_zero_branch_pos]
            p_from = vi * current
            rows = self._zero_branch_rows.astype(np.intp, copy=False)
            zbr = ppc.get("zero_branch")
            if isinstance(zbr, np.ndarray) and zbr.size:
                zbr[rows, DC_ZERO_BRANCH_COLS["p"]] = p_from
                zbr[rows, DC_ZERO_BRANCH_COLS["current"]] = current

        if self._apply_dcdc_i.size:
            conv_pos = self._apply_dcdc_pos
            p_from = dcdc_power[2 * conv_pos]
            p_to = dcdc_power[2 * conv_pos + 1]
            v_from = voltage[self._apply_dcdc_i]
            v_to = voltage[self._apply_dcdc_j]
            i_from = np.zeros_like(p_from)
            i_to = np.zeros_like(p_to)
            np.divide(p_from, v_from, out=i_from, where=np.abs(v_from) > self.min_current_voltage)
            np.divide(p_to, v_to, out=i_to, where=np.abs(v_to) > self.min_current_voltage)
            rows = self._dcdc_rows.astype(np.intp, copy=False)
            dcdc = ppc.get("dcdc")
            if isinstance(dcdc, np.ndarray) and dcdc.size:
                dcdc[rows, DC_DCDC_COLS["i_p"]] = p_from
                dcdc[rows, DC_DCDC_COLS["j_p"]] = p_to
                dcdc[rows, DC_DCDC_COLS["i_c"]] = i_from
                dcdc[rows, DC_DCDC_COLS["j_c"]] = i_to

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        print("Estimated DC node voltages:")
        for pos, name in enumerate(self._node_name_by_pos[:limit]):
            print(f"  {str(name):10s} V={voltage[pos]:.9f}")
        if self.n_nodes > limit:
            print(f"  ... {self.n_nodes - limit} more nodes")

        if self._zero_branch_names.size:
            print("Estimated zero-branch currents:")
            for pos, name in enumerate(self._zero_branch_names[:limit]):
                print(f"  {str(name):10s} I={switch_current[pos]:.9f}")
            if self._zero_branch_names.size > limit:
                print(f"  ... {self._zero_branch_names.size - limit} more zero branches")
        if self._break_names.size:
            print("Estimated break currents:")
            offset = self._zero_branch_names.size
            for pos, name in enumerate(self._break_names[:limit]):
                print(f"  {str(name):10s} I={switch_current[offset + pos]:.9f}")
            if self._break_names.size > limit:
                print(f"  ... {self._break_names.size - limit} more breaks")

        if self._dcdc_names.size:
            print("Estimated DCDC port powers:")
            for pos, name in enumerate(self._dcdc_names[:limit]):
                base = 2 * pos
                print(f"  {str(name):10s} P_FROM={dcdc_power[base]:.9f} P_TO={dcdc_power[base + 1]:.9f}")
            if self._dcdc_names.size > limit:
                print(f"  ... {self._dcdc_names.size - limit} more DCDC converters")

        if self._v_generator_names.size:
            print("Estimated voltage-source generator powers:")
            for pos, name in enumerate(self._v_generator_names[:limit]):
                print(f"  {str(name):10s} P={v_generator_power[pos]:.9f}")
            if self._v_generator_names.size > limit:
                print(f"  ... {self._v_generator_names.size - limit} more voltage-source generators")


_ORIGINAL_DC_RUN_POWER_FLOW_SEED = DCStateEstimator._run_power_flow_seed


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
    parser = argparse.ArgumentParser(description="DC weighted least-squares state estimation.")
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="DC network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat voltage start instead of power-flow seed.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated DC states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    parser.add_argument("--profile", action="store_true", help="Print initialization profile timings.")
    parser.add_argument("--result-mode", default="full", help="SEResult payload mode: full, summary, array, or none.")
    parser.add_argument("--se-result", default=None, help="Write SEResult blocks to a new E file.")
    args = parser.parse_args(argv)

    estimator = DCStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        parameter_file=Path(args.para),
        flat_start=args.flat_start,
        profile=args.profile,
        auto_prepare=False,
    )
    estimator.prepare()

    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    se_result = estimator.run(
        result_mode=args.result_mode if args.se_result else "none",
        remove_bad_data=args.remove_bad_data,
        bad_threshold=bad_threshold,
        max_remove=args.max_remove,
        verbose=not args.quiet,
    )
    _print_observability(estimator.observability_result)
    result = estimator.estimate_result
    removed = estimator.removed_bad_data
    if removed:
        print("Removed bad data:")
        for item in removed:
            print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")
    bad_items = estimator.bad_items
    normalized = estimator.normalized_residual
    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
    )
    _print_bad_data(bad_items, normalized, bad_threshold)
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
