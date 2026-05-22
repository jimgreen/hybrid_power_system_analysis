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


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.dc_lf import DCPowerFlowCalc
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
)
from model.ppc_topology import build_dc_ppc_with_topology_from_e_file, ensure_dc_ppc_topology
from model.meas_array_model import (
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_list_from_meas_ppc,
    measurement_table_from_meas_ppc,
    sync_meas_ppc_from_measurement_table,
)
from model.meas_type import (
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCBreakConstraint,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_CODES,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCGenerator,
    DEVICE_TYPE_DCLoad,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCSwitchConstraint,
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
    Measurement,
    MeasurementList,
    MeasurementTable,
    MeasurementTableView,
    ObservabilityResult,
    TableBackedMeasurementList,
    measurement_from_table_row,
    measurement_table_from_measurements,
    measurement_table_status_code,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from secore.se_math import (
    NormalEquationSolver,
    SparseJacobianBuilder,
    _normal_equation_structural_pattern,
    build_normal_equations,
    is_sparse_matrix,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    observability_rank_details,
    observability_weak_direction,
    targeted_redundancy_count,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata
from secore.se_array_plan import (
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    measurement_table_take,
    rows_by_device_type_code,
    take_measurement_view,
)
from secore.se_result import SEResult, build_seresult_summary, normalize_seresult_result_mode
from unit_system import dc_current_base_ka


DEFAULT_CASE = model_file("dc", "dc_net_30.e")
DEFAULT_MEAS = measurement_file("dc", "dc_net_30.meas")

_DEVICE_TYPE_CODES = DEVICE_TYPE_CODES
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
_ACTIVE_NAME_ID_KEY_BITS = 32
_ACTIVE_MEAS_TYPE_KEY_BITS = 8
def _measurement_type_code_lookup(meas_type_codes: Sequence[int]) -> np.ndarray:
    lookup = np.full(_MAX_MEAS_TYPE_CODE + 1, -1, dtype=np.int16)
    for value in np.asarray(meas_type_codes, dtype=np.int16):
        code = int(value)
        if 0 <= int(code) < lookup.size:
            lookup[int(code)] = int(value)
    return lookup


def _measurement_table_from_measurements(measurements: Sequence["Measurement"]) -> MeasurementTable:
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=_DEVICE_TYPE_CODES,
    )

def load_dc_ppc_from_e_file(file_name) -> Dict:
    """Read a DC E file into PPC with topology arrays attached."""
    return build_dc_ppc_with_topology_from_e_file(file_name)


def _build_dc_se_network_from_ppc_dict(ppc: Dict) -> SimpleNamespace:
    """Create the DC SE network holder without materializing device objects."""
    ensure_dc_ppc_topology(ppc)
    topology_arrays = ppc["_topology_arrays"]
    base = ppc["base"]
    return SimpleNamespace(
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
        network: Optional[object] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
        profile: bool = False,
        auto_prepare: bool = True,
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

    def _dc_ppc_dict(self) -> Dict:
        ppc = getattr(self.network, "ppc", None)
        if not isinstance(ppc, dict):
            ppc = getattr(self.network, "_array_model", None)
        if not isinstance(ppc, dict):
            raise RuntimeError("DC SE requires a PPC-backed DC network")
        return ppc

    @staticmethod
    def _ppc_names_for_rows(names, rows: np.ndarray, prefix: str, idx_values: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        names_array = np.asarray(names if names is not None else (), dtype=object)
        if names_array.size and rows.size and int(np.max(rows)) < names_array.size:
            return names_array[rows.astype(np.intp, copy=False)].astype(object, copy=False)
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
        self._node_name_by_pos = self._ppc_names_for_rows(bus_names, first_node_rows, "bus", self._node_idx_by_pos)
        self._node_vbase_by_pos = bus[first_node_rows.astype(np.intp, copy=False), DC_BUS_COLS["vbase"]].astype(np.float64, copy=True)
        self._node_voltage_by_pos = bus[first_node_rows.astype(np.intp, copy=False), DC_BUS_COLS["voltage"]].astype(np.float64, copy=True)
        self._node_voltage_file_base_by_pos = self.u_scale * self._node_vbase_by_pos
        self._node_current_file_base_by_pos = np.asarray(
            [self.i_scale * dc_current_base_ka(self.p_base_kW, float(vbase)) for vbase in self._node_vbase_by_pos],
            dtype=np.float64,
        )

        node_rows = np.arange(int(topology_arrays.node_ids.size), dtype=np.int64)
        raw_bus_pos = topology_arrays.node_to_bus_pos.astype(np.int64, copy=False)
        raw_solver_pos = np.full(raw_bus_pos.size, -1, dtype=np.int64)
        valid_raw_bus = (raw_bus_pos >= 0) & (raw_bus_pos < bus_solver_pos.size)
        raw_solver_pos[valid_raw_bus] = bus_solver_pos[raw_bus_pos[valid_raw_bus].astype(np.intp, copy=False)]
        alive_node_rows = node_rows[raw_solver_pos >= 0]
        raw_node_names = self._ppc_names_for_rows(
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
            names = self._ppc_names_for_rows(ppc.get(name_key), rows, prefix, table[row_pos, cols["idx"]])
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
            self._switch_rows,
            self._switch_names,
            self._switch_i_node,
            self._switch_j_node,
            self._switch_i_pos,
            self._switch_j_pos,
            _switch_unused,
        ) = terminal_context("switch", "switch_name", DC_SWITCH_COLS, "switch")
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
            names = self._ppc_names_for_rows(ppc.get(name_key), rows, prefix, table[row_pos, cols["idx"]])
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
            self._dcdc_names = self._ppc_names_for_rows(ppc.get("dcdc_name"), self._dcdc_rows, "dcdc", dcdc[row_pos, DC_DCDC_COLS["idx"]])
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
        measurements: Optional[Sequence[Measurement]] = None,
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
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        if network is None:
            self.network = self._load_network(self.e_file)
        else:
            self.network = network
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            if bool(getattr(self, "_array_only_runtime", False)):
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
            else:
                self.meas_ppc = copy_meas_ppc(build_meas_ppc_from_e_file(self.meas_file))
                self.measurements = measurement_list_from_meas_ppc(self.meas_ppc)
        elif isinstance(measurements, dict) and measurements.get("format") == "meas_ppc_v1":
            self.meas_ppc = copy_meas_ppc(measurements)
            if bool(getattr(self, "_array_only_runtime", False)):
                self.measurements = self._measurement_sequence_from_table(
                    measurement_table_from_meas_ppc(self.meas_ppc, include_strings=False),
                    normalized=bool(self.meas_ppc.get("normalized", False)),
                )
            else:
                self.measurements = measurement_list_from_meas_ppc(self.meas_ppc)
        elif isinstance(measurements, MeasurementList):
            self.measurements = measurements
            self.meas_ppc = None
        else:
            self.measurements = list(measurements)
            self.meas_ppc = None
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)
        stage_start = time.perf_counter()
        self._build_array_device_context()
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
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
                labels = [str(label) for label in self._state_meta_arrays_ref()["legacy_label"].tolist()]
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
    ) -> "DCStateEstimator":
        self._defer_prepare_finalize_pending = False
        if not measurements_already_normalized:
            stage_start = time.perf_counter()
            self._disable_unavailable_measurements()
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
            self._jacobian_builder = SparseJacobianBuilder((0, self.n_state))
            self._jacobian_builder._assume_fixed_pattern = True
            self._measurement_plan_cache = {}
        self._prepared = True
        return self

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    def _refresh_active_measurement_indexes(self) -> None:
        """Rebuild active measurement arrays and the vectorized measurement plan."""
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=not bool(getattr(self, "_array_only_runtime", False)),
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
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._observability_matrix_cache = None
        self._measurement_plan_cache = {}
        self._active_measurement_plan = self._measurement_plan(self.active_measurements)

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

    def _shrink_active_measurement_plan(self, plan: Dict[str, np.ndarray], removed_pos: int) -> Dict[str, np.ndarray]:
        removed_pos = int(removed_pos)

        def shrink_rows(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            row_array = np.asarray(rows, dtype=np.int64)
            keep = row_array != removed_pos
            kept = row_array[keep]
            kept = kept - (kept > removed_pos)
            return kept.astype(np.int64, copy=False), keep

        merged: Dict[str, np.ndarray] = {}
        merged["handled_mask"] = np.delete(np.asarray(plan["handled_mask"], dtype=bool), removed_pos)

        for row_key, value_keys in (
            ("node_rows", ("node_pos", "node_col")),
            ("branch_rows", ("branch_kind", "branch_i", "branch_j", "branch_i_col", "branch_j_col", "branch_inv_r")),
            ("load_rows", ("load_kind", "load_pos", "load_col", "load_pv0", "load_pv1", "load_pv2")),
            ("gen_rows", ("gen_kind", "gen_ctrl", "gen_pos", "gen_col", "gen_p_col", "gen_vgen_pos", "gen_p_set", "gen_i_set")),
            ("switch_rows", ("switch_kind", "switch_i", "switch_j", "switch_i_col", "switch_j_col", "switch_col", "switch_pos")),
            ("constraint_rows", ("constraint_i", "constraint_j", "constraint_i_col", "constraint_j_col")),
            ("dcdc_rows", ("dcdc_kind", "dcdc_i", "dcdc_j", "dcdc_i_col", "dcdc_j_col", "dcdc_p_col", "dcdc_q_col", "dcdc_pos")),
        ):
            shrunk_rows, keep = shrink_rows(plan[row_key])
            merged[row_key] = shrunk_rows
            for value_key in value_keys:
                merged[value_key] = np.asarray(plan[value_key])[keep]
        DCStateEstimator._populate_kind_masks(merged)
        return merged

    def _shrink_active_measurement_indexes(self, removed_pos: int) -> MeasurementList:
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        self.active_measurements = take_measurement_view(self.active_measurements, keep_rows)
        self.active_measurement_table = self.active_measurements.table
        self.active_measurement_rows = np.arange(len(self.active_measurements), dtype=np.int64)
        self.measurement_table = self.active_measurement_table
        self.active_z = np.asarray(self.active_measurement_table.value, dtype=np.float64)
        self.active_weight = np.asarray(self.active_measurement_table.weight, dtype=np.float64)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        if not hasattr(self, "n_state"):
            return self.active_measurements
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._observability_matrix_cache = None
        if hasattr(self, "_active_measurement_plan"):
            self._active_measurement_plan = self._shrink_active_measurement_plan(self._active_measurement_plan, removed_pos)
            self._measurement_plan_cache = {
                id(self.active_measurements): (self.active_measurements, self._active_measurement_plan)
            }
        else:
            self._measurement_plan_cache = {}
            self._active_measurement_plan = self._measurement_plan(self.active_measurements)
        return self.active_measurements

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]):
        if measurements is None:
            return self.active_measurements
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return measurements
        if isinstance(measurements, list):
            return measurements
        return list(measurements)

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

    def _measurement_vectors(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
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

    def _cache_observability_matrix(
        self,
        result: ObservabilityResult,
        x: np.ndarray,
        measurements: Sequence[Measurement],
        H,
    ) -> None:
        self._observability_matrix_cache = {
            "result": result,
            "measurements": measurements,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "H": H,
            "normal_pattern": None,
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
        if cached_x is None or cached_x.shape != np.asarray(x).shape or not np.array_equal(cached_x, x):
            return None
        return cache

    def _disable_unavailable_measurements(self) -> None:
        """Keep invalid/off-topology measurement rows out of unit conversion and WLS."""
        table = getattr(self.measurements, "table", None)
        table_valid = table.valid if table is not None and len(table.valid) == len(self.measurements) else None
        table_status = measurement_table_status_code(table) if table_valid is not None else None
        if table_valid is not None:
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
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
            return dict(cached)
        self._refresh_measurement_summary_cache()
        return dict(getattr(self, "_node_voltage_measurement_cache", {}))

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
        components = getattr(self, "zero_tie_components", None)
        if pos < 0 or component_by_pos is None or not components:
            return None
        component = components[int(component_by_pos[int(pos)])]
        for member_pos in component:
            member_idx = int(self._node_idx_by_pos[int(member_pos)])
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_pseudo_is_covered_by_pos(self, device_type_code: int, device_pos: int, meas_type_code: int) -> bool:
        """Check whether a voltage pseudo row is redundant because the node already has real V data."""
        node_idx = self._voltage_measurement_node_idx_from_pos(device_type_code, device_pos, meas_type_code)
        return self._real_voltage_observation_value_for_node(node_idx) is not None

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live DC topology terminals used when choosing island voltage references."""
        degrees = np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.int64)
        for left, right in (
            (self._branch_i_pos, self._branch_j_pos),
            (self._zero_branch_i_pos, self._zero_branch_j_pos),
            (self._switch_i_pos, self._switch_j_pos),
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
        return dict(zip(self._node_idx_by_pos.tolist(), degrees.tolist()))

    def _select_reference_nodes(self) -> np.ndarray:
        """Choose one measured high-degree DC voltage reference per live DC topology island."""
        references = []
        voltage_measurements = self.node_voltage_measurements
        degrees = getattr(self, "_node_degree_by_pos", np.zeros(int(getattr(self, "n_nodes", 0)), dtype=np.int64))
        topology_arrays = self._dc_topology_arrays
        bus_solver_pos = self._bus_solver_pos
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
            measured = [
                int(pos)
                for pos in solver_pos.tolist()
                if int(self._node_idx_by_pos[int(pos)]) in voltage_measurements
            ]
            if measured:
                references.append(max(measured, key=lambda pos: (int(degrees[int(pos)]), -int(self._node_idx_by_pos[int(pos)]))))
                continue
            ref_bus = int(topology_arrays.island_reference_bus_pos[island_pos])
            ref_solver = int(bus_solver_pos[ref_bus]) if 0 <= ref_bus < bus_solver_pos.size else -1
            references.append(ref_solver if ref_solver >= 0 else int(solver_pos[0]))
        return np.asarray(references, dtype=np.int64)

    def _build_zero_tie_voltage_layout(self) -> None:
        """Compress DC voltage states across closed switches and zero branches."""
        n = int(self.n_nodes)
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

        for left, right in (
            (self._zero_branch_i_pos, self._zero_branch_j_pos),
            (self._switch_i_pos, self._switch_j_pos),
            (self._break_i_pos, self._break_j_pos),
        ):
            for i, j in zip(np.asarray(left, dtype=np.int64).tolist(), np.asarray(right, dtype=np.int64).tolist()):
                if 0 <= int(i) < n and 0 <= int(j) < n:
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

        node_voltage_state = np.full(n, -1, dtype=np.int32)
        voltage_state_pos = []
        self.voltage_state_nodes: List[Sequence[int]] = []
        self.ref_voltages: Dict[int, float] = {}
        reference_voltage_by_pos = {}
        for pos in np.asarray(getattr(self, "references", ()), dtype=np.int64).tolist():
            if not (0 <= int(pos) < self._node_idx_by_pos.size):
                continue
            node_idx = int(self._node_idx_by_pos[int(pos)])
            reference_voltage_by_pos[int(pos)] = float(
                self.node_voltage_measurements.get(node_idx, self._node_voltage_by_pos[int(pos)])
            )
        state_col = 0
        voltage_expand_pos = []
        voltage_expand_col = []
        for component in components:
            fixed_pos = next((int(pos) for pos in component if int(pos) in reference_voltage_by_pos), None)
            if fixed_pos is not None:
                fixed_voltage = max(float(reference_voltage_by_pos[fixed_pos]), self.voltage_floor)
                for pos in component:
                    self.ref_voltages[int(pos)] = fixed_voltage
                continue
            voltage_state_pos.append(component[0])
            self.voltage_state_nodes.append(component)
            voltage_expand_pos.extend(component)
            voltage_expand_col.extend([state_col] * len(component))
            for pos in component:
                node_voltage_state[pos] = state_col
            state_col += 1

        self.node_voltage_state = node_voltage_state
        self.voltage_state_pos = np.asarray(voltage_state_pos, dtype=np.int32)
        self.voltage_col = node_voltage_state.copy()
        self.n_voltage = int(self.voltage_state_pos.size)
        self._voltage_expand_pos = np.asarray(voltage_expand_pos, dtype=np.int64)
        self._voltage_expand_col = np.asarray(voltage_expand_col, dtype=np.int64)
        if self.ref_voltages:
            refs = sorted(self.ref_voltages.items())
            self._ref_voltage_pos = np.asarray([pos for pos, _value in refs], dtype=np.int64)
            self._ref_voltage_value = np.asarray([value for _pos, value in refs], dtype=np.float64)
        else:
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

        constraint_i = np.concatenate((self._zero_branch_i_pos, self._switch_i_pos, self._break_i_pos)).astype(np.int64, copy=False)
        constraint_j = np.concatenate((self._zero_branch_j_pos, self._switch_j_pos, self._break_j_pos)).astype(np.int64, copy=False)
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
        self._measurement_plan_device_pos_by_type_code = {}
        self._measurement_plan_meas_kind_by_type_code = {}
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
        self._measurement_plan_device_pos_by_type_code_id = self._measurement_plan_device_id_lookup_arrays()
        self._measurement_plan_meas_kind_code_by_type_code = {
            int(code): _measurement_type_code_lookup(kind_map)
            for code, kind_map in meas_kind_source.items()
        }

    def _ensure_measurement_plan_lookup_arrays(self) -> None:
        if not hasattr(self, "_measurement_plan_device_pos_by_type_code"):
            self._build_measurement_plan_lookup_arrays()

    def _measurement_plan_device_id_lookup_arrays(self) -> Dict[int, np.ndarray]:
        device_names = self._meas_ppc_device_names()
        if device_names.size == 0:
            return {}
        name_id_by_name = self._meas_ppc_device_name_ids()
        if not name_id_by_name:
            return {}

        def lookup_for(names: np.ndarray) -> np.ndarray:
            lookup = np.empty(device_names.size, dtype=np.int64)
            lookup.fill(-1)
            for pos, name in enumerate(np.asarray(names, dtype=object).tolist()):
                name_id = name_id_by_name.get(name)
                if name_id is not None and 0 <= int(name_id) < lookup.size:
                    lookup[int(name_id)] = int(pos)
            return lookup

        constraint_names = np.concatenate((self._zero_branch_names, self._switch_names, self._break_names))
        result: Dict[int, np.ndarray] = {
            DEVICE_TYPE_DCNode: lookup_for(self._raw_node_names_alive),
            DEVICE_TYPE_DCBranch: lookup_for(self._branch_names),
            DEVICE_TYPE_DCLoad: lookup_for(self._load_names),
            DEVICE_TYPE_DCGenerator: lookup_for(self._generator_names),
            DEVICE_TYPE_DCZeroBranch: lookup_for(self._zero_branch_names),
            DEVICE_TYPE_DCBreak: lookup_for(self._break_names),
            DEVICE_TYPE_DCZeroBranchConstraint: lookup_for(constraint_names),
            DEVICE_TYPE_DCBreakConstraint: lookup_for(constraint_names),
            DEVICE_TYPE_DCDCConverter: lookup_for(self._dcdc_names),
        }
        return result

    def _measurement_device_pos_array(self, table: MeasurementTable) -> np.ndarray:
        n_rows = int(table.idx.size)
        precomputed = getattr(table, "device_pos", None)
        if precomputed is not None and np.asarray(precomputed).size == n_rows:
            device_pos = np.asarray(precomputed, dtype=np.int64).copy()
        else:
            device_pos = np.empty(n_rows, dtype=np.int64)
            device_pos.fill(-1)
        if n_rows == 0 or np.all(device_pos >= 0):
            return device_pos
        self._ensure_measurement_plan_lookup_arrays()
        device_name_id = getattr(table, "device_name_id", None)
        if device_name_id is not None:
            device_name_id = np.asarray(device_name_id, dtype=np.int64)
            if device_name_id.size != n_rows:
                device_name_id = None
        id_maps = getattr(self, "_measurement_plan_device_pos_by_type_code_id", {})
        if device_name_id is not None and id_maps:
            for code_int, code_rows in rows_by_device_type_code(table).items():
                rows = np.asarray(code_rows, dtype=np.int64)
                rows = rows[device_pos[rows] < 0]
                if rows.size == 0:
                    continue
                lookup = id_maps.get(int(code_int))
                if lookup is None or lookup.size == 0:
                    continue
                ids = device_name_id[rows]
                valid = (ids >= 0) & (ids < lookup.size)
                if np.any(valid):
                    device_pos[rows[valid]] = lookup[ids[valid].astype(np.intp, copy=False)]
        if np.any(device_pos < 0):
            warnings.warn(
                "DC SE measurement device_pos contains unresolved rows; string fallback is disabled.",
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

    def _measurement_table_for_indexed_plan(self, measurements: Sequence[Measurement]) -> MeasurementTable:
        table = _measurement_table_from_measurements(measurements)
        self._ensure_table_meas_type_codes(table)
        table.device_pos = self._measurement_device_pos_array(table)
        return table

    @staticmethod
    def _load_network(e_file: Path) -> SimpleNamespace:
        """Read the DC case and build topology references used by measurements."""
        ppc = load_dc_ppc_from_e_file(e_file)
        return _build_dc_se_network_from_ppc_dict(ppc)

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
        gen = ppc.get("gen")
        load = ppc.get("load")
        dcdc = ppc.get("dcdc")

        def set_bus_voltage_by_idx(node_idx, value):
            node_ids = bus[:, DC_BUS_COLS["idx"]].astype(np.int64, copy=False)
            matches = np.flatnonzero(node_ids == int(node_idx))
            if matches.size:
                bus[int(matches[0]), DC_BUS_COLS["voltage"]] = max(float(value), 0.0)

        for device_type_code, ppc_row, meas_type_code, value in seed_rows:
            device_type_code = int(device_type_code)
            meas_type_code = int(meas_type_code)
            row = int(ppc_row)
            value = float(value)
            if device_type_code == DEVICE_TYPE_DCNode:
                if meas_type_code == MEAS_TYPE_V and 0 <= row < bus.shape[0]:
                    bus[row, DC_BUS_COLS["voltage"]] = max(value, 0.0)
                continue
            if device_type_code == DEVICE_TYPE_DCGenerator and gen is not None:
                if not (0 <= row < gen.shape[0]):
                    continue
                if meas_type_code == MEAS_TYPE_P_GEN:
                    gen[row, DC_GEN_COLS["p_set"]] = value
                    gen[row, DC_GEN_COLS["p"]] = value
                elif meas_type_code == MEAS_TYPE_V_GEN:
                    voltage = max(value, 0.0)
                    gen[row, DC_GEN_COLS["v_set"]] = voltage
                    set_bus_voltage_by_idx(gen[row, DC_GEN_COLS["node"]], voltage)
                elif meas_type_code == MEAS_TYPE_I_GEN:
                    gen[row, DC_GEN_COLS["i_set"]] = value
                    gen[row, DC_GEN_COLS["current"]] = value
                continue
            if device_type_code == DEVICE_TYPE_DCLoad and load is not None:
                if not (0 <= row < load.shape[0]):
                    continue
                if meas_type_code == MEAS_TYPE_P_LOAD:
                    load[row, DC_LOAD_COLS["pbase"]] = 1.0
                    load[row, DC_LOAD_COLS["pv0"]] = value
                    load[row, DC_LOAD_COLS["pv1"]] = 0.0
                    load[row, DC_LOAD_COLS["pv2"]] = 0.0
                    load[row, DC_LOAD_COLS["p"]] = value
                elif meas_type_code == MEAS_TYPE_V_LOAD:
                    set_bus_voltage_by_idx(load[row, DC_LOAD_COLS["node"]], value)
                elif meas_type_code == MEAS_TYPE_I_LOAD:
                    load[row, DC_LOAD_COLS["current"]] = value
                continue
            if device_type_code == DEVICE_TYPE_DCDCConverter and dcdc is not None:
                if not (0 <= row < dcdc.shape[0]):
                    continue
                if meas_type_code == MEAS_TYPE_P_FROM:
                    dcdc[row, DC_DCDC_COLS["p_set"]] = value
                    dcdc[row, DC_DCDC_COLS["i_p"]] = value
                elif meas_type_code == MEAS_TYPE_P_TO:
                    dcdc[row, DC_DCDC_COLS["j_p"]] = value
                elif meas_type_code == MEAS_TYPE_V_FROM:
                    set_bus_voltage_by_idx(dcdc[row, DC_DCDC_COLS["i_node"]], value)
                elif meas_type_code == MEAS_TYPE_V_TO:
                    set_bus_voltage_by_idx(dcdc[row, DC_DCDC_COLS["j_node"]], value)
                elif meas_type_code == MEAS_TYPE_I_FROM:
                    dcdc[row, DC_DCDC_COLS["i_set"]] = value
                    dcdc[row, DC_DCDC_COLS["i_c"]] = value
                elif meas_type_code == MEAS_TYPE_I_TO:
                    dcdc[row, DC_DCDC_COLS["j_c"]] = value

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
    def _apply_power_flow_seed_calc_state_to_network(network, ppc, calc) -> None:
        bus = ppc.get("bus")
        if hasattr(calc, "x") and hasattr(calc, "N"):
            voltage = np.asarray(calc.x[: calc.N], dtype=np.float64)
            if isinstance(bus, np.ndarray) and bus.size:
                node_pos = getattr(calc, "alive_node_dict", None)
                if node_pos is not None and hasattr(node_pos, "get"):
                    bus[:, DC_BUS_COLS["voltage"]] = 0.0
                    for row, node_idx in enumerate(bus[:, DC_BUS_COLS["idx"]].astype(np.int64, copy=False)):
                        pos = node_pos.get(int(node_idx), -1)
                        if 0 <= int(pos) < voltage.size:
                            bus[row, DC_BUS_COLS["voltage"]] = voltage[int(pos)]
                elif voltage.size == bus.shape[0]:
                    bus[:, DC_BUS_COLS["voltage"]] = voltage
            network.ppc = ppc
            if hasattr(network, "_array_model"):
                network._array_model = ppc

    def _apply_measurement_seed_to_network(self) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is None:
            seed_rows = ()
        setattr(self.network, "_se_power_flow_seed_rows", tuple(seed_rows))

    def _meas_ppc_device_names(self) -> np.ndarray:
        meas_ppc = getattr(self, "meas_ppc", None)
        if isinstance(meas_ppc, dict):
            return np.asarray(meas_ppc.get("device_names", ()), dtype=object)
        return np.asarray([], dtype=object)

    def _meas_ppc_device_name_ids(self) -> Dict[object, int]:
        meas_ppc = getattr(self, "meas_ppc", None)
        if isinstance(meas_ppc, dict):
            lookup = meas_ppc.get("device_name_id_by_name")
            if isinstance(lookup, dict):
                return lookup
        return {}

    @staticmethod
    def _active_measurement_pos_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
        if int(device_pos) < 0:
            return -1
        return (
            (int(device_type_code) << (_ACTIVE_NAME_ID_KEY_BITS + _ACTIVE_MEAS_TYPE_KEY_BITS))
            | (int(device_pos) << _ACTIVE_MEAS_TYPE_KEY_BITS)
            | int(meas_type_code)
        )

    @staticmethod
    def _active_measurement_pos_key_array(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> np.ndarray:
        return (
            (np.asarray(device_type_code, dtype=np.int64) << (_ACTIVE_NAME_ID_KEY_BITS + _ACTIVE_MEAS_TYPE_KEY_BITS))
            | (np.asarray(device_pos, dtype=np.int64) << _ACTIVE_MEAS_TYPE_KEY_BITS)
            | np.asarray(meas_type_code, dtype=np.int64)
        )

    def _seed_ppc_rows_from_device_pos(self, device_type_code: np.ndarray, device_pos: np.ndarray) -> np.ndarray:
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

    def _convert_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal state-estimation units."""
        if getattr(self.measurements, "normalized", False):
            self._refresh_measurement_summary_cache()
            seed_rows = []
            table = getattr(self.measurements, "table", None)
            if not bool(getattr(self, "flat_start", False)) and table is not None:
                code = np.asarray(table.device_type_code, dtype=np.int16)
                meas_type_code = self._ensure_table_meas_type_codes(table)
                device_pos = self._measurement_device_pos_array(table)
                active = np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                rows = np.flatnonzero(active & (device_pos >= 0))
                seed_mask = (
                    ((code[rows] == DEVICE_TYPE_DCNode) & (meas_type_code[rows] == MEAS_TYPE_V))
                    | (
                        (code[rows] == DEVICE_TYPE_DCGenerator)
                        & np.isin(meas_type_code[rows], np.asarray([MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN], dtype=np.int16))
                    )
                    | (
                        (code[rows] == DEVICE_TYPE_DCLoad)
                        & np.isin(meas_type_code[rows], np.asarray([MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD], dtype=np.int16))
                    )
                    | (
                        (code[rows] == DEVICE_TYPE_DCDCConverter)
                        & np.isin(
                            meas_type_code[rows],
                            np.asarray([MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO], dtype=np.int16),
                        )
                    )
                )
                selected = rows[seed_mask]
                seed_ppc_rows = self._seed_ppc_rows_from_device_pos(code[selected], device_pos[selected])
                valid_seed = seed_ppc_rows >= 0
                selected = selected[valid_seed]
                seed_ppc_rows = seed_ppc_rows[valid_seed]
                seed_rows = list(
                    zip(
                        code[selected].astype(np.int16, copy=False).tolist(),
                        seed_ppc_rows.astype(np.int64, copy=False).tolist(),
                        meas_type_code[selected].astype(np.int16, copy=False).tolist(),
                        table.value[selected].astype(np.float64, copy=False).tolist(),
                    )
                )
            self._power_flow_seed_rows = seed_rows
            if getattr(self, "meas_ppc", None) is not None:
                self.meas_ppc["normalized"] = True
                sync_meas_ppc_from_measurement_table(self.meas_ppc, self.measurements.table)
            return
        table = _measurement_table_from_measurements(self.measurements)
        self.measurement_table = table
        try:
            self.measurements.table = table
        except AttributeError:
            pass
        value = table.value
        valid = np.asarray(table.valid, dtype=bool)
        weight = np.asarray(table.weight, dtype=np.float64)
        active = valid & (weight > 0.0)
        scale = np.ones(value.size, dtype=np.float64)
        code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = self._ensure_table_meas_type_codes(table)
        device_pos = self._measurement_device_pos_array(table)
        status_code = measurement_table_status_code(table)

        def assign_single(rows: np.ndarray, node_pos_array: np.ndarray, p_code: int, v_code: int, i_code: int) -> None:
            if rows.size == 0:
                return
            pos = device_pos[rows].astype(np.int64, copy=False)
            valid_pos = (pos >= 0) & (pos < node_pos_array.size)
            if not np.any(valid_pos):
                return
            selected = rows[valid_pos]
            node_pos = node_pos_array[pos[valid_pos].astype(np.intp, copy=False)]
            in_range = (node_pos >= 0) & (node_pos < self._node_voltage_file_base_by_pos.size)
            if not np.any(in_range):
                return
            selected = selected[in_range]
            node_pos = node_pos[in_range].astype(np.intp, copy=False)
            mt = meas_type_code[selected]
            scale[selected[mt == int(p_code)]] = self.p_base
            v_rows = selected[mt == int(v_code)]
            if v_rows.size:
                scale[v_rows] = self._node_voltage_file_base_by_pos[node_pos[mt == int(v_code)]]
            i_rows = selected[mt == int(i_code)]
            if i_rows.size:
                scale[i_rows] = self._node_current_file_base_by_pos[node_pos[mt == int(i_code)]]

        def assign_terminal(
            rows: np.ndarray,
            i_pos_array: np.ndarray,
            j_pos_array: np.ndarray,
        ) -> None:
            if rows.size == 0:
                return
            pos = device_pos[rows].astype(np.int64, copy=False)
            valid_pos = (pos >= 0) & (pos < i_pos_array.size) & (pos < j_pos_array.size)
            if not np.any(valid_pos):
                return
            selected = rows[valid_pos]
            local_pos = pos[valid_pos].astype(np.intp, copy=False)
            i_pos = i_pos_array[local_pos]
            j_pos = j_pos_array[local_pos]
            mt = meas_type_code[selected]
            p_mask = (mt == MEAS_TYPE_P_FROM) | (mt == MEAS_TYPE_P_TO)
            if np.any(p_mask):
                scale[selected[p_mask]] = self.p_base
            v_from = mt == MEAS_TYPE_V_FROM
            if np.any(v_from):
                scale[selected[v_from]] = self._node_voltage_file_base_by_pos[i_pos[v_from].astype(np.intp, copy=False)]
            i_from = mt == MEAS_TYPE_I_FROM
            if np.any(i_from):
                scale[selected[i_from]] = self._node_current_file_base_by_pos[i_pos[i_from].astype(np.intp, copy=False)]
            v_to = mt == MEAS_TYPE_V_TO
            if np.any(v_to):
                scale[selected[v_to]] = self._node_voltage_file_base_by_pos[j_pos[v_to].astype(np.intp, copy=False)]
            i_to = mt == MEAS_TYPE_I_TO
            if np.any(i_to):
                scale[selected[i_to]] = self._node_current_file_base_by_pos[j_pos[i_to].astype(np.intp, copy=False)]

        node_rows = np.flatnonzero(active & (code == DEVICE_TYPE_DCNode) & (meas_type_code == MEAS_TYPE_V) & (device_pos >= 0))
        if node_rows.size:
            raw_pos = device_pos[node_rows].astype(np.intp, copy=False)
            solver_pos = self._raw_node_solver_pos_alive[raw_pos]
            valid_solver = (solver_pos >= 0) & (solver_pos < self._node_voltage_file_base_by_pos.size)
            if np.any(valid_solver):
                scale[node_rows[valid_solver]] = self._node_voltage_file_base_by_pos[solver_pos[valid_solver].astype(np.intp, copy=False)]

        assign_terminal(np.flatnonzero(active & (code == DEVICE_TYPE_DCBranch)), self._branch_i_pos, self._branch_j_pos)
        assign_terminal(np.flatnonzero(active & (code == DEVICE_TYPE_DCBreak)), self._break_i_pos, self._break_j_pos)
        assign_terminal(np.flatnonzero(active & (code == DEVICE_TYPE_DCZeroBranch)), self._zero_branch_i_pos, self._zero_branch_j_pos)
        assign_terminal(np.flatnonzero(active & (code == DEVICE_TYPE_DCDCConverter)), self._dcdc_i_pos, self._dcdc_j_pos)
        assign_single(
            np.flatnonzero(active & (code == DEVICE_TYPE_DCGenerator)),
            self._generator_pos,
            MEAS_TYPE_P_GEN,
            MEAS_TYPE_V_GEN,
            MEAS_TYPE_I_GEN,
        )
        assign_single(
            np.flatnonzero(active & (code == DEVICE_TYPE_DCLoad)),
            self._load_pos,
            MEAS_TYPE_P_LOAD,
            MEAS_TYPE_V_LOAD,
            MEAS_TYPE_I_LOAD,
        )
        constraint_rows = np.flatnonzero(
            active
            & (
                (code == DEVICE_TYPE_DCZeroBranchConstraint)
                | (code == DEVICE_TYPE_DCBreakConstraint)
                | (code == DEVICE_TYPE_DCSwitchConstraint)
            )
            & (meas_type_code == MEAS_TYPE_V_DIFF)
            & (device_pos >= 0)
        )
        if constraint_rows.size:
            constraint_i_pos = np.concatenate((self._zero_branch_i_pos, self._switch_i_pos, self._break_i_pos)).astype(
                np.int64,
                copy=False,
            )
            pos = device_pos[constraint_rows].astype(np.intp, copy=False)
            valid_pos = pos < constraint_i_pos.size
            if np.any(valid_pos):
                i_pos = constraint_i_pos[pos[valid_pos]]
                valid_node = (i_pos >= 0) & (i_pos < self._node_voltage_file_base_by_pos.size)
                if np.any(valid_node):
                    scale[constraint_rows[valid_pos][valid_node]] = self._node_voltage_file_base_by_pos[
                        i_pos[valid_node].astype(np.intp, copy=False)
                    ]
        value[active] = np.divide(value[active], scale[active], out=value[active].copy(), where=np.abs(scale[active]) > 1e-12)
        object_count = list.__len__(self.measurements) if isinstance(self.measurements, list) else 0
        if object_count == value.size and object_count > 0:
            for pos, meas in enumerate(list.__iter__(self.measurements)):
                meas.value = float(value[pos])

        active_rows = np.flatnonzero(active)
        valid_key_rows = active_rows[device_pos[active_rows] >= 0]
        active_measurement_keys = set(
            self._active_measurement_pos_key_array(
                code[valid_key_rows],
                device_pos[valid_key_rows],
                meas_type_code[valid_key_rows],
            ).tolist()
        )
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        seed_rows = []
        collect_power_flow_seed_rows = not bool(getattr(self, "flat_start", False))
        voltage_rows = active_rows[
            (status_code[active_rows] != MEAS_STATUS_PSEUDO)
            & np.isin(meas_type_code[active_rows], _VOLTAGE_MEASUREMENT_TYPE_CODES)
            & (device_pos[active_rows] >= 0)
        ]
        for row in voltage_rows.tolist():
            node_idx = self._voltage_measurement_node_idx_from_pos(
                int(code[row]),
                int(device_pos[row]),
                int(meas_type_code[row]),
            )
            if node_idx is None or self._node_pos_from_idx(node_idx) < 0:
                continue
            row_weight = float(weight[row])
            row_value = float(value[row])
            current = real_voltage_best.get(node_idx)
            if current is None or row_weight > current[0]:
                real_voltage_best[node_idx] = (row_weight, row_value)
            if int(code[row]) == DEVICE_TYPE_DCNode and int(meas_type_code[row]) == MEAS_TYPE_V:
                current = node_voltage_best.get(node_idx)
                if current is None or row_weight > current[0]:
                    node_voltage_best[node_idx] = (row_weight, row_value)
        if collect_power_flow_seed_rows:
            seed_mask = (
                (device_pos[active_rows] >= 0)
                & (
                    ((code[active_rows] == DEVICE_TYPE_DCNode) & (meas_type_code[active_rows] == MEAS_TYPE_V))
                    | (
                        (code[active_rows] == DEVICE_TYPE_DCGenerator)
                        & np.isin(meas_type_code[active_rows], np.asarray([MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN], dtype=np.int16))
                    )
                    | (
                        (code[active_rows] == DEVICE_TYPE_DCLoad)
                        & np.isin(meas_type_code[active_rows], np.asarray([MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD], dtype=np.int16))
                    )
                    | (
                        (code[active_rows] == DEVICE_TYPE_DCDCConverter)
                        & np.isin(
                            meas_type_code[active_rows],
                            np.asarray([MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO], dtype=np.int16),
                        )
                    )
                )
            )
            seed_selected = active_rows[seed_mask]
            seed_ppc_rows = self._seed_ppc_rows_from_device_pos(code[seed_selected], device_pos[seed_selected])
            valid_seed = seed_ppc_rows >= 0
            seed_selected = seed_selected[valid_seed]
            seed_ppc_rows = seed_ppc_rows[valid_seed]
            seed_rows = list(
                zip(
                    code[seed_selected].astype(np.int16, copy=False).tolist(),
                    seed_ppc_rows.astype(np.int64, copy=False).tolist(),
                    meas_type_code[seed_selected].astype(np.int16, copy=False).tolist(),
                    value[seed_selected].astype(np.float64, copy=False).tolist(),
                )
            )
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = int(table.idx.max()) if table.idx.size else 0
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }
        self._power_flow_seed_rows = seed_rows
        if getattr(self, "meas_ppc", None) is not None:
            self.meas_ppc["normalized"] = True
            sync_meas_ppc_from_measurement_table(self.meas_ppc, table)

    def _active_measurement_keys(self) -> set:
        """Return usable measurement keys at device and measurement-type granularity."""
        if not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        return set(self._active_measurement_key_cache)

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
        device_name_ids: Optional[Sequence[int]] = None,
        meas_type_codes: Optional[Sequence[int]] = None,
        device_positions: Optional[Sequence[int]] = None,
    ) -> MeasurementTable:
        row_count = max(
            len(values),
            len(names),
            0 if device_type_codes is None else len(device_type_codes),
            0 if device_name_ids is None else len(device_name_ids),
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
        if device_name_ids is None:
            device_name_id = np.full(row_count, -1, dtype=np.int64)
        else:
            device_name_id = np.asarray(device_name_ids, dtype=np.int64)
            if device_name_id.size != row_count:
                raise ValueError("pseudo measurement device_name_id array size does not match row count")
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
            device_name_id=device_name_id,
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
        device_type_codes: Optional[Sequence[int]] = None,
        device_name_ids: Optional[Sequence[int]] = None,
        meas_type_codes: Optional[Sequence[int]] = None,
        device_positions: Optional[Sequence[int]] = None,
    ) -> int:
        row_count = max(
            len(values),
            len(names),
            0 if device_type_codes is None else len(device_type_codes),
            0 if device_name_ids is None else len(device_name_ids),
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
            device_name_ids=device_name_ids,
            meas_type_codes=meas_type_codes,
            device_positions=device_positions,
        )
        table = getattr(self.measurements, "table", None)
        if table is None:
            warnings.warn(
                "DC SE pseudo measurement append requires a PPC-backed measurement table; skipped table append.",
                RuntimeWarning,
                stacklevel=2,
            )
            return int(next_idx)
        combined_table = concat_measurement_tables(table, appended_table)
        combined_table.rows_by_device_type_code = rows_by_device_type_code(combined_table)
        self.measurements = self._measurement_sequence_from_table(
            combined_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurement_table = combined_table
        if isinstance(self.measurements, TableBackedMeasurementList):
            self.measurements._table_prefix_size = int(combined_table.idx.size)
        self._max_measurement_idx = int(np.max(appended_table.idx))
        return int(next_idx) + row_count

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        active_measurement_keys = set()
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        real_voltage_best: Dict[int, Tuple[float, float]] = {}
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            max_idx = int(table.idx.max()) if table.idx.size else 0
            active = np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)
            active_rows = np.flatnonzero(active)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_type_code = self._ensure_table_meas_type_codes(table)
            device_pos = self._measurement_device_pos_array(table)
            valid_key_rows = active_rows[device_pos[active_rows] >= 0]
            active_measurement_keys = set(
                self._active_measurement_pos_key_array(
                    device_type_code[valid_key_rows],
                    device_pos[valid_key_rows],
                    meas_type_code[valid_key_rows],
                ).tolist()
            )
            status_code = measurement_table_status_code(table)
            voltage_rows = active_rows[
                (status_code[active_rows] != MEAS_STATUS_PSEUDO)
                & np.isin(meas_type_code[active_rows], _VOLTAGE_MEASUREMENT_TYPE_CODES)
                & (device_pos[active_rows] >= 0)
            ]
            for row in voltage_rows.tolist():
                weight = float(table.weight[row])
                value = float(table.value[row])
                node_idx = self._voltage_measurement_node_idx_from_pos(
                    int(device_type_code[row]),
                    int(device_pos[row]),
                    int(meas_type_code[row]),
                )
                if node_idx is None or self._node_pos_from_idx(node_idx) < 0:
                    continue
                if int(device_type_code[row]) == DEVICE_TYPE_DCNode and int(meas_type_code[row]) == MEAS_TYPE_V:
                    current = node_voltage_best.get(node_idx)
                    if current is None or weight > current[0]:
                        node_voltage_best[node_idx] = (weight, value)
                current = real_voltage_best.get(node_idx)
                if current is None or weight > current[0]:
                    real_voltage_best[node_idx] = (weight, value)
        else:
            max_idx = 0
            warnings.warn(
                "DC SE measurement summary requires a MeasurementTable with device_pos; object fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_voltage_observation_node_cache = {
            node_idx: value for node_idx, (_weight, value) in real_voltage_best.items()
        }

    def _topology_voltage_pseudo_seed(self, i_node: int, j_node: int, i_pos: int) -> float:
        """Pick a voltage seed for DC zero-impedance topology pseudo measurements."""
        for node_idx in (i_node, j_node):
            measured = self._real_voltage_observation_value_for_node(node_idx)
            if measured is not None:
                return max(float(measured), self.voltage_floor)
        if 0 <= int(i_pos) < self._node_voltage_by_pos.size:
            return max(float(self._node_voltage_by_pos[int(i_pos)]), self.voltage_floor)
        return 1.0

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak P/V priors for unmeasured DC topology-device states."""
        measured_keys = self._active_measurement_key_cache
        added_keys = set()
        topology_weight = float(self.pseudo_measurement_weight) * 1e-4
        store_strings = not bool(getattr(self, "_array_only_runtime", False))
        capacity = 2 * (self._zero_branch_names.size + self._break_names.size)
        pseudo_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_meas_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_device_name_ids = np.empty(capacity, dtype=np.int64)
        pseudo_device_positions = np.empty(capacity, dtype=np.int64)
        pseudo_meas_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_values = np.empty(capacity, dtype=np.float64)
        pseudo_weights = np.empty(capacity, dtype=np.float64)
        pseudo_count = 0

        def active_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
            return self._active_measurement_pos_key(device_type_code, device_pos, meas_type_code)

        def queue_topology_pseudo(
            device_type_code: int,
            device_type: str,
            device_name: str,
            device_pos: int,
            meas_type: str,
            meas_type_code: int,
            value: float,
        ) -> None:
            nonlocal pseudo_count
            if int(device_pos) < 0:
                warnings.warn(
                    f"DC SE skipped topology pseudo measurement for {device_type}:{device_name}; device_pos is missing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            key = active_key(device_type_code, device_pos, meas_type_code)
            if key >= 0 and (key in measured_keys or key in added_keys):
                return
            device_name = str(device_name) if store_strings else ""
            row = pseudo_count
            if store_strings:
                pseudo_names[row] = f"pseudo_{meas_type.lower()}_{device_name}"
                pseudo_device_types[row] = device_type
                pseudo_device_names[row] = device_name
                pseudo_meas_types[row] = meas_type
            pseudo_device_type_codes[row] = int(device_type_code)
            pseudo_device_name_ids[row] = -1
            pseudo_device_positions[row] = int(device_pos)
            pseudo_meas_type_codes[row] = int(meas_type_code)
            pseudo_values[row] = float(value)
            pseudo_weights[row] = topology_weight
            pseudo_count = row + 1
            if key >= 0:
                added_keys.add(key)

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
            for device_pos, name, i_node_value, j_node_value, i_pos_value, p_value in zip(
                np.arange(names.size, dtype=np.int64).tolist(),
                names.tolist(),
                i_node.tolist(),
                j_node.tolist(),
                i_pos.tolist(),
                np.asarray(p_values, dtype=np.float64).tolist(),
            ):
                name = str(name)
                has_terminal_measurement = False
                for meas_type_code in (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO):
                    key = active_key(device_type_code, int(device_pos), meas_type_code)
                    if key >= 0 and key in measured_keys:
                        has_terminal_measurement = True
                        break
                if not has_terminal_measurement:
                    queue_topology_pseudo(
                        device_type_code,
                        device_type,
                        name,
                        int(device_pos),
                        "P_FROM",
                        MEAS_TYPE_P_FROM,
                        float(p_value),
                    )
                if self._real_voltage_observation_value_for_node(int(i_node_value)) is None:
                    queue_topology_pseudo(
                        device_type_code,
                        device_type,
                        name,
                        int(device_pos),
                        "V_FROM",
                        MEAS_TYPE_V_FROM,
                        self._topology_voltage_pseudo_seed(int(i_node_value), int(j_node_value), int(i_pos_value)),
                    )
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_meas_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_values[:pseudo_count],
            weights=pseudo_weights[:pseudo_count],
            device_type_codes=pseudo_device_type_codes[:pseudo_count],
            device_name_ids=pseudo_device_name_ids[:pseudo_count],
            meas_type_codes=pseudo_meas_type_codes[:pseudo_count],
            device_positions=pseudo_device_positions[:pseudo_count],
        )
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        if not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        measured_keys = self._active_measurement_key_cache
        next_idx = self._next_measurement_idx()
        next_idx, topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        added_keys = set(topology_added_keys)
        store_strings = not bool(getattr(self, "_array_only_runtime", False))
        capacity = 2 * int(self._generator_names.size) + 2 * int(self._load_names.size) + 4 * int(self._dcdc_names.size)
        pseudo_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_names = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_meas_types = np.empty(capacity, dtype=object) if store_strings else np.asarray([], dtype=object)
        pseudo_device_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_device_name_ids = np.empty(capacity, dtype=np.int64)
        pseudo_device_positions = np.empty(capacity, dtype=np.int64)
        pseudo_meas_type_codes = np.empty(capacity, dtype=np.int16)
        pseudo_values = np.empty(capacity, dtype=np.float64)
        pseudo_count = 0

        def measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
            return self._active_measurement_pos_key(device_type_code, device_pos, meas_type_code)

        def has_measurement(device_type_code: int, device_pos: int, meas_type_code: int) -> bool:
            if int(device_pos) < 0:
                return False
            key = measurement_key(device_type_code, device_pos, meas_type_code)
            return key in measured_keys or key in added_keys

        def queue_pseudo(
            device_type: str,
            device_type_code: int,
            device_name: str,
            device_pos: int,
            meas_type: str,
            meas_type_code: int,
            value: float,
        ) -> None:
            nonlocal pseudo_count
            if int(device_pos) < 0:
                warnings.warn(
                    f"DC SE skipped pseudo measurement for {device_type}:{device_name}; device_pos is missing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            key = measurement_key(device_type_code, device_pos, meas_type_code)
            if key in measured_keys or key in added_keys:
                return
            row_device_name = str(device_name) if store_strings else ""
            row = pseudo_count
            if store_strings:
                pseudo_names[row] = f"pseudo_{meas_type.lower()}_{row_device_name}"
                pseudo_device_types[row] = device_type
                pseudo_device_names[row] = row_device_name
                pseudo_meas_types[row] = meas_type
            pseudo_device_type_codes[row] = int(device_type_code)
            pseudo_device_name_ids[row] = -1
            pseudo_device_positions[row] = int(device_pos)
            pseudo_meas_type_codes[row] = int(meas_type_code)
            pseudo_values[row] = float(value)
            pseudo_count = row + 1
            added_keys.add(key)

        gen_rows = self._generator_rows.astype(np.intp, copy=False)
        gen_table = self._gen_table
        for device_pos in range(int(self._generator_names.size)):
            name = str(self._generator_names[device_pos])
            node_idx = int(self._generator_node[device_pos])
            node_pos = int(self._generator_pos[device_pos])
            row = int(gen_rows[device_pos])
            p_value = float(gen_table[row, DC_GEN_COLS["p"]])
            if p_value == 0.0:
                ctrl = int(gen_table[row, DC_GEN_COLS["control_type"]])
                if ctrl == DC_CTRL_I and 0 <= int(node_pos) < self._node_voltage_by_pos.size:
                    p_value = float(gen_table[row, DC_GEN_COLS["i_set"]] * self._node_voltage_by_pos[node_pos])
                else:
                    p_value = float(gen_table[row, DC_GEN_COLS["p_set"]])
            queue_pseudo(
                "DCGenerator",
                DEVICE_TYPE_DCGenerator,
                name,
                device_pos,
                "P_GEN",
                MEAS_TYPE_P_GEN,
                p_value,
            )
            if (
                not has_measurement(DEVICE_TYPE_DCGenerator, device_pos, MEAS_TYPE_V_GEN)
                and self._real_voltage_observation_value_for_node(int(node_idx)) is None
            ):
                queue_pseudo(
                    "DCGenerator",
                    DEVICE_TYPE_DCGenerator,
                    name,
                    device_pos,
                    "V_GEN",
                    MEAS_TYPE_V_GEN,
                    float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0),
                )

        load_rows = self._load_rows.astype(np.intp, copy=False)
        load_table = self._load_table
        for device_pos in range(int(self._load_names.size)):
            name = str(self._load_names[device_pos])
            node_idx = int(self._load_node[device_pos])
            node_pos = int(self._load_pos[device_pos])
            row = int(load_rows[device_pos])
            voltage = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            p_value = float(load_table[row, DC_LOAD_COLS["p"]])
            if p_value == 0.0:
                pbase = float(load_table[row, DC_LOAD_COLS["pbase"]])
                p_value = pbase * (
                    float(load_table[row, DC_LOAD_COLS["pv0"]])
                    + float(load_table[row, DC_LOAD_COLS["pv1"]]) * voltage
                    + float(load_table[row, DC_LOAD_COLS["pv2"]]) * voltage * voltage
                )
            queue_pseudo(
                "DCLoad",
                DEVICE_TYPE_DCLoad,
                name,
                device_pos,
                "P_LOAD",
                MEAS_TYPE_P_LOAD,
                p_value,
            )
            if (
                not has_measurement(DEVICE_TYPE_DCLoad, device_pos, MEAS_TYPE_V_LOAD)
                and self._real_voltage_observation_value_for_node(int(node_idx)) is None
            ):
                queue_pseudo(
                    "DCLoad",
                    DEVICE_TYPE_DCLoad,
                    name,
                    device_pos,
                    "V_LOAD",
                    MEAS_TYPE_V_LOAD,
                    voltage,
                )

        dcdc_table = self._dc_ppc["dcdc"]
        dcdc_rows = self._dcdc_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._dcdc_names.size)):
            name = str(self._dcdc_names[device_pos])
            i_node = int(self._dcdc_i_node[device_pos])
            j_node = int(self._dcdc_j_node[device_pos])
            i_pos = int(self._dcdc_i_pos[device_pos])
            j_pos = int(self._dcdc_j_pos[device_pos])
            row = int(dcdc_rows[device_pos])
            queue_pseudo(
                "DCDCConverter",
                DEVICE_TYPE_DCDCConverter,
                name,
                device_pos,
                "P_FROM",
                MEAS_TYPE_P_FROM,
                float(dcdc_table[row, DC_DCDC_COLS["i_p"]]),
            )
            queue_pseudo(
                "DCDCConverter",
                DEVICE_TYPE_DCDCConverter,
                name,
                device_pos,
                "P_TO",
                MEAS_TYPE_P_TO,
                float(dcdc_table[row, DC_DCDC_COLS["j_p"]]),
            )
            if (
                not has_measurement(
                    DEVICE_TYPE_DCDCConverter,
                    device_pos,
                    MEAS_TYPE_V_FROM,
                )
                and self._real_voltage_observation_value_for_node(int(i_node)) is None
            ):
                queue_pseudo(
                    "DCDCConverter",
                    DEVICE_TYPE_DCDCConverter,
                    name,
                    device_pos,
                    "V_FROM",
                    MEAS_TYPE_V_FROM,
                    float(self._node_voltage_by_pos[i_pos] if 0 <= i_pos < self._node_voltage_by_pos.size else 1.0),
                )
            if (
                not has_measurement(
                    DEVICE_TYPE_DCDCConverter,
                    device_pos,
                    MEAS_TYPE_V_TO,
                )
                and self._real_voltage_observation_value_for_node(int(j_node)) is None
            ):
                queue_pseudo(
                    "DCDCConverter",
                    DEVICE_TYPE_DCDCConverter,
                    name,
                    device_pos,
                    "V_TO",
                    MEAS_TYPE_V_TO,
                    float(self._node_voltage_by_pos[j_pos] if 0 <= j_pos < self._node_voltage_by_pos.size else 1.0),
                )
        next_idx = self._append_pseudo_measurement_rows(
            next_idx,
            pseudo_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_device_names[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_meas_types[:pseudo_count] if store_strings else np.asarray([], dtype=object),
            pseudo_values[:pseudo_count],
            device_type_codes=pseudo_device_type_codes[:pseudo_count],
            device_name_ids=pseudo_device_name_ids[:pseudo_count],
            meas_type_codes=pseudo_meas_type_codes[:pseudo_count],
            device_positions=pseudo_device_positions[:pseudo_count],
        )
        if added_keys:
            measured_keys.update(added_keys)

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch DC rank deficiencies and optional post-observability redundancy."""
        total_added = 0
        max_count = max(0, int(self.targeted_pseudo_measurement_max))
        step = max(1, int(getattr(self, "targeted_pseudo_measurement_step", 10)))
        redundancy_target = targeted_redundancy_count(
            getattr(self, "n_state", 0),
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        )
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
                continue
            next_idx = self._next_measurement_idx()
            existing_keys = self._active_measurement_keys()
            added = 0
            refreshed = False
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
                break
            total_added += added
            if not refreshed:
                self._refresh_active_measurement_indexes()
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
        candidate_table = getattr(candidates, "table", None)
        if candidate_table is None:
            return 0
        selected_table = measurement_table_take(candidate_table, selected_rows)
        selected_count = int(selected_table.idx.size)
        self._append_pseudo_measurement_rows(
            next_idx,
            selected_table.name,
            selected_table.device_type,
            selected_table.device_name,
            selected_table.meas_type,
            selected_table.value,
            weights=selected_table.weight,
            device_type_codes=selected_table.device_type_code,
            device_name_ids=selected_table.device_name_id,
            meas_type_codes=selected_table.meas_type_code,
            device_positions=selected_table.device_pos,
        )
        if refresh:
            self._refresh_active_measurement_indexes()
        return selected_count

    def _add_redundant_observability_pseudo_measurements(self, max_add: int, refresh: bool = True) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add, refresh)

    def _observability_pseudo_candidate_measurements(self) -> MeasurementTableView:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        existing_keys = self._active_measurement_keys()
        candidate_keys = set()
        store_strings = not bool(getattr(self, "_array_only_runtime", False))
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
        device_name_ids = np.full(capacity, -1, dtype=np.int64)
        device_positions = np.empty(capacity, dtype=np.int64)
        meas_type_codes = np.empty(capacity, dtype=np.int16)
        values = np.empty(capacity, dtype=np.float64)
        count = 0

        def add(
            device_type_code: int,
            device_type: str,
            device_name: str,
            device_pos: int,
            meas_type_code: int,
            meas_type: str,
            value: float,
        ) -> None:
            nonlocal count
            device_type_code = int(device_type_code)
            device_pos = int(device_pos)
            meas_type_code = int(meas_type_code)
            if device_type_code <= 0 or device_pos < 0 or meas_type_code <= 0:
                return
            key = self._active_measurement_pos_key(device_type_code, device_pos, meas_type_code)
            if (
                meas_type_code in _VOLTAGE_MEASUREMENT_TYPE_CODES
                and self._voltage_pseudo_is_covered_by_pos(device_type_code, device_pos, meas_type_code)
            ):
                return
            if key in existing_keys or key in candidate_keys:
                return
            row = count
            if store_strings:
                names[row] = f"pseudo_obs_{meas_type.lower()}_{device_name}"
                device_types[row] = device_type
                device_names[row] = str(device_name)
                meas_types[row] = meas_type
            device_type_codes[row] = device_type_code
            device_positions[row] = device_pos
            meas_type_codes[row] = meas_type_code
            values[row] = float(value)
            count = row + 1
            candidate_keys.add(key)

        for device_pos, (name, node_pos) in enumerate(
            zip(self._raw_node_names_alive.tolist(), self._raw_node_solver_pos_alive.tolist())
        ):
            node_pos = int(node_pos)
            voltage = float(self._node_voltage_by_pos[node_pos] if 0 <= node_pos < self._node_voltage_by_pos.size else 1.0)
            add(DEVICE_TYPE_DCNode, "DCNode", str(name), int(device_pos), MEAS_TYPE_V, "V", voltage)

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
            name = str(self._load_names[device_pos])
            add(DEVICE_TYPE_DCLoad, "DCLoad", name, device_pos, MEAS_TYPE_P_LOAD, "P_LOAD", p_value)
            add(DEVICE_TYPE_DCLoad, "DCLoad", name, device_pos, MEAS_TYPE_V_LOAD, "V_LOAD", voltage)

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
            name = str(self._generator_names[device_pos])
            add(DEVICE_TYPE_DCGenerator, "DCGenerator", name, device_pos, MEAS_TYPE_P_GEN, "P_GEN", p_value)
            add(DEVICE_TYPE_DCGenerator, "DCGenerator", name, device_pos, MEAS_TYPE_V_GEN, "V_GEN", voltage)

        branch_table = np.asarray(self._dc_ppc.get("branch", np.zeros((0, len(DC_BRANCH_COLS)))), dtype=np.float64)
        branch_rows = self._branch_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._branch_names.size)):
            row = int(branch_rows[device_pos])
            if 0 <= row < branch_table.shape[0]:
                name = str(self._branch_names[device_pos])
                add(DEVICE_TYPE_DCBranch, "DCBranch", name, device_pos, MEAS_TYPE_P_FROM, "P_FROM", float(branch_table[row, DC_BRANCH_COLS["i_p"]]))
                add(DEVICE_TYPE_DCBranch, "DCBranch", name, device_pos, MEAS_TYPE_P_TO, "P_TO", float(branch_table[row, DC_BRANCH_COLS["j_p"]]))

        dcdc_table = np.asarray(self._dc_ppc.get("dcdc", np.zeros((0, len(DC_DCDC_COLS)))), dtype=np.float64)
        dcdc_rows = self._dcdc_rows.astype(np.intp, copy=False)
        for device_pos in range(int(self._dcdc_names.size)):
            row = int(dcdc_rows[device_pos])
            if 0 <= row < dcdc_table.shape[0]:
                name = str(self._dcdc_names[device_pos])
                add(DEVICE_TYPE_DCDCConverter, "DCDCConverter", name, device_pos, MEAS_TYPE_P_FROM, "P_FROM", float(dcdc_table[row, DC_DCDC_COLS["i_p"]]))
                add(DEVICE_TYPE_DCDCConverter, "DCDCConverter", name, device_pos, MEAS_TYPE_P_TO, "P_TO", float(dcdc_table[row, DC_DCDC_COLS["j_p"]]))

        candidate_table = self._pseudo_measurement_table(
            names[:count] if store_strings else np.asarray([], dtype=object),
            device_types[:count] if store_strings else np.asarray([], dtype=object),
            device_names[:count] if store_strings else np.asarray([], dtype=object),
            meas_types[:count] if store_strings else np.asarray([], dtype=object),
            values[:count],
            self.pseudo_measurement_weight,
            device_type_codes=device_type_codes[:count],
            device_name_ids=device_name_ids[:count],
            meas_type_codes=meas_type_codes[:count],
            device_positions=device_positions[:count],
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
        cache = self._observability_matrix_cache_for(observability, self.active_measurements, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, self.active_measurements)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return np.arange(min(int(max_add), candidate_count), dtype=np.int64)
        candidate_h = self.jacobian_sparse(x, candidates)
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
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

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
        max_add: int,
    ) -> Tuple[int, int]:
        """Translate a weak compact DC state into the smallest useful pseudo measurement."""
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
            key = self._active_measurement_pos_key(target_device_type_code, target_device_pos, target_meas_type_code)
            if (
                target_meas_type_code in _VOLTAGE_MEASUREMENT_TYPE_CODES
                and self._voltage_pseudo_is_covered_by_pos(
                    target_device_type_code,
                    target_device_pos,
                    target_meas_type_code,
                )
            ):
                return next_idx, 0
            if key in existing_keys:
                return next_idx, 0
            store_strings = not bool(getattr(self, "_array_only_runtime", False))
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}" if store_strings else ""
            new_idx = self._append_pseudo_measurement_rows(
                next_idx,
                np.asarray([pseudo_name], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([device_type], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([device_name], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([meas_type], dtype=object) if store_strings else np.asarray([], dtype=object),
                np.asarray([float(value)], dtype=np.float64),
                device_type_codes=np.asarray([target_device_type_code], dtype=np.int16),
                device_name_ids=np.asarray([-1], dtype=np.int64),
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

    def _measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        key = id(measurements)
        cached = self._measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        self._ensure_measurement_plan_lookup_arrays()
        plan_table = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=self._measurement_plan_device_pos_by_type_code,
            device_pos_by_type_code_id=getattr(self, "_measurement_plan_device_pos_by_type_code_id", {}),
            meas_kind_by_type_code=self._measurement_plan_meas_kind_by_type_code,
            meas_kind_code_by_type_code=getattr(self, "_measurement_plan_meas_kind_code_by_type_code", {}),
            require_index_arrays=True,
            table_builder=self._measurement_table_for_indexed_plan,
        )
        handled = np.asarray(plan_table.handled, dtype=bool).copy()
        row = plan_table.row
        device_type_code = plan_table.device_type_code
        meas_kind = plan_table.meas_kind
        device_pos = plan_table.device_pos

        node_code = DEVICE_TYPE_DCNode
        branch_code = DEVICE_TYPE_DCBranch
        load_code = DEVICE_TYPE_DCLoad
        gen_code = DEVICE_TYPE_DCGenerator
        zero_branch_code = DEVICE_TYPE_DCZeroBranch
        break_code = DEVICE_TYPE_DCBreak
        zero_constraint_code = DEVICE_TYPE_DCZeroBranchConstraint
        break_constraint_code = DEVICE_TYPE_DCBreakConstraint
        dcdc_code = DEVICE_TYPE_DCDCConverter

        node_rows = row[(device_type_code == node_code) & handled]
        node_device_pos = device_pos[node_rows]
        node_pos = self._node_plan_node_pos[node_device_pos]
        node_col = self._node_plan_col[node_device_pos]

        branch_rows = row[(device_type_code == branch_code) & handled]
        branch_device_pos = device_pos[branch_rows]
        branch_kind = meas_kind[branch_rows]
        branch_i = self._branch_plan_i[branch_device_pos]
        branch_j = self._branch_plan_j[branch_device_pos]
        branch_i_col = self._branch_plan_i_col[branch_device_pos]
        branch_j_col = self._branch_plan_j_col[branch_device_pos]
        branch_inv_r = self._branch_plan_inv_r[branch_device_pos]

        load_rows = row[(device_type_code == load_code) & handled]
        load_device_pos = device_pos[load_rows]
        load_kind = meas_kind[load_rows]
        load_pos = self._load_plan_pos[load_device_pos]
        load_col = self._load_plan_col[load_device_pos]
        load_pv0 = self._load_plan_pv0[load_device_pos]
        load_pv1 = self._load_plan_pv1[load_device_pos]
        load_pv2 = self._load_plan_pv2[load_device_pos]

        gen_rows = row[(device_type_code == gen_code) & handled]
        gen_device_pos = device_pos[gen_rows]
        gen_kind = meas_kind[gen_rows]
        gen_ctrl = self._generator_plan_ctrl[gen_device_pos]
        gen_pos = self._generator_plan_pos[gen_device_pos]
        gen_col = self._generator_plan_col[gen_device_pos]
        gen_p_col = self._generator_plan_p_col[gen_device_pos]
        gen_vgen_pos = self._generator_plan_vgen_pos[gen_device_pos]
        gen_p_set = self._generator_plan_p_set[gen_device_pos]
        gen_i_set = self._generator_plan_i_set[gen_device_pos]

        def _switch_plan_for_code(
            code: int,
            plan_i: np.ndarray,
            plan_j: np.ndarray,
            plan_i_col: np.ndarray,
            plan_j_col: np.ndarray,
            plan_current_col: np.ndarray,
            plan_current_pos: np.ndarray,
        ) -> Tuple[np.ndarray, ...]:
            rows = row[(device_type_code == code) & handled]
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

        zero_switch = _switch_plan_for_code(
            zero_branch_code,
            self._zero_branch_plan_i,
            self._zero_branch_plan_j,
            self._zero_branch_plan_i_col,
            self._zero_branch_plan_j_col,
            self._zero_branch_plan_current_col,
            self._zero_branch_plan_current_pos,
        )
        break_switch = _switch_plan_for_code(
            break_code,
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

        constraint_rows = np.concatenate(
            (
                row[(device_type_code == zero_constraint_code) & handled],
                row[(device_type_code == break_constraint_code) & handled],
            )
        ).astype(np.int64, copy=False)
        if constraint_rows.size:
            constraint_order = np.argsort(constraint_rows)
            constraint_rows = constraint_rows[constraint_order]
        constraint_device_pos = device_pos[constraint_rows]
        constraint_i = self._constraint_plan_i[constraint_device_pos]
        constraint_j = self._constraint_plan_j[constraint_device_pos]
        constraint_i_col = self._constraint_plan_i_col[constraint_device_pos]
        constraint_j_col = self._constraint_plan_j_col[constraint_device_pos]

        dcdc_rows = row[(device_type_code == dcdc_code) & handled]
        dcdc_device_pos = device_pos[dcdc_rows]
        dcdc_kind = meas_kind[dcdc_rows]
        dcdc_i = self._dcdc_plan_i[dcdc_device_pos]
        dcdc_j = self._dcdc_plan_j[dcdc_device_pos]
        dcdc_i_col = self._dcdc_plan_i_col[dcdc_device_pos]
        dcdc_j_col = self._dcdc_plan_j_col[dcdc_device_pos]
        dcdc_p_col = self._dcdc_plan_p_col[dcdc_device_pos]
        dcdc_q_col = self._dcdc_plan_q_col[dcdc_device_pos]
        dcdc_pos = self._dcdc_plan_pos[dcdc_device_pos]

        plan = {
            "handled_mask": handled,
            "node_rows": self._int_array(node_rows),
            "node_pos": self._int_array(node_pos),
            "node_col": self._int_array(node_col),
            "branch_rows": self._int_array(branch_rows),
            "branch_kind": self._int_array(branch_kind),
            "branch_i": self._int_array(branch_i),
            "branch_j": self._int_array(branch_j),
            "branch_i_col": self._int_array(branch_i_col),
            "branch_j_col": self._int_array(branch_j_col),
            "branch_inv_r": np.asarray(branch_inv_r, dtype=np.float64),
            "load_rows": self._int_array(load_rows),
            "load_kind": self._int_array(load_kind),
            "load_pos": self._int_array(load_pos),
            "load_col": self._int_array(load_col),
            "load_pv0": np.asarray(load_pv0, dtype=np.float64),
            "load_pv1": np.asarray(load_pv1, dtype=np.float64),
            "load_pv2": np.asarray(load_pv2, dtype=np.float64),
            "gen_rows": self._int_array(gen_rows),
            "gen_kind": self._int_array(gen_kind),
            "gen_ctrl": self._int_array(gen_ctrl),
            "gen_pos": self._int_array(gen_pos),
            "gen_col": self._int_array(gen_col),
            "gen_p_col": self._int_array(gen_p_col),
            "gen_vgen_pos": self._int_array(gen_vgen_pos),
            "gen_p_set": np.asarray(gen_p_set, dtype=np.float64),
            "gen_i_set": np.asarray(gen_i_set, dtype=np.float64),
            "switch_rows": self._int_array(switch_rows),
            "switch_kind": self._int_array(switch_kind),
            "switch_i": self._int_array(switch_i),
            "switch_j": self._int_array(switch_j),
            "switch_i_col": self._int_array(switch_i_col),
            "switch_j_col": self._int_array(switch_j_col),
            "switch_col": self._int_array(switch_col),
            "switch_pos": self._int_array(switch_pos),
            "constraint_rows": self._int_array(constraint_rows),
            "constraint_i": self._int_array(constraint_i),
            "constraint_j": self._int_array(constraint_j),
            "constraint_i_col": self._int_array(constraint_i_col),
            "constraint_j_col": self._int_array(constraint_j_col),
            "dcdc_rows": self._int_array(dcdc_rows),
            "dcdc_kind": self._int_array(dcdc_kind),
            "dcdc_i": self._int_array(dcdc_i),
            "dcdc_j": self._int_array(dcdc_j),
            "dcdc_i_col": self._int_array(dcdc_i_col),
            "dcdc_j_col": self._int_array(dcdc_j_col),
            "dcdc_p_col": self._int_array(dcdc_p_col),
            "dcdc_q_col": self._int_array(dcdc_q_col),
            "dcdc_pos": self._int_array(dcdc_pos),
        }
        # Precompute per-kind bool masks once. The fill_* helpers branch on
        # `kind == k` for each k in the relevant range; without these caches
        # every iteration re-evaluates the comparison even though `kind` is
        # immutable. Each mask is len(section_rows) so storage is trivial.
        self._populate_kind_masks(plan)
        if len(self._measurement_plan_cache) > 16:
            self._measurement_plan_cache.clear()
        self._measurement_plan_cache[key] = (measurements, plan)
        if hasattr(self, "active_measurements") and measurements is self.active_measurements:
            self._active_measurement_plan = plan
        return plan

    def _fill_measurement_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        switch_current: np.ndarray,
        dcdc_power: np.ndarray,
        v_generator_power: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate common DC measurement rows without per-device Python calls."""
        plan = self._measurement_plan(measurements)

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
        measurements: Optional[Sequence[Measurement]] = None,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate h(x): estimated values for each active DC measurement."""
        measurements = self._normalize_measurements(measurements)
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        n_meas = len(measurements)
        if out is None:
            values = np.zeros(n_meas, dtype=np.float64)
        else:
            values = out
            values.fill(0.0)
        vectorized_rows = self._fill_measurement_values_vectorized(
            values,
            measurements,
            voltage,
            switch_current,
            dcdc_power,
            v_generator_power,
        )
        if not np.all(vectorized_rows):
            missing = int(vectorized_rows.size - np.count_nonzero(vectorized_rows))
            warnings.warn(
                f"DC SE evaluate skipped {missing} non-vectorized measurement rows; object/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        return values

    @staticmethod
    def _add_indexed_values(
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

    @staticmethod
    def _add_derivative(H, row: int, col: int, value: float) -> None:
        if col < 0:
            return
        if hasattr(H, "add"):
            H.add(row, col, value)
        else:
            H[row, col] += value

    def _fill_jacobian_vectorized(
        self,
        H,
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        switch_current: np.ndarray,
        dcdc_power: np.ndarray,
        v_generator_power: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse/dense DC measurement Jacobian rows."""
        plan = self._measurement_plan(measurements)

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
        measurements: Optional[Sequence[Measurement]] = None,
        sparse: bool = False,
    ):
        """Assemble analytical DC measurement sensitivities for WLS."""
        measurements = self._normalize_measurements(measurements)
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        if sparse:
            if measurements is self.active_measurements:
                H = self._jacobian_builder
                H.reset()
            else:
                # Cache a fixed-pattern builder per `id(measurements)` so repeated
                # calls (e.g. from a hybrid parent) reuse the CSR pattern instead
                # of rebuilding it each call.
                cache = getattr(self, "_external_jacobian_builder_cache", None)
                if cache is None:
                    cache = {}
                    self._external_jacobian_builder_cache = cache
                key = id(measurements)
                cached = cache.get(key)
                if cached is not None and cached[0] is measurements:
                    H = cached[1]
                    H.reset()
                else:
                    H = SparseJacobianBuilder((len(measurements), self.n_state))
                    H._assume_fixed_pattern = True
                    if len(cache) > 4:
                        cache.clear()
                    cache[key] = (measurements, H)
        else:
            H = np.zeros((len(measurements), self.n_state), dtype=np.float64)
        vectorized_rows = self._fill_jacobian_vectorized(
            H,
            measurements,
            voltage,
            switch_current,
            dcdc_power,
            v_generator_power,
        )
        if not np.all(vectorized_rows):
            missing = int(vectorized_rows.size - np.count_nonzero(vectorized_rows))
            warnings.warn(
                f"DC SE jacobian skipped {missing} non-vectorized measurement rows; object/string fallback is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        return H.to_csr() if sparse else H

    def jacobian(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Assemble analytical DC measurement sensitivities as a dense array."""
        return self._assemble_jacobian(x, measurements, sparse=False)

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None):
        """Assemble analytical DC measurement sensitivities directly as sparse CSR."""
        return self._assemble_jacobian(x, measurements, sparse=True)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        """Use singular values of H to locate unobservable DC state combinations."""
        x = self.initial_state() if x is None else x
        measurements = self._normalize_measurements(measurements)
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            return ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])

        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.n_state,
            normal_matrix=normal_matrix,
            normal_factor_diag=normal_factor_diag,
        )
        result = ObservabilityResult(
            observable=rank == self.n_state,
            rank=rank,
            state_count=self.n_state,
            measurement_count=len(measurements),
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )
        self._cache_observability_matrix(result, x, measurements, H)
        return result

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        """Solve the weighted least-squares DC state estimate with damped Newton steps."""
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        measurements = self._normalize_measurements(measurements)
        if len(measurements) < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {len(measurements)} < {self.n_state}")

        x = self.initial_state() if x0 is None else x0.copy()
        if observability is None:
            if measurements is self.active_measurements and x0 is None:
                observability = self.observability_analysis()
            else:
                observability = self.observability_analysis(x, measurements)
        observability_cache = self._observability_matrix_cache_for(observability, measurements, x)
        cached_initial_H = observability_cache.get("H") if observability_cache is not None else None
        z, weight = self._measurement_vectors(measurements)
        uniform_weight = self.active_uniform_weight if measurements is self.active_measurements else self._uniform_weight(weight)
        weights_are_uniform = self.active_weights_are_uniform if measurements is self.active_measurements else uniform_weight is not None
        weighted_residual = None if weights_are_uniform else np.empty_like(weight)
        converged = False
        max_correction = np.inf
        objective = np.inf
        iteration = 0
        H = None
        gain = np.zeros((self.n_state, self.n_state), dtype=np.float64)
        final_quantities_current = False
        normal_solver = NormalEquationSolver(assume_fixed_pattern=measurements is self.active_measurements)
        normal_pattern = self._active_normal_pattern if measurements is self.active_measurements else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]

        if verbose:
            _print_iteration_header()

        # Pre-allocate evaluation buffers reused across iterations and line-search
        # candidates. On acceptance we swap pointers so the accepted vectors
        # become the "main" buffers without copying.
        n_meas = len(measurements)
        z_est = np.empty(n_meas, dtype=np.float64)
        residual = np.empty(n_meas, dtype=np.float64)
        cand_z_est = np.empty(n_meas, dtype=np.float64)
        cand_residual = np.empty(n_meas, dtype=np.float64)

        for iteration in range(1, self.max_iter + 1):
            self.evaluate(x, measurements, out=z_est)
            np.subtract(z, z_est, out=residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            objective = 0.5 * float(np.dot(weight * residual, residual))
            if iteration == 1 and cached_initial_H is not None:
                H = cached_initial_H
                cached_initial_H = None
            else:
                H = self.jacobian_sparse(x, measurements)
            if normal_pattern is None and is_sparse_matrix(H):
                normal_pattern = _normal_equation_structural_pattern(H)
                if measurements is self.active_measurements:
                    self._active_normal_pattern = normal_pattern
                if observability_cache is not None:
                    observability_cache["normal_pattern"] = normal_pattern
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            gain, rhs = build_normal_equations(
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=normal_pattern,
                assume_normal_pattern_matches=measurements is self.active_measurements,
            )
            dx, _ = normal_solver.solve(gain, rhs, return_factor_diag=False)

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
            # Try shorter steps when the full Gauss-Newton update raises the objective.
            for _ in range(8):
                candidate = x + step_scale * dx
                candidate[: self.n_voltage] = np.maximum(candidate[: self.n_voltage], self.voltage_floor)
                self.evaluate(candidate, measurements, out=cand_z_est)
                np.subtract(z, cand_z_est, out=cand_residual)
                candidate_objective = 0.5 * float(np.dot(weight * cand_residual, cand_residual))
                if candidate_objective <= objective or step_scale < 1e-3:
                    x = candidate
                    objective = candidate_objective
                    accepted_step = step_scale
                    accepted = True
                    # Swap so the accepted candidate becomes the "main" buffer.
                    z_est, cand_z_est = cand_z_est, z_est
                    residual, cand_residual = cand_residual, residual
                    break
                step_scale *= 0.5
            if not accepted:
                x += dx
                x[: self.n_voltage] = np.maximum(x[: self.n_voltage], self.voltage_floor)
                accepted_step = 1.0

            if verbose:
                self.evaluate(x, measurements, out=cand_z_est)
                np.subtract(z, cand_z_est, out=cand_residual)
                updated_residual_inf = float(np.linalg.norm(cand_residual, np.inf)) if cand_residual.size else 0.0
                updated_objective = 0.5 * float(np.dot(weight * cand_residual, cand_residual))
                _print_iteration(
                    iteration,
                    updated_objective,
                    updated_residual_inf,
                    max_correction,
                    accepted_step,
                    False,
                )

        if not final_quantities_current:
            self.evaluate(x, measurements, out=z_est)
            np.subtract(z, z_est, out=residual)
            objective = 0.5 * float(np.dot(weight * residual, residual))
            if final_diagnostics:
                H = self.jacobian_sparse(x, measurements)
                if normal_pattern is None and is_sparse_matrix(H):
                    normal_pattern = _normal_equation_structural_pattern(H)
                    if measurements is self.active_measurements:
                        self._active_normal_pattern = normal_pattern
                if weighted_residual is not None:
                    np.multiply(weight, residual, out=weighted_residual)
                gain, _ = build_normal_equations(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    normal_pattern=normal_pattern,
                    assume_normal_pattern_matches=measurements is self.active_measurements,
                )
        if not final_diagnostics:
            H = None
            gain = None
        array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
        if not array_only_result:
            self.apply_state(x)
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
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
            measurements=[] if array_only_result else measurements,
            observability=observability,
            measurement_table=getattr(measurements, "table", None),
        )

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Compute largest normalized residuals after accounting for measurement leverage."""
        profile_start = time.perf_counter() if self.profile_enabled else None
        gain_inverse_start = None
        measurement_table = getattr(result, "measurement_table", None)
        table_weight = getattr(measurement_table, "weight", None)
        if table_weight is not None and np.asarray(table_weight).size == result.residual.size:
            weights = np.asarray(table_weight, dtype=np.float64)
        else:
            weights = np.asarray([meas.weight for meas in result.measurements], dtype=np.float64)
        threshold = self.params.bad_threshold if threshold is None else threshold
        if result.residual.size and threshold > 0.0:
            normalized_upper_bound = np.abs(result.residual) / np.sqrt(1e-12)
            if float(normalized_upper_bound.max()) <= float(threshold):
                self._record_profile_time("bad_data.fast_residual_bound", 0.0)
                if profile_start is not None:
                    self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
                return [], normalized_upper_bound
        jacobian_measurements = (
            result.measurements
            if len(result.measurements) == result.residual.size
            else getattr(self, "active_measurements", result.measurements)
        )
        if result.H is None or result.gain is None:
            result.H = self.jacobian_sparse(result.x, jacobian_measurements)
            result.gain, _ = build_normal_equations(
                result.H,
                result.residual,
                weights,
            )
        R_diag = 1.0 / weights
        if self.profile_enabled:
            gain_inverse_start = time.perf_counter()
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
            meas = (
                measurement_from_table_row(measurement_table, row_pos)
                if measurement_table is not None
                else result.measurements[row_pos]
            )
            measured_value = (
                float(measurement_table.value[row_pos])
                if measurement_table is not None
                else float(meas.value)
            )
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
        elif array_only and not isinstance(getattr(self, "active_measurements", None), MeasurementTableView):
            self._refresh_active_measurement_indexes()
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        needs_bad_data = not skip_bad_data
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
        measurements = self.active_measurements
        removed: List[BadDataItem] = []
        x0 = self.initial_state()
        for round_idx in range(max_remove + 1):
            if verbose:
                print(f"Bad-data removal round {round_idx + 1}: measurements={len(measurements)}")
            result = self.estimate(measurements, x0=x0, verbose=verbose)
            bad_items, _ = self.identify_bad_data(result, threshold)
            if not bad_items:
                return result, removed
            worst = bad_items[0]
            removed.append(worst)
            remove_pos = result.measurements.index(worst.measurement)
            if measurements is self.active_measurements and result.measurements is self.active_measurements:
                measurements = self._shrink_active_measurement_indexes(remove_pos)
            else:
                keep_rows = np.concatenate(
                    (
                        np.arange(remove_pos, dtype=np.int64),
                        np.arange(remove_pos + 1, len(result.measurements), dtype=np.int64),
                    )
                )
                measurements = take_measurement_view(result.measurements, keep_rows)
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
