import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from efile_read import _read_efile_rows
from hybrid_lf import HybridPowerNetwork
from model.hybrid_array_model import (
    ACAC_COLS,
    ACAC_CONTROL_LABEL,
    DCAC_COLS,
    DCAC_CONTROL_LABEL,
)
from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows
from model.meas_array_model import (
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_list_from_meas_ppc,
    sync_meas_ppc_from_measurement_table,
)
from model.meas_type import (
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_DCACConverter,
    MEAS_TYPE_CODES,
    MEAS_TYPE_I_AC,
    MEAS_TYPE_I_DC,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_P_AC,
    MEAS_TYPE_P_DC,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_Q_AC,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_V_AC,
    MEAS_TYPE_V_DC,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_V_TO,
)
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    EstimateResult,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_PSEUDO,
    Measurement,
    MeasurementList,
    ObservabilityResult,
    is_pseudo_measurement,
    mark_measurement_invalid,
    mark_measurement_pseudo,
    measurement_from_table_row,
    measurement_table_from_measurements,
    measurement_table_status_code,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.ac_se import (
    ACStateEstimator,
    _build_ac_se_ppc_namespace,
)
from secore.dc_se import DCStateEstimator, _build_dc_se_ppc_namespace
from secore.se_array_plan import (
    MeasurementPartitions,
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    copy_measurement_view,
    extend_measurement_partitions,
    partition_measurements_by_code,
    take_measurement_view,
)
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    NormalEquationSolver,
    SparseJacobianBuilder,
    _normal_equation_structural_pattern,
    angle_residual_mask,
    build_normal_equations,
    is_sparse_matrix,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    observability_rank_details,
    observability_weak_direction,
    targeted_redundancy_count,
)
from secore.se_result import SEResult, build_seresult_summary, normalize_seresult_result_mode
from secore.state_metadata import StateMeta, state_labels_from_metadata, state_meta_at
from unit_system import ac_current_base_ka, dc_current_base_ka


DEFAULT_CASE = model_file("hybrid", "qinling.e")
DEFAULT_MEAS = measurement_file("hybrid", "qinling.meas")


def _read_measurements_direct(meas_file: Path):
    """Read Measurement rows through the shared PPC parser."""
    return measurement_list_from_meas_ppc(copy_meas_ppc(build_meas_ppc_from_e_file(meas_file)))


def _measurement_table_from_measurements(measurements: Sequence[Measurement]):
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=DEVICE_TYPE_CODES,
        angle_measurement_types=ANGLE_MEASUREMENT_TYPES,
    )


class _SELightweightHybridNetwork(SimpleNamespace):
    @property
    def total_nodes(self) -> int:
        return _side_alive_node_count(getattr(self, "ac", None)) + _side_alive_node_count(getattr(self, "dc", None))

    def prepare(self, verbose: bool = True):
        return [], [], [], []

    def check_topology(self):
        return [], [], [], []


def _array_device(idx, name=None, **values):
    return SimpleNamespace(idx=int(idx), name=str(name if name is not None else idx), **values)


def _name_at(names, pos: int, prefix: str, idx: int) -> str:
    if names is None or pos >= len(names):
        return f"{prefix}_{idx}"
    return str(names[pos])


def _side_topology(side):
    if side is None:
        return None
    topology = getattr(side, "topology", None) or getattr(side, "_topology_arrays", None)
    if topology is not None:
        return topology
    ppc = getattr(side, "ppc", None)
    return ppc.get("_topology_arrays") if isinstance(ppc, dict) else None


def _side_alive_node_count(side) -> int:
    topology = _side_topology(side)
    if topology is not None:
        bus_alive = getattr(topology, "bus_alive_mask", None)
        if bus_alive is not None:
            return int(np.count_nonzero(np.asarray(bus_alive, dtype=bool)))
        node_alive = getattr(topology, "node_alive_mask", None)
        if node_alive is not None:
            return int(np.count_nonzero(np.asarray(node_alive, dtype=bool)))
    nodes = getattr(side, "nodes", None)
    if nodes is not None:
        return sum(1 for node in nodes if getattr(node, "is_alive", True))
    ppc = getattr(side, "ppc", None)
    bus = ppc.get("bus") if isinstance(ppc, dict) else None
    return int(np.asarray(bus).shape[0]) if isinstance(bus, np.ndarray) else 0


def _side_has_alive_nodes(side) -> bool:
    return _side_alive_node_count(side) > 0


def _side_alive_node_lookup(side) -> Dict[int, bool]:
    topology = _side_topology(side)
    if topology is not None:
        node_ids = getattr(topology, "node_ids", None)
        node_alive = getattr(topology, "node_alive_mask", None)
        if node_ids is not None and node_alive is not None:
            ids = np.asarray(node_ids, dtype=np.int64)
            alive = np.asarray(node_alive, dtype=bool)
            if ids.size == alive.size:
                return {int(idx): bool(flag) for idx, flag in zip(ids, alive)}
    node_dict = getattr(side, "node_dict", None)
    if isinstance(node_dict, dict):
        return {int(idx): bool(getattr(node, "is_alive", True)) for idx, node in node_dict.items()}
    return {}


def _build_se_dcac_converters(ppc: Dict) -> List[SimpleNamespace]:
    rows = np.asarray(ppc.get("dcac", np.zeros((0, len(DCAC_COLS)), dtype=np.float64)))
    names = ppc.get("dcac_name")
    converters = []
    for pos, row in enumerate(rows):
        idx = int(row[DCAC_COLS["idx"]])
        converters.append(
            _array_device(
                idx,
                _name_at(names, pos, "dcac", idx),
                ac_node=int(row[DCAC_COLS["ac_node"]]),
                dc_node=int(row[DCAC_COLS["dc_node"]]),
                r1=float(row[DCAC_COLS["r1"]]),
                r2=float(row[DCAC_COLS["r2"]]),
                control_type=DCAC_CONTROL_LABEL.get(int(row[DCAC_COLS["control_type"]]), "DCV"),
                p_ac_set=float(row[DCAC_COLS["p_ac_set"]]),
                q_ac_set=float(row[DCAC_COLS["q_ac_set"]]),
                v_ac_set=float(row[DCAC_COLS["v_ac_set"]]),
                v_dc_set=float(row[DCAC_COLS["v_dc_set"]]),
                run_stat=int(row[DCAC_COLS["run_stat"]]),
                dc_p=float(row[DCAC_COLS["dc_p"]]),
                ac_p=float(row[DCAC_COLS["ac_p"]]),
                ac_q=float(row[DCAC_COLS["ac_q"]]),
                dc_i=float(row[DCAC_COLS["dc_i"]]),
                ac_i=float(row[DCAC_COLS["ac_i"]]),
                ac_node_obj=None,
                dc_node_obj=None,
                ac_isl_obj=None,
                dc_isl_obj=None,
                hybrid_isl=0,
                hybrid_isl_obj=None,
                is_alive=False,
            )
        )
    return converters


def _build_se_acac_converters(ppc: Dict) -> List[SimpleNamespace]:
    rows = np.asarray(ppc.get("acac", np.zeros((0, len(ACAC_COLS)), dtype=np.float64)))
    names = ppc.get("acac_name")
    converters = []
    for pos, row in enumerate(rows):
        idx = int(row[ACAC_COLS["idx"]])
        converters.append(
            _array_device(
                idx,
                _name_at(names, pos, "acac", idx),
                i_node=int(row[ACAC_COLS["i_node"]]),
                j_node=int(row[ACAC_COLS["j_node"]]),
                r1=float(row[ACAC_COLS["r1"]]),
                r2=float(row[ACAC_COLS["r2"]]),
                control_type=ACAC_CONTROL_LABEL.get(int(row[ACAC_COLS["control_type"]]), "PQQ"),
                p_set=float(row[ACAC_COLS["p_set"]]),
                i_q_set=float(row[ACAC_COLS["i_q_set"]]),
                j_q_set=float(row[ACAC_COLS["j_q_set"]]),
                i_v_set=float(row[ACAC_COLS["i_v_set"]]),
                j_v_set=float(row[ACAC_COLS["j_v_set"]]),
                run_stat=int(row[ACAC_COLS["run_stat"]]),
                i_p=float(row[ACAC_COLS["i_p"]]),
                i_q=float(row[ACAC_COLS["i_q"]]),
                j_p=float(row[ACAC_COLS["j_p"]]),
                j_q=float(row[ACAC_COLS["j_q"]]),
                i_i=float(row[ACAC_COLS["i_i"]]),
                j_i=float(row[ACAC_COLS["j_i"]]),
                i_node_obj=None,
                j_node_obj=None,
                i_isl_obj=None,
                j_isl_obj=None,
                hybrid_isl=0,
                hybrid_isl_obj=None,
                is_alive=False,
            )
        )
    return converters


def _assign_se_converter_topology(network: _SELightweightHybridNetwork) -> None:
    ac_nodes = getattr(network.ac, "node_dict", {})
    dc_nodes = getattr(network.dc, "node_dict", {})
    ac_alive = _side_alive_node_lookup(network.ac)
    dc_alive = _side_alive_node_lookup(network.dc)
    for conv in network.dcac_converters:
        ac_node = ac_nodes.get(int(conv.ac_node))
        dc_node = dc_nodes.get(int(conv.dc_node))
        conv.ac_node_obj = ac_node
        conv.dc_node_obj = dc_node
        conv.ac_isl_obj = None if ac_node is None else getattr(ac_node, "isl_obj", None)
        conv.dc_isl_obj = None if dc_node is None else getattr(dc_node, "isl_obj", None)
        conv.is_alive = bool(
            int(getattr(conv, "run_stat", 1)) == 1
            and ac_alive.get(int(conv.ac_node), getattr(ac_node, "is_alive", False))
            and dc_alive.get(int(conv.dc_node), getattr(dc_node, "is_alive", False))
        )
    for conv in network.acac_converters:
        i_node = ac_nodes.get(int(conv.i_node))
        j_node = ac_nodes.get(int(conv.j_node))
        conv.i_node_obj = i_node
        conv.j_node_obj = j_node
        conv.i_isl_obj = None if i_node is None else getattr(i_node, "isl_obj", None)
        conv.j_isl_obj = None if j_node is None else getattr(j_node, "isl_obj", None)
        conv.is_alive = bool(
            int(getattr(conv, "run_stat", 1)) == 1
            and ac_alive.get(int(conv.i_node), getattr(i_node, "is_alive", False))
            and ac_alive.get(int(conv.j_node), getattr(j_node, "is_alive", False))
        )


def _build_hybrid_se_network_from_ppc(ppc: Dict) -> _SELightweightHybridNetwork:
    ac_network = _build_ac_se_ppc_namespace(ppc["ac"], Path(ppc.get("source", "")) if ppc.get("source") else None)
    dc_network = _build_dc_se_ppc_namespace(ppc["dc"])
    network = _SELightweightHybridNetwork(
        _se_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=_build_se_dcac_converters(ppc),
        acac_converters=_build_se_acac_converters(ppc),
        hybrid_islands=[],
        ppc=ppc,
        _ac_ppc=ppc["ac"],
        _dc_ppc=ppc["dc"],
    )
    base = ppc["base"]
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    _assign_se_converter_topology(network)
    return network


class HybridStateEstimator:
    """AC/DC state-estimation orchestrator.

    AC and DC device rows are normalized, evaluated and differentiated by
    ACStateEstimator/DCStateEstimator.  This class owns only measurement
    partitioning, global WLS assembly and converter-side rows.
    """

    _AC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "ACNode",
            "ACBranch",
            "ACTransformer",
            "ACSwitch",
            "ACBreak",
            "ACZeroBranch",
            "ACGenerator",
            "ACLoad",
            "ACPowerBalance",
            "ACZeroBranchConstraint",
            "ACBreakConstraint",
        )
    )
    _DC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "DCNode",
            "DCBranch",
            "DCSwitch",
            "DCBreak",
            "DCZeroBranch",
            "DCZeroBranchConstraint",
            "DCSwitchConstraint",
            "DCBreakConstraint",
            "DCGenerator",
            "DCLoad",
            "DCDCConverter",
            "DCPowerBalance",
        )
    )
    _HYBRID_MEASUREMENT_DEVICE_TYPES = frozenset(("DCACConverter", "ACACConverter"))
    _VOLTAGE_MEASUREMENT_TYPES = frozenset(("V", "V_FROM", "V_TO", "V_GEN", "V_LOAD", "V_DC", "V_AC"))
    _VOLTAGE_MEASUREMENT_TYPE_TUPLE = tuple(_VOLTAGE_MEASUREMENT_TYPES)
    _MEASUREMENT_SIDE_BY_DEVICE_TYPE = {
        **{device_type: "ac" for device_type in _AC_MEASUREMENT_DEVICE_TYPES},
        **{device_type: "dc" for device_type in _DC_MEASUREMENT_DEVICE_TYPES},
        **{device_type: "hybrid" for device_type in _HYBRID_MEASUREMENT_DEVICE_TYPES},
    }
    _MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE = {
        **{
            DEVICE_TYPE_CODES[device_type]: "ac"
            for device_type in _AC_MEASUREMENT_DEVICE_TYPES
            if device_type in DEVICE_TYPE_CODES
        },
        **{
            DEVICE_TYPE_CODES[device_type]: "dc"
            for device_type in _DC_MEASUREMENT_DEVICE_TYPES
            if device_type in DEVICE_TYPE_CODES
        },
        **{
            DEVICE_TYPE_CODES[device_type]: "hybrid"
            for device_type in _HYBRID_MEASUREMENT_DEVICE_TYPES
            if device_type in DEVICE_TYPE_CODES
        },
    }
    _DCAC_MEASUREMENT_CODE = {
        "P_DC": 1,
        "P_AC": 2,
        "Q_AC": 3,
        "V_DC": 4,
        "I_DC": 5,
        "V_AC": 6,
        "I_AC": 7,
    }
    _ACAC_MEASUREMENT_CODE = {
        "P_FROM": 1,
        "Q_FROM": 2,
        "P_TO": 3,
        "Q_TO": 4,
        "V_FROM": 5,
        "I_FROM": 6,
        "V_TO": 7,
        "I_TO": 8,
    }
    _SUB_MEASUREMENT_SUMMARY_ATTRS = (
        "_active_device_key_cache",
        "_active_measurement_key_cache",
        "_max_measurement_idx",
        "_node_voltage_measurement_cache",
        "_real_voltage_observation_node_cache",
        "_real_power_measurement_seed_cache",
        "_has_valid_angle_measurements",
    )

    @dataclass(frozen=True)
    class _MeasurementSideBlock:
        rows: np.ndarray
        measurements: Sequence[Measurement]

    @dataclass(frozen=True)
    class _SubEstimatorMeasurementContext:
        measurements: Sequence[Measurement]
        summary_attrs: Optional[Dict[str, object]]

    @dataclass(frozen=True)
    class _HybridCodeBucket:
        """Pre-sliced state/jacobian info for a single (device-type, measurement-code) pair.

        Built once at plan construction; consumed every iteration by
        `_evaluate_hybrid_measurements` and `_append_hybrid_jacobian_plan` so they
        avoid per-iteration `codes == k` masks, column-validity filtering and
        default-template copies.
        """
        rows: np.ndarray         # int32, global measurement rows
        hx_p: np.ndarray         # int32, primary hybrid_x index (e.g. 3*pos for DCAC P_DC)
        hx_q: np.ndarray         # int32, secondary hybrid_x index (used by hypot codes)
        jcol_p: np.ndarray       # int32, jacobian col for the primary power state
        jcol_q: np.ndarray       # int32, jacobian col for the secondary power state
        v_default: np.ndarray    # float64, len == rows.size, default voltage per row
        v_use_x: np.ndarray      # bool,    len == rows.size, True where voltage comes from x
        v_x_index: np.ndarray    # int32,   filtered (len == v_use_x.sum()), index into x
        v_jcol: np.ndarray       # int32,   len == rows.size, voltage jacobian col (only valid where v_use_x)
        v_any_x: bool            # True iff v_use_x has any True entries

    @dataclass(frozen=True)
    class _HybridMeasurementPlan:
        dcac_rows: np.ndarray
        dcac_codes: np.ndarray
        dcac_pos: np.ndarray
        dcac_dc_v_col: np.ndarray
        dcac_ac_v_col: np.ndarray
        dcac_dc_v_default: np.ndarray
        dcac_ac_v_default: np.ndarray
        acac_rows: np.ndarray
        acac_codes: np.ndarray
        acac_pos: np.ndarray
        acac_from_v_col: np.ndarray
        acac_to_v_col: np.ndarray
        acac_from_v_default: np.ndarray
        acac_to_v_default: np.ndarray
        # Per-code buckets (indexed by 1-based measurement code).
        # DCAC: 1=P_DC 2=P_AC 3=Q_AC 4=V_DC 5=I_DC 6=V_AC 7=I_AC
        dcac_p_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_p_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_q_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_ac: "HybridStateEstimator._HybridCodeBucket"
        # ACAC: 1=P_FROM 2=Q_FROM 3=P_TO 4=Q_TO 5=V_FROM 6=I_FROM 7=V_TO 8=I_TO
        acac_p_from: "HybridStateEstimator._HybridCodeBucket"
        acac_q_from: "HybridStateEstimator._HybridCodeBucket"
        acac_p_to: "HybridStateEstimator._HybridCodeBucket"
        acac_q_to: "HybridStateEstimator._HybridCodeBucket"
        acac_v_from: "HybridStateEstimator._HybridCodeBucket"
        acac_i_from: "HybridStateEstimator._HybridCodeBucket"
        acac_v_to: "HybridStateEstimator._HybridCodeBucket"
        acac_i_to: "HybridStateEstimator._HybridCodeBucket"

    @dataclass(frozen=True)
    class _HybridSeedMeasurementPlan:
        measurement_row: np.ndarray
        state_col: np.ndarray

    @dataclass(frozen=True)
    class _HybridConverterMeasurementSpec:
        device_type: str
        device_name: str
        meas_type: str
        value: float
        device_type_code: int = 0
        device_pos: int = -1
        meas_type_code: int = 0

    @dataclass(frozen=True)
    class _MeasurementActivitySummary:
        max_idx: int
        measured_devices: set
        active_keys: set

    _ACTIVE_KEY_MEAS_BITS = 16
    _ACTIVE_KEY_POS_BITS = 32

    @staticmethod
    def _packed_device_key(device_type_code: int, device_pos: int) -> int:
        if int(device_type_code) <= 0 or int(device_pos) < 0:
            return -1
        return (int(device_type_code) << HybridStateEstimator._ACTIVE_KEY_POS_BITS) | int(device_pos)

    @staticmethod
    def _packed_measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
        if int(device_type_code) <= 0 or int(device_pos) < 0 or int(meas_type_code) <= 0:
            return -1
        return (
            (int(device_type_code) << (HybridStateEstimator._ACTIVE_KEY_POS_BITS + HybridStateEstimator._ACTIVE_KEY_MEAS_BITS))
            | (int(device_pos) << HybridStateEstimator._ACTIVE_KEY_MEAS_BITS)
            | int(meas_type_code)
        )

    @staticmethod
    def _packed_device_key_array(device_type_code: np.ndarray, device_pos: np.ndarray) -> np.ndarray:
        type_values = np.asarray(device_type_code, dtype=np.int64)
        pos_values = np.asarray(device_pos, dtype=np.int64)
        return (type_values << HybridStateEstimator._ACTIVE_KEY_POS_BITS) | pos_values

    @staticmethod
    def _packed_measurement_key_array(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> np.ndarray:
        type_values = np.asarray(device_type_code, dtype=np.int64)
        pos_values = np.asarray(device_pos, dtype=np.int64)
        meas_values = np.asarray(meas_type_code, dtype=np.int64)
        return (
            (type_values << (HybridStateEstimator._ACTIVE_KEY_POS_BITS + HybridStateEstimator._ACTIVE_KEY_MEAS_BITS))
            | (pos_values << HybridStateEstimator._ACTIVE_KEY_MEAS_BITS)
            | meas_values
        )

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
        self.observability_result = None
        self.estimate_result = None
        self.removed_bad_data: List[BadDataItem] = []
        self.bad_items: List[BadDataItem] = []
        self.normalized_residual = np.array([], dtype=np.float64)
        self.se_result = None
        self._real_voltage_seed_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._hybrid_seed_measurement_plan_cache: Dict[
            Tuple[int, int], HybridStateEstimator._HybridSeedMeasurementPlan
        ] = {}
        self.power_flow_state = np.array([], dtype=np.float64)
        self.flat_state = np.array([], dtype=np.float64)
        self._initial_state_cache_ready = False
        if auto_prepare:
            self.prepare()

    def prepare(self) -> "HybridStateEstimator":
        if self._prepared:
            return self
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        self.network = self._load_network(self.e_file)
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        self.p_base = float(getattr(self.network, "p_base", 1.0))
        self.p_base_kW = float(getattr(self.network, "p_base_kW", self.p_base))
        self.u_scale = float(getattr(self.network, "u_scale", 1.0))
        self.p_scale = float(getattr(self.network, "p_scale", 1.0))
        self.i_scale = float(getattr(self.network, "i_scale", 1.0))
        stage_start = time.perf_counter()
        self.meas_ppc = copy_meas_ppc(build_meas_ppc_from_e_file(self.meas_file))
        self.measurements = measurement_list_from_meas_ppc(self.meas_ppc)
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self._sub_measurement_sources_by_side = None
        self._sub_measurement_source_rows_by_side = None
        self._sub_measurement_summary_attrs_by_side = {}
        self._sub_measurement_global_summary_attrs = None
        self._sub_measurements_converted_by_side = {"ac": False, "dc": False}
        self._measurements_normalized = False

        self._sub_estimators_enabled = False
        self._delegate_estimator = None
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        stage_start = time.perf_counter()
        self._ac_sub_estimator = self._build_ac_sub_estimator()
        self._dc_sub_estimator = self._build_dc_sub_estimator()
        self._record_profile_time("init.prepare_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._build_device_maps()
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)

        if self._try_delegate_uncoupled_single_side():
            self._prepared = True
            self._record_profile_time("init.total", time.perf_counter() - profile_start)
            return self

        stage_start = time.perf_counter()
        self._disable_angle_measurements()
        self._disable_unavailable_measurements()
        self._convert_measurements_to_pu()
        self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._finalize_sub_estimators_after_measurement_prepare()
        self._build_device_maps()
        self._populate_measurement_device_pos_from_sub_estimators()
        self._record_profile_time("init.finalize_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._add_pseudo_power_measurements()
        self._populate_measurement_device_pos_from_sub_estimators()
        self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._build_state_layout()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        self.calc = self._calc_adapter()
        self.targeted_observability_pseudo_count = 0
        stage_start = time.perf_counter()
        self._refresh_active_measurement_state_layout()
        self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
        self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        self._prepared = True
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

    def _require_prepared(self, action: str) -> None:
        if not self._prepared:
            raise RuntimeError(f"Call prepare() before {action}.")

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    @staticmethod
    def _load_measurements(meas_file: Path) -> List[Measurement]:
        return _read_measurements_direct(meas_file)

    @staticmethod
    def _load_network(e_file: Path) -> HybridPowerNetwork:
        efile_rows = _read_efile_rows(e_file)
        ppc = build_hybrid_ppc_with_topology_from_efile_rows(e_file, efile_rows)
        return _build_hybrid_se_network_from_ppc(ppc)

    def _initial_measurement_sources_by_side(self) -> Dict[str, List[Measurement]]:
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        if sources_by_side is None or rows_by_side is None:
            partitions = partition_measurements_by_code(
                self.measurements,
                self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
                side_by_device_type=self._MEASUREMENT_SIDE_BY_DEVICE_TYPE,
                table_builder=_measurement_table_from_measurements,
                as_view=True,
                sides=("ac", "dc", "hybrid"),
            )
            sources_by_side = partitions.measurements
            self._sub_measurement_sources_by_side = sources_by_side
            self._sub_measurement_source_rows_by_side = partitions.rows
            self._sub_measurement_source_count = len(self.measurements)
        return sources_by_side

    def _measurements_for_sub_estimator(self, side: str, share_measurements: bool) -> List[Measurement]:
        if share_measurements and self._is_uncoupled_single_side(side):
            return self.measurements
        sources = self._initial_measurement_sources_by_side().get(side, ())
        defer_finalize = getattr(self, "network", None) is not None and self._defer_sub_prepare_finalize()
        if share_measurements or defer_finalize:
            return sources
        return copy_measurement_view(sources)

    def _defer_sub_prepare_finalize(self) -> bool:
        if self._is_uncoupled_single_side("ac") or self._is_uncoupled_single_side("dc"):
            return False
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        return bool(has_ac and has_dc)

    def _is_uncoupled_single_side(self, side: str) -> bool:
        if getattr(self.network, "dcac_converters", None) or getattr(self.network, "acac_converters", None):
            return False
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        if side == "ac":
            return has_ac and not has_dc
        if side == "dc":
            return has_dc and not has_ac
        return False

    def _defer_sub_active_measurement_preparation(self) -> bool:
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        has_coupling = bool(
            getattr(self.network, "dcac_converters", None)
            or getattr(self.network, "acac_converters", None)
        )
        return bool(self.flat_start and (has_coupling or (has_ac and has_dc)))

    def _build_ac_sub_estimator(self) -> Optional[ACStateEstimator]:
        if not _side_has_alive_nodes(getattr(self.network, "ac", None)):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("ac"))
        try:
            estimator = ACStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.ac if reuse_loaded else None,
                measurements=self._measurements_for_sub_estimator("ac", share_measurements) if reuse_loaded else None,
                prepare_active_measurements=not defer_active,
                defer_prepare_finalize=defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
            )
            estimator.prepare()
            return estimator
        except RuntimeError as exc:
            if "No alive AC nodes" in str(exc):
                return None
            raise

    def _build_dc_sub_estimator(self) -> Optional[DCStateEstimator]:
        if not _side_has_alive_nodes(getattr(self.network, "dc", None)):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("dc"))
        try:
            estimator = DCStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.dc if reuse_loaded else None,
                measurements=self._measurements_for_sub_estimator("dc", share_measurements) if reuse_loaded else None,
                prepare_active_measurements=not defer_active,
                defer_prepare_finalize=defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
            )
            estimator.prepare()
            return estimator
        except RuntimeError as exc:
            if "No alive DC nodes" in str(exc):
                return None
            raise

    def _build_device_maps(self) -> None:
        if hasattr(self, "_measurement_plan_device_lookup_by_master_ids_cache"):
            delattr(self, "_measurement_plan_device_lookup_by_master_ids_cache")
        ac = self._ac_sub_estimator
        dc = self._dc_sub_estimator
        self.ac_nodes = list(getattr(ac, "nodes", [])) if ac is not None else []
        self.dc_nodes = list(getattr(dc, "nodes", [])) if dc is not None else []
        self.ac_node_by_name = dict(getattr(ac, "node_by_name", {})) if ac is not None else {}
        self.dc_node_by_name = dict(getattr(dc, "node_by_name", {})) if dc is not None else {}
        self.ac_node_by_idx = dict(getattr(ac, "node_by_idx", {})) if ac is not None else {}
        self.dc_node_by_idx = dict(getattr(dc, "node_by_idx", {})) if dc is not None else {}
        self.ac_branch_by_name = dict(getattr(ac, "branch_by_name", {})) if ac is not None else {}
        self.ac_transformer_by_name = dict(getattr(ac, "transformer_by_name", {})) if ac is not None else {}
        self.ac_switch_by_name = dict(getattr(ac, "switch_by_name", {})) if ac is not None else {}
        self.ac_break_by_name = dict(getattr(ac, "break_by_name", {})) if ac is not None else {}
        self.ac_zero_branch_by_name = dict(getattr(ac, "zero_branch_by_name", {})) if ac is not None else {}
        self.ac_generator_by_name = dict(getattr(ac, "generator_by_name", {})) if ac is not None else {}
        self.ac_load_by_name = dict(getattr(ac, "load_by_name", {})) if ac is not None else {}
        self.dc_branch_by_name = dict(getattr(dc, "branch_by_name", {})) if dc is not None else {}
        self.dc_switch_by_name = dict(getattr(dc, "switch_by_name", {})) if dc is not None else {}
        self.dc_break_by_name = dict(getattr(dc, "break_by_name", {})) if dc is not None else {}
        self.dc_zero_branch_by_name = dict(getattr(dc, "zero_branch_by_name", {})) if dc is not None else {}
        self.dc_generator_by_name = dict(getattr(dc, "generator_by_name", {})) if dc is not None else {}
        self.dc_load_by_name = dict(getattr(dc, "load_by_name", {})) if dc is not None else {}
        self.dcdc_by_name = dict(getattr(dc, "dcdc_by_name", {})) if dc is not None else {}
        self.ac_branch_stamp_by_name = dict(getattr(ac, "branch_stamp_by_name", {})) if ac is not None else {}
        self.ac_transformer_stamp_by_name = dict(getattr(ac, "transformer_stamp_by_name", {})) if ac is not None else {}
        self.dcac_by_name = {
            conv.name: conv
            for conv in getattr(self.network, "dcac_converters", [])
            if getattr(conv, "is_alive", False)
        }
        self.acac_by_name = {
            conv.name: conv
            for conv in getattr(self.network, "acac_converters", [])
            if getattr(conv, "is_alive", False)
        }
        self.dcac_converters = sorted(self.dcac_by_name.values(), key=lambda item: item.idx)
        self.acac_converters = sorted(self.acac_by_name.values(), key=lambda item: item.idx)
        self.dcac_pos_by_name = {conv.name: pos for pos, conv in enumerate(self.dcac_converters)}
        self.acac_pos_by_name = {conv.name: pos for pos, conv in enumerate(self.acac_converters)}

    def _try_delegate_uncoupled_single_side(self) -> bool:
        if self.dcac_by_name or self.acac_by_name:
            return False
        if self._ac_sub_estimator is not None and self._dc_sub_estimator is None:
            self._adopt_delegate(self._ac_sub_estimator, "ac")
            return True
        if self._dc_sub_estimator is not None and self._ac_sub_estimator is None:
            self._adopt_delegate(self._dc_sub_estimator, "dc")
            return True
        return False

    def _adopt_delegate(self, delegate, side: str) -> None:
        self._delegate_estimator = delegate
        self._sub_estimators_enabled = True
        self.calc = self._calc_adapter()
        self.measurements = delegate.measurements
        self.active_measurements = delegate.active_measurements
        self.active_z = delegate.active_z
        self.active_weight = delegate.active_weight
        active_table = getattr(self.active_measurements, "table", None)
        if active_table is not None and len(active_table.idx) == len(self.active_measurements):
            self.active_angle_residual_mask = np.asarray(active_table.angle_mask, dtype=bool)
        else:
            self.active_angle_residual_mask = angle_residual_mask(self.active_measurements)
        self.state_meta = list(getattr(delegate, "state_meta", []))
        self.state_labels = list(delegate.state_labels)
        self.ac_state_labels = list(delegate.state_labels) if side == "ac" else []
        self.dc_state_labels = list(delegate.state_labels) if side == "dc" else []
        self.ac_state_layout = delegate.state_layout() if side == "ac" else {"state_labels": [], "n_state": 0}
        self.dc_state_layout = (
            {
                "state_labels": delegate.state_labels,
                "voltage_col": getattr(delegate, "voltage_col", np.array([], dtype=np.int32)),
                "n_state": delegate.n_state,
                "references": getattr(delegate, "references", []),
            }
            if side == "dc"
            else {"state_labels": [], "n_state": 0}
        )
        self.state_sides = [side] * len(self.state_labels)
        self.n_state = int(delegate.n_state)
        self.ac_n_state = int(delegate.n_state) if side == "ac" else 0
        self.dc_n_state = int(delegate.n_state) if side == "dc" else 0
        self.hybrid_n_state = 0
        self.dc_state_start = self.ac_n_state
        self.hybrid_state_start = self.ac_n_state + self.dc_n_state
        self.voltage_cols = np.asarray(getattr(delegate, "voltage_cols", []), dtype=np.int32)
        self.power_flow_state = delegate.initial_state()
        self.flat_state = delegate.initial_state()
        self.dc_reference_nodes = getattr(delegate, "references", []) if side == "dc" else []
        self.ac_reference_nodes = getattr(delegate, "references", []) if side == "ac" else []
        self.dc_node_voltage_measurements = getattr(delegate, "node_voltage_measurements", {}) if side == "dc" else {}
        self.ac_node_voltage_measurements = getattr(delegate, "_node_voltage_measurement_cache", {}) if side == "ac" else {}
        self.ac_theta_state_col = getattr(delegate, "angle_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.ac_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.dc_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "dc" else np.array([], dtype=np.int32)
        self.targeted_observability_pseudo_count = getattr(delegate, "targeted_observability_pseudo_count", 0)
        self._adopt_delegate_active_partition(side)
        self._partition_state_variables()

    def _adopt_delegate_active_partition(self, side: str) -> None:
        count = len(self.active_measurements)
        rows = np.arange(count, dtype=np.int32)
        empty_rows = np.array([], dtype=np.int32)
        if side == "ac":
            self.ac_meas_rows = rows
            self.dc_meas_rows = empty_rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = self.active_measurements
            self.dc_meas = []
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = rows.copy()
            self._active_dc_hybrid_rows = empty_rows
        elif side == "dc":
            self.ac_meas_rows = empty_rows
            self.dc_meas_rows = rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = []
            self.dc_meas = self.active_measurements
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = empty_rows
            self._active_dc_hybrid_rows = rows.copy()
        else:
            self._partition_active_measurements()
            return
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(count, dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(count, dtype=bool)
        if side == "ac" and count:
            self._active_ac_delegated_row_mask[:] = True
        elif side == "dc" and count:
            self._active_dc_delegated_row_mask[:] = True
        self._jacobian_static_skip = np.zeros(count, dtype=bool)

    def _delegate(self):
        return getattr(self, "_delegate_estimator", None) if getattr(self, "_sub_estimators_enabled", False) else None

    class _ACCalcAdapter:
        def __init__(self, estimator: ACStateEstimator):
            self.estimator = estimator
            ids = np.asarray(getattr(estimator, "_ac_node_id_lookup_ids", ()), dtype=np.int64)
            pos = np.asarray(getattr(estimator, "_ac_node_id_lookup_pos", ()), dtype=np.int64)
            self.node_pos = {int(idx): int(value) for idx, value in zip(ids, pos) if int(value) >= 0}
            self.N = int(getattr(estimator, "n_nodes", len(self.node_pos)))

        def _extract_state_vars(self, x, update_cache=False):
            theta, voltage = self.estimator._unpack_state(np.asarray(x, dtype=np.float64))
            return theta, voltage, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    class _DCCalcAdapter:
        def __init__(self, estimator: DCStateEstimator):
            self.estimator = estimator
            ids = np.asarray(getattr(estimator, "_node_idx_lookup_ids", ()), dtype=np.int64)
            pos = np.asarray(getattr(estimator, "_node_idx_lookup_pos", ()), dtype=np.int64)
            self.node_pos = {int(idx): int(value) for idx, value in zip(ids, pos) if int(value) >= 0}
            self.alive_node_dict = self.node_pos
            self.N = int(getattr(estimator, "n_nodes", len(self.node_pos)))

    def _calc_adapter(self):
        ac_calc = self._ACCalcAdapter(self._ac_sub_estimator) if self._ac_sub_estimator is not None else None
        dc_calc = self._DCCalcAdapter(self._dc_sub_estimator) if self._dc_sub_estimator is not None else None
        return SimpleNamespace(
            ac_calc=ac_calc,
            dc_calc=dc_calc,
            ac_size=int(getattr(self._ac_sub_estimator, "n_state", 0) or 0),
            dc_size=int(getattr(self._dc_sub_estimator, "n_state", 0) or 0),
            total_vars=int(getattr(self._ac_sub_estimator, "n_state", 0) or 0)
            + int(getattr(self._dc_sub_estimator, "n_state", 0) or 0)
            + 3 * len(getattr(self, "dcac_converters", []))
            + 4 * len(getattr(self, "acac_converters", [])),
        )

    @classmethod
    def _measurement_side(cls, meas: Measurement) -> Optional[str]:
        """Classify by device ownership/location, never by row label or device-name text."""
        if meas.device_type in cls._HYBRID_MEASUREMENT_DEVICE_TYPES:
            return "hybrid"
        if meas.device_type in cls._AC_MEASUREMENT_DEVICE_TYPES:
            return "ac"
        if meas.device_type in cls._DC_MEASUREMENT_DEVICE_TYPES:
            return "dc"
        return None

    @staticmethod
    def _measurement_key(meas: Measurement) -> Tuple[str, str, str]:
        return meas.device_type, meas.device_name, meas.meas_type

    def _partition_measurement_list(
        self,
        measurements: Sequence[Measurement],
    ) -> Tuple[List[Tuple[int, Measurement]], List[Tuple[int, Measurement]], List[Tuple[int, Measurement]]]:
        ac_rows: List[Tuple[int, Measurement]] = []
        dc_rows: List[Tuple[int, Measurement]] = []
        hybrid_rows: List[Tuple[int, Measurement]] = []
        for row, meas in enumerate(measurements):
            side = self._measurement_side(meas)
            if side == "ac":
                ac_rows.append((row, meas))
            elif side == "dc":
                dc_rows.append((row, meas))
            elif side == "hybrid":
                hybrid_rows.append((row, meas))
        return ac_rows, dc_rows, hybrid_rows

    def _partition_active_measurements(self) -> None:
        ac, dc, hybrid = self._partition_measurement_list(self.active_measurements)
        self.ac_meas_rows = np.asarray([row for row, _meas in ac], dtype=np.int32)
        self.dc_meas_rows = np.asarray([row for row, _meas in dc], dtype=np.int32)
        self.hybrid_meas_rows = np.asarray([row for row, _meas in hybrid], dtype=np.int32)
        self.ac_meas = [meas for _row, meas in ac]
        self.dc_meas = [meas for _row, meas in dc]
        self.hybrid_meas = [meas for _row, meas in hybrid]
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(self.ac_meas_rows, self.ac_meas),
            "dc": self._MeasurementSideBlock(self.dc_meas_rows, self.dc_meas),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = list(self.ac_meas)
        self._active_dc_sub_measurements = list(self.dc_meas)
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._ac_sub_to_hybrid_cols = np.arange(getattr(self, "ac_n_state", 0), dtype=np.int32)
        self._dc_sub_to_hybrid_cols = (
            np.arange(getattr(self, "dc_n_state", 0), dtype=np.int32) + int(getattr(self, "dc_state_start", 0))
        )
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)

    def _measurement_blocks_for(
        self,
        measurements: Sequence[Measurement],
    ) -> Dict[str, "HybridStateEstimator._MeasurementSideBlock"]:
        if measurements is self.active_measurements:
            return self._active_measurement_blocks
        ac, dc, hybrid = self._partition_measurement_list(measurements)
        return {
            "ac": self._MeasurementSideBlock(
                np.asarray([row for row, _meas in ac], dtype=np.int32),
                [meas for _row, meas in ac],
            ),
            "dc": self._MeasurementSideBlock(
                np.asarray([row for row, _meas in dc], dtype=np.int32),
                [meas for _row, meas in dc],
            ),
            "hybrid": self._MeasurementSideBlock(
                np.asarray([row for row, _meas in hybrid], dtype=np.int32),
                [meas for _row, meas in hybrid],
            ),
        }

    def _partition_state_variables(self) -> None:
        if (
            getattr(self, "ac_n_state", None) is not None
            and getattr(self, "dc_n_state", None) is not None
            and getattr(self, "hybrid_n_state", None) is not None
        ):
            ac_n = int(self.ac_n_state)
            dc_n = int(self.dc_n_state)
            hybrid_n = int(self.hybrid_n_state)
            dc_start = int(getattr(self, "dc_state_start", ac_n))
            hybrid_start = int(getattr(self, "hybrid_state_start", dc_start + dc_n))
            self.ac_state_cols = np.arange(ac_n, dtype=np.int32)
            self.dc_state_cols = dc_start + np.arange(dc_n, dtype=np.int32)
            self.hybrid_state_cols = hybrid_start + np.arange(hybrid_n, dtype=np.int32)
            self.ac_vars = list(self.state_labels[:ac_n])
            self.dc_vars = list(self.state_labels[dc_start : dc_start + dc_n])
            self.hybrid_vars = list(self.state_labels[hybrid_start : hybrid_start + hybrid_n])
            self.ac_state_slice = slice(0, ac_n)
            self.dc_state_slice = slice(dc_start, dc_start + dc_n)
            self.hybrid_state_slice = slice(hybrid_start, hybrid_start + hybrid_n)
            return
        self.ac_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "ac"],
            dtype=np.int32,
        )
        self.dc_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "dc"],
            dtype=np.int32,
        )
        self.hybrid_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "hybrid"],
            dtype=np.int32,
        )
        self.ac_vars = [self.state_labels[idx] for idx in self.ac_state_cols]
        self.dc_vars = [self.state_labels[idx] for idx in self.dc_state_cols]
        self.hybrid_vars = [self.state_labels[idx] for idx in self.hybrid_state_cols]
        self.ac_state_slice = self._cols_to_slice(self.ac_state_cols)
        self.dc_state_slice = self._cols_to_slice(self.dc_state_cols)
        self.hybrid_state_slice = self._cols_to_slice(self.hybrid_state_cols)

    @staticmethod
    def _cols_to_slice(cols: np.ndarray) -> slice:
        if cols.size == 0:
            return slice(0, 0)
        return slice(int(cols[0]), int(cols[-1]) + 1)

    def _disable_angle_measurements(self) -> None:
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            if table_size:
                angle_mask = np.asarray(getattr(table, "angle_mask", np.zeros(table_size, dtype=bool)), dtype=bool)
                if angle_mask.size != table_size:
                    angle_mask = np.isin(np.asarray(table.meas_type, dtype=object), tuple(ANGLE_MEASUREMENT_TYPES))
                elif not np.any(angle_mask):
                    angle_mask = np.isin(np.asarray(table.meas_type, dtype=object), tuple(ANGLE_MEASUREMENT_TYPES))
                if np.any(angle_mask):
                    table.valid[angle_mask] = False
                    measurement_table_status_code(table)[angle_mask] = MEAS_STATUS_INVALID
                    if self.flat_start:
                        table.value[angle_mask] = 0.0
                    if list.__len__(self.measurements) == table_size:
                        for pos in np.flatnonzero(angle_mask):
                            meas = list.__getitem__(self.measurements, int(pos))
                            mark_measurement_invalid(meas)
                            if self.flat_start:
                                meas.value = 0.0
            if table_size < len(self.measurements):
                for meas in self._iter_measurements_from(self.measurements, table_size):
                    if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                        mark_measurement_invalid(meas)
                        if self.flat_start:
                            meas.value = 0.0
            return
        for meas in self.measurements:
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                mark_measurement_invalid(meas)
                if self.flat_start:
                    meas.value = 0.0

    def _disable_unavailable_measurements(self) -> None:
        device_maps = {
            "ACNode": self.ac_node_by_name,
            "ACBranch": self.ac_branch_by_name,
            "ACTransformer": self.ac_transformer_by_name,
            "ACBreak": self.ac_break_by_name,
            "ACZeroBranch": self.ac_zero_branch_by_name,
            "ACGenerator": self.ac_generator_by_name,
            "ACLoad": self.ac_load_by_name,
            "ACPowerBalance": self.ac_node_by_name,
            "DCNode": self.dc_node_by_name,
            "DCBranch": self.dc_branch_by_name,
            "DCBreak": self.dc_break_by_name,
            "DCZeroBranch": self.dc_zero_branch_by_name,
            "DCZeroBranchConstraint": self.dc_zero_branch_by_name,
            "DCBreakConstraint": self.dc_break_by_name,
            "DCGenerator": self.dc_generator_by_name,
            "DCLoad": self.dc_load_by_name,
            "DCDCConverter": self.dcdc_by_name,
            "DCPowerBalance": self.dc_node_by_name,
            "DCACConverter": self.dcac_by_name,
            "ACACConverter": self.acac_by_name,
        }
        unsupported_switch_rows = {"ACSwitch", "ACSwitchConstraint", "DCSwitch", "DCSwitchConstraint"}
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            disabled_rows = []
            if table_size:
                valid = np.asarray(table.valid, dtype=bool)
                weight = np.asarray(table.weight, dtype=np.float64)
                active = valid & (weight > 0.0)
                known = np.zeros(table_size, dtype=bool)
                rows_by_code = getattr(table, "rows_by_device_type_code", None) or {}
                device_type = np.asarray(table.device_type, dtype=object)
                device_name = np.asarray(table.device_name, dtype=object)

                def rows_for_device_type(device_type_name: str) -> np.ndarray:
                    code = DEVICE_TYPE_CODES.get(device_type_name)
                    rows = rows_by_code.get(code) if code is not None else None
                    if rows is None:
                        rows = np.flatnonzero(device_type == device_type_name)
                    return np.asarray(rows, dtype=np.int64)

                for device_type_name in unsupported_switch_rows:
                    rows = rows_for_device_type(device_type_name)
                    if rows.size:
                        rows = rows[active[rows]]
                        if rows.size:
                            known[rows] = True
                            disabled_rows.append(rows)

                for device_type_name, devices in device_maps.items():
                    rows = rows_for_device_type(device_type_name)
                    if rows.size == 0:
                        continue
                    rows = rows[active[rows]]
                    if rows.size == 0:
                        continue
                    known[rows] = True
                    if not devices:
                        disabled_rows.append(rows)
                        continue
                    # `tolist()` + list comp avoids per-element numpy unboxing.
                    name_list = device_name[rows].tolist()
                    available = np.array(
                        [name in devices for name in name_list],
                        dtype=bool,
                    )
                    if not np.all(available):
                        disabled_rows.append(rows[~available])

                unknown_active = np.flatnonzero(active & ~known)
                if unknown_active.size:
                    disabled_rows.append(unknown_active)

                if disabled_rows:
                    rows = np.concatenate(disabled_rows).astype(np.int64, copy=False)
                    table.valid[rows] = False
                    measurement_table_status_code(table)[rows] = MEAS_STATUS_INVALID
                    if list.__len__(self.measurements) == table_size:
                        for row in rows:
                            mark_measurement_invalid(list.__getitem__(self.measurements, int(row)))
            if table_size < len(self.measurements):
                for meas in self._iter_measurements_from(self.measurements, table_size):
                    if not meas.valid or meas.weight <= 0.0:
                        continue
                    if meas.device_type in unsupported_switch_rows:
                        mark_measurement_invalid(meas)
                        continue
                    devices = device_maps.get(meas.device_type)
                    if devices is None or meas.device_name not in devices:
                        mark_measurement_invalid(meas)
            return
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            if meas.device_type in unsupported_switch_rows:
                mark_measurement_invalid(meas)
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                mark_measurement_invalid(meas)

    def _sub_attrs_snapshot(self, estimator) -> Dict[str, object]:
        names = (
            "measurements",
            "measurement_table",
            "_active_device_key_cache",
            "_active_measurement_key_cache",
            "_max_measurement_idx",
            "_node_voltage_measurement_cache",
            "_real_power_measurement_seed_cache",
            "_has_valid_angle_measurements",
        )
        return {name: getattr(estimator, name) for name in names if hasattr(estimator, name)}

    @staticmethod
    def _restore_sub_attrs(estimator, snapshot: Dict[str, object]) -> None:
        for name, value in snapshot.items():
            setattr(estimator, name, value)

    @staticmethod
    def _iter_measurements_from(measurements: Sequence[Measurement], start: int):
        table = getattr(measurements, "table", None)
        if table is not None and hasattr(measurements, "_incorporated_tail_size"):
            table_size = int(table.idx.size)
            if int(start) >= table_size:
                incorporated = measurements._incorporated_tail_size()
                raw_start = incorporated + int(start) - table_size
                raw_stop = list.__len__(measurements)
                for pos in range(raw_start, raw_stop):
                    yield list.__getitem__(measurements, pos)
                return
        for meas in measurements[int(start):]:
            yield meas

    @staticmethod
    def _max_measurement_idx_fast(measurements: Sequence[Measurement]) -> int:
        table = getattr(measurements, "table", None)
        count = len(measurements)
        if table is not None:
            table_size = int(table.idx.size)
            max_idx = int(table.idx.max()) if table_size else 0
            if table_size == count:
                return max_idx
            if table_size < count:
                for meas in HybridStateEstimator._iter_measurements_from(measurements, table_size):
                    if meas.idx > max_idx:
                        max_idx = int(meas.idx)
                return max_idx
        max_idx = 0
        for meas in measurements:
            if meas.idx > max_idx:
                max_idx = int(meas.idx)
        return max_idx

    @staticmethod
    def _measurement_name_set_fast(measurements: Sequence[Measurement]) -> set:
        table = getattr(measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            names = np.asarray(getattr(table, "name", np.asarray([], dtype=object)), dtype=object)
            if names.size == table_size:
                result = set(names.tolist())
            else:
                result = set()
            for meas in HybridStateEstimator._iter_measurements_from(measurements, table_size):
                result.add(meas.name)
            return result
        return {meas.name for meas in measurements}

    @staticmethod
    def _measurement_table_has_string_columns(table) -> bool:
        if table is None:
            return False
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        return (
            np.asarray(getattr(table, "device_type", np.asarray([], dtype=object)), dtype=object).size == row_count
            and np.asarray(getattr(table, "device_name", np.asarray([], dtype=object)), dtype=object).size == row_count
            and np.asarray(getattr(table, "meas_type", np.asarray([], dtype=object)), dtype=object).size == row_count
        )

    @staticmethod
    def _activity_key_sets_from_table(table, active_mask: np.ndarray) -> Tuple[set, set]:
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        device_pos = getattr(table, "device_pos", None)
        meas_type_code = getattr(table, "meas_type_code", None)
        if device_pos is not None and meas_type_code is not None:
            device_type_code = np.asarray(table.device_type_code, dtype=np.int64)
            device_pos = np.asarray(device_pos, dtype=np.int64)
            meas_type_code = np.asarray(meas_type_code, dtype=np.int64)
            if device_type_code.size == row_count and device_pos.size == row_count and meas_type_code.size == row_count:
                active = np.asarray(active_mask, dtype=bool)
                valid_device = active & (device_type_code > 0) & (device_pos >= 0)
                measured_devices = set(
                    HybridStateEstimator._packed_device_key_array(
                        device_type_code[valid_device],
                        device_pos[valid_device],
                    ).tolist()
                )
                valid_measurement = valid_device & (meas_type_code > 0)
                active_keys = set(
                    HybridStateEstimator._packed_measurement_key_array(
                        device_type_code[valid_measurement],
                        device_pos[valid_measurement],
                        meas_type_code[valid_measurement],
                    ).tolist()
                )
                return measured_devices, active_keys
        if HybridStateEstimator._measurement_table_has_string_columns(table):
            device_type = np.asarray(table.device_type, dtype=object)
            device_name = np.asarray(table.device_name, dtype=object)
            meas_type = np.asarray(table.meas_type, dtype=object)
            active = np.asarray(active_mask, dtype=bool)
            return (
                set(zip(device_type[active].tolist(), device_name[active].tolist())),
                set(zip(device_type[active].tolist(), device_name[active].tolist(), meas_type[active].tolist())),
            )
        return set(), set()

    def _sub_measurement_summary_attrs(
        self,
        measurements: Sequence[Measurement],
        max_idx: Optional[int] = None,
        voltage_node_mapper=None,
    ) -> Dict[str, object]:
        cached_attrs = self._cached_sub_measurement_summary_attrs(
            measurements,
            max_idx=max_idx,
            voltage_node_mapper=voltage_node_mapper,
        )
        if cached_attrs is not None:
            return cached_attrs
        active_device_keys = set()
        active_measurement_keys = set()
        voltage_best: Dict[int, Tuple[float, float]] = {}
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            active = table.valid & (table.weight > 0.0)
            active_device_keys, active_measurement_keys = self._activity_key_sets_from_table(table, active)
            has_strings = self._measurement_table_has_string_columns(table)
            if has_strings:
                device_type = np.asarray(table.device_type, dtype=object)
                device_name = np.asarray(table.device_name, dtype=object)
                meas_type = np.asarray(table.meas_type, dtype=object)
            if voltage_node_mapper is not None and np.any(active) and has_strings:
                status_code = measurement_table_status_code(table)
                voltage_mask = (
                    active
                    & (status_code != MEAS_STATUS_PSEUDO)
                    & np.isin(meas_type, self._VOLTAGE_MEASUREMENT_TYPE_TUPLE)
                )
                voltage_rows = np.flatnonzero(voltage_mask)
                for row, device_type, device_name, meas_type in zip(
                    voltage_rows,
                    device_type[voltage_mask],
                    device_name[voltage_mask],
                    meas_type[voltage_mask],
                ):
                    node_idx = voltage_node_mapper(str(device_type), str(device_name), str(meas_type))
                    if node_idx is not None:
                        weight = float(table.weight[row])
                        current = voltage_best.get(int(node_idx))
                        if current is None or weight > current[0]:
                            voltage_best[int(node_idx)] = (weight, float(table.value[row]))
        else:
            for meas in measurements:
                if meas.valid and meas.weight > 0.0:
                    active_device_keys.add((meas.device_type, meas.device_name))
                    active_measurement_keys.add((meas.device_type, meas.device_name, meas.meas_type))
                    if (
                        voltage_node_mapper is not None
                        and not is_pseudo_measurement(meas)
                        and str(meas.meas_type).upper() in self._VOLTAGE_MEASUREMENT_TYPES
                    ):
                        node_idx = voltage_node_mapper(meas.device_type, meas.device_name, meas.meas_type)
                        if node_idx is not None:
                            weight = float(meas.weight)
                            current = voltage_best.get(int(node_idx))
                            if current is None or weight > current[0]:
                                voltage_best[int(node_idx)] = (weight, float(meas.value))
        attrs = {
            "_active_device_key_cache": active_device_keys,
            "_active_measurement_key_cache": active_measurement_keys,
            "_max_measurement_idx": int(max_idx) if max_idx is not None else self._max_measurement_idx_fast(self.measurements),
        }
        if voltage_node_mapper is not None:
            attrs["_real_voltage_observation_node_cache"] = {
                node_idx: value for node_idx, (_weight, value) in voltage_best.items()
            }
        return attrs

    @staticmethod
    def _copy_summary_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
        copied = {}
        for name, value in attrs.items():
            if isinstance(value, set):
                copied[name] = set(value)
            elif isinstance(value, dict):
                copied[name] = dict(value)
            else:
                copied[name] = value
        return copied

    def _cache_sub_measurement_summary_attrs(self, side: str, attrs: Dict[str, object]) -> None:
        if not attrs:
            return
        cache = getattr(self, "_sub_measurement_summary_attrs_by_side", None)
        if cache is None:
            cache = {}
            self._sub_measurement_summary_attrs_by_side = cache
        cache[side] = self._copy_summary_attrs(attrs)

    def _cached_sub_measurement_summary_attrs(
        self,
        measurements: Sequence[Measurement],
        *,
        max_idx: Optional[int] = None,
        voltage_node_mapper=None,
    ) -> Optional[Dict[str, object]]:
        cache_by_side = getattr(self, "_sub_measurement_summary_attrs_by_side", None)
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        if not cache_by_side or not sources_by_side:
            return None
        side = None
        for candidate_side, source in sources_by_side.items():
            if measurements is source:
                side = candidate_side
                break
        if side is None:
            return None
        side_attrs = cache_by_side.get(side, {})
        if voltage_node_mapper is not None and "_real_voltage_observation_node_cache" not in side_attrs:
            return None
        summary_attrs = getattr(self, "_sub_measurement_global_summary_attrs", None)
        if summary_attrs is None:
            summary = self._measurement_activity_summary()
            summary_attrs = {
                "source_count": len(self.measurements),
                "active_device_keys": summary.measured_devices,
                "active_measurement_keys": summary.active_keys,
            }
            self._sub_measurement_global_summary_attrs = summary_attrs
        attrs = {
            "_active_device_key_cache": summary_attrs["active_device_keys"],
            "_active_measurement_key_cache": summary_attrs["active_measurement_keys"],
            "_max_measurement_idx": (
                int(max_idx)
                if max_idx is not None
                else self._max_measurement_idx_fast(self.measurements)
            ),
        }
        if "_real_voltage_observation_node_cache" in side_attrs:
            attrs["_real_voltage_observation_node_cache"] = dict(side_attrs["_real_voltage_observation_node_cache"])
        return attrs

    def _sub_measurement_context(
        self,
        measurements: Sequence[Measurement],
        *,
        refresh_summary: bool = True,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        summary_voltage_node_mapper=None,
    ) -> "HybridStateEstimator._SubEstimatorMeasurementContext":
        summary_attrs = None
        if summary_measurements is not None:
            summary_attrs = self._sub_measurement_summary_attrs(
                summary_measurements,
                summary_max_idx,
                voltage_node_mapper=summary_voltage_node_mapper,
            )
        elif not refresh_summary:
            summary_attrs = {}
        return self._SubEstimatorMeasurementContext(measurements=measurements, summary_attrs=summary_attrs)

    def _apply_sub_measurement_context(
        self,
        estimator,
        context: "HybridStateEstimator._SubEstimatorMeasurementContext",
    ) -> None:
        estimator.measurements = context.measurements
        if context.summary_attrs is None:
            refresh = getattr(estimator, "_refresh_measurement_summary_cache", None)
            if callable(refresh):
                refresh()
            return
        for name, value in context.summary_attrs.items():
            setattr(estimator, name, value)

    def _invoke_sub_estimator_methods(
        self,
        estimator,
        context: "HybridStateEstimator._SubEstimatorMeasurementContext",
        method_names: Sequence[str],
        *args,
        preserve_max_idx: bool = False,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ):
        if estimator is None:
            return None
        snapshot = self._sub_attrs_snapshot(estimator)
        max_idx = None
        result = None
        try:
            self._apply_sub_measurement_context(estimator, context)
            for index, method_name in enumerate(method_names):
                method = getattr(estimator, method_name, None)
                if not callable(method):
                    continue
                if index == 0:
                    result = method(*args)
                else:
                    method()
            if captured_attrs is not None:
                captured_attrs.clear()
                captured_attrs.update(
                    {
                        name: getattr(estimator, name)
                        for name in capture_attrs
                        if hasattr(estimator, name)
                    }
                )
            max_idx = getattr(estimator, "_max_measurement_idx", None)
            return result
        finally:
            self._restore_sub_attrs(estimator, snapshot)
            if preserve_max_idx and max_idx is not None:
                estimator._max_measurement_idx = max_idx

    def _call_sub_with_measurements(
        self,
        estimator,
        measurements: List[Measurement],
        method_name: str,
        *args,
        refresh_summary: bool = True,
        preserve_max_idx: bool = False,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ):
        context = self._sub_measurement_context(
            measurements,
            refresh_summary=refresh_summary,
            summary_measurements=summary_measurements,
            summary_max_idx=summary_max_idx,
        )
        return self._invoke_sub_estimator_methods(
            estimator,
            context,
            (method_name,),
            *args,
            preserve_max_idx=preserve_max_idx,
            capture_attrs=capture_attrs,
            captured_attrs=captured_attrs,
        )

    def _call_sub_sequence_with_measurements(
        self,
        estimator,
        measurements: List[Measurement],
        method_names: Sequence[str],
        *,
        refresh_summary: bool = True,
        preserve_max_idx: bool = False,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        summary_voltage_node_mapper=None,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        context = self._sub_measurement_context(
            measurements,
            refresh_summary=refresh_summary,
            summary_measurements=summary_measurements,
            summary_max_idx=summary_max_idx,
            summary_voltage_node_mapper=summary_voltage_node_mapper,
        )
        self._invoke_sub_estimator_methods(
            estimator,
            context,
            tuple(method_names),
            preserve_max_idx=preserve_max_idx,
            capture_attrs=capture_attrs,
            captured_attrs=captured_attrs,
        )

    def _adopt_sub_measurement_update(self, captured_attrs: Dict[str, object]) -> bool:
        """Adopt a sub-estimator MeasurementList when it replaced the shared view."""
        updated = captured_attrs.get("measurements")
        if updated is None or updated is self.measurements:
            return False
        table = captured_attrs.get("measurement_table")
        if table is None:
            table = getattr(updated, "table", None)
        self.measurements = updated
        if table is not None:
            self.measurement_table = table
            try:
                self.measurements.table = table
            except AttributeError:
                pass
        self._invalidate_measurement_activity_summary()
        return True

    def _meas_ppc_for_source_rows(self, source_rows: np.ndarray) -> Dict:
        meas_ppc = getattr(self, "meas_ppc", None)
        if not isinstance(meas_ppc, dict):
            return {}
        rows = np.asarray(source_rows, dtype=np.int64)
        row_count = int(rows.size)
        out = dict(meas_ppc)
        full_count = int(np.asarray(meas_ppc.get("idx_array", np.asarray([]))).size)
        for key in (
            "idx_array",
            "weight_array",
            "valid_array",
            "value_array",
            "status_array",
            "device_type_code_array",
            "device_name_id_array",
            "meas_type_code_array",
            "angle_mask_array",
            "device_pos",
            "scale",
            "from_pos",
            "to_pos",
            "available",
            "name",
            "device_type",
            "device_name",
            "meas_type",
        ):
            value = meas_ppc.get(key)
            if isinstance(value, np.ndarray) and int(value.shape[0]) == full_count:
                out[key] = value[rows.astype(np.intp, copy=False)].copy()
        meas = meas_ppc.get("meas")
        if isinstance(meas, np.ndarray) and meas.ndim == 2 and int(meas.shape[0]) == full_count:
            out["meas"] = meas[rows.astype(np.intp, copy=False), :].copy()
        device_type_code = out.get("device_type_code_array")
        if isinstance(device_type_code, np.ndarray) and int(device_type_code.size) == row_count:
            out["rows_by_device_type_code"] = {
                int(code): np.flatnonzero(device_type_code == code).astype(np.int64, copy=False)
                for code in np.unique(device_type_code)
            }
        else:
            out["rows_by_device_type_code"] = {}
        out["_mutable_runtime_arrays"] = True
        out["normalized"] = bool(meas_ppc.get("normalized", False))
        return out

    @staticmethod
    def _sub_voltage_node_mapper(estimator):
        mapper = getattr(estimator, "_voltage_measurement_node_idx", None)
        node_pos = getattr(estimator, "node_pos", {})
        if not callable(mapper):
            return None

        def voltage_node_mapper(device_type: str, device_name: str, meas_type: str):
            node_idx = mapper(device_type, device_name, meas_type)
            if node_idx is None or int(node_idx) not in node_pos:
                return None
            return int(node_idx)

        return voltage_node_mapper

    def _convert_measurements_to_pu(self) -> None:
        """Delegate AC/DC normalization and normalize only converter rows locally."""
        self._invalidate_measurement_activity_summary()
        if getattr(self, "_measurements_normalized", False):
            return
        sources = self._initial_measurement_sources_by_side()
        converted_by_sub = getattr(self, "_sub_measurements_converted_by_side", {})
        if not converted_by_sub.get("ac", False):
            captured_summary: Dict[str, object] = {}
            self._call_sub_with_measurements(
                self._ac_sub_estimator,
                sources["ac"],
                "_convert_measurements_to_pu",
                refresh_summary=False,
                capture_attrs=self._SUB_MEASUREMENT_SUMMARY_ATTRS,
                captured_attrs=captured_summary,
            )
            self._cache_sub_measurement_summary_attrs("ac", captured_summary)
            if not self._sub_estimator_updates_measurement_table(self._ac_sub_estimator):
                self._sync_partition_table_from_objects(sources["ac"])
            converted_by_sub["ac"] = True
            if hasattr(sources["ac"], "normalized"):
                sources["ac"].normalized = True
            if self._ac_sub_estimator is not None and hasattr(self._ac_sub_estimator.measurements, "normalized"):
                self._ac_sub_estimator.measurements.normalized = True
        if not converted_by_sub.get("dc", False):
            captured_summary = {}
            dc_rows = (
                np.asarray(self._sub_measurement_source_rows_by_side.get("dc", np.asarray([], dtype=np.int64)), dtype=np.int64)
                if isinstance(getattr(self, "_sub_measurement_source_rows_by_side", None), dict)
                else np.asarray([], dtype=np.int64)
            )
            dc_table = getattr(sources["dc"], "table", None)
            normalized_by_ppc = False
            if self._dc_sub_estimator is not None and dc_table is not None and dc_rows.size == int(dc_table.idx.size):
                dc_meas_ppc = self._meas_ppc_for_source_rows(dc_rows)
                normalize_from_ppc = getattr(self._dc_sub_estimator, "_normalize_measurements_to_pu_from_meas_ppc", None)
                if callable(normalize_from_ppc) and dc_meas_ppc:
                    self._dc_sub_estimator.meas_ppc = dc_meas_ppc
                    self._dc_sub_estimator.measurements = sources["dc"]
                    normalized_by_ppc = bool(normalize_from_ppc(dc_table, dc_meas_ppc))
                    captured_summary = {
                        name: getattr(self._dc_sub_estimator, name)
                        for name in self._SUB_MEASUREMENT_SUMMARY_ATTRS
                        if hasattr(self._dc_sub_estimator, name)
                    }
            if not normalized_by_ppc:
                self._call_sub_with_measurements(
                    self._dc_sub_estimator,
                    sources["dc"],
                    "_convert_measurements_to_pu",
                    refresh_summary=False,
                    capture_attrs=self._SUB_MEASUREMENT_SUMMARY_ATTRS,
                    captured_attrs=captured_summary,
                )
            self._cache_sub_measurement_summary_attrs("dc", captured_summary)
            if not self._sub_estimator_updates_measurement_table(self._dc_sub_estimator):
                self._sync_partition_table_from_objects(sources["dc"])
            converted_by_sub["dc"] = True
            if hasattr(sources["dc"], "normalized"):
                sources["dc"].normalized = True
            if self._dc_sub_estimator is not None and hasattr(self._dc_sub_estimator.measurements, "normalized"):
                self._dc_sub_estimator.measurements.normalized = True
        self._convert_hybrid_measurements_to_pu(sources["hybrid"])
        self._sync_measurement_table_from_partition_tables(sources)
        if hasattr(self.measurements, "normalized"):
            self.measurements.normalized = True
        self._measurements_normalized = True
        if getattr(self, "meas_ppc", None) is not None and getattr(self.measurements, "table", None) is not None:
            self.meas_ppc["normalized"] = True
            sync_meas_ppc_from_measurement_table(self.meas_ppc, self.measurements.table)

    def _device_id_lookup_from_names(self, names: Sequence[str]) -> np.ndarray:
        meas_ppc = getattr(self, "meas_ppc", {})
        name_to_id = meas_ppc.get("device_name_id_by_name", {}) if isinstance(meas_ppc, dict) else {}
        if not name_to_id:
            return np.asarray([], dtype=np.int64)
        ids = np.asarray([int(name_to_id.get(str(name), -1)) for name in names], dtype=np.int64)
        valid = ids >= 0
        if not np.any(valid):
            return np.asarray([], dtype=np.int64)
        lookup = np.empty(int(ids[valid].max()) + 1, dtype=np.int64)
        lookup.fill(-1)
        lookup[ids[valid].astype(np.intp, copy=False)] = np.flatnonzero(valid).astype(np.int64, copy=False)
        return lookup

    def _measurement_plan_device_lookup_by_master_ids(self) -> Dict[int, np.ndarray]:
        cached = getattr(self, "_measurement_plan_device_lookup_by_master_ids_cache", None)
        if cached is not None:
            return cached
        lookup_by_code: Dict[int, np.ndarray] = {}
        ac = getattr(self, "_ac_sub_estimator", None)
        if ac is not None:
            for code, name_rows in (getattr(ac, "_ac_measurement_plan_name_rows_by_type_code", {}) or {}).items():
                names, rows = name_rows
                rows = np.asarray(rows, dtype=np.int64)
                if rows.size == 0:
                    lookup_by_code[int(code)] = np.asarray([], dtype=np.int64)
                    continue
                row_names = np.asarray(names, dtype=object)[rows.astype(np.intp, copy=False)]
                lookup_by_code[int(code)] = self._device_id_lookup_from_names(row_names)
        dc = getattr(self, "_dc_sub_estimator", None)
        if dc is not None:
            constraint_names = np.concatenate(
                (
                    np.asarray(getattr(dc, "_zero_branch_names", ()), dtype=object),
                    np.asarray(getattr(dc, "_switch_names", ()), dtype=object),
                    np.asarray(getattr(dc, "_break_names", ()), dtype=object),
                )
            )
            dc_name_arrays = {
                "DCNode": getattr(dc, "_raw_node_names_alive", ()),
                "DCBranch": getattr(dc, "_branch_names", ()),
                "DCLoad": getattr(dc, "_load_names", ()),
                "DCGenerator": getattr(dc, "_generator_names", ()),
                "DCZeroBranch": getattr(dc, "_zero_branch_names", ()),
                "DCBreak": getattr(dc, "_break_names", ()),
                "DCZeroBranchConstraint": constraint_names,
                "DCBreakConstraint": constraint_names,
                "DCSwitchConstraint": constraint_names,
                "DCDCConverter": getattr(dc, "_dcdc_names", ()),
            }
            for name, values in dc_name_arrays.items():
                code = DEVICE_TYPE_CODES.get(name)
                if code is not None:
                    lookup_by_code[int(code)] = self._device_id_lookup_from_names(
                        np.asarray(values, dtype=object)
                    )
        lookup_by_code[int(DEVICE_TYPE_CODES["DCACConverter"])] = self._device_id_lookup_from_names(
            [conv.name for conv in self.dcac_converters]
        )
        lookup_by_code[int(DEVICE_TYPE_CODES["ACACConverter"])] = self._device_id_lookup_from_names(
            [conv.name for conv in self.acac_converters]
        )
        self._measurement_plan_device_lookup_by_master_ids_cache = lookup_by_code
        return lookup_by_code

    def _populate_measurement_device_pos_from_sub_estimators(self) -> None:
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            return
        n_rows = int(table.idx.size)
        if n_rows == 0:
            table.device_pos = np.asarray([], dtype=np.int64)
            return
        device_name_id = getattr(table, "device_name_id", None)
        if device_name_id is None or np.asarray(device_name_id).size != n_rows:
            return
        device_name_id = np.asarray(device_name_id, dtype=np.int64)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is not None and np.asarray(device_pos).size == n_rows:
            device_pos = np.asarray(device_pos, dtype=np.int64).copy()
        else:
            device_pos = np.empty(n_rows, dtype=np.int64)
            device_pos.fill(-1)

        lookup_by_code = self._measurement_plan_device_lookup_by_master_ids()

        rows_by_code = getattr(table, "rows_by_device_type_code", None) or {}
        for code, rows in rows_by_code.items():
            lookup = lookup_by_code.get(int(code))
            if lookup is None or lookup.size == 0:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            rows = rows[(rows >= 0) & (rows < n_rows)]
            if rows.size == 0:
                continue
            ids = device_name_id[rows.astype(np.intp, copy=False)]
            in_range = (ids >= 0) & (ids < lookup.size)
            if np.any(in_range):
                device_pos[rows[in_range].astype(np.intp, copy=False)] = lookup[ids[in_range].astype(np.intp, copy=False)]

        table.device_pos = device_pos
        cache = getattr(table, "_device_pos_plan_cache", None)
        if cache is not None:
            cache.clear()
        if isinstance(getattr(self, "meas_ppc", None), dict):
            self.meas_ppc["device_pos"] = device_pos

    def _finalize_sub_estimators_after_measurement_prepare(self) -> None:
        if self._ac_sub_estimator is not None and (
            getattr(self._ac_sub_estimator, "_defer_prepare_finalize_pending", False)
            or getattr(self._ac_sub_estimator, "_prepare_defer_finalize", False)
            or not hasattr(self._ac_sub_estimator, "voltage_col")
        ):
            self._ac_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )
        if self._dc_sub_estimator is not None and (
            getattr(self._dc_sub_estimator, "_defer_prepare_finalize_pending", False)
            or getattr(self._dc_sub_estimator, "_prepare_defer_finalize", False)
            or not hasattr(self._dc_sub_estimator, "voltage_col")
        ):
            self._dc_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )

    @staticmethod
    def _sub_estimator_updates_measurement_table(estimator) -> bool:
        return isinstance(estimator, (ACStateEstimator, DCStateEstimator))

    @staticmethod
    def _sync_partition_table_from_objects(measurements: Sequence[Measurement]) -> bool:
        table = getattr(measurements, "table", None)
        if table is None or len(table.idx) != len(measurements):
            return False
        for pos, meas in enumerate(measurements):
            table.valid[pos] = bool(meas.valid)
            table.weight[pos] = float(meas.weight)
            table.value[pos] = float(meas.value)
        return True

    def _hybrid_measurement_device_pos_array(self, table) -> np.ndarray:
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is not None and np.asarray(device_pos).size == row_count:
            return np.asarray(device_pos, dtype=np.int64)
        device_pos = np.empty(row_count, dtype=np.int64)
        device_pos.fill(-1)
        device_name_id = getattr(table, "device_name_id", None)
        if device_name_id is None or np.asarray(device_name_id).size != row_count:
            table.device_pos = device_pos
            return device_pos
        device_name_id = np.asarray(device_name_id, dtype=np.int64)
        code = np.asarray(table.device_type_code, dtype=np.int16)
        for device_type_code, converters in (
            (DEVICE_TYPE_DCACConverter, self.dcac_converters),
            (DEVICE_TYPE_ACACConverter, self.acac_converters),
        ):
            rows = np.flatnonzero(code == int(device_type_code))
            if rows.size == 0:
                continue
            lookup = self._device_id_lookup_from_names([conv.name for conv in converters])
            if lookup.size == 0:
                continue
            ids = device_name_id[rows.astype(np.intp, copy=False)]
            in_range = (ids >= 0) & (ids < lookup.size)
            if np.any(in_range):
                device_pos[rows[in_range].astype(np.intp, copy=False)] = lookup[ids[in_range].astype(np.intp, copy=False)]
        table.device_pos = device_pos
        return device_pos

    def _convert_hybrid_measurements_to_pu(self, measurements: Sequence[Measurement]) -> None:
        table = getattr(measurements, "table", None)
        has_table = table is not None and len(table.idx) == len(measurements)
        if not has_table:
            for meas in measurements:
                if not meas.valid or meas.weight <= 0.0:
                    continue
                scale = self._hybrid_measurement_scale(meas)
                meas.value = float(meas.value) / scale
            return

        active = np.asarray(table.valid, dtype=bool) & (np.asarray(table.weight, dtype=np.float64) > 0.0)
        if not np.any(active):
            return
        scale = np.ones(table.value.size, dtype=np.float64)
        code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
        if meas_code.size != table.value.size:
            meas_code = np.zeros(table.value.size, dtype=np.int16)
        device_pos = self._hybrid_measurement_device_pos_array(table)

        dcac_rows = np.flatnonzero(active & (code == DEVICE_TYPE_DCACConverter) & (device_pos >= 0))
        if dcac_rows.size:
            dcac_pos = device_pos[dcac_rows].astype(np.intp, copy=False)
            dc_nodes = np.asarray([int(conv.dc_node) for conv in self.dcac_converters], dtype=np.int64)
            ac_nodes = np.asarray([int(conv.ac_node) for conv in self.dcac_converters], dtype=np.int64)
            dcac_dc_v_scale = np.asarray([self._dc_voltage_file_base(int(node)) for node in dc_nodes], dtype=np.float64)
            dcac_dc_i_scale = np.asarray([self._dc_current_base(int(node)) for node in dc_nodes], dtype=np.float64)
            dcac_ac_v_scale = np.asarray([self._ac_voltage_file_base(int(node)) for node in ac_nodes], dtype=np.float64)
            dcac_ac_i_scale = np.asarray([self._ac_current_base(int(node)) for node in ac_nodes], dtype=np.float64)
            scale_vals = np.ones(dcac_rows.size, dtype=np.float64)
            dcac_meas = meas_code[dcac_rows]
            power_mask = (dcac_meas == MEAS_TYPE_P_DC) | (dcac_meas == MEAS_TYPE_P_AC) | (dcac_meas == MEAS_TYPE_Q_AC)
            scale_vals[power_mask] = self._power_file_base()
            mask = dcac_meas == MEAS_TYPE_V_DC
            scale_vals[mask] = dcac_dc_v_scale[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_I_DC
            scale_vals[mask] = dcac_dc_i_scale[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_V_AC
            scale_vals[mask] = dcac_ac_v_scale[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_I_AC
            scale_vals[mask] = dcac_ac_i_scale[dcac_pos[mask]]
            scale[dcac_rows] = scale_vals

        acac_rows = np.flatnonzero(active & (code == DEVICE_TYPE_ACACConverter) & (device_pos >= 0))
        if acac_rows.size:
            acac_pos = device_pos[acac_rows].astype(np.intp, copy=False)
            from_nodes = np.asarray([int(conv.i_node) for conv in self.acac_converters], dtype=np.int64)
            to_nodes = np.asarray([int(conv.j_node) for conv in self.acac_converters], dtype=np.int64)
            from_v_scale = np.asarray([self._ac_voltage_file_base(int(node)) for node in from_nodes], dtype=np.float64)
            from_i_scale = np.asarray([self._ac_current_base(int(node)) for node in from_nodes], dtype=np.float64)
            to_v_scale = np.asarray([self._ac_voltage_file_base(int(node)) for node in to_nodes], dtype=np.float64)
            to_i_scale = np.asarray([self._ac_current_base(int(node)) for node in to_nodes], dtype=np.float64)
            scale_vals = np.ones(acac_rows.size, dtype=np.float64)
            acac_meas = meas_code[acac_rows]
            power_mask = (
                (acac_meas == MEAS_TYPE_P_FROM)
                | (acac_meas == MEAS_TYPE_Q_FROM)
                | (acac_meas == MEAS_TYPE_P_TO)
                | (acac_meas == MEAS_TYPE_Q_TO)
            )
            scale_vals[power_mask] = self._power_file_base()
            mask = acac_meas == MEAS_TYPE_V_FROM
            scale_vals[mask] = from_v_scale[acac_pos[mask]]
            mask = acac_meas == MEAS_TYPE_I_FROM
            scale_vals[mask] = from_i_scale[acac_pos[mask]]
            mask = acac_meas == MEAS_TYPE_V_TO
            scale_vals[mask] = to_v_scale[acac_pos[mask]]
            mask = acac_meas == MEAS_TYPE_I_TO
            scale_vals[mask] = to_i_scale[acac_pos[mask]]
            scale[acac_rows] = scale_vals

        table.value[active] = np.divide(
            table.value[active],
            scale[active],
            out=table.value[active].copy(),
            where=np.abs(scale[active]) > 1e-12,
        )
        object_count = list.__len__(measurements) if isinstance(measurements, list) else 0
        if object_count == table.value.size and object_count > 0:
            for pos, meas in enumerate(list.__iter__(measurements)):
                meas.value = float(table.value[pos])
        else:
            source = getattr(measurements, "source", None)
            source_rows = getattr(measurements, "rows", None)
            source_count = list.__len__(source) if isinstance(source, list) else 0
            if source_count > 0 and source_rows is not None:
                for pos, source_row in enumerate(source_rows):
                    source[int(source_row)].value = float(table.value[pos])

    def _sync_measurement_table_from_partition_tables(
        self,
        sources: Dict[str, Sequence[Measurement]],
    ) -> None:
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or len(master_table.idx) != len(self.measurements):
            self._sync_measurement_table_from_objects()
            return
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        if rows_by_side is None:
            self._sync_measurement_table_from_objects()
            return
        for side, measurements in sources.items():
            rows = np.asarray(rows_by_side.get(side, ()), dtype=np.int64)
            table = getattr(measurements, "table", None)
            if table is None or len(table.idx) != len(measurements) or rows.size != len(measurements):
                self._sync_measurement_table_from_objects()
                return
            if rows.size == 0:
                continue
            master_table.valid[rows] = table.valid
            master_table.weight[rows] = table.weight
            master_table.value[rows] = table.value

    def _sync_measurement_table_from_objects(self) -> None:
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            return
        for pos, meas in enumerate(self.measurements):
            table.valid[pos] = bool(meas.valid)
            table.weight[pos] = float(meas.weight)
            table.value[pos] = float(meas.value)

    def _hybrid_measurement_scale(self, meas: Measurement) -> float:
        if meas.device_type == "DCACConverter":
            conv = self.dcac_by_name[meas.device_name]
            if meas.meas_type in ("P_DC", "P_AC", "Q_AC"):
                return self._power_file_base()
            if meas.meas_type == "V_DC":
                return self._dc_voltage_file_base(conv.dc_node)
            if meas.meas_type == "I_DC":
                return self._dc_current_base(conv.dc_node)
            if meas.meas_type == "V_AC":
                return self._ac_voltage_file_base(conv.ac_node)
            if meas.meas_type == "I_AC":
                return self._ac_current_base(conv.ac_node)
            return 1.0
        if meas.device_type == "ACACConverter":
            conv = self.acac_by_name[meas.device_name]
            return self._ac_terminal_scale(meas, conv)
        return 1.0

    def _ac_voltage_base(self, node_idx: int) -> float:
        estimator = getattr(self, "_ac_sub_estimator", None)
        if estimator is not None:
            value = self._lookup_node_value_by_idx(
                int(node_idx),
                getattr(estimator, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
            if value is not None:
                return value
        node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
        if int(node_idx) in node_dict:
            return float(node_dict[int(node_idx)].vbase)
        return 1.0

    def _dc_voltage_base(self, node_idx: int) -> float:
        estimator = getattr(self, "_dc_sub_estimator", None)
        if estimator is not None:
            value = self._lookup_node_value_by_idx(
                int(node_idx),
                getattr(estimator, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
            if value is not None:
                return value
        node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
        if int(node_idx) in node_dict:
            return float(node_dict[int(node_idx)].vbase)
        return 1.0

    def _ac_current_base(self, node_idx: int) -> float:
        return self.i_scale * ac_current_base_ka(self.p_base_kW, self._ac_voltage_base(node_idx))

    def _dc_current_base(self, node_idx: int) -> float:
        return self.i_scale * dc_current_base_ka(self.p_base_kW, self._dc_voltage_base(node_idx))

    def _ac_voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._ac_voltage_base(node_idx)

    def _dc_voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._dc_voltage_base(node_idx)

    def _power_file_base(self) -> float:
        return self.p_base

    def _ac_terminal_scale(self, meas: Measurement, device) -> float:
        if meas.meas_type.startswith(("P_", "Q_")):
            return self._power_file_base()
        if meas.meas_type.endswith("_FROM"):
            node_idx = device.i_node
        elif meas.meas_type.endswith("_TO"):
            node_idx = device.j_node
        else:
            return 1.0
        if meas.meas_type.startswith("V_"):
            return self._ac_voltage_file_base(node_idx)
        if meas.meas_type.startswith("I_"):
            return self._ac_current_base(node_idx)
        return 1.0

    @staticmethod
    def _lookup_node_value_by_idx(node_idx: int, ids: np.ndarray, pos: np.ndarray, values: np.ndarray) -> Optional[float]:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if ids.size == 0 or pos.size == 0 or values.size == 0:
            return None
        loc = int(np.searchsorted(ids, int(node_idx)))
        if loc < 0 or loc >= ids.size or int(ids[loc]) != int(node_idx):
            return None
        value_pos = int(pos[loc])
        if value_pos < 0 or value_pos >= values.size:
            return None
        return float(values[value_pos])

    def _append_pseudo_measurement(
        self,
        next_idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        value: float,
    ) -> int:
        measurement = Measurement.__new__(Measurement)
        measurement.idx = next_idx
        measurement.name = name
        measurement.device_type = device_type
        measurement.device_name = device_name
        measurement.meas_type = meas_type
        measurement.weight = self.pseudo_measurement_weight
        measurement.valid = True
        measurement.value = float(value)
        mark_measurement_pseudo(measurement)
        self.measurements.append(measurement)
        self._invalidate_measurement_activity_summary()
        return next_idx + 1

    def _hybrid_converter_spec_device_key(self, spec: "_HybridConverterMeasurementSpec") -> int:
        device_type_code = int(spec.device_type_code or DEVICE_TYPE_CODES.get(spec.device_type, 0))
        device_pos = int(spec.device_pos)
        if device_pos < 0:
            if device_type_code == DEVICE_TYPE_DCACConverter:
                device_pos = int(self.dcac_pos_by_name.get(spec.device_name, -1))
            elif device_type_code == DEVICE_TYPE_ACACConverter:
                device_pos = int(self.acac_pos_by_name.get(spec.device_name, -1))
        return self._packed_device_key(device_type_code, device_pos)

    def _add_pseudo_power_measurements(self) -> None:
        """Delegate AC/DC pseudo rows, then add missing converter-device pseudo rows."""
        if self._ac_sub_estimator is not None:
            add_balance = getattr(self._ac_sub_estimator, "_add_power_balance_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_balance):
                methods.append("_add_power_balance_constraint_measurements")
            max_idx = self._max_measurement_idx_fast(self.measurements)
            self._call_sub_sequence_with_measurements(
                self._ac_sub_estimator,
                self.measurements,
                methods,
                preserve_max_idx=True,
                summary_measurements=self._initial_measurement_sources_by_side()["ac"],
                summary_max_idx=max_idx,
                summary_voltage_node_mapper=self._sub_voltage_node_mapper(self._ac_sub_estimator),
                capture_attrs=("measurements", "measurement_table", "_max_measurement_idx"),
                captured_attrs=(captured_attrs := {}),
            )
            self._adopt_sub_measurement_update(captured_attrs)
            if callable(add_balance):
                self._disable_coupled_ac_power_balance_rows()
        if self._dc_sub_estimator is not None:
            add_constraints = getattr(self._dc_sub_estimator, "_add_zero_branch_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_constraints):
                methods.append("_add_zero_branch_constraint_measurements")
            max_idx = self._max_measurement_idx_fast(self.measurements)
            self._call_sub_sequence_with_measurements(
                self._dc_sub_estimator,
                self.measurements,
                methods,
                summary_measurements=self._initial_measurement_sources_by_side()["dc"],
                summary_max_idx=max_idx,
                summary_voltage_node_mapper=self._sub_voltage_node_mapper(self._dc_sub_estimator),
                capture_attrs=("measurements", "measurement_table", "_max_measurement_idx"),
                captured_attrs=(captured_attrs := {}),
            )
            self._adopt_sub_measurement_update(captured_attrs)
        self._populate_side_device_pseudo_values()
        self._add_hybrid_pseudo_measurements()

    def _disable_coupled_ac_power_balance_rows(self) -> None:
        coupled = self._converter_coupled_ac_node_names()
        coupled_pos = self._converter_coupled_ac_device_positions()
        if not coupled and coupled_pos.size == 0:
            return
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            if table_size:
                rows_by_code = getattr(table, "rows_by_device_type_code", None) or {}
                balance_code = DEVICE_TYPE_CODES.get("ACPowerBalance")
                balance_rows = rows_by_code.get(balance_code) if balance_code is not None else None
                if balance_rows is None:
                    balance_rows = np.flatnonzero(np.asarray(table.device_type, dtype=object) == "ACPowerBalance")
                else:
                    balance_rows = np.asarray(balance_rows, dtype=np.int64)
                if balance_rows.size:
                    matched = np.zeros(balance_rows.size, dtype=bool)
                    if coupled_pos.size:
                        device_pos = getattr(table, "device_pos", None)
                        if device_pos is not None and np.asarray(device_pos).size == table_size:
                            matched |= np.isin(
                                np.asarray(device_pos, dtype=np.int64)[balance_rows.astype(np.intp, copy=False)],
                                coupled_pos,
                            )
                    if coupled and self._measurement_table_has_string_columns(table):
                        names = table.device_name
                        matched |= np.fromiter(
                            (str(names[int(row)]) in coupled for row in balance_rows),
                            dtype=bool,
                            count=int(balance_rows.size),
                        )
                    if np.any(matched):
                        disable_rows = balance_rows[matched]
                        table.valid[disable_rows] = False
                        measurement_table_status_code(table)[disable_rows] = MEAS_STATUS_INVALID
                        self._invalidate_measurement_activity_summary()
            incorporated = (
                self.measurements._incorporated_tail_size()
                if hasattr(self.measurements, "_incorporated_tail_size")
                else 0
            )
            if isinstance(self.measurements, list):
                for pos in range(incorporated, list.__len__(self.measurements)):
                    meas = list.__getitem__(self.measurements, pos)
                    if meas.device_type == "ACPowerBalance" and meas.device_name in coupled:
                        mark_measurement_invalid(meas)
            return
        for meas in self.measurements:
            if meas.device_type == "ACPowerBalance" and meas.device_name in coupled:
                mark_measurement_invalid(meas)

    def _populate_side_device_pseudo_values(self) -> None:
        ac = self._ac_sub_estimator
        if ac is not None:
            for gen in self.ac_generator_by_name.values():
                gen.p, gen.q = ac._generator_pseudo_power(gen)
            for load in self.ac_load_by_name.values():
                load.p, load.q = ac._load_pseudo_power(load)
        dc = self._dc_sub_estimator
        if dc is not None:
            for gen in self.dc_generator_by_name.values():
                gen.p = dc._generator_pseudo_power(gen)
            for load in self.dc_load_by_name.values():
                load.p = dc._load_pseudo_power(load)

    def _add_hybrid_pseudo_measurements(self) -> None:
        next_idx = self._max_measurement_idx_fast(self.measurements) + 1
        measured_devices = self._active_hybrid_converter_measurement_devices()
        specs_to_add = []
        for spec in self._hybrid_converter_measurement_specs(source="pseudo"):
            device_key = self._hybrid_converter_spec_device_key(spec)
            if device_key in measured_devices or (spec.device_type, spec.device_name) in measured_devices:
                continue
            if self._voltage_pseudo_is_covered(spec.device_type, spec.device_name, spec.meas_type):
                continue
            specs_to_add.append(spec)
        for spec in specs_to_add:
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_{spec.meas_type.lower()}_{spec.device_name}",
                spec.device_type,
                spec.device_name,
                spec.meas_type,
                spec.value,
            )

    def _active_hybrid_converter_measurement_devices(self) -> set:
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            source_count = int(getattr(self, "_sub_measurement_source_count", len(self.measurements)))
            return self._active_hybrid_converter_measurement_devices_from_table(table, row_limit=source_count)
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        if sources_by_side is None:
            return self._active_hybrid_converter_measurement_devices_from(self.measurements)
        return self._active_hybrid_converter_measurement_devices_from(sources_by_side.get("hybrid", ()))

    def _active_hybrid_converter_measurement_devices_from_table(self, table, row_limit: Optional[int] = None) -> set:
        row_count = int(table.idx.size)
        limit = row_count if row_limit is None else max(0, min(int(row_limit), row_count))
        if limit <= 0:
            return set()
        active = (
            np.asarray(table.valid[:limit], dtype=bool)
            & (np.asarray(table.weight[:limit], dtype=np.float64) > 0.0)
        )
        if not np.any(active):
            return set()
        code = np.asarray(table.device_type_code[:limit], dtype=np.int64)
        device_pos = self._hybrid_measurement_device_pos_array(table)[:limit]
        converter_mask = (
            active
            & (device_pos >= 0)
            & ((code == DEVICE_TYPE_DCACConverter) | (code == DEVICE_TYPE_ACACConverter))
        )
        if np.any(converter_mask):
            return set(self._packed_device_key_array(code[converter_mask], device_pos[converter_mask]).tolist())
        if self._measurement_table_has_string_columns(table):
            device_type = np.asarray(table.device_type[:limit], dtype=object)
            string_mask = active & np.isin(device_type, tuple(self._HYBRID_MEASUREMENT_DEVICE_TYPES))
            return set(
                zip(
                    device_type[string_mask].tolist(),
                    np.asarray(table.device_name[:limit], dtype=object)[string_mask].tolist(),
                )
            )
        return set()

    def _active_hybrid_converter_measurement_devices_from(self, measurements: Sequence[Measurement]) -> set:
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return self._active_hybrid_converter_measurement_devices_from_table(table)
        measured_devices = set()
        for meas in measurements:
            if (
                meas.valid
                and meas.weight > 0.0
                and meas.device_type in self._HYBRID_MEASUREMENT_DEVICE_TYPES
            ):
                measured_devices.add((meas.device_type, meas.device_name))
        return measured_devices

    # Module-style constants reused across the 60K+ per-converter state-meta
    # rebuilds during prepare; previously the dicts were re-created on every
    # call which dominated state-layout construction for large grids.
    _AC_STATE_META_PREFIX_BY_KIND = {
        "angle": "AC_THETA",
        "voltage": "AC_V",
        "generator_p": "AC_GEN_P",
        "generator_q": "AC_GEN_Q",
        "load_p": "AC_LOAD_P",
        "load_q": "AC_LOAD_Q",
        "shunt_q": "AC_Q_SHUNT",
    }
    _DC_STATE_META_PREFIX_BY_KIND = {
        "voltage": "DC_V",
        "zero_current": "DC_I",
        "break_current": "DC_I",
        "dcdc_p_from": "DCDC_P_FROM",
        "dcdc_p_to": "DCDC_P_TO",
        "v_generator_p": "DC_VGEN_P",
    }

    @staticmethod
    def _ac_sub_state_meta_to_hybrid(meta: StateMeta) -> StateMeta:
        kind = meta.kind
        prefix = HybridStateEstimator._AC_STATE_META_PREFIX_BY_KIND.get(kind)
        if prefix is None:
            if kind == "zero_current" or kind == "break_current":
                prefix = "AC_I_RE" if meta.component == "re" else "AC_I_IM"
            else:
                prefix = f"AC_{kind.upper()}"
        device_name = meta.device_name
        # Construct StateMeta inline (rather than going through
        # `with_legacy_label`) to skip one Python-level function call per state.
        return StateMeta(
            "ac",
            kind,
            meta.device_type,
            device_name,
            terminal=meta.terminal,
            component=meta.component,
            legacy_label=f"{prefix}:{device_name}",
            device_pos=meta.device_pos,
            device_type_code=meta.device_type_code,
            meas_type_code=meta.meas_type_code,
        )

    @staticmethod
    def _dc_sub_state_meta_to_hybrid(meta: StateMeta) -> StateMeta:
        kind = meta.kind
        prefix = HybridStateEstimator._DC_STATE_META_PREFIX_BY_KIND.get(kind)
        if prefix is None:
            prefix = f"DC_{kind.upper()}"
        device_name = meta.device_name
        return StateMeta(
            "dc",
            kind,
            meta.device_type,
            device_name,
            terminal=meta.terminal,
            component=meta.component,
            legacy_label=f"{prefix}:{device_name}",
            device_pos=meta.device_pos,
            device_type_code=meta.device_type_code,
            meas_type_code=meta.meas_type_code,
        )

    def _build_state_layout(self) -> None:
        ac_n = int(getattr(self._ac_sub_estimator, "n_state", 0) or 0)
        dc_n = int(getattr(self._dc_sub_estimator, "n_state", 0) or 0)
        state_meta: List[StateMeta] = [
            StateMeta("ac", "sub_state", "ACSubsystem", str(idx), legacy_label=f"AC_STATE:{idx}")
            for idx in range(ac_n)
        ]
        state_meta.extend(
            StateMeta("dc", "sub_state", "DCSubsystem", str(idx), legacy_label=f"DC_STATE:{idx}")
            for idx in range(dc_n)
        )
        self.dcac_state_start = ac_n + dc_n
        for pos, conv in enumerate(self.dcac_converters):
            state_meta.extend(
                (
                    StateMeta("hybrid", "dcac_p_dc", "DCACConverter", conv.name, terminal="dc", component="p", legacy_label=f"DCAC_P_DC:{conv.name}", device_pos=pos),
                    StateMeta("hybrid", "dcac_p_ac", "DCACConverter", conv.name, terminal="ac", component="p", legacy_label=f"DCAC_P_AC:{conv.name}", device_pos=pos),
                    StateMeta("hybrid", "dcac_q_ac", "DCACConverter", conv.name, terminal="ac", component="q", legacy_label=f"DCAC_Q_AC:{conv.name}", device_pos=pos),
                )
            )
        self.acac_state_start = len(state_meta)
        for pos, conv in enumerate(self.acac_converters):
            state_meta.extend(
                (
                    StateMeta("hybrid", "acac_p_from", "ACACConverter", conv.name, terminal="from", component="p", legacy_label=f"ACAC_P_FROM:{conv.name}", device_pos=pos),
                    StateMeta("hybrid", "acac_q_from", "ACACConverter", conv.name, terminal="from", component="q", legacy_label=f"ACAC_Q_FROM:{conv.name}", device_pos=pos),
                    StateMeta("hybrid", "acac_p_to", "ACACConverter", conv.name, terminal="to", component="p", legacy_label=f"ACAC_P_TO:{conv.name}", device_pos=pos),
                    StateMeta("hybrid", "acac_q_to", "ACACConverter", conv.name, terminal="to", component="q", legacy_label=f"ACAC_Q_TO:{conv.name}", device_pos=pos),
                )
            )
        self.ac_n_state = ac_n
        self.dc_n_state = dc_n
        self.hybrid_n_state = 3 * len(self.dcac_converters) + 4 * len(self.acac_converters)
        self.dc_state_start = ac_n
        self.hybrid_state_start = ac_n + dc_n
        self.n_state = len(state_meta)
        self.state_meta = state_meta
        labels = [meta.legacy_label for meta in state_meta]
        self.state_labels = labels if all(labels) else state_labels_from_metadata(self.state_meta)
        self.state_sides = ["ac"] * ac_n + ["dc"] * dc_n + ["hybrid"] * self.hybrid_n_state
        self.ac_state_labels = self.state_labels[:ac_n]
        self.dc_state_labels = self.state_labels[ac_n : ac_n + dc_n]
        self.ac_state_layout = {"state_labels": self.ac_state_labels, "n_state": ac_n}
        self.dc_state_layout = {"state_labels": self.dc_state_labels, "n_state": dc_n}
        self.dcac_p_dc_state_col = np.asarray(
            [self.dcac_state_start + 3 * idx for idx in range(len(self.dcac_converters))],
            dtype=np.int32,
        )
        self.dcac_p_ac_state_col = self.dcac_p_dc_state_col + 1
        self.dcac_q_ac_state_col = self.dcac_p_dc_state_col + 2
        self.acac_p_from_state_col = np.asarray(
            [self.acac_state_start + 4 * idx for idx in range(len(self.acac_converters))],
            dtype=np.int32,
        )
        self.acac_q_from_state_col = self.acac_p_from_state_col + 1
        self.acac_p_to_state_col = self.acac_p_from_state_col + 2
        self.acac_q_to_state_col = self.acac_p_from_state_col + 3
        self.ac_reference_nodes = getattr(self._ac_sub_estimator, "references", []) if self._ac_sub_estimator is not None else []
        self.dc_reference_nodes = getattr(self._dc_sub_estimator, "references", []) if self._dc_sub_estimator is not None else []
        self.ac_node_voltage_measurements = (
            getattr(self._ac_sub_estimator, "_node_voltage_measurement_cache", {})
            if self._ac_sub_estimator is not None
            else {}
        )
        self.dc_node_voltage_measurements = (
            getattr(self._dc_sub_estimator, "node_voltage_measurements", {})
            if self._dc_sub_estimator is not None
            else {}
        )
        self.ac_theta_state_col = (
            np.asarray(getattr(self._ac_sub_estimator, "angle_col", []), dtype=np.int32)
            if self._ac_sub_estimator is not None
            else np.array([], dtype=np.int32)
        )
        self.ac_voltage_state_col = (
            np.asarray(getattr(self._ac_sub_estimator, "voltage_col", []), dtype=np.int32)
            if self._ac_sub_estimator is not None
            else np.array([], dtype=np.int32)
        )
        if self._dc_sub_estimator is not None:
            dc_voltage_col = np.asarray(getattr(self._dc_sub_estimator, "voltage_col", []), dtype=np.int32)
            self.dc_voltage_state_col = np.where(dc_voltage_col >= 0, dc_voltage_col + self.dc_state_start, -1).astype(
                np.int32,
                copy=False,
            )
        else:
            self.dc_voltage_state_col = np.array([], dtype=np.int32)
        voltage_col_parts = []
        if self.ac_voltage_state_col.size:
            voltage_col_parts.append(self.ac_voltage_state_col[self.ac_voltage_state_col >= 0].astype(np.int32, copy=False))
        if self.dc_voltage_state_col.size:
            voltage_col_parts.append(self.dc_voltage_state_col[self.dc_voltage_state_col >= 0].astype(np.int32, copy=False))
        self.voltage_cols = np.concatenate(voltage_col_parts) if voltage_col_parts else np.array([], dtype=np.int32)
        self._build_hybrid_converter_array_cache()
        self._partition_state_variables()

    def _build_hybrid_converter_array_cache(self) -> None:
        self._dcac_count = len(self.dcac_converters)
        self._acac_count = len(self.acac_converters)
        self._dcac_dc_node = np.asarray([int(conv.dc_node) for conv in self.dcac_converters], dtype=np.int64)
        self._dcac_ac_node = np.asarray([int(conv.ac_node) for conv in self.dcac_converters], dtype=np.int64)
        self._acac_i_node = np.asarray([int(conv.i_node) for conv in self.acac_converters], dtype=np.int64)
        self._acac_j_node = np.asarray([int(conv.j_node) for conv in self.acac_converters], dtype=np.int64)
        self._dcac_dc_v_col_by_pos = np.fromiter(
            (self._dc_voltage_col_for_node(int(node)) for node in self._dcac_dc_node),
            dtype=np.int32,
            count=self._dcac_count,
        )
        self._dcac_ac_v_col_by_pos = np.fromiter(
            (self._ac_voltage_col_for_node(int(node)) for node in self._dcac_ac_node),
            dtype=np.int32,
            count=self._dcac_count,
        )
        self._dcac_dc_v_default_by_pos = np.fromiter(
            (self._dc_voltage_default_for_node(int(node)) for node in self._dcac_dc_node),
            dtype=np.float64,
            count=self._dcac_count,
        )
        self._dcac_ac_v_default_by_pos = np.fromiter(
            (self._ac_voltage_default_for_node(int(node)) for node in self._dcac_ac_node),
            dtype=np.float64,
            count=self._dcac_count,
        )
        self._acac_from_v_col_by_pos = np.fromiter(
            (self._ac_voltage_col_for_node(int(node)) for node in self._acac_i_node),
            dtype=np.int32,
            count=self._acac_count,
        )
        self._acac_to_v_col_by_pos = np.fromiter(
            (self._ac_voltage_col_for_node(int(node)) for node in self._acac_j_node),
            dtype=np.int32,
            count=self._acac_count,
        )
        self._acac_from_v_default_by_pos = np.fromiter(
            (self._ac_voltage_default_for_node(int(node)) for node in self._acac_i_node),
            dtype=np.float64,
            count=self._acac_count,
        )
        self._acac_to_v_default_by_pos = np.fromiter(
            (self._ac_voltage_default_for_node(int(node)) for node in self._acac_j_node),
            dtype=np.float64,
            count=self._acac_count,
        )
        flat_parts = []
        nonflat_parts = []
        if self._dcac_count:
            dcac_flat = np.zeros(3 * self._dcac_count, dtype=np.float64)
            dcac_flat[1::3] = np.asarray(
                [float(getattr(conv, "p_ac_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_flat[2::3] = np.asarray(
                [float(getattr(conv, "q_ac_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat = np.empty(3 * self._dcac_count, dtype=np.float64)
            dcac_nonflat[0::3] = np.asarray(
                [float(getattr(conv, "dc_p", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat[1::3] = np.asarray(
                [float(getattr(conv, "ac_p", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat[2::3] = np.asarray(
                [float(getattr(conv, "ac_q", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            flat_parts.append(dcac_flat)
            nonflat_parts.append(dcac_nonflat)
        if self._acac_count:
            p_set = np.asarray([float(getattr(conv, "p_set", 0.0) or 0.0) for conv in self.acac_converters], dtype=np.float64)
            acac_flat = np.empty(4 * self._acac_count, dtype=np.float64)
            acac_flat[0::4] = p_set
            acac_flat[1::4] = np.asarray(
                [float(getattr(conv, "i_q_set", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            acac_flat[2::4] = -p_set
            acac_flat[3::4] = np.asarray(
                [float(getattr(conv, "j_q_set", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            acac_nonflat = np.empty(4 * self._acac_count, dtype=np.float64)
            acac_nonflat[0::4] = np.asarray(
                [float(getattr(conv, "i_p", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            acac_nonflat[1::4] = np.asarray(
                [float(getattr(conv, "i_q", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            acac_nonflat[2::4] = np.asarray(
                [float(getattr(conv, "j_p", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            acac_nonflat[3::4] = np.asarray(
                [float(getattr(conv, "j_q", 0.0) or 0.0) for conv in self.acac_converters],
                dtype=np.float64,
            )
            flat_parts.append(acac_flat)
            nonflat_parts.append(acac_nonflat)
        self._hybrid_seed_flat_base = (
            np.concatenate(flat_parts).astype(np.float64, copy=False)
            if flat_parts
            else np.zeros(self.hybrid_n_state, dtype=np.float64)
        )
        self._hybrid_seed_nonflat_base = (
            np.concatenate(nonflat_parts).astype(np.float64, copy=False)
            if nonflat_parts
            else np.zeros(self.hybrid_n_state, dtype=np.float64)
        )

    def _refresh_active_measurement_state_layout(self) -> None:
        self._invalidate_measurement_activity_summary()
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_measurements,
        )
        partitions = partition_measurements_by_code(
            active_view.measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            side_by_device_type=self._MEASUREMENT_SIDE_BY_DEVICE_TYPE,
            table_builder=_measurement_table_from_measurements,
            rows_by_device_type_code=active_view.rows_by_device_type_code,
            as_view=True,
            sides=("ac", "dc", "hybrid"),
        )

        self.active_measurements = active_view.measurements
        self._active_measurement_source_count = len(self.measurements)
        self._active_measurement_source_table = active_view.source_table
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._ac_sub_to_hybrid_cols = np.arange(getattr(self, "ac_n_state", 0), dtype=np.int32)
        self._dc_sub_to_hybrid_cols = (
            self.dc_state_start + np.arange(getattr(self, "dc_n_state", 0), dtype=np.int32)
        )
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(self.ac_meas_rows, self.ac_meas),
            "dc": self._MeasurementSideBlock(self.dc_meas_rows, self.dc_meas),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_angle_residual_mask = active_view.angle_mask
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask)) if self.active_angle_residual_mask is not None else False
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._invalidate_real_voltage_seed_cache()
        self._rebuild_initial_state_cache()

    def _incremental_update_active_measurement_state_layout(
        self,
        appended_measurements: Sequence[Measurement],
    ) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            return False
        if any((not meas.valid) or float(meas.weight) <= 0.0 for meas in appended_measurements):
            return False
        master_table = getattr(self.measurements, "table", None)
        active_table = getattr(self.active_measurements, "table", None)
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
        self.measurements.table = concat_measurement_tables(master_table, appended_list.table)

        active_view = append_active_measurement_view(
            build_active_measurement_view(self.active_measurements, table_builder=_measurement_table_from_measurements),
            appended_list,
            source_row_start=len(master_table.idx),
            table_builder=_measurement_table_from_measurements,
        )
        active_start = len(self.active_measurements)
        self.active_measurements = active_view.measurements
        self._active_measurement_source_count = len(self.measurements)
        self._active_measurement_source_table = self.measurements.table
        partitions = extend_measurement_partitions(
            self._active_measurement_blocks_as_partitions(),
            list(appended_list),
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            row_offset=active_start,
            side_by_device_type=self._MEASUREMENT_SIDE_BY_DEVICE_TYPE,
            table_builder=_measurement_table_from_measurements,
            sides=("ac", "dc", "hybrid"),
        )
        self.ac_meas_rows = np.asarray(partitions.rows["ac"], dtype=np.int32)
        self.dc_meas_rows = np.asarray(partitions.rows["dc"], dtype=np.int32)
        self.hybrid_meas_rows = np.asarray(partitions.rows["hybrid"], dtype=np.int32)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(self.ac_meas_rows, self.ac_meas),
            "dc": self._MeasurementSideBlock(self.dc_meas_rows, self.dc_meas),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        self.active_z = np.asarray(self.active_measurements.table.value, dtype=np.float64)
        self.active_weight = np.asarray(self.active_measurements.table.weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(self.active_measurements.table.angle_mask, dtype=bool)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        if self._has_real_voltage_seed_measurement(appended_measurements):
            self._invalidate_real_voltage_seed_cache()
            self._invalidate_initial_state_cache()
        self._invalidate_measurement_activity_summary()
        return True

    def _active_measurement_blocks_as_partitions(self):
        return MeasurementPartitions(
            measurements={"ac": self.ac_meas, "dc": self.dc_meas, "hybrid": self.hybrid_meas},
            rows={
                "ac": np.asarray(self.ac_meas_rows, dtype=np.int64),
                "dc": np.asarray(self.dc_meas_rows, dtype=np.int64),
                "hybrid": np.asarray(self.hybrid_meas_rows, dtype=np.int64),
            },
        )

    @staticmethod
    def _shrink_partition_rows(rows: Sequence[int], removed_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        row_array = np.asarray(rows, dtype=np.int64)
        keep = row_array != int(removed_pos)
        shrunk = row_array[keep]
        shrunk = shrunk - (shrunk > int(removed_pos))
        return shrunk.astype(np.int32, copy=False), keep

    def _shrink_active_measurement_state_layout(self, removed_pos: int) -> MeasurementList:
        removed_measurement = self.active_measurements[int(removed_pos)]
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        self.active_measurements = take_measurement_view(self.active_measurements, keep_rows)
        source_measurements = getattr(self, "measurements", self.active_measurements)
        self._active_measurement_source_count = len(source_measurements)
        self._active_measurement_source_table = getattr(source_measurements, "table", None)
        if not hasattr(self, "ac_meas_rows"):
            if self._has_real_voltage_seed_measurement([removed_measurement]):
                self._invalidate_real_voltage_seed_cache()
                self._invalidate_initial_state_cache()
            return self.active_measurements
        self.ac_meas_rows, ac_keep = self._shrink_partition_rows(self.ac_meas_rows, removed_pos)
        self.dc_meas_rows, dc_keep = self._shrink_partition_rows(self.dc_meas_rows, removed_pos)
        self.hybrid_meas_rows, hybrid_keep = self._shrink_partition_rows(self.hybrid_meas_rows, removed_pos)
        self.ac_meas = take_measurement_view(self.ac_meas, np.flatnonzero(ac_keep).astype(np.int64, copy=False))
        self.dc_meas = take_measurement_view(self.dc_meas, np.flatnonzero(dc_keep).astype(np.int64, copy=False))
        self.hybrid_meas = take_measurement_view(
            self.hybrid_meas,
            np.flatnonzero(hybrid_keep).astype(np.int64, copy=False),
        )
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(self.ac_meas_rows, self.ac_meas),
            "dc": self._MeasurementSideBlock(self.dc_meas_rows, self.dc_meas),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        active_table = self.active_measurements.table
        self.active_z = np.asarray(active_table.value, dtype=np.float64)
        self.active_weight = np.asarray(active_table.weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(active_table.angle_mask, dtype=bool)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        if not hasattr(self, "n_state"):
            return self.active_measurements
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._active_has_angle_residuals = bool(np.any(self.active_angle_residual_mask))
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        if self._has_real_voltage_seed_measurement([removed_measurement]):
            self._invalidate_real_voltage_seed_cache()
            self._invalidate_initial_state_cache()
        self._invalidate_measurement_activity_summary()
        return self.active_measurements

    def _build_hybrid_measurement_plan(
        self,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        dcac_code = int(DEVICE_TYPE_CODES["DCACConverter"])
        acac_code = int(DEVICE_TYPE_CODES["ACACConverter"])
        meas_kind_code_by_type_code = {
            dcac_code: self._measurement_kind_code_lookup(self._DCAC_MEASUREMENT_CODE),
            acac_code: self._measurement_kind_code_lookup(self._ACAC_MEASUREMENT_CODE),
        }
        plan_table = build_measurement_plan_table(
            block.measurements,
            device_pos_by_type_code={
                dcac_code: self.dcac_pos_by_name,
                acac_code: self.acac_pos_by_name,
            },
            meas_kind_by_type_code={
                dcac_code: self._DCAC_MEASUREMENT_CODE,
                acac_code: self._ACAC_MEASUREMENT_CODE,
            },
            device_pos_by_type_code_id=self._measurement_plan_device_lookup_by_master_ids(),
            meas_kind_code_by_type_code=meas_kind_code_by_type_code,
            require_index_arrays=True,
            table_builder=_measurement_table_from_measurements,
        )

        dcac_mask = plan_table.handled & (plan_table.device_type_code == dcac_code)
        dcac_local_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_pos = plan_table.device_pos[dcac_mask].astype(np.int32, copy=False)
        dcac_rows = block.rows[dcac_local_rows].astype(np.int32, copy=False)
        dcac_codes = plan_table.meas_kind[dcac_mask].astype(np.int8, copy=False)
        dcac_dc_v_col = self._dcac_dc_v_col_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.int32)
        dcac_ac_v_col = self._dcac_ac_v_col_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.int32)
        dcac_dc_v_default = (
            self._dcac_dc_v_default_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.float64)
        )
        dcac_ac_v_default = (
            self._dcac_ac_v_default_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.float64)
        )

        acac_mask = plan_table.handled & (plan_table.device_type_code == acac_code)
        acac_local_rows = plan_table.row[acac_mask].astype(np.int64, copy=False)
        acac_pos = plan_table.device_pos[acac_mask].astype(np.int32, copy=False)
        acac_rows = block.rows[acac_local_rows].astype(np.int32, copy=False)
        acac_codes = plan_table.meas_kind[acac_mask].astype(np.int8, copy=False)
        acac_from_v_col = self._acac_from_v_col_by_pos[acac_pos] if acac_pos.size else np.array([], dtype=np.int32)
        acac_to_v_col = self._acac_to_v_col_by_pos[acac_pos] if acac_pos.size else np.array([], dtype=np.int32)
        acac_from_v_default = (
            self._acac_from_v_default_by_pos[acac_pos] if acac_pos.size else np.array([], dtype=np.float64)
        )
        acac_to_v_default = (
            self._acac_to_v_default_by_pos[acac_pos] if acac_pos.size else np.array([], dtype=np.float64)
        )

        # Per-code buckets pre-slice all per-iteration indexing arrays so that
        # evaluate/jacobian can run without recomputing `codes == k` masks or
        # filtering invalid voltage columns. v_col-related fields default to
        # empty for codes that don't read a voltage state.
        dc_state_start = int(self.dc_state_start)

        def _empty_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            return self._HybridCodeBucket(
                rows=np.array([], dtype=np.int32),
                hx_p=np.array([], dtype=np.int32),
                hx_q=np.array([], dtype=np.int32),
                jcol_p=np.array([], dtype=np.int32),
                jcol_q=np.array([], dtype=np.int32),
                v_default=np.array([], dtype=np.float64),
                v_use_x=np.array([], dtype=bool),
                v_x_index=np.array([], dtype=np.int32),
                v_jcol=np.array([], dtype=np.int32),
                v_any_x=False,
            )

        def _voltage_fields(
            cols_subset: np.ndarray,
            defaults_subset: np.ndarray,
            offset: int,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
            """Pack voltage-gather info: (default, use_x, x_index, jcol, any_x)."""
            if cols_subset.size == 0:
                return (
                    np.array([], dtype=np.float64),
                    np.array([], dtype=bool),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    False,
                )
            cols_int = cols_subset.astype(np.int32, copy=False)
            use_x = cols_int >= 0
            any_x = bool(use_x.any())
            x_index = (cols_int[use_x] - int(offset)).astype(np.int32, copy=False) if any_x else np.array([], dtype=np.int32)
            return (
                defaults_subset.astype(np.float64, copy=False),
                use_x,
                x_index,
                cols_int,
                any_x,
            )

        def _build_dcac_linear_power_bucket(code: int, hx_offset: int, state_col_arr: np.ndarray) -> "HybridStateEstimator._HybridCodeBucket":
            """For DCAC codes 1/2/3: identity observation of a hybrid power state."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == code
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False) + hx_offset).astype(np.int32, copy=False)
            jcol_p = state_col_arr[pos_k].astype(np.int32, copy=False)
            return self._HybridCodeBucket(
                rows=rows_k,
                hx_p=hx_p,
                hx_q=np.array([], dtype=np.int32),
                jcol_p=jcol_p,
                jcol_q=np.array([], dtype=np.int32),
                v_default=np.array([], dtype=np.float64),
                v_use_x=np.array([], dtype=bool),
                v_x_index=np.array([], dtype=np.int32),
                v_jcol=np.array([], dtype=np.int32),
                v_any_x=False,
            )

        def _build_voltage_only_bucket(
            codes_arr: np.ndarray,
            rows_arr: np.ndarray,
            v_col_full: np.ndarray,
            v_default_full: np.ndarray,
            code: int,
            offset: int,
        ) -> "HybridStateEstimator._HybridCodeBucket":
            """For codes that observe a voltage directly (V_DC, V_AC, V_FROM, V_TO)."""
            if codes_arr.size == 0:
                return _empty_bucket()
            mask = codes_arr == code
            if not mask.any():
                return _empty_bucket()
            rows_k = rows_arr[mask]
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                v_col_full[mask], v_default_full[mask], offset,
            )
            return self._HybridCodeBucket(
                rows=rows_k,
                hx_p=np.array([], dtype=np.int32),
                hx_q=np.array([], dtype=np.int32),
                jcol_p=np.array([], dtype=np.int32),
                jcol_q=np.array([], dtype=np.int32),
                v_default=v_default,
                v_use_x=v_use_x,
                v_x_index=v_x_index,
                v_jcol=v_jcol,
                v_any_x=v_any_x,
            )

        def _build_dcac_i_dc_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            """DCAC code 5: i_dc = p_dc / v_dc."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == 5
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False)).astype(np.int32, copy=False)
            jcol_p = self.dcac_p_dc_state_col[pos_k].astype(np.int32, copy=False)
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                dcac_dc_v_col[mask], dcac_dc_v_default[mask], dc_state_start,
            )
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=np.array([], dtype=np.int32),
                jcol_p=jcol_p, jcol_q=np.array([], dtype=np.int32),
                v_default=v_default, v_use_x=v_use_x, v_x_index=v_x_index,
                v_jcol=v_jcol, v_any_x=v_any_x,
            )

        def _build_dcac_i_ac_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            """DCAC code 7: i_ac = hypot(p_ac, q_ac) / v_ac."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == 7
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False) + 1).astype(np.int32, copy=False)
            hx_q = (3 * pos_k.astype(np.int32, copy=False) + 2).astype(np.int32, copy=False)
            jcol_p = self.dcac_p_ac_state_col[pos_k].astype(np.int32, copy=False)
            jcol_q = self.dcac_q_ac_state_col[pos_k].astype(np.int32, copy=False)
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                dcac_ac_v_col[mask], dcac_ac_v_default[mask], 0,
            )
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=hx_q,
                jcol_p=jcol_p, jcol_q=jcol_q,
                v_default=v_default, v_use_x=v_use_x, v_x_index=v_x_index,
                v_jcol=v_jcol, v_any_x=v_any_x,
            )

        dcac_p_dc_bucket = _build_dcac_linear_power_bucket(1, 0, self.dcac_p_dc_state_col)
        dcac_p_ac_bucket = _build_dcac_linear_power_bucket(2, 1, self.dcac_p_ac_state_col)
        dcac_q_ac_bucket = _build_dcac_linear_power_bucket(3, 2, self.dcac_q_ac_state_col)
        dcac_v_dc_bucket = _build_voltage_only_bucket(
            dcac_codes, dcac_rows, dcac_dc_v_col, dcac_dc_v_default, 4, dc_state_start,
        )
        dcac_i_dc_bucket = _build_dcac_i_dc_bucket()
        dcac_v_ac_bucket = _build_voltage_only_bucket(
            dcac_codes, dcac_rows, dcac_ac_v_col, dcac_ac_v_default, 6, 0,
        )
        dcac_i_ac_bucket = _build_dcac_i_ac_bucket()

        acac_hybrid_base = 3 * len(self.dcac_converters)

        def _build_acac_linear_power_bucket(code: int, hx_offset: int, state_col_arr: np.ndarray) -> "HybridStateEstimator._HybridCodeBucket":
            if acac_pos.size == 0:
                return _empty_bucket()
            mask = acac_codes == code
            if not mask.any():
                return _empty_bucket()
            pos_k = acac_pos[mask]
            rows_k = acac_rows[mask]
            hx_p = (acac_hybrid_base + 4 * pos_k.astype(np.int32, copy=False) + hx_offset).astype(np.int32, copy=False)
            jcol_p = state_col_arr[pos_k].astype(np.int32, copy=False)
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=np.array([], dtype=np.int32),
                jcol_p=jcol_p, jcol_q=np.array([], dtype=np.int32),
                v_default=np.array([], dtype=np.float64),
                v_use_x=np.array([], dtype=bool),
                v_x_index=np.array([], dtype=np.int32),
                v_jcol=np.array([], dtype=np.int32),
                v_any_x=False,
            )

        def _build_acac_current_bucket(
            code: int,
            p_hx_offset: int,
            q_hx_offset: int,
            p_state_col: np.ndarray,
            q_state_col: np.ndarray,
            v_col_full: np.ndarray,
            v_default_full: np.ndarray,
        ) -> "HybridStateEstimator._HybridCodeBucket":
            """ACAC current codes (I_FROM=6, I_TO=8): hypot(p, q) / v."""
            if acac_pos.size == 0:
                return _empty_bucket()
            mask = acac_codes == code
            if not mask.any():
                return _empty_bucket()
            pos_k = acac_pos[mask]
            rows_k = acac_rows[mask]
            hx_p = (acac_hybrid_base + 4 * pos_k.astype(np.int32, copy=False) + p_hx_offset).astype(np.int32, copy=False)
            hx_q = (acac_hybrid_base + 4 * pos_k.astype(np.int32, copy=False) + q_hx_offset).astype(np.int32, copy=False)
            jcol_p = p_state_col[pos_k].astype(np.int32, copy=False)
            jcol_q = q_state_col[pos_k].astype(np.int32, copy=False)
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                v_col_full[mask], v_default_full[mask], 0,
            )
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=hx_q,
                jcol_p=jcol_p, jcol_q=jcol_q,
                v_default=v_default, v_use_x=v_use_x, v_x_index=v_x_index,
                v_jcol=v_jcol, v_any_x=v_any_x,
            )

        acac_p_from_bucket = _build_acac_linear_power_bucket(1, 0, self.acac_p_from_state_col)
        acac_q_from_bucket = _build_acac_linear_power_bucket(2, 1, self.acac_q_from_state_col)
        acac_p_to_bucket = _build_acac_linear_power_bucket(3, 2, self.acac_p_to_state_col)
        acac_q_to_bucket = _build_acac_linear_power_bucket(4, 3, self.acac_q_to_state_col)
        acac_v_from_bucket = _build_voltage_only_bucket(
            acac_codes, acac_rows, acac_from_v_col, acac_from_v_default, 5, 0,
        )
        acac_i_from_bucket = _build_acac_current_bucket(
            6, 0, 1,
            self.acac_p_from_state_col, self.acac_q_from_state_col,
            acac_from_v_col, acac_from_v_default,
        )
        acac_v_to_bucket = _build_voltage_only_bucket(
            acac_codes, acac_rows, acac_to_v_col, acac_to_v_default, 7, 0,
        )
        acac_i_to_bucket = _build_acac_current_bucket(
            8, 2, 3,
            self.acac_p_to_state_col, self.acac_q_to_state_col,
            acac_to_v_col, acac_to_v_default,
        )

        return self._HybridMeasurementPlan(
            dcac_rows=dcac_rows,
            dcac_codes=dcac_codes,
            dcac_pos=dcac_pos,
            dcac_dc_v_col=dcac_dc_v_col,
            dcac_ac_v_col=dcac_ac_v_col,
            dcac_dc_v_default=dcac_dc_v_default,
            dcac_ac_v_default=dcac_ac_v_default,
            acac_rows=acac_rows,
            acac_codes=acac_codes,
            acac_pos=acac_pos,
            acac_from_v_col=acac_from_v_col,
            acac_to_v_col=acac_to_v_col,
            acac_from_v_default=acac_from_v_default,
            acac_to_v_default=acac_to_v_default,
            dcac_p_dc=dcac_p_dc_bucket,
            dcac_p_ac=dcac_p_ac_bucket,
            dcac_q_ac=dcac_q_ac_bucket,
            dcac_v_dc=dcac_v_dc_bucket,
            dcac_i_dc=dcac_i_dc_bucket,
            dcac_v_ac=dcac_v_ac_bucket,
            dcac_i_ac=dcac_i_ac_bucket,
            acac_p_from=acac_p_from_bucket,
            acac_q_from=acac_q_from_bucket,
            acac_p_to=acac_p_to_bucket,
            acac_q_to=acac_q_to_bucket,
            acac_v_from=acac_v_from_bucket,
            acac_i_from=acac_i_from_bucket,
            acac_v_to=acac_v_to_bucket,
            acac_i_to=acac_i_to_bucket,
        )

    def _hybrid_measurement_plan_for(
        self,
        measurements: Sequence[Measurement],
        blocks: Optional[Dict[str, "HybridStateEstimator._MeasurementSideBlock"]] = None,
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        if measurements is self.active_measurements:
            return self._active_hybrid_measurement_plan
        side_blocks = self._measurement_blocks_for(measurements) if blocks is None else blocks
        return self._build_hybrid_measurement_plan(side_blocks["hybrid"])

    def _build_hybrid_seed_measurement_plan(
        self,
        measurements: Sequence[Measurement],
    ) -> "HybridStateEstimator._HybridSeedMeasurementPlan":
        dcac_code = int(DEVICE_TYPE_CODES["DCACConverter"])
        acac_code = int(DEVICE_TYPE_CODES["ACACConverter"])
        plan_table = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={
                dcac_code: self.dcac_pos_by_name,
                acac_code: self.acac_pos_by_name,
            },
            meas_kind_by_type_code={
                dcac_code: {"P_DC": 0, "P_AC": 1, "Q_AC": 2},
                acac_code: {"P_FROM": 0, "Q_FROM": 1, "P_TO": 2, "Q_TO": 3},
            },
            table_builder=_measurement_table_from_measurements,
        )
        active_mask = (
            plan_table.handled
            & np.asarray(plan_table.table.valid, dtype=bool)
            & (np.asarray(plan_table.table.weight, dtype=np.float64) > 0.0)
        )
        dcac_mask = active_mask & (plan_table.device_type_code == dcac_code)
        acac_mask = active_mask & (plan_table.device_type_code == acac_code)

        dcac_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_state_col = (
            3 * plan_table.device_pos[dcac_mask].astype(np.int64, copy=False)
            + plan_table.meas_kind[dcac_mask].astype(np.int64, copy=False)
        )
        acac_rows = plan_table.row[acac_mask].astype(np.int64, copy=False)
        acac_state_col = (
            3 * self._dcac_count
            + 4 * plan_table.device_pos[acac_mask].astype(np.int64, copy=False)
            + plan_table.meas_kind[acac_mask].astype(np.int64, copy=False)
        )
        return self._HybridSeedMeasurementPlan(
            measurement_row=np.concatenate((dcac_rows, acac_rows)).astype(np.int64, copy=False),
            state_col=np.concatenate((dcac_state_col, acac_state_col)).astype(np.int64, copy=False),
        )

    def _hybrid_converter_measurement_specs(
        self,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        def spec(device_type: str, device_name: str, meas_type: str, value: float, device_pos: int) -> "HybridStateEstimator._HybridConverterMeasurementSpec":
            return self._HybridConverterMeasurementSpec(
                device_type,
                device_name,
                meas_type,
                value,
                int(DEVICE_TYPE_CODES.get(device_type, 0)),
                int(device_pos),
                int(MEAS_TYPE_CODES.get(meas_type, 0)),
            )

        for pos, conv in enumerate(self.dcac_converters):
            p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "i_p", 0.0) or 0.0)
            p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_p", 0.0) or 0.0)
            q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_q", 0.0) or 0.0)
            specs.extend(
                (
                    spec("DCACConverter", conv.name, "P_DC", p_dc, pos),
                    spec("DCACConverter", conv.name, "P_AC", p_ac, pos),
                    spec("DCACConverter", conv.name, "Q_AC", q_ac, pos),
                    spec(
                        "DCACConverter",
                        conv.name,
                        "V_DC",
                        float(getattr(getattr(conv, "dc_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                    spec(
                        "DCACConverter",
                        conv.name,
                        "V_AC",
                        float(getattr(getattr(conv, "ac_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                )
            )
        for pos, conv in enumerate(self.acac_converters):
            specs.extend(
                (
                    spec("ACACConverter", conv.name, "P_FROM", float(getattr(conv, "i_p", 0.0) or 0.0), pos),
                    spec("ACACConverter", conv.name, "Q_FROM", float(getattr(conv, "i_q", 0.0) or 0.0), pos),
                    spec("ACACConverter", conv.name, "P_TO", float(getattr(conv, "j_p", 0.0) or 0.0), pos),
                    spec("ACACConverter", conv.name, "Q_TO", float(getattr(conv, "j_q", 0.0) or 0.0), pos),
                    spec(
                        "ACACConverter",
                        conv.name,
                        "V_FROM",
                        float(getattr(getattr(conv, "i_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                    spec(
                        "ACACConverter",
                        conv.name,
                        "V_TO",
                        float(getattr(getattr(conv, "j_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                )
            )
        return specs

    def state_layout(self) -> Dict[str, object]:
        return {
            "state_labels": self.state_labels,
            "state_meta": self.state_meta,
            "state_sides": self.state_sides,
            "ac_state_slice": self.ac_state_slice,
            "dc_state_slice": self.dc_state_slice,
            "hybrid_state_slice": self.hybrid_state_slice,
            "n_state": self.n_state,
        }

    def _hybrid_seed_vector(self, flat: Optional[bool] = None) -> np.ndarray:
        flat = self.flat_start if flat is None else bool(flat)
        base = self._hybrid_seed_flat_base if flat else self._hybrid_seed_nonflat_base
        x = base.copy()
        self._seed_hybrid_power_states_from_measurements(x)
        return x

    def _seed_hybrid_power_states_from_measurements(self, x: np.ndarray) -> None:
        measurements = getattr(self, "hybrid_meas", None) or self.measurements
        if measurements is getattr(self, "hybrid_meas", None) and hasattr(self, "_active_hybrid_measurement_plan"):
            plan = self._active_hybrid_measurement_plan
            dcac_mask = (plan.dcac_codes >= 1) & (plan.dcac_codes <= 3)
            acac_mask = (plan.acac_codes >= 1) & (plan.acac_codes <= 4)
            if not np.any(dcac_mask) and not np.any(acac_mask):
                return
            rows = []
            cols = []
            if np.any(dcac_mask):
                rows.append(plan.dcac_rows[dcac_mask].astype(np.int64, copy=False))
                cols.append(
                    (
                        3 * plan.dcac_pos[dcac_mask].astype(np.int64, copy=False)
                        + plan.dcac_codes[dcac_mask].astype(np.int64, copy=False)
                        - 1
                    )
                )
            if np.any(acac_mask):
                rows.append(plan.acac_rows[acac_mask].astype(np.int64, copy=False))
                cols.append(
                    (
                        3 * self._dcac_count
                        + 4 * plan.acac_pos[acac_mask].astype(np.int64, copy=False)
                        + plan.acac_codes[acac_mask].astype(np.int64, copy=False)
                        - 1
                    )
                )
            x[np.concatenate(cols)] = self.active_z[np.concatenate(rows)]
            return
        seed_plan = self._hybrid_seed_measurement_plan_for(measurements)
        if seed_plan.measurement_row.size == 0:
            return
        table = getattr(measurements, "table", None)
        if table is None or len(table.idx) != len(measurements):
            table = _measurement_table_from_measurements(measurements)
            try:
                measurements.table = table
            except AttributeError:
                pass
        x[seed_plan.state_col] = np.asarray(table.value, dtype=np.float64)[seed_plan.measurement_row]

    def _hybrid_seed_measurement_plan_for(
        self,
        measurements: Sequence[Measurement],
    ) -> "HybridStateEstimator._HybridSeedMeasurementPlan":
        key = (id(measurements), len(measurements))
        cached = self._hybrid_seed_measurement_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._build_hybrid_seed_measurement_plan(measurements)
        self._hybrid_seed_measurement_plan_cache[key] = plan
        return plan

    def initial_state(self, flat: Optional[bool] = None) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.initial_state()
        flat = self.flat_start if flat is None else bool(flat)
        if self._initial_state_cache_ready:
            cached = self.flat_state if flat else self.power_flow_state
            if cached.size == getattr(self, "n_state", 0):
                return cached.copy()
        if hasattr(self, "active_measurements") and getattr(self, "n_state", 0) > 0:
            self._rebuild_initial_state_cache(flat=flat)
            cached = self.flat_state if flat else self.power_flow_state
            if cached.size == getattr(self, "n_state", 0):
                return cached.copy()
        return self._build_initial_state(flat=flat)

    def _build_initial_state(self, flat: bool) -> np.ndarray:
        parts = []
        if self._ac_sub_estimator is not None:
            original = self._ac_sub_estimator.flat_start
            try:
                self._ac_sub_estimator.flat_start = flat
                parts.append(self._ac_sub_estimator.initial_state())
            finally:
                self._ac_sub_estimator.flat_start = original
        if self._dc_sub_estimator is not None:
            original = self._dc_sub_estimator.flat_start
            try:
                self._dc_sub_estimator.flat_start = flat
                parts.append(self._dc_sub_estimator.initial_state())
            finally:
                self._dc_sub_estimator.flat_start = original
        parts.append(self._hybrid_seed_vector(flat=flat))
        x = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        if not flat:
            self._seed_state_from_real_voltage_measurements(x)
        return x

    def _rebuild_initial_state_cache(self, flat: Optional[bool] = None) -> None:
        self._initial_state_cache_ready = False
        flat = self.flat_start if flat is None else bool(flat)
        if flat:
            self.flat_state = self._build_initial_state(flat=True)
        else:
            self.power_flow_state = self._build_initial_state(flat=False)
        self._initial_state_cache_ready = True

    def _invalidate_initial_state_cache(self) -> None:
        self._initial_state_cache_ready = False
        self._hybrid_seed_measurement_plan_cache = {}

    def _seed_state_from_real_voltage_measurements(self, x: np.ndarray) -> None:
        """Seed coupled voltage states from converter-side real voltage rows."""
        measurements = getattr(self, "hybrid_meas", None)
        if measurements is None:
            measurements = getattr(self, "active_measurements", self.measurements)
        cols, values = self._real_voltage_seed_arrays(measurements)
        if cols.size:
            x[cols] = values

    def _invalidate_real_voltage_seed_cache(self) -> None:
        self._real_voltage_seed_cache = {}

    @staticmethod
    def _has_real_voltage_seed_measurement(measurements: Sequence[Measurement]) -> bool:
        voltage_types = {"V", "V_FROM", "V_TO", "V_GEN", "V_LOAD", "V_DC", "V_AC"}
        for meas in measurements:
            if (
                bool(getattr(meas, "valid", True))
                and float(getattr(meas, "weight", 0.0)) > 0.0
                and not is_pseudo_measurement(meas)
                and str(getattr(meas, "meas_type", "")).upper() in voltage_types
            ):
                return True
        return False

    def _real_voltage_seed_arrays(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        key = (id(measurements), len(measurements))
        cached = self._real_voltage_seed_cache.get(key)
        if cached is not None:
            return cached
        if measurements is getattr(self, "hybrid_meas", None) and hasattr(self, "_active_hybrid_measurement_plan"):
            result = self._real_voltage_seed_arrays_from_active_hybrid_plan()
            self._real_voltage_seed_cache[key] = result
            return result
        best: Dict[int, Tuple[float, float]] = {}
        voltage_types = {"V", "V_FROM", "V_TO", "V_GEN", "V_LOAD", "V_DC", "V_AC"}
        for meas in measurements:
            if (
                not bool(getattr(meas, "valid", True))
                or float(getattr(meas, "weight", 0.0)) <= 0.0
                or is_pseudo_measurement(meas)
            ):
                continue
            meas_type = str(getattr(meas, "meas_type", "")).upper()
            if meas_type not in voltage_types:
                continue
            weight = float(meas.weight)
            value = float(meas.value)
            ac_node_idx = self._ac_voltage_measurement_node_idx(meas.device_type, meas.device_name, meas_type)
            if ac_node_idx is not None:
                col = self._ac_voltage_col_for_node(int(ac_node_idx))
                current = best.get(col)
                if current is None or weight > current[0]:
                    if col >= 0:
                        best[col] = (weight, value)
            dc_node_idx = self._dc_voltage_measurement_node_idx(meas.device_type, meas.device_name, meas_type)
            if dc_node_idx is not None:
                col = self._dc_voltage_col_for_node(int(dc_node_idx))
                current = best.get(col)
                if current is None or weight > current[0]:
                    if col >= 0:
                        best[col] = (weight, value)
        if best:
            cols = np.fromiter(best.keys(), dtype=np.int32, count=len(best))
            values = np.fromiter((item[1] for item in best.values()), dtype=np.float64, count=len(best))
        else:
            cols = np.array([], dtype=np.int32)
            values = np.array([], dtype=np.float64)
        self._real_voltage_seed_cache[key] = (cols, values)
        return cols, values

    def _real_voltage_seed_arrays_from_active_hybrid_plan(self) -> Tuple[np.ndarray, np.ndarray]:
        plan = self._active_hybrid_measurement_plan
        active_table = getattr(self.active_measurements, "table", None)
        status_code = (
            measurement_table_status_code(active_table)
            if active_table is not None and len(active_table.idx) == len(self.active_measurements)
            else None
        )
        row_chunks = []
        col_chunks = []

        def append_rows(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
            if status_code is not None and np.any(mask):
                selected_rows = rows[mask]
                mask_indices = np.flatnonzero(mask)
                real_mask = status_code[selected_rows.astype(np.int64, copy=False)] != MEAS_STATUS_PSEUDO
                selected_rows = selected_rows[real_mask]
                selected_cols = cols[mask_indices[real_mask]]
            else:
                selected_rows = rows[mask]
                selected_cols = cols[mask]
            valid_col = selected_cols >= 0
            if np.any(valid_col):
                row_chunks.append(selected_rows[valid_col].astype(np.int64, copy=False))
                col_chunks.append(selected_cols[valid_col].astype(np.int32, copy=False))

        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE["V_DC"], plan.dcac_rows, plan.dcac_dc_v_col)
        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE["V_AC"], plan.dcac_rows, plan.dcac_ac_v_col)
        append_rows(plan.acac_codes == self._ACAC_MEASUREMENT_CODE["V_FROM"], plan.acac_rows, plan.acac_from_v_col)
        append_rows(plan.acac_codes == self._ACAC_MEASUREMENT_CODE["V_TO"], plan.acac_rows, plan.acac_to_v_col)
        if not row_chunks:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        rows = np.concatenate(row_chunks).astype(np.int64, copy=False)
        cols = np.concatenate(col_chunks).astype(np.int32, copy=False)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        cols = cols[order]
        values = self.active_z[rows]
        weights = self.active_weight[rows]
        best: Dict[int, Tuple[float, float]] = {}
        for col, weight, value in zip(cols, weights, values):
            col_int = int(col)
            current = best.get(col_int)
            if current is None or float(weight) > current[0]:
                best[col_int] = (float(weight), float(value))
        if not best:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        return (
            np.fromiter(best.keys(), dtype=np.int32, count=len(best)),
            np.fromiter((item[1] for item in best.values()), dtype=np.float64, count=len(best)),
        )

    def _split_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        ac_x = x[: self.ac_n_state]
        dc_x = x[self.dc_state_start : self.dc_state_start + self.dc_n_state]
        hybrid_x = x[self.hybrid_state_start :]
        return ac_x, dc_x, hybrid_x

    def _expand_state(self, x: np.ndarray) -> np.ndarray:
        delegate = self._delegate()
        if isinstance(delegate, DCStateEstimator):
            compact = np.asarray(x, dtype=np.float64)
            voltage = np.ones(len(delegate.nodes), dtype=np.float64)
            for pos, node in enumerate(delegate.nodes):
                col = int(delegate.voltage_col[pos])
                if col >= 0:
                    voltage[pos] = compact[col]
                else:
                    voltage[pos] = float(
                        self.dc_node_voltage_measurements.get(node.idx, getattr(node, "voltage", 1.0) or 1.0)
                    )
            return np.concatenate((voltage, compact[delegate.n_voltage :]))
        return np.asarray(x, dtype=np.float64).copy()

    def _expand_state_mapped_only(self, x: np.ndarray) -> np.ndarray:
        return self._expand_state(x)

    def _ac_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._ac_sub_estimator is None:
            return None
        return np.asarray(x, dtype=np.float64)[: self.ac_n_state].copy()

    def _dc_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._dc_sub_estimator is None:
            return None
        start = self.dc_state_start
        return np.asarray(x, dtype=np.float64)[start : start + self.dc_n_state].copy()

    @staticmethod
    def _append_sparse_rows_unchecked(H, rows, cols, values) -> None:
        if rows is None or cols is None or values is None:
            return
        if hasattr(H, "rows") and hasattr(H, "cols") and hasattr(H, "data"):
            rr = np.asarray(rows, dtype=np.int32)
            cc = np.asarray(cols, dtype=np.int32)
            vv = np.asarray(values, dtype=np.float64)
            mask = (cc >= 0) & (vv != 0.0)
            if rr.size == 1 and cc.size:
                H.rows.extend([int(rr[0])] * int(np.count_nonzero(mask)))
                H.cols.extend(cc[mask].tolist())
                H.data.extend(vv[mask].tolist())
            elif rr.size == cc.size:
                H.rows.extend(rr[mask].tolist())
                H.cols.extend(cc[mask].tolist())
                H.data.extend(vv[mask].tolist())

    def _ac_branch_power_derivatives(self, *args, **kwargs):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    def _ac_branch_current_derivatives(self, *args, **kwargs):
        return np.array([], dtype=np.float64)

    def _converter_coupled_ac_node_names(self) -> set:
        names = set()
        for conv in self.dcac_converters:
            node = getattr(conv, "ac_node_obj", None)
            if node is not None:
                names.add(node.name)
        for conv in self.acac_converters:
            for attr in ("i_node_obj", "j_node_obj"):
                node = getattr(conv, attr, None)
                if node is not None:
                    names.add(node.name)
        return names

    def _converter_coupled_ac_device_positions(self) -> np.ndarray:
        node_ids = []
        dcac_ac_node = np.asarray(getattr(self, "_dcac_ac_node", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if dcac_ac_node.size:
            node_ids.extend(dcac_ac_node.tolist())
        else:
            node_ids.extend(int(conv.ac_node) for conv in self.dcac_converters)
        acac_i_node = np.asarray(getattr(self, "_acac_i_node", np.asarray([], dtype=np.int64)), dtype=np.int64)
        acac_j_node = np.asarray(getattr(self, "_acac_j_node", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if acac_i_node.size or acac_j_node.size:
            node_ids.extend(acac_i_node.tolist())
            node_ids.extend(acac_j_node.tolist())
        else:
            for conv in self.acac_converters:
                node_ids.append(int(conv.i_node))
                node_ids.append(int(conv.j_node))
        if not node_ids or self._ac_sub_estimator is None:
            return np.asarray([], dtype=np.int64)
        ac = self._ac_sub_estimator
        ids = np.asarray(getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        pos = np.asarray(getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if ids.size == 0 or pos.size != ids.size:
            return np.asarray([], dtype=np.int64)
        query = np.unique(np.asarray(node_ids, dtype=np.int64))
        loc = np.searchsorted(ids, query)
        valid = (loc >= 0) & (loc < ids.size)
        if np.any(valid):
            valid_locs = loc[valid].astype(np.intp, copy=False)
            valid[valid] = ids[valid_locs] == query[valid]
        if not np.any(valid):
            return np.asarray([], dtype=np.int64)
        return np.unique(pos[loc[valid].astype(np.intp, copy=False)].astype(np.int64, copy=False))

    def _ac_voltage_measurement_node_idx(
        self,
        device_type: str,
        device_name: str,
        meas_type: str,
    ) -> Optional[int]:
        """Return the AC-side node associated with a voltage measurement row."""
        if device_type == "ACNode":
            if meas_type == "V" and device_name in self.ac_node_by_name:
                return int(self.ac_node_by_name[device_name].idx)
            return None
        if device_type == "ACGenerator":
            gen = self.ac_generator_by_name.get(device_name)
            if meas_type == "V_GEN" and gen is not None:
                return int(gen.node)
            return None
        if device_type == "ACLoad":
            load = self.ac_load_by_name.get(device_name)
            if meas_type == "V_LOAD" and load is not None:
                return int(load.node)
            return None
        if device_type in ("ACBranch", "ACTransformer", "ACZeroBranch", "ACBreak", "ACACConverter"):
            if device_type == "ACBranch":
                dev = self.ac_branch_by_name.get(device_name)
            elif device_type == "ACTransformer":
                dev = self.ac_transformer_by_name.get(device_name)
            elif device_type == "ACZeroBranch":
                dev = self.ac_zero_branch_by_name.get(device_name)
            elif device_type == "ACBreak":
                dev = self.ac_break_by_name.get(device_name)
            else:
                dev = self.acac_by_name.get(device_name)
            if dev is None:
                return None
            if meas_type == "V_FROM":
                return int(dev.i_node)
            if meas_type == "V_TO":
                return int(dev.j_node)
            return None
        if device_type == "DCACConverter":
            conv = self.dcac_by_name.get(device_name)
            if meas_type == "V_AC" and conv is not None:
                return int(conv.ac_node)
        return None

    def _dc_voltage_measurement_node_idx(
        self,
        device_type: str,
        device_name: str,
        meas_type: str,
    ) -> Optional[int]:
        """Return the DC-side node associated with a voltage measurement row."""
        if device_type == "DCNode":
            if meas_type == "V" and device_name in self.dc_node_by_name:
                return int(self.dc_node_by_name[device_name].idx)
            return None
        if device_type == "DCGenerator":
            gen = self.dc_generator_by_name.get(device_name)
            if meas_type == "V_GEN" and gen is not None:
                return int(gen.node)
            return None
        if device_type == "DCLoad":
            load = self.dc_load_by_name.get(device_name)
            if meas_type == "V_LOAD" and load is not None:
                return int(load.node)
            return None
        if device_type in ("DCBranch", "DCZeroBranch", "DCBreak", "DCDCConverter"):
            if device_type == "DCBranch":
                dev = self.dc_branch_by_name.get(device_name)
            elif device_type == "DCZeroBranch":
                dev = self.dc_zero_branch_by_name.get(device_name)
            elif device_type == "DCBreak":
                dev = self.dc_break_by_name.get(device_name)
            else:
                dev = self.dcdc_by_name.get(device_name)
            if dev is None:
                return None
            if meas_type == "V_FROM":
                return int(dev.i_node)
            if meas_type == "V_TO":
                return int(dev.j_node)
            return None
        if device_type == "DCACConverter":
            conv = self.dcac_by_name.get(device_name)
            if meas_type == "V_DC" and conv is not None:
                return int(conv.dc_node)
        return None

    def _real_voltage_observation_nodes(self, side: str) -> Dict[int, float]:
        """Return side nodes covered by real usable voltage measurements on any device."""
        cache_name = f"_{side}_real_voltage_observation_node_cache"
        cache = getattr(self, cache_name, None)
        if cache is not None:
            return cache
        if side == "ac":
            mapper = self._ac_voltage_measurement_node_idx
            node_pos = getattr(self._ac_sub_estimator, "node_pos", {})
            sub_cache = getattr(self._ac_sub_estimator, "_real_voltage_observation_node_cache", None)
        else:
            mapper = self._dc_voltage_measurement_node_idx
            node_pos = getattr(self._dc_sub_estimator, "node_pos", {})
            sub_cache = getattr(self._dc_sub_estimator, "_real_voltage_observation_node_cache", None)
        best: Dict[int, Tuple[float, float]] = {}
        if isinstance(sub_cache, dict):
            for node_idx, value in sub_cache.items():
                best[int(node_idx)] = (float("inf"), float(value))
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            status_code = measurement_table_status_code(table)
            active_real = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
            )
            if self._measurement_table_has_string_columns(table):
                meas_type = np.asarray(table.meas_type, dtype=object)
                voltage_mask = active_real & np.isin(meas_type, self._VOLTAGE_MEASUREMENT_TYPE_TUPLE)
                voltage_rows = np.flatnonzero(voltage_mask)
                device_type = np.asarray(table.device_type, dtype=object)
                device_name = np.asarray(table.device_name, dtype=object)
                for row, dev_type, dev_name, meas_type_value in zip(
                    voltage_rows,
                    device_type[voltage_mask],
                    device_name[voltage_mask],
                    meas_type[voltage_mask],
                ):
                    node_idx = mapper(str(dev_type), str(dev_name), str(meas_type_value))
                    if node_idx is None or node_idx not in node_pos:
                        continue
                    weight = float(table.weight[row])
                    current = best.get(int(node_idx))
                    if current is None or weight > current[0]:
                        best[int(node_idx)] = (weight, float(table.value[row]))
            else:
                self._add_hybrid_converter_real_voltage_observations(side, table, active_real, best)
        else:
            for meas in self.measurements:
                if (
                    not meas.valid
                    or meas.weight <= 0.0
                    or is_pseudo_measurement(meas)
                    or str(meas.meas_type).upper() not in self._VOLTAGE_MEASUREMENT_TYPES
                ):
                    continue
                node_idx = mapper(meas.device_type, meas.device_name, meas.meas_type)
                if node_idx is None or node_idx not in node_pos:
                    continue
                current = best.get(node_idx)
                if current is None or float(meas.weight) > current[0]:
                    best[node_idx] = (float(meas.weight), float(meas.value))
        cache = {node_idx: value for node_idx, (_weight, value) in best.items()}
        setattr(self, cache_name, cache)
        return cache

    def _add_hybrid_converter_real_voltage_observations(
        self,
        side: str,
        table,
        active_real: np.ndarray,
        best: Dict[int, Tuple[float, float]],
    ) -> None:
        meas_type_code = getattr(table, "meas_type_code", None)
        if meas_type_code is None or np.asarray(meas_type_code).size != int(table.idx.size):
            return
        device_pos = self._hybrid_measurement_device_pos_array(table)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        table_weight = np.asarray(table.weight, dtype=np.float64)
        table_value = np.asarray(table.value, dtype=np.float64)

        def update(rows: np.ndarray, node_by_pos: np.ndarray) -> None:
            if rows.size == 0 or node_by_pos.size == 0:
                return
            pos = device_pos[rows.astype(np.intp, copy=False)]
            in_range = (pos >= 0) & (pos < node_by_pos.size)
            if not np.any(in_range):
                return
            rows_valid = rows[in_range]
            pos_valid = pos[in_range].astype(np.intp, copy=False)
            for row, node_idx in zip(rows_valid, node_by_pos[pos_valid]):
                weight = float(table_weight[int(row)])
                current = best.get(int(node_idx))
                if current is None or weight > current[0]:
                    best[int(node_idx)] = (weight, float(table_value[int(row)]))

        if side == "dc":
            dcac_dc_node = np.asarray(
                getattr(self, "_dcac_dc_node", [int(conv.dc_node) for conv in self.dcac_converters]),
                dtype=np.int64,
            )
            rows = np.flatnonzero(
                active_real
                & (device_type_code == DEVICE_TYPE_DCACConverter)
                & (meas_type_code == MEAS_TYPE_V_DC)
                & (device_pos >= 0)
            )
            update(rows, dcac_dc_node)
            return

        dcac_ac_node = np.asarray(
            getattr(self, "_dcac_ac_node", [int(conv.ac_node) for conv in self.dcac_converters]),
            dtype=np.int64,
        )
        acac_i_node = np.asarray(
            getattr(self, "_acac_i_node", [int(conv.i_node) for conv in self.acac_converters]),
            dtype=np.int64,
        )
        acac_j_node = np.asarray(
            getattr(self, "_acac_j_node", [int(conv.j_node) for conv in self.acac_converters]),
            dtype=np.int64,
        )
        update(
            np.flatnonzero(
                active_real
                & (device_type_code == DEVICE_TYPE_DCACConverter)
                & (meas_type_code == MEAS_TYPE_V_AC)
                & (device_pos >= 0)
            ),
            dcac_ac_node,
        )
        update(
            np.flatnonzero(
                active_real
                & (device_type_code == DEVICE_TYPE_ACACConverter)
                & (meas_type_code == MEAS_TYPE_V_FROM)
                & (device_pos >= 0)
            ),
            acac_i_node,
        )
        update(
            np.flatnonzero(
                active_real
                & (device_type_code == DEVICE_TYPE_ACACConverter)
                & (meas_type_code == MEAS_TYPE_V_TO)
                & (device_pos >= 0)
            ),
            acac_j_node,
        )

    def _real_voltage_observation_value_for_side_node(
        self,
        side: str,
        node_idx: Optional[int],
    ) -> Optional[float]:
        """Return a real side voltage on the node or its compressed zero-tie component."""
        if node_idx is None:
            return None
        observed = self._real_voltage_observation_nodes(side)
        if node_idx in observed:
            return float(observed[node_idx])
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator
        node_pos = getattr(estimator, "node_pos", {})
        pos = node_pos.get(node_idx)
        component_by_pos = getattr(estimator, "zero_tie_component_by_pos", None)
        components = getattr(estimator, "zero_tie_components", None)
        nodes = getattr(estimator, "nodes", None)
        if pos is None or component_by_pos is None or not components or nodes is None:
            return None
        component = components[int(component_by_pos[int(pos)])]
        for member_pos in component:
            member_idx = nodes[int(member_pos)].idx
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_pseudo_is_covered(self, device_type: str, device_name: str, meas_type: str) -> bool:
        """Check whether a voltage pseudo row is redundant because the node already has real V data."""
        ac_node_idx = self._ac_voltage_measurement_node_idx(device_type, device_name, meas_type)
        if ac_node_idx is not None:
            return self._real_voltage_observation_value_for_side_node("ac", ac_node_idx) is not None
        dc_node_idx = self._dc_voltage_measurement_node_idx(device_type, device_name, meas_type)
        if dc_node_idx is not None:
            return self._real_voltage_observation_value_for_side_node("dc", dc_node_idx) is not None
        return False

    def _active_measurement_keys(self) -> set:
        return set(self._measurement_activity_summary().active_keys)

    def _invalidate_measurement_activity_summary(self) -> None:
        if hasattr(self, "_measurement_activity_summary_cache"):
            delattr(self, "_measurement_activity_summary_cache")
        if hasattr(self, "_sub_measurement_global_summary_attrs"):
            self._sub_measurement_global_summary_attrs = None
        if hasattr(self, "_observability_pseudo_candidate_cache"):
            delattr(self, "_observability_pseudo_candidate_cache")
        if hasattr(self, "_ac_real_voltage_observation_node_cache"):
            delattr(self, "_ac_real_voltage_observation_node_cache")
        if hasattr(self, "_dc_real_voltage_observation_node_cache"):
            delattr(self, "_dc_real_voltage_observation_node_cache")

    def _measurement_activity_summary(self) -> "_MeasurementActivitySummary":
        cache = getattr(self, "_measurement_activity_summary_cache", None)
        if cache is not None:
            return cache
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            table = _measurement_table_from_measurements(self.measurements)
            try:
                self.measurements.table = table
            except AttributeError:
                pass
        active_table = self._current_active_measurement_table(table)
        if active_table is not None:
            max_idx = int(np.asarray(table.idx, dtype=np.int64).max()) if table.idx.size else 0
            active_mask = np.asarray(active_table.valid, dtype=bool) & (
                np.asarray(active_table.weight, dtype=np.float64) > 0.0
            )
            measured_devices, active_keys = self._activity_key_sets_from_table(active_table, active_mask)
            cache = self._MeasurementActivitySummary(
                max_idx=max_idx,
                measured_devices=measured_devices,
                active_keys=active_keys,
            )
            self._measurement_activity_summary_cache = cache
            return cache
        idx = np.asarray(table.idx, dtype=np.int64)
        valid = np.asarray(table.valid, dtype=bool)
        weight = np.asarray(table.weight, dtype=np.float64)
        active_mask = valid & (weight > 0.0)
        max_idx = int(idx.max()) if idx.size else 0
        measured_devices, active_keys = self._activity_key_sets_from_table(table, active_mask)
        cache = self._MeasurementActivitySummary(max_idx=max_idx, measured_devices=measured_devices, active_keys=active_keys)
        self._measurement_activity_summary_cache = cache
        return cache

    def _current_active_measurement_table(self, source_table):
        active_measurements = getattr(self, "active_measurements", None)
        active_table = getattr(active_measurements, "table", None)
        if active_measurements is None or active_table is None:
            return None
        if len(active_table.idx) != len(active_measurements):
            return None
        if getattr(self, "_active_measurement_source_count", None) != len(self.measurements):
            return None
        if getattr(self, "_active_measurement_source_table", None) is not source_table:
            return None
        return active_table

    def _next_measurement_idx(self) -> int:
        return int(self._measurement_activity_summary().max_idx) + 1

    @staticmethod
    def _converter_meas_type_for_state_meta(meta: StateMeta) -> Optional[str]:
        mapping = {
            "dcac_p_dc": "P_DC",
            "dcac_p_ac": "P_AC",
            "dcac_q_ac": "Q_AC",
            "acac_p_from": "P_FROM",
            "acac_q_from": "Q_FROM",
            "acac_p_to": "P_TO",
            "acac_q_to": "Q_TO",
        }
        return mapping.get(meta.kind)

    def _targeted_converter_specs_for_state(
        self,
        meta: StateMeta,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        meas_type = self._converter_meas_type_for_state_meta(meta)
        if meas_type is None:
            return []
        device_type_code = int(meta.device_type_code or DEVICE_TYPE_CODES.get(meta.device_type, 0))
        device_pos = int(meta.device_pos)
        meas_type_code = int(MEAS_TYPE_CODES.get(meas_type, 0))
        return [
            spec
            for spec in self._hybrid_converter_measurement_specs(source=source)
            if int(spec.device_type_code) == device_type_code
            and int(spec.device_pos) == device_pos
            and int(spec.meas_type_code) == meas_type_code
        ]

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch hybrid rank deficiencies and optional post-observability redundancy."""
        delegate = self._delegate()
        if delegate is not None:
            return int(getattr(delegate, "targeted_observability_pseudo_count", 0))
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
            existing_keys = self._active_measurement_keys()
            existing_names = self._measurement_name_set_fast(self.measurements)
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
                    existing_names,
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
                refreshed = self._incremental_update_active_measurement_state_layout(
                    self.measurements[measurement_count_before:]
                )
            if not refreshed:
                self._refresh_active_measurement_state_layout()
            observability = None
        if observability is None and total_added < max_count:
            observability = self.observability_analysis()
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
        self._invalidate_measurement_activity_summary()
        if refresh:
            refreshed = self._incremental_update_active_measurement_state_layout(
                self.measurements[measurement_count_before:]
            )
            if not refreshed:
                self._refresh_active_measurement_state_layout()
        return len(selected)

    def _add_redundant_observability_pseudo_measurements(self, max_add: int) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add)

    def _observability_pseudo_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        cache = getattr(self, "_observability_pseudo_candidate_cache", None)
        if cache is not None:
            return cache
        existing_keys = self._active_measurement_keys()
        existing_names = self._measurement_name_set_fast(self.measurements)
        candidates: List[Measurement] = []

        def add(
            device_type: str,
            device_name: str,
            meas_type: str,
            value: float,
            device_type_code: int = 0,
            device_pos: int = -1,
            meas_type_code: int = 0,
        ) -> None:
            device_type_code = int(device_type_code or DEVICE_TYPE_CODES.get(device_type, 0))
            meas_type_code = int(meas_type_code or MEAS_TYPE_CODES.get(meas_type, 0))
            key = self._packed_measurement_key(device_type_code, int(device_pos), meas_type_code)
            legacy_key = (device_type, device_name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if self._voltage_pseudo_is_covered(device_type, device_name, meas_type):
                return
            if key in existing_keys or legacy_key in existing_keys or pseudo_name in existing_names:
                return
            candidates.append(
                Measurement(
                    0,
                    pseudo_name,
                    device_type,
                    device_name,
                    meas_type,
                    self.pseudo_measurement_weight,
                    True,
                    float(value),
                    MEAS_STATUS_PSEUDO,
                )
            )
            existing_keys.add(key if key >= 0 else legacy_key)
            existing_names.add(pseudo_name)

        for spec in self._side_observability_pseudo_specs("ac"):
            add(spec.device_type, spec.device_name, spec.meas_type, spec.value)
        for spec in self._side_observability_pseudo_specs("dc"):
            add(spec.device_type, spec.device_name, spec.meas_type, spec.value)

        for spec in self._hybrid_converter_measurement_specs(source="flow"):
            add(
                spec.device_type,
                spec.device_name,
                spec.meas_type,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )

        self._observability_pseudo_candidate_cache = candidates
        return candidates

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: Sequence[Measurement],
        max_add: int,
    ) -> List[Measurement]:
        if max_add <= 0 or not candidates:
            return []
        x = self.initial_state()
        cache = self._observability_matrix_cache_for(observability, self.active_measurements, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, self.active_measurements)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return list(candidates[:max_add])
        candidate_h = None
        if cache is not None:
            candidate_cache = cache.setdefault("candidate_jacobians", {})
            signature = self._measurement_signature(candidates)
            candidate_h = candidate_cache.get(signature)
        if candidate_h is None:
            candidate_h = self.jacobian_sparse(x, candidates)
            if cache is not None:
                candidate_cache[signature] = candidate_h
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != len(candidates) or not np.any(scores > 0.0):
            return list(candidates[:max_add])
        order = np.argsort(-scores, kind="stable")
        selected = [candidates[int(pos)] for pos in order[:max_add] if scores[int(pos)] > 0.0]
        return selected or list(candidates[:max_add])

    def _side_observability_pseudo_specs(
        self,
        side: str,
    ) -> List["_HybridConverterMeasurementSpec"]:
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        if side == "ac":
            for node in sorted(self.ac_node_by_name.values(), key=lambda item: item.idx):
                specs.append(self._HybridConverterMeasurementSpec("ACNode", node.name, "V", float(getattr(node, "voltage", 1.0) or 1.0)))
            for load in sorted(self.ac_load_by_name.values(), key=lambda item: item.idx):
                p, q = self._ac_sub_estimator._load_pseudo_power(load) if self._ac_sub_estimator is not None else (0.0, 0.0)
                voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
                specs.extend(
                    (
                        self._HybridConverterMeasurementSpec("ACLoad", load.name, "P_LOAD", p),
                        self._HybridConverterMeasurementSpec("ACLoad", load.name, "Q_LOAD", q),
                        self._HybridConverterMeasurementSpec("ACLoad", load.name, "V_LOAD", voltage),
                    )
                )
            for gen in sorted(self.ac_generator_by_name.values(), key=lambda item: item.idx):
                p, q = self._ac_sub_estimator._generator_pseudo_power(gen) if self._ac_sub_estimator is not None else (0.0, 0.0)
                voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
                specs.extend(
                    (
                        self._HybridConverterMeasurementSpec("ACGenerator", gen.name, "P_GEN", p),
                        self._HybridConverterMeasurementSpec("ACGenerator", gen.name, "Q_GEN", q),
                        self._HybridConverterMeasurementSpec("ACGenerator", gen.name, "V_GEN", voltage),
                    )
                )
            for device_type, devices in (("ACBranch", self.ac_branch_by_name), ("ACTransformer", self.ac_transformer_by_name)):
                for dev in sorted(devices.values(), key=lambda item: item.idx):
                    specs.extend(
                        (
                            self._HybridConverterMeasurementSpec(device_type, dev.name, "P_FROM", float(getattr(dev, "i_p", 0.0) or 0.0)),
                            self._HybridConverterMeasurementSpec(device_type, dev.name, "Q_FROM", float(getattr(dev, "i_q", 0.0) or 0.0)),
                            self._HybridConverterMeasurementSpec(device_type, dev.name, "P_TO", float(getattr(dev, "j_p", 0.0) or 0.0)),
                            self._HybridConverterMeasurementSpec(device_type, dev.name, "Q_TO", float(getattr(dev, "j_q", 0.0) or 0.0)),
                        )
                    )
            return specs
        if side == "dc":
            for node in sorted(self.dc_node_by_name.values(), key=lambda item: item.idx):
                specs.append(self._HybridConverterMeasurementSpec("DCNode", node.name, "V", float(getattr(node, "voltage", 1.0) or 1.0)))
            for load in sorted(self.dc_load_by_name.values(), key=lambda item: item.idx):
                p = self._dc_sub_estimator._load_pseudo_power(load) if self._dc_sub_estimator is not None else 0.0
                voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
                specs.extend(
                    (
                        self._HybridConverterMeasurementSpec("DCLoad", load.name, "P_LOAD", p),
                        self._HybridConverterMeasurementSpec("DCLoad", load.name, "V_LOAD", voltage),
                    )
                )
            for gen in sorted(self.dc_generator_by_name.values(), key=lambda item: item.idx):
                p = self._dc_sub_estimator._generator_pseudo_power(gen) if self._dc_sub_estimator is not None else 0.0
                voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
                specs.extend(
                    (
                        self._HybridConverterMeasurementSpec("DCGenerator", gen.name, "P_GEN", p),
                        self._HybridConverterMeasurementSpec("DCGenerator", gen.name, "V_GEN", voltage),
                    )
                )
            for branch in sorted(self.dc_branch_by_name.values(), key=lambda item: item.idx):
                specs.extend(
                    (
                        self._HybridConverterMeasurementSpec("DCBranch", branch.name, "P_FROM", float(getattr(branch, "i_p", 0.0) or 0.0)),
                        self._HybridConverterMeasurementSpec("DCBranch", branch.name, "P_TO", float(getattr(branch, "j_p", 0.0) or 0.0)),
                    )
                )
            return specs
        return specs

    def _targeted_side_specs_for_state(
        self,
        meta: StateMeta,
        existing_keys: set,
    ) -> List["_HybridConverterMeasurementSpec"]:
        name = meta.device_name
        if meta.side == "ac" and meta.kind in ("zero_current", "break_current"):
            if name in self.ac_zero_branch_by_name:
                dev_type = "ACZeroBranch"
                dev = self.ac_zero_branch_by_name[name]
            elif name in self.ac_break_by_name:
                dev_type = "ACBreak"
                dev = self.ac_break_by_name[name]
            else:
                return []
            p = float(getattr(dev, "p", 0.0) or 0.0)
            q = float(getattr(dev, "q", 0.0) or 0.0)
            p_type = "P_TO" if (dev_type, name, "P_FROM") in existing_keys else "P_FROM"
            q_type = "Q_TO" if (dev_type, name, "Q_FROM") in existing_keys else "Q_FROM"
            return [
                self._HybridConverterMeasurementSpec(dev_type, name, p_type, -p if p_type == "P_TO" else p),
                self._HybridConverterMeasurementSpec(dev_type, name, q_type, -q if q_type == "Q_TO" else q),
            ]
        if meta.side == "dc" and meta.kind in ("zero_current", "break_current"):
            if name in self.dc_zero_branch_by_name:
                dev_type = "DCZeroBranch"
                dev = self.dc_zero_branch_by_name[name]
            elif name in self.dc_break_by_name:
                dev_type = "DCBreak"
                dev = self.dc_break_by_name[name]
            else:
                return []
            return [self._HybridConverterMeasurementSpec(dev_type, name, "I_FROM", float(getattr(dev, "current", 0.0) or 0.0))]
        return []

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        if max_add <= 0:
            return next_idx, 0
        meta = state_meta_at(self.state_meta, state_idx)
        if meta is None:
            return next_idx, 0
        name = meta.device_name
        if meta.side == "ac" and meta.kind == "angle":
            return next_idx, 0
        added = 0

        def add(device_type: str, meas_type: str, value: float) -> None:
            nonlocal next_idx, added
            if added >= max_add:
                return
            device_type_code = int(DEVICE_TYPE_CODES.get(device_type, 0))
            meas_type_code = int(MEAS_TYPE_CODES.get(meas_type, 0))
            key = self._packed_measurement_key(device_type_code, int(meta.device_pos), meas_type_code)
            legacy_key = (device_type, name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{name}"
            if self._voltage_pseudo_is_covered(device_type, name, meas_type):
                return
            if key in existing_keys or legacy_key in existing_keys or pseudo_name in existing_names:
                return
            next_idx = self._append_pseudo_measurement(next_idx, pseudo_name, device_type, name, meas_type, value)
            existing_keys.add(key if key >= 0 else legacy_key)
            existing_names.add(pseudo_name)
            added += 1

        if meta.kind == "voltage" and meta.device_type == "ACNode" and name in self.ac_node_by_name:
            node = self.ac_node_by_name[name]
            add("ACNode", "V", float(getattr(node, "voltage", 1.0) or 1.0))
            return next_idx, added
        if meta.kind == "voltage" and meta.device_type == "DCNode" and name in self.dc_node_by_name:
            node = self.dc_node_by_name[name]
            add("DCNode", "V", float(getattr(node, "voltage", 1.0) or 1.0))
            return next_idx, added
        converter_specs = self._targeted_converter_specs_for_state(meta, source="flow")
        if converter_specs:
            for spec in converter_specs:
                add(spec.device_type, spec.meas_type, spec.value)
            return next_idx, added
        side_specs = self._targeted_side_specs_for_state(meta, existing_keys)
        if side_specs:
            for spec in side_specs:
                add(spec.device_type, spec.meas_type, spec.value)
        return next_idx, added

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        if isinstance(measurements, list):
            return measurements
        return list(measurements)

    def _measurement_table(self, measurements: Sequence[Measurement]):
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return table
        table = _measurement_table_from_measurements(measurements)
        if isinstance(measurements, MeasurementList):
            measurements.table = table
        return table

    def _measurement_vectors(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
        table = self._measurement_table(measurements)
        return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)

    @staticmethod
    def _measurement_signature(measurements: Sequence[Measurement]) -> Tuple[Tuple[str, str, str, float, float, bool], ...]:
        return tuple(
            (
                str(meas.device_type),
                str(meas.device_name),
                str(meas.meas_type),
                float(meas.value),
                float(meas.weight),
                bool(meas.valid),
            )
            for meas in measurements
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
            "candidate_jacobians": {},
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

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _angle_residual_mask(self, measurements: Sequence[Measurement]) -> np.ndarray:
        if measurements is self.active_measurements:
            return self.active_angle_residual_mask
        table = self._measurement_table(measurements)
        return np.asarray(table.angle_mask, dtype=bool)

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if measurements is self.active_measurements:
            angle_mask = self.active_angle_residual_mask
            has_angle = self._active_has_angle_residuals
        else:
            angle_mask = self._angle_residual_mask(measurements)
            has_angle = bool(angle_mask is not None and np.any(angle_mask))
        if out is None:
            residual = np.subtract(z, z_est, dtype=np.float64)
        else:
            np.subtract(z, z_est, out=out)
            residual = out
        if has_angle:
            residual[angle_mask] = (residual[angle_mask] + np.pi) % (2.0 * np.pi) - np.pi
        return residual

    def _safe_current(self, p: np.ndarray, q: Optional[np.ndarray], v: np.ndarray) -> np.ndarray:
        power = np.abs(p) if q is None else np.hypot(p, q)
        return np.divide(power, v, out=np.zeros_like(power, dtype=np.float64), where=np.abs(v) > self.min_current_voltage)

    @staticmethod
    def _state_values_from_cols(
        x: np.ndarray,
        cols: np.ndarray,
        defaults: np.ndarray,
        offset: int = 0,
    ) -> np.ndarray:
        values = defaults.copy()
        mask = cols >= 0
        if np.any(mask):
            values[mask] = x[cols[mask] - int(offset)]
        return values

    @staticmethod
    def _lookup_node_pos_by_idx(node_idx: int, ids: np.ndarray, pos: np.ndarray) -> int:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        if ids.size == 0 or pos.size == 0:
            return -1
        loc = int(np.searchsorted(ids, int(node_idx)))
        if loc >= ids.size or int(ids[loc]) != int(node_idx):
            return -1
        value = int(pos[loc])
        return value if value >= 0 else -1

    def _ac_node_pos_for_idx(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        if ac is None:
            return -1
        return self._lookup_node_pos_by_idx(
            int(node_idx),
            getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    @staticmethod
    def _measurement_kind_code_lookup(kind_by_name: Dict[str, int]) -> np.ndarray:
        if not kind_by_name:
            return np.asarray([], dtype=np.int16)
        codes = [int(MEAS_TYPE_CODES.get(str(name), -1)) for name in kind_by_name]
        valid_codes = [code for code in codes if code >= 0]
        if not valid_codes:
            return np.asarray([], dtype=np.int16)
        lookup = np.empty(max(valid_codes) + 1, dtype=np.int16)
        lookup.fill(-1)
        for name, kind in kind_by_name.items():
            code = int(MEAS_TYPE_CODES.get(str(name), -1))
            if code >= 0:
                lookup[code] = int(kind)
        return lookup

    def _dc_node_pos_for_idx(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        if dc is None:
            return -1
        return self._lookup_node_pos_by_idx(
            int(node_idx),
            getattr(dc, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(dc, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    def _ac_voltage_col_for_node(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        pos = self._ac_node_pos_for_idx(int(node_idx))
        if ac is None or pos < 0:
            return -1
        col = int(ac.voltage_col[pos])
        return col if col >= 0 else -1

    def _ac_voltage_default_for_node(self, node_idx: int) -> float:
        ac = self._ac_sub_estimator
        pos = self._ac_node_pos_for_idx(int(node_idx))
        if ac is not None and pos >= 0:
            if pos in getattr(ac, "ref_voltages", {}):
                return float(ac.ref_voltages[pos])
            file_voltage = getattr(ac, "file_voltage", None)
            if file_voltage is not None and pos < int(np.asarray(file_voltage).size):
                value = float(np.asarray(file_voltage, dtype=np.float64)[pos])
                return value if value != 0.0 else 1.0
        node = self.ac_node_by_idx.get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _dc_voltage_default_for_node(self, node_idx: int) -> float:
        dc = self._dc_sub_estimator
        pos = self._dc_node_pos_for_idx(int(node_idx))
        if dc is not None and pos >= 0:
            if pos in getattr(dc, "ref_voltages", {}):
                return float(dc.ref_voltages[pos])
            node_voltage = getattr(dc, "_node_voltage_by_pos", None)
            if node_voltage is not None and pos < int(np.asarray(node_voltage).size):
                value = float(np.asarray(node_voltage, dtype=np.float64)[pos])
                return value if value != 0.0 else 1.0
        node = self.dc_node_by_idx.get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _dc_voltage_col_for_node(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        pos = self._dc_node_pos_for_idx(int(node_idx))
        if dc is None or pos < 0:
            return -1
        col = int(dc.voltage_col[pos])
        return self.dc_state_start + col if col >= 0 else -1

    def _evaluate_active_hybrid_measurements(
        self,
        values: np.ndarray,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
    ) -> None:
        self._evaluate_hybrid_measurements(values, ac_x, dc_x, hybrid_x, self._active_hybrid_measurement_plan)

    def _evaluate_hybrid_measurements(
        self,
        values: np.ndarray,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
        plan: "HybridStateEstimator._HybridMeasurementPlan",
    ) -> None:
        min_v = self.min_current_voltage

        def _gather_voltage(x: np.ndarray, bucket) -> np.ndarray:
            out = bucket.v_default.copy()
            if bucket.v_any_x:
                out[bucket.v_use_x] = x[bucket.v_x_index]
            return out

        # DCAC identity codes (1 P_DC, 2 P_AC, 3 Q_AC).
        for bucket in (plan.dcac_p_dc, plan.dcac_p_ac, plan.dcac_q_ac):
            if bucket.rows.size:
                values[bucket.rows] = hybrid_x[bucket.hx_p]

        # DCAC voltage-only codes (4 V_DC, 6 V_AC) and ACAC (5 V_FROM, 7 V_TO).
        for bucket, source_x in (
            (plan.dcac_v_dc, dc_x),
            (plan.dcac_v_ac, ac_x),
        ):
            if bucket.rows.size:
                values[bucket.rows] = _gather_voltage(source_x, bucket)

        # DCAC code 5 — I_DC = P_DC / V_DC.
        bucket = plan.dcac_i_dc
        if bucket.rows.size:
            p_dc_k = hybrid_x[bucket.hx_p]
            v_dc_k = _gather_voltage(dc_x, bucket)
            out = np.zeros(bucket.rows.size, dtype=np.float64)
            np.divide(p_dc_k, v_dc_k, out=out, where=np.abs(v_dc_k) > min_v)
            values[bucket.rows] = out

        # DCAC code 7 — I_AC = hypot(P_AC, Q_AC) / V_AC.
        bucket = plan.dcac_i_ac
        if bucket.rows.size:
            p_ac_k = hybrid_x[bucket.hx_p]
            q_ac_k = hybrid_x[bucket.hx_q]
            v_ac_k = _gather_voltage(ac_x, bucket)
            out = np.zeros(bucket.rows.size, dtype=np.float64)
            np.divide(np.hypot(p_ac_k, q_ac_k), v_ac_k, out=out, where=np.abs(v_ac_k) > min_v)
            values[bucket.rows] = out

        # ACAC identity codes (1 P_FROM, 2 Q_FROM, 3 P_TO, 4 Q_TO).
        for bucket in (plan.acac_p_from, plan.acac_q_from, plan.acac_p_to, plan.acac_q_to):
            if bucket.rows.size:
                values[bucket.rows] = hybrid_x[bucket.hx_p]

        # ACAC voltage-only codes (5 V_FROM, 7 V_TO).
        for bucket in (plan.acac_v_from, plan.acac_v_to):
            if bucket.rows.size:
                values[bucket.rows] = _gather_voltage(ac_x, bucket)

        # ACAC nonlinear current codes (6 I_FROM, 8 I_TO).
        for bucket in (plan.acac_i_from, plan.acac_i_to):
            if bucket.rows.size:
                p_k = hybrid_x[bucket.hx_p]
                q_k = hybrid_x[bucket.hx_q]
                v_k = _gather_voltage(ac_x, bucket)
                out = np.zeros(bucket.rows.size, dtype=np.float64)
                np.divide(np.hypot(p_k, q_k), v_k, out=out, where=np.abs(v_k) > min_v)
                values[bucket.rows] = out

    def evaluate(
        self,
        x: np.ndarray,
        measurements: Optional[Sequence[Measurement]] = None,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.evaluate(x, measurements)
        measurements = self._normalize_measurements(measurements)
        ac_x, dc_x, hybrid_x = self._split_state(x)
        n_meas = len(measurements)
        if out is None:
            values = np.zeros(n_meas, dtype=np.float64)
        else:
            values = out
            values.fill(0.0)
        blocks = self._measurement_blocks_for(measurements)
        ac_block = blocks["ac"]
        dc_block = blocks["dc"]
        if ac_block.measurements and self._ac_sub_estimator is not None:
            values[ac_block.rows] = self._ac_sub_estimator.evaluate(ac_x, ac_block.measurements)
        if dc_block.measurements and self._dc_sub_estimator is not None:
            values[dc_block.rows] = self._dc_sub_estimator.evaluate(dc_x, dc_block.measurements)
        self._evaluate_hybrid_measurements(
            values,
            ac_x,
            dc_x,
            hybrid_x,
            self._hybrid_measurement_plan_for(measurements, blocks),
        )
        return values

    def jacobian(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        return self.jacobian_sparse(x, measurements).toarray()

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None):
        delegate = self._delegate()
        if delegate is not None:
            return delegate.jacobian_sparse(x, measurements)
        measurements = self._normalize_measurements(measurements)
        ac_x, dc_x, hybrid_x = self._split_state(x)
        shape = (len(measurements), self.n_state)
        is_active = measurements is self.active_measurements
        H = self._jacobian_builder if is_active else SparseJacobianBuilder(shape)
        H.reset()
        blocks = self._measurement_blocks_for(measurements)
        self._append_sub_jacobian(H, self._ac_sub_estimator, ac_x, blocks["ac"], 0, "ac" if is_active else None)
        self._append_sub_jacobian(H, self._dc_sub_estimator, dc_x, blocks["dc"], self.dc_state_start, "dc" if is_active else None)
        self._append_hybrid_jacobian_plan(
            H,
            ac_x,
            dc_x,
            hybrid_x,
            self._hybrid_measurement_plan_for(measurements, blocks),
        )
        return H.to_csr()

    def _append_sub_jacobian(
        self,
        target,
        estimator,
        sub_x: np.ndarray,
        block: "HybridStateEstimator._MeasurementSideBlock",
        col_offset: int,
        cache_key: Optional[str] = None,
    ) -> None:
        """Stamp the sub-estimator's CSR jacobian into ``target``.

        When ``cache_key`` is provided (i.e. the parent is operating on its
        active measurements), the remapped row/column index arrays are cached
        after the first call. Subsequent calls skip the CSR→COO conversion,
        the row gather via ``block.rows[csr_rows]`` and the column offset
        arithmetic — only the freshly-computed ``sub_csr.data`` is stamped.

        Cache invalidation is driven by ``_refresh_active_measurement_state_layout``
        and friends, which reset ``self._sub_jacobian_stamp_cache`` whenever the
        active partition changes. We additionally guard against unexpected
        pattern shifts by checking nnz on every reuse.
        """
        if estimator is None or len(block.measurements) == 0:
            return
        sub_csr = estimator.jacobian_sparse(sub_x, block.measurements)
        if sub_csr.nnz == 0:
            return
        cache_store = self._sub_jacobian_stamp_cache if cache_key is not None else None
        cached = cache_store.get(cache_key) if cache_store is not None else None
        if cached is not None and cached["nnz"] == sub_csr.nnz:
            target.add_many(cached["parent_rows"], cached["parent_cols"], sub_csr.data)
            return

        # Cold path: derive row indices from indptr without going through tocoo().
        indptr = np.asarray(sub_csr.indptr)
        indices = np.asarray(sub_csr.indices)
        sub_rows = np.repeat(
            np.arange(sub_csr.shape[0], dtype=np.int64),
            np.diff(indptr).astype(np.int64, copy=False),
        )
        parent_rows = block.rows[sub_rows].astype(np.int32, copy=False)
        parent_cols = (indices.astype(np.int32, copy=False) + np.int32(col_offset))
        target.add_many(parent_rows, parent_cols, sub_csr.data)
        if cache_store is not None:
            cache_store[cache_key] = {
                "parent_rows": parent_rows,
                "parent_cols": parent_cols,
                "nnz": int(sub_csr.nnz),
            }

    @staticmethod
    def _append_vector_jacobian_entries(
        target,
        row_array: np.ndarray,
        col_array: np.ndarray,
        value_array,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        selected_cols = np.asarray(col_array, dtype=np.int32)[mask]
        valid = selected_cols >= 0
        if not np.any(valid):
            return
        selected_rows = row_array[mask][valid]
        selected_cols = selected_cols[valid]
        if np.isscalar(value_array):
            selected_values = np.full(selected_rows.size, float(value_array), dtype=np.float64)
        else:
            selected_values = np.asarray(value_array, dtype=np.float64)[mask][valid]
            valid_value = selected_values != 0.0
            if not np.any(valid_value):
                return
            selected_rows = selected_rows[valid_value]
            selected_cols = selected_cols[valid_value]
            selected_values = selected_values[valid_value]
        target.add_many(
            selected_rows.astype(np.int32, copy=False),
            selected_cols.astype(np.int32, copy=False),
            selected_values.astype(np.float64, copy=False),
        )

    @staticmethod
    def _masked_divide(numerator, denominator: np.ndarray, mask: np.ndarray) -> np.ndarray:
        denominator = np.asarray(denominator, dtype=np.float64)
        values = np.zeros_like(denominator, dtype=np.float64)
        np.divide(numerator, denominator, out=values, where=mask)
        return values

    def _append_active_hybrid_jacobian_entries(
        self,
        target,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
    ) -> None:
        self._append_hybrid_jacobian_plan(target, ac_x, dc_x, hybrid_x, self._active_hybrid_measurement_plan)

    def _append_hybrid_jacobian_plan(
        self,
        target,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
        plan: "HybridStateEstimator._HybridMeasurementPlan",
    ) -> None:
        min_v = self.min_current_voltage

        def _gather_voltage(x: np.ndarray, bucket) -> np.ndarray:
            out = bucket.v_default.copy()
            if bucket.v_any_x:
                out[bucket.v_use_x] = x[bucket.v_x_index]
            return out

        def _stamp_identity_power(bucket) -> None:
            """Identity derivative: 1.0 at jcol_p for every row."""
            if bucket.rows.size:
                target.add_many(
                    bucket.rows,
                    bucket.jcol_p,
                    np.ones(bucket.rows.size, dtype=np.float64),
                )

        def _stamp_identity_voltage(bucket) -> None:
            """Identity derivative: 1.0 at v_jcol where v_use_x (cols<0 filtered out)."""
            if bucket.rows.size and bucket.v_any_x:
                mask = bucket.v_use_x
                target.add_many(
                    bucket.rows[mask],
                    bucket.v_jcol[mask],
                    np.ones(int(mask.sum()), dtype=np.float64),
                )

        def _stamp_current_dc(bucket, x_for_v) -> None:
            """I_DC = p_dc / v_dc. Two jacobian terms."""
            if not bucket.rows.size:
                return
            p_dc_k = hybrid_x[bucket.hx_p]
            v_dc_k = _gather_voltage(x_for_v, bucket)
            valid = np.abs(v_dc_k) > min_v
            if not valid.any():
                return
            target.add_many(
                bucket.rows[valid],
                bucket.jcol_p[valid],
                1.0 / v_dc_k[valid],
            )
            v_mask = bucket.v_use_x & valid
            if v_mask.any():
                v_dc_sel = v_dc_k[v_mask]
                target.add_many(
                    bucket.rows[v_mask],
                    bucket.v_jcol[v_mask],
                    -p_dc_k[v_mask] / (v_dc_sel * v_dc_sel),
                )

        def _stamp_current_ac(bucket, x_for_v) -> None:
            """I = hypot(p, q) / v. Three jacobian terms."""
            if not bucket.rows.size:
                return
            p_k = hybrid_x[bucket.hx_p]
            q_k = hybrid_x[bucket.hx_q]
            v_k = _gather_voltage(x_for_v, bucket)
            s_k = np.hypot(p_k, q_k)
            valid = (np.abs(v_k) > min_v) & (s_k > 1e-12)
            if not valid.any():
                return
            s_v = s_k[valid] * v_k[valid]
            rows_v = bucket.rows[valid]
            target.add_many(rows_v, bucket.jcol_p[valid], p_k[valid] / s_v)
            target.add_many(rows_v, bucket.jcol_q[valid], q_k[valid] / s_v)
            v_mask = bucket.v_use_x & valid
            if v_mask.any():
                v_sel = v_k[v_mask]
                target.add_many(
                    bucket.rows[v_mask],
                    bucket.v_jcol[v_mask],
                    -s_k[v_mask] / (v_sel * v_sel),
                )

        # DCAC linear codes (1/2/3 — identity on power state; 4/6 — identity on voltage state).
        _stamp_identity_power(plan.dcac_p_dc)
        _stamp_identity_power(plan.dcac_p_ac)
        _stamp_identity_power(plan.dcac_q_ac)
        _stamp_identity_voltage(plan.dcac_v_dc)
        _stamp_identity_voltage(plan.dcac_v_ac)
        # DCAC nonlinear codes.
        _stamp_current_dc(plan.dcac_i_dc, dc_x)
        _stamp_current_ac(plan.dcac_i_ac, ac_x)

        # ACAC linear codes.
        _stamp_identity_power(plan.acac_p_from)
        _stamp_identity_power(plan.acac_q_from)
        _stamp_identity_power(plan.acac_p_to)
        _stamp_identity_power(plan.acac_q_to)
        _stamp_identity_voltage(plan.acac_v_from)
        _stamp_identity_voltage(plan.acac_v_to)
        # ACAC nonlinear current codes.
        _stamp_current_ac(plan.acac_i_from, ac_x)
        _stamp_current_ac(plan.acac_i_to, ac_x)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.observability_analysis(x, measurements, H, normal_matrix, normal_factor_diag)
        default_active_request = (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
        )
        if (
            default_active_request
            and self._initial_observability_cache is not None
        ):
            return self._initial_observability_cache
        measurements = self._normalize_measurements(measurements)
        x = self.initial_state() if x is None else x
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            result = ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])
            if default_active_request:
                self._initial_observability_cache = result
            return result
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
        if default_active_request:
            self._initial_observability_cache = result
        return result

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        delegate = self._delegate()
        if delegate is not None:
            array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
            previous_delegate_array_only = bool(getattr(delegate, "_array_only_estimate_result", False))
            delegate._array_only_estimate_result = array_only_result
            try:
                try:
                    result = delegate.estimate(
                        measurements,
                        x0,
                        verbose=verbose,
                        final_diagnostics=final_diagnostics,
                        observability=observability,
                    )
                except TypeError as exc:
                    if "final_diagnostics" not in str(exc):
                        raise
                    result = delegate.estimate(
                        measurements,
                        x0,
                        verbose=verbose,
                        observability=observability,
                    )
            finally:
                delegate._array_only_estimate_result = previous_delegate_array_only
            if array_only_result:
                result.measurements = []
            if solve_profile_start is not None:
                self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
            return result
        measurements = self._normalize_measurements(measurements)
        if len(measurements) < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {len(measurements)} < {self.n_state}")
        x = self.initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
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
        normal_solver = NormalEquationSolver(assume_fixed_pattern=measurements is self.active_measurements)
        normal_pattern = self._active_normal_pattern if measurements is self.active_measurements else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]
        if verbose:
            _print_iteration_header()
        converged = False
        objective = np.inf
        max_correction = np.inf
        residual_inf = np.inf
        H = None
        gain = None
        normal_factor_diag = None
        iteration = 0
        # Pre-allocate evaluation/residual buffers and reuse them across iterations
        # and line-search candidates. On acceptance we swap pointers so the accepted
        # vectors become the "main" buffers without copying.
        n_meas = len(measurements)
        z_est = np.empty(n_meas, dtype=np.float64)
        residual = np.empty(n_meas, dtype=np.float64)
        cand_z_est = np.empty(n_meas, dtype=np.float64)
        cand_residual = np.empty(n_meas, dtype=np.float64)

        for iteration in range(1, self.max_iter + 1):
            self.evaluate(x, measurements, out=z_est)
            self._measurement_residual(z, z_est, measurements, out=residual)
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
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
            dx, normal_factor_diag = normal_solver.solve(gain, rhs, return_factor_diag=final_diagnostics)
            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            if max_correction < self.tol:
                converged = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break
            step_scale = 1.0
            accepted = False
            nonfinite_candidates = 0
            for _attempt in range(12):
                candidate = x + step_scale * dx
                if self.voltage_cols.size:
                    candidate[self.voltage_cols] = np.maximum(candidate[self.voltage_cols], self.voltage_floor)
                self.evaluate(candidate, measurements, out=cand_z_est)
                self._measurement_residual(z, cand_z_est, measurements, out=cand_residual)
                candidate_objective = self._weighted_objective(weight, cand_residual)
                if np.isfinite(candidate_objective) and candidate_objective <= objective + 1e-12:
                    x = candidate
                    objective = candidate_objective
                    # Swap the accepted candidate buffers into the main role.
                    z_est, cand_z_est = cand_z_est, z_est
                    residual, cand_residual = cand_residual, residual
                    residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                    accepted = True
                    break
                if not np.isfinite(candidate_objective):
                    nonfinite_candidates += 1
                    if nonfinite_candidates >= 4:
                        break
                step_scale *= 0.5
            if not accepted:
                if nonfinite_candidates:
                    if verbose:
                        _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, False)
                    break
                x = x + dx
                if self.voltage_cols.size:
                    x[self.voltage_cols] = np.maximum(x[self.voltage_cols], self.voltage_floor)
            if verbose:
                _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, False)

        if H is None or gain is None:
            self.evaluate(x, measurements, out=z_est)
            self._measurement_residual(z, z_est, measurements, out=residual)
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if final_diagnostics:
                H = self.jacobian_sparse(x, measurements)
                if normal_pattern is None and is_sparse_matrix(H):
                    normal_pattern = _normal_equation_structural_pattern(H)
                    if measurements is self.active_measurements:
                        self._active_normal_pattern = normal_pattern
                if weighted_residual is not None:
                    np.multiply(weight, residual, out=weighted_residual)
                gain, _rhs = build_normal_equations(
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
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
        # Copy the buffer-backed vectors so callers (and the next estimate() invocation)
        # can't accidentally see them mutate as line-search scratch space later on.
        return EstimateResult(
            converged=converged,
            iterations=iteration,
            objective=objective,
            max_correction=max_correction,
            residual_inf=residual_inf,
            x=x,
            z_est=z_est.copy(),
            residual=residual.copy(),
            H=H,
            gain=gain,
            measurements=[] if array_only_result else (
                measurements if isinstance(measurements, MeasurementList) else list(measurements)
            ),
            observability=observability,
            measurement_table=getattr(measurements, "table", None),
        )

    def identify_bad_data(
        self,
        result: EstimateResult,
        threshold: Optional[float] = None,
    ) -> Tuple[List[BadDataItem], np.ndarray]:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.identify_bad_data(result, threshold)
        threshold = self.params.bad_threshold if threshold is None else threshold
        measurement_table = getattr(result, "measurement_table", None)
        table_weight = getattr(measurement_table, "weight", None)
        if table_weight is not None and np.asarray(table_weight).size == result.residual.size:
            weights = np.asarray(table_weight, dtype=np.float64)
        else:
            _z, weights = self._measurement_vectors(result.measurements)
        jacobian_measurements = (
            result.measurements
            if len(result.measurements) == result.residual.size
            else getattr(self, "active_measurements", result.measurements)
        )
        H = result.H if result.H is not None else self.jacobian_sparse(result.x, jacobian_measurements)
        gain = result.gain
        if gain is None:
            gain, _rhs = build_normal_equations(H, result.residual, weights)
        gain_inv = inverse_gain_for_bad_data(gain)
        if gain_inv is None:
            omega_diag = 1.0 / np.maximum(weights, 1e-12)
        else:
            leverage = measurement_leverage(H, gain_inv)
            omega_diag = np.maximum(1.0 / np.maximum(weights, 1e-12) - leverage, 1e-12)
        normalized = np.abs(result.residual) / np.sqrt(omega_diag)
        bad_items: List[BadDataItem] = []
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
        self._require_prepared("run()")
        mode = normalize_seresult_result_mode(result_mode)
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        array_only = mode == "array"
        if array_only and remove_bad_data:
            raise ValueError("result_mode='array' cannot be combined with remove_bad_data=True")
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
                    threshold=threshold,
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
        delegate = self._delegate()
        if delegate is not None:
            return delegate.estimate_with_bad_data_removal(
                threshold=threshold,
                max_remove=max_remove,
                verbose=verbose,
            )
        threshold = self.params.bad_threshold if threshold is None else threshold
        max_remove = self.params.bad_max_remove if max_remove is None else max_remove
        measurements = self.active_measurements
        removed: List[BadDataItem] = []
        x0 = self.initial_state()
        result = self.estimate(measurements=measurements, x0=x0, verbose=verbose)
        while len(removed) < max_remove:
            bad_items, _normalized = self.identify_bad_data(result, threshold)
            if not bad_items:
                break
            worst = bad_items[0]
            removed.append(worst)
            remove_pos = result.measurements.index(worst.measurement)
            keep_rows = np.concatenate(
                (
                    np.arange(remove_pos, dtype=np.int64),
                    np.arange(remove_pos + 1, len(result.measurements), dtype=np.int64),
                )
            )
            if measurements is self.active_measurements and result.measurements is self.active_measurements:
                measurements = self._shrink_active_measurement_state_layout(remove_pos)
            else:
                measurements = take_measurement_view(result.measurements, keep_rows)
            result = self.estimate(measurements=measurements, x0=result.x, verbose=verbose)
        return result, removed

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        delegate = self._delegate()
        if delegate is not None:
            delegate.print_state(x, limit=limit)
            return
        ac_x, dc_x, hybrid_x = self._split_state(x)
        if self._ac_sub_estimator is not None:
            print("Estimated AC subsystem state:")
            self._ac_sub_estimator.print_state(ac_x, limit=limit)
        if self._dc_sub_estimator is not None:
            print("Estimated DC subsystem state:")
            self._dc_sub_estimator.print_state(dc_x, limit=limit)
        if self.dcac_converters:
            print("Estimated DCAC converter states:")
            for conv in self.dcac_converters[:limit]:
                k = self.dcac_pos_by_name[conv.name]
                base = 3 * k
                print(
                    f"  {conv.name:14s} "
                    f"P_DC={hybrid_x[base]:.9f} P_AC={hybrid_x[base + 1]:.9f} Q_AC={hybrid_x[base + 2]:.9f}"
                )
        if self.acac_converters:
            print("Estimated ACAC converter states:")
            base0 = 3 * len(self.dcac_converters)
            for conv in self.acac_converters[:limit]:
                k = self.acac_pos_by_name[conv.name]
                base = base0 + 4 * k
                print(
                    f"  {conv.name:14s} "
                    f"P_FROM={hybrid_x[base]:.9f} Q_FROM={hybrid_x[base + 1]:.9f} "
                    f"P_TO={hybrid_x[base + 2]:.9f} Q_TO={hybrid_x[base + 3]:.9f}"
                )


def _print_observability(result: ObservabilityResult) -> None:
    print(
        "Observability: "
        f"observable={result.observable}, rank={result.rank}/{result.state_count}, "
        f"measurements={result.measurement_count}, deficiency={result.deficiency}"
    )
    if result.weak_states:
        print("Weak states:")
        for label, score in result.weak_states[:10]:
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


def _profile_time_items(estimator: HybridStateEstimator) -> List[Tuple[str, float]]:
    items = list(getattr(estimator, "profile_times", {}).items())
    delegate = estimator._delegate()
    if delegate is not None:
        items.extend(
            (f"delegate.{name}", value)
            for name, value in getattr(delegate, "profile_times", {}).items()
        )
    else:
        for side, sub_estimator in (
            ("ac", getattr(estimator, "_ac_sub_estimator", None)),
            ("dc", getattr(estimator, "_dc_sub_estimator", None)),
        ):
            items.extend(
                (f"{side}.{name}", value)
                for name, value in getattr(sub_estimator, "profile_times", {}).items()
            )
    return sorted(items)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid AC/DC weighted least-squares state estimation.")
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="Hybrid network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat hybrid state.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    parser.add_argument("--profile", action="store_true", help="Print initialization profile timings.")
    parser.add_argument("--result-mode", default="full", help="SEResult payload mode: full, summary, array, or none.")
    parser.add_argument("--se-result", default=None, help="Write SEResult blocks to a new E file.")
    args = parser.parse_args(argv)

    estimator = HybridStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        flat_start=args.flat_start,
        parameter_file=Path(args.para),
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
        f"converged={result.converged}, iterations={result.iterations}, "
        f"objective={result.objective:.6e}, max_dx={result.max_correction:.3e}, "
        f"residual_inf={result.residual_inf:.3e}"
    )
    _print_bad_data(bad_items, normalized, bad_threshold)
    if args.se_result and se_result is not None:
        se_result.write_e_file(Path(args.se_result))
    if args.print_state:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in _profile_time_items(estimator):
            print(f"  {name}={value:.6f}s")
    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
