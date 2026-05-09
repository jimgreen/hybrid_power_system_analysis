import argparse
import contextlib
import io
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
from ac_lf import matpower_branch_stamp, matpower_branch_stamp_vectorized
from hybrid_lf import HybridPowerNetwork
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    EstimateResult,
    Measurement,
    MeasurementList,
    ObservabilityResult,
    measurement_table_from_measurements,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.ac_se import ACStateEstimator, _read_measurements_direct as _read_table_measurements_direct
from secore.dc_se import DCStateEstimator
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
    measurement_residual as build_measurement_residual,
    observability_rank_details,
    observability_weak_direction,
    targeted_redundancy_count,
)
from secore.se_result import SEResult
from unit_system import ac_current_base_ka, dc_current_base_ka


DEFAULT_CASE = model_file("hybrid", "qinling.e")
DEFAULT_MEAS = measurement_file("hybrid", "qinling.meas")


def _read_measurements_direct(meas_file: Path):
    """Read Measurement rows through the shared table-backed SE parser."""
    return _read_table_measurements_direct(meas_file, Measurement)


def _measurement_table_from_measurements(measurements: Sequence[Measurement]):
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=DEVICE_TYPE_CODES,
        angle_measurement_types=ANGLE_MEASUREMENT_TYPES,
    )


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

    @dataclass(frozen=True)
    class _MeasurementSideBlock:
        rows: np.ndarray
        measurements: Sequence[Measurement]

    @dataclass(frozen=True)
    class _SubEstimatorMeasurementContext:
        measurements: Sequence[Measurement]
        summary_attrs: Optional[Dict[str, object]]

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

    @dataclass(frozen=True)
    class _MeasurementActivitySummary:
        max_idx: int
        measured_devices: set
        active_keys: set

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
        self.prepare()

    def prepare(self) -> "HybridStateEstimator":
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
        self.measurements = self._load_measurements(self.meas_file)
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
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
        self._record_profile_time("init.finalize_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._add_pseudo_power_measurements()
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

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    @staticmethod
    def _load_measurements(meas_file: Path) -> List[Measurement]:
        return _read_measurements_direct(meas_file)

    @staticmethod
    def _load_network(e_file: Path) -> HybridPowerNetwork:
        network = HybridPowerNetwork.read_from_file(e_file)
        with contextlib.redirect_stdout(io.StringIO()):
            ac_warnings, ac_errors, dc_warnings, dc_errors = network.prepare(verbose=False)
        if ac_errors or dc_errors:
            raise RuntimeError(
                f"Topology check failed for {e_file}: "
                f"ac_errors={ac_errors}, dc_errors={dc_errors}, "
                f"ac_warnings={ac_warnings}, dc_warnings={dc_warnings}"
            )
        return network

    def _initial_measurement_sources_by_side(self) -> Dict[str, List[Measurement]]:
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        if sources_by_side is None:
            partitions = partition_measurements_by_code(
                self.measurements,
                self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
                side_by_device_type=self._MEASUREMENT_SIDE_BY_DEVICE_TYPE,
                table_builder=_measurement_table_from_measurements,
                sides=("ac", "dc", "hybrid"),
            )
            sources_by_side = partitions.measurements
            self._sub_measurement_sources_by_side = sources_by_side
        return sources_by_side

    def _measurements_for_sub_estimator(self, side: str, share_measurements: bool) -> List[Measurement]:
        if share_measurements and self._is_uncoupled_single_side(side):
            return self.measurements
        sources = self._initial_measurement_sources_by_side().get(side, ())
        if share_measurements:
            return copy_measurement_view(sources)
        return copy_measurement_view(sources)

    def _defer_sub_prepare_finalize(self) -> bool:
        if self._is_uncoupled_single_side("ac") or self._is_uncoupled_single_side("dc"):
            return False
        has_ac = bool(getattr(getattr(self.network, "ac", None), "nodes", None))
        has_dc = bool(getattr(getattr(self.network, "dc", None), "nodes", None))
        return bool(has_ac and has_dc)

    def _is_uncoupled_single_side(self, side: str) -> bool:
        if getattr(self.network, "dcac_converters", None) or getattr(self.network, "acac_converters", None):
            return False
        has_ac = bool(getattr(getattr(self.network, "ac", None), "nodes", None))
        has_dc = bool(getattr(getattr(self.network, "dc", None), "nodes", None))
        if side == "ac":
            return has_ac and not has_dc
        if side == "dc":
            return has_dc and not has_ac
        return False

    def _defer_sub_active_measurement_preparation(self) -> bool:
        has_ac = bool(getattr(getattr(self.network, "ac", None), "nodes", None))
        has_dc = bool(getattr(getattr(self.network, "dc", None), "nodes", None))
        has_coupling = bool(
            getattr(self.network, "dcac_converters", None)
            or getattr(self.network, "acac_converters", None)
        )
        return bool(self.flat_start and (has_coupling or (has_ac and has_dc)))

    def _build_ac_sub_estimator(self) -> Optional[ACStateEstimator]:
        if not getattr(self.network.ac, "nodes", None):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("ac"))
        try:
            return ACStateEstimator(
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
            )
        except RuntimeError as exc:
            if "No alive AC nodes" in str(exc):
                return None
            raise

    def _build_dc_sub_estimator(self) -> Optional[DCStateEstimator]:
        if not getattr(self.network.dc, "nodes", None):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("dc"))
        try:
            return DCStateEstimator(
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
            )
        except RuntimeError as exc:
            if "No alive DC nodes" in str(exc):
                return None
            raise

    def _build_device_maps(self) -> None:
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
        self.active_angle_residual_mask = angle_residual_mask(self.active_measurements)
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
            self.ac_meas = list(self.active_measurements)
            self.dc_meas = []
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = rows.copy()
            self._active_dc_hybrid_rows = empty_rows
        elif side == "dc":
            self.ac_meas_rows = empty_rows
            self.dc_meas_rows = rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = []
            self.dc_meas = list(self.active_measurements)
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = empty_rows
            self._active_dc_hybrid_rows = rows.copy()
        else:
            self._partition_active_measurements()
            return
        self._active_ac_sub_measurements = list(self.ac_meas)
        self._active_dc_sub_measurements = list(self.dc_meas)
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
            self.node_pos = estimator.node_pos
            self.N = len(estimator.nodes)

        def _extract_state_vars(self, x, update_cache=False):
            theta, voltage = self.estimator._unpack_state(np.asarray(x, dtype=np.float64))
            return theta, voltage, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    class _DCCalcAdapter:
        def __init__(self, estimator: DCStateEstimator):
            self.estimator = estimator
            self.node_pos = estimator.node_pos
            self.alive_node_dict = estimator.node_pos
            self.N = len(estimator.nodes)

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
        for meas in self.measurements:
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                meas.valid = False
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
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            if meas.device_type in unsupported_switch_rows:
                meas.valid = False
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                meas.valid = False

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
    def _max_measurement_idx_fast(measurements: Sequence[Measurement]) -> int:
        table = getattr(measurements, "table", None)
        count = len(measurements)
        if table is not None:
            table_size = int(table.idx.size)
            max_idx = int(table.idx.max()) if table_size else 0
            if table_size == count:
                return max_idx
            if table_size < count:
                for meas in measurements[table_size:]:
                    if meas.idx > max_idx:
                        max_idx = int(meas.idx)
                return max_idx
        max_idx = 0
        for meas in measurements:
            if meas.idx > max_idx:
                max_idx = int(meas.idx)
        return max_idx

    def _sub_measurement_summary_attrs(
        self,
        measurements: Sequence[Measurement],
        max_idx: Optional[int] = None,
    ) -> Dict[str, object]:
        active_device_keys = set()
        active_measurement_keys = set()
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            active = table.valid & (table.weight > 0.0)
            for device_type, device_name, meas_type in zip(
                table.device_type[active],
                table.device_name[active],
                table.meas_type[active],
            ):
                active_device_keys.add((device_type, device_name))
                active_measurement_keys.add((device_type, device_name, meas_type))
        else:
            for meas in measurements:
                if meas.valid and meas.weight > 0.0:
                    active_device_keys.add((meas.device_type, meas.device_name))
                    active_measurement_keys.add((meas.device_type, meas.device_name, meas.meas_type))
        return {
            "_active_device_key_cache": active_device_keys,
            "_active_measurement_key_cache": active_measurement_keys,
            "_max_measurement_idx": int(max_idx) if max_idx is not None else self._max_measurement_idx_fast(self.measurements),
        }

    def _sub_measurement_context(
        self,
        measurements: Sequence[Measurement],
        *,
        refresh_summary: bool = True,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
    ) -> "HybridStateEstimator._SubEstimatorMeasurementContext":
        summary_attrs = None
        if summary_measurements is not None:
            summary_attrs = self._sub_measurement_summary_attrs(summary_measurements, summary_max_idx)
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
    ) -> None:
        context = self._sub_measurement_context(
            measurements,
            refresh_summary=refresh_summary,
            summary_measurements=summary_measurements,
            summary_max_idx=summary_max_idx,
        )
        self._invoke_sub_estimator_methods(
            estimator,
            context,
            tuple(method_names),
            preserve_max_idx=preserve_max_idx,
        )

    def _convert_measurements_to_pu(self) -> None:
        """Delegate AC/DC normalization and normalize only converter rows locally."""
        self._invalidate_measurement_activity_summary()
        if getattr(self, "_measurements_normalized", False):
            return
        sources = self._initial_measurement_sources_by_side()
        converted_by_sub = getattr(self, "_sub_measurements_converted_by_side", {})
        if not converted_by_sub.get("ac", False):
            self._call_sub_with_measurements(
                self._ac_sub_estimator,
                sources["ac"],
                "_convert_measurements_to_pu",
                refresh_summary=False,
            )
            converted_by_sub["ac"] = True
            if hasattr(sources["ac"], "normalized"):
                sources["ac"].normalized = True
            if self._ac_sub_estimator is not None and hasattr(self._ac_sub_estimator.measurements, "normalized"):
                self._ac_sub_estimator.measurements.normalized = True
        if not converted_by_sub.get("dc", False):
            self._call_sub_with_measurements(
                self._dc_sub_estimator,
                sources["dc"],
                "_convert_measurements_to_pu",
                refresh_summary=False,
            )
            converted_by_sub["dc"] = True
            if hasattr(sources["dc"], "normalized"):
                sources["dc"].normalized = True
            if self._dc_sub_estimator is not None and hasattr(self._dc_sub_estimator.measurements, "normalized"):
                self._dc_sub_estimator.measurements.normalized = True
        self._convert_hybrid_measurements_to_pu(sources["hybrid"])
        self._sync_measurement_table_from_objects()
        if hasattr(self.measurements, "normalized"):
            self.measurements.normalized = True
        self._measurements_normalized = True

    def _finalize_sub_estimators_after_measurement_prepare(self) -> None:
        if self._ac_sub_estimator is not None and getattr(
            self._ac_sub_estimator,
            "_defer_prepare_finalize_pending",
            False,
        ):
            self._ac_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )
        if self._dc_sub_estimator is not None and getattr(
            self._dc_sub_estimator,
            "_defer_prepare_finalize_pending",
            False,
        ):
            self._dc_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )

    def _convert_hybrid_measurements_to_pu(self, measurements: Sequence[Measurement]) -> None:
        for meas in measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            scale = self._hybrid_measurement_scale(meas)
            meas.value = float(meas.value) / scale

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
        return float(self.network.ac.node_dict[int(node_idx)].vbase)

    def _dc_voltage_base(self, node_idx: int) -> float:
        return float(self.network.dc.node_dict[int(node_idx)].vbase)

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
        self.measurements.append(measurement)
        self._invalidate_measurement_activity_summary()
        return next_idx + 1

    def _add_pseudo_power_measurements(self) -> None:
        """Delegate AC/DC pseudo rows, then add missing converter-device pseudo rows."""
        sources = self._initial_measurement_sources_by_side()
        if self._ac_sub_estimator is not None:
            add_balance = getattr(self._ac_sub_estimator, "_add_power_balance_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_balance):
                methods.append("_add_power_balance_constraint_measurements")
            self._call_sub_sequence_with_measurements(
                self._ac_sub_estimator,
                self.measurements,
                methods,
                preserve_max_idx=True,
                summary_measurements=sources["ac"],
                summary_max_idx=self._max_measurement_idx_fast(self.measurements),
            )
            if callable(add_balance):
                self._disable_coupled_ac_power_balance_rows()
        if self._dc_sub_estimator is not None:
            add_constraints = getattr(self._dc_sub_estimator, "_add_zero_branch_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_constraints):
                methods.append("_add_zero_branch_constraint_measurements")
            self._call_sub_sequence_with_measurements(
                self._dc_sub_estimator,
                self.measurements,
                methods,
                summary_measurements=sources["dc"],
                summary_max_idx=self._max_measurement_idx_fast(self.measurements),
            )
        self._populate_side_device_pseudo_values()
        self._add_hybrid_pseudo_measurements()

    def _disable_coupled_ac_power_balance_rows(self) -> None:
        coupled = self._converter_coupled_ac_node_names()
        if not coupled:
            return
        for meas in self.measurements:
            if meas.device_type == "ACPowerBalance" and meas.device_name in coupled:
                meas.valid = False

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
        summary = self._measurement_activity_summary()
        next_idx = int(summary.max_idx) + 1
        measured_devices = summary.measured_devices
        for spec in self._hybrid_converter_measurement_specs(source="pseudo"):
            if (spec.device_type, spec.device_name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_{spec.meas_type.lower()}_{spec.device_name}",
                spec.device_type,
                spec.device_name,
                spec.meas_type,
                spec.value,
            )

    @staticmethod
    def _ac_sub_state_label_to_hybrid(label: str) -> str:
        prefix, name = label.split(":", 1)
        prefix_map = {
            "theta": "AC_THETA",
            "V": "AC_V",
            "I_ZERO_RE": "AC_I_RE",
            "I_ZERO_IM": "AC_I_IM",
            "I_BREAK_RE": "AC_I_RE",
            "I_BREAK_IM": "AC_I_IM",
            "P_GEN": "AC_GEN_P",
            "Q_GEN": "AC_GEN_Q",
            "P_LOAD": "AC_LOAD_P",
            "Q_LOAD": "AC_LOAD_Q",
        }
        return f"{prefix_map.get(prefix, 'AC_' + prefix)}:{name}"

    @staticmethod
    def _dc_sub_state_label_to_hybrid(label: str) -> str:
        prefix, name = label.split(":", 1)
        prefix_map = {
            "V": "DC_V",
            "I_ZERO": "DC_I",
            "I_BREAK": "DC_I",
            "P_DCDC_FROM": "DCDC_P_FROM",
            "P_DCDC_TO": "DCDC_P_TO",
            "P_VGEN": "DC_VGEN_P",
        }
        return f"{prefix_map.get(prefix, 'DC_' + prefix)}:{name}"

    def _build_state_layout(self) -> None:
        ac_n = int(getattr(self._ac_sub_estimator, "n_state", 0) or 0)
        dc_n = int(getattr(self._dc_sub_estimator, "n_state", 0) or 0)
        labels: List[str] = []
        sides: List[str] = []
        if self._ac_sub_estimator is not None:
            labels.extend(self._ac_sub_state_label_to_hybrid(label) for label in self._ac_sub_estimator.state_labels)
            sides.extend(["ac"] * ac_n)
        if self._dc_sub_estimator is not None:
            labels.extend(self._dc_sub_state_label_to_hybrid(label) for label in self._dc_sub_estimator.state_labels)
            sides.extend(["dc"] * dc_n)
        self.dcac_state_start = len(labels)
        for conv in self.dcac_converters:
            labels.extend((f"DCAC_P_DC:{conv.name}", f"DCAC_P_AC:{conv.name}", f"DCAC_Q_AC:{conv.name}"))
            sides.extend(("hybrid", "hybrid", "hybrid"))
        self.acac_state_start = len(labels)
        for conv in self.acac_converters:
            labels.extend(
                (
                    f"ACAC_P_FROM:{conv.name}",
                    f"ACAC_Q_FROM:{conv.name}",
                    f"ACAC_P_TO:{conv.name}",
                    f"ACAC_Q_TO:{conv.name}",
                )
            )
            sides.extend(("hybrid", "hybrid", "hybrid", "hybrid"))
        self.ac_n_state = ac_n
        self.dc_n_state = dc_n
        self.hybrid_n_state = 3 * len(self.dcac_converters) + 4 * len(self.acac_converters)
        self.dc_state_start = ac_n
        self.hybrid_state_start = ac_n + dc_n
        self.n_state = len(labels)
        self.state_labels = labels
        self.state_sides = sides
        self.ac_state_labels = list(self._ac_sub_estimator.state_labels) if self._ac_sub_estimator is not None else []
        self.dc_state_labels = list(self._dc_sub_estimator.state_labels) if self._dc_sub_estimator is not None else []
        self.ac_state_layout = (
            self._ac_sub_estimator.state_layout()
            if self._ac_sub_estimator is not None
            else {"state_labels": [], "n_state": 0}
        )
        self.dc_state_layout = (
            {
                "state_labels": self._dc_sub_estimator.state_labels,
                "voltage_col": self._dc_sub_estimator.voltage_col,
                "n_state": self._dc_sub_estimator.n_state,
                "references": getattr(self._dc_sub_estimator, "references", []),
            }
            if self._dc_sub_estimator is not None
            else {"state_labels": [], "n_state": 0}
        )
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
        self.voltage_cols = np.asarray(
            [idx for idx, label in enumerate(labels) if label.startswith(("AC_V:", "DC_V:"))],
            dtype=np.int32,
        )
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
        self.dc_voltage_state_col = (
            np.asarray(getattr(self._dc_sub_estimator, "voltage_col", []), dtype=np.int32) + self.dc_state_start
            if self._dc_sub_estimator is not None
            else np.array([], dtype=np.int32)
        )
        self._partition_state_variables()

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
            sides=("ac", "dc", "hybrid"),
        )

        self.active_measurements = active_view.measurements
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = copy_measurement_view(self.ac_meas)
        self._active_dc_sub_measurements = copy_measurement_view(self.dc_meas)
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
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self.power_flow_state = self.initial_state(flat=False)
        self.flat_state = self.initial_state(flat=True)

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
        self._active_ac_sub_measurements = copy_measurement_view(self.ac_meas)
        self._active_dc_sub_measurements = copy_measurement_view(self.dc_meas)
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
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
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
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        self.active_measurements = take_measurement_view(self.active_measurements, keep_rows)
        if not hasattr(self, "ac_meas_rows"):
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
        self._active_ac_sub_measurements = copy_measurement_view(self.ac_meas)
        self._active_dc_sub_measurements = copy_measurement_view(self.dc_meas)
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
        self._active_normal_pattern = None
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._invalidate_measurement_activity_summary()
        return self.active_measurements

    def _build_hybrid_measurement_plan(
        self,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        ac_node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
        dc_node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
        dcac_code = int(DEVICE_TYPE_CODES["DCACConverter"])
        acac_code = int(DEVICE_TYPE_CODES["ACACConverter"])
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
            table_builder=_measurement_table_from_measurements,
        )

        dcac_mask = plan_table.handled & (plan_table.device_type_code == dcac_code)
        dcac_local_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_pos = plan_table.device_pos[dcac_mask].astype(np.int32, copy=False)
        dcac_rows = block.rows[dcac_local_rows].astype(np.int32, copy=False)
        dcac_codes = plan_table.meas_kind[dcac_mask].astype(np.int8, copy=False)
        dcac_dc_v_col = np.fromiter(
            (self._dc_voltage_col_for_node(self.dcac_converters[int(pos)].dc_node) for pos in dcac_pos),
            dtype=np.int32,
            count=dcac_pos.size,
        )
        dcac_ac_v_col = np.fromiter(
            (self._ac_voltage_col_for_node(self.dcac_converters[int(pos)].ac_node) for pos in dcac_pos),
            dtype=np.int32,
            count=dcac_pos.size,
        )
        dcac_dc_v_default = np.fromiter(
            (
                float(
                    getattr(
                        dc_node_dict.get(int(self.dcac_converters[int(pos)].dc_node)),
                        "voltage",
                        1.0,
                    )
                    or 1.0
                )
                for pos in dcac_pos
            ),
            dtype=np.float64,
            count=dcac_pos.size,
        )
        dcac_ac_v_default = np.fromiter(
            (
                float(
                    getattr(
                        ac_node_dict.get(int(self.dcac_converters[int(pos)].ac_node)),
                        "voltage",
                        1.0,
                    )
                    or 1.0
                )
                for pos in dcac_pos
            ),
            dtype=np.float64,
            count=dcac_pos.size,
        )

        acac_mask = plan_table.handled & (plan_table.device_type_code == acac_code)
        acac_local_rows = plan_table.row[acac_mask].astype(np.int64, copy=False)
        acac_pos = plan_table.device_pos[acac_mask].astype(np.int32, copy=False)
        acac_rows = block.rows[acac_local_rows].astype(np.int32, copy=False)
        acac_codes = plan_table.meas_kind[acac_mask].astype(np.int8, copy=False)
        acac_from_v_col = np.fromiter(
            (self._ac_voltage_col_for_node(self.acac_converters[int(pos)].i_node) for pos in acac_pos),
            dtype=np.int32,
            count=acac_pos.size,
        )
        acac_to_v_col = np.fromiter(
            (self._ac_voltage_col_for_node(self.acac_converters[int(pos)].j_node) for pos in acac_pos),
            dtype=np.int32,
            count=acac_pos.size,
        )
        acac_from_v_default = np.fromiter(
            (
                float(
                    getattr(
                        ac_node_dict.get(int(self.acac_converters[int(pos)].i_node)),
                        "voltage",
                        1.0,
                    )
                    or 1.0
                )
                for pos in acac_pos
            ),
            dtype=np.float64,
            count=acac_pos.size,
        )
        acac_to_v_default = np.fromiter(
            (
                float(
                    getattr(
                        ac_node_dict.get(int(self.acac_converters[int(pos)].j_node)),
                        "voltage",
                        1.0,
                    )
                    or 1.0
                )
                for pos in acac_pos
            ),
            dtype=np.float64,
            count=acac_pos.size,
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
            3 * len(self.dcac_converters)
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
        for conv in sorted(self.dcac_by_name.values(), key=lambda item: item.idx):
            p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "i_p", 0.0) or 0.0)
            p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_p", 0.0) or 0.0)
            q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_q", 0.0) or 0.0)
            specs.extend(
                (
                    self._HybridConverterMeasurementSpec("DCACConverter", conv.name, "P_DC", p_dc),
                    self._HybridConverterMeasurementSpec("DCACConverter", conv.name, "P_AC", p_ac),
                    self._HybridConverterMeasurementSpec("DCACConverter", conv.name, "Q_AC", q_ac),
                    self._HybridConverterMeasurementSpec(
                        "DCACConverter",
                        conv.name,
                        "V_DC",
                        float(getattr(getattr(conv, "dc_node_obj", None), "voltage", 1.0) or 1.0),
                    ),
                    self._HybridConverterMeasurementSpec(
                        "DCACConverter",
                        conv.name,
                        "V_AC",
                        float(getattr(getattr(conv, "ac_node_obj", None), "voltage", 1.0) or 1.0),
                    ),
                )
            )
        for conv in sorted(self.acac_by_name.values(), key=lambda item: item.idx):
            specs.extend(
                (
                    self._HybridConverterMeasurementSpec("ACACConverter", conv.name, "P_FROM", float(getattr(conv, "i_p", 0.0) or 0.0)),
                    self._HybridConverterMeasurementSpec("ACACConverter", conv.name, "Q_FROM", float(getattr(conv, "i_q", 0.0) or 0.0)),
                    self._HybridConverterMeasurementSpec("ACACConverter", conv.name, "P_TO", float(getattr(conv, "j_p", 0.0) or 0.0)),
                    self._HybridConverterMeasurementSpec("ACACConverter", conv.name, "Q_TO", float(getattr(conv, "j_q", 0.0) or 0.0)),
                    self._HybridConverterMeasurementSpec(
                        "ACACConverter",
                        conv.name,
                        "V_FROM",
                        float(getattr(getattr(conv, "i_node_obj", None), "voltage", 1.0) or 1.0),
                    ),
                    self._HybridConverterMeasurementSpec(
                        "ACACConverter",
                        conv.name,
                        "V_TO",
                        float(getattr(getattr(conv, "j_node_obj", None), "voltage", 1.0) or 1.0),
                    ),
                )
            )
        return specs

    def state_layout(self) -> Dict[str, object]:
        return {
            "state_labels": self.state_labels,
            "state_sides": self.state_sides,
            "ac_state_slice": self.ac_state_slice,
            "dc_state_slice": self.dc_state_slice,
            "hybrid_state_slice": self.hybrid_state_slice,
            "n_state": self.n_state,
        }

    def _hybrid_seed_vector(self, flat: Optional[bool] = None) -> np.ndarray:
        flat = self.flat_start if flat is None else bool(flat)
        x = np.zeros(self.hybrid_n_state, dtype=np.float64)
        offset = 0
        for conv in self.dcac_converters:
            defaults = (
                0.0 if flat else float(getattr(conv, "dc_p", 0.0) or 0.0),
                float(getattr(conv, "p_ac_set", 0.0) or 0.0) if flat else float(getattr(conv, "ac_p", 0.0) or 0.0),
                float(getattr(conv, "q_ac_set", 0.0) or 0.0) if flat else float(getattr(conv, "ac_q", 0.0) or 0.0),
            )
            x[offset : offset + 3] = defaults
            offset += 3
        for conv in self.acac_converters:
            defaults = (
                float(getattr(conv, "p_set", 0.0) or 0.0) if flat else float(getattr(conv, "i_p", 0.0) or 0.0),
                float(getattr(conv, "i_q_set", 0.0) or 0.0) if flat else float(getattr(conv, "i_q", 0.0) or 0.0),
                -float(getattr(conv, "p_set", 0.0) or 0.0) if flat else float(getattr(conv, "j_p", 0.0) or 0.0),
                float(getattr(conv, "j_q_set", 0.0) or 0.0) if flat else float(getattr(conv, "j_q", 0.0) or 0.0),
            )
            x[offset : offset + 4] = defaults
            offset += 4
        self._seed_hybrid_power_states_from_measurements(x)
        return x

    def _seed_hybrid_power_states_from_measurements(self, x: np.ndarray) -> None:
        seed_plan = self._build_hybrid_seed_measurement_plan(self.measurements)
        if seed_plan.measurement_row.size == 0:
            return
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            table = _measurement_table_from_measurements(self.measurements)
            try:
                self.measurements.table = table
            except AttributeError:
                pass
        x[seed_plan.state_col] = np.asarray(table.value, dtype=np.float64)[seed_plan.measurement_row]

    def initial_state(self, flat: Optional[bool] = None) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.initial_state()
        flat = self.flat_start if flat is None else bool(flat)
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
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

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

    def _active_measurement_keys(self) -> set:
        return set(self._measurement_activity_summary().active_keys)

    def _invalidate_measurement_activity_summary(self) -> None:
        if hasattr(self, "_measurement_activity_summary_cache"):
            delattr(self, "_measurement_activity_summary_cache")
        if hasattr(self, "_observability_pseudo_candidate_cache"):
            delattr(self, "_observability_pseudo_candidate_cache")

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
        idx = np.asarray(table.idx, dtype=np.int64)
        valid = np.asarray(table.valid, dtype=bool)
        weight = np.asarray(table.weight, dtype=np.float64)
        active_mask = valid & (weight > 0.0)
        max_idx = int(idx.max()) if idx.size else 0
        measured_devices = set(
            zip(
                np.asarray(table.device_type, dtype=object)[active_mask].tolist(),
                np.asarray(table.device_name, dtype=object)[active_mask].tolist(),
            )
        )
        active_keys = set(
            zip(
                np.asarray(table.device_type, dtype=object)[active_mask].tolist(),
                np.asarray(table.device_name, dtype=object)[active_mask].tolist(),
                np.asarray(table.meas_type, dtype=object)[active_mask].tolist(),
            )
        )
        cache = self._MeasurementActivitySummary(max_idx=max_idx, measured_devices=measured_devices, active_keys=active_keys)
        self._measurement_activity_summary_cache = cache
        return cache

    def _next_measurement_idx(self) -> int:
        return int(self._measurement_activity_summary().max_idx) + 1

    @staticmethod
    def _converter_meas_type_for_state_prefix(prefix: str) -> Optional[str]:
        mapping = {
            "DCAC_P_DC": "P_DC",
            "DCAC_P_AC": "P_AC",
            "DCAC_Q_AC": "Q_AC",
            "ACAC_P_FROM": "P_FROM",
            "ACAC_Q_FROM": "Q_FROM",
            "ACAC_P_TO": "P_TO",
            "ACAC_Q_TO": "Q_TO",
        }
        return mapping.get(prefix)

    def _targeted_converter_specs_for_state(
        self,
        prefix: str,
        name: str,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        meas_type = self._converter_meas_type_for_state_prefix(prefix)
        if meas_type is None:
            return []
        return [
            spec
            for spec in self._hybrid_converter_measurement_specs(source=source)
            if spec.device_name == name and spec.meas_type == meas_type
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
            existing_names = {meas.name for meas in self.measurements}
            added = 0
            refreshed = False
            measurement_count_before = len(self.measurements)
            remaining = batch_limit
            for label, _score in observability.weak_states:
                if added >= remaining:
                    break
                next_idx, added_count = self._append_targeted_observability_pseudo(
                    next_idx,
                    label,
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
        existing_names = {meas.name for meas in self.measurements}
        candidates: List[Measurement] = []

        def add(device_type: str, device_name: str, meas_type: str, value: float) -> None:
            key = (device_type, device_name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if key in existing_keys or pseudo_name in existing_names:
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
                )
            )
            existing_keys.add(key)
            existing_names.add(pseudo_name)

        for spec in self._side_observability_pseudo_specs("ac"):
            add(spec.device_type, spec.device_name, spec.meas_type, spec.value)
        for spec in self._side_observability_pseudo_specs("dc"):
            add(spec.device_type, spec.device_name, spec.meas_type, spec.value)

        for spec in self._hybrid_converter_measurement_specs(source="flow"):
            add(spec.device_type, spec.device_name, spec.meas_type, spec.value)

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
        direction = observability_weak_direction(H, self.state_labels, observability.weak_states)
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
        prefix: str,
        name: str,
        existing_keys: set,
    ) -> List["_HybridConverterMeasurementSpec"]:
        if prefix in ("AC_I_RE", "AC_I_IM"):
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
        if prefix == "DC_I":
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
        state_label: str,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        if ":" not in state_label or max_add <= 0:
            return next_idx, 0
        prefix, name = state_label.split(":", 1)
        if prefix == "AC_THETA":
            return next_idx, 0
        added = 0

        def add(device_type: str, meas_type: str, value: float) -> None:
            nonlocal next_idx, added
            if added >= max_add:
                return
            key = (device_type, name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{name}"
            if key in existing_keys or pseudo_name in existing_names:
                return
            next_idx = self._append_pseudo_measurement(next_idx, pseudo_name, device_type, name, meas_type, value)
            existing_keys.add(key)
            existing_names.add(pseudo_name)
            added += 1

        if prefix == "AC_V" and name in self.ac_node_by_name:
            node = self.ac_node_by_name[name]
            add("ACNode", "V", float(getattr(node, "voltage", 1.0) or 1.0))
            return next_idx, added
        if prefix == "DC_V" and name in self.dc_node_by_name:
            node = self.dc_node_by_name[name]
            add("DCNode", "V", float(getattr(node, "voltage", 1.0) or 1.0))
            return next_idx, added
        converter_specs = self._targeted_converter_specs_for_state(prefix, name, source="flow")
        if converter_specs:
            for spec in converter_specs:
                add(spec.device_type, spec.meas_type, spec.value)
            return next_idx, added
        side_specs = self._targeted_side_specs_for_state(prefix, name, existing_keys)
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

    def _measurement_residual(self, z: np.ndarray, z_est: np.ndarray, measurements: Sequence[Measurement]) -> np.ndarray:
        angle_mask = self._angle_residual_mask(measurements)
        return build_measurement_residual(z, z_est, angle_mask)

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

    def _ac_voltage_col_for_node(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        if ac is None or int(node_idx) not in ac.node_pos:
            return -1
        pos = int(ac.node_pos[int(node_idx)])
        col = int(ac.voltage_col[pos])
        return col if col >= 0 else -1

    def _dc_voltage_col_for_node(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        if dc is None or int(node_idx) not in dc.node_pos:
            return -1
        pos = int(dc.node_pos[int(node_idx)])
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
        rows = plan.dcac_rows
        if rows.size:
            codes = plan.dcac_codes
            pos = plan.dcac_pos
            base = 3 * pos
            p_dc = hybrid_x[base]
            p_ac = hybrid_x[base + 1]
            q_ac = hybrid_x[base + 2]
            v_dc = self._state_values_from_cols(
                dc_x,
                plan.dcac_dc_v_col,
                plan.dcac_dc_v_default,
                self.dc_state_start,
            )
            v_ac = self._state_values_from_cols(ac_x, plan.dcac_ac_v_col, plan.dcac_ac_v_default)
            for code, source in (
                (1, p_dc),
                (2, p_ac),
                (3, q_ac),
                (4, v_dc),
                (6, v_ac),
            ):
                mask = codes == code
                if np.any(mask):
                    values[rows[mask]] = source[mask]
            mask = codes == 5
            if np.any(mask):
                values[rows[mask]] = np.divide(
                    p_dc[mask],
                    v_dc[mask],
                    out=np.zeros(int(np.count_nonzero(mask)), dtype=np.float64),
                    where=np.abs(v_dc[mask]) > self.min_current_voltage,
                )
            mask = codes == 7
            if np.any(mask):
                values[rows[mask]] = np.divide(
                    np.hypot(p_ac[mask], q_ac[mask]),
                    v_ac[mask],
                    out=np.zeros(int(np.count_nonzero(mask)), dtype=np.float64),
                    where=np.abs(v_ac[mask]) > self.min_current_voltage,
                )

        rows = plan.acac_rows
        if rows.size:
            codes = plan.acac_codes
            pos = plan.acac_pos
            base = 3 * len(self.dcac_converters) + 4 * pos
            p_from = hybrid_x[base]
            q_from = hybrid_x[base + 1]
            p_to = hybrid_x[base + 2]
            q_to = hybrid_x[base + 3]
            v_from = self._state_values_from_cols(ac_x, plan.acac_from_v_col, plan.acac_from_v_default)
            v_to = self._state_values_from_cols(ac_x, plan.acac_to_v_col, plan.acac_to_v_default)
            for code, source in (
                (1, p_from),
                (2, q_from),
                (3, p_to),
                (4, q_to),
                (5, v_from),
                (7, v_to),
            ):
                mask = codes == code
                if np.any(mask):
                    values[rows[mask]] = source[mask]
            mask = codes == 6
            if np.any(mask):
                values[rows[mask]] = np.divide(
                    np.hypot(p_from[mask], q_from[mask]),
                    v_from[mask],
                    out=np.zeros(int(np.count_nonzero(mask)), dtype=np.float64),
                    where=np.abs(v_from[mask]) > self.min_current_voltage,
                )
            mask = codes == 8
            if np.any(mask):
                values[rows[mask]] = np.divide(
                    np.hypot(p_to[mask], q_to[mask]),
                    v_to[mask],
                    out=np.zeros(int(np.count_nonzero(mask)), dtype=np.float64),
                    where=np.abs(v_to[mask]) > self.min_current_voltage,
                )

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.evaluate(x, measurements)
        measurements = self._normalize_measurements(measurements)
        ac_x, dc_x, hybrid_x = self._split_state(x)
        values = np.zeros(len(measurements), dtype=np.float64)
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
        H = self._jacobian_builder if measurements is self.active_measurements else SparseJacobianBuilder(shape)
        H.reset()
        blocks = self._measurement_blocks_for(measurements)
        self._append_sub_jacobian(H, self._ac_sub_estimator, ac_x, blocks["ac"], 0)
        self._append_sub_jacobian(H, self._dc_sub_estimator, dc_x, blocks["dc"], self.dc_state_start)
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
    ) -> None:
        if estimator is None or len(block.measurements) == 0:
            return
        H = estimator.jacobian_sparse(sub_x, block.measurements).tocoo()
        if H.nnz == 0:
            return
        rows = block.rows[H.row].astype(np.int32, copy=False)
        cols = H.col.astype(np.int32, copy=False) + int(col_offset)
        data = H.data.astype(np.float64, copy=False)
        target.add_many(rows, cols, data)

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
        row_array = plan.dcac_rows
        if row_array.size:
            codes = plan.dcac_codes
            pos = plan.dcac_pos
            base = 3 * pos
            p_dc = hybrid_x[base]
            p_ac = hybrid_x[base + 1]
            q_ac = hybrid_x[base + 2]
            p_dc_col = self.dcac_p_dc_state_col[pos]
            p_ac_col = self.dcac_p_ac_state_col[pos]
            q_ac_col = self.dcac_q_ac_state_col[pos]
            v_dc_col = plan.dcac_dc_v_col
            v_ac_col = plan.dcac_ac_v_col
            v_dc = self._state_values_from_cols(dc_x, v_dc_col, plan.dcac_dc_v_default, self.dc_state_start)
            v_ac = self._state_values_from_cols(ac_x, v_ac_col, plan.dcac_ac_v_default)

            self._append_vector_jacobian_entries(target, row_array, p_dc_col, 1.0, codes == 1)
            self._append_vector_jacobian_entries(target, row_array, p_ac_col, 1.0, codes == 2)
            self._append_vector_jacobian_entries(target, row_array, q_ac_col, 1.0, codes == 3)
            self._append_vector_jacobian_entries(target, row_array, v_dc_col, 1.0, codes == 4)
            self._append_vector_jacobian_entries(target, row_array, v_ac_col, 1.0, codes == 6)

            valid = (codes == 5) & (np.abs(v_dc) > self.min_current_voltage)
            self._append_vector_jacobian_entries(
                target,
                row_array,
                p_dc_col,
                self._masked_divide(1.0, v_dc, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                v_dc_col,
                self._masked_divide(-p_dc, v_dc * v_dc, valid),
                valid,
            )

            s_ac = np.hypot(p_ac, q_ac)
            valid = (codes == 7) & (np.abs(v_ac) > self.min_current_voltage) & (s_ac > 1e-12)
            s_ac_v = s_ac * v_ac
            self._append_vector_jacobian_entries(
                target,
                row_array,
                p_ac_col,
                self._masked_divide(p_ac, s_ac_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                q_ac_col,
                self._masked_divide(q_ac, s_ac_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                v_ac_col,
                self._masked_divide(-s_ac, v_ac * v_ac, valid),
                valid,
            )

        row_array = plan.acac_rows
        if row_array.size:
            codes = plan.acac_codes
            pos = plan.acac_pos
            base = 3 * len(self.dcac_converters) + 4 * pos
            p_from = hybrid_x[base]
            q_from = hybrid_x[base + 1]
            p_to = hybrid_x[base + 2]
            q_to = hybrid_x[base + 3]
            p_from_col = self.acac_p_from_state_col[pos]
            q_from_col = self.acac_q_from_state_col[pos]
            p_to_col = self.acac_p_to_state_col[pos]
            q_to_col = self.acac_q_to_state_col[pos]
            v_from_col = plan.acac_from_v_col
            v_to_col = plan.acac_to_v_col
            v_from = self._state_values_from_cols(ac_x, v_from_col, plan.acac_from_v_default)
            v_to = self._state_values_from_cols(ac_x, v_to_col, plan.acac_to_v_default)

            self._append_vector_jacobian_entries(target, row_array, p_from_col, 1.0, codes == 1)
            self._append_vector_jacobian_entries(target, row_array, q_from_col, 1.0, codes == 2)
            self._append_vector_jacobian_entries(target, row_array, p_to_col, 1.0, codes == 3)
            self._append_vector_jacobian_entries(target, row_array, q_to_col, 1.0, codes == 4)
            self._append_vector_jacobian_entries(target, row_array, v_from_col, 1.0, codes == 5)
            self._append_vector_jacobian_entries(target, row_array, v_to_col, 1.0, codes == 7)

            s_from = np.hypot(p_from, q_from)
            valid = (codes == 6) & (np.abs(v_from) > self.min_current_voltage) & (s_from > 1e-12)
            s_from_v = s_from * v_from
            self._append_vector_jacobian_entries(
                target,
                row_array,
                p_from_col,
                self._masked_divide(p_from, s_from_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                q_from_col,
                self._masked_divide(q_from, s_from_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                v_from_col,
                self._masked_divide(-s_from, v_from * v_from, valid),
                valid,
            )

            s_to = np.hypot(p_to, q_to)
            valid = (codes == 8) & (np.abs(v_to) > self.min_current_voltage) & (s_to > 1e-12)
            s_to_v = s_to * v_to
            self._append_vector_jacobian_entries(
                target,
                row_array,
                p_to_col,
                self._masked_divide(p_to, s_to_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                q_to_col,
                self._masked_divide(q_to, s_to_v, valid),
                valid,
            )
            self._append_vector_jacobian_entries(
                target,
                row_array,
                v_to_col,
                self._masked_divide(-s_to, v_to * v_to, valid),
                valid,
            )

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
        if (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
            and self._initial_observability_cache is not None
        ):
            return self._initial_observability_cache
        measurements = self._normalize_measurements(measurements)
        x = self.initial_state() if x is None else x
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            return ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])
        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.state_labels,
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
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        delegate = self._delegate()
        if delegate is not None:
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
                result = delegate.estimate(measurements, x0, verbose=verbose, observability=observability)
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
        z_est = np.array([], dtype=np.float64)
        residual = np.array([], dtype=np.float64)

        for iteration in range(1, self.max_iter + 1):
            z_est = self.evaluate(x, measurements)
            residual = self._measurement_residual(z, z_est, measurements)
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
                candidate_z_est = self.evaluate(candidate, measurements)
                candidate_residual = self._measurement_residual(z, candidate_z_est, measurements)
                candidate_objective = self._weighted_objective(weight, candidate_residual)
                if np.isfinite(candidate_objective) and candidate_objective <= objective + 1e-12:
                    x = candidate
                    objective = candidate_objective
                    residual = candidate_residual
                    z_est = candidate_z_est
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
            z_est = self.evaluate(x, measurements)
            residual = self._measurement_residual(z, z_est, measurements)
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
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
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        return EstimateResult(
            converged=converged,
            iterations=iteration,
            objective=objective,
            max_correction=max_correction,
            residual_inf=residual_inf,
            x=x,
            z_est=z_est,
            residual=residual,
            H=H,
            gain=gain,
            measurements=measurements if isinstance(measurements, MeasurementList) else list(measurements),
            observability=observability,
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
        _z, weights = self._measurement_vectors(result.measurements)
        H = result.H if result.H is not None else self.jacobian_sparse(result.x, result.measurements)
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
            meas = result.measurements[int(idx)]
            bad_items.append(
                BadDataItem(
                    measurement=meas,
                    residual=float(result.residual[idx]),
                    normalized_residual=float(normalized[idx]),
                    estimated_value=float(result.z_est[idx]),
                    measured_value=float(meas.value),
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
    ) -> SEResult:
        """Build the structured state-estimation result snapshot after WLS."""
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        self.se_result = SEResult.from_estimate_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurements=self.measurements,
        )
        return self.se_result

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.estimate_with_bad_data_removal(threshold=threshold, max_remove=max_remove, verbose=verbose)
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
    )
    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    initial_observability = estimator.observability_analysis()
    _print_observability(initial_observability)
    if args.remove_bad_data:
        result, removed = estimator.estimate_with_bad_data_removal(
            threshold=bad_threshold,
            max_remove=args.max_remove,
            verbose=not args.quiet,
        )
        if removed:
            print("Removed bad data:")
            for item in removed:
                print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")
    else:
        result = estimator.estimate(verbose=not args.quiet, observability=initial_observability)

    profile_stage_start = time.perf_counter()
    bad_items, normalized = estimator.identify_bad_data(result, bad_threshold)
    estimator._record_profile_time("bad_data.identify", time.perf_counter() - profile_stage_start)
    print(
        "State estimation: "
        f"converged={result.converged}, iterations={result.iterations}, "
        f"objective={result.objective:.6e}, max_dx={result.max_correction:.3e}, "
        f"residual_inf={result.residual_inf:.3e}"
    )
    _print_bad_data(bad_items, normalized, bad_threshold)
    if args.se_result:
        profile_stage_start = time.perf_counter()
        se_result = estimator.build_se_result(result, bad_items=bad_items, normalized_residual=normalized)
        estimator._record_profile_time("se_result.build", time.perf_counter() - profile_stage_start)
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
