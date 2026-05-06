import argparse
import contextlib
import io
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import coo_matrix, csr_matrix, issparse
except Exception:
    coo_matrix = csr_matrix = None

    def issparse(_matrix):
        return False

try:
    from scipy.sparse.csgraph import maximum_bipartite_matching as sp_maximum_bipartite_matching
except Exception:
    sp_maximum_bipartite_matching = None


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_flow import matpower_branch_stamp, matpower_branch_stamp_vectorized
from ac_model import ACPowerNetwork
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from efile_read import EBook
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    SparseJacobianBuilder,
    angle_residual_mask,
    build_normal_equations,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    measurement_residual as build_measurement_residual,
    NormalEquationSolver,
    _normal_equation_structural_pattern,
    observability_rank_details,
    solve_normal_equations_with_factor,
    sparse_structural_rank,
    unanchored_angle_state_labels,
)
from unit_system import ac_current_base_ka


DEFAULT_CASE = ROOT_DIR / "data" / "ac" / "ieee39.e"
DEFAULT_MEAS = ROOT_DIR / "data" / "ac" / "ieee39.meas"


_DEVICE_TYPE_CODES = {
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
}

_TERMINAL_POWER_MEASUREMENT_TYPES = frozenset(("P_FROM", "Q_FROM", "P_TO", "Q_TO"))
_PSEUDO_DEVICE_SUMMARY_TYPES = frozenset(("ACGenerator", "ACLoad"))
_PSEUDO_MEASUREMENT_SUMMARY_TYPES = {
    "ACGenerator": frozenset(("P_GEN", "Q_GEN")),
    "ACLoad": frozenset(("P_LOAD", "Q_LOAD")),
    "ACZeroBranch": frozenset(("P_FROM", "Q_FROM", "V_FROM", "I_FROM")),
    "ACSwitch": frozenset(("P_FROM", "Q_FROM", "V_FROM", "I_FROM")),
}

_OBSERVABILITY_RESULT_CACHE = {}


def _file_cache_key(file_name: Path) -> Tuple[Path, int, int]:
    path = Path(file_name).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _read_measurements_direct(file_name: Path, measurement_cls):
    required_columns = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
    header = None
    column_index = None
    header_len = 0
    idx_col = name_col = dev_type_col = dev_name_col = meas_type_col = weight_col = valid_col = value_col = -1
    measurements = []
    append_measurement = measurements.append
    new_measurement = measurement_cls.__new__
    int_cell = int
    float_cell = float
    device_type_cache = {}
    measurement_type_cache = {}
    in_measurement = False
    with open(file_name, mode="rt", encoding="utf8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            first = raw_line[0] if raw_line else ""
            if not in_measurement:
                if first == "<" and raw_line.strip() == "<Measurement>":
                    in_measurement = True
                continue
            if first == "@":
                header = raw_line[1:].split()
                column_index = {name: idx for idx, name in enumerate(header)}
                missing = [name for name in required_columns if name not in column_index]
                if missing:
                    raise RuntimeError(f"{file_name} Measurement header is missing columns: {missing}")
                header_len = len(header)
                idx_col = column_index["idx"]
                name_col = column_index["name"]
                dev_type_col = column_index["dev_type"]
                dev_name_col = column_index["dev_name"]
                meas_type_col = column_index["meas_type"]
                weight_col = column_index["weight"]
                valid_col = column_index["valid"]
                value_col = column_index["value"]
                continue
            if first == "#":
                if header is None or column_index is None:
                    raise RuntimeError(f"{file_name} Measurement data appears before the header")
                row = raw_line[1:].split()
                if len(row) < header_len:
                    raise RuntimeError(f"Malformed Measurement row at line {line_no} in {file_name}")
                raw_device_type = row[dev_type_col]
                device_type = device_type_cache.get(raw_device_type)
                if device_type is None:
                    device_type = raw_device_type
                    device_type_cache[raw_device_type] = device_type
                raw_meas_type = row[meas_type_col]
                meas_type = measurement_type_cache.get(raw_meas_type)
                if meas_type is None:
                    meas_type = raw_meas_type.upper()
                    measurement_type_cache[raw_meas_type] = meas_type
                idx = int_cell(row[idx_col])
                name = row[name_col]
                device_name = row[dev_name_col]
                weight = float_cell(row[weight_col])
                valid = row[valid_col] == "1"
                value = float_cell(row[value_col])
                meas = new_measurement(measurement_cls)
                meas.idx = idx
                meas.name = name
                meas.device_type = device_type
                meas.device_name = device_name
                meas.meas_type = meas_type
                meas.weight = weight
                meas.valid = valid
                meas.value = value
                append_measurement(meas)
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line == "</Measurement>":
                break
            if first != "#":
                raise SyntaxError(f"Invalid Measurement row at line {line_no} in {file_name}")
    if not in_measurement:
        raise RuntimeError(f"{file_name} does not contain a <Measurement> block")
    if header is None:
        raise RuntimeError(f"{file_name} Measurement block does not contain a header")
    return measurements


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
    def read_from_file(cls, file_name: Path) -> List["Measurement"]:
        return _read_measurements_direct(file_name, cls)

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


def _print_iteration_header() -> None:
    print("Iteration process:")
    print("  iter objective      max_dx      norm_res    step   status")


def _print_iteration(
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
        self.voltage_floor = self.params.voltage_floor
        self.min_current_voltage = self.params.min_current_voltage

        profile_start = time.perf_counter()
        self.network = self._profile_call("init.load_network", self._load_network, self.e_file)
        self.measurements = self._profile_call("init.load_measurements", self._load_measurements, self.meas_file)
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)

        self.nodes = sorted([node for node in self.network.nodes if getattr(node, "is_alive", False)], key=lambda item: item.idx)
        if not self.nodes:
            raise RuntimeError("No alive AC nodes are available for state estimation")
        self.file_theta = np.asarray(
            [float(getattr(node, "angle", 0.0) or 0.0) for node in self.nodes],
            dtype=np.float64,
        )
        self.file_voltage = np.asarray(
            [float(getattr(node, "voltage", 1.0) or 1.0) for node in self.nodes],
            dtype=np.float64,
        )

        self.node_pos = {node.idx: pos for pos, node in enumerate(self.nodes)}
        self.node_by_name = {node.name: node for node in self.nodes}
        self.branch_by_name = {br.name: br for br in self.network.branches if getattr(br, "is_alive", False)}
        self.transformer_by_name = {tr.name: tr for tr in self.network.transformers if getattr(tr, "is_alive", False)}
        self.generator_by_name = {gen.name: gen for gen in self.network.generators if getattr(gen, "is_alive", False)}
        self.load_by_name = {load.name: load for load in self.network.loads if getattr(load, "is_alive", False)}
        self.zero_branch_by_name = {
            zbr.name: zbr
            for zbr in self.network.zero_branches
            if getattr(zbr, "is_alive", False)
        }
        self.switch_by_name = {sw.name: sw for sw in self.network.switches if getattr(sw, "is_alive", False)}

        self.zero_branches = sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
        self.switches = sorted(self.switch_by_name.values(), key=lambda item: item.idx)
        # Reference-bus voltage values must be known before the compact state layout
        # is built, so real file measurements are normalized before selecting them.
        self._profile_call("init.convert_measurements", self._convert_measurements_to_pu)
        self.node_voltage_measurements = self._profile_call("init.node_voltage_measurements", self._node_voltage_measurements)
        self.node_degrees = self._profile_call("init.node_degrees", self._node_incident_degrees)
        self.references = self._profile_call("init.references", self._select_reference_nodes)
        self.ref_idx = {self.node_pos[node.idx] for node in self.references}
        self.reference_voltage_by_pos = {
            self.node_pos[node.idx]: self.node_voltage_measurements[node.idx]
            for node in self.references
            if node.idx in self.node_voltage_measurements and node.idx in self.node_pos
        }
        self.reference_angle_by_pos = self._profile_call("init.reference_angles", self._reference_angle_offsets)
        self._profile_call("init.rebase_angles", self._rebase_angle_measurements)
        self._profile_call("init.zero_tie_layout", self._build_zero_tie_state_layout)
        self.zero_current_devices = [("Z", zbr) for zbr in self.zero_branches] + [("S", sw) for sw in self.switches]
        self.zero_current_pos = {(kind, dev.name): pos for pos, (kind, dev) in enumerate(self.zero_current_devices)}
        self.zero_branch_pos = {zbr.name: self.zero_current_pos[("Z", zbr.name)] for zbr in self.zero_branches}
        self.switch_pos = {sw.name: self.zero_current_pos[("S", sw.name)] for sw in self.switches}
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
        self.generator_order = sorted(self.generator_by_name.values(), key=lambda item: item.idx)
        self.load_order = sorted(self.load_by_name.values(), key=lambda item: item.idx)
        self.generator_pos_array = np.asarray([self.node_pos[gen.node] for gen in self.generator_order], dtype=np.int32)
        self.load_pos_array = np.asarray([self.node_pos[load.node] for load in self.load_order], dtype=np.int32)
        self.load_pv0_array = np.asarray([load.pv0 for load in self.load_order], dtype=np.float64)
        self.load_pv1_array = np.asarray([load.pv1 for load in self.load_order], dtype=np.float64)
        self.load_pv2_array = np.asarray([load.pv2 for load in self.load_order], dtype=np.float64)
        self.load_qv0_array = np.asarray([load.qv0 for load in self.load_order], dtype=np.float64)
        self.load_qv1_array = np.asarray([load.qv1 for load in self.load_order], dtype=np.float64)
        self.load_qv2_array = np.asarray([load.qv2 for load in self.load_order], dtype=np.float64)
        self.n_nodes = len(self.nodes)
        gen_seed = [self._generator_pseudo_power(gen) for gen in self.generator_order]
        load_seed = [self._load_pseudo_power(load) for load in self.load_order]
        self.initial_gen_p_array = np.asarray([p for p, _q in gen_seed], dtype=np.float64)
        self.initial_gen_q_array = np.asarray([q for _p, q in gen_seed], dtype=np.float64)
        self.initial_load_p_array = np.asarray([p for p, _q in load_seed], dtype=np.float64)
        self.initial_load_q_array = np.asarray([q for _p, q in load_seed], dtype=np.float64)
        self.n_generator_power = len(self.generator_order)
        self.n_load_power = len(self.load_order)
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
        self.generator_balance_minus_ones = -np.ones(self.n_generator_power, dtype=np.float64)
        self.load_balance_ones = np.ones(self.n_load_power, dtype=np.float64)
        self.base_switch_re = self.n_angle + self.n_voltage
        self.base_switch_im = self.base_switch_re + self.n_switch_current
        self.base_gen_p = self.base_switch_im + self.n_switch_current
        self.base_gen_q = self.base_gen_p + self.n_generator_power
        self.base_load_p = self.base_gen_q + self.n_generator_power
        self.base_load_q = self.base_load_p + self.n_load_power
        self.n_state = self.base_load_q + self.n_load_power
        self.gen_p_col_by_name = {
            gen.name: self.base_gen_p + idx for idx, gen in enumerate(self.generator_order)
        }
        self.gen_q_col_by_name = {
            gen.name: self.base_gen_q + idx for idx, gen in enumerate(self.generator_order)
        }
        self.generator_state_index_by_name = {
            gen.name: idx for idx, gen in enumerate(self.generator_order)
        }
        self.load_p_col_by_name = {
            load.name: self.base_load_p + idx for idx, load in enumerate(self.load_order)
        }
        self.load_q_col_by_name = {
            load.name: self.base_load_q + idx for idx, load in enumerate(self.load_order)
        }
        self.load_state_index_by_name = {
            load.name: idx for idx, load in enumerate(self.load_order)
        }
        self.state_labels = [f"theta:{self.nodes[pos].name}" for pos in self.angle_state_pos] + [
            f"V:{self.nodes[pos].name}" for pos in self.voltage_state_pos
        ]
        self.state_labels.extend(f"I_{kind}_RE:{dev.name}" for kind, dev in self.zero_current_devices)
        self.state_labels.extend(f"I_{kind}_IM:{dev.name}" for kind, dev in self.zero_current_devices)
        self.state_labels.extend(f"P_GEN:{gen.name}" for gen in self.generator_order)
        self.state_labels.extend(f"Q_GEN:{gen.name}" for gen in self.generator_order)
        self.state_labels.extend(f"P_LOAD:{load.name}" for load in self.load_order)
        self.state_labels.extend(f"Q_LOAD:{load.name}" for load in self.load_order)

        self.branch_stamp_by_name = self._profile_call(
            "init.branch_stamps",
            self._build_branch_stamp_map,
            list(self.branch_by_name.values()),
            False,
        )
        self.transformer_stamp_by_name = self._profile_call(
            "init.transformer_stamps",
            self._build_branch_stamp_map,
            list(self.transformer_by_name.values()),
            True,
        )

        self.Y = self._profile_call("init.y_matrix", self._build_y_matrix)
        self._profile_call("init.y_row_cache", self._prepare_y_row_cache)
        self.loads_at_pos = self._profile_call("init.group_loads", self._group_loads)
        self.generators_at_pos = self._profile_call("init.group_generators", self._group_generators)
        self.generator_share_by_name = self._profile_call("init.generator_shares", self._generator_shares)
        self._initial_observability_cache = None
        # Add priors after unit conversion because model objects are already normalized.
        self._profile_call("init.pseudo_measurements", self._add_pseudo_power_measurements)
        self._profile_call("init.seed_power_state", self._seed_power_state_arrays_from_measurements)
        self._profile_call("init.power_balance_constraints", self._add_power_balance_constraint_measurements)
        self.targeted_observability_pseudo_count = 0
        self._profile_call("init.active_refresh", self._refresh_active_measurement_indexes)
        self.targeted_observability_pseudo_count = self._profile_call(
            "init.targeted_observability",
            self._add_targeted_observability_pseudo_measurements,
        )
        self._record_profile_time("init.total", time.perf_counter() - profile_start)

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    def _profile_call(self, name: str, func, *args, **kwargs):
        if not self.profile_enabled:
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._record_profile_time(name, time.perf_counter() - start)

    def _default_observability_cache_key(self) -> Tuple[object, ...]:
        return (
            _file_cache_key(self.e_file),
            _file_cache_key(self.meas_file),
            bool(self.flat_start),
            int(self.targeted_pseudo_measurement_max),
            int(len(self.active_measurements)),
            int(self._max_measurement_idx),
            int(self.n_state),
        )

    def _install_active_plan_cache_entries(self) -> None:
        active_id = id(self.active_measurements)
        self._branch_transformer_vector_plan_cache[active_id] = (
            self.active_measurements,
            self._active_branch_transformer_vector_plan,
        )
        self._simple_jacobian_plan_cache[active_id] = (
            self.active_measurements,
            self._active_simple_jacobian_plan,
        )
        self._zero_current_vector_plan_cache[active_id] = (
            self.active_measurements,
            self._active_zero_current_vector_plan,
        )
        self._generator_measurement_plan_cache[active_id] = (
            self.active_measurements,
            self._active_generator_measurement_plan,
        )
        self._balance_measurement_plan_cache[active_id] = (
            self.active_measurements,
            self._active_balance_measurement_plan,
        )

    def _refresh_active_measurement_indexes(self) -> None:
        """Rebuild active measurement arrays and vectorized measurement plans."""
        self._initial_observability_cache = None
        active_measurements = []
        active_z = []
        active_weight = []
        active_angle_mask = []
        active_device_type_codes = []
        active_rows_by_device_type_code = {}
        append_active_measurement = active_measurements.append
        append_active_z = active_z.append
        append_active_weight = active_weight.append
        append_active_angle_mask = active_angle_mask.append
        append_active_device_type_code = active_device_type_codes.append
        device_type_code_get = _DEVICE_TYPE_CODES.get
        rows_by_type_setdefault = active_rows_by_device_type_code.setdefault
        angle_types = ANGLE_MEASUREMENT_TYPES
        max_idx = 0
        first_active_weight = None
        active_weights_are_uniform = True
        for meas in self.measurements:
            if meas.idx > max_idx:
                max_idx = int(meas.idx)
            if not meas.valid or meas.weight <= 0.0:
                continue
            active_row = len(active_measurements)
            append_active_measurement(meas)
            append_active_z(meas.value)
            append_active_weight(meas.weight)
            append_active_angle_mask(meas.meas_type in angle_types)
            device_type_code = device_type_code_get(meas.device_type, 0)
            append_active_device_type_code(device_type_code)
            if first_active_weight is None:
                first_active_weight = float(meas.weight)
            elif active_weights_are_uniform and float(meas.weight) != first_active_weight:
                active_weights_are_uniform = False
            rows_by_type_setdefault(device_type_code, []).append(active_row)
        self._max_measurement_idx = max_idx
        self.active_measurements = active_measurements
        self.active_z = np.asarray(active_z, dtype=np.float64)
        self.active_weight = np.asarray(active_weight, dtype=np.float64)
        self.active_angle_residual_mask = np.asarray(active_angle_mask, dtype=bool)
        self.active_device_type_codes = np.asarray(active_device_type_codes, dtype=np.int16)
        self.active_has_angle_residuals = any(active_angle_mask)
        self.active_weights_are_uniform = bool(active_weight) and active_weights_are_uniform
        self.active_uniform_weight = first_active_weight if self.active_weights_are_uniform else None
        self._active_rows_by_device_type_code = {
            int(device_type_code): tuple(rows)
            for device_type_code, rows in active_rows_by_device_type_code.items()
        }
        self._branch_transformer_vector_plan_cache = {}
        self._simple_jacobian_plan_cache = {}
        self._zero_current_vector_plan_cache = {}
        self._generator_measurement_plan_cache = {}
        self._balance_measurement_plan_cache = {}
        self._active_branch_transformer_vector_plan = None
        self._active_branch_transformer_vector_plan = self._branch_transformer_vector_plan(self.active_measurements)
        self._active_simple_jacobian_plan = None
        self._active_simple_jacobian_plan = self._simple_jacobian_plan(self.active_measurements)
        self._active_zero_current_vector_plan = None
        self._active_zero_current_vector_plan = self._zero_current_vector_plan(self.active_measurements)
        self._active_generator_measurement_plan = None
        self._active_generator_measurement_plan = self._generator_measurement_plan(self.active_measurements)
        self._active_balance_measurement_plan = None
        self._active_balance_measurement_plan = self._balance_measurement_plan(self.active_measurements)
        self.active_measurements_are_vectorized = bool(
            np.all(
                self._active_branch_transformer_vector_plan["handled_mask"]
                | self._active_simple_jacobian_plan["handled_mask"]
                | self._active_zero_current_vector_plan["handled_mask"]
                | self._active_generator_measurement_plan["handled_mask"]
                | self._active_balance_measurement_plan["handled_mask"]
            )
        )
        self._install_active_plan_cache_entries()

    def _measurement_rows_for_types(
        self,
        measurements: Sequence[Measurement],
        device_types: Tuple[str, ...],
    ):
        """Yield candidate measurement rows, using the active type index when available."""
        if measurements is self.active_measurements:
            rows_by_device_type_code = getattr(self, "_active_rows_by_device_type_code", None)
            if rows_by_device_type_code is not None:
                for device_type in device_types:
                    for row in rows_by_device_type_code.get(_DEVICE_TYPE_CODES.get(device_type, 0), ()):
                        yield row, measurements[row]
                return
        device_type_set = set(device_types)
        for row, meas in enumerate(measurements):
            if meas.device_type in device_type_set:
                yield row, meas

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        if isinstance(measurements, list):
            return measurements
        return list(measurements)

    def _measurement_vectors(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
        z = np.asarray([meas.value for meas in measurements], dtype=np.float64)
        weight = np.asarray([meas.weight for meas in measurements], dtype=np.float64)
        return z, weight

    @staticmethod
    def _uniform_weight(weight: np.ndarray) -> Optional[float]:
        if weight.size == 0:
            return None
        first_weight = float(weight[0])
        return first_weight if bool(np.all(weight == first_weight)) else None

    def _angle_residual_mask(self, measurements: Sequence[Measurement]) -> np.ndarray:
        if measurements is self.active_measurements:
            return self.active_angle_residual_mask
        return angle_residual_mask(measurements)

    def _has_angle_residuals(self, measurements: Sequence[Measurement], angle_mask: np.ndarray) -> bool:
        if measurements is self.active_measurements:
            return self.active_has_angle_residuals
        return bool(np.any(angle_mask))

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
    ) -> np.ndarray:
        angle_mask = self._angle_residual_mask(measurements)
        return build_measurement_residual(
            z,
            z_est,
            angle_mask,
            has_angle_residuals=self._has_angle_residuals(measurements, angle_mask),
        )

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _disable_angle_measurements(self) -> None:
        """Keep all phase-angle measurement rows out of WLS; flat starts store them as zero."""
        for meas in self.measurements:
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                meas.valid = False
                if self.flat_start:
                    meas.value = 0.0

    def _disable_unavailable_measurements(self) -> None:
        """Keep invalid/off-topology measurement rows out of unit conversion and WLS."""
        device_maps = {
            "ACNode": self.node_by_name,
            "ACBranch": self.branch_by_name,
            "ACTransformer": self.transformer_by_name,
            "ACSwitch": self.switch_by_name,
            "ACZeroBranch": self.zero_branch_by_name,
            "ACGenerator": self.generator_by_name,
            "ACLoad": self.load_by_name,
        }
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            if meas.device_type in ("ACZeroBranchConstraint", "ACSwitchConstraint"):
                meas.valid = False
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                meas.valid = False

    def _load_network(self, e_file: Path) -> ACPowerNetwork:
        """Read the AC case and build topology references used by measurements."""
        network = ACPowerNetwork()
        self._profile_call("init.network_read_file", network.read_from_file, e_file)
        self._profile_call("init.network_topo", network.topo)
        return network

    @staticmethod
    def _load_measurements(meas_file: Path) -> List[Measurement]:
        return Measurement.read_from_file(meas_file)

    def _node_current_base(self, node_idx: int) -> float:
        return self.i_scale * ac_current_base_ka(self.p_base_kW, self._voltage_base(node_idx))

    def _voltage_base(self, node_idx: int) -> float:
        return float(self.network.node_dict[node_idx].vbase)

    def _voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._voltage_base(node_idx)

    def _power_file_base(self) -> float:
        return self.p_base

    def _terminal_measurement_scale(self, meas: Measurement, device) -> float:
        if meas.meas_type.startswith(("P_", "Q_")):
            return self._power_file_base()
        if meas.meas_type.endswith("_FROM"):
            node_idx = device.i_node
        elif meas.meas_type.endswith("_TO"):
            node_idx = device.j_node
        else:
            return 1.0
        if meas.meas_type.startswith("V_"):
            return self._voltage_file_base(node_idx)
        if meas.meas_type.startswith("I_"):
            return self._node_current_base(node_idx)
        return 1.0

    def _measurement_scale(self, meas: Measurement) -> float:
        if meas.device_type == "ACNode":
            if meas.meas_type == "V":
                return self._voltage_file_base(self.node_by_name[meas.device_name].idx)
            return 1.0
        if meas.device_type == "ACBranch":
            return self._terminal_measurement_scale(meas, self.branch_by_name[meas.device_name])
        if meas.device_type == "ACTransformer":
            return self._terminal_measurement_scale(meas, self.transformer_by_name[meas.device_name])
        if meas.device_type == "ACZeroBranch":
            return self._terminal_measurement_scale(meas, self.zero_branch_by_name[meas.device_name])
        if meas.device_type == "ACSwitch":
            return self._terminal_measurement_scale(meas, self.switch_by_name[meas.device_name])
        if meas.device_type == "ACGenerator":
            gen = self.generator_by_name[meas.device_name]
            if meas.meas_type in ("P_GEN", "Q_GEN"):
                return self._power_file_base()
            if meas.meas_type == "V_GEN":
                return self._voltage_file_base(gen.node)
            if meas.meas_type == "I_GEN":
                return self._node_current_base(gen.node)
        if meas.device_type == "ACLoad":
            load = self.load_by_name[meas.device_name]
            if meas.meas_type in ("P_LOAD", "Q_LOAD"):
                return self._power_file_base()
            if meas.meas_type == "V_LOAD":
                return self._voltage_file_base(load.node)
            if meas.meas_type == "I_LOAD":
                return self._node_current_base(load.node)
        return 1.0

    def _measurement_value_to_internal_units(self, meas: Measurement) -> float:
        if meas.meas_type in ("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF"):
            return math.radians(float(meas.value))
        return float(meas.value) / self._measurement_scale(meas)

    def _convert_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal state-estimation units."""
        power_scale = self.p_base
        voltage_scale_by_node = {
            node_idx: self.u_scale * float(node.vbase)
            for node_idx, node in self.network.node_dict.items()
        }
        current_scale_by_node = {
            node_idx: self.i_scale * ac_current_base_ka(self.p_base_kW, float(node.vbase))
            for node_idx, node in self.network.node_dict.items()
        }
        node_voltage_scale_by_name = {
            name: voltage_scale_by_node[node.idx]
            for name, node in self.node_by_name.items()
        }
        branch_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.branch_by_name.items()
        }
        transformer_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.transformer_by_name.items()
        }
        zero_branch_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.zero_branch_by_name.items()
        }
        switch_terminal_scale_by_name = {
            name: (
                voltage_scale_by_node[device.i_node],
                current_scale_by_node[device.i_node],
                voltage_scale_by_node[device.j_node],
                current_scale_by_node[device.j_node],
            )
            for name, device in self.switch_by_name.items()
        }
        generator_node_scale_by_name = {
            name: (voltage_scale_by_node[gen.node], current_scale_by_node[gen.node])
            for name, gen in self.generator_by_name.items()
        }
        load_node_scale_by_name = {
            name: (voltage_scale_by_node[load.node], current_scale_by_node[load.node])
            for name, load in self.load_by_name.items()
        }
        max_idx = 0
        terminal_maps = {
            "ACBranch": branch_terminal_scale_by_name,
            "ACTransformer": transformer_terminal_scale_by_name,
            "ACZeroBranch": zero_branch_terminal_scale_by_name,
            "ACSwitch": switch_terminal_scale_by_name,
        }
        active_device_keys = set()
        active_measurement_keys = set()
        add_active_device_key = active_device_keys.add
        add_active_measurement_key = active_measurement_keys.add
        node_voltage_best: Dict[int, Tuple[float, float]] = {}
        power_seed_best: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        has_valid_angle_measurements = False

        for meas in self.measurements:
            if meas.idx > max_idx:
                max_idx = int(meas.idx)
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                if self.flat_start:
                    meas.value = 0.0
                meas.valid = False
                continue
            if not meas.valid or meas.weight <= 0.0:
                continue
            scale = 1.0
            mtype = meas.meas_type
            device_type = meas.device_type
            device_name = meas.device_name
            if device_type == "ACNode":
                node_scale = node_voltage_scale_by_name.get(device_name)
                if node_scale is None:
                    meas.valid = False
                    continue
                if mtype == "V":
                    scale = node_scale
                elif mtype in ("ANGLE", "THETA"):
                    has_valid_angle_measurements = True
                    meas.value = math.radians(float(meas.value))
                    continue
            elif device_type in terminal_maps:
                terminal_scales = terminal_maps[device_type].get(device_name)
                if terminal_scales is None:
                    meas.valid = False
                    continue
                if mtype in _TERMINAL_POWER_MEASUREMENT_TYPES:
                    scale = power_scale
                elif mtype == "V_FROM":
                    scale = terminal_scales[0]
                elif mtype == "I_FROM":
                    scale = terminal_scales[1]
                elif mtype == "V_TO":
                    scale = terminal_scales[2]
                elif mtype == "I_TO":
                    scale = terminal_scales[3]
            elif device_type == "ACZeroBranchConstraint":
                meas.valid = False
                continue
            elif device_type == "ACSwitchConstraint":
                meas.valid = False
                continue
            elif device_type == "ACGenerator":
                node_scales = generator_node_scale_by_name.get(device_name)
                if node_scales is None:
                    meas.valid = False
                    continue
                if mtype in ("P_GEN", "Q_GEN"):
                    scale = power_scale
                elif mtype == "V_GEN":
                    scale = node_scales[0]
                elif mtype == "I_GEN":
                    scale = node_scales[1]
            elif device_type == "ACLoad":
                node_scales = load_node_scale_by_name.get(device_name)
                if node_scales is None:
                    meas.valid = False
                    continue
                if mtype in ("P_LOAD", "Q_LOAD"):
                    scale = power_scale
                elif mtype == "V_LOAD":
                    scale = node_scales[0]
                elif mtype == "I_LOAD":
                    scale = node_scales[1]
            else:
                meas.valid = False
                continue
            meas.value = float(meas.value) / scale
            if device_type == "ACNode" and mtype == "V" and not meas.name.startswith("pseudo_"):
                node_idx = self.node_by_name[device_name].idx
                current = node_voltage_best.get(node_idx)
                if current is None or meas.weight > current[0]:
                    node_voltage_best[node_idx] = (float(meas.weight), float(meas.value))
            elif (
                (device_type == "ACGenerator" and mtype in ("P_GEN", "Q_GEN"))
                or (device_type == "ACLoad" and mtype in ("P_LOAD", "Q_LOAD"))
            ):
                key = (device_type, device_name, mtype)
                current = power_seed_best.get(key)
                if current is None or meas.weight > current[0]:
                    power_seed_best[key] = (float(meas.weight), float(meas.value))
            if device_type in _PSEUDO_DEVICE_SUMMARY_TYPES:
                add_active_device_key((device_type, device_name))
            if mtype in _PSEUDO_MEASUREMENT_SUMMARY_TYPES.get(device_type, ()):
                add_active_measurement_key((device_type, device_name, mtype))
        self._active_device_key_cache = active_device_keys
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = max_idx
        self._node_voltage_measurement_cache = {
            node_idx: value for node_idx, (_weight, value) in node_voltage_best.items()
        }
        self._real_power_measurement_seed_cache = power_seed_best
        self._has_valid_angle_measurements = has_valid_angle_measurements

    def _active_device_keys(self) -> set:
        """Return devices that already have at least one usable measurement."""
        if not hasattr(self, "_active_device_key_cache"):
            self._refresh_measurement_summary_cache()
        return set(self._active_device_key_cache)

    def _active_measurement_keys(self) -> set:
        """Return usable measurement keys at device and measurement-type granularity."""
        if not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        return set(self._active_measurement_key_cache)

    def _next_measurement_idx(self) -> int:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        return int(self._max_measurement_idx) + 1

    def _refresh_measurement_summary_cache(self) -> None:
        """Cache active measurement key sets and max row id for initialization scans."""
        active_device_keys = set()
        active_measurement_keys = set()
        max_idx = 0
        for meas in self.measurements:
            if meas.idx > max_idx:
                max_idx = int(meas.idx)
            if meas.valid and meas.weight > 0.0:
                active_device_keys.add((meas.device_type, meas.device_name))
                active_measurement_keys.add((meas.device_type, meas.device_name, meas.meas_type))
        self._active_device_key_cache = active_device_keys
        self._active_measurement_key_cache = active_measurement_keys
        self._max_measurement_idx = max_idx

    def _record_measurement_summary(self, meas: Measurement) -> None:
        if not hasattr(self, "_max_measurement_idx"):
            self._refresh_measurement_summary_cache()
        if meas.idx > self._max_measurement_idx:
            self._max_measurement_idx = int(meas.idx)
        if meas.valid and meas.weight > 0.0:
            self._active_device_key_cache.add((meas.device_type, meas.device_name))
            self._active_measurement_key_cache.add((meas.device_type, meas.device_name, meas.meas_type))

    def _append_pseudo_measurement(
        self,
        next_idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        value: float,
        *,
        record_summary: bool = True,
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
        if record_summary:
            self._record_measurement_summary(measurement)
        elif next_idx > getattr(self, "_max_measurement_idx", 0):
            self._max_measurement_idx = int(next_idx)
        return next_idx + 1

    @staticmethod
    def _generator_pseudo_power(gen) -> Tuple[float, float]:
        # Prefer solved output when available; otherwise use configured setpoints.
        p = getattr(gen, "p", None)
        q = getattr(gen, "q", None)
        if p is not None and q is not None and (abs(float(p)) > 1e-12 or abs(float(q)) > 1e-12):
            return float(p), float(q)
        return float(getattr(gen, "p_set", 0.0) or 0.0), float(getattr(gen, "q_set", 0.0) or 0.0)

    @staticmethod
    def _load_pseudo_power(load) -> Tuple[float, float]:
        # Loads may be voltage dependent, so evaluate the ZIP model at the current seed voltage.
        node = getattr(load, "node_obj", None)
        voltage = float(getattr(node, "voltage", 1.0) or 1.0)
        p = load.pv0 + load.pv1 * voltage + load.pv2 * voltage * voltage
        q = load.qv0 + load.qv1 * voltage + load.qv2 * voltage * voltage
        return float(p), float(q)

    def _active_angle_measurement_counts(self) -> Dict[int, int]:
        """Count usable local P/Q or direct angle measurements per AC node."""
        counts: Dict[int, int] = {}

        def add(pos: int, amount: int = 1) -> None:
            counts[pos] = counts.get(pos, 0) + amount

        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                if mtype in ("ANGLE", "THETA") and meas.device_name in self.node_by_name:
                    add(self.node_pos[self.node_by_name[meas.device_name].idx], 2)
                continue
            if not mtype.startswith(("P", "Q")):
                continue
            if meas.device_type == "ACBranch":
                dev = self.branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACTransformer":
                dev = self.transformer_by_name.get(meas.device_name)
            elif meas.device_type == "ACZeroBranch":
                dev = self.zero_branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACSwitch":
                dev = self.switch_by_name.get(meas.device_name)
            else:
                dev = None
            if dev is not None:
                if dev.i_node in self.node_pos:
                    add(self.node_pos[dev.i_node])
                if dev.j_node in self.node_pos:
                    add(self.node_pos[dev.j_node])
                continue
            if meas.device_type == "ACGenerator":
                dev = self.generator_by_name.get(meas.device_name)
                if dev is not None and dev.node in self.node_pos:
                    add(self.node_pos[dev.node])
        return counts

    def _add_pseudo_topology_measurements(self, next_idx: int) -> Tuple[int, set]:
        """Add weak priors for topology devices that have no usable measurement row."""
        measured_keys = self._active_measurement_key_cache
        added_keys = set()

        for device_type, devices in (
            ("ACZeroBranch", self.zero_branches),
            ("ACSwitch", self.switches),
        ):
            for dev in devices:
                voltage = float(getattr(getattr(dev, "i_node_obj", None), "voltage", 1.0) or 1.0)
                values = (
                    ("P_FROM", float(getattr(dev, "p", 0.0) or 0.0)),
                    ("Q_FROM", float(getattr(dev, "q", 0.0) or 0.0)),
                    ("V_FROM", voltage),
                    ("I_FROM", abs(getattr(dev, "current", 0.0) or 0.0)),
                )
                for meas_type, value in values:
                    key = (device_type, dev.name, meas_type)
                    if key in measured_keys or key in added_keys:
                        continue
                    next_idx = self._append_pseudo_measurement(
                        next_idx,
                        f"pseudo_{meas_type.lower()}_{dev.name}",
                        device_type,
                        dev.name,
                        meas_type,
                        value,
                        record_summary=False,
                    )
                    added_keys.add(key)
        return next_idx, added_keys

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        if not hasattr(self, "_active_device_key_cache") or not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        measured_devices = self._active_device_key_cache
        measured_keys = self._active_measurement_key_cache
        added_keys = set()
        next_idx = self._next_measurement_idx()
        next_idx, topology_added_keys = self._add_pseudo_topology_measurements(next_idx)
        added_keys.update(topology_added_keys)

        for gen in sorted(self.generator_by_name.values(), key=lambda item: item.idx):
            p, q = self._generator_pseudo_power(gen)
            voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
            key = ("ACGenerator", gen.name, "P_GEN")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_p_{gen.name}",
                    "ACGenerator",
                    gen.name,
                    "P_GEN",
                    p,
                    record_summary=False,
                )
                added_keys.add(key)
            key = ("ACGenerator", gen.name, "Q_GEN")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_q_{gen.name}",
                    "ACGenerator",
                    gen.name,
                    "Q_GEN",
                    q,
                    record_summary=False,
                )
                added_keys.add(key)
            if ("ACGenerator", gen.name) not in measured_devices:
                key = ("ACGenerator", gen.name, "V_GEN")
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_v_{gen.name}",
                    "ACGenerator",
                    gen.name,
                    "V_GEN",
                    voltage,
                    record_summary=False,
                )
                added_keys.add(key)

        for load in sorted(self.load_by_name.values(), key=lambda item: item.idx):
            p, q = self._load_pseudo_power(load)
            voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
            key = ("ACLoad", load.name, "P_LOAD")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_p_{load.name}",
                    "ACLoad",
                    load.name,
                    "P_LOAD",
                    p,
                    record_summary=False,
                )
                added_keys.add(key)
            key = ("ACLoad", load.name, "Q_LOAD")
            if key not in measured_keys and key not in added_keys:
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_q_{load.name}",
                    "ACLoad",
                    load.name,
                    "Q_LOAD",
                    q,
                    record_summary=False,
                )
                added_keys.add(key)
            if ("ACLoad", load.name) not in measured_devices:
                key = ("ACLoad", load.name, "V_LOAD")
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_v_{load.name}",
                    "ACLoad",
                    load.name,
                    "V_LOAD",
                    voltage,
                    record_summary=False,
                )
                added_keys.add(key)
        if added_keys:
            measured_keys.update(added_keys)
            measured_devices.update((device_type, device_name) for device_type, device_name, _meas_type in added_keys)

    def _seed_power_state_arrays_from_measurements(self) -> None:
        """Use the best available P/Q rows as initial values for explicit power states."""
        best = getattr(self, "_real_power_measurement_seed_cache", None)
        if best is None:
            best: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
            for meas in self.measurements:
                if not meas.valid or meas.weight <= 0.0:
                    continue
                if meas.device_type == "ACGenerator" and meas.meas_type in ("P_GEN", "Q_GEN"):
                    if meas.device_name not in self.generator_state_index_by_name:
                        continue
                elif meas.device_type == "ACLoad" and meas.meas_type in ("P_LOAD", "Q_LOAD"):
                    if meas.device_name not in self.load_state_index_by_name:
                        continue
                else:
                    continue
                key = (meas.device_type, meas.device_name, meas.meas_type)
                current = best.get(key)
                if current is None or meas.weight > current[0]:
                    best[key] = (float(meas.weight), float(meas.value))

        for (_device_type, name, meas_type), (_weight, value) in best.items():
            if meas_type == "P_GEN":
                self.initial_gen_p_array[self.generator_state_index_by_name[name]] = value
            elif meas_type == "Q_GEN":
                self.initial_gen_q_array[self.generator_state_index_by_name[name]] = value
            elif meas_type == "P_LOAD":
                self.initial_load_p_array[self.load_state_index_by_name[name]] = value
            elif meas_type == "Q_LOAD":
                self.initial_load_q_array[self.load_state_index_by_name[name]] = value

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch remaining AC rank deficiencies until observable or the configured cap is reached."""
        total_added = 0
        max_count = max(0, int(self.targeted_pseudo_measurement_max))
        observability = None
        while total_added < max_count:
            observability = self.observability_analysis()
            if observability.observable:
                break
            next_idx = self._next_measurement_idx()
            existing_keys = self._active_measurement_keys()
            existing_names = {meas.name for meas in self.measurements}
            added = 0
            remaining = max_count - total_added
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
                break
            total_added += added
            self._refresh_active_measurement_indexes()
            observability = None
        if observability is None and total_added < max_count:
            observability = self.observability_analysis()
        if total_added < max_count and observability is not None and not observability.observable:
            total_added += self._add_structural_rank_restoring_pseudo_measurements(max_count - total_added)
            observability = None
        if observability is not None:
            self._initial_observability_cache = observability
        return total_added

    def _rank_restoring_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidates from invalid real device rows, excluding node V and angles."""
        device_maps = {
            "ACBranch": self.branch_by_name,
            "ACTransformer": self.transformer_by_name,
            "ACSwitch": self.switch_by_name,
            "ACZeroBranch": self.zero_branch_by_name,
            "ACGenerator": self.generator_by_name,
            "ACLoad": self.load_by_name,
        }
        existing_keys = self._active_measurement_keys()
        existing_names = {meas.name for meas in self.measurements}
        seen_keys = set()
        candidates: List[Measurement] = []
        next_idx = self._next_measurement_idx()
        for meas in self.measurements:
            if meas.valid or meas.weight <= 0.0:
                continue
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                continue
            if meas.device_type == "ACNode" and meas.meas_type == "V":
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                continue
            key = (meas.device_type, meas.device_name, meas.meas_type)
            if key in existing_keys or key in seen_keys:
                continue
            pseudo_name = f"pseudo_rank_{meas.name}"
            if pseudo_name in existing_names:
                continue
            seen_keys.add(key)
            candidates.append(
                Measurement(
                    idx=next_idx + len(candidates),
                    name=pseudo_name,
                    device_type=meas.device_type,
                    device_name=meas.device_name,
                    meas_type=meas.meas_type,
                    weight=self.pseudo_measurement_weight,
                    valid=True,
                    value=self._measurement_value_to_internal_units(meas),
                )
            )
        return candidates

    def _rank_restoring_candidate_indices(self, candidates: Sequence[Measurement], max_add: int) -> List[int]:
        """Select candidate rows that participate in a higher structural-rank matching."""
        if max_add <= 0 or not candidates or sp_maximum_bipartite_matching is None:
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
        for local_idx in selected_indices:
            candidate = candidates[int(local_idx)]
            candidate.idx = next_idx
            next_idx += 1
            self.measurements.append(candidate)
        self._refresh_active_measurement_indexes()
        return len(selected_indices)

    def _unanchored_angle_state_labels(self) -> List[str]:
        """Return one AC angle state per structurally unanchored angle component."""
        H = self.jacobian_sparse(self.initial_state())
        return unanchored_angle_state_labels(H, self.state_labels, "theta:")

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_label: str,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        """Translate a weak compact AC state into the smallest useful pseudo measurement."""
        if ":" not in state_label:
            return next_idx, 0
        prefix, name = state_label.split(":", 1)
        added_total = 0

        def add(device_type: str, device_name: str, meas_type: str, value: float) -> Tuple[int, int]:
            nonlocal added_total
            if added_total >= max_add:
                return next_idx, 0
            key = (device_type, device_name, meas_type)
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if key in existing_keys or pseudo_name in existing_names:
                return next_idx, 0
            new_idx = self._append_pseudo_measurement(
                next_idx,
                pseudo_name,
                device_type,
                device_name,
                meas_type,
                value,
            )
            existing_keys.add(key)
            existing_names.add(pseudo_name)
            added_total += 1
            return new_idx, 1

        if prefix == "theta" and name in self.node_by_name:
            return next_idx, 0
        if prefix == "V" and name in self.node_by_name:
            return next_idx, 0
        if prefix in ("I_Z_RE", "I_Z_IM") and name in self.zero_branch_by_name:
            dev = self.zero_branch_by_name[name]
            next_idx, added_p = add("ACZeroBranch", name, "P_FROM", float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q = add("ACZeroBranch", name, "Q_FROM", float(getattr(dev, "q", 0.0) or 0.0))
            next_idx, added_p_to = add("ACZeroBranch", name, "P_TO", -float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q_to = add("ACZeroBranch", name, "Q_TO", -float(getattr(dev, "q", 0.0) or 0.0))
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if prefix in ("I_S_RE", "I_S_IM") and name in self.switch_by_name:
            dev = self.switch_by_name[name]
            next_idx, added_p = add("ACSwitch", name, "P_FROM", float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q = add("ACSwitch", name, "Q_FROM", float(getattr(dev, "q", 0.0) or 0.0))
            next_idx, added_p_to = add("ACSwitch", name, "P_TO", -float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q_to = add("ACSwitch", name, "Q_TO", -float(getattr(dev, "q", 0.0) or 0.0))
            return next_idx, added_p + added_q + added_p_to + added_q_to
        if prefix in ("P_GEN", "Q_GEN") and name in self.generator_by_name:
            p, q = self._generator_pseudo_power(self.generator_by_name[name])
            meas_type = prefix
            return add("ACGenerator", name, meas_type, p if meas_type == "P_GEN" else q)
        if prefix in ("P_LOAD", "Q_LOAD") and name in self.load_by_name:
            p, q = self._load_pseudo_power(self.load_by_name[name])
            meas_type = prefix
            return add("ACLoad", name, meas_type, p if meas_type == "P_LOAD" else q)
        return next_idx, 0

    def _add_zero_branch_constraint_measurements(self) -> None:
        """Inject ideal zero-impedance voltage equality constraints."""
        existing = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in self.measurements
            if meas.valid
            and meas.weight > 0.0
            and meas.device_type in ("ACZeroBranchConstraint", "ACSwitchConstraint")
        }
        next_idx = self._next_measurement_idx()
        weight = 10.0
        ideal_devices = [
            ("ACZeroBranchConstraint", zbr)
            for zbr in sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
        ]
        ideal_devices.extend(
            ("ACSwitchConstraint", sw)
            for sw in sorted(self.switch_by_name.values(), key=lambda item: item.idx)
        )
        for device_type, dev in ideal_devices:
            for meas_type in ("V_DIFF",):
                if (device_type, dev.name, meas_type) in existing:
                    continue
                self.measurements.append(
                    Measurement(
                        idx=next_idx,
                        name=f"constraint_{meas_type.lower()}_{dev.name}",
                        device_type=device_type,
                        device_name=dev.name,
                        meas_type=meas_type,
                        weight=weight,
                        valid=True,
                        value=0.0,
                    )
                )
                next_idx += 1

    def _add_power_balance_constraint_measurements(self) -> None:
        """Add nodal AC power-balance equations that tie P/Q states to the grid."""
        next_idx = self._next_measurement_idx()
        weight = 10.0
        new_measurement = Measurement.__new__
        append = self.measurements.append
        for node in self.nodes:
            for meas_type in ("P_BALANCE", "Q_BALANCE"):
                measurement = new_measurement(Measurement)
                measurement.idx = next_idx
                measurement.name = f"constraint_{meas_type.lower()}_{node.name}"
                measurement.device_type = "ACPowerBalance"
                measurement.device_name = node.name
                measurement.meas_type = meas_type
                measurement.weight = weight
                measurement.valid = True
                measurement.value = 0.0
                append(measurement)
                next_idx += 1
        self._max_measurement_idx = next_idx - 1 if next_idx > 0 else self._max_measurement_idx

    def _node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real ACNode voltage measurements keyed by node index."""
        cached = getattr(self, "_node_voltage_measurement_cache", None)
        if cached is not None:
            return dict(cached)
        best: Dict[int, Tuple[float, float]] = {}
        for meas in self.measurements:
            if (
                not meas.valid
                or meas.weight <= 0.0
                or meas.device_type != "ACNode"
                or meas.meas_type != "V"
                or meas.name.startswith("pseudo_")
                or meas.device_name not in self.node_by_name
            ):
                continue
            node_idx = self.node_by_name[meas.device_name].idx
            current = best.get(node_idx)
            if current is None or meas.weight > current[0]:
                best[node_idx] = (float(meas.weight), float(meas.value))
        return {node_idx: value for node_idx, (_weight, value) in best.items()}

    def _node_incident_degrees(self) -> Dict[int, int]:
        """Count live incident AC branches, transformers, switches and zero branches."""
        degrees = {node.idx: 0 for node in self.nodes}
        device_groups = (
            self.branch_by_name.values(),
            self.transformer_by_name.values(),
            self.zero_branch_by_name.values(),
            self.switch_by_name.values(),
        )
        for devices in device_groups:
            for dev in devices:
                if dev.i_node in degrees:
                    degrees[dev.i_node] += 1
                if dev.j_node in degrees:
                    degrees[dev.j_node] += 1
        return degrees

    def _select_reference_nodes(self):
        """Choose a measured high-degree V/angle reference for every live AC island."""
        references = []
        for island in self.network.islands:
            if not island.is_alive:
                continue
            candidates = [
                node
                for node in island.nodes
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
            elif island.nodes:
                references.append(sorted(island.nodes, key=lambda item: item.idx)[0])
        return references

    def _reference_angle_offsets(self) -> Dict[int, float]:
        """Map each node position to the original island reference angle."""
        ref_by_island = {getattr(node, "isl", None): node for node in self.references}
        offsets: Dict[int, float] = {}
        for island in self.network.islands:
            if not getattr(island, "is_alive", False):
                continue
            ref = ref_by_island.get(getattr(island, "idx", None))
            if ref is None:
                continue
            offset = float(getattr(ref, "angle", 0.0) or 0.0)
            for node in island.nodes:
                if node.idx in self.node_pos:
                    offsets[self.node_pos[node.idx]] = offset
        return offsets

    def _angle_reference_for_node(self, node_idx: int) -> float:
        pos = self.node_pos.get(node_idx)
        if pos is None:
            return 0.0
        return float(self.reference_angle_by_pos.get(pos, 0.0))

    def _node_angle_in_reference_frame(self, node) -> float:
        return float(getattr(node, "angle", 0.0) or 0.0) - self._angle_reference_for_node(node.idx)

    def _rebase_angle_measurements(self) -> None:
        """Convert absolute node angle measurements to the estimator reference frame."""
        if not getattr(self, "_has_valid_angle_measurements", True):
            return
        for meas in self.measurements:
            if (
                meas.valid
                and meas.weight > 0.0
                and meas.device_type == "ACNode"
                and meas.meas_type in ("ANGLE", "THETA")
                and meas.device_name in self.node_by_name
            ):
                node = self.node_by_name[meas.device_name]
                meas.value -= self._angle_reference_for_node(node.idx)

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

        for dev in [*self.zero_branches, *self.switches]:
            if dev.i_node in self.node_pos and dev.j_node in self.node_pos:
                union(self.node_pos[dev.i_node], self.node_pos[dev.j_node])

        groups: Dict[int, List[int]] = {}
        for pos in range(n):
            groups.setdefault(find(pos), []).append(pos)
        components = [sorted(group) for group in groups.values()]
        components.sort(key=lambda group: group[0])
        self.zero_tie_components = components

        self.angle_col = np.full(n, -1, dtype=np.int32)
        self.voltage_col = np.full(n, -1, dtype=np.int32)
        self.angle_state_pos: List[int] = []
        self.angle_state_nodes: List[np.ndarray] = []
        self.voltage_state_pos: List[int] = []
        self.voltage_state_nodes: List[np.ndarray] = []
        self.ref_angles: Dict[int, float] = {}
        self.ref_voltages: Dict[int, float] = {}

        for component in components:
            component_array = np.asarray(component, dtype=np.int32)
            ref_positions = [pos for pos in component if pos in self.ref_idx]
            if ref_positions:
                ref_pos = min(ref_positions, key=lambda pos: self.nodes[pos].idx)
                for pos in component:
                    self.ref_angles[pos] = 0.0
            else:
                col = len(self.angle_state_pos)
                rep_pos = component[0]
                self.angle_state_pos.append(rep_pos)
                self.angle_state_nodes.append(component_array)
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
                self.voltage_state_nodes.append(component_array)
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
        if self.angle_state_nodes:
            self._angle_unpack_nodes = np.concatenate(self.angle_state_nodes).astype(np.int32, copy=False)
            self._angle_unpack_cols = np.concatenate(
                [
                    np.full(nodes.size, col, dtype=np.int32)
                    for col, nodes in enumerate(self.angle_state_nodes)
                ]
            )
        else:
            self._angle_unpack_nodes = np.array([], dtype=np.int32)
            self._angle_unpack_cols = np.array([], dtype=np.int32)
        if self.voltage_state_nodes:
            self._voltage_unpack_nodes = np.concatenate(self.voltage_state_nodes).astype(np.int32, copy=False)
            self._voltage_unpack_cols = np.concatenate(
                [
                    np.full(nodes.size, col, dtype=np.int32)
                    for col, nodes in enumerate(self.voltage_state_nodes)
                ]
            )
        else:
            self._voltage_unpack_nodes = np.array([], dtype=np.int32)
            self._voltage_unpack_cols = np.array([], dtype=np.int32)
        voltage_mask = self.voltage_col >= 0
        self.voltage_col[voltage_mask] = self.n_angle + self.voltage_col[voltage_mask]

    def _build_y_matrix(self) -> np.ndarray:
        """Build the estimator admittance matrix with the same stamps as load flow."""
        n = len(self.nodes)
        if coo_matrix is not None:
            rows = []
            cols = []
            data = []

            for br in self.branch_by_name.values():
                i = self.node_pos[br.i_node]
                j = self.node_pos[br.j_node]
                yff, yft, ytf, ytt = self.branch_stamp_by_name[br.name]
                rows.extend((i, i, j, j))
                cols.extend((i, j, i, j))
                data.extend((yff, yft, ytf, ytt))

            for tr in self.transformer_by_name.values():
                i = self.node_pos[tr.i_node]
                j = self.node_pos[tr.j_node]
                yff, yft, ytf, ytt = self.transformer_stamp_by_name[tr.name]
                rows.extend((i, i, j, j))
                cols.extend((i, j, i, j))
                data.extend((yff, yft, ytf, ytt))

            for sc in self.network.shunt_compensators:
                if not getattr(sc, "is_alive", False):
                    continue
                if sc.node not in self.node_pos:
                    continue
                if sc.control_type in ("B", "Z") or sc.g_set != 0.0:
                    y_sh = complex(sc.g_set, sc.b_set)
                    if y_sh != 0.0:
                        pos = self.node_pos[sc.node]
                        rows.append(pos)
                        cols.append(pos)
                        data.append(y_sh)
            Y = coo_matrix((np.asarray(data, dtype=np.complex128), (np.asarray(rows), np.asarray(cols))), shape=(n, n)).tocsr()
            Y.sum_duplicates()
            return Y

        Y = np.zeros((n, n), dtype=np.complex128)

        for br in self.branch_by_name.values():
            i = self.node_pos[br.i_node]
            j = self.node_pos[br.j_node]
            yff, yft, ytf, ytt = self.branch_stamp_by_name[br.name]
            Y[i, i] += yff
            Y[i, j] += yft
            Y[j, i] += ytf
            Y[j, j] += ytt

        for tr in self.transformer_by_name.values():
            i = self.node_pos[tr.i_node]
            j = self.node_pos[tr.j_node]
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[tr.name]
            Y[i, i] += yff
            Y[i, j] += yft
            Y[j, i] += ytf
            Y[j, j] += ytt

        for sc in self.network.shunt_compensators:
            if not getattr(sc, "is_alive", False):
                continue
            if sc.node not in self.node_pos:
                continue
            if sc.control_type in ("B", "Z") or sc.g_set != 0.0:
                Y[self.node_pos[sc.node], self.node_pos[sc.node]] += complex(sc.g_set, sc.b_set)
        return Y

    def _prepare_y_row_cache(self) -> None:
        """Cache sparse Y-row topology used repeatedly by generator Jacobian rows."""
        n = len(self.nodes)
        self._y_row_nodes = []
        self._y_row_y_conj = []
        self._y_row_off_mask = [None] * n
        self._y_row_off_nodes = [None] * n
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

    def _group_loads(self) -> Dict[int, List]:
        grouped: Dict[int, List] = {}
        for load in self.load_by_name.values():
            grouped.setdefault(self.node_pos[load.node], []).append(load)
        return grouped

    def _group_generators(self) -> Dict[int, List]:
        grouped: Dict[int, List] = {}
        for gen in self.generator_by_name.values():
            grouped.setdefault(self.node_pos[gen.node], []).append(gen)
        return grouped

    def _generator_shares(self) -> Dict[str, float]:
        """Split a node-level solved injection among co-located generators."""
        shares = {}
        for gens in self.generators_at_pos.values():
            alphas = [float(gen.alpha) for gen in gens if getattr(gen, "alpha", None) is not None]
            total_alpha = sum(alphas)
            for gen in gens:
                if total_alpha > 0.0 and getattr(gen, "alpha", None) is not None:
                    shares[gen.name] = float(gen.alpha) / total_alpha
                else:
                    shares[gen.name] = 1.0 / len(gens)
        return shares

    def initial_state(self) -> np.ndarray:
        if self.flat_start:
            theta = np.zeros(len(self.nodes), dtype=np.float64)
            voltage = np.ones(len(self.nodes), dtype=np.float64)
            return self._pack_state(theta, voltage, rebase_angles=False)
        return self._file_state()

    def _file_state(self) -> np.ndarray:
        """Pack the E-file voltage/angle snapshot in the selected reference frame."""
        return self._pack_state(self.file_theta, self.file_voltage, rebase_angles=True)

    def _pack_state(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
        switch_current: Optional[np.ndarray] = None,
        gen_p: Optional[np.ndarray] = None,
        gen_q: Optional[np.ndarray] = None,
        load_p: Optional[np.ndarray] = None,
        load_q: Optional[np.ndarray] = None,
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
            x[self.base_load_q : self.n_state] = (
                self.initial_load_q_array if load_q is None else np.asarray(load_q, dtype=np.float64)
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

    def _load_power(self, voltage: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        p_values, q_values, p_load, q_load = self._load_power_arrays(voltage)
        load_power = {
            load.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, load in enumerate(self.load_order)
        }
        return load_power, p_load, q_load

    def _load_power_arrays(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-load ZIP powers plus node-level load totals."""
        p_load = np.zeros(len(self.nodes), dtype=np.float64)
        q_load = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), p_load, q_load
        vm = voltage[self.load_pos_array]
        p_values = self.load_pv0_array + self.load_pv1_array * vm + self.load_pv2_array * vm * vm
        q_values = self.load_qv0_array + self.load_qv1_array * vm + self.load_qv2_array * vm * vm
        np.add.at(p_load, self.load_pos_array, p_values)
        np.add.at(q_load, self.load_pos_array, q_values)
        return p_values, q_values, p_load, q_load

    def _load_power_totals(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute only node-level load totals for generator balance equations."""
        _, _, p_load, q_load = self._load_power_arrays(voltage)
        return p_load, q_load

    def _load_power_from_state(self, x: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        """Return load P/Q states and node totals used by balance equations."""
        p_values = np.asarray(x[self.base_load_p : self.base_load_q], dtype=np.float64)
        q_values = np.asarray(x[self.base_load_q : self.n_state], dtype=np.float64)
        p_load = np.zeros(len(self.nodes), dtype=np.float64)
        q_load = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size:
            p_load = np.bincount(self.load_pos_array, weights=p_values, minlength=self.n_nodes)
            q_load = np.bincount(self.load_pos_array, weights=q_values, minlength=self.n_nodes)
        load_power = {
            load.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, load in enumerate(self.load_order)
        }
        return load_power, p_load, q_load

    def _load_power_totals_from_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level load totals from explicit P_LOAD/Q_LOAD states."""
        if not self.load_pos_array.size:
            return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(self.n_nodes, dtype=np.float64)
        return (
            np.bincount(self.load_pos_array, weights=x[self.base_load_p : self.base_load_q], minlength=self.n_nodes),
            np.bincount(self.load_pos_array, weights=x[self.base_load_q : self.n_state], minlength=self.n_nodes),
        )

    def _generator_power_from_state(self, x: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
        """Return generator P/Q states and node totals used by balance equations."""
        p_values = np.asarray(x[self.base_gen_p : self.base_gen_q], dtype=np.float64)
        q_values = np.asarray(x[self.base_gen_q : self.base_load_p], dtype=np.float64)
        p_gen = np.zeros(len(self.nodes), dtype=np.float64)
        q_gen = np.zeros(len(self.nodes), dtype=np.float64)
        if self.generator_pos_array.size:
            p_gen = np.bincount(self.generator_pos_array, weights=p_values, minlength=self.n_nodes)
            q_gen = np.bincount(self.generator_pos_array, weights=q_values, minlength=self.n_nodes)
        gen_power = {
            gen.name: (float(p_values[idx]), float(q_values[idx]))
            for idx, gen in enumerate(self.generator_order)
        }
        return gen_power, p_gen, q_gen

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
        """Evaluate nodal mismatch S_net + S_switch + S_load - S_gen."""
        s_network = voltage_complex * np.conj(self.Y.dot(voltage_complex))
        p_switch, q_switch = self._switch_power_injections(voltage_complex, switch_current)
        p_load, q_load = self._load_power_totals_from_state(x)
        p_gen, q_gen = self._generator_power_totals_from_state(x)
        return (
            s_network.real + p_switch + p_load - p_gen,
            s_network.imag + q_switch + q_load - q_gen,
        )

    def _generator_power(
        self,
        voltage_complex: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        switch_current: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """Infer generator injections from network balance and local load demand."""
        p_gen_total, q_gen_total = self._generator_power_totals(
            voltage_complex,
            p_load,
            q_load,
            switch_current,
        )
        gen_power = {}
        for pos, gens in self.generators_at_pos.items():
            for gen in gens:
                share = self.generator_share_by_name[gen.name]
                gen_power[gen.name] = (share * p_gen_total[pos], share * q_gen_total[pos])
        return gen_power

    def _generator_power_totals(
        self,
        voltage_complex: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer node-level generator injections from network, load and zero-current balances."""
        s_network = voltage_complex * np.conj(self.Y.dot(voltage_complex))
        if self.n_switch_current:
            p_switch = np.zeros(len(self.nodes), dtype=np.float64)
            q_switch = np.zeros(len(self.nodes), dtype=np.float64)
            s_from = voltage_complex[self.zero_current_i] * np.conj(switch_current)
            s_to = voltage_complex[self.zero_current_j] * np.conj(-switch_current)
            np.add.at(p_switch, self.zero_current_i, s_from.real)
            np.add.at(q_switch, self.zero_current_i, s_from.imag)
            np.add.at(p_switch, self.zero_current_j, s_to.real)
            np.add.at(q_switch, self.zero_current_j, s_to.imag)
        else:
            p_switch = q_switch = 0.0
        p_gen_total = s_network.real + p_load
        q_gen_total = s_network.imag + q_load
        p_gen_total = p_gen_total + p_switch
        q_gen_total = q_gen_total + q_switch
        return p_gen_total, q_gen_total

    def _branch_power(self, device, is_transformer: bool, voltage_complex: np.ndarray) -> Tuple[complex, complex]:
        i = self.node_pos[device.i_node]
        j = self.node_pos[device.j_node]
        if is_transformer:
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[device.name]
        else:
            yff, yft, ytf, ytt = self.branch_stamp_by_name[device.name]
        vi = voltage_complex[i]
        vj = voltage_complex[j]
        sij = vi * np.conj(yff * vi + yft * vj)
        sji = vj * np.conj(ytf * vi + ytt * vj)
        return sij, sji

    def _branch_current(self, device, is_transformer: bool, voltage_complex: np.ndarray) -> Tuple[complex, complex]:
        i = self.node_pos[device.i_node]
        j = self.node_pos[device.j_node]
        if is_transformer:
            yff, yft, ytf, ytt = self.transformer_stamp_by_name[device.name]
        else:
            yff, yft, ytf, ytt = self.branch_stamp_by_name[device.name]
        vi = voltage_complex[i]
        vj = voltage_complex[j]
        return yff * vi + yft * vj, ytf * vi + ytt * vj

    def _power_current(self, p: float, q: float, voltage: float) -> float:
        if abs(voltage) <= self.min_current_voltage:
            return 0.0
        return float(np.hypot(p, q) / voltage)

    def _calc_quantities(self, x: np.ndarray):
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        load_power, _, _ = self._load_power_from_state(x)
        gen_power, _, _ = self._generator_power_from_state(x)
        return theta, voltage, voltage_complex, load_power, gen_power, switch_current

    @staticmethod
    def _int_array(values: Sequence[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _bool_array(values: Sequence[bool]) -> np.ndarray:
        return np.asarray(values, dtype=bool)

    @staticmethod
    def _complex_array(values: Sequence[complex]) -> np.ndarray:
        return np.asarray(values, dtype=np.complex128)

    @staticmethod
    def _build_branch_stamp_map(devices: Sequence[object], with_tap: bool) -> Dict[str, Tuple[complex, complex, complex, complex]]:
        """Build MATPOWER branch stamps in one vectorized pass, then attach them by device name."""
        if not devices:
            return {}
        if with_tap:
            yff, yft, ytf, ytt = matpower_branch_stamp_vectorized(
                [dev.r for dev in devices],
                [dev.x for dev in devices],
                [dev.b for dev in devices],
                [dev.tap for dev in devices],
                [dev.shift for dev in devices],
            )
        else:
            yff, yft, ytf, ytt = matpower_branch_stamp_vectorized(
                [dev.r for dev in devices],
                [dev.x for dev in devices],
                [dev.b for dev in devices],
            )
        return {
            dev.name: (complex(yff[idx]), complex(yft[idx]), complex(ytf[idx]), complex(ytt[idx]))
            for idx, dev in enumerate(devices)
        }

    def _branch_transformer_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_branch_transformer_vector_plan", None)
            if active_plan is not None:
                return active_plan
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
        handled_mask = np.zeros(len(measurements), dtype=bool)
        voltage_rows = []
        voltage_pos = []
        voltage_cols = []
        power_rows = []
        power_is_p = []
        power_own = []
        power_other = []
        power_y_self = []
        power_y_mutual = []
        current_rows = []
        current_own = []
        current_other = []
        current_y_self = []
        current_y_mutual = []

        for row, meas in self._measurement_rows_for_types(measurements, ("ACBranch", "ACTransformer")):
            if meas.device_type == "ACBranch":
                device = self.branch_by_name[meas.device_name]
                yff, yft, ytf, ytt = self.branch_stamp_by_name[device.name]
            elif meas.device_type == "ACTransformer":
                device = self.transformer_by_name[meas.device_name]
                yff, yft, ytf, ytt = self.transformer_stamp_by_name[device.name]
            else:
                continue

            i = self.node_pos[device.i_node]
            j = self.node_pos[device.j_node]
            mtype = meas.meas_type
            if mtype == "V_FROM":
                voltage_rows.append(row)
                voltage_pos.append(i)
                voltage_cols.append(self.voltage_col[i])
            elif mtype == "V_TO":
                voltage_rows.append(row)
                voltage_pos.append(j)
                voltage_cols.append(self.voltage_col[j])
            elif mtype in ("P_FROM", "Q_FROM"):
                power_rows.append(row)
                power_is_p.append(mtype == "P_FROM")
                power_own.append(i)
                power_other.append(j)
                power_y_self.append(yff)
                power_y_mutual.append(yft)
            elif mtype in ("P_TO", "Q_TO"):
                power_rows.append(row)
                power_is_p.append(mtype == "P_TO")
                power_own.append(j)
                power_other.append(i)
                power_y_self.append(ytt)
                power_y_mutual.append(ytf)
            elif mtype == "I_FROM":
                current_rows.append(row)
                current_own.append(i)
                current_other.append(j)
                current_y_self.append(yff)
                current_y_mutual.append(yft)
            elif mtype == "I_TO":
                current_rows.append(row)
                current_own.append(j)
                current_other.append(i)
                current_y_self.append(ytt)
                current_y_mutual.append(ytf)
            else:
                raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")
            handled_mask[row] = True

        return {
            "handled_mask": handled_mask,
            "voltage_rows": self._int_array(voltage_rows),
            "voltage_pos": self._int_array(voltage_pos),
            "voltage_cols": self._int_array(voltage_cols),
            "power_rows": self._int_array(power_rows),
            "power_is_p": self._bool_array(power_is_p),
            "power_own": self._int_array(power_own),
            "power_other": self._int_array(power_other),
            "power_y_self": self._complex_array(power_y_self),
            "power_y_mutual": self._complex_array(power_y_mutual),
            "current_rows": self._int_array(current_rows),
            "current_own": self._int_array(current_own),
            "current_other": self._int_array(current_other),
            "current_y_self": self._complex_array(current_y_self),
            "current_y_mutual": self._complex_array(current_y_mutual),
        }

    def _fill_branch_transformer_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACBranch/ACTransformer terminal P/Q/V/I measurements."""
        plan = self._branch_transformer_vector_plan(measurements)
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

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        measurements = self._normalize_measurements(measurements)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        load_power = gen_power = None
        branch_flow_cache = {}
        values = np.zeros(len(measurements), dtype=np.float64)
        vectorized_branch_rows = self._fill_branch_transformer_values_vectorized(
            values,
            measurements,
            voltage,
            voltage_complex,
        )
        vectorized_zero_rows = self._fill_zero_current_values_vectorized(
            values,
            measurements,
            theta,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_simple_rows = self._fill_simple_values_vectorized(
            values,
            measurements,
            x,
            theta,
            voltage,
        )
        vectorized_generator_rows = self._fill_generator_values_vectorized(
            values,
            measurements,
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_values_vectorized(
            values,
            measurements,
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
        if (measurements is self.active_measurements and self.active_measurements_are_vectorized) or np.all(vectorized_rows):
            return values

        for row, meas in enumerate(measurements):
            if vectorized_rows[row]:
                continue
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                node = self.node_by_name[meas.device_name]
                pos = self.node_pos[node.idx]
                if mtype == "V":
                    values[row] = voltage[pos]
                elif mtype in ("ANGLE", "THETA"):
                    values[row] = theta[pos]
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")
            elif meas.device_type == "ACBranch":
                br = self.branch_by_name[meas.device_name]
                if mtype == "V_FROM":
                    values[row] = voltage[self.node_pos[br.i_node]]
                elif mtype == "V_TO":
                    values[row] = voltage[self.node_pos[br.j_node]]
                else:
                    cache_key = ("ACBranch", br.name)
                    if cache_key not in branch_flow_cache:
                        sij, sji = self._branch_power(br, False, voltage_complex)
                        iij, iji = self._branch_current(br, False, voltage_complex)
                        branch_flow_cache[cache_key] = (sij, sji, iij, iji)
                    sij, sji, iij, iji = branch_flow_cache[cache_key]
                    values[row] = self._flow_value(mtype, sij, sji, iij, iji)
            elif meas.device_type == "ACTransformer":
                tr = self.transformer_by_name[meas.device_name]
                if mtype == "V_FROM":
                    values[row] = voltage[self.node_pos[tr.i_node]]
                elif mtype == "V_TO":
                    values[row] = voltage[self.node_pos[tr.j_node]]
                else:
                    cache_key = ("ACTransformer", tr.name)
                    if cache_key not in branch_flow_cache:
                        sij, sji = self._branch_power(tr, True, voltage_complex)
                        iij, iji = self._branch_current(tr, True, voltage_complex)
                        branch_flow_cache[cache_key] = (sij, sji, iij, iji)
                    sij, sji, iij, iji = branch_flow_cache[cache_key]
                    values[row] = self._flow_value(mtype, sij, sji, iij, iji)
            elif meas.device_type == "ACGenerator":
                gen = self.generator_by_name[meas.device_name]
                node = self.node_by_name[gen.node_obj.name]
                if gen_power is None:
                    gen_power, _, _ = self._generator_power_from_state(x)
                p, q = gen_power[gen.name]
                if mtype == "P_GEN":
                    values[row] = p
                elif mtype == "Q_GEN":
                    values[row] = q
                elif mtype == "V_GEN":
                    values[row] = voltage[self.node_pos[node.idx]]
                elif mtype == "I_GEN":
                    values[row] = self._power_current(p, q, voltage[self.node_pos[node.idx]])
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")
            elif meas.device_type == "ACLoad":
                load = self.load_by_name[meas.device_name]
                node = self.node_by_name[load.node_obj.name]
                if load_power is None:
                    load_power, _, _ = self._load_power_from_state(x)
                p, q = load_power[load.name]
                if mtype == "P_LOAD":
                    values[row] = p
                elif mtype == "Q_LOAD":
                    values[row] = q
                elif mtype == "V_LOAD":
                    values[row] = voltage[self.node_pos[node.idx]]
                elif mtype == "I_LOAD":
                    values[row] = self._power_current(p, q, voltage[self.node_pos[node.idx]])
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")
            elif meas.device_type == "ACZeroBranch":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[zbr.i_node]] - voltage[self.node_pos[zbr.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[zbr.i_node]] - theta[self.node_pos[zbr.j_node]]
                else:
                    current = switch_current[self.zero_branch_pos[zbr.name]]
                    values[row] = self._zero_current_measurement_value(zbr, current, mtype, voltage, voltage_complex)
            elif meas.device_type == "ACZeroBranchConstraint":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[zbr.i_node]] - voltage[self.node_pos[zbr.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[zbr.i_node]] - theta[self.node_pos[zbr.j_node]]
                else:
                    raise RuntimeError(f"Unsupported ACZeroBranchConstraint measurement type: {mtype}")
            elif meas.device_type == "ACSwitchConstraint":
                sw = self.switch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    values[row] = voltage[self.node_pos[sw.i_node]] - voltage[self.node_pos[sw.j_node]]
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    values[row] = theta[self.node_pos[sw.i_node]] - theta[self.node_pos[sw.j_node]]
                else:
                    raise RuntimeError(f"Unsupported ACSwitchConstraint measurement type: {mtype}")
            elif meas.device_type == "ACSwitch":
                sw = self.switch_by_name[meas.device_name]
                current = switch_current[self.switch_pos[sw.name]]
                values[row] = self._zero_current_measurement_value(sw, current, mtype, voltage, voltage_complex)
            elif meas.device_type == "ACPowerBalance":
                pos = self.node_pos[self.node_by_name[meas.device_name].idx]
                p_balance, q_balance = self._power_balance_totals(x, voltage_complex, switch_current)
                if mtype == "P_BALANCE":
                    values[row] = p_balance[pos]
                elif mtype == "Q_BALANCE":
                    values[row] = q_balance[pos]
                else:
                    raise RuntimeError(f"Unsupported ACPowerBalance measurement type: {mtype}")
            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")
        return values

    @staticmethod
    def _flow_value(meas_type: str, s_from: complex, s_to: complex, i_from: complex = 0.0, i_to: complex = 0.0) -> float:
        if meas_type == "P_FROM":
            return float(s_from.real)
        if meas_type == "Q_FROM":
            return float(s_from.imag)
        if meas_type == "P_TO":
            return float(s_to.real)
        if meas_type == "Q_TO":
            return float(s_to.imag)
        if meas_type == "I_FROM":
            return float(abs(i_from))
        if meas_type == "I_TO":
            return float(abs(i_to))
        raise RuntimeError(f"Unsupported flow measurement type: {meas_type}")

    def _zero_current_measurement_value(
        self,
        device,
        current: complex,
        meas_type: str,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> float:
        if meas_type.endswith("_FROM"):
            pos = self.node_pos[device.i_node]
            signed_current = current
        elif meas_type.endswith("_TO"):
            pos = self.node_pos[device.j_node]
            signed_current = -current
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

        if meas_type.startswith("P") or meas_type.startswith("Q"):
            power = voltage_complex[pos] * np.conj(signed_current)
            return float(power.real if meas_type.startswith("P") else power.imag)
        if meas_type.startswith("I"):
            return float(abs(current))
        if meas_type.startswith("V"):
            return float(voltage[pos])
        raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

    def _zero_current_vector_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_zero_current_vector_plan", None)
            if active_plan is not None:
                return active_plan
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
        handled_mask = np.zeros(len(measurements), dtype=bool)
        scalar_rows, scalar_cols, scalar_values = [], [], []
        voltage_rows, voltage_pos = [], []
        angle_diff_rows, angle_diff_i, angle_diff_j = [], [], []
        voltage_diff_rows, voltage_diff_i, voltage_diff_j = [], [], []
        power_rows, power_is_p, power_pos, power_current_idx, power_sign = [], [], [], [], []
        current_rows, current_idx = [], []

        def add_scalar(row: int, col: int, value: float) -> None:
            scalar_rows.append(row)
            scalar_cols.append(col)
            scalar_values.append(value)

        for row, meas in self._measurement_rows_for_types(measurements, ("ACZeroBranch", "ACSwitch")):
            if meas.device_type == "ACZeroBranch":
                device = self.zero_branch_by_name[meas.device_name]
                current_position = self.zero_branch_pos[device.name]
            elif meas.device_type == "ACSwitch":
                device = self.switch_by_name[meas.device_name]
                current_position = self.switch_pos[device.name]
            else:
                continue

            mtype = meas.meas_type
            if meas.device_type == "ACZeroBranch" and mtype == "V_DIFF":
                i = self.node_pos[device.i_node]
                j = self.node_pos[device.j_node]
                add_scalar(row, int(self.voltage_col[i]), 1.0)
                add_scalar(row, int(self.voltage_col[j]), -1.0)
                voltage_diff_rows.append(row)
                voltage_diff_i.append(i)
                voltage_diff_j.append(j)
                handled_mask[row] = True
                continue
            if meas.device_type == "ACZeroBranch" and mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                i = self.node_pos[device.i_node]
                j = self.node_pos[device.j_node]
                add_scalar(row, int(self.angle_col[i]), 1.0)
                add_scalar(row, int(self.angle_col[j]), -1.0)
                angle_diff_rows.append(row)
                angle_diff_i.append(i)
                angle_diff_j.append(j)
                handled_mask[row] = True
                continue

            if mtype.endswith("_FROM"):
                pos = self.node_pos[device.i_node]
                sign = 1.0
            elif mtype.endswith("_TO"):
                pos = self.node_pos[device.j_node]
                sign = -1.0
            else:
                raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")

            if mtype.startswith("P") or mtype.startswith("Q"):
                power_rows.append(row)
                power_is_p.append(mtype.startswith("P"))
                power_pos.append(pos)
                power_current_idx.append(current_position)
                power_sign.append(sign)
            elif mtype.startswith("I"):
                current_rows.append(row)
                current_idx.append(current_position)
            elif mtype.startswith("V"):
                voltage_rows.append(row)
                voltage_pos.append(pos)
            else:
                raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")
            handled_mask[row] = True

        return {
            "handled_mask": handled_mask,
            "scalar_rows": self._int_array(scalar_rows),
            "scalar_cols": self._int_array(scalar_cols),
            "scalar_values": np.asarray(scalar_values, dtype=np.float64),
            "voltage_rows": self._int_array(voltage_rows),
            "voltage_pos": self._int_array(voltage_pos),
            "angle_diff_rows": self._int_array(angle_diff_rows),
            "angle_diff_i": self._int_array(angle_diff_i),
            "angle_diff_j": self._int_array(angle_diff_j),
            "voltage_diff_rows": self._int_array(voltage_diff_rows),
            "voltage_diff_i": self._int_array(voltage_diff_i),
            "voltage_diff_j": self._int_array(voltage_diff_j),
            "power_rows": self._int_array(power_rows),
            "power_is_p": self._bool_array(power_is_p),
            "power_pos": self._int_array(power_pos),
            "power_current_idx": self._int_array(power_current_idx),
            "power_sign": np.asarray(power_sign, dtype=np.float64),
            "current_rows": self._int_array(current_rows),
            "current_idx": self._int_array(current_idx),
        }

    def _fill_zero_current_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACSwitch/ACZeroBranch P/Q/V/I measurements."""
        plan = self._zero_current_vector_plan(measurements)

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
        measurements: Sequence[Measurement],
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACSwitch/ACZeroBranch Jacobian rows from explicit current states."""
        plan = self._zero_current_vector_plan(measurements)

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

    def _branch_power_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex]:
        """Analytical derivatives for S_own = V_own * conj(y_self*V_own + y_mutual*V_other)."""
        angle = theta[own] - theta[other]
        exp_angle = np.exp(1j * angle)
        y_self_conj = np.conj(y_self)
        y_mutual_conj = np.conj(y_mutual)
        off = y_mutual_conj * voltage[own] * voltage[other] * exp_angle
        dtheta_own = 1j * off
        dtheta_other = -1j * off
        dvoltage_own = 2.0 * y_self_conj * voltage[own] + y_mutual_conj * voltage[other] * exp_angle
        dvoltage_other = y_mutual_conj * voltage[own] * exp_angle
        return dtheta_own, dtheta_other, dvoltage_own, dvoltage_other

    def _branch_current_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex, complex]:
        """Analytical derivatives of branch terminal current in polar-voltage variables."""
        v_own = voltage[own] * np.exp(1j * theta[own])
        v_other = voltage[other] * np.exp(1j * theta[other])
        current = y_self * v_own + y_mutual * v_other
        dtheta_own = 1j * y_self * v_own
        dtheta_other = 1j * y_mutual * v_other
        dvoltage_own = y_self * np.exp(1j * theta[own])
        dvoltage_other = y_mutual * np.exp(1j * theta[other])
        return current, dtheta_own, dtheta_other, dvoltage_own, dvoltage_other

    def _add_current_magnitude_derivatives(
        self,
        H: np.ndarray,
        row: int,
        own: int,
        other: int,
        current: complex,
        dtheta_own: complex,
        dtheta_other: complex,
        dvoltage_own: complex,
        dvoltage_other: complex,
    ) -> None:
        current_abs = abs(current)
        if current_abs <= 1e-12:
            return

        def pick(value: complex) -> float:
            # d|I| = real(conj(I) * dI) / |I| projects complex sensitivities to magnitude.
            return float(np.real(np.conj(current) * value) / current_abs)

        own_angle_col = self.angle_col[own]
        if own_angle_col >= 0:
            self._add_scalar_value(H, row, own_angle_col, pick(dtheta_own))
        other_angle_col = self.angle_col[other]
        if other_angle_col >= 0:
            self._add_scalar_value(H, row, other_angle_col, pick(dtheta_other))
        self._add_scalar_value(H, row, self.voltage_col[own], pick(dvoltage_own))
        self._add_scalar_value(H, row, self.voltage_col[other], pick(dvoltage_other))

    def _add_power_derivatives(
        self,
        H: np.ndarray,
        row: int,
        meas_type: str,
        own: int,
        other: int,
        dtheta_own: complex,
        dtheta_other: complex,
        dvoltage_own: complex,
        dvoltage_other: complex,
    ) -> None:
        if meas_type.startswith("P"):
            pick = np.real
        elif meas_type.startswith("Q"):
            pick = np.imag
        else:
            raise RuntimeError(f"Unsupported power measurement type: {meas_type}")

        own_angle_col = self.angle_col[own]
        if own_angle_col >= 0:
            self._add_scalar_value(H, row, own_angle_col, float(pick(dtheta_own)))
        other_angle_col = self.angle_col[other]
        if other_angle_col >= 0:
            self._add_scalar_value(H, row, other_angle_col, float(pick(dtheta_other)))
        self._add_scalar_value(H, row, self.voltage_col[own], float(pick(dvoltage_own)))
        self._add_scalar_value(H, row, self.voltage_col[other], float(pick(dvoltage_other)))

    def _add_indexed_values(
        self,
        H: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if hasattr(H, "add_many"):
            row_col_mask = (rows >= 0) & (cols >= 0)
            mask = row_col_mask if mask is None else (np.asarray(mask, dtype=bool) & row_col_mask)
            H.add_many(rows, cols, values, mask)
            return
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

    @staticmethod
    def _add_scalar_value(H, row: int, col: int, value: float) -> None:
        if col < 0:
            return
        if hasattr(H, "add"):
            H.add(row, col, value)
        else:
            H[row, col] += value

    def _fill_branch_transformer_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill ACBranch/ACTransformer P/Q/V/I rows to reduce Python call overhead."""
        plan = self._branch_transformer_vector_plan(measurements)
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
                H.add_many(
                    np.concatenate((rows, rows, rows, rows)),
                    np.concatenate((own_angle_cols, other_angle_cols, own_voltage_cols, other_voltage_cols)),
                    np.concatenate(values),
                )
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
                H.add_many(
                    np.concatenate((rows, rows, rows, rows)),
                    np.concatenate((own_angle_cols, other_angle_cols, own_voltage_cols, other_voltage_cols)),
                    np.concatenate(values),
                )
            else:
                self._add_indexed_values(H, rows, own_angle_cols, values[0])
                self._add_indexed_values(H, rows, other_angle_cols, values[1])
                self._add_indexed_values(H, rows, own_voltage_cols, values[2])
                self._add_indexed_values(H, rows, other_voltage_cols, values[3])

        return plan["handled_mask"]

    def _simple_jacobian_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_simple_jacobian_plan", None)
            if active_plan is not None:
                return active_plan
        key = id(measurements)
        cached = self._simple_jacobian_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        handled = np.zeros(len(measurements), dtype=bool)
        scalar_rows, scalar_cols, scalar_values = [], [], []
        value_voltage_rows, value_voltage_pos = [], []
        value_angle_rows, value_angle_pos = [], []
        value_voltage_diff_rows, value_voltage_diff_i, value_voltage_diff_j = [], [], []
        value_angle_diff_rows, value_angle_diff_i, value_angle_diff_j = [], [], []
        load_rows, load_pos, load_index, load_kind = [], [], [], []

        def add_scalar(row: int, col: int, value: float) -> None:
            scalar_rows.append(row)
            scalar_cols.append(col)
            scalar_values.append(value)

        for row, meas in self._measurement_rows_for_types(
            measurements,
            ("ACNode", "ACGenerator", "ACLoad", "ACZeroBranchConstraint", "ACSwitchConstraint"),
        ):
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                node = self.node_by_name[meas.device_name]
                pos = self.node_pos[node.idx]
                if mtype == "V":
                    add_scalar(row, int(self.voltage_col[pos]), 1.0)
                    value_voltage_rows.append(row)
                    value_voltage_pos.append(pos)
                    handled[row] = True
                elif mtype in ("ANGLE", "THETA"):
                    add_scalar(row, int(self.angle_col[pos]), 1.0)
                    value_angle_rows.append(row)
                    value_angle_pos.append(pos)
                    handled[row] = True

            elif meas.device_type == "ACGenerator" and mtype == "V_GEN":
                gen = self.generator_by_name[meas.device_name]
                pos = self.node_pos[gen.node]
                add_scalar(row, int(self.voltage_col[pos]), 1.0)
                value_voltage_rows.append(row)
                value_voltage_pos.append(pos)
                handled[row] = True

            elif meas.device_type == "ACLoad":
                load = self.load_by_name[meas.device_name]
                pos = self.node_pos[load.node]
                if mtype == "V_LOAD":
                    add_scalar(row, int(self.voltage_col[pos]), 1.0)
                    value_voltage_rows.append(row)
                    value_voltage_pos.append(pos)
                    handled[row] = True
                elif mtype in ("P_LOAD", "Q_LOAD", "I_LOAD"):
                    load_rows.append(row)
                    load_pos.append(pos)
                    load_index.append(self.load_state_index_by_name[load.name])
                    load_kind.append({"P_LOAD": 0, "Q_LOAD": 1, "I_LOAD": 2}[mtype])
                    handled[row] = True

            elif meas.device_type in ("ACZeroBranchConstraint", "ACSwitchConstraint"):
                device = (
                    self.zero_branch_by_name[meas.device_name]
                    if meas.device_type == "ACZeroBranchConstraint"
                    else self.switch_by_name[meas.device_name]
                )
                i = self.node_pos[device.i_node]
                j = self.node_pos[device.j_node]
                if mtype == "V_DIFF":
                    add_scalar(row, int(self.voltage_col[i]), 1.0)
                    add_scalar(row, int(self.voltage_col[j]), -1.0)
                    value_voltage_diff_rows.append(row)
                    value_voltage_diff_i.append(i)
                    value_voltage_diff_j.append(j)
                    handled[row] = True
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    add_scalar(row, int(self.angle_col[i]), 1.0)
                    add_scalar(row, int(self.angle_col[j]), -1.0)
                    value_angle_diff_rows.append(row)
                    value_angle_diff_i.append(i)
                    value_angle_diff_j.append(j)
                    handled[row] = True

        plan = {
            "handled_mask": handled,
            "scalar_rows": self._int_array(scalar_rows),
            "scalar_cols": self._int_array(scalar_cols),
            "scalar_values": np.asarray(scalar_values, dtype=np.float64),
            "value_voltage_rows": self._int_array(value_voltage_rows),
            "value_voltage_pos": self._int_array(value_voltage_pos),
            "value_angle_rows": self._int_array(value_angle_rows),
            "value_angle_pos": self._int_array(value_angle_pos),
            "value_voltage_diff_rows": self._int_array(value_voltage_diff_rows),
            "value_voltage_diff_i": self._int_array(value_voltage_diff_i),
            "value_voltage_diff_j": self._int_array(value_voltage_diff_j),
            "value_angle_diff_rows": self._int_array(value_angle_diff_rows),
            "value_angle_diff_i": self._int_array(value_angle_diff_i),
            "value_angle_diff_j": self._int_array(value_angle_diff_j),
            "load_rows": self._int_array(load_rows),
            "load_pos": self._int_array(load_pos),
            "load_index": self._int_array(load_index),
            "load_kind": self._int_array(load_kind),
        }
        if len(self._simple_jacobian_plan_cache) > 16:
            self._simple_jacobian_plan_cache.clear()
        self._simple_jacobian_plan_cache[key] = (measurements, plan)
        return plan

    def _fill_simple_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate node, load, generator-voltage and pseudo-constraint rows."""
        plan = self._simple_jacobian_plan(measurements)

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
            p_mask = kind == 0
            q_mask = kind == 1
            i_mask = kind == 2
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
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill node/load/generator-voltage and pseudo constraint Jacobian rows."""
        plan = self._simple_jacobian_plan(measurements)
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
            p_mask = kind == 0
            q_mask = kind == 1
            i_mask = kind == 2
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

    def _balance_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_balance_measurement_plan", None)
            if active_plan is not None:
                return active_plan
        key = id(measurements)
        cached = self._balance_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        handled = np.zeros(len(measurements), dtype=bool)
        rows = []
        pos = []
        kind = []
        p_row_by_pos = np.full(len(self.nodes), -1, dtype=np.int32)
        q_row_by_pos = np.full(len(self.nodes), -1, dtype=np.int32)
        for row, meas in self._measurement_rows_for_types(measurements, ("ACPowerBalance",)):
            if meas.device_type != "ACPowerBalance":
                continue
            if meas.device_name not in self.node_by_name:
                raise RuntimeError(f"Unknown ACPowerBalance node: {meas.device_name}")
            if meas.meas_type not in ("P_BALANCE", "Q_BALANCE"):
                raise RuntimeError(f"Unsupported ACPowerBalance measurement type: {meas.meas_type}")
            node_pos = self.node_pos[self.node_by_name[meas.device_name].idx]
            rows.append(row)
            pos.append(node_pos)
            is_q = meas.meas_type == "Q_BALANCE"
            kind.append(1 if is_q else 0)
            if is_q:
                q_row_by_pos[node_pos] = row
            else:
                p_row_by_pos[node_pos] = row
            handled[row] = True

        y_balance = []
        y_nodes = []
        y_conj = []
        for local_idx, node_pos in enumerate(pos):
            nodes = self._y_row_nodes[int(node_pos)]
            if nodes.size == 0:
                continue
            y_balance.append(np.full(nodes.size, local_idx, dtype=np.int32))
            y_nodes.append(nodes.astype(np.int32, copy=False))
            y_conj.append(self._y_row_y_conj[int(node_pos)].astype(np.complex128, copy=False))
        plan = {
            "handled_mask": handled,
            "rows": self._int_array(rows),
            "pos": self._int_array(pos),
            "pos_i64": np.asarray(pos, dtype=np.int64),
            "kind": self._int_array(kind),
            "p_row_by_pos": p_row_by_pos,
            "q_row_by_pos": q_row_by_pos,
            "y_balance": np.concatenate(y_balance) if y_balance else np.array([], dtype=np.int32),
            "y_nodes": np.concatenate(y_nodes) if y_nodes else np.array([], dtype=np.int32),
            "y_nodes_i64": (
                np.concatenate(y_nodes).astype(np.int64, copy=False) if y_nodes else np.array([], dtype=np.int64)
            ),
            "y_conj": np.concatenate(y_conj) if y_conj else np.array([], dtype=np.complex128),
        }
        if len(self._balance_measurement_plan_cache) > 16:
            self._balance_measurement_plan_cache.clear()
        self._balance_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _fill_balance_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate nodal P/Q power-balance mismatch rows."""
        plan = self._balance_measurement_plan(measurements)
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]
        p_balance, q_balance = self._power_balance_totals(x, voltage_complex, switch_current)
        pos = plan["pos"]
        values[rows] = np.where(plan["kind"] == 0, p_balance[pos], q_balance[pos])
        return plan["handled_mask"]

    def _fill_balance_jacobian_vectorized(
        self,
        H: np.ndarray,
        measurements: Sequence[Measurement],
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse nodal balance derivatives for network, switch and P/Q states."""
        plan = self._balance_measurement_plan(measurements)
        rows = plan["rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        pos = plan["pos_i64"]
        kind = plan["kind"]

        y_balance = plan["y_balance"]
        if y_balance.size:
            y_nodes = plan["y_nodes_i64"]
            y_conj = plan["y_conj"]
            y_pos = pos[y_balance]
            exp_delta = np.exp(1j * (theta[y_pos] - theta[y_nodes]))
            term = y_conj * voltage[y_pos] * voltage[y_nodes] * exp_delta

            off_mask = y_nodes != y_pos
            off_sum = np.zeros(pos.size, dtype=np.complex128)
            if np.any(off_mask):
                off_balance = y_balance[off_mask]
                off_nodes = y_nodes[off_mask]
                off_term = term[off_mask]
                off_sum = self._complex_bincount(off_balance, off_term, pos.size)

                theta_values = -1j * off_term
                theta_rows = rows[off_balance]
                theta_cols = self.angle_col[off_nodes]
                theta_kind = kind[off_balance]

                voltage_values = y_conj[off_mask] * voltage[y_pos[off_mask]] * exp_delta[off_mask]
                voltage_cols = self.voltage_col[off_nodes]
                self._add_indexed_value_blocks(
                    H,
                    (
                        (
                            theta_rows,
                            theta_cols,
                            np.where(theta_kind == 0, theta_values.real, theta_values.imag),
                        ),
                        (
                            theta_rows,
                            voltage_cols,
                            np.where(theta_kind == 0, voltage_values.real, voltage_values.imag),
                        ),
                    ),
                )

            sum_all = self._complex_bincount(y_balance, y_conj * voltage[y_nodes] * exp_delta, pos.size)
        else:
            off_sum = np.zeros(pos.size, dtype=np.complex128)
            sum_all = np.zeros(pos.size, dtype=np.complex128)

        own_theta_values = 1j * off_sum
        own_voltage_values = self._y_row_diag_conj[pos] * voltage[pos] + sum_all
        self._add_indexed_value_blocks(
            H,
            (
                (
                    rows,
                    self.angle_col[pos],
                    np.where(kind == 0, own_theta_values.real, own_theta_values.imag),
                ),
                (
                    rows,
                    self.voltage_col[pos],
                    np.where(kind == 0, own_voltage_values.real, own_voltage_values.imag),
                ),
            ),
        )

        p_row_by_pos = plan["p_row_by_pos"]
        q_row_by_pos = plan["q_row_by_pos"]
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

        return plan["handled_mask"]

    def _add_zero_current_measurement_derivatives(
        self,
        H: np.ndarray,
        row: int,
        meas_type: str,
        device,
        current: complex,
        current_idx: int,
        voltage_complex: np.ndarray,
        voltage: np.ndarray,
    ) -> None:
        """Fill H entries for switches/zero branches whose current is an explicit state."""
        if meas_type.endswith("_FROM"):
            pos = self.node_pos[device.i_node]
            sign = 1.0
        elif meas_type.endswith("_TO"):
            pos = self.node_pos[device.j_node]
            sign = -1.0
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

        if meas_type.startswith("P") or meas_type.startswith("Q"):
            signed_current = sign * current
            power = voltage_complex[pos] * np.conj(signed_current)
            pick = np.real if meas_type.startswith("P") else np.imag
            angle_col = self.angle_col[pos]
            dS_dtheta = 1j * power
            dS_dV = power / voltage[pos] if abs(voltage[pos]) > 1e-12 else 0.0
            dS_dIr = sign * voltage_complex[pos]
            dS_dIi = -1j * sign * voltage_complex[pos]
            if angle_col >= 0:
                self._add_scalar_value(H, row, angle_col, float(pick(dS_dtheta)))
            self._add_scalar_value(H, row, self.voltage_col[pos], float(pick(dS_dV)))
            self._add_scalar_value(H, row, self.base_switch_re + current_idx, float(pick(dS_dIr)))
            self._add_scalar_value(H, row, self.base_switch_im + current_idx, float(pick(dS_dIi)))
        elif meas_type.startswith("I"):
            current_abs = abs(current)
            if current_abs > 1e-12:
                self._add_scalar_value(H, row, self.base_switch_re + current_idx, current.real / current_abs)
                self._add_scalar_value(H, row, self.base_switch_im + current_idx, current.imag / current_abs)
        elif meas_type.startswith("V"):
            self._add_scalar_value(H, row, self.voltage_col[pos], 1.0)
        else:
            raise RuntimeError(f"Unsupported zero-current measurement type: {meas_type}")

    def _network_power_derivatives(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized derivatives of node network injection S = V * conj(YV)."""
        y_matrix = self.Y.toarray() if issparse(self.Y) else self.Y
        y_conj = np.conj(y_matrix)
        exp_delta = np.exp(1j * (theta[:, None] - theta[None, :]))
        term = y_conj * (voltage[:, None] * voltage[None, :]) * exp_delta
        off = term.copy()
        np.fill_diagonal(off, 0.0)

        dS_dtheta = -1j * off
        np.fill_diagonal(dS_dtheta, 1j * np.sum(off, axis=1))

        dS_dvoltage = y_conj * voltage[:, None] * exp_delta
        y_diag = np.diag(y_conj)
        d_self_from_off = np.sum(y_conj * voltage[None, :] * exp_delta, axis=1) - y_diag * voltage
        np.fill_diagonal(dS_dvoltage, 2.0 * y_diag * voltage + d_self_from_off)
        return dS_dtheta, dS_dvoltage

    def _network_power_derivative_rows(
        self,
        theta: np.ndarray,
        voltage: np.ndarray,
        rows: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute network power derivative rows needed by generator measurements."""
        rows = np.asarray(rows, dtype=np.int64)
        dS_dtheta = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.complex128)
        dS_dvoltage = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.complex128)
        if rows.size == 0:
            return dS_dtheta, dS_dvoltage

        y_rows = self.Y[rows, :].toarray() if issparse(self.Y) else self.Y[rows, :]
        y_conj_rows = np.conj(y_rows)
        exp_delta = np.exp(1j * (theta[rows, None] - theta[None, :]))
        term = y_conj_rows * (voltage[rows, None] * voltage[None, :]) * exp_delta
        off = term.copy()
        local = np.arange(rows.size)
        off[local, rows] = 0.0

        theta_rows = -1j * off
        theta_rows[local, rows] = 1j * np.sum(off, axis=1)

        voltage_rows = y_conj_rows * voltage[rows, None] * exp_delta
        y_diag_rows = np.conj(self.Y.diagonal()[rows] if issparse(self.Y) else self.Y[rows, rows])
        d_self_from_off = np.sum(y_conj_rows * voltage[None, :] * exp_delta, axis=1) - y_diag_rows * voltage[rows]
        voltage_rows[local, rows] = 2.0 * y_diag_rows * voltage[rows] + d_self_from_off

        dS_dtheta[rows, :] = theta_rows
        dS_dvoltage[rows, :] = voltage_rows
        return dS_dtheta, dS_dvoltage

    def _load_power_derivatives(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_derivative = np.zeros(len(self.nodes), dtype=np.float64)
        q_derivative = np.zeros(len(self.nodes), dtype=np.float64)
        if self.load_pos_array.size == 0:
            return p_derivative, q_derivative
        vm = voltage[self.load_pos_array]
        np.add.at(p_derivative, self.load_pos_array, self.load_pv1_array + 2.0 * self.load_pv2_array * vm)
        np.add.at(q_derivative, self.load_pos_array, self.load_qv1_array + 2.0 * self.load_qv2_array * vm)
        return p_derivative, q_derivative

    def _add_switch_injection_derivatives(
        self,
        dP: np.ndarray,
        dQ: np.ndarray,
        node_pos: int,
        switch_current: np.ndarray,
        voltage_complex: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
        share: float,
    ) -> None:
        """Add zero-branch current sensitivities into a generator injection derivative."""
        for idx, (_, dev) in enumerate(self.zero_current_devices):
            i = self.node_pos[dev.i_node]
            j = self.node_pos[dev.j_node]
            if node_pos != i and node_pos != j:
                continue

            current = switch_current[idx]
            if node_pos == i:
                s = voltage_complex[i] * np.conj(current)
                dS_dIr = voltage_complex[i]
                dS_dIi = -1j * voltage_complex[i]
            else:
                s = voltage_complex[j] * np.conj(-current)
                dS_dIr = -voltage_complex[j]
                dS_dIi = 1j * voltage_complex[j]

            angle_col = self.angle_col[node_pos]
            dS_dtheta = 1j * s
            dS_dV = s / voltage[node_pos] if abs(voltage[node_pos]) > 1e-12 else 0.0
            if angle_col >= 0:
                dP[angle_col] += share * dS_dtheta.real
                dQ[angle_col] += share * dS_dtheta.imag
            voltage_col = int(self.voltage_col[node_pos])
            if voltage_col >= 0:
                dP[voltage_col] += share * np.real(dS_dV)
                dQ[voltage_col] += share * np.imag(dS_dV)
            dP[self.base_switch_re + idx] += share * dS_dIr.real
            dQ[self.base_switch_re + idx] += share * dS_dIr.imag
            dP[self.base_switch_im + idx] += share * dS_dIi.real
            dQ[self.base_switch_im + idx] += share * dS_dIi.imag

    def _current_from_power_derivatives(
        self,
        p: float,
        q: float,
        voltage: float,
        dP: np.ndarray,
        dQ: np.ndarray,
        dV_col: int,
    ) -> np.ndarray:
        dI = np.zeros_like(dP)
        s_abs = float(np.hypot(p, q))
        if abs(voltage) <= self.min_current_voltage or s_abs <= 1e-12:
            return dI
        dI += (p * dP + q * dQ) / (s_abs * voltage)
        if dV_col >= 0:
            dI[dV_col] -= s_abs / (voltage * voltage)
        return dI

    def _network_power_derivative_entries(
        self,
        pos: int,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse dS/dtheta and dS/dV entries for one network injection row."""
        if issparse(self.Y):
            start, end = self.Y.indptr[pos], self.Y.indptr[pos + 1]
            node_cols = self.Y.indices[start:end]
            y_values = self.Y.data[start:end]
        else:
            row = self.Y[pos, :]
            node_cols = np.nonzero(row)[0]
            y_values = row[node_cols]

        if node_cols.size:
            y_conj = np.conj(y_values)
            exp_delta = np.exp(1j * (theta[pos] - theta[node_cols]))
            term = y_conj * voltage[pos] * voltage[node_cols] * exp_delta
            off_mask = node_cols != pos
            off_nodes = node_cols[off_mask]
            off_term = term[off_mask]
            theta_nodes = np.concatenate((off_nodes, np.asarray([pos], dtype=np.int32)))
            theta_values = np.concatenate((-1j * off_term, np.asarray([1j * np.sum(off_term)], dtype=np.complex128)))

            y_diag_values = y_values[node_cols == pos]
            y_diag_conj = np.conj(y_diag_values[0]) if y_diag_values.size else 0.0
            sum_all = np.sum(y_conj * voltage[node_cols] * exp_delta)
            off_voltage_values = y_conj[off_mask] * voltage[pos] * exp_delta[off_mask]
            diag_voltage = y_diag_conj * voltage[pos] + sum_all
            voltage_nodes = np.concatenate((off_nodes, np.asarray([pos], dtype=np.int32)))
            voltage_values = np.concatenate((off_voltage_values, np.asarray([diag_voltage], dtype=np.complex128)))
        else:
            theta_nodes = np.asarray([pos], dtype=np.int32)
            theta_values = np.asarray([0.0j], dtype=np.complex128)
            voltage_nodes = np.asarray([pos], dtype=np.int32)
            voltage_values = np.asarray([0.0j], dtype=np.complex128)

        return theta_nodes, theta_values, voltage_nodes, voltage_values

    def _cached_network_power_derivative_entries(
        self,
        pos: int,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return dS/dtheta and dS/dV entries using cached Y-row sparsity."""
        node_cols = self._y_row_nodes[pos]
        if node_cols.size == 0:
            theta_nodes = np.asarray([pos], dtype=np.int32)
            zero = np.asarray([0.0j], dtype=np.complex128)
            return theta_nodes, zero, theta_nodes, zero

        y_conj = self._y_row_y_conj[pos]
        off_mask = self._y_row_off_mask[pos]
        off_nodes = self._y_row_off_nodes[pos]
        if off_mask is None:
            off_mask = node_cols != pos
            off_nodes = node_cols[off_mask]
            self._y_row_off_mask[pos] = off_mask
            self._y_row_off_nodes[pos] = off_nodes
        exp_delta = np.exp(1j * (theta[pos] - theta[node_cols]))
        term = y_conj * voltage[pos] * voltage[node_cols] * exp_delta
        off_term = term[off_mask]

        theta_nodes = np.empty(off_nodes.size + 1, dtype=np.int32)
        theta_nodes[:-1] = off_nodes
        theta_nodes[-1] = pos
        theta_values = np.empty(theta_nodes.size, dtype=np.complex128)
        theta_values[:-1] = -1j * off_term
        theta_values[-1] = 1j * np.sum(off_term)

        voltage_nodes = theta_nodes.copy()
        voltage_values = np.empty(voltage_nodes.size, dtype=np.complex128)
        voltage_values[:-1] = y_conj[off_mask] * voltage[pos] * exp_delta[off_mask]
        voltage_values[-1] = self._y_row_diag_conj[pos] * voltage[pos] + np.sum(
            y_conj * voltage[node_cols] * exp_delta
        )
        return theta_nodes, theta_values, voltage_nodes, voltage_values

    def _generator_derivative_entries(
        self,
        gen,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sparse dP/dx and dQ/dx entries for generator power measurements."""
        pos = self.node_pos[gen.node]
        share = self.generator_share_by_name[gen.name]
        cols = []
        dP = []
        dQ = []

        theta_nodes, theta_values, voltage_nodes, voltage_values = self._cached_network_power_derivative_entries(
            pos,
            theta,
            voltage,
        )
        angle_cols = self.angle_col[theta_nodes]
        angle_mask = angle_cols >= 0
        if np.any(angle_mask):
            cols.extend(angle_cols[angle_mask].tolist())
            dP.extend((share * theta_values[angle_mask].real).tolist())
            dQ.extend((share * theta_values[angle_mask].imag).tolist())

        voltage_cols = self.voltage_col[voltage_nodes]
        voltage_mask = voltage_cols >= 0
        if np.any(voltage_mask):
            cols.extend(voltage_cols[voltage_mask].tolist())
            dP.extend((share * voltage_values[voltage_mask].real).tolist())
            dQ.extend((share * voltage_values[voltage_mask].imag).tolist())

        own_voltage_col = int(self.voltage_col[pos])
        if own_voltage_col >= 0:
            cols.append(own_voltage_col)
            dP.append(float(share * p_load_dv[pos]))
            dQ.append(float(share * q_load_dv[pos]))

        for idx, from_i in self.zero_current_by_node.get(pos, []):
            current = switch_current[idx]
            if from_i:
                s = voltage_complex[pos] * np.conj(current)
                dS_dIr = voltage_complex[pos]
                dS_dIi = -1j * voltage_complex[pos]
            else:
                s = voltage_complex[pos] * np.conj(-current)
                dS_dIr = -voltage_complex[pos]
                dS_dIi = 1j * voltage_complex[pos]
            dS_dtheta = 1j * s
            dS_dV = s / voltage[pos] if abs(voltage[pos]) > 1e-12 else 0.0
            angle_col = int(self.angle_col[pos])
            if angle_col >= 0:
                cols.append(angle_col)
                dP.append(float(share * dS_dtheta.real))
                dQ.append(float(share * dS_dtheta.imag))
            if own_voltage_col >= 0:
                cols.append(own_voltage_col)
                dP.append(float(share * np.real(dS_dV)))
                dQ.append(float(share * np.imag(dS_dV)))
            cols.extend((self.base_switch_re + idx, self.base_switch_im + idx))
            dP.extend((float(share * dS_dIr.real), float(share * dS_dIi.real)))
            dQ.extend((float(share * dS_dIr.imag), float(share * dS_dIi.imag)))

        return (
            np.asarray(cols, dtype=np.int32),
            np.asarray(dP, dtype=np.float64),
            np.asarray(dQ, dtype=np.float64),
        )

    @staticmethod
    def _add_sparse_generator_row(H, row: int, cols: np.ndarray, values: np.ndarray) -> None:
        if cols.size == 0:
            return
        rows = np.full(cols.size, int(row), dtype=np.int32)
        H.add_many(rows, cols, values)

    @staticmethod
    def _add_sparse_repeated_rows(H, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        if rows.size == 0 or cols.size == 0:
            return
        if rows.size == 1:
            H.add_many(np.full(cols.size, int(rows[0]), dtype=np.int32), cols, values)
            return
        H.add_many(
            np.repeat(rows.astype(np.int32, copy=False), cols.size),
            np.tile(cols.astype(np.int32, copy=False), rows.size),
            np.tile(values.astype(np.float64, copy=False), rows.size),
        )

    def _generator_measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, object]:
        if measurements is self.active_measurements:
            active_plan = getattr(self, "_active_generator_measurement_plan", None)
            if active_plan is not None:
                return active_plan
        key = id(measurements)
        cached = self._generator_measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        handled = np.zeros(len(measurements), dtype=bool)
        value_rows = []
        value_kind = []
        value_pos = []
        value_index = []
        for row, meas in self._measurement_rows_for_types(measurements, ("ACGenerator",)):
            if meas.device_type != "ACGenerator" or meas.meas_type == "V_GEN":
                continue
            if meas.meas_type not in ("P_GEN", "Q_GEN", "I_GEN"):
                continue
            gen = self.generator_by_name[meas.device_name]
            value_rows.append(row)
            value_kind.append({"P_GEN": 0, "Q_GEN": 1, "I_GEN": 2}[meas.meas_type])
            value_pos.append(self.node_pos[gen.node])
            value_index.append(self.generator_state_index_by_name[gen.name])
            handled[row] = True

        plan = {
            "handled_mask": handled,
            "value_rows": self._int_array(value_rows),
            "value_kind": self._int_array(value_kind),
            "value_pos": self._int_array(value_pos),
            "value_index": self._int_array(value_index),
        }
        if len(self._generator_measurement_plan_cache) > 16:
            self._generator_measurement_plan_cache.clear()
        self._generator_measurement_plan_cache[key] = (measurements, plan)
        return plan

    def _fill_generator_values_vectorized(
        self,
        values: np.ndarray,
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-evaluate ACGenerator P/Q/I measurements from explicit P/Q states."""
        plan = self._generator_measurement_plan(measurements)
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        kind = plan["value_kind"]
        row_values = np.zeros(rows.size, dtype=np.float64)
        p_mask = kind == 0
        q_mask = kind == 1
        i_mask = kind == 2
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
        measurements: Sequence[Measurement],
        x: np.ndarray,
        voltage: np.ndarray,
    ) -> np.ndarray:
        """Batch-fill sparse generator P/Q/I rows from explicit P/Q states."""
        plan = self._generator_measurement_plan(measurements)
        rows = plan["value_rows"]
        if rows.size == 0:
            return plan["handled_mask"]

        kind = plan["value_kind"]
        gen_idx = plan["value_index"]
        pos = plan["value_pos"]
        p = np.asarray(x[self.base_gen_p + gen_idx], dtype=np.float64)
        q = np.asarray(x[self.base_gen_q + gen_idx], dtype=np.float64)
        p_mask = kind == 0
        q_mask = kind == 1
        i_mask = kind == 2
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

    def _generator_derivative_vectors(
        self,
        gen,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        switch_current: np.ndarray,
        network_dtheta: np.ndarray,
        network_dvoltage: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build dP/dx and dQ/dx for a generator represented by node power balance."""
        pos = self.node_pos[gen.node]
        share = self.generator_share_by_name[gen.name]
        dP = np.zeros(self.n_state, dtype=np.float64)
        dQ = np.zeros(self.n_state, dtype=np.float64)
        angle_mask = self.angle_col >= 0
        if np.any(angle_mask):
            np.add.at(dP, self.angle_col[angle_mask], share * network_dtheta[pos, angle_mask].real)
            np.add.at(dQ, self.angle_col[angle_mask], share * network_dtheta[pos, angle_mask].imag)
        voltage_mask = self.voltage_col >= 0
        if np.any(voltage_mask):
            np.add.at(dP, self.voltage_col[voltage_mask], share * network_dvoltage[pos, voltage_mask].real)
            np.add.at(dQ, self.voltage_col[voltage_mask], share * network_dvoltage[pos, voltage_mask].imag)
        own_voltage_col = int(self.voltage_col[pos])
        if own_voltage_col >= 0:
            dP[own_voltage_col] += share * p_load_dv[pos]
            dQ[own_voltage_col] += share * q_load_dv[pos]
        self._add_switch_injection_derivatives(
            dP,
            dQ,
            pos,
            switch_current,
            voltage_complex,
            theta,
            voltage,
            share,
        )
        return dP, dQ

    def _assemble_jacobian(
        self,
        x: np.ndarray,
        measurements: Optional[Sequence[Measurement]] = None,
        sparse: bool = False,
    ):
        """Assemble the WLS measurement Jacobian H = dh(x)/dx."""
        measurements = self._normalize_measurements(measurements)
        theta, voltage = self._unpack_state(x)
        switch_current = self._switch_current_from_state(x)
        voltage_complex = self._complex_voltage(theta, voltage)
        H = (
            SparseJacobianBuilder((len(measurements), self.n_state))
            if sparse
            else np.zeros((len(measurements), self.n_state), dtype=np.float64)
        )
        branch_power_derivative_cache = {}
        branch_current_derivative_cache = {}
        vectorized_branch_rows = self._fill_branch_transformer_jacobian_vectorized(
            H,
            measurements,
            theta,
            voltage,
            voltage_complex,
        )
        vectorized_simple_rows = self._fill_simple_jacobian_vectorized(
            H,
            measurements,
            x,
            voltage,
        )
        vectorized_zero_rows = self._fill_zero_current_jacobian_vectorized(
            H,
            measurements,
            voltage,
            voltage_complex,
            switch_current,
        )
        vectorized_generator_rows = self._fill_generator_jacobian_sparse(
            H,
            measurements,
            x,
            voltage,
        )
        vectorized_balance_rows = self._fill_balance_jacobian_vectorized(
            H,
            measurements,
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
        if sparse and (
            (measurements is self.active_measurements and self.active_measurements_are_vectorized)
            or np.all(vectorized_rows)
        ):
            return H.to_csr()

        for row, meas in enumerate(measurements):
            if vectorized_rows[row]:
                continue
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                node = self.node_by_name[meas.device_name]
                pos = self.node_pos[node.idx]
                if mtype == "V":
                    H[row, self.voltage_col[pos]] = 1.0
                elif mtype in ("ANGLE", "THETA"):
                    angle_col = self.angle_col[pos]
                    if angle_col >= 0:
                        H[row, angle_col] = 1.0
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")

            elif meas.device_type == "ACBranch":
                br = self.branch_by_name[meas.device_name]
                i = self.node_pos[br.i_node]
                j = self.node_pos[br.j_node]
                yff, yft, ytf, ytt = self.branch_stamp_by_name[br.name]
                if mtype in ("P_FROM", "Q_FROM"):
                    cache_key = (meas.device_type, br.name, "from")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, i, j, *derivatives)
                elif mtype == "V_FROM":
                    H[row, self.voltage_col[i]] = 1.0
                elif mtype == "I_FROM":
                    cache_key = (meas.device_type, br.name, "from")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, i, j, *derivatives)
                elif mtype in ("P_TO", "Q_TO"):
                    cache_key = (meas.device_type, br.name, "to")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, j, i, *derivatives)
                elif mtype == "V_TO":
                    H[row, self.voltage_col[j]] = 1.0
                elif mtype == "I_TO":
                    cache_key = (meas.device_type, br.name, "to")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, j, i, *derivatives)
                else:
                    raise RuntimeError(f"Unsupported ACBranch measurement type: {mtype}")

            elif meas.device_type == "ACTransformer":
                tr = self.transformer_by_name[meas.device_name]
                i = self.node_pos[tr.i_node]
                j = self.node_pos[tr.j_node]
                yff, yft, ytf, ytt = self.transformer_stamp_by_name[tr.name]
                if mtype in ("P_FROM", "Q_FROM"):
                    cache_key = (meas.device_type, tr.name, "from")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, i, j, *derivatives)
                elif mtype == "V_FROM":
                    H[row, self.voltage_col[i]] = 1.0
                elif mtype == "I_FROM":
                    cache_key = (meas.device_type, tr.name, "from")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            i, j, yff, yft, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, i, j, *derivatives)
                elif mtype in ("P_TO", "Q_TO"):
                    cache_key = (meas.device_type, tr.name, "to")
                    if cache_key not in branch_power_derivative_cache:
                        branch_power_derivative_cache[cache_key] = self._branch_power_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_power_derivative_cache[cache_key]
                    self._add_power_derivatives(H, row, mtype, j, i, *derivatives)
                elif mtype == "V_TO":
                    H[row, self.voltage_col[j]] = 1.0
                elif mtype == "I_TO":
                    cache_key = (meas.device_type, tr.name, "to")
                    if cache_key not in branch_current_derivative_cache:
                        branch_current_derivative_cache[cache_key] = self._branch_current_derivatives(
                            j, i, ytt, ytf, theta, voltage
                        )
                    derivatives = branch_current_derivative_cache[cache_key]
                    self._add_current_magnitude_derivatives(H, row, j, i, *derivatives)
                else:
                    raise RuntimeError(f"Unsupported ACTransformer measurement type: {mtype}")

            elif meas.device_type == "ACLoad":
                load = self.load_by_name[meas.device_name]
                node = self.node_by_name[load.node_obj.name]
                pos = self.node_pos[node.idx]
                if mtype == "P_LOAD":
                    H[row, self.load_p_col_by_name[load.name]] = 1.0
                elif mtype == "Q_LOAD":
                    H[row, self.load_q_col_by_name[load.name]] = 1.0
                elif mtype == "V_LOAD":
                    H[row, self.voltage_col[pos]] = 1.0
                elif mtype == "I_LOAD":
                    vm = voltage[pos]
                    load_idx = self.load_state_index_by_name[load.name]
                    p = float(x[self.base_load_p + load_idx])
                    q = float(x[self.base_load_q + load_idx])
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(vm) > self.min_current_voltage:
                        H[row, self.base_load_p + load_idx] = p / (s_abs * vm)
                        H[row, self.base_load_q + load_idx] = q / (s_abs * vm)
                        H[row, self.voltage_col[pos]] = -s_abs / (vm * vm)
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")

            elif meas.device_type == "ACGenerator":
                gen = self.generator_by_name[meas.device_name]
                node = self.node_by_name[gen.node_obj.name]
                pos = self.node_pos[node.idx]
                if mtype == "V_GEN":
                    H[row, self.voltage_col[pos]] = 1.0
                    continue

                if mtype == "P_GEN":
                    H[row, self.gen_p_col_by_name[gen.name]] = 1.0
                elif mtype == "Q_GEN":
                    H[row, self.gen_q_col_by_name[gen.name]] = 1.0
                elif mtype == "I_GEN":
                    gen_idx = self.generator_state_index_by_name[gen.name]
                    p = float(x[self.base_gen_p + gen_idx])
                    q = float(x[self.base_gen_q + gen_idx])
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(voltage[pos]) > self.min_current_voltage:
                        H[row, self.base_gen_p + gen_idx] = p / (s_abs * voltage[pos])
                        H[row, self.base_gen_q + gen_idx] = q / (s_abs * voltage[pos])
                        H[row, self.voltage_col[pos]] = -s_abs / (voltage[pos] * voltage[pos])
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")

            elif meas.device_type == "ACZeroBranch":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[zbr.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[zbr.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[zbr.i_node]]
                    j_col = self.angle_col[self.node_pos[zbr.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    zbr_idx = self.zero_branch_pos[zbr.name]
                    self._add_zero_current_measurement_derivatives(
                        H,
                        row,
                        mtype,
                        zbr,
                        switch_current[zbr_idx],
                        zbr_idx,
                        voltage_complex,
                        voltage,
                    )

            elif meas.device_type == "ACZeroBranchConstraint":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[zbr.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[zbr.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[zbr.i_node]]
                    j_col = self.angle_col[self.node_pos[zbr.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    raise RuntimeError(f"Unsupported ACZeroBranchConstraint measurement type: {mtype}")

            elif meas.device_type == "ACSwitchConstraint":
                sw = self.switch_by_name[meas.device_name]
                if mtype == "V_DIFF":
                    H[row, self.voltage_col[self.node_pos[sw.i_node]]] = 1.0
                    H[row, self.voltage_col[self.node_pos[sw.j_node]]] = -1.0
                elif mtype in ("ANGLE_DIFF", "THETA_DIFF"):
                    i_col = self.angle_col[self.node_pos[sw.i_node]]
                    j_col = self.angle_col[self.node_pos[sw.j_node]]
                    if i_col >= 0:
                        H[row, i_col] = 1.0
                    if j_col >= 0:
                        H[row, j_col] = -1.0
                else:
                    raise RuntimeError(f"Unsupported ACSwitchConstraint measurement type: {mtype}")

            elif meas.device_type == "ACSwitch":
                sw = self.switch_by_name[meas.device_name]
                sw_idx = self.switch_pos[sw.name]
                self._add_zero_current_measurement_derivatives(
                    H,
                    row,
                    mtype,
                    sw,
                    switch_current[sw_idx],
                    sw_idx,
                    voltage_complex,
                    voltage,
                )

            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")
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
                self._initial_observability_cache = cached
                return cached
        x = self.initial_state() if x is None else x
        measurements = self._normalize_measurements(measurements)
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            return ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])

        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.state_labels,
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
            measurement_count=len(measurements),
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )
        if use_default_cache:
            if len(_OBSERVABILITY_RESULT_CACHE) > 32:
                _OBSERVABILITY_RESULT_CACHE.clear()
            _OBSERVABILITY_RESULT_CACHE[self._default_observability_cache_key()] = result
            self._initial_observability_cache = result
        return result

    def _has_structural_observability_certificate(self, H) -> bool:
        """Certify large sparse AC cases when numeric LU is conservative but structure is anchored."""
        rank = sparse_structural_rank(H)
        if rank != self.n_state:
            return False
        return not unanchored_angle_state_labels(H, self.state_labels, "theta:")

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
    ) -> EstimateResult:
        """Run weighted least squares with simple damping to avoid voltage divergence."""
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        measurements = self._normalize_measurements(measurements)
        if len(measurements) < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {len(measurements)} < {self.n_state}")

        x = self.initial_state() if x0 is None else x0.copy()
        if measurements is self.active_measurements and x0 is None:
            observability = self._profile_call("solve.observability", self.observability_analysis)
        else:
            observability = self._profile_call("solve.observability", self.observability_analysis, x, measurements)
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
        normal_factor_diag = None
        cached_z_est = None
        cached_residual = None
        cached_objective = None
        normal_solver = NormalEquationSolver()
        normal_pattern = None

        if verbose:
            _print_iteration_header()

        flat_restart_enabled = False
        iteration_limit = self.max_iter

        for iteration in range(1, iteration_limit + 1):
            if cached_z_est is None:
                z_est = self._profile_call("solve.evaluate", self.evaluate, x, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                residual = self._measurement_residual(z, z_est, measurements)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.residual_objective", time.perf_counter() - start)
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
                cached_z_est = cached_residual = cached_objective = None
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            H = self._profile_call("solve.jacobian", self.jacobian_sparse, x, measurements)
            if normal_pattern is None and issparse(H):
                normal_pattern = _normal_equation_structural_pattern(H)
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            gain, rhs = self._profile_call(
                "solve.normal_equations",
                build_normal_equations,
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=normal_pattern,
            )
            dx, normal_factor_diag = self._profile_call(
                "solve.factor_solve",
                normal_solver.solve,
                gain,
                rhs,
                return_factor_diag=False,
            )

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
                candidate_z_est = self._profile_call("solve.line_search_evaluate", self.evaluate, candidate, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                candidate_residual = self._measurement_residual(z, candidate_z_est, measurements)
                candidate_objective = self._weighted_objective(weight, candidate_residual)
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
                    cached_z_est = candidate_z_est
                    cached_residual = candidate_residual
                    cached_objective = candidate_objective
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
                if cached_residual is None:
                    cached_z_est = self._profile_call("solve.verbose_evaluate", self.evaluate, x, measurements)
                    cached_residual = self._measurement_residual(z, cached_z_est, measurements)
                    cached_objective = self._weighted_objective(weight, cached_residual)
                updated_residual = cached_residual
                updated_residual_inf = float(np.linalg.norm(updated_residual, np.inf)) if updated_residual.size else 0.0
                _print_iteration(
                    iteration,
                    cached_objective,
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
            )
            restart_result.iterations += iteration
            return restart_result

        if not final_quantities_current:
            if cached_z_est is None:
                z_est = self._profile_call("solve.final_evaluate", self.evaluate, x, measurements)
                start = time.perf_counter() if self.profile_enabled else None
                residual = self._measurement_residual(z, z_est, measurements)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.final_residual_objective", time.perf_counter() - start)
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
        if final_diagnostics and not final_quantities_current:
            H = self._profile_call("solve.final_jacobian", self.jacobian_sparse, x, measurements)
            if weighted_residual is not None:
                np.multiply(weight, residual, out=weighted_residual)
            gain, _ = self._profile_call(
                "solve.final_normal_equations",
                build_normal_equations,
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=normal_pattern,
            )
        elif not final_diagnostics:
            H = None
            gain = None
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
            z_est=z_est,
            residual=residual,
            H=H,
            gain=gain,
            measurements=measurements,
            observability=observability,
        )

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Return measurements whose normalized residual exceeds the bad-data threshold."""
        if result.H is None or result.gain is None:
            result.H = self.jacobian_sparse(result.x, result.measurements)
            result.gain, _ = build_normal_equations(
                result.H,
                result.residual,
                np.asarray([meas.weight for meas in result.measurements], dtype=np.float64),
            )
        threshold = self.params.bad_threshold if threshold is None else threshold
        weights = np.asarray([meas.weight for meas in result.measurements], dtype=np.float64)
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

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        threshold = self.params.bad_threshold if threshold is None else threshold
        max_remove = self.params.max_remove if max_remove is None else max_remove
        measurements = list(self.active_measurements)
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
            measurements = [meas for meas in measurements if meas.idx != worst.measurement.idx]
            x0 = result.x
        return result, removed

    def apply_state(self, x: np.ndarray) -> None:
        theta, voltage = self._unpack_state(x)
        for pos, node in enumerate(self.nodes):
            node.angle = float(theta[pos])
            node.voltage = float(voltage[pos])
        for idx, gen in enumerate(self.generator_order):
            gen.p = float(x[self.base_gen_p + idx])
            gen.q = float(x[self.base_gen_q + idx])
        for idx, load in enumerate(self.load_order):
            load.p = float(x[self.base_load_p + idx])
            load.q = float(x[self.base_load_q + idx])

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        theta, voltage = self._unpack_state(x)
        print("Estimated states:")
        for pos, node in enumerate(self.nodes[:limit]):
            print(f"  {node.name:10s} V={voltage[pos]:.9f} theta={theta[pos]:.9f} rad")
        if len(self.nodes) > limit:
            print(f"  ... {len(self.nodes) - limit} more nodes")


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
    )

    initial_observability = estimator.observability_analysis()
    _print_observability(initial_observability)

    if args.remove_bad_data:
        result, removed = estimator.estimate_with_bad_data_removal(
            args.bad_threshold,
            max_remove=args.max_remove,
            verbose=not args.quiet,
        )
        if removed:
            print("Removed bad data:")
            for item in removed:
                print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")
    else:
        result = estimator.estimate(verbose=not args.quiet, final_diagnostics=not args.skip_bad_data)

    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
    )
    _print_observability(result.observability)
    if not args.skip_bad_data:
        bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
        bad_items, normalized = estimator.identify_bad_data(result, bad_threshold)
        _print_bad_data(bad_items, normalized, bad_threshold)

    if args.print_state:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in sorted(estimator.profile_times.items()):
            print(f"  {name}={value:.6f}s")

    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
