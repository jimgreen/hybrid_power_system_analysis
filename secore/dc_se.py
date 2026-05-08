import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.dc_lf import DCPowerFlowCalc, load_dc_ppc_from_e_file
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    BUS_COLS as DC_BUS_COLS,
    DCDC_COLS as DC_DCDC_COLS,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    DCPowerNetwork,
    build_dc_network_from_ppc,
)
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    EstimateResult,
    GEN_CONTROL_KIND,
    GEN_MEASUREMENT_KIND,
    LOAD_MEASUREMENT_KIND,
    Measurement,
    MeasurementList,
    MeasurementTable,
    ObservabilityResult,
    TERMINAL_MEASUREMENT_KIND,
    measurement_table_from_measurements,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from efile_read import EBook  # Retained for compatibility; measurement loading uses the direct parser below.
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
from secore.se_array_plan import (
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    copy_measurement_view,
    take_measurement_view,
)
from secore.se_result import SEResult
from unit_system import dc_current_base_ka


DEFAULT_CASE = ROOT_DIR / "data" / "dc" / "dc_net_30.e"
DEFAULT_MEAS = ROOT_DIR / "data" / "dc" / "dc_net_30.meas"

_MEASUREMENT_REQUIRED_COLUMNS = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
_TERMINAL_KIND = TERMINAL_MEASUREMENT_KIND
_LOAD_MEASUREMENT_KIND = LOAD_MEASUREMENT_KIND
_GEN_MEASUREMENT_KIND = GEN_MEASUREMENT_KIND
_GEN_CONTROL_KIND = GEN_CONTROL_KIND
_DEVICE_TYPE_CODES = DEVICE_TYPE_CODES


def _measurement_table_from_measurements(measurements: Sequence["Measurement"]) -> MeasurementTable:
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=_DEVICE_TYPE_CODES,
    )


def _read_measurements_direct(meas_file: Path) -> MeasurementList:
    """Read only the Measurement block without materializing a generic EBook dict."""
    measurements: List[Measurement] = []
    idx_values = []
    name_values = []
    device_type_values = []
    device_name_values = []
    meas_type_values = []
    weight_values = []
    valid_values = []
    value_values = []
    device_type_code_values = []
    append_idx = idx_values.append
    append_name = name_values.append
    append_device_type = device_type_values.append
    append_device_name = device_name_values.append
    append_meas_type = meas_type_values.append
    append_weight = weight_values.append
    append_valid = valid_values.append
    append_value = value_values.append
    append_device_type_code = device_type_code_values.append
    in_measurement = False
    found_measurement_block = False
    header = None
    missing = ()
    new_measurement = Measurement.__new__
    append_measurement = measurements.append
    intern = sys.intern
    device_type_cache: Dict[str, str] = {}
    meas_type_cache: Dict[str, str] = {}
    device_type_cache_get = device_type_cache.get
    meas_type_cache_get = meas_type_cache.get
    device_type_code_get = _DEVICE_TYPE_CODES.get
    idx_col = name_col = dev_type_col = dev_name_col = meas_type_col = weight_col = valid_col = value_col = -1

    with Path(meas_file).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line or raw_line[0] in "\r\n":
                continue
            if not in_measurement:
                if raw_line.startswith("<Measurement>"):
                    in_measurement = True
                    found_measurement_block = True
                continue
            first = raw_line[0]
            if first == "<":
                if raw_line.strip() == "</Measurement>":
                    break
                raise SyntaxError(f"Invalid Measurement row at line {line_no} in {meas_file}")
            if first == "@":
                names = raw_line[1:].split()
                header = {name: idx for idx, name in enumerate(names)}
                missing = tuple(name for name in _MEASUREMENT_REQUIRED_COLUMNS if name not in header)
                if missing:
                    raise RuntimeError(f"{meas_file} Measurement header is missing columns: {missing}")
                idx_col = header["idx"]
                name_col = header["name"]
                dev_type_col = header["dev_type"]
                dev_name_col = header["dev_name"]
                meas_type_col = header["meas_type"]
                weight_col = header["weight"]
                valid_col = header["valid"]
                value_col = header["value"]
                continue
            if first != "#":
                raise SyntaxError(f"Invalid Measurement row at line {line_no} in {meas_file}")
            if header is None:
                raise RuntimeError(f"{meas_file} Measurement data appears before the header")

            fields = raw_line[1:].split()
            if len(fields) < len(header):
                raise RuntimeError(f"Malformed Measurement row at line {line_no} in {meas_file}")

            raw_device_type = fields[dev_type_col]
            device_type = device_type_cache_get(raw_device_type)
            if device_type is None:
                device_type = intern(raw_device_type)
                device_type_cache[raw_device_type] = device_type
            raw_meas_type = fields[meas_type_col]
            meas_type = meas_type_cache_get(raw_meas_type)
            if meas_type is None:
                meas_type = intern(raw_meas_type.upper())
                meas_type_cache[raw_meas_type] = meas_type

            idx = int(fields[idx_col])
            name = fields[name_col]
            device_name = fields[dev_name_col]
            weight = float(fields[weight_col])
            valid = bool(int(fields[valid_col]))
            value = float(fields[value_col])

            meas = new_measurement(Measurement)
            meas.idx = idx
            meas.name = name
            meas.device_type = device_type
            meas.device_name = device_name
            meas.meas_type = meas_type
            meas.weight = weight
            meas.valid = valid
            meas.value = value
            append_measurement(meas)
            append_idx(idx)
            append_name(name)
            append_device_type(device_type)
            append_device_name(device_name)
            append_meas_type(meas_type)
            append_weight(weight)
            append_valid(valid)
            append_value(value)
            append_device_type_code(device_type_code_get(device_type, 0))

    if not found_measurement_block:
        raise RuntimeError(f"{meas_file} does not contain a <Measurement> block")
    if header is None:
        raise RuntimeError(f"{meas_file} Measurement block does not contain a header")
    table = MeasurementTable(
        idx=np.asarray(idx_values, dtype=np.int64),
        name=np.asarray(name_values, dtype=object),
        device_type=np.asarray(device_type_values, dtype=object),
        device_name=np.asarray(device_name_values, dtype=object),
        meas_type=np.asarray(meas_type_values, dtype=object),
        weight=np.asarray(weight_values, dtype=np.float64),
        valid=np.asarray(valid_values, dtype=bool),
        value=np.asarray(value_values, dtype=np.float64),
        device_type_code=np.asarray(device_type_code_values, dtype=np.int16),
        angle_mask=np.zeros(len(measurements), dtype=bool),
    )
    return MeasurementList(measurements, table)


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
        network: Optional[DCPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
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
        self.prepare(
            network=network,
            measurements=measurements,
            prepare_active_measurements=prepare_active_measurements,
            defer_prepare_finalize=defer_prepare_finalize,
        )

    def prepare(
        self,
        *,
        network: Optional[DCPowerNetwork] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
    ) -> "DCStateEstimator":
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        if network is None:
            self.network = self._load_network(self.e_file)
        else:
            self.network = network
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if measurements is None:
            self.measurements = self._load_measurements(self.meas_file)
        elif isinstance(measurements, MeasurementList):
            self.measurements = measurements
        else:
            self.measurements = list(measurements)
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)
        stage_start = time.perf_counter()
        self._build_unit_scale_cache()

        self.nodes = sorted(
            getattr(self.network, "alive_buses", None)
            or [bus for bus in getattr(self.network, "buses", []) if getattr(bus, "is_alive", False)]
            or [node for node in self.network.nodes if getattr(node, "is_alive", False)],
            key=lambda item: item.idx,
        )
        if not self.nodes:
            raise RuntimeError("No alive DC nodes are available for state estimation")

        self.node_pos = {}
        self.node_by_name = {}
        self.node_by_idx = {}
        for pos, bus in enumerate(self.nodes):
            members = getattr(bus, "nodes", None) or [bus]
            for node in members:
                self.node_pos[node.idx] = pos
                self.node_by_name[node.name] = bus
                self.node_by_idx[node.idx] = bus
            self.node_pos.setdefault(bus.idx, pos)
            self.node_by_name.setdefault(bus.name, bus)
            self.node_by_idx.setdefault(bus.idx, bus)
        self.branch_by_name = {br.name: br for br in self.network.branches if getattr(br, "is_alive", False)}
        self.load_by_name = {load.name: load for load in self.network.loads if getattr(load, "is_alive", False)}
        self.generator_by_name = {gen.name: gen for gen in self.network.generators if getattr(gen, "is_alive", False)}
        self.switch_by_name = {sw.name: sw for sw in self.network.switches if getattr(sw, "is_alive", False)}
        self.break_by_name = {brk.name: brk for brk in getattr(self.network, "breakers", []) if getattr(brk, "is_alive", False)}
        self.zero_branch_by_name = {
            zbr.name: zbr
            for zbr in self.network.zero_branches
            if getattr(zbr, "is_alive", False)
        }
        self.dcdc_by_name = {
            conv.name: conv
            for conv in self.network.dcdc_converters
            if getattr(conv, "is_alive", False)
        }

        self.switches = sorted(self.switch_by_name.values(), key=lambda item: item.idx)
        self.breakers = sorted(self.break_by_name.values(), key=lambda item: item.idx)
        self.zero_branches = sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
        self._defer_prepare_finalize_pending = bool(defer_prepare_finalize)
        if defer_prepare_finalize:
            self._build_measurement_scale_cache()
            self.power_flow_seed_converged = False
            self.targeted_observability_pseudo_count = 0
            return self
        # Reference selection must see only active, unit-normalized real measurements.
        self.finalize_prepare(prepare_active_measurements=prepare_active_measurements)
        self._prepared = True
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

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
            self._build_measurement_scale_cache()
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
            run_seed = getattr(type(self), "_run_power_flow_seed", None)
            original_run_seed = globals().get("_ORIGINAL_DC_RUN_POWER_FLOW_SEED")
            if (
                not self.power_flow_seed_converged
                and original_run_seed is not None
                and run_seed is original_run_seed
            ):
                self._apply_measurement_seed_to_network(force_object=True)
            self._record_profile_time("seed.lf", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._sync_bus_state_from_members()
            self._record_profile_time("seed.refresh_file_state", time.perf_counter() - stage_start)
            self._record_profile_time("seed.total", time.perf_counter() - seed_start)
        stage_start = time.perf_counter()
        self.node_voltage_measurements = self._node_voltage_measurements()
        self.node_degrees = self._node_incident_degrees()
        self.references = self._select_reference_nodes()
        self._build_zero_tie_voltage_layout()
        self.zero_current_devices = [
            ("Z", zbr)
            for zbr in self.zero_branches
        ] + [("B", brk) for brk in self.breakers]
        self.dcdc_converters = sorted(self.dcdc_by_name.values(), key=lambda item: item.idx)
        self.v_generators = sorted(
            [gen for gen in self.generator_by_name.values() if gen.control_type == "V"],
            key=lambda item: item.idx,
        )

        self.zero_current_pos = {(kind, dev.name): pos for pos, (kind, dev) in enumerate(self.zero_current_devices)}
        self.switch_pos = {}
        self.break_pos = {brk.name: self.zero_current_pos[("B", brk.name)] for brk in self.breakers}
        self.zero_branch_pos = {
            zbr.name: self.zero_current_pos[("Z", zbr.name)]
            for zbr in self.zero_branches
            if ("Z", zbr.name) in self.zero_current_pos
        }
        self.dcdc_pos = {conv.name: pos for pos, conv in enumerate(self.dcdc_converters)}
        self.v_generator_pos = {gen.name: pos for pos, gen in enumerate(self.v_generators)}

        self.n_switch = len(self.zero_current_devices)
        self.n_dcdc_power = 2 * len(self.dcdc_converters)
        self.n_v_generator = len(self.v_generators)
        self.switch_start = self.n_voltage
        self.dcdc_start = self.switch_start + self.n_switch
        self.v_generator_start = self.dcdc_start + self.n_dcdc_power
        self.n_state = self.v_generator_start + self.n_v_generator

        self.state_labels = [f"V:{self.nodes[pos].name}" for pos in self.voltage_state_pos]
        self.state_labels.extend(f"I_ZERO:{zbr.name}" for zbr in self.zero_branches if zbr.name in self.zero_branch_pos)
        self.state_labels.extend(f"I_BREAK:{brk.name}" for brk in self.breakers)
        for conv in self.dcdc_converters:
            self.state_labels.append(f"P_DCDC_FROM:{conv.name}")
            self.state_labels.append(f"P_DCDC_TO:{conv.name}")
        self.state_labels.extend(f"P_VGEN:{gen.name}" for gen in self.v_generators)
        self._build_apply_state_index()
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
            self.active_measurements = MeasurementList(
                [],
                normalized=getattr(self.measurements, "normalized", False),
            )
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
        )
        self.measurement_table = active_view.source_table
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
    def _merge_measurement_plan(
        head: Dict[str, np.ndarray],
        tail: Dict[str, np.ndarray],
        row_offset: int,
    ) -> Dict[str, np.ndarray]:
        row_keys = {
            "node_rows",
            "branch_rows",
            "load_rows",
            "gen_rows",
            "switch_rows",
            "constraint_rows",
            "dcdc_rows",
        }
        merged: Dict[str, np.ndarray] = {}
        for key, head_value in head.items():
            tail_value = tail[key]
            if key in row_keys:
                merged[key] = np.concatenate(
                    (
                        np.asarray(head_value, dtype=np.int64),
                        np.asarray(tail_value, dtype=np.int64) + int(row_offset),
                    )
                ).astype(np.int64, copy=False)
            else:
                merged[key] = np.concatenate((np.asarray(head_value), np.asarray(tail_value)))
        return merged

    def _incremental_update_active_measurement_indexes(self, appended_measurements: Sequence[Measurement]) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
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
        self.measurement_table = concat_measurement_tables(master_table, appended_list.table)
        self.measurements.table = self.measurement_table
        active_view = append_active_measurement_view(
            build_active_measurement_view(self.active_measurements, table_builder=_measurement_table_from_measurements),
            appended_list,
            source_row_start=len(master_table.idx),
            table_builder=_measurement_table_from_measurements,
        )
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
        previous_plan = getattr(self, "_active_measurement_plan", None)
        if previous_plan is None:
            self._measurement_plan_cache = {}
            self._active_measurement_plan = self._measurement_plan(self.active_measurements)
        else:
            appended_plan = self._measurement_plan(appended_list)
            self._active_measurement_plan = self._merge_measurement_plan(previous_plan, appended_plan, len(active_table.idx))
            self._measurement_plan_cache = {id(self.active_measurements): (self.active_measurements, self._active_measurement_plan)}
        return True

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

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        if isinstance(measurements, list):
            return measurements
        return list(measurements)

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
        device_maps = {
            "DCNode": self.node_by_name,
            "DCBranch": self.branch_by_name,
            "DCBreak": self.break_by_name,
            "DCZeroBranch": self.zero_branch_by_name,
            "DCGenerator": self.generator_by_name,
            "DCLoad": self.load_by_name,
            "DCDCConverter": self.dcdc_by_name,
        }
        table = getattr(self.measurements, "table", None)
        table_valid = table.valid if table is not None and len(table.valid) == len(self.measurements) else None
        for pos, meas in enumerate(self.measurements):
            if not meas.valid or meas.weight <= 0.0:
                continue
            keep_valid = True
            if meas.device_type in ("DCSwitch", "DCSwitchConstraint"):
                keep_valid = False
            elif meas.device_type in ("DCZeroBranchConstraint", "DCBreakConstraint"):
                keep_valid = False
            else:
                devices = device_maps.get(meas.device_type)
                if devices is None or meas.device_name not in devices:
                    keep_valid = False
            if not keep_valid:
                meas.valid = False
                if table_valid is not None:
                    table_valid[pos] = False

    def _node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real DCNode voltage measurements keyed by DC node index."""
        best: Dict[int, Tuple[float, float]] = {}
        for meas in self.measurements:
            if (
                not meas.valid
                or meas.weight <= 0.0
                or meas.device_type != "DCNode"
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
        """Count live DC topology terminals used when choosing island voltage references."""
        degrees = {node.idx: 0 for node in self.nodes}
        device_groups = (
            self.branch_by_name.values(),
            self.zero_branch_by_name.values(),
            self.switch_by_name.values(),
            self.break_by_name.values(),
        )
        for devices in device_groups:
            for dev in devices:
                if dev.i_node in self.node_pos:
                    degrees[self.nodes[self.node_pos[dev.i_node]].idx] += 1
                if dev.j_node in self.node_pos:
                    degrees[self.nodes[self.node_pos[dev.j_node]].idx] += 1
        return degrees

    def _select_reference_nodes(self) -> List[object]:
        """Choose one measured high-degree DC voltage reference per live DC topology island."""
        references = []
        voltage_measurements = self.node_voltage_measurements
        degrees = self.node_degrees
        for island in self.network.islands:
            if not getattr(island, "is_alive", False):
                continue
            candidates = [
                node
                for node in island.buses
                if node.idx in self.node_by_idx and node.idx in voltage_measurements
            ]
            if candidates:
                references.append(
                    max(
                        candidates,
                        key=lambda node: (degrees.get(node.idx, 0), -int(node.idx)),
                    )
                )
            elif island.slack_nodes:
                references.append(sorted(island.slack_nodes, key=lambda item: item.idx)[0])
            elif island.buses:
                references.append(sorted(island.buses, key=lambda item: item.idx)[0])
        return references

    def _build_zero_tie_voltage_layout(self) -> None:
        """Compress DC voltage states across closed switches and zero branches."""
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

        for dev in [*self.zero_branches, *self.switches, *self.breakers]:
            if dev.i_node in self.node_pos and dev.j_node in self.node_pos:
                union(self.node_pos[dev.i_node], self.node_pos[dev.j_node])

        groups: Dict[int, List[int]] = {}
        for pos in range(n):
            groups.setdefault(find(pos), []).append(pos)
        components = [sorted(group) for group in groups.values()]
        components.sort(key=lambda group: group[0])
        self.zero_tie_components = components

        node_voltage_state = np.full(n, -1, dtype=np.int32)
        voltage_state_pos = []
        self.voltage_state_nodes: List[np.ndarray] = []
        self.ref_voltages: Dict[int, float] = {}
        reference_voltage_by_pos = {
            self.node_pos[node.idx]: self.node_voltage_measurements[node.idx]
            for node in self.references
            if node.idx in self.node_pos and node.idx in self.node_voltage_measurements
        }
        state_col = 0
        for component in components:
            component_array = np.asarray(component, dtype=np.int32)
            fixed_pos = next((int(pos) for pos in component if int(pos) in reference_voltage_by_pos), None)
            if fixed_pos is not None:
                fixed_voltage = max(float(reference_voltage_by_pos[fixed_pos]), self.voltage_floor)
                for pos in component:
                    self.ref_voltages[int(pos)] = fixed_voltage
                continue
            voltage_state_pos.append(component[0])
            self.voltage_state_nodes.append(component_array)
            for pos in component:
                node_voltage_state[pos] = state_col
            state_col += 1

        self.node_voltage_state = node_voltage_state
        self.voltage_state_pos = np.asarray(voltage_state_pos, dtype=np.int32)
        self.voltage_col = node_voltage_state.copy()
        self.n_voltage = int(self.voltage_state_pos.size)
        if self.voltage_state_nodes:
            self._voltage_expand_pos = np.concatenate(self.voltage_state_nodes).astype(np.int64, copy=False)
            self._voltage_expand_col = np.concatenate(
                [
                    np.full(nodes.size, state_col, dtype=np.int64)
                    for state_col, nodes in enumerate(self.voltage_state_nodes)
                ]
            )
        else:
            self._voltage_expand_pos = np.array([], dtype=np.int64)
            self._voltage_expand_col = np.array([], dtype=np.int64)
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
        self._apply_branch_devices = list(self.branch_by_name.values())
        self._apply_branch_i = self._int_array([self.node_pos[br.i_node] for br in self._apply_branch_devices])
        self._apply_branch_j = self._int_array([self.node_pos[br.j_node] for br in self._apply_branch_devices])
        self._apply_branch_inv_r = np.asarray(
            [1.0 / float(br.r) for br in self._apply_branch_devices],
            dtype=np.float64,
        )

        self._apply_load_devices = list(self.load_by_name.values())
        self._apply_load_pos = self._int_array([self.node_pos[load.node] for load in self._apply_load_devices])
        self._apply_load_pv0 = np.asarray([getattr(load, "pbase", 1.0) * load.pv0 for load in self._apply_load_devices], dtype=np.float64)
        self._apply_load_pv1 = np.asarray([getattr(load, "pbase", 1.0) * load.pv1 for load in self._apply_load_devices], dtype=np.float64)
        self._apply_load_pv2 = np.asarray([getattr(load, "pbase", 1.0) * load.pv2 for load in self._apply_load_devices], dtype=np.float64)

        gen_ctrl_map = {"V": 0, "P": 1, "I": 2}
        self._apply_generator_devices = list(self.generator_by_name.values())
        self._apply_generator_pos = self._int_array([self.node_pos[gen.node] for gen in self._apply_generator_devices])
        generator_ctrl = []
        generator_v_pos = []
        for gen in self._apply_generator_devices:
            ctrl = gen_ctrl_map.get(gen.control_type)
            if ctrl is None:
                raise RuntimeError(f"Unsupported DCGenerator control type: {gen.control_type}")
            generator_ctrl.append(ctrl)
            generator_v_pos.append(self.v_generator_pos[gen.name] if gen.control_type == "V" else -1)
        self._apply_generator_ctrl = self._int_array(generator_ctrl)
        self._apply_generator_v_pos = self._int_array(generator_v_pos)
        self._apply_generator_p_set = np.asarray(
            [gen.p_set for gen in self._apply_generator_devices],
            dtype=np.float64,
        )
        self._apply_generator_i_set = np.asarray(
            [gen.i_set for gen in self._apply_generator_devices],
            dtype=np.float64,
        )

        self._apply_switch_devices = []
        self._apply_switch_i = self._int_array([])
        self._apply_switch_j = self._int_array([])
        self._apply_switch_pos = self._int_array([])

        self._apply_break_devices = list(self.break_by_name.values())
        self._apply_break_i = self._int_array([self.node_pos[brk.i_node] for brk in self._apply_break_devices])
        self._apply_break_j = self._int_array([self.node_pos[brk.j_node] for brk in self._apply_break_devices])
        self._apply_break_pos = self._int_array([self.break_pos[brk.name] for brk in self._apply_break_devices])

        self._apply_zero_branch_devices = [
            zbr for zbr in self.zero_branch_by_name.values() if zbr.name in self.zero_branch_pos
        ]
        self._apply_zero_branch_i = self._int_array([self.node_pos[zbr.i_node] for zbr in self._apply_zero_branch_devices])
        self._apply_zero_branch_j = self._int_array([self.node_pos[zbr.j_node] for zbr in self._apply_zero_branch_devices])
        self._apply_zero_branch_pos = self._int_array([self.zero_branch_pos[zbr.name] for zbr in self._apply_zero_branch_devices])

        self._apply_dcdc_devices = list(self.dcdc_by_name.values())
        self._apply_dcdc_i = self._int_array([self.node_pos[conv.i_node] for conv in self._apply_dcdc_devices])
        self._apply_dcdc_j = self._int_array([self.node_pos[conv.j_node] for conv in self._apply_dcdc_devices])
        self._apply_dcdc_pos = self._int_array([self.dcdc_pos[conv.name] for conv in self._apply_dcdc_devices])

    def _build_measurement_plan_device_cache(self) -> None:
        """Cache per-device row-plan metadata shared by all measurement types."""
        self._node_plan_by_name = {
            name: (self.node_pos[node.idx], int(self.voltage_col[self.node_pos[node.idx]]))
            for name, node in self.node_by_name.items()
        }
        self._branch_plan_by_name = {}
        for br in self.branch_by_name.values():
            i = self.node_pos[br.i_node]
            j = self.node_pos[br.j_node]
            self._branch_plan_by_name[br.name] = (
                i,
                j,
                int(self.voltage_col[i]),
                int(self.voltage_col[j]),
                1.0 / br.r,
            )
        self._load_plan_by_name = {}
        for load in self.load_by_name.values():
            pos = self.node_pos[load.node]
            self._load_plan_by_name[load.name] = (
                pos,
                int(self.voltage_col[pos]),
                getattr(load, "pbase", 1.0) * load.pv0,
                getattr(load, "pbase", 1.0) * load.pv1,
                getattr(load, "pbase", 1.0) * load.pv2,
            )
        self._generator_plan_by_name = {}
        for gen in self.generator_by_name.values():
            ctrl = _GEN_CONTROL_KIND.get(gen.control_type)
            if ctrl is None:
                continue
            pos = self.node_pos[gen.node]
            vgen_pos = self.v_generator_pos[gen.name] if gen.control_type == "V" else -1
            power_col = self.v_generator_start + vgen_pos if vgen_pos >= 0 else -1
            self._generator_plan_by_name[gen.name] = (
                ctrl,
                pos,
                int(self.voltage_col[pos]),
                power_col,
                vgen_pos,
                gen.p_set,
                gen.i_set,
            )
        self._switch_plan_by_name = {}
        self._break_plan_by_name = {}
        for brk in self.break_by_name.values():
            i = self.node_pos[brk.i_node]
            j = self.node_pos[brk.j_node]
            current_pos = self.break_pos[brk.name]
            self._break_plan_by_name[brk.name] = (
                i,
                j,
                int(self.voltage_col[i]),
                int(self.voltage_col[j]),
                self.switch_start + current_pos,
                current_pos,
            )
        self._zero_branch_plan_by_name = {}
        for zbr in self.zero_branch_by_name.values():
            i = self.node_pos[zbr.i_node]
            j = self.node_pos[zbr.j_node]
            current_pos = self.zero_branch_pos.get(zbr.name, -1)
            self._zero_branch_plan_by_name[zbr.name] = (
                i,
                j,
                int(self.voltage_col[i]),
                int(self.voltage_col[j]),
                self.switch_start + current_pos if current_pos >= 0 else -1,
                current_pos,
            )
        self._constraint_plan_by_name = {
            name: (
                self.node_pos[dev.i_node],
                self.node_pos[dev.j_node],
                int(self.voltage_col[self.node_pos[dev.i_node]]),
                int(self.voltage_col[self.node_pos[dev.j_node]]),
            )
            for name, dev in {**self.zero_branch_by_name, **self.switch_by_name, **self.break_by_name}.items()
        }
        self._dcdc_plan_by_name = {}
        for conv in self.dcdc_by_name.values():
            conv_pos = self.dcdc_pos[conv.name]
            p_col = self.dcdc_start + 2 * conv_pos
            i = self.node_pos[conv.i_node]
            j = self.node_pos[conv.j_node]
            self._dcdc_plan_by_name[conv.name] = (
                i,
                j,
                int(self.voltage_col[i]),
                int(self.voltage_col[j]),
                p_col,
                p_col + 1,
                conv_pos,
            )
        self._build_measurement_plan_lookup_arrays()

    def _build_measurement_plan_lookup_arrays(self) -> None:
        node_items = list(self._node_plan_by_name.items())
        self._node_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(node_items)}
        self._node_plan_node_pos = self._int_array([plan[0] for _, plan in node_items])
        self._node_plan_col = self._int_array([plan[1] for _, plan in node_items])

        branch_items = list(self._branch_plan_by_name.items())
        self._branch_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(branch_items)}
        self._branch_plan_i = self._int_array([plan[0] for _, plan in branch_items])
        self._branch_plan_j = self._int_array([plan[1] for _, plan in branch_items])
        self._branch_plan_i_col = self._int_array([plan[2] for _, plan in branch_items])
        self._branch_plan_j_col = self._int_array([plan[3] for _, plan in branch_items])
        self._branch_plan_inv_r = np.asarray([plan[4] for _, plan in branch_items], dtype=np.float64)

        load_items = list(self._load_plan_by_name.items())
        self._load_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(load_items)}
        self._load_plan_pos = self._int_array([plan[0] for _, plan in load_items])
        self._load_plan_col = self._int_array([plan[1] for _, plan in load_items])
        self._load_plan_pv0 = np.asarray([plan[2] for _, plan in load_items], dtype=np.float64)
        self._load_plan_pv1 = np.asarray([plan[3] for _, plan in load_items], dtype=np.float64)
        self._load_plan_pv2 = np.asarray([plan[4] for _, plan in load_items], dtype=np.float64)

        gen_items = list(self._generator_plan_by_name.items())
        self._generator_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(gen_items)}
        self._generator_plan_ctrl = self._int_array([plan[0] for _, plan in gen_items])
        self._generator_plan_pos = self._int_array([plan[1] for _, plan in gen_items])
        self._generator_plan_col = self._int_array([plan[2] for _, plan in gen_items])
        self._generator_plan_p_col = self._int_array([plan[3] for _, plan in gen_items])
        self._generator_plan_vgen_pos = self._int_array([plan[4] for _, plan in gen_items])
        self._generator_plan_p_set = np.asarray([plan[5] for _, plan in gen_items], dtype=np.float64)
        self._generator_plan_i_set = np.asarray([plan[6] for _, plan in gen_items], dtype=np.float64)

        break_items = list(self._break_plan_by_name.items())
        self._break_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(break_items)}
        self._break_plan_i = self._int_array([plan[0] for _, plan in break_items])
        self._break_plan_j = self._int_array([plan[1] for _, plan in break_items])
        self._break_plan_i_col = self._int_array([plan[2] for _, plan in break_items])
        self._break_plan_j_col = self._int_array([plan[3] for _, plan in break_items])
        self._break_plan_current_col = self._int_array([plan[4] for _, plan in break_items])
        self._break_plan_current_pos = self._int_array([plan[5] for _, plan in break_items])

        zero_branch_items = list(self._zero_branch_plan_by_name.items())
        self._zero_branch_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(zero_branch_items)}
        self._zero_branch_plan_i = self._int_array([plan[0] for _, plan in zero_branch_items])
        self._zero_branch_plan_j = self._int_array([plan[1] for _, plan in zero_branch_items])
        self._zero_branch_plan_i_col = self._int_array([plan[2] for _, plan in zero_branch_items])
        self._zero_branch_plan_j_col = self._int_array([plan[3] for _, plan in zero_branch_items])
        self._zero_branch_plan_current_col = self._int_array([plan[4] for _, plan in zero_branch_items])
        self._zero_branch_plan_current_pos = self._int_array([plan[5] for _, plan in zero_branch_items])

        constraint_items = list(self._constraint_plan_by_name.items())
        self._constraint_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(constraint_items)}
        self._constraint_plan_i = self._int_array([plan[0] for _, plan in constraint_items])
        self._constraint_plan_j = self._int_array([plan[1] for _, plan in constraint_items])
        self._constraint_plan_i_col = self._int_array([plan[2] for _, plan in constraint_items])
        self._constraint_plan_j_col = self._int_array([plan[3] for _, plan in constraint_items])

        dcdc_items = list(self._dcdc_plan_by_name.items())
        self._dcdc_plan_name_to_pos = {name: pos for pos, (name, _) in enumerate(dcdc_items)}
        self._dcdc_plan_i = self._int_array([plan[0] for _, plan in dcdc_items])
        self._dcdc_plan_j = self._int_array([plan[1] for _, plan in dcdc_items])
        self._dcdc_plan_i_col = self._int_array([plan[2] for _, plan in dcdc_items])
        self._dcdc_plan_j_col = self._int_array([plan[3] for _, plan in dcdc_items])
        self._dcdc_plan_p_col = self._int_array([plan[4] for _, plan in dcdc_items])
        self._dcdc_plan_q_col = self._int_array([plan[5] for _, plan in dcdc_items])
        self._dcdc_plan_pos = self._int_array([plan[6] for _, plan in dcdc_items])

        self._measurement_plan_device_pos_by_type_code = {
            _DEVICE_TYPE_CODES["DCNode"]: self._node_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCBranch"]: self._branch_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCLoad"]: self._load_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCGenerator"]: self._generator_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCZeroBranch"]: self._zero_branch_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCBreak"]: self._break_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCZeroBranchConstraint"]: self._constraint_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCBreakConstraint"]: self._constraint_plan_name_to_pos,
            _DEVICE_TYPE_CODES["DCDCConverter"]: self._dcdc_plan_name_to_pos,
        }
        self._measurement_plan_meas_kind_by_type_code = {
            _DEVICE_TYPE_CODES["DCNode"]: {"V": 0},
            _DEVICE_TYPE_CODES["DCBranch"]: _TERMINAL_KIND,
            _DEVICE_TYPE_CODES["DCLoad"]: _LOAD_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["DCGenerator"]: _GEN_MEASUREMENT_KIND,
            _DEVICE_TYPE_CODES["DCZeroBranch"]: _TERMINAL_KIND,
            _DEVICE_TYPE_CODES["DCBreak"]: _TERMINAL_KIND,
            _DEVICE_TYPE_CODES["DCZeroBranchConstraint"]: {"V_DIFF": 0},
            _DEVICE_TYPE_CODES["DCBreakConstraint"]: {"V_DIFF": 0},
            _DEVICE_TYPE_CODES["DCDCConverter"]: _TERMINAL_KIND,
        }

    def _ensure_measurement_plan_lookup_arrays(self) -> None:
        if not hasattr(self, "_measurement_plan_device_pos_by_type_code"):
            self._build_measurement_plan_lookup_arrays()

    @staticmethod
    def _load_network(e_file: Path) -> DCPowerNetwork:
        """Read the DC case and build topology references used by measurements."""
        network = build_dc_network_from_ppc(load_dc_ppc_from_e_file(e_file))
        network.topo()
        return network

    @staticmethod
    def _run_power_flow_seed(network: DCPowerNetwork, params: StateEstimationParameters, e_file: Path) -> bool:
        original_ppc = getattr(network, "ppc", None)
        seed_tol = max(float(params.power_flow_tol), 1e-6)
        ppc = DCStateEstimator._power_flow_seed_ppc_from_network(network)
        if ppc is not None:
            network.ppc = ppc
            calc = DCPowerFlowCalc(network)
            calc.skip_lf_result = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()):
                        rc = calc.run(
                            tol=seed_tol,
                            max_iter=params.power_flow_max_iter,
                            min_voltage=params.power_flow_min_voltage,
                            result_mode="none",
                        )
            except Exception:
                if original_ppc is not None:
                    network.ppc = original_ppc
                return False
            if rc != 0 or not calc.converged:
                if original_ppc is not None:
                    network.ppc = original_ppc
                return False
            DCStateEstimator._apply_power_flow_seed_calc_state_to_network(network, ppc, calc)
            return True

        snapshot = DCStateEstimator._capture_power_flow_seed_snapshot(network)
        calc = DCPowerFlowCalc(network)
        calc.skip_lf_result = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = calc.run(
                        tol=seed_tol,
                        max_iter=params.power_flow_max_iter,
                        min_voltage=params.power_flow_min_voltage,
                    )
        except Exception:
            DCStateEstimator._restore_power_flow_seed_snapshot(snapshot)
            return False
        ok = bool(rc == 0 and calc.converged)
        if not ok:
            DCStateEstimator._restore_power_flow_seed_snapshot(snapshot)
            return False
        return ok

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
        if seed_rows is None:
            DCStateEstimator._sync_dc_network_to_ppc(network, ppc)
        else:
            DCStateEstimator._apply_power_flow_seed_rows_to_ppc(ppc, seed_rows)
        return ppc

    @staticmethod
    def _row_by_idx(array: np.ndarray, idx_col: int) -> Dict[int, int]:
        if array is None or array.size == 0:
            return {}
        return {int(row[idx_col]): pos for pos, row in enumerate(array)}

    @staticmethod
    def _row_by_name(names) -> Dict[str, int]:
        if names is None:
            return {}
        return {str(name): pos for pos, name in enumerate(names)}

    @staticmethod
    def _apply_power_flow_seed_rows_to_ppc(ppc, seed_rows) -> None:
        bus = ppc.get("bus")
        if bus is None:
            return
        bus_by_name = DCStateEstimator._row_by_name(ppc.get("bus_name", ()))
        bus_by_idx = DCStateEstimator._row_by_idx(bus, DC_BUS_COLS["idx"])
        gen = ppc.get("gen")
        gen_by_name = DCStateEstimator._row_by_name(ppc.get("gen_name", ()))
        load = ppc.get("load")
        load_by_name = DCStateEstimator._row_by_name(ppc.get("load_name", ()))
        dcdc = ppc.get("dcdc")
        dcdc_by_name = DCStateEstimator._row_by_name(ppc.get("dcdc_name", ()))

        def set_bus_voltage_by_idx(node_idx, value):
            row = bus_by_idx.get(int(node_idx))
            if row is not None:
                bus[row, DC_BUS_COLS["voltage"]] = max(float(value), 0.0)

        for device_type, device_name, meas_type, value in seed_rows:
            value = float(value)
            if device_type == "DCNode":
                if meas_type == "V":
                    row = bus_by_name.get(str(device_name))
                    if row is not None:
                        bus[row, DC_BUS_COLS["voltage"]] = max(value, 0.0)
                continue
            if device_type == "DCGenerator" and gen is not None:
                row = gen_by_name.get(str(device_name))
                if row is None:
                    continue
                if meas_type == "P_GEN":
                    gen[row, DC_GEN_COLS["p_set"]] = value
                    gen[row, DC_GEN_COLS["p"]] = value
                elif meas_type == "V_GEN":
                    voltage = max(value, 0.0)
                    gen[row, DC_GEN_COLS["v_set"]] = voltage
                    set_bus_voltage_by_idx(gen[row, DC_GEN_COLS["node"]], voltage)
                elif meas_type == "I_GEN":
                    gen[row, DC_GEN_COLS["i_set"]] = value
                    gen[row, DC_GEN_COLS["current"]] = value
                continue
            if device_type == "DCLoad" and load is not None:
                row = load_by_name.get(str(device_name))
                if row is None:
                    continue
                if meas_type == "P_LOAD":
                    load[row, DC_LOAD_COLS["pbase"]] = 1.0
                    load[row, DC_LOAD_COLS["pv0"]] = value
                    load[row, DC_LOAD_COLS["pv1"]] = 0.0
                    load[row, DC_LOAD_COLS["pv2"]] = 0.0
                    load[row, DC_LOAD_COLS["p"]] = value
                elif meas_type == "V_LOAD":
                    set_bus_voltage_by_idx(load[row, DC_LOAD_COLS["node"]], value)
                elif meas_type == "I_LOAD":
                    load[row, DC_LOAD_COLS["current"]] = value
                continue
            if device_type == "DCDCConverter" and dcdc is not None:
                row = dcdc_by_name.get(str(device_name))
                if row is None:
                    continue
                if meas_type in ("P_FROM", "P_DC", "P_IN"):
                    dcdc[row, DC_DCDC_COLS["p_set"]] = value
                    dcdc[row, DC_DCDC_COLS["i_p"]] = value
                elif meas_type in ("P_TO", "P_OUT"):
                    dcdc[row, DC_DCDC_COLS["j_p"]] = value
                elif meas_type == "V_FROM":
                    set_bus_voltage_by_idx(dcdc[row, DC_DCDC_COLS["i_node"]], value)
                elif meas_type == "V_TO":
                    set_bus_voltage_by_idx(dcdc[row, DC_DCDC_COLS["j_node"]], value)
                elif meas_type in ("I_FROM", "I_DC"):
                    dcdc[row, DC_DCDC_COLS["i_set"]] = value
                    dcdc[row, DC_DCDC_COLS["i_c"]] = value
                elif meas_type in ("I_TO", "I_OUT"):
                    dcdc[row, DC_DCDC_COLS["j_c"]] = value

    @staticmethod
    def _sync_dc_network_to_ppc(network, ppc) -> None:
        row_by_idx = DCStateEstimator._row_by_idx
        bus = ppc["bus"]
        bus_rows = row_by_idx(bus, DC_BUS_COLS["idx"])
        for node in getattr(network, "nodes", []) or []:
            row = bus_rows.get(int(getattr(node, "idx", -1)))
            if row is None:
                continue
            bus[row, DC_BUS_COLS["voltage"]] = float(getattr(node, "voltage", 1.0) or 1.0)
            bus[row, DC_BUS_COLS["run_stat"]] = int(getattr(node, "run_stat", 1))

        def sync_devices(devices, key, cols, attrs):
            array = ppc.get(key)
            if array is None or array.size == 0:
                return
            rows = row_by_idx(array, cols["idx"])
            for dev in devices or []:
                row = rows.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    continue
                for attr, col_name in attrs:
                    if col_name in cols and hasattr(dev, attr):
                        value = getattr(dev, attr)
                        if value is not None:
                            array[row, cols[col_name]] = value

        sync_devices(
            getattr(network, "branches", []),
            "branch",
            DC_BRANCH_COLS,
            (("run_stat", "run_stat"), ("i_p", "i_p"), ("j_p", "j_p"), ("current", "current")),
        )
        sync_devices(
            getattr(network, "loads", []),
            "load",
            DC_LOAD_COLS,
            (
                ("pbase", "pbase"),
                ("pv0", "pv0"),
                ("pv1", "pv1"),
                ("pv2", "pv2"),
                ("run_stat", "run_stat"),
                ("p", "p"),
                ("current", "current"),
            ),
        )
        sync_devices(
            getattr(network, "generators", []),
            "gen",
            DC_GEN_COLS,
            (
                ("p_set", "p_set"),
                ("v_set", "v_set"),
                ("i_set", "i_set"),
                ("run_stat", "run_stat"),
                ("p", "p"),
                ("current", "current"),
            ),
        )
        sync_devices(
            getattr(network, "zero_branches", []),
            "zero_branch",
            DC_ZERO_BRANCH_COLS,
            (("run_stat", "run_stat"), ("p", "p"), ("current", "current")),
        )
        switch_attrs = (("status", "status"), ("run_stat", "run_stat"), ("p", "p"), ("current", "current"))
        sync_devices(getattr(network, "switches", []), "switch", DC_SWITCH_COLS, switch_attrs)
        sync_devices(getattr(network, "breakers", []), "break", DC_BREAK_COLS, switch_attrs)
        sync_devices(
            getattr(network, "dcdc_converters", []),
            "dcdc",
            DC_DCDC_COLS,
            (
                ("p_set", "p_set"),
                ("i_set", "i_set"),
                ("v_set", "v_set"),
                ("run_stat", "run_stat"),
                ("i_p", "i_p"),
                ("j_p", "j_p"),
                ("i_c", "i_c"),
                ("j_c", "j_c"),
            ),
        )

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
                bus[:, DC_BUS_COLS["voltage"]] = voltage
            network.ppc = ppc
            if hasattr(network, "_array_model"):
                network._array_model = ppc
            row_by_idx = DCStateEstimator._row_by_idx
            bus_rows = row_by_idx(bus, DC_BUS_COLS["idx"]) if isinstance(bus, np.ndarray) else {}
            for node in getattr(network, "nodes", []) or []:
                row = bus_rows.get(int(getattr(node, "idx", -1)))
                if row is not None:
                    node.voltage = float(bus[row, DC_BUS_COLS["voltage"]])
            for bus_obj in getattr(network, "buses", []) or []:
                members = getattr(bus_obj, "nodes", ()) or ()
                if members:
                    bus_obj.voltage = float(getattr(members[0], "voltage", 0.0) or 0.0)
            return
        for bus_obj in getattr(network, "buses", []) or []:
            members = getattr(bus_obj, "nodes", ()) or ()
            if members:
                bus_obj.voltage = float(getattr(members[0], "voltage", 0.0) or 0.0)

    @staticmethod
    def _capture_power_flow_seed_snapshot(network):
        attrs = (
            "voltage",
            "p",
            "current",
            "i_p",
            "i_c",
            "j_p",
            "j_c",
            "p_set",
            "v_set",
            "i_set",
            "pbase",
            "pv0",
            "pv1",
            "pv2",
        )
        snapshot = []
        collections = (
            "nodes",
            "buses",
            "branches",
            "zero_branches",
            "switches",
            "breakers",
            "generators",
            "loads",
            "dcdc_converters",
        )
        seen = set()
        for collection in collections:
            for obj in getattr(network, collection, []) or []:
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                state = {attr: getattr(obj, attr) for attr in attrs if hasattr(obj, attr)}
                if state:
                    snapshot.append((obj, state))
        return snapshot

    @staticmethod
    def _restore_power_flow_seed_snapshot(snapshot) -> None:
        for obj, state in snapshot:
            for attr, value in state.items():
                setattr(obj, attr, value)

    @staticmethod
    def _set_existing_attr(obj, attr: str, value) -> None:
        if hasattr(obj, attr):
            setattr(obj, attr, value)

    def _set_node_voltage_object(self, node, value: float) -> None:
        voltage = max(float(value), self.voltage_floor)
        if node is None:
            return
        if hasattr(node, "voltage"):
            node.voltage = voltage
        for member in getattr(node, "nodes", ()) or ():
            if hasattr(member, "voltage"):
                member.voltage = voltage

    def _set_node_voltage_by_idx(self, node_idx: int, value: float) -> None:
        targets = []
        bus = self.node_by_idx.get(int(node_idx))
        if bus is not None:
            targets.append(bus)
        raw = getattr(self.network, "node_dict", {}).get(int(node_idx))
        if raw is not None:
            targets.append(raw)
        seen = set()
        for target in targets:
            if id(target) in seen:
                continue
            seen.add(id(target))
            self._set_node_voltage_object(target, value)

    def _set_node_voltage_by_name(self, node_name: str, value: float) -> None:
        bus = self.node_by_name.get(node_name)
        if bus is not None:
            self._set_node_voltage_object(bus, value)
            return
        for node in getattr(self.network, "nodes", []):
            if getattr(node, "name", None) == node_name:
                self._set_node_voltage_object(node, value)
                bus_obj = getattr(node, "bus_obj", None)
                if bus_obj is not None:
                    self._set_node_voltage_object(bus_obj, value)
                return

    def _sync_bus_state_from_members(self) -> None:
        for bus in self.nodes:
            members = getattr(bus, "nodes", ()) or ()
            if not members:
                continue
            member = next((node for node in members if getattr(node, "voltage", None) is not None), None)
            if member is None:
                continue
            if hasattr(bus, "voltage"):
                bus.voltage = float(getattr(member, "voltage", 1.0) or 1.0)

    def _apply_measurement_seed_to_network(self, *, force_object: bool = False) -> None:
        """Apply valid normalized measurements to network fields used by the LF seed."""
        seed_rows = getattr(self, "_power_flow_seed_rows", None)
        if seed_rows is not None:
            seed_rows = tuple(seed_rows)
            setattr(self.network, "_se_power_flow_seed_rows", seed_rows)
            run_seed = getattr(type(self), "_run_power_flow_seed", None)
            original_run_seed = globals().get("_ORIGINAL_DC_RUN_POWER_FLOW_SEED")
            if original_run_seed is not None and run_seed is not original_run_seed:
                force_object = True
            source = getattr(self.network, "ppc", None)
            if not (isinstance(source, dict) and source.get("format") == "dc_ppc_v1"):
                source = getattr(self.network, "_array_model", None)
            if not force_object and isinstance(source, dict) and source.get("format") == "dc_ppc_v1":
                return
            for device_type, device_name, meas_type, value in seed_rows:
                self._apply_power_flow_seed_row(device_type, device_name, meas_type, float(value))
            return
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            self._apply_power_flow_seed_row(
                meas.device_type,
                meas.device_name,
                meas.meas_type,
                float(meas.value),
            )

    def _apply_power_flow_seed_row(self, device_type: str, device_name: str, meas_type: str, value: float) -> None:
        if device_type == "DCNode":
            if meas_type == "V":
                self._set_node_voltage_by_name(device_name, value)
            return
        if device_type == "DCGenerator":
            gen = self.generator_by_name.get(device_name)
            if gen is None:
                return
            if meas_type == "P_GEN":
                self._set_existing_attr(gen, "p_set", value)
                self._set_existing_attr(gen, "p", value)
            elif meas_type == "V_GEN":
                voltage = max(value, self.voltage_floor)
                self._set_existing_attr(gen, "v_set", voltage)
                self._set_node_voltage_by_idx(gen.node, voltage)
            elif meas_type == "I_GEN":
                self._set_existing_attr(gen, "i_set", value)
                self._set_existing_attr(gen, "current", value)
            return
        if device_type == "DCLoad":
            load = self.load_by_name.get(device_name)
            if load is None:
                return
            if meas_type == "P_LOAD":
                self._set_existing_attr(load, "pbase", 1.0)
                self._set_existing_attr(load, "pv0", value)
                self._set_existing_attr(load, "pv1", 0.0)
                self._set_existing_attr(load, "pv2", 0.0)
                self._set_existing_attr(load, "p", value)
            elif meas_type == "V_LOAD":
                self._set_node_voltage_by_idx(load.node, value)
            elif meas_type == "I_LOAD":
                self._set_existing_attr(load, "current", value)
            return
        if device_type == "DCDCConverter":
            conv = self.dcdc_by_name.get(device_name)
            if conv is None:
                return
            if meas_type in ("P_FROM", "P_DC", "P_IN"):
                self._set_existing_attr(conv, "p_set", value)
                self._set_existing_attr(conv, "i_p", value)
            elif meas_type in ("P_TO", "P_OUT"):
                self._set_existing_attr(conv, "j_p", value)
            elif meas_type == "V_FROM":
                self._set_node_voltage_by_idx(conv.i_node, value)
            elif meas_type == "V_TO":
                self._set_node_voltage_by_idx(conv.j_node, value)
            elif meas_type in ("I_FROM", "I_DC"):
                self._set_existing_attr(conv, "i_set", value)
                self._set_existing_attr(conv, "i_c", value)
            elif meas_type in ("I_TO", "I_OUT"):
                self._set_existing_attr(conv, "j_c", value)

    @staticmethod
    def _load_measurements(meas_file: Path) -> MeasurementList:
        return _read_measurements_direct(meas_file)

    def _voltage_base(self, node_idx: int) -> float:
        return self._node_vbase_by_idx[int(node_idx)]

    def _node_current_base(self, node_idx: int) -> float:
        return self._current_file_base_by_idx[int(node_idx)]

    def _build_unit_scale_cache(self) -> None:
        """Cache per-node unit bases used by measurement conversion."""
        self._node_vbase_by_idx = {
            int(node.idx): float(node.vbase)
            for node in self.network.nodes
        }
        self._voltage_file_base_by_idx = {
            node_idx: self.u_scale * vbase
            for node_idx, vbase in self._node_vbase_by_idx.items()
        }
        self._current_file_base_by_idx = {
            node_idx: self.i_scale * dc_current_base_ka(self.p_base_kW, vbase)
            for node_idx, vbase in self._node_vbase_by_idx.items()
        }

    def _terminal_scale_tuple(self, i_node: int, j_node: int) -> Tuple[float, float, float, float, float, float]:
        """Return scale values ordered by _TERMINAL_KIND."""
        return (
            self.p_base,
            self._voltage_file_base_by_idx[int(i_node)],
            self._current_file_base_by_idx[int(i_node)],
            self.p_base,
            self._voltage_file_base_by_idx[int(j_node)],
            self._current_file_base_by_idx[int(j_node)],
        )

    def _build_measurement_scale_cache(self) -> None:
        """Cache file-unit scale factors used by the measurement normalization pass."""
        self._node_measurement_scale_by_name = {
            name: self._voltage_file_base_by_idx[int(node.idx)]
            for name, node in self.node_by_name.items()
        }
        self._branch_measurement_scale_by_name = {
            br.name: self._terminal_scale_tuple(br.i_node, br.j_node)
            for br in self.branch_by_name.values()
        }
        self._break_measurement_scale_by_name = {
            brk.name: self._terminal_scale_tuple(brk.i_node, brk.j_node)
            for brk in self.break_by_name.values()
        }
        self._zero_branch_measurement_scale_by_name = {
            zbr.name: self._terminal_scale_tuple(zbr.i_node, zbr.j_node)
            for zbr in self.zero_branch_by_name.values()
        }
        self._dcdc_measurement_scale_by_name = {
            conv.name: self._terminal_scale_tuple(conv.i_node, conv.j_node)
            for conv in self.dcdc_by_name.values()
        }
        self._generator_measurement_scale_by_name = {
            gen.name: (
                self.p_base,
                self._voltage_file_base_by_idx[int(gen.node)],
                self._current_file_base_by_idx[int(gen.node)],
            )
            for gen in self.generator_by_name.values()
        }
        self._load_measurement_scale_by_name = {
            load.name: (
                self.p_base,
                self._voltage_file_base_by_idx[int(load.node)],
                self._current_file_base_by_idx[int(load.node)],
            )
            for load in self.load_by_name.values()
        }
        self._constraint_measurement_scale_by_name = {
            name: self._voltage_file_base_by_idx[int(dev.i_node)]
            for name, dev in {**self.zero_branch_by_name, **self.switch_by_name, **self.break_by_name}.items()
        }

    def _convert_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal state-estimation units."""
        if getattr(self.measurements, "normalized", False):
            seed_rows = []
            for meas in self.measurements:
                if not meas.valid or meas.weight <= 0.0:
                    continue
                mtype = meas.meas_type
                if (
                    (meas.device_type == "DCNode" and mtype == "V")
                    or (meas.device_type == "DCGenerator" and mtype in ("P_GEN", "V_GEN", "I_GEN"))
                    or (meas.device_type == "DCLoad" and mtype in ("P_LOAD", "V_LOAD", "I_LOAD"))
                    or (
                        meas.device_type == "DCDCConverter"
                        and mtype
                        in (
                            "P_FROM",
                            "P_DC",
                            "P_IN",
                            "P_TO",
                            "P_OUT",
                            "V_FROM",
                            "V_TO",
                            "I_FROM",
                            "I_DC",
                            "I_TO",
                            "I_OUT",
                        )
                    )
                ):
                    seed_rows.append((meas.device_type, meas.device_name, mtype, float(meas.value)))
            self._power_flow_seed_rows = seed_rows
            return
        terminal_kind = _TERMINAL_KIND.get
        load_kind = _LOAD_MEASUREMENT_KIND.get
        gen_kind = _GEN_MEASUREMENT_KIND.get
        node_scale = self._node_measurement_scale_by_name.__getitem__
        branch_scale = self._branch_measurement_scale_by_name.__getitem__
        break_scale = self._break_measurement_scale_by_name.__getitem__
        zero_branch_scale = self._zero_branch_measurement_scale_by_name.__getitem__
        dcdc_scale = self._dcdc_measurement_scale_by_name.__getitem__
        gen_scale = self._generator_measurement_scale_by_name.__getitem__
        load_scale = self._load_measurement_scale_by_name.__getitem__
        constraint_scale = self._constraint_measurement_scale_by_name.__getitem__
        seed_rows = []
        table = getattr(self.measurements, "table", None)
        table_value = table.value if table is not None and len(table.value) == len(self.measurements) else None

        for pos, meas in enumerate(self.measurements):
            if not meas.valid or meas.weight <= 0.0:
                continue
            scale = 1.0
            mtype = meas.meas_type
            device_name = meas.device_name
            if meas.device_type == "DCNode":
                if mtype == "V":
                    scale = node_scale(device_name)
            elif meas.device_type == "DCBranch":
                kind = terminal_kind(mtype)
                if kind is not None:
                    scale = branch_scale(device_name)[kind]
            elif meas.device_type == "DCBreak":
                kind = terminal_kind(mtype)
                if kind is not None:
                    scale = break_scale(device_name)[kind]
            elif meas.device_type == "DCZeroBranch":
                kind = terminal_kind(mtype)
                if kind is not None:
                    scale = zero_branch_scale(device_name)[kind]
            elif meas.device_type == "DCZeroBranchConstraint":
                if mtype == "V_DIFF":
                    scale = constraint_scale(device_name)
            elif meas.device_type == "DCBreakConstraint":
                if mtype == "V_DIFF":
                    scale = constraint_scale(device_name)
            elif meas.device_type == "DCDCConverter":
                kind = terminal_kind(mtype)
                if kind is not None:
                    scale = dcdc_scale(device_name)[kind]
            elif meas.device_type == "DCGenerator":
                kind = gen_kind(mtype)
                if kind is not None:
                    scale = gen_scale(device_name)[kind]
            elif meas.device_type == "DCLoad":
                kind = load_kind(mtype)
                if kind is not None:
                    scale = load_scale(device_name)[kind]
            meas.value = float(meas.value) / scale
            if table_value is not None:
                table_value[pos] = meas.value
            if (
                (meas.device_type == "DCNode" and mtype == "V")
                or (meas.device_type == "DCGenerator" and mtype in ("P_GEN", "V_GEN", "I_GEN"))
                or (meas.device_type == "DCLoad" and mtype in ("P_LOAD", "V_LOAD", "I_LOAD"))
                or (
                    meas.device_type == "DCDCConverter"
                    and mtype
                    in (
                        "P_FROM",
                        "P_DC",
                        "P_IN",
                        "P_TO",
                        "P_OUT",
                        "V_FROM",
                        "V_TO",
                        "I_FROM",
                        "I_DC",
                        "I_TO",
                        "I_OUT",
                    )
                )
            ):
                seed_rows.append((meas.device_type, device_name, mtype, float(meas.value)))
        self._power_flow_seed_rows = seed_rows

    def _active_device_keys(self) -> set:
        """Return devices that already have at least one usable real measurement."""
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
    def _generator_pseudo_power(gen) -> float:
        # Prefer solved output when available; otherwise infer from the control mode.
        p = getattr(gen, "p", None)
        if p is not None:
            return float(p)
        control_type = str(getattr(gen, "control_type", "")).upper()
        if control_type == "I":
            node = getattr(gen, "node_obj", None)
            voltage = float(getattr(node, "voltage", 1.0) or 1.0)
            return float(getattr(gen, "i_set", 0.0) or 0.0) * voltage
        return float(getattr(gen, "p_set", 0.0) or 0.0)

    @staticmethod
    def _load_pseudo_power(load) -> float:
        # Loads may be voltage dependent, so evaluate the ZIP model at the current seed voltage.
        p = getattr(load, "p", None)
        if p is not None:
            return float(p)
        node = getattr(load, "node_obj", None)
        voltage = float(getattr(node, "voltage", 1.0) or 1.0)
        return float(getattr(load, "pbase", 1.0) * (load.pv0 + load.pv1 * voltage + load.pv2 * voltage * voltage))

    def _add_pseudo_topology_measurements(self, next_idx: int) -> int:
        """Add weak priors for topology devices that have no usable measurement row."""
        measured_keys = self._active_measurement_key_cache

        for device_type, devices in (
            ("DCZeroBranch", self.zero_branches),
            ("DCBreak", self.breakers),
        ):
            for dev in devices:
                voltage = float(getattr(getattr(dev, "i_node_obj", None), "voltage", 1.0) or 1.0)
                values = (
                    ("P_FROM", float(getattr(dev, "p", 0.0) or 0.0)),
                    ("V_FROM", voltage),
                    ("I_FROM", float(getattr(dev, "current", 0.0) or 0.0)),
                )
                for meas_type, value in values:
                    if (device_type, dev.name, meas_type) in measured_keys:
                        continue
                    next_idx = self._append_pseudo_measurement(
                        next_idx,
                        f"pseudo_{meas_type.lower()}_{dev.name}",
                        device_type,
                        dev.name,
                        meas_type,
                        value,
                    )
        return next_idx

    def _add_pseudo_power_measurements(self) -> None:
        """Add weak priors for devices whose file measurements are missing or invalid."""
        if not hasattr(self, "_active_device_key_cache") or not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        measured_devices = self._active_device_key_cache
        next_idx = self._next_measurement_idx()
        next_idx = self._add_pseudo_topology_measurements(next_idx)

        for gen in sorted(self.generator_by_name.values(), key=lambda item: item.idx):
            if ("DCGenerator", gen.name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{gen.name}",
                "DCGenerator",
                gen.name,
                "P_GEN",
                self._generator_pseudo_power(gen),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{gen.name}",
                "DCGenerator",
                gen.name,
                "V_GEN",
                float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0),
            )

        for load in sorted(self.load_by_name.values(), key=lambda item: item.idx):
            if ("DCLoad", load.name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{load.name}",
                "DCLoad",
                load.name,
                "P_LOAD",
                self._load_pseudo_power(load),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{load.name}",
                "DCLoad",
                load.name,
                "V_LOAD",
                float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0),
            )

        for conv in self.dcdc_converters:
            if ("DCDCConverter", conv.name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_from_{conv.name}",
                "DCDCConverter",
                conv.name,
                "P_FROM",
                float(getattr(conv, "i_p", 0.0) or 0.0),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_to_{conv.name}",
                "DCDCConverter",
                conv.name,
                "P_TO",
                float(getattr(conv, "j_p", 0.0) or 0.0),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_from_{conv.name}",
                "DCDCConverter",
                conv.name,
                "V_FROM",
                float(getattr(getattr(conv, "i_node_obj", None), "voltage", 1.0) or 1.0),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_to_{conv.name}",
                "DCDCConverter",
                conv.name,
                "V_TO",
                float(getattr(getattr(conv, "j_node_obj", None), "voltage", 1.0) or 1.0),
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
            next_idx = max((meas.idx for meas in self.measurements), default=0) + 1
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
                refreshed = self._incremental_update_active_measurement_indexes(
                    self.measurements[measurement_count_before:]
                )
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

    def _add_redundant_observability_pseudo_measurements(self, max_add: int, refresh: bool = True) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add, refresh)

    def _observability_pseudo_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
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

        for node in sorted(self.node_by_name.values(), key=lambda item: item.idx):
            add("DCNode", node.name, "V", float(getattr(node, "voltage", 1.0) or 1.0))

        for load in sorted(self.load_by_name.values(), key=lambda item: item.idx):
            voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
            add("DCLoad", load.name, "P_LOAD", self._load_pseudo_power(load))
            add("DCLoad", load.name, "V_LOAD", voltage)

        for gen in sorted(self.generator_by_name.values(), key=lambda item: item.idx):
            voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
            add("DCGenerator", gen.name, "P_GEN", self._generator_pseudo_power(gen))
            add("DCGenerator", gen.name, "V_GEN", voltage)

        for branch in sorted(self.branch_by_name.values(), key=lambda item: item.idx):
            add("DCBranch", branch.name, "P_FROM", float(getattr(branch, "i_p", 0.0) or 0.0))
            add("DCBranch", branch.name, "P_TO", float(getattr(branch, "j_p", 0.0) or 0.0))

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
        candidate_h = self.jacobian_sparse(x, candidates)
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != len(candidates) or not np.any(scores > 0.0):
            return list(candidates[:max_add])
        order = np.argsort(-scores, kind="stable")
        selected = [candidates[int(pos)] for pos in order[:max_add] if scores[int(pos)] > 0.0]
        return selected or list(candidates[:max_add])

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_label: str,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        """Translate a weak compact DC state into the smallest useful pseudo measurement."""
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

        if prefix == "V" and name in self.node_by_name:
            node = self.node_by_name[name]
            return add("DCNode", name, "V", float(getattr(node, "voltage", 1.0) or 1.0))
        if prefix == "I_ZERO" and name in self.zero_branch_by_name:
            dev = self.zero_branch_by_name[name]
            return add("DCZeroBranch", name, "I_FROM", float(getattr(dev, "current", 0.0) or 0.0))
        if prefix == "I_BREAK" and name in self.break_by_name:
            dev = self.break_by_name[name]
            return add("DCBreak", name, "I_FROM", float(getattr(dev, "current", 0.0) or 0.0))
        if prefix == "P_DCDC_FROM" and name in self.dcdc_by_name:
            return add("DCDCConverter", name, "P_FROM", float(getattr(self.dcdc_by_name[name], "i_p", 0.0) or 0.0))
        if prefix == "P_DCDC_TO" and name in self.dcdc_by_name:
            return add("DCDCConverter", name, "P_TO", float(getattr(self.dcdc_by_name[name], "j_p", 0.0) or 0.0))
        if prefix == "P_VGEN" and name in self.generator_by_name:
            return add("DCGenerator", name, "P_GEN", self._generator_pseudo_power(self.generator_by_name[name]))
        return next_idx, 0

    def _add_zero_branch_constraint_measurements(self) -> None:
        """Add ideal DC voltage-equality constraints for zero branches and closed switches."""
        if not hasattr(self, "_active_measurement_key_cache"):
            self._refresh_measurement_summary_cache()
        existing = {
            key
            for key in self._active_measurement_key_cache
            if key[0] in ("DCZeroBranchConstraint", "DCBreakConstraint")
        }
        next_idx = self._next_measurement_idx()
        weight = 10.0
        ideal_devices = [
            ("DCZeroBranchConstraint", zbr)
            for zbr in sorted(self.zero_branch_by_name.values(), key=lambda item: item.idx)
        ]
        ideal_devices.extend(
            ("DCBreakConstraint", brk)
            for brk in sorted(self.break_by_name.values(), key=lambda item: item.idx)
        )
        for device_type, dev in ideal_devices:
            key = (device_type, dev.name, "V_DIFF")
            if key in existing:
                continue
            measurement = Measurement.__new__(Measurement)
            measurement.idx = next_idx
            measurement.name = f"constraint_v_diff_{dev.name}"
            measurement.device_type = device_type
            measurement.device_name = dev.name
            measurement.meas_type = "V_DIFF"
            measurement.weight = weight
            measurement.valid = True
            measurement.value = 0.0
            self.measurements.append(measurement)
            self._record_measurement_summary(measurement)
            existing.add(key)
            next_idx += 1

    def initial_state(self) -> np.ndarray:
        x = np.zeros(self.n_state, dtype=np.float64)
        if self.flat_start:
            x[: self.n_voltage] = 1.0
            for gen in self.v_generators:
                pos = self.v_generator_start + self.v_generator_pos[gen.name]
                x[pos] = float(getattr(gen, "p_set", 0.0) or 0.0)
            return x

        for col, node_pos in enumerate(self.voltage_state_pos):
            node = self.nodes[int(node_pos)]
            x[col] = float(getattr(node, "voltage", 1.0) or 1.0)
        for zbr in self.zero_branches:
            if zbr.name not in self.zero_branch_pos:
                continue
            pos = self.switch_start + self.zero_branch_pos[zbr.name]
            x[pos] = float(getattr(zbr, "current", 0.0) or 0.0)
        for conv in self.dcdc_converters:
            pos = self.dcdc_start + 2 * self.dcdc_pos[conv.name]
            x[pos] = float(getattr(conv, "i_p", 0.0) or 0.0)
            x[pos + 1] = float(getattr(conv, "j_p", 0.0) or 0.0)
        for gen in self.v_generators:
            pos = self.v_generator_start + self.v_generator_pos[gen.name]
            x[pos] = float(getattr(gen, "p", 0.0) or 0.0)
        return x

    def _unpack_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split WLS state into node voltages, switch currents, converter powers and V-gen powers."""
        state_voltage = np.asarray(x[: self.n_voltage], dtype=np.float64).copy()
        state_voltage[state_voltage < self.voltage_floor] = self.voltage_floor
        voltage = np.ones(len(self.nodes), dtype=np.float64)
        if self._voltage_expand_pos.size:
            voltage[self._voltage_expand_pos] = state_voltage[self._voltage_expand_col]
        if self._ref_voltage_pos.size:
            voltage[self._ref_voltage_pos] = self._ref_voltage_value
        switch_current = np.asarray(x[self.switch_start : self.dcdc_start], dtype=np.float64)
        dcdc_power = np.asarray(x[self.dcdc_start : self.v_generator_start], dtype=np.float64)
        v_generator_power = np.asarray(x[self.v_generator_start :], dtype=np.float64)
        return voltage, switch_current, dcdc_power, v_generator_power

    @staticmethod
    def _safe_current(power: float, voltage: float, min_voltage: float) -> float:
        return float(power / voltage) if abs(voltage) > min_voltage else 0.0

    def _branch_values(self, br, voltage: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Return P/V/I at both ends of a resistive DC branch."""
        vi = voltage[self.node_pos[br.i_node]]
        vj = voltage[self.node_pos[br.j_node]]
        current = (vi - vj) / br.r
        return vi * current, vi, current, -vj * current, vj, -current

    def _load_values(self, load, voltage: np.ndarray) -> Tuple[float, float, float]:
        v = voltage[self.node_pos[load.node]]
        p = getattr(load, "pbase", 1.0) * (load.pv0 + load.pv1 * v + load.pv2 * v * v)
        return p, v, self._safe_current(p, v, self.min_current_voltage)

    def _generator_values(
        self,
        gen,
        voltage: np.ndarray,
        v_generator_power: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Evaluate generator terminal quantities according to its P/I/V control mode."""
        v = voltage[self.node_pos[gen.node]]
        if gen.control_type == "V":
            p = v_generator_power[self.v_generator_pos[gen.name]]
            current = self._safe_current(p, v, self.min_current_voltage)
        elif gen.control_type == "P":
            p = float(gen.p_set)
            current = self._safe_current(p, v, self.min_current_voltage)
        elif gen.control_type == "I":
            current = float(gen.i_set)
            p = current * v
        else:
            raise RuntimeError(f"Unsupported DCGenerator control type: {gen.control_type}")
        return p, v, current

    def _zero_branch_values(
        self,
        zbr,
        voltage: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        current = switch_current[self.zero_branch_pos[zbr.name]]
        vi = voltage[self.node_pos[zbr.i_node]]
        vj = voltage[self.node_pos[zbr.j_node]]
        return vi * current, vi, current, -vj * current, vj, -current

    def _break_values(
        self,
        brk,
        voltage: np.ndarray,
        switch_current: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        current = switch_current[self.break_pos[brk.name]]
        vi = voltage[self.node_pos[brk.i_node]]
        vj = voltage[self.node_pos[brk.j_node]]
        return vi * current, vi, current, -vj * current, vj, -current

    def _dcdc_values(
        self,
        conv,
        voltage: np.ndarray,
        dcdc_power: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        """Return measured port quantities for a DCDC converter whose port powers are states."""
        pos = 2 * self.dcdc_pos[conv.name]
        p_from = dcdc_power[pos]
        p_to = dcdc_power[pos + 1]
        v_from = voltage[self.node_pos[conv.i_node]]
        v_to = voltage[self.node_pos[conv.j_node]]
        return (
            p_from,
            v_from,
            self._safe_current(p_from, v_from, self.min_current_voltage),
            p_to,
            v_to,
            self._safe_current(p_to, v_to, self.min_current_voltage),
        )

    @staticmethod
    def _terminal_value(meas_type: str, values: Tuple[float, float, float, float, float, float]) -> float:
        p_from, v_from, i_from, p_to, v_to, i_to = values
        if meas_type == "P_FROM":
            return float(p_from)
        if meas_type == "V_FROM":
            return float(v_from)
        if meas_type == "I_FROM":
            return float(i_from)
        if meas_type == "P_TO":
            return float(p_to)
        if meas_type == "V_TO":
            return float(v_to)
        if meas_type == "I_TO":
            return float(i_to)
        raise RuntimeError(f"Unsupported terminal measurement type: {meas_type}")

    def _measurement_plan(self, measurements: Sequence[Measurement]) -> Dict[str, np.ndarray]:
        key = id(measurements)
        cached = self._measurement_plan_cache.get(key)
        if cached is not None and cached[0] is measurements:
            return cached[1]

        self._ensure_measurement_plan_lookup_arrays()
        plan_table = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=self._measurement_plan_device_pos_by_type_code,
            meas_kind_by_type_code=self._measurement_plan_meas_kind_by_type_code,
            table_builder=_measurement_table_from_measurements,
        )
        handled = np.asarray(plan_table.handled, dtype=bool).copy()
        row = plan_table.row
        device_type_code = plan_table.device_type_code
        meas_kind = plan_table.meas_kind
        device_pos = plan_table.device_pos

        node_code = _DEVICE_TYPE_CODES["DCNode"]
        branch_code = _DEVICE_TYPE_CODES["DCBranch"]
        load_code = _DEVICE_TYPE_CODES["DCLoad"]
        gen_code = _DEVICE_TYPE_CODES["DCGenerator"]
        zero_branch_code = _DEVICE_TYPE_CODES["DCZeroBranch"]
        break_code = _DEVICE_TYPE_CODES["DCBreak"]
        zero_constraint_code = _DEVICE_TYPE_CODES["DCZeroBranchConstraint"]
        break_constraint_code = _DEVICE_TYPE_CODES["DCBreakConstraint"]
        dcdc_code = _DEVICE_TYPE_CODES["DCDCConverter"]

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
            keep = (current_pos >= 0) | (kind == _TERMINAL_KIND["V_FROM"]) | (kind == _TERMINAL_KIND["V_TO"])
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
            i = plan["branch_i"]
            j = plan["branch_j"]
            kind = plan["branch_kind"]
            vi = voltage[i]
            vj = voltage[j]
            current = (vi - vj) * plan["branch_inv_r"]
            out = np.empty(rows.size, dtype=np.float64)
            out[kind == 0] = (vi * current)[kind == 0]
            out[kind == 1] = vi[kind == 1]
            out[kind == 2] = current[kind == 2]
            out[kind == 3] = (-vj * current)[kind == 3]
            out[kind == 4] = vj[kind == 4]
            out[kind == 5] = (-current)[kind == 5]
            values[rows] = out

        rows = plan["load_rows"]
        if rows.size:
            kind = plan["load_kind"]
            v = voltage[plan["load_pos"]]
            p = plan["load_pv0"] + plan["load_pv1"] * v + plan["load_pv2"] * v * v
            out = np.empty(rows.size, dtype=np.float64)
            out[kind == 0] = p[kind == 0]
            out[kind == 1] = v[kind == 1]
            i_mask = kind == 2
            out[i_mask] = 0.0
            if np.any(i_mask):
                valid = np.abs(v[i_mask]) > self.min_current_voltage
                idx = np.flatnonzero(i_mask)
                out[idx[valid]] = p[i_mask][valid] / v[i_mask][valid]
            values[rows] = out

        rows = plan["gen_rows"]
        if rows.size:
            kind = plan["gen_kind"]
            ctrl = plan["gen_ctrl"]
            v = voltage[plan["gen_pos"]]
            p = np.zeros(rows.size, dtype=np.float64)
            v_mask = ctrl == 0
            p_mask = ctrl == 1
            i_mask = ctrl == 2
            if np.any(v_mask):
                p[v_mask] = v_generator_power[plan["gen_vgen_pos"][v_mask]]
            p[p_mask] = plan["gen_p_set"][p_mask]
            p[i_mask] = plan["gen_i_set"][i_mask] * v[i_mask]
            current = np.zeros(rows.size, dtype=np.float64)
            fixed_current = i_mask
            current[fixed_current] = plan["gen_i_set"][fixed_current]
            derived_current = ~fixed_current
            valid = np.abs(v[derived_current]) > self.min_current_voltage
            idx = np.flatnonzero(derived_current)
            current[idx[valid]] = p[derived_current][valid] / v[derived_current][valid]
            out = np.empty(rows.size, dtype=np.float64)
            out[kind == 0] = p[kind == 0]
            out[kind == 1] = v[kind == 1]
            out[kind == 2] = current[kind == 2]
            values[rows] = out

        rows = plan["switch_rows"]
        if rows.size:
            kind = plan["switch_kind"]
            vi = voltage[plan["switch_i"]]
            vj = voltage[plan["switch_j"]]
            current = np.zeros(rows.size, dtype=np.float64)
            current_mask = plan["switch_pos"] >= 0
            if np.any(current_mask):
                current[current_mask] = switch_current[plan["switch_pos"][current_mask]]
            out = np.empty(rows.size, dtype=np.float64)
            out[kind == 0] = (vi * current)[kind == 0]
            out[kind == 1] = vi[kind == 1]
            out[kind == 2] = current[kind == 2]
            out[kind == 3] = (-vj * current)[kind == 3]
            out[kind == 4] = vj[kind == 4]
            out[kind == 5] = (-current)[kind == 5]
            values[rows] = out

        rows = plan["constraint_rows"]
        if rows.size:
            values[rows] = voltage[plan["constraint_i"]] - voltage[plan["constraint_j"]]

        rows = plan["dcdc_rows"]
        if rows.size:
            kind = plan["dcdc_kind"]
            conv_pos = plan["dcdc_pos"]
            p_from = dcdc_power[2 * conv_pos]
            p_to = dcdc_power[2 * conv_pos + 1]
            v_from = voltage[plan["dcdc_i"]]
            v_to = voltage[plan["dcdc_j"]]
            i_from = np.zeros(rows.size, dtype=np.float64)
            i_to = np.zeros(rows.size, dtype=np.float64)
            valid_from = np.abs(v_from) > self.min_current_voltage
            valid_to = np.abs(v_to) > self.min_current_voltage
            i_from[valid_from] = p_from[valid_from] / v_from[valid_from]
            i_to[valid_to] = p_to[valid_to] / v_to[valid_to]
            out = np.empty(rows.size, dtype=np.float64)
            out[kind == 0] = p_from[kind == 0]
            out[kind == 1] = v_from[kind == 1]
            out[kind == 2] = i_from[kind == 2]
            out[kind == 3] = p_to[kind == 3]
            out[kind == 4] = v_to[kind == 4]
            out[kind == 5] = i_to[kind == 5]
            values[rows] = out

        return plan["handled_mask"]

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Evaluate h(x): estimated values for each active DC measurement."""
        measurements = self._normalize_measurements(measurements)
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        values = np.zeros(len(measurements), dtype=np.float64)
        vectorized_rows = self._fill_measurement_values_vectorized(
            values,
            measurements,
            voltage,
            switch_current,
            dcdc_power,
            v_generator_power,
        )

        for row, meas in enumerate(measurements):
            if vectorized_rows[row]:
                continue
            mtype = meas.meas_type
            if meas.device_type == "DCNode":
                node = self.node_by_name[meas.device_name]
                if mtype != "V":
                    raise RuntimeError(f"Unsupported DCNode measurement type: {mtype}")
                values[row] = voltage[self.node_pos[node.idx]]
            elif meas.device_type == "DCBranch":
                br = self.branch_by_name[meas.device_name]
                values[row] = self._terminal_value(mtype, self._branch_values(br, voltage))
            elif meas.device_type == "DCLoad":
                load = self.load_by_name[meas.device_name]
                p, v, current = self._load_values(load, voltage)
                if mtype == "P_LOAD":
                    values[row] = p
                elif mtype == "V_LOAD":
                    values[row] = v
                elif mtype == "I_LOAD":
                    values[row] = current
                else:
                    raise RuntimeError(f"Unsupported DCLoad measurement type: {mtype}")
            elif meas.device_type == "DCGenerator":
                gen = self.generator_by_name[meas.device_name]
                p, v, current = self._generator_values(gen, voltage, v_generator_power)
                if mtype == "P_GEN":
                    values[row] = p
                elif mtype == "V_GEN":
                    values[row] = v
                elif mtype == "I_GEN":
                    values[row] = current
                else:
                    raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
            elif meas.device_type == "DCBreak":
                brk = self.break_by_name[meas.device_name]
                values[row] = self._terminal_value(mtype, self._break_values(brk, voltage, switch_current))
            elif meas.device_type == "DCZeroBranch":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype in ("V_FROM", "V_TO"):
                    node_idx = zbr.i_node if mtype == "V_FROM" else zbr.j_node
                    values[row] = voltage[self.node_pos[node_idx]]
                else:
                    values[row] = self._terminal_value(mtype, self._zero_branch_values(zbr, voltage, switch_current))
            elif meas.device_type == "DCZeroBranchConstraint":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCZeroBranchConstraint measurement type: {mtype}")
                values[row] = voltage[self.node_pos[zbr.i_node]] - voltage[self.node_pos[zbr.j_node]]
            elif meas.device_type == "DCBreakConstraint":
                brk = self.break_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCBreakConstraint measurement type: {mtype}")
                values[row] = voltage[self.node_pos[brk.i_node]] - voltage[self.node_pos[brk.j_node]]
            elif meas.device_type == "DCDCConverter":
                conv = self.dcdc_by_name[meas.device_name]
                values[row] = self._terminal_value(mtype, self._dcdc_values(conv, voltage, dcdc_power))
            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")
        return values

    @staticmethod
    def _add_indexed_values(
        H,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = cols >= 0
        else:
            mask = np.asarray(mask, dtype=bool) & (cols >= 0)
        if hasattr(H, "add_many"):
            H.add_many(rows, cols, values, mask)
            return
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
            kind = plan["branch_kind"]
            i = plan["branch_i"]
            j = plan["branch_j"]
            i_col = plan["branch_i_col"]
            j_col = plan["branch_j_col"]
            vi = voltage[i]
            vj = voltage[j]
            inv_r = plan["branch_inv_r"]

            mask = kind == 0
            self._add_indexed_values(H, rows, i_col, (2.0 * vi - vj) * inv_r, mask)
            self._add_indexed_values(H, rows, j_col, -vi * inv_r, mask)
            mask = kind == 1
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 2
            self._add_indexed_values(H, rows, i_col, inv_r, mask)
            self._add_indexed_values(H, rows, j_col, -inv_r, mask)
            mask = kind == 3
            self._add_indexed_values(H, rows, i_col, -vj * inv_r, mask)
            self._add_indexed_values(H, rows, j_col, (-vi + 2.0 * vj) * inv_r, mask)
            mask = kind == 4
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 5
            self._add_indexed_values(H, rows, i_col, -inv_r, mask)
            self._add_indexed_values(H, rows, j_col, inv_r, mask)

        rows = plan["load_rows"]
        if rows.size:
            kind = plan["load_kind"]
            pos = plan["load_pos"]
            col = plan["load_col"]
            v = voltage[pos]
            mask = kind == 0
            self._add_indexed_values(H, rows, col, plan["load_pv1"] + 2.0 * plan["load_pv2"] * v, mask)
            mask = kind == 1
            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 2
            self._add_indexed_values(H, rows, col, plan["load_pv2"] - plan["load_pv0"] / (v * v), mask)

        rows = plan["gen_rows"]
        if rows.size:
            kind = plan["gen_kind"]
            ctrl = plan["gen_ctrl"]
            pos = plan["gen_pos"]
            col = plan["gen_col"]
            v = voltage[pos]
            p_col = plan["gen_p_col"]
            p = np.zeros(rows.size, dtype=np.float64)
            v_ctrl = ctrl == 0
            p_ctrl = ctrl == 1
            i_ctrl = ctrl == 2
            if np.any(v_ctrl):
                p[v_ctrl] = v_generator_power[plan["gen_vgen_pos"][v_ctrl]]
            p[p_ctrl] = plan["gen_p_set"][p_ctrl]
            p[i_ctrl] = plan["gen_i_set"][i_ctrl] * v[i_ctrl]

            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), kind == 1)
            self._add_indexed_values(H, rows, p_col, np.ones(rows.size, dtype=np.float64), (kind == 0) & v_ctrl)
            self._add_indexed_values(H, rows, col, plan["gen_i_set"], (kind == 0) & i_ctrl)
            self._add_indexed_values(H, rows, p_col, 1.0 / v, (kind == 2) & v_ctrl)
            self._add_indexed_values(H, rows, col, -p / (v * v), (kind == 2) & v_ctrl)
            self._add_indexed_values(H, rows, col, -plan["gen_p_set"] / (v * v), (kind == 2) & p_ctrl)

        rows = plan["switch_rows"]
        if rows.size:
            kind = plan["switch_kind"]
            i = plan["switch_i"]
            j = plan["switch_j"]
            i_col = plan["switch_i_col"]
            j_col = plan["switch_j_col"]
            col = plan["switch_col"]
            vi = voltage[i]
            vj = voltage[j]
            current = np.zeros(rows.size, dtype=np.float64)
            current_mask = plan["switch_pos"] >= 0
            if np.any(current_mask):
                current[current_mask] = switch_current[plan["switch_pos"][current_mask]]

            mask = kind == 0
            self._add_indexed_values(H, rows, i_col, current, mask)
            self._add_indexed_values(H, rows, col, vi, mask)
            mask = kind == 1
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 2
            self._add_indexed_values(H, rows, col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 3
            self._add_indexed_values(H, rows, j_col, -current, mask)
            self._add_indexed_values(H, rows, col, -vj, mask)
            mask = kind == 4
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 5
            self._add_indexed_values(H, rows, col, -np.ones(rows.size, dtype=np.float64), mask)

        rows = plan["constraint_rows"]
        if rows.size:
            self._add_indexed_values(H, rows, plan["constraint_i_col"], np.ones(rows.size, dtype=np.float64))
            self._add_indexed_values(H, rows, plan["constraint_j_col"], -np.ones(rows.size, dtype=np.float64))

        rows = plan["dcdc_rows"]
        if rows.size:
            kind = plan["dcdc_kind"]
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

            mask = kind == 0
            self._add_indexed_values(H, rows, p_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 1
            self._add_indexed_values(H, rows, i_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 2
            self._add_indexed_values(H, rows, p_col, 1.0 / v_from, mask)
            self._add_indexed_values(H, rows, i_col, -p_from / (v_from * v_from), mask)
            mask = kind == 3
            self._add_indexed_values(H, rows, q_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 4
            self._add_indexed_values(H, rows, j_col, np.ones(rows.size, dtype=np.float64), mask)
            mask = kind == 5
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
            H = self._jacobian_builder if measurements is self.active_measurements else SparseJacobianBuilder((len(measurements), self.n_state))
            H.reset()
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
        if np.all(vectorized_rows):
            return H.to_csr() if sparse else H

        for row, meas in enumerate(measurements):
            if vectorized_rows[row]:
                continue
            mtype = meas.meas_type
            if meas.device_type == "DCNode":
                if mtype != "V":
                    raise RuntimeError(f"Unsupported DCNode measurement type: {mtype}")
                node = self.node_by_name[meas.device_name]
                self._add_derivative(H, row, int(self.voltage_col[self.node_pos[node.idx]]), 1.0)

            elif meas.device_type == "DCBranch":
                br = self.branch_by_name[meas.device_name]
                i = self.node_pos[br.i_node]
                j = self.node_pos[br.j_node]
                i_state = int(self.voltage_col[i])
                j_state = int(self.voltage_col[j])
                vi = voltage[i]
                vj = voltage[j]
                inv_r = 1.0 / br.r
                if mtype == "P_FROM":
                    self._add_derivative(H, row, i_state, (2.0 * vi - vj) * inv_r)
                    self._add_derivative(H, row, j_state, -vi * inv_r)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_state, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, i_state, inv_r)
                    self._add_derivative(H, row, j_state, -inv_r)
                elif mtype == "P_TO":
                    self._add_derivative(H, row, i_state, -vj * inv_r)
                    self._add_derivative(H, row, j_state, (-vi + 2.0 * vj) * inv_r)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_state, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, i_state, -inv_r)
                    self._add_derivative(H, row, j_state, inv_r)
                else:
                    raise RuntimeError(f"Unsupported DCBranch measurement type: {mtype}")

            elif meas.device_type == "DCLoad":
                load = self.load_by_name[meas.device_name]
                i = self.node_pos[load.node]
                i_state = int(self.voltage_col[i])
                v = voltage[i]
                if mtype == "P_LOAD":
                    self._add_derivative(H, row, i_state, getattr(load, "pbase", 1.0) * (load.pv1 + 2.0 * load.pv2 * v))
                elif mtype == "V_LOAD":
                    self._add_derivative(H, row, i_state, 1.0)
                elif mtype == "I_LOAD":
                    self._add_derivative(H, row, i_state, getattr(load, "pbase", 1.0) * (load.pv2 - load.pv0 / (v * v)))
                else:
                    raise RuntimeError(f"Unsupported DCLoad measurement type: {mtype}")

            elif meas.device_type == "DCGenerator":
                gen = self.generator_by_name[meas.device_name]
                i = self.node_pos[gen.node]
                i_state = int(self.voltage_col[i])
                v = voltage[i]
                # V-control generators expose active power as an estimator state;
                # P/I-control generators have fixed power or current setpoints.
                if gen.control_type == "V":
                    p_col = self.v_generator_start + self.v_generator_pos[gen.name]
                    p = v_generator_power[self.v_generator_pos[gen.name]]
                    if mtype == "P_GEN":
                        self._add_derivative(H, row, p_col, 1.0)
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, i_state, 1.0)
                    elif mtype == "I_GEN":
                        self._add_derivative(H, row, p_col, 1.0 / v)
                        self._add_derivative(H, row, i_state, -p / (v * v))
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                elif gen.control_type == "P":
                    if mtype == "P_GEN":
                        pass
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, i_state, 1.0)
                    elif mtype == "I_GEN":
                        self._add_derivative(H, row, i_state, -gen.p_set / (v * v))
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                elif gen.control_type == "I":
                    if mtype == "P_GEN":
                        self._add_derivative(H, row, i_state, gen.i_set)
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, i_state, 1.0)
                    elif mtype == "I_GEN":
                        pass
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                else:
                    raise RuntimeError(f"Unsupported DCGenerator control type: {gen.control_type}")

            elif meas.device_type == "DCBreak":
                brk = self.break_by_name[meas.device_name]
                i = self.node_pos[brk.i_node]
                j = self.node_pos[brk.j_node]
                i_state = int(self.voltage_col[i])
                j_state = int(self.voltage_col[j])
                i_col = self.switch_start + self.break_pos[brk.name]
                current = switch_current[self.break_pos[brk.name]]
                vi = voltage[i]
                vj = voltage[j]
                if mtype == "P_FROM":
                    self._add_derivative(H, row, i_state, current)
                    self._add_derivative(H, row, i_col, vi)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_state, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, i_col, 1.0)
                elif mtype == "P_TO":
                    self._add_derivative(H, row, j_state, -current)
                    self._add_derivative(H, row, i_col, -vj)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_state, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, i_col, -1.0)
                else:
                    raise RuntimeError(f"Unsupported DCBreak measurement type: {mtype}")

            elif meas.device_type == "DCZeroBranch":
                zbr = self.zero_branch_by_name[meas.device_name]
                i = self.node_pos[zbr.i_node]
                j = self.node_pos[zbr.j_node]
                i_state = int(self.voltage_col[i])
                j_state = int(self.voltage_col[j])
                i_col = self.switch_start + self.zero_branch_pos.get(zbr.name, -1)
                current = switch_current[self.zero_branch_pos[zbr.name]] if zbr.name in self.zero_branch_pos else 0.0
                vi = voltage[i]
                vj = voltage[j]
                if mtype == "P_FROM":
                    self._add_derivative(H, row, i_state, current)
                    self._add_derivative(H, row, i_col, vi)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_state, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, i_col, 1.0)
                elif mtype == "P_TO":
                    self._add_derivative(H, row, j_state, -current)
                    self._add_derivative(H, row, i_col, -vj)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_state, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, i_col, -1.0)
                else:
                    raise RuntimeError(f"Unsupported DCZeroBranch measurement type: {mtype}")

            elif meas.device_type == "DCZeroBranchConstraint":
                zbr = self.zero_branch_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCZeroBranchConstraint measurement type: {mtype}")
                self._add_derivative(H, row, int(self.voltage_col[self.node_pos[zbr.i_node]]), 1.0)
                self._add_derivative(H, row, int(self.voltage_col[self.node_pos[zbr.j_node]]), -1.0)

            elif meas.device_type == "DCBreakConstraint":
                brk = self.break_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCBreakConstraint measurement type: {mtype}")
                self._add_derivative(H, row, int(self.voltage_col[self.node_pos[brk.i_node]]), 1.0)
                self._add_derivative(H, row, int(self.voltage_col[self.node_pos[brk.j_node]]), -1.0)

            elif meas.device_type == "DCDCConverter":
                conv = self.dcdc_by_name[meas.device_name]
                i = self.node_pos[conv.i_node]
                j = self.node_pos[conv.j_node]
                i_state = int(self.voltage_col[i])
                j_state = int(self.voltage_col[j])
                p_col = self.dcdc_start + 2 * self.dcdc_pos[conv.name]
                q_col = p_col + 1
                p_from = dcdc_power[2 * self.dcdc_pos[conv.name]]
                p_to = dcdc_power[2 * self.dcdc_pos[conv.name] + 1]
                v_from = voltage[i]
                v_to = voltage[j]
                if mtype == "P_FROM":
                    self._add_derivative(H, row, p_col, 1.0)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_state, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, p_col, 1.0 / v_from)
                    self._add_derivative(H, row, i_state, -p_from / (v_from * v_from))
                elif mtype == "P_TO":
                    self._add_derivative(H, row, q_col, 1.0)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_state, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, q_col, 1.0 / v_to)
                    self._add_derivative(H, row, j_state, -p_to / (v_to * v_to))
                else:
                    raise RuntimeError(f"Unsupported DCDCConverter measurement type: {mtype}")
            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")
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
        normal_factor_diag = None
        normal_solver = NormalEquationSolver(assume_fixed_pattern=measurements is self.active_measurements)
        normal_pattern = self._active_normal_pattern if measurements is self.active_measurements else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]

        if verbose:
            _print_iteration_header()

        for iteration in range(1, self.max_iter + 1):
            z_est = self.evaluate(x, measurements)
            residual = z - z_est
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
            dx, normal_factor_diag = normal_solver.solve(gain, rhs, return_factor_diag=False)

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
                candidate_residual = z - self.evaluate(candidate, measurements)
                candidate_objective = 0.5 * float(np.dot(weight * candidate_residual, candidate_residual))
                if candidate_objective <= objective or step_scale < 1e-3:
                    x = candidate
                    objective = candidate_objective
                    accepted_step = step_scale
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                x += dx
                x[: self.n_voltage] = np.maximum(x[: self.n_voltage], self.voltage_floor)
                accepted_step = 1.0

            if verbose:
                updated_residual = z - self.evaluate(x, measurements)
                updated_residual_inf = float(np.linalg.norm(updated_residual, np.inf)) if updated_residual.size else 0.0
                updated_objective = 0.5 * float(np.dot(weight * updated_residual, updated_residual))
                _print_iteration(
                    iteration,
                    updated_objective,
                    updated_residual_inf,
                    max_correction,
                    accepted_step,
                    False,
                )

        if not final_quantities_current:
            z_est = self.evaluate(x, measurements)
            residual = z - z_est
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
            objective = 0.5 * float(np.dot(weight * residual, residual))
            normal_factor_diag = None
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
        """Compute largest normalized residuals after accounting for measurement leverage."""
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
        """Write estimated voltages, currents and powers back to the model objects."""
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)

        for pos, node in enumerate(self.nodes):
            node.voltage = float(voltage[pos])
            for member in getattr(node, "nodes", ()):
                member.voltage = node.voltage

        if self._apply_branch_devices:
            vi = voltage[self._apply_branch_i]
            vj = voltage[self._apply_branch_j]
            current = (vi - vj) * self._apply_branch_inv_r
            p_from = vi * current
            p_to = -vj * current
            for br, pf, pt, cur in zip(self._apply_branch_devices, p_from, p_to, current):
                br.i_p = float(pf)
                br.j_p = float(pt)
                br.current = float(cur)

        if self._apply_load_devices:
            v = voltage[self._apply_load_pos]
            p = self._apply_load_pv0 + self._apply_load_pv1 * v + self._apply_load_pv2 * v * v
            current = np.zeros_like(p)
            np.divide(p, v, out=current, where=np.abs(v) > self.min_current_voltage)
            for load, power, cur in zip(self._apply_load_devices, p, current):
                load.p = float(power)
                load.current = float(cur)

        if self._apply_generator_devices:
            v = voltage[self._apply_generator_pos]
            p = np.zeros(len(self._apply_generator_devices), dtype=np.float64)
            ctrl = self._apply_generator_ctrl
            v_mask = ctrl == 0
            p_mask = ctrl == 1
            i_mask = ctrl == 2
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
            for gen, power, cur in zip(self._apply_generator_devices, p, current):
                gen.p = float(power)
                gen.current = float(cur)

        if self._apply_switch_devices:
            vi = voltage[self._apply_switch_i]
            current = switch_current[self._apply_switch_pos]
            p_from = vi * current
            for sw, pf, cur in zip(self._apply_switch_devices, p_from, current):
                sw.p = float(pf)
                sw.current = float(cur)

        if self._apply_break_devices:
            vi = voltage[self._apply_break_i]
            current = switch_current[self._apply_break_pos]
            p_from = vi * current
            for brk, pf, cur in zip(self._apply_break_devices, p_from, current):
                brk.p = float(pf)
                brk.current = float(cur)

        if self._apply_zero_branch_devices:
            vi = voltage[self._apply_zero_branch_i]
            current = switch_current[self._apply_zero_branch_pos]
            p_from = vi * current
            for zbr, pf, cur in zip(self._apply_zero_branch_devices, p_from, current):
                zbr.p = float(pf)
                zbr.current = float(cur)

        if self._apply_dcdc_devices:
            conv_pos = self._apply_dcdc_pos
            p_from = dcdc_power[2 * conv_pos]
            p_to = dcdc_power[2 * conv_pos + 1]
            v_from = voltage[self._apply_dcdc_i]
            v_to = voltage[self._apply_dcdc_j]
            i_from = np.zeros_like(p_from)
            i_to = np.zeros_like(p_to)
            np.divide(p_from, v_from, out=i_from, where=np.abs(v_from) > self.min_current_voltage)
            np.divide(p_to, v_to, out=i_to, where=np.abs(v_to) > self.min_current_voltage)
            for conv, pf, pt, ifrom, ito in zip(self._apply_dcdc_devices, p_from, p_to, i_from, i_to):
                conv.i_p = float(pf)
                conv.j_p = float(pt)
                conv.i_c = float(ifrom)
                conv.j_c = float(ito)

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        voltage, switch_current, dcdc_power, v_generator_power = self._unpack_state(x)
        print("Estimated DC node voltages:")
        for pos, node in enumerate(self.nodes[:limit]):
            print(f"  {node.name:10s} V={voltage[pos]:.9f}")
        if len(self.nodes) > limit:
            print(f"  ... {len(self.nodes) - limit} more nodes")

        estimated_zero_branches = [zbr for zbr in self.zero_branches if zbr.name in self.zero_branch_pos]
        if estimated_zero_branches:
            print("Estimated zero-branch currents:")
            for zbr in estimated_zero_branches[:limit]:
                print(f"  {zbr.name:10s} I={switch_current[self.zero_branch_pos[zbr.name]]:.9f}")
            if len(estimated_zero_branches) > limit:
                print(f"  ... {len(estimated_zero_branches) - limit} more zero branches")
        if self.breakers:
            print("Estimated break currents:")
            for brk in self.breakers[:limit]:
                print(f"  {brk.name:10s} I={switch_current[self.break_pos[brk.name]]:.9f}")
            if len(self.breakers) > limit:
                print(f"  ... {len(self.breakers) - limit} more breaks")

        if self.dcdc_converters:
            print("Estimated DCDC port powers:")
            for conv in self.dcdc_converters[:limit]:
                pos = 2 * self.dcdc_pos[conv.name]
                print(f"  {conv.name:10s} P_FROM={dcdc_power[pos]:.9f} P_TO={dcdc_power[pos + 1]:.9f}")
            if len(self.dcdc_converters) > limit:
                print(f"  ... {len(self.dcdc_converters) - limit} more DCDC converters")

        if self.v_generators:
            print("Estimated voltage-source generator powers:")
            for gen in self.v_generators[:limit]:
                print(f"  {gen.name:10s} P={v_generator_power[self.v_generator_pos[gen.name]]:.9f}")
            if len(self.v_generators) > limit:
                print(f"  ... {len(self.v_generators) - limit} more voltage-source generators")


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
        result = estimator.estimate(verbose=not args.quiet, observability=initial_observability)

    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    profile_stage_start = time.perf_counter()
    bad_items, normalized = estimator.identify_bad_data(result, bad_threshold)
    estimator._record_profile_time("bad_data.identify", time.perf_counter() - profile_stage_start)
    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
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
        for name, value in sorted(estimator.profile_times.items()):
            print(f"  {name}={value:.6f}s")

    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
