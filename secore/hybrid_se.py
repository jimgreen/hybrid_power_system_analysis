import argparse
import contextlib
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import (
    SWITCH_COLS,
    ZERO_BRANCH_COLS,
)
from ac_lf import ACPowerFlowCalc, matpower_branch_stamp, matpower_branch_stamp_vectorized
from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork
from model.meas_model import (
    BadDataItem,
    EstimateResult,
    GEN_CONTROL_KIND,
    Measurement,
    ObservabilityResult,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.ac_se import (
    ACStateEstimator,
    _read_measurements_direct as _read_measurements_direct_shared,
)
from secore.dc_se import DCStateEstimator
from secore.se_math import (
    ANGLE_MEASUREMENT_TYPES,
    SparseJacobianBuilder,
    angle_residual_mask,
    build_normal_equations,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    measurement_residual as build_measurement_residual,
    observability_rank_details,
    solve_normal_equations_with_factor,
    sparse_structural_rank,
    unanchored_angle_state_labels,
)
from unit_system import ac_current_base_ka, dc_current_base_ka


DEFAULT_CASE = ROOT_DIR / "data" / "hybrid" / "qinling.e"
DEFAULT_MEAS = ROOT_DIR / "data" / "hybrid" / "qinling.meas"


def _read_measurements_direct(meas_file: Path) -> List[Measurement]:
    """Read Measurement rows through the shared AC/DC SE parser."""
    return _read_measurements_direct_shared(meas_file, Measurement)


class HybridStateEstimator:
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
    ):
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

        self.network, self.calc, self.flat_x, self.power_flow_x = self._load_case(self.e_file, self.params)
        self.ac_original_theta_spec = (
            self.calc.ac_calc.theta_spec.copy()
            if self.calc.ac_calc is not None and hasattr(self.calc.ac_calc, "theta_spec")
            else np.array([], dtype=np.float64)
        )
        self.p_base = float(self.network.p_base)
        self.p_base_kW = float(self.network.p_base_kW)
        self.u_scale = float(self.network.u_scale)
        self.p_scale = float(self.network.p_scale)
        self.i_scale = float(self.network.i_scale)
        self.measurements = self._load_measurements(self.meas_file)

        self.full_n_state = int(self.calc.total_vars)
        self.full_state_labels = self._build_state_labels()

        self.ac_nodes = sorted(
            [node for node in self.network.ac.nodes if getattr(node, "is_alive", False)],
            key=lambda item: item.idx,
        )
        self.dc_nodes = sorted(
            [node for node in self.network.dc.nodes if getattr(node, "is_alive", False)],
            key=lambda item: item.idx,
        )
        self.ac_node_by_name = {node.name: node for node in self.ac_nodes}
        self.dc_node_by_name = {node.name: node for node in self.dc_nodes}
        self.ac_branch_by_name = {
            br.name: br for br in self.network.ac.branches if getattr(br, "is_alive", False)
        }
        self.ac_transformer_by_name = {
            tr.name: tr for tr in self.network.ac.transformers if getattr(tr, "is_alive", False)
        }
        self.ac_switch_by_name = {
            sw.name: sw for sw in self.network.ac.switches if getattr(sw, "is_alive", False)
        }
        self.ac_break_by_name = {
            brk.name: brk for brk in getattr(self.network.ac, "breakers", []) if getattr(brk, "is_alive", False)
        }
        self.ac_zero_branch_by_name = {
            zbr.name: zbr for zbr in self.network.ac.zero_branches if getattr(zbr, "is_alive", False)
        }
        self.ac_generator_by_name = {
            gen.name: gen for gen in self.network.ac.generators if getattr(gen, "is_alive", False)
        }
        self.ac_load_by_name = {
            load.name: load for load in self.network.ac.loads if getattr(load, "is_alive", False)
        }
        self.dc_branch_by_name = {
            br.name: br for br in self.network.dc.branches if getattr(br, "is_alive", False)
        }
        self.dc_switch_by_name = {
            sw.name: sw for sw in self.network.dc.switches if getattr(sw, "is_alive", False)
        }
        self.dc_break_by_name = {
            brk.name: brk for brk in getattr(self.network.dc, "breakers", []) if getattr(brk, "is_alive", False)
        }
        self.dc_zero_branch_by_name = {
            zbr.name: zbr for zbr in self.network.dc.zero_branches if getattr(zbr, "is_alive", False)
        }
        self.dc_generator_by_name = {
            gen.name: gen for gen in self.network.dc.generators if getattr(gen, "is_alive", False)
        }
        self.dc_load_by_name = {
            load.name: load for load in self.network.dc.loads if getattr(load, "is_alive", False)
        }
        self.dcdc_by_name = {
            conv.name: conv for conv in self.network.dc.dcdc_converters if getattr(conv, "is_alive", False)
        }
        self.dcac_by_name = {
            conv.name: conv for conv in self.network.dcac_converters if getattr(conv, "is_alive", False)
        }
        self.acac_by_name = {
            conv.name: conv for conv in self.network.acac_converters if getattr(conv, "is_alive", False)
        }
        self._sub_estimators_enabled = False
        self._ac_sub_estimator = None
        self._dc_sub_estimator = None
        if self._init_uncoupled_sub_estimators():
            return

        # Files store named values; all estimator math below uses pu and radians.
        self._disable_angle_measurements()
        self._disable_unavailable_measurements()
        self._convert_measurements_to_pu()
        # Add priors after unit conversion because solved model objects are already normalized.
        self._add_pseudo_power_measurements()
        self._add_dc_zero_branch_constraint_measurements()
        self.targeted_observability_pseudo_count = 0
        self._last_written_state: Optional[np.ndarray] = None
        self._rebased_ac_angle_measurement_names = set()
        self._refresh_active_measurement_state_layout()
        self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()

    def _init_uncoupled_sub_estimators(self) -> bool:
        """Use AC/DC estimators as subsystem engines when no cross-domain devices exist.

        HybridStateEstimator remains the public orchestrator.  In this first
        composition path there are no converter coupling rows to assemble, so
        the subsystem estimator owns the numerical WLS work directly.
        """
        has_cross_domain_coupling = bool(self.dcac_by_name or self.acac_by_name)
        has_ac = bool(self.ac_nodes)
        has_dc = bool(self.dc_nodes)
        if has_ac and self._ac_sub_estimator is None:
            self._ac_sub_estimator = ACStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
            )
        if has_dc and self._dc_sub_estimator is None:
            self._dc_sub_estimator = DCStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
            )

        if has_cross_domain_coupling:
            return False

        if has_ac and has_dc:
            # Block-combining independent AC and DC estimators is kept out of
            # this slice; mixed networks with any DC side still use the legacy
            # hybrid path until converter coupling is composed explicitly.
            return False

        if has_ac:
            sub = self._ac_sub_estimator
            self._ac_sub_estimator = sub
            self._delegate_estimator = sub
            self._sub_estimators_enabled = True
            self.nodes = sub.nodes
            self.ac_nodes = sub.nodes
            self.dc_nodes = []
            self.ac_node_by_name = sub.node_by_name
            self.ac_branch_by_name = sub.branch_by_name
            self.ac_transformer_by_name = sub.transformer_by_name
            self.ac_generator_by_name = sub.generator_by_name
            self.ac_load_by_name = sub.load_by_name
            self.ac_switch_by_name = sub.switch_by_name
            self.ac_break_by_name = sub.break_by_name
            self.ac_zero_branch_by_name = sub.zero_branch_by_name
            self._build_ac_state_adapter(sub)
            self.measurements = sub.measurements
            self.active_measurements = sub.active_measurements
            self.active_z = sub.active_z
            self.active_weight = sub.active_weight
            self.active_angle_residual_mask = getattr(sub, "active_angle_residual_mask", angle_residual_mask(self.active_measurements))
            self.state_labels = list(sub.state_labels)
            self.n_state = int(sub.n_state)
            self.voltage_cols = getattr(sub, "voltage_cols", np.array([], dtype=np.int32))
            self.power_flow_state = sub._file_state().copy() if hasattr(sub, "_file_state") else sub.initial_state()
            self.flat_state = sub.initial_state()
            self.targeted_observability_pseudo_count = sub.targeted_observability_pseudo_count
            return True

        if has_dc:
            sub = self._dc_sub_estimator
            self._dc_sub_estimator = sub
            self._delegate_estimator = sub
            self._sub_estimators_enabled = True
            self.nodes = sub.nodes
            self.ac_nodes = []
            self.dc_nodes = sub.nodes
            self.dc_node_by_name = sub.node_by_name
            self.dc_branch_by_name = sub.branch_by_name
            self.dc_generator_by_name = sub.generator_by_name
            self.dc_load_by_name = sub.load_by_name
            self.dc_switch_by_name = sub.switch_by_name
            self.dc_break_by_name = sub.break_by_name
            self.dc_zero_branch_by_name = sub.zero_branch_by_name
            self.dcdc_by_name = sub.dcdc_by_name
            self.dc_reference_nodes = getattr(sub, "references", [])
            self.dc_node_voltage_measurements = sub.node_voltage_measurements
            self.dc_node_degrees = sub.node_degrees
            self.dc_voltage_state_col = sub.voltage_col.copy()
            self.measurements = sub.measurements
            self.active_measurements = sub.active_measurements
            self.active_z = sub.active_z
            self.active_weight = sub.active_weight
            self.active_angle_residual_mask = angle_residual_mask(self.active_measurements)
            self.state_labels = sub.state_labels
            self.n_state = sub.n_state
            self.voltage_cols = getattr(sub, "voltage_cols", np.array([], dtype=np.int32))
            self.power_flow_state = sub.initial_state()
            self.flat_state = sub.initial_state()
            self.targeted_observability_pseudo_count = sub.targeted_observability_pseudo_count
            return True

        return False

    def _delegate(self):
        return getattr(self, "_delegate_estimator", None) if getattr(self, "_sub_estimators_enabled", False) else None

    def _build_ac_state_adapter(self, sub: ACStateEstimator) -> None:
        """Reuse the canonical AC estimator layout for pure-AC delegation."""
        self.ac_state_layout = sub.state_layout()
        self.ac_node_by_idx = {node.idx: node for node in self.ac_nodes}
        self.ac_node_voltage_measurements = self._ac_node_voltage_measurements()
        self.ac_node_degrees = self._ac_node_incident_degrees()
        self.ac_reference_nodes = self._select_ac_reference_nodes()
        self.ac_reference_angle_by_pos = self._ac_reference_angle_offsets()
        self.ac_theta_state_col, self.ac_voltage_state_col = sub.state_cols_for_nodes(self.ac_nodes)
        self.ac_state_labels = list(sub.state_labels)
        self.ac_n_state = int(sub.n_state)
        self.ac_delegate_angle_delta_by_pos = self._build_ac_delegate_angle_delta(sub)

    def _build_ac_delegate_angle_delta(self, sub: ACStateEstimator) -> Dict[int, float]:
        """Map AC node positions from the delegate frame back to the hybrid reference frame."""
        if self.calc.ac_calc is None:
            return {}
        ac = self.calc.ac_calc
        theta, _voltage = sub._unpack_state(sub.initial_state())
        delta_by_pos: Dict[int, float] = {}
        for island in self.network.ac.islands:
            if not getattr(island, "is_alive", False):
                continue
            candidate = next((node for node in island.buses if node.idx in self.ac_node_by_name), None)
            if candidate is None:
                continue
            h_node = self.ac_node_by_name[candidate.name]
            if h_node.idx not in ac.node_pos or candidate.idx not in sub.node_pos:
                continue
            h_pos = int(ac.node_pos[h_node.idx])
            s_pos = int(sub.node_pos[candidate.idx])
            delta = float(self.power_flow_x[int(ac.theta_idx[h_pos])]) - float(theta[s_pos])
            for node in island.buses:
                if node.idx in ac.node_pos:
                    delta_by_pos[int(ac.node_pos[node.idx])] = delta
        return delta_by_pos

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
        return f"{prefix_map.get(prefix, prefix)}:{name}"

    @staticmethod
    def _ac_sub_state_label_to_hybrid(label: str) -> str:
        prefix, name = label.split(":", 1)
        if prefix == "theta":
            return f"AC_THETA:{name}"
        if prefix == "V":
            return f"AC_V:{name}"
        if prefix.startswith("I_") and prefix.endswith("_RE"):
            return f"AC_I_RE:{name}"
        if prefix.startswith("I_") and prefix.endswith("_IM"):
            return f"AC_I_IM:{name}"
        if prefix == "P_GEN":
            return f"AC_GEN_P:{name}"
        if prefix == "Q_GEN":
            return f"AC_GEN_Q:{name}"
        if prefix == "P_LOAD":
            return f"AC_LOAD_P:{name}"
        if prefix == "Q_LOAD":
            return f"AC_LOAD_Q:{name}"
        return label

    @staticmethod
    def _measurement_key(meas: Measurement) -> Tuple[str, str, str]:
        return (meas.device_type, meas.device_name, meas.meas_type)

    def _build_sub_estimator_composition_maps(self) -> None:
        """Map reusable subsystem estimator rows/columns into the hybrid state space."""
        self._ac_sub_to_hybrid_cols = np.array([], dtype=np.int32)
        self._active_ac_sub_measurements: List[Measurement] = []
        self._active_ac_sub_rows = np.array([], dtype=np.int32)
        self._active_ac_hybrid_rows = np.array([], dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(getattr(self, "active_measurements", [])), dtype=bool)
        self._dc_sub_to_hybrid_cols = np.array([], dtype=np.int32)
        self._active_dc_sub_measurements: List[Measurement] = []
        self._active_dc_sub_rows = np.array([], dtype=np.int32)
        self._active_dc_hybrid_rows = np.array([], dtype=np.int32)
        self._active_dc_delegated_row_mask = np.zeros(len(getattr(self, "active_measurements", [])), dtype=bool)
        self._build_ac_sub_estimator_composition_map()
        self._build_dc_sub_estimator_composition_map()

    def _build_ac_sub_estimator_composition_map(self) -> None:
        ac_sub = getattr(self, "_ac_sub_estimator", None)
        if ac_sub is None or self._delegate() is not None:
            return

        hybrid_col_by_label = {label: idx for idx, label in enumerate(self.state_labels)}
        mapped_cols = np.asarray(
            [hybrid_col_by_label.get(self._ac_sub_state_label_to_hybrid(label), -1) for label in ac_sub.state_labels],
            dtype=np.int32,
        )

        sub_rows_by_key: Dict[Tuple[str, str, str], List[int]] = {}
        for row, meas in enumerate(ac_sub.active_measurements):
            sub_rows_by_key.setdefault(self._measurement_key(meas), []).append(row)

        probe_x = ac_sub.initial_state()
        row_compatible = np.zeros(len(ac_sub.active_measurements), dtype=bool)
        try:
            probe_h = ac_sub.jacobian_sparse(probe_x, ac_sub.active_measurements).tocoo()
            row_compatible[:] = True
            if probe_h.nnz:
                sub_cols = probe_h.col.astype(np.int32, copy=False)
                sub_rows = probe_h.row.astype(np.int32, copy=False)
                bad = mapped_cols[sub_cols] < 0
                if np.any(bad):
                    row_compatible[np.unique(sub_rows[bad])] = False
        except Exception:
            row_compatible[:] = False

        hybrid_rows: List[int] = []
        sub_rows: List[int] = []
        sub_measurements: List[Measurement] = []
        for h_row, meas in enumerate(self.active_measurements):
            if meas.device_type.startswith("DC") or meas.device_type in ("DCDCConverter", "DCACConverter", "ACACConverter"):
                continue
            rows = sub_rows_by_key.get(self._measurement_key(meas))
            if not rows:
                continue
            sub_row = rows[0]
            sub_meas = ac_sub.active_measurements[sub_row]
            if not row_compatible[int(sub_row)]:
                continue
            rows.pop(0)
            hybrid_rows.append(h_row)
            sub_rows.append(sub_row)
            sub_measurements.append(sub_meas)

        if not hybrid_rows:
            return
        self._ac_sub_to_hybrid_cols = mapped_cols
        self._active_ac_sub_measurements = sub_measurements
        self._active_ac_sub_rows = np.asarray(sub_rows, dtype=np.int32)
        self._active_ac_hybrid_rows = np.asarray(hybrid_rows, dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True

    def _build_dc_sub_estimator_composition_map(self) -> None:
        dc_sub = getattr(self, "_dc_sub_estimator", None)
        if dc_sub is None or self._delegate() is not None:
            return

        hybrid_col_by_label = {label: idx for idx, label in enumerate(self.state_labels)}
        mapped_cols = []
        for label in dc_sub.state_labels:
            mapped_cols.append(hybrid_col_by_label.get(self._dc_sub_state_label_to_hybrid(label), -1))
        mapped_cols = np.asarray(mapped_cols, dtype=np.int32)
        if mapped_cols.size != dc_sub.n_state or np.any(mapped_cols < 0):
            return
        dc_sub_voltage_hybrid_cols = {
            int(mapped_cols[idx])
            for idx, label in enumerate(dc_sub.state_labels)
            if label.startswith("V:") and int(mapped_cols[idx]) >= 0
        }

        def dc_voltage_cols_compatible(meas: Measurement) -> bool:
            """Only delegate rows whose DC voltage dependencies use the same hybrid states."""
            dc = self.calc.dc_calc
            if dc is None:
                return False
            node_idxs: List[int] = []
            dtype = meas.device_type
            if dtype == "DCNode":
                node = self.dc_node_by_name.get(meas.device_name)
                if node is not None:
                    node_idxs.append(node.idx)
            elif dtype == "DCBranch":
                dev = self.dc_branch_by_name.get(meas.device_name)
                if dev is not None:
                    node_idxs.extend([dev.i_node, dev.j_node])
            elif dtype in ("DCSwitch", "DCZeroBranch", "DCBreak"):
                dev = (
                    self.dc_switch_by_name.get(meas.device_name)
                    if dtype == "DCSwitch"
                    else self.dc_zero_branch_by_name.get(meas.device_name)
                    if dtype == "DCZeroBranch"
                    else self.dc_break_by_name.get(meas.device_name)
                )
                if dev is not None:
                    node_idxs.extend([dev.i_node, dev.j_node])
            elif dtype == "DCLoad":
                dev = self.dc_load_by_name.get(meas.device_name)
                if dev is not None:
                    node_idxs.append(dev.node)
            elif dtype == "DCGenerator":
                dev = self.dc_generator_by_name.get(meas.device_name)
                if dev is not None:
                    node_idxs.append(dev.node)
            elif dtype == "DCDCConverter":
                dev = self.dcdc_by_name.get(meas.device_name)
                if dev is not None:
                    node_idxs.extend([dev.i_node, dev.j_node])
            else:
                return True

            for node_idx in node_idxs:
                if node_idx not in dc.alive_node_dict:
                    return False
                col = int(self.dc_voltage_state_col[int(dc.alive_node_dict[node_idx])])
                if col >= 0 and col not in dc_sub_voltage_hybrid_cols:
                    return False
            return True

        sub_rows_by_key: Dict[Tuple[str, str, str], List[int]] = {}
        for row, meas in enumerate(dc_sub.active_measurements):
            sub_rows_by_key.setdefault(self._measurement_key(meas), []).append(row)

        hybrid_rows: List[int] = []
        sub_rows: List[int] = []
        sub_measurements: List[Measurement] = []
        for h_row, meas in enumerate(self.active_measurements):
            rows = sub_rows_by_key.get(self._measurement_key(meas))
            if not rows:
                continue
            if not dc_voltage_cols_compatible(meas):
                continue
            sub_row = rows.pop(0)
            hybrid_rows.append(h_row)
            sub_rows.append(sub_row)
            sub_measurements.append(dc_sub.active_measurements[sub_row])

        if not hybrid_rows:
            return
        self._dc_sub_to_hybrid_cols = mapped_cols
        self._active_dc_sub_measurements = sub_measurements
        self._active_dc_sub_rows = np.asarray(sub_rows, dtype=np.int32)
        self._active_dc_hybrid_rows = np.asarray(hybrid_rows, dtype=np.int32)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True

    def _refresh_active_measurement_state_layout(self) -> None:
        """Rebuild active measurement arrays and all indexes that depend on them."""
        self.active_measurements = [m for m in self.measurements if m.valid and m.weight > 0.0]
        self._build_estimation_state_layout()
        self._rebase_ac_angle_measurements()
        self.active_measurements = [m for m in self.measurements if m.valid and m.weight > 0.0]
        self._append_ac_subsystem_active_measurements()
        self.active_z = np.fromiter((m.value for m in self.active_measurements), dtype=np.float64)
        self.active_weight = np.fromiter((m.weight for m in self.active_measurements), dtype=np.float64)
        self.active_angle_residual_mask = angle_residual_mask(self.active_measurements)
        self._build_sub_estimator_composition_maps()
        self._build_derivative_caches()
        self._build_static_jacobian_index()
        self._build_evaluation_index()
        self.power_flow_state = self._pack_estimation_state(self.power_flow_x, flat=False)
        self.flat_state = self._pack_estimation_state(self.flat_x, flat=True)
        self._last_written_state = None
        self._write_state(self.power_flow_state)

    def _append_ac_subsystem_active_measurements(self) -> None:
        """Append ACStateEstimator-owned internal rows to the hybrid active table."""
        ac_sub = getattr(self, "_ac_sub_estimator", None)
        if ac_sub is None or self._delegate() is not None:
            return
        coupled_ac_nodes = self._converter_coupled_ac_node_names()
        existing = {self._measurement_key(meas) for meas in self.active_measurements}
        for meas in ac_sub.active_measurements:
            key = self._measurement_key(meas)
            if key in existing:
                continue
            if meas.device_type != "ACPowerBalance":
                continue
            if meas.device_name in coupled_ac_nodes:
                continue
            if not meas.valid or meas.weight <= 0.0:
                continue
            self.active_measurements.append(meas)
            existing.add(key)

    def _converter_coupled_ac_node_names(self) -> set:
        names = set()
        for conv in self.dcac_by_name.values():
            node = self.ac_node_by_idx.get(conv.ac_node)
            if node is not None:
                names.add(node.name)
        for conv in self.acac_by_name.values():
            for node_idx in (conv.i_node, conv.j_node):
                node = self.ac_node_by_idx.get(node_idx)
                if node is not None:
                    names.add(node.name)
        return names

    def _angle_residual_mask(self, measurements: Sequence[Measurement]) -> np.ndarray:
        if measurements is self.active_measurements:
            return self.active_angle_residual_mask
        return angle_residual_mask(measurements)

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
    ) -> np.ndarray:
        return build_measurement_residual(z, z_est, self._angle_residual_mask(measurements))

    def _matches_active_measurements(self, measurements: Sequence[Measurement]) -> bool:
        return len(measurements) == len(self.active_measurements) and all(
            meas is active for meas, active in zip(measurements, self.active_measurements)
        )

    def _disable_angle_measurements(self) -> None:
        """Keep all AC phase-angle rows out of WLS; flat starts store them as zero."""
        for meas in self.measurements:
            if meas.meas_type in ANGLE_MEASUREMENT_TYPES:
                meas.valid = False
                if self.flat_start:
                    meas.value = 0.0

    def _disable_unavailable_measurements(self) -> None:
        """Keep invalid/off-topology measurement rows out of unit conversion and WLS."""
        device_maps = {
            "ACNode": self.ac_node_by_name,
            "DCNode": self.dc_node_by_name,
            "ACBranch": self.ac_branch_by_name,
            "ACTransformer": self.ac_transformer_by_name,
            "ACSwitch": self.ac_switch_by_name,
            "ACBreak": self.ac_break_by_name,
            "ACZeroBranch": self.ac_zero_branch_by_name,
            "ACGenerator": self.ac_generator_by_name,
            "ACLoad": self.ac_load_by_name,
            "DCBranch": self.dc_branch_by_name,
            "DCSwitch": self.dc_switch_by_name,
            "DCBreak": self.dc_break_by_name,
            "DCZeroBranch": self.dc_zero_branch_by_name,
            "DCZeroBranchConstraint": self.dc_zero_branch_by_name,
            "DCSwitchConstraint": self.dc_switch_by_name,
            "DCBreakConstraint": self.dc_break_by_name,
            "DCGenerator": self.dc_generator_by_name,
            "DCLoad": self.dc_load_by_name,
            "DCDCConverter": self.dcdc_by_name,
            "DCACConverter": self.dcac_by_name,
            "ACACConverter": self.acac_by_name,
        }
        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            devices = device_maps.get(meas.device_type)
            if devices is None or meas.device_name not in devices:
                meas.valid = False

    @staticmethod
    def _load_measurements(meas_file: Path) -> List[Measurement]:
        return _read_measurements_direct(meas_file)

    @staticmethod
    def _load_case(
        e_file: Path,
        params: StateEstimationParameters,
    ) -> Tuple[HybridPowerNetwork, HybridPowerFlowCalc, np.ndarray, np.ndarray]:
        """Load a hybrid case and build flat/E-file state seeds for the estimator."""
        network = HybridPowerNetwork.read_from_file(e_file)
        with contextlib.redirect_stdout(io.StringIO()):
            ac_warnings, ac_errors, dc_warnings, dc_errors = network.prepare(verbose=False)
        if ac_errors or dc_errors:
            raise RuntimeError(
                f"Topology check failed for {e_file}: "
                f"ac_errors={ac_errors}, dc_errors={dc_errors}, "
                f"ac_warnings={ac_warnings}, dc_warnings={dc_warnings}"
            )

        pure_ac = (
            bool(network.ac.nodes)
            and not network.dc.nodes
            and not network.dcac_converters
            and not network.acac_converters
        )
        calc = HybridPowerFlowCalc(
            network,
            tol=params.power_flow_tol,
            max_iter=params.power_flow_max_iter,
            min_voltage=params.power_flow_min_voltage,
            verbose=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
        if calc.ac_calc is not None and getattr(calc.ac_calc, "isl", None) is None:
            HybridStateEstimator._attach_array_ac_object_island(calc.ac_calc, network.ac)
        flat_x = calc.x.copy()
        file_x = HybridStateEstimator._file_state_vector(calc)
        if pure_ac and params.flat_start:
            power_flow_x = file_x
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
            if rc != 0 or not calc.converged:
                raise RuntimeError(
                    f"Hybrid power flow failed for {e_file}: "
                    f"rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}"
                )
            power_flow_x = calc.x.copy()
        calc._write_back(power_flow_x)
        return network, calc, flat_x, power_flow_x

    @staticmethod
    def _attach_array_ac_object_island(ac_calc: ACPowerFlowCalc, ac_grid) -> None:
        """Expose object device lists expected by SE while AC math uses array PPC."""
        ac_calc.isl = SimpleNamespace(
            buses=[
                bus
                for bus in getattr(ac_grid, "buses", [])
                if getattr(bus, "is_alive", False)
            ] or [node for node in ac_grid.nodes if getattr(node, "is_alive", False)],
            gens=ac_grid.generators,
            loads=ac_grid.loads,
            branches=ac_grid.branches,
            transformers=ac_grid.transformers,
            shunt_compensators=ac_grid.shunt_compensators,
            zero_branches=ac_grid.zero_branches,
            switches=ac_grid.switches,
            slack_nodes=[
                node
                for island in getattr(ac_grid, "islands", [])
                for node in getattr(island, "slack_nodes", [])
                if getattr(island, "is_alive", False)
            ],
        )

    @staticmethod
    def _file_state_vector(calc: HybridPowerFlowCalc) -> np.ndarray:
        """Project E-file node values into the hybrid Newton vector without rerunning LF."""
        x = calc.x.copy()
        ac = calc.ac_calc
        if ac is not None:
            theta = np.asarray([float(getattr(node, "angle", 0.0) or 0.0) for node in ac.node_list], dtype=np.float64)
            voltage = np.asarray([float(getattr(node, "voltage", 1.0) or 1.0) for node in ac.node_list], dtype=np.float64)
            if hasattr(ac, "theta_spec") and ac.theta_spec.size:
                ac.theta_spec[: theta.size] = theta
            if hasattr(ac, "V_spec") and ac.V_spec.size:
                ac.V_spec[: voltage.size] = voltage
            for pos, idx in ac.theta_idx.items():
                x[int(idx)] = theta[int(pos)]
            for pos, idx in ac.V_idx.items():
                x[int(ac.n_theta + idx)] = voltage[int(pos)]

        dc = calc.dc_calc
        if dc is not None:
            dc_start = calc.ac_size
            for node in getattr(dc, "alive_nodes", []):
                pos = dc.alive_node_dict.get(node.idx)
                if pos is not None:
                    x[dc_start + int(pos)] = float(getattr(node, "voltage", 1.0) or 1.0)
        return x

    def _ac_voltage_base(self, node_idx: int) -> float:
        return float(self.network.ac.node_dict[node_idx].vbase)

    def _dc_voltage_base(self, node_idx: int) -> float:
        return float(self.network.dc.node_dict[node_idx].vbase)

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

    def _convert_measurements_to_pu(self) -> None:
        """Normalize file measurement values to the internal hybrid state units."""
        def ac_terminal_scale(meas: Measurement, device) -> float:
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

        def dc_terminal_scale(meas: Measurement, device) -> float:
            if meas.meas_type.startswith("P_"):
                return self._power_file_base()
            if meas.meas_type.endswith("_FROM"):
                node_idx = device.i_node
            elif meas.meas_type.endswith("_TO"):
                node_idx = device.j_node
            else:
                return 1.0
            if meas.meas_type.startswith("V_"):
                return self._dc_voltage_file_base(node_idx)
            if meas.meas_type.startswith("I_"):
                return self._dc_current_base(node_idx)
            return 1.0

        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            scale = 1.0
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                if mtype == "V":
                    scale = self._ac_voltage_file_base(self.ac_node_by_name[meas.device_name].idx)
                elif mtype in ("ANGLE", "THETA"):
                    meas.value = math.radians(float(meas.value))
                    continue
            elif meas.device_type == "DCNode":
                if mtype == "V":
                    scale = self._dc_voltage_file_base(self.dc_node_by_name[meas.device_name].idx)
            elif meas.device_type == "ACBranch":
                scale = ac_terminal_scale(meas, self.ac_branch_by_name[meas.device_name])
            elif meas.device_type == "ACTransformer":
                scale = ac_terminal_scale(meas, self.ac_transformer_by_name[meas.device_name])
            elif meas.device_type == "ACSwitch":
                scale = ac_terminal_scale(meas, self.ac_switch_by_name[meas.device_name])
            elif meas.device_type == "ACBreak":
                scale = ac_terminal_scale(meas, self.ac_break_by_name[meas.device_name])
            elif meas.device_type == "ACZeroBranch":
                scale = ac_terminal_scale(meas, self.ac_zero_branch_by_name[meas.device_name])
            elif meas.device_type == "DCBranch":
                scale = dc_terminal_scale(meas, self.dc_branch_by_name[meas.device_name])
            elif meas.device_type == "DCSwitch":
                scale = dc_terminal_scale(meas, self.dc_switch_by_name[meas.device_name])
            elif meas.device_type == "DCBreak":
                scale = dc_terminal_scale(meas, self.dc_break_by_name[meas.device_name])
            elif meas.device_type == "DCZeroBranch":
                scale = dc_terminal_scale(meas, self.dc_zero_branch_by_name[meas.device_name])
            elif meas.device_type == "DCZeroBranchConstraint":
                if mtype == "V_DIFF":
                    scale = self._dc_voltage_file_base(self.dc_zero_branch_by_name[meas.device_name].i_node)
            elif meas.device_type == "DCSwitchConstraint":
                if mtype == "V_DIFF":
                    scale = self._dc_voltage_file_base(self.dc_switch_by_name[meas.device_name].i_node)
            elif meas.device_type == "DCBreakConstraint":
                if mtype == "V_DIFF":
                    scale = self._dc_voltage_file_base(self.dc_break_by_name[meas.device_name].i_node)
            elif meas.device_type == "DCDCConverter":
                scale = dc_terminal_scale(meas, self.dcdc_by_name[meas.device_name])
            elif meas.device_type == "ACGenerator":
                gen = self.ac_generator_by_name[meas.device_name]
                if mtype in ("P_GEN", "Q_GEN"):
                    scale = self._power_file_base()
                elif mtype == "V_GEN":
                    scale = self._ac_voltage_file_base(gen.node)
                elif mtype == "I_GEN":
                    scale = self._ac_current_base(gen.node)
            elif meas.device_type == "ACLoad":
                load = self.ac_load_by_name[meas.device_name]
                if mtype in ("P_LOAD", "Q_LOAD"):
                    scale = self._power_file_base()
                elif mtype == "V_LOAD":
                    scale = self._ac_voltage_file_base(load.node)
                elif mtype == "I_LOAD":
                    scale = self._ac_current_base(load.node)
            elif meas.device_type == "DCGenerator":
                gen = self.dc_generator_by_name[meas.device_name]
                if mtype == "P_GEN":
                    scale = self._power_file_base()
                elif mtype == "V_GEN":
                    scale = self._dc_voltage_file_base(gen.node)
                elif mtype == "I_GEN":
                    scale = self._dc_current_base(gen.node)
            elif meas.device_type == "DCLoad":
                load = self.dc_load_by_name[meas.device_name]
                if mtype == "P_LOAD":
                    scale = self._power_file_base()
                elif mtype == "V_LOAD":
                    scale = self._dc_voltage_file_base(load.node)
                elif mtype == "I_LOAD":
                    scale = self._dc_current_base(load.node)
            elif meas.device_type == "DCACConverter":
                conv = self.dcac_by_name[meas.device_name]
                if mtype in ("P_DC", "P_AC", "Q_AC"):
                    scale = self._power_file_base()
                elif mtype == "V_DC":
                    scale = self._dc_voltage_file_base(conv.dc_node)
                elif mtype == "I_DC":
                    scale = self._dc_current_base(conv.dc_node)
                elif mtype == "V_AC":
                    scale = self._ac_voltage_file_base(conv.ac_node)
                elif mtype == "I_AC":
                    scale = self._ac_current_base(conv.ac_node)
            elif meas.device_type == "ACACConverter":
                scale = ac_terminal_scale(meas, self.acac_by_name[meas.device_name])
            meas.value = float(meas.value) / scale

    def _active_device_keys(self) -> set:
        """Return devices that already have at least one usable real measurement."""
        return {
            (meas.device_type, meas.device_name)
            for meas in self.measurements
            if meas.valid and meas.weight > 0.0
        }

    def _active_measurement_keys(self) -> set:
        """Return usable measurement keys at device and measurement-type granularity."""
        return {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in self.measurements
            if meas.valid and meas.weight > 0.0
        }

    def _append_pseudo_measurement(
        self,
        next_idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        value: float,
    ) -> int:
        self.measurements.append(
            Measurement(
                idx=next_idx,
                name=name,
                device_type=device_type,
                device_name=device_name,
                meas_type=meas_type,
                weight=self.pseudo_measurement_weight,
                valid=True,
                value=float(value),
            )
        )
        return next_idx + 1

    @staticmethod
    def _load_zip_coefficients(load) -> Tuple[float, float, float, float, float, float]:
        """Return ZIP coefficients from the shared load field names."""
        pbase = float(getattr(load, "pbase", 1.0) or 0.0)
        qbase = float(getattr(load, "qbase", 1.0) or 0.0)
        p0 = pbase * float(getattr(load, "pv0", 0.0) or 0.0)
        p1 = pbase * float(getattr(load, "pv1", 0.0) or 0.0)
        p2 = pbase * float(getattr(load, "pv2", 0.0) or 0.0)
        q0 = qbase * float(getattr(load, "qv0", 0.0) or 0.0)
        q1 = qbase * float(getattr(load, "qv1", 0.0) or 0.0)
        q2 = qbase * float(getattr(load, "qv2", 0.0) or 0.0)
        return p0, p1, p2, q0, q1, q2

    @staticmethod
    def _ac_generator_pseudo_power(gen) -> Tuple[float, float]:
        # Hybrid load flow writes solved injections back before the estimator is built.
        p = getattr(gen, "p", None)
        q = getattr(gen, "q", None)
        if p is not None and q is not None:
            return float(p), float(q)
        return float(getattr(gen, "p_set", 0.0) or 0.0), float(getattr(gen, "q_set", 0.0) or 0.0)

    @staticmethod
    def _ac_load_pseudo_power(load) -> Tuple[float, float]:
        # Prefer solved load values; fall back to the ZIP model at the current seed voltage.
        p = getattr(load, "p", None)
        q = getattr(load, "q", None)
        if p is not None and q is not None:
            return float(p), float(q)
        node = getattr(load, "node_obj", None)
        voltage = float(getattr(node, "voltage", 1.0) or 1.0)
        p0, p1, p2, q0, q1, q2 = HybridStateEstimator._load_zip_coefficients(load)
        p = p0 + p1 * voltage + p2 * voltage * voltage
        q = q0 + q1 * voltage + q2 * voltage * voltage
        return float(p), float(q)

    @staticmethod
    def _dc_generator_pseudo_power(gen) -> float:
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
    def _dc_load_pseudo_power(load) -> float:
        # Prefer solved load values; fall back to the voltage-dependent load model.
        p = getattr(load, "p", None)
        if p is not None:
            return float(p)
        node = getattr(load, "node_obj", None)
        voltage = float(getattr(node, "voltage", 1.0) or 1.0)
        p0, p1, p2, _, _, _ = HybridStateEstimator._load_zip_coefficients(load)
        return float(p0 + p1 * voltage + p2 * voltage * voltage)

    def _active_ac_angle_measurement_counts(self) -> Dict[str, int]:
        """Count usable local P/Q or direct angle measurements per AC node."""
        counts: Dict[str, int] = {}

        def add(node_name: str, amount: int = 1) -> None:
            counts[node_name] = counts.get(node_name, 0) + amount

        for meas in self.measurements:
            if not meas.valid or meas.weight <= 0.0:
                continue
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                if mtype in ("ANGLE", "THETA") and meas.device_name in self.ac_node_by_name:
                    add(meas.device_name, 2)
                continue
            if not mtype.startswith(("P", "Q")):
                continue
            if meas.device_type == "ACBranch":
                dev = self.ac_branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACTransformer":
                dev = self.ac_transformer_by_name.get(meas.device_name)
            elif meas.device_type == "ACZeroBranch":
                dev = self.ac_zero_branch_by_name.get(meas.device_name)
            elif meas.device_type == "ACSwitch":
                dev = self.ac_switch_by_name.get(meas.device_name)
            elif meas.device_type == "ACBreak":
                dev = self.ac_break_by_name.get(meas.device_name)
            elif meas.device_type == "ACACConverter":
                dev = self.acac_by_name.get(meas.device_name)
            else:
                dev = None
            if dev is not None:
                for attr in ("i_node_obj", "j_node_obj"):
                    node = getattr(dev, attr, None)
                    if node is not None:
                        add(node.name)
                continue
            if meas.device_type == "DCACConverter":
                dev = self.dcac_by_name.get(meas.device_name)
                node = getattr(dev, "ac_node_obj", None) if dev is not None else None
                if node is not None:
                    add(node.name)
            elif meas.device_type == "ACGenerator":
                dev = self.ac_generator_by_name.get(meas.device_name)
                node = getattr(dev, "node_obj", None) if dev is not None else None
                if node is not None:
                    add(node.name)
        return counts

    def _add_pseudo_topology_measurements(self, next_idx: int) -> int:
        """Add weak priors for topology devices that have no usable measurement row."""
        measured_keys = self._active_measurement_keys()
        ac_node_voltage = self._ac_node_voltage_measurements() if self.calc.ac_calc is not None else {}
        dc_node_voltage = self._dc_node_voltage_measurements() if self.calc.dc_calc is not None else {}

        def ac_terminal_voltage(node_obj) -> float:
            if node_obj is not None and node_obj.idx in ac_node_voltage:
                return float(ac_node_voltage[node_obj.idx])
            return float(getattr(node_obj, "voltage", 1.0) or 1.0)

        def dc_terminal_voltage(node_obj) -> float:
            if node_obj is not None and node_obj.idx in dc_node_voltage:
                return float(dc_node_voltage[node_obj.idx])
            return float(getattr(node_obj, "voltage", 1.0) or 1.0)

        for device_type, devices in (
            ("ACZeroBranch", sorted(self.ac_zero_branch_by_name.values(), key=lambda item: item.idx)),
            ("ACSwitch", sorted(self.ac_switch_by_name.values(), key=lambda item: item.idx)),
            ("ACBreak", sorted(self.ac_break_by_name.values(), key=lambda item: item.idx)),
        ):
            for dev in devices:
                voltage = ac_terminal_voltage(getattr(dev, "i_node_obj", None))
                values = (
                    ("P_FROM", float(getattr(dev, "p", 0.0) or 0.0)),
                    ("Q_FROM", float(getattr(dev, "q", 0.0) or 0.0)),
                    ("V_FROM", voltage),
                    ("I_FROM", abs(getattr(dev, "current", 0.0) or 0.0)),
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

        for device_type, devices in (
            ("DCSwitch", sorted(self.dc_switch_by_name.values(), key=lambda item: item.idx)),
            ("DCBreak", sorted(self.dc_break_by_name.values(), key=lambda item: item.idx)),
            ("DCZeroBranch", sorted(self.dc_zero_branch_by_name.values(), key=lambda item: item.idx)),
        ):
            for dev in devices:
                voltage = dc_terminal_voltage(getattr(dev, "i_node_obj", None))
                values = [
                    ("P_FROM", float(getattr(dev, "p", 0.0) or 0.0)),
                    ("V_FROM", voltage),
                ]
                if not (device_type == "DCZeroBranch" and dev.i_node in dc_node_voltage):
                    values.append(("I_FROM", float(getattr(dev, "current", 0.0) or 0.0)))
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
        measured_devices = self._active_device_keys()
        next_idx = max((meas.idx for meas in self.measurements), default=0) + 1
        next_idx = self._add_pseudo_topology_measurements(next_idx)

        for gen in sorted(self.ac_generator_by_name.values(), key=lambda item: item.idx):
            if ("ACGenerator", gen.name) in measured_devices:
                continue
            p, q = self._ac_generator_pseudo_power(gen)
            voltage = float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0)
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{gen.name}",
                "ACGenerator",
                gen.name,
                "P_GEN",
                p,
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_q_{gen.name}",
                "ACGenerator",
                gen.name,
                "Q_GEN",
                q,
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{gen.name}",
                "ACGenerator",
                gen.name,
                "V_GEN",
                voltage,
            )

        for load in sorted(self.ac_load_by_name.values(), key=lambda item: item.idx):
            if ("ACLoad", load.name) in measured_devices:
                continue
            p, q = self._ac_load_pseudo_power(load)
            voltage = float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0)
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{load.name}",
                "ACLoad",
                load.name,
                "P_LOAD",
                p,
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_q_{load.name}",
                "ACLoad",
                load.name,
                "Q_LOAD",
                q,
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{load.name}",
                "ACLoad",
                load.name,
                "V_LOAD",
                voltage,
            )

        for gen in sorted(self.dc_generator_by_name.values(), key=lambda item: item.idx):
            if ("DCGenerator", gen.name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{gen.name}",
                "DCGenerator",
                gen.name,
                "P_GEN",
                self._dc_generator_pseudo_power(gen),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{gen.name}",
                "DCGenerator",
                gen.name,
                "V_GEN",
                float(getattr(getattr(gen, "node_obj", None), "voltage", 1.0) or 1.0),
            )

        for load in sorted(self.dc_load_by_name.values(), key=lambda item: item.idx):
            if ("DCLoad", load.name) in measured_devices:
                continue
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_p_{load.name}",
                "DCLoad",
                load.name,
                "P_LOAD",
                self._dc_load_pseudo_power(load),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_{load.name}",
                "DCLoad",
                load.name,
                "V_LOAD",
                float(getattr(getattr(load, "node_obj", None), "voltage", 1.0) or 1.0),
            )

        for conv in self.dcdc_by_name.values():
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
        for conv in self.dcac_by_name.values():
            if ("DCACConverter", conv.name) in measured_devices:
                continue
            for meas_type, attr in (("P_DC", "dc_p"), ("P_AC", "ac_p"), ("Q_AC", "ac_q")):
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_{meas_type.lower()}_{conv.name}",
                    "DCACConverter",
                    conv.name,
                    meas_type,
                    float(getattr(conv, attr, 0.0) or 0.0),
                )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_dc_{conv.name}",
                "DCACConverter",
                conv.name,
                "V_DC",
                float(getattr(getattr(conv, "dc_node_obj", None), "voltage", 1.0) or 1.0),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_ac_{conv.name}",
                "DCACConverter",
                conv.name,
                "V_AC",
                float(getattr(getattr(conv, "ac_node_obj", None), "voltage", 1.0) or 1.0),
            )
        for conv in self.acac_by_name.values():
            if ("ACACConverter", conv.name) in measured_devices:
                continue
            for meas_type, attr in (
                ("P_FROM", "i_p"),
                ("Q_FROM", "i_q"),
                ("P_TO", "j_p"),
                ("Q_TO", "j_q"),
            ):
                next_idx = self._append_pseudo_measurement(
                    next_idx,
                    f"pseudo_{meas_type.lower()}_{conv.name}",
                    "ACACConverter",
                    conv.name,
                    meas_type,
                    float(getattr(conv, attr, 0.0) or 0.0),
                )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_from_{conv.name}",
                "ACACConverter",
                conv.name,
                "V_FROM",
                float(getattr(getattr(conv, "i_node_obj", None), "voltage", 1.0) or 1.0),
            )
            next_idx = self._append_pseudo_measurement(
                next_idx,
                f"pseudo_v_to_{conv.name}",
                "ACACConverter",
                conv.name,
                "V_TO",
                float(getattr(getattr(conv, "j_node_obj", None), "voltage", 1.0) or 1.0),
            )

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch remaining rank deficiencies until observable or the configured cap is reached."""
        total_added = 0
        max_count = max(0, int(self.targeted_pseudo_measurement_max))
        while total_added < max_count:
            observability = self.observability_analysis()
            if observability.observable:
                break
            next_idx = max((meas.idx for meas in self.measurements), default=0) + 1
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
            self._refresh_active_measurement_state_layout()
        return total_added

    def _unanchored_ac_angle_state_labels(self) -> List[str]:
        """Return one AC angle state per structurally unanchored angle component."""
        H = self.jacobian_sparse(self.initial_state())
        return unanchored_angle_state_labels(H, self.state_labels, "AC_THETA:")

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_label: str,
        existing_keys: set,
        existing_names: set,
        max_add: int,
    ) -> Tuple[int, int]:
        """Translate a weak state label into the smallest useful pseudo measurement."""
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

        if prefix == "AC_THETA" and name in self.ac_node_by_name:
            return next_idx, 0
        if prefix == "AC_V" and name in self.ac_node_by_name:
            return next_idx, 0
        if prefix == "DC_V" and name in self.dc_node_by_name:
            return next_idx, 0

        if prefix in ("AC_I_RE", "AC_I_IM"):
            dev = self.ac_switch_by_name.get(name) or self.ac_zero_branch_by_name.get(name) or self.ac_break_by_name.get(name)
            if dev is None:
                return next_idx, 0
            device_type = (
                "ACSwitch"
                if name in self.ac_switch_by_name
                else "ACBreak"
                if name in self.ac_break_by_name
                else "ACZeroBranch"
            )
            next_idx, added_p = add(
                device_type,
                name,
                "P_FROM",
                float(getattr(dev, "p", 0.0) or 0.0),
            )
            next_idx, added_q = add(
                device_type,
                name,
                "Q_FROM",
                float(getattr(dev, "q", 0.0) or 0.0),
            )
            next_idx, added_p_to = add(device_type, name, "P_TO", -float(getattr(dev, "p", 0.0) or 0.0))
            next_idx, added_q_to = add(device_type, name, "Q_TO", -float(getattr(dev, "q", 0.0) or 0.0))
            return next_idx, added_p + added_q + added_p_to + added_q_to

        if prefix == "DC_I":
            dev = self.dc_switch_by_name.get(name) or self.dc_zero_branch_by_name.get(name) or self.dc_break_by_name.get(name)
            if dev is None:
                return next_idx, 0
            return add(
                "DCSwitch" if name in self.dc_switch_by_name else "DCBreak" if name in self.dc_break_by_name else "DCZeroBranch",
                name,
                "I_FROM",
                float(getattr(dev, "current", 0.0) or 0.0),
            )

        if prefix == "DCDC_P_FROM" and name in self.dcdc_by_name:
            return add("DCDCConverter", name, "P_FROM", float(getattr(self.dcdc_by_name[name], "i_p", 0.0) or 0.0))
        if prefix == "DCDC_P_TO" and name in self.dcdc_by_name:
            return add("DCDCConverter", name, "P_TO", float(getattr(self.dcdc_by_name[name], "j_p", 0.0) or 0.0))
        if prefix == "DC_VGEN_P" and name in self.dc_generator_by_name:
            return add("DCGenerator", name, "P_GEN", self._dc_generator_pseudo_power(self.dc_generator_by_name[name]))

        dcac_map = {
            "DCAC_P_DC": ("P_DC", "dc_p"),
            "DCAC_P_AC": ("P_AC", "ac_p"),
            "DCAC_Q_AC": ("Q_AC", "ac_q"),
        }
        if prefix in dcac_map and name in self.dcac_by_name:
            meas_type, attr = dcac_map[prefix]
            return add("DCACConverter", name, meas_type, float(getattr(self.dcac_by_name[name], attr, 0.0) or 0.0))

        acac_map = {
            "ACAC_P_FROM": ("P_FROM", "i_p"),
            "ACAC_Q_FROM": ("Q_FROM", "i_q"),
            "ACAC_P_TO": ("P_TO", "j_p"),
            "ACAC_Q_TO": ("Q_TO", "j_q"),
        }
        if prefix in acac_map and name in self.acac_by_name:
            meas_type, attr = acac_map[prefix]
            return add("ACACConverter", name, meas_type, float(getattr(self.acac_by_name[name], attr, 0.0) or 0.0))

        return next_idx, 0

    def _add_dc_zero_branch_constraint_measurements(self) -> None:
        """Add ideal DC voltage-equality constraints for zero branches and closed switches."""
        existing = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in self.measurements
            if meas.valid
            and meas.weight > 0.0
            and meas.device_type in ("DCZeroBranchConstraint", "DCSwitchConstraint", "DCBreakConstraint")
        }
        next_idx = max((meas.idx for meas in self.measurements), default=0) + 1
        weight = 10.0
        ideal_devices = [
            ("DCZeroBranchConstraint", zbr)
            for zbr in sorted(self.dc_zero_branch_by_name.values(), key=lambda item: item.idx)
        ]
        ideal_devices.extend(
            ("DCSwitchConstraint", sw)
            for sw in sorted(self.dc_switch_by_name.values(), key=lambda item: item.idx)
        )
        ideal_devices.extend(
            ("DCBreakConstraint", brk)
            for brk in sorted(self.dc_break_by_name.values(), key=lambda item: item.idx)
        )
        for device_type, dev in ideal_devices:
            key = (device_type, dev.name, "V_DIFF")
            if key in existing:
                continue
            self.measurements.append(
                Measurement(
                    idx=next_idx,
                    name=f"constraint_v_diff_{dev.name}",
                    device_type=device_type,
                    device_name=dev.name,
                    meas_type="V_DIFF",
                    weight=weight,
                    valid=True,
                    value=0.0,
                )
            )
            next_idx += 1

    def _build_state_labels(self) -> List[str]:
        """Create readable labels for the full hybrid Newton-vector layout."""
        labels = []
        if self.calc.ac_calc is not None:
            ac = self.calc.ac_calc
            node_names = [node.name for node in ac.node_list]
            labels.extend(f"AC_THETA:{node_names[int(pos)]}" for pos in ac.theta_unknown)
            labels.extend(f"AC_V:{node_names[int(pos)]}" for pos in ac.V_unknown)
            labels.extend(f"AC_PHI_RE:{idx}" for idx in range(ac.N_phi))
            labels.extend(f"AC_PHI_IM:{idx}" for idx in range(ac.N_phi))
        if self.calc.dc_calc is not None:
            dc = self.calc.dc_calc
            labels.extend(f"DC_V:{node.name}" for node in dc.alive_nodes)
            labels.extend(f"DC_PHI:{idx}" for idx in range(dc.N_phi))
            for idx in range(dc.N_dcdc):
                conv = dc.model.dcdc_converters[int(dc.dcdc_idx[idx])]
                labels.append(f"DCDC_P_FROM:{conv.name}")
                labels.append(f"DCDC_P_TO:{conv.name}")
        for conv, _, _, _ in self.calc.dcac_converters:
            labels.extend((f"DCAC_P_DC:{conv.name}", f"DCAC_P_AC:{conv.name}", f"DCAC_Q_AC:{conv.name}"))
        for conv, _, _, _ in self.calc.acac_converters:
            labels.extend(
                (
                    f"ACAC_P_FROM:{conv.name}",
                    f"ACAC_Q_FROM:{conv.name}",
                    f"ACAC_P_TO:{conv.name}",
                    f"ACAC_Q_TO:{conv.name}",
                )
            )
        if len(labels) < self.calc.total_vars:
            labels.extend(f"x:{idx}" for idx in range(len(labels), self.calc.total_vars))
        return labels[: self.calc.total_vars]

    def _ac_node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real ACNode voltage measurements keyed by AC node index."""
        best: Dict[int, Tuple[float, float]] = {}
        for meas in self.measurements:
            if (
                not meas.valid
                or meas.weight <= 0.0
                or meas.device_type != "ACNode"
                or meas.meas_type != "V"
                or meas.name.startswith("pseudo_")
                or meas.device_name not in self.ac_node_by_name
            ):
                continue
            node_idx = self.ac_node_by_name[meas.device_name].idx
            current = best.get(node_idx)
            if current is None or meas.weight > current[0]:
                best[node_idx] = (float(meas.weight), float(meas.value))
        return {node_idx: value for node_idx, (_weight, value) in best.items()}

    def _ac_node_incident_degrees(self) -> Dict[int, int]:
        """Count live AC branch, transformer, switch and zero-branch terminals."""
        degrees = {node.idx: 0 for node in self.ac_nodes}
        device_groups = (
            self.ac_branch_by_name.values(),
            self.ac_transformer_by_name.values(),
            self.ac_zero_branch_by_name.values(),
            self.ac_switch_by_name.values(),
            self.ac_break_by_name.values(),
        )
        for devices in device_groups:
            for dev in devices:
                if dev.i_node in degrees:
                    degrees[dev.i_node] += 1
                if dev.j_node in degrees:
                    degrees[dev.j_node] += 1
        return degrees

    def _select_ac_reference_nodes(self) -> List[object]:
        """Choose one measured high-degree AC V/angle reference per live AC island."""
        voltage_measurements = self.ac_node_voltage_measurements
        degrees = self.ac_node_degrees
        references = []
        for island in self.network.ac.islands:
            if not getattr(island, "is_alive", False):
                continue
            candidates = [
                node
                for node in island.buses
                if node.idx in self.ac_node_by_idx and node.idx in voltage_measurements
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

    def _ac_reference_angle_offsets(self) -> Dict[int, float]:
        """Map each AC node position to the original angle of its selected island reference."""
        ac = self.calc.ac_calc
        if ac is None:
            return {}
        ref_by_node_idx = {int(node.idx): node for node in self.ac_reference_nodes}
        offsets: Dict[int, float] = {}

        def original_angle(pos: int) -> float:
            return self._ac_original_angle_by_pos(pos)

        for island in self.network.ac.islands:
            if not getattr(island, "is_alive", False):
                continue
            ref = next((node for node in island.buses if int(node.idx) in ref_by_node_idx), None)
            if ref is None or ref.idx not in ac.node_pos:
                continue
            offset = original_angle(ac.node_pos[ref.idx])
            for node in island.buses:
                if node.idx in ac.node_pos:
                    offsets[int(ac.node_pos[node.idx])] = offset
        return offsets

    def _ac_angle_reference_for_node(self, node_idx: int) -> float:
        ac = self.calc.ac_calc
        if ac is None or node_idx not in ac.node_pos:
            return 0.0
        return float(self.ac_reference_angle_by_pos.get(int(ac.node_pos[node_idx]), 0.0))

    def _ac_original_angle_by_pos(self, node_pos: int) -> float:
        ac = self.calc.ac_calc
        if ac is not None and node_pos in ac.theta_idx:
            return float(self.power_flow_x[int(ac.theta_idx[node_pos])])
        if int(node_pos) < len(self.ac_original_theta_spec):
            return float(self.ac_original_theta_spec[int(node_pos)])
        return 0.0

    def _ac_original_node_angle(self, node_idx: int) -> float:
        ac = self.calc.ac_calc
        if ac is not None and node_idx in ac.node_pos:
            pos = int(ac.node_pos[node_idx])
            return self._ac_original_angle_by_pos(pos)
        node = self.ac_node_by_idx.get(node_idx)
        return float(getattr(node, "angle", 0.0) or 0.0)

    def _rebase_ac_angle_measurements(self) -> None:
        """Convert absolute ACNode angle measurements to the selected reference frame once."""
        if self.calc.ac_calc is None:
            return
        done = self._rebased_ac_angle_measurement_names
        for meas in self.measurements:
            if (
                meas.name not in done
                and meas.valid
                and meas.weight > 0.0
                and meas.device_type == "ACNode"
                and meas.meas_type in ("ANGLE", "THETA")
                and meas.device_name in self.ac_node_by_name
            ):
                node = self.ac_node_by_name[meas.device_name]
                meas.value -= self._ac_angle_reference_for_node(node.idx)
                done.add(meas.name)

    def _dc_node_voltage_measurements(self) -> Dict[int, float]:
        """Return valid real DCNode voltage measurements keyed by DC node index."""
        best: Dict[int, Tuple[float, float]] = {}
        for meas in self.measurements:
            if (
                not meas.valid
                or meas.weight <= 0.0
                or meas.device_type != "DCNode"
                or meas.meas_type != "V"
                or meas.name.startswith("pseudo_")
                or meas.device_name not in self.dc_node_by_name
            ):
                continue
            node_idx = self.dc_node_by_name[meas.device_name].idx
            current = best.get(node_idx)
            if current is None or meas.weight > current[0]:
                best[node_idx] = (float(meas.weight), float(meas.value))
        return {node_idx: value for node_idx, (_weight, value) in best.items()}

    def _dc_node_incident_degrees(self) -> Dict[int, int]:
        """Count live DC branch, switch and zero-branch terminals for reference selection."""
        degrees = {node.idx: 0 for node in self.dc_nodes}
        device_groups = (
            self.dc_branch_by_name.values(),
            self.dc_zero_branch_by_name.values(),
            self.dc_switch_by_name.values(),
            self.dc_break_by_name.values(),
        )
        for devices in device_groups:
            for dev in devices:
                if dev.i_node in degrees:
                    degrees[dev.i_node] += 1
                if dev.j_node in degrees:
                    degrees[dev.j_node] += 1
        return degrees

    def _select_dc_reference_nodes(self) -> List[object]:
        """Choose one measured high-degree DC voltage reference per live DC topology island."""
        references = []
        voltage_measurements = self.dc_node_voltage_measurements
        degrees = self.dc_node_degrees
        for island in self.network.dc.islands:
            if not getattr(island, "is_alive", False):
                continue
            candidates = [
                node
                for node in island.buses
                if node.idx in self.dc_node_by_idx and node.idx in voltage_measurements
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

    def _build_estimation_state_layout(self) -> None:
        """Build the compact WLS state layout from the larger hybrid load-flow vector."""
        self.state_labels: List[str] = []
        self.full_col_for_state: List[int] = []
        self.voltage_cols_list: List[int] = []

        self.ac_zero_re_cols = np.array([], dtype=np.int32)
        self.ac_zero_im_cols = np.array([], dtype=np.int32)
        self.ac_zero_phi_a = np.array([], dtype=np.int32)
        self.ac_zero_phi_b = np.array([], dtype=np.int32)
        self.ac_phi_free = np.array([], dtype=np.int32)
        self.ac_phi_pinv = np.zeros((0, 0), dtype=np.float64)

        self.dc_zero_cols = np.array([], dtype=np.int32)
        self.dc_zero_phi_a = np.array([], dtype=np.int32)
        self.dc_zero_phi_b = np.array([], dtype=np.int32)
        self.dc_phi_free = np.array([], dtype=np.int32)
        self.dc_phi_pinv = np.zeros((0, 0), dtype=np.float64)

        self.dc_v_generator_states: List[Tuple[object, int]] = []
        self.tied_full_cols: List[Tuple[int, int]] = []
        self.fixed_full_col_values: Dict[int, float] = {}
        self.ac_sub_seed_hybrid_cols = np.array([], dtype=np.int32)
        self.ac_sub_seed_sub_cols = np.array([], dtype=np.int32)
        self.ac_zero_current_cols_by_name: Dict[str, Tuple[int, int]] = {}
        self.dc_zero_current_col_by_name: Dict[str, int] = {}
        self.dc_v_generator_col_by_name: Dict[str, int] = {}
        self.ac_generator_p_col_by_name: Dict[str, int] = {}
        self.ac_generator_q_col_by_name: Dict[str, int] = {}
        self.ac_load_p_col_by_name: Dict[str, int] = {}
        self.ac_load_q_col_by_name: Dict[str, int] = {}
        self.ac_reference_nodes: List[object] = []
        self.ac_reference_voltage_by_pos: Dict[int, float] = {}
        self.ac_reference_angle_by_pos: Dict[int, float] = {}
        self.dc_reference_nodes: List[object] = []
        self.dc_reference_voltage_by_pos: Dict[int, float] = {}
        self.ac_virtual_theta_specs: List[Tuple[int, int]] = []
        self.ac_node_by_idx = {node.idx: node for node in self.ac_nodes}
        self.dc_node_by_idx = {node.idx: node for node in self.dc_nodes}

        def add_full(label: str, full_col: int, is_voltage: bool = False) -> int:
            # Full states map directly to the hybrid Newton vector.
            pos = len(self.state_labels)
            self.state_labels.append(label)
            self.full_col_for_state.append(int(full_col))
            if is_voltage:
                self.voltage_cols_list.append(pos)
            return pos

        def add_virtual(label: str) -> int:
            # Virtual states expose compact zero-impedance currents to the estimator.
            pos = len(self.state_labels)
            self.state_labels.append(label)
            self.full_col_for_state.append(-1)
            return pos

        if self.calc.ac_calc is not None:
            ac = self.calc.ac_calc
            node_names = [node.name for node in ac.node_list]
            ac_sub = getattr(self, "_ac_sub_estimator", None)
            ac_sub_hybrid_labels = (
                {self._ac_sub_state_label_to_hybrid(label) for label in ac_sub.state_labels}
                if ac_sub is not None and self._delegate() is None
                else None
            )

            def ac_sub_allows_state(label: str) -> bool:
                return ac_sub_hybrid_labels is None or label in ac_sub_hybrid_labels

            self.ac_node_voltage_measurements = self._ac_node_voltage_measurements()
            self.ac_node_degrees = self._ac_node_incident_degrees()
            self.ac_reference_nodes = self._select_ac_reference_nodes()
            self.ac_reference_angle_by_pos = self._ac_reference_angle_offsets()
            reference_voltage_by_pos = {
                ac.node_pos[node.idx]: self.ac_node_voltage_measurements[node.idx]
                for node in self.ac_reference_nodes
                if node.idx in ac.node_pos and node.idx in self.ac_node_voltage_measurements
            }
            fixed_theta_by_pos: Dict[int, float] = {int(pos): 0.0 for pos in reference_voltage_by_pos}
            fixed_voltage_by_pos: Dict[int, float] = {
                int(pos): max(float(value), self.voltage_floor)
                for pos, value in reference_voltage_by_pos.items()
            }
            comp_by_pos = {}
            for comp in ac.comp_nodes:
                comp_list = [int(pos) for pos in comp]
                for pos in comp_list:
                    comp_by_pos[pos] = comp_list
            for ref_pos, ref_voltage in list(fixed_voltage_by_pos.items()):
                for pos in comp_by_pos.get(int(ref_pos), [int(ref_pos)]):
                    fixed_theta_by_pos[int(pos)] = 0.0
                    fixed_voltage_by_pos[int(pos)] = ref_voltage
            self.ac_reference_voltage_by_pos = fixed_voltage_by_pos
            skip_theta = set()
            skip_voltage = set()
            for comp in ac.comp_nodes:
                if not comp:
                    continue
                # Nodes tied by zero impedance share voltage and angle in the estimator.
                fixed_positions = [int(pos) for pos in comp if int(pos) in fixed_theta_by_pos]
                if fixed_positions:
                    for pos in comp:
                        pos = int(pos)
                        skip_theta.add(pos)
                        skip_voltage.add(pos)
                    continue
                ref = int(comp[0])
                ref_theta_col = int(ac.theta_idx[ref]) if ref in ac.theta_idx else -1
                ref_voltage_col = int(ac.n_theta + ac.V_idx[ref]) if ref in ac.V_idx else -1
                for node_pos in comp[1:]:
                    pos = int(node_pos)
                    if pos in ac.theta_idx:
                        skip_theta.add(pos)
                        self.tied_full_cols.append((int(ac.theta_idx[pos]), ref_theta_col))
                    if pos in ac.V_idx:
                        skip_voltage.add(pos)
                        self.tied_full_cols.append((int(ac.n_theta + ac.V_idx[pos]), ref_voltage_col))
            for pos, value in fixed_theta_by_pos.items():
                skip_theta.add(pos)
                if pos in ac.theta_idx:
                    self.fixed_full_col_values[int(ac.theta_idx[pos])] = float(value)
                if hasattr(ac, "theta_spec") and pos < len(ac.theta_spec):
                    ac.theta_spec[pos] = float(value)
            for pos, value in fixed_voltage_by_pos.items():
                skip_voltage.add(pos)
                if pos in ac.V_idx:
                    self.fixed_full_col_values[int(ac.n_theta + ac.V_idx[pos])] = float(value)
                if hasattr(ac, "V_spec") and pos < len(ac.V_spec):
                    ac.V_spec[pos] = float(value)
            for node_pos in ac.theta_unknown:
                pos = int(node_pos)
                if pos in skip_theta:
                    continue
                label = f"AC_THETA:{node_names[pos]}"
                if ac_sub_allows_state(label):
                    add_full(label, int(ac.theta_idx[pos]))
            for pos in range(ac.N):
                if pos in skip_theta or pos in ac.theta_idx:
                    continue
                label = f"AC_THETA:{node_names[pos]}"
                if ac_sub_allows_state(label):
                    state_col = add_virtual(label)
                    self.ac_virtual_theta_specs.append((pos, state_col))
            for node_pos in ac.V_unknown:
                pos = int(node_pos)
                if pos in skip_voltage:
                    continue
                label = f"AC_V:{node_names[pos]}"
                if ac_sub_allows_state(label):
                    add_full(label, int(ac.n_theta + ac.V_idx[pos]), True)

            if ac.N_phi > 0 and ac.zero_a.size:
                re_cols = []
                im_cols = []
                for row in range(ac.zero_a.size):
                    zero_type = int(ac.zero_type[row])
                    if zero_type == 0:
                        dev = ac.isl.zero_branches[int(ac.zero_idx[row])]
                    elif zero_type == 2:
                        break_devices = getattr(ac.isl, "breakers", None)
                        if break_devices is None:
                            break_devices = sorted(self.ac_break_by_name.values(), key=lambda item: item.idx)
                        dev = break_devices[int(ac.zero_idx[row])]
                    else:
                        dev = ac.isl.switches[int(ac.zero_idx[row])]
                    re_col = add_virtual(f"AC_I_RE:{dev.name}")
                    im_col = add_virtual(f"AC_I_IM:{dev.name}")
                    re_cols.append(re_col)
                    im_cols.append(im_col)
                    self.ac_zero_current_cols_by_name[dev.name] = (re_col, im_col)
                self.ac_zero_re_cols = np.asarray(re_cols, dtype=np.int32)
                self.ac_zero_im_cols = np.asarray(im_cols, dtype=np.int32)
                self.ac_zero_phi_a = np.asarray(ac.zero_phi_a, dtype=np.int32)
                self.ac_zero_phi_b = np.asarray(ac.zero_phi_b, dtype=np.int32)
                self.ac_phi_free, self.ac_phi_pinv = self._build_phi_solver(
                    ac.N_phi,
                    np.asarray(ac.ref_phi_idx, dtype=np.int32),
                    self.ac_zero_phi_a,
                    self.ac_zero_phi_b,
                )

            for gen in sorted(self.ac_generator_by_name.values(), key=lambda item: item.idx):
                p_col = add_virtual(f"AC_GEN_P:{gen.name}")
                q_col = add_virtual(f"AC_GEN_Q:{gen.name}")
                self.ac_generator_p_col_by_name[gen.name] = p_col
                self.ac_generator_q_col_by_name[gen.name] = q_col
            for load in sorted(self.ac_load_by_name.values(), key=lambda item: item.idx):
                p_col = add_virtual(f"AC_LOAD_P:{load.name}")
                q_col = add_virtual(f"AC_LOAD_Q:{load.name}")
                self.ac_load_p_col_by_name[load.name] = p_col
                self.ac_load_q_col_by_name[load.name] = q_col

            if ac_sub is not None and self._delegate() is None:
                existing = {label: idx for idx, label in enumerate(self.state_labels)}
                seed_hybrid_cols: List[int] = []
                seed_sub_cols: List[int] = []
                for sub_col, sub_label in enumerate(ac_sub.state_labels):
                    hybrid_label = self._ac_sub_state_label_to_hybrid(sub_label)
                    if hybrid_label in existing:
                        continue
                    state_col = add_virtual(hybrid_label)
                    existing[hybrid_label] = state_col
                    if hybrid_label.startswith("AC_V:"):
                        self.voltage_cols_list.append(state_col)
                    seed_hybrid_cols.append(state_col)
                    seed_sub_cols.append(int(sub_col))
                self.ac_sub_seed_hybrid_cols = np.asarray(seed_hybrid_cols, dtype=np.int32)
                self.ac_sub_seed_sub_cols = np.asarray(seed_sub_cols, dtype=np.int32)

        if self.calc.dc_calc is not None:
            dc = self.calc.dc_calc
            dc_offset = self.calc.ac_size
            self.dc_node_voltage_measurements = self._dc_node_voltage_measurements()
            self.dc_node_degrees = self._dc_node_incident_degrees()
            self.dc_reference_nodes = self._select_dc_reference_nodes()
            reference_voltage_by_pos = {
                dc.alive_node_dict[node.idx]: self.dc_node_voltage_measurements[node.idx]
                for node in self.dc_reference_nodes
                if node.idx in dc.alive_node_dict and node.idx in self.dc_node_voltage_measurements
            }
            fixed_voltage_by_pos: Dict[int, float] = {
                int(pos): max(float(value), self.voltage_floor)
                for pos, value in reference_voltage_by_pos.items()
            }
            self.dc_reference_voltage_by_pos = fixed_voltage_by_pos
            fixed_voltage_pos = set(fixed_voltage_by_pos)
            for idx, node in enumerate(dc.alive_nodes):
                if idx in fixed_voltage_pos:
                    self.fixed_full_col_values[dc_offset + idx] = fixed_voltage_by_pos[int(idx)]
                    continue
                add_full(f"DC_V:{node.name}", dc_offset + idx, True)

            if dc.N_phi > 0 and dc.zero_phi_a.size:
                cols = []
                phi_a = []
                phi_b = []
                measured_zero_current = {
                    meas.device_name
                    for meas in self.active_measurements
                    if meas.device_type in ("DCZeroBranch", "DCBreak")
                    and meas.meas_type in ("P_FROM", "P_TO", "I_FROM", "I_TO")
                }
                for row in range(dc.zero_phi_a.size):
                    tp = str(dc.zero_type[row])
                    dev_idx = int(dc.zero_dev_idx[row])
                    if tp == "Z":
                        dev = dc.model.zero_branches[dev_idx]
                    elif tp == "B":
                        dev = dc.model.breakers[dev_idx]
                    else:
                        dev = dc.model.switches[dev_idx]
                    if tp == "Z" and dev.name not in measured_zero_current:
                        continue
                    col = add_virtual(f"DC_I:{dev.name}")
                    cols.append(col)
                    phi_a.append(int(dc.zero_phi_a[row]))
                    phi_b.append(int(dc.zero_phi_b[row]))
                    self.dc_zero_current_col_by_name[dev.name] = col
                self.dc_zero_cols = np.asarray(cols, dtype=np.int32)
                self.dc_zero_phi_a = np.asarray(phi_a, dtype=np.int32)
                self.dc_zero_phi_b = np.asarray(phi_b, dtype=np.int32)
                self.dc_phi_free, self.dc_phi_pinv = self._build_phi_solver(
                    dc.N_phi,
                    np.asarray(dc.ref_phi_idx, dtype=np.int32),
                    self.dc_zero_phi_a,
                    self.dc_zero_phi_b,
                )

            for idx in range(dc.N_dcdc):
                conv = dc.model.dcdc_converters[int(dc.dcdc_idx[idx])]
                base = dc_offset + dc.N + dc.N_phi + 2 * idx
                add_full(f"DCDC_P_FROM:{conv.name}", base)
                add_full(f"DCDC_P_TO:{conv.name}", base + 1)

            for gen in sorted(self.dc_generator_by_name.values(), key=lambda item: item.idx):
                if str(gen.control_type).upper() == "V":
                    pos = add_virtual(f"DC_VGEN_P:{gen.name}")
                    self.dc_v_generator_states.append((gen, pos))
                    self.dc_v_generator_col_by_name[gen.name] = pos

        for idx, (conv, _, _, _) in enumerate(self.calc.dcac_converters):
            base = self.calc.dcac_start + 3 * idx
            add_full(f"DCAC_P_DC:{conv.name}", base)
            add_full(f"DCAC_P_AC:{conv.name}", base + 1)
            add_full(f"DCAC_Q_AC:{conv.name}", base + 2)

        for idx, (conv, _, _, _) in enumerate(self.calc.acac_converters):
            base = self.calc.acac_start + 4 * idx
            add_full(f"ACAC_P_FROM:{conv.name}", base)
            add_full(f"ACAC_Q_FROM:{conv.name}", base + 1)
            add_full(f"ACAC_P_TO:{conv.name}", base + 2)
            add_full(f"ACAC_Q_TO:{conv.name}", base + 3)

        self.full_col_for_state = np.asarray(self.full_col_for_state, dtype=np.int32)
        self.mapped_state_cols = np.flatnonzero(self.full_col_for_state >= 0).astype(np.int32)
        self.mapped_full_cols = self.full_col_for_state[self.mapped_state_cols].astype(np.int32, copy=True)
        if self.fixed_full_col_values:
            fixed_items = sorted(self.fixed_full_col_values.items())
            self.fixed_full_cols = np.asarray([item[0] for item in fixed_items], dtype=np.int32)
            self.fixed_full_values = np.asarray([item[1] for item in fixed_items], dtype=np.float64)
        else:
            self.fixed_full_cols = np.array([], dtype=np.int32)
            self.fixed_full_values = np.array([], dtype=np.float64)
        self.voltage_cols = np.asarray(self.voltage_cols_list, dtype=np.int32)
        self.n_state = len(self.state_labels)
        self.full_to_state_col = np.full(self.full_n_state, -1, dtype=np.int32)
        for state_col, full_col in enumerate(self.full_col_for_state):
            if 0 <= int(full_col) < self.full_n_state:
                self.full_to_state_col[int(full_col)] = int(state_col)
        for target_col, source_col in self.tied_full_cols:
            if 0 <= source_col < self.full_n_state and 0 <= target_col < self.full_n_state:
                self.full_to_state_col[target_col] = self.full_to_state_col[source_col]
        self._build_state_column_lookup_arrays()

    def _build_state_column_lookup_arrays(self) -> None:
        """Cache full-vector to compact-state column lookups used by Jacobian assembly."""
        self.ac_theta_state_col = np.array([], dtype=np.int32)
        self.ac_voltage_state_col = np.array([], dtype=np.int32)
        self.dc_voltage_state_col = np.array([], dtype=np.int32)
        self.ac_theta_valid_pos = np.array([], dtype=np.int32)
        self.ac_theta_valid_cols = np.array([], dtype=np.int32)
        self.ac_voltage_valid_pos = np.array([], dtype=np.int32)
        self.ac_voltage_valid_cols = np.array([], dtype=np.int32)
        self.ac_theta_cols_need_add_at = False
        self.ac_voltage_cols_need_add_at = False
        self.dcdc_p_from_state_col = np.array([], dtype=np.int32)
        self.dcdc_p_to_state_col = np.array([], dtype=np.int32)
        self.dcac_p_dc_state_col = np.array([], dtype=np.int32)
        self.dcac_p_ac_state_col = np.array([], dtype=np.int32)
        self.dcac_q_ac_state_col = np.array([], dtype=np.int32)
        self.acac_p_from_state_col = np.array([], dtype=np.int32)
        self.acac_q_from_state_col = np.array([], dtype=np.int32)
        self.acac_p_to_state_col = np.array([], dtype=np.int32)
        self.acac_q_to_state_col = np.array([], dtype=np.int32)

        ac = self.calc.ac_calc
        if ac is not None:
            self.ac_theta_state_col = np.full(ac.N, -1, dtype=np.int32)
            self.ac_voltage_state_col = np.full(ac.N, -1, dtype=np.int32)
            for pos, idx in ac.theta_idx.items():
                self.ac_theta_state_col[int(pos)] = self._state_col_from_full(int(idx))
            for pos, state_col in getattr(self, "ac_virtual_theta_specs", []):
                self.ac_theta_state_col[int(pos)] = int(state_col)
            for pos, idx in ac.V_idx.items():
                self.ac_voltage_state_col[int(pos)] = self._state_col_from_full(int(ac.n_theta + idx))
            self.ac_theta_valid_pos = np.flatnonzero(self.ac_theta_state_col >= 0).astype(np.int32)
            self.ac_theta_valid_cols = self.ac_theta_state_col[self.ac_theta_valid_pos].astype(np.int32, copy=True)
            self.ac_voltage_valid_pos = np.flatnonzero(self.ac_voltage_state_col >= 0).astype(np.int32)
            self.ac_voltage_valid_cols = self.ac_voltage_state_col[self.ac_voltage_valid_pos].astype(np.int32, copy=True)
            if self.ac_theta_valid_cols.size:
                self.ac_theta_cols_need_add_at = np.unique(self.ac_theta_valid_cols).size != self.ac_theta_valid_cols.size
            if self.ac_voltage_valid_cols.size:
                self.ac_voltage_cols_need_add_at = np.unique(self.ac_voltage_valid_cols).size != self.ac_voltage_valid_cols.size

        dc = self.calc.dc_calc
        if dc is not None:
            full_cols = self.calc.ac_size + np.arange(dc.N, dtype=np.int32)
            self.dc_voltage_state_col = self.full_to_state_col[full_cols].astype(np.int32, copy=True)
            if dc.N_dcdc:
                idx = np.arange(dc.N_dcdc, dtype=np.int32)
                base = self.calc.ac_size + dc.N + dc.N_phi + 2 * idx
                self.dcdc_p_from_state_col = self.full_to_state_col[base].astype(np.int32, copy=True)
                self.dcdc_p_to_state_col = self.full_to_state_col[base + 1].astype(np.int32, copy=True)

        if self.calc.N_dcac:
            idx = np.arange(self.calc.N_dcac, dtype=np.int32)
            base = self.calc.dcac_start + 3 * idx
            self.dcac_p_dc_state_col = self.full_to_state_col[base].astype(np.int32, copy=True)
            self.dcac_p_ac_state_col = self.full_to_state_col[base + 1].astype(np.int32, copy=True)
            self.dcac_q_ac_state_col = self.full_to_state_col[base + 2].astype(np.int32, copy=True)

        if self.calc.N_acac:
            idx = np.arange(self.calc.N_acac, dtype=np.int32)
            base = self.calc.acac_start + 4 * idx
            self.acac_p_from_state_col = self.full_to_state_col[base].astype(np.int32, copy=True)
            self.acac_q_from_state_col = self.full_to_state_col[base + 1].astype(np.int32, copy=True)
            self.acac_p_to_state_col = self.full_to_state_col[base + 2].astype(np.int32, copy=True)
            self.acac_q_to_state_col = self.full_to_state_col[base + 3].astype(np.int32, copy=True)

    def _build_derivative_caches(self) -> None:
        """Cache static lookup tables used by the analytical hybrid Jacobian."""
        self.ac_branch_stamp_by_name = {}
        self.ac_transformer_stamp_by_name = {}
        self.ac_gen_share_by_name = {}
        self.ac_y_row_nodes = []
        self.ac_y_row_y_conj = []
        self.ac_y_row_off_mask = []
        self.ac_y_row_off_nodes = []
        self.ac_y_row_diag_conj = np.array([], dtype=np.complex128)
        self.ac_zero_current_by_node = {}
        self.dcdc_pos_by_name = {}
        self.dcac_pos_by_name = {}
        self.acac_pos_by_name = {}

        if self.calc.ac_calc is not None:
            ac = self.calc.ac_calc
            self.ac_branch_stamp_by_name = self._build_ac_stamp_map(
                [br for br in ac.isl.branches if getattr(br, "is_alive", False)],
                False,
            )
            self.ac_transformer_stamp_by_name = self._build_ac_stamp_map(
                [tr for tr in ac.isl.transformers if getattr(tr, "is_alive", False)],
                True,
            )
            gens_at_pos: Dict[int, List[object]] = {}
            for gen in ac.isl.gens:
                if getattr(gen, "is_alive", False) and gen.node in ac.node_pos:
                    gens_at_pos.setdefault(ac.node_pos[gen.node], []).append(gen)
            for gens in gens_at_pos.values():
                total_alpha = sum(float(gen.alpha) for gen in gens if getattr(gen, "alpha", None) is not None)
                for gen in gens:
                    if total_alpha > 0.0 and getattr(gen, "alpha", None) is not None:
                        self.ac_gen_share_by_name[gen.name] = float(gen.alpha) / total_alpha
                    else:
                        self.ac_gen_share_by_name[gen.name] = 1.0 / len(gens)
            self._prepare_ac_y_row_cache()
            self.ac_zero_current_by_node = {}
            for row, (a_raw, b_raw) in enumerate(zip(ac.zero_a, ac.zero_b)):
                a = int(a_raw)
                b = int(b_raw)
                self.ac_zero_current_by_node.setdefault(a, []).append((row, True))
                self.ac_zero_current_by_node.setdefault(b, []).append((row, False))

        if self.calc.dc_calc is not None:
            dc = self.calc.dc_calc
            self.dcdc_pos_by_name = {
                dc.model.dcdc_converters[int(dc.dcdc_idx[idx])].name: idx
                for idx in range(dc.N_dcdc)
            }

        self.dcac_pos_by_name = {
            conv.name: idx
            for idx, (conv, _, _, _) in enumerate(self.calc.dcac_converters)
        }
        self.acac_pos_by_name = {
            conv.name: idx
            for idx, (conv, _, _, _) in enumerate(self.calc.acac_converters)
        }

    @staticmethod
    def _build_ac_stamp_map(devices: Sequence[object], with_tap: bool) -> Dict[str, Tuple[complex, complex, complex, complex]]:
        """Build AC branch/transformer stamps in one vectorized call."""
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

    def _build_static_jacobian_index(self) -> None:
        """Pre-index measurement rows whose Jacobian entries are state independent."""
        skip = np.zeros(len(self.active_measurements), dtype=bool)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        dc_branch_power_rows: List[int] = []
        dc_branch_power_i_pos: List[int] = []
        dc_branch_power_j_pos: List[int] = []
        dc_branch_power_i_col: List[int] = []
        dc_branch_power_j_col: List[int] = []
        dc_branch_power_g: List[float] = []
        dc_branch_power_from: List[bool] = []
        dc_zero_power_rows: List[int] = []
        dc_zero_power_v_pos: List[int] = []
        dc_zero_power_v_col: List[int] = []
        dc_zero_power_i_col: List[int] = []
        dc_zero_power_sign: List[float] = []
        dcdc_current_rows: List[int] = []
        dcdc_current_p_col: List[int] = []
        dcdc_current_v_col: List[int] = []
        dcdc_current_v_pos: List[int] = []
        dcac_i_dc_rows: List[int] = []
        dcac_i_dc_p_col: List[int] = []
        dcac_i_dc_v_col: List[int] = []
        dcac_i_dc_v_pos: List[int] = []
        dcac_i_ac_rows: List[int] = []
        dcac_i_ac_p_col: List[int] = []
        dcac_i_ac_q_col: List[int] = []
        dcac_i_ac_v_col: List[int] = []
        dcac_i_ac_v_pos: List[int] = []
        ac_branch_power_rows: List[int] = []
        ac_branch_power_is_p: List[bool] = []
        ac_branch_power_own: List[int] = []
        ac_branch_power_other: List[int] = []
        ac_branch_power_y_self: List[complex] = []
        ac_branch_power_y_mutual: List[complex] = []
        ac_branch_current_rows: List[int] = []
        ac_branch_current_own: List[int] = []
        ac_branch_current_other: List[int] = []
        ac_branch_current_y_self: List[complex] = []
        ac_branch_current_y_mutual: List[complex] = []
        ac_load_rows: List[int] = []
        ac_load_kind: List[int] = []
        ac_load_v_pos: List[int] = []
        ac_load_v_col: List[int] = []
        ac_load_pv0: List[float] = []
        ac_load_pv1: List[float] = []
        ac_load_pv2: List[float] = []
        ac_load_qv0: List[float] = []
        ac_load_qv1: List[float] = []
        ac_load_qv2: List[float] = []
        ac_zero_current_rows: List[int] = []
        ac_zero_current_re_col: List[int] = []
        ac_zero_current_im_col: List[int] = []
        ac_zero_power_rows: List[int] = []
        ac_zero_power_is_p: List[bool] = []
        ac_zero_power_sign: List[float] = []
        ac_zero_power_pos: List[int] = []
        ac_zero_power_theta_col: List[int] = []
        ac_zero_power_v_col: List[int] = []
        ac_zero_power_re_col: List[int] = []
        ac_zero_power_im_col: List[int] = []
        dc_load_rows: List[int] = []
        dc_load_kind: List[int] = []
        dc_load_v_pos: List[int] = []
        dc_load_v_col: List[int] = []
        dc_load_pv0: List[float] = []
        dc_load_pv1: List[float] = []
        dc_load_pv2: List[float] = []
        dc_gen_i_v_rows: List[int] = []
        dc_gen_i_v_p_col: List[int] = []
        dc_gen_i_v_v_col: List[int] = []
        dc_gen_i_v_v_pos: List[int] = []
        dc_gen_i_p_rows: List[int] = []
        dc_gen_i_p_p: List[float] = []
        dc_gen_i_p_v_col: List[int] = []
        dc_gen_i_p_v_pos: List[int] = []
        dc_gen_p_i_rows: List[int] = []
        dc_gen_p_i_i: List[float] = []
        dc_gen_p_i_v_col: List[int] = []
        ac_gen_dynamic = {}
        ac = self.calc.ac_calc
        dc = self.calc.dc_calc

        def add(row: int, col: int, value: float = 1.0) -> None:
            if col >= 0 and value != 0.0:
                rows.append(int(row))
                cols.append(int(col))
                data.append(float(value))

        for row, meas in enumerate(self.active_measurements):
            mtype = meas.meas_type
            dtype = meas.device_type

            if dtype == "ACNode" and ac is not None:
                node = self.ac_node_by_name.get(meas.device_name)
                if node is None or node.idx not in ac.node_pos:
                    continue
                pos = ac.node_pos[node.idx]
                if mtype == "V":
                    add(row, int(self.ac_voltage_state_col[pos]))
                    skip[row] = True
                elif mtype in ("ANGLE", "THETA"):
                    add(row, int(self.ac_theta_state_col[pos]))
                    skip[row] = True

            elif dtype == "DCNode" and dc is not None and mtype == "V":
                node = self.dc_node_by_name.get(meas.device_name)
                if node is not None and node.idx in dc.alive_node_dict:
                    add(row, int(self.dc_voltage_state_col[dc.alive_node_dict[node.idx]]))
                    skip[row] = True

            elif dtype in ("DCZeroBranchConstraint", "DCSwitchConstraint", "DCBreakConstraint") and dc is not None:
                if mtype != "V_DIFF":
                    continue
                device = (
                    self.dc_zero_branch_by_name.get(meas.device_name)
                    if dtype == "DCZeroBranchConstraint"
                    else self.dc_break_by_name.get(meas.device_name)
                    if dtype == "DCBreakConstraint"
                    else self.dc_switch_by_name.get(meas.device_name)
                )
                if device is None or device.i_node not in dc.alive_node_dict or device.j_node not in dc.alive_node_dict:
                    continue
                i = dc.alive_node_dict[device.i_node]
                j = dc.alive_node_dict[device.j_node]
                add(row, int(self.dc_voltage_state_col[i]), 1.0)
                add(row, int(self.dc_voltage_state_col[j]), -1.0)
                skip[row] = True

            elif dtype in ("ACBranch", "ACTransformer") and ac is not None:
                if dtype == "ACTransformer":
                    device = self.ac_transformer_by_name.get(meas.device_name)
                    stamp = self.ac_transformer_stamp_by_name.get(meas.device_name)
                else:
                    device = self.ac_branch_by_name.get(meas.device_name)
                    stamp = self.ac_branch_stamp_by_name.get(meas.device_name)
                if device is None or stamp is None or device.i_node not in ac.node_pos or device.j_node not in ac.node_pos:
                    continue
                i = ac.node_pos[device.i_node]
                j = ac.node_pos[device.j_node]
                yff, yft, ytf, ytt = stamp
                if mtype == "V_FROM":
                    add(row, int(self.ac_voltage_state_col[i]))
                    skip[row] = True
                elif mtype == "V_TO":
                    add(row, int(self.ac_voltage_state_col[j]))
                    skip[row] = True
                elif mtype in ("P_FROM", "Q_FROM"):
                    ac_branch_power_rows.append(row)
                    ac_branch_power_is_p.append(mtype == "P_FROM")
                    ac_branch_power_own.append(i)
                    ac_branch_power_other.append(j)
                    ac_branch_power_y_self.append(yff)
                    ac_branch_power_y_mutual.append(yft)
                    skip[row] = True
                elif mtype in ("P_TO", "Q_TO"):
                    ac_branch_power_rows.append(row)
                    ac_branch_power_is_p.append(mtype == "P_TO")
                    ac_branch_power_own.append(j)
                    ac_branch_power_other.append(i)
                    ac_branch_power_y_self.append(ytt)
                    ac_branch_power_y_mutual.append(ytf)
                    skip[row] = True
                elif mtype == "I_FROM":
                    ac_branch_current_rows.append(row)
                    ac_branch_current_own.append(i)
                    ac_branch_current_other.append(j)
                    ac_branch_current_y_self.append(yff)
                    ac_branch_current_y_mutual.append(yft)
                    skip[row] = True
                elif mtype == "I_TO":
                    ac_branch_current_rows.append(row)
                    ac_branch_current_own.append(j)
                    ac_branch_current_other.append(i)
                    ac_branch_current_y_self.append(ytt)
                    ac_branch_current_y_mutual.append(ytf)
                    skip[row] = True

            elif dtype in ("ACSwitch", "ACZeroBranch", "ACBreak") and ac is not None:
                device = (
                    self.ac_switch_by_name.get(meas.device_name)
                    if dtype == "ACSwitch"
                    else self.ac_break_by_name.get(meas.device_name)
                    if dtype == "ACBreak"
                    else self.ac_zero_branch_by_name.get(meas.device_name)
                )
                if device is None:
                    continue
                if mtype in ("V_FROM", "V_TO"):
                    node_idx = device.i_node if mtype == "V_FROM" else device.j_node
                    if node_idx in ac.node_pos:
                        add(row, int(self.ac_voltage_state_col[ac.node_pos[node_idx]]))
                        skip[row] = True
                elif (
                    mtype in ("I_FROM", "I_TO", "P_FROM", "Q_FROM", "P_TO", "Q_TO")
                    and device.i_node in ac.node_pos
                    and device.name in self.ac_zero_current_cols_by_name
                ):
                    i = ac.node_pos[device.i_node]
                    re_col, im_col = self.ac_zero_current_cols_by_name[device.name]
                    if mtype in ("I_FROM", "I_TO"):
                        ac_zero_current_rows.append(row)
                        ac_zero_current_re_col.append(int(re_col))
                        ac_zero_current_im_col.append(int(im_col))
                    else:
                        ac_zero_power_rows.append(row)
                        ac_zero_power_is_p.append(mtype.startswith("P"))
                        ac_zero_power_sign.append(-1.0 if mtype.endswith("_TO") else 1.0)
                        ac_zero_power_pos.append(i)
                        ac_zero_power_theta_col.append(int(self.ac_theta_state_col[i]))
                        ac_zero_power_v_col.append(int(self.ac_voltage_state_col[i]))
                        ac_zero_power_re_col.append(int(re_col))
                        ac_zero_power_im_col.append(int(im_col))
                    skip[row] = True

            elif dtype == "ACLoad" and ac is not None:
                load = self.ac_load_by_name.get(meas.device_name)
                if load is None or load.node not in ac.node_pos:
                    continue
                pos = ac.node_pos[load.node]
                v_col = int(self.ac_voltage_state_col[pos])
                if mtype == "V_LOAD":
                    add(row, v_col)
                    skip[row] = True
                elif mtype in ("P_LOAD", "Q_LOAD", "I_LOAD"):
                    pass

            elif dtype == "ACGenerator" and ac is not None and mtype == "V_GEN":
                gen = self.ac_generator_by_name.get(meas.device_name)
                if gen is not None and gen.node in ac.node_pos:
                    add(row, int(self.ac_voltage_state_col[ac.node_pos[gen.node]]))
                    skip[row] = True

            elif dtype == "ACGenerator" and ac is not None and mtype in ("P_GEN", "Q_GEN", "I_GEN"):
                gen = self.ac_generator_by_name.get(meas.device_name)
                if gen is not None and gen.node in ac.node_pos:
                    pass

            elif dtype == "DCBranch" and dc is not None:
                br = self.dc_branch_by_name.get(meas.device_name)
                if br is None or br.i_node not in dc.alive_node_dict or br.j_node not in dc.alive_node_dict:
                    continue
                i = dc.alive_node_dict[br.i_node]
                j = dc.alive_node_dict[br.j_node]
                i_col = int(self.dc_voltage_state_col[i])
                j_col = int(self.dc_voltage_state_col[j])
                if mtype == "V_FROM":
                    add(row, i_col)
                    skip[row] = True
                elif mtype == "V_TO":
                    add(row, j_col)
                    skip[row] = True
                elif mtype == "I_FROM":
                    inv_r = 1.0 / br.r
                    add(row, i_col, inv_r)
                    add(row, j_col, -inv_r)
                    skip[row] = True
                elif mtype == "I_TO":
                    inv_r = 1.0 / br.r
                    add(row, i_col, -inv_r)
                    add(row, j_col, inv_r)
                    skip[row] = True
                elif mtype in ("P_FROM", "P_TO"):
                    dc_branch_power_rows.append(row)
                    dc_branch_power_i_pos.append(i)
                    dc_branch_power_j_pos.append(j)
                    dc_branch_power_i_col.append(i_col)
                    dc_branch_power_j_col.append(j_col)
                    dc_branch_power_g.append(1.0 / br.r)
                    dc_branch_power_from.append(mtype == "P_FROM")
                    skip[row] = True

            elif dtype in ("DCSwitch", "DCZeroBranch", "DCBreak") and dc is not None:
                device = (
                    self.dc_switch_by_name.get(meas.device_name)
                    if dtype == "DCSwitch"
                    else self.dc_break_by_name.get(meas.device_name)
                    if dtype == "DCBreak"
                    else self.dc_zero_branch_by_name.get(meas.device_name)
                )
                if device is None or device.i_node not in dc.alive_node_dict or device.j_node not in dc.alive_node_dict:
                    continue
                i = dc.alive_node_dict[device.i_node]
                j = dc.alive_node_dict[device.j_node]
                i_col = int(self.dc_voltage_state_col[i])
                j_col = int(self.dc_voltage_state_col[j])
                if mtype == "V_FROM":
                    add(row, i_col)
                    skip[row] = True
                elif mtype == "V_TO":
                    add(row, j_col)
                    skip[row] = True
                elif device.name not in self.dc_zero_current_col_by_name:
                    if mtype in ("I_FROM", "I_TO", "P_FROM", "P_TO"):
                        skip[row] = True
                    else:
                        continue
                elif mtype == "I_FROM":
                    current_col = int(self.dc_zero_current_col_by_name[device.name])
                    add(row, current_col)
                    skip[row] = True
                elif mtype == "I_TO":
                    current_col = int(self.dc_zero_current_col_by_name[device.name])
                    add(row, current_col, -1.0)
                    skip[row] = True
                elif mtype in ("P_FROM", "P_TO"):
                    current_col = int(self.dc_zero_current_col_by_name[device.name])
                    from_side = mtype == "P_FROM"
                    dc_zero_power_rows.append(row)
                    dc_zero_power_v_pos.append(i if from_side else j)
                    dc_zero_power_v_col.append(i_col if from_side else j_col)
                    dc_zero_power_i_col.append(current_col)
                    dc_zero_power_sign.append(1.0 if from_side else -1.0)
                    skip[row] = True

            elif dtype == "DCLoad" and dc is not None:
                load = self.dc_load_by_name.get(meas.device_name)
                if load is None or load.node not in dc.alive_node_dict:
                    continue
                pos = dc.alive_node_dict[load.node]
                v_col = int(self.dc_voltage_state_col[pos])
                if mtype == "V_LOAD":
                    add(row, v_col)
                    skip[row] = True
                elif mtype in ("P_LOAD", "I_LOAD"):
                    p0, p1, p2, _, _, _ = self._load_zip_coefficients(load)
                    dc_load_rows.append(row)
                    dc_load_kind.append(0 if mtype == "P_LOAD" else 1)
                    dc_load_v_pos.append(pos)
                    dc_load_v_col.append(v_col)
                    dc_load_pv0.append(p0)
                    dc_load_pv1.append(p1)
                    dc_load_pv2.append(p2)
                    skip[row] = True

            elif dtype == "DCGenerator" and dc is not None:
                gen = self.dc_generator_by_name.get(meas.device_name)
                if gen is None or gen.node not in dc.alive_node_dict:
                    continue
                pos = dc.alive_node_dict[gen.node]
                control_type = str(gen.control_type).upper()
                if mtype == "V_GEN":
                    add(row, int(self.dc_voltage_state_col[pos]))
                    skip[row] = True
                elif mtype == "P_GEN" and control_type == "V":
                    add(row, int(self.dc_v_generator_col_by_name[gen.name]))
                    skip[row] = True
                elif mtype == "I_GEN" and control_type == "V":
                    dc_gen_i_v_rows.append(row)
                    dc_gen_i_v_p_col.append(int(self.dc_v_generator_col_by_name[gen.name]))
                    dc_gen_i_v_v_col.append(int(self.dc_voltage_state_col[pos]))
                    dc_gen_i_v_v_pos.append(pos)
                    skip[row] = True
                elif mtype == "I_GEN" and control_type == "P":
                    dc_gen_i_p_rows.append(row)
                    dc_gen_i_p_p.append(float(gen.p_set))
                    dc_gen_i_p_v_col.append(int(self.dc_voltage_state_col[pos]))
                    dc_gen_i_p_v_pos.append(pos)
                    skip[row] = True
                elif mtype == "P_GEN" and control_type == "I":
                    dc_gen_p_i_rows.append(row)
                    dc_gen_p_i_i.append(float(gen.i_set))
                    dc_gen_p_i_v_col.append(int(self.dc_voltage_state_col[pos]))
                    skip[row] = True
                elif (mtype == "P_GEN" and control_type == "P") or (mtype == "I_GEN" and control_type == "I"):
                    skip[row] = True

            elif dtype == "DCDCConverter" and dc is not None:
                conv = self.dcdc_by_name.get(meas.device_name)
                if conv is None or conv.name not in self.dcdc_pos_by_name:
                    continue
                d_idx = self.dcdc_pos_by_name[conv.name]
                if mtype == "P_FROM":
                    add(row, int(self.dcdc_p_from_state_col[d_idx]))
                    skip[row] = True
                elif mtype == "P_TO":
                    add(row, int(self.dcdc_p_to_state_col[d_idx]))
                    skip[row] = True
                elif mtype == "V_FROM" and conv.i_node in dc.alive_node_dict:
                    add(row, int(self.dc_voltage_state_col[dc.alive_node_dict[conv.i_node]]))
                    skip[row] = True
                elif mtype == "V_TO" and conv.j_node in dc.alive_node_dict:
                    add(row, int(self.dc_voltage_state_col[dc.alive_node_dict[conv.j_node]]))
                    skip[row] = True
                elif mtype in ("I_FROM", "I_TO"):
                    d_idx = self.dcdc_pos_by_name[conv.name]
                    if mtype == "I_FROM" and conv.i_node in dc.alive_node_dict:
                        dcdc_current_rows.append(row)
                        dcdc_current_p_col.append(int(self.dcdc_p_from_state_col[d_idx]))
                        dcdc_current_v_pos.append(dc.alive_node_dict[conv.i_node])
                        dcdc_current_v_col.append(int(self.dc_voltage_state_col[dc.alive_node_dict[conv.i_node]]))
                        skip[row] = True
                    elif mtype == "I_TO" and conv.j_node in dc.alive_node_dict:
                        dcdc_current_rows.append(row)
                        dcdc_current_p_col.append(int(self.dcdc_p_to_state_col[d_idx]))
                        dcdc_current_v_pos.append(dc.alive_node_dict[conv.j_node])
                        dcdc_current_v_col.append(int(self.dc_voltage_state_col[dc.alive_node_dict[conv.j_node]]))
                        skip[row] = True

            elif dtype == "DCACConverter":
                conv = self.dcac_by_name.get(meas.device_name)
                if conv is None or conv.name not in self.dcac_pos_by_name:
                    continue
                k = self.dcac_pos_by_name[conv.name]
                _, ac_pos, dc_pos, _ = self.calc.dcac_converters[k]
                if mtype == "P_DC":
                    add(row, int(self.dcac_p_dc_state_col[k]))
                    skip[row] = True
                elif mtype == "P_AC":
                    add(row, int(self.dcac_p_ac_state_col[k]))
                    skip[row] = True
                elif mtype == "Q_AC":
                    add(row, int(self.dcac_q_ac_state_col[k]))
                    skip[row] = True
                elif mtype == "V_DC":
                    add(row, int(self.dc_voltage_state_col[dc_pos]))
                    skip[row] = True
                elif mtype == "V_AC":
                    add(row, int(self.ac_voltage_state_col[ac_pos]))
                    skip[row] = True
                elif mtype == "I_DC":
                    dcac_i_dc_rows.append(row)
                    dcac_i_dc_p_col.append(int(self.dcac_p_dc_state_col[k]))
                    dcac_i_dc_v_col.append(int(self.dc_voltage_state_col[dc_pos]))
                    dcac_i_dc_v_pos.append(dc_pos)
                    skip[row] = True
                elif mtype == "I_AC":
                    dcac_i_ac_rows.append(row)
                    dcac_i_ac_p_col.append(int(self.dcac_p_ac_state_col[k]))
                    dcac_i_ac_q_col.append(int(self.dcac_q_ac_state_col[k]))
                    dcac_i_ac_v_col.append(int(self.ac_voltage_state_col[ac_pos]))
                    dcac_i_ac_v_pos.append(ac_pos)
                    skip[row] = True

            elif dtype == "ACACConverter":
                conv = self.acac_by_name.get(meas.device_name)
                if conv is None or conv.name not in self.acac_pos_by_name:
                    continue
                k = self.acac_pos_by_name[conv.name]
                _, i_pos, j_pos, _ = self.calc.acac_converters[k]
                if mtype == "P_FROM":
                    add(row, int(self.acac_p_from_state_col[k]))
                    skip[row] = True
                elif mtype == "Q_FROM":
                    add(row, int(self.acac_q_from_state_col[k]))
                    skip[row] = True
                elif mtype == "P_TO":
                    add(row, int(self.acac_p_to_state_col[k]))
                    skip[row] = True
                elif mtype == "Q_TO":
                    add(row, int(self.acac_q_to_state_col[k]))
                    skip[row] = True
                elif mtype == "V_FROM":
                    add(row, int(self.ac_voltage_state_col[i_pos]))
                    skip[row] = True
                elif mtype == "V_TO":
                    add(row, int(self.ac_voltage_state_col[j_pos]))
                    skip[row] = True

        self._jacobian_static_skip = skip
        self._jacobian_static_rows = np.asarray(rows, dtype=np.int32)
        self._jacobian_static_cols = np.asarray(cols, dtype=np.int32)
        self._jacobian_static_data = np.asarray(data, dtype=np.float64)
        self._jac_dc_branch_power_rows = np.asarray(dc_branch_power_rows, dtype=np.int32)
        self._jac_dc_branch_power_i_pos = np.asarray(dc_branch_power_i_pos, dtype=np.int32)
        self._jac_dc_branch_power_j_pos = np.asarray(dc_branch_power_j_pos, dtype=np.int32)
        self._jac_dc_branch_power_i_col = np.asarray(dc_branch_power_i_col, dtype=np.int32)
        self._jac_dc_branch_power_j_col = np.asarray(dc_branch_power_j_col, dtype=np.int32)
        self._jac_dc_branch_power_g = np.asarray(dc_branch_power_g, dtype=np.float64)
        self._jac_dc_branch_power_from = np.asarray(dc_branch_power_from, dtype=bool)
        self._jac_dc_zero_power_rows = np.asarray(dc_zero_power_rows, dtype=np.int32)
        self._jac_dc_zero_power_v_pos = np.asarray(dc_zero_power_v_pos, dtype=np.int32)
        self._jac_dc_zero_power_v_col = np.asarray(dc_zero_power_v_col, dtype=np.int32)
        self._jac_dc_zero_power_i_col = np.asarray(dc_zero_power_i_col, dtype=np.int32)
        self._jac_dc_zero_power_sign = np.asarray(dc_zero_power_sign, dtype=np.float64)
        self._jac_dcdc_current_rows = np.asarray(dcdc_current_rows, dtype=np.int32)
        self._jac_dcdc_current_p_col = np.asarray(dcdc_current_p_col, dtype=np.int32)
        self._jac_dcdc_current_v_col = np.asarray(dcdc_current_v_col, dtype=np.int32)
        self._jac_dcdc_current_v_pos = np.asarray(dcdc_current_v_pos, dtype=np.int32)
        self._jac_dcac_i_dc_rows = np.asarray(dcac_i_dc_rows, dtype=np.int32)
        self._jac_dcac_i_dc_p_col = np.asarray(dcac_i_dc_p_col, dtype=np.int32)
        self._jac_dcac_i_dc_v_col = np.asarray(dcac_i_dc_v_col, dtype=np.int32)
        self._jac_dcac_i_dc_v_pos = np.asarray(dcac_i_dc_v_pos, dtype=np.int32)
        self._jac_dcac_i_ac_rows = np.asarray(dcac_i_ac_rows, dtype=np.int32)
        self._jac_dcac_i_ac_p_col = np.asarray(dcac_i_ac_p_col, dtype=np.int32)
        self._jac_dcac_i_ac_q_col = np.asarray(dcac_i_ac_q_col, dtype=np.int32)
        self._jac_dcac_i_ac_v_col = np.asarray(dcac_i_ac_v_col, dtype=np.int32)
        self._jac_dcac_i_ac_v_pos = np.asarray(dcac_i_ac_v_pos, dtype=np.int32)
        self._jac_ac_branch_power_rows = np.asarray(ac_branch_power_rows, dtype=np.int32)
        self._jac_ac_branch_power_is_p = np.asarray(ac_branch_power_is_p, dtype=bool)
        self._jac_ac_branch_power_own = np.asarray(ac_branch_power_own, dtype=np.int32)
        self._jac_ac_branch_power_other = np.asarray(ac_branch_power_other, dtype=np.int32)
        self._jac_ac_branch_power_y_self = np.asarray(ac_branch_power_y_self, dtype=np.complex128)
        self._jac_ac_branch_power_y_mutual = np.asarray(ac_branch_power_y_mutual, dtype=np.complex128)
        self._jac_ac_branch_current_rows = np.asarray(ac_branch_current_rows, dtype=np.int32)
        self._jac_ac_branch_current_own = np.asarray(ac_branch_current_own, dtype=np.int32)
        self._jac_ac_branch_current_other = np.asarray(ac_branch_current_other, dtype=np.int32)
        self._jac_ac_branch_current_y_self = np.asarray(ac_branch_current_y_self, dtype=np.complex128)
        self._jac_ac_branch_current_y_mutual = np.asarray(ac_branch_current_y_mutual, dtype=np.complex128)
        self._jac_ac_load_rows = np.asarray(ac_load_rows, dtype=np.int32)
        self._jac_ac_load_kind = np.asarray(ac_load_kind, dtype=np.int8)
        self._jac_ac_load_v_pos = np.asarray(ac_load_v_pos, dtype=np.int32)
        self._jac_ac_load_v_col = np.asarray(ac_load_v_col, dtype=np.int32)
        self._jac_ac_load_pv0 = np.asarray(ac_load_pv0, dtype=np.float64)
        self._jac_ac_load_pv1 = np.asarray(ac_load_pv1, dtype=np.float64)
        self._jac_ac_load_pv2 = np.asarray(ac_load_pv2, dtype=np.float64)
        self._jac_ac_load_qv0 = np.asarray(ac_load_qv0, dtype=np.float64)
        self._jac_ac_load_qv1 = np.asarray(ac_load_qv1, dtype=np.float64)
        self._jac_ac_load_qv2 = np.asarray(ac_load_qv2, dtype=np.float64)
        self._jac_ac_zero_current_rows = np.asarray(ac_zero_current_rows, dtype=np.int32)
        self._jac_ac_zero_current_re_col = np.asarray(ac_zero_current_re_col, dtype=np.int32)
        self._jac_ac_zero_current_im_col = np.asarray(ac_zero_current_im_col, dtype=np.int32)
        self._jac_ac_zero_power_rows = np.asarray(ac_zero_power_rows, dtype=np.int32)
        self._jac_ac_zero_power_is_p = np.asarray(ac_zero_power_is_p, dtype=bool)
        self._jac_ac_zero_power_sign = np.asarray(ac_zero_power_sign, dtype=np.float64)
        self._jac_ac_zero_power_pos = np.asarray(ac_zero_power_pos, dtype=np.int32)
        self._jac_ac_zero_power_theta_col = np.asarray(ac_zero_power_theta_col, dtype=np.int32)
        self._jac_ac_zero_power_v_col = np.asarray(ac_zero_power_v_col, dtype=np.int32)
        self._jac_ac_zero_power_re_col = np.asarray(ac_zero_power_re_col, dtype=np.int32)
        self._jac_ac_zero_power_im_col = np.asarray(ac_zero_power_im_col, dtype=np.int32)
        self._jac_dc_load_rows = np.asarray(dc_load_rows, dtype=np.int32)
        self._jac_dc_load_kind = np.asarray(dc_load_kind, dtype=np.int8)
        self._jac_dc_load_v_pos = np.asarray(dc_load_v_pos, dtype=np.int32)
        self._jac_dc_load_v_col = np.asarray(dc_load_v_col, dtype=np.int32)
        self._jac_dc_load_pv0 = np.asarray(dc_load_pv0, dtype=np.float64)
        self._jac_dc_load_pv1 = np.asarray(dc_load_pv1, dtype=np.float64)
        self._jac_dc_load_pv2 = np.asarray(dc_load_pv2, dtype=np.float64)
        self._jac_dc_gen_i_v_rows = np.asarray(dc_gen_i_v_rows, dtype=np.int32)
        self._jac_dc_gen_i_v_p_col = np.asarray(dc_gen_i_v_p_col, dtype=np.int32)
        self._jac_dc_gen_i_v_v_col = np.asarray(dc_gen_i_v_v_col, dtype=np.int32)
        self._jac_dc_gen_i_v_v_pos = np.asarray(dc_gen_i_v_v_pos, dtype=np.int32)
        self._jac_dc_gen_i_p_rows = np.asarray(dc_gen_i_p_rows, dtype=np.int32)
        self._jac_dc_gen_i_p_p = np.asarray(dc_gen_i_p_p, dtype=np.float64)
        self._jac_dc_gen_i_p_v_col = np.asarray(dc_gen_i_p_v_col, dtype=np.int32)
        self._jac_dc_gen_i_p_v_pos = np.asarray(dc_gen_i_p_v_pos, dtype=np.int32)
        self._jac_dc_gen_p_i_rows = np.asarray(dc_gen_p_i_rows, dtype=np.int32)
        self._jac_dc_gen_p_i_i = np.asarray(dc_gen_p_i_i, dtype=np.float64)
        self._jac_dc_gen_p_i_v_col = np.asarray(dc_gen_p_i_v_col, dtype=np.int32)
        self._jacobian_dynamic_rows = np.flatnonzero(~skip).astype(np.int32, copy=False)

        ac_gen_node_pos: List[int] = []
        ac_gen_node_group_by_pos: Dict[int, int] = {}
        ac_gen_items = []
        if ac is not None:
            for gen, p_rows, q_rows, i_rows in ac_gen_dynamic.values():
                pos = int(ac.node_pos[gen.node])
                group = ac_gen_node_group_by_pos.get(pos)
                if group is None:
                    group = len(ac_gen_node_pos)
                    ac_gen_node_group_by_pos[pos] = group
                    ac_gen_node_pos.append(pos)
                ac_gen_items.append(
                    (
                        group,
                        float(self.ac_gen_share_by_name.get(gen.name, 1.0)),
                        np.asarray(p_rows, dtype=np.int32),
                        np.asarray(q_rows, dtype=np.int32),
                        np.asarray(i_rows, dtype=np.int32),
                    )
                )
        self._jac_ac_generator_items = ac_gen_items
        n_ac_gen_groups = len(ac_gen_node_pos)
        single_group_seen = np.zeros(n_ac_gen_groups, dtype=bool)
        single_share = np.ones(n_ac_gen_groups, dtype=np.float64)
        single_p_row = np.full(n_ac_gen_groups, -1, dtype=np.int32)
        single_q_row = np.full(n_ac_gen_groups, -1, dtype=np.int32)
        single_i_row = np.full(n_ac_gen_groups, -1, dtype=np.int32)
        single_rows = True
        for group, share, p_rows, q_rows, i_rows in ac_gen_items:
            if single_group_seen[group] or p_rows.size > 1 or q_rows.size > 1 or i_rows.size > 1:
                single_rows = False
                break
            single_group_seen[group] = True
            single_share[group] = float(share)
            if p_rows.size:
                single_p_row[group] = int(p_rows[0])
            if q_rows.size:
                single_q_row[group] = int(q_rows[0])
            if i_rows.size:
                single_i_row[group] = int(i_rows[0])
        self._jac_ac_gen_single_rows = bool(single_rows)
        self._jac_ac_gen_single_share = single_share
        self._jac_ac_gen_single_p_row = single_p_row
        self._jac_ac_gen_single_q_row = single_q_row
        self._jac_ac_gen_single_i_row = single_i_row
        self._jac_ac_gen_node_pos = np.asarray(ac_gen_node_pos, dtype=np.int32)
        ac_gen_y_group: List[np.ndarray] = []
        ac_gen_y_nodes: List[np.ndarray] = []
        ac_gen_y_conj: List[np.ndarray] = []
        for group, pos in enumerate(self._jac_ac_gen_node_pos):
            nodes = self.ac_y_row_nodes[int(pos)]
            if nodes.size == 0:
                continue
            ac_gen_y_group.append(np.full(nodes.size, group, dtype=np.int32))
            ac_gen_y_nodes.append(nodes.astype(np.int32, copy=False))
            ac_gen_y_conj.append(self.ac_y_row_y_conj[int(pos)].astype(np.complex128, copy=False))
        if ac_gen_y_group:
            self._jac_ac_gen_y_group = np.concatenate(ac_gen_y_group)
            self._jac_ac_gen_y_nodes = np.concatenate(ac_gen_y_nodes)
            self._jac_ac_gen_y_conj = np.concatenate(ac_gen_y_conj)
        else:
            self._jac_ac_gen_y_group = np.array([], dtype=np.int32)
            self._jac_ac_gen_y_nodes = np.array([], dtype=np.int32)
            self._jac_ac_gen_y_conj = np.array([], dtype=np.complex128)

    def _measurement_device(self, meas: Measurement):
        dtype = meas.device_type
        name = meas.device_name
        if dtype == "ACNode":
            return self.ac_node_by_name[name]
        if dtype == "DCNode":
            return self.dc_node_by_name[name]
        if dtype == "ACBranch":
            return self.ac_branch_by_name[name]
        if dtype == "ACTransformer":
            return self.ac_transformer_by_name[name]
        if dtype == "ACSwitch":
            return self.ac_switch_by_name[name]
        if dtype == "ACBreak":
            return self.ac_break_by_name[name]
        if dtype == "ACZeroBranch":
            return self.ac_zero_branch_by_name[name]
        if dtype == "ACGenerator":
            return self.ac_generator_by_name[name]
        if dtype == "ACLoad":
            return self.ac_load_by_name[name]
        if dtype == "DCBranch":
            return self.dc_branch_by_name[name]
        if dtype == "DCSwitch":
            return self.dc_switch_by_name[name]
        if dtype == "DCBreak":
            return self.dc_break_by_name[name]
        if dtype == "DCZeroBranch":
            return self.dc_zero_branch_by_name[name]
        if dtype == "DCSwitchConstraint":
            return self.dc_switch_by_name[name]
        if dtype == "DCBreakConstraint":
            return self.dc_break_by_name[name]
        if dtype == "DCZeroBranchConstraint":
            return self.dc_zero_branch_by_name[name]
        if dtype == "DCGenerator":
            return self.dc_generator_by_name[name]
        if dtype == "DCLoad":
            return self.dc_load_by_name[name]
        if dtype == "DCDCConverter":
            return self.dcdc_by_name[name]
        if dtype == "DCACConverter":
            return self.dcac_by_name[name]
        if dtype == "ACACConverter":
            return self.acac_by_name[name]
        raise RuntimeError(f"Unsupported measurement device type: {dtype}")

    def _build_evaluation_index(self) -> None:
        """Group active measurements once so h(x) evaluation avoids per-row string dispatch."""
        groups = {}
        ac_delegated = getattr(self, "_active_ac_delegated_row_mask", np.zeros(len(self.active_measurements), dtype=bool))
        dc_delegated = getattr(self, "_active_dc_delegated_row_mask", np.zeros(len(self.active_measurements), dtype=bool))
        for row, meas in enumerate(self.active_measurements):
            if (row < ac_delegated.size and ac_delegated[row]) or (row < dc_delegated.size and dc_delegated[row]):
                continue
            key = (meas.device_type, meas.meas_type)
            bucket = groups.setdefault(key, [[], []])
            bucket[0].append(row)
            bucket[1].append(self._measurement_device(meas))
        self._eval_groups = [
            (key[0], key[1], np.asarray(rows, dtype=np.int32), devices)
            for key, (rows, devices) in groups.items()
        ]
        self._build_fast_evaluation_index()

    def _build_fast_evaluation_index(self) -> None:
        """Precompile active measurement evaluation into array lookup groups."""
        fast_groups = []
        supported = True
        ac = self.calc.ac_calc
        dc = self.calc.dc_calc

        for dtype, mtype, rows, devices in self._eval_groups:
            try:
                if dtype == "ACNode" and ac is not None:
                    pos = np.asarray([ac.node_pos[dev.idx] for dev in devices], dtype=np.int32)
                    fast_groups.append(("ACNode", mtype, rows, pos))

                elif dtype == "DCNode" and dc is not None and mtype == "V":
                    pos = np.asarray([dc.alive_node_dict[dev.idx] for dev in devices], dtype=np.int32)
                    fast_groups.append(("DCNode", mtype, rows, pos))

                elif dtype in ("DCZeroBranchConstraint", "DCSwitchConstraint", "DCBreakConstraint") and dc is not None:
                    if mtype != "V_DIFF":
                        supported = False
                        continue
                    i_pos = np.asarray([dc.alive_node_dict[dev.i_node] for dev in devices], dtype=np.int32)
                    j_pos = np.asarray([dc.alive_node_dict[dev.j_node] for dev in devices], dtype=np.int32)
                    fast_groups.append(("DCVoltageDiff", mtype, rows, i_pos, j_pos))

                elif dtype in ("ACBranch", "ACTransformer") and ac is not None:
                    own = []
                    other = []
                    y_self = []
                    y_mutual = []
                    for dev in devices:
                        i = ac.node_pos[dev.i_node]
                        j = ac.node_pos[dev.j_node]
                        stamp = (
                            self.ac_transformer_stamp_by_name[dev.name]
                            if dtype == "ACTransformer"
                            else self.ac_branch_stamp_by_name[dev.name]
                        )
                        yff, yft, ytf, ytt = stamp
                        if mtype.endswith("_TO"):
                            own.append(j)
                            other.append(i)
                            y_self.append(ytt)
                            y_mutual.append(ytf)
                        else:
                            own.append(i)
                            other.append(j)
                            y_self.append(yff)
                            y_mutual.append(yft)
                    fast_groups.append(
                        (
                            "ACLine",
                            mtype,
                            rows,
                            np.asarray(own, dtype=np.int32),
                            np.asarray(other, dtype=np.int32),
                            np.asarray(y_self, dtype=np.complex128),
                            np.asarray(y_mutual, dtype=np.complex128),
                        )
                    )

                elif dtype in ("ACSwitch", "ACZeroBranch", "ACBreak") and ac is not None:
                    if mtype in ("V_FROM", "V_TO"):
                        handled_rows = rows
                        handled_devices = devices
                        re_cols = np.array([], dtype=np.int32)
                        im_cols = np.array([], dtype=np.int32)
                    else:
                        handled = [
                            (row, dev)
                            for row, dev in zip(rows, devices)
                            if dev.name in self.ac_zero_current_cols_by_name
                        ]
                        if not handled:
                            continue
                        handled_rows = np.asarray([row for row, _ in handled], dtype=np.int32)
                        handled_devices = [dev for _, dev in handled]
                        re_list = []
                        im_list = []
                        for dev in handled_devices:
                            re_col, im_col = self.ac_zero_current_cols_by_name[dev.name]
                            re_list.append(re_col)
                            im_list.append(im_col)
                        re_cols = np.asarray(re_list, dtype=np.int32)
                        im_cols = np.asarray(im_list, dtype=np.int32)
                    i_pos = np.asarray([ac.node_pos[dev.i_node] for dev in handled_devices], dtype=np.int32)
                    j_pos = np.asarray([ac.node_pos[dev.j_node] for dev in handled_devices], dtype=np.int32)
                    fast_groups.append(
                        (
                            "ACZero",
                            mtype,
                            handled_rows,
                            i_pos,
                            j_pos,
                            re_cols,
                            im_cols,
                        )
                    )

                elif dtype == "ACGenerator" and ac is not None:
                    pos = np.asarray([ac.node_pos[dev.node] for dev in devices], dtype=np.int32)
                    p_cols = np.asarray([self.ac_generator_p_col_by_name[dev.name] for dev in devices], dtype=np.int32)
                    q_cols = np.asarray([self.ac_generator_q_col_by_name[dev.name] for dev in devices], dtype=np.int32)
                    fast_groups.append(("ACGenerator", mtype, rows, pos, p_cols, q_cols))

                elif dtype == "ACLoad" and ac is not None:
                    pos = np.asarray([ac.node_pos[dev.node] for dev in devices], dtype=np.int32)
                    p_cols = np.asarray([self.ac_load_p_col_by_name[dev.name] for dev in devices], dtype=np.int32)
                    q_cols = np.asarray([self.ac_load_q_col_by_name[dev.name] for dev in devices], dtype=np.int32)
                    fast_groups.append(
                        (
                            "ACLoad",
                            mtype,
                            rows,
                            pos,
                            p_cols,
                            q_cols,
                        )
                    )

                elif dtype == "DCBranch" and dc is not None:
                    i_pos = np.asarray([dc.alive_node_dict[dev.i_node] for dev in devices], dtype=np.int32)
                    j_pos = np.asarray([dc.alive_node_dict[dev.j_node] for dev in devices], dtype=np.int32)
                    inv_r = np.asarray([1.0 / dev.r for dev in devices], dtype=np.float64)
                    fast_groups.append(("DCBranch", mtype, rows, i_pos, j_pos, inv_r))

                elif dtype in ("DCSwitch", "DCZeroBranch", "DCBreak") and dc is not None:
                    if mtype in ("V_FROM", "V_TO"):
                        handled_rows = rows
                        handled_devices = devices
                        current_cols = np.array([], dtype=np.int32)
                    else:
                        handled = [
                            (row, dev)
                            for row, dev in zip(rows, devices)
                            if dev.name in self.dc_zero_current_col_by_name
                        ]
                        if not handled:
                            continue
                        handled_rows = np.asarray([row for row, _ in handled], dtype=np.int32)
                        handled_devices = [dev for _, dev in handled]
                        current_cols = np.asarray(
                            [self.dc_zero_current_col_by_name[dev.name] for dev in handled_devices],
                            dtype=np.int32,
                        )
                    i_pos = np.asarray([dc.alive_node_dict[dev.i_node] for dev in handled_devices], dtype=np.int32)
                    j_pos = np.asarray([dc.alive_node_dict[dev.j_node] for dev in handled_devices], dtype=np.int32)
                    fast_groups.append(("DCZero", mtype, handled_rows, i_pos, j_pos, current_cols))

                elif dtype == "DCGenerator" and dc is not None:
                    pos = np.asarray([dc.alive_node_dict[dev.node] for dev in devices], dtype=np.int32)
                    ctrl = np.asarray(
                        [GEN_CONTROL_KIND[str(dev.control_type).upper()] for dev in devices],
                        dtype=np.int8,
                    )
                    p_col = np.asarray(
                        [self.dc_v_generator_col_by_name.get(dev.name, -1) for dev in devices],
                        dtype=np.int32,
                    )
                    p_set = np.asarray([getattr(dev, "p_set", getattr(dev, "p", 0.0)) for dev in devices], dtype=np.float64)
                    i_set = np.asarray([getattr(dev, "i_set", 0.0) for dev in devices], dtype=np.float64)
                    fast_groups.append(("DCGenerator", mtype, rows, pos, ctrl, p_col, p_set, i_set))

                elif dtype == "DCLoad" and dc is not None:
                    pos = np.asarray([dc.alive_node_dict[dev.node] for dev in devices], dtype=np.int32)
                    coeff = np.asarray([self._load_zip_coefficients(dev) for dev in devices], dtype=np.float64)
                    fast_groups.append(
                        (
                            "DCLoad",
                            mtype,
                            rows,
                            pos,
                            coeff[:, 0],
                            coeff[:, 1],
                            coeff[:, 2],
                        )
                    )

                elif dtype == "DCDCConverter" and dc is not None:
                    d_idx = np.asarray([self.dcdc_pos_by_name[dev.name] for dev in devices], dtype=np.int32)
                    i_pos = np.asarray([dc.alive_node_dict[dev.i_node] for dev in devices], dtype=np.int32)
                    j_pos = np.asarray([dc.alive_node_dict[dev.j_node] for dev in devices], dtype=np.int32)
                    fast_groups.append(("DCDCConverter", mtype, rows, d_idx, i_pos, j_pos))

                elif dtype == "DCACConverter":
                    k = np.asarray([self.dcac_pos_by_name[dev.name] for dev in devices], dtype=np.int32)
                    ac_pos = np.asarray([self.calc.dcac_converters[int(idx)][1] for idx in k], dtype=np.int32)
                    dc_pos = np.asarray([self.calc.dcac_converters[int(idx)][2] for idx in k], dtype=np.int32)
                    fast_groups.append(("DCACConverter", mtype, rows, k, ac_pos, dc_pos))

                elif dtype == "ACACConverter":
                    k = np.asarray([self.acac_pos_by_name[dev.name] for dev in devices], dtype=np.int32)
                    i_pos = np.asarray([self.calc.acac_converters[int(idx)][1] for idx in k], dtype=np.int32)
                    j_pos = np.asarray([self.calc.acac_converters[int(idx)][2] for idx in k], dtype=np.int32)
                    fast_groups.append(("ACACConverter", mtype, rows, k, i_pos, j_pos))

                else:
                    supported = False
            except (KeyError, AttributeError, IndexError):
                supported = False

        self._fast_eval_groups = fast_groups
        self._active_eval_fast_supported = supported

    @staticmethod
    def _fill_values(values: np.ndarray, rows: np.ndarray, generator) -> None:
        values[rows] = np.fromiter(generator, dtype=np.float64, count=int(rows.size))

    def _evaluate_active_measurements_fast(self, x: np.ndarray) -> np.ndarray:
        """Evaluate active h(x) directly from compact state arrays."""
        state = np.asarray(x, dtype=np.float64)
        full_x = self._expand_state_mapped_only(state)
        ac = self.calc.ac_calc
        dc = self.calc.dc_calc

        ac_theta = ac_voltage = ac_voltage_complex = None
        if ac is not None:
            ac_x = full_x[: self.calc.ac_size]
            ac_theta, ac_voltage, _, _ = ac._extract_state_vars(ac_x, update_cache=False)
            ac_voltage_complex = ac_voltage * np.exp(1j * ac_theta)

        dc_voltage = None
        dcdc_power = np.array([], dtype=np.float64)
        if dc is not None:
            dc_x = full_x[self.calc.ac_size : self.calc.ac_size + self.calc.dc_size]
            dc_voltage = dc_x[: dc.N]
            dcdc_start = dc.N + dc.N_phi
            dcdc_power = dc_x[dcdc_start : dcdc_start + 2 * dc.N_dcdc] if dc.N_dcdc else np.array([], dtype=np.float64)

        dcac_power = (
            full_x[self.calc.dcac_start : self.calc.dcac_start + 3 * self.calc.N_dcac].reshape(self.calc.N_dcac, 3)
            if self.calc.N_dcac
            else np.zeros((0, 3), dtype=np.float64)
        )
        acac_power = (
            full_x[self.calc.acac_start : self.calc.acac_start + 4 * self.calc.N_acac].reshape(self.calc.N_acac, 4)
            if self.calc.N_acac
            else np.zeros((0, 4), dtype=np.float64)
        )

        values = np.zeros(len(self.active_measurements), dtype=np.float64)
        ac_generator_totals = None

        def ac_generator_power_totals():
            nonlocal ac_generator_totals
            if ac_generator_totals is None:
                s_network = ac_voltage_complex * np.conj(ac.Y.dot(ac_voltage_complex))
                p_load, q_load = self._ac_load_power_arrays(ac_voltage)
                p_zero, q_zero = self._ac_zero_power_arrays(state, ac_voltage_complex)
                ac_generator_totals = (s_network.real + p_load + p_zero, s_network.imag + q_load + q_zero)
            return ac_generator_totals

        for group in self._fast_eval_groups:
            kind = group[0]
            mtype = group[1]
            rows = group[2]

            if kind == "ACNode":
                pos = group[3]
                if mtype == "V":
                    values[rows] = ac_voltage[pos]
                elif mtype in ("ANGLE", "THETA"):
                    values[rows] = ac_theta[pos]

            elif kind == "DCNode":
                values[rows] = dc_voltage[group[3]]

            elif kind == "DCVoltageDiff":
                values[rows] = dc_voltage[group[3]] - dc_voltage[group[4]]

            elif kind == "ACLine":
                own, other, y_self, y_mutual = group[3], group[4], group[5], group[6]
                if mtype in ("V_FROM", "V_TO"):
                    values[rows] = ac_voltage[own]
                else:
                    current = y_self * ac_voltage_complex[own] + y_mutual * ac_voltage_complex[other]
                    if mtype in ("I_FROM", "I_TO"):
                        values[rows] = np.abs(current)
                    else:
                        power = ac_voltage_complex[own] * np.conj(current)
                        values[rows] = power.real if mtype.startswith("P") else power.imag

            elif kind == "ACZero":
                i_pos, j_pos, re_cols, im_cols = group[3], group[4], group[5], group[6]
                if mtype == "V_FROM":
                    values[rows] = ac_voltage[i_pos]
                elif mtype == "V_TO":
                    values[rows] = ac_voltage[j_pos]
                else:
                    current = state[re_cols] + 1j * state[im_cols]
                    if mtype in ("I_FROM", "I_TO"):
                        values[rows] = np.abs(current)
                        continue
                    power = ac_voltage_complex[i_pos] * np.conj(current)
                    sign = -1.0 if mtype.endswith("_TO") else 1.0
                    values[rows] = sign * (power.real if mtype.startswith("P") else power.imag)

            elif kind == "ACGenerator":
                pos, p_cols, q_cols = group[3], group[4], group[5]
                if mtype == "V_GEN":
                    values[rows] = ac_voltage[pos]
                else:
                    p = state[p_cols]
                    q = state[q_cols]
                    if mtype == "P_GEN":
                        values[rows] = p
                    elif mtype == "Q_GEN":
                        values[rows] = q
                    elif mtype == "I_GEN":
                        values[rows] = np.divide(np.hypot(p, q), ac_voltage[pos], out=np.zeros(rows.size), where=np.abs(ac_voltage[pos]) > self.min_current_voltage)

            elif kind == "ACLoad":
                pos, p_cols, q_cols = group[3], group[4], group[5]
                voltage = ac_voltage[pos]
                p = state[p_cols]
                q = state[q_cols]
                if mtype == "P_LOAD":
                    values[rows] = p
                elif mtype == "Q_LOAD":
                    values[rows] = q
                elif mtype == "V_LOAD":
                    values[rows] = voltage
                elif mtype == "I_LOAD":
                    values[rows] = np.divide(np.hypot(p, q), voltage, out=np.zeros(rows.size), where=np.abs(voltage) > self.min_current_voltage)

            elif kind == "DCBranch":
                i_pos, j_pos, inv_r = group[3], group[4], group[5]
                vi = dc_voltage[i_pos]
                vj = dc_voltage[j_pos]
                current = (vi - vj) * inv_r
                if mtype == "P_FROM":
                    values[rows] = vi * current
                elif mtype == "V_FROM":
                    values[rows] = vi
                elif mtype == "I_FROM":
                    values[rows] = current
                elif mtype == "P_TO":
                    values[rows] = -vj * current
                elif mtype == "V_TO":
                    values[rows] = vj
                elif mtype == "I_TO":
                    values[rows] = -current

            elif kind == "DCZero":
                i_pos, j_pos, current_cols = group[3], group[4], group[5]
                current = state[current_cols]
                if mtype == "P_FROM":
                    values[rows] = dc_voltage[i_pos] * current
                elif mtype == "V_FROM":
                    values[rows] = dc_voltage[i_pos]
                elif mtype == "I_FROM":
                    values[rows] = current
                elif mtype == "P_TO":
                    values[rows] = -dc_voltage[j_pos] * current
                elif mtype == "V_TO":
                    values[rows] = dc_voltage[j_pos]
                elif mtype == "I_TO":
                    values[rows] = -current

            elif kind == "DCGenerator":
                pos, ctrl, p_col, p_set, i_set = group[3], group[4], group[5], group[6], group[7]
                voltage = dc_voltage[pos]
                p = np.where(ctrl == 0, state[p_col], np.where(ctrl == 1, p_set, i_set * voltage))
                current = np.divide(p, voltage, out=np.zeros(rows.size), where=np.abs(voltage) > self.min_current_voltage)
                current = np.where(ctrl == 2, i_set, current)
                if mtype == "P_GEN":
                    values[rows] = p
                elif mtype == "V_GEN":
                    values[rows] = voltage
                elif mtype == "I_GEN":
                    values[rows] = current

            elif kind == "DCLoad":
                pos, pv0, pv1, pv2 = group[3], group[4], group[5], group[6]
                voltage = dc_voltage[pos]
                p = pv0 + pv1 * voltage + pv2 * voltage * voltage
                if mtype == "P_LOAD":
                    values[rows] = p
                elif mtype == "V_LOAD":
                    values[rows] = voltage
                elif mtype == "I_LOAD":
                    values[rows] = np.divide(p, voltage, out=np.zeros(rows.size), where=np.abs(voltage) > self.min_current_voltage)

            elif kind == "DCDCConverter":
                d_idx, i_pos, j_pos = group[3], group[4], group[5]
                p_from = dcdc_power[2 * d_idx]
                p_to = dcdc_power[2 * d_idx + 1]
                if mtype == "P_FROM":
                    values[rows] = p_from
                elif mtype == "V_FROM":
                    values[rows] = dc_voltage[i_pos]
                elif mtype == "I_FROM":
                    values[rows] = np.divide(p_from, dc_voltage[i_pos], out=np.zeros(rows.size), where=np.abs(dc_voltage[i_pos]) > self.min_current_voltage)
                elif mtype == "P_TO":
                    values[rows] = p_to
                elif mtype == "V_TO":
                    values[rows] = dc_voltage[j_pos]
                elif mtype == "I_TO":
                    values[rows] = np.divide(p_to, dc_voltage[j_pos], out=np.zeros(rows.size), where=np.abs(dc_voltage[j_pos]) > self.min_current_voltage)

            elif kind == "DCACConverter":
                k, ac_pos, dc_pos = group[3], group[4], group[5]
                dc_p = dcac_power[k, 0]
                ac_p = dcac_power[k, 1]
                ac_q = dcac_power[k, 2]
                if mtype == "P_DC":
                    values[rows] = dc_p
                elif mtype == "V_DC":
                    values[rows] = dc_voltage[dc_pos]
                elif mtype == "I_DC":
                    values[rows] = np.divide(dc_p, dc_voltage[dc_pos], out=np.zeros(rows.size), where=np.abs(dc_voltage[dc_pos]) > self.min_current_voltage)
                elif mtype == "P_AC":
                    values[rows] = ac_p
                elif mtype == "Q_AC":
                    values[rows] = ac_q
                elif mtype == "V_AC":
                    values[rows] = ac_voltage[ac_pos]
                elif mtype == "I_AC":
                    values[rows] = np.divide(np.hypot(ac_p, ac_q), ac_voltage[ac_pos], out=np.zeros(rows.size), where=np.abs(ac_voltage[ac_pos]) > self.min_current_voltage)

            elif kind == "ACACConverter":
                k, i_pos, j_pos = group[3], group[4], group[5]
                i_p = acac_power[k, 0]
                i_q = acac_power[k, 1]
                j_p = acac_power[k, 2]
                j_q = acac_power[k, 3]
                if mtype == "P_FROM":
                    values[rows] = i_p
                elif mtype == "Q_FROM":
                    values[rows] = i_q
                elif mtype == "V_FROM":
                    values[rows] = ac_voltage[i_pos]
                elif mtype == "I_FROM":
                    values[rows] = np.divide(np.hypot(i_p, i_q), ac_voltage[i_pos], out=np.zeros(rows.size), where=np.abs(ac_voltage[i_pos]) > self.min_current_voltage)
                elif mtype == "P_TO":
                    values[rows] = j_p
                elif mtype == "Q_TO":
                    values[rows] = j_q
                elif mtype == "V_TO":
                    values[rows] = ac_voltage[j_pos]
                elif mtype == "I_TO":
                    values[rows] = np.divide(np.hypot(j_p, j_q), ac_voltage[j_pos], out=np.zeros(rows.size), where=np.abs(ac_voltage[j_pos]) > self.min_current_voltage)

        return values

    def _evaluate_active_measurements(self, x: np.ndarray) -> np.ndarray:
        if self._active_eval_fast_supported:
            values = self._evaluate_active_measurements_fast(x)
            ac_sub_x = self._ac_sub_state_from_hybrid(x)
            if ac_sub_x is not None and self._active_ac_hybrid_rows.size:
                values[self._active_ac_hybrid_rows] = self._ac_sub_estimator.evaluate(
                    ac_sub_x,
                    self._active_ac_sub_measurements,
                )
            dc_sub_x = self._dc_sub_state_from_hybrid(x)
            if dc_sub_x is not None and self._active_dc_hybrid_rows.size:
                values[self._active_dc_hybrid_rows] = self._dc_sub_estimator.evaluate(
                    dc_sub_x,
                    self._active_dc_sub_measurements,
                )
            return values
        self._write_state(x)
        values = np.zeros(len(self.active_measurements), dtype=np.float64)
        fill = self._fill_values
        min_v = self.min_current_voltage
        for dtype, mtype, rows, devices in self._eval_groups:
            if dtype == "ACNode":
                if mtype == "V":
                    fill(values, rows, (dev.voltage for dev in devices))
                elif mtype in ("ANGLE", "THETA"):
                    fill(values, rows, (dev.angle for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")

            elif dtype == "DCNode":
                if mtype == "V":
                    fill(values, rows, (dev.voltage for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCNode measurement type: {mtype}")

            elif dtype in ("ACBranch", "ACTransformer"):
                if mtype == "P_FROM":
                    fill(values, rows, (dev.i_p for dev in devices))
                elif mtype == "Q_FROM":
                    fill(values, rows, (dev.i_q for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (dev.i_c for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (dev.j_p for dev in devices))
                elif mtype == "Q_TO":
                    fill(values, rows, (dev.j_q for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (dev.j_c for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported {dtype} measurement type: {mtype}")

            elif dtype in ("ACSwitch", "ACZeroBranch", "ACBreak"):
                if mtype == "P_FROM":
                    fill(values, rows, (float(getattr(dev, "p", 0.0) or 0.0) for dev in devices))
                elif mtype == "Q_FROM":
                    fill(values, rows, (float(getattr(dev, "q", 0.0) or 0.0) for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (abs(float(getattr(dev, "current", 0.0) or 0.0)) for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (-float(getattr(dev, "p", 0.0) or 0.0) for dev in devices))
                elif mtype == "Q_TO":
                    fill(values, rows, (-float(getattr(dev, "q", 0.0) or 0.0) for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (abs(float(getattr(dev, "current", 0.0) or 0.0)) for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported {dtype} measurement type: {mtype}")

            elif dtype == "ACGenerator":
                if mtype == "P_GEN":
                    fill(values, rows, (dev.p for dev in devices))
                elif mtype == "Q_GEN":
                    fill(values, rows, (dev.q for dev in devices))
                elif mtype == "V_GEN":
                    fill(values, rows, (dev.node_obj.voltage for dev in devices))
                elif mtype == "I_GEN":
                    fill(values, rows, (dev.current for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")

            elif dtype == "ACLoad":
                if mtype == "P_LOAD":
                    fill(values, rows, (dev.p for dev in devices))
                elif mtype == "Q_LOAD":
                    fill(values, rows, (dev.q for dev in devices))
                elif mtype == "V_LOAD":
                    fill(values, rows, (dev.node_obj.voltage for dev in devices))
                elif mtype == "I_LOAD":
                    fill(values, rows, (dev.current for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")

            elif dtype == "DCBranch":
                if mtype == "P_FROM":
                    fill(values, rows, (dev.i_p for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (float(getattr(dev, "current", 0.0) or 0.0) for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (dev.j_p for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (-float(getattr(dev, "current", 0.0) or 0.0) for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCBranch measurement type: {mtype}")

            elif dtype in ("DCSwitch", "DCZeroBranch", "DCBreak"):
                if mtype == "P_FROM":
                    fill(values, rows, (float(getattr(dev, "p", 0.0) or 0.0) for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (float(getattr(dev, "current", 0.0) or 0.0) for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (-dev.j_node_obj.voltage * float(getattr(dev, "current", 0.0) or 0.0) for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (-float(getattr(dev, "current", 0.0) or 0.0) for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported {dtype} measurement type: {mtype}")

            elif dtype == "DCGenerator":
                if mtype == "P_GEN":
                    fill(values, rows, (dev.p for dev in devices))
                elif mtype == "V_GEN":
                    fill(values, rows, (dev.node_obj.voltage for dev in devices))
                elif mtype == "I_GEN":
                    fill(values, rows, (dev.current for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")

            elif dtype == "DCLoad":
                if mtype == "P_LOAD":
                    fill(values, rows, (dev.p for dev in devices))
                elif mtype == "V_LOAD":
                    fill(values, rows, (dev.node_obj.voltage for dev in devices))
                elif mtype == "I_LOAD":
                    fill(values, rows, (dev.current for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCLoad measurement type: {mtype}")

            elif dtype == "DCDCConverter":
                if mtype == "P_FROM":
                    fill(values, rows, (dev.i_p for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (dev.i_c for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (dev.j_p for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (dev.j_c for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCDCConverter measurement type: {mtype}")

            elif dtype == "DCACConverter":
                if mtype == "P_DC":
                    fill(values, rows, (dev.dc_p for dev in devices))
                elif mtype == "V_DC":
                    fill(values, rows, (dev.dc_node_obj.voltage for dev in devices))
                elif mtype == "I_DC":
                    fill(values, rows, (dev.dc_i for dev in devices))
                elif mtype == "P_AC":
                    fill(values, rows, (dev.ac_p for dev in devices))
                elif mtype == "Q_AC":
                    fill(values, rows, (dev.ac_q for dev in devices))
                elif mtype == "V_AC":
                    fill(values, rows, (dev.ac_node_obj.voltage for dev in devices))
                elif mtype == "I_AC":
                    fill(values, rows, (dev.ac_i for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported DCACConverter measurement type: {mtype}")

            elif dtype == "ACACConverter":
                if mtype == "P_FROM":
                    fill(values, rows, (dev.i_p for dev in devices))
                elif mtype == "Q_FROM":
                    fill(values, rows, (dev.i_q for dev in devices))
                elif mtype == "V_FROM":
                    fill(values, rows, (dev.i_node_obj.voltage for dev in devices))
                elif mtype == "I_FROM":
                    fill(values, rows, (dev.i_i for dev in devices))
                elif mtype == "P_TO":
                    fill(values, rows, (dev.j_p for dev in devices))
                elif mtype == "Q_TO":
                    fill(values, rows, (dev.j_q for dev in devices))
                elif mtype == "V_TO":
                    fill(values, rows, (dev.j_node_obj.voltage for dev in devices))
                elif mtype == "I_TO":
                    fill(values, rows, (dev.j_i for dev in devices))
                else:
                    raise RuntimeError(f"Unsupported ACACConverter measurement type: {mtype}")

            else:
                raise RuntimeError(f"Unsupported measurement device type: {dtype}")
        return values

    def _prepare_ac_y_row_cache(self) -> None:
        """Cache sparse AC Y-matrix rows used by generator measurement derivatives."""
        ac = self.calc.ac_calc
        if ac is None:
            return
        self.ac_y_row_nodes = []
        self.ac_y_row_y_conj = []
        self.ac_y_row_off_mask = []
        self.ac_y_row_off_nodes = []
        self.ac_y_row_diag_conj = np.zeros(ac.N, dtype=np.complex128)
        y_matrix = ac.Y
        for pos in range(ac.N):
            if hasattr(y_matrix, "indptr") and hasattr(y_matrix, "indices") and hasattr(y_matrix, "data"):
                start, end = y_matrix.indptr[pos], y_matrix.indptr[pos + 1]
                nodes = y_matrix.indices[start:end].astype(np.int32, copy=True)
                y_values = y_matrix.data[start:end].astype(np.complex128, copy=True)
            else:
                row = y_matrix[pos, :]
                nodes = np.nonzero(row)[0].astype(np.int32)
                y_values = np.asarray(row[nodes], dtype=np.complex128)

            y_conj = np.conj(y_values)
            off_mask = nodes != pos
            diag_values = y_values[nodes == pos]
            self.ac_y_row_nodes.append(nodes)
            self.ac_y_row_y_conj.append(y_conj)
            self.ac_y_row_off_mask.append(off_mask)
            self.ac_y_row_off_nodes.append(nodes[off_mask])
            self.ac_y_row_diag_conj[pos] = np.conj(diag_values[0]) if diag_values.size else 0.0

    @staticmethod
    def _build_phi_solver(
        n_phi: int,
        ref_phi_idx: np.ndarray,
        phi_a: np.ndarray,
        phi_b: np.ndarray,
    ) -> Tuple[np.ndarray, object]:
        """Build a least-squares map from zero-branch currents back to node potentials."""
        if n_phi == 0 or phi_a.size == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 0), dtype=np.float64)

        ref_mask = np.zeros(n_phi, dtype=bool)
        if ref_phi_idx.size:
            ref_mask[ref_phi_idx.astype(np.int32)] = True
        free = np.where(~ref_mask)[0].astype(np.int32)
        if free.size == 0:
            return free, np.zeros((0, phi_a.size), dtype=np.float64)

        if phi_a.size == free.size:
            tree_solver = HybridStateEstimator._build_tree_phi_solver(n_phi, ref_mask, phi_a, phi_b)
            if tree_solver is not None:
                return free, tree_solver

        free_pos = {int(phi): idx for idx, phi in enumerate(free)}
        incidence = np.zeros((phi_a.size, free.size), dtype=np.float64)
        for row, (a, b) in enumerate(zip(phi_a.astype(np.int32), phi_b.astype(np.int32))):
            if int(a) in free_pos:
                incidence[row, free_pos[int(a)]] = 1.0
            if int(b) in free_pos:
                incidence[row, free_pos[int(b)]] -= 1.0
        return free, np.linalg.pinv(incidence)

    @staticmethod
    def _build_tree_phi_solver(
        n_phi: int,
        ref_mask: np.ndarray,
        phi_a: np.ndarray,
        phi_b: np.ndarray,
    ):
        """Precompute a tree traversal for phi_a - phi_b = current edge equations."""
        adjacency = [[] for _ in range(n_phi)]
        for edge_idx, (a_raw, b_raw) in enumerate(zip(phi_a.astype(np.int32), phi_b.astype(np.int32))):
            a = int(a_raw)
            b = int(b_raw)
            adjacency[a].append((b, int(edge_idx), -1.0))
            adjacency[b].append((a, int(edge_idx), 1.0))

        seen = ref_mask.copy()
        stack = [int(idx) for idx in np.flatnonzero(ref_mask)]
        order_nodes = []
        order_parents = []
        order_edges = []
        order_signs = []
        while stack:
            parent = stack.pop()
            for child, edge_idx, sign in adjacency[parent]:
                if seen[child]:
                    continue
                seen[child] = True
                order_nodes.append(child)
                order_parents.append(parent)
                order_edges.append(edge_idx)
                order_signs.append(sign)
                stack.append(child)
        if not np.all(seen):
            return None
        return (
            "tree",
            np.asarray(order_nodes, dtype=np.int32),
            np.asarray(order_parents, dtype=np.int32),
            np.asarray(order_edges, dtype=np.int32),
            np.asarray(order_signs, dtype=np.float64),
        )

    @staticmethod
    def _currents_to_phi(
        currents: np.ndarray,
        n_phi: int,
        free: np.ndarray,
        pinv,
    ) -> np.ndarray:
        # Zero-branch measurements estimate current differences; Newton equations need phi.
        phi = np.zeros(n_phi, dtype=np.float64)
        if n_phi and free.size:
            if isinstance(pinv, tuple) and pinv and pinv[0] == "tree":
                _, nodes, parents, edges, signs = pinv
                for node, parent, edge_idx, sign in zip(nodes, parents, edges, signs):
                    phi[int(node)] = phi[int(parent)] + float(sign) * currents[int(edge_idx)]
                return phi
            phi[free] = pinv @ currents
        return phi

    def _pack_estimation_state(self, full_x: np.ndarray, flat: bool = False) -> np.ndarray:
        """Project the full hybrid Newton vector to the smaller WLS estimation vector."""
        x = np.zeros(self.n_state, dtype=np.float64)
        if self.mapped_state_cols.size:
            x[self.mapped_state_cols] = full_x[self.mapped_full_cols]
        if not flat and self.calc.ac_calc is not None:
            ac = self.calc.ac_calc
            rebased_state_cols = set()
            for node_pos, full_col in ac.theta_idx.items():
                state_col = self._state_col_from_full(int(full_col))
                if state_col >= 0 and state_col not in rebased_state_cols:
                    x[state_col] -= self.ac_reference_angle_by_pos.get(int(node_pos), 0.0)
                    rebased_state_cols.add(state_col)
        if self.calc.ac_calc is not None and self.ac_virtual_theta_specs:
            ac = self.calc.ac_calc
            for node_pos, state_col in self.ac_virtual_theta_specs:
                if flat:
                    x[state_col] = 0.0
                else:
                    x[state_col] = (
                        self._ac_original_angle_by_pos(int(node_pos))
                        - self.ac_reference_angle_by_pos.get(int(node_pos), 0.0)
                    )

        if self.calc.ac_calc is not None and self.ac_zero_re_cols.size:
            ac = self.calc.ac_calc
            ac_x = full_x[: self.calc.ac_size]
            phi_re = ac_x[ac.base_phi_re : ac.base_phi_re + ac.N_phi]
            phi_im = ac_x[ac.base_phi_im : ac.base_phi_im + ac.N_phi]
            current = (
                phi_re[self.ac_zero_phi_a] - phi_re[self.ac_zero_phi_b]
                + 1j * (phi_im[self.ac_zero_phi_a] - phi_im[self.ac_zero_phi_b])
            )
            x[self.ac_zero_re_cols] = current.real
            x[self.ac_zero_im_cols] = current.imag
        for gen in self.ac_generator_by_name.values():
            if gen.name in self.ac_generator_p_col_by_name:
                x[self.ac_generator_p_col_by_name[gen.name]] = (
                    float(getattr(gen, "p_set", 0.0) or 0.0) if flat else float(getattr(gen, "p", 0.0) or 0.0)
                )
            if gen.name in self.ac_generator_q_col_by_name:
                x[self.ac_generator_q_col_by_name[gen.name]] = (
                    float(getattr(gen, "q_set", 0.0) or 0.0) if flat else float(getattr(gen, "q", 0.0) or 0.0)
                )
        for load in self.ac_load_by_name.values():
            if load.name in self.ac_load_p_col_by_name:
                x[self.ac_load_p_col_by_name[load.name]] = float(getattr(load, "p", 0.0) or 0.0)
            if load.name in self.ac_load_q_col_by_name:
                x[self.ac_load_q_col_by_name[load.name]] = float(getattr(load, "q", 0.0) or 0.0)

        if self.calc.dc_calc is not None and self.dc_zero_cols.size:
            dc = self.calc.dc_calc
            dc_phi_start = self.calc.ac_size + dc.N
            phi = full_x[dc_phi_start : dc_phi_start + dc.N_phi]
            x[self.dc_zero_cols] = phi[self.dc_zero_phi_a] - phi[self.dc_zero_phi_b]

        for gen, pos in self.dc_v_generator_states:
            if flat:
                x[pos] = float(getattr(gen, "p_set", 0.0) or 0.0)
            else:
                x[pos] = float(getattr(gen, "p", 0.0) or 0.0)
        if self.ac_sub_seed_hybrid_cols.size:
            ac_seed = self._ac_sub_seed_state(flat)
            if ac_seed is not None:
                x[self.ac_sub_seed_hybrid_cols] = ac_seed[self.ac_sub_seed_sub_cols]
        return x

    def _ac_sub_seed_state(self, flat: bool) -> Optional[np.ndarray]:
        ac_sub = getattr(self, "_ac_sub_estimator", None)
        if ac_sub is None:
            return None
        if flat and hasattr(ac_sub, "_pack_state"):
            theta = np.zeros(len(ac_sub.nodes), dtype=np.float64)
            voltage = np.ones(len(ac_sub.nodes), dtype=np.float64)
            return ac_sub._pack_state(theta, voltage, rebase_angles=False)
        if not flat and hasattr(ac_sub, "_file_state"):
            return ac_sub._file_state()
        return ac_sub.initial_state()

    def _apply_virtual_ac_specs(self, x: np.ndarray) -> None:
        """Write estimator-only AC reference replacements into the AC extractor specs."""
        if self.calc.ac_calc is None or not self.ac_virtual_theta_specs:
            return
        ac = self.calc.ac_calc
        for node_pos, state_col in self.ac_virtual_theta_specs:
            ac.theta_spec[int(node_pos)] = float(x[int(state_col)])

    def _expand_state(self, x: np.ndarray) -> np.ndarray:
        """Expand WLS state variables back to the full hybrid Newton vector layout."""
        delegate = self._delegate()
        if delegate is not None:
            full = self.power_flow_x.copy()
            if delegate is self._ac_sub_estimator and self.calc.ac_calc is not None:
                theta, voltage = delegate._unpack_state(np.asarray(x, dtype=np.float64))
                ac = self.calc.ac_calc
                for node in delegate.nodes:
                    h_node = self.ac_node_by_name.get(node.name)
                    if h_node is None or h_node.idx not in ac.node_pos or node.idx not in delegate.node_pos:
                        continue
                    h_pos = int(ac.node_pos[h_node.idx])
                    s_pos = int(delegate.node_pos[node.idx])
                    if h_pos in ac.theta_idx:
                        full[int(ac.theta_idx[h_pos])] = theta[s_pos]
                    if h_pos in ac.V_idx:
                        full[int(ac.n_theta + ac.V_idx[h_pos])] = voltage[s_pos]
                return full
            if delegate is self._dc_sub_estimator and self.calc.dc_calc is not None:
                state = np.asarray(x, dtype=np.float64)
                dc = self.calc.dc_calc
                dc_start = self.calc.ac_size
                for node_idx, h_pos in dc.alive_node_dict.items():
                    if node_idx not in delegate.node_pos:
                        continue
                    s_pos = int(delegate.node_pos[node_idx])
                    col = int(delegate.voltage_col[s_pos]) if hasattr(delegate, "voltage_col") else s_pos
                    if col >= 0:
                        full[dc_start + h_pos] = state[col]
                return full
        full = self.power_flow_x.copy()
        if self.mapped_state_cols.size:
            full[self.mapped_full_cols] = x[self.mapped_state_cols]
        for target_col, source_col in self.tied_full_cols:
            if source_col >= 0:
                full[target_col] = full[source_col]
        if self.fixed_full_cols.size:
            full[self.fixed_full_cols] = self.fixed_full_values
        self._apply_virtual_ac_specs(x)

        if self.calc.ac_calc is not None and self.ac_zero_re_cols.size:
            ac = self.calc.ac_calc
            current_re = x[self.ac_zero_re_cols]
            current_im = x[self.ac_zero_im_cols]
            phi_re = self._currents_to_phi(current_re, ac.N_phi, self.ac_phi_free, self.ac_phi_pinv)
            phi_im = self._currents_to_phi(current_im, ac.N_phi, self.ac_phi_free, self.ac_phi_pinv)
            full[ac.base_phi_re : ac.base_phi_re + ac.N_phi] = phi_re
            full[ac.base_phi_im : ac.base_phi_im + ac.N_phi] = phi_im

        if self.calc.dc_calc is not None and self.dc_zero_cols.size:
            dc = self.calc.dc_calc
            current = x[self.dc_zero_cols]
            phi = self._currents_to_phi(current, dc.N_phi, self.dc_phi_free, self.dc_phi_pinv)
            dc_phi_start = self.calc.ac_size + dc.N
            full[dc_phi_start : dc_phi_start + dc.N_phi] = phi
        return full

    def _expand_state_mapped_only(self, x: np.ndarray) -> np.ndarray:
        """Expand mapped state columns without reconstructing zero-branch phi variables.

        Jacobian assembly only needs node angles, voltages and converter powers.
        Zero-branch currents are read directly from the compact estimator state, so
        the least-squares phi reconstruction in _expand_state would be wasted work.
        """
        delegate = self._delegate()
        if delegate is not None:
            return self._expand_state(x)
        full = self.power_flow_x.copy()
        if self.mapped_state_cols.size:
            full[self.mapped_full_cols] = x[self.mapped_state_cols]
        for target_col, source_col in self.tied_full_cols:
            if source_col >= 0:
                full[target_col] = full[source_col]
        if self.fixed_full_cols.size:
            full[self.fixed_full_cols] = self.fixed_full_values
        self._apply_virtual_ac_specs(x)
        return full

    def _dc_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        cols = getattr(self, "_dc_sub_to_hybrid_cols", np.array([], dtype=np.int32))
        dc_sub = getattr(self, "_dc_sub_estimator", None)
        if dc_sub is None or cols.size != dc_sub.n_state or np.any(cols < 0):
            return None
        return np.asarray(x, dtype=np.float64)[cols]

    def _ac_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        cols = getattr(self, "_ac_sub_to_hybrid_cols", np.array([], dtype=np.int32))
        ac_sub = getattr(self, "_ac_sub_estimator", None)
        if ac_sub is None or cols.size != ac_sub.n_state:
            return None
        state = np.zeros(ac_sub.n_state, dtype=np.float64)
        mapped = cols >= 0
        state[mapped] = np.asarray(x, dtype=np.float64)[cols[mapped]]
        sub_initial = ac_sub.initial_state()
        state[~mapped] = sub_initial[~mapped]
        return state

    def _append_ac_sub_jacobian(self, H: SparseJacobianBuilder, x: np.ndarray) -> None:
        """Append reusable AC subsystem rows from ACStateEstimator into hybrid H."""
        if not getattr(self, "_active_ac_sub_measurements", None):
            return
        ac_sub_x = self._ac_sub_state_from_hybrid(x)
        if ac_sub_x is None:
            return
        sub_h = self._ac_sub_estimator.jacobian_sparse(ac_sub_x, self._active_ac_sub_measurements).tocoo()
        if sub_h.nnz == 0:
            return
        sub_cols = sub_h.col.astype(np.int32, copy=False)
        hybrid_cols = self._ac_sub_to_hybrid_cols[sub_cols]
        keep = hybrid_cols >= 0
        if not np.any(keep):
            return
        H._append_arrays(
            self._active_ac_hybrid_rows[sub_h.row.astype(np.int32, copy=False)[keep]],
            hybrid_cols[keep],
            sub_h.data.astype(np.float64, copy=False)[keep],
        )

    def _append_dc_sub_jacobian(self, H: SparseJacobianBuilder, x: np.ndarray) -> None:
        """Append reusable DC subsystem rows from DCStateEstimator into hybrid H."""
        if not getattr(self, "_active_dc_sub_measurements", None):
            return
        dc_sub_x = self._dc_sub_state_from_hybrid(x)
        if dc_sub_x is None:
            return
        sub_h = self._dc_sub_estimator.jacobian_sparse(dc_sub_x, self._active_dc_sub_measurements).tocoo()
        if sub_h.nnz == 0:
            return
        H._append_arrays(
            self._active_dc_hybrid_rows[sub_h.row.astype(np.int32, copy=False)],
            self._dc_sub_to_hybrid_cols[sub_h.col.astype(np.int32, copy=False)],
            sub_h.data.astype(np.float64, copy=False),
        )

    def initial_state(self) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.initial_state()
        return self.flat_state.copy() if self.flat_start else self.power_flow_state.copy()

    def _write_state(self, x: np.ndarray) -> None:
        """Expand an estimator state and refresh all model objects used by measurement evaluation."""
        state = np.asarray(x, dtype=np.float64)
        if (
            self._last_written_state is not None
            and self._last_written_state.shape == state.shape
            and np.array_equal(self._last_written_state, state)
        ):
            return

        full_x = self._expand_state(state)
        self.calc.x = full_x.copy()
        self.calc._write_back(self.calc.x)
        if self.calc.ac_calc is not None and getattr(self.calc.ac_calc, "array_mode", False):
            self._sync_ac_zero_objects_from_array_result()
        for gen in self.ac_generator_by_name.values():
            p_col = self.ac_generator_p_col_by_name.get(gen.name)
            q_col = self.ac_generator_q_col_by_name.get(gen.name)
            if p_col is not None:
                gen.p = float(state[p_col])
            if q_col is not None:
                gen.q = float(state[q_col])
            voltage = float(getattr(gen.node_obj, "voltage", 1.0) or 1.0)
            gen.current = self._safe_current(float(getattr(gen, "p", 0.0) or 0.0), float(getattr(gen, "q", 0.0) or 0.0), voltage)
        for load in self.ac_load_by_name.values():
            p_col = self.ac_load_p_col_by_name.get(load.name)
            q_col = self.ac_load_q_col_by_name.get(load.name)
            if p_col is not None:
                load.p = float(state[p_col])
            if q_col is not None:
                load.q = float(state[q_col])
            voltage = float(getattr(load.node_obj, "voltage", 1.0) or 1.0)
            load.current = self._safe_current(float(getattr(load, "p", 0.0) or 0.0), float(getattr(load, "q", 0.0) or 0.0), voltage)
        for gen, pos in self.dc_v_generator_states:
            gen.p = float(state[pos])
            voltage = float(gen.node_obj.voltage)
            gen.current = gen.p / voltage if abs(voltage) > self.min_current_voltage else 0.0
        self._last_written_state = state.copy()

    def _sync_ac_zero_objects_from_array_result(self) -> None:
        """Mirror array-mode ideal AC branch results not covered by HybridPowerFlowCalc."""
        ac_calc = self.calc.ac_calc
        result = getattr(ac_calc, "result", None)
        if not result:
            return

        zero_branch = result.get("zero_branch")
        zero_names = ac_calc.ppc.get("zero_branch_name")
        if zero_branch is not None and zero_names is not None:
            for row_idx in np.nonzero(zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] == 1)[0]:
                dev = self.ac_zero_branch_by_name.get(str(zero_names[int(row_idx)]))
                if dev is None:
                    continue
                dev.p = float(zero_branch[row_idx, ZERO_BRANCH_COLS["p"]])
                dev.q = float(zero_branch[row_idx, ZERO_BRANCH_COLS["q"]])
                dev.current = float(zero_branch[row_idx, ZERO_BRANCH_COLS["current"]])

        switch = result.get("switch")
        switch_names = ac_calc.ppc.get("switch_name")
        if switch is not None and switch_names is not None:
            mask = (switch[:, SWITCH_COLS["run_stat"]] == 1) & (switch[:, SWITCH_COLS["status"]] == 1)
            for row_idx in np.nonzero(mask)[0]:
                dev = self.ac_switch_by_name.get(str(switch_names[int(row_idx)]))
                if dev is None:
                    continue
                dev.p = float(switch[row_idx, SWITCH_COLS["p"]])
                dev.q = float(switch[row_idx, SWITCH_COLS["q"]])
                dev.current = float(switch[row_idx, SWITCH_COLS["current"]])

    def _safe_current(self, p: float, q_or_v: float, voltage: Optional[float] = None) -> float:
        if voltage is None:
            voltage = q_or_v
            q = 0.0
        else:
            q = q_or_v
        if abs(voltage) <= self.min_current_voltage:
            return 0.0
        return float(np.hypot(p, q) / voltage)

    @staticmethod
    def _add_derivative(H: np.ndarray, row: int, col: int, value: float) -> None:
        if col < 0 or value == 0.0:
            return
        if isinstance(H, SparseJacobianBuilder):
            H.rows.append(int(row))
            H.cols.append(int(col))
            H.data.append(float(value))
        else:
            H[row, col] += float(value)

    @staticmethod
    def _add_derivatives(H: np.ndarray, row: int, cols: Sequence[int], values: Sequence[float]) -> None:
        if isinstance(H, SparseJacobianBuilder):
            row_int = int(row)
            rows = H.rows
            h_cols = H.cols
            data = H.data
            for col, value in zip(cols, values):
                if col >= 0 and value != 0.0:
                    rows.append(row_int)
                    h_cols.append(int(col))
                    data.append(float(value))
        else:
            row_view = H[row]
            for col, value in zip(cols, values):
                if col >= 0 and value != 0.0:
                    row_view[int(col)] += float(value)

    @staticmethod
    def _add_measurement_row(H: np.ndarray, row: int, cols: np.ndarray, values: np.ndarray) -> None:
        """Append a sparse measurement row without materializing a dense H[row, :]."""
        if cols.size == 0:
            return
        mask = (cols >= 0) & (values != 0.0)
        if not np.any(mask):
            return
        cols_valid = cols[mask].astype(np.int32, copy=False)
        values_valid = values[mask].astype(np.float64, copy=False)
        if isinstance(H, SparseJacobianBuilder):
            row_int = int(row)
            H.rows.extend([row_int] * int(cols_valid.size))
            H.cols.extend(cols_valid.tolist())
            H.data.extend(values_valid.tolist())
        else:
            np.add.at(H[row], cols_valid, values_valid)

    def _append_fast_dynamic_jacobian(
        self,
        H: SparseJacobianBuilder,
        x: np.ndarray,
        ac_theta: Optional[np.ndarray],
        ac_voltage: Optional[np.ndarray],
        ac_voltage_complex: Optional[np.ndarray],
        dc_voltage: Optional[np.ndarray],
        delegated_row_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Append vectorized dynamic Jacobian rows for simple DC/converter measurements."""
        if ac_theta is not None and ac_voltage is not None and self._jac_ac_branch_power_rows.size:
            rows = self._jac_ac_branch_power_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                own = self._jac_ac_branch_power_own[keep]
                other = self._jac_ac_branch_power_other[keep]
                angle = ac_theta[own] - ac_theta[other]
                exp_angle = np.exp(1j * angle)
                y_self_conj = np.conj(self._jac_ac_branch_power_y_self[keep])
                y_mutual_conj = np.conj(self._jac_ac_branch_power_y_mutual[keep])
                off = y_mutual_conj * ac_voltage[own] * ac_voltage[other] * exp_angle
                dtheta_own = 1j * off
                dtheta_other = -1j * off
                dvoltage_own = 2.0 * y_self_conj * ac_voltage[own] + y_mutual_conj * ac_voltage[other] * exp_angle
                dvoltage_other = y_mutual_conj * ac_voltage[own] * exp_angle
                is_p = self._jac_ac_branch_power_is_p[keep]
                H.add_many(
                    np.concatenate((rows, rows, rows, rows)),
                    np.concatenate(
                        (
                            self.ac_theta_state_col[own],
                            self.ac_theta_state_col[other],
                            self.ac_voltage_state_col[own],
                            self.ac_voltage_state_col[other],
                        )
                    ),
                    np.concatenate(
                        (
                            np.where(is_p, dtheta_own.real, dtheta_own.imag),
                            np.where(is_p, dtheta_other.real, dtheta_other.imag),
                            np.where(is_p, dvoltage_own.real, dvoltage_own.imag),
                            np.where(is_p, dvoltage_other.real, dvoltage_other.imag),
                        )
                    ),
                )

        if (
            ac_theta is not None
            and ac_voltage is not None
            and ac_voltage_complex is not None
            and self._jac_ac_branch_current_rows.size
        ):
            rows = self._jac_ac_branch_current_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                own = self._jac_ac_branch_current_own[keep]
                other = self._jac_ac_branch_current_other[keep]
                y_self = self._jac_ac_branch_current_y_self[keep]
                y_mutual = self._jac_ac_branch_current_y_mutual[keep]
                v_own = ac_voltage_complex[own]
                v_other = ac_voltage_complex[other]
                current = y_self * v_own + y_mutual * v_other
                current_abs = np.abs(current)
                valid = current_abs > 1e-12
                if np.any(valid):
                    scale = np.zeros_like(current, dtype=np.complex128)
                    scale[valid] = np.conj(current[valid]) / current_abs[valid]
                    exp_own = np.exp(1j * ac_theta[own])
                    exp_other = np.exp(1j * ac_theta[other])
                    values = (
                        (scale * (1j * y_self * v_own)).real,
                        (scale * (1j * y_mutual * v_other)).real,
                        (scale * (y_self * exp_own)).real,
                        (scale * (y_mutual * exp_other)).real,
                    )
                    H.add_many(
                        np.concatenate((rows, rows, rows, rows)),
                        np.concatenate(
                            (
                                self.ac_theta_state_col[own],
                                self.ac_theta_state_col[other],
                                self.ac_voltage_state_col[own],
                                self.ac_voltage_state_col[other],
                            )
                        ),
                        np.concatenate(values),
                        np.tile(valid, 4),
                    )

        if ac_voltage is not None and self._jac_ac_load_rows.size:
            rows = self._jac_ac_load_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                kind = self._jac_ac_load_kind[keep]
                voltage = ac_voltage[self._jac_ac_load_v_pos[keep]]
                pv0 = self._jac_ac_load_pv0[keep]
                pv1 = self._jac_ac_load_pv1[keep]
                pv2 = self._jac_ac_load_pv2[keep]
                qv0 = self._jac_ac_load_qv0[keep]
                qv1 = self._jac_ac_load_qv1[keep]
                qv2 = self._jac_ac_load_qv2[keep]
                p = pv0 + pv1 * voltage + pv2 * voltage * voltage
                q = qv0 + qv1 * voltage + qv2 * voltage * voltage
                dp = pv1 + 2.0 * pv2 * voltage
                dq = qv1 + 2.0 * qv2 * voltage
                values = np.empty(rows.size, dtype=np.float64)
                p_mask = kind == 0
                q_mask = kind == 1
                i_mask = kind == 2
                values[p_mask] = dp[p_mask]
                values[q_mask] = dq[q_mask]
                if np.any(i_mask):
                    s_abs = np.hypot(p[i_mask], q[i_mask])
                    v_i = voltage[i_mask]
                    valid_i = (s_abs > 1e-12) & (np.abs(v_i) > self.min_current_voltage)
                    i_values = np.zeros(s_abs.size, dtype=np.float64)
                    if np.any(valid_i):
                        i_values[valid_i] = (
                            (p[i_mask][valid_i] * dp[i_mask][valid_i] + q[i_mask][valid_i] * dq[i_mask][valid_i])
                            / (s_abs[valid_i] * v_i[valid_i])
                            - s_abs[valid_i] / (v_i[valid_i] * v_i[valid_i])
                        )
                    values[i_mask] = i_values
                H.add_many(rows, self._jac_ac_load_v_col[keep], values)

        if ac_voltage is not None and ac_voltage_complex is not None and self._jac_ac_zero_current_rows.size:
            rows = self._jac_ac_zero_current_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                re_col = self._jac_ac_zero_current_re_col[keep]
                im_col = self._jac_ac_zero_current_im_col[keep]
                current_re = x[re_col]
                current_im = x[im_col]
                current_abs = np.hypot(current_re, current_im)
                valid = current_abs > 1e-12
                if np.any(valid):
                    H._append_arrays(
                        np.repeat(rows[valid], 2),
                        np.column_stack((re_col[valid], im_col[valid])).ravel(),
                        np.column_stack((current_re[valid] / current_abs[valid], current_im[valid] / current_abs[valid])).ravel(),
                    )

        if ac_voltage is not None and ac_voltage_complex is not None and self._jac_ac_zero_power_rows.size:
            rows = self._jac_ac_zero_power_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                pos = self._jac_ac_zero_power_pos[keep]
                sign = self._jac_ac_zero_power_sign[keep]
                is_p = self._jac_ac_zero_power_is_p[keep]
                theta_col = self._jac_ac_zero_power_theta_col[keep]
                v_col = self._jac_ac_zero_power_v_col[keep]
                re_col = self._jac_ac_zero_power_re_col[keep]
                im_col = self._jac_ac_zero_power_im_col[keep]
                voltage = ac_voltage[pos]
                voltage_complex = ac_voltage_complex[pos]
                current = x[re_col] + 1j * x[im_col]
                s = voltage_complex * np.conj(current)
                dtheta = 1j * s
                dvoltage = np.zeros(rows.size, dtype=np.complex128)
                valid_v = np.abs(voltage) > 1e-12
                dvoltage[valid_v] = s[valid_v] / voltage[valid_v]
                dcurrent_re = voltage_complex
                dcurrent_im = -1j * voltage_complex
                H.add_many(
                    rows,
                    theta_col,
                    sign * np.where(is_p, dtheta.real, dtheta.imag),
                )
                H._append_arrays(
                    np.repeat(rows, 3),
                    np.column_stack((v_col, re_col, im_col)).ravel(),
                    np.column_stack(
                        (
                            sign * np.where(is_p, dvoltage.real, dvoltage.imag),
                            sign * np.where(is_p, dcurrent_re.real, dcurrent_re.imag),
                            sign * np.where(is_p, dcurrent_im.real, dcurrent_im.imag),
                        )
                    ).ravel(),
                )

        if dc_voltage is not None and self._jac_dc_branch_power_rows.size:
            rows = self._jac_dc_branch_power_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if not np.any(keep):
                rows = rows[:0]
            else:
                rows = rows[keep]
            vi = dc_voltage[self._jac_dc_branch_power_i_pos]
            vj = dc_voltage[self._jac_dc_branch_power_j_pos]
            g = self._jac_dc_branch_power_g
            is_from = self._jac_dc_branch_power_from
            if delegated_row_mask is not None and rows.size:
                vi = vi[keep]
                vj = vj[keep]
                g = g[keep]
                is_from = is_from[keep]
                i_col = self._jac_dc_branch_power_i_col[keep]
                j_col = self._jac_dc_branch_power_j_col[keep]
            else:
                i_col = self._jac_dc_branch_power_i_col
                j_col = self._jac_dc_branch_power_j_col
            d_i = np.where(is_from, (2.0 * vi - vj) * g, -vj * g)
            d_j = np.where(is_from, -vi * g, (-vi + 2.0 * vj) * g)
            H._append_arrays(
                np.repeat(rows, 2),
                np.column_stack((i_col, j_col)).ravel(),
                np.column_stack((d_i, d_j)).ravel(),
            )

        if dc_voltage is not None and self._jac_dc_zero_power_rows.size:
            rows = self._jac_dc_zero_power_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if not np.any(keep):
                rows = rows[:0]
            else:
                rows = rows[keep]
            sign = self._jac_dc_zero_power_sign
            i_col = self._jac_dc_zero_power_i_col
            v_pos = self._jac_dc_zero_power_v_pos
            v_col = self._jac_dc_zero_power_v_col
            if delegated_row_mask is not None and rows.size:
                sign = sign[keep]
                i_col = i_col[keep]
                v_pos = v_pos[keep]
                v_col = v_col[keep]
            current = x[i_col]
            voltage = dc_voltage[v_pos]
            H._append_arrays(
                np.repeat(rows, 2),
                np.column_stack((v_col, i_col)).ravel(),
                np.column_stack((sign * current, sign * voltage)).ravel(),
            )

        if dc_voltage is not None and self._jac_dcdc_current_rows.size:
            rows = self._jac_dcdc_current_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if not np.any(keep):
                rows = rows[:0]
                p_col = self._jac_dcdc_current_p_col[:0]
                v_col = self._jac_dcdc_current_v_col[:0]
                v_pos = self._jac_dcdc_current_v_pos[:0]
            elif delegated_row_mask is not None:
                rows = rows[keep]
                p_col = self._jac_dcdc_current_p_col[keep]
                v_col = self._jac_dcdc_current_v_col[keep]
                v_pos = self._jac_dcdc_current_v_pos[keep]
            else:
                p_col = self._jac_dcdc_current_p_col
                v_col = self._jac_dcdc_current_v_col
                v_pos = self._jac_dcdc_current_v_pos
            voltage = dc_voltage[v_pos]
            valid = np.abs(voltage) > self.min_current_voltage
            if np.any(valid):
                p = x[p_col[valid]]
                v = voltage[valid]
                H._append_arrays(
                    np.repeat(rows[valid], 2),
                    np.column_stack((p_col[valid], v_col[valid])).ravel(),
                    np.column_stack((1.0 / v, -p / (v * v))).ravel(),
                )

        if dc_voltage is not None and self._jac_dcac_i_dc_rows.size:
            rows = self._jac_dcac_i_dc_rows
            p_col = self._jac_dcac_i_dc_p_col
            v_col = self._jac_dcac_i_dc_v_col
            voltage = dc_voltage[self._jac_dcac_i_dc_v_pos]
            valid = np.abs(voltage) > self.min_current_voltage
            if np.any(valid):
                p = x[p_col[valid]]
                v = voltage[valid]
                H._append_arrays(
                    np.repeat(rows[valid], 2),
                    np.column_stack((p_col[valid], v_col[valid])).ravel(),
                    np.column_stack((1.0 / v, -p / (v * v))).ravel(),
                )

        if ac_voltage is not None and self._jac_dcac_i_ac_rows.size:
            rows = self._jac_dcac_i_ac_rows
            p_col = self._jac_dcac_i_ac_p_col
            q_col = self._jac_dcac_i_ac_q_col
            v_col = self._jac_dcac_i_ac_v_col
            voltage = ac_voltage[self._jac_dcac_i_ac_v_pos]
            p = x[p_col]
            q = x[q_col]
            s_abs = np.hypot(p, q)
            valid = (s_abs > 1e-12) & (np.abs(voltage) > self.min_current_voltage)
            if np.any(valid):
                p_valid = p[valid]
                q_valid = q[valid]
                v_valid = voltage[valid]
                s_valid = s_abs[valid]
                H._append_arrays(
                    np.repeat(rows[valid], 3),
                    np.column_stack((p_col[valid], q_col[valid], v_col[valid])).ravel(),
                    np.column_stack(
                        (
                            p_valid / (s_valid * v_valid),
                            q_valid / (s_valid * v_valid),
                            -s_valid / (v_valid * v_valid),
                        )
                    ).ravel(),
                )

        if dc_voltage is not None and self._jac_dc_load_rows.size:
            rows = self._jac_dc_load_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                voltage = dc_voltage[self._jac_dc_load_v_pos[keep]]
                kind = self._jac_dc_load_kind[keep]
                v_col = self._jac_dc_load_v_col[keep]
                pv0 = self._jac_dc_load_pv0[keep]
                pv1 = self._jac_dc_load_pv1[keep]
                pv2 = self._jac_dc_load_pv2[keep]
                values = np.where(
                    kind == 0,
                    pv1 + 2.0 * pv2 * voltage,
                    pv2 - pv0 / (voltage * voltage),
                )
                H.add_many(rows, v_col, values)

        if dc_voltage is not None and self._jac_dc_gen_i_v_rows.size:
            rows = self._jac_dc_gen_i_v_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                rows = rows[keep]
                p_col = self._jac_dc_gen_i_v_p_col[keep]
                v_col = self._jac_dc_gen_i_v_v_col[keep]
                voltage = dc_voltage[self._jac_dc_gen_i_v_v_pos[keep]]
                valid = np.abs(voltage) > self.min_current_voltage
                if np.any(valid):
                    p = x[p_col[valid]]
                    v = voltage[valid]
                    H._append_arrays(
                        np.repeat(rows[valid], 2),
                        np.column_stack((p_col[valid], v_col[valid])).ravel(),
                        np.column_stack((1.0 / v, -p / (v * v))).ravel(),
                    )

        if dc_voltage is not None and self._jac_dc_gen_i_p_rows.size:
            rows = self._jac_dc_gen_i_p_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                voltage = dc_voltage[self._jac_dc_gen_i_p_v_pos[keep]]
                values = -self._jac_dc_gen_i_p_p[keep] / (voltage * voltage)
                H.add_many(rows[keep], self._jac_dc_gen_i_p_v_col[keep], values)

        if self._jac_dc_gen_p_i_rows.size:
            rows = self._jac_dc_gen_p_i_rows
            keep = np.ones(rows.size, dtype=bool) if delegated_row_mask is None else ~delegated_row_mask[rows]
            if np.any(keep):
                H.add_many(rows[keep], self._jac_dc_gen_p_i_v_col[keep], self._jac_dc_gen_p_i_i[keep])

    @staticmethod
    def _append_sparse_rows_from_template(
        H: SparseJacobianBuilder,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if rows.size == 0 or cols.size == 0:
            return
        mask = (cols >= 0) & (values != 0.0)
        if not np.any(mask):
            return
        cols_valid = cols[mask].astype(np.int32, copy=False)
        values_valid = values[mask].astype(np.float64, copy=False)
        if rows.size == 1:
            row_int = int(rows[0])
            H.rows.extend([row_int] * int(cols_valid.size))
            H.cols.extend(cols_valid.tolist())
            H.data.extend(values_valid.tolist())
            return
        H._append_arrays(
            np.repeat(rows.astype(np.int32, copy=False), cols_valid.size),
            np.tile(cols_valid, rows.size),
            np.tile(values_valid, rows.size),
        )

    @staticmethod
    def _append_sparse_rows_unchecked(
        H: SparseJacobianBuilder,
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if rows.size == 0 or cols.size == 0:
            return
        if rows.size == 1:
            row_int = int(rows[0])
            H.rows.extend([row_int] * int(cols.size))
            H.cols.extend(cols.astype(np.int32, copy=False).tolist())
            H.data.extend(values.astype(np.float64, copy=False).tolist())
            return
        H._append_arrays(
            np.repeat(rows.astype(np.int32, copy=False), cols.size),
            np.tile(cols.astype(np.int32, copy=False), rows.size),
            np.tile(values.astype(np.float64, copy=False), rows.size),
        )

    def _append_fast_ac_generator_jacobian(
        self,
        H: SparseJacobianBuilder,
        x: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        s_network: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        p_zero: np.ndarray,
        q_zero: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
        skip_rows: Optional[np.ndarray] = None,
    ) -> None:
        """Append AC generator P/Q/I rows while reusing each AC-node injection derivative.

        Multiple generators on the same AC node share the same nodal injection
        sensitivity. Their measurement rows differ only by the generator share,
        so the expensive Y-row derivative work is done once per unique node and
        then scaled into the individual generator rows.
        """
        if not self._jac_ac_generator_items:
            return
        node_pos = self._jac_ac_gen_node_pos.astype(np.int64, copy=False)
        if node_pos.size == 0:
            return

        entry_groups: List[np.ndarray] = []
        entry_cols: List[np.ndarray] = []
        entry_dP: List[np.ndarray] = []
        entry_dQ: List[np.ndarray] = []
        group_index = np.arange(node_pos.size, dtype=np.int32)

        def append_entries(groups: np.ndarray, cols: np.ndarray, dP: np.ndarray, dQ: np.ndarray) -> None:
            if groups.size == 0:
                return
            mask = cols >= 0
            if not np.any(mask):
                return
            entry_groups.append(groups[mask].astype(np.int32, copy=False))
            entry_cols.append(cols[mask].astype(np.int32, copy=False))
            entry_dP.append(dP[mask].astype(np.float64, copy=False))
            entry_dQ.append(dQ[mask].astype(np.float64, copy=False))

        y_group = self._jac_ac_gen_y_group
        if y_group.size:
            y_nodes = self._jac_ac_gen_y_nodes.astype(np.int64, copy=False)
            y_conj = self._jac_ac_gen_y_conj
            y_pos = node_pos[y_group]
            exp_delta = np.exp(1j * (theta[y_pos] - theta[y_nodes]))
            term = y_conj * voltage[y_pos] * voltage[y_nodes] * exp_delta

            off_mask = y_nodes != y_pos
            off_sum = np.zeros(node_pos.size, dtype=np.complex128)
            if np.any(off_mask):
                off_groups = y_group[off_mask]
                off_nodes = y_nodes[off_mask]
                off_term = term[off_mask]
                np.add.at(off_sum, off_groups, off_term)

                angle_cols = self.ac_theta_state_col[off_nodes]
                values = -1j * off_term
                append_entries(off_groups, angle_cols, values.real, values.imag)

                voltage_values = y_conj[off_mask] * voltage[y_pos[off_mask]] * exp_delta[off_mask]
                append_entries(
                    off_groups,
                    self.ac_voltage_state_col[off_nodes],
                    voltage_values.real,
                    voltage_values.imag,
                )

            sum_all = np.zeros(node_pos.size, dtype=np.complex128)
            np.add.at(sum_all, y_group, y_conj * voltage[y_nodes] * exp_delta)
        else:
            off_sum = np.zeros(node_pos.size, dtype=np.complex128)
            sum_all = np.zeros(node_pos.size, dtype=np.complex128)

        own_angle_cols = self.ac_theta_state_col[node_pos]
        own_angle_values = 1j * off_sum
        append_entries(group_index, own_angle_cols, own_angle_values.real, own_angle_values.imag)

        own_voltage_values = self.ac_y_row_diag_conj[node_pos] * voltage[node_pos] + sum_all
        append_entries(
            group_index,
            self.ac_voltage_state_col[node_pos],
            own_voltage_values.real,
            own_voltage_values.imag,
        )
        append_entries(
            group_index,
            self.ac_voltage_state_col[node_pos],
            p_load_dv[node_pos],
            q_load_dv[node_pos],
        )

        for group, pos in enumerate(node_pos):
            incidents = self.ac_zero_current_by_node.get(int(pos), [])
            if not incidents:
                continue
            theta_col = int(self.ac_theta_state_col[pos])
            voltage_col = int(self.ac_voltage_state_col[pos])
            for zero_idx, from_i in incidents:
                re_col = int(self.ac_zero_re_cols[zero_idx])
                im_col = int(self.ac_zero_im_cols[zero_idx])
                current = complex(x[re_col], x[im_col])
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
                cols = [voltage_col, re_col, im_col]
                dP = [float(np.real(dS_dV)), float(dS_dIr.real), float(dS_dIi.real)]
                dQ = [float(np.imag(dS_dV)), float(dS_dIr.imag), float(dS_dIi.imag)]
                if theta_col >= 0:
                    cols.insert(0, theta_col)
                    dP.insert(0, float(dS_dtheta.real))
                    dQ.insert(0, float(dS_dtheta.imag))
                append_entries(
                    np.full(len(cols), group, dtype=np.int32),
                    np.asarray(cols, dtype=np.int32),
                    np.asarray(dP, dtype=np.float64),
                    np.asarray(dQ, dtype=np.float64),
                )

        if entry_groups:
            entry_groups_array = np.concatenate(entry_groups)
            entry_cols_array = np.concatenate(entry_cols)
            entry_dP_array = np.concatenate(entry_dP)
            entry_dQ_array = np.concatenate(entry_dQ)
            order = np.argsort(entry_groups_array, kind="stable")
            entry_groups_array = entry_groups_array[order]
            entry_cols_array = entry_cols_array[order]
            entry_dP_array = entry_dP_array[order]
            entry_dQ_array = entry_dQ_array[order]
            group_starts = np.searchsorted(entry_groups_array, group_index, side="left")
            group_ends = np.searchsorted(entry_groups_array, group_index, side="right")
        else:
            entry_groups_array = np.array([], dtype=np.int32)
            entry_cols_array = np.array([], dtype=np.int32)
            entry_dP_array = np.array([], dtype=np.float64)
            entry_dQ_array = np.array([], dtype=np.float64)
            group_starts = np.zeros(node_pos.size, dtype=np.int64)
            group_ends = np.zeros(node_pos.size, dtype=np.int64)

        p_total = s_network.real[node_pos] + p_zero[node_pos] + p_load[node_pos]
        q_total = s_network.imag[node_pos] + q_zero[node_pos] + q_load[node_pos]
        if self._jac_ac_gen_single_rows:
            groups = entry_groups_array.astype(np.int32, copy=False)
            cols = entry_cols_array.astype(np.int32, copy=False)

            def append_single_rows(row_by_group: np.ndarray, values: np.ndarray) -> None:
                if groups.size == 0:
                    return
                rows = row_by_group[groups]
                mask = (rows >= 0) & (values != 0.0)
                if skip_rows is not None and np.any(mask):
                    active = np.nonzero(mask)[0]
                    mask[active] &= ~skip_rows[rows[active]]
                if np.any(mask):
                    H._append_arrays(rows[mask], cols[mask], values[mask])

            if groups.size:
                share = self._jac_ac_gen_single_share[groups]
                append_single_rows(self._jac_ac_gen_single_p_row, share * entry_dP_array)
                append_single_rows(self._jac_ac_gen_single_q_row, share * entry_dQ_array)

                i_rows = self._jac_ac_gen_single_i_row[groups]
                i_mask = i_rows >= 0
                if np.any(i_mask):
                    i_groups = groups[i_mask]
                    i_share = self._jac_ac_gen_single_share[i_groups]
                    p = i_share * p_total[i_groups]
                    q = i_share * q_total[i_groups]
                    s_abs = np.hypot(p, q)
                    v = voltage[node_pos[i_groups]]
                    valid = (s_abs > 1e-12) & (np.abs(v) > self.min_current_voltage)
                    if np.any(valid):
                        dP = i_share * entry_dP_array[i_mask]
                        dQ = i_share * entry_dQ_array[i_mask]
                        dI = np.zeros(i_groups.size, dtype=np.float64)
                        dI[valid] = (p[valid] * dP[valid] + q[valid] * dQ[valid]) / (s_abs[valid] * v[valid])
                        value_mask = valid & (dI != 0.0)
                        if np.any(value_mask):
                            H._append_arrays(
                                i_rows[i_mask][value_mask],
                                cols[i_mask][value_mask],
                                dI[value_mask],
                            )

            i_group_rows = self._jac_ac_gen_single_i_row
            i_groups = np.flatnonzero(i_group_rows >= 0).astype(np.int32, copy=False)
            if i_groups.size:
                if skip_rows is not None:
                    i_groups = i_groups[~skip_rows[i_group_rows[i_groups]]]
                if not i_groups.size:
                    return
                i_share = self._jac_ac_gen_single_share[i_groups]
                p = i_share * p_total[i_groups]
                q = i_share * q_total[i_groups]
                s_abs = np.hypot(p, q)
                v = voltage[node_pos[i_groups]]
                valid = (s_abs > 1e-12) & (np.abs(v) > self.min_current_voltage)
                if np.any(valid):
                    H.add_many(
                        i_group_rows[i_groups][valid],
                        self.ac_voltage_state_col[node_pos[i_groups]][valid],
                        -s_abs[valid] / (v[valid] * v[valid]),
                    )
            return

        append_rows = self._append_sparse_rows_unchecked

        def filter_skipped_rows(rows: np.ndarray) -> np.ndarray:
            if skip_rows is None or rows.size == 0:
                return rows
            return rows[~skip_rows[rows]]

        for group, share, p_rows, q_rows, i_rows in self._jac_ac_generator_items:
            group = int(group)
            pos = int(node_pos[group])
            voltage_col = int(self.ac_voltage_state_col[pos])
            start = int(group_starts[group])
            end = int(group_ends[group])
            cols = entry_cols_array[start:end]
            dP = share * entry_dP_array[start:end]
            dQ = share * entry_dQ_array[start:end]
            p_rows = filter_skipped_rows(p_rows)
            q_rows = filter_skipped_rows(q_rows)
            i_rows = filter_skipped_rows(i_rows)
            append_rows(H, p_rows, cols, dP)
            append_rows(H, q_rows, cols, dQ)
            if i_rows.size:
                p = share * p_total[group]
                q = share * q_total[group]
                s_abs = float(np.hypot(p, q))
                if s_abs > 1e-12 and abs(voltage[pos]) > self.min_current_voltage:
                    dI = (p * dP + q * dQ) / (s_abs * voltage[pos])
                    append_rows(H, i_rows, cols, dI)
                    extra = -s_abs / (voltage[pos] * voltage[pos])
                    H.add_many(
                        i_rows,
                        np.full(i_rows.size, voltage_col, dtype=np.int32),
                        np.full(i_rows.size, extra, dtype=np.float64),
                    )

    def _state_col_from_full(self, full_col: int) -> int:
        if 0 <= int(full_col) < self.full_to_state_col.size:
            return int(self.full_to_state_col[int(full_col)])
        return -1

    def _ac_theta_col(self, node_pos: int) -> int:
        pos = int(node_pos)
        if pos < 0 or pos >= self.ac_theta_state_col.size:
            return -1
        return int(self.ac_theta_state_col[pos])

    def _ac_voltage_col(self, node_pos: int) -> int:
        pos = int(node_pos)
        if pos < 0 or pos >= self.ac_voltage_state_col.size:
            return -1
        return int(self.ac_voltage_state_col[pos])

    def _dc_voltage_col(self, node_pos: int) -> int:
        pos = int(node_pos)
        if pos < 0 or pos >= self.dc_voltage_state_col.size:
            return -1
        return int(self.dc_voltage_state_col[pos])

    def _ac_branch_power_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex]:
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

    def _ac_branch_current_derivatives(
        self,
        own: int,
        other: int,
        y_self: complex,
        y_mutual: complex,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[complex, complex, complex, complex, complex]:
        v_own = voltage[own] * np.exp(1j * theta[own])
        v_other = voltage[other] * np.exp(1j * theta[other])
        current = y_self * v_own + y_mutual * v_other
        dtheta_own = 1j * y_self * v_own
        dtheta_other = 1j * y_mutual * v_other
        dvoltage_own = y_self * np.exp(1j * theta[own])
        dvoltage_other = y_mutual * np.exp(1j * theta[other])
        return current, dtheta_own, dtheta_other, dvoltage_own, dvoltage_other

    def _add_ac_power_derivatives(
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
        if meas_type[0] == "P":
            values = (
                dtheta_own.real,
                dtheta_other.real,
                dvoltage_own.real,
                dvoltage_other.real,
            )
        else:
            values = (
                dtheta_own.imag,
                dtheta_other.imag,
                dvoltage_own.imag,
                dvoltage_other.imag,
            )
        cols = (
            int(self.ac_theta_state_col[own]),
            int(self.ac_theta_state_col[other]),
            int(self.ac_voltage_state_col[own]),
            int(self.ac_voltage_state_col[other]),
        )
        if isinstance(H, SparseJacobianBuilder):
            row_int = int(row)
            rows = H.rows
            h_cols = H.cols
            data = H.data
            for col, value in zip(cols, values):
                if col >= 0 and value != 0.0:
                    rows.append(row_int)
                    h_cols.append(col)
                    data.append(float(value))
        else:
            self._add_derivatives(H, row, cols, values)

    def _add_ac_current_magnitude_derivatives(
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

        scale = np.conj(current) / current_abs
        cols = (
            int(self.ac_theta_state_col[own]),
            int(self.ac_theta_state_col[other]),
            int(self.ac_voltage_state_col[own]),
            int(self.ac_voltage_state_col[other]),
        )
        values = (
            (scale * dtheta_own).real,
            (scale * dtheta_other).real,
            (scale * dvoltage_own).real,
            (scale * dvoltage_other).real,
        )
        if isinstance(H, SparseJacobianBuilder):
            row_int = int(row)
            rows = H.rows
            h_cols = H.cols
            data = H.data
            for col, value in zip(cols, values):
                if col >= 0 and value != 0.0:
                    rows.append(row_int)
                    h_cols.append(col)
                    data.append(float(value))
        else:
            self._add_derivatives(H, row, cols, values)

    def _current_from_power_derivatives(
        self,
        p: float,
        q: float,
        voltage: float,
        dP: np.ndarray,
        dQ: np.ndarray,
        voltage_col: int,
    ) -> np.ndarray:
        dI = np.zeros_like(dP)
        s_abs = float(np.hypot(p, q))
        if abs(voltage) <= self.min_current_voltage or s_abs <= 1e-12:
            return dI
        dI += (p * dP + q * dQ) / (s_abs * voltage)
        if voltage_col >= 0:
            dI[voltage_col] -= s_abs / (voltage * voltage)
        return dI

    def _ac_network_power_derivatives(self, theta: np.ndarray, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ac = self.calc.ac_calc
        y_matrix = ac.Y.toarray() if hasattr(ac.Y, "toarray") else np.asarray(ac.Y)
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

    def _cached_ac_network_power_derivative_entries(
        self,
        pos: int,
        theta: np.ndarray,
        voltage: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse dS/dtheta and dS/dV entries for one AC nodal injection row."""
        node_cols = self.ac_y_row_nodes[pos]
        if node_cols.size == 0:
            nodes = np.asarray([pos], dtype=np.int32)
            zero = np.asarray([0.0j], dtype=np.complex128)
            return nodes, zero, nodes, zero

        y_conj = self.ac_y_row_y_conj[pos]
        off_mask = self.ac_y_row_off_mask[pos]
        off_nodes = self.ac_y_row_off_nodes[pos]
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
        voltage_values[-1] = self.ac_y_row_diag_conj[pos] * voltage[pos] + np.sum(
            y_conj * voltage[node_cols] * exp_delta
        )
        return theta_nodes, theta_values, voltage_nodes, voltage_values

    def _ac_load_power_arrays(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ac = self.calc.ac_calc
        p_load = np.zeros(ac.N, dtype=np.float64)
        q_load = np.zeros(ac.N, dtype=np.float64)
        if ac.load_pos.size:
            vm = voltage[ac.load_pos]
            p_vals = ac.load_pv0 + ac.load_pv1 * vm + ac.load_pv2 * vm * vm
            q_vals = ac.load_qv0 + ac.load_qv1 * vm + ac.load_qv2 * vm * vm
            np.add.at(p_load, ac.load_pos, p_vals)
            np.add.at(q_load, ac.load_pos, q_vals)
        return p_load, q_load

    def _ac_load_power_derivatives(self, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ac = self.calc.ac_calc
        dP = np.zeros(ac.N, dtype=np.float64)
        dQ = np.zeros(ac.N, dtype=np.float64)
        if ac.load_pos.size:
            vm = voltage[ac.load_pos]
            np.add.at(dP, ac.load_pos, ac.load_pv1 + 2.0 * vm * ac.load_pv2)
            np.add.at(dQ, ac.load_pos, ac.load_qv1 + 2.0 * vm * ac.load_qv2)
        return dP, dQ

    def _ac_zero_power_arrays(self, x: np.ndarray, voltage_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ac = self.calc.ac_calc
        p_zero = np.zeros(ac.N, dtype=np.float64)
        q_zero = np.zeros(ac.N, dtype=np.float64)
        if not self.ac_zero_current_cols_by_name:
            return p_zero, q_zero
        a = ac.zero_a.astype(np.int32, copy=False)
        b = ac.zero_b.astype(np.int32, copy=False)
        current = x[self.ac_zero_re_cols] + 1j * x[self.ac_zero_im_cols]
        s_a = voltage_complex[a] * np.conj(current)
        s_b = voltage_complex[b] * np.conj(-current)
        np.add.at(p_zero, a, s_a.real)
        np.add.at(q_zero, a, s_a.imag)
        np.add.at(p_zero, b, s_b.real)
        np.add.at(q_zero, b, s_b.imag)
        return p_zero, q_zero

    def _ac_generator_power_and_derivatives(
        self,
        gen,
        x: np.ndarray,
        theta: np.ndarray,
        voltage: np.ndarray,
        voltage_complex: np.ndarray,
        s_network: np.ndarray,
        p_load: np.ndarray,
        q_load: np.ndarray,
        p_zero: np.ndarray,
        q_zero: np.ndarray,
        p_load_dv: np.ndarray,
        q_load_dv: np.ndarray,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        ac = self.calc.ac_calc
        pos = ac.node_pos[gen.node]
        share = self.ac_gen_share_by_name.get(gen.name, 1.0)
        p = share * (s_network.real[pos] + p_zero[pos] + p_load[pos])
        q = share * (s_network.imag[pos] + q_zero[pos] + q_load[pos])

        cols_parts = []
        dP_parts = []
        dQ_parts = []
        theta_nodes, theta_values, voltage_nodes, voltage_values = self._cached_ac_network_power_derivative_entries(
            pos,
            theta,
            voltage,
        )
        theta_cols = self.ac_theta_state_col[theta_nodes]
        theta_mask = theta_cols >= 0
        if np.any(theta_mask):
            cols_parts.append(theta_cols[theta_mask])
            dP_parts.append(share * theta_values[theta_mask].real)
            dQ_parts.append(share * theta_values[theta_mask].imag)
        voltage_cols = self.ac_voltage_state_col[voltage_nodes]
        voltage_mask = voltage_cols >= 0
        if np.any(voltage_mask):
            cols_parts.append(voltage_cols[voltage_mask])
            dP_parts.append(share * voltage_values[voltage_mask].real)
            dQ_parts.append(share * voltage_values[voltage_mask].imag)
        voltage_col = int(self.ac_voltage_state_col[pos])
        if voltage_col >= 0:
            cols_parts.append(np.asarray([voltage_col], dtype=np.int32))
            dP_parts.append(np.asarray([share * p_load_dv[pos]], dtype=np.float64))
            dQ_parts.append(np.asarray([share * q_load_dv[pos]], dtype=np.float64))

        theta_col = int(self.ac_theta_state_col[pos])
        for row, from_i in self.ac_zero_current_by_node.get(pos, []):
            re_col = int(self.ac_zero_re_cols[row])
            im_col = int(self.ac_zero_im_cols[row])
            current = complex(x[re_col], x[im_col])
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
            local_cols = []
            local_dP = []
            local_dQ = []
            if theta_col >= 0:
                local_cols.append(theta_col)
                local_dP.append(share * dS_dtheta.real)
                local_dQ.append(share * dS_dtheta.imag)
            if voltage_col >= 0:
                local_cols.append(voltage_col)
                local_dP.append(share * np.real(dS_dV))
                local_dQ.append(share * np.imag(dS_dV))
            local_cols.extend((re_col, im_col))
            local_dP.extend((share * dS_dIr.real, share * dS_dIi.real))
            local_dQ.extend((share * dS_dIr.imag, share * dS_dIi.imag))
            cols_parts.append(np.asarray(local_cols, dtype=np.int32))
            dP_parts.append(np.asarray(local_dP, dtype=np.float64))
            dQ_parts.append(np.asarray(local_dQ, dtype=np.float64))

        if cols_parts:
            cols = np.concatenate(cols_parts).astype(np.int32, copy=False)
            dP = np.concatenate(dP_parts).astype(np.float64, copy=False)
            dQ = np.concatenate(dQ_parts).astype(np.float64, copy=False)
        else:
            cols = np.array([], dtype=np.int32)
            dP = np.array([], dtype=np.float64)
            dQ = np.array([], dtype=np.float64)
        return float(p), float(q), cols, dP, dQ

    @staticmethod
    def _value_from_terminal(
        meas_type: str,
        p_from: float,
        q_from: float,
        v_from: float,
        i_from: float,
        p_to: float,
        q_to: float,
        v_to: float,
        i_to: float,
    ) -> float:
        """Pick the requested P/Q/V/I terminal value from a two-terminal device tuple."""
        if meas_type == "P_FROM":
            return float(p_from)
        if meas_type == "Q_FROM":
            return float(q_from)
        if meas_type == "V_FROM":
            return float(v_from)
        if meas_type == "I_FROM":
            return float(i_from)
        if meas_type == "P_TO":
            return float(p_to)
        if meas_type == "Q_TO":
            return float(q_to)
        if meas_type == "V_TO":
            return float(v_to)
        if meas_type == "I_TO":
            return float(i_to)
        raise RuntimeError(f"Unsupported terminal measurement type: {meas_type}")

    def _ac_line_value(self, device, meas_type: str) -> float:
        if meas_type == "P_FROM":
            return float(device.i_p)
        if meas_type == "Q_FROM":
            return float(device.i_q)
        if meas_type == "V_FROM":
            return float(device.i_node_obj.voltage)
        if meas_type == "I_FROM":
            return float(device.i_c)
        if meas_type == "P_TO":
            return float(device.j_p)
        if meas_type == "Q_TO":
            return float(device.j_q)
        if meas_type == "V_TO":
            return float(device.j_node_obj.voltage)
        if meas_type == "I_TO":
            return float(device.j_c)
        raise RuntimeError(f"Unsupported AC terminal measurement type: {meas_type}")

    def _ac_zero_value(self, device, meas_type: str) -> float:
        p_from = float(getattr(device, "p", 0.0) or 0.0)
        q_from = float(getattr(device, "q", 0.0) or 0.0)
        if meas_type == "P_FROM":
            return p_from
        if meas_type == "Q_FROM":
            return q_from
        if meas_type == "V_FROM":
            return float(device.i_node_obj.voltage)
        if meas_type == "I_FROM":
            return abs(float(getattr(device, "current", 0.0) or 0.0))
        if meas_type == "P_TO":
            return -p_from
        if meas_type == "Q_TO":
            return -q_from
        if meas_type == "V_TO":
            return float(device.j_node_obj.voltage)
        if meas_type == "I_TO":
            return abs(float(getattr(device, "current", 0.0) or 0.0))
        raise RuntimeError(f"Unsupported AC zero-branch measurement type: {meas_type}")

    def _dc_line_value(self, device, meas_type: str) -> float:
        current = float(getattr(device, "current", 0.0) or 0.0)
        if meas_type == "P_FROM":
            return float(device.i_p)
        if meas_type == "V_FROM":
            return float(device.i_node_obj.voltage)
        if meas_type == "I_FROM":
            return current
        if meas_type == "P_TO":
            return float(device.j_p)
        if meas_type == "V_TO":
            return float(device.j_node_obj.voltage)
        if meas_type == "I_TO":
            return -current
        raise RuntimeError(f"Unsupported DC terminal measurement type: {meas_type}")

    def _dc_zero_value(self, device, meas_type: str) -> float:
        current = float(getattr(device, "current", 0.0) or 0.0)
        p_from = float(getattr(device, "p", 0.0) or 0.0)
        if meas_type == "P_FROM":
            return p_from
        if meas_type == "V_FROM":
            return float(device.i_node_obj.voltage)
        if meas_type == "I_FROM":
            return current
        if meas_type == "P_TO":
            return -float(device.j_node_obj.voltage) * current
        if meas_type == "V_TO":
            return float(device.j_node_obj.voltage)
        if meas_type == "I_TO":
            return -current
        raise RuntimeError(f"Unsupported DC zero-branch measurement type: {meas_type}")

    def _dcac_value(self, conv, meas_type: str) -> float:
        """Evaluate measurements on a DC/AC converter after the hybrid state is written back."""
        ac_v = float(conv.ac_node_obj.voltage)
        dc_v = float(conv.dc_node_obj.voltage)
        if meas_type == "P_DC":
            return float(conv.dc_p)
        if meas_type == "V_DC":
            return dc_v
        if meas_type == "I_DC":
            return float(conv.dc_i)
        if meas_type == "P_AC":
            return float(conv.ac_p)
        if meas_type == "Q_AC":
            return float(conv.ac_q)
        if meas_type == "V_AC":
            return ac_v
        if meas_type == "I_AC":
            return float(conv.ac_i)
        raise RuntimeError(f"Unsupported DCACConverter measurement type: {meas_type}")

    def _acac_value(self, conv, meas_type: str) -> float:
        if meas_type == "P_FROM":
            return float(conv.i_p)
        if meas_type == "Q_FROM":
            return float(conv.i_q)
        if meas_type == "V_FROM":
            return float(conv.i_node_obj.voltage)
        if meas_type == "I_FROM":
            return float(conv.i_i)
        if meas_type == "P_TO":
            return float(conv.j_p)
        if meas_type == "Q_TO":
            return float(conv.j_q)
        if meas_type == "V_TO":
            return float(conv.j_node_obj.voltage)
        if meas_type == "I_TO":
            return float(conv.j_i)
        raise RuntimeError(f"Unsupported ACACConverter measurement type: {meas_type}")

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Evaluate h(x) for AC, DC and converter measurements in one pass."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.evaluate(x, measurements)
        if measurements is None:
            measurements = self.active_measurements
        elif measurements is not self.active_measurements:
            measurements = list(measurements)
            if self._matches_active_measurements(measurements):
                measurements = self.active_measurements
        if measurements is self.active_measurements:
            return self._evaluate_active_measurements(x)
        self._write_state(x)
        values = np.zeros(len(measurements), dtype=np.float64)

        for row, meas in enumerate(measurements):
            mtype = meas.meas_type
            if meas.device_type == "ACNode":
                node = self.ac_node_by_name[meas.device_name]
                if mtype == "V":
                    values[row] = node.voltage
                elif mtype in ("ANGLE", "THETA"):
                    values[row] = node.angle
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")
            elif meas.device_type == "DCNode":
                node = self.dc_node_by_name[meas.device_name]
                if mtype != "V":
                    raise RuntimeError(f"Unsupported DCNode measurement type: {mtype}")
                values[row] = node.voltage
            elif meas.device_type == "ACBranch":
                values[row] = self._ac_line_value(self.ac_branch_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACTransformer":
                values[row] = self._ac_line_value(self.ac_transformer_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACSwitch":
                values[row] = self._ac_zero_value(self.ac_switch_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACBreak":
                values[row] = self._ac_zero_value(self.ac_break_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACZeroBranch":
                values[row] = self._ac_zero_value(self.ac_zero_branch_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACGenerator":
                gen = self.ac_generator_by_name[meas.device_name]
                if mtype == "P_GEN":
                    values[row] = gen.p
                elif mtype == "Q_GEN":
                    values[row] = gen.q
                elif mtype == "V_GEN":
                    values[row] = gen.node_obj.voltage
                elif mtype == "I_GEN":
                    values[row] = gen.current
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")
            elif meas.device_type == "ACLoad":
                load = self.ac_load_by_name[meas.device_name]
                if mtype == "P_LOAD":
                    values[row] = load.p
                elif mtype == "Q_LOAD":
                    values[row] = load.q
                elif mtype == "V_LOAD":
                    values[row] = load.node_obj.voltage
                elif mtype == "I_LOAD":
                    values[row] = load.current
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")
            elif meas.device_type == "DCBranch":
                values[row] = self._dc_line_value(self.dc_branch_by_name[meas.device_name], mtype)
            elif meas.device_type == "DCZeroBranchConstraint":
                zbr = self.dc_zero_branch_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCZeroBranchConstraint measurement type: {mtype}")
                values[row] = zbr.i_node_obj.voltage - zbr.j_node_obj.voltage
            elif meas.device_type == "DCSwitchConstraint":
                sw = self.dc_switch_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCSwitchConstraint measurement type: {mtype}")
                values[row] = sw.i_node_obj.voltage - sw.j_node_obj.voltage
            elif meas.device_type == "DCBreakConstraint":
                brk = self.dc_break_by_name[meas.device_name]
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported DCBreakConstraint measurement type: {mtype}")
                values[row] = brk.i_node_obj.voltage - brk.j_node_obj.voltage
            elif meas.device_type == "DCSwitch":
                values[row] = self._dc_zero_value(self.dc_switch_by_name[meas.device_name], mtype)
            elif meas.device_type == "DCBreak":
                values[row] = self._dc_zero_value(self.dc_break_by_name[meas.device_name], mtype)
            elif meas.device_type == "DCZeroBranch":
                values[row] = self._dc_zero_value(self.dc_zero_branch_by_name[meas.device_name], mtype)
            elif meas.device_type == "DCGenerator":
                gen = self.dc_generator_by_name[meas.device_name]
                if mtype == "P_GEN":
                    values[row] = gen.p
                elif mtype == "V_GEN":
                    values[row] = gen.node_obj.voltage
                elif mtype == "I_GEN":
                    values[row] = gen.current
                else:
                    raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
            elif meas.device_type == "DCLoad":
                load = self.dc_load_by_name[meas.device_name]
                if mtype == "P_LOAD":
                    values[row] = load.p
                elif mtype == "V_LOAD":
                    values[row] = load.node_obj.voltage
                elif mtype == "I_LOAD":
                    values[row] = load.current
                else:
                    raise RuntimeError(f"Unsupported DCLoad measurement type: {mtype}")
            elif meas.device_type == "DCDCConverter":
                conv = self.dcdc_by_name[meas.device_name]
                values[row] = self._value_from_terminal(
                    mtype,
                    float(conv.i_p),
                    0.0,
                    float(conv.i_node_obj.voltage),
                    float(conv.i_c),
                    float(conv.j_p),
                    0.0,
                    float(conv.j_node_obj.voltage),
                    float(conv.j_c),
                )
            elif meas.device_type == "DCACConverter":
                values[row] = self._dcac_value(self.dcac_by_name[meas.device_name], mtype)
            elif meas.device_type == "ACACConverter":
                values[row] = self._acac_value(self.acac_by_name[meas.device_name], mtype)
            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")
        return values

    def _assemble_jacobian(
        self,
        x: np.ndarray,
        measurements: Optional[Sequence[Measurement]] = None,
        sparse: bool = False,
    ):
        """Build H analytically over the compact hybrid state."""
        if measurements is None:
            measurements = self.active_measurements
        elif measurements is not self.active_measurements:
            measurements = list(measurements)
            if self._matches_active_measurements(measurements):
                measurements = self.active_measurements
        x = np.asarray(x, dtype=np.float64)
        active_sparse = sparse and measurements is self.active_measurements
        full_x = self._expand_state_mapped_only(x) if active_sparse else self._expand_state(x)
        ac = self.calc.ac_calc
        dc = self.calc.dc_calc

        ac_theta = ac_V = ac_Vc = None
        ac_s_network = None
        ac_p_load = ac_q_load = None
        ac_p_zero = ac_q_zero = None
        ac_p_load_dv = ac_q_load_dv = None
        ac_gen_cache = {}
        ac_branch_power_derivative_cache = {}
        ac_branch_current_derivative_cache = {}
        if ac is not None:
            ac_x = full_x[: self.calc.ac_size]
            ac_theta, ac_V, _, _ = ac._extract_state_vars(ac_x, update_cache=True)
            ac_Vc = ac._cache["Vc"]
            ac_p_load, ac_q_load = self._ac_load_power_arrays(ac_V)
            ac_p_zero, ac_q_zero = self._ac_zero_power_arrays(x, ac_Vc)

        dc_x = dc_V = dcdc_power = None
        if dc is not None:
            dc_x = full_x[self.calc.ac_size : self.calc.ac_size + self.calc.dc_size]
            dc_V = dc_x[: dc.N]
            dcdc_start = dc.N + dc.N_phi
            dcdc_power = dc_x[dcdc_start : dcdc_start + 2 * dc.N_dcdc] if dc.N_dcdc else np.array([])

        H = (
            SparseJacobianBuilder((len(measurements), self.n_state))
            if sparse
            else np.zeros((len(measurements), self.n_state), dtype=np.float64)
        )
        ac_delegated_mask = (
            self._active_ac_delegated_row_mask
            if active_sparse and getattr(self, "_active_ac_hybrid_rows", np.array([], dtype=np.int32)).size
            else None
        )
        dc_delegated_mask = (
            self._active_dc_delegated_row_mask
            if active_sparse and getattr(self, "_active_dc_hybrid_rows", np.array([], dtype=np.int32)).size
            else None
        )
        static_skip = None
        if active_sparse:
            if ac_delegated_mask is not None or dc_delegated_mask is not None:
                static_skip = self._jacobian_static_skip.copy()
            else:
                static_skip = self._jacobian_static_skip
            if ac_delegated_mask is not None:
                static_skip[ac_delegated_mask] = True
            if dc_delegated_mask is not None:
                static_skip[dc_delegated_mask] = True
            if self._jacobian_static_rows.size:
                delegated_mask = None
                if ac_delegated_mask is not None and dc_delegated_mask is not None:
                    delegated_mask = ac_delegated_mask | dc_delegated_mask
                elif ac_delegated_mask is not None:
                    delegated_mask = ac_delegated_mask
                elif dc_delegated_mask is not None:
                    delegated_mask = dc_delegated_mask
                if delegated_mask is None:
                    H._append_arrays(
                        self._jacobian_static_rows,
                        self._jacobian_static_cols,
                        self._jacobian_static_data,
                    )
                else:
                    keep_static = ~delegated_mask[self._jacobian_static_rows]
                    H._append_arrays(
                        self._jacobian_static_rows[keep_static],
                        self._jacobian_static_cols[keep_static],
                        self._jacobian_static_data[keep_static],
                    )
            dynamic_delegated_mask = dc_delegated_mask
            if ac_delegated_mask is not None and dc_delegated_mask is not None:
                dynamic_delegated_mask = ac_delegated_mask | dc_delegated_mask
            elif ac_delegated_mask is not None:
                dynamic_delegated_mask = ac_delegated_mask
            self._append_fast_dynamic_jacobian(H, x, ac_theta, ac_V, ac_Vc, dc_V, dynamic_delegated_mask)
            if ac_delegated_mask is not None:
                self._append_ac_sub_jacobian(H, x)
            if dc_delegated_mask is not None:
                self._append_dc_sub_jacobian(H, x)
            if ac is not None and self._jac_ac_generator_items:
                if ac_p_load_dv is None or ac_q_load_dv is None:
                    ac_p_load_dv, ac_q_load_dv = self._ac_load_power_derivatives(ac_V)
                if ac_s_network is None:
                    ac_s_network = ac_Vc * np.conj(ac.Y.dot(ac_Vc))
                self._append_fast_ac_generator_jacobian(
                    H,
                    x,
                    ac_theta,
                    ac_V,
                    ac_Vc,
                    ac_s_network,
                    ac_p_load,
                    ac_q_load,
                    ac_p_zero,
                    ac_q_zero,
                    ac_p_load_dv,
                    ac_q_load_dv,
                    skip_rows=ac_delegated_mask,
                )

        row_iter = (
            self._jacobian_dynamic_rows
            if static_skip is not None
            else range(len(measurements))
        )
        for row_raw in row_iter:
            row = int(row_raw)
            meas = measurements[row]
            if static_skip is not None and static_skip[row]:
                continue
            mtype = meas.meas_type

            if meas.device_type == "ACNode":
                node = self.ac_node_by_name[meas.device_name]
                pos = ac.node_pos[node.idx]
                if mtype == "V":
                    self._add_derivative(H, row, int(self.ac_voltage_state_col[pos]), 1.0)
                elif mtype in ("ANGLE", "THETA"):
                    self._add_derivative(H, row, int(self.ac_theta_state_col[pos]), 1.0)
                else:
                    raise RuntimeError(f"Unsupported ACNode measurement type: {mtype}")

            elif meas.device_type in ("ACBranch", "ACTransformer"):
                is_transformer = meas.device_type == "ACTransformer"
                device = (
                    self.ac_transformer_by_name[meas.device_name]
                    if is_transformer
                    else self.ac_branch_by_name[meas.device_name]
                )
                i = ac.node_pos[device.i_node]
                j = ac.node_pos[device.j_node]
                yff, yft, ytf, ytt = (
                    self.ac_transformer_stamp_by_name[device.name]
                    if is_transformer
                    else self.ac_branch_stamp_by_name[device.name]
                )
                if mtype in ("P_FROM", "Q_FROM"):
                    cache_key = (meas.device_type, device.name, "from")
                    if cache_key not in ac_branch_power_derivative_cache:
                        ac_branch_power_derivative_cache[cache_key] = self._ac_branch_power_derivatives(
                            i, j, yff, yft, ac_theta, ac_V
                        )
                    derivatives = ac_branch_power_derivative_cache[cache_key]
                    self._add_ac_power_derivatives(H, row, mtype, i, j, *derivatives)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, int(self.ac_voltage_state_col[i]), 1.0)
                elif mtype == "I_FROM":
                    cache_key = (meas.device_type, device.name, "from")
                    if cache_key not in ac_branch_current_derivative_cache:
                        ac_branch_current_derivative_cache[cache_key] = self._ac_branch_current_derivatives(
                            i, j, yff, yft, ac_theta, ac_V
                        )
                    derivatives = ac_branch_current_derivative_cache[cache_key]
                    self._add_ac_current_magnitude_derivatives(H, row, i, j, *derivatives)
                elif mtype in ("P_TO", "Q_TO"):
                    cache_key = (meas.device_type, device.name, "to")
                    if cache_key not in ac_branch_power_derivative_cache:
                        ac_branch_power_derivative_cache[cache_key] = self._ac_branch_power_derivatives(
                            j, i, ytt, ytf, ac_theta, ac_V
                        )
                    derivatives = ac_branch_power_derivative_cache[cache_key]
                    self._add_ac_power_derivatives(H, row, mtype, j, i, *derivatives)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, int(self.ac_voltage_state_col[j]), 1.0)
                elif mtype == "I_TO":
                    cache_key = (meas.device_type, device.name, "to")
                    if cache_key not in ac_branch_current_derivative_cache:
                        ac_branch_current_derivative_cache[cache_key] = self._ac_branch_current_derivatives(
                            j, i, ytt, ytf, ac_theta, ac_V
                        )
                    derivatives = ac_branch_current_derivative_cache[cache_key]
                    self._add_ac_current_magnitude_derivatives(H, row, j, i, *derivatives)
                else:
                    raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")

            elif meas.device_type in ("ACSwitch", "ACZeroBranch", "ACBreak"):
                device = (
                    self.ac_switch_by_name[meas.device_name]
                    if meas.device_type == "ACSwitch"
                    else self.ac_break_by_name[meas.device_name]
                    if meas.device_type == "ACBreak"
                    else self.ac_zero_branch_by_name[meas.device_name]
                )
                i = ac.node_pos[device.i_node]
                j = ac.node_pos[device.j_node]
                if mtype == "V_FROM":
                    self._add_derivative(H, row, int(self.ac_voltage_state_col[i]), 1.0)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, int(self.ac_voltage_state_col[j]), 1.0)
                elif device.name not in self.ac_zero_current_cols_by_name:
                    if mtype not in ("I_FROM", "I_TO", "P_FROM", "Q_FROM", "P_TO", "Q_TO"):
                        raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")
                    # Redundant zero-impedance ties between equal slack phasors have no
                    # explicit current state; their P/Q/I estimate and derivatives are zero.
                    continue
                elif mtype in ("I_FROM", "I_TO", "P_FROM", "Q_FROM", "P_TO", "Q_TO"):
                    re_col, im_col = self.ac_zero_current_cols_by_name[device.name]
                    current = complex(x[re_col], x[im_col])
                    current_abs = abs(current)
                    if mtype in ("I_FROM", "I_TO"):
                        if current_abs > 1e-12:
                            self._add_derivative(H, row, re_col, current.real / current_abs)
                            self._add_derivative(H, row, im_col, current.imag / current_abs)
                    else:
                        sign = -1.0 if mtype.endswith("_TO") else 1.0
                        s = ac_Vc[i] * np.conj(current)
                        pick = np.real if mtype.startswith("P") else np.imag
                        theta_col = int(self.ac_theta_state_col[i])
                        voltage_col = int(self.ac_voltage_state_col[i])
                        dS_dtheta = 1j * s
                        dS_dV = s / ac_V[i] if abs(ac_V[i]) > 1e-12 else 0.0
                        self._add_derivative(H, row, theta_col, sign * float(pick(dS_dtheta)))
                        self._add_derivative(H, row, voltage_col, sign * float(pick(dS_dV)))
                        self._add_derivative(H, row, re_col, sign * float(pick(ac_Vc[i])))
                        self._add_derivative(H, row, im_col, sign * float(pick(-1j * ac_Vc[i])))
                else:
                    raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")

            elif meas.device_type == "ACLoad":
                load = self.ac_load_by_name[meas.device_name]
                pos = ac.node_pos[load.node]
                v = ac_V[pos]
                voltage_col = int(self.ac_voltage_state_col[pos])
                p_col = self.ac_load_p_col_by_name[load.name]
                q_col = self.ac_load_q_col_by_name[load.name]
                p = float(x[p_col])
                q = float(x[q_col])
                if mtype == "P_LOAD":
                    self._add_derivative(H, row, p_col, 1.0)
                elif mtype == "Q_LOAD":
                    self._add_derivative(H, row, q_col, 1.0)
                elif mtype == "V_LOAD":
                    self._add_derivative(H, row, voltage_col, 1.0)
                elif mtype == "I_LOAD":
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(v) > self.min_current_voltage:
                        self._add_derivative(H, row, p_col, p / (s_abs * v))
                        self._add_derivative(H, row, q_col, q / (s_abs * v))
                        self._add_derivative(H, row, voltage_col, -s_abs / (v * v))
                else:
                    raise RuntimeError(f"Unsupported ACLoad measurement type: {mtype}")

            elif meas.device_type == "ACGenerator":
                gen = self.ac_generator_by_name[meas.device_name]
                pos = ac.node_pos[gen.node]
                voltage_col = int(self.ac_voltage_state_col[pos])
                if mtype == "V_GEN":
                    self._add_derivative(H, row, voltage_col, 1.0)
                    continue
                p_col = self.ac_generator_p_col_by_name[gen.name]
                q_col = self.ac_generator_q_col_by_name[gen.name]
                p = float(x[p_col])
                q = float(x[q_col])
                if mtype == "P_GEN":
                    self._add_derivative(H, row, p_col, 1.0)
                elif mtype == "Q_GEN":
                    self._add_derivative(H, row, q_col, 1.0)
                elif mtype == "I_GEN":
                    s_abs = float(np.hypot(p, q))
                    if s_abs > 1e-12 and abs(ac_V[pos]) > self.min_current_voltage:
                        scale = 1.0 / (s_abs * ac_V[pos])
                        self._add_derivative(H, row, p_col, p * scale)
                        self._add_derivative(H, row, q_col, q * scale)
                        self._add_derivative(H, row, voltage_col, -s_abs / (ac_V[pos] * ac_V[pos]))
                else:
                    raise RuntimeError(f"Unsupported ACGenerator measurement type: {mtype}")

            elif meas.device_type == "DCNode":
                node = self.dc_node_by_name[meas.device_name]
                if mtype != "V":
                    raise RuntimeError(f"Unsupported DCNode measurement type: {mtype}")
                self._add_derivative(H, row, int(self.dc_voltage_state_col[dc.alive_node_dict[node.idx]]), 1.0)

            elif meas.device_type in ("DCZeroBranchConstraint", "DCSwitchConstraint", "DCBreakConstraint"):
                if mtype != "V_DIFF":
                    raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")
                device = (
                    self.dc_zero_branch_by_name[meas.device_name]
                    if meas.device_type == "DCZeroBranchConstraint"
                    else self.dc_break_by_name[meas.device_name]
                    if meas.device_type == "DCBreakConstraint"
                    else self.dc_switch_by_name[meas.device_name]
                )
                i = dc.alive_node_dict[device.i_node]
                j = dc.alive_node_dict[device.j_node]
                self._add_derivative(H, row, int(self.dc_voltage_state_col[i]), 1.0)
                self._add_derivative(H, row, int(self.dc_voltage_state_col[j]), -1.0)

            elif meas.device_type == "DCBranch":
                br = self.dc_branch_by_name[meas.device_name]
                i = dc.alive_node_dict[br.i_node]
                j = dc.alive_node_dict[br.j_node]
                vi = dc_V[i]
                vj = dc_V[j]
                inv_r = 1.0 / br.r
                i_col = int(self.dc_voltage_state_col[i])
                j_col = int(self.dc_voltage_state_col[j])
                if mtype == "P_FROM":
                    self._add_derivative(H, row, i_col, (2.0 * vi - vj) * inv_r)
                    self._add_derivative(H, row, j_col, -vi * inv_r)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_col, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, i_col, inv_r)
                    self._add_derivative(H, row, j_col, -inv_r)
                elif mtype == "P_TO":
                    self._add_derivative(H, row, i_col, -vj * inv_r)
                    self._add_derivative(H, row, j_col, (-vi + 2.0 * vj) * inv_r)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_col, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, i_col, -inv_r)
                    self._add_derivative(H, row, j_col, inv_r)
                else:
                    raise RuntimeError(f"Unsupported DCBranch measurement type: {mtype}")

            elif meas.device_type in ("DCSwitch", "DCZeroBranch", "DCBreak"):
                device = (
                    self.dc_switch_by_name[meas.device_name]
                    if meas.device_type == "DCSwitch"
                    else self.dc_break_by_name[meas.device_name]
                    if meas.device_type == "DCBreak"
                    else self.dc_zero_branch_by_name[meas.device_name]
                )
                i = dc.alive_node_dict[device.i_node]
                j = dc.alive_node_dict[device.j_node]
                i_col = int(self.dc_voltage_state_col[i])
                j_col = int(self.dc_voltage_state_col[j])
                if mtype == "P_FROM":
                    if device.name not in self.dc_zero_current_col_by_name:
                        continue
                    current_col = self.dc_zero_current_col_by_name[device.name]
                    current = x[current_col]
                    self._add_derivative(H, row, i_col, current)
                    self._add_derivative(H, row, current_col, dc_V[i])
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_col, 1.0)
                elif mtype == "I_FROM":
                    if device.name not in self.dc_zero_current_col_by_name:
                        continue
                    current_col = self.dc_zero_current_col_by_name[device.name]
                    self._add_derivative(H, row, current_col, 1.0)
                elif mtype == "P_TO":
                    if device.name not in self.dc_zero_current_col_by_name:
                        continue
                    current_col = self.dc_zero_current_col_by_name[device.name]
                    current = x[current_col]
                    self._add_derivative(H, row, j_col, -current)
                    self._add_derivative(H, row, current_col, -dc_V[j])
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_col, 1.0)
                elif mtype == "I_TO":
                    if device.name not in self.dc_zero_current_col_by_name:
                        continue
                    current_col = self.dc_zero_current_col_by_name[device.name]
                    self._add_derivative(H, row, current_col, -1.0)
                else:
                    raise RuntimeError(f"Unsupported {meas.device_type} measurement type: {mtype}")

            elif meas.device_type == "DCLoad":
                load = self.dc_load_by_name[meas.device_name]
                pos = dc.alive_node_dict[load.node]
                v = dc_V[pos]
                voltage_col = int(self.dc_voltage_state_col[pos])
                p0, p1, p2, _, _, _ = self._load_zip_coefficients(load)
                if mtype == "P_LOAD":
                    self._add_derivative(H, row, voltage_col, p1 + 2.0 * p2 * v)
                elif mtype == "V_LOAD":
                    self._add_derivative(H, row, voltage_col, 1.0)
                elif mtype == "I_LOAD":
                    self._add_derivative(H, row, voltage_col, p2 - p0 / (v * v))
                else:
                    raise RuntimeError(f"Unsupported DCLoad measurement type: {mtype}")

            elif meas.device_type == "DCGenerator":
                gen = self.dc_generator_by_name[meas.device_name]
                pos = dc.alive_node_dict[gen.node]
                v = dc_V[pos]
                voltage_col = int(self.dc_voltage_state_col[pos])
                control_type = str(gen.control_type).upper()
                if control_type == "V":
                    p_col = self.dc_v_generator_col_by_name[gen.name]
                    p = x[p_col]
                    if mtype == "P_GEN":
                        self._add_derivative(H, row, p_col, 1.0)
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, voltage_col, 1.0)
                    elif mtype == "I_GEN":
                        self._add_derivative(H, row, p_col, 1.0 / v)
                        self._add_derivative(H, row, voltage_col, -p / (v * v))
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                elif control_type == "P":
                    if mtype == "P_GEN":
                        pass
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, voltage_col, 1.0)
                    elif mtype == "I_GEN":
                        self._add_derivative(H, row, voltage_col, -gen.p_set / (v * v))
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                elif control_type == "I":
                    if mtype == "P_GEN":
                        self._add_derivative(H, row, voltage_col, gen.i_set)
                    elif mtype == "V_GEN":
                        self._add_derivative(H, row, voltage_col, 1.0)
                    elif mtype == "I_GEN":
                        pass
                    else:
                        raise RuntimeError(f"Unsupported DCGenerator measurement type: {mtype}")
                else:
                    raise RuntimeError(f"Unsupported DCGenerator control type: {gen.control_type}")

            elif meas.device_type == "DCDCConverter":
                conv = self.dcdc_by_name[meas.device_name]
                d_idx = self.dcdc_pos_by_name[conv.name]
                i = dc.alive_node_dict[conv.i_node]
                j = dc.alive_node_dict[conv.j_node]
                p_from_col = int(self.dcdc_p_from_state_col[d_idx])
                p_to_col = int(self.dcdc_p_to_state_col[d_idx])
                i_col = int(self.dc_voltage_state_col[i])
                j_col = int(self.dc_voltage_state_col[j])
                p_from = dcdc_power[2 * d_idx]
                p_to = dcdc_power[2 * d_idx + 1]
                v_from = dc_V[i]
                v_to = dc_V[j]
                if mtype == "P_FROM":
                    self._add_derivative(H, row, p_from_col, 1.0)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_col, 1.0)
                elif mtype == "I_FROM":
                    self._add_derivative(H, row, p_from_col, 1.0 / v_from)
                    self._add_derivative(H, row, i_col, -p_from / (v_from * v_from))
                elif mtype == "P_TO":
                    self._add_derivative(H, row, p_to_col, 1.0)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_col, 1.0)
                elif mtype == "I_TO":
                    self._add_derivative(H, row, p_to_col, 1.0 / v_to)
                    self._add_derivative(H, row, j_col, -p_to / (v_to * v_to))
                else:
                    raise RuntimeError(f"Unsupported DCDCConverter measurement type: {mtype}")

            elif meas.device_type == "DCACConverter":
                conv = self.dcac_by_name[meas.device_name]
                k = self.dcac_pos_by_name[conv.name]
                _, ac_pos, dc_pos, _ = self.calc.dcac_converters[k]
                dc_p_col = int(self.dcac_p_dc_state_col[k])
                ac_p_col = int(self.dcac_p_ac_state_col[k])
                ac_q_col = int(self.dcac_q_ac_state_col[k])
                ac_v_col = int(self.ac_voltage_state_col[ac_pos])
                dc_v_col = int(self.dc_voltage_state_col[dc_pos])
                dc_p, ac_p, ac_q = full_x[self.calc.dcac_start + 3 * k : self.calc.dcac_start + 3 * k + 3]
                ac_v = ac_V[ac_pos]
                dc_v = dc_V[dc_pos]
                if mtype == "P_DC":
                    self._add_derivative(H, row, dc_p_col, 1.0)
                elif mtype == "V_DC":
                    self._add_derivative(H, row, dc_v_col, 1.0)
                elif mtype == "I_DC":
                    self._add_derivative(H, row, dc_p_col, 1.0 / dc_v)
                    self._add_derivative(H, row, dc_v_col, -dc_p / (dc_v * dc_v))
                elif mtype == "P_AC":
                    self._add_derivative(H, row, ac_p_col, 1.0)
                elif mtype == "Q_AC":
                    self._add_derivative(H, row, ac_q_col, 1.0)
                elif mtype == "V_AC":
                    self._add_derivative(H, row, ac_v_col, 1.0)
                elif mtype == "I_AC":
                    s_abs = float(np.hypot(ac_p, ac_q))
                    if s_abs > 1e-12 and abs(ac_v) > self.min_current_voltage:
                        self._add_derivative(H, row, ac_p_col, ac_p / (s_abs * ac_v))
                        self._add_derivative(H, row, ac_q_col, ac_q / (s_abs * ac_v))
                        self._add_derivative(H, row, ac_v_col, -s_abs / (ac_v * ac_v))
                else:
                    raise RuntimeError(f"Unsupported DCACConverter measurement type: {mtype}")

            elif meas.device_type == "ACACConverter":
                conv = self.acac_by_name[meas.device_name]
                k = self.acac_pos_by_name[conv.name]
                _, i_pos, j_pos, _ = self.calc.acac_converters[k]
                base = self.calc.acac_start + 4 * k
                i_p_col = int(self.acac_p_from_state_col[k])
                i_q_col = int(self.acac_q_from_state_col[k])
                j_p_col = int(self.acac_p_to_state_col[k])
                j_q_col = int(self.acac_q_to_state_col[k])
                i_v_col = int(self.ac_voltage_state_col[i_pos])
                j_v_col = int(self.ac_voltage_state_col[j_pos])
                i_p, i_q, j_p, j_q = full_x[base : base + 4]
                vi = ac_V[i_pos]
                vj = ac_V[j_pos]
                if mtype == "P_FROM":
                    self._add_derivative(H, row, i_p_col, 1.0)
                elif mtype == "Q_FROM":
                    self._add_derivative(H, row, i_q_col, 1.0)
                elif mtype == "V_FROM":
                    self._add_derivative(H, row, i_v_col, 1.0)
                elif mtype == "I_FROM":
                    s_abs = float(np.hypot(i_p, i_q))
                    if s_abs > 1e-12 and abs(vi) > self.min_current_voltage:
                        self._add_derivative(H, row, i_p_col, i_p / (s_abs * vi))
                        self._add_derivative(H, row, i_q_col, i_q / (s_abs * vi))
                        self._add_derivative(H, row, i_v_col, -s_abs / (vi * vi))
                elif mtype == "P_TO":
                    self._add_derivative(H, row, j_p_col, 1.0)
                elif mtype == "Q_TO":
                    self._add_derivative(H, row, j_q_col, 1.0)
                elif mtype == "V_TO":
                    self._add_derivative(H, row, j_v_col, 1.0)
                elif mtype == "I_TO":
                    s_abs = float(np.hypot(j_p, j_q))
                    if s_abs > 1e-12 and abs(vj) > self.min_current_voltage:
                        self._add_derivative(H, row, j_p_col, j_p / (s_abs * vj))
                        self._add_derivative(H, row, j_q_col, j_q / (s_abs * vj))
                        self._add_derivative(H, row, j_v_col, -s_abs / (vj * vj))
                else:
                    raise RuntimeError(f"Unsupported ACACConverter measurement type: {mtype}")

            else:
                raise RuntimeError(f"Unsupported measurement device type: {meas.device_type}")

        if not active_sparse:
            self._write_state(x)
        return H.to_csr() if sparse else H

    def jacobian(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        """Build the compact hybrid measurement Jacobian as a dense array."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.jacobian(x, measurements)
        if (
            measurements is None
            and (
                getattr(self, "_active_ac_hybrid_rows", np.array([], dtype=np.int32)).size
                or getattr(self, "_active_dc_hybrid_rows", np.array([], dtype=np.int32)).size
            )
        ):
            return self.jacobian_sparse(x).toarray()
        return self._assemble_jacobian(x, measurements, sparse=False)

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None):
        """Build the compact hybrid measurement Jacobian directly as sparse CSR."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.jacobian_sparse(x, measurements)
        return self._assemble_jacobian(x, measurements, sparse=True)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        """Rank-test the hybrid measurement Jacobian and report weak state directions."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.observability_analysis(
                x=x,
                measurements=measurements,
                H=H,
                normal_matrix=normal_matrix,
                normal_factor_diag=normal_factor_diag,
            )
        x = self.initial_state() if x is None else x
        measurements = self.active_measurements if measurements is None else list(measurements)
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
        return ObservabilityResult(
            observable=rank == self.n_state,
            rank=rank,
            state_count=self.n_state,
            measurement_count=len(measurements),
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )

    def _has_structural_observability_certificate(self, H) -> bool:
        """Certify sparse hybrid cases after AC angle components are structurally anchored."""
        rank = sparse_structural_rank(H)
        if rank != self.n_state:
            return False
        return not unanchored_angle_state_labels(H, self.state_labels, "AC_THETA:")

    def estimate(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> EstimateResult:
        """Run damped WLS on the compact hybrid state vector."""
        delegate = self._delegate()
        if delegate is not None:
            try:
                result = delegate.estimate(measurements=measurements, x0=x0, verbose=verbose, final_diagnostics=False)
            except TypeError:
                result = delegate.estimate(measurements=measurements, x0=x0, verbose=verbose)
            self._sync_from_delegate()
            return result
        measurements = self.active_measurements if measurements is None else list(measurements)
        if len(measurements) < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {len(measurements)} < {self.n_state}")

        x = self.initial_state() if x0 is None else x0.copy()
        if measurements is self.active_measurements:
            z = self.active_z
            weight = self.active_weight
        else:
            z = np.asarray([meas.value for meas in measurements], dtype=np.float64)
            weight = np.asarray([meas.weight for meas in measurements], dtype=np.float64)
        converged = False
        max_correction = np.inf
        objective = np.inf
        residual_inf = np.inf
        iteration = 0
        H = None
        gain = None
        final_quantities_current = False
        normal_factor_diag = None
        cached_z_est = None
        cached_residual = None
        cached_objective = None

        if verbose:
            _print_iteration_header()

        flat_restart_enabled = False
        iteration_limit = self.max_iter

        for iteration in range(1, iteration_limit + 1):
            if cached_z_est is None:
                z_est = self.evaluate(x, measurements)
                residual = self._measurement_residual(z, z_est, measurements)
                objective = 0.5 * float(np.dot(weight * residual, residual))
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
                cached_z_est = cached_residual = cached_objective = None
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            H = self.jacobian_sparse(x, measurements)
            gain, rhs = build_normal_equations(H, residual, weight)
            dx, normal_factor_diag = solve_normal_equations_with_factor(gain, rhs)

            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            if dx.size and not np.all(np.isfinite(dx)):
                final_quantities_current = True
                break
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
            # Preserve positive voltage states while reducing the step if the objective grows.
            for _ in range(20):
                candidate = x + step_scale * dx
                if self.voltage_cols.size:
                    candidate[self.voltage_cols] = np.maximum(candidate[self.voltage_cols], self.voltage_floor)
                candidate_z_est = self.evaluate(candidate, measurements)
                candidate_residual = self._measurement_residual(z, candidate_z_est, measurements)
                candidate_objective = 0.5 * float(np.dot(weight * candidate_residual, candidate_residual))
                finite_candidate = np.isfinite(candidate_objective) and np.all(np.isfinite(candidate_residual))
                if not finite_candidate:
                    nonfinite_candidates += 1
                    if nonfinite_candidates >= 8:
                        break
                    step_scale *= 0.5
                    continue
                nonfinite_candidates = 0
                if finite_candidate and candidate_objective <= objective:
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

            if verbose:
                if cached_residual is None:
                    cached_z_est = self.evaluate(x, measurements)
                    cached_residual = self._measurement_residual(z, cached_z_est, measurements)
                    cached_objective = 0.5 * float(np.dot(weight * cached_residual, cached_residual))
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

        if not converged and flat_restart_enabled:
            restart_result = self.estimate(measurements, x0=self.power_flow_state.copy(), verbose=verbose)
            restart_result.iterations += iteration
            return restart_result

        if not final_quantities_current:
            if cached_z_est is None:
                z_est = self.evaluate(x, measurements)
                residual = self._measurement_residual(z, z_est, measurements)
                objective = 0.5 * float(np.dot(weight * residual, residual))
            else:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
            H = self.jacobian_sparse(x, measurements)
            gain, _ = build_normal_equations(H, residual, weight)
            normal_factor_diag = None
        observability = self.observability_analysis(
            x,
            measurements,
            H=H,
            normal_matrix=gain,
            normal_factor_diag=normal_factor_diag,
        )
        self.apply_state(x)
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

    def _sync_from_delegate(self) -> None:
        delegate = self._delegate()
        if delegate is None:
            return
        self.measurements = delegate.measurements
        self.active_measurements = delegate.active_measurements
        self.active_z = delegate.active_z
        self.active_weight = delegate.active_weight
        self.active_angle_residual_mask = getattr(delegate, "active_angle_residual_mask", angle_residual_mask(self.active_measurements))
        self.state_labels = delegate.state_labels
        self.n_state = delegate.n_state
        self.voltage_cols = getattr(delegate, "voltage_cols", np.array([], dtype=np.int32))
        if hasattr(delegate, "power_flow_state"):
            self.power_flow_state = delegate.power_flow_state.copy()
        if hasattr(delegate, "flat_state"):
            self.flat_state = delegate.flat_state.copy()

    def identify_bad_data(self, result: EstimateResult, threshold: Optional[float] = None) -> Tuple[List[BadDataItem], np.ndarray]:
        """Flag hybrid measurements with normalized residuals above threshold."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.identify_bad_data(result, threshold)
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
        delegate = self._delegate()
        if delegate is not None:
            return delegate.estimate_with_bad_data_removal(threshold=threshold, max_remove=max_remove, verbose=verbose)
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
        delegate = self._delegate()
        if delegate is not None:
            delegate.apply_state(x)
            self._sync_from_delegate()
            return
        self._write_state(x)

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        delegate = self._delegate()
        if delegate is not None:
            delegate.print_state(x, limit=limit)
            return
        self._write_state(x)
        print("Estimated AC node states:")
        for node in self.ac_nodes[:limit]:
            print(f"  {node.name:14s} V={node.voltage:.9f} theta={node.angle:.9f} rad")
        if len(self.ac_nodes) > limit:
            print(f"  ... {len(self.ac_nodes) - limit} more AC nodes")
        print("Estimated DC node voltages:")
        for node in self.dc_nodes[:limit]:
            print(f"  {node.name:14s} V={node.voltage:.9f}")
        if len(self.dc_nodes) > limit:
            print(f"  ... {len(self.dc_nodes) - limit} more DC nodes")


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
    parser = argparse.ArgumentParser(description="Hybrid AC/DC weighted least-squares state estimation.")
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="Hybrid network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat hybrid state instead of power-flow seed.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated node states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    args = parser.parse_args(argv)

    estimator = HybridStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        parameter_file=Path(args.para),
        flat_start=args.flat_start,
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
        result = estimator.estimate(verbose=not args.quiet)

    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    bad_items, normalized = estimator.identify_bad_data(result, bad_threshold)
    print(
        "State estimation: "
        f"converged={result.converged}, "
        f"iter={result.iterations}, "
        f"objective={result.objective:.6e}, "
        f"max_dx={result.max_correction:.3e}, "
        f"norm_res={result.residual_inf:.3e}"
    )
    _print_observability(result.observability)
    _print_bad_data(bad_items, normalized, bad_threshold)

    if args.print_state:
        estimator.print_state(result.x)

    return 0 if result.converged and result.observability.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
