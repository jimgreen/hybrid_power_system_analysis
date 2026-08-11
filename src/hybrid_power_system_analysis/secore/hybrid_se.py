import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, eye as sparse_eye, hstack, vstack


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from paths import measurement_file, model_file
from efile_read import _read_efile_rows
from hybrid_lf import HybridPowerNetwork
from model.multi_energy_model import (
    MultiEnergyContext,
    attach_multi_energy_context,
    build_multi_energy_context_from_rows,
    electric_heat_balance_residual,
    hydrogen_electric_balance_residual,
    hydrogen_electric_dependent_value,
)
from model.ac_array_model import (
    BRANCH_COLS as AC_BRANCH_COLS,
    BUS_COLS as AC_BUS_COLS,
    GEN_COLS as AC_GEN_COLS,
    LOAD_COLS as AC_LOAD_COLS,
    SWITCH_COLS as AC_SWITCH_COLS,
    TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
    ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
)
from model.hybrid_array_model import (
    DCAC_COLS,
    DCAC_AC_CONTROL_LABEL,
    DCAC_DC_CONTROL_LABEL,
    DCAC_DEVICE_TYPE_LABEL,
)
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BUS_COLS as DC_BUS_COLS,
    CTRL_V as DC_CTRL_V,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
)
from model.ppc_topology import (
    build_ac_ppc_with_topology_from_efile_rows,
    build_dc_ppc_with_topology_from_efile_rows,
    build_hybrid_ppc_with_topology_from_efile_rows,
)
from model.meas_array_model import (
    attach_device_pos_from_name_arrays,
    build_meas_ppc_from_e_file,
    measurement_list_from_meas_ppc,
    sync_meas_ppc_from_measurement_table,
)
from model.meas_type import (
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACBreakConstraint,
    DEVICE_TYPE_ACBranch,
    DEVICE_TYPE_ACGenerator,
    DEVICE_TYPE_ACLoad,
    DEVICE_TYPE_ACNode,
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_ACPowerBalance,
    DEVICE_TYPE_ACSwitch,
    DEVICE_TYPE_ACSwitchConstraint,
    DEVICE_TYPE_ACTransformer,
    DEVICE_TYPE_ACThreeWindingTransformer,
    DEVICE_TYPE_ACZeroBranch,
    DEVICE_TYPE_ACZeroBranchConstraint,
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCBreakConstraint,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCGenerator,
    DEVICE_TYPE_DCLoad,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCSwitch,
    DEVICE_TYPE_DCSwitchConstraint,
    DEVICE_TYPE_DCZeroBranch,
    DEVICE_TYPE_DCZeroBranchConstraint,
    DEVICE_TYPE_NAMES,
    MEAS_TYPE_ANGLE,
    MEAS_TYPE_I_AC,
    MEAS_TYPE_I_DC,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_I_GEN,
    MEAS_TYPE_I_LOAD,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_I_THIRD,
    MEAS_TYPE_P_AC,
    MEAS_TYPE_P_DC,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_GEN,
    MEAS_TYPE_P_LOAD,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_P_THIRD,
    MEAS_TYPE_Q_AC,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_GEN,
    MEAS_TYPE_Q_LOAD,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_Q_THIRD,
    MEAS_TYPE_V,
    MEAS_TYPE_V_AC,
    MEAS_TYPE_V_DC,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_V_TO,
    MEAS_TYPE_V_THIRD,
    MEAS_TYPE_V_GEN,
    MEAS_TYPE_V_LOAD,
    MEAS_TYPE_CODES,
    MEAS_TYPE_NAMES,
)
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    EstimateResult,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_PSEUDO,
    Measurement,
    MeasurementList,
    MeasurementTable,
    MeasurementTableView,
    ObservabilityResult,
    TableBackedMeasurementList,
    is_pseudo_measurement,
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
from secore.gas_se import GasStateEstimator
from secore.heat_se import HeatStateEstimator
from secore.hydro_se import HydroStateEstimator
from secore.steam_se import SteamStateEstimator
from secore.se_array_plan import (
    MeasurementPartitions,
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    copy_measurement_view,
    extend_measurement_partitions,
    measurement_table_take,
    partition_measurements_by_code,
    rows_by_device_type_code,
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
    sparse_structural_rank,
    targeted_redundancy_count,
    unanchored_angle_state_indices,
)
from secore.se_result import (
    SEResult,
    build_seresult_full_from_table,
    build_seresult_summary,
    build_seresult_summary_from_table,
    normalize_seresult_result_mode,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata, state_meta_at
from unit_system import ac_current_base_ka, dc_current_base_ka


DEFAULT_CASE = model_file("hybrid", "qinling.e")
DEFAULT_MEAS = measurement_file("hybrid", "qinling.meas")


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
                ac_control_type=DCAC_AC_CONTROL_LABEL.get(int(row[DCAC_COLS["ac_control_type"]]), "NONE"),
                dc_control_type=DCAC_DC_CONTROL_LABEL.get(int(row[DCAC_COLS["dc_control_type"]]), "NONE"),
                p_ac_set=float(row[DCAC_COLS["p_ac_set"]]),
                p_dc_set=float(row[DCAC_COLS["p_dc_set"]]),
                q_ac_set=float(row[DCAC_COLS["q_ac_set"]]),
                v_ac_set=float(row[DCAC_COLS["v_ac_set"]]),
                v_dc_set=float(row[DCAC_COLS["v_dc_set"]]),
                run_stat=int(row[DCAC_COLS["run_stat"]]),
                dc_p=float(row[DCAC_COLS["dc_p"]]),
                ac_p=float(row[DCAC_COLS["ac_p"]]),
                ac_q=float(row[DCAC_COLS["ac_q"]]),
                dc_i=float(row[DCAC_COLS["dc_i"]]),
                ac_i=float(row[DCAC_COLS["ac_i"]]),
                dev_type=DCAC_DEVICE_TYPE_LABEL.get(
                    int(row[DCAC_COLS["dev_type"]]),
                    "DCACConverter",
                ),
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


def _assign_se_converter_topology(network: _SELightweightHybridNetwork) -> None:
    # Converter node/island object links are unused by the array-mode SE path
    # (only the load-flow path consumes them, and it wires its own). Alive flags
    # are derived from the topology arrays via the side alive lookups, so this
    # avoids touching ``network.dc.node_dict`` and keeps the DC object graph
    # unmaterialized for hybrid cold starts.
    ac_alive = _side_alive_node_lookup(network.ac)
    dc_alive = _side_alive_node_lookup(network.dc)
    for conv in network.dcac_converters:
        conv.ac_node_obj = None
        conv.dc_node_obj = None
        conv.ac_isl_obj = None
        conv.dc_isl_obj = None
        conv.is_alive = bool(
            int(getattr(conv, "run_stat", 1)) == 1
            and ac_alive.get(int(conv.ac_node), False)
            and dc_alive.get(int(conv.dc_node), False)
        )


def _build_hybrid_se_network_from_ppc(ppc: Dict) -> _SELightweightHybridNetwork:
    ac_network = _build_ac_se_ppc_namespace(ppc["ac"], Path(ppc.get("source", "")) if ppc.get("source") else None)
    dc_network = _build_dc_se_ppc_namespace(ppc["dc"])
    network = _SELightweightHybridNetwork(
        _se_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=_build_se_dcac_converters(ppc),
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


def _detect_se_rows_kind(rows) -> str:
    has_ac = bool(rows.get("ACNode", {}).get("rows"))
    has_dc = bool(rows.get("DCNode", {}).get("rows"))
    has_dcac = bool(rows.get("DCACConverter", {}).get("rows"))
    if has_dcac or (has_ac and has_dc):
        return "hybrid"
    if has_ac:
        return "ac"
    if has_dc:
        return "dc"
    if any(
        bool(rows.get(f"{prefix}Node", {}).get("rows"))
        for prefix in ("Heat", "Gas", "Hydro", "Steam")
    ):
        return "fluid"
    return "hybrid"


def _empty_se_side(base: Dict) -> SimpleNamespace:
    return SimpleNamespace(
        _se_lightweight=True,
        ppc=None,
        base=base,
        topology=None,
        _topology_arrays=None,
        p_base=float(base.get("p_base", 1.0)),
        p_base_kW=float(base.get("p_base_kW", base.get("p_base", 1.0))),
        u_scale=float(base.get("u_scale", 1.0)),
        p_scale=float(base.get("p_scale", 1.0)),
        i_scale=float(base.get("i_scale", 1.0)),
        nodes=[],
        node_dict={},
    )


def _build_se_network_from_fluid_only(file_name) -> _SELightweightHybridNetwork:
    base = {
        "p_base": 1.0,
        "p_base_kW": 1.0,
        "u_scale": 1.0,
        "p_scale": 1.0,
        "i_scale": 1.0,
    }
    return _SELightweightHybridNetwork(
        _se_lightweight=True,
        ac=_empty_se_side(base),
        dc=_empty_se_side(base),
        dcac_converters=[],
        hybrid_islands=[],
        ppc={
            "format": "hybrid_ppc_v1",
            "source": str(Path(file_name).resolve()),
            "base": base,
            "ac": None,
            "dc": None,
        },
        _ac_ppc=None,
        _dc_ppc=None,
        p_base=1.0,
        p_base_kW=1.0,
        u_scale=1.0,
        p_scale=1.0,
        i_scale=1.0,
    )


class _HybridStateLabelSlice:
    __slots__ = ("_parent", "_start", "_stop", "_step")

    def __init__(self, parent, start: int, stop: int, step: int):
        self._parent = parent
        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step)

    def __len__(self) -> int:
        return len(range(self._start, self._stop, self._step))

    def __iter__(self):
        for pos in range(self._start, self._stop, self._step):
            yield self._parent[pos]

    def __getitem__(self, key):
        values = range(self._start, self._stop, self._step)
        if isinstance(key, slice):
            sub = values[key]
            return _HybridStateLabelSlice(self._parent, sub.start, sub.stop, sub.step)
        return self._parent[values[int(key)]]

    def __contains__(self, value) -> bool:
        return any(label == value for label in self)

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def index(self, value) -> int:
        for pos, label in enumerate(self):
            if label == value:
                return pos
        raise ValueError(f"{value!r} is not in state labels")


class _HybridStateLabelView:
    __slots__ = ("ac_n", "dc_n", "dcac_names", "dcac_start", "n_state")

    def __init__(self, ac_n: int, dc_n: int, dcac_names: Sequence[str]):
        self.ac_n = int(ac_n)
        self.dc_n = int(dc_n)
        self.dcac_names = tuple(str(name) for name in dcac_names)
        self.dcac_start = self.ac_n + self.dc_n
        self.n_state = self.dcac_start + 3 * len(self.dcac_names)

    def __len__(self) -> int:
        return self.n_state

    def _label_at(self, pos: int) -> str:
        if pos < self.ac_n:
            return f"AC_STATE:{pos}"
        if pos < self.dcac_start:
            return f"DC_STATE:{pos - self.ac_n}"
        local = pos - self.dcac_start
        name = self.dcac_names[local // 3]
        kind = local % 3
        if kind == 0:
            return f"DCAC_P_DC:{name}"
        if kind == 1:
            return f"DCAC_P_AC:{name}"
        return f"DCAC_Q_AC:{name}"

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(self.n_state)[key]
            return _HybridStateLabelSlice(self, indices.start, indices.stop, indices.step)
        pos = int(key)
        if pos < 0:
            pos += self.n_state
        if pos < 0 or pos >= self.n_state:
            raise IndexError("state label index out of range")
        return self._label_at(pos)

    def __iter__(self):
        for pos in range(self.n_state):
            yield self._label_at(pos)

    def __contains__(self, value) -> bool:
        return any(label == value for label in self)

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def index(self, value) -> int:
        for pos, label in enumerate(self):
            if label == value:
                return pos
        raise ValueError(f"{value!r} is not in state labels")


class _HybridStateSideView:
    __slots__ = ("ac_n", "dc_n", "hybrid_n", "n_state")

    def __init__(self, ac_n: int, dc_n: int, hybrid_n: int):
        self.ac_n = int(ac_n)
        self.dc_n = int(dc_n)
        self.hybrid_n = int(hybrid_n)
        self.n_state = self.ac_n + self.dc_n + self.hybrid_n

    def __len__(self) -> int:
        return self.n_state

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[pos] for pos in range(self.n_state)[key]]
        pos = int(key)
        if pos < 0:
            pos += self.n_state
        if pos < self.ac_n:
            return "ac"
        if pos < self.ac_n + self.dc_n:
            return "dc"
        if pos < self.n_state:
            return "hybrid"
        raise IndexError("state side index out of range")

    def __iter__(self):
        for _ in range(self.ac_n):
            yield "ac"
        for _ in range(self.dc_n):
            yield "dc"
        for _ in range(self.hybrid_n):
            yield "hybrid"


class _DelegatedSequenceView:
    __slots__ = ("_delegate", "_attr")

    def __init__(self, delegate, attr: str):
        self._delegate = delegate
        self._attr = str(attr)

    def _target(self):
        return getattr(self._delegate, self._attr)

    def __len__(self) -> int:
        n_state = getattr(self._delegate, "n_state", None)
        if self._attr in {"state_labels", "state_meta"} and n_state is not None:
            return int(n_state)
        return len(self._target())

    def __getitem__(self, key):
        return self._target()[key]

    def __iter__(self):
        return iter(self._target())

    def __contains__(self, value) -> bool:
        return value in self._target()

    def __eq__(self, other) -> bool:
        try:
            return list(self._target()) == list(other)
        except TypeError:
            return False


def _state_meta_from_estimator_arrays(estimator, pos: int) -> Optional[StateMeta]:
    if estimator is None:
        return None
    pos = int(pos)
    arrays_ref = getattr(estimator, "_state_meta_arrays_ref", None)
    if callable(arrays_ref):
        arrays = arrays_ref()
        kind = np.asarray(arrays.get("kind", np.asarray([], dtype=object)), dtype=object)
        if 0 <= pos < kind.size:
            return StateMeta(
                str(np.asarray(arrays.get("side", np.asarray([], dtype=object)), dtype=object)[pos]),
                str(kind[pos]),
                str(np.asarray(arrays.get("device_type", np.asarray([], dtype=object)), dtype=object)[pos]),
                str(np.asarray(arrays.get("device_name", np.asarray([], dtype=object)), dtype=object)[pos]),
                terminal=str(np.asarray(arrays.get("terminal", np.asarray([], dtype=object)), dtype=object)[pos]),
                component=str(np.asarray(arrays.get("component", np.asarray([], dtype=object)), dtype=object)[pos]),
                legacy_label=str(np.asarray(arrays.get("legacy_label", np.asarray([], dtype=object)), dtype=object)[pos]),
                device_pos=int(np.asarray(arrays.get("device_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)[pos]),
                device_type_code=int(np.asarray(arrays.get("device_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)[pos]),
                meas_type_code=int(np.asarray(arrays.get("meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)[pos]),
            )
    return state_meta_at(getattr(estimator, "state_meta", ()), pos)


class _HybridStateMetaView:
    __slots__ = ("_labels", "_ac_estimator", "_dc_estimator")

    def __init__(self, labels: _HybridStateLabelView, ac_estimator=None, dc_estimator=None):
        self._labels = labels
        self._ac_estimator = ac_estimator
        self._dc_estimator = dc_estimator

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[pos] for pos in range(len(self))[key]]
        pos = int(key)
        labels = self._labels
        if pos < 0:
            pos += len(labels)
        if pos < labels.ac_n:
            meta = _state_meta_from_estimator_arrays(self._ac_estimator, pos)
            if meta is not None:
                return HybridStateEstimator._ac_sub_state_meta_to_hybrid(meta)
            return StateMeta("ac", "sub_state", "ACSubsystem", str(pos), legacy_label=labels[pos])
        if pos < labels.dcac_start:
            local = pos - labels.ac_n
            meta = _state_meta_from_estimator_arrays(self._dc_estimator, local)
            if meta is not None:
                return HybridStateEstimator._dc_sub_state_meta_to_hybrid(meta)
            return StateMeta("dc", "sub_state", "DCSubsystem", str(local), legacy_label=labels[pos])
        local = pos - labels.dcac_start
        conv_pos = local // 3
        name = labels.dcac_names[conv_pos]
        kind = local % 3
        if kind == 0:
            return StateMeta("hybrid", "dcac_p_dc", "DCACConverter", name, terminal="dc", component="p", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_DC)
        if kind == 1:
            return StateMeta("hybrid", "dcac_p_ac", "DCACConverter", name, terminal="ac", component="p", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_AC)
        return StateMeta("hybrid", "dcac_q_ac", "DCACConverter", name, terminal="ac", component="q", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_Q_AC)

    def __iter__(self):
        for pos in range(len(self)):
            yield self[pos]


@dataclass
class MultiEnergySEResult:
    electric: Optional[SEResult] = None
    fluids: Dict[str, SEResult] = field(default_factory=dict)
    fluid_estimators: Dict[str, object] = field(default_factory=dict)
    couplings: List[object] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    converged: bool = False
    observable: bool = False

    @property
    def bad_data_count(self) -> int:
        return sum(len(getattr(estimator, "bad_data", ())) for estimator in self.fluid_estimators.values())


@dataclass(frozen=True)
class _MultiEnergySEEndpoint:
    provider: str
    row: int
    domain: str
    device_type: str
    device_idx: int
    device_type_code: int
    meas_type_code: int
    device_pos: int


@dataclass(frozen=True)
class _MultiEnergySECouplingPlan:
    coupling: object
    t1: _MultiEnergySEEndpoint
    t2: _MultiEnergySEEndpoint
    t1_scale: float
    t2_scale: float
    normalization: float
    measurement_row: int
    control_col: int = -1
    prior_measurement_row: int = -1
    electric_heat: bool = False
    hydrogen_electric: bool = False
    controlled: Optional[_MultiEnergySEEndpoint] = None
    dependent: Optional[_MultiEnergySEEndpoint] = None
    electric: Optional[_MultiEnergySEEndpoint] = None
    heat_flow: Optional[_MultiEnergySEEndpoint] = None
    heat_temperature: Optional[_MultiEnergySEEndpoint] = None
    controlled_setpoint: float = 0.0
    controlled_measurement_rows: Tuple[int, ...] = ()
    dependent_measurement_rows: Tuple[int, ...] = ()
    electric_setpoint: float = 0.0
    return_temperature_col: int = -1
    supply_temperature_set: float = 0.0
    heat_capacity: float = 1.0
    power_scale: float = 1.0


class HybridStateEstimator:
    """AC/DC and fluid-network state-estimation orchestrator.

    AC and DC device rows are normalized, evaluated and differentiated by
    ACStateEstimator/DCStateEstimator.  This class owns only measurement
    partitioning, global WLS assembly and converter-side rows. Heat, gas,
    hydrogen and steam measurements are partitioned by device ownership and
    delegated to their specialized sparse WLS estimators.

    For a coupled hybrid network, the side estimators are measurement-model
    providers only: they do not run independent WLS iterations.  All AC, DC
    and converter states are assembled into one state vector, one sparse H,
    and one global normal equation solved jointly in each iteration.
    Multi-energy coupling rows remain in this layer for endpoint validation
    and aggregate result reporting.
    """

    _AC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "ACNode",
            "ACBranch",
            "ACTransformer",
            "ACThreeWindingTransformer",
            "AC3WTransformer",
            "ACBreak",
            "ACZeroBranch",
            "ACGenerator",
            "ACLoad",
            "ACPowerBalance",
            "ACZeroBranchConstraint",
            "ACBreakConstraint",
            "ACACConverter",
        )
    )
    _DC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "DCNode",
            "DCBranch",
            "DCBreak",
            "DCZeroBranch",
            "DCZeroBranchConstraint",
            "DCBreakConstraint",
            "DCGenerator",
            "DCLoad",
            "DCDCConverter",
            "DCPowerBalance",
        )
    )
    _HYBRID_MEASUREMENT_DEVICE_TYPES = frozenset(("DCACConverter",))
    _VOLTAGE_MEASUREMENT_TYPES = frozenset(
        ("V", "V_FROM", "V_TO", "V_THIRD", "V_GEN", "V_LOAD", "V_DC", "V_AC")
    )
    _VOLTAGE_MEASUREMENT_TYPE_TUPLE = tuple(_VOLTAGE_MEASUREMENT_TYPES)
    _MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE = {
        DEVICE_TYPE_ACNode: "ac",
        DEVICE_TYPE_ACBranch: "ac",
        DEVICE_TYPE_ACTransformer: "ac",
        DEVICE_TYPE_ACThreeWindingTransformer: "ac",
        DEVICE_TYPE_ACBreak: "ac",
        DEVICE_TYPE_ACZeroBranch: "ac",
        DEVICE_TYPE_ACGenerator: "ac",
        DEVICE_TYPE_ACLoad: "ac",
        DEVICE_TYPE_ACPowerBalance: "ac",
        DEVICE_TYPE_ACZeroBranchConstraint: "ac",
        DEVICE_TYPE_ACBreakConstraint: "ac",
        DEVICE_TYPE_ACACConverter: "ac",
        DEVICE_TYPE_DCNode: "dc",
        DEVICE_TYPE_DCBranch: "dc",
        DEVICE_TYPE_DCBreak: "dc",
        DEVICE_TYPE_DCZeroBranch: "dc",
        DEVICE_TYPE_DCZeroBranchConstraint: "dc",
        DEVICE_TYPE_DCBreakConstraint: "dc",
        DEVICE_TYPE_DCGenerator: "dc",
        DEVICE_TYPE_DCLoad: "dc",
        DEVICE_TYPE_DCDCConverter: "dc",
        DEVICE_TYPE_DCACConverter: "hybrid",
    }
    _DCAC_MEASUREMENT_CODE = {
        MEAS_TYPE_P_DC: 1,
        MEAS_TYPE_P_AC: 2,
        MEAS_TYPE_Q_AC: 3,
        MEAS_TYPE_V_DC: 4,
        MEAS_TYPE_I_DC: 5,
        MEAS_TYPE_V_AC: 6,
        MEAS_TYPE_I_AC: 7,
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
        plan_tables: Optional[object] = None

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
        # Per-code buckets (indexed by 1-based measurement code).
        # DCAC: 1=P_DC 2=P_AC 3=Q_AC 4=V_DC 5=I_DC 6=V_AC 7=I_AC
        dcac_p_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_p_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_q_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_ac: "HybridStateEstimator._HybridCodeBucket"

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

    def _warn_missing_device_pos(
        self,
        context: str,
        device_type_code: int,
        device_name: object = "",
        meas_type_code: int = 0,
    ) -> None:
        key = (str(context), int(device_type_code), int(meas_type_code))
        seen = getattr(self, "_missing_device_pos_warning_keys", None)
        if seen is None:
            seen = set()
            self._missing_device_pos_warning_keys = seen
        if key in seen:
            return
        seen.add(key)
        warnings.warn(
            "Hybrid SE measurement row missing device_pos; string/name-id fallback is disabled "
            f"(context={context}, device_type_code={int(device_type_code)}, "
            f"device_name={device_name}, meas_type_code={int(meas_type_code)}).",
            RuntimeWarning,
            stacklevel=2,
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
        solver_name = str(power_flow_linear_solver).strip().lower() if power_flow_linear_solver is not None else ""
        self.power_flow_linear_solver = solver_name or None

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
        self.multi_energy = MultiEnergyContext()
        self._fluid_only = False
        self.fluid_estimators: Dict[str, object] = {}
        self.fluid_se_rc: Dict[str, int] = {}
        self.fluid_se_errors: Dict[str, str] = {}
        self.multi_energy_result = MultiEnergySEResult()
        self.multi_energy_se_coupling_plans: List[_MultiEnergySECouplingPlan] = []
        self.multi_energy_estimate_result: Optional[EstimateResult] = None
        self.multi_energy_observability_result: Optional[ObservabilityResult] = None
        self._multi_energy_se_layout_ready = False
        if auto_prepare:
            self.prepare()

    def prepare(self) -> "HybridStateEstimator":
        result = self._prepare_electric()
        self._prepare_fluid_estimators()
        self._prepare_multi_energy_se_layout()
        return result

    def _prepare_electric(self) -> "HybridStateEstimator":
        if self._prepared:
            return self
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        efile_rows = _read_efile_rows(self.e_file)
        self.multi_energy = build_multi_energy_context_from_rows(
            efile_rows,
            source=self.e_file,
        )
        rows_kind = _detect_se_rows_kind(efile_rows)
        if rows_kind == "ac":
            ac_ppc = build_ac_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
            base = ac_ppc["base"]
            self.network = _SELightweightHybridNetwork(
                _se_lightweight=True,
                ac=_build_ac_se_ppc_namespace(ac_ppc, self.e_file),
                dc=_empty_se_side(base),
                dcac_converters=[],
                hybrid_islands=[],
                ppc={"format": "hybrid_ppc_v1", "source": str(Path(self.e_file).resolve()), "base": base, "ac": ac_ppc, "dc": None},
                _ac_ppc=ac_ppc,
                _dc_ppc=None,
            )
        elif rows_kind == "dc":
            dc_ppc = build_dc_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
            base = dc_ppc["base"]
            self.network = _SELightweightHybridNetwork(
                _se_lightweight=True,
                ac=_empty_se_side(base),
                dc=_build_dc_se_ppc_namespace(dc_ppc),
                dcac_converters=[],
                hybrid_islands=[],
                ppc={"format": "hybrid_ppc_v1", "source": str(Path(self.e_file).resolve()), "base": base, "ac": None, "dc": dc_ppc},
                _ac_ppc=None,
                _dc_ppc=dc_ppc,
            )
        elif rows_kind == "fluid":
            self.network = _build_se_network_from_fluid_only(self.e_file)
            self._fluid_only = True
        else:
            self.network = _build_hybrid_se_network_from_ppc(
                build_hybrid_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
            )
        attach_multi_energy_context(self.network, self.multi_energy)
        if rows_kind in ("ac", "dc"):
            base = self.network.ppc["base"]
            self.network.p_base = float(base["p_base"])
            self.network.u_scale = float(base["u_scale"])
            self.network.p_scale = float(base["p_scale"])
            self.network.i_scale = float(base["i_scale"])
            self.network.p_base_kW = float(base["p_base_kW"])
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        self.p_base = float(getattr(self.network, "p_base", 1.0))
        self.p_base_kW = float(getattr(self.network, "p_base_kW", self.p_base))
        self.u_scale = float(getattr(self.network, "u_scale", 1.0))
        self.p_scale = float(getattr(self.network, "p_scale", 1.0))
        self.i_scale = float(getattr(self.network, "i_scale", 1.0))
        if self._fluid_only:
            self._ac_sub_estimator = None
            self._dc_sub_estimator = None
            self._delegate_estimator = None
            self._prepared = True
            self._record_profile_time("init.total", time.perf_counter() - profile_start)
            return self
        if rows_kind in ("ac", "dc"):
            return self._prepare_uncoupled_direct_delegate(rows_kind, profile_start)
        stage_start = time.perf_counter()
        self.meas_ppc = build_meas_ppc_from_e_file(
            self.meas_file,
            include_strings=False,
            include_matrix=False,
        )
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
        self._populate_measurement_device_pos_from_sub_estimators(force=True)

        if self._try_delegate_uncoupled_single_side():
            self._prepared = True
            self._record_profile_time("init.total", time.perf_counter() - profile_start)
            return self

        stage_start = time.perf_counter()
        self._disable_angle_measurements()
        self._disable_ac_current_measurements()
        self._disable_unavailable_measurements()
        self._convert_measurements_to_pu()
        self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._finalize_sub_estimators_after_measurement_prepare()
        self._build_device_maps()
        self._populate_measurement_device_pos_from_sub_estimators(force=True)
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
        fast_observability = self._fast_active_observability_certificate()
        if fast_observability is not None:
            self._initial_observability_cache = fast_observability
            self.required_pseudo_measurement_count = 0
        else:
            self.required_pseudo_measurement_count = self._add_required_missing_pseudo_measurements()
        self._record_profile_time("init.required_missing_pseudo", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
        self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self.ac_angle_reference_anchor_count = self._add_ac_angle_reference_anchor_measurements()
        self._record_profile_time("init.add_ac_angle_reference_anchors", time.perf_counter() - stage_start)
        self._prepared = True
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

    def _prepare_fluid_estimators(self) -> None:
        if self.fluid_estimators or not self.multi_energy.fluid_networks:
            return
        all_measurements = Measurement.read_from_file(self.meas_file)
        estimator_types = {
            "heat": HeatStateEstimator,
            "gas": GasStateEstimator,
            "hydro": HydroStateEstimator,
            "steam": SteamStateEstimator,
        }
        for name, network in self.multi_energy.fluid_networks.items():
            prefix = network.prefix
            measurements = [
                measurement
                for measurement in all_measurements
                if str(measurement.device_type).startswith(prefix)
            ]
            estimator = estimator_types[name](
                network,
                measurements=measurements,
                parameters=self.params,
                flat_start=self.flat_start,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=False,
            )
            estimator.prepare()
            self.fluid_estimators[name] = estimator

    def _add_multi_energy_warning(self, message: str) -> None:
        if message not in self.multi_energy.warnings:
            self.multi_energy.warnings.append(message)

    def _electric_endpoint_measurement(self, terminal) -> Tuple[Optional[Measurement], str]:
        domain = str(terminal.domain)
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_source = device_type.endswith(("Generator", "Unit", "Source"))
        if domain not in {"ac", "dc"} or (not is_load and not is_source):
            return None, f"unsupported electric endpoint type {device_type}"

        ppc = getattr(self.network, "ppc", {}) or {}
        side_ppc = ppc.get(domain)
        if not isinstance(side_ppc, dict):
            return None, f"{domain} endpoint has no active estimator block"
        if domain == "ac":
            estimator = self._ac_sub_estimator
            table_key = "load" if is_load else "gen"
            columns = AC_LOAD_COLS if is_load else AC_GEN_COLS
            device_type_code = DEVICE_TYPE_ACLoad if is_load else DEVICE_TYPE_ACGenerator
            meas_type = "P_LOAD" if is_load else "P_GEN"
            meas_type_code = MEAS_TYPE_P_LOAD if is_load else MEAS_TYPE_P_GEN
        else:
            estimator = self._dc_sub_estimator
            table_key = "load" if is_load else "gen"
            columns = DC_LOAD_COLS if is_load else DC_GEN_COLS
            device_type_code = DEVICE_TYPE_DCLoad if is_load else DEVICE_TYPE_DCGenerator
            meas_type = "P_LOAD" if is_load else "P_GEN"
            meas_type_code = MEAS_TYPE_P_LOAD if is_load else MEAS_TYPE_P_GEN
        if estimator is None:
            return None, f"{domain} endpoint has no active state estimator"

        table = np.asarray(side_ppc.get(table_key, ()), dtype=np.float64)
        if table.ndim != 2 or table.size == 0:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        rows = np.flatnonzero(
            table[:, columns["idx"]].astype(np.int64, copy=False)
            == int(terminal.device_idx)
        )
        if rows.size == 0:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        table_row = int(rows[0])
        run_stat_col = columns.get("run_stat")
        if run_stat_col is not None and int(table[table_row, run_stat_col]) != 1:
            return None, f"inactive {device_type} idx={int(terminal.device_idx)}"

        if domain == "ac":
            positions = estimator._plan_pos_for_ppc_rows(
                device_type_code,
                np.asarray([table_row], dtype=np.int64),
                table.shape[0],
            )
            device_pos = int(positions[0]) if positions.size else -1
        else:
            active_rows = np.asarray(
                estimator._load_rows if is_load else estimator._generator_rows,
                dtype=np.int64,
            )
            positions = np.flatnonzero(active_rows == table_row)
            device_pos = int(positions[0]) if positions.size else -1
        if device_pos < 0:
            return None, f"{device_type} idx={int(terminal.device_idx)} is outside the live topology"

        measurement = Measurement(
            0,
            f"coupling_endpoint_{domain}_{device_type_code}_{device_pos}",
            "ACLoad" if domain == "ac" and is_load else
            "ACGenerator" if domain == "ac" else
            "DCLoad" if is_load else "DCGenerator",
            "",
            meas_type,
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=device_type_code,
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )
        return measurement, ""

    def _fluid_endpoint_measurement(self, terminal) -> Tuple[Optional[Measurement], str]:
        domain = str(terminal.domain)
        estimator = self.fluid_estimators.get(domain)
        if estimator is None:
            return None, f"{domain} endpoint has no active state estimator"
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_storage = device_type.endswith("Storage")
        is_source = device_type.endswith(("Source", "Unit", "Generator")) or is_storage
        if not is_load and not is_source:
            return None, f"unsupported fluid endpoint type {device_type}"
        if is_load:
            positions = [
                pos
                for pos, device in enumerate(estimator.network.loads)
                if int(device.idx) == int(terminal.device_idx)
            ]
        else:
            positions = [
                pos
                for pos, device in enumerate(estimator.network.sources)
                if bool(estimator.network.source_is_storage[pos]) == is_storage
                and int(device.idx) == int(terminal.device_idx)
            ]
        if not positions:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        device_pos = int(positions[0])
        runtime_device_type = (
            f"{estimator.network.prefix}Load"
            if is_load
            else f"{estimator.network.prefix}Storage"
            if is_storage
            else f"{estimator.network.prefix}Source"
        )
        measurement = Measurement(
            0,
            f"coupling_endpoint_{domain}_{device_pos}",
            runtime_device_type,
            "",
            "FLOW",
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=int(DEVICE_TYPE_CODES.get(runtime_device_type, 0)),
            meas_type_code=int(MEAS_TYPE_CODES.get("FLOW", 0)),
            device_pos=device_pos,
        )
        return measurement, ""

    def _heat_source_temperature_measurement(
        self,
        terminal,
    ) -> Tuple[Optional[Measurement], str]:
        if str(terminal.domain) != "heat" or not str(terminal.device_type).endswith(
            ("Source", "Storage")
        ):
            return None, "electric heating requires a HeatSource or HeatStorage endpoint"
        flow_measurement, reason = self._fluid_endpoint_measurement(terminal)
        if flow_measurement is None:
            return None, reason
        measurement = Measurement(
            0,
            f"coupling_endpoint_heat_temperature_{int(flow_measurement.device_pos)}",
            str(flow_measurement.device_type),
            "",
            "T_SUPPLY",
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=int(flow_measurement.device_type_code),
            meas_type_code=int(MEAS_TYPE_CODES.get("T_SUPPLY", 0)),
            device_pos=int(flow_measurement.device_pos),
        )
        return measurement, ""

    def _resolve_multi_energy_se_endpoint(self, terminal) -> Tuple[Optional[Measurement], str]:
        if str(terminal.domain) in {"ac", "dc"}:
            return self._electric_endpoint_measurement(terminal)
        return self._fluid_endpoint_measurement(terminal)

    def _multi_energy_terminal_setpoint(self, terminal) -> float:
        domain = str(terminal.domain)
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_storage = device_type.endswith("Storage")
        if domain in {"ac", "dc"}:
            side_ppc = (getattr(self.network, "ppc", {}) or {}).get(domain, {})
            table_key = "load" if is_load else "gen"
            columns = (
                AC_LOAD_COLS if domain == "ac" and is_load else
                AC_GEN_COLS if domain == "ac" else
                DC_LOAD_COLS if is_load else DC_GEN_COLS
            )
            table = np.asarray(side_ppc.get(table_key, ()), dtype=np.float64)
            if table.ndim != 2 or table.size == 0:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            rows = np.flatnonzero(
                table[:, columns["idx"]].astype(np.int64, copy=False)
                == int(terminal.device_idx)
            )
            if rows.size == 0:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            column = "pbase" if is_load else "p_set"
            return float(table[int(rows[0]), columns[column]])

        estimator = self.fluid_estimators[domain]
        network = estimator.network
        if is_load:
            positions = [
                pos
                for pos, device in enumerate(network.loads)
                if int(device.idx) == int(terminal.device_idx)
            ]
            if not positions:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            return float(network.load_flow_set[int(positions[0])])
        positions = [
            pos
            for pos, device in enumerate(network.sources)
            if bool(network.source_is_storage[pos]) == is_storage
            and int(device.idx) == int(terminal.device_idx)
        ]
        if not positions:
            raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
        return float(network.source_flow_set[int(positions[0])])

    def _multi_energy_endpoint_measurement_rows(
        self,
        endpoint: _MultiEnergySEEndpoint,
    ) -> Tuple[int, ...]:
        if endpoint.provider == "electric":
            measurements = self.active_measurements
            measurement_slice = self.electric_measurement_slice
        else:
            measurements = self.fluid_estimators[endpoint.provider].active_measurements
            measurement_slice = self.fluid_measurement_slices[endpoint.provider]
        table = measurements.table
        mask = (
            (np.asarray(table.device_type_code, dtype=np.int64) == endpoint.device_type_code)
            & (np.asarray(table.meas_type_code, dtype=np.int64) == endpoint.meas_type_code)
            & (np.asarray(table.device_pos, dtype=np.int64) == endpoint.device_pos)
        )
        return tuple(
            (measurement_slice.start + np.flatnonzero(mask)).astype(np.int64).tolist()
        )

    def _prepare_multi_energy_se_layout(self) -> None:
        if self._multi_energy_se_layout_ready or not self.fluid_estimators:
            return

        for estimator in self.fluid_estimators.values():
            estimator.analyze_observability(add_pseudo=True)

        self.electric_state_count = 0 if self._fluid_only else int(self.n_state)
        self.electric_state_slice = slice(0, self.electric_state_count)
        state_offset = self.electric_state_count
        self.fluid_state_slices: Dict[str, slice] = {}
        for name, estimator in self.fluid_estimators.items():
            self.fluid_state_slices[name] = slice(
                state_offset,
                state_offset + int(estimator.state_count),
            )
            state_offset += int(estimator.state_count)
        self.multi_energy_state_count = state_offset

        self.electric_measurement_count = 0 if self._fluid_only else len(self.active_measurements)
        self.electric_measurement_slice = slice(0, self.electric_measurement_count)
        measurement_offset = self.electric_measurement_count
        self.fluid_measurement_slices: Dict[str, slice] = {}
        for name, estimator in self.fluid_estimators.items():
            count = len(estimator.active_measurements)
            self.fluid_measurement_slices[name] = slice(
                measurement_offset,
                measurement_offset + count,
            )
            measurement_offset += count

        endpoint_measurements: Dict[str, MeasurementList] = {
            "electric": MeasurementList()
        }
        endpoint_measurements.update(
            {name: MeasurementList() for name in self.fluid_estimators}
        )
        endpoint_cache: Dict[Tuple[str, str, int, str], _MultiEnergySEEndpoint] = {}

        def register(
            terminal,
            endpoint_kind: str = "flow",
        ) -> Tuple[Optional[_MultiEnergySEEndpoint], str]:
            key = (
                str(terminal.domain),
                str(terminal.device_type),
                int(terminal.device_idx),
                str(endpoint_kind),
            )
            cached = endpoint_cache.get(key)
            if cached is not None:
                return cached, ""
            measurement, reason = (
                self._heat_source_temperature_measurement(terminal)
                if endpoint_kind == "temperature"
                else self._resolve_multi_energy_se_endpoint(terminal)
            )
            if measurement is None:
                return None, reason
            provider = "electric" if str(terminal.domain) in {"ac", "dc"} else str(terminal.domain)
            rows = endpoint_measurements[provider]
            endpoint = _MultiEnergySEEndpoint(
                provider=provider,
                row=len(rows),
                domain=str(terminal.domain),
                device_type=str(terminal.device_type),
                device_idx=int(terminal.device_idx),
                device_type_code=int(measurement.device_type_code),
                meas_type_code=int(measurement.meas_type_code),
                device_pos=int(measurement.device_pos),
            )
            rows.append(measurement)
            endpoint_cache[key] = endpoint
            return endpoint, ""

        pending = []
        for coupling in self.multi_energy.couplings:
            if not coupling.active or not coupling.supports_energy_balance:
                continue
            electric_scale = float(getattr(self.network, "p_base_kW", self.p_base_kW))
            if coupling.is_electric_heat_control:
                electric_terminal = coupling.electric_terminal
                heat_terminal = coupling.heat_terminal
                if electric_terminal is None or heat_terminal is None:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: missing electric or heat endpoint"
                    )
                    continue
                electric, reason = register(electric_terminal)
                if electric is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_flow, reason = register(heat_terminal)
                if heat_flow is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_temperature, reason = register(heat_terminal, "temperature")
                if heat_temperature is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_estimator = self.fluid_estimators["heat"]
                heat_network = heat_estimator.network
                source_pos = int(heat_flow.device_pos)
                return_node = int(heat_network.source_return_node_pos[source_pos])
                return_temperature_col = (
                    self.fluid_state_slices["heat"].start
                    + heat_estimator.base_temperature
                    + int(heat_network.return_temperature_state_by_node[return_node])
                )
                electric_setpoint = self._multi_energy_terminal_setpoint(electric_terminal)
                supply_temperature_set = float(
                    heat_network.source_supply_temperature_set[source_pos]
                )
                controlled = electric if coupling.control_type == "P" else heat_temperature
                dependent = heat_temperature if coupling.control_type == "P" else electric
                t1 = electric if coupling.t1 == electric_terminal else heat_temperature
                t2 = heat_temperature if coupling.t2 == heat_terminal else electric
                pending.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        electric_heat=True,
                        hydrogen_electric=False,
                        controlled=controlled,
                        dependent=dependent,
                        electric=electric,
                        heat_flow=heat_flow,
                        heat_temperature=heat_temperature,
                        electric_setpoint=float(electric_setpoint),
                        supply_temperature_set=supply_temperature_set,
                        controlled_setpoint=(
                            float(electric_setpoint)
                            if coupling.control_type == "P"
                            else supply_temperature_set
                        ),
                        dependent_setpoint=0.0,
                        return_temperature_col=return_temperature_col,
                        heat_capacity=max(
                            float(heat_network.medium.heat_capacity),
                            1.0e-12,
                        ),
                        power_scale=electric_scale,
                    )
                )
                continue
            t1, reason = register(coupling.t1)
            if t1 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            t2, reason = register(coupling.t2)
            if t2 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            if coupling.is_hydrogen_electric_control:
                controlled_terminal = coupling.controlled_terminal
                dependent_terminal = coupling.dependent_terminal
                controlled = t1 if coupling.t1 == controlled_terminal else t2
                dependent = t1 if coupling.t1 == dependent_terminal else t2
                controlled_setpoint = self._multi_energy_terminal_setpoint(
                    controlled_terminal
                )
                dependent_setpoint = hydrogen_electric_dependent_value(
                    coupling,
                    controlled_setpoint,
                    electric_scale,
                )
                pending.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        electric_heat=False,
                        hydrogen_electric=True,
                        controlled=controlled,
                        dependent=dependent,
                        controlled_setpoint=float(controlled_setpoint),
                        dependent_setpoint=float(dependent_setpoint),
                    )
                )
                continue
            factor = float(coupling.energy_factor)
            pending.append(
                SimpleNamespace(
                    coupling=coupling,
                    t1=t1,
                    t2=t2,
                    t1_scale=electric_scale if t1.domain in {"ac", "dc"} else factor,
                    t2_scale=electric_scale if t2.domain in {"ac", "dc"} else factor,
                    electric_heat=False,
                    hydrogen_electric=False,
                    controlled=None,
                    dependent=None,
                    controlled_setpoint=0.0,
                    dependent_setpoint=0.0,
                )
            )

        for measurements in endpoint_measurements.values():
            measurements.table = _measurement_table_from_measurements(measurements)
        self._multi_energy_endpoint_measurements = endpoint_measurements
        self._multi_energy_se_layout_ready = True

        initial_state = self._multi_energy_initial_state()
        endpoint_values, endpoint_jacobians = self._multi_energy_endpoint_runtime(
            initial_state,
            with_jacobian=True,
        )
        local_weight_max = 1.0
        for table in self._multi_energy_local_measurement_tables():
            if table.weight.size:
                local_weight_max = max(local_weight_max, float(np.max(table.weight)))
        self.multi_energy_coupling_weight = max(1.0e6, 100.0 * local_weight_max)
        self.multi_energy_coupling_prior_weight = local_weight_max
        control_measurements = MeasurementList()
        control_initial = []
        prepared = []
        for item in pending:
            coupling = item.coupling
            t1 = item.t1
            t2 = item.t2
            t1_scale = item.t1_scale
            t2_scale = item.t2_scale
            if item.electric_heat:
                source_flow = float(
                    endpoint_values[item.heat_flow.provider][item.heat_flow.row]
                )
                t_return = float(initial_state[item.return_temperature_col])
                coefficient = float(coupling.e2h_coeff)
                if coupling.control_type == "P":
                    dependent_setpoint = (
                        t_return
                        + item.electric_setpoint
                        * item.power_scale
                        * coefficient
                        / (source_flow * item.heat_capacity)
                        if abs(source_flow) > 1.0e-12
                        else item.supply_temperature_set
                    )
                    thermal_gap = dependent_setpoint - t_return
                else:
                    dependent_setpoint = (
                        source_flow
                        * item.heat_capacity
                        * (item.supply_temperature_set - t_return)
                        / (item.power_scale * coefficient)
                    )
                    thermal_gap = item.supply_temperature_set - t_return
                normalization = max(
                    1.0,
                    abs(item.electric_setpoint * item.power_scale * coefficient),
                    abs(source_flow * item.heat_capacity * thermal_gap),
                )
                control_col = self.multi_energy_state_count + len(control_initial)
                control_initial.append(float(dependent_setpoint))
                prepared.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        normalization=float(normalization),
                        control_col=control_col,
                        prior_measurement_row=-1,
                        electric_heat=True,
                        hydrogen_electric=False,
                        controlled=item.controlled,
                        dependent=item.dependent,
                        electric=item.electric,
                        heat_flow=item.heat_flow,
                        heat_temperature=item.heat_temperature,
                        controlled_setpoint=float(item.controlled_setpoint),
                        controlled_measurement_rows=(
                            self._multi_energy_endpoint_measurement_rows(item.controlled)
                        ),
                        dependent_measurement_rows=(
                            self._multi_energy_endpoint_measurement_rows(item.dependent)
                        ),
                        return_temperature_col=item.return_temperature_col,
                        supply_temperature_set=item.supply_temperature_set,
                        heat_capacity=item.heat_capacity,
                        power_scale=item.power_scale,
                        electric_setpoint=item.electric_setpoint,
                    )
                )
                continue
            t1_value = float(endpoint_values[t1.provider][t1.row])
            t2_value = float(endpoint_values[t2.provider][t2.row])
            endpoint = item.dependent if item.hydrogen_electric else t1
            endpoint_value = (
                float(endpoint_values[endpoint.provider][endpoint.row])
                if item.hydrogen_electric
                else t1_value
            )
            normalization = (
                max(1.0, abs(endpoint_value), abs(item.dependent_setpoint))
                if item.hydrogen_electric
                else max(
                    1.0,
                    abs(t1_value * t1_scale),
                    abs(t2_value * t2_scale),
                )
            )
            endpoint_row = endpoint_jacobians[endpoint.provider].getrow(endpoint.row)
            control_col = -1
            prior_measurement_row = -1
            if item.hydrogen_electric or endpoint_row.nnz == 0:
                control_col = self.multi_energy_state_count + len(control_initial)
                control_initial.append(
                    item.dependent_setpoint if item.hydrogen_electric else t1_value
                )
                if not item.hydrogen_electric:
                    prior_measurement_row = measurement_offset + len(control_measurements)
                    control_measurements.append(
                        Measurement(
                            prior_measurement_row + 1,
                            f"coupling_control_{coupling.table_name}_{coupling.idx}",
                            str(coupling.table_name),
                            str(coupling.name),
                            "T1_CONTROL",
                            self.multi_energy_coupling_prior_weight,
                            True,
                            t1_value,
                            status=MEAS_STATUS_PSEUDO,
                        )
                    )
            prepared.append(
                SimpleNamespace(
                    coupling=coupling,
                    t1=t1,
                    t2=t2,
                    t1_scale=float(t1_scale),
                    t2_scale=float(t2_scale),
                    normalization=float(normalization),
                    control_col=control_col,
                    prior_measurement_row=prior_measurement_row,
                    electric_heat=False,
                    hydrogen_electric=item.hydrogen_electric,
                    controlled=item.controlled,
                    dependent=item.dependent,
                    controlled_setpoint=item.controlled_setpoint,
                    controlled_measurement_rows=(
                        self._multi_energy_endpoint_measurement_rows(item.controlled)
                        if item.hydrogen_electric else ()
                    ),
                    dependent_measurement_rows=(
                        self._multi_energy_endpoint_measurement_rows(item.dependent)
                        if item.hydrogen_electric else ()
                    ),
                    electric=None,
                    heat_flow=None,
                    heat_temperature=None,
                    return_temperature_col=-1,
                    supply_temperature_set=0.0,
                    heat_capacity=1.0,
                    power_scale=1.0,
                    electric_setpoint=0.0,
                )
            )

        self._multi_energy_coupling_state_initial = np.asarray(control_initial, dtype=np.float64)
        self.multi_energy_coupling_state_slice = slice(
            self.multi_energy_state_count,
            self.multi_energy_state_count + len(control_initial),
        )
        self.multi_energy_state_count = self.multi_energy_coupling_state_slice.stop
        self.multi_energy_control_measurement_slice = slice(
            measurement_offset,
            measurement_offset + len(control_measurements),
        )
        energy_measurement_offset = self.multi_energy_control_measurement_slice.stop
        energy_measurements = MeasurementList()
        self.multi_energy_se_coupling_plans = []
        for item in prepared:
            coupling = item.coupling
            measurement_row = energy_measurement_offset + len(energy_measurements)
            self.multi_energy_se_coupling_plans.append(
                _MultiEnergySECouplingPlan(
                    coupling=coupling,
                    t1=item.t1,
                    t2=item.t2,
                    t1_scale=item.t1_scale,
                    t2_scale=item.t2_scale,
                    normalization=item.normalization,
                    measurement_row=measurement_row,
                    control_col=item.control_col,
                    prior_measurement_row=item.prior_measurement_row,
                    electric_heat=item.electric_heat,
                    hydrogen_electric=item.hydrogen_electric,
                    controlled=item.controlled,
                    dependent=item.dependent,
                    electric=item.electric,
                    heat_flow=item.heat_flow,
                    heat_temperature=item.heat_temperature,
                    controlled_setpoint=item.controlled_setpoint,
                    controlled_measurement_rows=item.controlled_measurement_rows,
                    dependent_measurement_rows=item.dependent_measurement_rows,
                    electric_setpoint=item.electric_setpoint,
                    return_temperature_col=item.return_temperature_col,
                    supply_temperature_set=item.supply_temperature_set,
                    heat_capacity=item.heat_capacity,
                    power_scale=item.power_scale,
                )
            )
            energy_measurements.append(
                Measurement(
                    measurement_row + 1,
                    f"coupling_{coupling.table_name}_{coupling.idx}",
                    str(coupling.table_name),
                    str(coupling.name),
                    "ENERGY_BALANCE",
                    self.multi_energy_coupling_weight,
                    True,
                    0.0,
                    status=MEAS_STATUS_PSEUDO,
                )
            )
        coupling_measurements = MeasurementList([*control_measurements, *energy_measurements])
        coupling_measurements.table = _measurement_table_from_measurements(coupling_measurements)
        self._multi_energy_coupling_measurements = coupling_measurements
        self.multi_energy_coupling_measurement_slice = slice(
            energy_measurement_offset,
            energy_measurement_offset + len(energy_measurements),
        )
        self.multi_energy_measurement_count = energy_measurement_offset + len(energy_measurements)
        self._multi_energy_measurement_table = self._build_multi_energy_measurement_table()
        self._multi_energy_z = np.asarray(self._multi_energy_measurement_table.value, dtype=np.float64)
        self._multi_energy_weight = np.asarray(self._multi_energy_measurement_table.weight, dtype=np.float64)
        self._multi_energy_angle_mask = np.asarray(
            self._multi_energy_measurement_table.angle_mask,
            dtype=bool,
        )
        self._multi_energy_has_angle_measurements = bool(np.any(self._multi_energy_angle_mask))
        voltage_cols = []
        if self.electric_state_count:
            voltage_cols.extend(np.asarray(self.voltage_cols, dtype=np.int64).tolist())
        self._multi_energy_voltage_cols = np.asarray(voltage_cols, dtype=np.int64)
        self._multi_energy_fluid_potential_cols = np.concatenate(
            [
                np.arange(state_slice.start, state_slice.start + estimator.n_potential, dtype=np.int64)
                for name, estimator in self.fluid_estimators.items()
                for state_slice in (self.fluid_state_slices[name],)
            ]
        ) if self.fluid_estimators else np.asarray([], dtype=np.int64)
        electric_labels = [] if self._fluid_only else list(self.state_labels)
        self.multi_energy_state_labels = electric_labels + [
            label
            for name, estimator in self.fluid_estimators.items()
            for label in estimator.state_labels
        ] + [
            f"Coupling:{plan.coupling.table_name}:{plan.coupling.name}:"
            f"{('TEMPERATURE' if plan.coupling.control_type == 'P' else 'POWER') if plan.electric_heat else 'DEPENDENT' if plan.hydrogen_electric else 'T1_CONTROL'}"
            for plan in self.multi_energy_se_coupling_plans
            if plan.control_col >= 0
        ]
        self._multi_energy_observability_cache = None

    def _multi_energy_local_measurement_tables(self) -> List[MeasurementTable]:
        tables = []
        if not self._fluid_only:
            tables.append(
                self._measurement_table_with_string_columns(
                    self._measurement_table(self.active_measurements)
                )
            )
        tables.extend(
            estimator.active_measurements.table
            for estimator in self.fluid_estimators.values()
        )
        return tables

    def _build_multi_energy_measurement_table(self) -> MeasurementTable:
        tables = self._multi_energy_local_measurement_tables()
        tables.append(self._multi_energy_coupling_measurements.table)
        combined = _measurement_table_from_measurements(MeasurementList())
        for table in tables:
            combined = concat_measurement_tables(combined, table)
        combined.idx = np.arange(1, combined.idx.size + 1, dtype=np.int64)
        return combined

    def _multi_energy_initial_state(self) -> np.ndarray:
        parts = []
        if not self._fluid_only:
            parts.append(np.asarray(self.initial_state(), dtype=np.float64))
        parts.extend(
            np.asarray(estimator.initial_state(), dtype=np.float64)
            for estimator in self.fluid_estimators.values()
        )
        coupling_initial = np.asarray(
            getattr(self, "_multi_energy_coupling_state_initial", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        )
        if coupling_initial.size:
            parts.append(coupling_initial.copy())
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)

    def _multi_energy_endpoint_runtime(self, x: np.ndarray, *, with_jacobian: bool):
        values = {}
        jacobians = {}
        for provider, measurements in self._multi_energy_endpoint_measurements.items():
            if not measurements:
                values[provider] = np.asarray([], dtype=np.float64)
                if with_jacobian:
                    state_slice = (
                        self.electric_state_slice
                        if provider == "electric"
                        else self.fluid_state_slices[provider]
                    )
                    jacobians[provider] = csr_matrix((0, state_slice.stop - state_slice.start))
                continue
            if provider == "electric":
                local_x = x[self.electric_state_slice]
                values[provider] = self.evaluate(local_x, measurements)
                if with_jacobian:
                    jacobians[provider] = self.jacobian_sparse(local_x, measurements)
            else:
                estimator = self.fluid_estimators[provider]
                local_x = x[self.fluid_state_slices[provider]]
                values[provider] = estimator.evaluate(local_x, measurements)
                if with_jacobian:
                    jacobians[provider] = estimator.jacobian_sparse(local_x, measurements)
        return values, jacobians

    def _evaluate_multi_energy(self, x: np.ndarray) -> np.ndarray:
        self._prepare_multi_energy_se_layout()
        state = np.asarray(x, dtype=np.float64)
        values = np.empty(self.multi_energy_measurement_count, dtype=np.float64)
        if self.electric_measurement_count:
            values[self.electric_measurement_slice] = self.evaluate(
                state[self.electric_state_slice],
                self.active_measurements,
            )
        for name, estimator in self.fluid_estimators.items():
            values[self.fluid_measurement_slices[name]] = estimator.evaluate(
                state[self.fluid_state_slices[name]],
            )
        endpoint_values, _endpoint_jacobians = self._multi_energy_endpoint_runtime(
            state,
            with_jacobian=False,
        )
        for plan in self.multi_energy_se_coupling_plans:
            if plan.electric_heat:
                dependent_value = float(state[plan.control_col])
                source_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                if plan.coupling.control_type == "P":
                    electric_power = plan.electric_setpoint
                    t_out = dependent_value
                else:
                    electric_power = dependent_value
                    t_out = plan.supply_temperature_set
                if plan.controlled_measurement_rows:
                    values[np.asarray(plan.controlled_measurement_rows, dtype=np.int64)] = (
                        plan.controlled_setpoint
                    )
                if plan.dependent_measurement_rows:
                    values[np.asarray(plan.dependent_measurement_rows, dtype=np.int64)] = (
                        dependent_value
                    )
                values[plan.measurement_row] = electric_heat_balance_residual(
                    plan.coupling,
                    electric_power,
                    source_flow,
                    t_out,
                    t_return,
                    plan.heat_capacity,
                    plan.power_scale,
                ) / plan.normalization
                continue
            if plan.hydrogen_electric:
                dependent_value = (
                    float(state[plan.control_col])
                    if plan.control_col >= 0
                    else float(
                        endpoint_values[plan.dependent.provider][plan.dependent.row]
                    )
                )
                expected = hydrogen_electric_dependent_value(
                    plan.coupling,
                    plan.controlled_setpoint,
                    float(getattr(self.network, "p_base_kW", self.p_base_kW)),
                )
                if plan.controlled_measurement_rows:
                    values[np.asarray(plan.controlled_measurement_rows, dtype=np.int64)] = (
                        plan.controlled_setpoint
                    )
                if plan.dependent_measurement_rows:
                    values[np.asarray(plan.dependent_measurement_rows, dtype=np.int64)] = (
                        dependent_value
                    )
                values[plan.measurement_row] = (
                    dependent_value - expected
                ) / plan.normalization
                continue
            if plan.control_col >= 0:
                t1_value = float(state[plan.control_col])
                values[plan.prior_measurement_row] = t1_value
            else:
                t1_value = float(endpoint_values[plan.t1.provider][plan.t1.row])
            t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
            values[plan.measurement_row] = (
                abs(t2_value) * plan.t2_scale
                - float(plan.coupling.efficiency) * abs(t1_value) * plan.t1_scale
            ) / plan.normalization
        return values

    @staticmethod
    def _append_multi_energy_endpoint_jacobian(
        rows: List[int],
        cols: List[int],
        data: List[float],
        *,
        target_row: int,
        endpoint: _MultiEnergySEEndpoint,
        endpoint_jacobians: Dict[str, csr_matrix],
        state_slice: slice,
        scale: float,
    ) -> None:
        jacobian_row = endpoint_jacobians[endpoint.provider].getrow(endpoint.row)
        start, stop = jacobian_row.indptr[0], jacobian_row.indptr[1]
        if stop <= start:
            return
        rows.extend([target_row] * (stop - start))
        cols.extend((state_slice.start + jacobian_row.indices[start:stop]).tolist())
        data.extend((scale * jacobian_row.data[start:stop]).tolist())

    def _multi_energy_jacobian_sparse(self, x: np.ndarray) -> csr_matrix:
        self._prepare_multi_energy_se_layout()
        state = np.asarray(x, dtype=np.float64)
        local_blocks = []
        if not self._fluid_only:
            local_blocks.append(
                self.jacobian_sparse(state[self.electric_state_slice], self.active_measurements)
            )
        local_blocks.extend(
            estimator.jacobian_sparse(state[self.fluid_state_slices[name]])
            for name, estimator in self.fluid_estimators.items()
        )
        local_jacobian = (
            block_diag(local_blocks, format="csr")
            if local_blocks
            else csr_matrix((0, self.multi_energy_state_count), dtype=np.float64)
        )
        if local_jacobian.shape[1] < self.multi_energy_state_count:
            local_jacobian = hstack(
                (
                    local_jacobian,
                    csr_matrix(
                        (
                            local_jacobian.shape[0],
                            self.multi_energy_state_count - local_jacobian.shape[1],
                        ),
                        dtype=np.float64,
                    ),
                ),
                format="csr",
            )
        if any(
            plan.hydrogen_electric or plan.electric_heat
            for plan in self.multi_energy_se_coupling_plans
        ):
            local_jacobian = local_jacobian.tolil()
            for plan in self.multi_energy_se_coupling_plans:
                if not (plan.hydrogen_electric or plan.electric_heat):
                    continue
                for row in plan.controlled_measurement_rows:
                    local_jacobian.rows[row] = []
                    local_jacobian.data[row] = []
                if plan.control_col < 0:
                    continue
                for row in plan.dependent_measurement_rows:
                    local_jacobian.rows[row] = [plan.control_col]
                    local_jacobian.data[row] = [1.0]
            local_jacobian = local_jacobian.tocsr()
        endpoint_values, endpoint_jacobians = self._multi_energy_endpoint_runtime(
            state,
            with_jacobian=True,
        )
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        added_row_offset = self.multi_energy_control_measurement_slice.start
        for plan in self.multi_energy_se_coupling_plans:
            local_row = plan.measurement_row - added_row_offset
            if plan.electric_heat:
                source_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                t_out = (
                    float(state[plan.control_col])
                    if plan.coupling.control_type == "P"
                    else plan.supply_temperature_set
                )
                flow_slice = self.fluid_state_slices[plan.heat_flow.provider]
                self._append_multi_energy_endpoint_jacobian(
                    rows,
                    cols,
                    data,
                    target_row=local_row,
                    endpoint=plan.heat_flow,
                    endpoint_jacobians=endpoint_jacobians,
                    state_slice=flow_slice,
                    scale=(
                        -plan.heat_capacity
                        * (t_out - t_return)
                        / plan.normalization
                    ),
                )
                rows.append(local_row)
                cols.append(plan.return_temperature_col)
                data.append(source_flow * plan.heat_capacity / plan.normalization)
                rows.append(local_row)
                cols.append(plan.control_col)
                data.append(
                    -source_flow * plan.heat_capacity / plan.normalization
                    if plan.coupling.control_type == "P"
                    else plan.power_scale
                    * float(plan.coupling.e2h_coeff)
                    / plan.normalization
                )
                continue
            if plan.hydrogen_electric:
                if plan.control_col >= 0:
                    rows.append(local_row)
                    cols.append(plan.control_col)
                    data.append(1.0 / plan.normalization)
                else:
                    dependent_slice = (
                        self.electric_state_slice
                        if plan.dependent.provider == "electric"
                        else self.fluid_state_slices[plan.dependent.provider]
                    )
                    self._append_multi_energy_endpoint_jacobian(
                        rows,
                        cols,
                        data,
                        target_row=local_row,
                        endpoint=plan.dependent,
                        endpoint_jacobians=endpoint_jacobians,
                        state_slice=dependent_slice,
                        scale=1.0 / plan.normalization,
                    )
                continue
            if plan.control_col >= 0:
                t1_value = float(state[plan.control_col])
                prior_row = plan.prior_measurement_row - added_row_offset
                rows.append(prior_row)
                cols.append(plan.control_col)
                data.append(1.0)
            else:
                t1_value = float(endpoint_values[plan.t1.provider][plan.t1.row])
            t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
            t1_sign = 1.0 if t1_value >= 0.0 else -1.0
            t2_sign = 1.0 if t2_value >= 0.0 else -1.0
            t1_slice = (
                self.electric_state_slice
                if plan.t1.provider == "electric"
                else self.fluid_state_slices[plan.t1.provider]
            )
            t2_slice = (
                self.electric_state_slice
                if plan.t2.provider == "electric"
                else self.fluid_state_slices[plan.t2.provider]
            )
            if plan.control_col >= 0:
                rows.append(local_row)
                cols.append(plan.control_col)
                data.append(
                    -float(plan.coupling.efficiency)
                    * t1_sign
                    * plan.t1_scale
                    / plan.normalization
                )
            else:
                self._append_multi_energy_endpoint_jacobian(
                    rows,
                    cols,
                    data,
                    target_row=local_row,
                    endpoint=plan.t1,
                    endpoint_jacobians=endpoint_jacobians,
                    state_slice=t1_slice,
                    scale=(
                        -float(plan.coupling.efficiency)
                        * t1_sign
                        * plan.t1_scale
                        / plan.normalization
                    ),
                )
            self._append_multi_energy_endpoint_jacobian(
                rows,
                cols,
                data,
                target_row=local_row,
                endpoint=plan.t2,
                endpoint_jacobians=endpoint_jacobians,
                state_slice=t2_slice,
                scale=t2_sign * plan.t2_scale / plan.normalization,
            )
        coupling_jacobian = coo_matrix(
            (
                np.asarray(data, dtype=np.float64),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(
                self.multi_energy_measurement_count - added_row_offset,
                self.multi_energy_state_count,
            ),
        ).tocsr()
        return vstack((local_jacobian, coupling_jacobian), format="csr")

    def _apply_multi_energy_state_bounds(self, x: np.ndarray) -> np.ndarray:
        if self._multi_energy_voltage_cols.size:
            x[self._multi_energy_voltage_cols] = np.maximum(
                x[self._multi_energy_voltage_cols],
                self.voltage_floor,
            )
        if self._multi_energy_fluid_potential_cols.size:
            x[self._multi_energy_fluid_potential_cols] = np.maximum(
                x[self._multi_energy_fluid_potential_cols],
                1.0e-12,
            )
        return x

    def _multi_energy_measurement_residual(
        self,
        z_est: np.ndarray,
        *,
        measurement_rows: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rows = None if measurement_rows is None else np.asarray(measurement_rows, dtype=np.int64)
        z = self._multi_energy_z if rows is None else self._multi_energy_z[rows]
        residual = (
            z - z_est
            if out is None
            else np.subtract(z, z_est, out=out)
        )
        mask = self._multi_energy_angle_mask if rows is None else self._multi_energy_angle_mask[rows]
        if np.any(mask):
            residual[mask] = (residual[mask] + np.pi) % (2.0 * np.pi) - np.pi
        return residual

    def _multi_energy_observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        H=None,
        measurement_rows: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        self._prepare_multi_energy_se_layout()
        state = self._multi_energy_initial_state() if x is None else np.asarray(x, dtype=np.float64)
        rows = (
            np.arange(self.multi_energy_measurement_count, dtype=np.int64)
            if measurement_rows is None
            else np.asarray(measurement_rows, dtype=np.int64)
        )
        jacobian = self._multi_energy_jacobian_sparse(state) if H is None else H
        if jacobian.shape[0] != rows.size:
            jacobian = jacobian[rows]
        measurement_count = int(rows.size)
        if matrix_is_empty(jacobian):
            result = ObservabilityResult(
                False,
                0,
                self.multi_energy_state_count,
                measurement_count,
                self.multi_energy_state_count,
                np.asarray([], dtype=np.float64),
                [],
            )
        else:
            structural_rank = sparse_structural_rank(jacobian)
            if structural_rank == self.multi_energy_state_count:
                result = ObservabilityResult(
                    True,
                    self.multi_energy_state_count,
                    self.multi_energy_state_count,
                    measurement_count,
                    0,
                    np.asarray([], dtype=np.float64),
                    [],
                )
            else:
                rank, deficiency, singular_values, weak_states = observability_rank_details(
                    jacobian,
                    self.multi_energy_state_count,
                )
                result = ObservabilityResult(
                    rank == self.multi_energy_state_count,
                    rank,
                    self.multi_energy_state_count,
                    measurement_count,
                    max(0, deficiency),
                    singular_values,
                    weak_states,
                )
        self.multi_energy_observability_result = result
        self._multi_energy_observability_cache = {
            "result": result,
            "x": state.copy(),
            "H": jacobian,
            "measurement_rows": rows.copy(),
        }
        return result

    def _estimate_multi_energy(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        measurement_rows: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        self._prepare_multi_energy_se_layout()
        x = self._multi_energy_initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        if x.size != self.multi_energy_state_count:
            raise ValueError(
                f"multi-energy initial state has {x.size} values; expected {self.multi_energy_state_count}"
            )
        rows = (
            np.arange(self.multi_energy_measurement_count, dtype=np.int64)
            if measurement_rows is None
            else np.asarray(measurement_rows, dtype=np.int64)
        )
        observability = observability or self._multi_energy_observability_analysis(
            x,
            measurement_rows=rows,
        )
        cached_H = None
        cache = self._multi_energy_observability_cache
        if cache is not None and cache.get("result") is observability:
            cached_x = np.asarray(cache.get("x"), dtype=np.float64)
            cached_rows = np.asarray(cache.get("measurement_rows"), dtype=np.int64)
            if (
                cached_x.shape == x.shape
                and np.array_equal(cached_x, x)
                and np.array_equal(cached_rows, rows)
            ):
                cached_H = cache.get("H")

        weight = self._multi_energy_weight[rows]
        uniform_weight = self._uniform_weight(weight)
        weights_are_uniform = uniform_weight is not None
        weighted_residual = None if weights_are_uniform else np.empty_like(weight)
        normal_solver = NormalEquationSolver(assume_fixed_pattern=True)
        normal_pattern = None
        z_est = np.empty(rows.size, dtype=np.float64)
        residual = np.empty_like(z_est)
        candidate_z_est = np.empty_like(z_est)
        candidate_residual = np.empty_like(z_est)
        converged = False
        objective = np.inf
        residual_inf = np.inf
        max_correction = np.inf
        scaled_correction = np.inf
        iteration = 0
        H = None
        gain = None
        if verbose:
            _print_iteration_header()

        for iteration in range(1, self.max_iter + 1):
            z_est[:] = self._evaluate_multi_energy(x)[rows]
            self._multi_energy_measurement_residual(
                z_est,
                measurement_rows=rows,
                out=residual,
            )
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if iteration == 1 and cached_H is not None:
                H = cached_H
                cached_H = None
            else:
                H = self._multi_energy_jacobian_sparse(x)[rows]
            if normal_pattern is None:
                normal_pattern = _normal_equation_structural_pattern(H)
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
                assume_normal_pattern_matches=True,
            )
            dx, _factor_diag = normal_solver.solve(
                gain,
                rhs,
                return_factor_diag=final_diagnostics,
            )
            if dx.size and not np.all(np.isfinite(dx)):
                break
            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            scaled_correction = (
                float(np.max(np.abs(dx) / np.maximum(1.0, np.abs(x))))
                if dx.size
                else 0.0
            )
            if scaled_correction < self.tol:
                converged = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break

            accepted = False
            stationary = False
            step_scale = 1.0
            for _attempt in range(16):
                candidate = self._apply_multi_energy_state_bounds(x + step_scale * dx)
                candidate_z_est[:] = self._evaluate_multi_energy(candidate)[rows]
                self._multi_energy_measurement_residual(
                    candidate_z_est,
                    measurement_rows=rows,
                    out=candidate_residual,
                )
                candidate_objective = self._weighted_objective(weight, candidate_residual)
                if np.isfinite(candidate_objective) and candidate_objective <= objective + 1.0e-12:
                    relative_objective_change = abs(objective - candidate_objective) / max(1.0, abs(objective))
                    x = candidate
                    z_est, candidate_z_est = candidate_z_est, z_est
                    residual, candidate_residual = candidate_residual, residual
                    objective = candidate_objective
                    residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                    accepted = True
                    stationary = bool(
                        relative_objective_change < self.tol
                        and scaled_correction < np.sqrt(self.tol)
                    )
                    break
                step_scale *= 0.5
            if not accepted:
                break
            if stationary:
                converged = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, True)
                break
            if verbose:
                _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, False)

        z_est[:] = self._evaluate_multi_energy(x)[rows]
        self._multi_energy_measurement_residual(
            z_est,
            measurement_rows=rows,
            out=residual,
        )
        objective = self._weighted_objective(weight, residual)
        residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
        if scaled_correction < self.tol:
            converged = True
        if final_diagnostics or H is None or gain is None:
            H = self._multi_energy_jacobian_sparse(x)[rows]
            normal_pattern = _normal_equation_structural_pattern(H)
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
                assume_normal_pattern_matches=True,
            )
        result = EstimateResult(
            converged=bool(converged and observability.observable),
            iterations=int(iteration),
            objective=float(objective),
            max_correction=float(max_correction),
            residual_inf=float(residual_inf),
            x=x,
            z_est=z_est.copy(),
            residual=residual.copy(),
            H=H,
            gain=gain,
            measurements=[],
            observability=observability,
            measurement_table=measurement_table_take(self._multi_energy_measurement_table, rows),
        )
        result.multi_energy_rows = rows.copy()
        self.multi_energy_estimate_result = result
        return result

    @staticmethod
    def _local_observability_result(H, state_count: int, measurement_count: int) -> ObservabilityResult:
        if state_count == 0:
            return ObservabilityResult(True, 0, 0, measurement_count, 0, np.asarray([]), [])
        rank, deficiency, singular_values, weak_states = observability_rank_details(H, state_count)
        return ObservabilityResult(
            rank == state_count,
            rank,
            state_count,
            measurement_count,
            max(0, deficiency),
            singular_values,
            weak_states,
        )

    def _multi_energy_local_result(
        self,
        global_result: EstimateResult,
        measurement_slice: slice,
        state_slice: slice,
        measurement_table: MeasurementTable,
    ) -> EstimateResult:
        selected_rows = np.asarray(
            getattr(
                global_result,
                "multi_energy_rows",
                np.arange(self.multi_energy_measurement_count, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        selected_positions = np.flatnonzero(
            (selected_rows >= measurement_slice.start)
            & (selected_rows < measurement_slice.stop)
        )
        local_rows = selected_rows[selected_positions] - measurement_slice.start
        local_table = measurement_table_take(measurement_table, local_rows)
        H = global_result.H[selected_positions, state_slice].tocsr()
        weights = np.asarray(local_table.weight, dtype=np.float64)
        gain, _rhs = build_normal_equations(
            H,
            global_result.residual[selected_positions],
            weights,
        )
        observability = self._local_observability_result(
            H,
            state_slice.stop - state_slice.start,
            int(selected_positions.size),
        )
        residual = global_result.residual[selected_positions].copy()
        return EstimateResult(
            converged=bool(global_result.converged and observability.observable),
            iterations=global_result.iterations,
            objective=self._weighted_objective(weights, residual),
            max_correction=global_result.max_correction,
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=global_result.x[state_slice].copy(),
            z_est=global_result.z_est[selected_positions].copy(),
            residual=residual,
            H=H,
            gain=gain,
            measurements=[],
            observability=observability,
            measurement_table=local_table,
        )

    def _build_result_from_table(
        self,
        result: EstimateResult,
        *,
        bad_items: Sequence[BadDataItem],
        normalized_residual: np.ndarray,
        result_mode: str,
        all_measurement_table: MeasurementTable,
    ) -> Optional[SEResult]:
        mode = normalize_seresult_result_mode(result_mode)
        if mode in {"none", "array"}:
            return None
        if mode == "summary":
            return build_seresult_summary_from_table(
                result,
                bad_items=bad_items,
                all_measurement_table=all_measurement_table,
            )
        return build_seresult_full_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurement_table=all_measurement_table,
        )

    def _commit_multi_energy_results(
        self,
        result: EstimateResult,
        *,
        result_mode: str,
        skip_bad_data: bool,
        threshold: float,
        removed_bad_data: Optional[Sequence[BadDataItem]] = None,
    ) -> Optional[SEResult]:
        self.multi_energy_estimate_result = result
        self.estimate_result = result
        self.observability_result = result.observability
        self.multi_energy_observability_result = result.observability
        self.removed_bad_data = list(removed_bad_data or ())
        if skip_bad_data:
            bad_items = []
            normalized = np.asarray([], dtype=np.float64)
        else:
            bad_items, normalized = self.identify_bad_data(result, threshold)
        self.bad_items = list(bad_items)
        self.normalized_residual = np.asarray(normalized, dtype=np.float64)
        self.se_result = self._build_result_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=self.normalized_residual,
            result_mode=result_mode,
            all_measurement_table=self._multi_energy_measurement_table,
        )

        electric_se_result = None
        if not self._fluid_only:
            electric_table = self._measurement_table(self.active_measurements)
            electric_estimate = self._multi_energy_local_result(
                result,
                self.electric_measurement_slice,
                self.electric_state_slice,
                electric_table,
            )
            self.electric_estimate_result = electric_estimate
            electric_se_result = self._build_result_from_table(
                electric_estimate,
                bad_items=[],
                normalized_residual=np.asarray([], dtype=np.float64),
                result_mode=result_mode,
                all_measurement_table=electric_table,
            )
            self.electric_se_result = electric_se_result

        self.fluid_se_rc = {}
        self.fluid_se_errors = {}
        for name, estimator in self.fluid_estimators.items():
            local_result = self._multi_energy_local_result(
                result,
                self.fluid_measurement_slices[name],
                self.fluid_state_slices[name],
                estimator.active_measurements.table,
            )
            estimator.result = local_result
            estimator.observability = local_result.observability
            estimator._write_back_state(local_result.x)
            if skip_bad_data:
                estimator.bad_data = []
                estimator.normalized_residual = np.asarray([], dtype=np.float64)
            else:
                local_bad_data, local_normalized = self.identify_bad_data(
                    local_result,
                    threshold,
                )
                estimator.bad_data = local_bad_data
                estimator.normalized_residual = local_normalized
            estimator.se_result = self._build_result_from_table(
                local_result,
                bad_items=estimator.bad_data,
                normalized_residual=estimator.normalized_residual,
                result_mode=result_mode,
                all_measurement_table=estimator.active_measurements.table,
            )
            self.fluid_se_rc[name] = 0 if local_result.converged else 1
            if not local_result.converged:
                self.fluid_se_errors[name] = (
                    f"fluid state estimation residual={float(local_result.residual_inf):.6e}"
                )

        self._build_multi_energy_result(electric_se_result)
        return None if self._fluid_only else self.se_result

    def _run_multi_energy(
        self,
        *,
        result_mode: str,
        remove_bad_data: bool,
        bad_threshold: Optional[float],
        max_remove: Optional[int],
        skip_bad_data: bool,
        verbose: bool,
        final_diagnostics: bool,
        observability: Optional[ObservabilityResult],
    ) -> Optional[SEResult]:
        threshold = self.params.bad_threshold if bad_threshold is None else float(bad_threshold)
        removed: List[BadDataItem] = []
        if remove_bad_data:
            result, removed = self._estimate_multi_energy_with_bad_data_removal(
                threshold=threshold,
                max_remove=max_remove,
                verbose=verbose,
                observability=observability,
            )
        else:
            result = self._estimate_multi_energy(
                verbose=verbose,
                final_diagnostics=final_diagnostics or not skip_bad_data,
                observability=observability,
            )
        return self._commit_multi_energy_results(
            result,
            result_mode=result_mode,
            skip_bad_data=skip_bad_data,
            threshold=threshold,
            removed_bad_data=removed,
        )

    def _estimate_multi_energy_with_bad_data_removal(
        self,
        *,
        threshold: float,
        max_remove: Optional[int],
        verbose: bool,
        observability: Optional[ObservabilityResult] = None,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        remove_limit = self.params.bad_max_remove if max_remove is None else int(max_remove)
        rows = np.arange(self.multi_energy_measurement_count, dtype=np.int64)
        x0 = None
        result = None
        removed: List[BadDataItem] = []
        current_observability = observability
        while True:
            result = self._estimate_multi_energy(
                x0=x0,
                measurement_rows=rows,
                verbose=verbose,
                final_diagnostics=True,
                observability=current_observability,
            )
            bad_items, _normalized = self.identify_bad_data(result, threshold)
            removable = []
            for item in bad_items:
                global_row = int(rows[int(item.row_pos)])
                if global_row < self.multi_energy_control_measurement_slice.start:
                    removable.append((item, global_row))
            if not removable or len(removed) >= remove_limit:
                break
            worst, global_row = removable[0]
            trial_rows = rows[rows != global_row]
            trial_observability = self._multi_energy_observability_analysis(
                result.x,
                measurement_rows=trial_rows,
            )
            if not trial_observability.observable:
                break
            removed.append(worst)
            rows = trial_rows
            x0 = result.x
            current_observability = trial_observability
        if result is None:
            raise RuntimeError("multi-energy bad-data estimation did not start")
        return result, removed

    def _prepare_uncoupled_direct_delegate(self, side: str, profile_start: float) -> "HybridStateEstimator":
        stage_start = time.perf_counter()
        self.meas_ppc = build_meas_ppc_from_e_file(
            self.meas_file,
            include_strings=False,
            include_matrix=False,
        )
        self.meas_ppc["_mutable_runtime_arrays"] = True
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
        self.dcac_converters = []

        stage_start = time.perf_counter()
        self._direct_delegate_fast_path = True
        if side == "ac":
            self._dc_sub_estimator = None
            self._ac_sub_estimator = ACStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.ac,
                measurements=self.meas_ppc,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            self._ac_sub_estimator.prepare()
            delegate = self._ac_sub_estimator
        elif side == "dc":
            self._ac_sub_estimator = None
            self._dc_sub_estimator = DCStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.dc,
                measurements=self.meas_ppc,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            self._dc_sub_estimator.prepare()
            delegate = self._dc_sub_estimator
        else:
            raise ValueError(f"Unsupported direct delegate side: {side}")
        self._record_profile_time("init.prepare_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._build_device_maps()
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._adopt_delegate(delegate, side)
        self.meas_ppc = getattr(delegate, "meas_ppc", self.meas_ppc)
        self._record_profile_time("init.adopt_delegate", time.perf_counter() - stage_start)
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
                table_builder=_measurement_table_from_measurements,
                as_view=False,
                sides=("ac", "dc", "hybrid"),
            )
            sources_by_side = partitions.measurements
            self._sub_measurement_sources_by_side = sources_by_side
            self._sub_measurement_source_rows_by_side = partitions.rows
            self._sub_measurement_source_count = len(self.measurements)
        return sources_by_side

    def _measurements_for_sub_estimator(self, side: str, share_measurements: bool):
        if share_measurements and self._is_uncoupled_single_side(side):
            meas_ppc = getattr(self, "meas_ppc", None)
            if isinstance(meas_ppc, dict):
                return meas_ppc
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
        if getattr(self.network, "dcac_converters", None):
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
        has_coupling = bool(getattr(self.network, "dcac_converters", None))
        return bool(self.flat_start and (has_coupling or (has_ac and has_dc)))

    def _sub_power_flow_linear_solver(self) -> Optional[str]:
        if self.power_flow_linear_solver:
            return self.power_flow_linear_solver
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        has_coupling = bool(getattr(self.network, "dcac_converters", None))
        return "umfpack" if has_dc or has_coupling else None

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
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
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
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            estimator.prepare()
            return estimator
        except RuntimeError as exc:
            if "No alive DC nodes" in str(exc):
                return None
            raise

    def _build_device_maps(self) -> None:
        if hasattr(self, "_hybrid_converter_device_key_array_cache"):
            delattr(self, "_hybrid_converter_device_key_array_cache")
        self.dcac_converters = sorted(
            (
                conv
                for conv in getattr(self.network, "dcac_converters", [])
                if getattr(conv, "is_alive", False)
            ),
            key=lambda item: item.idx,
        )
        self._install_compatibility_device_views()

    @staticmethod
    def _compat_names(ppc: Dict, name_key: str, table: np.ndarray, prefix: str) -> np.ndarray:
        names = np.asarray(ppc.get(name_key, ()), dtype=object)
        count = int(table.shape[0]) if table.ndim == 2 else 0
        if names.size == count:
            return names.astype(str, copy=False)
        idx = table[:, 0].astype(np.int64, copy=False) if count else np.asarray([], dtype=np.int64)
        return np.asarray([f"{prefix}_{int(value)}" for value in idx], dtype=object)

    @classmethod
    def _compat_devices(
        cls,
        ppc: Dict,
        table_key: str,
        name_key: str,
        cols: Dict[str, int],
        prefix: str,
        attr_cols: Dict[str, str],
    ) -> List[SimpleNamespace]:
        table = np.asarray(ppc.get(table_key, np.zeros((0, len(cols)))), dtype=np.float64)
        if table.ndim != 2 or table.shape[0] == 0:
            return []
        names = cls._compat_names(ppc, name_key, table, prefix)
        devices = []
        for row, name in zip(table, names):
            values = {
                attr: float(row[cols[column]]) if column in cols else 0.0
                for attr, column in attr_cols.items()
            }
            for attr in ("idx", "node", "i_node", "j_node", "run_stat", "status"):
                if attr in values:
                    values[attr] = int(values[attr])
            values["is_alive"] = bool(
                int(values.get("run_stat", 1)) == 1
                and int(values.get("status", 1)) == 1
            )
            devices.append(_array_device(values.pop("idx"), name, **values))
        return devices

    @classmethod
    def _compat_nodes(
        cls,
        ppc: Dict,
        cols: Dict[str, int],
        *,
        ac_side: bool,
    ) -> List[SimpleNamespace]:
        table = np.asarray(ppc.get("bus", np.zeros((0, len(cols)))), dtype=np.float64)
        if table.ndim != 2 or table.shape[0] == 0:
            return []
        names = cls._compat_names(ppc, "bus_name", table, "bus")
        nodes = []
        for row, name in zip(table, names):
            values = {
                "vbase": float(row[cols["vbase"]]),
                "voltage": float(row[cols["voltage"]]),
                "run_stat": int(row[cols["run_stat"]]),
                "is_alive": int(row[cols["run_stat"]]) == 1,
            }
            if ac_side:
                values["angle"] = float(row[cols["angle"]])
            nodes.append(_array_device(int(row[cols["idx"]]), name, **values))
        return nodes

    @staticmethod
    def _link_compat_nodes(devices: Sequence[SimpleNamespace], nodes_by_idx: Dict[int, SimpleNamespace]) -> None:
        for device in devices:
            if hasattr(device, "node"):
                device.node_obj = nodes_by_idx.get(int(device.node))
            if hasattr(device, "i_node"):
                device.i_node_obj = nodes_by_idx.get(int(device.i_node))
            if hasattr(device, "j_node"):
                device.j_node_obj = nodes_by_idx.get(int(device.j_node))

    @staticmethod
    def _compat_reference_nodes(estimator, node_ids: np.ndarray, nodes_by_idx: Dict[int, SimpleNamespace]):
        if estimator is None:
            return []
        references = np.asarray(getattr(estimator, "references", ()), dtype=np.int64)
        valid = references[(references >= 0) & (references < node_ids.size)]
        return [
            nodes_by_idx[int(node_ids[int(pos)])]
            for pos in valid
            if int(node_ids[int(pos)]) in nodes_by_idx
        ]

    def _install_compatibility_device_views(self) -> None:
        """Expose lightweight legacy lookup views without rebuilding network objects."""
        ac_ppc = getattr(getattr(self, "network", None), "_ac_ppc", None)
        dc_ppc = getattr(getattr(self, "network", None), "_dc_ppc", None)

        self.ac_nodes = self._compat_nodes(ac_ppc, AC_BUS_COLS, ac_side=True) if isinstance(ac_ppc, dict) else []
        self.dc_nodes = self._compat_nodes(dc_ppc, DC_BUS_COLS, ac_side=False) if isinstance(dc_ppc, dict) else []
        self.ac_node_by_name = {node.name: node for node in self.ac_nodes}
        self.ac_node_by_idx = {node.idx: node for node in self.ac_nodes}
        self.dc_node_by_name = {node.name: node for node in self.dc_nodes}
        self.dc_node_by_idx = {node.idx: node for node in self.dc_nodes}

        if isinstance(ac_ppc, dict):
            self.ac_branches = self._compat_devices(
                ac_ppc, "branch", "branch_name", AC_BRANCH_COLS, "branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.ac_transformers = self._compat_devices(
                ac_ppc, "transformer", "transformer_name", AC_TRANSFORMER_COLS, "transformer",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.ac_zero_branches = self._compat_devices(
                ac_ppc, "zero_branch", "zero_branch_name", AC_ZERO_BRANCH_COLS, "zero_branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_breakers = self._compat_devices(
                ac_ppc, "break", "break_name", AC_SWITCH_COLS, "break",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "status": "status", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_generators = self._compat_devices(
                ac_ppc, "gen", "gen_name", AC_GEN_COLS, "generator",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_loads = self._compat_devices(
                ac_ppc, "load", "load_name", AC_LOAD_COLS, "load",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
        else:
            self.ac_branches = self.ac_transformers = self.ac_zero_branches = self.ac_breakers = []
            self.ac_generators = self.ac_loads = []

        if isinstance(dc_ppc, dict):
            self.dc_branches = self._compat_devices(
                dc_ppc, "branch", "branch_name", DC_BRANCH_COLS, "branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.dc_zero_branches = self._compat_devices(
                dc_ppc, "zero_branch", "zero_branch_name", DC_ZERO_BRANCH_COLS, "zero_branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_breakers = self._compat_devices(
                dc_ppc, "break", "break_name", DC_SWITCH_COLS, "break",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "status": "status", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_generators = self._compat_devices(
                dc_ppc, "gen", "gen_name", DC_GEN_COLS, "generator",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_loads = self._compat_devices(
                dc_ppc, "load", "load_name", DC_LOAD_COLS, "load",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p"},
            )
        else:
            self.dc_branches = self.dc_zero_branches = self.dc_breakers = []
            self.dc_generators = self.dc_loads = []

        for devices, nodes in (
            (self.ac_branches, self.ac_node_by_idx),
            (self.ac_transformers, self.ac_node_by_idx),
            (self.ac_zero_branches, self.ac_node_by_idx),
            (self.ac_breakers, self.ac_node_by_idx),
            (self.ac_generators, self.ac_node_by_idx),
            (self.ac_loads, self.ac_node_by_idx),
            (self.dc_branches, self.dc_node_by_idx),
            (self.dc_zero_branches, self.dc_node_by_idx),
            (self.dc_breakers, self.dc_node_by_idx),
            (self.dc_generators, self.dc_node_by_idx),
            (self.dc_loads, self.dc_node_by_idx),
        ):
            self._link_compat_nodes(devices, nodes)

        ac = getattr(self, "_ac_sub_estimator", None)
        dc = getattr(self, "_dc_sub_estimator", None)
        ac_ids = np.asarray(getattr(ac, "_ac_node_ids", ()), dtype=np.int64)
        dc_ids = np.asarray(getattr(dc, "_node_idx_by_pos", ()), dtype=np.int64)
        self.ac_reference_nodes = self._compat_reference_nodes(ac, ac_ids, self.ac_node_by_idx)
        self.dc_reference_nodes = self._compat_reference_nodes(dc, dc_ids, self.dc_node_by_idx)

        self.ac_branch_by_name = {item.name: item for item in self.ac_branches}
        self.ac_zero_branch_by_name = {item.name: item for item in self.ac_zero_branches}
        self.ac_generator_by_name = {item.name: item for item in self.ac_generators}
        self.ac_load_by_name = {item.name: item for item in self.ac_loads}
        self.dc_zero_branch_by_name = {item.name: item for item in self.dc_zero_branches}
        self.dc_generator_by_name = {item.name: item for item in self.dc_generators}
        self.dc_load_by_name = {item.name: item for item in self.dc_loads}
        all_dcac = getattr(getattr(self, "network", None), "dcac_converters", self.dcac_converters)
        self.dcac_by_name = {item.name: item for item in all_dcac}

        if getattr(self, "network", None) is not None:
            if getattr(self.network, "ac", None) is not None:
                self.network.ac.alive_buses = [node for node in self.ac_nodes if node.is_alive]
            if getattr(self.network, "dc", None) is not None:
                self.network.dc.alive_buses = [node for node in self.dc_nodes if node.is_alive]

        for conv in self.dcac_converters:
            conv.ac_node_obj = self.ac_node_by_idx.get(int(conv.ac_node))
            conv.dc_node_obj = self.dc_node_by_idx.get(int(conv.dc_node))

    def _try_delegate_uncoupled_single_side(self) -> bool:
        if self.dcac_converters:
            return False
        if self._ac_sub_estimator is not None and self._dc_sub_estimator is None:
            self._adopt_delegate(self._ac_sub_estimator, "ac")
            return True
        if self._dc_sub_estimator is not None and self._ac_sub_estimator is None:
            self._adopt_delegate(self._dc_sub_estimator, "dc")
            return True
        return False

    def _adopt_delegate(self, delegate, side: str) -> None:
        direct_delegate = bool(getattr(self, "_direct_delegate_fast_path", False))
        self._delegate_estimator = delegate
        self._sub_estimators_enabled = True
        self.calc = self._calc_adapter()
        self.measurements = delegate.measurements
        self.active_measurements = delegate.active_measurements
        self.active_z = delegate.active_z
        self.active_weight = delegate.active_weight
        active_table = getattr(self.active_measurements, "table", None)
        delegate_angle_mask = getattr(delegate, "active_angle_residual_mask", None)
        if direct_delegate and delegate_angle_mask is not None:
            self.active_angle_residual_mask = delegate_angle_mask
        elif active_table is not None and len(active_table.idx) == len(self.active_measurements):
            self.active_angle_residual_mask = np.asarray(active_table.angle_mask, dtype=bool)
        else:
            self.active_angle_residual_mask = angle_residual_mask(self.active_measurements)
        delegate_state_labels = _DelegatedSequenceView(delegate, "state_labels") if direct_delegate else delegate.state_labels
        delegate_state_meta = _DelegatedSequenceView(delegate, "state_meta") if direct_delegate else getattr(delegate, "state_meta", [])
        self.state_meta = delegate_state_meta
        self.state_labels = delegate_state_labels
        self.ac_state_labels = delegate_state_labels if side == "ac" else []
        self.dc_state_labels = delegate_state_labels if side == "dc" else []
        self.ac_state_layout = (
            {"state_labels": delegate_state_labels, "n_state": int(delegate.n_state)}
            if direct_delegate and side == "ac"
            else delegate.state_layout() if side == "ac" else {"state_labels": [], "n_state": 0}
        )
        self.dc_state_layout = (
            {
                "state_labels": delegate_state_labels,
                "voltage_col": getattr(delegate, "voltage_col", np.array([], dtype=np.int32)),
                "n_state": delegate.n_state,
                "references": getattr(delegate, "references", []),
            }
            if side == "dc"
            else {"state_labels": [], "n_state": 0}
        )
        self.n_state = int(delegate.n_state)
        self.ac_n_state = int(delegate.n_state) if side == "ac" else 0
        self.dc_n_state = int(delegate.n_state) if side == "dc" else 0
        self.hybrid_n_state = 0
        self.dc_state_start = self.ac_n_state
        self.hybrid_state_start = self.ac_n_state + self.dc_n_state
        self.state_sides = _HybridStateSideView(self.ac_n_state, self.dc_n_state, 0)
        self.voltage_cols = np.asarray(getattr(delegate, "voltage_cols", []), dtype=np.int32)
        self.power_flow_state = np.array([], dtype=np.float64)
        self.flat_state = np.array([], dtype=np.float64)
        self._initial_state_cache_ready = False
        self.dc_reference_nodes = getattr(delegate, "references", []) if side == "dc" else []
        self.ac_reference_nodes = getattr(delegate, "references", []) if side == "ac" else []
        self.dc_node_voltage_measurements = getattr(delegate, "node_voltage_measurements", {}) if side == "dc" else {}
        self.ac_node_voltage_measurements = getattr(delegate, "_node_voltage_measurement_cache", {}) if side == "ac" else {}
        self.ac_theta_state_col = getattr(delegate, "angle_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.ac_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.dc_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "dc" else np.array([], dtype=np.int32)
        self.targeted_observability_pseudo_count = getattr(delegate, "targeted_observability_pseudo_count", 0)
        if direct_delegate:
            empty_rows = np.array([], dtype=np.int32)
            self.ac_meas_rows = empty_rows
            self.dc_meas_rows = empty_rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = []
            self.dc_meas = []
            self.hybrid_meas = []
            self._active_ac_sub_measurements = []
            self._active_dc_sub_measurements = []
            self._active_ac_sub_rows = empty_rows
            self._active_dc_sub_rows = empty_rows
            self._active_ac_hybrid_rows = empty_rows
            self._active_dc_hybrid_rows = empty_rows
            self._active_ac_delegated_row_mask = np.array([], dtype=bool)
            self._active_dc_delegated_row_mask = np.array([], dtype=bool)
            self._jacobian_static_skip = np.array([], dtype=bool)
        else:
            self._adopt_delegate_active_partition(side)
        if side == "ac":
            self.ac_state_cols = np.arange(self.n_state, dtype=np.int32)
            self.dc_state_cols = np.array([], dtype=np.int32)
            self.hybrid_state_cols = np.array([], dtype=np.int32)
            self.ac_vars = self.state_labels
            self.dc_vars = []
            self.hybrid_vars = []
            self.ac_state_slice = slice(0, self.n_state)
            self.dc_state_slice = slice(self.n_state, self.n_state)
            self.hybrid_state_slice = slice(self.n_state, self.n_state)
        else:
            self.ac_state_cols = np.array([], dtype=np.int32)
            self.dc_state_cols = np.arange(self.n_state, dtype=np.int32)
            self.hybrid_state_cols = np.array([], dtype=np.int32)
            self.ac_vars = []
            self.dc_vars = self.state_labels
            self.hybrid_vars = []
            self.ac_state_slice = slice(0, 0)
            self.dc_state_slice = slice(0, self.n_state)
            self.hybrid_state_slice = slice(self.n_state, self.n_state)
        self._install_compatibility_device_views()

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
        )

    def _partition_active_measurements(self) -> None:
        partitions = partition_measurements_by_code(
            self.active_measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            as_view=False,
            sides=("ac", "dc", "hybrid"),
        )
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
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
        partitions = partition_measurements_by_code(
            measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            as_view=False,
            sides=("ac", "dc", "hybrid"),
        )
        return {
            "ac": self._MeasurementSideBlock(
                partitions.rows["ac"].astype(np.int32, copy=False),
                partitions.measurements["ac"],
            ),
            "dc": self._MeasurementSideBlock(
                partitions.rows["dc"].astype(np.int32, copy=False),
                partitions.measurements["dc"],
            ),
            "hybrid": self._MeasurementSideBlock(
                partitions.rows["hybrid"].astype(np.int32, copy=False),
                partitions.measurements["hybrid"],
            ),
        }

    @staticmethod
    def _sub_measurement_plan_tables(estimator, measurements: Sequence[Measurement]):
        if estimator is None or len(measurements) == 0:
            return None
        return estimator.install_measurement_runtime(measurements)

    @staticmethod
    def _sub_measurement_runtime_input(block: "HybridStateEstimator._MeasurementSideBlock"):
        return block.plan_tables if block.plan_tables is not None else block.measurements

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
            if getattr(self, "_lazy_state_layout", False):
                self.ac_vars = self.state_labels[:ac_n]
                self.dc_vars = self.state_labels[dc_start : dc_start + dc_n]
                self.hybrid_vars = self.state_labels[hybrid_start : hybrid_start + hybrid_n]
            else:
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
        if table is not None and int(table.idx.size) == len(self.measurements):
            table_size = int(table.idx.size)
            angle_mask = np.asarray(getattr(table, "angle_mask", np.zeros(table_size, dtype=bool)), dtype=bool)
            if angle_mask.size != table_size:
                warnings.warn(
                    "Hybrid SE angle disable requires table.angle_mask; string fallback is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            if np.any(angle_mask):
                table.valid[angle_mask] = False
                measurement_table_status_code(table)[angle_mask] = MEAS_STATUS_INVALID
                if self.flat_start:
                    table.value[angle_mask] = 0.0
            return
        warnings.warn(
            "Hybrid SE angle disable requires table-backed measurements; Measurement object iteration is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _disable_ac_current_measurements(self) -> None:
        """Disable unsupported AC currents while retaining ideal-edge flow measurements."""
        table = getattr(self.measurements, "table", None)
        if table is None:
            warnings.warn(
                "Hybrid SE AC-current disable requires table-backed measurements; Measurement object iteration is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        if row_count == 0:
            return
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
        if device_type_code.size != row_count or meas_type_code.size != row_count:
            return
        ac_terminal_devices = np.asarray(
            (
                DEVICE_TYPE_ACBranch,
                DEVICE_TYPE_ACTransformer,
                DEVICE_TYPE_ACThreeWindingTransformer,
            ),
            dtype=np.int16,
        )
        ac_single_devices = np.asarray((DEVICE_TYPE_ACLoad, DEVICE_TYPE_ACGenerator), dtype=np.int16)
        ac_current_mask = (
            np.isin(device_type_code, ac_terminal_devices)
            & np.isin(
                meas_type_code,
                np.asarray((MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO, MEAS_TYPE_I_THIRD), dtype=np.int16),
            )
        )
        ac_current_mask |= (
            np.isin(device_type_code, ac_single_devices)
            & np.isin(meas_type_code, np.asarray((MEAS_TYPE_I_LOAD, MEAS_TYPE_I_GEN), dtype=np.int16))
        )
        ac_current_mask |= (device_type_code == DEVICE_TYPE_DCACConverter) & (meas_type_code == MEAS_TYPE_I_AC)
        if not np.any(ac_current_mask):
            return
        table.valid[ac_current_mask] = False
        measurement_table_status_code(table)[ac_current_mask] = MEAS_STATUS_INVALID
        self._invalidate_measurement_activity_summary()

    def _measurement_device_count_by_type_code(self) -> Dict[int, int]:
        ac = getattr(self, "_ac_sub_estimator", None)
        dc = getattr(self, "_dc_sub_estimator", None)
        counts: Dict[int, int] = {}
        if ac is not None:
            counts.update(ac.measurement_device_counts())
        if dc is not None:
            counts.update(dc.measurement_device_counts())
        counts[DEVICE_TYPE_DCACConverter] = int(len(getattr(self, "dcac_converters", ())))
        return counts

    def _disable_unavailable_measurements(self) -> None:
        device_count_by_code = self._measurement_device_count_by_type_code()
        unsupported_switch_codes = {
            DEVICE_TYPE_ACSwitch,
            DEVICE_TYPE_ACSwitchConstraint,
            DEVICE_TYPE_DCSwitch,
            DEVICE_TYPE_DCSwitchConstraint,
        }
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            disabled_rows = []
            if table_size:
                valid = np.asarray(table.valid, dtype=bool)
                weight = np.asarray(table.weight, dtype=np.float64)
                active = valid & (weight > 0.0)
                known = np.zeros(table_size, dtype=bool)
                rows_by_code = getattr(table, "rows_by_device_type_code", None)
                if rows_by_code is None:
                    rows_by_code = rows_by_device_type_code(table)
                device_pos = getattr(table, "device_pos", None)
                has_device_pos = device_pos is not None and np.asarray(device_pos).size == table_size
                if has_device_pos:
                    device_pos = np.asarray(device_pos, dtype=np.int64)

                for code in unsupported_switch_codes:
                    rows = rows_by_code.get(int(code))
                    rows = np.asarray(rows, dtype=np.int64) if rows is not None else np.asarray([], dtype=np.int64)
                    if rows.size:
                        rows = rows[active[rows]]
                        if rows.size:
                            known[rows] = True
                            disabled_rows.append(rows)

                for code, device_count in device_count_by_code.items():
                    rows = rows_by_code.get(int(code))
                    rows = np.asarray(rows, dtype=np.int64) if rows is not None else np.asarray([], dtype=np.int64)
                    if rows.size == 0:
                        continue
                    rows = rows[active[rows]]
                    if rows.size == 0:
                        continue
                    known[rows] = True
                    if int(device_count) <= 0 or not has_device_pos:
                        disabled_rows.append(rows)
                        continue
                    pos = device_pos[rows.astype(np.intp, copy=False)]
                    available = (pos >= 0) & (pos < int(device_count))
                    if not np.all(available):
                        disabled_rows.append(rows[~available])

                unknown_active = np.flatnonzero(active & ~known)
                if unknown_active.size:
                    disabled_rows.append(unknown_active)

                if disabled_rows:
                    rows = np.concatenate(disabled_rows).astype(np.int64, copy=False)
                    table.valid[rows] = False
                    measurement_table_status_code(table)[rows] = MEAS_STATUS_INVALID
            return

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
        if table is None or int(np.asarray(getattr(table, "idx", np.asarray([]))).size) != len(measurements):
            warnings.warn(
                "Hybrid SE requires table-backed measurements for max idx; Measurement object scan is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0
        return int(np.asarray(table.idx, dtype=np.int64).max()) if int(table.idx.size) else 0

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
            if voltage_node_mapper is not None and np.any(active):
                status_code = measurement_table_status_code(table)
                meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
                device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
                device_pos = self._hybrid_measurement_device_pos_array(table)
                if meas_type_code.size != table.idx.size or device_pos.size != table.idx.size:
                    self._warn_missing_device_pos("sub_summary_voltage", 0)
                    meas_type_code = np.asarray([], dtype=np.int16)
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                voltage_mask = (
                    active
                    & (status_code != MEAS_STATUS_PSEUDO)
                    & np.isin(meas_type_code, voltage_codes)
                    & (device_pos >= 0)
                )
                for row in np.flatnonzero(voltage_mask):
                    node_idx = voltage_node_mapper(
                        int(device_type_code[int(row)]),
                        int(device_pos[int(row)]),
                        int(meas_type_code[int(row)]),
                    )
                    if node_idx is not None:
                        weight = float(table.weight[row])
                        current = voltage_best.get(int(node_idx))
                        if current is None or weight > current[0]:
                            voltage_best[int(node_idx)] = (weight, float(table.value[row]))
        else:
            self._warn_missing_device_pos("sub_summary_no_table", 0)
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
            if name == "_real_voltage_observation_node_cache":
                copied[name] = value
            elif isinstance(value, set):
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
            summary_attrs = {
                "source_count": len(self.measurements),
            }
            self._sub_measurement_global_summary_attrs = summary_attrs
        attrs = {
            "_active_device_key_cache": set(),
            "_active_measurement_key_cache": set(),
            "_max_measurement_idx": (
                int(max_idx)
                if max_idx is not None
                else self._max_measurement_idx_fast(self.measurements)
            ),
        }
        if "_real_voltage_observation_node_cache" in side_attrs:
            attrs["_real_voltage_observation_node_cache"] = side_attrs["_real_voltage_observation_node_cache"]
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
        mapper = getattr(estimator, "voltage_measurement_node_idx", None)
        if not callable(mapper):
            return None

        def voltage_node_mapper(device_type_code: int, device_pos: int, meas_type_code: int):
            node_idx = mapper(device_type_code, device_pos, meas_type_code)
            return None if node_idx is None else int(node_idx)

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

    def _measurement_name_arrays_by_type_code(self, requested_codes=None) -> Dict[int, np.ndarray]:
        requested = None if requested_codes is None else {int(code) for code in requested_codes}
        out: Dict[int, np.ndarray] = {}
        ac = getattr(self, "_ac_sub_estimator", None)
        if ac is not None:
            out.update(ac.measurement_device_names(requested))
        dc = getattr(self, "_dc_sub_estimator", None)
        if dc is not None:
            out.update(dc.measurement_device_names(requested))
        if requested is None or DEVICE_TYPE_DCACConverter in requested:
            out[DEVICE_TYPE_DCACConverter] = np.asarray([conv.name for conv in self.dcac_converters], dtype=object)
        return out

    def _populate_measurement_device_pos_from_sub_estimators(
        self,
        *,
        force: bool = False,
        name_attach_codes=None,
    ) -> None:
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            return
        n_rows = int(table.idx.size)
        if n_rows == 0:
            table.device_pos = np.asarray([], dtype=np.int64)
            return
        meas_ppc = getattr(self, "meas_ppc", None)
        existing_device_pos = getattr(table, "device_pos", None)
        if existing_device_pos is not None and np.asarray(existing_device_pos).size == n_rows:
            device_pos = np.asarray(existing_device_pos, dtype=np.int64).copy()
        else:
            ppc_device_pos = meas_ppc.get("device_pos") if isinstance(meas_ppc, dict) else None
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == n_rows:
                device_pos = np.asarray(ppc_device_pos, dtype=np.int64).copy()
            else:
                device_pos = np.full(n_rows, -1, dtype=np.int64)

        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        source_count = int(getattr(self, "_sub_measurement_source_count", -1))
        can_reuse_partitions = (
            isinstance(sources_by_side, dict)
            and isinstance(rows_by_side, dict)
            and source_count == n_rows
        )
        if can_reuse_partitions:
            estimators = {"ac": self._ac_sub_estimator, "dc": self._dc_sub_estimator}
            for side, source in sources_by_side.items():
                estimator = estimators.get(side)
                estimator_measurements = getattr(estimator, "measurements", None) if estimator is not None else None
                source_table = getattr(estimator_measurements, "table", None)
                if source_table is None:
                    source_table = getattr(source, "table", None)
                rows = rows_by_side.get(side)
                source_pos = getattr(source_table, "device_pos", None) if source_table is not None else None
                if source_table is None or rows is None or source_pos is None:
                    continue
                rows = np.asarray(rows, dtype=np.int64)
                source_pos = np.asarray(source_pos, dtype=np.int64)
                if rows.size != source_pos.size or rows.size != int(getattr(source_table, "idx", np.asarray([])).size):
                    continue
                valid_rows = (rows >= 0) & (rows < n_rows) & (source_pos >= 0)
                if np.any(valid_rows):
                    device_pos[rows[valid_rows].astype(np.intp, copy=False)] = source_pos[
                        valid_rows
                    ].astype(np.int64, copy=False)

        previous_count = int(getattr(self, "_measurement_device_pos_populated_count", -1))
        unresolved = device_pos < 0
        name_attach_done_count = int(getattr(self, "_measurement_device_pos_name_attach_count", -1))
        should_attach = bool(force and np.any(unresolved) and name_attach_done_count != n_rows)
        if not should_attach and previous_count < 0:
            should_attach = bool(np.any(unresolved))

        if should_attach and isinstance(meas_ppc, dict):
            meas_ppc["device_pos"] = device_pos.copy()
            codes = np.asarray(getattr(table, "device_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            unresolved_codes = (
                np.unique(codes[unresolved])
                if codes.size == n_rows and np.any(unresolved)
                else np.asarray([], dtype=np.int16)
            )
            if name_attach_codes is not None:
                allowed_codes = np.asarray(tuple(int(code) for code in name_attach_codes), dtype=np.int16)
                unresolved_codes = unresolved_codes[np.isin(unresolved_codes, allowed_codes)]
            device_pos = attach_device_pos_from_name_arrays(
                meas_ppc,
                self._measurement_name_arrays_by_type_code(unresolved_codes),
            )
            if name_attach_codes is None:
                self._measurement_device_pos_name_attach_count = n_rows

        table.device_pos = device_pos
        cache = getattr(table, "_device_pos_plan_cache", None)
        if cache is not None:
            cache.clear()
        if can_reuse_partitions:
            for side, source in sources_by_side.items():
                source_table = getattr(source, "table", None)
                rows = rows_by_side.get(side)
                if source_table is None or rows is None:
                    continue
                rows = np.asarray(rows, dtype=np.int64)
                if rows.size == int(getattr(source_table, "idx", np.asarray([])).size):
                    source_table.device_pos = device_pos[rows.astype(np.intp, copy=False)].astype(np.int64, copy=True)
                    source_cache = getattr(source_table, "_device_pos_plan_cache", None)
                    if source_cache is not None:
                        source_cache.clear()
        if isinstance(getattr(self, "meas_ppc", None), dict):
            self.meas_ppc["device_pos"] = device_pos
        if not can_reuse_partitions:
            self._sub_measurement_sources_by_side = None
            self._sub_measurement_source_rows_by_side = None
        self._sub_measurement_source_count = n_rows
        self._measurement_device_pos_populated_count = n_rows

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
        meas_ppc = getattr(self, "meas_ppc", None)
        ppc_device_pos = meas_ppc.get("device_pos") if isinstance(meas_ppc, dict) else None
        if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == row_count:
            device_pos = np.asarray(ppc_device_pos, dtype=np.int64)
            table.device_pos = device_pos
            return device_pos
        device_pos = np.full(row_count, -1, dtype=np.int64)
        table.device_pos = device_pos
        return device_pos

    def _convert_hybrid_measurements_to_pu(self, measurements: Sequence[Measurement]) -> None:
        table = getattr(measurements, "table", None)
        has_table = table is not None and len(table.idx) == len(measurements)
        if not has_table:
            self._warn_missing_device_pos("convert_hybrid_measurements_no_table", DEVICE_TYPE_DCACConverter)
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
        self._ensure_hybrid_converter_scale_cache()

        dcac_rows = np.flatnonzero(active & (code == DEVICE_TYPE_DCACConverter) & (device_pos >= 0))
        if dcac_rows.size:
            dcac_pos = device_pos[dcac_rows].astype(np.intp, copy=False)
            scale_vals = np.ones(dcac_rows.size, dtype=np.float64)
            dcac_meas = meas_code[dcac_rows]
            power_mask = (dcac_meas == MEAS_TYPE_P_DC) | (dcac_meas == MEAS_TYPE_P_AC) | (dcac_meas == MEAS_TYPE_Q_AC)
            scale_vals[power_mask] = self._power_file_base()
            mask = dcac_meas == MEAS_TYPE_V_DC
            scale_vals[mask] = self._dcac_dc_v_file_scale_by_pos[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_I_DC
            scale_vals[mask] = self._dcac_dc_i_file_scale_by_pos[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_V_AC
            scale_vals[mask] = self._dcac_ac_v_file_scale_by_pos[dcac_pos[mask]]
            mask = dcac_meas == MEAS_TYPE_I_AC
            scale_vals[mask] = self._dcac_ac_i_file_scale_by_pos[dcac_pos[mask]]
            scale[dcac_rows] = scale_vals

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
            # A side partition may reject additional rows, but it must not
            # reactivate rows already rejected by Hybrid-level ownership or
            # device-position validation.
            master_table.valid[rows] &= np.asarray(table.valid, dtype=bool)
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

    @staticmethod
    def _lookup_node_values_by_idx(
        node_idx: np.ndarray,
        ids: np.ndarray,
        pos: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched_rows = np.zeros(nodes.size, dtype=bool)
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if nodes.size == 0 or ids.size == 0 or pos.size == 0 or values.size == 0:
            return out, matched_rows
        loc = np.searchsorted(ids, nodes)
        in_range = loc < ids.size
        if not np.any(in_range):
            return out, matched_rows
        rows = np.flatnonzero(in_range)
        loc_rows = loc[rows].astype(np.intp, copy=False)
        matched = ids[loc_rows] == nodes[rows]
        if not np.any(matched):
            return out, matched_rows
        matched_row_idx = rows[matched]
        value_pos = pos[loc[matched_row_idx].astype(np.intp, copy=False)]
        valid = (value_pos >= 0) & (value_pos < values.size)
        if np.any(valid):
            final_rows = matched_row_idx[valid]
            out[final_rows] = values[value_pos[valid].astype(np.intp, copy=False)]
            matched_rows[final_rows] = True
        return out, matched_rows

    def _ac_voltage_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched = np.zeros(nodes.size, dtype=bool)
        estimator = getattr(self, "_ac_sub_estimator", None)
        if estimator is not None and nodes.size:
            out, matched = self._lookup_node_values_by_idx(
                nodes,
                getattr(estimator, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
        if nodes.size and not np.all(matched):
            node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
            for row in np.flatnonzero(~matched):
                node = node_dict.get(int(nodes[int(row)]))
                if node is not None:
                    out[int(row)] = float(getattr(node, "vbase", 1.0) or 1.0)
        return out

    def _dc_voltage_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched = np.zeros(nodes.size, dtype=bool)
        estimator = getattr(self, "_dc_sub_estimator", None)
        if estimator is not None and nodes.size:
            out, matched = self._lookup_node_values_by_idx(
                nodes,
                getattr(estimator, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
        if nodes.size and not np.all(matched):
            node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
            for row in np.flatnonzero(~matched):
                node = node_dict.get(int(nodes[int(row)]))
                if node is not None:
                    out[int(row)] = float(getattr(node, "vbase", 1.0) or 1.0)
        return out

    def _ac_current_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        voltage_base = self._ac_voltage_base_array(node_idx)
        current = np.ones(voltage_base.size, dtype=np.float64)
        valid = np.abs(voltage_base) > 1e-12
        if np.any(valid):
            current[valid] = self.p_base_kW / (1000.0 * np.sqrt(3.0) * np.abs(voltage_base[valid]))
        return self.i_scale * current

    def _dc_current_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        voltage_base = self._dc_voltage_base_array(node_idx)
        current = np.ones(voltage_base.size, dtype=np.float64)
        valid = np.abs(voltage_base) > 1e-12
        if np.any(valid):
            current[valid] = self.p_base_kW / (1000.0 * np.abs(voltage_base[valid]))
        return self.i_scale * current

    def _ensure_hybrid_converter_node_cache(self) -> None:
        dcac_count = len(self.dcac_converters)
        if (
            getattr(self, "_dcac_count", None) == dcac_count
            and hasattr(self, "_dcac_dc_node")
            and hasattr(self, "_dcac_ac_node")
        ):
            return
        self._dcac_count = dcac_count
        self._dcac_dc_node = np.asarray([int(conv.dc_node) for conv in self.dcac_converters], dtype=np.int64)
        self._dcac_ac_node = np.asarray([int(conv.ac_node) for conv in self.dcac_converters], dtype=np.int64)
        if hasattr(self, "_hybrid_converter_scale_cache_key"):
            delattr(self, "_hybrid_converter_scale_cache_key")

    def _ensure_hybrid_converter_scale_cache(self) -> None:
        self._ensure_hybrid_converter_node_cache()
        cache_key = (
            int(self._dcac_count),
            float(getattr(self, "p_base_kW", 1.0)),
            float(getattr(self, "u_scale", 1.0)),
            float(getattr(self, "i_scale", 1.0)),
        )
        if getattr(self, "_hybrid_converter_scale_cache_key", None) == cache_key:
            return
        dcac_dc_v = self._dc_voltage_base_array(self._dcac_dc_node)
        dcac_ac_v = self._ac_voltage_base_array(self._dcac_ac_node)
        self._dcac_dc_v_file_scale_by_pos = self.u_scale * dcac_dc_v
        self._dcac_ac_v_file_scale_by_pos = self.u_scale * dcac_ac_v
        self._dcac_dc_i_file_scale_by_pos = self._dc_current_base_array(self._dcac_dc_node)
        self._dcac_ac_i_file_scale_by_pos = self._ac_current_base_array(self._dcac_ac_node)
        self._hybrid_converter_scale_cache_key = cache_key

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
        *,
        device_type_code: int = 0,
        device_pos: int = -1,
        meas_type_code: int = 0,
    ) -> int:
        device_type_code = int(device_type_code)
        meas_type_code = int(meas_type_code)
        device_pos = int(device_pos)
        if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("append_pseudo", device_type_code, device_name, meas_type_code)
            return next_idx
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
        measurement.device_type_code = device_type_code
        measurement.meas_type_code = meas_type_code
        measurement.device_pos = device_pos
        self._append_pseudo_measurement_objects((measurement,))
        return next_idx + 1

    def _measurement_spec(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
        value: float,
        device_name: str = "",
    ) -> "_HybridConverterMeasurementSpec":
        device_type_code = int(device_type_code)
        meas_type_code = int(meas_type_code)
        pos_value = int(device_pos)
        if device_type_code <= 0 or meas_type_code <= 0 or pos_value < 0:
            self._warn_missing_device_pos("measurement_spec", device_type_code, device_name, meas_type_code)
        device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
        meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
        return self._HybridConverterMeasurementSpec(
            device_type,
            device_name,
            meas_type,
            float(value),
            device_type_code,
            pos_value,
            meas_type_code,
        )

    def _pseudo_measurement_table_from_objects(
        self,
        measurements: Sequence[Measurement],
    ) -> MeasurementTable:
        count = len(measurements)
        idx = np.empty(count, dtype=np.int64)
        name = np.empty(count, dtype=object)
        device_type = np.empty(count, dtype=object)
        device_name = np.empty(count, dtype=object)
        meas_type = np.empty(count, dtype=object)
        weight = np.empty(count, dtype=np.float64)
        valid = np.ones(count, dtype=bool)
        value = np.empty(count, dtype=np.float64)
        device_type_code = np.empty(count, dtype=np.int16)
        meas_type_code = np.empty(count, dtype=np.int16)
        device_pos = np.empty(count, dtype=np.int64)
        angle_mask = np.zeros(count, dtype=bool)
        status_code = np.full(count, MEAS_STATUS_PSEUDO, dtype=np.int16)
        name_ids = np.full(count, -1, dtype=np.int64)
        rows_by_code: Dict[int, List[int]] = {}
        for pos, measurement in enumerate(measurements):
            idx[pos] = int(measurement.idx)
            name[pos] = measurement.name
            device_type[pos] = measurement.device_type
            device_name[pos] = measurement.device_name
            meas_type[pos] = measurement.meas_type
            weight[pos] = float(measurement.weight)
            valid[pos] = bool(measurement.valid)
            value[pos] = float(measurement.value)
            type_code = int(getattr(measurement, "device_type_code", 0))
            kind_code = int(getattr(measurement, "meas_type_code", 0))
            pos_value = int(getattr(measurement, "device_pos", -1))
            if type_code <= 0 or kind_code <= 0 or pos_value < 0:
                self._warn_missing_device_pos(
                    "pseudo_table",
                    type_code,
                    getattr(measurement, "device_name", ""),
                    kind_code,
                )
            device_type_code[pos] = type_code
            meas_type_code[pos] = kind_code
            device_pos[pos] = pos_value
            angle_mask[pos] = kind_code == MEAS_TYPE_ANGLE
            rows_by_code.setdefault(type_code, []).append(pos)
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
            status_code=status_code,
            rows_by_device_type_code={
                int(code): np.asarray(rows, dtype=np.int64)
                for code, rows in rows_by_code.items()
            },
            device_name_id=name_ids,
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )

    def _append_pseudo_measurement_objects(
        self,
        measurements: Sequence[Measurement],
    ) -> MeasurementTable:
        if not measurements:
            return MeasurementTable(
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
                device_name_id=np.asarray([], dtype=np.int64),
                meas_type_code=np.asarray([], dtype=np.int16),
                device_pos=np.asarray([], dtype=np.int64),
            )
        tail_table = self._pseudo_measurement_table_from_objects(measurements)
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or int(master_table.idx.size) != len(self.measurements):
            master_table = _measurement_table_from_measurements(self.measurements)
        new_table = concat_measurement_tables(master_table, tail_table)
        self.measurement_table = new_table
        self.measurements = TableBackedMeasurementList(
            new_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._invalidate_measurement_activity_summary()
        return tail_table

    def _measurement_table_with_string_columns(self, table: MeasurementTable) -> MeasurementTable:
        count = int(table.idx.size)
        if (
            int(np.asarray(table.name, dtype=object).size) == count
            and int(np.asarray(table.device_type, dtype=object).size) == count
            and int(np.asarray(table.device_name, dtype=object).size) == count
            and int(np.asarray(table.meas_type, dtype=object).size) == count
        ):
            return table
        idx = np.asarray(table.idx, dtype=np.int64)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
        device_pos = np.asarray(getattr(table, "device_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if meas_type_code.size != count:
            meas_type_code = np.zeros(count, dtype=np.int16)
        if device_pos.size != count:
            device_pos = np.full(count, -1, dtype=np.int64)
        name = (
            np.asarray(table.name, dtype=object)
            if int(np.asarray(table.name, dtype=object).size) == count
            else np.asarray([f"m{int(value)}" for value in idx], dtype=object)
        )
        device_type = (
            np.asarray(table.device_type, dtype=object)
            if int(np.asarray(table.device_type, dtype=object).size) == count
            else np.asarray([DEVICE_TYPE_NAMES.get(int(code), "") for code in device_type_code], dtype=object)
        )
        device_name = (
            np.asarray(table.device_name, dtype=object)
            if int(np.asarray(table.device_name, dtype=object).size) == count
            else self._device_name_array_from_table(table, device_type_code, device_pos)
        )
        meas_type = (
            np.asarray(table.meas_type, dtype=object)
            if int(np.asarray(table.meas_type, dtype=object).size) == count
            else np.asarray([MEAS_TYPE_NAMES.get(int(code), "") for code in meas_type_code], dtype=object)
        )
        return MeasurementTable(
            idx=idx,
            name=name,
            device_type=device_type,
            device_name=device_name,
            meas_type=meas_type,
            weight=np.asarray(table.weight, dtype=np.float64),
            valid=np.asarray(table.valid, dtype=bool),
            value=np.asarray(table.value, dtype=np.float64),
            device_type_code=device_type_code,
            angle_mask=np.asarray(table.angle_mask, dtype=bool),
            status_code=measurement_table_status_code(table).copy(),
            rows_by_device_type_code=getattr(table, "rows_by_device_type_code", None),
            device_name_id=getattr(table, "device_name_id", None),
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )

    def _device_name_array_from_table(
        self,
        table: MeasurementTable,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> np.ndarray:
        device_name_id = getattr(table, "device_name_id", None)
        meas_ppc = getattr(self, "meas_ppc", {})
        device_names = meas_ppc.get("device_names") if isinstance(meas_ppc, dict) else None
        if device_name_id is not None and isinstance(device_names, np.ndarray):
            ids = np.asarray(device_name_id, dtype=np.int64)
            names = np.asarray(device_names, dtype=object)
            if ids.size == int(device_type_code.size) and names.size:
                out = np.empty(ids.size, dtype=object)
                out[:] = ""
                valid = (ids >= 0) & (ids < names.size)
                if np.any(valid):
                    out[valid] = names[ids[valid].astype(np.intp, copy=False)]
                missing = ~valid
                if np.any(missing):
                    fallback = self._device_name_array_from_type_pos(device_type_code[missing], device_pos[missing])
                    out[missing] = fallback
                return out
        return self._device_name_array_from_type_pos(device_type_code, device_pos)

    def _device_name_array_from_type_pos(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> np.ndarray:
        names = np.empty(int(device_type_code.size), dtype=object)
        names[:] = ""
        type_codes = np.asarray(device_type_code, dtype=np.int16)
        positions = np.asarray(device_pos, dtype=np.int64)
        names_by_code = self._measurement_name_arrays_by_type_code(np.unique(type_codes))
        for code in np.unique(type_codes):
            rows = np.flatnonzero(type_codes == code)
            code_names = np.asarray(names_by_code.get(int(code), ()), dtype=object)
            row_pos = positions[rows]
            valid = (row_pos >= 0) & (row_pos < code_names.size)
            if np.any(valid):
                names[rows[valid]] = code_names[row_pos[valid].astype(np.intp, copy=False)]
            if np.any(~valid):
                names[rows[~valid]] = np.asarray(
                    [f"pos:{int(value)}" for value in row_pos[~valid]],
                    dtype=object,
                )
        return names

    def _hybrid_converter_spec_device_key(self, spec: "_HybridConverterMeasurementSpec") -> int:
        device_type_code = int(spec.device_type_code)
        device_pos = int(spec.device_pos)
        if device_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("converter_spec_key", device_type_code, spec.device_name, spec.meas_type_code)
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
        self._add_hybrid_pseudo_measurements()

    def _disable_coupled_ac_power_balance_rows(self) -> None:
        coupled_pos = self._converter_coupled_ac_device_positions()
        if coupled_pos.size == 0:
            return
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            if table_size:
                rows_by_code = getattr(table, "rows_by_device_type_code", None) or {}
                balance_code = int(DEVICE_TYPE_ACPowerBalance)
                balance_rows = rows_by_code.get(balance_code)
                if balance_rows is None:
                    balance_rows = np.flatnonzero(np.asarray(table.device_type_code, dtype=np.int16) == balance_code)
                else:
                    balance_rows = np.asarray(balance_rows, dtype=np.int64)
                if balance_rows.size:
                    matched = np.zeros(balance_rows.size, dtype=bool)
                    device_pos = getattr(table, "device_pos", None)
                    if device_pos is not None and np.asarray(device_pos).size == table_size:
                        matched |= np.isin(
                            np.asarray(device_pos, dtype=np.int64)[balance_rows.astype(np.intp, copy=False)],
                            coupled_pos,
                        )
                    else:
                        self._warn_missing_device_pos("disable_coupled_ac_power_balance", balance_code)
                    if np.any(matched):
                        disable_rows = balance_rows[matched]
                        table.valid[disable_rows] = False
                        measurement_table_status_code(table)[disable_rows] = MEAS_STATUS_INVALID
                        self._invalidate_measurement_activity_summary()
            return
        self._warn_missing_device_pos("disable_coupled_ac_power_balance_no_table", DEVICE_TYPE_ACPowerBalance)

    def _add_hybrid_pseudo_measurements(self) -> None:
        next_idx = self._max_measurement_idx_fast(self.measurements) + 1
        measured_devices = self._active_hybrid_converter_measurement_devices()
        converter_keys = self._hybrid_converter_device_key_array()
        if converter_keys.size and all(int(key) in measured_devices for key in converter_keys):
            return
        specs_to_add = []
        for spec in self._hybrid_converter_measurement_specs(source="pseudo"):
            if int(spec.device_type_code) != DEVICE_TYPE_DCACConverter:
                continue
            device_key = self._hybrid_converter_spec_device_key(spec)
            if device_key < 0 or device_key in measured_devices:
                continue
            if self._voltage_pseudo_is_covered_by_code(spec.device_type_code, spec.device_pos, spec.meas_type_code):
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
                device_type_code=spec.device_type_code,
                device_pos=spec.device_pos,
                meas_type_code=spec.meas_type_code,
            )

    def _hybrid_converter_device_key_array(self) -> np.ndarray:
        cached = getattr(self, "_hybrid_converter_device_key_array_cache", None)
        expected_size = len(self.dcac_converters)
        if cached is not None and int(np.asarray(cached).size) == expected_size:
            return cached
        parts = []
        if self.dcac_converters:
            parts.append(
                self._packed_device_key_array(
                    np.full(len(self.dcac_converters), DEVICE_TYPE_DCACConverter, dtype=np.int64),
                    np.arange(len(self.dcac_converters), dtype=np.int64),
                )
            )
        keys = np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, dtype=np.int64)
        self._hybrid_converter_device_key_array_cache = keys
        return keys

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
        converter_rows = active & (code == DEVICE_TYPE_DCACConverter)
        if not np.any(converter_rows):
            return set()
        converter_mask = converter_rows & (device_pos >= 0)
        if np.any(converter_mask):
            return set(self._packed_device_key_array(code[converter_mask], device_pos[converter_mask]).tolist())
        self._warn_missing_device_pos("active_hybrid_converter_measurement_devices", DEVICE_TYPE_DCACConverter)
        return set()

    def _active_hybrid_converter_measurement_devices_from(self, measurements: Sequence[Measurement]) -> set:
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return self._active_hybrid_converter_measurement_devices_from_table(table)
        self._warn_missing_device_pos("active_hybrid_converter_measurement_devices_sequence", DEVICE_TYPE_DCACConverter)
        return set()

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
        dcac_count = len(self.dcac_converters)
        hybrid_n = 3 * dcac_count
        self.ac_n_state = ac_n
        self.dc_n_state = dc_n
        self.hybrid_n_state = hybrid_n
        self.dc_state_start = ac_n
        self.hybrid_state_start = ac_n + dc_n
        self.dcac_state_start = ac_n + dc_n
        self.n_state = ac_n + dc_n + hybrid_n
        lazy_state_layout = self.n_state > 20000
        self._lazy_state_layout = bool(lazy_state_layout)
        if lazy_state_layout:
            labels = _HybridStateLabelView(
                ac_n,
                dc_n,
                (conv.name for conv in self.dcac_converters),
            )
            self.state_labels = labels
            self.state_meta = _HybridStateMetaView(labels, self._ac_sub_estimator, self._dc_sub_estimator)
            self.state_sides = _HybridStateSideView(ac_n, dc_n, hybrid_n)
            self.ac_state_labels = labels[:ac_n]
            self.dc_state_labels = labels[ac_n : ac_n + dc_n]
            self.ac_state_layout = {"state_labels": self.ac_state_labels, "n_state": ac_n}
            self.dc_state_layout = {"state_labels": self.dc_state_labels, "n_state": dc_n}
            self.dcac_p_dc_state_col = self.dcac_state_start + 3 * np.arange(dcac_count, dtype=np.int32)
            self.dcac_p_ac_state_col = self.dcac_p_dc_state_col + 1
            self.dcac_q_ac_state_col = self.dcac_p_dc_state_col + 2
            self._finish_state_layout_arrays()
            return
        state_meta: List[StateMeta] = []
        for idx in range(ac_n):
            meta = _state_meta_from_estimator_arrays(self._ac_sub_estimator, idx)
            if meta is None:
                state_meta.append(StateMeta("ac", "sub_state", "ACSubsystem", str(idx), legacy_label=f"AC_STATE:{idx}"))
            else:
                state_meta.append(self._ac_sub_state_meta_to_hybrid(meta))
        for idx in range(dc_n):
            meta = _state_meta_from_estimator_arrays(self._dc_sub_estimator, idx)
            if meta is None:
                state_meta.append(StateMeta("dc", "sub_state", "DCSubsystem", str(idx), legacy_label=f"DC_STATE:{idx}"))
            else:
                state_meta.append(self._dc_sub_state_meta_to_hybrid(meta))
        for pos, conv in enumerate(self.dcac_converters):
            state_meta.extend(
                (
                    StateMeta("hybrid", "dcac_p_dc", "DCACConverter", conv.name, terminal="dc", component="p", legacy_label=f"DCAC_P_DC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_DC),
                    StateMeta("hybrid", "dcac_p_ac", "DCACConverter", conv.name, terminal="ac", component="p", legacy_label=f"DCAC_P_AC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_AC),
                    StateMeta("hybrid", "dcac_q_ac", "DCACConverter", conv.name, terminal="ac", component="q", legacy_label=f"DCAC_Q_AC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_Q_AC),
                )
            )
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
        self._finish_state_layout_arrays()

    def _finish_state_layout_arrays(self) -> None:
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
        self._install_compatibility_device_views()

    def _build_hybrid_converter_array_cache(self) -> None:
        self._ensure_hybrid_converter_node_cache()
        self._dcac_dc_v_col_by_pos = self._dc_voltage_cols_for_nodes(self._dcac_dc_node)
        self._dcac_ac_v_col_by_pos = self._ac_voltage_cols_for_nodes(self._dcac_ac_node)
        self._dcac_dc_v_default_by_pos = self._dc_voltage_defaults_for_nodes(self._dcac_dc_node)
        self._dcac_ac_v_default_by_pos = self._ac_voltage_defaults_for_nodes(self._dcac_ac_node)
        flat_parts = []
        nonflat_parts = []
        if self._dcac_count:
            dcac_flat = np.zeros(3 * self._dcac_count, dtype=np.float64)
            # Terminal powers are always positive into the converter. A
            # lossless seed therefore has P_DC == -P_AC.
            p_ac_set = np.asarray(
                [float(getattr(conv, "p_ac_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            p_dc_set = np.asarray(
                [float(getattr(conv, "p_dc_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dc_p_control = np.asarray(
                [str(getattr(conv, "dc_control_type", "NONE")).upper() == "P" for conv in self.dcac_converters],
                dtype=bool,
            )
            dcac_flat[0::3] = np.where(dc_p_control, p_dc_set, -p_ac_set)
            dcac_flat[1::3] = np.where(dc_p_control, -p_dc_set, p_ac_set)
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
            materialize_measurements=False,
        )
        partitions = partition_measurements_by_code(
            active_view.measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
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
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
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

    @staticmethod
    def _preserve_incremental_appended_string_fields(
        table: MeasurementTable,
        base_count: int,
        appended_table: MeasurementTable,
    ) -> None:
        total_count = int(table.idx.size)
        base_count = int(base_count)
        appended_count = int(appended_table.idx.size)
        if base_count < 0 or base_count + appended_count != total_count:
            return
        for field_name in ("name", "device_type", "device_name", "meas_type"):
            values = np.asarray(getattr(table, field_name), dtype=object)
            if values.size == total_count:
                continue
            appended_values = np.asarray(getattr(appended_table, field_name), dtype=object)
            if appended_values.size != appended_count:
                continue
            merged = np.empty(total_count, dtype=object)
            merged[:base_count] = ""
            merged[base_count:] = appended_values
            setattr(table, field_name, merged)

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
        appended_count = len(appended_measurements)
        source_row_start = len(self.measurements) - appended_count
        if source_row_start < 0:
            return False
        if len(active_table.idx) != len(self.active_measurements):
            return False
        master_size = int(master_table.idx.size)
        if master_size == len(self.measurements):
            appended_rows = np.arange(source_row_start, master_size, dtype=np.int64)
            appended_table = measurement_table_take(master_table, appended_rows)
            base_table = measurement_table_take(master_table, np.arange(source_row_start, dtype=np.int64))
        elif master_size == source_row_start:
            appended_table = _measurement_table_from_measurements(appended_measurements)
            base_table = master_table
        else:
            return False

        appended_list = MeasurementList(
            list(appended_measurements),
            appended_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurements.table = (
            master_table
            if master_size == len(self.measurements)
            else concat_measurement_tables(base_table, appended_list.table)
        )

        active_start = len(self.active_measurements)
        active_view = append_active_measurement_view(
            build_active_measurement_view(
                self.active_measurements,
                table_builder=_measurement_table_from_measurements,
                materialize_measurements=False,
            ),
            appended_list,
            source_row_start=source_row_start,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=False,
        )
        self._preserve_incremental_appended_string_fields(active_view.table, active_start, appended_list.table)
        self.active_measurements = active_view.measurements
        self._active_measurement_source_count = len(self.measurements)
        self._active_measurement_source_table = self.measurements.table
        partitions = extend_measurement_partitions(
            self._active_measurement_blocks_as_partitions(),
            list(appended_list),
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            row_offset=active_start,
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
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
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
        active_table_before = getattr(self.active_measurements, "table", None)
        removed_measurement = (
            measurement_from_table_row(active_table_before, int(removed_pos))
            if active_table_before is not None and int(active_table_before.idx.size) > int(removed_pos)
            else None
        )
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        active_table = getattr(self.active_measurements, "table", None)
        if active_table is None:
            raise RuntimeError("Hybrid SE active measurement shrink requires an active MeasurementTable")
        self.active_measurements = MeasurementTableView(
            measurement_table_take(active_table, keep_rows),
            normalized=getattr(self.active_measurements, "normalized", False),
        )
        source_measurements = getattr(self, "measurements", self.active_measurements)
        self._active_measurement_source_count = len(source_measurements)
        self._active_measurement_source_table = getattr(source_measurements, "table", None)
        if not hasattr(self, "ac_meas_rows"):
            if removed_measurement is not None and self._has_real_voltage_seed_measurement([removed_measurement]):
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
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
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
        if removed_measurement is not None and self._has_real_voltage_seed_measurement([removed_measurement]):
            self._invalidate_real_voltage_seed_cache()
            self._invalidate_initial_state_cache()
        self._invalidate_measurement_activity_summary()
        return self.active_measurements

    @staticmethod
    def _empty_hybrid_code_bucket() -> "HybridStateEstimator._HybridCodeBucket":
        return HybridStateEstimator._HybridCodeBucket(
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

    def _empty_hybrid_measurement_plan(self) -> "HybridStateEstimator._HybridMeasurementPlan":
        empty_i32 = np.array([], dtype=np.int32)
        empty_i8 = np.array([], dtype=np.int8)
        empty_f64 = np.array([], dtype=np.float64)
        buckets = [self._empty_hybrid_code_bucket() for _ in range(7)]
        return self._HybridMeasurementPlan(
            dcac_rows=empty_i32,
            dcac_codes=empty_i8,
            dcac_pos=empty_i32,
            dcac_dc_v_col=empty_i32,
            dcac_ac_v_col=empty_i32,
            dcac_dc_v_default=empty_f64,
            dcac_ac_v_default=empty_f64,
            dcac_p_dc=buckets[0],
            dcac_p_ac=buckets[1],
            dcac_q_ac=buckets[2],
            dcac_v_dc=buckets[3],
            dcac_i_dc=buckets[4],
            dcac_v_ac=buckets[5],
            dcac_i_ac=buckets[6],
        )

    def _build_hybrid_measurement_plan(
        self,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        if len(block.measurements) == 0:
            return self._empty_hybrid_measurement_plan()
        dcac_code = int(DEVICE_TYPE_DCACConverter)
        meas_kind_code_by_type_code = {
            dcac_code: self._measurement_kind_code_lookup(self._DCAC_MEASUREMENT_CODE),
        }
        plan_table = build_measurement_plan_table(
            block.measurements,
            device_pos_by_type_code={},
            meas_kind_by_type_code={
                dcac_code: self._DCAC_MEASUREMENT_CODE,
            },
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
            """Build a direct DCAC terminal-voltage observation bucket."""
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

        return self._HybridMeasurementPlan(
            dcac_rows=dcac_rows,
            dcac_codes=dcac_codes,
            dcac_pos=dcac_pos,
            dcac_dc_v_col=dcac_dc_v_col,
            dcac_ac_v_col=dcac_ac_v_col,
            dcac_dc_v_default=dcac_dc_v_default,
            dcac_ac_v_default=dcac_ac_v_default,
            dcac_p_dc=dcac_p_dc_bucket,
            dcac_p_ac=dcac_p_ac_bucket,
            dcac_q_ac=dcac_q_ac_bucket,
            dcac_v_dc=dcac_v_dc_bucket,
            dcac_i_dc=dcac_i_dc_bucket,
            dcac_v_ac=dcac_v_ac_bucket,
            dcac_i_ac=dcac_i_ac_bucket,
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
        dcac_code = int(DEVICE_TYPE_DCACConverter)
        kind_lookup = np.full(max(MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC) + 1, -1, dtype=np.int16)
        kind_lookup[MEAS_TYPE_P_DC] = 0
        kind_lookup[MEAS_TYPE_P_AC] = 1
        kind_lookup[MEAS_TYPE_Q_AC] = 2
        plan_table = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={},
            meas_kind_by_type_code={},
            meas_kind_code_by_type_code={dcac_code: kind_lookup},
            require_index_arrays=True,
            table_builder=_measurement_table_from_measurements,
        )
        active_mask = (
            plan_table.handled
            & np.asarray(plan_table.table.valid, dtype=bool)
            & (np.asarray(plan_table.table.weight, dtype=np.float64) > 0.0)
        )
        dcac_mask = active_mask & (plan_table.device_type_code == dcac_code)

        dcac_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_state_col = (
            3 * plan_table.device_pos[dcac_mask].astype(np.int64, copy=False)
            + plan_table.meas_kind[dcac_mask].astype(np.int64, copy=False)
        )
        return self._HybridSeedMeasurementPlan(
            measurement_row=dcac_rows,
            state_col=dcac_state_col,
        )

    def _hybrid_converter_measurement_specs(
        self,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        def spec(device_type_code: int, device_name: str, meas_type_code: int, value: float, device_pos: int) -> "HybridStateEstimator._HybridConverterMeasurementSpec":
            return self._measurement_spec(device_type_code, int(device_pos), meas_type_code, value, str(device_name))

        for pos, conv in enumerate(self.dcac_converters):
            p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "i_p", 0.0) or 0.0)
            p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_p", 0.0) or 0.0)
            q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_q", 0.0) or 0.0)
            specs.extend(
                (
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_P_DC, p_dc, pos),
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_P_AC, p_ac, pos),
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_Q_AC, q_ac, pos),
                    spec(
                        DEVICE_TYPE_DCACConverter,
                        conv.name,
                        MEAS_TYPE_V_DC,
                        float(getattr(getattr(conv, "dc_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                    spec(
                        DEVICE_TYPE_DCACConverter,
                        conv.name,
                        MEAS_TYPE_V_AC,
                        float(getattr(getattr(conv, "ac_node_obj", None), "voltage", 1.0) or 1.0),
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
            if not np.any(dcac_mask):
                return
            state_col = (
                3 * plan.dcac_pos[dcac_mask].astype(np.int64, copy=False)
                + plan.dcac_codes[dcac_mask].astype(np.int64, copy=False)
                - 1
            )
            rows = plan.dcac_rows[dcac_mask].astype(np.int64, copy=False)
            x[state_col] = self.active_z[rows]
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
        voltage_codes = {
            MEAS_TYPE_V,
            MEAS_TYPE_V_FROM,
            MEAS_TYPE_V_TO,
            MEAS_TYPE_V_THIRD,
            MEAS_TYPE_V_GEN,
            MEAS_TYPE_V_LOAD,
            MEAS_TYPE_V_DC,
            MEAS_TYPE_V_AC,
        }
        for meas in measurements:
            if (
                bool(getattr(meas, "valid", True))
                and float(getattr(meas, "weight", 0.0)) > 0.0
                and not is_pseudo_measurement(meas)
                and int(getattr(meas, "meas_type_code", 0)) in voltage_codes
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
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            status_code = measurement_table_status_code(table)
            active_real = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
            )
            meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            device_pos = self._hybrid_measurement_device_pos_array(table)
            if meas_type_code.size == table.idx.size and device_pos.size == table.idx.size:
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                rows = np.flatnonzero(active_real & np.isin(meas_type_code, voltage_codes) & (device_pos >= 0))
                for row in rows:
                    side, node_idx = self._voltage_measurement_side_node_idx_from_code(
                        int(device_type_code[int(row)]),
                        int(device_pos[int(row)]),
                        int(meas_type_code[int(row)]),
                    )
                    if side == "ac" and node_idx is not None:
                        col = self._ac_voltage_col_for_node(int(node_idx))
                    elif side == "dc" and node_idx is not None:
                        col = self._dc_voltage_col_for_node(int(node_idx))
                    else:
                        continue
                    if col < 0:
                        continue
                    weight = float(table.weight[int(row)])
                    current = best.get(col)
                    if current is None or weight > current[0]:
                        best[col] = (weight, float(table.value[int(row)]))
        else:
            self._warn_missing_device_pos("real_voltage_seed_no_table", DEVICE_TYPE_ACNode)
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

        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE[MEAS_TYPE_V_DC], plan.dcac_rows, plan.dcac_dc_v_col)
        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE[MEAS_TYPE_V_AC], plan.dcac_rows, plan.dcac_ac_v_col)
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
            voltage, _, _, _ = delegate._unpack_state(compact)
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

    def _converter_coupled_ac_node_names(self) -> set:
        names = set()
        for conv in self.dcac_converters:
            node = getattr(conv, "ac_node_obj", None)
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
        solver_pos = pos[loc[valid].astype(np.intp, copy=False)].astype(np.int64, copy=False)
        plan_pos = np.asarray(getattr(ac, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        node_count = int(getattr(ac, "n_nodes", 0) or 0)
        if plan_pos.size and node_count > 0:
            solver_to_plan = np.full(node_count, -1, dtype=np.int64)
            valid_plan = (plan_pos >= 0) & (plan_pos < node_count)
            if np.any(valid_plan):
                solver_to_plan[plan_pos[valid_plan].astype(np.intp, copy=False)] = np.nonzero(valid_plan)[0]
                in_range = (solver_pos >= 0) & (solver_pos < solver_to_plan.size)
                mapped = np.full(solver_pos.size, -1, dtype=np.int64)
                mapped[in_range] = solver_to_plan[solver_pos[in_range].astype(np.intp, copy=False)]
                mapped = mapped[mapped >= 0]
                if mapped.size:
                    return np.unique(mapped)
        return np.unique(solver_pos)

    def _real_voltage_observation_nodes(self, side: str) -> Dict[int, float]:
        """Return side nodes covered by real usable voltage measurements on any device."""
        cache_name = f"_{side}_real_voltage_observation_node_cache"
        cache = getattr(self, cache_name, None)
        if cache is not None:
            return cache
        if side == "ac":
            sub_cache = getattr(self._ac_sub_estimator, "_real_voltage_observation_node_cache", None)
        else:
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
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            device_pos = self._hybrid_measurement_device_pos_array(table)
            if meas_type_code.size == table.idx.size and device_pos.size == table.idx.size:
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                voltage_rows = np.flatnonzero(active_real & np.isin(meas_type_code, voltage_codes) & (device_pos >= 0))
                for row in voltage_rows:
                    side_value, node_idx = self._voltage_measurement_side_node_idx_from_code(
                        int(device_type_code[int(row)]),
                        int(device_pos[int(row)]),
                        int(meas_type_code[int(row)]),
                    )
                    if side_value != side or node_idx is None:
                        continue
                    weight = float(table.weight[int(row)])
                    current = best.get(int(node_idx))
                    if current is None or weight > current[0]:
                        best[int(node_idx)] = (weight, float(table.value[int(row)]))
        else:
            self._warn_missing_device_pos("real_voltage_observation_no_table", DEVICE_TYPE_ACNode if side == "ac" else DEVICE_TYPE_DCNode)
        cache = {node_idx: value for node_idx, (_weight, value) in best.items()}
        setattr(self, cache_name, cache)
        return cache

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

    def _voltage_measurement_side_node_idx_from_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> Tuple[Optional[str], Optional[int]]:
        pos = int(device_pos)
        if pos < 0:
            return None, None
        code = int(device_type_code)
        meas_code = int(meas_type_code)
        if code == DEVICE_TYPE_DCACConverter:
            if meas_code == MEAS_TYPE_V_AC:
                nodes = np.asarray(getattr(self, "_dcac_ac_node", ()), dtype=np.int64)
                return ("ac", int(nodes[pos])) if pos < nodes.size else (None, None)
            if meas_code == MEAS_TYPE_V_DC:
                nodes = np.asarray(getattr(self, "_dcac_dc_node", ()), dtype=np.int64)
                return ("dc", int(nodes[pos])) if pos < nodes.size else (None, None)
            return None, None

        side = self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE.get(code)
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator if side == "dc" else None
        if estimator is None:
            return None, None
        node_idx = estimator.voltage_measurement_node_idx(code, pos, meas_code)
        if node_idx is None:
            return None, None
        return side, int(node_idx)

    def _voltage_pseudo_is_covered_by_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> bool:
        side, node_idx = self._voltage_measurement_side_node_idx_from_code(
            device_type_code,
            device_pos,
            meas_type_code,
        )
        if side is None or node_idx is None:
            return False
        return self._real_voltage_observation_value_for_side_node(side, node_idx) is not None

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

    def materialize_measurements(
        self,
        measurements: Optional[Sequence[Measurement]] = None,
    ) -> MeasurementList:
        """Materialize array-backed rows for diagnostics and compatibility only."""
        source = self.active_measurements if measurements is None else measurements
        table = getattr(source, "table", None)
        if table is None or int(table.idx.size) != len(source):
            return MeasurementList(list(source), normalized=getattr(source, "normalized", False))
        codes = np.asarray(table.device_type_code, dtype=np.int16)
        positions = self._hybrid_measurement_device_pos_array(table)
        meas_codes = np.asarray(
            getattr(table, "meas_type_code", np.zeros(table.idx.size, dtype=np.int16)),
            dtype=np.int16,
        )
        name_arrays = self._measurement_name_arrays_by_type_code(np.unique(codes))
        status = measurement_table_status_code(table)
        rows = []
        for row in range(int(table.idx.size)):
            measurement = measurement_from_table_row(table, row)
            type_code = int(codes[row])
            meas_code = int(meas_codes[row]) if meas_codes.size == table.idx.size else 0
            device_pos = int(positions[row]) if positions.size == table.idx.size else -1
            measurement.device_type = DEVICE_TYPE_NAMES.get(type_code, measurement.device_type)
            measurement.meas_type = MEAS_TYPE_NAMES.get(meas_code, measurement.meas_type)
            names = np.asarray(name_arrays.get(type_code, ()), dtype=object)
            if 0 <= device_pos < names.size:
                measurement.device_name = str(names[device_pos])
            if not measurement.name or measurement.name.startswith("m"):
                prefix = "pseudo" if int(status[row]) == MEAS_STATUS_PSEUDO else "measurement"
                measurement.name = f"{prefix}_{measurement.meas_type.lower()}_{measurement.device_name}"
            rows.append(measurement)
        return MeasurementList(rows, table=table, normalized=getattr(source, "normalized", False))

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
    def _converter_meas_type_code_for_state_meta(meta: StateMeta) -> int:
        mapping = {
            "dcac_p_dc": MEAS_TYPE_P_DC,
            "dcac_p_ac": MEAS_TYPE_P_AC,
            "dcac_q_ac": MEAS_TYPE_Q_AC,
        }
        return int(mapping.get(meta.kind, 0))

    def _targeted_converter_specs_for_state(
        self,
        meta: StateMeta,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        meas_type_code = self._converter_meas_type_code_for_state_meta(meta)
        if meas_type_code <= 0:
            return []
        device_type_code = int(meta.device_type_code)
        device_pos = int(meta.device_pos)
        return [
            spec
            for spec in self._hybrid_converter_measurement_specs(source=source)
            if int(spec.device_type_code) == device_type_code
            and int(spec.device_pos) == device_pos
            and int(spec.meas_type_code) == meas_type_code
        ]

    def _hybrid_direct_state_coverage_ok(self) -> bool:
        hybrid_n = int(getattr(self, "hybrid_n_state", 0))
        if hybrid_n <= 0:
            return True
        plan = getattr(self, "_active_hybrid_measurement_plan", None)
        if plan is None:
            return False
        start = int(getattr(self, "hybrid_state_start", 0))
        stop = start + hybrid_n
        covered = np.zeros(hybrid_n, dtype=bool)
        for bucket in (
            plan.dcac_p_dc,
            plan.dcac_p_ac,
            plan.dcac_q_ac,
        ):
            cols = np.asarray(bucket.jcol_p, dtype=np.int64)
            valid = (cols >= start) & (cols < stop)
            if np.any(valid):
                covered[(cols[valid] - start).astype(np.intp, copy=False)] = True
        return bool(np.all(covered))

    def _sub_block_observable_fast_certificate(
        self,
        estimator,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> bool:
        sub_n_state = int(getattr(estimator, "n_state", 0) or 0) if estimator is not None else 0
        if sub_n_state <= 0:
            return True
        if block is None or block.plan_tables is None or len(block.measurements) < sub_n_state:
            return False
        cached = getattr(estimator, "_initial_observability_cache", None)
        if (
            cached is not None
            and bool(getattr(cached, "observable", False))
            and int(getattr(cached, "state_count", -1)) == sub_n_state
        ):
            return True
        fast_builder = getattr(estimator, "_fast_active_observability_certificate", None)
        if callable(fast_builder):
            fast_result = fast_builder()
            if (
                fast_result is not None
                and bool(getattr(fast_result, "observable", False))
                and int(getattr(fast_result, "state_count", -1)) == sub_n_state
            ):
                return True
        if isinstance(estimator, DCStateEstimator) and self._dc_sub_direct_observability_certificate(estimator, block):
            return True
        try:
            result = estimator.observability_analysis(
                estimator.initial_state(),
                self._sub_measurement_runtime_input(block),
            )
        except Exception:
            return False
        return bool(getattr(result, "observable", False)) and int(getattr(result, "rank", -1)) == sub_n_state

    @staticmethod
    def _dc_sub_direct_observability_certificate(
        estimator: DCStateEstimator,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> bool:
        n_state = int(getattr(estimator, "n_state", 0) or 0)
        if n_state <= 0:
            return True
        if block is None or block.plan_tables is None or len(block.measurements) < n_state:
            return False
        try:
            plan = estimator._vector_plans_for_measurement_plan_tables(block.plan_tables)["simple"]
        except Exception:
            return False
        handled = np.asarray(plan.get("handled_mask", ()), dtype=bool)
        if handled.size != len(block.measurements) or not np.all(handled):
            return False
        covered = np.zeros(n_state, dtype=bool)

        def mark(cols, mask=None) -> None:
            values = np.asarray(cols, dtype=np.int64)
            valid = (values >= 0) & (values < n_state)
            if mask is not None:
                mask_values = np.asarray(mask, dtype=bool)
                if mask_values.size != values.size:
                    return
                valid &= mask_values
            if np.any(valid):
                covered[values[valid].astype(np.intp, copy=False)] = True

        def mask_at(masks, code: int, size: int) -> np.ndarray:
            if isinstance(masks, (tuple, list)) and 0 <= int(code) < len(masks):
                values = np.asarray(masks[int(code)], dtype=bool)
                if values.size == int(size):
                    return values
            return np.zeros(int(size), dtype=bool)

        mark(plan.get("node_col", ()))
        branch_size = np.asarray(plan.get("branch_i_col", ())).size
        branch_kind_masks = plan.get("branch_kind_masks", ())
        mark(plan.get("branch_i_col", ()), mask_at(branch_kind_masks, MEAS_TYPE_V_FROM, branch_size))
        mark(plan.get("branch_j_col", ()), mask_at(branch_kind_masks, MEAS_TYPE_V_TO, branch_size))
        load_size = np.asarray(plan.get("load_col", ())).size
        mark(plan.get("load_col", ()), mask_at(plan.get("load_kind_masks", ()), MEAS_TYPE_V_LOAD, load_size))
        gen_size = np.asarray(plan.get("gen_col", ())).size
        gen_kind_masks = plan.get("gen_kind_masks", ())
        gen_ctrl_masks = plan.get("gen_ctrl_masks", ())
        mark(plan.get("gen_col", ()), mask_at(gen_kind_masks, MEAS_TYPE_V_GEN, gen_size))
        mark(
            plan.get("gen_p_col", ()),
            mask_at(gen_kind_masks, MEAS_TYPE_P_GEN, gen_size) & mask_at(gen_ctrl_masks, DC_CTRL_V, gen_size),
        )
        switch_size = np.asarray(plan.get("switch_col", ())).size
        switch_kind_masks = plan.get("switch_kind_masks", ())
        switch_i_mask = mask_at(switch_kind_masks, MEAS_TYPE_I_FROM, switch_size) | mask_at(
            switch_kind_masks,
            MEAS_TYPE_I_TO,
            switch_size,
        )
        mark(plan.get("switch_col", ()), switch_i_mask)
        mark(plan.get("switch_i_col", ()), mask_at(switch_kind_masks, MEAS_TYPE_V_FROM, switch_size))
        mark(plan.get("switch_j_col", ()), mask_at(switch_kind_masks, MEAS_TYPE_V_TO, switch_size))
        dcdc_size = np.asarray(plan.get("dcdc_p_col", ())).size
        dcdc_kind_masks = plan.get("dcdc_kind_masks", ())
        mark(plan.get("dcdc_p_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_P_FROM, dcdc_size))
        mark(plan.get("dcdc_q_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_P_TO, dcdc_size))
        mark(plan.get("dcdc_i_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_V_FROM, dcdc_size))
        mark(plan.get("dcdc_j_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_V_TO, dcdc_size))
        return bool(np.all(covered))

    def _fast_active_observability_certificate(self) -> Optional[ObservabilityResult]:
        if not hasattr(self, "active_measurements") or not hasattr(self, "_active_measurement_blocks"):
            return None
        n_state = int(getattr(self, "n_state", 0) or 0)
        if n_state <= 0:
            return None
        measurement_count = len(self.active_measurements)
        if measurement_count < n_state:
            return None
        if targeted_redundancy_count(
            n_state,
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        ) > 0:
            return None
        if not self._hybrid_direct_state_coverage_ok():
            return None
        blocks = self._active_measurement_blocks
        if not self._sub_block_observable_fast_certificate(self._ac_sub_estimator, blocks.get("ac")):
            return None
        if not self._sub_block_observable_fast_certificate(self._dc_sub_estimator, blocks.get("dc")):
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

    def _add_required_missing_pseudo_measurements(self) -> int:
        """Add all required pseudo rows for missing P/Q/V topology states without a max-count cap."""
        delegate = self._delegate()
        if delegate is not None:
            return 0
        existing_keys = set(self._active_measurement_keys())
        required_specs = []
        power_codes_by_device = {
            DEVICE_TYPE_ACGenerator: {MEAS_TYPE_P_GEN, MEAS_TYPE_Q_GEN},
            DEVICE_TYPE_ACLoad: {MEAS_TYPE_P_LOAD, MEAS_TYPE_Q_LOAD},
            DEVICE_TYPE_ACZeroBranch: {MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM},
            DEVICE_TYPE_ACBreak: {MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM},
            DEVICE_TYPE_DCGenerator: {MEAS_TYPE_P_GEN},
            DEVICE_TYPE_DCLoad: {MEAS_TYPE_P_LOAD},
            DEVICE_TYPE_DCZeroBranch: {MEAS_TYPE_P_FROM},
            DEVICE_TYPE_DCBreak: {MEAS_TYPE_P_FROM},
            DEVICE_TYPE_DCACConverter: {MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC},
        }
        voltage_codes_by_device = {
            DEVICE_TYPE_ACNode: {MEAS_TYPE_V},
            DEVICE_TYPE_DCNode: {MEAS_TYPE_V},
        }

        def accept(spec) -> bool:
            device_type_code = int(spec.device_type_code)
            meas_type_code = int(spec.meas_type_code)
            return (
                meas_type_code in power_codes_by_device.get(device_type_code, set())
                or meas_type_code in voltage_codes_by_device.get(device_type_code, set())
            )

        for spec in self._side_observability_pseudo_specs("ac"):
            if accept(spec):
                required_specs.append(spec)
        for spec in self._side_observability_pseudo_specs("dc"):
            if accept(spec):
                required_specs.append(spec)
        for spec in self._hybrid_converter_measurement_specs(source="pseudo"):
            if accept(spec):
                required_specs.append(spec)
        if not required_specs:
            return 0

        idx_values = []
        device_type_codes = []
        device_positions = []
        meas_type_codes = []
        values = []
        next_idx = self._next_measurement_idx()
        for spec in required_specs:
            device_type_code = int(spec.device_type_code)
            meas_type_code = int(spec.meas_type_code)
            device_pos = int(spec.device_pos)
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                continue
            key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
            if key < 0 or key in existing_keys:
                continue
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                continue
            idx_values.append(next_idx)
            next_idx += 1
            device_type_codes.append(device_type_code)
            device_positions.append(device_pos)
            meas_type_codes.append(meas_type_code)
            values.append(float(spec.value))
            existing_keys.add(key)
        count = len(idx_values)
        if count == 0:
            return 0
        tail_table = MeasurementTable(
            idx=np.asarray(idx_values, dtype=np.int64),
            name=np.asarray([], dtype=object),
            device_type=np.asarray([], dtype=object),
            device_name=np.asarray([], dtype=object),
            meas_type=np.asarray([], dtype=object),
            weight=np.full(count, self.pseudo_measurement_weight, dtype=np.float64),
            valid=np.ones(count, dtype=bool),
            value=np.asarray(values, dtype=np.float64),
            device_type_code=np.asarray(device_type_codes, dtype=np.int16),
            angle_mask=np.zeros(count, dtype=bool),
            status_code=np.full(count, MEAS_STATUS_PSEUDO, dtype=np.int16),
            device_name_id=np.full(count, -1, dtype=np.int64),
            meas_type_code=np.asarray(meas_type_codes, dtype=np.int16),
            device_pos=np.asarray(device_positions, dtype=np.int64),
        )
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or int(master_table.idx.size) != len(self.measurements):
            master_table = _measurement_table_from_measurements(self.measurements)
        self.measurement_table = concat_measurement_tables(master_table, tail_table)
        self.measurements = TableBackedMeasurementList(
            self.measurement_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._invalidate_measurement_activity_summary()
        self._refresh_active_measurement_state_layout()
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        return count

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch remaining hybrid rank deficiencies without applying a pseudo-count cap."""
        delegate = self._delegate()
        if delegate is not None:
            return int(getattr(delegate, "targeted_observability_pseudo_count", 0))
        fast_observability = self._fast_active_observability_certificate()
        if fast_observability is not None:
            self._initial_observability_cache = fast_observability
            return 0
        observability = self.observability_analysis()
        if observability.observable:
            self._initial_observability_cache = observability
            return 0
        angle_col = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        active_angle_cols = angle_col[angle_col >= 0]
        if active_angle_cols.size:
            cache = self._observability_matrix_cache_for(observability, self.active_measurements, self.initial_state())
            H = cache.get("H") if cache is not None else self.jacobian_sparse(self.initial_state(), self.active_measurements)
            unanchored = unanchored_angle_state_indices(H, active_angle_cols)
            if unanchored and int(getattr(observability, "deficiency", 0) or 0) <= len(unanchored):
                return 0
        candidates = self._observability_pseudo_candidate_measurements()
        if not candidates:
            self._initial_observability_cache = observability
            return 0
        existing_keys = set(self._active_measurement_keys())
        next_idx = self._next_measurement_idx()
        selected = []
        for candidate in candidates:
            device_type_code = int(getattr(candidate, "device_type_code", 0))
            meas_type_code = int(getattr(candidate, "meas_type_code", 0))
            device_pos = int(getattr(candidate, "device_pos", -1))
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                continue
            key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
            if key < 0 or key in existing_keys:
                continue
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                continue
            candidate.idx = next_idx
            next_idx += 1
            candidate.device_type_code = device_type_code
            candidate.meas_type_code = meas_type_code
            candidate.device_pos = device_pos
            mark_measurement_pseudo(candidate)
            selected.append(candidate)
            existing_keys.add(key)
        if not selected:
            self._initial_observability_cache = observability
            return 0
        measurement_count_before = len(self.measurements)
        self._append_pseudo_measurement_objects(selected)
        appended = self.measurements[measurement_count_before:]
        if not self._incremental_update_active_measurement_state_layout(appended):
            self._refresh_active_measurement_state_layout()
        self._observability_matrix_cache = None
        self._initial_observability_cache = self.observability_analysis()
        return len(selected)

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
        if len(candidates) <= max_add:
            selected = list(candidates[:max_add])
        else:
            selected = self._select_weak_direction_pseudo_candidates(observability, candidates, max_add)
        if not selected:
            return 0
        next_idx = self._next_measurement_idx()
        valid_selected: List[Measurement] = []
        for candidate in selected:
            candidate.device_type_code = int(getattr(candidate, "device_type_code", 0))
            candidate.meas_type_code = int(getattr(candidate, "meas_type_code", 0))
            device_pos = int(getattr(candidate, "device_pos", -1))
            if candidate.device_type_code <= 0 or candidate.meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos(
                    "weak_direction_candidate",
                    candidate.device_type_code,
                    candidate.device_name,
                    candidate.meas_type_code,
                )
                continue
            candidate.idx = next_idx
            next_idx += 1
            candidate.device_pos = device_pos
            mark_measurement_pseudo(candidate)
            valid_selected.append(candidate)
        if not valid_selected:
            return 0
        measurement_count_before = len(self.measurements)
        self._append_pseudo_measurement_objects(valid_selected)
        if refresh:
            appended = self.measurements[measurement_count_before:]
            if not self._incremental_update_active_measurement_state_layout(appended):
                self._refresh_active_measurement_state_layout()
        return len(valid_selected)

    def _add_redundant_observability_pseudo_measurements(self, max_add: int) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add)

    def _add_ac_angle_reference_anchor_measurements(self) -> int:
        """Add one internal AC angle anchor for each structurally free angle component."""
        delegate = self._delegate()
        if delegate is not None:
            return 0
        ac = getattr(self, "_ac_sub_estimator", None)
        if ac is None or not hasattr(self, "active_measurements") or int(getattr(self, "n_state", 0)) <= 0:
            return 0
        cached_observability = getattr(self, "_initial_observability_cache", None)
        if (
            cached_observability is not None
            and bool(getattr(cached_observability, "observable", False))
            and int(getattr(cached_observability, "deficiency", 1)) == 0
        ):
            return 0
        angle_col = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        active_angle_cols = angle_col[angle_col >= 0]
        if active_angle_cols.size == 0:
            return 0
        node_plan_pos = np.asarray(getattr(ac, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        node_names = np.asarray(getattr(ac, "_ac_node_names", np.asarray([], dtype=object)), dtype=object)
        if node_plan_pos.size == 0 or node_names.size == 0:
            return 0
        max_solver_pos = int(node_plan_pos.max()) if node_plan_pos.size else -1
        if max_solver_pos < 0:
            return 0
        solver_to_plan = np.full(max_solver_pos + 1, -1, dtype=np.int64)
        valid_plan = (node_plan_pos >= 0) & (node_plan_pos <= max_solver_pos)
        if np.any(valid_plan):
            solver_to_plan[node_plan_pos[valid_plan].astype(np.intp, copy=False)] = np.flatnonzero(valid_plan).astype(
                np.int64,
                copy=False,
            )

        x = self.initial_state()
        H = self.jacobian_sparse(x, self.active_measurements)
        unanchored = unanchored_angle_state_indices(H, active_angle_cols)
        if not unanchored:
            return 0

        existing_keys = self._active_measurement_keys()
        measurements: List[Measurement] = []
        next_idx = self._next_measurement_idx()
        seen_state_cols = set()
        for state_col_raw in unanchored:
            state_col = int(state_col_raw)
            if state_col in seen_state_cols:
                continue
            seen_state_cols.add(state_col)
            solver_pos = -1
            meta = state_meta_at(self.state_meta, state_col)
            meta_pos = int(getattr(meta, "device_pos", -1)) if meta is not None else -1
            if 0 <= meta_pos < angle_col.size and int(angle_col[meta_pos]) == state_col:
                solver_pos = meta_pos
            if solver_pos < 0:
                matched = np.flatnonzero(angle_col == state_col)
                if matched.size == 0:
                    continue
                solver_pos = int(matched[0])
            if solver_pos < 0 or solver_pos >= solver_to_plan.size:
                continue
            device_pos = int(solver_to_plan[solver_pos])
            if device_pos < 0:
                continue
            device_name = str(node_names[device_pos]) if device_pos < node_names.size else f"ac_node_{solver_pos}"
            key = self._packed_measurement_key(DEVICE_TYPE_ACNode, device_pos, MEAS_TYPE_ANGLE)
            pseudo_name = f"pseudo_ref_angle_{device_name}"
            if key in existing_keys:
                continue
            value = float(x[state_col]) if 0 <= state_col < x.size else 0.0
            measurements.append(
                Measurement(
                    next_idx,
                    pseudo_name,
                    "ACNode",
                    device_name,
                    "ANGLE",
                    self.pseudo_measurement_weight,
                    True,
                    value,
                    MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_ACNode,
                    meas_type_code=MEAS_TYPE_ANGLE,
                    device_pos=device_pos,
                )
            )
            next_idx += 1
            existing_keys.add(key)

        if not measurements:
            return 0
        measurement_count_before = len(self.measurements)
        self._append_pseudo_measurement_objects(measurements)
        appended = self.measurements[measurement_count_before:]
        if not self._incremental_update_active_measurement_state_layout(appended):
            self._refresh_active_measurement_state_layout()
        return len(measurements)

    def _observability_pseudo_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        cache = getattr(self, "_observability_pseudo_candidate_cache", None)
        if cache is not None:
            return cache
        existing_keys = self._active_measurement_keys()
        candidates: List[Measurement] = []

        def add(
            device_name: str,
            value: float,
            device_type_code: int,
            device_pos: int = -1,
            meas_type_code: int = 0,
        ) -> None:
            device_type_code = int(device_type_code)
            meas_type_code = int(meas_type_code)
            device_pos = int(device_pos)
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos("observability_candidate", device_type_code, device_name, meas_type_code)
                return
            key = self._packed_measurement_key(device_type_code, int(device_pos), meas_type_code)
            device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
            meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                return
            if key < 0 or key in existing_keys:
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
                    device_type_code=device_type_code,
                    meas_type_code=meas_type_code,
                    device_pos=device_pos,
                )
            )
            existing_keys.add(key)

        for spec in self._side_observability_pseudo_specs("ac"):
            add(
                spec.device_name,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )
        for spec in self._side_observability_pseudo_specs("dc"):
            add(
                spec.device_name,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )

        for spec in self._hybrid_converter_measurement_specs(source="flow"):
            add(
                spec.device_name,
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
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator if side == "dc" else None
        if estimator is None:
            return []
        candidates = estimator._observability_pseudo_candidate_measurements()
        table = getattr(candidates, "table", None)
        if table is None:
            raise RuntimeError(f"{side.upper()} observability candidates must expose an array measurement table")
        row_count = int(table.idx.size)
        if row_count == 0:
            return []
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        device_pos = np.asarray(table.device_pos, dtype=np.int64)
        meas_type_code = np.asarray(table.meas_type_code, dtype=np.int16)
        values = np.asarray(table.value, dtype=np.float64)
        if not all(array.size == row_count for array in (device_type_code, device_pos, meas_type_code, values)):
            raise RuntimeError(f"{side.upper()} observability candidate arrays have inconsistent sizes")
        names_by_code = self._measurement_name_arrays_by_type_code(np.unique(device_type_code))
        table_names = np.asarray(table.device_name, dtype=object)
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        for row in range(row_count):
            code = int(device_type_code[row])
            pos = int(device_pos[row])
            names = np.asarray(names_by_code.get(code, ()), dtype=object)
            if 0 <= pos < names.size:
                device_name = str(names[pos])
            elif table_names.size == row_count:
                device_name = str(table_names[row])
            else:
                device_name = ""
            specs.append(
                self._measurement_spec(
                    code,
                    pos,
                    int(meas_type_code[row]),
                    float(values[row]),
                    device_name,
                )
            )
        return specs

    def _targeted_side_specs_for_state(
        self,
        meta: StateMeta,
        existing_keys: set,
    ) -> List["_HybridConverterMeasurementSpec"]:
        name = meta.device_name
        device_type_code = int(meta.device_type_code)
        device_pos = int(meta.device_pos)
        if device_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("targeted_side_state", device_type_code, name, meta.meas_type_code)
            return []
        if meta.side == "ac" and meta.kind in ("zero_current", "break_current"):
            if device_type_code not in (DEVICE_TYPE_ACZeroBranch, DEVICE_TYPE_ACBreak):
                return []
            p_from_key = self._packed_measurement_key(device_type_code, device_pos, MEAS_TYPE_P_FROM)
            q_from_key = self._packed_measurement_key(device_type_code, device_pos, MEAS_TYPE_Q_FROM)
            p_type_code = MEAS_TYPE_P_TO if p_from_key in existing_keys else MEAS_TYPE_P_FROM
            q_type_code = MEAS_TYPE_Q_TO if q_from_key in existing_keys else MEAS_TYPE_Q_FROM
            return [
                self._measurement_spec(device_type_code, device_pos, p_type_code, 0.0, name),
                self._measurement_spec(device_type_code, device_pos, q_type_code, 0.0, name),
            ]
        if meta.side == "dc" and meta.kind in ("zero_current", "break_current"):
            if device_type_code not in (DEVICE_TYPE_DCZeroBranch, DEVICE_TYPE_DCBreak):
                return []
            return [
                self._measurement_spec(
                    device_type_code,
                    device_pos,
                    MEAS_TYPE_I_FROM,
                    0.0,
                    name,
                )
            ]
        return []

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
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

        def add(
            device_type_code: int,
            meas_type_code: int,
            value: float,
            device_pos: int = -1,
        ) -> None:
            nonlocal next_idx, added
            if added >= max_add:
                return
            device_type_code = int(device_type_code)
            meas_type_code = int(meas_type_code)
            device_pos = int(device_pos if int(device_pos) >= 0 else meta.device_pos)
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos("targeted_add", device_type_code, name, meas_type_code)
                return
            key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
            device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
            meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{name}"
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                return
            if key < 0 or key in existing_keys:
                return
            new_next_idx = self._append_pseudo_measurement(
                next_idx,
                pseudo_name,
                device_type,
                name,
                meas_type,
                value,
                device_type_code=device_type_code,
                device_pos=device_pos,
                meas_type_code=meas_type_code,
            )
            if new_next_idx == next_idx:
                return
            next_idx = new_next_idx
            existing_keys.add(key)
            added += 1

        if meta.kind == "voltage" and int(meta.device_type_code) == DEVICE_TYPE_ACNode and int(meta.device_pos) >= 0:
            ac = self._ac_sub_estimator
            solver_pos = np.asarray(getattr(ac, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
            values = np.asarray(getattr(ac, "file_voltage", np.asarray([], dtype=np.float64)), dtype=np.float64)
            device_pos = int(meta.device_pos)
            voltage = 1.0
            if device_pos < solver_pos.size:
                node_pos = int(solver_pos[device_pos])
                if 0 <= node_pos < values.size:
                    voltage = float(values[node_pos])
            add(
                meta.device_type_code,
                MEAS_TYPE_V,
                voltage,
                meta.device_pos,
            )
            return next_idx, added
        if meta.kind == "voltage" and int(meta.device_type_code) == DEVICE_TYPE_DCNode and int(meta.device_pos) >= 0:
            dc = self._dc_sub_estimator
            solver_pos = np.asarray(getattr(dc, "_raw_node_solver_pos_alive", np.asarray([], dtype=np.int64)), dtype=np.int64)
            values = np.asarray(getattr(dc, "_node_voltage_by_pos", np.asarray([], dtype=np.float64)), dtype=np.float64)
            device_pos = int(meta.device_pos)
            voltage = 1.0
            if device_pos < solver_pos.size:
                node_pos = int(solver_pos[device_pos])
                if 0 <= node_pos < values.size:
                    voltage = float(values[node_pos])
            add(
                meta.device_type_code,
                MEAS_TYPE_V,
                voltage,
                meta.device_pos,
            )
            return next_idx, added
        converter_specs = self._targeted_converter_specs_for_state(meta, source="flow")
        if converter_specs:
            for spec in converter_specs:
                add(
                    spec.device_type_code,
                    spec.meas_type_code,
                    spec.value,
                    spec.device_pos,
                )
            return next_idx, added
        side_specs = self._targeted_side_specs_for_state(meta, existing_keys)
        if side_specs:
            for spec in side_specs:
                add(
                    spec.device_type_code,
                    spec.meas_type_code,
                    spec.value,
                    spec.device_pos,
                )
        return next_idx, added

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        table = getattr(measurements, "table", None)
        if table is not None and int(np.asarray(getattr(table, "idx", np.asarray([]))).size) == len(measurements):
            return measurements
        if isinstance(measurements, list):
            return measurements
        raise RuntimeError("Hybrid SE requires table-backed measurements; Measurement object iteration is disabled.")

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
    def _measurement_signature(measurements: Sequence[Measurement]) -> Tuple[Tuple[int, int, int, float, float, bool], ...]:
        return tuple(
            (
                int(getattr(meas, "device_type_code", 0)),
                int(getattr(meas, "device_pos", -1)),
                int(getattr(meas, "meas_type_code", 0)),
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

    @staticmethod
    def _lookup_node_pos_array_by_idx(node_idx: np.ndarray, ids: np.ndarray, pos: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        if nodes.size == 0 or ids.size == 0 or pos.size == 0:
            return out
        loc = np.searchsorted(ids, nodes)
        in_range = loc < ids.size
        if np.any(in_range):
            rows = np.flatnonzero(in_range)
            matched = ids[loc[rows].astype(np.intp, copy=False)] == nodes[rows]
            if np.any(matched):
                matched_rows = rows[matched]
                values = pos[loc[matched_rows].astype(np.intp, copy=False)]
                valid = values >= 0
                out[matched_rows[valid]] = values[valid]
        return out

    def _ac_node_pos_for_idx(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        if ac is None:
            return -1
        return self._lookup_node_pos_by_idx(
            int(node_idx),
            getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    def _ac_node_pos_for_idx_array(self, node_idx: np.ndarray) -> np.ndarray:
        ac = self._ac_sub_estimator
        if ac is None:
            return np.full(np.asarray(node_idx, dtype=np.int64).size, -1, dtype=np.int64)
        return self._lookup_node_pos_array_by_idx(
            node_idx,
            getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    @staticmethod
    def _measurement_kind_code_lookup(kind_by_code: Dict[int, int]) -> np.ndarray:
        if not kind_by_code:
            return np.asarray([], dtype=np.int16)
        valid_codes = [int(code) for code in kind_by_code if int(code) >= 0]
        if not valid_codes:
            return np.asarray([], dtype=np.int16)
        lookup = np.empty(max(valid_codes) + 1, dtype=np.int16)
        lookup.fill(-1)
        for code, kind in kind_by_code.items():
            code = int(code)
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

    def _dc_node_pos_for_idx_array(self, node_idx: np.ndarray) -> np.ndarray:
        dc = self._dc_sub_estimator
        if dc is None:
            return np.full(np.asarray(node_idx, dtype=np.int64).size, -1, dtype=np.int64)
        return self._lookup_node_pos_array_by_idx(
            node_idx,
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

    def _ac_voltage_cols_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int32)
        ac = self._ac_sub_estimator
        if ac is None or nodes.size == 0:
            return out
        pos = self._ac_node_pos_for_idx_array(nodes)
        voltage_col = np.asarray(getattr(ac, "voltage_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        rows = np.flatnonzero((pos >= 0) & (pos < voltage_col.size))
        if rows.size:
            cols = voltage_col[pos[rows].astype(np.intp, copy=False)]
            valid = cols >= 0
            out[rows[valid]] = cols[valid]
        return out

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
        node = getattr(getattr(self.network, "ac", None), "node_dict", {}).get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _ac_voltage_defaults_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        ac = self._ac_sub_estimator
        if ac is None or nodes.size == 0:
            return out
        pos = self._ac_node_pos_for_idx_array(nodes)
        valid_pos = pos >= 0
        file_voltage = getattr(ac, "file_voltage", None)
        if file_voltage is not None:
            voltage = np.asarray(file_voltage, dtype=np.float64)
            rows = np.flatnonzero(valid_pos & (pos < voltage.size))
            if rows.size:
                values = voltage[pos[rows].astype(np.intp, copy=False)]
                out[rows] = np.where(values != 0.0, values, 1.0)
        for ref_pos, value in (getattr(ac, "ref_voltages", {}) or {}).items():
            rows = np.flatnonzero(valid_pos & (pos == int(ref_pos)))
            if rows.size:
                out[rows] = float(value)
        invalid_rows = np.flatnonzero(~valid_pos)
        node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
        for row in invalid_rows:
            node = node_dict.get(int(nodes[row]))
            out[row] = float(getattr(node, "voltage", 1.0) or 1.0)
        return out

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
        node = getattr(getattr(self.network, "dc", None), "node_dict", {}).get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _dc_voltage_defaults_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        dc = self._dc_sub_estimator
        if dc is None or nodes.size == 0:
            return out
        pos = self._dc_node_pos_for_idx_array(nodes)
        valid_pos = pos >= 0
        node_voltage = getattr(dc, "_node_voltage_by_pos", None)
        if node_voltage is not None:
            voltage = np.asarray(node_voltage, dtype=np.float64)
            rows = np.flatnonzero(valid_pos & (pos < voltage.size))
            if rows.size:
                values = voltage[pos[rows].astype(np.intp, copy=False)]
                out[rows] = np.where(values != 0.0, values, 1.0)
        for ref_pos, value in (getattr(dc, "ref_voltages", {}) or {}).items():
            rows = np.flatnonzero(valid_pos & (pos == int(ref_pos)))
            if rows.size:
                out[rows] = float(value)
        invalid_rows = np.flatnonzero(~valid_pos)
        if invalid_rows.size:
            node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
            for row in invalid_rows:
                node = node_dict.get(int(nodes[row]))
                out[row] = float(getattr(node, "voltage", 1.0) or 1.0)
        return out

    def _dc_voltage_col_for_node(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        pos = self._dc_node_pos_for_idx(int(node_idx))
        if dc is None or pos < 0:
            return -1
        col = int(dc.voltage_col[pos])
        return self.dc_state_start + col if col >= 0 else -1

    def _dc_voltage_cols_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int32)
        dc = self._dc_sub_estimator
        if dc is None or nodes.size == 0:
            return out
        pos = self._dc_node_pos_for_idx_array(nodes)
        voltage_col = np.asarray(getattr(dc, "voltage_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        rows = np.flatnonzero((pos >= 0) & (pos < voltage_col.size))
        if rows.size:
            cols = voltage_col[pos[rows].astype(np.intp, copy=False)]
            valid = cols >= 0
            out[rows[valid]] = (self.dc_state_start + cols[valid]).astype(np.int32, copy=False)
        return out

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

        # DCAC voltage-only codes (4 V_DC, 6 V_AC).
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
            values[ac_block.rows] = self._ac_sub_estimator.evaluate(
                ac_x,
                self._sub_measurement_runtime_input(ac_block),
            )
        if dc_block.measurements and self._dc_sub_estimator is not None:
            values[dc_block.rows] = self._dc_sub_estimator.evaluate(
                dc_x,
                self._sub_measurement_runtime_input(dc_block),
            )
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
        builder = getattr(estimator, "_jacobian_builder", None)
        previous_fixed_pattern = getattr(builder, "_assume_fixed_pattern", None)
        if builder is not None:
            builder._assume_fixed_pattern = False
        try:
            sub_csr = estimator.jacobian_sparse(sub_x, self._sub_measurement_runtime_input(block))
        finally:
            if builder is not None and previous_fixed_pattern is not None:
                builder._assume_fixed_pattern = previous_fixed_pattern
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

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[Sequence[Measurement]] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        if self.fluid_estimators and measurements is None and normal_matrix is None and normal_factor_diag is None:
            return self._multi_energy_observability_analysis(x=x, H=H)
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
        if (
            normal_matrix is None
            and normal_factor_diag is None
            and is_sparse_matrix(H)
            and int(H.shape[0]) >= int(self.n_state)
        ):
            structural_rank = sparse_structural_rank(H)
            if structural_rank == self.n_state:
                ac_angle_cols = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
                unanchored_angles = unanchored_angle_state_indices(H, ac_angle_cols[ac_angle_cols >= 0])
                if not unanchored_angles:
                    result = ObservabilityResult(
                        observable=True,
                        rank=self.n_state,
                        state_count=self.n_state,
                        measurement_count=len(measurements),
                        deficiency=0,
                        singular_values=np.array([], dtype=np.float64),
                        weak_states=[],
                    )
                    self._cache_observability_matrix(result, x, measurements, H)
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
        if not self._prepared:
            self.prepare()
        if self.fluid_estimators and measurements is None:
            return self._estimate_multi_energy(
                x0=x0,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
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
            if dx.size and not np.all(np.isfinite(dx)):
                diag_scale = 1.0
                if is_sparse_matrix(gain):
                    diag_values = np.asarray(gain.diagonal(), dtype=np.float64)
                    finite_diag = diag_values[np.isfinite(diag_values) & (diag_values > 0.0)]
                    if finite_diag.size:
                        diag_scale = float(np.median(finite_diag))
                    identity = sparse_eye(gain.shape[0], format="csc", dtype=np.float64)
                    for damping_factor in (1e-10, 1e-8, 1e-6, 1e-4):
                        damped_gain = gain + identity * (diag_scale * damping_factor)
                        dx_candidate, normal_factor_diag = NormalEquationSolver().solve(
                            damped_gain,
                            rhs,
                            return_factor_diag=final_diagnostics,
                        )
                        if dx_candidate.size == 0 or np.all(np.isfinite(dx_candidate)):
                            dx = dx_candidate
                            self._record_profile_time("solve.damped_normal_fallback", damping_factor)
                            break
                else:
                    diag_values = np.diag(gain)
                    finite_diag = diag_values[np.isfinite(diag_values) & (diag_values > 0.0)]
                    if finite_diag.size:
                        diag_scale = float(np.median(finite_diag))
                    for damping_factor in (1e-10, 1e-8, 1e-6, 1e-4):
                        damped_gain = np.array(gain, dtype=np.float64, copy=True)
                        damped_gain[np.diag_indices_from(damped_gain)] += diag_scale * damping_factor
                        dx_candidate, normal_factor_diag = NormalEquationSolver().solve(
                            damped_gain,
                            rhs,
                            return_factor_diag=final_diagnostics,
                        )
                        if dx_candidate.size == 0 or np.all(np.isfinite(dx_candidate)):
                            dx = dx_candidate
                            self._record_profile_time("solve.damped_normal_fallback", damping_factor)
                            break
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
        result_measurements = []
        if not array_only_result:
            table = getattr(measurements, "table", None)
            result_measurements = measurements if table is not None and int(table.idx.size) == len(measurements) else list(measurements)
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
            measurements=result_measurements,
            observability=observability,
            measurement_table=getattr(measurements, "table", None),
        )

    def identify_bad_data(
        self,
        result: EstimateResult,
        threshold: Optional[float] = None,
    ) -> Tuple[List[BadDataItem], np.ndarray]:
        profile_start = time.perf_counter() if self.profile_enabled else None
        delegate = self._delegate()
        if delegate is not None:
            return delegate.identify_bad_data(result, threshold)
        threshold = self.params.bad_threshold if threshold is None else threshold
        if result.residual.size and threshold > 0.0:
            normalized_upper_bound = np.abs(result.residual) / np.sqrt(1e-12)
            if float(normalized_upper_bound.max()) <= float(threshold):
                self._record_profile_time("bad_data.fast_residual_bound", 0.0)
                if profile_start is not None:
                    self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
                return [], normalized_upper_bound
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
        if self.fluid_estimators and result is self.multi_energy_estimate_result:
            if bad_items is None or normalized_residual is None:
                computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
                if bad_items is None:
                    bad_items = computed_bad_items
                if normalized_residual is None:
                    normalized_residual = computed_normalized
            self.se_result = self._build_result_from_table(
                result,
                bad_items=bad_items,
                normalized_residual=np.asarray(normalized_residual, dtype=np.float64),
                result_mode=mode,
                all_measurement_table=self._multi_energy_measurement_table,
            )
            return self.se_result
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
            )
            return self.se_result
        self.se_result = build_seresult_full_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurement_table=getattr(self.measurements, "table", None),
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
        if not self._prepared:
            self.prepare()
        if self.fluid_estimators:
            return self._run_multi_energy(
                result_mode=result_mode,
                remove_bad_data=remove_bad_data,
                bad_threshold=bad_threshold,
                max_remove=max_remove,
                skip_bad_data=skip_bad_data,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
        electric_result = self._run_electric(
            result_mode=result_mode,
            remove_bad_data=remove_bad_data,
            bad_threshold=bad_threshold,
            max_remove=max_remove,
            skip_bad_data=skip_bad_data,
            verbose=verbose,
            final_diagnostics=final_diagnostics,
            observability=observability,
        )
        self._run_fluid_estimators()
        self._build_multi_energy_result(electric_result)
        return electric_result

    def _run_electric(
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
        if not self._prepared:
            self.prepare()
        if self._fluid_only:
            return None
        mode = normalize_seresult_result_mode(result_mode)
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        array_only = mode == "array"
        if array_only and remove_bad_data:
            raise ValueError("result_mode='array' cannot be combined with remove_bad_data=True")
        delegate = self._delegate()
        if delegate is not None:
            result = delegate.run(
                result_mode=mode,
                remove_bad_data=remove_bad_data,
                bad_threshold=threshold,
                max_remove=max_remove,
                skip_bad_data=skip_bad_data,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
            self.observability_result = getattr(delegate, "observability_result", None)
            self.estimate_result = getattr(delegate, "estimate_result", None)
            self.removed_bad_data = list(getattr(delegate, "removed_bad_data", []) or [])
            self.bad_items = list(getattr(delegate, "bad_items", []) or [])
            self.normalized_residual = np.asarray(
                getattr(delegate, "normalized_residual", np.array([], dtype=np.float64)),
                dtype=np.float64,
            )
            self.se_result = getattr(delegate, "se_result", None)
            return result
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

    def _run_fluid_estimators(self) -> None:
        self.fluid_se_rc = {}
        self.fluid_se_errors = {}
        for name, estimator in self.fluid_estimators.items():
            try:
                rc = int(estimator.run())
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                rc = 1
                self.fluid_se_errors[name] = str(exc)
            self.fluid_se_rc[name] = rc
            if rc != 0 and name not in self.fluid_se_errors:
                result = getattr(estimator, "result", None)
                self.fluid_se_errors[name] = (
                    "fluid state estimation did not converge"
                    if result is None
                    else f"fluid state estimation residual={float(result.residual_inf):.6e}"
                )

    def _build_multi_energy_coupling_results(self) -> List[object]:
        plan_by_key = {
            (plan.coupling.table_name, int(plan.coupling.idx)): plan
            for plan in self.multi_energy_se_coupling_plans
        }
        endpoint_values = {}
        if self.multi_energy_estimate_result is not None and self._multi_energy_se_layout_ready:
            endpoint_values, _endpoint_jacobians = self._multi_energy_endpoint_runtime(
                self.multi_energy_estimate_result.x,
                with_jacobian=False,
            )
        results = []
        for coupling in self.multi_energy.couplings:
            plan = plan_by_key.get((coupling.table_name, int(coupling.idx)))
            domains_ready = all(
                terminal.domain in {"ac", "dc"}
                or terminal.domain in self.fluid_estimators
                for terminal in (coupling.t1, coupling.t2)
            )
            t1_value = None
            t2_value = None
            residual = None
            if not coupling.active:
                status = "inactive"
            elif plan is not None and endpoint_values:
                if plan.electric_heat:
                    dependent_value = float(
                        self.multi_energy_estimate_result.x[plan.control_col]
                    )
                    source_flow = float(
                        endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                    )
                    t_return = float(
                        self.multi_energy_estimate_result.x[plan.return_temperature_col]
                    )
                    if coupling.control_type == "P":
                        electric_value = plan.electric_setpoint
                        t_out = dependent_value
                    else:
                        electric_value = dependent_value
                        t_out = plan.supply_temperature_set
                    if plan.t1 == plan.electric:
                        t1_value, t2_value = electric_value, t_out
                    else:
                        t1_value, t2_value = t_out, electric_value
                    residual = electric_heat_balance_residual(
                        coupling,
                        electric_value,
                        source_flow,
                        t_out,
                        t_return,
                        plan.heat_capacity,
                        plan.power_scale,
                    )
                elif plan.hydrogen_electric:
                    dependent_value = (
                        float(self.multi_energy_estimate_result.x[plan.control_col])
                        if plan.control_col >= 0
                        else float(
                            endpoint_values[plan.dependent.provider][plan.dependent.row]
                        )
                    )
                    controlled_value = float(plan.controlled_setpoint)
                    if plan.t1 == plan.controlled:
                        t1_value, t2_value = controlled_value, dependent_value
                    else:
                        t1_value, t2_value = dependent_value, controlled_value
                    electric_value = (
                        t1_value if plan.t1.domain in {"ac", "dc"} else t2_value
                    )
                    hydro_value = t1_value if plan.t1.domain == "hydro" else t2_value
                    residual = hydrogen_electric_balance_residual(
                        coupling,
                        electric_value,
                        hydro_value,
                        float(getattr(self.network, "p_base_kW", self.p_base_kW)),
                    )
                else:
                    t1_value = (
                        float(self.multi_energy_estimate_result.x[plan.control_col])
                        if plan.control_col >= 0
                        else float(endpoint_values[plan.t1.provider][plan.t1.row])
                    )
                    t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
                    residual = (
                        abs(t2_value) * plan.t2_scale
                        - float(coupling.efficiency) * abs(t1_value) * plan.t1_scale
                    )
                normalized_tolerance = max(
                    10.0 * self.tol,
                    3.0 / np.sqrt(max(self.multi_energy_coupling_weight, 1.0)),
                )
                tolerance = max(1.0e-7, normalized_tolerance * plan.normalization)
                status = "balanced" if abs(residual) <= tolerance else "mismatch"
            elif domains_ready:
                status = "associated"
            else:
                status = "unavailable"
            results.append(
                SimpleNamespace(
                    coupling=coupling,
                    table_name=coupling.table_name,
                    idx=coupling.idx,
                    name=coupling.name,
                    active=coupling.active,
                    control_type=coupling.control_type,
                    e2h_coeff=coupling.e2h_coeff,
                    h2e_coeff=coupling.h2e_coeff,
                    efficiency=coupling.efficiency,
                    energy_factor=coupling.energy_factor,
                    t1_value=t1_value,
                    t2_value=t2_value,
                    source_flow=(
                        source_flow
                        if plan is not None and plan.electric_heat and endpoint_values
                        else None
                    ),
                    supply_temperature=(
                        t_out
                        if plan is not None and plan.electric_heat and endpoint_values
                        else None
                    ),
                    return_temperature=(
                        t_return
                        if plan is not None and plan.electric_heat and endpoint_values
                        else None
                    ),
                    residual=residual,
                    status=status,
                )
            )
        return results

    def _build_multi_energy_result(self, electric_result: Optional[SEResult]) -> MultiEnergySEResult:
        fluid_results = {
            name: estimator.se_result
            for name, estimator in self.fluid_estimators.items()
            if getattr(estimator, "se_result", None) is not None
        }
        electric_converged = self._fluid_only or bool(
            self.estimate_result is not None and self.estimate_result.converged
        )
        electric_observable = self._fluid_only or bool(
            self.observability_result is not None and self.observability_result.observable
        )
        fluid_converged = all(
            self.fluid_se_rc.get(name, 1) == 0
            and getattr(estimator, "result", None) is not None
            and estimator.result.converged
            for name, estimator in self.fluid_estimators.items()
        )
        fluid_observable = all(
            getattr(estimator, "result", None) is not None
            and estimator.result.observability.observable
            for estimator in self.fluid_estimators.values()
        )
        result = MultiEnergySEResult(
            electric=electric_result,
            fluids=fluid_results,
            fluid_estimators=dict(self.fluid_estimators),
            couplings=self._build_multi_energy_coupling_results(),
            warnings=list(self.multi_energy.warnings),
            errors=dict(self.fluid_se_errors),
            converged=bool(electric_converged and fluid_converged),
            observable=bool(electric_observable and fluid_observable),
        )
        self.multi_energy_result = result
        if self.se_result is not None:
            self.se_result.multi_energy = result
        return result

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        if self.fluid_estimators:
            threshold_value = self.params.bad_threshold if threshold is None else float(threshold)
            return self._estimate_multi_energy_with_bad_data_removal(
                threshold=threshold_value,
                max_remove=max_remove,
                verbose=verbose,
            )
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
            for k, conv in enumerate(self.dcac_converters[:limit]):
                base = 3 * k
                print(
                    f"  {conv.name:14s} "
                    f"P_DC={hybrid_x[base]:.9f} P_AC={hybrid_x[base + 1]:.9f} Q_AC={hybrid_x[base + 2]:.9f}"
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
    for side, sub_estimator in estimator.fluid_estimators.items():
        items.extend(
            (f"{side}.{name}", value)
            for name, value in getattr(sub_estimator, "profile_times", {}).items()
        )
    return sorted(items)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid electric/heat/gas/hydrogen/steam weighted least-squares state estimation."
    )
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
    parser.add_argument(
        "--result-mode",
        default=None,
        choices=("full", "summary", "array", "none"),
        help="SEResult payload mode: full, summary, array, or none.",
    )
    parser.add_argument(
        "--power-flow-linear-solver",
        default=None,
        help="Sparse solver used by AC/DC load-flow seeds. Default: umfpack for hybrid cases with DC/coupling.",
    )
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
        power_flow_linear_solver=args.power_flow_linear_solver,
    )
    estimator.prepare()
    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    estimator.run(
        result_mode=args.result_mode if args.result_mode is not None else "none",
        remove_bad_data=args.remove_bad_data,
        bad_threshold=bad_threshold,
        max_remove=args.max_remove,
        verbose=not args.quiet,
    )
    if estimator.observability_result is not None:
        _print_observability(estimator.observability_result)
    result = estimator.estimate_result
    removed = estimator.removed_bad_data
    if removed:
        print("Removed bad data:")
        for item in removed:
            print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")
    bad_items = estimator.bad_items
    normalized = estimator.normalized_residual
    if result is not None:
        print(
            "State estimation: "
            f"converged={result.converged}, iterations={result.iterations}, "
            f"objective={result.objective:.6e}, max_dx={result.max_correction:.3e}, "
            f"residual_inf={result.residual_inf:.3e}"
        )
        _print_bad_data(bad_items, normalized, bad_threshold)
    if estimator.fluid_estimators:
        print("Fluid state estimation:")
        for name, fluid_estimator in estimator.fluid_estimators.items():
            fluid_result = fluid_estimator.result
            print(
                f"  {name}: converged={fluid_result.converged}, "
                f"observable={fluid_result.observability.observable}, "
                f"rank={fluid_result.observability.rank}/{fluid_result.observability.state_count}, "
                f"iterations={fluid_result.iterations}, residual={fluid_result.residual_inf:.6e}, "
                f"bad_data={len(fluid_estimator.bad_data)}"
            )
    if estimator.multi_energy_result.couplings:
        print("Multi-energy couplings:")
        for item in estimator.multi_energy_result.couplings:
            print(f"  {item.table_name}:{item.name}: status={item.status}")
    print(
        "Multi-energy result: "
        f"converged={estimator.multi_energy_result.converged}, "
        f"observable={estimator.multi_energy_result.observable}"
    )
    if args.print_state and result is not None:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in _profile_time_items(estimator):
            print(f"  {name}={value:.6f}s")
    return 0 if estimator.multi_energy_result.converged and estimator.multi_energy_result.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
