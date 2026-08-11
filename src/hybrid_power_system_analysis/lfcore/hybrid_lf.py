import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
LFCORE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, LFCORE_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_lf import (
    ACLFResult,
    AC_NODE_TYPE_PQ,
    ACPowerFlowCalc,
    ac_node_type_label,
    coo_matrix,
    csc_matrix,
    csr_matrix,
)
try:
    from lfcore.common import device_key as _lf_device_key, normalize_result_mode as _normalize_lf_result_mode
except ImportError:  # pragma: no cover - direct script import path
    from common import device_key as _lf_device_key, normalize_result_mode as _normalize_lf_result_mode
try:
    from lfcore.solver_common import (
        OPTIONAL_SPARSE_MISSING as _AC_OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _AC_OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        make_reusable_factorizer as _make_reusable_factorizer,
        resolve_linear_solver as _resolve_linear_solver,
    )
except ImportError:  # pragma: no cover - direct script import path
    from solver_common import (
        OPTIONAL_SPARSE_MISSING as _AC_OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _AC_OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        make_reusable_factorizer as _make_reusable_factorizer,
        resolve_linear_solver as _resolve_linear_solver,
    )
from scipy.sparse.linalg import spsolve as _scipy_spsolve
from dc_lf import DCLFResult, DCPowerFlowCalc
from gas_lf import GasPowerFlowCalc
from heat_lf import HeatPowerFlowCalc
from hydro_lf import HydroPowerFlowCalc
from steam_lf import SteamPowerFlowCalc
from scipy.sparse import block_diag
try:
    from _sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
except ImportError:  # pragma: no cover - package import path
    from ._sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from hybrid_model import HybridPowerNetwork
from model.multi_energy_model import (
    EnergyCoupling,
    MultiEnergyContext,
    attach_multi_energy_context,
    build_multi_energy_context_from_rows,
    electric_heat_balance_residual,
    hydrogen_electric_balance_residual,
    hydrogen_electric_dependent_value,
)
from ac_array_model import (
    ACAC_COLS,
    ACAC_SIDE_CONTROL_LABEL,
    BRANCH_COLS as AC_BRANCH_COLS,
    BREAK_COLS as AC_BREAK_COLS,
    BUS_COLS as AC_BUS_COLS,
    GEN_COLS as AC_GEN_COLS,
    LOAD_COLS as AC_LOAD_COLS,
    SHUNT_COLS as AC_SHUNT_COLS,
    SWITCH_COLS as AC_SWITCH_COLS,
    THREE_WINDING_TRANSFORMER_COLS as AC_THREE_WINDING_TRANSFORMER_COLS,
    TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
    ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
    build_ac_ppc_from_network,
    build_ac_ppc_from_mat_file,
)
from dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    BUS_COLS as DC_BUS_COLS,
    DCDC_COLS as DC_DCDC_COLS,
    DCDC_SIDE_CONTROL_LABEL,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    build_dc_ppc_from_network,
    dcdc_legacy_control_label,
)
from efile_read import _read_efile_rows
from hybrid_array_model import (
    DCAC_COLS,
    DCAC_AC_CONTROL_CODE,
    DCAC_AC_CONTROL_LABEL,
    DCAC_DC_CONTROL_CODE,
    DCAC_DC_CONTROL_LABEL,
    DCAC_DEVICE_TYPE_LABEL,
    validate_dcac_control_types,
)
from model.ppc_topology import (
    build_ac_ppc_with_topology_from_e_file,
    build_ac_ppc_with_topology_from_efile_rows,
    build_dc_ppc_with_topology_from_e_file,
    build_dc_ppc_with_topology_from_efile_rows,
    build_hybrid_ppc_with_topology_from_efile_rows,
    ensure_ac_ppc_topology,
    ensure_hybrid_ppc_topology,
)


DEFAULT_HYBRID_EFILE = model_file("hybrid", "hybrid_net_40.e")


def _node_count(network_part) -> int:
    try:
        return len(getattr(network_part, "nodes", []))
    except TypeError:
        return 0


def _ppc_has_operational_nodes(ppc, network_part) -> bool:
    topology = ppc.get("_topology_arrays") if isinstance(ppc, dict) else None
    node_alive = getattr(topology, "node_alive_mask", None)
    if node_alive is not None:
        return bool(np.any(np.asarray(node_alive, dtype=bool)))
    nodes = getattr(network_part, "nodes", ())
    alive_states = [getattr(node, "is_alive", None) for node in nodes]
    if any(state is not None for state in alive_states):
        return any(bool(state) for state in alive_states)
    return _node_count(network_part) > 0


def _zero_result_table_spec(columns, dynamic_names):
    return (
        max(columns.values()) + 1 if columns else 0,
        tuple(columns[name] for name in dynamic_names),
    )


def _zero_table_columns(ppc, key, table_spec):
    canonical_width, columns = table_spec
    source = None if ppc is None else ppc.get(key)
    if source is None:
        return np.zeros((0, canonical_width), dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2:
        return np.zeros((0, canonical_width), dtype=np.float64)
    result = source.copy()
    if result.size and columns:
        existing_columns = [
            column
            for column in columns
            if 0 <= column < result.shape[1]
        ]
        if existing_columns:
            result[:, existing_columns] = 0.0
    return result


_AC_ZERO_RESULT_TABLE_SPECS = {
    "bus": _zero_result_table_spec(AC_BUS_COLS, ("voltage", "angle")),
    "gen": _zero_result_table_spec(AC_GEN_COLS, ("p", "q", "current")),
    "load": _zero_result_table_spec(AC_LOAD_COLS, ("p", "q", "current")),
    "shunt": _zero_result_table_spec(AC_SHUNT_COLS, ("p", "q", "current")),
    "branch": _zero_result_table_spec(
        AC_BRANCH_COLS,
        ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"),
    ),
    "transformer": _zero_result_table_spec(
        AC_TRANSFORMER_COLS,
        ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"),
    ),
    "three_winding_transformer": _zero_result_table_spec(
        AC_THREE_WINDING_TRANSFORMER_COLS,
        ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"),
    ),
    "zero_branch": _zero_result_table_spec(AC_ZERO_BRANCH_COLS, ("p", "q", "current")),
    "switch": _zero_result_table_spec(AC_SWITCH_COLS, ("p", "q", "current")),
    "break": _zero_result_table_spec(AC_BREAK_COLS, ("p", "q", "current")),
    "acac": _zero_result_table_spec(ACAC_COLS, ("i_p", "i_q", "j_p", "j_q", "i_i", "j_i")),
}

_DC_ZERO_RESULT_TABLE_SPECS = {
    "bus": _zero_result_table_spec(DC_BUS_COLS, ("voltage",)),
    "branch": _zero_result_table_spec(DC_BRANCH_COLS, ("i_p", "j_p", "current")),
    "load": _zero_result_table_spec(DC_LOAD_COLS, ("p", "current")),
    "gen": _zero_result_table_spec(DC_GEN_COLS, ("p", "current")),
    "zero_branch": _zero_result_table_spec(DC_ZERO_BRANCH_COLS, ("p", "current")),
    "switch": _zero_result_table_spec(DC_SWITCH_COLS, ("p", "current")),
    "break": _zero_result_table_spec(DC_BREAK_COLS, ("p", "current")),
    "dcdc": _zero_result_table_spec(DC_DCDC_COLS, ("i_p", "j_p", "i_c", "j_c")),
}


def _zero_subgrid_result(ppc, table_specs):
    if not isinstance(ppc, dict):
        return None
    return {
        key: _zero_table_columns(ppc, key, table_spec)
        for key, table_spec in table_specs.items()
    }


_AC_SKIPPED_OBJECT_ATTRS = {
    "nodes": ("voltage", "angle"),
    "generators": ("p", "q", "current"),
    "loads": ("p", "q", "current"),
    "shunt_compensators": ("p", "q", "current"),
    "branches": ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"),
    "transformers": ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"),
    "three_winding_transformers": (
        "i_p",
        "i_q",
        "i_c",
        "j_p",
        "j_q",
        "j_c",
        "k_p",
        "k_q",
        "k_c",
    ),
    "zero_branches": ("p", "q", "current"),
    "switches": ("p", "q", "current"),
    "breakers": ("p", "q", "current"),
}

_DC_SKIPPED_OBJECT_ATTRS = {
    "nodes": ("voltage",),
    "generators": ("p", "current"),
    "loads": ("p", "current"),
    "branches": ("i_p", "j_p", "current"),
    "zero_branches": ("p", "current"),
    "switches": ("p", "current"),
    "breakers": ("p", "current"),
    "dcdc_converters": ("i_p", "j_p", "i_c", "j_c"),
}


def _zero_skipped_object_side(network_part, attribute_map):
    for collection_name, attributes in attribute_map.items():
        for device in getattr(network_part, collection_name, ()):
            if hasattr(device, "is_alive"):
                device.is_alive = False
            for attribute in attributes:
                setattr(device, attribute, 0.0)


def _default_hybrid_linear_solver(has_dc: bool) -> str:
    return "pyklu"


def _array_device(idx, name=None, **values):
    return SimpleNamespace(idx=int(idx), name=str(name if name is not None else idx), **values)


def _build_group_share_metadata(positions, setpoints, active_mask):
    positions = np.asarray(positions, dtype=np.int32)
    setpoints = np.asarray(setpoints, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)
    n = positions.size
    avg_set = setpoints.copy()
    rep_mask = np.zeros(n, dtype=bool)
    share_mask = np.zeros(n, dtype=bool)
    ref_idx = np.arange(n, dtype=np.int32)
    if n == 0 or not np.any(active_mask):
        return avg_set, rep_mask, share_mask, ref_idx
    active_idx = np.flatnonzero(active_mask).astype(np.int32, copy=False)
    pos_active = positions[active_idx]
    order = np.argsort(pos_active, kind="stable")
    sorted_idx = active_idx[order]
    sorted_pos = pos_active[order]
    start = 0
    while start < sorted_idx.size:
        stop = start + 1
        bus_pos = sorted_pos[start]
        while stop < sorted_idx.size and sorted_pos[stop] == bus_pos:
            stop += 1
        members = sorted_idx[start:stop]
        ref = int(members[0])
        avg = float(np.mean(setpoints[members]))
        avg_set[members] = avg
        rep_mask[ref] = True
        if members.size > 1:
            share_mask[members[1:]] = True
        ref_idx[members] = ref
        start = stop
    return avg_set, rep_mask, share_mask, ref_idx


class _LightweightHybridNetwork(SimpleNamespace):
    @property
    def total_nodes(self) -> int:
        return len(self.ac.nodes) + len(self.dc.nodes)

    @property
    def acac_converters(self):
        """Compatibility view; ACAC storage and calculation belong to the AC side."""
        return self.ac.acac_converters


class _PpcACNodeList:
    def __init__(self, facade, ppc):
        self.facade = facade
        self.ppc = ppc

    def __len__(self):
        return int(self.ppc["bus"].shape[0])

    def __iter__(self):
        result = getattr(self.facade, "result", None) or self.ppc
        names = self.ppc.get("bus_name")
        for pos, row in enumerate(result["bus"]):
            yield _array_device(
                row[AC_BUS_COLS["idx"]],
                None if names is None else names[pos],
                vbase=float(self.ppc["bus"][pos, AC_BUS_COLS["vbase"]]),
                voltage=float(row[AC_BUS_COLS["voltage"]]),
                angle=float(row[AC_BUS_COLS["angle"]]),
                run_stat=int(row[AC_BUS_COLS["run_stat"]]),
            )


class _PpcDCNodeList:
    def __init__(self, facade, ppc):
        self.facade = facade
        self.ppc = ppc

    def __len__(self):
        return int(self.ppc["bus"].shape[0]) if self.ppc is not None else 0

    def __iter__(self):
        if self.ppc is None:
            return
        result = getattr(self.facade, "result", None) or self.ppc
        names = self.ppc.get("bus_name")
        for pos, row in enumerate(result["bus"]):
            yield _array_device(
                row[DC_BUS_COLS["idx"]],
                None if names is None else names[pos],
                vbase=float(self.ppc["bus"][pos, DC_BUS_COLS["vbase"]]),
                voltage=float(row[DC_BUS_COLS["voltage"]]),
                run_stat=int(row[DC_BUS_COLS["run_stat"]]),
            )


class _PpcDeviceList:
    def __init__(self, facade, ppc, table_key, name_key, cols):
        self.facade = facade
        self.ppc = ppc
        self.table_key = table_key
        self.name_key = name_key
        self.cols = cols

    def __len__(self):
        table = self.ppc.get(self.table_key)
        return 0 if table is None else int(table.shape[0])

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        table = self.ppc.get(self.table_key)
        if table is None:
            return
        result = getattr(self.facade, "result", None) or self.ppc
        result_table = result.get(self.table_key, table)
        names = self.ppc.get(self.name_key)
        node_by_idx = {int(node.idx): node for node in self.facade.nodes}
        topology = self.ppc.get("_topology_arrays")
        device_topology = None if topology is None else topology.devices.get(self.table_key)
        alive_mask = None if device_topology is None else np.asarray(device_topology.alive_mask, dtype=bool)
        for pos, source_row in enumerate(table):
            row = result_table[pos] if pos < len(result_table) else source_row
            name = None if names is None or pos >= len(names) else names[pos]
            alive = None if alive_mask is None or pos >= alive_mask.size else bool(alive_mask[pos])
            yield _ppc_device(
                source_row,
                row,
                name,
                self.cols,
                node_by_idx,
                table_key=self.table_key,
                alive=alive,
            )

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        return list(self)[int(index)]


def _ppc_device(static_row, row, name, cols, node_by_idx, *, table_key=None, alive=None):
    int_fields = {
        "idx",
        "node",
        "i_node",
        "j_node",
        "k_node",
        "status",
        "run_stat",
        "control_type",
        "ac_control_type",
        "dc_control_type",
        "i_control_type",
        "j_control_type",
    }
    static_fields = {
        "node",
        "i_node",
        "j_node",
        "k_node",
        "r",
        "x",
        "b",
        "gt",
        "bt",
        "tap",
        "shift",
        "i_r",
        "i_x",
        "j_r",
        "j_x",
        "k_r",
        "k_x",
        "i_tap",
        "i_shift",
        "j_tap",
        "j_shift",
        "k_tap",
        "k_shift",
    }
    values = {}
    for attr, col in cols.items():
        source = static_row if attr in static_fields else row
        value = source[col]
        values[attr] = int(value) if attr in int_fields else float(value)
    if "node" in values:
        values["node_obj"] = node_by_idx.get(int(values["node"]))
    if "i_node" in values:
        values["i_node_obj"] = node_by_idx.get(int(values["i_node"]))
    if "j_node" in values:
        values["j_node_obj"] = node_by_idx.get(int(values["j_node"]))
    if "k_node" in values:
        values["k_node_obj"] = node_by_idx.get(int(values["k_node"]))
    run_stat = int(values.get("run_stat", 1))
    status = int(values.get("status", 1))
    values["is_alive"] = (run_stat == 1 and status == 1) if alive is None else bool(alive)
    if table_key == "dcdc":
        values["i_control_type"] = DCDC_SIDE_CONTROL_LABEL.get(values["i_control_type"], "P")
        values["j_control_type"] = DCDC_SIDE_CONTROL_LABEL.get(values["j_control_type"], "NONE")
        values["control_type"] = dcdc_legacy_control_label(
            values["i_control_type"],
            values["j_control_type"],
        )
    return _array_device(values.pop("idx", 0), name, **values)


class _PpcConverterList:
    def __init__(self, ppc, table_key, name_key, cols, label_map, field_specs, *, facade=None):
        self.ppc = ppc
        self.table_key = table_key
        self.name_key = name_key
        self.cols = cols
        self.label_map = label_map
        self.field_specs = field_specs
        self.facade = facade

    def __len__(self):
        table = self.ppc.get(self.table_key)
        return 0 if table is None else int(table.shape[0])

    def __bool__(self):
        return len(self) > 0

    def _item_at(self, pos):
        table = self.ppc.get(self.table_key)
        if table is None:
            raise IndexError(pos)
        size = int(table.shape[0])
        if pos < 0:
            pos += size
        if pos < 0 or pos >= size:
            raise IndexError(pos)
        result = getattr(self.facade, "result", None) if self.facade is not None else None
        result_table = table if not result else result.get(self.table_key, table)
        row = result_table[pos] if pos < len(result_table) else table[pos]
        names = self.ppc.get(self.name_key)
        name = None if names is None or pos >= len(names) else names[pos]
        values = {}
        for attr, col_name, kind in self.field_specs:
            raw = row[self.cols[col_name]]
            if kind == "int":
                values[attr] = int(raw)
            elif kind == "control":
                values[attr] = self.label_map.get(int(raw), "")
            elif kind == "acac_side":
                values[attr] = ACAC_SIDE_CONTROL_LABEL.get(int(raw), "")
            elif kind == "ac_control":
                values[attr] = DCAC_AC_CONTROL_LABEL.get(int(raw), "")
            elif kind == "dc_control":
                values[attr] = DCAC_DC_CONTROL_LABEL.get(int(raw), "")
            elif kind == "dcac_device_type":
                values[attr] = DCAC_DEVICE_TYPE_LABEL.get(int(raw), "DCACConverter")
            elif kind == "none":
                values[attr] = None
            else:
                values[attr] = float(raw)
        values["is_alive"] = int(row[self.cols["run_stat"]]) == 1
        return _array_device(row[self.cols["idx"]], name, **values)

    def __iter__(self):
        for pos in range(len(self)):
            yield self._item_at(pos)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._item_at(pos) for pos in range(start, stop, step)]
        return self._item_at(int(index))


def _lightweight_ac_network(ac_ppc):
    network = SimpleNamespace(
        _lf_lightweight=True,
        ppc=ac_ppc,
        result=None,
        islands=[],
        node_dict={},
    )
    network.nodes = _PpcACNodeList(network, ac_ppc)
    network.branches = _PpcDeviceList(network, ac_ppc, "branch", "branch_name", AC_BRANCH_COLS)
    network.transformers = _PpcDeviceList(network, ac_ppc, "transformer", "transformer_name", AC_TRANSFORMER_COLS)
    network.three_winding_transformers = _PpcDeviceList(
        network,
        ac_ppc,
        "three_winding_transformer",
        "three_winding_transformer_name",
        AC_THREE_WINDING_TRANSFORMER_COLS,
    )
    network.generators = _PpcDeviceList(network, ac_ppc, "gen", "gen_name", AC_GEN_COLS)
    network.loads = _PpcDeviceList(network, ac_ppc, "load", "load_name", AC_LOAD_COLS)
    network.shunt_compensators = _PpcDeviceList(network, ac_ppc, "shunt", "shunt_name", AC_SHUNT_COLS)
    network.zero_branches = _PpcDeviceList(network, ac_ppc, "zero_branch", "zero_branch_name", AC_ZERO_BRANCH_COLS)
    network.switches = _PpcDeviceList(network, ac_ppc, "switch", "switch_name", AC_SWITCH_COLS)
    network.breakers = _PpcDeviceList(network, ac_ppc, "break", "break_name", AC_BREAK_COLS)
    network.acac_converters = _PpcConverterList(
        ac_ppc,
        "acac",
        "acac_name",
        ACAC_COLS,
        None,
        (
            ("i_node", "i_node", "int"),
            ("j_node", "j_node", "int"),
            ("r1", "r1", "float"),
            ("r2", "r2", "float"),
            ("i_control_type", "i_control_type", "acac_side"),
            ("j_control_type", "j_control_type", "acac_side"),
            ("p_set", "p_set", "float"),
            ("i_q_set", "i_q_set", "float"),
            ("j_q_set", "j_q_set", "float"),
            ("i_v_set", "i_v_set", "float"),
            ("j_v_set", "j_v_set", "float"),
            ("run_stat", "run_stat", "int"),
            ("i_p", "i_p", "float"),
            ("i_q", "i_q", "float"),
            ("j_p", "j_p", "float"),
            ("j_q", "j_q", "float"),
            ("i_i", "i_i", "float"),
            ("j_i", "j_i", "float"),
            ("i_node_obj", "idx", "none"),
            ("j_node_obj", "idx", "none"),
        ),
        facade=network,
    )
    return network


def _lightweight_dc_network(dc_ppc=None):
    if dc_ppc is None:
        network = SimpleNamespace(
            _lf_lightweight=True,
            ppc=None,
            result=None,
            branches=[],
            loads=[],
            generators=[],
            zero_branches=[],
            switches=[],
            breakers=[],
            dcdc_converters=[],
            islands=[],
            node_dict={},
        )
        network.nodes = _PpcDCNodeList(network, None)
        return network
    network = SimpleNamespace(
        _lf_lightweight=True,
        ppc=dc_ppc,
        result=None,
        islands=[],
        node_dict={},
    )
    network.nodes = _PpcDCNodeList(network, dc_ppc)
    network.branches = _PpcDeviceList(network, dc_ppc, "branch", "branch_name", DC_BRANCH_COLS)
    network.loads = _PpcDeviceList(network, dc_ppc, "load", "load_name", DC_LOAD_COLS)
    network.generators = _PpcDeviceList(network, dc_ppc, "gen", "gen_name", DC_GEN_COLS)
    network.zero_branches = _PpcDeviceList(network, dc_ppc, "zero_branch", "zero_branch_name", DC_ZERO_BRANCH_COLS)
    network.switches = _PpcDeviceList(network, dc_ppc, "switch", "switch_name", DC_SWITCH_COLS)
    network.breakers = _PpcDeviceList(network, dc_ppc, "break", "break_name", DC_BREAK_COLS)
    network.dcdc_converters = _PpcDeviceList(network, dc_ppc, "dcdc", "dcdc_name", DC_DCDC_COLS)
    return network


def _detect_lf_rows_kind(rows) -> str:
    """Classify already-loaded E rows before choosing array-only or full hybrid loading."""
    has_ac = bool(rows.get("ACNode", {}).get("rows"))
    has_dc = bool(rows.get("DCNode", {}).get("rows"))
    has_hybrid_converter = bool(rows.get("DCACConverter", {}).get("rows"))
    if has_hybrid_converter or (has_ac and has_dc):
        return "hybrid"
    if has_ac:
        return "ac"
    if has_dc:
        return "dc"
    if any(bool(rows.get(f"{prefix}Node", {}).get("rows")) for prefix in ("Heat", "Gas", "Hydro", "Steam")):
        return "fluid"
    return "hybrid"


def _build_lf_network_from_fluid_only(file_name) -> _LightweightHybridNetwork:
    ac_network = SimpleNamespace(
        _lf_lightweight=True,
        ppc=None,
        result=None,
        nodes=[],
        branches=[],
        transformers=[],
        three_winding_transformers=[],
        generators=[],
        loads=[],
        shunt_compensators=[],
        zero_branches=[],
        switches=[],
        breakers=[],
        acac_converters=[],
        islands=[],
        node_dict={},
    )
    dc_network = SimpleNamespace(
        _lf_lightweight=True,
        ppc=None,
        result=None,
        nodes=[],
        branches=[],
        generators=[],
        loads=[],
        zero_branches=[],
        switches=[],
        breakers=[],
        dcdc_converters=[],
        islands=[],
        node_dict={},
    )
    network = _LightweightHybridNetwork(
        _lf_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=[],
        hybrid_islands=[],
        _ac_ppc=None,
        _dc_ppc=None,
        p_base=1.0,
        p_base_kW=1.0,
        u_scale=1.0,
        p_scale=1.0,
        i_scale=1.0,
        source=str(Path(file_name).resolve()),
    )
    return network


def _build_lf_network_from_single_ac_file(file_name, rows=None) -> _LightweightHybridNetwork:
    ac_ppc = (
        build_ac_ppc_with_topology_from_e_file(file_name)
        if rows is None
        else build_ac_ppc_with_topology_from_efile_rows(file_name, rows)
    )
    return _build_lf_network_from_single_ac_ppc(file_name, ac_ppc)


def _build_lf_network_from_single_ac_ppc(file_name, ac_ppc) -> _LightweightHybridNetwork:
    base = ac_ppc["base"]
    ac_network = _lightweight_ac_network(ac_ppc)
    dc_network = _lightweight_dc_network()
    network = _LightweightHybridNetwork(
        _lf_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=[],
        hybrid_islands=[],
    )
    network.ppc = {
        "format": "hybrid_ppc_v1",
        "source": str(file_name),
        "base": base,
        "ac": ac_ppc,
        "dc": None,
    }
    network._ac_ppc = ac_ppc
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    return network


def _build_lf_network_from_single_dc_file(file_name, rows=None) -> _LightweightHybridNetwork:
    dc_ppc = (
        build_dc_ppc_with_topology_from_e_file(file_name)
        if rows is None
        else build_dc_ppc_with_topology_from_efile_rows(file_name, rows)
    )
    base = dc_ppc["base"]
    ac_network = _lightweight_ac_network(
        {
            "base": {
                "p_base": float(base["p_base"]),
                "u_scale": float(base["u_scale"]),
                "p_scale": float(base["p_scale"]),
                "i_scale": float(base["i_scale"]),
                "p_base_kW": float(base["p_base_kW"]),
            },
            "bus": np.zeros((0, len(AC_BUS_COLS)), dtype=np.float64),
        }
    )
    dc_network = _lightweight_dc_network(dc_ppc)
    network = _LightweightHybridNetwork(
        _lf_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=[],
        hybrid_islands=[],
    )
    network.ppc = {"format": "hybrid_ppc_v1", "source": str(file_name), "base": dc_ppc["base"], "ac": None, "dc": dc_ppc}
    network._dc_ppc = dc_ppc
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    return network


def _build_lf_network_from_hybrid_rows(file_name, rows) -> _LightweightHybridNetwork:
    ppc = build_hybrid_ppc_with_topology_from_efile_rows(file_name, rows)
    ac_network = _lightweight_ac_network(ppc["ac"])
    dc_network = _lightweight_dc_network(ppc["dc"])
    network = _LightweightHybridNetwork(
        _lf_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=_PpcConverterList(
            ppc,
            "dcac",
            "dcac_name",
            DCAC_COLS,
            None,
            (
                ("ac_node", "ac_node", "int"),
                ("dc_node", "dc_node", "int"),
                ("r1", "r1", "float"),
                ("r2", "r2", "float"),
                ("ac_control_type", "ac_control_type", "ac_control"),
                ("dc_control_type", "dc_control_type", "dc_control"),
                ("p_ac_set", "p_ac_set", "float"),
                ("p_dc_set", "p_dc_set", "float"),
                ("q_ac_set", "q_ac_set", "float"),
                ("v_ac_set", "v_ac_set", "float"),
                ("v_dc_set", "v_dc_set", "float"),
                ("run_stat", "run_stat", "int"),
                ("dc_p", "dc_p", "float"),
                ("ac_p", "ac_p", "float"),
                ("ac_q", "ac_q", "float"),
                ("dc_i", "dc_i", "float"),
                ("ac_i", "ac_i", "float"),
                ("dev_type", "dev_type", "dcac_device_type"),
                ("ac_node_obj", "idx", "none"),
                ("dc_node_obj", "idx", "none"),
            ),
        ),
        hybrid_islands=[],
    )
    network.ppc = ppc
    network._ac_ppc = ppc["ac"]
    network._dc_ppc = ppc["dc"]
    base = ppc["base"]
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    return network


def _read_lf_network_from_file(file_name) -> HybridPowerNetwork:
    path = Path(file_name)
    if path.suffix.lower() in {".m", ".mat"}:
        ac_ppc = build_ac_ppc_from_mat_file(path)
        ac_ppc["source"] = str(path.resolve())
        return _build_lf_network_from_single_ac_ppc(path, ensure_ac_ppc_topology(ac_ppc))

    efile_rows = _read_efile_rows(file_name)
    file_kind = _detect_lf_rows_kind(efile_rows)
    if file_kind == "ac":
        network = _build_lf_network_from_single_ac_file(file_name, efile_rows)
    elif file_kind == "dc":
        network = _build_lf_network_from_single_dc_file(file_name, efile_rows)
    elif file_kind == "fluid":
        network = _build_lf_network_from_fluid_only(file_name)
    else:
        network = _build_lf_network_from_hybrid_rows(file_name, efile_rows)
    context = build_multi_energy_context_from_rows(efile_rows, source=file_name)
    return attach_multi_energy_context(network, context)


@dataclass
class DCACLFResult:
    dcac_converters: dict = field(default_factory=dict)


@dataclass
class HybridLFResult:
    arrays: dict = field(default_factory=dict)
    network: Optional[HybridPowerNetwork] = None
    ac_network: Any = None
    dc_network: Any = None
    calc: Optional["HybridPowerFlowCalc"] = None
    ac_calc: Optional[ACPowerFlowCalc] = None
    dc_calc: Optional[DCPowerFlowCalc] = None
    rc: int = -1
    ac_warnings: List[str] = field(default_factory=list)
    ac_errors: List[str] = field(default_factory=list)
    dc_warnings: List[str] = field(default_factory=list)
    dc_errors: List[str] = field(default_factory=list)
    ac: Optional[ACLFResult] = None
    dc: Optional[DCLFResult] = None
    dcac: DCACLFResult = field(default_factory=DCACLFResult)
    fluid: Dict[str, object] = field(default_factory=dict)
    fluid_calcs: Dict[str, object] = field(default_factory=dict)
    fluid_warnings: List[str] = field(default_factory=list)
    fluid_errors: Dict[str, str] = field(default_factory=dict)
    couplings: List[object] = field(default_factory=list)

    @property
    def lf_result(self) -> "HybridLFResult":
        return self

    @property
    def total_nodes(self) -> int:
        return 0 if self.network is None else self.network.total_nodes

    @property
    def total_fluid_nodes(self) -> int:
        context = None if self.network is None else getattr(self.network, "multi_energy", None)
        return 0 if context is None else int(context.total_fluid_nodes)

    @property
    def total_energy_nodes(self) -> int:
        return self.total_nodes + self.total_fluid_nodes

    @property
    def converged(self) -> bool:
        return (
            self.rc == 0
            and self.calc is not None
            and self.calc.converged
            and not self.ac_errors
            and not self.dc_errors
            and not self.fluid_errors
        )

    @property
    def global_jacobian_shape(self) -> Tuple[int, int]:
        return (0, 0) if self.calc is None else self.calc.last_jacobian_shape

    @property
    def has_ac(self) -> bool:
        return self.ac_network is not None and len(self.ac_network.nodes) > 0

    @property
    def has_dc(self) -> bool:
        return self.dc_network is not None and len(self.dc_network.nodes) > 0

    @property
    def has_dcac(self) -> bool:
        return self.network is not None and len(self.network.dcac_converters) > 0

class HybridPowerFlowCalc:
    """统一交直流 Newton 求解器。

    AC、DC 子网和 DC/AC 换流器变量在同一个全局状态向量中求解。
    ACACConverter 由 ACPowerFlowCalc 作为交流侧设备处理；本类只拼接
    AC、DC 子块和 DCACConverter 的跨域耦合方程。

    混联网中 ACPowerFlowCalc/DCPowerFlowCalc 只作为方程和导数提供器，
    不分别执行潮流迭代。每次 Newton 迭代只形成一个全局残差向量和一个
    全局稀疏 Jacobian，并对完整线性方程组做一次联合求解。

    Hybrid LF 继承 AC/DC 两侧的控制语义：同一 AC 或 DC 母线上若存在
    多个定压设备，可在 LF 中做代表控制方程和功率分摊。

    Heat、gas、hydrogen 和 steam 子模块只提供各自的状态、残差和稀疏
    Jacobian。本类把电气、流体和跨能源耦合块拼成一个全局 Newton 问题，
    每次迭代只做一次联合线性求解。仅当耦合表给出能量换算因子时才增加
    跨域控制状态和能量平衡方程，否则保留为已校验的端点关联。
    """

    def __init__(
        self,
        network,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        island=None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        keep_node_objects: bool = True,
        linear_solver: Optional[str] = None,
        result_mode: str = "full",
        verbose: bool = False,
    ):
        self.network = network
        self.island = island
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.verbose = verbose
        self.result_mode = self._normalize_result_mode(result_mode)
        self.keep_node_objects = False
        hybrid_ppc = getattr(network, "ppc", None)
        if isinstance(hybrid_ppc, dict) and ("ac" in hybrid_ppc or "dc" in hybrid_ppc):
            ensure_hybrid_ppc_topology(hybrid_ppc)
        elif not getattr(network, "_lf_lightweight", False):
            topo = getattr(network, "topo", None)
            if callable(topo):
                topo()
        self._ac_ppc = getattr(network, "_ac_ppc", None) or getattr(network.ac, "ppc", None)
        self._dc_ppc = getattr(network, "_dc_ppc", None) or getattr(network.dc, "ppc", None)
        self.has_ac = _ppc_has_operational_nodes(self._ac_ppc, network.ac)
        self.has_dc = _ppc_has_operational_nodes(self._dc_ppc, network.dc)
        self._skipped_ac_result = None
        if not self.has_ac:
            ac_result_source = self._ac_ppc
            if ac_result_source is None and not getattr(network.ac, "_lf_lightweight", False):
                ac_result_source = build_ac_ppc_from_network(network.ac)
            self._skipped_ac_result = _zero_subgrid_result(
                ac_result_source,
                _AC_ZERO_RESULT_TABLE_SPECS,
            )
        self._skipped_dc_result = None
        if not self.has_dc:
            dc_result_source = self._dc_ppc
            if dc_result_source is None and not getattr(network.dc, "_lf_lightweight", False):
                dc_result_source = build_dc_ppc_from_network(network.dc)
            self._skipped_dc_result = _zero_subgrid_result(
                dc_result_source,
                _DC_ZERO_RESULT_TABLE_SPECS,
            )
        # Hybrid 含 DC 网络时，PyKLU 在部分非对称/耦合矩阵上会失败；未显式指定时
        # 默认走 UMFPACK，若本机未安装则由 solver_common 回退到 SciPy/SuperLU。
        solver_name = str(linear_solver).strip().lower() if linear_solver is not None else ""
        self.linear_solver = solver_name or _default_hybrid_linear_solver(self.has_dc)
        self._linear_solver_resolved, self._linear_solver_fn = _resolve_linear_solver(self.linear_solver)
        # 实例级可选求解器黑名单: 失败时只在本实例回退 scipy, 不污染模块缓存。
        self._instance_solver_blacklist: set = set()
        self.ac_calc = self._build_ac_subcalc()
        self.dc_calc = self._build_dc_subcalc()
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.failure_reason = ""
        self.x = np.array([], dtype=np.float64)
        self.ac_size = 0
        self.dc_size = 0
        self.ac_eq = 0
        self.dc_eq = 0
        self.dcac_start = 0
        self.dcac_eq_start = 0
        self.dcac_converters = []
        self.N_dcac = 0
        self.total_vars = 0
        self.total_eq = 0
        self.electric_vars = 0
        self.electric_eq = 0
        self.dc_G = None
        self.last_jacobian_shape = (0, 0)
        self._residual_work = np.array([], dtype=np.float64)
        self._electric_residual_work = np.array([], dtype=np.float64)
        self._converter_ppc_mode = bool(
            getattr(network, "_lf_lightweight", False)
            and isinstance(getattr(network, "ppc", None), dict)
        )
        self._clear_converter_outputs()
        needs_ac_node_lookup = (
            not self._converter_ppc_mode
            and bool(getattr(network, "dcac_converters", []))
        )
        needs_dc_node_lookup = (
            not self._converter_ppc_mode
            and bool(getattr(network, "dcac_converters", []))
        )
        self._ac_node_obj_by_idx = (
            {int(node.idx): node for node in getattr(network.ac, "nodes", [])}
            if needs_ac_node_lookup
            else {}
        )
        self._dc_node_obj_by_idx = (
            {int(node.idx): node for node in getattr(network.dc, "nodes", [])}
            if needs_dc_node_lookup
            else {}
        )
        self._clear_dcac_arrays()
        self._clear_converter_jacobian_structure()
        self._clear_global_jacobian_pattern()
        self.lf_result = None
        self._single_ac_newton_block = False
        self._single_dc_newton_block = False
        self.result = {}
        self.multi_energy: MultiEnergyContext = getattr(
            network,
            "multi_energy",
            MultiEnergyContext(),
        )
        self.fluid_calcs = self._build_fluid_subcalcs()
        self.fluid_state_slices: Dict[str, slice] = {}
        self.fluid_eq_slices: Dict[str, slice] = {}
        self.energy_coupling_plans: List[object] = []
        self.energy_coupling_state_slice = slice(0, 0)
        self.energy_coupling_eq_slice = slice(0, 0)
        self.fluid_rc: Dict[str, int] = {}
        self.fluid_errors: Dict[str, str] = {}
        self.coupling_results: List[object] = []
        self.electric_converged = False
        self._prepared = False

    @classmethod
    def from_file_fast(
        cls,
        file_name,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        linear_solver: Optional[str] = None,
        result_mode: str = "array",
        verbose: bool = False,
    ) -> "HybridPowerFlowCalc":
        """Build a lightweight PPC-backed hybrid solver directly from file.

        Accepted inputs:
        - `.e` named-unit/network files

        This path uses `_read_lf_network_from_file()`, which returns a `_LightweightHybridNetwork`
        backed by PPC arrays and lightweight facades. It avoids `HybridPowerNetwork.read_from_file()`
        and the expensive Python object graph construction (tens of thousands of AC/DC node objects),
        while remaining API-compatible for LF array-mode workloads.
        """
        path = Path(file_name)
        if path.suffix.lower() != ".e":
            raise ValueError(f"HybridPowerFlowCalc.from_file_fast() only supports .e files, got: {path}")
        network = _read_lf_network_from_file(path)
        return cls(
            network,
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
            parameter_file=parameter_file,
            parameters=parameters,
            linear_solver=linear_solver,
            result_mode=result_mode,
            verbose=verbose,
        )

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        return _normalize_lf_result_mode(result_mode, "Hybrid")

    def _build_ac_subcalc(self):
        if not self.has_ac:
            return None
        source = self._ac_ppc if self._ac_ppc is not None else self.network.ac
        calc = ACPowerFlowCalc(
            source,
            parameters=self.params,
            keep_node_objects=False,
            linear_solver=self.linear_solver,
            result_mode="array",
            verbose=self.verbose,
        )
        calc._network_writeback = self.network.ac
        return calc

    def _build_dc_subcalc(self):
        if not self.has_dc:
            return None
        source = self._dc_ppc if self._dc_ppc is not None else self.network.dc
        calc = DCPowerFlowCalc(
            source,
            parameters=self.params,
            keep_node_objects=False,
            linear_solver=self.linear_solver,
            result_mode="array",
            verbose=self.verbose,
        )
        if not getattr(self.network.dc, "_lf_lightweight", False):
            calc._network_writeback = self.network.dc
        return calc

    def _build_fluid_subcalcs(self) -> Dict[str, object]:
        calc_types = {
            "heat": HeatPowerFlowCalc,
            "gas": GasPowerFlowCalc,
            "hydro": HydroPowerFlowCalc,
            "steam": SteamPowerFlowCalc,
        }
        return {
            name: calc_types[name](
                network,
                parameters=self.params,
                linear_solver="scipy",
                result_mode=self.result_mode,
                verbose=self.verbose,
            )
            for name, network in self.multi_energy.fluid_networks.items()
        }

    def _add_multi_energy_warning(self, message: str) -> None:
        if message not in self.multi_energy.warnings:
            self.multi_energy.warnings.append(message)

    @staticmethod
    def _device_row_by_idx(table: np.ndarray, columns: Dict[str, int], device_idx: int) -> int:
        if table.ndim != 2 or table.size == 0:
            return -1
        rows = np.flatnonzero(
            table[:, columns["idx"]].astype(np.int64, copy=False) == int(device_idx)
        )
        return int(rows[0]) if rows.size else -1

    def _resolve_energy_endpoint(
        self,
        terminal,
        *,
        adjustable: bool,
        allow_missing_fluid_balance: bool = False,
    ):
        domain = str(terminal.domain)
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_storage = device_type.endswith("Storage")
        is_source = device_type.endswith(("Source", "Generator", "Unit")) or is_storage
        if not is_load and not is_source:
            return None, f"unsupported endpoint type {device_type}"

        if domain in {"ac", "dc"}:
            calc = self.ac_calc if domain == "ac" else self.dc_calc
            ppc = self._ac_ppc if domain == "ac" else self._dc_ppc
            if calc is None or not isinstance(ppc, dict):
                return None, f"{domain} endpoint has no active solver block"
            if domain == "ac":
                columns = AC_LOAD_COLS if is_load else AC_GEN_COLS
            else:
                columns = DC_LOAD_COLS if is_load else DC_GEN_COLS
            table_key = "load" if is_load else "gen"
            table = np.asarray(ppc.get(table_key, ()), dtype=np.float64)
            row = self._device_row_by_idx(table, columns, terminal.device_idx)
            if row < 0 or int(table[row, columns["run_stat"]]) != 1:
                return None, f"missing or inactive {device_type} idx={terminal.device_idx}"
            node_id = int(table[row, columns["node"]])
            solver_pos = (
                self._ac_solver_pos(node_id)
                if domain == "ac"
                else self._dc_solver_pos(node_id)
            )
            if solver_pos < 0:
                return None, f"{device_type} idx={terminal.device_idx} is outside the live topology"
            if domain == "ac":
                balance_row = int(self._ac_balance_rows([solver_pos])[0][0])
                state_voltage_col = int(self.ac_calc.v_col_by_node[solver_pos])
            else:
                local_row = int(self.dc_calc.node_eq[solver_pos])
                balance_row = -1 if local_row < 0 else self.ac_eq + local_row
                state_voltage_col = self.ac_size + solver_pos
            if adjustable and balance_row < 0:
                return None, (
                    f"{device_type} idx={terminal.device_idx} is a reference endpoint; "
                    "its power cannot be released as a coupling control"
                )
            return SimpleNamespace(
                domain=domain,
                device_type=device_type,
                device_idx=int(terminal.device_idx),
                kind="load" if is_load else "source",
                table_key=table_key,
                table=table,
                columns=columns,
                row=row,
                solver_pos=solver_pos,
                balance_row=balance_row,
                state_voltage_col=state_voltage_col,
                correction_sign=1.0 if is_load else -1.0,
                energy_scale=float(getattr(self.network, "p_base_kW", 1.0)),
            ), ""

        calc = self.fluid_calcs.get(domain)
        if calc is None:
            return None, f"{domain} endpoint has no active solver block"
        network = calc.network
        if is_load:
            positions = [
                pos
                for pos, device in enumerate(network.loads)
                if int(device.idx) == int(terminal.device_idx)
            ]
        else:
            positions = [
                pos
                for pos, device in enumerate(network.sources)
                if bool(network.source_is_storage[pos]) == is_storage
                and int(device.idx) == int(terminal.device_idx)
            ]
        if not positions:
            return None, f"missing {device_type} idx={terminal.device_idx}"
        device_pos = int(positions[0])
        if is_load:
            node_pos = int(network.load_node_pos[device_pos])
            explicit_return = bool(network.load_explicit_return[device_pos])
            control_type = "FLOW"
        else:
            node_pos = int(network.source_node_pos[device_pos])
            explicit_return = bool(network.source_explicit_return[device_pos])
            control_type = str(network.source_control_type[device_pos])
        if adjustable and network.thermal:
            return None, (
                f"{device_type} idx={terminal.device_idx} changes transport energy equations; "
                "use a non-reference gas/hydrogen endpoint as coupling control"
            )
        if adjustable and network.steam and not is_load:
            return None, (
                f"{device_type} idx={terminal.device_idx} changes the steam enthalpy boundary; "
                "use a steam load as the released endpoint"
            )
        if adjustable and explicit_return:
            return None, (
                f"{device_type} idx={terminal.device_idx} has explicit return flow and "
                "cannot yet be released as one scalar coupling control"
            )
        if adjustable and not is_load and control_type != "FLOW":
            return None, (
                f"{device_type} idx={terminal.device_idx} is pressure-controlled; "
                "use it as the dependent endpoint, not the released endpoint"
            )
        local_row = int(network.balance_row_by_node[node_pos])
        if adjustable and local_row < 0 and not allow_missing_fluid_balance:
            return None, f"{device_type} idx={terminal.device_idx} has no hydraulic balance row"
        return SimpleNamespace(
            domain=domain,
            device_type=device_type,
            device_idx=int(terminal.device_idx),
            kind="load" if is_load else "storage" if is_storage else "source",
            device_pos=device_pos,
            solver_pos=node_pos,
            balance_row=(
                -1 if local_row < 0 else self.fluid_eq_slices[domain].start + local_row
            ),
            correction_sign=-1.0 if is_load else 1.0,
            energy_scale=1.0,
        ), ""

    def _energy_endpoint_value_and_derivative(self, endpoint, x: np.ndarray):
        domain = endpoint.domain
        if domain == "ac":
            if endpoint.kind == "source":
                return float(endpoint.table[endpoint.row, endpoint.columns["p_set"]]), (), ()
            ac_x = x[: self.electric_vars]
            _theta, voltage, _phi_re, _phi_im = self.ac_calc._extract_state_vars(
                ac_x,
                update_cache=False,
            )
            value_v = float(voltage[endpoint.solver_pos])
            row = endpoint.table[endpoint.row]
            pbase = float(row[endpoint.columns["pbase"]])
            pv0 = float(row[endpoint.columns["pv0"]])
            pv1 = float(row[endpoint.columns["pv1"]])
            pv2 = float(row[endpoint.columns["pv2"]])
            value = pbase * (pv0 + pv1 * value_v + pv2 * value_v * value_v)
            if endpoint.state_voltage_col < 0:
                return value, (), ()
            derivative = pbase * (pv1 + 2.0 * pv2 * value_v)
            return value, (endpoint.state_voltage_col,), (derivative,)

        if domain == "dc":
            if endpoint.kind == "source":
                return float(endpoint.table[endpoint.row, endpoint.columns["p_set"]]), (), ()
            value_v = float(x[self.ac_size + endpoint.solver_pos])
            row = endpoint.table[endpoint.row]
            pbase = float(row[endpoint.columns["pbase"]])
            pv0 = float(row[endpoint.columns["pv0"]])
            pv1 = float(row[endpoint.columns["pv1"]])
            pv2 = float(row[endpoint.columns["pv2"]])
            value = pbase * (pv0 + pv1 * value_v + pv2 * value_v * value_v)
            derivative = pbase * (pv1 + 2.0 * pv2 * value_v)
            return value, (endpoint.state_voltage_col,), (derivative,)

        calc = self.fluid_calcs[domain]
        network = calc.network
        if endpoint.kind == "load":
            return float(network.load_flow_set[endpoint.device_pos]), (), ()
        if str(network.source_control_type[endpoint.device_pos]) != "PRESSURE":
            return float(network.source_flow_set[endpoint.device_pos]), (), ()
        state_slice = self.fluid_state_slices[domain]
        source_flow, source_jacobian = calc.source_flows_and_jacobian(x[state_slice])
        derivative = source_jacobian.getrow(endpoint.device_pos).tocoo()
        columns = tuple((state_slice.start + derivative.col).tolist())
        return (
            float(source_flow[endpoint.device_pos]),
            columns,
            tuple(derivative.data.tolist()),
        )

    def _energy_endpoint_setpoint(self, endpoint) -> float:
        if endpoint.domain in {"ac", "dc"}:
            column = "pbase" if endpoint.kind == "load" else "p_set"
            return float(endpoint.table[endpoint.row, endpoint.columns[column]])
        network = self.fluid_calcs[endpoint.domain].network
        if endpoint.kind == "load":
            return float(network.load_flow_set[endpoint.device_pos])
        return float(network.source_flow_set[endpoint.device_pos])

    def _set_fixed_fluid_endpoint_value(self, endpoint, value: float) -> None:
        network = self.fluid_calcs[endpoint.domain].network
        value = float(value)
        if endpoint.kind == "load":
            old_value = float(network.load_flow_set[endpoint.device_pos])
            network.load_flow_set[endpoint.device_pos] = value
            network.demand[endpoint.solver_pos] += value - old_value
            network.loads[endpoint.device_pos].flow_set = value
            return
        old_value = float(network.source_flow_set[endpoint.device_pos])
        network.source_flow_set[endpoint.device_pos] = value
        network.fixed_injection[endpoint.solver_pos] += value - old_value
        network.sources[endpoint.device_pos].flow_set = value

    def _prepare_energy_coupling_plans(
        self,
        uncoupled_state: np.ndarray,
        state_offset: int,
        eq_offset: int,
    ) -> np.ndarray:
        self.energy_coupling_plans = []
        initial = []
        released_endpoints = set()
        for coupling in self.multi_energy.couplings:
            if not coupling.active or not coupling.supports_energy_balance:
                continue
            if coupling.is_electric_heat_control:
                electric_terminal = coupling.electric_terminal
                heat_terminal = coupling.heat_terminal
                if electric_terminal is None or heat_terminal is None:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: missing electric or heat endpoint"
                    )
                    continue
                electric, reason = self._resolve_energy_endpoint(
                    electric_terminal,
                    adjustable=False,
                )
                if electric is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat, reason = self._resolve_energy_endpoint(
                    heat_terminal,
                    adjustable=False,
                )
                if heat is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                if heat.domain != "heat" or heat.kind not in {"source", "storage"}:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: heat endpoint must be a HeatSource or HeatStorage"
                    )
                    continue
                adjustable_kind = "temperature" if coupling.control_type == "P" else "power"
                adjustable = heat if adjustable_kind == "temperature" else electric
                key = (
                    adjustable.domain,
                    adjustable.device_type,
                    adjustable.device_idx,
                    adjustable_kind,
                )
                if key in released_endpoints:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: endpoint {key} is already controlled by another coupling"
                    )
                    continue

                heat_calc = self.fluid_calcs["heat"]
                heat_network = heat_calc.network
                source_pos = int(heat.device_pos)
                supply_node = int(heat_network.source_supply_node_pos[source_pos])
                return_node = int(heat_network.source_return_node_pos[source_pos])
                heat_state_slice = self.fluid_state_slices["heat"]
                supply_temperature_col = (
                    heat_state_slice.start
                    + heat_calc.base_temperature
                    + int(heat_network.supply_temperature_state_by_node[supply_node])
                )
                return_temperature_col = (
                    heat_state_slice.start
                    + heat_calc.base_temperature
                    + int(heat_network.return_temperature_state_by_node[return_node])
                )
                heat_temperature_eq_row = (
                    self.fluid_eq_slices["heat"].start
                    + heat_calc.hydraulic_state_count
                    + int(heat_network.supply_temperature_state_by_node[supply_node])
                )
                electric_setpoint = self._energy_endpoint_setpoint(electric)
                source_flow, _flow_cols, _flow_derivatives = (
                    self._energy_endpoint_value_and_derivative(heat, uncoupled_state)
                )
                t_return = float(uncoupled_state[return_temperature_col])
                supply_temperature_set = float(
                    heat_network.source_supply_temperature_set[source_pos]
                )
                coefficient = float(coupling.e2h_coeff)
                heat_capacity = max(float(heat_network.medium.heat_capacity), 1.0e-12)
                power_scale = float(getattr(self.network, "p_base_kW", 1.0))
                if coupling.control_type == "P":
                    initial_value = (
                        t_return
                        + electric_setpoint * power_scale * coefficient
                        / (source_flow * heat_capacity)
                        if abs(source_flow) > 1.0e-12
                        else supply_temperature_set
                    )
                    controlled_setpoint = electric_setpoint
                else:
                    initial_value = (
                        source_flow
                        * heat_capacity
                        * (supply_temperature_set - t_return)
                        / (power_scale * coefficient)
                    )
                    controlled_setpoint = supply_temperature_set
                energy_scale = max(
                    1.0,
                    abs(electric_setpoint * power_scale * coefficient),
                    abs(
                        source_flow
                        * heat_capacity
                        * (supply_temperature_set - t_return)
                    ),
                )
                plan_pos = len(self.energy_coupling_plans)
                self.energy_coupling_plans.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=electric if coupling.t1 == electric_terminal else heat,
                        t2=heat if coupling.t2 == heat_terminal else electric,
                        electric=electric,
                        heat=heat,
                        controlled=(electric if coupling.control_type == "P" else heat),
                        dependent=(heat if coupling.control_type == "P" else electric),
                        adjustable=adjustable,
                        adjustable_kind=adjustable_kind,
                        controlled_setpoint=float(controlled_setpoint),
                        electric_setpoint=float(electric_setpoint),
                        supply_temperature_set=supply_temperature_set,
                        supply_temperature_col=supply_temperature_col,
                        return_temperature_col=return_temperature_col,
                        heat_temperature_eq_row=heat_temperature_eq_row,
                        heat_capacity=heat_capacity,
                        power_scale=power_scale,
                        state_col=state_offset + plan_pos,
                        eq_row=eq_offset + plan_pos,
                        normalization=energy_scale,
                        electric_heat=True,
                        hydrogen_electric=False,
                    )
                )
                released_endpoints.add(key)
                initial.append(float(initial_value))
                continue
            if coupling.is_hydrogen_electric_control:
                dependent_terminal = coupling.dependent_terminal
                controlled_terminal = coupling.controlled_terminal
                if dependent_terminal is None or controlled_terminal is None:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: missing electric or hydrogen endpoint"
                    )
                    continue
                t1_is_dependent = coupling.t1 == dependent_terminal
                t1, reason = self._resolve_energy_endpoint(
                    coupling.t1,
                    adjustable=t1_is_dependent,
                    allow_missing_fluid_balance=(
                        t1_is_dependent and coupling.t1.domain not in {"ac", "dc"}
                    ),
                )
                if t1 is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                t2, reason = self._resolve_energy_endpoint(
                    coupling.t2,
                    adjustable=not t1_is_dependent,
                    allow_missing_fluid_balance=(
                        not t1_is_dependent and coupling.t2.domain not in {"ac", "dc"}
                    ),
                )
                if t2 is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                dependent = t1 if t1_is_dependent else t2
                controlled = t2 if t1_is_dependent else t1
                if controlled.balance_row < 0 and controlled.domain in {"ac", "dc"}:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: controlled endpoint has no balance row"
                    )
                    continue
                if controlled.domain not in {"ac", "dc"} and controlled.kind != "load":
                    controlled_network = self.fluid_calcs[controlled.domain].network
                    if str(controlled_network.source_control_type[controlled.device_pos]) != "FLOW":
                        self._add_multi_energy_warning(
                            f"{coupling.table_name}:{coupling.name}: FLOW control requires a flow-controlled hydrogen endpoint"
                        )
                        continue
                key = (dependent.domain, dependent.device_type, dependent.device_idx)
                if key in released_endpoints:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: endpoint {key} is already controlled by another coupling"
                    )
                    continue
                controlled_setpoint = self._energy_endpoint_setpoint(controlled)
                dependent_initial = hydrogen_electric_dependent_value(
                    coupling,
                    controlled_setpoint,
                    float(getattr(self.network, "p_base_kW", 1.0)),
                )
                if dependent.domain not in {"ac", "dc"} and dependent.balance_row < 0:
                    self._set_fixed_fluid_endpoint_value(dependent, dependent_initial)
                plan_pos = len(self.energy_coupling_plans)
                self.energy_coupling_plans.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        controlled=controlled,
                        dependent=dependent,
                        adjustable=dependent,
                        controlled_setpoint=float(controlled_setpoint),
                        state_col=state_offset + plan_pos,
                        eq_row=eq_offset + plan_pos,
                        normalization=max(1.0, abs(float(dependent_initial))),
                        electric_heat=False,
                        hydrogen_electric=True,
                    )
                )
                released_endpoints.add(key)
                initial.append(float(dependent_initial))
                continue
            t1, reason = self._resolve_energy_endpoint(coupling.t1, adjustable=True)
            if t1 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            key = (t1.domain, t1.device_type, t1.device_idx)
            if key in released_endpoints:
                self._add_multi_energy_warning(
                    f"{coupling.table_name}:{coupling.name}: endpoint {key} is already controlled by another coupling"
                )
                continue
            t2, reason = self._resolve_energy_endpoint(coupling.t2, adjustable=False)
            if t2 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            t1_value, _columns, _derivatives = self._energy_endpoint_value_and_derivative(
                t1,
                uncoupled_state,
            )
            t1.energy_scale = (
                float(getattr(self.network, "p_base_kW", 1.0))
                if t1.domain in {"ac", "dc"}
                else float(coupling.energy_factor)
            )
            t2.energy_scale = (
                float(getattr(self.network, "p_base_kW", 1.0))
                if t2.domain in {"ac", "dc"}
                else float(coupling.energy_factor)
            )
            t2_value, _columns, _derivatives = self._energy_endpoint_value_and_derivative(
                t2,
                uncoupled_state,
            )
            normalization = max(
                1.0,
                abs(t1_value * t1.energy_scale),
                abs(t2_value * t2.energy_scale),
            )
            plan_pos = len(self.energy_coupling_plans)
            self.energy_coupling_plans.append(
                SimpleNamespace(
                    coupling=coupling,
                    t1=t1,
                    t2=t2,
                    state_col=state_offset + plan_pos,
                    eq_row=eq_offset + plan_pos,
                    normalization=normalization,
                    electric_heat=False,
                    hydrogen_electric=False,
                )
            )
            released_endpoints.add(key)
            initial.append(float(t1_value))
        return np.asarray(initial, dtype=np.float64)

    def _apply_energy_coupling_terms(
        self,
        x: np.ndarray,
        residual: np.ndarray,
        jacobian,
        *,
        jacobian_format: str,
    ):
        if not self.energy_coupling_plans:
            return jacobian
        rows = []
        cols = []
        data = []
        for plan in self.energy_coupling_plans:
            controlled_value = float(x[plan.state_col])
            if plan.electric_heat:
                electric_base, electric_cols, electric_derivatives = (
                    self._energy_endpoint_value_and_derivative(plan.electric, x)
                )
                source_flow, flow_cols, flow_derivatives = (
                    self._energy_endpoint_value_and_derivative(plan.heat, x)
                )
                t_return = float(x[plan.return_temperature_col])
                if plan.coupling.control_type == "P":
                    electric_power = plan.electric_setpoint
                    t_out = controlled_value
                    electric_target = plan.electric_setpoint
                else:
                    electric_power = controlled_value
                    t_out = plan.supply_temperature_set
                    electric_target = controlled_value

                if plan.electric.balance_row >= 0:
                    residual[plan.electric.balance_row] += (
                        plan.electric.correction_sign * (electric_target - electric_base)
                    )
                if plan.coupling.control_type == "P":
                    boundary_delta = controlled_value - plan.supply_temperature_set
                    residual[plan.heat_temperature_eq_row] -= source_flow * boundary_delta
                residual[plan.eq_row] = electric_heat_balance_residual(
                    plan.coupling,
                    electric_power,
                    source_flow,
                    t_out,
                    t_return,
                    plan.heat_capacity,
                    plan.power_scale,
                ) / plan.normalization
                if jacobian is None:
                    continue

                if plan.electric.balance_row >= 0:
                    for column, derivative in zip(electric_cols, electric_derivatives):
                        rows.append(plan.electric.balance_row)
                        cols.append(int(column))
                        data.append(-plan.electric.correction_sign * float(derivative))
                    if plan.coupling.control_type == "T_OUT":
                        rows.append(plan.electric.balance_row)
                        cols.append(plan.state_col)
                        data.append(plan.electric.correction_sign)

                if plan.coupling.control_type == "P":
                    rows.append(plan.heat_temperature_eq_row)
                    cols.append(plan.state_col)
                    data.append(-source_flow)
                    for column, derivative in zip(flow_cols, flow_derivatives):
                        rows.append(plan.heat_temperature_eq_row)
                        cols.append(int(column))
                        data.append(
                            -(controlled_value - plan.supply_temperature_set)
                            * float(derivative)
                        )

                thermal_gap = t_out - t_return
                for column, derivative in zip(flow_cols, flow_derivatives):
                    rows.append(plan.eq_row)
                    cols.append(int(column))
                    data.append(
                        -plan.heat_capacity
                        * thermal_gap
                        * float(derivative)
                        / plan.normalization
                    )
                rows.append(plan.eq_row)
                cols.append(plan.return_temperature_col)
                data.append(source_flow * plan.heat_capacity / plan.normalization)
                rows.append(plan.eq_row)
                cols.append(plan.state_col)
                data.append(
                    -source_flow * plan.heat_capacity / plan.normalization
                    if plan.coupling.control_type == "P"
                    else plan.power_scale
                    * float(plan.coupling.e2h_coeff)
                    / plan.normalization
                )
                continue
            if plan.hydrogen_electric:
                controlled_base, controlled_cols, controlled_derivatives = (
                    self._energy_endpoint_value_and_derivative(plan.controlled, x)
                )
                controlled_correction = plan.controlled_setpoint - controlled_base
                if plan.controlled.balance_row >= 0:
                    residual[plan.controlled.balance_row] += (
                        plan.controlled.correction_sign * controlled_correction
                    )
                dependent_base, dependent_cols, dependent_derivatives = (
                    self._energy_endpoint_value_and_derivative(plan.dependent, x)
                )
                dependent_correction = controlled_value - dependent_base
                if plan.dependent.balance_row >= 0:
                    residual[plan.dependent.balance_row] += (
                        plan.dependent.correction_sign * dependent_correction
                    )
                expected = hydrogen_electric_dependent_value(
                    plan.coupling,
                    plan.controlled_setpoint,
                    float(getattr(self.network, "p_base_kW", 1.0)),
                )
                residual[plan.eq_row] = (
                    controlled_value - expected
                ) / plan.normalization
                if jacobian is None:
                    continue
                for endpoint, columns, derivatives in (
                    (plan.controlled, controlled_cols, controlled_derivatives),
                    (plan.dependent, dependent_cols, dependent_derivatives),
                ):
                    if endpoint.balance_row < 0:
                        continue
                    for column, derivative in zip(columns, derivatives):
                        rows.append(endpoint.balance_row)
                        cols.append(int(column))
                        data.append(-endpoint.correction_sign * float(derivative))
                if plan.dependent.balance_row >= 0:
                    rows.append(plan.dependent.balance_row)
                    cols.append(plan.state_col)
                    data.append(plan.dependent.correction_sign)
                rows.append(plan.eq_row)
                cols.append(plan.state_col)
                data.append(1.0 / plan.normalization)
                continue
            base_value, base_cols, base_derivatives = (
                self._energy_endpoint_value_and_derivative(plan.t1, x)
            )
            correction = controlled_value - base_value
            residual[plan.t1.balance_row] += plan.t1.correction_sign * correction
            if jacobian is not None:
                rows.append(plan.t1.balance_row)
                cols.append(plan.state_col)
                data.append(plan.t1.correction_sign)
                for column, derivative in zip(base_cols, base_derivatives):
                    rows.append(plan.t1.balance_row)
                    cols.append(int(column))
                    data.append(-plan.t1.correction_sign * float(derivative))

            dependent_value, dependent_cols, dependent_derivatives = (
                self._energy_endpoint_value_and_derivative(plan.t2, x)
            )
            efficiency = float(plan.coupling.efficiency)
            residual[plan.eq_row] = (
                abs(dependent_value) * plan.t2.energy_scale
                - efficiency * abs(controlled_value) * plan.t1.energy_scale
            ) / plan.normalization
            if jacobian is None:
                continue
            controlled_sign = 1.0 if controlled_value >= 0.0 else -1.0
            dependent_sign = 1.0 if dependent_value >= 0.0 else -1.0
            rows.append(plan.eq_row)
            cols.append(plan.state_col)
            data.append(
                -efficiency
                * controlled_sign
                * plan.t1.energy_scale
                / plan.normalization
            )
            for column, derivative in zip(dependent_cols, dependent_derivatives):
                rows.append(plan.eq_row)
                cols.append(int(column))
                data.append(
                    dependent_sign
                    * plan.t2.energy_scale
                    * float(derivative)
                    / plan.normalization
                )

        if jacobian is None or not rows:
            return jacobian
        correction = coo_matrix(
            (
                np.asarray(data, dtype=np.float64),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(self.total_eq, self.total_vars),
        )
        jacobian = jacobian + correction.asformat(jacobian_format)
        return jacobian.asformat(jacobian_format)

    def _apply_final_energy_endpoint_value(self, endpoint, value, ac_result, dc_result) -> None:
        value = float(value)
        if endpoint.domain in {"ac", "dc"}:
            endpoint.table[endpoint.row, endpoint.columns["p"]] = value
            result = ac_result if endpoint.domain == "ac" else dc_result
            if isinstance(result, dict) and endpoint.table_key in result:
                result[endpoint.table_key][endpoint.row, endpoint.columns["p"]] = value
            return

        calc = self.fluid_calcs[endpoint.domain]
        result = calc.lf_result
        network = calc.network
        if endpoint.kind == "load":
            if "load_flow" in result.arrays:
                result.arrays["load_flow"][endpoint.device_pos] = value
            name = str(network.load_name[endpoint.device_pos])
            if name in result.loads:
                result.loads[name].flow = value
            return

        calc.source_flow[endpoint.device_pos] = value
        if "source_flow" in result.arrays:
            result.arrays["source_flow"][endpoint.device_pos] = value
        if endpoint.kind == "storage" and "storage_flow" in result.arrays:
            storage_rows = np.flatnonzero(network.storage_source_pos == endpoint.device_pos)
            if storage_rows.size:
                result.arrays["storage_flow"][int(storage_rows[0])] = value
        collection = result.storages if endpoint.kind == "storage" else result.sources
        name = str(network.source_name[endpoint.device_pos])
        if name in collection:
            collection[name].flow = value

    def _apply_final_coupling_endpoint_controls(self, ac_result, dc_result) -> None:
        for plan in self.energy_coupling_plans:
            if plan.electric_heat:
                electric_power = (
                    plan.electric_setpoint
                    if plan.coupling.control_type == "P"
                    else float(self.x[plan.state_col])
                )
                t_out = (
                    float(self.x[plan.state_col])
                    if plan.coupling.control_type == "P"
                    else plan.supply_temperature_set
                )
                self._apply_final_energy_endpoint_value(
                    plan.electric,
                    electric_power,
                    ac_result,
                    dc_result,
                )
                heat_calc = self.fluid_calcs["heat"]
                heat_result = heat_calc.lf_result
                source_pos = int(plan.heat.device_pos)
                if heat_result is not None:
                    heat_result.arrays.setdefault(
                        "source_t_out",
                        np.full(
                            len(heat_calc.network.sources),
                            np.nan,
                            dtype=np.float64,
                        ),
                    )[source_pos] = t_out
                    source_name = str(heat_calc.network.source_name[source_pos])
                    collection = (
                        heat_result.storages
                        if plan.heat.kind == "storage"
                        else heat_result.sources
                    )
                    if source_name in collection:
                        collection[source_name].t_out = t_out
                continue
            if plan.hydrogen_electric:
                self._apply_final_energy_endpoint_value(
                    plan.controlled,
                    plan.controlled_setpoint,
                    ac_result,
                    dc_result,
                )
                self._apply_final_energy_endpoint_value(
                    plan.dependent,
                    self.x[plan.state_col],
                    ac_result,
                    dc_result,
                )
            else:
                self._apply_final_energy_endpoint_value(
                    plan.t1,
                    self.x[plan.state_col],
                    ac_result,
                    dc_result,
                )

    def _clear_converter_outputs(self):
        ppc = getattr(self.network, "ppc", {}) or {}
        dcac = ppc.get("dcac")
        if dcac is not None and getattr(dcac, "size", 0):
            dcac[:, [DCAC_COLS[name] for name in ("dc_p", "ac_p", "ac_q", "dc_i", "ac_i")]] = 0.0
        if not self._converter_ppc_mode:
            for conv in getattr(self.network, "dcac_converters", ()):
                conv.is_alive = False
                for name in ("dc_p", "ac_p", "ac_q", "dc_i", "ac_i"):
                    setattr(conv, name, 0.0)

    def _sync_sub_result_modes(self) -> None:
        if self.ac_calc is not None:
            self.ac_calc.result_mode = "array"
            self.ac_calc.keep_node_objects = False
        if self.dc_calc is not None:
            self.dc_calc.result_mode = "array"
            self.dc_calc.keep_node_objects = False

    def _install_ac_converter_voltage_control_nodes(self):
        """Expose converter V-control terminals to the AC topology-aware LF layout."""
        if self.ac_calc is None:
            return
        controlled_nodes = []
        if self._converter_ppc_mode:
            dcac = np.asarray(self.network.ppc.get("dcac", ()), dtype=np.float64)
            if dcac.size:
                run = dcac[:, DCAC_COLS["run_stat"]].astype(np.int8, copy=False) == 1
                ac_control = dcac[:, DCAC_COLS["ac_control_type"]].astype(np.int8, copy=False)
                mask = run & np.isin(
                    ac_control,
                    (DCAC_AC_CONTROL_CODE["PV"], DCAC_AC_CONTROL_CODE["PH"]),
                )
                if np.any(mask):
                    controlled_nodes.append(dcac[mask, DCAC_COLS["ac_node"]].astype(np.int64, copy=False))

        else:
            for conv in self.network.dcac_converters:
                if conv.run_stat == 1 and str(conv.ac_control_type).upper() in {"PV", "PH"}:
                    controlled_nodes.append(np.asarray([conv.ac_node], dtype=np.int64))

        if controlled_nodes:
            self.ac_calc.ppc["_external_voltage_control_node_ids"] = np.unique(
                np.concatenate(controlled_nodes)
            ).astype(np.int64, copy=False)
        else:
            self.ac_calc.ppc.pop("_external_voltage_control_node_ids", None)

    def prepare(self):
        if self._prepared:
            return self.x
        electric_x = self._prepare_electric().copy()
        self.electric_vars = int(self.total_vars)
        self.electric_eq = int(self.total_eq)
        self._electric_residual_work = self._residual_work
        state_parts = [electric_x] if electric_x.size else []
        state_offset = self.electric_vars
        eq_offset = self.electric_eq
        self.fluid_state_slices = {}
        self.fluid_eq_slices = {}
        for name, calc in self.fluid_calcs.items():
            calc.result_mode = self.result_mode
            calc.verbose = self.verbose
            if not calc.prepared:
                calc.prepare()
            state_slice = slice(state_offset, state_offset + calc.total_vars)
            eq_slice = slice(eq_offset, eq_offset + calc.total_eq)
            self.fluid_state_slices[name] = state_slice
            self.fluid_eq_slices[name] = eq_slice
            state_parts.append(calc.x.copy())
            state_offset = state_slice.stop
            eq_offset = eq_slice.stop
        uncoupled_state = (
            np.concatenate(state_parts) if state_parts else np.empty(0, dtype=np.float64)
        )
        coupling_initial = self._prepare_energy_coupling_plans(
            uncoupled_state,
            state_offset,
            eq_offset,
        )
        if coupling_initial.size:
            state_parts.append(coupling_initial)
            self.energy_coupling_state_slice = slice(
                state_offset,
                state_offset + coupling_initial.size,
            )
            self.energy_coupling_eq_slice = slice(
                eq_offset,
                eq_offset + coupling_initial.size,
            )
            state_offset = self.energy_coupling_state_slice.stop
            eq_offset = self.energy_coupling_eq_slice.stop
        else:
            self.energy_coupling_state_slice = slice(state_offset, state_offset)
            self.energy_coupling_eq_slice = slice(eq_offset, eq_offset)
        self.x = np.concatenate(state_parts) if state_parts else np.empty(0, dtype=np.float64)
        self.total_vars = int(state_offset)
        self.total_eq = int(eq_offset)
        self.last_jacobian_shape = (self.total_eq, self.total_vars)
        if self.fluid_calcs:
            self._single_ac_newton_block = False
            self._single_dc_newton_block = False
        if self.verbose:
            fluid_vars = self.total_vars - self.electric_vars
            print(
                "Multi-energy prepare:",
                f"electric_vars={self.electric_vars}",
                f"fluid_vars={fluid_vars}",
                f"total_vars={self.total_vars}",
                f"total_eq={self.total_eq}",
            )
        self._prepared = True
        return self.x

    def _prepare_electric(self):
        """Build the global hybrid state vector and block equation layout.

        调用方无需先对 HybridPowerNetwork 显式执行 `prepare()` 或 `topo()`；
        HybridPowerFlowCalc.prepare() 会直接准备 AC/DC 子求解器和全局状态布局。
        """
        self._sync_sub_result_modes()
        self._install_ac_converter_voltage_control_nodes()
        parts = []
        if self.ac_calc is not None:
            self.ac_calc.verbose = self.verbose
            self.ac_calc.prepare()
            self.ac_size = self.ac_calc.total_vars
            self.ac_eq = self.ac_calc.total_eq
            parts.append(self.ac_calc.x.copy())
        if self.dc_calc is not None:
            self.dc_calc.prepare()
            self.dc_G = self.dc_calc.G
            dc_x = self.dc_calc.x
            self.dc_size = self.dc_calc.total_vars
            self.dc_eq = self.dc_calc.total_eq
            parts.append(dc_x.copy())
        self._prepare_dcac_converters()
        if not parts:
            self.x = np.empty(0, dtype=np.float64)
            self.total_vars = 0
            self.total_eq = 0
            self.electric_vars = 0
            self.electric_eq = 0
            self.last_jacobian_shape = (0, 0)
            self._residual_work = np.empty(0, dtype=np.float64)
            return self.x
        # 全局变量布局：[AC 内部变量（含 ACAC）][DC 内部变量][每台 DCAC: Pdc, Pac, Qac]
        self.dcac_start = self.ac_size + self.dc_size
        dcac_x = self._initial_dcac_x()
        if dcac_x.size:
            parts.append(dcac_x)
        self.dcac_eq_start = self.ac_eq + self.dc_eq
        self.x = np.concatenate(parts)
        self.total_vars = self.x.size
        self.total_eq = self.dcac_eq_start + self.N_dcac * 3
        self.electric_vars = int(self.total_vars)
        self.electric_eq = int(self.total_eq)
        # Variable/equation order is block diagonal first, then converter coupling rows.
        self.last_jacobian_shape = (self.total_eq, self.total_vars)
        self._residual_work = np.empty(self.total_eq, dtype=np.float64)
        self._single_ac_newton_block = (
            self.ac_calc is not None
            and self.dc_calc is None
            and self.N_dcac == 0
            and self.ac_size == self.total_vars
            and self.ac_eq == self.total_eq
        )
        self._single_dc_newton_block = (
            self.dc_calc is not None
            and self.ac_calc is None
            and self.N_dcac == 0
            and self.dc_size == self.total_vars
            and self.dc_eq == self.total_eq
        )
        self._cache_converter_jacobian_structure()
        if self._single_ac_newton_block or self._single_dc_newton_block:
            self._clear_global_jacobian_pattern()
        else:
            self._cache_global_jacobian_pattern()
        if self.verbose:
            print(
                "Hybrid prepare:",
                f"ac_vars={self.ac_size}",
                f"dc_vars={self.dc_size}",
                f"dcac_vars={self.N_dcac * 3}",
                f"total_vars={self.x.size}",
                f"total_eq={self.last_jacobian_shape[0]}",
            )
        return self.x

    def _split_x(self, x):
        """Return AC, DC and DCAC slices from the global Newton vector."""
        ac_x = x[:self.ac_size]
        dc_x = x[self.ac_size:self.ac_size + self.dc_size]
        dcac_x = x[self.dcac_start:self.dcac_start + self.N_dcac * 3]
        return ac_x, dc_x, dcac_x

    def _prepare_dcac_converters(self):
        """Validate live DCAC converters and map terminal nodes to AC/DC solver indices."""
        self.dcac_converters = []
        self._clear_dcac_arrays()
        if not self.has_ac or not self.has_dc:
            return
        if self._converter_ppc_mode:
            self._prepare_dcac_converters_from_ppc()
            return
        for conv in self.network.dcac_converters:
            conv.is_alive = False
            if conv.run_stat != 1:
                continue
            if conv.ac_node not in self.ac_calc.node_pos:
                continue
            if conv.dc_node not in self.dc_calc.alive_node_dict:
                continue
            conv.ac_node_obj = self._ac_node_obj_by_idx.get(int(conv.ac_node))
            conv.dc_node_obj = self._dc_node_obj_by_idx.get(int(conv.dc_node))
            ac_pos = self.ac_calc.node_pos[conv.ac_node]
            dc_pos = self.dc_calc.alive_node_dict[conv.dc_node]
            ac_ctrl, dc_ctrl = validate_dcac_control_types(
                getattr(conv, "ac_control_type", "NONE"),
                getattr(conv, "dc_control_type", "NONE"),
            )
            self._validate_dcac_ac_terminal_control(conv.idx, ac_pos, ac_ctrl)
            conv.is_alive = True
            self.dcac_converters.append((conv, ac_pos, dc_pos))
        self.N_dcac = len(self.dcac_converters)
        self._cache_dcac_arrays()

    def _ac_solver_pos(self, node_id: int) -> int:
        node_id = int(node_id)
        lookup = getattr(self.ac_calc, "_active_node_solver_lookup", None)
        if isinstance(lookup, np.ndarray) and lookup.size and 0 <= node_id < lookup.size:
            pos = int(lookup[node_id])
            if pos >= 0:
                return pos
        pos = self.ac_calc.node_pos.get(node_id) if getattr(self.ac_calc, "node_pos", None) else None
        if pos is not None:
            return int(pos)
        active_ids = getattr(self.ac_calc, "ppc_node_idx", np.array([], dtype=np.int64))
        if active_ids.size:
            matches = np.flatnonzero(active_ids.astype(np.int64, copy=False) == node_id)
            if matches.size:
                return int(matches[0])
        return -1

    def _dc_solver_pos(self, node_id: int) -> int:
        return int(self.dc_calc.alive_node_dict.get(int(node_id), -1))

    def _ac_balance_rows(self, positions):
        """Map AC solver positions to the P/Q balance rows that still exist."""
        positions = np.asarray(positions, dtype=np.int32)
        p_rows = np.asarray(self.ac_calc.theta_idx, dtype=np.int32)[positions]
        v_idx = np.asarray(self.ac_calc.V_idx, dtype=np.int32)[positions]
        q_rows = np.full(positions.size, -1, dtype=np.int32)
        active_q = v_idx >= 0
        q_rows[active_q] = self.ac_calc.n_theta + v_idx[active_q]
        return p_rows, q_rows

    def _validate_dcac_ac_terminal_control(self, converter_idx, ac_pos, ac_control_type):
        """Reject only duplicate AC voltage controls after topology grouping."""
        if ac_control_type not in {"PV", "PH"}:
            return
        node_type = self.ac_calc.node_type[int(ac_pos)]
        if node_type == AC_NODE_TYPE_PQ:
            return
        current_type = ac_node_type_label(node_type)
        raise ValueError(
            f"DCACConverter[{converter_idx}] 的 AC 端 {ac_control_type} 控制与"
            f"拓扑后母线已有的 {current_type} 控制重复"
        )

    def _prepare_dcac_converters_from_ppc(self):
        table = self.network.ppc.get("dcac")
        if table is None or table.size == 0:
            return
        rows = []
        ac_pos = []
        dc_pos = []
        ac_control_codes = []
        dc_control_codes = []
        for row_pos, row in enumerate(table):
            if int(row[DCAC_COLS["run_stat"]]) != 1:
                continue
            ac_node = int(row[DCAC_COLS["ac_node"]])
            dc_node = int(row[DCAC_COLS["dc_node"]])
            ac_solver_pos = self._ac_solver_pos(ac_node)
            if ac_solver_pos < 0:
                continue
            dc_solver_pos = self._dc_solver_pos(dc_node)
            if dc_solver_pos < 0:
                continue
            ac_control_code = int(row[DCAC_COLS["ac_control_type"]])
            dc_control_code = int(row[DCAC_COLS["dc_control_type"]])
            ac_ctrl_label, dc_ctrl_label = validate_dcac_control_types(
                DCAC_AC_CONTROL_LABEL.get(ac_control_code, "NONE"),
                DCAC_DC_CONTROL_LABEL.get(dc_control_code, "NONE"),
            )
            self._validate_dcac_ac_terminal_control(
                int(row[DCAC_COLS["idx"]]),
                ac_solver_pos,
                ac_ctrl_label,
            )
            rows.append(row_pos)
            ac_pos.append(ac_solver_pos)
            dc_pos.append(dc_solver_pos)
            ac_control_codes.append(DCAC_AC_CONTROL_CODE[ac_ctrl_label])
            dc_control_codes.append(DCAC_DC_CONTROL_CODE[dc_ctrl_label])
        self.N_dcac = len(rows)
        if self.N_dcac == 0:
            return
        self.dcac_row_pos = np.asarray(rows, dtype=np.int32)
        active = table[self.dcac_row_pos]
        self.dcac_idx = active[:, DCAC_COLS["idx"]].astype(np.int32, copy=True)
        names = self.network.ppc.get("dcac_name", np.asarray([], dtype=object))
        self.dcac_names = np.asarray(names[self.dcac_row_pos], dtype=object) if len(names) else np.asarray(
            [str(idx) for idx in self.dcac_idx],
            dtype=object,
        )
        self.dcac_ac_pos = np.asarray(ac_pos, dtype=np.int32)
        self.dcac_dc_pos = np.asarray(dc_pos, dtype=np.int32)
        self.dcac_ac_control_code = np.asarray(ac_control_codes, dtype=np.int8)
        self.dcac_dc_control_code = np.asarray(dc_control_codes, dtype=np.int8)
        self.dcac_r1 = active[:, DCAC_COLS["r1"]].astype(np.float64, copy=True)
        self.dcac_r2 = active[:, DCAC_COLS["r2"]].astype(np.float64, copy=True)
        self.dcac_v_dc_set = active[:, DCAC_COLS["v_dc_set"]].astype(np.float64, copy=True)
        self.dcac_v_ac_set = active[:, DCAC_COLS["v_ac_set"]].astype(np.float64, copy=True)
        self.dcac_p_ac_set = active[:, DCAC_COLS["p_ac_set"]].astype(np.float64, copy=True)
        self.dcac_p_dc_set = active[:, DCAC_COLS["p_dc_set"]].astype(np.float64, copy=True)
        self.dcac_q_ac_set = active[:, DCAC_COLS["q_ac_set"]].astype(np.float64, copy=True)
        self.dcac_ac_p_row, self.dcac_ac_q_row = self._ac_balance_rows(self.dcac_ac_pos)
        self.dcac_ac_p_eq_mask = self.dcac_ac_p_row >= 0
        self.dcac_ac_q_eq_mask = self.dcac_ac_q_row >= 0
        self.dcac_dc_eq = np.asarray([self.dc_calc.node_eq[int(pos)] for pos in self.dcac_dc_pos], dtype=np.int32)
        self.dcac_dc_eq_mask = self.dcac_dc_eq >= 0
        self.dcac_ac_theta_set = self.ac_calc.ppc_node_angle[self.dcac_ac_pos].astype(np.float64, copy=True)
        self.dcac_ac_theta_col = self.dcac_ac_p_row.copy()
        self.dcac_ac_v_col = self.dcac_ac_q_row.copy()
        self.dcac_dc_v_col = self.ac_size + self.dcac_dc_pos

    def _clear_dcac_arrays(self):
        self.dcac_devices = []
        self.dcac_row_pos = np.array([], dtype=np.int32)
        self.dcac_idx = np.array([], dtype=np.int32)
        self.dcac_names = np.array([], dtype=object)
        self.dcac_ac_pos = np.array([], dtype=np.int32)
        self.dcac_dc_pos = np.array([], dtype=np.int32)
        self.dcac_ac_control_code = np.array([], dtype=np.int8)
        self.dcac_dc_control_code = np.array([], dtype=np.int8)
        self.dcac_r1 = np.array([], dtype=np.float64)
        self.dcac_r2 = np.array([], dtype=np.float64)
        self.dcac_v_dc_set = np.array([], dtype=np.float64)
        self.dcac_v_ac_set = np.array([], dtype=np.float64)
        self.dcac_p_ac_set = np.array([], dtype=np.float64)
        self.dcac_p_dc_set = np.array([], dtype=np.float64)
        self.dcac_q_ac_set = np.array([], dtype=np.float64)
        self.dcac_ac_p_row = np.array([], dtype=np.int32)
        self.dcac_ac_q_row = np.array([], dtype=np.int32)
        self.dcac_ac_p_eq_mask = np.array([], dtype=bool)
        self.dcac_ac_q_eq_mask = np.array([], dtype=bool)
        self.dcac_dc_eq = np.array([], dtype=np.int32)
        self.dcac_dc_eq_mask = np.array([], dtype=bool)
        self.dcac_ac_theta_set = np.array([], dtype=np.float64)
        self.dcac_ac_theta_col = np.array([], dtype=np.int32)
        self.dcac_ac_v_col = np.array([], dtype=np.int32)
        self.dcac_dc_v_col = np.array([], dtype=np.int32)

    def _cache_dcac_arrays(self):
        """Cache DCAC converter metadata as arrays for residual/Jacobian assembly."""
        if not self.dcac_converters:
            self._clear_dcac_arrays()
            return
        self.dcac_ac_pos = np.asarray([item[1] for item in self.dcac_converters], dtype=np.int32)
        self.dcac_dc_pos = np.asarray([item[2] for item in self.dcac_converters], dtype=np.int32)
        convs = [item[0] for item in self.dcac_converters]
        control_pairs = [
            validate_dcac_control_types(conv.ac_control_type, conv.dc_control_type)
            for conv in convs
        ]
        self.dcac_ac_control_code = np.asarray(
            [DCAC_AC_CONTROL_CODE[ac_control] for ac_control, _dc_control in control_pairs],
            dtype=np.int8,
        )
        self.dcac_dc_control_code = np.asarray(
            [DCAC_DC_CONTROL_CODE[dc_control] for _ac_control, dc_control in control_pairs],
            dtype=np.int8,
        )
        self.dcac_devices = convs
        self.dcac_r1 = np.asarray([conv.r1 for conv in convs], dtype=np.float64)
        self.dcac_r2 = np.asarray([conv.r2 for conv in convs], dtype=np.float64)
        self.dcac_v_dc_set = np.asarray([conv.v_dc_set for conv in convs], dtype=np.float64)
        self.dcac_v_ac_set = np.asarray([conv.v_ac_set for conv in convs], dtype=np.float64)
        self.dcac_p_ac_set = np.asarray([conv.p_ac_set for conv in convs], dtype=np.float64)
        self.dcac_p_dc_set = np.asarray([conv.p_dc_set for conv in convs], dtype=np.float64)
        self.dcac_q_ac_set = np.asarray([conv.q_ac_set for conv in convs], dtype=np.float64)
        self.dcac_ac_p_row, self.dcac_ac_q_row = self._ac_balance_rows(self.dcac_ac_pos)
        self.dcac_ac_p_eq_mask = self.dcac_ac_p_row >= 0
        self.dcac_ac_q_eq_mask = self.dcac_ac_q_row >= 0
        self.dcac_dc_eq = np.asarray([self.dc_calc.node_eq[int(pos)] for pos in self.dcac_dc_pos], dtype=np.int32)
        self.dcac_dc_eq_mask = self.dcac_dc_eq >= 0
        self.dcac_ac_theta_set = self.ac_calc.ppc_node_angle[self.dcac_ac_pos].astype(np.float64, copy=True)
        self.dcac_ac_theta_col = self.dcac_ac_p_row.copy()
        self.dcac_ac_v_col = self.dcac_ac_q_row.copy()
        self.dcac_dc_v_col = self.ac_size + self.dcac_dc_pos

    def _clear_converter_jacobian_structure(self):
        """Reset cached sparse row/column patterns for converter Jacobian terms."""
        self.dcac_dc_p_col = np.array([], dtype=np.int32)
        self.dcac_ac_p_col = np.array([], dtype=np.int32)
        self.dcac_ac_q_col = np.array([], dtype=np.int32)
        self.dcac_eq_loss = np.array([], dtype=np.int32)
        self.dcac_eq_ctrl_1 = np.array([], dtype=np.int32)
        self.dcac_eq_ctrl_2 = np.array([], dtype=np.int32)
        self.dcac_loss_rows = np.array([], dtype=np.int32)
        self.dcac_loss_cols = np.array([], dtype=np.int32)
        self.dcac_dc_eq_rows = np.array([], dtype=np.int32)
        self.dcac_dc_eq_cols = np.array([], dtype=np.int32)
        self.dcac_ctrl_dc_v_mask = np.array([], dtype=bool)
        self.dcac_ctrl_ac_v_mask = np.array([], dtype=bool)
        self.dcac_ctrl_ac_p_mask = np.array([], dtype=bool)
        self.dcac_ctrl_dc_p_mask = np.array([], dtype=bool)
        self.dcac_ctrl_q_mask = np.array([], dtype=bool)
        self.dcac_ctrl_ac_theta_mask = np.array([], dtype=bool)
        self.dcac_dcv_avg_set = np.array([], dtype=np.float64)
        self.dcac_dcv_rep_mask = np.array([], dtype=bool)
        self.dcac_dcv_share_mask = np.array([], dtype=bool)
        self.dcac_dcv_ref_idx = np.array([], dtype=np.int32)
        self.dcac_acv_v_avg_set = np.array([], dtype=np.float64)
        self.dcac_acv_theta_avg_set = np.array([], dtype=np.float64)
        self.dcac_acv_rep_mask = np.array([], dtype=bool)
        self.dcac_acv_share_mask = np.array([], dtype=bool)
        self.dcac_acv_ref_idx = np.array([], dtype=np.int32)
        self.dcac_ones = np.array([], dtype=np.float64)
        self.dcac_dc_eq_ones = np.array([], dtype=np.float64)
        self.dcac_ctrl_rows = np.array([], dtype=np.int32)
        self.dcac_ctrl_cols = np.array([], dtype=np.int32)
        self.dcac_ctrl_data = np.array([], dtype=np.float64)

    def _clear_global_jacobian_pattern(self):
        self.global_jac_raw_data = np.array([], dtype=np.float64)
        self.global_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.global_jac_csr_indices = np.array([], dtype=np.int32)
        self.global_jac_csr_indptr = np.array([], dtype=np.int32)
        self.global_jac_csr_data = np.array([], dtype=np.float64)
        self.global_jac_csr_sum_plan = build_raw_sum_plan(self.global_jac_raw_to_csr_pos, 0)
        self.global_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.global_jac_csc_indices = np.array([], dtype=np.int32)
        self.global_jac_csc_indptr = np.array([], dtype=np.int32)
        self.global_jac_csc_data = np.array([], dtype=np.float64)
        self.global_jac_csc_sum_plan = build_raw_sum_plan(self.global_jac_raw_to_csc_pos, 0)
        self.global_jac_ac_slice = slice(0, 0)
        self.global_jac_dc_slice = slice(0, 0)
        self.global_jac_dcac_ac_p_slice = slice(0, 0)
        self.global_jac_dcac_ac_q_slice = slice(0, 0)
        self.global_jac_dcac_dc_eq_slice = slice(0, 0)
        self.global_jac_dcac_loss_slice = slice(0, 0)
        self.global_jac_dcac_ctrl_slice = slice(0, 0)

    def _cache_converter_jacobian_structure(self):
        """Precompute converter Jacobian row/column indices once per prepared case."""
        self._clear_converter_jacobian_structure()
        if self.N_dcac:
            idx = np.arange(self.N_dcac, dtype=np.int32)
            self.dcac_dc_p_col = self.dcac_start + 3 * idx
            self.dcac_ac_p_col = self.dcac_dc_p_col + 1
            self.dcac_ac_q_col = self.dcac_dc_p_col + 2
            self.dcac_eq_loss = self.dcac_eq_start + 3 * idx
            self.dcac_eq_ctrl_1 = self.dcac_eq_loss + 1
            self.dcac_eq_ctrl_2 = self.dcac_eq_loss + 2

            # ACV 表示 AC 侧成网定压：第一控制方程固定电压幅值，
            # 第二控制方程固定相角；DCV/ACP 保持原有定 Q 方程。
            variable_ac_v = self.dcac_ac_v_col >= 0
            self.dcac_loss_rows = np.concatenate(
                (
                    self.dcac_eq_loss,
                    self.dcac_eq_loss,
                    self.dcac_eq_loss,
                    self.dcac_eq_loss[variable_ac_v],
                    self.dcac_eq_loss,
                )
            ).astype(np.int32, copy=False)
            self.dcac_loss_cols = np.concatenate(
                (
                    self.dcac_dc_p_col,
                    self.dcac_ac_p_col,
                    self.dcac_ac_q_col,
                    self.dcac_ac_v_col[variable_ac_v],
                    self.dcac_dc_v_col,
                )
            ).astype(np.int32, copy=False)

            if self.dcac_dc_eq_mask.any():
                self.dcac_dc_eq_rows = self.ac_eq + self.dcac_dc_eq[self.dcac_dc_eq_mask]
                self.dcac_dc_eq_cols = self.dcac_dc_p_col[self.dcac_dc_eq_mask]
                self.dcac_dc_eq_ones = np.ones(self.dcac_dc_eq_rows.size, dtype=np.float64)
            self.dcac_ctrl_dc_v_mask = (
                (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["PQ"])
                & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["V"])
            )
            self.dcac_ctrl_ac_v_mask = (
                (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["PH"])
                & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["NONE"])
            )
            self.dcac_ctrl_ac_p_mask = (
                (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["PQ"])
                & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["NONE"])
            )
            self.dcac_ctrl_dc_p_mask = (
                (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["NONE"])
                & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["P"])
            )
            self.dcac_ctrl_ac_theta_mask = self.dcac_ctrl_ac_v_mask.copy()
            self.dcac_ctrl_q_mask = ~self.dcac_ctrl_ac_theta_mask
            (
                self.dcac_dcv_avg_set,
                self.dcac_dcv_rep_mask,
                self.dcac_dcv_share_mask,
                self.dcac_dcv_ref_idx,
            ) = _build_group_share_metadata(self.dcac_dc_pos, self.dcac_v_dc_set, self.dcac_ctrl_dc_v_mask)
            (
                self.dcac_acv_v_avg_set,
                self.dcac_acv_rep_mask,
                self.dcac_acv_share_mask,
                self.dcac_acv_ref_idx,
            ) = _build_group_share_metadata(self.dcac_ac_pos, self.dcac_v_ac_set, self.dcac_ctrl_ac_v_mask)
            (
                self.dcac_acv_theta_avg_set,
                _theta_rep,
                _theta_share,
                _theta_ref,
            ) = _build_group_share_metadata(self.dcac_ac_pos, self.dcac_ac_theta_set, self.dcac_ctrl_ac_v_mask)
            self.dcac_acv_rep_mask = self.dcac_acv_rep_mask | _theta_rep
            self.dcac_acv_share_mask = self.dcac_acv_share_mask | _theta_share
            self.dcac_acv_ref_idx = _theta_ref
            self.dcac_ones = np.ones(self.N_dcac, dtype=np.float64)
            ctrl_rows = []
            ctrl_cols = []
            ctrl_data = []
            if np.any(self.dcac_ctrl_ac_p_mask):
                idx_mask = np.flatnonzero(self.dcac_ctrl_ac_p_mask).astype(np.int32, copy=False)
                ctrl_rows.append(self.dcac_eq_ctrl_1[idx_mask])
                ctrl_cols.append(self.dcac_ac_p_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_ctrl_dc_p_mask):
                idx_mask = np.flatnonzero(self.dcac_ctrl_dc_p_mask).astype(np.int32, copy=False)
                ctrl_rows.append(self.dcac_eq_ctrl_1[idx_mask])
                ctrl_cols.append(self.dcac_dc_p_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_dcv_rep_mask):
                idx_mask = np.flatnonzero(self.dcac_dcv_rep_mask).astype(np.int32, copy=False)
                ctrl_rows.append(self.dcac_eq_ctrl_1[idx_mask])
                ctrl_cols.append(self.dcac_dc_v_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_dcv_share_mask):
                idx_mask = np.flatnonzero(self.dcac_dcv_share_mask).astype(np.int32, copy=False)
                ref = self.dcac_dcv_ref_idx[idx_mask]
                rows = np.repeat(self.dcac_eq_ctrl_1[idx_mask], 2)
                cols = np.empty(2 * idx_mask.size, dtype=np.int32)
                data = np.empty(2 * idx_mask.size, dtype=np.float64)
                cols[0::2] = self.dcac_dc_p_col[idx_mask]
                cols[1::2] = self.dcac_dc_p_col[ref]
                data[0::2] = 1.0
                data[1::2] = -1.0
                ctrl_rows.append(rows)
                ctrl_cols.append(cols)
                ctrl_data.append(data)
            if np.any(self.dcac_acv_rep_mask):
                idx_mask = np.flatnonzero(self.dcac_acv_rep_mask).astype(np.int32, copy=False)
                ctrl_rows.append(np.repeat(self.dcac_eq_ctrl_1[idx_mask], 1))
                ctrl_cols.append(self.dcac_ac_v_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_acv_share_mask):
                idx_mask = np.flatnonzero(self.dcac_acv_share_mask).astype(np.int32, copy=False)
                ref = self.dcac_acv_ref_idx[idx_mask]
                rows = np.repeat(self.dcac_eq_ctrl_1[idx_mask], 2)
                cols = np.empty(2 * idx_mask.size, dtype=np.int32)
                data = np.empty(2 * idx_mask.size, dtype=np.float64)
                cols[0::2] = self.dcac_ac_p_col[idx_mask]
                cols[1::2] = self.dcac_ac_p_col[ref]
                data[0::2] = 1.0
                data[1::2] = -1.0
                ctrl_rows.append(rows)
                ctrl_cols.append(cols)
                ctrl_data.append(data)
            q_mask = self.dcac_ctrl_q_mask & ~self.dcac_ctrl_ac_v_mask
            if np.any(q_mask):
                idx_mask = np.flatnonzero(q_mask).astype(np.int32, copy=False)
                ctrl_rows.append(self.dcac_eq_ctrl_2[idx_mask])
                ctrl_cols.append(self.dcac_ac_q_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_acv_rep_mask):
                idx_mask = np.flatnonzero(self.dcac_acv_rep_mask).astype(np.int32, copy=False)
                ctrl_rows.append(self.dcac_eq_ctrl_2[idx_mask])
                ctrl_cols.append(self.dcac_ac_theta_col[idx_mask])
                ctrl_data.append(np.ones(idx_mask.size, dtype=np.float64))
            if np.any(self.dcac_acv_share_mask):
                idx_mask = np.flatnonzero(self.dcac_acv_share_mask).astype(np.int32, copy=False)
                ref = self.dcac_acv_ref_idx[idx_mask]
                rows = np.repeat(self.dcac_eq_ctrl_2[idx_mask], 2)
                cols = np.empty(2 * idx_mask.size, dtype=np.int32)
                data = np.empty(2 * idx_mask.size, dtype=np.float64)
                cols[0::2] = self.dcac_ac_q_col[idx_mask]
                cols[1::2] = self.dcac_ac_q_col[ref]
                data[0::2] = 1.0
                data[1::2] = -1.0
                ctrl_rows.append(rows)
                ctrl_cols.append(cols)
                ctrl_data.append(data)
            if ctrl_rows:
                self.dcac_ctrl_rows = np.concatenate(ctrl_rows).astype(np.int32, copy=False)
                self.dcac_ctrl_cols = np.concatenate(ctrl_cols).astype(np.int32, copy=False)
                self.dcac_ctrl_data = np.concatenate(ctrl_data).astype(np.float64, copy=False)

    @staticmethod
    def _rows_from_csr_indptr(indptr):
        return np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))

    @staticmethod
    def _cols_from_csc_indptr(indptr):
        return np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))

    def _sub_jacobian_pattern(self, calc, is_dc=False):
        if calc is None or getattr(calc, "total_eq", 0) == 0 or getattr(calc, "total_vars", 0) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        if is_dc and hasattr(calc, "_dc_jac_csc_indices") and calc._dc_jac_csc_indices.size:
            return (
                calc._dc_jac_csc_indices.astype(np.int32, copy=True),
                self._cols_from_csc_indptr(calc._dc_jac_csc_indptr),
            )
        if hasattr(calc, "full_jac_csc_indices") and calc.full_jac_csc_indices.size:
            return (
                calc.full_jac_csc_indices.astype(np.int32, copy=True),
                self._cols_from_csc_indptr(calc.full_jac_csc_indptr),
            )
        if hasattr(calc, "standard_jac_csc_indices") and calc.standard_jac_csc_indices.size:
            return (
                calc.standard_jac_csc_indices.astype(np.int32, copy=True),
                self._cols_from_csc_indptr(calc.standard_jac_csc_indptr),
            )
        if is_dc and hasattr(calc, "_dc_jac_csr_indices") and calc._dc_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc._dc_jac_csr_indptr),
                calc._dc_jac_csr_indices.astype(np.int32, copy=True),
            )
        if hasattr(calc, "full_jac_csr_indices") and calc.full_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc.full_jac_csr_indptr),
                calc.full_jac_csr_indices.astype(np.int32, copy=True),
            )
        if hasattr(calc, "standard_jac_csr_indices") and calc.standard_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc.standard_jac_csr_indptr),
                calc.standard_jac_csr_indices.astype(np.int32, copy=True),
            )
        jac = calc.get_jacobi(calc.x)
        jac = jac.tocsr()
        return self._rows_from_csr_indptr(jac.indptr), jac.indices.astype(np.int32, copy=True)

    @staticmethod
    def _sub_jacobian_cached_values(calc, is_dc=False):
        """Return values aligned with the CSC pattern selected above."""
        candidates = []
        if is_dc:
            candidates.append(("_dc_jac_csc_indices", "_dc_jac_csc_data"))
        candidates.extend(
            (
                ("full_jac_csc_indices", "full_jac_csc_data"),
                ("standard_jac_csc_indices", "standard_jac_csc_data"),
            )
        )
        for index_name, data_name in candidates:
            indices = getattr(calc, index_name, None)
            data = getattr(calc, data_name, None)
            if indices is not None and data is not None and np.asarray(indices).size:
                return np.asarray(data, dtype=np.float64)
        return None

    def _cache_global_jacobian_pattern(self):
        """Precompute global hybrid Jacobian CSR pattern for repeated Newton iterations."""
        self._clear_global_jacobian_pattern()
        if self.electric_eq == 0 or self.electric_vars == 0:
            return

        rows_parts = []
        cols_parts = []
        raw_count = 0

        def add_part(name, rows, cols):
            nonlocal raw_count
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            if rows.size != cols.size:
                raise ValueError(f"Hybrid Jacobian pattern part {name!r} has mismatched row/column lengths")
            part_slice = slice(raw_count, raw_count + rows.size)
            setattr(self, f"global_jac_{name}_slice", part_slice)
            raw_count += rows.size
            if rows.size:
                rows_parts.append(rows)
                cols_parts.append(cols)

        if self.ac_calc is not None:
            ac_rows, ac_cols = self._sub_jacobian_pattern(self.ac_calc, is_dc=False)
            add_part("ac", ac_rows, ac_cols)
        if self.dc_calc is not None:
            dc_rows, dc_cols = self._sub_jacobian_pattern(self.dc_calc, is_dc=True)
            add_part("dc", dc_rows + self.ac_eq, dc_cols + self.ac_size)

        if self.N_dcac:
            add_part(
                "dcac_ac_p",
                self.dcac_ac_p_row[self.dcac_ac_p_eq_mask],
                self.dcac_ac_p_col[self.dcac_ac_p_eq_mask],
            )
            add_part(
                "dcac_ac_q",
                self.dcac_ac_q_row[self.dcac_ac_q_eq_mask],
                self.dcac_ac_q_col[self.dcac_ac_q_eq_mask],
            )
            add_part("dcac_dc_eq", self.dcac_dc_eq_rows, self.dcac_dc_eq_cols)
            add_part("dcac_loss", self.dcac_loss_rows, self.dcac_loss_cols)
            add_part("dcac_ctrl", self.dcac_ctrl_rows, self.dcac_ctrl_cols)

        self.global_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            self.global_jac_csr_indptr = np.zeros(self.electric_eq + 1, dtype=np.int32)
            self.global_jac_csc_indptr = np.zeros(self.electric_vars + 1, dtype=np.int32)
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        (
            self.global_jac_csr_indices,
            self.global_jac_csr_indptr,
            self.global_jac_raw_to_csr_pos,
        ) = build_compressed_pattern_from_raw_coords(raw_rows, raw_cols, self.electric_eq)
        (
            self.global_jac_csc_indices,
            self.global_jac_csc_indptr,
            self.global_jac_raw_to_csc_pos,
        ) = build_compressed_pattern_from_raw_coords(raw_cols, raw_rows, self.electric_vars)
        self.global_jac_csr_data = np.empty(self.global_jac_csr_indices.size, dtype=np.float64)
        self.global_jac_csc_data = np.empty(self.global_jac_csc_indices.size, dtype=np.float64)
        self.global_jac_csr_sum_plan = build_raw_sum_plan(self.global_jac_raw_to_csr_pos, self.global_jac_csr_data.size)
        self.global_jac_csc_sum_plan = build_raw_sum_plan(self.global_jac_raw_to_csc_pos, self.global_jac_csc_data.size)

    def _initial_dcac_x(self):
        if self.N_dcac == 0:
            return np.array([], dtype=np.float64)
        x = np.zeros(self.N_dcac * 3, dtype=np.float64)
        # Terminal powers are always positive into the converter, independent
        # of descriptive converter metadata.
        ac_p_control = (
            (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["PQ"])
            & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["NONE"])
        )
        dc_p_control = (
            (self.dcac_ac_control_code == DCAC_AC_CONTROL_CODE["NONE"])
            & (self.dcac_dc_control_code == DCAC_DC_CONTROL_CODE["P"])
        )
        ac_p = np.where(ac_p_control, self.dcac_p_ac_set, 0.0)
        dc_p = np.where(dc_p_control, self.dcac_p_dc_set, -ac_p)
        x[0::3] = dc_p
        x[1::3] = np.where(dc_p_control, -dc_p, ac_p)
        x[2::3] = self.dcac_q_ac_set
        return x

    def _cached_state_values(self, ac_x, dc_x):
        """Reuse AC sub-solver state cache after AC residual/Jacobian evaluation."""
        ac_theta = ac_V = None
        dc_V = None
        if self.ac_calc is not None:
            cache = getattr(self.ac_calc, "_cache", {})
            ac_theta = cache.get("theta")
            ac_V = cache.get("V")
            if ac_theta is None or ac_V is None:
                ac_theta, ac_V, _, _ = self.ac_calc._extract_state_vars(ac_x)
        if self.dc_calc is not None:
            dc_V = dc_x[:self.dc_calc.N]
        return ac_theta, ac_V, dc_V

    def _append_dcac_residuals(self, ac_f, dc_f, dcac_x, ac_theta, ac_V, dc_V, out=None):
        """Mutate AC/DC nodal residuals and return DC/AC converter residual rows."""
        dcac = dcac_x.reshape(self.N_dcac, 3)
        dc_p = dcac[:, 0]
        ac_p = dcac[:, 1]
        ac_q = dcac[:, 2]
        # DCAC terminal powers are positive from each grid into the converter,
        # so both terminal values enter the existing nodal residuals directly.
        # bincount 比 np.add.at 显著更快；ac_f / dc_f 是预分配的全局 F 切片，
        # 这里用 += 把贡献加进去。
        if np.any(self.dcac_ac_p_eq_mask):
            ac_f += np.bincount(
                self.dcac_ac_p_row[self.dcac_ac_p_eq_mask],
                weights=ac_p[self.dcac_ac_p_eq_mask],
                minlength=ac_f.size,
            )
        if np.any(self.dcac_ac_q_eq_mask):
            ac_f += np.bincount(
                self.dcac_ac_q_row[self.dcac_ac_q_eq_mask],
                weights=ac_q[self.dcac_ac_q_eq_mask],
                minlength=ac_f.size,
            )
        if self.dcac_dc_eq_mask.any():
            dc_eq_active = self.dcac_dc_eq[self.dcac_dc_eq_mask]
            dc_p_active = dc_p[self.dcac_dc_eq_mask]
            dc_f += np.bincount(dc_eq_active, weights=dc_p_active, minlength=dc_f.size)

        va = ac_V[self.dcac_ac_pos]
        vd = dc_V[self.dcac_dc_pos]
        va2 = va * va
        vd2 = vd * vd
        dcac_f = out if out is not None else np.empty(self.N_dcac * 3, dtype=np.float64)
        # r1+r2 converter loss equation in per-unit power/voltage variables.
        dcac_f[0::3] = (
            vd2 * va2 * (dc_p + ac_p)
            - self.dcac_r1 * dc_p * dc_p * va2
            - self.dcac_r2 * (ac_p * ac_p + ac_q * ac_q) * vd2
        )
        f_ctrl = dcac_f[1::3]
        f_ctrl[self.dcac_ctrl_ac_p_mask] = (
            ac_p[self.dcac_ctrl_ac_p_mask] - self.dcac_p_ac_set[self.dcac_ctrl_ac_p_mask]
        )
        f_ctrl[self.dcac_ctrl_dc_p_mask] = (
            dc_p[self.dcac_ctrl_dc_p_mask] - self.dcac_p_dc_set[self.dcac_ctrl_dc_p_mask]
        )
        f_ctrl[self.dcac_dcv_rep_mask] = vd[self.dcac_dcv_rep_mask] - self.dcac_dcv_avg_set[self.dcac_dcv_rep_mask]
        if np.any(self.dcac_dcv_share_mask):
            share_idx = np.flatnonzero(self.dcac_dcv_share_mask).astype(np.int32, copy=False)
            ref = self.dcac_dcv_ref_idx[share_idx]
            f_ctrl[share_idx] = dc_p[share_idx] - dc_p[ref]
        f_ctrl[self.dcac_acv_rep_mask] = va[self.dcac_acv_rep_mask] - self.dcac_acv_v_avg_set[self.dcac_acv_rep_mask]
        if np.any(self.dcac_acv_share_mask):
            share_idx = np.flatnonzero(self.dcac_acv_share_mask).astype(np.int32, copy=False)
            ref = self.dcac_acv_ref_idx[share_idx]
            f_ctrl[share_idx] = ac_p[share_idx] - ac_p[ref]
        dcac_f[1::3] = f_ctrl
        f_second = dcac_f[2::3]
        f_second[self.dcac_ctrl_q_mask] = ac_q[self.dcac_ctrl_q_mask] - self.dcac_q_ac_set[self.dcac_ctrl_q_mask]
        f_second[self.dcac_acv_rep_mask] = (
            ac_theta[self.dcac_ac_pos[self.dcac_acv_rep_mask]]
            - self.dcac_acv_theta_avg_set[self.dcac_acv_rep_mask]
        )
        if np.any(self.dcac_acv_share_mask):
            share_idx = np.flatnonzero(self.dcac_acv_share_mask).astype(np.int32, copy=False)
            ref = self.dcac_acv_ref_idx[share_idx]
            f_second[share_idx] = ac_q[share_idx] - ac_q[ref]
        dcac_f[2::3] = f_second
        return dcac_f

    def _fill_residual_work(self, ac_f, dc_f, dcac_x, ac_theta, ac_V, dc_V):
        """Fill the preallocated global residual vector and return it."""
        if self._electric_residual_work.size != self.electric_eq:
            self._electric_residual_work = np.empty(self.electric_eq, dtype=np.float64)
        F = self._electric_residual_work
        ac_view = None
        dc_view = None
        if self.ac_eq:
            ac_view = F[:self.ac_eq]
            ac_view[:] = ac_f
        if self.dc_eq:
            dc_view = F[self.ac_eq:self.ac_eq + self.dc_eq]
            dc_view[:] = dc_f
        if self.N_dcac:
            dcac_view = F[self.dcac_eq_start:self.electric_eq]
            self._append_dcac_residuals(ac_view, dc_view, dcac_x, ac_theta, ac_V, dc_V, out=dcac_view)
        return F

    def _get_electric_f(self, x: np.ndarray) -> np.ndarray:
        """Assemble AC/DC residuals and DCAC coupling equations."""
        # 纯 AC/纯 DC 文件没有跨域耦合时，直接复用子求解器残差，避免全局向量拆装。
        if self._single_ac_newton_block:
            return self.ac_calc.get_f(x)
        if self._single_dc_newton_block:
            return self.dc_calc.get_f(x)
        ac_x, dc_x, dcac_x = self._split_x(x)
        ac_f = None
        dc_f = None
        if self.ac_calc is not None:
            ac_f = self.ac_calc.get_f(ac_x)
        if self.dc_calc is not None:
            dc_f = self.dc_calc.get_f(dc_x)
        ac_theta = ac_V = dc_V = None
        if self.N_dcac:
            ac_theta, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        return self._fill_residual_work(ac_f, dc_f, dcac_x, ac_theta, ac_V, dc_V)

    def _get_electric_jacobi(self, x: np.ndarray) -> csr_matrix:
        """Build the global sparse Jacobian from sub-solver blocks plus converter couplings."""
        # 单块 Newton 的 Jacobian 与子系统完全一致，不需要再包一层 hybrid CSR。
        if self._single_ac_newton_block:
            jac = self.ac_calc.get_jacobi(x)
            self.last_jacobian_shape = jac.shape
            return jac
        if self._single_dc_newton_block:
            jac = self.dc_calc.get_jacobi(x)
            self.last_jacobian_shape = jac.shape
            return jac
        _residual, jac = self._build_electric_newton_system(
            x,
            return_jacobian=True,
            jacobian_format="csr",
        )
        return jac

    @staticmethod
    def _slice_len(part_slice):
        return int(part_slice.stop - part_slice.start)

    @staticmethod
    def _jacobian_values(jac_or_data):
        if jac_or_data is None:
            return None
        if getattr(jac_or_data, "format", None) is not None:
            return jac_or_data.data
        return np.asarray(jac_or_data, dtype=np.float64)

    def _fill_dcac_global_jacobian_data(self, raw, dcac_x, ac_V, dc_V):
        if not self.N_dcac:
            return
        dcac = dcac_x.reshape(self.N_dcac, 3)
        dc_p = dcac[:, 0]
        ac_p = dcac[:, 1]
        ac_q = dcac[:, 2]

        for part_slice in (
            self.global_jac_dcac_ac_p_slice,
            self.global_jac_dcac_ac_q_slice,
            self.global_jac_dcac_dc_eq_slice,
            self.global_jac_dcac_ctrl_slice,
        ):
            if self._slice_len(part_slice):
                if part_slice == self.global_jac_dcac_ctrl_slice:
                    raw[part_slice] = self.dcac_ctrl_data
                else:
                    raw[part_slice] = 1.0

        va = ac_V[self.dcac_ac_pos]
        vd = dc_V[self.dcac_dc_pos]
        va2 = va * va
        vd2 = vd * vd
        dc_p2 = dc_p * dc_p
        ac_i2_num = ac_p * ac_p + ac_q * ac_q
        d_va = 2.0 * va * vd2 * (dc_p + ac_p) - 2.0 * self.dcac_r1 * dc_p2 * va
        raw[self.global_jac_dcac_loss_slice] = np.concatenate(
            (
                vd2 * va2 - 2.0 * self.dcac_r1 * dc_p * va2,
                vd2 * va2 - 2.0 * self.dcac_r2 * ac_p * vd2,
                -2.0 * self.dcac_r2 * ac_q * vd2,
                d_va[self.dcac_ac_q_eq_mask],
                2.0 * vd * va2 * (dc_p + ac_p) - 2.0 * self.dcac_r2 * ac_i2_num * vd,
            )
        )

    def _assemble_jacobian_from_precomputed_pattern(
        self,
        ac_x,
        dc_x,
        dcac_x,
        ac_j=None,
        dc_j=None,
        ac_V=None,
        dc_V=None,
        *,
        matrix_format="csr",
    ):
        if self.global_jac_raw_data.size == 0:
            return None
        raw = self.global_jac_raw_data
        if ac_j is not None and self._slice_len(self.global_jac_ac_slice):
            raw[self.global_jac_ac_slice] = self._jacobian_values(ac_j)
        if dc_j is not None and self._slice_len(self.global_jac_dc_slice):
            raw[self.global_jac_dc_slice] = self._jacobian_values(dc_j)

        if self.N_dcac and (ac_V is None or dc_V is None):
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        self._fill_dcac_global_jacobian_data(raw, dcac_x, ac_V, dc_V)

        if matrix_format == "csc":
            apply_raw_sum_plan(self.global_jac_csc_data, raw, self.global_jac_csc_sum_plan)
            jac = csc_matrix(
                (self.global_jac_csc_data, self.global_jac_csc_indices, self.global_jac_csc_indptr),
                shape=(self.electric_eq, self.electric_vars),
                copy=False,
            )
        else:
            apply_raw_sum_plan(self.global_jac_csr_data, raw, self.global_jac_csr_sum_plan)
            jac = csr_matrix(
                (self.global_jac_csr_data, self.global_jac_csr_indices, self.global_jac_csr_indptr),
                shape=(self.electric_eq, self.electric_vars),
                copy=False,
            )
        self.last_jacobian_shape = jac.shape
        return jac

    def _assemble_jacobian(self, ac_x, dc_x, dcac_x, ac_j=None, dc_j=None, ac_V=None, dc_V=None, *, matrix_format="csr"):
        """Build the global sparse Jacobian from prepared sub-solver blocks."""
        precomputed = self._assemble_jacobian_from_precomputed_pattern(
            ac_x,
            dc_x,
            dcac_x,
            ac_j=ac_j,
            dc_j=dc_j,
            ac_V=ac_V,
            dc_V=dc_V,
            matrix_format=matrix_format,
        )
        if precomputed is not None:
            return precomputed

        row_parts = []
        col_parts = []
        data_parts = []
        if ac_j is not None:
            ac_coo = ac_j.tocoo()
            row_parts.append(ac_coo.row)
            col_parts.append(ac_coo.col)
            data_parts.append(ac_coo.data)
        target_shape = (self.electric_eq, self.electric_vars)
        if dc_j is not None:
            dc_coo = dc_j.tocoo()
            row_parts.append(dc_coo.row + self.ac_eq)
            col_parts.append(dc_coo.col + self.ac_size)
            data_parts.append(dc_coo.data)

        if self.N_dcac and (ac_V is None or dc_V is None):
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        if self.N_dcac:
            self._append_dcac_jacobian_terms(row_parts, col_parts, data_parts, dcac_x, ac_V, dc_V)

        if row_parts:
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            data = np.concatenate(data_parts)
            nonzero = data != 0.0
            if not np.all(nonzero):
                rows = rows[nonzero]
                cols = cols[nonzero]
                data = data[nonzero]
            jac = coo_matrix((data, (rows, cols)), shape=target_shape).tocsr()
        else:
            jac = coo_matrix(target_shape, dtype=np.float64).tocsr()
        if matrix_format == "csc":
            jac = jac.tocsc()
        self.last_jacobian_shape = jac.shape
        return jac

    def _build_electric_newton_system(
        self,
        x: np.ndarray,
        *,
        return_jacobian=True,
        jacobian_format="csc",
    ):
        """Build one global residual/Jacobian while reusing AC/DC equation caches."""
        if self._single_ac_newton_block:
            F, J = self.ac_calc._build_newton_system(
                x,
                return_jacobian=return_jacobian,
                jacobian_format=jacobian_format,
            )
            if J is not None:
                self.last_jacobian_shape = J.shape
            return F, J
        if self._single_dc_newton_block:
            F, J = self.dc_calc._build_newton_system(
                x,
                return_jacobian=return_jacobian,
                jacobian_format=jacobian_format,
            )
            if J is not None:
                self.last_jacobian_shape = J.shape
            return F, J
        ac_x, dc_x, dcac_x = self._split_x(x)
        ac_f = ac_j = None
        dc_f = dc_j = None
        if self.ac_calc is not None:
            ac_f, ac_j = self.ac_calc._build_newton_system(
                ac_x,
                return_jacobian=False,
                jacobian_format="csc",
            )
            if return_jacobian and ac_j is None and self._slice_len(self.global_jac_ac_slice):
                ac_j = self._sub_jacobian_cached_values(self.ac_calc, is_dc=False)
                if ac_j is None:
                    ac_j = self.ac_calc.get_jacobi(ac_x)
        if self.dc_calc is not None:
            dc_f, dc_j = self.dc_calc._build_newton_system(
                dc_x,
                return_jacobian=False,
                jacobian_format="csc",
            )
            if return_jacobian and dc_j is None and self._slice_len(self.global_jac_dc_slice):
                dc_j = self._sub_jacobian_cached_values(self.dc_calc, is_dc=True)
                if dc_j is None:
                    dc_j = self.dc_calc.get_jacobi(dc_x)

        ac_theta = ac_V = dc_V = None
        if self.N_dcac:
            ac_theta, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)

        F = self._fill_residual_work(ac_f, dc_f, dcac_x, ac_theta, ac_V, dc_V)
        if not return_jacobian:
            return F, None
        J = self._assemble_jacobian(
            ac_x,
            dc_x,
            dcac_x,
            ac_j,
            dc_j,
            ac_V,
            dc_V,
            matrix_format=jacobian_format,
        )
        return F, J

    def _build_newton_system(self, x: np.ndarray, *, return_jacobian=True, jacobian_format="csc"):
        """Assemble every energy-domain block into one sparse Newton system."""
        if not self.fluid_calcs:
            return self._build_electric_newton_system(
                x,
                return_jacobian=return_jacobian,
                jacobian_format=jacobian_format,
            )

        residual_parts = []
        jacobian_blocks = []
        if self.electric_eq:
            electric_f, electric_j = self._build_electric_newton_system(
                x[: self.electric_vars],
                return_jacobian=return_jacobian,
                jacobian_format="csc",
            )
            residual_parts.append(np.asarray(electric_f, dtype=np.float64).copy())
            if return_jacobian:
                jacobian_blocks.append(electric_j)

        for name, calc in self.fluid_calcs.items():
            state = x[self.fluid_state_slices[name]]
            residual, jacobian = calc._build_newton_system(
                state,
                return_jacobian=return_jacobian,
                jacobian_format="csc",
            )
            residual_parts.append(np.asarray(residual, dtype=np.float64))
            if return_jacobian:
                jacobian_blocks.append(jacobian)

        coupling_count = len(self.energy_coupling_plans)
        if coupling_count:
            residual_parts.append(np.zeros(coupling_count, dtype=np.float64))
            if return_jacobian:
                jacobian_blocks.append(
                    coo_matrix((coupling_count, coupling_count), dtype=np.float64)
                )

        residual = (
            np.concatenate(residual_parts)
            if residual_parts
            else np.empty(0, dtype=np.float64)
        )
        if not return_jacobian:
            self._apply_energy_coupling_terms(
                x,
                residual,
                None,
                jacobian_format=jacobian_format,
            )
            return residual, None
        jacobian = block_diag(jacobian_blocks, format=jacobian_format)
        jacobian = self._apply_energy_coupling_terms(
            x,
            residual,
            jacobian,
            jacobian_format=jacobian_format,
        )
        self.last_jacobian_shape = jacobian.shape
        return residual, jacobian

    def get_f(self, x: np.ndarray) -> np.ndarray:
        return self._build_newton_system(x, return_jacobian=False)[0]

    def get_jacobi(self, x: np.ndarray) -> csr_matrix:
        return self._build_newton_system(
            x,
            return_jacobian=True,
            jacobian_format="csr",
        )[1]

    def _append_dcac_jacobian_terms(self, row_parts, col_parts, data_parts, dcac_x, ac_V, dc_V):
        """Append DC/AC converter Jacobian entries to global COO buffers."""
        n = self.N_dcac
        dcac = dcac_x.reshape(n, 3)
        dc_p = dcac[:, 0]
        ac_p = dcac[:, 1]
        ac_q = dcac[:, 2]

        if np.any(self.dcac_ac_p_eq_mask):
            row_parts.append(self.dcac_ac_p_row[self.dcac_ac_p_eq_mask])
            col_parts.append(self.dcac_ac_p_col[self.dcac_ac_p_eq_mask])
            data_parts.append(self.dcac_ones[self.dcac_ac_p_eq_mask])
        if np.any(self.dcac_ac_q_eq_mask):
            row_parts.append(self.dcac_ac_q_row[self.dcac_ac_q_eq_mask])
            col_parts.append(self.dcac_ac_q_col[self.dcac_ac_q_eq_mask])
            data_parts.append(self.dcac_ones[self.dcac_ac_q_eq_mask])
        if self.dcac_dc_eq_rows.size:
            row_parts.append(self.dcac_dc_eq_rows)
            col_parts.append(self.dcac_dc_eq_cols)
            data_parts.append(self.dcac_dc_eq_ones)

        va = ac_V[self.dcac_ac_pos]
        vd = dc_V[self.dcac_dc_pos]
        va2 = va * va
        vd2 = vd * vd
        dc_p2 = dc_p * dc_p
        ac_i2_num = ac_p * ac_p + ac_q * ac_q
        d_va = 2.0 * va * vd2 * (dc_p + ac_p) - 2.0 * self.dcac_r1 * dc_p2 * va
        loss_data = np.concatenate(
            (
                vd2 * va2 - 2.0 * self.dcac_r1 * dc_p * va2,
                vd2 * va2 - 2.0 * self.dcac_r2 * ac_p * vd2,
                -2.0 * self.dcac_r2 * ac_q * vd2,
                d_va[self.dcac_ac_q_eq_mask],
                2.0 * vd * va2 * (dc_p + ac_p) - 2.0 * self.dcac_r2 * ac_i2_num * vd,
            )
        )
        row_parts.append(self.dcac_loss_rows)
        col_parts.append(self.dcac_loss_cols)
        data_parts.append(loss_data)

        if self.dcac_ctrl_rows.size:
            row_parts.append(self.dcac_ctrl_rows)
            col_parts.append(self.dcac_ctrl_cols)
            data_parts.append(self.dcac_ctrl_data)

    def _single_block(self):
        if self._single_ac_newton_block:
            return "ac", self.ac_calc, self.ac_eq, self.ac_size
        if self._single_dc_newton_block:
            return "dc", self.dc_calc, self.dc_eq, self.dc_size
        return None, None, 0, 0

    def _hybrid_summary(self):
        summary = {
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "normF": float(self.normF),
        }
        if self.failure_reason:
            summary["failure_reason"] = self.failure_reason
        return summary

    def _ac_array_summary(self, ac_result):
        if not ac_result:
            return None
        bus = ac_result.get("bus")
        if bus is None:
            return None
        return {
            "node_id": bus[:, AC_BUS_COLS["idx"]].astype(np.int64, copy=True),
            "voltage": bus[:, AC_BUS_COLS["voltage"]].copy(),
            "angle": bus[:, AC_BUS_COLS["angle"]].copy(),
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
            },
        }

    def _dc_array_summary(self, dc_result):
        if not dc_result:
            return None
        bus = dc_result.get("bus")
        if bus is None:
            return None
        return {
            "node_id": bus[:, DC_BUS_COLS["idx"]].astype(np.int64, copy=True),
            "voltage": bus[:, DC_BUS_COLS["voltage"]].copy(),
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
            },
        }

    def _set_array_result(self, ac_result, dc_result, dcac_result=None):
        self.result = {
            "ac": ac_result,
            "dc": dc_result,
            "dcac": (
                np.zeros((0, 5), dtype=np.float64)
                if dcac_result is None
                else dcac_result
            ),
            "summary": self._hybrid_summary(),
        }

    def _sync_single_subsolver_result(self, kind):
        ac_result = self.ac_calc.result if kind == "ac" else self._skipped_ac_result
        dc_result = self.dc_calc.result if kind == "dc" else self._skipped_dc_result
        if kind == "ac" and (
            self.result_mode == "full"
            or getattr(self.network.ac, "_lf_lightweight", False)
        ):
            self._write_ac_ppc_result_to_network()
        elif kind == "dc" and (
            self.result_mode == "full"
            or getattr(self.network.dc, "_lf_lightweight", False)
        ):
            self._write_dc_ppc_result_to_network()
        self._write_skipped_results_to_network()

        if self.result_mode == "full":
            self._set_array_result(ac_result, dc_result)
            self.lf_result = (
                None if getattr(self, "skip_lf_result", False) else self._build_lf_result()
            )
        elif self.result_mode == "array":
            self._set_array_result(ac_result, dc_result)
            self.lf_result = None
        elif self.result_mode == "summary":
            self.result = {
                "ac": self._ac_array_summary(ac_result),
                "dc": self._dc_array_summary(dc_result),
                "hybrid": self._hybrid_summary(),
            }
            self.lf_result = None
        else:
            self.result = {}
            self.lf_result = None

    def _write_skipped_results_to_network(self):
        if self.ac_calc is None:
            if self._skipped_ac_result is not None:
                self.network.ac.result = self._skipped_ac_result
            if not getattr(self.network.ac, "_lf_lightweight", False):
                _zero_skipped_object_side(self.network.ac, _AC_SKIPPED_OBJECT_ATTRS)
        if self.dc_calc is None:
            if self._skipped_dc_result is not None:
                self.network.dc.result = self._skipped_dc_result
            if not getattr(self.network.dc, "_lf_lightweight", False):
                _zero_skipped_object_side(self.network.dc, _DC_SKIPPED_OBJECT_ATTRS)

    def _finish_empty_system(self) -> int:
        self.converged = True
        self.iterations = 0
        self.normF = 0.0
        self.failure_reason = ""
        self._write_skipped_results_to_network()
        if self.result_mode == "none":
            self.result = {}
            self.lf_result = None
        elif self.result_mode == "summary":
            self.result = {
                "ac": self._ac_array_summary(self._skipped_ac_result),
                "dc": self._dc_array_summary(self._skipped_dc_result),
                "hybrid": self._hybrid_summary(),
            }
            self.lf_result = None
        else:
            self._set_array_result(self._skipped_ac_result, self._skipped_dc_result)
            self.lf_result = None if self.result_mode == "array" else self._build_lf_result()
        return 0

    @staticmethod
    def _call_subsolver_without_object_writeback(subcalc, callback):
        network_writeback = getattr(subcalc, "_network_writeback", None)
        subcalc._network_writeback = None
        try:
            return callback()
        finally:
            subcalc._network_writeback = network_writeback

    def _run_single_subsolver(self, kind, subcalc, eq_count, var_count):
        subcalc.verbose = self.verbose
        subcalc.result_mode = "array"
        subcalc.keep_node_objects = False
        subcalc.skip_lf_result = True
        rc = self._call_subsolver_without_object_writeback(
            subcalc,
            subcalc._run_newton_raphson,
        )
        self.x = subcalc.x
        self.converged = bool(subcalc.converged)
        self.iterations = int(subcalc.iterations)
        self.normF = float(subcalc.normF)
        self.last_jacobian_shape = (eq_count, var_count)
        self._sync_single_subsolver_result(kind)
        return rc

    def run(self, result_mode=None) -> int:
        electric_rc = self._run_electric(result_mode=result_mode)
        return self._finish_multi_energy_run(electric_rc)

    def _run_electric(self, result_mode=None) -> int:
        """Execute unified Newton iterations over the full hybrid state vector."""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
            self._sync_sub_result_modes()
        if self.x.size == 0:
            self.prepare()
        if (
            self.total_vars == 0
            and self.total_eq == 0
            and self.ac_calc is None
            and self.dc_calc is None
        ):
            return self._finish_empty_system()
        return self._run_newton_raphson()

    def _finish_multi_energy_run(self, electric_rc: int) -> int:
        global_converged = bool(self.converged)
        if self.electric_eq:
            electric_f = self.get_f(self.x)[: self.electric_eq]
            self.electric_converged = bool(
                np.linalg.norm(electric_f, np.inf) < self.tol
            )
        else:
            self.electric_converged = True
        self.fluid_rc = {}
        self.fluid_errors = {}
        for name, calc in self.fluid_calcs.items():
            rc = 0 if calc.converged else 1
            self.fluid_rc[name] = rc
            if rc != 0 or not calc.converged:
                self.fluid_errors[name] = str(
                    getattr(calc, "failure_reason", "") or "fluid load flow did not converge"
                )
        fluids_converged = all(
            rc == 0 and self.fluid_calcs[name].converged
            for name, rc in self.fluid_rc.items()
        )
        self.converged = bool(global_converged and self.electric_converged and fluids_converged)
        self.coupling_results = self._build_energy_coupling_results()
        self._attach_multi_energy_results()
        if self.fluid_errors:
            details = "; ".join(f"{name}: {reason}" for name, reason in self.fluid_errors.items())
            self.failure_reason = "; ".join(
                item for item in (self.failure_reason, details) if item
            )
        if electric_rc == 0 and self.converged:
            return 0
        return electric_rc if electric_rc != 0 else 1

    def _terminal_result_value(self, terminal) -> Tuple[Optional[float], str]:
        domain = terminal.domain
        device_type = terminal.device_type
        device_idx = int(terminal.device_idx)
        if domain in self.fluid_calcs:
            calc = self.fluid_calcs[domain]
            network = calc.network
            if device_type.endswith(("Source", "Storage")):
                for pos, device in enumerate(network.sources):
                    is_storage = device_type.endswith("Storage")
                    if (
                        bool(network.source_is_storage[pos]) == is_storage
                        and int(device.idx) == device_idx
                    ):
                        return float(calc.source_flow[pos]), "flow"
            if device_type.endswith("Load"):
                for pos, device in enumerate(network.loads):
                    if int(device.idx) == device_idx:
                        return float(network.load_flow_set[pos]), "flow"
            if device_type.endswith("Node"):
                for pos, device in enumerate(network.nodes):
                    if int(device.idx) == device_idx:
                        return float(calc.pressure[pos]), "pressure"
            return None, ""
        if domain not in {"ac", "dc"}:
            return None, ""
        ppc = self._ac_ppc if domain == "ac" else self._dc_ppc
        if not isinstance(ppc, dict):
            return None, ""
        if domain == "ac":
            table_key = "load" if device_type == "ACLoad" else "gen"
            columns = AC_LOAD_COLS if table_key == "load" else AC_GEN_COLS
        else:
            table_key = "load" if device_type == "DCLoad" else "gen"
            columns = DC_LOAD_COLS if table_key == "load" else DC_GEN_COLS
        table = np.asarray(ppc.get(table_key, ()), dtype=np.float64)
        if table.ndim != 2 or table.size == 0:
            return None, ""
        idx_col = columns.get("idx", 0)
        p_col = columns.get("p")
        if p_col is None:
            return None, ""
        rows = np.flatnonzero(table[:, idx_col].astype(np.int64, copy=False) == device_idx)
        if rows.size == 0:
            return None, ""
        return float(table[int(rows[0]), p_col]), "pu"

    def _build_energy_coupling_results(self) -> List[object]:
        results = []
        plans = {
            (plan.coupling.table_name, int(plan.coupling.idx)): plan
            for plan in self.energy_coupling_plans
        }
        for coupling in self.multi_energy.couplings:
            plan = plans.get((coupling.table_name, int(coupling.idx)))
            electric_value = None
            source_flow = None
            t_out = None
            t_return = None
            if plan is None:
                t1_value, t1_unit = self._terminal_result_value(coupling.t1)
                t2_value, t2_unit = self._terminal_result_value(coupling.t2)
            elif plan.electric_heat:
                electric_value = (
                    plan.electric_setpoint
                    if coupling.control_type == "P"
                    else float(self.x[plan.state_col])
                )
                t_out = (
                    float(self.x[plan.state_col])
                    if coupling.control_type == "P"
                    else plan.supply_temperature_set
                )
                source_flow, _columns, _derivatives = self._energy_endpoint_value_and_derivative(
                    plan.heat,
                    self.x,
                )
                t_return = float(self.x[plan.return_temperature_col])
                if plan.t1 is plan.electric:
                    t1_value, t2_value = electric_value, t_out
                    t1_unit, t2_unit = "pu", "degC"
                else:
                    t1_value, t2_value = t_out, electric_value
                    t1_unit, t2_unit = "degC", "pu"
            elif plan.hydrogen_electric:
                dependent_value = float(self.x[plan.state_col])
                controlled_value = float(plan.controlled_setpoint)
                if plan.t1 is plan.controlled:
                    t1_value, t2_value = controlled_value, dependent_value
                else:
                    t1_value, t2_value = dependent_value, controlled_value
                t1_unit = "pu" if plan.t1.domain in {"ac", "dc"} else "Nm3/h"
                t2_unit = "pu" if plan.t2.domain in {"ac", "dc"} else "Nm3/h"
            else:
                t1_value = float(self.x[plan.state_col])
                t1_unit = "pu" if plan.t1.domain in {"ac", "dc"} else "flow"
                t2_value, _columns, _derivatives = self._energy_endpoint_value_and_derivative(
                    plan.t2,
                    self.x,
                )
                t2_unit = "pu" if plan.t2.domain in {"ac", "dc"} else "flow"
            residual = None
            status = "inactive" if not coupling.active else "linked"
            if coupling.active and (t1_value is None or t2_value is None):
                status = "unavailable"
            elif coupling.active and coupling.is_electric_heat_control and plan is not None:
                residual = electric_heat_balance_residual(
                    coupling,
                    electric_value,
                    source_flow,
                    t_out,
                    t_return,
                    plan.heat_capacity,
                    plan.power_scale,
                )
                status = "balanced" if abs(residual) <= max(self.tol, 1e-9) else "mismatch"
            elif coupling.active and coupling.is_hydrogen_electric_control:
                electric_value = t1_value if coupling.t1.domain in {"ac", "dc"} else t2_value
                hydro_value = t1_value if coupling.t1.domain == "hydro" else t2_value
                residual = hydrogen_electric_balance_residual(
                    coupling,
                    electric_value,
                    hydro_value,
                    float(getattr(self.network, "p_base_kW", 1.0)),
                )
                status = "balanced" if abs(residual) <= max(self.tol, 1e-9) else "mismatch"
            elif coupling.active and coupling.supports_energy_balance:
                factor = float(coupling.energy_factor)
                t1_energy = (
                    abs(t1_value) * self.network.p_base_kW
                    if t1_unit == "pu"
                    else abs(t1_value) * factor
                )
                t2_energy = (
                    abs(t2_value) * self.network.p_base_kW
                    if t2_unit == "pu"
                    else abs(t2_value) * factor
                )
                residual = t2_energy - t1_energy * float(coupling.efficiency)
                status = "balanced" if abs(residual) <= max(self.tol, 1e-9) else "mismatch"
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
                    t1_value=t1_value,
                    t1_unit=t1_unit,
                    t2_value=t2_value,
                    t2_unit=t2_unit,
                    efficiency=coupling.efficiency,
                    energy_factor=coupling.energy_factor,
                    source_flow=source_flow,
                    t_out=t_out,
                    t_return=t_return,
                    residual=residual,
                    status=status,
                )
            )
        return results

    def _attach_multi_energy_results(self) -> None:
        fluid_results = {
            name: calc.lf_result for name, calc in self.fluid_calcs.items()
        }
        fluid_summary = {
            name: {
                "converged": bool(calc.converged),
                "iterations": int(calc.iterations),
                "residual": float(calc.normF),
                "nodes": len(calc.network.nodes),
                "edges": len(calc.network.edges),
            }
            for name, calc in self.fluid_calcs.items()
        }
        if isinstance(self.result, dict):
            self.result["fluids"] = fluid_summary
            self.result["couplings"] = self.coupling_results
        if self.result_mode == "full" and self.lf_result is None:
            self.lf_result = self._build_lf_result()
        if self.lf_result is not None:
            self.lf_result.rc = 0 if self.converged else -1
            self.lf_result.fluid = fluid_results
            self.lf_result.fluid_calcs = dict(self.fluid_calcs)
            self.lf_result.fluid_warnings = list(self.multi_energy.warnings)
            self.lf_result.fluid_errors = dict(self.fluid_errors)
            self.lf_result.couplings = list(self.coupling_results)

    def _run_newton_raphson(self) -> int:
        """Solve the complete hybrid Jacobian; delegate only uncoupled one-side files."""
        kind, subcalc, eq_count, var_count = self._single_block()
        if subcalc is not None:
            return self._run_single_subsolver(kind, subcalc, eq_count, var_count)

        self.converged = False
        self.iterations = 0
        self.failure_reason = ""
        x = self.x.copy()
        resolved_name = self._linear_solver_resolved
        solver_fn = self._linear_solver_fn
        reusable_factorizer = None

        # 若本实例已把 KLU/UMFPACK 加入黑名单, 直接走 scipy, 避免重复触发失败路径。
        if self._instance_solver_blacklist and resolved_name in self._instance_solver_blacklist:
            resolved_name = "scipy"
            solver_fn = _scipy_spsolve
            self._linear_solver_resolved = "scipy"
            self._linear_solver_fn = _scipy_spsolve

        def bounded_candidate(candidate):
            candidate = np.asarray(candidate, dtype=np.float64).copy()
            for name, calc in self.fluid_calcs.items():
                count = int(calc.network.free_node_pos.size)
                if count:
                    state_slice = self.fluid_state_slices[name]
                    start = state_slice.start
                    candidate[start : start + count] = np.maximum(
                        candidate[start : start + count],
                        calc._minimum_potential(),
                    )
            if self.energy_coupling_state_slice.stop > self.energy_coupling_state_slice.start:
                candidate[self.energy_coupling_state_slice] = np.maximum(
                    candidate[self.energy_coupling_state_slice],
                    0.0,
                )
            return candidate

        for it in range(self.max_iter):
            F, J = self._build_newton_system(x)
            self.iterations = it + 1
            self.normF = np.linalg.norm(F, np.inf)

            if it == 0 and not self._validate_initial_jacobian_columns(J):
                self.x = x
                if self.verbose:
                    print(self.failure_reason)
                return -1

            ac_f = F[:self.ac_eq]
            dc_f = F[self.ac_eq:self.ac_eq + self.dc_eq]
            ac_norm = np.linalg.norm(ac_f, np.inf) if ac_f.size else 0.0
            dc_norm = np.linalg.norm(dc_f, np.inf) if dc_f.size else 0.0
            if self.ac_calc is not None:
                self.ac_calc.iterations = self.iterations
                self.ac_calc.normF = ac_norm
            if self.dc_calc is not None:
                self.dc_calc.iterations = self.iterations
                self.dc_calc.normF = dc_norm
            fluid_norms = {}
            for name, calc in self.fluid_calcs.items():
                fluid_f = F[self.fluid_eq_slices[name]]
                fluid_norm = np.linalg.norm(fluid_f, np.inf) if fluid_f.size else 0.0
                fluid_norms[name] = fluid_norm
                calc.iterations = self.iterations
                calc.normF = fluid_norm

            if self.verbose:
                print(
                    f"Iter {it + 1}: |F| = {self.normF:.2e}, "
                    f"|F_ac| = {ac_norm:.2e}, "
                    f"|F_dc| = {dc_norm:.2e}, "
                    + ", ".join(
                        f"|F_{name}| = {value:.2e}"
                        for name, value in fluid_norms.items()
                    )
                )

            if self.normF < self.tol:
                if self.verbose:
                    print(f"收敛于第 {it + 1} 次迭代")
                self.converged = True
                self.x = x
                self._write_back()
                return 0

            try:
                if reusable_factorizer is None:
                    reusable_factorizer = _make_reusable_factorizer(J, resolved_name)
                factor = (
                    reusable_factorizer.factor(J)
                    if reusable_factorizer is not None
                    else _factor_jacobian(J, resolved_name, solver_fn)
                )
                delta = factor.solve(F)
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                # 可选稀疏求解器失败时, 只在本实例回退到 scipy, 避免污染模块缓存。
                if resolved_name not in {"scipy", "superlu", "default"}:
                    self._instance_solver_blacklist.add(resolved_name)
                    if self.verbose:
                        print(f"[hybrid_lf] 可选稀疏求解器 {resolved_name!r} 失败，回退到 scipy: {exc}")
                reusable_factorizer = None
                resolved_name = "scipy"
                solver_fn = _scipy_spsolve
                self._linear_solver_resolved = "scipy"
                self._linear_solver_fn = _scipy_spsolve
                delta = _scipy_spsolve(J, F)
            if not self.fluid_calcs:
                x -= delta
                continue
            # 方程定义为 F(x)=0，Newton 方向为 -J^{-1}F。流体势变量的
            # 非线性比电压方程更强，因此在同一个全局方向上做残差下降线搜索。
            accepted = False
            step = 1.0
            for _ in range(24):
                candidate = bounded_candidate(x - step * delta)
                candidate_f, _ = self._build_newton_system(
                    candidate,
                    return_jacobian=False,
                )
                candidate_norm = (
                    float(np.linalg.norm(candidate_f, np.inf))
                    if candidate_f.size
                    else 0.0
                )
                if np.isfinite(candidate_norm) and candidate_norm < self.normF:
                    x = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self.failure_reason = "global Newton line search could not reduce the residual"
                self.x = x
                self._write_back()
                return -1

        if self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未收敛")
        self.x = x
        self._write_back()
        return -1

    def _validate_initial_jacobian_columns(self, jacobian, tol=1e-14):
        """Reject states that have no effective equation at the initial point."""
        csc = jacobian if getattr(jacobian, "format", None) == "csc" else jacobian.tocsc()
        zero_columns = []
        for col in range(csc.shape[1]):
            start, stop = int(csc.indptr[col]), int(csc.indptr[col + 1])
            data = csc.data[start:stop]
            if data.size == 0 or np.max(np.abs(data)) <= tol:
                zero_columns.append(col)
        if not zero_columns:
            return True

        labels = [self._global_state_label(col) for col in zero_columns]
        self.failure_reason = (
            "初始雅可比矩阵存在零列，以下状态缺少独立控制方程: "
            + ", ".join(labels)
        )
        return False

    def _global_state_label(self, column):
        """Return a compact diagnostic label for a global Newton state column."""
        column = int(column)
        if column < self.electric_vars and column < self.ac_size:
            return f"AC_state[{column}]"
        dc_stop = self.ac_size + self.dc_size
        if column < self.electric_vars and column < dc_stop:
            local = column - self.ac_size
            if self.dc_calc is not None and local < self.dc_calc.N:
                node_ids = sorted(
                    int(node_id)
                    for node_id, solver_pos in self.dc_calc.alive_node_dict.items()
                    if int(solver_pos) == local
                )
                if len(node_ids) == 1:
                    return f"V_DC(node {node_ids[0]})"
                if node_ids:
                    return "V_DC(nodes " + ",".join(str(node_id) for node_id in node_ids) + ")"
            return f"DC_state[{local}]"
        if column < self.electric_vars:
            return f"DCAC_state[{column - self.dcac_start}]"
        for name, state_slice in self.fluid_state_slices.items():
            if state_slice.start <= column < state_slice.stop:
                calc = self.fluid_calcs[name]
                local = column - state_slice.start
                if local < calc.n_potential:
                    return f"{name}_pressure_potential[{local}]"
                if local < calc.base_temperature:
                    return f"{name}_regulated_flow[{local - calc.n_potential}]"
                if calc.network.thermal and local < calc.base_enthalpy:
                    return f"{name}_temperature[{local - calc.base_temperature}]"
                if calc.network.steam:
                    return f"{name}_enthalpy[{local - calc.base_enthalpy}]"
                return f"{name}_state[{local}]"
        for plan in self.energy_coupling_plans:
            if int(plan.state_col) == column:
                suffix = (
                    plan.adjustable_kind.upper()
                    if plan.electric_heat
                    else "DEPENDENT"
                    if plan.hydrogen_electric
                    else "T1"
                )
                return f"Coupling:{plan.coupling.table_name}:{plan.coupling.name}:{suffix}"
        return f"global_state[{column}]"

    def _write_summary_result(self):
        x = self.x
        ac_x, dc_x, dcac_x = self._split_x(x)
        ac_result = None
        dc_result = None
        if self.ac_calc is not None:
            self._set_ac_dcac_power_injections(dcac_x)
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc.iterations = self.iterations
            self.ac_calc._write_summary_result()
            ac_result = self.ac_calc.result
        if self.dc_calc is not None:
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.iterations = self.iterations
            self.dc_calc._write_summary_result()
            dc_result = self.dc_calc.result
        self.result = {
            "ac": ac_result,
            "dc": dc_result,
            "hybrid": self._hybrid_summary(),
        }
        self.lf_result = None

    def _write_dc_ppc_result_to_network(self) -> None:
        result = self.dc_calc.result if self.dc_calc is not None else None
        if not result:
            return
        if getattr(self.network.dc, "_lf_lightweight", False):
            self.network.dc.result = result
            return
        self.dc_calc._write_ppc_result_to_network()

    def _write_ac_ppc_result_to_network(self) -> None:
        """Delegate AC array-result writeback to ACPowerFlowCalc."""
        if self.ac_calc is None or not self.ac_calc.result:
            return
        self.ac_calc.write_array_result_to_network(self.network.ac)

    def _write_back_ppc(self):
        """Compatibility helper matching AC/DC naming.

        Hybrid LF delegates PPC write-back to the AC/DC subcalculators after splitting
        the global state vector. This helper prepares that state and triggers the PPC-only
        write-back path on each subsolver without building the final HybridLFResult.
        """
        x = self.x
        ac_x, dc_x, dcac_x = self._split_x(x)
        if self.ac_calc is not None:
            self._set_ac_dcac_power_injections(dcac_x)
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc.iterations = self.iterations
            self.ac_calc._write_back_ppc()
        if self.dc_calc is not None:
            self._set_dc_converter_power_injections(dcac_x)
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.iterations = self.iterations
            self.dc_calc._write_back_ppc()

    def _write_back(self):
        """Write final global state back into AC, DC and converter model objects."""
        x = self.x
        ac_x, dc_x, dcac_x = self._split_x(x)
        ac_result = self._skipped_ac_result
        dc_result = self._skipped_dc_result
        if self.ac_calc is not None:
            self._set_ac_dcac_power_injections(dcac_x)
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc.iterations = self.iterations
            self.ac_calc.result_mode = "array"
            self.ac_calc.keep_node_objects = False
            self.ac_calc.skip_lf_result = True
            self._call_subsolver_without_object_writeback(
                self.ac_calc,
                self.ac_calc._write_back,
            )
            ac_result = self.ac_calc.result
        if self.dc_calc is not None:
            self._set_dc_converter_power_injections(dcac_x)
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.iterations = self.iterations
            self.dc_calc.result_mode = "array"
            self.dc_calc.keep_node_objects = False
            self.dc_calc.skip_lf_result = True
            self._call_subsolver_without_object_writeback(
                self.dc_calc,
                self.dc_calc._write_back,
            )
            dc_result = self.dc_calc.result

        global_f = self.get_f(x) if self.fluid_calcs else np.empty(0, dtype=np.float64)
        for name, calc in self.fluid_calcs.items():
            state = x[self.fluid_state_slices[name]]
            fluid_f = global_f[self.fluid_eq_slices[name]]
            fluid_norm = np.linalg.norm(fluid_f, np.inf) if fluid_f.size else 0.0
            calc.result_mode = self.result_mode
            calc.verbose = self.verbose
            calc.commit_state(
                state,
                converged=bool(self.converged and fluid_norm < self.tol),
                iterations=self.iterations,
                normF=fluid_norm,
            )

        self._apply_final_coupling_endpoint_controls(ac_result, dc_result)

        self._write_skipped_results_to_network()

        dcac_result = np.zeros((self.N_dcac, 5), dtype=np.float64)
        ac_V = dc_V = None
        if self.N_dcac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        if self.N_dcac:
            dcac = dcac_x.reshape(self.N_dcac, 3)
            dcac_result[:, 0] = dcac[:, 0]
            dcac_result[:, 1] = dcac[:, 1]
            dcac_result[:, 2] = dcac[:, 2]
            dc_v = dc_V[self.dcac_dc_pos]
            ac_v = ac_V[self.dcac_ac_pos]
            dcac_result[:, 3] = np.divide(
                dcac[:, 0],
                dc_v,
                out=np.zeros(self.N_dcac, dtype=np.float64),
                where=np.abs(dc_v) > self.params.min_voltage,
            )
            dcac_result[:, 4] = np.divide(
                np.hypot(dcac[:, 1], dcac[:, 2]),
                ac_v,
                out=np.zeros(self.N_dcac, dtype=np.float64),
                where=np.abs(ac_v) > self.params.min_voltage,
            )
        if self.result_mode == "none":
            self.result = {}
            self.lf_result = None
            return
        if self.result_mode == "summary":
            self.result = {
                "ac": self._ac_array_summary(ac_result),
                "dc": self._dc_array_summary(dc_result),
                "hybrid": self._hybrid_summary(),
            }
            self.lf_result = None
            return
        if self.result_mode == "array":
            self._set_array_result(ac_result, dc_result, dcac_result)
            self.lf_result = None
            return

        self._set_array_result(ac_result, dc_result, dcac_result)
        skip_lf_result = getattr(self, "skip_lf_result", False)
        if self.ac_calc is not None:
            self._write_ac_ppc_result_to_network()
        if self.dc_calc is not None:
            self._write_dc_ppc_result_to_network()
        if self.N_dcac:
            dcac = dcac_x.reshape(self.N_dcac, 3)
            dc_p = dcac[:, 0]
            ac_p = dcac[:, 1]
            ac_q = dcac[:, 2]
            dc_v = dc_V[self.dcac_dc_pos]
            ac_v = ac_V[self.dcac_ac_pos]
            dc_i = np.divide(dc_p, dc_v, out=np.zeros_like(dc_p), where=np.abs(dc_v) > self.params.min_voltage)
            ac_i = np.divide(
                np.hypot(ac_p, ac_q),
                ac_v,
                out=np.zeros_like(ac_p),
                where=np.abs(ac_v) > self.params.min_voltage,
            )
            if self._converter_ppc_mode and self.dcac_row_pos.size:
                dcac_table = self.network.ppc["dcac"]
                dcac_table[self.dcac_row_pos, DCAC_COLS["dc_p"]] = dc_p
                dcac_table[self.dcac_row_pos, DCAC_COLS["ac_p"]] = ac_p
                dcac_table[self.dcac_row_pos, DCAC_COLS["ac_q"]] = ac_q
                dcac_table[self.dcac_row_pos, DCAC_COLS["dc_i"]] = dc_i
                dcac_table[self.dcac_row_pos, DCAC_COLS["ac_i"]] = ac_i
            if not self._converter_ppc_mode:
                for conv, p_dc, p_ac, q_ac, i_dc, i_ac in zip(self.dcac_devices, dc_p, ac_p, ac_q, dc_i, ac_i):
                    conv.dc_p = float(p_dc)
                    conv.ac_p = float(p_ac)
                    conv.ac_q = float(q_ac)
                    conv.dc_i = float(i_dc)
                    conv.ac_i = float(i_ac)
                for conv, dcv, acv in zip(self.dcac_devices, dc_v, ac_v):
                    if getattr(conv, "dc_node_obj", None) is not None:
                        conv.dc_node_obj.voltage = float(dcv)
                    if getattr(conv, "ac_node_obj", None) is not None:
                        conv.ac_node_obj.voltage = float(acv)
        if skip_lf_result:
            self.lf_result = None
        else:
            self.lf_result = self._build_lf_result(ac_V, dc_V)

    def _set_ac_dcac_power_injections(self, dcac_x) -> None:
        """Expose DCAC AC-terminal powers to AC writeback."""
        if self.ac_calc is None:
            return
        if self.N_dcac == 0:
            self.ac_calc._external_ac_p_injection = None
            self.ac_calc._external_ac_q_injection = None
            return

        p = np.zeros(self.ac_calc.N, dtype=np.float64)
        q = np.zeros(self.ac_calc.N, dtype=np.float64)
        dcac = dcac_x.reshape(self.N_dcac, 3)
        p += np.bincount(self.dcac_ac_pos, weights=dcac[:, 1], minlength=self.ac_calc.N)
        q += np.bincount(self.dcac_ac_pos, weights=dcac[:, 2], minlength=self.ac_calc.N)

        self.ac_calc._external_ac_p_injection = p
        self.ac_calc._external_ac_q_injection = q

    def _set_dc_converter_power_injections(self, dcac_x) -> None:
        """Expose final DC-side converter injections to DC result writeback."""
        if self.dc_calc is None:
            return
        if self.N_dcac == 0:
            self.dc_calc._external_dc_p_injection = None
            return

        dcac = dcac_x.reshape(self.N_dcac, 3)
        self.dc_calc._external_dc_p_injection = np.bincount(
            self.dcac_dc_pos,
            weights=dcac[:, 0],
            minlength=self.dc_calc.N,
        )

    @staticmethod
    def _ppc_converter_key(names, row_pos, idx) -> str:
        if names is not None and row_pos < len(names):
            name = str(names[row_pos])
            if name:
                return name
        return str(int(idx))

    def _build_ppc_converter_lf_result(self, result: HybridLFResult, ac_V=None, dc_V=None) -> None:
        """Build the Hybrid-owned DCAC full-result dictionary from PPC tables."""
        ppc = getattr(self.network, "ppc", {}) or {}
        if self.N_dcac and (ac_V is None or dc_V is None):
            ac_x, dc_x, _dcac_x = self._split_x(self.x)
            _theta, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)

        dcac_table = ppc.get("dcac")
        if dcac_table is not None and dcac_table.size:
            dc_v_by_row = np.zeros(dcac_table.shape[0], dtype=np.float64)
            ac_v_by_row = np.zeros(dcac_table.shape[0], dtype=np.float64)
            if self.dcac_row_pos.size and dc_V is not None and ac_V is not None:
                dc_v_by_row[self.dcac_row_pos] = dc_V[self.dcac_dc_pos]
                ac_v_by_row[self.dcac_row_pos] = ac_V[self.dcac_ac_pos]
            names = ppc.get("dcac_name")
            for row_pos, row in enumerate(dcac_table):
                if int(row[DCAC_COLS["run_stat"]]) != 1:
                    continue
                key = self._ppc_converter_key(names, row_pos, row[DCAC_COLS["idx"]])
                result.dcac.dcac_converters[key] = SimpleNamespace(
                    i_p=float(row[DCAC_COLS["dc_p"]]),
                    i_c=float(row[DCAC_COLS["dc_i"]]),
                    i_v=float(dc_v_by_row[row_pos]),
                    j_p=float(row[DCAC_COLS["ac_p"]]),
                    j_q=float(row[DCAC_COLS["ac_q"]]),
                    j_c=float(row[DCAC_COLS["ac_i"]]),
                    j_v=float(ac_v_by_row[row_pos]),
                )

    def _build_lf_result(self, ac_V=None, dc_V=None) -> HybridLFResult:
        ac_result = self.ac_calc._build_lf_result_from_ppc() if self.ac_calc is not None else None
        dc_result = self.dc_calc._build_lf_result_from_ppc() if self.dc_calc is not None else None
        result = HybridLFResult(
            arrays=dict(self.result),
            network=self.network,
            ac_network=self.network.ac,
            dc_network=self.network.dc,
            calc=self,
            ac_calc=self.ac_calc,
            dc_calc=self.dc_calc,
            rc=0 if self.converged else -1,
            ac=ac_result,
            dc=dc_result,
        )
        if self._converter_ppc_mode:
            self._build_ppc_converter_lf_result(result, ac_V, dc_V)
            return result

        for conv in getattr(self.network, "dcac_converters", []):
            if getattr(conv, "run_stat", 1) != 1:
                continue
            dc_v = float(getattr(conv.dc_node_obj, "voltage", 0.0) or 0.0)
            ac_v = float(getattr(conv.ac_node_obj, "voltage", 0.0) or 0.0)
            result.dcac.dcac_converters[_lf_device_key(conv)] = SimpleNamespace(
                i_p=float(getattr(conv, "dc_p", 0.0) or 0.0),
                i_c=float(getattr(conv, "dc_i", 0.0) or 0.0),
                i_v=dc_v,
                j_p=float(getattr(conv, "ac_p", 0.0) or 0.0),
                j_q=float(getattr(conv, "ac_q", 0.0) or 0.0),
                j_c=float(getattr(conv, "ac_i", 0.0) or 0.0),
                j_v=ac_v,
            )
        return result

    def _build_lf_result_from_ppc(self) -> HybridLFResult:
        """Compatibility helper matching AC/DC naming.

        Hybrid LF already stores PPC-backed AC/DC sub-results before building the
        final HybridLFResult, so this delegates to `_build_lf_result()`.
        """
        return self._build_lf_result()


def print_hybrid_result(calc: HybridPowerFlowCalc, rc: int) -> None:
    result = calc.lf_result
    if result is None:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, 返回码: {rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
        if calc.failure_reason:
            print(f"失败原因: {calc.failure_reason}")
        return
    print("\n=== 多能系统联合潮流计算结果 ===")
    print(
        f"节点总数: {result.total_energy_nodes} "
        f"(AC={len(result.ac_network.nodes)}, DC={len(result.dc_network.nodes)}, "
        f"流体={result.total_fluid_nodes})"
    )

    print("\n1. AC 节点电压:")
    for node in result.ac_network.nodes:
        print(f"   AC 节点 {node.idx}: V={node.voltage:.6f} pu, angle={node.angle:.6f} rad")

    print("\n2. DC 节点电压:")
    for node in result.dc_network.nodes:
        print(f"   DC 节点 {node.idx}: V={node.voltage:.6f} pu")

    print("\n3. AC 发电机:")
    for gen in result.ac_network.generators:
        print(f"   AC 发电机 {gen.idx} 节点 {gen.node}: P={gen.p:.6f} pu, Q={gen.q:.6f} pu")

    print("\n4. DC 发电机:")
    for gen in result.dc_network.generators:
        print(f"   DC 发电机 {gen.idx} 节点 {gen.node}: P={gen.p:.6f} pu, I={gen.current:.6f} pu")

    print("\n5. DC/DC 变流器:")
    for conv in result.dc_network.dcdc_converters:
        print(
            f"   DCDC {conv.idx} {conv.i_node}->{conv.j_node} "
            f"控制:i={conv.i_control_type},j={conv.j_control_type}: "
            f"Pi={conv.i_p:.6f} pu, Pj={conv.j_p:.6f} pu, loss={conv.i_p + conv.j_p:.6f} pu"
        )

    print("\n6. DC/AC 逆变器:")
    for conv in result.network.dcac_converters:
        print(
            f"   DCAC {conv.idx} {conv.dc_node}->AC{conv.ac_node} "
            f"控制:AC={conv.ac_control_type},DC={conv.dc_control_type}: "
            f"Pdc={conv.dc_p:.6f} pu, Pac={conv.ac_p:.6f} pu, Qac={conv.ac_q:.6f} pu, "
            f"loss={conv.dc_p + conv.ac_p:.6f} pu"
        )

    print("\n7. AC/AC 柔性互联:")
    for conv in result.ac_network.acac_converters:
        print(
            f"   ACAC {conv.idx} AC{conv.i_node}->AC{conv.j_node} "
            f"控制:i={conv.i_control_type},j={conv.j_control_type}: "
            f"Pi={conv.i_p:.6f} pu, Qi={conv.i_q:.6f} pu, "
            f"Pj={conv.j_p:.6f} pu, Qj={conv.j_q:.6f} pu, "
            f"loss={conv.i_p + conv.j_p:.6f} pu"
        )

    if result.ac is not None:
        print("\n7.1 AC 零阻抗支路:")
        for name, tie in result.ac.zero_branches.items():
            print(
                f"   {name}: P={tie.i_p:.6f} pu, Q={tie.i_q:.6f} pu, "
                f"I={tie.i_c:.6f} pu, V={tie.i_v:.6f} pu"
            )

        print("\n7.2 AC 刀闸:")
        for name, tie in result.ac.breakers.items():
            print(
                f"   {name}: P={tie.i_p:.6f} pu, Q={tie.i_q:.6f} pu, "
                f"I={tie.i_c:.6f} pu, V={tie.i_v:.6f} pu"
            )

    if result.dc is not None:
        print("\n7.3 DC 零阻抗支路:")
        for name, tie in result.dc.zero_branches.items():
            print(
                f"   {name}: P={tie.i_p:.6f} pu, I={tie.i_c:.6f} pu, "
                f"V={tie.i_v:.6f} pu"
            )

        print("\n7.4 DC 刀闸:")
        for name, tie in result.dc.breakers.items():
            print(
                f"   {name}: P={tie.i_p:.6f} pu, I={tie.i_c:.6f} pu, "
                f"V={tie.i_v:.6f} pu"
            )

    print("\n8. 收敛信息:")
    if result.ac_calc is not None:
        print(f"   AC: {'已收敛' if result.ac_calc.converged else '未收敛'}, iter={result.ac_calc.iterations}, normF={result.ac_calc.normF:.3e}")
    else:
        print("   AC: 文件中无 AC 子网")
    if result.dc_calc is not None:
        print(f"   DC: {'已收敛' if result.dc_calc.converged else '未收敛'}, iter={result.dc_calc.iterations}, normF={result.dc_calc.normF:.3e}")
    else:
        print("   DC: 文件中无 DC 子网")
    print(
        f"   Hybrid: {'已收敛' if result.converged else '未收敛'}, "
        f"返回码: {rc}, iter={calc.iterations}, normF={calc.normF:.3e}"
    )

    ac_gen = sum(gen.p for gen in result.ac_network.generators)
    ac_load = sum(load.p for load in result.ac_network.loads)
    dc_gen = sum(gen.p for gen in result.dc_network.generators)
    dc_load = sum(load.p for load in result.dc_network.loads)
    print("\n9. 功率汇总:")
    print(f"   AC: gen={ac_gen:.6f} pu, load={ac_load:.6f} pu, diff={ac_gen - ac_load:.6f} pu")
    print(f"   DC: gen={dc_gen:.6f} pu, load={dc_load:.6f} pu, diff={dc_gen - dc_load:.6f} pu")

    if result.fluid_calcs:
        print("\n10. 流体网络:")
        for name, fluid_calc in result.fluid_calcs.items():
            print(
                f"   {name}: {'已收敛' if fluid_calc.converged else '未收敛'}, "
                f"nodes={len(fluid_calc.network.nodes)}, edges={len(fluid_calc.network.edges)}, "
                f"iter={fluid_calc.iterations}, normF={fluid_calc.normF:.3e}"
            )
    if result.couplings:
        print("\n11. 多能耦合设备:")
        for item in result.couplings:
            print(
                f"   {item.table_name}:{item.name}: status={item.status}, "
                f"t1={item.t1_value} {item.t1_unit}, t2={item.t2_value} {item.t2_unit}, "
                f"residual={item.residual}"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid electric/heat/gas/hydrogen/steam power flow")
    parser.add_argument("file", nargs="?", default=str(DEFAULT_HYBRID_EFILE), help="Hybrid E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument(
        "--linear-solver",
        default=None,
        help="Sparse linear solver. Default: umfpack for hybrid cases with DC network, pyklu for AC-only cases.",
    )
    parser.add_argument("--result-mode", choices=("full", "array", "summary", "none"), default="full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    network = _read_lf_network_from_file(args.file)
    calc = HybridPowerFlowCalc(
        network,
        parameter_file=args.para,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        linear_solver=args.linear_solver,
        result_mode=args.result_mode,
        verbose=not args.quiet,
    )
    rc = calc.run()
    if not args.quiet and calc.result_mode == "full":
        print_hybrid_result(calc, rc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
        for name, fluid_calc in calc.fluid_calcs.items():
            print(
                f"{name}: {'已收敛' if fluid_calc.converged else '未收敛'}, "
                f"iter={fluid_calc.iterations}, normF={fluid_calc.normF:.3e}"
            )
        if calc.failure_reason:
            print(f"失败原因: {calc.failure_reason}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
