from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


_EMPTY_INT = np.array([], dtype=np.int32)
_EMPTY_BOOL = np.array([], dtype=bool)


@dataclass
class TerminalDeviceTopologyArrays:
    i_node_pos: np.ndarray
    j_node_pos: np.ndarray
    i_bus_pos: np.ndarray
    j_bus_pos: np.ndarray
    i_island_pos: np.ndarray
    j_island_pos: np.ndarray
    island_pos: np.ndarray
    alive_mask: np.ndarray


@dataclass
class SingleDeviceTopologyArrays:
    node_pos: np.ndarray
    bus_pos: np.ndarray
    island_pos: np.ndarray
    alive_mask: np.ndarray


@dataclass
class GridTopologyArrays:
    node_ids: np.ndarray
    node_run_mask: np.ndarray
    node_to_bus_pos: np.ndarray
    node_to_island_pos: np.ndarray
    bus_ids: np.ndarray
    bus_node_offsets: np.ndarray
    bus_node_indices: np.ndarray
    bus_to_island_pos: np.ndarray
    island_ids: np.ndarray
    island_bus_offsets: np.ndarray
    island_bus_indices: np.ndarray
    island_alive_mask: np.ndarray
    bus_alive_mask: np.ndarray
    node_alive_mask: np.ndarray
    island_reference_bus_pos: np.ndarray
    devices: Dict[str, object] = field(default_factory=dict)


def _empty_terminal_device(count: int = 0) -> TerminalDeviceTopologyArrays:
    values = np.full(int(count), -1, dtype=np.int32)
    return TerminalDeviceTopologyArrays(
        values.copy(),
        values.copy(),
        values.copy(),
        values.copy(),
        values.copy(),
        values.copy(),
        values.copy(),
        np.zeros(int(count), dtype=bool),
    )


def _empty_single_device(count: int = 0) -> SingleDeviceTopologyArrays:
    values = np.full(int(count), -1, dtype=np.int32)
    return SingleDeviceTopologyArrays(values.copy(), values.copy(), values.copy(), np.zeros(int(count), dtype=bool))


def _pos_parent(parent, item: int) -> int:
    root = int(item)
    if root < 0 or root >= len(parent) or parent[root] < 0:
        return -1
    while parent[root] != root:
        root = parent[root]
    item = int(item)
    while parent[item] != item:
        item, parent[item] = parent[item], root
    return root


def _union_pos_parent(parent, left: int, right: int) -> None:
    root_l = _pos_parent(parent, int(left))
    root_r = _pos_parent(parent, int(right))
    if root_l >= 0 and root_r >= 0 and root_l != root_r:
        parent[root_r] = root_l


def _make_node_pos_lookup(node_ids: np.ndarray):
    if node_ids.size == 0:
        return np.array([], dtype=np.int32)
    min_id = int(np.min(node_ids))
    max_id = int(np.max(node_ids))
    if min_id >= 0 and max_id <= max(1024, int(node_ids.size) * 4):
        lookup = np.full(max_id + 1, -1, dtype=np.int32)
        lookup[node_ids.astype(np.intp, copy=False)] = np.arange(node_ids.size, dtype=np.int32)
        return lookup
    return {int(node_id): int(pos) for pos, node_id in enumerate(node_ids)}


def _map_node_positions(values, lookup) -> np.ndarray:
    node_ids = np.asarray(values, dtype=np.int64)
    out = np.full(node_ids.shape, -1, dtype=np.int32)
    if node_ids.size == 0:
        return out
    if isinstance(lookup, np.ndarray):
        if lookup.size == 0:
            return out
        mask = (node_ids >= 0) & (node_ids < lookup.size)
        if np.any(mask):
            out[mask] = lookup[node_ids[mask].astype(np.intp, copy=False)]
        return out
    flat = out.reshape(-1)
    for pos, node_id in enumerate(node_ids.reshape(-1)):
        flat[pos] = lookup.get(int(node_id), -1)
    return out


def _table_count(table) -> int:
    return 0 if table is None else int(np.asarray(table).shape[0])


def _active_terminal_positions(
    table,
    i_col: int,
    j_col: int,
    run_col: int,
    node_lookup,
    running_node_mask: np.ndarray,
    status_col: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _EMPTY_INT, _EMPTY_INT, _EMPTY_BOOL
    i_pos = _map_node_positions(rows[:, i_col], node_lookup)
    j_pos = _map_node_positions(rows[:, j_col], node_lookup)
    active = (
        (rows[:, run_col].astype(np.int64, copy=False) == 1)
        & (i_pos >= 0)
        & (j_pos >= 0)
        & (i_pos != j_pos)
    )
    if status_col is not None:
        active &= rows[:, status_col].astype(np.int64, copy=False) == 1
    if running_node_mask.size:
        valid_i = i_pos >= 0
        valid_j = j_pos >= 0
        active[valid_i] &= running_node_mask[i_pos[valid_i]]
        active[valid_j] &= running_node_mask[j_pos[valid_j]]
    return i_pos, j_pos, active


def _union_terminal_table(
    parent,
    table,
    i_col: int,
    j_col: int,
    run_col: int,
    node_lookup,
    running_node_mask: np.ndarray,
    status_col: Optional[int] = None,
) -> None:
    i_pos, j_pos, active = _active_terminal_positions(
        table,
        i_col,
        j_col,
        run_col,
        node_lookup,
        running_node_mask,
        status_col=status_col,
    )
    if active.size == 0:
        return
    for left, right in zip(i_pos[active], j_pos[active]):
        _union_pos_parent(parent, int(left), int(right))


def _flatten_position_groups(groups: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    indices = []
    for group in groups:
        indices.extend(int(item) for item in group)
        offsets.append(len(indices))
    return np.asarray(offsets, dtype=np.int32), np.asarray(indices, dtype=np.int32)


def _make_compact_bus(bus_cls, idx: int, grouped_nodes):
    bus = bus_cls.__new__(bus_cls)
    bus.idx = int(idx)
    bus.nodes = grouped_nodes
    ref = grouped_nodes[0] if grouped_nodes else None
    bus.name = getattr(ref, "name", f"bus_{idx}")
    bus.vbase = getattr(ref, "vbase", 0.0)
    bus.voltage = getattr(ref, "voltage", 1.0)
    bus.run_stat = 1
    bus.isl = None
    bus.isl_obj = None
    bus.is_alive = False
    bus.generators = ()
    bus.loads = ()
    bus.branches = ()
    bus.switches = ()
    bus.breakers = ()
    bus.zero_branches = ()
    if bus_cls.__name__ == "ACBus":
        bus.angle = getattr(ref, "angle", 0.0)
        bus.transformers = ()
        bus.shunt_compensators = ()
        bus.v_gens = ()
    else:
        bus.v_set = 1.0
        bus.v_gens = ()
        bus.v_dcdcs = ()
        bus.is_slack = False
        bus.dcdc_converters = ()
    return bus


def _component_position_groups(
    node_ids: np.ndarray,
    node_run_mask: np.ndarray,
    edge_specs: Sequence[Tuple[np.ndarray, int, int, int, Optional[int]]],
    node_lookup,
) -> Sequence[Sequence[int]]:
    n_nodes = int(node_ids.size)
    running_positions = np.flatnonzero(node_run_mask)
    if running_positions.size == 0:
        return []

    left_chunks = []
    right_chunks = []
    for table, i_col, j_col, run_col, status_col in edge_specs:
        i_pos, j_pos, active = _active_terminal_positions(
            table,
            i_col,
            j_col,
            run_col,
            node_lookup,
            node_run_mask,
            status_col=status_col,
        )
        if active.size and np.any(active):
            left_chunks.append(i_pos[active])
            right_chunks.append(j_pos[active])

    if left_chunks:
        left = np.concatenate(left_chunks).astype(np.int32, copy=False)
        right = np.concatenate(right_chunks).astype(np.int32, copy=False)
        graph = coo_matrix((np.ones(left.size, dtype=np.int8), (left, right)), shape=(n_nodes, n_nodes))
        _count, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        labels = np.arange(n_nodes, dtype=np.int32)

    running_labels = labels[running_positions]
    order = np.lexsort((node_ids[running_positions], running_labels))
    sorted_positions = running_positions[order]
    sorted_labels = running_labels[order]
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]).astype(np.int64) + 1,
            np.array([sorted_positions.size], dtype=np.int64),
        )
    )
    groups = [
        sorted_positions[int(boundaries[pos]) : int(boundaries[pos + 1])].astype(int).tolist()
        for pos in range(boundaries.size - 1)
    ]
    groups.sort(key=lambda group: int(node_ids[group[0]]))
    return groups


def _build_base_topology_arrays(
    node_ids: np.ndarray,
    node_run_mask: np.ndarray,
    bus_edge_specs: Sequence[Tuple[np.ndarray, int, int, int, Optional[int]]],
    island_edge_specs: Sequence[Tuple[np.ndarray, int, int, int, Optional[int]]],
) -> GridTopologyArrays:
    node_ids = np.asarray(node_ids, dtype=np.int32)
    node_run_mask = np.asarray(node_run_mask, dtype=bool)
    n_nodes = int(node_ids.size)
    node_lookup = _make_node_pos_lookup(node_ids)
    bus_groups = _component_position_groups(node_ids, node_run_mask, bus_edge_specs, node_lookup)
    island_node_groups = _component_position_groups(node_ids, node_run_mask, island_edge_specs, node_lookup)

    node_to_bus_pos = np.full(n_nodes, -1, dtype=np.int32)
    bus_ids = np.empty(len(bus_groups), dtype=np.int32)
    for bus_pos, group in enumerate(bus_groups):
        bus_ids[bus_pos] = int(node_ids[group[0]])
        node_to_bus_pos[np.asarray(group, dtype=np.intp)] = int(bus_pos)
    bus_node_offsets, bus_node_indices = _flatten_position_groups(bus_groups)

    node_to_island_pos = np.full(n_nodes, -1, dtype=np.int32)
    for island_pos, group in enumerate(island_node_groups):
        node_to_island_pos[np.asarray(group, dtype=np.intp)] = int(island_pos)
    island_bus_groups = [[] for _group in island_node_groups]
    bus_to_island_pos = np.full(len(bus_groups), -1, dtype=np.int32)
    for bus_pos, group in enumerate(bus_groups):
        island_pos = int(node_to_island_pos[int(group[0])])
        if island_pos < 0:
            continue
        bus_to_island_pos[bus_pos] = island_pos
        island_bus_groups[island_pos].append(bus_pos)

    island_ids = np.arange(1, len(island_bus_groups) + 1, dtype=np.int32)
    island_bus_offsets, island_bus_indices = _flatten_position_groups(island_bus_groups)

    false_islands = np.zeros(len(island_bus_groups), dtype=bool)
    false_buses = np.zeros(len(bus_groups), dtype=bool)
    false_nodes = np.zeros(n_nodes, dtype=bool)
    reference = np.full(len(island_bus_groups), -1, dtype=np.int32)
    return GridTopologyArrays(
        node_ids=node_ids,
        node_run_mask=node_run_mask,
        node_to_bus_pos=node_to_bus_pos,
        node_to_island_pos=node_to_island_pos,
        bus_ids=bus_ids,
        bus_node_offsets=bus_node_offsets,
        bus_node_indices=bus_node_indices,
        bus_to_island_pos=bus_to_island_pos,
        island_ids=island_ids,
        island_bus_offsets=island_bus_offsets,
        island_bus_indices=island_bus_indices,
        island_alive_mask=false_islands,
        bus_alive_mask=false_buses,
        node_alive_mask=false_nodes,
        island_reference_bus_pos=reference,
    )


def _single_device_arrays(
    table,
    node_col: int,
    run_col: int,
    node_lookup,
    topology: GridTopologyArrays,
) -> SingleDeviceTopologyArrays:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _empty_single_device(0)
    node_pos = _map_node_positions(rows[:, node_col], node_lookup)
    bus_pos = np.full(count, -1, dtype=np.int32)
    island_pos = np.full(count, -1, dtype=np.int32)
    valid = node_pos >= 0
    if np.any(valid):
        bus_pos[valid] = topology.node_to_bus_pos[node_pos[valid]]
        island_pos[valid] = topology.node_to_island_pos[node_pos[valid]]
    node_alive = np.zeros(count, dtype=bool)
    if np.any(valid):
        node_alive[valid] = topology.node_alive_mask[node_pos[valid]]
    alive = (rows[:, run_col].astype(np.int64, copy=False) == 1) & valid & (island_pos >= 0) & node_alive
    return SingleDeviceTopologyArrays(node_pos, bus_pos, island_pos, alive)


def _terminal_device_arrays(
    table,
    i_col: int,
    j_col: int,
    run_col: int,
    node_lookup,
    topology: GridTopologyArrays,
    status_col: Optional[int] = None,
) -> TerminalDeviceTopologyArrays:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _empty_terminal_device(0)
    i_node_pos = _map_node_positions(rows[:, i_col], node_lookup)
    j_node_pos = _map_node_positions(rows[:, j_col], node_lookup)
    i_bus_pos = np.full(count, -1, dtype=np.int32)
    j_bus_pos = np.full(count, -1, dtype=np.int32)
    i_island_pos = np.full(count, -1, dtype=np.int32)
    j_island_pos = np.full(count, -1, dtype=np.int32)
    i_valid = i_node_pos >= 0
    j_valid = j_node_pos >= 0
    if np.any(i_valid):
        i_bus_pos[i_valid] = topology.node_to_bus_pos[i_node_pos[i_valid]]
        i_island_pos[i_valid] = topology.node_to_island_pos[i_node_pos[i_valid]]
    if np.any(j_valid):
        j_bus_pos[j_valid] = topology.node_to_bus_pos[j_node_pos[j_valid]]
        j_island_pos[j_valid] = topology.node_to_island_pos[j_node_pos[j_valid]]
    run_mask = rows[:, run_col].astype(np.int64, copy=False) == 1
    if status_col is not None:
        run_mask &= rows[:, status_col].astype(np.int64, copy=False) == 1
    valid = (
        run_mask
        & i_valid
        & j_valid
        & (i_node_pos != j_node_pos)
        & (i_island_pos >= 0)
        & (j_island_pos >= 0)
    )
    alive = valid.copy()
    if np.any(valid):
        alive[valid] &= topology.node_alive_mask[i_node_pos[valid]]
        alive[valid] &= topology.node_alive_mask[j_node_pos[valid]]
    island_pos = np.full(count, -1, dtype=np.int32)
    same_island = i_island_pos == j_island_pos
    island_pos[valid & same_island] = i_island_pos[valid & same_island]
    return TerminalDeviceTopologyArrays(
        i_node_pos,
        j_node_pos,
        i_bus_pos,
        j_bus_pos,
        i_island_pos,
        j_island_pos,
        island_pos,
        alive,
    )


def _mark_reference_bus(reference: np.ndarray, island_pos: int, bus_pos: int, bus_ids: np.ndarray) -> None:
    if island_pos < 0 or bus_pos < 0:
        return
    current = int(reference[island_pos])
    if current < 0 or int(bus_ids[bus_pos]) < int(bus_ids[current]):
        reference[island_pos] = int(bus_pos)


def prepare_ac_topology_ppc(ppc: Dict) -> GridTopologyArrays:
    """Build AC bus/island topology directly from ``ac_ppc_v1`` arrays."""
    from ac_array_model import (
        BREAK_COLS,
        BRANCH_COLS,
        BUS_COLS,
        CTRL_SLACK,
        GEN_COLS,
        LOAD_COLS,
        SHUNT_COLS,
        SWITCH_COLS,
        TRANSFORMER_COLS,
        ZERO_BRANCH_COLS,
        _empty,
    )

    bus = np.asarray(ppc["bus"], dtype=np.float64)
    branch = np.asarray(ppc["branch"], dtype=np.float64)
    transformer = np.asarray(ppc["transformer"], dtype=np.float64)
    gen = np.asarray(ppc["gen"], dtype=np.float64)
    load = np.asarray(ppc["load"], dtype=np.float64)
    shunt = np.asarray(ppc["shunt"], dtype=np.float64)
    zero_branch = np.asarray(ppc["zero_branch"], dtype=np.float64)
    switch = np.asarray(ppc["switch"], dtype=np.float64)
    breaker = np.asarray(ppc.get("break", _empty(len(BREAK_COLS))), dtype=np.float64)
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int32, copy=False)
    node_run_mask = bus[:, BUS_COLS["run_stat"]].astype(np.int64, copy=False) == 1
    topology = _build_base_topology_arrays(
        node_ids,
        node_run_mask,
        bus_edge_specs=((switch, SWITCH_COLS["i_node"], SWITCH_COLS["j_node"], SWITCH_COLS["run_stat"], SWITCH_COLS["status"]),),
        island_edge_specs=(
            (switch, SWITCH_COLS["i_node"], SWITCH_COLS["j_node"], SWITCH_COLS["run_stat"], SWITCH_COLS["status"]),
            (branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], None),
            (transformer, TRANSFORMER_COLS["i_node"], TRANSFORMER_COLS["j_node"], TRANSFORMER_COLS["run_stat"], None),
            (zero_branch, ZERO_BRANCH_COLS["i_node"], ZERO_BRANCH_COLS["j_node"], ZERO_BRANCH_COLS["run_stat"], None),
            (breaker, BREAK_COLS["i_node"], BREAK_COLS["j_node"], BREAK_COLS["run_stat"], BREAK_COLS["status"]),
        ),
    )
    node_lookup = _make_node_pos_lookup(topology.node_ids)
    gen_nodes = _map_node_positions(gen[:, GEN_COLS["node"]], node_lookup) if gen.size else _EMPTY_INT
    if gen_nodes.size:
        valid_gen = gen_nodes >= 0
        gen_islands = np.full(gen_nodes.shape, -1, dtype=np.int32)
        gen_buses = np.full(gen_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_gen):
            gen_islands[valid_gen] = topology.node_to_island_pos[gen_nodes[valid_gen]]
            gen_buses[valid_gen] = topology.node_to_bus_pos[gen_nodes[valid_gen]]
        slack_mask = (
            (gen[:, GEN_COLS["run_stat"]].astype(np.int64, copy=False) == 1)
            & valid_gen
            & (gen_islands >= 0)
            & (gen[:, GEN_COLS["control_type"]].astype(np.int64, copy=False) == CTRL_SLACK)
        )
        topology.island_alive_mask[gen_islands[slack_mask]] = True
        for island_pos, bus_pos in zip(gen_islands[slack_mask], gen_buses[slack_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    valid_bus = topology.bus_to_island_pos >= 0
    if np.any(valid_bus):
        topology.bus_alive_mask[valid_bus] = topology.island_alive_mask[topology.bus_to_island_pos[valid_bus]]
    valid_node = topology.node_to_island_pos >= 0
    if np.any(valid_node):
        topology.node_alive_mask[valid_node] = (
            topology.node_run_mask[valid_node] & topology.island_alive_mask[topology.node_to_island_pos[valid_node]]
        )

    topology.devices = {
        "branch": _terminal_device_arrays(branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], node_lookup, topology),
        "transformer": _terminal_device_arrays(
            transformer,
            TRANSFORMER_COLS["i_node"],
            TRANSFORMER_COLS["j_node"],
            TRANSFORMER_COLS["run_stat"],
            node_lookup,
            topology,
        ),
        "zero_branch": _terminal_device_arrays(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
        ),
        "switch": _terminal_device_arrays(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=SWITCH_COLS["status"],
        ),
        "break": _terminal_device_arrays(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=BREAK_COLS["status"],
        ),
        "gen": _single_device_arrays(gen, GEN_COLS["node"], GEN_COLS["run_stat"], node_lookup, topology),
        "load": _single_device_arrays(load, LOAD_COLS["node"], LOAD_COLS["run_stat"], node_lookup, topology),
        "shunt": _single_device_arrays(shunt, SHUNT_COLS["node"], SHUNT_COLS["run_stat"], node_lookup, topology),
    }
    return topology


def prepare_dc_topology_ppc(ppc: Dict) -> GridTopologyArrays:
    """Build DC bus/island topology directly from ``dc_ppc_v1`` arrays."""
    from dc_array_model import (
        BREAK_COLS,
        BRANCH_COLS,
        BUS_COLS,
        CTRL_V,
        DCDC_COLS,
        GEN_COLS,
        LOAD_COLS,
        SWITCH_COLS,
        ZERO_BRANCH_COLS,
        _empty,
    )

    bus = np.asarray(ppc["bus"], dtype=np.float64)
    branch = np.asarray(ppc["branch"], dtype=np.float64)
    gen = np.asarray(ppc["gen"], dtype=np.float64)
    load = np.asarray(ppc["load"], dtype=np.float64)
    zero_branch = np.asarray(ppc["zero_branch"], dtype=np.float64)
    switch = np.asarray(ppc["switch"], dtype=np.float64)
    breaker = np.asarray(ppc.get("break", _empty(len(BREAK_COLS))), dtype=np.float64)
    dcdc = np.asarray(ppc["dcdc"], dtype=np.float64)
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int32, copy=False)
    node_run_mask = bus[:, BUS_COLS["run_stat"]].astype(np.int64, copy=False) == 1
    topology = _build_base_topology_arrays(
        node_ids,
        node_run_mask,
        bus_edge_specs=((switch, SWITCH_COLS["i_node"], SWITCH_COLS["j_node"], SWITCH_COLS["run_stat"], SWITCH_COLS["status"]),),
        island_edge_specs=(
            (switch, SWITCH_COLS["i_node"], SWITCH_COLS["j_node"], SWITCH_COLS["run_stat"], SWITCH_COLS["status"]),
            (branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], None),
            (zero_branch, ZERO_BRANCH_COLS["i_node"], ZERO_BRANCH_COLS["j_node"], ZERO_BRANCH_COLS["run_stat"], None),
            (breaker, BREAK_COLS["i_node"], BREAK_COLS["j_node"], BREAK_COLS["run_stat"], BREAK_COLS["status"]),
        ),
    )
    node_lookup = _make_node_pos_lookup(topology.node_ids)
    gen_nodes = _map_node_positions(gen[:, GEN_COLS["node"]], node_lookup) if gen.size else _EMPTY_INT
    if gen_nodes.size:
        valid_gen = gen_nodes >= 0
        gen_islands = np.full(gen_nodes.shape, -1, dtype=np.int32)
        gen_buses = np.full(gen_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_gen):
            gen_islands[valid_gen] = topology.node_to_island_pos[gen_nodes[valid_gen]]
            gen_buses[valid_gen] = topology.node_to_bus_pos[gen_nodes[valid_gen]]
        v_mask = (
            (gen[:, GEN_COLS["run_stat"]].astype(np.int64, copy=False) == 1)
            & valid_gen
            & (gen_islands >= 0)
            & (gen[:, GEN_COLS["control_type"]].astype(np.int64, copy=False) == CTRL_V)
        )
        topology.island_alive_mask[gen_islands[v_mask]] = True
        for island_pos, bus_pos in zip(gen_islands[v_mask], gen_buses[v_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    dcdc_nodes = _map_node_positions(dcdc[:, DCDC_COLS["i_node"]], node_lookup) if dcdc.size else _EMPTY_INT
    if dcdc_nodes.size:
        dcdc_j_nodes = _map_node_positions(dcdc[:, DCDC_COLS["j_node"]], node_lookup)
        valid_dcdc = (dcdc_nodes >= 0) & (dcdc_j_nodes >= 0)
        dcdc_islands = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        dcdc_buses = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_dcdc):
            dcdc_islands[valid_dcdc] = topology.node_to_island_pos[dcdc_nodes[valid_dcdc]]
            dcdc_buses[valid_dcdc] = topology.node_to_bus_pos[dcdc_nodes[valid_dcdc]]
        v_mask = (
            (dcdc[:, DCDC_COLS["run_stat"]].astype(np.int64, copy=False) == 1)
            & valid_dcdc
            & (dcdc_islands >= 0)
            & (dcdc[:, DCDC_COLS["control_type"]].astype(np.int64, copy=False) == CTRL_V)
        )
        topology.island_alive_mask[dcdc_islands[v_mask]] = True
        for island_pos, bus_pos in zip(dcdc_islands[v_mask], dcdc_buses[v_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    valid_bus = topology.bus_to_island_pos >= 0
    if np.any(valid_bus):
        topology.bus_alive_mask[valid_bus] = topology.island_alive_mask[topology.bus_to_island_pos[valid_bus]]
    valid_node = topology.node_to_island_pos >= 0
    if np.any(valid_node):
        topology.node_alive_mask[valid_node] = (
            topology.node_run_mask[valid_node] & topology.island_alive_mask[topology.node_to_island_pos[valid_node]]
        )

    topology.devices = {
        "branch": _terminal_device_arrays(branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], node_lookup, topology),
        "zero_branch": _terminal_device_arrays(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
        ),
        "switch": _terminal_device_arrays(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=SWITCH_COLS["status"],
        ),
        "break": _terminal_device_arrays(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=BREAK_COLS["status"],
        ),
        "dcdc": _terminal_device_arrays(
            dcdc,
            DCDC_COLS["i_node"],
            DCDC_COLS["j_node"],
            DCDC_COLS["run_stat"],
            node_lookup,
            topology,
        ),
        "gen": _single_device_arrays(gen, GEN_COLS["node"], GEN_COLS["run_stat"], node_lookup, topology),
        "load": _single_device_arrays(load, LOAD_COLS["node"], LOAD_COLS["run_stat"], node_lookup, topology),
    }
    return topology


def _find_parent(parent, item):
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        item, parent[item] = parent[item], root
    return root


def _union_parent(parent, left, right):
    root_l = _find_parent(parent, left)
    root_r = _find_parent(parent, right)
    if root_l != root_r:
        parent[root_r] = root_l


def _make_parent_index(ids):
    ids = [int(item) for item in ids]
    if not ids:
        return {}
    min_id = min(ids)
    max_id = max(ids)
    if min_id >= 0 and max_id <= max(1024, len(ids) * 4):
        parent = [-1] * (max_id + 1)
        for item in ids:
            parent[item] = item
        return parent
    return {item: item for item in ids}


def _parent_contains(parent, item):
    item = int(item)
    if isinstance(parent, list):
        return 0 <= item < len(parent) and parent[item] >= 0
    return item in parent


def _terminal_pair(dev, running_ids, require_closed=False):
    left = int(dev.i_node)
    right = int(dev.j_node)
    if (
        dev.run_stat == 1
        and (not require_closed or getattr(dev, "status", 1) == 1)
        and _parent_contains(running_ids, left)
        and _parent_contains(running_ids, right)
        and left != right
    ):
        return left, right
    return None


def _device_seq(network, attr):
    return getattr(network, attr, None) or ()


def _sorted_by_idx(values):
    values = list(values)
    if len(values) < 2:
        return values
    previous = int(getattr(values[0], "idx", 0))
    for item in values[1:]:
        current = int(getattr(item, "idx", 0))
        if current < previous:
            return sorted(values, key=lambda entry: entry.idx)
        previous = current
    return values


def _topology_device_mask(topology: GridTopologyArrays, key: str, count: int) -> np.ndarray:
    device_topology = topology.devices.get(key) if topology.devices else None
    if device_topology is None:
        return np.zeros(int(count), dtype=bool)
    mask = np.asarray(getattr(device_topology, "alive_mask", np.zeros(int(count), dtype=bool)), dtype=bool)
    if mask.size == count:
        return mask
    out = np.zeros(int(count), dtype=bool)
    out[: min(count, mask.size)] = mask[: min(count, mask.size)]
    return out


def _topology_terminal_island_pos(topology: GridTopologyArrays, key: str, count: int) -> np.ndarray:
    device_topology = topology.devices.get(key) if topology.devices else None
    if device_topology is None:
        return np.full(int(count), -1, dtype=np.int32)
    values = np.asarray(getattr(device_topology, "island_pos", np.full(int(count), -1, dtype=np.int32)), dtype=np.int32)
    if values.size == count:
        return values
    out = np.full(int(count), -1, dtype=np.int32)
    out[: min(count, values.size)] = values[: min(count, values.size)]
    return out


def _topology_single_node_pos(topology: GridTopologyArrays, key: str, count: int) -> np.ndarray:
    device_topology = topology.devices.get(key) if topology.devices else None
    if device_topology is None:
        return np.full(int(count), -1, dtype=np.int32)
    values = np.asarray(getattr(device_topology, "node_pos", np.full(int(count), -1, dtype=np.int32)), dtype=np.int32)
    if values.size == count:
        return values
    out = np.full(int(count), -1, dtype=np.int32)
    out[: min(count, values.size)] = values[: min(count, values.size)]
    return out


def _topology_terminal_node_pos(topology: GridTopologyArrays, key: str, count: int) -> Tuple[np.ndarray, np.ndarray]:
    device_topology = topology.devices.get(key) if topology.devices else None
    if device_topology is None:
        empty = np.full(int(count), -1, dtype=np.int32)
        return empty, empty.copy()
    i_values = np.asarray(getattr(device_topology, "i_node_pos", np.full(int(count), -1, dtype=np.int32)), dtype=np.int32)
    j_values = np.asarray(getattr(device_topology, "j_node_pos", np.full(int(count), -1, dtype=np.int32)), dtype=np.int32)
    if i_values.size == count and j_values.size == count:
        return i_values, j_values
    i_out = np.full(int(count), -1, dtype=np.int32)
    j_out = np.full(int(count), -1, dtype=np.int32)
    i_out[: min(count, i_values.size)] = i_values[: min(count, i_values.size)]
    j_out[: min(count, j_values.size)] = j_values[: min(count, j_values.size)]
    return i_out, j_out


def _apply_common_node_bus_island_arrays(network, topology: GridTopologyArrays, bus_cls, island_factory, *, compact: bool = False):
    nodes = _device_seq(network, "nodes")
    node_dict = {int(node.idx): node for node in nodes}
    network.node_dict = node_dict

    node_alive_mask = topology.node_alive_mask
    for node_pos, node in enumerate(nodes):
        node.isl = 0
        node.isl_obj = None
        node.bus = None
        node.bus_obj = None
        node.is_alive = bool(node_alive_mask[node_pos]) if node_pos < node_alive_mask.size else False

    islands = []
    for pos, island_id in enumerate(topology.island_ids):
        island = island_factory(int(island_id), bool(topology.island_alive_mask[pos]))
        island.is_alive = bool(topology.island_alive_mask[pos])
        islands.append(island)
    network.islands = islands

    buses = []
    network.bus_dict = {}
    network.node_to_bus = {}
    for bus_pos, bus_id in enumerate(topology.bus_ids):
        start = int(topology.bus_node_offsets[bus_pos])
        end = int(topology.bus_node_offsets[bus_pos + 1])
        grouped_nodes = [nodes[int(node_pos)] for node_pos in topology.bus_node_indices[start:end]]
        bus = _make_compact_bus(bus_cls, int(bus_id), grouped_nodes) if compact else bus_cls(int(bus_id), grouped_nodes)
        bus.is_alive = bool(topology.bus_alive_mask[bus_pos])
        buses.append(bus)
        if not compact:
            network.bus_dict[int(bus.idx)] = bus
        island_pos = int(topology.bus_to_island_pos[bus_pos])
        island = islands[island_pos] if island_pos >= 0 else None
        if island is not None:
            bus.isl = int(island.idx)
            bus.isl_obj = island
            island.buses.append(bus)
        for node in grouped_nodes:
            node.bus = int(bus.idx)
            node.bus_obj = bus
            if not compact:
                network.node_to_bus[int(node.idx)] = bus
            if island is not None:
                node.isl = int(island.idx)
                node.isl_obj = island
    network.buses = buses
    return nodes, node_dict, buses, islands


def apply_ac_topology_arrays(
    network,
    topology: GridTopologyArrays,
    *,
    compact: bool = False,
    build_alive_maps: bool = True,
) -> None:
    """Populate AC object topology fields from precomputed ppc topology arrays.

    ``compact`` keeps only the object links required by SE compatibility code.
    The full reverse device lists remain the default for LF/full-result callers.
    """
    from model.ac_model import ACBus, ACIsl

    branches = _device_seq(network, "branches")
    transformers = _device_seq(network, "transformers")
    generators = _device_seq(network, "generators")
    loads = _device_seq(network, "loads")
    shunts = _device_seq(network, "shunt_compensators")
    zero_branches = _device_seq(network, "zero_branches")
    switches = _device_seq(network, "switches")
    breakers = _device_seq(network, "breakers")

    nodes, node_dict, buses, _islands = _apply_common_node_bus_island_arrays(
        network,
        topology,
        ACBus,
        lambda idx, is_alive: _make_ac_island(idx, ACIsl),
        compact=compact,
    )
    if compact:
        network.branch_dict = {}
        network.transformer_dict = {}
        network.generator_dict = {}
        network.load_dict = {}
        network.shunt_compensator_dict = {}
        network.zero_branch_dict = {}
        network.switch_dict = {}
        network.break_dict = {}
    else:
        network.branch_dict = {int(dev.idx): dev for dev in branches}
        network.transformer_dict = {int(dev.idx): dev for dev in transformers}
        network.generator_dict = {int(dev.idx): dev for dev in generators}
        network.load_dict = {int(dev.idx): dev for dev in loads}
        network.shunt_compensator_dict = {int(dev.idx): dev for dev in shunts}
        network.zero_branch_dict = {int(dev.idx): dev for dev in zero_branches}
        network.switch_dict = {int(dev.idx): dev for dev in switches}
        network.break_dict = {int(dev.idx): dev for dev in breakers}

    if not compact:
        for node in nodes:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.zero_branches = []
            node.transformers = []
            node.shunt_compensators = []
            node.v_gens = []

    gen_alive = _topology_device_mask(topology, "gen", len(generators))
    gen_node_pos = _topology_single_node_pos(topology, "gen", len(generators))
    for pos, gen in enumerate(generators):
        node_pos = int(gen_node_pos[pos]) if pos < gen_node_pos.size else -1
        node = nodes[node_pos] if 0 <= node_pos < len(nodes) else node_dict.get(int(gen.node))
        gen.node_obj = node
        gen.is_alive = bool(gen_alive[pos]) if pos < gen_alive.size else False
        if node is None:
            continue
        if not compact:
            node.generators.append(gen)
        island = node.isl_obj
        if int(getattr(gen, "run_stat", 1)) != 1 or island is None:
            continue
        if not compact:
            island.gens.append(gen)
        if gen.control_type in ("V", "SLACK", "PH"):
            if not compact:
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                island.v_gens.append(gen)
            slack_bus = node.bus_obj or node
            if slack_bus not in island.slack_nodes:
                island.slack_nodes.append(slack_bus)
        elif gen.control_type == "PV":
            if not compact:
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                island.v_gens.append(gen)

    load_alive = _topology_device_mask(topology, "load", len(loads))
    load_node_pos = _topology_single_node_pos(topology, "load", len(loads))
    for pos, load in enumerate(loads):
        node_pos = int(load_node_pos[pos]) if pos < load_node_pos.size else -1
        node = nodes[node_pos] if 0 <= node_pos < len(nodes) else node_dict.get(int(load.node))
        load.node_obj = node
        load.is_alive = bool(load_alive[pos]) if pos < load_alive.size else False
        if node is not None:
            if not compact:
                node.loads.append(load)
                if load.is_alive and node.isl_obj is not None:
                    node.isl_obj.loads.append(load)

    shunt_alive = _topology_device_mask(topology, "shunt", len(shunts))
    shunt_node_pos = _topology_single_node_pos(topology, "shunt", len(shunts))
    for pos, shunt in enumerate(shunts):
        node_pos = int(shunt_node_pos[pos]) if pos < shunt_node_pos.size else -1
        node = nodes[node_pos] if 0 <= node_pos < len(nodes) else node_dict.get(int(shunt.node))
        shunt.node_obj = node
        shunt.is_alive = bool(shunt_alive[pos]) if pos < shunt_alive.size else False
        if node is not None:
            if not compact:
                node.shunt_compensators.append(shunt)
                if shunt.is_alive and node.isl_obj is not None:
                    node.isl_obj.shunt_compensators.append(shunt)

    def finalize_terminal(devices, key: str, node_attr: str, island_attr: str):
        alive = _topology_device_mask(topology, key, len(devices))
        same_island_pos = _topology_terminal_island_pos(topology, key, len(devices))
        i_node_pos, j_node_pos = _topology_terminal_node_pos(topology, key, len(devices))
        for pos, dev in enumerate(devices):
            i_pos = int(i_node_pos[pos]) if pos < i_node_pos.size else -1
            j_pos = int(j_node_pos[pos]) if pos < j_node_pos.size else -1
            i_node = nodes[i_pos] if 0 <= i_pos < len(nodes) else node_dict.get(int(dev.i_node))
            j_node = nodes[j_pos] if 0 <= j_pos < len(nodes) else node_dict.get(int(dev.j_node))
            dev.i_node_obj = i_node
            dev.j_node_obj = j_node
            dev.is_alive = bool(alive[pos]) if pos < alive.size else False
            if not compact and i_node is not None:
                getattr(i_node, node_attr).append(dev)
            if not compact and j_node is not None:
                getattr(j_node, node_attr).append(dev)
            if (
                not compact
                and dev.is_alive
                and pos < same_island_pos.size
                and int(same_island_pos[pos]) >= 0
                and i_node is not None
            ):
                island = i_node.isl_obj
                if island is not None:
                    getattr(island, island_attr).append(dev)

    finalize_terminal(branches, "branch", "branches", "branches")
    finalize_terminal(transformers, "transformer", "transformers", "transformers")
    finalize_terminal(zero_branches, "zero_branch", "zero_branches", "zero_branches")
    finalize_terminal(switches, "switch", "switches", "switches")
    finalize_terminal(breakers, "break", "breakers", "breakers")

    if build_alive_maps:
        network.alive_nodes = [bus for bus in buses if bus.is_alive]
        network.alive_buses = network.alive_nodes
        network.alive_branch_by_name = {br.name: br for br in branches if br.is_alive}
        network.alive_transformer_by_name = {tr.name: tr for tr in transformers if tr.is_alive}
        network.alive_generator_by_name = {gen.name: gen for gen in generators if gen.is_alive}
        network.alive_load_by_name = {load.name: load for load in loads if load.is_alive}
        network.alive_zero_branch_by_name = {zbr.name: zbr for zbr in zero_branches if zbr.is_alive}
        network.alive_switch_by_name = {sw.name: sw for sw in switches if sw.is_alive}
        network.alive_break_by_name = {brk.name: brk for brk in breakers if brk.is_alive}
        network.alive_zero_branches = _sorted_by_idx(network.alive_zero_branch_by_name.values())
        network.alive_switches = _sorted_by_idx(network.alive_switch_by_name.values())
        network.alive_breakers = _sorted_by_idx(network.alive_break_by_name.values())
        network.alive_generator_order = _sorted_by_idx(network.alive_generator_by_name.values())
        network.alive_load_order = _sorted_by_idx(network.alive_load_by_name.values())
    else:
        network.alive_nodes = [bus for bus in buses if bus.is_alive]
        network.alive_buses = network.alive_nodes
        network.alive_branch_by_name = {}
        network.alive_transformer_by_name = {}
        network.alive_generator_by_name = {}
        network.alive_load_by_name = {}
        network.alive_zero_branch_by_name = {}
        network.alive_switch_by_name = {}
        network.alive_break_by_name = {}
        network.alive_zero_branches = []
        network.alive_switches = []
        network.alive_breakers = []
        network.alive_generator_order = []
        network.alive_load_order = []


def apply_dc_topology_arrays(
    network,
    topology: GridTopologyArrays,
    *,
    compact: bool = False,
    build_alive_maps: bool = True,
) -> None:
    """Populate DC object topology fields from precomputed ppc topology arrays.

    ``compact`` keeps only the object links required by SE compatibility code.
    The full reverse device lists remain the default for LF/full-result callers.
    """
    from model.dc_model import DCBus, DCIsl

    branches = _device_seq(network, "branches")
    generators = _device_seq(network, "generators")
    loads = _device_seq(network, "loads")
    zero_branches = _device_seq(network, "zero_branches")
    switches = _device_seq(network, "switches")
    breakers = _device_seq(network, "breakers")
    dcdc_converters = _device_seq(network, "dcdc_converters")

    nodes, node_dict, buses, _islands = _apply_common_node_bus_island_arrays(
        network,
        topology,
        DCBus,
        lambda idx, is_alive: DCIsl(idx, is_alive),
        compact=compact,
    )
    if compact:
        network.switch_dict = {}
        network.break_dict = {}
        network.load_dict = {}
        network.generator_dict = {}
        network.zero_branch_dict = {}
        network.zero_branche_dict = network.zero_branch_dict
        network.branch_dict = {}
        network.branche_dict = network.branch_dict
        network.dcdc_converter_dict = {}
    else:
        network.switch_dict = {int(dev.idx): dev for dev in switches}
        network.break_dict = {int(dev.idx): dev for dev in breakers}
        network.load_dict = {int(dev.idx): dev for dev in loads}
        network.generator_dict = {int(dev.idx): dev for dev in generators}
        network.zero_branch_dict = {int(dev.idx): dev for dev in zero_branches}
        network.zero_branche_dict = network.zero_branch_dict
        network.branch_dict = {int(dev.idx): dev for dev in branches}
        network.branche_dict = network.branch_dict
        network.dcdc_converter_dict = {int(dev.idx): dev for dev in dcdc_converters}

    for node in nodes:
        node.v_set = 0.0
        node.is_slack = False
        if not compact:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.dcdc_converters = []
            node.zero_branches = []
            node.v_gens = []
            node.v_dcdcs = []

    gen_alive = _topology_device_mask(topology, "gen", len(generators))
    gen_node_pos = _topology_single_node_pos(topology, "gen", len(generators))
    for pos, gen in enumerate(generators):
        node_pos = int(gen_node_pos[pos]) if pos < gen_node_pos.size else -1
        node = nodes[node_pos] if 0 <= node_pos < len(nodes) else node_dict.get(int(gen.node))
        gen.node_obj = node
        gen.is_alive = bool(gen_alive[pos]) if pos < gen_alive.size else False
        if node is None:
            continue
        if not compact:
            node.generators.append(gen)
        if int(getattr(gen, "run_stat", 1)) != 1 or node.isl_obj is None:
            continue
        if not compact:
            node.isl_obj.gens.append(gen)
        if gen.control_type == "V":
            node.v_set = float(getattr(gen, "v_set", node.v_set))
            if node.bus_obj is not None:
                node.bus_obj.v_set = float(getattr(gen, "v_set", node.bus_obj.v_set))
            slack_bus = node.bus_obj or node
            if slack_bus not in node.isl_obj.slack_nodes:
                node.isl_obj.slack_nodes.append(slack_bus)
            if compact:
                node.is_slack = True
                if node.bus_obj is not None:
                    node.bus_obj.is_slack = True
            if not compact:
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)

    dcdc_alive = _topology_device_mask(topology, "dcdc", len(dcdc_converters))
    dcdc_i_node_pos, dcdc_j_node_pos = _topology_terminal_node_pos(topology, "dcdc", len(dcdc_converters))
    for pos, conv in enumerate(dcdc_converters):
        i_pos = int(dcdc_i_node_pos[pos]) if pos < dcdc_i_node_pos.size else -1
        j_pos = int(dcdc_j_node_pos[pos]) if pos < dcdc_j_node_pos.size else -1
        i_node = nodes[i_pos] if 0 <= i_pos < len(nodes) else node_dict.get(int(conv.i_node))
        j_node = nodes[j_pos] if 0 <= j_pos < len(nodes) else node_dict.get(int(conv.j_node))
        conv.i_node_obj = i_node
        conv.j_node_obj = j_node
        conv.is_alive = bool(dcdc_alive[pos]) if pos < dcdc_alive.size else False
        if not compact and i_node is not None:
            i_node.dcdc_converters.append(conv)
        if not compact and j_node is not None:
            j_node.dcdc_converters.append(conv)
        if int(getattr(conv, "run_stat", 1)) != 1 or i_node is None or j_node is None:
            continue
        if not compact and i_node.isl_obj is not None:
            i_node.isl_obj.dcdc_converters.append(conv)
        if not compact and j_node.isl_obj is not None:
            j_node.isl_obj.dcdc_converters.append(conv)
        if conv.control_type == "V" and i_node.isl_obj is not None:
            i_node.v_set = float(getattr(conv, "v_set", i_node.v_set))
            if i_node.bus_obj is not None:
                i_node.bus_obj.v_set = float(getattr(conv, "v_set", i_node.bus_obj.v_set))
            slack_bus = i_node.bus_obj or i_node
            if slack_bus not in i_node.isl_obj.slack_nodes:
                i_node.isl_obj.slack_nodes.append(slack_bus)
            if compact:
                i_node.is_slack = True
                if i_node.bus_obj is not None:
                    i_node.bus_obj.is_slack = True
            if not compact:
                i_node.v_dcdcs.append(conv)
                if i_node.bus_obj is not None:
                    i_node.bus_obj.v_dcdcs.append(conv)
                i_node.isl_obj.v_dcdcs.append(conv)

    load_alive = _topology_device_mask(topology, "load", len(loads))
    load_node_pos = _topology_single_node_pos(topology, "load", len(loads))
    for pos, load in enumerate(loads):
        node_pos = int(load_node_pos[pos]) if pos < load_node_pos.size else -1
        node = nodes[node_pos] if 0 <= node_pos < len(nodes) else node_dict.get(int(load.node))
        load.node_obj = node
        load.is_alive = bool(load_alive[pos]) if pos < load_alive.size else False
        if node is not None:
            if not compact:
                node.loads.append(load)
                if int(getattr(load, "run_stat", 1)) == 1 and node.isl_obj is not None:
                    node.isl_obj.loads.append(load)

    def finalize_terminal(devices, key: str, node_attr: str, island_attr: str, append_to_both_islands: bool = False):
        alive = _topology_device_mask(topology, key, len(devices))
        same_island_pos = _topology_terminal_island_pos(topology, key, len(devices))
        i_node_pos, j_node_pos = _topology_terminal_node_pos(topology, key, len(devices))
        for pos, dev in enumerate(devices):
            i_pos = int(i_node_pos[pos]) if pos < i_node_pos.size else -1
            j_pos = int(j_node_pos[pos]) if pos < j_node_pos.size else -1
            i_node = nodes[i_pos] if 0 <= i_pos < len(nodes) else node_dict.get(int(dev.i_node))
            j_node = nodes[j_pos] if 0 <= j_pos < len(nodes) else node_dict.get(int(dev.j_node))
            dev.i_node_obj = i_node
            dev.j_node_obj = j_node
            dev.is_alive = bool(alive[pos]) if pos < alive.size else False
            if not compact and i_node is not None:
                getattr(i_node, node_attr).append(dev)
            if not compact and j_node is not None:
                getattr(j_node, node_attr).append(dev)
            if not dev.is_alive:
                continue
            if compact:
                continue
            if append_to_both_islands:
                if i_node is not None and i_node.isl_obj is not None:
                    getattr(i_node.isl_obj, island_attr).append(dev)
                if j_node is not None and j_node.isl_obj is not None and j_node.isl_obj is not getattr(i_node, "isl_obj", None):
                    getattr(j_node.isl_obj, island_attr).append(dev)
            elif pos < same_island_pos.size and int(same_island_pos[pos]) >= 0 and i_node is not None and i_node.isl_obj is not None:
                getattr(i_node.isl_obj, island_attr).append(dev)

    finalize_terminal(branches, "branch", "branches", "branches")
    finalize_terminal(zero_branches, "zero_branch", "zero_branches", "zero_branches")
    finalize_terminal(switches, "switch", "switches", "switches")
    finalize_terminal(breakers, "break", "breakers", "breakers")

    for bus in buses:
        if bus.isl_obj is not None and (len(bus.v_gens) + len(bus.v_dcdcs)) > 0:
            bus.is_slack = True
            if bus not in bus.isl_obj.slack_nodes:
                bus.isl_obj.slack_nodes.append(bus)

    if build_alive_maps:
        network.alive_buses = [bus for bus in buses if bus.is_alive]
        network.alive_nodes = network.alive_buses
        network.alive_branch_by_name = {br.name: br for br in branches if br.is_alive}
        network.alive_generator_by_name = {gen.name: gen for gen in generators if gen.is_alive}
        network.alive_load_by_name = {load.name: load for load in loads if load.is_alive}
        network.alive_zero_branch_by_name = {zbr.name: zbr for zbr in zero_branches if zbr.is_alive}
        network.alive_switch_by_name = {sw.name: sw for sw in switches if sw.is_alive}
        network.alive_break_by_name = {brk.name: brk for brk in breakers if brk.is_alive}
        network.alive_dcdc_by_name = {conv.name: conv for conv in dcdc_converters if conv.is_alive}
        network.alive_zero_branches = _sorted_by_idx(network.alive_zero_branch_by_name.values())
        network.alive_switches = _sorted_by_idx(network.alive_switch_by_name.values())
        network.alive_breakers = _sorted_by_idx(network.alive_break_by_name.values())
        network.alive_generator_order = _sorted_by_idx(network.alive_generator_by_name.values())
        network.alive_load_order = _sorted_by_idx(network.alive_load_by_name.values())
        network.alive_dcdc_order = _sorted_by_idx(network.alive_dcdc_by_name.values())
    else:
        network.alive_buses = [bus for bus in buses if bus.is_alive]
        network.alive_nodes = network.alive_buses
        network.alive_branch_by_name = {}
        network.alive_generator_by_name = {}
        network.alive_load_by_name = {}
        network.alive_zero_branch_by_name = {}
        network.alive_switch_by_name = {}
        network.alive_break_by_name = {}
        network.alive_dcdc_by_name = {}
        network.alive_zero_branches = []
        network.alive_switches = []
        network.alive_breakers = []
        network.alive_generator_order = []
        network.alive_load_order = []
        network.alive_dcdc_order = []


def _make_ac_bus(grouped_nodes, bus_cls):
    return bus_cls(grouped_nodes[0].idx, grouped_nodes)


def _make_ac_island(idx, island_cls):
    island = island_cls(idx, False)
    if not hasattr(island, "transformers"):
        island.transformers = []
    if not hasattr(island, "shunt_compensators"):
        island.shunt_compensators = []
    return island


def prepare_ac_topology(network) -> None:
    """Build AC object topology fields through the shared fast path.

    The resulting object graph matches the fields used by LF and SE:
    ``node_dict``, bus grouping through closed switches, live islands, device
    back-references, and alive device name caches.
    """
    from model.ac_model import ACBus, ACIsl

    nodes = _device_seq(network, "nodes")
    branches = _device_seq(network, "branches")
    transformers = _device_seq(network, "transformers")
    generators = _device_seq(network, "generators")
    loads = _device_seq(network, "loads")
    shunts = _device_seq(network, "shunt_compensators")
    zero_branches = _device_seq(network, "zero_branches")
    switches = _device_seq(network, "switches")
    breakers = _device_seq(network, "breakers")

    node_dict = {node.idx: node for node in nodes}
    network.node_dict = node_dict
    network.branch_dict = {dev.idx: dev for dev in branches}
    network.transformer_dict = {dev.idx: dev for dev in transformers}
    network.generator_dict = {dev.idx: dev for dev in generators}
    network.load_dict = {dev.idx: dev for dev in loads}
    network.shunt_compensator_dict = {dev.idx: dev for dev in shunts}
    network.zero_branch_dict = {dev.idx: dev for dev in zero_branches}
    network.switch_dict = {dev.idx: dev for dev in switches}
    network.break_dict = {dev.idx: dev for dev in breakers}

    for node in nodes:
        node.isl = 0
        node.isl_obj = None
        node.bus = None
        node.bus_obj = None
        node.is_alive = False
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.zero_branches = []
        node.transformers = []
        node.shunt_compensators = []
        node.v_gens = []

    running_nodes = [node for node in nodes if node.run_stat == 1]
    running_ids = _make_parent_index(node.idx for node in running_nodes)
    parent = running_ids
    if isinstance(running_ids, list):
        running_id_count = len(running_ids)

        def contains_running_node(node_idx: int) -> bool:
            return 0 <= node_idx < running_id_count and running_ids[node_idx] >= 0

    else:

        def contains_running_node(node_idx: int) -> bool:
            return node_idx in running_ids

    def live_terminal_pair(dev, require_closed=False):
        left = int(dev.i_node)
        right = int(dev.j_node)
        if (
            dev.run_stat == 1
            and (not require_closed or getattr(dev, "status", 1) == 1)
            and left != right
            and contains_running_node(left)
            and contains_running_node(right)
        ):
            return left, right
        return None

    for dev in switches:
        pair = live_terminal_pair(dev, require_closed=True)
        if pair is not None:
            _union_parent(parent, pair[0], pair[1])

    root_to_nodes = {}
    for node in running_nodes:
        root_to_nodes.setdefault(_find_parent(parent, node.idx), []).append(node)

    network.buses = []
    network.bus_dict = {}
    network.node_to_bus = {}
    buses = network.buses
    bus_dict = network.bus_dict
    node_to_bus = network.node_to_bus
    for grouped_nodes in sorted(root_to_nodes.values(), key=lambda group: min(node.idx for node in group)):
        if len(grouped_nodes) > 1:
            grouped_nodes.sort(key=lambda item: item.idx)
        bus = _make_ac_bus(grouped_nodes, ACBus)
        buses.append(bus)
        bus_dict[bus.idx] = bus
        for node in grouped_nodes:
            node.bus = bus.idx
            node.bus_obj = bus
            node_to_bus[node.idx] = bus

    bus_parent = _make_parent_index(bus.idx for bus in buses)
    get_bus = node_to_bus.get
    union_bus_parent = _union_parent

    def connect_bus_device(dev, require_closed=False):
        pair = live_terminal_pair(dev, require_closed=require_closed)
        if pair is None:
            return
        i_bus = get_bus(pair[0])
        j_bus = get_bus(pair[1])
        if i_bus is not None and j_bus is not None and i_bus.idx != j_bus.idx:
            union_bus_parent(bus_parent, i_bus.idx, j_bus.idx)

    for dev in branches:
        connect_bus_device(dev)
    for dev in transformers:
        connect_bus_device(dev)
    for dev in zero_branches:
        connect_bus_device(dev)
    for dev in breakers:
        connect_bus_device(dev, require_closed=True)

    root_to_island = {}
    for bus in buses:
        root = _find_parent(bus_parent, bus.idx)
        island = root_to_island.get(root)
        if island is None:
            island = _make_ac_island(len(root_to_island) + 1, ACIsl)
            root_to_island[root] = island
        bus.isl = island.idx
        bus.isl_obj = island
        island.buses.append(bus)
        for node in bus.nodes:
            node.isl = island.idx
            node.isl_obj = island
    network.islands = list(root_to_island.values())

    for gen in generators:
        gen.node_obj = node_dict.get(gen.node)
        gen.is_alive = False
        if gen.node_obj is None:
            continue
        gen.node_obj.generators.append(gen)
        island = gen.node_obj.isl_obj
        if gen.run_stat != 1 or island is None:
            continue
        island.gens.append(gen)
        if gen.control_type in ("V", "SLACK", "PH"):
            island.is_alive = True
            gen.node_obj.v_gens.append(gen)
            if gen.node_obj.bus_obj is not None:
                gen.node_obj.bus_obj.v_gens.append(gen)
            island.v_gens.append(gen)
            slack_bus = gen.node_obj.bus_obj or gen.node_obj
            if slack_bus not in island.slack_nodes:
                island.slack_nodes.append(slack_bus)
        elif gen.control_type == "PV":
            gen.node_obj.v_gens.append(gen)
            if gen.node_obj.bus_obj is not None:
                gen.node_obj.bus_obj.v_gens.append(gen)
            island.v_gens.append(gen)

    for bus in buses:
        bus.is_alive = bus.run_stat == 1 and bus.isl_obj is not None and bus.isl_obj.is_alive
    for node in nodes:
        node.is_alive = node.run_stat == 1 and node.isl_obj is not None and node.isl_obj.is_alive
    for gen in generators:
        gen.is_alive = gen.node_obj is not None and gen.run_stat == 1 and gen.node_obj.is_alive

    for load in loads:
        load.node_obj = node_dict.get(load.node)
        load.is_alive = load.node_obj is not None and load.run_stat == 1 and load.node_obj.is_alive
        if load.node_obj is not None:
            load.node_obj.loads.append(load)
            if load.is_alive:
                load.node_obj.isl_obj.loads.append(load)

    for shunt in shunts:
        shunt.node_obj = node_dict.get(shunt.node)
        shunt.is_alive = shunt.node_obj is not None and shunt.run_stat == 1 and shunt.node_obj.is_alive
        if shunt.node_obj is not None:
            shunt.node_obj.shunt_compensators.append(shunt)
            if shunt.is_alive:
                shunt.node_obj.isl_obj.shunt_compensators.append(shunt)

    def finalize_branch_like(dev, attr_name, require_closed=False):
        dev.i_node_obj = node_dict.get(dev.i_node)
        dev.j_node_obj = node_dict.get(dev.j_node)
        closed = (not require_closed) or getattr(dev, "status", 1) == 1
        dev.is_alive = (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.run_stat == 1
            and closed
            and dev.i_node_obj.is_alive
            and dev.j_node_obj.is_alive
        )
        if dev.i_node_obj is not None:
            getattr(dev.i_node_obj, attr_name).append(dev)
        if dev.j_node_obj is not None:
            getattr(dev.j_node_obj, attr_name).append(dev)
        if dev.is_alive and dev.i_node_obj.isl_obj is dev.j_node_obj.isl_obj:
            getattr(dev.i_node_obj.isl_obj, attr_name).append(dev)

    for dev in branches:
        finalize_branch_like(dev, "branches")
    for dev in transformers:
        finalize_branch_like(dev, "transformers")
    for dev in zero_branches:
        finalize_branch_like(dev, "zero_branches")
    for dev in breakers:
        finalize_branch_like(dev, "breakers", require_closed=True)
    for dev in switches:
        finalize_branch_like(dev, "switches", require_closed=True)

    network.alive_nodes = [bus for bus in buses if bus.is_alive]
    network.alive_buses = network.alive_nodes
    network.alive_branch_by_name = {br.name: br for br in branches if br.is_alive}
    network.alive_transformer_by_name = {tr.name: tr for tr in transformers if tr.is_alive}
    network.alive_generator_by_name = {gen.name: gen for gen in generators if gen.is_alive}
    network.alive_load_by_name = {load.name: load for load in loads if load.is_alive}
    network.alive_zero_branch_by_name = {zbr.name: zbr for zbr in zero_branches if zbr.is_alive}
    network.alive_switch_by_name = {sw.name: sw for sw in switches if sw.is_alive}
    network.alive_break_by_name = {brk.name: brk for brk in breakers if brk.is_alive}
    network.alive_zero_branches = _sorted_by_idx(network.alive_zero_branch_by_name.values())
    network.alive_switches = _sorted_by_idx(network.alive_switch_by_name.values())
    network.alive_breakers = _sorted_by_idx(network.alive_break_by_name.values())
    network.alive_generator_order = _sorted_by_idx(network.alive_generator_by_name.values())
    network.alive_load_order = _sorted_by_idx(network.alive_load_by_name.values())


def _make_dc_bus(grouped_nodes, bus_cls):
    return bus_cls(grouped_nodes[0].idx, grouped_nodes)


def _make_dc_island(idx, island_cls):
    return island_cls(idx, False)


def prepare_dc_topology(network) -> None:
    """Build DC object topology fields through the shared fast path."""
    from model.dc_model import DCBus, DCIsl

    nodes = _device_seq(network, "nodes")
    branches = _device_seq(network, "branches")
    generators = _device_seq(network, "generators")
    loads = _device_seq(network, "loads")
    zero_branches = _device_seq(network, "zero_branches")
    switches = _device_seq(network, "switches")
    breakers = _device_seq(network, "breakers")
    dcdc_converters = _device_seq(network, "dcdc_converters")

    node_dict = {node.idx: node for node in nodes}
    network.node_dict = node_dict
    network.switch_dict = {dev.idx: dev for dev in switches}
    network.break_dict = {dev.idx: dev for dev in breakers}
    network.load_dict = {dev.idx: dev for dev in loads}
    network.generator_dict = {dev.idx: dev for dev in generators}
    network.zero_branch_dict = {dev.idx: dev for dev in zero_branches}
    network.zero_branche_dict = network.zero_branch_dict
    network.branch_dict = {dev.idx: dev for dev in branches}
    network.branche_dict = network.branch_dict
    network.dcdc_converter_dict = {dev.idx: dev for dev in dcdc_converters}

    for node in nodes:
        node.isl = 0
        node.isl_obj = None
        node.bus = None
        node.bus_obj = None
        node.is_alive = False
        node.v_set = 0.0
        node.is_slack = False
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.dcdc_converters = []
        node.zero_branches = []
        node.v_gens = []
        node.v_dcdcs = []

    running_nodes = [node for node in nodes if node.run_stat == 1]
    running_ids = _make_parent_index(node.idx for node in running_nodes)
    parent = running_ids
    if isinstance(running_ids, list):
        running_id_count = len(running_ids)

        def contains_running_node(node_idx: int) -> bool:
            return 0 <= node_idx < running_id_count and running_ids[node_idx] >= 0

    else:

        def contains_running_node(node_idx: int) -> bool:
            return node_idx in running_ids

    def live_terminal_pair(dev, require_closed=False):
        left = int(dev.i_node)
        right = int(dev.j_node)
        if (
            dev.run_stat == 1
            and (not require_closed or getattr(dev, "status", 1) == 1)
            and left != right
            and contains_running_node(left)
            and contains_running_node(right)
        ):
            return left, right
        return None

    for dev in switches:
        pair = live_terminal_pair(dev, require_closed=True)
        if pair is not None:
            _union_parent(parent, pair[0], pair[1])

    root_to_nodes = {}
    for node in running_nodes:
        root_to_nodes.setdefault(_find_parent(parent, node.idx), []).append(node)

    network.buses = []
    network.bus_dict = {}
    network.node_to_bus = {}
    buses = network.buses
    bus_dict = network.bus_dict
    node_to_bus = network.node_to_bus
    for grouped_nodes in sorted(root_to_nodes.values(), key=lambda group: min(node.idx for node in group)):
        if len(grouped_nodes) > 1:
            grouped_nodes.sort(key=lambda item: item.idx)
        bus = _make_dc_bus(grouped_nodes, DCBus)
        buses.append(bus)
        bus_dict[bus.idx] = bus
        for node in grouped_nodes:
            node.bus = bus.idx
            node.bus_obj = bus
            node_to_bus[node.idx] = bus

    bus_parent = _make_parent_index(bus.idx for bus in buses)
    get_bus = node_to_bus.get
    union_bus_parent = _union_parent

    def connect_bus_device(dev, require_closed=False):
        pair = live_terminal_pair(dev, require_closed=require_closed)
        if pair is None:
            return
        i_bus = get_bus(pair[0])
        j_bus = get_bus(pair[1])
        if i_bus is not None and j_bus is not None and i_bus.idx != j_bus.idx:
            union_bus_parent(bus_parent, i_bus.idx, j_bus.idx)

    for dev in branches:
        connect_bus_device(dev)
    for dev in zero_branches:
        connect_bus_device(dev)
    for dev in breakers:
        connect_bus_device(dev, require_closed=True)

    root_to_island = {}
    for bus in buses:
        root = _find_parent(bus_parent, bus.idx)
        island = root_to_island.get(root)
        if island is None:
            island = _make_dc_island(len(root_to_island) + 1, DCIsl)
            root_to_island[root] = island
        bus.isl = island.idx
        bus.isl_obj = island
        for node in bus.nodes:
            node.isl = island.idx
            node.isl_obj = island
    network.islands = list(root_to_island.values())
    get_node = node_dict.get

    for gen in generators:
        node = get_node(gen.node)
        gen.node_obj = node
        if node is not None:
            node.generators.append(gen)
        if gen.run_stat == 0:
            continue
        if node is None or node.isl_obj is None:
            continue
        node.isl_obj.gens.append(gen)
        if gen.control_type == "V":
            node.v_gens.append(gen)
            if node.bus_obj is not None:
                node.bus_obj.v_gens.append(gen)
            node.isl_obj.v_gens.append(gen)

    for conv in dcdc_converters:
        i_node = get_node(conv.i_node)
        j_node = get_node(conv.j_node)
        conv.i_node_obj = i_node
        conv.j_node_obj = j_node
        if i_node is not None:
            i_node.dcdc_converters.append(conv)
        if j_node is not None:
            j_node.dcdc_converters.append(conv)
        if conv.run_stat == 0:
            continue
        if i_node is None or j_node is None:
            continue
        if i_node.isl_obj is None or j_node.isl_obj is None:
            continue
        node = i_node
        i_node.isl_obj.dcdc_converters.append(conv)
        j_node.isl_obj.dcdc_converters.append(conv)
        if conv.control_type == "V":
            node.v_dcdcs.append(conv)
            if node.bus_obj is not None:
                node.bus_obj.v_dcdcs.append(conv)
            node.isl_obj.v_dcdcs.append(conv)

    for load in loads:
        node = get_node(load.node)
        load.node_obj = node
        if node is not None:
            node.loads.append(load)
        if load.run_stat == 0:
            continue
        if node is not None and node.isl_obj is not None:
            node.isl_obj.loads.append(load)
    for dev in switches:
        i_node = get_node(dev.i_node)
        j_node = get_node(dev.j_node)
        dev.i_node_obj = i_node
        dev.j_node_obj = j_node
        if i_node is not None:
            i_node.switches.append(dev)
        if j_node is not None:
            j_node.switches.append(dev)
        if i_node is None or j_node is None or dev.run_stat == 0 or dev.status == 0:
            continue
        if i_node.isl_obj and j_node.isl_obj and i_node.isl_obj == j_node.isl_obj:
            i_node.isl_obj.switches.append(dev)
    for dev in branches:
        i_node = get_node(dev.i_node)
        j_node = get_node(dev.j_node)
        dev.i_node_obj = i_node
        dev.j_node_obj = j_node
        if i_node is not None:
            i_node.branches.append(dev)
        if j_node is not None:
            j_node.branches.append(dev)
        if i_node is None or j_node is None or dev.run_stat == 0:
            continue
        if i_node.isl_obj and j_node.isl_obj and i_node.isl_obj == j_node.isl_obj:
            i_node.isl_obj.branches.append(dev)
    for dev in zero_branches:
        i_node = get_node(dev.i_node)
        j_node = get_node(dev.j_node)
        dev.i_node_obj = i_node
        dev.j_node_obj = j_node
        if i_node is not None:
            i_node.zero_branches.append(dev)
        if j_node is not None:
            j_node.zero_branches.append(dev)
        if i_node is None or j_node is None or dev.run_stat == 0:
            continue
        if i_node.isl_obj and j_node.isl_obj and i_node.isl_obj == j_node.isl_obj:
            i_node.isl_obj.zero_branches.append(dev)
    for dev in breakers:
        i_node = get_node(dev.i_node)
        j_node = get_node(dev.j_node)
        dev.i_node_obj = i_node
        dev.j_node_obj = j_node
        if i_node is not None:
            i_node.breakers.append(dev)
        if j_node is not None:
            j_node.breakers.append(dev)
        if i_node is None or j_node is None or dev.run_stat == 0 or dev.status == 0:
            continue
        if i_node.isl_obj and j_node.isl_obj and i_node.isl_obj == j_node.isl_obj:
            i_node.isl_obj.breakers.append(dev)

    for bus in buses:
        if bus.isl_obj is None:
            continue
        bus.isl_obj.buses.append(bus)
        if len(bus.v_gens) + len(bus.v_dcdcs) > 0:
            bus.isl_obj.slack_nodes.append(bus)

    for island in network.islands:
        island.is_alive = len(island.slack_nodes) + len(island.v_dcdcs) >= 1

    for bus in buses:
        bus.is_alive = bus.run_stat == 1 and bus.isl_obj is not None and bus.isl_obj.is_alive
    for node in nodes:
        node.is_alive = node.run_stat == 1 and node.isl_obj is not None and node.isl_obj.is_alive

    network.alive_buses = [bus for bus in buses if bus.is_alive]
    network.alive_nodes = network.alive_buses

    for load in loads:
        node = load.node_obj
        load.is_alive = node is not None and node.isl_obj is not None and load.run_stat == 1 and node.isl_obj.is_alive
    for gen in generators:
        node = gen.node_obj
        gen.is_alive = node is not None and node.isl_obj is not None and gen.run_stat == 1 and node.isl_obj.is_alive
    for dev in branches:
        dev.is_alive = (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.run_stat == 1
            and dev.i_node_obj.is_alive
            and dev.j_node_obj.is_alive
        )
    for dev in zero_branches:
        dev.is_alive = (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.run_stat == 1
            and dev.i_node_obj.is_alive
            and dev.j_node_obj.is_alive
        )
    for dev in breakers:
        dev.is_alive = (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.run_stat == 1
            and dev.status == 1
            and dev.i_node_obj.is_alive
            and dev.j_node_obj.is_alive
        )
    for dev in switches:
        dev.is_alive = (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.status == 1
            and dev.run_stat == 1
            and dev.i_node_obj.is_alive
            and dev.j_node_obj.is_alive
        )
    for conv in dcdc_converters:
        conv.is_alive = (
            conv.i_node_obj is not None
            and conv.j_node_obj is not None
            and conv.run_stat == 1
            and conv.i_node_obj.is_alive
            and conv.j_node_obj.is_alive
        )

    network.alive_branch_by_name = {br.name: br for br in branches if br.is_alive}
    network.alive_generator_by_name = {gen.name: gen for gen in generators if gen.is_alive}
    network.alive_load_by_name = {load.name: load for load in loads if load.is_alive}
    network.alive_zero_branch_by_name = {zbr.name: zbr for zbr in zero_branches if zbr.is_alive}
    network.alive_switch_by_name = {sw.name: sw for sw in switches if sw.is_alive}
    network.alive_break_by_name = {brk.name: brk for brk in breakers if brk.is_alive}
    network.alive_dcdc_by_name = {conv.name: conv for conv in dcdc_converters if conv.is_alive}
    network.alive_zero_branches = _sorted_by_idx(network.alive_zero_branch_by_name.values())
    network.alive_switches = _sorted_by_idx(network.alive_switch_by_name.values())
    network.alive_breakers = _sorted_by_idx(network.alive_break_by_name.values())
    network.alive_generator_order = _sorted_by_idx(network.alive_generator_by_name.values())
    network.alive_load_order = _sorted_by_idx(network.alive_load_by_name.values())
    network.alive_dcdc_order = _sorted_by_idx(network.alive_dcdc_by_name.values())
