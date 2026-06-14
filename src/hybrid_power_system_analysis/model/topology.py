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


@dataclass
class TerminalDeviceTopologyInput:
    i_node_pos: np.ndarray
    j_node_pos: np.ndarray
    run_mask: np.ndarray


@dataclass
class SingleDeviceTopologyInput:
    node_pos: np.ndarray
    run_mask: np.ndarray


@dataclass
class GridTopologyInput:
    node_ids: np.ndarray
    node_run_mask: np.ndarray
    node_lookup: object
    terminals: Dict[str, TerminalDeviceTopologyInput] = field(default_factory=dict)
    singles: Dict[str, SingleDeviceTopologyInput] = field(default_factory=dict)


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


def _terminal_topology_input(
    table,
    i_col: int,
    j_col: int,
    run_col: int,
    node_lookup,
    status_col: Optional[int] = None,
) -> TerminalDeviceTopologyInput:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return TerminalDeviceTopologyInput(_EMPTY_INT, _EMPTY_INT, _EMPTY_BOOL)
    run_mask = rows[:, run_col].astype(np.int64, copy=False) == 1
    if status_col is not None:
        run_mask &= rows[:, status_col].astype(np.int64, copy=False) == 1
    return TerminalDeviceTopologyInput(
        _map_node_positions(rows[:, i_col], node_lookup),
        _map_node_positions(rows[:, j_col], node_lookup),
        run_mask,
    )


def _single_topology_input(table, node_col: int, run_col: int, node_lookup) -> SingleDeviceTopologyInput:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)
    return SingleDeviceTopologyInput(
        _map_node_positions(rows[:, node_col], node_lookup),
        rows[:, run_col].astype(np.int64, copy=False) == 1,
    )


def _compatible_terminal_input(precomputed, count: int) -> bool:
    return (
        precomputed is not None
        and getattr(precomputed, "i_node_pos", _EMPTY_INT).shape[0] == count
        and getattr(precomputed, "j_node_pos", _EMPTY_INT).shape[0] == count
        and getattr(precomputed, "run_mask", _EMPTY_BOOL).shape[0] == count
    )


def _compatible_single_input(precomputed, count: int) -> bool:
    return (
        precomputed is not None
        and getattr(precomputed, "node_pos", _EMPTY_INT).shape[0] == count
        and getattr(precomputed, "run_mask", _EMPTY_BOOL).shape[0] == count
    )


def _compatible_grid_topology_input(precomputed, node_count: int) -> bool:
    return (
        precomputed is not None
        and getattr(precomputed, "node_ids", _EMPTY_INT).shape[0] == node_count
        and getattr(precomputed, "node_run_mask", _EMPTY_BOOL).shape[0] == node_count
        and getattr(precomputed, "node_lookup", None) is not None
    )


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
    precomputed: Optional[TerminalDeviceTopologyInput] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _EMPTY_INT, _EMPTY_INT, _EMPTY_BOOL
    if _compatible_terminal_input(precomputed, count):
        i_pos = precomputed.i_node_pos
        j_pos = precomputed.j_node_pos
        run_mask = precomputed.run_mask
    else:
        i_pos = _map_node_positions(rows[:, i_col], node_lookup)
        j_pos = _map_node_positions(rows[:, j_col], node_lookup)
        run_mask = rows[:, run_col].astype(np.int64, copy=False) == 1
        if status_col is not None:
            run_mask &= rows[:, status_col].astype(np.int64, copy=False) == 1
    active = (
        run_mask
        & (i_pos >= 0)
        & (j_pos >= 0)
        & (i_pos != j_pos)
    )
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
        bus.acac_converters = ()
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
    edge_specs: Sequence[Tuple],
    node_lookup,
) -> Sequence[Sequence[int]]:
    n_nodes = int(node_ids.size)
    running_positions = np.flatnonzero(node_run_mask)
    if running_positions.size == 0:
        return []

    left_chunks = []
    right_chunks = []
    for edge_spec in edge_specs:
        table, i_col, j_col, run_col, status_col = edge_spec[:5]
        precomputed = edge_spec[5] if len(edge_spec) > 5 else None
        i_pos, j_pos, active = _active_terminal_positions(
            table,
            i_col,
            j_col,
            run_col,
            node_lookup,
            node_run_mask,
            status_col=status_col,
            precomputed=precomputed,
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


def _component_position_group_arrays(
    node_ids: np.ndarray,
    node_run_mask: np.ndarray,
    edge_specs: Sequence[Tuple],
    node_lookup,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = int(node_ids.size)
    item_to_group_pos = np.full(n_nodes, -1, dtype=np.int32)
    running_positions = np.flatnonzero(node_run_mask).astype(np.int32, copy=False)
    if running_positions.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            _EMPTY_INT,
            item_to_group_pos,
        )

    left_chunks = []
    right_chunks = []
    for edge_spec in edge_specs:
        table, i_col, j_col, run_col, status_col = edge_spec[:5]
        precomputed = edge_spec[5] if len(edge_spec) > 5 else None
        i_pos, j_pos, active = _active_terminal_positions(
            table,
            i_col,
            j_col,
            run_col,
            node_lookup,
            node_run_mask,
            status_col=status_col,
            precomputed=precomputed,
        )
        if active.size and np.any(active):
            left_chunks.append(i_pos[active])
            right_chunks.append(j_pos[active])

    if not left_chunks:
        order = np.argsort(node_ids[running_positions], kind="stable")
        indices = running_positions[order].astype(np.int32, copy=False)
        group_ids = node_ids[indices].astype(np.int32, copy=False)
        offsets = np.arange(indices.size + 1, dtype=np.int32)
        item_to_group_pos[indices] = np.arange(indices.size, dtype=np.int32)
        return group_ids, offsets, indices, item_to_group_pos

    left = np.concatenate(left_chunks).astype(np.int32, copy=False)
    right = np.concatenate(right_chunks).astype(np.int32, copy=False)
    graph = coo_matrix((np.ones(left.size, dtype=np.int8), (left, right)), shape=(n_nodes, n_nodes))
    _count, labels = connected_components(graph, directed=False, return_labels=True)

    running_labels = labels[running_positions]
    order = np.lexsort((node_ids[running_positions], running_labels))
    sorted_positions = running_positions[order].astype(np.int32, copy=False)
    sorted_labels = running_labels[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]).astype(np.int64) + 1,
            np.asarray([sorted_positions.size], dtype=np.int64),
        )
    )
    counts = np.diff(boundaries).astype(np.int32, copy=False)
    first_positions = sorted_positions[boundaries[:-1]]
    group_ids_unsorted = node_ids[first_positions].astype(np.int32, copy=False)
    group_order = np.argsort(group_ids_unsorted, kind="stable")

    old_group_for_sorted = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    old_to_new = np.empty(counts.size, dtype=np.int32)
    old_to_new[group_order] = np.arange(counts.size, dtype=np.int32)
    new_group_for_sorted = old_to_new[old_group_for_sorted]
    index_order = np.argsort(new_group_for_sorted, kind="stable")
    indices = sorted_positions[index_order].astype(np.int32, copy=False)
    item_to_group_pos[indices] = new_group_for_sorted[index_order]

    offsets = np.empty(counts.size + 1, dtype=np.int32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts[group_order], dtype=np.int32)
    return group_ids_unsorted[group_order], offsets, indices, item_to_group_pos


def _build_base_topology_arrays(
    node_ids: np.ndarray,
    node_run_mask: np.ndarray,
    bus_edge_specs: Sequence[Tuple],
    island_edge_specs: Sequence[Tuple],
    node_lookup=None,
) -> GridTopologyArrays:
    node_ids = np.asarray(node_ids, dtype=np.int32)
    node_run_mask = np.asarray(node_run_mask, dtype=bool)
    n_nodes = int(node_ids.size)
    if node_lookup is None:
        node_lookup = _make_node_pos_lookup(node_ids)
    bus_ids, bus_node_offsets, bus_node_indices, node_to_bus_pos = _component_position_group_arrays(
        node_ids,
        node_run_mask,
        bus_edge_specs,
        node_lookup,
    )
    _island_node_ids, _island_node_offsets, _island_node_indices, node_to_island_pos = _component_position_group_arrays(
        node_ids,
        node_run_mask,
        island_edge_specs,
        node_lookup,
    )

    bus_to_island_pos = np.full(bus_ids.size, -1, dtype=np.int32)
    if bus_ids.size:
        bus_first_node_pos = bus_node_indices[bus_node_offsets[:-1]]
        bus_to_island_pos[:] = node_to_island_pos[bus_first_node_pos]

    island_count = int(_island_node_ids.size)
    island_ids = np.arange(1, island_count + 1, dtype=np.int32)
    valid_bus = bus_to_island_pos >= 0
    if np.any(valid_bus):
        bus_positions = np.flatnonzero(valid_bus).astype(np.int32, copy=False)
        island_for_bus = bus_to_island_pos[bus_positions]
        order = np.lexsort((bus_positions, island_for_bus))
        island_bus_indices = bus_positions[order].astype(np.int32, copy=False)
        counts = np.bincount(island_for_bus, minlength=island_count).astype(np.int32, copy=False)
        island_bus_offsets = np.empty(island_count + 1, dtype=np.int32)
        island_bus_offsets[0] = 0
        island_bus_offsets[1:] = np.cumsum(counts, dtype=np.int32)
    else:
        island_bus_offsets = np.zeros(island_count + 1, dtype=np.int32)
        island_bus_indices = _EMPTY_INT

    false_islands = np.zeros(island_count, dtype=bool)
    false_buses = np.zeros(bus_ids.size, dtype=bool)
    false_nodes = np.zeros(n_nodes, dtype=bool)
    reference = np.full(island_count, -1, dtype=np.int32)
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
    precomputed: Optional[SingleDeviceTopologyInput] = None,
) -> SingleDeviceTopologyArrays:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _empty_single_device(0)
    if _compatible_single_input(precomputed, count):
        node_pos = precomputed.node_pos
        run_mask = precomputed.run_mask
    else:
        node_pos = _map_node_positions(rows[:, node_col], node_lookup)
        run_mask = rows[:, run_col].astype(np.int64, copy=False) == 1
    bus_pos = np.full(count, -1, dtype=np.int32)
    island_pos = np.full(count, -1, dtype=np.int32)
    valid = node_pos >= 0
    if np.any(valid):
        bus_pos[valid] = topology.node_to_bus_pos[node_pos[valid]]
        island_pos[valid] = topology.node_to_island_pos[node_pos[valid]]
    node_alive = np.zeros(count, dtype=bool)
    if np.any(valid):
        node_alive[valid] = topology.node_alive_mask[node_pos[valid]]
    alive = run_mask & valid & (island_pos >= 0) & node_alive
    return SingleDeviceTopologyArrays(node_pos, bus_pos, island_pos, alive)


def _terminal_device_arrays(
    table,
    i_col: int,
    j_col: int,
    run_col: int,
    node_lookup,
    topology: GridTopologyArrays,
    status_col: Optional[int] = None,
    precomputed: Optional[TerminalDeviceTopologyInput] = None,
) -> TerminalDeviceTopologyArrays:
    rows = np.asarray(table)
    count = int(rows.shape[0]) if rows.size else 0
    if count == 0:
        return _empty_terminal_device(0)
    if _compatible_terminal_input(precomputed, count):
        i_node_pos = precomputed.i_node_pos
        j_node_pos = precomputed.j_node_pos
        run_mask = precomputed.run_mask
    else:
        i_node_pos = _map_node_positions(rows[:, i_col], node_lookup)
        j_node_pos = _map_node_positions(rows[:, j_col], node_lookup)
        run_mask = rows[:, run_col].astype(np.int64, copy=False) == 1
        if status_col is not None:
            run_mask &= rows[:, status_col].astype(np.int64, copy=False) == 1
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


def build_ac_topology_input_ppc(ppc: Dict) -> GridTopologyInput:
    """Precompute AC PPC node positions and run/status masks for topology."""
    try:
        from .ac_array_model import (
            BREAK_COLS,
            ACAC_COLS,
            BRANCH_COLS,
            BUS_COLS,
            GEN_COLS,
            LOAD_COLS,
            SHUNT_COLS,
            SWITCH_COLS,
            TRANSFORMER_COLS,
            ZERO_BRANCH_COLS,
            _empty,
        )
    except ImportError:  # pragma: no cover - top-level module import path
        from ac_array_model import (
            BREAK_COLS,
            ACAC_COLS,
            BRANCH_COLS,
            BUS_COLS,
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
    acac = np.asarray(ppc.get("acac", _empty(len(ACAC_COLS))), dtype=np.float64)
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int32, copy=False)
    node_run_mask = bus[:, BUS_COLS["run_stat"]].astype(np.int64, copy=False) == 1
    node_lookup = _make_node_pos_lookup(node_ids)
    terminals = {
        "branch": _terminal_topology_input(
            branch,
            BRANCH_COLS["i_node"],
            BRANCH_COLS["j_node"],
            BRANCH_COLS["run_stat"],
            node_lookup,
        ),
        "transformer": _terminal_topology_input(
            transformer,
            TRANSFORMER_COLS["i_node"],
            TRANSFORMER_COLS["j_node"],
            TRANSFORMER_COLS["run_stat"],
            node_lookup,
        ),
        "zero_branch": _terminal_topology_input(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
        ),
        "switch": _terminal_topology_input(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            status_col=SWITCH_COLS["status"],
        ),
        "break": _terminal_topology_input(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            status_col=BREAK_COLS["status"],
        ),
        "acac": _terminal_topology_input(
            acac,
            ACAC_COLS["i_node"],
            ACAC_COLS["j_node"],
            ACAC_COLS["run_stat"],
            node_lookup,
        ),
    }
    singles = {
        "gen": _single_topology_input(gen, GEN_COLS["node"], GEN_COLS["run_stat"], node_lookup),
        "load": _single_topology_input(load, LOAD_COLS["node"], LOAD_COLS["run_stat"], node_lookup),
        "shunt": _single_topology_input(shunt, SHUNT_COLS["node"], SHUNT_COLS["run_stat"], node_lookup),
    }
    return GridTopologyInput(node_ids, node_run_mask, node_lookup, terminals=terminals, singles=singles)


def prepare_ac_topology_ppc(ppc: Dict) -> GridTopologyArrays:
    """Build AC bus/island topology directly from ``ac_ppc_v1`` arrays."""
    try:
        from .ac_array_model import (
            BREAK_COLS,
            ACAC_COLS,
            BRANCH_COLS,
            CTRL_SLACK,
            GEN_COLS,
            LOAD_COLS,
            SHUNT_COLS,
            SWITCH_COLS,
            TRANSFORMER_COLS,
            ZERO_BRANCH_COLS,
            _empty,
        )
    except ImportError:  # pragma: no cover - top-level module import path
        from ac_array_model import (
            BREAK_COLS,
            ACAC_COLS,
            BRANCH_COLS,
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
    acac = np.asarray(ppc.get("acac", _empty(len(ACAC_COLS))), dtype=np.float64)
    topology_input = ppc.get("_topology_input")
    if not _compatible_grid_topology_input(topology_input, bus.shape[0] if bus.size else 0):
        topology_input = build_ac_topology_input_ppc(ppc)
        ppc["_topology_input"] = topology_input
    node_ids = topology_input.node_ids
    node_run_mask = topology_input.node_run_mask
    terminals = topology_input.terminals
    singles = topology_input.singles
    node_lookup = topology_input.node_lookup
    topology = _build_base_topology_arrays(
        node_ids,
        node_run_mask,
        bus_edge_specs=(
            (
                switch,
                SWITCH_COLS["i_node"],
                SWITCH_COLS["j_node"],
                SWITCH_COLS["run_stat"],
                SWITCH_COLS["status"],
                terminals.get("switch"),
            ),
        ),
        island_edge_specs=(
            (
                switch,
                SWITCH_COLS["i_node"],
                SWITCH_COLS["j_node"],
                SWITCH_COLS["run_stat"],
                SWITCH_COLS["status"],
                terminals.get("switch"),
            ),
            (branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], None, terminals.get("branch")),
            (
                transformer,
                TRANSFORMER_COLS["i_node"],
                TRANSFORMER_COLS["j_node"],
                TRANSFORMER_COLS["run_stat"],
                None,
                terminals.get("transformer"),
            ),
            (
                zero_branch,
                ZERO_BRANCH_COLS["i_node"],
                ZERO_BRANCH_COLS["j_node"],
                ZERO_BRANCH_COLS["run_stat"],
                None,
                terminals.get("zero_branch"),
            ),
            (
                breaker,
                BREAK_COLS["i_node"],
                BREAK_COLS["j_node"],
                BREAK_COLS["run_stat"],
                BREAK_COLS["status"],
                terminals.get("break"),
            ),
        ),
        node_lookup=node_lookup,
    )
    gen_input = singles.get("gen")
    gen_count = gen.shape[0] if gen.size else 0
    if _compatible_single_input(gen_input, gen_count):
        gen_nodes = gen_input.node_pos
    elif gen.size:
        gen_nodes = _map_node_positions(gen[:, GEN_COLS["node"]], node_lookup)
    else:
        gen_nodes = _EMPTY_INT
    if gen_nodes.size:
        valid_gen = gen_nodes >= 0
        gen_islands = np.full(gen_nodes.shape, -1, dtype=np.int32)
        gen_buses = np.full(gen_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_gen):
            gen_islands[valid_gen] = topology.node_to_island_pos[gen_nodes[valid_gen]]
            gen_buses[valid_gen] = topology.node_to_bus_pos[gen_nodes[valid_gen]]
        if _compatible_single_input(gen_input, gen.shape[0]):
            gen_run_mask = gen_input.run_mask
        else:
            gen_run_mask = gen[:, GEN_COLS["run_stat"]].astype(np.int64, copy=False) == 1
        slack_mask = (
            gen_run_mask
            & valid_gen
            & (gen_islands >= 0)
            & (gen[:, GEN_COLS["control_type"]].astype(np.int64, copy=False) == CTRL_SLACK)
        )
        topology.island_alive_mask[gen_islands[slack_mask]] = True
        for island_pos, bus_pos in zip(gen_islands[slack_mask], gen_buses[slack_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    external_ref_nodes = np.asarray(ppc.get("_external_angle_reference_node_ids", _EMPTY_INT), dtype=np.int64)
    if external_ref_nodes.size:
        external_ref_pos = _map_node_positions(external_ref_nodes, node_lookup)
        valid_ref = external_ref_pos >= 0
        if np.any(valid_ref):
            ref_node_pos = external_ref_pos[valid_ref]
            valid_ref_indices = np.flatnonzero(valid_ref)
            valid_ref[valid_ref_indices] &= topology.node_run_mask[ref_node_pos]
        if np.any(valid_ref):
            ref_node_pos = external_ref_pos[valid_ref]
            ref_islands = topology.node_to_island_pos[ref_node_pos]
            ref_buses = topology.node_to_bus_pos[ref_node_pos]
            valid_island = ref_islands >= 0
            topology.island_alive_mask[ref_islands[valid_island]] = True
            for island_pos, bus_pos in zip(ref_islands[valid_island], ref_buses[valid_island]):
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
        "branch": _terminal_device_arrays(
            branch,
            BRANCH_COLS["i_node"],
            BRANCH_COLS["j_node"],
            BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("branch"),
        ),
        "transformer": _terminal_device_arrays(
            transformer,
            TRANSFORMER_COLS["i_node"],
            TRANSFORMER_COLS["j_node"],
            TRANSFORMER_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("transformer"),
        ),
        "zero_branch": _terminal_device_arrays(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("zero_branch"),
        ),
        "switch": _terminal_device_arrays(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=SWITCH_COLS["status"],
            precomputed=terminals.get("switch"),
        ),
        "break": _terminal_device_arrays(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=BREAK_COLS["status"],
            precomputed=terminals.get("break"),
        ),
        "acac": _terminal_device_arrays(
            acac,
            ACAC_COLS["i_node"],
            ACAC_COLS["j_node"],
            ACAC_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("acac"),
        ),
        "gen": _single_device_arrays(
            gen,
            GEN_COLS["node"],
            GEN_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=singles.get("gen"),
        ),
        "load": _single_device_arrays(
            load,
            LOAD_COLS["node"],
            LOAD_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=singles.get("load"),
        ),
        "shunt": _single_device_arrays(
            shunt,
            SHUNT_COLS["node"],
            SHUNT_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=singles.get("shunt"),
        ),
    }
    return topology


def build_dc_topology_input_ppc(ppc: Dict) -> GridTopologyInput:
    """Precompute DC PPC node positions and run/status masks for topology."""
    try:
        from .dc_array_model import (
            BREAK_COLS,
            BRANCH_COLS,
            BUS_COLS,
            DCDC_COLS,
            GEN_COLS,
            LOAD_COLS,
            SWITCH_COLS,
            ZERO_BRANCH_COLS,
            _empty,
        )
    except ImportError:  # pragma: no cover - top-level module import path
        from dc_array_model import (
            BREAK_COLS,
            BRANCH_COLS,
            BUS_COLS,
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
    node_lookup = _make_node_pos_lookup(node_ids)
    terminals = {
        "branch": _terminal_topology_input(
            branch,
            BRANCH_COLS["i_node"],
            BRANCH_COLS["j_node"],
            BRANCH_COLS["run_stat"],
            node_lookup,
        ),
        "zero_branch": _terminal_topology_input(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
        ),
        "switch": _terminal_topology_input(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            status_col=SWITCH_COLS["status"],
        ),
        "break": _terminal_topology_input(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            status_col=BREAK_COLS["status"],
        ),
        "dcdc": _terminal_topology_input(
            dcdc,
            DCDC_COLS["i_node"],
            DCDC_COLS["j_node"],
            DCDC_COLS["run_stat"],
            node_lookup,
        ),
    }
    singles = {
        "gen": _single_topology_input(gen, GEN_COLS["node"], GEN_COLS["run_stat"], node_lookup),
        "load": _single_topology_input(load, LOAD_COLS["node"], LOAD_COLS["run_stat"], node_lookup),
    }
    return GridTopologyInput(node_ids, node_run_mask, node_lookup, terminals=terminals, singles=singles)


def prepare_dc_topology_ppc(ppc: Dict) -> GridTopologyArrays:
    """Build DC bus/island topology directly from ``dc_ppc_v1`` arrays."""
    try:
        from .dc_array_model import (
            BREAK_COLS,
            BRANCH_COLS,
            CTRL_SLACK,
            CTRL_V,
            DCDC_COLS,
            GEN_COLS,
            LOAD_COLS,
            SWITCH_COLS,
            ZERO_BRANCH_COLS,
            _empty,
        )
    except ImportError:  # pragma: no cover - top-level module import path
        from dc_array_model import (
            BREAK_COLS,
            BRANCH_COLS,
            CTRL_SLACK,
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
    topology_input = ppc.get("_topology_input")
    if not _compatible_grid_topology_input(topology_input, bus.shape[0] if bus.size else 0):
        topology_input = build_dc_topology_input_ppc(ppc)
        ppc["_topology_input"] = topology_input
    node_ids = topology_input.node_ids
    node_run_mask = topology_input.node_run_mask
    terminals = topology_input.terminals
    singles = topology_input.singles
    node_lookup = topology_input.node_lookup
    topology = _build_base_topology_arrays(
        node_ids,
        node_run_mask,
        bus_edge_specs=(
            (
                switch,
                SWITCH_COLS["i_node"],
                SWITCH_COLS["j_node"],
                SWITCH_COLS["run_stat"],
                SWITCH_COLS["status"],
                terminals.get("switch"),
            ),
        ),
        island_edge_specs=(
            (
                switch,
                SWITCH_COLS["i_node"],
                SWITCH_COLS["j_node"],
                SWITCH_COLS["run_stat"],
                SWITCH_COLS["status"],
                terminals.get("switch"),
            ),
            (branch, BRANCH_COLS["i_node"], BRANCH_COLS["j_node"], BRANCH_COLS["run_stat"], None, terminals.get("branch")),
            (
                zero_branch,
                ZERO_BRANCH_COLS["i_node"],
                ZERO_BRANCH_COLS["j_node"],
                ZERO_BRANCH_COLS["run_stat"],
                None,
                terminals.get("zero_branch"),
            ),
            (
                breaker,
                BREAK_COLS["i_node"],
                BREAK_COLS["j_node"],
                BREAK_COLS["run_stat"],
                BREAK_COLS["status"],
                terminals.get("break"),
            ),
        ),
        node_lookup=node_lookup,
    )
    gen_input = singles.get("gen")
    gen_count = gen.shape[0] if gen.size else 0
    if _compatible_single_input(gen_input, gen_count):
        gen_nodes = gen_input.node_pos
    elif gen.size:
        gen_nodes = _map_node_positions(gen[:, GEN_COLS["node"]], node_lookup)
    else:
        gen_nodes = _EMPTY_INT
    if gen_nodes.size:
        valid_gen = gen_nodes >= 0
        gen_islands = np.full(gen_nodes.shape, -1, dtype=np.int32)
        gen_buses = np.full(gen_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_gen):
            gen_islands[valid_gen] = topology.node_to_island_pos[gen_nodes[valid_gen]]
            gen_buses[valid_gen] = topology.node_to_bus_pos[gen_nodes[valid_gen]]
        if _compatible_single_input(gen_input, gen.shape[0]):
            gen_run_mask = gen_input.run_mask
        else:
            gen_run_mask = gen[:, GEN_COLS["run_stat"]].astype(np.int64, copy=False) == 1
        v_mask = (
            gen_run_mask
            & valid_gen
            & (gen_islands >= 0)
            & (gen[:, GEN_COLS["control_type"]].astype(np.int64, copy=False) == CTRL_V)
        )
        topology.island_alive_mask[gen_islands[v_mask]] = True
        for island_pos, bus_pos in zip(gen_islands[v_mask], gen_buses[v_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    dcdc_input = terminals.get("dcdc")
    dcdc_count = dcdc.shape[0] if dcdc.size else 0
    if _compatible_terminal_input(dcdc_input, dcdc_count):
        dcdc_nodes = dcdc_input.i_node_pos
    elif dcdc.size:
        dcdc_nodes = _map_node_positions(dcdc[:, DCDC_COLS["i_node"]], node_lookup)
    else:
        dcdc_nodes = _EMPTY_INT
    if dcdc_nodes.size:
        if _compatible_terminal_input(dcdc_input, dcdc.shape[0]):
            dcdc_j_nodes = dcdc_input.j_node_pos
            dcdc_run_mask = dcdc_input.run_mask
        else:
            dcdc_j_nodes = _map_node_positions(dcdc[:, DCDC_COLS["j_node"]], node_lookup)
            dcdc_run_mask = dcdc[:, DCDC_COLS["run_stat"]].astype(np.int64, copy=False) == 1
        valid_dcdc = (dcdc_nodes >= 0) & (dcdc_j_nodes >= 0)
        dcdc_i_islands = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        dcdc_j_islands = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        dcdc_i_buses = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        dcdc_j_buses = np.full(dcdc_nodes.shape, -1, dtype=np.int32)
        if np.any(valid_dcdc):
            dcdc_i_islands[valid_dcdc] = topology.node_to_island_pos[dcdc_nodes[valid_dcdc]]
            dcdc_j_islands[valid_dcdc] = topology.node_to_island_pos[dcdc_j_nodes[valid_dcdc]]
            dcdc_i_buses[valid_dcdc] = topology.node_to_bus_pos[dcdc_nodes[valid_dcdc]]
            dcdc_j_buses[valid_dcdc] = topology.node_to_bus_pos[dcdc_j_nodes[valid_dcdc]]
        i_control = dcdc[:, DCDC_COLS["i_control_type"]].astype(np.int64, copy=False)
        j_control = dcdc[:, DCDC_COLS["j_control_type"]].astype(np.int64, copy=False)
        valid_i_island = dcdc_i_islands >= 0
        valid_j_island = dcdc_j_islands >= 0
        i_v_mask = (
            dcdc_run_mask
            & valid_dcdc
            & valid_i_island
            & (i_control == CTRL_V)
            & (j_control == CTRL_SLACK)
        )
        j_v_mask = (
            dcdc_run_mask
            & valid_dcdc
            & valid_j_island
            & (j_control == CTRL_V)
            & (i_control == CTRL_SLACK)
        )
        linked_v_mask = i_v_mask | j_v_mask
        linked_i_alive = linked_v_mask & valid_i_island
        linked_j_alive = linked_v_mask & valid_j_island
        topology.island_alive_mask[dcdc_i_islands[linked_i_alive]] = True
        topology.island_alive_mask[dcdc_j_islands[linked_j_alive]] = True
        for island_pos, bus_pos in zip(dcdc_i_islands[i_v_mask], dcdc_i_buses[i_v_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)
        for island_pos, bus_pos in zip(dcdc_j_islands[j_v_mask], dcdc_j_buses[j_v_mask]):
            _mark_reference_bus(topology.island_reference_bus_pos, int(island_pos), int(bus_pos), topology.bus_ids)

    external_ref_nodes = np.asarray(ppc.get("_external_voltage_reference_node_ids", _EMPTY_INT), dtype=np.int64)
    if external_ref_nodes.size:
        external_ref_pos = _map_node_positions(external_ref_nodes, node_lookup)
        valid_ref = external_ref_pos >= 0
        if np.any(valid_ref):
            ref_node_pos = external_ref_pos[valid_ref]
            valid_ref_indices = np.flatnonzero(valid_ref)
            valid_ref[valid_ref_indices] &= topology.node_run_mask[ref_node_pos]
        if np.any(valid_ref):
            ref_node_pos = external_ref_pos[valid_ref]
            ref_islands = topology.node_to_island_pos[ref_node_pos]
            ref_buses = topology.node_to_bus_pos[ref_node_pos]
            valid_island = ref_islands >= 0
            topology.island_alive_mask[ref_islands[valid_island]] = True
            for island_pos, bus_pos in zip(ref_islands[valid_island], ref_buses[valid_island]):
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
        "branch": _terminal_device_arrays(
            branch,
            BRANCH_COLS["i_node"],
            BRANCH_COLS["j_node"],
            BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("branch"),
        ),
        "zero_branch": _terminal_device_arrays(
            zero_branch,
            ZERO_BRANCH_COLS["i_node"],
            ZERO_BRANCH_COLS["j_node"],
            ZERO_BRANCH_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("zero_branch"),
        ),
        "switch": _terminal_device_arrays(
            switch,
            SWITCH_COLS["i_node"],
            SWITCH_COLS["j_node"],
            SWITCH_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=SWITCH_COLS["status"],
            precomputed=terminals.get("switch"),
        ),
        "break": _terminal_device_arrays(
            breaker,
            BREAK_COLS["i_node"],
            BREAK_COLS["j_node"],
            BREAK_COLS["run_stat"],
            node_lookup,
            topology,
            status_col=BREAK_COLS["status"],
            precomputed=terminals.get("break"),
        ),
        "dcdc": _terminal_device_arrays(
            dcdc,
            DCDC_COLS["i_node"],
            DCDC_COLS["j_node"],
            DCDC_COLS["run_stat"],
            node_lookup,
            topology,
            precomputed=terminals.get("dcdc"),
        ),
        "gen": _single_device_arrays(gen, GEN_COLS["node"], GEN_COLS["run_stat"], node_lookup, topology, precomputed=singles.get("gen")),
        "load": _single_device_arrays(load, LOAD_COLS["node"], LOAD_COLS["run_stat"], node_lookup, topology, precomputed=singles.get("load")),
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


def _apply_topology_alive_flags(devices, topology: GridTopologyArrays, key: str) -> None:
    alive = _topology_device_mask(topology, key, len(devices))
    for pos, dev in enumerate(devices):
        dev.is_alive = bool(alive[pos]) if pos < alive.size else False


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

    # Pre-materialize numpy arrays into Python lists once; the loops below
    # iterate by Python index and per-iteration numpy scalar access (with
    # boxing) was a measurable cost for large grids.
    node_alive_mask = topology.node_alive_mask
    node_alive_list = node_alive_mask.tolist() if node_alive_mask.size else []
    n_node_alive = len(node_alive_list)
    for node_pos, node in enumerate(nodes):
        node.isl = 0
        node.isl_obj = None
        node.bus = None
        node.bus_obj = None
        node.is_alive = node_alive_list[node_pos] if node_pos < n_node_alive else False

    island_ids_list = topology.island_ids.tolist()
    island_alive_list = topology.island_alive_mask.tolist()
    islands = []
    append_island = islands.append
    for pos in range(len(island_ids_list)):
        is_alive = bool(island_alive_list[pos])
        island = island_factory(int(island_ids_list[pos]), is_alive)
        island.is_alive = is_alive
        append_island(island)
    network.islands = islands

    bus_ids_list = topology.bus_ids.tolist()
    bus_offsets_list = topology.bus_node_offsets.tolist()
    bus_node_indices_list = topology.bus_node_indices.tolist()
    bus_alive_list = topology.bus_alive_mask.tolist()
    bus_to_island_list = topology.bus_to_island_pos.tolist()
    buses = []
    network.bus_dict = {}
    network.node_to_bus = {}
    node_to_bus = network.node_to_bus
    bus_dict = network.bus_dict
    for bus_pos in range(len(bus_ids_list)):
        bus_id = int(bus_ids_list[bus_pos])
        start = bus_offsets_list[bus_pos]
        end = bus_offsets_list[bus_pos + 1]
        grouped_nodes = [nodes[node_pos] for node_pos in bus_node_indices_list[start:end]]
        bus = _make_compact_bus(bus_cls, bus_id, grouped_nodes) if compact else bus_cls(bus_id, grouped_nodes)
        bus.is_alive = bool(bus_alive_list[bus_pos])
        buses.append(bus)
        if not compact:
            bus_dict[bus_id] = bus
        island_pos = bus_to_island_list[bus_pos]
        island = islands[island_pos] if island_pos >= 0 else None
        if island is not None:
            bus.isl = int(island.idx)
            bus.isl_obj = island
            island.buses.append(bus)
        for node in grouped_nodes:
            node.bus = bus_id
            node.bus_obj = bus
            if not compact:
                node_to_bus[int(node.idx)] = bus
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
    populate_device_links: bool = True,
) -> None:
    """Populate AC object topology fields from precomputed ppc topology arrays.

    ``compact`` keeps only the object links required by SE compatibility code.
    The full reverse device lists remain the default for LF/full-result callers.
    ``populate_device_links=False`` skips device ``*_node_obj`` backfill and
    only refreshes device alive flags, which keeps array-only callers on the
    cheap topology path.
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

    def finalize_alive_maps() -> None:
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

    if compact:
        network.branch_dict = {}
        network.transformer_dict = {}
        network.generator_dict = {}
        network.load_dict = {}
        network.shunt_compensator_dict = {}
        network.zero_branch_dict = {}
        network.zero_branche_dict = network.zero_branch_dict
        network.switch_dict = {}
        network.break_dict = {}
        network.branche_dict = network.branch_dict
    else:
        network.branch_dict = {int(dev.idx): dev for dev in branches}
        network.branche_dict = network.branch_dict
        network.transformer_dict = {int(dev.idx): dev for dev in transformers}
        network.generator_dict = {int(dev.idx): dev for dev in generators}
        network.load_dict = {int(dev.idx): dev for dev in loads}
        network.shunt_compensator_dict = {int(dev.idx): dev for dev in shunts}
        network.zero_branch_dict = {int(dev.idx): dev for dev in zero_branches}
        network.zero_branche_dict = network.zero_branch_dict
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

    if not populate_device_links:
        _apply_topology_alive_flags(generators, topology, "gen")
        _apply_topology_alive_flags(loads, topology, "load")
        _apply_topology_alive_flags(shunts, topology, "shunt")
        _apply_topology_alive_flags(branches, topology, "branch")
        _apply_topology_alive_flags(transformers, topology, "transformer")
        _apply_topology_alive_flags(zero_branches, topology, "zero_branch")
        _apply_topology_alive_flags(switches, topology, "switch")
        _apply_topology_alive_flags(breakers, topology, "break")
        finalize_alive_maps()
        return

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

    finalize_alive_maps()


def apply_dc_topology_arrays(
    network,
    topology: GridTopologyArrays,
    *,
    compact: bool = False,
    build_alive_maps: bool = True,
    populate_device_links: bool = True,
) -> None:
    """Populate DC object topology fields from precomputed ppc topology arrays.

    ``compact`` keeps only the object links required by SE compatibility code.
    The full reverse device lists remain the default for LF/full-result callers.
    ``populate_device_links=False`` skips device ``*_node_obj`` backfill and
    only refreshes device alive flags for array-only callers.
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

    def finalize_alive_maps() -> None:
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

    if not populate_device_links:
        _apply_topology_alive_flags(generators, topology, "gen")
        _apply_topology_alive_flags(loads, topology, "load")
        _apply_topology_alive_flags(dcdc_converters, topology, "dcdc")
        _apply_topology_alive_flags(branches, topology, "branch")
        _apply_topology_alive_flags(zero_branches, topology, "zero_branch")
        _apply_topology_alive_flags(switches, topology, "switch")
        _apply_topology_alive_flags(breakers, topology, "break")
        finalize_alive_maps()
        return

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
        if getattr(conv, "i_control_type", getattr(conv, "control_type", "")) in ("V", "CTRL_V"):
            v_node = i_node
        elif getattr(conv, "j_control_type", "") in ("V", "CTRL_V"):
            v_node = j_node
        else:
            v_node = None
        if v_node is not None and v_node.isl_obj is not None:
            v_node.v_set = float(getattr(conv, "v_set", v_node.v_set))
            if v_node.bus_obj is not None:
                v_node.bus_obj.v_set = float(getattr(conv, "v_set", v_node.bus_obj.v_set))
            slack_bus = v_node.bus_obj or v_node
            if slack_bus not in v_node.isl_obj.slack_nodes:
                v_node.isl_obj.slack_nodes.append(slack_bus)
            if compact:
                v_node.is_slack = True
                if v_node.bus_obj is not None:
                    v_node.bus_obj.is_slack = True
            if not compact:
                v_node.v_dcdcs.append(conv)
                if v_node.bus_obj is not None:
                    v_node.bus_obj.v_dcdcs.append(conv)
                v_node.isl_obj.v_dcdcs.append(conv)

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

    finalize_alive_maps()


def _make_ac_bus(grouped_nodes, bus_cls):
    return bus_cls(grouped_nodes[0].idx, grouped_nodes)


def _make_ac_island(idx, island_cls):
    island = island_cls(idx, False)
    if not hasattr(island, "transformers"):
        island.transformers = []
    if not hasattr(island, "shunt_compensators"):
        island.shunt_compensators = []
    if not hasattr(island, "acac_converters"):
        island.acac_converters = []
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
    acac_converters = _device_seq(network, "acac_converters")

    node_dict = {node.idx: node for node in nodes}
    network.node_dict = node_dict
    network.branch_dict = {dev.idx: dev for dev in branches}
    network.transformer_dict = {dev.idx: dev for dev in transformers}
    network.generator_dict = {dev.idx: dev for dev in generators}
    network.load_dict = {dev.idx: dev for dev in loads}
    network.shunt_compensator_dict = {dev.idx: dev for dev in shunts}
    network.zero_branch_dict = {dev.idx: dev for dev in zero_branches}
    network.zero_branche_dict = network.zero_branch_dict
    network.switch_dict = {dev.idx: dev for dev in switches}
    network.break_dict = {dev.idx: dev for dev in breakers}
    network.acac_converter_dict = {dev.idx: dev for dev in acac_converters}
    network.branche_dict = network.branch_dict

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
        node.acac_converters = []
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
    for dev in acac_converters:
        finalize_branch_like(dev, "acac_converters")

    network.alive_nodes = [bus for bus in buses if bus.is_alive]
    network.alive_buses = network.alive_nodes
    network.alive_branch_by_name = {br.name: br for br in branches if br.is_alive}
    network.alive_transformer_by_name = {tr.name: tr for tr in transformers if tr.is_alive}
    network.alive_generator_by_name = {gen.name: gen for gen in generators if gen.is_alive}
    network.alive_load_by_name = {load.name: load for load in loads if load.is_alive}
    network.alive_zero_branch_by_name = {zbr.name: zbr for zbr in zero_branches if zbr.is_alive}
    network.alive_switch_by_name = {sw.name: sw for sw in switches if sw.is_alive}
    network.alive_break_by_name = {brk.name: brk for brk in breakers if brk.is_alive}
    network.alive_acac_converter_by_name = {conv.name: conv for conv in acac_converters if conv.is_alive}
    network.alive_zero_branches = _sorted_by_idx(network.alive_zero_branch_by_name.values())
    network.alive_switches = _sorted_by_idx(network.alive_switch_by_name.values())
    network.alive_breakers = _sorted_by_idx(network.alive_break_by_name.values())
    network.alive_acac_converters = _sorted_by_idx(network.alive_acac_converter_by_name.values())
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
        i_node.isl_obj.dcdc_converters.append(conv)
        j_node.isl_obj.dcdc_converters.append(conv)
        if getattr(conv, "i_control_type", getattr(conv, "control_type", "")) in ("V", "CTRL_V"):
            node = i_node
        elif getattr(conv, "j_control_type", "") in ("V", "CTRL_V"):
            node = j_node
        else:
            node = None
        if node is not None:
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
