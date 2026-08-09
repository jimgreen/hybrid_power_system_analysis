from __future__ import annotations

import argparse
import heapq
import html
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

try:
    from efile_read import EBook
except ImportError:  # pragma: no cover - package import path fallback
    from hybrid_power_system_analysis.efile_read import EBook


NODE_TABLES = ("ACNode", "DCNode")
INJECTION_TABLES = ("ACGenerator", "DCGenerator", "ACLoad", "DCLoad", "ACUnit", "DCUnit")
EDGE_TABLES = (
    "ACBranch",
    "ACTransformer",
    "ACSwitch",
    "ACBreak",
    "DCBranch",
    "DCSwitch",
    "DCBreak",
    "DCDCConverter",
    "DCACConverter",
    "ACACConverter",
)

EDGE_STYLE = {
    "ACBranch": ("#1f5f9f", "交流线路"),
    "ACTransformer": ("#7b61a8", "变压器"),
    "ACSwitch": ("#4c78a8", "交流刀闸"),
    "ACBreak": ("#24476f", "交流断路器"),
    "DCBranch": ("#c46a17", "直流线路"),
    "DCSwitch": ("#d38a31", "直流刀闸"),
    "DCBreak": ("#9c4f0f", "直流断路器"),
    "DCDCConverter": ("#be6b00", "DC/DC"),
    "DCACConverter": ("#2f8a45", "DC/AC"),
    "ACACConverter": ("#6f63bc", "AC/AC"),
}

DEVICE_EDGE_TABLES = {
    "ACSwitch",
    "DCSwitch",
    "ACBreak",
    "DCBreak",
    "DCDCConverter",
    "DCACConverter",
    "ACACConverter",
}


@dataclass(frozen=True)
class StationNode:
    key: str
    idx: int
    name: str
    side: str
    table: str
    vbase: str = ""
    run_stat: int = 1
    is_bus: bool = False


@dataclass(frozen=True)
class StationEdge:
    name: str
    table: str
    source: str
    target: str
    run_stat: int = 1
    status: int | None = None


@dataclass(frozen=True)
class StationInjection:
    idx: int
    name: str
    table: str
    node: str
    kind: str
    side: str
    run_stat: int = 1


@dataclass
class StationGraph:
    nodes: dict[str, StationNode] = field(default_factory=dict)
    edges: list[StationEdge] = field(default_factory=list)
    injections: list[StationInjection] = field(default_factory=list)


@dataclass
class StationLayout:
    positions: dict[str, tuple[float, float]]
    routes: dict[int, list[tuple[float, float]]]
    width: int
    height: int
    crossing_count: int


@dataclass(frozen=True)
class RenderResult:
    input_file: Path
    output_file: Path
    node_count: int
    edge_count: int
    crossing_count: int


def _as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _node_key(side: str, idx) -> str:
    return f"{side}:{_as_int(idx)}"


def _is_running(row: dict) -> bool:
    return _as_int(row.get("run_stat", 1), 1) == 1


def _is_closed(row: dict) -> bool:
    if not _is_running(row):
        return False
    if "status" in row:
        return _as_int(row.get("status", 0), 0) == 1
    return True


def _add_edge(
    graph: StationGraph,
    table: str,
    row: dict,
    source_side: str,
    source_col: str,
    target_side: str,
    target_col: str,
) -> None:
    if not _is_closed(row):
        return
    source = _node_key(source_side, row.get(source_col))
    target = _node_key(target_side, row.get(target_col))
    if source == target:
        return
    if source not in graph.nodes or target not in graph.nodes:
        return
    graph.edges.append(
        StationEdge(
            name=str(row.get("name", row.get("idx", table))),
            table=table,
            source=source,
            target=target,
            run_stat=_as_int(row.get("run_stat", 1), 1),
            status=_as_int(row["status"]) if "status" in row else None,
        )
    )


def _add_injection(graph: StationGraph, table: str, row: dict, side: str, node_col: str) -> None:
    if not _is_running(row):
        return
    node = _node_key(side, row.get(node_col))
    if node not in graph.nodes:
        return
    kind = "load" if "Load" in table else "generator"
    graph.injections.append(
        StationInjection(
            idx=_as_int(row.get("idx")),
            name=str(row.get("name", row.get("idx", table))),
            table=table,
            node=node,
            kind=kind,
            side=side,
            run_stat=_as_int(row.get("run_stat", 1), 1),
        )
    )


def parse_station_graph(e_file: str | Path) -> StationGraph:
    book = EBook(e_file)
    graph = StationGraph()
    declared_bus_nodes = set()
    for table, side in (("ACRealBs", "AC"), ("DCRealBs", "DC")):
        block = book.data.get(table)
        for row in getattr(block, "data", []):
            if _is_running(row):
                declared_bus_nodes.add(_node_key(side, row.get("node")))

    for table in NODE_TABLES:
        side = "AC" if table.startswith("AC") else "DC"
        block = book.data.get(table)
        if block is None:
            continue
        for row in block.data:
            if not _is_running(row):
                continue
            idx = _as_int(row.get("idx"))
            key = _node_key(side, idx)
            graph.nodes[key] = StationNode(
                key=key,
                idx=idx,
                name=str(row.get("name", key)),
                side=side,
                table=table,
                vbase=str(row.get("vbase", "")),
                run_stat=_as_int(row.get("run_stat", 1), 1),
                is_bus=key in declared_bus_nodes,
            )

    edge_specs = {
        "ACBranch": ("AC", "i_node", "AC", "j_node"),
        "ACTransformer": ("AC", "i_node", "AC", "j_node"),
        "ACSwitch": ("AC", "i_node", "AC", "j_node"),
        "ACBreak": ("AC", "i_node", "AC", "j_node"),
        "DCBranch": ("DC", "i_node", "DC", "j_node"),
        "DCSwitch": ("DC", "i_node", "DC", "j_node"),
        "DCBreak": ("DC", "i_node", "DC", "j_node"),
        "DCDCConverter": ("DC", "i_node", "DC", "j_node"),
        "DCACConverter": ("AC", "ac_node", "DC", "dc_node"),
        "ACACConverter": ("AC", "i_node", "AC", "j_node"),
    }
    for table in EDGE_TABLES:
        block = book.data.get(table)
        if block is None:
            continue
        spec = edge_specs[table]
        for row in block.data:
            _add_edge(graph, table, row, *spec)

    injection_specs = {
        "ACGenerator": ("AC", "node"),
        "DCGenerator": ("DC", "node"),
        "ACLoad": ("AC", "node"),
        "DCLoad": ("DC", "node"),
        "ACUnit": ("AC", "node"),
        "DCUnit": ("DC", "node"),
    }
    for table in INJECTION_TABLES:
        block = book.data.get(table)
        if block is None:
            continue
        side, node_col = injection_specs[table]
        for row in block.data:
            _add_injection(graph, table, row, side, node_col)
    return graph


def _build_adjacency(graph: StationGraph, side: str) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {key: set() for key, node in graph.nodes.items() if node.side == side}
    for edge in graph.edges:
        source = graph.nodes[edge.source]
        target = graph.nodes[edge.target]
        if source.side == side and target.side == side:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set()).add(edge.source)
    return adjacency


def _choose_root(graph: StationGraph, keys: Iterable[str], adjacency: dict[str, set[str]]) -> str:
    key_list = list(keys)
    if not key_list:
        raise ValueError("empty side")
    declared_buses = [key for key in key_list if graph.nodes[key].is_bus]
    candidates = declared_buses or key_list
    return max(candidates, key=lambda key: (len(adjacency.get(key, ())), -graph.nodes[key].idx))


def _natural_node_order(graph: StationGraph, key: str):
    node = graph.nodes[key]
    return 0 if node.is_bus else 1, node.idx, node.key


def _bfs_component_levels(start: str, levels: dict[str, int], adjacency: dict[str, set[str]]) -> None:
    levels[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in sorted(adjacency.get(current, ()), key=lambda key: key):
            if nxt in levels:
                continue
            levels[nxt] = levels[current] + 1
            queue.append(nxt)


def _bfs_levels(graph: StationGraph, root: str, keys: set[str], adjacency: dict[str, set[str]]) -> dict[str, int]:
    levels: dict[str, int] = {}
    _bfs_component_levels(root, levels, adjacency)
    for key in sorted(keys, key=lambda item: _natural_node_order(graph, item)):
        if key not in levels:
            component_root = _choose_root(graph, {key} | (keys - set(levels)), adjacency)
            _bfs_component_levels(component_root, levels, adjacency)
    return levels


def _order_nodes(graph: StationGraph, root: str, keys: set[str], adjacency: dict[str, set[str]]) -> list[str]:
    visited = set()
    order = []

    def walk(key: str) -> None:
        visited.add(key)
        order.append(key)
        neighbors = sorted(
            (n for n in adjacency.get(key, ()) if n not in visited),
            key=lambda item: (len(adjacency.get(item, ())), _natural_node_order(graph, item)),
        )
        for neighbor in neighbors:
            walk(neighbor)

    walk(root)
    for key in sorted(keys, key=lambda item: _natural_node_order(graph, item)):
        if key not in visited:
            walk(key)
    return order


def _layout_side(
    graph: StationGraph,
    side: str,
    y_root: float,
    y_step: float,
    x_margin: float,
    x_step: float,
) -> dict[str, tuple[float, float]]:
    keys = {key for key, node in graph.nodes.items() if node.side == side}
    if not keys:
        return {}
    adjacency = _build_adjacency(graph, side)
    root = _choose_root(graph, keys, adjacency)
    levels = _bfs_levels(graph, root, keys, adjacency)
    order = _order_nodes(graph, root, keys, adjacency)

    by_level: dict[int, list[str]] = defaultdict(list)
    for key in order:
        by_level[levels[key]].append(key)

    positions = {}
    for level in sorted(by_level):
        row = by_level[level]
        width = max(0, len(row) - 1) * x_step
        start_x = x_margin - width / 2
        for pos, key in enumerate(row):
            positions[key] = (start_x + pos * x_step, y_root + level * y_step)
    return positions


def _route_edge(
    edge: StationEdge,
    graph: StationGraph,
    positions: dict[str, tuple[float, float]],
    side_bounds: dict[str, tuple[float, float]],
    lane: float,
) -> list[tuple[float, float]]:
    x1, y1 = positions[edge.source]
    x2, y2 = positions[edge.target]
    if abs(x1 - x2) < 1e-9 or abs(y1 - y2) < 1e-9:
        return [(x1, y1), (x2, y2)]
    source_side = graph.nodes[edge.source].side
    target_side = graph.nodes[edge.target].side
    same_side = source_side == target_side
    if same_side:
        min_y, max_y = side_bounds[source_side]
        bend_y = min_y - 45.0 - lane if source_side == "AC" else max_y + 45.0 + lane
        return [(x1, y1), (x1, bend_y), (x2, bend_y), (x2, y2)]
    return [(x1, y1), (x2, y1), (x2, y2)]


def _simplify_route(route: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(route) <= 2:
        return route
    simplified = [route[0]]
    for index, point in enumerate(route[1:-1], start=1):
        prev = simplified[-1]
        nxt = route[index + 1]
        if (abs(prev[0] - point[0]) < 1e-9 and abs(point[0] - nxt[0]) < 1e-9) or (
            abs(prev[1] - point[1]) < 1e-9 and abs(point[1] - nxt[1]) < 1e-9
        ):
            continue
        simplified.append(point)
    simplified.append(route[-1])
    return simplified


def _point_on_segment(point: tuple[float, float], segment) -> bool:
    (x, y) = point
    (x1, y1), (x2, y2) = segment
    if abs(x1 - x2) < 1e-9:
        return abs(x - x1) < 1e-9 and min(y1, y2) < y < max(y1, y2)
    if abs(y1 - y2) < 1e-9:
        return abs(y - y1) < 1e-9 and min(x1, x2) < x < max(x1, x2)
    return False


def _segment_blocked(
    segment,
    used_segments: list,
    blocked_points: set[tuple[float, float]],
    endpoints: set[tuple[float, float]],
) -> bool:
    for point in blocked_points:
        if point not in endpoints and _point_on_segment(point, segment):
            return True
    for used in used_segments:
        if _segment_crosses(segment, used):
            return True
    return False


def _orthogonal_grid_route(
    start: tuple[float, float],
    target: tuple[float, float],
    x_grid: list[float],
    y_grid: list[float],
    used_segments: list,
    blocked_points: set[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    start_idx = (x_grid.index(start[0]), y_grid.index(start[1]))
    target_idx = (x_grid.index(target[0]), y_grid.index(target[1]))
    endpoints = {start, target}
    blocked = {
        (x_grid.index(x), y_grid.index(y))
        for x, y in blocked_points
        if (x, y) not in endpoints and x in x_grid and y in y_grid
    }
    queue: list[tuple[float, int, tuple[int, int], tuple[int, int] | None]] = []
    heapq.heappush(queue, (0.0, 0, start_idx, None))
    best: dict[tuple[tuple[int, int], tuple[int, int] | None], float] = {(start_idx, None): 0.0}
    parent: dict[tuple[tuple[int, int], tuple[int, int] | None], tuple[tuple[int, int], tuple[int, int] | None]] = {}
    sequence = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    final_state = None
    while queue:
        cost, _seq, current, previous_direction = heapq.heappop(queue)
        state = (current, previous_direction)
        if cost > best.get(state, math.inf):
            continue
        if current == target_idx:
            final_state = state
            break
        cx, cy = current
        for direction in directions:
            nx, ny = cx + direction[0], cy + direction[1]
            if nx < 0 or ny < 0 or nx >= len(x_grid) or ny >= len(y_grid):
                continue
            nxt = (nx, ny)
            if nxt in blocked:
                continue
            segment = ((x_grid[cx], y_grid[cy]), (x_grid[nx], y_grid[ny]))
            if _segment_blocked(segment, used_segments, blocked_points, endpoints):
                continue
            step = abs(x_grid[nx] - x_grid[cx]) + abs(y_grid[ny] - y_grid[cy])
            turn_penalty = 18.0 if previous_direction is not None and previous_direction != direction else 0.0
            target_bias = 0.03 * (abs(x_grid[nx] - target[0]) + abs(y_grid[ny] - target[1]))
            next_cost = cost + step + turn_penalty + target_bias
            next_state = (nxt, direction)
            if next_cost >= best.get(next_state, math.inf):
                continue
            best[next_state] = next_cost
            parent[next_state] = state
            sequence += 1
            heapq.heappush(queue, (next_cost, sequence, nxt, direction))

    if final_state is None:
        return None
    cells = []
    state = final_state
    while True:
        cells.append(state[0])
        if state not in parent:
            break
        state = parent[state]
    cells.reverse()
    return _simplify_route([(x_grid[x], y_grid[y]) for x, y in cells])


def _build_grid_routes(
    graph: StationGraph,
    positions: dict[str, tuple[float, float]],
    side_bounds: dict[str, tuple[float, float]],
) -> dict[int, list[tuple[float, float]]]:
    xs = {x for x, _ in positions.values()}
    ys = {y for _, y in positions.values()}
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    for step in range(-3, 4):
        ys.add(min_y - 55.0 * (step + 4))
        ys.add(max_y + 55.0 * (step + 4))
    left = min_x - 170.0
    right = max_x + 170.0
    top = min_y - 260.0
    bottom = max_y + 260.0
    grid_step = 55.0
    count = int(math.ceil((right - left) / grid_step))
    for idx in range(count + 1):
        xs.add(round(left + idx * grid_step, 1))
    count = int(math.ceil((bottom - top) / grid_step))
    for idx in range(count + 1):
        ys.add(round(top + idx * grid_step, 1))
    x_grid = sorted(xs)
    y_grid = sorted(ys)

    blocked_points = set(positions.values())
    used_segments = []
    routes: dict[int, list[tuple[float, float]]] = {}

    def priority(item: tuple[int, StationEdge]):
        idx, edge = item
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        cross_rank = 0 if graph.nodes[edge.source].side != graph.nodes[edge.target].side else 1
        length = abs(x1 - x2) + abs(y1 - y2)
        return cross_rank, -length, idx

    for idx, edge in sorted(enumerate(graph.edges), key=priority):
        start = positions[edge.source]
        target = positions[edge.target]
        route = _orthogonal_grid_route(start, target, x_grid, y_grid, used_segments, blocked_points)
        if route is None:
            source_side = graph.nodes[edge.source].side
            target_side = graph.nodes[edge.target].side
            route = _route_edge(edge, graph, positions, side_bounds, 0.0 if source_side != target_side else 16.0)
        routes[idx] = route
        used_segments.extend(_segments(route))
    return routes


def _segments(route: list[tuple[float, float]]):
    for left, right in zip(route, route[1:]):
        if left == right:
            continue
        yield left, right


def _between(value: float, a: float, b: float) -> bool:
    return min(a, b) < value < max(a, b)


def _segment_crosses(a, b) -> bool:
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    a_vertical = abs(x1 - x2) < 1e-9
    b_vertical = abs(x3 - x4) < 1e-9
    if a_vertical == b_vertical:
        return False
    if a_vertical:
        return _between(x1, x3, x4) and _between(y3, y1, y2)
    return _between(x3, x1, x2) and _between(y1, y3, y4)


def _count_crossings(routes: dict[int, list[tuple[float, float]]]) -> int:
    segs = []
    for idx, route in routes.items():
        for segment in _segments(route):
            segs.append((idx, segment))
    count = 0
    for i, (left_idx, left) in enumerate(segs):
        for right_idx, right in segs[i + 1:]:
            if left_idx == right_idx:
                continue
            if set(left) & set(right):
                continue
            if _segment_crosses(left, right):
                count += 1
    return count


def _is_bus_node(node: StationNode) -> bool:
    return node.is_bus


def _station_graph_adjacency(graph: StationGraph) -> dict[str, set[str]]:
    adjacency = {key: set() for key in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    return adjacency


def _distances_from(start: str, adjacency: dict[str, set[str]]) -> dict[str, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _station_bay_components(
    graph: StationGraph,
    adjacency: dict[str, set[str]],
) -> list[set[str]]:
    bus_nodes = {key for key, node in graph.nodes.items() if node.is_bus}
    remaining = set(graph.nodes) - bus_nodes
    components = []
    while remaining:
        start = min(remaining, key=lambda key: (graph.nodes[key].side, graph.nodes[key].idx, key))
        component = set()
        queue = deque([start])
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return sorted(
        components,
        key=lambda component: min(
            (graph.nodes[key].idx, 0 if graph.nodes[key].side == "AC" else 1, key)
            for key in component
        ),
    )


def _route_station_edge(
    edge: StationEdge,
    graph: StationGraph,
    positions: dict[str, tuple[float, float]],
    bus_by_side: dict[str, str],
    channel_index: int,
) -> list[tuple[float, float]]:
    x1, y1 = positions[edge.source]
    x2, y2 = positions[edge.target]
    source_node = graph.nodes[edge.source]
    target_node = graph.nodes[edge.target]
    if edge.source in bus_by_side.values() or edge.target in bus_by_side.values():
        bus = source_node if _is_bus_node(source_node) else target_node
        other_x, other_y = (x2, y2) if _is_bus_node(source_node) else (x1, y1)
        bus_y = positions[bus.key][1]
        return [(other_x, other_y), (other_x, bus_y)]
    if abs(x1 - x2) < 1e-9 or abs(y1 - y2) < 1e-9:
        return [(x1, y1), (x2, y2)]
    top = min(y1, y2) - 45.0 - channel_index * 18.0
    bottom = max(y1, y2) + 45.0 + channel_index * 18.0
    channel_y = top if source_node.side == "AC" and target_node.side == "AC" else bottom
    return [(x1, y1), (x1, channel_y), (x2, channel_y), (x2, y2)]


def _layout_station_bays(graph: StationGraph) -> StationLayout:
    if not graph.nodes:
        raise ValueError("E 文件中没有可绘制的 ACNode/DCNode")

    bus_by_side: dict[str, str] = {}
    for side in ("AC", "DC"):
        buses = [node for node in graph.nodes.values() if node.side == side and node.is_bus]
        if buses:
            bus_by_side[side] = min(buses, key=lambda node: node.idx).key

    adjacency = _station_graph_adjacency(graph)
    components = _station_bay_components(graph, adjacency)
    x_margin = 75.0
    x_step = min(70.0, 1050.0 / max(1, len(components) - 1))
    component_x = {
        frozenset(component): x_margin + index * x_step
        for index, component in enumerate(components)
    }
    node_x = {
        key: x
        for component, x in component_x.items()
        for key in component
    }
    width = int(max(920.0, 2 * x_margin + max(0, len(components) - 1) * x_step))
    ac_bus_y = 265.0
    dc_bus_y = 395.0
    positions: dict[str, tuple[float, float]] = {}
    side_distances = {
        side: _distances_from(bus_key, _build_adjacency(graph, side))
        for side, bus_key in bus_by_side.items()
    }
    injection_nodes = {injection.node for injection in graph.injections}
    cross_domain_nodes = {
        key
        for edge in graph.edges
        if graph.nodes[edge.source].side != graph.nodes[edge.target].side
        for key in (edge.source, edge.target)
    }

    center_x = width / 2
    for key, node in graph.nodes.items():
        if _is_bus_node(node):
            y = ac_bus_y if node.side == "AC" else dc_bus_y
            positions[key] = (center_x, y)
            continue
        distance = side_distances.get(node.side, {}).get(key)
        if node.side == "AC":
            if distance is not None:
                offset = min(155.0, 65.0 + max(0, distance - 1) * 45.0)
            elif key in injection_nodes:
                offset = 155.0
            elif key in cross_domain_nodes:
                offset = 65.0
            else:
                offset = 108.0
            y = ac_bus_y - offset
        else:
            if distance is not None:
                offset = min(160.0, 36.0 + max(0, distance - 1) * 42.0)
            elif key in injection_nodes:
                offset = 160.0
            elif key in cross_domain_nodes:
                offset = 78.0
            else:
                offset = 110.0
            y = dc_bus_y + offset
        positions[key] = (node_x[key], y)

    routes = {}
    channel_index = 0
    for idx, edge in enumerate(graph.edges):
        if edge.source not in positions or edge.target not in positions:
            continue
        route = _route_station_edge(edge, graph, positions, bus_by_side, channel_index)
        routes[idx] = route
        if len(route) > 2:
            channel_index += 1

    height = 660
    return StationLayout(
        positions=positions,
        routes=routes,
        width=width,
        height=height,
        crossing_count=_count_crossings(routes),
    )


def layout_station_graph(graph: StationGraph) -> StationLayout:
    for side in ("AC", "DC"):
        side_keys = {
            key
            for key, node in graph.nodes.items()
            if node.side == side
        }
        if not side_keys or any(graph.nodes[key].is_bus for key in side_keys):
            continue
        adjacency = _build_adjacency(graph, side)
        root = _choose_root(graph, side_keys, adjacency)
        graph.nodes[root] = replace(graph.nodes[root], is_bus=True)

    return _layout_station_bays(graph)


def _svg_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _device_placement(edge: StationEdge, route: list[tuple[float, float]]):
    if edge.table not in DEVICE_EDGE_TABLES:
        return None
    best_index = 0
    best_length = -1.0
    for index, (left, right) in enumerate(_segments(route)):
        length = abs(right[0] - left[0]) + abs(right[1] - left[1])
        if length > best_length:
            best_index = index
            best_length = length
    left = route[best_index]
    right = route[best_index + 1]
    horizontal = abs(left[1] - right[1]) < 1e-9
    cx = (left[0] + right[0]) / 2
    cy = (left[1] + right[1]) / 2
    half_gap = 26.0 if "Converter" in edge.table else 15.0
    if horizontal:
        sign = 1.0 if right[0] >= left[0] else -1.0
        terminal_a = (cx - sign * half_gap, cy)
        terminal_b = (cx + sign * half_gap, cy)
        orientation = "horizontal"
    else:
        sign = 1.0 if right[1] >= left[1] else -1.0
        terminal_a = (cx, cy - sign * half_gap)
        terminal_b = (cx, cy + sign * half_gap)
        orientation = "vertical"
    return best_index, (cx, cy), terminal_a, terminal_b, orientation


def _split_route_for_device(edge: StationEdge, route: list[tuple[float, float]]):
    placement = _device_placement(edge, route)
    if placement is None:
        return None
    segment_index, _center, terminal_a, terminal_b, _orientation = placement
    first = route[: segment_index + 1] + [terminal_a]
    second = [terminal_b] + route[segment_index + 1 :]
    return placement, _simplify_route(first), _simplify_route(second)


def _device_symbol(edge: StationEdge, placement) -> str:
    if placement is None:
        return ""
    _segment_index, (x, y), terminal_a, terminal_b, orientation = placement
    label = {
        "ACSwitch": "QS",
        "DCSwitch": "QS",
        "ACBreak": "QF",
        "DCBreak": "QF",
        "DCDCConverter": "DC/DC",
        "DCACConverter": "AC/DC",
        "ACACConverter": "AC/AC",
    }[edge.table]
    terminal_markup = (
        f'<circle cx="{terminal_a[0]:.1f}" cy="{terminal_a[1]:.1f}" r="3.2" class="device-terminal"/>'
        f'<circle cx="{terminal_b[0]:.1f}" cy="{terminal_b[1]:.1f}" r="3.2" class="device-terminal"/>'
    )
    if "Converter" in edge.table:
        if orientation == "horizontal":
            rect = f'<rect x="{x - 23:.1f}" y="{y - 12:.1f}" width="46" height="24" rx="3" class="converter"/>'
        else:
            rect = f'<rect x="{x - 16:.1f}" y="{y - 24:.1f}" width="32" height="48" rx="3" class="converter"/>'
        return (
            f'<g class="converter-device" data-device-name="{html.escape(edge.name)}">'
            f'{rect}'
            f'</g>'
        )
    if orientation == "horizontal":
        rect = f'<rect x="{x - 11:.1f}" y="{y - 8:.1f}" width="22" height="16" class="breaker"/>'
    else:
        rect = f'<rect x="{x - 8:.1f}" y="{y - 11:.1f}" width="16" height="22" class="breaker"/>'
    return (
        f'<g class="two-terminal-device" data-device-name="{html.escape(edge.name)}">'
        f'{terminal_markup}'
        f'{rect}'
        f'</g>'
    )


def _injection_anchor(
    injection: StationInjection,
    graph: StationGraph,
    layout: StationLayout,
    order_index: int,
    order_count: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x, y = layout.positions[injection.node]
    node = graph.nodes[injection.node]
    offset = (order_index - (order_count - 1) / 2.0) * 24.0
    if injection.kind == "generator":
        direction = -1.0 if node.side == "AC" else 1.0
    else:
        direction = 1.0 if node.side == "AC" else -1.0
    terminal = (x + offset, y + direction * 18.0)
    symbol = (x + offset, y + direction * 40.0)
    return (x, y), terminal, symbol


def _injection_symbol(
    injection: StationInjection,
    graph: StationGraph,
    layout: StationLayout,
    order_index: int,
    order_count: int,
) -> str:
    source, terminal, symbol = _injection_anchor(injection, graph, layout, order_index, order_count)
    color = "#2f7d32" if injection.kind == "generator" else "#8a3ffc"
    label = "G" if injection.kind == "generator" else "L"
    escaped_name = html.escape(injection.name)
    direction = 1.0 if symbol[1] >= source[1] else -1.0
    if injection.kind == "generator":
        lead_end = (symbol[0], symbol[1] - direction * 12.0)
        body = (
            f'<circle cx="{symbol[0]:.1f}" cy="{symbol[1]:.1f}" r="12" class="injection-generator"/>'
        )
    else:
        apex = (symbol[0], symbol[1] - direction * 13.0)
        base_y = symbol[1] + direction * 10.0
        lead_end = apex
        points = (
            f'{apex[0]:.1f},{apex[1]:.1f} '
            f'{symbol[0] - 13:.1f},{base_y:.1f} '
            f'{symbol[0] + 13:.1f},{base_y:.1f}'
        )
        body = (
            f'<polygon points="{points}" class="injection-load"/>'
        )
    return (
        f'<g class="single-terminal-injection" data-injection-name="{escaped_name}">'
        f'<polyline class="injection-lead" points="{_svg_points([source, terminal, lead_end])}" stroke="{color}"/>'
        f'<circle cx="{terminal[0]:.1f}" cy="{terminal[1]:.1f}" r="3.2" class="injection-terminal"/>'
        f'{body}'
        f'</g>'
    )


def render_station_svg(graph: StationGraph, layout: StationLayout, title: str = "厂站接线图") -> str:
    styles = """
    <style>
      .background { fill: #ffffff; }
      .title { font: 700 22px Arial, "Microsoft YaHei", sans-serif; fill: #1f2933; }
      .subtitle { font: 13px Arial, "Microsoft YaHei", sans-serif; fill: #5b6775; }
      .node-label { font: 12px Arial, "Microsoft YaHei", sans-serif; fill: #18212f; text-anchor: middle; }
      .node-meta { font: 10px Arial, "Microsoft YaHei", sans-serif; fill: #64748b; text-anchor: middle; }
      .edge-label { font: 10px Arial, "Microsoft YaHei", sans-serif; fill: #334155; text-anchor: middle; }
      .symbol-label { font: 9px Arial, "Microsoft YaHei", sans-serif; fill: #263238; text-anchor: middle; dominant-baseline: middle; }
      .branch-route { fill: none; stroke-width: 2.2; stroke-linecap: square; stroke-linejoin: miter; }
      .busbar-ac { stroke: #1f5f9f; stroke-width: 5; stroke-linecap: square; }
      .busbar-dc { stroke: #c46a17; stroke-width: 5; stroke-linecap: square; }
      .breaker { fill: #fff; stroke: #334155; stroke-width: 1.2; }
      .converter { fill: #f8fafc; stroke: #2f8a45; stroke-width: 1.2; }
      .device-terminal { fill: #ffffff; stroke: #263238; stroke-width: 1.1; }
      .injection-lead { fill: none; stroke-width: 1.4; stroke-linecap: square; stroke-linejoin: miter; }
      .injection-terminal { fill: #ffffff; stroke: #263238; stroke-width: 1.1; }
      .injection-generator { fill: #eef8ee; stroke: #2f7d32; stroke-width: 1.3; }
      .injection-load { fill: #f4efff; stroke: #8a3ffc; stroke-width: 1.3; }
      .injection-label { font: 9px Arial, "Microsoft YaHei", sans-serif; fill: #4b5563; text-anchor: middle; }
      .legend { font: 12px Arial, "Microsoft YaHei", sans-serif; fill: #334155; }
    </style>
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout.width}" height="{layout.height}" viewBox="0 0 {layout.width} {layout.height}">',
        styles,
        f'<rect class="background" x="0" y="0" width="{layout.width}" height="{layout.height}"/>',
        f'<text x="32" y="36" class="title">{html.escape(title)}</text>',
        f'<text x="32" y="58" class="subtitle">正交布线：横平竖直；节点数 {len(graph.nodes)}，连接数 {len(graph.edges)}，检测交叉 {layout.crossing_count}</text>',
    ]

    bus_extents: dict[str, tuple[float, float]] = {}
    for key, node in graph.nodes.items():
        if not _is_bus_node(node) or key not in layout.positions:
            continue
        connected_x = []
        for edge in graph.edges:
            other = None
            if edge.source == key:
                other = edge.target
            elif edge.target == key:
                other = edge.source
            if other is not None and other in layout.positions:
                connected_x.append(layout.positions[other][0])
        if connected_x:
            bus_extents[key] = (min(connected_x) - 35.0, max(connected_x) + 35.0)
        else:
            x, _y = layout.positions[key]
            bus_extents[key] = (x - 42.0, x + 42.0)

    hidden_node_labels = {
        endpoint
        for edge in graph.edges
        if "Converter" in edge.table
        for endpoint in (edge.source, edge.target)
    }
    converter_connection_points: dict[str, tuple[float, float]] = {}
    converter_placements = {}
    for idx, edge in enumerate(graph.edges):
        if "Converter" not in edge.table:
            continue
        route = layout.routes.get(idx)
        if not route:
            continue
        placement = _device_placement(edge, route)
        if placement is None:
            continue
        converter_placements[idx] = placement
        _segment_index, _center, terminal_a, terminal_b, _orientation = placement
        converter_connection_points[edge.source] = terminal_a
        converter_connection_points[edge.target] = terminal_b

    for idx, edge in enumerate(graph.edges):
        route = layout.routes.get(idx)
        if not route:
            continue
        color, _label = EDGE_STYLE.get(edge.table, ("#475569", edge.table))
        dash = ' stroke-dasharray="7 5"' if "Switch" in edge.table else ""
        if "Converter" in edge.table:
            placement = converter_placements.get(idx) or _device_placement(edge, route)
            parts.append(_device_symbol(edge, placement))
            continue
        route = list(route)
        if edge.source in converter_connection_points:
            route[0] = converter_connection_points[edge.source]
        if edge.target in converter_connection_points:
            route[-1] = converter_connection_points[edge.target]
        route = _simplify_route(route)
        split = _split_route_for_device(edge, route)
        if split is None:
            parts.append(
                f'<polyline class="branch-route" data-name="{html.escape(edge.name)}" points="{_svg_points(route)}" '
                f'stroke="{color}"{dash}/>'
            )
            mid = route[len(route) // 2]
        else:
            placement, first_route, second_route = split
            if len(first_route) >= 2:
                parts.append(
                    f'<polyline class="branch-route" data-name="{html.escape(edge.name)}" points="{_svg_points(first_route)}" '
                    f'stroke="{color}"{dash}/>'
                )
            if len(second_route) >= 2:
                parts.append(
                    f'<polyline class="branch-route" data-name="{html.escape(edge.name)}" points="{_svg_points(second_route)}" '
                    f'stroke="{color}"{dash}/>'
                )
            parts.append(_device_symbol(edge, placement))
            mid = placement[1]

    injections_by_node: dict[str, list[StationInjection]] = defaultdict(list)
    for injection in graph.injections:
        if injection.node in layout.positions:
            injections_by_node[injection.node].append(injection)
    for node_key, injections in injections_by_node.items():
        ordered = sorted(injections, key=lambda item: (item.kind, item.table, item.idx))
        for order_index, injection in enumerate(ordered):
            parts.append(_injection_symbol(injection, graph, layout, order_index, len(ordered)))

    for key, node in graph.nodes.items():
        if key not in layout.positions:
            continue
        x, y = layout.positions[key]
        if _is_bus_node(node):
            cls = "busbar-ac" if node.side == "AC" else "busbar-dc"
            bus_left, bus_right = bus_extents.get(key, (x - 42.0, x + 42.0))
            parts.append(f'<line x1="{bus_left:.1f}" y1="{y:.1f}" x2="{bus_right:.1f}" y2="{y:.1f}" class="{cls}"/>')

    legend_x = layout.width - 250
    parts.append(f'<g transform="translate({legend_x},28)">')
    parts.append('<text x="0" y="0" class="legend">图例</text>')
    for row, table in enumerate(("ACBranch", "DCBranch", "ACBreak", "DCDCConverter", "DCACConverter")):
        color, label = EDGE_STYLE[table]
        y = 24 + row * 20
        parts.append(f'<line x1="0" y1="{y}" x2="34" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="44" y="{y + 4}" class="legend">{label}</text>')
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(part for part in parts if part)


def render_station_svg_file(
    e_file: str | Path,
    svg_file: str | Path,
    *,
    title: str = "厂站接线图",
) -> RenderResult:
    input_path = Path(e_file)
    output_path = Path(svg_file)
    graph = parse_station_graph(input_path)
    layout = layout_station_graph(graph)
    svg = render_station_svg(graph, layout, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return RenderResult(
        input_file=input_path,
        output_file=output_path,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        crossing_count=layout.crossing_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an E-file station single-line diagram as orthogonal SVG.")
    parser.add_argument("input", help="Input E file.")
    parser.add_argument("output", help="Output SVG file.")
    parser.add_argument("--title", default="厂站接线图", help="SVG title.")
    args = parser.parse_args(argv)
    result = render_station_svg_file(args.input, args.output, title=args.title)
    print(f"nodes={result.node_count} edges={result.edge_count} crossings={result.crossing_count}")
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
