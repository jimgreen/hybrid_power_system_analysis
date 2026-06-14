import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
PKG_DIR = SRC_DIR / "hybrid_power_system_analysis"
for path in (SRC_DIR, PKG_DIR, PKG_DIR / "model", PKG_DIR / "lfcore"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from hybrid_lf import HybridPowerFlowCalc
from hybrid_model import HybridPowerNetwork


DEFAULT_CASE = ROOT_DIR / "data" / "model" / "hybrid" / "qinling.e"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts"

AC_BUS_COLOR = "#1f77b4"
DC_BUS_COLOR = "#ff7f0e"
AC_BRANCH_COLOR = "#4e79a7"
AC_XFMR_COLOR = "#9467bd"
DC_BRANCH_COLOR = "#f28e2b"
DCDC_COLOR = "#cc6600"
DCAC_COLOR = "#2ca02c"
TOPO_LINK_COLOR = "#7f7f7f"
BACKBONE_BAR_COLOR = "#5f6368"
BACKBONE_TAP_COLOR = "#8a8f96"


def _flat_start_network(network: HybridPowerNetwork) -> None:
    for node in getattr(network.ac, "nodes", []):
        node.voltage = 1.0
        node.angle = 0.0
    for bus in getattr(network.ac, "buses", []):
        bus.voltage = 1.0
        bus.angle = 0.0
    for node in getattr(network.dc, "nodes", []):
        node.voltage = 1.0
    for bus in getattr(network.dc, "buses", []):
        bus.voltage = 1.0


def _node_key(side: str, idx: int) -> str:
    return f"{side}:{idx}"


def _bus_name(bus) -> str:
    return str(getattr(bus, "name", f"bus_{getattr(bus, 'idx', 0)}"))


def _node_bus(node):
    return getattr(node, "bus_obj", None) or getattr(node, "bus", None) or node


def _is_alive(device) -> bool:
    return bool(getattr(device, "is_alive", False))


def _alive_ac_buses(network: HybridPowerNetwork):
    return [bus for bus in getattr(network.ac, "alive_buses", []) if getattr(bus, "is_alive", False)]


def _alive_dc_buses(network: HybridPowerNetwork):
    return [bus for bus in getattr(network.dc, "alive_buses", []) if getattr(bus, "is_alive", False)]


def _normalize_layout(raw_pos, x_range, y_center=0.0, y_span=1.8):
    if not raw_pos:
        return {}
    xs = np.asarray([xy[0] for xy in raw_pos.values()], dtype=np.float64)
    ys = np.asarray([xy[1] for xy in raw_pos.values()], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    dx = x_max - x_min
    dy = y_max - y_min
    if dx <= 1e-12:
        dx = 1.0
    if dy <= 1e-12:
        dy = 1.0

    x0, x1 = x_range
    out = {}
    for key, (x, y) in raw_pos.items():
        px = x0 + (float(x) - x_min) / dx * (x1 - x0)
        py = y_center + ((float(y) - y_min) / dy - 0.5) * y_span
        out[key] = (px, py)
    return out


def _ac_topology_edges(network: HybridPowerNetwork):
    edges = []
    for dev in list(getattr(network.ac, "branches", [])) + list(getattr(network.ac, "transformers", [])):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        edges.append((_node_key("ac", int(i_bus.idx)), _node_key("ac", int(j_bus.idx))))
    return edges


def _dc_topology_edges(network: HybridPowerNetwork):
    edges = []
    for dev in list(getattr(network.dc, "branches", [])) + list(getattr(network.dc, "dcdc_converters", [])):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        edges.append((_node_key("dc", int(i_bus.idx)), _node_key("dc", int(j_bus.idx))))
    return edges


def _dcac_edges(network: HybridPowerNetwork):
    edges = []
    for dev in getattr(network, "dcac_converters", []):
        if not _is_alive(dev):
            continue
        ac_node = getattr(dev, "ac_node_obj", None)
        dc_node = getattr(dev, "dc_node_obj", None)
        if ac_node is None or dc_node is None:
            continue
        ac_bus = _node_bus(ac_node)
        dc_bus = _node_bus(dc_node)
        edges.append((_node_key("ac", int(ac_bus.idx)), _node_key("dc", int(dc_bus.idx))))
    return edges


def _is_closed_topology_link(device) -> bool:
    if not _is_alive(device):
        return False
    run_stat = int(getattr(device, "run_stat", 0) or 0)
    status = int(getattr(device, "status", 0) or 0)
    return run_stat == 1 and status == 1


def _topology_aux_links(network: HybridPowerNetwork):
    links = []
    for side, group in (("ac", list(getattr(network.ac, "switches", [])) + list(getattr(network.ac, "breakers", []))),
                        ("dc", list(getattr(network.dc, "switches", [])) + list(getattr(network.dc, "breakers", [])))):
        for dev in group:
            if not _is_closed_topology_link(dev):
                continue
            i_node = getattr(dev, "i_node_obj", None)
            j_node = getattr(dev, "j_node_obj", None)
            if i_node is None or j_node is None:
                continue
            i_bus = _node_bus(i_node)
            j_bus = _node_bus(j_node)
            if i_bus is None or j_bus is None:
                continue
            links.append((side, dev, _node_key(side, int(i_bus.idx)), _node_key(side, int(j_bus.idx))))
    return links


def _topology_backbones(network: HybridPowerNetwork, positions):
    backbones = []
    for side in ("ac", "dc"):
        links = [(i_key, j_key) for link_side, _dev, i_key, j_key in _topology_aux_links(network) if link_side == side and i_key != j_key]
        if not links:
            continue
        counts = {}
        for i_key, j_key in links:
            counts[i_key] = counts.get(i_key, 0) + 1
            counts[j_key] = counts.get(j_key, 0) + 1
        if not counts:
            continue
        hub_key = max(counts.items(), key=lambda item: item[1])[0]
        member_keys = {hub_key}
        for i_key, j_key in links:
            if i_key == hub_key:
                member_keys.add(j_key)
            elif j_key == hub_key:
                member_keys.add(i_key)
        if len(member_keys) <= 1:
            continue
        hub_pos = positions.get(hub_key)
        if hub_pos is None:
            continue
        ys = [positions[key][1] for key in member_keys if key in positions]
        if not ys:
            continue
        hub_name = None
        for name, info in _name_to_bus_map(network).items():
            node_key = _node_key(info[0], info[1])
            if node_key == hub_key:
                hub_name = name
                break
        if hub_name == "ac_bus":
            spine_x = 0.68
        elif hub_name == "dc_bus_720v":
            spine_x = -0.02
        elif side == "ac":
            spine_x = hub_pos[0] - 0.26
        else:
            spine_x = hub_pos[0] + 0.12
        backbones.append(
            {
                "side": side,
                "hub_key": hub_key,
                "spine_x": spine_x,
                "y_min": min(ys),
                "y_max": max(ys),
                "member_keys": member_keys,
                "label": "Main AC Bus" if side == "ac" else "Main DC Bus",
            }
        )
    return backbones


def _draw_backbones(ax, network: HybridPowerNetwork, positions, alpha=0.9, linewidth=1.2, zorder=0, compact=False):
    for backbone in _topology_backbones(network, positions):
        spine_x = backbone["spine_x"]
        y_min = backbone["y_min"]
        y_max = backbone["y_max"]
        tap_half = 0.018 if not compact else 0.014
        endcap_half = 0.022 if not compact else 0.018
        bar_width = max(5.5, linewidth * 4.2)
        tap_width = max(2.8, linewidth * 2.1)
        ax.plot(
            [spine_x, spine_x],
            [y_min, y_max],
            color=BACKBONE_BAR_COLOR,
            linewidth=bar_width,
            alpha=min(1.0, alpha),
            solid_capstyle="butt",
            zorder=zorder,
        )
        ax.plot(
            [spine_x - endcap_half, spine_x + endcap_half],
            [y_min, y_min],
            color=BACKBONE_BAR_COLOR,
            linewidth=tap_width,
            alpha=min(1.0, alpha),
            solid_capstyle="butt",
            zorder=zorder + 1,
        )
        ax.plot(
            [spine_x - endcap_half, spine_x + endcap_half],
            [y_max, y_max],
            color=BACKBONE_BAR_COLOR,
            linewidth=tap_width,
            alpha=min(1.0, alpha),
            solid_capstyle="butt",
            zorder=zorder + 1,
        )
        hub_key = backbone["hub_key"]
        if hub_key in positions:
            hub_pos = positions[hub_key]
            _plot_polyline(ax, [(spine_x, hub_pos[1]), hub_pos], color=BACKBONE_TAP_COLOR, linestyle="-", linewidth=max(1.3, linewidth), alpha=alpha, zorder=zorder + 2)
        for member_key in backbone["member_keys"]:
            if member_key == hub_key or member_key not in positions:
                continue
            x, y = positions[member_key]
            _plot_polyline(ax, [(x, y), (spine_x, y)], color=BACKBONE_TAP_COLOR, linestyle="-", linewidth=max(1.3, linewidth), alpha=alpha, zorder=zorder + 2)
            ax.plot(
                [spine_x - tap_half, spine_x + tap_half],
                [y, y],
                color=BACKBONE_BAR_COLOR,
                linewidth=tap_width,
                alpha=min(1.0, alpha),
                solid_capstyle="butt",
                zorder=zorder + 3,
            )
        label_y = y_max + (0.028 if not compact else 0.02)
        label_dx = -0.03 if backbone["side"] == "ac" else 0.03
        label_ha = "right" if backbone["side"] == "ac" else "left"
        ax.text(
            spine_x + label_dx,
            label_y,
            backbone["label"],
            fontsize=7 if not compact else 6,
            ha=label_ha,
            va="bottom",
            color=BACKBONE_BAR_COLOR,
            zorder=zorder + 4,
        )


def _build_hybrid_graph(network: HybridPowerNetwork):
    graph = nx.Graph()
    ac_buses = _alive_ac_buses(network)
    dc_buses = _alive_dc_buses(network)
    for bus in ac_buses:
        graph.add_node(_node_key("ac", int(bus.idx)), side="ac", bus=bus)
    for bus in dc_buses:
        graph.add_node(_node_key("dc", int(bus.idx)), side="dc", bus=bus)
    for left, right in _ac_topology_edges(network):
        if left != right:
            graph.add_edge(left, right, edge_type="ac")
    for left, right in _dc_topology_edges(network):
        if left != right:
            graph.add_edge(left, right, edge_type="dc")
    for left, right in _dcac_edges(network):
        if left != right:
            graph.add_edge(left, right, edge_type="dcac")
    return graph


def _normalized_y_map(graph: nx.Graph, seed: int = 17):
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        order = list(graph.nodes())
        raw_pos = {node: (0.0, -float(idx)) for idx, node in enumerate(order)}
    else:
        raw_pos = nx.spring_layout(
            graph,
            seed=seed,
            k=1.6 / math.sqrt(max(1, graph.number_of_nodes())),
            iterations=300,
        )
    ys = np.asarray([pt[1] for pt in raw_pos.values()], dtype=np.float64)
    y_min = float(ys.min())
    y_max = float(ys.max())
    dy = y_max - y_min
    if dy <= 1e-12:
        dy = 1.0
    return {key: 0.95 - 1.9 * ((float(val[1]) - y_min) / dy) for key, val in raw_pos.items()}


def _side_depths(graph: nx.Graph, side: str):
    side_graph = nx.Graph()
    side_nodes = [node for node, data in graph.nodes(data=True) if data.get("side") == side]
    side_graph.add_nodes_from(side_nodes)
    anchor_nodes = set()
    for left, right, data in graph.edges(data=True):
        edge_type = data.get("edge_type")
        if edge_type == side and left in side_graph and right in side_graph:
            side_graph.add_edge(left, right)
        elif edge_type == "dcac":
            if side == "ac" and left.startswith("ac:"):
                anchor_nodes.add(left)
            elif side == "dc" and right.startswith("dc:"):
                anchor_nodes.add(right)
    if not side_nodes:
        return {}, set()
    if not anchor_nodes:
        anchor_nodes = {max(side_nodes, key=lambda node: graph.degree(node))}
    depths = {}
    for anchor in sorted(anchor_nodes):
        lengths = nx.single_source_shortest_path_length(side_graph, anchor)
        for node, dist in lengths.items():
            prev = depths.get(node)
            if prev is None or dist < prev:
                depths[node] = dist
    max_depth = max(depths.values()) if depths else 0
    for node in side_nodes:
        depths.setdefault(node, max_depth + 1)
    return depths, anchor_nodes


def _x_from_depth(depth: int, max_depth: int, side: str) -> float:
    denom = max(1, max_depth)
    ratio = float(depth) / float(denom)
    if side == "ac":
        return -0.18 - 0.82 * ratio
    return 0.18 + 0.82 * ratio


def _spread_positions(positions, min_gap=0.07):
    grouped = {}
    for key, (x, y) in positions.items():
        bucket = round(float(x), 3)
        grouped.setdefault(bucket, []).append((key, y))
    out = dict(positions)
    for bucket, items in grouped.items():
        items.sort(key=lambda item: item[1], reverse=True)
        ys = [item[1] for item in items]
        if len(ys) <= 1:
            continue
        adjusted = [ys[0]]
        for y in ys[1:]:
            adjusted.append(min(y, adjusted[-1] - min_gap))
        for idx in range(len(adjusted) - 2, -1, -1):
            adjusted[idx] = max(adjusted[idx], adjusted[idx + 1] + min_gap)
        center_old = float(np.mean(ys))
        center_new = float(np.mean(adjusted))
        adjusted = [max(-1.0, min(1.0, y - center_new + center_old)) for y in adjusted]
        for (key, _y_old), y_new in zip(items, adjusted):
            out[key] = (positions[key][0], y_new)
    return out


def _polyline_points(start, end, style="same-side"):
    x0, y0 = start
    x1, y1 = end
    if style == "cross-side":
        xm = 0.5 * (x0 + x1)
        return [start, (xm, y0), (xm, y1), end]
    if abs(y0 - y1) <= 1e-9:
        return [start, end]
    xm = 0.5 * (x0 + x1)
    return [start, (xm, y0), (xm, y1), end]


def _polyline_midpoint(points):
    if len(points) == 2:
        return 0.5 * (points[0][0] + points[1][0]), 0.5 * (points[0][1] + points[1][1])
    mid_idx = max(0, len(points) // 2 - 1)
    p0 = points[mid_idx]
    p1 = points[mid_idx + 1]
    return 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])


def _plot_polyline(ax, points, color, linewidth=1.2, alpha=0.9, linestyle="-", zorder=1):
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=zorder)


def _name_to_bus_map(network: HybridPowerNetwork):
    mapping = {}
    for bus in _alive_ac_buses(network):
        mapping[_bus_name(bus)] = ("ac", int(bus.idx))
    for bus in _alive_dc_buses(network):
        mapping[_bus_name(bus)] = ("dc", int(bus.idx))
    return mapping


def _set_named_pos(positions, name_map, name, x, y):
    info = name_map.get(name)
    if info is None:
        return
    side, idx = info
    positions[_node_key(side, idx)] = (float(x), float(y))


def _extract_num(name: str, prefix: str) -> int:
    m = re.match(rf"{re.escape(prefix)}(\d+)", name)
    return int(m.group(1)) if m else 0


def _qinling_custom_positions(network: HybridPowerNetwork):
    name_map = _name_to_bus_map(network)
    required = {"dc_bus_720v", "grid_inv_dc", "grid_inv_ac", "ac_bus"}
    if not required.issubset(name_map):
        return None

    positions = {}
    x = {
        "src": -0.98,
        "rect": -0.76,
        "dcsw": -0.58,
        "line": -0.40,
        "dc_bus": -0.12,
        "inv_dc": 0.10,
        "conv": 0.24,
        "inv_ac": 0.38,
        "ac_bus": 0.58,
        "ac_load": 0.86,
        "ac_stub": 0.72,
        "dc_low": -0.94,
        "dc_mid": -0.74,
        "dc_tie": -0.52,
        "dc_fc": -0.86,
        "ess_line_r": 0.16,
        "ess_mid_r": 0.31,
        "ess_low_r": 0.46,
    }

    wind_names = sorted([name for name in name_map if name.startswith("wt") and name.endswith("_src")], key=lambda n: _extract_num(n, "wt"))
    wind_rows = np.linspace(0.96, 0.14, num=max(1, len(wind_names)))
    for y, src_name in zip(wind_rows, wind_names):
        idx = _extract_num(src_name, "wt")
        _set_named_pos(positions, name_map, f"wt{idx:02d}_src", x["src"], y)
        _set_named_pos(positions, name_map, f"wt{idx:02d}_rect", x["rect"], y)
        _set_named_pos(positions, name_map, f"wt{idx:02d}_dc_sw", x["dcsw"], y)
        _set_named_pos(positions, name_map, f"wt{idx:02d}_line_dc", x["line"], y)

    lower_rows = {
        "pv01": -0.08,
        "pv02": -0.26,
        "pv03": -0.44,
        "ess01": -0.62,
        "ess02": -0.80,
        "ess03": -0.98,
        "ess04": -1.16,
        "ess05": -1.34,
        "fc01": -0.18,
    }
    for prefix, y in lower_rows.items():
        if prefix.startswith("pv"):
            _set_named_pos(positions, name_map, f"{prefix}_300v", x["dc_low"], y)
            _set_named_pos(positions, name_map, f"{prefix}_dc_sw", x["dc_mid"], y)
            if prefix != "pv02":
                _set_named_pos(positions, name_map, f"{prefix}_line_dc", x["dc_tie"], y)
        elif prefix.startswith("ess"):
            _set_named_pos(positions, name_map, f"{prefix}_300v", x["ess_low_r"], y)
            _set_named_pos(positions, name_map, f"{prefix}_720v", x["ess_mid_r"], y)
            if prefix not in {"ess01", "ess03", "ess05"}:
                _set_named_pos(positions, name_map, f"{prefix}_line_dc", x["ess_line_r"], y)
        elif prefix == "fc01":
            _set_named_pos(positions, name_map, "fc01_src", x["dc_fc"], y)

    _set_named_pos(positions, name_map, "dc_bus_720v", x["dc_bus"], -0.30)
    _set_named_pos(positions, name_map, "grid_inv_dc", x["inv_dc"], -0.30)
    _set_named_pos(positions, name_map, "inv_line_dc", x["dc_tie"], -0.30)
    _set_named_pos(positions, name_map, "grid_inv_ac", x["inv_ac"], -0.30)
    _set_named_pos(positions, name_map, "ac_bus", x["ac_bus"], -0.30)

    ac_right_rows = {
        "diesel_sw": 0.18,
        "diesel_node": 0.18,
        "load2_sw": -0.02,
        "ac_load_2": -0.02,
        "h2_load_sw": -0.22,
        "h2_load": -0.22,
        "ac_load_1": -0.42,
    }
    for name, y in ac_right_rows.items():
        x_pos = x["ac_stub"] if name.endswith("_sw") else x["ac_load"]
        _set_named_pos(positions, name_map, name, x_pos, y)

    for key in list(name_map):
        if key.startswith("wt") and key.endswith("_line_dc") and key not in {f"wt{i:02d}_line_dc" for i in range(1, 11)}:
            _set_named_pos(positions, name_map, key, x["line"], 0.0)

    for key, (_side, idx) in name_map.items():
        node_key = _node_key(_side, idx)
        if node_key not in positions:
            positions[node_key] = (0.0, 0.0)
    return positions


def build_positions(network: HybridPowerNetwork):
    custom = _qinling_custom_positions(network)
    if custom is not None:
        return custom
    graph = _build_hybrid_graph(network)
    y_map = _normalized_y_map(graph)
    ac_depths, _ac_anchors = _side_depths(graph, "ac")
    dc_depths, _dc_anchors = _side_depths(graph, "dc")
    ac_max_depth = max(ac_depths.values()) if ac_depths else 0
    dc_max_depth = max(dc_depths.values()) if dc_depths else 0
    positions = {}
    for node, data in graph.nodes(data=True):
        side = data.get("side")
        y = y_map.get(node, 0.0)
        if side == "ac":
            positions[node] = (_x_from_depth(ac_depths.get(node, ac_max_depth), ac_max_depth, "ac"), y)
        else:
            positions[node] = (_x_from_depth(dc_depths.get(node, dc_max_depth), dc_max_depth, "dc"), y)
    return _spread_positions(positions)


def _draw_arrow(ax, start, end, color, linewidth, alpha=0.9, zorder=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10 + 2 * linewidth,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(patch)


def _draw_arrow_path(ax, points, color, linewidth, alpha=0.9, zorder=2):
    if len(points) < 2:
        return
    if len(points) > 2:
        _plot_polyline(ax, points[:-1], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    _draw_arrow(ax, points[-2], points[-1], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _flow_style(value, scale):
    magnitude = abs(float(value))
    ratio = magnitude / scale if scale > 1e-12 else 0.0
    width = 0.8 + 4.0 * ratio
    alpha = min(0.35 + 0.55 * ratio, 0.95)
    return width, alpha


def _converter_midpoint(start, end, offset=0.0):
    return 0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]) + offset


def _collect_flow_scale(network: HybridPowerNetwork) -> float:
    magnitudes = []
    for dev in getattr(network.ac, "branches", []):
        if _is_alive(dev):
            magnitudes.append(math.hypot(float(getattr(dev, "i_p", 0.0) or 0.0), float(getattr(dev, "i_q", 0.0) or 0.0)))
            magnitudes.append(math.hypot(float(getattr(dev, "j_p", 0.0) or 0.0), float(getattr(dev, "j_q", 0.0) or 0.0)))
    for dev in getattr(network.ac, "transformers", []):
        if _is_alive(dev):
            magnitudes.append(math.hypot(float(getattr(dev, "i_p", 0.0) or 0.0), float(getattr(dev, "i_q", 0.0) or 0.0)))
            magnitudes.append(math.hypot(float(getattr(dev, "j_p", 0.0) or 0.0), float(getattr(dev, "j_q", 0.0) or 0.0)))
    for dev in getattr(network.dc, "branches", []):
        if _is_alive(dev):
            magnitudes.append(abs(float(getattr(dev, "i_p", 0.0) or 0.0)))
            magnitudes.append(abs(float(getattr(dev, "j_p", 0.0) or 0.0)))
    for dev in getattr(network.dc, "dcdc_converters", []):
        if _is_alive(dev):
            magnitudes.append(abs(float(getattr(dev, "i_p", 0.0) or 0.0)))
            magnitudes.append(abs(float(getattr(dev, "j_p", 0.0) or 0.0)))
    for dev in getattr(network, "dcac_converters", []):
        if _is_alive(dev):
            magnitudes.append(abs(float(getattr(dev, "dc_p", 0.0) or 0.0)))
            magnitudes.append(abs(float(getattr(dev, "ac_p", 0.0) or 0.0)))
    return max(magnitudes) if magnitudes else 1.0


def _save_figure(fig, output_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bus_degree_map(network: HybridPowerNetwork):
    degree = {}

    def add(name):
        degree.setdefault(name, 0)

    def bump(name):
        degree[name] = degree.get(name, 0) + 1

    for bus in _alive_ac_buses(network):
        add(_bus_name(bus))
    for bus in _alive_dc_buses(network):
        add(_bus_name(bus))

    for dev in list(getattr(network.ac, "branches", [])) + list(getattr(network.ac, "transformers", [])):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        bump(_bus_name(_node_bus(i_node)))
        bump(_bus_name(_node_bus(j_node)))
    for dev in list(getattr(network.dc, "branches", [])) + list(getattr(network.dc, "dcdc_converters", [])):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        bump(_bus_name(_node_bus(i_node)))
        bump(_bus_name(_node_bus(j_node)))
    for dev in getattr(network, "dcac_converters", []):
        if not _is_alive(dev):
            continue
        ac_node = getattr(dev, "ac_node_obj", None)
        dc_node = getattr(dev, "dc_node_obj", None)
        if ac_node is None or dc_node is None:
            continue
        bump(_bus_name(_node_bus(ac_node)))
        bump(_bus_name(_node_bus(dc_node)))
    return degree


def _is_key_bus(bus_name: str, degree_map) -> bool:
    lowered = bus_name.lower()
    if degree_map.get(bus_name, 0) != 2:
        return True
    for token in ("grid", "dc_bus", "inv", "rect", "src"):
        if token in lowered:
            return True
    return False


def _draw_bus_layer(ax, network: HybridPowerNetwork, positions, compact: bool = False) -> None:
    degree_map = _bus_degree_map(network) if compact else {}
    for bus in _alive_ac_buses(network):
        key = _node_key("ac", int(bus.idx))
        if key not in positions:
            continue
        x, y = positions[key]
        ax.scatter([x], [y], s=150 if not compact else 135, c=AC_BUS_COLOR, edgecolors="black", linewidths=0.6, zorder=4)
        name = _bus_name(bus)
        if compact and not _is_key_bus(name, degree_map):
            continue
        label = f"{name}\nV={float(bus.voltage):.4f}"
        if not compact:
            label += f"\nA={float(bus.angle):.4f}"
        ax.text(x - 0.025, y + 0.025, label, fontsize=6.5 if not compact else 6.0, ha="right", va="bottom", color="#123b5d")

    for bus in _alive_dc_buses(network):
        key = _node_key("dc", int(bus.idx))
        if key not in positions:
            continue
        x, y = positions[key]
        ax.scatter([x], [y], s=130 if not compact else 120, marker="s", c=DC_BUS_COLOR, edgecolors="black", linewidths=0.6, zorder=4)
        name = _bus_name(bus)
        if compact and not _is_key_bus(name, degree_map):
            continue
        label = f"{name}\nV={float(bus.voltage):.4f}"
        ax.text(x + 0.025, y + 0.025, label, fontsize=6.5 if not compact else 6.0, ha="left", va="bottom", color="#8a3d00")


def draw_topology(network: HybridPowerNetwork, positions, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title("Qinling Hybrid Network Topology")
    ax.axis("off")

    _draw_backbones(ax, network, positions, alpha=0.85, linewidth=1.2, zorder=0, compact=False)

    for bus in _alive_ac_buses(network):
        key = _node_key("ac", int(bus.idx))
        if key not in positions:
            continue
        x, y = positions[key]
        ax.scatter([x], [y], s=140, c=AC_BUS_COLOR, edgecolors="black", linewidths=0.6, zorder=3)
        ax.text(x - 0.02, y + 0.02, _bus_name(bus), fontsize=7, ha="right", va="bottom", color="#123b5d")

    for bus in _alive_dc_buses(network):
        key = _node_key("dc", int(bus.idx))
        if key not in positions:
            continue
        x, y = positions[key]
        ax.scatter([x], [y], s=120, marker="s", c=DC_BUS_COLOR, edgecolors="black", linewidths=0.6, zorder=3)
        ax.text(x + 0.02, y + 0.02, _bus_name(bus), fontsize=7, ha="left", va="bottom", color="#8a3d00")

    for dev in getattr(network.ac, "branches", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("ac", int(i_bus.idx))
        j_key = _node_key("ac", int(j_bus.idx))
        if i_key in positions and j_key in positions:
            _plot_polyline(ax, _polyline_points(positions[i_key], positions[j_key]), color=AC_BRANCH_COLOR, linewidth=1.2, alpha=0.85, zorder=1)

    for dev in getattr(network.ac, "transformers", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("ac", int(i_bus.idx))
        j_key = _node_key("ac", int(j_bus.idx))
        if i_key in positions and j_key in positions:
            _plot_polyline(ax, _polyline_points(positions[i_key], positions[j_key]), color=AC_XFMR_COLOR, linewidth=1.4, alpha=0.9, zorder=1)

    for dev in getattr(network.dc, "branches", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("dc", int(i_bus.idx))
        j_key = _node_key("dc", int(j_bus.idx))
        if i_key in positions and j_key in positions:
            _plot_polyline(ax, _polyline_points(positions[i_key], positions[j_key]), color=DC_BRANCH_COLOR, linewidth=1.2, alpha=0.85, zorder=1)

    for dev in getattr(network.dc, "dcdc_converters", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("dc", int(i_bus.idx))
        j_key = _node_key("dc", int(j_bus.idx))
        if i_key in positions and j_key in positions:
            _plot_polyline(ax, _polyline_points(positions[i_key], positions[j_key]), color=DCDC_COLOR, linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)

    for idx, conv in enumerate(getattr(network, "dcac_converters", [])):
        if not _is_alive(conv):
            continue
        ac_node = getattr(conv, "ac_node_obj", None)
        dc_node = getattr(conv, "dc_node_obj", None)
        if ac_node is None or dc_node is None:
            continue
        ac_bus = _node_bus(ac_node)
        dc_bus = _node_bus(dc_node)
        ac_key = _node_key("ac", int(ac_bus.idx))
        dc_key = _node_key("dc", int(dc_bus.idx))
        if ac_key not in positions or dc_key not in positions:
            continue
        start = positions[ac_key]
        end = positions[dc_key]
        points = _polyline_points(start, end, style="cross-side")
        _plot_polyline(ax, points, color=DCAC_COLOR, linewidth=1.4, alpha=0.9, zorder=2)
        mx, my = _polyline_midpoint(points)
        my += 0.015 * ((idx % 3) - 1)
        ax.scatter([mx], [my], s=40, c=DCAC_COLOR, marker="D", edgecolors="black", linewidths=0.4, zorder=4)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AC_BUS_COLOR, markeredgecolor="black", label="AC bus", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=DC_BUS_COLOR, markeredgecolor="black", label="DC bus", markersize=8),
        Line2D([0], [0], color=AC_BRANCH_COLOR, lw=1.4, label="AC branch"),
        Line2D([0], [0], color=AC_XFMR_COLOR, lw=1.6, label="AC transformer"),
        Line2D([0], [0], color=DC_BRANCH_COLOR, lw=1.4, label="DC branch"),
        Line2D([0], [0], color=DCDC_COLOR, lw=1.4, linestyle="--", label="DCDC converter"),
        Line2D([0], [0], color=DCAC_COLOR, lw=1.6, label="DCAC converter"),
        Line2D([0], [0], color=BACKBONE_BAR_COLOR, lw=3.0, label="main switch/break bus"),
    ]
    ax.legend(handles=legend, loc="upper center", ncol=4, frameon=False)
    _save_figure(fig, output_stem)


def draw_powerflow(network: HybridPowerNetwork, positions, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_title("Qinling Hybrid Power Flow")
    ax.axis("off")
    flow_scale = _collect_flow_scale(network)
    _draw_backbones(ax, network, positions, alpha=0.55, linewidth=0.95, zorder=0, compact=False)
    _draw_bus_layer(ax, network, positions, compact=False)

    def draw_ac_flow(dev, color):
        if not _is_alive(dev):
            return
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            return
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("ac", int(i_bus.idx))
        j_key = _node_key("ac", int(j_bus.idx))
        if i_key not in positions or j_key not in positions:
            return
        start = positions[i_key]
        end = positions[j_key]
        forward = float(getattr(dev, "i_p", 0.0) or 0.0) >= 0.0
        p = float(getattr(dev, "i_p" if forward else "j_p", 0.0) or 0.0)
        q = float(getattr(dev, "i_q" if forward else "j_q", 0.0) or 0.0)
        width, alpha = _flow_style(math.hypot(p, q), flow_scale)
        points = _polyline_points(start if forward else end, end if forward else start)
        _draw_arrow_path(ax, points, color=color, linewidth=width, alpha=alpha)
        mx, my = _polyline_midpoint(points)
        ax.text(mx, my, f"P={p:.3f}\nQ={q:.3f}", fontsize=6, ha="center", va="center", color=color)

    for dev in getattr(network.ac, "branches", []):
        draw_ac_flow(dev, "#245b8f")
    for dev in getattr(network.ac, "transformers", []):
        draw_ac_flow(dev, "#7d4aa8")

    for dev in getattr(network.dc, "branches", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("dc", int(i_bus.idx))
        j_key = _node_key("dc", int(j_bus.idx))
        if i_key not in positions or j_key not in positions:
            continue
        start = positions[i_key]
        end = positions[j_key]
        forward = float(getattr(dev, "i_p", 0.0) or 0.0) >= 0.0
        p = float(getattr(dev, "i_p" if forward else "j_p", 0.0) or 0.0)
        width, alpha = _flow_style(abs(p), flow_scale)
        points = _polyline_points(start if forward else end, end if forward else start)
        _draw_arrow_path(ax, points, color="#c76b00", linewidth=width, alpha=alpha)
        mx, my = _polyline_midpoint(points)
        ax.text(mx, my, f"P={p:.3f}", fontsize=6, ha="center", va="center", color="#8a3d00")

    for dev in getattr(network.dc, "dcdc_converters", []):
        if not _is_alive(dev):
            continue
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            continue
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("dc", int(i_bus.idx))
        j_key = _node_key("dc", int(j_bus.idx))
        if i_key not in positions or j_key not in positions:
            continue
        start = positions[i_key]
        end = positions[j_key]
        p = float(getattr(dev, "i_p", 0.0) or 0.0)
        width, alpha = _flow_style(abs(p), flow_scale)
        points = _polyline_points(start if p >= 0.0 else end, end if p >= 0.0 else start)
        _draw_arrow_path(ax, points, color="#b55a00", linewidth=width, alpha=alpha)
        mx, my = _polyline_midpoint(points)
        ax.text(mx, my, f"DCDC\nP={p:.3f}", fontsize=6, ha="center", va="center", color="#8a3d00")

    for idx, conv in enumerate(getattr(network, "dcac_converters", [])):
        if not _is_alive(conv):
            continue
        ac_node = getattr(conv, "ac_node_obj", None)
        dc_node = getattr(conv, "dc_node_obj", None)
        if ac_node is None or dc_node is None:
            continue
        ac_bus = _node_bus(ac_node)
        dc_bus = _node_bus(dc_node)
        ac_key = _node_key("ac", int(ac_bus.idx))
        dc_key = _node_key("dc", int(dc_bus.idx))
        if ac_key not in positions or dc_key not in positions:
            continue
        start = positions[ac_key]
        end = positions[dc_key]
        p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0)
        p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0)
        q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0)
        width, alpha = _flow_style(max(abs(p_dc), abs(p_ac)), flow_scale)
        forward = p_dc >= 0.0
        points = _polyline_points(start if forward else end, end if forward else start, style="cross-side")
        _draw_arrow_path(ax, points, color=DCAC_COLOR, linewidth=width, alpha=alpha, zorder=3)
        mx, my = _polyline_midpoint(points)
        my += 0.02 * ((idx % 4) - 1.5)
        ax.scatter([mx], [my], s=48, c=DCAC_COLOR, marker="D", edgecolors="black", linewidths=0.4, zorder=5)
        label = f"{getattr(conv, 'name', f'dcac_{idx}')}\nPdc={p_dc:.3f}\nPac={p_ac:.3f}\nQac={q_ac:.3f}"
        ax.text(mx, my, label, fontsize=5.8, ha="center", va="center", color="#176117")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AC_BUS_COLOR, markeredgecolor="black", label="AC bus", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=DC_BUS_COLOR, markeredgecolor="black", label="DC bus", markersize=8),
        Line2D([0], [0], color="#245b8f", lw=2.0, label="AC flow"),
        Line2D([0], [0], color="#c76b00", lw=2.0, label="DC flow"),
        Line2D([0], [0], color=DCAC_COLOR, lw=2.0, label="DCAC flow"),
        Line2D([0], [0], color=BACKBONE_BAR_COLOR, lw=3.0, label="main switch/break bus"),
    ]
    ax.legend(handles=legend, loc="upper center", ncol=5, frameon=False)
    _save_figure(fig, output_stem)


def draw_powerflow_compact(network: HybridPowerNetwork, positions, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title("Qinling Hybrid Power Flow (Compact)")
    ax.axis("off")
    flow_scale = _collect_flow_scale(network)
    compact_threshold = 0.18 * flow_scale
    _draw_backbones(ax, network, positions, alpha=0.75, linewidth=1.0, zorder=0, compact=True)
    _draw_bus_layer(ax, network, positions, compact=True)

    def should_label(value, force=False):
        return force or abs(float(value)) >= compact_threshold

    def draw_ac_flow(dev, color):
        if not _is_alive(dev):
            return
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            return
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("ac", int(i_bus.idx))
        j_key = _node_key("ac", int(j_bus.idx))
        if i_key not in positions or j_key not in positions:
            return
        start = positions[i_key]
        end = positions[j_key]
        forward = float(getattr(dev, "i_p", 0.0) or 0.0) >= 0.0
        p = float(getattr(dev, "i_p" if forward else "j_p", 0.0) or 0.0)
        q = float(getattr(dev, "i_q" if forward else "j_q", 0.0) or 0.0)
        width, alpha = _flow_style(math.hypot(p, q), flow_scale)
        points = _polyline_points(start if forward else end, end if forward else start)
        _draw_arrow_path(ax, points, color=color, linewidth=max(1.1, width), alpha=alpha)
        if should_label(math.hypot(p, q)):
            mx, my = _polyline_midpoint(points)
            ax.text(mx, my, f"P={p:.3f}\nQ={q:.3f}", fontsize=5.8, ha="center", va="center", color=color)

    def draw_dc_flow(dev, color, label_prefix="", force_label=False):
        if not _is_alive(dev):
            return
        i_node = getattr(dev, "i_node_obj", None)
        j_node = getattr(dev, "j_node_obj", None)
        if i_node is None or j_node is None:
            return
        i_bus = _node_bus(i_node)
        j_bus = _node_bus(j_node)
        i_key = _node_key("dc", int(i_bus.idx))
        j_key = _node_key("dc", int(j_bus.idx))
        if i_key not in positions or j_key not in positions:
            return
        start = positions[i_key]
        end = positions[j_key]
        p = float(getattr(dev, "i_p", 0.0) or 0.0)
        points = _polyline_points(start if p >= 0.0 else end, end if p >= 0.0 else start)
        width, alpha = _flow_style(abs(p), flow_scale)
        _draw_arrow_path(ax, points, color=color, linewidth=max(1.1, width), alpha=alpha)
        if should_label(p, force=force_label):
            mx, my = _polyline_midpoint(points)
            text = f"{label_prefix}P={p:.3f}" if label_prefix else f"P={p:.3f}"
            ax.text(mx, my, text, fontsize=5.8, ha="center", va="center", color=color)

    for dev in getattr(network.ac, "branches", []):
        draw_ac_flow(dev, "#245b8f")
    for dev in getattr(network.ac, "transformers", []):
        draw_ac_flow(dev, "#7d4aa8")
    for dev in getattr(network.dc, "branches", []):
        draw_dc_flow(dev, "#c76b00")
    for dev in getattr(network.dc, "dcdc_converters", []):
        draw_dc_flow(dev, "#b55a00", label_prefix="DCDC\n", force_label=True)

    for idx, conv in enumerate(getattr(network, "dcac_converters", [])):
        if not _is_alive(conv):
            continue
        ac_node = getattr(conv, "ac_node_obj", None)
        dc_node = getattr(conv, "dc_node_obj", None)
        if ac_node is None or dc_node is None:
            continue
        ac_bus = _node_bus(ac_node)
        dc_bus = _node_bus(dc_node)
        ac_key = _node_key("ac", int(ac_bus.idx))
        dc_key = _node_key("dc", int(dc_bus.idx))
        if ac_key not in positions or dc_key not in positions:
            continue
        start = positions[ac_key]
        end = positions[dc_key]
        p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0)
        p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0)
        q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0)
        points = _polyline_points(start if p_dc >= 0.0 else end, end if p_dc >= 0.0 else start, style="cross-side")
        width, alpha = _flow_style(max(abs(p_dc), abs(p_ac)), flow_scale)
        _draw_arrow_path(ax, points, color=DCAC_COLOR, linewidth=max(1.3, width), alpha=alpha, zorder=3)
        mx, my = _polyline_midpoint(points)
        ax.scatter([mx], [my], s=42, c=DCAC_COLOR, marker="D", edgecolors="black", linewidths=0.4, zorder=5)
        if should_label(max(abs(p_dc), abs(p_ac)), force=True):
            label = f"{getattr(conv, 'name', f'dcac_{idx}')}\nPdc={p_dc:.3f}\nPac={p_ac:.3f}"
            if abs(q_ac) >= compact_threshold:
                label += f"\nQac={q_ac:.3f}"
            ax.text(mx, my + 0.02, label, fontsize=5.8, ha="center", va="bottom", color="#176117")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AC_BUS_COLOR, markeredgecolor="black", label="AC bus", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=DC_BUS_COLOR, markeredgecolor="black", label="DC bus", markersize=8),
        Line2D([0], [0], color="#245b8f", lw=2.0, label="AC flow"),
        Line2D([0], [0], color="#c76b00", lw=2.0, label="DC flow"),
        Line2D([0], [0], color=DCAC_COLOR, lw=2.0, label="DCAC flow"),
        Line2D([0], [0], color=BACKBONE_BAR_COLOR, lw=3.0, label="main switch/break bus"),
    ]
    ax.legend(handles=legend, loc="upper center", ncol=6, frameon=False)
    _save_figure(fig, output_stem)


def build_network(case_path: Path, flat_start: bool) -> HybridPowerNetwork:
    network = HybridPowerNetwork.read_from_file(case_path)
    if flat_start:
        _flat_start_network(network)
    network.prepare(verbose=False)
    return network


def run_power_flow(network: HybridPowerNetwork) -> HybridPowerFlowCalc:
    calc = HybridPowerFlowCalc(network, verbose=False, result_mode="full")
    calc.prepare()
    rc = calc.run()
    if rc != 0 or not getattr(calc, "converged", False):
        raise RuntimeError(
            f"Hybrid LF did not converge: rc={rc}, converged={getattr(calc, 'converged', None)}, iter={getattr(calc, 'iterations', None)}"
        )
    return calc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate layered Qinling topology and power-flow diagrams.")
    parser.add_argument("case", nargs="?", default=str(DEFAULT_CASE), help="Hybrid E file path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--flat-start", action="store_true", help="Use flat start before LF solve.")
    args = parser.parse_args(argv)

    case_path = Path(args.case).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_network = build_network(case_path, flat_start=False)
    positions = build_positions(topology_network)
    draw_topology(topology_network, positions, output_dir / "qinling_topology")

    powerflow_network = build_network(case_path, flat_start=bool(args.flat_start))
    run_power_flow(powerflow_network)
    draw_powerflow(powerflow_network, positions, output_dir / "qinling_powerflow")
    draw_powerflow_compact(powerflow_network, positions, output_dir / "qinling_powerflow_compact")

    print(output_dir / "qinling_topology.svg")
    print(output_dir / "qinling_topology.png")
    print(output_dir / "qinling_powerflow.svg")
    print(output_dir / "qinling_powerflow.png")
    print(output_dir / "qinling_powerflow_compact.svg")
    print(output_dir / "qinling_powerflow_compact.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
