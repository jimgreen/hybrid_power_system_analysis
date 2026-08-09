from pathlib import Path
import xml.etree.ElementTree as ET


def _sample_station_e() -> str:
    return "\n".join(
        [
            "<ACNode>",
            "@ idx name vbase voltage angle isl run_stat",
            "# 1 bus 110 110 0 0 1",
            "# 2 line1 110 110 0 0 1",
            "# 3 line2 110 110 0 0 1",
            "# 4 gen1 110 110 0 0 1",
            "</ACNode>",
            "",
            "<ACBranch>",
            "@ idx name i_node j_node r x b run_stat",
            "# 1 bay_line1 1 2 0.01 0.05 0 1",
            "# 2 bay_line2 1 3 0.01 0.05 0 1",
            "# 3 bay_gen1 1 4 0.01 0.05 0 1",
            "</ACBranch>",
            "",
            "<ACGenerator>",
            "@ idx name node control_type p_set q_set v_set alpha run_stat",
            "# 1 gen1_unit 4 V 0 0 110 1 1",
            "</ACGenerator>",
            "",
        ]
    )


def test_station_svg_routes_orthogonal_svg(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    e_file = tmp_path / "station.e"
    svg_file = tmp_path / "station.svg"
    e_file.write_text(_sample_station_e(), encoding="utf-8")

    result = render_station_svg_file(e_file, svg_file)

    assert result.node_count == 4
    assert result.edge_count == 3
    assert svg_file.exists()
    text = svg_file.read_text(encoding="utf-8")
    assert "厂站接线图" in text
    assert 'data-name="bay_line1"' in text

    root = ET.fromstring(text)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    polylines = root.findall(".//svg:polyline", ns)
    assert len(polylines) >= 3
    for polyline in polylines:
        if polyline.attrib.get("class") != "branch-route":
            continue
        points = [
            tuple(float(v) for v in pair.split(","))
            for pair in polyline.attrib["points"].split()
        ]
        assert len(points) >= 2
        for left, right in zip(points, points[1:]):
            assert left[0] == right[0] or left[1] == right[1]


def test_station_svg_reports_no_crossings_for_simple_radial_case(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import parse_station_graph, layout_station_graph

    e_file = tmp_path / "station.e"
    e_file.write_text(_sample_station_e(), encoding="utf-8")

    graph = parse_station_graph(e_file)
    layout = layout_station_graph(graph)

    assert layout.crossing_count == 0


def test_station_svg_layout_is_independent_of_device_names(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import parse_station_graph, layout_station_graph

    original_file = tmp_path / "original.e"
    renamed_file = tmp_path / "renamed.e"
    original_file.write_text(_sample_station_e(), encoding="utf-8")
    renamed = (
        _sample_station_e()
        .replace("bus", "opaque_node_a")
        .replace("line1", "opaque_node_b")
        .replace("line2", "opaque_node_c")
        .replace("gen1", "opaque_node_d")
        .replace("bay_opaque_node_b", "edge_alpha")
        .replace("bay_opaque_node_c", "edge_beta")
        .replace("bay_opaque_node_d", "edge_gamma")
        .replace("opaque_node_d_unit", "source_alpha")
    )
    renamed_file.write_text(renamed, encoding="utf-8")

    original_graph = parse_station_graph(original_file)
    renamed_graph = parse_station_graph(renamed_file)
    original_layout = layout_station_graph(original_graph)
    renamed_layout = layout_station_graph(renamed_graph)

    assert original_layout.positions == renamed_layout.positions
    assert original_layout.routes == renamed_layout.routes
    assert original_layout.crossing_count == renamed_layout.crossing_count


def test_station_svg_qinling_uses_low_crossing_station_layout():
    from hybrid_power_system_analysis.drawcore.station_svg import parse_station_graph, layout_station_graph

    graph = parse_station_graph(Path("data/model/hybrid/qinling.e"))
    layout = layout_station_graph(graph)

    assert layout.crossing_count <= 5


def test_station_svg_qinling_ac_bus_only_spans_connected_bays(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    ac_bus = root.find('.//svg:line[@class="busbar-ac"]', ns)
    assert ac_bus is not None
    assert float(ac_bus.attrib["x1"]) > 720.0


def test_station_svg_switch_and_breaker_are_two_terminal_devices(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    for name in ("sw_diesel_ac", "sw_wt01_dc"):
        group = root.find(f'.//svg:g[@data-device-name="{name}"]', ns)
        assert group is not None
        assert group.attrib["class"] == "two-terminal-device"
        terminals = group.findall('svg:circle[@class="device-terminal"]', ns)
        assert len(terminals) == 2


def test_station_svg_converter_is_rectangle_without_terminals(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    for name in ("wt01_rect", "pv01_dcdc"):
        group = root.find(f'.//svg:g[@data-device-name="{name}"]', ns)
        assert group is not None
        assert group.attrib["class"] == "converter-device"
        assert group.find('svg:rect[@class="converter"]', ns) is not None
        assert group.findall('svg:circle[@class="device-terminal"]', ns) == []
        assert root.findall(f'.//svg:polyline[@data-name="{name}"]', ns) == []


def test_station_svg_converter_terminal_node_labels_are_hidden(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    text_values = {text.text for text in root.findall(".//svg:text", ns)}

    assert "grid_inv_ac" not in text_values
    assert "grid_inv_dc" not in text_values
    assert "wt01_rect" not in text_values
    assert "pv01_300v" not in text_values


def test_station_svg_element_labels_are_hidden(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    hidden_classes = {"node-label", "node-meta", "edge-label", "symbol-label", "injection-label"}
    visible_element_labels = [
        text
        for text in root.findall(".//svg:text", ns)
        if text.attrib.get("class") in hidden_classes
    ]

    assert visible_element_labels == []


def test_station_svg_ac_load_triangles_point_toward_connection(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    for name in ("load_ac_1", "load_ac_2", "h2_load"):
        group = root.find(f'.//svg:g[@data-injection-name="{name}"]', ns)
        assert group is not None
        polygon = group.find('svg:polygon[@class="injection-load"]', ns)
        assert polygon is not None
        points = [
            tuple(float(value) for value in pair.split(","))
            for pair in polygon.attrib["points"].split()
        ]
        apex = min(points, key=lambda point: point[1])
        lead = group.find('svg:polyline[@class="injection-lead"]', ns)
        assert lead is not None
        lead_end = tuple(float(value) for value in lead.attrib["points"].split()[-1].split(","))

        assert lead_end == apex


def test_station_svg_generator_and_load_are_single_terminal_injections(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    for name in ("wt01_10kw", "diesel_300kw", "load_ac_1", "pv01_vsrc"):
        group = root.find(f'.//svg:g[@data-injection-name="{name}"]', ns)
        assert group is not None
        assert group.attrib["class"] == "single-terminal-injection"
        terminals = group.findall('svg:circle[@class="injection-terminal"]', ns)
        assert len(terminals) == 1


def test_station_svg_qinling_layout_is_compact(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    assert float(root.attrib["width"]) <= 1250.0
    assert float(root.attrib["height"]) <= 660.0


def test_station_svg_qinling_spreads_dc_injection_bays_across_lower_area():
    from hybrid_power_system_analysis.drawcore.station_svg import parse_station_graph, layout_station_graph

    graph = parse_station_graph(Path("data/model/hybrid/qinling.e"))
    layout = layout_station_graph(graph)
    dc_bus_y = next(
        layout.positions[key][1]
        for key, node in graph.nodes.items()
        if node.side == "DC" and node.is_bus
    )
    dc_injection_positions = [
        layout.positions[injection.node]
        for injection in graph.injections
        if injection.side == "DC" and not graph.nodes[injection.node].is_bus
    ]

    assert max(x for x, _y in dc_injection_positions) - min(x for x, _y in dc_injection_positions) >= 500.0
    assert all(y > dc_bus_y for _x, y in dc_injection_positions)


def test_station_svg_connection_nodes_are_not_displayed(tmp_path):
    from hybrid_power_system_analysis.drawcore.station_svg import render_station_svg_file

    svg_file = tmp_path / "qinling.svg"
    render_station_svg_file(Path("data/model/hybrid/qinling.e"), svg_file)

    root = ET.fromstring(svg_file.read_text(encoding="utf-8"))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    connection_nodes = [
        circle
        for circle in root.findall(".//svg:circle", ns)
        if circle.attrib.get("class") in {"node-ac", "node-dc"}
    ]

    assert connection_nodes == []
