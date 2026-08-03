import unittest
from pathlib import Path

import numpy as np


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _build_auto_ph_ppc(node_ids, branches, generators):
    from model.ac_array_model import build_ac_ppc_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["IEEE", "auto_ph", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[node_id, f"bus_{node_id}", 380, 380, 0, 1] for node_id in node_ids],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [
                [row + 1, f"branch_{i_node}_{j_node}", i_node, j_node, 0.01, 0.1, 0.0, 1]
                for row, (i_node, j_node) in enumerate(branches)
            ],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_max run_stat",
            [
                [idx, f"gen_{idx}", node, control, p_set, 0.0, 380, alpha, p_max, 1]
                for idx, node, control, p_set, alpha, p_max in generators
            ],
        ),
    }
    return build_ac_ppc_from_efile_rows(Path("auto_ph.e"), rows)


def _build_ac_island_ppc(
    *,
    breaker_status=0,
    downstream_node_run_stat=1,
    include_real_bus=True,
    real_bus_node=1,
):
    from model.ac_array_model import build_ac_ppc_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "rooted_ac", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [
                [1, "main_bus", 380, 380, 0, 1],
                [2, "source_branch", 380, 380, 0, downstream_node_run_stat],
            ],
        ),
        "ACBreak": _table(
            "idx name i_node j_node status run_stat",
            [[1, "source_breaker", 1, 2, breaker_status, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [
                [1, "main_slack", 1, "PH", 0, 0, 380, 1, 1],
                [2, "isolated_slack", 2, "PH", 0, 0, 380, 1, 1],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "main_load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1]],
        ),
    }
    if include_real_bus:
        rows["ACRealBs"] = _table(
            "idx name node run_stat",
            [[1, "declared_real_bus", real_bus_node, 1]],
        )
    return build_ac_ppc_from_efile_rows(Path("ac_island.e"), rows)


def _build_dc_island_ppc(*, breaker_status=0, real_bus_node=1):
    from model.dc_array_model import build_dc_ppc_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "rooted_dc", 100, "V", "kW", "A"]]),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [
                [1, "main_bus", 750, 750, 1],
                [2, "converter_grid_side", 750, 750, 1],
                [3, "converter_source_side", 400, 400, 1],
                [4, "source_node", 400, 400, 1],
            ],
        ),
        "DCRealBs": _table(
            "idx name node run_stat",
            [[1, "declared_main_bus", 1, 1]],
        ),
        "DCBreak": _table(
            "idx name i_node j_node status run_stat",
            [[1, "source_breaker", 1, 2, breaker_status, 1]],
        ),
        "DCBranch": _table(
            "idx name i_node j_node r run_stat",
            [[1, "source_line", 3, 4, 0.01, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [
                [1, "main_voltage_source", 1, "V", 750, 0, 0, 1],
                [2, "isolated_power_source", 4, "P", 400, 10, 0, 1],
            ],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "main_load", 1, 10, 1, 0, 0, 1]],
        ),
        "DCDCConverter": _table(
            "idx name i_node j_node i_control_type j_control_type p_set i_set v_set run_stat r1 r2",
            [[1, "source_converter", 3, 2, "V", "NONE", 0, 0, 400, 1, 0, 0]],
        ),
    }
    rows["DCRealBs"]["rows"][0][2] = real_bus_node
    return build_dc_ppc_from_efile_rows(Path("dc_island.e"), rows)


def _build_hybrid_island_ppc(
    *,
    dc_breaker_status=1,
    connect_ac_terminal=False,
    ac_control_type="PH",
    dc_control_type="NONE",
):
    from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "rooted_hybrid", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [
                [1, "ac_main_bus", 380, 380, 0, 1],
                [2, "converter_ac_terminal", 380, 380, 0, 1],
            ],
        ),
        "ACRealBs": _table("idx name node run_stat", [[1, "ac_root", 1, 1]]),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [
                [1, "ac_main_slack", 1, "PH", 0, 0, 380, 1, 1],
                [2, "converter_side_source", 2, "PQ", 5, 0, 380, 1, 1],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "ac_main_load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [
                [1, "dc_main_bus", 750, 750, 1],
                [2, "converter_dc_terminal", 750, 750, 1],
            ],
        ),
        "DCRealBs": _table("idx name node run_stat", [[1, "dc_root", 1, 1]]),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "dc_main_voltage_source", 1, "V", 750, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "dc_main_load", 1, 10, 1, 0, 0, 1]],
        ),
        "DCBreak": _table(
            "idx name i_node j_node status run_stat",
            [[1, "converter_breaker", 1, 2, dc_breaker_status, 1]],
        ),
        "DCACConverter": _table(
            "idx name ac_node dc_node ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat r1 r2",
            [
                [
                    1,
                    "grid_converter",
                    2,
                    2,
                    ac_control_type,
                    dc_control_type,
                    0,
                    0,
                    380,
                    750,
                    1,
                    0,
                    0,
                ]
            ],
        ),
    }
    if connect_ac_terminal:
        rows["ACBranch"] = _table(
            "idx name i_node j_node r x b run_stat",
            [[1, "converter_ac_tie", 1, 2, 0.01, 0.1, 0.0, 1]],
        )
    return build_hybrid_ppc_with_topology_from_efile_rows(Path("hybrid_island.e"), rows)


def _build_acac_source_to_unreferenced_load_ppc():
    from model.ac_array_model import build_ac_ppc_from_efile_rows
    from model.ppc_topology import ensure_ac_ppc_topology

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "acac_local_ref", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "source-node", 380, 380, 0, 1], [2, "load-node", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "source-only-slack", 1, "PH", 0, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "unreferenced-load", 2, 10, 1, 0, 0, 2, 1, 0, 0, 1]],
        ),
        "ACACConverter": _table(
            "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
            [[1, "pq-link", 1, 2, 0, 0, "PQ", "PQ", 0, 0, 0, 380, 380, 1]],
        ),
    }
    return ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("acac_local_ref.e"), rows))


def _build_acac_ph_reference_hybrid_ppc():
    from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "acac_ph_ref", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "slack-node", 380, 380, 0, 1], [2, "ph-node", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "slack-source", 1, "PH", 0, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [
                [1, "slack-side-load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1],
                [2, "ph-side-load", 2, 10, 1, 0, 0, 2, 1, 0, 0, 1],
            ],
        ),
        "ACACConverter": _table(
            "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
            [[1, "pq-ph-link", 1, 2, 0, 0, "PQ", "PH", 0, 0, 0, 380, 380, 1]],
        ),
    }
    return build_hybrid_ppc_with_topology_from_efile_rows(Path("acac_ph_ref.e"), rows)


def _build_acac_ph_with_pv_hybrid_ppc():
    from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "acac_ph_pv", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "pq-node", 380, 380, 0, 1], [2, "ph-pv-node", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "ph-side-pv", 2, "PV", 5, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [
                [1, "pq-side-load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1],
                [2, "ph-side-load", 2, 10, 1, 0, 0, 2, 1, 0, 0, 1],
            ],
        ),
        "ACACConverter": _table(
            "idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
            [[1, "pq-ph-link", 1, 2, 0, 0, "PQ", "PH", 0, 0, 0, 380, 380, 1]],
        ),
    }
    return build_hybrid_ppc_with_topology_from_efile_rows(Path("acac_ph_pv.e"), rows)


def _build_dcac_ph_with_pv_hybrid_ppc():
    from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "dcac_ph_pv", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "converter-ac-node", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "converter-side-pv", 1, "PV", 5, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "converter-ac-load", 1, 10, 1, 0, 0, 2, 1, 0, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [[1, "converter-dc-node", 750, 750, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "converter-dc-load", 1, 10, 1, 0, 0, 1]],
        ),
        "DCACConverter": _table(
            "idx name ac_node dc_node ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat r1 r2",
            [[1, "ph-none-link", 1, 1, "PH", "NONE", 0, 0, 380, 750, 1, 0, 0]],
        ),
    }
    return build_hybrid_ppc_with_topology_from_efile_rows(Path("dcac_ph_pv.e"), rows)


def _build_dcac_v_with_dcdc_hybrid_ppc():
    from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "dcac_v_dcdc", 100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "converter-ac-node", 380, 380, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [
                [1, "dc-source-node", 750, 750, 1],
                [2, "converter-dc-node", 750, 750, 1],
            ],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "dc-source", 1, "V", 750, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "converter-side-load", 2, 10, 1, 0, 0, 1]],
        ),
        "DCDCConverter": _table(
            "idx name i_node j_node i_control_type j_control_type p_set i_set v_set run_stat r1 r2",
            [[1, "source-load-link", 1, 2, "P", "NONE", 0, 0, 750, 1, 0, 0]],
        ),
        "DCACConverter": _table(
            "idx name ac_node dc_node ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat r1 r2",
            [[1, "none-v-link", 1, 2, "NONE", "V", 0, 0, 380, 750, 1, 0, 0]],
        ),
    }
    return build_hybrid_ppc_with_topology_from_efile_rows(Path("dcac_v_dcdc.e"), rows)


def _build_dcdc_source_to_unreferenced_load_ppc():
    from model.dc_array_model import build_dc_ppc_from_efile_rows
    from model.ppc_topology import ensure_dc_ppc_topology

    rows = {
        "Model": _table("path name p_base u_unit p_unit i_unit", [["test", "dcdc_local_ref", 100, "V", "kW", "A"]]),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [[1, "source-node", 750, 750, 1], [2, "load-node", 750, 750, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "source-only-voltage", 1, "V", 750, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "unreferenced-load", 2, 10, 1, 0, 0, 1]],
        ),
        "DCDCConverter": _table(
            "idx name i_node j_node i_control_type j_control_type p_set i_set v_set run_stat r1 r2",
            [[1, "none-link", 1, 2, "P", "NONE", 0, 0, 750, 1, 0, 0]],
        ),
    }
    return ensure_dc_ppc_topology(build_dc_ppc_from_efile_rows(Path("dcdc_local_ref.e"), rows))


class TopologyHelperTest(unittest.TestCase):
    def test_ac_single_balance_source_without_other_generation_or_load_is_dead(self):
        from model.topology import prepare_ac_topology_ppc

        arrays = prepare_ac_topology_ppc(_build_ac_island_ppc(breaker_status=0))

        self.assertEqual([True, False], arrays.island_alive_mask.tolist())
        self.assertEqual([True, False], arrays.devices["gen"].alive_mask.tolist())
        self.assertEqual([True, False], arrays.node_alive_mask.tolist())

    def test_ac_real_bus_metadata_does_not_select_which_island_is_alive(self):
        from model.topology import prepare_ac_topology_ppc

        root_on_loaded_island = prepare_ac_topology_ppc(
            _build_ac_island_ppc(breaker_status=0, real_bus_node=1)
        )
        root_on_source_only_island = prepare_ac_topology_ppc(
            _build_ac_island_ppc(breaker_status=0, real_bus_node=2)
        )
        no_real_bus = prepare_ac_topology_ppc(
            _build_ac_island_ppc(breaker_status=0, include_real_bus=False)
        )

        expected = [True, False]
        self.assertEqual(expected, root_on_loaded_island.island_alive_mask.tolist())
        self.assertEqual(expected, root_on_source_only_island.island_alive_mask.tolist())
        self.assertEqual(expected, no_real_bus.island_alive_mask.tolist())

    def test_ac_closed_breaker_restores_source_branch_to_loaded_island(self):
        from model.topology import prepare_ac_topology_ppc

        arrays = prepare_ac_topology_ppc(_build_ac_island_ppc(breaker_status=1))

        self.assertEqual([True], arrays.island_alive_mask.tolist())
        self.assertEqual([True, True], arrays.devices["gen"].alive_mask.tolist())

    def test_ac_island_viability_honors_node_run_state(self):
        from model.topology import prepare_ac_topology_ppc

        arrays = prepare_ac_topology_ppc(
            _build_ac_island_ppc(breaker_status=1, downstream_node_run_stat=0)
        )

        self.assertEqual([True], arrays.island_alive_mask.tolist())
        self.assertEqual([True, False], arrays.devices["gen"].alive_mask.tolist())
        self.assertEqual([True, False], arrays.node_alive_mask.tolist())

    def test_dc_converter_reference_without_balance_source_or_load_is_dead(self):
        from model.topology import prepare_dc_topology_ppc

        open_arrays = prepare_dc_topology_ppc(_build_dc_island_ppc(breaker_status=0))
        closed_arrays = prepare_dc_topology_ppc(_build_dc_island_ppc(breaker_status=1))

        self.assertEqual([True, False, False], open_arrays.island_alive_mask.tolist())
        self.assertEqual([True, False], open_arrays.devices["gen"].alive_mask.tolist())
        self.assertEqual([False], open_arrays.devices["dcdc"].alive_mask.tolist())
        self.assertEqual([True, True], closed_arrays.island_alive_mask.tolist())
        self.assertEqual([True, True], closed_arrays.devices["gen"].alive_mask.tolist())
        self.assertEqual([True], closed_arrays.devices["dcdc"].alive_mask.tolist())

    def test_ac_object_topology_applies_device_composition_viability(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.topology import prepare_ac_topology

        network = build_ac_network_from_ppc(_build_ac_island_ppc(breaker_status=0))
        prepare_ac_topology(network)

        self.assertEqual([True, False], [island.is_alive for island in network.islands])
        self.assertEqual([True, False], [gen.is_alive for gen in network.generators])

    def test_dc_object_topology_applies_device_composition_through_dcdc(self):
        from model.dc_array_model import build_dc_network_from_ppc
        from model.topology import prepare_dc_topology

        network = build_dc_network_from_ppc(_build_dc_island_ppc(breaker_status=0))
        prepare_dc_topology(network)

        connected = build_dc_network_from_ppc(_build_dc_island_ppc(breaker_status=1))
        prepare_dc_topology(connected)

        self.assertEqual([True, False, False], [island.is_alive for island in network.islands])
        self.assertEqual([True, False], [gen.is_alive for gen in network.generators])
        self.assertFalse(network.dcdc_converters[0].is_alive)
        self.assertEqual([True, True], [island.is_alive for island in connected.islands])
        self.assertTrue(connected.dcdc_converters[0].is_alive)

    def test_hybrid_pq_none_converter_cannot_keep_dangling_dc_terminal_alive(self):
        ppc = _build_hybrid_island_ppc(
            dc_breaker_status=0,
            connect_ac_terminal=True,
            ac_control_type="PQ",
            dc_control_type="NONE",
        )

        self.assertEqual([True], ppc["ac"]["_topology_arrays"].island_alive_mask.tolist())
        self.assertEqual([True, False], ppc["dc"]["_topology_arrays"].island_alive_mask.tolist())
        self.assertFalse(ppc["dc"]["_topology_arrays"].node_alive_mask[1])

    def test_hybrid_pq_none_converter_cannot_keep_dangling_ac_terminal_alive(self):
        ppc = _build_hybrid_island_ppc(
            dc_breaker_status=1,
            connect_ac_terminal=False,
            ac_control_type="PQ",
            dc_control_type="NONE",
        )

        self.assertEqual([True, False], ppc["ac"]["_topology_arrays"].island_alive_mask.tolist())
        self.assertEqual([True], ppc["dc"]["_topology_arrays"].island_alive_mask.tolist())

    def test_hybrid_converter_remains_connected_when_both_endpoint_islands_have_local_references(self):
        ppc = _build_hybrid_island_ppc(
            dc_breaker_status=1,
            connect_ac_terminal=False,
            ac_control_type="PH",
            dc_control_type="NONE",
        )

        self.assertEqual([True, True], ppc["ac"]["_topology_arrays"].island_alive_mask.tolist())
        self.assertEqual([True], ppc["dc"]["_topology_arrays"].island_alive_mask.tolist())

    def test_acac_does_not_count_load_from_unreferenced_endpoint(self):
        ppc = _build_acac_source_to_unreferenced_load_ppc()
        arrays = ppc["_topology_arrays"]
        self.assertEqual([False, False], arrays.island_alive_mask.tolist())
        self.assertEqual([False], arrays.devices["acac"].alive_mask.tolist())

    def test_acac_ph_endpoint_reference_has_ppc_object_parity(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.dc_array_model import build_dc_network_from_ppc
        from model.hybrid_array_model import build_hybrid_model_from_ppc

        ppc = _build_acac_ph_reference_hybrid_ppc()
        object_ppc = dict(ppc)
        object_ppc["ac_network"] = build_ac_network_from_ppc(ppc["ac"])
        object_ppc["dc_network"] = build_dc_network_from_ppc(ppc["dc"])
        network = build_hybrid_model_from_ppc(object_ppc)

        network.topo()

        ppc_alive = ppc["ac"]["_topology_arrays"].island_alive_mask.tolist()
        object_alive = [island.is_alive for island in network.ac.islands]
        self.assertEqual([True, True], object_alive)
        self.assertEqual(object_alive, ppc_alive)
        self.assertEqual(
            [True],
            ppc["ac"]["_topology_arrays"].devices["acac"].alive_mask.tolist(),
        )
        self.assertTrue(network.acac_converters[0].is_alive)

    def test_acac_ph_reference_suppresses_object_auto_slack_with_ppc_parity(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.dc_array_model import build_dc_network_from_ppc
        from model.hybrid_array_model import build_hybrid_model_from_ppc

        ppc = _build_acac_ph_with_pv_hybrid_ppc()
        object_ppc = dict(ppc)
        object_ppc["ac_network"] = build_ac_network_from_ppc(ppc["ac"])
        object_ppc["dc_network"] = build_dc_network_from_ppc(ppc["dc"])
        network = build_hybrid_model_from_ppc(object_ppc)

        network.topo()

        ppc_alive = ppc["ac"]["_topology_arrays"].island_alive_mask.tolist()
        object_alive = [island.is_alive for island in network.ac.islands]
        self.assertEqual([False, False], ppc_alive)
        self.assertEqual(ppc_alive, object_alive)
        self.assertEqual([], ppc["ac"]["_auto_slack_gen_rows"].tolist())
        self.assertEqual([], network.ac._auto_slack_generators)
        self.assertFalse(network.acac_converters[0].is_alive)

    def test_dcac_ph_reference_suppresses_object_auto_slack_with_ppc_parity(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.dc_array_model import build_dc_network_from_ppc
        from model.hybrid_array_model import build_hybrid_model_from_ppc

        ppc = _build_dcac_ph_with_pv_hybrid_ppc()
        object_ppc = dict(ppc)
        object_ppc["ac_network"] = build_ac_network_from_ppc(ppc["ac"])
        object_ppc["dc_network"] = build_dc_network_from_ppc(ppc["dc"])
        network = build_hybrid_model_from_ppc(object_ppc)

        network.topo()

        ppc_ac_alive = ppc["ac"]["_topology_arrays"].island_alive_mask.tolist()
        object_ac_alive = [island.is_alive for island in network.ac.islands]
        self.assertEqual([False], ppc_ac_alive)
        self.assertEqual(ppc_ac_alive, object_ac_alive)
        self.assertEqual([], ppc["ac"]["_auto_slack_gen_rows"].tolist())
        self.assertEqual([], network.ac._auto_slack_generators)
        self.assertFalse(network.dcac_converters[0].is_alive)

    def test_removing_last_dcac_ph_reference_reconsiders_ac_auto_slack(self):
        from model.hybrid_array_model import DCAC_COLS
        from model.ppc_topology import ensure_hybrid_ppc_topology

        ppc = _build_dcac_ph_with_pv_hybrid_ppc()
        previous_topology = ppc["ac"]["_topology_arrays"]
        ppc["dcac"] = np.empty((0, len(DCAC_COLS)), dtype=np.float64)

        ensure_hybrid_ppc_topology(ppc)

        self.assertNotIn("_hybrid_dcac_angle_reference_node_ids", ppc["ac"])
        self.assertNotIn("_hybrid_dcac_voltage_reference_pu", ppc["ac"])
        self.assertNotIn("_external_angle_reference_node_ids", ppc["ac"])
        self.assertIsNot(previous_topology, ppc["ac"]["_topology_arrays"])
        self.assertEqual([0], ppc["ac"]["_auto_slack_gen_rows"].tolist())
        self.assertEqual([True], ppc["ac"]["_topology_arrays"].island_alive_mask.tolist())

    def test_removing_last_dcac_v_reference_cannot_keep_dcdc_component_alive(self):
        from model.hybrid_array_model import DCAC_COLS
        from model.ppc_topology import ensure_hybrid_ppc_topology

        ppc = _build_dcac_v_with_dcdc_hybrid_ppc()
        self.assertEqual([True, True], ppc["dc"]["_topology_arrays"].island_alive_mask.tolist())
        previous_topology = ppc["dc"]["_topology_arrays"]
        ppc["dcac"] = np.empty((0, len(DCAC_COLS)), dtype=np.float64)

        ensure_hybrid_ppc_topology(ppc)

        self.assertNotIn("_hybrid_dcac_voltage_reference_node_ids", ppc["dc"])
        self.assertNotIn("_hybrid_dcac_voltage_reference_pu", ppc["dc"])
        self.assertNotIn("_external_voltage_reference_node_ids", ppc["dc"])
        self.assertIsNot(previous_topology, ppc["dc"]["_topology_arrays"])
        self.assertEqual([False, False], ppc["dc"]["_topology_arrays"].island_alive_mask.tolist())
        self.assertEqual(
            [False],
            ppc["dc"]["_topology_arrays"].devices["dcdc"].alive_mask.tolist(),
        )

    def test_removing_dcac_references_preserves_caller_owned_external_metadata(self):
        from model.hybrid_array_model import DCAC_COLS
        from model.ppc_topology import ensure_hybrid_ppc_topology

        ppc = _build_hybrid_island_ppc(
            dc_breaker_status=1,
            ac_control_type="PH",
            dc_control_type="V",
        )
        ppc["ac"]["_external_angle_reference_node_ids"] = np.asarray([1], dtype=np.int64)
        ppc["ac"]["_external_voltage_reference_pu"] = np.asarray([0.97], dtype=np.float64)
        ppc["dc"]["_external_voltage_reference_node_ids"] = np.asarray([1], dtype=np.int64)
        ppc["dc"]["_external_voltage_reference_pu"] = np.asarray([1.02], dtype=np.float64)
        ensure_hybrid_ppc_topology(ppc)
        ppc["dcac"] = np.empty((0, len(DCAC_COLS)), dtype=np.float64)

        ensure_hybrid_ppc_topology(ppc)

        np.testing.assert_array_equal(
            np.asarray([1], dtype=np.int64),
            ppc["ac"]["_external_angle_reference_node_ids"],
        )
        np.testing.assert_allclose(
            np.asarray([0.97], dtype=np.float64),
            ppc["ac"]["_external_voltage_reference_pu"],
        )
        np.testing.assert_array_equal(
            np.asarray([1], dtype=np.int64),
            ppc["dc"]["_external_voltage_reference_node_ids"],
        )
        np.testing.assert_allclose(
            np.asarray([1.02], dtype=np.float64),
            ppc["dc"]["_external_voltage_reference_pu"],
        )
        self.assertNotIn("_hybrid_dcac_angle_reference_node_ids", ppc["ac"])
        self.assertNotIn("_hybrid_dcac_voltage_reference_node_ids", ppc["dc"])

    def test_dcdc_does_not_count_load_from_unreferenced_endpoint(self):
        ppc = _build_dcdc_source_to_unreferenced_load_ppc()
        arrays = ppc["_topology_arrays"]
        self.assertEqual([False, False], arrays.island_alive_mask.tolist())
        self.assertEqual([False], arrays.devices["dcdc"].alive_mask.tolist())

    def test_hybrid_object_topology_matches_ppc_local_reference_filter(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.dc_array_model import build_dc_network_from_ppc
        from model.hybrid_array_model import build_hybrid_model_from_ppc

        ppc = _build_hybrid_island_ppc(
            dc_breaker_status=0,
            connect_ac_terminal=True,
            ac_control_type="PQ",
            dc_control_type="NONE",
        )
        object_ppc = dict(ppc)
        object_ppc["ac_network"] = build_ac_network_from_ppc(ppc["ac"])
        object_ppc["dc_network"] = build_dc_network_from_ppc(ppc["dc"])
        network = build_hybrid_model_from_ppc(object_ppc)

        network.topo()

        self.assertEqual(
            ppc["ac"]["_topology_arrays"].island_alive_mask.tolist(),
            [island.is_alive for island in network.ac.islands],
        )
        self.assertEqual(
            ppc["dc"]["_topology_arrays"].island_alive_mask.tolist(),
            [island.is_alive for island in network.dc.islands],
        )
        self.assertFalse(network.dcac_converters[0].is_alive)

    def test_hybrid_device_composition_propagates_across_running_dcac(self):
        connected = _build_hybrid_island_ppc(dc_breaker_status=1)
        isolated = _build_hybrid_island_ppc(dc_breaker_status=0)

        connected_ac = connected["ac"]["_topology_arrays"]
        connected_dc = connected["dc"]["_topology_arrays"]
        isolated_ac = isolated["ac"]["_topology_arrays"]
        isolated_dc = isolated["dc"]["_topology_arrays"]

        self.assertEqual([True, True], connected_ac.island_alive_mask.tolist())
        self.assertEqual([True], connected_dc.island_alive_mask.tolist())
        self.assertEqual([True, False], isolated_ac.island_alive_mask.tolist())
        self.assertEqual([True, False], isolated_dc.island_alive_mask.tolist())

    def test_hybrid_object_topology_uses_combined_ac_dc_device_composition(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.dc_array_model import build_dc_network_from_ppc
        from model.hybrid_array_model import build_hybrid_model_from_ppc

        def build_network(dc_breaker_status):
            ppc = _build_hybrid_island_ppc(dc_breaker_status=dc_breaker_status)
            object_ppc = dict(ppc)
            object_ppc["ac_network"] = build_ac_network_from_ppc(ppc["ac"])
            object_ppc["dc_network"] = build_dc_network_from_ppc(ppc["dc"])
            return build_hybrid_model_from_ppc(object_ppc)

        connected = build_network(1)
        connected.topo()
        isolated = build_network(0)
        isolated.topo()

        self.assertEqual([True, True], [island.is_alive for island in connected.ac.islands])
        self.assertEqual([True], [island.is_alive for island in connected.dc.islands])
        self.assertTrue(connected.dcac_converters[0].is_alive)
        self.assertEqual([True, False], [island.is_alive for island in isolated.ac.islands])
        self.assertEqual([True, False], [island.is_alive for island in isolated.dc.islands])
        self.assertFalse(isolated.dcac_converters[0].is_alive)

    def test_hybrid_topology_rebuilds_local_references_when_dcac_state_changes(self):
        from model.ac_array_model import CTRL_SLACK, GEN_COLS
        from model.hybrid_array_model import (
            DCAC_AC_CONTROL_CODE,
            DCAC_COLS,
            DCAC_DC_CONTROL_CODE,
        )
        from model.ppc_topology import ensure_hybrid_ppc_topology

        ppc = _build_hybrid_island_ppc(dc_breaker_status=1)
        ppc["ac"]["gen"][1, GEN_COLS["control_type"]] = CTRL_SLACK
        ppc["dcac"][0, DCAC_COLS["ac_control_type"]] = DCAC_AC_CONTROL_CODE["PQ"]
        ppc["dcac"][0, DCAC_COLS["dc_control_type"]] = DCAC_DC_CONTROL_CODE["NONE"]
        ppc["dcac"][0, DCAC_COLS["run_stat"]] = 0
        ensure_hybrid_ppc_topology(ppc)

        self.assertEqual([True, False], ppc["ac"]["_topology_arrays"].island_alive_mask.tolist())

        ppc["dcac"][0, DCAC_COLS["run_stat"]] = 1
        ensure_hybrid_ppc_topology(ppc)
        ac_topology = ppc["ac"]["_topology_arrays"]

        self.assertEqual([True, True], ac_topology.island_alive_mask.tolist())
        self.assertGreaterEqual(int(ac_topology.island_reference_bus_pos[1]), 0)

    def test_ac_ppc_topology_auto_selects_one_capacity_ranked_pv_per_island(self):
        from model.topology import prepare_ac_topology_ppc

        ppc = _build_auto_ph_ppc(
            node_ids=[1, 2, 3, 4],
            branches=[(1, 2), (3, 4)],
            generators=[
                (1, 1, "PV", 90, 1.0, 80),
                (2, 2, "PV", 10, 1.0, 100),
                (3, 3, "PV", 40, 1.0, np.nan),
                (4, 4, "PV", 30, 5.0, np.nan),
            ],
        )

        arrays = prepare_ac_topology_ppc(ppc)

        self.assertEqual([1, 2], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([True, True], arrays.island_alive_mask.tolist())
        self.assertEqual(
            [
                int(arrays.node_to_bus_pos[1]),
                int(arrays.node_to_bus_pos[2]),
            ],
            arrays.island_reference_bus_pos.tolist(),
        )

    def test_ac_object_topology_uses_auto_selected_pv_as_balance_generator(self):
        from model.ac_array_model import build_ac_network_from_ppc
        from model.topology import prepare_ac_topology

        ppc = _build_auto_ph_ppc(
            node_ids=[1, 2, 3, 4],
            branches=[(1, 2), (3, 4)],
            generators=[
                (1, 1, "PV", 90, 1.0, 80),
                (2, 2, "PV", 10, 1.0, 100),
                (3, 3, "PV", 40, 1.0, np.nan),
                (4, 4, "PV", 30, 5.0, np.nan),
            ],
        )
        network = build_ac_network_from_ppc(ppc)

        prepare_ac_topology(network)

        self.assertEqual([True, True], [island.is_alive for island in network.islands])
        self.assertEqual(
            [2, 3],
            [gen.idx for gen in network._auto_slack_generators],
        )

    def test_ac_ppc_topology_auto_slack_falls_back_to_alpha_then_smallest_idx(self):
        from model.topology import prepare_ac_topology_ppc

        ppc = _build_auto_ph_ppc(
            node_ids=[1, 2, 3],
            branches=[(1, 2), (2, 3)],
            generators=[
                (20, 1, "PV", 10, 1.0, np.nan),
                (30, 2, "PV", -10, 2.0, np.nan),
                (10, 3, "PV", 10, 2.0, np.nan),
            ],
        )

        arrays = prepare_ac_topology_ppc(ppc)

        self.assertEqual([2], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([True], arrays.island_alive_mask.tolist())
        self.assertEqual(int(arrays.node_to_bus_pos[2]), int(arrays.island_reference_bus_pos[0]))

    def test_ac_ppc_topology_external_reference_does_not_replace_balance_generator(self):
        from model.topology import prepare_ac_topology_ppc

        explicit = _build_auto_ph_ppc(
            node_ids=[1, 2],
            branches=[(1, 2)],
            generators=[
                (1, 1, "SLACK", 0, 1.0, 10),
                (2, 2, "PV", 0, 1.0, 100),
            ],
        )
        explicit_arrays = prepare_ac_topology_ppc(explicit)

        external = _build_auto_ph_ppc(
            node_ids=[1, 2],
            branches=[(1, 2)],
            generators=[
                (1, 1, "PV", 0, 1.0, 10),
                (2, 2, "PV", 0, 1.0, 100),
            ],
        )
        external["_external_angle_reference_node_ids"] = np.asarray([1], dtype=np.int64)
        external_arrays = prepare_ac_topology_ppc(external)

        self.assertEqual([], explicit["_auto_slack_gen_rows"].tolist())
        self.assertEqual(int(explicit_arrays.node_to_bus_pos[0]), int(explicit_arrays.island_reference_bus_pos[0]))
        self.assertEqual([], external["_auto_slack_gen_rows"].tolist())
        self.assertEqual([False], external_arrays.island_alive_mask.tolist())
        self.assertEqual([-1], external_arrays.island_reference_bus_pos.tolist())

    def test_ac_ppc_topology_leaves_island_dead_when_no_online_pv_exists(self):
        from model.topology import prepare_ac_topology_ppc

        ppc = _build_auto_ph_ppc(
            node_ids=[1],
            branches=[],
            generators=[(1, 1, "PQ", 10, 1.0, 100)],
        )

        arrays = prepare_ac_topology_ppc(ppc)

        self.assertEqual([], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([False], arrays.island_alive_mask.tolist())
        self.assertEqual([-1], arrays.island_reference_bus_pos.tolist())

    def test_ac_ppc_topology_refreshes_auto_slack_when_generator_ranking_changes(self):
        from model.ac_array_model import GEN_COLS
        from model.ppc_topology import ensure_ac_ppc_topology

        ppc = _build_auto_ph_ppc(
            node_ids=[1, 2],
            branches=[(1, 2)],
            generators=[
                (1, 1, "PV", 0, 1.0, 100),
                (2, 2, "PV", 0, 1.0, 50),
            ],
        )

        ensure_ac_ppc_topology(ppc)
        self.assertEqual([0], ppc["_auto_slack_gen_rows"].tolist())

        ppc["gen"][0, GEN_COLS["p_max"]] = 10
        ppc["gen"][1, GEN_COLS["p_max"]] = 200
        ensure_ac_ppc_topology(ppc)
        self.assertEqual([1], ppc["_auto_slack_gen_rows"].tolist())

        ppc["gen"][1, GEN_COLS["run_stat"]] = 0
        ensure_ac_ppc_topology(ppc)
        self.assertEqual([], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([False], ppc["_topology_arrays"].island_alive_mask.tolist())

    def test_ac_ppc_topology_refreshes_auto_slack_when_islands_merge(self):
        from model.ac_array_model import BRANCH_COLS
        from model.ppc_topology import ensure_ac_ppc_topology

        ppc = _build_auto_ph_ppc(
            node_ids=[1, 2],
            branches=[(1, 2)],
            generators=[
                (1, 1, "PV", 0, 1.0, 100),
                (2, 2, "PV", 0, 1.0, 50),
            ],
        )
        ppc["branch"][0, BRANCH_COLS["run_stat"]] = 0
        ppc.pop("_topology_input", None)

        ensure_ac_ppc_topology(ppc)
        self.assertEqual(2, len(ppc["_topology_arrays"].island_ids))
        self.assertEqual([], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([False, False], ppc["_topology_arrays"].island_alive_mask.tolist())

        ppc["branch"][0, BRANCH_COLS["run_stat"]] = 1
        ensure_ac_ppc_topology(ppc)

        self.assertEqual(1, len(ppc["_topology_arrays"].island_ids))
        self.assertEqual([0], ppc["_auto_slack_gen_rows"].tolist())
        self.assertEqual([True], ppc["_topology_arrays"].island_alive_mask.tolist())

    def test_parent_index_uses_dense_storage_for_contiguous_ids(self):
        from model.topology import _make_parent_index, _parent_contains, _union_parent, _find_parent

        parent = _make_parent_index([1, 2, 3, 4])

        self.assertIsInstance(parent, list)
        self.assertTrue(_parent_contains(parent, 3))
        self.assertFalse(_parent_contains(parent, 0))
        _union_parent(parent, 1, 4)
        self.assertEqual(_find_parent(parent, 1), _find_parent(parent, 4))

    def test_parent_index_uses_sparse_storage_for_large_gaps(self):
        from model.topology import _make_parent_index, _parent_contains

        parent = _make_parent_index([1, 1_000_000])

        self.assertIsInstance(parent, dict)
        self.assertTrue(_parent_contains(parent, 1_000_000))
        self.assertFalse(_parent_contains(parent, 2))

    def test_ac_ppc_topology_matches_object_topology(self):
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import prepare_ac_topology, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        network = build_ac_network_from_ppc(ppc)
        prepare_ac_topology(network)

        arrays = prepare_ac_topology_ppc(ppc)

        self.assertEqual([node.idx for node in network.nodes], arrays.node_ids.tolist())
        self.assertEqual([bus.idx for bus in network.buses], arrays.bus_ids.tolist())
        self.assertEqual([island.idx for island in network.islands], arrays.island_ids.tolist())
        node_bus_by_idx = {node.idx: node.bus for node in network.nodes}
        node_island_by_idx = {node.idx: node.isl for node in network.nodes}
        node_alive_by_idx = {node.idx: node.is_alive for node in network.nodes}
        for pos, node_id in enumerate(arrays.node_ids):
            bus_pos = int(arrays.node_to_bus_pos[pos])
            island_pos = int(arrays.node_to_island_pos[pos])
            self.assertEqual(node_bus_by_idx[int(node_id)], int(arrays.bus_ids[bus_pos]) if bus_pos >= 0 else None)
            self.assertEqual(node_island_by_idx[int(node_id)] or 0, int(arrays.island_ids[island_pos]) if island_pos >= 0 else 0)
            self.assertEqual(node_alive_by_idx[int(node_id)], bool(arrays.node_alive_mask[pos]))
        self.assertEqual([bus.is_alive for bus in network.buses], arrays.bus_alive_mask.tolist())
        self.assertEqual([island.is_alive for island in network.islands], arrays.island_alive_mask.tolist())
        self.assertTrue(np.array_equal([br.is_alive for br in network.branches], arrays.devices["branch"].alive_mask))
        self.assertTrue(np.array_equal([tr.is_alive for tr in network.transformers], arrays.devices["transformer"].alive_mask))
        self.assertTrue(np.array_equal([gen.is_alive for gen in network.generators], arrays.devices["gen"].alive_mask))
        self.assertTrue(np.array_equal([load.is_alive for load in network.loads], arrays.devices["load"].alive_mask))
        self.assertTrue(np.array_equal([sw.is_alive for sw in network.switches], arrays.devices["switch"].alive_mask))
        self.assertTrue(np.array_equal([brk.is_alive for brk in network.breakers], arrays.devices["break"].alive_mask))
        self.assertTrue(np.array_equal([zbr.is_alive for zbr in network.zero_branches], arrays.devices["zero_branch"].alive_mask))

    def test_ppc_topology_uses_sparse_component_search(self):
        from unittest.mock import patch

        from model.ac_array_model import build_ac_ppc_from_e_file
        from model.topology import prepare_ac_topology_ppc
        import model.topology as topology

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")

        with patch.object(topology, "_union_terminal_table", side_effect=AssertionError("ppc topology should not use Python union loop")):
            arrays = prepare_ac_topology_ppc(ppc)

        self.assertTrue(arrays.bus_ids.size)

    def test_ac_ppc_topology_consumes_precomputed_terminal_positions(self):
        from unittest.mock import patch

        import model.topology as topology
        from model.ac_array_model import build_ac_ppc_from_e_file
        from model.topology import prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        self.assertIn("_topology_input", ppc)

        with patch.object(topology, "_map_node_positions", side_effect=AssertionError("precomputed topology input should be reused")):
            arrays = prepare_ac_topology_ppc(ppc)

        self.assertTrue(arrays.bus_ids.size)
        self.assertTrue(arrays.devices["branch"].i_node_pos.size)

    def test_ac_apply_ppc_topology_arrays_matches_object_topology(self):
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import apply_ac_topology_arrays, prepare_ac_topology, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        expected = build_ac_network_from_ppc(ppc)
        prepare_ac_topology(expected)

        actual = build_ac_network_from_ppc(ppc)
        arrays = prepare_ac_topology_ppc(ppc)
        apply_ac_topology_arrays(actual, arrays)

        self.assertEqual([bus.idx for bus in expected.buses], [bus.idx for bus in actual.buses])
        self.assertEqual([isl.idx for isl in expected.islands], [isl.idx for isl in actual.islands])
        self.assertEqual([bus.is_alive for bus in expected.buses], [bus.is_alive for bus in actual.buses])
        self.assertEqual([isl.is_alive for isl in expected.islands], [isl.is_alive for isl in actual.islands])
        self.assertEqual(
            {node.idx: (node.bus, node.isl, node.is_alive) for node in expected.nodes},
            {node.idx: (node.bus, node.isl, node.is_alive) for node in actual.nodes},
        )
        self.assertEqual(set(expected.alive_branch_by_name), set(actual.alive_branch_by_name))
        self.assertEqual(set(expected.alive_transformer_by_name), set(actual.alive_transformer_by_name))
        self.assertEqual(set(expected.alive_generator_by_name), set(actual.alive_generator_by_name))
        self.assertEqual(set(expected.alive_load_by_name), set(actual.alive_load_by_name))
        self.assertEqual(set(expected.alive_switch_by_name), set(actual.alive_switch_by_name))
        self.assertEqual(set(expected.alive_break_by_name), set(actual.alive_break_by_name))
        self.assertEqual(set(expected.alive_zero_branch_by_name), set(actual.alive_zero_branch_by_name))

    def test_ac_apply_ppc_topology_arrays_compact_skips_reverse_device_lists(self):
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import apply_ac_topology_arrays, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        network = build_ac_network_from_ppc(ppc)
        arrays = prepare_ac_topology_ppc(ppc)

        apply_ac_topology_arrays(network, arrays, compact=True)

        self.assertTrue(network.alive_branch_by_name)
        self.assertTrue(network.alive_generator_by_name)
        self.assertTrue(any(island.slack_nodes for island in network.islands if island.is_alive))
        self.assertTrue(all(getattr(gen, "node_obj", None) is not None for gen in network.generators))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is not None for branch in network.branches))
        self.assertTrue(all(len(getattr(node, "branches", ())) == 0 for node in network.nodes))
        self.assertTrue(all(len(getattr(node, "generators", ())) == 0 for node in network.nodes))
        self.assertTrue(all(len(getattr(island, "branches", ())) == 0 for island in network.islands))
        self.assertTrue(all(len(getattr(island, "gens", ())) == 0 for island in network.islands))
        self.assertFalse(network.bus_dict)
        self.assertFalse(network.branch_dict)
        self.assertFalse(network.generator_dict)

    def test_ac_apply_ppc_topology_arrays_compact_can_skip_alive_maps(self):
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import apply_ac_topology_arrays, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        network = build_ac_network_from_ppc(ppc)
        arrays = prepare_ac_topology_ppc(ppc)

        apply_ac_topology_arrays(network, arrays, compact=True, build_alive_maps=False)

        self.assertTrue(network.alive_buses)
        self.assertFalse(network.alive_branch_by_name)
        self.assertFalse(network.alive_generator_by_name)
        self.assertFalse(network.alive_zero_branches)
        self.assertTrue(any(island.slack_nodes for island in network.islands if island.is_alive))
        self.assertTrue(all(getattr(gen, "node_obj", None) is not None for gen in network.generators))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is not None for branch in network.branches))

    def test_ac_apply_ppc_topology_arrays_compact_uses_lightweight_bus_construction(self):
        from unittest.mock import patch

        import model.ac_model as ac_model
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import apply_ac_topology_arrays, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        network = build_ac_network_from_ppc(ppc)
        arrays = prepare_ac_topology_ppc(ppc)

        with patch.object(ac_model.ACBus, "__init__", side_effect=AssertionError("compact topology should not allocate full bus state")):
            apply_ac_topology_arrays(network, arrays, compact=True, build_alive_maps=False)

        self.assertTrue(network.buses)
        self.assertTrue(all(getattr(bus, "nodes", None) for bus in network.buses))

    def test_ac_apply_ppc_topology_arrays_can_skip_device_object_backfill(self):
        from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
        from model.topology import apply_ac_topology_arrays, prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "ac" / "ieee39.e")
        network = build_ac_network_from_ppc(ppc)
        arrays = prepare_ac_topology_ppc(ppc)

        apply_ac_topology_arrays(
            network,
            arrays,
            compact=True,
            build_alive_maps=False,
            populate_device_links=False,
        )

        self.assertTrue(np.array_equal([gen.is_alive for gen in network.generators], arrays.devices["gen"].alive_mask))
        self.assertTrue(np.array_equal([br.is_alive for br in network.branches], arrays.devices["branch"].alive_mask))
        self.assertTrue(all(getattr(gen, "node_obj", None) is None for gen in network.generators))
        self.assertTrue(all(getattr(load, "node_obj", None) is None for load in network.loads))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is None for branch in network.branches))
        self.assertTrue(all(getattr(branch, "j_node_obj", None) is None for branch in network.branches))

    def test_dc_ppc_topology_matches_object_topology(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from model.topology import prepare_dc_topology, prepare_dc_topology_ppc

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network = build_dc_network_from_ppc(ppc)
        prepare_dc_topology(network)

        arrays = prepare_dc_topology_ppc(ppc)

        self.assertEqual([node.idx for node in network.nodes], arrays.node_ids.tolist())
        self.assertEqual([bus.idx for bus in network.buses], arrays.bus_ids.tolist())
        self.assertEqual([island.idx for island in network.islands], arrays.island_ids.tolist())
        node_bus_by_idx = {node.idx: node.bus for node in network.nodes}
        node_island_by_idx = {node.idx: node.isl for node in network.nodes}
        node_alive_by_idx = {node.idx: node.is_alive for node in network.nodes}
        for pos, node_id in enumerate(arrays.node_ids):
            bus_pos = int(arrays.node_to_bus_pos[pos])
            island_pos = int(arrays.node_to_island_pos[pos])
            self.assertEqual(node_bus_by_idx[int(node_id)], int(arrays.bus_ids[bus_pos]) if bus_pos >= 0 else None)
            self.assertEqual(node_island_by_idx[int(node_id)] or 0, int(arrays.island_ids[island_pos]) if island_pos >= 0 else 0)
            self.assertEqual(node_alive_by_idx[int(node_id)], bool(arrays.node_alive_mask[pos]))
        self.assertEqual([bus.is_alive for bus in network.buses], arrays.bus_alive_mask.tolist())
        self.assertEqual([island.is_alive for island in network.islands], arrays.island_alive_mask.tolist())
        self.assertTrue(np.array_equal([br.is_alive for br in network.branches], arrays.devices["branch"].alive_mask))
        self.assertTrue(np.array_equal([gen.is_alive for gen in network.generators], arrays.devices["gen"].alive_mask))
        self.assertTrue(np.array_equal([load.is_alive for load in network.loads], arrays.devices["load"].alive_mask))
        self.assertTrue(np.array_equal([sw.is_alive for sw in network.switches], arrays.devices["switch"].alive_mask))
        self.assertTrue(np.array_equal([brk.is_alive for brk in network.breakers], arrays.devices["break"].alive_mask))
        self.assertTrue(np.array_equal([zbr.is_alive for zbr in network.zero_branches], arrays.devices["zero_branch"].alive_mask))
        self.assertTrue(np.array_equal([conv.is_alive for conv in network.dcdc_converters], arrays.devices["dcdc"].alive_mask))

    def test_dc_apply_ppc_topology_arrays_matches_object_topology(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from model.topology import apply_dc_topology_arrays, prepare_dc_topology, prepare_dc_topology_ppc

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        expected = build_dc_network_from_ppc(ppc)
        prepare_dc_topology(expected)

        actual = build_dc_network_from_ppc(ppc)
        arrays = prepare_dc_topology_ppc(ppc)
        apply_dc_topology_arrays(actual, arrays)

        self.assertEqual([bus.idx for bus in expected.buses], [bus.idx for bus in actual.buses])
        self.assertEqual([isl.idx for isl in expected.islands], [isl.idx for isl in actual.islands])
        self.assertEqual([bus.is_alive for bus in expected.buses], [bus.is_alive for bus in actual.buses])
        self.assertEqual([isl.is_alive for isl in expected.islands], [isl.is_alive for isl in actual.islands])
        self.assertEqual(
            {node.idx: (node.bus, node.isl, node.is_alive, node.is_slack) for node in expected.nodes},
            {node.idx: (node.bus, node.isl, node.is_alive, node.is_slack) for node in actual.nodes},
        )
        self.assertEqual(set(expected.alive_branch_by_name), set(actual.alive_branch_by_name))
        self.assertEqual(set(expected.alive_generator_by_name), set(actual.alive_generator_by_name))
        self.assertEqual(set(expected.alive_load_by_name), set(actual.alive_load_by_name))
        self.assertEqual(set(expected.alive_switch_by_name), set(actual.alive_switch_by_name))
        self.assertEqual(set(expected.alive_break_by_name), set(actual.alive_break_by_name))
        self.assertEqual(set(expected.alive_zero_branch_by_name), set(actual.alive_zero_branch_by_name))
        self.assertEqual(set(expected.alive_dcdc_by_name), set(actual.alive_dcdc_by_name))

    def test_dc_apply_ppc_topology_arrays_compact_skips_reverse_device_lists(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from model.topology import apply_dc_topology_arrays, prepare_dc_topology_ppc

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network = build_dc_network_from_ppc(ppc)
        arrays = prepare_dc_topology_ppc(ppc)

        apply_dc_topology_arrays(network, arrays, compact=True)

        self.assertTrue(network.alive_branch_by_name)
        self.assertTrue(network.alive_generator_by_name)
        self.assertTrue(any(island.slack_nodes for island in network.islands if island.is_alive))
        self.assertTrue(all(getattr(gen, "node_obj", None) is not None for gen in network.generators))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is not None for branch in network.branches))
        self.assertTrue(all(len(getattr(node, "branches", ())) == 0 for node in network.nodes))
        self.assertTrue(all(len(getattr(node, "generators", ())) == 0 for node in network.nodes))
        self.assertTrue(all(len(getattr(island, "branches", ())) == 0 for island in network.islands))
        self.assertTrue(all(len(getattr(island, "gens", ())) == 0 for island in network.islands))
        self.assertFalse(network.bus_dict)
        self.assertFalse(network.branch_dict)
        self.assertFalse(network.generator_dict)

    def test_dc_apply_ppc_topology_arrays_compact_can_skip_alive_maps(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from model.topology import apply_dc_topology_arrays, prepare_dc_topology_ppc

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network = build_dc_network_from_ppc(ppc)
        arrays = prepare_dc_topology_ppc(ppc)

        apply_dc_topology_arrays(network, arrays, compact=True, build_alive_maps=False)

        self.assertTrue(network.alive_buses)
        self.assertFalse(network.alive_branch_by_name)
        self.assertFalse(network.alive_generator_by_name)
        self.assertFalse(network.alive_zero_branches)
        self.assertTrue(any(island.slack_nodes for island in network.islands if island.is_alive))
        self.assertTrue(all(getattr(gen, "node_obj", None) is not None for gen in network.generators))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is not None for branch in network.branches))

    def test_dc_apply_ppc_topology_arrays_can_skip_device_object_backfill(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from model.topology import apply_dc_topology_arrays, prepare_dc_topology_ppc

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network = build_dc_network_from_ppc(ppc)
        arrays = prepare_dc_topology_ppc(ppc)

        apply_dc_topology_arrays(
            network,
            arrays,
            compact=True,
            build_alive_maps=False,
            populate_device_links=False,
        )

        self.assertTrue(np.array_equal([gen.is_alive for gen in network.generators], arrays.devices["gen"].alive_mask))
        self.assertTrue(np.array_equal([br.is_alive for br in network.branches], arrays.devices["branch"].alive_mask))
        self.assertTrue(np.array_equal([conv.is_alive for conv in network.dcdc_converters], arrays.devices["dcdc"].alive_mask))
        self.assertTrue(all(getattr(gen, "node_obj", None) is None for gen in network.generators))
        self.assertTrue(all(getattr(load, "node_obj", None) is None for load in network.loads))
        self.assertTrue(all(getattr(branch, "i_node_obj", None) is None for branch in network.branches))
        self.assertTrue(all(getattr(branch, "j_node_obj", None) is None for branch in network.branches))
        self.assertTrue(all(getattr(conv, "i_node_obj", None) is None for conv in network.dcdc_converters))
        self.assertTrue(all(getattr(conv, "j_node_obj", None) is None for conv in network.dcdc_converters))

    def test_common_ppc_topology_builders_attach_topology_arrays(self):
        from model.ppc_topology import (
            build_ac_ppc_with_topology_from_e_file,
            build_dc_ppc_with_topology_from_e_file,
            build_hybrid_ppc_with_topology_from_efile_rows,
        )
        from efile_read import _read_efile_rows

        root = Path(__file__).resolve().parents[1]

        ac_ppc = build_ac_ppc_with_topology_from_e_file(root / "data" / "model" / "ac" / "ieee39.e")
        self.assertIn("_topology_arrays", ac_ppc)
        self.assertGreater(ac_ppc["_topology_arrays"].bus_ids.size, 0)

        dc_ppc = build_dc_ppc_with_topology_from_e_file(root / "data" / "model" / "dc" / "dc_net_30.e")
        self.assertIn("_topology_arrays", dc_ppc)
        self.assertGreater(dc_ppc["_topology_arrays"].bus_ids.size, 0)

        hybrid_file = root / "data" / "model" / "hybrid" / "hybrid_net_40.e"
        hybrid_ppc = build_hybrid_ppc_with_topology_from_efile_rows(hybrid_file, _read_efile_rows(hybrid_file))
        self.assertIn("_topology_arrays", hybrid_ppc["ac"])
        self.assertIn("_topology_arrays", hybrid_ppc["dc"])
        self.assertGreater(hybrid_ppc["ac"]["_topology_arrays"].bus_ids.size, 0)
        self.assertGreater(hybrid_ppc["dc"]["_topology_arrays"].bus_ids.size, 0)

    def test_common_hybrid_ppc_topology_builder_keeps_ppc_only_path(self):
        from unittest.mock import patch

        import model.hybrid_array_model as hybrid_array_model
        from efile_read import _read_efile_rows
        from model.ppc_topology import build_hybrid_ppc_with_topology_from_efile_rows

        root = Path(__file__).resolve().parents[1]
        hybrid_file = root / "data" / "model" / "hybrid" / "hybrid_net_40.e"

        with patch.object(
            hybrid_array_model,
            "build_hybrid_ppc_from_efile_rows",
            side_effect=AssertionError("common topology builder should not build full HybridPowerNetwork"),
        ):
            ppc = build_hybrid_ppc_with_topology_from_efile_rows(hybrid_file, _read_efile_rows(hybrid_file))

        self.assertEqual("hybrid_ppc_v1", ppc["format"])
        self.assertIn("_topology_arrays", ppc["ac"])
        self.assertIn("_topology_arrays", ppc["dc"])


if __name__ == "__main__":
    unittest.main()
