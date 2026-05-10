import unittest
from pathlib import Path

import numpy as np


class TopologyHelperTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
