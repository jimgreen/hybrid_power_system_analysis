import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "model"))
sys.path.insert(0, str(ROOT_DIR / "lfcore"))


class ACPPCFlowTest(unittest.TestCase):
    def test_ppc_flow_applies_automatic_ph_reference_to_each_island(self):
        from ac_array_model import build_ac_ppc_from_efile_rows
        from ac_lf import AC_NODE_TYPE_SLACK, ACPowerFlowCalc
        from model.ppc_topology import ensure_ac_ppc_topology

        def table(header, rows):
            return {"header_list": header.split(), "rows": rows}

        rows = {
            "Model": table("path name p_base u_unit p_unit i_unit", [["IEEE", "two_islands", 100, "V", "kW", "A"]]),
            "ACNode": table(
                "idx name vbase voltage angle run_stat",
                [[idx, f"bus_{idx}", 380, 380, 0, 1] for idx in range(1, 5)],
            ),
            "ACBranch": table(
                "idx name i_node j_node r x b run_stat",
                [
                    [1, "branch_1_2", 1, 2, 0.01, 0.1, 0.0, 1],
                    [2, "branch_3_4", 3, 4, 0.01, 0.1, 0.0, 1],
                ],
            ),
            "ACGenerator": table(
                "idx name node control_type p_set q_set v_set alpha p_max run_stat",
                [
                    [1, "gen_1", 1, "PV", 0, 0, 380, 1.0, 80, 1],
                    [2, "gen_2", 3, "PV", 0, 0, 380, 1.0, 60, 1],
                ],
            ),
            "ACLoad": table(
                "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
                [
                    [1, "load_1", 2, 10, 1, 0, 0, 2, 1, 0, 0, 1],
                    [2, "load_2", 4, 5, 1, 0, 0, 1, 1, 0, 0, 1],
                ],
            ),
        }
        ppc = ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("two_islands.e"), rows))
        self.assertEqual([0, 1], ppc["_auto_slack_gen_rows"].tolist())

        calc = ACPowerFlowCalc(ppc, tol=1e-10, max_iter=50, verbose=False)
        calc.prepare()

        self.assertEqual(2, int(np.count_nonzero(calc.node_type == AC_NODE_TYPE_SLACK)))
        self.assertEqual(0, calc.run())
        self.assertTrue(calc.converged)

    def test_external_angle_reference_does_not_promote_an_additional_pv_slack(self):
        from ac_array_model import CTRL_PV, GEN_COLS, build_ac_ppc_from_e_file
        from ac_lf import AC_NODE_TYPE_SLACK, ACPowerFlowCalc
        from model.ppc_topology import ensure_ac_ppc_topology

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ac_net_10.e")
        ppc["gen"][:, GEN_COLS["control_type"]] = CTRL_PV
        ppc["_external_angle_reference_node_ids"] = np.asarray([1], dtype=np.int64)
        ppc.pop("_topology_arrays", None)

        ensure_ac_ppc_topology(ppc)
        self.assertEqual([], ppc["_auto_slack_gen_rows"].tolist())

        calc = ACPowerFlowCalc(ppc, verbose=False)
        calc.prepare()

        self.assertEqual(0, int(np.count_nonzero(calc.node_type == AC_NODE_TYPE_SLACK)))
        self.assertEqual(-1, calc.slack_node)

    def test_automatic_ph_only_absorbs_residual_power_for_its_generator_row(self):
        from ac_array_model import GEN_COLS, build_ac_ppc_from_efile_rows
        from ac_lf import ACPowerFlowCalc
        from model.ppc_topology import ensure_ac_ppc_topology

        def table(header, rows):
            return {"header_list": header.split(), "rows": rows}

        rows = {
            "Model": table("path name p_base u_unit p_unit i_unit", [["IEEE", "shared_bus", 100, "V", "kW", "A"]]),
            "ACNode": table(
                "idx name vbase voltage angle run_stat",
                [[1, "source", 380, 380, 0, 1], [2, "load", 380, 380, 0, 1]],
            ),
            "ACBranch": table(
                "idx name i_node j_node r x b run_stat",
                [[1, "branch", 1, 2, 0.01, 0.1, 0.0, 1]],
            ),
            "ACGenerator": table(
                "idx name node control_type p_set q_set v_set alpha p_max run_stat",
                [
                    [1, "auto_ph", 1, "PV", 0, 0, 380, 1.0, 100, 1],
                    [2, "fixed_pv", 1, "PV", 40, 0, 380, 1.0, 50, 1],
                ],
            ),
            "ACLoad": table(
                "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
                [[1, "load", 2, 50, 1, 0, 0, 10, 1, 0, 0, 1]],
            ),
        }
        ppc = ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("shared_bus.e"), rows))
        self.assertEqual([0], ppc["_auto_slack_gen_rows"].tolist())

        calc = ACPowerFlowCalc(ppc, tol=1e-10, max_iter=50, verbose=False)
        self.assertEqual(0, calc.run())

        result_gen = calc.result["gen"]
        self.assertAlmostEqual(0.4, result_gen[1, GEN_COLS["p"]], places=10)
        self.assertAlmostEqual(
            np.sum(result_gen[:, GEN_COLS["p"]]) - 0.4,
            result_gen[0, GEN_COLS["p"]],
            places=10,
        )

    def test_ac_topology_contracts_closed_switches_to_buses_before_islands(self):
        from ac_model import ACPowerNetwork

        network = ACPowerNetwork()
        network.add_node(1, 110.0)
        network.nodes[-1].name = "n1"
        network.add_node(2, 110.0)
        network.nodes[-1].name = "n2"
        network.add_node(3, 110.0)
        network.nodes[-1].name = "n3"
        network.add_generator(1, 1, "SLACK", 1.0, 0.0, 1.0)
        network.generators[-1].name = "g1"
        network.add_switch(1, 1, 2, 1)
        network.switches[-1].name = "sw_1_2"
        network.add_switch(2, 2, 3, 0)
        network.switches[-1].name = "sw_2_3"
        network.add_branch(1, 2, 3, 0.01, 0.05, 0.0)
        network.branches[-1].name = "br_2_3"

        network.topo()

        self.assertEqual(2, len(network.buses))
        self.assertEqual(["n1", "n2"], [node.name for node in network.node_dict[1].bus_obj.nodes])
        self.assertIs(network.node_dict[1].bus_obj, network.node_dict[2].bus_obj)
        self.assertIsNot(network.node_dict[2].bus_obj, network.node_dict[3].bus_obj)
        self.assertEqual(1, len(network.islands))
        self.assertEqual(2, len(network.islands[0].buses))

    def test_ac_break_is_parsed_as_distinct_zero_tie_device(self):
        from ac_array_model import SWITCH_COLS, build_ac_ppc_from_e_file
        from ac_model import ACPowerNetwork, ACBreak

        source = ROOT_DIR / "data" / "model" / "ac" / "ac_net_10.e"
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "ac_break.e"
            text = source.read_text(encoding="utf-8")
            switch_start = text.index("<ACSwitch>")
            switch_end = text.index("</ACSwitch>", switch_start) + len("</ACSwitch>")
            break_start = text.index("<ACBreak>")
            break_end = text.index("</ACBreak>", break_start) + len("</ACBreak>")
            text = (
                text[:switch_start]
                + "<ACSwitch>\n@ idx name   i_node j_node status run_stat p q current\n</ACSwitch>\n\n"
                + "<ACBreak>\n@ idx name   i_node j_node status run_stat p q current\n"
                + "# 0   brk_7_8 7      8      1      1        0 0 0\n"
                + "</ACBreak>"
                + text[break_end:]
            )
            case_path.write_text(text, encoding="utf-8")

            ppc = build_ac_ppc_from_e_file(case_path)
            network = ACPowerNetwork()
            with contextlib.redirect_stdout(io.StringIO()):
                network.read_from_file(case_path)
                network.topo()

        self.assertEqual(0, ppc["switch"].shape[0])
        self.assertEqual(1, ppc["break"].shape[0])
        self.assertEqual("brk_7_8", ppc["break_name"][0])
        self.assertEqual(7, int(ppc["break"][0, SWITCH_COLS["i_node"]]))
        self.assertEqual(8, int(ppc["break"][0, SWITCH_COLS["j_node"]]))
        self.assertEqual(1, len(network.breakers))
        self.assertIsInstance(network.breakers[0], ACBreak)
        self.assertEqual("brk_7_8", network.breakers[0].name)
        self.assertTrue(network.node_dict[7].isl_obj is network.node_dict[8].isl_obj)

    def test_object_multi_island_flow_keeps_breaker_zero_ties(self):
        from scipy.sparse.linalg import MatrixRankWarning, spsolve
        from ac_lf import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "hybrid" / "qinling_100.e"
        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)
            network.topo()
            calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50)
            calc.prepare()
            f = calc.get_f(calc.x)
            j = calc.get_jacobi(calc.x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            delta = spsolve(j, f)

        self.assertGreater(calc.N_phi, 0)
        self.assertFalse(any(item.category is MatrixRankWarning for item in caught))
        self.assertTrue(np.isfinite(delta).all())

    def test_object_y_matrix_uses_vectorized_branch_stamps(self):
        import ac_lf
        from ac_lf import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"
        expected_network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            expected_network.read_from_file(case_path)
            expected_network.topo()
            expected_calc = ACPowerFlowCalc(expected_network, tol=1e-8, max_iter=50)
            expected_calc.prepare()

        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)
            network.topo()
        calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50)

        original_stamp = ac_lf.matpower_branch_stamp

        def fail_scalar_stamp(*args, **kwargs):
            raise AssertionError("AC Y matrix should use vectorized branch stamps")

        ac_lf.matpower_branch_stamp = fail_scalar_stamp
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
        finally:
            ac_lf.matpower_branch_stamp = original_stamp

        np.testing.assert_allclose(calc.Y.toarray(), expected_calc.Y.toarray(), atol=1e-12)

    def test_ppc_flow_matches_object_flow_for_ieee300(self):
        from ac_array_model import BUS_COLS, build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"

        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)
            network.topo()
            warnings, errors = network.check_topo()
            object_calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50)
            object_calc.prepare()
            object_rc = object_calc.run()
        self.assertEqual([], errors)
        self.assertEqual(0, object_rc)
        self.assertTrue(object_calc.converged)

        ppc = build_ac_ppc_from_e_file(case_path)
        ppc_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            ppc_calc.prepare()
            ppc_rc = ppc_calc.run()
        self.assertEqual(0, ppc_rc)
        self.assertTrue(ppc_calc.converged)

        self.assertFalse(object_calc.keep_node_objects)
        self.assertEqual([], object_calc.node_list)
        object_voltage = object_calc.result["bus"][:, BUS_COLS["voltage"]]
        object_angle = object_calc.result["bus"][:, BUS_COLS["angle"]]

        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["voltage"]], object_voltage, atol=1e-10)
        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["angle"]], object_angle, atol=1e-10)

    def test_ac_lf_rejects_pq_decoupled_algorithm(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)

        with self.assertRaises(TypeError):
            ACPowerFlowCalc(ppc, algorithm="pq")

    def test_invalid_power_flow_algorithm_parameter_is_removed(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        with self.assertRaises(TypeError):
            ACPowerFlowCalc(ppc, algorithm="bad")

    def test_ac_power_flow_can_load_e_file_through_efile_reader_path(self):
        import ac_lf
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        previous_builder = getattr(ac_lf, "build_ac_ppc_from_network", None)
        previous_common_loader = ac_lf.build_ac_ppc_with_topology_from_e_file
        calls = []

        def counted_common_loader(path):
            calls.append(Path(path).name)
            return previous_common_loader(path)

        def reject_network_builder(*_args, **_kwargs):
            raise AssertionError("AC LF should use the shared E-to-PPC topology loader")

        ac_lf.build_ac_ppc_from_network = reject_network_builder
        ac_lf.build_ac_ppc_with_topology_from_e_file = counted_common_loader
        try:
            ppc = ac_lf.load_ac_ppc_from_e_file(case_path)
            calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        finally:
            if previous_builder is None:
                del ac_lf.build_ac_ppc_from_network
            else:
                ac_lf.build_ac_ppc_from_network = previous_builder
            ac_lf.build_ac_ppc_with_topology_from_e_file = previous_common_loader

        self.assertEqual(["ieee300.e"], calls)
        self.assertFalse(hasattr(calc, "array_mode"))
        self.assertEqual("ac_ppc_v1", calc.ppc["format"])
        self.assertIn("_topology_arrays", calc.ppc)

    def test_network_input_uses_array_kernel_and_writes_back_objects(self):
        from ac_array_model import BUS_COLS, GEN_COLS
        from ac_lf import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)

        calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50)
        self.assertFalse(hasattr(calc, "array_mode"))

        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        bus_by_idx = {
            int(row[BUS_COLS["idx"]]): row
            for row in calc.result["bus"]
        }
        gen_by_idx = {
            int(row[GEN_COLS["idx"]]): row
            for row in calc.result["gen"]
        }
        first_node = network.nodes[0]
        first_gen = network.generators[0]
        self.assertAlmostEqual(bus_by_idx[first_node.idx][BUS_COLS["voltage"]], first_node.voltage)
        self.assertAlmostEqual(bus_by_idx[first_node.idx][BUS_COLS["angle"]], first_node.angle)
        self.assertAlmostEqual(gen_by_idx[first_gen.idx][GEN_COLS["p"]], first_gen.p)
        self.assertAlmostEqual(gen_by_idx[first_gen.idx][GEN_COLS["q"]], first_gen.q)

    def test_build_ac_ppc_from_network_reflects_network_objects(self):
        from ac_array_model import BUS_COLS, build_ac_ppc_from_e_file, build_ac_ppc_from_network
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        expected = build_ac_ppc_from_e_file(case_path)
        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)

        network.nodes[0].voltage = 0.987654
        ppc = build_ac_ppc_from_network(network)

        for key in ("branch", "transformer", "gen", "load", "shunt", "zero_branch", "switch", "break"):
            np.testing.assert_allclose(ppc[key], expected[key])
        self.assertEqual(0.987654, ppc["bus"][0, BUS_COLS["voltage"]])
        np.testing.assert_allclose(ppc["bus"][1:], expected["bus"][1:])
        np.testing.assert_array_equal(ppc["bus_name"], expected["bus_name"])

    def test_ac_generator_p_max_round_trips_between_e_ppc_and_object_model(self):
        from ac_array_model import (
            GEN_COLS,
            build_ac_network_from_ppc,
            build_ac_ppc_from_e_file,
            build_ac_ppc_from_network,
            build_matpower_ppc_from_ac_ppc,
        )

        content = """<Model>
@ path name p_base u_unit p_unit i_unit
# IEEE p_max_case 100 V kW A
</Model>

<ACNode>
@ idx name vbase run_stat
# 1 bus_1 380 1
</ACNode>

<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha p_max run_stat
# 7 gen_7 1 PV 20 0 380 1 80 1
</ACGenerator>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "p_max.e"
            case_path.write_text(content, encoding="utf-8")
            ppc = build_ac_ppc_from_e_file(case_path)

        self.assertAlmostEqual(0.8, ppc["gen"][0, GEN_COLS["p_max"]])

        network = build_ac_network_from_ppc(ppc)
        self.assertAlmostEqual(0.8, network.generators[0].p_max)

        round_trip = build_ac_ppc_from_network(network)
        self.assertAlmostEqual(0.8, round_trip["gen"][0, GEN_COLS["p_max"]])

        matpower = build_matpower_ppc_from_ac_ppc(round_trip)
        self.assertAlmostEqual(80.0, matpower["gen"][0, 8])

    def test_ac_generator_without_p_max_uses_nan_capacity(self):
        from ac_array_model import GEN_COLS, build_ac_ppc_from_e_file

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ac_net_10.e")

        self.assertTrue(np.isnan(ppc["gen"][:, GEN_COLS["p_max"]]).all())

    def test_legacy_ac_ppc_generator_width_is_upgraded_before_use(self):
        from ac_array_model import (
            CTRL_PV,
            GEN_COLS,
            build_ac_network_from_ppc,
            build_ac_ppc_from_e_file,
            build_matpower_ppc_from_ac_ppc,
        )
        from model.topology import prepare_ac_topology_ppc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ac_net_10.e")
        ppc["gen"][:, GEN_COLS["control_type"]] = CTRL_PV
        ppc["gen"] = ppc["gen"][:, :11].copy()
        ppc.pop("_topology_input", None)

        arrays = prepare_ac_topology_ppc(ppc)

        self.assertEqual(len(GEN_COLS), ppc["gen"].shape[1])
        self.assertTrue(np.isnan(ppc["gen"][:, GEN_COLS["p_max"]]).all())
        self.assertTrue(arrays.island_alive_mask.any())
        self.assertTrue(ppc["_auto_slack_gen_rows"].size)

        network = build_ac_network_from_ppc(ppc)
        self.assertTrue(all(gen.p_max is None for gen in network.generators))
        self.assertTrue(build_matpower_ppc_from_ac_ppc(ppc)["gen"].size)

    def test_build_ac_ppc_from_e_file_builds_directly_from_loaded_rows(self):
        import ac_array_model

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        original_network_builder = ac_array_model.build_ac_ppc_from_network
        original_model_builder = ac_array_model._build_ac_ppc_from_model
        original_file_factory = ac_array_model.efile_factory_from_file
        original_rows_factory = ac_array_model.efile_factory_from_rows

        def reject_object_path(*_args, **_kwargs):
            raise AssertionError("AC E-file PPC load should not build dynamic model/network objects")

        ac_array_model.build_ac_ppc_from_network = reject_object_path
        ac_array_model._build_ac_ppc_from_model = reject_object_path
        ac_array_model.efile_factory_from_file = reject_object_path
        ac_array_model.efile_factory_from_rows = reject_object_path
        try:
            ppc = ac_array_model.build_ac_ppc_from_e_file(case_path)
        finally:
            ac_array_model.build_ac_ppc_from_network = original_network_builder
            ac_array_model._build_ac_ppc_from_model = original_model_builder
            ac_array_model.efile_factory_from_file = original_file_factory
            ac_array_model.efile_factory_from_rows = original_rows_factory

        self.assertEqual("ac_ppc_v1", ppc["format"])
        self.assertEqual(300, ppc["bus"].shape[0])

    def test_ac_ppc_matpower_mat_read_write(self):
        from scipy.io import loadmat, savemat

        from ac_array_model import (
            BRANCH_COLS,
            BUS_COLS,
            GEN_COLS,
            LOAD_COLS,
            SHUNT_COLS,
            TRANSFORMER_COLS,
            build_ac_ppc_from_mat_file,
            save_ac_ppc_to_mat_file,
        )

        bus = np.array(
            [
                [1, 3, 0, 0, 0, 0, 1, 1.02, 5.0, 230, 1, 1.1, 0.9],
                [2, 1, 50, 20, 1.0, 3.0, 1, 0.98, -2.0, 230, 1, 1.1, 0.9],
            ],
            dtype=np.float64,
        )
        gen = np.zeros((1, 21), dtype=np.float64)
        gen[0, 0] = 1
        gen[0, 1] = 80
        gen[0, 2] = 15
        gen[0, 5] = 1.02
        gen[0, 6] = 100
        gen[0, 7] = 1
        gen[0, 8] = 200
        branch = np.array(
            [
                [1, 2, 0.01, 0.05, 0.02, 0, 0, 0, 0, 0, 1, -360, 360],
                [1, 2, 0.02, 0.08, 0.04, 0, 0, 0, 1.1, 10, 1, -360, 360],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mat_path = Path(tmp_dir) / "case2.mat"
            savemat(str(mat_path), {"mpc": {"version": "2", "baseMVA": 100.0, "bus": bus, "gen": gen, "branch": branch}})

            ppc = build_ac_ppc_from_mat_file(mat_path)
            self.assertEqual("ac_ppc_v1", ppc["format"])
            self.assertEqual(100.0, ppc["base"]["p_base"])
            self.assertEqual((2, len(BUS_COLS)), ppc["bus"].shape)
            self.assertEqual((1, len(BRANCH_COLS)), ppc["branch"].shape)
            self.assertEqual((1, len(TRANSFORMER_COLS)), ppc["transformer"].shape)
            self.assertEqual((1, len(GEN_COLS)), ppc["gen"].shape)
            self.assertEqual((1, len(LOAD_COLS)), ppc["load"].shape)
            self.assertEqual((2, len(SHUNT_COLS)), ppc["shunt"].shape)
            self.assertAlmostEqual(0.5, ppc["load"][0, LOAD_COLS["pbase"]])
            self.assertAlmostEqual(0.2, ppc["load"][0, LOAD_COLS["qbase"]])
            self.assertAlmostEqual(0.01, ppc["shunt"][0, SHUNT_COLS["g_set"]])
            self.assertAlmostEqual(0.03, ppc["shunt"][0, SHUNT_COLS["b_set"]])
            self.assertAlmostEqual(0.02, ppc["transformer"][0, TRANSFORMER_COLS["bt"]])
            self.assertAlmostEqual(1.1, ppc["transformer"][0, TRANSFORMER_COLS["tap"]])
            self.assertAlmostEqual(10.0, ppc["transformer"][0, TRANSFORMER_COLS["shift"]])
            self.assertAlmostEqual(2.0, ppc["gen"][0, GEN_COLS["p_max"]])

            out_path = Path(tmp_dir) / "roundtrip.mat"
            returned = save_ac_ppc_to_mat_file(ppc, out_path)
            self.assertEqual(out_path, returned)
            saved = loadmat(str(out_path), squeeze_me=True, struct_as_record=False)
            self.assertIn("mpc", saved)
            saved_mpc = saved["mpc"]
            self.assertAlmostEqual(100.0, float(saved_mpc.baseMVA))
            self.assertEqual((2, 13), np.asarray(saved_mpc.bus).shape)
            self.assertEqual((1, 21), np.atleast_2d(np.asarray(saved_mpc.gen)).shape)
            self.assertEqual((2, 13), np.asarray(saved_mpc.branch).shape)
            self.assertAlmostEqual(200.0, np.atleast_2d(np.asarray(saved_mpc.gen))[0, 8])

    def test_ac_ppc_matpower_export_maps_acac_ports_to_gen_or_load(self):
        from ac_array_model import (
            ACAC_COLS,
            BUS_COLS,
            build_matpower_ppc_from_ac_ppc,
        )

        bus = np.zeros((2, len(BUS_COLS)), dtype=np.float64)
        bus[:, BUS_COLS["idx"]] = [1, 2]
        bus[:, BUS_COLS["vbase"]] = [230.0, 230.0]
        bus[:, BUS_COLS["voltage"]] = [1.02, 0.98]
        bus[:, BUS_COLS["run_stat"]] = 1.0

        acac = np.zeros((1, len(ACAC_COLS)), dtype=np.float64)
        acac[0, ACAC_COLS["idx"]] = 1
        acac[0, ACAC_COLS["i_node"]] = 1
        acac[0, ACAC_COLS["j_node"]] = 2
        acac[0, ACAC_COLS["i_control_type"]] = 1
        acac[0, ACAC_COLS["j_control_type"]] = 0
        acac[0, ACAC_COLS["i_v_set"]] = 1.04
        acac[0, ACAC_COLS["run_stat"]] = 1.0
        acac[0, ACAC_COLS["i_p"]] = -0.30
        acac[0, ACAC_COLS["i_q"]] = -0.10
        acac[0, ACAC_COLS["j_p"]] = 0.31
        acac[0, ACAC_COLS["j_q"]] = 0.12

        ppc = {
            "format": "ac_ppc_v1",
            "base": {"p_base": 100.0},
            "bus": bus,
            "acac": acac,
        }

        matpower = build_matpower_ppc_from_ac_ppc(ppc)

        self.assertEqual((2, 13), matpower["bus"].shape)
        self.assertEqual((1, 21), matpower["gen"].shape)
        self.assertEqual(2.0, matpower["bus"][0, 1])
        self.assertAlmostEqual(30.0, matpower["gen"][0, 1])
        self.assertAlmostEqual(10.0, matpower["gen"][0, 2])
        self.assertAlmostEqual(1.04, matpower["gen"][0, 5])
        self.assertAlmostEqual(31.0, matpower["bus"][1, 2])
        self.assertAlmostEqual(12.0, matpower["bus"][1, 3])

    def test_ac_and_hybrid_lf_load_matpower_m_files_as_ac_ppc(self):
        import ac_lf
        import hybrid_lf
        from ac_array_model import BUS_COLS
        from ac_lf import ACPowerFlowCalc
        from hybrid_lf import HybridPowerFlowCalc

        case_text = """function mpc = case2
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
  1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;
  2 1 20 8 0 0 1 1 0 230 1 1.1 0.9;
];
mpc.gen = [
  1 30 0 100 -100 1 100 1 100 0 0 0 0 0 0 0 0 0 0 0 0;
];
mpc.branch = [
  1 2 0.01 0.05 0.02 0 0 0 0 0 1 -360 360;
];
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "case2.m"
            case_path.write_text(case_text, encoding="utf-8")

            ppc = ac_lf.load_ac_ppc_from_e_file(case_path)
            self.assertEqual("ac_ppc_v1", ppc["format"])
            self.assertIn("_topology_arrays", ppc)
            self.assertEqual((2, len(BUS_COLS)), ppc["bus"].shape)
            ac_calc = ACPowerFlowCalc(ppc, result_mode="array", verbose=False)
            self.assertEqual(0, ac_calc.run())
            self.assertTrue(ac_calc.converged)

            network = hybrid_lf._read_lf_network_from_file(case_path)
            self.assertIsNotNone(network._ac_ppc)
            self.assertIsNone(network.ppc.get("dc"))
            hybrid_calc = HybridPowerFlowCalc(network, result_mode="array", verbose=False)
            self.assertEqual(0, hybrid_calc.run())
            self.assertTrue(hybrid_calc.converged)

    def test_ac_lf_script_entry_loads_e_file_once_through_array_path(self):
        source = (ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "ac_lf.py").read_text(
            encoding="utf-8"
        )
        main_block = source.split("def main", 1)[1].split('if __name__ == "__main__":', 1)[0]

        self.assertIn("load_ac_ppc_from_e_file(args.file)", main_block)
        self.assertIn("ACPowerFlowCalc(", main_block)
        self.assertIn("verbose=not args.quiet", main_block)
        self.assertIn("calc.run()", main_block)
        self.assertNotIn("calc.prepare()", main_block)
        self.assertNotIn("_run_with_optional_output", main_block)
        self.assertNotIn("--algorithm", main_block)
        self.assertNotIn("ACPowerFlowCalc.from_e_file", main_block)
        self.assertNotIn("read_from_file", main_block)
        self.assertNotIn("ACPowerNetwork", main_block)

    def test_ac_lf_pq_decoupled_code_is_removed(self):
        source = (ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "ac_lf.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("_run_pq_decoupled", source)
        self.assertNotIn("_cache_pq_decoupled_matrices", source)
        self.assertNotIn("_build_fast_decoupled_b_matrix", source)
        self.assertNotIn("pq_Bp", source)
        self.assertNotIn("pq_Bpp", source)
        self.assertNotIn("used_algorithm", source)
        self.assertNotIn("self.algorithm", source)
        self.assertNotIn('algorithm == "pq"', source)
        self.assertNotIn('"pq"', source.split("def main", 1)[1].split('if __name__ == "__main__":', 1)[0])

    def test_ac_run_prepares_when_called_directly(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e")
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="array")

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertGreater(calc.x.size, 0)
        self.assertIn("bus", calc.result)

    def test_ac_node_type_uses_numeric_codes_in_solver_path(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e")
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertTrue(np.issubdtype(calc.node_type.dtype, np.integer))
        self.assertEqual((calc.N,), calc.node_type.shape)
        self.assertGreater(np.unique(calc.node_type).size, 1)

        checked_sources = [
            ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "ac_lf.py",
            ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "hybrid_lf.py",
        ]
        offenders = []
        for source_path in checked_sources:
            for line_no, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
                if "node_type" not in line:
                    continue
                if "label" in line.lower() or "LABEL" in line:
                    continue
                if any(token in line for token in ("'PQ'", '"PQ"', "'PV'", '"PV"', "'SLACK'", '"SLACK"')):
                    offenders.append(f"{source_path.name}:{line_no}:{line.strip()}")
        self.assertEqual([], offenders)

    def test_full_ac_result_exposes_array_result_tables(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e")
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="full")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertIsNotNone(calc.lf_result)
        self.assertIn("bus", calc.lf_result.arrays)
        self.assertIn("branch", calc.lf_result.arrays)
        self.assertIs(calc.lf_result.arrays["bus"], calc.result["bus"])
        self.assertIs(calc.lf_result.arrays["branch"], calc.result["branch"])

    def test_ac_lf_benchmark_uses_nr_only(self):
        from lfcore import ac_lf_benchmark

        with contextlib.redirect_stdout(io.StringIO()):
            result = ac_lf_benchmark.run_case("ieee300", repeats=1, profile=False)

        self.assertNotIn("algorithm", result)
        self.assertEqual(0, result["rc"])
        self.assertTrue(result["converged"])

    def test_state_extraction_reuses_iteration_work_arrays(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        theta1, voltage1, _, _ = calc._extract_state_vars(calc.x, update_cache=True)
        theta2, voltage2, _, _ = calc._extract_state_vars(calc.x, update_cache=True)

        self.assertIs(theta1, theta2)
        self.assertIs(voltage1, voltage2)
        self.assertIs(theta2, calc._cache["theta"])
        self.assertIs(voltage2, calc._cache["V"])

    def test_ppc_standard_jacobian_avoids_sparse_block_stack(self):
        import ac_lf
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
        self.assertEqual(0, calc.N_phi)
        self.assertEqual(0, calc.full_jac_raw_data.size)
        self.assertEqual(0, calc.full_jac_csr_indices.size)

        original_hstack = ac_lf.hstack
        original_vstack = ac_lf.vstack

        def reject_stack(*_args, **_kwargs):
            raise AssertionError("standard PPC Jacobian should be assembled directly")

        ac_lf.hstack = reject_stack
        ac_lf.vstack = reject_stack
        try:
            calc.get_jacobi(calc.x)
        finally:
            ac_lf.hstack = original_hstack
            ac_lf.vstack = original_vstack

    def test_ppc_standard_jacobian_caches_coordinate_pattern(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        rows = calc.standard_jac_rows
        cols = calc.standard_jac_cols
        self.assertIsInstance(rows, np.ndarray)
        self.assertIsInstance(cols, np.ndarray)
        self.assertEqual(np.int32, rows.dtype)
        self.assertEqual(np.int32, cols.dtype)

        with contextlib.redirect_stdout(io.StringIO()):
            first = calc.get_jacobi(calc.x)
            second = calc.get_jacobi(calc.x)

        self.assertIs(rows, calc.standard_jac_rows)
        self.assertIs(cols, calc.standard_jac_cols)
        np.testing.assert_array_equal(first.indptr, second.indptr)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_allclose(first.data, second.data, atol=1e-12)

    def test_newton_system_returns_solver_ready_csc_jacobian(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        _f, solver_jac = calc._build_newton_system(calc.x)
        public_jac = calc.get_jacobi(calc.x)

        self.assertEqual("csc", solver_jac.format)
        np.testing.assert_allclose(solver_jac.toarray(), public_jac.toarray(), atol=1e-12)

    def test_ppc_zero_branch_jacobian_caches_coordinate_pattern(self):
        from hybrid_array_model import build_hybrid_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        _, ppc = build_hybrid_ppc_from_e_file(ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = ACPowerFlowCalc(ppc["ac"], tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertGreater(calc.N_phi, 0)
        for name in (
            "zero_top_left_rows",
            "zero_top_left_cols",
            "zero_top_right_rows",
            "zero_top_right_cols",
            "zero_bottom_left_rows",
            "zero_bottom_left_cols",
            "zero_bottom_right_rows",
            "zero_bottom_right_cols",
        ):
            self.assertIsInstance(getattr(calc, name), np.ndarray)
        self.assertGreater(calc.zero_top_right_rows.size, 0)
        self.assertGreater(calc.zero_bottom_right_rows.size, 0)

        with contextlib.redirect_stdout(io.StringIO()):
            first = calc.get_jacobi(calc.x)
            second = calc.get_jacobi(calc.x)

        np.testing.assert_array_equal(first.indptr, second.indptr)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_allclose(first.data, second.data, atol=1e-12)

    def test_ppc_zero_branch_jacobian_reuses_precomputed_csr_pattern(self):
        import ac_lf
        from hybrid_array_model import build_hybrid_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        _, ppc = build_hybrid_ppc_from_e_file(ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = ACPowerFlowCalc(ppc["ac"], tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertGreater(calc.N_phi, 0)
        expected = calc.get_jacobi(calc.x).toarray()
        self.assertGreater(calc.full_jac_csr_indices.size, 0)

        original_coo_matrix = ac_lf.coo_matrix
        original_hstack = ac_lf.hstack
        original_vstack = ac_lf.vstack

        def reject_sparse_rebuild(*_args, **_kwargs):
            raise AssertionError("AC zero-branch Jacobian should refresh precomputed CSR data")

        ac_lf.coo_matrix = reject_sparse_rebuild
        ac_lf.hstack = reject_sparse_rebuild
        ac_lf.vstack = reject_sparse_rebuild
        try:
            actual = calc.get_jacobi(calc.x).toarray()
        finally:
            ac_lf.coo_matrix = original_coo_matrix
            ac_lf.hstack = original_hstack
            ac_lf.vstack = original_vstack

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_full_jacobian_csr_pattern_maps_duplicate_raw_coordinates(self):
        from scipy.sparse import coo_matrix, csr_matrix

        from ac_lf import _build_csr_pattern_from_raw_coords

        raw_rows = np.asarray([1, 0, 1, 0, 1, 2], dtype=np.int32)
        raw_cols = np.asarray([2, 1, 2, 1, 3, 0], dtype=np.int32)
        raw_data = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)

        indices, indptr, raw_to_csr = _build_csr_pattern_from_raw_coords(
            raw_rows,
            raw_cols,
            n_rows=3,
        )
        actual_data = np.bincount(raw_to_csr, weights=raw_data, minlength=indices.size)
        actual = csr_matrix((actual_data, indices, indptr), shape=(3, 4))
        expected = coo_matrix((raw_data, (raw_rows, raw_cols)), shape=(3, 4)).tocsr()

        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.data, expected.data, atol=1e-12)

    def test_ppc_zero_branch_jacobian_uses_precomputed_kind_groups(self):
        from hybrid_array_model import build_hybrid_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        _, ppc = build_hybrid_ppc_from_e_file(ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        calc = ACPowerFlowCalc(ppc["ac"], tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertGreater(calc.N_phi, 0)
        expected = calc.get_jacobi(calc.x).toarray()

        def reject_kind_mask_dispatch(*_args, **_kwargs):
            raise AssertionError("zero-branch Jacobian should use precomputed kind groups")

        calc._fill_indexed_kind_data = reject_kind_mask_dispatch
        actual = calc.get_jacobi(calc.x).toarray()

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_verbose_false_suppresses_ac_prepare_and_iteration_output(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ieee300.e")
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="none", verbose=False)

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            calc.prepare()
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertEqual("", captured.getvalue())

    def test_result_mode_skips_full_ac_result_backfill(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        ppc = build_ac_ppc_from_e_file(ROOT_DIR / "data" / "model" / "ac" / "ieee300.e")
        ppc.pop("_pf_static", None)

        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="none")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        none_calls = []
        original_none_write_back_ppc = calc._write_back_ppc

        def counted_none_write_back_ppc():
            none_calls.append(calc.result_mode)
            return original_none_write_back_ppc()

        calc._write_back_ppc = counted_none_write_back_ppc
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(["none"], none_calls)
        self.assertEqual({}, calc.result)
        self.assertIsNone(getattr(calc, "lf_result", None))
        self.assertTrue(hasattr(calc, "x"))

        summary_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="summary")
        with contextlib.redirect_stdout(io.StringIO()):
            summary_calc.prepare()
        summary_calls = []
        original_summary_write_back_ppc = summary_calc._write_back_ppc

        def counted_summary_write_back_ppc():
            summary_calls.append(summary_calc.result_mode)
            return original_summary_write_back_ppc()

        summary_calc._write_back_ppc = counted_summary_write_back_ppc
        with contextlib.redirect_stdout(io.StringIO()):
            rc = summary_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(summary_calc.converged)
        self.assertEqual(["summary"], summary_calls)
        self.assertEqual({"node_id", "voltage", "angle", "summary"}, set(summary_calc.result))
        self.assertEqual(summary_calc.N, summary_calc.result["voltage"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["angle"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["node_id"].size)

        array_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            array_calc.prepare()
        self.assertEqual([], array_calc.node_list)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = array_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(array_calc.converged)
        self.assertIn("bus", array_calc.result)
        self.assertIn("branch", array_calc.result)
        self.assertIsNone(getattr(array_calc, "lf_result", None))

    def test_ppc_prepare_uses_shared_ppc_topology(self):
        from model import ppc_topology
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        ppc.pop("_pf_static", None)
        ppc.pop("_topology_arrays", None)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)

        original_prepare_topology = ppc_topology.network_topology.prepare_ac_topology_ppc
        calls = []

        def wrapped_prepare_topology(arg):
            calls.append(arg)
            return original_prepare_topology(arg)

        ppc_topology.network_topology.prepare_ac_topology_ppc = wrapped_prepare_topology
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
        finally:
            ppc_topology.network_topology.prepare_ac_topology_ppc = original_prepare_topology

        self.assertEqual([ppc], calls)
        self.assertIn("_topology_arrays", ppc)
        self.assertGreater(calc.N, 0)

    def test_ppc_prepare_ignores_static_cache_field(self):
        from ac_lf import ACPowerFlowCalc
        from hybrid_array_model import build_hybrid_ppc_from_e_file

        _, hybrid_ppc = build_hybrid_ppc_from_e_file(ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        ppc = hybrid_ppc["ac"]
        first = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            first.prepare()

        self.assertNotIn("_pf_static", ppc)

        stale_static = {"N": -1}
        ppc["_pf_static"] = stale_static
        second = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            second.prepare()

        self.assertIs(stale_static, ppc["_pf_static"])
        self.assertEqual(first.total_vars, second.total_vars)
        self.assertEqual(first.total_eq, second.total_eq)

    def test_efile_factory_from_file_returns_fresh_model_without_cache(self):
        from efile_read import efile_factory_from_file

        content = "\n".join(
            [
                "<PowerBase>",
                "@ p_base u_unit p_unit i_unit",
                "# 100 kV MW kA",
                "</PowerBase>",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "factory.e"
            path.write_text(content, encoding="utf-8")
            first = efile_factory_from_file(path)
            second = efile_factory_from_file(path)

        self.assertIsNot(first, second)
        self.assertEqual(first.PowerBase[0].p_base, second.PowerBase[0].p_base)


if __name__ == "__main__":
    unittest.main()
