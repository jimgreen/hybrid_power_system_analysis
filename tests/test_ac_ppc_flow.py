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
        from ac_array_model import build_ac_ppc_from_e_file
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

        object_voltage = np.asarray([node.voltage for node in object_calc.node_list])
        object_angle = np.asarray([node.angle for node in object_calc.node_list])

        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["voltage"]], object_voltage, atol=1e-10)
        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["angle"]], object_angle, atol=1e-10)

    def test_ppc_flow_supports_pq_decoupled_algorithm(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)

        nr_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        pq_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=80, algorithm="pq")
        with contextlib.redirect_stdout(io.StringIO()):
            nr_calc.prepare()
            nr_rc = nr_calc.run()
            pq_calc.prepare()
            pq_rc = pq_calc.run()

        self.assertEqual("nr", nr_calc.algorithm)
        self.assertEqual("pq", pq_calc.algorithm)
        self.assertEqual(0, nr_rc)
        self.assertEqual(0, pq_rc)
        self.assertTrue(pq_calc.converged)
        self.assertEqual("pq", pq_calc.used_algorithm)
        self.assertLess(pq_calc.iterations, 50)

        cols = ppc["bus_cols"]
        np.testing.assert_allclose(
            pq_calc.result["bus"][:, cols["voltage"]],
            nr_calc.result["bus"][:, cols["voltage"]],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pq_calc.result["bus"][:, cols["angle"]],
            nr_calc.result["bus"][:, cols["angle"]],
            atol=1e-5,
        )

    def test_invalid_power_flow_algorithm_is_rejected(self):
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        with self.assertRaises(ValueError):
            ACPowerFlowCalc(ppc, algorithm="bad")

    def test_ac_power_flow_can_load_e_file_through_efile_reader_path(self):
        import ac_lf
        from ac_array_model import build_ac_ppc_from_network as original_builder
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        previous_builder = getattr(ac_lf, "build_ac_ppc_from_network", None)
        previous_file_builder = getattr(ac_lf, "build_ac_ppc_from_e_file", None)
        calls = []

        def counted_builder(network):
            calls.append((network.__class__.__name__, len(network.nodes)))
            return original_builder(network)

        def reject_file_builder(*_args, **_kwargs):
            raise AssertionError("AC LF should build ppc from an already loaded ACPowerNetwork")

        ac_lf.build_ac_ppc_from_network = counted_builder
        if previous_file_builder is not None:
            ac_lf.build_ac_ppc_from_e_file = reject_file_builder
        try:
            ppc = ac_lf.load_ac_ppc_from_e_file(case_path)
            calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        finally:
            if previous_builder is None:
                del ac_lf.build_ac_ppc_from_network
            else:
                ac_lf.build_ac_ppc_from_network = previous_builder
            if previous_file_builder is not None:
                ac_lf.build_ac_ppc_from_e_file = previous_file_builder

        self.assertEqual([("ACPowerNetwork", 300)], calls)
        self.assertTrue(calc.array_mode)
        self.assertEqual("ac_ppc_v1", calc.ppc["format"])

    def test_network_input_uses_array_kernel_and_writes_back_objects(self):
        from ac_array_model import BUS_COLS, GEN_COLS
        from ac_lf import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        network = ACPowerNetwork()
        with contextlib.redirect_stdout(io.StringIO()):
            network.read_from_file(case_path)

        calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50)
        self.assertTrue(calc.array_mode)

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

    def test_build_ac_ppc_from_e_file_delegates_through_network_model(self):
        import ac_array_model

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ac_array_model.clear_ac_ppc_cache(case_path)
        original = ac_array_model.build_ac_ppc_from_network
        calls = []

        def counted_builder(network):
            calls.append((network.__class__.__name__, len(network.nodes)))
            return original(network)

        ac_array_model.build_ac_ppc_from_network = counted_builder
        try:
            ppc = ac_array_model.build_ac_ppc_from_e_file(case_path)
        finally:
            ac_array_model.build_ac_ppc_from_network = original

        self.assertEqual([("ACPowerNetwork", 300)], calls)
        self.assertEqual("ac_ppc_v1", ppc["format"])

    def test_ac_lf_script_entry_loads_e_file_once_through_array_path(self):
        source = (ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "ac_lf.py").read_text(
            encoding="utf-8"
        )
        main_block = source.split("def main", 1)[1].split('if __name__ == "__main__":', 1)[0]

        self.assertIn("load_ac_ppc_from_e_file(args.file)", main_block)
        self.assertIn("ACPowerFlowCalc(", main_block)
        self.assertIn("verbose=not args.quiet", main_block)
        self.assertIn("calc.prepare()", main_block)
        self.assertIn("calc.run()", main_block)
        self.assertNotIn("_run_with_optional_output", main_block)
        self.assertNotIn("ACPowerFlowCalc.from_e_file", main_block)
        self.assertNotIn("read_from_file", main_block)
        self.assertNotIn("ACPowerNetwork", main_block)

    def test_ac_lf_benchmark_accepts_algorithm_option(self):
        from lfcore import ac_lf_benchmark

        with contextlib.redirect_stdout(io.StringIO()):
            result = ac_lf_benchmark.run_case("ieee300", repeats=1, profile=False, algorithm="pq")

        self.assertEqual("pq", result["algorithm"])
        self.assertEqual(0, result["rc"])
        self.assertTrue(result["converged"])
        self.assertLess(result["iterations"], 50)

    def test_pq_algorithm_reuses_fixed_b_matrix_factorization(self):
        import ac_lf
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)

        original_splu = ac_lf.splu
        original_spsolve = ac_lf.spsolve
        factor_shapes = []

        def wrapped_splu(matrix, *args, **kwargs):
            factor_shapes.append(matrix.shape)
            return original_splu(matrix, *args, **kwargs)

        def reject_spsolve(*_args, **_kwargs):
            raise AssertionError("PQ iterations should reuse fixed B' and B'' factorizations")

        ac_lf.splu = wrapped_splu
        try:
            calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=80, algorithm="pq")
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
            ac_lf.spsolve = reject_spsolve
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = calc.run()
            finally:
                ac_lf.spsolve = original_spsolve
        finally:
            ac_lf.splu = original_splu

        self.assertEqual(0, rc)
        self.assertEqual("pq", calc.used_algorithm)
        self.assertEqual([calc.pq_Bp.shape, calc.pq_Bpp.shape], factor_shapes)

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

        def reject_full_backfill():
            raise AssertionError("result_mode='none' should skip full AC result backfill")

        calc._write_back_ppc = reject_full_backfill
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual({}, calc.result)
        self.assertIsNone(getattr(calc, "lf_result", None))
        self.assertTrue(hasattr(calc, "x"))

        summary_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="summary")
        with contextlib.redirect_stdout(io.StringIO()):
            summary_calc.prepare()
        summary_calc._write_back_ppc = reject_full_backfill
        with contextlib.redirect_stdout(io.StringIO()):
            rc = summary_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(summary_calc.converged)
        self.assertEqual({"node_id", "voltage", "angle", "summary"}, set(summary_calc.result))
        self.assertEqual(summary_calc.N, summary_calc.result["voltage"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["angle"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["node_id"].size)

        array_calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            array_calc.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = array_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(array_calc.converged)
        self.assertIn("bus", array_calc.result)
        self.assertIn("branch", array_calc.result)
        self.assertIsNone(getattr(array_calc, "lf_result", None))

    def test_ppc_prepare_uses_sparse_connected_components(self):
        import ac_lf
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_lf import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        ppc.pop("_pf_static", None)
        calc = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)

        original_connected_components = ac_lf.connected_components
        calls = []

        def wrapped_connected_components(*args, **kwargs):
            calls.append(args[0].shape)
            return original_connected_components(*args, **kwargs)

        ac_lf.connected_components = wrapped_connected_components
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
        finally:
            ac_lf.connected_components = original_connected_components

        self.assertEqual([(ppc["bus"].shape[0], ppc["bus"].shape[0])], calls)
        self.assertGreater(calc.N, 0)

    def test_ppc_prepare_reuses_static_cache_for_second_calc(self):
        import ac_lf
        from ac_lf import ACPowerFlowCalc
        from hybrid_array_model import build_hybrid_ppc_from_e_file

        _, hybrid_ppc = build_hybrid_ppc_from_e_file(ROOT_DIR / "data" / "model" / "hybrid" / "hybrid_net_40.e")
        ppc = hybrid_ppc["ac"]
        first = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            first.prepare()

        self.assertIn("_pf_static", ppc)

        original_connected_components = ac_lf.connected_components

        def reject_connected_components(*_args, **_kwargs):
            raise AssertionError("second PPC prepare should reuse cached static data")

        ac_lf.connected_components = reject_connected_components
        try:
            second = ACPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
            with contextlib.redirect_stdout(io.StringIO()):
                second.prepare()
        finally:
            ac_lf.connected_components = original_connected_components

        self.assertEqual(first.total_vars, second.total_vars)
        self.assertEqual(first.total_eq, second.total_eq)

    def test_efile_factory_from_file_returns_fresh_model_without_cache(self):
        from efile_read import efile_factory_from_file

        content = "\n".join(
            [
                "<PowerBase>",
                "@ p_base u_scale p_scale i_scale",
                "# 100 1.0 0.001 1.0",
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
