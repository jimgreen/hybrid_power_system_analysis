import contextlib
import io
import tempfile
import unittest
from pathlib import Path


class DCLargeCaseGenerationTest(unittest.TestCase):
    def test_dc_array_network_replaces_object_model_loader(self):
        from model.dc_array_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()

        self.assertEqual("dc_ppc_v1", network.ppc["format"])
        self.assertEqual(30, len(network.nodes))
        self.assertEqual(37, len(network.branches))
        self.assertEqual("nd_1", network.nodes[0].name)
        self.assertEqual(1.0, network.nodes[0].voltage)
        self.assertEqual(1, network.nodes[0].run_stat)
        self.assertIs(network.branches[0].i_node_obj, network.nodes[0])
        self.assertTrue(any(isl.is_alive for isl in network.islands))

    def test_dc_programs_import_array_model_network(self):
        root = Path(__file__).resolve().parents[1]
        checked_files = [
            root / "benchmarks" / "benchmark_flat_lf_se.py",
            root / "scripts" / "generate_dc_large_cases.py",
            root / "scripts" / "update_meas_from_lf.py",
            root / "src" / "hybrid_power_system_analysis" / "secore" / "dc_se.py",
            root / "src" / "hybrid_power_system_analysis" / "lfcore" / "dc_lf.py",
        ]

        for file_path in checked_files:
            source = file_path.read_text(encoding="utf-8")
            self.assertNotIn("dc_model import DCPowerNetwork", source, str(file_path))
            self.assertNotIn("import dc_model", source, str(file_path))

    def test_dc_solver_accepts_optional_solver_name_and_falls_back(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network, linear_solver="not-installed-solver")

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual("not-installed-solver", calc.linear_solver)

    def test_dc_solver_prepare_uses_array_model_fast_path(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)

        with contextlib.redirect_stdout(io.StringIO()):
            G, x = calc.prepare()

        self.assertTrue(calc.array_mode)
        self.assertEqual((30, 30), G.shape)
        self.assertEqual(70, x.size)
        np.testing.assert_array_equal(calc.branch_i[:3], np.asarray([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(calc.branch_j[:3], np.asarray([1, 2, 3], dtype=np.int32))
        self.assertEqual(9, calc.N_dcdc)

    def test_dc_solver_accepts_ppc_without_network_topology(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        calc = DCPowerFlowCalc(ppc)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.array_mode)
        self.assertTrue(calc.converged)
        self.assertEqual("dc_ppc_v1", calc.ppc["format"])
        self.assertIn("bus", calc.result)
        self.assertEqual(ppc["bus"].shape, calc.result["bus"].shape)

    def test_dc_ppc_prepare_reuses_static_cache_for_second_calc(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        ppc.pop("_dc_pf_static", None)

        with contextlib.redirect_stdout(io.StringIO()):
            cold_G, cold_x = DCPowerFlowCalc(ppc).prepare()
        self.assertIn("_dc_pf_static", ppc)

        original_topology = DCPowerFlowCalc._prepare_direct_ppc_topology

        def reject_topology(*_args, **_kwargs):
            raise AssertionError("warm DC ppc prepare should reuse cached static topology")

        DCPowerFlowCalc._prepare_direct_ppc_topology = reject_topology
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                warm_calc = DCPowerFlowCalc(ppc)
                warm_G, warm_x = warm_calc.prepare()
        finally:
            DCPowerFlowCalc._prepare_direct_ppc_topology = original_topology

        self.assertEqual(cold_G.shape, warm_G.shape)
        self.assertEqual(cold_x.shape, warm_x.shape)
        self.assertTrue(warm_calc.array_mode)
        self.assertEqual(cold_x.tolist(), warm_x.tolist())

        cold_x[0] = 123.0
        with contextlib.redirect_stdout(io.StringIO()):
            _again_G, again_x = DCPowerFlowCalc(ppc).prepare()
        self.assertNotEqual(123.0, again_x[0])

    def test_dc_jacobian_reuses_precomputed_csr_pattern(self):
        import numpy as np
        import lfcore.dc_lf as dc_lf
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        ppc.pop("_dc_pf_static", None)
        calc = DCPowerFlowCalc(ppc)
        with contextlib.redirect_stdout(io.StringIO()):
            G, x = calc.prepare()

        expected = calc.get_jacobi(G, x).toarray()
        self.assertGreater(calc._dc_jac_csr_indices.size, 0)

        original_coo_matrix = dc_lf.coo_matrix

        def reject_coo_matrix(*_args, **_kwargs):
            raise AssertionError("DC Jacobian should refresh precomputed CSR data without COO rebuild")

        dc_lf.coo_matrix = reject_coo_matrix
        try:
            actual = calc.get_jacobi(G, x).toarray()
        finally:
            dc_lf.coo_matrix = original_coo_matrix

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_result_mode_skips_full_dc_result_backfill(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        ppc.pop("_dc_pf_static", None)

        calc = DCPowerFlowCalc(ppc, result_mode="none")

        def reject_full_backfill(_x):
            raise AssertionError("result_mode='none' should skip full DC result backfill")

        calc._write_back_ppc = reject_full_backfill
        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual({}, calc.result)
        self.assertTrue(hasattr(calc, "x"))

        summary_calc = DCPowerFlowCalc(ppc, result_mode="summary")
        summary_calc._write_back_ppc = reject_full_backfill
        with contextlib.redirect_stdout(io.StringIO()):
            rc = summary_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(summary_calc.converged)
        self.assertEqual({"node_id", "voltage", "summary"}, set(summary_calc.result))
        self.assertEqual(summary_calc.N, summary_calc.result["voltage"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["node_id"].size)

        array_calc = DCPowerFlowCalc(ppc, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = array_calc.run()

        self.assertEqual(0, rc)
        self.assertTrue(array_calc.converged)
        self.assertIn("bus", array_calc.result)
        self.assertIn("branch", array_calc.result)
        self.assertIsNone(getattr(array_calc, "lf_result", None))

    def test_dc_power_flow_can_load_e_file_through_efile_reader_path(self):
        import lfcore.dc_lf as dc_lf
        from lfcore.dc_lf import DCPowerFlowCalc

        case_path = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        original_loader = dc_lf._build_dc_ppc_from_e_file
        calls = []

        def counted_loader(file_name):
            calls.append(Path(file_name).name)
            return original_loader(file_name)

        dc_lf._build_dc_ppc_from_e_file = counted_loader
        try:
            ppc = dc_lf.load_dc_ppc_from_e_file(case_path)
            network = dc_lf._dc_network_from_ppc(ppc)
            calc = DCPowerFlowCalc(network)
        finally:
            dc_lf._build_dc_ppc_from_e_file = original_loader

        self.assertEqual(["dc_net_30.e"], calls)
        self.assertTrue(calc.array_mode)
        self.assertEqual("dc_ppc_v1", calc.ppc["format"])

    def test_update_meas_snapshot_supports_dc_breaker_measurements(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        scripts_dir = root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import update_meas_from_lf

        snapshot, _info = update_meas_from_lf.solve_dc(root / "data" / "model" / "dc" / "dc_net_30.e")
        breaker = next(item for item in snapshot.dc.breakers if getattr(item, "is_alive", False))

        self.assertIsNotNone(snapshot.value("DCBreak", breaker.name, "P_FROM"))
        self.assertIsNotNone(snapshot.value("DCBreak", breaker.name, "V_FROM"))
        self.assertIsNotNone(snapshot.value("DCBreak", breaker.name, "I_FROM"))

    def test_run_reuses_combined_dc_newton_system_builder(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        class CountingDCPowerFlowCalc(DCPowerFlowCalc):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.combined_calls = 0
                self.public_f_calls = 0
                self.public_j_calls = 0

            def _build_newton_system(self, G, x):
                self.combined_calls += 1
                return super()._build_newton_system(G, x)

            def get_f(self, x, terms=None):
                self.public_f_calls += 1
                return super().get_f(x, terms=terms)

            def get_jacobi(self, G, x, terms=None):
                self.public_j_calls += 1
                return super().get_jacobi(G, x, terms=terms)

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = CountingDCPowerFlowCalc(network)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertGreater(calc.combined_calls, 0)
        self.assertEqual(0, calc.public_f_calls)
        self.assertEqual(0, calc.public_j_calls)

    def test_build_dc_ppc_from_e_file_creates_cached_array_model(self):
        import numpy as np
        from model.dc_array_model import (
            BRANCH_COLS,
            BUS_COLS,
            DCDC_COLS,
            GEN_COLS,
            LOAD_COLS,
            build_dc_ppc_from_e_file,
        )

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        ppc = build_dc_ppc_from_e_file(e_file)
        cached_ppc = build_dc_ppc_from_e_file(e_file)

        self.assertIs(ppc, cached_ppc)
        self.assertEqual("dc_ppc_v1", ppc["format"])
        self.assertEqual((30, len(BUS_COLS)), ppc["bus"].shape)
        self.assertEqual((37, len(BRANCH_COLS)), ppc["branch"].shape)
        self.assertEqual((16, len(LOAD_COLS)), ppc["load"].shape)
        self.assertEqual((14, len(GEN_COLS)), ppc["gen"].shape)
        self.assertEqual((9, len(DCDC_COLS)), ppc["dcdc"].shape)
        self.assertEqual(1.0, ppc["bus"][0, BUS_COLS["voltage"]])
        self.assertEqual(100.0, ppc["bus"][0, BUS_COLS["vbase"]])
        self.assertEqual(100.0, ppc["base"]["p_base"])
        np.testing.assert_array_equal(ppc["bus_name"][:3], np.asarray(["nd_1", "nd_2", "nd_3"], dtype=object))

    def test_build_dc_ppc_from_network_reflects_network_objects(self):
        import numpy as np
        from model.dc_array_model import BUS_COLS, DCPowerNetwork, build_dc_ppc_from_e_file, build_dc_ppc_from_network

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        expected = build_dc_ppc_from_e_file(e_file)
        network = DCPowerNetwork()
        network.read_from_file(e_file)

        network.nodes[0].voltage = 0.987654
        ppc = build_dc_ppc_from_network(network)

        for key in ("branch", "load", "gen", "zero_branch", "switch", "break", "dcdc"):
            np.testing.assert_allclose(ppc[key], expected[key])
        self.assertEqual(0.987654, ppc["bus"][0, BUS_COLS["voltage"]])
        np.testing.assert_allclose(ppc["bus"][1:], expected["bus"][1:])
        np.testing.assert_array_equal(ppc["bus_name"], expected["bus_name"])

    def test_build_dc_ppc_from_efile_rows_matches_file_builder(self):
        import numpy as np
        from efile_read import _read_efile_rows
        from model.dc_array_model import build_dc_ppc_from_e_file, build_dc_ppc_from_efile_rows

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        expected = build_dc_ppc_from_e_file(e_file)
        actual = build_dc_ppc_from_efile_rows(e_file, _read_efile_rows(e_file))

        for key in ("bus", "branch", "load", "gen", "zero_branch", "switch", "break", "dcdc"):
            self.assertEqual(expected[key].shape, actual[key].shape)
            np.testing.assert_allclose(expected[key], actual[key], atol=0.0)

    def test_build_dc_ppc_from_e_file_delegates_through_network_model(self):
        from model import dc_array_model

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        dc_array_model.clear_dc_ppc_cache(e_file)
        original = dc_array_model.build_dc_ppc_from_network
        calls = []

        def counted_builder(network):
            calls.append((network.__class__.__name__, len(network.nodes)))
            return original(network)

        dc_array_model.build_dc_ppc_from_network = counted_builder
        try:
            ppc = dc_array_model.build_dc_ppc_from_e_file(e_file)
        finally:
            dc_array_model.build_dc_ppc_from_network = original

        self.assertEqual([("DCPowerNetwork", 30)], calls)
        self.assertEqual("dc_ppc_v1", ppc["format"])

    def test_dcdc_residual_and_jacobian_use_vectorized_control_arrays(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        class NonIterableControls:
            def __iter__(self):
                raise AssertionError("DCDC control equations should use cached vectorized masks")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            G, x = calc.prepare()

        expected_f = calc.get_f(x)
        expected_j = calc.get_jacobi(G, x).toarray()
        calc.dcdc_ctrl = NonIterableControls()

        np.testing.assert_allclose(calc.get_f(x), expected_f, atol=1e-12)
        np.testing.assert_allclose(calc.get_jacobi(G, x).toarray(), expected_j, atol=1e-12)

    def test_update_lf_info_uses_cached_branch_arrays(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        class NonIterableBranches:
            def __iter__(self):
                raise AssertionError("DC load-flow backfill should use cached branch arrays")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = calc.prepare()
        calc.runtime_params = calc.params
        network.branches = NonIterableBranches()

        calc.update_lf_info(x)

        self.assertTrue(all(node.voltage > 0.0 for node in network.nodes if node.is_alive))

    def test_lf_result_uses_cached_voltage_lookup(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = calc.prepare()
        calc.runtime_params = calc.params
        calc.skip_lf_result = True
        calc.update_lf_info(x)

        def reject_node_voltage(*_args, **_kwargs):
            raise AssertionError("DC LF result building should use a cached voltage lookup")

        calc._node_voltage = reject_node_voltage
        result = calc._build_lf_result()

        self.assertEqual(len(network.nodes), len(result.nodes))
        self.assertTrue(result.branches)

    def test_generates_solvable_dc_case_and_measurements(self):
        from generate_dc_large_cases import generate_dc_case_files
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import DCPowerNetwork
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            e_file = Path(tmp_dir) / "dc_net_60.e"
            meas_file = Path(tmp_dir) / "dc_net_60.meas"
            generate_dc_case_files(60, e_file, meas_file)

            network = DCPowerNetwork()
            network.read_from_file(e_file)
            network.topo()
            with contextlib.redirect_stdout(io.StringIO()):
                warnings, errors = network.check_topo()
            self.assertEqual([], errors)
            self.assertEqual(60, len(network.nodes))
            self.assertEqual(6, sum(1 for gen in network.generators if gen.control_type == "V"))

            calc = DCPowerFlowCalc(network)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
            self.assertEqual(0, rc)
            self.assertTrue(calc.converged)

            quiet_network = DCPowerNetwork()
            quiet_network.read_from_file(e_file)
            quiet_network.topo()
            quiet_calc = DCPowerFlowCalc(quiet_network)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                quiet_network.check_topo()
                quiet_calc.run()
            self.assertEqual("", output.getvalue())

            estimator = DCStateEstimator(e_file=e_file, meas_file=meas_file, max_iter=20)
            result = estimator.estimate()
            self.assertTrue(result.converged)
            self.assertTrue(result.observability.observable)
            self.assertLess(result.residual_inf, 1e-7)


if __name__ == "__main__":
    unittest.main()
