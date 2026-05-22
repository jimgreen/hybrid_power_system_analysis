import contextlib
import io
import tempfile
import unittest
from pathlib import Path


class DCLargeCaseGenerationTest(unittest.TestCase):
    def test_dc_array_network_replaces_object_model_loader(self):
        from model.dc_array_model import build_dc_network_from_ppc, build_dc_ppc_from_e_file
        from dc_model import DCPowerNetwork

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network = build_dc_network_from_ppc(ppc)
        network.topo()

        self.assertIsInstance(network, DCPowerNetwork)
        self.assertEqual("dc_ppc_v1", network.ppc["format"])
        self.assertEqual(30, len(network.nodes))
        self.assertEqual(37, len(network.branches))
        self.assertEqual("nd_1", network.nodes[0].name)
        self.assertEqual(1.0, network.nodes[0].voltage)
        self.assertEqual(1, network.nodes[0].run_stat)
        self.assertIs(network.branches[0].i_node_obj, network.nodes[0])
        self.assertTrue(any(isl.is_alive for isl in network.islands))

    def test_dc_programs_do_not_import_dc_array_model_network_facade(self):
        root = Path(__file__).resolve().parents[1]
        checked_files = [
            root / "benchmarks" / "benchmark_flat_lf_se.py",
            root / "scripts" / "generate_dc_large_cases.py",
            root / "scripts" / "update_meas_from_lf.py",
            root / "src" / "hybrid_power_system_analysis" / "model" / "hybrid_model.py",
            root / "src" / "hybrid_power_system_analysis" / "secore" / "dc_se.py",
            root / "src" / "hybrid_power_system_analysis" / "lfcore" / "dc_lf.py",
        ]

        for file_path in checked_files:
            source = file_path.read_text(encoding="utf-8")
            self.assertNotIn("dc_array_model import DCPowerNetwork", source, str(file_path))

    def test_dc_solver_accepts_optional_solver_name_and_falls_back(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork

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
        from model.dc_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)

        with contextlib.redirect_stdout(io.StringIO()):
            prepared = calc.prepare()

        self.assertIsNone(prepared)
        self.assertFalse(hasattr(calc, "array_mode"))
        self.assertFalse(calc.keep_node_objects)
        self.assertEqual((26, 26), calc.G.shape)
        self.assertEqual(53, calc.x.size)
        np.testing.assert_array_equal(calc.branch_i[:3], np.asarray([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(calc.branch_j[:3], np.asarray([1, 2, 2], dtype=np.int32))
        self.assertEqual(9, calc.N_dcdc)

    def test_dc_solver_run_prepares_and_exposes_ac_style_state_access(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        calc = DCPowerFlowCalc(ppc, result_mode="array")

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()
            residual, jacobian = calc._build_newton_system(calc.x)
            public_jacobian = calc.get_jacobi(calc.x)

        self.assertEqual(0, rc)
        self.assertTrue(calc.converged)
        self.assertEqual(calc.total_eq, residual.size)
        self.assertEqual("csc", jacobian.format)
        np.testing.assert_allclose(jacobian.toarray(), public_jacobian.toarray(), atol=1e-12)

    def test_dc_solver_accepts_ppc_without_network_topology(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        calc = DCPowerFlowCalc(ppc)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = calc.run()

        self.assertEqual(0, rc)
        self.assertFalse(hasattr(calc, "array_mode"))
        self.assertTrue(calc.converged)
        self.assertEqual("dc_ppc_v1", calc.ppc["format"])
        self.assertIn("bus", calc.result)
        self.assertEqual(ppc["bus"].shape, calc.result["bus"].shape)

    def test_dc_ppc_prepare_ignores_static_cache_field(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")

        with contextlib.redirect_stdout(io.StringIO()):
            cold_calc = DCPowerFlowCalc(ppc)
            cold_calc.prepare()
            cold_G = cold_calc.G
            cold_x = cold_calc.x
        self.assertNotIn("_dc_pf_static", ppc)

        stale_static = {"N": -1}
        ppc["_dc_pf_static"] = stale_static
        with contextlib.redirect_stdout(io.StringIO()):
            warm_calc = DCPowerFlowCalc(ppc)
            warm_calc.prepare()
            warm_G = warm_calc.G
            warm_x = warm_calc.x

        self.assertIs(stale_static, ppc["_dc_pf_static"])
        self.assertEqual(cold_G.shape, warm_G.shape)
        self.assertEqual(cold_x.shape, warm_x.shape)
        self.assertFalse(hasattr(warm_calc, "array_mode"))
        self.assertEqual(cold_x.tolist(), warm_x.tolist())

        cold_x[0] = 123.0
        with contextlib.redirect_stdout(io.StringIO()):
            again_calc = DCPowerFlowCalc(ppc)
            again_calc.prepare()
            again_x = again_calc.x
        self.assertNotEqual(123.0, again_x[0])

    def test_dc_array_mode_uses_dense_lookup(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")

        calc = DCPowerFlowCalc(ppc, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        self.assertNotIn("_dc_pf_static", ppc)
        self.assertNotIsInstance(calc.alive_node_dict, dict)
        self.assertIn(1, calc.alive_node_dict)
        self.assertEqual(calc.alive_node_dict[1], calc.alive_node_dict.get(1))

    def test_dc_jacobian_reuses_precomputed_csr_pattern(self):
        import numpy as np
        import lfcore.dc_lf as dc_lf
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        calc = DCPowerFlowCalc(ppc)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
            x = calc.x

        expected = calc.get_jacobi(x).toarray()
        self.assertGreater(calc._dc_jac_csr_indices.size, 0)

        original_coo_matrix = dc_lf.coo_matrix

        def reject_coo_matrix(*_args, **_kwargs):
            raise AssertionError("DC Jacobian should refresh precomputed CSR data without COO rebuild")

        dc_lf.coo_matrix = reject_coo_matrix
        try:
            actual = calc.get_jacobi(x).toarray()
        finally:
            dc_lf.coo_matrix = original_coo_matrix

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_newton_system_returns_solver_ready_csc_jacobian(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        calc = DCPowerFlowCalc(ppc)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        _f, solver_jac = calc._build_newton_system(calc.x)
        public_jac = calc.get_jacobi(calc.x)

        self.assertEqual("csc", solver_jac.format)
        np.testing.assert_allclose(solver_jac.toarray(), public_jac.toarray(), atol=1e-12)

    def test_result_mode_skips_full_dc_result_backfill(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")

        calc = DCPowerFlowCalc(ppc, result_mode="none")

        stale_result = object()
        calc.lf_result = stale_result
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
        self.assertIsNone(calc.lf_result)
        self.assertTrue(hasattr(calc, "x"))

        summary_calc = DCPowerFlowCalc(ppc, result_mode="summary")
        summary_calc.lf_result = stale_result
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
        self.assertEqual({"node_id", "voltage", "summary"}, set(summary_calc.result))
        self.assertEqual(summary_calc.N, summary_calc.result["voltage"].size)
        self.assertEqual(summary_calc.N, summary_calc.result["node_id"].size)
        self.assertIsNone(summary_calc.lf_result)

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
        original_common_loader = dc_lf.build_dc_ppc_with_topology_from_e_file
        calls = []

        def counted_common_loader(file_name):
            calls.append(Path(file_name).name)
            return original_common_loader(file_name)

        self.assertFalse(hasattr(dc_lf, "_build_dc_ppc_from_e_file"))
        dc_lf.build_dc_ppc_with_topology_from_e_file = counted_common_loader
        try:
            ppc = dc_lf.load_dc_ppc_from_e_file(case_path)
            network = dc_lf._dc_network_from_ppc(ppc)
            calc = DCPowerFlowCalc(network)
        finally:
            dc_lf.build_dc_ppc_with_topology_from_e_file = original_common_loader

        self.assertEqual(["dc_net_30.e"], calls)
        self.assertFalse(hasattr(calc, "array_mode"))
        self.assertEqual("dc_ppc_v1", calc.ppc["format"])
        self.assertIn("_topology_arrays", calc.ppc)

    def test_dc_prepare_ensures_ppc_topology_before_direct_topology_step(self):
        import lfcore.dc_lf as dc_lf
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        case_path = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        ppc = build_dc_ppc_from_e_file(case_path)
        ppc.pop("_topology_arrays", None)
        calc = DCPowerFlowCalc(ppc, tol=1e-8, max_iter=50)
        original_prepare_topology = DCPowerFlowCalc._prepare_direct_ppc_topology
        original_ensure_topology = dc_lf.ensure_dc_ppc_topology
        calls = []
        ensure_calls = []

        def counted_ensure_topology(ppc_arg):
            ensure_calls.append(ppc_arg)
            return original_ensure_topology(ppc_arg)

        def wrapped_prepare_topology(self):
            calls.append(self.ppc.get("_topology_arrays") is not None)
            return original_prepare_topology(self)

        dc_lf.ensure_dc_ppc_topology = counted_ensure_topology
        DCPowerFlowCalc._prepare_direct_ppc_topology = wrapped_prepare_topology
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
        finally:
            dc_lf.ensure_dc_ppc_topology = original_ensure_topology
            DCPowerFlowCalc._prepare_direct_ppc_topology = original_prepare_topology

        self.assertEqual([ppc], ensure_calls)
        self.assertEqual([True], calls)
        self.assertIn("_topology_arrays", ppc)

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
        from model.dc_model import DCPowerNetwork

        class CountingDCPowerFlowCalc(DCPowerFlowCalc):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.combined_calls = 0
                self.public_f_calls = 0
                self.public_j_calls = 0

            def _build_newton_system(self, x, **kwargs):
                self.combined_calls += 1
                return super()._build_newton_system(x, **kwargs)

            def get_f(self, x):
                self.public_f_calls += 1
                return super().get_f(x)

            def get_jacobi(self, x):
                self.public_j_calls += 1
                return super().get_jacobi(x)

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
        for key in (
            "bus_cols",
            "branch_cols",
            "load_cols",
            "gen_cols",
            "zero_branch_cols",
            "switch_cols",
            "break_cols",
            "dcdc_cols",
            "ctrl",
        ):
            self.assertIn(key, ppc)
        self.assertIs(ppc["bus_cols"], BUS_COLS)
        np.testing.assert_array_equal(ppc["bus_name"][:3], np.asarray(["nd_1", "nd_2", "nd_3"], dtype=object))

    def test_build_dc_ppc_from_network_reflects_network_objects(self):
        import numpy as np
        from model.dc_array_model import BUS_COLS, build_dc_ppc_from_e_file, build_dc_ppc_from_network
        from model.dc_model import DCPowerNetwork

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

    def test_build_dc_ppc_from_e_file_builds_directly_from_loaded_rows(self):
        from model import dc_array_model

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        dc_array_model.clear_dc_ppc_cache(e_file)
        original_network_builder = dc_array_model.build_dc_ppc_from_network
        original_model_builder = dc_array_model._build_dc_ppc_from_model
        original_file_factory = dc_array_model.efile_factory_from_file
        original_rows_factory = dc_array_model.efile_factory_from_rows

        def reject_object_path(*_args, **_kwargs):
            raise AssertionError("DC E-file PPC load should not build dynamic model/network objects")

        dc_array_model.build_dc_ppc_from_network = reject_object_path
        dc_array_model._build_dc_ppc_from_model = reject_object_path
        dc_array_model.efile_factory_from_file = reject_object_path
        dc_array_model.efile_factory_from_rows = reject_object_path
        try:
            ppc = dc_array_model.build_dc_ppc_from_e_file(e_file)
        finally:
            dc_array_model.build_dc_ppc_from_network = original_network_builder
            dc_array_model._build_dc_ppc_from_model = original_model_builder
            dc_array_model.efile_factory_from_file = original_file_factory
            dc_array_model.efile_factory_from_rows = original_rows_factory

        self.assertEqual("dc_ppc_v1", ppc["format"])
        self.assertEqual(30, ppc["bus"].shape[0])

    def test_dc_rows_for_uses_cached_column_rows_like_ac_model(self):
        from efile_read import _read_efile_rows
        from model import dc_array_model

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        rows = _read_efile_rows(e_file)
        columns, table_rows = dc_array_model._rows_for(rows, "DCNode")

        self.assertTrue(hasattr(table_rows, "raw_column"))
        idx_col = columns["idx"]
        first = table_rows.raw_column(idx_col, "")
        second = table_rows.raw_column(idx_col, "")

        self.assertIs(first, second)

    def test_dc_ppc_builds_and_reuses_topology_input_like_ac_model(self):
        import numpy as np
        from model import topology
        from model.dc_array_model import build_dc_ppc_from_e_file, clear_dc_ppc_cache

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        clear_dc_ppc_cache(e_file)
        ppc = build_dc_ppc_from_e_file(e_file)

        self.assertIn("_topology_input", ppc)
        self.assertIn("branch", ppc["_topology_input"].terminals)
        self.assertIn("dcdc", ppc["_topology_input"].terminals)
        self.assertIn("gen", ppc["_topology_input"].singles)

        expected = topology.prepare_dc_topology_ppc({key: value for key, value in ppc.items() if key != "_topology_input"})
        original_mapper = topology._map_node_positions

        def reject_mapping(*_args, **_kwargs):
            raise AssertionError("DC topology should reuse precomputed PPC topology input")

        topology._map_node_positions = reject_mapping
        try:
            actual = topology.prepare_dc_topology_ppc(ppc)
        finally:
            topology._map_node_positions = original_mapper

        np.testing.assert_array_equal(actual.node_to_bus_pos, expected.node_to_bus_pos)
        np.testing.assert_array_equal(actual.node_to_island_pos, expected.node_to_island_pos)

    def test_dc_e_file_name_arrays_are_lazy_until_materialized(self):
        import numpy as np
        from model.dc_array_model import build_dc_ppc_from_e_file, clear_dc_ppc_cache

        e_file = Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e"
        clear_dc_ppc_cache(e_file)
        ppc = build_dc_ppc_from_e_file(e_file)
        bus_names = ppc["bus_name"]

        self.assertNotIsInstance(bus_names, np.ndarray)
        self.assertEqual(30, len(bus_names))
        self.assertEqual("nd_1", bus_names[0])
        np.testing.assert_array_equal(bus_names[:3], np.asarray(["nd_1", "nd_2", "nd_3"], dtype=object))
        np.testing.assert_array_equal(np.asarray(bus_names)[:3], np.asarray(["nd_1", "nd_2", "nd_3"], dtype=object))

    def test_dc_array_prepare_reuses_ppc_topology_device_positions(self):
        import contextlib
        import io
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        with contextlib.redirect_stdout(io.StringIO()):
            expected_calc = DCPowerFlowCalc(ppc, result_mode="array")
            expected_calc.prepare()

        original_mapper = DCPowerFlowCalc.__dict__["_map_nodes_with_lookup"]

        def reject_mapping(*_args, **_kwargs):
            raise AssertionError("direct DC array prepare should reuse PPC topology device positions")

        DCPowerFlowCalc._map_nodes_with_lookup = staticmethod(reject_mapping)
        try:
            calc = DCPowerFlowCalc(ppc, result_mode="array")
            with contextlib.redirect_stdout(io.StringIO()):
                calc.prepare()
        finally:
            DCPowerFlowCalc._map_nodes_with_lookup = original_mapper

        self.assertEqual(expected_calc.G.shape, calc.G.shape)
        self.assertEqual(expected_calc.x.size, calc.x.size)

    def test_dc_array_mode_eliminates_dcdc_j_power_variables_and_matches_full(self):
        import contextlib
        import io
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_array_model import build_dc_ppc_from_e_file

        ppc = build_dc_ppc_from_e_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")

        full_calc = DCPowerFlowCalc(ppc, result_mode="full")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, full_calc.run())

        array_calc = DCPowerFlowCalc(ppc, result_mode="array")
        with contextlib.redirect_stdout(io.StringIO()):
            array_calc.prepare()

        self.assertEqual(array_calc.N + array_calc.N_phi + array_calc.N_dcdc, array_calc.total_vars)
        self.assertEqual(array_calc.total_vars, array_calc.x.size)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, array_calc.run())

        np.testing.assert_allclose(
            array_calc.result["bus"][:, 2],
            full_calc.result["bus"][:, 2],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            array_calc.result["dcdc"][:, [10, 11]],
            full_calc.result["dcdc"][:, [10, 11]],
            atol=1e-8,
        )

    def test_dcdc_residual_and_jacobian_use_vectorized_control_arrays(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork

        class NonIterableControls:
            def __iter__(self):
                raise AssertionError("DCDC control equations should use cached vectorized masks")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
            x = calc.x

        expected_f = calc.get_f(x)
        expected_j = calc.get_jacobi(x).toarray()
        calc.dcdc_ctrl = NonIterableControls()

        np.testing.assert_allclose(calc.get_f(x), expected_f, atol=1e-12)
        np.testing.assert_allclose(calc.get_jacobi(x).toarray(), expected_j, atol=1e-12)

    def test_write_back_uses_cached_branch_arrays(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork

        class NonIterableBranches:
            def __iter__(self):
                raise AssertionError("DC load-flow backfill should use cached branch arrays")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
            x = calc.x
        network.branches = NonIterableBranches()

        calc.x = x
        calc._write_back()

        self.assertTrue(all(node.voltage > 0.0 for node in network.nodes if node.is_alive))

    def test_lf_result_uses_cached_voltage_lookup(self):
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "model" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()
            x = calc.x
        calc.skip_lf_result = True
        calc.x = x
        calc._write_back()

        def reject_node_voltage(*_args, **_kwargs):
            raise AssertionError("DC LF result building should use a cached voltage lookup")

        calc._node_voltage = reject_node_voltage
        result = calc._build_lf_result()

        self.assertEqual(len(network.nodes), len(result.nodes))
        self.assertTrue(result.branches)

    def test_generates_solvable_dc_case_and_measurements(self):
        from generate_dc_large_cases import generate_dc_case_files
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork
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
