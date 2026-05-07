import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))


def _ensure_block_column(text, block_name, column_name, default_value):
    start = text.index(f"<{block_name}>")
    end = text.index(f"</{block_name}>", start)
    block = text[start:end]
    lines = block.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("@"))
    header = lines[header_idx].split()
    if column_name in header[1:]:
        return text
    lines[header_idx] = lines[header_idx] + f" {column_name}"
    for i, line in enumerate(lines):
        if line.startswith("#"):
            lines[i] = line + f" {default_value}"
    return text[:start] + "\n".join(lines) + "\n" + text[end:]


def _set_block_value(text, block_name, row_idx, column_name, value):
    start = text.index(f"<{block_name}>")
    end = text.index(f"</{block_name}>", start)
    block = text[start:end]
    lines = block.splitlines()
    header = next(line for line in lines if line.startswith("@")).split()[1:]
    column_idx = header.index(column_name)
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] == str(row_idx):
            parts[column_idx + 1] = str(value)
            lines[i] = " ".join(parts)
            break
    else:
        raise AssertionError(f"{block_name}[{row_idx}] not found")
    return text[:start] + "\n".join(lines) + "\n" + text[end:]


class HybridNetFlowSelfContainedTest(unittest.TestCase):
    def test_hybrid_net_flow_does_not_import_network_classes(self):
        import hybrid_net_flow

        source = Path(hybrid_net_flow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from hybrid_net_model import ACPowerNetwork", source)
        self.assertNotIn("DCPowerNetwork", source)
        self.assertFalse(hasattr(hybrid_net_flow, "ACPowerNetwork"))
        self.assertFalse(hasattr(hybrid_net_flow, "DCPowerNetwork"))
        self.assertFalse(hasattr(hybrid_net_flow, "HybridACGrid"))
        self.assertFalse(hasattr(hybrid_net_flow, "HybridDCGrid"))
        self.assertTrue(hasattr(hybrid_net_flow, "HybridPowerNetwork"))

    def test_hybrid_net_40_runs_from_self_contained_network(self):
        from scipy.sparse import issparse
        import hybrid_net_flow

        result = hybrid_net_flow.run_hybrid_power_flow(
            ROOT / "data" / "hybrid" / "hybrid_net_40.e",
            verbose=False,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.total_nodes, 40)
        self.assertEqual(len(result.ac_network.nodes), 10)
        self.assertEqual(len(result.dc_network.nodes), 30)
        self.assertTrue(result.has_acac)
        self.assertEqual(result.calc.N_acac, 1)
        self.assertGreater(result.global_jacobian_shape[0], 0)
        self.assertEqual(result.global_jacobian_shape[0], result.global_jacobian_shape[1])

        ac_x, dc_x, _, _ = result.calc._split_x(result.calc.x)
        with contextlib.redirect_stdout(io.StringIO()):
            ac_j = result.calc.ac_calc.get_jacobi(ac_x)
            dc_j = result.calc.dc_calc.get_jacobi(result.calc.dc_G, dc_x)
            hybrid_j = result.calc.get_jacobi(result.calc.x)
        self.assertTrue(issparse(ac_j))
        self.assertTrue(issparse(dc_j))
        self.assertTrue(issparse(hybrid_j))

    def test_hybrid_jacobian_builds_converter_terms_in_one_sparse_pass(self):
        from scipy.sparse import issparse
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "hybrid" / "hybrid_net_40.e"
        )
        network.prepare(verbose=False)
        calc = hybrid_net_flow.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        def fail_old_converter_sparse_path(*_args, **_kwargs):
            raise AssertionError("converter coupling terms should be appended to the global COO data")

        calc._get_dcac_jacobi = fail_old_converter_sparse_path
        calc._get_acac_jacobi = fail_old_converter_sparse_path

        jac = calc.get_jacobi(calc.x)

        self.assertTrue(issparse(jac))
        self.assertEqual(jac.shape, (calc.total_eq, calc.total_vars))

    def test_hybrid_newton_uses_shared_sparse_solver(self):
        import lfcore.hybrid_lf as hybrid_lf

        network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "hybrid" / "hybrid_net_40.e")
        calc = hybrid_lf.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        original_solver = hybrid_lf.solve_sparse_system
        calls = []

        def counted_solver(matrix, rhs, solver_name="scipy"):
            calls.append((matrix.shape, rhs.shape, solver_name))
            return original_solver(matrix, rhs, solver_name)

        hybrid_lf.solve_sparse_system = counted_solver
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = calc.run()
        finally:
            hybrid_lf.solve_sparse_system = original_solver

        self.assertEqual(0, rc)
        self.assertTrue(calls)
        self.assertTrue(all(shape == (calc.total_eq, calc.total_vars) for shape, _rhs_shape, _solver in calls))

    def test_converter_terms_reuse_ac_state_cache(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "hybrid" / "hybrid_net_40.e"
        )
        network.prepare(verbose=False)
        calc = hybrid_net_flow.HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        original_extract = calc.ac_calc._extract_state_vars
        call_count = 0

        def counted_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_extract(*args, **kwargs)

        calc.ac_calc._extract_state_vars = counted_extract
        calc.get_f(calc.x)
        self.assertEqual(1, call_count)

        call_count = 0
        calc.get_jacobi(calc.x)
        self.assertEqual(1, call_count)

        call_count = 0
        calc._write_back(calc.x)
        self.assertEqual(1, call_count)

    def test_hybrid_network_load_uses_array_model_for_all_case_shapes(self):
        import lfcore.hybrid_lf as hybrid_lf

        ac_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "ac" / "ieee300.e")
        dc_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "dc" / "dc_net_30.e")
        hybrid_network = hybrid_lf.HybridPowerNetwork.read_from_file(ROOT / "data" / "hybrid" / "hybrid_net_40.e")

        self.assertEqual("hybrid_ppc_v1", ac_network.ppc["format"])
        self.assertEqual("hybrid_ppc_v1", dc_network.ppc["format"])
        self.assertEqual("hybrid_ppc_v1", hybrid_network.ppc["format"])
        self.assertTrue(hasattr(ac_network, "_ac_ppc"))
        self.assertTrue(hasattr(dc_network, "_dc_ppc"))
        self.assertTrue(hasattr(hybrid_network, "_ac_ppc"))
        self.assertTrue(hasattr(hybrid_network, "_dc_ppc"))
        self.assertTrue(ac_network.ac.nodes)
        self.assertTrue(dc_network.dc.nodes)
        self.assertTrue(hybrid_network.dcac_converters or hybrid_network.acac_converters)

    def test_hybrid_ppc_builds_sub_ppc_from_shared_model(self):
        import model.hybrid_array_model as hybrid_array_model

        case_path = ROOT / "data" / "hybrid" / "hybrid_net_40.e"
        original_factory = hybrid_array_model.efile_factory_from_file
        original_ac_builder = hybrid_array_model.build_ac_ppc_from_model
        original_dc_builder = hybrid_array_model.build_dc_ppc_from_model
        original_ac_file_builder = getattr(hybrid_array_model, "build_ac_ppc_from_e_file", None)
        original_dc_file_builder = getattr(hybrid_array_model, "build_dc_ppc_from_e_file", None)

        calls = {"factory": 0, "ac_model": 0, "dc_model": 0}

        def counted_factory(*args, **kwargs):
            calls["factory"] += 1
            return original_factory(*args, **kwargs)

        def counted_ac_builder(model, **kwargs):
            calls["ac_model"] += 1
            self.assertTrue(getattr(model, "ACNode", []))
            return original_ac_builder(model, **kwargs)

        def counted_dc_builder(model, **kwargs):
            calls["dc_model"] += 1
            self.assertTrue(getattr(model, "DCNode", []))
            return original_dc_builder(model, **kwargs)

        def reject_file_builder(*_args, **_kwargs):
            raise AssertionError("hybrid ppc builder should use the shared model, not file-level sub builders")

        hybrid_array_model.efile_factory_from_file = counted_factory
        hybrid_array_model.build_ac_ppc_from_model = counted_ac_builder
        hybrid_array_model.build_dc_ppc_from_model = counted_dc_builder
        if original_ac_file_builder is not None:
            hybrid_array_model.build_ac_ppc_from_e_file = reject_file_builder
        if original_dc_file_builder is not None:
            hybrid_array_model.build_dc_ppc_from_e_file = reject_file_builder
        try:
            _network, ppc = hybrid_array_model.build_hybrid_ppc_from_e_file(case_path)
        finally:
            hybrid_array_model.efile_factory_from_file = original_factory
            hybrid_array_model.build_ac_ppc_from_model = original_ac_builder
            hybrid_array_model.build_dc_ppc_from_model = original_dc_builder
            if original_ac_file_builder is None:
                if hasattr(hybrid_array_model, "build_ac_ppc_from_e_file"):
                    del hybrid_array_model.build_ac_ppc_from_e_file
            else:
                hybrid_array_model.build_ac_ppc_from_e_file = original_ac_file_builder
            if original_dc_file_builder is None:
                if hasattr(hybrid_array_model, "build_dc_ppc_from_e_file"):
                    del hybrid_array_model.build_dc_ppc_from_e_file
            else:
                hybrid_array_model.build_dc_ppc_from_e_file = original_dc_file_builder

        self.assertEqual(1, calls["factory"])
        self.assertEqual(1, calls["ac_model"])
        self.assertEqual(1, calls["dc_model"])
        self.assertIs(ppc["ac_network"].ppc, ppc["ac"])
        self.assertIs(ppc["dc_network"].ppc, ppc["dc"])
        self.assertEqual(10, len(ppc["ac_network"].nodes))
        self.assertEqual(30, len(ppc["dc_network"].nodes))
        self.assertEqual(10, ppc["ac"]["bus"].shape[0])
        self.assertEqual(30, ppc["dc"]["bus"].shape[0])
        self.assertGreater(ppc["dcac"].shape[0], 0)
        self.assertGreater(ppc["acac"].shape[0], 0)

    def test_lf_loader_builds_real_ac_network_from_ppc(self):
        import lfcore.hybrid_lf as hybrid_lf
        from ac_model import ACPowerNetwork

        network = hybrid_lf._read_lf_network_from_file(ROOT / "data" / "hybrid" / "hybrid_net_40.e")

        self.assertFalse(hasattr(hybrid_lf, "_build_lf_ac_facade"))
        self.assertTrue(hasattr(hybrid_lf, "_build_lf_ac_network"))
        self.assertIsInstance(network.ac, ACPowerNetwork)
        self.assertFalse(getattr(network.ac, "_lf_lightweight", False))
        self.assertEqual(10, len(network.ac.nodes))
        self.assertTrue(network.ac.breakers)

    def test_hybrid_ppc_model_build_delegates_to_ac_dc_array_helpers(self):
        import model.hybrid_array_model as hybrid_array_model

        _network, ppc = hybrid_array_model.build_hybrid_ppc_from_e_file(ROOT / "data" / "hybrid" / "hybrid_net_40.e")

        self.assertFalse(hasattr(hybrid_array_model, "_build_ac_network"))
        self.assertFalse(hasattr(hybrid_array_model, "_build_dc_network"))
        self.assertFalse(hasattr(hybrid_array_model, "_empty_ac_ppc"))
        self.assertFalse(hasattr(hybrid_array_model, "_empty_dc_ppc"))

        self.assertFalse(hasattr(hybrid_array_model, "build_ac_network_from_ppc"))
        self.assertFalse(hasattr(hybrid_array_model, "build_dc_network_from_ppc"))
        model = hybrid_array_model.build_hybrid_model_from_ppc(ppc)

        from model.hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork

        self.assertIsInstance(model, HybridPowerNetwork)
        self.assertIs(model.ac, ppc["ac_network"])
        self.assertIs(model.dc, ppc["dc_network"])
        self.assertTrue(all(isinstance(conv, DCACConverter) for conv in model.DCACConverter))
        self.assertTrue(all(isinstance(conv, ACACConverter) for conv in model.ACACConverter))
        self.assertEqual(10, len(model.ac.nodes))
        self.assertEqual(30, len(model.dc.nodes))

    def test_run_hybrid_power_flow_returns_hybrid_lf_result(self):
        import lfcore.hybrid_lf as hybrid_lf

        result = hybrid_lf.run_hybrid_power_flow(
            ROOT / "data" / "hybrid" / "hybrid_net_40.e",
            verbose=False,
        )

        self.assertFalse(hasattr(hybrid_lf, "HybridPowerFlowResult"))
        self.assertIsInstance(result, hybrid_lf.HybridLFResult)
        self.assertIs(result.lf_result, result)
        self.assertTrue(result.converged)
        self.assertIs(result.network, result.calc.network)
        self.assertIsNotNone(result.ac)
        self.assertIsNotNone(result.dc)
        self.assertTrue(result.dcac.dcac_converters)
        self.assertTrue(result.acac.acac_converters)

    def test_converter_initial_values_and_writeback_use_cached_arrays(self):
        from lfcore.hybrid_lf import HybridPowerFlowCalc, HybridPowerNetwork

        class NonIterableConverters:
            def __iter__(self):
                raise AssertionError("hybrid converter helpers should use cached converter arrays")

        network = HybridPowerNetwork.read_from_file(ROOT / "data" / "hybrid" / "hybrid_net_40.e")
        network.prepare(verbose=False)
        calc = HybridPowerFlowCalc(network, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            calc.prepare()

        expected_dcac = calc._initial_dcac_x()
        expected_acac = calc._initial_acac_x()
        calc.dcac_converters = NonIterableConverters()
        calc.acac_converters = NonIterableConverters()

        self.assertGreater(calc.N_dcac, 0)
        self.assertGreater(calc.N_acac, 0)
        self.assertEqual(expected_dcac.tolist(), calc._initial_dcac_x().tolist())
        self.assertEqual(expected_acac.tolist(), calc._initial_acac_x().tolist())
        calc._write_back(calc.x)

    def test_hybrid_topology_builds_hybrid_islands(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "hybrid" / "hybrid_net_40.e"
        )

        network.topo()

        from ac_model import ACPowerNetwork
        from dc_model import DCPowerNetwork

        self.assertIsInstance(network.ac, ACPowerNetwork)
        self.assertIsInstance(network.dc, DCPowerNetwork)
        self.assertEqual(len(network.ac.islands), 1)
        self.assertEqual(len(network.dc.islands), 3)
        self.assertEqual(len(network.hybrid_islands), 1)

        island = network.hybrid_islands[0]
        self.assertIsInstance(island, hybrid_net_flow.HybridIsland)
        self.assertTrue(island.is_alive)
        self.assertEqual(len(island.ac_islands), 1)
        self.assertEqual(len(island.dc_islands), 3)
        self.assertEqual(len(island.ac_nodes), 10)
        self.assertEqual(len(island.dc_nodes), 26)
        self.assertEqual(len(island.dcac_converters), 3)
        self.assertEqual(len(island.dcdc_converters), 9)
        self.assertEqual(len(island.acac_converters), 1)
        self.assertIs(network.ac.islands[0].hybrid_isl_obj, island)
        self.assertTrue(all(dc_isl.hybrid_isl_obj is island for dc_isl in network.dc.islands))

    def test_acac_converter_is_solved_inside_hybrid_newton_system(self):
        import hybrid_net_flow

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "hybrid_acac.e"
            source_path = ROOT / "data" / "hybrid" / "hybrid_net_40.e"
            source_text = source_path.read_text(encoding="utf-8").split("<ACACConverter>")[0].rstrip()
            acac_block = """

<ACACConverter>
@ idx name     i_node j_node r1   r2   control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat i_p  i_q  j_p  j_q  i_i  j_i
# 0   acac_3_4 3      4      0.01 0.01 PQQ          5.0   0.0     0.0     0.0     0.0     1        0.0  0.0 0.0 0.0 0.0 0.0
</ACACConverter>
"""
            case_path.write_text(source_text + acac_block, encoding="utf-8")

            result = hybrid_net_flow.run_hybrid_power_flow(case_path, verbose=False)

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))
        self.assertTrue(result.has_acac)
        self.assertEqual(len(result.network.acac_converters), 1)
        self.assertEqual(result.calc.N_acac, 1)
        self.assertEqual(
            result.global_jacobian_shape[0],
            result.calc.ac_eq + result.calc.dc_eq + result.calc.N_dcac * 3 + result.calc.N_acac * 4,
        )
        self.assertEqual(result.global_jacobian_shape[0], result.global_jacobian_shape[1])

        conv = result.network.acac_converters[0]
        self.assertAlmostEqual(conv.i_p, 0.05, places=8)
        self.assertAlmostEqual(conv.i_q, 0.0, places=8)
        self.assertAlmostEqual(conv.j_q, 0.0, places=8)
        self.assertIsNotNone(conv.j_p)
        self.assertGreater(conv.i_i, 0.0)
        self.assertGreater(conv.j_i, 0.0)

    def test_topology_counts_cross_ac_dc_converters_as_node_references(self):
        import hybrid_net_flow

        network = hybrid_net_flow.HybridPowerNetwork.read_from_file(
            ROOT / "data" / "hybrid" / "qinling.e"
        )

        ac_warnings, ac_errors, dc_warnings, dc_errors = network.prepare(verbose=False)

        self.assertEqual(ac_errors, [])
        self.assertEqual(dc_errors, [])
        self.assertEqual(ac_warnings, [])
        self.assertEqual(dc_warnings, [])

    def test_main_hybrid_power_flow_skips_topology_diagnostics(self):
        import hybrid_net_flow
        from ac_model import ACPowerNetwork
        from dc_array_model import DCPowerNetwork

        original_ac_check_topo = ACPowerNetwork.check_topo
        original_dc_check_topo = DCPowerNetwork.check_topo

        def reject_check_topo(*_args, **_kwargs):
            raise AssertionError("main hybrid load-flow path should not call check_topo")

        ACPowerNetwork.check_topo = reject_check_topo
        DCPowerNetwork.check_topo = reject_check_topo
        try:
            result = hybrid_net_flow.run_hybrid_power_flow(
                ROOT / "data" / "hybrid" / "qinling.e",
                verbose=False,
            )
        finally:
            ACPowerNetwork.check_topo = original_ac_check_topo
            DCPowerNetwork.check_topo = original_dc_check_topo

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))

    def test_node_run_stat_zero_removes_attached_converter_from_solution(self):
        import hybrid_net_flow

        text = (ROOT / "data" / "hybrid" / "qinling.e").read_text(encoding="utf-8")
        text = _ensure_block_column(text, "ACNode", "run_stat", "1")
        text = _ensure_block_column(text, "DCNode", "run_stat", "1")
        for block_name, row_idx in (
            ("ACNode", 0),
            ("ACNode", 10),
            ("DCNode", 1),
            ("DCNode", 28),
            ("ACBranch", 0),
            ("DCBranch", 0),
            ("ACGenerator", 0),
            ("DCBreak", 0),
        ):
            text = _set_block_value(text, block_name, row_idx, "run_stat", "0")

        with tempfile.TemporaryDirectory() as tmpdir:
            case_path = Path(tmpdir) / "qinling_node_out.e"
            case_path.write_text(text, encoding="utf-8")
            result = hybrid_net_flow.run_hybrid_power_flow(case_path, verbose=False)

        self.assertTrue(result.converged, (result.ac_errors, result.dc_errors, result.calc.normF))
        self.assertEqual(len(result.network.dcac_converters), 11)
        self.assertEqual(result.calc.N_dcac, 10)
        self.assertNotIn(result.network.dcac_converters[0], [item[0] for item in result.calc.dcac_converters])


if __name__ == "__main__":
    unittest.main()
