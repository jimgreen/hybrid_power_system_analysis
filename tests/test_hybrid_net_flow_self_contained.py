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
        self.assertTrue(hasattr(hybrid_net_flow, "HybridACGrid"))
        self.assertTrue(hasattr(hybrid_net_flow, "HybridDCGrid"))

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

    def test_converter_initial_values_and_writeback_use_cached_arrays(self):
        from lfcore.hybrid_flow import HybridPowerFlowCalc, HybridPowerNetwork

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

        self.assertIsInstance(network.ac, hybrid_net_flow.HybridACGrid)
        self.assertIsInstance(network.dc, hybrid_net_flow.HybridDCGrid)
        self.assertEqual(len(network.ac.islands), 1)
        self.assertEqual(len(network.dc.islands), 3)
        self.assertEqual(len(network.hybrid_islands), 1)

        island = network.hybrid_islands[0]
        self.assertIsInstance(island, hybrid_net_flow.HybridIsland)
        self.assertTrue(island.is_alive)
        self.assertEqual(len(island.ac_islands), 1)
        self.assertEqual(len(island.dc_islands), 3)
        self.assertEqual(len(island.ac_nodes), 10)
        self.assertEqual(len(island.dc_nodes), 30)
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
            ("DCSwitch", 0),
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
