import contextlib
import inspect
import io
import math
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "model"))


class NetworkFactoryLoadingTest(unittest.TestCase):
    def test_model_loading_api_signatures_are_simplified(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf
        from lfcore.ac_lf import load_ac_ppc_from_e_file
        from lfcore.dc_lf import load_dc_ppc_from_e_file
        from model.ac_array_model import build_ac_ppc_from_model, build_ac_ppc_from_network
        from model.dc_array_model import DCPowerNetwork, build_dc_ppc_from_model, build_dc_ppc_from_network
        from model.hybrid_array_model import build_hybrid_ppc_from_e_file
        from model.hybrid_model import HybridPowerNetwork

        self.assertEqual(["network"], list(inspect.signature(build_ac_ppc_from_network).parameters))
        self.assertEqual(["network"], list(inspect.signature(build_dc_ppc_from_network).parameters))
        self.assertEqual(["model"], list(inspect.signature(build_ac_ppc_from_model).parameters))
        self.assertEqual(["model"], list(inspect.signature(build_dc_ppc_from_model).parameters))
        self.assertEqual(["file_path"], list(inspect.signature(build_hybrid_ppc_from_e_file).parameters))
        self.assertEqual(["file_name"], list(inspect.signature(load_ac_ppc_from_e_file).parameters))
        self.assertEqual(["file_name"], list(inspect.signature(load_dc_ppc_from_e_file).parameters))
        self.assertEqual(["self", "file_name"], list(inspect.signature(DCPowerNetwork.read_from_file).parameters))
        self.assertEqual(["file_name"], list(inspect.signature(HybridPowerNetwork.read_from_file).parameters))
        self.assertFalse(hasattr(ac_lf, "load_ac_network_from_e_file"))
        self.assertFalse(hasattr(dc_lf, "load_dc_network_from_e_file"))

    def test_empty_ppc_helpers_are_removed_from_array_modules(self):
        import model.ac_array_model as ac_array_model
        import model.dc_array_model as dc_array_model

        self.assertFalse(hasattr(ac_array_model, "build_empty_ac_ppc"))
        self.assertFalse(hasattr(dc_array_model, "build_empty_dc_ppc"))

    def test_cache_ppc_helpers_are_removed_from_array_modules(self):
        for rel_path in (
            "src/hybrid_power_system_analysis/model/ac_array_model.py",
            "src/hybrid_power_system_analysis/model/dc_array_model.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("def build_ac_ppc_from_cache", source, rel_path)
            self.assertNotIn("def build_dc_ppc_from_cache", source, rel_path)

    def test_model_array_imports_do_not_use_importerror_fallbacks(self):
        for rel_path in (
            "src/hybrid_power_system_analysis/model/ac_array_model.py",
            "src/hybrid_power_system_analysis/model/dc_array_model.py",
            "src/hybrid_power_system_analysis/model/hybrid_array_model.py",
            "src/hybrid_power_system_analysis/model/hybrid_model.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("except ImportError", source, rel_path)

    def test_required_scipy_imports_do_not_have_missing_dependency_fallbacks(self):
        forbidden_tokens = (
            "except ModuleNotFoundError",
            "SCIPY_AVAILABLE = False",
            "Small numpy-backed scipy sparse subset",
            "coo_matrix = csr_matrix = None",
            "SP_COO_MATRIX = None",
            "SP_STRUCTURAL_RANK = None",
            "SP_SPLU = None",
            "DPOSV = None",
            "CHO_FACTOR = None",
            "sp_maximum_bipartite_matching = None",
        )
        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
            "src/hybrid_power_system_analysis/secore/ac_se.py",
            "src/hybrid_power_system_analysis/secore/se_math.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, rel_path)

    def test_ac_power_network_reads_from_in_memory_model_without_reopening_file(self):
        import ac_model
        from efile_read import efile_factory_from_file

        case_path = ROOT_DIR / "data" / "ac" / "ieee39.e"
        model = efile_factory_from_file(case_path)

        expected = ac_model.ACPowerNetwork()
        expected.read_from_file(case_path)
        expected_node = next(node for node in expected.nodes if node.name == "bus_1")

        original_factory = ac_model.efile_factory_from_file

        def reject_file_factory(*_args, **_kwargs):
            raise AssertionError("read_from_model should consume the in-memory model without reopening the file")

        ac_model.efile_factory_from_file = reject_file_factory
        try:
            network = ac_model.ACPowerNetwork()
            old_method_name = "read_from_" + "cache"
            self.assertFalse(hasattr(network, old_method_name))
            network.read_from_model(model)
        finally:
            ac_model.efile_factory_from_file = original_factory

        node = next(item for item in network.nodes if item.name == "bus_1")
        self.assertEqual(len(expected.nodes), len(network.nodes))
        self.assertEqual(len(expected.branches), len(network.branches))
        self.assertEqual(len(expected.transformers), len(network.transformers))
        self.assertAlmostEqual(expected_node.voltage, node.voltage)
        self.assertAlmostEqual(expected_node.angle, node.angle, places=10)
        self.assertLess(abs(node.angle), math.pi)

        with contextlib.redirect_stdout(io.StringIO()):
            network.topo()
        self.assertTrue(any(isl.is_alive for isl in network.islands))

    def test_ac_read_from_file_uses_factory_then_delegates_to_read_from_model(self):
        import ac_model

        case_path = ROOT_DIR / "data" / "ac" / "ieee39.e"
        original_factory = ac_model.efile_factory_from_file
        original_read_from_model = ac_model.ACPowerNetwork.read_from_model
        calls = []

        def counted_factory(file_name):
            model = original_factory(file_name)
            calls.append(("factory", Path(file_name).name, bool(getattr(model, "ACNode", []))))
            return model

        def counted_read_from_model(self, model):
            calls.append(("read_from_model", bool(getattr(model, "ACNode", [])), bool(getattr(model, "PowerBase", []))))
            return original_read_from_model(self, model)

        ac_model.efile_factory_from_file = counted_factory
        ac_model.ACPowerNetwork.read_from_model = counted_read_from_model
        try:
            network = ac_model.ACPowerNetwork()
            network.read_from_file(case_path)
        finally:
            ac_model.efile_factory_from_file = original_factory
            ac_model.ACPowerNetwork.read_from_model = original_read_from_model

        self.assertEqual(
            [("factory", "ieee39.e", True), ("read_from_model", True, True)],
            calls,
        )
        self.assertEqual(39, len(network.nodes))

    def test_dc_power_network_reads_from_in_memory_model_without_reopening_file(self):
        import model.dc_model as dc_model
        from efile_read import efile_factory_from_file

        case_path = ROOT_DIR / "data" / "dc" / "dc_net_30.e"
        model = efile_factory_from_file(case_path)

        expected = dc_model.DCPowerNetwork()
        expected.read_from_file(case_path)

        original_factory = dc_model.efile_factory_from_file

        def reject_file_factory(*_args, **_kwargs):
            raise AssertionError("read_from_model should consume the in-memory model without reopening the file")

        dc_model.efile_factory_from_file = reject_file_factory
        try:
            network = dc_model.DCPowerNetwork()
            old_method_name = "read_from_" + "cache"
            self.assertFalse(hasattr(network, old_method_name))
            network.read_from_model(model)
        finally:
            dc_model.efile_factory_from_file = original_factory

        self.assertEqual(len(expected.nodes), len(network.nodes))
        self.assertEqual(len(expected.branches), len(network.branches))
        self.assertEqual(len(expected.generators), len(network.generators))
        self.assertEqual(len(expected.loads), len(network.loads))
        self.assertEqual(len(expected.dcdc_converters), len(network.dcdc_converters))
        self.assertEqual(len(expected.breakers), len(network.breakers))
        self.assertEqual(expected.nodes[0].name, network.nodes[0].name)
        self.assertAlmostEqual(expected.nodes[0].voltage, network.nodes[0].voltage)

        with contextlib.redirect_stdout(io.StringIO()):
            network.topo()
        self.assertTrue(any(isl.is_alive for isl in network.islands))

    def test_dc_read_from_file_uses_factory_then_delegates_to_read_from_model(self):
        import model.dc_model as dc_model

        case_path = ROOT_DIR / "data" / "dc" / "dc_net_30.e"
        original_factory = dc_model.efile_factory_from_file
        original_read_from_model = dc_model.DCPowerNetwork.read_from_model
        calls = []

        def counted_factory(file_name):
            model = original_factory(file_name)
            calls.append(("factory", Path(file_name).name, bool(getattr(model, "DCNode", []))))
            return model

        def counted_read_from_model(self, model):
            calls.append(("read_from_model", bool(getattr(model, "DCNode", [])), bool(getattr(model, "PowerBase", []))))
            return original_read_from_model(self, model)

        dc_model.efile_factory_from_file = counted_factory
        dc_model.DCPowerNetwork.read_from_model = counted_read_from_model
        try:
            network = dc_model.DCPowerNetwork()
            network.read_from_file(case_path)
        finally:
            dc_model.efile_factory_from_file = original_factory
            dc_model.DCPowerNetwork.read_from_model = original_read_from_model

        self.assertEqual(
            [("factory", "dc_net_30.e", True), ("read_from_model", True, True)],
            calls,
        )
        self.assertEqual(30, len(network.nodes))


if __name__ == "__main__":
    unittest.main()
