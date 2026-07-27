import contextlib
import inspect
import io
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


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
        import model.dc_array_model as dc_array_model
        from model.dc_array_model import build_dc_ppc_from_model, build_dc_ppc_from_network
        from model.dc_model import DCPowerNetwork
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
        self.assertFalse(hasattr(dc_array_model, "DCPowerNetwork"))

    def test_dc_lf_public_api_matches_ac_lf_flow(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf
        from lfcore.ac_lf import ACPowerFlowCalc
        from lfcore.dc_lf import DCPowerFlowCalc

        ac_init = [
            name
            for name in inspect.signature(ACPowerFlowCalc.__init__).parameters
            if name != "self"
        ]
        dc_init = [
            name
            for name in inspect.signature(DCPowerFlowCalc.__init__).parameters
            if name != "self"
        ]
        self.assertEqual(ac_init, dc_init)
        self.assertNotIn("algorithm", ac_init)
        self.assertNotIn("jacobian_refresh_period", ac_init)
        for method_name in ("prepare", "get_f", "get_jacobi", "_build_newton_system", "_run_newton_raphson", "run"):
            with self.subTest(method=method_name):
                self.assertEqual(
                    inspect.signature(getattr(ACPowerFlowCalc, method_name)),
                    inspect.signature(getattr(DCPowerFlowCalc, method_name)),
                )
        self.assertIn("return self._run_newton_raphson()", inspect.getsource(ACPowerFlowCalc.run))
        self.assertIn("return self._run_newton_raphson()", inspect.getsource(DCPowerFlowCalc.run))
        self.assertTrue(hasattr(DCPowerFlowCalc, "_prepare_from_ppc"))
        self.assertIn("self._prepare_from_ppc()", inspect.getsource(DCPowerFlowCalc.prepare))
        self.assertEqual(
            inspect.signature(ACPowerFlowCalc._write_summary_result),
            inspect.signature(DCPowerFlowCalc._write_summary_result),
        )
        self.assertEqual(
            inspect.signature(ACPowerFlowCalc._write_back),
            inspect.signature(DCPowerFlowCalc._write_back),
        )
        self.assertEqual(
            inspect.signature(ACPowerFlowCalc._write_back_ppc),
            inspect.signature(DCPowerFlowCalc._write_back_ppc),
        )
        self.assertEqual(
            list(inspect.signature(ac_lf.print_ac_result).parameters),
            list(inspect.signature(dc_lf.print_dc_result).parameters),
        )

        dc_source = inspect.getsource(DCPowerFlowCalc._run_newton_raphson)
        self.assertIn("delta = factor.solve(F)", dc_source)
        self.assertIn("x -= delta", dc_source)
        self.assertNotIn("factor.solve(-F)", dc_source)
        self.assertNotIn("x += delta", dc_source)
        self.assertIn("self._write_back()", dc_source)
        self.assertNotIn("update_lf_info", dc_source)
        self.assertNotIn("runtime_params", dc_source)
        self.assertNotIn("divergence_threshold", dc_source)
        self.assertNotIn("np.linalg.lstsq", dc_source)

        self.assertNotIn("calc.model", inspect.getsource(dc_lf.print_dc_result))
        dc_main_source = inspect.getsource(dc_lf.main)
        self.assertIn("print_dc_result(calc, rc)", dc_main_source)
        self.assertNotIn("开始直流电网潮流计算", dc_main_source)

        dc_prepare_source = inspect.getsource(DCPowerFlowCalc._prepare_from_ppc)
        self.assertIn("预处理完成：节点数", dc_prepare_source)
        self.assertNotIn('print("self.N = "', dc_prepare_source)
        self.assertNotIn('print("self.N_phi = "', dc_prepare_source)
        self.assertNotIn('print("self.N_dcdc = "', dc_prepare_source)
        self.assertNotIn("print(x)", dc_prepare_source)
        self.assertNotIn('print("total_vars"', dc_prepare_source)
        self.assertNotIn('print("total_eq"', dc_prepare_source)

        ac_summary_source = inspect.getsource(ACPowerFlowCalc._write_summary_result)
        dc_summary_source = inspect.getsource(DCPowerFlowCalc._write_summary_result)
        self.assertIn("self.lf_result = None", ac_summary_source)
        self.assertIn("self.lf_result = None", dc_summary_source)
        for dc_only_field in ("node_count", "total_vars", "total_eq", "v_min", "v_max", "v_mean"):
            self.assertNotIn(dc_only_field, dc_summary_source)

    def test_lf_refresh_period_and_algorithm_branches_are_removed(self):
        from lfcore.ac_lf import ACPowerFlowCalc
        from lfcore.dc_lf import DCPowerFlowCalc
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        for cls in (ACPowerFlowCalc, DCPowerFlowCalc, HybridPowerFlowCalc):
            params = inspect.signature(cls.__init__).parameters
            self.assertNotIn("jacobian_refresh_period", params)
        for cls in (ACPowerFlowCalc, DCPowerFlowCalc):
            params = inspect.signature(cls.__init__).parameters
            self.assertNotIn("algorithm", params)

        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
            "src/hybrid_power_system_analysis/lfcore/hybrid_lf.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("jacobian_refresh_period", source, rel_path)
            self.assertNotIn("refresh_period", source, rel_path)
            self.assertNotIn("steps_since_refresh", source, rel_path)
            self.assertNotIn("used_algorithm", source, rel_path)
            self.assertNotIn("self.algorithm", source, rel_path)
            self.assertNotIn("--algorithm", source, rel_path)

    def test_lf_sparse_solver_helpers_are_shared(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf
        import lfcore.hybrid_lf as hybrid_lf
        import lfcore.solver_common as solver_common

        for module in (ac_lf, dc_lf, hybrid_lf):
            self.assertIs(module._resolve_linear_solver, solver_common.resolve_linear_solver)
            self.assertIs(module._factor_jacobian, solver_common.factor_jacobian)
        for module in (ac_lf, dc_lf):
            self.assertIs(module._load_named_sparse_solver, solver_common.load_named_sparse_solver)
            self.assertIs(module._OPTIONAL_SPARSE_SOLVERS, solver_common.OPTIONAL_SPARSE_SOLVERS)
            self.assertIs(module._OPTIONAL_SPARSE_MISSING, solver_common.OPTIONAL_SPARSE_MISSING)

        forbidden_tokens = (
            "def _load_named_sparse_solver",
            "def _resolve_linear_solver",
            "class _CallableFactor",
            "def _get_pyklu_cls",
            "def _factor_jacobian",
            "_OPTIONAL_SOLVER_CANDIDATES =",
        )
        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, rel_path)

    def test_lf_topology_helpers_are_shared(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.common as lf_common
        import lfcore.dc_lf as dc_lf
        import lfcore.hybrid_lf as hybrid_lf

        self.assertIs(ac_lf._device_key, lf_common.device_key)
        self.assertIs(dc_lf._device_key, lf_common.device_key)
        self.assertIs(hybrid_lf._lf_device_key, lf_common.device_key)
        self.assertIs(ac_lf.find_spanning_tree_edges, lf_common.find_spanning_tree_edges)
        self.assertIs(dc_lf.find_spanning_tree_edges, lf_common.find_spanning_tree_edges)
        self.assertEqual([0, 1], lf_common.find_spanning_tree_edges([(0, 1), (1, 2), (0, 2)], 3))

        forbidden_tokens = (
            "def _device_key",
            "def find_spanning_tree_edges",
        )
        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, rel_path)

    def test_ac_dc_array_model_helpers_are_shared(self):
        import model.ac_array_model as ac_array_model
        import model.array_common as array_common
        import model.dc_array_model as dc_array_model

        shared_names = (
            "_empty",
            "_rows_for",
            "_cell",
            "_float_cell",
            "_int_cell",
            "_float_column",
            "_int_column",
            "_code_column",
            "_names_from_rows",
            "_base_from_rows",
            "_scale_by_node",
            "_raw_vbase_by_node",
            "_value",
            "_float_value",
            "_int_value",
            "_code_value",
        )
        for name in shared_names:
            with self.subTest(helper=name):
                self.assertIs(getattr(ac_array_model, name), getattr(array_common, name))
                self.assertIs(getattr(dc_array_model, name), getattr(array_common, name))

        for rel_path in (
            "src/hybrid_power_system_analysis/model/ac_array_model.py",
            "src/hybrid_power_system_analysis/model/dc_array_model.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("class _EFileTableRows", source, rel_path)
            for name in shared_names:
                self.assertNotIn(f"def {name}", source, rel_path)

    def test_ac_dc_array_model_public_builders_are_parallel(self):
        import model.ac_array_model as ac_array_model
        import model.dc_array_model as dc_array_model

        def normalized_public_builders(module, kind):
            names = {
                name
                for name in dir(module)
                if name.startswith(f"build_{kind}_") and not name.startswith(f"build_{kind}_ppc_with")
            }
            return {name.replace(f"_{kind}_", "_grid_").replace(f"{kind}_", "grid_", 1) for name in names}

        self.assertEqual(
            normalized_public_builders(ac_array_model, "ac"),
            normalized_public_builders(dc_array_model, "dc"),
        )

    def test_ac_dc_network_public_flow_methods_are_parallel(self):
        import model.ac_model as ac_model
        import model.dc_model as dc_model

        for method_name in ("read_from_model", "read_from_file", "_load_from_model", "format_assoc", "topo"):
            with self.subTest(method=method_name):
                self.assertEqual(
                    inspect.signature(getattr(ac_model.ACPowerNetwork, method_name)),
                    inspect.signature(getattr(dc_model.DCPowerNetwork, method_name)),
                )

    def test_ac_dc_power_flow_reuses_existing_network_ppc(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf

        ac_ppc = {"format": "ac_ppc_v1"}
        dc_ppc = {"format": "dc_ppc_v1"}
        ac_network = SimpleNamespace(nodes=[], ppc=ac_ppc)
        dc_network = SimpleNamespace(nodes=[], ppc=dc_ppc)
        original_ac_builder = ac_lf.build_ac_ppc_from_network
        original_dc_builder = dc_lf.build_dc_ppc_from_network

        def reject_builder(*_args, **_kwargs):
            raise AssertionError("constructor should reuse existing network.ppc")

        ac_lf.build_ac_ppc_from_network = reject_builder
        dc_lf.build_dc_ppc_from_network = reject_builder
        try:
            ac_calc = ac_lf.ACPowerFlowCalc(ac_network)
            dc_calc = dc_lf.DCPowerFlowCalc(dc_network)
        finally:
            ac_lf.build_ac_ppc_from_network = original_ac_builder
            dc_lf.build_dc_ppc_from_network = original_dc_builder

        self.assertIs(ac_ppc, ac_calc.ppc)
        self.assertIs(dc_ppc, dc_calc.ppc)
        self.assertFalse(hasattr(ac_calc, "array_mode"))
        self.assertFalse(hasattr(dc_calc, "array_mode"))

    def test_ac_dc_ppc_from_network_preserves_source(self):
        from model.ac_array_model import build_ac_ppc_from_network
        from model.ac_model import ACPowerNetwork
        from model.dc_array_model import build_dc_ppc_from_network
        from model.dc_model import DCPowerNetwork

        ac_network = ACPowerNetwork()
        ac_network.source = "ac_source.e"
        dc_network = DCPowerNetwork()
        dc_network.source = "dc_source.e"

        self.assertEqual("ac_source.e", build_ac_ppc_from_network(ac_network)["source"])
        self.assertEqual("dc_source.e", build_dc_ppc_from_network(dc_network)["source"])

    def test_ac_dc_model_dictionary_aliases_are_consistent(self):
        import model.ac_model as ac_model
        import model.dc_model as dc_model

        ac_network = ac_model.ACPowerNetwork()
        dc_network = dc_model.DCPowerNetwork()

        self.assertIs(ac_network.zero_branch_dict, ac_network.zero_branche_dict)
        self.assertIs(dc_network.zero_branch_dict, dc_network.zero_branche_dict)
        self.assertIs(ac_network.branch_dict, ac_network.branche_dict)
        self.assertIs(dc_network.branch_dict, dc_network.branche_dict)

    def test_lf_result_mode_accepts_only_current_modes(self):
        from lfcore.ac_lf import ACPowerFlowCalc
        import lfcore.common as lf_common
        from lfcore.dc_lf import DCPowerFlowCalc
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        for mode in ("full", "array", "summary", "none"):
            with self.subTest(mode=mode):
                self.assertEqual(mode, lf_common.normalize_result_mode(mode, "LF"))
                self.assertEqual(mode, ACPowerFlowCalc._normalize_result_mode(mode))
                self.assertEqual(mode, DCPowerFlowCalc._normalize_result_mode(mode))
                self.assertEqual(mode, HybridPowerFlowCalc._normalize_result_mode(mode))
        for old_mode in ("all", "complete", "arrays", "ppc", "brief", "minimal", "off", "raw", "skip"):
            with self.subTest(old_mode=old_mode):
                with self.assertRaises(ValueError):
                    ACPowerFlowCalc._normalize_result_mode(old_mode)
                with self.assertRaises(ValueError):
                    DCPowerFlowCalc._normalize_result_mode(old_mode)
                with self.assertRaises(ValueError):
                    HybridPowerFlowCalc._normalize_result_mode(old_mode)

        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
            "src/hybrid_power_system_analysis/lfcore/hybrid_lf.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn('modes = {"full"', source, rel_path)
            self.assertNotIn("mode = str(result_mode", source, rel_path)

    def test_lf_old_compatibility_hooks_are_removed(self):
        checked_files = {
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py": (
                "def _cache_static_arrays",
                "def _load_ppc_static",
                "def _store_ppc_static",
                "_pf_static",
            ),
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py": (
                "_build_dc_ppc_from_e_file",
                "_DIRECT_PPC_STATIC_ATTRS",
                "_DIRECT_PPC_SHAPE_KEYS",
                "def _load_direct_ppc_static",
                "def _store_direct_ppc_static",
                "_dc_pf_static",
            ),
            "src/hybrid_power_system_analysis/lfcore/hybrid_lf.py": (),
        }
        for rel_path, extra_tokens in checked_files.items():
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("_run_with_optional_output", source, rel_path)
            self.assertNotIn("solve_sparse_system", source, rel_path)
            self.assertNotIn("monkey-patch", source, rel_path)
            self.assertNotIn("兼容", source, rel_path)
            self.assertNotIn("compatibility", source, rel_path)
            self.assertNotIn("standard_jac_csr_order", source, rel_path)
            for token in extra_tokens:
                self.assertNotIn(token, source, rel_path)

    def test_hybrid_lf_run_matches_ac_dc_control_flow(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf
        import lfcore.hybrid_lf as hybrid_lf
        from lfcore.ac_lf import ACPowerFlowCalc
        from lfcore.dc_lf import DCPowerFlowCalc
        from lfcore.hybrid_lf import HybridPowerFlowCalc

        self.assertFalse(hasattr(hybrid_lf, "run_hybrid_power_flow"))
        self.assertFalse(hasattr(hybrid_lf, "_hybrid_result_from_calc"))
        self.assertIn("return self._run_newton_raphson()", inspect.getsource(hybrid_lf.HybridPowerFlowCalc.run))
        self.assertTrue(hasattr(hybrid_lf.HybridPowerFlowCalc, "_run_newton_raphson"))

        ac_init = [name for name in inspect.signature(ACPowerFlowCalc.__init__).parameters if name != "self"]
        hybrid_init = [name for name in inspect.signature(HybridPowerFlowCalc.__init__).parameters if name != "self"]
        self.assertEqual(ac_init, hybrid_init)
        self.assertEqual(inspect.signature(ACPowerFlowCalc._build_newton_system), inspect.signature(HybridPowerFlowCalc._build_newton_system))
        self.assertEqual(inspect.signature(ACPowerFlowCalc._run_newton_raphson), inspect.signature(HybridPowerFlowCalc._run_newton_raphson))
        self.assertEqual(inspect.signature(ACPowerFlowCalc.get_f), inspect.signature(HybridPowerFlowCalc.get_f))
        self.assertEqual(inspect.signature(ACPowerFlowCalc.get_jacobi), inspect.signature(HybridPowerFlowCalc.get_jacobi))
        self.assertEqual(inspect.signature(ACPowerFlowCalc._write_summary_result), inspect.signature(HybridPowerFlowCalc._write_summary_result))
        self.assertEqual(inspect.signature(ACPowerFlowCalc._write_back), inspect.signature(HybridPowerFlowCalc._write_back))
        self.assertEqual(
            list(inspect.signature(ac_lf.print_ac_result).parameters),
            list(inspect.signature(hybrid_lf.print_hybrid_result).parameters),
        )
        self.assertEqual(
            inspect.signature(ac_lf.print_ac_result).return_annotation,
            inspect.signature(hybrid_lf.print_hybrid_result).return_annotation,
        )

        hybrid_source = inspect.getsource(HybridPowerFlowCalc._run_newton_raphson)
        self.assertIn("delta = factor.solve(F)", hybrid_source)
        self.assertIn("x -= delta", hybrid_source)
        self.assertIn("self._write_back()", hybrid_source)
        self.assertNotIn("factor.solve(-F)", hybrid_source)
        self.assertNotIn("x += delta", hybrid_source)
        self.assertNotIn("_finish_result", hybrid_source)
        self.assertFalse(hasattr(HybridPowerFlowCalc, "_finish_result"))
        self.assertFalse(hasattr(HybridPowerFlowCalc, "_write_none_result"))
        self.assertFalse(hasattr(HybridPowerFlowCalc, "_write_array_result"))

        self.assertIn("self.result =", inspect.getsource(HybridPowerFlowCalc._write_summary_result))
        self.assertIn("self.lf_result = None", inspect.getsource(HybridPowerFlowCalc._write_summary_result))
        hybrid_summary_source = inspect.getsource(HybridPowerFlowCalc._hybrid_summary)
        self.assertNotIn('"total_vars"', hybrid_summary_source)
        self.assertNotIn('"total_eq"', hybrid_summary_source)
        self.assertNotIn(
            'self.lf_result = {"ac":',
            inspect.getsource(HybridPowerFlowCalc._sync_single_subsolver_result),
        )
        hybrid_main_source = inspect.getsource(hybrid_lf.main)
        self.assertIn("print_hybrid_result(calc, rc)", hybrid_main_source)
        self.assertNotIn("print_hybrid_result(calc.lf_result)", hybrid_main_source)
        self.assertNotIn("calc.model", inspect.getsource(HybridPowerFlowCalc._build_dc_subcalc))
        self.assertNotIn("calc.net", inspect.getsource(HybridPowerFlowCalc._build_dc_subcalc))

        source = (ROOT_DIR / "src" / "hybrid_power_system_analysis" / "lfcore" / "hybrid_lf.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "def run_hybrid_power_flow",
            "def _hybrid_result_from_calc",
            "def _append_converter_residuals",
            "ac_warnings=None",
            "ac_errors=None",
            "dc_warnings=None",
            "dc_errors=None",
            "result.ac_warnings =",
            "result.ac_errors =",
            "result.dc_warnings =",
            "result.dc_errors =",
            '"converged": False',
        ):
            self.assertNotIn(token, source)

    def test_ac_dc_lf_array_mode_switch_is_removed(self):
        import lfcore.ac_lf as ac_lf
        import lfcore.dc_lf as dc_lf

        ac_calc = ac_lf.ACPowerFlowCalc({"format": "ac_ppc_v1"})
        dc_calc = dc_lf.DCPowerFlowCalc({"format": "dc_ppc_v1"})

        self.assertFalse(hasattr(ac_calc, "array_mode"))
        self.assertFalse(hasattr(dc_calc, "array_mode"))

        for rel_path in (
            "src/hybrid_power_system_analysis/lfcore/ac_lf.py",
            "src/hybrid_power_system_analysis/lfcore/dc_lf.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8-sig")
            self.assertNotIn("self.array_mode", source, rel_path)
            self.assertNotIn("calc.array_mode", source, rel_path)
            self.assertNotIn("if True:", source, rel_path)
            self.assertNotIn("if False:", source, rel_path)
            self.assertNotIn("return\n\n        theta, V, phi_re, phi_im", source, rel_path)
            self.assertNotIn("model = self.net", source, rel_path)
            self.assertNotIn("alive_branch_tuple", source, rel_path)
            self.assertNotIn("alive_dcdc_tuples", source, rel_path)
            self.assertNotIn("def _prepare_zero_branches", source, rel_path)
            self.assertNotIn("def _build_y_matrix", source, rel_path)
            self.assertNotIn("def _prepare_node_parameters", source, rel_path)
            self.assertNotIn("def _device_voltage", source, rel_path)
            self.assertNotIn("def _get_jacobi_loop", source, rel_path)
            self.assertNotIn("def _result_node_voltage", source, rel_path)
            self.assertNotIn("def _live_ppc_terminal_pair", source, rel_path)
            self.assertNotIn("def _direct_ppc_pair_positions", source, rel_path)
            self.assertNotIn("def _component_labels_from_edges", source, rel_path)
            self.assertNotIn("def build_jacobian_matrix", source, rel_path)

    def test_ppc_base_compatibility_helper_is_removed(self):
        self.assertFalse((ROOT_DIR / "src" / "hybrid_power_system_analysis" / "model" / "ppc_base.py").exists())
        for rel_path in (
            "src/hybrid_power_system_analysis/model/ac_array_model.py",
            "src/hybrid_power_system_analysis/model/dc_array_model.py",
            "src/hybrid_power_system_analysis/model/hybrid_array_model.py",
            "src/hybrid_power_system_analysis/lfcore/hybrid_lf.py",
            "src/hybrid_power_system_analysis/secore/ac_se.py",
            "src/hybrid_power_system_analysis/secore/hybrid_se.py",
        ):
            source = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("ppc_base", source, rel_path)
            self.assertNotIn("base[0]", source, rel_path)
            self.assertNotIn("base[1]", source, rel_path)
            self.assertNotIn("base[2]", source, rel_path)
            self.assertNotIn("base[3]", source, rel_path)
            self.assertNotIn("base[4]", source, rel_path)

    def test_empty_ppc_helpers_are_removed_from_array_modules(self):
        import model.ac_array_model as ac_array_model
        import model.dc_array_model as dc_array_model

        self.assertFalse(hasattr(ac_array_model, "build_empty_ac_ppc"))
        self.assertFalse(hasattr(dc_array_model, "build_empty_dc_ppc"))
        self.assertFalse(hasattr(dc_array_model, "DCPowerNetwork"))

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

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"
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

        case_path = ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"
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

        case_path = ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"
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

        case_path = ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"
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

    def test_dc_model_initializes_topology_containers_like_ac_model(self):
        import model.ac_model as ac_model
        import model.dc_model as dc_model

        ac_network = ac_model.ACPowerNetwork()
        dc_network = dc_model.DCPowerNetwork()

        self.assertIsInstance(ac_network.islands, list)
        self.assertIsInstance(dc_network.islands, list)

    def test_dc_model_coerces_generic_rows_like_ac_model(self):
        import model.dc_model as dc_model

        model = SimpleNamespace(
            p_base=100.0,
            p_base_kW=100000.0,
            u_scale=1.0,
            p_scale=1.0,
            i_scale=1.0,
            DCNode=[SimpleNamespace(idx=1, name="n1", vbase=10.0, voltage=1.0, run_stat=1)],
            DCBranch=[SimpleNamespace(idx=2, name="br2", i_node=1, j_node=1, r=0.01, run_stat=1)],
            DCGenerator=[SimpleNamespace(idx=3, name="gen3", node=1, control_type="V", p_set=0.0, v_set=1.0, i_set=0.0, run_stat=1)],
            DCLoad=[SimpleNamespace(idx=4, name="load4", node=1, pbase=1.0, pv0=0.0, pv1=0.0, pv2=0.0, run_stat=1)],
            DCSwitch=[SimpleNamespace(idx=5, name="sw5", i_node=1, j_node=1, status=1, run_stat=1)],
            DCBreak=[SimpleNamespace(idx=6, name="brk6", i_node=1, j_node=1, status=1, run_stat=1)],
            DCZeroBranch=[SimpleNamespace(idx=7, name="zbr7", i_node=1, j_node=1, run_stat=1)],
            DCDCConverter=[
                SimpleNamespace(
                    idx=8,
                    name="dcdc8",
                    i_node=1,
                    j_node=1,
                    r1=0.01,
                    r2=0.01,
                    i_control_type="P",
                    j_control_type="NONE",
                    p_set=0.0,
                    i_set=0.0,
                    v_set=1.0,
                    run_stat=1,
                )
            ],
        )

        network = dc_model.DCPowerNetwork()
        network.model = model
        network._load_from_model(units_already_normalized=True)

        self.assertIsInstance(network.nodes[0], dc_model.DCNode)
        self.assertIsInstance(network.branches[0], dc_model.DCBranch)
        self.assertIsInstance(network.generators[0], dc_model.DCGenerator)
        self.assertIsInstance(network.loads[0], dc_model.DCLoad)
        self.assertIsInstance(network.switches[0], dc_model.DCSwitch)
        self.assertIsInstance(network.breakers[0], dc_model.DCBreak)
        self.assertIsInstance(network.zero_branches[0], dc_model.DCZeroBranch)
        self.assertIsInstance(network.dcdc_converters[0], dc_model.DCDCConverter)
        self.assertEqual("n1", network.nodes[0].name)
        self.assertEqual("brk6", network.breakers[0].name)


if __name__ == "__main__":
    unittest.main()
