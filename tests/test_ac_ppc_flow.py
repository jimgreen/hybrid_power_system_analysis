import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "model"))
sys.path.insert(0, str(ROOT_DIR / "lfcore"))


class ACPPCFlowTest(unittest.TestCase):
    def test_object_y_matrix_uses_vectorized_branch_stamps(self):
        import ac_lf
        from ac_flow import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "ac" / "ieee39.e"
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
        from ac_flow import ACPowerFlowCalc
        from ac_model import ACPowerNetwork

        case_path = ROOT_DIR / "data" / "ac" / "ieee300.e"

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
        ppc_calc = ACPowerFlowCalc.from_ppc(ppc, tol=1e-8, max_iter=50)
        with contextlib.redirect_stdout(io.StringIO()):
            ppc_calc.prepare()
            ppc_rc = ppc_calc.run()
        self.assertEqual(0, ppc_rc)
        self.assertTrue(ppc_calc.converged)

        object_voltage = np.asarray([node.voltage for node in object_calc.node_list])
        object_angle = np.asarray([node.angle for node in object_calc.node_list])

        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["voltage"]], object_voltage, atol=1e-10)
        np.testing.assert_allclose(ppc_calc.result["bus"][:, ppc["bus_cols"]["angle"]], object_angle, atol=1e-10)

    def test_ppc_standard_jacobian_avoids_sparse_block_stack(self):
        import ac_lf
        from ac_array_model import build_ac_ppc_from_e_file
        from ac_flow import ACPowerFlowCalc

        case_path = ROOT_DIR / "data" / "ac" / "ieee300.e"
        ppc = build_ac_ppc_from_e_file(case_path)
        calc = ACPowerFlowCalc.from_ppc(ppc, tol=1e-8, max_iter=50)
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

    def test_efile_dict_cache_reuses_unchanged_parse(self):
        from efile_read import clear_efile_cache, read_efile_dict_cached

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
            path = Path(tmp_dir) / "cache.e"
            path.write_text(content, encoding="utf-8")
            clear_efile_cache()
            first = read_efile_dict_cached(path)
            second = read_efile_dict_cached(path)

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
