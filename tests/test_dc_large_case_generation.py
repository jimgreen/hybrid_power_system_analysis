import contextlib
import io
import tempfile
import unittest
from pathlib import Path


class DCLargeCaseGenerationTest(unittest.TestCase):
    def test_dcdc_residual_and_jacobian_use_vectorized_control_arrays(self):
        import numpy as np
        from lfcore.dc_lf import DCPowerFlowCalc
        from model.dc_model import DCPowerNetwork

        class NonIterableControls:
            def __iter__(self):
                raise AssertionError("DCDC control equations should use cached vectorized masks")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "dc" / "dc_net_30.e")
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
        from model.dc_model import DCPowerNetwork

        class NonIterableBranches:
            def __iter__(self):
                raise AssertionError("DC load-flow backfill should use cached branch arrays")

        network = DCPowerNetwork()
        network.read_from_file(Path(__file__).resolve().parents[1] / "data" / "dc" / "dc_net_30.e")
        network.topo()
        calc = DCPowerFlowCalc(network)
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = calc.prepare()
        calc.runtime_params = calc.params
        network.branches = NonIterableBranches()

        calc.update_lf_info(x)

        self.assertTrue(all(node.voltage > 0.0 for node in network.nodes if node.is_alive))

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
