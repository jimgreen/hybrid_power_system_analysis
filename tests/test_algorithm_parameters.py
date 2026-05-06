import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_lf_para(path: Path, tol: str = "1e-5", max_iter: str = "7") -> None:
    path.write_text(
        "\n".join(
            [
                "<PowerFlowParameter>",
                "@ name value",
                f"# tol {tol}",
                f"# max_iter {max_iter}",
                "# min_voltage 0.123",
                "# divergence_threshold 98765",
                "</PowerFlowParameter>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_se_para(
    path: Path,
    max_iter: str = "3",
    pseudo_weight: str = "0.002",
    pseudo_obs_max: str = "17",
) -> None:
    path.write_text(
        "\n".join(
            [
                "<StateEstimationParameter>",
                "@ name value",
                "# tol 1e-5",
                f"# max_iter {max_iter}",
                "# diff_step 1e-7",
                "# flat_start 1",
                "# bad_threshold 2.5",
                "# max_remove 4",
                f"# pseudo_measurement_weight {pseudo_weight}",
                f"# targeted_pseudo_measurement_max {pseudo_obs_max}",
                "# voltage_floor 0.07",
                "# min_current_voltage 0.02",
                "# power_flow_tol 1e-6",
                "# power_flow_max_iter 9",
                "# power_flow_min_voltage 0.03",
                "</StateEstimationParameter>",
                "",
            ]
        ),
        encoding="utf-8",
    )


class AlgorithmParameterFileTest(unittest.TestCase):
    def test_power_flow_classes_read_algorithm_parameters_from_lf_para(self):
        from algorithm_parameters import load_lf_parameters
        from lfcore.ac_flow import ACPowerFlowCalc
        from lfcore.dc_flow import DCPowerFlowCalc
        from lfcore.hybrid_flow import HybridPowerFlowCalc, HybridPowerNetwork

        with tempfile.TemporaryDirectory() as tmp_dir:
            para_file = Path(tmp_dir) / "lf.para"
            _write_lf_para(para_file)

            params = load_lf_parameters(para_file)
            self.assertEqual(1e-5, params.tol)
            self.assertEqual(7, params.max_iter)
            self.assertEqual(0.123, params.min_voltage)
            self.assertEqual(98765.0, params.divergence_threshold)

            ac_calc = ACPowerFlowCalc(object(), parameter_file=para_file)
            self.assertEqual(params.tol, ac_calc.tol)
            self.assertEqual(params.max_iter, ac_calc.max_iter)
            self.assertEqual(params.min_voltage, ac_calc.min_voltage)

            dc_calc = DCPowerFlowCalc(object(), parameter_file=para_file)
            self.assertEqual(params.tol, dc_calc.params.tol)
            self.assertEqual(params.max_iter, dc_calc.params.max_iter)
            self.assertEqual(params.min_voltage, dc_calc.params.min_voltage)

            network = HybridPowerNetwork.read_from_file(ROOT_DIR / "data" / "hybrid" / "hybrid_net_40.e")
            hybrid_calc = HybridPowerFlowCalc(network, parameter_file=para_file, verbose=False)
            self.assertEqual(params.tol, hybrid_calc.tol)
            self.assertEqual(params.max_iter, hybrid_calc.max_iter)
            self.assertEqual(params.min_voltage, hybrid_calc.params.min_voltage)

    def test_state_estimator_reads_algorithm_parameters_from_se_para(self):
        from algorithm_parameters import load_se_parameters
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            para_file = Path(tmp_dir) / "se.para"
            meas_file = Path(tmp_dir) / "voltage_only.meas"
            _write_se_para(para_file)
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 vm_nd0 ACNode nd_0 V 1.0 1 1.06",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            params = load_se_parameters(para_file)
            self.assertEqual(17, params.targeted_pseudo_measurement_max)

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ac_net_10.e",
                meas_file=meas_file,
                parameter_file=para_file,
            )

            self.assertEqual(1e-5, estimator.tol)
            self.assertEqual(3, estimator.max_iter)
            self.assertEqual(1e-7, estimator.diff_step)
            self.assertTrue(estimator.flat_start)
            self.assertEqual(0.002, estimator.pseudo_measurement_weight)
            self.assertEqual(17, estimator.targeted_pseudo_measurement_max)
            self.assertEqual(0.07, estimator.voltage_floor)
            self.assertEqual(0.02, estimator.min_current_voltage)

            pseudo_weights = {
                meas.weight
                for meas in estimator.active_measurements
                if meas.name.startswith("pseudo_")
            }
            self.assertEqual({0.002}, pseudo_weights)


if __name__ == "__main__":
    unittest.main()
