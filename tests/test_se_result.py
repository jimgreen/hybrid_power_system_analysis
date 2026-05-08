import unittest
from pathlib import Path

import numpy as np

from model.meas_model import BadDataItem, EstimateResult, Measurement, ObservabilityResult


ROOT_DIR = Path(__file__).resolve().parents[1]


class SEResultTest(unittest.TestCase):
    def test_estimate_result_is_exported_from_measurement_model(self):
        import model.meas_model as meas_model

        self.assertIs(EstimateResult, meas_model.EstimateResult)

    def test_nested_tables_store_rows_and_export_dicts(self):
        from secore.se_result import SEResult

        result = SEResult()
        meas = Measurement(1, "v_n1", "DCNode", "n1", "V", 2.0, True, 1.01)

        result.normal_measurements.append(
            meas,
            estimated_value=1.0,
            residual=0.01,
            normalized_residual=0.2,
        )

        self.assertIsInstance(result.statistics, SEResult.StatisticsTable)
        self.assertIsInstance(result.prefiltered_measurements, SEResult.PrefilteredMeasurementTable)
        self.assertIsInstance(result.pseudo_measurements, SEResult.PseudoMeasurementTable)
        self.assertIsInstance(result.bad_data, SEResult.BadDataTable)
        self.assertIsInstance(result.normal_measurements, SEResult.NormalMeasurementTable)
        self.assertEqual(1, len(result.normal_measurements))
        self.assertEqual(
            {
                "idx": 1,
                "name": "v_n1",
                "device_type": "DCNode",
                "device_name": "n1",
                "meas_type": "V",
                "weight": 2.0,
                "valid": True,
                "value": 1.01,
                "estimated_value": 1.0,
                "residual": 0.01,
                "normalized_residual": 0.2,
                "reason": "",
                "source": "",
            },
            result.normal_measurements.to_dicts()[0],
        )

    def test_builds_tables_from_estimate_result(self):
        from secore.se_result import SEResult

        good = Measurement(1, "v_n1", "DCNode", "n1", "V", 1.0, True, 1.0)
        pseudo = Measurement(2, "pseudo_v_n2", "DCNode", "n2", "V", 1e-6, True, 1.0)
        filtered = Measurement(3, "v_off", "DCNode", "n3", "V", 1.0, False, 0.0)
        bad_meas = Measurement(4, "v_bad", "DCNode", "n4", "V", 1.0, True, 2.0)
        obs = ObservabilityResult(
            observable=True,
            rank=2,
            state_count=2,
            measurement_count=3,
            deficiency=0,
            singular_values=np.array([2.0, 1.0]),
            weak_states=[],
        )
        estimate = EstimateResult(
            converged=True,
            iterations=3,
            objective=1.25,
            max_correction=1e-8,
            residual_inf=0.2,
            x=np.array([1.0, 1.0]),
            z_est=np.array([1.0, 1.0, 1.8]),
            residual=np.array([0.0, 0.0, 0.2]),
            H=None,
            gain=None,
            measurements=[good, pseudo, bad_meas],
            observability=obs,
        )
        bad = BadDataItem(
            measurement=bad_meas,
            residual=0.2,
            normalized_residual=4.0,
            estimated_value=1.8,
            measured_value=2.0,
        )

        result = SEResult.from_estimate_result(
            estimate,
            bad_items=[bad],
            normalized_residual=np.array([0.0, 0.0, 4.0]),
            prefiltered_measurements=[(filtered, "invalid row")],
        )

        self.assertEqual(3, result.statistics.iterations)
        self.assertEqual(2, result.statistics.state_count)
        self.assertEqual(3, result.statistics.measurement_count)
        self.assertEqual(1, result.statistics.prefiltered_measurement_count)
        self.assertEqual(1, result.statistics.pseudo_measurement_count)
        self.assertEqual(1, result.statistics.bad_data_count)
        self.assertEqual(1, result.statistics.normal_measurement_count)
        self.assertEqual("invalid row", result.prefiltered_measurements[0].reason)
        self.assertEqual("pseudo_v_n2", result.pseudo_measurements[0].name)
        self.assertEqual("v_bad", result.bad_data[0].name)
        self.assertEqual(4.0, result.bad_data[0].normalized_residual)
        self.assertEqual("v_n1", result.normal_measurements[0].name)

    def test_state_estimators_build_se_result_after_estimation(self):
        from secore.ac_se import ACStateEstimator
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator
        from secore.se_result import SEResult

        good = Measurement(1, "v_n1", "DCNode", "n1", "V", 1.0, True, 1.0)
        pseudo = Measurement(2, "pseudo_v_n2", "DCNode", "n2", "V", 1e-4, True, 1.0)
        filtered = Measurement(3, "v_off", "DCNode", "n3", "V", 1.0, False, 0.0)
        bad_meas = Measurement(4, "v_bad", "DCNode", "n4", "V", 1.0, True, 2.0)
        obs = ObservabilityResult(True, 2, 2, 3, 0, np.array([2.0, 1.0]), [])
        estimate = EstimateResult(
            converged=True,
            iterations=2,
            objective=0.5,
            max_correction=1e-9,
            residual_inf=0.2,
            x=np.array([1.0, 1.0]),
            z_est=np.array([1.0, 1.0, 1.8]),
            residual=np.array([0.0, 0.0, 0.2]),
            H=None,
            gain=None,
            measurements=[good, pseudo, bad_meas],
            observability=obs,
        )
        bad_item = BadDataItem(bad_meas, 0.2, 4.0, 1.8, 2.0)

        for estimator_cls in (ACStateEstimator, DCStateEstimator, HybridStateEstimator):
            estimator = estimator_cls.__new__(estimator_cls)
            estimator.measurements = [good, pseudo, filtered, bad_meas]
            estimator.identify_bad_data = lambda _result, _threshold=None: (
                [bad_item],
                np.array([0.0, 0.0, 4.0]),
            )

            se_result = estimator.build_se_result(estimate)

            self.assertIsInstance(se_result, SEResult)
            self.assertEqual(1, len(se_result.prefiltered_measurements))
            self.assertEqual(1, len(se_result.pseudo_measurements))
            self.assertEqual(1, len(se_result.bad_data))
            self.assertEqual(1, len(se_result.normal_measurements))
            self.assertIs(se_result, estimator.se_result)

    def test_estimate_returns_estimate_result_and_builds_se_result_afterwards(self):
        from secore.dc_se import DCStateEstimator
        from secore.se_result import SEResult

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )

        observability = estimator.observability_analysis()
        result = estimator.estimate(observability=observability)

        self.assertIsInstance(result, EstimateResult)
        self.assertIs(result.observability, observability)
        se_result = estimator.build_se_result(result)
        self.assertIsInstance(se_result, SEResult)
        self.assertEqual(se_result.statistics.iterations, result.iterations)
        self.assertEqual(se_result.statistics.measurement_count, len(result.measurements))

    def test_dc_main_calls_build_se_result_after_bad_data_analysis(self):
        import contextlib
        import io
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        calls = []
        original_build = DCStateEstimator.build_se_result

        def counted_build(self, result, *args, **kwargs):
            calls.append((result, kwargs))
            return original_build(self, result, *args, **kwargs)

        DCStateEstimator.build_se_result = counted_build
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = dc_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "dc" / "dc_net_30.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "dc" / "dc_net_30.meas"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            DCStateEstimator.build_se_result = original_build

        self.assertEqual(0, rc)
        self.assertEqual(1, len(calls))
        self.assertIsInstance(calls[0][0], EstimateResult)
        self.assertIn("bad_items", calls[0][1])
        self.assertIn("normalized_residual", calls[0][1])

    def test_hybrid_targeted_pseudos_loop_until_observable(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator.__new__(HybridStateEstimator)
        estimator.targeted_pseudo_measurement_max = 5
        estimator.measurements = []
        observability_calls = []

        def observability_analysis():
            observability_calls.append(1)
            if len(observability_calls) == 1:
                return ObservabilityResult(False, 1, 2, 1, 1, np.array([]), [("AC_I_RE:zbr_1", 0.0)])
            return ObservabilityResult(True, 2, 2, 2, 0, np.array([]), [])

        def append_targeted(next_idx, state_label, existing_keys, existing_names, max_add):
            self.assertEqual("AC_I_RE:zbr_1", state_label)
            pseudo = Measurement(next_idx, f"pseudo_obs_{next_idx}", "ACZeroBranch", "zbr_1", "P_FROM", 1e-4, True, 0.0)
            estimator.measurements.append(pseudo)
            existing_keys.add((pseudo.device_type, pseudo.device_name, pseudo.meas_type))
            existing_names.add(pseudo.name)
            return next_idx + 1, 1

        estimator.observability_analysis = observability_analysis
        estimator._active_measurement_keys = lambda: set()
        estimator._append_targeted_observability_pseudo = append_targeted
        estimator._refresh_active_measurement_state_layout = lambda: None

        self.assertEqual(1, estimator._add_targeted_observability_pseudo_measurements())
        self.assertEqual(2, len(observability_calls))


if __name__ == "__main__":
    unittest.main()
