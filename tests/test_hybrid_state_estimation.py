import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class HybridStateEstimationTest(unittest.TestCase):
    @staticmethod
    def _all_valid_measurement_file(tmp_dir, source: Path) -> Path:
        """Build a temporary measurement file with every existing row marked valid."""
        target = Path(tmp_dir) / source.name
        lines = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                parts = line.split()
                if len(parts) >= 9:
                    parts[7] = "1"
                    line = " ".join(parts)
            lines.append(line)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def test_hybrid_jacobian_uses_direct_derivatives_without_repeated_evaluation(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )

        original_evaluate = estimator.evaluate
        call_count = 0

        def counted_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_evaluate(*args, **kwargs)

        estimator.evaluate = counted_evaluate
        H = estimator.jacobian(estimator.initial_state())

        self.assertEqual((len(estimator.active_measurements), estimator.n_state), H.shape)
        self.assertLessEqual(call_count, 1)

    def test_hybrid_dc_pseudo_and_constraints_use_prepared_side_summary(self):
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        original_refresh = DCStateEstimator._refresh_measurement_summary_cache
        call_count = 0

        def counted_refresh(self):
            nonlocal call_count
            call_count += 1
            return original_refresh(self)

        DCStateEstimator._refresh_measurement_summary_cache = counted_refresh
        try:
            HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
                flat_start=True,
            )
        finally:
            DCStateEstimator._refresh_measurement_summary_cache = original_refresh

        self.assertEqual(0, call_count)

    def test_hybrid_converter_pseudo_summary_is_single_pass(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        before_count = len(estimator.measurements)

        def fail_redundant_scan(*_args, **_kwargs):
            raise AssertionError("hybrid converter pseudo preparation should use one summary scan")

        estimator._active_device_keys = fail_redundant_scan
        estimator._add_hybrid_pseudo_measurements()

        self.assertEqual(before_count, len(estimator.measurements))

    def test_hybrid_measurement_loader_reuses_table_backed_parser(self):
        from model.meas_model import MeasurementList
        from secore.hybrid_se import HybridStateEstimator

        measurements = HybridStateEstimator._load_measurements(ROOT_DIR / "data" / "hybrid" / "qinling.meas")

        self.assertIsInstance(measurements, MeasurementList)
        self.assertIsNotNone(measurements.table)
        self.assertEqual(len(measurements), len(measurements.table.idx))
        self.assertGreater(len(measurements), 0)
        self.assertEqual("ACNode", measurements[0].device_type)
        self.assertEqual("V", measurements[0].meas_type)

    def test_hybrid_side_measurement_slices_preserve_table_cache(self):
        from model.meas_model import MeasurementList
        from secore.hybrid_se import HybridStateEstimator, Measurement

        estimator = HybridStateEstimator.__new__(HybridStateEstimator)
        estimator.measurements = HybridStateEstimator._load_measurements(
            ROOT_DIR / "data" / "hybrid" / "qinling.meas"
        )
        estimator._sub_measurements_converted_by_side = {}

        sources = estimator._initial_measurement_sources_by_side()
        ac_for_sub = estimator._measurements_for_sub_estimator("ac", share_measurements=False)

        self.assertIsInstance(sources["ac"], MeasurementList)
        self.assertIsNotNone(sources["ac"].table)
        self.assertIsInstance(ac_for_sub, MeasurementList)
        self.assertIs(ac_for_sub.table, sources["ac"].table)
        self.assertIsNot(ac_for_sub, sources["ac"])
        before_count = len(sources["ac"])
        ac_for_sub.append(Measurement(-1, "pseudo_probe", "ACNode", "n", "V", 1.0, True, 1.0))
        self.assertEqual(before_count, len(sources["ac"]))

    def test_ieee3k_flat_start_does_not_add_angle_pseudos(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
        )
        ac = estimator.calc.ac_calc
        full_x = estimator._expand_state(estimator.initial_state())
        theta, _voltage, _, _ = ac._extract_state_vars(full_x[: estimator.calc.ac_size], update_cache=False)

        self.assertFalse(
            any(
                meas.name.startswith(("pseudo_angle_", "pseudo_obs_angle_", "constraint_angle"))
                or meas.meas_type in ("ANGLE", "THETA", "ANGLE_DIFF", "THETA_DIFF")
                for meas in estimator.active_measurements
                if meas.name.startswith(("pseudo_", "constraint_"))
            )
        )
        np.testing.assert_allclose(theta, 0.0)

    def test_ieee3k_flat_start_first_step_keeps_angles_zero_without_angle_pseudos(self):
        import warnings
        from scipy.sparse.linalg import MatrixRankWarning
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
            max_iter=1,
        )
        x0 = estimator.initial_state()
        ac = estimator.calc.ac_calc

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.estimate(verbose=False)
        full_x = estimator._expand_state(x0)
        theta, _voltage, _, _ = ac._extract_state_vars(full_x[: estimator.calc.ac_size], update_cache=False)

        np.testing.assert_allclose(theta, 0.0)
        self.assertTrue(np.isfinite(result.objective))
        self.assertFalse(any(meas.meas_type in ("ANGLE", "THETA") for meas in result.measurements))
        self.assertFalse(any(isinstance(w.message, MatrixRankWarning) for w in caught))

    def test_ac_angle_residuals_wrap_across_two_pi(self):
        from dataclasses import replace
        from secore.hybrid_se import HybridStateEstimator, Measurement

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee300.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee300.meas",
            flat_start=False,
        )
        x0 = estimator.initial_state()
        meas = Measurement(
            idx=-1,
            name="angle_wrap_probe",
            device_type="ACNode",
            device_name=estimator.ac_nodes[0].name,
            meas_type="ANGLE",
            weight=1.0,
            valid=True,
            value=0.0,
        )
        single_z_est = estimator.evaluate(x0, [meas])
        wrapped_meas = replace(meas, value=float(single_z_est[0] - 2.0 * np.pi + 0.04))

        residual = estimator._measurement_residual(
            np.array([wrapped_meas.value], dtype=np.float64),
            single_z_est,
            [wrapped_meas],
        )

        self.assertAlmostEqual(0.04, float(residual[0]), places=12)

    def test_ieee39_flat_start_reuses_ac_state_estimator_path(self):
        from secore.ac_se import ACStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            flat_start=True,
        )

        ac_estimator = ACStateEstimator(**kwargs)
        hybrid_estimator = HybridStateEstimator(**kwargs)

        ac_result = ac_estimator.estimate(verbose=False, final_diagnostics=False)
        hybrid_result = hybrid_estimator.estimate(verbose=False)

        self.assertIsInstance(hybrid_estimator._ac_sub_estimator, ACStateEstimator)
        self.assertTrue(hybrid_result.converged)
        self.assertEqual(ac_result.iterations, hybrid_result.iterations)
        self.assertAlmostEqual(ac_result.objective, hybrid_result.objective, places=14)
        self.assertAlmostEqual(ac_result.residual_inf, hybrid_result.residual_inf, places=12)

    def test_ieee39_flat_start_exposes_ac_state_layout_contract(self):
        from secore.ac_se import ACStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            flat_start=True,
        )

        ac_estimator = ACStateEstimator(**kwargs)
        hybrid_estimator = HybridStateEstimator(**kwargs)

        self.assertEqual(ac_estimator.state_labels, hybrid_estimator.ac_state_labels)
        self.assertEqual(ac_estimator.n_state, hybrid_estimator.ac_n_state)
        self.assertEqual(ac_estimator.state_layout()["state_labels"], hybrid_estimator.ac_state_layout["state_labels"])
        self.assertEqual(ac_estimator.state_layout()["n_state"], hybrid_estimator.ac_state_layout["n_state"])
        self.assertIsInstance(hybrid_estimator._delegate(), ACStateEstimator)
        self.assertEqual(hybrid_estimator.n_state, hybrid_estimator.ac_state_cols.size)
        self.assertFalse(hybrid_estimator.dc_state_cols.size)
        self.assertFalse(hybrid_estimator.hybrid_state_cols.size)
        np.testing.assert_allclose(ac_estimator.initial_state(), hybrid_estimator.initial_state())

    def test_ieee3k_flat_start_keeps_angle_state_zero(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
            max_iter=25,
        )
        ac = estimator.calc.ac_calc

        full_x = estimator._expand_state(estimator.initial_state())
        theta, _voltage, _, _ = ac._extract_state_vars(full_x[: estimator.calc.ac_size], update_cache=False)

        np.testing.assert_allclose(theta, 0.0)

    def test_ac_reference_node_uses_highest_degree_node_with_valid_voltage_measurement(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        ac = estimator.calc.ac_calc
        ref = estimator.ac_node_by_name["bus_16"]
        ref_pos = ac.node_pos[ref.idx]
        ref_voltage = next(
            meas.value
            for meas in estimator.measurements
            if meas.device_type == "ACNode"
            and meas.device_name == "bus_16"
            and meas.meas_type == "V"
            and meas.valid
        )

        self.assertEqual(["bus_16"], [node.name for node in estimator.ac_reference_nodes])
        self.assertEqual(-1, int(estimator.ac_theta_state_col[ref_pos]))
        self.assertEqual(-1, int(estimator.ac_voltage_state_col[ref_pos]))

        full_x = estimator._expand_state(estimator.initial_state())
        theta, voltage, _, _ = ac._extract_state_vars(full_x[: estimator.calc.ac_size], update_cache=False)
        self.assertAlmostEqual(0.0, theta[ref_pos])
        self.assertAlmostEqual(ref_voltage, voltage[ref_pos])

    def test_nonflat_start_reuses_ac_measurement_seeded_power_flow(self):
        from secore.ac_se import ACStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "ac" / "ieee300.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee300.meas",
            flat_start=False,
        )

        ac_estimator = ACStateEstimator(**kwargs)
        estimator = HybridStateEstimator(**kwargs)

        self.assertIsInstance(estimator._delegate(), ACStateEstimator)
        self.assertFalse(estimator.flat_start)
        self.assertFalse(ac_estimator.flat_start)
        self.assertEqual(ac_estimator.state_labels, estimator.ac_state_labels)
        np.testing.assert_allclose(ac_estimator.initial_state(), estimator.initial_state())
        self.assertFalse(any(meas.name == "pseudo_angle_bus_9025" for meas in estimator.measurements))

    def test_zero_tied_ac_angle_state_rebases_reference_only_once(self):
        from secore.ac_se import ACStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=False,
        )

        ac_estimator = ACStateEstimator(**kwargs)
        estimator = HybridStateEstimator(**kwargs)

        self.assertIsInstance(estimator._delegate(), ACStateEstimator)
        np.testing.assert_allclose(ac_estimator.initial_state(), estimator.initial_state())

    def test_ieee3k_nonflat_seed_matches_load_flow_measurements_after_zero_ties(self):
        from secore.ac_se import ACStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=False,
        )

        ac_estimator = ACStateEstimator(**kwargs)
        estimator = HybridStateEstimator(**kwargs)
        x0 = estimator.initial_state()
        z_est = estimator.evaluate(x0)
        ac_z_est = ac_estimator.evaluate(ac_estimator.initial_state())
        row, meas = next(
            (idx, item)
            for idx, item in enumerate(estimator.active_measurements)
            if item.name == "pt_line_196_2040_c07"
        )

        self.assertAlmostEqual(ac_z_est[row], z_est[row], places=6)
        self.assertAlmostEqual(meas.value, z_est[row], places=6)

    def test_dc_nonflat_start_runs_measurement_seeded_power_flow(self):
        import secore.dc_se as dc_se
        from secore.hybrid_se import HybridStateEstimator

        original_seed = getattr(dc_se.DCStateEstimator, "_run_power_flow_seed", None)
        calls = []

        def fake_seed(network, _params, _e_file):
            nd_1 = network.node_dict[0]
            self.assertAlmostEqual(1.6, float(nd_1.voltage))
            calls.append(True)
            for node in network.nodes:
                if getattr(node, "is_alive", False):
                    node.voltage = 1.23

        dc_se.DCStateEstimator._run_power_flow_seed = staticmethod(fake_seed)
        try:
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=False,
            )
        finally:
            if original_seed is not None:
                dc_se.DCStateEstimator._run_power_flow_seed = staticmethod(original_seed)

        self.assertFalse(estimator.flat_start)
        self.assertTrue(calls)
        self.assertIsInstance(estimator._delegate(), dc_se.DCStateEstimator)

    def test_nonflat_ac_delegate_uses_array_mode_power_flow_seed(self):
        import secore.ac_se as ac_se
        from secore.hybrid_se import HybridStateEstimator

        original_calc = ac_se.ACPowerFlowCalc
        calls = []

        class FakePowerFlowCalc:
            def __init__(self, model, **_kwargs):
                self.model = model
                self.ppc = model if isinstance(model, dict) else getattr(model, "ppc", None)
                self.converged = False
                self.iterations = 0
                self.normF = 0.0
                calls.append(isinstance(model, dict))

            def prepare(self):
                self.testcase.assertAlmostEqual(
                    119.0641444 / 115.0,
                    float(self.ppc["bus"][1, ac_se.BUS_COLS["voltage"]]),
                )

            def run(self):
                self.converged = True
                self.iterations = 1
                self.result = {
                    key: value.copy()
                    for key, value in self.ppc.items()
                    if isinstance(value, np.ndarray)
                }
                self.result["bus"][:, ac_se.BUS_COLS["voltage"]] = 1.13
                self.result["bus"][:, ac_se.BUS_COLS["angle"]] = 0.07
                return 0

        FakePowerFlowCalc.testcase = self
        ac_se.ACPowerFlowCalc = FakePowerFlowCalc
        try:
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee300.e",
                meas_file=ROOT_DIR / "data" / "ac" / "ieee300.meas",
                flat_start=False,
            )
        finally:
            ac_se.ACPowerFlowCalc = original_calc

        self.assertEqual([True], calls)
        self.assertIsInstance(estimator._delegate(), ac_se.ACStateEstimator)
        _theta, voltage = estimator._delegate()._unpack_state(estimator.initial_state())
        np.testing.assert_allclose(voltage[estimator._delegate().voltage_state_pos], 1.13)

    def test_dc_reference_nodes_use_highest_degree_nodes_with_valid_voltage_measurements(self):
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        dc_estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        dc = estimator.calc.dc_calc
        expected_refs = [node.name for node in dc_estimator.references]

        self.assertEqual(expected_refs, [node.name for node in estimator.dc_reference_nodes])
        full_x = estimator._expand_state(estimator.initial_state())
        dc_voltage = full_x[estimator.calc.ac_size : estimator.calc.ac_size + dc.N]
        for name in expected_refs:
            node = estimator.dc_node_by_name[name]
            pos = dc.alive_node_dict[node.idx]
            sub_pos = estimator._dc_sub_estimator.node_pos[node.idx]
            ref_voltage = estimator.dc_node_voltage_measurements[node.idx]
            self.assertEqual(-1, int(estimator.dc_voltage_state_col[sub_pos]))
            self.assertAlmostEqual(ref_voltage, dc_voltage[pos])

    def test_dc_net_30_matches_dc_state_estimator_result(self):
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        kwargs = dict(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )

        dc_estimator = DCStateEstimator(**kwargs)
        hybrid_estimator = HybridStateEstimator(**kwargs)

        dc_result = dc_estimator.estimate(verbose=False)
        hybrid_result = hybrid_estimator.estimate(verbose=False)
        dc_bad_items, dc_normalized = dc_estimator.identify_bad_data(dc_result)
        hybrid_bad_items, hybrid_normalized = hybrid_estimator.identify_bad_data(hybrid_result)

        self.assertIsInstance(hybrid_estimator._dc_sub_estimator, DCStateEstimator)
        self.assertEqual(dc_estimator.state_labels, hybrid_estimator.state_labels)
        self.assertEqual(len(dc_estimator.active_measurements), len(hybrid_estimator.active_measurements))
        self.assertEqual(dc_result.converged, hybrid_result.converged)
        self.assertEqual(dc_result.iterations, hybrid_result.iterations)
        self.assertAlmostEqual(dc_result.objective, hybrid_result.objective, places=14)
        self.assertAlmostEqual(dc_result.max_correction, hybrid_result.max_correction, places=14)
        self.assertAlmostEqual(dc_result.residual_inf, hybrid_result.residual_inf, places=14)
        np.testing.assert_allclose(dc_result.x, hybrid_result.x, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(dc_result.z_est, hybrid_result.z_est, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(dc_result.residual, hybrid_result.residual, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(dc_normalized, hybrid_normalized, rtol=0.0, atol=0.0)
        self.assertEqual(len(dc_bad_items), len(hybrid_bad_items))
        self.assertFalse(hybrid_estimator.ac_state_cols.size)
        self.assertEqual(hybrid_estimator.n_state, hybrid_estimator.dc_state_cols.size)
        self.assertFalse(hybrid_estimator.hybrid_state_cols.size)

    def test_mixed_network_reuses_dc_state_estimator_jacobian_block(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        x = estimator.initial_state()
        hybrid_h = estimator.jacobian_sparse(x)
        dc_x = estimator._dc_sub_state_from_hybrid(x)
        dc_h = estimator._dc_sub_estimator.jacobian_sparse(dc_x, estimator._active_dc_sub_measurements)
        hybrid_dc_h = hybrid_h[estimator._active_dc_hybrid_rows, :][:, estimator._dc_sub_to_hybrid_cols]

        self.assertGreater(estimator._active_dc_hybrid_rows.size, 0)
        self.assertEqual(dc_h.shape, hybrid_dc_h.shape)
        self.assertEqual(dc_h.nnz, hybrid_h[estimator._active_dc_hybrid_rows, :].nnz)
        diff = (hybrid_dc_h - dc_h).tocoo()
        self.assertEqual(0, diff.nnz)

    def test_mixed_network_reuses_ac_state_estimator_jacobian_block(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        x = estimator.initial_state()
        hybrid_h = estimator.jacobian_sparse(x)
        ac_x = estimator._ac_sub_state_from_hybrid(x)
        ac_h = estimator._ac_sub_estimator.jacobian_sparse(ac_x, estimator._active_ac_sub_measurements)
        hybrid_ac_h = hybrid_h[estimator._active_ac_hybrid_rows, :][:, estimator._ac_sub_to_hybrid_cols]

        self.assertGreater(estimator._active_ac_hybrid_rows.size, 0)
        self.assertEqual(ac_h.shape, hybrid_ac_h.shape)
        self.assertEqual(ac_h.nnz, hybrid_h[estimator._active_ac_hybrid_rows, :].nnz)
        diff = (hybrid_ac_h - ac_h).tocoo()
        self.assertEqual(0, diff.nnz)

    def test_active_evaluate_and_jacobian_reuse_cached_measurement_partition(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        x = estimator.initial_state()

        def fail_if_repartitioned(_measurements):
            raise AssertionError("active measurement partition should be cached")

        estimator._partition_measurement_list = fail_if_repartitioned

        z_est = estimator.evaluate(x)
        H = estimator.jacobian_sparse(x)

        self.assertEqual(len(estimator.active_measurements), z_est.size)
        self.assertEqual((len(estimator.active_measurements), estimator.n_state), H.shape)

    def test_active_hybrid_converter_rows_use_vectorized_path(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        x = estimator.initial_state()
        eval_calls = {"count": 0}
        jac_calls = {"count": 0}
        original_eval = estimator._evaluate_hybrid_measurements
        original_jac = estimator._append_hybrid_jacobian_plan

        def counted_eval(*args, **kwargs):
            eval_calls["count"] += 1
            return original_eval(*args, **kwargs)

        def counted_jac(*args, **kwargs):
            jac_calls["count"] += 1
            return original_jac(*args, **kwargs)

        estimator._evaluate_hybrid_measurements = counted_eval
        estimator._append_hybrid_jacobian_plan = counted_jac

        z_est = estimator.evaluate(x)
        H = estimator.jacobian_sparse(x)

        self.assertEqual(len(estimator.active_measurements), z_est.size)
        self.assertEqual((len(estimator.active_measurements), estimator.n_state), H.shape)
        self.assertGreater(estimator.hybrid_meas_rows.size, 0)
        self.assertEqual(1, eval_calls["count"])
        self.assertEqual(1, jac_calls["count"])

    def test_non_active_hybrid_converter_subset_uses_plan_path(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        x = estimator.initial_state()
        self.assertGreater(len(estimator.hybrid_meas), 0)
        subset = list(estimator.hybrid_meas[: min(8, len(estimator.hybrid_meas))])
        self.assertNotEqual(subset, estimator.active_measurements)
        eval_calls = {"count": 0}
        jac_calls = {"count": 0}
        original_eval = estimator._evaluate_hybrid_measurements
        original_jac = estimator._append_hybrid_jacobian_plan

        def counted_eval(*args, **kwargs):
            eval_calls["count"] += 1
            return original_eval(*args, **kwargs)

        def counted_jac(*args, **kwargs):
            jac_calls["count"] += 1
            return original_jac(*args, **kwargs)

        estimator._evaluate_hybrid_measurements = counted_eval
        estimator._append_hybrid_jacobian_plan = counted_jac

        z_est = estimator.evaluate(x, subset)
        H = estimator.jacobian_sparse(x, subset)

        self.assertEqual(len(subset), z_est.size)
        self.assertEqual((len(subset), estimator.n_state), H.shape)
        self.assertEqual(1, eval_calls["count"])
        self.assertEqual(1, jac_calls["count"])

    def test_hybrid_measurement_plan_uses_measurement_plan_table(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator
        from secore.se_array_plan import build_measurement_plan_table as original_builder

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        calls = {"count": 0}

        def counted_builder(*args, **kwargs):
            calls["count"] += 1
            return original_builder(*args, **kwargs)

        previous_builder = getattr(hybrid_se, "build_measurement_plan_table", None)
        hybrid_se.build_measurement_plan_table = counted_builder
        try:
            estimator._build_hybrid_measurement_plan(
                estimator._MeasurementSideBlock(estimator.hybrid_meas_rows, estimator.hybrid_meas)
            )
        finally:
            if previous_builder is None:
                del hybrid_se.build_measurement_plan_table
            else:
                hybrid_se.build_measurement_plan_table = previous_builder

        self.assertGreater(calls["count"], 0)

    def test_hybrid_seed_uses_measurement_plan_table(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator
        from secore.se_array_plan import build_measurement_plan_table as original_builder

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        calls = {"count": 0}

        def counted_builder(*args, **kwargs):
            calls["count"] += 1
            return original_builder(*args, **kwargs)

        previous_builder = getattr(hybrid_se, "build_measurement_plan_table", None)
        hybrid_se.build_measurement_plan_table = counted_builder
        try:
            estimator._hybrid_seed_vector(flat=True)
        finally:
            if previous_builder is None:
                del hybrid_se.build_measurement_plan_table
            else:
                hybrid_se.build_measurement_plan_table = previous_builder

        self.assertGreater(calls["count"], 0)

    def test_converter_pseudo_and_candidate_share_facade(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        calls = []
        original = estimator._hybrid_converter_measurement_specs

        def counted(*args, **kwargs):
            calls.append((args, tuple(sorted(kwargs.items()))))
            return original(*args, **kwargs)

        estimator._hybrid_converter_measurement_specs = counted
        estimator._add_hybrid_pseudo_measurements()
        estimator._observability_pseudo_candidate_measurements()

        self.assertGreaterEqual(len(calls), 2)

    def test_hybrid_estimator_drops_row_fallback_helpers(self):
        from secore.hybrid_se import HybridStateEstimator

        self.assertFalse(hasattr(HybridStateEstimator, "_evaluate_hybrid_measurement"))
        self.assertFalse(hasattr(HybridStateEstimator, "_append_hybrid_jacobian_entries"))

    def test_mixed_network_sub_estimators_reuse_loaded_network_and_measurements(self):
        from unittest.mock import patch

        from secore.ac_se import ACStateEstimator
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        def fail_load(*_args, **_kwargs):
            raise AssertionError("sub-estimator should reuse hybrid-loaded data")

        with patch.object(ACStateEstimator, "_load_network", side_effect=fail_load), patch.object(
            ACStateEstimator,
            "_load_measurements",
            side_effect=fail_load,
        ), patch.object(DCStateEstimator, "_load_network", side_effect=fail_load), patch.object(
            DCStateEstimator,
            "_load_measurements",
            side_effect=fail_load,
        ):
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
                flat_start=True,
            )

        self.assertIsNotNone(estimator._ac_sub_estimator)
        self.assertIsNotNone(estimator._dc_sub_estimator)
        self.assertGreater(len(estimator.ac_meas), 0)
        self.assertGreater(len(estimator.dc_meas), 0)
        self.assertEqual(len(estimator.network.ac.alive_buses), len(estimator._ac_sub_estimator.nodes))

    def test_pure_single_side_delegate_does_not_copy_measurement_objects(self):
        from unittest.mock import patch

        from secore.hybrid_se import HybridStateEstimator

        ac_measurements = HybridStateEstimator._load_measurements(ROOT_DIR / "data" / "ac" / "ieee39.meas")
        with patch.object(HybridStateEstimator, "_load_measurements", return_value=ac_measurements):
            ac_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
                flat_start=True,
            )

        dc_measurements = HybridStateEstimator._load_measurements(ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
        with patch.object(HybridStateEstimator, "_load_measurements", return_value=dc_measurements):
            dc_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )

        self.assertIsNotNone(ac_estimator._delegate())
        self.assertIsNotNone(dc_estimator._delegate())
        self.assertIn(id(ac_measurements[0]), {id(meas) for meas in ac_estimator.measurements})
        self.assertIn(id(dc_measurements[0]), {id(meas) for meas in dc_estimator.measurements})

    def test_pure_single_side_delegate_skips_generic_active_measurement_partition(self):
        from unittest.mock import patch

        from secore.hybrid_se import HybridStateEstimator

        def reject_partition(self, _measurements):
            raise AssertionError("pure single-side delegation should assign rows directly")

        with patch.object(HybridStateEstimator, "_partition_measurement_list", reject_partition):
            ac_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
                flat_start=True,
            )
            dc_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )

        self.assertEqual(len(ac_estimator.active_measurements), ac_estimator.ac_meas_rows.size)
        self.assertEqual(0, ac_estimator.dc_meas_rows.size)
        self.assertEqual(len(dc_estimator.active_measurements), dc_estimator.dc_meas_rows.size)
        self.assertEqual(0, dc_estimator.ac_meas_rows.size)

    def test_pure_single_side_delegate_passes_loaded_measurements_without_side_partition_scan(self):
        from unittest.mock import patch

        from secore.hybrid_se import HybridStateEstimator

        def reject_side_partition(self):
            raise AssertionError("pure single-side delegation should pass loaded measurements directly")

        with patch.object(HybridStateEstimator, "_initial_measurement_sources_by_side", reject_side_partition):
            ac_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
                flat_start=True,
            )
            dc_estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )

        self.assertIsNotNone(ac_estimator._delegate())
        self.assertIsNotNone(dc_estimator._delegate())

    def test_mixed_network_sub_estimators_defer_own_active_measurement_preparation(self):
        from unittest.mock import patch

        from secore.ac_se import ACStateEstimator
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        def fail_refresh(*_args, **_kwargs):
            raise AssertionError("hybrid mixed path should own active measurement preparation")

        with patch.object(ACStateEstimator, "_refresh_active_measurement_indexes", side_effect=fail_refresh), patch.object(
            DCStateEstimator,
            "_refresh_active_measurement_indexes",
            side_effect=fail_refresh,
        ):
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
                flat_start=True,
            )

        self.assertGreater(len(estimator.active_measurements), 0)
        self.assertGreater(len(estimator.ac_meas), 0)
        self.assertGreater(len(estimator.dc_meas), 0)

    def test_mixed_network_includes_ac_sub_estimator_power_balance_rows(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        ac_balance_rows = [
            row
            for row, meas in enumerate(estimator.active_measurements)
            if meas.device_type == "ACPowerBalance"
        ]
        coupled_ac_nodes = estimator._converter_coupled_ac_node_names()
        expected_balance_count = sum(
            2 for node in estimator._ac_sub_estimator.nodes if node.name not in coupled_ac_nodes
        )
        delegated_rows = set(int(row) for row in estimator._active_ac_hybrid_rows)

        self.assertEqual(expected_balance_count, len(ac_balance_rows))
        self.assertTrue(all(row in delegated_rows for row in ac_balance_rows))

    def test_mixed_network_partitions_active_measurements_into_ac_dc_and_hybrid_blocks(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        partitioned = list(estimator.ac_meas) + list(estimator.dc_meas) + list(estimator.hybrid_meas)
        partitioned_ids = {id(meas) for meas in partitioned}
        active_ids = {id(meas) for meas in estimator.active_measurements}
        ac_rows = set(int(row) for row in estimator.ac_meas_rows)
        dc_rows = set(int(row) for row in estimator.dc_meas_rows)
        hybrid_rows = set(int(row) for row in estimator.hybrid_meas_rows)

        self.assertEqual(len(estimator.active_measurements), len(partitioned))
        self.assertEqual(active_ids, partitioned_ids)
        self.assertFalse(ac_rows & dc_rows)
        self.assertFalse(ac_rows & hybrid_rows)
        self.assertFalse(dc_rows & hybrid_rows)
        self.assertTrue(estimator.ac_meas)
        self.assertTrue(estimator.dc_meas)
        self.assertTrue(estimator.hybrid_meas)
        self.assertTrue(all(meas.device_type in estimator._AC_MEASUREMENT_DEVICE_TYPES for meas in estimator.ac_meas))
        self.assertTrue(all(meas.device_type in estimator._DC_MEASUREMENT_DEVICE_TYPES for meas in estimator.dc_meas))
        self.assertTrue(all(meas.device_type in estimator._HYBRID_MEASUREMENT_DEVICE_TYPES for meas in estimator.hybrid_meas))
        self.assertTrue(set(int(row) for row in estimator._active_ac_hybrid_rows).issubset(ac_rows))
        self.assertTrue(set(int(row) for row in estimator._active_dc_hybrid_rows).issubset(dc_rows))
        self.assertFalse(set(int(row) for row in estimator._active_ac_hybrid_rows) & hybrid_rows)
        self.assertFalse(set(int(row) for row in estimator._active_dc_hybrid_rows) & hybrid_rows)

    def test_measurement_partition_uses_device_ownership_not_device_name_text(self):
        from secore.hybrid_se import HybridStateEstimator, Measurement

        dc_named_like_ac = Measurement(1, "m1", "DCNode", "ac_named_dc_bus", "V", 1.0, True, 1.0)
        ac_named_like_dc = Measurement(2, "m2", "ACNode", "dc_named_ac_bus", "V", 1.0, True, 1.0)
        hybrid_named_like_dc = Measurement(3, "m3", "DCACConverter", "dc_named_converter", "P_AC", 1.0, True, 1.0)

        self.assertEqual("dc", HybridStateEstimator._measurement_side(dc_named_like_ac))
        self.assertEqual("ac", HybridStateEstimator._measurement_side(ac_named_like_dc))
        self.assertEqual("hybrid", HybridStateEstimator._measurement_side(hybrid_named_like_dc))

    def test_measurement_pu_conversion_delegates_ac_dc_and_keeps_hybrid_local(self):
        from secore.hybrid_se import HybridStateEstimator, Measurement

        calls = []

        class FakeSubEstimator:
            def __init__(self, side):
                self.side = side
                self.measurements = []

            def _convert_measurements_to_pu(self):
                calls.append((self.side, tuple(meas.device_type for meas in self.measurements)))
                for meas in self.measurements:
                    meas.value += 10.0 if self.side == "ac" else 20.0

            def _refresh_measurement_summary_cache(self):
                raise AssertionError("unit conversion should not refresh sub measurement summary")

        estimator = HybridStateEstimator.__new__(HybridStateEstimator)
        estimator._ac_sub_estimator = FakeSubEstimator("ac")
        estimator._dc_sub_estimator = FakeSubEstimator("dc")
        estimator._measurements_normalized = False
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.u_scale = 1.0
        estimator.i_scale = 1.0
        estimator.dcac_by_name = {
            "conv": SimpleNamespace(name="conv", dc_node=1, ac_node=2),
        }
        estimator.acac_by_name = {}
        estimator.network = SimpleNamespace(
            ac=SimpleNamespace(node_dict={2: SimpleNamespace(vbase=10.0)}),
            dc=SimpleNamespace(node_dict={1: SimpleNamespace(vbase=1.0)}),
        )
        estimator.measurements = [
            Measurement(1, "ac_v", "ACNode", "ac_bus", "V", 1.0, True, 1.0),
            Measurement(2, "dc_v", "DCNode", "dc_bus", "V", 1.0, True, 2.0),
            Measurement(3, "conv_p", "DCACConverter", "conv", "P_AC", 1.0, True, 50.0),
        ]

        estimator._convert_measurements_to_pu()

        self.assertEqual([("ac", ("ACNode",)), ("dc", ("DCNode",))], calls)
        self.assertAlmostEqual(estimator.measurements[0].value, 11.0)
        self.assertAlmostEqual(estimator.measurements[1].value, 22.0)
        self.assertAlmostEqual(estimator.measurements[2].value, 0.5)
        self.assertFalse(hasattr(HybridStateEstimator, "_add_dc_zero_branch_constraint_measurements"))

    def test_measurement_pu_conversion_updates_master_table_values(self):
        from model.meas_model import MeasurementList, measurement_table_from_measurements
        from secore.hybrid_se import HybridStateEstimator, Measurement

        class FakeSubEstimator:
            def __init__(self, delta):
                self.delta = delta
                self.measurements = []

            def _convert_measurements_to_pu(self):
                for meas in self.measurements:
                    meas.value += self.delta

        estimator = HybridStateEstimator.__new__(HybridStateEstimator)
        estimator._ac_sub_estimator = FakeSubEstimator(10.0)
        estimator._dc_sub_estimator = FakeSubEstimator(20.0)
        estimator._measurements_normalized = False
        estimator._sub_measurements_converted_by_side = {"ac": False, "dc": False}
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.u_scale = 1.0
        estimator.i_scale = 1.0
        estimator.dcac_by_name = {
            "conv": SimpleNamespace(name="conv", dc_node=1, ac_node=2),
        }
        estimator.acac_by_name = {}
        estimator.network = SimpleNamespace(
            ac=SimpleNamespace(node_dict={2: SimpleNamespace(vbase=10.0)}),
            dc=SimpleNamespace(node_dict={1: SimpleNamespace(vbase=1.0)}),
        )
        measurements = [
            Measurement(1, "ac_v", "ACNode", "ac_bus", "V", 1.0, True, 1.0),
            Measurement(2, "dc_v", "DCNode", "dc_bus", "V", 1.0, True, 2.0),
            Measurement(3, "conv_p", "DCACConverter", "conv", "P_AC", 1.0, True, 50.0),
        ]
        estimator.measurements = MeasurementList(
            measurements,
            measurement_table_from_measurements(measurements),
        )

        estimator._convert_measurements_to_pu()

        np.testing.assert_allclose(estimator.measurements.table.value, np.array([11.0, 22.0, 0.5]))

    def test_measurement_pu_conversion_skips_sides_already_converted_by_sub_estimators(self):
        from secore.hybrid_se import HybridStateEstimator, Measurement

        class FakeSubEstimator:
            def _convert_measurements_to_pu(self):
                raise AssertionError("already converted side measurements should not be converted again")

        estimator = HybridStateEstimator.__new__(HybridStateEstimator)
        estimator._ac_sub_estimator = FakeSubEstimator()
        estimator._dc_sub_estimator = FakeSubEstimator()
        estimator._measurements_normalized = False
        estimator._sub_measurements_converted_by_side = {"ac": True, "dc": True}
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.u_scale = 1.0
        estimator.i_scale = 1.0
        estimator.dcac_by_name = {
            "conv": SimpleNamespace(name="conv", dc_node=1, ac_node=2),
        }
        estimator.acac_by_name = {}
        estimator.network = SimpleNamespace(
            ac=SimpleNamespace(node_dict={2: SimpleNamespace(vbase=10.0)}),
            dc=SimpleNamespace(node_dict={1: SimpleNamespace(vbase=1.0)}),
        )
        estimator.measurements = [
            Measurement(1, "ac_v", "ACNode", "ac_bus", "V", 1.0, True, 1.0),
            Measurement(2, "dc_v", "DCNode", "dc_bus", "V", 1.0, True, 2.0),
            Measurement(3, "conv_p", "DCACConverter", "conv", "P_AC", 1.0, True, 50.0),
        ]

        estimator._convert_measurements_to_pu()

        self.assertAlmostEqual(estimator.measurements[0].value, 1.0)
        self.assertAlmostEqual(estimator.measurements[1].value, 2.0)
        self.assertAlmostEqual(estimator.measurements[2].value, 0.5)

    def test_mixed_network_partitions_state_variables_into_ac_dc_and_hybrid_blocks(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        ac_cols = set(int(col) for col in estimator.ac_state_cols)
        dc_cols = set(int(col) for col in estimator.dc_state_cols)
        hybrid_cols = set(int(col) for col in estimator.hybrid_state_cols)
        all_cols = ac_cols | dc_cols | hybrid_cols

        self.assertEqual(set(range(estimator.n_state)), all_cols)
        self.assertFalse(ac_cols & dc_cols)
        self.assertFalse(ac_cols & hybrid_cols)
        self.assertFalse(dc_cols & hybrid_cols)
        self.assertTrue(ac_cols)
        self.assertTrue(dc_cols)
        self.assertTrue(hybrid_cols)
        self.assertEqual([estimator.state_labels[col] for col in estimator.ac_state_cols], estimator.ac_vars)
        self.assertEqual([estimator.state_labels[col] for col in estimator.dc_state_cols], estimator.dc_vars)
        self.assertEqual([estimator.state_labels[col] for col in estimator.hybrid_state_cols], estimator.hybrid_vars)
        self.assertEqual(slice(0, estimator.ac_state_cols.size), estimator.ac_state_slice)
        self.assertEqual(
            slice(estimator.ac_state_cols.size, estimator.ac_state_cols.size + estimator.dc_state_cols.size),
            estimator.dc_state_slice,
        )
        self.assertEqual(
            slice(estimator.ac_state_cols.size + estimator.dc_state_cols.size, estimator.n_state),
            estimator.hybrid_state_slice,
        )
        self.assertTrue(all(estimator.state_sides[int(col)] == "ac" for col in estimator.ac_state_cols))
        self.assertTrue(all(estimator.state_sides[int(col)] == "dc" for col in estimator.dc_state_cols))
        self.assertTrue(all(estimator.state_sides[int(col)] == "hybrid" for col in estimator.hybrid_state_cols))

    def test_state_partition_uses_layout_side_metadata_not_label_text(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        expected_ac_cols = estimator.ac_state_cols.copy()
        expected_dc_cols = estimator.dc_state_cols.copy()
        expected_hybrid_cols = estimator.hybrid_state_cols.copy()
        estimator.state_labels = [f"ambiguous_state_{idx}" for idx in range(estimator.n_state)]
        estimator._partition_state_variables()

        np.testing.assert_array_equal(expected_ac_cols, estimator.ac_state_cols)
        np.testing.assert_array_equal(expected_dc_cols, estimator.dc_state_cols)
        np.testing.assert_array_equal(expected_hybrid_cols, estimator.hybrid_state_cols)

    def test_dc_side_measurements_are_delegated_by_device_ownership(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )

        delegated_rows = set(int(row) for row in estimator._active_dc_hybrid_rows)
        hybrid_only_rows = [
            row
            for row, meas in enumerate(estimator.active_measurements)
            if meas.device_type == "DCNode"
            and meas.device_name == "wt01_line_dc"
            and meas.meas_type == "V"
        ]

        self.assertEqual(1, len(hybrid_only_rows))
        self.assertIn(hybrid_only_rows[0], delegated_rows)

    def test_targeted_zero_current_pseudo_uses_to_side_when_from_side_exists(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
        )
        device_name = next(iter(estimator.ac_zero_branch_by_name))
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = {
            ("ACZeroBranch", device_name, "P_FROM"),
            ("ACZeroBranch", device_name, "Q_FROM"),
        }
        existing_names = set()

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            f"AC_I_RE:{device_name}",
            existing_keys,
            existing_names,
            2,
        )

        self.assertEqual(2, added)
        self.assertIn(("ACZeroBranch", device_name, "P_TO"), existing_keys)
        self.assertIn(("ACZeroBranch", device_name, "Q_TO"), existing_keys)

    def test_targeted_node_voltage_states_add_pseudo_measurements(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
        )
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = set()
        existing_names = set()

        next_idx, ac_added = estimator._append_targeted_observability_pseudo(
            next_idx,
            "AC_V:wt02_src",
            existing_keys,
            existing_names,
            1,
        )
        _, dc_added = estimator._append_targeted_observability_pseudo(
            next_idx,
            "DC_V:wt01_dc_sw",
            existing_keys,
            existing_names,
            1,
        )

        self.assertEqual(1, ac_added)
        self.assertEqual(1, dc_added)
        self.assertIn(("ACNode", "wt02_src", "V"), existing_keys)
        self.assertIn(("DCNode", "wt01_dc_sw", "V"), existing_keys)

    def test_sparse_jacobian_matches_dense_jacobian(self):
        from scipy.sparse import issparse
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )

        x = estimator.initial_state()
        dense = estimator.jacobian(x)
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_sparse_jacobian_vectorizes_repeated_hybrid_device_rows(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        generic_keys = {
            (meas.device_type, meas.meas_type)
            for row, meas in enumerate(estimator.active_measurements)
            if not estimator._jacobian_static_skip[row]
            and not estimator._active_ac_delegated_row_mask[row]
            and not estimator._active_dc_delegated_row_mask[row]
        }

        for key in (
            ("ACLoad", "P_LOAD"),
            ("ACLoad", "Q_LOAD"),
            ("ACLoad", "I_LOAD"),
            ("ACSwitch", "P_FROM"),
            ("ACSwitch", "Q_FROM"),
            ("ACSwitch", "I_FROM"),
            ("ACSwitch", "P_TO"),
            ("ACSwitch", "Q_TO"),
            ("ACSwitch", "I_TO"),
            ("DCGenerator", "I_GEN"),
        ):
            self.assertNotIn(key, generic_keys)

        x = estimator.initial_state()
        np.testing.assert_allclose(estimator.jacobian_sparse(x).toarray(), estimator.jacobian(x), atol=1e-10)

    def test_active_evaluation_uses_state_arrays_without_model_writeback(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        x = estimator.initial_state()
        if estimator.voltage_cols.size:
            x = x.copy()
            x[estimator.voltage_cols[0]] *= 1.001
        if estimator.dcac_p_ac_state_col.size:
            x = x.copy()
            x[estimator.dcac_p_ac_state_col[0]] += 1e-4

        expected = estimator.evaluate(x, list(estimator.active_measurements))

        def fail_writeback(_x):
            raise AssertionError("active evaluation should use array formulas without model writeback")

        estimator._write_state = fail_writeback
        actual = estimator.evaluate(x)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_active_sparse_jacobian_uses_state_arrays_without_model_writeback(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        x = estimator.initial_state()
        expected = estimator.jacobian_sparse(x).toarray()

        def fail_writeback(_x):
            raise AssertionError("active sparse Jacobian should use array formulas without model writeback")

        estimator._write_state = fail_writeback
        actual = estimator.jacobian_sparse(x).toarray()

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_mapped_state_expansion_uses_cached_mapping_arrays(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )
        x = estimator.initial_state()
        expected = estimator._expand_state_mapped_only(x)

        class NonComparableMapping:
            def __ge__(self, _other):
                raise AssertionError("mapped-state expansion should reuse cached mapping arrays")

            def __array__(self, *_args, **_kwargs):
                raise AssertionError("mapped-state expansion should reuse cached mapping arrays")

        estimator.full_col_for_state = NonComparableMapping()
        actual = estimator._expand_state_mapped_only(x)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_initialization_uses_vectorized_ac_branch_stamps(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        original_stamp = hybrid_se.matpower_branch_stamp

        def fail_scalar_stamp(*args, **kwargs):
            raise AssertionError("hybrid state estimator should use vectorized AC branch stamps")

        hybrid_se.matpower_branch_stamp = fail_scalar_stamp
        try:
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            )
        finally:
            hybrid_se.matpower_branch_stamp = original_stamp

        self.assertTrue(estimator.ac_branch_stamp_by_name)

    def test_estimate_reuses_converged_iteration_sparse_jacobian(self):
        from scipy.sparse import issparse
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            max_iter=20,
        )

        original_jacobian = estimator.jacobian_sparse
        call_count = 0

        def counted_jacobian(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_jacobian(*args, **kwargs)

        estimator.jacobian_sparse = counted_jacobian
        result = estimator.estimate()

        self.assertTrue(result.converged)
        self.assertTrue(issparse(result.H))
        self.assertEqual(result.iterations, call_count)

    def test_main_runs_observability_before_estimation_and_does_not_repeat_it(self):
        import contextlib
        import io
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        events = []
        original_observability = HybridStateEstimator.observability_analysis
        original_estimate = HybridStateEstimator.estimate
        test_case = self

        def counted_observability(self, *args, **kwargs):
            events.append("observability")
            return original_observability(self, *args, **kwargs)

        def counted_estimate(self, *args, **kwargs):
            events.append("estimate")
            test_case.assertIsNotNone(kwargs.get("observability"))
            observability_calls = events.count("observability")
            result = original_estimate(self, *args, **kwargs)
            test_case.assertEqual(observability_calls, events.count("observability"))
            return result

        output = io.StringIO()
        HybridStateEstimator.observability_analysis = counted_observability
        HybridStateEstimator.estimate = counted_estimate
        try:
            with contextlib.redirect_stdout(output):
                rc = hybrid_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "hybrid" / "qinling.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "hybrid" / "qinling.meas"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            HybridStateEstimator.observability_analysis = original_observability
            HybridStateEstimator.estimate = original_estimate

        self.assertEqual(0, rc)
        self.assertEqual(["observability", "estimate"], events[-2:])
        self.assertEqual(1, output.getvalue().count("Observability:"))
        self.assertLess(output.getvalue().index("Observability:"), output.getvalue().index("State estimation:"))

    def test_main_returns_failure_when_estimation_does_not_converge(self):
        import contextlib
        import io
        import secore.hybrid_se as hybrid_se

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = hybrid_se.main(
                [
                    "--case",
                    str(ROOT_DIR / "data" / "hybrid" / "qinling.e"),
                    "--meas",
                    str(ROOT_DIR / "data" / "hybrid" / "qinling.meas"),
                    "--flat-start",
                    "--quiet",
                    "--max-iter",
                    "0",
                ]
            )

        self.assertEqual(1, rc)
        self.assertIn("converged=False", output.getvalue())

    def test_mixed_estimate_reuses_fixed_pattern_normal_equation_solver(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
            max_iter=5,
        )

        solver_instances = []

        class SpyNormalSolver:
            def __init__(self, assume_fixed_pattern=False):
                self.assume_fixed_pattern = bool(assume_fixed_pattern)
                self.solve_calls = 0
                solver_instances.append(self)

            def solve(self, gain, rhs, return_factor_diag=True):
                self.solve_calls += 1
                return np.zeros_like(rhs), np.ones_like(rhs)

        self.assertTrue(hasattr(hybrid_se, "NormalEquationSolver"))
        original_solver = hybrid_se.NormalEquationSolver
        hybrid_se.NormalEquationSolver = SpyNormalSolver
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.NormalEquationSolver = original_solver

        self.assertTrue(result.converged)
        self.assertEqual(1, len(solver_instances))
        self.assertTrue(solver_instances[0].assume_fixed_pattern)
        self.assertEqual(1, solver_instances[0].solve_calls)

    def test_mixed_estimate_reuses_normal_equation_structural_pattern(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            flat_start=True,
            max_iter=5,
        )

        original_builder = hybrid_se.build_normal_equations
        builder_calls = []

        def counted_builder(H, residual, weight, **kwargs):
            builder_calls.append(kwargs)
            return original_builder(H, residual, weight, **kwargs)

        class ZeroStepSolver:
            def __init__(self, assume_fixed_pattern=False):
                pass

            def solve(self, gain, rhs, return_factor_diag=True):
                return np.zeros_like(rhs), np.ones_like(rhs)

        self.assertTrue(hasattr(hybrid_se, "NormalEquationSolver"))
        original_solver = hybrid_se.NormalEquationSolver
        hybrid_se.build_normal_equations = counted_builder
        hybrid_se.NormalEquationSolver = ZeroStepSolver
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.build_normal_equations = original_builder
            hybrid_se.NormalEquationSolver = original_solver

        self.assertTrue(result.converged)
        self.assertTrue(builder_calls)
        self.assertIsNotNone(builder_calls[0].get("normal_pattern"))
        self.assertTrue(builder_calls[0].get("assume_normal_pattern_matches"))
        self.assertIn("weighted_residual", builder_calls[0])

    def test_estimate_rejects_nonfinite_line_search_candidates(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
            max_iter=20,
        )
        original_evaluate = estimator.evaluate
        evaluate_count = 0

        def nonfinite_candidate_evaluate(x, measurements=None):
            nonlocal evaluate_count
            evaluate_count += 1
            if evaluate_count == 1:
                return original_evaluate(x, measurements)
            measurements = estimator.active_measurements if measurements is None else list(measurements)
            return np.full(len(measurements), np.nan)

        class FiniteStepSolver:
            def __init__(self, assume_fixed_pattern=False):
                pass

            def solve(self, gain, rhs, return_factor_diag=True):
                return np.full(estimator.n_state, 1e-3), np.ones(estimator.n_state)

        original_solver = hybrid_se.NormalEquationSolver
        estimator.evaluate = nonfinite_candidate_evaluate
        hybrid_se.NormalEquationSolver = FiniteStepSolver
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.NormalEquationSolver = original_solver

        self.assertFalse(result.converged)
        self.assertLessEqual(evaluate_count, 10)
        self.assertTrue(np.isfinite(result.residual_inf))

    def test_estimate_uses_precomputed_observability_without_reanalysis(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )
        observability = estimator.observability_analysis()

        original = hybrid_se.observability_rank_details

        def counted_rank_details(*args, **kwargs):
            raise AssertionError("estimate should reuse precomputed observability and not re-run rank analysis")

        hybrid_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate(observability=observability)
        finally:
            hybrid_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertIs(observability, result.observability)

    def test_estimate_reuses_observability_normal_pattern_cache(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )
        observability = estimator.observability_analysis()

        original_builder = hybrid_se.build_normal_equations
        normal_pattern_seen = False

        def counted_builder(H, residual, weight, **kwargs):
            nonlocal normal_pattern_seen
            normal_pattern_seen = kwargs.get("normal_pattern") is not None
            return original_builder(H, residual, weight, **kwargs)

        hybrid_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate(observability=observability)
        finally:
            hybrid_se.build_normal_equations = original_builder

        self.assertTrue(result.converged)
        self.assertTrue(normal_pattern_seen)

    def test_estimate_passes_file_weights_to_normal_equation_builder(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        self.assertTrue(hasattr(hybrid_se, "build_normal_equations"))
        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )

        original = hybrid_se.build_normal_equations
        non_unit_weight_seen = False

        def counted_builder(H, residual, weight, **kwargs):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight, **kwargs)

        hybrid_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.build_normal_equations = original

        self.assertTrue(result.converged)
        self.assertTrue(non_unit_weight_seen)

    def test_estimate_uses_cholesky_solver_when_available(self):
        import secore.se_math as se_math
        from secore.hybrid_se import HybridStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

            original_solve = np.linalg.solve
            call_count = 0

            def counted_solve(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return original_solve(*args, **kwargs)

            np.linalg.solve = counted_solve
            try:
                result = estimator.estimate()
            finally:
                np.linalg.solve = original_solve

        self.assertTrue(result.converged)
        self.assertEqual(0, call_count)

    def test_observability_uses_cholesky_fast_path_when_observable(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

            original_svd = np.linalg.svd
            original_eigvalsh = np.linalg.eigvalsh
            calls = []

            def counted_svd(*args, **kwargs):
                calls.append(("svd", dict(kwargs)))
                return original_svd(*args, **kwargs)

            def counted_eigvalsh(*args, **kwargs):
                calls.append(("eigvalsh", dict(kwargs)))
                return original_eigvalsh(*args, **kwargs)

            np.linalg.svd = counted_svd
            np.linalg.eigvalsh = counted_eigvalsh
            try:
                result = estimator.observability_analysis()
            finally:
                np.linalg.svd = original_svd
                np.linalg.eigvalsh = original_eigvalsh

        self.assertTrue(result.observable)
        self.assertEqual(0, len(calls))
        self.assertEqual(0, result.singular_values.size)

    def test_jacobian_reuses_ac_branch_derivatives_per_terminal(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        branch_name = next(iter(estimator.ac_branch_by_name))
        wanted_types = {"P_FROM", "Q_FROM", "I_FROM", "P_TO", "Q_TO", "I_TO"}
        measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACBranch"
            and meas.device_name == branch_name
            and meas.meas_type in wanted_types
        ]
        self.assertEqual(wanted_types, {meas.meas_type for meas in measurements})

        original_power = estimator._ac_branch_power_derivatives
        original_current = estimator._ac_branch_current_derivatives
        power_calls = 0
        current_calls = 0

        def counted_power(*args, **kwargs):
            nonlocal power_calls
            power_calls += 1
            return original_power(*args, **kwargs)

        def counted_current(*args, **kwargs):
            nonlocal current_calls
            current_calls += 1
            return original_current(*args, **kwargs)

        estimator._ac_branch_power_derivatives = counted_power
        estimator._ac_branch_current_derivatives = counted_current
        estimator.jacobian(estimator.initial_state(), measurements)

        self.assertEqual(0, power_calls)
        self.assertEqual(0, current_calls)

    def test_ac_generator_sparse_jacobian_batches_row_appends(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "hybrid" / "qinling.meas")
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        original_append = estimator._append_sparse_rows_unchecked
        call_count = 0

        def counted_append(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_append(*args, **kwargs)

        estimator._append_sparse_rows_unchecked = counted_append
        estimator.jacobian_sparse(estimator.initial_state())

        self.assertEqual(0, call_count)

    def test_hybrid_analytic_jacobian_matches_finite_difference_sample(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )

        sampled_measurements = []
        wanted = {
            ("ACBranch", "P_FROM"),
            ("ACBranch", "I_TO"),
            ("ACGenerator", "I_GEN"),
            ("DCBranch", "P_FROM"),
            ("DCGenerator", "I_GEN"),
            ("DCDCConverter", "I_TO"),
            ("DCACConverter", "I_AC"),
        }
        for meas in estimator.active_measurements:
            key = (meas.device_type, meas.meas_type)
            if key in wanted:
                sampled_measurements.append(meas)
                wanted.remove(key)
            if not wanted:
                break
        self.assertFalse(wanted)

        x = estimator.initial_state()
        H = estimator.jacobian(x, sampled_measurements)
        H_num = np.zeros_like(H)
        for col in range(estimator.n_state):
            step = 1e-6 * max(1.0, abs(x[col]))
            xp = x.copy()
            xm = x.copy()
            xp[col] += step
            xm[col] -= step
            if estimator.voltage_cols.size:
                xp[estimator.voltage_cols] = np.maximum(xp[estimator.voltage_cols], 0.05)
                xm[estimator.voltage_cols] = np.maximum(xm[estimator.voltage_cols], 0.05)
            H_num[:, col] = (estimator.evaluate(xp, sampled_measurements) - estimator.evaluate(xm, sampled_measurements)) / (
                2.0 * step
            )

        self.assertLess(float(np.max(np.abs(H - H_num))), 1e-5)

    def test_adds_low_weight_pseudo_power_measurements_for_unmetered_generators_and_loads(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "voltage_only.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 vm_ac_bus ACNode ac_bus V 1.0 1 380",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        pseudo = [meas for meas in estimator.active_measurements if meas.name.startswith("pseudo_")]
        pseudo_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in pseudo}

        self.assertIn(("ACGenerator", "diesel_300kw", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "diesel_300kw", "Q_GEN"), pseudo_keys)
        self.assertIn(("DCGenerator", "dc_bus_vctrl", "P_GEN"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        selected_ac_loads = {
            load.name
            for load in estimator.ac_load_by_name.values()
        }
        pseudo_ac_loads = [
            meas
            for meas in pseudo
            if meas.device_type == "ACLoad"
        ]
        self.assertEqual(
            selected_ac_loads,
            {meas.device_name for meas in pseudo_ac_loads},
        )
        self.assertEqual(3 * len(selected_ac_loads), len(pseudo_ac_loads))
        selected_load_name = sorted(selected_ac_loads)[0]
        ac_load_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "ACLoad"
            and meas.device_name == selected_load_name
            and meas.meas_type == "P_LOAD"
        )
        dc_gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "DCGenerator"
            and meas.device_name == "dc_bus_vctrl"
            and meas.meas_type == "P_GEN"
        )
        self.assertAlmostEqual(ac_load_p.value, estimator.ac_load_by_name[selected_load_name].p)
        self.assertAlmostEqual(dc_gen_p.value, estimator.dc_generator_by_name["dc_bus_vctrl"].p)

    def test_hybrid_observability_pseudo_preparation_does_not_emit_runtime_warnings(self):
        import warnings

        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "voltage_only.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 vm_ac_bus ACNode ac_bus V 1.0 1 380",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                HybridStateEstimator(
                    e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                    meas_file=meas_file,
                )

    def test_pseudo_measurements_are_device_level_for_hybrid_converters(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "device_level.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_dcdc DCDCConverter pv01_dcdc V_FROM 1.0 1 300",
                        "# 2 v_dcac DCACConverter wt01_rect V_AC 1.0 1 300",
                        "# 3 bad_dcdc DCDCConverter pv02_dcdc P_FROM 1.0 0 0",
                        "# 4 bad_dcac DCACConverter wt02_rect P_AC 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }
        regular_pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_") and not meas.name.startswith("pseudo_obs_")
        }

        self.assertNotIn(("DCDCConverter", "pv01_dcdc", "P_FROM"), regular_pseudo_keys)
        self.assertNotIn(("DCDCConverter", "pv01_dcdc", "P_TO"), regular_pseudo_keys)
        self.assertNotIn(("DCACConverter", "wt01_rect", "P_AC"), regular_pseudo_keys)
        self.assertNotIn(("DCACConverter", "wt01_rect", "P_DC"), regular_pseudo_keys)
        self.assertIn(("DCDCConverter", "pv02_dcdc", "P_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "pv02_dcdc", "P_TO"), pseudo_keys)
        self.assertIn(("DCDCConverter", "pv02_dcdc", "V_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "pv02_dcdc", "V_TO"), pseudo_keys)
        self.assertIn(("DCACConverter", "wt02_rect", "P_DC"), pseudo_keys)
        self.assertIn(("DCACConverter", "wt02_rect", "P_AC"), pseudo_keys)
        self.assertIn(("DCACConverter", "wt02_rect", "Q_AC"), pseudo_keys)
        self.assertIn(("DCACConverter", "wt02_rect", "V_DC"), pseudo_keys)
        self.assertIn(("DCACConverter", "wt02_rect", "V_AC"), pseudo_keys)

    def test_adds_low_weight_pseudo_measurements_for_unmetered_hybrid_nodes_and_switches(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "invalid_hybrid_topology_devices.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_ac_ok ACNode wt01_src V 1.0 1 300",
                        "# 2 v_ac_bad ACNode wt02_src V 1.0 0 300",
                        "# 3 v_dc_ok DCNode dc_bus_720v V 1.0 1 720",
                        "# 4 v_dc_bad DCNode wt01_dc_sw V 1.0 0 720",
                        "# 5 p_ac_sw_bad ACSwitch sw_diesel_ac P_FROM 1.0 0 0",
                        "# 6 p_dc_sw_bad DCSwitch sw_wt01_dc P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertNotIn(("ACNode", "wt01_src", "V"), pseudo_keys)
        self.assertNotIn(("ACNode", "wt02_src", "V"), pseudo_keys)
        self.assertNotIn(("DCNode", "dc_bus_720v", "V"), pseudo_keys)
        self.assertNotIn(("DCNode", "wt01_dc_sw", "V"), pseudo_keys)
        self.assertFalse(
            any(
                device_type in ("ACNode", "DCNode") and meas_type == "V"
                for device_type, _name, meas_type in pseudo_keys
            )
        )
        for meas_type in ("P_FROM", "Q_FROM", "V_FROM", "I_FROM"):
            self.assertIn(("ACBreak", "sw_diesel_ac", meas_type), pseudo_keys)
        for meas_type in ("P_FROM", "V_FROM", "I_FROM"):
            self.assertIn(("DCBreak", "sw_wt01_dc", meas_type), pseudo_keys)

    def test_adds_low_weight_pseudo_measurements_for_unmetered_hybrid_zero_branches(self):
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "invalid_zero_branch_devices.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_nd_1 DCNode nd_1 V 1.0 1 160",
                        "# 2 p_zbr_bad DCZeroBranch zbr_1_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        for meas_type in ("P_FROM", "V_FROM", "I_FROM"):
            self.assertIn(("DCZeroBranch", "zbr_1_2", meas_type), pseudo_keys)

    def test_qinling_hybrid_state_estimation_with_converter_measurements(self):
        from secore.hybrid_se import HybridStateEstimator

        meas_file = ROOT_DIR / "data" / "hybrid" / "qinling.meas"
        self.assertTrue(meas_file.exists())

        with tempfile.TemporaryDirectory() as tmp_dir:
            all_valid_meas = self._all_valid_measurement_file(tmp_dir, meas_file)
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
                meas_file=all_valid_meas,
                max_iter=20,
            )

        device_types = {meas.device_type for meas in estimator.active_measurements}
        meas_types = {meas.meas_type for meas in estimator.active_measurements}
        for device_type in (
            "ACNode",
            "DCNode",
            "ACBranch",
            "DCBranch",
            "ACBreak",
            "DCBreak",
            "ACGenerator",
            "DCGenerator",
            "ACLoad",
            "DCDCConverter",
            "DCACConverter",
        ):
            self.assertIn(device_type, device_types)
        for meas_type in (
            "V",
            "P_FROM",
            "I_FROM",
            "P_TO",
            "I_TO",
            "P_GEN",
            "I_GEN",
            "P_LOAD",
            "I_LOAD",
            "P_AC",
            "P_DC",
            "I_AC",
            "I_DC",
        ):
            self.assertIn(meas_type, meas_types)

        result = estimator.estimate()
        self.assertTrue(result.converged)
        self.assertTrue(result.observability.observable)
        self.assertLess(result.residual_inf, 1e-6)

    def test_pure_ac_ieee3k_allows_zero_tied_equal_slack_nodes(self):
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
        )

        self.assertGreater(estimator.n_state, 0)
        self.assertGreater(len(estimator.active_measurements), estimator.n_state)

    def test_pure_dc_dc_net_3000_adds_pseudo_measurements_for_unmetered_zero_branch_current_states(self):
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "dc" / "dc_net_3000.meas")
            dc_estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_3000.e",
                meas_file=meas_file,
                flat_start=True,
            )
            estimator = HybridStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_3000.e",
                meas_file=meas_file,
                flat_start=True,
            )

        self.assertEqual(dc_estimator.state_labels, estimator.state_labels)
        self.assertEqual(
            [node.name for node in dc_estimator.references],
            [node.name for node in estimator.dc_reference_nodes],
        )
        self.assertEqual(
            int(np.count_nonzero(dc_estimator.voltage_col < 0)),
            int(np.count_nonzero(estimator.dc_voltage_state_col < 0)),
        )
        self.assertTrue(any(label.startswith("I_ZERO:zbr_") for label in estimator.state_labels))

        dc_result = dc_estimator.estimate(verbose=False)
        result = estimator.estimate(verbose=False)
        self.assertEqual(dc_result.observability.observable, result.observability.observable)
        self.assertEqual(dc_result.converged, result.converged)
        self.assertEqual(dc_result.iterations, result.iterations)
        self.assertAlmostEqual(dc_result.objective, result.objective, places=12)
        self.assertAlmostEqual(dc_result.residual_inf, result.residual_inf, places=12)

    def test_hybrid_se_adds_dc_ideal_branch_voltage_constraints(self):
        from secore.dc_se import DCStateEstimator
        from secore.hybrid_se import HybridStateEstimator

        dc_estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )

        dc_constraint_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in dc_estimator.active_measurements
            if meas.device_type in ("DCZeroBranchConstraint", "DCBreakConstraint")
        }
        hybrid_constraint_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.device_type in ("DCZeroBranchConstraint", "DCBreakConstraint")
        }
        self.assertEqual(dc_constraint_keys, hybrid_constraint_keys)

    def test_cli_does_not_build_seresult_without_output_file(self):
        import contextlib
        import io
        import secore.hybrid_se as hybrid_se

        original_build = hybrid_se.HybridStateEstimator.build_se_result

        def reject_build(*_args, **_kwargs):
            raise AssertionError("SEResult details should be built only when --se-result is requested")

        hybrid_se.HybridStateEstimator.build_se_result = reject_build
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = hybrid_se.main(
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
            hybrid_se.HybridStateEstimator.build_se_result = original_build

        self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
