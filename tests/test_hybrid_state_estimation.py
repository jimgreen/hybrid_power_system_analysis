import unittest
from pathlib import Path
import tempfile

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

    def test_ac_reference_angle_rebases_nonflat_state_without_angle_pseudos(self):
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
        self.assertNotAlmostEqual(meas.value, z_est[row], places=6)

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
            1
            for meas in estimator._ac_sub_estimator.active_measurements
            if meas.device_type == "ACPowerBalance" and meas.device_name not in coupled_ac_nodes
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

    def test_dc_sub_delegation_excludes_hybrid_only_voltage_states(self):
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
        self.assertNotIn(hybrid_only_rows[0], delegated_rows)

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

    def test_targeted_node_voltage_states_do_not_add_pseudo_measurements(self):
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

        self.assertEqual(0, ac_added)
        self.assertEqual(0, dc_added)
        self.assertNotIn(("ACNode", "wt02_src", "V"), existing_keys)
        self.assertNotIn(("DCNode", "wt01_dc_sw", "V"), existing_keys)

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

        original_solver = hybrid_se.solve_normal_equations_with_factor

        def finite_test_step(gain, rhs):
            return np.full(estimator.n_state, 1e-3), np.ones(estimator.n_state)

        estimator.evaluate = nonfinite_candidate_evaluate
        hybrid_se.solve_normal_equations_with_factor = finite_test_step
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.solve_normal_equations_with_factor = original_solver

        self.assertFalse(result.converged)
        self.assertLessEqual(evaluate_count, 10)
        self.assertTrue(np.isfinite(result.residual_inf))

    def test_estimate_reuses_gain_matrix_for_observability(self):
        import secore.hybrid_se as hybrid_se
        from secore.hybrid_se import HybridStateEstimator

        estimator = HybridStateEstimator(
            e_file=ROOT_DIR / "data" / "hybrid" / "qinling.e",
            meas_file=ROOT_DIR / "data" / "hybrid" / "qinling.meas",
        )

        original = hybrid_se.observability_rank_details
        normal_matrix_seen = False

        def counted_rank_details(*args, **kwargs):
            nonlocal normal_matrix_seen
            normal_matrix_seen = kwargs.get("normal_matrix") is not None
            return original(*args, **kwargs)

        hybrid_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate()
        finally:
            hybrid_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(normal_matrix_seen)

    def test_estimate_reuses_cholesky_factor_for_observability(self):
        import secore.hybrid_se as hybrid_se
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

            original = hybrid_se.observability_rank_details
            factor_seen = False

            def counted_rank_details(*args, **kwargs):
                nonlocal factor_seen
                factor_seen = kwargs.get("normal_factor_diag") is not None
                return original(*args, **kwargs)

            hybrid_se.observability_rank_details = counted_rank_details
            try:
                result = estimator.estimate()
            finally:
                hybrid_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(factor_seen)

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

        def counted_builder(H, residual, weight):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight)

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

        self.assertEqual(2, power_calls)
        self.assertEqual(2, current_calls)

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
            ("ACSwitch", "P_FROM"),
            ("ACGenerator", "I_GEN"),
            ("DCBranch", "P_FROM"),
            ("DCSwitch", "I_FROM"),
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
        self.assertIn(("ACLoad", "load_ac_1", "P_LOAD"), pseudo_keys)
        self.assertIn(("ACLoad", "load_ac_1", "Q_LOAD"), pseudo_keys)
        self.assertIn(("DCGenerator", "dc_bus_vctrl", "P_GEN"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        ac_load_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "ACLoad"
            and meas.device_name == "load_ac_1"
            and meas.meas_type == "P_LOAD"
        )
        dc_gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "DCGenerator"
            and meas.device_name == "dc_bus_vctrl"
            and meas.meas_type == "P_GEN"
        )
        self.assertAlmostEqual(ac_load_p.value, estimator.ac_load_by_name["load_ac_1"].p)
        self.assertAlmostEqual(dc_gen_p.value, estimator.dc_generator_by_name["dc_bus_vctrl"].p)

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
            "ACSwitch",
            "DCSwitch",
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


if __name__ == "__main__":
    unittest.main()
