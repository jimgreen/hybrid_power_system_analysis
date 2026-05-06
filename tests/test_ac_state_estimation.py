import unittest
from pathlib import Path
import tempfile

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class ACStateEstimationTest(unittest.TestCase):
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

    def test_build_normal_equations_accepts_sparse_jacobian(self):
        from scipy.sparse import csr_matrix
        from secore.se_math import build_normal_equations

        H_dense = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, -3.0, 0.0],
                [4.0, 0.0, 5.0],
            ]
        )
        residual = np.array([0.5, -1.0, 2.0])
        weight = np.array([0.2, 3.0, 1.5])

        gain, rhs = build_normal_equations(csr_matrix(H_dense), residual, weight)

        np.testing.assert_allclose(gain, H_dense.T @ (weight[:, None] * H_dense))
        np.testing.assert_allclose(rhs, H_dense.T @ (weight * residual))
        self.assertIsInstance(gain, np.ndarray)

    def test_ieee3k_flat_start_does_not_add_angle_pseudos(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
        )
        theta, _voltage = estimator._unpack_state(estimator.initial_state())

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
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
            max_iter=1,
        )
        x0 = estimator.initial_state()

        result = estimator.estimate(verbose=False)
        theta, _voltage = estimator._unpack_state(x0)

        np.testing.assert_allclose(theta, 0.0)
        self.assertTrue(np.isfinite(result.objective))
        self.assertFalse(any(meas.meas_type in ("ANGLE", "THETA") for meas in result.measurements))

    def test_angle_residuals_wrap_across_two_pi(self):
        from dataclasses import replace
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee300.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee300.meas",
            flat_start=False,
        )
        x0 = estimator.initial_state()
        meas = Measurement(
            idx=-1,
            name="angle_wrap_probe",
            device_type="ACNode",
            device_name=estimator.nodes[0].name,
            meas_type="ANGLE",
            weight=1.0,
            valid=True,
            value=0.0,
        )
        single_z_est = estimator.evaluate(x0, [meas])
        wrapped_meas = replace(meas, value=float(single_z_est[0] + 2.0 * np.pi - 0.025))

        residual = estimator._measurement_residual(
            np.array([wrapped_meas.value], dtype=np.float64),
            single_z_est,
            [wrapped_meas],
        )

        self.assertAlmostEqual(-0.025, float(residual[0]), places=12)

    def test_ieee3k_flat_start_keeps_angle_state_zero(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3k.meas",
            flat_start=True,
            max_iter=25,
        )

        theta, _voltage = estimator._unpack_state(estimator.initial_state())

        np.testing.assert_allclose(theta, 0.0)

    def test_reference_node_uses_highest_degree_node_with_valid_voltage_measurement(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        ref = estimator.node_by_name["bus_16"]
        ref_pos = estimator.node_pos[ref.idx]
        ref_voltage = next(
            meas.value
            for meas in estimator.measurements
            if meas.device_type == "ACNode"
            and meas.device_name == "bus_16"
            and meas.meas_type == "V"
            and meas.valid
        )

        self.assertEqual(["bus_16"], [node.name for node in estimator.references])
        self.assertEqual(-1, int(estimator.angle_col[ref_pos]))
        self.assertEqual(-1, int(estimator.voltage_col[ref_pos]))

        theta, voltage = estimator._unpack_state(estimator.initial_state())
        self.assertAlmostEqual(0.0, theta[ref_pos])
        self.assertAlmostEqual(ref_voltage, voltage[ref_pos])

    def test_reference_angle_rebases_nonflat_state_without_angle_pseudos(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee300.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee300.meas",
            flat_start=False,
        )
        ref_angle = float(getattr(estimator.references[0], "angle", 0.0) or 0.0)
        node = estimator.node_by_name["bus_9025"]
        node_pos = estimator.node_pos[node.idx]

        theta, _voltage = estimator._unpack_state(estimator.initial_state())

        expected_angle = float(getattr(node, "angle", 0.0) or 0.0) - ref_angle
        self.assertAlmostEqual(expected_angle, theta[node_pos])
        self.assertFalse(any(meas.name == "pseudo_angle_bus_9025" for meas in estimator.measurements))

    def test_targeted_zero_current_pseudo_uses_to_side_when_from_side_exists(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )
        device_name = next(iter(estimator.zero_branch_by_name))
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = {
            ("ACZeroBranch", device_name, "P_FROM"),
            ("ACZeroBranch", device_name, "Q_FROM"),
        }
        existing_names = set()

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            f"I_Z_RE:{device_name}",
            existing_keys,
            existing_names,
            2,
        )

        self.assertEqual(2, added)
        self.assertIn(("ACZeroBranch", device_name, "P_TO"), existing_keys)
        self.assertIn(("ACZeroBranch", device_name, "Q_TO"), existing_keys)

    def test_targeted_node_voltage_state_does_not_add_pseudo_measurement(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = set()
        existing_names = set()

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            "V:nd_2",
            existing_keys,
            existing_names,
            1,
        )

        self.assertEqual(0, added)
        self.assertNotIn(("ACNode", "nd_2", "V"), existing_keys)

    def test_ieee3w_adds_rank_restoring_pseudos_without_node_voltage_or_angles(self):
        from secore.ac_se import ACStateEstimator
        from secore.se_math import ANGLE_MEASUREMENT_TYPES

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee3w.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee3w.meas",
            flat_start=True,
        )

        rank_pseudos = [
            meas
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_rank_")
        ]
        self.assertTrue(all(meas.weight == estimator.pseudo_measurement_weight for meas in rank_pseudos))
        self.assertFalse(
            any(
                (meas.device_type == "ACNode" and meas.meas_type == "V")
                or meas.meas_type in ANGLE_MEASUREMENT_TYPES
                for meas in rank_pseudos
            )
        )

        observability = estimator.observability_analysis()
        self.assertTrue(observability.observable)
        self.assertEqual(observability.state_count, observability.rank)

    def test_sparse_jacobian_builder_relies_on_coo_to_csr_duplicate_merge(self):
        from unittest.mock import patch
        from scipy.sparse._csr import csr_matrix
        from secore.se_math import SparseJacobianBuilder

        builder = SparseJacobianBuilder((1, 1))
        builder.add_many(
            np.array([0, 0], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float64),
        )

        original = csr_matrix.sum_duplicates
        call_count = 0

        def counted_sum_duplicates(self):
            nonlocal call_count
            call_count += 1
            return original(self)

        with patch.object(csr_matrix, "sum_duplicates", counted_sum_duplicates):
            matrix = builder.to_csr()

        np.testing.assert_allclose(matrix.toarray(), np.array([[3.0]]))
        self.assertEqual(call_count, 1)

    def test_adds_low_weight_pseudo_power_measurements_for_unmetered_generators_and_loads(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "voltage_only.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 vm_bus_1 ACNode bus_1 V 1.0 1 358.587342",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        pseudo = [meas for meas in estimator.active_measurements if meas.name.startswith("pseudo_")]
        pseudo_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in pseudo}

        self.assertIn(("ACGenerator", "gen_30_0", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_30_0", "Q_GEN"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "P_LOAD"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "Q_LOAD"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "ACGenerator"
            and meas.device_name == "gen_30_0"
            and meas.meas_type == "P_GEN"
        )
        load_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "ACLoad"
            and meas.device_name == "load_1"
            and meas.meas_type == "P_LOAD"
        )
        self.assertAlmostEqual(gen_p.value, 2.5)
        self.assertAlmostEqual(load_p.value, 0.976)

    def test_does_not_duplicate_existing_generator_or_load_power_measurements(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertNotIn(("ACGenerator", "gen_30_0", "P_GEN"), pseudo_keys)
        self.assertNotIn(("ACGenerator", "gen_30_0", "Q_GEN"), pseudo_keys)
        self.assertNotIn(("ACLoad", "load_1", "P_LOAD"), pseudo_keys)
        self.assertNotIn(("ACLoad", "load_1", "Q_LOAD"), pseudo_keys)

    def test_pseudo_measurements_are_device_level_for_ac_sources_and_loads(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "device_level.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_gen ACGenerator gen_30_0 V_GEN 1.0 1 345",
                        "# 2 v_load ACLoad load_1 V_LOAD 1.0 1 345",
                        "# 3 p_bad ACGenerator gen_31_1 P_GEN 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertIn(("ACGenerator", "gen_30_0", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_30_0", "Q_GEN"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "P_LOAD"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "Q_LOAD"), pseudo_keys)
        self.assertNotIn(("ACGenerator", "gen_30_0", "V_GEN"), pseudo_keys)
        self.assertNotIn(("ACLoad", "load_1", "V_LOAD"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_31_1", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_31_1", "Q_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_31_1", "V_GEN"), pseudo_keys)

    def test_adds_low_weight_pseudo_measurements_for_unmetered_nodes_switches_and_zero_branches(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "invalid_topology_devices.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_nd_1 ACNode nd_1 V 1.0 1 100",
                        "# 2 v_nd_2_bad ACNode nd_2 V 1.0 0 100",
                        "# 3 p_sw_bad ACSwitch sw_0_1 P_FROM 1.0 0 0",
                        "# 4 p_zbr_bad ACZeroBranch zbr_1_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertNotIn(("ACNode", "nd_1", "V"), pseudo_keys)
        self.assertNotIn(("ACNode", "nd_2", "V"), pseudo_keys)
        self.assertFalse(any(device_type == "ACNode" and meas_type == "V" for device_type, _name, meas_type in pseudo_keys))
        for meas_type in ("P_FROM", "Q_FROM", "V_FROM", "I_FROM"):
            self.assertIn(("ACSwitch", "sw_0_1", meas_type), pseudo_keys)
            self.assertIn(("ACZeroBranch", "zbr_1_2", meas_type), pseudo_keys)

    def test_jacobian_uses_direct_derivatives_without_repeated_evaluation(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
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

    def test_sparse_jacobian_matches_dense_jacobian(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        x = estimator.initial_state()
        dense = estimator.jacobian(x)
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_sparse_jacobian_batches_switch_and_zero_branch_measurements(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        active_devices = {meas.device_type for meas in estimator.active_measurements}
        self.assertIn("ACSwitch", active_devices)
        self.assertIn("ACZeroBranch", active_devices)

        def fail_scalar_zero_current_path(*args, **kwargs):
            raise AssertionError("sparse zero-current rows must be assembled in batches")

        estimator._add_zero_current_measurement_derivatives = fail_scalar_zero_current_path
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_generator_and_load_power_are_explicit_state_variables(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        old_state_count = estimator.n_angle + estimator.n_voltage + 2 * estimator.n_switch_current
        self.assertEqual(
            old_state_count + 2 * len(estimator.generator_order) + 2 * len(estimator.load_order),
            estimator.n_state,
        )

        gen = estimator.generator_order[0]
        load = estimator.load_order[0]
        self.assertIn(f"P_GEN:{gen.name}", estimator.state_labels)
        self.assertIn(f"Q_GEN:{gen.name}", estimator.state_labels)
        self.assertIn(f"P_LOAD:{load.name}", estimator.state_labels)
        self.assertIn(f"Q_LOAD:{load.name}", estimator.state_labels)

    def test_generator_and_load_power_measurements_select_power_states(self):
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        gen = estimator.generator_order[0]
        load = estimator.load_order[0]
        measurements = [
            Measurement(1, "gen_p", "ACGenerator", gen.name, "P_GEN", 1.0, True, 0.0),
            Measurement(2, "gen_q", "ACGenerator", gen.name, "Q_GEN", 1.0, True, 0.0),
            Measurement(3, "load_p", "ACLoad", load.name, "P_LOAD", 1.0, True, 0.0),
            Measurement(4, "load_q", "ACLoad", load.name, "Q_LOAD", 1.0, True, 0.0),
        ]

        H = estimator.jacobian_sparse(estimator.initial_state(), measurements).toarray()

        expected_cols = [
            estimator.gen_p_col_by_name[gen.name],
            estimator.gen_q_col_by_name[gen.name],
            estimator.load_p_col_by_name[load.name],
            estimator.load_q_col_by_name[load.name],
        ]
        for row, col in enumerate(expected_cols):
            nz = np.flatnonzero(np.abs(H[row]) > 1e-12)
            self.assertEqual([col], nz.tolist())
            self.assertEqual(1.0, H[row, col])

    def test_generator_and_load_current_measurements_depend_on_power_states(self):
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        gen = next(
            item
            for item in estimator.generator_order
            if estimator.voltage_col[estimator.node_pos[item.node]] >= 0
        )
        load = next(
            item
            for item in estimator.load_order
            if estimator.voltage_col[estimator.node_pos[item.node]] >= 0
        )
        x = estimator.initial_state()
        x[estimator.gen_p_col_by_name[gen.name]] = 1.2
        x[estimator.gen_q_col_by_name[gen.name]] = 0.5
        x[estimator.load_p_col_by_name[load.name]] = 0.8
        x[estimator.load_q_col_by_name[load.name]] = 0.3
        measurements = [
            Measurement(1, "gen_i", "ACGenerator", gen.name, "I_GEN", 1.0, True, 0.0),
            Measurement(2, "load_i", "ACLoad", load.name, "I_LOAD", 1.0, True, 0.0),
        ]

        H = estimator.jacobian_sparse(x, measurements).toarray()

        gen_voltage_col = estimator.voltage_col[estimator.node_pos[gen.node]]
        load_voltage_col = estimator.voltage_col[estimator.node_pos[load.node]]
        self.assertNotEqual(0.0, H[0, estimator.gen_p_col_by_name[gen.name]])
        self.assertNotEqual(0.0, H[0, estimator.gen_q_col_by_name[gen.name]])
        self.assertNotEqual(0.0, H[0, gen_voltage_col])
        self.assertNotEqual(0.0, H[1, estimator.load_p_col_by_name[load.name]])
        self.assertNotEqual(0.0, H[1, estimator.load_q_col_by_name[load.name]])
        self.assertNotEqual(0.0, H[1, load_voltage_col])

    def test_adds_power_balance_equations_for_every_ac_node(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        balance_rows = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACPowerBalance"
        ]
        balance_keys = {(meas.device_name, meas.meas_type) for meas in balance_rows}

        self.assertEqual(2 * len(estimator.nodes), len(balance_rows))
        for node in estimator.nodes:
            self.assertIn((node.name, "P_BALANCE"), balance_keys)
            self.assertIn((node.name, "Q_BALANCE"), balance_keys)

    def test_sparse_jacobian_batches_generator_power_measurements(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        def fail_per_row_generator_path(*args, **kwargs):
            raise AssertionError("sparse generator rows must be assembled in batches")

        estimator._add_sparse_generator_row = fail_per_row_generator_path
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_sparse_jacobian_batches_generator_triplets_once(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        def fail_repeated_generator_rows(*args, **kwargs):
            raise AssertionError("sparse generator triplets must be emitted in one batch")

        estimator._add_sparse_repeated_rows = fail_repeated_generator_rows
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_sparse_generator_jacobian_uses_cached_y_rows(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        def fail_uncached_y_row_path(*args, **kwargs):
            raise AssertionError("sparse generator Jacobian must use cached Y-row metadata")

        estimator._network_power_derivative_entries = fail_uncached_y_row_path
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_sparse_generator_jacobian_vectorizes_all_generator_entries(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        def fail_single_generator_derivative(*args, **kwargs):
            raise AssertionError("sparse generator Jacobian must vectorize all generator entries")

        estimator._generator_derivative_entries = fail_single_generator_derivative
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_evaluate_batches_generator_measurements(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        generator_measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACGenerator"
        ]
        self.assertIn("I_GEN", {meas.meas_type for meas in generator_measurements})
        expected = estimator.evaluate(x, generator_measurements)

        def fail_scalar_current(*args, **kwargs):
            raise AssertionError("generator values must be evaluated in batches")

        estimator._power_current = fail_scalar_current
        actual = estimator.evaluate(x, generator_measurements)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_evaluate_batches_load_measurements(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
        )
        x = estimator.initial_state()
        load_measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACLoad"
        ]
        self.assertIn("I_LOAD", {meas.meas_type for meas in load_measurements})
        expected = estimator.evaluate(x, load_measurements)

        def fail_scalar_power_current(*args, **kwargs):
            raise AssertionError("load values must be evaluated in batches")

        estimator._power_current = fail_scalar_power_current
        actual = estimator.evaluate(x, load_measurements)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_estimate_reuses_converged_iteration_sparse_jacobian(self):
        from scipy.sparse import issparse
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
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

    def test_estimate_reuses_accepted_step_evaluation(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
            max_iter=20,
        )

        original_evaluate = estimator.evaluate
        call_count = 0

        def counted_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_evaluate(*args, **kwargs)

        estimator.evaluate = counted_evaluate
        result = estimator.estimate()

        self.assertTrue(result.converged)
        self.assertLessEqual(call_count, result.iterations + 1)

    def test_initialization_prepares_simple_measurement_plan(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        cached = estimator._simple_jacobian_plan_cache.get(id(estimator.active_measurements))
        self.assertIsNotNone(cached)
        self.assertIs(cached[0], estimator.active_measurements)

    def test_initialization_prepares_active_measurement_vectors(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        np.testing.assert_allclose(
            estimator.active_z,
            np.asarray([meas.value for meas in estimator.active_measurements], dtype=np.float64),
        )
        np.testing.assert_allclose(
            estimator.active_weight,
            np.asarray([meas.weight for meas in estimator.active_measurements], dtype=np.float64),
        )

    def test_initialization_uses_vectorized_branch_stamps(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_stamp = ac_se.matpower_branch_stamp

        def fail_scalar_stamp(*args, **kwargs):
            raise AssertionError("AC state estimator should use vectorized branch stamps")

        ac_se.matpower_branch_stamp = fail_scalar_stamp
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            )
        finally:
            ac_se.matpower_branch_stamp = original_stamp

        self.assertTrue(estimator.branch_stamp_by_name)
        self.assertTrue(estimator.transformer_stamp_by_name)

    def test_nonconverged_estimate_reuses_factor_for_observability(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
            max_iter=1,
        )
        estimator.tol = 0.0

        original = ac_se.observability_rank_details
        factor_seen = False

        def counted_rank_details(*args, **kwargs):
            nonlocal factor_seen
            factor_seen = kwargs.get("normal_factor_diag") is not None
            return original(*args, **kwargs)

        ac_se.observability_rank_details = counted_rank_details
        try:
            estimator.estimate()
        finally:
            ac_se.observability_rank_details = original

        self.assertTrue(factor_seen)

    def test_estimate_reuses_gain_matrix_for_observability(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        original = ac_se.observability_rank_details
        normal_matrix_seen = False

        def counted_rank_details(*args, **kwargs):
            nonlocal normal_matrix_seen
            normal_matrix_seen = kwargs.get("normal_matrix") is not None
            return original(*args, **kwargs)

        ac_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate()
        finally:
            ac_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(normal_matrix_seen)

    def test_estimate_reuses_cholesky_factor_for_observability(self):
        import secore.ac_se as ac_se
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        original = ac_se.observability_rank_details
        factor_seen = False

        def counted_rank_details(*args, **kwargs):
            nonlocal factor_seen
            factor_seen = kwargs.get("normal_factor_diag") is not None
            return original(*args, **kwargs)

        ac_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate()
        finally:
            ac_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(factor_seen)

    def test_estimate_passes_file_weights_to_normal_equation_builder(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        self.assertTrue(hasattr(ac_se, "build_normal_equations"))
        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        original = ac_se.build_normal_equations
        non_unit_weight_seen = False

        def counted_builder(H, residual, weight):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight)

        ac_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate()
        finally:
            ac_se.build_normal_equations = original

        self.assertTrue(result.converged)
        self.assertTrue(non_unit_weight_seen)

    def test_estimate_uses_cholesky_solver_when_available(self):
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
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

    def test_normal_equation_solver_uses_lapack_posv_when_available(self):
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        try:
            from scipy.linalg.lapack import dposv as original_dposv
        except Exception:
            self.skipTest("SciPy LAPACK dposv is not available")
        self.assertIsNotNone(getattr(se_math, "DPOSV", None))

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        call_count = 0

        def counted_dposv(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_dposv(*args, **kwargs)

        previous = se_math.DPOSV
        se_math.DPOSV = counted_dposv
        try:
            result = estimator.estimate()
        finally:
            se_math.DPOSV = previous

        self.assertTrue(result.converged)
        self.assertGreaterEqual(call_count, 1)

    def test_observability_uses_cholesky_fast_path_when_observable(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
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

    def test_large_sparse_observability_avoids_dense_svd_fallback(self):
        from scipy.sparse import eye
        from secore.se_math import observability_rank_details

        n_state = 2100
        H = eye(n_state, format="csr")[:-1, :]
        labels = [f"x{i}" for i in range(n_state)]
        normal = H.T @ H

        original_svd = np.linalg.svd

        def fail_svd(*args, **kwargs):
            raise AssertionError("large sparse observability should not use dense SVD fallback")

        np.linalg.svd = fail_svd
        try:
            rank, deficiency, singular_values, weak_states = observability_rank_details(
                H,
                labels,
                normal_matrix=normal,
            )
        finally:
            np.linalg.svd = original_svd

        self.assertLess(rank, n_state)
        self.assertGreater(deficiency, 0)
        self.assertEqual(0, singular_values.size)
        self.assertTrue(weak_states)

    def test_observability_uses_lapack_cholesky_when_available(self):
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        try:
            from scipy.linalg.lapack import dpotrf  # noqa: F401
        except Exception:
            self.skipTest("SciPy LAPACK dpotrf is not available")
        self.assertIsNotNone(getattr(se_math, "DPOTRF", None))

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        original_cholesky = np.linalg.cholesky
        call_count = 0

        def counted_cholesky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_cholesky(*args, **kwargs)

        np.linalg.cholesky = counted_cholesky
        try:
            result = estimator.observability_analysis()
        finally:
            np.linalg.cholesky = original_cholesky

        self.assertTrue(result.observable)
        self.assertEqual(0, call_count)

    def test_evaluate_vectorizes_ac_branch_flow_values(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "ac" / "ieee39.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        branch_name = next(iter(estimator.branch_by_name))
        wanted_types = {"P_FROM", "Q_FROM", "I_FROM", "P_TO", "Q_TO", "I_TO"}
        measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACBranch"
            and meas.device_name == branch_name
            and meas.meas_type in wanted_types
        ]
        self.assertEqual(wanted_types, {meas.meas_type for meas in measurements})

        original_power = estimator._branch_power
        original_current = estimator._branch_current
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

        estimator._branch_power = counted_power
        estimator._branch_current = counted_current
        estimator.evaluate(estimator.initial_state(), measurements)

        self.assertEqual(0, power_calls)
        self.assertEqual(0, current_calls)

    def test_evaluate_and_jacobian_reuse_branch_vector_plan(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )
        builder = getattr(estimator, "_build_branch_transformer_vector_plan", None)
        self.assertIsNotNone(builder)

        call_count = 0

        def counted_builder(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return builder(*args, **kwargs)

        estimator._build_branch_transformer_vector_plan = counted_builder
        x = estimator.initial_state()
        estimator.evaluate(x, estimator.active_measurements)
        estimator.jacobian(x, estimator.active_measurements)

        self.assertEqual(0, call_count)

    def test_initializes_active_branch_vector_plan(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        self.assertIn(id(estimator.active_measurements), estimator._branch_transformer_vector_plan_cache)

    def test_jacobian_vectorizes_ac_branch_derivatives(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "ac" / "ieee39.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        branch_name = next(iter(estimator.branch_by_name))
        wanted_types = {"P_FROM", "Q_FROM", "I_FROM", "P_TO", "Q_TO", "I_TO"}
        measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACBranch"
            and meas.device_name == branch_name
            and meas.meas_type in wanted_types
        ]
        self.assertEqual(wanted_types, {meas.meas_type for meas in measurements})

        original_power = estimator._branch_power_derivatives
        original_current = estimator._branch_current_derivatives
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

        estimator._branch_power_derivatives = counted_power
        estimator._branch_current_derivatives = counted_current
        estimator.jacobian(estimator.initial_state(), measurements)

        self.assertEqual(0, power_calls)
        self.assertEqual(0, current_calls)

    def test_jacobian_reuses_generator_derivative_vectors_per_generator(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        gen_name = next(iter(estimator.generator_by_name))
        wanted_types = {"P_GEN", "Q_GEN", "I_GEN"}
        measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACGenerator"
            and meas.device_name == gen_name
            and meas.meas_type in wanted_types
        ]
        self.assertEqual(wanted_types, {meas.meas_type for meas in measurements})

        original = estimator._generator_derivative_vectors
        call_count = 0

        def counted_derivatives(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        estimator._generator_derivative_vectors = counted_derivatives
        estimator.jacobian(estimator.initial_state(), measurements)

        self.assertEqual(0, call_count)

    def test_generator_jacobian_uses_needed_network_derivative_rows_only(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        generator_measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACGenerator" and meas.meas_type in ("P_GEN", "Q_GEN", "I_GEN")
        ]
        self.assertTrue(generator_measurements)

        original = estimator._network_power_derivatives
        call_count = 0

        def counted_full_derivatives(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        estimator._network_power_derivatives = counted_full_derivatives
        estimator.jacobian(estimator.initial_state(), generator_measurements)

        self.assertEqual(0, call_count)

    def test_analytic_jacobian_matches_finite_difference(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ieee39.meas",
        )

        x = estimator.initial_state()
        H = estimator.jacobian(x)
        H_num = np.zeros_like(H)
        for col in range(estimator.n_state):
            step = 1e-6 * max(1.0, abs(x[col]))
            xp = x.copy()
            xm = x.copy()
            xp[col] += step
            xm[col] -= step
            H_num[:, col] = (estimator.evaluate(xp) - estimator.evaluate(xm)) / (2.0 * step)

        self.assertLess(float(np.max(np.abs(H - H_num))), 1e-5)

    def test_ac_net_30_estimates_with_current_voltage_and_switch_measurements(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "ac" / "ac_net_30.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                max_iter=20,
            )

        active_types = {meas.meas_type for meas in estimator.active_measurements}
        active_devices = {meas.device_type for meas in estimator.active_measurements}
        self.assertIn("I_FROM", active_types)
        self.assertIn("I_TO", active_types)
        self.assertIn("I_GEN", active_types)
        self.assertIn("I_LOAD", active_types)
        self.assertIn("ACSwitch", active_devices)

        result = estimator.estimate()
        self.assertTrue(result.converged)
        self.assertTrue(result.observability.observable)
        self.assertLess(result.residual_inf, 1e-6)

    def test_ac_net_30_estimates_with_zero_branch_measurements(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "ac" / "ac_net_30.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                max_iter=20,
            )

        zero_branch_measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACZeroBranch"
        ]
        zero_branch_types = {meas.meas_type for meas in zero_branch_measurements}
        self.assertEqual(4 * len(estimator.zero_branches), len(zero_branch_measurements))
        self.assertIn("P_FROM", zero_branch_types)
        self.assertIn("Q_FROM", zero_branch_types)
        self.assertIn("V_FROM", zero_branch_types)
        self.assertIn("I_FROM", zero_branch_types)
        for zero_branch in estimator.zero_branches:
            per_device_types = {
                meas.meas_type
                for meas in zero_branch_measurements
                if meas.device_name == zero_branch.name
            }
            self.assertEqual({"P_FROM", "Q_FROM", "V_FROM", "I_FROM"}, per_device_types)

        result = estimator.estimate()
        self.assertTrue(result.converged)
        self.assertTrue(result.observability.observable)
        self.assertLess(result.residual_inf, 1e-6)

    def test_ieee3k_zero_branch_compression_makes_flat_start_observable(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "ac" / "ieee3k.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "ac" / "ieee3k.e",
                meas_file=meas_file,
                flat_start=True,
                max_iter=30,
            )

        constraint_types = {
            meas.device_type
            for meas in estimator.active_measurements
            if meas.device_type in ("ACZeroBranchConstraint", "ACSwitchConstraint")
        }
        self.assertEqual(set(), constraint_types)
        for zero_branch in estimator.zero_branches:
            i = estimator.node_pos[zero_branch.i_node]
            j = estimator.node_pos[zero_branch.j_node]
            self.assertEqual(estimator.voltage_col[i], estimator.voltage_col[j])
            self.assertEqual(estimator.angle_col[i], estimator.angle_col[j])

        initial_observability = estimator.observability_analysis()
        self.assertTrue(initial_observability.observable)
        self.assertEqual(initial_observability.state_count, initial_observability.rank)

        result = estimator.estimate(verbose=False)
        self.assertTrue(result.converged)
        self.assertTrue(result.observability.observable)

    def test_closed_ac_switches_share_voltage_and_angle_states(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )

        switch_constraints = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACSwitchConstraint"
        ]
        self.assertEqual([], switch_constraints)
        for switch in estimator.switches:
            i = estimator.node_pos[switch.i_node]
            j = estimator.node_pos[switch.j_node]
            self.assertEqual(estimator.voltage_col[i], estimator.voltage_col[j])
            self.assertEqual(estimator.angle_col[i], estimator.angle_col[j])

    def test_compact_state_unpack_uses_precomputed_bulk_indexes(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )

        self.assertEqual(
            int(np.sum(estimator.angle_col >= 0)),
            int(estimator._angle_unpack_nodes.size),
        )
        self.assertEqual(
            int(np.sum(estimator.voltage_col >= 0)),
            int(estimator._voltage_unpack_nodes.size),
        )
        self.assertEqual(estimator._angle_unpack_nodes.size, estimator._angle_unpack_cols.size)
        self.assertEqual(estimator._voltage_unpack_nodes.size, estimator._voltage_unpack_cols.size)

    def test_sparse_normal_solver_disables_dynamic_pivoting_for_speed(self):
        from scipy.sparse import csc_matrix
        import secore.se_math as se_math

        calls = []
        original_splu = se_math.SP_SPLU

        class FakeLU:
            def __init__(self):
                self.U = csc_matrix(np.eye(2))

            def solve(self, rhs):
                return np.asarray(rhs, dtype=np.float64)

        def fake_splu(matrix, **kwargs):
            calls.append(kwargs)
            return FakeLU()

        se_math.SP_SPLU = fake_splu
        try:
            dx, diag = se_math.solve_normal_equations_with_factor(csc_matrix(np.eye(2)), np.ones(2))
        finally:
            se_math.SP_SPLU = original_splu

        self.assertTrue(calls)
        self.assertEqual(0.0, calls[0]["diag_pivot_thresh"])
        np.testing.assert_allclose(dx, np.ones(2))
        np.testing.assert_allclose(diag, np.ones(2))

    def test_ac_net_30_analytic_jacobian_matches_finite_difference(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "ac" / "ac_net_30.meas",
        )

        x = estimator.initial_state()
        H = estimator.jacobian(x)
        H_num = np.zeros_like(H)
        for col in range(estimator.n_state):
            step = 1e-6 * max(1.0, abs(x[col]))
            xp = x.copy()
            xm = x.copy()
            xp[col] += step
            xm[col] -= step
            H_num[:, col] = (estimator.evaluate(xp) - estimator.evaluate(xm)) / (2.0 * step)

        self.assertLess(float(np.max(np.abs(H - H_num))), 1e-5)


if __name__ == "__main__":
    unittest.main()
