import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

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

    def test_summary_cache_uses_table_and_primes_voltage_observation_cache(self):
        from model.meas_model import (
            MEAS_STATUS_PSEUDO,
            Measurement,
            MeasurementList,
            measurement_table_from_measurements,
        )
        from secore.ac_se import ACStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("summary cache should use the cached table")

        rows = [
            Measurement(1, "node_v", "ACNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "node_v_status_pseudo", "ACNode", "n2", "V", 3.0, True, 0.99, MEAS_STATUS_PSEUDO),
        ]
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1), "n2": SimpleNamespace(idx=2)}
        estimator.node_pos = {1: 0, 2: 1}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}
        estimator.branch_by_name = {}
        estimator.transformer_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}

        estimator._refresh_measurement_summary_cache()

        self.assertEqual({("ACNode", "n1"), ("ACNode", "n2")}, estimator._active_device_key_cache)
        self.assertEqual(2, estimator._max_measurement_idx)
        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_nodes())

    def test_standard_measurement_parser_uses_status_fast_path_without_status_column(self):
        import secore.ac_se as ac_se
        from model.meas_model import MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL, Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@idx name dev_type dev_name meas_type weight valid value",
                        "#1 v1 ACNode n1 V 2.0 1 1.01",
                        "#2 v2 ACNode n2 V 2.0 0 0.99",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            original_normalize = ac_se.normalize_measurement_status

            def fail_normalize(*_args, **_kwargs):
                raise AssertionError("standard rows without status should use direct status codes")

            ac_se.normalize_measurement_status = fail_normalize
            try:
                measurements = ac_se._read_measurements_direct(meas_file, Measurement)
            finally:
                ac_se.normalize_measurement_status = original_normalize

        self.assertEqual([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID], [meas.status for meas in measurements])
        np.testing.assert_array_equal(measurements.table.status_code, np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID]))
        np.testing.assert_array_equal(measurements.table.valid, np.array([True, False]))

    def test_standard_measurement_parser_builds_device_type_rows_while_reading(self):
        from unittest.mock import patch

        import secore.ac_se as ac_se
        from model.meas_model import DEVICE_TYPE_CODES, Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v1 ACNode n1 V 1.0 1 10.0",
                        "# 2 p1 ACLoad load P_LOAD 1.0 1 20.0",
                        "# 3 v2 ACNode n2 V 1.0 1 11.0",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(ac_se.np, "unique", side_effect=AssertionError("row cache should be built during parsing")):
                measurements = ac_se._read_measurements_direct(meas_file, Measurement)

        cache = measurements.table.rows_by_device_type_code
        np.testing.assert_array_equal(cache[DEVICE_TYPE_CODES["ACNode"]], np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(cache[DEVICE_TYPE_CODES["ACLoad"]], np.array([1], dtype=np.int64))

    def test_standard_measurement_parser_can_return_table_backed_rows_without_object_list(self):
        import secore.ac_se as ac_se
        from model.meas_model import DEVICE_TYPE_CODES, Measurement, TableBackedMeasurementList

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@idx name dev_type dev_name meas_type weight valid value",
                        "#1 v1 ACNode n1 V 2.0 1 1.01",
                        "#2 p1 ACLoad load P_LOAD 3.0 1 5.0",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            class RejectMeasurement(Measurement):
                def __new__(cls, *_args, **_kwargs):
                    raise AssertionError("table-only parsing should not instantiate Measurement rows")

            measurements = ac_se._read_measurements_direct(meas_file, RejectMeasurement, table_only=True)

        self.assertIsInstance(measurements, TableBackedMeasurementList)
        self.assertEqual(2, len(measurements))
        self.assertEqual(0, list.__len__(measurements))
        np.testing.assert_array_equal(measurements.table.device_type_code, np.array([DEVICE_TYPE_CODES["ACNode"], DEVICE_TYPE_CODES["ACLoad"]], dtype=np.int16))
        self.assertEqual("v1", measurements[0].name)
        self.assertEqual("P_LOAD", measurements[1].meas_type)

    def test_summary_cache_maps_only_voltage_rows(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("summary cache should use the cached table")

        rows = [
            Measurement(1, "node_v", "ACNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "load_p", "ACLoad", "load_1", "P_LOAD", 2.0, True, 0.5),
            Measurement(3, "load_q", "ACLoad", "load_1", "Q_LOAD", 2.0, True, 0.2),
        ]
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1)}
        estimator.node_pos = {1: 0}
        estimator.generator_by_name = {}
        estimator.load_by_name = {"load_1": SimpleNamespace(node=1)}
        estimator.branch_by_name = {}
        estimator.transformer_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        mapped = []

        def voltage_mapper(_device_type, device_name, meas_type):
            if not str(meas_type).startswith("V"):
                raise AssertionError("power rows should not be passed to the voltage mapper")
            mapped.append((device_name, meas_type))
            return {"n1": 1}.get(device_name)

        estimator._voltage_measurement_node_idx = voltage_mapper

        estimator._refresh_measurement_summary_cache()

        self.assertEqual([("n1", "V")], mapped)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_node_cache)
        self.assertIn(("ACLoad", "load_1", "P_LOAD"), estimator._active_measurement_key_cache)

    def test_real_voltage_observation_uses_table_and_maps_only_voltage_rows(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("real voltage observation should use the cached table")

        rows = [
            Measurement(1, "node_v", "ACNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "load_p", "ACLoad", "load_1", "P_LOAD", 2.0, True, 0.5),
            Measurement(3, "load_q", "ACLoad", "load_1", "Q_LOAD", 2.0, True, 0.2),
        ]
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_pos = {1: 0}
        mapped = []

        def voltage_mapper(_device_type, device_name, meas_type):
            if not str(meas_type).startswith("V"):
                raise AssertionError("power rows should not be passed to the voltage mapper")
            mapped.append((device_name, meas_type))
            return {"n1": 1}.get(device_name)

        estimator._voltage_measurement_node_idx = voltage_mapper

        observed = estimator._real_voltage_observation_nodes()

        self.assertEqual([("n1", "V")], mapped)
        self.assertEqual({1: 1.02}, observed)

    def test_conversion_primes_voltage_observation_cache(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        rows = [Measurement(1, "node_v", "ACNode", "n1", "V", 2.0, True, 10.2)]
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = MeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.flat_start = False
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.u_scale = 1.0
        estimator.i_scale = 1.0
        estimator.network = SimpleNamespace(node_dict={1: SimpleNamespace(vbase=10.0)})
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1)}
        estimator.node_pos = {1: 0}
        estimator.branch_by_name = {}
        estimator.transformer_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}

        estimator._convert_measurements_to_pu()

        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_node_cache)

    def test_conversion_uses_table_without_iterating_table_backed_measurements(self):
        from model.meas_model import Measurement, TableBackedMeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        class NoIterTableBackedMeasurementList(TableBackedMeasurementList):
            def __iter__(self):
                raise AssertionError("AC conversion should use the measurement table directly")

        rows = [Measurement(1, "node_v", "ACNode", "n1", "V", 2.0, True, 10.2)]
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = NoIterTableBackedMeasurementList(measurement_table_from_measurements(rows))
        estimator.flat_start = False
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.u_scale = 1.0
        estimator.i_scale = 1.0
        estimator.network = SimpleNamespace(node_dict={1: SimpleNamespace(vbase=10.0)})
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1)}
        estimator.node_pos = {1: 0}
        estimator.node_by_idx = {1: estimator.node_by_name["n1"]}
        estimator.branch_by_name = {}
        estimator.transformer_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}

        estimator._convert_measurements_to_pu()

        np.testing.assert_allclose(estimator.measurements.table.value, np.array([1.02]))
        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)

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

    def test_sparse_jacobian_builder_retains_explicit_zero_entries(self):
        from secore.se_math import SparseJacobianBuilder

        builder = SparseJacobianBuilder((2, 3))
        builder.add(0, 0, 0.0)
        builder.add_many(
            np.array([0, 1, 1]),
            np.array([1, 1, 2]),
            np.array([0.0, 2.0, 0.0]),
        )
        H = builder.to_csr()

        self.assertEqual(4, H.nnz)
        self.assertIn(0.0, H.data)
        np.testing.assert_allclose(H.toarray(), np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))

    def test_sparse_normal_equations_keep_structural_zero_pattern(self):
        from secore.se_math import SparseJacobianBuilder, build_normal_equations

        builder = SparseJacobianBuilder((1, 2))
        builder.add_many(np.array([0, 0]), np.array([0, 1]), np.array([1.0, 0.0]))
        H = builder.to_csr()
        gain, rhs = build_normal_equations(
            H,
            np.array([3.0]),
            np.array([1.0]),
            dense_gain_limit=0,
        )

        self.assertEqual(4, gain.nnz)
        np.testing.assert_allclose(gain.toarray(), np.array([[1.0, 0.0], [0.0, 0.0]]))
        np.testing.assert_allclose(rhs, np.array([3.0, 0.0]))

    def test_build_normal_equations_uses_precomputed_uniform_weight_flag(self):
        from scipy.sparse import csr_matrix
        import secore.se_math as se_math

        H_dense = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        residual = np.array([0.5, -1.0, 2.0])
        weight = np.array([2.0, 2.0, 2.0])

        original_all = se_math.np.all

        def fail_all(*_args, **_kwargs):
            raise AssertionError("uniform weight fast path should not scan the weight vector")

        se_math.np.all = fail_all
        try:
            gain, rhs = se_math.build_normal_equations(
                csr_matrix(H_dense),
                residual,
                weight,
                uniform_weight=2.0,
            )
        finally:
            se_math.np.all = original_all

        np.testing.assert_allclose(gain.toarray() if hasattr(gain, "toarray") else gain, 2.0 * (H_dense.T @ H_dense))
        np.testing.assert_allclose(rhs, 2.0 * (H_dense.T @ residual))

    def test_build_normal_equations_uses_precomputed_nonuniform_weight_flag(self):
        from scipy.sparse import csr_matrix
        import secore.se_math as se_math

        H_dense = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        residual = np.array([0.5, -1.0, 2.0])
        weight = np.array([1.0, 2.0, 3.0])

        original_all = se_math.np.all

        def fail_all(*_args, **_kwargs):
            raise AssertionError("known nonuniform weight path should not scan the weight vector")

        se_math.np.all = fail_all
        try:
            gain, rhs = se_math.build_normal_equations(
                csr_matrix(H_dense),
                residual,
                weight,
                weights_are_uniform=False,
            )
        finally:
            se_math.np.all = original_all

        np.testing.assert_allclose(gain.toarray() if hasattr(gain, "toarray") else gain, H_dense.T @ (weight[:, None] * H_dense))
        np.testing.assert_allclose(rhs, H_dense.T @ (weight * residual))

    def test_build_normal_equations_uses_supplied_weighted_residual(self):
        from scipy.sparse import csr_matrix
        from secore.se_math import build_normal_equations

        H_dense = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        residual = np.array([100.0, 100.0, 100.0])
        weight = np.array([1.0, 2.0, 3.0])
        weighted_residual = np.array([0.5, -2.0, 6.0])

        gain, rhs = build_normal_equations(
            csr_matrix(H_dense),
            residual,
            weight,
            weights_are_uniform=False,
            weighted_residual=weighted_residual,
        )

        np.testing.assert_allclose(gain.toarray() if hasattr(gain, "toarray") else gain, H_dense.T @ (weight[:, None] * H_dense))
        np.testing.assert_allclose(rhs, H_dense.T @ weighted_residual)

    def test_active_vectorized_evaluate_and_jacobian_skip_full_mask_scan(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        x = estimator.initial_state()
        self.assertTrue(estimator.active_measurements_are_vectorized)

        original_all = ac_se.np.all

        def fail_all(*_args, **_kwargs):
            raise AssertionError("active vectorized path should not scan the handled mask")

        ac_se.np.all = fail_all
        try:
            z_est = estimator.evaluate(x)
            H = estimator.jacobian_sparse(x)
        finally:
            ac_se.np.all = original_all

        self.assertEqual(len(estimator.active_measurements), z_est.size)
        self.assertEqual((len(estimator.active_measurements), estimator.n_state), H.shape)

    def test_refresh_active_measurements_uses_unified_plan_tables(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )

        original_builder = ac_se.build_measurement_plan_table
        call_count = 0

        def counted_builder(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_builder(*args, **kwargs)

        ac_se.build_measurement_plan_table = counted_builder
        try:
            estimator._refresh_active_measurement_indexes()
        finally:
            ac_se.build_measurement_plan_table = original_builder

        self.assertEqual(0, call_count)
        self.assertTrue(estimator.active_measurements_are_vectorized)

    def test_measurement_plan_lookup_rebuild_uses_ppc_rows_not_device_object_maps(self):
        from secore.ac_se import ACStateEstimator

        class RejectDeviceMap(dict):
            def items(self):
                raise AssertionError("measurement plan lookup should use PPC rows, not device object maps")

            def values(self):
                raise AssertionError("measurement plan lookup should use PPC rows, not device object maps")

            def get(self, *_args, **_kwargs):
                raise AssertionError("measurement plan lookup should use PPC row state arrays, not name maps")

            def __contains__(self, _key):
                raise AssertionError("measurement plan lookup should use PPC rows, not device object maps")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        ppc = estimator.network.ppc
        topology = ppc["_topology_arrays"]

        for attr in (
            "node_pos",
            "node_by_name",
            "branch_by_name",
            "transformer_by_name",
            "generator_by_name",
            "load_by_name",
            "zero_branch_by_name",
            "break_by_name",
            "branch_stamp_by_name",
            "transformer_stamp_by_name",
            "zero_branch_pos",
            "break_pos",
            "generator_state_index_by_name",
            "load_state_index_by_name",
        ):
            setattr(estimator, attr, RejectDeviceMap())

        estimator._build_measurement_plan_lookup_arrays()

        self.assertEqual(int(np.count_nonzero(topology.node_alive_mask)), estimator._ac_node_plan_pos.size)
        self.assertEqual(int(np.count_nonzero(topology.devices["branch"].alive_mask)), estimator._ac_branch_plan_i.size)
        self.assertEqual(
            int(np.count_nonzero(topology.devices["transformer"].alive_mask)),
            estimator._ac_transformer_plan_i.size,
        )
        self.assertEqual(int(np.count_nonzero(topology.devices["gen"].alive_mask)), estimator._ac_generator_plan_index.size)
        self.assertEqual(int(np.count_nonzero(topology.devices["load"].alive_mask)), estimator._ac_load_plan_index.size)

    def test_observability_pseudo_candidates_enumerate_ppc_rows_not_device_maps(self):
        from secore.ac_se import ACStateEstimator

        class RejectDeviceMap(dict):
            def items(self):
                raise AssertionError("pseudo candidate enumeration should use PPC rows")

            def values(self):
                raise AssertionError("pseudo candidate enumeration should use PPC rows")

            def get(self, *_args, **_kwargs):
                raise AssertionError("pseudo candidate enumeration should use PPC rows")

            def __contains__(self, _key):
                raise AssertionError("pseudo candidate enumeration should use PPC rows")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        estimator._voltage_pseudo_is_covered = lambda *_args, **_kwargs: False
        estimator._active_measurement_key_cache = set()
        estimator._active_device_key_cache = set()
        for attr in (
            "node_by_name",
            "branch_by_name",
            "transformer_by_name",
            "generator_by_name",
            "load_by_name",
        ):
            setattr(estimator, attr, RejectDeviceMap())

        candidates = estimator._observability_pseudo_candidate_measurements()
        candidate_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in candidates}

        self.assertIn(("ACNode", "bus_1", "V"), candidate_keys)
        self.assertIn(("ACLoad", "load_1", "P_LOAD"), candidate_keys)
        self.assertIn(("ACGenerator", "gen_30_0", "P_GEN"), candidate_keys)
        self.assertIn(("ACBranch", "line_1_2", "P_FROM"), candidate_keys)
        self.assertIn(("ACTransformer", "tr_2_30", "P_FROM"), candidate_keys)

    def test_regular_pseudo_measurements_enumerate_ppc_rows_not_device_lists(self):
        from secore.ac_se import ACStateEstimator

        class RejectIterable(list):
            def __iter__(self):
                raise AssertionError("regular pseudo generation should use PPC rows")

        class RejectDeviceMap(dict):
            def values(self):
                raise AssertionError("regular pseudo generation should use PPC rows")

            def get(self, *_args, **_kwargs):
                raise AssertionError("regular pseudo generation should use PPC rows")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator._voltage_measurement_node_idx = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("regular pseudo generation should use PPC node ids")
        )
        estimator._voltage_pseudo_is_covered = lambda *_args, **_kwargs: False
        estimator._active_measurement_key_cache = set()
        estimator._active_device_key_cache = set()
        estimator.zero_branches = RejectIterable()
        estimator.breakers = RejectIterable()
        estimator.generator_order = RejectIterable()
        estimator.load_order = RejectIterable()
        estimator.generator_by_name = RejectDeviceMap()
        estimator.load_by_name = RejectDeviceMap()

        estimator._add_pseudo_power_measurements()
        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertIn(("ACGenerator", "gen_30_0", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_30_0", "Q_GEN"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "P_LOAD"), pseudo_keys)
        self.assertIn(("ACLoad", "load_1", "Q_LOAD"), pseudo_keys)

    def test_voltage_measurement_node_lookup_uses_ppc_rows_not_device_maps(self):
        from secore.ac_se import ACStateEstimator

        class RejectDeviceMap(dict):
            def get(self, *_args, **_kwargs):
                raise AssertionError("voltage node lookup should use PPC row maps")

            def __contains__(self, _key):
                raise AssertionError("voltage node lookup should use PPC row maps")

            def __getitem__(self, _key):
                raise AssertionError("voltage node lookup should use PPC row maps")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        for attr in (
            "node_by_name",
            "branch_by_name",
            "transformer_by_name",
            "generator_by_name",
            "load_by_name",
            "zero_branch_by_name",
            "break_by_name",
        ):
            setattr(estimator, attr, RejectDeviceMap())

        self.assertEqual(1, estimator._voltage_measurement_node_idx("ACNode", "bus_1", "V"))
        self.assertEqual(30, estimator._voltage_measurement_node_idx("ACGenerator", "gen_30_0", "V_GEN"))
        self.assertEqual(1, estimator._voltage_measurement_node_idx("ACLoad", "load_1", "V_LOAD"))
        self.assertEqual(1, estimator._voltage_measurement_node_idx("ACBranch", "line_1_2", "V_FROM"))
        self.assertEqual(2, estimator._voltage_measurement_node_idx("ACBranch", "line_1_2", "V_TO"))
        self.assertEqual(30, estimator._voltage_measurement_node_idx("ACTransformer", "tr_2_30", "V_TO"))

    def test_file_backed_measurement_normalization_uses_meas_ppc_ids(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            auto_prepare=False,
        )

        def reject_voltage_lookup(*_args, **_kwargs):
            raise AssertionError("file-backed normalization should use meas PPC ids")

        estimator._voltage_measurement_node_idx = reject_voltage_lookup
        estimator.prepare(prepare_active_measurements=False)

        self.assertTrue(estimator.meas_ppc["normalized"])
        self.assertTrue(estimator.measurements.normalized)
        self.assertTrue(estimator._node_voltage_measurement_cache)

    def test_zero_tie_state_layout_uses_ppc_topology_not_device_lists(self):
        from secore.ac_se import ACStateEstimator

        class RejectIterable(list):
            def __iter__(self):
                raise AssertionError("zero-tie state layout should use PPC topology rows")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        expected_components = [tuple(component) for component in estimator.zero_tie_components]
        estimator.zero_branches = RejectIterable()
        estimator.switches = RejectIterable()
        estimator.breakers = RejectIterable()

        estimator._build_zero_tie_state_layout()

        self.assertEqual(expected_components, [tuple(component) for component in estimator.zero_tie_components])

    def test_active_measurement_rows_use_cached_table_rows(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        class FailingList(list):
            def __getitem__(self, _index):
                raise AssertionError("active row lookup should use cached table rows")

        estimator.active_measurements = FailingList(estimator.active_measurements)

        rows = estimator._active_measurement_rows_for_types(("ACNode", "ACBranch"))

        self.assertGreater(rows.size, 0)
        self.assertTrue(np.issubdtype(rows.dtype, np.integer))

    def test_active_fallback_rows_can_use_table_without_iterating_measurements(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        class TableBackedList(list):
            def __iter__(self):
                raise AssertionError("fallback rows should use cached table arrays")

        measurements = TableBackedList(estimator.active_measurements)
        measurements.table = estimator.active_measurement_table

        zero_mask = np.zeros(len(measurements), dtype=bool)

        def no_vectorized_rows(*_args, **_kwargs):
            return zero_mask

        estimator._fill_branch_transformer_values_vectorized = no_vectorized_rows
        estimator._fill_zero_current_values_vectorized = no_vectorized_rows
        estimator._fill_simple_values_vectorized = no_vectorized_rows
        estimator._fill_generator_values_vectorized = no_vectorized_rows
        # Keep balance vectorization enabled because Jacobian fallback does not carry
        # a scalar ACPowerBalance branch and normally relies on the vectorized plan.
        estimator._fill_branch_transformer_jacobian_vectorized = no_vectorized_rows
        estimator._fill_zero_current_jacobian_vectorized = no_vectorized_rows
        estimator._fill_simple_jacobian_vectorized = no_vectorized_rows
        estimator._fill_generator_jacobian_sparse = no_vectorized_rows

        x = estimator.initial_state()
        z_est = estimator.evaluate(x, measurements)
        H = estimator.jacobian_sparse(x, measurements)

        self.assertEqual(len(measurements), z_est.size)
        self.assertEqual((len(measurements), estimator.n_state), H.shape)

    def test_measurement_residual_skips_angle_mask_scan_when_flag_false(self):
        import secore.se_math as se_math

        original_any = se_math.np.any

        def fail_any(*_args, **_kwargs):
            raise AssertionError("no-angle residual fast path should not scan the angle mask")

        se_math.np.any = fail_any
        try:
            residual = se_math.measurement_residual(
                np.array([3.0, 5.0]),
                np.array([1.0, 2.0]),
                np.array([False, False]),
                has_angle_residuals=False,
            )
        finally:
            se_math.np.any = original_any

        np.testing.assert_allclose(residual, np.array([2.0, 3.0]))

    def test_load_measurements_uses_meas_ppc_parser(self):
        import secore.ac_se as ac_se
        from model.meas_model import TableBackedMeasurementList
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Other>",
                        "@ idx name",
                        "# 1 ignored",
                        "</Other>",
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_ebook = ac_se.EBook

            class RejectEBook:
                def __init__(self, *_args, **_kwargs):
                    raise AssertionError("measurement loading should not use the generic EBook parser")

            def reject_direct_parser(*_args, **_kwargs):
                raise AssertionError("ACStateEstimator measurement loading should use meas PPC")

            ac_se.EBook = RejectEBook
            original_direct = ac_se._read_measurements_direct
            ac_se._read_measurements_direct = reject_direct_parser
            try:
                measurements = ACStateEstimator._load_measurements(meas_file)
            finally:
                ac_se.EBook = original_ebook
                ac_se._read_measurements_direct = original_direct

        self.assertEqual(1, len(measurements))
        self.assertIsInstance(measurements, TableBackedMeasurementList)
        self.assertEqual(0, list.__len__(measurements))
        measurement = measurements[0]
        self.assertEqual(7, measurement.idx)
        self.assertEqual("vm_bus_1", measurement.name)
        self.assertEqual("ACNode", measurement.device_type)
        self.assertEqual("bus_1", measurement.device_name)
        self.assertEqual("V", measurement.meas_type)
        self.assertEqual(2.5, measurement.weight)
        self.assertTrue(measurement.valid)
        self.assertEqual(345.6, measurement.value)

    def test_measurement_read_from_file_uses_direct_measurement_parser(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator, Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_read_direct = ac_se._read_measurements_direct
            calls = []

            def counted_read_direct(file_name, measurement_cls, *args, **kwargs):
                calls.append(Path(file_name))
                return original_read_direct(file_name, measurement_cls, *args, **kwargs)

            ac_se._read_measurements_direct = counted_read_direct
            try:
                measurements = Measurement.read_from_file(meas_file)
                loaded_by_estimator = ACStateEstimator._load_measurements(meas_file)
            finally:
                ac_se._read_measurements_direct = original_read_direct

        self.assertEqual([meas_file], calls)
        self.assertEqual(1, len(measurements))
        self.assertEqual("V", measurements[0].meas_type)
        self.assertEqual(1, len(loaded_by_estimator))
        self.assertEqual("V", loaded_by_estimator[0].meas_type)

    def test_measurement_read_from_file_bypasses_intermediate_row_objects(self):
        import efile_read
        from secore.ac_se import Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original_factory = efile_read.efile_factory_from_file

            def reject_factory(_file_name):
                raise AssertionError("measurement loading should use raw rows, not row objects")

            efile_read.efile_factory_from_file = reject_factory
            try:
                measurements = Measurement.read_from_file(meas_file)
            finally:
                efile_read.efile_factory_from_file = original_factory

        self.assertEqual(1, len(measurements))
        self.assertEqual("V", measurements[0].meas_type)

    def test_measurement_read_from_file_bypasses_generic_raw_efile_parser(self):
        import secore.ac_se as ac_se
        import efile_read
        from secore.ac_se import Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Other>",
                        "@ idx name",
                        "# 1 ignored",
                        "</Other>",
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertFalse(hasattr(efile_read, "read_efile_rows_cached"))
            measurements = Measurement.read_from_file(meas_file)

        self.assertEqual(1, len(measurements))
        self.assertEqual("V", measurements[0].meas_type)

    def test_measurement_read_from_file_builds_column_table(self):
        from secore.ac_se import Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "# 8 p_branch ACBranch br_1 p_from 3.5 0 12.0",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            measurements = Measurement.read_from_file(meas_file)

        table = measurements.table
        self.assertEqual(2, len(measurements))
        np.testing.assert_array_equal(table.idx, np.array([7, 8], dtype=np.int64))
        np.testing.assert_array_equal(table.device_type, np.array(["ACNode", "ACBranch"], dtype=object))
        np.testing.assert_array_equal(table.meas_type, np.array(["V", "P_FROM"], dtype=object))
        np.testing.assert_allclose(table.weight, np.array([2.5, 3.5]))
        np.testing.assert_array_equal(table.valid, np.array([True, False]))
        np.testing.assert_allclose(table.value, np.array([345.6, 12.0]))

    def test_measurement_read_from_file_can_normalize_in_parse(self):
        from types import SimpleNamespace
        from secore.ac_se import Measurement

        context = SimpleNamespace(
            flat_start=False,
            p_base=100000.0,
            p_base_kW=100000.0,
            u_scale=1000.0,
            i_scale=1.0,
            network=SimpleNamespace(node_dict={1: SimpleNamespace(vbase=345.0)}),
            node_by_name={"bus_1": SimpleNamespace(idx=1)},
            branch_by_name={},
            transformer_by_name={},
            zero_branch_by_name={},
            switch_by_name={},
            generator_by_name={},
            load_by_name={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345000.0",
                        "# 8 theta_bus_1 ACNode bus_1 angle 1.0 1 90.0",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            measurements = Measurement.read_from_file(meas_file, scale_context=context)

        self.assertTrue(getattr(measurements, "normalized", False))
        self.assertAlmostEqual(1.0, measurements[0].value)
        self.assertAlmostEqual(np.pi / 2, measurements[1].value)
        np.testing.assert_allclose(measurements.table.value, np.array([1.0, np.pi / 2]))

    def test_measurement_read_from_file_does_not_share_warm_template_cache(self):
        from secore.ac_se import Measurement

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "minimal.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 7 vm_bus_1 ACNode bus_1 v 2.5 1 345.6",
                        "</Measurement>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            measurements = Measurement.read_from_file(meas_file)
            repeated_measurements = Measurement.read_from_file(meas_file)

        self.assertEqual(1, len(measurements))
        self.assertEqual(1, len(repeated_measurements))
        self.assertIsNot(measurements[0], repeated_measurements[0])
        repeated_measurements[0].value = 1.0
        self.assertEqual(345.6, measurements[0].value)

    def test_measurement_objects_use_slots_and_support_dataclass_replace(self):
        from dataclasses import replace
        from secore.ac_se import Measurement

        self.assertTrue(hasattr(Measurement, "__slots__"))
        measurement = Measurement(1, "m1", "ACNode", "n1", "V", 1.0, True, 2.0)
        updated = replace(measurement, value=3.0)

        self.assertEqual(2.0, measurement.value)
        self.assertEqual(3.0, updated.value)

    def test_efile_factory_skips_generic_conversion_for_text_columns(self):
        import efile_read

        class_name = "FastEfileFactoryText"
        data = {
            class_name: {
                "header_list": ["idx", "name", "value"],
                "data": [{"idx": "7", "name": "load_1", "value": "3.5"}],
            }
        }
        original_convert = efile_read._convert_cell

        def fail_text_conversion(value):
            if value == "load_1":
                raise AssertionError("text columns should not use generic numeric conversion")
            return original_convert(value)

        efile_read._convert_cell = fail_text_conversion
        try:
            model = efile_read.efile_factory(data)
        finally:
            efile_read._convert_cell = original_convert

        row = getattr(model, class_name)[0]
        self.assertEqual(7, row.idx)
        self.assertEqual("load_1", row.name)
        self.assertEqual(3.5, row.value)

    def test_efile_factory_uses_typed_numeric_converters(self):
        import efile_read

        class_name = "FastEfileFactoryTyped"
        data = {
            class_name: {
                "header_list": ["idx", "name", "value"],
                "data": [{"idx": "7", "name": "load_1", "value": "3.5"}],
            }
        }
        original_convert = efile_read._convert_cell

        def fail_generic_conversion(value):
            raise AssertionError(f"typed E-file factory should not call _convert_cell for {value!r}")

        efile_read._convert_cell = fail_generic_conversion
        try:
            model = efile_read.efile_factory(data)
        finally:
            efile_read._convert_cell = original_convert

        row = getattr(model, class_name)[0]
        self.assertEqual(7, row.idx)
        self.assertEqual("load_1", row.name)
        self.assertEqual(3.5, row.value)

    def test_efile_factory_infers_column_types_from_first_nonempty_sample(self):
        import efile_read

        class_name = "FastEfileFactoryInference"
        data = {
            class_name: {
                "header_list": ["idx", "name", "value"],
                "data": [
                    {"idx": str(idx), "name": f"load_{idx}", "value": f"{idx}.5"}
                    for idx in range(100)
                ],
            }
        }
        original_kind = efile_read._numeric_cell_kind
        calls = 0

        def counted_kind(value):
            nonlocal calls
            calls += 1
            return original_kind(value)

        efile_read._numeric_cell_kind = counted_kind
        try:
            model = efile_read.efile_factory(data)
        finally:
            efile_read._numeric_cell_kind = original_kind

        self.assertLessEqual(calls, 3)
        self.assertEqual(100, len(getattr(model, class_name)))

    def test_efile_factory_integer_inferred_column_accepts_float_text(self):
        import efile_read

        class_name = "FastEfileFactoryMixedNumeric"
        data = {
            class_name: {
                "header_list": ["idx", "value"],
                "data": [
                    {"idx": "1", "value": "0"},
                    {"idx": "2", "value": "0.5"},
                ],
            }
        }

        model = efile_read.efile_factory(data)
        rows = getattr(model, class_name)

        self.assertEqual(0, rows[0].value)
        self.assertEqual(0.5, rows[1].value)

    def test_efile_factory_generates_direct_initializers(self):
        import efile_read

        cls = efile_read._class_factory(
            "FastEfileFactoryDirectInit",
            ["idx", "name", "value"],
            [efile_read._safe_int_cell, efile_read._identity_cell, efile_read._safe_float_cell],
        )

        self.assertEqual((), cls.__init__.__code__.co_freevars)
        row = cls({"idx": "7", "name": "load_1", "value": "3.5"})
        self.assertEqual(7, row.idx)
        self.assertEqual("load_1", row.name)
        self.assertEqual(3.5, row.value)

    def test_efile_factory_inlines_builtin_column_converters(self):
        import efile_read

        cls = efile_read._class_factory(
            "FastEfileFactoryInlineConverters",
            ["idx", "name", "value"],
            [efile_read._safe_int_cell, efile_read._identity_cell, efile_read._safe_float_cell],
        )
        self.assertFalse(any(name.startswith("_converter_") for name in cls.__init__.__code__.co_names))
        original_int = efile_read._safe_int_cell
        original_float = efile_read._safe_float_cell

        def reject_converter(value):
            raise AssertionError(f"generated initializer should inline converter for {value!r}")

        efile_read._safe_int_cell = reject_converter
        efile_read._safe_float_cell = reject_converter
        try:
            row = cls({"idx": "7", "name": "load_1", "value": "3.5"})
            mixed = cls({"idx": "7.25", "name": "load_2", "value": "text"})
        finally:
            efile_read._safe_int_cell = original_int
            efile_read._safe_float_cell = original_float

        self.assertEqual(7, row.idx)
        self.assertEqual("load_1", row.name)
        self.assertEqual(3.5, row.value)
        self.assertEqual(7.25, mixed.idx)
        self.assertEqual("text", mixed.value)

    def test_efile_row_factory_builds_equivalent_objects_without_row_dicts(self):
        import efile_read

        data = {
            "FastEfileRows": {
                "table_name": "FastEfileRows",
                "header_list": ["idx", "name", "value"],
                "rows": [["7", "load_1", "3.5"], ["8", "load_2", ""]],
                "lv": 0,
            }
        }

        model = efile_read.efile_factory_from_rows(data)
        rows = model.FastEfileRows

        self.assertEqual(7, rows[0].idx)
        self.assertEqual("load_1", rows[0].name)
        self.assertEqual(3.5, rows[0].value)
        self.assertEqual(8, rows[1].idx)
        self.assertEqual("", rows[1].value)

    def test_ac_model_prunes_obsolete_direct_cache_helpers(self):
        import model.ac_model as ac_model

        obsolete_names = (
            "_AC_DIRECT_BUILDER_CACHE",
            "_AC_DIRECT_FLOAT_ATTRS",
            "_AC_DIRECT_INT_ATTRS",
            "_AC_DIRECT_TABLES",
            "_build_ac_model_direct",
            "_build_ac_model_from_cache",
            "_direct_cell_assignment",
            "_float_arg",
            "_generated_ac_table_builder",
            "_generated_direct_row_lines",
            "_header_attr_lines",
            "_header_index",
            "_int_arg",
            "_parse_direct_cell",
            "_row_dict",
            "_set_present_attrs",
            "_str_arg",
            "_to_float",
            "_to_int",
        )

        self.assertEqual([], [name for name in obsolete_names if hasattr(ac_model, name)])

    def test_ac_network_load_uses_ppc_file_loader(self):
        import secore.ac_se
        from secore.ac_se import ACStateEstimator

        previous_builder = getattr(secore.ac_se, "build_ac_ppc_from_network", None)
        previous_common_loader = secore.ac_se.build_ac_ppc_with_topology_from_e_file
        calls = []

        def counted_common_loader(path):
            calls.append(Path(path).name)
            return previous_common_loader(path)

        def reject_builder(*_args, **_kwargs):
            raise AssertionError("AC SE should use the shared E-to-PPC topology loader")

        secore.ac_se.build_ac_ppc_from_network = reject_builder
        secore.ac_se.build_ac_ppc_with_topology_from_e_file = counted_common_loader
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            )
        finally:
            if previous_builder is None:
                del secore.ac_se.build_ac_ppc_from_network
            else:
                secore.ac_se.build_ac_ppc_from_network = previous_builder
            secore.ac_se.build_ac_ppc_with_topology_from_e_file = previous_common_loader

        self.assertEqual(["ieee39.e"], calls)
        self.assertTrue(estimator.nodes)
        self.assertTrue(hasattr(estimator.network, "ppc"))
        self.assertTrue(hasattr(estimator.network, "topology"))
        self.assertIn("_topology_arrays", estimator.network.ppc)

    def test_load_network_returns_ppc_namespace_without_acpowernetwork_devices(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_facade_builder = ac_se._build_ac_se_network_from_ppc_dict

        def reject_facade_builder(*_args, **_kwargs):
            raise AssertionError("_load_network should return a PPC namespace without building ACPowerNetwork devices")

        ac_se._build_ac_se_network_from_ppc_dict = reject_facade_builder
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
                prepare_active_measurements=False,
            )
        finally:
            ac_se._build_ac_se_network_from_ppc_dict = original_facade_builder

        self.assertEqual("ac_ppc_v1", estimator.network.ppc["format"])
        self.assertEqual({"ppc", "base", "topology"}, set(vars(estimator.network)))
        self.assertIs(estimator.network.topology, estimator.network.ppc["_topology_arrays"])
        self.assertFalse(hasattr(estimator.network, "branches"))
        self.assertFalse(hasattr(estimator.network, "generators"))
        self.assertFalse(hasattr(estimator.network, "loads"))
        self.assertFalse(hasattr(estimator.network, "nodes"))

    def test_refresh_active_measurements_reuses_all_active_measurement_table(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        measurements = MeasurementList(
            [
                Measurement(1, "m1", "ACNode", "n1", "V", 1.0, True, 1.0),
                Measurement(2, "m2", "ACLoad", "l1", "P_LOAD", 1.0, True, 0.2),
            ]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.measurements = measurements
        estimator.n_state = 1
        estimator._branch_transformer_vector_plan_cache = {}
        estimator._simple_jacobian_plan_cache = {}
        estimator._zero_current_vector_plan_cache = {}
        estimator._generator_measurement_plan_cache = {}
        estimator._balance_measurement_plan_cache = {}

        def empty_plan(active_measurements):
            return {"handled_mask": np.zeros(len(active_measurements), dtype=bool)}

        def full_plan(active_measurements):
            return {"handled_mask": np.ones(len(active_measurements), dtype=bool)}

        estimator._branch_transformer_vector_plan = empty_plan
        estimator._simple_jacobian_plan = full_plan
        estimator._zero_current_vector_plan = empty_plan
        estimator._generator_measurement_plan = empty_plan
        estimator._balance_measurement_plan = empty_plan

        estimator._refresh_active_measurement_indexes()

        self.assertIs(estimator.active_measurements, measurements)
        self.assertIs(estimator.active_measurement_table, measurements.table)
        np.testing.assert_allclose(estimator.active_z, measurements.table.value)

    def test_prepare_preserves_provided_measurement_list_cache(self):
        from secore.ac_se import ACStateEstimator

        loader = ACStateEstimator.__new__(ACStateEstimator)
        network = loader._load_network(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e")
        measurements = ACStateEstimator._load_measurements(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas")
        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            network=network,
            measurements=measurements,
            flat_start=True,
            prepare_active_measurements=False,
        )

        self.assertIs(estimator.measurements, measurements)
        self.assertIsNotNone(estimator.measurements.table)

    def test_active_balance_plan_uses_type_index(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        node = type("Node", (), {"idx": 1, "name": "n1"})()
        measurements = MeasurementList(
            [
                Measurement(1, "bad", "ACPowerBalance", "missing", "P_BALANCE", 1.0, True, 0.0),
                Measurement(2, "good", "ACPowerBalance", "n1", "P_BALANCE", 1.0, True, 0.0),
            ]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.active_measurements = measurements
        estimator.active_measurement_table = measurements.table
        estimator.node_by_name = {"n1": node}
        estimator.node_pos = {1: 0}
        estimator.nodes = [node]
        estimator._y_row_nodes = [np.array([], dtype=np.int32)]
        estimator._y_row_y_conj = [np.array([], dtype=np.complex128)]
        estimator._active_measurement_rows_for_types = lambda device_types: np.array([1], dtype=np.int64)

        plan = estimator._build_balance_measurement_plan(measurements)

        np.testing.assert_array_equal(plan["rows"], np.array([1]))

    def test_active_balance_plan_avoids_small_full_allocations(self):
        import secore.ac_se as ac_se
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator

        node = type("Node", (), {"idx": 1, "name": "n1"})()
        measurements = MeasurementList(
            [
                Measurement(1, "p", "ACPowerBalance", "n1", "P_BALANCE", 1.0, True, 0.0),
                Measurement(2, "q", "ACPowerBalance", "n1", "Q_BALANCE", 1.0, True, 0.0),
            ]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.active_measurements = measurements
        estimator.active_measurement_table = measurements.table
        estimator.node_by_name = {"n1": node}
        estimator.node_pos = {1: 0}
        estimator.nodes = [node]
        estimator._y_row_nodes = [np.array([0, 1], dtype=np.int32)]
        estimator._y_row_y_conj = [np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)]
        estimator._active_measurement_rows_for_types = lambda device_types: np.array([0, 1], dtype=np.int64)

        original_full = ac_se.np.full

        def reject_full(*_args, **_kwargs):
            raise AssertionError("active balance plan should batch y-term allocation")

        ac_se.np.full = reject_full
        try:
            plan = estimator._build_balance_measurement_plan(measurements)
        finally:
            ac_se.np.full = original_full

        np.testing.assert_array_equal(plan["balance_pos"], np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(plan["y_balance"], np.array([0, 0], dtype=np.int32))

    def test_ac_state_estimator_does_not_use_profile_call_wrapper(self):
        source = (ROOT_DIR / "src" / "hybrid_power_system_analysis" / "secore" / "ac_se.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_profile_call", source)

    def test_normalize_model_named_units_reuses_current_base_per_node(self):
        import unit_system
        from efile_read import efile_factory

        data = {
            "PowerBase": {
                "header_list": ["p_base", "u_scale", "p_scale", "i_scale"],
                "data": [{"p_base": "100000", "u_scale": "1000", "p_scale": "1000", "i_scale": "1"}],
            },
            "ACNode": {
                "header_list": ["idx", "name", "vbase", "voltage", "angle", "run_stat"],
                "data": [{"idx": "1", "name": "n1", "vbase": "345", "voltage": "345", "angle": "0", "run_stat": "1"}],
            },
            "ACBranch": {
                "header_list": ["idx", "name", "i_node", "j_node", "r", "x", "b", "run_stat", "i_c", "j_c"],
                "data": [
                    {"idx": str(idx), "name": f"br_{idx}", "i_node": "1", "j_node": "1", "r": "0", "x": "0", "b": "0", "run_stat": "1", "i_c": "10", "j_c": "20"}
                    for idx in range(20)
                ],
            },
        }
        model = efile_factory(data)
        original = unit_system.ac_current_base_ka
        calls = 0

        def counted_current_base(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        unit_system.ac_current_base_ka = counted_current_base
        try:
            unit_system.normalize_model_named_units(model)
        finally:
            unit_system.ac_current_base_ka = original

        self.assertEqual(1, calls)

    def test_check_topo_scans_voltage_consistency_once(self):
        from model.ac_model import ACPowerNetwork

        net = ACPowerNetwork()
        n1 = net.add_node(1, 1.0, 1.0, 0.0)
        n1.name = "n1"
        n2 = net.add_node(2, 1.0, 1.0, 0.0)
        n2.name = "n2"
        n3 = net.add_node(3, 1.0, 1.0, 0.0)
        n3.name = "n3"
        n4 = net.add_node(4, 1.0, 1.0, 0.0)
        n4.name = "n4"
        br1 = net.add_branch(1, 1, 2, 0.0)
        br1.name = "br1"
        br2 = net.add_branch(2, 3, 4, 0.0)
        br2.name = "br2"
        gen1 = net.add_generator(1, 1, "SLACK", 0.0, 0.0, 1.0)
        gen1.name = "g1"
        gen2 = net.add_generator(2, 3, "SLACK", 0.0, 0.0, 1.0)
        gen2.name = "g2"
        net.format_assoc()
        net.topo()

        class CountedBranches(list):
            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        counted = CountedBranches(net.branches)
        counted.iterations = 0
        net.branches = counted

        net.check_topo()

        self.assertLessEqual(counted.iterations, 2)

    def test_estimator_load_network_skips_topology_check(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_check_topo = ac_se.ACPowerNetwork.check_topo

        def reject_check_topo(self):
            raise AssertionError("main state-estimation load path should not call check_topo")

        ac_se.ACPowerNetwork.check_topo = reject_check_topo
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.ACPowerNetwork.check_topo = original_check_topo

        self.assertTrue(estimator.nodes)

    def test_estimator_load_network_uses_ppc_topology_arrays_without_object_topology(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_prepare_topology = ac_se.network_topology.prepare_ac_topology

        def reject_object_topology(*_args, **_kwargs):
            raise AssertionError("AC SE array load should apply ppc topology arrays directly")

        ac_se.network_topology.prepare_ac_topology = reject_object_topology
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.network_topology.prepare_ac_topology = original_prepare_topology

        self.assertTrue(estimator.nodes)
        self.assertIsNotNone(getattr(estimator.network, "topology", None))

    def test_measurement_unit_conversion_uses_precomputed_node_scales(self):
        from secore.ac_se import ACStateEstimator, Measurement

        class Device:
            def __init__(self, i_node=None, j_node=None, node=None, idx=None):
                self.i_node = i_node
                self.j_node = j_node
                self.node = node
                self.idx = idx if idx is not None else node

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.p_base = 100.0
        estimator.p_base_kW = 100.0
        estimator.i_scale = 1.0
        estimator.u_scale = 1000.0
        estimator.node_by_name = {"n1": Device(node=1, idx=1), "n2": Device(node=2, idx=2)}
        estimator.branch_by_name = {"br": Device(i_node=1, j_node=2)}
        estimator.transformer_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.switch_by_name = {}
        estimator.generator_by_name = {"gen": Device(node=1)}
        estimator.load_by_name = {"load": Device(node=2)}
        estimator.network = type(
            "Network",
            (),
            {"node_dict": {1: type("Node", (), {"vbase": 1.0})(), 2: type("Node", (), {"vbase": 2.0})()}},
        )()
        estimator.measurements = [
            Measurement(1, "v1", "ACNode", "n1", "V", 1.0, True, 1000.0),
            Measurement(2, "i_from", "ACBranch", "br", "I_FROM", 1.0, True, 10.0),
            Measurement(3, "i_to", "ACBranch", "br", "I_TO", 1.0, True, 20.0),
            Measurement(4, "i_gen", "ACGenerator", "gen", "I_GEN", 1.0, True, 10.0),
            Measurement(5, "i_load", "ACLoad", "load", "I_LOAD", 1.0, True, 20.0),
        ]
        estimator._node_current_base = lambda node_idx: (_ for _ in ()).throw(
            AssertionError("_convert_measurements_to_pu should use precomputed current scales")
        )
        estimator._voltage_file_base = lambda node_idx: (_ for _ in ()).throw(
            AssertionError("_convert_measurements_to_pu should use precomputed voltage scales")
        )

        estimator._convert_measurements_to_pu()

        self.assertAlmostEqual(1.0, estimator.measurements[0].value)

    def test_measurement_unit_conversion_bypasses_per_measurement_terminal_dispatch(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        estimator.measurements = estimator._load_measurements(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas")
        estimator._disable_angle_measurements()
        estimator._disable_unavailable_measurements()

        def reject_terminal_scale(*_args, **_kwargs):
            raise AssertionError("unit conversion should use precomputed terminal scale maps")

        estimator._terminal_measurement_scale = reject_terminal_scale
        estimator._convert_measurements_to_pu()

        self.assertTrue(estimator._active_measurement_key_cache)

    def test_prepare_fuses_file_measurement_load_and_normalization(self):
        from secore.ac_se import ACStateEstimator
        from model.meas_model import TableBackedMeasurementList

        original_convert = ACStateEstimator._convert_measurements_to_pu

        def reject_convert(self):
            raise AssertionError("file-backed prepare should load normalized measurements directly")

        ACStateEstimator._convert_measurements_to_pu = reject_convert
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ACStateEstimator._convert_measurements_to_pu = original_convert

        self.assertTrue(getattr(estimator.measurements, "normalized", False))
        self.assertIsInstance(estimator.measurements, TableBackedMeasurementList)
        self.assertEqual("meas_ppc_v1", estimator.meas_ppc["format"])
        self.assertTrue(estimator.meas_ppc["normalized"])
        self.assertTrue(estimator.estimate(final_diagnostics=False).converged)

    def test_file_backed_measurement_normalization_updates_meas_ppc_directly(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_sync = ac_se.sync_meas_ppc_from_measurement_table

        def reject_table_sync(*_args, **_kwargs):
            raise AssertionError("file-backed AC SE should update meas PPC directly")

        ac_se.sync_meas_ppc_from_measurement_table = reject_table_sync
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.sync_meas_ppc_from_measurement_table = original_sync

        self.assertTrue(getattr(estimator.measurements, "normalized", False))
        self.assertTrue(estimator.meas_ppc["normalized"])
        self.assertTrue(estimator.estimate_result is None)

    def test_power_balance_constraints_extend_measurement_table_without_row_objects(self):
        import secore.ac_se as ac_se
        from model.meas_model import DEVICE_TYPE_CODES, MEAS_STATUS_NORMAL, TableBackedMeasurementList
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        before_count = len(estimator.measurements)
        expected_added = 2 * len(estimator.nodes)
        original_measurement = ac_se.Measurement

        class RejectMeasurement:
            def __new__(cls, *_args, **_kwargs):
                raise AssertionError("power-balance constraints should be appended as table rows")

        ac_se.Measurement = RejectMeasurement
        try:
            estimator._add_power_balance_constraint_measurements()
        finally:
            ac_se.Measurement = original_measurement

        self.assertIsInstance(estimator.measurements, TableBackedMeasurementList)
        self.assertEqual(before_count + expected_added, len(estimator.measurements))
        self.assertEqual(0, list.__len__(estimator.measurements))
        tail = estimator.measurements.table
        balance_rows = np.arange(before_count, before_count + expected_added)
        np.testing.assert_array_equal(
            tail.device_type_code[balance_rows],
            np.full(expected_added, DEVICE_TYPE_CODES["ACPowerBalance"], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            tail.status_code[balance_rows],
            np.full(expected_added, MEAS_STATUS_NORMAL, dtype=np.int16),
        )

    def test_standard_measurement_file_uses_vectorized_parser(self):
        import secore.ac_se as ac_se
        from model.meas_model import Measurement, TableBackedMeasurementList

        self.assertFalse(hasattr(ac_se, "_read_standard_measurement_lines"))
        measurements = ac_se._read_measurements_direct(
            ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            Measurement,
            table_only=True,
        )

        self.assertIsInstance(measurements, TableBackedMeasurementList)
        self.assertIsNotNone(measurements.table.rows_by_device_type_code)
        self.assertEqual(0, int(measurements.table.idx[0]))
        self.assertEqual("ACNode", str(measurements.table.device_type[0]))
        self.assertEqual("V", str(measurements.table.meas_type[0]))

    def test_ac_se_dead_legacy_helpers_are_not_exposed(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        for name in (
            "_decode_cached_bytes_array",
            "_decode_bytes_array",
            "_measurement_list_from_table_arrays",
            "_build_ac_se_network_from_ppc",
        ):
            self.assertFalse(hasattr(ac_se, name), name)

        for name in (
            "_measurement_rows_for_types",
            "_measurement_row_indexes_for_types",
            "_measurement_row_count",
            "_active_angle_measurement_counts",
            "_add_redundant_observability_pseudo_measurements",
            "_unanchored_angle_state_indices",
            "_node_angle_in_reference_frame",
            "_load_power",
            "_load_power_arrays",
            "_load_power_totals",
            "_generator_power",
            "_generator_power_totals",
            "_bool_array",
            "_network_power_derivatives",
            "_network_power_derivative_rows",
            "_load_power_derivatives",
            "_current_from_power_derivatives",
        ):
            self.assertFalse(hasattr(ACStateEstimator, name), name)

    def test_table_backed_measurement_tail_concatenation_skips_sequence_getitem(self):
        import model.meas_model as meas_model
        from model.meas_model import Measurement, TableBackedMeasurementList, measurement_table_from_measurements

        base_table = measurement_table_from_measurements(
            [
                Measurement(1, "v1", "ACNode", "n1", "V", 1.0, True, 1.0),
                Measurement(2, "v2", "ACNode", "n2", "V", 1.0, True, 1.0),
            ],
            device_type_codes={"ACNode": 1, "ACPowerBalance": 2},
            angle_measurement_types=frozenset(),
        )
        measurements = TableBackedMeasurementList(base_table, normalized=True)
        measurements.append(Measurement(3, "pb", "ACPowerBalance", "n1", "P_BALANCE", 1.0, True, 0.0))

        original_getitem = TableBackedMeasurementList.__getitem__
        calls = 0

        def counted_getitem(self, index):
            nonlocal calls
            calls += 1
            return original_getitem(self, index)

        TableBackedMeasurementList.__getitem__ = counted_getitem
        try:
            table = meas_model.measurement_table_from_measurements(
                measurements,
                device_type_codes={"ACNode": 1, "ACPowerBalance": 2},
                angle_measurement_types=frozenset(),
            )
        finally:
            TableBackedMeasurementList.__getitem__ = original_getitem

        self.assertEqual(0, calls)
        self.assertEqual(3, int(table.idx.size))

    def test_estimator_skips_separate_unavailable_measurement_scan(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original = ac_se.ACStateEstimator._disable_unavailable_measurements

        def reject_scan(self):
            raise AssertionError("availability filtering should be folded into unit conversion")

        ac_se.ACStateEstimator._disable_unavailable_measurements = reject_scan
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.ACStateEstimator._disable_unavailable_measurements = original

        self.assertTrue(estimator.active_measurements)

    def test_estimator_skips_separate_measurement_summary_scan_after_conversion(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original = ac_se.ACStateEstimator._refresh_measurement_summary_cache

        def reject_summary_scan(self):
            raise AssertionError("measurement summary should be populated during unit conversion")

        ac_se.ACStateEstimator._refresh_measurement_summary_cache = reject_summary_scan
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.ACStateEstimator._refresh_measurement_summary_cache = original

        self.assertTrue(estimator.active_measurements)

    def test_pseudo_initialization_reads_summary_cache_without_key_copies(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_active_devices = ac_se.ACStateEstimator._active_device_keys
        original_active_measurements = ac_se.ACStateEstimator._active_measurement_keys

        def reject_active_devices(self):
            raise AssertionError("pseudo initialization should read the cached key set directly")

        def reject_active_measurements(self):
            raise AssertionError("pseudo initialization should read the cached key set directly")

        ac_se.ACStateEstimator._active_device_keys = reject_active_devices
        ac_se.ACStateEstimator._active_measurement_keys = reject_active_measurements
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
                flat_start=True,
            )
        finally:
            ac_se.ACStateEstimator._active_device_keys = original_active_devices
            ac_se.ACStateEstimator._active_measurement_keys = original_active_measurements

        self.assertTrue(estimator.active_measurements)

    def test_ieee3k_flat_start_does_not_add_angle_pseudos(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee300.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee300.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas",
            flat_start=True,
            max_iter=25,
        )

        theta, _voltage = estimator._unpack_state(estimator.initial_state())

        np.testing.assert_allclose(theta, 0.0)

    def test_reference_node_uses_highest_degree_node_with_valid_voltage_measurement(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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

    def test_nonflat_start_runs_measurement_seeded_power_flow(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_seed = getattr(ac_se.ACStateEstimator, "_run_power_flow_seed", None)
        calls = []

        def fake_seed(network, _params, _e_file):
            seed_ppc = ACStateEstimator._power_flow_seed_ppc_from_network(network)
            bus = seed_ppc["bus"]
            bus_idx = network.ppc["bus_cols"]["idx"]
            voltage_col = network.ppc["bus_cols"]["voltage"]
            angle_col = network.ppc["bus_cols"]["angle"]
            bus_2_row = int(np.flatnonzero(bus[:, bus_idx] == 2)[0])
            self.assertAlmostEqual(119.053271 / 115.0, float(bus[bus_2_row, voltage_col]))
            calls.append(True)
            network.ppc["bus"][:, voltage_col] = 1.11
            network.ppc["bus"][:, angle_col] = 0.05
            return True

        ac_se.ACStateEstimator._run_power_flow_seed = staticmethod(fake_seed)
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee300.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee300.meas",
                flat_start=False,
            )
        finally:
            if original_seed is not None:
                ac_se.ACStateEstimator._run_power_flow_seed = staticmethod(original_seed)

        self.assertFalse(estimator.flat_start)
        self.assertTrue(calls)
        _theta, voltage = estimator._unpack_state(estimator.initial_state())
        voltage_state = voltage[estimator.voltage_state_pos]
        np.testing.assert_allclose(voltage_state, 1.11)
        self.assertFalse(any(meas.name == "pseudo_angle_bus_9025" for meas in estimator.measurements))

    def test_nonflat_start_uses_array_mode_power_flow_seed_when_ppc_is_available(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_calc = ac_se.ACPowerFlowCalc
        original_sync = ACStateEstimator._sync_ac_network_to_ppc
        calls = []

        def reject_full_sync(*_args, **_kwargs):
            raise AssertionError("cached seed rows should update seed ppc without full network sync")

        class FakePowerFlowCalc:
            def __init__(self, model, **_kwargs):
                self.model = model
                self.ppc = model if isinstance(model, dict) else getattr(model, "ppc", None)
                self.converged = False
                self.iterations = 0
                self.normF = 0.0
                calls.append(isinstance(model, dict))

            def prepare(self):
                bus_2_row = 1
                self.testcase.assertAlmostEqual(
                    119.053271 / 115.0,
                    float(self.ppc["bus"][bus_2_row, ac_se.BUS_COLS["voltage"]]),
                )

            def run(self):
                self.testcase.assertTrue(
                    getattr(self, "skip_lf_result", False),
                    "SE LF seed should skip detailed LFResult construction",
                )
                self.converged = True
                self.iterations = 1
                self.result = {
                    key: value.copy()
                    for key, value in self.ppc.items()
                    if isinstance(value, np.ndarray)
                }
                self.result["bus"][:, ac_se.BUS_COLS["voltage"]] = 1.12
                self.result["bus"][:, ac_se.BUS_COLS["angle"]] = 0.06
                return 0

        FakePowerFlowCalc.testcase = self
        ac_se.ACPowerFlowCalc = FakePowerFlowCalc
        ACStateEstimator._sync_ac_network_to_ppc = staticmethod(reject_full_sync)
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee300.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee300.meas",
                flat_start=False,
            )
        finally:
            ac_se.ACPowerFlowCalc = original_calc
            ACStateEstimator._sync_ac_network_to_ppc = staticmethod(original_sync)

        self.assertEqual([True], calls)
        self.assertTrue(estimator.power_flow_seed_converged)
        _theta, voltage = estimator._unpack_state(estimator.initial_state())
        np.testing.assert_allclose(voltage[estimator.voltage_state_pos], 1.12)

    def test_power_flow_seed_uses_cached_seed_rows(self):
        from secore.ac_se import ACStateEstimator

        class Device:
            pass

        class FailingMeasurements(list):
            def __iter__(self):
                raise AssertionError("power-flow seed should not rescan all measurements")

        node = Device()
        node.idx = 1
        node.name = "n1"
        node.voltage = 1.0
        node.angle = 0.0
        gen = Device()
        gen.node = 1
        gen.p_set = 0.0
        gen.q_set = 0.0
        gen.p = 0.0
        gen.q = 0.0
        load = Device()
        load.node = 1
        load.pbase = 1.0
        load.qbase = 1.0
        load.pv0 = 0.0
        load.pv1 = 0.0
        load.pv2 = 0.0
        load.qv0 = 0.0
        load.qv1 = 0.0
        load.qv2 = 0.0
        load.p = 0.0
        load.q = 0.0

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.voltage_floor = 0.1
        estimator.node_by_name = {"n1": node}
        estimator.node_by_idx = {1: node}
        estimator.generator_by_name = {"g1": gen}
        estimator.load_by_name = {"l1": load}
        estimator.network = type("Network", (), {"node_dict": {1: node}, "nodes": [node]})()
        estimator.measurements = FailingMeasurements()
        estimator._power_flow_seed_rows = [
            ("ACNode", "n1", "V", 1.05),
            ("ACGenerator", "g1", "P_GEN", 0.7),
            ("ACGenerator", "g1", "Q_GEN", 0.2),
            ("ACLoad", "l1", "P_LOAD", 0.3),
            ("ACLoad", "l1", "Q_LOAD", 0.1),
        ]

        estimator._apply_measurement_seed_to_network()

        self.assertAlmostEqual(1.05, node.voltage)
        self.assertAlmostEqual(0.7, gen.p_set)
        self.assertAlmostEqual(0.2, gen.q_set)
        self.assertAlmostEqual(0.3, load.pv0)
        self.assertAlmostEqual(0.1, load.qv0)

    def test_array_power_flow_seed_defers_object_seed_application(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.network = type("Network", (), {"_array_model": {"format": "ac_ppc_v1"}})()
        estimator.measurements = []
        estimator._power_flow_seed_rows = [("ACNode", "n1", "V", 1.05)]
        estimator._apply_power_flow_seed_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("array-mode seed should be applied to ppc, not object network")
        )

        estimator._apply_measurement_seed_to_network()

        self.assertEqual(tuple(estimator._power_flow_seed_rows), estimator.network._se_power_flow_seed_rows)

    def test_lightweight_power_flow_seed_writeback_skips_terminal_device_scans(self):
        from types import SimpleNamespace

        import numpy as np
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        class NonIterableDevices:
            def __bool__(self):
                return True

            def __iter__(self):
                raise AssertionError("lightweight seed writeback should not scan terminal devices")

        bus = np.zeros((1, len(ac_se.BUS_COLS)), dtype=np.float64)
        bus[0, ac_se.BUS_COLS["idx"]] = 1
        bus[0, ac_se.BUS_COLS["voltage"]] = 1.08
        bus[0, ac_se.BUS_COLS["angle"]] = 0.02
        gen = np.zeros((1, len(ac_se.GEN_COLS)), dtype=np.float64)
        gen[0, ac_se.GEN_COLS["idx"]] = 1
        gen[0, ac_se.GEN_COLS["p"]] = 0.7
        gen[0, ac_se.GEN_COLS["q"]] = 0.2
        load = np.zeros((1, len(ac_se.LOAD_COLS)), dtype=np.float64)
        load[0, ac_se.LOAD_COLS["idx"]] = 1
        load[0, ac_se.LOAD_COLS["p"]] = 0.3
        load[0, ac_se.LOAD_COLS["q"]] = 0.1
        shunt = np.zeros((1, len(ac_se.SHUNT_COLS)), dtype=np.float64)
        shunt[0, ac_se.SHUNT_COLS["idx"]] = 1
        shunt[0, ac_se.SHUNT_COLS["q"]] = -0.05
        ppc = {
            "bus": bus,
            "gen": gen,
            "load": load,
            "shunt": shunt,
            "branch": np.zeros((1, len(ac_se.BRANCH_COLS)), dtype=np.float64),
            "transformer": np.zeros((1, len(ac_se.TRANSFORMER_COLS)), dtype=np.float64),
            "zero_branch": np.zeros((1, len(ac_se.ZERO_BRANCH_COLS)), dtype=np.float64),
            "switch": np.zeros((1, len(ac_se.SWITCH_COLS)), dtype=np.float64),
            "break": np.zeros((1, len(ac_se.BREAK_COLS)), dtype=np.float64),
        }
        node = SimpleNamespace(idx=1, voltage=1.0, angle=0.0)
        bus_obj = SimpleNamespace(nodes=[node], voltage=1.0, angle=0.0)
        gen_obj = SimpleNamespace(idx=1, p=0.0, q=0.0, current=0.0)
        load_obj = SimpleNamespace(idx=1, p=0.0, q=0.0, current=0.0)
        shunt_obj = SimpleNamespace(idx=1, p=0.0, q=0.0, current=0.0)
        network = SimpleNamespace(
            _se_lightweight=True,
            nodes=[node],
            buses=[bus_obj],
            generators=[gen_obj],
            loads=[load_obj],
            shunt_compensators=[shunt_obj],
            branches=NonIterableDevices(),
            transformers=NonIterableDevices(),
            zero_branches=NonIterableDevices(),
            switches=NonIterableDevices(),
            breakers=NonIterableDevices(),
        )

        ACStateEstimator._apply_power_flow_seed_ppc_to_network(network, ppc)

        self.assertAlmostEqual(1.08, node.voltage)
        self.assertAlmostEqual(0.02, node.angle)
        self.assertAlmostEqual(0.7, gen_obj.p)
        self.assertAlmostEqual(0.2, gen_obj.q)
        self.assertAlmostEqual(0.3, load_obj.p)
        self.assertAlmostEqual(0.1, load_obj.q)
        self.assertAlmostEqual(-0.05, shunt_obj.q)

    def test_targeted_zero_current_pseudo_uses_to_side_when_from_side_exists(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )
        device_name = next(iter(estimator.zero_branch_by_name))
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = {
            ("ACZeroBranch", device_name, "P_FROM"),
            ("ACZeroBranch", device_name, "Q_FROM"),
        }
        existing_names = set()
        target_col = next(
            idx
            for idx, meta in enumerate(estimator.state_meta)
            if meta.device_type == "ACZeroBranch" and meta.device_name == device_name and meta.component == "re"
        )

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            target_col,
            existing_keys,
            existing_names,
            2,
        )

        self.assertEqual(2, added)
        self.assertIn(("ACZeroBranch", device_name, "P_TO"), existing_keys)
        self.assertIn(("ACZeroBranch", device_name, "Q_TO"), existing_keys)

    def test_targeted_node_voltage_state_adds_pseudo_measurement(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "no_real_voltage.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 p_bad ACLoad load_1 P_LOAD 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = set()
        existing_names = set()
        target_col, target_meta = next(
            (idx, meta)
            for idx, meta in enumerate(estimator.state_meta)
            if meta.kind == "voltage" and meta.device_type == "ACNode"
        )

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            target_col,
            existing_keys,
            existing_names,
            1,
        )

        self.assertEqual(1, added)
        self.assertIn(("ACNode", target_meta.device_name, "V"), existing_keys)

    def test_ieee3w_adds_rank_restoring_pseudos_without_node_voltage_or_angles(self):
        from secore.ac_se import ACStateEstimator
        from secore.se_math import ANGLE_MEASUREMENT_TYPES

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3w.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee3w.meas",
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

    def test_sparse_jacobian_builder_caches_fixed_pattern_chunk_slot_plan(self):
        from secore.se_math import SparseJacobianBuilder

        builder = SparseJacobianBuilder((2, 3))
        builder._assume_fixed_pattern = True
        rows = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        cols = np.array([0, 2, 0, 0, 2], dtype=np.int32)
        builder.add_many(rows, cols, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        first = builder.to_csr()
        np.testing.assert_allclose(first.toarray(), np.array([[1.0, 0.0, 2.0], [7.0, 0.0, 5.0]]))

        builder.reset()
        builder.add_many(rows, cols, np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        second = builder.to_csr()

        self.assertTrue(builder._cached_chunk_slot_plans)
        np.testing.assert_allclose(second.toarray(), np.array([[10.0, 0.0, 20.0], [70.0, 0.0, 50.0]]))

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
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        pseudo = [meas for meas in estimator.active_measurements if meas.name.startswith("pseudo_")]
        pseudo_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in pseudo}

        self.assertIn(("ACGenerator", "gen_30_0", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_30_0", "Q_GEN"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "ACGenerator"
            and meas.device_name == "gen_30_0"
            and meas.meas_type == "P_GEN"
        )
        self.assertAlmostEqual(gen_p.value, 2.5)

    def test_ac_unmetered_load_pseudo_measurements_cover_all_unmetered_loads(self):
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
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=meas_file,
            )

        selected_loads = {
            load.name
            for load in estimator.load_by_name.values()
        }
        pseudo_loads = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACLoad"
            and meas.name.startswith(("pseudo_p_", "pseudo_q_", "pseudo_v_"))
        ]
        pseudo_load_devices = {meas.device_name for meas in pseudo_loads}
        pseudo_load_keys = {(meas.device_name, meas.meas_type) for meas in pseudo_loads}

        self.assertEqual(selected_loads, pseudo_load_devices)
        voltage_covered_loads = {"load_1"}
        self.assertEqual(3 * len(selected_loads) - len(voltage_covered_loads), len(pseudo_loads))
        for load_name in selected_loads:
            self.assertIn((load_name, "P_LOAD"), pseudo_load_keys)
            self.assertIn((load_name, "Q_LOAD"), pseudo_load_keys)
            if load_name in voltage_covered_loads:
                self.assertNotIn((load_name, "V_LOAD"), pseudo_load_keys)
            else:
                self.assertIn((load_name, "V_LOAD"), pseudo_load_keys)

    def test_ac_targeted_pseudo_uses_ratio_target_and_step_between_reanalysis(self):
        from model.meas_model import Measurement
        from secore.ac_se import ACStateEstimator, ObservabilityResult
        from secore.se_math import targeted_redundancy_count

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "voltage_only.meas"
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
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_10.e",
                meas_file=meas_file,
            )

        base_count = len(estimator.active_measurements)
        target_redundancy = base_count - estimator.n_state + 5
        estimator.targeted_pseudo_measurement_redundancy_ratio = target_redundancy / estimator.n_state
        estimator.targeted_pseudo_measurement_step = 2
        batches = []

        def observable_now():
            return ObservabilityResult(
                True,
                estimator.n_state,
                estimator.n_state,
                len(estimator.active_measurements),
                0,
                np.array([]),
                [],
            )

        def add_batch(_observability, max_add):
            batches.append(max_add)
            next_idx = max((meas.idx for meas in estimator.measurements), default=0) + 1
            for offset in range(max_add):
                estimator.measurements.append(
                    Measurement(
                        next_idx + offset,
                        f"pseudo_extra_test_{next_idx + offset}",
                        "ACNode",
                        "nd_0",
                        "V",
                        estimator.pseudo_measurement_weight,
                        True,
                        1.0,
                    )
                )
            estimator._refresh_active_measurement_indexes()
            return max_add

        estimator.observability_analysis = observable_now
        estimator._add_weak_direction_observability_pseudo_measurements = add_batch
        expected_redundancy = targeted_redundancy_count(
            estimator.n_state,
            estimator.targeted_pseudo_measurement_redundancy_ratio,
        )
        expected_added = expected_redundancy - (base_count - estimator.n_state)
        added = estimator._add_targeted_observability_pseudo_measurements()

        self.assertEqual(expected_added, added)
        self.assertEqual([2] * (expected_added // 2) + ([1] if expected_added % 2 else []), batches)
        self.assertEqual(base_count + expected_added, len(estimator.active_measurements))
        self.assertGreaterEqual(
            len(estimator.active_measurements),
            estimator.n_state + expected_redundancy,
        )

    def test_ac_weak_direction_candidates_include_voltage_and_branch_power_rows(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "no_real_voltage.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 p_bad ACLoad load_1 P_LOAD 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                flat_start=True,
                prepare_active_measurements=False,
            )

        keys = {
            (meas.device_type, meas.meas_type)
            for meas in estimator._observability_pseudo_candidate_measurements()
        }

        self.assertIn(("ACLoad", "V_LOAD"), keys)
        self.assertIn(("ACNode", "V"), keys)
        self.assertIn(("ACBranch", "P_FROM"), keys)
        self.assertIn(("ACBranch", "Q_FROM"), keys)

    def test_does_not_duplicate_existing_generator_or_load_power_measurements(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
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

    def test_adds_low_weight_pseudo_pq_measurements_for_unmetered_ac_topology_devices(self):
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
                        "# 3 p_sw_bad ACBreak sw_0_1 P_FROM 1.0 0 0",
                        "# 4 p_zbr_bad ACZeroBranch zbr_1_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        regular_pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_") and not meas.name.startswith("pseudo_obs_")
        }

        self.assertNotIn(("ACNode", "nd_1", "V"), regular_pseudo_keys)
        self.assertNotIn(("ACNode", "nd_2", "V"), regular_pseudo_keys)
        self.assertFalse(
            any(device_type == "ACNode" and meas_type == "V" for device_type, _name, meas_type in regular_pseudo_keys)
        )
        for meas_type in ("P_FROM", "Q_FROM"):
            self.assertIn(("ACBreak", "sw_0_1", meas_type), regular_pseudo_keys)
            self.assertIn(("ACZeroBranch", "zbr_1_2", meas_type), regular_pseudo_keys)
        self.assertNotIn(("ACBreak", "sw_0_1", "V_FROM"), regular_pseudo_keys)
        self.assertNotIn(("ACZeroBranch", "zbr_1_2", "V_FROM"), regular_pseudo_keys)
        self.assertNotIn(("ACBreak", "sw_0_1", "I_FROM"), regular_pseudo_keys)
        self.assertNotIn(("ACZeroBranch", "zbr_1_2", "I_FROM"), regular_pseudo_keys)

    def test_device_voltage_pseudo_is_skipped_when_peer_device_on_node_has_real_voltage(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "peer_device_voltage.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_load_31 ACLoad load_31 V_LOAD 1.0 1 345",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_") and not meas.name.startswith("pseudo_obs_")
        }

        self.assertIn(("ACGenerator", "gen_31_1", "P_GEN"), pseudo_keys)
        self.assertIn(("ACGenerator", "gen_31_1", "Q_GEN"), pseudo_keys)
        self.assertNotIn(("ACGenerator", "gen_31_1", "V_GEN"), pseudo_keys)

    def test_jacobian_uses_direct_derivatives_without_repeated_evaluation(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            flat_start=True,
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        active_devices = {meas.device_type for meas in estimator.active_measurements}
        self.assertIn("ACBreak", active_devices)
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            flat_start=True,
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        cached = estimator._simple_jacobian_plan_cache.get(id(estimator.active_measurements))
        self.assertIsNotNone(cached)
        self.assertIs(cached[0], estimator.active_measurements)

    def test_active_measurement_plans_use_dedicated_fast_path(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        class RejectCache(dict):
            def get(self, *args, **kwargs):
                raise AssertionError("active plan lookup should not touch generic cache")

        plan_cases = [
            (
                "_active_branch_transformer_vector_plan",
                "_branch_transformer_vector_plan",
                "_branch_transformer_vector_plan_cache",
            ),
            (
                "_active_zero_current_vector_plan",
                "_zero_current_vector_plan",
                "_zero_current_vector_plan_cache",
            ),
            (
                "_active_simple_jacobian_plan",
                "_simple_jacobian_plan",
                "_simple_jacobian_plan_cache",
            ),
            (
                "_active_balance_measurement_plan",
                "_balance_measurement_plan",
                "_balance_measurement_plan_cache",
            ),
            (
                "_active_generator_measurement_plan",
                "_generator_measurement_plan",
                "_generator_measurement_plan_cache",
            ),
        ]
        for active_attr, method_name, cache_attr in plan_cases:
            expected_plan = getattr(estimator, active_attr)
            setattr(estimator, cache_attr, RejectCache())

            plan = getattr(estimator, method_name)(estimator.active_measurements)

            self.assertIs(expected_plan, plan)

    def test_active_measurement_plans_use_measurement_plan_table(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator
        from secore.se_array_plan import build_measurement_plan_table as original_builder

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )
        for attr in (
            "_active_branch_transformer_vector_plan",
            "_active_simple_jacobian_plan",
            "_active_zero_current_vector_plan",
            "_active_generator_measurement_plan",
            "_active_balance_measurement_plan",
        ):
            setattr(estimator, attr, None)
        estimator._branch_transformer_vector_plan_cache = {}
        estimator._simple_jacobian_plan_cache = {}
        estimator._zero_current_vector_plan_cache = {}
        estimator._generator_measurement_plan_cache = {}
        estimator._balance_measurement_plan_cache = {}

        calls = []

        def counted_builder(*args, **kwargs):
            calls.append(1)
            return original_builder(*args, **kwargs)

        previous_builder = getattr(ac_se, "build_measurement_plan_table", None)
        ac_se.build_measurement_plan_table = counted_builder
        try:
            estimator._branch_transformer_vector_plan(estimator.active_measurements)
            estimator._simple_jacobian_plan(estimator.active_measurements)
            estimator._zero_current_vector_plan(estimator.active_measurements)
            estimator._generator_measurement_plan(estimator.active_measurements)
            estimator._balance_measurement_plan(estimator.active_measurements)
        finally:
            if previous_builder is None:
                del ac_se.build_measurement_plan_table
            else:
                ac_se.build_measurement_plan_table = previous_builder

        self.assertGreaterEqual(len(calls), 5)

    def test_active_measurement_plans_are_estimator_local(self):
        from secore.ac_se import ACStateEstimator

        first = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        second = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        self.assertIsNot(
            first._active_branch_transformer_vector_plan,
            second._active_branch_transformer_vector_plan,
        )
        np.testing.assert_allclose(first.evaluate(first.initial_state()), second.evaluate(second.initial_state()))

    def test_active_measurement_summary_reuses_initialization_cache(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        expected_device_keys = set(estimator._active_device_keys())
        expected_measurement_keys = set(estimator._active_measurement_keys())
        expected_next_idx = max(meas.idx for meas in estimator.measurements) + 1

        class RejectIteration(list):
            def __iter__(self):
                raise AssertionError("active measurement summary should use cached values")

        estimator.measurements = RejectIteration(estimator.measurements)

        self.assertEqual(expected_device_keys, estimator._active_device_keys())
        self.assertEqual(expected_measurement_keys, estimator._active_measurement_keys())
        self.assertEqual(expected_next_idx, estimator._next_measurement_idx())

    def test_initialization_prepares_active_measurement_vectors(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            )
        finally:
            ac_se.matpower_branch_stamp = original_stamp

        self.assertTrue(estimator.branch_stamp_by_name)
        self.assertTrue(estimator.transformer_stamp_by_name)

    def test_initialization_defers_scalar_branch_stamp_map_materialization(self):
        from secore.ac_se import ACStateEstimator

        original_build_map = ACStateEstimator._build_branch_stamp_map

        def reject_prepare_stamp(_devices, _with_tap):
            raise AssertionError("branch stamp maps should be lazy during AC SE prepare")

        ACStateEstimator._build_branch_stamp_map = staticmethod(reject_prepare_stamp)
        try:
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
                meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            )
        finally:
            ACStateEstimator._build_branch_stamp_map = original_build_map

        self.assertTrue(estimator.branch_stamp_by_name)
        branch_name = next(iter(estimator.branch_by_name))
        self.assertEqual(4, len(estimator.branch_stamp_by_name[branch_name]))

    def test_ppc_prepare_keeps_branch_and_transformer_device_maps_lazy(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )

        self.assertFalse(isinstance(estimator.branch_by_name, dict))
        self.assertFalse(isinstance(estimator.transformer_by_name, dict))
        self.assertEqual(0, estimator.branch_by_name.materialized_count)
        self.assertEqual(0, estimator.transformer_by_name.materialized_count)
        branch_name = next(iter(estimator.branch_by_name))
        self.assertEqual(0, estimator.branch_by_name.materialized_count)
        self.assertEqual(branch_name, estimator.branch_by_name[branch_name].name)
        self.assertEqual(1, estimator.branch_by_name.materialized_count)

    def test_node_incident_degrees_use_ppc_topology_not_branch_device_maps(self):
        from secore.ac_se import ACStateEstimator

        class RejectDeviceMap(dict):
            def values(self):
                raise AssertionError("node degrees should use PPC topology arrays")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator.branch_by_name = RejectDeviceMap()
        estimator.transformer_by_name = RejectDeviceMap()

        degrees = estimator._node_incident_degrees()

        self.assertGreater(degrees[1], 0)
        self.assertGreater(degrees[2], 0)

    def test_estimate_skips_final_observability_analysis(self):
        from secore.ac_se import ACStateEstimator, ObservabilityResult

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        initial_observability = ObservabilityResult(
            True,
            estimator.n_state,
            estimator.n_state,
            len(estimator.active_measurements),
            0,
            np.array([]),
            [],
        )
        calls = 0

        def counted_observability(*args, **kwargs):
            nonlocal calls
            calls += 1
            return initial_observability

        estimator.observability_analysis = counted_observability
        result = estimator.estimate()

        self.assertTrue(result.converged)
        self.assertIs(initial_observability, result.observability)
        self.assertEqual(1, calls)

    def test_custom_measurement_estimate_gets_pre_estimation_observability_once(self):
        from secore.ac_se import ACStateEstimator, ObservabilityResult

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        measurements = estimator.active_measurements[:]
        initial_observability = ObservabilityResult(
            True,
            estimator.n_state,
            estimator.n_state,
            len(measurements),
            0,
            np.array([]),
            [],
        )
        calls = 0

        def counted_observability(*args, **kwargs):
            nonlocal calls
            calls += 1
            self.assertIs(measurements, args[1] if len(args) > 1 else kwargs.get("measurements"))
            return initial_observability

        estimator.observability_analysis = counted_observability
        result = estimator.estimate(measurements)

        self.assertTrue(result.converged)
        self.assertIs(initial_observability, result.observability)
        self.assertEqual(1, calls)

    def test_estimate_passes_file_weights_to_normal_equation_builder(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        self.assertTrue(hasattr(ac_se, "build_normal_equations"))
        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        original = ac_se.build_normal_equations
        non_unit_weight_seen = False

        def counted_builder(H, residual, weight, **kwargs):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight, **kwargs)

        ac_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate()
        finally:
            ac_se.build_normal_equations = original

        self.assertTrue(result.converged)
        self.assertTrue(non_unit_weight_seen)

    def test_main_skip_bad_data_skips_post_estimation_bad_data_analysis(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_identify = ACStateEstimator.identify_bad_data

        def reject_identify(*_args, **_kwargs):
            raise AssertionError("--skip-bad-data should bypass identify_bad_data")

        ACStateEstimator.identify_bad_data = reject_identify
        try:
            rc = ac_se.main(
                [
                    "--case",
                    str(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"),
                    "--meas",
                    str(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"),
                    "--para",
                    str(ROOT_DIR / "data" / "se" / "se.para"),
                    "--flat-start",
                    "--quiet",
                    "--skip-bad-data",
                ]
            )
        finally:
            ACStateEstimator.identify_bad_data = original_identify

        self.assertEqual(0, rc)

    def test_main_runs_observability_before_estimation_and_does_not_repeat_it(self):
        import contextlib
        import io
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        events = []
        original_prepare = ACStateEstimator.prepare
        original_observability = ACStateEstimator.observability_analysis
        original_estimate = ACStateEstimator.estimate
        original_run = ACStateEstimator.run
        test_case = self

        def counted_prepare(self, *args, **kwargs):
            events.append("prepare")
            return original_prepare(self, *args, **kwargs)

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

        def counted_run(self, *args, **kwargs):
            test_case.assertNotIn("observability", kwargs)
            test_case.assertFalse(getattr(self, "_prepared", False))
            return original_run(self, *args, **kwargs)

        output = io.StringIO()
        ACStateEstimator.prepare = counted_prepare
        ACStateEstimator.observability_analysis = counted_observability
        ACStateEstimator.estimate = counted_estimate
        ACStateEstimator.run = counted_run
        try:
            with contextlib.redirect_stdout(output):
                rc = ac_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"),
                        "--para",
                        str(ROOT_DIR / "data" / "se" / "se.para"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            ACStateEstimator.prepare = original_prepare
            ACStateEstimator.observability_analysis = original_observability
            ACStateEstimator.estimate = original_estimate
            ACStateEstimator.run = original_run

        self.assertEqual(0, rc)
        self.assertEqual("prepare", events[0])
        self.assertEqual(1, events.count("prepare"))
        self.assertEqual(["observability", "estimate"], events[-2:])
        self.assertEqual(1, output.getvalue().count("Observability:"))
        self.assertLess(output.getvalue().index("Observability:"), output.getvalue().index("State estimation:"))

    def test_main_does_not_build_seresult_without_output_file(self):
        import contextlib
        import io
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        original_build = ACStateEstimator.build_se_result

        def reject_build(*_args, **_kwargs):
            raise AssertionError("SEResult details should be built only when --se-result is requested")

        ACStateEstimator.build_se_result = reject_build
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = ac_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"),
                        "--para",
                        str(ROOT_DIR / "data" / "se" / "se.para"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            ACStateEstimator.build_se_result = original_build

        self.assertEqual(0, rc)

    def test_main_honors_explicit_return_mode_without_output_file(self):
        import contextlib
        import io
        import secore.ac_se as ac_se
        from model.meas_model import ObservabilityResult
        from secore.ac_se import ACStateEstimator

        original_run = ACStateEstimator.run
        calls = []

        def fake_run(self, *args, **kwargs):
            calls.append(kwargs.get("return_mode"))
            observability = ObservabilityResult(True, 0, 0, 0, 0, np.array([]), [])
            self.observability_result = observability
            self.estimate_result = SimpleNamespace(
                converged=True,
                iterations=0,
                objective=0.0,
                max_correction=0.0,
                residual_inf=0.0,
                observability=observability,
                x=np.array([], dtype=np.float64),
            )
            self.removed_bad_data = []
            self.bad_items = []
            self.normalized_residual = np.array([], dtype=np.float64)
            return None

        ACStateEstimator.run = fake_run
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = ac_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"),
                        "--para",
                        str(ROOT_DIR / "data" / "se" / "se.para"),
                        "--flat-start",
                        "--quiet",
                        "--return-mode",
                        "array",
                    ]
                )
        finally:
            ACStateEstimator.run = original_run

        self.assertEqual(0, rc)
        self.assertEqual(["array"], calls)

    def test_estimate_can_skip_final_diagnostic_jacobian_and_gain(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )
        original_jacobian = estimator.jacobian_sparse
        calls = 0

        def counted_jacobian(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_jacobian(*args, **kwargs)

        estimator.jacobian_sparse = counted_jacobian
        result = estimator.estimate(final_diagnostics=False)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, calls)
        self.assertIsNone(result.H)
        self.assertIsNone(result.gain)

    def test_run_summary_return_mode_limits_seresult_only(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            auto_prepare=False,
        )
        self.assertFalse(estimator._prepared)
        estimator.prepare()
        se_result = estimator.run(
            return_mode="summary",
            verbose=False,
            skip_bad_data=True,
            final_diagnostics=False,
        )
        result = estimator.estimate_result

        self.assertIs(se_result, estimator.se_result)
        self.assertTrue(result.converged)
        self.assertIs(estimator.observability_result, result.observability)
        self.assertFalse(hasattr(result, "return_mode"))
        self.assertGreater(result.x.size, 0)
        self.assertGreater(result.z_est.size, 0)
        self.assertGreater(result.residual.size, 0)
        self.assertIsNone(result.H)
        self.assertIsNone(result.gain)
        self.assertEqual(result.iterations, se_result.statistics.iterations)
        self.assertEqual(0, len(se_result.prefiltered_measurements))
        self.assertEqual(0, len(se_result.pseudo_measurements))
        self.assertEqual(0, len(se_result.bad_data))
        self.assertEqual(0, len(se_result.normal_measurements))

    def test_run_prepares_on_demand_like_ac_lf(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            auto_prepare=False,
        )

        self.assertFalse(estimator._prepared)
        se_result = estimator.run(
            return_mode="array",
            skip_bad_data=True,
            final_diagnostics=False,
            verbose=False,
        )

        self.assertIsNone(se_result)
        self.assertTrue(estimator._prepared)
        self.assertTrue(estimator.estimate_result.converged)

    def test_run_array_return_mode_keeps_estimate_arrays_only(self):
        import secore.ac_se as ac_se_module
        from secore.ac_se import ACStateEstimator
        from secore.se_result import SEResult

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            auto_prepare=False,
        )
        estimator.prepare()
        original_build = ACStateEstimator.build_se_result
        original_summary = ac_se_module.build_seresult_summary
        original_identify = ACStateEstimator.identify_bad_data
        original_from_estimate = SEResult.from_estimate_result
        original_apply_state = ACStateEstimator.apply_state

        def reject_seresult_path(*_args, **_kwargs):
            raise AssertionError("array return_mode should not build SEResult payloads")

        def reject_bad_data(*_args, **_kwargs):
            raise AssertionError("array return_mode should not run post-estimation bad-data analysis")

        def reject_apply_state(*_args, **_kwargs):
            raise AssertionError("array return_mode should not write estimated state back to model objects")

        def reject_full_tables(*_args, **_kwargs):
            raise AssertionError("array return_mode should not build full SEResult measurement tables")

        ACStateEstimator.build_se_result = reject_seresult_path
        ac_se_module.build_seresult_summary = reject_seresult_path
        ACStateEstimator.identify_bad_data = reject_bad_data
        ACStateEstimator.apply_state = reject_apply_state
        SEResult.from_estimate_result = reject_full_tables
        try:
            se_result = estimator.run(
                return_mode="array",
                verbose=False,
            )
        finally:
            ACStateEstimator.build_se_result = original_build
            ac_se_module.build_seresult_summary = original_summary
            ACStateEstimator.identify_bad_data = original_identify
            ACStateEstimator.apply_state = original_apply_state
            SEResult.from_estimate_result = original_from_estimate
        result = estimator.estimate_result

        self.assertIsNone(se_result)
        self.assertIsNone(estimator.se_result)
        self.assertTrue(result.converged)
        self.assertGreater(result.x.size, 0)
        self.assertGreater(result.z_est.size, 0)
        self.assertGreater(result.residual.size, 0)
        self.assertIsNone(result.H)
        self.assertIsNone(result.gain)
        self.assertEqual(0, len(result.measurements))
        self.assertEqual([], estimator.bad_items)
        self.assertEqual(0, estimator.normalized_residual.size)

    def test_estimate_uses_cholesky_solver_when_available(self):
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
        normal = H.T @ H

        original_svd = np.linalg.svd

        def fail_svd(*args, **kwargs):
            raise AssertionError("large sparse observability should not use dense SVD fallback")

        np.linalg.svd = fail_svd
        try:
            rank, deficiency, singular_values, weak_states = observability_rank_details(
                H,
                n_state,
                normal_matrix=normal,
            )
        finally:
            np.linalg.svd = original_svd

        self.assertLess(rank, n_state)
        self.assertGreater(deficiency, 0)
        self.assertEqual(0, singular_values.size)
        self.assertTrue(weak_states)

    def test_observability_rank_cannot_exceed_measurement_row_count(self):
        from scipy.sparse import csr_matrix
        from secore.se_math import observability_rank_details

        H = csr_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        rank, deficiency, _singular_values, _weak_states = observability_rank_details(
            H,
            3,
            normal_factor_diag=np.ones(3),
        )

        self.assertLessEqual(rank, H.shape[0])
        self.assertEqual(2, rank)
        self.assertEqual(1, deficiency)

    def test_observability_uses_lapack_cholesky_when_available(self):
        import secore.se_math as se_math
        from secore.ac_se import ACStateEstimator

        try:
            from scipy.linalg.lapack import dpotrf  # noqa: F401
        except Exception:
            self.skipTest("SciPy LAPACK dpotrf is not available")
        self.assertIsNotNone(getattr(se_math, "DPOTRF", None))

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        self.assertIn(id(estimator.active_measurements), estimator._branch_transformer_vector_plan_cache)

    def test_jacobian_vectorizes_ac_branch_derivatives(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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

    def test_generator_jacobian_does_not_require_dense_network_derivatives(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
        )

        generator_measurements = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "ACGenerator" and meas.meas_type in ("P_GEN", "Q_GEN", "I_GEN")
        ]
        self.assertTrue(generator_measurements)
        self.assertFalse(hasattr(estimator, "_network_power_derivatives"))

        H = estimator.jacobian(estimator.initial_state(), generator_measurements)
        self.assertEqual(len(generator_measurements), H.shape[0])

    def test_analytic_jacobian_matches_finite_difference(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
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
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
                meas_file=meas_file,
                max_iter=20,
            )

        active_types = {meas.meas_type for meas in estimator.active_measurements}
        active_devices = {meas.device_type for meas in estimator.active_measurements}
        self.assertIn("I_FROM", active_types)
        self.assertIn("I_TO", active_types)
        self.assertIn("I_GEN", active_types)
        self.assertIn("I_LOAD", active_types)
        self.assertIn("ACBreak", active_devices)

        result = estimator.estimate()
        self.assertTrue(result.converged)
        self.assertTrue(result.observability.observable)
        self.assertLess(result.residual_inf, 1e-6)

    def test_ac_net_30_estimates_with_zero_branch_measurements(self):
        from secore.ac_se import ACStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
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
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "ac" / "ieee3k.meas")
            estimator = ACStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee3k.e",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
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
        original_cholmod_analyze = se_math.CHOLMOD_ANALYZE
        original_cholmod_cholesky = se_math.CHOLMOD_CHOLESKY

        class FakeLU:
            def __init__(self):
                self.U = csc_matrix(np.eye(2))

            def solve(self, rhs):
                return np.asarray(rhs, dtype=np.float64)

        def fake_splu(matrix, **kwargs):
            calls.append(kwargs)
            return FakeLU()

        se_math.SP_SPLU = fake_splu
        se_math.CHOLMOD_ANALYZE = None
        se_math.CHOLMOD_CHOLESKY = None
        try:
            dx, diag = se_math.solve_normal_equations_with_factor(csc_matrix(np.eye(2)), np.ones(2))
        finally:
            se_math.SP_SPLU = original_splu
            se_math.CHOLMOD_ANALYZE = original_cholmod_analyze
            se_math.CHOLMOD_CHOLESKY = original_cholmod_cholesky

        self.assertTrue(calls)
        self.assertEqual(0.0, calls[0]["diag_pivot_thresh"])
        self.assertEqual("MMD_AT_PLUS_A", calls[0]["permc_spec"])
        np.testing.assert_allclose(dx, np.ones(2))
        np.testing.assert_allclose(diag, np.ones(2))

    def test_normal_equation_assembly_plan_matches_sparse_reference(self):
        from scipy.sparse import csr_matrix
        import secore.se_math as se_math

        H = csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 2.0],
                    [0.0, -3.0, 4.0],
                    [5.0, 6.0, 0.0],
                    [0.0, 7.0, 8.0],
                ],
                dtype=np.float64,
            )
        )
        residual = np.array([0.5, -0.25, 1.25, -1.5], dtype=np.float64)
        weight = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)

        plan = se_math.NormalEquationAssemblyPlan.from_jacobian(H)
        gain, rhs = se_math.build_normal_equations(
            H,
            residual,
            weight,
            dense_gain_limit=0,
            weights_are_uniform=False,
            normal_assembly_plan=plan,
        )
        expected_gain = H.T @ H.multiply(weight[:, None])
        expected_rhs = H.T @ (weight * residual)

        np.testing.assert_allclose(gain.toarray(), expected_gain.toarray())
        np.testing.assert_allclose(rhs, np.asarray(expected_rhs).ravel())

        H2 = H.copy()
        H2.data = H2.data * np.linspace(0.5, 1.5, H2.data.size)
        residual2 = residual * 0.25
        gain2, rhs2 = se_math.build_normal_equations(
            H2,
            residual2,
            weight,
            dense_gain_limit=0,
            weights_are_uniform=False,
            normal_assembly_plan=plan,
        )
        expected_gain2 = H2.T @ H2.multiply(weight[:, None])
        expected_rhs2 = H2.T @ (weight * residual2)
        np.testing.assert_allclose(gain2.toarray(), expected_gain2.toarray())
        np.testing.assert_allclose(rhs2, np.asarray(expected_rhs2).ravel())

    def test_normal_equation_assembly_plan_uses_pair_count_threshold(self):
        from scipy.sparse import csr_matrix
        import secore.se_math as se_math

        H = csr_matrix(np.ones((3, 3), dtype=np.float64))

        self.assertFalse(
            se_math.NormalEquationAssemblyPlan.direct_assembly_is_reasonable(H, max_pair_count=26)
        )
        self.assertTrue(
            se_math.NormalEquationAssemblyPlan.direct_assembly_is_reasonable(H, max_pair_count=27)
        )

    def test_estimate_reuses_normal_equation_assembly_plan_for_active_measurements(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
        )

        original_builder = ac_se.build_normal_equations
        plan_seen = []

        def counted_builder(*args, **kwargs):
            plan_seen.append(kwargs.get("normal_assembly_plan"))
            return original_builder(*args, **kwargs)

        ac_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate(final_diagnostics=False)
        finally:
            ac_se.build_normal_equations = original_builder

        self.assertTrue(result.converged)
        self.assertTrue(any(plan is not None for plan in plan_seen))

    def test_cholmod_solver_reuses_symbolic_analysis_for_same_sparse_pattern(self):
        from scipy.sparse import csc_matrix
        import secore.se_math as se_math

        analyze_calls = []
        numeric_calls = []
        original_cholmod_analyze = se_math.CHOLMOD_ANALYZE
        original_cholmod_cholesky = se_math.CHOLMOD_CHOLESKY

        class FakeCholmodFactor:
            def __init__(self, matrix):
                self.matrix = matrix

            def cholesky_inplace(self, matrix):
                numeric_calls.append(matrix.copy())
                self.matrix = matrix

            def __call__(self, rhs):
                return np.linalg.solve(self.matrix.toarray(), rhs)

        def fake_analyze(matrix):
            analyze_calls.append(matrix.copy())
            return FakeCholmodFactor(matrix)

        se_math.CHOLMOD_ANALYZE = fake_analyze
        se_math.CHOLMOD_CHOLESKY = None
        try:
            solver = se_math.NormalEquationSolver()
            gain_1 = csc_matrix([[4.0, 1.0], [1.0, 3.0]])
            gain_2 = csc_matrix([[5.0, 2.0], [2.0, 6.0]])
            dx_1, _ = solver.solve(gain_1, np.array([1.0, 2.0]))
            dx_2, _ = solver.solve(gain_2, np.array([3.0, 4.0]))
        finally:
            se_math.CHOLMOD_ANALYZE = original_cholmod_analyze
            se_math.CHOLMOD_CHOLESKY = original_cholmod_cholesky

        self.assertEqual(1, len(analyze_calls))
        self.assertEqual(2, len(numeric_calls))
        np.testing.assert_allclose(dx_1, np.linalg.solve(gain_1.toarray(), np.array([1.0, 2.0])))
        np.testing.assert_allclose(dx_2, np.linalg.solve(gain_2.toarray(), np.array([3.0, 4.0])))

    def test_normal_equation_solver_disables_repeated_failed_cholmod_attempts(self):
        from scipy.sparse import csc_matrix
        import secore.se_math as se_math

        original_cholmod_analyze = se_math.CHOLMOD_ANALYZE
        original_cholmod_cholesky = se_math.CHOLMOD_CHOLESKY
        calls = 0

        def failing_analyze(_matrix):
            nonlocal calls
            calls += 1
            raise RuntimeError("cholmod unavailable for this matrix")

        se_math.CHOLMOD_ANALYZE = failing_analyze
        se_math.CHOLMOD_CHOLESKY = None
        try:
            solver = se_math.NormalEquationSolver()
            gain = csc_matrix([[4.0, 1.0], [1.0, 3.0]])
            dx_1, _ = solver.solve(gain, np.array([1.0, 2.0]))
            dx_2, _ = solver.solve(gain, np.array([3.0, 4.0]))
        finally:
            se_math.CHOLMOD_ANALYZE = original_cholmod_analyze
            se_math.CHOLMOD_CHOLESKY = original_cholmod_cholesky

        self.assertEqual(1, calls)
        np.testing.assert_allclose(dx_1, np.linalg.solve(gain.toarray(), np.array([1.0, 2.0])))
        np.testing.assert_allclose(dx_2, np.linalg.solve(gain.toarray(), np.array([3.0, 4.0])))

    def test_targeted_pseudo_reuses_observable_rank_result(self):
        from secore.ac_se import ACStateEstimator, ObservabilityResult

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        estimator.targeted_pseudo_measurement_max = 10
        estimator.measurements = []
        calls = 0

        def observable():
            nonlocal calls
            calls += 1
            return ObservabilityResult(True, 1, 1, 1, 0, np.array([]), [])

        estimator.observability_analysis = observable
        estimator._active_measurement_keys = lambda: set()
        estimator._append_targeted_observability_pseudo = lambda *args, **kwargs: (1, 0)
        estimator._refresh_active_measurement_indexes = lambda: None
        estimator._add_structural_rank_restoring_pseudo_measurements = lambda max_add: 0

        self.assertEqual(0, estimator._add_targeted_observability_pseudo_measurements())
        self.assertEqual(1, calls)

    def test_targeted_pseudo_small_batch_avoids_full_active_refresh(self):
        from secore.ac_se import ACStateEstimator, ObservabilityResult

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator._refresh_active_measurement_indexes()
        initial_active_count = len(estimator.active_measurements)
        observable_result = ObservabilityResult(
            observable=True,
            rank=estimator.n_state,
            state_count=estimator.n_state,
            measurement_count=len(estimator.active_measurements),
            deficiency=0,
            singular_values=np.ones(1, dtype=np.float64),
            weak_states=[],
        )
        target_node = estimator.node_by_name["bus_2"]
        target_pos = estimator.node_pos[target_node.idx]
        target_col = int(estimator.voltage_col[target_pos])
        estimator.state_labels = [f"opaque_state_{idx}" for idx in range(estimator.n_state)]
        non_observable_result = ObservabilityResult(
            observable=False,
            rank=max(estimator.n_state - 1, 0),
            state_count=estimator.n_state,
            measurement_count=len(estimator.active_measurements),
            deficiency=1,
            singular_values=np.ones(1, dtype=np.float64),
            weak_states=[(target_col, 1.0)],
        )
        results = [non_observable_result, observable_result]
        estimator.observability_analysis = lambda: results.pop(0) if results else observable_result
        estimator.targeted_pseudo_measurement_max = 1
        estimator.targeted_pseudo_measurement_step = 1
        estimator.targeted_pseudo_measurement_redundancy_ratio = 0.0
        estimator._refresh_active_measurement_indexes = lambda: (_ for _ in ()).throw(
            AssertionError("AC targeted pseudo append should update active layout incrementally")
        )
        estimator._add_structural_rank_restoring_pseudo_measurements = lambda max_add: 0

        added = estimator._add_targeted_observability_pseudo_measurements()

        self.assertEqual(1, added)
        self.assertEqual(initial_active_count + 1, len(estimator.active_measurements))

    def test_incremental_updater_reuses_shared_se_array_plan_helpers(self):
        import secore.ac_se as ac_se
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator._refresh_active_measurement_indexes()
        additions = [Measurement(500000, "ac_v2", "ACNode", "bus_2", "V", 5.0, True, 1.01)]
        estimator.measurements.extend(additions)
        calls = {"active": 0}
        original_active = ac_se.append_active_measurement_view

        def counted_active(*args, **kwargs):
            calls["active"] += 1
            return original_active(*args, **kwargs)

        ac_se.append_active_measurement_view = counted_active
        try:
            refreshed = estimator._incremental_update_active_measurement_indexes(additions)
        finally:
            ac_se.append_active_measurement_view = original_active

        self.assertTrue(refreshed)
        self.assertEqual(1, calls["active"])

    def test_incremental_updater_reuses_existing_active_measurement_plans(self):
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ieee39.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator._refresh_active_measurement_indexes()
        additions = [Measurement(500001, "ac_v3", "ACNode", "bus_2", "V", 5.0, True, 1.02)]
        estimator.measurements.extend(additions)
        active_len = len(estimator.active_measurements)
        calls = {
            "branch_full": 0,
            "branch_append": 0,
            "simple_full": 0,
            "simple_append": 0,
            "zero_full": 0,
            "zero_append": 0,
            "generator_full": 0,
            "generator_append": 0,
            "balance_full": 0,
            "balance_append": 0,
        }

        def count_builder(name, original):
            def wrapped(measurements):
                key = f"{name}_{'append' if len(measurements) == 1 else 'full' if len(measurements) == active_len + 1 else 'other'}"
                if key in calls:
                    calls[key] += 1
                return original(measurements)

            return wrapped

        original_branch = estimator._build_branch_transformer_vector_plan
        original_simple = estimator._build_simple_jacobian_plan
        original_zero = estimator._build_zero_current_vector_plan
        original_generator = estimator._build_generator_measurement_plan
        original_balance = estimator._build_balance_measurement_plan
        estimator._build_branch_transformer_vector_plan = count_builder("branch", original_branch)
        estimator._build_simple_jacobian_plan = count_builder("simple", original_simple)
        estimator._build_zero_current_vector_plan = count_builder("zero", original_zero)
        estimator._build_generator_measurement_plan = count_builder("generator", original_generator)
        estimator._build_balance_measurement_plan = count_builder("balance", original_balance)
        try:
            refreshed = estimator._incremental_update_active_measurement_indexes(additions)
        finally:
            estimator._build_branch_transformer_vector_plan = original_branch
            estimator._build_simple_jacobian_plan = original_simple
            estimator._build_zero_current_vector_plan = original_zero
            estimator._build_generator_measurement_plan = original_generator
            estimator._build_balance_measurement_plan = original_balance

        self.assertTrue(refreshed)
        self.assertEqual(0, calls["branch_full"])
        self.assertEqual(0, calls["simple_full"])
        self.assertEqual(0, calls["zero_full"])
        self.assertEqual(0, calls["generator_full"])
        self.assertEqual(0, calls["balance_full"])
        self.assertEqual(1, calls["branch_append"])
        self.assertEqual(1, calls["simple_append"])
        self.assertEqual(1, calls["zero_append"])
        self.assertEqual(1, calls["generator_append"])
        self.assertEqual(1, calls["balance_append"])

    def test_bad_data_removal_reuses_table_backed_measurement_subset(self):
        from types import SimpleNamespace
        from model.meas_model import BadDataItem, EstimateResult, MeasurementList, ObservabilityResult, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        m1 = Measurement(1, "m1", "ACNode", "n1", "V", 1.0, True, 1.0)
        m2 = Measurement(2, "m2", "ACNode", "n2", "V", 1.0, True, 1.0)
        measurements = MeasurementList([m1, m2], measurement_table_from_measurements([m1, m2]))
        estimator.active_measurements = measurements
        estimator.params = SimpleNamespace(bad_threshold=3.0, max_remove=1)
        estimator.initial_state = lambda: np.array([0.0], dtype=np.float64)
        result1 = EstimateResult(True, 1, 0.0, 0.0, 0.0, np.array([0.0]), np.zeros(2), np.zeros(2), None, None, measurements, ObservabilityResult(True, 1, 1, 2, 0, np.array([1.0]), []))
        result2 = EstimateResult(True, 1, 0.0, 0.0, 0.0, np.array([0.0]), np.zeros(1), np.zeros(1), None, None, None, ObservabilityResult(True, 1, 1, 1, 0, np.array([1.0]), []))
        seen = []

        def estimate(measurements, x0=None, verbose=False):
            seen.append(measurements)
            return result1 if len(seen) == 1 else result2

        estimator.estimate = estimate
        estimator.identify_bad_data = lambda result, threshold=None: ([BadDataItem(m1, 1.0, 4.0, 0.0, 1.0)] if result is result1 else [], np.array([]))

        estimator.estimate_with_bad_data_removal()

        self.assertEqual(2, len(seen))
        self.assertIsInstance(seen[1], MeasurementList)
        self.assertIsNotNone(seen[1].table)
        self.assertEqual([m2], list(seen[1]))

    def test_bad_data_removal_reuses_active_shrink_update(self):
        from types import SimpleNamespace
        from model.meas_model import BadDataItem, EstimateResult, MeasurementList, ObservabilityResult, measurement_table_from_measurements
        from secore.ac_se import ACStateEstimator, Measurement

        estimator = ACStateEstimator.__new__(ACStateEstimator)
        m1 = Measurement(1, "m1", "ACNode", "n1", "V", 1.0, True, 1.0)
        m2 = Measurement(2, "m2", "ACNode", "n2", "V", 1.0, True, 1.0)
        measurements = MeasurementList([m1, m2], measurement_table_from_measurements([m1, m2]))
        estimator.active_measurements = measurements
        estimator.active_measurement_table = measurements.table
        estimator.params = SimpleNamespace(bad_threshold=3.0, max_remove=1)
        estimator.initial_state = lambda: np.array([0.0], dtype=np.float64)
        result1 = EstimateResult(True, 1, 0.0, 0.0, 0.0, np.array([0.0]), np.zeros(2), np.zeros(2), None, None, measurements, ObservabilityResult(True, 1, 1, 2, 0, np.array([1.0]), []))
        result2 = EstimateResult(True, 1, 0.0, 0.0, 0.0, np.array([0.0]), np.zeros(1), np.zeros(1), None, None, None, ObservabilityResult(True, 1, 1, 1, 0, np.array([1.0]), []))
        seen = []
        shrink_calls = []

        def estimate(measurements, x0=None, verbose=False):
            seen.append(measurements)
            return result1 if len(seen) == 1 else result2

        def shrink(remove_pos):
            shrink_calls.append(remove_pos)
            estimator.active_measurements = MeasurementList([m2], measurement_table_from_measurements([m2]))
            estimator.active_measurement_table = estimator.active_measurements.table
            return estimator.active_measurements

        estimator.estimate = estimate
        estimator.identify_bad_data = lambda result, threshold=None: ([BadDataItem(m1, 1.0, 4.0, 0.0, 1.0)] if result is result1 else [], np.array([]))
        estimator._shrink_active_measurement_indexes = shrink

        estimator.estimate_with_bad_data_removal()

        self.assertEqual([0], shrink_calls)
        self.assertIs(seen[0], measurements)
        self.assertIs(seen[1], estimator.active_measurements)

    def test_initial_observability_reuses_targeted_pseudo_result(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
        )

        original_jacobian_sparse = estimator.jacobian_sparse
        calls = 0

        def counted_jacobian(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_jacobian_sparse(*args, **kwargs)

        estimator.jacobian_sparse = counted_jacobian
        result = estimator.observability_analysis()

        self.assertTrue(result.observable)
        self.assertEqual(0, calls)

    def test_sparse_observability_disables_dynamic_pivoting_for_speed(self):
        from scipy.sparse import csc_matrix, eye
        import secore.se_math as se_math

        calls = []
        original_splu = se_math.SP_SPLU
        original_cholmod = se_math.CHOLMOD_CHOLESKY

        class FakeLU:
            def __init__(self):
                self.U = csc_matrix(np.eye(2))

        def fake_splu(matrix, **kwargs):
            calls.append(kwargs)
            return FakeLU()

        se_math.SP_SPLU = fake_splu
        se_math.CHOLMOD_CHOLESKY = None
        try:
            rank, deficiency, _singular_values, _weak_states = se_math.observability_rank_details(
                eye(2, format="csr"),
                2,
            )
        finally:
            se_math.SP_SPLU = original_splu
            se_math.CHOLMOD_CHOLESKY = original_cholmod

        self.assertEqual(2, rank)
        self.assertEqual(0, deficiency)
        self.assertTrue(calls)
        self.assertEqual(0.0, calls[0]["diag_pivot_thresh"])
        self.assertEqual("MMD_AT_PLUS_A", calls[0]["permc_spec"])

    def test_ac_net_30_analytic_jacobian_matches_finite_difference(self):
        from secore.ac_se import ACStateEstimator

        estimator = ACStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "ac" / "ac_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "ac" / "ac_net_30.meas",
            flat_start=True,
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
