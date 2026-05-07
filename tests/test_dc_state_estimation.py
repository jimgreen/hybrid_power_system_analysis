import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class DCStateEstimationTest(unittest.TestCase):
    def test_dc_topology_contracts_closed_switches_to_buses_before_islands(self):
        from model.dc_model import DCPowerNetwork as ObjectDCPowerNetwork

        network = ObjectDCPowerNetwork()
        network.add_node(1, 100.0)
        network.nodes[-1].name = "n1"
        network.add_node(2, 100.0)
        network.nodes[-1].name = "n2"
        network.add_node(3, 100.0)
        network.nodes[-1].name = "n3"
        network.add_generator(1, 1, "V", 1.0, 1.0, 0.0)
        network.generators[-1].name = "g1"
        network.add_switch(1, 1, 2, 1)
        network.switches[-1].name = "sw_1_2"
        network.add_switch(2, 2, 3, 0)
        network.switches[-1].name = "sw_2_3"
        network.add_branch(1, 2, 3, 0.01)
        network.branches[-1].name = "br_2_3"

        network.topo()

        self.assertEqual(2, len(network.buses))
        self.assertEqual(["n1", "n2"], [node.name for node in network.node_dict[1].bus_obj.nodes])
        self.assertIs(network.node_dict[1].bus_obj, network.node_dict[2].bus_obj)
        self.assertIsNot(network.node_dict[2].bus_obj, network.node_dict[3].bus_obj)
        self.assertEqual(1, len(network.islands))
        self.assertEqual(2, len(network.islands[0].buses))

    def test_dc_break_is_parsed_as_distinct_zero_tie_device(self):
        from model.dc_array_model import SWITCH_COLS, build_dc_ppc_from_e_file
        from model.dc_model import DCBreak, DCPowerNetwork as ObjectDCPowerNetwork

        source = ROOT_DIR / "data" / "dc" / "dc_net_30.e"
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "dc_break.e"
            text = source.read_text(encoding="utf-8")
            switch_start = text.index("<DCSwitch>")
            switch_end = text.index("</DCSwitch>", switch_start) + len("</DCSwitch>")
            break_start = text.index("<DCBreak>")
            break_end = text.index("</DCBreak>", break_start) + len("</DCBreak>")
            text = (
                text[:switch_start]
                + "<DCSwitch>\n@ idx name     i_node j_node status run_stat p current\n</DCSwitch>\n\n"
                + "<DCBreak>\n@ idx name     i_node j_node status run_stat p current\n"
                + "# 0   brk_0_1   0      1      1      1        0 0\n"
                + "</DCBreak>"
                + text[break_end:]
            )
            case_path.write_text(text, encoding="utf-8")

            ppc = build_dc_ppc_from_e_file(case_path)
            network = ObjectDCPowerNetwork()
            network.read_from_file(case_path)
            network.topo()

        self.assertEqual(0, ppc["switch"].shape[0])
        self.assertEqual(1, ppc["break"].shape[0])
        self.assertEqual("brk_0_1", ppc["break_name"][0])
        self.assertEqual(0, int(ppc["break"][0, SWITCH_COLS["i_node"]]))
        self.assertEqual(1, int(ppc["break"][0, SWITCH_COLS["j_node"]]))
        self.assertEqual(1, len(network.breakers))
        self.assertIsInstance(network.breakers[0], DCBreak)
        self.assertEqual("brk_0_1", network.breakers[0].name)
        self.assertTrue(network.node_dict[0].isl_obj is network.node_dict[1].isl_obj)

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

    def test_measurement_loader_bypasses_generic_ebook_parser(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator
        from model.meas_model import MeasurementList

        original_ebook = dc_se.EBook

        def fail_ebook(*args, **kwargs):
            raise AssertionError("generic EBook parser should not be used for DC measurements")

        dc_se.EBook = fail_ebook
        try:
            measurements = DCStateEstimator._load_measurements(ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
        finally:
            dc_se.EBook = original_ebook

        self.assertGreater(len(measurements), 0)
        self.assertIsInstance(measurements, MeasurementList)
        self.assertIsNotNone(measurements.table)
        self.assertEqual(len(measurements), len(measurements.table.idx))
        self.assertEqual("DCNode", measurements[0].device_type)
        self.assertEqual("V", measurements[0].meas_type)

    def test_dc_network_load_uses_array_model_by_default(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )

        self.assertEqual("model.dc_array_model", estimator.network.__class__.__module__)
        self.assertEqual("dc_ppc_v1", estimator.network.ppc["format"])

    def test_dc_network_load_uses_dc_lf_efile_loader(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original = dc_se.load_dc_ppc_from_e_file
        calls = []

        def counted_loader(path, *args, **kwargs):
            calls.append(Path(path).name)
            return original(path, *args, **kwargs)

        dc_se.load_dc_ppc_from_e_file = counted_loader
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.load_dc_ppc_from_e_file = original

        self.assertEqual(["dc_net_30.e"], calls)
        self.assertEqual("dc_ppc_v1", estimator.network.ppc["format"])

    def test_estimator_load_network_skips_topology_check(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original_check_topo = dc_se.DCPowerNetwork.check_topo

        def reject_check_topo(self):
            raise AssertionError("main DC state-estimation load path should not call check_topo")

        dc_se.DCPowerNetwork.check_topo = reject_check_topo
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.DCPowerNetwork.check_topo = original_check_topo

        self.assertTrue(estimator.nodes)

    def test_adds_low_weight_pseudo_power_measurements_for_unmetered_generators_and_loads(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "voltage_only.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 vm_nd_1 DCNode nd_1 V 1.0 1 100",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
            )

        pseudo = [meas for meas in estimator.active_measurements if meas.name.startswith("pseudo_")]
        pseudo_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in pseudo}

        self.assertIn(("DCGenerator", "gen_v1", "P_GEN"), pseudo_keys)
        self.assertIn(("DCLoad", "load_1", "P_LOAD"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "DCGenerator"
            and meas.device_name == "gen_v1"
            and meas.meas_type == "P_GEN"
        )
        load_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "DCLoad"
            and meas.device_name == "load_1"
            and meas.meas_type == "P_LOAD"
        )
        self.assertAlmostEqual(gen_p.value, estimator.generator_by_name["gen_v1"].p)
        self.assertAlmostEqual(load_p.value, estimator.load_by_name["load_1"].p)

    def test_reference_nodes_use_highest_degree_nodes_with_valid_voltage_measurements(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        expected_refs = ["nd_11", "nd_21", "nd_26"]

        self.assertEqual(expected_refs, [node.name for node in estimator.references])
        voltage, _switch_current, _dcdc_power, _vgen_power = estimator._unpack_state(estimator.initial_state())
        for name in expected_refs:
            node = estimator.node_by_name[name]
            pos = estimator.node_pos[node.idx]
            ref_voltage = estimator.node_voltage_measurements[node.idx]
            self.assertEqual(-1, int(estimator.voltage_col[pos]))
            self.assertAlmostEqual(ref_voltage, voltage[pos])

    def test_targeted_node_voltage_state_does_not_add_pseudo_measurement(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
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
        self.assertNotIn(("DCNode", "nd_2", "V"), existing_keys)

    def test_pseudo_measurements_are_device_level_for_dc_sources_loads_and_converters(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "device_level.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_gen DCGenerator gen_v1 V_GEN 1.0 1 100",
                        "# 2 v_load DCLoad load_1 V_LOAD 1.0 1 100",
                        "# 3 v_conv DCDCConverter conv_1 V_FROM 1.0 1 100",
                        "# 4 p_bad DCDCConverter conv_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
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

        self.assertNotIn(("DCGenerator", "gen_v1", "P_GEN"), regular_pseudo_keys)
        self.assertNotIn(("DCLoad", "load_1", "P_LOAD"), regular_pseudo_keys)
        self.assertNotIn(("DCDCConverter", "conv_1", "P_FROM"), regular_pseudo_keys)
        self.assertNotIn(("DCDCConverter", "conv_1", "P_TO"), regular_pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "P_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "P_TO"), pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "V_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "V_TO"), pseudo_keys)

    def test_adds_low_weight_pseudo_measurements_for_unmetered_nodes_breaks_and_zero_branches(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "invalid_topology_devices.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_nd_1 DCNode nd_1 V 1.0 1 160",
                        "# 2 v_nd_2_bad DCNode nd_2 V 1.0 0 160",
                        "# 3 p_brk_bad DCBreak sw_0_1 P_FROM 1.0 0 0",
                        "# 4 p_zbr_bad DCZeroBranch zbr_1_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_")
        }

        self.assertNotIn(("DCNode", "nd_1", "V"), pseudo_keys)
        self.assertNotIn(("DCNode", "nd_2", "V"), pseudo_keys)
        self.assertFalse(any(device_type == "DCNode" and meas_type == "V" for device_type, _name, meas_type in pseudo_keys))
        for meas_type in ("P_FROM", "V_FROM", "I_FROM"):
            self.assertIn(("DCBreak", "sw_0_1", meas_type), pseudo_keys)
            self.assertIn(("DCZeroBranch", "zbr_1_2", meas_type), pseudo_keys)
        self.assertIn("zbr_1_2", estimator.zero_branch_pos)

    def test_dc_zero_branches_are_compressed_like_closed_switches(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "zero_branch.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 pz DCZeroBranch zbr_1_2 P_FROM 1.0 1 0",
                        "# 2 vz DCZeroBranch zbr_1_2 V_FROM 1.0 1 100",
                        "# 3 iz DCZeroBranch zbr_1_2 I_FROM 1.0 1 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        active_zero = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "DCZeroBranch"
        ]
        self.assertEqual({"P_FROM", "V_FROM", "I_FROM"}, {meas.meas_type for meas in active_zero})
        self.assertIn("I_ZERO:zbr_1_2", estimator.state_labels)
        constraint_types = {
            meas.device_type
            for meas in estimator.active_measurements
            if meas.device_type in ("DCZeroBranchConstraint", "DCSwitchConstraint")
        }
        self.assertEqual(set(), constraint_types)

        zbr = estimator.zero_branch_by_name["zbr_1_2"]
        self.assertEqual(
            estimator.voltage_col[estimator.node_pos[zbr.i_node]],
            estimator.voltage_col[estimator.node_pos[zbr.j_node]],
        )
        sw = estimator.break_by_name["sw_0_1"]
        self.assertEqual(
            estimator.voltage_col[estimator.node_pos[sw.i_node]],
            estimator.voltage_col[estimator.node_pos[sw.j_node]],
        )

    def test_dc_net_30_estimation_observability_and_bad_data(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                max_iter=20,
            )

        observability = estimator.observability_analysis()
        self.assertTrue(observability.observable)
        self.assertEqual(observability.rank, observability.state_count)

        result = estimator.estimate()
        self.assertTrue(result.converged)
        self.assertLess(result.residual_inf, 1e-6)

        bad_items, normalized = estimator.identify_bad_data(result, threshold=3.0)
        self.assertEqual([], bad_items)
        self.assertLess(float(normalized.max()), 1e-3)

        bad_measurements = list(estimator.active_measurements)
        voltage_idx = next(i for i, meas in enumerate(bad_measurements) if meas.meas_type == "V")
        bad_measurements[voltage_idx] = replace(
            bad_measurements[voltage_idx],
            value=bad_measurements[voltage_idx].value + 5.0,
        )
        bad_result = estimator.estimate(bad_measurements)
        bad_items, _ = estimator.identify_bad_data(bad_result, threshold=3.0)
        self.assertGreaterEqual(len(bad_items), 1)
        self.assertEqual(bad_measurements[voltage_idx].idx, bad_items[0].measurement.idx)

    def test_jacobian_uses_direct_derivatives_without_repeated_evaluation(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
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
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )

        x = estimator.initial_state()
        dense = estimator.jacobian(x)
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_evaluate_batches_device_measurements(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )
        x = estimator.initial_state()
        expected = estimator.evaluate(x)

        def fail_scalar_value_path(*args, **kwargs):
            raise AssertionError("DC measurements must be evaluated in vectorized batches")

        estimator._branch_values = fail_scalar_value_path
        estimator._load_values = fail_scalar_value_path
        estimator._generator_values = fail_scalar_value_path
        estimator._switch_values = fail_scalar_value_path
        estimator._dcdc_values = fail_scalar_value_path
        actual = estimator.evaluate(x)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_sparse_jacobian_batches_device_measurements(self):
        from scipy.sparse import issparse
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )
        x = estimator.initial_state()
        dense = estimator.jacobian(x)

        def fail_scalar_derivative_path(*args, **kwargs):
            raise AssertionError("DC sparse Jacobian must be assembled in vectorized batches")

        estimator._add_derivative = fail_scalar_derivative_path
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_active_measurement_arrays_are_cached_for_estimation(self):
        from model.meas_model import MeasurementList
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )

        self.assertIsInstance(estimator.active_measurements, MeasurementList)
        self.assertIsNotNone(estimator.active_measurement_table)
        self.assertIs(estimator.active_measurements.table, estimator.active_measurement_table)
        self.assertTrue(hasattr(estimator, "active_z"))
        self.assertTrue(hasattr(estimator, "active_weight"))
        np.testing.assert_allclose(
            estimator.active_z,
            np.asarray(estimator.active_measurement_table.value, dtype=np.float64),
        )
        np.testing.assert_allclose(
            estimator.active_weight,
            np.asarray(estimator.active_measurement_table.weight, dtype=np.float64),
        )

    def test_apply_state_batches_device_value_calculation(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )

        def fail_scalar_value_path(*args, **kwargs):
            raise AssertionError("DC apply_state should calculate device values in vectorized batches")

        estimator._branch_values = fail_scalar_value_path
        estimator._load_values = fail_scalar_value_path
        estimator._generator_values = fail_scalar_value_path
        estimator._switch_values = fail_scalar_value_path
        estimator._dcdc_values = fail_scalar_value_path

        estimator.apply_state(estimator.initial_state())

        self.assertTrue(all(node.voltage > 0.0 for node in estimator.nodes))

    def test_estimate_reuses_converged_iteration_sparse_jacobian(self):
        from scipy.sparse import issparse
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
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

    def test_flat_start_does_not_run_power_flow_seed(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original_run = dc_se.DCPowerFlowCalc.run
        call_count = 0

        def counted_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_run(*args, **kwargs)

        dc_se.DCPowerFlowCalc.run = counted_run
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.DCPowerFlowCalc.run = original_run

        self.assertTrue(estimator.flat_start)
        self.assertEqual(0, call_count)

    def test_estimate_reuses_gain_matrix_for_observability(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )

        original = dc_se.observability_rank_details
        normal_matrix_seen = False

        def counted_rank_details(*args, **kwargs):
            nonlocal normal_matrix_seen
            normal_matrix_seen = kwargs.get("normal_matrix") is not None
            return original(*args, **kwargs)

        dc_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate()
        finally:
            dc_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(normal_matrix_seen)

    def test_estimate_reuses_cholesky_factor_for_observability(self):
        import secore.dc_se as dc_se
        import secore.se_math as se_math
        from secore.dc_se import DCStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
            )

            original = dc_se.observability_rank_details
            factor_seen = False

            def counted_rank_details(*args, **kwargs):
                nonlocal factor_seen
                factor_seen = kwargs.get("normal_factor_diag") is not None
                return original(*args, **kwargs)

            dc_se.observability_rank_details = counted_rank_details
            try:
                result = estimator.estimate()
            finally:
                dc_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertTrue(factor_seen)

    def test_estimate_passes_file_weights_to_normal_equation_builder(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        self.assertTrue(hasattr(dc_se, "build_normal_equations"))
        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
        )

        original = dc_se.build_normal_equations
        non_unit_weight_seen = False

        def counted_builder(H, residual, weight):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight)

        dc_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate()
        finally:
            dc_se.build_normal_equations = original

        self.assertTrue(result.converged)
        self.assertTrue(non_unit_weight_seen)

    def test_estimate_uses_cholesky_solver_when_available(self):
        import secore.se_math as se_math
        from secore.dc_se import DCStateEstimator

        if se_math.CHO_FACTOR is None or se_math.CHO_SOLVE is None:
            self.skipTest("SciPy Cholesky solver is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
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
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
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

    def test_analytic_jacobian_matches_finite_difference(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "dc" / "dc_net_30.meas",
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

        self.assertLess(float(np.max(np.abs(H - H_num))), 1e-6)


if __name__ == "__main__":
    unittest.main()
