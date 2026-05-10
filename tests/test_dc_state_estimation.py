import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class DCStateEstimationTest(unittest.TestCase):
    def test_dc_state_layout_and_nonflat_seed_use_cached_arrays(self):
        import contextlib
        import io

        from secore.dc_se import DCStateEstimator

        class NonIterable:
            def __iter__(self):
                raise AssertionError("DC initial_state should use cached seed arrays")

        with contextlib.redirect_stdout(io.StringIO()):
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=False,
            )
        expected = estimator.initial_state()

        layout = estimator.state_layout()
        self.assertIs(layout["state_meta"], estimator.state_meta)
        self.assertIs(layout["voltage_col"], estimator.voltage_col)
        self.assertEqual(estimator.n_state, layout["n_state"])

        estimator.voltage_state_pos = NonIterable()
        estimator.zero_branches = NonIterable()
        estimator.dcdc_converters = NonIterable()
        estimator.v_generators = NonIterable()

        np.testing.assert_allclose(estimator.initial_state(), expected, atol=0.0)

    def test_summary_cache_uses_table_and_primes_voltage_observation_cache(self):
        from model.meas_model import (
            MEAS_STATUS_PSEUDO,
            Measurement,
            MeasurementList,
            measurement_table_from_measurements,
        )
        from secore.dc_se import DCStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("summary cache should use the cached table")

        rows = [
            Measurement(1, "node_v", "DCNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "node_v_status_pseudo", "DCNode", "n2", "V", 3.0, True, 0.99, MEAS_STATUS_PSEUDO),
        ]
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1), "n2": SimpleNamespace(idx=2)}
        estimator.node_pos = {1: 0, 2: 1}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}
        estimator.branch_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        estimator.dcdc_by_name = {}

        estimator._refresh_measurement_summary_cache()

        self.assertEqual({("DCNode", "n1"), ("DCNode", "n2")}, estimator._active_device_key_cache)
        self.assertEqual(2, estimator._max_measurement_idx)
        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_nodes())

    def test_measurement_parser_uses_status_fast_path_without_status_column(self):
        import secore.dc_se as dc_se
        from model.meas_model import MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@idx name dev_type dev_name meas_type weight valid value",
                        "#1 v1 DCNode n1 V 2.0 1 1.01",
                        "#2 v2 DCNode n2 V 2.0 0 0.99",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            original_normalize = dc_se.normalize_measurement_status

            def fail_normalize(*_args, **_kwargs):
                raise AssertionError("rows without status should use direct status codes")

            dc_se.normalize_measurement_status = fail_normalize
            try:
                measurements = dc_se._read_measurements_direct(meas_file)
            finally:
                dc_se.normalize_measurement_status = original_normalize

        self.assertEqual([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID], [meas.status for meas in measurements])
        np.testing.assert_array_equal(measurements.table.status_code, np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID]))
        np.testing.assert_array_equal(measurements.table.valid, np.array([True, False]))

    def test_summary_cache_maps_only_voltage_rows(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("summary cache should use the cached table")

        rows = [
            Measurement(1, "node_v", "DCNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "load_p", "DCLoad", "load_1", "P_LOAD", 2.0, True, 0.5),
            Measurement(3, "load_i", "DCLoad", "load_1", "I_LOAD", 2.0, True, 0.2),
        ]
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1)}
        estimator.node_pos = {1: 0}
        estimator.generator_by_name = {}
        estimator.load_by_name = {"load_1": SimpleNamespace(node=1)}
        estimator.branch_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        estimator.dcdc_by_name = {}
        mapped = []

        def voltage_mapper(_device_type, device_name, meas_type):
            if not str(meas_type).startswith("V"):
                raise AssertionError("non-voltage rows should not be passed to the voltage mapper")
            mapped.append((device_name, meas_type))
            return {"n1": 1}.get(device_name)

        estimator._voltage_measurement_node_idx = voltage_mapper

        estimator._refresh_measurement_summary_cache()

        self.assertEqual([("n1", "V")], mapped)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_node_cache)
        self.assertIn(("DCLoad", "load_1", "P_LOAD"), estimator._active_measurement_key_cache)

    def test_real_voltage_observation_uses_table_and_maps_only_voltage_rows(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("real voltage observation should use the cached table")

        rows = [
            Measurement(1, "node_v", "DCNode", "n1", "V", 2.0, True, 1.02),
            Measurement(2, "load_p", "DCLoad", "load_1", "P_LOAD", 2.0, True, 0.5),
            Measurement(3, "load_i", "DCLoad", "load_1", "I_LOAD", 2.0, True, 0.2),
        ]
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = NoIterMeasurementList(rows, measurement_table_from_measurements(rows))
        estimator.node_pos = {1: 0}
        mapped = []

        def voltage_mapper(_device_type, device_name, meas_type):
            if not str(meas_type).startswith("V"):
                raise AssertionError("non-voltage rows should not be passed to the voltage mapper")
            mapped.append((device_name, meas_type))
            return {"n1": 1}.get(device_name)

        estimator._voltage_measurement_node_idx = voltage_mapper

        observed = estimator._real_voltage_observation_nodes()

        self.assertEqual([("n1", "V")], mapped)
        self.assertEqual({1: 1.02}, observed)

    def test_conversion_primes_summary_and_voltage_observation_cache(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator

        rows = [Measurement(1, "node_v", "DCNode", "n1", "V", 2.0, True, 10.2)]
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = MeasurementList(rows, measurement_table_from_measurements(rows))
        estimator._node_measurement_scale_by_name = {"n1": 10.0}
        estimator._branch_measurement_scale_by_name = {}
        estimator._break_measurement_scale_by_name = {}
        estimator._zero_branch_measurement_scale_by_name = {}
        estimator._dcdc_measurement_scale_by_name = {}
        estimator._generator_measurement_scale_by_name = {}
        estimator._load_measurement_scale_by_name = {}
        estimator._constraint_measurement_scale_by_name = {}
        estimator.node_by_name = {"n1": SimpleNamespace(idx=1)}
        estimator.node_pos = {1: 0}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}
        estimator.branch_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.break_by_name = {}
        estimator.dcdc_by_name = {}

        estimator._convert_measurements_to_pu()

        self.assertEqual({("DCNode", "n1")}, estimator._active_device_key_cache)
        self.assertEqual({("DCNode", "n1", "V")}, estimator._active_measurement_key_cache)
        self.assertEqual(1, estimator._max_measurement_idx)
        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_nodes())

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

        source = ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"
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
                + "# 1   brk_1_2   1      2      1      1        0 0\n"
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
        self.assertEqual("brk_1_2", ppc["break_name"][0])
        self.assertEqual(1, int(ppc["break"][0, SWITCH_COLS["i_node"]]))
        self.assertEqual(2, int(ppc["break"][0, SWITCH_COLS["j_node"]]))
        self.assertEqual(1, len(network.breakers))
        self.assertIsInstance(network.breakers[0], DCBreak)
        self.assertEqual("brk_1_2", network.breakers[0].name)
        self.assertTrue(network.node_dict[1].isl_obj is network.node_dict[2].isl_obj)

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
            measurements = DCStateEstimator._load_measurements(ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas")
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.DCPowerNetwork.check_topo = original_check_topo

        self.assertTrue(estimator.nodes)

    def test_estimator_load_network_uses_ppc_topology_arrays_without_object_topology(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original_prepare_topology = dc_se.network_topology.prepare_dc_topology

        def reject_object_topology(*_args, **_kwargs):
            raise AssertionError("DC SE array load should apply ppc topology arrays directly")

        dc_se.network_topology.prepare_dc_topology = reject_object_topology
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.network_topology.prepare_dc_topology = original_prepare_topology

        self.assertTrue(estimator.nodes)
        self.assertIsNotNone(getattr(estimator.network, "_topology_arrays", None))

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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
            )

        pseudo = [meas for meas in estimator.active_measurements if meas.name.startswith("pseudo_")]
        pseudo_keys = {(meas.device_type, meas.device_name, meas.meas_type) for meas in pseudo}

        self.assertIn(("DCGenerator", "gen_v1", "P_GEN"), pseudo_keys)
        self.assertTrue(all(0.0 < meas.weight < 1.0 for meas in pseudo))

        gen_p = next(
            meas
            for meas in pseudo
            if meas.device_type == "DCGenerator"
            and meas.device_name == "gen_v1"
            and meas.meas_type == "P_GEN"
        )
        self.assertAlmostEqual(gen_p.value, estimator.generator_by_name["gen_v1"].p)

    def test_dc_unmetered_load_pseudo_measurements_cover_all_unmetered_loads(self):
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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
            )

        selected_loads = {
            load.name
            for load in estimator.load_by_name.values()
        }
        pseudo_loads = [
            meas
            for meas in estimator.active_measurements
            if meas.device_type == "DCLoad"
            and meas.name.startswith(("pseudo_p_", "pseudo_v_"))
        ]
        pseudo_load_devices = {meas.device_name for meas in pseudo_loads}
        pseudo_load_keys = {(meas.device_name, meas.meas_type) for meas in pseudo_loads}

        self.assertEqual(selected_loads, pseudo_load_devices)
        voltage_covered_loads = {
            load_name
            for load_name in selected_loads
            if estimator._voltage_pseudo_is_covered("DCLoad", load_name, "V_LOAD")
        }
        self.assertEqual(2 * len(selected_loads) - len(voltage_covered_loads), len(pseudo_loads))
        for load_name in selected_loads:
            self.assertIn((load_name, "P_LOAD"), pseudo_load_keys)
            if load_name in voltage_covered_loads:
                self.assertNotIn((load_name, "V_LOAD"), pseudo_load_keys)
            else:
                self.assertIn((load_name, "V_LOAD"), pseudo_load_keys)

    def test_reference_nodes_use_highest_degree_nodes_with_valid_voltage_measurements(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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

    def test_targeted_node_voltage_state_adds_pseudo_measurement(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "no_real_voltage.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 p_bad DCLoad load_1 P_LOAD 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )
        next_idx = max(meas.idx for meas in estimator.measurements) + 1
        existing_keys = set()
        existing_names = set()
        target_col, target_meta = next(
            (idx, meta)
            for idx, meta in enumerate(estimator.state_meta)
            if meta.kind == "voltage" and meta.device_type == "DCNode"
        )

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            target_col,
            existing_keys,
            existing_names,
            1,
        )

        self.assertEqual(1, added)
        self.assertIn(("DCNode", target_meta.device_name, "V"), existing_keys)

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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
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

    def test_dc_converter_voltage_pseudo_is_skipped_per_terminal_when_node_has_real_voltage(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "converter_terminal_voltage.meas"
            meas_file.write_text(
                "\n".join(
                    [
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v_nd_9 DCNode nd_9 V 1.0 1 100",
                        "# 2 p_conv_bad DCDCConverter conv_2 P_FROM 1.0 0 0",
                        "</Measurement>",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_") and not meas.name.startswith("pseudo_obs_")
        }

        self.assertIn(("DCDCConverter", "conv_2", "P_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "P_TO"), pseudo_keys)
        self.assertNotIn(("DCDCConverter", "conv_2", "V_FROM"), pseudo_keys)
        self.assertIn(("DCDCConverter", "conv_2", "V_TO"), pseudo_keys)

    def test_dc_pseudo_measurements_reuse_measurement_summary_cache(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )

        def fail_redundant_scan(*_args, **_kwargs):
            raise AssertionError("pseudo measurement preparation should use one summary scan")

        estimator._active_device_keys = fail_redundant_scan
        estimator._active_measurement_keys = fail_redundant_scan
        estimator._add_pseudo_power_measurements()

        pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.measurements
            if meas.name.startswith("pseudo_")
        }
        self.assertNotIn(("DCBreak", "sw_0_1", "V_FROM"), pseudo_keys)
        self.assertTrue(hasattr(estimator, "_active_device_key_cache"))
        self.assertTrue(hasattr(estimator, "_active_measurement_key_cache"))

    def test_dc_constraint_measurements_update_measurement_summary_cache(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )
        estimator._refresh_measurement_summary_cache()
        estimator._add_zero_branch_constraint_measurements()

        self.assertIn(
            ("DCBreakConstraint", "sw_0_1", "V_DIFF"),
            estimator._active_measurement_key_cache,
        )
        self.assertEqual(
            max(int(meas.idx) for meas in estimator.measurements),
            estimator._max_measurement_idx,
        )

    def test_adds_low_weight_pseudo_pv_measurements_for_unmetered_dc_topology_devices(self):
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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=meas_file,
                flat_start=True,
            )

        regular_pseudo_keys = {
            (meas.device_type, meas.device_name, meas.meas_type)
            for meas in estimator.active_measurements
            if meas.name.startswith("pseudo_") and not meas.name.startswith("pseudo_obs_")
        }

        self.assertNotIn(("DCNode", "nd_1", "V"), regular_pseudo_keys)
        for meas_type in ("P_FROM",):
            self.assertIn(("DCBreak", "sw_0_1", meas_type), regular_pseudo_keys)
            self.assertIn(("DCZeroBranch", "zbr_1_2", meas_type), regular_pseudo_keys)
        self.assertNotIn(("DCBreak", "sw_0_1", "V_FROM"), regular_pseudo_keys)
        self.assertNotIn(("DCZeroBranch", "zbr_1_2", "V_FROM"), regular_pseudo_keys)
        self.assertNotIn(("DCBreak", "sw_0_1", "I_FROM"), regular_pseudo_keys)
        self.assertNotIn(("DCZeroBranch", "zbr_1_2", "I_FROM"), regular_pseudo_keys)
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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
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
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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

    def test_evaluate_returns_after_full_vectorized_fill_without_iterating_measurements(self):
        from model.meas_model import MeasurementList
        from secore.dc_se import DCStateEstimator

        class NoIterMeasurementList(MeasurementList):
            def __iter__(self):
                raise AssertionError("fully vectorized DC evaluate should not iterate measurement objects")

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )
        x = estimator.initial_state()
        expected = estimator.evaluate(x)
        wrapped = NoIterMeasurementList(estimator.active_measurements, estimator.active_measurements.table)

        actual = estimator.evaluate(x, wrapped)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_sparse_jacobian_batches_device_measurements(self):
        from scipy.sparse import issparse
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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

    def test_active_measurement_plan_uses_table_without_iterating_measurements(self):
        from model.meas_model import Measurement, MeasurementTable
        from secore.dc_se import DCStateEstimator

        class TableBackedSequence:
            def __init__(self, table):
                self.table = table

            def __len__(self):
                return len(self.table.idx)

            def __iter__(self):
                raise AssertionError("active DC measurement plan should use the cached table")

        table = MeasurementTable(
            idx=np.array([1], dtype=np.int64),
            name=np.array(["m1"], dtype=object),
            device_type=np.array(["DCNode"], dtype=object),
            device_name=np.array(["n1"], dtype=object),
            meas_type=np.array(["V"], dtype=object),
            weight=np.array([1.0], dtype=np.float64),
            valid=np.array([True], dtype=bool),
            value=np.array([1.0], dtype=np.float64),
            device_type_code=np.array([11], dtype=np.int16),
            angle_mask=np.array([False], dtype=bool),
        )
        measurements = TableBackedSequence(table)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.active_measurements = measurements
        estimator.active_measurement_table = table
        estimator._measurement_plan_cache = {}
        estimator._node_plan_by_name = {"n1": (0, 0)}
        estimator._branch_plan_by_name = {}
        estimator._load_plan_by_name = {}
        estimator._generator_plan_by_name = {}
        estimator._zero_branch_plan_by_name = {}
        estimator._break_plan_by_name = {}
        estimator._constraint_plan_by_name = {}
        estimator._dcdc_plan_by_name = {}

        plan = estimator._measurement_plan(measurements)

        np.testing.assert_array_equal(plan["node_rows"], np.array([0]))

    def test_measurement_plan_ignores_rows_without_device_position(self):
        from model.meas_model import MeasurementTable
        from secore.dc_se import DCStateEstimator

        class TableBackedSequence:
            def __init__(self, table):
                self.table = table

            def __len__(self):
                return len(self.table.idx)

            def __iter__(self):
                raise AssertionError("active DC measurement plan should use the cached table")

        table = MeasurementTable(
            idx=np.array([1], dtype=np.int64),
            name=np.array(["m1"], dtype=object),
            device_type=np.array(["DCNode"], dtype=object),
            device_name=np.array(["missing"], dtype=object),
            meas_type=np.array(["V"], dtype=object),
            weight=np.array([1.0], dtype=np.float64),
            valid=np.array([True], dtype=bool),
            value=np.array([1.0], dtype=np.float64),
            device_type_code=np.array([11], dtype=np.int16),
            angle_mask=np.array([False], dtype=bool),
        )
        measurements = TableBackedSequence(table)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator._measurement_plan_cache = {}
        estimator._node_plan_by_name = {}
        estimator._branch_plan_by_name = {}
        estimator._load_plan_by_name = {}
        estimator._generator_plan_by_name = {}
        estimator._zero_branch_plan_by_name = {}
        estimator._break_plan_by_name = {}
        estimator._constraint_plan_by_name = {}
        estimator._dcdc_plan_by_name = {}

        plan = estimator._measurement_plan(measurements)

        self.assertEqual(0, plan["node_rows"].size)
        np.testing.assert_array_equal(plan["handled_mask"], np.array([False], dtype=bool))

    def test_refresh_active_measurements_reuses_all_active_measurement_table(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator

        measurements = MeasurementList(
            [
                Measurement(1, "m1", "DCNode", "n1", "V", 1.0, True, 1.0),
                Measurement(2, "m2", "DCLoad", "l1", "P_LOAD", 1.0, True, 0.2),
            ]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = measurements
        estimator.n_state = 1
        estimator._measurement_plan = lambda active_measurements: {}

        estimator._refresh_active_measurement_indexes()

        self.assertIs(estimator.active_measurements, measurements)
        self.assertIs(estimator.active_measurement_table, measurements.table)
        np.testing.assert_allclose(estimator.active_z, measurements.table.value)

    def test_targeted_pseudo_small_batch_avoids_full_active_refresh(self):
        from secore.dc_se import DCStateEstimator, ObservabilityResult

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
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
        target_pos = int(np.flatnonzero(estimator.voltage_col >= 0)[0])
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
            AssertionError("DC targeted pseudo append should update active layout incrementally")
        )

        added = estimator._add_targeted_observability_pseudo_measurements()

        self.assertEqual(1, added)
        self.assertEqual(initial_active_count + 1, len(estimator.active_measurements))

    def test_incremental_updater_reuses_shared_se_array_plan_helpers(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator, Measurement

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        additions = [Measurement(500000, "dc_v2", "DCNode", "dc1", "V", 5.0, True, 1.01)]
        estimator.measurements.extend(additions)
        calls = {"active": 0}
        original_active = dc_se.append_active_measurement_view

        def counted_active(*args, **kwargs):
            calls["active"] += 1
            return original_active(*args, **kwargs)

        dc_se.append_active_measurement_view = counted_active
        try:
            refreshed = estimator._incremental_update_active_measurement_indexes(additions)
        finally:
            dc_se.append_active_measurement_view = original_active

        self.assertTrue(refreshed)
        self.assertEqual(1, calls["active"])

    def test_incremental_updater_reuses_existing_active_measurement_plan(self):
        from secore.dc_se import DCStateEstimator, Measurement

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        additions = [Measurement(500001, "dc_v3", "DCNode", "dc1", "V", 5.0, True, 1.02)]
        estimator.measurements.extend(additions)
        original_plan = estimator._measurement_plan
        active_len = len(estimator.active_measurements)
        calls = {"full": 0, "append": 0}

        def counted_plan(measurements):
            if len(measurements) == active_len + 1:
                calls["full"] += 1
            elif len(measurements) == 1:
                calls["append"] += 1
            return original_plan(measurements)

        estimator._measurement_plan = counted_plan
        try:
            refreshed = estimator._incremental_update_active_measurement_indexes(additions)
        finally:
            estimator._measurement_plan = original_plan

        self.assertTrue(refreshed)
        self.assertEqual(0, calls["full"])
        self.assertEqual(1, calls["append"])

    def test_bad_data_removal_reuses_table_backed_measurement_subset(self):
        from types import SimpleNamespace
        from model.meas_model import BadDataItem, EstimateResult, MeasurementList, ObservabilityResult, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator, Measurement

        estimator = DCStateEstimator.__new__(DCStateEstimator)
        m1 = Measurement(1, "m1", "DCNode", "n1", "V", 1.0, True, 1.0)
        m2 = Measurement(2, "m2", "DCNode", "n2", "V", 1.0, True, 1.0)
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
        from secore.dc_se import DCStateEstimator, Measurement

        estimator = DCStateEstimator.__new__(DCStateEstimator)
        m1 = Measurement(1, "m1", "DCNode", "n1", "V", 1.0, True, 1.0)
        m2 = Measurement(2, "m2", "DCNode", "n2", "V", 1.0, True, 1.0)
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

    def test_disable_unavailable_measurements_updates_cached_table(self):
        from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
        from secore.dc_se import DCStateEstimator

        measurements = MeasurementList(
            [Measurement(1, "m1", "DCNode", "missing", "V", 1.0, True, 1.0)]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = measurements
        estimator.node_by_name = {}
        estimator.branch_by_name = {}
        estimator.break_by_name = {}
        estimator.zero_branch_by_name = {}
        estimator.generator_by_name = {}
        estimator.load_by_name = {}
        estimator.dcdc_by_name = {}
        estimator.n_state = 1
        estimator._measurement_plan = lambda active_measurements: {}

        estimator._disable_unavailable_measurements()
        estimator._refresh_active_measurement_indexes()

        self.assertFalse(measurements[0].valid)
        self.assertFalse(bool(measurements.table.valid[0]))
        self.assertEqual(0, len(estimator.active_measurements))

    def test_prepare_preserves_provided_measurement_list_cache(self):
        from secore.dc_se import DCStateEstimator

        network = DCStateEstimator._load_network(ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e")
        measurements = DCStateEstimator._load_measurements(ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas")
        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            network=network,
            measurements=measurements,
            flat_start=True,
            prepare_active_measurements=False,
        )

        self.assertIs(estimator.measurements, measurements)
        self.assertIsNotNone(estimator.measurements.table)

    def test_apply_state_batches_device_value_calculation(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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

    def test_estimate_reuses_observability_jacobian_for_first_wls_iteration(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            max_iter=20,
        )

        observability = estimator.observability_analysis()
        original_jacobian = estimator.jacobian_sparse
        call_count = 0

        def counted_jacobian(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_jacobian(*args, **kwargs)

        estimator.jacobian_sparse = counted_jacobian
        result = estimator.estimate(observability=observability)

        self.assertTrue(result.converged)
        self.assertIs(observability, result.observability)
        self.assertEqual(result.iterations - 1, call_count)

    def test_active_sparse_jacobian_reuses_fixed_pattern_builder(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )

        self.assertTrue(hasattr(estimator, "_jacobian_builder"))
        self.assertTrue(estimator._jacobian_builder._assume_fixed_pattern)

        x = estimator.initial_state()
        first = estimator.jacobian_sparse(x)
        second = estimator.jacobian_sparse(x)

        np.testing.assert_array_equal(first.indptr, second.indptr)
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_allclose(first.data, second.data)

    def test_estimate_reuses_fixed_pattern_normal_equation_solver(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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

        self.assertTrue(hasattr(dc_se, "NormalEquationSolver"))
        original_solver = dc_se.NormalEquationSolver
        dc_se.NormalEquationSolver = SpyNormalSolver
        try:
            result = estimator.estimate()
        finally:
            dc_se.NormalEquationSolver = original_solver

        self.assertTrue(result.converged)
        self.assertEqual(1, len(solver_instances))
        self.assertTrue(solver_instances[0].assume_fixed_pattern)
        self.assertEqual(1, solver_instances[0].solve_calls)

    def test_estimate_reuses_normal_equation_structural_pattern(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            max_iter=5,
        )

        original_builder = dc_se.build_normal_equations
        builder_calls = []

        def counted_builder(H, residual, weight, **kwargs):
            builder_calls.append(kwargs)
            return original_builder(H, residual, weight, **kwargs)

        class ZeroStepSolver:
            def __init__(self, assume_fixed_pattern=False):
                pass

            def solve(self, gain, rhs, return_factor_diag=True):
                return np.zeros_like(rhs), np.ones_like(rhs)

        self.assertTrue(hasattr(dc_se, "NormalEquationSolver"))
        original_solver = dc_se.NormalEquationSolver
        dc_se.build_normal_equations = counted_builder
        dc_se.NormalEquationSolver = ZeroStepSolver
        try:
            result = estimator.estimate()
        finally:
            dc_se.build_normal_equations = original_builder
            dc_se.NormalEquationSolver = original_solver

        self.assertTrue(result.converged)
        self.assertTrue(builder_calls)
        self.assertIsNotNone(builder_calls[0].get("normal_pattern"))
        self.assertTrue(builder_calls[0].get("assume_normal_pattern_matches"))
        self.assertIn("weighted_residual", builder_calls[0])

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
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=True,
            )
        finally:
            dc_se.DCPowerFlowCalc.run = original_run

        self.assertTrue(estimator.flat_start)
        self.assertEqual(0, call_count)

    def test_nonflat_start_runs_measurement_seeded_power_flow(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original_seed = getattr(dc_se.DCStateEstimator, "_run_power_flow_seed", None)
        calls = []

        def fake_seed(network, _params, _e_file):
            nd_1 = network.node_dict[1]
            self.assertAlmostEqual(1.6, float(nd_1.voltage))
            calls.append(True)
            for node in network.nodes:
                if getattr(node, "is_alive", False):
                    node.voltage = 1.23

        dc_se.DCStateEstimator._run_power_flow_seed = staticmethod(fake_seed)
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=False,
            )
        finally:
            if original_seed is not None:
                dc_se.DCStateEstimator._run_power_flow_seed = staticmethod(original_seed)

        self.assertFalse(estimator.flat_start)
        self.assertTrue(calls)
        voltage, _switch_current, _dcdc_power, _vgen_power = estimator._unpack_state(estimator.initial_state())
        voltage_state = voltage[estimator.voltage_state_pos]
        np.testing.assert_allclose(voltage_state, 1.23)

    def test_nonflat_start_syncs_measurement_seed_to_array_power_flow_ppc(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        original_calc = dc_se.DCPowerFlowCalc
        original_sync = DCStateEstimator._sync_dc_network_to_ppc
        calls = []

        def reject_full_sync(*_args, **_kwargs):
            raise AssertionError("cached seed rows should update seed ppc without full network sync")

        class FakePowerFlowCalc:
            def __init__(self, model):
                self.model = model
                self.ppc = getattr(model, "ppc", None)
                self.converged = False
                self.iterations = 0
                self.normF = 0.0
                calls.append(isinstance(self.ppc, dict))

            def run(self, **_kwargs):
                self.testcase.assertTrue(
                    getattr(self, "skip_lf_result", False),
                    "SE LF seed should skip detailed LFResult construction",
                )
                self.testcase.assertAlmostEqual(
                    1.6,
                    float(self.ppc["bus"][0, dc_se.DC_BUS_COLS["voltage"]]),
                )
                self.converged = True
                self.iterations = 1
                for node in self.model.nodes:
                    if getattr(node, "is_alive", False):
                        node.voltage = 1.24
                return 0

        FakePowerFlowCalc.testcase = self
        dc_se.DCPowerFlowCalc = FakePowerFlowCalc
        DCStateEstimator._sync_dc_network_to_ppc = staticmethod(reject_full_sync)
        try:
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
                meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
                flat_start=False,
            )
        finally:
            dc_se.DCPowerFlowCalc = original_calc
            DCStateEstimator._sync_dc_network_to_ppc = staticmethod(original_sync)

        self.assertEqual([True], calls)
        self.assertTrue(estimator.power_flow_seed_converged)
        voltage, _switch_current, _dcdc_power, _vgen_power = estimator._unpack_state(estimator.initial_state())
        np.testing.assert_allclose(voltage[estimator.voltage_state_pos], 1.24)

    def test_power_flow_seed_uses_cached_seed_rows(self):
        from secore.dc_se import DCStateEstimator

        class Device:
            pass

        class FailingMeasurements(list):
            def __iter__(self):
                raise AssertionError("power-flow seed should not rescan all measurements")

        node = Device()
        node.idx = 1
        node.name = "n1"
        node.voltage = 1.0
        gen = Device()
        gen.node = 1
        gen.p_set = 0.0
        gen.p = 0.0
        load = Device()
        load.node = 1
        load.pbase = 1.0
        load.pv0 = 0.0
        load.pv1 = 0.0
        load.pv2 = 0.0
        load.p = 0.0

        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.voltage_floor = 0.1
        estimator.node_by_name = {"n1": node}
        estimator.node_by_idx = {1: node}
        estimator.generator_by_name = {"g1": gen}
        estimator.load_by_name = {"l1": load}
        estimator.dcdc_by_name = {}
        estimator.network = type("Network", (), {"node_dict": {1: node}, "nodes": [node]})()
        estimator.measurements = FailingMeasurements()
        estimator._power_flow_seed_rows = [
            ("DCNode", "n1", "V", 1.05),
            ("DCGenerator", "g1", "P_GEN", 0.7),
            ("DCLoad", "l1", "P_LOAD", 0.3),
        ]

        estimator._apply_measurement_seed_to_network()

        self.assertAlmostEqual(1.05, node.voltage)
        self.assertAlmostEqual(0.7, gen.p_set)
        self.assertAlmostEqual(0.3, load.pv0)

    def test_array_power_flow_seed_defers_object_seed_application(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.network = type("Network", (), {"ppc": {"format": "dc_ppc_v1"}})()
        estimator.measurements = []
        estimator._power_flow_seed_rows = [("DCNode", "n1", "V", 1.05)]
        estimator._apply_power_flow_seed_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("array-mode seed should be applied to ppc, not object network")
        )

        estimator._apply_measurement_seed_to_network()

        self.assertEqual(tuple(estimator._power_flow_seed_rows), estimator.network._se_power_flow_seed_rows)

    def test_estimate_uses_precomputed_observability_without_reanalysis(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )
        observability = estimator.observability_analysis()

        original = dc_se.observability_rank_details

        def counted_rank_details(*args, **kwargs):
            raise AssertionError("estimate should reuse precomputed observability and not re-run rank analysis")

        dc_se.observability_rank_details = counted_rank_details
        try:
            result = estimator.estimate(observability=observability)
        finally:
            dc_se.observability_rank_details = original

        self.assertTrue(result.converged)
        self.assertIs(observability, result.observability)

    def test_estimate_reuses_observability_normal_pattern_cache(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )
        observability = estimator.observability_analysis()

        original_builder = dc_se.build_normal_equations
        normal_pattern_seen = False

        def counted_builder(H, residual, weight, **kwargs):
            nonlocal normal_pattern_seen
            normal_pattern_seen = kwargs.get("normal_pattern") is not None
            return original_builder(H, residual, weight, **kwargs)

        dc_se.build_normal_equations = counted_builder
        try:
            result = estimator.estimate(observability=observability)
        finally:
            dc_se.build_normal_equations = original_builder

        self.assertTrue(result.converged)
        self.assertTrue(normal_pattern_seen)

    def test_estimate_passes_file_weights_to_normal_equation_builder(self):
        import secore.dc_se as dc_se
        from secore.dc_se import DCStateEstimator

        self.assertTrue(hasattr(dc_se, "build_normal_equations"))
        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )

        original = dc_se.build_normal_equations
        non_unit_weight_seen = False

        def counted_builder(H, residual, weight, **kwargs):
            nonlocal non_unit_weight_seen
            non_unit_weight_seen = bool(np.any(weight != 1.0))
            return original(H, residual, weight, **kwargs)

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
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
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

    def test_cli_runs_observability_before_estimation_and_does_not_repeat_it(self):
        import contextlib
        import io
        import secore.dc_se as dc_se

        events = []
        original_prepare = dc_se.DCStateEstimator.prepare
        original_observability = dc_se.DCStateEstimator.observability_analysis
        original_estimate = dc_se.DCStateEstimator.estimate
        original_run = dc_se.DCStateEstimator.run
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
            test_case.assertTrue(getattr(self, "_prepared", False))
            return original_run(self, *args, **kwargs)

        output = io.StringIO()
        dc_se.DCStateEstimator.prepare = counted_prepare
        dc_se.DCStateEstimator.observability_analysis = counted_observability
        dc_se.DCStateEstimator.estimate = counted_estimate
        dc_se.DCStateEstimator.run = counted_run
        try:
            with contextlib.redirect_stdout(output):
                code = dc_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            dc_se.DCStateEstimator.prepare = original_prepare
            dc_se.DCStateEstimator.observability_analysis = original_observability
            dc_se.DCStateEstimator.estimate = original_estimate
            dc_se.DCStateEstimator.run = original_run

        self.assertEqual(0, code)
        self.assertEqual("prepare", events[0])
        self.assertEqual(1, events.count("prepare"))
        self.assertEqual(["observability", "estimate"], events[-2:])
        self.assertEqual(1, output.getvalue().count("Observability:"))
        self.assertLess(output.getvalue().index("Observability:"), output.getvalue().index("State estimation:"))

    def test_cli_does_not_build_seresult_without_output_file(self):
        import contextlib
        import io
        import secore.dc_se as dc_se

        original_build = dc_se.DCStateEstimator.build_se_result

        def reject_build(*_args, **_kwargs):
            raise AssertionError("SEResult details should be built only when --se-result is requested")

        dc_se.DCStateEstimator.build_se_result = reject_build
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = dc_se.main(
                    [
                        "--case",
                        str(ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"),
                        "--meas",
                        str(ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas"),
                        "--flat-start",
                        "--quiet",
                    ]
                )
        finally:
            dc_se.DCStateEstimator.build_se_result = original_build

        self.assertEqual(0, code)

    def test_run_summary_return_mode_limits_seresult_only(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            auto_prepare=False,
        )
        self.assertFalse(estimator._prepared)
        estimator.prepare()
        se_result = estimator.run(return_mode="summary", verbose=False, skip_bad_data=True)
        result = estimator.estimate_result

        self.assertIs(se_result, estimator.se_result)
        self.assertTrue(result.converged)
        self.assertIs(estimator.observability_result, result.observability)
        self.assertFalse(hasattr(result, "return_mode"))
        self.assertGreater(result.x.size, 0)
        self.assertGreater(result.z_est.size, 0)
        self.assertGreater(result.residual.size, 0)
        self.assertEqual(result.iterations, se_result.statistics.iterations)
        self.assertEqual(0, len(se_result.prefiltered_measurements))
        self.assertEqual(0, len(se_result.pseudo_measurements))
        self.assertEqual(0, len(se_result.bad_data))
        self.assertEqual(0, len(se_result.normal_measurements))

    def test_run_array_return_mode_keeps_estimate_arrays_only(self):
        import secore.dc_se as dc_se_module
        from secore.dc_se import DCStateEstimator
        from secore.se_result import SEResult

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            auto_prepare=False,
        )
        estimator.prepare()
        original_build = DCStateEstimator.build_se_result
        original_summary = dc_se_module.build_seresult_summary
        original_identify = DCStateEstimator.identify_bad_data
        original_from_estimate = SEResult.from_estimate_result

        def reject_seresult_path(*_args, **_kwargs):
            raise AssertionError("array return_mode should not build SEResult payloads")

        def reject_bad_data(*_args, **_kwargs):
            raise AssertionError("array return_mode should not run post-estimation bad-data analysis")

        def reject_full_tables(*_args, **_kwargs):
            raise AssertionError("array return_mode should not build full SEResult measurement tables")

        DCStateEstimator.build_se_result = reject_seresult_path
        dc_se_module.build_seresult_summary = reject_seresult_path
        DCStateEstimator.identify_bad_data = reject_bad_data
        SEResult.from_estimate_result = reject_full_tables
        try:
            se_result = estimator.run(return_mode="array", verbose=False)
        finally:
            DCStateEstimator.build_se_result = original_build
            dc_se_module.build_seresult_summary = original_summary
            DCStateEstimator.identify_bad_data = original_identify
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
        self.assertEqual([], estimator.bad_items)
        self.assertEqual(0, estimator.normalized_residual.size)

    def test_observability_uses_cholesky_fast_path_when_observable(self):
        from secore.dc_se import DCStateEstimator

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = self._all_valid_measurement_file(tmp_dir, ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas")
            estimator = DCStateEstimator(
                e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
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
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
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
