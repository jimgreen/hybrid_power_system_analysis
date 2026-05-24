import unittest
from pathlib import Path
import tempfile

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class DCStateEstimationTest(unittest.TestCase):
    @staticmethod
    def _prime_minimal_dc_node_plan(estimator) -> None:
        from model.meas_type import DEVICE_TYPE_DCNode, MEAS_TYPE_V

        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        estimator._measurement_plan_device_pos_by_type_code = {}
        lookup = np.full(MEAS_TYPE_V + 1, -1, dtype=np.int16)
        lookup[MEAS_TYPE_V] = MEAS_TYPE_V
        estimator._measurement_plan_meas_kind_code_by_type_code = {DEVICE_TYPE_DCNode: lookup}
        estimator._node_plan_node_pos = np.array([0], dtype=np.int64)
        estimator._node_plan_col = np.array([0], dtype=np.int64)
        estimator._branch_plan_i = empty_i
        estimator._branch_plan_j = empty_i
        estimator._branch_plan_i_col = empty_i
        estimator._branch_plan_j_col = empty_i
        estimator._branch_plan_inv_r = empty_f
        estimator._load_plan_pos = empty_i
        estimator._load_plan_col = empty_i
        estimator._load_plan_pv0 = empty_f
        estimator._load_plan_pv1 = empty_f
        estimator._load_plan_pv2 = empty_f
        estimator._generator_plan_ctrl = empty_i
        estimator._generator_plan_pos = empty_i
        estimator._generator_plan_col = empty_i
        estimator._generator_plan_p_col = empty_i
        estimator._generator_plan_vgen_pos = empty_i
        estimator._generator_plan_p_set = empty_f
        estimator._generator_plan_i_set = empty_f
        estimator._zero_branch_plan_i = empty_i
        estimator._zero_branch_plan_j = empty_i
        estimator._zero_branch_plan_i_col = empty_i
        estimator._zero_branch_plan_j_col = empty_i
        estimator._zero_branch_plan_current_col = empty_i
        estimator._zero_branch_plan_current_pos = empty_i
        estimator._break_plan_i = empty_i
        estimator._break_plan_j = empty_i
        estimator._break_plan_i_col = empty_i
        estimator._break_plan_j_col = empty_i
        estimator._break_plan_current_col = empty_i
        estimator._break_plan_current_pos = empty_i
        estimator._constraint_plan_i = empty_i
        estimator._constraint_plan_j = empty_i
        estimator._constraint_plan_i_col = empty_i
        estimator._constraint_plan_j_col = empty_i
        estimator._dcdc_plan_i = empty_i
        estimator._dcdc_plan_j = empty_i
        estimator._dcdc_plan_i_col = empty_i
        estimator._dcdc_plan_j_col = empty_i
        estimator._dcdc_plan_p_col = empty_i
        estimator._dcdc_plan_q_col = empty_i
        estimator._dcdc_plan_pos = empty_i

    @staticmethod
    def _prime_minimal_dc_runtime_arrays(estimator, *, n_nodes: int = 1) -> None:
        empty_i = np.asarray([], dtype=np.int64)
        estimator._node_voltage_file_base_by_pos = np.ones(n_nodes, dtype=np.float64)
        estimator._node_current_file_base_by_pos = np.ones(n_nodes, dtype=np.float64)
        estimator._node_idx_by_pos = np.arange(1, n_nodes + 1, dtype=np.int64)
        estimator._node_idx_lookup_ids = estimator._node_idx_by_pos.copy()
        estimator._node_idx_lookup_pos = np.arange(n_nodes, dtype=np.int64)
        estimator._raw_node_idx_alive = estimator._node_idx_by_pos.copy()
        estimator._raw_node_solver_pos_alive = np.arange(n_nodes, dtype=np.int64)
        estimator.p_base = 1.0
        estimator._branch_i_pos = empty_i
        estimator._branch_j_pos = empty_i
        estimator._zero_branch_i_pos = empty_i
        estimator._zero_branch_j_pos = empty_i
        estimator._switch_i_pos = empty_i
        estimator._switch_j_pos = empty_i
        estimator._break_i_pos = empty_i
        estimator._break_j_pos = empty_i
        estimator._dcdc_i_pos = empty_i
        estimator._dcdc_j_pos = empty_i
        estimator._generator_pos = empty_i
        estimator._load_pos = np.asarray([0], dtype=np.int64)

    @staticmethod
    def _measurement_keys_from_table(estimator, table, mask=None) -> set:
        if mask is None:
            mask = np.ones(int(table.idx.size), dtype=bool)
        rows = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64, copy=False)
        if rows.size == 0:
            return set()
        keys = estimator._active_measurement_key_array(
            np.asarray(table.device_type_code, dtype=np.int16)[rows],
            np.asarray(table.device_pos, dtype=np.int64)[rows],
            np.asarray(table.meas_type_code, dtype=np.int16)[rows],
        )
        return set(keys.tolist())

    def _pseudo_measurement_keys(self, estimator) -> set:
        from model.meas_model import MEAS_STATUS_PSEUDO, measurement_table_status_code

        table = estimator.active_measurement_table
        status = measurement_table_status_code(table)
        return self._measurement_keys_from_table(estimator, table, status == MEAS_STATUS_PSEUDO)

    @staticmethod
    def _device_pos(names: np.ndarray, name: str) -> int:
        matches = np.flatnonzero(np.asarray(names, dtype=object) == str(name))
        if matches.size == 0:
            raise AssertionError(f"missing device name {name}")
        return int(matches[0])

    @staticmethod
    def _dc_measurement_table(
        idx,
        device_type_code,
        device_pos,
        meas_type_code,
        value,
        *,
        weight=None,
        valid=None,
        status_code=None,
    ):
        from model.meas_model import MEAS_STATUS_NORMAL, MeasurementTable

        idx = np.asarray(idx, dtype=np.int64)
        row_count = int(idx.size)
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        device_pos = np.asarray(device_pos, dtype=np.int64)
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
        value = np.asarray(value, dtype=np.float64)
        if weight is None:
            weight = np.ones(row_count, dtype=np.float64)
        else:
            weight = np.asarray(weight, dtype=np.float64)
        if valid is None:
            valid = np.ones(row_count, dtype=bool)
        else:
            valid = np.asarray(valid, dtype=bool)
        if status_code is None:
            status_code = np.full(row_count, MEAS_STATUS_NORMAL, dtype=np.int16)
        else:
            status_code = np.asarray(status_code, dtype=np.int16)
        rows_by_code = {
            int(code): np.flatnonzero(device_type_code == code).astype(np.int64, copy=False)
            for code in np.unique(device_type_code)
        }
        return MeasurementTable(
            idx=idx,
            name=np.asarray([], dtype=object),
            device_type=np.asarray([], dtype=object),
            device_name=np.asarray([], dtype=object),
            meas_type=np.asarray([], dtype=object),
            weight=weight,
            valid=valid,
            value=value,
            device_type_code=device_type_code,
            angle_mask=np.zeros(row_count, dtype=bool),
            status_code=status_code,
            rows_by_device_type_code=rows_by_code,
            device_name_id=np.full(row_count, -1, dtype=np.int64),
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )

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
        from model import meas_type as mt
        from model.meas_model import MEAS_STATUS_PSEUDO, MeasurementTableView
        from secore.dc_se import DCStateEstimator

        table = self._dc_measurement_table(
            [1, 2],
            [mt.DEVICE_TYPE_DCNode, mt.DEVICE_TYPE_DCNode],
            [0, 1],
            [mt.MEAS_TYPE_V, mt.MEAS_TYPE_V],
            [1.02, 0.99],
            weight=[2.0, 3.0],
            status_code=[0, MEAS_STATUS_PSEUDO],
        )
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = MeasurementTableView(table, normalized=True)
        self._prime_minimal_dc_runtime_arrays(estimator, n_nodes=2)

        estimator._refresh_measurement_summary_cache()

        self.assertFalse(hasattr(DCStateEstimator, "_active_device_keys_ref"))
        self.assertFalse(hasattr(estimator, "_active_device_key_cache"))
        self.assertEqual(2, estimator._max_measurement_idx)
        self.assertEqual({1: 1.02}, estimator._node_voltage_measurement_cache)
        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_nodes())

    def test_summary_cache_maps_only_voltage_rows(self):
        from model import meas_type as mt
        from model.meas_model import MeasurementTableView
        from secore.dc_se import DCStateEstimator

        table = self._dc_measurement_table(
            [1, 2, 3],
            [mt.DEVICE_TYPE_DCNode, mt.DEVICE_TYPE_DCLoad, mt.DEVICE_TYPE_DCLoad],
            [0, 0, 0],
            [mt.MEAS_TYPE_V, mt.MEAS_TYPE_P_LOAD, mt.MEAS_TYPE_I_LOAD],
            [1.02, 0.5, 0.2],
            weight=[2.0, 2.0, 2.0],
        )
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = MeasurementTableView(table, normalized=True)
        self._prime_minimal_dc_runtime_arrays(estimator)

        estimator._refresh_measurement_summary_cache()

        self.assertEqual({1: 1.02}, estimator._real_voltage_observation_node_cache)
        self.assertTrue(all(isinstance(key, int) for key in estimator._active_measurement_key_cache))

    def test_real_voltage_observation_uses_table_and_maps_only_voltage_rows(self):
        from model import meas_type as mt
        from model.meas_model import MeasurementTableView
        from secore.dc_se import DCStateEstimator

        table = self._dc_measurement_table(
            [1, 2, 3],
            [mt.DEVICE_TYPE_DCNode, mt.DEVICE_TYPE_DCLoad, mt.DEVICE_TYPE_DCLoad],
            [0, 0, 0],
            [mt.MEAS_TYPE_V, mt.MEAS_TYPE_P_LOAD, mt.MEAS_TYPE_I_LOAD],
            [1.02, 0.5, 0.2],
            weight=[2.0, 2.0, 2.0],
        )
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.measurements = MeasurementTableView(table, normalized=True)
        self._prime_minimal_dc_runtime_arrays(estimator)

        observed = estimator._real_voltage_observation_nodes()

        self.assertEqual({1: 1.02}, observed)

    def test_conversion_primes_summary_and_voltage_observation_cache(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )

        self.assertFalse(hasattr(estimator, "_active_device_key_cache"))
        self.assertTrue(all(isinstance(key, int) for key in estimator._active_measurement_key_cache))
        self.assertGreater(estimator._max_measurement_idx, 0)
        self.assertTrue(estimator._node_voltage_measurement_cache)
        self.assertTrue(estimator._real_voltage_observation_nodes())

    def test_dc_topology_contracts_closed_switches_to_buses_before_islands(self):
        from model.ppc_topology import build_dc_ppc_with_topology_from_e_file

        case_text = """<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV kW kA
</PowerBase>

<DCNode>
@ idx name vbase voltage isl run_stat
# 1 n1 100 100 0 1
# 2 n2 100 100 0 1
# 3 n3 100 100 0 1
</DCNode>

<DCBranch>
@ idx name i_node j_node r run_stat
# 1 br_2_3 2 3 0.01 1
</DCBranch>

<DCLoad>
@ idx name node pbase pv0 pv1 pv2 run_stat
</DCLoad>

<DCGenerator>
@ idx name node control_type v_set p_set i_set run_stat
# 1 g1 1 V 1.0 1.0 0.0 1
</DCGenerator>

<DCZeroBranch>
@ idx name i_node j_node run_stat
</DCZeroBranch>

<DCSwitch>
@ idx name i_node j_node status run_stat
# 1 sw_1_2 1 2 1 1
# 2 sw_2_3 2 3 0 1
</DCSwitch>

<DCBreak>
@ idx name i_node j_node status run_stat
</DCBreak>

<DCDCConverter>
@ idx name i_node j_node r1 r2 control_type p_set i_set v_set run_stat
</DCDCConverter>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "dc_topology.e"
            case_path.write_text(case_text, encoding="utf-8")
            topology = build_dc_ppc_with_topology_from_e_file(case_path)["_topology_arrays"]

        self.assertEqual([1, 3], topology.bus_ids.tolist())
        self.assertEqual([0, 0, 1], topology.node_to_bus_pos.tolist())
        self.assertEqual([0, 2], topology.island_bus_offsets.tolist())
        self.assertEqual([0, 1], topology.island_bus_indices.tolist())
        switch_topology = topology.devices["switch"]
        self.assertTrue(bool(switch_topology.alive_mask[0]))
        self.assertFalse(bool(switch_topology.alive_mask[1]))
        self.assertEqual(int(switch_topology.i_bus_pos[0]), int(switch_topology.j_bus_pos[0]))
        self.assertNotEqual(int(switch_topology.i_bus_pos[1]), int(switch_topology.j_bus_pos[1]))

    def test_dc_break_is_parsed_as_distinct_zero_tie_device(self):
        from model.dc_array_model import SWITCH_COLS, build_dc_ppc_from_e_file
        from model.ppc_topology import ensure_dc_ppc_topology

        source = ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_path = Path(tmp_dir) / "dc_break.e"
            text = source.read_text(encoding="utf-8")
            switch_start = text.index("<DCSwitch>")
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
            ensure_dc_ppc_topology(ppc)
            break_topology = ppc["_topology_arrays"].devices["break"]

        self.assertEqual(0, ppc["switch"].shape[0])
        self.assertEqual(1, ppc["break"].shape[0])
        self.assertEqual("brk_1_2", ppc["break_name"][0])
        self.assertEqual(1, int(ppc["break"][0, SWITCH_COLS["i_node"]]))
        self.assertEqual(2, int(ppc["break"][0, SWITCH_COLS["j_node"]]))
        self.assertTrue(bool(break_topology.alive_mask[0]))
        self.assertNotEqual(int(break_topology.i_bus_pos[0]), int(break_topology.j_bus_pos[0]))
        self.assertEqual(int(break_topology.i_island_pos[0]), int(break_topology.j_island_pos[0]))

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

    def test_dc_network_load_uses_array_model_by_default(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )

        self.assertTrue(getattr(estimator.network, "_se_lightweight", False))
        self.assertEqual("dc_ppc_v1", estimator.network.ppc["format"])
        self.assertEqual("meas_ppc_v1", estimator.meas_ppc["format"])

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

    def test_adds_low_weight_pseudo_power_measurements_for_unmetered_generators_and_loads(self):
        from model import meas_type as mt
        from model.meas_model import MEAS_STATUS_PSEUDO, measurement_table_status_code
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

        pseudo_keys = self._pseudo_measurement_keys(estimator)
        gen_pos = self._device_pos(estimator._generator_names, "gen_v1")
        gen_p_key = DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCGenerator, gen_pos, mt.MEAS_TYPE_P_GEN)

        self.assertIn(gen_p_key, pseudo_keys)
        table = estimator.active_measurement_table
        pseudo_mask = measurement_table_status_code(table) == MEAS_STATUS_PSEUDO
        self.assertTrue(np.all((table.weight[pseudo_mask] > 0.0) & (table.weight[pseudo_mask] < 1.0)))

    def test_dc_unmetered_load_pseudo_measurements_cover_all_unmetered_loads(self):
        from model import meas_type as mt
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

        pseudo_keys = self._pseudo_measurement_keys(estimator)
        for load_pos in range(int(estimator._load_names.size)):
            p_key = DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCLoad, load_pos, mt.MEAS_TYPE_P_LOAD)
            v_key = DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCLoad, load_pos, mt.MEAS_TYPE_V_LOAD)
            self.assertIn(p_key, pseudo_keys)
            if estimator._voltage_pseudo_is_covered_by_pos(
                mt.DEVICE_TYPE_DCLoad,
                load_pos,
                mt.MEAS_TYPE_V_LOAD,
            ):
                self.assertNotIn(v_key, pseudo_keys)
            else:
                self.assertIn(v_key, pseudo_keys)

    def test_reference_nodes_use_highest_degree_nodes_with_valid_voltage_measurements(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
        )
        expected_refs = ["nd_11", "nd_21", "nd_26"]

        ref_pos = np.asarray(estimator.references, dtype=np.int64)
        self.assertEqual(expected_refs, [str(estimator._node_name_by_pos[int(pos)]) for pos in ref_pos])
        voltage, _switch_current, _dcdc_power, _vgen_power = estimator._unpack_state(estimator.initial_state())
        for pos in ref_pos:
            pos = int(pos)
            node_idx = int(estimator._node_idx_by_pos[pos])
            ref_voltage = estimator.node_voltage_measurements[node_idx]
            self.assertEqual(-1, int(estimator.voltage_col[pos]))
            self.assertAlmostEqual(ref_voltage, voltage[pos])

    def test_targeted_node_voltage_state_adds_pseudo_measurement(self):
        from model import meas_type as mt
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
        next_idx = int(estimator.measurements.table.idx.max()) + 1
        existing_keys = set()
        arrays = estimator._state_meta_arrays_ref()
        target_col = int(
            np.flatnonzero(
                (arrays["kind"] == "voltage")
                & (arrays["device_type_code"] == mt.DEVICE_TYPE_DCNode)
            )[0]
        )
        target_device_pos = int(arrays["device_pos"][target_col])

        _, added = estimator._append_targeted_observability_pseudo(
            next_idx,
            target_col,
            existing_keys,
            1,
        )

        self.assertEqual(1, added)
        expected_key = DCStateEstimator._active_measurement_key(
            mt.DEVICE_TYPE_DCNode,
            target_device_pos,
            mt.MEAS_TYPE_V,
        )
        self.assertIn(expected_key, existing_keys)

    def test_pseudo_measurements_are_measurement_level_for_dc_sources_loads_and_converters(self):
        from model import meas_type as mt
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

        pseudo_keys = self._pseudo_measurement_keys(estimator)
        gen_pos = self._device_pos(estimator._generator_names, "gen_v1")
        load_pos = self._device_pos(estimator._load_names, "load_1")
        conv1_pos = self._device_pos(estimator._dcdc_names, "conv_1")
        conv2_pos = self._device_pos(estimator._dcdc_names, "conv_2")

        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCGenerator, gen_pos, mt.MEAS_TYPE_P_GEN),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCLoad, load_pos, mt.MEAS_TYPE_P_LOAD),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv1_pos, mt.MEAS_TYPE_P_FROM),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv1_pos, mt.MEAS_TYPE_P_TO),
            pseudo_keys,
        )
        self.assertNotIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv1_pos, mt.MEAS_TYPE_V_FROM),
            pseudo_keys,
        )
        for meas_type_code in (mt.MEAS_TYPE_P_FROM, mt.MEAS_TYPE_P_TO, mt.MEAS_TYPE_V_FROM, mt.MEAS_TYPE_V_TO):
            self.assertIn(
                DCStateEstimator._active_measurement_key(
                    mt.DEVICE_TYPE_DCDCConverter,
                    conv2_pos,
                    meas_type_code,
                ),
                pseudo_keys,
            )

    def test_dc_converter_voltage_pseudo_is_skipped_per_terminal_when_node_has_real_voltage(self):
        from model import meas_type as mt
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

        pseudo_keys = self._pseudo_measurement_keys(estimator)
        conv_pos = self._device_pos(estimator._dcdc_names, "conv_2")

        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv_pos, mt.MEAS_TYPE_P_FROM),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv_pos, mt.MEAS_TYPE_P_TO),
            pseudo_keys,
        )
        self.assertNotIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv_pos, mt.MEAS_TYPE_V_FROM),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCDCConverter, conv_pos, mt.MEAS_TYPE_V_TO),
            pseudo_keys,
        )

    def test_dc_pseudo_measurements_reuse_measurement_summary_cache(self):
        from model import meas_type as mt
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )

        def fail_redundant_scan(*_args, **_kwargs):
            raise AssertionError("pseudo measurement preparation should use one summary scan")

        estimator._active_measurement_keys = fail_redundant_scan
        estimator._add_pseudo_power_measurements()

        break_pos = int(np.flatnonzero(estimator._break_names == "sw_0_1")[0])
        break_v_from_key = DCStateEstimator._active_measurement_key(
            mt.DEVICE_TYPE_DCBreak,
            break_pos,
            mt.MEAS_TYPE_V_FROM,
        )
        self.assertNotIn(break_v_from_key, estimator._active_measurement_key_cache)
        self.assertTrue(all(isinstance(key, int) for key in estimator._active_measurement_key_cache))
        self.assertFalse(hasattr(estimator, "_active_device_key_cache"))
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
        _next_idx, added_keys = estimator._add_pseudo_topology_measurements(estimator._next_measurement_idx())

        self.assertTrue(added_keys)
        self.assertTrue(added_keys.issubset(estimator._active_measurement_key_cache))
        self.assertTrue(all(isinstance(key, int) for key in estimator._active_measurement_key_cache))
        self.assertEqual(
            int(estimator.measurements.table.idx.max()),
            estimator._max_measurement_idx,
        )

    def test_adds_low_weight_pseudo_pv_measurements_for_unmetered_dc_topology_devices(self):
        from model import meas_type as mt
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

        pseudo_keys = self._pseudo_measurement_keys(estimator)
        break_pos = self._device_pos(estimator._break_names, "sw_0_1")
        zbr_pos = self._device_pos(estimator._zero_branch_names, "zbr_1_2")

        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCBreak, break_pos, mt.MEAS_TYPE_P_FROM),
            pseudo_keys,
        )
        self.assertIn(
            DCStateEstimator._active_measurement_key(mt.DEVICE_TYPE_DCZeroBranch, zbr_pos, mt.MEAS_TYPE_P_FROM),
            pseudo_keys,
        )
        for device_type_code, device_pos in (
            (mt.DEVICE_TYPE_DCBreak, break_pos),
            (mt.DEVICE_TYPE_DCZeroBranch, zbr_pos),
        ):
            self.assertNotIn(
                DCStateEstimator._active_measurement_key(device_type_code, device_pos, mt.MEAS_TYPE_V_FROM),
                pseudo_keys,
            )
            self.assertNotIn(
                DCStateEstimator._active_measurement_key(device_type_code, device_pos, mt.MEAS_TYPE_I_FROM),
                pseudo_keys,
            )

    def test_dc_zero_branches_are_compressed_like_closed_switches(self):
        from model import meas_type as mt
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

        active_keys = self._measurement_keys_from_table(estimator, estimator.active_measurement_table)
        zbr_pos = self._device_pos(estimator._zero_branch_names, "zbr_1_2")
        for meas_type_code in (mt.MEAS_TYPE_P_FROM, mt.MEAS_TYPE_V_FROM, mt.MEAS_TYPE_I_FROM):
            self.assertIn(
                DCStateEstimator._active_measurement_key(
                    mt.DEVICE_TYPE_DCZeroBranch,
                    zbr_pos,
                    meas_type_code,
                ),
                active_keys,
            )
        arrays = estimator._state_meta_arrays_ref()
        zero_current_mask = (
            (arrays["kind"] == "zero_current")
            & (arrays["device_type_code"] == mt.DEVICE_TYPE_DCZeroBranch)
            & (arrays["device_pos"] == zbr_pos)
        )
        self.assertTrue(np.any(zero_current_mask))

        zbr_i = int(estimator._zero_branch_i_pos[zbr_pos])
        zbr_j = int(estimator._zero_branch_j_pos[zbr_pos])
        self.assertEqual(
            estimator.voltage_col[zbr_i],
            estimator.voltage_col[zbr_j],
        )
        break_pos = self._device_pos(estimator._break_names, "sw_0_1")
        break_i = int(estimator._break_i_pos[break_pos])
        break_j = int(estimator._break_j_pos[break_pos])
        self.assertEqual(
            estimator.voltage_col[break_i],
            estimator.voltage_col[break_j],
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
        self.assertLess(float(normalized.max()), 3e-3)

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
        H = estimator.jacobian_sparse(estimator.initial_state())

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
        dense = estimator._assemble_jacobian(x, sparse=False)
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
        from model.meas_model import MeasurementTableView
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )
        x = estimator.initial_state()
        expected = estimator.evaluate(x)
        wrapped = estimator.active_measurements

        self.assertIsInstance(wrapped, MeasurementTableView)
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
        dense = estimator._assemble_jacobian(x, sparse=False)

        self.assertFalse(hasattr(estimator, "_add_derivative"))
        sparse = estimator.jacobian_sparse(x)

        self.assertTrue(issparse(sparse))
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.toarray(), atol=1e-10)

    def test_active_measurement_arrays_are_cached_for_estimation(self):
        from model.meas_model import MeasurementTableView
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
        )

        self.assertIsInstance(estimator.active_measurements, MeasurementTableView)
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
        from model.meas_type import MEAS_TYPE_V
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
            device_name=np.array(["n1"], dtype=object),
            meas_type=np.array(["V"], dtype=object),
            weight=np.array([1.0], dtype=np.float64),
            valid=np.array([True], dtype=bool),
            value=np.array([1.0], dtype=np.float64),
            device_type_code=np.array([11], dtype=np.int16),
            angle_mask=np.array([False], dtype=bool),
            meas_type_code=np.array([MEAS_TYPE_V], dtype=np.int16),
            device_pos=np.array([0], dtype=np.int64),
        )
        measurements = TableBackedSequence(table)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.active_measurements = measurements
        estimator.active_measurement_table = table
        estimator._measurement_plan_cache = {}
        self._prime_minimal_dc_node_plan(estimator)

        plan = estimator._measurement_plan(measurements)

        np.testing.assert_array_equal(plan["node_rows"], np.array([0]))

    def test_measurement_plan_ignores_rows_without_device_position(self):
        from model.meas_type import MEAS_TYPE_V
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
            meas_type_code=np.array([MEAS_TYPE_V], dtype=np.int16),
            device_pos=np.array([-1], dtype=np.int64),
        )
        measurements = TableBackedSequence(table)
        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator._measurement_plan_cache = {}
        self._prime_minimal_dc_node_plan(estimator)

        plan = estimator._measurement_plan(measurements)

        self.assertEqual(0, plan["node_rows"].size)
        np.testing.assert_array_equal(plan["handled_mask"], np.array([False], dtype=bool))

    def test_refresh_active_measurements_reuses_all_active_measurement_table(self):
        from model.meas_model import MeasurementTableView
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            prepare_active_measurements=False,
        )

        estimator._refresh_active_measurement_indexes()

        self.assertIsInstance(estimator.active_measurements, MeasurementTableView)
        self.assertIs(estimator.active_measurement_table, estimator.active_measurements.table)
        np.testing.assert_allclose(estimator.active_z, estimator.active_measurement_table.value)

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

    def test_apply_state_batches_device_value_calculation(self):
        import secore.dc_se as dc_se
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

        bus = estimator._dc_ppc["bus"]
        self.assertTrue(
            np.all(bus[estimator._raw_node_rows_alive.astype(np.intp), dc_se.DC_BUS_COLS["voltage"]] > 0.0)
        )

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
        self.assertLessEqual(call_count, result.iterations)

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

    def test_array_power_flow_seed_defers_object_seed_application(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator.__new__(DCStateEstimator)
        estimator.network = type("Network", (), {"ppc": {"format": "dc_ppc_v1"}})()
        estimator._power_flow_seed_rows = {
            "measurement_key": np.asarray([1], dtype=np.int64),
            "ppc_row": np.asarray([0], dtype=np.int64),
            "value": np.asarray([1.05], dtype=np.float64),
        }
        estimator._apply_power_flow_seed_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("array-mode seed should be applied to ppc, not object network")
        )

        estimator._apply_measurement_seed_to_network()

        self.assertIs(estimator._power_flow_seed_rows, estimator.network._se_power_flow_seed_rows)

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

    def test_run_summary_result_mode_limits_seresult_only(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            auto_prepare=False,
        )
        self.assertFalse(estimator._prepared)
        estimator.prepare()
        se_result = estimator.run(result_mode="summary", verbose=False, skip_bad_data=True)
        result = estimator.estimate_result

        self.assertIs(se_result, estimator.se_result)
        self.assertTrue(result.converged)
        self.assertIs(estimator.observability_result, result.observability)
        self.assertFalse(hasattr(result, "result_mode"))
        self.assertGreater(result.x.size, 0)
        self.assertGreater(result.z_est.size, 0)
        self.assertGreater(result.residual.size, 0)
        self.assertEqual(result.iterations, se_result.statistics.iterations)
        self.assertEqual(0, len(se_result.prefiltered_measurements))
        self.assertEqual(0, len(se_result.pseudo_measurements))
        self.assertEqual(0, len(se_result.bad_data))
        self.assertEqual(0, len(se_result.normal_measurements))

    def test_run_array_result_mode_keeps_estimate_arrays_only(self):
        import secore.dc_se as dc_se_module
        from secore.dc_se import DCStateEstimator
        from model.meas_model import MeasurementTableView
        from secore.se_result import SEResult

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            auto_prepare=False,
        )
        estimator.prepare()
        original_build = DCStateEstimator.build_se_result
        original_summary = dc_se_module.build_seresult_summary_from_table
        original_full = dc_se_module.build_seresult_full_from_table
        original_identify = DCStateEstimator.identify_bad_data
        original_from_estimate = SEResult.from_estimate_result
        original_apply_state = DCStateEstimator.apply_state

        def reject_seresult_path(*_args, **_kwargs):
            raise AssertionError("array result_mode should not build SEResult payloads")

        bad_data_calls = 0

        def counted_bad_data(self, result, threshold=None):
            nonlocal bad_data_calls
            bad_data_calls += 1
            return original_identify(self, result, threshold)

        def reject_apply_state(*_args, **_kwargs):
            raise AssertionError("array result_mode should not write estimated state back to model objects")

        def reject_full_tables(*_args, **_kwargs):
            raise AssertionError("array result_mode should not build full SEResult measurement tables")

        DCStateEstimator.build_se_result = reject_seresult_path
        dc_se_module.build_seresult_summary_from_table = reject_seresult_path
        dc_se_module.build_seresult_full_from_table = reject_seresult_path
        DCStateEstimator.identify_bad_data = counted_bad_data
        DCStateEstimator.apply_state = reject_apply_state
        SEResult.from_estimate_result = reject_full_tables
        try:
            se_result = estimator.run(result_mode="array", verbose=False)
        finally:
            DCStateEstimator.build_se_result = original_build
            dc_se_module.build_seresult_summary_from_table = original_summary
            dc_se_module.build_seresult_full_from_table = original_full
            DCStateEstimator.identify_bad_data = original_identify
            DCStateEstimator.apply_state = original_apply_state
            SEResult.from_estimate_result = original_from_estimate
        result = estimator.estimate_result

        self.assertIsNone(se_result)
        self.assertIsNone(estimator.se_result)
        self.assertTrue(result.converged)
        self.assertGreater(result.x.size, 0)
        self.assertGreater(result.z_est.size, 0)
        self.assertGreater(result.residual.size, 0)
        self.assertIsInstance(estimator.active_measurements, MeasurementTableView)
        self.assertIsNotNone(result.H)
        self.assertIsNotNone(result.gain)
        self.assertEqual(0, len(result.measurements))
        self.assertEqual(1, bad_data_calls)
        self.assertEqual([], estimator.bad_items)
        self.assertEqual(result.residual.size, estimator.normalized_residual.size)

    def test_run_array_result_mode_uses_stringless_measurement_ppc(self):
        from secore.dc_se import DCStateEstimator
        from model.meas_model import MeasurementTableView

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            auto_prepare=False,
        )

        estimator.run(result_mode="array", verbose=False, skip_bad_data=True)

        self.assertIsInstance(estimator.active_measurements, MeasurementTableView)
        self.assertEqual(0, estimator.measurement_table.device_name.size)
        self.assertEqual(0, estimator.measurement_table.device_type.size)
        self.assertEqual(0, estimator.measurement_table.meas_type.size)
        self.assertTrue(estimator.estimate_result.converged)

    def test_run_array_result_mode_uses_integer_active_key_caches(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            auto_prepare=False,
        )

        estimator.run(result_mode="array", verbose=False, skip_bad_data=True)

        self.assertTrue(estimator._active_measurement_key_cache)
        self.assertTrue(all(isinstance(key, int) for key in estimator._active_measurement_key_cache))

    def test_flat_start_array_result_mode_skips_power_flow_seed_rows(self):
        from secore.dc_se import DCStateEstimator

        estimator = DCStateEstimator(
            e_file=ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e",
            meas_file=ROOT_DIR / "data" / "meas" / "dc" / "dc_net_30.meas",
            flat_start=True,
            auto_prepare=False,
        )

        estimator.run(result_mode="array", verbose=False, skip_bad_data=True)

        self.assertEqual(0, int(estimator._power_flow_seed_rows["measurement_key"].size))
        self.assertEqual(0, int(estimator._power_flow_seed_rows["ppc_row"].size))
        self.assertEqual(0, int(estimator._power_flow_seed_rows["value"].size))
        self.assertTrue(estimator.estimate_result.converged)

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
        H = estimator.jacobian_sparse(x).toarray()
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
