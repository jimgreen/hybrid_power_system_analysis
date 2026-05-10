import unittest

import numpy as np

from model.meas_model import (
    DEVICE_TYPE_CODES,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    Measurement,
    MeasurementList,
    MeasurementView,
    is_pseudo_measurement,
    measurement_table_from_measurements,
    measurement_table_status_code,
)


class SEArrayPlanTest(unittest.TestCase):
    @staticmethod
    def _measurements():
        rows = [
            Measurement(1, "ac_v", "ACNode", "ac_bus", "V", 1.0, True, 1.02),
            Measurement(2, "dc_v", "DCNode", "dc_bus", "V", 0.0, True, 0.98),
            Measurement(3, "conv_p", "DCACConverter", "conv", "P_AC", 2.0, True, 3.0),
            Measurement(4, "ac_p", "ACLoad", "load", "P_LOAD", 4.0, False, 1.2),
        ]
        return MeasurementList(rows, measurement_table_from_measurements(rows))

    def test_build_active_measurement_view_preserves_all_active_table(self):
        from secore.se_array_plan import build_active_measurement_view

        measurements = MeasurementList(
            [
                Measurement(1, "m1", "ACNode", "n1", "V", 1.0, True, 1.0),
                Measurement(2, "m2", "ACNode", "n2", "V", 2.0, True, 2.0),
            ]
        )
        measurements.table = measurement_table_from_measurements(measurements)

        view = build_active_measurement_view(measurements)

        self.assertTrue(view.all_active)
        self.assertIs(view.measurements, measurements)
        self.assertIs(view.table, measurements.table)
        np.testing.assert_array_equal(view.source_rows, np.array([0, 1], dtype=np.int64))
        np.testing.assert_allclose(view.z, np.array([1.0, 2.0]))
        np.testing.assert_allclose(view.weight, np.array([1.0, 2.0]))

    def test_build_active_measurement_view_returns_table_backed_subset(self):
        from secore.se_array_plan import build_active_measurement_view

        measurements = self._measurements()

        view = build_active_measurement_view(measurements)

        self.assertFalse(view.all_active)
        self.assertIsInstance(view.measurements, MeasurementList)
        self.assertIs(view.measurements.table, view.table)
        self.assertEqual(["ac_v", "conv_p"], [meas.name for meas in view.measurements])
        np.testing.assert_array_equal(view.source_rows, np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(view.table.idx, np.array([1, 3], dtype=np.int64))
        np.testing.assert_allclose(view.z, np.array([1.02, 3.0]))
        np.testing.assert_allclose(view.weight, np.array([1.0, 2.0]))

    def test_partition_measurements_by_code_preserves_side_tables(self):
        from secore.se_array_plan import partition_measurements_by_code

        measurements = self._measurements()
        side_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: "ac",
            DEVICE_TYPE_CODES["ACLoad"]: "ac",
            DEVICE_TYPE_CODES["DCNode"]: "dc",
            DEVICE_TYPE_CODES["DCACConverter"]: "hybrid",
        }

        partitions = partition_measurements_by_code(
            measurements,
            side_by_code,
            sides=("ac", "dc", "hybrid"),
        )

        self.assertEqual(["ac_v", "ac_p"], [meas.name for meas in partitions.measurements["ac"]])
        self.assertEqual(["dc_v"], [meas.name for meas in partitions.measurements["dc"]])
        self.assertEqual(["conv_p"], [meas.name for meas in partitions.measurements["hybrid"]])
        self.assertIsInstance(partitions.measurements["ac"], MeasurementList)
        self.assertNotIsInstance(partitions.measurements["ac"], MeasurementView)
        self.assertIsNotNone(partitions.measurements["ac"].table)
        np.testing.assert_array_equal(partitions.rows["ac"], np.array([0, 3], dtype=np.int64))
        np.testing.assert_array_equal(partitions.measurements["ac"].table.idx, np.array([1, 4], dtype=np.int64))

    def test_partition_measurements_by_code_can_return_side_views(self):
        from secore.se_array_plan import partition_measurements_by_code

        measurements = self._measurements()
        side_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: "ac",
            DEVICE_TYPE_CODES["ACLoad"]: "ac",
            DEVICE_TYPE_CODES["DCNode"]: "dc",
            DEVICE_TYPE_CODES["DCACConverter"]: "hybrid",
        }

        partitions = partition_measurements_by_code(
            measurements,
            side_by_code,
            sides=("ac", "dc", "hybrid"),
            as_view=True,
        )

        self.assertIsInstance(partitions.measurements["ac"], MeasurementView)
        self.assertIs(partitions.measurements["ac"][0], measurements[0])
        self.assertIs(partitions.measurements["ac"][1], measurements[3])
        np.testing.assert_array_equal(partitions.measurements["ac"].table.idx, np.array([1, 4], dtype=np.int64))

    def test_measurement_status_is_explicit_and_preserved_in_table_slices(self):
        from secore.se_array_plan import measurement_table_take

        rows = [
            Measurement(1, "pseudo_named_real", "ACNode", "n1", "V", 1.0, True, 1.0, MEAS_STATUS_NORMAL),
            Measurement(2, "normal_named_pseudo", "ACNode", "n2", "V", 1.0, True, 1.0, MEAS_STATUS_PSEUDO),
            Measurement(3, "invalid", "ACNode", "n3", "V", 1.0, True, 1.0, MEAS_STATUS_INVALID),
        ]
        table = measurement_table_from_measurements(rows)
        sliced = measurement_table_take(table, np.array([0, 2], dtype=np.int64))

        self.assertFalse(is_pseudo_measurement(rows[0]))
        self.assertTrue(is_pseudo_measurement(rows[1]))
        self.assertFalse(rows[2].valid)
        np.testing.assert_array_equal(
            measurement_table_status_code(table),
            np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_PSEUDO, MEAS_STATUS_INVALID], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            measurement_table_status_code(sliced),
            np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID], dtype=np.int16),
        )

    def test_measurement_table_take_preserves_device_type_row_cache(self):
        from model.meas_model import MeasurementTable
        from secore.se_array_plan import measurement_table_take, rows_by_device_type_code

        table = MeasurementTable(
            idx=np.array([1, 2, 3, 4], dtype=np.int64),
            name=np.array(["ac1", "dc1", "ac2", "hyb1"], dtype=object),
            device_type=np.array(["ACNode", "DCNode", "ACLoad", "DCACConverter"], dtype=object),
            device_name=np.array(["a1", "d1", "l1", "c1"], dtype=object),
            meas_type=np.array(["V", "V", "P_LOAD", "P_AC"], dtype=object),
            weight=np.ones(4, dtype=np.float64),
            valid=np.ones(4, dtype=bool),
            value=np.ones(4, dtype=np.float64),
            device_type_code=np.array(
                [
                    DEVICE_TYPE_CODES["ACNode"],
                    DEVICE_TYPE_CODES["DCNode"],
                    DEVICE_TYPE_CODES["ACLoad"],
                    DEVICE_TYPE_CODES["DCACConverter"],
                ],
                dtype=np.int16,
            ),
            angle_mask=np.zeros(4, dtype=bool),
            rows_by_device_type_code={
                DEVICE_TYPE_CODES["ACNode"]: np.array([0], dtype=np.int64),
                DEVICE_TYPE_CODES["DCNode"]: np.array([1], dtype=np.int64),
                DEVICE_TYPE_CODES["ACLoad"]: np.array([2], dtype=np.int64),
                DEVICE_TYPE_CODES["DCACConverter"]: np.array([3], dtype=np.int64),
            },
        )

        sliced = measurement_table_take(table, np.array([0, 2, 3], dtype=np.int64))

        cache = rows_by_device_type_code(sliced)
        np.testing.assert_array_equal(cache[DEVICE_TYPE_CODES["ACNode"]], np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(cache[DEVICE_TYPE_CODES["ACLoad"]], np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(cache[DEVICE_TYPE_CODES["DCACConverter"]], np.array([2], dtype=np.int64))
        self.assertNotIn(DEVICE_TYPE_CODES["DCNode"], cache)

    def test_partition_measurements_by_code_can_fallback_to_device_type(self):
        from secore.se_array_plan import partition_measurements_by_code

        measurements = MeasurementList(
            [Measurement(1, "dc_balance", "DCPowerBalance", "dc_bus", "P_BALANCE", 1.0, True, 0.0)]
        )
        measurements.table = measurement_table_from_measurements(measurements)
        self.assertEqual(0, int(measurements.table.device_type_code[0]))

        partitions = partition_measurements_by_code(
            measurements,
            {},
            side_by_device_type={"DCPowerBalance": "dc"},
            sides=("ac", "dc", "hybrid"),
        )

        self.assertEqual(["dc_balance"], [meas.name for meas in partitions.measurements["dc"]])
        np.testing.assert_array_equal(partitions.rows["dc"], np.array([0], dtype=np.int64))

    def test_partition_measurements_by_code_reuses_precomputed_code_rows(self):
        from secore.se_array_plan import build_active_measurement_view, partition_measurements_by_code

        measurements = self._measurements()
        view = build_active_measurement_view(measurements)
        side_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: "ac",
            DEVICE_TYPE_CODES["DCACConverter"]: "hybrid",
        }

        partitions = partition_measurements_by_code(
            view.measurements,
            side_by_code,
            rows_by_device_type_code=view.rows_by_device_type_code,
            sides=("ac", "dc", "hybrid"),
        )

        self.assertEqual(["ac_v"], [meas.name for meas in partitions.measurements["ac"]])
        self.assertEqual([], [meas.name for meas in partitions.measurements["dc"]])
        self.assertEqual(["conv_p"], [meas.name for meas in partitions.measurements["hybrid"]])
        np.testing.assert_array_equal(partitions.rows["ac"], np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(partitions.rows["hybrid"], np.array([1], dtype=np.int64))

    def test_partition_measurements_by_code_passes_side_row_cache_to_table_slice(self):
        import secore.se_array_plan as se_array_plan
        from secore.se_array_plan import partition_measurements_by_code, rows_by_device_type_code

        measurements = self._measurements()
        side_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: "ac",
            DEVICE_TYPE_CODES["ACLoad"]: "ac",
            DEVICE_TYPE_CODES["DCNode"]: "dc",
            DEVICE_TYPE_CODES["DCACConverter"]: "hybrid",
        }
        original = se_array_plan._slice_cached_rows_by_device_type_code

        def reject_generic_slice(*_args, **_kwargs):
            raise AssertionError("partition already knows side code rows and should pass them through")

        se_array_plan._slice_cached_rows_by_device_type_code = reject_generic_slice
        try:
            partitions = partition_measurements_by_code(
                measurements,
                side_by_code,
                sides=("ac", "dc", "hybrid"),
                as_view=True,
            )
        finally:
            se_array_plan._slice_cached_rows_by_device_type_code = original

        ac_cache = rows_by_device_type_code(partitions.measurements["ac"].table)
        np.testing.assert_array_equal(ac_cache[DEVICE_TYPE_CODES["ACNode"]], np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(ac_cache[DEVICE_TYPE_CODES["ACLoad"]], np.array([1], dtype=np.int64))

    def test_build_measurement_plan_table_uses_cached_table_without_iterating(self):
        from model.meas_model import MeasurementTable
        from secore.se_array_plan import build_measurement_plan_table

        class TableBackedSequence:
            def __init__(self, table):
                self.table = table

            def __len__(self):
                return len(self.table.idx)

            def __iter__(self):
                raise AssertionError("MeasurementPlanTable should use the cached table")

        table = MeasurementTable(
            idx=np.array([1, 2, 3, 4], dtype=np.int64),
            name=np.array(["node_v", "load_p", "missing_load", "bad_node"], dtype=object),
            device_type=np.array(["DCNode", "DCLoad", "DCLoad", "DCNode"], dtype=object),
            device_name=np.array(["n1", "l1", "missing", "n2"], dtype=object),
            meas_type=np.array(["V", "P_LOAD", "P_LOAD", "BAD"], dtype=object),
            weight=np.ones(4, dtype=np.float64),
            valid=np.ones(4, dtype=bool),
            value=np.ones(4, dtype=np.float64),
            device_type_code=np.array(
                [DEVICE_TYPE_CODES["DCNode"], DEVICE_TYPE_CODES["DCLoad"], DEVICE_TYPE_CODES["DCLoad"], DEVICE_TYPE_CODES["DCNode"]],
                dtype=np.int16,
            ),
            angle_mask=np.zeros(4, dtype=bool),
        )
        measurements = TableBackedSequence(table)

        plan = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={
                DEVICE_TYPE_CODES["DCNode"]: {"n1": 3},
                DEVICE_TYPE_CODES["DCLoad"]: {"l1": 5},
            },
            meas_kind_by_type_code={
                DEVICE_TYPE_CODES["DCNode"]: {"V": 0},
                DEVICE_TYPE_CODES["DCLoad"]: {"P_LOAD": 0, "V_LOAD": 1, "I_LOAD": 2},
            },
        )

        self.assertIs(plan.table, table)
        np.testing.assert_array_equal(plan.row, np.array([0, 1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(
            plan.device_type_code,
            np.array(
                [DEVICE_TYPE_CODES["DCNode"], DEVICE_TYPE_CODES["DCLoad"], DEVICE_TYPE_CODES["DCLoad"], DEVICE_TYPE_CODES["DCNode"]],
                dtype=np.int16,
            ),
        )
        np.testing.assert_array_equal(plan.meas_kind, np.array([0, 0, 0, -1], dtype=np.int16))
        np.testing.assert_array_equal(plan.device_pos, np.array([3, 5, -1, -1], dtype=np.int64))
        np.testing.assert_array_equal(plan.handled, np.array([True, True, False, False], dtype=bool))

    def test_build_measurement_plan_table_reuses_cached_device_positions(self):
        from model.meas_model import MeasurementTable
        from secore.se_array_plan import build_measurement_plan_table

        class TableBackedSequence:
            def __init__(self, table):
                self.table = table

            def __len__(self):
                return len(self.table.idx)

        class FailAfterFirstMap(dict):
            def __init__(self, values):
                super().__init__(values)
                self.fail = False

            def get(self, key, default=None):
                if self.fail:
                    raise AssertionError("device positions should be cached for the same table and map")
                return super().get(key, default)

        table = MeasurementTable(
            idx=np.array([1, 2, 3], dtype=np.int64),
            name=np.array(["v1", "p1", "v2"], dtype=object),
            device_type=np.array(["ACNode", "ACLoad", "ACNode"], dtype=object),
            device_name=np.array(["n1", "l1", "n2"], dtype=object),
            meas_type=np.array(["V", "P_LOAD", "BAD"], dtype=object),
            weight=np.ones(3, dtype=np.float64),
            valid=np.ones(3, dtype=bool),
            value=np.ones(3, dtype=np.float64),
            device_type_code=np.array(
                [DEVICE_TYPE_CODES["ACNode"], DEVICE_TYPE_CODES["ACLoad"], DEVICE_TYPE_CODES["ACNode"]],
                dtype=np.int16,
            ),
            angle_mask=np.zeros(3, dtype=bool),
            rows_by_device_type_code={
                DEVICE_TYPE_CODES["ACNode"]: np.array([0, 2], dtype=np.int64),
                DEVICE_TYPE_CODES["ACLoad"]: np.array([1], dtype=np.int64),
            },
        )
        node_pos = FailAfterFirstMap({"n1": 10, "n2": 11})
        load_pos = FailAfterFirstMap({"l1": 20})
        measurements = TableBackedSequence(table)
        device_pos_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: node_pos,
            DEVICE_TYPE_CODES["ACLoad"]: load_pos,
        }

        first = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=device_pos_by_code,
            meas_kind_by_type_code={
                DEVICE_TYPE_CODES["ACNode"]: {"V": 0},
                DEVICE_TYPE_CODES["ACLoad"]: {"P_LOAD": 1},
            },
        )
        node_pos.fail = True
        load_pos.fail = True
        second = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code=device_pos_by_code,
            meas_kind_by_type_code={
                DEVICE_TYPE_CODES["ACNode"]: {"BAD": 2},
                DEVICE_TYPE_CODES["ACLoad"]: {"P_LOAD": 1},
            },
        )

        np.testing.assert_array_equal(first.device_pos, np.array([10, 20, 11], dtype=np.int64))
        np.testing.assert_array_equal(second.device_pos, first.device_pos)
        np.testing.assert_array_equal(second.meas_kind, np.array([-1, 1, 2], dtype=np.int16))

    def test_build_measurement_plan_table_reuses_cached_measurement_kinds(self):
        from model.meas_model import MeasurementTable
        from secore.se_array_plan import build_measurement_plan_table

        class TableBackedSequence:
            def __init__(self, table):
                self.table = table

            def __len__(self):
                return len(self.table.idx)

        class FailAfterFirstMap(dict):
            def __init__(self, values):
                super().__init__(values)
                self.fail = False

            def get(self, key, default=None):
                if self.fail:
                    raise AssertionError("measurement kinds should be cached for the same table and map")
                return super().get(key, default)

        table = MeasurementTable(
            idx=np.array([1, 2, 3], dtype=np.int64),
            name=np.array(["v1", "p1", "bad"], dtype=object),
            device_type=np.array(["ACNode", "ACLoad", "ACNode"], dtype=object),
            device_name=np.array(["n1", "l1", "n2"], dtype=object),
            meas_type=np.array(["V", "P_LOAD", "BAD"], dtype=object),
            weight=np.ones(3, dtype=np.float64),
            valid=np.ones(3, dtype=bool),
            value=np.ones(3, dtype=np.float64),
            device_type_code=np.array(
                [DEVICE_TYPE_CODES["ACNode"], DEVICE_TYPE_CODES["ACLoad"], DEVICE_TYPE_CODES["ACNode"]],
                dtype=np.int16,
            ),
            angle_mask=np.zeros(3, dtype=bool),
            rows_by_device_type_code={
                DEVICE_TYPE_CODES["ACNode"]: np.array([0, 2], dtype=np.int64),
                DEVICE_TYPE_CODES["ACLoad"]: np.array([1], dtype=np.int64),
            },
        )
        node_kind = FailAfterFirstMap({"V": 0})
        load_kind = FailAfterFirstMap({"P_LOAD": 1})
        measurements = TableBackedSequence(table)
        meas_kind_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: node_kind,
            DEVICE_TYPE_CODES["ACLoad"]: load_kind,
        }

        first = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={
                DEVICE_TYPE_CODES["ACNode"]: {"n1": 10, "n2": 11},
                DEVICE_TYPE_CODES["ACLoad"]: {"l1": 20},
            },
            meas_kind_by_type_code=meas_kind_by_code,
        )
        node_kind.fail = True
        load_kind.fail = True
        second = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={
                DEVICE_TYPE_CODES["ACNode"]: {"n1": 10, "n2": 11},
                DEVICE_TYPE_CODES["ACLoad"]: {"l1": 20},
            },
            meas_kind_by_type_code=meas_kind_by_code,
        )

        np.testing.assert_array_equal(first.meas_kind, np.array([0, 1, -1], dtype=np.int16))
        np.testing.assert_array_equal(second.meas_kind, first.meas_kind)

    def test_append_active_measurement_view_extends_active_table_and_rows(self):
        from secore.se_array_plan import append_active_measurement_view, build_active_measurement_view

        measurements = self._measurements()
        view = build_active_measurement_view(measurements)
        additions = [
            Measurement(5, "ac_v2", "ACNode", "ac_bus_2", "V", 5.0, True, 1.01),
            Measurement(6, "dc_skip", "DCNode", "dc_bus_2", "V", 0.0, True, 0.99),
        ]

        updated = append_active_measurement_view(view, additions)

        self.assertEqual(["ac_v", "conv_p", "ac_v2"], [meas.name for meas in updated.measurements])
        np.testing.assert_array_equal(updated.source_rows, np.array([0, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(updated.table.idx, np.array([1, 3, 5], dtype=np.int64))
        np.testing.assert_allclose(updated.z, np.array([1.02, 3.0, 1.01]))
        np.testing.assert_allclose(updated.weight, np.array([1.0, 2.0, 5.0]))

    def test_extend_measurement_partitions_appends_side_rows_and_tables(self):
        from secore.se_array_plan import build_active_measurement_view, extend_measurement_partitions, partition_measurements_by_code

        measurements = self._measurements()
        view = build_active_measurement_view(measurements)
        side_by_code = {
            DEVICE_TYPE_CODES["ACNode"]: "ac",
            DEVICE_TYPE_CODES["DCNode"]: "dc",
            DEVICE_TYPE_CODES["DCACConverter"]: "hybrid",
        }
        partitions = partition_measurements_by_code(
            view.measurements,
            side_by_code,
            sides=("ac", "dc", "hybrid"),
        )
        additions = [
            Measurement(5, "ac_v2", "ACNode", "ac_bus_2", "V", 5.0, True, 1.01),
            Measurement(6, "hyb_p2", "DCACConverter", "conv_2", "P_AC", 6.0, True, 4.0),
        ]

        updated = extend_measurement_partitions(
            partitions,
            additions,
            side_by_code,
            sides=("ac", "dc", "hybrid"),
        )

        self.assertEqual(["ac_v", "ac_v2"], [meas.name for meas in updated.measurements["ac"]])
        self.assertEqual(["conv_p", "hyb_p2"], [meas.name for meas in updated.measurements["hybrid"]])
        np.testing.assert_array_equal(updated.rows["ac"], np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(updated.rows["hybrid"], np.array([1, 3], dtype=np.int64))
        np.testing.assert_array_equal(updated.measurements["ac"].table.idx, np.array([1, 5], dtype=np.int64))
        np.testing.assert_array_equal(updated.measurements["hybrid"].table.idx, np.array([3, 6], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
