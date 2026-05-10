import unittest

import numpy as np

from model.meas_model import (
    DEVICE_TYPE_CODES,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    Measurement,
    MeasurementList,
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
        self.assertIsNotNone(partitions.measurements["ac"].table)
        np.testing.assert_array_equal(partitions.rows["ac"], np.array([0, 3], dtype=np.int64))
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
