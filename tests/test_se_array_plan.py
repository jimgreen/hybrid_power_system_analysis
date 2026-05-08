import unittest

import numpy as np

from model.meas_model import DEVICE_TYPE_CODES, Measurement, MeasurementList, measurement_table_from_measurements


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


if __name__ == "__main__":
    unittest.main()
