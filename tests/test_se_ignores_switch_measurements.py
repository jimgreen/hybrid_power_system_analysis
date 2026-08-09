import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "hybrid_power_system_analysis"))

from model.meas_model import MEAS_STATUS_INVALID
from model.meas_type import (
    DEVICE_TYPE_ACSwitch,
    DEVICE_TYPE_ACSwitchConstraint,
    DEVICE_TYPE_DCSwitch,
    DEVICE_TYPE_DCSwitchConstraint,
)
from scripts.update_meas_from_lf import Snapshot, parse_measurement_rows, rewrite_measurements
from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator


AC_CASE = ROOT / "data" / "model" / "ac" / "ac_net_30.e"
AC_MEASUREMENTS = ROOT / "data" / "meas" / "ac" / "ac_net_30.meas"
DC_CASE = ROOT / "data" / "model" / "dc" / "dc_net_30.e"
DC_MEASUREMENTS = ROOT / "data" / "meas" / "dc" / "dc_net_30.meas"
HYBRID_CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_converter_mix.e"
HYBRID_MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_converter_mix.meas"


def _measurement_file_with_rows(source: Path, directory: Path, rows) -> Path:
    target = directory / source.name
    text = source.read_text(encoding="utf-8")
    inserted = "\n".join(f"# {row}" for row in rows)
    target.write_text(
        text.replace("</Measurement>", f"{inserted}\n</Measurement>"),
        encoding="utf-8",
    )
    return target


def _assert_switch_rows_invalid(testcase, estimator, switch_codes) -> None:
    table = estimator.measurements.table
    mask = np.isin(
        np.asarray(table.device_type_code, dtype=np.int16),
        np.asarray(tuple(switch_codes), dtype=np.int16),
    )
    testcase.assertTrue(np.any(mask))
    testcase.assertFalse(np.any(np.asarray(table.valid, dtype=bool)[mask]))
    testcase.assertTrue(
        np.all(np.asarray(table.status_code, dtype=np.int16)[mask] == MEAS_STATUS_INVALID)
    )

    active_table = estimator.active_measurements.table
    testcase.assertFalse(
        np.any(
            np.isin(
                np.asarray(active_table.device_type_code, dtype=np.int16),
                np.asarray(tuple(switch_codes), dtype=np.int16),
            )
        )
    )


class SwitchMeasurementExclusionTest(unittest.TestCase):
    def test_ac_se_ignores_all_switch_measurement_kinds(self):
        rows = (
            "900001 ac_sw_p ACSwitch sw_2_3 P_FROM 100 1 1",
            "900002 ac_sw_q ACSwitch sw_2_3 Q_TO 100 1 1",
            "900003 ac_sw_v ACSwitch sw_2_3 V_FROM 100 1 1",
            "900004 ac_sw_i ACSwitch sw_2_3 I_TO 100 1 1",
            "900005 ac_sw_a ACSwitch sw_2_3 ANGLE_DIFF 100 1 1",
            "900006 ac_sw_c ACSwitchConstraint sw_2_3 V_DIFF 100 1 1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            meas_file = _measurement_file_with_rows(AC_MEASUREMENTS, Path(tmp), rows)
            estimator = ACStateEstimator(AC_CASE, meas_file, flat_start=True)

        _assert_switch_rows_invalid(
            self,
            estimator,
            (DEVICE_TYPE_ACSwitch, DEVICE_TYPE_ACSwitchConstraint),
        )
        self.assertNotIn("ACSwitch", np.asarray(estimator._state_meta_arrays_ref()["device_type"], dtype=object))

    def test_dc_se_ignores_switch_rows_and_does_not_publish_switch_measurement_devices(self):
        rows = (
            "900101 dc_sw_p DCSwitch sw_2_3 P_FROM 100 1 1",
            "900102 dc_sw_v DCSwitch sw_2_3 V_TO 100 1 1",
            "900103 dc_sw_i DCSwitch sw_2_3 I_FROM 100 1 1",
            "900104 dc_sw_c DCSwitchConstraint sw_2_3 V_DIFF 100 1 1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            meas_file = _measurement_file_with_rows(DC_MEASUREMENTS, Path(tmp), rows)
            estimator = DCStateEstimator(DC_CASE, meas_file, flat_start=True)

        _assert_switch_rows_invalid(
            self,
            estimator,
            (DEVICE_TYPE_DCSwitch, DEVICE_TYPE_DCSwitchConstraint),
        )
        counts = estimator.measurement_device_counts()
        names = estimator.measurement_device_names()
        self.assertNotIn(DEVICE_TYPE_DCSwitch, counts)
        self.assertNotIn(DEVICE_TYPE_DCSwitchConstraint, counts)
        self.assertNotIn(DEVICE_TYPE_DCSwitch, names)
        self.assertNotIn(DEVICE_TYPE_DCSwitchConstraint, names)
        self.assertNotIn("DCSwitch", np.asarray(estimator._state_meta_arrays_ref()["device_type"], dtype=object))

    def test_hybrid_se_drops_switch_rows_before_side_partitioning(self):
        rows = (
            "900201 dc_sw_p DCSwitch sw_2_3 P_FROM 100 1 1",
            "900202 dc_sw_c DCSwitchConstraint sw_2_3 V_DIFF 100 1 1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            meas_file = _measurement_file_with_rows(HYBRID_MEASUREMENTS, Path(tmp), rows)
            estimator = HybridStateEstimator(HYBRID_CASE, meas_file, flat_start=True)

        _assert_switch_rows_invalid(
            self,
            estimator,
            (DEVICE_TYPE_DCSwitch, DEVICE_TYPE_DCSwitchConstraint),
        )
        self.assertNotIn("ACSwitch", estimator._AC_MEASUREMENT_DEVICE_TYPES)
        self.assertNotIn("DCSwitch", estimator._DC_MEASUREMENT_DEVICE_TYPES)
        self.assertNotIn("DCSwitchConstraint", estimator._DC_MEASUREMENT_DEVICE_TYPES)
        self.assertNotIn(DEVICE_TYPE_ACSwitch, estimator._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE)
        self.assertNotIn(DEVICE_TYPE_DCSwitch, estimator._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE)
        self.assertNotIn(DEVICE_TYPE_DCSwitchConstraint, estimator._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE)

    def test_lf_snapshot_does_not_expose_switch_measurement_values(self):
        root = SimpleNamespace(p_base=100.0, p_base_kW=100000.0, u_scale=1.0, i_scale=1.0)
        switch = SimpleNamespace(name="sw", p=1.0, q=2.0, current=3.0)
        empty = {
            "nodes": [],
            "branches": [],
            "transformers": [],
            "breakers": [],
            "zero_branches": [],
            "generators": [],
            "loads": [],
        }
        ac = SimpleNamespace(**empty, switches=[switch])
        dc = SimpleNamespace(
            nodes=[],
            branches=[],
            switches=[switch],
            breakers=[],
            zero_branches=[],
            dcdc_converters=[],
            generators=[],
            loads=[],
        )
        snapshot = Snapshot(root, ac_grid=ac, dc_grid=dc)

        for device_type in ("ACSwitch", "DCSwitch"):
            for meas_type in ("P_FROM", "Q_FROM", "V_FROM", "I_FROM", "ANGLE_DIFF"):
                self.assertIsNone(snapshot.value(device_type, "sw", meas_type))

    def test_measurement_rewrite_marks_existing_switch_rows_invalid(self):
        root = SimpleNamespace(p_base=100.0, p_base_kW=100000.0, u_scale=1.0, i_scale=1.0)
        snapshot = Snapshot(root)
        with tempfile.TemporaryDirectory() as tmp:
            meas_file = Path(tmp) / "switch.meas"
            meas_file.write_text(
                "<Measurement>\n"
                "@ idx name dev_type dev_name meas_type weight valid value\n"
                "# 1 ac_sw ACSwitch sw P_FROM 100 1 123\n"
                "# 2 dc_sw DCSwitch sw I_TO 100 1 456\n"
                "</Measurement>\n",
                encoding="utf-8",
            )
            updated, missing = rewrite_measurements(meas_file, snapshot)
            _before, rows, _after = parse_measurement_rows(meas_file)

        self.assertEqual(0, updated)
        self.assertEqual(0, missing)
        self.assertEqual(["0", "0"], [row[6] for row in rows])


if __name__ == "__main__":
    unittest.main()
