import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class MeasurementArrayModelTest(unittest.TestCase):
    def test_build_meas_ppc_from_e_file_parses_standard_rows_without_measurement_objects(self):
        from model.meas_model import DEVICE_TYPE_CODES, MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL, Measurement
        from model.meas_array_model import MEAS_COLS, MEAS_TYPE_CODES, build_meas_ppc_from_e_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value",
                        "# 1 v1 ACNode n1 V 2.0 1 110.0",
                        "# 2 p1 ACLoad load_1 P_LOAD 3.5 0 -20.25",
                        "# 10003 long_measurement_name ACGenerator gen_with_long_name Q_GEN 12.25 1 -0.000123456",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(Measurement, "__init__", side_effect=AssertionError("meas PPC must not instantiate rows")):
                ppc = build_meas_ppc_from_e_file(meas_file)

        self.assertEqual("meas_ppc_v1", ppc["format"])
        self.assertEqual(str(meas_file.resolve()), ppc["source"])
        self.assertEqual((3, len(MEAS_COLS)), ppc["meas"].shape)
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["idx"]], np.array([1.0, 2.0, 10003.0]))
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["device_type_code"]],
            np.array(
                [
                    DEVICE_TYPE_CODES["ACNode"],
                    DEVICE_TYPE_CODES["ACLoad"],
                    DEVICE_TYPE_CODES["ACGenerator"],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["meas_type_code"]],
            np.array([MEAS_TYPE_CODES["V"], MEAS_TYPE_CODES["P_LOAD"], MEAS_TYPE_CODES["Q_GEN"]], dtype=np.float64),
        )
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["valid"]], np.array([1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["status"]],
            np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL], dtype=np.float64),
        )
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["value"]], np.array([110.0, -20.25, -0.000123456]))
        self.assertEqual(["v1", "p1", "long_measurement_name"], ppc["name"].tolist())
        self.assertEqual(["n1", "load_1", "gen_with_long_name"], ppc["device_name"].tolist())
        np.testing.assert_array_equal(
            ppc["rows_by_device_type_code"][DEVICE_TYPE_CODES["ACNode"]],
            np.array([0], dtype=np.int64),
        )

    def test_build_meas_ppc_matches_existing_parser_for_ieee39_core_columns(self):
        from model.meas_array_model import MEAS_COLS, build_meas_ppc_from_e_file
        from model.meas_model import Measurement
        from secore.ac_se import _read_measurements_direct

        meas_file = ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"
        ppc = build_meas_ppc_from_e_file(meas_file)
        old_measurements = _read_measurements_direct(meas_file, Measurement, table_only=True)
        old_table = old_measurements.table

        self.assertEqual(old_table.idx.size, ppc["meas"].shape[0])
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["idx"]].astype(np.int64), old_table.idx)
        np.testing.assert_array_equal(ppc["name"], old_table.name)
        np.testing.assert_array_equal(ppc["device_name"], old_table.device_name)
        np.testing.assert_array_equal(ppc["device_type"], old_table.device_type)
        np.testing.assert_array_equal(ppc["meas_type"], old_table.meas_type)
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["weight"]], old_table.weight)
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["valid"]].astype(bool), old_table.valid)
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["value"]], old_table.value)
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["device_type_code"]].astype(np.int16),
            old_table.device_type_code,
        )
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["angle_mask"]].astype(bool), old_table.angle_mask)

    def test_standard_parser_reuses_repeated_device_name_strings(self):
        from model.meas_array_model import build_meas_ppc_from_e_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@idx name dev_type dev_name meas_type weight valid value",
                        "#1 p1 ACLoad same_load P_LOAD 1.0 1 2.0",
                        "#2 q1 ACLoad same_load Q_LOAD 1.0 1 1.0",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            ppc = build_meas_ppc_from_e_file(meas_file)

        self.assertIs(ppc["device_name"][0], ppc["device_name"][1])


if __name__ == "__main__":
    unittest.main()
