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

    def test_build_meas_ppc_populates_ieee39_core_table_columns(self):
        from model.meas_array_model import MEAS_COLS, build_meas_ppc_from_e_file, measurement_table_from_meas_ppc

        meas_file = ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"
        ppc = build_meas_ppc_from_e_file(meas_file)
        table = measurement_table_from_meas_ppc(ppc)

        self.assertEqual(table.idx.size, ppc["meas"].shape[0])
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["idx"]].astype(np.int64), table.idx)
        np.testing.assert_array_equal(ppc["name"], table.name)
        np.testing.assert_array_equal(ppc["device_name"], table.device_name)
        np.testing.assert_array_equal(ppc["device_type"], table.device_type)
        np.testing.assert_array_equal(ppc["meas_type"], table.meas_type)
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["weight"]], table.weight)
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["valid"]].astype(bool), table.valid)
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["value"]], table.value)
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["device_type_code"]].astype(np.int16),
            table.device_type_code,
        )
        np.testing.assert_array_equal(ppc["meas"][:, MEAS_COLS["angle_mask"]].astype(bool), table.angle_mask)

    def test_measurement_table_from_meas_ppc_can_skip_string_columns(self):
        from model.meas_array_model import build_meas_ppc_from_e_file, measurement_table_from_meas_ppc

        meas_file = ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"
        ppc = build_meas_ppc_from_e_file(meas_file)
        table = measurement_table_from_meas_ppc(ppc, include_strings=False)

        self.assertEqual(ppc["meas"].shape[0], table.idx.size)
        self.assertEqual(0, table.name.size)
        self.assertEqual(0, table.device_type.size)
        self.assertEqual(0, table.device_name.size)
        self.assertEqual(0, table.meas_type.size)
        self.assertEqual(table.idx.size, table.device_type_code.size)
        self.assertEqual(table.idx.size, table.meas_type_code.size)
        self.assertEqual(table.idx.size, table.device_name_id.size)

    def test_build_meas_ppc_from_e_file_can_skip_row_string_columns(self):
        from model.meas_model import MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL
        from model.meas_array_model import MEAS_COLS, MEAS_TYPE_CODES, build_meas_ppc_from_e_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value status",
                        "# 1 p1 ACLoad load_1 P_LOAD 1.0 1 2.0",
                        "# 2 q1 ACLoad load_1 Q_LOAD 1.0 1 1.0 0 extra_col",
                        "# 3 v1 ACNode node_1 V 2.5 0 110.0",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            ppc = build_meas_ppc_from_e_file(meas_file, include_strings=False)

        self.assertEqual((3, len(MEAS_COLS)), ppc["meas"].shape)
        self.assertEqual(0, ppc["name"].size)
        self.assertEqual(0, ppc["device_type"].size)
        self.assertEqual(0, ppc["device_name"].size)
        self.assertEqual(0, ppc["meas_type"].size)
        self.assertEqual(["load_1", "node_1"], ppc["device_names"].tolist())
        np.testing.assert_array_equal(ppc["device_name_id_array"], np.array([0, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(
            ppc["meas_type_code_array"],
            np.array([MEAS_TYPE_CODES["P_LOAD"], MEAS_TYPE_CODES["Q_LOAD"], MEAS_TYPE_CODES["V"]], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["status"]],
            np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID], dtype=np.float64),
        )

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

    def test_direct_measurement_parser_accepts_variable_width_rows(self):
        from model.meas_model import MEAS_STATUS_INVALID, MEAS_STATUS_NORMAL
        from model.meas_array_model import MEAS_COLS, MEAS_TYPE_CODES, build_meas_ppc_from_e_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            meas_file = Path(tmp_dir) / "case.meas"
            meas_file.write_text(
                "\n".join(
                    (
                        "<Measurement>",
                        "@ idx name dev_type dev_name meas_type weight valid value status",
                        "# 1 p1 ACLoad load_1 P_LOAD 1.0 1 2.0",
                        "# 2 q1 ACLoad load_1 Q_LOAD 1.0 1 1.0 0 extra_col",
                        "# 3 i1 ACLoad load_1 I_LOAD 1.0 1 0.1 1",
                        "</Measurement>",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            ppc = build_meas_ppc_from_e_file(meas_file)

        self.assertEqual((3, len(MEAS_COLS)), ppc["meas"].shape)
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["meas_type_code"]],
            np.array(
                [MEAS_TYPE_CODES["P_LOAD"], MEAS_TYPE_CODES["Q_LOAD"], MEAS_TYPE_CODES["I_LOAD"]],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["status"]],
            np.array([MEAS_STATUS_NORMAL, MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID], dtype=np.float64),
        )

    def test_build_meas_ppc_from_efile_rows_accepts_variable_width_rows(self):
        from model.meas_array_model import MEAS_COLS, MEAS_TYPE_CODES, build_meas_ppc_from_efile_rows

        rows = {
            "Measurement": {
                "header_list": ["idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value"],
                "rows": [
                    ["1", "p1", "ACLoad", "load_1", "P_LOAD", "1.0", "1", "2.0"],
                    ["2", "q1", "ACLoad", "load_1", "Q_LOAD", "1.0", "1", "1.0", "ignored_extra"],
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ppc = build_meas_ppc_from_efile_rows(Path(tmp_dir) / "case.meas", rows)

        self.assertEqual((2, len(MEAS_COLS)), ppc["meas"].shape)
        np.testing.assert_array_equal(
            ppc["meas"][:, MEAS_COLS["meas_type_code"]],
            np.array([MEAS_TYPE_CODES["P_LOAD"], MEAS_TYPE_CODES["Q_LOAD"]], dtype=np.float64),
        )
        np.testing.assert_allclose(ppc["meas"][:, MEAS_COLS["value"]], np.array([2.0, 1.0]))

    def test_measurement_table_from_meas_ppc_carries_runtime_index_arrays(self):
        from model.meas_array_model import copy_meas_ppc, build_meas_ppc_from_e_file, measurement_table_from_meas_ppc

        meas_file = ROOT_DIR / "data" / "meas" / "ac" / "ieee39.meas"
        ppc = build_meas_ppc_from_e_file(meas_file)
        row_count = int(ppc["meas"].shape[0])
        ppc["device_pos"] = np.arange(row_count, dtype=np.int64)
        ppc["scale"] = np.linspace(1.0, 2.0, row_count, dtype=np.float64)
        ppc["from_pos"] = np.arange(row_count, dtype=np.int64) + 10
        ppc["to_pos"] = np.arange(row_count, dtype=np.int64) + 20

        copied = copy_meas_ppc(ppc)
        table = measurement_table_from_meas_ppc(copied)

        np.testing.assert_array_equal(table.device_pos, ppc["device_pos"])
        np.testing.assert_array_equal(copied["device_pos"], ppc["device_pos"])
        np.testing.assert_array_equal(copied["scale"], ppc["scale"])
        np.testing.assert_array_equal(copied["from_pos"], ppc["from_pos"])
        np.testing.assert_array_equal(copied["to_pos"], ppc["to_pos"])
        self.assertIsNot(copied["device_pos"], ppc["device_pos"])
        self.assertIsNot(copied["scale"], ppc["scale"])


if __name__ == "__main__":
    unittest.main()
