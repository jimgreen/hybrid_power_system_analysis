import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class EFileReadPerformanceTest(unittest.TestCase):
    def test_rows_reader_unquoted_rows_use_inline_fast_split(self):
        import efile_read

        content = "\n".join(
            [
                "<ACNode>",
                "@ idx name vbase voltage",
                "# 1 n1 110 110",
                "# 2 n2 220 220",
                "</ACNode>",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fast_rows.e"
            path.write_text(content, encoding="utf-8")
            original_split_row = efile_read._split_data_row

            def forbidden_split_row(*args, **kwargs):
                raise AssertionError("_read_efile_rows should inline unquoted row splitting")

            efile_read._split_data_row = forbidden_split_row
            try:
                data = efile_read._read_efile_rows(path)
            finally:
                efile_read._split_data_row = original_split_row

        self.assertEqual(["1", "n1", "110", "110"], data["ACNode"]["rows"][0])
        self.assertEqual(["2", "n2", "220", "220"], data["ACNode"]["rows"][1])

    def test_unquoted_rows_use_fast_split_without_regex(self):
        import efile_read
        from efile_read import EBook

        content = "\n".join(
            [
                "<Measurement>",
                "@ idx name dev_type dev_name meas_type weight valid value",
                "# 1 vm_bus_1 ACNode bus_1 V 1.0 1 358.587342",
                "# 2 vm_bus_2 ACNode bus_2 V 2.0 1 361.7304645",
                "</Measurement>",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fast.meas"
            path.write_text(content, encoding="utf-8")
            original_split = efile_read.re.split

            def forbidden_regex_split(*args, **kwargs):
                raise AssertionError("unquoted E rows should use str.split fast path")

            efile_read.re.split = forbidden_regex_split
            try:
                data = EBook(path).to_dict()
            finally:
                efile_read.re.split = original_split

        rows = data["Measurement"]["data"]
        self.assertEqual("ACNode", rows[0]["dev_type"])
        self.assertEqual("bus_2", rows[1]["dev_name"])

    def test_quoted_rows_keep_regex_aware_split(self):
        from efile_read import EBook

        content = "\n".join(
            [
                "<Device>",
                "@ idx name description value",
                "# 1 dev_1 'quoted value with spaces' 2.5",
                "</Device>",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "quoted.e"
            path.write_text(content, encoding="utf-8")
            data = EBook(path).to_dict()

        row = data["Device"]["data"][0]
        self.assertEqual("'quoted value with spaces'", row["description"])
        self.assertEqual("2.5", row["value"])


if __name__ == "__main__":
    unittest.main()
