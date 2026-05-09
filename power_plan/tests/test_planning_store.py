import shutil
import sys
import unittest
from pathlib import Path

from openpyxl import Workbook


WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))

import planning_store


class PlanningStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = WEB_ROOT / "tests" / "tmp_planning_store"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)
        self.store = planning_store.PlanningStore(root=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_validate_scheme_name_accepts_chinese_letters_numbers(self):
        self.assertEqual(planning_store.validate_scheme_name("方案A-01"), "方案A-01")

    def test_validate_scheme_name_removes_whitespace_and_invisible_chars(self):
        self.assertEqual(planning_store.validate_scheme_name(" 方 案\r\nA\u200b "), "方案A")

    def test_validate_scheme_name_rejects_path_chars(self):
        for name in ("", "../bad", "a/b", "a\\b", ".", ".."):
            with self.assertRaises(ValueError):
                planning_store.validate_scheme_name(name)

    def test_create_scheme_writes_default_workbook(self):
        payload = self.store.create_scheme("方案A")

        workbook = self.tmp_dir / "方案A" / "parameters.xlsx"
        self.assertTrue(workbook.exists())
        self.assertEqual(payload["scheme"], "方案A")
        self.assertEqual(len(payload["time_series"]), 8760)
        self.assertIn("diesel_generators", payload)
        self.assertIn("storage_battery_packs", payload)
        self.assertIn("hydrogen_tanks", payload)
        self.assertIn("planning_parameters", payload)
        self.assertEqual(payload["planning_parameters"][0]["design_life_years"], 20)
        self.assertEqual(payload["planning_parameters"][0]["planning_load_factor"], 1.0)
        self.assertFalse(payload["planning_parameters"][0]["storage_frequency_regulation_enabled"])
        self.assertEqual(payload["validation"][0]["level"], "ok")
        for key in planning_store.DEFAULT_DEVICE_ROWS:
            self.assertIn("quantity_lower", payload[key][0])
            self.assertIn("quantity_upper", payload[key][0])
            self.assertNotIn("design_capacity_lower", payload[key][0])
            self.assertNotIn("design_capacity_upper", payload[key][0])
            self.assertEqual(payload[key][0]["quantity_lower"], 0)
            self.assertEqual(payload[key][0]["quantity_upper"], 0)
        self.assertIn("generation_efficiency", payload["photovoltaics"][0])
        self.assertNotIn("cut_in_wind_speed", payload["photovoltaics"][0])
        self.assertNotIn("cut_out_wind_speed", payload["photovoltaics"][0])

    def test_list_copy_and_rename_schemes(self):
        self.store.create_scheme("方案A")
        self.store.copy_scheme("方案A", "方案B")
        self.store.rename_scheme("方案B", "方案C")

        names = [item["name"] for item in self.store.list_schemes()]
        self.assertEqual(names, ["方案A", "方案C"])
        self.assertTrue((self.tmp_dir / "方案C" / "parameters.xlsx").exists())
        self.assertFalse((self.tmp_dir / "方案B").exists())

    def test_delete_scheme_removes_scheme_folder(self):
        self.store.create_scheme("方案A")
        self.store.create_scheme("方案B")

        result = self.store.delete_scheme("方案A")

        self.assertEqual(result["deleted"], "方案A")
        self.assertFalse((self.tmp_dir / "方案A").exists())
        self.assertTrue((self.tmp_dir / "方案B" / "parameters.xlsx").exists())

    def test_delete_scheme_rejects_missing_scheme(self):
        with self.assertRaises(FileNotFoundError):
            self.store.delete_scheme("不存在")

    def test_write_and_read_scheme_round_trip(self):
        self.store.create_scheme("方案A")
        payload = self.store.read_scheme("方案A")
        payload["time_series"][0]["wind_speed"] = 8.5
        payload["time_series"][0]["temperature"] = -12.5
        payload["diesel_generators"][0]["quantity_lower"] = 1
        payload["diesel_generators"][0]["quantity_upper"] = 3
        payload["hydrogen_tanks"][0]["hydrogen_tank_capacity"] = 300
        payload["photovoltaics"][0]["generation_efficiency"] = 0.82
        payload["planning_parameters"][0]["design_life_years"] = 25
        payload["planning_parameters"][0]["diesel_price"] = 0.76
        payload["planning_parameters"][0]["storage_frequency_regulation_enabled"] = True

        self.store.write_scheme("方案A", payload)
        saved = self.store.read_scheme("方案A")

        self.assertEqual(saved["time_series"][0]["wind_speed"], 8.5)
        self.assertEqual(saved["time_series"][0]["temperature"], -12.5)
        self.assertEqual(saved["diesel_generators"][0]["quantity_lower"], 1)
        self.assertEqual(saved["diesel_generators"][0]["quantity_upper"], 3)
        self.assertEqual(saved["hydrogen_tanks"][0]["hydrogen_tank_capacity"], 300)
        self.assertEqual(saved["photovoltaics"][0]["generation_efficiency"], 0.82)
        self.assertEqual(saved["planning_parameters"][0]["design_life_years"], 25)
        self.assertEqual(saved["planning_parameters"][0]["diesel_price"], 0.76)
        self.assertTrue(saved["planning_parameters"][0]["storage_frequency_regulation_enabled"])

    def test_read_scheme_overview_defers_time_series_rows(self):
        self.store.create_scheme("方案A")

        overview = self.store.read_scheme_overview("方案A")

        self.assertEqual(overview["scheme"], "方案A")
        self.assertNotIn("time_series", overview)
        self.assertFalse(overview["time_series_loaded"])
        self.assertEqual(overview["time_series_count"], 8760)
        self.assertIn("diesel_generators", overview)
        self.assertIn("planning_parameters", overview)
        self.assertEqual(overview["planning_parameters"][0]["frequency_security_upper"], 1.5)
        self.assertFalse(any(item["level"] == "error" for item in overview["validation"]))

    def test_read_time_series_returns_only_time_series_rows(self):
        self.store.create_scheme("方案A")

        payload = self.store.read_time_series("方案A")

        self.assertEqual(payload["scheme"], "方案A")
        self.assertEqual(len(payload["time_series"]), 8760)
        self.assertNotIn("diesel_generators", payload)
        self.assertNotIn("planning_parameters", payload)

    def test_default_time_series_includes_temperature(self):
        payload = planning_store.default_payload("方案A")

        self.assertIn("temperature", payload["time_series"][0])
        self.assertEqual(payload["time_series"][0]["temperature"], 0)

    def test_read_legacy_time_series_without_temperature_keeps_load(self):
        self.store.create_scheme("方案A")
        workbook_path = self.tmp_dir / "方案A" / "parameters.xlsx"
        workbook = Workbook()
        workbook.remove(workbook.active)
        sheet = workbook.create_sheet("8760时序数据")
        sheet.append(["hour_index", "datetime", "wind_speed", "solar_irradiance", "load"])
        for hour in range(1, 8761):
            sheet.append([hour, f"H{hour:04d}", 8.5 if hour == 1 else 0, 310 if hour == 1 else 0, 123.4 if hour == 1 else 0])
        for key, (sheet_name, headers) in planning_store.SHEET_SPECS.items():
            if key == "time_series":
                continue
            device_sheet = workbook.create_sheet(sheet_name)
            device_sheet.append(headers)
        workbook.save(workbook_path)

        payload = self.store.read_scheme("方案A")

        self.assertEqual(payload["time_series"][0]["load"], 123.4)
        self.assertEqual(payload["time_series"][0]["temperature"], "")
        self.assertIn("planning_parameters", payload)
        self.assertEqual(payload["planning_parameters"][0]["design_life_years"], 20)

    def test_read_legacy_photovoltaic_sheet_does_not_shift_removed_capacity_columns(self):
        self.store.create_scheme("方案A")
        workbook_path = self.tmp_dir / "方案A" / "parameters.xlsx"
        workbook = Workbook()
        workbook.remove(workbook.active)
        time_sheet = workbook.create_sheet("8760时序数据")
        time_sheet.append(planning_store.SHEET_SPECS["time_series"][1])
        for row in planning_store.default_time_series():
            time_sheet.append([row.get(header, "") for header in planning_store.SHEET_SPECS["time_series"][1]])
        for key, (sheet_name, headers) in planning_store.SHEET_SPECS.items():
            if key in {"time_series", "photovoltaics"}:
                continue
            device_sheet = workbook.create_sheet(sheet_name)
            device_sheet.append(headers)
        pv_sheet = workbook.create_sheet("光伏参数")
        pv_sheet.append([
            "name",
            "capacity",
            "design_capacity_lower",
            "design_capacity_upper",
            "cost",
            "cut_in_wind_speed",
            "cut_out_wind_speed",
            "quantity_lower",
            "quantity_upper",
        ])
        pv_sheet.append(["旧光伏", 50, 10, 90, 3.5, 1, 2, 4, 5])
        workbook.save(workbook_path)

        payload = self.store.read_scheme("方案A")

        self.assertEqual(payload["photovoltaics"][0]["cost"], 3.5)
        self.assertEqual(payload["photovoltaics"][0]["generation_efficiency"], "")
        self.assertEqual(payload["photovoltaics"][0]["quantity_lower"], 4)
        self.assertEqual(payload["photovoltaics"][0]["quantity_upper"], 5)

    def test_validate_quantity_limits(self):
        payload = planning_store.default_payload("方案A")
        payload["fuel_cells"][0]["quantity_lower"] = 5
        payload["fuel_cells"][0]["quantity_upper"] = 2

        messages = planning_store.validate_payload(payload)

        self.assertTrue(any("数据上限不能小于数据下限" in item["message"] for item in messages))

    def test_validate_planning_parameter_ranges(self):
        payload = planning_store.default_payload("方案A")
        payload["planning_parameters"][0]["planning_load_factor"] = 12
        payload["planning_parameters"][0]["green_power_ratio_lower"] = 1.2
        payload["planning_parameters"][0]["frequency_security_upper"] = 1.1
        payload["planning_parameters"][0]["frequency_security_lower"] = 1.3

        messages = planning_store.validate_payload(payload)

        self.assertTrue(any("规划负荷系数(0.1-10.0)不能大于10" in item["message"] for item in messages))
        self.assertTrue(any("绿电电量占比下限(0.0-1.0)不能大于1" in item["message"] for item in messages))
        self.assertTrue(any("频率安全上限不能小于频率安全下限" in item["message"] for item in messages))


if __name__ == "__main__":
    unittest.main()
