import shutil
import sys
import unittest
from pathlib import Path


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
        self.assertEqual(payload["validation"][0]["level"], "ok")

    def test_list_copy_and_rename_schemes(self):
        self.store.create_scheme("方案A")
        self.store.copy_scheme("方案A", "方案B")
        self.store.rename_scheme("方案B", "方案C")

        names = [item["name"] for item in self.store.list_schemes()]
        self.assertEqual(names, ["方案A", "方案C"])
        self.assertTrue((self.tmp_dir / "方案C" / "parameters.xlsx").exists())
        self.assertFalse((self.tmp_dir / "方案B").exists())

    def test_write_and_read_scheme_round_trip(self):
        self.store.create_scheme("方案A")
        payload = self.store.read_scheme("方案A")
        payload["time_series"][0]["wind_speed"] = 8.5
        payload["diesel_generators"][0]["design_capacity_upper"] = 650
        payload["hydrogen_tanks"][0]["hydrogen_tank_capacity"] = 300

        self.store.write_scheme("方案A", payload)
        saved = self.store.read_scheme("方案A")

        self.assertEqual(saved["time_series"][0]["wind_speed"], 8.5)
        self.assertEqual(saved["diesel_generators"][0]["design_capacity_upper"], 650)
        self.assertEqual(saved["hydrogen_tanks"][0]["hydrogen_tank_capacity"], 300)

    def test_validate_design_capacity_limits(self):
        payload = planning_store.default_payload("方案A")
        payload["wind_turbines"][0]["design_capacity_lower"] = 10
        payload["wind_turbines"][0]["design_capacity_upper"] = 1

        messages = planning_store.validate_payload(payload)

        self.assertTrue(any("设计容量上限不能小于下限" in item["message"] for item in messages))

    def test_validate_design_capacity_allows_upper_equal_to_lower(self):
        payload = planning_store.default_payload("方案A")
        payload["storage_pcs"][0]["design_capacity_lower"] = 10
        payload["storage_pcs"][0]["design_capacity_upper"] = 10

        messages = planning_store.validate_payload(payload)

        self.assertFalse(any(item["level"] == "error" for item in messages))


if __name__ == "__main__":
    unittest.main()
