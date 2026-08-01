import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path


class PrimaryFrequencyResponseTest(unittest.TestCase):
    def test_power_deficit_makes_frequency_drop_and_storage_responds_first(self):
        from hybrid_power_system_analysis.simu.primary_frequency_response import (
            DieselGovernor,
            Disturbance,
            GridFormingStorage,
            SystemFrequencyModel,
            simulate_primary_frequency_response,
        )

        result = simulate_primary_frequency_response(
            system=SystemFrequencyModel(f_nom_hz=50.0, s_base_mw=10.0, inertia_s=4.0, damping_mw_per_hz=0.2),
            diesel=DieselGovernor(
                reserve_mw=2.0,
                droop_mw_per_hz=0.8,
                time_constant_s=2.0,
                ramp_mw_per_s=0.5,
            ),
            storage=GridFormingStorage(
                discharge_limit_mw=2.0,
                charge_limit_mw=2.0,
                droop_mw_per_hz=1.5,
                inertia_mw_s_per_hz=0.4,
                response_time_s=0.1,
                energy_mwh=4.0,
                initial_soc=0.8,
                min_soc=0.2,
                max_soc=0.95,
            ),
            disturbance=Disturbance(start_s=1.0, deficit_mw=2.0),
            duration_s=20.0,
            dt_s=0.02,
        )

        self.assertLess(result.nadir_hz, 50.0)
        self.assertGreater(result.nadir_time_s, 1.0)
        self.assertGreater(max(result.storage_power_mw), max(result.diesel_power_mw[:20]))

    def test_storage_soc_limit_prevents_discharge_below_minimum(self):
        from hybrid_power_system_analysis.simu.primary_frequency_response import (
            DieselGovernor,
            Disturbance,
            GridFormingStorage,
            SystemFrequencyModel,
            simulate_primary_frequency_response,
        )

        result = simulate_primary_frequency_response(
            system=SystemFrequencyModel(f_nom_hz=50.0, s_base_mw=10.0, inertia_s=3.0, damping_mw_per_hz=0.0),
            diesel=DieselGovernor(
                reserve_mw=0.0,
                droop_mw_per_hz=0.0,
                time_constant_s=1.0,
                ramp_mw_per_s=0.0,
            ),
            storage=GridFormingStorage(
                discharge_limit_mw=3.0,
                charge_limit_mw=3.0,
                droop_mw_per_hz=5.0,
                response_time_s=0.05,
                energy_mwh=0.001,
                initial_soc=0.2,
                min_soc=0.2,
                max_soc=0.95,
                deadband_hz=0.0,
            ),
            disturbance=Disturbance(start_s=0.0, deficit_mw=4.0),
            duration_s=2.0,
            dt_s=0.1,
        )

        self.assertGreaterEqual(min(result.storage_soc), 0.2)
        self.assertEqual(0.0, max(result.storage_power_mw))

    def test_invalid_time_step_is_rejected(self):
        from hybrid_power_system_analysis.simu.primary_frequency_response import (
            DieselGovernor,
            Disturbance,
            GridFormingStorage,
            SystemFrequencyModel,
            simulate_primary_frequency_response,
        )

        with self.assertRaisesRegex(ValueError, "dt_s"):
            simulate_primary_frequency_response(
                system=SystemFrequencyModel(),
                diesel=DieselGovernor(
                    reserve_mw=1.0,
                    droop_mw_per_hz=1.0,
                    time_constant_s=1.0,
                    ramp_mw_per_s=1.0,
                ),
                storage=GridFormingStorage(
                    discharge_limit_mw=1.0,
                    charge_limit_mw=1.0,
                    droop_mw_per_hz=1.0,
                ),
                disturbance=Disturbance(start_s=0.0, deficit_mw=1.0),
                duration_s=1.0,
                dt_s=0.0,
            )

    def test_parallel_diesels_and_storages_have_individual_curves_and_totals(self):
        from hybrid_power_system_analysis.simu.primary_frequency_response import (
            DieselGovernor,
            Disturbance,
            GridFormingStorage,
            SystemFrequencyModel,
            simulate_primary_frequency_response,
        )

        result = simulate_primary_frequency_response(
            system=SystemFrequencyModel(f_nom_hz=50.0, s_base_mw=10.0, inertia_s=4.0, damping_mw_per_hz=1.0),
            diesels=[
                DieselGovernor(
                    name="diesel_slow",
                    reserve_mw=1.2,
                    droop_mw_per_hz=0.55,
                    time_constant_s=3.0,
                    ramp_mw_per_s=0.25,
                ),
                DieselGovernor(
                    name="diesel_fast",
                    reserve_mw=0.8,
                    droop_mw_per_hz=0.45,
                    time_constant_s=1.2,
                    ramp_mw_per_s=0.55,
                ),
            ],
            storages=[
                GridFormingStorage(
                    name="bess_1",
                    discharge_limit_mw=1.5,
                    charge_limit_mw=1.0,
                    droop_mw_per_hz=1.0,
                    inertia_mw_s_per_hz=0.25,
                    response_time_s=0.08,
                    energy_mwh=3.0,
                    initial_soc=0.8,
                    min_soc=0.2,
                    max_soc=0.95,
                ),
                GridFormingStorage(
                    name="bess_2",
                    discharge_limit_mw=0.8,
                    charge_limit_mw=0.8,
                    droop_mw_per_hz=0.7,
                    inertia_mw_s_per_hz=0.15,
                    response_time_s=0.15,
                    energy_mwh=1.5,
                    initial_soc=0.65,
                    min_soc=0.25,
                    max_soc=0.9,
                ),
            ],
            disturbance=Disturbance(start_s=1.0, deficit_mw=2.0),
            duration_s=20.0,
            dt_s=0.02,
        )

        self.assertEqual(["diesel_slow", "diesel_fast"], result.diesel_unit_names)
        self.assertEqual(["bess_1", "bess_2"], result.storage_unit_names)
        self.assertEqual(2, len(result.diesel_units_power_mw))
        self.assertEqual(2, len(result.storage_units_power_mw))
        self.assertEqual(2, len(result.storage_units_soc))
        early_idx = result.time_s.index(3.0)
        self.assertGreater(result.diesel_units_power_mw[1][early_idx], result.diesel_units_power_mw[0][early_idx])
        self.assertGreater(max(result.storage_units_power_mw[0]), max(result.storage_units_power_mw[1]))

        for idx in range(len(result.time_s)):
            self.assertAlmostEqual(
                result.diesel_power_mw[idx],
                result.diesel_units_power_mw[0][idx] + result.diesel_units_power_mw[1][idx],
                places=9,
            )
            self.assertAlmostEqual(
                result.storage_power_mw[idx],
                result.storage_units_power_mw[0][idx] + result.storage_units_power_mw[1][idx],
                places=9,
            )
        self.assertLess(result.nadir_hz, 50.0)

    def test_cli_writes_csv_with_frequency_and_resource_columns(self):
        root_dir = Path(__file__).resolve().parents[1]
        output_dir = root_dir / "tmp_test" / "primary_frequency_response"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "response.csv"

        completed = subprocess.run(
            [
                sys.executable,
                str(root_dir / "scripts" / "simulate_primary_frequency_response.py"),
                "--duration-s",
                "2",
                "--dt-s",
                "0.1",
                "--deficit-mw",
                "1.5",
                "--output",
                str(output_csv),
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("nadir_hz", completed.stdout)
        with output_csv.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))

        self.assertGreater(len(rows), 2)
        self.assertEqual(
            [
                "time_s",
                "frequency_hz",
                "delta_frequency_hz",
                "diesel_power_mw",
                "storage_power_mw",
                "storage_soc",
                "power_deficit_mw",
            ],
            list(rows[0].keys()),
        )

    def test_cli_typical_parallel_case_writes_unit_columns(self):
        root_dir = Path(__file__).resolve().parents[1]
        output_dir = root_dir / "tmp_test" / "primary_frequency_response"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "parallel_response.csv"

        completed = subprocess.run(
            [
                sys.executable,
                str(root_dir / "scripts" / "simulate_primary_frequency_response.py"),
                "--case",
                "parallel",
                "--duration-s",
                "2",
                "--dt-s",
                "0.1",
                "--output",
                str(output_csv),
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("diesel_units=2", completed.stdout)
        self.assertIn("storage_units=2", completed.stdout)
        with output_csv.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))

        self.assertIn("diesel_power_mw__diesel_slow", rows[0])
        self.assertIn("diesel_power_mw__diesel_fast", rows[0])
        self.assertIn("storage_power_mw__bess_1", rows[0])
        self.assertIn("storage_power_mw__bess_2", rows[0])
        self.assertIn("storage_soc__bess_1", rows[0])
        self.assertIn("storage_soc__bess_2", rows[0])

    def test_cli_reads_input_json_and_writes_separate_csv_and_summary_files(self):
        root_dir = Path(__file__).resolve().parents[1]
        work_dir = root_dir / "tmp_test" / "primary_frequency_response"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_json = work_dir / "case_input.json"
        output_csv = work_dir / "case_curve.csv"
        summary_json = work_dir / "case_summary.json"
        input_json.write_text(
            json.dumps(
                {
                    "system": {
                        "f_nom_hz": 50.0,
                        "s_base_mw": 10.0,
                        "inertia_s": 4.0,
                        "damping_mw_per_hz": 1.0,
                    },
                    "simulation": {
                        "duration_s": 3.0,
                        "dt_s": 0.1,
                    },
                    "disturbance": {
                        "start_s": 1.0,
                        "deficit_mw": 2.0,
                    },
                    "diesels": [
                        {
                            "name": "diesel_a",
                            "reserve_mw": 1.0,
                            "droop_mw_per_hz": 0.5,
                            "time_constant_s": 1.5,
                            "ramp_mw_per_s": 0.5,
                        },
                        {
                            "name": "diesel_b",
                            "reserve_mw": 1.0,
                            "droop_mw_per_hz": 0.4,
                            "time_constant_s": 2.5,
                            "ramp_mw_per_s": 0.3,
                        },
                    ],
                    "storages": [
                        {
                            "name": "bess_a",
                            "discharge_limit_mw": 1.0,
                            "charge_limit_mw": 1.0,
                            "droop_mw_per_hz": 1.0,
                            "inertia_mw_s_per_hz": 0.2,
                            "response_time_s": 0.1,
                            "energy_mwh": 2.0,
                            "initial_soc": 0.75,
                            "min_soc": 0.2,
                            "max_soc": 0.95,
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(root_dir / "scripts" / "simulate_primary_frequency_response.py"),
                "--input",
                str(input_json),
                "--output",
                str(output_csv),
                "--summary-output",
                str(summary_json),
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertTrue(output_csv.exists())
        self.assertTrue(summary_json.exists())
        self.assertIn("input=", completed.stdout)
        self.assertIn("summary=", completed.stdout)
        with output_csv.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        self.assertIn("diesel_power_mw__diesel_a", rows[0])
        self.assertIn("diesel_power_mw__diesel_b", rows[0])
        self.assertIn("storage_power_mw__bess_a", rows[0])

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertEqual(str(input_json), summary["input_file"])
        self.assertEqual(str(output_csv), summary["curve_output_file"])
        self.assertEqual(["diesel_a", "diesel_b"], summary["diesel_unit_names"])
        self.assertEqual(["bess_a"], summary["storage_unit_names"])
        self.assertLess(summary["nadir_hz"], 50.0)

    def test_cli_rejects_same_input_and_output_file(self):
        root_dir = Path(__file__).resolve().parents[1]
        work_dir = root_dir / "tmp_test" / "primary_frequency_response"
        work_dir.mkdir(parents=True, exist_ok=True)
        same_path = work_dir / "same_file.json"
        same_path.write_text("{}", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(root_dir / "scripts" / "simulate_primary_frequency_response.py"),
                "--input",
                str(same_path),
                "--output",
                str(same_path),
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("must be different files", completed.stderr)


if __name__ == "__main__":
    unittest.main()
