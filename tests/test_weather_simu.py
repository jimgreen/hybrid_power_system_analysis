import unittest
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT_DIR / "tmp_test" / "weather_simu_tests"


class WeatherSimulationTest(unittest.TestCase):
    def test_update_weather_file_uses_matching_minute_from_csv(self):
        from efile_read import EBook
        from simu.weather_simu import update_weather_file

        tmp = TMP_ROOT / "update_existing"
        tmp.mkdir(parents=True, exist_ok=True)
        csv_path = tmp / "weather.csv"
        weather_path = tmp / "weather.e"
        csv_path.write_text(
            "\n".join(
                [
                    "minute,time,wind_speed_mps,solar_irradiance_w_m2,air_temp_c,load_kw",
                    "0,00:00,1.0,0.0,20.0,100.0",
                    "1,00:01,2.5,3.0,21.0,101.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        weather_path.write_text(
            "\n".join(
                [
                    "<Weather>",
                    "@ name value",
                    "# wind_speed_mps 0",
                    "# solar_irradiance_w_m2 0",
                    "# air_temp_c 0",
                    "# load_kw 0",
                    "</Weather>",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        result = update_weather_file(weather_path, csv_path, datetime(2026, 5, 7, 0, 1))

        block = EBook(weather_path).data["Weather"]
        row = block.data[0]
        self.assertEqual(5, result.updated)
        self.assertEqual(1, result.minute)
        self.assertEqual(["time", "wind_speed_mps", "solar_irradiance_w_m2", "air_temp_c", "load_kw"], block.header_list)
        self.assertEqual("00:01:00", row["time"])
        self.assertEqual("2.5", row["wind_speed_mps"])
        self.assertEqual("3.0", row["solar_irradiance_w_m2"])
        self.assertEqual("21.0", row["air_temp_c"])
        self.assertEqual("101.0", row["load_kw"])
        self.assertNotIn("\t", weather_path.read_text(encoding="utf-8"))

    def test_update_weather_file_interpolates_between_minute_rows(self):
        from efile_read import EBook
        from simu.weather_simu import update_weather_file

        tmp = TMP_ROOT / "interpolate"
        tmp.mkdir(parents=True, exist_ok=True)
        csv_path = tmp / "weather.csv"
        weather_path = tmp / "weather.e"
        csv_path.write_text(
            "\n".join(
                [
                    "minute,time,wind_speed_mps,solar_irradiance_w_m2,air_temp_c,load_kw",
                    "0,00:00,10.0,0.0,20.0,100.0",
                    "1,00:01,20.0,60.0,22.0,160.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        result = update_weather_file(weather_path, csv_path, datetime(2026, 5, 7, 0, 0, 30))

        row = EBook(weather_path).data["Weather"].data[0]
        self.assertEqual(0, result.minute)
        self.assertEqual("00:00:30", row["time"])
        self.assertEqual("15.000", row["wind_speed_mps"])
        self.assertEqual("30.000", row["solar_irradiance_w_m2"])
        self.assertEqual("21.000", row["air_temp_c"])
        self.assertEqual("130.000", row["load_kw"])

    def test_default_weather_simulation_period_is_five_seconds(self):
        from simu.weather_simu import DEFAULT_PERIOD_SECONDS, WeatherSimConfig

        self.assertEqual(5.0, DEFAULT_PERIOD_SECONDS)
        self.assertEqual(5.0, WeatherSimConfig().period_seconds)

    def test_update_weather_file_creates_default_weather_file_when_missing(self):
        from efile_read import EBook
        from simu.weather_simu import update_weather_file

        tmp = TMP_ROOT / "create_missing"
        tmp.mkdir(parents=True, exist_ok=True)
        csv_path = tmp / "weather.csv"
        weather_path = tmp / "weather.e"
        csv_path.write_text(
            "\n".join(
                [
                    "minute,time,wind_speed_mps,solar_irradiance_w_m2,air_temp_c,load_kw",
                    "75,01:15,12.0,0.0,19.5,88.8",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if weather_path.exists():
            weather_path.unlink()

        result = update_weather_file(weather_path, csv_path, datetime(2026, 5, 7, 1, 15))

        block = EBook(weather_path).data["Weather"]
        row = block.data[0]
        self.assertEqual(5, result.updated)
        self.assertEqual(["time", "wind_speed_mps", "solar_irradiance_w_m2", "air_temp_c", "load_kw"], block.header_list)
        self.assertEqual("01:15:00", row["time"])
        self.assertEqual("12.0", row["wind_speed_mps"])
        self.assertEqual("88.8", row["load_kw"])

    def test_run_loop_prints_update_summary(self):
        from simu.weather_simu import WeatherSimConfig, run_loop

        tmp = TMP_ROOT / "print_summary"
        tmp.mkdir(parents=True, exist_ok=True)
        csv_path = tmp / "weather.csv"
        weather_path = tmp / "weather.e"
        csv_path.write_text(
            "\n".join(
                [
                    "minute,time,wind_speed_mps,solar_irradiance_w_m2,air_temp_c,load_kw",
                    "1,00:01,2.5,3.0,21.0,101.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        printed = []
        config = WeatherSimConfig(
            weather_file=weather_path,
            weather_csv=csv_path,
            loop_count=1,
            log_file=tmp / "weather.log",
        )

        rc = run_loop(
            config,
            now_func=lambda: datetime(2026, 5, 7, 0, 1),
            print_func=printed.append,
        )

        self.assertEqual(0, rc)
        self.assertTrue(any("第 1 轮气象数据更新完成" in line for line in printed))
        self.assertTrue(any("wind=2.5" in line and "load=101.0" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
