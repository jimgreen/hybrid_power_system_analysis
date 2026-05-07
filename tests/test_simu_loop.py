import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT_DIR / "tmp_test" / "simu_loop_tests"


class SimulationLoopTest(unittest.TestCase):
    def test_overlay_file_updates_matching_model_rows_by_name_and_idx(self):
        from efile_read import EBook
        from simu.simu_loop import apply_overlay_file

        tmp = TMP_ROOT / "overlay"
        tmp.mkdir(parents=True, exist_ok=True)
        model_path = tmp / "model.e"
        overlay_path = tmp / "dev_ctrl.e"
        model_path.write_text(
            "\n".join(
                [
                    "<ACLoad>",
                    "@ idx name node pbase run_stat",
                    "# 1 load_1 1 10.0 1",
                    "# 2 load_2 2 20.0 1",
                    "</ACLoad>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        overlay_path.write_text(
            "\n".join(
                [
                    "<ACLoad>",
                    "@ name pbase run_stat",
                    "# load_1 12.5 0",
                    "</ACLoad>",
                    "<UnknownWeather>",
                    "@ name value",
                    "# wind_speed 6.7",
                    "</UnknownWeather>",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        book = EBook(model_path)
        changed = apply_overlay_file(book, overlay_path)

        rows = book.data["ACLoad"].data
        self.assertEqual(2, changed)
        self.assertEqual("12.5", rows[0]["pbase"])
        self.assertEqual("0", rows[0]["run_stat"])
        self.assertEqual("20.0", rows[1]["pbase"])

    def test_run_once_writes_real_and_noisy_scada_measurement_files(self):
        from simu.simu_loop import SimulationConfig, run_once

        class FakeSnapshot:
            def value(self, dev_type, dev_name, meas_type):
                self.last_key = (dev_type, dev_name, meas_type)
                return 100.0

        tmp = TMP_ROOT / "run_once"
        tmp.mkdir(parents=True, exist_ok=True)
        model_path = tmp / "ieee39.e"
        meas_path = tmp / "meas.e"
        real_path = tmp / "real.e"
        scada_path = tmp / "scada.e"
        model_path.write_text(
            "\n".join(
                [
                    "<ACNode>",
                    "@ idx name vbase voltage angle isl run_stat",
                    "# 0 bus_1 345 345 0 0 1",
                    "</ACNode>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        meas_path.write_text(
            "\n".join(
                [
                    "<Measurement>",
                    "@ idx name dev_type dev_name meas_type weight valid value",
                    "# 1 vm_bus_1 ACNode bus_1 V 4.0 1 0.0",
                    "</Measurement>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        config = SimulationConfig(
            model_file=model_path,
            meas_file=meas_path,
            weather_file=tmp / "weather.e",
            dev_stat_file=tmp / "dev_stat.e",
            yt_ctrl_file=tmp / "yt_ctrl.e",
            real_file=real_path,
            scada_file=scada_path,
            period_seconds=60.0,
            noise_std=0.0,
            random_seed=123,
        )

        result = run_once(config, solver=lambda _path: (FakeSnapshot(), "fake"))

        real_text = real_path.read_text(encoding="utf-8")
        scada_text = scada_path.read_text(encoding="utf-8")
        self.assertEqual(1, result.updated)
        self.assertEqual(0, result.missing)
        self.assertIn("100", real_text)
        self.assertEqual(real_text, scada_text)

    def test_simulation_logger_writes_to_configured_log_file(self):
        from simu.simu_loop import setup_logger

        tmp = TMP_ROOT / "logging"
        tmp.mkdir(parents=True, exist_ok=True)
        log_path = tmp / "simu.log"

        logger = setup_logger(log_path)
        logger.info("cycle message")

        text = log_path.read_text(encoding="utf-8")
        self.assertIn("INFO", text)
        self.assertIn("cycle message", text)

    def test_storage_soc_update_keeps_dev_stat_columns_space_aligned(self):
        from efile_read import EBook
        from simu.simu_loop import update_storage_soc

        tmp = TMP_ROOT / "soc_alignment"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "dev_stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<StorageStatus>",
                    "@ dev_type idx name run_stat soc_curr",
                    "# ESS 3 ess01 1 0.5",
                    "</StorageStatus>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook(
            {
                "DCDCConverter": [
                    {
                        "idx": "3",
                        "name": "ess01_dcdc",
                        "p_set": "60",
                        "run_stat": "1",
                    }
                ]
            }
        )

        changed = update_storage_soc(dev_stat, model, 60.0)

        text = dev_stat.read_text(encoding="utf-8")
        self.assertEqual(1, changed)
        self.assertNotIn("\t", text)
        self.assertIn("@ dev_type  idx  name", text)

    def test_run_loop_logs_cycle_summary(self):
        from simu.simu_loop import SimulationConfig, SimulationResult, run_loop, setup_logger

        tmp = TMP_ROOT / "run_loop_logging"
        tmp.mkdir(parents=True, exist_ok=True)
        log_path = tmp / "loop.log"
        logger = setup_logger(log_path)
        config = SimulationConfig(
            model_file=tmp / "ieee39.e",
            meas_file=tmp / "meas.e",
            weather_file=tmp / "weather.e",
            dev_stat_file=tmp / "dev_stat.e",
            yt_ctrl_file=tmp / "yt_ctrl.e",
            real_file=tmp / "real.e",
            scada_file=tmp / "scada.e",
            loop_count=1,
        )

        def fake_run_once(_config, rng=None):
            return SimulationResult(
                real_file=config.real_file,
                scada_file=config.scada_file,
                updated=3,
                missing=0,
                overlay_updates=2,
                solver_info="iter=1, normF=0.0",
            )

        rc = run_loop(config, logger=logger, run_once_func=fake_run_once)

        text = log_path.read_text(encoding="utf-8")
        self.assertEqual(0, rc)
        self.assertIn("仿真循环启动", text)
        self.assertIn("第 1 轮仿真完成", text)
        self.assertIn("updated=3", text)


if __name__ == "__main__":
    unittest.main()
