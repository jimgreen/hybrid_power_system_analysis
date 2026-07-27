import re
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT_DIR / "tmp_test" / "simu_loop_tests"


class SimulationLoopTest(unittest.TestCase):
    def test_default_realtime_input_file_names_follow_station_convention(self):
        from simu.simu_loop import default_config

        config = default_config()

        self.assertEqual("model.e", config.model_file.name)
        self.assertEqual("meas.e", config.meas_file.name)
        self.assertEqual("stat.e", config.dev_stat_file.name)
        self.assertEqual("weather.e", config.weather_file.name)
        self.assertEqual("device.e", config.dev_define_file.name)

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

    def test_measurement_snapshot_columns_align_header_and_rows(self):
        from simu.simu_loop import write_measurement_snapshot

        tmp = TMP_ROOT / "measurement_alignment"
        tmp.mkdir(parents=True, exist_ok=True)
        output = tmp / "real.e"

        write_measurement_snapshot(
            output,
            [],
            [
                ["0", "vm_short", "ACNode", "bus_1", "V", "3.0", "1", "100"],
                ["12", "p_long_measurement", "ACGenerator", "diesel_300kw", "P_GEN", "10.0", "0", "-123.456"],
            ],
            [],
        )

        lines = output.read_text(encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("@"))
        row = next(line for line in lines if line.startswith("#"))
        header_starts = [match.start() for match in re.finditer(r"\S+", header)][1:]
        row_starts = [match.start() for match in re.finditer(r"\S+", row)][1:]

        self.assertEqual(header_starts, row_starts)
        self.assertNotIn("\t", output.read_text(encoding="utf-8"))

    def test_simulate_once_interface_reads_files_and_writes_outputs(self):
        from simu.simu_loop import simulate_once

        class FakeSnapshot:
            def value(self, dev_type, dev_name, meas_type):
                return 77.0

        tmp = TMP_ROOT / "simulate_once_interface"
        tmp.mkdir(parents=True, exist_ok=True)
        model_path = tmp / "qinling.e"
        meas_path = tmp / "meas.e"
        weather_path = tmp / "weather.e"
        dev_stat_path = tmp / "dev_stat.e"
        yt_ctrl_path = tmp / "yt_ctrl.e"
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
        weather_path.write_text("<Weather>\n@ name value\n</Weather>\n", encoding="utf-8")
        dev_stat_path.write_text("<DeviceRunStatus>\n@ dev_type idx name run_stat\n</DeviceRunStatus>\n", encoding="utf-8")
        yt_ctrl_path.write_text("<GeneratorSetpoint>\n@ dev_type idx name run_stat p_set q_set v_set\n</GeneratorSetpoint>\n", encoding="utf-8")

        result = simulate_once(
            model_file=model_path,
            meas_file=meas_path,
            weather_file=weather_path,
            dev_stat_file=dev_stat_path,
            yt_ctrl_file=yt_ctrl_path,
            real_file=real_path,
            scada_file=scada_path,
            noise_std=0.0,
            random_seed=123,
            solver=lambda _path: (FakeSnapshot(), "fake"),
        )

        self.assertEqual(real_path, result.real_file)
        self.assertEqual(scada_path, result.scada_file)
        self.assertEqual(1, result.updated)
        self.assertIn("77", real_path.read_text(encoding="utf-8"))
        self.assertEqual(real_path.read_text(encoding="utf-8"), scada_path.read_text(encoding="utf-8"))

    def test_dev_stat_converter_setpoint_updates_dcac_pq_setpoints(self):
        from efile_read import EBook
        from simu.simu_loop import apply_dev_stat_file

        tmp = TMP_ROOT / "converter_setpoint"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "dev_stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<ConverterSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set",
                    "# DCACConverter 10 grid_inv_acp 1 -320 45",
                    "</ConverterSetpoint>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook(
            {
                "DCACConverter": [
                    {
                        "idx": "10",
                        "name": "grid_inv_acp",
                        "p_ac_set": "-350",
                        "q_ac_set": "0",
                        "v_ac_set": "0",
                        "run_stat": "0",
                    }
                ]
            }
        )

        changed = apply_dev_stat_file(model, dev_stat)

        row = model.data["DCACConverter"].data[0]
        self.assertEqual(3, changed)
        self.assertEqual("-320", row["p_ac_set"])
        self.assertEqual("45", row["q_ac_set"])
        self.assertEqual("1", row["run_stat"])

    def test_dev_stat_set_value_updates_setpoints_by_type(self):
        from efile_read import EBook
        from simu.simu_loop import apply_dev_stat_file

        tmp = TMP_ROOT / "set_value"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<SetValue>",
                    "@ dev_type dev_name set_type set_value",
                    "# ACGenerator diesel_300kw p_set 120",
                    "# ACGenerator diesel_300kw q_set 35",
                    "# ACGenerator diesel_300kw v_set 381",
                    "# DCACConverter grid_inv_acp p_set -320",
                    "# DCACConverter grid_inv_acp q_set 45",
                    "# DCACConverter grid_inv_acp v_set 379",
                    "# ACLoad load_ac_1 p_set 250",
                    "# ACLoad load_ac_1 q_set 90",
                    "</SetValue>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook(
            {
                "ACGenerator": [
                    {
                        "idx": "1",
                        "name": "diesel_300kw",
                        "p_set": "0",
                        "q_set": "0",
                        "v_set": "380",
                        "run_stat": "1",
                    }
                ],
                "DCACConverter": [
                    {
                        "idx": "10",
                        "name": "grid_inv_acp",
                        "p_ac_set": "-350",
                        "q_ac_set": "0",
                        "v_ac_set": "0",
                        "run_stat": "1",
                    }
                ],
                "ACLoad": [
                    {
                        "idx": "1",
                        "name": "load_ac_1",
                        "pv0": "100",
                        "qv0": "40",
                        "run_stat": "1",
                    }
                ],
            }
        )

        changed = apply_dev_stat_file(model, dev_stat)

        gen = model.data["ACGenerator"].data[0]
        conv = model.data["DCACConverter"].data[0]
        load = model.data["ACLoad"].data[0]
        self.assertEqual(8, changed)
        self.assertEqual("120", gen["p_set"])
        self.assertEqual("35", gen["q_set"])
        self.assertEqual("381", gen["v_set"])
        self.assertEqual("-320", conv["p_ac_set"])
        self.assertEqual("45", conv["q_ac_set"])
        self.assertEqual("379", conv["v_ac_set"])
        self.assertEqual("250", load["pv0"])
        self.assertEqual("90", load["qv0"])

    def test_dev_stat_run_stat_updates_all_device_types(self):
        from efile_read import EBook
        from simu.simu_loop import apply_dev_stat_file

        tmp = TMP_ROOT / "run_stat"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<RunStat>",
                    "@ dev_type dev_name run_stat",
                    "# ACNode bus_1 0",
                    "# DCDCConverter ess01_dcdc 0",
                    "# ACBreak sw_diesel_ac 0",
                    "</RunStat>",
                    "<CbOpenStat>",
                    "@ dev_type dev_name status",
                    "# ACBreak sw_diesel_ac 0",
                    "</CbOpenStat>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook(
            {
                "ACNode": [{"idx": "1", "name": "bus_1", "run_stat": "1"}],
                "DCDCConverter": [{"idx": "1", "name": "ess01_dcdc", "run_stat": "1"}],
                "ACBreak": [{"idx": "1", "name": "sw_diesel_ac", "run_stat": "1", "status": "1"}],
            }
        )

        changed = apply_dev_stat_file(model, dev_stat)

        self.assertEqual(4, changed)
        self.assertEqual("0", model.data["ACNode"].data[0]["run_stat"])
        self.assertEqual("0", model.data["DCDCConverter"].data[0]["run_stat"])
        self.assertEqual("0", model.data["ACBreak"].data[0]["run_stat"])
        self.assertEqual("0", model.data["ACBreak"].data[0]["status"])

    def test_storage_status_without_run_stat_does_not_blank_converter_run_stat(self):
        from efile_read import EBook
        from simu.simu_loop import apply_dev_stat_file

        tmp = TMP_ROOT / "storage_status_no_run_stat"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<RunStat>",
                    "@ dev_type dev_name run_stat",
                    "# ESS ess01 1",
                    "</RunStat>",
                    "<StorageStatus>",
                    "@ dev_type idx name soc_curr",
                    "# ESS 4 ess01 0.38",
                    "</StorageStatus>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook({"DCDCConverter": [{"idx": "4", "name": "ess01_dcdc", "run_stat": "1"}]})

        changed = apply_dev_stat_file(model, dev_stat)

        self.assertEqual(0, changed)
        self.assertEqual("1", model.data["DCDCConverter"].data[0]["run_stat"])

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
                    "<StorageSoc>",
                    "@ dev_type idx name run_stat soc_curr",
                    "# ESS 3 ess01 1 0.5",
                    "</StorageSoc>",
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
        self.assertIn("<StorageSoc>", text)
        self.assertIn("@ dev_type  idx  name", text)

    def test_realtime_inputs_apply_device_define_weather_limits_and_storage_constraints(self):
        from efile_read import EBook
        from simu.simu_loop import apply_realtime_inputs

        tmp = TMP_ROOT / "device_define_limits"
        tmp.mkdir(parents=True, exist_ok=True)
        model_path = tmp / "qinling.e"
        weather_path = tmp / "weather.e"
        dev_stat_path = tmp / "dev_stat.e"
        dev_define_path = tmp / "dev_define.e"
        yt_ctrl_path = tmp / "yt_ctrl.e"
        work_dir = tmp / "work"

        model_path.write_text(
            "\n".join(
                [
                    "<ACLoad>",
                    "@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
                    "# 1 load_ac_1 1 1.0 100 0 0 1.0 50 0 0 1",
                    "</ACLoad>",
                    "<ACGenerator>",
                    "@ idx name node control_type p_set q_set v_set alpha run_stat",
                    "# 1 diesel_300kw 2 V 0 0 380 1.0 1",
                    "</ACGenerator>",
                    "<DCDCConverter>",
                    "@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_set v_set run_stat",
                    "# 1 pv01_dcdc 1 2 0.01 0.01 P NONE 0 0 0 1",
                    "# 2 ess01_dcdc 2 3 0.01 0.01 P NONE 0 0 0 1",
                    "</DCDCConverter>",
                    "<DCACConverter>",
                    "@ idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat",
                    "# 1 wt01_rect 1 1 0.01 0.01 PQ NONE 0 0 0 0 1",
                    "</DCACConverter>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        weather_path.write_text(
            "\n".join(
                [
                    "<Weather>",
                    "@ time wind_speed_mps solar_irradiance_w_m2 air_temp_c load_kw",
                    "# 00:00:00 10 500 35 80",
                    "</Weather>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        dev_stat_path.write_text(
            "\n".join(
                [
                    "<GeneratorSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set v_set",
                    "# ACGenerator 1 diesel_300kw 1 10 0 380",
                    "# DCDCConverter 1 pv01_dcdc 1 80 0 0",
                    "# DCDCConverter 2 ess01_dcdc 1 40 0 0",
                    "</GeneratorSetpoint>",
                    "<ConverterSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set",
                    "# DCACConverter 1 wt01_rect 1 30 0",
                    "</ConverterSetpoint>",
                    "<LoadSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set",
                    "# ACLoad 1 load_ac_1 1 100 50",
                    "</LoadSetpoint>",
                    "<StorageSoc>",
                    "@ dev_type idx name run_stat soc_curr",
                    "# ESS 1 ess01 1 0.29",
                    "</StorageSoc>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        dev_define_path.write_text(
            "\n".join(
                [
                    "<wind_generator>",
                    "@ id name p_max p_min p_fur rated_power rated_wind_speed cut_in_speed cut_out_speed",
                    "# 1 wind_1 20 0 0 20 15 5 30",
                    "</wind_generator>",
                    "<pv_generator>",
                    "@ id name p_max p_min p_fur rated_power temp_coefficient reference_irradiance reference_temperature",
                    "# 1 pv_1 100 0 0 100 -0.004 1000 25",
                    "</pv_generator>",
                    "<diesel_generator>",
                    "@ id name p_max p_min",
                    "# 1 diesel_1 300 30",
                    "</diesel_generator>",
                    "<energyconsumer>",
                    "@ id name temp_base temp_factor",
                    "# 1 load_1 5 0",
                    "</energyconsumer>",
                    "<estorage>",
                    "@ id name emva soc_max soc_min soc_cur charge_p_max dis_charge_p_max",
                    "# 1 ess_1 50 0.9 0.3 0.5 20 20",
                    "</estorage>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        yt_ctrl_path.write_text(
            "\n".join(
                [
                    "<GeneratorSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set v_set",
                    "# ACGenerator 1 diesel_300kw 1 10 0 380",
                    "# DCDCConverter 1 pv01_dcdc 1 80 0 0",
                    "# DCDCConverter 2 ess01_dcdc 1 40 0 0",
                    "</GeneratorSetpoint>",
                    "<ConverterSetpoint>",
                    "@ dev_type idx name run_stat p_set q_set",
                    "# DCACConverter 1 wt01_rect 1 30 0",
                    "</ConverterSetpoint>",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        merged_model, changed, _model = apply_realtime_inputs(
            model_path,
            weather_path,
            dev_stat_path,
            yt_ctrl_path,
            dev_define_path,
            work_dir,
        )

        merged = EBook(merged_model)
        self.assertGreater(changed, 0)
        self.assertEqual("80", str(merged.data["ACLoad"].data[0]["pv0"]))
        self.assertEqual("40", str(merged.data["ACLoad"].data[0]["qv0"]))
        self.assertAlmostEqual(2.5, float(merged.data["DCACConverter"].data[0]["p_ac_set"]))
        self.assertAlmostEqual(48.0, float(merged.data["DCDCConverter"].data[0]["p_set"]))
        self.assertEqual("0", str(merged.data["DCDCConverter"].data[1]["p_set"]))
        self.assertEqual("30", str(merged.data["ACGenerator"].data[0]["p_set"]))

    def test_load_model_uses_96_point_curve_and_temperature_parameters(self):
        from efile_read import EBook
        from simu.simu_loop import apply_load_model

        curve_columns = [f"p{idx:03d}" for idx in range(1, 97)]
        curve_row = {"id": "1", "name": "load_ac_1"}
        curve_row.update({column: "1.0" for column in curve_columns})
        curve_row["p003"] = "0.5"
        model = EBook(
            {
                "ACLoad": [
                    {
                        "idx": "1",
                        "name": "load_ac_1",
                        "node": "1",
                        "pbase": "1.0",
                        "pv0": "100",
                        "pv1": "0",
                        "pv2": "0",
                        "qbase": "1.0",
                        "qv0": "50",
                        "qv1": "0",
                        "qv2": "0",
                        "run_stat": "1",
                    }
                ]
            }
        )
        dev_define = EBook(
            {
                "load_curve_96": [curve_row],
                "load_temperature": [
                    {
                        "id": "1",
                        "name": "load_ac_1",
                        "temp_base": "20",
                        "temp_factor": "0.02",
                    }
                ],
            }
        )

        changed = apply_load_model(
            model,
            dev_define,
            {
                "time_minutes": 30.0,
                "air_temp_c": 30.0,
            },
        )

        row = model.data["ACLoad"].data[0]
        self.assertEqual(2, changed)
        self.assertEqual("60", str(row["pv0"]))
        self.assertEqual("30", str(row["qv0"]))

    def test_storage_soc_update_uses_device_define_capacity_and_soc_bounds(self):
        from efile_read import EBook
        from simu.simu_loop import update_storage_soc

        tmp = TMP_ROOT / "soc_define_bounds"
        tmp.mkdir(parents=True, exist_ok=True)
        dev_stat = tmp / "dev_stat.e"
        dev_stat.write_text(
            "\n".join(
                [
                    "<StorageSoc>",
                    "@ dev_type idx name run_stat soc_curr",
                    "# ESS 1 ess01 1 0.5",
                    "</StorageSoc>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        dev_define = tmp / "dev_define.e"
        dev_define.write_text(
            "\n".join(
                [
                    "<estorage>",
                    "@ id name emva soc_max soc_min soc_cur charge_p_max dis_charge_p_max",
                    "# 1 ess_1 100 0.8 0.2 0.5 20 20",
                    "</estorage>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        model = EBook(
            {
                "DCDCConverter": [
                    {
                        "idx": "1",
                        "name": "ess01_dcdc",
                        "p_set": "50",
                        "run_stat": "1",
                    }
                ]
            }
        )

        changed = update_storage_soc(dev_stat, model, 3600.0, dev_define)

        self.assertEqual(1, changed)
        stat_book = EBook(dev_stat)
        self.assertEqual("0.2", str(stat_book.data["StorageSoc"].data[0]["soc_curr"]))

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

    def test_run_loop_step_mode_runs_fixed_steps_without_sleep(self):
        import simu.simu_loop as simu_loop
        from simu.simu_loop import SimulationConfig, SimulationResult, run_loop, setup_logger

        tmp = TMP_ROOT / "run_loop_step_mode"
        tmp.mkdir(parents=True, exist_ok=True)
        log_path = tmp / "loop.log"
        logger = setup_logger(log_path)
        config = SimulationConfig(
            model_file=tmp / "model.e",
            meas_file=tmp / "meas.e",
            weather_file=tmp / "weather.e",
            dev_stat_file=tmp / "stat.e",
            yt_ctrl_file=tmp / "yt_ctrl.e",
            real_file=tmp / "real.e",
            scada_file=tmp / "scada.e",
            loop_count=3,
            step_mode=True,
        )
        calls = []

        def fake_run_once(_config, rng=None):
            calls.append(len(calls) + 1)
            return SimulationResult(
                real_file=config.real_file,
                scada_file=config.scada_file,
                updated=len(calls),
                missing=0,
                overlay_updates=0,
                solver_info="iter=1, normF=0.0",
            )

        original_sleep = simu_loop.time.sleep
        try:
            simu_loop.time.sleep = lambda _seconds: (_ for _ in ()).throw(AssertionError("step mode must not sleep"))
            rc = run_loop(config, logger=logger, run_once_func=fake_run_once)
        finally:
            simu_loop.time.sleep = original_sleep

        text = log_path.read_text(encoding="utf-8")
        self.assertEqual(0, rc)
        self.assertEqual([1, 2, 3], calls)
        self.assertIn("步进模式", text)
        self.assertIn("第 3 轮仿真完成", text)


if __name__ == "__main__":
    unittest.main()
