import re
import json
import threading
from urllib.request import Request, urlopen
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT_DIR / "tmp_test" / "polar_microgrid_sim_tests"


def _write_minimal_inputs(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.e").write_text(
        "\n".join(
            [
                "<ACLoad>",
                "@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
                "# 1 load_ac_1 1 1.0 100 0 0 1.0 40 0 0 1",
                "</ACLoad>",
                "<ACGenerator>",
                "@ idx name node control_type p_set q_set v_set alpha run_stat",
                "# 1 diesel_300kw 1 V 80 0 380 1.0 1",
                "</ACGenerator>",
                "<ACBreak>",
                "@ idx name i_node j_node status run_stat",
                "# 1 sw_diesel_ac 1 2 1 1",
                "</ACBreak>",
                "<DCDCConverter>",
                "@ idx name i_node j_node r1 r2 control_type p_set i_set v_set run_stat",
                "# 1 ess01_dcdc 1 2 0.01 0.01 P 20 0 0 1",
                "</DCDCConverter>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (base / "meas.e").write_text(
        "\n".join(
            [
                "<Measurement>",
                "@ idx name dev_type dev_name meas_type weight valid value",
                "# 1 p_load ACLoad load_ac_1 P_LOAD 4.0 1 0",
                "</Measurement>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (base / "stat.e").write_text(
        "\n".join(
            [
                "<RunStat>",
                "@ dev_type dev_name run_stat",
                "# ACLoad load_ac_1 1",
                "# ACGenerator diesel_300kw 1",
                "# ACBreak sw_diesel_ac 1",
                "# DCDCConverter ess01_dcdc 1",
                "# ESS ess01 1",
                "</RunStat>",
                "<CbOpenStat>",
                "@ dev_type dev_name status",
                "# ACBreak sw_diesel_ac 1",
                "</CbOpenStat>",
                "<SetValue>",
                "@ dev_type dev_name set_type set_value",
                "# ACGenerator diesel_300kw p_set 80",
                "# ACLoad load_ac_1 p_set 100",
                "</SetValue>",
                "<StorageSoc>",
                "@ dev_type idx name soc_curr",
                "# ESS 1 ess01 0.5",
                "</StorageSoc>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (base / "weather.e").write_text(
        "\n".join(
            [
                "<Weather>",
                "@ time wind_speed_mps air_temp_c air_pressure_hpa solar_irradiance_w_m2 humidity_pct load_kw",
                "# 00:00:00 12  -18  960  0  72  100",
                "</Weather>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (base / "device.e").write_text("<estorage>\n@ id name emva soc_max soc_min soc_cur charge_p_max dis_charge_p_max\n</estorage>\n", encoding="utf-8")


def test_student_commands_are_written_to_runtime_stat_and_measurements_are_published():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "inputs"
    runtime_dir = TMP_ROOT / "runtime"
    _write_minimal_inputs(sim_dir)

    def fake_kernel(config):
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "123.4"]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "124.0"]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, "fake")

    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=fake_kernel)
    accepted = service.apply_student_commands(
        {
            "run_status": [{"dev_type": "ACBreak", "dev_name": "sw_diesel_ac", "run_stat": 0}],
            "set_values": [
                {"dev_type": "ACGenerator", "dev_name": "diesel_300kw", "set_type": "p_set", "set_value": 120},
                {"dev_type": "ACLoad", "dev_name": "load_ac_1", "set_type": "p_set", "set_value": 150},
            ],
        },
        source="student-01",
    )
    result = service.step()

    stat_text = (runtime_dir / "stat.e").read_text(encoding="utf-8")
    assert accepted == {"run_status": 1, "set_values": 2}
    assert "<RunStat>" in stat_text
    assert "<SetValue>" in stat_text
    assert re.search(r"^#\s+ACBreak\s+sw_diesel_ac\s+0\s*$", stat_text, re.MULTILINE)
    assert re.search(r"^#\s+ACGenerator\s+diesel_300kw\s+p_set\s+120\s*$", stat_text, re.MULTILINE)
    assert re.search(r"^#\s+ACLoad\s+load_ac_1\s+p_set\s+150\s*$", stat_text, re.MULTILINE)
    assert result["measurements"]["scada"][0]["value"] == 124.0
    assert result["clock"]["minute"] == 1


def test_clock_control_supports_start_pause_speed_and_single_step():
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "clock_inputs"
    runtime_dir = TMP_ROOT / "clock_runtime"
    _write_minimal_inputs(sim_dir)
    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=lambda _config: None)

    assert service.control_clock({"action": "start", "speed": 4})["state"] == "running"
    assert service.snapshot()["clock"]["speed"] == 4.0
    assert service.control_clock({"action": "pause"})["state"] == "paused"
    assert service.control_clock({"action": "step"})["minute"] == 1
    assert service.control_clock({"action": "stop"})["state"] == "stopped"


def test_local_curves_faults_and_modes_are_projected_before_kernel_call():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "local_inputs"
    runtime_dir = TMP_ROOT / "local_runtime"
    _write_minimal_inputs(sim_dir)

    observed = {}

    def fake_kernel(config):
        observed["weather"] = config.weather_file.read_text(encoding="utf-8")
        observed["stat"] = config.dev_stat_file.read_text(encoding="utf-8")
        observed["model"] = config.model_file.read_text(encoding="utf-8")
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "123.4"]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "124.0"]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, "fake")

    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=fake_kernel)
    service.set_curves(
        {
            "weather": [
                {
                    "minute": 0,
                    "wind_speed_mps": 18,
                    "air_temp_c": -22,
                    "air_pressure_hpa": 950,
                    "solar_irradiance_w_m2": 300,
                    "humidity_pct": 80,
                }
            ],
            "loads": {"load_ac_1": [{"minute": 0, "p_kw": 140}]},
        }
    )
    service.set_local_settings(
        {
            "device_faults": [
                {
                    "dev_type": "ACBreak",
                    "dev_name": "sw_diesel_ac",
                    "start_minute": 0,
                    "clear_minute": 2,
                    "run_stat": 0,
                    "status": 0,
                }
            ],
            "measurement_faults": [
                {"name": "p_load", "fault_type": "zero", "start_minute": 0, "clear_minute": 2}
            ],
            "modes": [{"dev_type": "ACGenerator", "dev_name": "diesel_300kw", "mode": "PQ"}],
        }
    )
    result = service.step()

    assert re.search(r"^#\s+00:00:00\s+18\s+-22\s+950\s+300\s+80\s+140\s*$", observed["weather"], re.MULTILINE)
    assert re.search(r"^#\s+ACBreak\s+sw_diesel_ac\s+0\s*$", observed["stat"], re.MULTILINE)
    assert re.search(r"^#\s+ACBreak\s+sw_diesel_ac\s+0\s*$", observed["stat"], re.MULTILINE)
    assert re.search(r"^#\s+1\s+diesel_300kw\s+1\s+PQ\s+80\s+0\s+380\s+1.0\s+1\s*$", observed["model"], re.MULTILINE)
    assert result["measurements"]["scada"][0]["value"] == 0.0


def test_snapshot_exposes_measurement_definitions_before_first_step():
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "measurement_definition_inputs"
    runtime_dir = TMP_ROOT / "measurement_definition_runtime"
    _write_minimal_inputs(sim_dir)
    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=lambda _config: None)

    snapshot = service.snapshot()

    assert snapshot["measurements"]["definitions"][0]["name"] == "p_load"
    assert snapshot["measurements"]["definitions"][0]["dev_name"] == "load_ac_1"


def test_measurement_fault_normal_is_noop_and_dead_fault_can_use_median_value():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "measurement_fault_inputs"
    runtime_dir = TMP_ROOT / "measurement_fault_runtime"
    _write_minimal_inputs(sim_dir)

    def fake_kernel(config):
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "123.4"]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "124.0"]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, "fake")

    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=fake_kernel)
    service.set_local_settings(
        {
            "measurement_faults": [
                {"name": "p_load", "fault_type": "normal", "start_minute": 0, "clear_minute": 2, "bias": 99}
            ]
        }
    )
    normal_result = service.step()

    assert normal_result["measurements"]["scada"][0]["value"] == 124.0

    service.control_clock({"action": "stop"})
    service.set_local_settings(
        {
            "measurement_faults": [
                {
                    "name": "p_load",
                    "fault_type": "dead",
                    "start_minute": 0,
                    "clear_minute": 2,
                    "median": 88.8,
                    "bias": 1.2,
                }
            ]
        }
    )
    dead_result = service.step()

    assert dead_result["measurements"]["scada"][0]["value"] == 90.0


def test_http_server_exposes_snapshot_command_and_step_endpoints():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.server import make_http_server
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "http_inputs"
    runtime_dir = TMP_ROOT / "http_runtime"
    _write_minimal_inputs(sim_dir)

    def fake_kernel(config):
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "222.2"]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "223.0"]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, "fake")

    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=fake_kernel)
    server = make_http_server(("127.0.0.1", 0), service, role="simulator")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        command = json.dumps(
            {"run_status": [{"dev_type": "ACBreak", "dev_name": "sw_diesel_ac", "run_stat": 0}]}
        ).encode("utf-8")
        with urlopen(
            Request(
                f"{base_url}/api/student/commands",
                data=command,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        ) as response:
            accepted = json.loads(response.read().decode("utf-8"))
        with urlopen(
            Request(f"{base_url}/api/step", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"),
            timeout=5,
        ) as response:
            stepped = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base_url}/api/snapshot", timeout=5) as response:
            snapshot = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert accepted == {"run_status": 1, "set_values": 0}
    assert stepped["measurements"]["scada"][0]["value"] == 223.0
    assert snapshot["clock"]["minute"] == 1


def test_web_consoles_are_split_into_home_and_topic_pages():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web"
    expected = {
        "simulator": ["overview", "curves", "faults", "modes", "runtime", "logs", "measurements"],
        "trainee": ["overview", "measurements", "controls", "commands", "history"],
    }

    for role, pages in expected.items():
        html = (web_root / role / "index.html").read_text(encoding="utf-8")
        js = (web_root / role / "app.js").read_text(encoding="utf-8")
        assert 'data-default-page="overview"' in html
        assert 'class="page-nav"' in html
        assert html.count('class="page-section') >= len(pages)
        for page in pages:
            assert f'data-page="{page}"' in html
            assert f'data-nav-page="{page}"' in html
        assert "function showPage" in js
        assert "location.hash" in js


def test_simulator_curve_page_uses_dense_curve_editor_and_hourly_table():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")

    assert 'data-page="curves"' in html
    assert 'id="curveEditorChart"' in html
    assert 'id="hourlyCurveTable"' in html
    assert 'id="saveCurves"' in html
    assert 'data-curve-role="dense-chart"' in html
    assert 'data-curve-role="hourly-table"' in html
    assert "9760点" in html
    assert "const CURVE_POINT_COUNT = 9760" in js
    assert "function applyCurveDrag" in js
    assert "function renderHourlyTable" in js
    assert "function resizeCurveCanvas" in js
    assert "contenteditable" in js
    assert "saveCurves" in js


def test_simulator_fault_page_uses_device_and_measurement_subtabs():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")

    assert 'data-page="faults"' in html
    assert 'data-fault-tab="devices"' in html
    assert 'data-fault-tab="measurements"' in html
    assert 'id="deviceFaultTable"' in html
    assert 'id="measurementFaultTable"' in html
    assert 'id="saveDeviceFaults"' in html
    assert 'id="saveMeasurementFaults"' in html
    assert "设备名称" in html
    assert "运行状态" in html
    assert "故障启始时刻" in html
    assert "结束时刻" in html
    assert "量测状态" in html
    assert "正常" in html
    assert "死数" in html
    assert "0值" in html
    assert "中值" in html
    assert "误差" in html
    assert "function setFaultTab" in js
    assert "function renderDeviceFaultTable" in js
    assert "function renderMeasurementFaultTable" in js
    assert "data-device-field" in js
    assert "data-meas-field" in js


def test_simulator_mode_page_uses_device_tree_and_editable_mode_table():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-page="modes"' in html
    assert 'class="mode-page-layout"' in html
    assert 'id="modeDeviceTree"' in html
    assert 'id="modeDeviceTable"' in html
    assert 'id="pushModes"' in html
    assert "设备树" in html
    assert "设备名称" in html
    assert "运行模式" in html
    assert "function syncModesFromDevices" in js
    assert "function renderModeDeviceTree" in js
    assert "function renderModeDeviceTable" in js
    assert "function setModeFilter" in js
    assert "data-mode-tree-type" in js
    assert "data-mode-device-index" in js
    assert "data-mode-field" in js
    assert "mode-page-layout" in css
    assert "mode-tree-panel" in css
    assert "mode-editor-table" in css


def test_simulator_has_runtime_log_and_realtime_measurement_pages():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-nav-page="logs"' in html
    assert 'data-nav-page="measurements"' in html
    assert 'data-page="logs"' in html
    assert 'data-page="measurements"' in html
    assert 'id="runtimeLogTable"' in html
    assert 'id="runtimeLogSummary"' in html
    assert 'id="measurementCompareTable"' in html
    assert 'id="measurementCompareSummary"' in html
    assert "运行日志" in html
    assert "量测值与真值" in html
    assert "真值" in html
    assert "量测值" in html
    assert "偏差" in html
    assert "function appendRuntimeLog" in js
    assert "function renderRuntimeLogs" in js
    assert "function measurementCompareRows" in js
    assert "function renderMeasurementCompareTable" in js
    assert "runtime-log-table" in js
    assert "measurement-compare-table" in js
    assert "runtime-log-wrap" in css
    assert "measurement-compare-wrap" in css
