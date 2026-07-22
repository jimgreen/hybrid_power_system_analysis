import re
import json
import threading
import time
import uuid
from urllib.request import Request, urlopen
from pathlib import Path

import pytest


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


def test_year_curve_mode_uses_absolute_minute_for_weather_projection():
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "year_curve_inputs"
    runtime_dir = TMP_ROOT / "year_curve_runtime"
    _write_minimal_inputs(sim_dir)
    observed = {}

    def fake_kernel(config):
        observed["weather"] = config.weather_file.read_text(encoding="utf-8")
        return None

    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=fake_kernel)
    service.set_curves(
        {
            "mode": "year",
            "point_count": 8760,
            "time_step_minutes": 60,
            "weather": [
                {"minute": 0, "wind_speed_mps": 10},
                {"minute": 1440, "wind_speed_mps": 20},
            ],
            "loads": {},
        }
    )
    service.control_clock({"minute": 1440})
    service.step()

    assert re.search(r"^#\s+00:00:00\s+20\b", observed["weather"], re.MULTILINE)


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


def test_multi_model_manager_keeps_runtime_clock_and_settings_isolated():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator, SimulationModelSpec

    sim_a = TMP_ROOT / "multi_inputs_a"
    sim_b = TMP_ROOT / "multi_inputs_b"
    runtime_dir = TMP_ROOT / "multi_runtime"
    _write_minimal_inputs(sim_a)
    _write_minimal_inputs(sim_b)

    def fake_kernel(config):
        model_id = config.real_file.parent.name
        value = "301.0" if model_id == "station_a" else "402.0"
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", f"p_load_{model_id}", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", value]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", f"p_load_{model_id}", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", value]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, f"fake-{model_id}")

    manager = MultiModelSimulator(
        [
            SimulationModelSpec("station_a", sim_a, "考察站A"),
            SimulationModelSpec("station_b", sim_b, "考察站B"),
        ],
        runtime_dir=runtime_dir,
        kernel=fake_kernel,
    )

    assert [item["id"] for item in manager.models()] == ["station_a", "station_b"]
    assert manager.default_model_id == "station_a"

    manager.control_clock({"action": "start", "speed": 5}, model_id="station_a")
    manager.set_curves({"weather": [{"minute": 0, "wind_speed_mps": 33}], "loads": {}}, model_id="station_b")
    stepped_b = manager.step(model_id="station_b")

    snapshot_a = manager.snapshot("station_a")
    snapshot_b = manager.snapshot("station_b")

    assert snapshot_a["model"]["id"] == "station_a"
    assert snapshot_b["model"]["id"] == "station_b"
    assert snapshot_a["clock"]["state"] == "running"
    assert snapshot_a["clock"]["speed"] == 5.0
    assert snapshot_a["clock"]["minute"] == 0
    assert snapshot_b["clock"]["state"] == "stopped"
    assert snapshot_b["clock"]["minute"] == 1
    assert snapshot_a["curves"]["weather"] == []
    assert snapshot_b["curves"]["weather"][0]["wind_speed_mps"] == 33
    assert stepped_b["measurements"]["scada"][0]["name"] == "p_load_station_b"
    assert snapshot_a["summary"]["runtime_dir"].endswith("station_a")
    assert snapshot_b["summary"]["runtime_dir"].endswith("station_b")


def test_multi_model_manager_clones_current_model_folder_and_runtime_state():
    from simu.simu_loop import SimulationResult, write_measurement_snapshot
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator

    sim_root = TMP_ROOT / f"clone_model_root_{uuid.uuid4().hex[:8]}"
    runtime_dir = TMP_ROOT / f"clone_model_runtime_{uuid.uuid4().hex[:8]}"
    _write_minimal_inputs(sim_root)

    def fake_kernel(config):
        write_measurement_snapshot(
            config.real_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "155.0"]],
            [],
        )
        write_measurement_snapshot(
            config.scada_file,
            [],
            [["1", "p_load", "ACLoad", "load_ac_1", "P_LOAD", "4.0", "1", "156.0"]],
            [],
        )
        return SimulationResult(config.real_file, config.scada_file, 1, 0, 3, "fake")

    manager = MultiModelSimulator.discover(sim_dir=sim_root, runtime_dir=runtime_dir, kernel=fake_kernel)
    manager.set_curves(
        {
            "mode": "day",
            "weather": [{"minute": 0, "wind_speed_mps": 31}],
            "loads": {"load_ac_1": [{"minute": 0, "p_kw": 188}]},
        },
        model_id="default",
    )
    manager.set_local_settings(
        {
            "device_faults": [{"dev_type": "ACBreak", "dev_name": "sw_diesel_ac", "start_minute": 10}],
            "measurement_faults": [{"name": "p_load", "fault_type": "dead"}],
            "modes": [{"dev_type": "ACGenerator", "dev_name": "diesel_300kw", "mode": "PQ"}],
        },
        model_id="default",
    )
    manager.apply_student_commands(
        {"set_values": [{"dev_type": "ACLoad", "dev_name": "load_ac_1", "set_type": "p_set", "set_value": 188}]},
        source="student-01",
        model_id="default",
    )
    manager.step("default")

    clone_name = f"station_copy_{uuid.uuid4().hex[:8]}"
    model = manager.clone_model("default", clone_name)
    snapshot = manager.snapshot(clone_name)
    clone_dir = sim_root / "models" / clone_name

    assert model["id"] == clone_name
    assert clone_dir.is_dir()
    assert (clone_dir / "model.e").exists()
    assert (clone_dir / "curves.json").exists()
    assert (clone_dir / "local_settings.json").exists()
    assert (clone_dir / "commands.json").exists()
    assert snapshot["curves"]["weather"][0]["wind_speed_mps"] == 31
    assert snapshot["settings"]["measurement_faults"][0]["fault_type"] == "dead"
    assert snapshot["commands"]["history"][0]["source"] == "student-01"
    assert snapshot["measurements"]["scada"][0]["value"] == 156.0


def test_multi_model_manager_discovers_models_from_specified_subdirectories():
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator

    sim_root = TMP_ROOT / f"specified_models_root_{uuid.uuid4().hex[:8]}"
    runtime_dir = TMP_ROOT / f"specified_models_runtime_{uuid.uuid4().hex[:8]}"
    models_dir = sim_root / "model_cases"
    _write_minimal_inputs(sim_root)
    _write_minimal_inputs(sim_root / "models" / "ignored_default_models_dir")
    _write_minimal_inputs(models_dir / "station_b")
    _write_minimal_inputs(models_dir / "station_a")
    (sim_root / "models.json").write_text(
        json.dumps({"models": [{"id": "stale_manifest_model", "sim_dir": "."}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    manager = MultiModelSimulator.discover(
        sim_dir=sim_root,
        runtime_dir=runtime_dir,
        models_dir=models_dir,
        kernel=lambda _config: None,
    )
    models = manager.models()

    assert [model["id"] for model in models] == ["station_a", "station_b"]
    assert manager.default_model_id == "station_a"
    assert all(Path(model["sim_dir"]).parent == models_dir for model in models)


def test_multi_model_manager_deduplicates_model_names_and_rejects_duplicate_clone_names():
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator

    sim_root = TMP_ROOT / f"dedupe_model_root_{uuid.uuid4().hex[:8]}"
    runtime_dir = TMP_ROOT / f"dedupe_model_runtime_{uuid.uuid4().hex[:8]}"
    _write_minimal_inputs(sim_root)
    _write_minimal_inputs(sim_root / "models" / "默认模型")
    _write_minimal_inputs(sim_root / "models" / "默认模型_copy")

    manager = MultiModelSimulator.discover(sim_dir=sim_root, runtime_dir=runtime_dir, kernel=lambda _config: None)
    models = manager.models()

    assert [model["id"] for model in models] == ["默认模型", "默认模型_copy"]
    assert [model["name"] for model in models].count("默认模型") == 1
    with pytest.raises(ValueError, match="模型已存在"):
        manager.clone_model("默认模型", "默认模型")
    with pytest.raises(ValueError, match="模型已存在"):
        manager.clone_model("默认模型", "默认模型_copy")


def test_http_server_routes_model_scoped_requests_to_independent_simulators():
    from hybrid_power_system_analysis.polar_microgrid_sim.server import make_http_server
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator, SimulationModelSpec

    sim_a = TMP_ROOT / "http_multi_inputs_a"
    sim_b = TMP_ROOT / "http_multi_inputs_b"
    runtime_dir = TMP_ROOT / "http_multi_runtime"
    _write_minimal_inputs(sim_a)
    _write_minimal_inputs(sim_b)
    manager = MultiModelSimulator(
        [
            SimulationModelSpec("station_a", sim_a, "考察站A"),
            SimulationModelSpec("station_b", sim_b, "考察站B"),
        ],
        runtime_dir=runtime_dir,
        kernel=lambda _config: None,
    )
    server = make_http_server(("127.0.0.1", 0), manager, role="simulator")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base_url}/api/models", timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
        clock_payload = json.dumps({"action": "start", "speed": 7}).encode("utf-8")
        with urlopen(
            Request(
                f"{base_url}/api/clock?model_id=station_b",
                data=clock_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        ) as response:
            clock_b = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base_url}/api/snapshot?model_id=station_a", timeout=5) as response:
            snapshot_a = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base_url}/api/snapshot?model_id=station_b", timeout=5) as response:
            snapshot_b = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert [item["id"] for item in models["models"]] == ["station_a", "station_b"]
    assert models["active_model_id"] == "station_a"
    assert clock_b["state"] == "running"
    assert clock_b["speed"] == 7.0
    assert snapshot_a["clock"]["state"] == "stopped"
    assert snapshot_b["clock"]["state"] == "running"
    assert snapshot_b["model"]["id"] == "station_b"


def test_http_server_can_clone_current_model_folder():
    from hybrid_power_system_analysis.polar_microgrid_sim.server import make_http_server
    from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator

    sim_root = TMP_ROOT / f"http_clone_root_{uuid.uuid4().hex[:8]}"
    runtime_dir = TMP_ROOT / f"http_clone_runtime_{uuid.uuid4().hex[:8]}"
    _write_minimal_inputs(sim_root)
    manager = MultiModelSimulator.discover(sim_dir=sim_root, runtime_dir=runtime_dir, kernel=lambda _config: None)
    server = make_http_server(("127.0.0.1", 0), manager, role="simulator")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    clone_name = f"station_http_copy_{uuid.uuid4().hex[:8]}"
    try:
        clone_payload = json.dumps({"name": clone_name}).encode("utf-8")
        with urlopen(
            Request(
                f"{base_url}/api/models/clone?model_id=default",
                data=clone_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        ) as response:
            cloned = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base_url}/api/models", timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert cloned["model"]["id"] == clone_name
    assert cloned["active_model_id"] == clone_name
    assert clone_name in [item["id"] for item in models["models"]]
    assert (sim_root / "models" / clone_name / "model.e").exists()


def test_clock_worker_advances_running_clock_and_obeys_pause_stop():
    from hybrid_power_system_analysis.polar_microgrid_sim.server import start_clock_worker
    from hybrid_power_system_analysis.polar_microgrid_sim.service import PolarMicrogridSimulator

    sim_dir = TMP_ROOT / "worker_clock_inputs"
    runtime_dir = TMP_ROOT / "worker_clock_runtime"
    _write_minimal_inputs(sim_dir)
    service = PolarMicrogridSimulator(sim_dir=sim_dir, runtime_dir=runtime_dir, kernel=lambda _config: None)
    stop_event = threading.Event()
    worker = start_clock_worker(service, stop_event)
    try:
        service.control_clock({"action": "start", "speed": 10})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and service.snapshot()["clock"]["minute"] < 2:
            time.sleep(0.05)
        running_clock = service.snapshot()["clock"]

        service.control_clock({"action": "pause"})
        paused_minute = service.snapshot()["clock"]["minute"]
        time.sleep(0.25)
        paused_clock = service.snapshot()["clock"]

        stopped_clock = service.control_clock({"action": "stop"})
    finally:
        stop_event.set()
        worker.join(timeout=2)

    assert running_clock["minute"] >= 1
    assert paused_clock["state"] == "paused"
    assert paused_clock["minute"] == paused_minute
    assert stopped_clock["state"] == "stopped"
    assert stopped_clock["minute"] == 0
    assert stopped_clock["speed"] == 1.0


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


def test_web_consoles_support_switching_between_independent_models():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web"
    for role in ("simulator", "trainee"):
        html = (web_root / role / "index.html").read_text(encoding="utf-8")
        js = (web_root / role / "app.js").read_text(encoding="utf-8")
        css = (web_root / role / "styles.css").read_text(encoding="utf-8")

        assert 'id="modelSelector"' in html
        assert 'id="activeModelName"' in html
        assert "显示模型" in html
        assert "function modelScopedPath" in js
        assert "function loadModels" in js
        assert "function setActiveModel" in js
        assert "activeModelId" in js
        assert 'api("/api/models", { modelScoped: false })' in js
        assert "model_id" in js
        assert "model-switcher" in css


def test_simulator_clock_buttons_are_actionable_controls():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")

    clock_buttons = re.findall(r"<button\b[^>]*data-clock=\"([^\"]+)\"[^>]*>", html)
    assert clock_buttons == ["start", "pause", "stop", "step", "slower", "faster"]
    for button_html in re.findall(r"<button\b[^>]*data-clock=\"[^\"]+\"[^>]*>", html):
        assert 'type="button"' in button_html
    for action in ("start", "pause", "stop"):
        pattern = rf"<button\b(?=[^>]*data-clock=\"{action}\")(?=[^>]*aria-pressed=\")[^>]*>"
        assert re.search(pattern, html)

    assert "function renderClock" in js
    assert "function controlClock" in js
    assert 'api("/api/clock", { method: "POST", body: JSON.stringify({ action }) })' in js
    assert 'document.querySelectorAll("[data-clock]")' in js
    assert "button.addEventListener(\"click\", () => controlClock(button.dataset.clock))" in js
    assert "button.disabled = isBusy" in js
    assert "button.setAttribute(\"aria-pressed\", isActive ? \"true\" : \"false\")" in js


def test_simulator_can_clone_current_model_from_header():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'id="cloneModelButton"' in html
    assert 'id="cloneModelDialog"' in html
    assert 'id="cloneModelName"' in html
    assert "复制新模型" in html
    assert "function openCloneModelDialog" in js
    assert "function cloneCurrentModel" in js
    assert "function normalizeModels" in js
    assert "function isModelNameTaken" in js
    assert "function uniqueCloneName" in js
    assert 'api("/api/models/clone"' in js
    assert "setActiveModel(newModelId" in js
    assert "model-toolbar" in css
    assert "modal-backdrop" in css


def test_simulator_curve_page_uses_dense_curve_editor_and_hourly_table():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-page="curves"' in html
    assert 'id="curveTree"' in html
    assert 'id="curveTreeSummary"' in html
    assert 'class="curve-page-layout"' in html
    assert 'class="curve-workspace"' in html
    assert 'id="curveEditorChart"' in html
    assert 'id="hourlyCurveTable"' in html
    assert 'id="saveCurves"' in html
    assert 'data-curve-mode="year"' in html
    assert 'data-curve-mode="day"' in html
    assert 'id="curveTableTitle"' in html
    assert 'id="curveTableSummary"' in html
    assert 'data-curve-role="dense-chart"' in html
    assert 'data-curve-role="hourly-table"' in html
    assert "环境曲线" in html
    assert "负荷曲线" in html
    assert "风" in html
    assert "光" in html
    assert "温" in html
    assert "8760点" in html
    assert "const CURVE_MODES" in js
    assert "pointCount: 8760" in js
    assert "stepMinutes: 60" in js
    assert "pointCount: 1440" in js
    assert "stepMinutes: 1" in js
    assert "function curveModeConfig" in js
    assert "function setCurveMode" in js
    assert "function renderCurveModeControls" in js
    assert "function updateCurveModeLabels" in js
    assert "function formatCurveTableTime" in js
    assert "function curveLoadDevices" in js
    assert "function renderCurveTree" in js
    assert "function setActiveCurve" in js
    assert "function selectedCurveKeys" in js
    assert "function visibleCurveKeys" in js
    assert "function setSelectedCurves" in js
    assert "function toggleCurveSelection" in js
    assert "function selectCurveFamily" in js
    assert "function curveEditKey" in js
    assert "function setCurveEditKey" in js
    assert "function cancelCurveEditSelection" in js
    assert "function curveKeyAtPointer" in js
    assert "function applyCurveDrag" in js
    assert "function renderHourlyTable" in js
    assert "function resizeCurveCanvas" in js
    assert "data-curve-tree-type" in js
    assert "data-curve-family" in js
    assert "aria-pressed" in js
    assert "data-curve-key" in js
    assert "data-index" in js
    assert "data-curve-mode" in js
    assert "selectedCurveKeys" in js
    assert "curveEditKey" in js
    assert 'event.button === 2' in js
    assert 'canvas.addEventListener("contextmenu"' in js
    assert "load:" in js
    assert "contenteditable" in js
    assert "saveCurves" in js
    assert "curve-tree-panel" in css
    assert "curve-workspace" in css
    assert "curve-chart-panel" in css


def test_simulator_fault_page_uses_device_and_measurement_subtabs():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-page="faults"' in html
    assert 'data-fault-tab="devices"' in html
    assert 'data-fault-tab="measurements"' in html
    assert 'class="fault-split-layout"' in html
    assert 'id="faultDeviceTree"' in html
    assert 'id="faultDeviceTreeSummary"' in html
    assert 'id="faultMeasurementTree"' in html
    assert 'id="faultMeasurementTreeSummary"' in html
    assert 'aria-label="量测故障设备树"' in html
    assert 'id="deviceFaultTable"' in html
    assert 'id="measurementFaultTable"' in html
    assert 'id="saveDeviceFaults"' in html
    assert 'id="saveMeasurementFaults"' in html
    assert "设备树" in html
    assert "全部设备" in html
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
    assert "function renderFaultDeviceTree" in js
    assert "function renderFaultMeasurementTree" in js
    assert "function setDeviceFaultFilter" in js
    assert "function setMeasurementFaultFilter" in js
    assert "function filteredFaultDevices" in js
    assert "function filteredFaultMeasurements" in js
    assert "function faultMeasurementDevices" in js
    assert "function renderDeviceFaultTable" in js
    assert "function renderMeasurementFaultTable" in js
    assert "data-fault-device-tree-type" in js
    assert "data-fault-measurement-tree-type" in js
    assert "data-fault-measurement-tree-name" in js
    assert "data-device-field" in js
    assert "data-meas-field" in js
    assert "fault-split-layout" in css
    assert "fault-tree-panel" in css
    assert "fault-table-panel" in css


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


def test_simulator_device_trees_support_collapsible_type_groups():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-nav-page="runtime">控制指令</button>' in html
    assert 'aria-label="控制指令设备树"' in html
    assert "计算监视" not in html
    assert "collapsedDeviceTreeGroups" in js
    assert "function isDeviceTreeGroupCollapsed" in js
    assert "function toggleDeviceTreeGroup" in js
    assert "function deviceTreeTypeAttrs" in js
    assert "function deviceTreeTypeLabel" in js
    assert "function deviceTreeChildren" in js
    assert 'data-tree-toggle-scope="${escapeHtml(scope)}"' in js
    for scope in ("model", "faultDevice", "faultMeasurement", "mode", "runtime", "measurement"):
        assert f'isDeviceTreeGroupCollapsed("{scope}", devType)' in js
        assert f'deviceTreeTypeAttrs("{scope}", devType, isCollapsed)' in js
    assert "dataset.treeToggleScope" in js
    assert "dataset.treeToggleGroup" in js
    assert "aria-expanded" in js
    assert "tree-title" in css
    assert "tree-toggle" in css
    assert ".tree-type.is-collapsed .tree-toggle" in css


def test_simulator_runtime_page_uses_device_tree_table_and_trace_chart():
    web_root = ROOT_DIR / "src" / "hybrid_power_system_analysis" / "polar_microgrid_sim" / "web" / "simulator"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    js = (web_root / "app.js").read_text(encoding="utf-8")
    css = (web_root / "styles.css").read_text(encoding="utf-8")

    assert 'data-page="runtime"' in html
    assert 'class="runtime-page-layout"' in html
    assert 'id="runtimeDeviceTree"' in html
    assert 'id="runtimeTreeSummary"' in html
    assert 'id="deviceTable"' in html
    assert 'id="runtimeDeviceSummary"' in html
    assert 'id="runtimeTraceWindow"' in html
    assert 'id="runtimeTraceChart"' in html
    assert 'aria-label="控制指令设备树"' in html
    assert '<option value="120">2小时</option>' in html
    assert 'data-runtime-role="device-table"' in html
    assert 'data-runtime-role="trace-chart"' in html
    assert "控制指令" in html
    assert "实时值" in html
    assert "量测值" in html
    assert "时间窗口" in html
    assert "function renderRuntimeDeviceTree" in js
    assert "function setRuntimeDeviceFilter" in js
    assert "function renderRuntimeDeviceTable" in js
    assert "function appendRuntimeTrace" in js
    assert "function drawRuntimeTraceChart" in js
    assert "function runtimeTraceWindowPoints" in js
    assert "function runtimeTraceWindowRange" in js
    assert "function runtimeTraceAxisTicks" in js
    assert "function runtimeAxisTickLabel" in js
    assert "function traceAxisStepMinutes" in js
    assert "function traceWindowAlignmentMinutes" in js
    assert "function alignedTraceWindowRange" in js
    assert "axisStepMinutes" in js
    assert "alignmentMinutes" in js
    assert "Math.floor(latestMinute / alignmentMinutes)" in js
    assert "runtimeFormatWindowSpan" not in js
    assert "xForMinute" in js
    assert "data-runtime-tree-type" in js
    assert "data-runtime-tree-name" in js
    assert "runtimeTraceHistory" in js
    assert "runtime-page-layout" in css
    assert "runtime-tree-panel" in css
    assert "runtime-workspace" in css
    assert "runtime-device-wrap" in css
    assert "runtime-trace-legend" in css


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
    assert 'class="measurement-page-layout"' in html
    assert 'class="measurement-workspace"' in html
    assert 'id="measurementCompareDeviceTree"' in html
    assert 'id="measurementCompareTreeSummary"' in html
    assert 'id="measurementTraceWindow"' in html
    assert 'id="measurementTraceChart"' in html
    assert '<option value="120">2小时</option>' in html
    assert 'data-measurement-role="trace-chart"' in html
    assert "运行日志" in html
    assert "量测值与真值" in html
    assert "测点跟踪曲线" in html
    assert "时间窗口" in html
    assert "设备树" in html
    assert "全部设备" in html
    assert "真值" in html
    assert "量测值" in html
    assert "偏差" in html
    assert "function appendRuntimeLog" in js
    assert "function renderRuntimeLogs" in js
    assert "function measurementCompareRows" in js
    assert "function measurementCompareDevices" in js
    assert "function filteredMeasurementCompareRows" in js
    assert "function renderMeasurementCompareDeviceTree" in js
    assert "function setMeasurementCompareFilter" in js
    assert "function appendMeasurementTrace" in js
    assert "function measurementTraceWindowRange" in js
    assert "function measurementTraceWindowPoints" in js
    assert "function drawMeasurementTraceChart" in js
    assert "alignedTraceWindowRange(history, windowMinutes, fallbackMinute)" in js
    assert "function initMeasurementMonitor" in js
    assert "function renderMeasurementCompareTable" in js
    assert "data-measurement-select-key" in js
    assert "selectedMeasurementKey" in js
    assert "data-measurement-tree-type" in js
    assert "data-measurement-tree-name" in js
    assert "runtime-log-table" in js
    assert "measurement-compare-table" in js
    assert "runtime-log-wrap" in css
    assert "measurement-compare-wrap" in css
    assert "measurement-page-layout" in css
    assert "measurement-workspace" in css
    assert "measurement-tree-panel" in css
    assert "measurement-table-panel" in css
    assert "measurement-trace-panel" in css
    assert "measurement-trace-legend" in css
