import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))

import server


class FakeCursor:
    def __init__(self, rows_by_sql):
        self.rows_by_sql = rows_by_sql
        self.last_sql = ""
        self.last_params = ()
        self.executed = []

    def execute(self, sql, params=()):
        self.last_sql = " ".join(sql.split())
        self.last_params = params
        self.executed.append((self.last_sql, params))

    def fetchall(self):
        return self.rows_by_sql.get(self.last_sql, [])

    def fetchone(self):
        rows = self.fetchall()
        if isinstance(rows, dict):
            return rows
        return rows[0] if rows else None

    def close(self):
        pass


class FakeConnection:
    def __init__(self, rows_by_sql):
        self.cursor_obj = FakeCursor(rows_by_sql)
        self.committed = False

    def cursor(self, *args, **kwargs):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        pass


class PowerPlanServerTest(unittest.TestCase):
    def setUp(self):
        self._original_data_source = server.DATA_SOURCE
        self._original_simu_runtime = server.SIMU_RUNTIME
        server.DATA_SOURCE = server.CsvDataSource()
        server.SIMU_RUNTIME = server.SimuRuntime()

    def tearDown(self):
        server.DATA_SOURCE = self._original_data_source
        server.SIMU_RUNTIME = self._original_simu_runtime

    def test_api_payload_contains_all_monitor_sections(self):
        payload = server.build_snapshot()

        self.assertEqual(payload["system"], "南极秦岭站综合能量管理系统")
        self.assertIn("simu", payload)
        self.assertIn("scada", payload)
        self.assertIn("agc", payload)
        self.assertIn("summary", payload)

    def test_api_response_is_json_for_known_endpoint(self):
        status, headers, body = server.handle_api_path("/api/scada")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["section"], "SCADA在线监视")
        self.assertIn("metrics", data)
        self.assertIn("alarms", data)
        self.assertIn("charts", data)
        self.assertIn("summary", data)

    def test_monitor_pages_include_dynamic_rendering_blocks(self):
        payload = server.build_snapshot()

        for key in ("simu", "scada", "agc"):
            section = payload[key]
            self.assertGreaterEqual(len(section["metrics"]), 4)
            self.assertGreaterEqual(len(section["alarms"]), 4)
            self.assertIsInstance(section["summary"], list)
            self.assertGreaterEqual(len(section["summary"]), 4)

        self.assertIn("bars", payload["simu"]["charts"])
        self.assertIn("daily", payload["simu"]["charts"])
        self.assertIn("state", payload["simu"])
        self.assertIn("topology", payload["simu"])
        self.assertIn("columns", payload["scada"]["charts"])
        self.assertIn("stations", payload["scada"])
        self.assertIn("units", payload["agc"])
        self.assertIn("reserve", payload["agc"])

    def test_simu_page_has_four_daily_curves_and_state(self):
        payload = server.build_snapshot(force_reload=True)
        simu = payload["simu"]

        curve_names = {curve["name"] for curve in simu["charts"]["daily"]}
        self.assertEqual(curve_names, {"风速", "温度", "太阳辐射", "负荷"})
        self.assertGreaterEqual(len(simu["charts"]["daily"][0]["points"]), 24)
        self.assertEqual(simu["state"]["status"], "STOPPED")
        self.assertIn("cursor_hour", simu["state"])

    def test_simu_control_actions_update_runtime_state(self):
        controller = server.SimuRuntime()

        controller.apply("start")
        self.assertEqual(controller.snapshot()["status"], "RUNNING")
        controller.apply("faster")
        self.assertEqual(controller.snapshot()["speed"], 2.0)
        controller.apply("slower")
        self.assertEqual(controller.snapshot()["speed"], 1.0)
        controller.apply("stop")
        self.assertEqual(controller.snapshot()["status"], "STOPPED")
        controller.apply("reset")
        state = controller.snapshot()
        self.assertEqual(state["status"], "STOPPED")
        self.assertEqual(state["cursor_hour"], 0.0)

    def test_optimization_api_start_stop_and_logs(self):
        original_runtime = server.OPTIMIZATION_RUNTIME
        server.OPTIMIZATION_RUNTIME = server.OptimizationRuntime()
        try:
            status, headers, body = server.handle_api_path("/api/optimization/status")
            initial = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            self.assertEqual(initial["status"], "待启动")
            self.assertIn("metrics", initial)
            self.assertIn("results", initial)
            self.assertIn("logs", initial)

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "start", "scheme": "方案A"}, ensure_ascii=False).encode("utf-8"),
            )
            started = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(started["state"]["status"], "运行中")
            self.assertEqual(started["state"]["scheme"], "方案A")
            self.assertTrue(started["state"]["start_time"])
            self.assertFalse(started["state"]["end_time"])
            self.assertTrue(any("启动优化规划" in item["message"] for item in started["state"]["logs"]))

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "start", "scheme": "方案A"}, ensure_ascii=False).encode("utf-8"),
            )
            duplicate = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 409)
            self.assertEqual(duplicate["error"], "running")
            self.assertIn("正在运行，无法再次启动", duplicate["message"])

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "stop", "scheme": "方案A"}, ensure_ascii=False).encode("utf-8"),
            )
            stopped = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(stopped["state"]["status"], "已停止")
            self.assertTrue(stopped["state"]["end_time"])
            self.assertTrue(any("停止优化规划" in item["message"] for item in stopped["state"]["logs"]))

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "stop", "scheme": "方案A"}, ensure_ascii=False).encode("utf-8"),
            )
            not_running = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 409)
            self.assertEqual(not_running["error"], "not_running")
            self.assertIn("没有运行", not_running["message"])

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "bad"}).encode("utf-8"),
            )
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(body.decode("utf-8"))["error"], "bad_request")
        finally:
            server.OPTIMIZATION_RUNTIME = original_runtime

    def test_optimization_overview_results_are_three_requested_tables(self):
        runtime = server.OptimizationRuntime()
        payload = runtime.apply("start", scheme="方案A")

        tables = payload["results"]["overview_tables"]
        self.assertEqual([table["title"] for table in tables], ["规划结果", "规划年指标", "规划年效益"])
        self.assertEqual(len(tables), 3)
        self.assertTrue(any(row["设备类型"] == "柴发" and "设计台数" in row for row in tables[0]["rows"]))
        self.assertTrue(any(row["设备类型"] == "储能" and "设计台数" in row for row in tables[0]["rows"]))
        annual_metric_names = {row["指标"] for row in tables[1]["rows"]}
        for name in (
            "柴发总容量",
            "风电总容量",
            "光伏总容量",
            "氢能总容量",
            "储能总容量",
            "负荷总电量",
            "柴发总电量",
            "风能总电量",
            "光伏总电量",
            "弃电量",
            "储能总电量",
            "制氢总量",
            "燃料电池发电量",
        ):
            self.assertIn(name, annual_metric_names)
        benefit_names = {row["指标"] for row in tables[2]["rows"]}
        for name in (
            "总成本",
            "度电成本",
            "建设成本",
            "柴油消耗量",
            "运行成本",
            "绿电占比",
            "弃电占比",
            "最高频率",
            "最低频率",
            "频率安全风险点",
        ):
            self.assertIn(name, benefit_names)

    def test_snapshot_reads_values_from_csv_files(self):
        payload = server.build_snapshot(force_reload=True)

        simu_load = next(item for item in payload["simu"]["metrics"] if item["label"] == "总有功负荷")
        self.assertEqual(simu_load["value"], 486.2)
        self.assertEqual(payload["simu"]["charts"]["bars"][0]["label"], "线路 L12")
        self.assertEqual(payload["agc"]["reserve"]["score"], 97.4)

    def test_snapshot_reloads_after_configured_interval(self):
        data_dir = WEB_ROOT / "tests" / "tmp_data_source"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True)
        try:
            (data_dir / "summary.csv").write_text("key,value,unit\nrunning_days,1,天\n", encoding="utf-8")
            (data_dir / "metrics.csv").write_text(
                "page,label,value,unit,status\nsimu,总有功负荷,100,MW,normal\n",
                encoding="utf-8",
            )
            (data_dir / "alarms.csv").write_text("page,time,object,message,status\n", encoding="utf-8")
            (data_dir / "simu_bars.csv").write_text("label,value,unit\n线路 L12,10,%\n", encoding="utf-8")
            (data_dir / "simu_topology.csv").write_text("id,status,value\nBUS-101,ok,1.0 p.u.\n", encoding="utf-8")
            (data_dir / "scada_columns.csv").write_text("label,value,unit\nNOW,20,%\n", encoding="utf-8")
            (data_dir / "scada_stations.csv").write_text("name,status,detail\n主站 A,normal,延迟 1 ms\n", encoding="utf-8")
            (data_dir / "agc_units.csv").write_text("name,percent,power,unit\nGEN-01,50,10,MW\n", encoding="utf-8")
            (data_dir / "agc_reserve.csv").write_text("score,up,down,response,cycle\n88,1,2,3,4\n", encoding="utf-8")
            (data_dir / "page_summary.csv").write_text("page,label,value,status\nsimu,刷新周期,1 s,normal\n", encoding="utf-8")

            reader = server.CsvDataSource(data_dir=data_dir, reload_interval=0)
            first = reader.snapshot()
            (data_dir / "metrics.csv").write_text(
                "page,label,value,unit,status\nsimu,总有功负荷,200,MW,normal\n",
                encoding="utf-8",
            )
            second = reader.snapshot()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        first_load = next(item for item in first["simu"]["metrics"] if item["label"] == "总有功负荷")
        second_load = next(item for item in second["simu"]["metrics"] if item["label"] == "总有功负荷")
        self.assertEqual(first_load["value"], 100)
        self.assertEqual(second_load["value"], 200)

    def test_mysql_data_source_builds_snapshot_from_database_rows(self):
        rows = {
            "SELECT `key`, value, unit FROM overview_summary ORDER BY display_order, id": [
                {"key": "running_days", "value": "449", "unit": "天"},
            ],
            "SELECT page, label, value, unit, status FROM metrics ORDER BY page, display_order, id": [
                {"page": "simu", "label": "总有功负荷", "value": "486.2", "unit": "MW", "status": "normal"},
            ],
            "SELECT page, time, object, message, status FROM alarms ORDER BY page, display_order, id": [],
            "SELECT page, label, value, status FROM page_summary ORDER BY page, display_order, id": [
                {"page": "simu", "label": "刷新周期", "value": "2 s", "status": "normal"},
            ],
            "SELECT label, value, unit FROM simu_bars ORDER BY display_order, id": [
                {"label": "线路 L12", "value": "72", "unit": "%"},
            ],
            "SELECT id, status, value FROM simu_topology ORDER BY display_order, id": [
                {"id": "BUS-101", "status": "ok", "value": "1.018 p.u."},
            ],
            "SELECT hour, wind_speed, temperature, solar_irradiance, load_value FROM simu_daily_curves ORDER BY hour": [
                {"hour": "0", "wind_speed": "7.8", "temperature": "-18", "solar_irradiance": "0", "load_value": "138"},
            ],
            "SELECT label, value, unit FROM scada_columns ORDER BY display_order, id": [],
            "SELECT name, status, detail FROM scada_stations ORDER BY display_order, id": [],
            "SELECT name, percent, power, unit FROM agc_units ORDER BY display_order, id": [],
            "SELECT score, up, down, response, cycle FROM agc_reserve ORDER BY id LIMIT 1": {
                "score": "97.4", "up": "164", "down": "120", "response": "8.2", "cycle": "4",
            },
        }
        fake_connection = FakeConnection(rows)
        source = server.MySqlDataSource(connector_factory=lambda config: fake_connection, reload_interval=0)

        payload = source.snapshot()

        self.assertEqual(payload["summary"]["running_days"], 449)
        self.assertEqual(payload["simu"]["metrics"][0]["value"], 486.2)
        self.assertEqual(payload["simu"]["charts"]["daily"][0]["name"], "风速")
        self.assertEqual(payload["agc"]["reserve"]["score"], 97.4)

    def test_mysql_data_source_persists_simu_state(self):
        fake_connection = FakeConnection({})
        source = server.MySqlDataSource(connector_factory=lambda config: fake_connection, reload_interval=0)

        source.save_simu_state({"sim_time": "01:30", "speed": 2.0, "status": "RUNNING"})

        executed_sql, params = fake_connection.cursor_obj.executed[-1]
        self.assertIn("UPDATE simu_state", executed_sql)
        self.assertEqual(params, ("01:30", 2.0, "RUNNING"))
        self.assertTrue(fake_connection.committed)

    def test_unknown_api_path_returns_404_json(self):
        status, headers, body = server.handle_api_path("/api/not-found")

        self.assertEqual(status, 404)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(json.loads(body.decode("utf-8"))["error"], "not_found")

    def test_planning_api_create_read_save_copy_rename(self):
        planning_root = WEB_ROOT / "tests" / "tmp_planning_api"
        shutil.rmtree(planning_root, ignore_errors=True)
        planning_root.mkdir(parents=True)
        original_store = server.PLANNING_STORE
        server.PLANNING_STORE = server.planning_store.PlanningStore(root=planning_root)
        try:
            status, headers, body = server.handle_planning_api_path(
                "/api/planning/schemes",
                "POST",
                json.dumps({"name": "方案A"}).encode("utf-8"),
            )
            self.assertEqual(status, 200)
            created = json.loads(body.decode("utf-8"))
            self.assertEqual(created["scheme"], "方案A")
            self.assertEqual(len(created["time_series"]), 8760)
            self.assertIn("planning_parameters", created)

            created["time_series"][0]["load"] = 123.4
            created["planning_parameters"][0]["design_life_years"] = 30
            created["planning_parameters"][0]["storage_frequency_regulation_enabled"] = True
            status, headers, body = server.handle_planning_api_path(
                "/api/planning/schemes/方案A",
                "PUT",
                json.dumps(created, ensure_ascii=False).encode("utf-8"),
            )
            self.assertEqual(status, 200)

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes/方案A", "GET", b"")
            loaded = json.loads(body.decode("utf-8"))
            self.assertEqual(loaded["time_series"][0]["load"], 123.4)
            self.assertEqual(loaded["planning_parameters"][0]["design_life_years"], 30)
            self.assertTrue(loaded["planning_parameters"][0]["storage_frequency_regulation_enabled"])

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes/方案A/overview", "GET", b"")
            overview = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertNotIn("time_series", overview)
            self.assertFalse(overview["time_series_loaded"])
            self.assertEqual(overview["time_series_count"], 8760)
            self.assertIn("diesel_generators", overview)
            self.assertIn("planning_parameters", overview)
            self.assertEqual(overview["planning_parameters"][0]["design_life_years"], 30)

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes/方案A/time-series", "GET", b"")
            time_payload = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(time_payload["time_series"][0]["load"], 123.4)
            self.assertNotIn("diesel_generators", time_payload)

            status, headers, body = server.handle_planning_api_path(
                "/api/planning/schemes/copy",
                "POST",
                json.dumps({"source": "方案A", "target": "方案B"}, ensure_ascii=False).encode("utf-8"),
            )
            self.assertEqual(status, 200)

            status, headers, body = server.handle_planning_api_path(
                "/api/planning/schemes/rename",
                "POST",
                json.dumps({"source": "方案B", "target": "方案C"}, ensure_ascii=False).encode("utf-8"),
            )
            self.assertEqual(status, 200)

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes", "GET", b"")
            names = [item["name"] for item in json.loads(body.decode("utf-8"))["schemes"]]
            self.assertEqual(names, ["方案A", "方案C"])
        finally:
            server.PLANNING_STORE = original_store
            shutil.rmtree(planning_root, ignore_errors=True)

    def test_planning_api_delete_scheme(self):
        planning_root = WEB_ROOT / "tests" / "tmp_planning_delete_api"
        shutil.rmtree(planning_root, ignore_errors=True)
        planning_root.mkdir(parents=True)
        original_store = server.PLANNING_STORE
        server.PLANNING_STORE = server.planning_store.PlanningStore(root=planning_root)
        try:
            server.PLANNING_STORE.create_scheme("方案A")
            server.PLANNING_STORE.create_scheme("方案B")

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes/方案A", "DELETE", b"")

            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.decode("utf-8"))["deleted"], "方案A")
            self.assertFalse((planning_root / "方案A").exists())
            names = [item["name"] for item in server.PLANNING_STORE.list_schemes()]
            self.assertEqual(names, ["方案B"])
        finally:
            server.PLANNING_STORE = original_store
            shutil.rmtree(planning_root, ignore_errors=True)

    def test_planning_api_rejects_bad_scheme_name(self):
        status, headers, body = server.handle_planning_api_path(
            "/api/planning/schemes",
            "POST",
            json.dumps({"name": "../bad"}).encode("utf-8"),
        )

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body.decode("utf-8"))["error"], "bad_request")

    def test_planning_weather_history_endpoint_validates_year_before_current_year(self):
        status, headers, body = server.handle_planning_api_path(
            "/api/planning/weather-history",
            "POST",
            json.dumps({"latitude": 10, "longitude": 20, "year": 9999}).encode("utf-8"),
        )

        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 400)
        self.assertEqual(data["error"], "bad_request")
        self.assertIn("历史数据年必须小于当前年", data["message"])

    def test_parse_nasa_power_hourly_response_requires_8760_rows(self):
        payload = {
            "header": {"fill_value": -999},
            "properties": {
                "parameter": {
                    "WS10M": {"2025010100": 7.1},
                    "ALLSKY_SFC_SW_DWN": {"2025010100": 0},
                    "T2M": {"2025010100": -12.5},
                }
            },
        }

        with self.assertRaises(server.WeatherHistoryError):
            server.parse_nasa_power_hourly_response(payload, 2025)

    def test_planning_geocode_endpoint_fills_coordinates_from_place_name(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(url, timeout):
            self.assertIn("geocoding-api.open-meteo.com", url)
            return FakeResponse(
                {
                    "results": [
                        {
                            "name": "北京",
                            "latitude": 39.9075,
                            "longitude": 116.39723,
                            "country": "中国",
                            "admin1": "北京市",
                        }
                    ]
                }
            )

        with patch.object(server, "urlopen_with_user_agent", side_effect=fake_urlopen):
            status, headers, body = server.handle_planning_api_path(
                "/api/planning/geocode",
                "POST",
                json.dumps({"place": "北京"}, ensure_ascii=False).encode("utf-8"),
            )

        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(data["latitude"], 39.9075)
        self.assertEqual(data["longitude"], 116.39723)
        self.assertEqual(data["source"], "Open-Meteo Geocoding API")

    def test_planning_geocode_prefers_amap_when_key_is_configured(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "status": "1",
                        "geocodes": [
                            {
                                "formatted_address": "北京市",
                                "province": "北京市",
                                "city": "北京市",
                                "district": [],
                                "location": "116.407526,39.904030",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ).encode("utf-8")

        original_key = server.AMAP_WEB_SERVICE_KEY
        server.AMAP_WEB_SERVICE_KEY = "test-key"
        try:
            def fake_urlopen(url, timeout):
                self.assertIn("restapi.amap.com", url)
                self.assertIn("key=test-key", url)
                return FakeResponse()

            with patch.object(server, "urlopen_with_user_agent", side_effect=fake_urlopen):
                status, headers, body = server.handle_planning_api_path(
                    "/api/planning/geocode",
                    "POST",
                    json.dumps({"place": "北京"}, ensure_ascii=False).encode("utf-8"),
                )
        finally:
            server.AMAP_WEB_SERVICE_KEY = original_key

        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(data["latitude"], 39.90403)
        self.assertEqual(data["longitude"], 116.407526)
        self.assertEqual(data["source"], "高德地图 Web 服务地理编码 API")

    def test_planning_map_config_exposes_amap_key_when_configured(self):
        original_key = server.AMAP_WEB_SERVICE_KEY
        server.AMAP_WEB_SERVICE_KEY = "test-key"
        try:
            status, headers, body = server.handle_planning_api_path("/api/planning/map-config", "GET", b"")
        finally:
            server.AMAP_WEB_SERVICE_KEY = original_key

        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(data["amap_key"], "test-key")
        self.assertEqual(data["preferred_provider"], "amap")

    def test_planning_geocode_endpoint_falls_back_to_nominatim(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(url, timeout):
            if "geocoding-api.open-meteo.com" in url:
                return FakeResponse({"results": []})
            self.assertIn("nominatim.openstreetmap.org", url)
            return FakeResponse([{"lat": "39.9042", "lon": "116.4074", "display_name": "北京"}])

        with patch.object(server, "urlopen_with_user_agent", side_effect=fake_urlopen):
            status, headers, body = server.handle_planning_api_path(
                "/api/planning/geocode",
                "POST",
                json.dumps({"place": "北京"}, ensure_ascii=False).encode("utf-8"),
            )

        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(data["latitude"], 39.9042)
        self.assertEqual(data["longitude"], 116.4074)
        self.assertEqual(data["source"], "OpenStreetMap Nominatim")

    def test_planning_page_has_current_scheme_display(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="currentSchemeName"', html)
        self.assertIn("当前方案:", html)
        self.assertIn(".current-scheme", css)
        self.assertIn("display: flex", css)
        self.assertIn("justify-content: flex-start", css)
        self.assertIn("white-space: nowrap", css)
        self.assertIn("text-overflow: ellipsis", css)
        self.assertLess(html.index('id="currentSchemeName"'), html.index(">时序数据<"))

    def test_planning_page_uses_requested_product_and_time_series_labels(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")

        self.assertIn(">微电网风光氢储联合规划系统<", html)
        self.assertIn(">时序数据<", html)
        self.assertNotIn(">电网规划系统<", html)
        self.assertNotIn(">电网规划列表<", html)
        self.assertNotIn(">8760时序数据<", html)

    def test_optimization_page_has_requested_three_area_layout(self):
        html = (WEB_ROOT / "optimize.html").read_text(encoding="utf-8")
        planning_html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn(">微电网风光氢储联合规划系统<", html)
        self.assertIn('<a class="active" href="optimize.html">启动优化</a>', html)
        self.assertIn('<aside class="scheme-rail">', html)
        self.assertIn('<div class="scheme-list-title">方案列表</div>', html)
        self.assertIn('id="schemeList"', html)
        self.assertIn('class="optimization-panel"', html)
        self.assertIn('class="optimization-command-card"', html)
        self.assertIn('id="startOptimization"', html)
        self.assertIn('id="stopOptimization"', html)
        for label in ("当前状态", "启动时刻", "结束时刻", "度电成本", "绿电占比"):
            self.assertIn(label, html)
        for tab in ("结果概览", "绿电结果", "安全结果"):
            self.assertIn(tab, html)
        self.assertIn('id="overviewResult"', html)
        self.assertIn('id="greenResult"', html)
        self.assertIn('id="safetyResult"', html)
        self.assertIn('id="optimizationLogs"', html)
        self.assertIn('assets/optimize.js', html)
        self.assertIn('href="optimize.html">启动优化</a>', planning_html)
        self.assertIn(".optimization-panel", css)
        self.assertIn("grid-template-rows: auto 14px minmax(220px, var(--optimization-result-height, 1fr)) 14px minmax(120px, var(--optimization-log-height, 24vh))", css)

    def test_optimization_frontend_polls_status_and_binds_controls(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")

        self.assertIn("/api/planning/schemes", script)
        self.assertIn("/api/optimization/status", script)
        self.assertIn("/api/optimization/control", script)
        self.assertIn("startOptimization", script)
        self.assertIn("stopOptimization", script)
        self.assertIn("正在运行，无法再次启动", script)
        self.assertIn("没有运行", script)
        self.assertIn("alert(data.message", script)
        self.assertIn("setInterval", script)
        self.assertIn("scheduleOptimizationPolling", script)
        self.assertIn("state.pollDelay = data.status === \"运行中\" ? 1000 : 4000", script)
        self.assertIn("renderOptimizationLogs", script)
        self.assertIn("scrollTop", script)
        self.assertIn("data-result-tab", script)
        self.assertIn("结果概览", script)
        self.assertIn("绿电结果", script)
        self.assertIn("安全结果", script)

    def test_optimization_page_has_draggable_result_and_log_resize_handles(self):
        html = (WEB_ROOT / "optimize.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="optimizationResultResizeHandle"', html)
        self.assertIn('id="optimizationLogResizeHandle"', html)
        self.assertIn('role="separator"', html)
        self.assertIn('aria-label="调整规划结果高度"', html)
        self.assertIn('aria-label="调整运行日志高度"', html)
        self.assertIn('aria-orientation="horizontal"', html)
        self.assertIn("bindOptimizationResultResizeHandle", script)
        self.assertIn("bindOptimizationLogResizeHandle", script)
        self.assertIn("optimizationResultHeight", script)
        self.assertIn("optimizationLogHeight", script)
        self.assertIn("--optimization-result-height", script)
        self.assertIn("--optimization-log-height", script)
        self.assertIn("pointerdown", script)
        self.assertIn("setPointerCapture", script)
        self.assertIn("ArrowUp", script)
        self.assertIn("ArrowDown", script)
        self.assertIn(".optimization-result-resize-handle", css)
        self.assertIn(".optimization-log-resize-handle", css)
        self.assertIn("cursor: row-resize", css)
        self.assertIn("--optimization-result-height", css)
        self.assertIn("--optimization-log-height", css)

    def test_optimization_overview_frontend_renders_three_parallel_tables(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("renderOverviewTables", script)
        self.assertIn("overview_tables", script)
        self.assertIn("optimization-overview-grid", script)
        for title in ("规划结果", "规划年指标", "规划年效益"):
            self.assertIn(title, script)
        for field in ("设备类型", "设计台数", "指标", "数值", "单位"):
            self.assertIn(field, script)
        self.assertIn(".optimization-overview-grid", css)
        self.assertIn("grid-template-columns: repeat(3, minmax(260px, 1fr))", css)
        self.assertIn(".overview-table-card", css)

    def test_planning_scheme_rail_only_shows_scheme_list_title(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")
        rail = html.split('<aside class="scheme-rail">', 1)[1].split("</aside>", 1)[0]

        self.assertNotIn("方案管理", rail)
        self.assertIn("方案列表", rail)
        self.assertIn('id="schemeList"', rail)
        self.assertIn(".scheme-list-title", css)
        self.assertIn("color: #102b2a", css)
        self.assertIn("font-size: 18px", css)
        self.assertIn("font-weight: 900", css)

    def test_planning_page_save_button_is_in_scheme_actions(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        current_scheme_panel = html.split('<div class="current-scheme-panel">', 1)[1].split('<div class="tabs"', 1)[0]
        editor_header = html.split('<div class="editor-header">', 1)[1].split("</div>\n\n        <section", 1)[0]
        topbar = html.split('<header class="topbar">', 1)[1].split("</header>", 1)[0]
        rail = html.split('<aside class="scheme-rail">', 1)[1].split("</aside>", 1)[0]

        self.assertNotIn('id="saveScheme"', current_scheme_panel)
        self.assertIn('class="scheme-actions"', editor_header)
        self.assertIn("margin-left: auto", (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8"))
        self.assertIn('id="saveScheme"', editor_header)
        self.assertIn('id="renameScheme"', editor_header)
        self.assertIn('id="copyScheme"', editor_header)
        self.assertIn('id="deleteScheme"', editor_header)
        self.assertIn(">修改名称<", editor_header)
        self.assertNotIn("修改方案名称", editor_header)
        self.assertNotIn("修改方案名", editor_header)
        self.assertNotIn('id="saveScheme"', topbar)
        self.assertNotIn('id="saveScheme"', rail)
        self.assertLess(html.index('id="currentSchemeName"'), html.index(">时序数据<"))
        self.assertLess(html.index(">时序数据<"), html.index('id="saveScheme"'))

    def test_planning_scheme_actions_are_horizontal(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("flex-wrap: nowrap", css)
        self.assertIn("overflow-x: auto", css)
        self.assertIn("white-space: nowrap", css)

    def test_planning_page_has_device_filter_tags(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="deviceFilters"', html)
        self.assertIn("renderDeviceFilters", script)
        self.assertIn("visibleDevices", script)
        self.assertIn("deviceGroups", script)
        self.assertIn("data-device-group", script)
        self.assertNotIn("<h2>设备类型显示</h2>", html)
        self.assertNotIn("默认全部显示，取消勾选则隐藏对应表格。", html)
        self.assertIn('class="device-filter-row"', html)
        device_filter_card = html.split('<div class="device-filter-card">', 1)[1].split('<div id="deviceTables"', 1)[0]
        self.assertIn('id="deviceFilters"', device_filter_card)
        self.assertIn('id="deviceJump"', device_filter_card)
        self.assertLess(device_filter_card.index('id="deviceFilters"'), device_filter_card.index('id="deviceJump"'))
        self.assertIn(".device-filter-row", css)
        self.assertIn("flex-wrap: nowrap", css)
        self.assertIn("overflow-x: auto", css)
        self.assertIn("justify-content: flex-end", css)
        self.assertIn("margin-left: auto", css)
        self.assertIn("min-width: 0", css)
        for group_name in ("风光柴", "氢储能", "电储能"):
            self.assertIn(group_name, script)
        self.assertLess(script.index('"电储能"'), script.index('"氢储能"'))
        self.assertLess(script.index('"储能PCS"'), script.index('"电制氢"'))
        for name in ("柴发", "风机", "光伏", "储能PCS", "储能电池组", "电制氢", "储氢罐", "燃料电池"):
            self.assertIn(name, script)

    def test_planning_page_has_planning_parameters_tab_and_summary_panel(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('data-tab="planning"', html)
        self.assertIn('id="planningTab"', html)
        self.assertIn('id="planningParametersTable"', html)
        self.assertNotIn("参数随当前方案保存到 XLSX 文件。", html)
        self.assertLess(html.index('data-tab="devices"'), html.index('data-tab="planning"'))
        self.assertLess(html.index('data-tab="planning"'), html.index('data-tab="limits"'))
        self.assertIn('data-summary-tab="planning"', html)
        self.assertIn('data-summary-panel="planning"', html)
        self.assertIn('id="planningSummary"', html)
        self.assertIn("planningParameterSpecs", script)
        self.assertIn("renderPlanningParameters", script)
        self.assertIn("renderPlanningParameterSummaryTable", script)
        self.assertIn("collectPlanningParameterWarnings", script)
        self.assertIn("planning_parameters", script)
        self.assertIn(".planning-parameters-card", css)
        self.assertIn("#planningTab #planningParametersTable", css)
        for label in (
            "设计使用年限(年)",
            "柴油价格(万元/吨)",
            "规划负荷系数(0.1-10.0)",
            "绿电电量占比下限(0.0-1.0)",
            "储能是否参与调频",
            "负荷扰动系数(0.0-0.5)",
            "是否考虑频率安全约束",
            "频率安全上限(1.0-1.5)",
            "频率安全下限(1.0-1.5)",
            "是否考虑扰动后功率平衡",
            "是否考虑新能源N-1",
            "是否考虑负荷扰动",
        ):
            self.assertIn(label, script)

    def test_planning_boolean_parameters_use_yes_no_selects(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("planning-bool-select", script)
        self.assertIn('<option value="true"', script)
        self.assertIn('<option value="false"', script)
        self.assertIn(">是</option>", script)
        self.assertIn(">否</option>", script)
        self.assertIn('input.type === "checkbox"', script)
        self.assertIn('input.tagName === "SELECT"', script)
        self.assertNotIn('type="checkbox" data-planning-key', script)
        self.assertIn(".planning-bool-select", css)

    def test_planning_save_has_parameter_alarm_validation(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertIn("collectSaveWarnings", script)
        self.assertIn("参数校验未通过", script)
        self.assertIn("数据下限(台)", script)
        self.assertIn("数据上限(台)", script)
        self.assertIn("数据上限不能小于数据下限", script)
        self.assertIn("频率安全上限不能小于频率安全下限", script)
        self.assertIn("规划负荷系数(0.1-10.0)", script)
        self.assertNotIn("设计容量上限不能小于下限", script)

    def test_planning_scheme_actions_validate_duplicates_and_delete_current_scheme(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertIn("schemeNameExists", script)
        self.assertIn("normalizeSchemeName", script)
        self.assertIn("\\s\\u0000-\\u001f", script)
        self.assertIn("方案名称已存在", script)
        self.assertIn("deleteScheme", script)
        self.assertIn("DELETE", script)
        self.assertIn("确认删除方案", script)
        self.assertIn("selectNextSchemeAfterDelete", script)

    def test_planning_overview_page_has_statistics_histograms_and_candidate_device_list(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertIn("方案概览", html)
        self.assertNotIn("方案汇总", html)
        self.assertNotIn('id="schemeOverview"', html)
        self.assertNotIn("overviewHost", script)
        self.assertNotIn('id="summaryStats"', html)
        self.assertNotIn("时序统计量", html)
        self.assertIn('id="summaryCharts"', html)
        self.assertIn('id="quantitySummary"', html)
        self.assertIn("待选设备列表", html)
        self.assertIn('class="summary-tabs"', html)
        self.assertIn('data-summary-tab="charts"', html)
        self.assertIn('data-summary-tab="devices"', html)
        self.assertIn('data-summary-tab="planning"', html)
        self.assertIn('data-summary-panel="charts"', html)
        self.assertIn('data-summary-panel="devices"', html)
        self.assertIn('data-summary-panel="planning"', html)
        self.assertNotIn("设计容量约束", html)
        self.assertIn("bindSummaryTabs", script)
        self.assertIn("data-summary-panel", script)
        self.assertIn("renderSchemeSummary", script)
        self.assertNotIn("renderStatsTable", script)
        self.assertIn("renderCandidateDeviceTable", script)
        self.assertIn("capacityValue", script)
        self.assertIn("calculateSeriesStats", script)
        self.assertIn("buildHistogram", script)
        for name in ("风速", "太阳辐照", "温度", "负荷", "最大值", "最小值", "平均值", "数据下限(台)", "数据上限(台)"):
            self.assertIn(name, script)
        self.assertNotIn("formatFixed2", script)
        self.assertIn("formatInteger", script)
        self.assertNotIn(">频数</text>", script)
        self.assertIn("yAxis", script)
        self.assertIn("formatInteger(count)", script)
        self.assertIn("formatInteger(bin.count)", script)
        self.assertIn("histogram-bar", script)
        self.assertIn("data-bin-range", script)
        self.assertIn("data-bin-count", script)
        self.assertIn("onHistogramMouseMove", script)
        self.assertIn("横坐标", script)
        self.assertIn("纵坐标", script)

    def test_planning_overview_page_scrolls_when_content_overflows(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn(".summary-page", css)
        self.assertIn(".summary-switcher", css)
        self.assertIn(".summary-tab-panel.active", css)
        self.assertIn("flex: 1 1 auto", css)
        self.assertIn("overflow: auto", css)

    def test_planning_overview_charts_and_tables_adapt_to_available_height(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        summary_layout_script = script.split("function applyAdaptiveSummaryLayout()", 1)[1].split("function bindTimeResizeHandle()", 1)[0]

        self.assertIn("--summary-panel-height", css)
        self.assertIn("--summary-table-height", css)
        self.assertIn("--summary-histogram-grid-height", css)
        self.assertIn("--summary-histogram-svg-height", css)
        self.assertIn("height: var(--summary-table-height", css)
        self.assertIn("max-height: none", css)
        self.assertNotIn("min(50vh, 560px)", css)
        self.assertIn("height: var(--summary-histogram-grid-height", css)
        self.assertNotIn("min(52vh, 620px)", css)
        self.assertIn("grid-template-rows: repeat(2, minmax(0, 1fr))", css)
        self.assertIn("flex-direction: column", css)
        self.assertIn("applyAdaptiveSummaryLayout", script)
        self.assertIn("summaryTabs", script)
        self.assertIn("summary-histogram-grid-height", script)
        self.assertIn("summary-table-height", script)
        self.assertNotIn("Math.min(560", summary_layout_script)
        self.assertNotIn("Math.min(620", summary_layout_script)

    def test_planning_overview_table_rows_wrap_and_use_adaptive_height(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("#quantitySummary table,", css)
        self.assertIn("#planningSummary table", css)
        self.assertIn("table-layout: fixed", css)
        self.assertIn("#quantitySummary th,", css)
        self.assertIn("#quantitySummary td,", css)
        self.assertIn("#planningSummary th,", css)
        self.assertIn("#planningSummary td", css)
        self.assertNotIn("#limitsTab .data-table table", css)
        self.assertIn("white-space: normal", css)
        self.assertIn("overflow-wrap: anywhere", css)
        self.assertIn("line-height: 1.45", css)
        self.assertIn("min-height: 48px", css)
        self.assertIn("vertical-align: top", css)

    def test_planning_layout_constrains_page_height_to_viewport(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("height: 100vh", css)
        self.assertIn("overflow: hidden", css)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr)", css)
        self.assertIn("#timeTab #timeTable", css)
        self.assertIn("height: var(--time-chart-height, clamp(180px, 28vh, 300px))", css)

    def test_planning_layout_adapts_chart_and_table_heights_to_viewport(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertIn("--panel-table-max-height", css)
        self.assertIn("--time-table-height", css)
        self.assertIn("height: var(--time-table-height", css)
        self.assertIn("max-height: var(--panel-table-max-height", css)
        self.assertIn("#timeTab.tab-panel.active", css)
        self.assertIn("syncAdaptiveLayout", script)
        self.assertIn("applyAdaptiveTimeSeriesLayout", script)
        self.assertIn("ResizeObserver", script)
        self.assertIn("timeChartManualHeight", script)
        self.assertIn("Math.round(tableHeight)", script)

    def test_planning_frontend_defers_time_series_loading(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertIn("/overview", script)
        self.assertIn("/time-series", script)
        self.assertIn("ensureTimeSeriesLoaded", script)
        self.assertIn("ensureTimeSeriesForActiveTab", script)
        self.assertIn("shouldAutoLoadTimeSeries", script)
        self.assertIn("timeSeriesLoaded", script)
        self.assertIn("时序数据尚未加载", script)
        self.assertIn("进入时序数据或方案概览", script)
        self.assertIn("自动加载", script)
        self.assertNotIn("8760时序数据", script)
        self.assertNotIn("data-load-time-series", script)
        self.assertNotIn("点击加载", script)

    def test_planning_time_series_page_includes_temperature(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertNotIn("8760点曲线板", html)
        for curve_key, label in (
            ("wind_speed", "风速"),
            ("solar_irradiance", "太阳辐照"),
            ("temperature", "温度"),
            ("load", "负荷"),
        ):
            self.assertIn(f'<button type="button" data-curve="{curve_key}"', html)
            self.assertIn(f">{label}</button>", html)
        self.assertNotIn('type="radio"', html)
        self.assertNotIn('name="timeCurve"', html)
        self.assertNotIn('type="checkbox" data-curve', html)
        self.assertIn('class="curve-switch-row"', html)
        self.assertIn('class="curve-button active"', html)
        self.assertIn('aria-pressed="true"', html)
        self.assertLess(html.index('class="weather-import-bar"'), html.index('class="curve-switch-row"'))
        self.assertIn("temperature", script)
        self.assertIn("温度", script)

    def test_planning_time_series_page_can_fetch_geocoded_weather_history(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        for element_id in (
            "weatherPlace",
            "geocodePlace",
            "openCoordinatePicker",
            "weatherLatitude",
            "weatherLongitude",
            "weatherYear",
            "fetchWeatherHistory",
            "weatherImportStatus",
            "mapPickerModal",
            "mapPickerCanvas",
            "closeMapPicker",
            "confirmMapPoint",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('id="weatherYear" type="number" min="2001" step="1" value="2024"', html)
        weather_bar = html.split('<div class="weather-import-bar"', 1)[1].split("</div>", 1)[0]
        modal = html.split('<div id="mapPickerModal"', 1)[1].split('<div id="timeResizeHandle"', 1)[0]
        self.assertIn(">坐标选择<", weather_bar)
        self.assertNotIn('id="weatherPlace"', weather_bar)
        self.assertNotIn('id="geocodePlace"', weather_bar)
        self.assertNotIn(">地图选点</button>", weather_bar)
        self.assertIn('id="weatherPlace"', modal)
        self.assertIn('id="geocodePlace"', modal)
        self.assertIn("根据地名查找坐标", modal)
        self.assertIn("/api/planning/map-config", script)
        self.assertIn("/api/planning/geocode", script)
        self.assertIn("/api/planning/weather-history", script)
        self.assertIn("openCoordinatePicker", script)
        self.assertIn("loadAmapScript", script)
        self.assertIn("initAmapPicker", script)
        self.assertIn("setMapPoint", script)
        self.assertIn("未配置地图 Key", script)
        self.assertIn("geocodePlace", script)
        self.assertIn("fetchWeatherHistory", script)
        self.assertIn("validateWeatherInputs", script)
        self.assertIn("历史数据年必须", script)
        self.assertIn("rows.length !== 8760", script)
        self.assertIn("未更新数据", script)
        self.assertIn("wind_speed: weather.wind_speed", script)
        self.assertIn("solar_irradiance: weather.solar_irradiance", script)
        self.assertIn("temperature: weather.temperature", script)
        self.assertNotIn("load: weather.load", script)
        self.assertIn("气象已更新", script)
        self.assertNotIn("风速、太阳辐照和温度数据", script)
        self.assertIn(".weather-import-bar", css)
        self.assertIn(".coordinate-search-row", css)
        self.assertIn(".weather-import-status.error", css)
        self.assertIn(".map-picker-modal", css)
        self.assertIn(".map-picker-canvas", css)

    def test_planning_time_series_table_uses_month_tabs(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="monthTabs"', html)
        self.assertIn('class="time-table-toolbar"', html)
        self.assertNotIn('id="prevPage"', html)
        self.assertNotIn('id="nextPage"', html)
        self.assertNotIn("<h2>小时级数据</h2>", html)
        toolbar = html.split('<div class="time-table-toolbar">', 1)[1].split('<div id="timeTable"', 1)[0]
        self.assertLess(toolbar.index('id="monthTabs"'), toolbar.index('id="pageInfo"'))
        self.assertIn(".time-table-toolbar", css)
        self.assertIn("flex-wrap: nowrap", css)
        self.assertIn("justify-content: flex-start", css)
        self.assertIn("margin-left: auto", css)
        self.assertIn("text-align: right", css)
        self.assertIn("monthRanges", script)
        self.assertIn("renderMonthTabs", script)
        self.assertIn("1月", script)
        self.assertIn("12月", script)

    def test_planning_time_series_chart_height_has_resizable_splitter(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="timeResizeHandle"', html)
        self.assertIn('role="separator"', html)
        self.assertIn("调整时序图高度", html)
        self.assertLess(html.index('id="timeChart"'), html.index('id="timeResizeHandle"'))
        self.assertLess(html.index('id="timeResizeHandle"'), html.index('id="timeTable"'))
        self.assertIn("bindTimeResizeHandle", script)
        self.assertIn("pointerdown", script)
        self.assertIn("--time-chart-height", script)
        self.assertIn("svg.clientHeight", script)
        self.assertIn(".time-resize-handle", css)
        self.assertIn("cursor: row-resize", css)
        self.assertIn("height: var(--time-chart-height", css)

    def test_planning_time_series_input_refreshes_visible_chart(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("function onTimeInput", script)
        self.assertIn("renderChart();", script)
        self.assertIn("selectedCurveSpec", script)
        self.assertIn('[data-curve][aria-pressed="true"]', script)
        self.assertIn("selectCurve", script)
        self.assertIn("时间（月）", script)
        self.assertIn("monthRanges", script)
        self.assertIn("yTicks", script)
        self.assertIn('stroke="${color}"', script)
        self.assertIn("onChartMouseMove", script)
        self.assertIn("hideChartCursor", script)
        self.assertIn('id="chartCursor"', script)
        self.assertIn('id="chartCursorLine"', script)
        self.assertIn('id="chartCursorPoint"', script)
        self.assertIn("mousemove", script)
        self.assertIn("mouseleave", script)
        self.assertIn(".chart-cursor", css)

    def test_planning_device_fields_follow_latest_parameter_names(self):
        script = (WEB_ROOT / "assets" / "planning.js").read_text(encoding="utf-8")

        self.assertNotIn("design_capacity_lower", script)
        self.assertNotIn("design_capacity_upper", script)
        self.assertIn("generation_efficiency", script)
        self.assertIn("发电效率(0-1.0)", script)
        self.assertIn("氢-电效率(kWh/Nm3)", script)
        self.assertIn("电-氢效率(Nm3/kWh)", script)
        self.assertIn("切入风速(m/s)", script)
        self.assertIn("切出风速(m/s)", script)
        self.assertIn("成本(万元/台)", script)
        self.assertIn("油耗率(kg/kWh)", script)
        self.assertIn("功率上限(kW)", script)
        self.assertIn("功率下限(kW)", script)

    def test_static_path_resolves_index(self):
        resolved = server.resolve_static_path("/")

        self.assertEqual(resolved.name, "index.html")
        self.assertTrue(resolved.exists())

    def test_planning_assets_are_cache_busted_and_static_js_css_are_no_cache(self):
        html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")

        self.assertIn("assets/planning.css?v=", html)
        self.assertIn("assets/planning.js?v=", html)
        self.assertEqual(server.resolve_static_path("/assets/planning.js?v=test").name, "planning.js")
        self.assertIn('".css", ".js"', (WEB_ROOT / "server.py").read_text(encoding="utf-8"))

    def test_static_path_rejects_directory_traversal(self):
        with self.assertRaises(ValueError):
            server.resolve_static_path("/../README.md")


if __name__ == "__main__":
    unittest.main()
