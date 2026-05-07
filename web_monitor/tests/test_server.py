import json
import shutil
import sys
import unittest
from pathlib import Path


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


class WebMonitorServerTest(unittest.TestCase):
    def setUp(self):
        self._original_data_source = server.DATA_SOURCE
        server.DATA_SOURCE = server.CsvDataSource()

    def tearDown(self):
        server.DATA_SOURCE = self._original_data_source

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

    def test_static_path_resolves_index(self):
        resolved = server.resolve_static_path("/")

        self.assertEqual(resolved.name, "index.html")
        self.assertTrue(resolved.exists())

    def test_static_path_rejects_directory_traversal(self):
        with self.assertRaises(ValueError):
            server.resolve_static_path("/../README.md")


if __name__ == "__main__":
    unittest.main()
