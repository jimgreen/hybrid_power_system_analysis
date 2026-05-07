import json
import shutil
import sys
import unittest
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))

import server


class WebMonitorServerTest(unittest.TestCase):
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
        self.assertIn("topology", payload["simu"])
        self.assertIn("columns", payload["scada"]["charts"])
        self.assertIn("stations", payload["scada"])
        self.assertIn("units", payload["agc"])
        self.assertIn("reserve", payload["agc"])

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
