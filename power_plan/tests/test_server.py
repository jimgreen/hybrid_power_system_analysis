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

    def test_api_payload_excludes_removed_monitor_sections(self):
        payload = server.build_snapshot()

        self.assertEqual(payload["system"], "考察站风-光-氢-储-柴联合规划系统")
        self.assertIn("timestamp", payload)
        self.assertIn("summary", payload)
        self.assertNotIn("simu", payload)
        self.assertNotIn("scada", payload)
        self.assertNotIn("agc", payload)

    def test_index_page_has_visual_planning_entry_buttons(self):
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn('background-image: url("assets/main-dashboard-bg.png?v=20260510-title-safe")', html)
        self.assertIn("background-size: contain", html)
        self.assertIn('<link rel="icon" href="data:,">', html)
        self.assertIn(".screen::before", html)
        self.assertIn("filter: saturate(1.08) brightness(0.74) contrast(1.08)", html)
        self.assertIn(".energy-side", html)
        self.assertIn(".energy-left", html)
        self.assertIn(".energy-right", html)
        energy_side_css = html.split(".energy-side {", 1)[1].split("}", 1)[0]
        self.assertIn("top: 50%", energy_side_css)
        self.assertIn("min-height: clamp(320px, 41vh, 400px)", energy_side_css)
        self.assertIn("align-content: space-around", energy_side_css)
        self.assertIn("transform: translateY(-29%)", energy_side_css)
        self.assertIn("clip-path: polygon", html)
        self.assertIn("box-shadow:", html)
        self.assertIn("color: #21d5ff", html)
        self.assertIn('class="feature-entry-grid"', html)
        self.assertIn('aria-label="规划功能快捷入口"', html)
        self.assertEqual(html.count('class="feature-entry"'), 3)
        self.assertEqual(html.count('class="feature-icon"'), 3)
        self.assertIn('class="energy-side energy-left"', html)
        self.assertIn('class="energy-side energy-right"', html)
        self.assertIn('<strong>参数维护</strong>', html)
        self.assertIn('<strong>算法启动</strong>', html)
        self.assertIn('<strong>结果评估</strong>', html)
        self.assertNotIn("规划参数维护", html)
        self.assertNotIn("规划算法启动", html)
        self.assertNotIn("规划结果评估", html)
        self.assertIn('href="planning.html"', html)
        self.assertIn('href="optimize.html"', html)
        self.assertIn('href="optimize.html#overviewResult"', html)
        self.assertIn(".feature-entry-grid", html)
        self.assertIn("grid-template-columns: repeat(3, minmax(0, 1fr))", html)
        self.assertIn("top: 50%", html)
        self.assertIn("transform: translate(-50%, -29%)", html)
        self.assertIn(".feature-icon svg", html)
        self.assertNotIn("hot-nav", html)
        self.assertNotIn("quick-links", html)
        self.assertNotIn("系统主导航", html)
        self.assertNotIn("在线监视快捷入口", html)
        self.assertNotIn("SIMU在线监视", html)

    def test_power_plan_pages_share_dark_hud_visual_theme(self):
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")
        planning_html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        optimize_html = (WEB_ROOT / "optimize.html").read_text(encoding="utf-8")

        self.assertIn("assets/planning.css?v=20260510-dark-hud", planning_html)
        self.assertIn("assets/planning.css?v=20260510-dark-hud", optimize_html)
        self.assertIn('url("main-dashboard-bg.png?v=20260510-title-safe")', css)
        self.assertIn("--hud-cyan: #21d5ff", css)
        self.assertIn("--hud-panel:", css)
        self.assertIn("rgba(20, 190, 255, 0.64)", css)
        self.assertIn(".scheme-rail,", css)
        self.assertIn(".optimization-command-card,", css)
        self.assertIn("background: var(--hud-panel)", css)
        self.assertIn("color: var(--hud-text)", css)

    def test_monitor_static_pages_are_removed(self):
        for filename in ("simu.html", "scada.html", "agc.html"):
            self.assertFalse((WEB_ROOT / filename).exists())

    def test_auth_pages_and_topbars_include_user_controls(self):
        login_html = (WEB_ROOT / "login.html").read_text(encoding="utf-8")
        register_html = (WEB_ROOT / "register.html").read_text(encoding="utf-8")
        users_html = (WEB_ROOT / "users.html").read_text(encoding="utf-8")
        planning_html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        optimize_html = (WEB_ROOT / "optimize.html").read_text(encoding="utf-8")
        index_html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn('id="loginForm"', login_html)
        self.assertIn('id="registerForm"', register_html)
        self.assertIn('body data-admin-page="true"', users_html)
        self.assertIn('id="usersTable"', users_html)
        for html in (planning_html, optimize_html, index_html, users_html):
            self.assertIn("data-auth-user", html)
            self.assertIn("data-auth-username", html)
            self.assertIn("data-logout", html)
            self.assertIn("assets/auth.js", html)
        self.assertIn("data-admin-only", planning_html)
        self.assertIn("data-admin-only", optimize_html)
        self.assertIn("data-admin-only", index_html)
        self.assertIn(".user-status", css)
        self.assertIn(".auth-card", css)
        self.assertIn("font-size: 17px", css)

    def test_sqlite_user_store_registers_first_user_as_admin_and_authenticates(self):
        db_path = WEB_ROOT / "tests" / "tmp_users.sqlite3"
        db_path.unlink(missing_ok=True)
        try:
            store = server.UserStore(db_path)
            admin = store.create_user("adminA", "secret1")
            normal = store.create_user("userA", "secret2")

            self.assertEqual(admin["role"], "admin")
            self.assertEqual(normal["role"], "user")
            self.assertEqual(store.authenticate("adminA", "secret1")["id"], admin["id"])
            with self.assertRaises(ValueError):
                store.authenticate("adminA", "bad-password")
            token = store.create_session(admin["id"])
            self.assertEqual(store.user_for_session(token)["username"], "adminA")
            store.delete_session(token)
            self.assertIsNone(store.user_for_session(token))
        finally:
            db_path.unlink(missing_ok=True)

    def test_auth_and_user_management_api_use_sqlite_sessions(self):
        original_store = server.USER_STORE
        db_path = WEB_ROOT / "tests" / "tmp_auth_api.sqlite3"
        db_path.unlink(missing_ok=True)
        server.USER_STORE = server.UserStore(db_path)
        try:
            status, headers, body = server.handle_auth_api_path(
                "/api/auth/register",
                "POST",
                json.dumps({"username": "adminA", "password": "secret1"}).encode("utf-8"),
            )
            data = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(data["user"]["role"], "admin")
            self.assertIn("Set-Cookie", headers)
            token = headers["Set-Cookie"].split("=", 1)[1].split(";", 1)[0]

            status, headers, body = server.handle_auth_api_path("/api/auth/me", "GET", token=token)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.decode("utf-8"))["user"]["username"], "adminA")

            status, headers, body = server.handle_users_api_path("/api/users", "GET", b"", {"id": 1, "username": "adminA", "role": "admin"})
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(body.decode("utf-8"))["users"]), 1)

            status, headers, body = server.handle_users_api_path("/api/users", "GET", b"", {"id": 2, "username": "userA", "role": "user"})
            self.assertEqual(status, 403)
        finally:
            server.USER_STORE = original_store
            db_path.unlink(missing_ok=True)

    def test_api_response_is_json_for_known_endpoint(self):
        status, headers, body = server.handle_api_path("/api/overview")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["system"], "考察站风-光-氢-储-柴联合规划系统")
        self.assertIn("summary", data)
        self.assertNotIn("simu", data)
        self.assertNotIn("scada", data)
        self.assertNotIn("agc", data)

    def test_removed_monitor_api_endpoints_return_not_found(self):
        for path in ("/api/simu", "/api/scada", "/api/agc"):
            status, headers, body = server.handle_api_path(path)
            data = json.loads(body.decode("utf-8"))

            self.assertEqual(status, 404)
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            self.assertEqual(data["error"], "not_found")
            self.assertEqual(data["path"], path)

        status, headers, body = server.handle_control_path(
            "/api/simu/control",
            json.dumps({"action": "start"}).encode("utf-8"),
        )
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(status, 404)
        self.assertEqual(data["error"], "not_found")

    def test_optimization_api_start_stop_and_logs(self):
        original_runtime = server.OPTIMIZATION_RUNTIME
        server.OPTIMIZATION_RUNTIME = server.OptimizationRuntimeManager()
        try:
            status, headers, body = server.handle_api_path("/api/optimization/status?scheme=方案A")
            initial = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
            self.assertEqual(initial["status"], "待启动")
            self.assertEqual(initial["scheme"], "方案A")
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
                json.dumps({"action": "start", "scheme": "方案B"}, ensure_ascii=False).encode("utf-8"),
            )
            second_started = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(second_started["state"]["status"], "运行中")
            self.assertEqual(second_started["state"]["scheme"], "方案B")

            status, headers, body = server.handle_api_path("/api/optimization/status?scheme=方案A")
            scheme_a = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(scheme_a["status"], "运行中")
            self.assertEqual(scheme_a["scheme"], "方案A")
            self.assertIn("方案A", scheme_a["running_schemes"])
            self.assertIn("方案B", scheme_a["running_schemes"])

            status, headers, body = server.handle_api_path("/api/optimization/status?scheme=方案B")
            scheme_b = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(scheme_b["status"], "运行中")
            self.assertEqual(scheme_b["scheme"], "方案B")
            self.assertTrue(any("方案：方案A" in item["message"] for item in scheme_a["logs"]))
            self.assertFalse(any("方案：方案B" in item["message"] for item in scheme_a["logs"]))
            self.assertTrue(any("方案：方案B" in item["message"] for item in scheme_b["logs"]))

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

            status, headers, body = server.handle_api_path("/api/optimization/status?scheme=方案B")
            scheme_b_after_a_stop = json.loads(body.decode("utf-8"))
            self.assertEqual(status, 200)
            self.assertEqual(scheme_b_after_a_stop["status"], "运行中")
            self.assertEqual(scheme_b_after_a_stop["scheme"], "方案B")

            status, headers, body = server.handle_control_path(
                "/api/optimization/control",
                json.dumps({"action": "bad"}).encode("utf-8"),
            )
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(body.decode("utf-8"))["error"], "bad_request")
        finally:
            server.OPTIMIZATION_RUNTIME = original_runtime

    def test_optimization_overview_results_are_two_tables_with_ratio_disks(self):
        runtime = server.OptimizationRuntime()
        payload = runtime.apply("start", scheme="方案A")

        tables = payload["results"]["overview_tables"]
        self.assertEqual([table["title"] for table in tables], ["规划结果", "规划年指标"])
        self.assertEqual(len(tables), 2)
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
            "总成本",
            "绿电占比",
            "频率风险点",
        ):
            self.assertIn(name, annual_metric_names)
        self.assertNotIn("规划年效益", [table["title"] for table in tables])

        disks = payload["results"]["overview_disks"]
        self.assertEqual([disk["title"] for disk in disks], ["成本构成", "电量构成"])
        self.assertEqual(disks[0]["left_label"], "运行成本")
        self.assertEqual(disks[0]["right_label"], "建设成本")
        self.assertEqual(disks[1]["left_label"], "柴发电量")
        self.assertEqual(disks[1]["right_label"], "新能源电量")

    def test_optimization_green_result_has_summary_table_and_daily_curve(self):
        runtime = server.OptimizationRuntime()
        payload = runtime.apply("start", scheme="方案A")

        green_rows = payload["results"]["green_table"]
        metric_names = {row["指标"] for row in green_rows}
        for name in (
            "负荷总电量",
            "柴发总电量",
            "风机总发电量",
            "光伏总发电量",
            "电储总发电量",
            "氢储总发电量",
            "新能源总弃电量",
            "柴油消耗",
            "制氢总量",
        ):
            self.assertIn(name, metric_names)
        self.assertTrue(all(set(row) == {"指标", "数值", "单位"} for row in green_rows))
        units = {row["指标"]: row["单位"] for row in green_rows}
        self.assertEqual(units["负荷总电量"], "kWh")
        self.assertEqual(units["柴发总电量"], "kWh")
        self.assertEqual(units["风机总发电量"], "kWh")
        self.assertEqual(units["光伏总发电量"], "kWh")
        self.assertEqual(units["电储总发电量"], "kWh")
        self.assertEqual(units["氢储总发电量"], "kWh")
        self.assertEqual(units["新能源总弃电量"], "%")
        self.assertEqual(units["柴油消耗"], "吨")
        self.assertEqual(units["制氢总量"], "Nm3")

        daily = payload["results"]["curves"]["green_daily"]
        self.assertEqual(len(daily), 365)
        self.assertEqual(daily[0]["day"], 1)
        self.assertEqual(daily[-1]["day"], 365)
        for field in (
            "diesel_energy",
            "wind_energy",
            "pv_energy",
            "hydrogen_energy",
            "storage_discharge_energy",
            "load_energy",
            "hydrogen_production_energy",
            "storage_charge_energy",
        ):
            self.assertIn(field, daily[0])
            self.assertIsInstance(daily[0][field], (int, float))

    def test_optimization_safety_result_has_summary_table_and_daily_frequency_curve(self):
        runtime = server.OptimizationRuntime()
        payload = runtime.apply("start", scheme="方案A")

        safety_rows = payload["results"]["safety_table"]
        metric_names = {row["指标"] for row in safety_rows}
        for name in (
            "向上扰动最大量",
            "向下扰动最大量",
            "最高频率",
            "最低频率",
            "频率安全风险小时数",
        ):
            self.assertIn(name, metric_names)
        units = {row["指标"]: row["单位"] for row in safety_rows}
        self.assertEqual(units["向上扰动最大量"], "kW")
        self.assertEqual(units["向下扰动最大量"], "kW")
        self.assertEqual(units["最高频率"], "Hz")
        self.assertEqual(units["最低频率"], "Hz")
        self.assertEqual(units["频率安全风险小时数"], "h")

        daily = payload["results"]["curves"]["safety_daily"]
        self.assertEqual(len(daily), 365)
        self.assertEqual(daily[0]["day"], 1)
        self.assertEqual(daily[-1]["day"], 365)
        self.assertGreater(daily[0]["frequency_max"], 50)
        self.assertLess(daily[0]["frequency_min"], 50)
        for field in ("frequency_max", "frequency_min"):
            self.assertIn(field, daily[0])
            self.assertIsInstance(daily[0][field], (int, float))

    def test_snapshot_reads_summary_from_csv_files(self):
        payload = server.build_snapshot(force_reload=True)

        self.assertEqual(payload["system"], "考察站风-光-氢-储-柴联合规划系统")
        self.assertIn("summary", payload)
        self.assertNotIn("simu", payload)
        self.assertNotIn("scada", payload)
        self.assertNotIn("agc", payload)

    def test_snapshot_reloads_after_configured_interval(self):
        data_dir = WEB_ROOT / "tests" / "tmp_data_source"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True)
        try:
            (data_dir / "summary.csv").write_text("key,value,unit\nrunning_days,1,天\n", encoding="utf-8")

            reader = server.CsvDataSource(data_dir=data_dir, reload_interval=0)
            first = reader.snapshot()
            (data_dir / "summary.csv").write_text("key,value,unit\nrunning_days,2,天\n", encoding="utf-8")
            second = reader.snapshot()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        self.assertEqual(first["summary"]["running_days"], 1)
        self.assertEqual(second["summary"]["running_days"], 2)

    def test_mysql_data_source_builds_snapshot_from_database_rows(self):
        rows = {
            "SELECT `key`, value, unit FROM overview_summary ORDER BY display_order, id": [
                {"key": "running_days", "value": "449", "unit": "天"},
            ],
        }
        fake_connection = FakeConnection(rows)
        source = server.MySqlDataSource(connector_factory=lambda config: fake_connection, reload_interval=0)

        payload = source.snapshot()

        self.assertEqual(payload["summary"]["running_days"], 449)
        self.assertNotIn("simu", payload)
        self.assertNotIn("scada", payload)
        self.assertNotIn("agc", payload)
        self.assertEqual(
            fake_connection.cursor_obj.executed,
            [("SELECT `key`, value, unit FROM overview_summary ORDER BY display_order, id", ())],
        )

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

        self.assertIn(">考察站风-光-氢-储-柴联合规划系统<", html)
        self.assertIn(">时序数据<", html)
        self.assertNotIn(">电网规划系统<", html)
        self.assertNotIn(">微电网风光氢储联合规划系统<", html)
        self.assertNotIn(">电网规划列表<", html)
        self.assertNotIn(">8760时序数据<", html)

    def test_optimization_page_has_requested_three_area_layout(self):
        html = (WEB_ROOT / "optimize.html").read_text(encoding="utf-8")
        planning_html = (WEB_ROOT / "planning.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn(">考察站风-光-氢-储-柴联合规划系统<", html)
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
        for tab in ("结果概览", "供能分析", "安全评估"):
            self.assertIn(tab, html)
        self.assertNotIn("绿电结果", html)
        self.assertNotIn("安全结果", html)
        self.assertIn('id="overviewResult"', html)
        self.assertIn('id="greenResult"', html)
        self.assertIn('id="safetyResult"', html)
        self.assertIn('id="optimizationLogs"', html)
        self.assertIn('assets/optimize.js', html)
        self.assertIn('href="optimize.html">启动优化</a>', planning_html)
        self.assertIn(".optimization-panel", css)
        self.assertIn("grid-template-rows: var(--optimization-command-height, max-content) 14px minmax(220px, var(--optimization-result-height, 1fr)) 14px minmax(120px, var(--optimization-log-height, 24vh))", css)

    def test_optimization_frontend_polls_status_and_binds_controls(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")

        self.assertIn("/api/planning/schemes", script)
        self.assertIn("/api/optimization/status", script)
        self.assertIn("/api/optimization/control", script)
        self.assertIn("startOptimization", script)
        self.assertIn("stopOptimization", script)
        self.assertIn("updateOptimizationActions", script)
        self.assertIn("startButton.disabled = !hasScheme || isRunning", script)
        self.assertIn("stopButton.disabled = !hasScheme || !isRunning", script)
        self.assertIn("classList.toggle(\"is-disabled\"", script)
        self.assertIn("classList.toggle(\"is-active\"", script)
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
        self.assertIn("供能分析", script)
        self.assertIn("安全评估", script)
        self.assertNotIn("绿电结果", script)
        self.assertNotIn("安全结果", script)
        self.assertIn("optimizationStatusPath", script)
        self.assertIn("defaultOptimizationState", script)
        self.assertIn("scheme=", script)
        self.assertIn("encodeURIComponent(scheme)", script)
        self.assertIn("refreshOptimizationStatus().catch(showError)", script)
        self.assertIn(".optimization-actions button.is-disabled", css := (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8"))
        self.assertIn(".optimization-actions button.is-active", css)

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
        self.assertIn("lockOptimizationCommandHeight", script)
        self.assertIn("--optimization-command-height", script)
        self.assertIn("optimizationResizableContentHeight() - safeHeight", script)
        self.assertIn("optimizationResultHeight", script)
        self.assertIn("optimizationLogHeight", script)
        self.assertIn("--optimization-result-height", script)
        self.assertIn("--optimization-log-height", script)
        self.assertIn("pointerdown", script)
        self.assertIn("setPointerCapture", script)
        self.assertIn("ArrowUp", script)
        self.assertIn("ArrowDown", script)
        result_resize_script = script.split("function bindOptimizationResultResizeHandle()", 1)[1].split("function bindOptimizationLogResizeHandle()", 1)[0]
        self.assertIn("applyHeight(startHeight - (moveEvent.clientY - startY))", result_resize_script)
        self.assertNotIn("applyHeight(startHeight + moveEvent.clientY - startY)", result_resize_script)
        self.assertIn(".optimization-result-resize-handle", css)
        self.assertIn(".optimization-log-resize-handle", css)
        result_tabs_css = css.split(".result-tabs {", 1)[1].split("}", 1)[0]
        self.assertIn("flex: 0 0 auto", result_tabs_css)
        self.assertIn("min-height: 38px", result_tabs_css)
        result_tab_css = css.split(".result-tab {", 1)[1].split("}", 1)[0]
        self.assertIn("height: 36px", result_tab_css)
        self.assertIn("display: inline-flex", result_tab_css)
        self.assertIn("cursor: row-resize", css)
        self.assertIn("--optimization-result-height", css)
        self.assertIn("--optimization-log-height", css)

    def test_optimization_overview_frontend_renders_two_tables_and_ratio_disks(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("renderOverviewTables", script)
        self.assertIn("overview_tables", script)
        self.assertIn("renderOverviewDisks", script)
        self.assertIn("overview_disks", script)
        self.assertIn("overview-ratio-stack", script)
        self.assertIn("optimization-overview-grid", script)
        for title in ("规划结果", "规划年指标"):
            self.assertIn(title, script)
        self.assertNotIn("规划年效益", script)
        for label in ("运行成本", "建设成本", "柴发电量", "新能源电量"):
            self.assertIn(label, script)
        for field in ("设备类型", "设计台数", "指标", "数值", "单位"):
            self.assertIn(field, script)
        self.assertIn(".optimization-overview-grid", css)
        self.assertIn("grid-template-columns: minmax(260px, 1fr) minmax(360px, 0.95fr) minmax(260px, 1fr)", css)
        self.assertIn(".overview-table-card", css)
        self.assertIn(".overview-ratio-stack", css)
        ratio_stack_css = css.split(".overview-ratio-stack {", 1)[1].split("}", 1)[0]
        self.assertIn("display: grid", ratio_stack_css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", ratio_stack_css)
        self.assertIn("overflow: auto", ratio_stack_css)
        ratio_card_css = css.split(".ratio-disk-card {", 1)[1].split("}", 1)[0]
        self.assertIn("grid-template-rows: auto minmax(0, 1fr)", ratio_card_css)
        self.assertIn("min-height: 172px", ratio_card_css)
        self.assertIn("border: 1px solid #d7e4e0", ratio_card_css)
        self.assertIn(".ratio-disk", css)
        ratio_disk_css = css.split(".ratio-disk {", 1)[1].split("}", 1)[0]
        self.assertIn("width: clamp(96px, 7.4vw, 116px)", ratio_disk_css)
        self.assertIn("conic-gradient", css)

    def test_optimization_green_frontend_renders_daily_stacked_chart_and_table(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("renderGreenResult", script)
        self.assertIn("green_table", script)
        self.assertIn("green_daily", script)
        self.assertIn("renderGreenDailyChart", script)
        self.assertIn("bindResultColumnResizeHandles", script)
        self.assertIn("bindAdaptiveResultCharts", script)
        self.assertIn("bindChartHoverCursors", script)
        self.assertIn("updateChartHoverCursor", script)
        self.assertIn("renderGreenHoverTooltip", script)
        self.assertIn("ResizeObserver", script)
        self.assertIn("resultChartMargins", script)
        self.assertIn("chartTickIndexes", script)
        self.assertIn("height >= 200", script)
        self.assertIn("width < 620", script)
        self.assertNotIn("right: 22", script)
        self.assertIn('data-result-column-resize="green"', script)
        self.assertIn('data-result-chart-viewport="green"', script)
        self.assertIn("--green-result-table-width", script)
        self.assertIn("green-daily-chart", script)
        self.assertIn("green-zero-line", script)
        self.assertIn("green-zero-label", script)
        self.assertIn('data-chart-hover="green"', script)
        self.assertIn('data-chart-hover-line="green"', script)
        self.assertIn('data-chart-hover-tooltip="green"', script)
        self.assertNotIn("green-axis-label", script)
        self.assertNotIn("green-y-axis-title", script)
        self.assertNotIn("green-x-axis-title", script)
        self.assertNotIn(">kWh<", script)
        self.assertNotIn("日序号", script)
        self.assertNotIn("renderGreenDailyDataTable", script)
        self.assertNotIn("green-daily-data-table", script)
        self.assertNotIn("日曲线测试数据", script)
        self.assertNotIn("第${day}日", script)
        green_render_script = script.split("function renderGreenResult", 1)[1].split("function renderGreenDailyChart", 1)[0]
        self.assertIn('"单位": row["单位"] || ""', green_render_script)
        self.assertLess(green_render_script.index("green-result-table"), green_render_script.index("green-chart-card"))
        for label in ("柴发日电量", "风电日电量", "光伏日电量", "氢能日电量", "储能放电量", "负荷电量", "制氢电量", "储能充电量"):
            self.assertIn(label, script)
        for metric in (
            '"指标": "负荷总电量", "数值": "-", "单位": "kWh"',
            '"指标": "柴发总电量", "数值": "-", "单位": "kWh"',
            '"指标": "风机总发电量", "数值": "-", "单位": "kWh"',
            '"指标": "光伏总发电量", "数值": "-", "单位": "kWh"',
            '"指标": "电储总发电量", "数值": "-", "单位": "kWh"',
            '"指标": "氢储总发电量", "数值": "-", "单位": "kWh"',
            '"指标": "新能源总弃电量", "数值": "-", "单位": "%"',
            '"指标": "柴油消耗", "数值": "-", "单位": "吨"',
            '"指标": "制氢总量", "数值": "-", "单位": "Nm3"',
        ):
            self.assertIn(metric, script)
        self.assertNotIn("负荷总电量(kWh)", script)

        self.assertIn(".green-result-layout", css)
        green_layout_css = css.split(".green-result-layout {", 1)[1].split("}", 1)[0]
        self.assertIn("display: grid", green_layout_css)
        self.assertIn("grid-template-columns: minmax(260px, var(--green-result-table-width, 34%)) 10px minmax(0, 1fr)", green_layout_css)
        self.assertIn(".result-column-resize-handle", css)
        result_column_resize_css = css.split(".result-column-resize-handle {", 1)[1].split("}", 1)[0]
        self.assertIn("cursor: col-resize", result_column_resize_css)
        self.assertIn(".green-daily-chart", css)
        green_chart_card_css = css.split(".green-chart-card {", 1)[1].split("}", 1)[0]
        self.assertIn("display: grid", green_chart_card_css)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr)", green_chart_card_css)
        self.assertIn(".green-chart-viewport", css)
        green_chart_viewport_css = css.split(".green-chart-viewport,", 1)[1].split("{", 1)[1].split("}", 1)[0]
        self.assertIn("height: 100%", green_chart_viewport_css)
        self.assertIn("min-height: 0", green_chart_viewport_css)
        self.assertIn(".green-chart-svg", css)
        self.assertIn(".chart-hover-line", css)
        self.assertIn(".chart-hover-tooltip", css)
        chart_tooltip_css = css.split(".chart-hover-tooltip {", 1)[1].split("}", 1)[0]
        self.assertIn("position: absolute", chart_tooltip_css)
        self.assertIn("pointer-events: none", chart_tooltip_css)
        self.assertIn('preserveAspectRatio="xMidYMid meet"', script)
        self.assertNotIn('preserveAspectRatio="none"', script)
        self.assertIn('resultChartSize("green"', script)
        green_chart_svg_css = css.split(".green-chart-svg {", 1)[1].split("}", 1)[0]
        self.assertIn("width: 100%", green_chart_svg_css)
        self.assertIn("height: 100%", green_chart_svg_css)
        self.assertIn("min-height: 0", green_chart_svg_css)
        self.assertIn(".green-chart-legend", css)
        self.assertIn(".green-result-table", css)
        self.assertNotIn(".green-daily-data-table", css)
        self.assertNotIn(".green-daily-data-section", css)

    def test_optimization_safety_frontend_renders_frequency_chart_and_table(self):
        script = (WEB_ROOT / "assets" / "optimize.js").read_text(encoding="utf-8")
        css = (WEB_ROOT / "assets" / "planning.css").read_text(encoding="utf-8")

        self.assertIn("renderSafetyResult", script)
        self.assertIn("safety_table", script)
        self.assertIn("safety_daily", script)
        self.assertIn("renderSafetyDailyChart", script)
        self.assertIn("bindResultColumnResizeHandles", script)
        self.assertIn("bindAdaptiveResultCharts", script)
        self.assertIn("bindChartHoverCursors", script)
        self.assertIn("updateChartHoverCursor", script)
        self.assertIn("renderSafetyHoverTooltip", script)
        self.assertIn("ResizeObserver", script)
        self.assertIn("resultChartMargins", script)
        self.assertIn("chartTickIndexes", script)
        self.assertIn("height >= 200", script)
        self.assertIn("width < 620", script)
        self.assertNotIn("right: 22", script)
        self.assertIn('data-result-column-resize="safety"', script)
        self.assertIn('data-result-chart-viewport="safety"', script)
        self.assertIn("--safety-result-table-width", script)
        self.assertIn("safety-frequency-chart", script)
        self.assertIn("safety-center-line", script)
        self.assertIn("safety-zero-label", script)
        self.assertIn('data-chart-hover="safety"', script)
        self.assertIn('data-chart-hover-line="safety"', script)
        self.assertIn('data-chart-hover-tooltip="safety"', script)
        self.assertIn("formatFrequencyDeviation", script)
        self.assertNotIn("safety-axis-label", script)
        self.assertNotIn("safety-y-axis-title", script)
        self.assertNotIn("safety-x-axis-title", script)
        self.assertNotIn(">Hz<", script)
        self.assertNotIn("日序号", script)
        self.assertNotIn(">50Hz<", script)
        self.assertIn("向上频率最大值", script)
        self.assertIn("向下频率最小值", script)
        self.assertNotIn("第${day}日", script)
        safety_render_script = script.split("function renderSafetyResult", 1)[1].split("function renderSafetyDailyChart", 1)[0]
        self.assertLess(safety_render_script.index("safety-result-table"), safety_render_script.index("safety-chart-card"))
        for metric in (
            "向上扰动最大量",
            "向下扰动最大量",
            "最高频率",
            "最低频率",
            "频率安全风险小时数",
        ):
            self.assertIn(metric, script)

        self.assertIn(".safety-result-layout", css)
        safety_layout_css = css.split(".safety-result-layout {", 1)[1].split("}", 1)[0]
        self.assertIn("display: grid", safety_layout_css)
        self.assertIn("grid-template-columns: minmax(260px, var(--safety-result-table-width, 34%)) 10px minmax(0, 1fr)", safety_layout_css)
        self.assertIn(".result-column-resize-handle", css)
        result_column_resize_css = css.split(".result-column-resize-handle {", 1)[1].split("}", 1)[0]
        self.assertIn("cursor: col-resize", result_column_resize_css)
        self.assertIn(".safety-chart-card", css)
        safety_chart_card_css = css.split(".safety-chart-card {", 1)[1].split("}", 1)[0]
        self.assertIn("display: grid", safety_chart_card_css)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr)", safety_chart_card_css)
        self.assertIn(".safety-chart-viewport", css)
        safety_chart_viewport_css = css.split(".green-chart-viewport,", 1)[1].split("{", 1)[1].split("}", 1)[0]
        self.assertIn("height: 100%", safety_chart_viewport_css)
        self.assertIn("min-height: 0", safety_chart_viewport_css)
        self.assertIn(".safety-chart-svg", css)
        self.assertIn(".chart-hover-line", css)
        self.assertIn(".chart-hover-tooltip", css)
        chart_tooltip_css = css.split(".chart-hover-tooltip {", 1)[1].split("}", 1)[0]
        self.assertIn("position: absolute", chart_tooltip_css)
        self.assertIn("pointer-events: none", chart_tooltip_css)
        self.assertIn('preserveAspectRatio="xMidYMid meet"', script)
        self.assertNotIn('preserveAspectRatio="none"', script)
        self.assertIn('resultChartSize("safety"', script)
        safety_chart_svg_css = css.split(".safety-chart-svg {", 1)[1].split("}", 1)[0]
        self.assertIn("width: 100%", safety_chart_svg_css)
        self.assertIn("height: 100%", safety_chart_svg_css)
        self.assertIn("min-height: 0", safety_chart_svg_css)
        self.assertIn(".safety-center-line", css)
        self.assertIn(".safety-result-table", css)

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
        self.assertNotIn("时间（月）", script)
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
