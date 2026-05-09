# Grid Planning Parameter Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the parameter-maintenance part of the grid-planning Web site: scheme folders, XLSX persistence, APIs, and a native HTML/JS editor for 8760 curves plus device parameter tables.

**Architecture:** Keep the existing `power_plan` lightweight server and add a focused planning persistence module. The backend owns scheme-folder safety, XLSX read/write, default workbook creation, validation, and API routing; the frontend owns editing state, chart rendering, table rendering, and user actions.

**Tech Stack:** Python 3.11, `http.server`, `openpyxl`, `unittest`, native HTML/CSS/JavaScript, SVG for the 8760 curve panel.

---

## File Structure

- Create: `power_plan/planning_store.py` — scheme validation, default data, XLSX read/write, list/copy/rename operations.
- Modify: `power_plan/server.py` — `/api/planning/...` routes and PUT support.
- Modify: `power_plan/requirements.txt` — add `openpyxl>=3.1.0`.
- Create: `power_plan/planning.html` — planning UI shell and parameter-maintenance workspace.
- Create: `power_plan/assets/planning.css` — planning page styles.
- Create: `power_plan/assets/planning.js` — scheme actions, 8760 chart/table, device tables, validation summary.
- Create: `power_plan/tests/test_planning_store.py` — XLSX store tests.
- Modify: `power_plan/tests/test_server.py` — planning API tests.
- Modify: `power_plan/index.html` — link to planning page.

---

### Task 1: Add XLSX Dependency

**Files:**
- Modify: `power_plan/requirements.txt`

- [ ] **Step 1: Add dependency line**

Replace `power_plan/requirements.txt` with:

```txt
PyMySQL>=1.1.0
openpyxl>=3.1.0
```

- [ ] **Step 2: Verify dependency file**

Run: `Get-Content power_plan/requirements.txt`

Expected output contains:

```txt
PyMySQL>=1.1.0
openpyxl>=3.1.0
```

- [ ] **Step 3: Commit**

```powershell
git add power_plan/requirements.txt
git commit -m "chore: add planning xlsx dependency"
```

---

### Task 2: Create Planning Store and Default Workbook

**Files:**
- Create: `power_plan/planning_store.py`
- Create: `power_plan/tests/test_planning_store.py`

- [ ] **Step 1: Write failing tests**

Create `power_plan/tests/test_planning_store.py`:

```python
import shutil
import sys
import unittest
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_ROOT))

import planning_store


class PlanningStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = WEB_ROOT / "tests" / "tmp_planning_store"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)
        self.store = planning_store.PlanningStore(root=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_validate_scheme_name_accepts_chinese_letters_numbers(self):
        self.assertEqual(planning_store.validate_scheme_name("方案A-01"), "方案A-01")

    def test_validate_scheme_name_rejects_path_chars(self):
        for name in ("", "../bad", "a/b", "a\\b", ".", ".."):
            with self.assertRaises(ValueError):
                planning_store.validate_scheme_name(name)

    def test_create_scheme_writes_default_workbook(self):
        payload = self.store.create_scheme("方案A")

        workbook = self.tmp_dir / "方案A" / "parameters.xlsx"
        self.assertTrue(workbook.exists())
        self.assertEqual(payload["scheme"], "方案A")
        self.assertEqual(len(payload["time_series"]), 8760)
        self.assertIn("diesel_generators", payload)
        self.assertIn("storage_battery_packs", payload)
        self.assertIn("hydrogen_tanks", payload)
        self.assertEqual(payload["validation"][0]["level"], "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest power_plan/tests/test_planning_store.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'planning_store'`.

- [ ] **Step 3: Create `power_plan/planning_store.py`**

Use this initial implementation:

```python
from __future__ import annotations

import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook

WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEME_ROOT = WEB_ROOT / "planning_schemes"
WORKBOOK_NAME = "parameters.xlsx"

SHEET_SPECS: dict[str, tuple[str, list[str]]] = {
    "time_series": ("8760时序数据", ["hour_index", "datetime", "wind_speed", "solar_irradiance", "load"]),
    "diesel_generators": ("柴发参数", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "power_upper", "power_lower", "fuel_rate"]),
    "wind_turbines": ("风机参数", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]),
    "photovoltaics": ("光伏参数", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]),
    "storage_pcs": ("储能PCS参数", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]),
    "storage_battery_packs": ("储能电池组参数", ["name", "battery_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]),
    "hydrogen_electrolyzers": ("电制氢参数", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "electric_to_hydrogen_efficiency"]),
    "hydrogen_tanks": ("储氢罐参数", ["name", "hydrogen_tank_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]),
    "fuel_cells": ("燃料电池参数", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "hydrogen_to_electric_efficiency"]),
}

DEFAULT_DEVICE_ROWS: dict[str, list[dict[str, Any]]] = {
    "diesel_generators": [{"name": "柴发1", "capacity": 100, "design_capacity_lower": 0, "design_capacity_upper": 500, "cost": 0, "power_upper": 100, "power_lower": 20, "fuel_rate": 0.26}],
    "wind_turbines": [{"name": "风机1", "capacity": 50, "design_capacity_lower": 0, "design_capacity_upper": 1000, "cost": 0, "cut_in_wind_speed": 3, "cut_out_wind_speed": 25}],
    "photovoltaics": [{"name": "光伏1", "capacity": 50, "design_capacity_lower": 0, "design_capacity_upper": 1000, "cost": 0, "cut_in_wind_speed": 0, "cut_out_wind_speed": 0}],
    "storage_pcs": [{"name": "储能PCS1", "power_capacity": 50, "design_capacity_lower": 0, "design_capacity_upper": 500, "cost": 0}],
    "storage_battery_packs": [{"name": "储能电池组1", "battery_capacity": 200, "design_capacity_lower": 0, "design_capacity_upper": 2000, "cost": 0}],
    "hydrogen_electrolyzers": [{"name": "电制氢1", "power_capacity": 50, "design_capacity_lower": 0, "design_capacity_upper": 500, "cost": 0, "electric_to_hydrogen_efficiency": 0.7}],
    "hydrogen_tanks": [{"name": "储氢罐1", "hydrogen_tank_capacity": 100, "design_capacity_lower": 0, "design_capacity_upper": 2000, "cost": 0}],
    "fuel_cells": [{"name": "燃料电池1", "power_capacity": 50, "design_capacity_lower": 0, "design_capacity_upper": 500, "cost": 0, "hydrogen_to_electric_efficiency": 0.55}],
}

INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*]')


def validate_scheme_name(name: str) -> str:
    clean = str(name or "").strip()
    if clean in {"", ".", ".."} or INVALID_NAME_RE.search(clean) or ".." in clean:
        raise ValueError("方案名称不能为空，且不能包含路径或非法字符")
    return clean


def default_time_series() -> list[dict[str, Any]]:
    return [{"hour_index": hour, "datetime": f"H{hour:04d}", "wind_speed": 0, "solar_irradiance": 0, "load": 0} for hour in range(1, 8761)]


def default_payload(scheme: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"scheme": scheme, "time_series": default_time_series(), "validation": []}
    for key in SHEET_SPECS:
        if key != "time_series":
            payload[key] = deepcopy(DEFAULT_DEVICE_ROWS[key])
    payload["capacity_limits"] = []
    return payload


@dataclass
class PlanningStore:
    root: Path = DEFAULT_SCHEME_ROOT

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def scheme_dir(self, name: str) -> Path:
        clean = validate_scheme_name(name)
        path = (self.root / clean).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("方案路径越界")
        return path

    def workbook_path(self, name: str) -> Path:
        return self.scheme_dir(name) / WORKBOOK_NAME

    def create_scheme(self, name: str) -> dict[str, Any]:
        clean = validate_scheme_name(name)
        folder = self.scheme_dir(clean)
        if folder.exists():
            raise FileExistsError(f"方案已存在: {clean}")
        folder.mkdir(parents=True)
        self.write_scheme(clean, default_payload(clean))
        return self.read_scheme(clean)

    def write_scheme(self, name: str, payload: dict[str, Any]) -> None:
        clean = validate_scheme_name(name)
        folder = self.scheme_dir(clean)
        folder.mkdir(parents=True, exist_ok=True)
        workbook = build_workbook(payload | {"scheme": clean})
        tmp_path = folder / f".{WORKBOOK_NAME}.tmp"
        final_path = folder / WORKBOOK_NAME
        workbook.save(tmp_path)
        tmp_path.replace(final_path)

    def read_scheme(self, name: str) -> dict[str, Any]:
        clean = validate_scheme_name(name)
        path = self.workbook_path(clean)
        if not path.exists():
            raise FileNotFoundError(f"方案参数文件不存在: {path}")
        payload = read_workbook(path, clean)
        payload["validation"] = validate_payload(payload)
        return payload


def build_workbook(payload: dict[str, Any]) -> Workbook:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for key, (sheet_name, headers) in SHEET_SPECS.items():
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(headers)
        for row in payload.get(key, []):
            sheet.append([row.get(header, "") for header in headers])
    return workbook


def read_workbook(path: Path, scheme: str) -> dict[str, Any]:
    workbook = load_workbook(path, data_only=True)
    payload: dict[str, Any] = {"scheme": scheme, "validation": [], "capacity_limits": []}
    for key, (sheet_name, headers) in SHEET_SPECS.items():
        if sheet_name not in workbook.sheetnames:
            payload[key] = []
            payload["validation"].append({"level": "error", "message": f"缺少工作表: {sheet_name}"})
            continue
        sheet = workbook[sheet_name]
        rows = []
        for values in sheet.iter_rows(min_row=2, values_only=True):
            if values is None or all(value is None for value in values):
                continue
            rows.append({header: values[index] if index < len(values) and values[index] is not None else "" for index, header in enumerate(headers)})
        payload[key] = rows
    return payload


def validate_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    time_series = payload.get("time_series", [])
    if len(time_series) != 8760:
        messages.append({"level": "error", "message": f"8760时序数据行数应为8760，当前为{len(time_series)}"})
    else:
        messages.append({"level": "ok", "message": "8760时序数据行数正确"})
    for key in SHEET_SPECS:
        if key == "time_series":
            continue
        for index, row in enumerate(payload.get(key, []), start=1):
            lower = row.get("design_capacity_lower", "")
            upper = row.get("design_capacity_upper", "")
            if lower != "" and upper != "" and float(lower) > float(upper):
                messages.append({"level": "error", "message": f"{SHEET_SPECS[key][0]}第{index}行设计容量下限大于上限"})
    return messages
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest power_plan/tests/test_planning_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add power_plan/planning_store.py power_plan/tests/test_planning_store.py
git commit -m "feat: add planning xlsx store"
```

---

### Task 3: Add Scheme List, Copy, Rename, and Round Trip Save

**Files:**
- Modify: `power_plan/planning_store.py`
- Modify: `power_plan/tests/test_planning_store.py`

- [ ] **Step 1: Add failing tests**

Append inside `PlanningStoreTest`:

```python
    def test_list_copy_and_rename_schemes(self):
        self.store.create_scheme("方案A")
        self.store.copy_scheme("方案A", "方案B")
        self.store.rename_scheme("方案B", "方案C")

        names = [item["name"] for item in self.store.list_schemes()]
        self.assertEqual(names, ["方案A", "方案C"])
        self.assertTrue((self.tmp_dir / "方案C" / "parameters.xlsx").exists())
        self.assertFalse((self.tmp_dir / "方案B").exists())

    def test_write_and_read_scheme_round_trip(self):
        self.store.create_scheme("方案A")
        payload = self.store.read_scheme("方案A")
        payload["time_series"][0]["wind_speed"] = 8.5
        payload["diesel_generators"][0]["design_capacity_upper"] = 650
        payload["hydrogen_tanks"][0]["hydrogen_tank_capacity"] = 300

        self.store.write_scheme("方案A", payload)
        saved = self.store.read_scheme("方案A")

        self.assertEqual(saved["time_series"][0]["wind_speed"], 8.5)
        self.assertEqual(saved["diesel_generators"][0]["design_capacity_upper"], 650)
        self.assertEqual(saved["hydrogen_tanks"][0]["hydrogen_tank_capacity"], 300)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest power_plan/tests/test_planning_store.py -q`

Expected: FAIL with missing `copy_scheme`, `rename_scheme`, or `list_schemes`.

- [ ] **Step 3: Add methods to `PlanningStore`**

Insert after `create_scheme`:

```python
    def list_schemes(self) -> list[dict[str, Any]]:
        schemes: list[dict[str, Any]] = []
        for folder in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not folder.is_dir():
                continue
            workbook = folder / WORKBOOK_NAME
            schemes.append({"name": folder.name, "has_workbook": workbook.exists(), "modified_at": workbook.stat().st_mtime if workbook.exists() else None})
        return schemes

    def copy_scheme(self, source: str, target: str) -> dict[str, Any]:
        source_dir = self.scheme_dir(source)
        target_dir = self.scheme_dir(target)
        if not source_dir.exists():
            raise FileNotFoundError(f"源方案不存在: {source}")
        if target_dir.exists():
            raise FileExistsError(f"目标方案已存在: {target}")
        shutil.copytree(source_dir, target_dir)
        return self.read_scheme(target)

    def rename_scheme(self, source: str, target: str) -> dict[str, Any]:
        source_dir = self.scheme_dir(source)
        target_dir = self.scheme_dir(target)
        if not source_dir.exists():
            raise FileNotFoundError(f"源方案不存在: {source}")
        if target_dir.exists():
            raise FileExistsError(f"目标方案已存在: {target}")
        source_dir.rename(target_dir)
        return self.read_scheme(target)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest power_plan/tests/test_planning_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add power_plan/planning_store.py power_plan/tests/test_planning_store.py
git commit -m "feat: manage planning schemes"
```

---

### Task 4: Add Planning API Routes

**Files:**
- Modify: `power_plan/server.py`
- Modify: `power_plan/tests/test_server.py`

- [ ] **Step 1: Add failing API tests**

Append inside `PowerPlanServerTest`:

```python
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

            created["time_series"][0]["load"] = 123.4
            status, headers, body = server.handle_planning_api_path(
                "/api/planning/schemes/方案A",
                "PUT",
                json.dumps(created, ensure_ascii=False).encode("utf-8"),
            )
            self.assertEqual(status, 200)

            status, headers, body = server.handle_planning_api_path("/api/planning/schemes/方案A", "GET", b"")
            loaded = json.loads(body.decode("utf-8"))
            self.assertEqual(loaded["time_series"][0]["load"], 123.4)

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

    def test_planning_api_rejects_bad_scheme_name(self):
        status, headers, body = server.handle_planning_api_path(
            "/api/planning/schemes",
            "POST",
            json.dumps({"name": "../bad"}).encode("utf-8"),
        )

        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body.decode("utf-8"))["error"], "bad_request")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest power_plan/tests/test_server.py -q`

Expected: FAIL with missing `handle_planning_api_path` or `PLANNING_STORE`.

- [ ] **Step 3: Import store and add route handler**

In `power_plan/server.py`, add `import planning_store` after imports. Add `PLANNING_STORE = planning_store.PlanningStore()` after `DATA_SOURCE = MySqlDataSource()`. Add this after `_json_response`:

```python
def _read_json_body(body: bytes) -> dict:
    try:
        return json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("请求体不是合法 JSON") from exc


def handle_planning_api_path(path: str, method: str = "GET", body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
    prefix = "/api/planning/schemes"
    try:
        if path == prefix and method == "GET":
            return _json_response({"schemes": PLANNING_STORE.list_schemes()})
        if path == prefix and method == "POST":
            payload = _read_json_body(body)
            return _json_response(PLANNING_STORE.create_scheme(str(payload.get("name", ""))))
        if path == f"{prefix}/copy" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(PLANNING_STORE.copy_scheme(str(payload.get("source", "")), str(payload.get("target", ""))))
        if path == f"{prefix}/rename" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(PLANNING_STORE.rename_scheme(str(payload.get("source", "")), str(payload.get("target", ""))))
        if path.startswith(f"{prefix}/"):
            name = unquote(path[len(prefix) + 1 :])
            if method == "GET":
                return _json_response(PLANNING_STORE.read_scheme(name))
            if method == "PUT":
                payload = _read_json_body(body)
                PLANNING_STORE.write_scheme(name, payload)
                return _json_response(PLANNING_STORE.read_scheme(name))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
    return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)
```

- [ ] **Step 4: Wire GET, POST, and PUT**

At the top of `handle_api_path`, add:

```python
    if path.startswith("/api/planning/"):
        return handle_planning_api_path(path, "GET", b"")
```

In `PowerPlanHandler.do_POST`, before `handle_control_path`, add:

```python
            if parsed.path.startswith("/api/planning/"):
                status, headers, response_body = handle_planning_api_path(parsed.path, "POST", body)
                self._send(status, headers, response_body)
                return
```

Add this method to `PowerPlanHandler`:

```python
    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        if parsed.path.startswith("/api/planning/"):
            status, headers, response_body = handle_planning_api_path(parsed.path, "PUT", body)
            self._send(status, headers, response_body)
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not Found")
```

- [ ] **Step 5: Run API tests**

Run: `python -m pytest power_plan/tests/test_server.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add power_plan/server.py power_plan/tests/test_server.py
git commit -m "feat: add planning scheme api"
```

---

### Task 5: Build Planning Page UI

**Files:**
- Create: `power_plan/planning.html`
- Create: `power_plan/assets/planning.css`
- Create: `power_plan/assets/planning.js`

- [ ] **Step 1: Create `planning.html`**

Create a shell with these required IDs: `saveScheme`, `schemeList`, `createScheme`, `copyScheme`, `renameScheme`, `timeChart`, `chartTip`, `timeTable`, `prevPage`, `nextPage`, `pageInfo`, `deviceJump`, `deviceTables`, `limitSummary`, `schemeSummary`, and `validationList`. Include tabs with `data-tab="time"`, `data-tab="devices"`, and `data-tab="limits"`. Include script `assets/planning.js` and stylesheet `assets/planning.css`.

- [ ] **Step 2: Add CSS**

Create `power_plan/assets/planning.css` with layout classes for `.planning-shell`, `.topbar`, `.workspace`, `.scheme-rail`, `.editor-panel`, `.summary-rail`, `.tabs`, `.tab-panel`, `.chart-card`, `.table-card`, `.device-card`, `.time-chart`, `.data-table`, and responsive collapse at `max-width: 1100px`.

- [ ] **Step 3: Add JavaScript implementation**

Create `power_plan/assets/planning.js` with:

```javascript
const state = { schemes: [], currentScheme: "", payload: null, page: 0, pageSize: 168 };
const deviceSpecs = [
  ["diesel_generators", "柴发", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "power_upper", "power_lower", "fuel_rate"]],
  ["wind_turbines", "风机", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]],
  ["photovoltaics", "光伏", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]],
  ["storage_pcs", "储能PCS", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["storage_battery_packs", "储能电池组", ["name", "battery_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["hydrogen_electrolyzers", "电制氢", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "electric_to_hydrogen_efficiency"]],
  ["hydrogen_tanks", "储氢罐", ["name", "hydrogen_tank_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["fuel_cells", "燃料电池", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "hydrogen_to_electric_efficiency"]],
];
```

Implement these functions with the exact names so future tests can target them: `bindTabs`, `bindActions`, `api`, `loadSchemes`, `renderSchemes`, `selectScheme`, `createScheme`, `copyScheme`, `renameScheme`, `saveScheme`, `renderAll`, `renderChart`, `renderTimeTable`, `onTimeInput`, `renderDeviceTables`, `deviceTable`, `onDeviceInput`, `addDeviceRow`, `deleteDeviceRow`, `renderLimitSummary`, `renderSummary`, `validateLocal`, `coerceInput`, and `escapeHtml`.

`renderChart` must draw three SVG polylines or paths for `wind_speed`, `solar_irradiance`, and `load` when checked. `renderTimeTable` must show `168` rows per page. `renderDeviceTables` must render all eight device tables on one webpage, and each table must include `design_capacity_lower` and `design_capacity_upper` columns.

- [ ] **Step 4: Static file smoke check**

Run:

```powershell
python - <<'PY'
from pathlib import Path
for path in ['power_plan/planning.html','power_plan/assets/planning.css','power_plan/assets/planning.js']:
    print(path, Path(path).exists())
PY
```

Expected:

```txt
power_plan/planning.html True
power_plan/assets/planning.css True
power_plan/assets/planning.js True
```

- [ ] **Step 5: Commit**

```powershell
git add power_plan/planning.html power_plan/assets/planning.css power_plan/assets/planning.js
git commit -m "feat: add planning parameter editor"
```

---

### Task 6: Link Planning Page From Existing Home

**Files:**
- Modify: `power_plan/index.html`

- [ ] **Step 1: Add planning navigation item**

In `power_plan/index.html`, replace:

```html
      <a href="simu.html">实时控制</a>
      <a href="agc.html">优化调度</a>
```

with:

```html
      <a href="planning.html">参数维护</a>
      <a href="planning.html">电网规划</a>
```

- [ ] **Step 2: Verify link exists**

Run: `Select-String -Path power_plan/index.html -Pattern 'planning.html'`

Expected: output shows two `planning.html` matches.

- [ ] **Step 3: Commit**

```powershell
git add power_plan/index.html
git commit -m "feat: link planning page from monitor home"
```

---

### Task 7: Full Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run planning store tests**

Run: `python -m pytest power_plan/tests/test_planning_store.py -q`

Expected: PASS.

- [ ] **Step 2: Run server tests**

Run: `python -m pytest power_plan/tests/test_server.py -q`

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest -q`

Expected: PASS. If unrelated existing tests fail, record exact failing names and confirm the planning tests still pass.

- [ ] **Step 4: Manual browser verification**

Run: `python power_plan/server.py --host 127.0.0.1 --port 8866`

Open: `http://127.0.0.1:8866/planning.html`

Verify:
- Create scheme `方案A`.
- Edit hour 1 wind speed, solar irradiance, and load in the lower table.
- Confirm the upper 8760 curve panel redraws.
- Add one row in each device table.
- Edit design capacity lower and upper in the device tables.
- Open the design-capacity summary tab and confirm those values are reflected.
- Save, refresh, select `方案A`, and confirm edited values persist.
- Copy `方案A` to `方案B`.
- Rename `方案B` to `方案C`.
- Confirm `power_plan/planning_schemes/方案A/parameters.xlsx` and `power_plan/planning_schemes/方案C/parameters.xlsx` exist.

- [ ] **Step 5: Commit verification fixes only if files changed**

```powershell
git status --short
git add power_plan/planning_store.py power_plan/server.py power_plan/planning.html power_plan/assets/planning.css power_plan/assets/planning.js power_plan/tests/test_planning_store.py power_plan/tests/test_server.py power_plan/index.html power_plan/requirements.txt
git commit -m "test: verify planning parameter maintenance"
```

---

## Self-Review

- Spec coverage: The plan covers scheme folders, XLSX persistence, new/copy/rename/save, 8760 curve plus table, unified device parameter page, design capacity lower/upper inside device tables, summary validation, and existing lightweight server integration.
- Placeholder scan: The plan contains concrete file paths, commands, code blocks, and expected results.
- Type consistency: Backend payload keys match frontend keys: `time_series`, `diesel_generators`, `wind_turbines`, `photovoltaics`, `storage_pcs`, `storage_battery_packs`, `hydrogen_electrolyzers`, `hydrogen_tanks`, and `fuel_cells`.

