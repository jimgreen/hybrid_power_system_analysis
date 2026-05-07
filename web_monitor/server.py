#!/usr/bin/env python3
"""Static web server and JSON API for the web_monitor dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


WEB_ROOT = Path(__file__).resolve().parent
DATA_DIR = WEB_ROOT / "data"


def _coerce_value(value: str) -> float | int | str:
    value = value.strip()
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _metric(label: str, value: float | int | str, unit: str, status: str = "normal") -> dict:
    return {"label": label, "value": value, "unit": unit, "status": status}


def _summary(label: str, value: str, status: str = "normal") -> dict:
    return {"label": label, "value": value, "status": status}


def _time_to_hour(value: str) -> float:
    hour, minute = value.split(":", 1)
    return int(hour) + int(minute) / 60


def _hour_to_time(value: float) -> str:
    value = value % 24
    hour = int(value)
    minute = int(round((value - hour) * 60)) % 60
    return f"{hour:02d}:{minute:02d}"


class SimuRuntime:
    """In-memory runtime state for simulation controls."""

    def __init__(self, initial_time: str = "00:00", speed: float = 1.0, status: str = "STOPPED") -> None:
        self.cursor_hour = _time_to_hour(initial_time)
        self.speed = speed
        self.status = status
        self._last_tick = time.monotonic()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            self._advance_locked()
            return {
                "sim_time": _hour_to_time(self.cursor_hour),
                "cursor_hour": round(self.cursor_hour, 3),
                "speed": self.speed,
                "status": self.status,
            }

    def apply(self, action: str) -> dict:
        with self._lock:
            self._advance_locked()
            if action == "start":
                self.status = "RUNNING"
            elif action == "faster":
                self.speed = min(16.0, self.speed * 2)
            elif action == "slower":
                self.speed = max(0.25, self.speed / 2)
            elif action == "stop":
                self.status = "STOPPED"
            elif action == "reset":
                self.cursor_hour = 0.0
                self.speed = 1.0
                self.status = "STOPPED"
            else:
                raise ValueError(f"unknown SIMU action: {action}")
            self._last_tick = time.monotonic()
            return {
                "sim_time": _hour_to_time(self.cursor_hour),
                "cursor_hour": round(self.cursor_hour, 3),
                "speed": self.speed,
                "status": self.status,
            }

    def _advance_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_tick
        if self.status == "RUNNING":
            # One real second advances one simulated minute at speed=1.
            self.cursor_hour = (self.cursor_hour + elapsed * self.speed / 60) % 24
        self._last_tick = now


class CsvDataSource:
    """Periodically reload dashboard data from CSV files."""

    def __init__(self, data_dir: Path = DATA_DIR, reload_interval: float = 1.0) -> None:
        self.data_dir = Path(data_dir)
        self.reload_interval = reload_interval
        self._snapshot: dict | None = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def snapshot(self, force_reload: bool = False) -> dict:
        now = time.monotonic()
        with self._lock:
            expired = now - self._loaded_at >= self.reload_interval
            if force_reload or self._snapshot is None or expired:
                self._snapshot = self._load_snapshot()
                self._loaded_at = now
            return self._snapshot

    def _rows(self, filename: str) -> list[dict[str, str]]:
        path = self.data_dir / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]

    def _by_page(self, rows: list[dict[str, str]], page: str) -> list[dict[str, str]]:
        return [row for row in rows if row.get("page") == page]

    def _load_snapshot(self) -> dict:
        metrics = self._rows("metrics.csv")
        alarms = self._rows("alarms.csv")
        page_summary = self._rows("page_summary.csv")
        agc_units = [
            {
                "name": row.get("name", ""),
                "percent": _coerce_value(row.get("percent", "")),
                "power": _coerce_value(row.get("power", "")),
                "unit": row.get("unit", ""),
            }
            for row in self._rows("agc_units.csv")
        ]

        return {
            "system": "南极秦岭站综合能量管理系统",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": self._load_overview_summary(),
            "simu": {
                "section": "SIMU在线监视",
                "metrics": self._load_metrics(metrics, "simu"),
                "alarms": self._load_alarms(alarms, "simu"),
                "charts": {
                    "bars": self._load_label_values("simu_bars.csv"),
                    "daily": self._load_simu_daily_curves(),
                },
                "topology": self._load_topology(),
                "summary": self._load_page_summary(page_summary, "simu"),
                "state": SIMU_RUNTIME.snapshot(),
            },
            "scada": {
                "section": "SCADA在线监视",
                "metrics": self._load_metrics(metrics, "scada"),
                "alarms": self._load_alarms(alarms, "scada"),
                "charts": {"columns": self._load_label_values("scada_columns.csv")},
                "stations": self._load_stations(),
                "summary": self._load_page_summary(page_summary, "scada"),
            },
            "agc": {
                "section": "AGC在线监视",
                "metrics": self._load_metrics(metrics, "agc"),
                "alarms": self._load_alarms(alarms, "agc"),
                "charts": {"units": agc_units},
                "units": agc_units,
                "reserve": self._load_agc_reserve(),
                "summary": self._load_page_summary(page_summary, "agc"),
            },
        }

    def _load_overview_summary(self) -> dict:
        return {
            row.get("key", ""): _coerce_value(row.get("value", ""))
            for row in self._rows("summary.csv")
            if row.get("key")
        }

    def _load_metrics(self, rows: list[dict[str, str]], page: str) -> list[dict]:
        return [
            _metric(
                row.get("label", ""),
                _coerce_value(row.get("value", "")),
                row.get("unit", ""),
                row.get("status", "normal"),
            )
            for row in self._by_page(rows, page)
        ]

    def _load_alarms(self, rows: list[dict[str, str]], page: str) -> list[dict]:
        return [
            {
                "time": row.get("time", ""),
                "object": row.get("object", ""),
                "message": row.get("message", ""),
                "status": row.get("status", ""),
            }
            for row in self._by_page(rows, page)
        ]

    def _load_page_summary(self, rows: list[dict[str, str]], page: str) -> list[dict]:
        return [
            _summary(row.get("label", ""), row.get("value", ""), row.get("status", "normal"))
            for row in self._by_page(rows, page)
        ]

    def _load_label_values(self, filename: str) -> list[dict]:
        return [
            {"label": row.get("label", ""), "value": _coerce_value(row.get("value", "")), "unit": row.get("unit", "")}
            for row in self._rows(filename)
        ]

    def _load_topology(self) -> list[dict]:
        return [
            {"id": row.get("id", ""), "status": row.get("status", "normal"), "value": row.get("value", "")}
            for row in self._rows("simu_topology.csv")
        ]

    def _load_simu_daily_curves(self) -> list[dict]:
        rows = self._rows("simu_daily_curves.csv")
        specs = [
            ("wind_speed", "风速", "m/s"),
            ("temperature", "温度", "℃"),
            ("solar_irradiance", "太阳辐射", "W/m²"),
            ("load", "负荷", "kW"),
        ]
        curves = []
        for key, name, unit in specs:
            points = [
                {"hour": _coerce_value(row.get("hour", "")), "value": _coerce_value(row.get(key, ""))}
                for row in rows
            ]
            curves.append({"key": key, "name": name, "unit": unit, "points": points})
        return curves

    def _load_stations(self) -> list[dict]:
        return [
            {"name": row.get("name", ""), "status": row.get("status", "normal"), "detail": row.get("detail", "")}
            for row in self._rows("scada_stations.csv")
        ]

    def _load_agc_reserve(self) -> dict:
        rows = self._rows("agc_reserve.csv")
        if not rows:
            return {"score": 0, "up": 0, "down": 0, "response": 0, "cycle": 0}
        row = rows[0]
        return {
            "score": _coerce_value(row.get("score", "")),
            "up": _coerce_value(row.get("up", "")),
            "down": _coerce_value(row.get("down", "")),
            "response": _coerce_value(row.get("response", "")),
            "cycle": _coerce_value(row.get("cycle", "")),
        }


def _load_initial_simu_runtime() -> SimuRuntime:
    rows = CsvDataSource(DATA_DIR)._rows("simu_state.csv")
    if not rows:
        return SimuRuntime()
    row = rows[0]
    return SimuRuntime(
        initial_time=row.get("sim_time", "00:00"),
        speed=float(_coerce_value(row.get("speed", "1.0")) or 1.0),
        status=row.get("status", "STOPPED"),
    )


SIMU_RUNTIME = _load_initial_simu_runtime()
DATA_SOURCE = CsvDataSource()


def build_snapshot(force_reload: bool = False) -> dict:
    """Build a snapshot from CSV files, reloading periodically."""
    return DATA_SOURCE.snapshot(force_reload=force_reload)


def _json_response(payload: dict, status: int = 200) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
    }, body


def handle_api_path(path: str) -> tuple[int, dict[str, str], bytes]:
    snapshot = build_snapshot()
    routes = {
        "/api/health": {"ok": True, "timestamp": snapshot["timestamp"]},
        "/api/overview": snapshot,
        "/api/simu": snapshot["simu"],
        "/api/scada": snapshot["scada"],
        "/api/agc": snapshot["agc"],
    }
    if path not in routes:
        return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)
    return _json_response(routes[path])


def handle_control_path(path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
    if path != "/api/simu/control":
        return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
        action = str(payload.get("action", ""))
        state = SIMU_RUNTIME.apply(action)
    except (ValueError, json.JSONDecodeError) as exc:
        return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
    DATA_SOURCE.snapshot(force_reload=True)
    return _json_response({"ok": True, "state": state})


def resolve_static_path(request_path: str) -> Path:
    parsed_path = unquote(urlparse(request_path).path)
    if parsed_path == "/":
        parsed_path = "/index.html"

    relative = parsed_path.lstrip("/")
    candidate = (WEB_ROOT / relative).resolve()
    if WEB_ROOT not in candidate.parents and candidate != WEB_ROOT:
        raise ValueError(f"path escapes web root: {request_path}")
    if candidate.is_dir():
        candidate = candidate / "index.html"
    return candidate


class WebMonitorHandler(BaseHTTPRequestHandler):
    server_version = "WebMonitor/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            status, headers, body = handle_api_path(parsed.path)
            self._send(status, headers, body)
            return

        try:
            path = resolve_static_path(self.path)
        except ValueError:
            self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
            return

        if not path.exists() or not path.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        headers = {
            "Content-Type": content_type,
            "Cache-Control": "no-cache" if path.suffix == ".html" else "public, max-age=3600",
        }
        self._send(HTTPStatus.OK, headers, path.read_bytes())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        if parsed.path.startswith("/api/"):
            status, headers, response_body = handle_control_path(parsed.path, body)
            self._send(status, headers, response_body)
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, status: int, text: str) -> None:
        self._send(status, {"Content-Type": "text/plain; charset=utf-8"}, text.encode("utf-8"))

    def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(int(status))
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8866) -> None:
    server = ThreadingHTTPServer((host, port), WebMonitorHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the web monitor server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8866, type=int)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
