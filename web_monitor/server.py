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
                "charts": {"bars": self._load_label_values("simu_bars.csv")},
                "topology": self._load_topology(),
                "summary": self._load_page_summary(page_summary, "simu"),
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
