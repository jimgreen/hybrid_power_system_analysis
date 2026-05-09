#!/usr/bin/env python3
"""Static web server and JSON API for the power_plan dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import threading
import time
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, unquote, urlparse
from urllib.request import urlopen

import planning_store


WEB_ROOT = Path(__file__).resolve().parent
DATA_DIR = WEB_ROOT / "data"
VENDOR_DIR = WEB_ROOT / "vendor"
NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
AMAP_WEB_SERVICE_KEY = os.environ.get("POWER_PLAN_AMAP_KEY") or os.environ.get("AMAP_WEB_SERVICE_KEY") or os.environ.get("AMAP_KEY")
NASA_POWER_PARAMETERS = {
    "wind_speed": "WS10M",
    "solar_irradiance": "ALLSKY_SFC_SW_DWN",
    "temperature": "T2M",
}
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
DB_CONFIG = {
    "host": os.environ.get("POWER_PLAN_DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("POWER_PLAN_DB_PORT", "3306")),
    "user": os.environ.get("POWER_PLAN_DB_USER", "root"),
    "password": os.environ.get("POWER_PLAN_DB_PASSWORD", "scadaems"),
    "database": os.environ.get("POWER_PLAN_DB_NAME", "scadaems"),
    "charset": "utf8mb4",
}


def _coerce_value(value: str) -> float | int | str:
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
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


def _mysql_connector(config: dict):
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("缺少 MySQL 驱动，请先安装: python -m pip install PyMySQL") from exc
    return pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        charset=config.get("charset", "utf8mb4"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


class MySqlDataSource:
    """Periodically reload dashboard data from MySQL."""

    def __init__(
        self,
        config: dict | None = None,
        reload_interval: float = 1.0,
        connector_factory=_mysql_connector,
    ) -> None:
        self.config = config or DB_CONFIG
        self.reload_interval = reload_interval
        self.connector_factory = connector_factory
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

    def save_simu_state(self, state: dict) -> None:
        sql = """
            UPDATE simu_state
            SET sim_time = %s, speed = %s, status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """
        self._execute(sql, (state["sim_time"], state["speed"], state["status"]), commit=True)

    def _connect(self):
        return self.connector_factory(self.config)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                if rows is None:
                    return []
                if isinstance(rows, dict):
                    return [rows]
                return list(rows)
            finally:
                cursor.close()
        finally:
            connection.close()

    def _query_one(self, sql: str, params: tuple = ()) -> dict | None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                return cursor.fetchone()
            finally:
                cursor.close()
        finally:
            connection.close()

    def _execute(self, sql: str, params: tuple = (), commit: bool = False) -> None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                if commit:
                    connection.commit()
            finally:
                cursor.close()
        finally:
            connection.close()

    def _load_snapshot(self) -> dict:
        metrics = self._query("SELECT page, label, value, unit, status FROM metrics ORDER BY page, display_order, id")
        alarms = self._query("SELECT page, time, object, message, status FROM alarms ORDER BY page, display_order, id")
        page_summary = self._query("SELECT page, label, value, status FROM page_summary ORDER BY page, display_order, id")
        agc_units = [
            {
                "name": row.get("name", ""),
                "percent": _coerce_value(str(row.get("percent", ""))),
                "power": _coerce_value(str(row.get("power", ""))),
                "unit": row.get("unit", ""),
            }
            for row in self._query("SELECT name, percent, power, unit FROM agc_units ORDER BY display_order, id")
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
                    "bars": self._load_label_values("SELECT label, value, unit FROM simu_bars ORDER BY display_order, id"),
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
                "charts": {
                    "columns": self._load_label_values(
                        "SELECT label, value, unit FROM scada_columns ORDER BY display_order, id"
                    )
                },
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
            row.get("key", ""): _coerce_value(str(row.get("value", "")))
            for row in self._query("SELECT `key`, value, unit FROM overview_summary ORDER BY display_order, id")
            if row.get("key")
        }

    def _by_page(self, rows: list[dict], page: str) -> list[dict]:
        return [row for row in rows if row.get("page") == page]

    def _load_metrics(self, rows: list[dict], page: str) -> list[dict]:
        return [
            _metric(
                row.get("label", ""),
                _coerce_value(str(row.get("value", ""))),
                row.get("unit", ""),
                row.get("status", "normal"),
            )
            for row in self._by_page(rows, page)
        ]

    def _load_alarms(self, rows: list[dict], page: str) -> list[dict]:
        return [
            {
                "time": row.get("time", ""),
                "object": row.get("object", ""),
                "message": row.get("message", ""),
                "status": row.get("status", ""),
            }
            for row in self._by_page(rows, page)
        ]

    def _load_page_summary(self, rows: list[dict], page: str) -> list[dict]:
        return [
            _summary(row.get("label", ""), row.get("value", ""), row.get("status", "normal"))
            for row in self._by_page(rows, page)
        ]

    def _load_label_values(self, sql: str) -> list[dict]:
        return [
            {
                "label": row.get("label", ""),
                "value": _coerce_value(str(row.get("value", ""))),
                "unit": row.get("unit", ""),
            }
            for row in self._query(sql)
        ]

    def _load_topology(self) -> list[dict]:
        return [
            {"id": row.get("id", ""), "status": row.get("status", "normal"), "value": row.get("value", "")}
            for row in self._query("SELECT id, status, value FROM simu_topology ORDER BY display_order, id")
        ]

    def _load_simu_daily_curves(self) -> list[dict]:
        rows = self._query(
            "SELECT hour, wind_speed, temperature, solar_irradiance, load_value FROM simu_daily_curves ORDER BY hour"
        )
        specs = [
            ("wind_speed", "风速", "m/s"),
            ("temperature", "温度", "℃"),
            ("solar_irradiance", "太阳辐射", "W/m²"),
            ("load_value", "负荷", "kW"),
        ]
        curves = []
        for key, name, unit in specs:
            points = [
                {"hour": _coerce_value(str(row.get("hour", ""))), "value": _coerce_value(str(row.get(key, "")))}
                for row in rows
            ]
            curves.append({"key": key, "name": name, "unit": unit, "points": points})
        return curves

    def _load_stations(self) -> list[dict]:
        return [
            {"name": row.get("name", ""), "status": row.get("status", "normal"), "detail": row.get("detail", "")}
            for row in self._query("SELECT name, status, detail FROM scada_stations ORDER BY display_order, id")
        ]

    def _load_agc_reserve(self) -> dict:
        row = self._query_one("SELECT score, up, down, response, cycle FROM agc_reserve ORDER BY id LIMIT 1")
        if not row:
            return {"score": 0, "up": 0, "down": 0, "response": 0, "cycle": 0}
        return {
            "score": _coerce_value(str(row.get("score", ""))),
            "up": _coerce_value(str(row.get("up", ""))),
            "down": _coerce_value(str(row.get("down", ""))),
            "response": _coerce_value(str(row.get("response", ""))),
            "cycle": _coerce_value(str(row.get("cycle", ""))),
        }


def _load_initial_simu_runtime() -> SimuRuntime:
    try:
        row = MySqlDataSource(reload_interval=0)._query_one("SELECT sim_time, speed, status FROM simu_state WHERE id = 1")
    except Exception:
        row = None
    if not row:
        return SimuRuntime()
    return SimuRuntime(
        initial_time=row.get("sim_time", "00:00"),
        speed=float(_coerce_value(row.get("speed", "1.0")) or 1.0),
        status=row.get("status", "STOPPED"),
    )


SIMU_RUNTIME = _load_initial_simu_runtime()
DATA_SOURCE = MySqlDataSource()
PLANNING_STORE = planning_store.PlanningStore()


def build_snapshot(force_reload: bool = False) -> dict:
    """Build a snapshot from CSV files, reloading periodically."""
    return DATA_SOURCE.snapshot(force_reload=force_reload)


def _json_response(payload: dict, status: int = 200) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
    }, body


def _read_json_body(body: bytes) -> dict:
    try:
        return json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("请求体不是合法 JSON") from exc


class WeatherHistoryError(RuntimeError):
    """Raised when historical weather data cannot be fetched or parsed."""


class GeocodingError(RuntimeError):
    """Raised when a place name cannot be resolved to coordinates."""


def geocode_place_name(place: str) -> dict:
    query_text = str(place or "").strip()
    if not query_text:
        raise ValueError("地名不能为空")
    errors: list[str] = []
    providers = []
    if AMAP_WEB_SERVICE_KEY:
        providers.append(geocode_with_amap)
    providers.extend([geocode_with_open_meteo, geocode_with_nominatim])
    for provider in providers:
        try:
            return provider(query_text)
        except GeocodingError as exc:
            errors.append(str(exc))
    raise GeocodingError("；".join(errors) or "未找到该地名对应的经纬度坐标")


def geocode_with_amap(place: str) -> dict:
    query = urlencode({"address": place, "output": "JSON", "key": AMAP_WEB_SERVICE_KEY})
    url = f"{AMAP_GEOCODING_URL}?{query}"
    try:
        with urlopen_with_user_agent(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise GeocodingError(f"高德地名解析接口返回错误: HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise GeocodingError(f"高德地名解析接口连接失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeocodingError("高德地名解析接口返回内容不是合法 JSON") from exc
    if str(data.get("status")) != "1":
        raise GeocodingError(f"高德地名解析失败: {data.get('info') or data.get('infocode') or '未知错误'}")
    geocodes = data.get("geocodes", [])
    if not geocodes:
        raise GeocodingError("高德未找到该地名对应的经纬度坐标")
    first = geocodes[0]
    location = str(first.get("location", ""))
    try:
        longitude_text, latitude_text = location.split(",", 1)
        longitude = float(longitude_text)
        latitude = float(latitude_text)
    except (ValueError, AttributeError) as exc:
        raise GeocodingError("高德地名解析结果缺少有效经纬度") from exc
    display_parts = [
        first.get("formatted_address"),
        first.get("province"),
        first.get("city"),
        first.get("district"),
    ]
    display_name = "，".join(str(part) for part in display_parts if part and part != [])
    return {
        "place": place,
        "display_name": display_name or place,
        "latitude": latitude,
        "longitude": longitude,
        "source": "高德地图 Web 服务地理编码 API",
    }


def geocode_with_open_meteo(place: str) -> dict:
    query = urlencode({"name": place, "count": 1, "language": "zh", "format": "json"})
    url = f"{OPEN_METEO_GEOCODING_URL}?{query}"
    try:
        with urlopen_with_user_agent(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise GeocodingError(f"Open-Meteo 地名解析接口返回错误: HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise GeocodingError(f"Open-Meteo 地名解析接口连接失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeocodingError("Open-Meteo 地名解析接口返回内容不是合法 JSON") from exc
    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        raise GeocodingError("Open-Meteo 未找到该地名对应的经纬度坐标")
    first = results[0]
    try:
        latitude = float(first["latitude"])
        longitude = float(first["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingError("Open-Meteo 地名解析结果缺少有效经纬度") from exc
    display_parts = [
        first.get("name"),
        first.get("admin2"),
        first.get("admin1"),
        first.get("country"),
    ]
    display_name = "，".join(str(part) for part in display_parts if part)
    return {
        "place": place,
        "display_name": display_name or place,
        "latitude": latitude,
        "longitude": longitude,
        "source": "Open-Meteo Geocoding API",
    }


def geocode_with_nominatim(place: str) -> dict:
    query = urlencode({"q": place, "format": "json", "limit": 1, "accept-language": "zh-CN"})
    url = f"{NOMINATIM_SEARCH_URL}?{query}"
    try:
        with urlopen_with_user_agent(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise GeocodingError(f"Nominatim 地名解析接口返回错误: HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise GeocodingError(f"Nominatim 地名解析接口连接失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeocodingError("Nominatim 地名解析接口返回内容不是合法 JSON") from exc
    if not isinstance(data, list) or not data:
        raise GeocodingError("Nominatim 未找到该地名对应的经纬度坐标")
    first = data[0]
    try:
        latitude = float(first["lat"])
        longitude = float(first["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingError("Nominatim 地名解析结果缺少有效经纬度") from exc
    return {
        "place": place,
        "display_name": first.get("display_name", place),
        "latitude": latitude,
        "longitude": longitude,
        "source": "OpenStreetMap Nominatim",
    }


def urlopen_with_user_agent(url: str, timeout: int):
    from urllib.request import Request

    request = Request(url, headers={"User-Agent": "power-plan-local-web/1.0"})
    return urlopen(request, timeout=timeout)


def fetch_weather_history(latitude: float, longitude: float, year: int) -> dict:
    latitude, longitude, year = validate_weather_history_inputs(latitude, longitude, year)
    query = urlencode(
        {
            "parameters": ",".join(NASA_POWER_PARAMETERS.values()),
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON",
            "time-standard": "LST",
        }
    )
    url = f"{NASA_POWER_HOURLY_URL}?{query}"
    try:
        with urlopen_with_user_agent(url, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise WeatherHistoryError(f"历史气象数据接口返回错误: HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise WeatherHistoryError(f"历史气象数据接口连接失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise WeatherHistoryError("历史气象数据接口返回内容不是合法 JSON") from exc
    rows = parse_nasa_power_hourly_response(data, year)
    return {
        "source": "NASA POWER Hourly API",
        "source_url": url,
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "rows": rows,
    }


def validate_weather_history_inputs(latitude: float, longitude: float, year: int) -> tuple[float, float, int]:
    try:
        latitude_number = float(latitude)
        longitude_number = float(longitude)
        year_number = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError("经纬度和历史数据年必须为数值") from exc
    current_year = datetime.now().year
    if not -90 <= latitude_number <= 90:
        raise ValueError("纬度范围应为 -90 到 90")
    if not -180 <= longitude_number <= 180:
        raise ValueError("经度范围应为 -180 到 180")
    if year_number < 2001:
        raise ValueError("NASA POWER 小时历史数据年份不能早于 2001")
    if year_number >= current_year:
        raise ValueError(f"历史数据年必须小于当前年 {current_year}")
    return latitude_number, longitude_number, year_number


def parse_nasa_power_hourly_response(data: dict, year: int) -> list[dict]:
    parameters = data.get("properties", {}).get("parameter", {})
    fill_value = data.get("header", {}).get("fill_value", -999)
    missing = [api_name for api_name in NASA_POWER_PARAMETERS.values() if api_name not in parameters]
    if missing:
        raise WeatherHistoryError(f"历史气象数据缺少字段: {', '.join(missing)}")
    wind_values = parameters[NASA_POWER_PARAMETERS["wind_speed"]]
    keys = sorted(key for key in wind_values if str(key).startswith(str(year)) and str(key)[4:8] != "0229")
    rows = []
    for hour_index, key in enumerate(keys, start=1):
        row = {"hour_index": hour_index, "datetime": power_hour_key_to_datetime(str(key))}
        for field, api_name in NASA_POWER_PARAMETERS.items():
            value = parameters.get(api_name, {}).get(key)
            if value in (None, "", fill_value):
                raise WeatherHistoryError(f"历史气象数据在 {key} 缺少 {api_name}")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise WeatherHistoryError(f"历史气象数据在 {key} 的 {api_name} 不是数值") from exc
            row[field] = round(number, 4)
        rows.append(row)
    if len(rows) != 8760:
        raise WeatherHistoryError(f"历史气象数据小时数应为8760，当前为{len(rows)}")
    return rows


def power_hour_key_to_datetime(key: str) -> str:
    return f"{key[0:4]}-{key[4:6]}-{key[6:8]} {key[8:10]}:00"


def handle_planning_api_path(path: str, method: str = "GET", body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
    prefix = "/api/planning/schemes"
    try:
        if path == "/api/planning/map-config" and method == "GET":
            return _json_response({"amap_key": AMAP_WEB_SERVICE_KEY, "preferred_provider": "amap" if AMAP_WEB_SERVICE_KEY else "manual"})
        if path == "/api/planning/weather-history" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(
                fetch_weather_history(payload.get("latitude"), payload.get("longitude"), payload.get("year"))
            )
        if path == "/api/planning/geocode" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(geocode_place_name(str(payload.get("place", ""))))
        if path == prefix and method == "GET":
            return _json_response({"schemes": PLANNING_STORE.list_schemes()})
        if path == prefix and method == "POST":
            payload = _read_json_body(body)
            return _json_response(PLANNING_STORE.create_scheme(str(payload.get("name", ""))))
        if path == f"{prefix}/copy" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(
                PLANNING_STORE.copy_scheme(str(payload.get("source", "")), str(payload.get("target", "")))
            )
        if path == f"{prefix}/rename" and method == "POST":
            payload = _read_json_body(body)
            return _json_response(
                PLANNING_STORE.rename_scheme(str(payload.get("source", "")), str(payload.get("target", "")))
            )
        if path.startswith(f"{prefix}/") and path.endswith("/overview") and method == "GET":
            name = unquote(path[len(prefix) + 1 : -len("/overview")])
            return _json_response(PLANNING_STORE.read_scheme_overview(name))
        if path.startswith(f"{prefix}/") and path.endswith("/time-series") and method == "GET":
            name = unquote(path[len(prefix) + 1 : -len("/time-series")])
            return _json_response(PLANNING_STORE.read_time_series(name))
        if path.startswith(f"{prefix}/"):
            name = unquote(path[len(prefix) + 1 :])
            if method == "GET":
                return _json_response(PLANNING_STORE.read_scheme(name))
            if method == "PUT":
                payload = _read_json_body(body)
                PLANNING_STORE.write_scheme(name, payload)
                return _json_response(PLANNING_STORE.read_scheme(name))
            if method == "DELETE":
                return _json_response(PLANNING_STORE.delete_scheme(name))
    except WeatherHistoryError as exc:
        return _json_response({"error": "weather_history_error", "message": str(exc)}, HTTPStatus.BAD_GATEWAY)
    except GeocodingError as exc:
        return _json_response({"error": "geocoding_error", "message": str(exc)}, HTTPStatus.BAD_GATEWAY)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
    return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)


def handle_api_path(path: str) -> tuple[int, dict[str, str], bytes]:
    if path.startswith("/api/planning/"):
        return handle_planning_api_path(path, "GET", b"")
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
        DATA_SOURCE.save_simu_state(state)
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


class PowerPlanHandler(BaseHTTPRequestHandler):
    server_version = "PowerPlan/1.0"

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
        no_cache_suffixes = {".html", ".css", ".js"}
        headers = {
            "Content-Type": content_type,
            "Cache-Control": "no-cache" if path.suffix in no_cache_suffixes else "public, max-age=3600",
        }
        self._send(HTTPStatus.OK, headers, path.read_bytes())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        if parsed.path.startswith("/api/"):
            if parsed.path.startswith("/api/planning/"):
                status, headers, response_body = handle_planning_api_path(parsed.path, "POST", body)
                self._send(status, headers, response_body)
                return
            status, headers, response_body = handle_control_path(parsed.path, body)
            self._send(status, headers, response_body)
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not Found")

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        if parsed.path.startswith("/api/planning/"):
            status, headers, response_body = handle_planning_api_path(parsed.path, "PUT", body)
            self._send(status, headers, response_body)
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not Found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/planning/"):
            status, headers, response_body = handle_planning_api_path(parsed.path, "DELETE", b"")
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
    server = ThreadingHTTPServer((host, port), PowerPlanHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the power plan server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8866, type=int)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()


