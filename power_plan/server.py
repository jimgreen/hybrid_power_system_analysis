#!/usr/bin/env python3
"""Static web server and JSON API for the power_plan dashboard."""

from __future__ import annotations

import argparse
from http.cookies import SimpleCookie
import csv
import hashlib
import hmac
import json
import math
import mimetypes
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from contextlib import closing
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, unquote, urlparse
from urllib.request import urlopen

import planning_store


WEB_ROOT = Path(__file__).resolve().parent
DATA_DIR = WEB_ROOT / "data"
VENDOR_DIR = WEB_ROOT / "vendor"
USER_DB_PATH = Path(os.environ.get("POWER_PLAN_USER_DB", WEB_ROOT / "power_plan_users.sqlite3"))
SESSION_COOKIE_NAME = "power_plan_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


class UserStore:
    """SQLite-backed user and session store."""

    def __init__(self, db_path: Path = USER_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            connection.commit()

    def user_count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(self, username: str, password: str, role: str | None = None) -> dict:
        clean_username = self._clean_username(username)
        if len(password or "") < 6:
            raise ValueError("密码长度不能少于6位")
        with self._lock:
            user_role = role or ("admin" if self.user_count() == 0 else "user")
            if user_role not in {"admin", "user"}:
                raise ValueError("用户角色不合法")
            salt, password_hash = self._hash_password(password)
            try:
                with closing(self._connect()) as connection:
                    cursor = connection.execute(
                        """
                        INSERT INTO users (username, password_salt, password_hash, role, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (clean_username, salt, password_hash, user_role, _now_iso()),
                    )
                    connection.commit()
                    user_id = int(cursor.lastrowid)
            except sqlite3.IntegrityError as exc:
                raise ValueError("用户名已存在") from exc
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("用户创建失败")
        return user

    def authenticate(self, username: str, password: str) -> dict:
        clean_username = self._clean_username(username)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, username, password_salt, password_hash, role, created_at FROM users WHERE username = ?",
                (clean_username,),
            ).fetchone()
        if not row or not self._verify_password(password or "", row["password_salt"], row["password_hash"]):
            raise ValueError("用户名或密码错误")
        return self._public_user(row)

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, _now_iso(), now + SESSION_MAX_AGE_SECONDS),
            )
            connection.commit()
        return token

    def delete_session(self, token: str) -> None:
        if not token:
            return
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()

    def user_for_session(self, token: str) -> dict | None:
        if not token:
            return None
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            row = connection.execute(
                """
                SELECT users.id, users.username, users.role, users.created_at
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = ? AND sessions.expires_at > ?
                """,
                (token, now),
            ).fetchone()
            connection.commit()
        return self._public_user(row) if row else None

    def list_users(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY id"
            ).fetchall()
        return [self._public_user(row) for row in rows]

    def get_user_by_id(self, user_id: int) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, username, role, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return self._public_user(row) if row else None

    def update_role(self, user_id: int, role: str) -> dict:
        if role not in {"admin", "user"}:
            raise ValueError("用户角色不合法")
        with closing(self._connect()) as connection:
            connection.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            connection.commit()
        user = self.get_user_by_id(user_id)
        if not user:
            raise FileNotFoundError("用户不存在")
        return user

    def delete_user(self, user_id: int, current_user_id: int | None = None) -> None:
        if current_user_id is not None and int(user_id) == int(current_user_id):
            raise ValueError("不能删除当前登录用户")
        with closing(self._connect()) as connection:
            cursor = connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
            connection.commit()
        if cursor.rowcount == 0:
            raise FileNotFoundError("用户不存在")

    @staticmethod
    def _clean_username(username: str) -> str:
        clean = str(username or "").strip()
        if len(clean) < 2 or len(clean) > 32:
            raise ValueError("用户名长度应为2-32位")
        if any(ord(char) < 32 or char.isspace() for char in clean):
            raise ValueError("用户名不能包含空格或不可见字符")
        return clean

    @staticmethod
    def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
        password_salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(password_salt), 120_000)
        return password_salt, digest.hex()

    @classmethod
    def _verify_password(cls, password: str, salt: str, expected_hash: str) -> bool:
        _, actual_hash = cls._hash_password(password, salt)
        return hmac.compare_digest(actual_hash, expected_hash)

    @staticmethod
    def _public_user(row) -> dict:
        return {
            "id": int(row["id"]),
            "username": row["username"],
            "role": row["role"],
            "created_at": row["created_at"],
        }


class OptimizationRuntime:
    """In-memory runtime state for the planning optimization page."""

    def __init__(self, scheme: str = "") -> None:
        self.status = "待启动"
        self.scheme = str(scheme or "").strip()
        self.start_time = ""
        self.end_time = ""
        self.progress = 0
        self._started_monotonic = 0.0
        self._last_progress_log = -1
        self._logs: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._append_log_unlocked("info", "优化规划待启动")

    def snapshot(self) -> dict:
        with self._lock:
            self._advance_locked()
            return self._payload_unlocked()

    def apply(self, action: str, scheme: str = "") -> dict:
        with self._lock:
            self._advance_locked()
            target_scheme = str(scheme or self.scheme or "未选择方案").strip() or "未选择方案"
            if action == "start":
                if self.status == "运行中":
                    if self.scheme == target_scheme:
                        raise OptimizationStateError("running", f"方案“{target_scheme}”正在运行，无法再次启动")
                    raise OptimizationStateError("running", f"方案“{self.scheme}”正在运行，无法启动方案“{target_scheme}”")
                self.status = "运行中"
                self.scheme = target_scheme
                self.start_time = _now_text()
                self.end_time = ""
                self.progress = 0
                self._started_monotonic = time.monotonic()
                self._last_progress_log = -1
                self._append_log_unlocked("ok", f"启动优化规划，方案：{self.scheme}")
                self._append_log_unlocked("info", "后台优化规划程序已启动")
            elif action == "stop":
                if self.status != "运行中" or self.scheme != target_scheme:
                    raise OptimizationStateError("not_running", f"方案“{target_scheme}”没有运行")
                self.status = "已停止"
                self.end_time = _now_text()
                self._append_log_unlocked("warn", "停止优化规划")
            else:
                raise ValueError(f"unknown optimization action: {action}")
            return self._payload_unlocked()

    def _advance_locked(self) -> None:
        if self.status != "运行中":
            return
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        self.progress = min(100, int(elapsed * 3))
        progress_bucket = min(100, (self.progress // 10) * 10)
        if progress_bucket >= 10 and progress_bucket != self._last_progress_log:
            self._last_progress_log = progress_bucket
            self._append_log_unlocked("info", f"优化迭代进度 {progress_bucket}%")
        if self.progress >= 100:
            self.status = "已完成"
            self.end_time = _now_text()
            self._append_log_unlocked("ok", "优化规划完成")

    def _payload_unlocked(self) -> dict:
        return {
            "status": self.status,
            "scheme": self.scheme,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "progress": self.progress,
            "metrics": self._metrics_unlocked(),
            "results": self._results_unlocked(),
            "logs": list(self._logs),
        }

    def _metrics_unlocked(self) -> list[dict]:
        if self.status == "待启动":
            cost: float | str = "-"
            green_ratio: float | str = "-"
        else:
            cost = round(max(0.42, 0.78 - self.progress * 0.002), 3)
            green_ratio = round(min(92.0, 52.0 + self.progress * 0.34), 1)
        return [
            {"label": "当前状态", "value": self.status, "unit": ""},
            {"label": "启动时刻", "value": self.start_time or "-", "unit": ""},
            {"label": "结束时刻", "value": self.end_time or "-", "unit": ""},
            {"label": "度电成本", "value": cost, "unit": "元/kWh"},
            {"label": "绿电占比", "value": green_ratio, "unit": "%"},
        ]

    def _results_unlocked(self) -> dict:
        cost = round(max(0.42, 0.78 - self.progress * 0.002), 3)
        green_ratio = round(min(92.0, 52.0 + self.progress * 0.34), 1)
        reserve_margin = round(max(12.0, 28.0 - self.progress * 0.05), 1)
        frequency_margin = round(1.08 + min(0.12, self.progress * 0.001), 3)
        curtailed_ratio = round(max(1.0, 9.0 - self.progress * 0.04), 1)
        diesel_energy = round(max(480, 1580 - self.progress * 7.2), 1)
        wind_energy = round(2260 + self.progress * 8.6, 1)
        pv_energy = round(1880 + self.progress * 7.4, 1)
        storage_energy = round(720 + self.progress * 3.5, 1)
        hydrogen_production = round(148 + self.progress * 1.1, 1)
        fuel_cell_energy = round(420 + self.progress * 2.6, 1)
        construction_cost = 2180.0
        operation_cost = round(980 - self.progress * 1.4, 1)
        renewable_energy = round(wind_energy + pv_energy + fuel_cell_energy, 1)
        load_energy_kwh = 6840.0 * 1000
        diesel_energy_kwh = diesel_energy * 1000
        wind_energy_kwh = wind_energy * 1000
        pv_energy_kwh = pv_energy * 1000
        storage_energy_kwh = storage_energy * 1000
        storage_charge_energy_kwh = round(storage_energy_kwh / 0.9, 1)
        fuel_cell_energy_kwh = fuel_cell_energy * 1000
        hydrogen_production_nm3 = hydrogen_production * 10000
        hydrogen_production_energy_kwh = hydrogen_production_nm3 / 0.2
        green_daily = self._green_daily_curve_unlocked(
            load_energy_kwh,
            diesel_energy_kwh,
            wind_energy_kwh,
            pv_energy_kwh,
            fuel_cell_energy_kwh,
            hydrogen_production_energy_kwh,
            storage_energy_kwh,
            storage_charge_energy_kwh,
        )
        safety_daily = self._safety_daily_frequency_curve_unlocked()
        highest_frequency = max(point["frequency_max"] for point in safety_daily)
        lowest_frequency = min(point["frequency_min"] for point in safety_daily)
        upward_disturbance = round(max(860.0, 1480.0 - self.progress * 5.8), 1)
        downward_disturbance = round(max(720.0, 1310.0 - self.progress * 5.1), 1)
        frequency_risk_hours = max(0, int(round(68 - self.progress * 0.55)))
        return {
            "overview_tables": [
                {
                    "title": "规划结果",
                    "rows": [
                        {"设备类型": "柴发", "设计台数": 2, "单台容量": 320, "总容量": 640, "单位": "kW"},
                        {"设备类型": "风机", "设计台数": 6, "单台容量": 120, "总容量": 720, "单位": "kW"},
                        {"设备类型": "光伏", "设计台数": 18, "单台容量": 55, "总容量": 990, "单位": "kW"},
                        {"设备类型": "储能", "设计台数": 4, "单台容量": 250, "总容量": 1000, "单位": "kWh"},
                        {"设备类型": "电制氢", "设计台数": 2, "单台容量": 180, "总容量": 360, "单位": "kW"},
                        {"设备类型": "储氢罐", "设计台数": 3, "单台容量": 420, "总容量": 1260, "单位": "Nm3"},
                        {"设备类型": "燃料电池", "设计台数": 2, "单台容量": 160, "总容量": 320, "单位": "kW"},
                    ],
                },
                {
                    "title": "规划年指标",
                    "rows": [
                        {"指标": "柴发总容量", "数值": 640, "单位": "kW"},
                        {"指标": "风电总容量", "数值": 720, "单位": "kW"},
                        {"指标": "光伏总容量", "数值": 990, "单位": "kW"},
                        {"指标": "氢能总容量", "数值": 320, "单位": "kW"},
                        {"指标": "储能总容量", "数值": 1000, "单位": "kWh"},
                        {"指标": "负荷总电量", "数值": 6840.0, "单位": "MWh"},
                        {"指标": "柴发总电量", "数值": diesel_energy, "单位": "MWh"},
                        {"指标": "风能总电量", "数值": wind_energy, "单位": "MWh"},
                        {"指标": "光伏总电量", "数值": pv_energy, "单位": "MWh"},
                        {"指标": "弃电量", "数值": round(420 - self.progress * 2.1, 1), "单位": "MWh"},
                        {"指标": "储能总电量", "数值": storage_energy, "单位": "MWh"},
                        {"指标": "制氢总量", "数值": hydrogen_production, "单位": "万Nm3"},
                        {"指标": "燃料电池发电量", "数值": fuel_cell_energy, "单位": "MWh"},
                        {"指标": "总成本", "数值": round(3280 - self.progress * 3.2, 1), "单位": "万元"},
                        {"指标": "绿电占比", "数值": green_ratio, "单位": "%"},
                        {"指标": "频率风险点", "数值": max(0, 6 - self.progress // 18), "单位": "个"},
                    ],
                },
            ],
            "overview_disks": [
                {
                    "title": "成本构成",
                    "left_label": "运行成本",
                    "left_value": operation_cost,
                    "right_label": "建设成本",
                    "right_value": construction_cost,
                    "unit": "万元",
                },
                {
                    "title": "电量构成",
                    "left_label": "柴发电量",
                    "left_value": diesel_energy,
                    "right_label": "新能源电量",
                    "right_value": renewable_energy,
                    "unit": "MWh",
                },
            ],
            "overview": [
                {"指标": "度电成本", "数值": cost, "单位": "元/kWh", "说明": "基于当前候选方案的综合成本估计"},
                {"指标": "绿电占比", "数值": green_ratio, "单位": "%", "说明": "风光与氢储供电占比"},
                {"指标": "优化进度", "数值": self.progress, "单位": "%", "说明": self.status},
            ],
            "green": [
                {"指标": "风电消纳率", "数值": round(min(98.0, 82.0 + self.progress * 0.08), 1), "单位": "%", "说明": "风机出力消纳水平"},
                {"指标": "光伏消纳率", "数值": round(min(98.5, 84.0 + self.progress * 0.07), 1), "单位": "%", "说明": "光伏出力消纳水平"},
                {"指标": "弃电率", "数值": round(max(1.0, 9.0 - self.progress * 0.04), 1), "单位": "%", "说明": "新能源未利用电量占比"},
            ],
            "green_table": [
                {"指标": "负荷总电量", "数值": round(load_energy_kwh, 1), "单位": "kWh"},
                {"指标": "柴发总电量", "数值": round(diesel_energy_kwh, 1), "单位": "kWh"},
                {"指标": "风机总发电量", "数值": round(wind_energy_kwh, 1), "单位": "kWh"},
                {"指标": "光伏总发电量", "数值": round(pv_energy_kwh, 1), "单位": "kWh"},
                {"指标": "电储总发电量", "数值": round(storage_energy_kwh, 1), "单位": "kWh"},
                {"指标": "氢储总发电量", "数值": round(fuel_cell_energy_kwh, 1), "单位": "kWh"},
                {"指标": "新能源总弃电量", "数值": curtailed_ratio, "单位": "%"},
                {"指标": "柴油消耗", "数值": round(diesel_energy * 0.24, 1), "单位": "吨"},
                {"指标": "制氢总量", "数值": round(hydrogen_production_nm3, 1), "单位": "Nm3"},
            ],
            "safety": [
                {"指标": "备用裕度", "数值": reserve_margin, "单位": "%", "说明": "负荷扰动后的可用备用"},
                {"指标": "频率安全裕度", "数值": frequency_margin, "单位": "p.u.", "说明": "频率约束裕度"},
                {"指标": "N-1校核", "数值": "通过" if self.progress >= 35 else "计算中", "单位": "", "说明": "新能源 N-1 约束校核"},
            ],
            "safety_table": [
                {"指标": "向上扰动最大量", "数值": upward_disturbance, "单位": "kW"},
                {"指标": "向下扰动最大量", "数值": downward_disturbance, "单位": "kW"},
                {"指标": "最高频率", "数值": highest_frequency, "单位": "Hz"},
                {"指标": "最低频率", "数值": lowest_frequency, "单位": "Hz"},
                {"指标": "频率安全风险小时数", "数值": frequency_risk_hours, "单位": "h"},
            ],
            "curves": {
                "overview": [
                    {"label": "20%", "value": round(cost + 0.08, 3)},
                    {"label": "40%", "value": round(cost + 0.05, 3)},
                    {"label": "60%", "value": round(cost + 0.03, 3)},
                    {"label": "80%", "value": round(cost + 0.01, 3)},
                    {"label": "当前", "value": cost},
                ],
                "green": [
                    {"label": "风电", "value": round(min(98.0, 82.0 + self.progress * 0.08), 1)},
                    {"label": "光伏", "value": round(min(98.5, 84.0 + self.progress * 0.07), 1)},
                    {"label": "氢储", "value": round(min(90.0, 66.0 + self.progress * 0.09), 1)},
                    {"label": "总体", "value": green_ratio},
                ],
                "green_daily": green_daily,
                "safety": [
                    {"label": "备用", "value": reserve_margin},
                    {"label": "频率", "value": round(frequency_margin * 10, 2)},
                    {"label": "N-1", "value": 100 if self.progress >= 35 else max(10, self.progress)},
                ],
                "safety_daily": safety_daily,
            },
        }

    def _green_daily_curve_unlocked(
        self,
        load_total: float,
        diesel_total: float,
        wind_total: float,
        pv_total: float,
        hydrogen_total: float,
        hydrogen_production_total: float,
        storage_discharge_total: float,
        storage_charge_total: float,
    ) -> list[dict[str, float | int]]:
        days = list(range(1, 366))

        def seasonal(day: int, phase_shift: float, amplitude: float, second: float = 0.0) -> float:
            phase = 2 * math.pi * (day - 1) / 365
            return max(0.05, 1 + amplitude * math.sin(phase + phase_shift) + second * math.sin(phase * 2 + phase_shift / 2))

        raw = {
            "load_energy": [seasonal(day, -0.25, 0.12, 0.05) for day in days],
            "diesel_energy": [seasonal(day, 0.8, 0.18, 0.03) for day in days],
            "wind_energy": [seasonal(day, 1.25, 0.22, 0.05) for day in days],
            "pv_energy": [seasonal(day, -1.15, 0.34, 0.04) for day in days],
            "hydrogen_energy": [seasonal(day, 0.35, 0.16, 0.03) for day in days],
            "hydrogen_production_energy": [seasonal(day, -0.75, 0.2, 0.04) for day in days],
            "storage_discharge_energy": [seasonal(day, 0.1, 0.11, 0.06) for day in days],
            "storage_charge_energy": [seasonal(day, -0.9, 0.18, 0.05) for day in days],
        }
        totals = {
            "load_energy": load_total,
            "diesel_energy": diesel_total,
            "wind_energy": wind_total,
            "pv_energy": pv_total,
            "hydrogen_energy": hydrogen_total,
            "hydrogen_production_energy": hydrogen_production_total,
            "storage_discharge_energy": storage_discharge_total,
            "storage_charge_energy": storage_charge_total,
        }
        scaled: dict[str, list[float]] = {}
        for key, values in raw.items():
            raw_total = sum(values) or 1
            scaled[key] = [round(totals[key] * value / raw_total, 1) for value in values]

        return [
            {
                "day": day,
                "diesel_energy": scaled["diesel_energy"][index],
                "wind_energy": scaled["wind_energy"][index],
                "pv_energy": scaled["pv_energy"][index],
                "hydrogen_energy": scaled["hydrogen_energy"][index],
                "storage_discharge_energy": scaled["storage_discharge_energy"][index],
                "load_energy": scaled["load_energy"][index],
                "hydrogen_production_energy": scaled["hydrogen_production_energy"][index],
                "storage_charge_energy": scaled["storage_charge_energy"][index],
            }
            for index, day in enumerate(days)
        ]

    def _safety_daily_frequency_curve_unlocked(self) -> list[dict[str, float | int]]:
        days = list(range(1, 366))
        max_deviation_base = max(0.07, 0.22 - self.progress * 0.0012)
        min_deviation_base = max(0.06, 0.2 - self.progress * 0.001)
        points = []
        for day in days:
            phase = 2 * math.pi * (day - 1) / 365
            frequency_max = 50 + max_deviation_base + 0.035 * math.sin(phase + 0.7) + 0.012 * math.sin(phase * 2.4)
            frequency_min = 50 - min_deviation_base - 0.032 * math.cos(phase - 0.25) - 0.01 * math.sin(phase * 2.1)
            points.append(
                {
                    "day": day,
                    "frequency_max": round(frequency_max, 3),
                    "frequency_min": round(frequency_min, 3),
                }
            )
        return points

    def _append_log_unlocked(self, level: str, message: str) -> None:
        self._logs.append({"time": _now_text(), "level": level, "message": message})
        if len(self._logs) > 120:
            del self._logs[:-120]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class OptimizationStateError(RuntimeError):
    """Raised when optimization start/stop violates the current runtime state."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class OptimizationRuntimeManager:
    """Holds independent optimization runtimes for multiple schemes."""

    def __init__(self) -> None:
        self._runtimes: dict[str, OptimizationRuntime] = {}
        self._lock = threading.Lock()

    def snapshot(self, scheme: str = "") -> dict:
        runtime = self._runtime_for_scheme(scheme)
        payload = runtime.snapshot()
        payload["running_schemes"] = self.running_schemes()
        return payload

    def apply(self, action: str, scheme: str = "") -> dict:
        runtime = self._runtime_for_scheme(scheme)
        payload = runtime.apply(action, scheme=self._scheme_name(scheme))
        payload["running_schemes"] = self.running_schemes()
        return payload

    def running_schemes(self) -> list[str]:
        with self._lock:
            runtimes = list(self._runtimes.items())
        running = []
        for scheme, runtime in runtimes:
            if runtime.snapshot()["status"] == "运行中":
                running.append(scheme)
        return running

    def _runtime_for_scheme(self, scheme: str = "") -> OptimizationRuntime:
        name = self._scheme_name(scheme)
        with self._lock:
            if name not in self._runtimes:
                self._runtimes[name] = OptimizationRuntime(scheme=name)
            return self._runtimes[name]

    @staticmethod
    def _scheme_name(scheme: str = "") -> str:
        return str(scheme or "未选择方案").strip() or "未选择方案"


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
        return {
            "system": "考察站风-光-氢-储-柴联合规划系统",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": self._load_overview_summary(),
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
        return {
            "system": "考察站风-光-氢-储-柴联合规划系统",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": self._load_overview_summary(),
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
    return SimuRuntime()


SIMU_RUNTIME = _load_initial_simu_runtime()
OPTIMIZATION_RUNTIME = OptimizationRuntimeManager()
DATA_SOURCE = CsvDataSource()
PLANNING_STORE = planning_store.PlanningStore()
USER_STORE = UserStore()


def build_snapshot(force_reload: bool = False) -> dict:
    """Build a snapshot from CSV files, reloading periodically."""
    return DATA_SOURCE.snapshot(force_reload=force_reload)


def _json_response(payload: dict, status: int = 200, extra_headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
    }
    if extra_headers:
        headers.update(extra_headers)
    return status, headers, body


def _read_json_body(body: bytes) -> dict:
    try:
        return json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("请求体不是合法 JSON") from exc


def _session_cookie(token: str, max_age: int = SESSION_MAX_AGE_SECONDS) -> str:
    return f"{SESSION_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"


def _expired_session_cookie() -> str:
    return f"{SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"


def _session_token_from_cookie(cookie_header: str | None) -> str:
    if not cookie_header:
        return ""
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return ""
    morsel = cookie.get(SESSION_COOKIE_NAME)
    return morsel.value if morsel else ""


def handle_auth_api_path(path: str, method: str, body: bytes = b"", token: str = "") -> tuple[int, dict[str, str], bytes]:
    current_user = USER_STORE.user_for_session(token)
    try:
        if path == "/api/auth/me" and method == "GET":
            if not current_user:
                return _json_response({"error": "unauthorized", "message": "请先登录"}, HTTPStatus.UNAUTHORIZED)
            return _json_response({"user": current_user})
        if path == "/api/auth/register" and method == "POST":
            payload = _read_json_body(body)
            user = USER_STORE.create_user(str(payload.get("username", "")), str(payload.get("password", "")))
            session_token = USER_STORE.create_session(user["id"])
            return _json_response({"ok": True, "user": user}, extra_headers={"Set-Cookie": _session_cookie(session_token)})
        if path == "/api/auth/login" and method == "POST":
            payload = _read_json_body(body)
            user = USER_STORE.authenticate(str(payload.get("username", "")), str(payload.get("password", "")))
            session_token = USER_STORE.create_session(user["id"])
            return _json_response({"ok": True, "user": user}, extra_headers={"Set-Cookie": _session_cookie(session_token)})
        if path == "/api/auth/logout" and method == "POST":
            USER_STORE.delete_session(token)
            return _json_response({"ok": True}, extra_headers={"Set-Cookie": _expired_session_cookie()})
    except ValueError as exc:
        return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
    return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)


def handle_users_api_path(path: str, method: str, body: bytes, current_user: dict | None) -> tuple[int, dict[str, str], bytes]:
    if not current_user:
        return _json_response({"error": "unauthorized", "message": "请先登录"}, HTTPStatus.UNAUTHORIZED)
    if current_user.get("role") != "admin":
        return _json_response({"error": "forbidden", "message": "需要管理员权限"}, HTTPStatus.FORBIDDEN)
    try:
        if path == "/api/users" and method == "GET":
            return _json_response({"users": USER_STORE.list_users()})
        if path.startswith("/api/users/"):
            user_id = int(path.rsplit("/", 1)[1])
            if method == "PUT":
                payload = _read_json_body(body)
                return _json_response({"user": USER_STORE.update_role(user_id, str(payload.get("role", "")))})
            if method == "DELETE":
                USER_STORE.delete_user(user_id, current_user_id=current_user["id"])
                return _json_response({"ok": True})
    except FileNotFoundError as exc:
        return _json_response({"error": "not_found", "message": str(exc)}, HTTPStatus.NOT_FOUND)
    except (TypeError, ValueError) as exc:
        return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
    return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)


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


def handle_api_path(path: str, query: str = "") -> tuple[int, dict[str, str], bytes]:
    parsed_api_path = urlparse(path)
    if parsed_api_path.query and not query:
        query = parsed_api_path.query
        path = parsed_api_path.path
    query_params = parse_qs(query)
    if path.startswith("/api/planning/"):
        return handle_planning_api_path(path, "GET", b"")
    if path == "/api/optimization/status":
        scheme = query_params.get("scheme", [""])[0]
        return _json_response(OPTIMIZATION_RUNTIME.snapshot(scheme=scheme))
    snapshot = build_snapshot()
    routes = {
        "/api/health": {"ok": True, "timestamp": snapshot["timestamp"]},
        "/api/overview": snapshot,
    }
    if path not in routes:
        return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)
    return _json_response(routes[path])


def handle_control_path(path: str, body: bytes) -> tuple[int, dict[str, str], bytes]:
    if path == "/api/optimization/control":
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            action = str(payload.get("action", ""))
            scheme = str(payload.get("scheme", ""))
            state = OPTIMIZATION_RUNTIME.apply(action, scheme=scheme)
        except OptimizationStateError as exc:
            return _json_response({"error": exc.code, "message": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json_response({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
        return _json_response({"ok": True, "state": state})
    return _json_response({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)


def _unauthorized_response() -> tuple[int, dict[str, str], bytes]:
    return _json_response({"error": "unauthorized", "message": "请先登录"}, HTTPStatus.UNAUTHORIZED)


def _redirect_response(location: str) -> tuple[int, dict[str, str], bytes]:
    return HTTPStatus.FOUND, {"Location": location, "Content-Type": "text/plain; charset=utf-8"}, b""


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

    def _current_user(self) -> dict | None:
        token = _session_token_from_cookie(self.headers.get("Cookie"))
        return USER_STORE.user_for_session(token)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            token = _session_token_from_cookie(self.headers.get("Cookie"))
            current_user = USER_STORE.user_for_session(token)
            if parsed.path.startswith("/api/auth/"):
                status, headers, body = handle_auth_api_path(parsed.path, "GET", b"", token)
                self._send(status, headers, body)
                return
            if parsed.path.startswith("/api/users"):
                status, headers, body = handle_users_api_path(parsed.path, "GET", b"", current_user)
                self._send(status, headers, body)
                return
            if parsed.path != "/api/health" and not current_user:
                status, headers, body = _unauthorized_response()
                self._send(status, headers, body)
                return
            status, headers, body = handle_api_path(parsed.path, parsed.query)
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

        if path.suffix == ".html":
            current_user = self._current_user()
            public_pages = {"login.html", "register.html"}
            if path.name not in public_pages and not current_user:
                next_path = parsed.path or "/index.html"
                self._send(*_redirect_response(f"/login.html?next={urlencode({'next': next_path})[5:]}"))
                return
            if path.name == "users.html" and current_user and current_user.get("role") != "admin":
                self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if path.name in public_pages and current_user:
                self._send(*_redirect_response("/index.html"))
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
            token = _session_token_from_cookie(self.headers.get("Cookie"))
            current_user = USER_STORE.user_for_session(token)
            if parsed.path.startswith("/api/auth/"):
                status, headers, response_body = handle_auth_api_path(parsed.path, "POST", body, token)
                self._send(status, headers, response_body)
                return
            if parsed.path.startswith("/api/users"):
                status, headers, response_body = handle_users_api_path(parsed.path, "POST", body, current_user)
                self._send(status, headers, response_body)
                return
            if not current_user:
                status, headers, response_body = _unauthorized_response()
                self._send(status, headers, response_body)
                return
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
        token = _session_token_from_cookie(self.headers.get("Cookie"))
        current_user = USER_STORE.user_for_session(token)
        if parsed.path.startswith("/api/users"):
            status, headers, response_body = handle_users_api_path(parsed.path, "PUT", body, current_user)
            self._send(status, headers, response_body)
            return
        if parsed.path.startswith("/api/") and not current_user:
            status, headers, response_body = _unauthorized_response()
            self._send(status, headers, response_body)
            return
        if parsed.path.startswith("/api/planning/"):
            status, headers, response_body = handle_planning_api_path(parsed.path, "PUT", body)
            self._send(status, headers, response_body)
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not Found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        token = _session_token_from_cookie(self.headers.get("Cookie"))
        current_user = USER_STORE.user_for_session(token)
        if parsed.path.startswith("/api/users"):
            status, headers, response_body = handle_users_api_path(parsed.path, "DELETE", b"", current_user)
            self._send(status, headers, response_body)
            return
        if parsed.path.startswith("/api/") and not current_user:
            status, headers, response_body = _unauthorized_response()
            self._send(status, headers, response_body)
            return
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


