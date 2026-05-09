"""XLSX-backed scheme storage for grid-planning parameter maintenance."""

from __future__ import annotations

import re
import shutil
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook


WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEME_ROOT = WEB_ROOT / "planning_schemes"
WORKBOOK_NAME = "parameters.xlsx"

SHEET_SPECS: dict[str, tuple[str, list[str]]] = {
    "time_series": ("8760时序数据", ["hour_index", "datetime", "wind_speed", "solar_irradiance", "load", "temperature"]),
    "diesel_generators": (
        "柴发参数",
        [
            "name",
            "capacity",
            "cost",
            "power_upper",
            "power_lower",
            "fuel_rate",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "wind_turbines": (
        "风机参数",
        [
            "name",
            "capacity",
            "cost",
            "cut_in_wind_speed",
            "cut_out_wind_speed",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "photovoltaics": (
        "光伏参数",
        [
            "name",
            "capacity",
            "cost",
            "generation_efficiency",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "storage_pcs": (
        "储能PCS参数",
        ["name", "power_capacity", "cost", "quantity_lower", "quantity_upper"],
    ),
    "storage_battery_packs": (
        "储能电池组参数",
        ["name", "battery_capacity", "cost", "quantity_lower", "quantity_upper"],
    ),
    "hydrogen_electrolyzers": (
        "电制氢参数",
        [
            "name",
            "power_capacity",
            "cost",
            "electric_to_hydrogen_efficiency",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "hydrogen_tanks": (
        "储氢罐参数",
        [
            "name",
            "hydrogen_tank_capacity",
            "cost",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "fuel_cells": (
        "燃料电池参数",
        [
            "name",
            "power_capacity",
            "cost",
            "hydrogen_to_electric_efficiency",
            "quantity_lower",
            "quantity_upper",
        ],
    ),
    "planning_parameters": (
        "规划参数",
        [
            "design_life_years",
            "diesel_price",
            "planning_load_factor",
            "green_power_ratio_lower",
            "storage_frequency_regulation_enabled",
            "load_disturbance_factor",
            "frequency_security_constraint_enabled",
            "frequency_security_upper",
            "frequency_security_lower",
            "post_disturbance_power_balance_enabled",
            "renewable_n_1_enabled",
            "load_disturbance_enabled",
        ],
    ),
}

DEFAULT_DEVICE_ROWS: dict[str, list[dict[str, Any]]] = {
    "diesel_generators": [
        {
            "name": "柴发1",
            "capacity": 100,
            "cost": 0,
            "power_upper": 100,
            "power_lower": 20,
            "fuel_rate": 0.26,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "wind_turbines": [
        {
            "name": "风机1",
            "capacity": 50,
            "cost": 0,
            "cut_in_wind_speed": 3,
            "cut_out_wind_speed": 25,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "photovoltaics": [
        {
            "name": "光伏1",
            "capacity": 50,
            "cost": 0,
            "generation_efficiency": 0.8,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "storage_pcs": [
        {
            "name": "储能PCS1",
            "power_capacity": 50,
            "cost": 0,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "storage_battery_packs": [
        {
            "name": "储能电池组1",
            "battery_capacity": 200,
            "cost": 0,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "hydrogen_electrolyzers": [
        {
            "name": "电制氢1",
            "power_capacity": 50,
            "cost": 0,
            "electric_to_hydrogen_efficiency": 0.7,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "hydrogen_tanks": [
        {
            "name": "储氢罐1",
            "hydrogen_tank_capacity": 100,
            "cost": 0,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
    "fuel_cells": [
        {
            "name": "燃料电池1",
            "power_capacity": 50,
            "cost": 0,
            "hydrogen_to_electric_efficiency": 0.55,
            "quantity_lower": 0,
            "quantity_upper": 0,
        }
    ],
}

DEFAULT_PLANNING_PARAMETERS: dict[str, Any] = {
    "design_life_years": 20,
    "diesel_price": 0,
    "planning_load_factor": 1.0,
    "green_power_ratio_lower": 0,
    "storage_frequency_regulation_enabled": False,
    "load_disturbance_factor": 0,
    "frequency_security_constraint_enabled": False,
    "frequency_security_upper": 1.5,
    "frequency_security_lower": 1.0,
    "post_disturbance_power_balance_enabled": False,
    "renewable_n_1_enabled": False,
    "load_disturbance_enabled": False,
}

INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*]')


def sanitize_scheme_name(name: str) -> str:
    return "".join(
        char
        for char in str(name or "")
        if not char.isspace() and unicodedata.category(char) not in {"Cc", "Cf"}
    )


def validate_scheme_name(name: str) -> str:
    clean = sanitize_scheme_name(name)
    if clean in {"", ".", ".."} or INVALID_NAME_RE.search(clean) or ".." in clean:
        raise ValueError("方案名称不能为空，且不能包含路径或非法字符")
    return clean


def default_time_series() -> list[dict[str, Any]]:
    return [
        {
            "hour_index": hour,
            "datetime": f"H{hour:04d}",
            "wind_speed": 0,
            "solar_irradiance": 0,
            "load": 0,
            "temperature": 0,
        }
        for hour in range(1, 8761)
    ]


def default_payload(scheme: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"scheme": scheme, "time_series": default_time_series(), "validation": []}
    for key in DEFAULT_DEVICE_ROWS:
        payload[key] = deepcopy(DEFAULT_DEVICE_ROWS[key])
    payload["planning_parameters"] = [deepcopy(DEFAULT_PLANNING_PARAMETERS)]
    payload["capacity_limits"] = []
    return payload


def default_rows_for_key(key: str) -> list[dict[str, Any]]:
    if key == "time_series":
        return default_time_series()
    if key == "planning_parameters":
        return [deepcopy(DEFAULT_PLANNING_PARAMETERS)]
    return deepcopy(DEFAULT_DEVICE_ROWS.get(key, []))


@dataclass
class PlanningStore:
    root: Path = DEFAULT_SCHEME_ROOT

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def scheme_dir(self, name: str) -> Path:
        clean = validate_scheme_name(name)
        path = (self.root / clean).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("方案路径越界")
        return path

    def workbook_path(self, name: str) -> Path:
        return self.scheme_dir(name) / WORKBOOK_NAME

    def list_schemes(self) -> list[dict[str, Any]]:
        schemes: list[dict[str, Any]] = []
        for folder in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not folder.is_dir():
                continue
            workbook = folder / WORKBOOK_NAME
            schemes.append(
                {
                    "name": folder.name,
                    "has_workbook": workbook.exists(),
                    "modified_at": workbook.stat().st_mtime if workbook.exists() else None,
                }
            )
        return schemes

    def create_scheme(self, name: str) -> dict[str, Any]:
        clean = validate_scheme_name(name)
        folder = self.scheme_dir(clean)
        if folder.exists():
            raise FileExistsError(f"方案已存在: {clean}")
        folder.mkdir(parents=True)
        self.write_scheme(clean, default_payload(clean))
        return self.read_scheme(clean)

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

    def delete_scheme(self, name: str) -> dict[str, str]:
        clean = validate_scheme_name(name)
        folder = self.scheme_dir(clean)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"方案不存在: {clean}")
        shutil.rmtree(folder)
        return {"deleted": clean}

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
        payload["time_series_loaded"] = True
        payload["time_series_count"] = len(payload.get("time_series", []))
        return payload

    def read_scheme_overview(self, name: str) -> dict[str, Any]:
        clean = validate_scheme_name(name)
        path = self.workbook_path(clean)
        if not path.exists():
            raise FileNotFoundError(f"方案参数文件不存在: {path}")
        payload = read_workbook(path, clean, include_keys=[key for key in SHEET_SPECS if key != "time_series"])
        payload["time_series_loaded"] = False
        payload["time_series_count"] = count_sheet_rows(path, SHEET_SPECS["time_series"][0])
        payload["validation"] = validate_payload(payload, require_time_series=False)
        return payload

    def read_time_series(self, name: str) -> dict[str, Any]:
        clean = validate_scheme_name(name)
        path = self.workbook_path(clean)
        if not path.exists():
            raise FileNotFoundError(f"方案参数文件不存在: {path}")
        payload = read_workbook(path, clean, include_keys=["time_series"])
        payload["time_series_loaded"] = True
        payload["time_series_count"] = len(payload.get("time_series", []))
        payload["validation"] = validate_payload(payload)
        return payload


def build_workbook(payload: dict[str, Any]) -> Workbook:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for key, (sheet_name, headers) in SHEET_SPECS.items():
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(headers)
        rows = payload.get(key, default_rows_for_key(key))
        if key == "planning_parameters" and isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            sheet.append([row.get(header, "") for header in headers])
    return workbook


def read_workbook(path: Path, scheme: str, include_keys: list[str] | None = None) -> dict[str, Any]:
    selected_keys = set(include_keys) if include_keys is not None else set(SHEET_SPECS)
    workbook = load_workbook(path, data_only=True, read_only=True)
    payload: dict[str, Any] = {"scheme": scheme, "validation": [], "capacity_limits": []}
    try:
        for key, (sheet_name, headers) in SHEET_SPECS.items():
            if key not in selected_keys:
                continue
            if sheet_name not in workbook.sheetnames:
                payload[key] = default_rows_for_key(key)
                if key != "planning_parameters":
                    payload["validation"].append({"level": "error", "message": f"缺少工作表: {sheet_name}"})
                continue
            sheet = workbook[sheet_name]
            header_values = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1), [])]
            has_named_header = any(value is not None for value in header_values)
            header_index = {
                str(value): index
                for index, value in enumerate(header_values)
                if value is not None and str(value) in headers
            }
            rows = []
            for values in sheet.iter_rows(min_row=2, values_only=True):
                if values is None or all(value is None for value in values):
                    continue
                row = {}
                for index, header in enumerate(headers):
                    if header in header_index:
                        source_index = header_index[header]
                    elif has_named_header:
                        row[header] = ""
                        continue
                    else:
                        source_index = index
                    row[header] = values[source_index] if source_index < len(values) and values[source_index] is not None else ""
                rows.append(row)
            payload[key] = (rows or default_rows_for_key(key)) if key == "planning_parameters" else rows
    finally:
        workbook.close()
    return payload


def count_sheet_rows(path: Path, sheet_name: str) -> int:
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            return 0
        sheet = workbook[sheet_name]
        return max(0, (sheet.max_row or 1) - 1)
    finally:
        workbook.close()


def validate_payload(payload: dict[str, Any], require_time_series: bool = True) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if require_time_series:
        time_series = payload.get("time_series", [])
        if len(time_series) != 8760:
            messages.append({"level": "error", "message": f"8760时序数据行数应为8760，当前为{len(time_series)}"})
        else:
            messages.append({"level": "ok", "message": "8760时序数据行数正确"})
    elif "time_series_count" in payload:
        messages.append({"level": "ok", "message": f"8760时序数据延迟加载，当前行数为{payload['time_series_count']}"})

    for key in DEFAULT_DEVICE_ROWS:
        for index, row in enumerate(payload.get(key, []), start=1):
            quantity_lower = row.get("quantity_lower", "")
            quantity_upper = row.get("quantity_upper", "")
            if quantity_lower == "" or quantity_upper == "":
                continue
            try:
                quantity_lower_number = float(quantity_lower)
                quantity_upper_number = float(quantity_upper)
            except (TypeError, ValueError):
                messages.append({"level": "error", "message": f"{SHEET_SPECS[key][0]}第{index}行数量上下限不是数值"})
                continue
            if quantity_lower_number > quantity_upper_number:
                messages.append({"level": "error", "message": f"{SHEET_SPECS[key][0]}第{index}行数据上限不能小于数据下限"})
    messages.extend(validate_planning_parameters(payload))
    return messages


def first_planning_parameter_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("planning_parameters")
    if isinstance(rows, dict):
        return rows
    if isinstance(rows, list) and rows:
        return rows[0]
    return deepcopy(DEFAULT_PLANNING_PARAMETERS)


def validate_planning_parameters(payload: dict[str, Any]) -> list[dict[str, str]]:
    row = first_planning_parameter_row(payload)
    messages: list[dict[str, str]] = []

    def number_in_range(key: str, label: str, minimum: float | None = None, maximum: float | None = None) -> float | None:
        value = row.get(key, "")
        try:
            number = float(value)
        except (TypeError, ValueError):
            messages.append({"level": "error", "message": f"{label}必须为数值"})
            return None
        if minimum is not None and number < minimum:
            messages.append({"level": "error", "message": f"{label}不能小于{minimum:g}"})
        if maximum is not None and number > maximum:
            messages.append({"level": "error", "message": f"{label}不能大于{maximum:g}"})
        return number

    design_life = number_in_range("design_life_years", "设计使用年限(年)", 1)
    if design_life is not None and not float(design_life).is_integer():
        messages.append({"level": "error", "message": "设计使用年限(年)必须为整数"})
    number_in_range("diesel_price", "柴油价格(万元/吨)", 0)
    number_in_range("planning_load_factor", "规划负荷系数(0.1-10.0)", 0.1, 10)
    number_in_range("green_power_ratio_lower", "绿电电量占比下限(0.0-1.0)", 0, 1)
    number_in_range("load_disturbance_factor", "负荷扰动系数(0.0-0.5)", 0, 0.5)
    frequency_upper = number_in_range("frequency_security_upper", "频率安全上限(1.0-1.5)", 1, 1.5)
    frequency_lower = number_in_range("frequency_security_lower", "频率安全下限(1.0-1.5)", 1, 1.5)
    if frequency_upper is not None and frequency_lower is not None and frequency_upper < frequency_lower:
        messages.append({"level": "error", "message": "频率安全上限不能小于频率安全下限"})
    return messages
