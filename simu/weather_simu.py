"""Periodic updater for realtime weather/load E-file values."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SIMU_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from efile_read import EBook  # noqa: E402


DEFAULT_WEATHER_FILE = SIMU_DIR / "weather.e"
DEFAULT_WEATHER_CSV = SIMU_DIR / "weather.csv"
DEFAULT_LOG_DIR = ROOT_DIR / "log"
DEFAULT_PERIOD_SECONDS = 5.0
WEATHER_FIELDS = ("wind_speed_mps", "solar_irradiance_w_m2", "air_temp_c", "load_kw")
FIELD_ALIASES = {
    "wind_speed": "wind_speed_mps",
    "windspeed": "wind_speed_mps",
    "wind_speed_mps": "wind_speed_mps",
    "solar_irradiance": "solar_irradiance_w_m2",
    "solar_irradiance_w_m2": "solar_irradiance_w_m2",
    "solar": "solar_irradiance_w_m2",
    "air_temp": "air_temp_c",
    "air_temp_c": "air_temp_c",
    "temperature": "air_temp_c",
    "temp": "air_temp_c",
    "load": "load_kw",
    "load_kw": "load_kw",
}


@dataclass(frozen=True)
class WeatherSimConfig:
    weather_file: Path = DEFAULT_WEATHER_FILE
    weather_csv: Path = DEFAULT_WEATHER_CSV
    period_seconds: float = DEFAULT_PERIOD_SECONDS
    loop_count: Optional[int] = None
    log_file: Optional[Path] = None


@dataclass(frozen=True)
class WeatherUpdateResult:
    weather_file: Path
    minute: int
    timestamp: str
    updated: int
    values: Dict[str, str]


def _default_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_DIR / f"weather_simu_{timestamp}.log"


def setup_logger(log_file: Path) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("WeatherSimulation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def write_ebook_aligned(book: EBook, file_path: Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for block in book.data.values():
        header = list(block.header_list)
        widths = [len(name) for name in header]
        for row in block.data:
            for idx, name in enumerate(header):
                widths[idx] = max(widths[idx], len(str(row.get(name, ""))))
        parts.append(f"<{block.name}>\n")
        parts.append("@ " + "  ".join(f"{header[idx]:<{widths[idx]}}" for idx in range(len(header))).rstrip() + "\n")
        for row in block.data:
            parts.append("# " + "  ".join(f"{str(row.get(name, '')):<{widths[idx]}}" for idx, name in enumerate(header)).rstrip() + "\n")
        parts.append(f"</{block.name}>\n")
    file_path.write_text("".join(parts), encoding="utf-8")


def minute_of_day(now: datetime) -> int:
    return now.hour * 60 + now.minute


def second_of_day(now: datetime) -> float:
    return now.hour * 3600.0 + now.minute * 60.0 + now.second + now.microsecond / 1_000_000.0


def load_weather_rows(csv_file: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with Path(csv_file).open("rt", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row:
                continue
            minute_text = row.get("minute")
            if minute_text in (None, ""):
                continue
            minute = int(float(minute_text))
            values = {field: str(row[field]) for field in WEATHER_FIELDS if field in row}
            rows[minute] = values
    if not rows:
        raise RuntimeError(f"{csv_file} does not contain weather rows")
    return rows


def _format_interpolated(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.3f}"


def _interpolate_values(
    before: Dict[str, str],
    after: Dict[str, str],
    fraction: float,
) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for field in WEATHER_FIELDS:
        if field not in before:
            continue
        if field not in after or fraction <= 0.0:
            values[field] = before[field]
            continue
        try:
            start = float(before[field])
            end = float(after[field])
        except (TypeError, ValueError):
            values[field] = before[field]
            continue
        values[field] = _format_interpolated(start + (end - start) * fraction)
    return values


def weather_values_for_time(csv_file: Path, now: datetime) -> Dict[str, str]:
    rows = load_weather_rows(csv_file)
    second = second_of_day(now)
    minute_float = second / 60.0
    base_minute = int(minute_float) % 1440
    fraction = minute_float - int(minute_float)
    if fraction <= 0.0 and base_minute in rows:
        return rows[base_minute]
    ordered_minutes = sorted(rows)
    earlier = [item for item in ordered_minutes if item <= base_minute]
    before_minute = earlier[-1] if earlier else ordered_minutes[-1]
    later = [item for item in ordered_minutes if item > before_minute]
    after_minute = later[0] if later else ordered_minutes[0]
    if before_minute == after_minute:
        return rows[before_minute]

    span = after_minute - before_minute
    offset = minute_float - before_minute
    if span <= 0:
        span += 1440
    if offset < 0:
        offset += 1440
    fraction = max(0.0, min(1.0, offset / span))
    return _interpolate_values(rows[before_minute], rows[after_minute], fraction)


def _canonical_name(name: object) -> Optional[str]:
    if name is None:
        return None
    return FIELD_ALIASES.get(str(name).strip().lower())


def _default_weather_book(values: Dict[str, str]) -> EBook:
    row = {"time": values.get("time", "00:00:00")}
    row.update({field: values.get(field, "0") for field in WEATHER_FIELDS})
    return EBook({"Weather": [row]})


def _weather_block_is_columnar(book: EBook) -> bool:
    block = book.data.get("Weather")
    if block is None:
        return False
    return all(column in block.header_list for column in ("time", *WEATHER_FIELDS))


def _update_named_value_rows(book: EBook, values: Dict[str, str]) -> int:
    updated = 0
    for block in book.data.values():
        if "name" not in block.header_list or "value" not in block.header_list:
            continue
        for row in block.data:
            field = _canonical_name(row.get("name"))
            if field is None or field not in values:
                continue
            if str(row.get("value", "")) != str(values[field]):
                row["value"] = values[field]
                updated += 1
    return updated


def _update_direct_columns(book: EBook, values: Dict[str, str], timestamp: str) -> int:
    updated = 0
    for block in book.data.values():
        canonical_columns = {column: _canonical_name(column) for column in block.header_list}
        canonical_columns = {column: field for column, field in canonical_columns.items() if field in values}
        has_time = "time" in block.header_list
        if not canonical_columns and not has_time:
            continue
        for row in block.data:
            if has_time and str(row.get("time", "")) != timestamp:
                row["time"] = timestamp
                updated += 1
            for column, field in canonical_columns.items():
                if str(row.get(column, "")) != str(values[field]):
                    row[column] = values[field]
                    updated += 1
    return updated


def update_weather_file(weather_file: Path, csv_file: Path, now: Optional[datetime] = None) -> WeatherUpdateResult:
    now = now or datetime.now()
    minute = minute_of_day(now)
    timestamp = now.strftime("%H:%M:%S")
    values = weather_values_for_time(csv_file, now)
    values = {"time": timestamp, **values}
    weather_file = Path(weather_file)

    weather_exists = weather_file.exists()
    migrated_to_columnar = False
    if weather_file.exists():
        book = EBook(weather_file)
        if not _weather_block_is_columnar(book):
            book = _default_weather_book(values)
            migrated_to_columnar = True
    else:
        book = _default_weather_book(values)

    updated = 0 if _weather_block_is_columnar(book) else _update_named_value_rows(book, values)
    updated += _update_direct_columns(book, values, timestamp)
    if migrated_to_columnar and updated == 0:
        updated = len(WEATHER_FIELDS) + 1
    if updated == 0 and not weather_exists:
        updated = len(WEATHER_FIELDS) + 1

    write_ebook_aligned(book, weather_file)
    return WeatherUpdateResult(weather_file=weather_file, minute=minute, timestamp=timestamp, updated=updated, values=values)


def format_update_message(cycle: int, result: WeatherUpdateResult) -> str:
    return (
        f"第 {cycle} 轮气象数据更新完成: minute={result.minute} updated={result.updated} "
        f"time={result.timestamp} "
        f"wind={result.values.get('wind_speed_mps')} "
        f"solar={result.values.get('solar_irradiance_w_m2')} "
        f"temp={result.values.get('air_temp_c')} "
        f"load={result.values.get('load_kw')} "
        f"file={result.weather_file}"
    )


def run_loop(
    config: WeatherSimConfig,
    logger: Optional[logging.Logger] = None,
    now_func: Callable[[], datetime] = datetime.now,
    print_func: Callable[[str], None] = print,
) -> int:
    logger = logger or setup_logger(config.log_file or _default_log_file())
    count = 0
    start_message = (
        f"气象仿真启动 weather_file={config.weather_file} weather_csv={config.weather_csv} "
        f"period={config.period_seconds} count={config.loop_count}"
    )
    logger.info(start_message)
    print_func(start_message)
    while config.loop_count is None or count < config.loop_count:
        started = time.monotonic()
        try:
            result = update_weather_file(config.weather_file, config.weather_csv, now_func())
        except Exception as exc:
            failure_message = f"第 {count + 1} 轮气象数据更新失败: {exc}"
            logger.exception(failure_message)
            print_func(failure_message)
            return 1
        message = format_update_message(count + 1, result)
        logger.info(message)
        print_func(message)
        count += 1
        if config.loop_count is not None and count >= config.loop_count:
            break
        sleep_seconds = max(0.0, float(config.period_seconds) - (time.monotonic() - started))
        wait_message = f"等待 {sleep_seconds:.3f} 秒后进入下一轮气象数据更新"
        logger.info(wait_message)
        print_func(wait_message)
        time.sleep(sleep_seconds)
    end_message = f"气象仿真结束，共完成 {count} 轮"
    logger.info(end_message)
    print_func(end_message)
    return 0


def parse_args(argv: Sequence[str]) -> WeatherSimConfig:
    parser = argparse.ArgumentParser(description="Update weather.e from minute-level weather.csv periodically.")
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER_FILE), help="Weather E file to update.")
    parser.add_argument("--csv", default=str(DEFAULT_WEATHER_CSV), help="Minute-level weather CSV file.")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD_SECONDS, help="Loop period in seconds.")
    parser.add_argument("--log", default=str(_default_log_file()), help="Weather simulation log file.")
    parser.add_argument("--once", action="store_true", help="Run one update and exit.")
    parser.add_argument("--count", type=int, default=None, help="Run a fixed number of updates and exit.")
    args = parser.parse_args(argv)
    loop_count = 1 if args.once else args.count
    return WeatherSimConfig(
        weather_file=Path(args.weather).resolve(),
        weather_csv=Path(args.csv).resolve(),
        period_seconds=args.period,
        loop_count=loop_count,
        log_file=Path(args.log).resolve() if args.log else None,
    )


def main(argv: Sequence[str]) -> int:
    return run_loop(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
