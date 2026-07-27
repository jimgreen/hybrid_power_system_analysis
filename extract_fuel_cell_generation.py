import csv
import gzip
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


SOURCE_ROOT = Path(os.environ["HISDATA_2026_ROOT"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])
PERIODS = [
    ("2026-07-04", "2026-07-04 00:00:00", "2026-07-04 23:59:59"),
    ("2026-04-29", "2026-04-29 00:00:00", "2026-04-29 23:59:59"),
    ("2026-03-10", "2026-03-10 00:00:00", "2026-03-10 23:59:59"),
]

FIELDS = {
    "self_power": "\u71c3\u6599\u7535\u6c60_\u7cfb\u7edf\u8f93\u51fa\u529f\u7387",
    "outlet_power": "\u71c3\u6599\u7535\u6c60_\u7cfb\u7edf\u51c0\u8f93\u51fa\u529f\u7387",
    "inlet_pressure": "\u5236\u6c22\u7cfb\u7edf_\u71c3\u7535\u4f9b\u6c14\u538b\u529b\uff08MPa\uff09",
    "hydrogen_tank_outlet_pressure": "\u5236\u6c22\u7cfb\u7edf_\u9ad8\u538b\u7f50\u7ec4\u51fa\u53e3\u538b\u529b\uff08MPa\uff09",
    "cell_avg_voltage": "\u71c3\u6599\u7535\u6c60_\u5355\u901a\u9053\u5e73\u5747\u7535\u6c60\u7535\u538b",
    "cell_min_voltage": "\u71c3\u6599\u7535\u6c60_\u5355\u901a\u9053\u6700\u4f4e\u7535\u6c60\u7535\u538b",
    "coolant_outlet_temperature": "\u71c3\u6599\u7535\u6c60_\u51b7\u5374\u6c34\u51fa\u53e3\u6e29\u5ea6",
    "coolant_inlet_temperature": "\u71c3\u6599\u7535\u6c60_\u51b7\u5374\u6c34\u5165\u53e3\u6e29\u5ea6",
}
TEMPERATURE_FIELDS = [
    "\u71c3\u6599\u7535\u6c60_\u6e29\u5ea6\uff08\u5de6\u524d\uff09",
    "\u71c3\u6599\u7535\u6c60_\u6e29\u5ea6\uff08\u5de6\u540e\uff09",
    "\u71c3\u6599\u7535\u6c60_\u6e29\u5ea6\uff08\u53f3\u524d\uff09",
    "\u71c3\u6599\u7535\u6c60_\u6e29\u5ea6\uff08\u53f3\u540e\uff09",
]
WEATHER_FIELDS = {
    "ambient_temperature": "\u6c14\u8c61\u4eea\u6c14\u6e29",
    "ambient_wind_speed": "\u6c14\u8c61\u4eea\u98ce\u901fM",
    "solar_irradiance": "\u8f90\u7167\u4f20\u611f\u5668\u6e29\u5ea6\u8865\u507f\u8f90\u5c04\uff08SGR\u7684\u51c0\u8f90\u5c04\uff09",
}


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


result = {}
for date_text, start_text, end_text in PERIODS:
    start = datetime.strptime(start_text, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_text, "%Y-%m-%d %H:%M:%S")
    month = date_text[5:7]
    day_dir = SOURCE_ROOT / month / date_text
    buckets = defaultdict(lambda: defaultdict(list))

    for hour in range(start.hour, end.hour + 1):
        source = day_dir / f"{hour:02d}" / "\u6c22\u80fd\u5b50\u7ad92025_yc.csv.gz"
        if not source.exists():
            continue
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            time_pos = header.index("time")
            positions = {key: header.index(name) for key, name in FIELDS.items()}
            temp_positions = [header.index(name) for name in TEMPERATURE_FIELDS]
            for row in reader:
                timestamp = datetime.strptime(row[time_pos], "%Y-%m-%d %H:%M:%S")
                if timestamp < start or timestamp > end:
                    continue
                minute = timestamp.strftime("%Y-%m-%d %H:%M")
                for key, pos in positions.items():
                    value = numeric(row[pos])
                    if value is not None:
                        buckets[minute][key].append(value)
                temperatures = [numeric(row[pos]) for pos in temp_positions]
                temperatures = [value for value in temperatures if value is not None]
                if temperatures:
                    buckets[minute]["cabin_temperature"].append(sum(temperatures) / len(temperatures))

        weather_source = day_dir / f"{hour:02d}" / "\u521b\u65b0\u533a\u52a8\u6001\u63a7\u5236\u5668_yc.csv.gz"
        if weather_source.exists():
            with gzip.open(weather_source, "rt", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                time_pos = header.index("time")
                positions = {key: header.index(name) for key, name in WEATHER_FIELDS.items()}
                for row in reader:
                    timestamp = datetime.strptime(row[time_pos], "%Y-%m-%d %H:%M:%S")
                    if timestamp < start or timestamp > end:
                        continue
                    minute = timestamp.strftime("%Y-%m-%d %H:%M")
                    for key, pos in positions.items():
                        value = numeric(row[pos])
                        if value is not None:
                            buckets[minute][key].append(value)

    rows = []
    current = start.replace(second=0)
    last_minute = end.replace(second=0)
    while current <= last_minute:
        minute = current.strftime("%Y-%m-%d %H:%M")
        values = buckets.get(minute, {})
        row = {"time": current.strftime("%Y-%m-%d %H:%M:%S")}
        for key in ["cabin_temperature", "ambient_temperature", "ambient_wind_speed", "solar_irradiance", "inlet_pressure", "hydrogen_tank_outlet_pressure", "self_power", "outlet_power", "cell_avg_voltage", "cell_min_voltage", "coolant_outlet_temperature", "coolant_inlet_temperature"]:
            samples = values.get(key, [])
            row[key] = sum(samples) / len(samples) if samples else None
        row["sample_count"] = len(values.get("self_power", []))
        rows.append(row)
        current += timedelta(minutes=1)
    result[date_text] = {"start": start_text, "end": end_text, "rows": rows}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
print(json.dumps({date: {"rows": len(payload["rows"]), "complete": sum(r["sample_count"] > 0 for r in payload["rows"])} for date, payload in result.items()}, ensure_ascii=False))
