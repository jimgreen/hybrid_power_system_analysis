import csv
import gzip
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


SOURCE_ROOT = Path(os.environ["HISDATA_ROOT"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])
DATES = ["2025-12-04", "2025-12-05", "2025-12-16", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22", "2025-12-23", "2025-12-24"]

DYNAMIC_FIELDS = {
    "power": "\u5e76\u7f51\u5f00\u5173\u67dc\u5236\u6c22\u8bbe\u5907\u6709\u529f\u529f\u7387",
    "wind": "\u6c14\u8c61\u4eea\u98ce\u901fM",
    "irradiance": "\u8f90\u7167\u4f20\u611f\u5668\u6e29\u5ea6\u8865\u507f\u8f90\u5c04\uff08SGR\u7684\u51c0\u8f90\u5c04\uff09",
    "temperature": "\u6c14\u8c61\u4eea\u6c14\u6e29",
}
HYDROGEN_FIELDS = {
    "production_rate": "\u5236\u6c22\u6846\u67b6_\u6c22\u6c14\u5b9e\u65f6\u4ea7\u91cf\uff08m3/h\uff09",
    "production_total": "\u5236\u6c22\u6846\u67b6_\u6c22\u6c14\u603b\u4ea7\u91cf\uff08m3\uff09",
}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_file(path, wanted, buckets):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {key: header.index(name) for key, name in wanted.items()}
        time_pos = header.index("time")
        for row in reader:
            if len(row) <= max(positions.values()):
                continue
            minute = row[time_pos][:16]
            for key, pos in positions.items():
                value = number(row[pos])
                if value is not None:
                    buckets[minute][key].append(value)


result = {}
for date_text in DATES:
    day_dir = SOURCE_ROOT / date_text
    buckets = defaultdict(lambda: defaultdict(list))
    for hour_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
        dynamic = hour_dir / "\u521b\u65b0\u533a\u52a8\u6001\u63a7\u5236\u5668_yc.csv.gz"
        hydrogen = hour_dir / "\u6c22\u80fd\u5b50\u7ad9_yc.csv.gz"
        if dynamic.exists():
            read_file(dynamic, DYNAMIC_FIELDS, buckets)
        if hydrogen.exists():
            read_file(hydrogen, HYDROGEN_FIELDS, buckets)

    start = datetime.strptime(date_text, "%Y-%m-%d")
    rows = []
    for offset in range(24 * 60):
        stamp = start + timedelta(minutes=offset)
        minute = stamp.strftime("%Y-%m-%d %H:%M")
        values = buckets.get(minute, {})
        row = {"time": stamp.strftime("%Y-%m-%d %H:%M:%S")}
        for key in ["power", "production_rate", "wind", "irradiance", "temperature"]:
            samples = values.get(key, [])
            row[key] = sum(samples) / len(samples) if samples else None
        total_samples = values.get("production_total", [])
        row["production_total"] = total_samples[-1] if total_samples else None
        row["sample_count"] = len(values.get("power", []))
        rows.append(row)
    result[date_text] = rows

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
print(json.dumps({date: {"rows": len(rows), "minutes_with_power": sum(r["power"] is not None for r in rows)} for date, rows in result.items()}, ensure_ascii=False))
