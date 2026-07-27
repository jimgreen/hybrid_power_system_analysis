import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(os.environ["HISDATA_ROOT"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])
WIND_FIELD = "\u6c14\u8c61\u4eea\u98ce\u901fM"
SOURCE_FILE = "\u521b\u65b0\u533a\u52a8\u6001\u63a7\u5236\u5668_yc.csv.gz"


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_day(day_text):
    day = datetime.strptime(day_text, "%Y-%m-%d")
    day_dir = ROOT / day_text[:4] / day_text[5:7] / day_text
    sums = {}
    counts = {}
    source_files = 0

    for hour in range(24):
        source = day_dir / f"{hour:02d}" / SOURCE_FILE
        if not source.exists():
            continue
        source_files += 1
        with gzip.open(source, "rt", encoding="utf-8-sig", errors="replace", newline="") as stream:
            header = next(csv.reader([stream.readline()]))
            wind_pos = header.index(WIND_FIELD)
            reverse_splits = len(header) - wind_pos
            for line in stream:
                if len(line) < 20:
                    continue
                parts = line.rsplit(",", reverse_splits)
                if len(parts) <= 1:
                    continue
                value = numeric(parts[1])
                if value is not None:
                    minute = line[:16]
                    sums[minute] = sums.get(minute, 0.0) + value
                    counts[minute] = counts.get(minute, 0) + 1

    rows = []
    for minute_index in range(1440):
        timestamp = day + timedelta(minutes=minute_index)
        key = timestamp.strftime("%Y-%m-%d %H:%M")
        value = sums[key] / counts[key] if key in counts else None
        rows.append({
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed": value,
            "valid": 1 if value is not None else 0,
        })
    return day_text, rows, source_files


def main():
    day_folders = []
    for year in ["2025", "2026"]:
        year_dir = ROOT / year
        if not year_dir.exists():
            continue
        for month_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            for day_dir in sorted(path for path in month_dir.iterdir() if path.is_dir()):
                try:
                    datetime.strptime(day_dir.name, "%Y-%m-%d")
                    day_folders.append(day_dir.name)
                except ValueError:
                    pass

    monthly = {}
    file_counts = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for day_text, rows, source_files in executor.map(extract_day, day_folders, chunksize=1):
            month = day_text[:7]
            monthly.setdefault(month, []).extend(rows)
            file_counts[day_text] = source_files

    for rows in monthly.values():
        rows.sort(key=lambda row: row["time"])

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(monthly, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "months": {month: {"rows": len(rows), "valid": sum(row["valid"] for row in rows)} for month, rows in sorted(monthly.items())},
        "days": len(day_folders),
        "source_files": sum(file_counts.values()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
