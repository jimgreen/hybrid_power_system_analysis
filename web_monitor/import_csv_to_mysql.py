#!/usr/bin/env python3
"""Initialize MySQL tables and import web_monitor/data CSV seed data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEMA_PATH = ROOT / "mysql_schema.sql"
VENDOR_DIR = ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import pymysql
import server


def connect(database: str | None = None):
    config = dict(server.DB_CONFIG)
    if database is None:
        config.pop("database", None)
    else:
        config["database"] = database
    return pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config.get("database"),
        charset="utf8mb4",
        autocommit=False,
    )


def rows(filename: str) -> list[dict[str, str]]:
    with (DATA_DIR / filename).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def execute_schema() -> None:
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = [statement.strip() for statement in sql_text.split(";") if statement.strip()]
    connection = connect(database=None)
    try:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()


def truncate_tables(cursor) -> None:
    for table in (
        "overview_summary",
        "metrics",
        "alarms",
        "page_summary",
        "simu_bars",
        "simu_topology",
        "simu_daily_curves",
        "simu_state",
        "scada_columns",
        "scada_stations",
        "agc_units",
        "agc_reserve",
    ):
        cursor.execute(f"TRUNCATE TABLE {table}")


def insert_seed_data() -> None:
    connection = connect(database=server.DB_CONFIG["database"])
    try:
        with connection.cursor() as cursor:
            truncate_tables(cursor)

            for order, row in enumerate(rows("summary.csv"), start=1):
                cursor.execute(
                    "INSERT INTO overview_summary (`key`, value, unit, display_order) VALUES (%s, %s, %s, %s)",
                    (row["key"], row["value"], row.get("unit", ""), order),
                )

            for order, row in enumerate(rows("metrics.csv"), start=1):
                cursor.execute(
                    "INSERT INTO metrics (page, label, value, unit, status, display_order) VALUES (%s, %s, %s, %s, %s, %s)",
                    (row["page"], row["label"], row["value"], row.get("unit", ""), row.get("status", "normal"), order),
                )

            for order, row in enumerate(rows("alarms.csv"), start=1):
                cursor.execute(
                    "INSERT INTO alarms (page, time, object, message, status, display_order) VALUES (%s, %s, %s, %s, %s, %s)",
                    (row["page"], row["time"], row["object"], row["message"], row["status"], order),
                )

            for order, row in enumerate(rows("page_summary.csv"), start=1):
                cursor.execute(
                    "INSERT INTO page_summary (page, label, value, status, display_order) VALUES (%s, %s, %s, %s, %s)",
                    (row["page"], row["label"], row["value"], row.get("status", "normal"), order),
                )

            for order, row in enumerate(rows("simu_bars.csv"), start=1):
                cursor.execute(
                    "INSERT INTO simu_bars (label, value, unit, display_order) VALUES (%s, %s, %s, %s)",
                    (row["label"], row["value"], row.get("unit", ""), order),
                )

            for order, row in enumerate(rows("simu_topology.csv"), start=1):
                cursor.execute(
                    "INSERT INTO simu_topology (id, status, value, display_order) VALUES (%s, %s, %s, %s)",
                    (row["id"], row.get("status", "normal"), row["value"], order),
                )

            for row in rows("simu_daily_curves.csv"):
                cursor.execute(
                    "INSERT INTO simu_daily_curves (hour, wind_speed, temperature, solar_irradiance, load_value) VALUES (%s, %s, %s, %s, %s)",
                    (row["hour"], row["wind_speed"], row["temperature"], row["solar_irradiance"], row["load"]),
                )

            for row in rows("simu_state.csv"):
                cursor.execute(
                    "INSERT INTO simu_state (id, sim_time, speed, status) VALUES (1, %s, %s, %s)",
                    (row["sim_time"], row["speed"], row["status"]),
                )

            for order, row in enumerate(rows("scada_columns.csv"), start=1):
                cursor.execute(
                    "INSERT INTO scada_columns (label, value, unit, display_order) VALUES (%s, %s, %s, %s)",
                    (row["label"], row["value"], row.get("unit", ""), order),
                )

            for order, row in enumerate(rows("scada_stations.csv"), start=1):
                cursor.execute(
                    "INSERT INTO scada_stations (name, status, detail, display_order) VALUES (%s, %s, %s, %s)",
                    (row["name"], row.get("status", "normal"), row["detail"], order),
                )

            for order, row in enumerate(rows("agc_units.csv"), start=1):
                cursor.execute(
                    "INSERT INTO agc_units (name, percent, power, unit, display_order) VALUES (%s, %s, %s, %s, %s)",
                    (row["name"], row["percent"], row["power"], row.get("unit", "MW"), order),
                )

            for row in rows("agc_reserve.csv"):
                cursor.execute(
                    "INSERT INTO agc_reserve (id, score, up, down, response, cycle) VALUES (1, %s, %s, %s, %s, %s)",
                    (row["score"], row["up"], row["down"], row["response"], row["cycle"]),
                )

        connection.commit()
    finally:
        connection.close()


def main() -> None:
    execute_schema()
    insert_seed_data()
    print("MySQL schema initialized and CSV seed data imported.")


if __name__ == "__main__":
    main()
