"""Periodic load-flow based simulator for SCADA measurement snapshots."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT_DIR = _find_project_root()
SIMU_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
for path in (PACKAGE_DIR, PACKAGE_DIR / "lfcore", PACKAGE_DIR / "model", ROOT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from paths import measurement_file, model_file
from update_meas_from_lf import (  # noqa: E402
    ANGLE_TYPES,
    MEAS_HEADER,
    VALUE_TYPES,
    Snapshot,
    format_number,
    parse_measurement_rows,
)
from ac_lf import ACPowerFlowCalc  # noqa: E402
from ac_model import ACPowerNetwork  # noqa: E402
from hybrid_lf import run_hybrid_power_flow  # noqa: E402


DEFAULT_MODEL_FILE = SIMU_DIR / "qinling.e"
DEFAULT_MODEL_FALLBACK = model_file("hybrid", "qinling.e")
DEFAULT_MEAS_FILE = SIMU_DIR / "meas.e"
DEFAULT_MEAS_FALLBACK = measurement_file("hybrid", "qinling.meas")
DEFAULT_WEATHER_FILE = SIMU_DIR / "weather.e"
DEFAULT_DEV_STAT_FILE = SIMU_DIR / "dev_stat.e"
DEFAULT_YT_CTRL_FILE = SIMU_DIR / "yt_ctrl.e"
DEFAULT_REAL_FILE = SIMU_DIR / "real.e"
DEFAULT_SCADA_FILE = SIMU_DIR / "scada.e"
DEFAULT_LOG_DIR = ROOT_DIR / "log"
DEFAULT_PERIOD_SECONDS = 60.0
DEFAULT_STORAGE_CAPACITY_KWH = 50.0


@dataclass(frozen=True)
class SimulationConfig:
    model_file: Path
    meas_file: Path
    weather_file: Path
    dev_stat_file: Path
    real_file: Path
    scada_file: Path
    yt_ctrl_file: Path = DEFAULT_YT_CTRL_FILE
    period_seconds: float = DEFAULT_PERIOD_SECONDS
    noise_std: Optional[float] = None
    random_seed: Optional[int] = None
    loop_count: Optional[int] = None
    log_file: Optional[Path] = None


@dataclass(frozen=True)
class SimulationResult:
    real_file: Path
    scada_file: Path
    updated: int
    missing: int
    overlay_updates: int
    solver_info: str


def _existing_or_fallback(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def default_config() -> SimulationConfig:
    return SimulationConfig(
        model_file=_existing_or_fallback(DEFAULT_MODEL_FILE, DEFAULT_MODEL_FALLBACK),
        meas_file=_existing_or_fallback(DEFAULT_MEAS_FILE, DEFAULT_MEAS_FALLBACK),
        weather_file=DEFAULT_WEATHER_FILE,
        dev_stat_file=DEFAULT_DEV_STAT_FILE,
        yt_ctrl_file=DEFAULT_YT_CTRL_FILE,
        real_file=DEFAULT_REAL_FILE,
        scada_file=DEFAULT_SCADA_FILE,
        log_file=_default_log_file(),
    )


def _default_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_DIR / f"simu_loop_{timestamp}.log"


def setup_logger(log_file: Path) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("SimulationLoop")
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


def _row_key(row) -> Tuple[Optional[str], Optional[str]]:
    name = row.get("name")
    idx = row.get("idx")
    return (None if name in (None, "") else str(name), None if idx in (None, "") else str(idx))


def _find_target_row(rows, overlay_row):
    name, idx = _row_key(overlay_row)
    if name is not None:
        for row in rows:
            if str(row.get("name", "")) == name:
                return row
    if idx is not None:
        for row in rows:
            if str(row.get("idx", "")) == idx:
                return row
    return None


def apply_overlay_file(model_book: EBook, overlay_file: Path) -> int:
    """Apply matching rows from weather/dev-control E files onto a model book.

    A block is applied only when the model has the same block name.  Rows match
    by ``name`` first, then by ``idx``.  Only columns already present in the
    model block are overwritten, so auxiliary weather blocks can coexist with
    the simulator without breaking the base network model.
    """
    overlay_file = Path(overlay_file)
    if not overlay_file.exists():
        return 0

    overlay_book = EBook(overlay_file)
    changed = 0
    for table_name, overlay_block in overlay_book.data.items():
        model_block = model_book.data.get(table_name)
        if model_block is None:
            continue
        writable_columns = set(model_block.header_list) - {"idx", "name"}
        if not writable_columns:
            continue
        for overlay_row in overlay_block.data:
            target_row = _find_target_row(model_block.data, overlay_row)
            if target_row is None:
                continue
            for column in overlay_block.header_list:
                if column not in writable_columns:
                    continue
                new_value = overlay_row[column]
                if str(target_row.get(column, "")) != str(new_value):
                    target_row[column] = new_value
                    changed += 1
    return changed


def _rows_by_name(block) -> Dict[str, dict]:
    return {str(row.get("name", "")): row for row in block.data}


def _set_row_value(row: dict, column: str, value) -> int:
    if column not in row:
        return 0
    text = str(value)
    if str(row.get(column, "")) == text:
        return 0
    row[column] = text
    return 1


def _model_row(model_book: EBook, dev_type: str, name: str) -> Optional[dict]:
    block = model_book.data.get(dev_type)
    if block is None:
        return None
    return _rows_by_name(block).get(str(name))


def _apply_setpoint_row(model_book: EBook, row: dict) -> int:
    dev_type = str(row.get("dev_type", ""))
    target = _model_row(model_book, dev_type, row.get("name", ""))
    if target is None:
        return 0
    changed = 0
    if dev_type == "DCACConverter":
        mapping = {"p_set": "p_ac_set", "q_set": "q_ac_set", "v_set": "v_ac_set"}
    else:
        mapping = {"p_set": "p_set", "q_set": "q_set", "v_set": "v_set"}
    for src, dst in mapping.items():
        value = row.get(src, "")
        if value != "":
            changed += _set_row_value(target, dst, value)
    if row.get("run_stat", "") != "":
        changed += _set_row_value(target, "run_stat", row["run_stat"])
    return changed


def apply_dev_stat_file(model_book: EBook, dev_stat_file: Path) -> int:
    dev_stat_file = Path(dev_stat_file)
    if not dev_stat_file.exists():
        return 0
    stat_book = EBook(dev_stat_file)
    changed = 0

    block = stat_book.data.get("DeviceRunStatus")
    if block is not None:
        for row in block.data:
            target = _model_row(model_book, row.get("dev_type", ""), row.get("name", ""))
            if target is not None:
                changed += _set_row_value(target, "run_stat", row.get("run_stat", ""))

    block = stat_book.data.get("SwitchBreakerStatus")
    if block is not None:
        for row in block.data:
            target = _model_row(model_book, row.get("dev_type", ""), row.get("name", ""))
            if target is not None:
                changed += _set_row_value(target, "run_stat", row.get("run_stat", ""))
                changed += _set_row_value(target, "status", row.get("status", ""))

    block = stat_book.data.get("GeneratorSetpoint")
    if block is not None:
        for row in block.data:
            changed += _apply_setpoint_row(model_book, row)

    block = stat_book.data.get("ConverterSetpoint")
    if block is not None:
        for row in block.data:
            changed += _apply_setpoint_row(model_book, row)

    block = stat_book.data.get("LoadSetpoint")
    if block is not None:
        for row in block.data:
            target = _model_row(model_book, row.get("dev_type", ""), row.get("name", ""))
            if target is None:
                continue
            changed += _set_row_value(target, "run_stat", row.get("run_stat", ""))
            if str(row.get("dev_type", "")) == "ACLoad":
                changed += _set_row_value(target, "pv0", row.get("p_set", ""))
                changed += _set_row_value(target, "qv0", row.get("q_set", ""))
            else:
                changed += _set_row_value(target, "p_set", row.get("p_set", ""))
                changed += _set_row_value(target, "q_set", row.get("q_set", ""))
    return changed


def _weather_values(weather_file: Path) -> Dict[str, float]:
    weather_file = Path(weather_file)
    if not weather_file.exists():
        return {}
    book = EBook(weather_file)
    block = book.data.get("Weather")
    if block is None or not block.data:
        return {}
    row = block.data[0]
    if "name" in block.header_list and "value" in block.header_list:
        raw = {str(item.get("name")): item.get("value") for item in block.data}
    else:
        raw = row
    values = {}
    for key in ("wind_speed_mps", "solar_irradiance_w_m2", "air_temp_c", "load_kw"):
        try:
            values[key] = float(raw[key])
        except (KeyError, TypeError, ValueError):
            pass
    return values


def _wind_power_kw(speed: float, rated_power: float = 10.0) -> float:
    if speed < 5.0 or speed >= 30.0:
        return 0.0
    if speed >= 15.0:
        return rated_power
    return rated_power * ((speed - 5.0) / 10.0) ** 3


def apply_weather_file(model_book: EBook, weather_file: Path) -> int:
    values = _weather_values(weather_file)
    if not values:
        return 0
    changed = 0

    if "wind_speed_mps" in values:
        wind_kw = format_number(_wind_power_kw(values["wind_speed_mps"]))
        block = model_book.data.get("DCACConverter")
        if block is not None:
            for row in block.data:
                if str(row.get("name", "")).startswith("wt"):
                    changed += _set_row_value(row, "p_ac_set", wind_kw)

    if "solar_irradiance_w_m2" in values:
        scale = max(0.0, min(1.0, values["solar_irradiance_w_m2"] / 1000.0))
        block = model_book.data.get("DCDCConverter")
        if block is not None:
            for row in block.data:
                if str(row.get("name", "")).startswith("pv"):
                    try:
                        rated = float(row.get("p_set", 0.0))
                    except (TypeError, ValueError):
                        rated = 0.0
                    changed += _set_row_value(row, "p_set", format_number(rated * scale))

    if "load_kw" in values:
        block = model_book.data.get("ACLoad")
        if block is not None and block.data:
            base_loads = []
            total = 0.0
            for row in block.data:
                try:
                    p = float(row.get("pbase", 1.0)) * float(row.get("pv0", 0.0))
                    q = float(row.get("qbase", 1.0)) * float(row.get("qv0", 0.0))
                except (TypeError, ValueError):
                    p, q = 0.0, 0.0
                base_loads.append((row, p, q))
                total += p
            if total > 0.0:
                for row, p, q in base_loads:
                    new_p = values["load_kw"] * p / total
                    new_q = values["load_kw"] * q / total
                    changed += _set_row_value(row, "pv0", format_number(new_p))
                    changed += _set_row_value(row, "qv0", format_number(new_q))
    return changed


def apply_yt_ctrl_file(model_book: EBook, yt_ctrl_file: Path) -> int:
    yt_ctrl_file = Path(yt_ctrl_file)
    if not yt_ctrl_file.exists():
        return 0
    ctrl_book = EBook(yt_ctrl_file)
    changed = 0
    for table_name in ("GeneratorSetpoint", "StorageStatus"):
        block = ctrl_book.data.get(table_name)
        if block is None:
            continue
        for row in block.data:
            if table_name == "StorageStatus":
                ess_name = str(row.get("name", ""))
                target = _model_row(model_book, "DCDCConverter", f"{ess_name}_dcdc")
                if target is not None and row.get("p_set", "") != "":
                    changed += _set_row_value(target, "p_set", row["p_set"])
            else:
                changed += _apply_setpoint_row(model_book, row)
    changed += apply_overlay_file(model_book, yt_ctrl_file)
    return changed


def update_storage_soc(dev_stat_file: Path, model_book: EBook, period_seconds: float) -> int:
    dev_stat_file = Path(dev_stat_file)
    if not dev_stat_file.exists():
        return 0
    stat_book = EBook(dev_stat_file)
    block = stat_book.data.get("StorageStatus")
    if block is None:
        return 0
    dcdc = model_book.data.get("DCDCConverter")
    dcdc_by_name = _rows_by_name(dcdc) if dcdc is not None else {}
    changed = 0
    for row in block.data:
        ess_name = str(row.get("name", ""))
        source = dcdc_by_name.get(f"{ess_name}_dcdc")
        if source is None:
            continue
        try:
            soc = float(row.get("soc_curr", 0.5))
            p_set = float(source.get("p_set", 0.0))
        except (TypeError, ValueError):
            continue
        next_soc = max(0.0, min(1.0, soc - p_set * float(period_seconds) / 3600.0 / DEFAULT_STORAGE_CAPACITY_KWH))
        changed += _set_row_value(row, "soc_curr", format_number(next_soc))
    if changed:
        write_ebook_aligned(stat_book, dev_stat_file)
    return changed


def apply_realtime_inputs(
    model_file: Path,
    weather_file: Path,
    dev_stat_file: Path,
    yt_ctrl_file: Path,
    work_dir: Path,
) -> Tuple[Path, int, EBook]:
    model_book = EBook(model_file)
    changed = 0
    changed += apply_dev_stat_file(model_book, dev_stat_file)
    changed += apply_weather_file(model_book, weather_file)
    changed += apply_yt_ctrl_file(model_book, yt_ctrl_file)
    work_dir.mkdir(parents=True, exist_ok=True)
    merged_model = work_dir / "merged_model.e"
    write_ebook_aligned(model_book, merged_model)
    return merged_model, changed, model_book


def solve_ac_snapshot(e_file: Path) -> Tuple[Snapshot, str]:
    network = ACPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    calc = ACPowerFlowCalc(network)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"AC load flow failed for {e_file}: rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return Snapshot(network, ac_grid=network), f"iter={calc.iterations}, normF={calc.normF:.3e}"


def solve_hybrid_snapshot(e_file: Path) -> Tuple[Snapshot, str]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_hybrid_power_flow(e_file, verbose=False)
    if not result.converged:
        raise RuntimeError(
            f"Hybrid load flow failed for {e_file}: rc={result.rc}, "
            f"iter={result.calc.iterations}, normF={result.calc.normF:.3e}, "
            f"ac_errors={result.ac_errors}, dc_errors={result.dc_errors}"
        )
    network = result.network
    snapshot = Snapshot(
        network,
        ac_grid=network.ac,
        dc_grid=network.dc,
        dcac_converters=network.dcac_converters,
        acac_converters=network.acac_converters,
    )
    _add_zero_impedance_devices_from_file(snapshot, e_file)
    _link_snapshot_terminal_objects(snapshot)
    return snapshot, f"iter={result.calc.iterations}, normF={result.calc.normF:.3e}"


def _add_zero_impedance_devices_from_file(snapshot: Snapshot, e_file: Path) -> None:
    book = EBook(e_file)
    specs = (
        ("ACSwitch", snapshot.ac_devices, ("p", "q", "current")),
        ("ACBreak", snapshot.ac_devices, ("p", "q", "current")),
        ("DCSwitch", snapshot.dc_devices, ("p", "current")),
        ("DCBreak", snapshot.dc_devices, ("p", "current")),
    )
    for table_name, target, value_fields in specs:
        block = book.data.get(table_name)
        if block is None:
            continue
        devices = target.setdefault(table_name, {})
        for row in block.data:
            name = str(row.get("name", ""))
            if name in devices:
                continue
            values = {
                "idx": int(row.get("idx", 0)),
                "name": name,
                "i_node": int(row.get("i_node", 0)),
                "j_node": int(row.get("j_node", 0)),
                "status": int(row.get("status", 1)),
                "run_stat": int(row.get("run_stat", 1)),
            }
            values.update({field: 0.0 for field in value_fields})
            devices[name] = SimpleNamespace(**values)


def _link_snapshot_terminal_objects(snapshot: Snapshot) -> None:
    if "ACBreak" not in snapshot.ac_devices:
        snapshot.ac_devices["ACBreak"] = snapshot._by_name(getattr(snapshot.ac, "breakers", []))
    if "DCBreak" not in snapshot.dc_devices:
        snapshot.dc_devices["DCBreak"] = snapshot._by_name(getattr(snapshot.dc, "breakers", []))

    for device_type in ("ACBranch", "ACTransformer", "ACSwitch", "ACZeroBranch", "ACBreak"):
        for dev in snapshot.ac_devices.get(device_type, {}).values():
            if getattr(dev, "i_node_obj", None) is None:
                dev.i_node_obj = snapshot.ac_nodes_by_idx.get(getattr(dev, "i_node", None))
            if getattr(dev, "j_node_obj", None) is None:
                dev.j_node_obj = snapshot.ac_nodes_by_idx.get(getattr(dev, "j_node", None))
    for device_type in ("ACGenerator", "ACLoad"):
        for dev in snapshot.ac_devices.get(device_type, {}).values():
            if getattr(dev, "node_obj", None) is None:
                dev.node_obj = snapshot.ac_nodes_by_idx.get(getattr(dev, "node", None))

    for device_type in ("DCBranch", "DCSwitch", "DCZeroBranch", "DCBreak", "DCDCConverter"):
        for dev in snapshot.dc_devices.get(device_type, {}).values():
            if getattr(dev, "i_node_obj", None) is None:
                dev.i_node_obj = snapshot.dc_nodes_by_idx.get(getattr(dev, "i_node", None))
            if getattr(dev, "j_node_obj", None) is None:
                dev.j_node_obj = snapshot.dc_nodes_by_idx.get(getattr(dev, "j_node", None))
    for device_type in ("DCGenerator", "DCLoad"):
        for dev in snapshot.dc_devices.get(device_type, {}).values():
            if getattr(dev, "node_obj", None) is None:
                dev.node_obj = snapshot.dc_nodes_by_idx.get(getattr(dev, "node", None))
    for dev in snapshot.dcac_by_name.values():
        if getattr(dev, "ac_node_obj", None) is None:
            dev.ac_node_obj = snapshot.ac_nodes_by_idx.get(getattr(dev, "ac_node", None))
        if getattr(dev, "dc_node_obj", None) is None:
            dev.dc_node_obj = snapshot.dc_nodes_by_idx.get(getattr(dev, "dc_node", None))


def _measurement_value(snapshot, row: Sequence[str]) -> Optional[float]:
    dev_type, dev_name, meas_type = row[2], row[3], row[4].upper()
    if dev_type == "ACBreak":
        dev = snapshot.ac_devices.get("ACBreak", {}).get(dev_name)
        return None if dev is None else snapshot._ac_zero_value(dev, meas_type)
    if dev_type == "DCBreak":
        dev = snapshot.dc_devices.get("DCBreak", {}).get(dev_name)
        return None if dev is None else snapshot._dc_zero_value(dev, meas_type)
    value = snapshot.value(dev_type, dev_name, meas_type)
    if value is None and (meas_type in VALUE_TYPES or meas_type in ANGLE_TYPES):
        return None
    return value


def build_real_rows(meas_file: Path, snapshot) -> Tuple[List[str], List[List[str]], List[str], int, int]:
    before, rows, after = parse_measurement_rows(meas_file)
    updated = 0
    missing = 0
    for row in rows:
        value = _measurement_value(snapshot, row)
        if value is None:
            missing += 1
            continue
        row[7] = format_number(float(value))
        updated += 1
    return before, rows, after, updated, missing


def _row_noise_sigma(row: Sequence[str], noise_std: Optional[float]) -> float:
    if noise_std is not None:
        return max(0.0, float(noise_std))
    try:
        weight = float(row[5])
    except (TypeError, ValueError):
        return 0.0
    if weight <= 0.0:
        return 0.0
    return 1.0 / math.sqrt(weight)


def add_noise_to_rows(rows: Sequence[Sequence[str]], noise_std: Optional[float], rng: random.Random) -> List[List[str]]:
    noisy_rows: List[List[str]] = []
    for source_row in rows:
        row = list(source_row)
        sigma = _row_noise_sigma(row, noise_std)
        if sigma > 0.0:
            try:
                row[7] = format_number(float(row[7]) + rng.gauss(0.0, sigma))
            except (TypeError, ValueError):
                pass
        noisy_rows.append(row)
    return noisy_rows


def render_measurement_snapshot_aligned(before: Sequence[str], rows: Sequence[Sequence[str]], after: Sequence[str]) -> str:
    widths = [len(header) for header in MEAS_HEADER]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    parts: List[str] = []
    parts.extend(line + "\n" for line in before if line)
    parts.append("<Measurement>\n")
    parts.append("@ " + "  ".join(f"{MEAS_HEADER[idx]:<{widths[idx]}}" for idx in range(len(MEAS_HEADER))).rstrip() + "\n")
    for row in rows:
        parts.append("# " + "  ".join(f"{str(cell):<{widths[idx]}}" for idx, cell in enumerate(row)).rstrip() + "\n")
    parts.append("</Measurement>\n")
    parts.extend(line + "\n" for line in after if line)
    return "".join(parts)


def write_measurement_snapshot(path: Path, before: Sequence[str], rows: Sequence[Sequence[str]], after: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_measurement_snapshot_aligned(before, rows, after), encoding="utf-8")


def run_once(
    config: SimulationConfig,
    solver: Callable[[Path], Tuple[object, str]] = solve_hybrid_snapshot,
    rng: Optional[random.Random] = None,
) -> SimulationResult:
    rng = rng or random.Random(config.random_seed)
    work_dir = config.real_file.parent / ".simu_loop_work"
    model_file, overlay_updates, model_book = apply_realtime_inputs(
        config.model_file,
        config.weather_file,
        config.dev_stat_file,
        config.yt_ctrl_file,
        work_dir,
    )
    snapshot, solver_info = solver(model_file)
    soc_updates = update_storage_soc(config.dev_stat_file, model_book, config.period_seconds)

    before, real_rows, after, updated, missing = build_real_rows(config.meas_file, snapshot)
    write_measurement_snapshot(config.real_file, before, real_rows, after)
    scada_rows = add_noise_to_rows(real_rows, config.noise_std, rng)
    write_measurement_snapshot(config.scada_file, before, scada_rows, after)
    return SimulationResult(
        real_file=config.real_file,
        scada_file=config.scada_file,
        updated=updated,
        missing=missing,
        overlay_updates=overlay_updates + soc_updates,
        solver_info=solver_info,
    )


def simulate_once(
    model_file: Optional[Path] = None,
    meas_file: Optional[Path] = None,
    weather_file: Optional[Path] = None,
    dev_stat_file: Optional[Path] = None,
    yt_ctrl_file: Optional[Path] = None,
    real_file: Optional[Path] = None,
    scada_file: Optional[Path] = None,
    period_seconds: float = DEFAULT_PERIOD_SECONDS,
    noise_std: Optional[float] = None,
    random_seed: Optional[int] = None,
    solver: Callable[[Path], Tuple[object, str]] = solve_hybrid_snapshot,
) -> SimulationResult:
    defaults = default_config()
    config = SimulationConfig(
        model_file=Path(model_file or defaults.model_file).resolve(),
        meas_file=Path(meas_file or defaults.meas_file).resolve(),
        weather_file=Path(weather_file or defaults.weather_file).resolve(),
        dev_stat_file=Path(dev_stat_file or defaults.dev_stat_file).resolve(),
        yt_ctrl_file=Path(yt_ctrl_file or defaults.yt_ctrl_file).resolve(),
        real_file=Path(real_file or defaults.real_file).resolve(),
        scada_file=Path(scada_file or defaults.scada_file).resolve(),
        period_seconds=period_seconds,
        noise_std=noise_std,
        random_seed=random_seed,
        loop_count=1,
        log_file=defaults.log_file,
    )
    return run_once(config, solver=solver)


def _result_message(cycle: int, result: SimulationResult) -> str:
    return (
        f"第 {cycle} 轮仿真完成: updated={result.updated}, missing={result.missing}, "
        f"overlays={result.overlay_updates}, {result.solver_info}, "
        f"real={result.real_file}, scada={result.scada_file}"
    )


def run_loop(
    config: SimulationConfig,
    logger: Optional[logging.Logger] = None,
    run_once_func: Callable[..., SimulationResult] = run_once,
) -> int:
    logger = logger or setup_logger(config.log_file or _default_log_file())
    rng = random.Random(config.random_seed)
    count = 0
    logger.info(
        "仿真循环启动 model=%s meas=%s weather=%s dev_stat=%s yt_ctrl=%s real=%s scada=%s period=%s noise_std=%s count=%s seed=%s",
        config.model_file,
        config.meas_file,
        config.weather_file,
        config.dev_stat_file,
        config.yt_ctrl_file,
        config.real_file,
        config.scada_file,
        config.period_seconds,
        config.noise_std,
        config.loop_count,
        config.random_seed,
    )
    while config.loop_count is None or count < config.loop_count:
        started = time.monotonic()
        try:
            result = run_once_func(config, rng=rng)
        except Exception:
            logger.exception("第 %s 轮仿真失败", count + 1)
            return 1
        logger.info(_result_message(count + 1, result))
        count += 1
        if config.loop_count is not None and count >= config.loop_count:
            break
        sleep_seconds = max(0.0, float(config.period_seconds) - (time.monotonic() - started))
        logger.info("等待 %.3f 秒后进入下一轮仿真", sleep_seconds)
        time.sleep(sleep_seconds)
    logger.info("仿真循环结束，共完成 %s 轮", count)
    return 0


def parse_args(argv: Sequence[str]) -> SimulationConfig:
    defaults = default_config()
    parser = argparse.ArgumentParser(description="Run periodic qinling hybrid load-flow simulation and write real/scada E files.")
    parser.add_argument("--model", default=str(defaults.model_file), help="Network model E file, default: simu/qinling.e or data/hybrid/qinling.e.")
    parser.add_argument("--meas", default=str(defaults.meas_file), help="Measurement definition E file, default: simu/meas.e or data/hybrid/qinling.meas.")
    parser.add_argument("--weather", default=str(defaults.weather_file), help="Realtime weather E overlay file.")
    parser.add_argument("--dev-stat", default=str(defaults.dev_stat_file), help="Device status E file.")
    parser.add_argument("--yt-ctrl", default=str(defaults.yt_ctrl_file), help="Remote control E file.")
    parser.add_argument("--real", default=str(defaults.real_file), help="Output real-value E file.")
    parser.add_argument("--scada", default=str(defaults.scada_file), help="Output noisy SCADA E file.")
    parser.add_argument("--period", type=float, default=defaults.period_seconds, help="Loop period in seconds.")
    parser.add_argument("--noise-std", type=float, default=None, help="Absolute Gaussian noise sigma. If omitted, use 1/sqrt(weight).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible SCADA noise.")
    parser.add_argument("--log", default=str(defaults.log_file), help="Simulation log file.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--count", type=int, default=None, help="Run a fixed number of cycles and exit.")
    args = parser.parse_args(argv)
    loop_count = 1 if args.once else args.count
    return SimulationConfig(
        model_file=Path(args.model).resolve(),
        meas_file=Path(args.meas).resolve(),
        weather_file=Path(args.weather).resolve(),
        dev_stat_file=Path(args.dev_stat).resolve(),
        yt_ctrl_file=Path(args.yt_ctrl).resolve(),
        real_file=Path(args.real).resolve(),
        scada_file=Path(args.scada).resolve(),
        period_seconds=args.period,
        noise_std=args.noise_std,
        random_seed=args.seed,
        loop_count=loop_count,
        log_file=Path(args.log).resolve() if args.log else None,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    return run_loop(config)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
