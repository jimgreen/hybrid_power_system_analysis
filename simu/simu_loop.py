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
from typing import Callable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
SIMU_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, ROOT_DIR / "lfcore", ROOT_DIR / "model"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from update_meas_from_lf import (  # noqa: E402
    ANGLE_TYPES,
    MEAS_HEADER,
    VALUE_TYPES,
    Snapshot,
    format_number,
    parse_measurement_rows,
    render_measurement_file,
)
from ac_lf import ACPowerFlowCalc  # noqa: E402
from ac_model import ACPowerNetwork  # noqa: E402


DEFAULT_MODEL_FILE = SIMU_DIR / "ieee39.e"
DEFAULT_MODEL_FALLBACK = ROOT_DIR / "data" / "ac" / "ieee39.e"
DEFAULT_MEAS_FILE = SIMU_DIR / "meas.e"
DEFAULT_MEAS_FALLBACK = ROOT_DIR / "data" / "ac" / "ieee39.meas"
DEFAULT_WEATHER_FILE = SIMU_DIR / "weather.e"
DEFAULT_DEV_CTRL_FILE = SIMU_DIR / "dev_ctrl.e"
DEFAULT_REAL_FILE = SIMU_DIR / "real.e"
DEFAULT_SCADA_FILE = SIMU_DIR / "scada.e"
DEFAULT_LOG_DIR = ROOT_DIR / "log"
DEFAULT_PERIOD_SECONDS = 60.0


@dataclass(frozen=True)
class SimulationConfig:
    model_file: Path
    meas_file: Path
    weather_file: Path
    dev_ctrl_file: Path
    real_file: Path
    scada_file: Path
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
        dev_ctrl_file=DEFAULT_DEV_CTRL_FILE,
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


def apply_realtime_inputs(model_file: Path, weather_file: Path, dev_ctrl_file: Path, work_dir: Path) -> Tuple[Path, int]:
    model_book = EBook(model_file)
    changed = 0
    changed += apply_overlay_file(model_book, weather_file)
    changed += apply_overlay_file(model_book, dev_ctrl_file)
    if changed == 0:
        return Path(model_file), 0

    work_dir.mkdir(parents=True, exist_ok=True)
    merged_model = work_dir / "merged_model.e"
    model_book.apply_to_file(merged_model)
    return merged_model, changed


def solve_ac_snapshot(e_file: Path) -> Tuple[Snapshot, str]:
    network = ACPowerNetwork()
    network.read_from_file(e_file)
    network.topo()
    calc = ACPowerFlowCalc(network)
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prepare()
        rc = calc.run()
    if rc != 0 or not calc.converged:
        raise RuntimeError(f"AC load flow failed for {e_file}: rc={rc}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return Snapshot(network, ac_grid=network), f"iter={calc.iterations}, normF={calc.normF:.3e}"


def _measurement_value(snapshot, row: Sequence[str]) -> Optional[float]:
    dev_type, dev_name, meas_type = row[2], row[3], row[4].upper()
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


def write_measurement_snapshot(path: Path, before: Sequence[str], rows: Sequence[Sequence[str]], after: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_measurement_file(before, rows, after), encoding="utf-8")


def run_once(
    config: SimulationConfig,
    solver: Callable[[Path], Tuple[object, str]] = solve_ac_snapshot,
    rng: Optional[random.Random] = None,
) -> SimulationResult:
    rng = rng or random.Random(config.random_seed)
    work_dir = config.real_file.parent / ".simu_loop_work"
    model_file, overlay_updates = apply_realtime_inputs(
        config.model_file,
        config.weather_file,
        config.dev_ctrl_file,
        work_dir,
    )
    snapshot, solver_info = solver(model_file)

    before, real_rows, after, updated, missing = build_real_rows(config.meas_file, snapshot)
    write_measurement_snapshot(config.real_file, before, real_rows, after)
    scada_rows = add_noise_to_rows(real_rows, config.noise_std, rng)
    write_measurement_snapshot(config.scada_file, before, scada_rows, after)
    return SimulationResult(
        real_file=config.real_file,
        scada_file=config.scada_file,
        updated=updated,
        missing=missing,
        overlay_updates=overlay_updates,
        solver_info=solver_info,
    )


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
        "仿真循环启动 model=%s meas=%s weather=%s dev_ctrl=%s real=%s scada=%s period=%s noise_std=%s count=%s seed=%s",
        config.model_file,
        config.meas_file,
        config.weather_file,
        config.dev_ctrl_file,
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
    parser = argparse.ArgumentParser(description="Run periodic AC load-flow simulation and write real/scada E files.")
    parser.add_argument("--model", default=str(defaults.model_file), help="Network model E file, default: simu/ieee39.e or data/ac/ieee39.e.")
    parser.add_argument("--meas", default=str(defaults.meas_file), help="Measurement definition E file, default: simu/meas.e or data/ac/ieee39.meas.")
    parser.add_argument("--weather", default=str(defaults.weather_file), help="Realtime weather E overlay file.")
    parser.add_argument("--dev-ctrl", default=str(defaults.dev_ctrl_file), help="Device status/control E overlay file.")
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
        dev_ctrl_file=Path(args.dev_ctrl).resolve(),
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
