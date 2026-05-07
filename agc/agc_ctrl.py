import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
AGC_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, AGC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook, efile_factory


DEFAULT_MODEL_FILE = AGC_DIR / "model_real.e"
DEFAULT_LOG_DIR = ROOT_DIR / "log"
DEFAULT_CTRL_TRIGGER = 60
DEFAULT_GATHER_TRIGGER = 10
DEFAULT_BALANCE_DEADBAND = 1e-3
DEFAULT_DIESEL_MIN_DEADBAND = 1e-3
DEFAULT_STORAGE_SOC_DEADBAND = 0.0
DEFAULT_CONTROL_STEP = 1e9


@dataclass(frozen=True)
class FesClient:
    get_yc_value: Callable
    get_yx_value: Callable
    schema: object = None
    context: object = None


class SystemConfig:
    """Runtime state for the AGC loop."""

    def __init__(
        self,
        model,
        ctrl_trigger: int = DEFAULT_CTRL_TRIGGER,
        gather_trigger: int = DEFAULT_GATHER_TRIGGER,
        log_file: Optional[Path] = None,
        fes_client: Optional[FesClient] = None,
    ):
        self.model = model
        self.ctrl_trigger = max(1, int(ctrl_trigger))
        self.gather_trigger = max(1, int(gather_trigger))
        self.fes_client = fes_client or _load_fes_client()
        self.is_running = True
        self.control_count = 0
        self.gather_count = 0
        self.logger = self._setup_logger(log_file or _default_log_file())

    @staticmethod
    def _setup_logger(log_file: Path) -> logging.Logger:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("AGCControlSystem")
        logger.setLevel(logging.INFO)
        logger.propagate = False
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

    def send_yt_cmd(self, yt, value: float) -> None:
        self.logger.info("下发遥调指令 %s rtu=%s pnt=%s value=%s", yt.name, yt.rtu, yt.pnt, value)


def _default_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_DIR / f"control_system_{timestamp}.log"


def _load_fes_client() -> FesClient:
    try:
        from fes_interface import get_yc_value, get_yx_value
    except ImportError:
        return FesClient(
            get_yc_value=lambda schema, context, rtu, pnt: (False, "", None),
            get_yx_value=lambda schema, context, rtu, pnt: (False, "", None),
        )

    try:
        from sys_para import CONTEXT, SCHEMA
    except ImportError:
        SCHEMA = None
        CONTEXT = None
    return FesClient(get_yc_value=get_yc_value, get_yx_value=get_yx_value, schema=SCHEMA, context=CONTEXT)


def _load_control_functions() -> Tuple[Callable, Callable, Callable]:
    try:
        from do_one_ctrl import agc_ctrl_stg_check, do_one_ctrl, do_renew_gen_fur_calc

        return do_one_ctrl, agc_ctrl_stg_check, do_renew_gen_fur_calc
    except ImportError:
        def do_one_ctrl(logger, model):
            do_wind_solar_priority_ctrl(logger, model)

        def agc_ctrl_stg_check(logger, model):
            agc_ctrl_stg_check_default(logger, model)

        def do_renew_gen_fur_calc(logger, model, wind_speed, solar_irrad, env_temp):
            do_renew_gen_fur_calc_default(logger, model, wind_speed, solar_irrad, env_temp)

        return do_one_ctrl, agc_ctrl_stg_check, do_renew_gen_fur_calc


def _object_id(item) -> str:
    return str(getattr(item, "id", ""))


def _build_id_map(items: Iterable) -> Dict[str, object]:
    return {_object_id(item): item for item in items}


def _find_by_id(mapping: Dict[str, object], item_id, logger: logging.Logger, label: str):
    item = mapping.get(str(item_id))
    if item is None:
        logger.info("not find %s %s", label, item_id)
    return item


def get_para(model, name: str, default, cast):
    if not hasattr(model, "_para_by_name"):
        model._para_by_name = {str(para.name): para for para in getattr(model, "para", [])}
    para = model._para_by_name.get(name)
    if para is None:
        return default
    try:
        return cast(para.value)
    except (TypeError, ValueError):
        return default


def get_para_int(model, name: str, default: int) -> int:
    return get_para(model, name, default, int)


def get_para_float(model, name: str, default: float) -> float:
    return get_para(model, name, default, float)


def get_para_str(model, name: str, default: str) -> str:
    return get_para(model, name, default, str)


def initialize_system(
    emodel,
    log_file: Optional[Path] = None,
    ctrl_trigger: Optional[int] = None,
    gather_trigger: Optional[int] = None,
) -> SystemConfig:
    """Initialize device links and runtime config."""
    ctrl = ctrl_trigger if ctrl_trigger is not None else get_para_int(emodel, "CTRL_TRIGGER", DEFAULT_CTRL_TRIGGER)
    gather = gather_trigger if gather_trigger is not None else get_para_int(emodel, "GATHER_TRIGGER", DEFAULT_GATHER_TRIGGER)
    config = SystemConfig(emodel, ctrl_trigger=ctrl, gather_trigger=gather, log_file=log_file)
    emodel.config = config
    yc_by_id = _build_id_map(getattr(emodel, "yc", []))
    yx_by_id = _build_id_map(getattr(emodel, "yx", []))
    emodel.yc_by_id = yc_by_id
    emodel.yx_by_id = yx_by_id

    emodel.renew_generator = list(getattr(emodel, "wind_generator", [])) + list(getattr(emodel, "pv_generator", []))
    for gen in getattr(emodel, "diesel_generator", []):
        gen.p_yc = _find_by_id(yc_by_id, gen.p_yc_id, config.logger, "柴油发电机遥测")
    for gen in emodel.renew_generator:
        gen.p_yc = _find_by_id(yc_by_id, gen.p_yc_id, config.logger, "新能源遥测")
    for load in getattr(emodel, "energyconsumer", []):
        load.p_yc = _find_by_id(yc_by_id, load.p_yc_id, config.logger, "负荷遥测")
    for estore in getattr(emodel, "estorage", []):
        estore.p_yc = _find_by_id(yc_by_id, estore.p_yc_id, config.logger, "储能功率遥测")
        estore.soc_yc = _find_by_id(yc_by_id, estore.soc_yc_id, config.logger, "储能 SOC 遥测")

    env_by_name = {str(env.name): env for env in getattr(emodel, "env_para", [])}
    emodel.wind_speed = _find_by_id(yc_by_id, getattr(env_by_name.get("wind_speed"), "yc_id", None), config.logger, "风速遥测")
    emodel.solor_irrad = _find_by_id(yc_by_id, getattr(env_by_name.get("solor_irrad"), "yc_id", None), config.logger, "辐照遥测")
    emodel.env_temp = _find_by_id(yc_by_id, getattr(env_by_name.get("env_temp"), "yc_id", None), config.logger, "气温遥测")
    return config


def _telemetry_value(yobj, default: float = 0.0) -> float:
    if yobj is None:
        return default
    value = getattr(yobj, "value", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_running(device) -> bool:
    return int(getattr(device, "run_stat", 1) or 0) == 1


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        lower, upper = upper, lower
    return max(lower, min(upper, value))


def _current_control_value(device) -> float:
    return _as_float(getattr(device, "p_ctrl", getattr(device, "p_cur", 0.0)))


def _device_float(device, attr_names: Tuple[str, ...], default=None):
    for attr_name in attr_names:
        value = _as_float(getattr(device, attr_name, None), None)
        if value is not None:
            return value
    return default


def _control_step(model, device, kind: str) -> float:
    device_step = _device_float(device, ("p_step",), None)
    if device_step is not None and device_step >= 0.0:
        return device_step
    specific = get_para_float(model, f"{kind.upper()}_CONTROL_STEP", -1.0)
    if specific >= 0.0:
        return specific
    return get_para_float(model, "CONTROL_STEP", DEFAULT_CONTROL_STEP)


def _limit_control_step(model, device, target: float, kind: str) -> float:
    before = _current_control_value(device)
    step = _control_step(model, device, kind)
    if step < 0.0:
        return target
    return _clamp(target, before - step, before + step)


def _set_p_ctrl(model, device, target: float, kind: str, lower: float, upper: float) -> Tuple[float, float]:
    before = _current_control_value(device)
    bounded = _clamp(target, lower, upper)
    after = _clamp(_limit_control_step(model, device, bounded, kind), lower, upper)
    device.p_ctrl = after
    return before, after


def _sum_attr(devices: Iterable, attr_name: str) -> float:
    return sum(_as_float(getattr(device, attr_name, 0.0)) for device in devices if _is_running(device))


def _running_diesel_min_output(model) -> float:
    return sum(
        _as_float(getattr(gen, "p_min", 0.0))
        for gen in getattr(model, "diesel_generator", [])
        if _is_running(gen)
    )


def _running_diesel_output(model) -> float:
    return sum(
        _as_float(getattr(gen, "p_ctrl", getattr(gen, "p_cur", 0.0)))
        for gen in getattr(model, "diesel_generator", [])
        if _is_running(gen)
    )


def _diesel_min_deadband(model, gen) -> float:
    device_deadband = _device_float(gen, ("p_dead",), None)
    if device_deadband is not None:
        return max(0.0, device_deadband)
    return max(0.0, get_para_float(model, "DIESEL_MIN_DEADBAND", DEFAULT_DIESEL_MIN_DEADBAND))


def _running_diesel_min_bounds(model) -> Tuple[float, float, float]:
    diesel_min = 0.0
    lower = 0.0
    upper = 0.0
    for gen in getattr(model, "diesel_generator", []):
        if not _is_running(gen):
            continue
        p_min = _as_float(getattr(gen, "p_min", 0.0))
        deadband = _diesel_min_deadband(model, gen)
        diesel_min += p_min
        lower += p_min - deadband
        upper += p_min + deadband
    return diesel_min, lower, upper


def _renewable_output(model) -> float:
    return (
        _sum_attr(getattr(model, "wind_generator", []), "p_ctrl")
        + _sum_attr(getattr(model, "pv_generator", []), "p_ctrl")
    )


def _renewable_available_output(model) -> float:
    return (
        _sum_attr(getattr(model, "wind_generator", []), "p_fur")
        + _sum_attr(getattr(model, "pv_generator", []), "p_fur")
    )


def _load_target(model) -> float:
    load = _sum_attr(getattr(model, "energyconsumer", []), "p_fur")
    if load <= 0.0:
        load = _sum_attr(getattr(model, "energyconsumer", []), "p_cur")
    return load


def _controlled_power_mismatch(model) -> float:
    load = _load_target(model)
    generation = (
        _renewable_output(model)
        + _running_diesel_output(model)
        + _sum_attr(getattr(model, "estorage", []), "p_ctrl")
    )
    return load - generation


def _storage_soc_deadband(model, estore) -> float:
    device_deadband = _device_float(estore, ("soc_dead",), None)
    if device_deadband is not None:
        return max(0.0, device_deadband)
    return max(0.0, get_para_float(model, "STORAGE_SOC_DEADBAND", DEFAULT_STORAGE_SOC_DEADBAND))


def _can_charge_storage(model, estore) -> bool:
    soc = _as_float(getattr(estore, "soc_cur", 0.0))
    soc_max = _as_float(getattr(estore, "soc_max", 1.0), 1.0)
    return soc < soc_max - _storage_soc_deadband(model, estore)


def _can_discharge_storage(model, estore) -> bool:
    soc = _as_float(getattr(estore, "soc_cur", 0.0))
    soc_min = _as_float(getattr(estore, "soc_min", 0.0), 0.0)
    return soc > soc_min + _storage_soc_deadband(model, estore)


def _storage_by_soc(model, reverse: bool) -> list:
    return sorted(
        getattr(model, "estorage", []),
        key=lambda estore: _as_float(getattr(estore, "soc_cur", 0.0)),
        reverse=reverse,
    )


def _dispatch_limited(devices: Iterable, demand: float, attr_name: str, max_attr: str, min_attr: str = None) -> float:
    """Dispatch devices in order and return remaining positive demand."""
    remaining = max(0.0, float(demand))
    for device in devices:
        if remaining <= 0.0 or not _is_running(device):
            setattr(device, attr_name, 0.0)
            continue
        lower = _as_float(getattr(device, min_attr, 0.0), 0.0) if min_attr else 0.0
        upper = _as_float(getattr(device, max_attr, 0.0), 0.0)
        value = _clamp(remaining, lower if remaining >= lower else 0.0, upper)
        setattr(device, attr_name, value)
        remaining -= value
    return remaining


def _find_first_by_name(items: Iterable, keywords: Tuple[str, ...]):
    for item in items:
        name = str(getattr(item, "name", ""))
        if any(keyword in name for keyword in keywords):
            return item
    return None


def do_renew_gen_fur_calc_default(logger, model, wind_speed: float, solar_irrad: float, env_temp: float) -> None:
    """Forecast renewable active power for the next control period.

    Wind and PV are treated as must-take resources by the default controller.
    Forecasts are clipped by each device's p_min/p_max so downstream dispatch can
    use p_fur directly without rechecking physical bounds.
    """
    solar_irrad = max(0.0, _as_float(solar_irrad))
    wind_speed = max(0.0, _as_float(wind_speed))
    env_temp = _as_float(env_temp, 25.0)

    for pv in getattr(model, "pv_generator", []):
        p_max = _as_float(getattr(pv, "p_max", 0.0))
        p_min = _as_float(getattr(pv, "p_min", 0.0))
        rated = _as_float(getattr(pv, "rated_power", p_max), p_max)
        ref_irrad = max(_as_float(getattr(pv, "reference_irradiance", 1000.0), 1000.0), 1e-6)
        ref_temp = _as_float(getattr(pv, "reference_temperature", 25.0), 25.0)
        temp_coef = _as_float(getattr(pv, "temp_coefficient", 0.0), 0.0)
        raw = rated * solar_irrad / ref_irrad * (1.0 + temp_coef * (env_temp - ref_temp))
        pv.p_fur = _clamp(raw if _is_running(pv) else 0.0, p_min, p_max)
        logger.info("光伏预测 %s p_fur=%.3f", pv.name, pv.p_fur)

    for wind in getattr(model, "wind_generator", []):
        p_max = _as_float(getattr(wind, "p_max", 0.0))
        p_min = _as_float(getattr(wind, "p_min", 0.0))
        rated = _as_float(getattr(wind, "rated_power", p_max), p_max)
        cut_in = _as_float(getattr(wind, "cut_in_speed", 0.0), 0.0)
        cut_out = _as_float(getattr(wind, "cut_out_speed", 1e9), 1e9)
        rated_speed = max(_as_float(getattr(wind, "rated_wind_speed", cut_out), cut_out), cut_in + 1e-6)
        if not _is_running(wind) or wind_speed < cut_in or wind_speed >= cut_out:
            raw = 0.0
        elif wind_speed >= rated_speed:
            raw = rated
        else:
            ratio = (wind_speed - cut_in) / (rated_speed - cut_in)
            raw = rated * ratio ** 3
        wind.p_fur = _clamp(raw, p_min, p_max)
        logger.info("风机预测 %s p_fur=%.3f", wind.name, wind.p_fur)


def _dispatch_renewable_priority(logger, model, load_target: float) -> float:
    """Dispatch wind and PV first and return remaining load deficit."""
    demand = max(0.0, load_target)
    renewable_units = []
    for group_name, devices in (("风机", getattr(model, "wind_generator", [])), ("光伏", getattr(model, "pv_generator", []))):
        for gen in devices:
            if not _is_running(gen):
                gen.p_ctrl = 0.0
                continue
            available = max(0.0, _as_float(getattr(gen, "p_fur", getattr(gen, "p_cur", 0.0))))
            p_max = _as_float(getattr(gen, "p_max", available), available)
            renewable_units.append((group_name, gen, min(available, p_max), p_max))

    total_available = sum(available for _, _, available, _ in renewable_units)
    if total_available <= 0.0:
        return demand

    scale = min(1.0, demand / total_available)
    supplied = 0.0
    for group_name, gen, available, p_max in renewable_units:
        target = available * scale
        before, after = _set_p_ctrl(model, gen, target, "renew", 0.0, p_max)
        supplied += after
        logger.info(
            "%s协同出力 %s p_ctrl %.3f -> %.3f p_fur=%.3f",
            group_name,
            gen.name,
            before,
            after,
            available,
        )
    remaining = max(0.0, demand - supplied)
    return remaining


def _charge_storage(logger, model, surplus: float) -> float:
    remaining = max(0.0, surplus)
    for estore in _storage_by_soc(model, reverse=False):
        if not _is_running(estore):
            estore.p_ctrl = 0.0
            continue
        soc = _as_float(getattr(estore, "soc_cur", 0.0))
        if not _can_charge_storage(model, estore):
            estore.p_ctrl = 0.0
            continue
        limit = _as_float(getattr(estore, "charge_p_max", 0.0))
        target = -min(remaining, limit)
        before, after = _set_p_ctrl(model, estore, target, "storage", -limit, limit)
        absorbed = max(0.0, -after)
        remaining -= absorbed
        logger.info("储能优先充电 %s p_ctrl %.3f -> %.3f soc=%.3f", estore.name, before, after, soc)
        if remaining <= 0.0:
            break
    return remaining


def _discharge_storage(logger, model, deficit: float) -> float:
    remaining = max(0.0, deficit)
    for estore in _storage_by_soc(model, reverse=True):
        if not _is_running(estore):
            estore.p_ctrl = 0.0
            continue
        soc = _as_float(getattr(estore, "soc_cur", 0.0))
        if not _can_discharge_storage(model, estore):
            estore.p_ctrl = 0.0
            continue
        limit = _as_float(getattr(estore, "dis_charge_p_max", 0.0))
        target = min(remaining, limit)
        before, after = _set_p_ctrl(model, estore, target, "storage", -limit, limit)
        supplied = max(0.0, after)
        remaining -= supplied
        logger.info("储能补缺放电 %s p_ctrl %.3f -> %.3f soc=%.3f", estore.name, before, after, soc)
        if remaining <= 0.0:
            break
    return remaining


def _dispatch_diesel(logger, model, deficit: float) -> float:
    remaining = max(0.0, deficit)
    for gen in getattr(model, "diesel_generator", []):
        if not _is_running(gen):
            gen.p_ctrl = 0.0
            continue
        p_min = _as_float(getattr(gen, "p_min", 0.0))
        p_max = _as_float(getattr(gen, "p_max", 0.0))
        target = _clamp(remaining, p_min if remaining >= p_min else 0.0, p_max)
        before, after = _set_p_ctrl(model, gen, target, "diesel", 0.0, p_max)
        remaining -= after
        logger.info("柴发补缺 %s p_ctrl %.3f -> %.3f", gen.name, before, after)
        if remaining <= 0.0:
            break
    return remaining


def _curtail_renewable(logger, model, surplus: float) -> float:
    remaining = max(0.0, surplus)
    for devices in (getattr(model, "pv_generator", []), getattr(model, "wind_generator", [])):
        for gen in devices:
            if remaining <= 0.0:
                return 0.0
            before = _as_float(getattr(gen, "p_ctrl", 0.0))
            lower = _as_float(getattr(gen, "p_min", 0.0))
            reducible = max(0.0, before - lower)
            if reducible <= 0.0:
                continue
            target = before - min(reducible, remaining)
            _, after = _set_p_ctrl(model, gen, target, "renew", lower, before)
            reduced = before - after
            remaining -= reduced
            logger.info("富余无法消纳，弃风弃光 %s p_ctrl %.3f -> %.3f", gen.name, before, after)
    return remaining


def _restore_renewable(logger, model, shortage: float) -> float:
    remaining = max(0.0, shortage)
    for devices in (getattr(model, "wind_generator", []), getattr(model, "pv_generator", [])):
        for gen in devices:
            if remaining <= 0.0:
                return 0.0
            if not _is_running(gen):
                continue
            before = _as_float(getattr(gen, "p_ctrl", 0.0))
            available = _as_float(getattr(gen, "p_fur", before), before)
            headroom = max(0.0, available - before)
            if headroom <= 0.0:
                continue
            target = before + min(headroom, remaining)
            _, after = _set_p_ctrl(model, gen, target, "renew", before, available)
            increased = after - before
            remaining -= increased
            logger.info("减少新能源弃电 %s p_ctrl %.3f -> %.3f", gen.name, before, after)
    return remaining


def _reduce_storage_discharge(logger, model, surplus: float) -> float:
    remaining = max(0.0, surplus)
    for estore in _storage_by_soc(model, reverse=False):
        if remaining <= 0.0:
            return 0.0
        before = _as_float(getattr(estore, "p_ctrl", 0.0))
        if not _is_running(estore) or before <= 0.0:
            continue
        target = before - min(before, remaining)
        _, after = _set_p_ctrl(model, estore, target, "storage", 0.0, before)
        reduced = before - after
        remaining -= reduced
        logger.info("柴发低于下限，减少储能放电 %s p_ctrl %.3f -> %.3f", estore.name, before, after)
    return remaining


def _reduce_storage_charge(logger, model, shortage: float) -> float:
    remaining = max(0.0, shortage)
    for estore in _storage_by_soc(model, reverse=True):
        if remaining <= 0.0:
            return 0.0
        before = _as_float(getattr(estore, "p_ctrl", 0.0))
        if not _is_running(estore) or before >= 0.0:
            continue
        target = before + min(-before, remaining)
        _, after = _set_p_ctrl(model, estore, target, "storage", before, 0.0)
        increased = after - before
        remaining -= increased
        logger.info("柴发高于下限，减少储能充电 %s p_ctrl %.3f -> %.3f", estore.name, before, after)
    return remaining


def _force_storage_charge(logger, model, surplus: float) -> float:
    remaining = max(0.0, surplus)
    for estore in _storage_by_soc(model, reverse=False):
        if remaining <= 0.0:
            return 0.0
        if not _is_running(estore):
            continue
        if not _can_charge_storage(model, estore):
            continue
        before = _as_float(getattr(estore, "p_ctrl", 0.0))
        limit = _as_float(getattr(estore, "charge_p_max", 0.0))
        headroom = max(0.0, limit + min(0.0, before))
        charge = min(headroom, remaining)
        if charge <= 0.0:
            continue
        target = before - charge
        _, after = _set_p_ctrl(model, estore, target, "storage", -limit, limit)
        actual_charge = before - after
        remaining -= actual_charge
        logger.info("新能源已降至下限，储能转充电 %s p_ctrl %.3f -> %.3f", estore.name, before, after)
    return remaining


def _force_storage_discharge(logger, model, shortage: float) -> float:
    remaining = max(0.0, shortage)
    for estore in _storage_by_soc(model, reverse=True):
        if remaining <= 0.0:
            return 0.0
        if not _is_running(estore):
            continue
        if not _can_discharge_storage(model, estore):
            continue
        before = _as_float(getattr(estore, "p_ctrl", 0.0))
        limit = _as_float(getattr(estore, "dis_charge_p_max", 0.0))
        headroom = max(0.0, limit - max(0.0, before))
        discharge = min(headroom, remaining)
        if discharge <= 0.0:
            continue
        target = before + discharge
        _, after = _set_p_ctrl(model, estore, target, "storage", -limit, limit)
        actual_discharge = after - before
        remaining -= actual_discharge
        logger.info("新能源已恢复至最大可发，储能转放电 %s p_ctrl %.3f -> %.3f", estore.name, before, after)
    return remaining


def _apply_diesel_min_output_coordination(logger, model) -> None:
    diesel_min, diesel_lower, diesel_upper = _running_diesel_min_bounds(model)
    if diesel_min <= 0.0:
        return
    diesel_output = _running_diesel_output(model)

    if diesel_output < diesel_lower:
        excess = diesel_lower - diesel_output
        logger.info(
            "柴发出力 %.3f 低于下限死区边界 %.3f，按储能放电->新能源弃电->储能充电修正",
            diesel_output,
            diesel_lower,
        )
        excess = _reduce_storage_discharge(logger, model, excess)
        excess = _curtail_renewable(logger, model, excess)
        excess = _force_storage_charge(logger, model, excess)
    elif diesel_output > diesel_upper:
        shortage = diesel_output - diesel_upper
        logger.info(
            "柴发出力 %.3f 高于下限死区边界 %.3f，按储能充电->减少弃电->储能放电修正",
            diesel_output,
            diesel_upper,
        )
        shortage = _reduce_storage_charge(logger, model, shortage)
        shortage = _restore_renewable(logger, model, shortage)
        shortage = _force_storage_discharge(logger, model, shortage)
    else:
        mismatch = _controlled_power_mismatch(model)
        if mismatch > DEFAULT_BALANCE_DEADBAND:
            logger.info(
                "柴发出力 %.3f 位于下限死区内，功率缺额 %.3f 由新能源+储能联合补足",
                diesel_output,
                mismatch,
            )
            mismatch = _restore_renewable(logger, model, mismatch)
            mismatch = _force_storage_discharge(logger, model, mismatch)
        elif mismatch < -DEFAULT_BALANCE_DEADBAND:
            surplus = -mismatch
            logger.info(
                "柴发出力 %.3f 位于下限死区内，功率富余 %.3f 由储能+新能源联合消纳",
                diesel_output,
                surplus,
            )
            surplus = _reduce_storage_discharge(logger, model, surplus)
            surplus = _curtail_renewable(logger, model, surplus)
            surplus = _force_storage_charge(logger, model, surplus)


def _send_hydrogen_absorb(logger, model, surplus: float) -> float:
    if surplus <= 0.0:
        return 0.0
    yt = _find_first_by_name(getattr(model, "yt", []), ("制氢", "电解"))
    if yt is None:
        return surplus
    model.config.send_yt_cmd(yt, surplus)
    logger.info("风光富余转制氢 %s value=%.3f", yt.name, surplus)
    return 0.0


def _send_fuel_cell_support(logger, model, deficit: float) -> float:
    if deficit <= 0.0:
        return 0.0
    yt = _find_first_by_name(getattr(model, "yt", []), ("燃料电池", "燃电"))
    if yt is None:
        return deficit
    model.config.send_yt_cmd(yt, deficit)
    logger.info("缺额转燃料电池支撑 %s value=%.3f", yt.name, deficit)
    return 0.0


def do_wind_solar_priority_ctrl(logger, model) -> None:
    """Default AGC strategy: wind/PV first, storage/hydrogen for surplus, storage/diesel for deficit."""
    load_target = _load_target(model)
    logger.info("AGC 风光优先控制开始: load_target=%.3f", load_target)

    remaining_load = _dispatch_renewable_priority(logger, model, load_target)
    renew_used = load_target - remaining_load
    renew_available = (
        _sum_attr(getattr(model, "wind_generator", []), "p_fur")
        + _sum_attr(getattr(model, "pv_generator", []), "p_fur")
    )
    renewable_surplus = max(0.0, renew_available - renew_used)
    if renewable_surplus > DEFAULT_BALANCE_DEADBAND:
        logger.info("风光富余 %.3f，优先储能充电，其次制氢，最后弃风弃光", renewable_surplus)
        renewable_surplus = _charge_storage(logger, model, renewable_surplus)
        renewable_surplus = _send_hydrogen_absorb(logger, model, renewable_surplus)
        renewable_surplus = _curtail_renewable(logger, model, renewable_surplus)

    if remaining_load > DEFAULT_BALANCE_DEADBAND:
        logger.info("风光不足 %.3f，优先储能放电，其次柴发，最后燃料电池", remaining_load)
        remaining_load = _discharge_storage(logger, model, remaining_load)
        remaining_load = _dispatch_diesel(logger, model, remaining_load)
        remaining_load = _send_fuel_cell_support(logger, model, remaining_load)

    _apply_diesel_min_output_coordination(logger, model)
    model.agc_balance_mismatch = _controlled_power_mismatch(model)
    logger.info("AGC 风光优先控制结束: mismatch=%.3f", model.agc_balance_mismatch)


def agc_ctrl_stg_check_default(logger, model) -> None:
    mismatch = _as_float(getattr(model, "agc_balance_mismatch", 0.0))
    if abs(mismatch) > DEFAULT_BALANCE_DEADBAND:
        logger.warning("AGC 策略校验: 仍有功率缺额 %.3f", mismatch)
    else:
        logger.info("AGC 策略校验: 功率平衡满足死区")


def _update_yc(config: SystemConfig) -> None:
    client = config.fes_client
    for yc in getattr(config.model, "yc", []):
        result, name, value = client.get_yc_value(client.schema, client.context, yc.rtu, yc.pnt)
        if result:
            yc.value = value
        else:
            config.logger.info("无法更新遥测数据 %s rtu=%s pnt=%s，使用本地值 %s", yc.name, yc.rtu, yc.pnt, yc.value)


def _update_yx(config: SystemConfig) -> None:
    client = config.fes_client
    for yx in getattr(config.model, "yx", []):
        result, name, value = client.get_yx_value(client.schema, client.context, yx.rtu, yx.pnt)
        if result:
            yx.value = value
        else:
            config.logger.info("无法更新遥信数据 %s rtu=%s pnt=%s", yx.name, yx.rtu, yx.pnt)


def _log_power_group(logger: logging.Logger, title: str, devices: Iterable, attr_name: str = "p_cur") -> None:
    logger.info("采集<%s>设备实时信息", title)
    for device in devices:
        value = _telemetry_value(getattr(device, "p_yc", None))
        setattr(device, attr_name, value)
        logger.info("    %s.有功: %.2f", device.name, value)


def data_gather(config: SystemConfig) -> None:
    """Gather telemetry and refresh model runtime fields."""
    _update_yc(config)
    _update_yx(config)
    model = config.model
    _log_power_group(config.logger, "柴油发电机", getattr(model, "diesel_generator", []))
    _log_power_group(config.logger, "光伏", getattr(model, "pv_generator", []))
    _log_power_group(config.logger, "风机", getattr(model, "wind_generator", []))

    config.logger.info("采集<负荷>设备实时信息")
    for load in getattr(model, "energyconsumer", []):
        load.p_cur = load.p_fur = _telemetry_value(getattr(load, "p_yc", None))
        load.p_add = 0.0
        config.logger.info("    %s.有功: %.2f", load.name, load.p_cur)

    config.logger.info("采集<储能>设备实时信息")
    for estore in getattr(model, "estorage", []):
        estore.p_cur = _telemetry_value(getattr(estore, "p_yc", None))
        estore.soc_cur = _telemetry_value(getattr(estore, "soc_yc", None))
        config.logger.info("    %s.有功: %.2f", estore.name, estore.p_cur)
        config.logger.info("    %s.soc: %.2f", estore.name, estore.soc_cur)

    config.logger.info("采集天气信息:")
    config.logger.info("    风速:%.3fm/s", _telemetry_value(getattr(model, "wind_speed", None)))
    config.logger.info("    辐照:%.3fW/m^2", _telemetry_value(getattr(model, "solor_irrad", None)))
    config.logger.info("    气温:%.3f摄氏度", _telemetry_value(getattr(model, "env_temp", None)))


def _wait_until_next_period(period_s: int) -> None:
    now = time.time()
    sleep_s = period_s - (now % period_s)
    if sleep_s < period_s:
        time.sleep(max(0.0, sleep_s))


def agc_control_loop(config: SystemConfig, max_control_cycles: Optional[int] = None) -> None:
    """Run periodic data gathering and AGC decisions."""
    do_one_ctrl, agc_ctrl_stg_check, do_renew_gen_fur_calc = _load_control_functions()
    config.logger.info("=" * 60)
    config.logger.info("风-光-储-柴-荷 联合控制系统启动")
    config.logger.info("控制周期: %s秒, 采集周期: %s秒", config.ctrl_trigger, config.gather_trigger)
    config.logger.info("-" * 60)
    _wait_until_next_period(config.ctrl_trigger)

    next_gather = time.monotonic()
    next_control = time.monotonic()
    try:
        while config.is_running:
            now = time.monotonic()
            if now >= next_gather:
                config.logger.info("[采集周期 %s]", config.gather_count)
                data_gather(config)
                do_renew_gen_fur_calc(
                    config.logger,
                    config.model,
                    _telemetry_value(getattr(config.model, "wind_speed", None)),
                    _telemetry_value(getattr(config.model, "solor_irrad", None)),
                    _telemetry_value(getattr(config.model, "env_temp", None)),
                )
                config.gather_count += 1
                next_gather += config.gather_trigger
                config.logger.info("-" * 60)

            if now >= next_control:
                config.logger.info("[控制周期 %s]", config.control_count)
                do_one_ctrl(config.logger, config.model)
                agc_ctrl_stg_check(config.logger, config.model)
                config.control_count += 1
                next_control += config.ctrl_trigger
                config.logger.info("-" * 60)
                if max_control_cycles is not None and config.control_count >= max_control_cycles:
                    config.is_running = False

            sleep_to = min(next_gather, next_control) - time.monotonic()
            time.sleep(max(0.05, min(0.5, sleep_to)))
    except KeyboardInterrupt:
        config.logger.info("程序被用户中断")
    finally:
        config.logger.info("最终统计:")
        config.logger.info("  总共执行了 %s 个控制周期", config.control_count)
        config.logger.info("  总共采集了 %s 次数据", config.gather_count)


def load_model(model_file: Path):
    return efile_factory(EBook(model_file).to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AGC control loop for wind/solar/storage/diesel/load coordination.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_FILE), help="AGC model E file.")
    parser.add_argument("--log-file", default=None, help="Log file path.")
    parser.add_argument("--ctrl-trigger", type=int, default=None, help="Control period in seconds. Default uses E file or 60.")
    parser.add_argument("--gather-trigger", type=int, default=None, help="Gather period in seconds. Default uses E file or 10.")
    parser.add_argument("--max-control-cycles", type=int, default=None, help="Stop after N control cycles; useful for tests.")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    model_file = Path(args.model)
    log_file = None if args.log_file is None else Path(args.log_file)
    print(f"日志文件: {log_file or _default_log_file()}")
    print("系统启动中...")
    model = load_model(model_file)
    config = initialize_system(
        model,
        log_file=log_file,
        ctrl_trigger=args.ctrl_trigger,
        gather_trigger=args.gather_trigger,
    )
    agc_control_loop(config, max_control_cycles=args.max_control_cycles)
    config.logger.info("系统运行结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
