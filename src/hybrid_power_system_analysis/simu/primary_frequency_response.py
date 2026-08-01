"""Primary frequency response calculation for power deficit disturbances."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemFrequencyModel:
    f_nom_hz: float = 50.0
    s_base_mw: float = 10.0
    inertia_s: float = 4.0
    damping_mw_per_hz: float = 0.0


@dataclass(frozen=True)
class DieselGovernor:
    reserve_mw: float
    droop_mw_per_hz: float
    time_constant_s: float
    ramp_mw_per_s: float
    deadband_hz: float = 0.03
    name: str = ""


@dataclass(frozen=True)
class GridFormingStorage:
    discharge_limit_mw: float
    charge_limit_mw: float
    droop_mw_per_hz: float
    inertia_mw_s_per_hz: float = 0.0
    response_time_s: float = 0.1
    energy_mwh: float = 1.0
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9
    deadband_hz: float = 0.01
    name: str = ""


@dataclass(frozen=True)
class Disturbance:
    start_s: float
    deficit_mw: float
    end_s: float | None = None


@dataclass(frozen=True)
class FrequencyResponseResult:
    time_s: list[float]
    frequency_hz: list[float]
    delta_frequency_hz: list[float]
    diesel_power_mw: list[float]
    storage_power_mw: list[float]
    storage_soc: list[float]
    power_deficit_mw: list[float]
    diesel_unit_names: list[str] = field(default_factory=list)
    diesel_units_power_mw: list[list[float]] = field(default_factory=list)
    storage_unit_names: list[str] = field(default_factory=list)
    storage_units_power_mw: list[list[float]] = field(default_factory=list)
    storage_units_soc: list[list[float]] = field(default_factory=list)

    @property
    def nadir_hz(self) -> float:
        return min(self.frequency_hz)

    @property
    def nadir_time_s(self) -> float:
        return self.time_s[self.frequency_hz.index(self.nadir_hz)]

    @property
    def final_frequency_hz(self) -> float:
        return self.frequency_hz[-1]

    @property
    def final_delta_frequency_hz(self) -> float:
        return self.delta_frequency_hz[-1]

    @property
    def max_diesel_power_mw(self) -> float:
        return max(self.diesel_power_mw)

    @property
    def max_storage_power_mw(self) -> float:
        return max(self.storage_power_mw)


def simulate_primary_frequency_response(
    *,
    system: SystemFrequencyModel,
    disturbance: Disturbance,
    duration_s: float,
    dt_s: float,
    diesel: DieselGovernor | None = None,
    storage: GridFormingStorage | None = None,
    diesels: list[DieselGovernor] | None = None,
    storages: list[GridFormingStorage] | None = None,
) -> FrequencyResponseResult:
    """Simulate primary frequency response after a power deficit disturbance."""

    diesel_units, storage_units = _normalize_units(diesel, storage, diesels, storages)
    _validate_inputs(system, diesel_units, storage_units, disturbance, duration_s, dt_s)
    steps = int(round(duration_s / dt_s))
    f_hz = system.f_nom_hz
    diesel_powers = [0.0 for _ in diesel_units]
    storage_powers = [0.0 for _ in storage_units]
    storage_socs = [unit.initial_soc for unit in storage_units]
    previous_dfdt = 0.0
    diesel_names = _unit_names(diesel_units, "diesel")
    storage_names = _unit_names(storage_units, "bess")

    time_s: list[float] = []
    frequency_hz: list[float] = []
    delta_frequency_hz: list[float] = []
    diesel_power_mw: list[float] = []
    storage_power_mw: list[float] = []
    storage_soc_values: list[float] = []
    power_deficit_mw: list[float] = []
    diesel_units_power_mw: list[list[float]] = [[] for _ in diesel_units]
    storage_units_power_mw: list[list[float]] = [[] for _ in storage_units]
    storage_units_soc: list[list[float]] = [[] for _ in storage_units]

    for idx in range(steps + 1):
        t_s = round(idx * dt_s, 12)
        deficit = _disturbance_power(disturbance, t_s)
        diesel_total = sum(diesel_powers)
        storage_total = sum(storage_powers)
        weighted_storage_soc = _weighted_storage_soc(storage_socs, storage_units)

        time_s.append(t_s)
        frequency_hz.append(f_hz)
        delta_frequency_hz.append(f_hz - system.f_nom_hz)
        diesel_power_mw.append(diesel_total)
        storage_power_mw.append(storage_total)
        storage_soc_values.append(weighted_storage_soc)
        power_deficit_mw.append(deficit)
        for unit_idx, power in enumerate(diesel_powers):
            diesel_units_power_mw[unit_idx].append(power)
        for unit_idx, power in enumerate(storage_powers):
            storage_units_power_mw[unit_idx].append(power)
            storage_units_soc[unit_idx].append(storage_socs[unit_idx])

        if idx == steps:
            break

        delta_f = f_hz - system.f_nom_hz

        for unit_idx, unit in enumerate(diesel_units):
            diesel_ref = _clamp(
                -unit.droop_mw_per_hz * _apply_deadband(delta_f, unit.deadband_hz),
                0.0,
                unit.reserve_mw,
            )
            diesel_target_delta = (diesel_ref - diesel_powers[unit_idx]) / unit.time_constant_s * dt_s
            diesel_delta = _clamp(
                diesel_target_delta,
                -unit.ramp_mw_per_s * dt_s,
                unit.ramp_mw_per_s * dt_s,
            )
            diesel_powers[unit_idx] = _clamp(diesel_powers[unit_idx] + diesel_delta, 0.0, unit.reserve_mw)

        for unit_idx, unit in enumerate(storage_units):
            storage_ref = (
                -unit.droop_mw_per_hz * _apply_deadband(delta_f, unit.deadband_hz)
                - unit.inertia_mw_s_per_hz * previous_dfdt
            )
            response_alpha = min(1.0, dt_s / unit.response_time_s)
            storage_power = storage_powers[unit_idx] + (storage_ref - storage_powers[unit_idx]) * response_alpha
            storage_power = _clamp(storage_power, -unit.charge_limit_mw, unit.discharge_limit_mw)
            storage_power = _apply_storage_energy_limits(storage_power, storage_socs[unit_idx], unit, dt_s)
            storage_powers[unit_idx] = storage_power
            storage_socs[unit_idx] = storage_socs[unit_idx] - storage_power * dt_s / 3600.0 / unit.energy_mwh
            storage_socs[unit_idx] = _clamp(storage_socs[unit_idx], unit.min_soc, unit.max_soc)

        power_balance = sum(diesel_powers) + sum(storage_powers) - deficit - system.damping_mw_per_hz * delta_f
        previous_dfdt = system.f_nom_hz / (2.0 * system.inertia_s * system.s_base_mw) * power_balance
        f_hz = f_hz + previous_dfdt * dt_s

    return FrequencyResponseResult(
        time_s=time_s,
        frequency_hz=frequency_hz,
        delta_frequency_hz=delta_frequency_hz,
        diesel_power_mw=diesel_power_mw,
        storage_power_mw=storage_power_mw,
        storage_soc=storage_soc_values,
        power_deficit_mw=power_deficit_mw,
        diesel_unit_names=diesel_names,
        diesel_units_power_mw=diesel_units_power_mw,
        storage_unit_names=storage_names,
        storage_units_power_mw=storage_units_power_mw,
        storage_units_soc=storage_units_soc,
    )


def _disturbance_power(disturbance: Disturbance, t_s: float) -> float:
    if t_s < disturbance.start_s:
        return 0.0
    if disturbance.end_s is not None and t_s >= disturbance.end_s:
        return 0.0
    return disturbance.deficit_mw


def _validate_inputs(
    system: SystemFrequencyModel,
    diesels: list[DieselGovernor],
    storages: list[GridFormingStorage],
    disturbance: Disturbance,
    duration_s: float,
    dt_s: float,
) -> None:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if system.f_nom_hz <= 0.0:
        raise ValueError("system.f_nom_hz must be positive")
    if system.s_base_mw <= 0.0:
        raise ValueError("system.s_base_mw must be positive")
    if system.inertia_s <= 0.0:
        raise ValueError("system.inertia_s must be positive")
    if system.damping_mw_per_hz < 0.0:
        raise ValueError("system.damping_mw_per_hz must be non-negative")
    for idx, diesel in enumerate(diesels, start=1):
        prefix = f"diesels[{idx}]"
        if diesel.reserve_mw < 0.0:
            raise ValueError(f"{prefix}.reserve_mw must be non-negative")
        if diesel.droop_mw_per_hz < 0.0:
            raise ValueError(f"{prefix}.droop_mw_per_hz must be non-negative")
        if diesel.time_constant_s <= 0.0:
            raise ValueError(f"{prefix}.time_constant_s must be positive")
        if diesel.ramp_mw_per_s < 0.0:
            raise ValueError(f"{prefix}.ramp_mw_per_s must be non-negative")
        if diesel.deadband_hz < 0.0:
            raise ValueError(f"{prefix}.deadband_hz must be non-negative")
    for idx, storage in enumerate(storages, start=1):
        prefix = f"storages[{idx}]"
        if storage.discharge_limit_mw < 0.0:
            raise ValueError(f"{prefix}.discharge_limit_mw must be non-negative")
        if storage.charge_limit_mw < 0.0:
            raise ValueError(f"{prefix}.charge_limit_mw must be non-negative")
        if storage.droop_mw_per_hz < 0.0:
            raise ValueError(f"{prefix}.droop_mw_per_hz must be non-negative")
        if storage.inertia_mw_s_per_hz < 0.0:
            raise ValueError(f"{prefix}.inertia_mw_s_per_hz must be non-negative")
        if storage.response_time_s <= 0.0:
            raise ValueError(f"{prefix}.response_time_s must be positive")
        if storage.energy_mwh <= 0.0:
            raise ValueError(f"{prefix}.energy_mwh must be positive")
        if not 0.0 <= storage.min_soc <= storage.initial_soc <= storage.max_soc <= 1.0:
            raise ValueError(f"{prefix} SOC values must satisfy 0 <= min_soc <= initial_soc <= max_soc <= 1")
        if storage.deadband_hz < 0.0:
            raise ValueError(f"{prefix}.deadband_hz must be non-negative")
    if disturbance.start_s < 0.0:
        raise ValueError("disturbance.start_s must be non-negative")
    if disturbance.end_s is not None and disturbance.end_s < disturbance.start_s:
        raise ValueError("disturbance.end_s must be greater than or equal to start_s")


def _normalize_units(
    diesel: DieselGovernor | None,
    storage: GridFormingStorage | None,
    diesels: list[DieselGovernor] | None,
    storages: list[GridFormingStorage] | None,
) -> tuple[list[DieselGovernor], list[GridFormingStorage]]:
    if diesel is not None and diesels is not None:
        raise ValueError("Use either diesel or diesels, not both")
    if storage is not None and storages is not None:
        raise ValueError("Use either storage or storages, not both")
    diesel_units = list(diesels) if diesels is not None else ([diesel] if diesel is not None else [])
    storage_units = list(storages) if storages is not None else ([storage] if storage is not None else [])
    return diesel_units, storage_units


def _unit_names(units: list[DieselGovernor] | list[GridFormingStorage], prefix: str) -> list[str]:
    names: list[str] = []
    counts: dict[str, int] = {}
    for idx, unit in enumerate(units, start=1):
        base_name = unit.name.strip() if unit.name else f"{prefix}_{idx}"
        count = counts.get(base_name, 0) + 1
        counts[base_name] = count
        names.append(base_name if count == 1 else f"{base_name}_{count}")
    return names


def _weighted_storage_soc(socs: list[float], storages: list[GridFormingStorage]) -> float:
    if not socs:
        return 0.0
    total_energy = sum(storage.energy_mwh for storage in storages)
    if total_energy <= 0.0:
        return 0.0
    return sum(soc * storage.energy_mwh for soc, storage in zip(socs, storages)) / total_energy


def _apply_storage_energy_limits(
    power_mw: float,
    soc: float,
    storage: GridFormingStorage,
    dt_s: float,
) -> float:
    dt_h = dt_s / 3600.0
    if power_mw > 0.0:
        max_discharge_mw = max(0.0, (soc - storage.min_soc) * storage.energy_mwh / dt_h)
        return min(power_mw, max_discharge_mw)
    if power_mw < 0.0:
        max_charge_mw = max(0.0, (storage.max_soc - soc) * storage.energy_mwh / dt_h)
        return max(power_mw, -max_charge_mw)
    return power_mw


def _apply_deadband(delta_f_hz: float, deadband_hz: float) -> float:
    if abs(delta_f_hz) <= deadband_hz:
        return 0.0
    if delta_f_hz > 0.0:
        return delta_f_hz - deadband_hz
    return delta_f_hz + deadband_hz


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
