"""Shared E-file facade for electric, heat, gas, hydrogen, and steam systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from efile_read import _read_efile_rows, efile_factory_from_rows
from model.gas_model import build_gas_network_from_model
from model.heat_model import build_heat_network_from_model
from model.hydro_model import build_hydro_network_from_model
from model.steam_model import build_steam_network_from_model
from model.effective_state import propagate_composite_run_states


FLUID_SYSTEM_PREFIXES = ("Heat", "Gas", "Hydro", "Steam")
FLUID_SYSTEM_KEYS = tuple(prefix.lower() for prefix in FLUID_SYSTEM_PREFIXES)
HYDROGEN_ELECTRIC_CONTROL_TYPES = frozenset(("P", "FLOW"))
GAS_ELECTRIC_CONTROL_TYPES = frozenset(("P", "FLOW"))
STEAM_ELECTRIC_CONTROL_TYPES = frozenset(("P", "FLOW"))
ELECTRIC_HEAT_CONTROL_TYPES = frozenset(("P", "T_OUT"))
GAS_HEAT_CONTROL_TYPES = frozenset(("FLOW", "T_OUT"))

_NETWORK_BUILDERS = {
    "heat": build_heat_network_from_model,
    "gas": build_gas_network_from_model,
    "hydro": build_hydro_network_from_model,
    "steam": build_steam_network_from_model,
}
_DOMAIN_ALIASES = {
    "ace": "ac",
    "ac": "ac",
    "dce": "dc",
    "dc": "dc",
    "heat": "heat",
    "gas": "gas",
    "hydro": "hydro",
    "h2": "hydro",
    "hydrogen": "hydro",
    "steam": "steam",
}
_COUPLING_TABLE_RE = re.compile(
    r"^(AcE|DcE|Heat|Gas|Hydro|H2|Hydrogen|Steam)2"
    r"(AcE|DcE|Heat|Gas|Hydro|H2|Hydrogen|Steam)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EnergyTerminal:
    domain: str
    device_type: str
    device_idx: int
    reference_field: str


@dataclass(frozen=True)
class EnergyCoupling:
    table_name: str
    idx: int
    name: str
    run_stat: int
    t1: EnergyTerminal
    t2: EnergyTerminal
    efficiency: float = 1.0
    e2h_coeff: Optional[float] = None
    h2e_coeff: Optional[float] = None
    e2g_coeff: Optional[float] = None
    g2e_coeff: Optional[float] = None
    e2s_coeff: Optional[float] = None
    s2e_coeff: Optional[float] = None
    g2h_coeff: Optional[float] = None
    energy_factor: Optional[float] = None
    control_type: str = "MONITOR"
    active: bool = True

    @property
    def supports_energy_balance(self) -> bool:
        if self.is_electric_heat_control:
            return self.e2h_coeff is not None and self.e2h_coeff > 0.0
        if self.is_hydrogen_electric_control:
            coefficient = self.hydrogen_electric_coeff
            return coefficient is not None and coefficient > 0.0
        if self.is_gas_electric_control:
            coefficient = self.gas_electric_coeff
            return coefficient is not None and coefficient > 0.0
        if self.is_steam_electric_control:
            coefficient = self.steam_electric_coeff
            return coefficient is not None and coefficient > 0.0
        if self.is_gas_heat_control:
            return self.g2h_coeff is not None and self.g2h_coeff > 0.0
        return self.energy_factor is not None

    @property
    def hydrogen_electric_direction(self) -> Optional[str]:
        match = _COUPLING_TABLE_RE.match(self.table_name)
        if match is None:
            return None
        source_domain = _normalize_domain(match.group(1))
        target_domain = _normalize_domain(match.group(2))
        if source_domain in {"ac", "dc"} and target_domain == "hydro":
            return "E2H"
        if source_domain == "hydro" and target_domain in {"ac", "dc"}:
            return "H2E"
        return None

    @property
    def is_hydrogen_electric_control(self) -> bool:
        return (
            self.hydrogen_electric_direction is not None
            and self.control_type in HYDROGEN_ELECTRIC_CONTROL_TYPES
        )

    @property
    def gas_electric_direction(self) -> Optional[str]:
        match = _COUPLING_TABLE_RE.match(self.table_name)
        if match is None:
            return None
        source_domain = _normalize_domain(match.group(1))
        target_domain = _normalize_domain(match.group(2))
        if source_domain in {"ac", "dc"} and target_domain == "gas":
            return "E2G"
        if source_domain == "gas" and target_domain in {"ac", "dc"}:
            return "G2E"
        return None

    @property
    def is_gas_electric_control(self) -> bool:
        return (
            self.gas_electric_direction is not None
            and self.control_type in GAS_ELECTRIC_CONTROL_TYPES
        )

    @property
    def steam_electric_direction(self) -> Optional[str]:
        match = _COUPLING_TABLE_RE.match(self.table_name)
        if match is None:
            return None
        source_domain = _normalize_domain(match.group(1))
        target_domain = _normalize_domain(match.group(2))
        if source_domain in {"ac", "dc"} and target_domain == "steam":
            return "E2S"
        if source_domain == "steam" and target_domain in {"ac", "dc"}:
            return "S2E"
        return None

    @property
    def is_steam_electric_control(self) -> bool:
        return (
            self.steam_electric_direction is not None
            and self.control_type in STEAM_ELECTRIC_CONTROL_TYPES
        )

    @property
    def electric_heat_direction(self) -> Optional[str]:
        match = _COUPLING_TABLE_RE.match(self.table_name)
        if match is None:
            return None
        source_domain = _normalize_domain(match.group(1))
        target_domain = _normalize_domain(match.group(2))
        if source_domain in {"ac", "dc"} and target_domain == "heat":
            return "E2HEAT"
        return None

    @property
    def is_electric_heat_control(self) -> bool:
        return (
            self.electric_heat_direction is not None
            and self.control_type in ELECTRIC_HEAT_CONTROL_TYPES
        )

    @property
    def gas_heat_direction(self) -> Optional[str]:
        match = _COUPLING_TABLE_RE.match(self.table_name)
        if match is None:
            return None
        source_domain = _normalize_domain(match.group(1))
        target_domain = _normalize_domain(match.group(2))
        return "G2HEAT" if source_domain == "gas" and target_domain == "heat" else None

    @property
    def is_gas_heat_control(self) -> bool:
        return (
            self.gas_heat_direction is not None
            and self.control_type in GAS_HEAT_CONTROL_TYPES
        )

    @property
    def hydrogen_electric_coeff(self) -> Optional[float]:
        if self.hydrogen_electric_direction == "E2H":
            return self.e2h_coeff
        if self.hydrogen_electric_direction == "H2E":
            return self.h2e_coeff
        return None

    @property
    def gas_electric_coeff(self) -> Optional[float]:
        if self.gas_electric_direction == "E2G":
            return self.e2g_coeff
        if self.gas_electric_direction == "G2E":
            return self.g2e_coeff
        return None

    @property
    def steam_electric_coeff(self) -> Optional[float]:
        if self.steam_electric_direction == "E2S":
            return self.e2s_coeff
        if self.steam_electric_direction == "S2E":
            return self.s2e_coeff
        return None

    @property
    def electric_terminal(self) -> Optional[EnergyTerminal]:
        for terminal in (self.t1, self.t2):
            if terminal.domain in {"ac", "dc"}:
                return terminal
        return None

    @property
    def hydro_terminal(self) -> Optional[EnergyTerminal]:
        for terminal in (self.t1, self.t2):
            if terminal.domain == "hydro":
                return terminal
        return None

    @property
    def gas_terminal(self) -> Optional[EnergyTerminal]:
        for terminal in (self.t1, self.t2):
            if terminal.domain == "gas":
                return terminal
        return None

    @property
    def steam_terminal(self) -> Optional[EnergyTerminal]:
        for terminal in (self.t1, self.t2):
            if terminal.domain == "steam":
                return terminal
        return None

    @property
    def heat_terminal(self) -> Optional[EnergyTerminal]:
        for terminal in (self.t1, self.t2):
            if terminal.domain == "heat":
                return terminal
        return None

    @property
    def controlled_terminal(self) -> Optional[EnergyTerminal]:
        if self.is_electric_heat_control:
            return self.electric_terminal if self.control_type == "P" else self.heat_terminal
        if self.is_gas_heat_control:
            return self.gas_terminal if self.control_type == "FLOW" else self.heat_terminal
        if self.is_hydrogen_electric_control:
            fluid_terminal = self.hydro_terminal
        elif self.is_gas_electric_control:
            fluid_terminal = self.gas_terminal
        elif self.is_steam_electric_control:
            fluid_terminal = self.steam_terminal
        else:
            return None
        return self.electric_terminal if self.control_type == "P" else fluid_terminal

    @property
    def dependent_terminal(self) -> Optional[EnergyTerminal]:
        if self.is_electric_heat_control:
            return self.heat_terminal if self.control_type == "P" else self.electric_terminal
        if self.is_gas_heat_control:
            return self.heat_terminal if self.control_type == "FLOW" else self.gas_terminal
        if self.is_hydrogen_electric_control:
            fluid_terminal = self.hydro_terminal
        elif self.is_gas_electric_control:
            fluid_terminal = self.gas_terminal
        elif self.is_steam_electric_control:
            fluid_terminal = self.steam_terminal
        else:
            return None
        return fluid_terminal if self.control_type == "P" else self.electric_terminal


@dataclass
class MultiEnergyContext:
    source: Optional[Path] = None
    model: Optional[object] = None
    fluid_networks: Dict[str, object] = field(default_factory=dict)
    couplings: List[EnergyCoupling] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_fluid_nodes(self) -> int:
        return sum(len(network.nodes) for network in self.fluid_networks.values())

    @property
    def active_couplings(self) -> List[EnergyCoupling]:
        return [coupling for coupling in self.couplings if coupling.active]


def _text(row, name: str, default: str = "") -> str:
    value = getattr(row, name, default)
    return str(default if value in (None, "") else value)


def _float(row, names: Sequence[str], default: Optional[float]) -> Optional[float]:
    for name in names:
        value = getattr(row, name, None)
        if value not in (None, "", "-"):
            return float(value)
    return default


def _int(row, name: str, default: int = 0) -> int:
    value = getattr(row, name, default)
    return int(default if value in (None, "", "-") else value)


def _normalize_domain(token: str) -> Optional[str]:
    return _DOMAIN_ALIASES.get(str(token).lower())


def normalize_energy_coupling_control_type(value: str) -> str:
    token = str(value or "MONITOR").strip().upper().replace("-", "_")
    aliases = {
        "POWER": "P",
        "P_SET": "P",
        "CONST_P": "P",
        "GAS_FLOW": "FLOW",
        "H2_FLOW": "FLOW",
        "FLOW_SET": "FLOW",
        "CONST_FLOW": "FLOW",
        "OUTLET_TEMPERATURE": "T_OUT",
        "T_OUT_SET": "T_OUT",
        "CONST_T_OUT": "T_OUT",
        "TEMPERATURE": "T_OUT",
        "NONE": "MONITOR",
    }
    return aliases.get(token, token)


def hydrogen_electric_dependent_value(
    coupling: EnergyCoupling,
    controlled_value: float,
    p_base_kw: float,
) -> float:
    """Convert the active endpoint setpoint into the dependent endpoint value.

    Electric endpoint values use per-unit power, fluid endpoint values use Nm3/h.
    Electrolyzers use ``e2h_coeff`` in Nm3/kWh; fuel cells use
    ``h2e_coeff`` in kWh/Nm3.
    """
    if not coupling.is_hydrogen_electric_control:
        raise ValueError(f"{coupling.table_name}:{coupling.name} is not P/FLOW controlled")
    coefficient = coupling.hydrogen_electric_coeff
    power_base = float(p_base_kw)
    value = float(controlled_value)
    if coefficient is None or coefficient <= 0.0:
        field = "e2h_coeff" if coupling.hydrogen_electric_direction == "E2H" else "h2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if power_base <= 0.0:
        raise ValueError("electric power base must be positive")
    if value < 0.0:
        raise ValueError(f"{coupling.table_name}:{coupling.name} setpoint must be non-negative")

    if coupling.control_type == "P":
        power_kw = value * power_base
        return (
            power_kw * coefficient
            if coupling.hydrogen_electric_direction == "E2H"
            else power_kw / coefficient
        )

    return (
        value / coefficient / power_base
        if coupling.hydrogen_electric_direction == "E2H"
        else value * coefficient / power_base
    )


def hydrogen_electric_balance_residual(
    coupling: EnergyCoupling,
    electric_power_pu: float,
    hydro_flow: float,
    p_base_kw: float,
) -> float:
    """Return the dimensional conversion mismatch for an electric/H2 device."""
    power_kw = float(electric_power_pu) * float(p_base_kw)
    flow = float(hydro_flow)
    coefficient = coupling.hydrogen_electric_coeff
    if coefficient is None or coefficient <= 0.0:
        field = "e2h_coeff" if coupling.hydrogen_electric_direction == "E2H" else "h2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if coupling.hydrogen_electric_direction == "E2H":
        return flow - power_kw * coefficient
    if coupling.hydrogen_electric_direction == "H2E":
        return power_kw - flow * coefficient
    raise ValueError(f"{coupling.table_name}:{coupling.name} is not an electric/H2 coupling")


def gas_electric_dependent_value(
    coupling: EnergyCoupling,
    controlled_value: float,
    p_base_kw: float,
) -> float:
    """Convert a P/FLOW gas-electric setpoint into its dependent endpoint."""
    if not coupling.is_gas_electric_control:
        raise ValueError(f"{coupling.table_name}:{coupling.name} is not P/FLOW controlled")
    coefficient = coupling.gas_electric_coeff
    power_base = float(p_base_kw)
    value = float(controlled_value)
    if coefficient is None or coefficient <= 0.0:
        field = "e2g_coeff" if coupling.gas_electric_direction == "E2G" else "g2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if power_base <= 0.0:
        raise ValueError("electric power base must be positive")
    if value < 0.0:
        raise ValueError(f"{coupling.table_name}:{coupling.name} setpoint must be non-negative")

    if coupling.control_type == "P":
        power_kw = value * power_base
        return (
            power_kw * coefficient
            if coupling.gas_electric_direction == "E2G"
            else power_kw / coefficient
        )
    return (
        value / coefficient / power_base
        if coupling.gas_electric_direction == "E2G"
        else value * coefficient / power_base
    )


def gas_electric_balance_residual(
    coupling: EnergyCoupling,
    electric_power_pu: float,
    gas_flow: float,
    p_base_kw: float,
) -> float:
    """Return gas/electric conversion mismatch in flow or kW units."""
    power_kw = float(electric_power_pu) * float(p_base_kw)
    flow = float(gas_flow)
    coefficient = coupling.gas_electric_coeff
    if coefficient is None or coefficient <= 0.0:
        field = "e2g_coeff" if coupling.gas_electric_direction == "E2G" else "g2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if coupling.gas_electric_direction == "E2G":
        return flow - power_kw * coefficient
    if coupling.gas_electric_direction == "G2E":
        return power_kw - flow * coefficient
    raise ValueError(f"{coupling.table_name}:{coupling.name} is not an electric/gas coupling")


def steam_electric_dependent_value(
    coupling: EnergyCoupling,
    controlled_value: float,
    p_base_kw: float,
    steam_enthalpy: float,
    condensate_enthalpy: float,
) -> float:
    """Convert a steam/electric setpoint using mass flow and specific enthalpy.

    Steam flow is interpreted as t/h and enthalpy as kJ/kg. ``e2s_coeff`` and
    ``s2e_coeff`` are dimensionless conversion efficiencies.
    """
    if not coupling.is_steam_electric_control:
        raise ValueError(f"{coupling.table_name}:{coupling.name} is not P/FLOW controlled")
    coefficient = coupling.steam_electric_coeff
    power_base = float(p_base_kw)
    value = float(controlled_value)
    enthalpy_power = (
        float(steam_enthalpy) - float(condensate_enthalpy)
    ) / 3.6
    if coefficient is None or coefficient <= 0.0:
        field = "e2s_coeff" if coupling.steam_electric_direction == "E2S" else "s2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if power_base <= 0.0:
        raise ValueError("electric power base must be positive")
    if enthalpy_power <= 0.0:
        raise ValueError(
            f"{coupling.table_name}:{coupling.name} steam enthalpy must exceed condensate enthalpy"
        )
    if value < 0.0:
        raise ValueError(f"{coupling.table_name}:{coupling.name} setpoint must be non-negative")

    if coupling.control_type == "P":
        power_kw = value * power_base
        return (
            power_kw * coefficient / enthalpy_power
            if coupling.steam_electric_direction == "E2S"
            else power_kw / (enthalpy_power * coefficient)
        )
    return (
        value * enthalpy_power / coefficient / power_base
        if coupling.steam_electric_direction == "E2S"
        else value * enthalpy_power * coefficient / power_base
    )


def steam_electric_balance_residual(
    coupling: EnergyCoupling,
    electric_power_pu: float,
    steam_flow: float,
    steam_enthalpy: float,
    condensate_enthalpy: float,
    p_base_kw: float,
) -> float:
    """Return steam/electric conversion mismatch in kW."""
    power_kw = float(electric_power_pu) * float(p_base_kw)
    steam_kw = (
        float(steam_flow)
        * (float(steam_enthalpy) - float(condensate_enthalpy))
        / 3.6
    )
    coefficient = coupling.steam_electric_coeff
    if coefficient is None or coefficient <= 0.0:
        field = "e2s_coeff" if coupling.steam_electric_direction == "E2S" else "s2e_coeff"
        raise ValueError(f"{coupling.table_name}:{coupling.name} {field} must be positive")
    if coupling.steam_electric_direction == "E2S":
        return steam_kw - power_kw * coefficient
    if coupling.steam_electric_direction == "S2E":
        return power_kw - steam_kw * coefficient
    raise ValueError(f"{coupling.table_name}:{coupling.name} is not an electric/steam coupling")


def gas_heat_balance_residual(
    coupling: EnergyCoupling,
    gas_flow: float,
    heat_flow: float,
    supply_temperature: float,
    return_temperature: float,
    heat_capacity: float,
) -> float:
    """Return gas input minus delivered heat in kW."""
    if not coupling.is_gas_heat_control:
        raise ValueError(f"{coupling.table_name}:{coupling.name} is not gas/heat controlled")
    coefficient = coupling.g2h_coeff
    if coefficient is None or coefficient <= 0.0:
        raise ValueError(f"{coupling.table_name}:{coupling.name} g2h_coeff must be positive")
    return (
        float(gas_flow) * float(coefficient)
        - float(heat_flow)
        * float(heat_capacity)
        * (float(supply_temperature) - float(return_temperature))
    )


def electric_heat_balance_residual(
    coupling: EnergyCoupling,
    electric_power_pu: float,
    source_flow: float,
    t_out: float,
    t_return: float,
    heat_capacity: float,
    p_base_kw: float,
) -> float:
    """Return electric input minus delivered heat in kW."""
    if not coupling.is_electric_heat_control:
        raise ValueError(f"{coupling.table_name}:{coupling.name} is not an electric/heat coupling")
    coefficient = coupling.e2h_coeff
    if coefficient is None or coefficient <= 0.0:
        raise ValueError(f"{coupling.table_name}:{coupling.name} e2h_coeff must be positive")
    return (
        float(electric_power_pu) * float(p_base_kw) * float(coefficient)
        - float(source_flow)
        * float(heat_capacity)
        * (float(t_out) - float(t_return))
    )


def _domain_from_reference_field(reference_field: str, fallback: str) -> str:
    token = "".join(char for char in reference_field.lower() if char.isalnum())
    token = token.removeprefix("idx").removesuffix("t1").removesuffix("t2")
    for prefix, domain in (
        ("ace", "ac"),
        ("ac", "ac"),
        ("dce", "dc"),
        ("dc", "dc"),
        ("hydrogen", "hydro"),
        ("hydro", "hydro"),
        ("h2", "hydro"),
        ("heat", "heat"),
        ("gas", "gas"),
        ("steam", "steam"),
    ):
        if token.startswith(prefix):
            return domain
    return fallback


def _endpoint_device_type(domain: str, reference_field: str) -> str:
    token = "".join(char for char in reference_field.lower() if char.isalnum())
    token = token.removeprefix("idx").removesuffix("t1").removesuffix("t2")
    if domain == "ac":
        return "ACLoad" if "load" in token else "ACGenerator"
    if domain == "dc":
        return "DCLoad" if "load" in token else "DCGenerator"
    prefix = "Hydro" if domain == "hydro" else domain.capitalize()
    if "load" in token:
        return f"{prefix}Load"
    if "storage" in token or "tank" in token:
        return f"{prefix}Storage"
    return f"{prefix}Source"


def _endpoint_reference_fields(row) -> Tuple[Optional[str], Optional[str]]:
    names = tuple(vars(row))
    t1_fields = [name for name in names if name.startswith("idx_") and name.endswith("_t1")]
    t2_fields = [name for name in names if name.startswith("idx_") and name.endswith("_t2")]
    if t1_fields and t2_fields:
        return t1_fields[0], t2_fields[0]
    index_fields = [
        name
        for name in names
        if name.startswith("idx_") and name not in {"idx", "index"}
    ]
    return (index_fields[0], index_fields[1]) if len(index_fields) >= 2 else (None, None)


def _device_row_by_idx(model, device_type: str, device_idx: int):
    for row in getattr(model, device_type, ()) or ():
        if _int(row, "idx", -1) == int(device_idx):
            return row
    return None


def _parse_couplings(model) -> Tuple[List[EnergyCoupling], List[str]]:
    couplings: List[EnergyCoupling] = []
    warnings: List[str] = []
    for table_name, rows in vars(model).items():
        if not isinstance(rows, list):
            continue
        match = _COUPLING_TABLE_RE.match(table_name)
        if match is None:
            continue
        t1_domain = _normalize_domain(match.group(1))
        t2_domain = _normalize_domain(match.group(2))
        if t1_domain is None or t2_domain is None:
            continue
        for position, row in enumerate(rows):
            t1_field, t2_field = _endpoint_reference_fields(row)
            if t1_field is None or t2_field is None:
                warnings.append(
                    f"{table_name} row {position + 1} has no two endpoint index fields"
                )
                continue
            row_t1_domain = _domain_from_reference_field(t1_field, t1_domain)
            row_t2_domain = _domain_from_reference_field(t2_field, t2_domain)
            t1_idx = _int(row, t1_field, -1)
            t2_idx = _int(row, t2_field, -1)
            t1_type = _endpoint_device_type(row_t1_domain, t1_field)
            t2_type = _endpoint_device_type(row_t2_domain, t2_field)
            t1_row = _device_row_by_idx(model, t1_type, t1_idx)
            t2_row = _device_row_by_idx(model, t2_type, t2_idx)
            coupling_idx = _int(row, "idx", position + 1)
            coupling_name = _text(row, "name", f"{table_name}_{coupling_idx}")
            if t1_row is None:
                warnings.append(
                    f"{table_name}:{coupling_name} references missing {t1_type} idx={t1_idx}"
                )
            if t2_row is None:
                warnings.append(
                    f"{table_name}:{coupling_name} references missing {t2_type} idx={t2_idx}"
                )
            row_alive = _int(row, "run_stat", 1) == 1
            endpoints_alive = all(
                endpoint is not None and _int(endpoint, "run_stat", 1) == 1
                for endpoint in (t1_row, t2_row)
            )
            direction = None
            for fluid_domain, token in (("hydro", "H"), ("gas", "G"), ("steam", "S")):
                if t1_domain in {"ac", "dc"} and t2_domain == fluid_domain:
                    direction = f"E2{token}"
                    break
                if t1_domain == fluid_domain and t2_domain in {"ac", "dc"}:
                    direction = f"{token}2E"
                    break
            electric_heat = t1_domain in {"ac", "dc"} and t2_domain == "heat"
            gas_heat = t1_domain == "gas" and t2_domain == "heat"
            coefficient_field = {
                "E2H": "e2h_coeff",
                "H2E": "h2e_coeff",
                "E2G": "e2g_coeff",
                "G2E": "g2e_coeff",
                "E2S": "e2s_coeff",
                "S2E": "s2e_coeff",
            }.get(direction)
            if electric_heat:
                coefficient_field = "e2h_coeff"
            elif gas_heat:
                coefficient_field = "g2h_coeff"
            coefficient = (
                _float(row, (coefficient_field,), None)
                if coefficient_field is not None
                else None
            )
            default_control_type = (
                "P"
                if electric_heat and coefficient is not None
                else "FLOW"
                if gas_heat and coefficient is not None
                else
                "FLOW"
                if direction in {"E2H", "E2G", "E2S"} and coefficient is not None
                else "P"
                if direction in {"H2E", "G2E", "S2E"} and coefficient is not None
                else "MONITOR"
            )
            control_type = normalize_energy_coupling_control_type(
                _text(row, "control_type", default_control_type)
            )
            direct_control = (
                (direction in {"E2H", "H2E"} and control_type in HYDROGEN_ELECTRIC_CONTROL_TYPES)
                or (direction in {"E2G", "G2E"} and control_type in GAS_ELECTRIC_CONTROL_TYPES)
                or (direction in {"E2S", "S2E"} and control_type in STEAM_ELECTRIC_CONTROL_TYPES)
                or (electric_heat and control_type in ELECTRIC_HEAT_CONTROL_TYPES)
                or (gas_heat and control_type in GAS_HEAT_CONTROL_TYPES)
            )
            if direct_control and (coefficient is None or coefficient <= 0.0):
                unit = (
                    "kWh/kWh"
                    if electric_heat
                    else "kWh/Nm3"
                    if direction == "G2E" or gas_heat
                    else "Nm3/kWh"
                    if direction == "E2G"
                    else "dimensionless"
                    if direction in {"E2S", "S2E"}
                    else "Nm3/kWh"
                    if direction == "E2H"
                    else "kWh/Nm3"
                )
                warnings.append(
                    f"{table_name}:{coupling_name} requires positive {coefficient_field} ({unit})"
                )
                coefficient = 0.0
            generic_efficiency = _float(
                row,
                ("efficiency", "eta", "conversion_efficiency"),
                1.0,
            )
            couplings.append(
                EnergyCoupling(
                    table_name=table_name,
                    idx=coupling_idx,
                    name=coupling_name,
                    run_stat=_int(row, "run_stat", 1),
                    t1=EnergyTerminal(row_t1_domain, t1_type, t1_idx, t1_field),
                    t2=EnergyTerminal(row_t2_domain, t2_type, t2_idx, t2_field),
                    efficiency=float(generic_efficiency),
                    e2h_coeff=(
                        float(coefficient)
                        if (direction == "E2H" or electric_heat) and coefficient is not None
                        else None
                    ),
                    h2e_coeff=(
                        float(coefficient)
                        if direction == "H2E" and coefficient is not None
                        else None
                    ),
                    e2g_coeff=(
                        float(coefficient)
                        if direction == "E2G" and coefficient is not None
                        else None
                    ),
                    g2e_coeff=(
                        float(coefficient)
                        if direction == "G2E" and coefficient is not None
                        else None
                    ),
                    e2s_coeff=(
                        float(coefficient)
                        if direction == "E2S" and coefficient is not None
                        else None
                    ),
                    s2e_coeff=(
                        float(coefficient)
                        if direction == "S2E" and coefficient is not None
                        else None
                    ),
                    g2h_coeff=(
                        float(coefficient)
                        if gas_heat and coefficient is not None
                        else None
                    ),
                    energy_factor=_float(
                        row,
                        ("energy_factor", "conversion_factor", "heating_value", "calorific_value"),
                        None,
                    ),
                    control_type=control_type,
                    active=bool(row_alive and endpoints_alive),
                )
            )
    return couplings, warnings


def build_multi_energy_context_from_model(model, source=None) -> MultiEnergyContext:
    source_path = None if source is None else Path(source)
    networks: Dict[str, object] = {}
    warnings: List[str] = []
    for key, prefix in zip(FLUID_SYSTEM_KEYS, FLUID_SYSTEM_PREFIXES):
        if not (getattr(model, f"{prefix}Node", ()) or ()):
            continue
        try:
            networks[key] = _NETWORK_BUILDERS[key](model, source=source_path)
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{prefix} network is unavailable: {exc}")
    couplings, coupling_warnings = _parse_couplings(model)
    warnings.extend(coupling_warnings)
    return MultiEnergyContext(
        source=source_path,
        model=model,
        fluid_networks=networks,
        couplings=couplings,
        warnings=warnings,
    )


def build_multi_energy_context_from_rows(rows, source=None) -> MultiEnergyContext:
    effective_rows, overrides = propagate_composite_run_states(rows)
    context = build_multi_energy_context_from_model(
        efile_factory_from_rows(effective_rows),
        source=source,
    )
    context.warnings.extend(
        f"{item['dev_type']}:{item['dev_name']} disabled by "
        f"{item['ancestor_type']}:{item['ancestor_name']}"
        for item in overrides
        if str(item.get("dev_type", "")).startswith(FLUID_SYSTEM_PREFIXES)
    )
    return context


def load_multi_energy_context_from_e_file(file_name) -> MultiEnergyContext:
    return build_multi_energy_context_from_rows(
        _read_efile_rows(file_name),
        source=file_name,
    )


def attach_multi_energy_context(network, context: MultiEnergyContext):
    network.multi_energy = context
    network.fluid_networks = context.fluid_networks
    network.energy_couplings = context.couplings
    return network
