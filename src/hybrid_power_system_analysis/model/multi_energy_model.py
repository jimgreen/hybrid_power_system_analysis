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
    energy_factor: Optional[float] = None
    control_type: str = "MONITOR"
    active: bool = True

    @property
    def supports_energy_balance(self) -> bool:
        if self.is_hydrogen_electric_control:
            coefficient = self.hydrogen_electric_coeff
            return coefficient is not None and coefficient > 0.0
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
    def hydrogen_electric_coeff(self) -> Optional[float]:
        if self.hydrogen_electric_direction == "E2H":
            return self.e2h_coeff
        if self.hydrogen_electric_direction == "H2E":
            return self.h2e_coeff
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
    def controlled_terminal(self) -> Optional[EnergyTerminal]:
        if not self.is_hydrogen_electric_control:
            return None
        return self.electric_terminal if self.control_type == "P" else self.hydro_terminal

    @property
    def dependent_terminal(self) -> Optional[EnergyTerminal]:
        if not self.is_hydrogen_electric_control:
            return None
        return self.hydro_terminal if self.control_type == "P" else self.electric_terminal


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
    Electrolyzer efficiency is Nm3/kWh; fuel-cell efficiency is kWh/Nm3.
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
            if t1_domain in {"ac", "dc"} and t2_domain == "hydro":
                direction = "E2H"
            elif t1_domain == "hydro" and t2_domain in {"ac", "dc"}:
                direction = "H2E"
            default_control_type = (
                "FLOW"
                if direction == "E2H"
                else "P"
                if direction == "H2E"
                else "MONITOR"
            )
            control_type = normalize_energy_coupling_control_type(
                _text(row, "control_type", default_control_type)
            )
            coefficient_names = (
                (
                    "e2h_coeff",
                    "electric_to_hydrogen_efficiency",
                    "electric_gas_efficiency",
                    "e2h_efficiency",
                    "efficiency",
                    "eta",
                    "conversion_efficiency",
                )
                if direction == "E2H"
                else (
                    "h2e_coeff",
                    "hydrogen_to_electric_efficiency",
                    "gas_electric_efficiency",
                    "h2e_efficiency",
                    "efficiency",
                    "eta",
                    "conversion_efficiency",
                )
            )
            direct_control = (
                direction is not None
                and control_type in HYDROGEN_ELECTRIC_CONTROL_TYPES
            )
            coefficient = _float(
                row,
                coefficient_names,
                None if direct_control else 1.0,
            )
            if direct_control and (coefficient is None or coefficient <= 0.0):
                unit = "Nm3/kWh" if direction == "E2H" else "kWh/Nm3"
                field = "e2h_coeff" if direction == "E2H" else "h2e_coeff"
                warnings.append(
                    f"{table_name}:{coupling_name} requires positive {field} ({unit})"
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
                    e2h_coeff=(float(coefficient) if direction == "E2H" else None),
                    h2e_coeff=(float(coefficient) if direction == "H2E" else None),
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
