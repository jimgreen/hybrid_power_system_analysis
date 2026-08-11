"""Array-oriented steady-state models for heat, gas, and hydrogen networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from efile_read import efile_factory_from_file
from paths import resolve_project_file


PRESSURE_CONTROL = "PRESSURE"
FLOW_CONTROL = "FLOW"
PASSIVE_CONTROL = "PASSIVE"
RATIO_CONTROL = "RATIO"
GAIN_CONTROL = "GAIN"
CLOSED_CONTROL = "CLOSED"


def _float(row, name: str, default: float = 0.0) -> float:
    value = getattr(row, name, default)
    if value in (None, "", "-"):
        return float(default)
    return float(value)


def _int(row, name: str, default: int = 0) -> int:
    value = getattr(row, name, default)
    if value in (None, "", "-"):
        return int(default)
    return int(value)


def _optional_int(row, name: str) -> Optional[int]:
    value = getattr(row, name, None)
    if value in (None, "", "-"):
        return None
    return int(value)


def _text(row, name: str, default: str = "") -> str:
    value = getattr(row, name, default)
    return str(default if value in (None, "") else value)


def normalize_control_type(value, default: str = PASSIVE_CONTROL) -> str:
    text = str(value or default).strip().upper()
    aliases = {
        "P": PRESSURE_CONTROL,
        "V": PRESSURE_CONTROL,
        "PRESSURE": PRESSURE_CONTROL,
        "SLACK": PRESSURE_CONTROL,
        "Q": FLOW_CONTROL,
        "M": FLOW_CONTROL,
        "MASS": FLOW_CONTROL,
        "FLOW": FLOW_CONTROL,
        "FIXED_FLOW": FLOW_CONTROL,
        "OPEN": PASSIVE_CONTROL,
        "PASSIVE": PASSIVE_CONTROL,
        "NONE": PASSIVE_CONTROL,
        "RATIO": RATIO_CONTROL,
        "GAIN": GAIN_CONTROL,
        "BOOST": GAIN_CONTROL,
        "CLOSED": CLOSED_CONTROL,
        "OFF": CLOSED_CONTROL,
    }
    return aliases.get(text, text)


def _source_uses_explicit_return(item: "FluidSource") -> bool:
    return item.supply_node is not None and item.return_node is not None


def _load_uses_explicit_return(item: "FluidLoad") -> bool:
    return item.supply_node is not None and item.return_node is not None


def _exchanger_primary_is_explicit(item: "HeatExchanger") -> bool:
    return item.primary_supply_node is not None and item.primary_return_node is not None


def _exchanger_secondary_is_explicit(item: "HeatExchanger") -> bool:
    return item.secondary_return_node is not None and item.secondary_supply_node is not None


@dataclass(slots=True)
class FluidMedium:
    density: float = 0.8
    compressibility: float = 1.0
    molar_mass: float = 0.018
    temperature: float = 288.15
    heat_capacity: float = 4.186
    ambient_temperature: float = 20.0
    flow_factor: Optional[float] = None
    reference_temperature: float = 100.0
    reference_enthalpy: float = 2676.0
    ambient_enthalpy: float = 419.0
    feedwater_enthalpy: float = 419.0

    def conductance_factor(self, *, potential_power: int) -> float:
        if potential_power != 2:
            return 1.0
        if self.flow_factor is not None:
            return max(float(self.flow_factor), 1e-12)
        z = max(float(self.compressibility), 1e-12)
        temperature = max(float(self.temperature), 1e-12)
        molar_mass = max(float(self.molar_mass), 1e-12)
        return float(np.sqrt((molar_mass / 0.018) * (288.15 / temperature) / z))


@dataclass(slots=True)
class FluidNode:
    idx: int
    name: str
    pressure: float
    supply_temperature: float = 80.0
    return_temperature: float = 50.0
    enthalpy: float = 2800.0
    temperature: float = 100.0
    run_stat: int = 1


@dataclass(slots=True)
class FluidSource:
    idx: int
    name: str
    node: Optional[int]
    control_type: str
    pressure_set: float
    flow_set: float = 0.0
    alpha: float = 1.0
    flow_min: float = -np.inf
    flow_max: float = np.inf
    supply_temperature: float = 80.0
    enthalpy_set: float = 3000.0
    run_stat: int = 1
    supply_node: Optional[int] = None
    return_node: Optional[int] = None


@dataclass(slots=True)
class FluidStorage(FluidSource):
    """Bidirectional source-equivalent storage; positive flow discharges to the node."""


@dataclass(slots=True)
class FluidLoad:
    idx: int
    name: str
    node: Optional[int]
    flow_set: float
    heat_power: float = 0.0
    condensate_enthalpy: float = 419.0
    run_stat: int = 1
    supply_node: Optional[int] = None
    return_node: Optional[int] = None


@dataclass(slots=True)
class FluidEdge:
    idx: int
    name: str
    i_node: int
    j_node: int
    kind: str
    control_type: str = PASSIVE_CONTROL
    conductance: float = 1.0
    flow_set: float = 0.0
    ratio: float = 1.0
    pressure_gain: float = 0.0
    heat_loss: float = 0.0
    run_stat: int = 1


@dataclass(slots=True)
class HeatExchanger:
    idx: int
    name: str
    i_node: Optional[int]
    j_node: Optional[int]
    control_type: str
    primary_flow: float
    secondary_flow: float
    heat_set: float = 0.0
    effectiveness: float = 1.0
    heat_loss: float = 0.0
    run_stat: int = 1
    primary_supply_node: Optional[int] = None
    primary_return_node: Optional[int] = None
    secondary_return_node: Optional[int] = None
    secondary_supply_node: Optional[int] = None


@dataclass
class FluidNetwork:
    prefix: str
    potential_power: int
    thermal: bool = False
    steam: bool = False
    medium: FluidMedium = field(default_factory=FluidMedium)
    nodes: List[FluidNode] = field(default_factory=list)
    sources: List[FluidSource] = field(default_factory=list)
    storages: List[FluidStorage] = field(default_factory=list)
    loads: List[FluidLoad] = field(default_factory=list)
    pipes: List[FluidEdge] = field(default_factory=list)
    valves: List[FluidEdge] = field(default_factory=list)
    controllers: List[FluidEdge] = field(default_factory=list)
    heat_exchangers: List[HeatExchanger] = field(default_factory=list)
    explicit_return: bool = False
    source: Optional[Path] = None

    def prepare(self) -> "FluidNetwork":
        known_sources = {id(item) for item in self.sources}
        self.sources.extend(
            item for item in self.storages if id(item) not in known_sources
        )
        live_nodes = [node for node in self.nodes if int(node.run_stat) == 1]
        if not live_nodes:
            raise RuntimeError(f"{self.prefix} network has no live nodes")
        self.nodes = live_nodes
        self.node_idx = np.asarray([node.idx for node in live_nodes], dtype=np.int64)
        self.node_name = np.asarray([node.name for node in live_nodes], dtype=object)
        self.node_pressure = np.asarray([node.pressure for node in live_nodes], dtype=np.float64)
        self.node_temperature = np.asarray([node.temperature for node in live_nodes], dtype=np.float64)
        self.node_supply_temperature = np.asarray(
            [node.supply_temperature for node in live_nodes], dtype=np.float64
        )
        self.node_return_temperature = np.asarray(
            [node.return_temperature for node in live_nodes], dtype=np.float64
        )
        self.node_enthalpy = np.asarray([node.enthalpy for node in live_nodes], dtype=np.float64)
        self.node_pos_by_idx = {int(idx): pos for pos, idx in enumerate(self.node_idx.tolist())}
        self.node_pos_by_name = {str(name): pos for pos, name in enumerate(self.node_name.tolist())}

        if self.thermal:
            self.sources = [
                item
                for item in self.sources
                if int(item.run_stat) == 1
                and (
                    (
                        _source_uses_explicit_return(item)
                        and int(item.supply_node) in self.node_pos_by_idx
                        and int(item.return_node) in self.node_pos_by_idx
                    )
                    or (
                        not _source_uses_explicit_return(item)
                        and int(item.node) in self.node_pos_by_idx
                    )
                )
            ]
            self.loads = [
                item
                for item in self.loads
                if int(item.run_stat) == 1
                and (
                    (
                        _load_uses_explicit_return(item)
                        and int(item.supply_node) in self.node_pos_by_idx
                        and int(item.return_node) in self.node_pos_by_idx
                    )
                    or (
                        not _load_uses_explicit_return(item)
                        and int(item.node) in self.node_pos_by_idx
                    )
                )
            ]
            self.heat_exchangers = [
                item
                for item in self.heat_exchangers
                if int(item.run_stat) == 1
                and int(
                    item.primary_supply_node
                    if _exchanger_primary_is_explicit(item)
                    else item.i_node
                )
                in self.node_pos_by_idx
                and int(
                    item.primary_return_node
                    if _exchanger_primary_is_explicit(item)
                    else item.i_node
                )
                in self.node_pos_by_idx
                and int(
                    item.secondary_return_node
                    if _exchanger_secondary_is_explicit(item)
                    else item.j_node
                )
                in self.node_pos_by_idx
                and int(
                    item.secondary_supply_node
                    if _exchanger_secondary_is_explicit(item)
                    else item.j_node
                )
                in self.node_pos_by_idx
            ]
        else:
            self.sources = [
                item
                for item in self.sources
                if int(item.run_stat) == 1 and int(item.node) in self.node_pos_by_idx
            ]
            self.loads = [
                item
                for item in self.loads
                if int(item.run_stat) == 1 and int(item.node) in self.node_pos_by_idx
            ]
            self.heat_exchangers = []
        self.storages = [
            item for item in self.sources if isinstance(item, FluidStorage)
        ]
        all_edges = [*self.pipes, *self.valves, *self.controllers]
        self.edges = [
            edge
            for edge in all_edges
            if int(edge.run_stat) == 1
            and normalize_control_type(edge.control_type) != CLOSED_CONTROL
            and int(edge.i_node) in self.node_pos_by_idx
            and int(edge.j_node) in self.node_pos_by_idx
        ]
        self.edge_name = np.asarray([edge.name for edge in self.edges], dtype=object)
        self.edge_kind = np.asarray([edge.kind for edge in self.edges], dtype=object)
        self.edge_control_type = np.asarray(
            [normalize_control_type(edge.control_type) for edge in self.edges], dtype=object
        )
        self.edge_i = np.asarray(
            [self.node_pos_by_idx[int(edge.i_node)] for edge in self.edges], dtype=np.int64
        )
        self.edge_j = np.asarray(
            [self.node_pos_by_idx[int(edge.j_node)] for edge in self.edges], dtype=np.int64
        )
        factor = self.medium.conductance_factor(potential_power=self.potential_power)
        self.edge_conductance = np.asarray(
            [max(float(edge.conductance) * factor, 1e-12) for edge in self.edges], dtype=np.float64
        )
        self.edge_flow_set = np.asarray([edge.flow_set for edge in self.edges], dtype=np.float64)
        self.edge_ratio = np.asarray([max(edge.ratio, 1e-12) for edge in self.edges], dtype=np.float64)
        self.edge_pressure_gain = np.asarray([edge.pressure_gain for edge in self.edges], dtype=np.float64)
        self.edge_heat_loss = np.asarray([max(edge.heat_loss, 0.0) for edge in self.edges], dtype=np.float64)
        self.edge_pos_by_name = {str(name): pos for pos, name in enumerate(self.edge_name.tolist())}

        self.passive_edge_pos = np.flatnonzero(self.edge_control_type == PASSIVE_CONTROL).astype(np.int64)
        self.fixed_flow_edge_pos = np.flatnonzero(self.edge_control_type == FLOW_CONTROL).astype(np.int64)
        regulated_mask = np.isin(self.edge_control_type, (RATIO_CONTROL, GAIN_CONTROL))
        self.regulated_edge_pos = np.flatnonzero(regulated_mask).astype(np.int64)

        n_node = len(self.nodes)
        n_edge = len(self.edges)
        if n_edge:
            cols = np.repeat(np.arange(n_edge, dtype=np.int64), 2)
            rows = np.empty(2 * n_edge, dtype=np.int64)
            rows[0::2] = self.edge_i
            rows[1::2] = self.edge_j
            values = np.tile(np.asarray([-1.0, 1.0]), n_edge)
            self.incidence = coo_matrix((values, (rows, cols)), shape=(n_node, n_edge)).tocsr()
        else:
            self.incidence = csr_matrix((n_node, 0), dtype=np.float64)

        if n_edge:
            pressure_adjacency = abs(self.incidence) @ abs(self.incidence).T
            pressure_adjacency.setdiag(0.0)
            pressure_adjacency.eliminate_zeros()
            self.island_count, self.node_island = connected_components(
                pressure_adjacency, directed=False
            )
        else:
            self.island_count = n_node
            self.node_island = np.arange(n_node, dtype=np.int32)

        topology_i = self.edge_i.tolist()
        topology_j = self.edge_j.tolist()
        if self.thermal:
            for item in self.sources:
                if _source_uses_explicit_return(item):
                    topology_i.append(self.node_pos_by_idx[int(item.return_node)])
                    topology_j.append(self.node_pos_by_idx[int(item.supply_node)])
            for item in self.loads:
                if _load_uses_explicit_return(item):
                    topology_i.append(self.node_pos_by_idx[int(item.supply_node)])
                    topology_j.append(self.node_pos_by_idx[int(item.return_node)])
            for item in self.heat_exchangers:
                if _exchanger_primary_is_explicit(item):
                    topology_i.append(self.node_pos_by_idx[int(item.primary_supply_node)])
                    topology_j.append(self.node_pos_by_idx[int(item.primary_return_node)])
                if _exchanger_secondary_is_explicit(item):
                    topology_i.append(self.node_pos_by_idx[int(item.secondary_return_node)])
                    topology_j.append(self.node_pos_by_idx[int(item.secondary_supply_node)])
        if topology_i:
            rows = np.asarray([*topology_i, *topology_j], dtype=np.int64)
            cols = np.asarray([*topology_j, *topology_i], dtype=np.int64)
            adjacency = coo_matrix(
                (np.ones(rows.size, dtype=np.float64), (rows, cols)),
                shape=(n_node, n_node),
            ).tocsr()
            self.thermal_circuit_count, self.node_thermal_circuit = connected_components(
                adjacency, directed=False
            )
        else:
            self.thermal_circuit_count = n_node
            self.node_thermal_circuit = np.arange(n_node, dtype=np.int32)

        if self.thermal:
            island_modes: Dict[int, bool] = {}

            def register_node_mode(node_idx: int, explicit: bool, device_name: str) -> None:
                node_pos = self.node_pos_by_idx[int(node_idx)]
                island = int(self.node_island[node_pos])
                previous = island_modes.get(island)
                if previous is not None and previous != bool(explicit):
                    raise ValueError(
                        f"heat hydraulic island {island} mixes explicit and implicit return models; "
                        f"conflict found at {device_name}"
                    )
                island_modes[island] = bool(explicit)

            for item in self.sources:
                explicit = _source_uses_explicit_return(item)
                if explicit:
                    register_node_mode(int(item.supply_node), True, item.name)
                    register_node_mode(int(item.return_node), True, item.name)
                else:
                    register_node_mode(int(item.node), False, item.name)
            for item in self.loads:
                explicit = _load_uses_explicit_return(item)
                if explicit:
                    register_node_mode(int(item.supply_node), True, item.name)
                    register_node_mode(int(item.return_node), True, item.name)
                else:
                    register_node_mode(int(item.node), False, item.name)
            for item in self.heat_exchangers:
                primary_explicit = _exchanger_primary_is_explicit(item)
                secondary_explicit = _exchanger_secondary_is_explicit(item)
                if primary_explicit:
                    register_node_mode(int(item.primary_supply_node), True, item.name)
                    register_node_mode(int(item.primary_return_node), True, item.name)
                else:
                    register_node_mode(int(item.i_node), False, item.name)
                if secondary_explicit:
                    register_node_mode(int(item.secondary_return_node), True, item.name)
                    register_node_mode(int(item.secondary_supply_node), True, item.name)
                else:
                    register_node_mode(int(item.j_node), False, item.name)
            self.node_explicit_return = np.asarray(
                [island_modes.get(int(island), False) for island in self.node_island.tolist()],
                dtype=bool,
            )
            self.explicit_return = bool(np.any(self.node_explicit_return))
            self.mixed_return = bool(
                self.explicit_return and np.any(~self.node_explicit_return)
            )
            self.supply_temperature_state_by_node = np.arange(n_node, dtype=np.int64)
            self.return_temperature_state_by_node = self.supply_temperature_state_by_node.copy()
            implicit_nodes = np.flatnonzero(~self.node_explicit_return).astype(np.int64)
            self.return_temperature_state_by_node[implicit_nodes] = (
                n_node + np.arange(implicit_nodes.size, dtype=np.int64)
            )
            self.temperature_state_count = n_node + implicit_nodes.size
            self.initial_temperature_state = np.empty(
                self.temperature_state_count, dtype=np.float64
            )
            self.initial_temperature_state[:n_node] = np.where(
                self.node_explicit_return,
                self.node_temperature,
                self.node_supply_temperature,
            )
            if implicit_nodes.size:
                self.initial_temperature_state[n_node:] = self.node_return_temperature[
                    implicit_nodes
                ]
        else:
            self.node_explicit_return = np.zeros(n_node, dtype=bool)
            self.mixed_return = False
            self.supply_temperature_state_by_node = np.empty(0, dtype=np.int64)
            self.return_temperature_state_by_node = np.empty(0, dtype=np.int64)
            self.temperature_state_count = 0
            self.initial_temperature_state = np.empty(0, dtype=np.float64)

        if self.thermal:
            self.source_explicit_return = np.asarray(
                [_source_uses_explicit_return(item) for item in self.sources], dtype=bool
            )
            self.load_explicit_return = np.asarray(
                [_load_uses_explicit_return(item) for item in self.loads], dtype=bool
            )
            self.source_supply_node_pos = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.supply_node if explicit else item.node)
                    ]
                    for item, explicit in zip(self.sources, self.source_explicit_return.tolist())
                ],
                dtype=np.int64,
            )
            self.source_return_node_pos = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.return_node if explicit else item.node)
                    ]
                    for item, explicit in zip(self.sources, self.source_explicit_return.tolist())
                ],
                dtype=np.int64,
            )
            self.load_supply_node_pos = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.supply_node if explicit else item.node)
                    ]
                    for item, explicit in zip(self.loads, self.load_explicit_return.tolist())
                ],
                dtype=np.int64,
            )
            self.load_return_node_pos = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.return_node if explicit else item.node)
                    ]
                    for item, explicit in zip(self.loads, self.load_explicit_return.tolist())
                ],
                dtype=np.int64,
            )
            self.source_node_pos = self.source_supply_node_pos
            self.load_node_pos = self.load_supply_node_pos
        else:
            self.source_explicit_return = np.zeros(len(self.sources), dtype=bool)
            self.load_explicit_return = np.zeros(len(self.loads), dtype=bool)
            self.source_node_pos = np.asarray(
                [self.node_pos_by_idx[int(item.node)] for item in self.sources], dtype=np.int64
            )
            self.load_node_pos = np.asarray(
                [self.node_pos_by_idx[int(item.node)] for item in self.loads], dtype=np.int64
            )
            self.source_supply_node_pos = self.source_node_pos
            self.source_return_node_pos = self.source_node_pos
            self.load_supply_node_pos = self.load_node_pos
            self.load_return_node_pos = self.load_node_pos
        self.source_name = np.asarray([item.name for item in self.sources], dtype=object)
        self.source_is_storage = np.asarray(
            [isinstance(item, FluidStorage) for item in self.sources],
            dtype=bool,
        )
        self.storage_source_pos = np.flatnonzero(self.source_is_storage).astype(np.int64)
        self.storage_name = self.source_name[self.storage_source_pos]
        self.load_name = np.asarray([item.name for item in self.loads], dtype=object)
        self.source_pos_by_name = {str(name): pos for pos, name in enumerate(self.source_name.tolist())}
        self.storage_pos_by_name = {
            str(self.source_name[pos]): int(pos)
            for pos in self.storage_source_pos.tolist()
        }
        self.load_pos_by_name = {str(name): pos for pos, name in enumerate(self.load_name.tolist())}
        self.source_control_type = np.asarray(
            [normalize_control_type(item.control_type, FLOW_CONTROL) for item in self.sources], dtype=object
        )
        self.source_pressure_set = np.asarray([item.pressure_set for item in self.sources], dtype=np.float64)
        self.source_flow_set = np.asarray([item.flow_set for item in self.sources], dtype=np.float64)
        self.source_alpha = np.asarray([max(item.alpha, 0.0) for item in self.sources], dtype=np.float64)
        self.source_flow_min = np.asarray([item.flow_min for item in self.sources], dtype=np.float64)
        self.source_flow_max = np.asarray([item.flow_max for item in self.sources], dtype=np.float64)
        self.source_supply_temperature = np.asarray(
            [item.supply_temperature for item in self.sources], dtype=np.float64
        )
        self.source_enthalpy_set = np.asarray([item.enthalpy_set for item in self.sources], dtype=np.float64)
        self.load_flow_set = np.asarray([max(item.flow_set, 0.0) for item in self.loads], dtype=np.float64)
        self.load_heat_power = np.asarray([max(item.heat_power, 0.0) for item in self.loads], dtype=np.float64)
        self.load_condensate_enthalpy = np.asarray(
            [item.condensate_enthalpy for item in self.loads], dtype=np.float64
        )
        self.exchanger_name = np.asarray([item.name for item in self.heat_exchangers], dtype=object)
        self.exchanger_pos_by_name = {
            str(name): pos for pos, name in enumerate(self.exchanger_name.tolist())
        }
        if self.thermal:
            self.exchanger_primary_explicit = np.asarray(
                [_exchanger_primary_is_explicit(item) for item in self.heat_exchangers], dtype=bool
            )
            self.exchanger_secondary_explicit = np.asarray(
                [_exchanger_secondary_is_explicit(item) for item in self.heat_exchangers], dtype=bool
            )
            self.exchanger_primary_supply = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.primary_supply_node if explicit else item.i_node)
                    ]
                    for item, explicit in zip(
                        self.heat_exchangers, self.exchanger_primary_explicit.tolist()
                    )
                ],
                dtype=np.int64,
            )
            self.exchanger_primary_return = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.primary_return_node if explicit else item.i_node)
                    ]
                    for item, explicit in zip(
                        self.heat_exchangers, self.exchanger_primary_explicit.tolist()
                    )
                ],
                dtype=np.int64,
            )
            self.exchanger_secondary_return = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.secondary_return_node if explicit else item.j_node)
                    ]
                    for item, explicit in zip(
                        self.heat_exchangers, self.exchanger_secondary_explicit.tolist()
                    )
                ],
                dtype=np.int64,
            )
            self.exchanger_secondary_supply = np.asarray(
                [
                    self.node_pos_by_idx[
                        int(item.secondary_supply_node if explicit else item.j_node)
                    ]
                    for item, explicit in zip(
                        self.heat_exchangers, self.exchanger_secondary_explicit.tolist()
                    )
                ],
                dtype=np.int64,
            )
            self.exchanger_i = self.exchanger_primary_supply
            self.exchanger_j = self.exchanger_secondary_return
        else:
            self.exchanger_primary_explicit = np.zeros(0, dtype=bool)
            self.exchanger_secondary_explicit = np.zeros(0, dtype=bool)
            self.exchanger_i = np.asarray(
                [self.node_pos_by_idx[int(item.i_node)] for item in self.heat_exchangers], dtype=np.int64
            )
            self.exchanger_j = np.asarray(
                [self.node_pos_by_idx[int(item.j_node)] for item in self.heat_exchangers], dtype=np.int64
            )
        self.exchanger_control_type = np.asarray(
            [str(item.control_type).strip().upper() for item in self.heat_exchangers], dtype=object
        )
        self.exchanger_primary_flow = np.asarray(
            [max(item.primary_flow, 0.0) for item in self.heat_exchangers], dtype=np.float64
        )
        self.exchanger_secondary_flow = np.asarray(
            [max(item.secondary_flow, 0.0) for item in self.heat_exchangers], dtype=np.float64
        )
        self.exchanger_heat_set = np.asarray(
            [max(item.heat_set, 0.0) for item in self.heat_exchangers], dtype=np.float64
        )
        self.exchanger_effectiveness = np.asarray(
            [min(max(item.effectiveness, 0.0), 1.0) for item in self.heat_exchangers], dtype=np.float64
        )
        self.exchanger_heat_loss = np.asarray(
            [min(max(item.heat_loss, 0.0), 1.0) for item in self.heat_exchangers], dtype=np.float64
        )

        self.fixed_injection = np.zeros(n_node, dtype=np.float64)
        self.demand = np.zeros(n_node, dtype=np.float64)
        flow_sources = np.flatnonzero(self.source_control_type == FLOW_CONTROL)
        if self.thermal:
            if flow_sources.size:
                explicit = flow_sources[self.source_explicit_return[flow_sources]]
                implicit = flow_sources[~self.source_explicit_return[flow_sources]]
                if explicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.source_return_node_pos[explicit],
                        -self.source_flow_set[explicit],
                    )
                    np.add.at(
                        self.fixed_injection,
                        self.source_supply_node_pos[explicit],
                        self.source_flow_set[explicit],
                    )
                if implicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.source_node_pos[implicit],
                        self.source_flow_set[implicit],
                    )
            if self.load_node_pos.size:
                explicit = np.flatnonzero(self.load_explicit_return)
                implicit = np.flatnonzero(~self.load_explicit_return)
                if explicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.load_supply_node_pos[explicit],
                        -self.load_flow_set[explicit],
                    )
                    np.add.at(
                        self.fixed_injection,
                        self.load_return_node_pos[explicit],
                        self.load_flow_set[explicit],
                    )
                if implicit.size:
                    np.add.at(self.demand, self.load_node_pos[implicit], self.load_flow_set[implicit])
            if self.exchanger_i.size:
                primary_explicit = np.flatnonzero(self.exchanger_primary_explicit)
                primary_implicit = np.flatnonzero(~self.exchanger_primary_explicit)
                secondary_explicit = np.flatnonzero(self.exchanger_secondary_explicit)
                secondary_implicit = np.flatnonzero(~self.exchanger_secondary_explicit)
                if primary_explicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.exchanger_primary_supply[primary_explicit],
                        -self.exchanger_primary_flow[primary_explicit],
                    )
                    np.add.at(
                        self.fixed_injection,
                        self.exchanger_primary_return[primary_explicit],
                        self.exchanger_primary_flow[primary_explicit],
                    )
                if primary_implicit.size:
                    np.add.at(
                        self.demand,
                        self.exchanger_primary_supply[primary_implicit],
                        self.exchanger_primary_flow[primary_implicit],
                    )
                if secondary_explicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.exchanger_secondary_return[secondary_explicit],
                        -self.exchanger_secondary_flow[secondary_explicit],
                    )
                    np.add.at(
                        self.fixed_injection,
                        self.exchanger_secondary_supply[secondary_explicit],
                        self.exchanger_secondary_flow[secondary_explicit],
                    )
                if secondary_implicit.size:
                    np.add.at(
                        self.fixed_injection,
                        self.exchanger_secondary_supply[secondary_implicit],
                        self.exchanger_secondary_flow[secondary_implicit],
                    )
        else:
            if flow_sources.size:
                np.add.at(
                    self.fixed_injection,
                    self.source_node_pos[flow_sources],
                    self.source_flow_set[flow_sources],
                )
            if self.load_node_pos.size:
                np.add.at(self.demand, self.load_node_pos, self.load_flow_set)

        pressure_sources = np.flatnonzero(self.source_control_type == PRESSURE_CONTROL).astype(np.int64)
        fixed_pressure_by_node: Dict[int, List[float]] = {}
        for source_pos in pressure_sources.tolist():
            node_pos = int(self.source_node_pos[source_pos])
            fixed_pressure_by_node.setdefault(node_pos, []).append(float(self.source_pressure_set[source_pos]))
        self.warnings = []
        for island in range(int(self.island_count)):
            island_nodes = np.flatnonzero(self.node_island == island)
            if any(int(node) in fixed_pressure_by_node for node in island_nodes.tolist()):
                continue
            anchor = int(island_nodes[0])
            fixed_pressure_by_node[anchor] = [float(self.node_pressure[anchor])]
            self.warnings.append(
                f"island {island} has no pressure-controlled source; node {self.node_name[anchor]} is the pressure anchor"
            )
        self.fixed_node_pos = np.asarray(sorted(fixed_pressure_by_node), dtype=np.int64)
        self.fixed_pressure = np.asarray(
            [np.mean(fixed_pressure_by_node[int(pos)]) for pos in self.fixed_node_pos], dtype=np.float64
        )
        self.fixed_potential = self.fixed_pressure ** int(self.potential_power)
        fixed_mask = np.zeros(n_node, dtype=bool)
        fixed_mask[self.fixed_node_pos] = True
        self.free_node_pos = np.flatnonzero(~fixed_mask).astype(np.int64)
        self.free_state_by_node = np.full(n_node, -1, dtype=np.int64)
        self.free_state_by_node[self.free_node_pos] = np.arange(self.free_node_pos.size, dtype=np.int64)
        self.regulated_state_by_edge = np.full(n_edge, -1, dtype=np.int64)
        self.regulated_state_by_edge[self.regulated_edge_pos] = (
            self.free_node_pos.size + np.arange(self.regulated_edge_pos.size, dtype=np.int64)
        )
        if self.thermal:
            explicit_pressure_sources = pressure_sources[
                self.source_explicit_return[pressure_sources]
            ]
            self.pressure_source_group_nodes = np.asarray(
                sorted(
                    {
                        int(self.source_supply_node_pos[pos])
                        for pos in explicit_pressure_sources.tolist()
                    }
                ),
                dtype=np.int64,
            )
            self.pressure_source_groups = [
                explicit_pressure_sources[
                    self.source_supply_node_pos[explicit_pressure_sources] == int(node_pos)
                ].astype(np.int64)
                for node_pos in self.pressure_source_group_nodes.tolist()
            ]
            self.pressure_source_group_by_source = np.full(len(self.sources), -1, dtype=np.int64)
            for group_pos, source_positions in enumerate(self.pressure_source_groups):
                self.pressure_source_group_by_source[source_positions] = group_pos
            balance_nodes = np.concatenate((self.free_node_pos, self.pressure_source_group_nodes))
            self.balance_node_pos = np.asarray(sorted(set(balance_nodes.tolist())), dtype=np.int64)
            self.balance_row_by_node = np.full(n_node, -1, dtype=np.int64)
            self.balance_row_by_node[self.balance_node_pos] = np.arange(
                self.balance_node_pos.size, dtype=np.int64
            )
        else:
            self.pressure_source_group_nodes = np.empty(0, dtype=np.int64)
            self.pressure_source_groups = []
            self.pressure_source_group_by_source = np.full(len(self.sources), -1, dtype=np.int64)
            self.balance_node_pos = self.free_node_pos.copy()
            self.balance_row_by_node = self.free_state_by_node.copy()
        self.prepared = True
        return self

    def initial_potential(self) -> np.ndarray:
        pressure = np.maximum(self.node_pressure, 1e-6)
        potential = pressure ** int(self.potential_power)
        if self.fixed_node_pos.size:
            potential[self.fixed_node_pos] = self.fixed_potential
        return potential


def _medium_from_model(model, prefix: str, thermal: bool, steam: bool = False) -> FluidMedium:
    rows = list(getattr(model, f"{prefix}Medium", ()) or ())
    if not rows:
        return FluidMedium(
            density=998.0 if thermal else (4.0 if steam else (0.08375 if prefix == "Hydro" else 0.8)),
            molar_mass=0.002016 if prefix == "Hydro" else 0.018,
            heat_capacity=2.08 if steam else 4.186,
        )
    row = rows[0]
    flow_factor_value = getattr(row, "flow_factor", None)
    return FluidMedium(
        density=_float(row, "density", 998.0 if thermal else 0.8),
        compressibility=_float(row, "compressibility", 1.0),
        molar_mass=_float(row, "molar_mass", 0.002016 if prefix == "Hydro" else 0.018),
        temperature=_float(row, "temperature", 288.15),
        heat_capacity=_float(row, "heat_capacity", 4.186),
        ambient_temperature=_float(row, "ambient_temperature", 20.0),
        flow_factor=None if flow_factor_value in (None, "", "-") else float(flow_factor_value),
        reference_temperature=_float(row, "reference_temperature", 100.0),
        reference_enthalpy=_float(row, "reference_enthalpy", 2676.0),
        ambient_enthalpy=_float(row, "ambient_enthalpy", 419.0),
        feedwater_enthalpy=_float(row, "feedwater_enthalpy", 419.0),
    )


def _parse_nodes(model, prefix: str, thermal: bool) -> List[FluidNode]:
    nodes = []
    for row in getattr(model, f"{prefix}Node", ()) or ():
        nodes.append(
            FluidNode(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"{prefix.lower()}_node_{_int(row, 'idx')}"),
                pressure=_float(row, "pressure", _float(row, "p_set", 1.0)),
                supply_temperature=_float(row, "supply_temperature", 80.0 if thermal else 20.0),
                return_temperature=_float(row, "return_temperature", 50.0 if thermal else 20.0),
                enthalpy=_float(row, "enthalpy", 2800.0),
                temperature=_float(row, "temperature", 100.0),
                run_stat=_int(row, "run_stat", 1),
            )
        )
    return nodes


def _parse_sources(model, prefix: str, thermal: bool) -> List[FluidSource]:
    sources = []
    for row in getattr(model, f"{prefix}Source", ()) or ():
        sources.append(
            FluidSource(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"{prefix.lower()}_source_{_int(row, 'idx')}"),
                node=_optional_int(row, "node"),
                control_type=normalize_control_type(_text(row, "control_type", PRESSURE_CONTROL)),
                pressure_set=_float(row, "pressure_set", _float(row, "p_set", 1.0)),
                flow_set=_float(row, "flow_set", 0.0),
                alpha=_float(row, "alpha", 1.0),
                flow_min=_float(row, "flow_min", -np.inf),
                flow_max=_float(row, "flow_max", np.inf),
                supply_temperature=_float(row, "supply_temperature", 80.0 if thermal else 20.0),
                enthalpy_set=_float(row, "enthalpy_set", _float(row, "h_set", 3000.0)),
                run_stat=_int(row, "run_stat", 1),
                supply_node=_optional_int(row, "supply_node"),
                return_node=_optional_int(row, "return_node"),
            )
        )
    return sources


def _parse_storages(model, prefix: str, thermal: bool) -> List[FluidStorage]:
    storages = []
    for row in getattr(model, f"{prefix}Storage", ()) or ():
        max_charge_flow = abs(_float(row, "max_charge_flow", np.inf))
        max_discharge_flow = abs(_float(row, "max_discharge_flow", np.inf))
        storages.append(
            FluidStorage(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"{prefix.lower()}_storage_{_int(row, 'idx')}"),
                node=_optional_int(row, "node"),
                control_type=normalize_control_type(
                    _text(row, "control_type", FLOW_CONTROL),
                    FLOW_CONTROL,
                ),
                pressure_set=_float(row, "pressure_set", _float(row, "p_set", 1.0)),
                flow_set=_float(row, "flow_set", _float(row, "mass_flow", 0.0)),
                alpha=_float(row, "alpha", 1.0),
                flow_min=_float(row, "flow_min", -max_charge_flow),
                flow_max=_float(row, "flow_max", max_discharge_flow),
                supply_temperature=_float(row, "supply_temperature", 80.0 if thermal else 20.0),
                enthalpy_set=_float(row, "enthalpy_set", _float(row, "h_set", 3000.0)),
                run_stat=_int(row, "run_stat", 1),
                supply_node=_optional_int(row, "supply_node"),
                return_node=_optional_int(row, "return_node"),
            )
        )
    return storages


def _parse_loads(model, prefix: str) -> List[FluidLoad]:
    loads = []
    for row in getattr(model, f"{prefix}Load", ()) or ():
        loads.append(
            FluidLoad(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"{prefix.lower()}_load_{_int(row, 'idx')}"),
                node=_optional_int(row, "node"),
                flow_set=_float(row, "flow_set", _float(row, "mass_flow", 0.0)),
                heat_power=_float(row, "heat_power", 0.0),
                condensate_enthalpy=_float(row, "condensate_enthalpy", 419.0),
                run_stat=_int(row, "run_stat", 1),
                supply_node=_optional_int(row, "supply_node"),
                return_node=_optional_int(row, "return_node"),
            )
        )
    return loads


def _parse_edges(model, prefix: str, block_suffix: str, kind: str, default_control: str) -> List[FluidEdge]:
    edges = []
    for row in getattr(model, f"{prefix}{block_suffix}", ()) or ():
        edges.append(
            FluidEdge(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"{prefix.lower()}_{kind}_{_int(row, 'idx')}"),
                i_node=_int(row, "i_node"),
                j_node=_int(row, "j_node"),
                kind=kind,
                control_type=normalize_control_type(_text(row, "control_type", default_control), default_control),
                conductance=_float(row, "conductance", 1.0),
                flow_set=_float(row, "flow_set", 0.0),
                ratio=_float(row, "ratio", 1.0),
                pressure_gain=_float(row, "pressure_gain", _float(row, "gain", 0.0)),
                heat_loss=_float(row, "heat_loss", 0.0),
                run_stat=_int(row, "run_stat", 1),
            )
        )
    return edges


def _parse_heat_exchangers(model) -> List[HeatExchanger]:
    exchangers = []
    for row in getattr(model, "HeatExchanger", ()) or ():
        exchangers.append(
            HeatExchanger(
                idx=_int(row, "idx"),
                name=_text(row, "name", f"heat_exchanger_{_int(row, 'idx')}"),
                i_node=_optional_int(row, "i_node"),
                j_node=_optional_int(row, "j_node"),
                control_type=_text(row, "control_type", "EFFECTIVENESS").strip().upper(),
                primary_flow=_float(row, "primary_flow", _float(row, "i_flow", 0.0)),
                secondary_flow=_float(row, "secondary_flow", _float(row, "j_flow", 0.0)),
                heat_set=_float(row, "heat_set", 0.0),
                effectiveness=_float(row, "effectiveness", 1.0),
                heat_loss=_float(row, "heat_loss", 0.0),
                run_stat=_int(row, "run_stat", 1),
                primary_supply_node=_optional_int(row, "primary_supply_node"),
                primary_return_node=_optional_int(row, "primary_return_node"),
                secondary_return_node=_optional_int(row, "secondary_return_node"),
                secondary_supply_node=_optional_int(row, "secondary_supply_node"),
            )
        )
    return exchangers


def _validate_heat_ports(
    sources: Sequence[FluidSource],
    loads: Sequence[FluidLoad],
    exchangers: Sequence[HeatExchanger],
) -> bool:
    any_explicit = False

    def register(name: str, implicit_ports, explicit_ports) -> None:
        nonlocal any_explicit
        implicit = all(value is not None for value in implicit_ports)
        explicit = all(value is not None for value in explicit_ports)
        if implicit == explicit:
            raise ValueError(
                f"{name} must define exactly one complete heat-port layout: "
                "node/i_node+j_node for implicit return, or explicit supply/return ports"
            )
        any_explicit = any_explicit or explicit

    for item in sources:
        if int(item.run_stat) == 1:
            register(item.name, (item.node,), (item.supply_node, item.return_node))
    for item in loads:
        if int(item.run_stat) == 1:
            register(item.name, (item.node,), (item.supply_node, item.return_node))
    for item in exchangers:
        if int(item.run_stat) == 1:
            register(
                f"{item.name} primary side",
                (item.i_node,),
                (item.primary_supply_node, item.primary_return_node),
            )
            register(
                f"{item.name} secondary side",
                (item.j_node,),
                (item.secondary_return_node, item.secondary_supply_node),
            )
    return any_explicit


def build_fluid_network_from_model(
    model,
    *,
    prefix: str,
    potential_power: int,
    thermal: bool = False,
    steam: bool = False,
    controller_suffix: Optional[str] = None,
    source: Optional[Path] = None,
) -> FluidNetwork:
    controller_suffix = controller_suffix or ("Pump" if thermal else "Compressor")
    sources = _parse_sources(model, prefix, thermal)
    storages = _parse_storages(model, prefix, thermal)
    loads = _parse_loads(model, prefix)
    heat_exchangers = _parse_heat_exchangers(model) if thermal else []
    network = FluidNetwork(
        prefix=prefix,
        potential_power=int(potential_power),
        thermal=bool(thermal),
        steam=bool(steam),
        medium=_medium_from_model(model, prefix, thermal, steam),
        nodes=_parse_nodes(model, prefix, thermal),
        sources=sources,
        storages=storages,
        loads=loads,
        pipes=_parse_edges(model, prefix, "Pipe", "pipe", PASSIVE_CONTROL),
        valves=_parse_edges(model, prefix, "Valve", "valve", PASSIVE_CONTROL),
        controllers=_parse_edges(
            model,
            prefix,
            controller_suffix,
            "pump" if thermal else ("pressure_reducer" if steam else "compressor"),
            GAIN_CONTROL if thermal else RATIO_CONTROL,
        ),
        heat_exchangers=heat_exchangers,
        explicit_return=(
            _validate_heat_ports([*sources, *storages], loads, heat_exchangers)
            if thermal
            else False
        ),
        source=source,
    )
    return network.prepare()


def load_fluid_network_from_e_file(
    file_name,
    *,
    prefix: str,
    potential_power: int,
    thermal: bool = False,
    steam: bool = False,
    controller_suffix: Optional[str] = None,
) -> FluidNetwork:
    path = resolve_project_file(file_name).resolve()
    model = efile_factory_from_file(path)
    return build_fluid_network_from_model(
        model,
        prefix=prefix,
        potential_power=potential_power,
        thermal=thermal,
        steam=steam,
        controller_suffix=controller_suffix,
        source=path,
    )


def iter_device_names(network: FluidNetwork) -> Iterable[str]:
    yield from network.node_name.tolist()
    yield from network.edge_name.tolist()
    yield from network.source_name.tolist()
    yield from network.load_name.tolist()
