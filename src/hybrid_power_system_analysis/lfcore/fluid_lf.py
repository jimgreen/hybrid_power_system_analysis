"""Shared sparse Newton load-flow kernel for steady fluid networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence
import sys

import numpy as np
from scipy.sparse import bmat, coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from lfcore.common import allocate_limited_residual, normalize_result_mode
from model.fluid_model import (
    GAIN_CONTROL,
    PRESSURE_CONTROL,
    RATIO_CONTROL,
    FluidNetwork,
    FluidStorage,
)


THERMAL_ZERO_FLOW_TOLERANCE = 1.0e-10


@dataclass
class FluidLFResult:
    """Fluid load-flow outputs.

    ``full`` mode populates the named device collections. Edge ``i_*`` and
    ``j_*`` flow/heat values are positive into the edge at that terminal, so
    their sum is the edge loss. Other modes retain only ``arrays``.
    """

    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    pipes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    valves: Dict[str, SimpleNamespace] = field(default_factory=dict)
    controllers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    pumps: Dict[str, SimpleNamespace] = field(default_factory=dict)
    compressors: Dict[str, SimpleNamespace] = field(default_factory=dict)
    pressure_reducers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    heat_exchangers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    sources: Dict[str, SimpleNamespace] = field(default_factory=dict)
    storages: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)


class FluidPowerFlowCalc:
    """Solve pressure, mass flow, and optional heat transport in one network."""

    result_class = FluidLFResult

    def __init__(
        self,
        network: FluidNetwork,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_pressure: float = 1e-6,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        linear_solver: str = "scipy",
        result_mode: str = "full",
        verbose: bool = False,
    ):
        if not hasattr(network, "prepare") or not hasattr(network, "potential_power"):
            raise ValueError("FluidPowerFlowCalc requires a FluidNetwork-compatible input")
        self.network = network
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
        )
        self.tol = float(self.params.tol)
        self.max_iter = int(self.params.max_iter)
        self.min_pressure = max(float(min_pressure), 1e-12)
        self.linear_solver = str(linear_solver or "scipy").strip().lower()
        self.result_mode = normalize_result_mode(result_mode, f"{network.prefix} LF")
        self.verbose = bool(verbose)
        self.prepared = False
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.x = np.empty(0, dtype=np.float64)
        self.hydraulic_state_count = 0
        self.base_temperature = 0
        self.base_enthalpy = 0
        self.total_vars = 0
        self.total_eq = 0
        self.potential = np.empty(0, dtype=np.float64)
        self.pressure = np.empty(0, dtype=np.float64)
        self.edge_flow = np.empty(0, dtype=np.float64)
        self.source_flow = np.empty(0, dtype=np.float64)
        self.pressure_source_group_flow = np.empty(0, dtype=np.float64)
        self.heat_temperature_state = np.empty(0, dtype=np.float64)
        self.supply_temperature = np.empty(0, dtype=np.float64)
        self.return_temperature = np.empty(0, dtype=np.float64)
        self.enthalpy = np.empty(0, dtype=np.float64)
        self.temperature = np.empty(0, dtype=np.float64)
        self.lf_result = self.result_class()
        self.failure_reason = ""
        self.hydraulic_presolve: Dict[str, object] = {}
        self.excluded_device_keys: set[tuple[str, int]] = set()
        self.thermal_zero_flow_edge_pos = np.empty(0, dtype=np.int64)

    def _transport_flow_tolerance(self) -> float:
        return THERMAL_ZERO_FLOW_TOLERANCE if self.network.thermal else 1.0e-12

    def _active_transport_flow(self, flow: float) -> bool:
        """Return whether a hydraulic flow has a defined thermal direction."""
        return abs(float(flow)) > self._transport_flow_tolerance()

    def _zero_flow_thermal_edges(self, edge_flow: np.ndarray) -> np.ndarray:
        return np.flatnonzero(
            np.abs(np.asarray(edge_flow, dtype=np.float64))
            <= THERMAL_ZERO_FLOW_TOLERANCE
        ).astype(np.int64)

    def _configure_state_layout(self) -> None:
        net = self.network
        n_pressure_group = int(net.pressure_source_group_nodes.size) if net.thermal else 0
        self.base_pressure_source_group_flow = (
            net.free_node_pos.size + net.regulated_edge_pos.size
        )
        self.hydraulic_state_count = self.base_pressure_source_group_flow + n_pressure_group
        self.base_temperature = self.hydraulic_state_count
        self.base_enthalpy = self.base_temperature + (
            int(net.temperature_state_count) if net.thermal else 0
        )
        self.total_vars = self.base_enthalpy + (len(net.nodes) if net.steam else 0)
        self.total_eq = self.total_vars

    def _initial_hydraulic_state(self) -> np.ndarray:
        net = self.network
        potential = net.initial_potential()
        state = np.empty(self.hydraulic_state_count, dtype=np.float64)
        state[: net.free_node_pos.size] = potential[net.free_node_pos]
        if net.regulated_edge_pos.size:
            state[
                net.free_node_pos.size : self.base_pressure_source_group_flow
            ] = net.edge_flow_set[net.regulated_edge_pos]
        if net.thermal and net.pressure_source_group_nodes.size:
            state[
                self.base_pressure_source_group_flow : self.hydraulic_state_count
            ] = self._initial_pressure_source_group_flows()
        return state

    def _solve_hydraulic_initial_state(self) -> tuple[np.ndarray, dict]:
        state = self._initial_hydraulic_state()
        minimum = self._minimum_potential()
        converged = False
        norm = np.inf
        failure_reason = ""
        iterations = 0
        potential = self.network.initial_potential()
        edge_flow = np.zeros(len(self.network.edges), dtype=np.float64)
        for iteration in range(1, self.max_iter + 1):
            residual, jacobian, potential, edge_flow = self._hydraulic_residual_and_jacobian(state)
            iterations = iteration
            norm = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if norm < self.tol:
                converged = True
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    correction = np.asarray(spsolve(csc_matrix(jacobian), -residual), dtype=np.float64)
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                failure_reason = str(exc)
                break
            if not np.all(np.isfinite(correction)):
                failure_reason = "non-finite hydraulic Newton correction"
                break
            accepted = False
            step = 1.0
            for _ in range(24):
                candidate = state + step * correction
                if self.network.free_node_pos.size:
                    candidate[: self.network.free_node_pos.size] = np.maximum(
                        candidate[: self.network.free_node_pos.size], minimum
                    )
                candidate_residual = self._hydraulic_residual_and_jacobian(
                    candidate,
                    return_jacobian=False,
                )[0]
                candidate_norm = (
                    float(np.linalg.norm(candidate_residual, np.inf))
                    if candidate_residual.size
                    else 0.0
                )
                if np.isfinite(candidate_norm) and candidate_norm < norm:
                    state = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                failure_reason = "hydraulic line search could not reduce the residual"
                break

        residual, _jacobian, potential, edge_flow = self._hydraulic_residual_and_jacobian(
            state,
            return_jacobian=False,
        )
        norm = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
        converged = bool(norm < self.tol)
        source_flow = self._source_flows(state, edge_flow)[0]
        return state, {
            "converged": converged,
            "iterations": iterations,
            "residual": norm,
            "failure_reason": failure_reason,
            "potential": potential,
            "edge_flow": edge_flow,
            "source_flow": source_flow,
        }

    @staticmethod
    def _device_node_indices(item) -> tuple[int, ...]:
        values = []
        for field_name in (
            "node",
            "supply_node",
            "return_node",
            "i_node",
            "j_node",
            "primary_supply_node",
            "primary_return_node",
            "secondary_return_node",
            "secondary_supply_node",
        ):
            value = getattr(item, field_name, None)
            if value is not None:
                values.append(int(value))
        return tuple(dict.fromkeys(values))

    def _classify_hydraulic_islands(
        self,
        edge_flow: np.ndarray,
        source_flow: np.ndarray,
        *,
        flow_tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        net = self.network
        activity = np.zeros(int(net.island_count), dtype=np.float64)

        for edge_pos, flow in enumerate(edge_flow.tolist()):
            island = int(net.node_island[int(net.edge_i[edge_pos])])
            activity[island] = max(activity[island], abs(float(flow)))
        for source_pos, flow in enumerate(source_flow.tolist()):
            magnitude = abs(float(flow))
            if not bool(net.source_is_pressure_controlled[source_pos]):
                magnitude = max(
                    magnitude,
                    abs(float(net.source_flow_set[source_pos])),
                )
            for node_pos in (
                int(net.source_supply_node_pos[source_pos]),
                int(net.source_return_node_pos[source_pos]),
            ):
                island = int(net.node_island[node_pos])
                activity[island] = max(activity[island], magnitude)
        for load_pos, flow in enumerate(net.load_flow_set.tolist()):
            for node_pos in (
                int(net.load_supply_node_pos[load_pos]),
                int(net.load_return_node_pos[load_pos]),
            ):
                island = int(net.node_island[node_pos])
                activity[island] = max(activity[island], abs(float(flow)))
        for exchanger_pos in range(int(net.exchanger_i.size)):
            primary_magnitude = abs(float(net.exchanger_primary_flow[exchanger_pos]))
            secondary_magnitude = abs(float(net.exchanger_secondary_flow[exchanger_pos]))
            for node_pos in (
                int(net.exchanger_primary_supply[exchanger_pos]),
                int(net.exchanger_primary_return[exchanger_pos]),
            ):
                island = int(net.node_island[node_pos])
                activity[island] = max(activity[island], primary_magnitude)
            for node_pos in (
                int(net.exchanger_secondary_return[exchanger_pos]),
                int(net.exchanger_secondary_supply[exchanger_pos]),
            ):
                island = int(net.node_island[node_pos])
                activity[island] = max(activity[island], secondary_magnitude)

        active = np.flatnonzero(activity > flow_tolerance).astype(np.int64)
        dead = np.flatnonzero(activity <= flow_tolerance).astype(np.int64)
        return active, dead

    def _excluded_thermal_devices(self, active_islands: Sequence[int]) -> list[dict]:
        net = self.network
        active_islands = set(np.asarray(active_islands, dtype=np.int64).tolist())

        def touches_excluded_island(item) -> bool:
            return any(
                int(net.node_island[net.node_pos_by_idx[node_idx]]) not in active_islands
                for node_idx in self._device_node_indices(item)
            )

        devices = []
        collections = (
            (
                net.sources,
                lambda item: f"{net.prefix}Storage"
                if isinstance(item, FluidStorage)
                else f"{net.prefix}Source",
            ),
            (net.loads, lambda _item: f"{net.prefix}Load"),
            (net.pipes, lambda _item: f"{net.prefix}Pipe"),
            (net.valves, lambda _item: f"{net.prefix}Valve"),
            (
                net.controllers,
                lambda item: f"{net.prefix}{str(item.kind).replace('_', ' ').title().replace(' ', '')}",
            ),
            (net.heat_exchangers, lambda _item: "HeatExchanger"),
        )
        for collection, device_type_for in collections:
            for item in collection:
                if not touches_excluded_island(item):
                    continue
                device_type = str(device_type_for(item))
                device_idx = int(item.idx)
                self.excluded_device_keys.add((device_type, device_idx))
                devices.append(
                    {
                        "device_type": device_type,
                        "idx": device_idx,
                        "name": str(item.name),
                    }
                )
        return devices

    def _compact_thermal_network(self, active_islands: Sequence[int]) -> None:
        net = self.network
        active_islands = np.asarray(active_islands, dtype=np.int64)
        keep_nodes = np.isin(net.node_island, active_islands)
        keep_node_indices = {
            int(net.node_idx[pos]) for pos in np.flatnonzero(keep_nodes).tolist()
        }

        def device_is_active(item) -> bool:
            nodes = self._device_node_indices(item)
            return bool(nodes) and all(node in keep_node_indices for node in nodes)

        net.nodes = [node for node in net.nodes if int(node.idx) in keep_node_indices]
        net.sources = [item for item in net.sources if device_is_active(item)]
        net.storages = [item for item in net.storages if device_is_active(item)]
        net.loads = [item for item in net.loads if device_is_active(item)]
        net.pipes = [item for item in net.pipes if device_is_active(item)]
        net.valves = [item for item in net.valves if device_is_active(item)]
        net.controllers = [item for item in net.controllers if device_is_active(item)]
        net.heat_exchangers = [
            item for item in net.heat_exchangers if device_is_active(item)
        ]
        net.prepared = False
        net.prepare()

    def _prepare_empty_thermal_block(self, metadata: dict) -> "FluidPowerFlowCalc":
        self.network.nodes = []
        self.network.sources = []
        self.network.storages = []
        self.network.loads = []
        self.network.pipes = []
        self.network.valves = []
        self.network.controllers = []
        self.network.heat_exchangers = []
        self.network.edges = []
        self.network.island_count = 0
        self.network.node_island = np.empty(0, dtype=np.int32)
        self.network.node_idx = np.empty(0, dtype=np.int64)
        self.network.node_name = np.empty(0, dtype=object)
        self.network.node_explicit_return = np.empty(0, dtype=bool)
        self.network.edge_i = np.empty(0, dtype=np.int64)
        self.network.edge_j = np.empty(0, dtype=np.int64)
        self.network.edge_heat_loss = np.empty(0, dtype=np.float64)
        self.network.incidence = csr_matrix((0, 0), dtype=np.float64)
        self.network.source_name = np.empty(0, dtype=object)
        self.network.source_is_storage = np.empty(0, dtype=bool)
        self.network.source_is_pressure_controlled = np.empty(0, dtype=bool)
        self.network.storage_source_pos = np.empty(0, dtype=np.int64)
        self.network.load_name = np.empty(0, dtype=object)
        self.network.load_flow_set = np.empty(0, dtype=np.float64)
        self.network.load_heat_power = np.empty(0, dtype=np.float64)
        self.network.source_supply_node_pos = np.empty(0, dtype=np.int64)
        self.network.source_return_node_pos = np.empty(0, dtype=np.int64)
        self.network.load_supply_node_pos = np.empty(0, dtype=np.int64)
        self.network.load_return_node_pos = np.empty(0, dtype=np.int64)
        self.network.exchanger_i = np.empty(0, dtype=np.int64)
        self.network.exchanger_j = np.empty(0, dtype=np.int64)
        self.network.fixed_temperature_state_pos = np.empty(0, dtype=np.int64)
        self.network.fixed_temperature = np.empty(0, dtype=np.float64)
        self.network.free_node_pos = np.empty(0, dtype=np.int64)
        self.network.pressure_source_group_nodes = np.empty(0, dtype=np.int64)
        self.hydraulic_state_count = 0
        self.base_pressure_source_group_flow = 0
        self.base_temperature = 0
        self.base_enthalpy = 0
        self.total_vars = 0
        self.total_eq = 0
        self.x = np.empty(0, dtype=np.float64)
        self.potential = np.empty(0, dtype=np.float64)
        self.pressure = np.empty(0, dtype=np.float64)
        self.edge_flow = np.empty(0, dtype=np.float64)
        self.source_flow = np.empty(0, dtype=np.float64)
        self.pressure_source_group_flow = np.empty(0, dtype=np.float64)
        self.heat_temperature_state = np.empty(0, dtype=np.float64)
        self.supply_temperature = np.empty(0, dtype=np.float64)
        self.return_temperature = np.empty(0, dtype=np.float64)
        self.temperature = np.empty(0, dtype=np.float64)
        self.enthalpy = np.empty(0, dtype=np.float64)
        self.hydraulic_presolve = metadata
        self.converged = True
        self.iterations = 0
        self.normF = 0.0
        self.prepared = True
        return self

    def _initial_pressure_source_group_flows(self) -> np.ndarray:
        """Balance each explicit-return supply island at the initial state."""
        net = self.network
        group_count = int(net.pressure_source_group_nodes.size)
        initial = np.zeros(group_count, dtype=np.float64)
        groups_by_island: Dict[int, list[int]] = {}
        for group_pos, node_pos in enumerate(net.pressure_source_group_nodes.tolist()):
            island = int(net.node_island[int(node_pos)])
            groups_by_island.setdefault(island, []).append(group_pos)

        for island, group_positions in groups_by_island.items():
            island_nodes = np.flatnonzero(net.node_island == island)
            target_total = float(
                np.sum(net.demand[island_nodes])
                - np.sum(net.fixed_injection[island_nodes])
            )
            source_positions = np.concatenate(
                [net.pressure_source_groups[pos] for pos in group_positions]
            )
            allocated = allocate_limited_residual(
                net.source_flow_set[source_positions],
                target_total,
                lower=net.source_flow_min[source_positions],
                upper=net.source_flow_max[source_positions],
                alpha=net.source_alpha[source_positions],
            )
            offset = 0
            for group_pos in group_positions:
                group_size = int(net.pressure_source_groups[group_pos].size)
                initial[group_pos] = float(np.sum(allocated[offset : offset + group_size]))
                offset += group_size
        return initial

    def prepare(self) -> "FluidPowerFlowCalc":
        if self.prepared:
            return self
        net = self.network.prepare()
        self._configure_state_layout()
        hydraulic_initial = None
        if net.thermal:
            model_hydraulic_initial = self._initial_hydraulic_state()
            original_island_count = int(net.island_count)
            original_node_names = net.node_name.copy()
            original_node_island = net.node_island.copy()
            original_node_island_by_idx = {
                int(node_idx): int(island)
                for node_idx, island in zip(
                    net.node_idx.tolist(),
                    original_node_island.tolist(),
                )
            }
            # This is a physical zero-flow test, independent of the Newton residual tolerance.
            flow_tolerance = THERMAL_ZERO_FLOW_TOLERANCE
            original_dead_islands: set[int] = set()
            excluded_devices_by_key: Dict[tuple[str, int], dict] = {}
            self.excluded_device_keys = set()
            presolve_iterations = 0
            presolve_passes = 0

            while True:
                hydraulic_initial, presolve = self._solve_hydraulic_initial_state()
                presolve_passes += 1
                presolve_iterations += int(presolve["iterations"])
                if not presolve["converged"]:
                    raise RuntimeError(
                        "heat hydraulic initialization did not converge: "
                        f"residual={presolve['residual']:.6e}; {presolve['failure_reason']}"
                    )
                active_islands, dead_islands = self._classify_hydraulic_islands(
                    presolve["edge_flow"],
                    presolve["source_flow"],
                    flow_tolerance=flow_tolerance,
                )
                if dead_islands.size == 0:
                    break

                for node_pos in np.flatnonzero(
                    np.isin(net.node_island, dead_islands)
                ).tolist():
                    original_dead_islands.add(
                        original_node_island_by_idx[int(net.node_idx[node_pos])]
                    )
                for item in self._excluded_thermal_devices(active_islands):
                    key = (str(item["device_type"]), int(item["idx"]))
                    excluded_devices_by_key[key] = item
                if active_islands.size == 0:
                    break
                self._compact_thermal_network(active_islands)
                net = self.network
                self._configure_state_layout()

            dead_island_ids = sorted(original_dead_islands)
            active_island_ids = sorted(
                set(range(original_island_count)) - original_dead_islands
            )
            dead_node_names = original_node_names[
                np.isin(original_node_island, dead_island_ids)
            ].tolist()
            self.thermal_zero_flow_edge_pos = self._zero_flow_thermal_edges(
                presolve["edge_flow"]
            )
            metadata = {
                "converged": True,
                "iterations": presolve_iterations,
                "passes": presolve_passes,
                "residual": float(presolve["residual"]),
                "flow_tolerance": float(flow_tolerance),
                "original_island_count": original_island_count,
                "active_island_ids": active_island_ids,
                "dead_island_ids": dead_island_ids,
                "dead_node_names": [str(name) for name in dead_node_names],
                "excluded_devices": list(excluded_devices_by_key.values()),
                "zero_flow_thermal_edge_positions": (
                    self.thermal_zero_flow_edge_pos.tolist()
                ),
                "zero_flow_thermal_edge_names": [
                    str(net.edge_name[pos])
                    for pos in self.thermal_zero_flow_edge_pos.tolist()
                ],
                "initial_state_policy": (
                    "presolved_after_dead_island_compaction"
                    if dead_island_ids
                    else "model"
                ),
                "presolved_state_used": bool(dead_island_ids),
            }
            if dead_island_ids:
                warning = (
                    f"excluded {len(dead_island_ids)} zero-flow heat hydraulic island(s) "
                    f"before global load flow: islands={dead_island_ids}"
                )
                if not active_island_ids:
                    if warning not in net.warnings:
                        net.warnings.append(warning)
                    return self._prepare_empty_thermal_block(metadata)
                if warning not in net.warnings:
                    net.warnings.append(warning)
            else:
                # The pre-solve is still required to classify every island.  With no
                # topology reduction, retain the model state because it can follow a
                # better path in the complete electric/fluid Newton problem.
                hydraulic_initial = model_hydraulic_initial
            self.hydraulic_presolve = metadata

        potential = net.initial_potential()
        n_pressure_group = int(net.pressure_source_group_nodes.size) if net.thermal else 0
        x = np.empty(self.total_vars, dtype=np.float64)
        if hydraulic_initial is None:
            hydraulic_initial = self._initial_hydraulic_state()
        x[: self.hydraulic_state_count] = hydraulic_initial
        if net.thermal:
            x[self.base_temperature : self.base_enthalpy] = net.initial_temperature_state
        elif net.steam:
            x[self.base_enthalpy :] = net.node_enthalpy
        self.x = x
        self.potential = potential
        self.pressure = np.maximum(potential, self._minimum_potential()) ** (1.0 / net.potential_power)
        self.edge_flow = np.zeros(len(net.edges), dtype=np.float64)
        self.source_flow = net.source_flow_set.copy()
        self.pressure_source_group_flow = (
            x[self.base_pressure_source_group_flow : self.hydraulic_state_count].copy()
            if n_pressure_group
            else np.empty(0, dtype=np.float64)
        )
        self.heat_temperature_state = net.initial_temperature_state.copy()
        self.supply_temperature = net.node_supply_temperature.copy()
        self.return_temperature = net.node_return_temperature.copy()
        if net.thermal and net.temperature_state_count:
            self._sync_heat_temperature_views()
        self.enthalpy = net.node_enthalpy.copy()
        self.temperature = self._steam_temperature(self.enthalpy) if net.steam else np.empty(0)
        self.prepared = True
        return self

    def _sync_heat_temperature_views(self) -> None:
        net = self.network
        self.supply_temperature = self.heat_temperature_state[
            net.supply_temperature_state_by_node
        ]
        self.return_temperature = self.heat_temperature_state[
            net.return_temperature_state_by_node
        ]
        self.temperature = self.supply_temperature.copy()

    def _minimum_potential(self) -> float:
        return self.min_pressure ** int(self.network.potential_power)

    def _state_potential_and_flow(
        self,
        x: np.ndarray,
        *,
        return_derivative: bool = True,
    ):
        net = self.network
        potential = net.initial_potential()
        potential[net.free_node_pos] = x[: net.free_node_pos.size]
        edge_flow = net.edge_flow_set.copy()
        if net.regulated_edge_pos.size:
            edge_flow[net.regulated_edge_pos] = x[
                net.free_node_pos.size : self.base_pressure_source_group_flow
            ]
        passive = net.passive_edge_pos
        derivative = (
            np.zeros(len(net.edges), dtype=np.float64)
            if return_derivative
            else None
        )
        if passive.size:
            delta = potential[net.edge_i[passive]] - potential[net.edge_j[passive]]
            edge_flow[passive] = net.edge_conductance[passive] * np.sign(delta) * np.sqrt(np.abs(delta))
            if return_derivative:
                derivative[passive] = 0.5 * net.edge_conductance[passive] / np.sqrt(
                    np.maximum(np.abs(delta), 1e-10)
                )
        return potential, edge_flow, derivative

    def _hydraulic_residual_and_jacobian(
        self,
        x: np.ndarray,
        *,
        return_jacobian: bool = True,
    ):
        net = self.network
        potential, edge_flow, derivative = self._state_potential_and_flow(
            x,
            return_derivative=return_jacobian,
        )
        node_balance = net.fixed_injection - net.demand + net.incidence @ edge_flow
        source_flow, source_group_shares = self._explicit_pressure_source_flows(
            x,
            return_shares=return_jacobian,
        )
        if net.thermal and net.pressure_source_group_nodes.size:
            for group_pos, source_positions in enumerate(net.pressure_source_groups):
                flows = source_flow[source_positions]
                np.add.at(
                    node_balance,
                    net.source_return_node_pos[source_positions],
                    -flows,
                )
                np.add.at(
                    node_balance,
                    net.source_supply_node_pos[source_positions],
                    flows,
                )
        n_balance = int(net.balance_node_pos.size)
        n_reg = int(net.regulated_edge_pos.size)
        residual = np.empty(n_balance + n_reg, dtype=np.float64)
        residual[:n_balance] = node_balance[net.balance_node_pos]

        for local_pos, edge_pos in enumerate(net.regulated_edge_pos.tolist()):
            row = n_balance + local_pos
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            control = str(net.edge_control_type[edge_pos])
            if control == RATIO_CONTROL:
                ratio2 = float(net.edge_ratio[edge_pos] ** net.potential_power)
                residual[row] = potential[j] - ratio2 * potential[i]
            elif control == GAIN_CONTROL:
                residual[row] = potential[j] - potential[i] - net.edge_pressure_gain[edge_pos]
            else:
                residual[row] = edge_flow[edge_pos] - net.edge_flow_set[edge_pos]

        if not return_jacobian:
            return residual, None, potential, edge_flow

        rows = []
        cols = []
        data = []
        for edge_pos in net.passive_edge_pos.tolist():
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            i_row = int(net.balance_row_by_node[i])
            j_row = int(net.balance_row_by_node[j])
            di = int(net.free_state_by_node[i])
            dj = int(net.free_state_by_node[j])
            value = float(derivative[edge_pos])
            if i_row >= 0:
                if di >= 0:
                    rows.append(i_row)
                    cols.append(di)
                    data.append(-value)
                if dj >= 0:
                    rows.append(i_row)
                    cols.append(dj)
                    data.append(value)
            if j_row >= 0:
                if dj >= 0:
                    rows.append(j_row)
                    cols.append(dj)
                    data.append(-value)
                if di >= 0:
                    rows.append(j_row)
                    cols.append(di)
                    data.append(value)

        for edge_pos in net.regulated_edge_pos.tolist():
            state_col = int(net.regulated_state_by_edge[edge_pos])
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            i_row = int(net.balance_row_by_node[i])
            j_row = int(net.balance_row_by_node[j])
            if i_row >= 0:
                rows.append(i_row)
                cols.append(state_col)
                data.append(-1.0)
            if j_row >= 0:
                rows.append(j_row)
                cols.append(state_col)
                data.append(1.0)

        for group_pos, source_positions in enumerate(net.pressure_source_groups):
            state_col = self.base_pressure_source_group_flow + group_pos
            shares = source_group_shares[group_pos]
            for local_pos, source_pos in enumerate(source_positions.tolist()):
                share = float(shares[local_pos])
                supply_row = int(
                    net.balance_row_by_node[net.source_supply_node_pos[source_pos]]
                )
                return_row = int(
                    net.balance_row_by_node[net.source_return_node_pos[source_pos]]
                )
                if supply_row >= 0:
                    rows.append(supply_row)
                    cols.append(state_col)
                    data.append(share)
                if return_row >= 0:
                    rows.append(return_row)
                    cols.append(state_col)
                    data.append(-share)

        for local_pos, edge_pos in enumerate(net.regulated_edge_pos.tolist()):
            row = n_balance + local_pos
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            i_col = int(net.free_state_by_node[i])
            j_col = int(net.free_state_by_node[j])
            control = str(net.edge_control_type[edge_pos])
            if control == RATIO_CONTROL:
                ratio2 = float(net.edge_ratio[edge_pos] ** net.potential_power)
                if i_col >= 0:
                    rows.append(row)
                    cols.append(i_col)
                    data.append(-ratio2)
                if j_col >= 0:
                    rows.append(row)
                    cols.append(j_col)
                    data.append(1.0)
            elif control == GAIN_CONTROL:
                if i_col >= 0:
                    rows.append(row)
                    cols.append(i_col)
                    data.append(-1.0)
                if j_col >= 0:
                    rows.append(row)
                    cols.append(j_col)
                    data.append(1.0)
            else:
                rows.append(row)
                cols.append(int(net.regulated_state_by_edge[edge_pos]))
                data.append(1.0)

        jacobian = coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(residual.size, self.hydraulic_state_count),
        ).tocsc()
        return residual, jacobian, potential, edge_flow

    def _explicit_pressure_source_flows(
        self,
        x: np.ndarray,
        *,
        return_shares: bool = True,
    ):
        net = self.network
        source_flow = net.source_flow_set.copy()
        shares_by_group = [] if return_shares else None
        for group_pos, source_positions in enumerate(net.pressure_source_groups):
            target = float(x[self.base_pressure_source_group_flow + group_pos])
            allocated = allocate_limited_residual(
                net.source_flow_set[source_positions],
                target,
                lower=net.source_flow_min[source_positions],
                upper=net.source_flow_max[source_positions],
                alpha=net.source_alpha[source_positions],
            )
            source_flow[source_positions] = allocated
            if return_shares:
                step = max(1e-7, abs(target) * 1e-7)
                allocated_plus = allocate_limited_residual(
                    net.source_flow_set[source_positions],
                    target + step,
                    lower=net.source_flow_min[source_positions],
                    upper=net.source_flow_max[source_positions],
                    alpha=net.source_alpha[source_positions],
                )
                shares_by_group.append((allocated_plus - allocated) / step)
        return source_flow, shares_by_group

    def _edge_flow_jacobian(self, potential: np.ndarray) -> csr_matrix:
        """Return d(edge_flow)/d(hydraulic_state) for the prepared topology."""
        net = self.network
        rows = []
        cols = []
        data = []
        for edge_pos in net.passive_edge_pos.tolist():
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            delta = float(potential[i] - potential[j])
            derivative = 0.5 * float(net.edge_conductance[edge_pos]) / np.sqrt(
                max(abs(delta), 1e-10)
            )
            i_col = int(net.free_state_by_node[i])
            j_col = int(net.free_state_by_node[j])
            if i_col >= 0:
                rows.append(edge_pos)
                cols.append(i_col)
                data.append(derivative)
            if j_col >= 0:
                rows.append(edge_pos)
                cols.append(j_col)
                data.append(-derivative)
        for edge_pos in net.regulated_edge_pos.tolist():
            rows.append(edge_pos)
            cols.append(int(net.regulated_state_by_edge[edge_pos]))
            data.append(1.0)
        return coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(len(net.edges), self.hydraulic_state_count),
        ).tocsr()

    @staticmethod
    def _allocation_shares(base, target, lower, upper, alpha) -> tuple[np.ndarray, np.ndarray]:
        allocated = allocate_limited_residual(
            base,
            target,
            lower=lower,
            upper=upper,
            alpha=alpha,
        )
        step = max(1e-7, abs(float(target)) * 1e-7)
        allocated_plus = allocate_limited_residual(
            base,
            float(target) + step,
            lower=lower,
            upper=upper,
            alpha=alpha,
        )
        return allocated, (allocated_plus - allocated) / step

    def _source_flows(
        self,
        x: np.ndarray,
        edge_flow: np.ndarray,
        edge_jacobian: Optional[csr_matrix] = None,
    ) -> tuple[np.ndarray, Optional[csr_matrix]]:
        """Evaluate balancing-source flows, optionally with hydraulic derivatives."""
        net = self.network
        return_jacobian = edge_jacobian is not None
        source_flow = net.source_flow_set.copy()
        rows = [] if return_jacobian else None
        cols = [] if return_jacobian else None
        data = [] if return_jacobian else None
        handled = np.zeros(len(net.sources), dtype=bool)

        for group_pos, source_positions in enumerate(net.pressure_source_groups):
            target = float(x[self.base_pressure_source_group_flow + group_pos])
            if return_jacobian:
                allocated, shares = self._allocation_shares(
                    net.source_flow_set[source_positions],
                    target,
                    net.source_flow_min[source_positions],
                    net.source_flow_max[source_positions],
                    net.source_alpha[source_positions],
                )
            else:
                allocated = allocate_limited_residual(
                    net.source_flow_set[source_positions],
                    target,
                    lower=net.source_flow_min[source_positions],
                    upper=net.source_flow_max[source_positions],
                    alpha=net.source_alpha[source_positions],
                )
            source_flow[source_positions] = allocated
            handled[source_positions] = True
            if return_jacobian:
                state_col = self.base_pressure_source_group_flow + group_pos
                for local_pos, source_pos in enumerate(source_positions.tolist()):
                    rows.append(source_pos)
                    cols.append(state_col)
                    data.append(float(shares[local_pos]))

        pressure_sources = np.flatnonzero(
            net.source_is_pressure_controlled
        ).astype(np.int64)
        pressure_sources = pressure_sources[~handled[pressure_sources]]
        node_edge_balance = np.asarray(net.incidence @ edge_flow, dtype=np.float64)
        node_edge_jacobian = (
            (net.incidence @ edge_jacobian).tocsr()
            if return_jacobian
            else None
        )
        if pressure_sources.size:
            source_nodes = net.source_node_pos[pressure_sources]
            for node_pos in np.unique(source_nodes).tolist():
                positions = pressure_sources[source_nodes == int(node_pos)]
                target = float(
                    net.demand[int(node_pos)]
                    - net.fixed_injection[int(node_pos)]
                    - node_edge_balance[int(node_pos)]
                )
                if return_jacobian:
                    allocated, shares = self._allocation_shares(
                        net.source_flow_set[positions],
                        target,
                        net.source_flow_min[positions],
                        net.source_flow_max[positions],
                        net.source_alpha[positions],
                    )
                else:
                    allocated = allocate_limited_residual(
                        net.source_flow_set[positions],
                        target,
                        lower=net.source_flow_min[positions],
                        upper=net.source_flow_max[positions],
                        alpha=net.source_alpha[positions],
                    )
                source_flow[positions] = allocated
                if return_jacobian:
                    target_derivative = -node_edge_jacobian.getrow(int(node_pos))
                    target_coo = target_derivative.tocoo()
                    for local_pos, source_pos in enumerate(positions.tolist()):
                        share = float(shares[local_pos])
                        if share == 0.0 or target_coo.nnz == 0:
                            continue
                        rows.extend([source_pos] * target_coo.nnz)
                        cols.extend(target_coo.col.tolist())
                        data.extend((share * target_coo.data).tolist())

        if not return_jacobian:
            return source_flow, None

        jacobian = coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(len(net.sources), self.hydraulic_state_count),
        ).tocsr()
        return source_flow, jacobian

    def _source_flows_and_jacobian(
        self,
        x: np.ndarray,
        edge_flow: np.ndarray,
        edge_jacobian: csr_matrix,
    ) -> tuple[np.ndarray, csr_matrix]:
        """Evaluate balancing-source flows and their hydraulic derivatives."""
        source_flow, jacobian = self._source_flows(x, edge_flow, edge_jacobian)
        return source_flow, jacobian

    def source_flows_and_jacobian(self, x: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
        """Evaluate source-flow outputs and derivatives for a local or global solver."""
        hydraulic_x = np.asarray(x[: self.hydraulic_state_count], dtype=np.float64)
        potential, edge_flow, _derivative = self._state_potential_and_flow(hydraulic_x)
        edge_jacobian = self._edge_flow_jacobian(potential)
        return self._source_flows_and_jacobian(
            hydraulic_x,
            edge_flow,
            edge_jacobian,
        )

    @staticmethod
    def _append_scaled_sparse_row(rows, cols, data, row: int, sparse_row, scale: float) -> None:
        if scale == 0.0:
            return
        coo = sparse_row.tocoo()
        if coo.nnz == 0:
            return
        rows.extend([int(row)] * coo.nnz)
        cols.extend(coo.col.tolist())
        data.extend((float(scale) * coo.data).tolist())

    def _transport_state_components(self, edge_flow: np.ndarray, count: int) -> np.ndarray:
        """Return transport-equation components for temperature or enthalpy states."""
        net = self.network
        parent = np.arange(count, dtype=np.int64)

        def find(pos: int) -> int:
            while parent[pos] != pos:
                parent[pos] = parent[parent[pos]]
                pos = int(parent[pos])
            return pos

        def union(left: int, right: int) -> None:
            left_root = find(int(left))
            right_root = find(int(right))
            if left_root != right_root:
                parent[right_root] = left_root

        for edge_pos, flow in enumerate(edge_flow.tolist()):
            if not self._active_transport_flow(flow):
                continue
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            if net.thermal:
                union(
                    int(net.supply_temperature_state_by_node[i]),
                    int(net.supply_temperature_state_by_node[j]),
                )
                if not net.node_explicit_return[i] and not net.node_explicit_return[j]:
                    union(
                        int(net.return_temperature_state_by_node[i]),
                        int(net.return_temperature_state_by_node[j]),
                    )
            else:
                union(i, j)

        if net.thermal:
            for load_pos in range(len(net.loads)):
                if (
                    float(net.load_flow_set[load_pos])
                    <= THERMAL_ZERO_FLOW_TOLERANCE
                ):
                    continue
                union(
                    int(
                        net.supply_temperature_state_by_node[
                            net.load_supply_node_pos[load_pos]
                        ]
                    ),
                    int(
                        net.return_temperature_state_by_node[
                            net.load_return_node_pos[load_pos]
                        ]
                    ),
                )
            for exchanger_pos in range(int(net.exchanger_i.size)):
                if (
                    float(net.exchanger_primary_flow[exchanger_pos])
                    <= THERMAL_ZERO_FLOW_TOLERANCE
                    or float(net.exchanger_secondary_flow[exchanger_pos])
                    <= THERMAL_ZERO_FLOW_TOLERANCE
                ):
                    continue
                states = (
                    int(
                        net.supply_temperature_state_by_node[
                            net.exchanger_primary_supply[exchanger_pos]
                        ]
                    ),
                    int(
                        net.return_temperature_state_by_node[
                            net.exchanger_primary_return[exchanger_pos]
                        ]
                    ),
                    int(
                        net.return_temperature_state_by_node[
                            net.exchanger_secondary_return[exchanger_pos]
                        ]
                    ),
                    int(
                        net.supply_temperature_state_by_node[
                            net.exchanger_secondary_supply[exchanger_pos]
                        ]
                    ),
                )
                if str(net.exchanger_control_type[exchanger_pos]) == "EFFECTIVENESS":
                    for state in states[1:]:
                        union(states[0], state)
                else:
                    union(states[0], states[1])
                    union(states[2], states[3])

        return np.asarray([find(pos) for pos in range(count)], dtype=np.int64)

    def _thermal_source_injection(self, source_pos: int, flow: float):
        """Return the thermal inlet represented by a signed source flow."""
        net = self.network
        if flow > THERMAL_ZERO_FLOW_TOLERANCE:
            node_pos = int(net.source_supply_node_pos[source_pos])
            state = int(net.supply_temperature_state_by_node[node_pos])
            return (
                state,
                float(flow),
                float(net.source_supply_temperature_set[source_pos]),
                1.0,
            )
        if (
            flow < -THERMAL_ZERO_FLOW_TOLERANCE
            and bool(net.source_is_storage[source_pos])
            and bool(net.source_explicit_return[source_pos])
        ):
            node_pos = int(net.source_return_node_pos[source_pos])
            state = int(net.return_temperature_state_by_node[node_pos])
            return (
                state,
                float(-flow),
                float(net.source_return_temperature_set[source_pos]),
                -1.0,
            )
        return None

    def _unanchored_pressure_source_groups(
        self,
        edge_flow: np.ndarray,
        source_flow: np.ndarray,
        incoming_mass: np.ndarray,
        source_state: np.ndarray,
    ) -> list[np.ndarray]:
        """Find zero-flow pressure sources in transport components without a boundary."""
        labels = self._transport_state_components(edge_flow, incoming_mass.size)
        anchored = set(
            labels[
                np.flatnonzero(incoming_mass <= THERMAL_ZERO_FLOW_TOLERANCE)
            ].tolist()
        )
        injection_state = source_state.copy()
        reverse_storage = (
            (source_flow < -THERMAL_ZERO_FLOW_TOLERANCE)
            & self.network.source_is_storage
            & self.network.source_explicit_return
        )
        if np.any(reverse_storage):
            injection_state[reverse_storage] = (
                self.network.return_temperature_state_by_node[
                    self.network.source_return_node_pos[reverse_storage]
                ]
            )
        injecting_sources = np.flatnonzero(
            (source_flow > THERMAL_ZERO_FLOW_TOLERANCE) | reverse_storage
        )
        anchored.update(labels[injection_state[injecting_sources]].tolist())
        candidates = np.flatnonzero(
            self.network.source_is_pressure_controlled
            & (np.abs(source_flow) <= THERMAL_ZERO_FLOW_TOLERANCE)
        )
        groups = []
        for label in np.unique(labels[source_state[candidates]]).tolist():
            if int(label) in anchored:
                continue
            positions = candidates[labels[source_state[candidates]] == int(label)]
            if positions.size:
                groups.append(positions)
        return groups

    def _heat_transport_residual_and_jacobian(
        self,
        x: np.ndarray,
        edge_flow: np.ndarray,
        edge_jacobian: Optional[csr_matrix],
        source_flow: np.ndarray,
        source_jacobian: Optional[csr_matrix],
    ) -> tuple[np.ndarray, Optional[csr_matrix], Optional[csc_matrix]]:
        net = self.network
        return_jacobian = edge_jacobian is not None and source_jacobian is not None
        temperature = np.asarray(x[self.base_temperature : self.base_enthalpy], dtype=np.float64)
        count = int(net.temperature_state_count)
        cp = max(float(net.medium.heat_capacity), 1e-12)
        ambient = float(net.medium.ambient_temperature)
        residual = np.zeros(count, dtype=np.float64)
        incoming_mass = np.zeros(count, dtype=np.float64)
        temp_rows = [] if return_jacobian else None
        temp_cols = [] if return_jacobian else None
        temp_data = [] if return_jacobian else None
        cross_rows = [] if return_jacobian else None
        cross_cols = [] if return_jacobian else None
        cross_data = [] if return_jacobian else None

        def add_temp(row: int, col: int, value: float) -> None:
            if return_jacobian:
                temp_rows.append(int(row))
                temp_cols.append(int(col))
                temp_data.append(float(value))

        def add_transport(edge_pos: int, upstream: int, downstream: int, flow: float) -> None:
            mass = abs(float(flow))
            if not self._active_transport_flow(mass):
                return
            loss = float(net.edge_heat_loss[edge_pos])
            attenuation = float(np.exp(-loss / max(mass, 1e-9)))
            upstream_value = float(temperature[upstream])
            downstream_value = float(temperature[downstream])
            transport_gap = downstream_value - ambient - attenuation * (upstream_value - ambient)
            incoming_mass[downstream] += mass
            residual[downstream] += mass * transport_gap
            add_temp(downstream, downstream, mass)
            add_temp(downstream, upstream, -mass * attenuation)
            if return_jacobian:
                derivative_mass = transport_gap - attenuation * loss / max(mass, 1e-9) * (
                    upstream_value - ambient
                )
                derivative_flow = np.sign(float(flow)) * derivative_mass
                self._append_scaled_sparse_row(
                    cross_rows,
                    cross_cols,
                    cross_data,
                    downstream,
                    edge_jacobian.getrow(edge_pos),
                    derivative_flow,
                )

        for edge_pos, flow in enumerate(edge_flow.tolist()):
            if not self._active_transport_flow(flow):
                continue
            if flow > 0.0:
                upstream_node = int(net.edge_i[edge_pos])
                downstream_node = int(net.edge_j[edge_pos])
            else:
                upstream_node = int(net.edge_j[edge_pos])
                downstream_node = int(net.edge_i[edge_pos])
            add_transport(
                edge_pos,
                int(net.supply_temperature_state_by_node[upstream_node]),
                int(net.supply_temperature_state_by_node[downstream_node]),
                flow,
            )
            if not net.node_explicit_return[upstream_node]:
                add_transport(
                    edge_pos,
                    int(net.return_temperature_state_by_node[downstream_node]),
                    int(net.return_temperature_state_by_node[upstream_node]),
                    flow,
                )

        source_state = net.supply_temperature_state_by_node[net.source_supply_node_pos]
        for source_pos, flow in enumerate(source_flow.tolist()):
            injection = self._thermal_source_injection(source_pos, flow)
            if injection is None:
                continue
            state, mass, temperature_set, flow_derivative_scale = injection
            gap = float(temperature[state] - temperature_set)
            incoming_mass[state] += mass
            residual[state] += mass * gap
            add_temp(state, state, mass)
            if return_jacobian:
                self._append_scaled_sparse_row(
                    cross_rows,
                    cross_cols,
                    cross_data,
                    state,
                    source_jacobian.getrow(source_pos),
                    flow_derivative_scale * gap,
                )

        for load_pos in range(len(net.loads)):
            mass = float(net.load_flow_set[load_pos])
            if mass <= THERMAL_ZERO_FLOW_TOLERANCE:
                continue
            supply_node = int(net.load_supply_node_pos[load_pos])
            return_node = int(net.load_return_node_pos[load_pos])
            supply_state = int(net.supply_temperature_state_by_node[supply_node])
            return_state = int(net.return_temperature_state_by_node[return_node])
            incoming_mass[return_state] += mass
            residual[return_state] += (
                mass * (temperature[return_state] - temperature[supply_state])
                + float(net.load_heat_power[load_pos]) / cp
            )
            add_temp(return_state, return_state, mass)
            add_temp(return_state, supply_state, -mass)

        for exchanger_pos in range(int(net.exchanger_i.size)):
            primary_supply = int(net.exchanger_primary_supply[exchanger_pos])
            primary_return = int(net.exchanger_primary_return[exchanger_pos])
            secondary_return = int(net.exchanger_secondary_return[exchanger_pos])
            secondary_supply = int(net.exchanger_secondary_supply[exchanger_pos])
            ps = int(net.supply_temperature_state_by_node[primary_supply])
            pr = int(net.return_temperature_state_by_node[primary_return])
            sr = int(net.return_temperature_state_by_node[secondary_return])
            ss = int(net.supply_temperature_state_by_node[secondary_supply])
            primary_mass = float(net.exchanger_primary_flow[exchanger_pos])
            secondary_mass = float(net.exchanger_secondary_flow[exchanger_pos])
            if (
                primary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
                or secondary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
            ):
                continue
            incoming_mass[ss] += secondary_mass
            incoming_mass[pr] += primary_mass
            if str(net.exchanger_control_type[exchanger_pos]) == "EFFECTIVENESS":
                transfer = float(net.exchanger_effectiveness[exchanger_pos]) * min(
                    primary_mass, secondary_mass
                )
                secondary_transfer = (1.0 - float(net.exchanger_heat_loss[exchanger_pos])) * transfer
                residual[ss] += (
                    secondary_mass * temperature[ss]
                    - secondary_transfer * temperature[ps]
                    - (secondary_mass - secondary_transfer) * temperature[sr]
                )
                add_temp(ss, ss, secondary_mass)
                add_temp(ss, ps, -secondary_transfer)
                add_temp(ss, sr, -(secondary_mass - secondary_transfer))
                residual[pr] += (
                    primary_mass * temperature[pr]
                    - (primary_mass - transfer) * temperature[ps]
                    - transfer * temperature[sr]
                )
                add_temp(pr, pr, primary_mass)
                add_temp(pr, ps, -(primary_mass - transfer))
                add_temp(pr, sr, -transfer)
            else:
                primary_heat = float(net.exchanger_heat_set[exchanger_pos])
                secondary_heat = (1.0 - float(net.exchanger_heat_loss[exchanger_pos])) * primary_heat
                residual[ss] += secondary_mass * (temperature[ss] - temperature[sr]) - secondary_heat / cp
                add_temp(ss, ss, secondary_mass)
                add_temp(ss, sr, -secondary_mass)
                residual[pr] += primary_mass * (temperature[pr] - temperature[ps]) + primary_heat / cp
                add_temp(pr, pr, primary_mass)
                add_temp(pr, ps, -primary_mass)

        anchor_mass = 1e-6 * max(
            1.0,
            float(np.max(np.abs(net.demand), initial=0.0)),
            float(np.max(np.abs(net.fixed_injection), initial=0.0)),
        )
        for source_positions in self._unanchored_pressure_source_groups(
            edge_flow,
            source_flow,
            incoming_mass,
            source_state,
        ):
            weights = np.maximum(net.source_alpha[source_positions], 0.0)
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones(source_positions.size, dtype=np.float64)
            weights = weights / float(np.sum(weights))
            for local_pos, source_pos in enumerate(source_positions.tolist()):
                state = int(source_state[source_pos])
                mass = anchor_mass * float(weights[local_pos])
                gap = float(temperature[state] - net.source_supply_temperature[source_pos])
                incoming_mass[state] += mass
                residual[state] += mass * gap
                add_temp(state, state, mass)

        for state_pos in range(count):
            if incoming_mass[state_pos] > THERMAL_ZERO_FLOW_TOLERANCE:
                continue
            residual[state_pos] = temperature[state_pos] - net.initial_temperature_state[state_pos]
            add_temp(state_pos, state_pos, 1.0)

        fixed_states = net.fixed_temperature_state_pos
        if fixed_states.size:
            residual[fixed_states] = temperature[fixed_states] - net.fixed_temperature

        if not return_jacobian:
            return residual, None, None

        fixed_mask = np.zeros(count, dtype=bool)
        fixed_mask[fixed_states] = True
        temp_rows_array = np.asarray(temp_rows, dtype=np.int64)
        temp_cols_array = np.asarray(temp_cols, dtype=np.int64)
        temp_data_array = np.asarray(temp_data, dtype=np.float64)
        if temp_rows_array.size:
            keep = ~fixed_mask[temp_rows_array]
            temp_rows_array = temp_rows_array[keep]
            temp_cols_array = temp_cols_array[keep]
            temp_data_array = temp_data_array[keep]
        if fixed_states.size:
            temp_rows_array = np.concatenate((temp_rows_array, fixed_states))
            temp_cols_array = np.concatenate((temp_cols_array, fixed_states))
            temp_data_array = np.concatenate(
                (temp_data_array, np.ones(fixed_states.size, dtype=np.float64))
            )

        cross_rows_array = np.asarray(cross_rows, dtype=np.int64)
        cross_cols_array = np.asarray(cross_cols, dtype=np.int64)
        cross_data_array = np.asarray(cross_data, dtype=np.float64)
        if cross_rows_array.size:
            keep = ~fixed_mask[cross_rows_array]
            cross_rows_array = cross_rows_array[keep]
            cross_cols_array = cross_cols_array[keep]
            cross_data_array = cross_data_array[keep]

        cross = coo_matrix(
            (
                cross_data_array,
                (cross_rows_array, cross_cols_array),
            ),
            shape=(count, self.hydraulic_state_count),
        ).tocsr()
        local = coo_matrix(
            (
                temp_data_array,
                (temp_rows_array, temp_cols_array),
            ),
            shape=(count, count),
        ).tocsc()
        return residual, cross, local

    def _steam_transport_residual_and_jacobian(
        self,
        x: np.ndarray,
        edge_flow: np.ndarray,
        edge_jacobian: Optional[csr_matrix],
        source_flow: np.ndarray,
        source_jacobian: Optional[csr_matrix],
    ) -> tuple[np.ndarray, Optional[csr_matrix], Optional[csc_matrix]]:
        net = self.network
        return_jacobian = edge_jacobian is not None and source_jacobian is not None
        enthalpy = np.asarray(x[self.base_enthalpy :], dtype=np.float64)
        count = len(net.nodes)
        ambient = float(net.medium.ambient_enthalpy)
        residual = np.zeros(count, dtype=np.float64)
        incoming_mass = np.zeros(count, dtype=np.float64)
        local_rows = [] if return_jacobian else None
        local_cols = [] if return_jacobian else None
        local_data = [] if return_jacobian else None
        cross_rows = [] if return_jacobian else None
        cross_cols = [] if return_jacobian else None
        cross_data = [] if return_jacobian else None

        def add_local(row: int, col: int, value: float) -> None:
            if return_jacobian:
                local_rows.append(int(row))
                local_cols.append(int(col))
                local_data.append(float(value))

        for edge_pos, flow in enumerate(edge_flow.tolist()):
            if abs(flow) <= 1e-12:
                continue
            if flow > 0.0:
                upstream = int(net.edge_i[edge_pos])
                downstream = int(net.edge_j[edge_pos])
            else:
                upstream = int(net.edge_j[edge_pos])
                downstream = int(net.edge_i[edge_pos])
            mass = abs(float(flow))
            loss = float(net.edge_heat_loss[edge_pos])
            attenuation = float(np.exp(-loss / max(mass, 1e-9)))
            gap = float(enthalpy[downstream] - ambient - attenuation * (enthalpy[upstream] - ambient))
            incoming_mass[downstream] += mass
            residual[downstream] += mass * gap
            add_local(downstream, downstream, mass)
            add_local(downstream, upstream, -mass * attenuation)
            if return_jacobian:
                derivative_mass = gap - attenuation * loss / max(mass, 1e-9) * (
                    enthalpy[upstream] - ambient
                )
                self._append_scaled_sparse_row(
                    cross_rows,
                    cross_cols,
                    cross_data,
                    downstream,
                    edge_jacobian.getrow(edge_pos),
                    np.sign(float(flow)) * derivative_mass,
                )

        source_state = net.source_node_pos
        for source_pos, flow in enumerate(source_flow.tolist()):
            if flow <= 1e-12:
                continue
            node = int(net.source_node_pos[source_pos])
            gap = float(enthalpy[node] - net.source_enthalpy_set[source_pos])
            incoming_mass[node] += flow
            residual[node] += flow * gap
            add_local(node, node, flow)
            if return_jacobian:
                self._append_scaled_sparse_row(
                    cross_rows,
                    cross_cols,
                    cross_data,
                    node,
                    source_jacobian.getrow(source_pos),
                    gap,
                )

        anchor_mass = 1e-6 * max(
            1.0,
            float(np.max(np.abs(net.demand), initial=0.0)),
            float(np.max(np.abs(net.fixed_injection), initial=0.0)),
        )
        for source_positions in self._unanchored_pressure_source_groups(
            edge_flow,
            source_flow,
            incoming_mass,
            source_state,
        ):
            weights = np.maximum(net.source_alpha[source_positions], 0.0)
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones(source_positions.size, dtype=np.float64)
            weights = weights / float(np.sum(weights))
            for local_pos, source_pos in enumerate(source_positions.tolist()):
                node = int(source_state[source_pos])
                mass = anchor_mass * float(weights[local_pos])
                gap = float(enthalpy[node] - net.source_enthalpy_set[source_pos])
                incoming_mass[node] += mass
                residual[node] += mass * gap
                add_local(node, node, mass)

        for node in range(count):
            if incoming_mass[node] > 1e-12:
                continue
            residual[node] = enthalpy[node] - net.node_enthalpy[node]
            add_local(node, node, 1.0)

        if not return_jacobian:
            return residual, None, None

        cross = coo_matrix(
            (
                np.asarray(cross_data, dtype=np.float64),
                (np.asarray(cross_rows), np.asarray(cross_cols)),
            ),
            shape=(count, self.hydraulic_state_count),
        ).tocsr()
        local = coo_matrix(
            (
                np.asarray(local_data, dtype=np.float64),
                (np.asarray(local_rows), np.asarray(local_cols)),
            ),
            shape=(count, count),
        ).tocsc()
        return residual, cross, local

    def _residual_and_jacobian(
        self,
        x: np.ndarray,
        *,
        return_jacobian: bool = True,
    ):
        if self.total_vars == 0 and self.total_eq == 0:
            empty = np.empty(0, dtype=np.float64)
            jacobian = csc_matrix((0, 0), dtype=np.float64) if return_jacobian else None
            return empty, jacobian, empty, empty
        hydraulic_x = np.asarray(x[: self.hydraulic_state_count], dtype=np.float64)
        hydraulic_residual, hydraulic_jacobian, potential, edge_flow = (
            self._hydraulic_residual_and_jacobian(
                hydraulic_x,
                return_jacobian=return_jacobian,
            )
        )
        if not self.network.thermal and not self.network.steam:
            return hydraulic_residual, hydraulic_jacobian, potential, edge_flow

        if return_jacobian:
            edge_jacobian = self._edge_flow_jacobian(potential)
            source_flow, source_jacobian = self._source_flows_and_jacobian(
                hydraulic_x,
                edge_flow,
                edge_jacobian,
            )
        else:
            edge_jacobian = None
            source_flow, source_jacobian = self._source_flows(
                hydraulic_x,
                edge_flow,
            )
        self._joint_source_flow = source_flow
        if self.network.thermal:
            transport_residual, cross_jacobian, transport_jacobian = (
                self._heat_transport_residual_and_jacobian(
                    x,
                    edge_flow,
                    edge_jacobian,
                    source_flow,
                    source_jacobian,
                )
            )
        else:
            transport_residual, cross_jacobian, transport_jacobian = (
                self._steam_transport_residual_and_jacobian(
                    x,
                    edge_flow,
                    edge_jacobian,
                    source_flow,
                    source_jacobian,
                )
            )
        residual = np.concatenate((hydraulic_residual, transport_residual))
        if not return_jacobian:
            return residual, None, potential, edge_flow
        jacobian = bmat(
            [
                [hydraulic_jacobian, None],
                [cross_jacobian, transport_jacobian],
            ],
            format="csc",
        )
        return residual, jacobian, potential, edge_flow

    def get_f(self, x: np.ndarray) -> np.ndarray:
        return self._residual_and_jacobian(x, return_jacobian=False)[0]

    def get_jacobi(self, x: np.ndarray) -> csc_matrix:
        return self._residual_and_jacobian(x)[1]

    def _build_newton_system(self, x: np.ndarray, *, return_jacobian=True, jacobian_format="csc"):
        residual, jacobian, _potential, _edge_flow = self._residual_and_jacobian(
            x,
            return_jacobian=return_jacobian,
        )
        if not return_jacobian:
            return residual, None
        return residual, jacobian if jacobian_format == "csc" else jacobian.tocsr()

    def run(self, result_mode=None) -> int:
        if result_mode is not None:
            self.result_mode = normalize_result_mode(result_mode, f"{self.network.prefix} LF")
        if not self.prepared:
            self.prepare()
        if self.total_vars == 0 and self.total_eq == 0:
            self.converged = True
            self.iterations = 0
            self.normF = 0.0
            self.failure_reason = ""
            self._allocate_source_flows()
            self._write_back()
            return 0
        self.converged = False
        self.iterations = 0
        x = self.x.copy()
        minimum = self._minimum_potential()
        for iteration in range(1, self.max_iter + 1):
            residual, jacobian, potential, edge_flow = self._residual_and_jacobian(x)
            self.iterations = iteration
            self.normF = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if self.verbose:
                print(f"Iter {iteration}: |F| = {self.normF:.6e}")
            if self.normF < self.tol:
                self.converged = True
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    correction = spsolve(csc_matrix(jacobian), -residual)
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                self.failure_reason = str(exc)
                break
            correction = np.asarray(correction, dtype=np.float64)
            if not np.all(np.isfinite(correction)):
                self.failure_reason = "non-finite Newton correction"
                break
            accepted = False
            step = 1.0
            for _ in range(24):
                candidate = x + step * correction
                if self.network.free_node_pos.size:
                    candidate[: self.network.free_node_pos.size] = np.maximum(
                        candidate[: self.network.free_node_pos.size], minimum
                    )
                candidate_residual, _ = self._build_newton_system(
                    candidate,
                    return_jacobian=False,
                )
                candidate_norm = (
                    float(np.linalg.norm(candidate_residual, np.inf)) if candidate_residual.size else 0.0
                )
                if np.isfinite(candidate_norm) and candidate_norm < self.normF:
                    x = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self.failure_reason = "line search could not reduce the residual"
                break

        residual, _, _potential, _edge_flow = self._residual_and_jacobian(
            x,
            return_jacobian=False,
        )
        self.normF = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
        if self.normF < self.tol:
            self.converged = True
        self.commit_state(
            x,
            converged=self.converged,
            iterations=self.iterations,
            normF=self.normF,
        )
        return 0 if self.converged else 1

    def commit_state(
        self,
        x: np.ndarray,
        *,
        converged: Optional[bool] = None,
        iterations: Optional[int] = None,
        normF: Optional[float] = None,
    ) -> None:
        """Write a state solved by either the local or global Newton loop."""
        state = np.asarray(x, dtype=np.float64).copy()
        if self.total_vars == 0 and self.total_eq == 0:
            self.x = state
            self.normF = 0.0 if normF is None else float(normF)
            if converged is not None:
                self.converged = bool(converged)
            if iterations is not None:
                self.iterations = int(iterations)
            self._allocate_source_flows()
            self._write_back()
            return
        residual, _jacobian, potential, edge_flow = self._residual_and_jacobian(
            state,
            return_jacobian=False,
        )
        self.x = state
        self.potential = potential
        self.pressure = np.maximum(potential, self._minimum_potential()) ** (
            1.0 / self.network.potential_power
        )
        self.edge_flow = edge_flow
        self.source_flow, _source_jacobian = self._source_flows(
            state[: self.hydraulic_state_count],
            edge_flow,
        )
        if self.network.pressure_source_group_nodes.size:
            self.pressure_source_group_flow = state[
                self.base_pressure_source_group_flow : self.hydraulic_state_count
            ].copy()
        if self.network.thermal:
            self.heat_temperature_state = state[
                self.base_temperature : self.base_enthalpy
            ].copy()
            self._sync_heat_temperature_views()
        elif self.network.steam:
            self.enthalpy = state[self.base_enthalpy :].copy()
            self.temperature = self._steam_temperature(self.enthalpy)
        self.normF = (
            float(np.linalg.norm(residual, np.inf)) if normF is None else float(normF)
        )
        if converged is not None:
            self.converged = bool(converged)
        if iterations is not None:
            self.iterations = int(iterations)
        self._write_back()

    def _allocate_source_flows(self) -> None:
        net = self.network
        if not net.nodes:
            self.source_flow = np.empty(0, dtype=np.float64)
            self.pressure_source_group_flow = np.empty(0, dtype=np.float64)
            return
        source_flow, _ = self._explicit_pressure_source_flows(
            self.x,
            return_shares=False,
        )
        if net.pressure_source_group_nodes.size:
            self.pressure_source_group_flow = self.x[
                self.base_pressure_source_group_flow : self.hydraulic_state_count
            ].copy()
        node_edge_balance = np.asarray(net.incidence @ self.edge_flow, dtype=np.float64)
        for node_pos in range(len(net.nodes)):
            source_positions = np.flatnonzero(net.source_node_pos == node_pos)
            if source_positions.size == 0:
                continue
            pressure_positions = source_positions[
                net.source_is_pressure_controlled[source_positions]
            ]
            if net.thermal and pressure_positions.size:
                pressure_positions = pressure_positions[
                    ~net.source_explicit_return[pressure_positions]
                ]
            if pressure_positions.size == 0:
                continue
            pressure_target = float(
                net.demand[node_pos] - node_edge_balance[node_pos] - net.fixed_injection[node_pos]
            )
            source_flow[pressure_positions] = allocate_limited_residual(
                net.source_flow_set[pressure_positions],
                pressure_target,
                lower=net.source_flow_min[pressure_positions],
                upper=net.source_flow_max[pressure_positions],
                alpha=net.source_alpha[pressure_positions],
            )
        self.source_flow = source_flow

    def _edge_attenuation(self) -> np.ndarray:
        if not self.network.thermal and not self.network.steam:
            return np.ones(len(self.network.edges), dtype=np.float64)
        mass = np.abs(self.edge_flow)
        attenuation = np.ones(mass.size, dtype=np.float64)
        active = mass > self._transport_flow_tolerance()
        attenuation[active] = np.exp(
            -self.network.edge_heat_loss[active] / np.maximum(mass[active], 1e-9)
        )
        return attenuation

    def _heat_edge_result_arrays(self) -> Dict[str, np.ndarray]:
        """Return endpoint temperatures and heat flows for heat-network edges."""
        net = self.network
        count = len(net.edges)
        flow = np.asarray(self.edge_flow, dtype=np.float64)
        active = np.abs(flow) > THERMAL_ZERO_FLOW_TOLERANCE
        attenuation = self._edge_attenuation()
        nan = np.full(count, np.nan, dtype=np.float64)
        values = {
            "edge_thermal_active": active,
            "edge_i_temperature": nan.copy(),
            "edge_j_temperature": nan.copy(),
            "edge_i_supply_temperature": nan.copy(),
            "edge_j_supply_temperature": nan.copy(),
            "edge_i_return_temperature": nan.copy(),
            "edge_j_return_temperature": nan.copy(),
            "edge_i_supply_heat_power": nan.copy(),
            "edge_j_supply_heat_power": nan.copy(),
            "edge_i_return_heat_power": nan.copy(),
            "edge_j_return_heat_power": nan.copy(),
            "edge_i_heat_power": np.zeros(count, dtype=np.float64),
            "edge_j_heat_power": np.zeros(count, dtype=np.float64),
            "edge_heat_loss": np.zeros(count, dtype=np.float64),
        }
        cp = float(net.medium.heat_capacity)
        ambient = float(net.medium.ambient_temperature)
        for edge_pos in range(count):
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            edge_flow = float(flow[edge_pos])
            factor = float(attenuation[edge_pos])
            if net.node_explicit_return[i]:
                if not active[edge_pos]:
                    i_temperature = float(self.temperature[i])
                    j_temperature = float(self.temperature[j])
                elif edge_flow > 0.0:
                    i_temperature = float(self.temperature[i])
                    j_temperature = ambient + (i_temperature - ambient) * factor
                else:
                    j_temperature = float(self.temperature[j])
                    i_temperature = ambient + (j_temperature - ambient) * factor
                i_heat_power = edge_flow * cp * (i_temperature - ambient)
                j_heat_power = -edge_flow * cp * (j_temperature - ambient)
                values["edge_i_temperature"][edge_pos] = i_temperature
                values["edge_j_temperature"][edge_pos] = j_temperature
            else:
                if not active[edge_pos]:
                    i_supply = float(self.supply_temperature[i])
                    j_supply = float(self.supply_temperature[j])
                    i_return = float(self.return_temperature[i])
                    j_return = float(self.return_temperature[j])
                elif edge_flow > 0.0:
                    i_supply = float(self.supply_temperature[i])
                    j_supply = ambient + (i_supply - ambient) * factor
                    j_return = float(self.return_temperature[j])
                    i_return = ambient + (j_return - ambient) * factor
                else:
                    j_supply = float(self.supply_temperature[j])
                    i_supply = ambient + (j_supply - ambient) * factor
                    i_return = float(self.return_temperature[i])
                    j_return = ambient + (i_return - ambient) * factor
                i_supply_heat_power = edge_flow * cp * (i_supply - ambient)
                j_supply_heat_power = -edge_flow * cp * (j_supply - ambient)
                i_return_heat_power = -edge_flow * cp * (i_return - ambient)
                j_return_heat_power = edge_flow * cp * (j_return - ambient)
                i_heat_power = i_supply_heat_power + i_return_heat_power
                j_heat_power = j_supply_heat_power + j_return_heat_power
                values["edge_i_supply_temperature"][edge_pos] = i_supply
                values["edge_j_supply_temperature"][edge_pos] = j_supply
                values["edge_i_return_temperature"][edge_pos] = i_return
                values["edge_j_return_temperature"][edge_pos] = j_return
                values["edge_i_supply_heat_power"][edge_pos] = i_supply_heat_power
                values["edge_j_supply_heat_power"][edge_pos] = j_supply_heat_power
                values["edge_i_return_heat_power"][edge_pos] = i_return_heat_power
                values["edge_j_return_heat_power"][edge_pos] = j_return_heat_power
            values["edge_i_heat_power"][edge_pos] = i_heat_power
            values["edge_j_heat_power"][edge_pos] = j_heat_power
            values["edge_heat_loss"][edge_pos] = i_heat_power + j_heat_power
        return values

    def _solve_heat_temperatures(self) -> None:
        net = self.network
        n_temperature = int(net.temperature_state_count)
        cp = max(float(net.medium.heat_capacity), 1e-12)
        ambient = float(net.medium.ambient_temperature)
        attenuation = self._edge_attenuation()

        rows = []
        cols = []
        data = []
        rhs = np.zeros(n_temperature, dtype=np.float64)
        incoming_mass = np.zeros(n_temperature, dtype=np.float64)

        def add(row: int, col: int, value: float) -> None:
            rows.append(int(row))
            cols.append(int(col))
            data.append(float(value))

        def add_transport(
            upstream_state: int,
            downstream_state: int,
            mass: float,
            factor: float,
        ) -> None:
            incoming_mass[downstream_state] += mass
            add(downstream_state, upstream_state, -mass * factor)
            rhs[downstream_state] += mass * (1.0 - factor) * ambient

        for edge_pos, flow in enumerate(self.edge_flow.tolist()):
            if not self._active_transport_flow(flow):
                continue
            if flow > 0.0:
                upstream, downstream = int(net.edge_i[edge_pos]), int(net.edge_j[edge_pos])
            else:
                upstream, downstream = int(net.edge_j[edge_pos]), int(net.edge_i[edge_pos])
            mass = abs(float(flow))
            factor = float(attenuation[edge_pos])
            if net.node_explicit_return[upstream]:
                add_transport(
                    int(net.supply_temperature_state_by_node[upstream]),
                    int(net.supply_temperature_state_by_node[downstream]),
                    mass,
                    factor,
                )
            else:
                add_transport(
                    int(net.supply_temperature_state_by_node[upstream]),
                    int(net.supply_temperature_state_by_node[downstream]),
                    mass,
                    factor,
                )
                add_transport(
                    int(net.return_temperature_state_by_node[downstream]),
                    int(net.return_temperature_state_by_node[upstream]),
                    mass,
                    factor,
                )
        for source_pos, flow in enumerate(self.source_flow.tolist()):
            injection = self._thermal_source_injection(source_pos, flow)
            if injection is None:
                continue
            state, mass, temperature_set, _flow_derivative_scale = injection
            incoming_mass[state] += mass
            rhs[state] += mass * temperature_set
        for load_pos in range(len(net.loads)):
            mass = float(net.load_flow_set[load_pos])
            if mass <= THERMAL_ZERO_FLOW_TOLERANCE:
                continue
            supply_node = int(net.load_supply_node_pos[load_pos])
            return_node = int(net.load_return_node_pos[load_pos])
            supply_state = int(net.supply_temperature_state_by_node[supply_node])
            return_state = int(net.return_temperature_state_by_node[return_node])
            incoming_mass[return_state] += mass
            add(return_state, supply_state, -mass)
            rhs[return_state] -= float(net.load_heat_power[load_pos]) / cp

        for exchanger_pos in range(int(net.exchanger_i.size)):
            primary_supply = int(net.exchanger_primary_supply[exchanger_pos])
            primary_return = int(net.exchanger_primary_return[exchanger_pos])
            secondary_return = int(net.exchanger_secondary_return[exchanger_pos])
            secondary_supply = int(net.exchanger_secondary_supply[exchanger_pos])
            primary_supply_state = int(
                net.supply_temperature_state_by_node[primary_supply]
            )
            primary_return_state = int(
                net.return_temperature_state_by_node[primary_return]
            )
            secondary_return_state = int(
                net.return_temperature_state_by_node[secondary_return]
            )
            secondary_supply_state = int(
                net.supply_temperature_state_by_node[secondary_supply]
            )
            primary_mass = float(net.exchanger_primary_flow[exchanger_pos])
            secondary_mass = float(net.exchanger_secondary_flow[exchanger_pos])
            if (
                primary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
                or secondary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
            ):
                continue
            control = str(net.exchanger_control_type[exchanger_pos])
            loss_factor = 1.0 - float(net.exchanger_heat_loss[exchanger_pos])
            incoming_mass[secondary_supply_state] += secondary_mass
            incoming_mass[primary_return_state] += primary_mass
            if control == "EFFECTIVENESS":
                transfer_mass = float(net.exchanger_effectiveness[exchanger_pos]) * min(
                    primary_mass, secondary_mass
                )
                secondary_transfer_mass = loss_factor * transfer_mass
                add(
                    secondary_supply_state,
                    primary_supply_state,
                    -secondary_transfer_mass,
                )
                add(
                    secondary_supply_state,
                    secondary_return_state,
                    -(secondary_mass - secondary_transfer_mass),
                )
                add(
                    primary_return_state,
                    primary_supply_state,
                    -(primary_mass - transfer_mass),
                )
                add(primary_return_state, secondary_return_state, -transfer_mass)
            else:
                primary_heat = float(net.exchanger_heat_set[exchanger_pos])
                secondary_heat = loss_factor * primary_heat
                add(secondary_supply_state, secondary_return_state, -secondary_mass)
                rhs[secondary_supply_state] += secondary_heat / cp
                add(primary_return_state, primary_supply_state, -primary_mass)
                rhs[primary_return_state] -= primary_heat / cp

        for state_pos in range(n_temperature):
            if incoming_mass[state_pos] <= THERMAL_ZERO_FLOW_TOLERANCE:
                add(state_pos, state_pos, 1.0)
                rhs[state_pos] = net.initial_temperature_state[state_pos]
            else:
                add(state_pos, state_pos, incoming_mass[state_pos])
        fixed_states = net.fixed_temperature_state_pos
        if fixed_states.size:
            fixed_mask = np.zeros(n_temperature, dtype=bool)
            fixed_mask[fixed_states] = True
            row_array = np.asarray(rows, dtype=np.int64)
            col_array = np.asarray(cols, dtype=np.int64)
            data_array = np.asarray(data, dtype=np.float64)
            keep = ~fixed_mask[row_array]
            rows = row_array[keep].tolist()
            cols = col_array[keep].tolist()
            data = data_array[keep].tolist()
            rows.extend(fixed_states.tolist())
            cols.extend(fixed_states.tolist())
            data.extend(np.ones(fixed_states.size, dtype=np.float64).tolist())
            rhs[fixed_states] = net.fixed_temperature
        matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_temperature, n_temperature)
        ).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                temperature = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
            except MatrixRankWarning as exc:
                raise RuntimeError(
                    "heat transport equations are singular; fixed-heat exchanger loops require a temperature anchor"
                ) from exc
        if not np.all(np.isfinite(temperature)):
            raise RuntimeError("heat transport produced non-finite supply/return temperatures")
        self.heat_temperature_state = temperature
        self._sync_heat_temperature_views()

    def _heat_exchanger_quantities(self):
        net = self.network
        count = int(net.exchanger_i.size)
        primary_heat = np.zeros(count, dtype=np.float64)
        secondary_heat = np.zeros(count, dtype=np.float64)
        primary_out = np.zeros(count, dtype=np.float64)
        secondary_out = np.zeros(count, dtype=np.float64)
        cp = max(float(net.medium.heat_capacity), 1e-12)
        for pos in range(count):
            i = int(net.exchanger_i[pos])
            j = int(net.exchanger_j[pos])
            primary_mass = float(net.exchanger_primary_flow[pos])
            secondary_mass = float(net.exchanger_secondary_flow[pos])
            if (
                primary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
                or secondary_mass <= THERMAL_ZERO_FLOW_TOLERANCE
            ):
                primary_return = int(net.exchanger_primary_return[pos])
                secondary_supply = int(net.exchanger_secondary_supply[pos])
                primary_out[pos] = self.return_temperature[primary_return]
                secondary_out[pos] = self.supply_temperature[secondary_supply]
                continue
            if str(net.exchanger_control_type[pos]) == "EFFECTIVENESS":
                primary_heat[pos] = (
                    net.exchanger_effectiveness[pos]
                    * min(primary_mass, secondary_mass)
                    * cp
                    * (self.supply_temperature[i] - self.return_temperature[j])
                )
            else:
                primary_heat[pos] = net.exchanger_heat_set[pos]
            secondary_heat[pos] = primary_heat[pos] * (1.0 - net.exchanger_heat_loss[pos])
            primary_out[pos] = self.supply_temperature[i] - primary_heat[pos] / (primary_mass * cp)
            secondary_out[pos] = self.return_temperature[j] + secondary_heat[pos] / (secondary_mass * cp)
        return primary_heat, secondary_heat, primary_out, secondary_out

    def _steam_temperature(self, enthalpy) -> np.ndarray:
        net = self.network
        cp = max(float(net.medium.heat_capacity), 1e-12)
        return (
            float(net.medium.reference_temperature)
            + (np.asarray(enthalpy, dtype=np.float64) - float(net.medium.reference_enthalpy)) / cp
        )

    def _solve_steam_enthalpy(self) -> None:
        net = self.network
        n = len(net.nodes)
        attenuation = self._edge_attenuation()
        ambient_enthalpy = float(net.medium.ambient_enthalpy)
        rows = []
        cols = []
        data = []
        rhs = np.zeros(n, dtype=np.float64)
        incoming_mass = np.zeros(n, dtype=np.float64)
        for edge_pos, flow in enumerate(self.edge_flow.tolist()):
            if abs(flow) <= 1e-12:
                continue
            if flow > 0.0:
                upstream, downstream = int(net.edge_i[edge_pos]), int(net.edge_j[edge_pos])
            else:
                upstream, downstream = int(net.edge_j[edge_pos]), int(net.edge_i[edge_pos])
            mass = abs(float(flow))
            factor = float(attenuation[edge_pos])
            incoming_mass[downstream] += mass
            rows.append(downstream)
            cols.append(upstream)
            data.append(-mass * factor)
            rhs[downstream] += mass * (1.0 - factor) * ambient_enthalpy
        for source_pos, flow in enumerate(self.source_flow.tolist()):
            if flow <= 1e-12:
                continue
            node_pos = int(net.source_node_pos[source_pos])
            incoming_mass[node_pos] += flow
            rhs[node_pos] += flow * net.source_enthalpy_set[source_pos]
        for node_pos in range(n):
            rows.append(node_pos)
            cols.append(node_pos)
            if incoming_mass[node_pos] <= 1e-12:
                data.append(1.0)
                rhs[node_pos] = net.node_enthalpy[node_pos]
            else:
                data.append(incoming_mass[node_pos])
        matrix = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                enthalpy = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
            except MatrixRankWarning as exc:
                raise RuntimeError("steam enthalpy transport equations are singular") from exc
        if not np.all(np.isfinite(enthalpy)):
            raise RuntimeError("steam enthalpy transport produced non-finite values")
        self.enthalpy = enthalpy
        self.temperature = self._steam_temperature(enthalpy)

    def _write_back(self) -> None:
        net = self.network
        edge_i_pressure = self.pressure[net.edge_i].copy()
        edge_j_pressure = self.pressure[net.edge_j].copy()
        source_supply_pressure = self.pressure[net.source_supply_node_pos].copy()
        source_return_pressure = self.pressure[net.source_return_node_pos].copy()
        load_supply_pressure = self.pressure[net.load_supply_node_pos].copy()
        load_return_pressure = self.pressure[net.load_return_node_pos].copy()
        arrays = {
            "node_pressure": self.pressure.copy(),
            "edge_flow": self.edge_flow.copy(),
            "edge_i_pressure": edge_i_pressure,
            "edge_j_pressure": edge_j_pressure,
            "source_flow": self.source_flow.copy(),
            "source_supply_pressure": source_supply_pressure,
            "source_return_pressure": source_return_pressure,
            "storage_flow": self.source_flow[net.storage_source_pos].copy(),
            "load_flow": net.load_flow_set.copy(),
            "load_supply_pressure": load_supply_pressure,
            "load_return_pressure": load_return_pressure,
        }
        heat_edge_arrays = None
        if net.thermal:
            heat_edge_arrays = self._heat_edge_result_arrays()
            arrays.update(heat_edge_arrays)
            arrays["node_supply_temperature"] = self.supply_temperature.copy()
            arrays["node_return_temperature"] = self.return_temperature.copy()
            arrays["node_temperature"] = self.temperature.copy()
            arrays["node_explicit_return"] = net.node_explicit_return.copy()
            source_supply_temperature = self.supply_temperature[
                net.source_supply_node_pos
            ].copy()
            source_return_temperature = self.return_temperature[
                net.source_return_node_pos
            ].copy()
            source_heat_power = (
                self.source_flow
                * float(net.medium.heat_capacity)
                * (source_supply_temperature - source_return_temperature)
            )
            arrays["source_supply_temperature"] = source_supply_temperature
            arrays["source_return_temperature"] = source_return_temperature
            arrays["source_heat_power"] = source_heat_power
            arrays["storage_heat_power"] = source_heat_power[
                net.storage_source_pos
            ].copy()
            arrays["load_supply_temperature"] = self.supply_temperature[
                net.load_supply_node_pos
            ].copy()
            arrays["load_return_temperature"] = self.return_temperature[
                net.load_return_node_pos
            ].copy()
            arrays["load_heat_power"] = net.load_heat_power.copy()
            if net.exchanger_i.size:
                primary_heat, secondary_heat, primary_out, secondary_out = self._heat_exchanger_quantities()
                arrays["exchanger_primary_heat"] = primary_heat
                arrays["exchanger_secondary_heat"] = secondary_heat
                arrays["exchanger_primary_out_temperature"] = primary_out
                arrays["exchanger_secondary_out_temperature"] = secondary_out
        metadata = (
            {"hydraulic_presolve": dict(self.hydraulic_presolve)}
            if self.hydraulic_presolve
            else {}
        )
        result = self.result_class(arrays=arrays, metadata=metadata)
        if self.result_mode in {"none", "array", "summary"}:
            self.lf_result = result
            return

        for pos, node in enumerate(net.nodes):
            values = {
                "idx": int(node.idx),
                "name": str(node.name),
                "device_type": f"{net.prefix}Node",
                "pressure": float(self.pressure[pos]),
                "island": int(net.node_island[pos]),
            }
            if net.thermal:
                if net.node_explicit_return[pos]:
                    values["temperature"] = float(self.temperature[pos])
                else:
                    values.update(
                        supply_temperature=float(self.supply_temperature[pos]),
                        return_temperature=float(self.return_temperature[pos]),
                    )
            result.nodes[node.name] = SimpleNamespace(**values)
        for edge_pos, edge in enumerate(net.edges):
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            values = {
                "idx": int(edge.idx),
                "name": str(edge.name),
                "kind": str(edge.kind),
                "device_type": (
                    f"{net.prefix}"
                    + str(edge.kind).replace("_", " ").title().replace(" ", "")
                ),
                "flow": float(self.edge_flow[edge_pos]),
                "i_flow": float(self.edge_flow[edge_pos]),
                "j_flow": float(-self.edge_flow[edge_pos]),
                "i_pressure": float(self.pressure[i]),
                "j_pressure": float(self.pressure[j]),
            }
            if net.thermal:
                thermal_active = bool(
                    heat_edge_arrays["edge_thermal_active"][edge_pos]
                )
                values["thermal_active"] = thermal_active
                if net.node_explicit_return[i]:
                    values.update(
                        i_temperature=float(
                            heat_edge_arrays["edge_i_temperature"][edge_pos]
                        ),
                        j_temperature=float(
                            heat_edge_arrays["edge_j_temperature"][edge_pos]
                        ),
                    )
                else:
                    values.update(
                        i_supply_temperature=float(
                            heat_edge_arrays["edge_i_supply_temperature"][edge_pos]
                        ),
                        j_supply_temperature=float(
                            heat_edge_arrays["edge_j_supply_temperature"][edge_pos]
                        ),
                        i_return_temperature=float(
                            heat_edge_arrays["edge_i_return_temperature"][edge_pos]
                        ),
                        j_return_temperature=float(
                            heat_edge_arrays["edge_j_return_temperature"][edge_pos]
                        ),
                        i_supply_heat_power=float(
                            heat_edge_arrays["edge_i_supply_heat_power"][edge_pos]
                        ),
                        j_supply_heat_power=float(
                            heat_edge_arrays["edge_j_supply_heat_power"][edge_pos]
                        ),
                        i_return_heat_power=float(
                            heat_edge_arrays["edge_i_return_heat_power"][edge_pos]
                        ),
                        j_return_heat_power=float(
                            heat_edge_arrays["edge_j_return_heat_power"][edge_pos]
                        ),
                    )
                values.update(
                    i_heat_power=float(
                        heat_edge_arrays["edge_i_heat_power"][edge_pos]
                    ),
                    j_heat_power=float(
                        heat_edge_arrays["edge_j_heat_power"][edge_pos]
                    ),
                    heat_loss=float(heat_edge_arrays["edge_heat_loss"][edge_pos]),
                )
            namespace = SimpleNamespace(**values)
            if edge.kind == "pipe":
                result.pipes[edge.name] = namespace
            elif edge.kind == "valve":
                result.valves[edge.name] = namespace
            else:
                result.controllers[edge.name] = namespace
                if edge.kind == "pump":
                    result.pumps[edge.name] = namespace
                elif edge.kind == "compressor":
                    result.compressors[edge.name] = namespace
                elif edge.kind == "pressure_reducer":
                    result.pressure_reducers[edge.name] = namespace
        for source_pos, source in enumerate(net.sources):
            supply_node = int(net.source_supply_node_pos[source_pos])
            return_node = int(net.source_return_node_pos[source_pos])
            values = {
                "idx": int(source.idx),
                "name": str(source.name),
                "device_type": (
                    f"{net.prefix}Storage"
                    if bool(net.source_is_storage[source_pos])
                    else f"{net.prefix}Source"
                ),
                "flow": float(self.source_flow[source_pos]),
            }
            if net.thermal:
                values.update(
                    supply_pressure=float(self.pressure[supply_node]),
                    return_pressure=float(self.pressure[return_node]),
                    supply_temperature=float(self.supply_temperature[supply_node]),
                    return_temperature=float(self.return_temperature[return_node]),
                    heat_power=float(
                        self.source_flow[source_pos]
                        * net.medium.heat_capacity
                        * (
                            self.supply_temperature[supply_node]
                            - self.return_temperature[return_node]
                        )
                    ),
                )
                if not net.source_explicit_return[source_pos]:
                    values["pressure"] = float(self.pressure[supply_node])
            else:
                values["pressure"] = float(self.pressure[supply_node])
            collection = (
                result.storages
                if bool(net.source_is_storage[source_pos])
                else result.sources
            )
            collection[source.name] = SimpleNamespace(**values)
        for load_pos, load in enumerate(net.loads):
            supply_node = int(net.load_supply_node_pos[load_pos])
            return_node = int(net.load_return_node_pos[load_pos])
            values = {
                "idx": int(load.idx),
                "name": str(load.name),
                "device_type": f"{net.prefix}Load",
                "flow": float(net.load_flow_set[load_pos]),
            }
            if net.thermal:
                values.update(
                    supply_pressure=float(self.pressure[supply_node]),
                    return_pressure=float(self.pressure[return_node]),
                    supply_temperature=float(self.supply_temperature[supply_node]),
                    return_temperature=float(self.return_temperature[return_node]),
                    heat_power=float(net.load_heat_power[load_pos]),
                )
                if not net.load_explicit_return[load_pos]:
                    values["pressure"] = float(self.pressure[supply_node])
            else:
                values["pressure"] = float(self.pressure[supply_node])
            result.loads[load.name] = SimpleNamespace(**values)
        if net.thermal and net.exchanger_i.size:
            primary_heat, secondary_heat, primary_out, secondary_out = self._heat_exchanger_quantities()
            for pos, exchanger in enumerate(net.heat_exchangers):
                primary_supply = int(net.exchanger_primary_supply[pos])
                primary_return = int(net.exchanger_primary_return[pos])
                secondary_return = int(net.exchanger_secondary_return[pos])
                secondary_supply = int(net.exchanger_secondary_supply[pos])
                result.heat_exchangers[exchanger.name] = SimpleNamespace(
                    primary_flow=float(net.exchanger_primary_flow[pos]),
                    secondary_flow=float(net.exchanger_secondary_flow[pos]),
                    primary_supply_pressure=float(self.pressure[primary_supply]),
                    primary_return_pressure=float(self.pressure[primary_return]),
                    secondary_return_pressure=float(self.pressure[secondary_return]),
                    secondary_supply_pressure=float(self.pressure[secondary_supply]),
                    primary_in_temperature=float(self.supply_temperature[primary_supply]),
                    primary_out_temperature=float(primary_out[pos]),
                    secondary_in_temperature=float(self.return_temperature[secondary_return]),
                    secondary_out_temperature=float(secondary_out[pos]),
                    primary_heat=float(primary_heat[pos]),
                    secondary_heat=float(secondary_heat[pos]),
                    thermal_active=bool(
                        net.exchanger_primary_flow[pos]
                        > THERMAL_ZERO_FLOW_TOLERANCE
                        and net.exchanger_secondary_flow[pos]
                        > THERMAL_ZERO_FLOW_TOLERANCE
                    ),
                )
        self.lf_result = result


def print_fluid_result(calc: FluidPowerFlowCalc, rc: int) -> None:
    net = calc.network
    print(f"{net.prefix} load flow: {'converged' if rc == 0 else 'not converged'}")
    print(
        f"  nodes={len(net.nodes)}, edges={len(net.edges)}, islands={net.island_count}, "
        f"iterations={calc.iterations}, residual={calc.normF:.6e}"
    )
    for warning in net.warnings:
        print(f"  warning: {warning}")
    if calc.result_mode != "full":
        return
    print("  node results:")
    for pos, (name, item) in enumerate(calc.lf_result.nodes.items()):
        if net.steam:
            print(
                f"    {name}: pressure={item.pressure:.6f}, "
                f"temperature={item.temperature:.6f}, "
                f"enthalpy={item.enthalpy:.6f}"
            )
        elif net.thermal:
            if net.node_explicit_return[pos]:
                print(
                    f"    {name}: pressure={item.pressure:.6f}, "
                    f"temperature={item.temperature:.6f}"
                )
            else:
                print(
                    f"    {name}: pressure={item.pressure:.6f}, "
                    f"Ts={item.supply_temperature:.6f}, Tr={item.return_temperature:.6f}"
                )
        else:
            print(f"    {name}: pressure={item.pressure:.6f}")
    def edge_text(item) -> str:
        text = (
            f"flow={item.flow:.6f}, Pi={item.i_pressure:.6f}, "
            f"Pj={item.j_pressure:.6f}"
        )
        if hasattr(item, "i_supply_temperature"):
            text += (
                f", Tsi={item.i_supply_temperature:.6f}, "
                f"Tsj={item.j_supply_temperature:.6f}, "
                f"Tri={item.i_return_temperature:.6f}, "
                f"Trj={item.j_return_temperature:.6f}"
            )
        elif hasattr(item, "i_temperature"):
            text += (
                f", Ti={item.i_temperature:.6f}, "
                f"Tj={item.j_temperature:.6f}"
            )
        if hasattr(item, "i_enthalpy"):
            text += f", hi={item.i_enthalpy:.6f}, hj={item.j_enthalpy:.6f}"
        if hasattr(item, "i_heat_power"):
            text += (
                f", Qi={item.i_heat_power:.6f}, "
                f"Qj={item.j_heat_power:.6f}, loss={item.heat_loss:.6f}"
            )
        return text

    categorized_controllers = set()
    edge_collections = (
        ("pipes", calc.lf_result.pipes),
        ("valves", calc.lf_result.valves),
        ("pumps", calc.lf_result.pumps),
        ("compressors", calc.lf_result.compressors),
        ("pressure reducers", calc.lf_result.pressure_reducers),
    )
    for label, collection in edge_collections:
        if not collection:
            continue
        print(f"  {label}:")
        categorized_controllers.update(collection)
        for name, item in collection.items():
            print(f"    {name}: {edge_text(item)}")
    other_controllers = {
        name: item
        for name, item in calc.lf_result.controllers.items()
        if name not in categorized_controllers
    }
    if other_controllers:
        print("  controllers:")
        for name, item in other_controllers.items():
            print(f"    {name}: {edge_text(item)}")

    def terminal_text(item) -> str:
        text = f"flow={item.flow:.6f}"
        if hasattr(item, "supply_pressure"):
            text += (
                f", Ps={item.supply_pressure:.6f}, "
                f"Pr={item.return_pressure:.6f}"
            )
        elif hasattr(item, "pressure"):
            text += f", pressure={item.pressure:.6f}"
        if hasattr(item, "supply_temperature"):
            text += (
                f", Ts={item.supply_temperature:.6f}, "
                f"Tr={item.return_temperature:.6f}"
            )
        elif hasattr(item, "temperature"):
            text += f", temperature={item.temperature:.6f}"
        if hasattr(item, "enthalpy"):
            text += f", enthalpy={item.enthalpy:.6f}"
        if hasattr(item, "heat_power"):
            text += f", heat={item.heat_power:.6f}"
        return text

    for label, collection in (
        ("sources", calc.lf_result.sources),
        ("storages (positive flow=discharge)", calc.lf_result.storages),
        ("loads", calc.lf_result.loads),
    ):
        if not collection:
            continue
        print(f"  {label}:")
        for name, item in collection.items():
            print(f"    {name}: {terminal_text(item)}")
    if calc.lf_result.heat_exchangers:
        print("  heat exchangers:")
        for name, item in calc.lf_result.heat_exchangers.items():
            print(
                f"    {name}: primary_flow={item.primary_flow:.6f}, "
                f"secondary_flow={item.secondary_flow:.6f}, "
                f"primary_pressure={item.primary_supply_pressure:.6f}"
                f"->{item.primary_return_pressure:.6f}, "
                f"secondary_pressure={item.secondary_return_pressure:.6f}"
                f"->{item.secondary_supply_pressure:.6f}, "
                f"primary_temperature={item.primary_in_temperature:.6f}"
                f"->{item.primary_out_temperature:.6f}, "
                f"secondary_temperature={item.secondary_in_temperature:.6f}"
                f"->{item.secondary_out_temperature:.6f}, "
                f"primary_heat={item.primary_heat:.6f}, "
                f"secondary_heat={item.secondary_heat:.6f}"
            )
