"""Shared sparse Newton load-flow kernel for steady fluid networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional
import sys

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
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
)


@dataclass
class FluidLFResult:
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    pipes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    valves: Dict[str, SimpleNamespace] = field(default_factory=dict)
    controllers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    heat_exchangers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    sources: Dict[str, SimpleNamespace] = field(default_factory=dict)
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

    def prepare(self) -> "FluidPowerFlowCalc":
        net = self.network.prepare()
        potential = net.initial_potential()
        n_pressure_group = int(net.pressure_source_group_nodes.size) if net.thermal else 0
        self.base_pressure_source_group_flow = (
            net.free_node_pos.size + net.regulated_edge_pos.size
        )
        x = np.empty(
            self.base_pressure_source_group_flow + n_pressure_group,
            dtype=np.float64,
        )
        x[: net.free_node_pos.size] = potential[net.free_node_pos]
        if net.regulated_edge_pos.size:
            x[
                net.free_node_pos.size : self.base_pressure_source_group_flow
            ] = net.edge_flow_set[net.regulated_edge_pos]
        if n_pressure_group:
            x[self.base_pressure_source_group_flow :] = np.asarray(
                [
                    np.sum(net.source_flow_set[source_positions])
                    for source_positions in net.pressure_source_groups
                ],
                dtype=np.float64,
            )
        self.x = x
        self.potential = potential
        self.pressure = np.maximum(potential, self._minimum_potential()) ** (1.0 / net.potential_power)
        self.edge_flow = np.zeros(len(net.edges), dtype=np.float64)
        self.source_flow = net.source_flow_set.copy()
        self.pressure_source_group_flow = (
            x[self.base_pressure_source_group_flow :].copy()
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

    def _state_potential_and_flow(self, x: np.ndarray):
        net = self.network
        potential = net.initial_potential()
        potential[net.free_node_pos] = x[: net.free_node_pos.size]
        edge_flow = net.edge_flow_set.copy()
        if net.regulated_edge_pos.size:
            edge_flow[net.regulated_edge_pos] = x[
                net.free_node_pos.size : self.base_pressure_source_group_flow
            ]
        passive = net.passive_edge_pos
        derivative = np.zeros(len(net.edges), dtype=np.float64)
        if passive.size:
            delta = potential[net.edge_i[passive]] - potential[net.edge_j[passive]]
            edge_flow[passive] = net.edge_conductance[passive] * np.sign(delta) * np.sqrt(np.abs(delta))
            derivative[passive] = 0.5 * net.edge_conductance[passive] / np.sqrt(
                np.maximum(np.abs(delta), 1e-10)
            )
        return potential, edge_flow, derivative

    def _residual_and_jacobian(self, x: np.ndarray):
        net = self.network
        potential, edge_flow, derivative = self._state_potential_and_flow(x)
        node_balance = net.fixed_injection - net.demand + net.incidence @ edge_flow
        source_flow, source_group_shares = self._explicit_pressure_source_flows(x)
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
                residual[row] = potential[j] - ratio2 * potential[i]
                if i_col >= 0:
                    rows.append(row)
                    cols.append(i_col)
                    data.append(-ratio2)
                if j_col >= 0:
                    rows.append(row)
                    cols.append(j_col)
                    data.append(1.0)
            elif control == GAIN_CONTROL:
                residual[row] = potential[j] - potential[i] - net.edge_pressure_gain[edge_pos]
                if i_col >= 0:
                    rows.append(row)
                    cols.append(i_col)
                    data.append(-1.0)
                if j_col >= 0:
                    rows.append(row)
                    cols.append(j_col)
                    data.append(1.0)
            else:
                residual[row] = edge_flow[edge_pos] - net.edge_flow_set[edge_pos]
                rows.append(row)
                cols.append(int(net.regulated_state_by_edge[edge_pos]))
                data.append(1.0)

        jacobian = coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(residual.size, x.size),
        ).tocsc()
        return residual, jacobian, potential, edge_flow

    def _explicit_pressure_source_flows(self, x: np.ndarray):
        net = self.network
        source_flow = net.source_flow_set.copy()
        shares_by_group = []
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

    def run(self, result_mode=None) -> int:
        if result_mode is not None:
            self.result_mode = normalize_result_mode(result_mode, f"{self.network.prefix} LF")
        if not self.prepared:
            self.prepare()
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
                candidate_residual, _, _, _ = self._residual_and_jacobian(candidate)
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

        residual, _, potential, edge_flow = self._residual_and_jacobian(x)
        self.x = x
        self.potential = potential
        self.pressure = np.maximum(potential, minimum) ** (1.0 / self.network.potential_power)
        self.edge_flow = edge_flow
        self.normF = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
        if self.normF < self.tol:
            self.converged = True
        self._allocate_source_flows()
        if self.network.thermal:
            self._solve_heat_temperatures()
        elif self.network.steam:
            self._solve_steam_enthalpy()
        self._write_back()
        return 0 if self.converged else 1

    def _allocate_source_flows(self) -> None:
        net = self.network
        source_flow, _ = self._explicit_pressure_source_flows(self.x)
        if net.pressure_source_group_nodes.size:
            self.pressure_source_group_flow = self.x[
                self.base_pressure_source_group_flow :
            ].copy()
        node_edge_balance = np.asarray(net.incidence @ self.edge_flow, dtype=np.float64)
        for node_pos in range(len(net.nodes)):
            source_positions = np.flatnonzero(net.source_node_pos == node_pos)
            if source_positions.size == 0:
                continue
            pressure_positions = source_positions[
                net.source_control_type[source_positions] == PRESSURE_CONTROL
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
        mass = np.maximum(np.abs(self.edge_flow), 1e-9)
        return np.exp(-self.network.edge_heat_loss / mass)

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
            if abs(flow) <= 1e-12:
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
            if flow > 1e-12:
                supply_node = int(net.source_supply_node_pos[source_pos])
                supply_state = int(net.supply_temperature_state_by_node[supply_node])
                incoming_mass[supply_state] += flow
                rhs[supply_state] += flow * net.source_supply_temperature[source_pos]
        for load_pos in range(len(net.loads)):
            mass = float(net.load_flow_set[load_pos])
            if mass <= 1e-12:
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
            if primary_mass <= 1e-12 or secondary_mass <= 1e-12:
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
            if incoming_mass[state_pos] <= 1e-12:
                add(state_pos, state_pos, 1.0)
                rhs[state_pos] = net.initial_temperature_state[state_pos]
            else:
                add(state_pos, state_pos, incoming_mass[state_pos])
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
            primary_mass = max(float(net.exchanger_primary_flow[pos]), 1e-12)
            secondary_mass = max(float(net.exchanger_secondary_flow[pos]), 1e-12)
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
        arrays = {
            "node_pressure": self.pressure.copy(),
            "edge_flow": self.edge_flow.copy(),
            "source_flow": self.source_flow.copy(),
            "load_flow": net.load_flow_set.copy(),
        }
        if net.thermal:
            arrays["node_supply_temperature"] = self.supply_temperature.copy()
            arrays["node_return_temperature"] = self.return_temperature.copy()
            arrays["node_temperature"] = self.temperature.copy()
            arrays["node_explicit_return"] = net.node_explicit_return.copy()
            if net.exchanger_i.size:
                primary_heat, secondary_heat, primary_out, secondary_out = self._heat_exchanger_quantities()
                arrays["exchanger_primary_heat"] = primary_heat
                arrays["exchanger_secondary_heat"] = secondary_heat
                arrays["exchanger_primary_out_temperature"] = primary_out
                arrays["exchanger_secondary_out_temperature"] = secondary_out
        result = self.result_class(arrays=arrays)
        if self.result_mode in {"none", "array", "summary"}:
            self.lf_result = result
            return

        for pos, node in enumerate(net.nodes):
            values = {
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
        attenuation = self._edge_attenuation()
        for edge_pos, edge in enumerate(net.edges):
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            values = {
                "i_flow": float(self.edge_flow[edge_pos]),
                "j_flow": float(-self.edge_flow[edge_pos]),
                "i_pressure": float(self.pressure[i]),
                "j_pressure": float(self.pressure[j]),
            }
            if net.thermal:
                flow = float(self.edge_flow[edge_pos])
                factor = float(attenuation[edge_pos])
                if net.node_explicit_return[i]:
                    if flow >= 0.0:
                        i_temperature = self.temperature[i]
                        j_temperature = net.medium.ambient_temperature + (
                            self.temperature[i] - net.medium.ambient_temperature
                        ) * factor
                    else:
                        j_temperature = self.temperature[j]
                        i_temperature = net.medium.ambient_temperature + (
                            self.temperature[j] - net.medium.ambient_temperature
                        ) * factor
                    values.update(
                        i_temperature=float(i_temperature),
                        j_temperature=float(j_temperature),
                    )
                else:
                    if flow >= 0.0:
                        i_supply = self.supply_temperature[i]
                        j_supply = net.medium.ambient_temperature + (
                            self.supply_temperature[i] - net.medium.ambient_temperature
                        ) * factor
                        j_return = self.return_temperature[j]
                        i_return = net.medium.ambient_temperature + (
                            self.return_temperature[j] - net.medium.ambient_temperature
                        ) * factor
                    else:
                        j_supply = self.supply_temperature[j]
                        i_supply = net.medium.ambient_temperature + (
                            self.supply_temperature[j] - net.medium.ambient_temperature
                        ) * factor
                        i_return = self.return_temperature[i]
                        j_return = net.medium.ambient_temperature + (
                            self.return_temperature[i] - net.medium.ambient_temperature
                        ) * factor
                    values.update(
                        i_supply_temperature=float(i_supply),
                        j_supply_temperature=float(j_supply),
                        i_return_temperature=float(i_return),
                        j_return_temperature=float(j_return),
                    )
            namespace = SimpleNamespace(**values)
            if edge.kind == "pipe":
                result.pipes[edge.name] = namespace
            elif edge.kind == "valve":
                result.valves[edge.name] = namespace
            else:
                result.controllers[edge.name] = namespace
        for source_pos, source in enumerate(net.sources):
            supply_node = int(net.source_supply_node_pos[source_pos])
            return_node = int(net.source_return_node_pos[source_pos])
            values = {"flow": float(self.source_flow[source_pos])}
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
            result.sources[source.name] = SimpleNamespace(**values)
        for load_pos, load in enumerate(net.loads):
            supply_node = int(net.load_supply_node_pos[load_pos])
            return_node = int(net.load_return_node_pos[load_pos])
            values = {"flow": float(net.load_flow_set[load_pos])}
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
        if net.thermal:
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
    print("  edge flows:")
    for collection in (calc.lf_result.pipes, calc.lf_result.valves, calc.lf_result.controllers):
        for name, item in collection.items():
            print(f"    {name}: i_flow={item.i_flow:.6f}, j_flow={item.j_flow:.6f}")
    print("  source flows:")
    for name, item in calc.lf_result.sources.items():
        print(f"    {name}: flow={item.flow:.6f}")
    if calc.lf_result.heat_exchangers:
        print("  heat exchangers:")
        for name, item in calc.lf_result.heat_exchangers.items():
            print(
                f"    {name}: heat={item.primary_heat:.6f}, secondary_heat={item.secondary_heat:.6f}, "
                f"primary={item.primary_in_temperature:.6f}->{item.primary_out_temperature:.6f}, "
                f"secondary={item.secondary_in_temperature:.6f}->{item.secondary_out_temperature:.6f}"
            )
