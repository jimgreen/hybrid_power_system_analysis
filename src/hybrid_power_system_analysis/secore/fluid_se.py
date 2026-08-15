"""Shared sparse WLS state estimator for heat, gas, and hydrogen networks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import DEFAULT_SE_PARAMETER_FILE, StateEstimationParameters, load_se_parameters
from lfcore.common import allocate_limited_residual
from model.fluid_model import FluidNetwork
from model.meas_model import (
    MEAS_STATUS_PSEUDO,
    BadDataItem,
    EstimateResult,
    Measurement,
    MeasurementList,
    ObservabilityResult,
    measurement_table_from_measurements,
)
from model.meas_type import DEVICE_TYPE_CODES, MEAS_TYPE_CODES
from secore.se_result import SEResult


SUPPORTED_MEASUREMENT_TYPES = frozenset(
    {
        "PRESSURE",
        "PRESSURE_FROM",
        "PRESSURE_TO",
        "FLOW",
        "FLOW_FROM",
        "FLOW_TO",
        "T_SUPPLY",
        "T_RETURN",
        "TS_FROM",
        "TS_TO",
        "TR_FROM",
        "TR_TO",
        "HEAT",
        "ENTHALPY",
        "TEMPERATURE",
        "H_FROM",
        "H_TO",
        "T_FROM",
        "T_TO",
    }
)
_DEVICE_TYPE_ALIASES = {
    "HeatSource2": "HeatSource",
    "HeatLoad2": "HeatLoad",
}


class FluidStateEstimator:
    """Estimate fluid-network pressure/flow and optional thermal states."""

    def __init__(
        self,
        network: FluidNetwork,
        measurement_file=None,
        *,
        measurements: Optional[Sequence[Measurement]] = None,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        flat_start: Optional[bool] = None,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        parameter_file=DEFAULT_SE_PARAMETER_FILE,
        parameters: Optional[StateEstimationParameters] = None,
        verbose: bool = False,
    ):
        if not hasattr(network, "prepare") or not hasattr(network, "potential_power"):
            raise ValueError("FluidStateEstimator requires a FluidNetwork-compatible input")
        self.network = network
        self.measurement_file = None if measurement_file is None else Path(measurement_file)
        self._provided_measurements = measurements
        self.params = (parameters or load_se_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            flat_start=flat_start,
            bad_threshold=bad_threshold,
            max_remove=max_remove,
        )
        self.tol = float(self.params.tol)
        self.max_iter = int(self.params.max_iter)
        self.bad_threshold = float(self.params.bad_threshold)
        self.verbose = bool(verbose)
        self.prepared = False
        self.measurements: MeasurementList = MeasurementList()
        self.active_measurements: MeasurementList = MeasurementList()
        self.prefiltered_measurements: List[Tuple[Measurement, str]] = []
        self.measurement_groups: Dict[Tuple[str, str], np.ndarray] = {}
        self.state_labels: List[str] = []
        self.result: Optional[EstimateResult] = None
        self.bad_data: List[BadDataItem] = []
        self.normalized_residual = np.empty(0, dtype=np.float64)
        self.se_result = SEResult()

    def prepare(self) -> "FluidStateEstimator":
        net = self.network.prepare()
        self.n_potential = len(net.nodes)
        self.base_regulated_flow = self.n_potential
        self.n_regulated = int(net.regulated_edge_pos.size)
        self.base_supply_temperature = self.base_regulated_flow + self.n_regulated
        self.base_temperature = self.base_supply_temperature
        self.base_return_temperature = self.base_temperature + (
            len(net.nodes) if net.thermal else 0
        )
        self.base_enthalpy = self.base_temperature + (
            net.temperature_state_count if net.thermal else 0
        )
        self.state_count = self.base_enthalpy + (len(net.nodes) if net.steam else 0)
        self.regulated_state_by_edge = np.full(len(net.edges), -1, dtype=np.int64)
        if net.regulated_edge_pos.size:
            self.regulated_state_by_edge[net.regulated_edge_pos] = (
                self.base_regulated_flow + np.arange(net.regulated_edge_pos.size, dtype=np.int64)
            )
        self.state_labels = [f"{net.prefix}Node:{name}:PRESSURE" for name in net.node_name.tolist()]
        self.state_labels.extend(
            f"{self._controller_device_type()}:{net.edge_name[pos]}:FLOW"
            for pos in net.regulated_edge_pos.tolist()
        )
        if net.thermal:
            self.state_labels.extend(
                f"HeatNode:{name}:{'TEMPERATURE' if net.node_explicit_return[pos] else 'T_SUPPLY'}"
                for pos, name in enumerate(net.node_name.tolist())
            )
            self.state_labels.extend(
                f"HeatNode:{net.node_name[pos]}:T_RETURN"
                for pos in np.flatnonzero(~net.node_explicit_return).tolist()
            )
        if net.steam:
            self.state_labels.extend(f"SteamNode:{name}:ENTHALPY" for name in net.node_name.tolist())
        self.measurements = self._load_measurements()
        self._install_measurement_runtime()
        self.prepared = True
        return self

    def _load_measurements(self) -> MeasurementList:
        if self._provided_measurements is not None:
            return MeasurementList(list(self._provided_measurements))
        if self.measurement_file is None:
            raise ValueError("measurement_file or measurements is required")
        loaded = Measurement.read_from_file(self.measurement_file)
        return MeasurementList(list(loaded))

    def _device_position(self, measurement: Measurement) -> int:
        net = self.network
        device_type = _DEVICE_TYPE_ALIASES.get(
            str(measurement.device_type),
            str(measurement.device_type),
        )
        name = str(measurement.device_name)
        prefix = net.prefix
        if device_type == f"{prefix}Node":
            return int(net.node_pos_by_name.get(name, -1))
        if device_type in {
            f"{prefix}Pipe",
            f"{prefix}Valve",
            self._controller_device_type(),
        }:
            return int(net.edge_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Source":
            return int(net.source_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Storage":
            return int(net.storage_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Load":
            return int(net.load_pos_by_name.get(name, -1))
        if net.thermal and device_type == "HeatExchanger":
            return int(net.exchanger_pos_by_name.get(name, -1))
        return -1

    def _controller_device_type(self) -> str:
        if self.network.thermal:
            return "HeatPump"
        if self.network.steam:
            return "SteamPressureReducer"
        return f"{self.network.prefix}Compressor"

    def _source_device_types(self) -> frozenset[str]:
        prefix = self.network.prefix
        return frozenset((f"{prefix}Source", f"{prefix}Storage"))

    def _install_measurement_runtime(self) -> None:
        prefix = self.network.prefix
        active = []
        prefiltered = []
        for measurement in self.measurements:
            measurement.device_type = _DEVICE_TYPE_ALIASES.get(
                str(measurement.device_type),
                str(measurement.device_type),
            )
            measurement.meas_type = str(measurement.meas_type).upper()
            measurement.device_type_code = int(DEVICE_TYPE_CODES.get(str(measurement.device_type), 0))
            measurement.meas_type_code = int(MEAS_TYPE_CODES.get(measurement.meas_type, 0))
            measurement.device_pos = self._device_position(measurement)
            reason = ""
            if not str(measurement.device_type).startswith(prefix):
                reason = "foreign network measurement"
            elif measurement.device_pos < 0:
                reason = "unknown device"
            elif measurement.meas_type not in SUPPORTED_MEASUREMENT_TYPES:
                reason = "unsupported measurement type"
            elif not bool(measurement.valid):
                reason = "invalid"
            elif float(measurement.weight) <= 0.0:
                reason = "zero weight"
            if reason:
                measurement.valid = False
                prefiltered.append((measurement, reason))
            else:
                active.append(measurement)
        self.prefiltered_measurements = prefiltered
        self.active_measurements = MeasurementList(active)
        self._rebuild_measurement_groups()

    def _rebuild_measurement_groups(self) -> None:
        groups = defaultdict(list)
        for row, measurement in enumerate(self.active_measurements):
            groups[(str(measurement.device_type), str(measurement.meas_type))].append(row)
        self.measurement_groups = {
            key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()
        }
        self.active_measurements.table = measurement_table_from_measurements(self.active_measurements)

    def initial_state(self, flat_start: Optional[bool] = None) -> np.ndarray:
        if not self.prepared:
            self.prepare()
        net = self.network
        use_flat = self.params.flat_start if flat_start is None else bool(flat_start)
        if use_flat:
            pressure = np.empty(len(net.nodes), dtype=np.float64)
            for island in range(int(net.island_count)):
                island_nodes = np.flatnonzero(net.node_island == island)
                fixed_rows = np.flatnonzero(
                    net.node_island[net.fixed_node_pos] == island
                )
                if fixed_rows.size:
                    pressure_value = float(np.mean(net.fixed_pressure[fixed_rows]))
                else:
                    pressure_value = float(np.mean(net.node_pressure[island_nodes]))
                pressure[island_nodes] = max(pressure_value, 1e-6)
        else:
            pressure = np.maximum(net.node_pressure, 1e-6).copy()
        x = np.empty(self.state_count, dtype=np.float64)
        x[: self.n_potential] = pressure ** int(net.potential_power)
        if self.n_regulated:
            x[self.base_regulated_flow : self.base_supply_temperature] = net.edge_flow_set[
                net.regulated_edge_pos
            ]
        if net.thermal:
            if use_flat:
                source_rows = np.flatnonzero(net.source_supply_temperature > 0.0)
                supply = (
                    float(np.mean(net.source_supply_temperature[source_rows]))
                    if source_rows.size
                    else float(np.mean(net.initial_temperature_state))
                )
                return_value = float(np.mean(net.node_return_temperature))
                temperature = np.full(
                    net.temperature_state_count, supply, dtype=np.float64
                )
                implicit_return_states = np.unique(
                    net.return_temperature_state_by_node[~net.node_explicit_return]
                )
                temperature[implicit_return_states] = return_value
                if net.fixed_temperature_state_pos.size:
                    temperature[
                        net.fixed_temperature_state_pos
                    ] = net.fixed_temperature
                x[self.base_temperature : self.base_enthalpy] = temperature
            else:
                x[self.base_temperature : self.base_enthalpy] = net.initial_temperature_state
        elif net.steam:
            if use_flat:
                source_rows = np.flatnonzero(net.source_enthalpy_set > 0.0)
                enthalpy = (
                    float(np.mean(net.source_enthalpy_set[source_rows]))
                    if source_rows.size
                    else float(np.mean(net.node_enthalpy))
                )
                x[self.base_enthalpy :] = enthalpy
            else:
                x[self.base_enthalpy :] = net.node_enthalpy
        return x

    def _flow_state(self, x: np.ndarray):
        net = self.network
        potential = np.asarray(x[: self.n_potential], dtype=np.float64)
        flow = net.edge_flow_set.copy()
        rows = []
        cols = []
        data = []
        for edge_pos in net.passive_edge_pos.tolist():
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            delta = float(potential[i] - potential[j])
            conductance = float(net.edge_conductance[edge_pos])
            flow[edge_pos] = conductance * np.sign(delta) * np.sqrt(abs(delta))
            derivative = 0.5 * conductance / np.sqrt(max(abs(delta), 1e-10))
            rows.extend((edge_pos, edge_pos))
            cols.extend((i, j))
            data.extend((derivative, -derivative))
        for edge_pos in net.regulated_edge_pos.tolist():
            state_pos = int(self.regulated_state_by_edge[edge_pos])
            flow[edge_pos] = x[state_pos]
            rows.append(edge_pos)
            cols.append(state_pos)
            data.append(1.0)
        derivative = coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(len(net.edges), self.state_count),
        ).tocsr()
        return potential, flow, derivative

    def _source_flow_and_derivative(
        self,
        x: np.ndarray,
        flow: np.ndarray,
        flow_derivative: csr_matrix,
    ):
        net = self.network
        source_flow = net.source_flow_set.copy()
        source_derivatives = [csr_matrix((1, self.state_count), dtype=np.float64) for _ in net.sources]
        node_edge_balance = np.asarray(net.incidence @ flow, dtype=np.float64)
        node_target_derivative = -(net.incidence @ flow_derivative).tocsr()
        for source_positions in net.pressure_source_groups:
            node_pos = int(net.source_supply_node_pos[source_positions[0]])
            target = float(
                net.demand[node_pos]
                - net.fixed_injection[node_pos]
                - node_edge_balance[node_pos]
            )
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
            shares = (allocated_plus - allocated) / step
            target_row = node_target_derivative.getrow(node_pos)
            for local, source_pos in enumerate(source_positions.tolist()):
                source_derivatives[source_pos] = target_row * float(shares[local])
        for node_pos in range(len(net.nodes)):
            positions = np.flatnonzero(net.source_node_pos == node_pos)
            pressure_positions = positions[
                net.source_is_pressure_controlled[positions]
            ]
            if net.thermal and pressure_positions.size:
                pressure_positions = pressure_positions[
                    ~net.source_explicit_return[pressure_positions]
                ]
            if pressure_positions.size == 0:
                continue
            target = float(net.demand[node_pos] - node_edge_balance[node_pos] - net.fixed_injection[node_pos])
            allocated = allocate_limited_residual(
                net.source_flow_set[pressure_positions],
                target,
                lower=net.source_flow_min[pressure_positions],
                upper=net.source_flow_max[pressure_positions],
                alpha=net.source_alpha[pressure_positions],
            )
            source_flow[pressure_positions] = allocated
            step = max(1e-7, abs(target) * 1e-7)
            allocated_plus = allocate_limited_residual(
                net.source_flow_set[pressure_positions],
                target + step,
                lower=net.source_flow_min[pressure_positions],
                upper=net.source_flow_max[pressure_positions],
                alpha=net.source_alpha[pressure_positions],
            )
            shares = (allocated_plus - allocated) / step
            target_row = node_target_derivative.getrow(node_pos)
            for local, source_pos in enumerate(pressure_positions.tolist()):
                source_derivatives[source_pos] = target_row * float(shares[local])
        return source_flow, source_derivatives

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        if not self.prepared:
            self.prepare()
        active = self.active_measurements if measurements is None else measurements
        potential, flow, flow_derivative = self._flow_state(x)
        source_flow, _ = self._source_flow_and_derivative(x, flow, flow_derivative)
        pressure = np.maximum(potential, 1e-12) ** (1.0 / self.network.potential_power)
        if self.network.thermal:
            temperature_state = x[self.base_temperature : self.base_enthalpy]
            supply = temperature_state[
                self.network.supply_temperature_state_by_node
            ]
            return_temperature = temperature_state[
                self.network.return_temperature_state_by_node
            ]
        else:
            supply = None
            return_temperature = None
        exchanger_values = self._heat_exchanger_state(x) if self.network.thermal else None
        enthalpy = x[self.base_enthalpy :] if self.network.steam else None
        steam_temperature = self._steam_temperature(enthalpy) if enthalpy is not None else None
        values = np.empty(len(active), dtype=np.float64)
        net = self.network
        for row, measurement in enumerate(active):
            device_type = str(measurement.device_type)
            meas_type = str(measurement.meas_type)
            pos = int(measurement.device_pos)
            if device_type == f"{net.prefix}Node":
                if meas_type == "PRESSURE":
                    values[row] = pressure[pos]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[pos]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[pos]
                elif meas_type == "TEMPERATURE" and supply is not None:
                    values[row] = supply[pos]
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[pos]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[pos]
                else:
                    values[row] = np.nan
            elif device_type in {
                f"{net.prefix}Pipe",
                f"{net.prefix}Valve",
                self._controller_device_type(),
            }:
                i = int(net.edge_i[pos])
                j = int(net.edge_j[pos])
                edge_values = {
                    "FLOW": flow[pos],
                    "FLOW_FROM": flow[pos],
                    "FLOW_TO": -flow[pos],
                    "PRESSURE_FROM": pressure[i],
                    "PRESSURE_TO": pressure[j],
                }
                if supply is not None:
                    if net.node_explicit_return[i]:
                        factor = float(
                            np.exp(
                                -net.edge_heat_loss[pos]
                                / max(abs(float(flow[pos])), 1e-9)
                            )
                        )
                        ambient = float(net.medium.ambient_temperature)
                        if flow[pos] >= 0.0:
                            t_from = supply[i]
                            t_to = ambient + (supply[i] - ambient) * factor
                        else:
                            t_to = supply[j]
                            t_from = ambient + (supply[j] - ambient) * factor
                        edge_values.update(T_FROM=t_from, T_TO=t_to)
                    else:
                        edge_values.update(
                            TS_FROM=supply[i],
                            TS_TO=supply[j],
                            TR_FROM=return_temperature[i],
                            TR_TO=return_temperature[j],
                        )
                if enthalpy is not None:
                    factor = float(
                        np.exp(-net.edge_heat_loss[pos] / max(abs(float(flow[pos])), 1e-9))
                    )
                    ambient = float(net.medium.ambient_enthalpy)
                    if flow[pos] >= 0.0:
                        h_from = enthalpy[i]
                        h_to = ambient + (enthalpy[i] - ambient) * factor
                    else:
                        h_to = enthalpy[j]
                        h_from = ambient + (enthalpy[j] - ambient) * factor
                    edge_values.update(
                        H_FROM=h_from,
                        H_TO=h_to,
                        T_FROM=self._steam_temperature([h_from])[0],
                        T_TO=self._steam_temperature([h_to])[0],
                    )
                values[row] = edge_values.get(meas_type, np.nan)
            elif device_type in self._source_device_types():
                supply_node = int(net.source_supply_node_pos[pos])
                return_node = int(net.source_return_node_pos[pos])
                if meas_type == "FLOW":
                    values[row] = source_flow[pos]
                elif meas_type == "PRESSURE":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_FROM":
                    values[row] = pressure[return_node]
                elif meas_type == "PRESSURE_TO":
                    values[row] = pressure[supply_node]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[supply_node]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[return_node]
                elif (
                    meas_type == "HEAT"
                    and supply is not None
                    and return_temperature is not None
                ):
                    values[row] = (
                        source_flow[pos]
                        * float(net.medium.heat_capacity)
                        * (supply[supply_node] - return_temperature[return_node])
                    )
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[supply_node]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[supply_node]
                else:
                    values[row] = np.nan
            elif device_type == f"{net.prefix}Load":
                supply_node = int(net.load_supply_node_pos[pos])
                return_node = int(net.load_return_node_pos[pos])
                if meas_type == "FLOW":
                    values[row] = net.load_flow_set[pos]
                elif meas_type == "PRESSURE":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_FROM":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_TO":
                    values[row] = pressure[return_node]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[supply_node]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[return_node]
                elif meas_type == "HEAT" and supply is not None:
                    if net.load_explicit_return[pos]:
                        values[row] = (
                            net.load_flow_set[pos]
                            * net.medium.heat_capacity
                            * (supply[supply_node] - return_temperature[return_node])
                        )
                    else:
                        values[row] = net.load_heat_power[pos]
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[supply_node]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[supply_node]
                elif meas_type == "HEAT" and enthalpy is not None:
                    values[row] = net.load_flow_set[pos] * (
                        enthalpy[supply_node] - net.load_condensate_enthalpy[pos]
                    )
                else:
                    values[row] = np.nan
            elif device_type == "HeatExchanger" and exchanger_values is not None:
                primary_supply = int(net.exchanger_primary_supply[pos])
                primary_return = int(net.exchanger_primary_return[pos])
                secondary_return = int(net.exchanger_secondary_return[pos])
                secondary_supply = int(net.exchanger_secondary_supply[pos])
                primary_heat, secondary_heat, primary_out, secondary_out = exchanger_values
                exchanger_measurements = {
                    "FLOW_FROM": net.exchanger_primary_flow[pos],
                    "FLOW_TO": net.exchanger_secondary_flow[pos],
                    "PRESSURE_FROM": pressure[primary_supply],
                    "PRESSURE_TO": pressure[secondary_return],
                    "TS_FROM": supply[primary_supply],
                    "TR_FROM": primary_out[pos],
                    "TR_TO": return_temperature[secondary_return],
                    "TS_TO": secondary_out[pos],
                    "HEAT": primary_heat[pos],
                }
                values[row] = exchanger_measurements.get(meas_type, np.nan)
            else:
                values[row] = np.nan
        return values

    def _steam_temperature(self, enthalpy) -> np.ndarray:
        net = self.network
        cp = max(float(net.medium.heat_capacity), 1e-12)
        return (
            float(net.medium.reference_temperature)
            + (np.asarray(enthalpy, dtype=np.float64) - float(net.medium.reference_enthalpy)) / cp
        )

    def _heat_exchanger_state(self, x: np.ndarray):
        net = self.network
        temperature_state = x[self.base_temperature : self.base_enthalpy]
        supply = temperature_state[net.supply_temperature_state_by_node]
        return_temperature = temperature_state[net.return_temperature_state_by_node]
        count = int(net.exchanger_i.size)
        primary_heat = np.zeros(count, dtype=np.float64)
        secondary_heat = np.zeros(count, dtype=np.float64)
        primary_out = np.zeros(count, dtype=np.float64)
        secondary_out = np.zeros(count, dtype=np.float64)
        cp = max(float(net.medium.heat_capacity), 1e-12)
        for pos in range(count):
            primary_supply = int(net.exchanger_primary_supply[pos])
            secondary_return = int(net.exchanger_secondary_return[pos])
            primary_mass = max(float(net.exchanger_primary_flow[pos]), 1e-12)
            secondary_mass = max(float(net.exchanger_secondary_flow[pos]), 1e-12)
            if str(net.exchanger_control_type[pos]) == "EFFECTIVENESS":
                primary_heat[pos] = (
                    net.exchanger_effectiveness[pos]
                    * min(primary_mass, secondary_mass)
                    * cp
                    * (supply[primary_supply] - return_temperature[secondary_return])
                )
            else:
                primary_heat[pos] = net.exchanger_heat_set[pos]
            secondary_heat[pos] = primary_heat[pos] * (1.0 - net.exchanger_heat_loss[pos])
            primary_out[pos] = (
                supply[primary_supply] - primary_heat[pos] / (primary_mass * cp)
            )
            secondary_out[pos] = (
                return_temperature[secondary_return]
                + secondary_heat[pos] / (secondary_mass * cp)
            )
        return primary_heat, secondary_heat, primary_out, secondary_out

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> csr_matrix:
        active = self.active_measurements if measurements is None else measurements
        potential, flow, flow_derivative = self._flow_state(x)
        source_flow, source_derivatives = self._source_flow_and_derivative(
            x, flow, flow_derivative
        )
        net = self.network
        rows = []
        cols = []
        data = []

        def append_sparse_row(row_pos: int, sparse_row, scale: float = 1.0) -> None:
            sparse_row = sparse_row.tocsr()
            start, stop = sparse_row.indptr[0], sparse_row.indptr[1]
            count = stop - start
            if count:
                rows.extend([row_pos] * count)
                cols.extend(sparse_row.indices[start:stop].tolist())
                data.extend((sparse_row.data[start:stop] * scale).tolist())

        pressure_derivative = (1.0 / net.potential_power) * np.maximum(potential, 1e-12) ** (
            1.0 / net.potential_power - 1.0
        )
        for row, measurement in enumerate(active):
            device_type = str(measurement.device_type)
            meas_type = str(measurement.meas_type)
            pos = int(measurement.device_pos)
            if device_type == f"{net.prefix}Node":
                if meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(pos)
                    data.append(float(pressure_derivative[pos]))
                elif meas_type == "T_SUPPLY" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "T_RETURN" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.return_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "TEMPERATURE" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "ENTHALPY" and net.steam:
                    rows.append(row)
                    cols.append(self.base_enthalpy + pos)
                    data.append(1.0)
                elif meas_type == "TEMPERATURE" and net.steam:
                    rows.append(row)
                    cols.append(self.base_enthalpy + pos)
                    data.append(1.0 / max(float(net.medium.heat_capacity), 1e-12))
                continue
            if device_type in {
                f"{net.prefix}Pipe",
                f"{net.prefix}Valve",
                self._controller_device_type(),
            }:
                i = int(net.edge_i[pos])
                j = int(net.edge_j[pos])
                if meas_type in {"FLOW", "FLOW_FROM"}:
                    append_sparse_row(row, flow_derivative.getrow(pos))
                elif meas_type == "FLOW_TO":
                    append_sparse_row(row, flow_derivative.getrow(pos), -1.0)
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(i)
                    data.append(float(pressure_derivative[i]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(j)
                    data.append(float(pressure_derivative[j]))
                elif net.thermal and meas_type in {
                    "TS_FROM",
                    "TS_TO",
                    "TR_FROM",
                    "TR_TO",
                    "T_FROM",
                    "T_TO",
                }:
                    if net.node_explicit_return[i]:
                        terminal_from = meas_type.endswith("FROM")
                        flow_value = float(flow[pos])
                        upstream_node = i if flow_value >= 0.0 else j
                        upstream_terminal_is_from = flow_value >= 0.0
                        upstream_col = (
                            self.base_temperature
                            + int(net.supply_temperature_state_by_node[upstream_node])
                        )
                        if terminal_from == upstream_terminal_is_from:
                            rows.append(row)
                            cols.append(upstream_col)
                            data.append(1.0)
                        else:
                            mass = max(abs(flow_value), 1e-9)
                            loss = float(net.edge_heat_loss[pos])
                            factor = float(np.exp(-loss / mass))
                            rows.append(row)
                            cols.append(upstream_col)
                            data.append(factor)
                            if loss > 0.0:
                                temperature_state = x[
                                    self.base_temperature : self.base_enthalpy
                                ]
                                upstream_temperature = temperature_state[
                                    net.supply_temperature_state_by_node[upstream_node]
                                ]
                                da_dq = (
                                    factor
                                    * loss
                                    * np.sign(flow_value or 1.0)
                                    / (mass * mass)
                                )
                                append_sparse_row(
                                    row,
                                    flow_derivative.getrow(pos),
                                    (upstream_temperature - net.medium.ambient_temperature)
                                    * da_dq,
                                )
                    else:
                        node_pos = i if meas_type.endswith("FROM") else j
                        state_map = (
                            net.supply_temperature_state_by_node
                            if meas_type.startswith("TS")
                            else net.return_temperature_state_by_node
                        )
                        rows.append(row)
                        cols.append(
                            self.base_temperature + int(state_map[node_pos])
                        )
                        data.append(1.0)
                elif net.steam and meas_type in {"H_FROM", "H_TO", "T_FROM", "T_TO"}:
                    terminal_from = meas_type.endswith("FROM")
                    flow_value = float(flow[pos])
                    upstream_node = i if flow_value >= 0.0 else j
                    upstream_terminal_is_from = flow_value >= 0.0
                    output_scale = 1.0 if meas_type.startswith("H") else 1.0 / max(
                        float(net.medium.heat_capacity), 1e-12
                    )
                    if terminal_from == upstream_terminal_is_from:
                        rows.append(row)
                        cols.append(self.base_enthalpy + upstream_node)
                        data.append(output_scale)
                    else:
                        mass = max(abs(flow_value), 1e-9)
                        loss = float(net.edge_heat_loss[pos])
                        factor = float(np.exp(-loss / mass))
                        rows.append(row)
                        cols.append(self.base_enthalpy + upstream_node)
                        data.append(factor * output_scale)
                        if loss > 0.0:
                            enthalpy = x[self.base_enthalpy :]
                            da_dq = factor * loss * np.sign(flow_value or 1.0) / (mass * mass)
                            flow_scale = (
                                (float(enthalpy[upstream_node]) - float(net.medium.ambient_enthalpy))
                                * da_dq
                                * output_scale
                            )
                            append_sparse_row(row, flow_derivative.getrow(pos), flow_scale)
                continue
            if device_type in self._source_device_types():
                supply_node = int(net.source_supply_node_pos[pos])
                return_node = int(net.source_return_node_pos[pos])
                if meas_type == "FLOW":
                    append_sparse_row(row, source_derivatives[pos])
                elif meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(return_node)
                    data.append(float(pressure_derivative[return_node]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif net.thermal and meas_type in {"T_SUPPLY", "T_RETURN"}:
                    node_pos = supply_node if meas_type == "T_SUPPLY" else return_node
                    state_map = (
                        net.supply_temperature_state_by_node
                        if meas_type == "T_SUPPLY"
                        else net.return_temperature_state_by_node
                    )
                    rows.append(row)
                    cols.append(self.base_temperature + int(state_map[node_pos]))
                    data.append(1.0)
                elif net.thermal and meas_type == "HEAT":
                    supply_col = (
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[supply_node])
                    )
                    return_col = (
                        self.base_temperature
                        + int(net.return_temperature_state_by_node[return_node])
                    )
                    source_value = float(source_flow[pos])
                    heat_capacity = float(net.medium.heat_capacity)
                    temperature = x[self.base_temperature : self.base_enthalpy]
                    supply_state = int(
                        net.supply_temperature_state_by_node[supply_node]
                    )
                    return_state = int(
                        net.return_temperature_state_by_node[return_node]
                    )
                    temperature_gap = float(
                        temperature[supply_state] - temperature[return_state]
                    )
                    append_sparse_row(
                        row,
                        source_derivatives[pos],
                        heat_capacity * temperature_gap,
                    )
                    rows.extend((row, row))
                    cols.extend((supply_col, return_col))
                    data.extend(
                        (
                            source_value * heat_capacity,
                            -source_value * heat_capacity,
                        )
                    )
                elif net.steam and meas_type in {"ENTHALPY", "TEMPERATURE"}:
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(
                        1.0
                        if meas_type == "ENTHALPY"
                        else 1.0 / max(float(net.medium.heat_capacity), 1e-12)
                    )
                continue
            if device_type == f"{net.prefix}Load":
                supply_node = int(net.load_supply_node_pos[pos])
                return_node = int(net.load_return_node_pos[pos])
                if meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(return_node)
                    data.append(float(pressure_derivative[return_node]))
                elif net.thermal and meas_type in {"T_SUPPLY", "T_RETURN"}:
                    node_pos = supply_node if meas_type == "T_SUPPLY" else return_node
                    state_map = (
                        net.supply_temperature_state_by_node
                        if meas_type == "T_SUPPLY"
                        else net.return_temperature_state_by_node
                    )
                    rows.append(row)
                    cols.append(self.base_temperature + int(state_map[node_pos]))
                    data.append(1.0)
                elif net.thermal and meas_type == "HEAT" and net.load_explicit_return[pos]:
                    scale = float(net.load_flow_set[pos] * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend(
                        (
                            self.base_temperature
                            + int(net.supply_temperature_state_by_node[supply_node]),
                            self.base_temperature
                            + int(net.return_temperature_state_by_node[return_node]),
                        )
                    )
                    data.extend((scale, -scale))
                elif net.steam and meas_type in {"ENTHALPY", "TEMPERATURE"}:
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(
                        1.0
                        if meas_type == "ENTHALPY"
                        else 1.0 / max(float(net.medium.heat_capacity), 1e-12)
                    )
                elif net.steam and meas_type == "HEAT":
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(float(net.load_flow_set[pos]))
                # Fixed HeatLoad heat-power measurements do not add a state derivative.
                continue
            if device_type == "HeatExchanger" and net.thermal:
                primary_supply = int(net.exchanger_primary_supply[pos])
                secondary_return = int(net.exchanger_secondary_return[pos])
                primary_mass = max(float(net.exchanger_primary_flow[pos]), 1e-12)
                secondary_mass = max(float(net.exchanger_secondary_flow[pos]), 1e-12)
                loss_factor = 1.0 - float(net.exchanger_heat_loss[pos])
                if str(net.exchanger_control_type[pos]) == "EFFECTIVENESS":
                    heat_gradient = (
                        net.exchanger_effectiveness[pos]
                        * min(primary_mass, secondary_mass)
                        * net.medium.heat_capacity
                    )
                else:
                    heat_gradient = 0.0
                ts_i = self.base_temperature + int(
                    net.supply_temperature_state_by_node[primary_supply]
                )
                tr_j = self.base_temperature + int(
                    net.return_temperature_state_by_node[secondary_return]
                )
                if meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(primary_supply)
                    data.append(float(pressure_derivative[primary_supply]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(secondary_return)
                    data.append(float(pressure_derivative[secondary_return]))
                elif meas_type == "TS_FROM":
                    rows.append(row)
                    cols.append(ts_i)
                    data.append(1.0)
                elif meas_type == "TR_TO":
                    rows.append(row)
                    cols.append(tr_j)
                    data.append(1.0)
                elif meas_type == "HEAT" and heat_gradient:
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((heat_gradient, -heat_gradient))
                elif meas_type == "TR_FROM":
                    scale = heat_gradient / (primary_mass * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((1.0 - scale, scale))
                elif meas_type == "TS_TO":
                    scale = loss_factor * heat_gradient / (secondary_mass * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((scale, 1.0 - scale))
        return coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(len(active), self.state_count),
        ).tocsr()

    @staticmethod
    def _rank(H: csr_matrix) -> int:
        if H.shape[1] == 0:
            return 0
        if max(H.shape) <= 1000:
            return int(np.linalg.matrix_rank(H.toarray()))
        return int(structural_rank(H))

    def _observability_result(self, H: csr_matrix) -> ObservabilityResult:
        rank = self._rank(H)
        state_count = int(H.shape[1])
        column_norm = np.sqrt(np.asarray(H.power(2).sum(axis=0)).reshape(-1))
        weak_order = np.argsort(column_norm)
        weak = [(int(pos), float(column_norm[pos])) for pos in weak_order[: min(20, state_count)]]
        return ObservabilityResult(
            observable=rank == state_count,
            rank=rank,
            state_count=state_count,
            measurement_count=int(H.shape[0]),
            deficiency=max(0, state_count - rank),
            singular_values=np.asarray([], dtype=np.float64),
            weak_states=weak,
        )

    def _pseudo_candidates(self) -> List[Measurement]:
        net = self.network
        existing = {
            (str(item.device_type), str(item.device_name), str(item.meas_type))
            for item in self.active_measurements
        }
        candidates = []
        next_idx = max((int(item.idx) for item in self.measurements), default=0) + 1

        def add(device_type: str, device_name: str, meas_type: str, value: float, device_pos: int) -> None:
            nonlocal next_idx
            if (device_type, device_name, meas_type) in existing:
                return
            candidates.append(
                Measurement(
                    next_idx,
                    f"pseudo_{device_name}_{meas_type.lower()}",
                    device_type,
                    device_name,
                    meas_type,
                    self.params.pseudo_measurement_weight,
                    True,
                    float(value),
                    status=MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_CODES[device_type],
                    meas_type_code=MEAS_TYPE_CODES[meas_type],
                    device_pos=device_pos,
                )
            )
            next_idx += 1

        for pos, name in enumerate(net.node_name.tolist()):
            add(f"{net.prefix}Node", str(name), "PRESSURE", net.node_pressure[pos], pos)
        for edge_pos in net.regulated_edge_pos.tolist():
            edge = net.edges[edge_pos]
            device_type = self._controller_device_type()
            add(device_type, edge.name, "FLOW_FROM", net.edge_flow_set[edge_pos], edge_pos)
        if net.thermal:
            for pos, name in enumerate(net.node_name.tolist()):
                if net.node_explicit_return[pos]:
                    add(
                        "HeatNode",
                        str(name),
                        "TEMPERATURE",
                        net.node_temperature[pos],
                        pos,
                    )
                else:
                    add(
                        "HeatNode",
                        str(name),
                        "T_SUPPLY",
                        net.node_supply_temperature[pos],
                        pos,
                    )
                    add(
                        "HeatNode",
                        str(name),
                        "T_RETURN",
                        net.node_return_temperature[pos],
                        pos,
                    )
        if net.steam:
            for pos, name in enumerate(net.node_name.tolist()):
                add("SteamNode", str(name), "ENTHALPY", net.node_enthalpy[pos], pos)
        return candidates

    def analyze_observability(self, *, add_pseudo: bool = True) -> ObservabilityResult:
        if not self.prepared:
            self.prepare()
        x = self.initial_state()
        H = self.jacobian_sparse(x)
        result = self._observability_result(H)
        if result.observable or not add_pseudo:
            self.observability = result
            return result
        current_rank = result.rank
        max_add = int(self.params.targeted_pseudo_measurement_max)
        added = 0
        for candidate in self._pseudo_candidates():
            trial = MeasurementList([*self.active_measurements, candidate])
            trial_H = self.jacobian_sparse(x, trial)
            rank = self._rank(trial_H)
            if rank <= current_rank:
                continue
            self.measurements.append(candidate)
            self.active_measurements.append(candidate)
            current_rank = rank
            added += 1
            if current_rank == self.state_count or added >= max_add:
                break
        self._rebuild_measurement_groups()
        H = self.jacobian_sparse(x)
        result = self._observability_result(H)
        self.observability = result
        return result

    def estimate(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        observability: Optional[ObservabilityResult] = None,
    ) -> EstimateResult:
        if not self.prepared:
            self.prepare()
        obs = observability or self.analyze_observability(add_pseudo=True)
        x = self.initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        z = np.asarray([item.value for item in self.active_measurements], dtype=np.float64)
        weight = np.asarray([item.weight for item in self.active_measurements], dtype=np.float64)
        converged = False
        max_correction = np.inf
        objective = np.inf
        iteration = 0
        H = csr_matrix((len(z), self.state_count), dtype=np.float64)
        gain = csr_matrix((self.state_count, self.state_count), dtype=np.float64)
        minimum = 1e-12
        for iteration in range(1, self.max_iter + 1):
            z_est = self.evaluate(x)
            residual = z - z_est
            objective = float(np.dot(weight, residual * residual))
            H = self.jacobian_sparse(x)
            weighted_H = diags(weight) @ H
            gain = (H.T @ weighted_H).tocsc()
            rhs = H.T @ (weight * residual)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    dx = np.asarray(spsolve(gain, rhs, use_umfpack=False), dtype=np.float64)
            except (MatrixRankWarning, RuntimeError, ValueError):
                break
            if not np.all(np.isfinite(dx)):
                break
            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            if self.verbose:
                print(
                    f"Iter {iteration}: objective={objective:.6e}, "
                    f"max_dx={max_correction:.6e}, residual={np.linalg.norm(residual, np.inf):.6e}"
                )
            if max_correction < self.tol:
                converged = True
                break
            accepted = False
            step = 1.0
            for _ in range(20):
                candidate = x + step * dx
                candidate[: self.n_potential] = np.maximum(candidate[: self.n_potential], minimum)
                candidate_residual = z - self.evaluate(candidate)
                candidate_objective = float(np.dot(weight, candidate_residual * candidate_residual))
                if np.isfinite(candidate_objective) and candidate_objective <= objective + 1e-14:
                    x = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        z_est = self.evaluate(x)
        residual = z - z_est
        objective = float(np.dot(weight, residual * residual))
        if max_correction < self.tol:
            converged = True
        H = self.jacobian_sparse(x)
        gain = (H.T @ (diags(weight) @ H)).tocsr()
        result = EstimateResult(
            converged=bool(converged and obs.observable),
            iterations=int(iteration),
            objective=objective,
            max_correction=float(max_correction),
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=x,
            z_est=z_est,
            residual=residual,
            H=H,
            gain=gain,
            measurements=list(self.active_measurements),
            observability=obs,
            measurement_table=measurement_table_from_measurements(self.active_measurements),
        )
        self.result = result
        self._write_back_state(x)
        return result

    def identify_bad_data(
        self,
        result: Optional[EstimateResult] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[BadDataItem], np.ndarray]:
        result = result or self.result
        if result is None:
            raise RuntimeError("state estimation has not run")
        threshold_value = self.bad_threshold if threshold is None else float(threshold)
        weights = np.asarray([item.weight for item in result.measurements], dtype=np.float64)
        normalized = np.sqrt(np.maximum(weights, 0.0)) * np.abs(result.residual)
        bad = []
        for pos in np.flatnonzero(normalized > threshold_value).tolist():
            measurement = result.measurements[pos]
            bad.append(
                BadDataItem(
                    measurement=measurement,
                    residual=float(result.residual[pos]),
                    normalized_residual=float(normalized[pos]),
                    estimated_value=float(result.z_est[pos]),
                    measured_value=float(measurement.value),
                    row_pos=pos,
                )
            )
        self.bad_data = bad
        self.normalized_residual = normalized
        return bad, normalized

    def build_se_result(self, result: Optional[EstimateResult] = None) -> SEResult:
        result = result or self.result
        if result is None:
            raise RuntimeError("state estimation has not run")
        self.se_result = SEResult.from_estimate_result(
            result,
            bad_items=self.bad_data,
            normalized_residual=self.normalized_residual,
            prefiltered_measurements=self.prefiltered_measurements,
            all_measurements=self.measurements,
        )
        return self.se_result

    def run(self) -> int:
        if not self.prepared:
            self.prepare()
        observability = self.analyze_observability(add_pseudo=True)
        result = self.estimate(observability=observability)
        self.identify_bad_data(result)
        self.build_se_result(result)
        return 0 if result.converged else 1

    def _write_back_state(self, x: np.ndarray) -> None:
        net = self.network
        pressure = np.maximum(x[: self.n_potential], 1e-12) ** (1.0 / net.potential_power)
        temperature_state = (
            x[self.base_temperature : self.base_enthalpy] if net.thermal else None
        )
        for pos, node in enumerate(net.nodes):
            node.pressure = float(pressure[pos])
            if net.thermal:
                node.supply_temperature = float(
                    temperature_state[net.supply_temperature_state_by_node[pos]]
                )
                node.return_temperature = float(
                    temperature_state[net.return_temperature_state_by_node[pos]]
                )
                if net.node_explicit_return[pos]:
                    node.temperature = node.supply_temperature
            elif net.steam:
                node.enthalpy = float(x[self.base_enthalpy + pos])
                node.temperature = float(self._steam_temperature([node.enthalpy])[0])


def print_fluid_se_result(estimator: FluidStateEstimator, rc: int) -> None:
    result = estimator.result
    if result is None:
        print(f"{estimator.network.prefix} state estimation did not run")
        return
    obs = result.observability
    print(f"{estimator.network.prefix} state estimation: {'converged' if rc == 0 else 'not converged'}")
    print(
        f"  states={obs.state_count}, measurements={obs.measurement_count}, rank={obs.rank}, "
        f"pseudo={estimator.se_result.statistics.pseudo_measurement_count}"
    )
    print(
        f"  iterations={result.iterations}, objective={result.objective:.6e}, "
        f"residual={result.residual_inf:.6e}, bad_data={len(estimator.bad_data)}"
    )
