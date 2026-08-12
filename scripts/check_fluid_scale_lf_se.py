"""Generate and benchmark scalable heat, gas, hydrogen, and steam LF/SE cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (SRC, SRC / "model", SRC / "lfcore", SRC / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gas_lf import GasPowerFlowCalc
from gas_se import GasStateEstimator
from heat_lf import HeatPowerFlowCalc
from heat_se import HeatStateEstimator
from hydro_lf import HydroPowerFlowCalc
from hydro_se import HydroStateEstimator
from model.gas_model import load_gas_network_from_e_file
from model.heat_model import load_heat_network_from_e_file
from model.hydro_model import load_hydro_network_from_e_file
from model.meas_model import Measurement
from model.steam_model import load_steam_network_from_e_file
from steam_lf import SteamPowerFlowCalc
from steam_se import SteamStateEstimator


DEFAULT_SIZES = (10, 50, 100, 500, 1000, 2000, 5000)
DEFAULT_NETWORKS = ("heat", "gas", "hydro", "steam")
NETWORK_RUNTIME = {
    "heat": (
        load_heat_network_from_e_file,
        HeatPowerFlowCalc,
        HeatStateEstimator,
    ),
    "gas": (
        load_gas_network_from_e_file,
        GasPowerFlowCalc,
        GasStateEstimator,
    ),
    "hydro": (
        load_hydro_network_from_e_file,
        HydroPowerFlowCalc,
        HydroStateEstimator,
    ),
    "steam": (
        load_steam_network_from_e_file,
        SteamPowerFlowCalc,
        SteamStateEstimator,
    ),
}


def _block(name: str, header: str, rows: Iterable[str]) -> str:
    body = "\n".join(f"# {row}" for row in rows)
    return f"<{name}>\n@ {header}\n{body}\n</{name}>\n"


def _tree_depth(node_idx: int) -> int:
    return int(math.floor(math.log2(max(int(node_idx), 1))))


def _generic_edge_kind(child_idx: int) -> str:
    if child_idx == 2 or child_idx % 41 == 0:
        return "controller"
    if child_idx == 3 or child_idx % 17 == 0:
        return "valve"
    return "pipe"


def _generate_compressible_case(network_type: str, node_count: int) -> str:
    if network_type == "gas":
        prefix = "Gas"
        medium_header = "density compressibility molar_mass temperature flow_factor"
        medium_row = "0.8 1.0 0.018 288.15 1.0"
        controller_suffix = "Compressor"
    elif network_type == "hydro":
        prefix = "Hydro"
        medium_header = "density compressibility molar_mass temperature flow_factor"
        medium_row = "0.08375 1.0 0.002016 288.15 1.0"
        controller_suffix = "Compressor"
    elif network_type == "steam":
        prefix = "Steam"
        medium_header = (
            "density compressibility molar_mass temperature heat_capacity "
            "ambient_enthalpy reference_temperature reference_enthalpy "
            "feedwater_enthalpy flow_factor"
        )
        medium_row = "4.0 1.0 0.018 473.15 2.08 419.0 100.0 2676.0 419.0 1.0"
        controller_suffix = "PressureReducer"
    else:
        raise ValueError(f"unsupported compressible network: {network_type}")

    node_rows = []
    for idx in range(1, node_count + 1):
        pressure = 10.0 - 0.015 * _tree_depth(idx)
        if network_type == "steam":
            node_rows.append(
                f"{idx} {network_type}_n{idx} {pressure:.12g} 3000.0 255.769230769 1"
            )
        else:
            node_rows.append(f"{idx} {network_type}_n{idx} {pressure:.12g} 1")

    leaves = list(range(node_count // 2 + 1, node_count + 1))
    load_flow = 1.0 / len(leaves)
    load_rows = []
    for load_idx, node_idx in enumerate(leaves, start=1):
        if network_type == "steam":
            load_rows.append(
                f"{load_idx} steam_load_{load_idx} {node_idx} "
                f"{load_flow:.17g} 419.0 1"
            )
        else:
            load_rows.append(
                f"{load_idx} {network_type}_load_{load_idx} {node_idx} "
                f"{load_flow:.17g} 1"
            )

    pipe_rows = []
    valve_rows = []
    controller_rows = []
    pipe_idx = valve_idx = controller_idx = 0
    for child in range(2, node_count + 1):
        parent = child // 2
        kind = _generic_edge_kind(child)
        heat_loss = 1e-6 * (1 + child % 3)
        if kind == "controller":
            controller_idx += 1
            if network_type == "steam":
                controller_rows.append(
                    f"{controller_idx} steam_reducer_{parent}_{child} {parent} {child} "
                    f"RATIO 1.0 0.0 {heat_loss:.17g} 1"
                )
            else:
                controller_rows.append(
                    f"{controller_idx} {network_type}_compressor_{parent}_{child} "
                    f"{parent} {child} RATIO 1.0 0.0 1"
                )
        elif kind == "valve":
            valve_idx += 1
            if network_type == "steam":
                valve_rows.append(
                    f"{valve_idx} steam_valve_{parent}_{child} {parent} {child} "
                    f"OPEN 1.0 0.0 {heat_loss:.17g} 1"
                )
            else:
                valve_rows.append(
                    f"{valve_idx} {network_type}_valve_{parent}_{child} "
                    f"{parent} {child} OPEN 1.0 0.0 1"
                )
        else:
            pipe_idx += 1
            if network_type == "steam":
                pipe_rows.append(
                    f"{pipe_idx} steam_pipe_{parent}_{child} {parent} {child} "
                    f"1.0 {heat_loss:.17g} 1"
                )
            else:
                pipe_rows.append(
                    f"{pipe_idx} {network_type}_pipe_{parent}_{child} "
                    f"{parent} {child} 1.0 1"
                )

    blocks = [_block(f"{prefix}Medium", medium_header, [medium_row])]
    if network_type == "steam":
        blocks.append(
            _block(
                "SteamNode",
                "idx name pressure enthalpy temperature run_stat",
                node_rows,
            )
        )
        blocks.append(
            _block(
                "SteamSource",
                (
                    "idx name node control_type pressure_set flow_set alpha "
                    "flow_min flow_max enthalpy_set run_stat"
                ),
                ["1 steam_source 1 PRESSURE 10.0 0.0 1.0 0.0 2.0 3000.0 1"],
            )
        )
        blocks.append(
            _block(
                "SteamLoad",
                "idx name node flow_set condensate_enthalpy run_stat",
                load_rows,
            )
        )
        blocks.append(
            _block(
                "SteamPipe",
                "idx name i_node j_node conductance heat_loss run_stat",
                pipe_rows,
            )
        )
        blocks.append(
            _block(
                "SteamValve",
                (
                    "idx name i_node j_node control_type conductance flow_set "
                    "heat_loss run_stat"
                ),
                valve_rows,
            )
        )
        blocks.append(
            _block(
                "SteamPressureReducer",
                (
                    "idx name i_node j_node control_type ratio flow_set "
                    "heat_loss run_stat"
                ),
                controller_rows,
            )
        )
    else:
        blocks.append(
            _block(f"{prefix}Node", "idx name pressure run_stat", node_rows)
        )
        blocks.append(
            _block(
                f"{prefix}Source",
                (
                    "idx name node control_type pressure_set flow_set alpha "
                    "flow_min flow_max run_stat"
                ),
                [
                    f"1 {network_type}_source 1 PRESSURE 10.0 0.0 "
                    "1.0 0.0 2.0 1"
                ],
            )
        )
        blocks.append(
            _block(
                f"{prefix}Load",
                "idx name node flow_set run_stat",
                load_rows,
            )
        )
        blocks.append(
            _block(
                f"{prefix}Pipe",
                "idx name i_node j_node conductance run_stat",
                pipe_rows,
            )
        )
        blocks.append(
            _block(
                f"{prefix}Valve",
                "idx name i_node j_node control_type conductance flow_set run_stat",
                valve_rows,
            )
        )
        blocks.append(
            _block(
                f"{prefix}{controller_suffix}",
                "idx name i_node j_node control_type ratio flow_set run_stat",
                controller_rows,
            )
        )
    return "\n".join(blocks)


def _heat_edge_kind(edge_idx: int) -> str:
    if edge_idx == 1 or edge_idx % 41 == 0:
        return "pump"
    if edge_idx == 2 or edge_idx % 17 == 0:
        return "valve"
    return "pipe"


def _generate_heat_case(node_count: int) -> str:
    supply_count = max(2, node_count // 4)
    return_count = max(2, node_count // 4)
    implicit_count = node_count - supply_count - return_count
    if implicit_count < 2:
        raise ValueError("heat benchmark requires at least two implicit-return nodes")

    supply_nodes = list(range(1, supply_count + 1))
    return_nodes = list(range(supply_count + 1, supply_count + return_count + 1))
    implicit_nodes = list(range(supply_count + return_count + 1, node_count + 1))
    node_rows = []
    for local, node_idx in enumerate(supply_nodes, start=1):
        pressure = 10.0 - 0.01 * _tree_depth(local)
        node_rows.append(
            f"{node_idx} heat_supply_n{local} {pressure:.12g} 90.0 90.0 70.0 1"
        )
    for local, node_idx in enumerate(return_nodes, start=1):
        pressure = 5.0 + 0.01 * _tree_depth(local)
        node_rows.append(
            f"{node_idx} heat_return_n{local} {pressure:.12g} 70.0 90.0 70.0 1"
        )
    for local, node_idx in enumerate(implicit_nodes, start=1):
        pressure = 8.0 - 0.01 * _tree_depth(local)
        node_rows.append(
            f"{node_idx} heat_secondary_n{local} {pressure:.12g} 82.0 82.0 62.0 1"
        )

    pipe_rows = []
    valve_rows = []
    pump_rows = []
    edge_idx = 0
    pipe_idx = valve_idx = pump_idx = 0

    def add_component_edges(nodes: Sequence[int], reverse: bool = False) -> None:
        nonlocal edge_idx, pipe_idx, valve_idx, pump_idx
        for local_child in range(2, len(nodes) + 1):
            local_parent = local_child // 2
            parent = nodes[local_parent - 1]
            child = nodes[local_child - 1]
            i_node, j_node = (child, parent) if reverse else (parent, child)
            edge_idx += 1
            kind = _heat_edge_kind(edge_idx)
            heat_loss = 5e-7 * (1 + edge_idx % 3)
            if kind == "pump":
                pump_idx += 1
                pump_rows.append(
                    f"{pump_idx} heat_pump_{i_node}_{j_node} {i_node} {j_node} "
                    f"GAIN 0.0 0.0 {heat_loss:.17g} 1"
                )
            elif kind == "valve":
                valve_idx += 1
                valve_rows.append(
                    f"{valve_idx} heat_valve_{i_node}_{j_node} {i_node} {j_node} "
                    f"OPEN 10.0 0.0 {heat_loss:.17g} 1"
                )
            else:
                pipe_idx += 1
                pipe_rows.append(
                    f"{pipe_idx} heat_pipe_{i_node}_{j_node} {i_node} {j_node} "
                    f"10.0 {heat_loss:.17g} 1"
                )

    add_component_edges(supply_nodes)
    add_component_edges(return_nodes, reverse=True)
    add_component_edges(implicit_nodes)

    primary_supply_leaves = supply_nodes[len(supply_nodes) // 2 :]
    primary_return_leaves = return_nodes[len(return_nodes) // 2 :]
    primary_flow = 0.25 / len(primary_supply_leaves)
    load_rows = []
    for load_idx, (supply_node, return_node) in enumerate(
        zip(primary_supply_leaves, primary_return_leaves), start=1
    ):
        load_rows.append(
            f"{load_idx} primary_explicit_load_{load_idx} - {supply_node} "
            f"{return_node} {primary_flow:.17g} "
            f"{primary_flow * 4.186 * 8.0:.17g} 1"
        )

    secondary_leaves = implicit_nodes[len(implicit_nodes) // 2 :]
    secondary_flow = 1.0 / len(secondary_leaves)
    secondary_start = len(load_rows) + 1
    for load_idx, node_idx in enumerate(secondary_leaves, start=secondary_start):
        load_rows.append(
            f"{load_idx} secondary_load_{load_idx - secondary_start + 1} {node_idx} - - "
            f"{secondary_flow:.17g} {secondary_flow * 4.186 * 10.0:.17g} 1"
        )

    blocks = [
        _block(
            "HeatMedium",
            "density heat_capacity ambient_temperature temperature flow_factor",
            ["998.0 4.186 20.0 353.15 1.0"],
        ),
        _block(
            "HeatNode",
            (
                "idx name pressure temperature supply_temperature "
                "return_temperature run_stat"
            ),
            node_rows,
        ),
        _block(
            "HeatSource",
            (
                "idx name node supply_node return_node control_type pressure_set "
                "flow_set alpha flow_min flow_max supply_temperature_set run_stat"
            ),
            [
                f"1 primary_source - {supply_nodes[0]} {return_nodes[0]} PRESSURE "
                "10.0 0.0 1.0 0.0 3.0 90.0 1"
            ],
        ),
        _block(
            "HeatLoad",
            (
                "idx name node supply_node return_node mass_flow heat_power run_stat"
            ),
            load_rows,
        ),
        _block(
            "HeatPipe",
            "idx name i_node j_node conductance heat_loss run_stat",
            pipe_rows,
        ),
        _block(
            "HeatValve",
            (
                "idx name i_node j_node control_type conductance flow_set "
                "heat_loss run_stat"
            ),
            valve_rows,
        ),
        _block(
            "HeatPump",
            (
                "idx name i_node j_node control_type pressure_gain flow_set "
                "heat_loss run_stat"
            ),
            pump_rows,
        ),
        _block(
            "HeatExchanger",
            (
                "idx name i_node j_node primary_supply_node primary_return_node "
                "secondary_return_node secondary_supply_node control_type "
                "primary_flow secondary_flow heat_set effectiveness heat_loss run_stat"
            ),
            [
                f"1 three_port_exchanger - {implicit_nodes[0]} {supply_nodes[-1]} "
                f"{return_nodes[-1]} - - EFFECTIVENESS 1.0 1.0 0.0 0.8 0.02 1"
            ],
        ),
    ]
    return "\n".join(blocks)


def generate_case(network_type: str, node_count: int, destination: Path) -> Path:
    if int(node_count) < 10:
        raise ValueError("fluid scale benchmark requires at least 10 nodes")
    if network_type == "heat":
        text = _generate_heat_case(int(node_count))
    else:
        text = _generate_compressible_case(network_type, int(node_count))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


def _controller_device_type(network) -> str:
    if network.thermal:
        return "HeatPump"
    if network.steam:
        return "SteamPressureReducer"
    return f"{network.prefix}Compressor"


def _edge_device_type(network, edge_pos: int) -> str:
    kind = str(network.edge_kind[edge_pos])
    if kind == "pipe":
        return f"{network.prefix}Pipe"
    if kind == "valve":
        return f"{network.prefix}Valve"
    return _controller_device_type(network)


def build_measurements_from_lf(network, calc) -> list[Measurement]:
    measurements: list[Measurement] = []
    result_arrays = getattr(getattr(calc, "lf_result", None), "arrays", {})
    calculated_load_flow = np.asarray(
        result_arrays.get("load_flow", network.load_flow_set),
        dtype=np.float64,
    )
    if calculated_load_flow.shape != network.load_flow_set.shape:
        calculated_load_flow = network.load_flow_set

    def add(device_type: str, device_name: str, meas_type: str, value: float) -> None:
        idx = len(measurements) + 1
        measurements.append(
            Measurement(
                idx,
                f"scale_meas_{idx}",
                device_type,
                device_name,
                meas_type,
                10.0,
                True,
                float(value),
            )
        )

    for pos, name in enumerate(network.node_name.tolist()):
        add(f"{network.prefix}Node", str(name), "PRESSURE", calc.pressure[pos])
        if network.thermal:
            if network.node_explicit_return[pos]:
                add("HeatNode", str(name), "TEMPERATURE", calc.temperature[pos])
            else:
                add("HeatNode", str(name), "T_SUPPLY", calc.supply_temperature[pos])
                add("HeatNode", str(name), "T_RETURN", calc.return_temperature[pos])
        elif network.steam:
            add("SteamNode", str(name), "ENTHALPY", calc.enthalpy[pos])

    for edge_pos, name in enumerate(network.edge_name.tolist()):
        add(
            _edge_device_type(network, edge_pos),
            str(name),
            "FLOW_FROM",
            calc.edge_flow[edge_pos],
        )
    source_t_out = np.asarray(
        result_arrays.get(
            "source_t_out",
            np.full(len(network.sources), np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    source_supply_temperature = np.asarray(
        result_arrays.get(
            "source_supply_temperature",
            np.full(len(network.sources), np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    source_return_temperature = np.asarray(
        result_arrays.get(
            "source_return_temperature",
            np.full(len(network.sources), np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    for source_pos, name in enumerate(network.source_name.tolist()):
        source_device_type = (
            f"{network.prefix}Storage"
            if bool(network.source_is_storage[source_pos])
            else f"{network.prefix}Source"
        )
        add(
            source_device_type,
            str(name),
            "FLOW",
            calc.source_flow[source_pos],
        )
        if network.thermal:
            t_supply = (
                float(source_t_out[source_pos])
                if np.isfinite(source_t_out[source_pos])
                else float(source_supply_temperature[source_pos])
            )
            if bool(network.source_is_storage[source_pos]):
                t_return = float(source_return_temperature[source_pos])
                add(source_device_type, str(name), "T_SUPPLY", t_supply)
                add(source_device_type, str(name), "T_RETURN", t_return)
                add(
                    source_device_type,
                    str(name),
                    "HEAT",
                    calc.source_flow[source_pos]
                    * float(network.medium.heat_capacity)
                    * (t_supply - t_return),
                )
            elif np.isfinite(source_t_out[source_pos]):
                add(
                    source_device_type,
                    str(name),
                    "T_SUPPLY",
                    t_supply,
                )
    for load_pos, name in enumerate(network.load_name.tolist()):
        add(
            f"{network.prefix}Load",
            str(name),
            "FLOW",
            calculated_load_flow[load_pos],
        )
    if network.thermal and network.exchanger_i.size:
        primary_heat, _, primary_out, secondary_out = calc._heat_exchanger_quantities()
        for pos, name in enumerate(network.exchanger_name.tolist()):
            primary_supply = int(network.exchanger_primary_supply[pos])
            secondary_return = int(network.exchanger_secondary_return[pos])
            add("HeatExchanger", str(name), "FLOW_FROM", network.exchanger_primary_flow[pos])
            add("HeatExchanger", str(name), "FLOW_TO", network.exchanger_secondary_flow[pos])
            add("HeatExchanger", str(name), "TS_FROM", calc.supply_temperature[primary_supply])
            add("HeatExchanger", str(name), "TR_FROM", primary_out[pos])
            add("HeatExchanger", str(name), "TR_TO", calc.return_temperature[secondary_return])
            add("HeatExchanger", str(name), "TS_TO", secondary_out[pos])
            add("HeatExchanger", str(name), "HEAT", primary_heat[pos])
    return measurements


def write_measurement_file(measurements: Sequence[Measurement], destination: Path) -> Path:
    lines = [
        "<Measurement>",
        "@ idx name dev_type dev_name meas_type weight valid value",
    ]
    lines.extend(
        (
            f"# {item.idx} {item.name} {item.device_type} {item.device_name} "
            f"{item.meas_type} {item.weight:.17g} {int(item.valid)} {item.value:.17g}"
        )
        for item in measurements
    )
    lines.append("</Measurement>")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return destination


def _maximum(values) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _state_accuracy(estimator, result, lf_calc) -> dict[str, float]:
    network = estimator.network
    x = np.asarray(result.x, dtype=np.float64)
    estimated_pressure = np.maximum(x[: estimator.n_potential], 1e-12) ** (
        1.0 / network.potential_power
    )
    _, estimated_edge_flow, flow_derivative = estimator._flow_state(x)
    estimated_source_flow, _ = estimator._source_flow_and_derivative(
        x, estimated_edge_flow, flow_derivative
    )
    accuracy = {
        "pressure_max_abs_error": _maximum(estimated_pressure - lf_calc.pressure),
        "edge_flow_max_abs_error": _maximum(estimated_edge_flow - lf_calc.edge_flow),
        "source_flow_max_abs_error": _maximum(estimated_source_flow - lf_calc.source_flow),
    }
    if network.thermal:
        estimated_temperature = x[estimator.base_temperature : estimator.base_enthalpy]
        accuracy["temperature_max_abs_error"] = _maximum(
            estimated_temperature - lf_calc.heat_temperature_state
        )
    elif network.steam:
        accuracy["enthalpy_max_abs_error"] = _maximum(
            x[estimator.base_enthalpy :] - lf_calc.enthalpy
        )
    return accuracy


def benchmark_case(
    network_type: str,
    node_count: int,
    case_file: Path,
    measurement_file: Path | None = None,
) -> dict[str, object]:
    loader, lf_class, estimator_class = NETWORK_RUNTIME[network_type]
    internal_start = time.perf_counter()
    phases: dict[str, float] = {}

    stage = time.perf_counter()
    lf_network = loader(case_file)
    phases["lf_network_load_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    lf_calc = lf_class(
        lf_network,
        tol=1e-9,
        max_iter=100,
        result_mode="array",
        verbose=False,
    )
    phases["lf_construct_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    lf_calc.prepare()
    phases["lf_prepare_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    lf_rc = lf_calc.run()
    phases["lf_run_s"] = time.perf_counter() - stage
    if lf_rc != 0:
        raise RuntimeError(
            f"{network_type}-{node_count} LF did not converge: residual={lf_calc.normF}"
        )

    stage = time.perf_counter()
    measurements = build_measurements_from_lf(lf_network, lf_calc)
    phases["measurement_build_s"] = time.perf_counter() - stage
    measurement_file = (
        case_file.with_suffix(".meas")
        if measurement_file is None
        else measurement_file
    )
    measurement_file.parent.mkdir(parents=True, exist_ok=True)
    stage = time.perf_counter()
    write_measurement_file(measurements, measurement_file)
    phases["measurement_write_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    se_network = loader(case_file)
    phases["se_network_load_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    estimator = estimator_class(
        se_network,
        measurement_file,
        flat_start=True,
        tol=1e-8,
        max_iter=30,
        verbose=False,
    )
    phases["se_construct_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    estimator.prepare()
    phases["se_prepare_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    observability = estimator.analyze_observability(add_pseudo=True)
    phases["se_observability_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    result = estimator.estimate(observability=observability)
    phases["se_estimate_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    estimator.identify_bad_data(result)
    estimator.build_se_result(result)
    phases["se_diagnostics_s"] = time.perf_counter() - stage
    if not result.converged:
        raise RuntimeError(
            f"{network_type}-{node_count} SE did not converge: "
            f"rank={observability.rank}/{observability.state_count}, "
            f"residual={result.residual_inf}"
        )

    accuracy = _state_accuracy(estimator, result, lf_calc)
    return {
        "network": network_type,
        "nodes": int(node_count),
        "case_file": str(case_file),
        "measurement_file": str(measurement_file),
        "lf": {
            "converged": bool(lf_calc.converged),
            "iterations": int(lf_calc.iterations),
            "residual_inf": float(lf_calc.normF),
            "edges": int(len(lf_network.edges)),
            "sources": int(len(lf_network.sources)),
            "loads": int(len(lf_network.loads)),
            "valves": int(np.count_nonzero(lf_network.edge_kind == "valve")),
            "controllers": int(
                np.count_nonzero(
                    ~np.isin(lf_network.edge_kind, ("pipe", "valve"))
                )
            ),
            "heat_exchangers": int(len(lf_network.heat_exchangers)),
        },
        "se": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "states": int(observability.state_count),
            "rank": int(observability.rank),
            "measurements": int(observability.measurement_count),
            "pseudo_measurements": int(
                estimator.se_result.statistics.pseudo_measurement_count
            ),
            "bad_data": int(len(estimator.bad_data)),
            "residual_inf": float(result.residual_inf),
            "objective": float(result.objective),
        },
        "accuracy": accuracy,
        "phases": phases,
        "internal_total_s": time.perf_counter() - internal_start,
    }


def _worker(
    network_type: str,
    node_count: int,
    case_file: Path,
    measurement_file: Path,
) -> int:
    result = benchmark_case(
        network_type,
        node_count,
        case_file,
        measurement_file,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def _run_worker(
    network_type: str,
    node_count: int,
    case_file: Path,
    measurement_file: Path,
) -> dict[str, object]:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--network",
        network_type,
        "--size",
        str(node_count),
        "--case",
        str(case_file),
        "--measurement-file",
        str(measurement_file),
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    process_wall = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"{network_type}-{node_count} worker failed with exit code "
            f"{completed.returncode}:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{network_type}-{node_count} worker returned no output")
    result = json.loads(lines[-1])
    result["process_wall_s"] = process_wall
    result["startup_import_overhead_s"] = max(
        0.0, process_wall - float(result["internal_total_s"])
    )
    return result


def _flatten_result(result: dict[str, object]) -> dict[str, object]:
    flat = {
        "network": result["network"],
        "nodes": result["nodes"],
        "process_wall_s": result["process_wall_s"],
        "internal_total_s": result["internal_total_s"],
        "startup_import_overhead_s": result["startup_import_overhead_s"],
    }
    for section in ("lf", "se", "accuracy", "phases"):
        for key, value in result[section].items():
            flat[f"{section}_{key}"] = value
    return flat


def write_reports(results: Sequence[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(list(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    rows = [_flatten_result(result) for result in results]
    fieldnames = list(rows[0])
    known_fields = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known_fields:
                fieldnames.append(key)
                known_fields.add(key)
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fluid Network Scale LF/SE Benchmark",
        "",
        (
            "| Network | Nodes | LF iter | LF residual | SE states | SE iter | "
            "SE residual | Max state error | LF run (s) | SE obs (s) | "
            "SE estimate (s) | Process wall (s) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        accuracy = result["accuracy"]
        max_error = max(float(value) for value in accuracy.values())
        lines.append(
            f"| {result['network']} | {result['nodes']} | "
            f"{result['lf']['iterations']} | {result['lf']['residual_inf']:.3e} | "
            f"{result['se']['states']} | {result['se']['iterations']} | "
            f"{result['se']['residual_inf']:.3e} | {max_error:.3e} | "
            f"{result['phases']['lf_run_s']:.6f} | "
            f"{result['phases']['se_observability_s']:.6f} | "
            f"{result['phases']['se_estimate_s']:.6f} | "
            f"{result['process_wall_s']:.6f} |"
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and benchmark scalable heat/gas/hydrogen/steam LF and SE cases"
    )
    parser.add_argument("--networks", nargs="+", choices=DEFAULT_NETWORKS, default=DEFAULT_NETWORKS)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "fluid_scale_benchmark",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "data" / "model",
    )
    parser.add_argument(
        "--measurement-root",
        type=Path,
        default=ROOT / "data" / "meas",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--network", choices=DEFAULT_NETWORKS, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--measurement-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        if (
            args.network is None
            or args.size is None
            or args.case is None
            or args.measurement_file is None
        ):
            parser.error(
                "worker mode requires --network, --size, --case, and --measurement-file"
            )
        return _worker(
            args.network,
            args.size,
            args.case.resolve(),
            args.measurement_file.resolve(),
        )

    output_dir = args.output_dir.resolve()
    model_root = args.model_root.resolve()
    measurement_root = args.measurement_root.resolve()
    cases = []
    for network_type in args.networks:
        for node_count in args.sizes:
            stem = f"{network_type}_scale_{node_count}"
            case_file = model_root / network_type / f"{stem}.e"
            measurement_file = measurement_root / network_type / f"{stem}.meas"
            generate_case(network_type, node_count, case_file)
            cases.append((network_type, node_count, case_file, measurement_file))
    if args.generate_only:
        print(
            f"Generated {len(cases)} models below {model_root}; "
            "measurement files are created by the LF benchmark run"
        )
        return 0

    results = []
    for network_type, node_count, case_file, measurement_file in cases:
        print(f"Running {network_type}-{node_count}...", flush=True)
        result = _run_worker(
            network_type,
            node_count,
            case_file,
            measurement_file,
        )
        results.append(result)
        print(
            f"  LF {result['phases']['lf_run_s']:.6f}s, "
            f"SE {result['phases']['se_estimate_s']:.6f}s, "
            f"max error {max(result['accuracy'].values()):.3e}",
            flush=True,
        )
        write_reports(results, output_dir)
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
