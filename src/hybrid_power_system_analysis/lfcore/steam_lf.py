"""Steady-state steam-network pressure, mass-flow, and enthalpy calculation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lfcore.fluid_lf import FluidLFResult, FluidPowerFlowCalc, print_fluid_result
from model.steam_model import load_steam_network_from_e_file
from paths import model_file


DEFAULT_CASE = model_file("steam", "steam_net_5.e")


class SteamLFResult(FluidLFResult):
    pass


class SteamPowerFlowCalc(FluidPowerFlowCalc):
    result_class = SteamLFResult

    def _write_back(self) -> None:
        super()._write_back()
        net = self.network
        attenuation = self._edge_attenuation()
        ambient = float(net.medium.ambient_enthalpy)
        feedwater = float(net.medium.feedwater_enthalpy)
        edge_i_enthalpy = np.empty(len(net.edges), dtype=np.float64)
        edge_j_enthalpy = np.empty(len(net.edges), dtype=np.float64)
        for edge_pos, edge in enumerate(net.edges):
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            factor = float(attenuation[edge_pos])
            if self.edge_flow[edge_pos] >= 0.0:
                i_enthalpy = self.enthalpy[i]
                j_enthalpy = ambient + (self.enthalpy[i] - ambient) * factor
            else:
                j_enthalpy = self.enthalpy[j]
                i_enthalpy = ambient + (self.enthalpy[j] - ambient) * factor
            edge_i_enthalpy[edge_pos] = i_enthalpy
            edge_j_enthalpy[edge_pos] = j_enthalpy
        edge_i_temperature = self._steam_temperature(edge_i_enthalpy)
        edge_j_temperature = self._steam_temperature(edge_j_enthalpy)
        edge_i_heat_power = self.edge_flow * (edge_i_enthalpy - feedwater)
        edge_j_heat_power = -self.edge_flow * (edge_j_enthalpy - feedwater)
        edge_heat_loss = edge_i_heat_power + edge_j_heat_power
        source_enthalpy = net.source_enthalpy_set.copy()
        source_temperature = self._steam_temperature(source_enthalpy)
        source_heat_power = self.source_flow * (source_enthalpy - feedwater)
        load_enthalpy = self.enthalpy[net.load_node_pos].copy()
        load_temperature = self.temperature[net.load_node_pos].copy()
        load_heat_power = net.load_flow_set * (
            load_enthalpy - net.load_condensate_enthalpy
        )
        self.lf_result.arrays.update(
            node_enthalpy=self.enthalpy.copy(),
            node_temperature=self.temperature.copy(),
            edge_i_enthalpy=edge_i_enthalpy,
            edge_j_enthalpy=edge_j_enthalpy,
            edge_i_temperature=edge_i_temperature,
            edge_j_temperature=edge_j_temperature,
            edge_i_heat_power=edge_i_heat_power,
            edge_j_heat_power=edge_j_heat_power,
            edge_heat_loss=edge_heat_loss,
            source_enthalpy=source_enthalpy,
            source_temperature=source_temperature,
            source_heat_power=source_heat_power,
            storage_enthalpy=source_enthalpy[net.storage_source_pos].copy(),
            storage_temperature=source_temperature[net.storage_source_pos].copy(),
            storage_heat_power=source_heat_power[net.storage_source_pos].copy(),
            load_enthalpy=load_enthalpy,
            load_temperature=load_temperature,
            load_heat_power=load_heat_power,
        )
        if self.result_mode != "full":
            return
        for pos, node in enumerate(net.nodes):
            result = self.lf_result.nodes[node.name]
            result.enthalpy = float(self.enthalpy[pos])
            result.temperature = float(self.temperature[pos])
        for edge_pos, edge in enumerate(net.edges):
            collection = (
                self.lf_result.pipes
                if edge.kind == "pipe"
                else self.lf_result.valves
                if edge.kind == "valve"
                else self.lf_result.controllers
            )
            result = collection[edge.name]
            i_enthalpy = edge_i_enthalpy[edge_pos]
            j_enthalpy = edge_j_enthalpy[edge_pos]
            result.i_enthalpy = float(i_enthalpy)
            result.j_enthalpy = float(j_enthalpy)
            result.i_temperature = float(edge_i_temperature[edge_pos])
            result.j_temperature = float(edge_j_temperature[edge_pos])
            result.i_heat_power = float(edge_i_heat_power[edge_pos])
            result.j_heat_power = float(edge_j_heat_power[edge_pos])
            result.heat_loss = float(edge_heat_loss[edge_pos])
        for source_pos, source in enumerate(net.sources):
            collection = (
                self.lf_result.storages
                if bool(net.source_is_storage[source_pos])
                else self.lf_result.sources
            )
            result = collection[source.name]
            result.enthalpy = float(source_enthalpy[source_pos])
            result.temperature = float(source_temperature[source_pos])
            result.heat_power = float(source_heat_power[source_pos])
        for load_pos, load in enumerate(net.loads):
            result = self.lf_result.loads[load.name]
            result.enthalpy = float(load_enthalpy[load_pos])
            result.temperature = float(load_temperature[load_pos])
            result.heat_power = float(load_heat_power[load_pos])


def print_steam_result(calc: SteamPowerFlowCalc, rc: int) -> None:
    print_fluid_result(calc, rc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Steady-state steam-network load flow")
    parser.add_argument("case", nargs="?", default=str(DEFAULT_CASE))
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--result-mode", default="full", choices=("full", "summary", "array", "none"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    calc = SteamPowerFlowCalc(
        load_steam_network_from_e_file(args.case),
        tol=args.tol,
        max_iter=args.max_iter,
        result_mode=args.result_mode,
        verbose=args.verbose,
    )
    rc = calc.run()
    print_steam_result(calc, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
