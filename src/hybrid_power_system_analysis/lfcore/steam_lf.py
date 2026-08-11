"""Steady-state steam-network pressure, mass-flow, and enthalpy calculation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
        self.lf_result.arrays["node_enthalpy"] = self.enthalpy.copy()
        self.lf_result.arrays["node_temperature"] = self.temperature.copy()
        if self.result_mode != "full":
            return
        for pos, node in enumerate(net.nodes):
            result = self.lf_result.nodes[node.name]
            result.enthalpy = float(self.enthalpy[pos])
            result.temperature = float(self.temperature[pos])
        attenuation = self._edge_attenuation()
        ambient = float(net.medium.ambient_enthalpy)
        for edge_pos, edge in enumerate(net.edges):
            collection = (
                self.lf_result.pipes
                if edge.kind == "pipe"
                else self.lf_result.valves
                if edge.kind == "valve"
                else self.lf_result.controllers
            )
            result = collection[edge.name]
            i = int(net.edge_i[edge_pos])
            j = int(net.edge_j[edge_pos])
            factor = float(attenuation[edge_pos])
            if self.edge_flow[edge_pos] >= 0.0:
                i_enthalpy = self.enthalpy[i]
                j_enthalpy = ambient + (self.enthalpy[i] - ambient) * factor
            else:
                j_enthalpy = self.enthalpy[j]
                i_enthalpy = ambient + (self.enthalpy[j] - ambient) * factor
            result.i_enthalpy = float(i_enthalpy)
            result.j_enthalpy = float(j_enthalpy)
            result.i_temperature = float(self._steam_temperature([i_enthalpy])[0])
            result.j_temperature = float(self._steam_temperature([j_enthalpy])[0])
            result.heat_loss = float(abs(self.edge_flow[edge_pos]) * abs(i_enthalpy - j_enthalpy))
        for source_pos, source in enumerate(net.sources):
            result = self.lf_result.sources[source.name]
            result.enthalpy = float(net.source_enthalpy_set[source_pos])
            result.temperature = float(self._steam_temperature([net.source_enthalpy_set[source_pos]])[0])
            result.heat_power = float(
                self.source_flow[source_pos]
                * (net.source_enthalpy_set[source_pos] - net.medium.feedwater_enthalpy)
            )
        for load_pos, load in enumerate(net.loads):
            node_pos = int(net.load_node_pos[load_pos])
            result = self.lf_result.loads[load.name]
            result.enthalpy = float(self.enthalpy[node_pos])
            result.temperature = float(self.temperature[node_pos])
            result.heat_power = float(
                net.load_flow_set[load_pos]
                * (self.enthalpy[node_pos] - net.load_condensate_enthalpy[load_pos])
            )


def print_steam_result(calc: SteamPowerFlowCalc, rc: int) -> None:
    print_fluid_result(calc, rc)
    if calc.result_mode != "full":
        return
    print("  steam node energy states:")
    for name, item in calc.lf_result.nodes.items():
        print(
            f"    {name}: pressure={item.pressure:.6f}, "
            f"enthalpy={item.enthalpy:.6f}, temperature={item.temperature:.6f}"
        )
    print("  steam loads:")
    for name, item in calc.lf_result.loads.items():
        print(f"    {name}: flow={item.flow:.6f}, heat={item.heat_power:.6f}")


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
