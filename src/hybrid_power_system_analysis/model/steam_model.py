"""Steam-network E-file model facade."""

from model.fluid_model import FluidNetwork, build_fluid_network_from_model, load_fluid_network_from_e_file


def build_steam_network_from_model(model, source=None) -> FluidNetwork:
    return build_fluid_network_from_model(
        model,
        prefix="Steam",
        potential_power=2,
        thermal=False,
        steam=True,
        controller_suffix="PressureReducer",
        source=source,
    )


def load_steam_network_from_e_file(file_name) -> FluidNetwork:
    return load_fluid_network_from_e_file(
        file_name,
        prefix="Steam",
        potential_power=2,
        thermal=False,
        steam=True,
        controller_suffix="PressureReducer",
    )
