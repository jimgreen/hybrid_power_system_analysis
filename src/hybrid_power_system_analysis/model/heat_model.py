"""District-heating E-file model facade."""

from model.fluid_model import FluidNetwork, build_fluid_network_from_model, load_fluid_network_from_e_file


def build_heat_network_from_model(model, source=None) -> FluidNetwork:
    return build_fluid_network_from_model(
        model,
        prefix="Heat",
        potential_power=1,
        thermal=True,
        source=source,
    )


def load_heat_network_from_e_file(file_name) -> FluidNetwork:
    return load_fluid_network_from_e_file(
        file_name,
        prefix="Heat",
        potential_power=1,
        thermal=True,
    )
