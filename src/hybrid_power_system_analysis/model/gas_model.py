"""Gas-network E-file model facade."""

from model.fluid_model import FluidNetwork, build_fluid_network_from_model, load_fluid_network_from_e_file


def build_gas_network_from_model(model, source=None) -> FluidNetwork:
    return build_fluid_network_from_model(
        model,
        prefix="Gas",
        potential_power=2,
        thermal=False,
        source=source,
    )


def load_gas_network_from_e_file(file_name) -> FluidNetwork:
    return load_fluid_network_from_e_file(
        file_name,
        prefix="Gas",
        potential_power=2,
        thermal=False,
    )
