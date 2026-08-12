from pathlib import Path
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from model.meas_model import Measurement


ROOT = Path(__file__).resolve().parents[1]


def _fluid_storage_case(case_type):
    if case_type == "heat":
        from heat_lf import HeatPowerFlowCalc
        from heat_se import HeatStateEstimator
        from model.heat_model import build_heat_network_from_model

        return (
            "Heat",
            "heat_net_3.e",
            2.0,
            build_heat_network_from_model,
            HeatPowerFlowCalc,
            HeatStateEstimator,
        )
    if case_type == "gas":
        from gas_lf import GasPowerFlowCalc
        from gas_se import GasStateEstimator
        from model.gas_model import build_gas_network_from_model

        return (
            "Gas",
            "gas_net_3.e",
            1.0,
            build_gas_network_from_model,
            GasPowerFlowCalc,
            GasStateEstimator,
        )
    if case_type == "hydro":
        from hydro_lf import HydroPowerFlowCalc
        from hydro_se import HydroStateEstimator
        from model.hydro_model import build_hydro_network_from_model

        return (
            "Hydro",
            "hydro_net_3.e",
            0.5,
            build_hydro_network_from_model,
            HydroPowerFlowCalc,
            HydroStateEstimator,
        )

    from model.steam_model import build_steam_network_from_model
    from steam_lf import SteamPowerFlowCalc
    from steam_se import SteamStateEstimator

    return (
        "Steam",
        "steam_net_5.e",
        1.0,
        build_steam_network_from_model,
        SteamPowerFlowCalc,
        SteamStateEstimator,
    )


def _model_with_storage(case_type, storage_flow):
    from efile_read import efile_factory_from_file

    prefix, case_name, *_rest = _fluid_storage_case(case_type)
    model = efile_factory_from_file(
        ROOT / "data" / "model" / case_type / case_name
    )
    storage_name = f"{case_type}_storage"
    setattr(
        model,
        f"{prefix}Storage",
        [
            SimpleNamespace(
                idx=1,
                name=storage_name,
                node=3,
                control_type="FLOW",
                pressure_set=1.0,
                flow_set=float(storage_flow),
                alpha=1.0,
                flow_min=-1.0,
                flow_max=1.0,
                supply_temperature=87.0,
                enthalpy_set=3110.0,
                run_stat=1,
                supply_node=None,
                return_node=None,
            )
        ],
    )
    return model, storage_name


@pytest.mark.parametrize(
    ("case_type", "loader_name", "calc_name", "expected_source", "expected_edges"),
    (
        (
            "gas",
            "load_gas_network_from_e_file",
            "GasPowerFlowCalc",
            1.0,
            np.asarray([1.0, 0.5, 0.8, 0.8]),
        ),
        (
            "hydro",
            "load_hydro_network_from_e_file",
            "HydroPowerFlowCalc",
            0.5,
            np.asarray([0.5, 0.25, 0.4, 0.4]),
        ),
    ),
)
def test_compressible_fluid_load_flow_converges_and_balances(
    case_type,
    loader_name,
    calc_name,
    expected_source,
    expected_edges,
):
    if case_type == "gas":
        from gas_lf import GasPowerFlowCalc
        from model.gas_model import load_gas_network_from_e_file

        loader = load_gas_network_from_e_file
        calc_class = GasPowerFlowCalc
    else:
        from hydro_lf import HydroPowerFlowCalc
        from model.hydro_model import load_hydro_network_from_e_file

        loader = load_hydro_network_from_e_file
        calc_class = HydroPowerFlowCalc

    network = loader(ROOT / "data" / "model" / case_type / f"{case_type}_net_3.e")
    calc = calc_class(network, linear_solver="scipy", tol=1e-10, max_iter=100, result_mode="full")

    assert calc.run() == 0
    assert calc.converged
    assert calc.normF < 1e-10
    np.testing.assert_allclose(calc.edge_flow, expected_edges, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(calc.source_flow, [expected_source], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(np.sum(calc.source_flow), np.sum(network.load_flow_set), atol=1e-10)
    node_balance = np.zeros(len(network.nodes), dtype=np.float64)
    np.add.at(node_balance, network.source_node_pos, calc.source_flow)
    np.add.at(node_balance, network.load_node_pos, -network.load_flow_set)
    node_balance += network.incidence @ calc.edge_flow
    np.testing.assert_allclose(node_balance, 0.0, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("case_type", ("heat", "gas", "hydro", "steam"))
@pytest.mark.parametrize("storage_flow", (0.1, -0.1))
def test_fluid_storage_is_bidirectional_source_in_lf_and_se(case_type, storage_flow):
    from model.fluid_model import PRESSURE_CONTROL
    from model.meas_type import DEVICE_TYPE_CODES
    from scripts.check_fluid_scale_lf_se import build_measurements_from_lf

    (
        prefix,
        _case_name,
        base_source_flow,
        build_network,
        calc_class,
        estimator_class,
    ) = _fluid_storage_case(case_type)
    model, storage_name = _model_with_storage(case_type, storage_flow)
    network = build_network(model)

    assert len(network.storages) == 1
    assert network.storage_source_pos.tolist() == [1]
    assert network.source_is_storage.tolist() == [False, True]
    assert network.storage_pos_by_name == {storage_name: 1}
    assert DEVICE_TYPE_CODES[f"{prefix}Storage"] > 0

    calc = calc_class(
        network,
        tol=1e-10,
        max_iter=100,
        result_mode="full",
    )
    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.source_flow,
        [base_source_flow - storage_flow, storage_flow],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(calc.lf_result.arrays["storage_flow"], [storage_flow])
    assert storage_name in calc.lf_result.storages
    assert storage_name not in calc.lf_result.sources
    assert calc.lf_result.storages[storage_name].flow == pytest.approx(storage_flow)

    node_balance = (
        network.fixed_injection
        - network.demand
        + np.asarray(network.incidence @ calc.edge_flow, dtype=np.float64)
    )
    pressure_sources = np.flatnonzero(network.source_control_type == PRESSURE_CONTROL)
    for source_pos in pressure_sources.tolist():
        flow = float(calc.source_flow[source_pos])
        if network.thermal and network.source_explicit_return[source_pos]:
            node_balance[network.source_return_node_pos[source_pos]] -= flow
            node_balance[network.source_supply_node_pos[source_pos]] += flow
        else:
            node_balance[network.source_node_pos[source_pos]] += flow
    np.testing.assert_allclose(node_balance, 0.0, rtol=0.0, atol=1e-9)

    measurements = build_measurements_from_lf(network, calc)
    storage_measurements = [
        measurement
        for measurement in measurements
        if measurement.device_type == f"{prefix}Storage"
    ]
    expected_types = (
        {"FLOW", "T_SUPPLY", "T_RETURN", "HEAT"}
        if case_type == "heat"
        else {"FLOW"}
    )
    assert {item.meas_type for item in storage_measurements} == expected_types
    flow_measurement = next(
        item for item in storage_measurements if item.meas_type == "FLOW"
    )
    assert flow_measurement.value == pytest.approx(storage_flow)

    estimator = estimator_class(
        network,
        measurements=measurements,
        flat_start=True,
        tol=1e-8,
        max_iter=100,
    )
    assert estimator.run() == 0
    assert estimator.result.converged
    assert not any(
        measurement.device_type == f"{prefix}Storage"
        for measurement, _reason in estimator.prefiltered_measurements
    )


def test_heat_storage_e_file_models_charge_discharge_and_heat_measurements():
    from heat_lf import HeatPowerFlowCalc
    from heat_se import HeatStateEstimator
    from model.heat_model import load_heat_network_from_e_file
    from scripts.check_fluid_scale_lf_se import build_measurements_from_lf

    case_file = ROOT / "data" / "model" / "heat" / "heat_storage.e"
    network = load_heat_network_from_e_file(case_file)

    assert [item.name for item in network.storages] == [
        "heat_storage_discharge",
        "heat_storage_charge",
    ]
    assert network.source_is_storage.tolist() == [False, True, True]
    np.testing.assert_allclose(network.source_flow_set, [0.0, 0.2, -0.1])
    np.testing.assert_allclose(
        network.source_return_temperature_set,
        [50.0, 55.0, 60.0],
    )

    calc = HeatPowerFlowCalc(
        network,
        tol=1.0e-10,
        max_iter=100,
        result_mode="full",
    )
    assert calc.run() == 0
    np.testing.assert_allclose(calc.source_flow, [0.9, 0.2, -0.1], atol=1.0e-9)
    np.testing.assert_allclose(calc.lf_result.arrays["storage_flow"], [0.2, -0.1])
    assert calc.lf_result.storages["heat_storage_discharge"].heat_power > 0.0
    assert calc.lf_result.storages["heat_storage_charge"].heat_power < 0.0

    return_pipe = network.edge_pos_by_name["heat_return_pipe"]
    upstream_node = int(network.edge_i[return_pipe])
    storage_return_node = int(network.source_return_node_pos[2])
    expected_return_temperature = (
        calc.temperature[upstream_node] + 0.1 * 60.0
    ) / 1.1
    assert calc.temperature[storage_return_node] == pytest.approx(
        expected_return_temperature,
        abs=1.0e-9,
    )

    measurements = build_measurements_from_lf(network, calc)
    for storage_name in ("heat_storage_discharge", "heat_storage_charge"):
        storage_types = {
            item.meas_type
            for item in measurements
            if item.device_type == "HeatStorage" and item.device_name == storage_name
        }
        assert storage_types == {"FLOW", "T_SUPPLY", "T_RETURN", "HEAT"}

    measurement_file = ROOT / "data" / "meas" / "heat" / "heat_storage.meas"
    stored_measurements = list(Measurement.read_from_file(measurement_file))
    assert [
        (item.device_type, item.device_name, item.meas_type)
        for item in stored_measurements
    ] == [
        (item.device_type, item.device_name, item.meas_type)
        for item in measurements
    ]
    np.testing.assert_allclose(
        [item.value for item in stored_measurements],
        [item.value for item in measurements],
        atol=1.0e-12,
    )

    estimator = HeatStateEstimator(
        load_heat_network_from_e_file(case_file),
        measurement_file,
        flat_start=True,
        tol=1.0e-9,
        max_iter=100,
    )
    assert estimator.run() == 0
    assert estimator.result.converged
    assert estimator.result.observability.observable
    assert estimator.result.residual_inf < 1.0e-8
    assert not any(
        item.device_type == "HeatStorage"
        for item, _reason in estimator.prefiltered_measurements
    )
    heat_rows = np.asarray(
        [
            row
            for row, item in enumerate(estimator.active_measurements)
            if item.device_type == "HeatStorage" and item.meas_type == "HEAT"
        ],
        dtype=np.int64,
    )
    analytic = estimator.jacobian_sparse(estimator.result.x).toarray()[heat_rows]
    numeric = np.zeros_like(analytic)
    for column in range(estimator.state_count):
        step = 1.0e-6 * max(1.0, abs(float(estimator.result.x[column])))
        upper = estimator.result.x.copy()
        lower = estimator.result.x.copy()
        upper[column] += step
        lower[column] -= step
        numeric[:, column] = (
            estimator.evaluate(upper)[heat_rows]
            - estimator.evaluate(lower)[heat_rows]
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2.0e-6, atol=1.0e-7)


def test_heat_load_flow_solves_hydraulics_supply_and_return_temperature():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    network = load_heat_network_from_e_file(ROOT / "data" / "model" / "heat" / "heat_net_3.e")
    calc = HeatPowerFlowCalc(network, tol=1e-10, max_iter=100, result_mode="full")

    assert calc.run() == 0
    np.testing.assert_allclose(calc.pressure, [10.0, 6.0, 8.0, 5.75, 5.26], atol=1e-9)
    np.testing.assert_allclose(calc.edge_flow, [2.0, 0.7, 1.5, 1.5], atol=1e-9)
    np.testing.assert_allclose(calc.source_flow, [2.0], atol=1e-9)
    assert np.all(np.isfinite(calc.supply_temperature))
    assert np.all(np.isfinite(calc.return_temperature))
    assert calc.supply_temperature[0] > calc.supply_temperature[-1]
    assert sum(item.heat_power for item in calc.lf_result.loads.values()) == pytest.approx(110.0)


def test_heat_exchanger_couples_temperature_without_merging_hydraulic_islands():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / "heat_exchanger.e"
    network = load_heat_network_from_e_file(case_file)
    calc = HeatPowerFlowCalc(network, tol=1e-12, max_iter=50, result_mode="full")

    assert calc.run() == 0
    assert network.island_count == 2
    np.testing.assert_allclose(calc.edge_flow, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(calc.source_flow, [1.0, 0.0], atol=1e-12)
    exchanger = calc.lf_result.heat_exchangers["main_heat_exchanger"]
    assert exchanger.primary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.secondary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.primary_in_temperature > exchanger.primary_out_temperature
    assert exchanger.secondary_out_temperature > exchanger.secondary_in_temperature
    assert exchanger.primary_out_temperature == pytest.approx(calc.return_temperature[1])
    assert exchanger.secondary_out_temperature == pytest.approx(calc.supply_temperature[2])


def test_fixed_heat_exchanger_uses_secondary_temperature_anchor():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / "heat_exchanger.e"
    network = load_heat_network_from_e_file(case_file)
    network.heat_exchangers[0].control_type = "HEAT"
    network.heat_exchangers[0].heat_set = 20.0
    network.loads[0].flow_set = 1.2
    calc = HeatPowerFlowCalc(network, tol=1e-10, max_iter=50, result_mode="full")

    assert calc.run() == 0
    np.testing.assert_allclose(calc.source_flow, [1.0, 0.2], atol=1e-9)
    exchanger = calc.lf_result.heat_exchangers["main_heat_exchanger"]
    assert exchanger.primary_heat == pytest.approx(20.0)
    assert exchanger.secondary_heat == pytest.approx(20.0)
    assert exchanger.primary_out_temperature == pytest.approx(
        exchanger.primary_in_temperature - 20.0 / network.medium.heat_capacity
    )


@pytest.mark.parametrize(
    ("case_type", "case_name"),
    (("heat", "heat_net_3.e"), ("steam", "steam_net_5.e")),
)
def test_joint_fluid_lf_jacobian_matches_finite_difference(case_type, case_name):
    if case_type == "heat":
        from heat_lf import HeatPowerFlowCalc as PowerFlowCalc
        from model.heat_model import load_heat_network_from_e_file as load_network
    else:
        from model.steam_model import load_steam_network_from_e_file as load_network
        from steam_lf import SteamPowerFlowCalc as PowerFlowCalc

    case_file = ROOT / "data" / "model" / case_type / case_name
    calc = PowerFlowCalc(
        load_network(case_file),
        tol=1e-12,
        max_iter=100,
        result_mode="array",
    )
    assert calc.run() == 0

    state = calc.x.copy()
    analytic = calc.get_jacobi(state).toarray()
    numeric = np.zeros_like(analytic)
    for column in range(state.size):
        step = 1e-6 * max(1.0, abs(float(state[column])))
        upper = state.copy()
        lower = state.copy()
        upper[column] += step
        lower[column] -= step
        numeric[:, column] = (calc.get_f(upper) - calc.get_f(lower)) / (2.0 * step)

    cross = analytic[calc.hydraulic_state_count :, : calc.hydraulic_state_count]
    assert np.max(np.abs(cross)) > 0.0
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    ("case_type", "case_name"),
    (
        ("heat", "heat_net_3.e"),
        ("gas", "gas_net_3.e"),
        ("hydro", "hydro_net_3.e"),
        ("steam", "steam_net_5.e"),
    ),
)
def test_fluid_residual_only_path_skips_sparse_jacobian_build(
    monkeypatch,
    case_type,
    case_name,
):
    if case_type == "heat":
        from heat_lf import HeatPowerFlowCalc as PowerFlowCalc
        from model.heat_model import load_heat_network_from_e_file as load_network
    elif case_type == "gas":
        from gas_lf import GasPowerFlowCalc as PowerFlowCalc
        from model.gas_model import load_gas_network_from_e_file as load_network
    elif case_type == "hydro":
        from hydro_lf import HydroPowerFlowCalc as PowerFlowCalc
        from model.hydro_model import load_hydro_network_from_e_file as load_network
    else:
        from model.steam_model import load_steam_network_from_e_file as load_network
        from steam_lf import SteamPowerFlowCalc as PowerFlowCalc

    calc = PowerFlowCalc(
        load_network(ROOT / "data" / "model" / case_type / case_name),
        result_mode="array",
    ).prepare()
    state = calc.x.copy()
    expected, jacobian = calc._build_newton_system(state, return_jacobian=True)
    assert jacobian is not None

    fluid_module = importlib.import_module(calc.__class__.__mro__[1].__module__)

    def fail_sparse_build(*_args, **_kwargs):
        raise AssertionError("residual-only evaluation must not build sparse Jacobian data")

    monkeypatch.setattr(fluid_module, "coo_matrix", fail_sparse_build)
    monkeypatch.setattr(fluid_module, "bmat", fail_sparse_build)
    monkeypatch.setattr(calc, "_edge_flow_jacobian", fail_sparse_build)

    actual, residual_jacobian = calc._build_newton_system(
        state,
        return_jacobian=False,
    )

    assert residual_jacobian is None
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_explicit_return_heat_network_solves_dual_port_devices_and_four_port_exchanger():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / "heat_explicit_return.e"
    network = load_heat_network_from_e_file(case_file)
    calc = HeatPowerFlowCalc(network, tol=1e-12, max_iter=50, result_mode="full")

    assert network.explicit_return
    assert not network.mixed_return
    assert network.island_count == 4
    assert network.thermal_circuit_count == 2
    assert np.all(network.node_explicit_return)
    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, 1.0, atol=1e-12)
    np.testing.assert_allclose(calc.source_flow, [1.0], atol=1e-12)
    np.testing.assert_allclose(
        calc.pressure,
        [10.0, 9.0, 5.0, 4.0, 8.0, 7.0, 3.0, 2.0],
        atol=1e-12,
    )
    node_balance = network.fixed_injection - network.demand + network.incidence @ calc.edge_flow
    np.add.at(node_balance, network.source_return_node_pos, -calc.source_flow)
    np.add.at(node_balance, network.source_supply_node_pos, calc.source_flow)
    np.testing.assert_allclose(node_balance, 0.0, atol=1e-12)
    exchanger = calc.lf_result.heat_exchangers["explicit_heat_exchanger"]
    assert exchanger.primary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.secondary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.primary_out_temperature == pytest.approx(calc.temperature[2])
    assert exchanger.secondary_out_temperature == pytest.approx(calc.temperature[4])


def test_three_port_heat_exchanger_couples_explicit_and_implicit_return_networks():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / "heat_three_port_exchanger.e"
    network = load_heat_network_from_e_file(case_file)
    calc = HeatPowerFlowCalc(network, tol=1e-12, max_iter=50, result_mode="full")

    assert network.explicit_return
    assert network.mixed_return
    np.testing.assert_array_equal(
        network.node_explicit_return,
        [True, True, True, True, False, False],
    )
    np.testing.assert_array_equal(network.exchanger_primary_explicit, [True])
    np.testing.assert_array_equal(network.exchanger_secondary_explicit, [False])
    assert network.temperature_state_count == 8
    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, 1.0, atol=1e-12)
    np.testing.assert_allclose(calc.source_flow, [1.0], atol=1e-12)
    exchanger = calc.lf_result.heat_exchangers["three_port_heat_exchanger"]
    assert exchanger.primary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.secondary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.primary_out_temperature == pytest.approx(calc.temperature[2])
    assert exchanger.secondary_out_temperature == pytest.approx(calc.supply_temperature[4])


def test_reverse_three_port_heat_exchanger_supports_implicit_primary_side():
    from heat_lf import HeatPowerFlowCalc
    from model.heat_model import load_heat_network_from_e_file

    case_file = (
        ROOT
        / "data"
        / "model"
        / "heat"
        / "heat_three_port_exchanger_reverse.e"
    )
    network = load_heat_network_from_e_file(case_file)
    calc = HeatPowerFlowCalc(network, tol=1e-12, max_iter=50, result_mode="full")

    np.testing.assert_array_equal(
        network.node_explicit_return,
        [False, True, True, True, True],
    )
    np.testing.assert_array_equal(network.exchanger_primary_explicit, [False])
    np.testing.assert_array_equal(network.exchanger_secondary_explicit, [True])
    assert network.temperature_state_count == 6
    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, 1.0, atol=1e-12)
    np.testing.assert_allclose(calc.source_flow, [1.0], atol=1e-12)
    exchanger = calc.lf_result.heat_exchangers["reverse_three_port_heat_exchanger"]
    assert exchanger.primary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.secondary_heat == pytest.approx(50.0, abs=1e-10)
    assert exchanger.primary_out_temperature == pytest.approx(calc.return_temperature[0])
    assert exchanger.secondary_out_temperature == pytest.approx(calc.temperature[2])


def test_heat_exchanger_measurements_converge_from_flat_start():
    from heat_se import HeatStateEstimator
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / "heat_exchanger.e"
    meas_file = ROOT / "data" / "meas" / "heat" / "heat_exchanger.meas"
    estimator = HeatStateEstimator(
        load_heat_network_from_e_file(case_file),
        meas_file,
        flat_start=True,
        tol=1e-10,
        max_iter=20,
    )

    assert estimator.run() == 0
    assert estimator.result.observability.observable
    assert estimator.result.residual_inf < 1e-10
    assert not estimator.bad_data
    exchanger_rows = [
        pos
        for pos, measurement in enumerate(estimator.result.measurements)
        if measurement.device_type == "HeatExchanger"
    ]
    assert len(exchanger_rows) == 7
    np.testing.assert_allclose(estimator.result.residual[exchanger_rows], 0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("stem", "expected_states"),
    (
        ("heat_explicit_return", 16),
        ("heat_three_port_exchanger", 14),
        ("heat_three_port_exchanger_reverse", 11),
    ),
)
def test_explicit_and_three_port_heat_state_estimation_recovers_load_flow(
    stem,
    expected_states,
):
    from heat_lf import HeatPowerFlowCalc
    from heat_se import HeatStateEstimator
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / f"{stem}.e"
    meas_file = ROOT / "data" / "meas" / "heat" / f"{stem}.meas"
    lf = HeatPowerFlowCalc(
        load_heat_network_from_e_file(case_file),
        tol=1e-12,
        max_iter=50,
        result_mode="array",
    )
    assert lf.run() == 0
    estimator = HeatStateEstimator(
        load_heat_network_from_e_file(case_file),
        meas_file,
        flat_start=True,
        tol=1e-10,
        max_iter=30,
    )

    assert estimator.run() == 0
    assert estimator.state_count == expected_states
    assert estimator.result.observability.rank == expected_states
    assert estimator.result.residual_inf < 1e-10
    assert not estimator.bad_data
    np.testing.assert_allclose(
        [node.pressure for node in estimator.network.nodes],
        lf.pressure,
        atol=1e-9,
    )
    estimated_temperature = estimator.result.x[
        estimator.base_temperature : estimator.base_enthalpy
    ]
    np.testing.assert_allclose(estimated_temperature, lf.heat_temperature_state, atol=1e-9)


@pytest.mark.parametrize(
    "stem",
    (
        "heat_explicit_return",
        "heat_three_port_exchanger",
        "heat_three_port_exchanger_reverse",
    ),
)
def test_explicit_heat_measurement_jacobian_matches_finite_difference(stem):
    from heat_lf import HeatPowerFlowCalc
    from heat_se import HeatStateEstimator
    from model.heat_model import load_heat_network_from_e_file

    case_file = ROOT / "data" / "model" / "heat" / f"{stem}.e"
    meas_file = ROOT / "data" / "meas" / "heat" / f"{stem}.meas"
    lf = HeatPowerFlowCalc(
        load_heat_network_from_e_file(case_file),
        tol=1e-12,
        max_iter=50,
        result_mode="array",
    )
    assert lf.run() == 0
    estimator = HeatStateEstimator(
        load_heat_network_from_e_file(case_file),
        meas_file,
        flat_start=True,
    )
    estimator.prepare()
    x = estimator.initial_state(flat_start=False)
    x[: estimator.n_potential] = lf.pressure
    x[estimator.base_temperature : estimator.base_enthalpy] = lf.heat_temperature_state
    analytic = estimator.jacobian_sparse(x).toarray()
    numeric = np.zeros_like(analytic)
    for col in range(estimator.state_count):
        step = 1e-6 * max(1.0, abs(float(x[col])))
        plus = x.copy()
        minus = x.copy()
        plus[col] += step
        minus[col] -= step
        numeric[:, col] = (
            estimator.evaluate(plus) - estimator.evaluate(minus)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("case_type", ("gas", "hydro", "heat"))
def test_fluid_state_estimation_flat_start_recovers_load_flow_state(case_type):
    if case_type == "gas":
        from gas_lf import GasPowerFlowCalc
        from gas_se import GasStateEstimator
        from model.gas_model import load_gas_network_from_e_file

        loader = load_gas_network_from_e_file
        calc_class = GasPowerFlowCalc
        estimator_class = GasStateEstimator
    elif case_type == "hydro":
        from hydro_lf import HydroPowerFlowCalc
        from hydro_se import HydroStateEstimator
        from model.hydro_model import load_hydro_network_from_e_file

        loader = load_hydro_network_from_e_file
        calc_class = HydroPowerFlowCalc
        estimator_class = HydroStateEstimator
    else:
        from heat_lf import HeatPowerFlowCalc
        from heat_se import HeatStateEstimator
        from model.heat_model import load_heat_network_from_e_file

        loader = load_heat_network_from_e_file
        calc_class = HeatPowerFlowCalc
        estimator_class = HeatStateEstimator

    case_file = ROOT / "data" / "model" / case_type / f"{case_type}_net_3.e"
    meas_file = ROOT / "data" / "meas" / case_type / f"{case_type}_net_3.meas"
    lf = calc_class(loader(case_file), tol=1e-12, max_iter=100, result_mode="array")
    assert lf.run() == 0
    estimator = estimator_class(loader(case_file), meas_file, flat_start=True, tol=1e-8, max_iter=30)

    assert estimator.run() == 0
    assert estimator.result.converged
    assert estimator.result.observability.observable
    assert estimator.result.residual_inf < 1e-7
    assert not estimator.bad_data
    estimated_pressure = np.asarray([node.pressure for node in estimator.network.nodes])
    np.testing.assert_allclose(estimated_pressure, lf.pressure, rtol=0.0, atol=1e-7)
    if case_type == "heat":
        np.testing.assert_allclose(
            [node.supply_temperature for node in estimator.network.nodes],
            lf.supply_temperature,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            [node.return_temperature for node in estimator.network.nodes],
            lf.return_temperature,
            rtol=0.0,
            atol=1e-7,
        )


def test_gas_observability_adds_only_missing_regulated_flow_pseudo_measurement():
    from gas_se import GasStateEstimator
    from model.gas_model import load_gas_network_from_e_file

    case_file = ROOT / "data" / "model" / "gas" / "gas_net_3.e"
    network = load_gas_network_from_e_file(case_file)
    pressure_values = [5.0, 4.898979485566356, 4.996959075277683, 4.932504434868761, 4.907096901427564]
    measurements = [
        Measurement(
            idx=pos + 1,
            name=f"gas_pressure_{pos + 1}",
            device_type="GasNode",
            device_name=f"gas_n{pos + 1}",
            meas_type="PRESSURE",
            weight=10000.0,
            valid=True,
            value=value,
        )
        for pos, value in enumerate(pressure_values)
    ]
    estimator = GasStateEstimator(network, measurements=measurements, flat_start=True, max_iter=30)
    estimator.prepare()

    before = estimator.analyze_observability(add_pseudo=False)
    after = estimator.analyze_observability(add_pseudo=True)

    assert before.deficiency == 1
    assert after.observable
    pseudo = [item for item in estimator.active_measurements if item.status == 2]
    assert len(pseudo) == 1
    assert pseudo[0].device_type == "GasCompressor"
    assert pseudo[0].meas_type == "FLOW_FROM"


def test_steam_load_flow_balances_mass_and_enthalpy():
    from model.steam_model import load_steam_network_from_e_file
    from steam_lf import SteamPowerFlowCalc

    case_file = ROOT / "data" / "model" / "steam" / "steam_net_5.e"
    network = load_steam_network_from_e_file(case_file)
    calc = SteamPowerFlowCalc(network, tol=1e-12, max_iter=100, result_mode="full")

    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, [1.0, 0.5, 0.8, 0.8], atol=1e-9)
    np.testing.assert_allclose(calc.source_flow, [1.0], atol=1e-9)
    assert np.all(np.diff(calc.enthalpy) < 0.0)
    source_heat = sum(item.heat_power for item in calc.lf_result.sources.values())
    load_heat = sum(item.heat_power for item in calc.lf_result.loads.values())
    network_loss = sum(
        item.heat_loss
        for collection in (
            calc.lf_result.pipes,
            calc.lf_result.valves,
            calc.lf_result.controllers,
        )
        for item in collection.values()
    )
    assert source_heat == pytest.approx(load_heat + network_loss, abs=1e-9)


def test_steam_state_estimation_flat_start_recovers_pressure_and_enthalpy():
    from model.steam_model import load_steam_network_from_e_file
    from steam_lf import SteamPowerFlowCalc
    from steam_se import SteamStateEstimator

    case_file = ROOT / "data" / "model" / "steam" / "steam_net_5.e"
    meas_file = ROOT / "data" / "meas" / "steam" / "steam_net_5.meas"
    lf = SteamPowerFlowCalc(
        load_steam_network_from_e_file(case_file),
        tol=1e-12,
        max_iter=100,
        result_mode="array",
    )
    assert lf.run() == 0
    estimator = SteamStateEstimator(
        load_steam_network_from_e_file(case_file),
        meas_file,
        flat_start=True,
        tol=1e-8,
        max_iter=30,
    )

    assert estimator.run() == 0
    assert estimator.result.observability.observable
    assert estimator.result.residual_inf < 1e-7
    assert not estimator.bad_data
    np.testing.assert_allclose(
        [node.pressure for node in estimator.network.nodes],
        lf.pressure,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        [node.enthalpy for node in estimator.network.nodes],
        lf.enthalpy,
        atol=1e-7,
    )


def test_steam_observability_adds_missing_node_enthalpy_pseudos():
    from model.steam_model import load_steam_network_from_e_file
    from steam_se import SteamStateEstimator

    case_file = ROOT / "data" / "model" / "steam" / "steam_net_5.e"
    pressure = [5.0, 4.898979485566356, 4.409081537009721, 4.33589667773576, 4.306971093471606]
    measurements = [
        Measurement(
            idx=pos + 1,
            name=f"steam_pressure_{pos + 1}",
            device_type="SteamNode",
            device_name=f"steam_n{pos + 1}",
            meas_type="PRESSURE",
            weight=10.0,
            valid=True,
            value=value,
        )
        for pos, value in enumerate(pressure)
    ]
    measurements.append(
        Measurement(
            idx=6,
            name="steam_reducer_flow",
            device_type="SteamPressureReducer",
            device_name="steam_reducer_23",
            meas_type="FLOW_FROM",
            weight=10.0,
            valid=True,
            value=0.8,
        )
    )
    estimator = SteamStateEstimator(
        load_steam_network_from_e_file(case_file),
        measurements=measurements,
        flat_start=True,
    )
    estimator.prepare()

    before = estimator.analyze_observability(add_pseudo=False)
    after = estimator.analyze_observability(add_pseudo=True)

    assert before.deficiency == 5
    assert after.observable
    pseudo = [item for item in estimator.active_measurements if item.status == 2]
    assert len(pseudo) == 5
    assert {item.device_type for item in pseudo} == {"SteamNode"}
    assert {item.meas_type for item in pseudo} == {"ENTHALPY"}


def test_steam_terminal_enthalpy_jacobian_matches_finite_difference():
    from model.steam_model import load_steam_network_from_e_file
    from steam_lf import SteamPowerFlowCalc
    from steam_se import SteamStateEstimator

    case_file = ROOT / "data" / "model" / "steam" / "steam_net_5.e"
    meas_file = ROOT / "data" / "meas" / "steam" / "steam_net_5.meas"
    lf = SteamPowerFlowCalc(
        load_steam_network_from_e_file(case_file),
        tol=1e-12,
        max_iter=100,
        result_mode="array",
    )
    assert lf.run() == 0
    estimator = SteamStateEstimator(load_steam_network_from_e_file(case_file), meas_file, flat_start=True)
    estimator.prepare()
    x = estimator.initial_state()
    x[: estimator.n_potential] = lf.pressure**2
    x[estimator.base_regulated_flow : estimator.base_supply_temperature] = lf.edge_flow[
        estimator.network.regulated_edge_pos
    ]
    x[estimator.base_enthalpy :] = lf.enthalpy
    analytic = estimator.jacobian_sparse(x).toarray()
    target_rows = [
        pos
        for pos, measurement in enumerate(estimator.active_measurements)
        if measurement.meas_type in {"H_TO", "T_TO"}
    ]
    numeric = np.zeros((len(target_rows), estimator.state_count), dtype=np.float64)
    for col in range(estimator.state_count):
        step = 1e-6 * max(1.0, abs(float(x[col])))
        plus = x.copy()
        minus = x.copy()
        plus[col] += step
        minus[col] -= step
        numeric[:, col] = (
            estimator.evaluate(plus)[target_rows] - estimator.evaluate(minus)[target_rows]
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic[target_rows], numeric, rtol=2e-5, atol=2e-7)
