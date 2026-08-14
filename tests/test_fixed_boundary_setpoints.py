from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "model"))
sys.path.insert(0, str(SRC_DIR / "lfcore"))
sys.path.insert(0, str(SRC_DIR / "secore"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))


def _empty(width):
    return np.zeros((0, width), dtype=np.float64)


def test_ac_voltage_sanitizer_covers_generators_shunts_and_acac_terminals():
    from ac_array_model import (
        ACAC_COLS,
        ACAC_SIDE_CONTROL_CODE,
        CTRL_PQ,
        CTRL_PV,
        CTRL_SLACK,
        GEN_COLS,
        SHUNT_COLS,
        SHUNT_Q,
        SHUNT_V,
        sanitize_ac_voltage_setpoints,
    )

    gen = _empty(len(GEN_COLS))
    gen = np.vstack((gen, np.zeros((3, len(GEN_COLS)))))
    gen[:, GEN_COLS["idx"]] = [1, 2, 3]
    gen[:, GEN_COLS["control_type"]] = [CTRL_SLACK, CTRL_PV, CTRL_PQ]
    gen[:, GEN_COLS["v_set"]] = [0.0, 2.0, 0.0]
    gen[:, GEN_COLS["run_stat"]] = 1.0

    shunt = np.zeros((2, len(SHUNT_COLS)), dtype=np.float64)
    shunt[:, SHUNT_COLS["idx"]] = [1, 2]
    shunt[:, SHUNT_COLS["control_type"]] = [SHUNT_V, SHUNT_Q]
    shunt[:, SHUNT_COLS["v_set"]] = 0.0
    shunt[:, SHUNT_COLS["run_stat"]] = 1.0

    acac = np.zeros((1, len(ACAC_COLS)), dtype=np.float64)
    acac[0, ACAC_COLS["idx"]] = 1
    acac[0, ACAC_COLS["i_control_type"]] = ACAC_SIDE_CONTROL_CODE["PV"]
    acac[0, ACAC_COLS["j_control_type"]] = ACAC_SIDE_CONTROL_CODE["PQ"]
    acac[0, ACAC_COLS["i_v_set"]] = np.nan
    acac[0, ACAC_COLS["j_v_set"]] = 0.0
    acac[0, ACAC_COLS["run_stat"]] = 1.0

    ppc = {"gen": gen, "shunt": shunt, "acac": acac}
    corrections = sanitize_ac_voltage_setpoints(ppc)

    np.testing.assert_allclose(gen[:2, GEN_COLS["v_set"]], 1.0)
    assert gen[2, GEN_COLS["v_set"]] == 0.0
    assert shunt[0, SHUNT_COLS["v_set"]] == 1.0
    assert shunt[1, SHUNT_COLS["v_set"]] == 0.0
    assert acac[0, ACAC_COLS["i_v_set"]] == 1.0
    assert acac[0, ACAC_COLS["j_v_set"]] == 0.0
    assert len(corrections) == 4


def test_ac_auto_slack_generator_voltage_is_sanitized():
    from ac_array_model import CTRL_PQ, GEN_COLS, sanitize_ac_voltage_setpoints

    gen = np.zeros((1, len(GEN_COLS)), dtype=np.float64)
    gen[0, GEN_COLS["idx"]] = 7
    gen[0, GEN_COLS["control_type"]] = CTRL_PQ
    gen[0, GEN_COLS["v_set"]] = 0.0
    gen[0, GEN_COLS["run_stat"]] = 1.0
    ppc = {"gen": gen, "_auto_slack_gen_rows": np.asarray([0], dtype=np.int32)}

    sanitize_ac_voltage_setpoints(ppc)

    assert gen[0, GEN_COLS["v_set"]] == 1.0


def test_dc_voltage_sanitizer_covers_generators_and_dcdc_controlled_side_only():
    from dc_array_model import (
        CTRL_NONE,
        CTRL_P,
        CTRL_V,
        DCDC_COLS,
        GEN_COLS,
        sanitize_dc_voltage_setpoints,
    )

    gen = np.zeros((2, len(GEN_COLS)), dtype=np.float64)
    gen[:, GEN_COLS["idx"]] = [1, 2]
    gen[:, GEN_COLS["control_type"]] = [CTRL_V, CTRL_P]
    gen[:, GEN_COLS["v_set"]] = [0.0, 0.0]
    gen[:, GEN_COLS["run_stat"]] = 1.0

    dcdc = np.zeros((3, len(DCDC_COLS)), dtype=np.float64)
    dcdc[:, DCDC_COLS["idx"]] = [1, 2, 3]
    dcdc[:, DCDC_COLS["i_control_type"]] = [CTRL_V, CTRL_NONE, CTRL_P]
    dcdc[:, DCDC_COLS["j_control_type"]] = [CTRL_NONE, CTRL_V, CTRL_NONE]
    dcdc[:, DCDC_COLS["v_set"]] = [0.0, np.inf, 0.0]
    dcdc[:, DCDC_COLS["run_stat"]] = 1.0

    ppc = {"gen": gen, "dcdc": dcdc}
    corrections = sanitize_dc_voltage_setpoints(ppc)

    assert gen[0, GEN_COLS["v_set"]] == 1.0
    assert gen[1, GEN_COLS["v_set"]] == 0.0
    np.testing.assert_allclose(dcdc[:2, DCDC_COLS["v_set"]], 1.0)
    assert dcdc[2, DCDC_COLS["v_set"]] == 0.0
    assert len(corrections) == 3


def test_dcac_voltage_sanitizer_checks_only_active_voltage_control_terminals():
    from hybrid_array_model import (
        DCAC_AC_CONTROL_CODE,
        DCAC_COLS,
        DCAC_DC_CONTROL_CODE,
        sanitize_dcac_voltage_setpoints,
    )

    dcac = np.zeros((3, len(DCAC_COLS)), dtype=np.float64)
    dcac[:, DCAC_COLS["idx"]] = [1, 2, 3]
    dcac[:, DCAC_COLS["ac_control_type"]] = [
        DCAC_AC_CONTROL_CODE["PH"],
        DCAC_AC_CONTROL_CODE["PQ"],
        DCAC_AC_CONTROL_CODE["PQ"],
    ]
    dcac[:, DCAC_COLS["dc_control_type"]] = [
        DCAC_DC_CONTROL_CODE["NONE"],
        DCAC_DC_CONTROL_CODE["V"],
        DCAC_DC_CONTROL_CODE["NONE"],
    ]
    dcac[:, DCAC_COLS["v_ac_set"]] = [0.0, 0.0, 0.0]
    dcac[:, DCAC_COLS["v_dc_set"]] = [0.0, 1.7, 0.0]
    dcac[:, DCAC_COLS["run_stat"]] = 1.0

    ppc = {"dcac": dcac}
    corrections = sanitize_dcac_voltage_setpoints(ppc)

    assert dcac[0, DCAC_COLS["v_ac_set"]] == 1.0
    assert dcac[0, DCAC_COLS["v_dc_set"]] == 0.0
    assert dcac[1, DCAC_COLS["v_ac_set"]] == 0.0
    assert dcac[1, DCAC_COLS["v_dc_set"]] == 1.0
    assert dcac[2, DCAC_COLS["v_ac_set"]] == 0.0
    assert dcac[2, DCAC_COLS["v_dc_set"]] == 0.0
    assert len(corrections) == 2


def test_ac_state_estimator_sanitizes_direct_ppc_voltage_controls():
    from model.ac_array_model import CTRL_SLACK, GEN_COLS
    from model.ppc_topology import build_ac_ppc_with_topology_from_e_file
    from secore.ac_se import ACStateEstimator

    ppc = build_ac_ppc_with_topology_from_e_file(
        ROOT_DIR / "data" / "model" / "ac" / "ieee39.e"
    )
    control = ppc["gen"][:, GEN_COLS["control_type"]].astype(np.int8, copy=False)
    row = int(np.flatnonzero(control == CTRL_SLACK)[0])
    ppc["gen"][row, GEN_COLS["v_set"]] = 0.0
    ppc.pop("_setpoint_corrections", None)
    estimator = ACStateEstimator.__new__(ACStateEstimator)
    estimator.network = SimpleNamespace(ppc=ppc)

    estimator._ac_ppc_dict()

    assert ppc["gen"][row, GEN_COLS["v_set"]] == pytest.approx(1.0)
    assert any(
        item["device_type"] == "ACGenerator"
        for item in ppc["_setpoint_corrections"]
    )


def test_dc_state_estimator_sanitizes_direct_ppc_voltage_controls():
    from model.dc_array_model import CTRL_V, GEN_COLS
    from model.ppc_topology import build_dc_ppc_with_topology_from_e_file
    from secore.dc_se import DCStateEstimator

    ppc = build_dc_ppc_with_topology_from_e_file(
        ROOT_DIR / "data" / "model" / "dc" / "dc_net_30.e"
    )
    control = ppc["gen"][:, GEN_COLS["control_type"]].astype(np.int8, copy=False)
    row = int(np.flatnonzero(control == CTRL_V)[0])
    ppc["gen"][row, GEN_COLS["v_set"]] = 0.0
    ppc.pop("_setpoint_corrections", None)
    estimator = DCStateEstimator.__new__(DCStateEstimator)
    estimator.network = SimpleNamespace(ppc=ppc)

    estimator._dc_ppc_dict()

    assert ppc["gen"][row, GEN_COLS["v_set"]] == pytest.approx(1.0)
    assert any(
        item["device_type"] == "DCGenerator"
        for item in ppc["_setpoint_corrections"]
    )


def test_qinling_global_jacobian_lf_and_se_remain_consistent(tmp_path):
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
    from secore.hybrid_se import HybridStateEstimator
    from update_meas_from_lf import (
        Snapshot,
        _reconstruct_ac_ideal_edge_flows,
        _reconstruct_dc_ideal_edge_flows,
        parse_measurement_rows,
        rewrite_measurements,
    )

    case_file = ROOT_DIR / "data" / "model" / "hybrid" / "qinling.e"
    source_meas = ROOT_DIR / "data" / "meas" / "hybrid" / "qinling.meas"
    generated_meas = tmp_path / "qinling.meas"
    generated_meas.write_text(source_meas.read_text(encoding="utf-8"), encoding="utf-8")

    network = _read_lf_network_from_file(case_file)
    calc = HybridPowerFlowCalc(
        network,
        result_mode="full",
        max_iter=50,
        verbose=False,
    )
    calc.prepare()

    # Converter controls close directions that are intentionally external to
    # the AC subblock, so rank must be checked after global block assembly.
    _, initial_jacobian = calc._build_newton_system(calc.x)
    initial_dense = initial_jacobian.toarray()
    singular_values = np.linalg.svd(initial_dense, compute_uv=False)
    assert initial_jacobian.shape == (calc.total_vars, calc.total_vars)
    assert np.linalg.matrix_rank(initial_dense) == calc.total_vars
    assert singular_values[-1] > 1e-3

    assert calc.run() == 0
    assert calc.converged
    assert calc.normF < 1e-8

    ac_ideal_residual = _reconstruct_ac_ideal_edge_flows(
        network.ac,
        network.dcac_converters,
        network.acac_converters,
    )
    dc_ideal_residual = _reconstruct_dc_ideal_edge_flows(
        network.dc,
        network.dcac_converters,
    )
    assert ac_ideal_residual < 1e-8
    assert dc_ideal_residual < 1e-8

    snapshot = Snapshot(
        network,
        ac_grid=network.ac,
        dc_grid=network.dc,
        dcac_converters=network.dcac_converters,
        acac_converters=network.acac_converters,
    )
    _, original_rows, _ = parse_measurement_rows(source_meas)
    updated, missing = rewrite_measurements(generated_meas, snapshot)
    _, generated_rows, _ = parse_measurement_rows(generated_meas)
    assert updated > 0
    assert missing == 0
    assert [row[6:] for row in generated_rows] == [row[6:] for row in original_rows]

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=generated_meas,
        flat_start=True,
        max_iter=50,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator.run(result_mode="full", remove_bad_data=False, verbose=False)

    result = estimator.estimate_result
    assert result.converged
    assert result.observability.observable
    assert result.residual_inf < 1e-8
    assert estimator.electric_estimate_result.residual_inf < 1e-8
    assert estimator.multi_energy_result.converged
    assert estimator.multi_energy_result.observable
    assert all(
        fluid_estimator.result.converged
        and fluid_estimator.result.residual_inf < 1e-8
        for fluid_estimator in estimator.fluid_estimators.values()
    )
    assert all(
        abs(float(coupling.residual)) < 1e-10
        for coupling in estimator.multi_energy_result.couplings
    )


@pytest.mark.parametrize(
    ("prefix", "potential_power", "thermal", "steam", "node_pressure"),
    (
        ("Gas", 2, False, False, 5.0),
        ("Hydro", 2, False, False, 35.0),
        ("Steam", 2, False, True, 4.5),
        ("Heat", 1, True, False, 10.0),
    ),
)
def test_fluid_pressure_control_zero_falls_back_to_connected_node(
    prefix,
    potential_power,
    thermal,
    steam,
    node_pressure,
):
    from model.fluid_model import FluidNetwork, FluidNode, FluidSource, PRESSURE_CONTROL

    node = FluidNode(
        idx=1,
        name="anchor",
        pressure=node_pressure,
        supply_temperature=85.0,
        return_temperature=50.0,
        run_stat=1,
    )
    source = FluidSource(
        idx=1,
        name="source",
        node=1,
        control_type=PRESSURE_CONTROL,
        pressure_set=0.0,
        supply_temperature_set=85.0,
        return_temperature_set=50.0,
    )
    network = FluidNetwork(
        prefix=prefix,
        potential_power=potential_power,
        thermal=thermal,
        steam=steam,
        nodes=[node],
        sources=[source],
    ).prepare()

    assert network.source_pressure_set[0] == pytest.approx(node_pressure)
    assert network.fixed_pressure[0] == pytest.approx(node_pressure)
    assert any(item["field"] == "pressure_set" for item in network.setpoint_corrections)


def test_non_pressure_control_source_keeps_zero_pressure_placeholder():
    from model.fluid_model import FLOW_CONTROL, FluidNetwork, FluidNode, FluidSource

    network = FluidNetwork(
        prefix="Gas",
        potential_power=2,
        nodes=[FluidNode(idx=1, name="node", pressure=5.0)],
        sources=[
            FluidSource(
                idx=1,
                name="flow_source",
                node=1,
                control_type=FLOW_CONTROL,
                pressure_set=0.0,
                flow_set=0.0,
            )
        ],
    ).prepare()

    assert network.source_pressure_set[0] == 0.0


def test_heat_fixed_temperatures_fall_back_to_connected_node_values():
    from model.fluid_model import FLOW_CONTROL, FluidNetwork, FluidNode, FluidStorage

    network = FluidNetwork(
        prefix="Heat",
        potential_power=1,
        thermal=True,
        nodes=[
            FluidNode(
                idx=1,
                name="tank_bus",
                pressure=10.0,
                supply_temperature=88.0,
                return_temperature=52.0,
            )
        ],
        storages=[
            FluidStorage(
                idx=1,
                name="tank",
                node=1,
                control_type=FLOW_CONTROL,
                pressure_set=10.0,
                supply_temperature_set=0.0,
                return_temperature_set=np.nan,
            )
        ],
    ).prepare()

    assert network.source_supply_temperature_set[0] == pytest.approx(88.0)
    assert network.source_return_temperature_set[0] == pytest.approx(52.0)
    np.testing.assert_allclose(network.fixed_temperature, [88.0, 52.0])
    assert {item["field"] for item in network.setpoint_corrections} >= {
        "supply_temperature_set",
        "return_temperature_set",
    }


def test_invalid_automatic_fluid_pressure_anchor_uses_positive_nominal_fallback():
    from model.fluid_model import FluidNetwork, FluidNode

    network = FluidNetwork(
        prefix="Gas",
        potential_power=2,
        nodes=[FluidNode(idx=1, name="invalid_anchor", pressure=0.0)],
    ).prepare()

    assert network.fixed_pressure[0] == pytest.approx(1.0)
    assert network.node_pressure[0] == pytest.approx(1.0)
    assert any(item["device_type"] == "GasNode" for item in network.setpoint_corrections)


def test_heat_node_zero_temperature_initial_values_are_replaced():
    from model.fluid_model import FluidNetwork, FluidNode

    network = FluidNetwork(
        prefix="Heat",
        potential_power=1,
        thermal=True,
        nodes=[
            FluidNode(
                idx=1,
                name="invalid_temperature_node",
                pressure=10.0,
                supply_temperature=0.0,
                return_temperature=0.0,
                temperature=0.0,
            )
        ],
    ).prepare()

    assert network.node_supply_temperature[0] == pytest.approx(80.0)
    assert network.node_return_temperature[0] == pytest.approx(50.0)
    assert network.node_temperature[0] == pytest.approx(80.0)
    assert {item["field"] for item in network.setpoint_corrections} >= {
        "supply_temperature",
        "return_temperature",
        "temperature",
    }


def test_steam_node_zero_enthalpy_initial_value_is_replaced():
    from model.fluid_model import FluidNetwork, FluidNode

    network = FluidNetwork(
        prefix="Steam",
        potential_power=2,
        steam=True,
        nodes=[FluidNode(idx=1, name="invalid_enthalpy_node", pressure=5.0, enthalpy=0.0)],
    ).prepare()

    assert network.node_enthalpy[0] == pytest.approx(3000.0)
    assert any(
        item["field"] == "enthalpy" for item in network.setpoint_corrections
    )


def test_steam_fixed_enthalpy_falls_back_to_connected_node_value():
    from model.fluid_model import FluidNetwork, FluidNode, FluidSource, PRESSURE_CONTROL

    network = FluidNetwork(
        prefix="Steam",
        potential_power=2,
        steam=True,
        nodes=[FluidNode(idx=1, name="steam_bus", pressure=5.0, enthalpy=3200.0)],
        sources=[
            FluidSource(
                idx=1,
                name="boiler",
                node=1,
                control_type=PRESSURE_CONTROL,
                pressure_set=5.0,
                enthalpy_set=0.0,
            )
        ],
    ).prepare()

    assert network.source_enthalpy_set[0] == pytest.approx(3200.0)
    assert any(
        item["field"] == "enthalpy_set" for item in network.setpoint_corrections
    )
