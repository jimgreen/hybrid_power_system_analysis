from pathlib import Path

import numpy as np

from heat_lf import HeatPowerFlowCalc
from lfcore.hybrid_lf import HybridPowerFlowCalc
from model.fluid_model import (
    PASSIVE_CONTROL,
    PRESSURE_CONTROL,
    FluidEdge,
    HeatExchanger,
    FluidLoad,
    FluidNetwork,
    FluidNode,
    FluidSource,
)


ROOT = Path(__file__).resolve().parents[1]


def _heat_network_with_dead_island() -> FluidNetwork:
    return FluidNetwork(
        prefix="Heat",
        potential_power=1,
        thermal=True,
        nodes=[
            FluidNode(1, "active_source", 10.0, 90.0, 50.0),
            FluidNode(2, "active_load", 9.0, 85.0, 50.0),
            FluidNode(3, "active_zero_flow_leaf", 10.0, 80.0, 50.0),
            FluidNode(4, "dead_anchor", 7.0, 70.0, 45.0),
            FluidNode(5, "dead_leaf", 7.0, 70.0, 45.0),
        ],
        sources=[
            FluidSource(
                idx=1,
                name="active_heat_source",
                node=1,
                control_type=PRESSURE_CONTROL,
                pressure_set=10.0,
                flow_min=0.0,
                flow_max=5.0,
                supply_temperature=90.0,
            )
        ],
        loads=[FluidLoad(1, "active_heat_load", 2, 1.0, heat_power=20.0)],
        pipes=[
            FluidEdge(1, "active_pipe", 1, 2, "pipe", PASSIVE_CONTROL, 1.0),
            FluidEdge(2, "active_zero_flow_pipe", 1, 3, "pipe", PASSIVE_CONTROL, 1.0),
            FluidEdge(3, "dead_pipe", 4, 5, "pipe", PASSIVE_CONTROL, 1.0),
        ],
    )


def _all_dead_heat_network() -> FluidNetwork:
    return FluidNetwork(
        prefix="Heat",
        potential_power=1,
        thermal=True,
        nodes=[
            FluidNode(1, "dead_1", 6.0, 60.0, 40.0),
            FluidNode(2, "dead_2", 6.0, 60.0, 40.0),
        ],
        sources=[
            FluidSource(
                1,
                "idle_pressure_source",
                1,
                PRESSURE_CONTROL,
                6.0,
                flow_set=2.0,
                flow_min=0.0,
                flow_max=5.0,
            )
        ],
        pipes=[FluidEdge(1, "dead_pipe", 1, 2, "pipe", PASSIVE_CONTROL, 1.0)],
    )


def test_heat_prepare_presolves_hydraulics_and_excludes_only_zero_flow_island():
    calc = HeatPowerFlowCalc(
        _heat_network_with_dead_island(),
        tol=1.0e-10,
        max_iter=50,
        result_mode="full",
    ).prepare()

    metadata = calc.hydraulic_presolve
    assert metadata["converged"]
    assert metadata["original_island_count"] == 2
    assert metadata["active_island_ids"] == [0]
    assert metadata["dead_island_ids"] == [1]
    assert metadata["dead_node_names"] == ["dead_anchor", "dead_leaf"]
    assert metadata["initial_state_policy"] == "presolved_after_dead_island_compaction"
    assert metadata["presolved_state_used"]
    assert [node.name for node in calc.network.nodes] == [
        "active_source",
        "active_load",
        "active_zero_flow_leaf",
    ]
    assert [edge.name for edge in calc.network.edges] == [
        "active_pipe",
        "active_zero_flow_pipe",
    ]
    assert metadata["zero_flow_thermal_edge_positions"] == [1]
    assert metadata["zero_flow_thermal_edge_names"] == ["active_zero_flow_pipe"]
    assert calc.hydraulic_state_count == 2
    assert calc.total_vars == 8

    hydraulic_residual = calc._hydraulic_residual_and_jacobian(
        calc.x[: calc.hydraulic_state_count],
        return_jacobian=False,
    )[0]
    assert np.linalg.norm(hydraulic_residual, np.inf) < calc.tol

    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, [1.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(calc.source_flow, [1.0], atol=1.0e-10)
    np.testing.assert_array_equal(
        calc.lf_result.arrays["edge_thermal_active"],
        [True, False],
    )
    zero_flow_result = calc.lf_result.pipes["active_zero_flow_pipe"]
    assert not zero_flow_result.thermal_active
    assert zero_flow_result.i_supply_temperature == calc.supply_temperature[0]
    assert zero_flow_result.j_supply_temperature == calc.supply_temperature[2]
    assert zero_flow_result.i_return_temperature == calc.return_temperature[0]
    assert zero_flow_result.j_return_temperature == calc.return_temperature[2]
    assert calc.lf_result.metadata["hydraulic_presolve"]["dead_island_ids"] == [1]


def test_active_island_zero_flow_edge_is_hydraulic_only_until_flow_recovers():
    calc = HeatPowerFlowCalc(
        _heat_network_with_dead_island(),
        tol=1.0e-10,
        max_iter=50,
        result_mode="array",
    ).prepare()

    zero_edge_pos = calc.network.edge_pos_by_name["active_zero_flow_pipe"]
    zero_leaf_pos = calc.network.node_pos_by_name["active_zero_flow_leaf"]
    zero_leaf_state = int(
        calc.network.supply_temperature_state_by_node[zero_leaf_pos]
    )
    residual, jacobian, _potential, edge_flow = calc._residual_and_jacobian(calc.x)
    thermal_row = calc.hydraulic_state_count + zero_leaf_state

    assert abs(float(edge_flow[zero_edge_pos])) <= 1.0e-10
    assert jacobian.getrow(thermal_row).nnz == 1
    assert jacobian[thermal_row, calc.base_temperature + zero_leaf_state] == 1.0
    assert residual[thermal_row] == 0.0

    recovered = calc.x.copy()
    recovered[calc.network.free_state_by_node[zero_leaf_pos]] -= 1.0
    _residual, recovered_jacobian, _potential, recovered_flow = (
        calc._residual_and_jacobian(recovered)
    )

    assert abs(float(recovered_flow[zero_edge_pos])) > 1.0e-10
    assert recovered_jacobian.getrow(thermal_row).nnz > 1


def test_active_island_zero_flow_heat_exchanger_has_no_thermal_transfer():
    network = _heat_network_with_dead_island()
    network.heat_exchangers.append(
        HeatExchanger(
            idx=1,
            name="zero_flow_heat_exchanger",
            i_node=1,
            j_node=2,
            control_type="EFFECTIVENESS",
            primary_flow=0.0,
            secondary_flow=0.0,
            effectiveness=0.8,
            heat_loss=0.02,
        )
    )
    calc = HeatPowerFlowCalc(
        network,
        tol=1.0e-10,
        max_iter=50,
        result_mode="full",
    )

    assert calc.run() == 0
    exchanger = calc.lf_result.heat_exchangers["zero_flow_heat_exchanger"]
    primary_node = calc.network.node_pos_by_name["active_source"]
    secondary_node = calc.network.node_pos_by_name["active_load"]

    assert not exchanger.thermal_active
    assert exchanger.primary_heat == 0.0
    assert exchanger.secondary_heat == 0.0
    assert exchanger.primary_out_temperature == calc.return_temperature[primary_node]
    assert exchanger.secondary_out_temperature == calc.supply_temperature[secondary_node]
    assert np.all(
        np.isfinite(
            [
                exchanger.primary_out_temperature,
                exchanger.secondary_out_temperature,
                exchanger.primary_heat,
                exchanger.secondary_heat,
            ]
        )
    )


def test_all_zero_flow_heat_network_becomes_a_zero_size_solver_block():
    calc = HeatPowerFlowCalc(
        _all_dead_heat_network(),
        tol=1.0e-10,
        max_iter=20,
        result_mode="full",
    ).prepare()

    assert calc.hydraulic_presolve["dead_island_ids"] == [0]
    assert calc.hydraulic_presolve["active_island_ids"] == []
    assert (
        calc.hydraulic_presolve["initial_state_policy"]
        == "presolved_after_dead_island_compaction"
    )
    assert calc.hydraulic_presolve["presolved_state_used"]
    assert calc.total_vars == 0
    assert calc.total_eq == 0
    assert calc.x.size == 0
    assert calc.run() == 0
    assert calc.converged
    assert calc.iterations == 0
    assert calc.normF == 0.0
    assert calc.lf_result.metadata["hydraulic_presolve"]["dead_node_names"] == [
        "dead_1",
        "dead_2",
    ]
    assert calc.lf_result.metadata["hydraulic_presolve"]["excluded_devices"][0] == {
        "device_type": "HeatSource",
        "idx": 1,
        "name": "idle_pressure_source",
    }


def test_zero_net_injection_island_with_hydraulic_circulation_remains_active():
    network = FluidNetwork(
        prefix="Heat",
        potential_power=1,
        thermal=True,
        nodes=[
            FluidNode(1, "circulation_anchor", 10.0, 80.0, 50.0),
            FluidNode(2, "circulation_leaf", 9.0, 80.0, 50.0),
        ],
        sources=[
            FluidSource(
                1,
                "circulation_pressure_source",
                1,
                PRESSURE_CONTROL,
                10.0,
                flow_min=-5.0,
                flow_max=5.0,
            )
        ],
        pipes=[FluidEdge(1, "circulation_pipe", 1, 2, "pipe", PASSIVE_CONTROL, 1.0)],
        controllers=[
            FluidEdge(
                1,
                "circulation_pump",
                2,
                1,
                "pump",
                "GAIN",
                pressure_gain=1.0,
            )
        ],
    )
    calc = HeatPowerFlowCalc(
        network,
        tol=1.0e-10,
        max_iter=50,
        result_mode="full",
    ).prepare()

    assert calc.hydraulic_presolve["dead_island_ids"] == []
    assert calc.hydraulic_presolve["active_island_ids"] == [0]
    assert calc.hydraulic_presolve["initial_state_policy"] == "model"
    assert not calc.hydraulic_presolve["presolved_state_used"]
    assert len(calc.network.nodes) == 2
    assert calc.run() == 0
    np.testing.assert_allclose(calc.edge_flow, [1.0, 1.0], atol=1.0e-10)
    np.testing.assert_allclose(calc.source_flow, [0.0], atol=1.0e-10)


def test_hybrid_global_layout_uses_only_active_heat_islands(tmp_path):
    heat = """
<HeatMedium>
@ density heat_capacity ambient_temperature temperature flow_factor
# 998.0 4.186 20.0 353.15 1.0
</HeatMedium>

<HeatNode>
@ idx name pressure supply_temperature return_temperature run_stat
# 1 active_source 10.0 90.0 50.0 1
# 2 active_load 9.0 85.0 50.0 1
# 3 dead_anchor 7.0 70.0 45.0 1
# 4 dead_leaf 7.0 70.0 45.0 1
</HeatNode>

<HeatSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 active_heat_source 1 PRESSURE 10.0 0.0 1.0 0.0 5.0 90.0 1
</HeatSource>

<HeatLoad>
@ idx name node mass_flow heat_power run_stat
# 1 active_heat_load 2 1.0 20.0 1
</HeatLoad>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 active_pipe 1 2 1.0 0.0 1
# 2 dead_pipe 3 4 1.0 0.0 1
</HeatPipe>
""".strip()
    case_file = tmp_path / "hybrid_with_dead_heat_island.e"
    case_file.write_text(
        (ROOT / "data" / "model" / "ac" / "ac_net_10.e").read_text(encoding="utf8")
        + "\n\n"
        + heat
        + "\n",
        encoding="utf8",
    )

    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        tol=1.0e-9,
        max_iter=100,
        result_mode="full",
        linear_solver="scipy",
    )
    calc.prepare()
    heat_calc = calc.fluid_calcs["heat"]
    heat_slice = calc.fluid_state_slices["heat"]

    assert heat_calc.hydraulic_presolve["dead_island_ids"] == [1]
    assert heat_slice.stop - heat_slice.start == heat_calc.total_vars == 5
    assert calc.total_vars == calc.electric_vars + heat_calc.total_vars
    assert calc.total_eq == calc.electric_eq + heat_calc.total_eq

    assert calc.run() == 0
    assert calc.converged
    assert calc.last_jacobian_shape == (calc.total_eq, calc.total_vars)
    assert calc.result["fluids"]["heat"]["dead_islands"] == 1


def test_hybrid_skips_coupling_whose_heat_endpoint_is_on_a_dead_island(tmp_path):
    heat = """
<HeatNode>
@ idx name pressure supply_temperature return_temperature run_stat
# 1 dead_source_node 7.0 70.0 45.0 1
# 2 dead_leaf 7.0 70.0 45.0 1
</HeatNode>

<HeatSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature run_stat
# 1 dead_heat_source 1 PRESSURE 7.0 0.0 1.0 0.0 5.0 70.0 1
</HeatSource>

<HeatPipe>
@ idx name i_node j_node conductance heat_loss run_stat
# 1 dead_pipe 1 2 1.0 0.0 1
</HeatPipe>

<AcE2Heat>
@ idx name run_stat control_type idx_ac_load_t1 idx_heat_unit_t2 e2h_coeff
# 1 dead_island_heater 1 P 1 1 0.95
</AcE2Heat>
""".strip()
    case_file = tmp_path / "hybrid_with_dead_heat_coupling.e"
    case_file.write_text(
        (ROOT / "data" / "model" / "ac" / "ac_net_10.e").read_text(encoding="utf8")
        + "\n\n"
        + heat
        + "\n",
        encoding="utf8",
    )

    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        tol=1.0e-9,
        max_iter=100,
        result_mode="full",
        linear_solver="scipy",
    )
    calc.prepare()

    assert calc.fluid_calcs["heat"].total_vars == 0
    assert calc.energy_coupling_plans == []
    assert any(
        "dead_island_heater" in warning
        and "zero-flow heat hydraulic island" in warning
        for warning in calc.multi_energy.warnings
    )
    assert calc.run() == 0
    assert calc.converged
