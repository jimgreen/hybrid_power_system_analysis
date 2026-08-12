from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
from scripts.check_hybrid_converter_all_modes_1k import (
    EXPECTED_ACAC_MODES,
    EXPECTED_DCAC_MODES,
    EXPECTED_DCDC_MODES,
    _decode_control_modes,
    _network_statistics,
)
from scripts.generate_hybrid_multi_energy_1k import COUPLING_TYPES
from scripts.generate_hybrid_multi_energy_5k import COUPLINGS_PER_TYPE, NODE_COUNTS
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_multi_energy_5k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_multi_energy_5k.meas"


def test_multi_energy_5k_structure_and_control_coverage():
    book = EBook(CASE)
    node_blocks = {
        "ac": "ACNode",
        "dc": "DCNode",
        "heat": "HeatNode",
        "gas": "GasNode",
        "hydro": "HydroNode",
        "steam": "SteamNode",
    }
    actual_counts = {
        domain: len(book.data[block_name].data)
        for domain, block_name in node_blocks.items()
    }
    assert actual_counts == NODE_COUNTS
    assert sum(actual_counts.values()) == 5000
    assert all(
        len(book.data[table_name].data) == COUPLINGS_PER_TYPE
        for table_name in COUPLING_TYPES
    )

    network = _read_lf_network_from_file(CASE)
    calc = HybridPowerFlowCalc(network, result_mode="array", linear_solver="scipy")
    calc.prepare()
    modes = {
        key: frozenset(tuple(mode) for mode in values)
        for key, values in _decode_control_modes(calc).items()
    }
    assert modes == {
        "acac": EXPECTED_ACAC_MODES,
        "dcdc": EXPECTED_DCDC_MODES,
        "dcac": EXPECTED_DCAC_MODES,
    }
    statistics = _network_statistics(calc)
    assert statistics["acac_count"] == 4
    assert statistics["dcdc_count"] == 6
    assert statistics["dcac_count"] == 4
    assert statistics["dcac_device_types"] == {
        "DCACConverter": 3,
        "ACDCConverter": 1,
    }
    assert len(calc.energy_coupling_plans) == 220
    assert {plan.coupling.table_name for plan in calc.energy_coupling_plans} == set(
        COUPLING_TYPES
    )
    assert not calc.multi_energy.warnings
    hydrogen_plans = [
        plan
        for plan in calc.energy_coupling_plans
        if plan.coupling.hydrogen_electric_direction is not None
    ]
    assert {plan.coupling.control_type for plan in hydrogen_plans} == {"P", "FLOW"}
    assert all(
        plan.coupling.e2h_coeff is not None
        for plan in hydrogen_plans
        if plan.coupling.hydrogen_electric_direction == "E2H"
    )
    assert all(
        plan.coupling.h2e_coeff is not None
        for plan in hydrogen_plans
        if plan.coupling.hydrogen_electric_direction == "H2E"
    )
    electric_heat_plans = [
        plan for plan in calc.energy_coupling_plans if plan.coupling.is_electric_heat_control
    ]
    assert {plan.coupling.control_type for plan in electric_heat_plans} == {"P", "T_OUT"}
    assert all(plan.coupling.e2h_coeff is not None for plan in electric_heat_plans)
    gas_electric_plans = [
        plan for plan in calc.energy_coupling_plans if plan.coupling.is_gas_electric_control
    ]
    assert {plan.coupling.control_type for plan in gas_electric_plans} == {"P", "FLOW"}
    assert all(plan.coupling.g2e_coeff is not None for plan in gas_electric_plans)
    steam_electric_plans = [
        plan for plan in calc.energy_coupling_plans if plan.coupling.is_steam_electric_control
    ]
    assert {plan.coupling.control_type for plan in steam_electric_plans} == {"P", "FLOW"}
    assert all(plan.coupling.s2e_coeff is not None for plan in steam_electric_plans)
    gas_heat_plans = [
        plan for plan in calc.energy_coupling_plans if plan.coupling.is_gas_heat_control
    ]
    assert {plan.coupling.control_type for plan in gas_heat_plans} == {"FLOW", "T_OUT"}
    assert all(plan.coupling.g2h_coeff is not None for plan in gas_heat_plans)
    for table_name in ("Steam2AcE", "Steam2DcE"):
        plans = [
            plan
            for plan in calc.energy_coupling_plans
            if plan.coupling.table_name == table_name
        ]
        assert len(plans) == COUPLINGS_PER_TYPE
        assert all(plan.t1.domain == "steam" and plan.t1.kind == "load" for plan in plans)


def test_multi_energy_5k_lf_is_one_global_newton_problem(monkeypatch):
    calc = HybridPowerFlowCalc.from_file_fast(
        CASE,
        result_mode="array",
        linear_solver="scipy",
        tol=1e-8,
        max_iter=50,
        verbose=False,
    )
    calc.prepare()

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid LF must not launch a separate fluid Newton loop")

    for fluid_calc in calc.fluid_calcs.values():
        monkeypatch.setattr(fluid_calc, "run", fail_local_run)

    assert calc.run() == 0
    assert calc.converged
    assert calc.total_vars == calc.total_eq == 8468
    assert calc.iterations <= 15
    residual = calc.get_f(calc.x)
    jacobian = calc.get_jacobi(calc.x)
    assert float(np.max(np.abs(residual))) < 1e-8
    assert jacobian.shape == (8468, 8468)
    assert jacobian.nnz > 30000
    assert {item.status for item in calc.coupling_results} == {"balanced"}

    selected = {}
    for plan in calc.energy_coupling_plans:
        selected.setdefault(plan.coupling.table_name, plan)
    for plan in selected.values():
        column = int(plan.state_col)
        nonzero_rows = jacobian.getcol(column).nonzero()[0]
        expected_balance_row = (
            int(plan.heat_temperature_eq_row)
            if (
                (plan.electric_heat and plan.coupling.control_type == "P")
                or (plan.gas_heat and plan.coupling.control_type == "FLOW")
            )
            else int(plan.t1.balance_row)
        )
        assert expected_balance_row in nonzero_rows
        assert int(plan.eq_row) in nonzero_rows
        step = 1e-6 * max(1.0, abs(float(calc.x[column])))
        upper = calc.x.copy()
        lower = calc.x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (calc.get_f(upper) - calc.get_f(lower)) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian.getcol(column).toarray().ravel(),
            numeric,
            rtol=2e-6,
            atol=5e-8,
        )


def test_multi_energy_5k_se_is_observable_accurate_and_joint(monkeypatch):
    estimator = HybridStateEstimator(
        CASE,
        MEASUREMENTS,
        flat_start=True,
        max_iter=50,
        auto_prepare=False,
    )
    estimator.prepare()
    observability = estimator.observability_analysis()
    assert observability.observable
    assert observability.rank == observability.state_count == 8670

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid SE must not launch a separate fluid WLS loop")

    for fluid_estimator in estimator.fluid_estimators.values():
        monkeypatch.setattr(fluid_estimator, "run", fail_local_run)

    estimator.run(result_mode="array", skip_bad_data=True, verbose=False)
    result = estimator.multi_energy_estimate_result
    assert result.converged
    assert result.iterations <= 12
    assert result.residual_inf < 5e-8
    assert result.objective < 1e-10
    expected_measurement_count = (
        estimator.electric_measurement_count
        + sum(
            measurement_slice.stop - measurement_slice.start
            for measurement_slice in estimator.fluid_measurement_slices.values()
        )
        + (
            estimator.multi_energy_control_measurement_slice.stop
            - estimator.multi_energy_control_measurement_slice.start
        )
        + (
            estimator.multi_energy_coupling_measurement_slice.stop
            - estimator.multi_energy_coupling_measurement_slice.start
        )
    )
    assert estimator.multi_energy_measurement_count == expected_measurement_count
    assert result.H.shape == (
        estimator.multi_energy_measurement_count,
        estimator.multi_energy_state_count,
    )
    assert result.H.nnz > 50000
    assert all(rc == 0 for rc in estimator.fluid_se_rc.values())
    assert {item.status for item in estimator.multi_energy_result.couplings} == {
        "balanced"
    }
    coupling_rows = estimator.multi_energy_coupling_measurement_slice
    assert float(np.max(np.abs(result.residual[coupling_rows]))) < 1e-10

    analytic = estimator._multi_energy_jacobian_sparse(result.x)
    selected_rows = {}
    for plan in estimator.multi_energy_se_coupling_plans:
        selected_rows.setdefault(plan.coupling.table_name, int(plan.measurement_row))
    for row in selected_rows.values():
        columns = analytic.getrow(row).indices
        assert columns.size
        column = int(columns[0])
        step = 1e-6 * max(1.0, abs(float(result.x[column])))
        upper = result.x.copy()
        lower = result.x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (
            estimator._evaluate_multi_energy(upper)
            - estimator._evaluate_multi_energy(lower)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic.getcol(column).toarray().ravel(),
            numeric,
            rtol=2e-6,
            atol=5e-8,
        )
