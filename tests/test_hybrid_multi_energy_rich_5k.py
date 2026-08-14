from collections import Counter
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hybrid_power_system_analysis"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efile_read import EBook
from lfcore.hybrid_lf import HybridPowerFlowCalc
from model.meas_type import DEVICE_TYPE_CODES
from scripts.check_hybrid_converter_all_modes_1k import (
    EXPECTED_ACAC_MODES,
    EXPECTED_DCAC_MODES,
    EXPECTED_DCDC_MODES,
    _decode_control_modes,
)
from scripts.generate_hybrid_multi_energy_rich_5k import (
    COUPLING_TYPES,
    COUPLINGS_PER_TYPE,
    NODE_COUNTS,
    STORAGE_COUNTS,
)
from secore.hybrid_se import HybridStateEstimator


CASE = ROOT / "data" / "model" / "hybrid" / "hybrid_multi_energy_rich_5k.e"
MEASUREMENTS = ROOT / "data" / "meas" / "hybrid" / "hybrid_multi_energy_rich_5k.meas"


def test_rich_5k_structure_storage_and_control_coverage():
    book = EBook(CASE)
    node_blocks = {
        "ac": "ACNode",
        "dc": "DCNode",
        "heat": "HeatNode",
        "gas": "GasNode",
        "hydro": "HydroNode",
        "steam": "SteamNode",
    }
    actual_nodes = {
        domain: len(book.data[block].data)
        for domain, block in node_blocks.items()
    }
    assert actual_nodes == NODE_COUNTS
    assert sum(actual_nodes.values()) == 5000
    assert {
        table: len(book.data[table].data)
        for table in COUPLING_TYPES
    } == dict.fromkeys(COUPLING_TYPES, COUPLINGS_PER_TYPE)
    assert {
        table: len(book.data[table].data)
        for table in STORAGE_COUNTS
    } == STORAGE_COUNTS
    assert len(book.data["ACACConverter"].data) == 40
    assert len(book.data["DCDCConverter"].data) == 42
    assert len(book.data["DCACConverter"].data) == 80
    assert Counter(
        str(row["dev_type"])
        for row in book.data["DCACConverter"].data
    ) == {"DCACConverter": 40, "ACDCConverter": 40}

    calc = HybridPowerFlowCalc.from_file_fast(
        CASE,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )
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
    assert len(calc.energy_coupling_plans) == len(COUPLING_TYPES) * COUPLINGS_PER_TYPE
    assert not calc.multi_energy.warnings
    assert {
        plan.coupling.control_type
        for plan in calc.energy_coupling_plans
        if plan.coupling.is_electric_heat_control
    } == {"P", "T_OUT"}
    assert {
        plan.coupling.control_type
        for plan in calc.energy_coupling_plans
        if plan.coupling.is_gas_heat_control
    } == {"FLOW", "T_OUT"}
    heat_plans = [
        plan
        for plan in calc.energy_coupling_plans
        if plan.coupling.is_electric_heat_control
        or plan.coupling.is_gas_heat_control
    ]
    assert heat_plans
    assert {plan.heat.kind for plan in heat_plans} == {"source"}

    heat_sources = {
        int(row["idx"]): row
        for row in book.data["HeatSource"].data
    }
    single_port_tables = ("DcE2Heat", "AcE2Heat", "Gas2Heat")
    dual_port_tables = ("DcE2Heat2", "AcE2Heat2")
    for table_name in single_port_tables:
        for coupling in book.data[table_name].data:
            assert "idx_heat_storage_t2" not in coupling
            source = heat_sources[int(coupling["idx_heat_unit_t2"])]
            assert source["node"] != "-"
            assert source["supply_node"] == "-"
            assert source["return_node"] == "-"
    for table_name in dual_port_tables:
        for coupling in book.data[table_name].data:
            assert "idx_heat_storage_t2" not in coupling
            source = heat_sources[int(coupling["idx_heat_unit_t2"])]
            assert source["node"] == "-"
            assert source["supply_node"] != "-"
            assert source["return_node"] != "-"

    heat_network = calc.fluid_calcs["heat"].network
    for plan in heat_plans:
        pos = int(plan.heat.device_pos)
        expected_explicit = plan.coupling.table_name.endswith("Heat2")
        assert bool(heat_network.source_explicit_return[pos]) is expected_explicit

    exchangers = book.data["HeatExchanger"].data
    assert exchangers
    for exchanger in exchangers:
        assert exchanger["primary_supply_node"] != "-"
        assert exchanger["primary_return_node"] != "-"
        assert exchanger["j_node"] != "-"
        assert exchanger["secondary_supply_node"] == "-"
        assert exchanger["secondary_return_node"] == "-"
    for primary_supply, primary_return, secondary_supply, secondary_return in zip(
        heat_network.exchanger_primary_supply.tolist(),
        heat_network.exchanger_primary_return.tolist(),
        heat_network.exchanger_secondary_supply.tolist(),
        heat_network.exchanger_secondary_return.tolist(),
    ):
        assert heat_network.node_explicit_return[primary_supply]
        assert heat_network.node_explicit_return[primary_return]
        assert not heat_network.node_explicit_return[secondary_supply]
        assert not heat_network.node_explicit_return[secondary_return]
        assert heat_network.node_island[primary_supply] != heat_network.node_island[secondary_supply]
    for predicate in (
        "is_hydrogen_electric_control",
        "is_gas_electric_control",
        "is_steam_electric_control",
    ):
        assert {
            plan.coupling.control_type
            for plan in calc.energy_coupling_plans
            if getattr(plan.coupling, predicate)
        } == {"P", "FLOW"}

    expected_directions = {
        "heat": (30, 30),
        "gas": (10, 10),
        "hydro": (10, 10),
        "steam": (10, 10),
    }
    for domain, fluid_calc in calc.fluid_calcs.items():
        network = fluid_calc.network
        storage_flow = network.source_flow_set[network.source_is_storage]
        assert storage_flow.size == STORAGE_COUNTS[f"{domain.capitalize()}Storage"]
        assert (
            int(np.count_nonzero(storage_flow > 0.0)),
            int(np.count_nonzero(storage_flow < 0.0)),
        ) == expected_directions[domain]


def test_rich_5k_lf_is_one_global_problem_with_storage_results(monkeypatch):
    calc = HybridPowerFlowCalc.from_file_fast(
        CASE,
        result_mode="full",
        linear_solver="scipy",
        tol=1.0e-8,
        max_iter=50,
        verbose=False,
    )
    calc.prepare()

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid LF must not run a separate fluid Newton solve")

    for fluid_calc in calc.fluid_calcs.values():
        monkeypatch.setattr(fluid_calc, "run", fail_local_run)

    assert calc.run() == 0
    assert calc.converged
    expected_size = 8916 - STORAGE_COUNTS["HydroStorage"]
    assert calc.total_vars == calc.total_eq == expected_size
    assert calc.iterations <= 15
    residual = calc.get_f(calc.x)
    jacobian = calc.get_jacobi(calc.x)
    assert float(np.max(np.abs(residual))) < 1.0e-8
    assert jacobian.shape == (expected_size, expected_size)
    assert jacobian.nnz > 32000
    assert len(calc.coupling_results) == len(COUPLING_TYPES) * COUPLINGS_PER_TYPE
    assert {result.status for result in calc.coupling_results} == {"balanced"}

    for domain, fluid_calc in calc.fluid_calcs.items():
        expected_count = STORAGE_COUNTS[f"{domain.capitalize()}Storage"]
        assert len(fluid_calc.lf_result.storages) == expected_count
        storage_flow = fluid_calc.lf_result.arrays["storage_flow"]
        assert storage_flow.size == expected_count
        assert np.all(np.isfinite(storage_flow))
        assert np.any(np.abs(storage_flow) > 1.0e-12)
        if domain != "hydro":
            assert np.any(storage_flow > 0.0)
            assert np.any(storage_flow < 0.0)
    hydro_calc = calc.fluid_calcs["hydro"]
    hydro_network = hydro_calc.network
    hydro_storage_sources = hydro_network.storage_source_pos
    np.testing.assert_allclose(
        hydro_calc.pressure[
            hydro_network.source_node_pos[hydro_storage_sources]
        ],
        hydro_network.source_pressure_set[hydro_storage_sources],
        rtol=0.0,
        atol=1.0e-10,
    )
    heat_arrays = calc.fluid_calcs["heat"].lf_result.arrays
    assert heat_arrays["storage_heat_power"].size == STORAGE_COUNTS["HeatStorage"]


def test_rich_5k_se_is_observable_accurate_joint_and_storage_aware(monkeypatch):
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
    assert observability.rank == observability.state_count == 9184

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid SE must not run a separate fluid WLS solve")

    for fluid_estimator in estimator.fluid_estimators.values():
        monkeypatch.setattr(fluid_estimator, "run", fail_local_run)

    estimator.run(
        result_mode="array",
        skip_bad_data=True,
        final_diagnostics=True,
        observability=observability,
        verbose=False,
    )
    result = estimator.multi_energy_estimate_result
    assert result.converged
    assert result.iterations <= 12
    assert result.residual_inf < 5.0e-8
    assert result.objective < 1.0e-10
    # All 80 DCAC AC-terminal current measurements are active coupling rows.
    assert result.H.shape == (26925, 9184)
    assert result.H.nnz > 53700
    coupling_rows = estimator.multi_energy_coupling_measurement_slice
    assert float(np.max(np.abs(result.residual[coupling_rows]))) < 1.0e-10
    assert all(rc == 0 for rc in estimator.fluid_se_rc.values())
    assert {item.status for item in estimator.multi_energy_result.couplings} == {
        "balanced"
    }

    table = result.measurement_table
    for device_type, expected_count in (
        ("HeatStorage", 240),
        ("GasStorage", 20),
        ("HydroStorage", 20),
        ("SteamStorage", 20),
    ):
        rows = np.flatnonzero(
            np.asarray(table.device_type_code, dtype=np.int64)
            == DEVICE_TYPE_CODES[device_type]
        )
        assert rows.size == expected_count
        assert float(np.max(np.abs(result.residual[rows]))) < 1.0e-10
