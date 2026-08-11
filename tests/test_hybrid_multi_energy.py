from pathlib import Path

import numpy as np
import pytest

from lfcore.hybrid_lf import HybridPowerFlowCalc
from model.meas_model import Measurement, MeasurementList, measurement_table_from_measurements
from secore.hybrid_se import HybridStateEstimator


ROOT = Path(__file__).resolve().parents[1]


def _measurement_rows(path: Path):
    rows = []
    in_block = False
    for raw_line in path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if line == "<Measurement>":
            in_block = True
        elif line == "</Measurement>":
            in_block = False
        elif in_block and line.startswith("#"):
            rows.append(line[1:].strip().split())
    return rows


def _build_multi_energy_case(tmp_path: Path):
    model_parts = [
        (ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e").read_text(
            encoding="utf8"
        )
    ]
    measurement_files = [
        ROOT / "data" / "meas" / "hybrid" / "hybrid_net_40.meas"
    ]
    for name, stem in (
        ("heat", "heat_net_3"),
        ("gas", "gas_net_3"),
        ("hydro", "hydro_net_3"),
        ("steam", "steam_net_5"),
    ):
        model_parts.append(
            (ROOT / "data" / "model" / name / f"{stem}.e").read_text(encoding="utf8")
        )
        measurement_files.append(
            ROOT / "data" / "meas" / name / f"{stem}.meas"
        )
    model_parts.append(
        """
<AcE2Heat>
@ idx name run_stat idx_ac_load_t1 idx_heat_unit_t2 efficiency energy_factor
# 1 electric_boiler 1 1 1 0.95 100.0
</AcE2Heat>

<Gas2AcE>
@ idx name run_stat idx_ac_unit_t1 idx_gas_load_t2 efficiency energy_factor
# 1 gas_turbine 1 2 1 0.40 100.0
</Gas2AcE>

<AcE2Hydro>
@ idx name run_stat idx_ac_load_t1 idx_h2_unit_t2 efficiency energy_factor
# 1 electrolyzer 1 2 1 0.75 100.0
</AcE2Hydro>

<Steam2DcE>
@ idx name run_stat idx_steam_load_t1 idx_dc_unit_t2 efficiency energy_factor
# 1 steam_generator 1 1 2 0.35 1500.0
</Steam2DcE>

<Gas2Heat>
@ idx name run_stat idx_gas_load_t1 idx_heat_unit_t2 efficiency energy_factor
# 1 gas_boiler 1 2 1 0.90 100.0
</Gas2Heat>
""".strip()
    )
    case_file = tmp_path / "multi_energy.e"
    case_file.write_text("\n\n".join(model_parts) + "\n", encoding="utf8")

    rows = []
    for measurement_file in measurement_files:
        rows.extend(_measurement_rows(measurement_file))
    meas_file = tmp_path / "multi_energy.meas"
    output = [
        "<Measurement>",
        "@ idx name dev_type dev_name meas_type weight valid value",
    ]
    for idx, row in enumerate(rows, start=1):
        row[0] = str(idx)
        output.append("# " + " ".join(row))
    output.append("</Measurement>")
    meas_file.write_text("\n".join(output) + "\n", encoding="utf8")
    return case_file, meas_file


def test_hybrid_lf_runs_all_fluid_networks_and_parses_couplings(tmp_path):
    case_file, _ = _build_multi_energy_case(tmp_path)
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    assert calc.converged
    assert set(calc.fluid_calcs) == {"heat", "gas", "hydro", "steam"}
    assert all(item.converged for item in calc.fluid_calcs.values())
    assert set(calc.lf_result.fluid) == {"heat", "gas", "hydro", "steam"}
    assert calc.lf_result.total_fluid_nodes == 20
    assert len(calc.coupling_results) == 5
    assert {item.status for item in calc.coupling_results} == {"balanced"}
    assert not calc.lf_result.fluid_errors
    steam_plan = next(
        plan
        for plan in calc.energy_coupling_plans
        if plan.coupling.table_name == "Steam2DcE"
    )
    assert steam_plan.t1.domain == "steam"
    assert steam_plan.t1.kind == "load"
    assert steam_plan.t2.domain == "dc"


def test_hybrid_storage_endpoint_keeps_storage_identity_and_source_balance(tmp_path):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    storage_blocks = """
<HydroStorage>
@ idx name node control_type flow_set alpha flow_min flow_max run_stat
# 1 hydrogen_buffer 3 FLOW 0.0 1.0 -2.0 2.0 1
</HydroStorage>

<Hydro2DcE>
@ idx name run_stat idx_h2_storage_t1 idx_dc_unit_t2 efficiency energy_factor
# 1 hydrogen_buffer_supply 1 1 2 1.0 100.0
</Hydro2DcE>
""".strip()
    case_file.write_text(
        case_file.read_text(encoding="utf8") + "\n\n" + storage_blocks + "\n",
        encoding="utf8",
    )

    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    assert calc.run() == 0
    plan = next(
        plan
        for plan in calc.energy_coupling_plans
        if plan.coupling.name == "hydrogen_buffer_supply"
    )
    assert plan.t1.kind == "storage"
    assert plan.t1.device_type == "HydroStorage"
    storage_flow = float(calc.x[plan.state_col])
    assert calc.fluid_calcs["hydro"].lf_result.storages["hydrogen_buffer"].flow == pytest.approx(
        storage_flow
    )

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    terminal = next(
        coupling.t1
        for coupling in estimator.multi_energy.couplings
        if coupling.name == "hydrogen_buffer_supply"
    )
    measurement, reason = estimator._fluid_endpoint_measurement(terminal)
    assert reason == ""
    assert measurement.device_type == "HydroStorage"
    assert measurement.device_pos == 1


def test_hybrid_lf_uses_one_global_newton_system_without_fluid_run(tmp_path, monkeypatch):
    case_file, _ = _build_multi_energy_case(tmp_path)
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )
    calc.prepare()

    expected_vars = calc.electric_vars + sum(
        item.total_vars for item in calc.fluid_calcs.values()
    ) + len(calc.energy_coupling_plans)
    expected_eq = calc.electric_eq + sum(
        item.total_eq for item in calc.fluid_calcs.values()
    ) + len(calc.energy_coupling_plans)
    residual, jacobian = calc._build_newton_system(calc.x)
    assert calc.total_vars == expected_vars
    assert calc.total_eq == expected_eq
    assert residual.shape == (expected_eq,)
    assert jacobian.shape == (expected_eq, expected_vars)

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid LF must not start a separate fluid Newton loop")

    for fluid_calc in calc.fluid_calcs.values():
        monkeypatch.setattr(fluid_calc, "run", fail_local_run)

    assert calc.run() == 0
    assert all(item.iterations == calc.iterations for item in calc.fluid_calcs.values())

    analytic = calc.get_jacobi(calc.x).toarray()
    assert analytic[
        calc.energy_coupling_eq_slice,
        : calc.energy_coupling_state_slice.start,
    ].any()
    assert analytic[
        : calc.energy_coupling_eq_slice.start,
        calc.energy_coupling_state_slice,
    ].any()
    checked_columns = set()
    for plan in calc.energy_coupling_plans:
        checked_columns.add(plan.state_col)
        for endpoint in (plan.t1, plan.t2):
            _value, columns, _derivatives = calc._energy_endpoint_value_and_derivative(
                endpoint,
                calc.x,
            )
            checked_columns.update(columns)
    for column in checked_columns:
        step = 1e-6 * max(1.0, abs(float(calc.x[column])))
        upper = calc.x.copy()
        lower = calc.x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (calc.get_f(upper) - calc.get_f(lower)) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[:, column],
            numeric,
            rtol=2e-6,
            atol=5e-8,
        )

def test_hybrid_multi_energy_lf_and_se_are_repeatable(tmp_path):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )
    assert calc.run() == 0
    first_lf_state = calc.x.copy()
    first_coupling_values = np.asarray(
        [calc.x[plan.state_col] for plan in calc.energy_coupling_plans],
        dtype=np.float64,
    )
    calc.prepare()
    assert calc.run() == 0
    np.testing.assert_allclose(calc.x, first_lf_state, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        [calc.x[plan.state_col] for plan in calc.energy_coupling_plans],
        first_coupling_values,
        rtol=1e-9,
        atol=1e-10,
    )

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator.run(result_mode="array", skip_bad_data=True, verbose=False)
    first_se_state = estimator.multi_energy_estimate_result.x.copy()
    estimator.prepare()
    estimator.run(result_mode="array", skip_bad_data=True, verbose=False)
    np.testing.assert_allclose(
        estimator.multi_energy_estimate_result.x,
        first_se_state,
        rtol=1e-9,
        atol=1e-10,
    )


def test_hybrid_multi_energy_bad_data_removal_keeps_global_layout(tmp_path):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._multi_energy_measurement_table.value[0] += 10.0
    estimator._multi_energy_z = estimator._multi_energy_measurement_table.value.copy()

    estimator.run(
        result_mode="array",
        remove_bad_data=True,
        bad_threshold=3.0,
        max_remove=1,
        skip_bad_data=False,
        verbose=False,
    )

    assert estimator.multi_energy_estimate_result.converged
    assert len(estimator.removed_bad_data) == 1
    assert estimator.removed_bad_data[0].row_pos == 0
    assert estimator.multi_energy_estimate_result.H.shape == (
        estimator.multi_energy_measurement_count - 1,
        estimator.multi_energy_state_count,
    )
    assert all(rc == 0 for rc in estimator.fluid_se_rc.values())


def test_hybrid_se_uses_one_global_wls_without_fluid_run(tmp_path, monkeypatch):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )

    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()

    expected_states = estimator.electric_state_count + sum(
        item.state_count for item in estimator.fluid_estimators.values()
    ) + (
        estimator.multi_energy_coupling_state_slice.stop
        - estimator.multi_energy_coupling_state_slice.start
    )
    expected_measurements = estimator.electric_measurement_count + sum(
        len(item.active_measurements) for item in estimator.fluid_estimators.values()
    ) + len(estimator.multi_energy_se_coupling_plans) + (
        estimator.multi_energy_control_measurement_slice.stop
        - estimator.multi_energy_control_measurement_slice.start
    )
    x0 = estimator._multi_energy_initial_state()
    z_est = estimator._evaluate_multi_energy(x0)
    jacobian = estimator._multi_energy_jacobian_sparse(x0)

    assert estimator.multi_energy_state_count == expected_states
    assert estimator.multi_energy_measurement_count == expected_measurements
    assert z_est.shape == (expected_measurements,)
    assert jacobian.shape == (expected_measurements, expected_states)
    assert jacobian[
        estimator.multi_energy_coupling_measurement_slice,
        :,
    ].nnz > 0

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("Hybrid SE must not start a separate fluid WLS loop")

    for fluid_estimator in estimator.fluid_estimators.values():
        monkeypatch.setattr(fluid_estimator, "run", fail_local_run)

    estimator.run(result_mode="summary", skip_bad_data=True, verbose=False)

    assert set(estimator.fluid_estimators) == {"heat", "gas", "hydro", "steam"}
    assert all(rc == 0 for rc in estimator.fluid_se_rc.values())
    assert all(item.result.converged for item in estimator.fluid_estimators.values())
    assert all(
        item.result.observability.observable
        for item in estimator.fluid_estimators.values()
    )
    assert not estimator.fluid_se_errors
    assert len(estimator.multi_energy_result.couplings) == 5
    assert {item.status for item in estimator.multi_energy_result.couplings} == {"balanced"}
    assert estimator.multi_energy_estimate_result.converged
    assert estimator.multi_energy_result.observable

    analytic = estimator._multi_energy_jacobian_sparse(
        estimator.multi_energy_estimate_result.x
    ).toarray()
    coupling_rows = range(
        estimator.multi_energy_coupling_measurement_slice.start,
        estimator.multi_energy_coupling_measurement_slice.stop,
    )
    checked_columns = set()
    for row in coupling_rows:
        checked_columns.update(np.flatnonzero(analytic[row]).tolist())
    for column in checked_columns:
        x = estimator.multi_energy_estimate_result.x
        step = 1e-6 * max(1.0, abs(float(x[column])))
        upper = x.copy()
        lower = x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (
            estimator._evaluate_multi_energy(upper)
            - estimator._evaluate_multi_energy(lower)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[:, column],
            numeric,
            rtol=2e-6,
            atol=5e-8,
        )


@pytest.mark.parametrize(
    ("name", "stem"),
    (
        ("heat", "heat_net_3"),
        ("gas", "gas_net_3"),
        ("hydro", "hydro_net_3"),
        ("steam", "steam_net_5"),
    ),
)
def test_hybrid_lf_and_se_support_pure_fluid_cases(name, stem):
    case_file = ROOT / "data" / "model" / name / f"{stem}.e"
    meas_file = ROOT / "data" / "meas" / name / f"{stem}.meas"

    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    assert calc.run() == 0
    assert calc.converged
    assert set(calc.fluid_calcs) == {name}
    assert calc.fluid_calcs[name].converged
    assert calc.lf_result is not None
    assert calc.lf_result.total_nodes == 0
    assert calc.lf_result.total_fluid_nodes > 0

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    assert estimator.run(result_mode="summary", skip_bad_data=True, verbose=False) is None
    assert estimator.estimate_result is estimator.multi_energy_estimate_result
    assert estimator.estimate_result.converged
    assert set(estimator.fluid_estimators) == {name}
    assert estimator.fluid_se_rc[name] == 0
    assert estimator.fluid_estimators[name].result.converged
    assert estimator.multi_energy_result.converged
    assert estimator.multi_energy_result.observable


def test_partition_sync_does_not_reactivate_hybrid_rejected_measurement():
    rows = [
        Measurement(1, "stale", "DCACConverter", "old_name", "V_DC", 1.0, False, 720.0),
        Measurement(2, "current", "DCACConverter", "conv_1", "V_DC", 1.0, True, 720.0),
    ]
    master = MeasurementList(rows)
    master.table = measurement_table_from_measurements(rows)
    partition_rows = [
        Measurement(1, "stale", "DCACConverter", "old_name", "V_DC", 1.0, True, 720.0),
        Measurement(2, "current", "DCACConverter", "conv_1", "V_DC", 1.0, True, 720.0),
    ]
    partition = MeasurementList(partition_rows)
    partition.table = measurement_table_from_measurements(partition_rows)

    estimator = HybridStateEstimator.__new__(HybridStateEstimator)
    estimator.measurements = master
    estimator._sub_measurement_source_rows_by_side = {
        "hybrid": [0, 1],
    }

    estimator._sync_measurement_table_from_partition_tables({"hybrid": partition})

    assert master.table.valid.tolist() == [False, True]
