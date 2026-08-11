from pathlib import Path

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
@ idx name run_stat idx_dc_unit_t1 idx_steam_load_t2 efficiency energy_factor
# 1 steam_generator 1 2 1 0.35 100.0
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
    assert {item.status for item in calc.coupling_results} <= {"balanced", "mismatch"}
    assert not calc.lf_result.fluid_errors


def test_hybrid_se_partitions_and_estimates_all_fluid_measurements(tmp_path):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )

    estimator.prepare()
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
    assert {item.status for item in estimator.multi_energy_result.couplings} == {"associated"}
    assert estimator.multi_energy_result.observable


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
    assert estimator.estimate_result is None
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
