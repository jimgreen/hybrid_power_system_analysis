from pathlib import Path
import re

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
@ idx name run_stat control_type idx_ac_load_t1 idx_h2_unit_t2 efficiency energy_factor
# 1 electrolyzer 1 MONITOR 2 1 0.75 100.0
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


def _replace_model_block(text: str, block_name: str, replacement: str) -> str:
    pattern = rf"<{block_name}>.*?</{block_name}>"
    updated, count = re.subn(pattern, replacement.strip(), text, count=1, flags=re.DOTALL)
    assert count == 1
    return updated


def _build_direct_hydrogen_conversion_case(
    tmp_path: Path,
    *,
    table_name: str,
    control_type: str | None,
    hydro_reference_node: bool = False,
):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    text = case_file.read_text(encoding="utf8")
    text = _replace_model_block(
        text,
        "HydroSource",
        f"""
<HydroSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max run_stat
# 1 hydro_source 1 PRESSURE 3.00 0.0 1.0 0.0 2.0 1
# 2 electrolyzer_h2 {1 if hydro_reference_node else 3} FLOW 2.98 0.4 1.0 0.0 10.0 1
</HydroSource>
""",
    )
    if table_name in {"AcE2Hydro", "DcE2Hydro"}:
        control_header = " control_type" if control_type is not None else ""
        control_value = f" {control_type}" if control_type is not None else ""
        electric_reference = (
            "idx_ac_load_t1" if table_name == "AcE2Hydro" else "idx_dc_load_t1"
        )
        electric_idx = 7 if table_name == "AcE2Hydro" else 16
        replacement = f"""
<{table_name}>
@ idx name run_stat{control_header} {electric_reference} idx_h2_unit_t2 e2h_coeff
# 1 direct_electrolyzer 1{control_value} {electric_idx} 2 0.02
</{table_name}>
"""
        if table_name == "AcE2Hydro":
            text = _replace_model_block(text, table_name, replacement)
        else:
            text += f"\n{replacement}"
    else:
        if hydro_reference_node:
            text = _replace_model_block(
                text,
                "HydroLoad",
                """
<HydroLoad>
@ idx name node flow_set run_stat
# 1 hydro_load_2 2 0.10 1
# 2 hydro_load_4 4 0.15 1
# 3 fuel_cell_h2 1 0.25 1
</HydroLoad>
""",
            )
        control_header = " control_type" if control_type is not None else ""
        control_value = f" {control_type}" if control_type is not None else ""
        electric_reference = (
            "idx_ac_unit_t1" if table_name == "Hydro2AcE" else "idx_dc_unit_t1"
        )
        electric_idx = 4 if table_name == "Hydro2AcE" else 3
        text += f"""

<{table_name}>
@ idx name run_stat{control_header} {electric_reference} idx_h2_load_t2 h2e_coeff
# 1 direct_fuel_cell 1{control_value} {electric_idx} 3 600.0
</{table_name}>
"""
    case_file.write_text(text, encoding="utf8")
    return case_file, meas_file


def _build_direct_gas_or_steam_conversion_case(
    tmp_path: Path,
    *,
    domain: str,
    control_type: str,
):
    electric_model = ROOT / "data" / "model" / "ac" / "ac_net_10.e"
    fluid_model = (
        ROOT
        / "data"
        / "model"
        / domain
        / ("gas_net_3.e" if domain == "gas" else "steam_net_5.e")
    )
    text = "\n\n".join(
        (
            electric_model.read_text(encoding="utf8"),
            fluid_model.read_text(encoding="utf8"),
        )
    )
    if domain == "gas":
        table_name = "Gas2AcE"
        fluid_field = "idx_gas_load_t1"
        coefficient_field = "g2e_coeff"
        coefficient = 10.0
        generator_power = 2.0
    else:
        table_name = "Steam2AcE"
        fluid_field = "idx_steam_load_t1"
        electric_field = "idx_ac_unit_t2"
        coefficient_field = "s2e_coeff"
        coefficient = 0.35
        generator_power = 50.0
    if domain == "gas":
        electric_field = "idx_ac_unit_t2"
        power_pattern = r"(#\s+4\s+gen_pq9\s+10\s+PQ\s+)50(?=\s+10)"
    else:
        power_pattern = r"(#\s+4\s+gen_pq9\s+10\s+PQ\s+)50(?=\s+10)"
    text, count = re.subn(power_pattern, rf"\g<1>{generator_power}", text, count=1)
    assert count == 1
    text = _replace_model_block(
        text + f"\n\n<{table_name}></{table_name}>\n",
        table_name,
        f"""
<{table_name}>
@ idx name run_stat control_type {fluid_field} {electric_field} {coefficient_field}
# 1 direct_{domain}_generator 1 {control_type} 1 4 {coefficient}
</{table_name}>
""",
    )
    case_file = tmp_path / f"direct_{domain}_{control_type.lower()}.e"
    case_file.write_text(text, encoding="utf8")

    measurement_paths = (
        ROOT / "data" / "meas" / "hybrid" / "hybrid_net_40.meas",
        ROOT
        / "data"
        / "meas"
        / domain
        / ("gas_net_3.meas" if domain == "gas" else "steam_net_5.meas"),
    )
    rows = []
    for measurement_path in measurement_paths:
        for row in _measurement_rows(measurement_path):
            if measurement_path.name != "hybrid_net_40.meas" or row[2].startswith("AC"):
                rows.append(row)
    meas_file = tmp_path / f"direct_{domain}_{control_type.lower()}.meas"
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


def _build_direct_gas_heat_case(tmp_path: Path, *, control_type: str):
    model_paths = (
        ROOT / "data" / "model" / "gas" / "gas_net_3.e",
        ROOT / "data" / "model" / "heat" / "heat_net_3.e",
    )
    coupling = f"""
<Gas2Heat>
@ idx name run_stat control_type idx_gas_load_t1 idx_heat_unit_t2 g2h_coeff
# 1 direct_gas_boiler 1 {control_type} 1 1 1700.0
</Gas2Heat>
""".strip()
    case_file = tmp_path / f"direct_gas_heat_{control_type.lower()}.e"
    case_file.write_text(
        "\n\n".join(path.read_text(encoding="utf8") for path in model_paths)
        + "\n\n"
        + coupling
        + "\n",
        encoding="utf8",
    )

    rows = []
    for domain, stem in (("gas", "gas_net_3"), ("heat", "heat_net_3")):
        rows.extend(
            _measurement_rows(ROOT / "data" / "meas" / domain / f"{stem}.meas")
        )
    meas_file = tmp_path / f"direct_gas_heat_{control_type.lower()}.meas"
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


def _build_direct_electric_heat_case(
    tmp_path: Path,
    *,
    table_name: str,
    control_type: str,
    supply_temperature_set: float = 90.0,
    e2h_coeff: float = 2.0,
):
    model_parts = [
        (ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e").read_text(
            encoding="utf8"
        ),
        (ROOT / "data" / "model" / "heat" / "heat_net_3.e").read_text(
            encoding="utf8"
        ),
    ]
    electric_text = model_parts[0]
    if table_name.startswith("AcE2Heat"):
        electric_text, count = re.subn(
            r"(#\s+1\s+load_3\s+4\s+)1\.0",
            r"\g<1>100.0",
            electric_text,
            count=1,
        )
    else:
        electric_text, count = re.subn(
            r"(#\s+1\s+load_1\s+1\s+)1\.0",
            r"\g<1>100.0",
            electric_text,
            count=1,
        )
    assert count == 1
    model_parts[0] = electric_text
    heat_text = model_parts[1]
    heat_text = _replace_model_block(
        heat_text,
        "HeatSource",
        f"""
<HeatSource>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature_set run_stat
# 1 heat_source 1 PRESSURE 10.0 0.0 1.0 0.0 5.0 {supply_temperature_set} 1
</HeatSource>
""",
    )
    model_parts[1] = heat_text
    endpoint_field = (
        "idx_ac_load_t1"
        if table_name.startswith("AcE2Heat")
        else "idx_dc_load_t1"
    )
    coupling = f"""
<{table_name}>
@ idx name run_stat control_type {endpoint_field} idx_heat_unit_t2 e2h_coeff
# 1 direct_electric_heater 1 {control_type} 1 1 {e2h_coeff}
</{table_name}>
""".strip()
    case_file = tmp_path / f"{table_name.lower()}_{control_type.lower()}.e"
    case_file.write_text("\n\n".join((*model_parts, coupling)) + "\n", encoding="utf8")

    rows = []
    for measurement_file in (
        ROOT / "data" / "meas" / "hybrid" / "hybrid_net_40.meas",
        ROOT / "data" / "meas" / "heat" / "heat_net_3.meas",
    ):
        rows.extend(_measurement_rows(measurement_file))
    meas_file = tmp_path / f"{table_name.lower()}_{control_type.lower()}.meas"
    output = [
        "<Measurement>",
        "@ idx name dev_type dev_name meas_type weight valid value",
    ]
    for idx, row in enumerate(rows, start=1):
        row[0] = str(idx)
        output.append("# " + " ".join(row))
    output.append(
        f"# {len(rows) + 1} heat_source_t_supply HeatSource heat_source "
        f"T_SUPPLY 10 1 {supply_temperature_set}"
    )
    output.append("</Measurement>")
    meas_file.write_text("\n".join(output) + "\n", encoding="utf8")
    return case_file, meas_file


def _build_explicit_return_electric_heat_case(
    tmp_path: Path,
    *,
    control_type: str,
) -> Path:
    electric_text = (
        ROOT / "data" / "model" / "hybrid" / "hybrid_net_40.e"
    ).read_text(encoding="utf8")
    electric_text, count = re.subn(
        r"(#\s+1\s+load_3\s+4\s+)1\.0",
        r"\g<1>10.0",
        electric_text,
        count=1,
    )
    assert count == 1

    heat_text = (
        ROOT / "data" / "model" / "heat" / "heat_explicit_return.e"
    ).read_text(encoding="utf8")
    heat_text = heat_text.replace(
        "flow_min flow_max supply_temperature run_stat",
        "flow_min flow_max supply_temperature_set run_stat",
    )
    heat_text, count = re.subn(
        r"(?m)^(#\s+\d+\s+\S+\s+\d+\s+\d+\s+1\.0)\s+0\.0\s+1$",
        r"\1 0.1 1",
        heat_text,
    )
    assert count == 4

    coupling = f"""
<AcE2Heat>
@ idx name run_stat control_type idx_ac_load_t1 idx_heat_unit_t2 e2h_coeff
# 1 explicit_return_heater 1 {control_type} 1 1 2.0
</AcE2Heat>
""".strip()
    case_file = tmp_path / f"explicit_return_electric_heat_{control_type.lower()}.e"
    case_file.write_text(
        "\n\n".join((electric_text, heat_text, coupling)) + "\n",
        encoding="utf8",
    )
    return case_file


def _build_direct_electric_heat_storage_case(
    tmp_path: Path,
    *,
    table_name: str,
    control_type: str,
):
    case_file, meas_file = _build_direct_electric_heat_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
        supply_temperature_set=85.0,
        e2h_coeff=2.0,
    )
    text = case_file.read_text(encoding="utf8")
    text += """

<HeatStorage>
@ idx name node control_type pressure_set flow_set alpha flow_min flow_max supply_temperature_set return_temperature_set run_stat
# 1 thermal_buffer 1 FLOW 10.0 0.2 1.0 -0.5 0.5 85.0 50.0 1
</HeatStorage>
"""
    electric_field = (
        "idx_ac_load_t1" if table_name.startswith("AcE2Heat") else "idx_dc_load_t1"
    )
    text = _replace_model_block(
        text,
        table_name,
        f"""
<{table_name}>
@ idx name run_stat control_type {electric_field} idx_heat_storage_t2 e2h_coeff
# 1 storage_electric_heater 1 {control_type} 1 1 2.0
</{table_name}>
""",
    )
    case_file.write_text(text, encoding="utf8")
    measurement_text = meas_file.read_text(encoding="utf8")
    measurement_rows = _measurement_rows(meas_file)
    storage_measurements = (
        ("storage_flow", "FLOW", 0.2),
        ("storage_t_supply", "T_SUPPLY", 85.0),
        ("storage_t_return", "T_RETURN", 50.0),
        ("storage_heat", "HEAT", 0.0),
    )
    added_rows = [
        (
            f"# {len(measurement_rows) + offset} {name} HeatStorage "
            f"thermal_buffer {meas_type} 10 1 {value}"
        )
        for offset, (name, meas_type, value) in enumerate(
            storage_measurements,
            start=1,
        )
    ]
    measurement_text = measurement_text.replace(
        "</Measurement>",
        "\n".join((*added_rows, "</Measurement>")),
    )
    meas_file.write_text(measurement_text, encoding="utf8")
    return case_file, meas_file


@pytest.mark.parametrize(
    "table_name",
    ("AcE2Heat", "AcE2Heat2", "DcE2Heat", "DcE2Heat2"),
)
@pytest.mark.parametrize("control_type", ("P", "T_OUT"))
def test_direct_electric_heat_control_uses_e2h_coeff_and_sparse_jacobian(
    tmp_path,
    table_name,
    control_type,
):
    case_file, _meas_file = _build_direct_electric_heat_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
        supply_temperature_set=90.0,
        e2h_coeff=2.0,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    plan = next(
        item
        for item in calc.energy_coupling_plans
        if item.coupling.name == "direct_electric_heater"
    )
    result = next(
        item for item in calc.coupling_results if item.name == "direct_electric_heater"
    )
    assert plan.electric_heat
    assert plan.coupling.e2h_coeff == pytest.approx(2.0)
    assert plan.coupling.control_type == control_type
    assert plan.adjustable_kind == ("temperature" if control_type == "P" else "power")
    assert result.status == "balanced"
    assert abs(float(result.residual)) < 1.0e-8
    assert calc.fluid_calcs["heat"].network.source_supply_temperature_set[0] == pytest.approx(90.0)

    analytic = calc.get_jacobi(calc.x).toarray()
    checked_columns = set(np.flatnonzero(analytic[plan.eq_row]).tolist())
    checked_columns.add(plan.state_col)
    for column in checked_columns:
        step = 1.0e-6 * max(1.0, abs(float(calc.x[column])))
        upper = calc.x.copy()
        lower = calc.x.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (calc.get_f(upper) - calc.get_f(lower)) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[:, column],
            numeric,
            rtol=3.0e-6,
            atol=2.0e-7,
        )


def test_direct_electric_heat_p_control_keeps_inactive_supply_temperature_set(tmp_path):
    case_file, _meas_file = _build_direct_electric_heat_case(
        tmp_path,
        table_name="AcE2Heat",
        control_type="P",
        supply_temperature_set=72.0,
        e2h_coeff=2.0,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    calc.prepare()
    plan = next(item for item in calc.energy_coupling_plans if item.coupling.is_electric_heat_control)
    assert calc.run() == 0
    assert calc.fluid_calcs["heat"].network.source_supply_temperature_set[0] == pytest.approx(72.0)
    assert float(calc.x[plan.state_col]) != pytest.approx(72.0)


@pytest.mark.parametrize("control_type", ("P", "T_OUT"))
def test_direct_electric_heat_supports_explicit_return_source(tmp_path, control_type):
    case_file = _build_explicit_return_electric_heat_case(
        tmp_path,
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    calc.prepare()

    heat_calc = calc.fluid_calcs["heat"]
    initial_group_flow = heat_calc.x[
        heat_calc.base_pressure_source_group_flow : heat_calc.hydraulic_state_count
    ]
    np.testing.assert_allclose(initial_group_flow, [1.0], rtol=0.0, atol=1.0e-12)

    assert calc.run() == 0
    plan = next(item for item in calc.energy_coupling_plans if item.electric_heat)
    result = next(
        item for item in calc.coupling_results if item.name == "explicit_return_heater"
    )
    assert result.status == "balanced"
    assert abs(float(result.residual)) < 1.0e-8
    assert heat_calc.network.source_supply_temperature_set[0] == pytest.approx(90.0)
    if control_type == "P":
        assert calc.x[plan.supply_temperature_col] == pytest.approx(
            calc.x[plan.state_col],
            abs=1.0e-8,
        )
    else:
        assert calc.x[plan.supply_temperature_col] == pytest.approx(90.0, abs=1.0e-8)


@pytest.mark.parametrize(
    "table_name",
    ("AcE2Heat", "AcE2Heat2", "DcE2Heat", "DcE2Heat2"),
)
@pytest.mark.parametrize("control_type", ("P", "T_OUT"))
def test_direct_electric_heat_accepts_heat_storage_endpoint(
    tmp_path,
    table_name,
    control_type,
):
    case_file, meas_file = _build_direct_electric_heat_storage_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    plan = next(item for item in calc.energy_coupling_plans if item.electric_heat)
    result = next(
        item for item in calc.coupling_results if item.name == "storage_electric_heater"
    )
    assert plan.heat.kind == "storage"
    assert plan.heat.device_type == "HeatStorage"
    assert result.status == "balanced"
    assert abs(float(result.residual)) < 1.0e-8
    assert "thermal_buffer" in calc.fluid_calcs["heat"].lf_result.storages
    assert "thermal_buffer" not in calc.fluid_calcs["heat"].lf_result.sources

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()
    se_plan = next(
        item
        for item in estimator.multi_energy_se_coupling_plans
        if item.coupling.name == "storage_electric_heater"
    )
    assert se_plan.heat_flow.device_type == "HeatStorage"
    assert se_plan.heat_temperature.device_type == "HeatStorage"


@pytest.mark.parametrize(
    "table_name",
    ("AcE2Heat", "AcE2Heat2", "DcE2Heat", "DcE2Heat2"),
)
@pytest.mark.parametrize("control_type", ("P", "T_OUT"))
def test_hybrid_se_uses_coupled_heat_storage_outlet_for_heat_measurement(
    tmp_path,
    table_name,
    control_type,
):
    case_file, meas_file = _build_direct_electric_heat_storage_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
    )
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()
    plan = next(
        item
        for item in estimator.multi_energy_se_coupling_plans
        if item.coupling.name == "storage_electric_heater"
    )
    state = estimator._multi_energy_initial_state()
    if control_type == "P":
        state[plan.control_col] += 3.0

    values = estimator._evaluate_multi_energy(state)
    endpoint_values, _ = estimator._multi_energy_endpoint_runtime(
        state,
        with_jacobian=False,
    )
    source_flow = float(endpoint_values[plan.heat_flow.provider][plan.heat_flow.row])
    outlet_temperature = (
        float(state[plan.control_col])
        if control_type == "P"
        else float(plan.supply_temperature_set)
    )
    return_temperature = float(state[plan.return_temperature_col])
    expected_heat = (
        source_flow
        * float(plan.heat_capacity)
        * (outlet_temperature - return_temperature)
    )

    assert plan.heat_power_measurement_rows
    for row in plan.heat_power_measurement_rows:
        assert values[row] == pytest.approx(expected_heat, rel=1.0e-10, abs=1.0e-12)

    analytic = estimator._multi_energy_jacobian_sparse(state)
    checked_columns = {plan.return_temperature_col}
    if control_type == "P":
        checked_columns.add(plan.control_col)
    heat_rows = np.asarray(plan.heat_power_measurement_rows, dtype=np.int64)
    for column in checked_columns:
        step = 1.0e-6 * max(1.0, abs(float(state[column])))
        upper = state.copy()
        lower = state.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (
            estimator._evaluate_multi_energy(upper)
            - estimator._evaluate_multi_energy(lower)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[heat_rows, column].toarray().ravel(),
            numeric[heat_rows],
            rtol=3.0e-6,
            atol=2.0e-7,
        )


@pytest.mark.parametrize("table_name", ("AcE2Heat", "DcE2Heat"))
@pytest.mark.parametrize("control_type", ("P", "T_OUT"))
def test_hybrid_se_maps_direct_electric_heat_control_rows_and_jacobian(
    tmp_path,
    table_name,
    control_type,
):
    case_file, meas_file = _build_direct_electric_heat_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
        supply_temperature_set=90.0,
        e2h_coeff=2.0,
    )
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()
    plan = next(
        item
        for item in estimator.multi_energy_se_coupling_plans
        if item.coupling.name == "direct_electric_heater"
    )

    assert plan.electric_heat
    assert plan.control_col >= 0
    assert plan.controlled_measurement_rows
    assert plan.dependent_measurement_rows
    state = estimator._multi_energy_initial_state()
    analytic = estimator._multi_energy_jacobian_sparse(state)
    checked_rows = (
        plan.controlled_measurement_rows
        + plan.dependent_measurement_rows
        + (plan.measurement_row,)
    )
    # Hydraulic pressure states use a square-root flow law and are non-smooth at
    # the flat-start zero-drop point. Their delegated Jacobian is covered by the
    # fluid estimator tests; verify the new coupling and temperature columns here.
    checked_columns = {plan.control_col, plan.return_temperature_col}
    for column in checked_columns:
        step = 1.0e-6 * max(1.0, abs(float(state[column])))
        upper = state.copy()
        lower = state.copy()
        upper[column] += step
        lower[column] -= step
        numeric = (
            estimator._evaluate_multi_energy(upper)
            - estimator._evaluate_multi_energy(lower)
        ) / (2.0 * step)
        row_index = np.asarray(checked_rows, dtype=np.int64)
        np.testing.assert_allclose(
            analytic[row_index, column].toarray().ravel(),
            numeric[row_index],
            rtol=3.0e-6,
            atol=2.0e-7,
        )
    for row in plan.controlled_measurement_rows:
        assert analytic.getrow(row).nnz == 0
    for row in plan.dependent_measurement_rows:
        jacobian_row = analytic.getrow(row)
        np.testing.assert_array_equal(jacobian_row.indices, [plan.control_col])
        np.testing.assert_allclose(jacobian_row.data, [1.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("table_name", "control_type", "adjustable_domain"),
    (
        ("AcE2Hydro", "P", "hydro"),
        ("AcE2Hydro", "FLOW", "ac"),
        ("DcE2Hydro", "P", "hydro"),
        ("DcE2Hydro", "FLOW", "dc"),
        ("Hydro2AcE", "P", "hydro"),
        ("Hydro2AcE", "FLOW", "ac"),
        ("Hydro2DcE", "P", "hydro"),
        ("Hydro2DcE", "FLOW", "dc"),
    ),
)
def test_direct_hydrogen_control_supports_pressure_reference_node_endpoint(
    tmp_path,
    table_name,
    control_type,
    adjustable_domain,
):
    case_file, _meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name=table_name,
        control_type=control_type,
        hydro_reference_node=True,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    result = next(
        item
        for item in calc.coupling_results
        if item.name in {"direct_electrolyzer", "direct_fuel_cell"}
    )
    plan = next(item for item in calc.energy_coupling_plans if item.coupling is result.coupling)
    assert plan.adjustable.domain == adjustable_domain
    assert result.status == "balanced"
    electric_value = result.t1_value
    hydrogen_flow = result.t2_value
    if result.coupling.t2.domain in {"ac", "dc"}:
        electric_value, hydrogen_flow = hydrogen_flow, electric_value
    electric_kw = float(electric_value) * float(calc.network.p_base_kW)
    if table_name in {"AcE2Hydro", "DcE2Hydro"}:
        assert hydrogen_flow == pytest.approx(electric_kw * 0.02)
    else:
        assert electric_kw == pytest.approx(hydrogen_flow * 600.0)


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


@pytest.mark.parametrize(
    ("control_type", "adjustable_domain"),
    (("P", "hydro"), ("FLOW", "ac")),
)
def test_electrolyzer_control_mode_selects_endpoint_and_e2h_coeff(
    tmp_path,
    control_type,
    adjustable_domain,
):
    case_file, _meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name="AcE2Hydro",
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    plan = next(
        item
        for item in calc.energy_coupling_plans
        if item.coupling.name == "direct_electrolyzer"
    )
    result = next(
        item for item in calc.coupling_results if item.name == "direct_electrolyzer"
    )
    assert plan.adjustable.domain == adjustable_domain
    electric_kw = float(result.t1_value) * float(calc.network.p_base_kW)
    hydrogen_flow = float(result.t2_value)
    assert hydrogen_flow == pytest.approx(0.02 * electric_kw, rel=1e-7, abs=1e-9)
    if control_type == "FLOW":
        assert hydrogen_flow == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("control_type", "adjustable_domain"),
    (("P", "hydro"), ("FLOW", "dc")),
)
def test_fuel_cell_control_mode_selects_endpoint_and_h2e_coeff(
    tmp_path,
    control_type,
    adjustable_domain,
):
    case_file, _meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name="Hydro2DcE",
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    plan = next(
        item
        for item in calc.energy_coupling_plans
        if item.coupling.name == "direct_fuel_cell"
    )
    result = next(
        item for item in calc.coupling_results if item.name == "direct_fuel_cell"
    )
    assert plan.adjustable.domain == adjustable_domain
    electric_kw = float(result.t1_value) * float(calc.network.p_base_kW)
    hydrogen_flow = float(result.t2_value)
    assert electric_kw == pytest.approx(600.0 * hydrogen_flow, rel=1e-7, abs=1e-9)
    if control_type == "FLOW":
        assert hydrogen_flow == pytest.approx(0.25)


@pytest.mark.parametrize("domain", ("gas", "steam"))
@pytest.mark.parametrize("control_type", ("P", "FLOW"))
def test_gas_and_steam_electric_controls_join_lf_newton_system(
    tmp_path,
    domain,
    control_type,
):
    case_file, _meas_file = _build_direct_gas_or_steam_conversion_case(
        tmp_path,
        domain=domain,
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )

    assert calc.run() == 0
    plan = next(
        item
        for item in calc.energy_coupling_plans
        if item.coupling.name == f"direct_{domain}_generator"
    )
    result = next(
        item
        for item in calc.coupling_results
        if item.name == f"direct_{domain}_generator"
    )
    assert plan.dependent.domain == (domain if control_type == "P" else "ac")
    assert result.status == "balanced"
    assert abs(float(result.residual)) < 1.0e-8

    jacobian = calc.get_jacobi(calc.x).tocsr()
    column = int(plan.state_col)
    step = 1.0e-6 * max(1.0, abs(float(calc.x[column])))
    upper = calc.x.copy()
    lower = calc.x.copy()
    upper[column] += step
    lower[column] -= step
    finite_difference = (calc.get_f(upper) - calc.get_f(lower)) / (2.0 * step)
    np.testing.assert_allclose(
        jacobian.getcol(column).toarray().ravel(),
        finite_difference,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    if domain == "steam":
        assert jacobian[plan.eq_row, plan.steam_enthalpy_col] != 0.0


@pytest.mark.parametrize("domain", ("gas", "steam"))
@pytest.mark.parametrize("control_type", ("P", "FLOW"))
def test_gas_and_steam_electric_controls_join_se_wls(
    tmp_path,
    domain,
    control_type,
):
    case_file, meas_file = _build_direct_gas_or_steam_conversion_case(
        tmp_path,
        domain=domain,
        control_type=control_type,
    )
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()
    plan = next(
        item
        for item in estimator.multi_energy_se_coupling_plans
        if item.coupling.name == f"direct_{domain}_generator"
    )
    state = estimator._multi_energy_initial_state()
    analytic = estimator._multi_energy_jacobian_sparse(state)
    step = 1.0e-6 * max(1.0, abs(float(state[plan.control_col])))
    upper = state.copy()
    lower = state.copy()
    upper[plan.control_col] += step
    lower[plan.control_col] -= step
    numeric = (
        estimator._evaluate_multi_energy(upper)
        - estimator._evaluate_multi_energy(lower)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic.getcol(plan.control_col).toarray().ravel(),
        numeric,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    if domain == "steam":
        assert analytic[plan.measurement_row, plan.steam_enthalpy_col] != 0.0

    estimator.run(result_mode="array", skip_bad_data=True, verbose=False)
    assert estimator.multi_energy_estimate_result.converged
    result = next(
        item
        for item in estimator.multi_energy_result.couplings
        if item.name == f"direct_{domain}_generator"
    )
    assert result.status == "balanced"


@pytest.mark.parametrize("control_type", ("FLOW", "T_OUT"))
def test_gas_heat_control_uses_thermal_power_in_lf_and_se(tmp_path, control_type):
    case_file, meas_file = _build_direct_gas_heat_case(
        tmp_path,
        control_type=control_type,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="full",
        linear_solver="scipy",
        verbose=False,
    )
    assert calc.run() == 0
    plan = next(item for item in calc.energy_coupling_plans if item.gas_heat)
    result = next(
        item for item in calc.coupling_results if item.name == "direct_gas_boiler"
    )
    assert result.status == "balanced"
    assert abs(float(result.residual)) < 1.0e-8
    jacobian = calc.get_jacobi(calc.x).tocsr()
    assert jacobian[plan.eq_row, plan.state_col] != 0.0
    assert jacobian[plan.eq_row, plan.return_temperature_col] != 0.0

    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator.run(result_mode="array", skip_bad_data=True, verbose=False)
    assert estimator.multi_energy_estimate_result.converged
    se_plan = next(
        item for item in estimator.multi_energy_se_coupling_plans if item.gas_heat
    )
    assert estimator.multi_energy_estimate_result.H[
        se_plan.measurement_row,
        se_plan.return_temperature_col,
    ] != 0.0
    se_result = next(
        item
        for item in estimator.multi_energy_result.couplings
        if item.name == "direct_gas_boiler"
    )
    assert se_result.status == "balanced"


@pytest.mark.parametrize(
    ("table_name", "expected_control", "coefficient_field"),
    (
        ("AcE2Hydro", "FLOW", "e2h_coeff"),
        ("DcE2Hydro", "FLOW", "e2h_coeff"),
        ("Hydro2AcE", "P", "h2e_coeff"),
        ("Hydro2DcE", "P", "h2e_coeff"),
    ),
)
def test_hydrogen_conversion_uses_direction_specific_default_control(
    tmp_path,
    table_name,
    expected_control,
    coefficient_field,
):
    case_file, _meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name=table_name,
        control_type=None,
    )
    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )

    calc.prepare()
    coupling = next(
        item
        for item in calc.multi_energy.couplings
        if item.table_name == table_name and item.is_hydrogen_electric_control
    )
    assert coupling.control_type == expected_control
    assert getattr(coupling, coefficient_field) is not None
    other_field = "h2e_coeff" if coefficient_field == "e2h_coeff" else "e2h_coeff"
    assert getattr(coupling, other_field) is None


@pytest.mark.parametrize(
    ("table_name", "coefficient_field"),
    (("AcE2Hydro", "e2h_coeff"), ("Hydro2DcE", "h2e_coeff")),
)
def test_direct_hydrogen_control_does_not_fallback_to_generic_efficiency(
    tmp_path,
    table_name,
    coefficient_field,
):
    case_file, _meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name=table_name,
        control_type="P",
    )
    text = case_file.read_text(encoding="utf8")
    pattern = rf"(<{table_name}>.*?</{table_name}>)"
    block = re.search(pattern, text, flags=re.DOTALL)
    assert block is not None
    legacy_block = block.group(1).replace(coefficient_field, "efficiency", 1)
    case_file.write_text(
        text[: block.start()] + legacy_block + text[block.end() :],
        encoding="utf8",
    )

    calc = HybridPowerFlowCalc.from_file_fast(
        case_file,
        result_mode="array",
        linear_solver="scipy",
        verbose=False,
    )
    calc.prepare()
    coupling = next(
        item
        for item in calc.multi_energy.couplings
        if item.table_name == table_name and item.name.startswith("direct_")
    )

    assert getattr(coupling, coefficient_field) == 0.0
    assert not coupling.supports_energy_balance
    assert all(plan.coupling is not coupling for plan in calc.energy_coupling_plans)
    assert any(
        coefficient_field in warning
        for warning in calc.multi_energy.warnings
    )


def test_hybrid_se_maps_flow_controlled_fuel_cell_power_row_to_coupling_state(
    tmp_path,
):
    case_file, meas_file = _build_direct_hydrogen_conversion_case(
        tmp_path,
        table_name="Hydro2DcE",
        control_type="FLOW",
    )
    estimator = HybridStateEstimator(
        e_file=case_file,
        meas_file=meas_file,
        flat_start=True,
        auto_prepare=False,
    )
    estimator.prepare()
    estimator._prepare_multi_energy_se_layout()
    plan = next(
        item
        for item in estimator.multi_energy_se_coupling_plans
        if item.coupling.name == "direct_fuel_cell"
    )

    assert plan.control_col >= 0
    assert plan.dependent_measurement_rows
    state = estimator._multi_energy_initial_state()
    analytic = estimator._multi_energy_jacobian_sparse(state)
    step = 1.0e-6
    upper = state.copy()
    lower = state.copy()
    upper[plan.control_col] += step
    lower[plan.control_col] -= step
    numeric = (
        estimator._evaluate_multi_energy(upper)
        - estimator._evaluate_multi_energy(lower)
    ) / (2.0 * step)

    for row in plan.dependent_measurement_rows:
        jacobian_row = analytic.getrow(row)
        np.testing.assert_array_equal(jacobian_row.indices, [plan.control_col])
        np.testing.assert_allclose(jacobian_row.data, [1.0], rtol=0.0, atol=0.0)
        assert numeric[row] == pytest.approx(1.0, rel=1.0e-9, abs=1.0e-9)
    for row in plan.controlled_measurement_rows:
        assert analytic.getrow(row).nnz == 0
        assert numeric[row] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)


def test_hybrid_storage_endpoint_keeps_storage_identity_and_source_balance(tmp_path):
    case_file, meas_file = _build_multi_energy_case(tmp_path)
    storage_blocks = """
<HydroStorage>
@ idx name node control_type flow_set alpha flow_min flow_max run_stat
# 1 hydrogen_buffer 3 FLOW 0.0 1.0 -2.0 2.0 1
</HydroStorage>

<Hydro2DcE>
@ idx name run_stat control_type idx_h2_storage_t1 idx_dc_unit_t2 efficiency energy_factor
# 1 hydrogen_buffer_supply 1 MONITOR 1 2 1.0 100.0
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
