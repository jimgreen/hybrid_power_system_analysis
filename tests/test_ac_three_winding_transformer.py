import copy
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT_DIR / "data" / "model" / "ac" / "ac_three_winding_transformer.e"


def _run_power_flow(ppc):
    from lfcore.ac_lf import ACPowerFlowCalc

    calc = ACPowerFlowCalc(ppc, tol=1e-10, max_iter=50, linear_solver="scipy")
    assert calc.run() == 0
    assert calc.converged
    return calc


def _explicit_star_ppc(ppc):
    from model.ac_array_model import (
        BUS_COLS,
        THREE_WINDING_TRANSFORMER_COLS,
        TRANSFORMER_COLS,
    )

    out = copy.deepcopy(ppc)
    out.pop("_topology_input", None)
    out.pop("_topology_arrays", None)
    out.pop("_pf_static", None)

    cols3 = THREE_WINDING_TRANSFORMER_COLS
    row3 = np.asarray(out["three_winding_transformer"], dtype=np.float64)[0]
    star_node = int(np.max(out["bus"][:, BUS_COLS["idx"]])) + 1
    star_bus = np.zeros((1, len(BUS_COLS)), dtype=np.float64)
    star_bus[0, BUS_COLS["idx"]] = star_node
    star_bus[0, BUS_COLS["vbase"]] = out["bus"][0, BUS_COLS["vbase"]]
    star_bus[0, BUS_COLS["voltage"]] = 1.0
    star_bus[0, BUS_COLS["angle"]] = 0.0
    star_bus[0, BUS_COLS["run_stat"]] = 1
    out["bus"] = np.vstack((out["bus"], star_bus))
    out["bus_name"] = np.append(np.asarray(out["bus_name"], dtype=object), "tr3_star")

    transformer = np.zeros((3, len(TRANSFORMER_COLS)), dtype=np.float64)
    for pos, terminal in enumerate(("i", "j", "k")):
        transformer[pos, TRANSFORMER_COLS["idx"]] = 100 + pos
        transformer[pos, TRANSFORMER_COLS["i_node"]] = row3[cols3[f"{terminal}_node"]]
        transformer[pos, TRANSFORMER_COLS["j_node"]] = star_node
        transformer[pos, TRANSFORMER_COLS["r"]] = row3[cols3[f"{terminal}_r"]]
        transformer[pos, TRANSFORMER_COLS["x"]] = row3[cols3[f"{terminal}_x"]]
        transformer[pos, TRANSFORMER_COLS["tap"]] = row3[cols3[f"{terminal}_tap"]]
        transformer[pos, TRANSFORMER_COLS["shift"]] = row3[cols3[f"{terminal}_shift"]]
        transformer[pos, TRANSFORMER_COLS["run_stat"]] = 1
    transformer[0, TRANSFORMER_COLS["gt"]] = row3[cols3["gt"]]
    transformer[0, TRANSFORMER_COLS["bt"]] = row3[cols3["bt"]]
    out["transformer"] = np.vstack((out["transformer"], transformer))
    out["transformer_name"] = np.append(
        np.asarray(out["transformer_name"], dtype=object),
        np.asarray(["tr3_i_star", "tr3_j_star", "tr3_k_star"], dtype=object),
    )
    out["three_winding_transformer"] = np.zeros((0, len(cols3)), dtype=np.float64)
    out["three_winding_transformer_name"] = np.asarray([], dtype=object)
    return out


def _three_terminal_measurements():
    from model import meas_type as mt
    from model.meas_model import Measurement, MeasurementList

    types = (
        "P_FROM",
        "Q_FROM",
        "V_FROM",
        "I_FROM",
        "P_TO",
        "Q_TO",
        "V_TO",
        "I_TO",
        "P_THIRD",
        "Q_THIRD",
        "V_THIRD",
        "I_THIRD",
    )
    return MeasurementList(
        [
            Measurement(
                pos + 1,
                f"tr3_{meas_type.lower()}",
                "ACThreeWindingTransformer",
                "tr3_main",
                meas_type,
                1.0,
                True,
                0.0,
                device_type_code=mt.DEVICE_TYPE_ACThreeWindingTransformer,
                meas_type_code=mt.MEAS_TYPE_CODES[meas_type],
            )
            for pos, meas_type in enumerate(types)
        ],
        normalized=True,
    )


def _measurements_from_power_flow(calc):
    from model import meas_type as mt
    from model.ac_array_model import (
        BUS_COLS,
        GEN_COLS,
        LOAD_COLS,
        THREE_WINDING_TRANSFORMER_COLS,
    )
    from model.meas_model import Measurement, MeasurementList

    rows = []

    def add(device_type, device_name, meas_type, value):
        rows.append(
            Measurement(
                len(rows) + 1,
                f"m_{len(rows) + 1}",
                device_type,
                device_name,
                meas_type,
                100.0,
                True,
                float(value),
                device_type_code=mt.DEVICE_TYPE_CODES[device_type],
                meas_type_code=mt.MEAS_TYPE_CODES[meas_type],
            )
        )

    for row, name in zip(calc.result["bus"], calc.ppc["bus_name"]):
        add("ACNode", str(name), "V", row[BUS_COLS["voltage"]])
        add("ACNode", str(name), "ANGLE", row[BUS_COLS["angle"]])
    for row, name in zip(calc.result["gen"], calc.ppc["gen_name"]):
        add("ACGenerator", str(name), "P_GEN", row[GEN_COLS["p"]])
        add("ACGenerator", str(name), "Q_GEN", row[GEN_COLS["q"]])
    for row, name in zip(calc.result["load"], calc.ppc["load_name"]):
        add("ACLoad", str(name), "P_LOAD", row[LOAD_COLS["p"]])
        add("ACLoad", str(name), "Q_LOAD", row[LOAD_COLS["q"]])
    cols = THREE_WINDING_TRANSFORMER_COLS
    row = calc.result["three_winding_transformer"][0]
    for meas_type, attr in (
        ("P_FROM", "i_p"),
        ("Q_FROM", "i_q"),
        ("V_FROM", None),
        ("I_FROM", "i_c"),
        ("P_TO", "j_p"),
        ("Q_TO", "j_q"),
        ("V_TO", None),
        ("I_TO", "j_c"),
        ("P_THIRD", "k_p"),
        ("Q_THIRD", "k_q"),
        ("V_THIRD", None),
        ("I_THIRD", "k_c"),
    ):
        if attr is None:
            terminal = {"V_FROM": "i", "V_TO": "j", "V_THIRD": "k"}[meas_type]
            node_idx = int(row[cols[f"{terminal}_node"]])
            bus_row = next(item for item in calc.result["bus"] if int(item[BUS_COLS["idx"]]) == node_idx)
            value = bus_row[BUS_COLS["voltage"]]
        else:
            value = row[cols[attr]]
        add("ACThreeWindingTransformer", "tr3_main", meas_type, value)
    return MeasurementList(rows, normalized=True)


def test_three_winding_stamp_matches_explicit_star_kron_reduction():
    from lfcore.ac_lf import three_winding_transformer_stamp_vectorized

    y = three_winding_transformer_stamp_vectorized(
        [0.002],
        [0.04],
        [0.003],
        [0.05],
        [0.004],
        [0.06],
        gt=[0.0001],
        bt=[-0.001],
        i_tap=[1.02],
        i_shift=[2.0],
        j_tap=[1.0],
        j_shift=[0.0],
        k_tap=[0.98],
        k_shift=[-1.0],
    )[0]

    z = np.asarray([0.002 + 0.04j, 0.003 + 0.05j, 0.004 + 0.06j])
    winding_y = 1.0 / z
    tap = np.asarray([1.02 * np.exp(1j * np.deg2rad(2.0)), 1.0, 0.98 * np.exp(-1j * np.deg2rad(1.0))])
    full = np.zeros((4, 4), dtype=np.complex128)
    for terminal in range(3):
        full[terminal, terminal] += winding_y[terminal] / abs(tap[terminal]) ** 2
        full[terminal, 3] -= winding_y[terminal] / np.conj(tap[terminal])
        full[3, terminal] -= winding_y[terminal] / tap[terminal]
        full[3, 3] += winding_y[terminal]
    full[0, 0] += (0.0001 - 0.001j) / abs(tap[0]) ** 2
    expected = full[:3, :3] - np.outer(full[:3, 3], full[3, :3]) / full[3, 3]

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_three_winding_stamp_preserves_single_zero_star_arm_limit():
    from lfcore.ac_lf import three_winding_transformer_stamp_vectorized

    z_j = 0.003 + 0.05j
    z_k = 0.004 + 0.06j
    y_j = 1.0 / z_j
    y_k = 1.0 / z_k
    tap = np.asarray(
        [
            1.02 * np.exp(1j * np.deg2rad(2.0)),
            1.0,
            0.98 * np.exp(-1j * np.deg2rad(1.0)),
        ]
    )
    actual = three_winding_transformer_stamp_vectorized(
        [0.0],
        [0.0],
        [z_j.real],
        [z_j.imag],
        [z_k.real],
        [z_k.imag],
        gt=[0.0001],
        bt=[-0.001],
        i_tap=[abs(tap[0])],
        i_shift=[np.degrees(np.angle(tap[0]))],
        j_tap=[abs(tap[1])],
        j_shift=[np.degrees(np.angle(tap[1]))],
        k_tap=[abs(tap[2])],
        k_shift=[np.degrees(np.angle(tap[2]))],
    )[0]

    expected = np.zeros((3, 3), dtype=np.complex128)
    for terminal, admittance in ((1, y_j), (2, y_k)):
        expected[0, 0] += admittance / abs(tap[0]) ** 2
        expected[terminal, terminal] += admittance / abs(tap[terminal]) ** 2
        expected[0, terminal] -= admittance / (np.conj(tap[0]) * tap[terminal])
        expected[terminal, 0] -= admittance / (np.conj(tap[terminal]) * tap[0])
    expected[0, 0] += (0.0001 - 0.001j) / abs(tap[0]) ** 2

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_three_winding_stamp_rejects_multiple_zero_star_arms():
    import pytest

    from lfcore.ac_lf import three_winding_transformer_stamp_vectorized

    with pytest.raises(ValueError, match="more than one zero-impedance winding"):
        three_winding_transformer_stamp_vectorized(
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.004],
            [0.06],
        )


def test_pairwise_impedance_alias_imports_into_array_and_object_models(tmp_path):
    from model.ac_array_model import THREE_WINDING_TRANSFORMER_COLS, build_ac_ppc_from_e_file
    from model.ac_model import ACPowerNetwork

    case_file = tmp_path / "pairwise_three_winding.e"
    case_file.write_text(
        """<PowerBase>
@ p_base u_unit p_unit i_unit
# 100 kV MW kA
</PowerBase>
<ACNode>
@ idx name vbase voltage angle isl run_stat
# 1 i 110 110 0 0 1
# 2 j 110 110 0 0 1
# 3 k 110 110 0 0 1
</ACNode>
<ACGenerator>
@ idx name node control_type p_set q_set v_set alpha run_stat
# 1 slack 1 V 0 0 110 1 1
</ACGenerator>
<AC3WTransformer>
@ idx name i_node j_node k_node ij_r ij_x ik_r ik_x jk_r jk_x tap_i shift_i tap_j shift_j tap_k shift_k run_stat
# 1 pairwise_tr3 1 2 3 0.003 0.09 0.004 0.10 0.005 0.11 1.01 1.0 1.0 0.0 0.99 -1.0 1
</AC3WTransformer>
""",
        encoding="utf-8",
    )

    ppc = build_ac_ppc_from_e_file(case_file)
    cols = THREE_WINDING_TRANSFORMER_COLS
    row = ppc["three_winding_transformer"][0]
    expected = (0.001, 0.04, 0.002, 0.05, 0.003, 0.06)
    actual = tuple(row[cols[name]] for name in ("i_r", "i_x", "j_r", "j_x", "k_r", "k_x"))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        [row[cols["i_tap"]], row[cols["i_shift"]], row[cols["k_tap"]], row[cols["k_shift"]]],
        [1.01, 1.0, 0.99, -1.0],
    )

    network = ACPowerNetwork()
    network.read_from_file(case_file)
    transformer = network.three_winding_transformers[0]
    np.testing.assert_allclose(
        [
            transformer.i_r,
            transformer.i_x,
            transformer.j_r,
            transformer.j_x,
            transformer.k_r,
            transformer.k_x,
        ],
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [transformer.i_tap, transformer.i_shift, transformer.k_tap, transformer.k_shift],
        [1.01, 1.0, 0.99, -1.0],
    )


def test_three_winding_topology_connects_all_terminals_and_requires_all_alive():
    from model.ac_array_model import BUS_COLS, build_ac_network_from_ppc, build_ac_ppc_from_e_file
    from model.topology import build_ac_topology_input_ppc, prepare_ac_topology_ppc

    ppc = build_ac_ppc_from_e_file(CASE_FILE)
    topology = prepare_ac_topology_ppc(ppc)
    device = topology.devices["three_winding_transformer"]
    assert device.alive_mask.tolist() == [True]
    assert len({int(device.i_island_pos[0]), int(device.j_island_pos[0]), int(device.k_island_pos[0])}) == 1
    assert len({int(device.i_bus_pos[0]), int(device.j_bus_pos[0]), int(device.k_bus_pos[0])}) == 3

    offline = copy.deepcopy(ppc)
    offline.pop("_topology_arrays", None)
    offline["bus"][2, BUS_COLS["run_stat"]] = 0
    offline["_topology_input"] = build_ac_topology_input_ppc(offline)
    offline_topology = prepare_ac_topology_ppc(offline)
    assert offline_topology.devices["three_winding_transformer"].alive_mask.tolist() == [False]
    assert int(offline_topology.node_to_island_pos[0]) != int(offline_topology.node_to_island_pos[1])

    offline_network = build_ac_network_from_ppc(offline)
    offline_network.topo()
    assert offline_network.node_dict[1].isl_obj is not offline_network.node_dict[2].isl_obj


def test_three_winding_network_ppc_roundtrip_preserves_parameters_names_and_results():
    from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file, build_ac_ppc_from_network

    ppc = build_ac_ppc_from_e_file(CASE_FILE)
    network = build_ac_network_from_ppc(ppc)
    transformer = network.three_winding_transformers[0]
    for pos, attr in enumerate(("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"), start=1):
        setattr(transformer, attr, pos / 10.0)

    roundtrip = build_ac_network_from_ppc(build_ac_ppc_from_network(network))
    restored = roundtrip.three_winding_transformers[0]
    assert restored.name == "tr3_main"
    for attr in (
        "i_node",
        "j_node",
        "k_node",
        "i_r",
        "i_x",
        "j_r",
        "j_x",
        "k_r",
        "k_x",
        "gt",
        "bt",
        "i_tap",
        "i_shift",
        "j_tap",
        "j_shift",
        "k_tap",
        "k_shift",
        "run_stat",
        "i_p",
        "i_q",
        "i_c",
        "j_p",
        "j_q",
        "j_c",
        "k_p",
        "k_q",
        "k_c",
    ):
        assert getattr(restored, attr) == getattr(transformer, attr)


def test_three_winding_load_flow_matches_explicit_star_and_returns_all_terminal_results():
    from model.ac_array_model import BUS_COLS, build_ac_ppc_from_e_file

    direct_ppc = build_ac_ppc_from_e_file(CASE_FILE)
    direct = _run_power_flow(direct_ppc)
    explicit = _run_power_flow(_explicit_star_ppc(direct_ppc))

    direct_bus = {int(row[BUS_COLS["idx"]]): row for row in direct.result["bus"]}
    explicit_bus = {int(row[BUS_COLS["idx"]]): row for row in explicit.result["bus"]}
    for node_idx in (1, 2, 3):
        np.testing.assert_allclose(
            direct_bus[node_idx][[BUS_COLS["voltage"], BUS_COLS["angle"]]],
            explicit_bus[node_idx][[BUS_COLS["voltage"], BUS_COLS["angle"]]],
            rtol=1e-9,
            atol=1e-10,
        )

    row = direct.result["three_winding_transformer"][0]
    assert np.isfinite(row).all()
    result = direct.lf_result
    terminal_result = result.three_winding_transformers["tr3_main"]
    for attr in ("i_p", "i_q", "i_c", "i_v", "j_p", "j_q", "j_c", "j_v", "k_p", "k_q", "k_c", "k_v"):
        assert np.isfinite(getattr(terminal_result, attr))


def test_ac_load_flow_text_output_includes_three_winding_transformer(capsys):
    from lfcore.ac_lf import print_ac_result
    from model.ac_array_model import build_ac_ppc_from_e_file

    calc = _run_power_flow(build_ac_ppc_from_e_file(CASE_FILE))
    print_ac_result(calc, 0)
    output = capsys.readouterr().out

    assert "三绕组主变 1 tr3_main" in output
    assert "三绕组主变:" in output


def test_matpower_export_adds_one_internal_bus_and_three_winding_branches():
    from model.ac_array_model import (
        MP_BR_R,
        MP_BR_X,
        MP_F_BUS,
        MP_T_BUS,
        THREE_WINDING_TRANSFORMER_COLS,
        build_ac_ppc_from_e_file,
        build_matpower_ppc_from_ac_ppc,
    )

    ppc = build_ac_ppc_from_e_file(CASE_FILE)
    exported = build_matpower_ppc_from_ac_ppc(ppc)
    assert exported["bus"].shape[0] == ppc["bus"].shape[0] + 1
    assert exported["branch"].shape[0] == 3
    star_bus = int(exported["bus"][-1, 0])
    assert set(exported["branch"][:, MP_F_BUS].astype(int)) == {1, 2, 3}
    assert set(exported["branch"][:, MP_T_BUS].astype(int)) == {star_bus}
    cols = THREE_WINDING_TRANSFORMER_COLS
    source = ppc["three_winding_transformer"][0]
    np.testing.assert_allclose(exported["branch"][:, MP_BR_R], [source[cols["i_r"]], source[cols["j_r"]], source[cols["k_r"]]])
    np.testing.assert_allclose(exported["branch"][:, MP_BR_X], [source[cols["i_x"]], source[cols["j_x"]], source[cols["k_x"]]])


def test_matpower_export_eliminates_single_zero_star_arm_without_zero_branch():
    from lfcore.ac_lf import matpower_transformer_stamp_vectorized, three_winding_transformer_stamp_vectorized
    from model.ac_array_model import (
        MP_BR_R,
        MP_BR_X,
        MP_F_BUS,
        MP_GS,
        MP_BS,
        MP_SHIFT,
        MP_TAP,
        MP_T_BUS,
        THREE_WINDING_TRANSFORMER_COLS,
        build_ac_ppc_from_e_file,
        build_matpower_ppc_from_ac_ppc,
    )

    ppc = build_ac_ppc_from_e_file(CASE_FILE)
    cols = THREE_WINDING_TRANSFORMER_COLS
    source = ppc["three_winding_transformer"][0]
    source[cols["i_r"]] = 0.0
    source[cols["i_x"]] = 0.0
    exported = build_matpower_ppc_from_ac_ppc(ppc)

    assert exported["bus"].shape[0] == ppc["bus"].shape[0]
    assert exported["branch"].shape[0] == 2
    assert np.all(np.hypot(exported["branch"][:, MP_BR_R], exported["branch"][:, MP_BR_X]) > 1e-12)

    exported_y = np.zeros((3, 3), dtype=np.complex128)
    for row in exported["branch"]:
        tap = row[MP_TAP] if abs(row[MP_TAP]) > 1e-12 else 1.0
        yff, yft, ytf, ytt = matpower_transformer_stamp_vectorized(
            [row[MP_BR_R]],
            [row[MP_BR_X]],
            tap=[tap],
            shift=[row[MP_SHIFT]],
        )
        i = int(row[MP_F_BUS]) - 1
        j = int(row[MP_T_BUS]) - 1
        exported_y[i, i] += yff[0]
        exported_y[i, j] += yft[0]
        exported_y[j, i] += ytf[0]
        exported_y[j, j] += ytt[0]
    exported_y[np.arange(3), np.arange(3)] += (
        exported["bus"][:, MP_GS] + 1j * exported["bus"][:, MP_BS]
    ) / exported["baseMVA"]

    direct_y = three_winding_transformer_stamp_vectorized(
        [source[cols["i_r"]]],
        [source[cols["i_x"]]],
        [source[cols["j_r"]]],
        [source[cols["j_x"]]],
        [source[cols["k_r"]]],
        [source[cols["k_x"]]],
        gt=[source[cols["gt"]]],
        bt=[source[cols["bt"]]],
        i_tap=[source[cols["i_tap"]]],
        i_shift=[source[cols["i_shift"]]],
        j_tap=[source[cols["j_tap"]]],
        j_shift=[source[cols["j_shift"]]],
        k_tap=[source[cols["k_tap"]]],
        k_shift=[source[cols["k_shift"]]],
    )[0]
    np.testing.assert_allclose(exported_y, direct_y, rtol=1e-12, atol=1e-12)


def test_offline_three_winding_transformer_clears_stale_terminal_results():
    from model.ac_array_model import BUS_COLS, LOAD_COLS, THREE_WINDING_TRANSFORMER_COLS, build_ac_ppc_from_e_file
    from model.topology import build_ac_topology_input_ppc

    ppc = build_ac_ppc_from_e_file(CASE_FILE)
    cols = THREE_WINDING_TRANSFORMER_COLS
    result_attrs = ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c")
    ppc["three_winding_transformer"][0, [cols[attr] for attr in result_attrs]] = 9.0
    # Keep a loaded slack island alive so this test reaches stale-result cleanup.
    ppc["load"][0, LOAD_COLS["node"]] = ppc["bus"][0, BUS_COLS["idx"]]
    ppc["bus"][2, BUS_COLS["run_stat"]] = 0
    ppc.pop("_topology_arrays", None)
    ppc["_topology_input"] = build_ac_topology_input_ppc(ppc)

    calc = _run_power_flow(ppc)
    result_row = calc.result["three_winding_transformer"][0]
    np.testing.assert_array_equal(result_row[[cols[attr] for attr in result_attrs]], np.zeros(len(result_attrs)))


def test_ac_state_estimation_supports_all_three_terminal_measurements_and_jacobian():
    from model import meas_type as mt
    from secore.ac_se import ACStateEstimator

    measurements = _three_terminal_measurements()
    estimator = ACStateEstimator(
        e_file=CASE_FILE,
        measurements=measurements,
        flat_start=True,
        prepare_active_measurements=False,
    )
    estimator.prepare(prepare_active_measurements=False)
    x = estimator.initial_state()
    values = estimator.evaluate(x, estimator.measurements)
    theta, voltage = estimator._unpack_state(x)
    voltage_complex = voltage * np.exp(1j * theta)
    terminal_pos = np.asarray(
        [
            estimator._ac_three_winding_transformer_plan_i[0],
            estimator._ac_three_winding_transformer_plan_j[0],
            estimator._ac_three_winding_transformer_plan_k[0],
        ],
        dtype=np.int64,
    )
    terminal_voltage = voltage_complex[terminal_pos]
    terminal_current = estimator._ac_three_winding_transformer_plan_y[0] @ terminal_voltage
    terminal_power = terminal_voltage * np.conj(terminal_current)
    expected_by_code = {
        mt.MEAS_TYPE_P_FROM: terminal_power[0].real,
        mt.MEAS_TYPE_Q_FROM: terminal_power[0].imag,
        mt.MEAS_TYPE_V_FROM: voltage[terminal_pos[0]],
        mt.MEAS_TYPE_I_FROM: abs(terminal_current[0]),
        mt.MEAS_TYPE_P_TO: terminal_power[1].real,
        mt.MEAS_TYPE_Q_TO: terminal_power[1].imag,
        mt.MEAS_TYPE_V_TO: voltage[terminal_pos[1]],
        mt.MEAS_TYPE_I_TO: abs(terminal_current[1]),
        mt.MEAS_TYPE_P_THIRD: terminal_power[2].real,
        mt.MEAS_TYPE_Q_THIRD: terminal_power[2].imag,
        mt.MEAS_TYPE_V_THIRD: voltage[terminal_pos[2]],
        mt.MEAS_TYPE_I_THIRD: abs(terminal_current[2]),
    }
    expected = np.asarray(
        [expected_by_code[int(code)] for code in estimator.measurement_table.meas_type_code],
        dtype=np.float64,
    )
    np.testing.assert_allclose(values, expected, rtol=1e-11, atol=1e-11)

    analytic = estimator.jacobian_sparse(x, estimator.measurements).toarray()
    numeric = np.zeros_like(analytic)
    for col in range(estimator.n_state):
        step = 1e-6 * max(1.0, abs(x[col]))
        xp = x.copy()
        xm = x.copy()
        xp[col] += step
        xm[col] -= step
        numeric[:, col] = (
            estimator.evaluate(xp, estimator.measurements)
            - estimator.evaluate(xm, estimator.measurements)
        ) / (2.0 * step)
    assert float(np.max(np.abs(analytic - numeric))) < 2e-5


def test_ac_state_estimation_applies_current_policy_to_third_terminal(tmp_path):
    from model import meas_type as mt
    from secore.ac_se import ACStateEstimator

    meas_file = tmp_path / "three_winding_current.meas"
    meas_file.write_text(
        """<Measurement>
@ idx name dev_type dev_name meas_type weight valid value
# 1 i_from  ACThreeWindingTransformer tr3_main I_FROM  1 1 0.1
# 2 i_to    ACThreeWindingTransformer tr3_main I_TO    1 1 0.1
# 3 i_third ACThreeWindingTransformer tr3_main I_THIRD 1 1 0.1
</Measurement>
""",
        encoding="utf-8",
    )
    estimator = ACStateEstimator(e_file=CASE_FILE, meas_file=meas_file, flat_start=True)
    estimator.prepare(prepare_active_measurements=False)

    table = estimator.measurement_table
    current_rows = np.isin(
        table.meas_type_code,
        (mt.MEAS_TYPE_I_FROM, mt.MEAS_TYPE_I_TO, mt.MEAS_TYPE_I_THIRD),
    )
    assert current_rows.tolist() == [True, True, True]
    assert table.valid[current_rows].tolist() == [False, False, False]


def test_ac_state_estimation_scales_third_terminal_named_units(tmp_path):
    from secore.ac_se import ACStateEstimator

    meas_file = tmp_path / "three_winding_named_units.meas"
    meas_file.write_text(
        """<Measurement>
@ idx name dev_type dev_name meas_type weight valid value
# 1 p_third ACThreeWindingTransformer tr3_main P_THIRD 1 1  10
# 2 q_third ACThreeWindingTransformer tr3_main Q_THIRD 1 1   5
# 3 v_third ACThreeWindingTransformer tr3_main V_THIRD 1 1 110
</Measurement>
""",
        encoding="utf-8",
    )
    estimator = ACStateEstimator(e_file=CASE_FILE, meas_file=meas_file, flat_start=True)
    estimator.prepare(prepare_active_measurements=False)

    np.testing.assert_allclose(estimator.measurement_table.value, [0.1, 0.05, 1.0])
    assert estimator.measurement_table.valid.tolist() == [True, True, True]


def test_ac_state_estimation_includes_three_winding_pseudo_candidates_and_incident_degree():
    from model import meas_type as mt
    from model.meas_model import Measurement, MeasurementList
    from secore.ac_se import ACStateEstimator

    disabled = Measurement(
        1,
        "disabled",
        "ACNode",
        "bus_i",
        "V",
        1.0,
        False,
        1.0,
        device_type_code=mt.DEVICE_TYPE_ACNode,
        meas_type_code=mt.MEAS_TYPE_V,
    )
    estimator = ACStateEstimator(
        e_file=CASE_FILE,
        measurements=MeasurementList([disabled], normalized=True),
        flat_start=True,
        prepare_active_measurements=False,
    )
    estimator.prepare(prepare_active_measurements=False)
    candidates = estimator._observability_pseudo_candidate_measurements()
    table = candidates.table
    rows = table.device_type_code == mt.DEVICE_TYPE_ACThreeWindingTransformer
    assert set(table.meas_type_code[rows].tolist()) == {
        mt.MEAS_TYPE_P_FROM,
        mt.MEAS_TYPE_Q_FROM,
        mt.MEAS_TYPE_P_TO,
        mt.MEAS_TYPE_Q_TO,
        mt.MEAS_TYPE_P_THIRD,
        mt.MEAS_TYPE_Q_THIRD,
    }
    np.testing.assert_array_equal(estimator.node_degree_array, np.ones(3, dtype=np.int32))


def test_ac_state_estimation_converges_with_three_winding_measurements():
    from model.ac_array_model import build_ac_ppc_from_e_file
    from secore.ac_se import ACStateEstimator

    truth = _run_power_flow(build_ac_ppc_from_e_file(CASE_FILE))
    estimator = ACStateEstimator(
        e_file=CASE_FILE,
        measurements=_measurements_from_power_flow(truth),
        flat_start=True,
        max_iter=20,
    )
    estimator.prepare()
    result = estimator.estimate(final_diagnostics=False)
    assert result.converged
    assert result.observability.observable
    assert result.residual_inf < 1e-7


def test_ac_state_estimation_power_flow_seed_overlay_keeps_three_winding_results():
    from secore.ac_se import ACStateEstimator

    seed = {"three_winding_transformer": np.zeros((1, 1), dtype=np.float64)}
    replacement = np.ones((1, 1), dtype=np.float64)
    overlaid = ACStateEstimator._overlay_power_flow_seed_result_ppc(
        seed,
        {"three_winding_transformer": replacement},
    )
    assert overlaid["three_winding_transformer"] is replacement


def test_hybrid_state_estimation_routes_three_winding_measurements_to_ac_side():
    from model import meas_type as mt
    from secore.hybrid_se import HybridStateEstimator

    assert HybridStateEstimator._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE[
        mt.DEVICE_TYPE_ACThreeWindingTransformer
    ] == "ac"
    assert "ACThreeWindingTransformer" in HybridStateEstimator._AC_MEASUREMENT_DEVICE_TYPES
    assert "AC3WTransformer" in HybridStateEstimator._AC_MEASUREMENT_DEVICE_TYPES
    assert "V_THIRD" in HybridStateEstimator._VOLTAGE_MEASUREMENT_TYPES


def test_hybrid_state_estimation_runs_with_third_terminal_measurements(tmp_path):
    from model.ac_array_model import BUS_COLS, THREE_WINDING_TRANSFORMER_COLS, build_ac_ppc_from_e_file
    from secore.hybrid_se import HybridStateEstimator

    truth = _run_power_flow(build_ac_ppc_from_e_file(CASE_FILE))
    cols = THREE_WINDING_TRANSFORMER_COLS
    transformer = truth.result["three_winding_transformer"][0]
    k_node = int(transformer[cols["k_node"]])
    k_bus = next(row for row in truth.result["bus"] if int(row[BUS_COLS["idx"]]) == k_node)
    meas_file = tmp_path / "hybrid_three_winding.meas"
    meas_file.write_text(
        """<Measurement>
@ idx name dev_type dev_name meas_type weight valid value
# 1 p_third ACThreeWindingTransformer tr3_main P_THIRD 100 1 {p_third:.12g}
# 2 q_third ACThreeWindingTransformer tr3_main Q_THIRD 100 1 {q_third:.12g}
# 3 v_third ACThreeWindingTransformer tr3_main V_THIRD 100 1 {v_third:.12g}
</Measurement>
""".format(
            p_third=100.0 * transformer[cols["k_p"]],
            q_third=100.0 * transformer[cols["k_q"]],
            v_third=110.0 * k_bus[BUS_COLS["voltage"]],
        ),
        encoding="utf-8",
    )

    estimator = HybridStateEstimator(
        e_file=CASE_FILE,
        meas_file=meas_file,
        flat_start=True,
        max_iter=20,
    )
    estimator.prepare()
    result = estimator.estimate(final_diagnostics=False)

    assert result.converged
    assert result.observability.observable
    assert float(np.max(np.abs(result.residual[:3]))) < 2e-7


def test_sparse_jacobian_fixed_refresh_ignores_all_negative_column_chunks():
    from secore.se_math import SparseJacobianBuilder

    builder = SparseJacobianBuilder((2, 3))
    builder._assume_fixed_pattern = True

    def fill(first_value, last_values):
        builder.add_many(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([first_value], dtype=np.float64),
        )
        builder.add_many(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([-1, -1], dtype=np.int32),
            np.asarray([99.0, 99.0], dtype=np.float64),
        )
        builder.add_many(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 2], dtype=np.int32),
            np.asarray(last_values, dtype=np.float64),
        )
        return builder.to_csr()

    first = fill(1.0, [2.0, 3.0]).toarray()
    builder.reset()
    second = fill(10.0, [20.0, 30.0]).toarray()

    np.testing.assert_allclose(first, [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(second, [[10.0, 20.0, 0.0], [0.0, 0.0, 30.0]])


def test_hybrid_model_roundtrip_exposes_three_winding_transformer_collection():
    from efile_read import efile_factory_from_file
    from model.hybrid_array_model import build_hybrid_ppc_from_model

    source_model = efile_factory_from_file(CASE_FILE)
    network, _ppc = build_hybrid_ppc_from_model(CASE_FILE, source_model)

    assert network.ACThreeWindingTransformer is network.ac.three_winding_transformers
    assert [device.name for device in network.ACThreeWindingTransformer] == ["tr3_main"]


def test_hybrid_topology_reference_count_includes_three_winding_terminals():
    from model.ac_array_model import build_ac_network_from_ppc, build_ac_ppc_from_e_file
    from model.dc_model import DCPowerNetwork
    from model.hybrid_model import HybridPowerNetwork

    ac_network = build_ac_network_from_ppc(build_ac_ppc_from_e_file(CASE_FILE))
    ac_network.topo()
    network = HybridPowerNetwork(ac_network, DCPowerNetwork(), [], [])

    assert network._ac_node_ref_count() == {1: 2, 2: 2, 3: 2}


def test_hybrid_load_flow_lightweight_facade_exposes_three_winding_results():
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file

    network = _read_lf_network_from_file(CASE_FILE)
    assert len(network.ac.three_winding_transformers) == 1
    source_device = network.ac.three_winding_transformers[0]
    assert isinstance(source_device.k_node, int)
    assert source_device.k_node_obj.idx == 3

    calc = HybridPowerFlowCalc(network, linear_solver="scipy", tol=1e-10, max_iter=50)
    assert calc.run() == 0
    result_device = network.ac.three_winding_transformers[0]
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"):
        assert np.isfinite(getattr(result_device, attr))


def test_hybrid_load_flow_writes_three_winding_results_to_full_model():
    from lfcore.hybrid_lf import HybridPowerFlowCalc
    from model.hybrid_model import HybridPowerNetwork

    network = HybridPowerNetwork.read_from_file(CASE_FILE)
    calc = HybridPowerFlowCalc(network, linear_solver="scipy", tol=1e-10, max_iter=50)
    assert calc.run() == 0

    result_device = network.ac.three_winding_transformers[0]
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"):
        assert np.isfinite(getattr(result_device, attr))


def test_hybrid_load_flow_marks_offline_three_winding_transformer_dead_and_clears_results():
    from lfcore.hybrid_lf import HybridPowerFlowCalc
    from model.ac_array_model import BUS_COLS, THREE_WINDING_TRANSFORMER_COLS
    from model.hybrid_model import HybridPowerNetwork
    from model.topology import build_ac_topology_input_ppc

    network = HybridPowerNetwork.read_from_file(CASE_FILE)
    ppc = network._ac_ppc
    cols = THREE_WINDING_TRANSFORMER_COLS
    result_attrs = ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c")
    ppc["three_winding_transformer"][0, [cols[attr] for attr in result_attrs]] = 9.0
    ppc["bus"][2, BUS_COLS["run_stat"]] = 0
    ppc.pop("_topology_arrays", None)
    ppc["_topology_input"] = build_ac_topology_input_ppc(ppc)

    calc = HybridPowerFlowCalc(network, linear_solver="scipy", tol=1e-10, max_iter=50)
    assert calc.run() == 0

    result_device = network.ac.three_winding_transformers[0]
    assert result_device.is_alive is False
    np.testing.assert_array_equal(
        [getattr(result_device, attr) for attr in result_attrs],
        np.zeros(len(result_attrs)),
    )


def test_hybrid_load_flow_lightweight_facade_marks_offline_three_winding_transformer_dead():
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
    from model.ac_array_model import BUS_COLS, THREE_WINDING_TRANSFORMER_COLS
    from model.topology import build_ac_topology_input_ppc

    network = _read_lf_network_from_file(CASE_FILE)
    ppc = network._ac_ppc
    cols = THREE_WINDING_TRANSFORMER_COLS
    result_attrs = ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c")
    ppc["three_winding_transformer"][0, [cols[attr] for attr in result_attrs]] = 9.0
    ppc["bus"][2, BUS_COLS["run_stat"]] = 0
    ppc.pop("_topology_arrays", None)
    ppc["_topology_input"] = build_ac_topology_input_ppc(ppc)

    calc = HybridPowerFlowCalc(network, linear_solver="scipy", tol=1e-10, max_iter=50)
    assert calc.run() == 0

    result_device = network.ac.three_winding_transformers[0]
    assert result_device.is_alive is False
    np.testing.assert_array_equal(
        [getattr(result_device, attr) for attr in result_attrs],
        np.zeros(len(result_attrs)),
    )
