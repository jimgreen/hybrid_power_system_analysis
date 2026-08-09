from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _build_ac_multi_balance_ppc():
    from ac_array_model import build_ac_ppc_from_efile_rows
    from model.ppc_topology import ensure_ac_ppc_topology

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "ac_balance_sharing", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "bus_1", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_min p_max q_min q_max run_stat",
            [
                [1, "slack_1", 1, "PH", 20, 10, 380, 1, 0, 80, -50, 80, 1],
                [2, "slack_2", 1, "PH", 10, 20, 380, 2, 0, 45, -20, 60, 1],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load_1", 1, 100, 1, 0, 0, 100, 1, 0, 0, 1]],
        ),
    }
    return ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("ac_balance_sharing.e"), rows))


def _build_dc_multi_balance_ppc():
    from dc_array_model import build_dc_ppc_from_efile_rows
    from model.ppc_topology import ensure_dc_ppc_topology

    rows = {
        "PowerBase": _table(
            "p_base u_unit p_unit i_unit",
            [[100, "V", "kW", "A"]],
        ),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [[1, "bus_1", 100, 100, 0, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type p_set v_set i_set p_min p_max alpha run_stat",
            [
                [1, "v_1", 1, "V", 20, 100, 0, 0, 80, 1, 1],
                [2, "v_2", 1, "V", 10, 100, 0, 0, 45, 2, 1],
            ],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "load_1", 1, 100, 1, 0, 0, 1]],
        ),
    }
    return ensure_dc_ppc_topology(build_dc_ppc_from_efile_rows(Path("dc_balance_sharing.e"), rows))


def _build_hybrid_dcac_balance_network(
    dev_type="DCACConverter",
    p_ac_set=30,
    *,
    ac_control_type="PQ",
    dc_control_type="NONE",
    p_dc_set=0,
    q_ac_set=0,
    r1=0,
    r2=0,
):
    from lfcore.hybrid_lf import _build_lf_network_from_hybrid_rows

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "hybrid_dcac_balance", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "ac_bus", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_min p_max q_min q_max run_stat",
            [[1, "ac_slack", 1, "PH", 0, 0, 380, 1, -100, 100, -100, 100, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "ac_load", 1, 10, 1, 0, 0, 0, 1, 0, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [[1, "dc_bus", 750, 750, 0, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set p_max p_min i_set alpha run_stat",
            [[1, "dc_slack", 1, "V", 750, 0, 100, -100, 0, 1, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "dc_load", 1, 20, 1, 0, 0, 1]],
        ),
        "DCACConverter": _table(
            "idx name dev_type ac_node dc_node ac_control_type dc_control_type "
            "p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat r1 r2",
            [[
                1,
                "converter",
                dev_type,
                1,
                1,
                ac_control_type,
                dc_control_type,
                p_ac_set,
                p_dc_set,
                q_ac_set,
                380,
                750,
                1,
                r1,
                r2,
            ]],
        ),
    }
    return _build_lf_network_from_hybrid_rows(Path("hybrid_dcac_balance.e"), rows)


def test_dcac_legacy_combined_control_column_is_rejected():
    from hybrid_array_model import build_hybrid_ppc_from_e_file

    source = ROOT_DIR / "data" / "model" / "hybrid" / "qinling.e"
    lines = []
    in_dcac = False
    ac_control_index = -1
    dc_control_index = -1
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "<DCACConverter>":
            in_dcac = True
        elif stripped == "</DCACConverter>":
            in_dcac = False
        elif in_dcac and stripped.startswith("@"):
            parts = line.split()
            ac_control_index = parts.index("ac_control_type")
            dc_control_index = parts.index("dc_control_type")
            parts[ac_control_index] = "control_type"
            parts.pop(dc_control_index)
            line = " ".join(parts)
        elif in_dcac and stripped.startswith("#"):
            parts = line.split()
            parts[ac_control_index] = "ACP"
            parts.pop(dc_control_index)
            line = " ".join(parts)
        lines.append(line)

    with tempfile.TemporaryDirectory() as temporary:
        legacy_file = Path(temporary) / "legacy_dcac.e"
        legacy_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="control_type"):
            build_hybrid_ppc_from_e_file(legacy_file)


@pytest.mark.parametrize(
    ("ac_control_type", "dc_control_type", "expected_ac_code", "expected_dc_code"),
    (
        ("PQ", "NONE", 1, 0),
        ("PQ", "V", 1, 2),
        ("NONE", "P", 0, 1),
    ),
)
def test_hybrid_dcac_kernel_keeps_separate_terminal_control_codes(
    ac_control_type,
    dc_control_type,
    expected_ac_code,
    expected_dc_code,
):
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    network = _build_hybrid_dcac_balance_network(
        ac_control_type=ac_control_type,
        dc_control_type=dc_control_type,
        p_ac_set=30,
        p_dc_set=30,
    )
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        verbose=False,
    )

    calc.prepare()

    assert not hasattr(calc, "dcac_ctrl_code")
    np.testing.assert_array_equal(calc.dcac_ac_control_code, [expected_ac_code])
    np.testing.assert_array_equal(calc.dcac_dc_control_code, [expected_dc_code])


def test_ac_balance_devices_apply_p_set_then_limited_alpha_sharing():
    from ac_array_model import GEN_COLS
    from ac_lf import ACPowerFlowCalc

    ppc = _build_ac_multi_balance_ppc()
    calc = ACPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")
    assert calc.run() == 0
    assert calc.converged

    gen = calc.result["gen"]
    np.testing.assert_allclose(
        gen[:, GEN_COLS["p"]],
        [0.55, 0.45],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gen[:, GEN_COLS["q"]],
        [0.5454545454545454, 0.4545454545454546],
        rtol=0.0,
        atol=1e-12,
    )


def test_dc_balance_devices_apply_p_set_then_limited_alpha_sharing():
    from dc_array_model import GEN_COLS
    from dc_lf import DCPowerFlowCalc

    ppc = _build_dc_multi_balance_ppc()
    calc = DCPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")
    assert calc.run() == 0
    assert calc.converged

    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [0.55, 0.45],
        rtol=0.0,
        atol=1e-12,
    )


def test_ac_balance_devices_exceed_limits_to_finish_unallocated_residual():
    from ac_array_model import GEN_COLS, LOAD_COLS
    from ac_lf import ACPowerFlowCalc

    ppc = _build_ac_multi_balance_ppc()
    ppc["load"][0, LOAD_COLS["pbase"]] = 2.0
    ppc["load"][0, LOAD_COLS["qbase"]] = 2.0
    calc = ACPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [1.05, 0.95],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["q"]],
        [1.1, 0.9],
        rtol=0.0,
        atol=1e-12,
    )


def test_ac_voltage_control_generator_and_shunt_share_reactive_balance():
    from ac_array_model import GEN_COLS, SHUNT_COLS, build_ac_ppc_from_efile_rows
    from ac_lf import ACPowerFlowCalc
    from model.ppc_topology import ensure_ac_ppc_topology

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "ac_shunt_q_sharing", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [
                [1, "bus_1", 380, 380, 0, 1],
                [2, "bus_2", 380, 380, 0, 1],
            ],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "slack", 1, "PH", 0, 20, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load", 1, 0, 1, 0, 0, 100, 1, 0, 0, 1]],
        ),
        "ACShuntCompensator": _table(
            "idx name node control_type q_set g_set b_set v_set run_stat",
            [
                [1, "fixed_q", 1, "Q", 20, 0, 0, 0, 1],
                [2, "voltage_q", 1, "V", 10, 0, 0, 380, 1],
                [3, "fixed_z", 1, "Z", 0, 0, -0.1, 0, 1],
            ],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [[1, "line", 1, 2, 0.01, 0.1, 0, 1]],
        ),
    }
    ppc = ensure_ac_ppc_topology(
        build_ac_ppc_from_efile_rows(Path("ac_shunt_q_sharing.e"), rows)
    )
    calc = ACPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["q"]],
        [0.5],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.result["shunt"][:, SHUNT_COLS["q"]],
        [0.2, 0.4, 0.1],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.result["shunt"][:, SHUNT_COLS["current"]],
        [0.2, 0.4, 0.1],
        rtol=0.0,
        atol=1e-12,
    )


def test_dc_balance_devices_exceed_limits_to_finish_unallocated_residual():
    from dc_array_model import GEN_COLS, LOAD_COLS
    from dc_lf import DCPowerFlowCalc

    ppc = _build_dc_multi_balance_ppc()
    ppc["load"][0, LOAD_COLS["pbase"]] = 2.0
    calc = DCPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [1.05, 0.95],
        rtol=0.0,
        atol=1e-12,
    )


def test_hybrid_dc_balance_writeback_includes_dcac_terminal_power():
    from dc_array_model import GEN_COLS, LOAD_COLS
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    network = _build_hybrid_dcac_balance_network()
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        verbose=False,
    )

    assert calc.run() == 0
    assert calc.converged

    dc_gen = calc.result["dc"]["gen"][:, GEN_COLS["p"]]
    dc_load = calc.result["dc"]["load"][:, LOAD_COLS["p"]]
    dcac_p = calc.result["dcac"][:, 0]
    np.testing.assert_allclose(dcac_p, [-0.3], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dc_gen, [-0.1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        float(np.sum(dc_load) - np.sum(dc_gen) + np.sum(dcac_p)),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_hybrid_dcac_device_types_share_terminal_power_signs():
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    solved = {}
    for p_ac_set, expected_terminal_powers in (
        (30, [-0.3, 0.3]),
        (-30, [0.3, -0.3]),
    ):
        for dev_type in ("ACDCConverter", "DCACConverter"):
            network = _build_hybrid_dcac_balance_network(dev_type, p_ac_set)
            converter = network.dcac_converters[0]
            calc = HybridPowerFlowCalc(
                network,
                linear_solver="scipy",
                result_mode="array",
                verbose=False,
            )

            assert converter.dev_type == dev_type
            assert calc.run() == 0
            assert calc.converged
            solved[(dev_type, p_ac_set)] = calc.result["dcac"][0].copy()
            np.testing.assert_allclose(
                solved[(dev_type, p_ac_set)][:2],
                expected_terminal_powers,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.sum(solved[(dev_type, p_ac_set)][:2]),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )

        np.testing.assert_allclose(
            solved[("ACDCConverter", p_ac_set)],
            solved[("DCACConverter", p_ac_set)],
            rtol=0.0,
            atol=1e-12,
        )


def test_hybrid_dcac_descriptive_dev_type_does_not_change_calculation():
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    solved = {}
    for dev_type in (
        "DCACConverter",
        "wind-acdc-converter",
        "grid-acdc-converter",
        "storage-dcac-converter",
    ):
        network = _build_hybrid_dcac_balance_network(dev_type, p_ac_set=-30)
        calc = HybridPowerFlowCalc(
            network,
            linear_solver="scipy",
            result_mode="array",
            verbose=False,
        )

        assert calc.run() == 0
        assert calc.converged
        solved[dev_type] = calc.result["dcac"][0].copy()
        np.testing.assert_allclose(
            solved[dev_type][:2],
            [0.3, -0.3],
            rtol=0.0,
            atol=1e-12,
        )

    for dev_type, result in solved.items():
        np.testing.assert_allclose(
            result,
            solved["DCACConverter"],
            rtol=0.0,
            atol=1e-12,
            err_msg=f"descriptive dev_type changed calculation: {dev_type}",
        )


def test_hybrid_dcac_dc_p_control_uses_dc_setpoint_and_ac_q_setpoint():
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    solved = {}
    for p_dc_set, expected_terminal_powers in (
        (30, [0.3, -0.3, 0.07]),
        (-30, [-0.3, 0.3, 0.07]),
    ):
        for dev_type in ("ACDCConverter", "DCACConverter"):
            network = _build_hybrid_dcac_balance_network(
                dev_type,
                p_ac_set=0,
                ac_control_type="NONE",
                dc_control_type="P",
                p_dc_set=p_dc_set,
                q_ac_set=7,
            )
            converter = network.dcac_converters[0]
            calc = HybridPowerFlowCalc(
                network,
                linear_solver="scipy",
                result_mode="array",
                verbose=False,
            )

            assert converter.p_dc_set == p_dc_set / 100.0
            assert not hasattr(converter, "control_type")
            assert converter.ac_control_type == "NONE"
            assert converter.dc_control_type == "P"
            assert calc.run() == 0
            assert calc.converged
            solved[(dev_type, p_dc_set)] = calc.result["dcac"][0].copy()
            np.testing.assert_allclose(
                solved[(dev_type, p_dc_set)][:3],
                expected_terminal_powers,
                rtol=0.0,
                atol=1e-12,
            )

            residual = calc.get_f(calc.x)
            np.testing.assert_allclose(
                residual[calc.dcac_eq_start:calc.dcac_eq_start + 3],
                0.0,
                rtol=0.0,
                atol=1e-12,
            )

        np.testing.assert_allclose(
            solved[("ACDCConverter", p_dc_set)],
            solved[("DCACConverter", p_dc_set)],
            rtol=0.0,
            atol=1e-12,
        )


def test_hybrid_dcac_dc_p_control_jacobian_matches_finite_difference():
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    network = _build_hybrid_dcac_balance_network(
        p_ac_set=0,
        ac_control_type="NONE",
        dc_control_type="P",
        p_dc_set=30,
        q_ac_set=7,
    )
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        verbose=False,
    )
    calc.prepare()

    x = calc.x.copy()
    row = int(calc.dcac_eq_ctrl_1[0])
    col = int(calc.dcac_dc_p_col[0])
    analytic = float(calc.get_jacobi(x)[row, col])
    step = 1e-7
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[col] += step
    x_minus[col] -= step
    finite_difference = float(
        (calc.get_f(x_plus)[row] - calc.get_f(x_minus)[row]) / (2.0 * step)
    )

    np.testing.assert_allclose(analytic, 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(finite_difference, analytic, rtol=0.0, atol=1e-9)


def test_hybrid_dcac_dc_p_control_preserves_terminal_signs_with_losses():
    from ac_array_model import BUS_COLS as AC_BUS_COLS
    from dc_array_model import BUS_COLS as DC_BUS_COLS
    from lfcore.hybrid_lf import HybridPowerFlowCalc

    solved = {}
    for p_dc_set in (30, -30):
        for dev_type in ("ACDCConverter", "DCACConverter"):
            network = _build_hybrid_dcac_balance_network(
                dev_type,
                p_ac_set=0,
                ac_control_type="NONE",
                dc_control_type="P",
                p_dc_set=p_dc_set,
                q_ac_set=7,
                r1=0.01,
                r2=0.01,
            )
            calc = HybridPowerFlowCalc(
                network,
                linear_solver="scipy",
                result_mode="array",
                verbose=False,
            )

            assert calc.run() == 0
            dc_p, ac_p, ac_q = calc.result["dcac"][0, :3]
            solved[(dev_type, p_dc_set)] = calc.result["dcac"][0].copy()
            np.testing.assert_allclose(dc_p, p_dc_set / 100.0, rtol=0.0, atol=1e-12)
            assert np.signbit(ac_p) != np.signbit(dc_p)
            assert dc_p + ac_p > 0.0

            va = calc.ac_calc.result["bus"][0, AC_BUS_COLS["voltage"]]
            vd = calc.dc_calc.result["bus"][0, DC_BUS_COLS["voltage"]]
            loss_residual = (
                vd * vd * va * va * (dc_p + ac_p)
                - 0.01 * dc_p * dc_p * va * va
                - 0.01 * (ac_p * ac_p + ac_q * ac_q) * vd * vd
            )
            np.testing.assert_allclose(loss_residual, 0.0, rtol=0.0, atol=1e-10)

        np.testing.assert_allclose(
            solved[("ACDCConverter", p_dc_set)],
            solved[("DCACConverter", p_dc_set)],
            rtol=0.0,
            atol=1e-12,
        )


def test_ac_balance_devices_use_rated_capacity_when_zero_p_max_is_placeholder():
    from ac_array_model import GEN_COLS
    from ac_lf import ACPowerFlowCalc
    from model.ppc_topology import ensure_ac_ppc_topology
    from ac_array_model import build_ac_ppc_from_efile_rows

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "ac_zero_pmax_placeholder", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "bus_1", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_min p_max q_min q_max run_stat rated_capacity",
            [
                [1, "slack_1", 1, "PH", 0, 0, 380, 1, 0, 0, 0, 0, 1, 100],
                [2, "slack_2", 1, "PH", 0, 0, 380, 1, 0, 0, 0, 0, 1, 100],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load_1", 1, 100, 1, 0, 0, 100, 1, 0, 0, 1]],
        ),
    }
    ppc = ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("ac_zero_pmax_placeholder.e"), rows))
    calc = ACPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["q"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )


def test_dc_balance_devices_use_rated_capacity_when_zero_p_max_is_placeholder():
    from dc_array_model import GEN_COLS
    from dc_lf import DCPowerFlowCalc
    from model.ppc_topology import ensure_dc_ppc_topology
    from dc_array_model import build_dc_ppc_from_efile_rows

    rows = {
        "PowerBase": _table(
            "p_base u_unit p_unit i_unit",
            [[100, "V", "kW", "A"]],
        ),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [[1, "bus_1", 100, 100, 0, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type p_set v_set i_set p_min p_max alpha run_stat rated_capacity",
            [
                [1, "v_1", 1, "V", 0, 100, 0, 0, 0, 1, 1, 100],
                [2, "v_2", 1, "V", 0, 100, 0, 0, 0, 1, 1, 100],
            ],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "load_1", 1, 100, 1, 0, 0, 1]],
        ),
    }
    ppc = ensure_dc_ppc_topology(build_dc_ppc_from_efile_rows(Path("dc_zero_pmax_placeholder.e"), rows))
    calc = DCPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )


def test_ac_object_model_balance_devices_use_rated_capacity_placeholder():
    from ac_array_model import GEN_COLS, _build_ac_ppc_from_model
    from ac_lf import ACPowerFlowCalc
    from model.ppc_topology import ensure_ac_ppc_topology

    model = SimpleNamespace(
        Model=[
            SimpleNamespace(
                path="test",
                name="ac_object_zero_pmax_placeholder",
                p_base=100,
                u_unit="V",
                p_unit="kW",
                i_unit="A",
            )
        ],
        ACNode=[
            SimpleNamespace(idx=1, name="bus_1", vbase=380, voltage=380, angle=0, run_stat=1)
        ],
        ACGenerator=[
            SimpleNamespace(
                idx=1,
                name="slack_1",
                node=1,
                control_type="PH",
                p_set=0,
                q_set=0,
                v_set=380,
                alpha=1,
                p_min=0,
                p_max=0,
                q_min=0,
                q_max=0,
                run_stat=1,
                rated_capacity=100,
            ),
            SimpleNamespace(
                idx=2,
                name="slack_2",
                node=1,
                control_type="PH",
                p_set=0,
                q_set=0,
                v_set=380,
                alpha=1,
                p_min=0,
                p_max=0,
                q_min=0,
                q_max=0,
                run_stat=1,
                rated_capacity=100,
            ),
        ],
        ACLoad=[
            SimpleNamespace(
                idx=1,
                name="load_1",
                node=1,
                pbase=100,
                pv0=1,
                pv1=0,
                pv2=0,
                qbase=100,
                qv0=1,
                qv1=0,
                qv2=0,
                run_stat=1,
            )
        ],
    )
    _, ppc = _build_ac_ppc_from_model(model)
    ppc = ensure_ac_ppc_topology(ppc)
    calc = ACPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(
        ppc["gen"][:, GEN_COLS["p_max"]],
        [1.0, 1.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(ppc["_gen_q_min"], [-1.0, -1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ppc["_gen_q_max"], [1.0, 1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["q"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )


def test_dc_object_model_balance_devices_use_rated_capacity_placeholder():
    from dc_array_model import GEN_COLS, _build_dc_ppc_from_model
    from dc_lf import DCPowerFlowCalc
    from model.ppc_topology import ensure_dc_ppc_topology

    model = SimpleNamespace(
        PowerBase=[
            SimpleNamespace(
                p_base=100,
                u_unit="V",
                p_unit="kW",
                i_unit="A",
            )
        ],
        DCNode=[
            SimpleNamespace(idx=1, name="bus_1", vbase=100, voltage=100, isl=0, run_stat=1)
        ],
        DCGenerator=[
            SimpleNamespace(
                idx=1,
                name="v_1",
                node=1,
                control_type="V",
                p_set=0,
                v_set=100,
                i_set=0,
                p_min=0,
                p_max=0,
                alpha=1,
                run_stat=1,
                rated_capacity=100,
            ),
            SimpleNamespace(
                idx=2,
                name="v_2",
                node=1,
                control_type="V",
                p_set=0,
                v_set=100,
                i_set=0,
                p_min=0,
                p_max=0,
                alpha=1,
                run_stat=1,
                rated_capacity=100,
            ),
        ],
        DCLoad=[
            SimpleNamespace(
                idx=1,
                name="load_1",
                node=1,
                pbase=100,
                pv0=1,
                pv1=0,
                pv2=0,
                run_stat=1,
            )
        ],
    )
    _, ppc = _build_dc_ppc_from_model(model)
    ppc = ensure_dc_ppc_topology(ppc)
    calc = DCPowerFlowCalc(ppc, linear_solver="scipy", result_mode="array")

    assert calc.run() == 0
    assert calc.converged
    np.testing.assert_allclose(ppc["_gen_p_max"], [1.0, 1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        calc.result["gen"][:, GEN_COLS["p"]],
        [0.5, 0.5],
        rtol=0.0,
        atol=1e-12,
    )


def test_residual_allocator_handles_missing_limits_and_negative_residual():
    from lfcore.common import allocate_limited_residual

    np.testing.assert_allclose(
        allocate_limited_residual([0.2, 0.1], 1.0, alpha=[1.0, 2.0]),
        [0.43333333333333335, 0.5666666666666667],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        allocate_limited_residual(
            [0.8, 0.6],
            0.5,
            lower=[0.0, 0.3],
            upper=[1.0, 1.0],
            alpha=[1.0, 2.0],
        ),
        [0.2, 0.3],
        rtol=0.0,
        atol=1e-12,
    )


def test_residual_allocator_exceeds_bounds_only_after_headroom_is_exhausted():
    from lfcore.common import allocate_limited_residual

    np.testing.assert_allclose(
        allocate_limited_residual(
            [0.2, 0.1],
            2.0,
            lower=[0.0, 0.0],
            upper=[0.8, 0.45],
            alpha=[1.0, 2.0],
        ),
        [1.05, 0.95],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        allocate_limited_residual(
            [0.0, 0.0],
            -1.0,
            lower=[-0.2, -0.4],
            upper=[0.5, 0.5],
            alpha=[1.0, 1.0],
        ),
        [-0.4, -0.6],
        rtol=0.0,
        atol=1e-12,
    )
