from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
