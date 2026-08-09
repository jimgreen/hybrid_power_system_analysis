import contextlib
import io
from pathlib import Path

import numpy as np


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _ac_rows():
    return {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "ac_full_zero_ties", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[idx, f"ac_n{idx}", 380, 380, 0, 1] for idx in range(1, 4)],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_min p_max q_min q_max run_stat",
            [[1, "ac_gen", 1, "PH", 0, 0, 380, 1, -200, 200, -200, 200, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "ac_load", 3, 40, 1, 0, 0, 20, 1, 0, 0, 1]],
        ),
        "ACZeroBranch": _table(
            "idx name i_node j_node run_stat",
            [[1, "ac_zero", 1, 2, 1]],
        ),
        "ACBreak": _table(
            "idx name i_node j_node status run_stat",
            [[1, "ac_break", 2, 3, 1, 1]],
        ),
    }


def _dc_rows():
    return {
        "PowerBase": _table(
            "p_base u_unit p_unit i_unit",
            [[100, "V", "kW", "A"]],
        ),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [[idx, f"dc_n{idx}", 100, 100, 0, 1] for idx in range(1, 4)],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set p_max p_min i_set alpha run_stat",
            [[1, "dc_gen", 1, "V", 100, 0, 200, -200, 0, 1, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "dc_load", 3, 40, 1, 0, 0, 1]],
        ),
        "DCZeroBranch": _table(
            "idx name i_node j_node run_stat",
            [[1, "dc_zero", 1, 2, 1]],
        ),
        "DCBreak": _table(
            "idx name i_node j_node status run_stat",
            [[1, "dc_break", 2, 3, 1, 1]],
        ),
    }


def _assert_ac_tie_result(result):
    expected_current = np.hypot(0.4, 0.2)
    for collection, name in (
        (result.zero_branches, "ac_zero"),
        (result.breakers, "ac_break"),
    ):
        tie = collection[name]
        np.testing.assert_allclose(
            [tie.i_p, tie.i_q, tie.i_c, tie.i_v],
            [0.4, 0.2, expected_current, 1.0],
            rtol=0.0,
            atol=1e-12,
        )


def _assert_dc_tie_result(result):
    for collection, name in (
        (result.zero_branches, "dc_zero"),
        (result.breakers, "dc_break"),
    ):
        tie = collection[name]
        np.testing.assert_allclose(
            [tie.i_p, tie.i_c, tie.i_v],
            [0.4, 0.4, 1.0],
            rtol=0.0,
            atol=1e-12,
        )


def test_ac_full_mode_builds_and_prints_zero_branch_and_break_results():
    from ac_lf import ACPowerFlowCalc, print_ac_result
    from model.ac_array_model import build_ac_ppc_from_efile_rows
    from model.ppc_topology import ensure_ac_ppc_topology

    ppc = ensure_ac_ppc_topology(
        build_ac_ppc_from_efile_rows(Path("ac_full_zero_ties.e"), _ac_rows())
    )
    calc = ACPowerFlowCalc(
        ppc,
        linear_solver="scipy",
        result_mode="full",
        verbose=False,
        tol=1e-10,
    )

    assert calc.run() == 0
    _assert_ac_tie_result(calc.lf_result)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_ac_result(calc, 0)
    text = output.getvalue()
    assert "ac_zero" in text
    assert "ac_break" in text
    assert "功率=0.400000 + j 0.200000 pu" in text


def test_dc_full_mode_builds_and_prints_zero_branch_and_break_results():
    from dc_lf import DCPowerFlowCalc, print_dc_result
    from model.dc_array_model import build_dc_ppc_from_efile_rows
    from model.ppc_topology import ensure_dc_ppc_topology

    ppc = ensure_dc_ppc_topology(
        build_dc_ppc_from_efile_rows(Path("dc_full_zero_ties.e"), _dc_rows())
    )
    calc = DCPowerFlowCalc(
        ppc,
        linear_solver="scipy",
        result_mode="full",
        verbose=False,
        tol=1e-10,
    )

    assert calc.run() == 0
    _assert_dc_tie_result(calc.lf_result)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_dc_result(calc, 0)
    text = output.getvalue()
    assert "dc_zero" in text
    assert "dc_break" in text
    assert "功率: 0.400000 pu" in text


def test_hybrid_full_mode_exposes_and_prints_ac_dc_zero_branch_and_break_results():
    from lfcore.hybrid_lf import (
        HybridPowerFlowCalc,
        _build_lf_network_from_hybrid_rows,
        print_hybrid_result,
    )

    rows = {**_ac_rows(), **_dc_rows()}
    network = _build_lf_network_from_hybrid_rows(Path("hybrid_full_zero_ties.e"), rows)
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="full",
        verbose=False,
        tol=1e-10,
    )

    assert calc.run() == 0
    assert calc.lf_result.ac is not None
    assert calc.lf_result.dc is not None
    _assert_ac_tie_result(calc.lf_result.ac)
    _assert_dc_tie_result(calc.lf_result.dc)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_hybrid_result(calc, 0)
    text = output.getvalue()
    for name in ("ac_zero", "ac_break", "dc_zero", "dc_break"):
        assert name in text
