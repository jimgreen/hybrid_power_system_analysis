from pathlib import Path
import sys

import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "model"))
sys.path.insert(0, str(SRC_DIR / "lfcore"))


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _ac_rows(*, status, closed_status, closed_status_set):
    return {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "ac_closed_status", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "source", 380, 380, 0, 1], [2, "load", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "slack", 1, "PH", 0, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [
                [1, "source_load", 1, 1, 1, 0, 0, 0.2, 1, 0, 0, 1],
                [2, "remote_load", 2, 10, 1, 0, 0, 2, 1, 0, 0, 1],
            ],
        ),
        "ACBreak": _table(
            "idx name i_node j_node status closed_status closed_status_set run_stat",
            [[1, "breaker", 1, 2, status, closed_status, closed_status_set, 1]],
        ),
    }


def _dc_rows(*, status, closed_status, closed_status_set):
    return {
        "PowerBase": _table(
            "p_base u_unit p_unit i_unit",
            [[100, "V", "kW", "A"]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [[1, "source", 100, 100, 1], [2, "load", 100, 100, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "slack", 1, "V", 100, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "source_load", 1, 1, 1, 0, 0, 1], [2, "remote_load", 2, 10, 1, 0, 0, 1]],
        ),
        "DCBreak": _table(
            "idx name i_node j_node status closed_status closed_status_set run_stat",
            [[1, "breaker", 1, 2, status, closed_status, closed_status_set, 1]],
        ),
    }


@pytest.mark.parametrize(
    ("status", "closed_status", "closed_status_set", "expected_closed"),
    [(0, 0, 1, 1), (1, 1, 0, 0)],
)
def test_ac_lf_uses_closed_status_set_and_returns_closed_status(
    status,
    closed_status,
    closed_status_set,
    expected_closed,
):
    from lfcore.ac_lf import ACPowerFlowCalc
    from model.ac_array_model import (
        BREAK_COLS,
        build_ac_network_from_ppc,
        build_ac_ppc_from_efile_rows,
    )

    ppc = build_ac_ppc_from_efile_rows(
        Path("ac_closed_status.e"),
        _ac_rows(
            status=status,
            closed_status=closed_status,
            closed_status_set=closed_status_set,
        ),
    )
    network = build_ac_network_from_ppc(ppc)
    calc = ACPowerFlowCalc(network, linear_solver="scipy", result_mode="full", verbose=False)

    assert calc.run() == 0
    result_row = calc.result["break"][0]
    assert int(result_row[BREAK_COLS["status"]]) == status
    assert int(result_row[BREAK_COLS["closed_status_set"]]) == closed_status_set
    assert int(result_row[BREAK_COLS["closed_status"]]) == expected_closed
    assert calc.lf_result.breakers["breaker"].closed_status == expected_closed
    assert network.breakers[0].closed_status == expected_closed
    assert bool(calc._ppc_topology.devices["break"].alive_mask[0]) is bool(expected_closed)


@pytest.mark.parametrize(
    ("status", "closed_status", "closed_status_set", "expected_closed"),
    [(0, 0, 1, 1), (1, 1, 0, 0)],
)
def test_dc_lf_uses_closed_status_set_and_returns_closed_status(
    status,
    closed_status,
    closed_status_set,
    expected_closed,
):
    from lfcore.dc_lf import DCPowerFlowCalc
    from model.dc_array_model import (
        BREAK_COLS,
        build_dc_network_from_ppc,
        build_dc_ppc_from_efile_rows,
    )

    ppc = build_dc_ppc_from_efile_rows(
        Path("dc_closed_status.e"),
        _dc_rows(
            status=status,
            closed_status=closed_status,
            closed_status_set=closed_status_set,
        ),
    )
    network = build_dc_network_from_ppc(ppc)
    calc = DCPowerFlowCalc(network, linear_solver="scipy", result_mode="full", verbose=False)

    assert calc.run() == 0
    result_row = calc.result["break"][0]
    assert int(result_row[BREAK_COLS["status"]]) == status
    assert int(result_row[BREAK_COLS["closed_status_set"]]) == closed_status_set
    assert int(result_row[BREAK_COLS["closed_status"]]) == expected_closed
    assert calc.lf_result.breakers["breaker"].closed_status == expected_closed
    assert network.breakers[0].closed_status == expected_closed
    assert bool(calc._ppc_topology.devices["break"].alive_mask[0]) is bool(expected_closed)


@pytest.mark.parametrize("system", ["ac", "dc"])
@pytest.mark.parametrize(
    ("status", "closed_status", "closed_status_set", "expected_closed"),
    [(0, 0, 1, 1), (1, 1, 0, 0)],
)
def test_lf_switch_topology_uses_closed_status_set(
    system,
    status,
    closed_status,
    closed_status_set,
    expected_closed,
):
    if system == "ac":
        from lfcore.ac_lf import ACPowerFlowCalc as PowerFlowCalc
        from model.ac_array_model import SWITCH_COLS, build_ac_ppc_from_efile_rows

        rows = _ac_rows(
            status=status,
            closed_status=closed_status,
            closed_status_set=closed_status_set,
        )
        rows["ACSwitch"] = rows.pop("ACBreak")
        builder = build_ac_ppc_from_efile_rows
    else:
        from lfcore.dc_lf import DCPowerFlowCalc as PowerFlowCalc
        from model.dc_array_model import SWITCH_COLS, build_dc_ppc_from_efile_rows

        rows = _dc_rows(
            status=status,
            closed_status=closed_status,
            closed_status_set=closed_status_set,
        )
        rows["DCSwitch"] = rows.pop("DCBreak")
        builder = build_dc_ppc_from_efile_rows

    ppc = builder(Path(f"{system}_switch_closed_status.e"), rows)
    calc = PowerFlowCalc(ppc, linear_solver="scipy", result_mode="array", verbose=False)

    assert calc.run() == 0
    assert int(calc.result["switch"][0, SWITCH_COLS["status"]]) == status
    assert (
        int(calc.result["switch"][0, SWITCH_COLS["closed_status_set"]])
        == closed_status_set
    )
    assert int(calc.result["switch"][0, SWITCH_COLS["closed_status"]]) == expected_closed
    assert bool(calc._ppc_topology.devices["switch"].alive_mask[0]) is bool(expected_closed)


def test_hybrid_lf_applies_ac_and_dc_closed_status_boundaries():
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_hybrid_rows
    from model.ac_array_model import BREAK_COLS as AC_BREAK_COLS
    from model.dc_array_model import BREAK_COLS as DC_BREAK_COLS

    rows = {
        **_ac_rows(status=0, closed_status=0, closed_status_set=1),
        **_dc_rows(status=1, closed_status=1, closed_status_set=0),
    }
    network = _build_lf_network_from_hybrid_rows(Path("hybrid_closed_status.e"), rows)
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="full",
        verbose=False,
    )

    assert calc.run() == 0
    ac_break = calc.result["ac"]["break"][0]
    dc_break = calc.result["dc"]["break"][0]
    assert int(ac_break[AC_BREAK_COLS["closed_status"]]) == 1
    assert int(dc_break[DC_BREAK_COLS["closed_status"]]) == 0
    assert calc.lf_result.ac.breakers["breaker"].closed_status == 1
    assert calc.lf_result.dc.breakers["breaker"].closed_status == 0


def test_legacy_status_populates_setpoint_and_result_fields():
    from model.ac_array_model import BREAK_COLS, build_ac_ppc_from_efile_rows

    rows = _ac_rows(status=0, closed_status=0, closed_status_set=0)
    rows["ACBreak"] = _table(
        "idx name i_node j_node status run_stat",
        [[1, "legacy_breaker", 1, 2, 1, 1]],
    )
    ppc = build_ac_ppc_from_efile_rows(Path("legacy_status.e"), rows)
    breaker = ppc["break"][0]

    assert int(breaker[BREAK_COLS["status"]]) == 1
    assert int(breaker[BREAK_COLS["closed_status_set"]]) == 1
    assert int(breaker[BREAK_COLS["closed_status"]]) == 1
