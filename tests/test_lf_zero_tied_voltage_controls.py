from pathlib import Path
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "hybrid_power_system_analysis"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "model"))
sys.path.insert(0, str(SRC_DIR / "lfcore"))


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _ac_zero_tied_ph_ppc():
    from ac_array_model import build_ac_ppc_from_efile_rows
    from model.ppc_topology import ensure_ac_ppc_topology

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["IEEE", "zero_tied_ph", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [
                [1, "source_1", 380, 380, 0, 1],
                [2, "source_2", 380, 380, 0, 1],
                [3, "source_bus", 380, 380, 0, 1],
                [4, "load_bus", 380, 380, 0, 1],
            ],
        ),
        "ACSwitch": _table(
            "idx name i_node j_node status run_stat",
            [[1, "switch_1_3", 1, 3, 1, 1]],
        ),
        "ACZeroBranch": _table(
            "idx name i_node j_node run_stat p q current",
            [[1, "zbr_2_3", 2, 3, 1, 0, 0, 0]],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [[1, "branch_3_4", 3, 4, 0.01, 0.1, 0.0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha p_max run_stat",
            [
                [1, "ph_1", 1, "PH", 0, 0, 380.0, 1.0, 100, 1],
                [2, "ph_2", 2, "PH", 0, 0, 395.2, 3.0, 100, 1],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load", 4, 50, 1, 0, 0, 10, 1, 0, 0, 1]],
        ),
    }
    return ensure_ac_ppc_topology(build_ac_ppc_from_efile_rows(Path("zero_tied_ph.e"), rows))


def _dc_zero_tied_v_ppc():
    from dc_array_model import build_dc_ppc_from_efile_rows
    from model.ppc_topology import ensure_dc_ppc_topology

    rows = {
        "PowerBase": _table("p_base u_unit p_unit i_unit", [[100, "V", "kW", "A"]]),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [
                [1, "source_1", 100, 100, 0, 1],
                [2, "source_2", 100, 100, 0, 1],
                [3, "source_bus", 100, 100, 0, 1],
                [4, "load_bus", 100, 100, 0, 1],
            ],
        ),
        "DCSwitch": _table(
            "idx name i_node j_node status run_stat",
            [[1, "switch_1_3", 1, 3, 1, 1]],
        ),
        "DCZeroBranch": _table(
            "idx name i_node j_node run_stat p current",
            [[1, "zbr_2_3", 2, 3, 1, 0, 0]],
        ),
        "DCBranch": _table(
            "idx name i_node j_node r run_stat",
            [[1, "branch_3_4", 3, 4, 0.1, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "v_1", 1, "V", 98, 0, 0, 1], [2, "v_2", 2, "V", 102, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "load", 4, 50, 1, 0, 0, 1]],
        ),
    }
    return ensure_dc_ppc_topology(build_dc_ppc_from_efile_rows(Path("zero_tied_v.e"), rows))


def _hybrid_zero_tied_ph_with_fixed_pq_converter_network(include_acac=False, include_floating_dc=False):
    from lfcore.hybrid_lf import _build_lf_network_from_hybrid_rows

    rows = {
        "PowerBase": _table("p_base u_unit p_unit i_unit", [[100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle isl run_stat",
            [
                [1, "source_1", 380, 380, 0, 0, 1],
                [2, "source_2", 380, 380, 0, 0, 1],
                [3, "source_bus", 380, 380, 0, 0, 1],
                [4, "zero_tie_bus", 380, 380, 0, 0, 1],
                [5, "load_bus", 380, 380, 0, 0, 1],
            ],
        ),
        "ACZeroBranch": _table(
            "idx name i_node j_node run_stat p q current",
            [
                [1, "zbr_1_3", 1, 3, 1, 0, 0, 0],
                [2, "zbr_2_3", 2, 3, 1, 0, 0, 0],
                [3, "zbr_4_5", 4, 5, 1, 0, 0, 0],
            ],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [[1, "branch_3_4", 3, 4, 0.01, 0.1, 0.0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [
                [1, "ph_1", 1, "PH", 0, 0, 380.0, 1.0, 1],
                [2, "ph_2", 2, "PH", 0, 0, 395.2, 3.0, 1],
            ],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load", 5, 50, 1, 0, 0, 10, 1, 0, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage isl run_stat",
            [[1, "dc_source", 100, 100, 0, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "dc_v", 1, "V", 100, 0, 0, 1]],
        ),
        "DCACConverter": _table(
            (
                "idx name ac_node dc_node r1 r2 ac_control_type dc_control_type "
                "p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat"
            ),
            [[1, "fixed_pq", 3, 1, 0.01, 0.01, "PQ", "NONE", 10, 0, 2, 0, 0, 1]],
        ),
    }
    if include_acac:
        rows["ACACConverter"] = _table(
            (
                "idx name i_node j_node r1 r2 i_control_type j_control_type "
                "p_set i_q_set j_q_set i_v_set j_v_set run_stat"
            ),
            [[1, "fixed_pq_acac", 3, 5, 0.01, 0.01, "PQ", "PQ", 5, 1, 1.5, 0, 0, 1]],
        )
    if include_floating_dc:
        rows["DCNode"]["rows"].append([2, "floating_dc", 100, 100, 0, 1])
        rows["DCGenerator"]["rows"].append([2, "floating_p", 2, "P", 100, 0, 0, 1])
        rows["DCDCConverter"] = _table(
            (
                "idx name dev_type i_node j_node r1 r2 i_control_type j_control_type "
                "p_set i_set v_set run_stat"
            ),
            [[1, "floating_dcdc", "dcdc-converter", 1, 2, 0, 0, "V", "NONE", 0, 0, 100, 1]],
        )
    return _build_lf_network_from_hybrid_rows(Path("zero_tied_hybrid.e"), rows)


def _hybrid_zero_tied_acac_pv_network():
    from lfcore.hybrid_lf import _build_lf_network_from_hybrid_rows

    rows = {
        "PowerBase": _table("p_base u_unit p_unit i_unit", [[100, "V", "kW", "A"]]),
        "ACNode": _table(
            "idx name vbase voltage angle isl run_stat",
            [
                [1, "acac_pv_1", 380, 380, 0, 0, 1],
                [2, "acac_pv_2", 380, 380, 0, 0, 1],
                [3, "source", 380, 380, 0, 0, 1],
                [4, "acac_pq", 380, 380, 0, 0, 1],
            ],
        ),
        "ACZeroBranch": _table(
            "idx name i_node j_node run_stat p q current",
            [[1, "zbr_1_2", 1, 2, 1, 0, 0, 0]],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [
                [1, "branch_3_1", 3, 1, 0.01, 0.1, 0.0, 1],
                [2, "branch_3_4", 3, 4, 0.01, 0.1, 0.0, 1],
            ],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "ph", 3, "PH", 0, 0, 380.0, 1.0, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "load", 4, 10, 1, 0, 0, 2, 1, 0, 0, 1]],
        ),
        "ACACConverter": _table(
            (
                "idx name i_node j_node r1 r2 i_control_type j_control_type "
                "p_set i_q_set j_q_set i_v_set j_v_set run_stat"
            ),
            [
                [1, "acac_1", 1, 4, 0.01, 0.01, "PV", "PQ", 5, 0, 1, 380.0, 0, 1],
                [2, "acac_2", 2, 4, 0.01, 0.01, "PV", "PQ", 5, 0, 1, 395.2, 0, 1],
            ],
        ),
    }
    return _build_lf_network_from_hybrid_rows(Path("zero_tied_acac_pv.e"), rows)


def test_ac_lf_unifies_zero_tied_ph_controls_before_newton():
    from ac_array_model import BUS_COLS, GEN_COLS
    from ac_lf import ACPowerFlowCalc

    ppc = _ac_zero_tied_ph_ppc()
    topology = ppc["_topology_arrays"]
    assert topology.node_to_bus_pos[0] == topology.node_to_bus_pos[2]
    assert topology.node_to_bus_pos[1] != topology.node_to_bus_pos[2]

    calc = ACPowerFlowCalc(
        ppc,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )
    calc.prepare()

    jacobian = calc.get_jacobi(calc.x)
    assert calc.total_vars == calc.total_eq
    assert np.linalg.matrix_rank(jacobian.toarray()) == calc.total_vars
    assert calc.run() == 0

    result = calc.result
    source_rows = np.isin(result["bus"][:, BUS_COLS["idx"]].astype(int), [1, 2, 3])
    assert np.allclose(result["bus"][source_rows, BUS_COLS["voltage"]], 1.02, atol=1e-9)
    generated_p = result["gen"][:, GEN_COLS["p"]]
    generated_q = result["gen"][:, GEN_COLS["q"]]
    assert np.isclose(generated_p[1] / generated_p[0], 3.0, rtol=1e-8, atol=1e-8)
    assert np.isclose(generated_q[1] / generated_q[0], 1.0, rtol=1e-8, atol=1e-8)


def test_dc_lf_unifies_zero_tied_v_controls_before_newton():
    from dc_array_model import BUS_COLS, GEN_COLS
    from dc_lf import DCPowerFlowCalc

    ppc = _dc_zero_tied_v_ppc()
    topology = ppc["_topology_arrays"]
    assert topology.node_to_bus_pos[0] == topology.node_to_bus_pos[2]
    assert topology.node_to_bus_pos[1] != topology.node_to_bus_pos[2]

    calc = DCPowerFlowCalc(
        ppc,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )
    calc.prepare()

    jacobian = calc.get_jacobi(calc.x)
    assert calc.total_vars == calc.total_eq
    assert np.linalg.matrix_rank(jacobian.toarray()) == calc.total_vars
    assert calc.run() == 0

    result = calc.result
    source_rows = np.isin(result["bus"][:, BUS_COLS["idx"]].astype(int), [1, 2, 3])
    assert np.allclose(result["bus"][source_rows, BUS_COLS["voltage"]], 1.0, atol=1e-9)
    generated_p = result["gen"][:, GEN_COLS["p"]]
    assert np.isclose(generated_p[0], generated_p[1], rtol=1e-9, atol=1e-9)


def test_dc_lf_refreshes_topology_after_voltage_control_table_growth():
    from dc_array_model import BUS_COLS, GEN_COLS
    from dc_lf import DCPowerFlowCalc

    ppc = _dc_zero_tied_v_ppc()
    new_gen = ppc["gen"][0].copy()
    new_gen[GEN_COLS["idx"]] = 3
    new_gen[GEN_COLS["v_set"]] = 1.04
    ppc["gen"] = np.vstack((ppc["gen"], new_gen))

    calc = DCPowerFlowCalc(
        ppc,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )
    assert calc.run() == 0

    result = calc.result
    source_rows = np.isin(result["bus"][:, BUS_COLS["idx"]].astype(int), [1, 2, 3])
    assert np.allclose(result["bus"][source_rows, BUS_COLS["voltage"]], (0.98 + 1.02 + 1.04) / 3.0)


def test_hybrid_lf_allows_fixed_pq_converter_on_zero_tied_ph_bus():
    from ac_array_model import BUS_COLS as AC_BUS_COLS, GEN_COLS as AC_GEN_COLS
    from hybrid_lf import HybridPowerFlowCalc

    network = _hybrid_zero_tied_ph_with_fixed_pq_converter_network()
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )
    calc.prepare()

    jacobian = calc.get_jacobi(calc.x)
    assert calc.total_vars == calc.total_eq
    assert np.linalg.matrix_rank(jacobian.toarray()) == calc.total_vars
    assert calc.run() == 0

    ac_result = calc.result["ac"]
    source_rows = np.isin(ac_result["bus"][:, AC_BUS_COLS["idx"]].astype(int), [1, 2, 3])
    assert np.allclose(ac_result["bus"][source_rows, AC_BUS_COLS["voltage"]], 1.02, atol=1e-9)
    generated_p = ac_result["gen"][:, AC_GEN_COLS["p"]]
    generated_q = ac_result["gen"][:, AC_GEN_COLS["q"]]
    assert np.isclose(generated_p[1] / generated_p[0], 3.0, rtol=1e-8, atol=1e-8)
    assert np.isclose(generated_q[1] / generated_q[0], 1.0, rtol=1e-8, atol=1e-8)

    converter = calc.result["dcac"][0]
    assert np.isclose(converter[1], 0.1, atol=1e-10)
    assert np.isclose(converter[2], 0.02, atol=1e-10)


def test_hybrid_lf_allows_fixed_pq_acac_terminal_on_zero_tied_ph_bus():
    from ac_array_model import ACAC_COLS
    from hybrid_lf import HybridPowerFlowCalc

    network = _hybrid_zero_tied_ph_with_fixed_pq_converter_network(include_acac=True)
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )

    assert calc.run() == 0
    converter = calc.result["ac"]["acac"][0]
    assert np.isclose(converter[ACAC_COLS["i_p"]], 0.05, atol=1e-10)
    assert np.isclose(converter[ACAC_COLS["i_q"]], 0.01, atol=1e-10)
    assert np.isclose(converter[ACAC_COLS["j_q"]], 0.015, atol=1e-10)


def test_hybrid_lf_unifies_zero_tied_acac_pv_controls():
    from ac_array_model import ACAC_COLS, BUS_COLS as AC_BUS_COLS
    from hybrid_lf import HybridPowerFlowCalc

    network = _hybrid_zero_tied_acac_pv_network()
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        tol=1e-10,
        max_iter=30,
        verbose=False,
    )
    calc.prepare()

    jacobian = calc.get_jacobi(calc.x)
    assert np.linalg.matrix_rank(jacobian.toarray()) == calc.total_vars
    assert calc.run() == 0

    ac_result = calc.result["ac"]
    controlled_rows = np.isin(ac_result["bus"][:, AC_BUS_COLS["idx"]].astype(int), [1, 2])
    assert np.allclose(ac_result["bus"][controlled_rows, AC_BUS_COLS["voltage"]], 1.02, atol=1e-9)
    assert np.isclose(
        calc.result["ac"]["acac"][0, ACAC_COLS["i_q"]],
        calc.result["ac"]["acac"][1, ACAC_COLS["i_q"]],
        atol=1e-9,
    )


def test_hybrid_lf_skips_floating_dc_island_and_solves_referenced_side():
    from dc_array_model import BUS_COLS as DC_BUS_COLS, DCDC_COLS
    from hybrid_lf import HybridPowerFlowCalc

    network = _hybrid_zero_tied_ph_with_fixed_pq_converter_network(include_floating_dc=True)
    calc = HybridPowerFlowCalc(
        network,
        linear_solver="scipy",
        result_mode="array",
        max_iter=30,
        verbose=False,
    )

    assert calc.run() == 0
    assert calc.failure_reason == ""
    assert np.all(np.isfinite(calc.x))

    dc_bus = calc.result["dc"]["bus"]
    dc_bus_by_idx = {
        int(row[DC_BUS_COLS["idx"]]): row
        for row in dc_bus
    }
    assert dc_bus_by_idx[1][DC_BUS_COLS["voltage"]] > 0.0
    assert dc_bus_by_idx[2][DC_BUS_COLS["run_stat"]] == 1
    assert dc_bus_by_idx[2][DC_BUS_COLS["voltage"]] == 0.0

    topology = calc.dc_calc._ppc_topology
    node_rows = calc.dc_calc.ppc["bus"][:, DC_BUS_COLS["idx"]].astype(int)
    node_2_row = int(np.flatnonzero(node_rows == 2)[0])
    assert not topology.node_alive_mask[node_2_row]
    assert not topology.devices["dcdc"].alive_mask[0]

    dcdc = calc.result["dc"]["dcdc"]
    assert dcdc.shape[0] == 1
    assert dcdc[0, DCDC_COLS["run_stat"]] == 1
    assert np.allclose(
        dcdc[0, [DCDC_COLS[name] for name in ("i_p", "j_p", "i_c", "j_c")]],
        0.0,
    )


def test_ac_multi_slack_island_is_warning_and_power_flow_continues():
    from ac_lf import ACPowerFlowCalc
    from ac_model import ACPowerNetwork

    network = ACPowerNetwork()
    for idx, name in ((1, "source_1"), (2, "source_2"), (3, "load_bus")):
        node = network.add_node(idx, 110.0)
        node.name = name
    for idx, node_idx, voltage in ((1, 1, 1.00), (2, 2, 1.02)):
        gen = network.add_generator(idx, node_idx, "SLACK", 0.0, 0.0, voltage, alpha=1.0)
        gen.name = f"slack_{idx}"
    network.add_branch(1, 1, 3, 0.01, 0.10, 0.0).name = "branch_1_3"
    network.add_branch(2, 2, 3, 0.01, 0.10, 0.0).name = "branch_2_3"
    network.add_load(1, 3, 0.30, 1.0, 0.0, 0.0, 0.10, 1.0, 0.0, 0.0).name = "load"

    network.topo()
    topology_warnings, topology_errors = network.check_topo()

    assert any("存在多个平衡节点" in message for message in topology_warnings)
    assert not any("存在多个平衡节点" in message for message in topology_errors)
    calc = ACPowerFlowCalc(network, linear_solver="scipy", result_mode="array", tol=1e-10, max_iter=30)
    assert calc.run() == 0
    assert calc.converged


def test_dc_multi_voltage_island_is_warning_and_power_flow_continues():
    from dc_lf import DCPowerFlowCalc
    from dc_model import DCPowerNetwork

    network = DCPowerNetwork()
    for idx, name in ((1, "source_1"), (2, "source_2"), (3, "load_bus")):
        node = network.add_node(idx, 100.0)
        node.name = name
    for idx, node_idx, voltage in ((1, 1, 0.98), (2, 2, 1.02)):
        gen = network.add_generator(idx, node_idx, "V", 0.0, voltage, 0.0)
        gen.name = f"voltage_source_{idx}"
    network.add_branch(1, 1, 3, 0.10).name = "branch_1_3"
    network.add_branch(2, 2, 3, 0.10).name = "branch_2_3"
    network.add_load(1, 3, 0.30, 1.0, 0.0, 0.0).name = "load"

    network.topo()
    topology_warnings, topology_errors = network.check_topo()

    assert any("存在多个定V节点" in message for message in topology_warnings)
    assert not any("多个电压控制源" in message for message in topology_errors)
    calc = DCPowerFlowCalc(network, linear_solver="scipy", result_mode="array", tol=1e-10, max_iter=30)
    assert calc.run() == 0
    assert calc.converged
