from pathlib import Path


def _table(header, rows):
    return {"header_list": header.split(), "rows": rows}


def _composite_ac_rows(*, parent_run_stat=0, load_run_stat=1, nested=False):
    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "composite_state", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [
                [1, "source_bus", 380, 380, 0, 1],
                [2, "load_bus", 380, 380, 0, 1],
            ],
        ),
        "ACBranch": _table(
            "idx name i_node j_node r x b run_stat",
            [[1, "source_line", 1, 2, 0.01, 0.05, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "source", 1, "PH", 0, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "electrolyzer_load", 2, 10, 1, 0, 0, 2, 1, 0, 0, load_run_stat]],
        ),
        "HydroSource": _table(
            "idx name run_stat",
            [[1, "hydrogen_source", 1]],
        ),
        "AcE2Hydro": _table(
            "idx name run_stat idx_ac_load_t1 idx_h2_unit_t2",
            [[1, "electrolyzer", 1 if nested else parent_run_stat, 1, 1]],
        ),
    }
    if nested:
        rows["PlantComposite"] = _table(
            "idx name run_stat idx_ac_e2_hydro_t1",
            [[1, "hydrogen_plant", parent_run_stat, 1]],
        )
    return rows


def test_offline_composite_parent_disables_referenced_load_without_mutating_source_rows():
    from model.ac_array_model import LOAD_COLS
    from model.ppc_topology import build_ac_ppc_with_topology_from_efile_rows

    rows = _composite_ac_rows(parent_run_stat=0, load_run_stat=1)
    source_load_row = rows["ACLoad"]["rows"][0]
    source_run_stat_col = rows["ACLoad"]["header_list"].index("run_stat")

    ppc = build_ac_ppc_with_topology_from_efile_rows(Path("composite_state.e"), rows)

    assert source_load_row[source_run_stat_col] == 1
    assert ppc["load"][0, LOAD_COLS["run_stat"]] == 0
    assert ppc["_topology_arrays"].devices["load"].alive_mask.tolist() == [False]
    override = ppc["_effective_run_state_overrides"][0]
    assert override["dev_type"] == "ACLoad"
    assert override["dev_name"] == "electrolyzer_load"
    assert override["reason"] == "上级组合设备退运"
    assert override["ancestor_type"] == "AcE2Hydro"


def test_direct_e_rows_to_ppc_boundary_applies_composite_effective_state():
    from model.ac_array_model import LOAD_COLS, build_ac_ppc_from_efile_rows

    rows = _composite_ac_rows(parent_run_stat=0, load_run_stat=1)

    ppc = build_ac_ppc_from_efile_rows(Path("direct_composite_state.e"), rows)

    assert ppc["load"][0, LOAD_COLS["run_stat"]] == 0
    assert ppc["_effective_run_state_overrides"][0]["dev_type"] == "ACLoad"


def test_offline_nested_composite_ancestor_propagates_through_online_composite():
    from model.ac_array_model import LOAD_COLS
    from model.ppc_topology import build_ac_ppc_with_topology_from_efile_rows

    rows = _composite_ac_rows(parent_run_stat=0, load_run_stat=1, nested=True)

    ppc = build_ac_ppc_with_topology_from_efile_rows(Path("nested_composite_state.e"), rows)

    assert rows["AcE2Hydro"]["rows"][0][2] == 1
    assert rows["ACLoad"]["rows"][0][-1] == 1
    assert ppc["load"][0, LOAD_COLS["run_stat"]] == 0
    overrides = {
        (item["dev_type"], item["dev_name"]): item
        for item in ppc["_effective_run_state_overrides"]
    }
    assert overrides[("AcE2Hydro", "electrolyzer")]["ancestor_type"] == "PlantComposite"
    assert overrides[("ACLoad", "electrolyzer_load")]["ancestor_type"] == "PlantComposite"


def test_online_composite_parent_does_not_revive_child_with_own_offline_state():
    from model.ac_array_model import LOAD_COLS
    from model.ppc_topology import build_ac_ppc_with_topology_from_efile_rows

    rows = _composite_ac_rows(parent_run_stat=1, load_run_stat=0)

    ppc = build_ac_ppc_with_topology_from_efile_rows(Path("child_offline_state.e"), rows)

    assert rows["ACLoad"]["rows"][0][-1] == 0
    assert ppc["load"][0, LOAD_COLS["run_stat"]] == 0
    assert not any(
        item["dev_type"] == "ACLoad"
        for item in ppc["_effective_run_state_overrides"]
    )


def test_offline_composite_propagates_to_explicit_branch_and_terminal_members():
    from model.ac_array_model import BRANCH_COLS, BUS_COLS
    from model.ppc_topology import build_ac_ppc_with_topology_from_efile_rows

    rows = _composite_ac_rows(parent_run_stat=1, load_run_stat=1)
    rows["FeederComposite"] = _table(
        "idx name run_stat idx_ac_branch_t1 idx_ac_node_t2",
        [[1, "load_feeder", 0, 1, 2]],
    )

    ppc = build_ac_ppc_with_topology_from_efile_rows(Path("branch_composite_state.e"), rows)

    assert rows["ACBranch"]["rows"][0][-1] == 1
    assert rows["ACNode"]["rows"][1][-1] == 1
    assert ppc["branch"][0, BRANCH_COLS["run_stat"]] == 0
    assert ppc["bus"][1, BUS_COLS["run_stat"]] == 0
    overridden_types = {item["dev_type"] for item in ppc["_effective_run_state_overrides"]}
    assert {"ACBranch", "ACNode"} <= overridden_types


def test_hydro_to_dc_composite_disables_referenced_dc_generator():
    from model.dc_array_model import GEN_COLS
    from model.ppc_topology import build_dc_ppc_with_topology_from_efile_rows

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "dc_composite_state", 100, "V", "kW", "A"]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [
                [1, "source_bus", 750, 750, 1],
                [2, "fuel_cell_bus", 750, 750, 1],
            ],
        ),
        "DCBranch": _table(
            "idx name i_node j_node r run_stat",
            [[1, "fuel_cell_line", 1, 2, 0.01, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [
                [1, "voltage_source", 1, "V", 750, 0, 0, 1],
                [2, "fuel_cell_source", 2, "P", 750, 10, 0, 1],
            ],
        ),
        "HydroLoad": _table(
            "idx name run_stat",
            [[1, "hydrogen_load", 1]],
        ),
        "Hydro2DcE": _table(
            "idx name run_stat idx_dc_unit_t1 idx_h2_load_t2",
            [[1, "fuel_cell", 0, 2, 1]],
        ),
    }

    ppc = build_dc_ppc_with_topology_from_efile_rows(Path("dc_composite_state.e"), rows)

    assert rows["DCGenerator"]["rows"][1][-1] == 1
    assert ppc["gen"][1, GEN_COLS["run_stat"]] == 0
    assert ppc["_topology_arrays"].devices["gen"].alive_mask.tolist() == [False, False]
    override = next(
        item
        for item in ppc["_effective_run_state_overrides"]
        if item["dev_type"] == "DCGenerator"
    )
    assert override["ancestor_type"] == "Hydro2DcE"


def test_load_flow_writes_zero_results_for_effectively_offline_composite_child():
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _build_lf_network_from_single_ac_file

    rows = _composite_ac_rows(parent_run_stat=0, load_run_stat=1)
    rows["ACLoad"]["rows"].append(
        [2, "base_load", 2, 5, 1, 0, 0, 1, 1, 0, 0, 1]
    )
    network = _build_lf_network_from_single_ac_file(Path("composite_flow.e"), rows)

    calc = HybridPowerFlowCalc(network, verbose=False)
    assert calc.run() == 0
    assert calc.converged

    composite_load = next(load for load in network.ac.loads if load.name == "electrolyzer_load")
    base_load = next(load for load in network.ac.loads if load.name == "base_load")
    assert not composite_load.is_alive
    assert (composite_load.p, composite_load.q, composite_load.current) == (0.0, 0.0, 0.0)
    assert base_load.is_alive
    assert base_load.p > 0


def test_hybrid_ppc_boundary_propagates_state_to_referenced_dcac_converter():
    from model.hybrid_array_model import DCAC_COLS, build_hybrid_ppc_only_from_efile_rows

    rows = {
        "Model": _table(
            "path name p_base u_unit p_unit i_unit",
            [["test", "hybrid_composite_state", 100, "V", "kW", "A"]],
        ),
        "ACNode": _table(
            "idx name vbase voltage angle run_stat",
            [[1, "ac_bus", 380, 380, 0, 1]],
        ),
        "ACGenerator": _table(
            "idx name node control_type p_set q_set v_set alpha run_stat",
            [[1, "ac_source", 1, "PH", 0, 0, 380, 1, 1]],
        ),
        "ACLoad": _table(
            "idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat",
            [[1, "ac_load", 1, 5, 1, 0, 0, 1, 1, 0, 0, 1]],
        ),
        "DCNode": _table(
            "idx name vbase voltage run_stat",
            [[1, "dc_bus", 750, 750, 1]],
        ),
        "DCGenerator": _table(
            "idx name node control_type v_set p_set i_set run_stat",
            [[1, "dc_source", 1, "V", 750, 0, 0, 1]],
        ),
        "DCLoad": _table(
            "idx name node pbase pv0 pv1 pv2 run_stat",
            [[1, "dc_load", 1, 5, 1, 0, 0, 1]],
        ),
        "DCACConverter": _table(
            "idx name ac_node dc_node ac_control_type dc_control_type p_ac_set q_ac_set v_ac_set v_dc_set run_stat r1 r2",
            [[1, "grid_converter", 1, 1, "PQ", "NONE", 0, 0, 380, 750, 1, 0, 0]],
        ),
        "ConverterComposite": _table(
            "idx name run_stat idx_dcac_converter_t1",
            [[1, "converter_assembly", 0, 1]],
        ),
    }

    ppc = build_hybrid_ppc_only_from_efile_rows(Path("hybrid_composite_state.e"), rows)

    assert rows["DCACConverter"]["rows"][0][10] == 1
    assert ppc["dcac"][0, DCAC_COLS["run_stat"]] == 0
    override = next(
        item
        for item in ppc["_effective_run_state_overrides"]
        if item["dev_type"] == "DCACConverter"
    )
    assert override["ancestor_name"] == "converter_assembly"
