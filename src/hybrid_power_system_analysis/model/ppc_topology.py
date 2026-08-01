"""Shared E-file to PPC loaders with precomputed topology arrays."""

import hashlib
from pathlib import Path
from typing import Dict

import numpy as np

from efile_read import _read_efile_rows
from model import topology as network_topology
from model.ac_array_model import (
    ACAC_COLS as AC_ACAC_COLS,
    BRANCH_COLS as AC_BRANCH_COLS,
    BREAK_COLS as AC_BREAK_COLS,
    BUS_COLS as AC_BUS_COLS,
    CTRL_SLACK as AC_CTRL_SLACK,
    GEN_COLS as AC_GEN_COLS,
    LOAD_COLS as AC_LOAD_COLS,
    SWITCH_COLS as AC_SWITCH_COLS,
    THREE_WINDING_TRANSFORMER_COLS as AC_THREE_WINDING_TRANSFORMER_COLS,
    TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
    ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
    build_ac_ppc_from_e_file,
    build_ac_ppc_from_efile_rows,
    ensure_ac_ppc_gen_columns,
)
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    BUS_COLS as DC_BUS_COLS,
    DCDC_COLS as DC_DCDC_COLS,
    CTRL_V as DC_CTRL_V,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    build_dc_ppc_from_e_file,
    build_dc_ppc_from_efile_rows,
)
from model.hybrid_array_model import (
    DCAC_COLS,
    build_hybrid_ppc_only_from_efile_rows,
)


def _update_signature_table(digest, ppc: Dict, key: str, columns) -> None:
    table = np.asarray(ppc.get(key, ()), dtype=np.float64)
    digest.update(key.encode("ascii"))
    digest.update(np.asarray(table.shape, dtype=np.int64).tobytes())
    if table.ndim == 2 and table.size:
        values = np.ascontiguousarray(table[:, columns])
        digest.update(values.tobytes())


def _ac_topology_signature(ppc: Dict) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    _update_signature_table(
        digest,
        ppc,
        "gen",
        [
            AC_GEN_COLS["idx"],
            AC_GEN_COLS["node"],
            AC_GEN_COLS["control_type"],
            AC_GEN_COLS["p_set"],
            AC_GEN_COLS["alpha"],
            AC_GEN_COLS["run_stat"],
            AC_GEN_COLS["p_max"],
        ],
    )
    _update_signature_table(digest, ppc, "bus", [AC_BUS_COLS["idx"], AC_BUS_COLS["run_stat"]])
    _update_signature_table(
        digest,
        ppc,
        "load",
        [AC_LOAD_COLS["node"], AC_LOAD_COLS["run_stat"]],
    )
    _update_signature_table(
        digest,
        ppc,
        "branch",
        [AC_BRANCH_COLS["i_node"], AC_BRANCH_COLS["j_node"], AC_BRANCH_COLS["run_stat"]],
    )
    _update_signature_table(
        digest,
        ppc,
        "transformer",
        [AC_TRANSFORMER_COLS["i_node"], AC_TRANSFORMER_COLS["j_node"], AC_TRANSFORMER_COLS["run_stat"]],
    )
    _update_signature_table(
        digest,
        ppc,
        "three_winding_transformer",
        [
            AC_THREE_WINDING_TRANSFORMER_COLS["i_node"],
            AC_THREE_WINDING_TRANSFORMER_COLS["j_node"],
            AC_THREE_WINDING_TRANSFORMER_COLS["k_node"],
            AC_THREE_WINDING_TRANSFORMER_COLS["run_stat"],
        ],
    )
    _update_signature_table(
        digest,
        ppc,
        "zero_branch",
        [AC_ZERO_BRANCH_COLS["i_node"], AC_ZERO_BRANCH_COLS["j_node"], AC_ZERO_BRANCH_COLS["run_stat"]],
    )
    for key, columns in (
        (
            "switch",
            [
                AC_SWITCH_COLS["i_node"],
                AC_SWITCH_COLS["j_node"],
                AC_SWITCH_COLS["status"],
                AC_SWITCH_COLS["run_stat"],
            ],
        ),
        (
            "break",
            [
                AC_BREAK_COLS["i_node"],
                AC_BREAK_COLS["j_node"],
                AC_BREAK_COLS["status"],
                AC_BREAK_COLS["run_stat"],
            ],
        ),
        (
            "acac",
            [AC_ACAC_COLS["i_node"], AC_ACAC_COLS["j_node"], AC_ACAC_COLS["run_stat"]],
        ),
    ):
        _update_signature_table(digest, ppc, key, columns)
    external_refs = np.ascontiguousarray(
        np.asarray(ppc.get("_external_angle_reference_node_ids", ()), dtype=np.int64)
    )
    digest.update(np.asarray(external_refs.shape, dtype=np.int64).tobytes())
    digest.update(external_refs.tobytes())
    digest.update(bytes((bool(ppc.get("_defer_operational_island_filter", False)),)))
    return digest.digest()


def _dc_topology_signature(ppc: Dict) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    _update_signature_table(
        digest,
        ppc,
        "gen",
        [
            DC_GEN_COLS["idx"],
            DC_GEN_COLS["node"],
            DC_GEN_COLS["control_type"],
            DC_GEN_COLS["run_stat"],
        ],
    )
    _update_signature_table(digest, ppc, "bus", [DC_BUS_COLS["idx"], DC_BUS_COLS["run_stat"]])
    _update_signature_table(
        digest,
        ppc,
        "load",
        [DC_LOAD_COLS["node"], DC_LOAD_COLS["run_stat"]],
    )
    _update_signature_table(
        digest,
        ppc,
        "branch",
        [DC_BRANCH_COLS["i_node"], DC_BRANCH_COLS["j_node"], DC_BRANCH_COLS["run_stat"]],
    )
    _update_signature_table(
        digest,
        ppc,
        "zero_branch",
        [
            DC_ZERO_BRANCH_COLS["i_node"],
            DC_ZERO_BRANCH_COLS["j_node"],
            DC_ZERO_BRANCH_COLS["run_stat"],
        ],
    )
    for key, columns in (
        (
            "switch",
            [
                DC_SWITCH_COLS["i_node"],
                DC_SWITCH_COLS["j_node"],
                DC_SWITCH_COLS["status"],
                DC_SWITCH_COLS["run_stat"],
            ],
        ),
        (
            "break",
            [
                DC_BREAK_COLS["i_node"],
                DC_BREAK_COLS["j_node"],
                DC_BREAK_COLS["status"],
                DC_BREAK_COLS["run_stat"],
            ],
        ),
        (
            "dcdc",
            [
                DC_DCDC_COLS["i_node"],
                DC_DCDC_COLS["j_node"],
                DC_DCDC_COLS["i_control_type"],
                DC_DCDC_COLS["j_control_type"],
                DC_DCDC_COLS["run_stat"],
            ],
        ),
    ):
        _update_signature_table(digest, ppc, key, columns)
    external_refs = np.ascontiguousarray(
        np.asarray(ppc.get("_external_voltage_reference_node_ids", ()), dtype=np.int64)
    )
    digest.update(np.asarray(external_refs.shape, dtype=np.int64).tobytes())
    digest.update(external_refs.tobytes())
    digest.update(bytes((bool(ppc.get("_defer_operational_island_filter", False)),)))
    return digest.digest()


def _hybrid_converter_topology_signature(ppc: Dict) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    _update_signature_table(
        digest,
        ppc,
        "dcac",
        [
            DCAC_COLS["ac_node"],
            DCAC_COLS["dc_node"],
            DCAC_COLS["ac_control_type"],
            DCAC_COLS["dc_control_type"],
            DCAC_COLS["run_stat"],
        ],
    )
    return digest.digest()


def ensure_ac_ppc_topology(ppc: Dict) -> Dict:
    """Attach AC PPC topology arrays when they are not already present."""
    ensure_ac_ppc_gen_columns(ppc)
    signature = _ac_topology_signature(ppc)
    if ppc.get("_topology_arrays") is None or ppc.get("_ac_topology_signature") != signature:
        ppc.pop("_topology_input", None)
        ppc["_topology_arrays"] = network_topology.prepare_ac_topology_ppc(ppc)
        ppc["_ac_topology_signature"] = signature
    return ppc


def ensure_dc_ppc_topology(ppc: Dict) -> Dict:
    """Attach DC PPC topology arrays when they are not already present."""
    signature = _dc_topology_signature(ppc)
    if ppc.get("_topology_arrays") is None or ppc.get("_dc_topology_signature") != signature:
        ppc.pop("_topology_input", None)
        ppc["_topology_arrays"] = network_topology.prepare_dc_topology_ppc(ppc)
        ppc["_dc_topology_signature"] = signature
    return ppc


def ensure_hybrid_ppc_topology(ppc: Dict) -> Dict:
    """Attach AC/DC topology arrays to a hybrid PPC dictionary."""
    ac_ppc = ppc.get("ac")
    dc_ppc = ppc.get("dc")
    hybrid_signature = _hybrid_converter_topology_signature(ppc)
    if ppc.get("_hybrid_converter_topology_signature") != hybrid_signature:
        if ac_ppc is not None:
            ac_ppc.pop("_topology_arrays", None)
        if dc_ppc is not None:
            dc_ppc.pop("_topology_arrays", None)
        ppc["_hybrid_converter_topology_signature"] = hybrid_signature
    if ac_ppc is not None:
        ac_ppc["_defer_operational_island_filter"] = True
    if dc_ppc is not None:
        dc_ppc["_defer_operational_island_filter"] = True
    if dc_ppc is not None:
        _attach_hybrid_dc_reference_nodes(ppc)
        ensure_dc_ppc_topology(dc_ppc)
    if ac_ppc is not None:
        _attach_hybrid_ac_reference_nodes(ppc)
        ensure_ac_ppc_topology(ac_ppc)
    _apply_hybrid_operational_island_filter(ppc)
    return ppc


def _apply_hybrid_operational_island_filter(ppc: Dict) -> None:
    ac_ppc = ppc.get("ac")
    dc_ppc = ppc.get("dc")
    ac_topology = None if ac_ppc is None else ac_ppc.get("_topology_arrays")
    dc_topology = None if dc_ppc is None else dc_ppc.get("_topology_arrays")
    if ac_topology is None and dc_topology is None:
        return

    ac_input = None if ac_ppc is None else ac_ppc.get("_topology_input")
    dc_input = None if dc_ppc is None else dc_ppc.get("_topology_input")
    ac_linked = ()
    dc_linked = ()
    if ac_input is not None:
        ac_linked = (ac_input.terminals.get("acac"),)
    if dc_input is not None:
        dc_linked = (dc_input.terminals.get("dcdc"),)

    dcac = np.asarray(ppc.get("dcac", ()), dtype=np.float64)
    if dcac.size:
        dcac_ac_nodes = dcac[:, DCAC_COLS["ac_node"]]
        dcac_dc_nodes = dcac[:, DCAC_COLS["dc_node"]]
        dcac_run = dcac[:, DCAC_COLS["run_stat"]].astype(np.int64, copy=False) == 1
    else:
        dcac_ac_nodes = np.empty(0, dtype=np.int64)
        dcac_dc_nodes = np.empty(0, dtype=np.int64)
        dcac_run = np.empty(0, dtype=bool)

    ac_gen_rows = np.empty(0, dtype=np.int32)
    ac_gen_islands = np.empty(0, dtype=np.int32)
    ac_load_islands = np.empty(0, dtype=np.int32)
    ac_balance_islands = np.empty(0, dtype=np.int32)
    if ac_topology is not None and ac_input is not None:
        ac_gen_rows, ac_gen_islands = network_topology._online_single_device_rows_and_islands(
            ac_topology,
            ac_input.singles.get("gen"),
        )
        _ac_load_rows, ac_load_islands = network_topology._online_single_device_rows_and_islands(
            ac_topology,
            ac_input.singles.get("load"),
        )
        if ac_gen_rows.size:
            ac_gen = np.asarray(ac_ppc["gen"], dtype=np.float64)
            balance_rows = (
                ac_gen[ac_gen_rows, AC_GEN_COLS["control_type"]].astype(
                    np.int64,
                    copy=False,
                )
                == AC_CTRL_SLACK
            )
            auto_slack_rows = np.asarray(
                ac_ppc.get("_auto_slack_gen_rows", ()),
                dtype=np.int32,
            )
            if auto_slack_rows.size:
                balance_rows |= np.isin(ac_gen_rows, auto_slack_rows)
            ac_balance_islands = ac_gen_islands[balance_rows]

    dc_gen_rows = np.empty(0, dtype=np.int32)
    dc_gen_islands = np.empty(0, dtype=np.int32)
    dc_load_islands = np.empty(0, dtype=np.int32)
    dc_balance_islands = np.empty(0, dtype=np.int32)
    if dc_topology is not None and dc_input is not None:
        dc_gen_rows, dc_gen_islands = network_topology._online_single_device_rows_and_islands(
            dc_topology,
            dc_input.singles.get("gen"),
        )
        _dc_load_rows, dc_load_islands = network_topology._online_single_device_rows_and_islands(
            dc_topology,
            dc_input.singles.get("load"),
        )
        if dc_gen_rows.size:
            dc_gen = np.asarray(dc_ppc["gen"], dtype=np.float64)
            balance_rows = (
                dc_gen[dc_gen_rows, DC_GEN_COLS["control_type"]].astype(
                    np.int64,
                    copy=False,
                )
                == DC_CTRL_V
            )
            dc_balance_islands = dc_gen_islands[balance_rows]

    ac_operational, dc_operational = network_topology.hybrid_operational_island_masks(
        ac_topology,
        dc_topology,
        ac_balance_generator_islands=ac_balance_islands,
        ac_generator_islands=ac_gen_islands,
        ac_load_islands=ac_load_islands,
        dc_balance_generator_islands=dc_balance_islands,
        dc_generator_islands=dc_gen_islands,
        dc_load_islands=dc_load_islands,
        ac_linked_terminals=ac_linked,
        dc_linked_terminals=dc_linked,
        dcac_ac_node_ids=dcac_ac_nodes,
        dcac_dc_node_ids=dcac_dc_nodes,
        dcac_run_mask=dcac_run,
    )
    if ac_topology is not None:
        network_topology.apply_operational_island_filter(
            ac_topology,
            ac_operational,
            ac_input,
        )
        auto_slack_rows = np.asarray(ac_ppc.get("_auto_slack_gen_rows", ()), dtype=np.int32)
        gen_topology = ac_topology.devices.get("gen")
        if auto_slack_rows.size and gen_topology is not None:
            valid = (auto_slack_rows >= 0) & (auto_slack_rows < gen_topology.alive_mask.size)
            auto_slack_rows = auto_slack_rows[valid]
            ac_ppc["_auto_slack_gen_rows"] = auto_slack_rows[
                gen_topology.alive_mask[auto_slack_rows]
            ]
    if dc_topology is not None:
        network_topology.apply_operational_island_filter(
            dc_topology,
            dc_operational,
            dc_input,
        )


def _attach_hybrid_dc_reference_nodes(ppc: Dict) -> None:
    ac_ppc = ppc.get("ac")
    dc_ppc = ppc.get("dc")
    dcac = ppc.get("dcac")
    if dc_ppc is None or dcac is None or getattr(dcac, "size", 0) == 0:
        return
    ctrl = dcac[:, DCAC_COLS["dc_control_type"]].astype(int, copy=False)
    run = dcac[:, DCAC_COLS["run_stat"]].astype(int, copy=False) == 1
    dcv_mask = run & (ctrl == 2)
    if ac_ppc is not None and dcv_mask.any():
        ac_bus = ac_ppc.get("bus")
        if ac_bus is not None and getattr(ac_bus, "size", 0):
            active_ac_ids = ac_bus[
                ac_bus[:, AC_BUS_COLS["run_stat"]].astype(int, copy=False) == 1,
                AC_BUS_COLS["idx"],
            ].astype(int, copy=False)
            dcv_mask &= np.isin(dcac[:, DCAC_COLS["ac_node"]].astype(int, copy=False), active_ac_ids)
    if not dcv_mask.any():
        had_external_refs = "_external_voltage_reference_node_ids" in dc_ppc or "_external_voltage_reference_pu" in dc_ppc
        dc_ppc.pop("_external_voltage_reference_node_ids", None)
        dc_ppc.pop("_external_voltage_reference_pu", None)
        if had_external_refs:
            dc_ppc.pop("_topology_arrays", None)
        return

    ref_nodes, unique_pos = np.unique(
        dcac[dcv_mask, DCAC_COLS["dc_node"]].astype(int, copy=False),
        return_index=True,
    )
    if ref_nodes.size:
        ref_values = dcac[dcv_mask, DCAC_COLS["v_dc_set"]][unique_pos].astype(
            float,
            copy=False,
        )
        old_nodes = np.asarray(dc_ppc.get("_external_voltage_reference_node_ids", []), dtype=int)
        old_values = np.asarray(dc_ppc.get("_external_voltage_reference_pu", []), dtype=float)
        changed = (
            old_nodes.shape != ref_nodes.shape
            or old_values.shape != ref_values.shape
            or not np.array_equal(old_nodes, ref_nodes)
            or not np.allclose(old_values, ref_values)
        )
        dc_ppc["_external_voltage_reference_node_ids"] = ref_nodes.astype(int, copy=False)
        dc_ppc["_external_voltage_reference_pu"] = ref_values
        if changed:
            dc_ppc.pop("_topology_arrays", None)
    else:
        had_external_refs = "_external_voltage_reference_node_ids" in dc_ppc or "_external_voltage_reference_pu" in dc_ppc
        dc_ppc.pop("_external_voltage_reference_node_ids", None)
        dc_ppc.pop("_external_voltage_reference_pu", None)
        if had_external_refs:
            dc_ppc.pop("_topology_arrays", None)


def _attach_hybrid_ac_reference_nodes(ppc: Dict) -> None:
    ac_ppc = ppc.get("ac")
    dc_ppc = ppc.get("dc")
    dcac = ppc.get("dcac")
    if ac_ppc is None or dc_ppc is None or dcac is None or getattr(dcac, "size", 0) == 0:
        return
    dcac = dcac
    ctrl = dcac[:, DCAC_COLS["ac_control_type"]].astype(int, copy=False)
    run = dcac[:, DCAC_COLS["run_stat"]].astype(int, copy=False) == 1
    acv_mask = run & (ctrl == 3)
    if not acv_mask.any():
        had_external_refs = "_external_angle_reference_node_ids" in ac_ppc or "_external_voltage_reference_pu" in ac_ppc
        ac_ppc.pop("_external_angle_reference_node_ids", None)
        ac_ppc.pop("_external_voltage_reference_pu", None)
        if had_external_refs:
            ac_ppc.pop("_topology_arrays", None)
        return

    dc_topology = dc_ppc.get("_topology_arrays")
    if dc_topology is not None:
        dc_bus = dc_ppc.get("bus")
        alive_dc_ids = dc_bus[dc_topology.node_alive_mask, DC_BUS_COLS["idx"]].astype(int, copy=False)
        acv_mask &= np.isin(dcac[:, DCAC_COLS["dc_node"]].astype(int, copy=False), alive_dc_ids)

    ref_nodes, unique_pos = np.unique(
        dcac[acv_mask, DCAC_COLS["ac_node"]].astype(int, copy=False),
        return_index=True,
    )
    if ref_nodes.size:
        ref_values = dcac[acv_mask, DCAC_COLS["v_ac_set"]][unique_pos].astype(
            float,
            copy=False,
        )
        old_nodes = np.asarray(ac_ppc.get("_external_angle_reference_node_ids", []), dtype=int)
        old_values = np.asarray(ac_ppc.get("_external_voltage_reference_pu", []), dtype=float)
        changed = (
            old_nodes.shape != ref_nodes.shape
            or old_values.shape != ref_values.shape
            or not np.array_equal(old_nodes, ref_nodes)
            or not np.allclose(old_values, ref_values)
        )
        ac_ppc["_external_angle_reference_node_ids"] = ref_nodes.astype(int, copy=False)
        ac_ppc["_external_voltage_reference_pu"] = ref_values
        if changed:
            ac_ppc.pop("_topology_arrays", None)
    else:
        had_external_refs = "_external_angle_reference_node_ids" in ac_ppc or "_external_voltage_reference_pu" in ac_ppc
        ac_ppc.pop("_external_angle_reference_node_ids", None)
        ac_ppc.pop("_external_voltage_reference_pu", None)
        if had_external_refs:
            ac_ppc.pop("_topology_arrays", None)


def build_ac_ppc_with_topology_from_e_file(file_path) -> Dict:
    """Read an AC E file once into PPC and attach topology arrays."""
    source = Path(file_path).resolve()
    ppc = build_ac_ppc_from_e_file(source)
    ppc["source"] = str(source)
    return ensure_ac_ppc_topology(ppc)


def build_ac_ppc_with_topology_from_efile_rows(file_path, rows) -> Dict:
    """Build AC PPC from already loaded E rows and attach topology arrays."""
    source = Path(file_path).resolve()
    ppc = build_ac_ppc_from_efile_rows(source, rows)
    ppc["source"] = str(source)
    return ensure_ac_ppc_topology(ppc)


def build_dc_ppc_with_topology_from_e_file(file_path) -> Dict:
    """Read a DC E file once into PPC and attach topology arrays."""
    source = Path(file_path).resolve()
    ppc = build_dc_ppc_from_e_file(source)
    ppc["source"] = str(source)
    return ensure_dc_ppc_topology(ppc)


def build_dc_ppc_with_topology_from_efile_rows(file_path, rows) -> Dict:
    """Build DC PPC from already loaded E rows and attach topology arrays."""
    source = Path(file_path).resolve()
    ppc = build_dc_ppc_from_efile_rows(source, rows)
    ppc["source"] = str(source)
    return ensure_dc_ppc_topology(ppc)


def build_hybrid_ppc_with_topology_from_efile_rows(file_path, rows) -> Dict:
    """Build hybrid PPC from already loaded E rows and attach AC/DC topology arrays."""
    source = Path(file_path).resolve()
    ppc = build_hybrid_ppc_only_from_efile_rows(source, rows)
    ppc["source"] = str(source)
    if ppc.get("ac") is not None:
        ppc["ac"]["source"] = str(source)
    if ppc.get("dc") is not None:
        ppc["dc"]["source"] = str(source)
    return ensure_hybrid_ppc_topology(ppc)


def build_hybrid_ppc_with_topology_from_e_file(file_path) -> Dict:
    """Read a hybrid E file once into PPC and attach AC/DC topology arrays."""
    source = Path(file_path).resolve()
    rows = _read_efile_rows(source)
    return build_hybrid_ppc_with_topology_from_efile_rows(source, rows)
