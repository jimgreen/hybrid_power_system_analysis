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
    GEN_COLS as AC_GEN_COLS,
    SWITCH_COLS as AC_SWITCH_COLS,
    THREE_WINDING_TRANSFORMER_COLS as AC_THREE_WINDING_TRANSFORMER_COLS,
    TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
    ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
    build_ac_ppc_from_e_file,
    build_ac_ppc_from_efile_rows,
    ensure_ac_ppc_gen_columns,
)
from model.dc_array_model import BUS_COLS as DC_BUS_COLS
from model.dc_array_model import build_dc_ppc_from_e_file, build_dc_ppc_from_efile_rows
from model.hybrid_array_model import (
    DCAC_COLS,
    DCAC_CONTROL_CODE,
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
    topology = ppc.get("_topology_arrays")
    if topology is None:
        topology = network_topology.prepare_dc_topology_ppc(ppc)
        ppc["_topology_arrays"] = topology
    return ppc


def ensure_hybrid_ppc_topology(ppc: Dict) -> Dict:
    """Attach AC/DC topology arrays to a hybrid PPC dictionary."""
    dc_ppc = ppc.get("dc")
    if dc_ppc is not None:
        _attach_hybrid_dc_reference_nodes(ppc)
        ensure_dc_ppc_topology(dc_ppc)
    ac_ppc = ppc.get("ac")
    if ac_ppc is not None:
        _attach_hybrid_ac_reference_nodes(ppc)
        ensure_ac_ppc_topology(ac_ppc)
    return ppc


def _attach_hybrid_dc_reference_nodes(ppc: Dict) -> None:
    ac_ppc = ppc.get("ac")
    dc_ppc = ppc.get("dc")
    dcac = ppc.get("dcac")
    if dc_ppc is None or dcac is None or getattr(dcac, "size", 0) == 0:
        return
    ctrl = dcac[:, DCAC_COLS["control_type"]].astype(int, copy=False)
    run = dcac[:, DCAC_COLS["run_stat"]].astype(int, copy=False) == 1
    dcv_mask = run & (ctrl == DCAC_CONTROL_CODE["DCV"])
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
    ctrl = dcac[:, DCAC_COLS["control_type"]].astype(int, copy=False)
    run = dcac[:, DCAC_COLS["run_stat"]].astype(int, copy=False) == 1
    acv_mask = run & (ctrl == DCAC_CONTROL_CODE["ACV"])
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
