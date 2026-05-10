"""Shared E-file to PPC loaders with precomputed topology arrays."""

from pathlib import Path
from typing import Dict

from efile_read import _read_efile_rows
from model import topology as network_topology
from model.ac_array_model import build_ac_ppc_from_e_file, build_ac_ppc_from_efile_rows
from model.dc_array_model import build_dc_ppc_from_e_file, build_dc_ppc_from_efile_rows
from model.hybrid_array_model import (
    build_hybrid_ppc_only_from_efile_rows,
)


def ensure_ac_ppc_topology(ppc: Dict) -> Dict:
    """Attach AC PPC topology arrays when they are not already present."""
    if ppc.get("_topology_arrays") is None:
        ppc["_topology_arrays"] = network_topology.prepare_ac_topology_ppc(ppc)
    return ppc


def ensure_dc_ppc_topology(ppc: Dict) -> Dict:
    """Attach DC PPC topology arrays when they are not already present."""
    if ppc.get("_topology_arrays") is None:
        ppc["_topology_arrays"] = network_topology.prepare_dc_topology_ppc(ppc)
    return ppc


def ensure_hybrid_ppc_topology(ppc: Dict) -> Dict:
    """Attach AC/DC topology arrays to a hybrid PPC dictionary."""
    ac_ppc = ppc.get("ac")
    if ac_ppc is not None:
        ensure_ac_ppc_topology(ac_ppc)
    dc_ppc = ppc.get("dc")
    if dc_ppc is not None:
        ensure_dc_ppc_topology(dc_ppc)
    return ppc


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
