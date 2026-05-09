import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
for path in (MODEL_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_array_model import build_ac_ppc_from_model
from dc_array_model import build_dc_ppc_from_model
from efile_read import efile_factory_from_file
from unit_system import normalize_model_named_units


DCAC_CONTROL_CODE = {"DCV": 0, "ACV": 1, "ACP": 2}
DCAC_CONTROL_LABEL = {value: key for key, value in DCAC_CONTROL_CODE.items()}
ACAC_CONTROL_CODE = {"PQQ": 0, "PVQ": 1, "PQV": 2, "PVV": 3}
ACAC_CONTROL_LABEL = {value: key for key, value in ACAC_CONTROL_CODE.items()}

DCAC_COLS = {
    "idx": 0,
    "ac_node": 1,
    "dc_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_ac_set": 6,
    "q_ac_set": 7,
    "v_ac_set": 8,
    "v_dc_set": 9,
    "run_stat": 10,
    "dc_p": 11,
    "ac_p": 12,
    "ac_q": 13,
    "dc_i": 14,
    "ac_i": 15,
}
ACAC_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_set": 6,
    "i_q_set": 7,
    "j_q_set": 8,
    "i_v_set": 9,
    "j_v_set": 10,
    "run_stat": 11,
    "i_p": 12,
    "i_q": 13,
    "j_p": 14,
    "j_q": 15,
    "i_i": 16,
    "j_i": 17,
}

def _attr(obj, attr: str, default=""):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_attr(obj, attr: str, default: float = 0.0) -> float:
    value = _attr(obj, attr, default)
    return default if value in (None, "") else float(value)


def _int_attr(obj, attr: str, default: int = 0) -> int:
    value = _attr(obj, attr, default)
    return default if value in (None, "") else int(float(value))


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _names(devices, count: int, fallback_prefix: str) -> np.ndarray:
    return np.asarray(
        [
            str(_attr(dev, "name", f"{fallback_prefix}_{idx}") or f"{fallback_prefix}_{idx}")
            for idx, dev in enumerate(devices[:count])
        ],
        dtype=object,
    )


def _build_dcac(model) -> Tuple[np.ndarray, np.ndarray]:
    converters = list(getattr(model, "DCACConverter", []))
    if not converters:
        return _empty(len(DCAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(converters), len(DCAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "dcac")
    for pos, conv in enumerate(converters):
        out[pos, DCAC_COLS["idx"]] = _int_attr(conv, "idx", pos)
        out[pos, DCAC_COLS["ac_node"]] = _int_attr(conv, "ac_node")
        out[pos, DCAC_COLS["dc_node"]] = _int_attr(conv, "dc_node")
        out[pos, DCAC_COLS["r1"]] = _float_attr(conv, "r1")
        out[pos, DCAC_COLS["r2"]] = _float_attr(conv, "r2")
        out[pos, DCAC_COLS["control_type"]] = DCAC_CONTROL_CODE.get(
            str(_attr(conv, "control_type", "DCV")).upper(),
            0,
        )
        out[pos, DCAC_COLS["p_ac_set"]] = _float_attr(conv, "p_ac_set")
        out[pos, DCAC_COLS["q_ac_set"]] = _float_attr(conv, "q_ac_set")
        out[pos, DCAC_COLS["v_ac_set"]] = _float_attr(conv, "v_ac_set")
        out[pos, DCAC_COLS["v_dc_set"]] = _float_attr(conv, "v_dc_set")
        out[pos, DCAC_COLS["run_stat"]] = _float_attr(conv, "run_stat", 1.0)
        out[pos, DCAC_COLS["dc_p"]] = _float_attr(conv, "dc_p")
        out[pos, DCAC_COLS["ac_p"]] = _float_attr(conv, "ac_p")
        out[pos, DCAC_COLS["ac_q"]] = _float_attr(conv, "ac_q")
        out[pos, DCAC_COLS["dc_i"]] = _float_attr(conv, "dc_i")
        out[pos, DCAC_COLS["ac_i"]] = _float_attr(conv, "ac_i")
    return out, names


def _build_acac(model) -> Tuple[np.ndarray, np.ndarray]:
    converters = list(getattr(model, "ACACConverter", []))
    if not converters:
        return _empty(len(ACAC_COLS)), np.asarray([], dtype=object)
    out = np.zeros((len(converters), len(ACAC_COLS)), dtype=np.float64)
    names = _names(converters, len(converters), "acac")
    for pos, conv in enumerate(converters):
        out[pos, ACAC_COLS["idx"]] = _int_attr(conv, "idx", pos)
        out[pos, ACAC_COLS["i_node"]] = _int_attr(conv, "i_node")
        out[pos, ACAC_COLS["j_node"]] = _int_attr(conv, "j_node")
        out[pos, ACAC_COLS["r1"]] = _float_attr(conv, "r1")
        out[pos, ACAC_COLS["r2"]] = _float_attr(conv, "r2")
        out[pos, ACAC_COLS["control_type"]] = ACAC_CONTROL_CODE.get(
            str(_attr(conv, "control_type", "PQQ")).upper(),
            0,
        )
        out[pos, ACAC_COLS["p_set"]] = _float_attr(conv, "p_set")
        out[pos, ACAC_COLS["i_q_set"]] = _float_attr(conv, "i_q_set")
        out[pos, ACAC_COLS["j_q_set"]] = _float_attr(conv, "j_q_set")
        out[pos, ACAC_COLS["i_v_set"]] = _float_attr(conv, "i_v_set")
        out[pos, ACAC_COLS["j_v_set"]] = _float_attr(conv, "j_v_set")
        out[pos, ACAC_COLS["run_stat"]] = _float_attr(conv, "run_stat", 1.0)
        out[pos, ACAC_COLS["i_p"]] = _float_attr(conv, "i_p")
        out[pos, ACAC_COLS["i_q"]] = _float_attr(conv, "i_q")
        out[pos, ACAC_COLS["j_p"]] = _float_attr(conv, "j_p")
        out[pos, ACAC_COLS["j_q"]] = _float_attr(conv, "j_q")
        out[pos, ACAC_COLS["i_i"]] = _float_attr(conv, "i_i")
        out[pos, ACAC_COLS["j_i"]] = _float_attr(conv, "j_i")
    return out, names


def build_hybrid_ppc_from_e_file(file_path):
    model = efile_factory_from_file(file_path)
    normalize_model_named_units(model)
    ac_network, ac_ppc = build_ac_ppc_from_model(model)
    ac_ppc["source"] = str(file_path)
    ac_network.ppc = ac_ppc
    dc_network, dc_ppc = build_dc_ppc_from_model(model)
    dc_network.ppc = dc_ppc
    dcac, dcac_name = _build_dcac(model)
    acac, acac_name = _build_acac(model)
    ppc = {
        "format": "hybrid_ppc_v1",
        "source": str(file_path),
        "base": ac_ppc["base"],
        "ac": ac_ppc,
        "dc": dc_ppc,
        "ac_network": ac_network,
        "dc_network": dc_network,
        "dcac": dcac,
        "acac": acac,
        "dcac_name": dcac_name,
        "acac_name": acac_name,
        "dcac_cols": DCAC_COLS,
        "acac_cols": ACAC_COLS,
    }
    network = build_hybrid_model_from_ppc(ppc)
    return network, ppc



def build_hybrid_model_from_ppc(ppc: Dict):
    if __package__ == "model":
        from model.hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork
    else:
        from hybrid_model import ACACConverter, DCACConverter, HybridPowerNetwork

    ac_network = ppc["ac_network"]
    dc_network = ppc["dc_network"]
    dcac = [
        DCACConverter(
            int(row[DCAC_COLS["idx"]]),
            int(row[DCAC_COLS["ac_node"]]),
            int(row[DCAC_COLS["dc_node"]]),
            float(row[DCAC_COLS["r1"]]),
            float(row[DCAC_COLS["r2"]]),
            DCAC_CONTROL_LABEL.get(int(row[DCAC_COLS["control_type"]]), "DCV"),
            float(row[DCAC_COLS["p_ac_set"]]),
            float(row[DCAC_COLS["q_ac_set"]]),
            float(row[DCAC_COLS["v_ac_set"]]),
            float(row[DCAC_COLS["v_dc_set"]]),
            int(row[DCAC_COLS["run_stat"]]),
        )
        for pos, row in enumerate(ppc["dcac"])
    ]
    acac = [
        ACACConverter(
            int(row[ACAC_COLS["idx"]]),
            int(row[ACAC_COLS["i_node"]]),
            int(row[ACAC_COLS["j_node"]]),
            float(row[ACAC_COLS["r1"]]),
            float(row[ACAC_COLS["r2"]]),
            ACAC_CONTROL_LABEL.get(int(row[ACAC_COLS["control_type"]]), "PQQ"),
            float(row[ACAC_COLS["p_set"]]),
            float(row[ACAC_COLS["i_q_set"]]),
            float(row[ACAC_COLS["j_q_set"]]),
            float(row[ACAC_COLS["i_v_set"]]),
            float(row[ACAC_COLS["j_v_set"]]),
            int(row[ACAC_COLS["run_stat"]]),
        )
        for pos, row in enumerate(ppc["acac"])
    ]
    for pos, conv in enumerate(dcac):
        conv.name = str(ppc["dcac_name"][pos])
        row = ppc["dcac"][pos]
        conv.dc_p = float(row[DCAC_COLS["dc_p"]])
        conv.ac_p = float(row[DCAC_COLS["ac_p"]])
        conv.ac_q = float(row[DCAC_COLS["ac_q"]])
        conv.dc_i = float(row[DCAC_COLS["dc_i"]])
        conv.ac_i = float(row[DCAC_COLS["ac_i"]])
        conv.is_alive = False
    for pos, conv in enumerate(acac):
        conv.name = str(ppc["acac_name"][pos])
        row = ppc["acac"][pos]
        conv.i_p = float(row[ACAC_COLS["i_p"]])
        conv.i_q = float(row[ACAC_COLS["i_q"]])
        conv.j_p = float(row[ACAC_COLS["j_p"]])
        conv.j_q = float(row[ACAC_COLS["j_q"]])
        conv.i_i = float(row[ACAC_COLS["i_i"]])
        conv.j_i = float(row[ACAC_COLS["j_i"]])
        conv.is_alive = False

    model = HybridPowerNetwork(ac=ac_network, dc=dc_network, dcac_converters=dcac, acac_converters=acac)
    base = ppc["base"]
    model.p_base = float(base[0])
    model.u_scale = float(base[1])
    model.p_scale = float(base[2])
    model.i_scale = float(base[3])
    model.p_base_kW = float(base[4])
    model.ac.ppc = ppc["ac"]
    model.dc.ppc = ppc["dc"]
    model.ppc = ppc
    model._ac_ppc = ppc["ac"]
    model._dc_ppc = ppc["dc"]
    model.ACNode = ac_network.nodes
    model.ACBranch = ac_network.branches
    model.ACLoad = ac_network.loads
    model.ACGenerator = ac_network.generators
    model.ACZeroBranch = ac_network.zero_branches
    model.ACSwitch = ac_network.switches
    model.ACBreak = getattr(ac_network, "breakers", [])
    model.ACTransformer = ac_network.transformers
    model.ACShuntCompensator = ac_network.shunt_compensators
    model.DCNode = dc_network.nodes
    model.DCBranch = dc_network.branches
    model.DCLoad = dc_network.loads
    model.DCGenerator = dc_network.generators
    model.DCZeroBranch = dc_network.zero_branches
    model.DCSwitch = dc_network.switches
    model.DCBreak = getattr(dc_network, "breakers", [])
    model.DCDCConverter = dc_network.dcdc_converters
    model.DCShuntCompensator = []
    model.DCACConverter = dcac
    model.ACACConverter = acac
    return model
