import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from efile_read import efile_factory_from_file
from paths import resolve_project_file

MODEL_DIR = Path(__file__).resolve().parent
for path in (MODEL_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CTRL_PQ = 0
CTRL_P = 1
CTRL_PV = 2
CTRL_SLACK = 3
CTRL_CODE = {
    "PQ": CTRL_PQ,
    "P": CTRL_P,
    "PV": CTRL_PV,
    "V": CTRL_SLACK,
    "SLACK": CTRL_SLACK,
    "PH": CTRL_SLACK,
}

SHUNT_Q = 0
SHUNT_V = 1
SHUNT_B = 2
SHUNT_Z = 3
SHUNT_CODE = {
    "Q": SHUNT_Q,
    "V": SHUNT_V,
    "B": SHUNT_B,
    "Z": SHUNT_Z,
}

BUS_COLS = {
    "idx": 0,
    "vbase": 1,
    "voltage": 2,
    "angle": 3,
    "isl": 4,
    "run_stat": 5,
}
BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "b": 5,
    "run_stat": 6,
    "i_p": 7,
    "i_q": 8,
    "i_c": 9,
    "j_p": 10,
    "j_q": 11,
    "j_c": 12,
}
TRANSFORMER_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "b": 5,
    "tap": 6,
    "shift": 7,
    "run_stat": 8,
    "i_p": 9,
    "i_q": 10,
    "i_c": 11,
    "j_p": 12,
    "j_q": 13,
    "j_c": 14,
}
GEN_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "p_set": 3,
    "q_set": 4,
    "v_set": 5,
    "alpha": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
}
LOAD_COLS = {
    "idx": 0,
    "node": 1,
    "pbase": 2,
    "pv0": 3,
    "pv1": 4,
    "pv2": 5,
    "qbase": 6,
    "qv0": 7,
    "qv1": 8,
    "qv2": 9,
    "run_stat": 10,
    "p": 11,
    "q": 12,
    "current": 13,
}
SHUNT_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "q_set": 3,
    "g_set": 4,
    "b_set": 5,
    "v_set": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
}
ZERO_BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "run_stat": 3,
    "p": 4,
    "q": 5,
    "current": 6,
}
SWITCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "status": 3,
    "run_stat": 4,
    "p": 5,
    "q": 6,
    "current": 7,
}
BREAK_COLS = SWITCH_COLS

_AC_PPC_CACHE = {}
_AC_PPC_CACHE_LOCK = threading.Lock()


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def clear_ac_ppc_cache(file_path=None) -> None:
    with _AC_PPC_CACHE_LOCK:
        if file_path is None:
            _AC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _AC_PPC_CACHE.pop(path, None)


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def build_ac_ppc_from_e_file(file_path) -> Dict:
    """Build a MATPOWER-like NumPy dictionary for AC power-flow fast paths.

    The returned arrays are in pu/radians and should be treated as read-only by callers.
    """
    file_key = _file_cache_key(file_path)
    with _AC_PPC_CACHE_LOCK:
        cached = _AC_PPC_CACHE.get(file_key[0])
        if cached is not None and cached[0] == file_key:
            return cached[1]

    model = efile_factory_from_file(file_key[0])
    _network, ppc = build_ac_ppc_from_model(model)
    ppc["source"] = str(file_key[0])
    with _AC_PPC_CACHE_LOCK:
        _AC_PPC_CACHE[file_key[0]] = (file_key, ppc)
    return ppc


def _value(obj, attr: str, default=0.0):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_value(obj, attr: str, default: float = 0.0) -> float:
    return float(_value(obj, attr, default))


def _int_value(obj, attr: str, default: int = 0) -> int:
    return int(float(_value(obj, attr, default)))


def _name_array(devices, prefix: str) -> np.ndarray:
    return np.asarray(
        [str(getattr(dev, "name", "") or f"{prefix}_{_int_value(dev, 'idx', pos)}") for pos, dev in enumerate(devices)],
        dtype=object,
    )


def _code_value(value, mapping: Dict[str, int], default_label: str) -> int:
    if value in (None, ""):
        return mapping[default_label]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return mapping.get(str(value).upper(), mapping[default_label])


def build_ac_ppc_from_network(network) -> Dict:
    """Build an AC ppc dictionary from an already loaded ACPowerNetwork."""
    nodes = list(getattr(network, "nodes", []))
    branches = list(getattr(network, "branches", []))
    transformers = list(getattr(network, "transformers", []))
    generators = list(getattr(network, "generators", []))
    loads = list(getattr(network, "loads", []))
    shunts = list(getattr(network, "shunt_compensators", []))
    zero_branches = list(getattr(network, "zero_branches", []))
    switches = list(getattr(network, "switches", []))
    breakers = list(getattr(network, "breakers", []))

    p_base = float(getattr(network, "p_base", 1.0))
    u_scale = float(getattr(network, "u_scale", 1.0))
    p_scale = float(getattr(network, "p_scale", 1.0))
    i_scale = float(getattr(network, "i_scale", 1.0))

    bus = np.zeros((len(nodes), len(BUS_COLS)), dtype=np.float64)
    for row, node in enumerate(nodes):
        bus[row, BUS_COLS["idx"]] = _int_value(node, "idx")
        bus[row, BUS_COLS["vbase"]] = _float_value(node, "vbase")
        bus[row, BUS_COLS["voltage"]] = _float_value(node, "voltage", 1.0)
        bus[row, BUS_COLS["angle"]] = _float_value(node, "angle", 0.0)
        bus[row, BUS_COLS["isl"]] = _float_value(node, "isl", 0.0)
        bus[row, BUS_COLS["run_stat"]] = _float_value(node, "run_stat", 1.0)
    bus_names = _name_array(nodes, "bus")

    branch = np.zeros((len(branches), len(BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(branches):
        branch[row, BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        branch[row, BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        branch[row, BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        branch[row, BRANCH_COLS["r"]] = _float_value(dev, "r")
        branch[row, BRANCH_COLS["x"]] = _float_value(dev, "x")
        branch[row, BRANCH_COLS["b"]] = _float_value(dev, "b")
        branch[row, BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        branch[row, BRANCH_COLS["i_p"]] = _float_value(dev, "i_p")
        branch[row, BRANCH_COLS["i_q"]] = _float_value(dev, "i_q")
        branch[row, BRANCH_COLS["i_c"]] = _float_value(dev, "i_c")
        branch[row, BRANCH_COLS["j_p"]] = _float_value(dev, "j_p")
        branch[row, BRANCH_COLS["j_q"]] = _float_value(dev, "j_q")
        branch[row, BRANCH_COLS["j_c"]] = _float_value(dev, "j_c")

    transformer = np.zeros((len(transformers), len(TRANSFORMER_COLS)), dtype=np.float64)
    for row, dev in enumerate(transformers):
        transformer[row, TRANSFORMER_COLS["idx"]] = _int_value(dev, "idx")
        transformer[row, TRANSFORMER_COLS["i_node"]] = _int_value(dev, "i_node")
        transformer[row, TRANSFORMER_COLS["j_node"]] = _int_value(dev, "j_node")
        transformer[row, TRANSFORMER_COLS["r"]] = _float_value(dev, "r")
        transformer[row, TRANSFORMER_COLS["x"]] = _float_value(dev, "x")
        transformer[row, TRANSFORMER_COLS["b"]] = _float_value(dev, "b")
        transformer[row, TRANSFORMER_COLS["tap"]] = _float_value(dev, "tap", 1.0)
        transformer[row, TRANSFORMER_COLS["shift"]] = _float_value(dev, "shift")
        transformer[row, TRANSFORMER_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        transformer[row, TRANSFORMER_COLS["i_p"]] = _float_value(dev, "i_p")
        transformer[row, TRANSFORMER_COLS["i_q"]] = _float_value(dev, "i_q")
        transformer[row, TRANSFORMER_COLS["i_c"]] = _float_value(dev, "i_c")
        transformer[row, TRANSFORMER_COLS["j_p"]] = _float_value(dev, "j_p")
        transformer[row, TRANSFORMER_COLS["j_q"]] = _float_value(dev, "j_q")
        transformer[row, TRANSFORMER_COLS["j_c"]] = _float_value(dev, "j_c")

    gen = np.zeros((len(generators), len(GEN_COLS)), dtype=np.float64)
    for row, dev in enumerate(generators):
        gen[row, GEN_COLS["idx"]] = _int_value(dev, "idx")
        gen[row, GEN_COLS["node"]] = _int_value(dev, "node")
        gen[row, GEN_COLS["control_type"]] = _code_value(_value(dev, "control_type", "PQ"), CTRL_CODE, "PQ")
        gen[row, GEN_COLS["p_set"]] = _float_value(dev, "p_set")
        gen[row, GEN_COLS["q_set"]] = _float_value(dev, "q_set")
        gen[row, GEN_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        gen[row, GEN_COLS["alpha"]] = _float_value(dev, "alpha", 1.0)
        gen[row, GEN_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        gen[row, GEN_COLS["p"]] = _float_value(dev, "p")
        gen[row, GEN_COLS["q"]] = _float_value(dev, "q")
        gen[row, GEN_COLS["current"]] = _float_value(dev, "current")

    load = np.zeros((len(loads), len(LOAD_COLS)), dtype=np.float64)
    for row, dev in enumerate(loads):
        load[row, LOAD_COLS["idx"]] = _int_value(dev, "idx")
        load[row, LOAD_COLS["node"]] = _int_value(dev, "node")
        load[row, LOAD_COLS["pbase"]] = _float_value(dev, "pbase", 1.0)
        load[row, LOAD_COLS["pv0"]] = _float_value(dev, "pv0")
        load[row, LOAD_COLS["pv1"]] = _float_value(dev, "pv1")
        load[row, LOAD_COLS["pv2"]] = _float_value(dev, "pv2")
        load[row, LOAD_COLS["qbase"]] = _float_value(dev, "qbase", 1.0)
        load[row, LOAD_COLS["qv0"]] = _float_value(dev, "qv0")
        load[row, LOAD_COLS["qv1"]] = _float_value(dev, "qv1")
        load[row, LOAD_COLS["qv2"]] = _float_value(dev, "qv2")
        load[row, LOAD_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        load[row, LOAD_COLS["p"]] = _float_value(dev, "p")
        load[row, LOAD_COLS["q"]] = _float_value(dev, "q")
        load[row, LOAD_COLS["current"]] = _float_value(dev, "current")

    shunt = np.zeros((len(shunts), len(SHUNT_COLS)), dtype=np.float64)
    for row, dev in enumerate(shunts):
        shunt[row, SHUNT_COLS["idx"]] = _int_value(dev, "idx")
        shunt[row, SHUNT_COLS["node"]] = _int_value(dev, "node")
        shunt[row, SHUNT_COLS["control_type"]] = _code_value(_value(dev, "control_type", "Q"), SHUNT_CODE, "Q")
        shunt[row, SHUNT_COLS["q_set"]] = _float_value(dev, "q_set")
        shunt[row, SHUNT_COLS["g_set"]] = _float_value(dev, "g_set")
        shunt[row, SHUNT_COLS["b_set"]] = _float_value(dev, "b_set")
        shunt[row, SHUNT_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        shunt[row, SHUNT_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        shunt[row, SHUNT_COLS["p"]] = _float_value(dev, "p")
        shunt[row, SHUNT_COLS["q"]] = _float_value(dev, "q")
        shunt[row, SHUNT_COLS["current"]] = _float_value(dev, "current")

    zero_branch = np.zeros((len(zero_branches), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(zero_branches):
        zero_branch[row, ZERO_BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        zero_branch[row, ZERO_BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        zero_branch[row, ZERO_BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        zero_branch[row, ZERO_BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        zero_branch[row, ZERO_BRANCH_COLS["p"]] = _float_value(dev, "p")
        zero_branch[row, ZERO_BRANCH_COLS["q"]] = _float_value(dev, "q")
        zero_branch[row, ZERO_BRANCH_COLS["current"]] = _float_value(dev, "current")

    def build_switch_like(devices):
        out = np.zeros((len(devices), len(SWITCH_COLS)), dtype=np.float64)
        for row, dev in enumerate(devices):
            out[row, SWITCH_COLS["idx"]] = _int_value(dev, "idx")
            out[row, SWITCH_COLS["i_node"]] = _int_value(dev, "i_node")
            out[row, SWITCH_COLS["j_node"]] = _int_value(dev, "j_node")
            out[row, SWITCH_COLS["status"]] = _float_value(dev, "status", 1.0)
            out[row, SWITCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
            out[row, SWITCH_COLS["p"]] = _float_value(dev, "p")
            out[row, SWITCH_COLS["q"]] = _float_value(dev, "q")
            out[row, SWITCH_COLS["current"]] = _float_value(dev, "current")
        return out

    ppc = {
        "format": "ac_ppc_v1",
        "source": str(getattr(network, "source", getattr(network, "file_name", "<network>"))),
        "base": np.asarray([p_base, u_scale, p_scale, i_scale, getattr(network, "p_base_kW", p_base / p_scale)], dtype=np.float64),
        "bus": bus,
        "branch": branch,
        "transformer": transformer,
        "gen": gen,
        "load": load,
        "shunt": shunt,
        "zero_branch": zero_branch,
        "switch": build_switch_like(switches),
        "break": build_switch_like(breakers),
        "bus_name": bus_names,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
    }
    ppc.update(
        branch_name=_name_array(branches, "branch"),
        transformer_name=_name_array(transformers, "transformer"),
        gen_name=_name_array(generators, "gen"),
        load_name=_name_array(loads, "load"),
        shunt_name=_name_array(shunts, "shunt"),
        zero_branch_name=_name_array(zero_branches, "zero_branch"),
        switch_name=_name_array(switches, "switch"),
        break_name=_name_array(breakers, "break"),
    )
    return ppc


def build_ac_ppc_from_model(model):
    from ac_model import ACPowerNetwork

    network = ACPowerNetwork()
    network.model = model
    network._load_from_model()
    ppc = build_ac_ppc_from_network(network)
    network.ppc = ppc
    return network, ppc


def _list_ppc_names(ppc: Dict, key: str, prefix: str, count: int) -> List[str]:
    values = ppc.get(key)
    if values is None:
        return [f"{prefix}_{idx}" for idx in range(count)]
    return [str(value) for value in values]


def build_ac_network_from_ppc(ppc: Dict):
    from ac_model import (
        ACBreak,
        ACBranch,
        ACGenerator,
        ACLoad,
        ACNode,
        ACPowerNetwork,
        ACShuntCompensator,
        ACSwitch,
        ACTransformer,
        ACZeroBranch,
    )

    ctrl_name = {CTRL_PQ: "PQ", CTRL_P: "P", CTRL_PV: "PV", CTRL_SLACK: "V"}
    shunt_ctrl_name = {SHUNT_Q: "Q", SHUNT_V: "V", SHUNT_B: "B", SHUNT_Z: "Z"}
    bus_names = _list_ppc_names(ppc, "bus_name", "bus", ppc["bus"].shape[0])
    branch_names = _list_ppc_names(ppc, "branch_name", "branch", ppc["branch"].shape[0])
    transformer_names = _list_ppc_names(ppc, "transformer_name", "transformer", ppc["transformer"].shape[0])
    gen_names = _list_ppc_names(ppc, "gen_name", "gen", ppc["gen"].shape[0])
    load_names = _list_ppc_names(ppc, "load_name", "load", ppc["load"].shape[0])
    shunt_names = _list_ppc_names(ppc, "shunt_name", "shunt", ppc["shunt"].shape[0])
    zero_branch_names = _list_ppc_names(ppc, "zero_branch_name", "zero_branch", ppc["zero_branch"].shape[0])
    switch_names = _list_ppc_names(ppc, "switch_name", "switch", ppc["switch"].shape[0])
    break_names = _list_ppc_names(ppc, "break_name", "break", ppc.get("break", _empty(len(BREAK_COLS))).shape[0])

    network = ACPowerNetwork()
    base = ppc["base"]
    network.ppc = ppc
    network.p_base = float(base[0])
    network.u_scale = float(base[1])
    network.p_scale = float(base[2])
    network.i_scale = float(base[3])
    network.p_base_kW = float(base[4])

    network.nodes = [
        ACNode(
            int(row[BUS_COLS["idx"]]),
            float(row[BUS_COLS["vbase"]]),
            float(row[BUS_COLS["voltage"]]),
            float(row[BUS_COLS["angle"]]),
            int(row[BUS_COLS["run_stat"]]),
        )
        for row in ppc["bus"]
    ]
    for node in network.nodes:
        node.isl = None
        node.isl_obj = None
        node.v_gens = []
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.zero_branches = []
        node.transformers = []
        node.shunt_compensators = []

    network.branches = [
        ACBranch(
            int(row[BRANCH_COLS["idx"]]),
            int(row[BRANCH_COLS["i_node"]]),
            int(row[BRANCH_COLS["j_node"]]),
            float(row[BRANCH_COLS["r"]]),
            float(row[BRANCH_COLS["x"]]),
            float(row[BRANCH_COLS["b"]]),
            int(row[BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["branch"]
    ]
    network.transformers = [
        ACTransformer(
            int(row[TRANSFORMER_COLS["idx"]]),
            int(row[TRANSFORMER_COLS["i_node"]]),
            int(row[TRANSFORMER_COLS["j_node"]]),
            float(row[TRANSFORMER_COLS["r"]]),
            float(row[TRANSFORMER_COLS["x"]]),
            float(row[TRANSFORMER_COLS["tap"]]),
            float(row[TRANSFORMER_COLS["shift"]]),
            float(row[TRANSFORMER_COLS["b"]]),
            int(row[TRANSFORMER_COLS["run_stat"]]),
        )
        for row in ppc["transformer"]
    ]
    network.generators = [
        ACGenerator(
            int(row[GEN_COLS["idx"]]),
            int(row[GEN_COLS["node"]]),
            ctrl_name.get(int(row[GEN_COLS["control_type"]]), "PQ"),
            float(row[GEN_COLS["p_set"]]),
            float(row[GEN_COLS["q_set"]]),
            float(row[GEN_COLS["v_set"]]),
            float(row[GEN_COLS["alpha"]]),
            int(row[GEN_COLS["run_stat"]]),
        )
        for row in ppc["gen"]
    ]
    network.loads = [
        ACLoad(
            int(row[LOAD_COLS["idx"]]),
            int(row[LOAD_COLS["node"]]),
            float(row[LOAD_COLS["pbase"]]),
            float(row[LOAD_COLS["pv0"]]),
            float(row[LOAD_COLS["pv1"]]),
            float(row[LOAD_COLS["pv2"]]),
            float(row[LOAD_COLS["qbase"]]),
            float(row[LOAD_COLS["qv0"]]),
            float(row[LOAD_COLS["qv1"]]),
            float(row[LOAD_COLS["qv2"]]),
            int(row[LOAD_COLS["run_stat"]]),
        )
        for row in ppc["load"]
    ]
    network.shunt_compensators = [
        ACShuntCompensator(
            int(row[SHUNT_COLS["idx"]]),
            int(row[SHUNT_COLS["node"]]),
            shunt_ctrl_name.get(int(row[SHUNT_COLS["control_type"]]), "Q"),
            float(row[SHUNT_COLS["q_set"]]),
            float(row[SHUNT_COLS["g_set"]]),
            float(row[SHUNT_COLS["b_set"]]),
            float(row[SHUNT_COLS["v_set"]]),
            int(row[SHUNT_COLS["run_stat"]]),
        )
        for row in ppc["shunt"]
    ]
    network.zero_branches = [
        ACZeroBranch(
            int(row[ZERO_BRANCH_COLS["idx"]]),
            int(row[ZERO_BRANCH_COLS["i_node"]]),
            int(row[ZERO_BRANCH_COLS["j_node"]]),
            int(row[ZERO_BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["zero_branch"]
    ]
    network.switches = [
        ACSwitch(
            int(row[SWITCH_COLS["idx"]]),
            int(row[SWITCH_COLS["i_node"]]),
            int(row[SWITCH_COLS["j_node"]]),
            int(row[SWITCH_COLS["status"]]),
            int(row[SWITCH_COLS["run_stat"]]),
        )
        for row in ppc["switch"]
    ]
    network.breakers = [
        ACBreak(
            int(row[BREAK_COLS["idx"]]),
            int(row[BREAK_COLS["i_node"]]),
            int(row[BREAK_COLS["j_node"]]),
            int(row[BREAK_COLS["status"]]),
            int(row[BREAK_COLS["run_stat"]]),
        )
        for row in ppc.get("break", _empty(len(BREAK_COLS)))
    ]
    network.node_dict = {}
    network.switch_dict = {}
    network.break_dict = {}
    network.load_dict = {}
    network.generator_dict = {}
    network.zero_branch_dict = {}
    network.branch_dict = {}
    network.transformer_dict = {}
    network.shunt_compensator_dict = {}
    network.islands = []
    for obj, name in zip(network.nodes, bus_names):
        obj.name = name
        obj.is_alive = False
    for obj, row, name in zip(network.branches, ppc["branch"], branch_names):
        obj.name = name
        obj.i_p = float(row[BRANCH_COLS["i_p"]])
        obj.i_q = float(row[BRANCH_COLS["i_q"]])
        obj.i_c = float(row[BRANCH_COLS["i_c"]])
        obj.j_p = float(row[BRANCH_COLS["j_p"]])
        obj.j_q = float(row[BRANCH_COLS["j_q"]])
        obj.j_c = float(row[BRANCH_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(network.transformers, ppc["transformer"], transformer_names):
        obj.name = name
        obj.i_p = float(row[TRANSFORMER_COLS["i_p"]])
        obj.i_q = float(row[TRANSFORMER_COLS["i_q"]])
        obj.i_c = float(row[TRANSFORMER_COLS["i_c"]])
        obj.j_p = float(row[TRANSFORMER_COLS["j_p"]])
        obj.j_q = float(row[TRANSFORMER_COLS["j_q"]])
        obj.j_c = float(row[TRANSFORMER_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(network.generators, ppc["gen"], gen_names):
        obj.name = name
        obj.p = float(row[GEN_COLS["p"]])
        obj.q = float(row[GEN_COLS["q"]])
        obj.current = float(row[GEN_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.loads, ppc["load"], load_names):
        obj.name = name
        obj.p = float(row[LOAD_COLS["p"]])
        obj.q = float(row[LOAD_COLS["q"]])
        obj.current = float(row[LOAD_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.shunt_compensators, ppc["shunt"], shunt_names):
        obj.name = name
        obj.p = float(row[SHUNT_COLS["p"]])
        obj.q = float(row[SHUNT_COLS["q"]])
        obj.current = float(row[SHUNT_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.zero_branches, ppc["zero_branch"], zero_branch_names):
        obj.name = name
        obj.p = float(row[ZERO_BRANCH_COLS["p"]])
        obj.q = float(row[ZERO_BRANCH_COLS["q"]])
        obj.current = float(row[ZERO_BRANCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.switches, ppc["switch"], switch_names):
        obj.name = name
        obj.p = float(row[SWITCH_COLS["p"]])
        obj.q = float(row[SWITCH_COLS["q"]])
        obj.current = float(row[SWITCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.breakers, ppc.get("break", _empty(len(BREAK_COLS))), break_names):
        obj.name = name
        obj.p = float(row[BREAK_COLS["p"]])
        obj.q = float(row[BREAK_COLS["q"]])
        obj.current = float(row[BREAK_COLS["current"]])
        obj.is_alive = False
    return network


