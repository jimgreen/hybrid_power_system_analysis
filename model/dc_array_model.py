import threading
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

from efile_read import efile_factory_from_file

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


CTRL_P = 0
CTRL_V = 1
CTRL_I = 2
CTRL_CODE = {
    "P": CTRL_P,
    "V": CTRL_V,
    "I": CTRL_I,
}

BUS_COLS = {
    "idx": 0,
    "vbase": 1,
    "voltage": 2,
    "isl": 3,
    "run_stat": 4,
}
BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "run_stat": 4,
    "i_p": 5,
    "j_p": 6,
    "current": 7,
}
LOAD_COLS = {
    "idx": 0,
    "node": 1,
    "pbase": 2,
    "pv0": 3,
    "pv1": 4,
    "pv2": 5,
    "run_stat": 6,
    "p": 7,
    "current": 8,
}
GEN_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "p_set": 3,
    "v_set": 4,
    "i_set": 5,
    "run_stat": 6,
    "p": 7,
    "current": 8,
}
ZERO_BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "run_stat": 3,
    "p": 4,
    "current": 5,
}
SWITCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "status": 3,
    "run_stat": 4,
    "p": 5,
    "current": 6,
}
BREAK_COLS = SWITCH_COLS
DCDC_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r1": 3,
    "r2": 4,
    "control_type": 5,
    "p_set": 6,
    "i_set": 7,
    "v_set": 8,
    "run_stat": 9,
    "i_p": 10,
    "j_p": 11,
    "i_c": 12,
    "j_c": 13,
}

_DC_PPC_CACHE = {}
_DC_PPC_CACHE_LOCK = threading.Lock()


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = Path(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def clear_dc_ppc_cache(file_path=None) -> None:
    with _DC_PPC_CACHE_LOCK:
        if file_path is None:
            _DC_PPC_CACHE.clear()
        else:
            path = Path(file_path).resolve()
            _DC_PPC_CACHE.pop(path, None)

def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


class _ArrayDevice(SimpleNamespace):
    __hash__ = object.__hash__


def _device(
    idx,
    name,
    **values,
):
    obj = _ArrayDevice(idx=int(idx), name=str(name), **values)
    obj.is_alive = False
    return obj


def _node_maps(bus: np.ndarray):
    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
    return {int(node_id): pos for pos, node_id in enumerate(node_ids)}


def build_dc_ppc_from_e_file(file_path) -> Dict:
    """Build a DC ppc from an E file through the shared model factory."""
    file_key = _file_cache_key(file_path)
    with _DC_PPC_CACHE_LOCK:
        cached = _DC_PPC_CACHE.get(file_key[0])
        if cached is not None and cached[0] == file_key:
            return cached[1]

    model = efile_factory_from_file(file_key[0])
    _network, ppc = build_dc_ppc_from_model(model)
    ppc["source"] = str(file_key[0])
    with _DC_PPC_CACHE_LOCK:
        _DC_PPC_CACHE[file_key[0]] = (file_key, ppc)
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


def build_dc_ppc_from_network(network) -> Dict:
    """Build a DC ppc dictionary from an already loaded DCPowerNetwork."""
    nodes = list(getattr(network, "nodes", []))
    branches = list(getattr(network, "branches", []))
    loads = list(getattr(network, "loads", []))
    generators = list(getattr(network, "generators", []))
    zero_branches = list(getattr(network, "zero_branches", []))
    switches = list(getattr(network, "switches", []))
    breakers = list(getattr(network, "breakers", []))
    dcdcs = list(getattr(network, "dcdc_converters", []))

    p_base = float(getattr(network, "p_base", 1.0))
    u_scale = float(getattr(network, "u_scale", 1.0))
    p_scale = float(getattr(network, "p_scale", 1.0))
    i_scale = float(getattr(network, "i_scale", 1.0))
    p_base_kw = float(getattr(network, "p_base_kW", p_base / p_scale))

    bus = np.zeros((len(nodes), len(BUS_COLS)), dtype=np.float64)
    for row, node in enumerate(nodes):
        bus[row, BUS_COLS["idx"]] = _int_value(node, "idx")
        bus[row, BUS_COLS["vbase"]] = _float_value(node, "vbase")
        bus[row, BUS_COLS["voltage"]] = _float_value(node, "voltage", 1.0)
        bus[row, BUS_COLS["isl"]] = _float_value(node, "isl", 0.0)
        bus[row, BUS_COLS["run_stat"]] = _float_value(node, "run_stat", 1.0)

    branch = np.zeros((len(branches), len(BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(branches):
        branch[row, BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        branch[row, BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        branch[row, BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        branch[row, BRANCH_COLS["r"]] = _float_value(dev, "r")
        branch[row, BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        branch[row, BRANCH_COLS["i_p"]] = _float_value(dev, "i_p")
        branch[row, BRANCH_COLS["j_p"]] = _float_value(dev, "j_p")
        branch[row, BRANCH_COLS["current"]] = _float_value(dev, "current")

    load = np.zeros((len(loads), len(LOAD_COLS)), dtype=np.float64)
    for row, dev in enumerate(loads):
        load[row, LOAD_COLS["idx"]] = _int_value(dev, "idx")
        load[row, LOAD_COLS["node"]] = _int_value(dev, "node")
        load[row, LOAD_COLS["pbase"]] = _float_value(dev, "pbase", 1.0)
        load[row, LOAD_COLS["pv0"]] = _float_value(dev, "pv0")
        load[row, LOAD_COLS["pv1"]] = _float_value(dev, "pv1")
        load[row, LOAD_COLS["pv2"]] = _float_value(dev, "pv2")
        load[row, LOAD_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        load[row, LOAD_COLS["p"]] = _float_value(dev, "p")
        load[row, LOAD_COLS["current"]] = _float_value(dev, "current")

    gen = np.zeros((len(generators), len(GEN_COLS)), dtype=np.float64)
    for row, dev in enumerate(generators):
        gen[row, GEN_COLS["idx"]] = _int_value(dev, "idx")
        gen[row, GEN_COLS["node"]] = _int_value(dev, "node")
        gen[row, GEN_COLS["control_type"]] = _code_value(_value(dev, "control_type", "P"), CTRL_CODE, "P")
        gen[row, GEN_COLS["p_set"]] = _float_value(dev, "p_set")
        gen[row, GEN_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        gen[row, GEN_COLS["i_set"]] = _float_value(dev, "i_set")
        gen[row, GEN_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        gen[row, GEN_COLS["p"]] = _float_value(dev, "p")
        gen[row, GEN_COLS["current"]] = _float_value(dev, "current")

    zero_branch = np.zeros((len(zero_branches), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(zero_branches):
        zero_branch[row, ZERO_BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        zero_branch[row, ZERO_BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        zero_branch[row, ZERO_BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        zero_branch[row, ZERO_BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        zero_branch[row, ZERO_BRANCH_COLS["p"]] = _float_value(dev, "p")
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
            out[row, SWITCH_COLS["current"]] = _float_value(dev, "current")
        return out

    dcdc = np.zeros((len(dcdcs), len(DCDC_COLS)), dtype=np.float64)
    for row, dev in enumerate(dcdcs):
        dcdc[row, DCDC_COLS["idx"]] = _int_value(dev, "idx")
        dcdc[row, DCDC_COLS["i_node"]] = _int_value(dev, "i_node")
        dcdc[row, DCDC_COLS["j_node"]] = _int_value(dev, "j_node")
        dcdc[row, DCDC_COLS["r1"]] = _float_value(dev, "r1")
        dcdc[row, DCDC_COLS["r2"]] = _float_value(dev, "r2")
        dcdc[row, DCDC_COLS["control_type"]] = _code_value(_value(dev, "control_type", "P"), CTRL_CODE, "P")
        dcdc[row, DCDC_COLS["p_set"]] = _float_value(dev, "p_set")
        dcdc[row, DCDC_COLS["i_set"]] = _float_value(dev, "i_set")
        dcdc[row, DCDC_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        dcdc[row, DCDC_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        dcdc[row, DCDC_COLS["i_p"]] = _float_value(dev, "i_p")
        dcdc[row, DCDC_COLS["j_p"]] = _float_value(dev, "j_p")
        dcdc[row, DCDC_COLS["i_c"]] = _float_value(dev, "i_c")
        dcdc[row, DCDC_COLS["j_c"]] = _float_value(dev, "j_c")

    ppc = {
        "format": "dc_ppc_v1",
        "base": {
            "p_base": p_base,
            "u_scale": u_scale,
            "p_scale": p_scale,
            "i_scale": i_scale,
            "p_base_kW": p_base_kw,
        },
        "bus": bus,
        "branch": branch,
        "load": load,
        "gen": gen,
        "zero_branch": zero_branch,
        "switch": build_switch_like(switches),
        "break": build_switch_like(breakers),
        "dcdc": dcdc,
        "node_pos": _node_maps(bus) if len(bus) else {},
    }
    ppc.update(
        bus_name=_name_array(nodes, "bus"),
        branch_name=_name_array(branches, "branch"),
        load_name=_name_array(loads, "load"),
        gen_name=_name_array(generators, "gen"),
        zero_branch_name=_name_array(zero_branches, "zero_branch"),
        switch_name=_name_array(switches, "switch"),
        break_name=_name_array(breakers, "break"),
        dcdc_name=_name_array(dcdcs, "dcdc"),
    )
    return ppc


def build_dc_ppc_from_model(model):
    from dc_model import DCPowerNetwork as ObjectDCPowerNetwork

    network = ObjectDCPowerNetwork()
    network.model = model
    network._load_from_model()
    ppc = build_dc_ppc_from_network(network)
    network.ppc = ppc
    return network, ppc




def build_dc_network_from_ppc(ppc: Dict):
    network = DCPowerNetwork()
    network.ppc = ppc
    base = ppc["base"]
    network.p_base = float(base["p_base"])
    network.p_base_kW = float(base["p_base_kW"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network._load_objects_from_ppc(ppc)
    return network


class DCIsl:
    def __init__(self, idx, is_alive):
        self.idx = idx
        self.is_alive = is_alive
        self.buses = []
        self.gens = []
        self.loads = []
        self.branches = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.slack_nodes = []
        self.v_gens = []
        self.v_dcdcs = []


class DCPowerNetwork:
    """Object-compatible DC network facade backed by dc_ppc_v1 arrays."""

    def __init__(self):
        self.ppc = None
        self.nodes = []
        self.branches = []
        self.loads = []
        self.generators = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.buses = []
        self.islands = []
        self.node_dict = {}
        self.bus_dict = {}
        self.node_to_bus = {}
        self.switch_dict = {}
        self.break_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branche_dict = {}
        self.branche_dict = {}
        self.dcdc_converter_dict = {}

    def add_node(self, idx, vbase, voltage=1.0, run_stat=1):
        node = _device(idx, f"nd_{idx}", vbase=float(vbase), voltage=float(voltage), run_stat=int(run_stat))
        node.isl = None
        node.isl_obj = None
        node.v_set = 1.0
        node.v_gens = []
        node.v_dcdcs = []
        node.is_slack = False
        node.bus = None
        node.bus_obj = None
        self.nodes.append(node)
        return node

    def add_branch(self, idx, i_node, j_node, r, run_stat=1):
        br = _device(
            idx,
            f"br_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            r=float(r),
            run_stat=int(run_stat),
            current=None,
            i_p=None,
            j_p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.branches.append(br)
        return br

    def add_load(self, idx, node, pbase, pv0, pv1, pv2, run_stat=1):
        ld = _device(
            idx,
            f"load_{idx}",
            node=int(node),
            pbase=float(pbase),
            pv0=float(pv0),
            pv1=float(pv1),
            pv2=float(pv2),
            run_stat=int(run_stat),
            p=None,
            current=None,
            node_obj=None,
        )
        self.loads.append(ld)
        return ld

    def add_generator(self, idx, node, control_type, p_set, v_set, i_set, run_stat=1):
        gen = _device(
            idx,
            f"gen_{idx}",
            node=int(node),
            control_type=str(control_type),
            p_set=float(p_set),
            v_set=float(v_set),
            i_set=float(i_set),
            run_stat=int(run_stat),
            p=None,
            current=None,
            node_obj=None,
        )
        self.generators.append(gen)
        return gen

    def add_zero_branch(self, idx, i_node, j_node, run_stat=1):
        zbr = _device(
            idx,
            f"zbr_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.zero_branches.append(zbr)
        return zbr

    def add_switch(self, idx, i_node, j_node, status, run_stat=1):
        sw = _device(
            idx,
            f"sw_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            status=int(status),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.switches.append(sw)
        return sw

    def add_break(self, idx, i_node, j_node, status, run_stat=1):
        brk = _device(
            idx,
            f"brk_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            status=int(status),
            run_stat=int(run_stat),
            current=None,
            p=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.breakers.append(brk)
        return brk

    def add_dcdc_converter(self, idx, i_node, j_node, r1, r2, control_type, p_set, i_set, v_set, run_stat=1):
        conv = _device(
            idx,
            f"dcdc_{idx}",
            i_node=int(i_node),
            j_node=int(j_node),
            r1=float(r1),
            r2=float(r2),
            control_type=str(control_type),
            p_set=float(p_set),
            i_set=float(i_set),
            v_set=float(v_set),
            run_stat=int(run_stat),
            i_p=None,
            j_p=None,
            i_c=None,
            j_c=None,
            i_node_obj=None,
            j_node_obj=None,
        )
        self.dcdc_converters.append(conv)
        return conv

    def read_from_file(self, file_name):
        self.ppc = build_dc_ppc_from_e_file(file_name)
        base = self.ppc["base"]
        self.p_base = float(base["p_base"])
        self.p_base_kW = float(base["p_base_kW"])
        self.u_scale = float(base["u_scale"])
        self.p_scale = float(base["p_scale"])
        self.i_scale = float(base["i_scale"])
        self._load_objects_from_ppc(self.ppc)

    def _load_objects_from_ppc(self, ppc):
        ctrl_label = {CTRL_P: "P", CTRL_V: "V", CTRL_I: "I"}
        self.nodes = []
        for row, name in zip(ppc["bus"], ppc.get("bus_name", [])):
            node = _device(
                row[BUS_COLS["idx"]],
                name,
                vbase=float(row[BUS_COLS["vbase"]]),
                voltage=float(row[BUS_COLS["voltage"]]),
                run_stat=int(row[BUS_COLS["run_stat"]]),
            )
            node.isl = None
            node.isl_obj = None
            node.v_set = 1.0
            node.v_gens = []
            node.v_dcdcs = []
            node.is_slack = False
            node.bus = None
            node.bus_obj = None
            self.nodes.append(node)

        self.branches = []
        for row, name in zip(ppc["branch"], ppc.get("branch_name", [])):
            self.branches.append(
                _device(
                    row[BRANCH_COLS["idx"]],
                    name,
                    i_node=int(row[BRANCH_COLS["i_node"]]),
                    j_node=int(row[BRANCH_COLS["j_node"]]),
                    r=float(row[BRANCH_COLS["r"]]),
                    run_stat=int(row[BRANCH_COLS["run_stat"]]),
                    i_p=float(row[BRANCH_COLS["i_p"]]),
                    j_p=float(row[BRANCH_COLS["j_p"]]),
                    current=float(row[BRANCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.loads = []
        for row, name in zip(ppc["load"], ppc.get("load_name", [])):
            self.loads.append(
                _device(
                    row[LOAD_COLS["idx"]],
                    name,
                    node=int(row[LOAD_COLS["node"]]),
                    pbase=float(row[LOAD_COLS["pbase"]]),
                    pv0=float(row[LOAD_COLS["pv0"]]),
                    pv1=float(row[LOAD_COLS["pv1"]]),
                    pv2=float(row[LOAD_COLS["pv2"]]),
                    run_stat=int(row[LOAD_COLS["run_stat"]]),
                    p=float(row[LOAD_COLS["p"]]),
                    current=float(row[LOAD_COLS["current"]]),
                    node_obj=None,
                )
            )

        self.generators = []
        for row, name in zip(ppc["gen"], ppc.get("gen_name", [])):
            self.generators.append(
                _device(
                    row[GEN_COLS["idx"]],
                    name,
                    node=int(row[GEN_COLS["node"]]),
                    control_type=ctrl_label[int(row[GEN_COLS["control_type"]])],
                    p_set=float(row[GEN_COLS["p_set"]]),
                    v_set=float(row[GEN_COLS["v_set"]]),
                    i_set=float(row[GEN_COLS["i_set"]]),
                    run_stat=int(row[GEN_COLS["run_stat"]]),
                    p=float(row[GEN_COLS["p"]]),
                    current=float(row[GEN_COLS["current"]]),
                    node_obj=None,
                )
            )

        self.zero_branches = []
        for row, name in zip(ppc["zero_branch"], ppc.get("zero_branch_name", [])):
            self.zero_branches.append(
                _device(
                    row[ZERO_BRANCH_COLS["idx"]],
                    name,
                    i_node=int(row[ZERO_BRANCH_COLS["i_node"]]),
                    j_node=int(row[ZERO_BRANCH_COLS["j_node"]]),
                    run_stat=int(row[ZERO_BRANCH_COLS["run_stat"]]),
                    p=float(row[ZERO_BRANCH_COLS["p"]]),
                    current=float(row[ZERO_BRANCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.switches = []
        for row, name in zip(ppc["switch"], ppc.get("switch_name", [])):
            self.switches.append(
                _device(
                    row[SWITCH_COLS["idx"]],
                    name,
                    i_node=int(row[SWITCH_COLS["i_node"]]),
                    j_node=int(row[SWITCH_COLS["j_node"]]),
                    status=int(row[SWITCH_COLS["status"]]),
                    run_stat=int(row[SWITCH_COLS["run_stat"]]),
                    p=float(row[SWITCH_COLS["p"]]),
                    current=float(row[SWITCH_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.breakers = []
        for row, name in zip(ppc.get("break", _empty(len(BREAK_COLS))), ppc.get("break_name", [])):
            self.breakers.append(
                _device(
                    row[BREAK_COLS["idx"]],
                    name,
                    i_node=int(row[BREAK_COLS["i_node"]]),
                    j_node=int(row[BREAK_COLS["j_node"]]),
                    status=int(row[BREAK_COLS["status"]]),
                    run_stat=int(row[BREAK_COLS["run_stat"]]),
                    p=float(row[BREAK_COLS["p"]]),
                    current=float(row[BREAK_COLS["current"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

        self.dcdc_converters = []
        for row, name in zip(ppc["dcdc"], ppc.get("dcdc_name", [])):
            self.dcdc_converters.append(
                _device(
                    row[DCDC_COLS["idx"]],
                    name,
                    i_node=int(row[DCDC_COLS["i_node"]]),
                    j_node=int(row[DCDC_COLS["j_node"]]),
                    r1=float(row[DCDC_COLS["r1"]]),
                    r2=float(row[DCDC_COLS["r2"]]),
                    control_type=ctrl_label[int(row[DCDC_COLS["control_type"]])],
                    p_set=float(row[DCDC_COLS["p_set"]]),
                    i_set=float(row[DCDC_COLS["i_set"]]),
                    v_set=float(row[DCDC_COLS["v_set"]]),
                    run_stat=int(row[DCDC_COLS["run_stat"]]),
                    i_p=float(row[DCDC_COLS["i_p"]]),
                    j_p=float(row[DCDC_COLS["j_p"]]),
                    i_c=float(row[DCDC_COLS["i_c"]]),
                    j_c=float(row[DCDC_COLS["j_c"]]),
                    i_node_obj=None,
                    j_node_obj=None,
                )
            )

    def format_assoc(self):
        self.node_dict = {node.idx: node for node in self.nodes}
        self.switch_dict = {sw.idx: sw for sw in self.switches}
        self.break_dict = {brk.idx: brk for brk in self.breakers}
        self.load_dict = {ld.idx: ld for ld in self.loads}
        self.generator_dict = {gen.idx: gen for gen in self.generators}
        self.zero_branche_dict = {zbr.idx: zbr for zbr in self.zero_branches}
        self.branche_dict = {br.idx: br for br in self.branches}
        self.dcdc_converter_dict = {conv.idx: conv for conv in self.dcdc_converters}

        for node in self.nodes:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.dcdc_converters = []
            node.zero_branches = []
            node.is_alive = False
            node.bus = None
            node.bus_obj = None

        for gen in self.generators:
            gen.node_obj = self.node_dict.get(gen.node, None)
            if gen.node_obj:
                gen.node_obj.generators.append(gen)
        for ld in self.loads:
            ld.node_obj = self.node_dict.get(ld.node, None)
            if ld.node_obj:
                ld.node_obj.loads.append(ld)
        for br in self.branches:
            br.i_node_obj = self.node_dict.get(br.i_node, None)
            br.j_node_obj = self.node_dict.get(br.j_node, None)
            if br.i_node_obj:
                br.i_node_obj.branches.append(br)
            if br.j_node_obj:
                br.j_node_obj.branches.append(br)
        for sw in self.switches:
            sw.i_node_obj = self.node_dict.get(sw.i_node, None)
            sw.j_node_obj = self.node_dict.get(sw.j_node, None)
            if sw.i_node_obj:
                sw.i_node_obj.switches.append(sw)
            if sw.j_node_obj:
                sw.j_node_obj.switches.append(sw)
        for brk in self.breakers:
            brk.i_node_obj = self.node_dict.get(brk.i_node, None)
            brk.j_node_obj = self.node_dict.get(brk.j_node, None)
            if brk.i_node_obj:
                brk.i_node_obj.breakers.append(brk)
            if brk.j_node_obj:
                brk.j_node_obj.breakers.append(brk)
        for conv in self.dcdc_converters:
            conv.i_node_obj = self.node_dict.get(conv.i_node, None)
            conv.j_node_obj = self.node_dict.get(conv.j_node, None)
            if conv.i_node_obj:
                conv.i_node_obj.dcdc_converters.append(conv)
            if conv.j_node_obj:
                conv.j_node_obj.dcdc_converters.append(conv)
        for zbr in self.zero_branches:
            zbr.i_node_obj = self.node_dict.get(zbr.i_node, None)
            zbr.j_node_obj = self.node_dict.get(zbr.j_node, None)
            if zbr.i_node_obj:
                zbr.i_node_obj.zero_branches.append(zbr)
            if zbr.j_node_obj:
                zbr.j_node_obj.zero_branches.append(zbr)

    def topo(self):
        if len(self.node_dict) == 0:
            self.format_assoc()
        for node in self.nodes:
            node.isl = 0
            node.isl_obj = None
            node.bus = None
            node.bus_obj = None
            node.is_alive = False

        running_nodes = [node for node in self.nodes if node.run_stat == 1]
        running_node_ids = {node.idx for node in running_nodes}
        parent = {node.idx: node.idx for node in running_nodes}

        def find(parents, node_idx):
            root = node_idx
            while parents[root] != root:
                root = parents[root]
            while parents[node_idx] != node_idx:
                next_idx = parents[node_idx]
                parents[node_idx] = root
                node_idx = next_idx
            return root

        def union(parents, left, right):
            root_l = find(parents, left)
            root_r = find(parents, right)
            if root_l != root_r:
                parents[root_r] = root_l

        def live_terminal_pair(dev, require_closed=False):
            if (
                dev.run_stat == 1
                and (not require_closed or getattr(dev, "status", 1) == 1)
                and dev.i_node in running_node_ids
                and dev.j_node in running_node_ids
                and dev.i_node != dev.j_node
            ):
                return dev.i_node, dev.j_node
            return None

        if self.switches:
            for dev in self.switches:
                pair = live_terminal_pair(dev, require_closed=True)
                if pair is not None:
                    union(parent, pair[0], pair[1])

        root_to_nodes = {}
        for node in running_nodes:
            root_to_nodes.setdefault(find(parent, node.idx), []).append(node)
        self.buses = []
        self.bus_dict = {}
        self.node_to_bus = {}
        for nodes in sorted(root_to_nodes.values(), key=lambda group: min(node.idx for node in group)):
            nodes.sort(key=lambda item: item.idx)
            bus = _device(
                nodes[0].idx,
                getattr(nodes[0], "name", f"bus_{nodes[0].idx}"),
                nodes=list(nodes),
                vbase=float(getattr(nodes[0], "vbase", 0.0)),
                voltage=float(getattr(nodes[0], "voltage", 1.0)),
                run_stat=1,
                isl=0,
                isl_obj=None,
                v_set=1.0,
                v_gens=[],
                v_dcdcs=[],
                is_slack=False,
                generators=[],
                loads=[],
                branches=[],
                switches=[],
                breakers=[],
                dcdc_converters=[],
                zero_branches=[],
            )
            self.buses.append(bus)
            self.bus_dict[bus.idx] = bus
            for node in nodes:
                node.bus = bus.idx
                node.bus_obj = bus
                self.node_to_bus[node.idx] = bus

        bus_parent = {bus.idx: bus.idx for bus in self.buses}

        def add_bus_edge(dev, require_closed=False):
            pair = live_terminal_pair(dev, require_closed=require_closed)
            if pair is None:
                return
            i_bus = self.node_to_bus.get(pair[0])
            j_bus = self.node_to_bus.get(pair[1])
            if i_bus is not None and j_bus is not None and i_bus.idx != j_bus.idx:
                union(bus_parent, i_bus.idx, j_bus.idx)

        for dev in self.branches:
            add_bus_edge(dev)
        for dev in self.zero_branches:
            add_bus_edge(dev)
        for dev in self.breakers:
            add_bus_edge(dev, require_closed=True)

        self.islands = []
        island_idx = 0
        root_to_island = {}
        for bus in self.buses:
            root = find(bus_parent, bus.idx)
            island = root_to_island.get(root)
            if island is None:
                island_idx += 1
                island = DCIsl(island_idx, True)
                root_to_island[root] = island
                self.islands.append(island)
            bus.isl = island.idx
            bus.isl_obj = island
            for node in bus.nodes:
                node.isl = island.idx
                node.isl_obj = island

        self.det_isl_alive_stat()

    def det_isl_alive_stat(self):
        for isl in self.islands:
            isl.is_alive = False
            isl.slack_nodes = []
            isl.v_gens = []
            isl.v_dcdcs = []
            isl.buses = []
            isl.gens = []
            isl.loads = []
            isl.branches = []
            isl.dcdc_converters = []
            isl.zero_branches = []
            isl.switches = []
            isl.breakers = []

        for node in self.nodes:
            node.v_gens = []
            node.v_dcdcs = []
            node.v_set = 0.0
            node.is_slack = False
        for bus in self.buses:
            bus.v_gens = []
            bus.v_dcdcs = []
            bus.v_set = 0.0
            bus.is_slack = False
            bus.generators = []
            bus.loads = []
            bus.branches = []
            bus.switches = []
            bus.breakers = []
            bus.dcdc_converters = []
            bus.zero_branches = []

        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            node = gen.node_obj
            if node is None or node.isl_obj is None:
                continue
            node.isl_obj.gens.append(gen)
            if gen.control_type == "V":
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)

        for dcdc in self.dcdc_converters:
            if dcdc.run_stat == 0:
                continue
            if dcdc.i_node_obj is None or dcdc.j_node_obj is None:
                continue
            if dcdc.i_node_obj.isl_obj is None or dcdc.j_node_obj.isl_obj is None:
                continue
            node = dcdc.i_node_obj
            dcdc.i_node_obj.isl_obj.dcdc_converters.append(dcdc)
            dcdc.j_node_obj.isl_obj.dcdc_converters.append(dcdc)
            if dcdc.control_type == "V":
                node.v_dcdcs.append(dcdc)
                if node.bus_obj is not None:
                    node.bus_obj.v_dcdcs.append(dcdc)
                node.isl_obj.v_dcdcs.append(dcdc)

        for load in self.loads:
            if load.run_stat == 0:
                continue
            if load.node_obj is None or load.node_obj.isl_obj is None:
                continue
            load.node_obj.isl_obj.loads.append(load)
        for switch in self.switches:
            if switch.i_node_obj is None or switch.j_node_obj is None:
                continue
            if switch.run_stat == 0 or switch.status == 0:
                continue
            if switch.i_node_obj.isl_obj and switch.j_node_obj.isl_obj and switch.i_node_obj.isl_obj == switch.j_node_obj.isl_obj:
                switch.i_node_obj.isl_obj.switches.append(switch)
        for br in self.branches:
            if br.run_stat == 0:
                continue
            if br.i_node_obj is None or br.j_node_obj is None:
                continue
            if br.i_node_obj.isl_obj and br.j_node_obj.isl_obj and br.i_node_obj.isl_obj == br.j_node_obj.isl_obj:
                br.i_node_obj.isl_obj.branches.append(br)
        for zbr in self.zero_branches:
            if zbr.run_stat == 0:
                continue
            if zbr.i_node_obj is None or zbr.j_node_obj is None:
                continue
            if zbr.i_node_obj.isl_obj and zbr.j_node_obj.isl_obj and zbr.i_node_obj.isl_obj == zbr.j_node_obj.isl_obj:
                zbr.i_node_obj.isl_obj.zero_branches.append(zbr)
        for brk in self.breakers:
            if brk.run_stat == 0 or brk.status == 0:
                continue
            if brk.i_node_obj is None or brk.j_node_obj is None:
                continue
            if brk.i_node_obj.isl_obj and brk.j_node_obj.isl_obj and brk.i_node_obj.isl_obj == brk.j_node_obj.isl_obj:
                brk.i_node_obj.isl_obj.breakers.append(brk)

        for bus in self.buses:
            if bus.isl_obj is None:
                continue
            bus.isl_obj.buses.append(bus)
            if len(bus.v_gens) + len(bus.v_dcdcs) > 0:
                bus.isl_obj.slack_nodes.append(bus)

        for isl in self.islands:
            if len(isl.slack_nodes) + len(isl.v_dcdcs) >= 1:
                isl.is_alive = True

        for bus in self.buses:
            bus.is_alive = bus.run_stat == 1 and bus.isl_obj is not None and bus.isl_obj.is_alive

        for node in self.nodes:
            node.is_alive = node.run_stat == 1 and node.isl_obj is not None and node.isl_obj.is_alive
        self.alive_buses = [bus for bus in self.buses if bus.is_alive]
        for load in self.loads:
            node = load.node_obj
            load.is_alive = node is not None and node.isl_obj is not None and load.run_stat == 1 and node.isl_obj.is_alive
        for gen in self.generators:
            node = gen.node_obj
            gen.is_alive = node is not None and node.isl_obj is not None and gen.run_stat == 1 and node.isl_obj.is_alive
        for br in self.branches:
            br.is_alive = (
                br.i_node_obj is not None
                and br.j_node_obj is not None
                and br.run_stat == 1
                and br.i_node_obj.is_alive
                and br.j_node_obj.is_alive
            )
        for zbr in self.zero_branches:
            zbr.is_alive = (
                zbr.i_node_obj is not None
                and zbr.j_node_obj is not None
                and zbr.run_stat == 1
                and zbr.i_node_obj.is_alive
                and zbr.j_node_obj.is_alive
            )
        for brk in self.breakers:
            brk.is_alive = (
                brk.i_node_obj is not None
                and brk.j_node_obj is not None
                and brk.run_stat == 1
                and brk.status == 1
                and brk.i_node_obj.is_alive
                and brk.j_node_obj.is_alive
            )
        for sw in self.switches:
            sw.is_alive = (
                sw.i_node_obj is not None
                and sw.j_node_obj is not None
                and sw.status == 1
                and sw.run_stat == 1
                and sw.i_node_obj.is_alive
                and sw.j_node_obj.is_alive
            )
        for conv in self.dcdc_converters:
            conv.is_alive = (
                conv.i_node_obj is not None
                and conv.j_node_obj is not None
                and conv.run_stat == 1
                and conv.i_node_obj.is_alive
                and conv.j_node_obj.is_alive
            )

    def print_isl_info(self):
        for isl in self.islands:
            print(f"isl {isl.idx} is_alive = {isl.is_alive}")
            print(f"    buses = {len(isl.buses)}:")
            for node in isl.buses:
                print(f"        {node.idx} {node.name} vbase: {node.vbase}")
            print(f"    gens = {len(isl.gens)}:")
            for gen in isl.gens:
                print(f"        {gen.idx} {gen.name} node = {gen.node} control_type = {gen.control_type}")
            print(f"    loads = {len(isl.loads)}:")
            for load in isl.loads:
                print(f"        {load.idx} {load.name} node = {load.node}")
            print(f"    branches = {len(isl.branches)}:")
            for br in isl.branches:
                print(f"        {br.idx} {br.name} i_node = {br.i_node} j_node = {br.j_node} r = {br.r}")
            print(f"    switches = {len(isl.switches)}:")
            for sw in isl.switches:
                print(f"        {sw.idx} {sw.name} i_node = {sw.i_node} j_node = {sw.j_node} status = {sw.status}")
            print(f"    zero_branches = {len(isl.zero_branches)}:")
            for zbr in isl.zero_branches:
                print(f"        {zbr.idx} {zbr.name} i_node = {zbr.i_node} j_node = {zbr.j_node}")
            print(f"    breakers = {len(getattr(isl, 'breakers', []))}:")
            for brk in getattr(isl, "breakers", []):
                print(f"        {brk.idx} {brk.name} i_node = {brk.i_node} j_node = {brk.j_node} status = {brk.status}")
            print(f"    dcdc_converters = {len(isl.dcdc_converters)}:")
            for dcc in isl.dcdc_converters:
                print(f"        {dcc.idx} {dcc.name} i_node = {dcc.i_node} j_node = {dcc.j_node} r1 = {dcc.r1} r2 = {dcc.r2} control_type = {dcc.control_type}")

    def check_topo(self):
        errors = []
        warns = []
        if len(self.islands) == 0:
            self.topo()

        node_ref_count = {node.idx: 0 for node in self.nodes}

        def check_node(node_idx, dev_type, dev):
            if node_idx not in self.node_dict:
                errors.append(f"设备 {dev_type}[{dev.idx}] {dev.name} 引用的节点 {node_idx} 不存在")
            elif self.node_dict[node_idx].run_stat == 1:
                node_ref_count[node_idx] += 1

        for br in self.branches:
            if br.run_stat:
                check_node(br.i_node, "Branch", br)
                check_node(br.j_node, "Branch", br)
        for zbr in self.zero_branches:
            if zbr.run_stat:
                check_node(zbr.i_node, "ZeroBranch", zbr)
                check_node(zbr.j_node, "ZeroBranch", zbr)
        for sw in self.switches:
            if sw.run_stat:
                check_node(sw.i_node, "Switch", sw)
                check_node(sw.j_node, "Switch", sw)
        for brk in self.breakers:
            if brk.run_stat and brk.status:
                check_node(brk.i_node, "Break", brk)
                check_node(brk.j_node, "Break", brk)
        for ld in self.loads:
            if ld.run_stat:
                check_node(ld.node, "Load", ld)
        for gen in self.generators:
            if gen.run_stat:
                check_node(gen.node, "Generator", gen)
        for dcdc in self.dcdc_converters:
            if dcdc.run_stat:
                check_node(dcdc.i_node, "DCDCConverter", dcdc)
                check_node(dcdc.j_node, "DCDCConverter", dcdc)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {node.name} 未关联任何设备")
            if node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {node.name} 单端悬空，请检查！")

        for isl in self.islands:
            vbase_set = {int(bus.vbase * 1000) for bus in isl.buses}
            if len(vbase_set) > 1:
                str_info = f"岛屿 {isl.idx} 内节点电压基值不一致:"
                for vbase in vbase_set:
                    str_info += f" {vbase / 1000.0 :.2f}"
                errors.append(str_info)
            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V节点:"
                for node in isl.slack_nodes:
                    str_info += f" {node.name}"
                warns.append(str_info)
            if len(isl.v_dcdcs) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V变流器:"
                for dcdc in isl.v_dcdcs:
                    str_info += f" {dcdc.name}"
                warns.append(str_info)
            if len(isl.slack_nodes) + len(isl.v_dcdcs) == 0:
                errors.append(f"岛屿 {isl.idx} , 内无电压控制源（定V节点或定V变流器）")
            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} , 内有多个电压控制源（定V节点或定V变流器）:"
                for node in isl.slack_nodes:
                    str_info += f" node-{node.name}"
                errors.append(str_info)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if len(node.v_gens) + len(node.v_dcdcs) <= 1:
                continue
            if len(node.v_gens) + len(node.v_dcdcs) >= 2:
                errors.append(f"松弛节点 {node.idx} 上的定V发电机与定V变流器数量之和超过1，请检查拓扑！")
            node.v_set = 0.0
            if len(node.v_gens) >= 1:
                node.v_set = node.v_gens[0].v_set
            if len(node.v_dcdcs) > 1:
                node.v_set = node.v_dcdcs[0].v_set
            node.is_slack = True

        return warns, errors
