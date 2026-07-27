class ACIsl:
    def __init__(self, idx, is_alive):
        self.idx = idx
        self.is_alive = is_alive
        self.buses = []
        self.gens = []
        self.loads = []
        self.branches = []
        self.three_winding_transformers = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.acac_converters = []
        self.slack_nodes = []
        self.v_gens = []

class ACNode:
    def __init__(self, idx, vbase, voltage=None, angle=None, run_stat=1):
        self.idx = idx
        self.vbase = vbase
        self.voltage = voltage
        self.angle = angle
        self.run_stat = run_stat
        self.isl = None
        self.isl_obj = None
        self.bus = None
        self.bus_obj = None


class ACBus:
    def __init__(self, idx, nodes=None):
        self.idx = idx
        if nodes is None:
            self.nodes = []
        elif isinstance(nodes, list):
            self.nodes = nodes
        else:
            self.nodes = list(nodes)
        ref = self.nodes[0] if self.nodes else None
        self.name = getattr(ref, "name", f"bus_{idx}")
        self.vbase = getattr(ref, "vbase", 0.0)
        self.voltage = getattr(ref, "voltage", 1.0)
        self.angle = getattr(ref, "angle", 0.0)
        self.run_stat = 1
        self.isl = None
        self.isl_obj = None
        self.is_alive = False
        self.generators = []
        self.loads = []
        self.branches = []
        self.switches = []
        self.breakers = []
        self.zero_branches = []
        self.transformers = []
        self.three_winding_transformers = []
        self.shunt_compensators = []
        self.acac_converters = []
        self.v_gens = []

class ACBranch:
    def __init__(self, idx, i_node, j_node, r, x, b, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r = r
        self.x = x
        self.b = b
        self.run_stat = run_stat
        self.i_p = None
        self.i_q = None
        self.i_c = None
        self.j_p = None
        self.j_q = None
        self.j_c = None
        self.i_node_obj = None
        self.j_node_obj = None

class ACLoad:
    def __init__(self, idx, node, pbase, pv0, pv1, pv2, qbase, qv0, qv1, qv2, run_stat=1):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.pbase = pbase
        self.pv0 = pv0
        self.pv1 = pv1
        self.pv2 = pv2
        self.qbase = qbase
        self.qv0 = qv0
        self.qv1 = qv1
        self.qv2 = qv2

        self.p = None
        self.q = None
        self.current = None
        self.node_obj = None


class ACShuntCompensator:
    def __init__(self, idx, node, control_type='Q', q_set=0.0, g_set=0.0, b_set=0.0, v_set=1.0, run_stat=1):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.control_type = control_type
        self.q_set = q_set
        self.g_set = g_set
        self.b_set = b_set
        self.v_set = v_set
        self.p = None
        self.q = None
        self.current = None
        self.is_alive = False

class ACGenerator:
    def __init__(self, idx, node, control_type, p_set, q_set, v_set, alpha=None, run_stat=1, p_max=None):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.control_type = control_type
        self.p_set = p_set
        self.q_set = q_set
        self.v_set = v_set
        self.alpha = alpha
        self.p_max = p_max
        self.p = None
        self.q = None
        self.current = None
        self.node_obj = None

class ACZeroBranch:
    def __init__(self, idx, i_node, j_node, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.run_stat = run_stat
        self.p = None
        self.q = None
        self.current = None
        self.i_node_obj = None
        self.j_node_obj = None

class ACSwitch:
    def __init__(self, idx, i_node, j_node, status, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.status = status
        self.run_stat = run_stat
        self.current = None
        self.p = None
        self.q = None
        self.current = None
        self.i_node_obj = None
        self.j_node_obj = None


class ACBreak(ACSwitch):
    pass


class ACTransformer:
    """AC transformer with a T-type single-ended shunt model.

    ``gt`` and ``bt`` are the conductance and susceptance of the i-side
    grounding branch. They are not line-style total charging terms. The optional
    legacy ``b`` argument is accepted only for old E files and is converted to
    ``bt = b / 2`` to preserve the previous per-end charging magnitude.
    """

    def __init__(self, idx, i_node, j_node, r, x, tap, shift, gt=0.0, bt=0.0, run_stat=1, b=None):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r = r
        self.x = x
        self.gt = gt
        self.bt = float(b) / 2.0 if b is not None and bt == 0.0 else bt
        self.tap = tap
        self.shift = shift
        self.run_stat = run_stat
        self.is_alive = True
        self.current = None
        self.i_p = None
        self.i_q = None
        self.i_c = None
        self.j_p = None
        self.j_q = None
        self.j_c = None


class ACThreeWindingTransformer:
    """Three-winding AC transformer represented by a star equivalent.

    ``i_r/i_x``, ``j_r/j_x`` and ``k_r/k_x`` are the three winding leakage
    impedances from each external terminal to the eliminated internal star
    point. ``gt + j*bt`` is the i-side single-ended magnetizing admittance.
    Each winding may have its own complex tap ratio.
    """

    def __init__(
        self,
        idx,
        i_node,
        j_node,
        k_node,
        i_r,
        i_x,
        j_r,
        j_x,
        k_r,
        k_x,
        i_tap=1.0,
        i_shift=0.0,
        j_tap=1.0,
        j_shift=0.0,
        k_tap=1.0,
        k_shift=0.0,
        gt=0.0,
        bt=0.0,
        run_stat=1,
    ):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.k_node = k_node
        self.i_r = i_r
        self.i_x = i_x
        self.j_r = j_r
        self.j_x = j_x
        self.k_r = k_r
        self.k_x = k_x
        self.gt = gt
        self.bt = bt
        self.i_tap = i_tap
        self.i_shift = i_shift
        self.j_tap = j_tap
        self.j_shift = j_shift
        self.k_tap = k_tap
        self.k_shift = k_shift
        self.run_stat = run_stat
        self.is_alive = True
        self.i_p = None
        self.i_q = None
        self.i_c = None
        self.j_p = None
        self.j_q = None
        self.j_c = None
        self.k_p = None
        self.k_q = None
        self.k_c = None
        self.i_node_obj = None
        self.j_node_obj = None
        self.k_node_obj = None


ACAC_SIDE_CONTROL_TYPES = {"Q", "V"}
ACAC_LEGACY_TO_PAIR = {
    "PQQ": ("Q", "Q"),
    "PVQ": ("V", "Q"),
    "PQV": ("Q", "V"),
    "PVV": ("V", "V"),
}
ACAC_PAIR_TO_LEGACY = {value: key for key, value in ACAC_LEGACY_TO_PAIR.items()}


def acac_control_pair_from_legacy(control_type):
    label = str(control_type or "PQQ").upper()
    if label not in ACAC_LEGACY_TO_PAIR:
        raise ValueError(f"未知 ACACConverter 控制模式: {control_type}")
    return ACAC_LEGACY_TO_PAIR[label]


def acac_legacy_control_label(i_control_type, j_control_type):
    i_label = str(i_control_type or "Q").upper()
    j_label = str(j_control_type or "Q").upper()
    legacy = ACAC_PAIR_TO_LEGACY.get((i_label, j_label))
    if legacy is None:
        raise ValueError(f"不支持的 ACACConverter 控制组合: ({i_label}, {j_label})")
    return legacy


class ACACConverter:
    def __init__(
        self,
        idx,
        i_node,
        j_node,
        r1,
        r2,
        i_control_type,
        j_control_type,
        p_set,
        i_q_set,
        j_q_set,
        i_v_set,
        j_v_set,
        run_stat=1,
    ):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r1 = r1
        self.r2 = r2
        self.i_control_type = str(i_control_type or "Q").upper()
        self.j_control_type = str(j_control_type or "Q").upper()
        self.p_set = p_set
        self.i_q_set = i_q_set
        self.j_q_set = j_q_set
        self.i_v_set = i_v_set
        self.j_v_set = j_v_set
        self.run_stat = run_stat
        self.i_p = None
        self.i_q = None
        self.j_p = None
        self.j_q = None
        self.i_i = None
        self.j_i = None
        self.i_node_obj = None
        self.j_node_obj = None

    @property
    def control_type(self):
        return acac_legacy_control_label(self.i_control_type, self.j_control_type)

    @control_type.setter
    def control_type(self, value):
        i_control_type, j_control_type = acac_control_pair_from_legacy(value)
        self.i_control_type = i_control_type
        self.j_control_type = j_control_type

from efile_read import efile_factory_from_file, efile_factory_from_rows
from unit_system import normalize_model_named_units


_AC_ROW_CLASS_BY_TABLE = {
    "ACNode": ACNode,
    "ACBranch": ACBranch,
    "ACLoad": ACLoad,
    "ACGenerator": ACGenerator,
    "ACShuntCompensator": ACShuntCompensator,
    "ACZeroBranch": ACZeroBranch,
    "ACSwitch": ACSwitch,
    "ACBreak": ACBreak,
    "ACTransformer": ACTransformer,
    "ACThreeWindingTransformer": ACThreeWindingTransformer,
    "ACACConverter": ACACConverter,
}


_AC_ROW_DEFAULT_ATTRS = {
    "ACNode": {
        "idx": 0,
        "name": "",
        "vbase": 0.0,
        "voltage": 1.0,
        "angle": 0.0,
        "isl": 0,
        "run_stat": 1,
        "isl_obj": None,
    },
    "ACBranch": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "r": 0.0,
        "x": 0.0,
        "b": 0.0,
        "run_stat": 1,
        "i_p": None,
        "i_q": None,
        "i_c": None,
        "j_p": None,
        "j_q": None,
        "j_c": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
    "ACLoad": {
        "idx": 0,
        "name": "",
        "node": 0,
        "pbase": 1.0,
        "pv0": 0.0,
        "pv1": 0.0,
        "pv2": 0.0,
        "qbase": 1.0,
        "qv0": 0.0,
        "qv1": 0.0,
        "qv2": 0.0,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "node_obj": None,
    },
    "ACGenerator": {
        "idx": 0,
        "name": "",
        "node": 0,
        "control_type": "",
        "p_set": 0.0,
        "q_set": 0.0,
        "v_set": 1.0,
        "alpha": None,
        "p_max": None,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "node_obj": None,
    },
    "ACShuntCompensator": {
        "idx": 0,
        "name": "",
        "node": 0,
        "control_type": "Q",
        "q_set": 0.0,
        "g_set": 0.0,
        "b_set": 0.0,
        "v_set": 1.0,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "is_alive": False,
        "node_obj": None,
    },
    "ACZeroBranch": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
    "ACSwitch": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "status": 1,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
    "ACBreak": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "status": 1,
        "run_stat": 1,
        "p": None,
        "q": None,
        "current": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
    "ACTransformer": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "r": 0.0,
        "x": 0.0,
        "gt": 0.0,
        "bt": 0.0,
        "tap": 1.0,
        "shift": 0.0,
        "run_stat": 1,
        "is_alive": True,
        "current": None,
        "i_p": None,
        "i_q": None,
        "i_c": None,
        "j_p": None,
        "j_q": None,
        "j_c": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
    "ACThreeWindingTransformer": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "k_node": 0,
        "i_r": 0.0,
        "i_x": 0.0,
        "j_r": 0.0,
        "j_x": 0.0,
        "k_r": 0.0,
        "k_x": 0.0,
        "gt": 0.0,
        "bt": 0.0,
        "i_tap": 1.0,
        "i_shift": 0.0,
        "j_tap": 1.0,
        "j_shift": 0.0,
        "k_tap": 1.0,
        "k_shift": 0.0,
        "run_stat": 1,
        "is_alive": True,
        "i_p": None,
        "i_q": None,
        "i_c": None,
        "j_p": None,
        "j_q": None,
        "j_c": None,
        "k_p": None,
        "k_q": None,
        "k_c": None,
        "i_node_obj": None,
        "j_node_obj": None,
        "k_node_obj": None,
    },
    "ACACConverter": {
        "idx": 0,
        "name": "",
        "i_node": 0,
        "j_node": 0,
        "r1": 0.0,
        "r2": 0.0,
        "i_control_type": "Q",
        "j_control_type": "Q",
        "p_set": 0.0,
        "i_q_set": 0.0,
        "j_q_set": 0.0,
        "i_v_set": 1.0,
        "j_v_set": 1.0,
        "run_stat": 1,
        "i_p": None,
        "i_q": None,
        "j_p": None,
        "j_q": None,
        "i_i": None,
        "j_i": None,
        "i_node_obj": None,
        "j_node_obj": None,
    },
}


def _float_row_alias(row_values, names, default=0.0):
    for name in names:
        value = row_values.get(name)
        if value not in (None, ""):
            return float(value)
    return float(default)


def _normalize_three_winding_row_values(row_values, values):
    direct_aliases = {
        f"{terminal}_{part}": (f"{terminal}_{part}", f"{part}_{terminal}")
        for terminal in ("i", "j", "k")
        for part in ("r", "x")
    }
    if all(any(name in row_values for name in aliases) for aliases in direct_aliases.values()):
        for attr, aliases in direct_aliases.items():
            values[attr] = _float_row_alias(row_values, aliases)
    else:
        z_ij = complex(
            _float_row_alias(row_values, ("ij_r", "ji_r", "r_ij", "r_ji")),
            _float_row_alias(row_values, ("ij_x", "ji_x", "x_ij", "x_ji")),
        )
        z_ik = complex(
            _float_row_alias(row_values, ("ik_r", "ki_r", "r_ik", "r_ki")),
            _float_row_alias(row_values, ("ik_x", "ki_x", "x_ik", "x_ki")),
        )
        z_jk = complex(
            _float_row_alias(row_values, ("jk_r", "kj_r", "r_jk", "r_kj")),
            _float_row_alias(row_values, ("jk_x", "kj_x", "x_jk", "x_kj")),
        )
        z_i = 0.5 * (z_ij + z_ik - z_jk)
        z_j = 0.5 * (z_ij + z_jk - z_ik)
        z_k = 0.5 * (z_ik + z_jk - z_ij)
        for terminal, impedance in (("i", z_i), ("j", z_j), ("k", z_k)):
            values[f"{terminal}_r"] = float(impedance.real)
            values[f"{terminal}_x"] = float(impedance.imag)

    values["gt"] = _float_row_alias(row_values, ("gt", "g"), values.get("gt", 0.0))
    values["bt"] = _float_row_alias(row_values, ("bt", "b"), values.get("bt", 0.0))
    for terminal in ("i", "j", "k"):
        values[f"{terminal}_tap"] = _float_row_alias(
            row_values,
            (f"{terminal}_tap", f"tap_{terminal}"),
            values.get(f"{terminal}_tap", 1.0),
        )
        values[f"{terminal}_shift"] = _float_row_alias(
            row_values,
            (f"{terminal}_shift", f"shift_{terminal}"),
            values.get(f"{terminal}_shift", 0.0),
        )


def _coerce_ac_rows(rows, table_name):
    row_cls = _AC_ROW_CLASS_BY_TABLE[table_name]
    defaults = _AC_ROW_DEFAULT_ATTRS.get(table_name, {})
    output = []
    for row in rows:
        if isinstance(row, row_cls):
            if table_name == "ACTransformer":
                if not hasattr(row, "gt"):
                    row.gt = 0.0
                if not hasattr(row, "bt") and hasattr(row, "b"):
                    row.bt = float(row.b) / 2.0
            elif table_name == "ACThreeWindingTransformer":
                _normalize_three_winding_row_values(row.__dict__, row.__dict__)
            elif table_name == "ACACConverter":
                if not hasattr(row, "i_control_type") or not hasattr(row, "j_control_type"):
                    i_ctrl, j_ctrl = acac_control_pair_from_legacy(getattr(row, "control_type", "PQQ"))
                    row.i_control_type = str(getattr(row, "i_control_type", i_ctrl)).upper()
                    row.j_control_type = str(getattr(row, "j_control_type", j_ctrl)).upper()
            output.append(row)
            continue
        row_values = getattr(row, "__dict__", {})
        obj = row_cls.__new__(row_cls)
        obj.__dict__.update(defaults)
        obj.__dict__.update(row_values)
        if table_name == "ACTransformer":
            # Compatibility for older E files/objects that used one total
            # shunt susceptance column b. The new transformer model is T-type
            # with a single grounding branch on the i side, so preserve the
            # old per-end magnitude by mapping bt = b / 2.
            values = obj.__dict__
            if "gt" not in row_values:
                values["gt"] = 0.0
            if "bt" not in row_values and "b" in row_values:
                values["bt"] = float(values["b"]) / 2.0
        elif table_name == "ACThreeWindingTransformer":
            _normalize_three_winding_row_values(row_values, obj.__dict__)
        elif table_name == "ACACConverter":
            values = obj.__dict__
            if "i_control_type" not in row_values or "j_control_type" not in row_values:
                i_ctrl, j_ctrl = acac_control_pair_from_legacy(values.get("control_type", "PQQ"))
                values.setdefault("i_control_type", i_ctrl)
                values.setdefault("j_control_type", j_ctrl)
            values["i_control_type"] = str(values.get("i_control_type", "Q")).upper()
            values["j_control_type"] = str(values.get("j_control_type", "Q")).upper()
        output.append(obj)
    return output


class ACPowerNetwork:
    def __init__(self):
        self.nodes = []
        self.branches = []
        self.loads = []
        self.generators = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.transformers = []
        self.three_winding_transformers = []
        self.shunt_compensators = []
        self.acac_converters = []
        self.islands = []
        self.buses = []

        self.node_dict = {}
        self.bus_dict = {}
        self.node_to_bus = {}
        self.switch_dict = {}
        self.break_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branch_dict = {}
        self.branch_dict = {}
        self.zero_branche_dict = self.zero_branch_dict
        self.branche_dict = self.branch_dict
        self.transformer_dict = {}
        self.three_winding_transformer_dict = {}
        self.shunt_compensator_dict = {}
        self.acac_converter_dict = {}

    def add_node(self, idx, vbase, voltage=1.0, angle=0.0, run_stat=1):
        node = ACNode(idx, vbase, voltage, angle, run_stat)
        self.nodes.append(node)
        return node

    def add_branch(self, idx, i_node, j_node, r, x=0.0, b=0.0, run_stat=1):
        br = ACBranch(idx, i_node, j_node, r, x, b, run_stat)
        self.branches.append(br)
        return br

    def add_load(self, idx, node, pbase, pv0, pv1, pv2, qbase, qv0, qv1, qv2, run_stat=1):
        ld = ACLoad(idx, node, pbase, pv0, pv1, pv2, qbase, qv0, qv1, qv2, run_stat)
        self.loads.append(ld)
        return ld

    def add_generator(self, idx, node, control_type, p_set,q_set, v_set, alpha=None, run_stat=1, p_max=None):
        gen = ACGenerator(idx, node, control_type, p_set, q_set, v_set, alpha, run_stat, p_max)
        self.generators.append(gen)
        return gen

    def add_zero_branch(self, idx, i_node, j_node, run_stat=1):
        zb = ACZeroBranch(idx, i_node, j_node, run_stat)
        self.zero_branches.append(zb)
        return zb

    def add_switch(self, idx, i_node, j_node, status, run_stat=1):
        sw = ACSwitch(idx, i_node, j_node, status, run_stat)
        self.switches.append(sw)
        return sw

    def add_break(self, idx, i_node, j_node, status, run_stat=1):
        brk = ACBreak(idx, i_node, j_node, status, run_stat)
        self.breakers.append(brk)
        return brk

    def add_transformer(self, idx, i_node, j_node, r, x, tap, shift, gt=0.0, bt=0.0, run_stat=1, b=None):
        trfm = ACTransformer(idx, i_node, j_node, r, x, tap, shift, gt, bt, run_stat, b=b)
        self.transformers.append(trfm)
        return trfm

    def add_three_winding_transformer(
        self,
        idx,
        i_node,
        j_node,
        k_node,
        i_r,
        i_x,
        j_r,
        j_x,
        k_r,
        k_x,
        i_tap=1.0,
        i_shift=0.0,
        j_tap=1.0,
        j_shift=0.0,
        k_tap=1.0,
        k_shift=0.0,
        gt=0.0,
        bt=0.0,
        run_stat=1,
    ):
        trfm = ACThreeWindingTransformer(
            idx,
            i_node,
            j_node,
            k_node,
            i_r,
            i_x,
            j_r,
            j_x,
            k_r,
            k_x,
            i_tap,
            i_shift,
            j_tap,
            j_shift,
            k_tap,
            k_shift,
            gt,
            bt,
            run_stat,
        )
        self.three_winding_transformers.append(trfm)
        return trfm

    def add_acac_converter(
        self,
        idx,
        i_node,
        j_node,
        r1,
        r2,
        i_control_type,
        j_control_type,
        p_set,
        i_q_set,
        j_q_set,
        i_v_set,
        j_v_set,
        run_stat=1,
    ):
        conv = ACACConverter(
            idx,
            i_node,
            j_node,
            r1,
            r2,
            i_control_type,
            j_control_type,
            p_set,
            i_q_set,
            j_q_set,
            i_v_set,
            j_v_set,
            run_stat,
        )
        self.acac_converters.append(conv)
        return conv

    def _load_from_model(self, *, units_already_normalized: bool = False):
        if units_already_normalized:
            self.p_base = float(self.model.p_base)
            self.p_base_kW = float(self.model.p_base_kW)
            self.u_scale = float(self.model.u_scale)
            self.p_scale = float(self.model.p_scale)
            self.i_scale = float(self.model.i_scale)
        else:
            self.p_base = normalize_model_named_units(self.model)
            self.p_base_kW = float(self.model.p_base_kW)
            self.u_scale = float(self.model.u_scale)
            self.p_scale = float(self.model.p_scale)
            self.i_scale = float(self.model.i_scale)
        self.branches = _coerce_ac_rows(getattr(self.model, 'ACBranch', []), "ACBranch")
        self.nodes = _coerce_ac_rows(getattr(self.model, 'ACNode', []), "ACNode")
        self.generators = _coerce_ac_rows(getattr(self.model, 'ACGenerator', []), "ACGenerator")
        self.loads = _coerce_ac_rows(getattr(self.model, 'ACLoad', []), "ACLoad")
        self.switches = _coerce_ac_rows(getattr(self.model, 'ACSwitch', []), "ACSwitch")
        self.breakers = _coerce_ac_rows(getattr(self.model, 'ACBreak', []), "ACBreak")
        self.zero_branches = _coerce_ac_rows(getattr(self.model, 'ACZeroBranch', []), "ACZeroBranch")
        self.transformers = _coerce_ac_rows(getattr(self.model, 'ACTransformer', []), "ACTransformer")
        three_winding_rows = getattr(self.model, 'ACThreeWindingTransformer', None)
        if three_winding_rows is None:
            three_winding_rows = getattr(self.model, 'AC3WTransformer', [])
        self.three_winding_transformers = _coerce_ac_rows(
            three_winding_rows,
            "ACThreeWindingTransformer",
        )
        self.shunt_compensators = _coerce_ac_rows(getattr(self.model, 'ACShuntCompensator', []), "ACShuntCompensator")
        self.acac_converters = _coerce_ac_rows(getattr(self.model, 'ACACConverter', []), "ACACConverter")
        self.node_dict = {}
        self.bus_dict = {}
        self.node_to_bus = {}
        self.switch_dict = {}
        self.break_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branch_dict = {}
        self.branch_dict = {}
        self.zero_branche_dict = self.zero_branch_dict
        self.branche_dict = self.branch_dict
        self.transformer_dict = {}
        self.three_winding_transformer_dict = {}
        self.shunt_compensator_dict = {}
        self.acac_converter_dict = {}
        self.islands = []
        self.buses = []

    def read_from_model(self, model):
        self.model = efile_factory_from_rows(model) if isinstance(model, dict) else model
        self._load_from_model()

    def read_from_file(self, file_name):
        self.source = str(file_name)
        self.read_from_model(efile_factory_from_file(file_name))

    def format_assoc(self):
        """
        重新形成索引，把 i_node, j_node, node，等都替换成实际对象的引用。
        """

        # 建立节点索引到节点对象的映射
        self.node_dict = {node.idx: node for node in self.nodes}
        self.switch_dict = {sw.idx: sw for sw in self.switches}
        self.break_dict = {brk.idx: brk for brk in self.breakers}
        self.load_dict = {ld.idx: ld for ld in self.loads}
        self.generator_dict = {gen.idx: gen for gen in self.generators}
        self.zero_branch_dict = {zbr.idx: zbr for zbr in self.zero_branches}
        self.branch_dict = {br.idx: br for br in self.branches}
        self.zero_branche_dict = self.zero_branch_dict
        self.branche_dict = self.branch_dict
        self.transformer_dict = {trfm.idx: trfm for trfm in self.transformers}
        self.three_winding_transformer_dict = {
            trfm.idx: trfm for trfm in self.three_winding_transformers
        }
        self.shunt_compensator_dict = {scp.idx: scp for scp in self.shunt_compensators}
        self.acac_converter_dict = {conv.idx: conv for conv in self.acac_converters}

        for node in self.nodes:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.zero_branches = []
            node.transformers = []
            node.three_winding_transformers = []
            node.shunt_compensators = []
            node.acac_converters = []
            node.bus = None
            node.bus_obj = None

        # 建立设备到节点之间的连接关系。
        for gen in self.generators:
            gen.node_obj = self.node_dict.get(gen.node, None)
            if gen.node_obj:
                gen.node_obj.generators.append(gen)

        for ld in self.loads:
            ld.node_obj = self.node_dict.get(ld.node, None)
            if ld.node_obj:
                ld.node_obj.loads.append(ld)

        for scp in self.shunt_compensators:
            scp.node_obj = self.node_dict.get(scp.node, None)
            if scp.node_obj:
                scp.node_obj.shunt_compensators.append(scp)

        for nb in self.branches:
            nb.i_node_obj = self.node_dict.get(nb.i_node, None)
            nb.j_node_obj = self.node_dict.get(nb.j_node, None)
            if nb.i_node_obj:
                nb.i_node_obj.branches.append(nb)
            if nb.j_node_obj:
                nb.j_node_obj.branches.append(nb)

        for nb in self.transformers:
            nb.i_node_obj = self.node_dict.get(nb.i_node, None)
            nb.j_node_obj = self.node_dict.get(nb.j_node, None)
            if nb.i_node_obj:
                nb.i_node_obj.transformers.append(nb)
            if nb.j_node_obj:
                nb.j_node_obj.transformers.append(nb)

        for nb in self.three_winding_transformers:
            nb.i_node_obj = self.node_dict.get(nb.i_node, None)
            nb.j_node_obj = self.node_dict.get(nb.j_node, None)
            nb.k_node_obj = self.node_dict.get(nb.k_node, None)
            for node_obj in (nb.i_node_obj, nb.j_node_obj, nb.k_node_obj):
                if node_obj:
                    node_obj.three_winding_transformers.append(nb)

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

        for zbr in self.zero_branches:
            zbr.i_node_obj = self.node_dict.get(zbr.i_node, None)
            zbr.j_node_obj = self.node_dict.get(zbr.j_node, None)
            if zbr.i_node_obj:
                zbr.i_node_obj.zero_branches.append(zbr)
            if zbr.j_node_obj:
                zbr.j_node_obj.zero_branches.append(zbr)

        for conv in self.acac_converters:
            conv.i_node_obj = self.node_dict.get(conv.i_node, None)
            conv.j_node_obj = self.node_dict.get(conv.j_node, None)
            if conv.i_node_obj:
                conv.i_node_obj.acac_converters.append(conv)
            if conv.j_node_obj:
                conv.j_node_obj.acac_converters.append(conv)

    def topo(self):
        from model import topology as network_topology

        network_topology.prepare_ac_topology(self)

    def det_isl_alive_stat(self):

        for isl in self.islands:
            isl.is_alive = False
            isl.slack_nodes = []
            isl.v_gens = []
            isl.buses = []
            isl.gens = []
            isl.loads = []
            isl.branches = []
            isl.zero_branches = []
            isl.switches = []
            isl.breakers = []
            isl.transformers = []
            isl.three_winding_transformers = []
            isl.shunt_compensators = []
            isl.acac_converters = []

        # 检查节点
        for node in self.nodes:
            node.v_gens = []

        for bus in self.buses:
            bus.v_gens = []
            bus.generators = []
            bus.loads = []
            bus.branches = []
            bus.switches = []
            bus.breakers = []
            bus.zero_branches = []
            bus.transformers = []
            bus.three_winding_transformers = []
            bus.shunt_compensators = []
            bus.acac_converters = []
            if bus.isl_obj is not None:
                bus.isl_obj.buses.append(bus)

        # 检查发电机
        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            node = gen.node_obj
            if node is None or node.isl_obj is None:
                continue
            node.isl_obj.gens.append(gen)
            if gen.control_type in ['V', 'SLACK', 'PH']:
                node.isl_obj.is_alive = True
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)
                slack_bus = node.bus_obj or node
                if slack_bus not in node.isl_obj.slack_nodes:
                    node.isl_obj.slack_nodes.append(slack_bus)
            elif gen.control_type == 'PV':
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)

        for load in self.loads:
            if load.run_stat == 0:
                continue
            if load.node_obj is None or load.node_obj.isl_obj is None:
                continue
            load.node_obj.isl_obj.loads.append(load)

        for scp in self.shunt_compensators:
            if scp.run_stat == 0:
                continue
            if scp.node_obj is None or scp.node_obj.isl_obj is None:
                continue
            scp.node_obj.isl_obj.shunt_compensators.append(scp)

        for switch in self.switches:
            if switch.i_node_obj is None or switch.j_node_obj is None:
                continue
            if switch.run_stat == 0 or switch.status == 0:
                continue
            if switch.i_node_obj.isl_obj and switch.j_node_obj.isl_obj and switch.i_node_obj.isl_obj ==  switch.j_node_obj.isl_obj:
                switch.i_node_obj.isl_obj.switches.append(switch)

        for br in self.branches:
            if br.run_stat == 0:
                continue
            if br.i_node_obj is None or br.j_node_obj is None:
                continue
            if br.i_node_obj.isl_obj and br.j_node_obj.isl_obj and br.i_node_obj.isl_obj ==  br.j_node_obj.isl_obj:
                br.i_node_obj.isl_obj.branches.append(br)


        for trfm in self.transformers:
            if trfm.run_stat == 0:
                continue
            if trfm.i_node_obj is None or trfm.j_node_obj is None:
                continue
            if trfm.i_node_obj.isl_obj and trfm.j_node_obj.isl_obj and trfm.i_node_obj.isl_obj ==  trfm.j_node_obj.isl_obj:
                trfm.i_node_obj.isl_obj.transformers.append(trfm)

        for trfm in self.three_winding_transformers:
            if trfm.run_stat == 0:
                continue
            terminal_nodes = (trfm.i_node_obj, trfm.j_node_obj, trfm.k_node_obj)
            if any(node is None or node.isl_obj is None for node in terminal_nodes):
                continue
            if terminal_nodes[0].isl_obj is terminal_nodes[1].isl_obj is terminal_nodes[2].isl_obj:
                terminal_nodes[0].isl_obj.three_winding_transformers.append(trfm)


        for zbr in self.zero_branches:
            if zbr.run_stat == 0:
                continue
            if zbr.i_node_obj is None or zbr.j_node_obj is None:
                continue
            if zbr.i_node_obj.isl_obj and zbr.j_node_obj.isl_obj and zbr.i_node_obj.isl_obj ==  zbr.j_node_obj.isl_obj:
                zbr.i_node_obj.isl_obj.zero_branches.append(zbr)

        for brk in self.breakers:
            if brk.run_stat == 0 or brk.status == 0:
                continue
            if brk.i_node_obj is None or brk.j_node_obj is None:
                continue
            if brk.i_node_obj.isl_obj and brk.j_node_obj.isl_obj and brk.i_node_obj.isl_obj == brk.j_node_obj.isl_obj:
                brk.i_node_obj.isl_obj.breakers.append(brk)

        for conv in self.acac_converters:
            if conv.run_stat == 0:
                continue
            if conv.i_node_obj is None or conv.j_node_obj is None:
                continue
            if conv.i_node_obj.isl_obj and conv.j_node_obj.isl_obj and conv.i_node_obj.isl_obj == conv.j_node_obj.isl_obj:
                conv.i_node_obj.isl_obj.acac_converters.append(conv)

        for isl in self.islands:
            if len(isl.slack_nodes)  >= 1:
                isl.is_alive = True

        for bus in self.buses:
            bus.is_alive = bus.run_stat == 1 and bus.isl_obj is not None and bus.isl_obj.is_alive

        for node in self.nodes:
            node.is_alive = node.run_stat == 1 and node.isl_obj is not None and node.isl_obj.is_alive

        self.alive_buses = [bus for bus in self.buses if bus.is_alive]


        for load in self.loads:
            node = load.node_obj
            load.is_alive = (
                node is not None
                and node.isl_obj is not None
                and load.run_stat == 1
                and node.isl_obj.is_alive
            )

        for gen in self.generators:
            node = gen.node_obj
            gen.is_alive = (
                node is not None
                and node.isl_obj is not None
                and gen.run_stat == 1
                and node.isl_obj.is_alive
            )

        for scp in self.shunt_compensators:
            node = scp.node_obj
            scp.is_alive = (
                node is not None
                and node.isl_obj is not None
                and scp.run_stat == 1
                and node.isl_obj.is_alive
            )

        for br in self.branches:
            if br.i_node_obj is None or br.j_node_obj is None or br.run_stat == 0:
                br.is_alive = False
                continue
            br.is_alive = br.i_node_obj.is_alive and br.j_node_obj.is_alive

        for trfm in self.transformers:
            if trfm.i_node_obj is None or trfm.j_node_obj is None or trfm.run_stat == 0:
                trfm.is_alive = False
                continue
            trfm.is_alive = trfm.i_node_obj.is_alive and trfm.j_node_obj.is_alive

        for trfm in self.three_winding_transformers:
            terminal_nodes = (trfm.i_node_obj, trfm.j_node_obj, trfm.k_node_obj)
            trfm.is_alive = (
                trfm.run_stat == 1
                and all(node is not None and node.is_alive for node in terminal_nodes)
                and terminal_nodes[0].isl_obj is terminal_nodes[1].isl_obj
                and terminal_nodes[0].isl_obj is terminal_nodes[2].isl_obj
            )

        for zbr in self.zero_branches:
            if zbr.i_node_obj is None or zbr.j_node_obj is None or zbr.run_stat == 0:
                zbr.is_alive = False
                continue
            zbr.is_alive = zbr.i_node_obj.is_alive and zbr.j_node_obj.is_alive

        for brk in self.breakers:
            if brk.i_node_obj is None or brk.j_node_obj is None or brk.run_stat == 0 or brk.status == 0:
                brk.is_alive = False
                continue
            brk.is_alive = brk.i_node_obj.is_alive and brk.j_node_obj.is_alive

        for sw in self.switches:
            if sw.i_node_obj is None or sw.j_node_obj is None or sw.status == 0 or sw.run_stat == 0:
                sw.is_alive = False
                continue
            sw.is_alive = sw.i_node_obj.is_alive and sw.j_node_obj.is_alive

        for conv in self.acac_converters:
            if conv.i_node_obj is None or conv.j_node_obj is None or conv.run_stat == 0:
                conv.is_alive = False
                continue
            conv.is_alive = conv.i_node_obj.is_alive and conv.j_node_obj.is_alive

    def print_isl_info(self):
        for isl in self.islands:
            print(f"isl {isl.idx} is_alive = {isl.is_alive}")
            print(f"    buses = {len(isl.buses)}:")
            for node in isl.buses:
                is_slack = node in isl.slack_nodes
                print(f"        {node.idx} {node.name} vbase: {node.vbase} slack:{int(is_slack)}")
            print(f"    gens = {len(isl.gens)}:")
            for gen in isl.gens:
                print(f"        {gen.idx} {gen.name} node = {gen.node} control_type = {gen.control_type}")
            print(f"    loads = {len(isl.loads)}:")
            for load in isl.loads:
                print(f"        {load.idx} {load.name} node = {load.node}")

            print(f"    branches = {len(isl.branches)}:")
            for br in isl.branches:
                print(f"        {br.idx} {br.name} i_node = {br.i_node} j_node = {br.j_node} r = {br.r} x = {br.x} b = {br.b}")

            print(f"    transformers = {len(isl.transformers)}:")
            for trfm in isl.transformers:
                print(f"        {trfm.idx} {trfm.name} i_node = {trfm.i_node} j_node = {trfm.j_node} r = {trfm.r} x = {trfm.x} gt = {trfm.gt} bt = {trfm.bt} tap = {trfm.tap} shift = {trfm.shift}")

            print(f"    switches = {len(isl.switches)}:")
            for sw in isl.switches:
                print(f"        {sw.idx} {sw.name} i_node = {sw.i_node} j_node = {sw.j_node} status = {sw.status}")

            print(f"    zero_branches = {len(isl.zero_branches)}:")
            for zbr in isl.zero_branches:
                print(f"        {zbr.idx} {zbr.name} i_node = {zbr.i_node} j_node = {zbr.j_node}")

            print(f"    shunt_compensators = {len(isl.shunt_compensators)}:")
            for scp in isl.shunt_compensators:
                print(f"        {scp.idx} {scp.name} node = {scp.node} g = {scp.g_set} b = {scp.b_set}")

            print(f"    acac_converters = {len(isl.acac_converters)}:")
            for conv in isl.acac_converters:
                print(
                    f"        {conv.idx} {conv.name} i_node = {conv.i_node} "
                    f"j_node = {conv.j_node} control_type = {conv.control_type}"
                )

    def check_topo(self):
        """
        检查网络模型的完整性、一致性和合理性。
        返回一个错误信息列表，若列表为空则表示所有检查通过。
        """
        errors = []
        warns = []
        # 确保拓扑岛已划分
        if len(self.islands) == 0:
            self.topo()

        # 记录每个节点的引用次数
        node_ref_count = {node.idx: 0 for node in self.nodes}

        # 辅助函数：检查节点是否存在，并增加引用计数
        def check_node(node_idx, dev_type, dev):
            if node_idx not in self.node_dict:
                errors.append(f"设备 {dev_type}[{dev.idx}] {dev.name} 引用的节点 {node_idx} 不存在")
            elif self.node_dict[node_idx].run_stat == 1:
                node_ref_count[node_idx] += 1

        # 遍历所有设备，检查节点存在性
        for br in self.branches:
            if br.run_stat == 0:
                continue
            check_node(br.i_node, 'Branch', br)
            check_node(br.j_node, 'Branch', br)

        for trfm in self.transformers:
            if trfm.run_stat == 0:
                continue
            check_node(trfm.i_node, 'Trfm', trfm)
            check_node(trfm.j_node, 'Trfm', trfm)

        for trfm in self.three_winding_transformers:
            if trfm.run_stat == 0:
                continue
            check_node(trfm.i_node, 'ThreeWindingTrfm', trfm)
            check_node(trfm.j_node, 'ThreeWindingTrfm', trfm)
            check_node(trfm.k_node, 'ThreeWindingTrfm', trfm)

        for zb in self.zero_branches:
            if zb.run_stat == 0:
                continue
            check_node(zb.i_node, 'ZeroBranch', zb)
            check_node(zb.j_node, 'ZeroBranch', zb)

        for brk in self.breakers:
            if brk.run_stat == 0 or brk.status == 0:
                continue
            check_node(brk.i_node, 'Break', brk)
            check_node(brk.j_node, 'Break', brk)

        for sw in self.switches:
            if sw.run_stat == 0:
                continue
            check_node(sw.i_node, 'Switch', sw)
            check_node(sw.j_node, 'Switch', sw)

        for ld in self.loads:
            if ld.run_stat == 0:
                continue
            check_node(ld.node, 'Load', ld)

        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            check_node(gen.node, 'Generator', gen)

        for scp in self.shunt_compensators:
            if scp.run_stat == 0:
                continue
            check_node(scp.node, 'ShuntCompensator', scp)

        for conv in self.acac_converters:
            if conv.run_stat == 0:
                continue
            check_node(conv.i_node, 'ACACConverter', conv)
            check_node(conv.j_node, 'ACACConverter', conv)
            if conv.i_node == conv.j_node:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 两端不能连接同一个 AC 节点")
            i_ctrl = str(getattr(conv, "i_control_type", "Q")).upper()
            j_ctrl = str(getattr(conv, "j_control_type", "Q")).upper()
            if i_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} i_control_type {i_ctrl} 不支持")
            if j_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} j_control_type {j_ctrl} 不支持")

        # 检查节点悬空
        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {node.name} 未关联任何设备")
            if node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {node.name} 单端悬空，请检查！")

        # 电压基值一致性只与支路两端有关，不需要按岛重复扫描全部设备。
        for dev in self.branches:
            if dev.run_stat != 1:
                continue
            if dev.i_node_obj is None or dev.j_node_obj is None:
                continue
            if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                continue
            if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                str_info = f" 支路 {dev.idx} {dev.name} 两端节点的电压碁值不同:{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                errors.append(str_info)

        for dev in self.switches:
            if dev.run_stat != 1:
                continue
            if dev.i_node_obj is None or dev.j_node_obj is None:
                continue
            if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                continue
            if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                str_info = f" 开关 {dev.idx} {dev.name} 两端节点的电压碁值不同:{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                errors.append(str_info)

        for dev in self.zero_branches:
            if dev.run_stat != 1:
                continue
            if dev.i_node_obj is None or dev.j_node_obj is None:
                continue
            if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                continue
            if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                str_info = f" 零阻抗支路 {dev.idx} {dev.name} 两端节点的电压碁值不同:{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                errors.append(str_info)

        for dev in self.breakers:
            if dev.run_stat != 1 or dev.status != 1:
                continue
            if dev.i_node_obj is None or dev.j_node_obj is None:
                continue
            if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                continue
            if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                str_info = f" 断路器 {dev.idx} {dev.name} 两端节点的电压碁值不同:{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                errors.append(str_info)

        for dev in self.acac_converters:
            if dev.run_stat != 1:
                continue
            if dev.i_node_obj is None or dev.j_node_obj is None:
                continue
            if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                continue
            if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                str_info = (
                    f" ACACConverter {dev.idx} {dev.name} 两端节点的电压碁值不同:"
                    f"{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                )
                errors.append(str_info)

        # 检查每个岛屿
        for isl in self.islands:
            # 电压控制源唯一性（松弛节点或定V发电机）
            if len(isl.slack_nodes) > 1:
                if self._multi_slack_nodes_are_zero_tied(isl.slack_nodes):
                    str_info = f"岛屿 {isl.idx} 存在多个零阻抗等值相连的平衡节点:"
                    for node in isl.slack_nodes:
                        str_info += f" {node.name}"
                    warns.append(str_info)
                else:
                    str_info = f"岛屿 {isl.idx} 存在多个平衡节点:"
                    for node in isl.slack_nodes:
                        str_info += f" {node.name}"
                    errors.append(str_info)

            if len(isl.slack_nodes)  == 0:
                warns.append(f"岛屿 {isl.idx} , 无平衡节点，跳过潮流计算")

        # 检查平衡节点与定V发电机的一致性
        slack_node_set = {slack_node for isl in self.islands for slack_node in isl.slack_nodes}
        for node in self.nodes:
            if node.run_stat != 1:
                continue
            has_slack_gen = any(gen.control_type in ['V', 'SLACK', 'PH'] for gen in getattr(node, 'v_gens', []))
            if node in slack_node_set and not has_slack_gen:
                errors.append(f"平衡节点 {node.idx} {node.name} 未关联任何平衡发电机设备")

        return warns, errors

    def _multi_slack_nodes_are_zero_tied(self, slack_nodes):
        """Return True when same-setpoint slack nodes are shorted by zero branches.

        Large synthetic IEEE cases intentionally connect copied voltage-source
        buses through ACZeroBranch devices. In that topology several V sources
        represent one ideal equal-voltage bus, so topology validation should
        warn instead of rejecting the case as ordinary multi-slack operation.
        """
        if len(slack_nodes) <= 1:
            return False
        slack_ids = {node.idx for node in slack_nodes}
        adj = {node.idx: set() for node in slack_nodes}
        for zbr in [*self.zero_branches, *self.breakers]:
            if getattr(zbr, "run_stat", 1) != 1:
                continue
            if isinstance(zbr, ACBreak) and getattr(zbr, "status", 1) != 1:
                continue
            if zbr.i_node in slack_ids and zbr.j_node in slack_ids:
                adj[zbr.i_node].add(zbr.j_node)
                adj[zbr.j_node].add(zbr.i_node)
        visited = set()
        stack = [next(iter(slack_ids))]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adj[current] - visited)
        if visited != slack_ids:
            return False

        first = slack_nodes[0]
        first_voltage = float(getattr(first, "voltage", 1.0) or 1.0)
        first_angle = float(getattr(first, "angle", 0.0) or 0.0)
        first_vbase = float(getattr(first, "vbase", 0.0) or 0.0)
        for node in slack_nodes[1:]:
            if abs(float(getattr(node, "vbase", 0.0) or 0.0) - first_vbase) > 1e-9:
                return False
            if abs(float(getattr(node, "voltage", 1.0) or 1.0) - first_voltage) > 1e-9:
                return False
            if abs(float(getattr(node, "angle", 0.0) or 0.0) - first_angle) > 1e-9:
                return False
        return True



if __name__ == "__main__":

    net = ACPowerNetwork()
    net.read_from_file("data/ac_net_10.e")

    net.topo()

    net.print_isl_info()

