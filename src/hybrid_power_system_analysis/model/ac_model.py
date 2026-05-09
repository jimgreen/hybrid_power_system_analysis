class ACIsl:
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
        self.nodes = list(nodes or [])
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
        self.shunt_compensators = []
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
    def __init__(self, idx, node, control_type, p_set, q_set, v_set, alpha=None, run_stat=1):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.control_type = control_type
        self.p_set = p_set
        self.q_set = q_set
        self.v_set = v_set
        self.alpha = alpha
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
    def __init__(self, idx, i_node, j_node, r, x, tap, shift, b=0.0, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r = r
        self.x = x
        self.b = b
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
        "b": 0.0,
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
}


def _coerce_ac_rows(rows, table_name):
    row_cls = _AC_ROW_CLASS_BY_TABLE[table_name]
    defaults = _AC_ROW_DEFAULT_ATTRS.get(table_name, {})
    output = []
    for row in rows:
        if isinstance(row, row_cls):
            output.append(row)
            continue
        obj = row_cls.__new__(row_cls)
        obj.__dict__.update(defaults)
        obj.__dict__.update(getattr(row, "__dict__", {}))
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
        self.shunt_compensators = []
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
        self.transformer_dict = {}
        self.shunt_compensator_dict = {}

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

    def add_generator(self, idx, node, control_type, p_set,q_set, v_set, alpha=None, run_stat=1):
        gen = ACGenerator(idx, node, control_type, p_set, q_set, v_set, alpha, run_stat)
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

    def add_transformer(self, idx, i_node, j_node, r, x, tap, shift, b=0.0, run_stat=1):
        trfm = ACTransformer(idx, i_node, j_node, r, x, tap, shift, b, run_stat)
        self.transformers.append(trfm)
        return trfm

    def _load_from_model(self):
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
        self.shunt_compensators = _coerce_ac_rows(getattr(self.model, 'ACShuntCompensator', []), "ACShuntCompensator")
        self.node_dict = {}
        self.bus_dict = {}
        self.node_to_bus = {}
        self.switch_dict = {}
        self.break_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branch_dict = {}
        self.branch_dict = {}
        self.transformer_dict = {}
        self.shunt_compensator_dict = {}
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
        self.transformer_dict = {trfm.idx: trfm for trfm in self.transformers}
        self.shunt_compensator_dict = {scp.idx: scp for scp in self.shunt_compensators}

        for node in self.nodes:
            node.generators = []
            node.loads = []
            node.branches = []
            node.switches = []
            node.breakers = []
            node.zero_branches = []
            node.transformers = []
            node.shunt_compensators = []
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

    def topo(self):

        if len(self.node_dict) == 0:
            self.format_assoc()

        # 重置所有节点的拓扑岛号和母线归属。
        for node in self.nodes:
            node.isl = 0
            node.isl_obj = None
            node.bus = None
            node.bus_obj = None
            node.is_alive = False

        running_nodes = [node for node in self.nodes if node.run_stat == 1]
        running_node_ids = {node.idx for node in running_nodes}
        bus_parent = {node.idx: node.idx for node in running_nodes}

        def find(parent, node_idx):
            root = node_idx
            while parent[root] != root:
                root = parent[root]
            while parent[node_idx] != node_idx:
                next_idx = parent[node_idx]
                parent[node_idx] = root
                node_idx = next_idx
            return root

        def union(parent, left, right):
            root_l = find(parent, left)
            root_r = find(parent, right)
            if root_l != root_r:
                parent[root_r] = root_l

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
                    union(bus_parent, pair[0], pair[1])

        root_to_nodes = {}
        for node in running_nodes:
            root_to_nodes.setdefault(find(bus_parent, node.idx), []).append(node)
        self.buses = []
        self.bus_dict = {}
        self.node_to_bus = {}
        for nodes in sorted(root_to_nodes.values(), key=lambda group: min(node.idx for node in group)):
            nodes.sort(key=lambda item: item.idx)
            bus = ACBus(nodes[0].idx, nodes)
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
        for dev in self.transformers:
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
                island = ACIsl(island_idx, True)
                root_to_island[root] = island
                self.islands.append(island)
            bus.isl = island.idx
            bus.isl_obj = island
            for node in bus.nodes:
                node.isl = island.idx
                node.isl_obj = island

        self.det_isl_alive_stat()
        # Cold-start state estimation does not reuse module-level topology templates.
        # Avoid the extra full-model scan needed to build the warm-cache template.

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
            isl.shunt_compensators = []

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
            bus.shunt_compensators = []
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
                print(f"        {trfm.idx} {trfm.name} i_node = {trfm.i_node} j_node = {trfm.j_node} r = {trfm.r} x = {trfm.x} b = {trfm.b} tap = {trfm.tap} shift = {trfm.shift}")

            print(f"    switches = {len(isl.switches)}:")
            for sw in isl.switches:
                print(f"        {sw.idx} {sw.name} i_node = {sw.i_node} j_node = {sw.j_node} status = {sw.status}")

            print(f"    zero_branches = {len(isl.zero_branches)}:")
            for zbr in isl.zero_branches:
                print(f"        {zbr.idx} {zbr.name} i_node = {zbr.i_node} j_node = {zbr.j_node}")

            print(f"    shunt_compensators = {len(isl.shunt_compensators)}:")
            for scp in isl.shunt_compensators:
                print(f"        {scp.idx} {scp.name} node = {scp.node} g = {scp.g_set} b = {scp.b_set}")

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

