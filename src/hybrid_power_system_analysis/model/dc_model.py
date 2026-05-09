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

class DCNode:
    def __init__(self, idx, vbase, voltage, run_stat=1):
        self.idx = idx
        self.vbase = vbase
        self.voltage = voltage
        self.run_stat = run_stat
        self.isl = None
        self.isl_obj = None
        self.v_set = 1.0
        self.v_gens = []
        self.v_dcdcs = []
        self.is_slack = False
        self.bus = None
        self.bus_obj = None


class DCBus:
    def __init__(self, idx, nodes=None):
        self.idx = idx
        self.nodes = list(nodes or [])
        ref = self.nodes[0] if self.nodes else None
        self.name = getattr(ref, "name", f"bus_{idx}")
        self.vbase = getattr(ref, "vbase", 0.0)
        self.voltage = getattr(ref, "voltage", 1.0)
        self.run_stat = 1
        self.isl = None
        self.isl_obj = None
        self.is_alive = False
        self.v_set = 1.0
        self.v_gens = []
        self.v_dcdcs = []
        self.is_slack = False
        self.generators = []
        self.loads = []
        self.branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.zero_branches = []

class DCBranch:
    def __init__(self, idx, i_node, j_node, r, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r = r
        self.run_stat = run_stat
        self.current = None
        self.i_p = None
        self.j_p = None
        self.i_node_obj = None
        self.j_node_obj = None

class DCLoad:
    def __init__(self, idx, node, pbase, pv0, pv1, pv2, run_stat=1):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.pbase = pbase
        self.pv0 = pv0
        self.pv1 = pv1
        self.pv2 = pv2

        self.p = None
        self.current = None
        self.node_obj = None

class DCGenerator:
    def __init__(self, idx, node, control_type, p_set, v_set, i_set, run_stat=1):
        self.idx = idx
        self.node = node
        self.run_stat = run_stat
        self.control_type = control_type
        self.p_set = p_set
        self.v_set = v_set
        self.i_set = i_set
        self.p = None
        self.current = None
        self.node_obj = None

class DCZeroBranch:
    def __init__(self, idx, i_node, j_node, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.run_stat = run_stat
        self.current = None
        self.p = None
        self.i_node_obj = None
        self.j_node_obj = None

class DCSwitch:
    def __init__(self, idx, i_node, j_node, status, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.status = status
        self.run_stat = run_stat
        self.current = None
        self.p = None
        self.i_node_obj = None
        self.j_node_obj = None

class DCBreak(DCSwitch):
    pass

class DCDCConverter:
    def __init__(self, idx, i_node, j_node, r1, r2, control_type, p_set, i_set, v_set, run_stat=1):
        self.idx = idx
        self.i_node = i_node
        self.j_node = j_node
        self.r1 = r1
        self.r2 = r2
        self.control_type = control_type
        self.p_set = p_set
        self.i_set = i_set
        self.v_set = v_set
        self.run_stat = run_stat
        self.i_p = None
        self.j_p = None
        self.i_c = None
        self.j_c = None
        self.i_node_obj = None
        self.j_node_obj = None



from efile_read import efile_factory_from_file, efile_factory_from_rows
from unit_system import normalize_model_named_units


class DCPowerNetwork:
    def __init__(self):

        self.nodes = []
        self.branches = []
        self.loads = []
        self.generators = []
        self.zero_branches = []
        self.switches = []
        self.breakers = []
        self.dcdc_converters = []
        self.buses = []

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
        node = DCNode(idx, vbase, voltage, run_stat)
        self.nodes.append(node)
        return node

    def add_branch(self, idx, i_node, j_node, r, run_stat=1):
        br = DCBranch(idx, i_node, j_node, r, run_stat)
        self.branches.append(br)
        return br

    def add_load(self, idx, node, pbase, pv0, pv1, pv2, run_stat=1):
        ld = DCLoad(idx, node, pbase, pv0, pv1, pv2, run_stat)
        self.loads.append(ld)
        return ld

    def add_generator(self, idx, node, control_type, p_set,v_set, i_set, run_stat=1):
        gen = DCGenerator(idx, node, control_type, p_set, v_set, i_set, run_stat)
        self.generators.append(gen)
        return gen

    def add_zero_branch(self, idx, i_node, j_node, run_stat=1):
        zb = DCZeroBranch(idx, i_node, j_node, run_stat)
        self.zero_branches.append(zb)
        return zb

    def add_switch(self, idx, i_node, j_node, status, run_stat=1):
        sw = DCSwitch(idx, i_node, j_node, status, run_stat)
        self.switches.append(sw)
        return sw

    def add_break(self, idx, i_node, j_node, status, run_stat=1):
        brk = DCBreak(idx, i_node, j_node, status, run_stat)
        self.breakers.append(brk)
        return brk

    def add_dcdc_converter(self, idx, i_node, j_node, r1, r2, control_type, p_set, i_set, v_set, run_stat=1):
        dc = DCDCConverter(idx, i_node, j_node, r1, r2, control_type, p_set, i_set, v_set, run_stat)
        self.dcdc_converters.append(dc)
        return dc

    def _load_from_model(self):
        self.p_base = normalize_model_named_units(self.model)
        self.p_base_kW = float(self.model.p_base_kW)
        self.u_scale = float(self.model.u_scale)
        self.p_scale = float(self.model.p_scale)
        self.i_scale = float(self.model.i_scale)
        self.branches = getattr(self.model, 'DCBranch', [])
        self.nodes = getattr(self.model, 'DCNode', [])
        self.generators = getattr(self.model, 'DCGenerator', [])
        self.loads = getattr(self.model, 'DCLoad', [])
        self.dcdc_converters = getattr(self.model, 'DCDCConverter', [])
        self.switches = getattr(self.model, 'DCSwitch', [])
        self.zero_branches = getattr(self.model, 'DCZeroBranch', [])
        self.breakers = [
            self._coerce_break(row)
            for row in getattr(self.model, 'DCBreak', [])
        ]
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

    def read_from_model(self, model):
        self.model = efile_factory_from_rows(model) if isinstance(model, dict) else model
        self._load_from_model()

    def read_from_file(self, file_name):
        self.source = str(file_name)
        self.read_from_model(efile_factory_from_file(file_name))

    @staticmethod
    def _coerce_break(row):
        brk = DCBreak(
            getattr(row, "idx", 0),
            getattr(row, "i_node", 0),
            getattr(row, "j_node", 0),
            getattr(row, "status", 1),
            getattr(row, "run_stat", 1),
        )
        brk.name = getattr(row, "name", f"brk_{brk.idx}")
        brk.p = getattr(row, "p", None)
        brk.current = getattr(row, "current", None)
        return brk

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

        for nb in self.branches:
            nb.i_node_obj = self.node_dict.get(nb.i_node, None)
            nb.j_node_obj = self.node_dict.get(nb.j_node, None)
            if nb.i_node_obj:
                nb.i_node_obj.branches.append(nb)
            if nb.j_node_obj:
                nb.j_node_obj.branches.append(nb)

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

        for cv in self.dcdc_converters:
            cv.i_node_obj = self.node_dict.get(cv.i_node, None)
            cv.j_node_obj = self.node_dict.get(cv.j_node, None)
            if cv.i_node_obj:
                cv.i_node_obj.dcdc_converters.append(cv)
            if cv.j_node_obj:
                cv.j_node_obj.dcdc_converters.append(cv)

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
            bus = DCBus(nodes[0].idx, nodes)
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

        # 检查发电机
        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            node = gen.node_obj
            if node is None or node.isl_obj is None:
                continue
            node.isl_obj.gens.append(gen)
            if gen.control_type == 'V':
                node.v_gens.append(gen)
                if node.bus_obj is not None:
                    node.bus_obj.v_gens.append(gen)
                node.isl_obj.v_gens.append(gen)

        # 检查 DC/DC 换流器
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
            if dcdc.control_type == 'V':
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
            if switch.i_node_obj.isl_obj and switch.j_node_obj.isl_obj and switch.i_node_obj.isl_obj ==  switch.j_node_obj.isl_obj:
                switch.i_node_obj.isl_obj.switches.append(switch)

        for br in self.branches:
            if br.run_stat == 0:
                continue
            if br.i_node_obj is None or br.j_node_obj is None:
                continue
            if br.i_node_obj.isl_obj and br.j_node_obj.isl_obj and br.i_node_obj.isl_obj ==  br.j_node_obj.isl_obj:
                br.i_node_obj.isl_obj.branches.append(br)

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
            if brk.i_node_obj.isl_obj and brk.j_node_obj.isl_obj and brk.i_node_obj.isl_obj ==  brk.j_node_obj.isl_obj:
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

        for br in self.branches:
            if br.i_node_obj is None or br.j_node_obj is None or br.run_stat == 0:
                br.is_alive = False
                continue
            br.is_alive = br.i_node_obj.is_alive and br.j_node_obj.is_alive

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

        for conv in self.dcdc_converters:
            if conv.i_node_obj is None or conv.j_node_obj is None or conv.run_stat == 0:
                conv.is_alive = False
                continue
            conv.is_alive = conv.i_node_obj.is_alive and conv.j_node_obj.is_alive

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
            for brk in getattr(isl, 'breakers', []):
                print(f"        {brk.idx} {brk.name} i_node = {brk.i_node} j_node = {brk.j_node} status = {brk.status}")

            print(f"    dcdc_converters = {len(isl.dcdc_converters)}:")
            for dcc in isl.dcdc_converters:
                print(f"        {dcc.idx} {dcc.name} i_node = {dcc.i_node} j_node = {dcc.j_node} r1 = {dcc.r1} r2 = {dcc.r2} control_type = {dcc.control_type}")

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
        for zb in self.zero_branches:
            if zb.run_stat == 0:
                continue
            check_node(zb.i_node, 'ZeroBranch', zb)
            check_node(zb.j_node, 'ZeroBranch', zb)
        for sw in self.switches:
            if sw.run_stat == 0:
                continue
            check_node(sw.i_node, 'Switch', sw)
            check_node(sw.j_node, 'Switch', sw)
        for brk in self.breakers:
            if brk.run_stat == 0 or brk.status == 0:
                continue
            check_node(brk.i_node, 'Break', brk)
            check_node(brk.j_node, 'Break', brk)
        for ld in self.loads:
            if ld.run_stat == 0:
                continue
            check_node(ld.node, 'Load', ld)
        for gen in self.generators:
            if gen.run_stat == 0:
                continue
            check_node(gen.node, 'Generator', gen)
        for dcdc in self.dcdc_converters:
            if dcdc.run_stat == 0:
                continue
            check_node(dcdc.i_node, 'DCDCConverter', dcdc)
            check_node(dcdc.j_node, 'DCDCConverter', dcdc)

        # 检查节点悬空
        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {node.name} 未关联任何设备")
            if node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {node.name} 单端悬空，请检查！")

        # 检查每个岛屿
        for isl in self.islands:
            # 1. 电压基值一致性
            vbase_set = {int(bus.vbase*1000) for bus in isl.buses}
            if len(vbase_set) > 1:
                str_info = f"岛屿 {isl.idx} 内节点电压基值不一致:"
                for vbase in vbase_set:
                    str_info += f" {vbase / 1000.0 :.2f}"
                errors.append(str_info)

            # 2. 电压控制源唯一性（松弛节点或定V发电机）
            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V节点:"
                for node in isl.slack_nodes:
                    str_info += f" {node.name}"
                warns.append(str_info)

            if len(isl.v_dcdcs) > 1:
                str_info = f"岛屿 {isl.idx} 存在多个定V变流器:"
                for dcdcs in isl.v_dcdcs:
                    str_info += f" {dcdcs.name}"
                warns.append(str_info)

            if len(isl.slack_nodes) +  len(isl.v_dcdcs) == 0:
                errors.append(f"岛屿 {isl.idx} , 内无电压控制源（定V节点或定V变流器）")

            if len(isl.slack_nodes) > 1:
                str_info = f"岛屿 {isl.idx} , 内有多个电压控制源（定V节点或定V变流器）:"
                for node in isl.slack_nodes:
                    str_info += f" node-{node.name}"
                errors.append(str_info)


        # 检查松弛节点与定V发电机的一致性
        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if len(node.v_gens)  + len(node.v_dcdcs) <= 1:
                continue

            if len(node.v_gens) + len(node.v_dcdcs) >= 2:
                errors.append(
                    f"松弛节点 {node.idx} 上的定V发电机与定V变流器数量之和超过1，请检查拓扑！")

            node.v_set = 0.0
            if len(node.v_gens) >= 1:
                node.v_set = node.v_gens[0].v_set

            if len(node.v_dcdcs) > 1:
                node.v_set = node.v_dcdcs[0].v_set

            node.is_slack = True

        return warns, errors
