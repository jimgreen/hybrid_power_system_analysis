from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import List, Tuple

MODEL_DIR = Path(__file__).resolve().parent
for path in (MODEL_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_model import ACACConverter, ACAC_CONTROL_TYPES, ACPowerNetwork
from dc_model import DCPowerNetwork
from hybrid_array_model import DCAC_CONTROL_PARSE_CODE, build_hybrid_ppc_from_e_file

class DCACConverter:
    def __init__(
        self,
        idx,
        ac_node,
        dc_node,
        r1,
        r2,
        control_type,
        p_ac_set,
        q_ac_set,
        v_ac_set,
        v_dc_set,
        run_stat=1,
    ):
        self.idx = idx
        self.ac_node = ac_node
        self.dc_node = dc_node
        self.r1 = r1
        self.r2 = r2
        self.control_type = control_type
        self.p_ac_set = p_ac_set
        self.q_ac_set = q_ac_set
        self.v_ac_set = v_ac_set
        self.v_dc_set = v_dc_set
        self.run_stat = run_stat
        self.dc_p = None
        self.ac_p = None
        self.ac_q = None
        self.dc_i = None
        self.ac_i = None
        self.ac_node_obj = None
        self.dc_node_obj = None

class HybridIsland:
    def __init__(self, idx: int):
        self.idx = idx
        self.is_alive = False
        self.ac_islands = []
        self.dc_islands = []
        self.ac_nodes = []
        self.dc_nodes = []
        self.dcac_converters = []
        self.dcdc_converters = []
        self.acac_converters = []

    def add_ac_island(self, island):
        self.ac_islands.append(island)
        island.hybrid_isl = self.idx
        island.hybrid_isl_obj = self
        self.ac_nodes.extend(island.buses)
        self.is_alive = self.is_alive or island.is_alive

    def add_dc_island(self, island):
        self.dc_islands.append(island)
        island.hybrid_isl = self.idx
        island.hybrid_isl_obj = self
        self.dc_nodes.extend(island.buses)
        self.is_alive = self.is_alive or island.is_alive

    def add_dcac_converter(self, conv):
        self.dcac_converters.append(conv)
        conv.hybrid_isl = self.idx
        conv.hybrid_isl_obj = self

    def add_dcdc_converter(self, conv):
        self.dcdc_converters.append(conv)
        conv.hybrid_isl = self.idx
        conv.hybrid_isl_obj = self

    def add_acac_converter(self, conv):
        self.acac_converters.append(conv)
        conv.hybrid_isl = self.idx
        conv.hybrid_isl_obj = self


@dataclass
class HybridPowerNetwork:
    ac: ACPowerNetwork
    dc: DCPowerNetwork
    dcac_converters: List
    acac_converters: List
    hybrid_islands: List[HybridIsland] = field(default_factory=list)

    def __post_init__(self):
        self.ac.acac_converters = self.acac_converters

    @classmethod
    def read_from_file(cls, file_name) -> "HybridPowerNetwork":
        network, _ppc = build_hybrid_ppc_from_e_file(file_name)
        return network

    @property
    def total_nodes(self) -> int:
        return len(self.ac.nodes) + len(self.dc.nodes)

    def topo(self):
        if self.ac.nodes:
            self.ac.topo()
        if self.dc.nodes:
            self.dc.topo()
        self._build_hybrid_topo()

    def _build_hybrid_topo(self):
        ac_islands = self.ac.islands
        dc_islands = self.dc.islands
        n_ac_islands = len(ac_islands)
        total_islands = n_ac_islands + len(dc_islands)
        for pos, island in enumerate(ac_islands):
            island._hybrid_pos = pos
            island.hybrid_isl = 0
            island.hybrid_isl_obj = None
        for offset, island in enumerate(dc_islands, start=n_ac_islands):
            island._hybrid_pos = offset
            island.hybrid_isl = 0
            island.hybrid_isl_obj = None
        for conv in self.dc.dcdc_converters:
            conv.hybrid_isl = 0
            conv.hybrid_isl_obj = None
        for conv in self.dcac_converters:
            conv.hybrid_isl = 0
            conv.hybrid_isl_obj = None
        for conv in self.acac_converters:
            conv.hybrid_isl = 0
            conv.hybrid_isl_obj = None

        parent = list(range(total_islands))

        def find(pos):
            root = pos
            while parent[root] != root:
                root = parent[root]
            while parent[pos] != pos:
                pos, parent[pos] = parent[pos], root
            return root

        def union(left, right):
            if left is None or right is None:
                return
            left_root = find(left._hybrid_pos)
            right_root = find(right._hybrid_pos)
            if left_root != right_root:
                parent[right_root] = left_root

        for conv in self.dcac_converters:
            ac_node = self.ac.node_dict.get(conv.ac_node)
            dc_node = self.dc.node_dict.get(conv.dc_node)
            conv.ac_node_obj = ac_node
            conv.dc_node_obj = dc_node
            conv.ac_isl_obj = None if ac_node is None else ac_node.isl_obj
            conv.dc_isl_obj = None if dc_node is None else dc_node.isl_obj
            conv.is_alive = (
                conv.run_stat == 1
                and ac_node is not None
                and dc_node is not None
                and ac_node.is_alive
                and dc_node.is_alive
            )
            if conv.is_alive:
                union(conv.ac_isl_obj, conv.dc_isl_obj)

        for conv in self.acac_converters:
            i_node = self.ac.node_dict.get(conv.i_node)
            j_node = self.ac.node_dict.get(conv.j_node)
            conv.i_node_obj = i_node
            conv.j_node_obj = j_node
            conv.i_isl_obj = None if i_node is None else i_node.isl_obj
            conv.j_isl_obj = None if j_node is None else j_node.isl_obj
            conv.is_alive = (
                conv.run_stat == 1
                and i_node is not None
                and j_node is not None
                and i_node.is_alive
                and j_node.is_alive
            )
            if conv.is_alive:
                union(conv.i_isl_obj, conv.j_isl_obj)

        for conv in self.dc.dcdc_converters:
            if not getattr(conv, "is_alive", False):
                continue
            union(
                None if getattr(conv, "i_node_obj", None) is None else conv.i_node_obj.isl_obj,
                None if getattr(conv, "j_node_obj", None) is None else conv.j_node_obj.isl_obj,
            )

        grouped = {}
        for island in ac_islands:
            root = find(island._hybrid_pos)
            grouped.setdefault(root, []).append((True, island))
        for island in dc_islands:
            root = find(island._hybrid_pos)
            grouped.setdefault(root, []).append((False, island))

        self.hybrid_islands = []
        island_by_root = {}
        for idx, (root, islands) in enumerate(grouped.items(), start=1):
            hybrid_island = HybridIsland(idx)
            self.hybrid_islands.append(hybrid_island)
            island_by_root[root] = hybrid_island
            for is_ac_island, island in islands:
                if is_ac_island:
                    hybrid_island.add_ac_island(island)
                else:
                    hybrid_island.add_dc_island(island)

        for conv in self.dcac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            ac_island = getattr(conv, "ac_isl_obj", None)
            dc_island = getattr(conv, "dc_isl_obj", None)
            if ac_island is None or dc_island is None:
                continue
            hybrid_island = island_by_root[find(ac_island._hybrid_pos)]
            hybrid_island.add_dcac_converter(conv)

        for conv in self.dc.dcdc_converters:
            if not getattr(conv, "is_alive", False) or conv.i_node_obj is None or conv.j_node_obj is None:
                continue
            if conv.i_node_obj.isl_obj is None or conv.j_node_obj.isl_obj is None:
                continue
            hybrid_island = island_by_root[find(conv.i_node_obj.isl_obj._hybrid_pos)]
            hybrid_island.add_dcdc_converter(conv)

        for conv in self.acac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            i_island = getattr(conv, "i_isl_obj", None)
            j_island = getattr(conv, "j_isl_obj", None)
            if i_island is None or j_island is None:
                continue
            hybrid_island = island_by_root[find(i_island._hybrid_pos)]
            hybrid_island.add_acac_converter(conv)

    def print_hybrid_isl_info(self):
        for island in self.hybrid_islands:
            print(f"Hybrid island {island.idx} is_alive = {island.is_alive}")
            print(
                f"    ac_islands={len(island.ac_islands)} "
                f"dc_islands={len(island.dc_islands)} "
                f"ac_nodes={len(island.ac_nodes)} "
                f"dc_nodes={len(island.dc_nodes)}"
            )
            print(
                f"    dcac_converters={len(island.dcac_converters)} "
                f"dcdc_converters={len(island.dcdc_converters)} "
                f"acac_converters={len(island.acac_converters)}"
            )

    @staticmethod
    def _add_ref(refs, node_idx):
        refs[node_idx] = refs.get(node_idx, 0) + 1

    def _external_node_refs(self):
        ac_refs = {}
        dc_refs = {}
        for conv in self.dcac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            self._add_ref(ac_refs, conv.ac_node)
            self._add_ref(dc_refs, conv.dc_node)
        for conv in self.acac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            self._add_ref(ac_refs, conv.i_node)
            self._add_ref(ac_refs, conv.j_node)
        return ac_refs, dc_refs

    def prepare(self, verbose: bool = True) -> Tuple[List[str], List[str], List[str], List[str]]:
        self.topo()
        if verbose and self.ac.nodes:
            self.ac.print_isl_info()
        if verbose and self.dc.nodes:
            self.dc.print_isl_info()
        if verbose:
            self.print_hybrid_isl_info()
        return [], [], [], []

    def check_topology(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        if not self.hybrid_islands:
            self.topo()
        ac_refs, dc_refs = self._external_node_refs()
        ac_warnings, ac_errors = self._check_ac_topo(ac_refs) if self.ac.nodes else ([], [])
        dc_warnings, dc_errors = self._check_dc_topo(dc_refs) if self.dc.nodes else ([], [])
        dc_errors.extend(self._check_dcac_topo())
        dc_errors.extend(self._check_acac_topo())
        return ac_warnings, ac_errors, dc_warnings, dc_errors

    def _check_ac_topo(self, extra_node_refs=None):
        warnings, errors = self.ac.check_topo()
        return self._filter_node_ref_messages(self.ac, warnings, errors, extra_node_refs or {}, self._ac_node_ref_count())

    def _check_dc_topo(self, extra_node_refs=None):
        warnings, errors = self.dc.check_topo()
        return self._filter_node_ref_messages(self.dc, warnings, errors, extra_node_refs or {}, self._dc_node_ref_count())

    @staticmethod
    def _filter_node_ref_messages(grid, warnings, errors, extra_refs, base_counts):
        warning_set = set(warnings)
        error_set = set(errors)
        for node in grid.nodes:
            if node.run_stat != 1:
                continue
            count = base_counts.get(node.idx, 0) + extra_refs.get(node.idx, 0)
            no_ref_msg = f"节点 {node.idx} {node.name} 未关联任何设备"
            single_ref_msg = f"节点 {node.idx} {node.name} 单端悬空，请检查！"
            if count > 0:
                error_set.discard(no_ref_msg)
            if count != 1:
                warning_set.discard(single_ref_msg)
        return list(warning_set), list(error_set)

    def _ac_node_ref_count(self):
        counts = {node.idx: 0 for node in self.ac.nodes}

        def add(node_idx):
            node = self.ac.node_dict.get(node_idx)
            if node is not None and node.run_stat == 1:
                counts[node_idx] += 1

        for br in self.ac.branches:
            if br.run_stat:
                add(br.i_node)
                add(br.j_node)
        for tr in self.ac.transformers:
            if tr.run_stat:
                add(tr.i_node)
                add(tr.j_node)
        for zb in self.ac.zero_branches:
            if zb.run_stat:
                add(zb.i_node)
                add(zb.j_node)
        for sw in self.ac.switches:
            if sw.run_stat:
                add(sw.i_node)
                add(sw.j_node)
        for load in self.ac.loads:
            if load.run_stat:
                add(load.node)
        for gen in self.ac.generators:
            if gen.run_stat:
                add(gen.node)
        for shunt in self.ac.shunt_compensators:
            if shunt.run_stat:
                add(shunt.node)
        return counts

    def _dc_node_ref_count(self):
        counts = {node.idx: 0 for node in self.dc.nodes}

        def add(node_idx):
            node = self.dc.node_dict.get(node_idx)
            if node is not None and node.run_stat == 1:
                counts[node_idx] += 1

        for br in self.dc.branches:
            if br.run_stat:
                add(br.i_node)
                add(br.j_node)
        for zb in self.dc.zero_branches:
            if zb.run_stat:
                add(zb.i_node)
                add(zb.j_node)
        for sw in self.dc.switches:
            if sw.run_stat:
                add(sw.i_node)
                add(sw.j_node)
        for load in self.dc.loads:
            if load.run_stat:
                add(load.node)
        for gen in self.dc.generators:
            if gen.run_stat:
                add(gen.node)
        for dcdc in self.dc.dcdc_converters:
            if dcdc.run_stat:
                add(dcdc.i_node)
                add(dcdc.j_node)
        return counts

    def _check_dcac_topo(self) -> List[str]:
        errors = []
        if not self.dcac_converters:
            return errors
        if not self.ac.nodes:
            return ["存在 DCACConverter，但文件中没有 ACNode"]
        if not self.dc.nodes:
            return ["存在 DCACConverter，但文件中没有 DCNode"]

        for conv in self.dcac_converters:
            if conv.run_stat != 1:
                continue
            if conv.ac_node not in self.ac.node_dict:
                errors.append(f"DCACConverter[{conv.idx}] {getattr(conv, 'name', '')} 引用的 AC 节点 {conv.ac_node} 不存在")
            if conv.dc_node not in self.dc.node_dict:
                errors.append(f"DCACConverter[{conv.idx}] {getattr(conv, 'name', '')} 引用的 DC 节点 {conv.dc_node} 不存在")
            if conv.ac_node not in self.ac.node_dict or conv.dc_node not in self.dc.node_dict:
                continue
            if str(conv.control_type).upper() not in DCAC_CONTROL_PARSE_CODE:
                errors.append(f"DCACConverter[{conv.idx}] {getattr(conv, 'name', '')} 控制模式 {conv.control_type} 不支持")
        return errors

    def _check_acac_topo(self) -> List[str]:
        errors = []
        if not self.acac_converters:
            return errors
        if not self.ac.nodes:
            return ["存在 ACACConverter，但文件中没有 ACNode"]

        for conv in self.acac_converters:
            if conv.run_stat != 1:
                continue
            if conv.i_node not in self.ac.node_dict:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 引用的 AC 节点 {conv.i_node} 不存在")
            if conv.j_node not in self.ac.node_dict:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 引用的 AC 节点 {conv.j_node} 不存在")
            if conv.i_node not in self.ac.node_dict or conv.j_node not in self.ac.node_dict:
                continue
            if conv.i_node == conv.j_node:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 两端不能连接同一个 AC 节点")
            if str(conv.control_type).upper() not in ACAC_CONTROL_TYPES:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 控制模式 {conv.control_type} 不支持")
        return errors
