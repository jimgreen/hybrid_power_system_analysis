import argparse
import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
LFCORE_DIR = Path(__file__).resolve().parent
if str(LFCORE_DIR) not in sys.path:
    sys.path.insert(0, str(LFCORE_DIR))

from efile_read import efile_factory_from_file_cached
from ac_lf import ACPowerFlowCalc
from dc_lf import DCPowerFlowCalc
from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from unit_system import normalize_model_named_units


DEFAULT_HYBRID_EFILE = ROOT_DIR / "data" / "hybrid" / "hybrid_net_40.e"
ACAC_CONTROL_TYPES = {"PQQ", "PVQ", "PQV", "PVV"}


class _ACIsland:
    def __init__(self, idx: int):
        self.idx = idx
        self.reset()

    def reset(self):
        self.is_alive = False
        self.nodes = []
        self.gens = []
        self.loads = []
        self.branches = []
        self.zero_branches = []
        self.switches = []
        self.transformers = []
        self.shunt_compensators = []
        self.slack_nodes = []
        self.v_gens = []


class _DCIsland:
    def __init__(self, idx: int):
        self.idx = idx
        self.reset()

    def reset(self):
        self.is_alive = False
        self.nodes = []
        self.gens = []
        self.loads = []
        self.branches = []
        self.zero_branches = []
        self.switches = []
        self.dcdc_converters = []
        self.slack_nodes = []
        self.v_gens = []
        self.v_dcdcs = []


class _GridBase:
    @staticmethod
    def _is_running(dev) -> bool:
        return dev.run_stat == 1

    @staticmethod
    def _same_live_island(dev) -> bool:
        return (
            dev.i_node_obj is not None
            and dev.j_node_obj is not None
            and dev.i_node_obj.isl_obj is not None
            and dev.i_node_obj.isl_obj == dev.j_node_obj.isl_obj
        )

    @staticmethod
    def _name(dev) -> str:
        return getattr(dev, "name", str(getattr(dev, "idx", "")))


class HybridACGrid(_GridBase):
    def __init__(self, model):
        self.model = model
        self.nodes = getattr(model, "ACNode", [])
        self.branches = getattr(model, "ACBranch", [])
        self.loads = getattr(model, "ACLoad", [])
        self.generators = getattr(model, "ACGenerator", [])
        self.zero_branches = getattr(model, "ACZeroBranch", [])
        self.switches = getattr(model, "ACSwitch", [])
        self.transformers = getattr(model, "ACTransformer", [])
        self.shunt_compensators = getattr(model, "ACShuntCompensator", [])
        self.islands = []
        self.node_dict = {}
        self.switch_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branch_dict = {}
        self.branch_dict = {}
        self.transformer_dict = {}
        self.shunt_compensator_dict = {}

    def format_assoc(self):
        """Rebuild AC device dictionaries and attach each device to its terminal nodes."""
        self.node_dict = {node.idx: node for node in self.nodes}
        self.switch_dict = {sw.idx: sw for sw in self.switches}
        self.load_dict = {ld.idx: ld for ld in self.loads}
        self.generator_dict = {gen.idx: gen for gen in self.generators}
        self.zero_branch_dict = {zbr.idx: zbr for zbr in self.zero_branches}
        self.branch_dict = {br.idx: br for br in self.branches}
        self.transformer_dict = {tr.idx: tr for tr in self.transformers}
        self.shunt_compensator_dict = {sc.idx: sc for sc in self.shunt_compensators}

        for node in self.nodes:
            node.v_gens = []
            node.is_alive = False

        node_get = self.node_dict.get
        for gen in self.generators:
            gen.node_obj = node_get(gen.node)
        for ld in self.loads:
            ld.node_obj = node_get(ld.node)
        for sc in self.shunt_compensators:
            sc.node_obj = node_get(sc.node)
        for br in self.branches:
            br.i_node_obj = node_get(br.i_node)
            br.j_node_obj = node_get(br.j_node)
        for tr in self.transformers:
            tr.i_node_obj = node_get(tr.i_node)
            tr.j_node_obj = node_get(tr.j_node)
        for sw in self.switches:
            sw.i_node_obj = node_get(sw.i_node)
            sw.j_node_obj = node_get(sw.j_node)
        for zbr in self.zero_branches:
            zbr.i_node_obj = node_get(zbr.i_node)
            zbr.j_node_obj = node_get(zbr.j_node)

    def topo(self):
        """Find AC connectivity islands using running branches, transformers and closed switches."""
        self.format_assoc()
        self.islands = []
        for node in self.nodes:
            node.isl = 0
            node.isl_obj = None

        running_nodes = [node for node in self.nodes if node.run_stat == 1]
        adj = {node: [] for node in running_nodes}

        def add_edge(dev):
            if (
                dev.run_stat == 1
                and dev.i_node_obj in adj
                and dev.j_node_obj in adj
                and dev.i_node_obj != dev.j_node_obj
            ):
                adj[dev.i_node_obj].append(dev.j_node_obj)
                adj[dev.j_node_obj].append(dev.i_node_obj)

        for dev in self.branches:
            add_edge(dev)
        for dev in self.transformers:
            add_edge(dev)
        for dev in self.zero_branches:
            add_edge(dev)
        for dev in self.switches:
            if dev.status == 1:
                add_edge(dev)

        island_idx = 0
        for node in running_nodes:
            if node.isl != 0:
                continue
            island_idx += 1
            island = _ACIsland(island_idx)
            self.islands.append(island)
            node.isl = island_idx
            node.isl_obj = island
            stack = [node]
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt.isl == 0:
                        nxt.isl = island_idx
                        nxt.isl_obj = island
                        stack.append(nxt)

        self.det_isl_alive_stat()

    def det_isl_alive_stat(self):
        """Populate live AC islands and mark devices that participate in load flow."""
        for island in self.islands:
            island.reset()

        for node in self.nodes:
            node.v_gens = []
            if node.isl_obj is not None:
                node.isl_obj.nodes.append(node)

        for gen in self.generators:
            if gen.run_stat != 1 or gen.node_obj is None or gen.node_obj.isl_obj is None:
                continue
            island = gen.node_obj.isl_obj
            island.gens.append(gen)
            if gen.control_type in ("V", "SLACK", "PH"):
                island.is_alive = True
                gen.node_obj.v_gens.append(gen)
                island.v_gens.append(gen)
                if gen.node_obj not in island.slack_nodes:
                    island.slack_nodes.append(gen.node_obj)
            elif gen.control_type == "PV":
                gen.node_obj.v_gens.append(gen)
                island.v_gens.append(gen)

        for ld in self.loads:
            if ld.run_stat == 1 and ld.node_obj is not None and ld.node_obj.isl_obj is not None:
                ld.node_obj.isl_obj.loads.append(ld)
        for sc in self.shunt_compensators:
            if sc.run_stat == 1 and sc.node_obj is not None and sc.node_obj.isl_obj is not None:
                sc.node_obj.isl_obj.shunt_compensators.append(sc)
        for sw in self.switches:
            if sw.run_stat == 1 and sw.status == 1 and self._same_live_island(sw):
                sw.i_node_obj.isl_obj.switches.append(sw)
        for br in self.branches:
            if br.run_stat == 1 and self._same_live_island(br):
                br.i_node_obj.isl_obj.branches.append(br)
        for tr in self.transformers:
            if tr.run_stat == 1 and self._same_live_island(tr):
                tr.i_node_obj.isl_obj.transformers.append(tr)
        for zbr in self.zero_branches:
            if zbr.run_stat == 1 and self._same_live_island(zbr):
                zbr.i_node_obj.isl_obj.zero_branches.append(zbr)

        for island in self.islands:
            island.is_alive = len(island.slack_nodes) >= 1

        for node in self.nodes:
            node.is_alive = node.isl_obj is not None and node.isl_obj.is_alive
        for ld in self.loads:
            ld.is_alive = ld.node_obj is not None and ld.run_stat == 1 and ld.node_obj.is_alive
        for gen in self.generators:
            gen.is_alive = gen.node_obj is not None and gen.run_stat == 1 and gen.node_obj.is_alive
        for sc in self.shunt_compensators:
            sc.is_alive = sc.node_obj is not None and sc.run_stat == 1 and sc.node_obj.is_alive
        for dev in [*self.branches, *self.transformers, *self.zero_branches]:
            dev.is_alive = (
                dev.run_stat == 1
                and dev.i_node_obj is not None
                and dev.j_node_obj is not None
                and dev.i_node_obj.is_alive
                and dev.j_node_obj.is_alive
            )
        for sw in self.switches:
            sw.is_alive = (
                sw.run_stat == 1
                and sw.status == 1
                and sw.i_node_obj is not None
                and sw.j_node_obj is not None
                and sw.i_node_obj.is_alive
                and sw.j_node_obj.is_alive
            )

    def print_isl_info(self):
        for island in self.islands:
            print(f"AC island {island.idx} is_alive = {island.is_alive}")
            print(f"    nodes = {len(island.nodes)}")
            print(f"    gens = {len(island.gens)}")
            print(f"    loads = {len(island.loads)}")
            print(f"    branches = {len(island.branches)}")
            print(f"    transformers = {len(island.transformers)}")
            print(f"    switches = {len(island.switches)}")
            print(f"    zero_branches = {len(island.zero_branches)}")
            print(f"    shunt_compensators = {len(island.shunt_compensators)}")

    def _slack_reference(self, node):
        """Return the fixed phasor imposed by the running slack generator on a node."""
        for gen in getattr(node, "v_gens", []):
            if gen.run_stat == 1 and str(gen.control_type).upper() in {"V", "SLACK", "PH"}:
                return float(gen.v_set), float(getattr(node, "angle", 0.0) or 0.0)
        return float(getattr(node, "voltage", 1.0) or 1.0), float(getattr(node, "angle", 0.0) or 0.0)

    @staticmethod
    def _angle_close(left: float, right: float, tol: float = 1e-10) -> bool:
        diff = (left - right + np.pi) % (2.0 * np.pi) - np.pi
        return abs(diff) <= tol

    def _slack_nodes_are_redundant_zero_ties(self, island) -> bool:
        """Allow equal fixed-voltage slack nodes only when ideal ties make them one node."""
        slack_nodes = list(getattr(island, "slack_nodes", []))
        if len(slack_nodes) <= 1:
            return True

        parent = {node.idx: node.idx for node in island.nodes}

        def find(node_idx):
            root = node_idx
            while parent[root] != root:
                root = parent[root]
            while parent[node_idx] != node_idx:
                node_idx, parent[node_idx] = parent[node_idx], root
            return root

        def union(left, right):
            if left not in parent or right not in parent:
                return
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for dev in island.zero_branches:
            if dev.run_stat == 1:
                union(dev.i_node, dev.j_node)
        for dev in island.switches:
            if dev.run_stat == 1 and getattr(dev, "status", 0) == 1:
                union(dev.i_node, dev.j_node)

        first_root = find(slack_nodes[0].idx)
        if any(find(node.idx) != first_root for node in slack_nodes[1:]):
            return False

        ref_voltage, ref_angle = self._slack_reference(slack_nodes[0])
        for node in slack_nodes[1:]:
            voltage, angle = self._slack_reference(node)
            if abs(voltage - ref_voltage) > 1e-10 or not self._angle_close(angle, ref_angle):
                return False
        return True

    def check_topo(self, extra_node_refs=None):
        if not self.islands:
            self.topo()
        errors = []
        warns = []
        node_ref_count = {node.idx: 0 for node in self.nodes}
        for node_idx, count in (extra_node_refs or {}).items():
            if node_idx in node_ref_count:
                node_ref_count[node_idx] += count

        def check_node(node_idx, dev_type, dev):
            if node_idx not in self.node_dict:
                errors.append(f"设备 {dev_type}[{dev.idx}] {self._name(dev)} 引用的节点 {node_idx} 不存在")
            elif self.node_dict[node_idx].run_stat == 1:
                node_ref_count[node_idx] += 1

        for br in self.branches:
            if br.run_stat == 1:
                check_node(br.i_node, "ACBranch", br)
                check_node(br.j_node, "ACBranch", br)
        for tr in self.transformers:
            if tr.run_stat == 1:
                check_node(tr.i_node, "ACTransformer", tr)
                check_node(tr.j_node, "ACTransformer", tr)
        for zbr in self.zero_branches:
            if zbr.run_stat == 1:
                check_node(zbr.i_node, "ACZeroBranch", zbr)
                check_node(zbr.j_node, "ACZeroBranch", zbr)
        for sw in self.switches:
            if sw.run_stat == 1:
                check_node(sw.i_node, "ACSwitch", sw)
                check_node(sw.j_node, "ACSwitch", sw)
        for ld in self.loads:
            if ld.run_stat == 1:
                check_node(ld.node, "ACLoad", ld)
        for gen in self.generators:
            if gen.run_stat == 1:
                check_node(gen.node, "ACGenerator", gen)
        for sc in self.shunt_compensators:
            if sc.run_stat == 1:
                check_node(sc.node, "ACShuntCompensator", sc)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {self._name(node)} 未关联任何设备")
            elif node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {self._name(node)} 单端悬空，请检查！")

        for dev_type, devices in (
            ("支路", self.branches),
            ("开关", self.switches),
            ("零阻抗支路", self.zero_branches),
        ):
            for dev in devices:
                if dev.run_stat != 1:
                    continue
                if dev.i_node_obj is None or dev.j_node_obj is None:
                    continue
                if dev.i_node_obj.run_stat != 1 or dev.j_node_obj.run_stat != 1:
                    continue
                if abs(dev.i_node_obj.vbase - dev.j_node_obj.vbase) > 0.1:
                    errors.append(
                        f"{dev_type} {dev.idx} {self._name(dev)} 两端节点的电压基值不同:"
                        f"{dev.i_node_obj.vbase} {dev.j_node_obj.vbase}"
                    )

        for island in self.islands:
            if len(island.slack_nodes) > 1:
                names = " ".join(self._name(node) for node in island.slack_nodes)
                if self._slack_nodes_are_redundant_zero_ties(island):
                    warns.append(f"岛屿 {island.idx} 存在多个零阻抗等值平衡节点，按冗余参考处理: {names}")
                else:
                    errors.append(f"岛屿 {island.idx} 存在多个平衡节点: {names}")
            if len(island.slack_nodes) == 0:
                warns.append(f"岛屿 {island.idx} , 无平衡节点，跳过潮流计算")

        return warns, errors


class HybridDCGrid(_GridBase):
    def __init__(self, model):
        self.model = model
        self.nodes = getattr(model, "DCNode", [])
        self.branches = getattr(model, "DCBranch", [])
        self.loads = getattr(model, "DCLoad", [])
        self.generators = getattr(model, "DCGenerator", [])
        self.zero_branches = getattr(model, "DCZeroBranch", [])
        self.switches = getattr(model, "DCSwitch", [])
        self.dcdc_converters = getattr(model, "DCDCConverter", [])
        self.islands = []
        self.node_dict = {}
        self.switch_dict = {}
        self.load_dict = {}
        self.generator_dict = {}
        self.zero_branch_dict = {}
        self.branch_dict = {}
        self.dcdc_converter_dict = {}

    def format_assoc(self):
        """Rebuild DC device dictionaries and attach each device to its terminal nodes."""
        self.node_dict = {node.idx: node for node in self.nodes}
        self.switch_dict = {sw.idx: sw for sw in self.switches}
        self.load_dict = {ld.idx: ld for ld in self.loads}
        self.generator_dict = {gen.idx: gen for gen in self.generators}
        self.zero_branch_dict = {zbr.idx: zbr for zbr in self.zero_branches}
        self.branch_dict = {br.idx: br for br in self.branches}
        self.dcdc_converter_dict = {conv.idx: conv for conv in self.dcdc_converters}

        for node in self.nodes:
            node.v_gens = []
            node.v_dcdcs = []
            node.v_set = 0.0
            node.is_slack = False
            node.is_alive = False

        node_get = self.node_dict.get
        for gen in self.generators:
            gen.node_obj = node_get(gen.node)
        for ld in self.loads:
            ld.node_obj = node_get(ld.node)
        for br in self.branches:
            br.i_node_obj = node_get(br.i_node)
            br.j_node_obj = node_get(br.j_node)
        for sw in self.switches:
            sw.i_node_obj = node_get(sw.i_node)
            sw.j_node_obj = node_get(sw.j_node)
        for zbr in self.zero_branches:
            zbr.i_node_obj = node_get(zbr.i_node)
            zbr.j_node_obj = node_get(zbr.j_node)
        for conv in self.dcdc_converters:
            conv.i_node_obj = node_get(conv.i_node)
            conv.j_node_obj = node_get(conv.j_node)

    def topo(self):
        """Find DC connectivity islands using running resistive/zero branches and closed switches."""
        self.format_assoc()
        self.islands = []
        for node in self.nodes:
            node.isl = 0
            node.isl_obj = None

        running_nodes = [node for node in self.nodes if node.run_stat == 1]
        adj = {node: [] for node in running_nodes}

        def add_edge(dev):
            if (
                dev.run_stat == 1
                and dev.i_node_obj in adj
                and dev.j_node_obj in adj
                and dev.i_node_obj != dev.j_node_obj
            ):
                adj[dev.i_node_obj].append(dev.j_node_obj)
                adj[dev.j_node_obj].append(dev.i_node_obj)

        for dev in self.branches:
            add_edge(dev)
        for dev in self.zero_branches:
            add_edge(dev)
        for dev in self.switches:
            if dev.status == 1:
                add_edge(dev)

        island_idx = 0
        for node in running_nodes:
            if node.isl != 0:
                continue
            island_idx += 1
            island = _DCIsland(island_idx)
            self.islands.append(island)
            node.isl = island_idx
            node.isl_obj = island
            stack = [node]
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt.isl == 0:
                        nxt.isl = island_idx
                        nxt.isl_obj = island
                        stack.append(nxt)

        self.det_isl_alive_stat()

    def det_isl_alive_stat(self):
        """Populate live DC islands, including V references from generators and DCDC converters."""
        for island in self.islands:
            island.reset()

        for node in self.nodes:
            node.v_gens = []
            node.v_dcdcs = []
            node.v_set = 0.0
            node.is_slack = False

        for gen in self.generators:
            if gen.run_stat != 1 or gen.node_obj is None or gen.node_obj.isl_obj is None:
                continue
            island = gen.node_obj.isl_obj
            island.gens.append(gen)
            if gen.control_type == "V":
                gen.node_obj.v_gens.append(gen)
                island.v_gens.append(gen)

        for conv in self.dcdc_converters:
            if conv.run_stat != 1 or conv.i_node_obj is None or conv.j_node_obj is None:
                continue
            if conv.i_node_obj.isl_obj is not None:
                conv.i_node_obj.isl_obj.dcdc_converters.append(conv)
            if conv.j_node_obj.isl_obj is not None and conv.j_node_obj.isl_obj != conv.i_node_obj.isl_obj:
                conv.j_node_obj.isl_obj.dcdc_converters.append(conv)
            if conv.control_type == "V" and conv.i_node_obj.isl_obj is not None:
                conv.i_node_obj.v_dcdcs.append(conv)
                conv.i_node_obj.isl_obj.v_dcdcs.append(conv)

        for ld in self.loads:
            if ld.run_stat == 1 and ld.node_obj is not None and ld.node_obj.isl_obj is not None:
                ld.node_obj.isl_obj.loads.append(ld)
        for sw in self.switches:
            if sw.run_stat == 1 and sw.status == 1 and self._same_live_island(sw):
                sw.i_node_obj.isl_obj.switches.append(sw)
        for br in self.branches:
            if br.run_stat == 1 and self._same_live_island(br):
                br.i_node_obj.isl_obj.branches.append(br)
        for zbr in self.zero_branches:
            if zbr.run_stat == 1 and self._same_live_island(zbr):
                zbr.i_node_obj.isl_obj.zero_branches.append(zbr)

        for node in self.nodes:
            if node.isl_obj is not None:
                node.isl_obj.nodes.append(node)
            refs = node.v_gens + node.v_dcdcs
            if refs and node.isl_obj is not None:
                node.isl_obj.slack_nodes.append(node)
                node.is_slack = True
                node.v_set = refs[0].v_set

        for island in self.islands:
            island.is_alive = len(island.slack_nodes) >= 1

        for node in self.nodes:
            node.is_alive = node.isl_obj is not None and node.isl_obj.is_alive
        for ld in self.loads:
            ld.is_alive = ld.node_obj is not None and ld.run_stat == 1 and ld.node_obj.is_alive
        for gen in self.generators:
            gen.is_alive = gen.node_obj is not None and gen.run_stat == 1 and gen.node_obj.is_alive
        for dev in [*self.branches, *self.zero_branches]:
            dev.is_alive = (
                dev.run_stat == 1
                and dev.i_node_obj is not None
                and dev.j_node_obj is not None
                and dev.i_node_obj.is_alive
                and dev.j_node_obj.is_alive
            )
        for sw in self.switches:
            sw.is_alive = (
                sw.run_stat == 1
                and sw.status == 1
                and sw.i_node_obj is not None
                and sw.j_node_obj is not None
                and sw.i_node_obj.is_alive
                and sw.j_node_obj.is_alive
            )
        for conv in self.dcdc_converters:
            conv.is_alive = (
                conv.run_stat == 1
                and conv.i_node_obj is not None
                and conv.j_node_obj is not None
                and conv.i_node_obj.is_alive
                and conv.j_node_obj.is_alive
            )

    def print_isl_info(self):
        for island in self.islands:
            print(f"DC island {island.idx} is_alive = {island.is_alive}")
            print(f"    nodes = {len(island.nodes)}")
            print(f"    gens = {len(island.gens)}")
            print(f"    loads = {len(island.loads)}")
            print(f"    branches = {len(island.branches)}")
            print(f"    switches = {len(island.switches)}")
            print(f"    zero_branches = {len(island.zero_branches)}")
            print(f"    dcdc_converters = {len(island.dcdc_converters)}")

    def check_topo(self, extra_node_refs=None):
        if not self.islands:
            self.topo()
        errors = []
        warns = []
        node_ref_count = {node.idx: 0 for node in self.nodes}
        for node_idx, count in (extra_node_refs or {}).items():
            if node_idx in node_ref_count:
                node_ref_count[node_idx] += count

        def check_node(node_idx, dev_type, dev):
            if node_idx not in self.node_dict:
                errors.append(f"设备 {dev_type}[{dev.idx}] {self._name(dev)} 引用的节点 {node_idx} 不存在")
            elif self.node_dict[node_idx].run_stat == 1:
                node_ref_count[node_idx] += 1

        for br in self.branches:
            if br.run_stat == 1:
                check_node(br.i_node, "DCBranch", br)
                check_node(br.j_node, "DCBranch", br)
        for zbr in self.zero_branches:
            if zbr.run_stat == 1:
                check_node(zbr.i_node, "DCZeroBranch", zbr)
                check_node(zbr.j_node, "DCZeroBranch", zbr)
        for sw in self.switches:
            if sw.run_stat == 1:
                check_node(sw.i_node, "DCSwitch", sw)
                check_node(sw.j_node, "DCSwitch", sw)
        for ld in self.loads:
            if ld.run_stat == 1:
                check_node(ld.node, "DCLoad", ld)
        for gen in self.generators:
            if gen.run_stat == 1:
                check_node(gen.node, "DCGenerator", gen)
        for conv in self.dcdc_converters:
            if conv.run_stat == 1:
                check_node(conv.i_node, "DCDCConverter", conv)
                check_node(conv.j_node, "DCDCConverter", conv)

        for node in self.nodes:
            if node.run_stat != 1:
                continue
            if node_ref_count[node.idx] == 0:
                errors.append(f"节点 {node.idx} {self._name(node)} 未关联任何设备")
            elif node_ref_count[node.idx] == 1:
                warns.append(f"节点 {node.idx} {self._name(node)} 单端悬空，请检查！")

        for island in self.islands:
            vbase_set = {int(node.vbase * 1000) for node in island.nodes}
            if len(vbase_set) > 1:
                values = " ".join(f"{v / 1000.0:.2f}" for v in sorted(vbase_set))
                errors.append(f"岛屿 {island.idx} 内节点电压基值不一致: {values}")
            if len(island.slack_nodes) > 1:
                names = " ".join(self._name(node) for node in island.slack_nodes)
                errors.append(f"岛屿 {island.idx} 存在多个定V节点: {names}")
            if len(island.v_dcdcs) > 1:
                names = " ".join(self._name(conv) for conv in island.v_dcdcs)
                errors.append(f"岛屿 {island.idx} 存在多个定V变流器: {names}")
            if len(island.slack_nodes) + len(island.v_dcdcs) == 0:
                errors.append(f"岛屿 {island.idx} , 内无电压控制源（定V节点或定V变流器）")

        for node in self.nodes:
            if len(node.v_gens) + len(node.v_dcdcs) > 1:
                errors.append(f"松弛节点 {node.idx} 上的定V发电机与定V变流器数量之和超过1，请检查拓扑！")

        return warns, errors


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
        self.ac_nodes.extend(island.nodes)
        self.is_alive = self.is_alive or island.is_alive

    def add_dc_island(self, island):
        self.dc_islands.append(island)
        island.hybrid_isl = self.idx
        island.hybrid_isl_obj = self
        self.dc_nodes.extend(island.nodes)
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
    ac: HybridACGrid
    dc: HybridDCGrid
    dcac_converters: List
    acac_converters: List
    hybrid_islands: List[HybridIsland] = field(default_factory=list)

    @classmethod
    def read_from_file(cls, file_name) -> "HybridPowerNetwork":
        path = str(file_name)
        model = efile_factory_from_file_cached(path)
        normalize_model_named_units(model)
        network = cls(
            ac=HybridACGrid(model),
            dc=HybridDCGrid(model),
            dcac_converters=getattr(model, "DCACConverter", []),
            acac_converters=getattr(model, "ACACConverter", []),
        )
        network.p_base = float(model.p_base)
        network.p_base_kW = float(model.p_base_kW)
        network.u_scale = float(model.u_scale)
        network.p_scale = float(model.p_scale)
        network.i_scale = float(model.i_scale)
        return network

    @property
    def total_nodes(self) -> int:
        return len(self.ac.nodes) + len(self.dc.nodes)

    def topo(self):
        """Run AC/DC topology separately, then merge them through live converters."""
        if self.ac.nodes:
            self.ac.topo()
        if self.dc.nodes:
            self.dc.topo()
        self._build_hybrid_topo()

    def _build_hybrid_topo(self):
        """Union AC and DC islands connected by DCAC, DCDC or ACAC converters."""
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
            # Path-compressed union-find keeps converter-driven island merging simple.
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
            if not conv.is_alive:
                continue
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
            if not conv.is_alive:
                continue
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
        """Count converter terminals as node references for AC/DC topology checks."""
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
        """Prepare topology and return AC/DC warnings and errors without running Newton."""
        self.topo()
        if verbose and self.ac.nodes:
            self.ac.print_isl_info()
        if verbose and self.dc.nodes:
            self.dc.print_isl_info()
        if verbose:
            self.print_hybrid_isl_info()
        return [], [], [], []

    def check_topology(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Run optional topology diagnostics outside the main power-flow path."""
        if not self.hybrid_islands:
            self.topo()
        ac_refs, dc_refs = self._external_node_refs()
        ac_warnings, ac_errors = self.ac.check_topo(ac_refs) if self.ac.nodes else ([], [])
        dc_warnings, dc_errors = self.dc.check_topo(dc_refs) if self.dc.nodes else ([], [])
        dc_errors.extend(self._check_dcac_topo())
        dc_errors.extend(self._check_acac_topo())
        return ac_warnings, ac_errors, dc_warnings, dc_errors

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
            if str(conv.control_type).upper() not in {"DCV", "ACV", "ACP"}:
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


@dataclass
class HybridPowerFlowResult:
    network: HybridPowerNetwork
    ac_network: Any
    dc_network: Any
    calc: "HybridPowerFlowCalc"
    ac: Optional[ACPowerFlowCalc]
    dc: Optional[DCPowerFlowCalc]
    rc: int
    ac_warnings: List[str]
    ac_errors: List[str]
    dc_warnings: List[str]
    dc_errors: List[str]

    @property
    def total_nodes(self) -> int:
        return self.network.total_nodes

    @property
    def converged(self) -> bool:
        return self.rc == 0 and self.calc.converged and not self.ac_errors and not self.dc_errors

    @property
    def global_jacobian_shape(self) -> Tuple[int, int]:
        return self.calc.last_jacobian_shape

    @property
    def has_ac(self) -> bool:
        return len(self.ac_network.nodes) > 0

    @property
    def has_dc(self) -> bool:
        return len(self.dc_network.nodes) > 0

    @property
    def has_dcac(self) -> bool:
        return len(self.network.dcac_converters) > 0

    @property
    def has_acac(self) -> bool:
        return len(self.network.acac_converters) > 0


def _run_with_optional_output(verbose: bool, func, *args, **kwargs):
    if verbose:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class HybridPowerFlowCalc:
    """统一交直流 Newton 求解器。

    AC、DC 子网和 DC/AC 换流器变量在同一个全局状态向量中求解。
    """

    def __init__(
        self,
        network: HybridPowerNetwork,
        tol=None,
        max_iter=None,
        min_voltage=None,
        verbose=True,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
    ):
        self.network = network
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.verbose = verbose
        self.has_ac = len(network.ac.nodes) > 0
        self.has_dc = len(network.dc.nodes) > 0
        self.ac_calc = ACPowerFlowCalc(network.ac, parameters=self.params) if self.has_ac else None
        self.dc_calc = DCPowerFlowCalc(network.dc, parameters=self.params) if self.has_dc else None
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.x = np.array([], dtype=np.float64)
        self.ac_size = 0
        self.dc_size = 0
        self.ac_eq = 0
        self.dc_eq = 0
        self.dcac_start = 0
        self.dcac_eq_start = 0
        self.dcac_converters = []
        self.N_dcac = 0
        self.acac_start = 0
        self.acac_eq_start = 0
        self.acac_converters = []
        self.N_acac = 0
        self.total_vars = 0
        self.total_eq = 0
        self.dc_G = None
        self.last_jacobian_shape = (0, 0)
        self._clear_dcac_arrays()
        self._clear_acac_arrays()
        self._clear_converter_jacobian_structure()

    def prepare(self):
        """Build the global hybrid state vector and block equation layout."""
        parts = []
        if self.ac_calc is not None:
            _run_with_optional_output(self.verbose, self.ac_calc.prepare)
            self.ac_size = self.ac_calc.total_vars
            self.ac_eq = self.ac_calc.total_eq
            parts.append(self.ac_calc.x.copy())
        if self.dc_calc is not None:
            self.dc_G, dc_x = _run_with_optional_output(self.verbose, self.dc_calc.prepare)
            self.dc_size = self.dc_calc.total_vars
            self.dc_eq = self.dc_calc.total_eq
            parts.append(dc_x.copy())
        self._prepare_dcac_converters()
        self._prepare_acac_converters()
        if not parts:
            raise RuntimeError("E 文件中没有 ACNode 或 DCNode，无法进行交直流潮流计算")
        self.dcac_start = self.ac_size + self.dc_size
        dcac_x = self._initial_dcac_x()
        if dcac_x.size:
            parts.append(dcac_x)
        self.acac_start = self.dcac_start + dcac_x.size
        acac_x = self._initial_acac_x()
        if acac_x.size:
            parts.append(acac_x)
        self.dcac_eq_start = self.ac_eq + self.dc_eq
        self.acac_eq_start = self.dcac_eq_start + self.N_dcac * 3
        self.x = np.concatenate(parts)
        self.total_vars = self.x.size
        self.total_eq = self.acac_eq_start + self.N_acac * 4
        # Variable/equation order is block diagonal first, then converter coupling rows.
        self.last_jacobian_shape = (self.total_eq, self.total_vars)
        self._cache_converter_jacobian_structure()
        if self.verbose:
            print(
                "Hybrid prepare:",
                f"ac_vars={self.ac_size}",
                f"dc_vars={self.dc_size}",
                f"dcac_vars={self.N_dcac * 3}",
                f"acac_vars={self.N_acac * 4}",
                f"total_vars={self.x.size}",
                f"total_eq={self.last_jacobian_shape[0]}",
            )
        return self.x

    def _split_x(self, x):
        """Return AC, DC, DCAC and ACAC slices from the global Newton vector."""
        ac_x = x[:self.ac_size]
        dc_x = x[self.ac_size:self.ac_size + self.dc_size]
        dcac_x = x[self.dcac_start:self.dcac_start + self.N_dcac * 3]
        acac_x = x[self.acac_start:self.acac_start + self.N_acac * 4]
        return ac_x, dc_x, dcac_x, acac_x

    def _prepare_dcac_converters(self):
        """Validate live DCAC converters and map terminal nodes to AC/DC solver indices."""
        self.dcac_converters = []
        self._clear_dcac_arrays()
        if not self.has_ac or not self.has_dc:
            return
        for conv in self.network.dcac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            if conv.ac_node not in self.ac_calc.node_pos:
                raise ValueError(f"DCACConverter[{conv.idx}] 引用的 AC 节点 {conv.ac_node} 不存在或不带电")
            if conv.dc_node not in self.dc_calc.alive_node_dict:
                raise ValueError(f"DCACConverter[{conv.idx}] 引用的 DC 节点 {conv.dc_node} 不存在或不带电")
            ac_pos = self.ac_calc.node_pos[conv.ac_node]
            if self.ac_calc.node_type[ac_pos] != "PQ":
                raise ValueError(f"DCACConverter[{conv.idx}] 的 AC 节点必须是 PQ 节点，当前为 {self.ac_calc.node_type[ac_pos]}")
            dc_pos = self.dc_calc.alive_node_dict[conv.dc_node]
            ctrl = str(conv.control_type).upper()
            if ctrl not in {"DCV", "ACV", "ACP"}:
                raise ValueError(f"未知 DCACConverter 控制模式: {conv.control_type}")
            self.dcac_converters.append((conv, ac_pos, dc_pos, ctrl))
        self.N_dcac = len(self.dcac_converters)
        self._cache_dcac_arrays()

    def _clear_dcac_arrays(self):
        self.dcac_devices = []
        self.dcac_ac_pos = np.array([], dtype=np.int32)
        self.dcac_dc_pos = np.array([], dtype=np.int32)
        self.dcac_ctrl_code = np.array([], dtype=np.int8)
        self.dcac_r1 = np.array([], dtype=np.float64)
        self.dcac_r2 = np.array([], dtype=np.float64)
        self.dcac_v_dc_set = np.array([], dtype=np.float64)
        self.dcac_v_ac_set = np.array([], dtype=np.float64)
        self.dcac_p_ac_set = np.array([], dtype=np.float64)
        self.dcac_q_ac_set = np.array([], dtype=np.float64)
        self.dcac_ac_p_row = np.array([], dtype=np.int32)
        self.dcac_ac_q_row = np.array([], dtype=np.int32)
        self.dcac_dc_eq = np.array([], dtype=np.int32)
        self.dcac_dc_eq_mask = np.array([], dtype=bool)
        self.dcac_ac_v_col = np.array([], dtype=np.int32)
        self.dcac_dc_v_col = np.array([], dtype=np.int32)

    def _cache_dcac_arrays(self):
        """Cache DCAC converter metadata as arrays for residual/Jacobian assembly."""
        if not self.dcac_converters:
            self._clear_dcac_arrays()
            return
        ctrl_map = {"DCV": 0, "ACV": 1, "ACP": 2}
        self.dcac_ac_pos = np.asarray([item[1] for item in self.dcac_converters], dtype=np.int32)
        self.dcac_dc_pos = np.asarray([item[2] for item in self.dcac_converters], dtype=np.int32)
        self.dcac_ctrl_code = np.asarray([ctrl_map[item[3]] for item in self.dcac_converters], dtype=np.int8)
        convs = [item[0] for item in self.dcac_converters]
        self.dcac_devices = convs
        self.dcac_r1 = np.asarray([conv.r1 for conv in convs], dtype=np.float64)
        self.dcac_r2 = np.asarray([conv.r2 for conv in convs], dtype=np.float64)
        self.dcac_v_dc_set = np.asarray([conv.v_dc_set for conv in convs], dtype=np.float64)
        self.dcac_v_ac_set = np.asarray([conv.v_ac_set for conv in convs], dtype=np.float64)
        self.dcac_p_ac_set = np.asarray([conv.p_ac_set for conv in convs], dtype=np.float64)
        self.dcac_q_ac_set = np.asarray([conv.q_ac_set for conv in convs], dtype=np.float64)
        self.dcac_ac_p_row = np.asarray(
            [self.ac_calc.theta_idx[int(pos)] for pos in self.dcac_ac_pos],
            dtype=np.int32,
        )
        self.dcac_ac_q_row = np.asarray(
            [self.ac_calc.n_theta + self.ac_calc.V_idx[int(pos)] for pos in self.dcac_ac_pos],
            dtype=np.int32,
        )
        self.dcac_dc_eq = np.asarray([self.dc_calc.node_eq[int(pos)] for pos in self.dcac_dc_pos], dtype=np.int32)
        self.dcac_dc_eq_mask = self.dcac_dc_eq >= 0
        self.dcac_ac_v_col = self.dcac_ac_q_row.copy()
        self.dcac_dc_v_col = self.ac_size + self.dcac_dc_pos

    def _prepare_acac_converters(self):
        """Validate live ACAC converters and map both AC terminals to solver indices."""
        self.acac_converters = []
        self._clear_acac_arrays()
        if not self.has_ac:
            return
        for conv in self.network.acac_converters:
            if not getattr(conv, "is_alive", False):
                continue
            if conv.i_node not in self.ac_calc.node_pos:
                raise ValueError(f"ACACConverter[{conv.idx}] 引用的 AC 节点 {conv.i_node} 不存在或不带电")
            if conv.j_node not in self.ac_calc.node_pos:
                raise ValueError(f"ACACConverter[{conv.idx}] 引用的 AC 节点 {conv.j_node} 不存在或不带电")
            i_pos = self.ac_calc.node_pos[conv.i_node]
            j_pos = self.ac_calc.node_pos[conv.j_node]
            if i_pos == j_pos:
                raise ValueError(f"ACACConverter[{conv.idx}] 两端不能连接同一个 AC 节点")
            if self.ac_calc.node_type[i_pos] != "PQ":
                raise ValueError(f"ACACConverter[{conv.idx}] 的 i 侧 AC 节点必须是 PQ 节点，当前为 {self.ac_calc.node_type[i_pos]}")
            if self.ac_calc.node_type[j_pos] != "PQ":
                raise ValueError(f"ACACConverter[{conv.idx}] 的 j 侧 AC 节点必须是 PQ 节点，当前为 {self.ac_calc.node_type[j_pos]}")
            ctrl = str(conv.control_type).upper()
            if ctrl not in ACAC_CONTROL_TYPES:
                raise ValueError(f"未知 ACACConverter 控制模式: {conv.control_type}")
            self.acac_converters.append((conv, i_pos, j_pos, ctrl))
        self.N_acac = len(self.acac_converters)
        self._cache_acac_arrays()

    def _clear_acac_arrays(self):
        self.acac_devices = []
        self.acac_i_pos = np.array([], dtype=np.int32)
        self.acac_j_pos = np.array([], dtype=np.int32)
        self.acac_ctrl_code = np.array([], dtype=np.int8)
        self.acac_r1 = np.array([], dtype=np.float64)
        self.acac_r2 = np.array([], dtype=np.float64)
        self.acac_p_set = np.array([], dtype=np.float64)
        self.acac_i_q_set = np.array([], dtype=np.float64)
        self.acac_j_q_set = np.array([], dtype=np.float64)
        self.acac_i_v_set = np.array([], dtype=np.float64)
        self.acac_j_v_set = np.array([], dtype=np.float64)
        self.acac_i_p_row = np.array([], dtype=np.int32)
        self.acac_i_q_row = np.array([], dtype=np.int32)
        self.acac_j_p_row = np.array([], dtype=np.int32)
        self.acac_j_q_row = np.array([], dtype=np.int32)
        self.acac_i_v_col = np.array([], dtype=np.int32)
        self.acac_j_v_col = np.array([], dtype=np.int32)

    def _clear_converter_jacobian_structure(self):
        """Reset cached sparse row/column patterns for converter Jacobian terms."""
        self.dcac_dc_p_col = np.array([], dtype=np.int32)
        self.dcac_ac_p_col = np.array([], dtype=np.int32)
        self.dcac_ac_q_col = np.array([], dtype=np.int32)
        self.dcac_eq_loss = np.array([], dtype=np.int32)
        self.dcac_eq_ctrl_1 = np.array([], dtype=np.int32)
        self.dcac_eq_ctrl_2 = np.array([], dtype=np.int32)
        self.dcac_loss_rows = np.array([], dtype=np.int32)
        self.dcac_loss_cols = np.array([], dtype=np.int32)
        self.dcac_dc_eq_rows = np.array([], dtype=np.int32)
        self.dcac_dc_eq_cols = np.array([], dtype=np.int32)
        self.dcac_ctrl_dc_v_mask = np.array([], dtype=bool)
        self.dcac_ctrl_ac_v_mask = np.array([], dtype=bool)
        self.dcac_ctrl_ac_p_mask = np.array([], dtype=bool)
        self.dcac_ones = np.array([], dtype=np.float64)
        self.dcac_dc_eq_ones = np.array([], dtype=np.float64)

        self.acac_i_p_col = np.array([], dtype=np.int32)
        self.acac_i_q_col = np.array([], dtype=np.int32)
        self.acac_j_p_col = np.array([], dtype=np.int32)
        self.acac_j_q_col = np.array([], dtype=np.int32)
        self.acac_eq_loss = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_1 = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_2 = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_3 = np.array([], dtype=np.int32)
        self.acac_loss_rows = np.array([], dtype=np.int32)
        self.acac_loss_cols = np.array([], dtype=np.int32)
        self.acac_q_i_mask = np.array([], dtype=bool)
        self.acac_v_i_mask = np.array([], dtype=bool)
        self.acac_q_j_mask = np.array([], dtype=bool)
        self.acac_v_j_mask = np.array([], dtype=bool)
        self.acac_ones = np.array([], dtype=np.float64)

    def _cache_converter_jacobian_structure(self):
        """Precompute converter Jacobian row/column indices once per prepared case."""
        self._clear_converter_jacobian_structure()
        if self.N_dcac:
            idx = np.arange(self.N_dcac, dtype=np.int32)
            self.dcac_dc_p_col = self.dcac_start + 3 * idx
            self.dcac_ac_p_col = self.dcac_dc_p_col + 1
            self.dcac_ac_q_col = self.dcac_dc_p_col + 2
            self.dcac_eq_loss = self.dcac_eq_start + 3 * idx
            self.dcac_eq_ctrl_1 = self.dcac_eq_loss + 1
            self.dcac_eq_ctrl_2 = self.dcac_eq_loss + 2

            self.dcac_loss_rows = np.repeat(self.dcac_eq_loss, 5)
            self.dcac_loss_cols = np.empty(self.N_dcac * 5, dtype=np.int32)
            self.dcac_loss_cols[0::5] = self.dcac_dc_p_col
            self.dcac_loss_cols[1::5] = self.dcac_ac_p_col
            self.dcac_loss_cols[2::5] = self.dcac_ac_q_col
            self.dcac_loss_cols[3::5] = self.dcac_ac_v_col
            self.dcac_loss_cols[4::5] = self.dcac_dc_v_col

            if self.dcac_dc_eq_mask.any():
                self.dcac_dc_eq_rows = self.ac_eq + self.dcac_dc_eq[self.dcac_dc_eq_mask]
                self.dcac_dc_eq_cols = self.dcac_dc_p_col[self.dcac_dc_eq_mask]
                self.dcac_dc_eq_ones = np.ones(self.dcac_dc_eq_rows.size, dtype=np.float64)
            self.dcac_ctrl_dc_v_mask = self.dcac_ctrl_code == 0
            self.dcac_ctrl_ac_v_mask = self.dcac_ctrl_code == 1
            self.dcac_ctrl_ac_p_mask = self.dcac_ctrl_code == 2
            self.dcac_ones = np.ones(self.N_dcac, dtype=np.float64)

        if self.N_acac:
            idx = np.arange(self.N_acac, dtype=np.int32)
            self.acac_i_p_col = self.acac_start + 4 * idx
            self.acac_i_q_col = self.acac_i_p_col + 1
            self.acac_j_p_col = self.acac_i_p_col + 2
            self.acac_j_q_col = self.acac_i_p_col + 3
            self.acac_eq_loss = self.acac_eq_start + 4 * idx
            self.acac_eq_ctrl_1 = self.acac_eq_loss + 1
            self.acac_eq_ctrl_2 = self.acac_eq_loss + 2
            self.acac_eq_ctrl_3 = self.acac_eq_loss + 3

            self.acac_loss_rows = np.repeat(self.acac_eq_loss, 6)
            self.acac_loss_cols = np.empty(self.N_acac * 6, dtype=np.int32)
            self.acac_loss_cols[0::6] = self.acac_i_p_col
            self.acac_loss_cols[1::6] = self.acac_i_q_col
            self.acac_loss_cols[2::6] = self.acac_j_p_col
            self.acac_loss_cols[3::6] = self.acac_j_q_col
            self.acac_loss_cols[4::6] = self.acac_i_v_col
            self.acac_loss_cols[5::6] = self.acac_j_v_col

            self.acac_q_i_mask = (self.acac_ctrl_code == 0) | (self.acac_ctrl_code == 2)
            self.acac_v_i_mask = ~self.acac_q_i_mask
            self.acac_q_j_mask = (self.acac_ctrl_code == 0) | (self.acac_ctrl_code == 1)
            self.acac_v_j_mask = ~self.acac_q_j_mask
            self.acac_ones = np.ones(self.N_acac, dtype=np.float64)

    def _cache_acac_arrays(self):
        """Cache ACAC converter metadata as arrays for residual/Jacobian assembly."""
        if not self.acac_converters:
            self._clear_acac_arrays()
            return
        ctrl_map = {"PQQ": 0, "PVQ": 1, "PQV": 2, "PVV": 3}
        self.acac_i_pos = np.asarray([item[1] for item in self.acac_converters], dtype=np.int32)
        self.acac_j_pos = np.asarray([item[2] for item in self.acac_converters], dtype=np.int32)
        self.acac_ctrl_code = np.asarray([ctrl_map[item[3]] for item in self.acac_converters], dtype=np.int8)
        convs = [item[0] for item in self.acac_converters]
        self.acac_devices = convs
        self.acac_r1 = np.asarray([conv.r1 for conv in convs], dtype=np.float64)
        self.acac_r2 = np.asarray([conv.r2 for conv in convs], dtype=np.float64)
        self.acac_p_set = np.asarray([conv.p_set for conv in convs], dtype=np.float64)
        self.acac_i_q_set = np.asarray([conv.i_q_set for conv in convs], dtype=np.float64)
        self.acac_j_q_set = np.asarray([conv.j_q_set for conv in convs], dtype=np.float64)
        self.acac_i_v_set = np.asarray([conv.i_v_set for conv in convs], dtype=np.float64)
        self.acac_j_v_set = np.asarray([conv.j_v_set for conv in convs], dtype=np.float64)
        self.acac_i_p_row = np.asarray([self.ac_calc.theta_idx[int(pos)] for pos in self.acac_i_pos], dtype=np.int32)
        self.acac_i_q_row = np.asarray(
            [self.ac_calc.n_theta + self.ac_calc.V_idx[int(pos)] for pos in self.acac_i_pos],
            dtype=np.int32,
        )
        self.acac_j_p_row = np.asarray([self.ac_calc.theta_idx[int(pos)] for pos in self.acac_j_pos], dtype=np.int32)
        self.acac_j_q_row = np.asarray(
            [self.ac_calc.n_theta + self.ac_calc.V_idx[int(pos)] for pos in self.acac_j_pos],
            dtype=np.int32,
        )
        self.acac_i_v_col = self.acac_i_q_row.copy()
        self.acac_j_v_col = self.acac_j_q_row.copy()

    def _initial_dcac_x(self):
        if not self.dcac_converters:
            return np.array([], dtype=np.float64)
        x = np.zeros(self.N_dcac * 3, dtype=np.float64)
        ac_p = np.where(self.dcac_ctrl_code == 2, self.dcac_p_ac_set, 0.0)
        x[0::3] = -ac_p
        x[1::3] = ac_p
        x[2::3] = self.dcac_q_ac_set
        return x

    def _initial_acac_x(self):
        if not self.acac_converters:
            return np.array([], dtype=np.float64)
        x = np.zeros(self.N_acac * 4, dtype=np.float64)
        x[0::4] = self.acac_p_set
        x[1::4] = self.acac_i_q_set
        x[2::4] = -self.acac_p_set
        x[3::4] = self.acac_j_q_set
        return x

    def _state_values(self, ac_x, dc_x):
        """Extract node voltage arrays needed by converter equations from sub-solver states."""
        ac_theta = ac_V = None
        dc_V = None
        if self.ac_calc is not None:
            ac_theta, ac_V, _, _ = self.ac_calc._extract_state_vars(ac_x)
        if self.dc_calc is not None:
            dc_V = dc_x[:self.dc_calc.N]
        return ac_theta, ac_V, dc_V

    def _cached_state_values(self, ac_x, dc_x):
        """Reuse AC sub-solver state cache after AC residual/Jacobian evaluation."""
        ac_theta = ac_V = None
        dc_V = None
        if self.ac_calc is not None:
            cache = getattr(self.ac_calc, "_cache", {})
            ac_theta = cache.get("theta")
            ac_V = cache.get("V")
            if ac_theta is None or ac_V is None:
                ac_theta, ac_V, _, _ = self.ac_calc._extract_state_vars(ac_x)
        if self.dc_calc is not None:
            dc_V = dc_x[:self.dc_calc.N]
        return ac_theta, ac_V, dc_V

    def get_f(self, x):
        """Assemble global residuals for AC, DC, DCAC and ACAC equations."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        parts = []
        ac_f = None
        dc_f = None
        if self.ac_calc is not None:
            ac_f = self.ac_calc.get_f(ac_x)
            parts.append(ac_f)
        if self.dc_calc is not None:
            dc_f = self.dc_calc.get_f(dc_x)
            parts.append(dc_f)
        ac_V = dc_V = None
        if self.N_dcac or self.N_acac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        if self.N_dcac:
            dcac = dcac_x.reshape(self.N_dcac, 3)
            dc_p = dcac[:, 0]
            ac_p = dcac[:, 1]
            ac_q = dcac[:, 2]
            # Converter port powers are injected into the existing AC/DC nodal balance rows.
            np.add.at(ac_f, self.dcac_ac_p_row, ac_p)
            np.add.at(ac_f, self.dcac_ac_q_row, ac_q)
            if self.dcac_dc_eq_mask.any():
                np.add.at(dc_f, self.dcac_dc_eq[self.dcac_dc_eq_mask], dc_p[self.dcac_dc_eq_mask])

            va = ac_V[self.dcac_ac_pos]
            vd = dc_V[self.dcac_dc_pos]
            va2 = va * va
            vd2 = vd * vd
            dcac_f = np.empty(self.N_dcac * 3, dtype=np.float64)
            # r1+r2 converter loss equation in per-unit power/voltage variables.
            dcac_f[0::3] = (
                vd2 * va2 * (dc_p + ac_p)
                - self.dcac_r1 * dc_p * dc_p * va2
                - self.dcac_r2 * (ac_p * ac_p + ac_q * ac_q) * vd2
            )
            f_ctrl = np.empty(self.N_dcac, dtype=np.float64)
            f_ctrl[self.dcac_ctrl_dc_v_mask] = (
                vd[self.dcac_ctrl_dc_v_mask] - self.dcac_v_dc_set[self.dcac_ctrl_dc_v_mask]
            )
            f_ctrl[self.dcac_ctrl_ac_v_mask] = (
                va[self.dcac_ctrl_ac_v_mask] - self.dcac_v_ac_set[self.dcac_ctrl_ac_v_mask]
            )
            f_ctrl[self.dcac_ctrl_ac_p_mask] = (
                ac_p[self.dcac_ctrl_ac_p_mask] - self.dcac_p_ac_set[self.dcac_ctrl_ac_p_mask]
            )
            dcac_f[1::3] = f_ctrl
            dcac_f[2::3] = ac_q - self.dcac_q_ac_set
            parts.append(dcac_f)
        if self.N_acac:
            acac = acac_x.reshape(self.N_acac, 4)
            i_p = acac[:, 0]
            i_q = acac[:, 1]
            j_p = acac[:, 2]
            j_q = acac[:, 3]
            # ACAC port powers couple two AC PQ nodes inside the same global system.
            np.add.at(ac_f, self.acac_i_p_row, i_p)
            np.add.at(ac_f, self.acac_i_q_row, i_q)
            np.add.at(ac_f, self.acac_j_p_row, j_p)
            np.add.at(ac_f, self.acac_j_q_row, j_q)

            vi = ac_V[self.acac_i_pos]
            vj = ac_V[self.acac_j_pos]
            vi2 = vi * vi
            vj2 = vj * vj
            acac_f = np.empty(self.N_acac * 4, dtype=np.float64)
            acac_f[0::4] = (
                vi2 * vj2 * (i_p + j_p)
                - self.acac_r1 * (i_p * i_p + i_q * i_q) * vj2
                - self.acac_r2 * (j_p * j_p + j_q * j_q) * vi2
            )
            acac_f[1::4] = i_p - self.acac_p_set
            f2 = np.empty(self.N_acac, dtype=np.float64)
            f3 = np.empty(self.N_acac, dtype=np.float64)
            f2[self.acac_q_i_mask] = i_q[self.acac_q_i_mask] - self.acac_i_q_set[self.acac_q_i_mask]
            f2[self.acac_v_i_mask] = vi[self.acac_v_i_mask] - self.acac_i_v_set[self.acac_v_i_mask]
            f3[self.acac_q_j_mask] = j_q[self.acac_q_j_mask] - self.acac_j_q_set[self.acac_q_j_mask]
            f3[self.acac_v_j_mask] = vj[self.acac_v_j_mask] - self.acac_j_v_set[self.acac_v_j_mask]
            acac_f[2::4] = f2
            acac_f[3::4] = f3
            parts.append(acac_f)
        return np.concatenate(parts)

    def get_jacobi(self, x):
        """Build the global sparse Jacobian from sub-solver blocks plus converter couplings."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        row_parts = []
        col_parts = []
        data_parts = []
        if self.ac_calc is not None:
            ac_j = self.ac_calc.get_jacobi(ac_x).tocoo()
            row_parts.append(ac_j.row)
            col_parts.append(ac_j.col)
            data_parts.append(ac_j.data)
        target_shape = (self.total_eq, self.total_vars)
        if self.dc_calc is not None:
            dc_j = self.dc_calc.get_jacobi(self.dc_G, dc_x).tocoo()
            row_parts.append(dc_j.row + self.ac_eq)
            col_parts.append(dc_j.col + self.ac_size)
            data_parts.append(dc_j.data)

        ac_V = dc_V = None
        if self.N_dcac or self.N_acac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        if self.N_dcac:
            self._append_dcac_jacobian_terms(row_parts, col_parts, data_parts, dcac_x, ac_V, dc_V)
        if self.N_acac:
            self._append_acac_jacobian_terms(row_parts, col_parts, data_parts, acac_x, ac_V)

        if row_parts:
            rows = np.concatenate(row_parts)
            cols = np.concatenate(col_parts)
            data = np.concatenate(data_parts)
            nonzero = data != 0.0
            if not np.all(nonzero):
                rows = rows[nonzero]
                cols = cols[nonzero]
                data = data[nonzero]
            jac = coo_matrix((data, (rows, cols)), shape=target_shape).tocsr()
        else:
            jac = coo_matrix(target_shape, dtype=np.float64).tocsr()
        self.last_jacobian_shape = jac.shape
        return jac

    def _append_dcac_jacobian_terms(self, row_parts, col_parts, data_parts, dcac_x, ac_V, dc_V):
        """Append DC/AC converter Jacobian entries to global COO buffers."""
        n = self.N_dcac
        dcac = dcac_x.reshape(n, 3)
        dc_p = dcac[:, 0]
        ac_p = dcac[:, 1]
        ac_q = dcac[:, 2]

        row_parts.append(self.dcac_ac_p_row)
        col_parts.append(self.dcac_ac_p_col)
        data_parts.append(self.dcac_ones)
        row_parts.append(self.dcac_ac_q_row)
        col_parts.append(self.dcac_ac_q_col)
        data_parts.append(self.dcac_ones)
        if self.dcac_dc_eq_rows.size:
            row_parts.append(self.dcac_dc_eq_rows)
            col_parts.append(self.dcac_dc_eq_cols)
            data_parts.append(self.dcac_dc_eq_ones)

        va = ac_V[self.dcac_ac_pos]
        vd = dc_V[self.dcac_dc_pos]
        va2 = va * va
        vd2 = vd * vd
        dc_p2 = dc_p * dc_p
        ac_i2_num = ac_p * ac_p + ac_q * ac_q
        loss_data = np.empty(n * 5, dtype=np.float64)
        loss_data[0::5] = vd2 * va2 - 2.0 * self.dcac_r1 * dc_p * va2
        loss_data[1::5] = vd2 * va2 - 2.0 * self.dcac_r2 * ac_p * vd2
        loss_data[2::5] = -2.0 * self.dcac_r2 * ac_q * vd2
        loss_data[3::5] = 2.0 * va * vd2 * (dc_p + ac_p) - 2.0 * self.dcac_r1 * dc_p2 * va
        loss_data[4::5] = 2.0 * vd * va2 * (dc_p + ac_p) - 2.0 * self.dcac_r2 * ac_i2_num * vd
        row_parts.append(self.dcac_loss_rows)
        col_parts.append(self.dcac_loss_cols)
        data_parts.append(loss_data)

        row_parts.append(self.dcac_eq_ctrl_2)
        col_parts.append(self.dcac_ac_q_col)
        data_parts.append(self.dcac_ones)
        for mask, ctrl_col in (
            (self.dcac_ctrl_dc_v_mask, self.dcac_dc_v_col),
            (self.dcac_ctrl_ac_v_mask, self.dcac_ac_v_col),
            (self.dcac_ctrl_ac_p_mask, self.dcac_ac_p_col),
        ):
            if np.any(mask):
                row_parts.append(self.dcac_eq_ctrl_1[mask])
                col_parts.append(ctrl_col[mask])
                data_parts.append(self.dcac_ones[mask])

    def _append_acac_jacobian_terms(self, row_parts, col_parts, data_parts, acac_x, ac_V):
        """Append AC/AC converter Jacobian entries to global COO buffers."""
        n = self.N_acac
        acac = acac_x.reshape(n, 4)
        i_p = acac[:, 0]
        i_q = acac[:, 1]
        j_p = acac[:, 2]
        j_q = acac[:, 3]

        row_parts.append(self.acac_i_p_row)
        col_parts.append(self.acac_i_p_col)
        data_parts.append(self.acac_ones)
        row_parts.append(self.acac_i_q_row)
        col_parts.append(self.acac_i_q_col)
        data_parts.append(self.acac_ones)
        row_parts.append(self.acac_j_p_row)
        col_parts.append(self.acac_j_p_col)
        data_parts.append(self.acac_ones)
        row_parts.append(self.acac_j_q_row)
        col_parts.append(self.acac_j_q_col)
        data_parts.append(self.acac_ones)

        vi = ac_V[self.acac_i_pos]
        vj = ac_V[self.acac_j_pos]
        vi2 = vi * vi
        vj2 = vj * vj
        i_s2 = i_p * i_p + i_q * i_q
        j_s2 = j_p * j_p + j_q * j_q
        loss_data = np.empty(n * 6, dtype=np.float64)
        loss_data[0::6] = vi2 * vj2 - 2.0 * self.acac_r1 * i_p * vj2
        loss_data[1::6] = -2.0 * self.acac_r1 * i_q * vj2
        loss_data[2::6] = vi2 * vj2 - 2.0 * self.acac_r2 * j_p * vi2
        loss_data[3::6] = -2.0 * self.acac_r2 * j_q * vi2
        loss_data[4::6] = 2.0 * vi * vj2 * (i_p + j_p) - 2.0 * self.acac_r2 * j_s2 * vi
        loss_data[5::6] = 2.0 * vj * vi2 * (i_p + j_p) - 2.0 * self.acac_r1 * i_s2 * vj
        row_parts.append(self.acac_loss_rows)
        col_parts.append(self.acac_loss_cols)
        data_parts.append(loss_data)

        row_parts.append(self.acac_eq_ctrl_1)
        col_parts.append(self.acac_i_p_col)
        data_parts.append(self.acac_ones)
        for mask, rows_src, cols_src in (
            (self.acac_q_i_mask, self.acac_eq_ctrl_2, self.acac_i_q_col),
            (self.acac_v_i_mask, self.acac_eq_ctrl_2, self.acac_i_v_col),
            (self.acac_q_j_mask, self.acac_eq_ctrl_3, self.acac_j_q_col),
            (self.acac_v_j_mask, self.acac_eq_ctrl_3, self.acac_j_v_col),
        ):
            if np.any(mask):
                row_parts.append(rows_src[mask])
                col_parts.append(cols_src[mask])
                data_parts.append(self.acac_ones[mask])

    def run(self):
        """Execute unified Newton iterations over the full hybrid state vector."""
        if self.x.size == 0:
            self.prepare()

        self.converged = False
        x = self.x.copy()
        for it in range(self.max_iter):
            F = self.get_f(x)
            self.iterations = it + 1
            self.normF = np.linalg.norm(F, np.inf)

            ac_f = F[:self.ac_eq]
            dc_f = F[self.ac_eq:self.ac_eq + self.dc_eq]
            ac_norm = np.linalg.norm(ac_f, np.inf) if ac_f.size else 0.0
            dc_norm = np.linalg.norm(dc_f, np.inf) if dc_f.size else 0.0
            if self.ac_calc is not None:
                self.ac_calc.iterations = self.iterations
                self.ac_calc.normF = ac_norm
            if self.dc_calc is not None:
                self.dc_calc.iterations = self.iterations
                self.dc_calc.normF = dc_norm

            if self.verbose:
                print(
                    f"Hybrid iter {it}: "
                    f"|F|={self.normF:.3e}, "
                    f"|F_ac|={ac_norm:.3e}, "
                    f"|F_dc|={dc_norm:.3e}"
                )

            if self.normF < self.tol:
                self.converged = True
                self.x = x
                self._write_back(x)
                return 0

            J = self.get_jacobi(x)
            delta = spsolve(J, -F)
            x += delta

        self.x = x
        self._write_back(x)
        return -1

    def _write_back(self, x):
        """Write final global state back into AC, DC and converter model objects."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        if self.ac_calc is not None:
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc._write_back()
        if self.dc_calc is not None:
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.update_lf_info(dc_x)
        ac_V = dc_V = None
        if self.N_dcac or self.N_acac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        if self.N_dcac:
            dcac = dcac_x.reshape(self.N_dcac, 3)
            dc_p = dcac[:, 0]
            ac_p = dcac[:, 1]
            ac_q = dcac[:, 2]
            dc_v = dc_V[self.dcac_dc_pos]
            ac_v = ac_V[self.dcac_ac_pos]
            dc_i = np.divide(dc_p, dc_v, out=np.zeros_like(dc_p), where=np.abs(dc_v) > self.params.min_voltage)
            ac_i = np.divide(
                np.hypot(ac_p, ac_q),
                ac_v,
                out=np.zeros_like(ac_p),
                where=np.abs(ac_v) > self.params.min_voltage,
            )
            for conv, p_dc, p_ac, q_ac, i_dc, i_ac in zip(self.dcac_devices, dc_p, ac_p, ac_q, dc_i, ac_i):
                conv.dc_p = float(p_dc)
                conv.ac_p = float(p_ac)
                conv.ac_q = float(q_ac)
                conv.dc_i = float(i_dc)
                conv.ac_i = float(i_ac)
        if self.N_acac:
            acac = acac_x.reshape(self.N_acac, 4)
            i_p = acac[:, 0]
            i_q = acac[:, 1]
            j_p = acac[:, 2]
            j_q = acac[:, 3]
            vi = ac_V[self.acac_i_pos]
            vj = ac_V[self.acac_j_pos]
            i_i = np.divide(
                np.hypot(i_p, i_q),
                vi,
                out=np.zeros_like(i_p),
                where=np.abs(vi) > self.params.min_voltage,
            )
            j_i = np.divide(
                np.hypot(j_p, j_q),
                vj,
                out=np.zeros_like(j_p),
                where=np.abs(vj) > self.params.min_voltage,
            )
            for conv, p_i, q_i, p_j, q_j, cur_i, cur_j in zip(self.acac_devices, i_p, i_q, j_p, j_q, i_i, j_i):
                conv.i_p = float(p_i)
                conv.i_q = float(q_i)
                conv.j_p = float(p_j)
                conv.j_q = float(q_j)
                conv.i_i = float(cur_i)
                conv.j_i = float(cur_j)


def run_hybrid_power_flow(
    file_name=DEFAULT_HYBRID_EFILE,
    tol=None,
    max_iter=None,
    min_voltage=None,
    verbose=True,
    parameter_file=DEFAULT_LF_PARAMETER_FILE,
    parameters: Optional[PowerFlowParameters] = None,
) -> HybridPowerFlowResult:
    network = HybridPowerNetwork.read_from_file(file_name)

    ac_warnings, ac_errors, dc_warnings, dc_errors = _run_with_optional_output(verbose, network.prepare, verbose)

    for warn in ac_warnings:
        if verbose:
            print("  AC 警告:", warn)
    for error in ac_errors:
        if verbose:
            print("  AC 错误:", error)
    for warn in dc_warnings:
        if verbose:
            print("  DC 警告:", warn)
    for error in dc_errors:
        if verbose:
            print("  DC 错误:", error)

    calc = HybridPowerFlowCalc(
        network,
        tol=tol,
        max_iter=max_iter,
        min_voltage=min_voltage,
        verbose=verbose,
        parameter_file=parameter_file,
        parameters=parameters,
    )

    if ac_errors or dc_errors:
        return HybridPowerFlowResult(
            network=network,
            ac_network=network.ac,
            dc_network=network.dc,
            calc=calc,
            ac=calc.ac_calc,
            dc=calc.dc_calc,
            rc=-1,
            ac_warnings=ac_warnings,
            ac_errors=ac_errors,
            dc_warnings=dc_warnings,
            dc_errors=dc_errors,
        )

    _run_with_optional_output(verbose, calc.prepare)
    rc = _run_with_optional_output(verbose, calc.run)

    return HybridPowerFlowResult(
        network=network,
        ac_network=network.ac,
        dc_network=network.dc,
        calc=calc,
        ac=calc.ac_calc,
        dc=calc.dc_calc,
        rc=rc,
        ac_warnings=ac_warnings,
        ac_errors=ac_errors,
        dc_warnings=dc_warnings,
        dc_errors=dc_errors,
    )


def print_hybrid_result(result: HybridPowerFlowResult):
    print("\n=== 交直流联合潮流计算结果 ===")
    print(f"节点总数: {result.total_nodes} (AC={len(result.ac_network.nodes)}, DC={len(result.dc_network.nodes)})")

    print("\n1. AC 节点电压:")
    for node in result.ac_network.nodes:
        print(f"   AC 节点 {node.idx}: V={node.voltage:.6f} pu, angle={node.angle:.6f} rad")

    print("\n2. DC 节点电压:")
    for node in result.dc_network.nodes:
        print(f"   DC 节点 {node.idx}: V={node.voltage:.6f} pu")

    print("\n3. AC 发电机:")
    for gen in result.ac_network.generators:
        print(f"   AC 发电机 {gen.idx} 节点 {gen.node}: P={gen.p:.6f} pu, Q={gen.q:.6f} pu")

    print("\n4. DC 发电机:")
    for gen in result.dc_network.generators:
        print(f"   DC 发电机 {gen.idx} 节点 {gen.node}: P={gen.p:.6f} pu, I={gen.current:.6f} pu")

    print("\n5. DC/DC 变流器:")
    for conv in result.dc_network.dcdc_converters:
        print(
            f"   DCDC {conv.idx} {conv.i_node}->{conv.j_node}: "
            f"Pi={conv.i_p:.6f} pu, Pj={conv.j_p:.6f} pu, loss={conv.i_p + conv.j_p:.6f} pu"
        )

    print("\n6. DC/AC 逆变器:")
    for conv in result.network.dcac_converters:
        print(
            f"   DCAC {conv.idx} {conv.dc_node}->AC{conv.ac_node} 控制:{conv.control_type}: "
            f"Pdc={conv.dc_p:.6f} pu, Pac={conv.ac_p:.6f} pu, Qac={conv.ac_q:.6f} pu, "
            f"loss={conv.dc_p + conv.ac_p:.6f} pu"
        )

    print("\n7. AC/AC 柔性互联:")
    for conv in result.network.acac_converters:
        print(
            f"   ACAC {conv.idx} AC{conv.i_node}->AC{conv.j_node} 控制:{conv.control_type}: "
            f"Pi={conv.i_p:.6f} pu, Qi={conv.i_q:.6f} pu, "
            f"Pj={conv.j_p:.6f} pu, Qj={conv.j_q:.6f} pu, "
            f"loss={conv.i_p + conv.j_p:.6f} pu"
        )

    print("\n8. 收敛信息:")
    if result.ac is not None:
        print(f"   AC: {'已收敛' if result.ac.converged else '未收敛'}, iter={result.ac.iterations}, normF={result.ac.normF:.3e}")
    else:
        print("   AC: 文件中无 AC 子网")
    if result.dc is not None:
        print(f"   DC: {'已收敛' if result.dc.converged else '未收敛'}, iter={result.dc.iterations}, normF={result.dc.normF:.3e}")
    else:
        print("   DC: 文件中无 DC 子网")
    print(f"   Hybrid: {'已收敛' if result.converged else '未收敛'}")

    ac_gen = sum(gen.p for gen in result.ac_network.generators)
    ac_load = sum(load.p for load in result.ac_network.loads)
    dc_gen = sum(gen.p for gen in result.dc_network.generators)
    dc_load = sum(load.p for load in result.dc_network.loads)
    print("\n9. 功率汇总:")
    print(f"   AC: gen={ac_gen:.6f} pu, load={ac_load:.6f} pu, diff={ac_gen - ac_load:.6f} pu")
    print(f"   DC: gen={dc_gen:.6f} pu, load={dc_load:.6f} pu, diff={dc_gen - dc_load:.6f} pu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid AC/DC power flow")
    parser.add_argument("file", nargs="?", default=str(DEFAULT_HYBRID_EFILE), help="hybrid E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_hybrid_power_flow(
        args.file,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        verbose=not args.quiet,
        parameter_file=args.para,
    )
    if not args.quiet:
        print_hybrid_result(result)
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
