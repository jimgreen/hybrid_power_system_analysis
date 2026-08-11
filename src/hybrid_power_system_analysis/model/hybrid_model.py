from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

MODEL_DIR = Path(__file__).resolve().parent
for path in (MODEL_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_model import (
    ACACConverter,
    ACAC_SIDE_CONTROL_TYPES,
    ACPowerNetwork,
    acac_legacy_control_label,
)
from dc_model import DCPowerNetwork
from efile_read import _read_efile_rows
from hybrid_array_model import (
    build_hybrid_ppc_from_efile_rows,
    normalize_dcac_ac_control_type,
    normalize_dcac_dc_control_type,
    normalize_dcac_device_type,
    validate_dcac_control_types,
)
from model.multi_energy_model import (
    attach_multi_energy_context,
    build_multi_energy_context_from_rows,
)


class DCACConverter:
    """AC/DC converter with terminal-oriented active-power references.

    Terminal powers are positive from the connected grid into the converter.
    Thus DC-to-AC transfer has ``dc_p > 0`` and ``ac_p < 0``; AC-to-DC
    transfer has the opposite signs. A lossless transfer has
    ``dc_p + ac_p == 0``. ``dev_type`` is retained as equipment metadata and
    never changes this sign convention. AC-side setpoints follow ``ac_p``;
    DC-side active-power setpoints follow ``dc_p``.
    """

    def __init__(
        self,
        idx,
        ac_node,
        dc_node,
        r1,
        r2,
        ac_control_type,
        dc_control_type,
        p_ac_set,
        q_ac_set,
        v_ac_set,
        v_dc_set,
        run_stat=1,
        dev_type="DCACConverter",
        p_dc_set=0.0,
    ):
        self.idx = idx
        self.ac_node = ac_node
        self.dc_node = dc_node
        self.r1 = r1
        self.r2 = r2
        self.ac_control_type = normalize_dcac_ac_control_type(ac_control_type)
        self.dc_control_type = normalize_dcac_dc_control_type(dc_control_type)
        validate_dcac_control_types(self.ac_control_type, self.dc_control_type)
        self.p_ac_set = p_ac_set
        self.p_dc_set = p_dc_set
        self.q_ac_set = q_ac_set
        self.v_ac_set = v_ac_set
        self.v_dc_set = v_dc_set
        self.run_stat = run_stat
        self.dev_type = normalize_dcac_device_type(dev_type)
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
    fluid_networks: Dict[str, object] = field(default_factory=dict)
    energy_couplings: List[object] = field(default_factory=list)
    multi_energy: Optional[object] = None

    def __post_init__(self):
        self.ac.acac_converters = self.acac_converters

    @classmethod
    def read_from_file(cls, file_name) -> "HybridPowerNetwork":
        rows = _read_efile_rows(file_name)
        network, _ppc = build_hybrid_ppc_from_efile_rows(file_name, rows)
        attach_multi_energy_context(
            network,
            build_multi_energy_context_from_rows(rows, source=file_name),
        )
        return network

    @property
    def total_nodes(self) -> int:
        return len(self.ac.nodes) + len(self.dc.nodes)

    @property
    def total_energy_nodes(self) -> int:
        return self.total_nodes + sum(
            len(network.nodes) for network in self.fluid_networks.values()
        )

    def _physical_ac_angle_reference_node_ids(self):
        ac_nodes = {int(node.idx): node for node in self.ac.nodes}
        dc_nodes = {int(node.idx): node for node in self.dc.nodes}
        reference_ids = set()
        for conv in self.dcac_converters:
            ac_node = ac_nodes.get(int(conv.ac_node))
            dc_node = dc_nodes.get(int(conv.dc_node))
            if (
                conv.run_stat == 1
                and ac_node is not None
                and dc_node is not None
                and ac_node.run_stat == 1
                and dc_node.run_stat == 1
                and str(conv.ac_control_type).upper() == "PH"
            ):
                reference_ids.add(int(ac_node.idx))
        for conv in self.acac_converters:
            i_node = ac_nodes.get(int(conv.i_node))
            j_node = ac_nodes.get(int(conv.j_node))
            if (
                conv.run_stat != 1
                or i_node is None
                or j_node is None
                or i_node.run_stat != 1
                or j_node.run_stat != 1
            ):
                continue
            if str(conv.i_control_type).upper() == "PH":
                reference_ids.add(int(i_node.idx))
            if str(conv.j_control_type).upper() == "PH":
                reference_ids.add(int(j_node.idx))
        return reference_ids

    def topo(self):
        previous_references = getattr(
            self.ac,
            "_hybrid_angle_reference_node_ids",
            None,
        )
        self.ac._hybrid_angle_reference_node_ids = (
            self._physical_ac_angle_reference_node_ids()
        )
        try:
            self._topo_with_hybrid_references()
        finally:
            if previous_references is None:
                self.ac.__dict__.pop("_hybrid_angle_reference_node_ids", None)
            else:
                self.ac._hybrid_angle_reference_node_ids = previous_references

    def _topo_with_hybrid_references(self):
        grids = [grid for grid in (self.ac, self.dc) if grid.nodes]
        previous_defer = {
            id(grid): getattr(grid, "_defer_operational_island_filter", None)
            for grid in grids
        }
        try:
            for grid in grids:
                grid._defer_operational_island_filter = True
                grid.topo()
        finally:
            for grid in grids:
                previous = previous_defer[id(grid)]
                if previous is None:
                    grid.__dict__.pop("_defer_operational_island_filter", None)
                else:
                    grid._defer_operational_island_filter = previous

        ac_operational, dc_operational = self._build_hybrid_topo(
            evaluate_operational=True,
        )
        overrides = ((self.ac, ac_operational), (self.dc, dc_operational))
        previous_overrides = {
            id(grid): getattr(grid, "_operational_island_ids_override", None)
            for grid, _island_ids in overrides
            if grid.nodes
        }
        try:
            for grid, island_ids in overrides:
                if not grid.nodes:
                    continue
                grid._operational_island_ids_override = island_ids
                grid.topo()
        finally:
            for grid, _island_ids in overrides:
                if not grid.nodes:
                    continue
                previous = previous_overrides[id(grid)]
                if previous is None:
                    grid.__dict__.pop("_operational_island_ids_override", None)
                else:
                    grid._operational_island_ids_override = previous
        self._build_hybrid_topo()

    def _local_reference_island_ids(self):
        ac_ids = {id(island) for island in self.ac.islands if island.is_alive}
        dc_ids = {id(island) for island in self.dc.islands if island.is_alive}
        for conv in self.dcac_converters:
            ac_node = self.ac.node_dict.get(conv.ac_node)
            dc_node = self.dc.node_dict.get(conv.dc_node)
            if (
                conv.run_stat != 1
                or ac_node is None
                or dc_node is None
                or ac_node.run_stat != 1
                or dc_node.run_stat != 1
                or ac_node.isl_obj is None
                or dc_node.isl_obj is None
            ):
                continue
            if str(conv.ac_control_type).upper() == "PH":
                ac_ids.add(id(ac_node.isl_obj))
            if str(conv.dc_control_type).upper() == "V":
                dc_ids.add(id(dc_node.isl_obj))
        for conv in self.acac_converters:
            i_node = self.ac.node_dict.get(conv.i_node)
            j_node = self.ac.node_dict.get(conv.j_node)
            if conv.run_stat != 1 or i_node is None or j_node is None:
                continue
            if i_node.run_stat != 1 or j_node.run_stat != 1:
                continue
            if i_node.isl_obj is not None and str(conv.i_control_type).upper() == "PH":
                ac_ids.add(id(i_node.isl_obj))
            if j_node.isl_obj is not None and str(conv.j_control_type).upper() == "PH":
                ac_ids.add(id(j_node.isl_obj))
        return ac_ids, dc_ids

    def _build_hybrid_topo(self, evaluate_operational: bool = False):
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

        ac_local_ids, dc_local_ids = self._local_reference_island_ids()

        def eligible_link(
            physical_alive,
            left_island,
            right_island,
            left_ids,
            right_ids,
            left_node,
            right_node,
        ):
            if not physical_alive:
                return False
            if evaluate_operational:
                return id(left_island) in left_ids and id(right_island) in right_ids
            return bool(left_node.is_alive and right_node.is_alive)

        for conv in self.dcac_converters:
            ac_node = self.ac.node_dict.get(conv.ac_node)
            dc_node = self.dc.node_dict.get(conv.dc_node)
            conv.ac_node_obj = ac_node
            conv.dc_node_obj = dc_node
            conv.ac_isl_obj = None if ac_node is None else ac_node.isl_obj
            conv.dc_isl_obj = None if dc_node is None else dc_node.isl_obj
            physical_alive = (
                conv.run_stat == 1
                and ac_node is not None
                and dc_node is not None
                and getattr(ac_node, "run_stat", 0) == 1
                and getattr(dc_node, "run_stat", 0) == 1
                and conv.ac_isl_obj is not None
                and conv.dc_isl_obj is not None
            )
            link_alive = eligible_link(
                physical_alive,
                conv.ac_isl_obj,
                conv.dc_isl_obj,
                ac_local_ids,
                dc_local_ids,
                ac_node,
                dc_node,
            )
            conv.is_alive = link_alive
            if link_alive:
                union(conv.ac_isl_obj, conv.dc_isl_obj)

        for conv in self.acac_converters:
            i_node = self.ac.node_dict.get(conv.i_node)
            j_node = self.ac.node_dict.get(conv.j_node)
            conv.i_node_obj = i_node
            conv.j_node_obj = j_node
            conv.i_isl_obj = None if i_node is None else i_node.isl_obj
            conv.j_isl_obj = None if j_node is None else j_node.isl_obj
            physical_alive = (
                conv.run_stat == 1
                and i_node is not None
                and j_node is not None
                and getattr(i_node, "run_stat", 0) == 1
                and getattr(j_node, "run_stat", 0) == 1
                and conv.i_isl_obj is not None
                and conv.j_isl_obj is not None
            )
            link_alive = eligible_link(
                physical_alive,
                conv.i_isl_obj,
                conv.j_isl_obj,
                ac_local_ids,
                ac_local_ids,
                i_node,
                j_node,
            )
            conv.is_alive = link_alive
            if link_alive:
                union(conv.i_isl_obj, conv.j_isl_obj)

        for conv in self.dc.dcdc_converters:
            i_node = getattr(conv, "i_node_obj", None)
            j_node = getattr(conv, "j_node_obj", None)
            physical_alive = (
                getattr(conv, "run_stat", 0) == 1
                and i_node is not None
                and j_node is not None
                and getattr(i_node, "run_stat", 0) == 1
                and getattr(j_node, "run_stat", 0) == 1
                and i_node.isl_obj is not None
                and j_node.isl_obj is not None
            )
            link_alive = eligible_link(
                physical_alive,
                None if i_node is None else i_node.isl_obj,
                None if j_node is None else j_node.isl_obj,
                dc_local_ids,
                dc_local_ids,
                i_node,
                j_node,
            )
            conv.is_alive = link_alive
            if link_alive:
                union(i_node.isl_obj, j_node.isl_obj)

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

        if evaluate_operational:
            return self._evaluate_hybrid_operational_islands(ac_local_ids, dc_local_ids)
        return set(), set()

    def _evaluate_hybrid_operational_islands(self, ac_local_ids, dc_local_ids):
        ac_operational = set()
        dc_operational = set()
        ac_auto_balance_ids = {
            id(gen) for gen in getattr(self.ac, "_auto_slack_generators", ())
        }
        for hybrid_island in self.hybrid_islands:
            has_reference = any(
                id(island) in ac_local_ids for island in hybrid_island.ac_islands
            ) or any(
                id(island) in dc_local_ids for island in hybrid_island.dc_islands
            )
            generator_count = 0
            balance_count = 0
            load_count = 0

            for gen in self.ac.generators:
                node = getattr(gen, "node_obj", None)
                if (
                    getattr(gen, "run_stat", 0) != 1
                    or node is None
                    or getattr(node, "run_stat", 0) != 1
                    or node.isl_obj is None
                    or node.isl_obj.hybrid_isl_obj is not hybrid_island
                ):
                    continue
                generator_count += 1
                if (
                    str(getattr(gen, "control_type", "")).upper()
                    in {"V", "SLACK", "PH"}
                    or id(gen) in ac_auto_balance_ids
                ):
                    balance_count += 1

            for gen in self.dc.generators:
                node = getattr(gen, "node_obj", None)
                if (
                    getattr(gen, "run_stat", 0) != 1
                    or node is None
                    or getattr(node, "run_stat", 0) != 1
                    or node.isl_obj is None
                    or node.isl_obj.hybrid_isl_obj is not hybrid_island
                ):
                    continue
                generator_count += 1
                if str(getattr(gen, "control_type", "")).upper() == "V":
                    balance_count += 1

            for loads, node_dict in (
                (self.ac.loads, self.ac.node_dict),
                (self.dc.loads, self.dc.node_dict),
            ):
                for load in loads:
                    node = getattr(load, "node_obj", None) or node_dict.get(int(load.node))
                    if (
                        getattr(load, "run_stat", 0) == 1
                        and node is not None
                        and getattr(node, "run_stat", 0) == 1
                        and node.isl_obj is not None
                        and node.isl_obj.hybrid_isl_obj is hybrid_island
                    ):
                        load_count += 1

            source_only = (
                balance_count == 1 and generator_count == 1 and load_count == 0
            )
            hybrid_island.is_alive = (
                has_reference and balance_count > 0 and not source_only
            )
            if hybrid_island.is_alive:
                ac_operational.update(island.idx for island in hybrid_island.ac_islands)
                dc_operational.update(island.idx for island in hybrid_island.dc_islands)
            for conv in (
                *hybrid_island.dcac_converters,
                *hybrid_island.dcdc_converters,
                *hybrid_island.acac_converters,
            ):
                conv.is_alive = bool(conv.is_alive and hybrid_island.is_alive)

        return ac_operational, dc_operational

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
        for tr in getattr(self.ac, "three_winding_transformers", ()):
            if tr.run_stat:
                add(tr.i_node)
                add(tr.j_node)
                add(tr.k_node)
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
            ac_ctrl = getattr(conv, "ac_control_type", "NONE")
            dc_ctrl = getattr(conv, "dc_control_type", "NONE")
            try:
                ac_ctrl, dc_ctrl = validate_dcac_control_types(ac_ctrl, dc_ctrl)
            except ValueError as exc:
                errors.append(
                    f"DCACConverter[{conv.idx}] {getattr(conv, 'name', '')} {exc}"
                )
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
            i_ctrl = str(getattr(conv, "i_control_type", "PQ")).upper()
            j_ctrl = str(getattr(conv, "j_control_type", "PQ")).upper()
            if i_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} i_control_type {i_ctrl} 不支持")
            if j_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                errors.append(f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} j_control_type {j_ctrl} 不支持")
            if i_ctrl in ACAC_SIDE_CONTROL_TYPES and j_ctrl in ACAC_SIDE_CONTROL_TYPES:
                try:
                    acac_legacy_control_label(i_ctrl, j_ctrl)
                except ValueError:
                    errors.append(
                        f"ACACConverter[{conv.idx}] {getattr(conv, 'name', '')} 控制组合 "
                        f"({i_ctrl}, {j_ctrl}) 当前程序不支持"
                    )
        return errors
