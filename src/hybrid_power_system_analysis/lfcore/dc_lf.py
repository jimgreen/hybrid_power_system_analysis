"""
为了在不引入小电阻、最小二乘或后处理分组的情况下处理零阻抗支路，我们提出一种节点电位变量法。该方法在原始网络中引入辅助变量（节点电位ϕ），用ϕ的差值表示零阻抗支路电流，从而：
1、自动实现并联支路等分流（同一节点对间的ϕ差相同）。
2、通过电压相等约束和ϕ定标使方程组闭合，无需额外假设（如无环流）。
3、所有变量（节点电压V和节点电位ϕ）在全局牛顿-拉夫逊迭代中统一求解。

核心思想：
对于零阻抗支路构成的每个连通分量：
1、引入一组节点电位变量 ϕ，每个原始节点对应一个。
2、每条零阻抗支路电流Iij=ϕi−ϕj，方向按预设，例如从 i到 j
3、节点电流平衡由功率平衡方程隐式满足，无需额外方程。
4、电压相等约束 Vi=Vj,只需取生成树上的Nc-1条（独立）。
5、在每个分量内固定一个ϕ=0，消除平移自由度。

这样，总变量数 = N+Nϕ（Nϕ为所有零阻抗节点总数），总方程数 = N（功率平衡/电压给定）+ ∑(nc−1)+C（ϕ定标），两者相等，系统可解。


预处理：识别零阻抗支路的连通分量，为每个分量建立局部节点编号，并选择生成树（用于电压约束）。
变量定义：每个原始电压节点一个V变量；每个零阻抗节点一个ϕ变量（按分量连续存储）。

方程系统：
1、未知节点：功率平衡方程，包含电阻支路、恒电流、恒功率，以及零阻抗支路电流项（用ϕ表示）。
2、已知节点：电压给定方程。
3、零阻抗支路：每个分量取生成树上的支路，添加电压相等方程 Vi=Vj。
4、ϕ定标：每个分量固定一个ϕ=0。
雅可比矩阵：计算所有偏导数，包括V对V，V对ϕ,以及电压约束和定标方程。
迭代求解：牛顿-拉夫逊法，收敛后得到所有变量。
结果计算：零阻抗支路电流由ϕ差直接给出，并联支路电流自动相等。

优点
1、无需小电阻近似，精确处理。
2、无需后处理分组，所有变量统一迭代。
3、并联支路电流自动相等
4、环网电流由ϕ的分布决定，具有最小二乘意义，但无需显式最小二乘。

此方案满足用户所有要求：不引入最小二乘法（数学上等价但实现为线性系统），不做分组和分段求解，不假设连支电流为零，且能处理任意零阻抗支路拓扑。
"""

import argparse
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

try:
    from scipy.sparse.linalg import use_solver as _scipy_use_solver
    _scipy_use_solver(useUmfpack=False)
except Exception:
    pass
from collections import deque
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from _sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
except ImportError:  # pragma: no cover - package import path
    from ._sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from model import topology as network_topology
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BUS_COLS as DC_BUS_COLS,
    CTRL_I as DC_CTRL_I,
    CTRL_P as DC_CTRL_P,
    CTRL_SLACK as DC_CTRL_SLACK,
    CTRL_V as DC_CTRL_V,
    DCDC_COLS as DC_DCDC_COLS,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    build_dc_network_from_ppc,
    build_dc_ppc_from_network,
)
from model.ppc_topology import build_dc_ppc_with_topology_from_e_file, ensure_dc_ppc_topology
try:
    from lfcore.common import (
        find_spanning_tree_edges,
        normalize_result_mode as _normalize_lf_result_mode,
    )
except ImportError:  # pragma: no cover - direct script import path
    from common import (
        find_spanning_tree_edges,
        normalize_result_mode as _normalize_lf_result_mode,
    )
try:
    from lfcore.solver_common import (
        OPTIONAL_SPARSE_MISSING as _OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        resolve_linear_solver as _resolve_linear_solver,
    )
except ImportError:  # pragma: no cover - direct script import path
    from solver_common import (
        OPTIONAL_SPARSE_MISSING as _OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        resolve_linear_solver as _resolve_linear_solver,
    )


@dataclass
class DCLFResult:
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    zero_branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    breakers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    dcdc_converters: Dict[str, SimpleNamespace] = field(default_factory=dict)
    generators: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)


class _DenseNodePositionMap:
    """Dictionary-like node-id lookup backed by the dense PPC node map."""

    __slots__ = ("lookup", "node_ids", "positions")

    def __init__(self, lookup, node_ids, positions):
        self.lookup = lookup
        self.node_ids = node_ids
        self.positions = positions

    def get(self, key, default=None):
        key = int(key)
        if self.lookup is not None and 0 <= key < self.lookup.size:
            pos = int(self.lookup[key])
            return pos if pos >= 0 else default
        return default

    def __contains__(self, key):
        return self.get(key, -1) >= 0

    def __getitem__(self, key):
        value = self.get(key, None)
        if value is None:
            raise KeyError(key)
        return value

    def items(self):
        for node_id, pos in zip(self.node_ids, self.positions):
            yield int(node_id), int(pos)

    def keys(self):
        for node_id in self.node_ids:
            yield int(node_id)

    def values(self):
        for pos in self.positions:
            yield int(pos)

    def __len__(self):
        return int(self.node_ids.size)


def load_dc_ppc_from_e_file(file_name) -> Dict:
    """Read a DC E file into PPC with topology arrays attached."""
    return build_dc_ppc_with_topology_from_e_file(file_name)


def _dc_network_from_ppc(ppc):
    ensure_dc_ppc_topology(ppc)
    network = build_dc_network_from_ppc(ppc)
    network_topology.apply_dc_topology_arrays(network, ppc["_topology_arrays"])
    return network


class DCPowerFlowCalc:
    """直流潮流计算器，使用节点电压、零阻抗 phi 和 DCDC 端口功率统一求解。

    状态向量布局为 ``V``、``phi``、``Pdc_i/Pdc_j``。节点功率平衡、定压
    节点约束、零阻抗电压相等约束、phi 参考约束和 DC-DC 控制/损耗方程
    在同一个 Newton 系统中求解。
    """

    def __init__(
        self,
        network,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        island=None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        keep_node_objects: bool = True,
        linear_solver: str = "pyklu",
        result_mode: str = "full",
        verbose: bool = False,
    ):
        self._network_writeback = None
        if isinstance(network, dict) and network.get("format") == "dc_ppc_v1":
            self.ppc = network
        elif island is None and hasattr(network, "nodes"):
            self._network_writeback = network
            existing_ppc = getattr(network, "ppc", None)
            if isinstance(existing_ppc, dict) and existing_ppc.get("format") == "dc_ppc_v1":
                self.ppc = existing_ppc
            else:
                self.ppc = build_dc_ppc_from_network(network)
            if hasattr(network, "source") and "source" not in self.ppc:
                self.ppc["source"] = str(getattr(network, "source"))
        else:
            raise ValueError("DCPowerFlowCalc requires dc_ppc_v1 or DCPowerNetwork input")
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.min_voltage = self.params.min_voltage
        self.target_island = island
        # 用户传入的求解器名原样保留，便于上层日志/测试断言；实际 callable
        # 由 _resolve_linear_solver 决定，未安装时回退 SuperLU。
        self.linear_solver = str(linear_solver or "pyklu").strip().lower()
        self._linear_solver_resolved, self._linear_solver_fn = _resolve_linear_solver(self.linear_solver)
        self.result_mode = self._normalize_result_mode(result_mode)
        self.keep_node_objects = False
        self._cache_csr_jacobian_pattern = self.result_mode == "full"
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.verbose = bool(verbose)
        self.result: Dict = {}
        self.lf_result = None
        self.G = None
        self.x = np.array([], dtype=np.float64)

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        return _normalize_lf_result_mode(result_mode, "DC")

    def _alive_node_lookup_array(self):
        """Return a dense node-id to active solver-position lookup when possible."""
        cached = getattr(self, "_alive_node_lookup", None)
        if cached is not None:
            return cached
        if self.alive_node_ids.size == 0:
            self._alive_node_lookup = np.array([], dtype=np.int32)
            return self._alive_node_lookup
        if np.any(self.alive_node_ids < 0):
            return None
        max_node_id = int(np.max(self.alive_node_ids))
        lookup = np.full(max_node_id + 1, -1, dtype=np.int32)
        for node_id, pos in self.alive_node_dict.items():
            lookup[int(node_id)] = int(pos)
        self._alive_node_lookup = lookup
        return lookup

    @staticmethod
    def _map_nodes_with_lookup(node_ids, lookup):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        mapped = np.full(node_ids.shape, -1, dtype=np.int32)
        if lookup is None or lookup.size == 0:
            return mapped
        valid = (node_ids >= 0) & (node_ids < lookup.size)
        mapped[valid] = lookup[node_ids[valid]]
        return mapped

    def _topology_terminal_solver_positions(self, key: str, count: int, *, require_distinct_solver: bool = False):
        if self.keep_node_objects:
            return None
        topology = getattr(self, "_ppc_topology", None)
        device_topology = getattr(topology, "devices", {}).get(key) if topology is not None else None
        solver_lookup = getattr(self, "_bus_pos_to_solver_pos", None)
        if device_topology is None or solver_lookup is None or solver_lookup.size == 0:
            return None
        alive = np.asarray(getattr(device_topology, "alive_mask", np.zeros(int(count), dtype=bool)), dtype=bool)
        if alive.size != int(count):
            return None
        i_bus = np.asarray(device_topology.i_bus_pos, dtype=np.int32)
        j_bus = np.asarray(device_topology.j_bus_pos, dtype=np.int32)
        if i_bus.size != int(count) or j_bus.size != int(count):
            return None
        i_solver = np.full(int(count), -1, dtype=np.int32)
        j_solver = np.full(int(count), -1, dtype=np.int32)
        i_valid = (i_bus >= 0) & (i_bus < solver_lookup.size)
        j_valid = (j_bus >= 0) & (j_bus < solver_lookup.size)
        if np.any(i_valid):
            i_solver[i_valid] = solver_lookup[i_bus[i_valid]]
        if np.any(j_valid):
            j_solver[j_valid] = solver_lookup[j_bus[j_valid]]
        mask = alive & (i_solver >= 0) & (j_solver >= 0)
        if require_distinct_solver:
            mask &= i_solver != j_solver
        rows = np.nonzero(mask)[0].astype(np.int32)
        return rows, i_solver[mask].astype(np.int32, copy=False), j_solver[mask].astype(np.int32, copy=False)

    def _topology_single_solver_positions(self, key: str, count: int):
        if self.keep_node_objects:
            return None
        topology = getattr(self, "_ppc_topology", None)
        device_topology = getattr(topology, "devices", {}).get(key) if topology is not None else None
        solver_lookup = getattr(self, "_bus_pos_to_solver_pos", None)
        if device_topology is None or solver_lookup is None or solver_lookup.size == 0:
            return None
        alive = np.asarray(getattr(device_topology, "alive_mask", np.zeros(int(count), dtype=bool)), dtype=bool)
        if alive.size != int(count):
            return None
        bus_pos = np.asarray(device_topology.bus_pos, dtype=np.int32)
        if bus_pos.size != int(count):
            return None
        solver_pos = np.full(int(count), -1, dtype=np.int32)
        valid = (bus_pos >= 0) & (bus_pos < solver_lookup.size)
        if np.any(valid):
            solver_pos[valid] = solver_lookup[bus_pos[valid]]
        mask = alive & (solver_pos >= 0)
        rows = np.nonzero(mask)[0].astype(np.int32)
        return rows, solver_pos[mask].astype(np.int32, copy=False)

    def _should_eliminate_dcdc_j_power(self) -> bool:
        return bool(not self.keep_node_objects)

    def _dcdc_j_power_from_loss(self, pi, vi, vj):
        pi = np.asarray(pi, dtype=np.float64)
        vi = np.asarray(vi, dtype=np.float64)
        vj = np.asarray(vj, dtype=np.float64)
        vi2 = vi * vi
        vj2 = vj * vj
        a = self.dcdc_r2 * vi2
        b = vi2 * vj2
        c = vj2 * pi * (vi2 - self.dcdc_r1 * pi)
        disc = np.maximum(b * b + 4.0 * a * c, 0.0)
        sqrt_disc = np.sqrt(disc)
        pj_linear = np.divide(-c, b, out=np.zeros_like(c), where=np.abs(b) > 1e-12)
        pj_quad = np.divide(b - sqrt_disc, 2.0 * a, out=pj_linear.copy(), where=np.abs(a) > 1e-12)

        denom = b - 2.0 * self.dcdc_r2 * pj_quad * vi2
        safe = np.abs(denom) > 1e-12
        dpj_dpi = np.divide(
            -vj2 * (vi2 - 2.0 * self.dcdc_r1 * pi),
            denom,
            out=np.zeros_like(pi),
            where=safe,
        )
        df_dvi = 2.0 * vi * vj2 * (pi + pj_quad) - 2.0 * self.dcdc_r2 * pj_quad * pj_quad * vi
        df_dvj = 2.0 * vj * vi2 * (pi + pj_quad) - 2.0 * self.dcdc_r1 * pi * pi * vj
        dpj_dvi = np.divide(-df_dvi, denom, out=np.zeros_like(pi), where=safe)
        dpj_dvj = np.divide(-df_dvj, denom, out=np.zeros_like(pi), where=safe)
        return pj_quad, dpj_dpi, dpj_dvi, dpj_dvj

    def _prepare_direct_ppc_topology(self):
        """Build active DC solver-node mapping directly from dc_ppc_v1 arrays."""
        ppc = self.ppc
        topology = ppc["_topology_arrays"]
        self._ppc_topology = topology
        if not np.any(topology.node_alive_mask):
            self.alive_nodes = []
            self.alive_node_dict = {}
            self.alive_node_ids = np.array([], dtype=np.int32)
            self.N = 0
            self._alive_node_lookup = np.array([], dtype=np.int32)
            return

        if self.keep_node_objects:
            solver_node_ids = topology.node_ids[topology.node_alive_mask].astype(np.int32, copy=False)
            self.alive_node_dict = {int(node_id): int(pos) for pos, node_id in enumerate(solver_node_ids)}
            self.alive_node_ids = solver_node_ids.copy()
            self.N = int(solver_node_ids.size)
            if solver_node_ids.size and np.all(solver_node_ids >= 0):
                self._alive_node_lookup = np.full(int(solver_node_ids.max()) + 1, -1, dtype=np.int32)
                self._alive_node_lookup[solver_node_ids.astype(np.intp)] = np.arange(self.N, dtype=np.int32)
            else:
                self._alive_node_lookup = np.array([], dtype=np.int32)
        else:
            active_bus_pos = np.flatnonzero(topology.bus_alive_mask).astype(np.int32)
            self._active_bus_pos = active_bus_pos
            self._bus_pos_to_solver_pos = np.full(topology.bus_alive_mask.size, -1, dtype=np.int32)
            if active_bus_pos.size:
                self._bus_pos_to_solver_pos[active_bus_pos] = np.arange(active_bus_pos.size, dtype=np.int32)
            if active_bus_pos.size:
                node_bus_pos = topology.node_to_bus_pos.astype(np.int32, copy=False)
                node_mask = topology.node_alive_mask & (node_bus_pos >= 0)
                node_pos = np.flatnonzero(node_mask).astype(np.int32, copy=False)
                alive_pos_values = self._bus_pos_to_solver_pos[node_bus_pos[node_pos]]
                valid = alive_pos_values >= 0
                node_pos = node_pos[valid]
                alive_pos_values = alive_pos_values[valid].astype(np.int32, copy=False)
                self.alive_node_ids = topology.node_ids[node_pos].astype(np.int32, copy=True)
                self.N = int(active_bus_pos.size)
                if self.alive_node_ids.size and np.all(self.alive_node_ids >= 0):
                    self._alive_node_lookup = np.full(int(self.alive_node_ids.max()) + 1, -1, dtype=np.int32)
                    self._alive_node_lookup[self.alive_node_ids.astype(np.intp)] = alive_pos_values
                else:
                    self._alive_node_lookup = np.array([], dtype=np.int32)
                self.alive_node_dict = _DenseNodePositionMap(
                    self._alive_node_lookup,
                    self.alive_node_ids,
                    alive_pos_values,
                )
            else:
                self.alive_node_dict = {}
                self.alive_node_ids = np.array([], dtype=np.int32)
                self.N = 0
                self._alive_node_lookup = np.array([], dtype=np.int32)
                self._active_bus_pos = np.array([], dtype=np.int32)
                self._bus_pos_to_solver_pos = np.array([], dtype=np.int32)

        # Direct ppc result/writeback paths use alive_node_dict and result arrays; constructing
        # SimpleNamespace node facades here is pure cold-start overhead for large ppc cases.
        self.alive_nodes = []

    def prepare(self):
        """预处理：合并带电拓扑岛，初始化参数并定义变量/方程索引。"""
        self._prepare_from_ppc()

    def _prepare_from_ppc(self):
        """Prepare a Newton system from an already arrayized DC ppc dictionary."""
        ensure_dc_ppc_topology(self.ppc)
        self._prepare_direct_ppc_topology()

        if self.N == 0:
            raise ValueError("电网中没有活节点")

        self.P_const = np.zeros(self.N, dtype=np.float64)   # 注入为正：P型发电机 - P型负荷
        self.I_shunt = np.zeros(self.N, dtype=np.float64)   # 消耗为正：负荷电流 - 发电电流
        self.slack_gen_info = {}
        node_lookup = self._alive_node_lookup_array()

        # ---------- 1. 数据预处理 ----------
        ppc = self.ppc

        branch = ppc["branch"]
        if branch.size:
            branch_positions = self._topology_terminal_solver_positions("branch", branch.shape[0])
            if branch_positions is not None:
                self.branch_idx, self.branch_i, self.branch_j = branch_positions
                branch_mask = np.zeros(branch.shape[0], dtype=bool)
                branch_mask[self.branch_idx] = True
            else:
                branch_i_all = self._map_nodes_with_lookup(branch[:, DC_BRANCH_COLS["i_node"]], node_lookup)
                branch_j_all = self._map_nodes_with_lookup(branch[:, DC_BRANCH_COLS["j_node"]], node_lookup)
                branch_mask = (
                    (branch[:, DC_BRANCH_COLS["run_stat"]] == 1)
                    & (branch_i_all >= 0)
                    & (branch_j_all >= 0)
                )
                self.branch_idx = np.nonzero(branch_mask)[0].astype(np.int32)
                self.branch_i = branch_i_all[branch_mask].astype(np.int32, copy=False)
                self.branch_j = branch_j_all[branch_mask].astype(np.int32, copy=False)
            self.branch_r = branch[branch_mask, DC_BRANCH_COLS["r"]].astype(np.float64, copy=False)
        else:
            branch_mask = np.array([], dtype=bool)
            self.branch_idx = self.branch_i = self.branch_j = np.array([], dtype=np.int32)
            self.branch_r = np.array([], dtype=np.float64)
        if np.any(self.branch_r <= 0.0):
            bad = int(np.where(self.branch_r <= 0.0)[0][0])
            raise ValueError(f"支路电阻必须为正数: r={self.branch_r[bad]}")
        load = ppc["load"]
        if load.size:
            load_positions = self._topology_single_solver_positions("load", load.shape[0])
            if load_positions is not None:
                load_rows, load_pos = load_positions
                load_mask = np.zeros(load.shape[0], dtype=bool)
                load_mask[load_rows] = True
            else:
                load_pos_all = self._map_nodes_with_lookup(load[:, DC_LOAD_COLS["node"]], node_lookup)
                load_mask = (load[:, DC_LOAD_COLS["run_stat"]] == 1) & (load_pos_all >= 0)
                load_pos = load_pos_all[load_mask]
            load_pbase = load[load_mask, DC_LOAD_COLS["pbase"]].astype(np.float64, copy=False)
            load_pv0 = load_pbase * load[load_mask, DC_LOAD_COLS["pv0"]]
            load_pv1 = load_pbase * load[load_mask, DC_LOAD_COLS["pv1"]]
            load_pv2 = load_pbase * load[load_mask, DC_LOAD_COLS["pv2"]]
            self.P_const += np.bincount(load_pos, weights=-load_pv0, minlength=self.P_const.size)
            self.I_shunt += np.bincount(load_pos, weights=load_pv1, minlength=self.I_shunt.size)
            nz_load = load_pv2 != 0.0
            load_nodes_arr = load_pos[nz_load].astype(np.int32, copy=False)
            load_g_arr = load_pv2[nz_load]
        else:
            load_nodes_arr = np.array([], dtype=np.int32)
            load_g_arr = np.array([], dtype=np.float64)

        gen = ppc["gen"]
        if gen.size:
            gen_positions = self._topology_single_solver_positions("gen", gen.shape[0])
            if gen_positions is not None:
                gen_rows, gen_pos = gen_positions
                gen_mask = np.zeros(gen.shape[0], dtype=bool)
                gen_mask[gen_rows] = True
            else:
                gen_pos_all = self._map_nodes_with_lookup(gen[:, DC_GEN_COLS["node"]], node_lookup)
                gen_mask = (gen[:, DC_GEN_COLS["run_stat"]] == 1) & (gen_pos_all >= 0)
                gen_pos = gen_pos_all[gen_mask]
                gen_rows = np.nonzero(gen_mask)[0]
            gen_active = gen[gen_mask]
            gen_ctrl = gen_active[:, DC_GEN_COLS["control_type"]].astype(np.int8, copy=False)
            p_mask = gen_ctrl == DC_CTRL_P
            i_mask = gen_ctrl == DC_CTRL_I
            v_mask = gen_ctrl == DC_CTRL_V
            if np.any(p_mask):
                self.P_const += np.bincount(
                    gen_pos[p_mask],
                    weights=gen_active[p_mask, DC_GEN_COLS["p_set"]],
                    minlength=self.P_const.size,
                )
            if np.any(i_mask):
                self.I_shunt += np.bincount(
                    gen_pos[i_mask],
                    weights=-gen_active[i_mask, DC_GEN_COLS["i_set"]],
                    minlength=self.I_shunt.size,
                )
            if np.any(v_mask):
                for row_idx, node in zip(gen_rows[v_mask], gen_pos[v_mask]):
                    self.slack_gen_info.setdefault(int(node), []).append(int(row_idx))
            bad_ctrl = ~(p_mask | i_mask | v_mask)
            if np.any(bad_ctrl):
                raise ValueError(f"未知发电机控制类型: {gen_ctrl[int(np.where(bad_ctrl)[0][0])]}")
        self.alive_loads = []
        self.alive_generators = []

        # G 矩阵只包含线性电导；恒功率、恒电流和二次负荷项分开放入方程。
        rows_parts = []
        cols_parts = []
        data_parts = []
        if load_nodes_arr.size:
            rows_parts.append(load_nodes_arr)
            cols_parts.append(load_nodes_arr)
            data_parts.append(load_g_arr)

        if self.branch_idx.size:
            branch_g = 1.0 / self.branch_r
            rows_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_i, self.branch_j)))
            cols_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_j, self.branch_i)))
            data_parts.append(np.concatenate((branch_g, branch_g, -branch_g, -branch_g)))

        if rows_parts:
            G_rows = np.concatenate(rows_parts)
            G_cols = np.concatenate(cols_parts)
            G_data = np.concatenate(data_parts)
            G = csr_matrix((G_data, (G_rows, G_cols)), shape=(self.N, self.N))
        else:
            G = csr_matrix((self.N, self.N), dtype=np.float64)
        G.sum_duplicates()
        self.G = G

        # ---------- 2. 零阻抗支路处理（节点电位法） ----------
        self.zero_edges = []

        def append_zero_edges(table_key, table, cols, code):
            if table is None or not table.size:
                return
            positions = self._topology_terminal_solver_positions(
                table_key,
                table.shape[0],
                require_distinct_solver=True,
            )
            if positions is not None:
                rows, i_pos, j_pos = positions
            else:
                i_all = self._map_nodes_with_lookup(table[:, cols["i_node"]], node_lookup)
                j_all = self._map_nodes_with_lookup(table[:, cols["j_node"]], node_lookup)
                mask = (
                    (table[:, cols["run_stat"]] == 1)
                    & (i_all >= 0)
                    & (j_all >= 0)
                    & (i_all != j_all)
                )
                if "status" in cols:
                    mask &= table[:, cols["status"]] == 1
                rows = np.nonzero(mask)[0].astype(np.int32)
                i_pos = i_all[mask].astype(np.int32, copy=False)
                j_pos = j_all[mask].astype(np.int32, copy=False)
            if rows.size:
                self.zero_edges.extend(
                    (code, int(dev_idx), int(i_node), int(j_node))
                    for dev_idx, i_node, j_node in zip(rows, i_pos, j_pos)
                )

        append_zero_edges("zero_branch", self.ppc["zero_branch"], DC_ZERO_BRANCH_COLS, "Z")
        append_zero_edges("switch", self.ppc["switch"], DC_SWITCH_COLS, "S")
        append_zero_edges("break", self.ppc.get("break"), DC_BREAK_COLS, "B")

        zero_adj = {}
        for edge_idx, (_, _, i_node, j_node) in enumerate(self.zero_edges):
            zero_adj.setdefault(i_node, []).append((edge_idx, j_node))
            zero_adj.setdefault(j_node, []).append((edge_idx, i_node))

        visited_nodes = set()
        edge_used = np.zeros(len(self.zero_edges), dtype=bool)
        comp_nodes = []
        comp_edge_indices = []

        for start in zero_adj:
            if start in visited_nodes:
                continue
            q = deque([start])
            visited_nodes.add(start)
            nodes = []
            edges_idx = []
            while q:
                u = q.popleft()
                nodes.append(u)
                for edge_idx, v in zero_adj[u]:
                    if not edge_used[edge_idx]:
                        edge_used[edge_idx] = True
                        edges_idx.append(edge_idx)
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        q.append(v)
            if len(nodes) > 1:
                comp_nodes.append(nodes)
                comp_edge_indices.append(edges_idx)

        self.comp_nodes = comp_nodes
        self.comp_tree_edges = []
        for nodes, edge_indices in zip(comp_nodes, comp_edge_indices):
            if len(edge_indices) == len(nodes) - 1:
                self.comp_tree_edges.append(list(edge_indices))
                continue
            local_idx = {node: i for i, node in enumerate(nodes)}
            local_edges = []
            orig_indices = []
            for edge_idx in edge_indices:
                _, _, i_node, j_node = self.zero_edges[edge_idx]
                local_edges.append((local_idx[i_node], local_idx[j_node]))
                orig_indices.append(edge_idx)
            tree_local_idx = find_spanning_tree_edges(local_edges, len(nodes))
            self.comp_tree_edges.append([orig_indices[i] for i in tree_local_idx])

        self.N_phi = sum(len(nodes) for nodes in self.comp_nodes)
        phi_node = []
        self.ref_phi_idx = []
        for nodes in self.comp_nodes:
            # 每个零阻抗连通分量固定一个 phi 参考，其余 phi 差值代表支路电流。
            self.ref_phi_idx.append(len(phi_node))
            phi_node.extend(nodes)

        node_to_phi = np.full(self.N, -1, dtype=np.int32)
        if phi_node:
            node_to_phi[np.asarray(phi_node, dtype=np.int32)] = np.arange(len(phi_node), dtype=np.int32)

        self.zero_branch_info = []
        for tp, dev_idx, i_node, j_node in self.zero_edges:
            phi_a = int(node_to_phi[i_node])
            phi_b = int(node_to_phi[j_node])
            if phi_a < 0 or phi_b < 0:
                raise RuntimeError("节点不在 phi 变量中")
            self.zero_branch_info.append((tp, dev_idx, i_node, j_node, phi_a, phi_b))

        if self.zero_branch_info:
            self.zero_type = np.asarray([item[0] for item in self.zero_branch_info], dtype=object)
            self.zero_dev_idx = np.asarray([item[1] for item in self.zero_branch_info], dtype=np.int32)
            self.zero_i = np.asarray([item[2] for item in self.zero_branch_info], dtype=np.int32)
            self.zero_j = np.asarray([item[3] for item in self.zero_branch_info], dtype=np.int32)
            self.zero_phi_a = np.asarray([item[4] for item in self.zero_branch_info], dtype=np.int32)
            self.zero_phi_b = np.asarray([item[5] for item in self.zero_branch_info], dtype=np.int32)
        else:
            self.zero_type = np.array([], dtype=object)
            self.zero_dev_idx = self.zero_i = self.zero_j = np.array([], dtype=np.int32)
            self.zero_phi_a = self.zero_phi_b = np.array([], dtype=np.int32)

        zero_constraint_edges = [edge_idx for edges in self.comp_tree_edges for edge_idx in edges]
        if zero_constraint_edges:
            self.zero_con_i = np.asarray([self.zero_edges[idx][2] for idx in zero_constraint_edges], dtype=np.int32)
            self.zero_con_j = np.asarray([self.zero_edges[idx][3] for idx in zero_constraint_edges], dtype=np.int32)
        else:
            self.zero_con_i = self.zero_con_j = np.array([], dtype=np.int32)
        self.ref_phi_idx = np.asarray(self.ref_phi_idx, dtype=np.int32)

        # ---------- 3. DC-DC变流器 ----------
        dcdc = self.ppc["dcdc"]
        if dcdc.size:
            dcdc_positions = self._topology_terminal_solver_positions("dcdc", dcdc.shape[0])
            if dcdc_positions is not None:
                self.dcdc_idx, self.dcdc_i, self.dcdc_j = dcdc_positions
                dcdc_mask = np.zeros(dcdc.shape[0], dtype=bool)
                dcdc_mask[self.dcdc_idx] = True
            else:
                dcdc_i_all = self._map_nodes_with_lookup(dcdc[:, DC_DCDC_COLS["i_node"]], node_lookup)
                dcdc_j_all = self._map_nodes_with_lookup(dcdc[:, DC_DCDC_COLS["j_node"]], node_lookup)
                dcdc_mask = (
                    (dcdc[:, DC_DCDC_COLS["run_stat"]] == 1)
                    & (dcdc_i_all >= 0)
                    & (dcdc_j_all >= 0)
                )
                self.dcdc_idx = np.nonzero(dcdc_mask)[0].astype(np.int32)
                self.dcdc_i = dcdc_i_all[dcdc_mask].astype(np.int32, copy=False)
                self.dcdc_j = dcdc_j_all[dcdc_mask].astype(np.int32, copy=False)
            dcdc_active = dcdc[dcdc_mask]
            self.dcdc_i_ctrl_code = dcdc_active[:, DC_DCDC_COLS["i_control_type"]].astype(np.int8, copy=False)
            self.dcdc_j_ctrl_code = dcdc_active[:, DC_DCDC_COLS["j_control_type"]].astype(np.int8, copy=False)
            self.dcdc_p_set = dcdc_active[:, DC_DCDC_COLS["p_set"]].astype(np.float64, copy=False)
            self.dcdc_i_set = dcdc_active[:, DC_DCDC_COLS["i_set"]].astype(np.float64, copy=False)
            self.dcdc_v_set = dcdc_active[:, DC_DCDC_COLS["v_set"]].astype(np.float64, copy=False)
            self.dcdc_r1 = dcdc_active[:, DC_DCDC_COLS["r1"]].astype(np.float64, copy=False)
            self.dcdc_r2 = dcdc_active[:, DC_DCDC_COLS["r2"]].astype(np.float64, copy=False)
        else:
            self.dcdc_idx = self.dcdc_i = self.dcdc_j = np.array([], dtype=np.int32)
            self.dcdc_i_ctrl_code = np.array([], dtype=np.int8)
            self.dcdc_j_ctrl_code = np.array([], dtype=np.int8)
            self.dcdc_p_set = self.dcdc_i_set = self.dcdc_v_set = np.array([], dtype=np.float64)
            self.dcdc_r1 = self.dcdc_r2 = np.array([], dtype=np.float64)
        self.N_dcdc = self.dcdc_idx.size
        self.dcdc_ctrl = self.dcdc_i_ctrl_code
        valid_dcdc_ctrl = np.asarray([DC_CTRL_P, DC_CTRL_V, DC_CTRL_I, DC_CTRL_SLACK], dtype=np.int8)
        i_active_ctrl = self.dcdc_i_ctrl_code != DC_CTRL_SLACK
        j_active_ctrl = self.dcdc_j_ctrl_code != DC_CTRL_SLACK
        bad_ctrl = (
            ~np.isin(self.dcdc_i_ctrl_code, valid_dcdc_ctrl)
            | ~np.isin(self.dcdc_j_ctrl_code, valid_dcdc_ctrl)
            | (i_active_ctrl == j_active_ctrl)
        )
        if np.any(bad_ctrl):
            bad_pos = int(np.flatnonzero(bad_ctrl)[0])
            raise ValueError(
                "DCDCConverter 控制类型必须且只能一端为 CTRL_P/CTRL_V/CTRL_I，另一端为 SLACK；"
                f"第 {bad_pos + 1} 个活动 DCDC 为 "
                f"i_control_type={int(self.dcdc_i_ctrl_code[bad_pos])}, "
                f"j_control_type={int(self.dcdc_j_ctrl_code[bad_pos])}"
            )

        # ---------- 4. 确定松弛节点 ----------
        gen = self.ppc["gen"]
        self.slack_nodes = {
            node: float(gen[int(gens[0]), DC_GEN_COLS["v_set"]])
            for node, gens in self.slack_gen_info.items()
        }
        self.slack_node_arr = np.fromiter(self.slack_nodes.keys(), dtype=np.int32, count=len(self.slack_nodes))
        self.slack_value_arr = np.fromiter(self.slack_nodes.values(), dtype=np.float64, count=len(self.slack_nodes))

        # ---------- 5. 变量定义 ----------
        # 前 N 个变量是节点电压；随后是零阻抗连通分量的 phi。
        # array/summary/none 模式消去 DCDC j 端功率，j 端由损耗方程闭式回算。
        self.dcdc_eliminate_pj = bool(self.N_dcdc and self._should_eliminate_dcdc_j_power())
        dcdc_var_count = self.N_dcdc if self.dcdc_eliminate_pj else self.N_dcdc * 2
        self.total_vars = self.N + self.N_phi + dcdc_var_count
        x = np.zeros(self.total_vars, dtype=np.float64)
        x[:self.N] = 1.0
        if self.slack_node_arr.size:
            x[self.slack_node_arr] = self.slack_value_arr

        # ---------- 6. 节点分类 ----------
        known_mask = np.zeros(self.N, dtype=bool)
        if self.slack_node_arr.size:
            known_mask[self.slack_node_arr] = True
        self.unknown_nodes = np.where(~known_mask)[0].astype(np.int32)
        self.n_unknown = self.unknown_nodes.size
        self.n_known = self.slack_node_arr.size
        self.node_eq = np.full(self.N, -1, dtype=np.int32)
        self.node_eq[self.unknown_nodes] = np.arange(self.n_unknown, dtype=np.int32)

        self.n_zero_constraint = self.zero_con_i.size
        self.n_phi_fix = self.ref_phi_idx.size
        self.n_dcdc = self.N_dcdc

        dcdc_eq_count = self.n_dcdc if self.dcdc_eliminate_pj else self.n_dcdc * 2
        self.total_eq = self.n_unknown + self.n_known + self.n_zero_constraint + self.n_phi_fix + dcdc_eq_count
        if self.total_vars != self.total_eq:
            if self.verbose:
                print(f"警告：变量数({self.total_vars})与方程数({self.total_eq})不匹配，请检查零阻抗支路设置。")

        if self.verbose:
            print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")

        self.eq_unknown_start = 0
        self.eq_known_start = self.eq_unknown_start + self.n_unknown
        self.eq_zero_start = self.eq_known_start + self.n_known
        self.eq_phi_start = self.eq_zero_start + self.n_zero_constraint
        self.eq_dcdc_start = self.eq_phi_start + self.n_phi_fix

        if not self.keep_node_objects:
            self.unknown_map = {}
        else:
            self.unknown_map = {int(node): int(i) for i, node in enumerate(self.unknown_nodes)}
        self.zero_con_rows = self.eq_zero_start + np.arange(self.n_zero_constraint, dtype=np.int32)
        self.phi_fix_rows = self.eq_phi_start + np.arange(self.n_phi_fix, dtype=np.int32)
        self.dcdc_seq = np.arange(self.N_dcdc, dtype=np.int32)
        if self.dcdc_eliminate_pj:
            self.dcdc_p_col = self.N + self.N_phi + self.dcdc_seq
            self.dcdc_q_col = np.array([], dtype=np.int32)
            self.dcdc_eq_ctrl = self.eq_dcdc_start + self.dcdc_seq
            self.dcdc_eq_loss = np.array([], dtype=np.int32)
        else:
            self.dcdc_p_col = self.N + self.N_phi + 2 * self.dcdc_seq
            self.dcdc_q_col = self.dcdc_p_col + 1
            self.dcdc_eq_ctrl = self.eq_dcdc_start + 2 * self.dcdc_seq
            self.dcdc_eq_loss = self.dcdc_eq_ctrl + 1
        self.dcdc_i_ctrl_p_mask = self.dcdc_i_ctrl_code == DC_CTRL_P
        self.dcdc_i_ctrl_v_mask = self.dcdc_i_ctrl_code == DC_CTRL_V
        self.dcdc_i_ctrl_i_mask = self.dcdc_i_ctrl_code == DC_CTRL_I
        self.dcdc_j_ctrl_p_mask = self.dcdc_j_ctrl_code == DC_CTRL_P
        self.dcdc_j_ctrl_v_mask = self.dcdc_j_ctrl_code == DC_CTRL_V
        self.dcdc_j_ctrl_i_mask = self.dcdc_j_ctrl_code == DC_CTRL_I
        if self.N_dcdc:
            if np.any(self.dcdc_i_ctrl_v_mask):
                x[self.dcdc_i[self.dcdc_i_ctrl_v_mask]] = self.dcdc_v_set[self.dcdc_i_ctrl_v_mask]
            if np.any(self.dcdc_j_ctrl_v_mask):
                x[self.dcdc_j[self.dcdc_j_ctrl_v_mask]] = self.dcdc_v_set[self.dcdc_j_ctrl_v_mask]
            pi_seed = self.P_const[self.dcdc_i].astype(np.float64, copy=True)
            if np.any(self.dcdc_i_ctrl_p_mask):
                pi_seed[self.dcdc_i_ctrl_p_mask] = self.dcdc_p_set[self.dcdc_i_ctrl_p_mask]
            if np.any(self.dcdc_i_ctrl_i_mask):
                pi_seed[self.dcdc_i_ctrl_i_mask] = (
                    self.dcdc_i_set[self.dcdc_i_ctrl_i_mask] * x[self.dcdc_i[self.dcdc_i_ctrl_i_mask]]
                )
            x[self.dcdc_p_col] = pi_seed
            if not self.dcdc_eliminate_pj:
                q_seed = -pi_seed
                if np.any(self.dcdc_j_ctrl_p_mask):
                    q_seed[self.dcdc_j_ctrl_p_mask] = self.dcdc_p_set[self.dcdc_j_ctrl_p_mask]
                if np.any(self.dcdc_j_ctrl_i_mask):
                    q_seed[self.dcdc_j_ctrl_i_mask] = (
                        self.dcdc_i_set[self.dcdc_j_ctrl_i_mask] * x[self.dcdc_j[self.dcdc_j_ctrl_i_mask]]
                    )
                x[self.dcdc_q_col] = q_seed
        self.dcdc_ones = np.ones(self.N_dcdc, dtype=np.float64)
        self._prepare_static_jacobian_indices()
        self.G = G
        self.x = x

        return

    def _prepare_static_jacobian_indices(self):
        """Precompute Jacobian row/column indices that are constant across Newton steps."""
        self.zero_i_eq = self.node_eq[self.zero_i] if self.zero_i.size else np.array([], dtype=np.int32)
        self.zero_j_eq = self.node_eq[self.zero_j] if self.zero_j.size else np.array([], dtype=np.int32)
        self.zero_i_unknown_mask = self.zero_i_eq >= 0
        self.zero_j_unknown_mask = self.zero_j_eq >= 0
        self.zero_i_unknown_count = int(np.count_nonzero(self.zero_i_unknown_mask))
        self.zero_j_unknown_count = int(np.count_nonzero(self.zero_j_unknown_mask))
        if self.zero_i_unknown_count:
            self.zero_i_rows_jac = np.repeat(self.zero_i_eq[self.zero_i_unknown_mask], 3)
            self.zero_i_cols_jac = np.empty(3 * self.zero_i_unknown_count, dtype=np.int32)
            self.zero_i_cols_jac[0::3] = self.N + self.zero_phi_a[self.zero_i_unknown_mask]
            self.zero_i_cols_jac[1::3] = self.N + self.zero_phi_b[self.zero_i_unknown_mask]
            self.zero_i_cols_jac[2::3] = self.zero_i[self.zero_i_unknown_mask]
        else:
            self.zero_i_rows_jac = self.zero_i_cols_jac = np.array([], dtype=np.int32)
        if self.zero_j_unknown_count:
            self.zero_j_rows_jac = np.repeat(self.zero_j_eq[self.zero_j_unknown_mask], 3)
            self.zero_j_cols_jac = np.empty(3 * self.zero_j_unknown_count, dtype=np.int32)
            self.zero_j_cols_jac[0::3] = self.N + self.zero_phi_a[self.zero_j_unknown_mask]
            self.zero_j_cols_jac[1::3] = self.N + self.zero_phi_b[self.zero_j_unknown_mask]
            self.zero_j_cols_jac[2::3] = self.zero_j[self.zero_j_unknown_mask]
        else:
            self.zero_j_rows_jac = self.zero_j_cols_jac = np.array([], dtype=np.int32)

        if self.n_known:
            self.known_rows_jac = self.eq_known_start + np.arange(self.n_known, dtype=np.int32)
            self.known_cols_jac = self.slack_node_arr
            self.known_data_jac = np.ones(self.n_known, dtype=np.float64)
        else:
            self.known_rows_jac = self.known_cols_jac = np.array([], dtype=np.int32)
            self.known_data_jac = np.array([], dtype=np.float64)

        if self.n_zero_constraint:
            self.zero_con_rows_jac = np.repeat(self.zero_con_rows, 2)
            self.zero_con_cols_jac = np.empty(2 * self.n_zero_constraint, dtype=np.int32)
            self.zero_con_data_jac = np.empty(2 * self.n_zero_constraint, dtype=np.float64)
            self.zero_con_cols_jac[0::2] = self.zero_con_i
            self.zero_con_cols_jac[1::2] = self.zero_con_j
            self.zero_con_data_jac[0::2] = 1.0
            self.zero_con_data_jac[1::2] = -1.0
        else:
            self.zero_con_rows_jac = self.zero_con_cols_jac = np.array([], dtype=np.int32)
            self.zero_con_data_jac = np.array([], dtype=np.float64)

        if self.n_phi_fix:
            self.phi_fix_cols_jac = self.N + self.ref_phi_idx
            self.phi_fix_data_jac = np.ones(self.n_phi_fix, dtype=np.float64)
        else:
            self.phi_fix_cols_jac = np.array([], dtype=np.int32)
            self.phi_fix_data_jac = np.array([], dtype=np.float64)

        if self.N_dcdc:
            self.dcdc_i_eq = self.node_eq[self.dcdc_i]
            self.dcdc_j_eq = self.node_eq[self.dcdc_j]
            self.dcdc_i_unknown_mask = self.dcdc_i_eq >= 0
            self.dcdc_j_unknown_mask = self.dcdc_j_eq >= 0
            self.dcdc_i_unknown_idx = np.where(self.dcdc_i_unknown_mask)[0].astype(np.int32)
            self.dcdc_j_unknown_idx = np.where(self.dcdc_j_unknown_mask)[0].astype(np.int32)
            self.dcdc_i_eq_rows_jac = self.dcdc_i_eq[self.dcdc_i_unknown_mask]
            self.dcdc_i_eq_cols_jac = self.dcdc_p_col[self.dcdc_i_unknown_idx]
            self.dcdc_i_eq_data_jac = np.ones(self.dcdc_i_unknown_idx.size, dtype=np.float64)
            if self.dcdc_eliminate_pj:
                self.dcdc_j_eq_rows_jac = np.repeat(self.dcdc_j_eq[self.dcdc_j_unknown_mask], 3)
                self.dcdc_j_eq_cols_jac = np.empty(3 * self.dcdc_j_unknown_idx.size, dtype=np.int32)
                self.dcdc_j_eq_cols_jac[0::3] = self.dcdc_p_col[self.dcdc_j_unknown_idx]
                self.dcdc_j_eq_cols_jac[1::3] = self.dcdc_i[self.dcdc_j_unknown_idx]
                self.dcdc_j_eq_cols_jac[2::3] = self.dcdc_j[self.dcdc_j_unknown_idx]
                self.dcdc_j_eq_data_jac = np.empty(3 * self.dcdc_j_unknown_idx.size, dtype=np.float64)
            else:
                self.dcdc_j_eq_rows_jac = self.dcdc_j_eq[self.dcdc_j_unknown_mask]
                self.dcdc_j_eq_cols_jac = self.dcdc_q_col[self.dcdc_j_unknown_idx]
                self.dcdc_j_eq_data_jac = np.ones(self.dcdc_j_unknown_idx.size, dtype=np.float64)

            ctrl_static_rows = []
            ctrl_static_cols = []
            ctrl_static_data = []

            i_p_idx = np.flatnonzero(self.dcdc_i_ctrl_p_mask).astype(np.int32, copy=False)
            if i_p_idx.size:
                ctrl_static_rows.append(self.dcdc_eq_ctrl[i_p_idx])
                ctrl_static_cols.append(self.dcdc_p_col[i_p_idx])
                ctrl_static_data.append(np.ones(i_p_idx.size, dtype=np.float64))
            i_v_idx = np.flatnonzero(self.dcdc_i_ctrl_v_mask).astype(np.int32, copy=False)
            if i_v_idx.size:
                ctrl_static_rows.append(self.dcdc_eq_ctrl[i_v_idx])
                ctrl_static_cols.append(self.dcdc_i[i_v_idx])
                ctrl_static_data.append(np.ones(i_v_idx.size, dtype=np.float64))
            i_i_idx = np.flatnonzero(self.dcdc_i_ctrl_i_mask).astype(np.int32, copy=False)
            if i_i_idx.size:
                ctrl_static_rows.append(np.repeat(self.dcdc_eq_ctrl[i_i_idx], 2))
                cols = np.empty(2 * i_i_idx.size, dtype=np.int32)
                cols[0::2] = self.dcdc_p_col[i_i_idx]
                cols[1::2] = self.dcdc_i[i_i_idx]
                data = np.empty(2 * i_i_idx.size, dtype=np.float64)
                data[0::2] = 1.0
                data[1::2] = -self.dcdc_i_set[i_i_idx]
                ctrl_static_cols.append(cols)
                ctrl_static_data.append(data)
            j_v_idx = np.flatnonzero(self.dcdc_j_ctrl_v_mask).astype(np.int32, copy=False)
            if j_v_idx.size:
                ctrl_static_rows.append(self.dcdc_eq_ctrl[j_v_idx])
                ctrl_static_cols.append(self.dcdc_j[j_v_idx])
                ctrl_static_data.append(np.ones(j_v_idx.size, dtype=np.float64))
            if not self.dcdc_eliminate_pj:
                j_p_idx = np.flatnonzero(self.dcdc_j_ctrl_p_mask).astype(np.int32, copy=False)
                if j_p_idx.size:
                    ctrl_static_rows.append(self.dcdc_eq_ctrl[j_p_idx])
                    ctrl_static_cols.append(self.dcdc_q_col[j_p_idx])
                    ctrl_static_data.append(np.ones(j_p_idx.size, dtype=np.float64))
                j_i_idx = np.flatnonzero(self.dcdc_j_ctrl_i_mask).astype(np.int32, copy=False)
                if j_i_idx.size:
                    ctrl_static_rows.append(np.repeat(self.dcdc_eq_ctrl[j_i_idx], 2))
                    cols = np.empty(2 * j_i_idx.size, dtype=np.int32)
                    cols[0::2] = self.dcdc_q_col[j_i_idx]
                    cols[1::2] = self.dcdc_j[j_i_idx]
                    data = np.empty(2 * j_i_idx.size, dtype=np.float64)
                    data[0::2] = 1.0
                    data[1::2] = -self.dcdc_i_set[j_i_idx]
                    ctrl_static_cols.append(cols)
                    ctrl_static_data.append(data)
            if ctrl_static_rows:
                self.dcdc_ctrl_static_rows_jac = np.concatenate(ctrl_static_rows).astype(np.int32, copy=False)
                self.dcdc_ctrl_static_cols_jac = np.concatenate(ctrl_static_cols).astype(np.int32, copy=False)
                self.dcdc_ctrl_static_data_jac = np.concatenate(ctrl_static_data).astype(np.float64, copy=False)
            else:
                self.dcdc_ctrl_static_rows_jac = self.dcdc_ctrl_static_cols_jac = np.array([], dtype=np.int32)
                self.dcdc_ctrl_static_data_jac = np.array([], dtype=np.float64)
            if self.dcdc_eliminate_pj:
                self.dcdc_ctrl_j_dynamic_idx = np.flatnonzero(
                    self.dcdc_j_ctrl_p_mask | self.dcdc_j_ctrl_i_mask
                ).astype(np.int32, copy=False)
                if self.dcdc_ctrl_j_dynamic_idx.size:
                    idx = self.dcdc_ctrl_j_dynamic_idx
                    self.dcdc_ctrl_j_dynamic_i_mask = self.dcdc_j_ctrl_i_mask[idx]
                    self.dcdc_ctrl_j_dynamic_rows_jac = np.repeat(self.dcdc_eq_ctrl[idx], 3)
                    self.dcdc_ctrl_j_dynamic_cols_jac = np.empty(3 * idx.size, dtype=np.int32)
                    self.dcdc_ctrl_j_dynamic_cols_jac[0::3] = self.dcdc_p_col[idx]
                    self.dcdc_ctrl_j_dynamic_cols_jac[1::3] = self.dcdc_i[idx]
                    self.dcdc_ctrl_j_dynamic_cols_jac[2::3] = self.dcdc_j[idx]
                    self.dcdc_ctrl_j_dynamic_data_jac = np.empty(3 * idx.size, dtype=np.float64)
                else:
                    self.dcdc_ctrl_j_dynamic_i_mask = np.array([], dtype=bool)
                    self.dcdc_ctrl_j_dynamic_rows_jac = self.dcdc_ctrl_j_dynamic_cols_jac = np.array([], dtype=np.int32)
                    self.dcdc_ctrl_j_dynamic_data_jac = np.array([], dtype=np.float64)
            else:
                self.dcdc_ctrl_j_dynamic_idx = np.array([], dtype=np.int32)
                self.dcdc_ctrl_j_dynamic_i_mask = np.array([], dtype=bool)
                self.dcdc_ctrl_j_dynamic_rows_jac = self.dcdc_ctrl_j_dynamic_cols_jac = np.array([], dtype=np.int32)
                self.dcdc_ctrl_j_dynamic_data_jac = np.array([], dtype=np.float64)
            if self.dcdc_eliminate_pj:
                self.dcdc_loss_rows_jac = np.array([], dtype=np.int32)
                self.dcdc_loss_cols_jac = np.array([], dtype=np.int32)
            else:
                self.dcdc_loss_rows_jac = np.repeat(self.dcdc_eq_loss, 4)
                self.dcdc_loss_cols_jac = np.empty(self.N_dcdc * 4, dtype=np.int32)
                self.dcdc_loss_cols_jac[0::4] = self.dcdc_p_col
                self.dcdc_loss_cols_jac[1::4] = self.dcdc_q_col
                self.dcdc_loss_cols_jac[2::4] = self.dcdc_i
                self.dcdc_loss_cols_jac[3::4] = self.dcdc_j
        else:
            self.dcdc_i_unknown_idx = self.dcdc_j_unknown_idx = np.array([], dtype=np.int32)
            self.dcdc_ctrl_static_rows_jac = self.dcdc_ctrl_static_cols_jac = np.array([], dtype=np.int32)
            self.dcdc_ctrl_static_data_jac = np.array([], dtype=np.float64)
            self.dcdc_ctrl_j_dynamic_idx = np.array([], dtype=np.int32)
            self.dcdc_ctrl_j_dynamic_i_mask = np.array([], dtype=bool)
            self.dcdc_ctrl_j_dynamic_rows_jac = self.dcdc_ctrl_j_dynamic_cols_jac = np.array([], dtype=np.int32)
            self.dcdc_ctrl_j_dynamic_data_jac = np.array([], dtype=np.float64)
        self._prepare_jacobian_csr_pattern()

    def _prepare_jacobian_csr_pattern(self):
        """Precompute the DC Jacobian CSR pattern and raw-entry accumulation map."""
        rows_parts = []
        cols_parts = []
        raw_count = 0

        def add_part(name, rows, cols):
            nonlocal raw_count
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            if rows.size != cols.size:
                raise ValueError(f"Jacobian pattern part {name!r} has mismatched row/column lengths")
            part_slice = slice(raw_count, raw_count + rows.size)
            setattr(self, f"_dc_jac_{name}_slice", part_slice)
            raw_count += rows.size
            if rows.size:
                rows_parts.append(rows)
                cols_parts.append(cols)

        unknown_rows = []
        unknown_cols = []
        unknown_row_nodes = []
        unknown_col_nodes = []
        unknown_g_data = []
        if self.n_unknown:
            G_csr = self.G.tocsr()
            counts = np.diff(G_csr.indptr).astype(np.int32, copy=False)
            if G_csr.indices.size:
                all_row_nodes = np.repeat(np.arange(self.N, dtype=np.int32), counts)
                all_eq_rows = self.node_eq[all_row_nodes]
                keep = all_eq_rows >= 0
                if np.any(keep):
                    unknown_rows = all_eq_rows[keep].astype(np.int32, copy=True)
                    unknown_cols = G_csr.indices[keep].astype(np.int32, copy=True)
                    unknown_row_nodes = all_row_nodes[keep].astype(np.int32, copy=True)
                    unknown_col_nodes = unknown_cols.copy()
                    unknown_g_data = G_csr.data[keep].astype(np.float64, copy=True)
                else:
                    unknown_rows = np.array([], dtype=np.int32)
                    unknown_cols = np.array([], dtype=np.int32)
                    unknown_row_nodes = np.array([], dtype=np.int32)
                    unknown_col_nodes = np.array([], dtype=np.int32)
                    unknown_g_data = np.array([], dtype=np.float64)
            else:
                unknown_rows = np.array([], dtype=np.int32)
                unknown_cols = np.array([], dtype=np.int32)
                unknown_row_nodes = np.array([], dtype=np.int32)
                unknown_col_nodes = np.array([], dtype=np.int32)
                unknown_g_data = np.array([], dtype=np.float64)

            diag_seen = np.zeros(self.n_unknown, dtype=bool)
            if np.asarray(unknown_rows).size:
                diag_mask = np.asarray(unknown_col_nodes) == np.asarray(unknown_row_nodes)
                if np.any(diag_mask):
                    diag_seen[np.asarray(unknown_rows, dtype=np.int32)[diag_mask]] = True
            missing_diag = np.flatnonzero(~diag_seen).astype(np.int32, copy=False)
            if missing_diag.size:
                missing_nodes = self.unknown_nodes[missing_diag].astype(np.int32, copy=False)
                unknown_rows = np.concatenate((np.asarray(unknown_rows, dtype=np.int32), missing_diag))
                unknown_cols = np.concatenate((np.asarray(unknown_cols, dtype=np.int32), missing_nodes))
                unknown_row_nodes = np.concatenate((np.asarray(unknown_row_nodes, dtype=np.int32), missing_nodes))
                unknown_col_nodes = np.concatenate((np.asarray(unknown_col_nodes, dtype=np.int32), missing_nodes))
                unknown_g_data = np.concatenate((
                    np.asarray(unknown_g_data, dtype=np.float64),
                    np.zeros(missing_diag.size, dtype=np.float64),
                ))
        self._dc_jac_unknown_row_nodes = np.asarray(unknown_row_nodes, dtype=np.int32)
        self._dc_jac_unknown_col_nodes = np.asarray(unknown_col_nodes, dtype=np.int32)
        self._dc_jac_unknown_g_data = np.asarray(unknown_g_data, dtype=np.float64)
        self._dc_jac_unknown_diag_mask = self._dc_jac_unknown_row_nodes == self._dc_jac_unknown_col_nodes
        add_part("unknown", unknown_rows, unknown_cols)

        add_part("zero_i", self.zero_i_rows_jac, self.zero_i_cols_jac)
        add_part("zero_j", self.zero_j_rows_jac, self.zero_j_cols_jac)
        add_part("dcdc_i", self.dcdc_i_eq_rows_jac if self.N_dcdc else [], self.dcdc_i_eq_cols_jac if self.N_dcdc else [])
        add_part("dcdc_j", self.dcdc_j_eq_rows_jac if self.N_dcdc else [], self.dcdc_j_eq_cols_jac if self.N_dcdc else [])
        add_part("known", self.known_rows_jac, self.known_cols_jac)
        add_part("zero_con", self.zero_con_rows_jac, self.zero_con_cols_jac)
        add_part("phi_fix", self.phi_fix_rows, self.phi_fix_cols_jac)
        add_part(
            "dcdc_ctrl_static",
            self.dcdc_ctrl_static_rows_jac if self.N_dcdc else [],
            self.dcdc_ctrl_static_cols_jac if self.N_dcdc else [],
        )
        add_part(
            "dcdc_ctrl_j_dynamic",
            self.dcdc_ctrl_j_dynamic_rows_jac if self.N_dcdc else [],
            self.dcdc_ctrl_j_dynamic_cols_jac if self.N_dcdc else [],
        )
        add_part("dcdc_loss", self.dcdc_loss_rows_jac if self.N_dcdc else [], self.dcdc_loss_cols_jac if self.N_dcdc else [])

        self._dc_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            self._dc_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
            self._dc_jac_csr_indices = np.array([], dtype=np.int32)
            self._dc_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
            self._dc_jac_csr_data = np.array([], dtype=np.float64)
            self._dc_jac_csr_sum_plan = build_raw_sum_plan(self._dc_jac_raw_to_csr_pos, 0)
            self._dc_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
            self._dc_jac_csc_indices = np.array([], dtype=np.int32)
            self._dc_jac_csc_indptr = np.zeros(self.total_vars + 1, dtype=np.int32)
            self._dc_jac_csc_data = np.array([], dtype=np.float64)
            self._dc_jac_csc_sum_plan = build_raw_sum_plan(self._dc_jac_raw_to_csc_pos, 0)
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        if self._cache_csr_jacobian_pattern:
            (
                self._dc_jac_csr_indices,
                self._dc_jac_csr_indptr,
                self._dc_jac_raw_to_csr_pos,
            ) = build_compressed_pattern_from_raw_coords(raw_rows, raw_cols, self.total_eq)
        else:
            self._dc_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
            self._dc_jac_csr_indices = np.array([], dtype=np.int32)
            self._dc_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
            self._dc_jac_csr_sum_plan = build_raw_sum_plan(self._dc_jac_raw_to_csr_pos, 0)
        (
            self._dc_jac_csc_indices,
            self._dc_jac_csc_indptr,
            self._dc_jac_raw_to_csc_pos,
        ) = build_compressed_pattern_from_raw_coords(raw_cols, raw_rows, self.total_vars)
        self._dc_jac_csr_data = np.empty(self._dc_jac_csr_indices.size, dtype=np.float64)
        self._dc_jac_csc_data = np.empty(self._dc_jac_csc_indices.size, dtype=np.float64)
        if self._cache_csr_jacobian_pattern:
            self._dc_jac_csr_sum_plan = build_raw_sum_plan(self._dc_jac_raw_to_csr_pos, self._dc_jac_csr_data.size)
        self._dc_jac_csc_sum_plan = build_raw_sum_plan(self._dc_jac_raw_to_csc_pos, self._dc_jac_csc_data.size)

    @staticmethod
    def _slice_len(part_slice):
        return int(part_slice.stop - part_slice.start)

    def _fill_jacobian_raw_data(self, terms):
        """Refresh raw Jacobian entry data in the precomputed insertion order."""
        raw = self._dc_jac_raw_data
        V = terms["V"]
        GV = terms["GV"]

        part_slice = self._dc_jac_unknown_slice
        if self._slice_len(part_slice):
            data = raw[part_slice]
            row_nodes = self._dc_jac_unknown_row_nodes
            data[:] = V[row_nodes] * self._dc_jac_unknown_g_data
            diag_mask = self._dc_jac_unknown_diag_mask
            if np.any(diag_mask):
                diag_nodes = row_nodes[diag_mask]
                data[diag_mask] += GV[diag_nodes] + self.I_shunt[diag_nodes]

        if self.zero_i.size:
            current = terms["zero_current"]
            part_slice = self._dc_jac_zero_i_slice
            if self.zero_i_unknown_count:
                data = raw[part_slice]
                mask = self.zero_i_unknown_mask
                data[0::3] = V[self.zero_i[mask]]
                data[1::3] = -V[self.zero_i[mask]]
                data[2::3] = current[mask]
            part_slice = self._dc_jac_zero_j_slice
            if self.zero_j_unknown_count:
                data = raw[part_slice]
                mask = self.zero_j_unknown_mask
                data[0::3] = -V[self.zero_j[mask]]
                data[1::3] = V[self.zero_j[mask]]
                data[2::3] = -current[mask]

        part_slice = self._dc_jac_dcdc_i_slice
        if self._slice_len(part_slice):
            raw[part_slice] = self.dcdc_i_eq_data_jac
        part_slice = self._dc_jac_dcdc_j_slice
        if self._slice_len(part_slice):
            if getattr(self, "dcdc_eliminate_pj", False):
                idx = self.dcdc_j_unknown_idx
                data = raw[part_slice]
                data[0::3] = terms["dcdc_dpj_dpi"][idx]
                data[1::3] = terms["dcdc_dpj_dvi"][idx]
                data[2::3] = terms["dcdc_dpj_dvj"][idx]
            else:
                raw[part_slice] = self.dcdc_j_eq_data_jac
        part_slice = self._dc_jac_known_slice
        if self._slice_len(part_slice):
            raw[part_slice] = self.known_data_jac
        part_slice = self._dc_jac_zero_con_slice
        if self._slice_len(part_slice):
            raw[part_slice] = self.zero_con_data_jac
        part_slice = self._dc_jac_phi_fix_slice
        if self._slice_len(part_slice):
            raw[part_slice] = self.phi_fix_data_jac

        if self.N_dcdc:
            part_slice = self._dc_jac_dcdc_ctrl_static_slice
            if self._slice_len(part_slice):
                raw[part_slice] = self.dcdc_ctrl_static_data_jac
            part_slice = self._dc_jac_dcdc_ctrl_j_dynamic_slice
            if self._slice_len(part_slice):
                idx = self.dcdc_ctrl_j_dynamic_idx
                data = raw[part_slice]
                data[0::3] = terms["dcdc_dpj_dpi"][idx]
                data[1::3] = terms["dcdc_dpj_dvi"][idx]
                data[2::3] = terms["dcdc_dpj_dvj"][idx]
                if np.any(self.dcdc_ctrl_j_dynamic_i_mask):
                    third = data[2::3]
                    third[self.dcdc_ctrl_j_dynamic_i_mask] -= self.dcdc_i_set[idx[self.dcdc_ctrl_j_dynamic_i_mask]]

            if not getattr(self, "dcdc_eliminate_pj", False):
                vi = terms["dcdc_vi"]
                vj = terms["dcdc_vj"]
                pi = terms["dcdc_pi"]
                pj = terms["dcdc_pj"]
                vi2 = terms["dcdc_vi2"]
                vj2 = terms["dcdc_vj2"]
                pi2 = terms["dcdc_pi2"]
                pj2 = terms["dcdc_pj2"]
                data = raw[self._dc_jac_dcdc_loss_slice]
                data[0::4] = vi2 * vj2 - 2.0 * self.dcdc_r1 * pi * vj2
                data[1::4] = vi2 * vj2 - 2.0 * self.dcdc_r2 * pj * vi2
                data[2::4] = 2.0 * vi * vj2 * (pi + pj) - 2.0 * self.dcdc_r2 * pj2 * vi
                data[3::4] = 2.0 * vj * vi2 * (pi + pj) - 2.0 * self.dcdc_r1 * pi2 * vj

    def _get_jacobi_from_precomputed_pattern(self, terms, *, matrix_format="csr", build_matrix=True):
        if not hasattr(self, "_dc_jac_raw_data"):
            return None
        if self._dc_jac_raw_data.size == 0:
            if not build_matrix:
                return np.array([], dtype=np.float64)
            shape = (self.total_eq, self.total_vars)
            return csc_matrix(shape, dtype=np.float64) if matrix_format == "csc" else csr_matrix(shape, dtype=np.float64)
        self._fill_jacobian_raw_data(terms)

        if matrix_format == "csc":
            apply_raw_sum_plan(self._dc_jac_csc_data, self._dc_jac_raw_data, self._dc_jac_csc_sum_plan)
            if not build_matrix:
                return self._dc_jac_csc_data
            return csc_matrix(
                (self._dc_jac_csc_data, self._dc_jac_csc_indices, self._dc_jac_csc_indptr),
                shape=(self.total_eq, self.total_vars),
                copy=False,
            )
        if not self._dc_jac_raw_to_csr_pos.size:
            return None

        apply_raw_sum_plan(self._dc_jac_csr_data, self._dc_jac_raw_data, self._dc_jac_csr_sum_plan)
        if not build_matrix:
            return self._dc_jac_csr_data
        return csr_matrix(
            (self._dc_jac_csr_data, self._dc_jac_csr_indices, self._dc_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )

    def _eval_newton_terms(self, G, x):
        """Evaluate DC Newton quantities shared by residual and Jacobian."""
        V = x[:self.N]
        phi = x[self.N:self.N + self.N_phi]
        if self.N_dcdc > 0:
            dcdc_width = self.N_dcdc if getattr(self, "dcdc_eliminate_pj", False) else self.N_dcdc * 2
            Pdc = x[self.N + self.N_phi:self.N + self.N_phi + dcdc_width]
        else:
            Pdc = np.array([])
        GV = G.dot(V)
        terms = {
            "V": V,
            "phi": phi,
            "Pdc": Pdc,
            "GV": GV,
        }
        if self.zero_i.size:
            terms["zero_current"] = phi[self.zero_phi_a] - phi[self.zero_phi_b]
        else:
            terms["zero_current"] = np.array([], dtype=np.float64)
        if self.N_dcdc:
            vi = V[self.dcdc_i]
            vj = V[self.dcdc_j]
            if getattr(self, "dcdc_eliminate_pj", False):
                pi = Pdc
                pj, dpj_dpi, dpj_dvi, dpj_dvj = self._dcdc_j_power_from_loss(pi, vi, vj)
            else:
                pi = Pdc[0::2]
                pj = Pdc[1::2]
                dpj_dpi = dpj_dvi = dpj_dvj = np.array([], dtype=np.float64)
            terms.update(
                dcdc_vi=vi,
                dcdc_vj=vj,
                dcdc_pi=pi,
                dcdc_pj=pj,
                dcdc_dpj_dpi=dpj_dpi,
                dcdc_dpj_dvi=dpj_dvi,
                dcdc_dpj_dvj=dpj_dvj,
                dcdc_vi2=vi * vi,
                dcdc_vj2=vj * vj,
                dcdc_pi2=pi * pi,
                dcdc_pj2=pj * pj,
            )
        return terms

    def _get_jacobi_from_terms(self, G, x, terms, *, matrix_format="csr", build_matrix=True):
        """组装 DC Newton 方程的稀疏 Jacobian。"""
        precomputed = self._get_jacobi_from_precomputed_pattern(
            terms,
            matrix_format=matrix_format,
            build_matrix=build_matrix,
        )
        if precomputed is not None:
            return precomputed

        V = terms["V"]
        GV = terms["GV"]
        rows_parts = []
        cols_parts = []
        data_parts = []

        # 8.1 未知节点功率平衡方程行：F = V * (G @ V) + I_shunt * V - P_const
        if self.n_unknown:
            Jv = G.multiply(V[:, None]).tocsr()
            Jv.setdiag(Jv.diagonal() + GV + self.I_shunt)
            J_unknown = Jv[self.unknown_nodes, :].tocoo()
            rows_parts.append(J_unknown.row.astype(np.int32))
            cols_parts.append(J_unknown.col.astype(np.int32))
            data_parts.append(J_unknown.data.astype(np.float64))

        # 8.2 零阻抗支路功率注入对 V/phi 的偏导
        if self.zero_i.size:
            current = terms["zero_current"]

            if self.zero_i_unknown_count:
                data = np.empty(3 * self.zero_i_unknown_count, dtype=np.float64)
                data[0::3] = V[self.zero_i[self.zero_i_unknown_mask]]
                data[1::3] = -V[self.zero_i[self.zero_i_unknown_mask]]
                data[2::3] = current[self.zero_i_unknown_mask]
                rows_parts.append(self.zero_i_rows_jac)
                cols_parts.append(self.zero_i_cols_jac)
                data_parts.append(data)

            if self.zero_j_unknown_count:
                data = np.empty(3 * self.zero_j_unknown_count, dtype=np.float64)
                data[0::3] = -V[self.zero_j[self.zero_j_unknown_mask]]
                data[1::3] = V[self.zero_j[self.zero_j_unknown_mask]]
                data[2::3] = -current[self.zero_j_unknown_mask]
                rows_parts.append(self.zero_j_rows_jac)
                cols_parts.append(self.zero_j_cols_jac)
                data_parts.append(data)

        # 8.3 DC-DC功率变量对节点功率方程的偏导
        if self.N_dcdc:
            if self.dcdc_i_unknown_idx.size:
                rows_parts.append(self.dcdc_i_eq_rows_jac)
                cols_parts.append(self.dcdc_i_eq_cols_jac)
                data_parts.append(self.dcdc_i_eq_data_jac)
            if self.dcdc_j_unknown_idx.size:
                rows_parts.append(self.dcdc_j_eq_rows_jac)
                cols_parts.append(self.dcdc_j_eq_cols_jac)
                if getattr(self, "dcdc_eliminate_pj", False):
                    idx = self.dcdc_j_unknown_idx
                    data = np.empty(3 * idx.size, dtype=np.float64)
                    data[0::3] = terms["dcdc_dpj_dpi"][idx]
                    data[1::3] = terms["dcdc_dpj_dvi"][idx]
                    data[2::3] = terms["dcdc_dpj_dvj"][idx]
                    data_parts.append(data)
                else:
                    data_parts.append(self.dcdc_j_eq_data_jac)

        # 8.4 松弛节点电压方程行
        if self.n_known:
            rows_parts.append(self.known_rows_jac)
            cols_parts.append(self.known_cols_jac)
            data_parts.append(self.known_data_jac)

        # 8.5 零阻抗电压约束行
        if self.n_zero_constraint:
            rows_parts.append(self.zero_con_rows_jac)
            cols_parts.append(self.zero_con_cols_jac)
            data_parts.append(self.zero_con_data_jac)

        # 8.6 φ参考固定行
        if self.n_phi_fix:
            rows_parts.append(self.phi_fix_rows)
            cols_parts.append(self.phi_fix_cols_jac)
            data_parts.append(self.phi_fix_data_jac)

        # 8.7 DC-DC方程行
        if self.N_dcdc:
            if self.dcdc_ctrl_static_rows_jac.size:
                rows_parts.append(self.dcdc_ctrl_static_rows_jac)
                cols_parts.append(self.dcdc_ctrl_static_cols_jac)
                data_parts.append(self.dcdc_ctrl_static_data_jac)
            if self.dcdc_ctrl_j_dynamic_idx.size:
                idx = self.dcdc_ctrl_j_dynamic_idx
                data = np.empty(3 * idx.size, dtype=np.float64)
                data[0::3] = terms["dcdc_dpj_dpi"][idx]
                data[1::3] = terms["dcdc_dpj_dvi"][idx]
                data[2::3] = terms["dcdc_dpj_dvj"][idx]
                if np.any(self.dcdc_ctrl_j_dynamic_i_mask):
                    third = data[2::3]
                    third[self.dcdc_ctrl_j_dynamic_i_mask] -= self.dcdc_i_set[idx[self.dcdc_ctrl_j_dynamic_i_mask]]
                rows_parts.append(self.dcdc_ctrl_j_dynamic_rows_jac)
                cols_parts.append(self.dcdc_ctrl_j_dynamic_cols_jac)
                data_parts.append(data)

            if not getattr(self, "dcdc_eliminate_pj", False):
                vi = terms["dcdc_vi"]
                vj = terms["dcdc_vj"]
                pi = terms["dcdc_pi"]
                pj = terms["dcdc_pj"]
                vi2 = terms["dcdc_vi2"]
                vj2 = terms["dcdc_vj2"]
                pi2 = terms["dcdc_pi2"]
                pj2 = terms["dcdc_pj2"]
                loss_data = np.empty(self.N_dcdc * 4, dtype=np.float64)
                loss_data[0::4] = vi2 * vj2 - 2.0 * self.dcdc_r1 * pi * vj2
                loss_data[1::4] = vi2 * vj2 - 2.0 * self.dcdc_r2 * pj * vi2
                loss_data[2::4] = 2.0 * vi * vj2 * (pi + pj) - 2.0 * self.dcdc_r2 * pj2 * vi
                loss_data[3::4] = 2.0 * vj * vi2 * (pi + pj) - 2.0 * self.dcdc_r1 * pi2 * vj
                rows_parts.append(self.dcdc_loss_rows_jac)
                cols_parts.append(self.dcdc_loss_cols_jac)
                data_parts.append(loss_data)

        if rows_parts:
            J_rows = np.concatenate(rows_parts)
            J_cols = np.concatenate(cols_parts)
            J_data = np.concatenate(data_parts)
        else:
            J_rows = J_cols = np.array([], dtype=np.int32)
            J_data = np.array([], dtype=np.float64)

        jac = coo_matrix((J_data, (J_rows, J_cols)), shape=(self.total_eq, self.total_vars)).tocsr()
        return jac.tocsc() if matrix_format == "csc" else jac

    def get_jacobi(self, x: np.ndarray) -> csr_matrix:
        """Public Jacobian API for tests and external callers."""
        terms = self._eval_newton_terms(self.G, x)
        return self._get_jacobi_from_terms(self.G, x, terms)

    def _get_f_from_terms(self, x, terms):
        """计算 DC 残差：节点功率平衡、参考电压、零阻抗约束和 DCDC 约束。"""
        V = terms["V"]
        phi = terms["phi"]

        P_inj = V * terms["GV"] + self.I_shunt * V - self.P_const

        if self.zero_i.size:
            current = terms["zero_current"]
            P_inj += np.bincount(self.zero_i, weights=V[self.zero_i] * current, minlength=P_inj.size)
            P_inj += np.bincount(self.zero_j, weights=-V[self.zero_j] * current, minlength=P_inj.size)

        if self.N_dcdc:
            P_inj += np.bincount(self.dcdc_i, weights=terms["dcdc_pi"], minlength=P_inj.size)
            P_inj += np.bincount(self.dcdc_j, weights=terms["dcdc_pj"], minlength=P_inj.size)

        F = np.zeros(self.total_eq, dtype=np.float64)

        if self.n_unknown:
            F[self.eq_unknown_start:self.eq_known_start] = P_inj[self.unknown_nodes]

        if self.n_known:
            F[self.eq_known_start:self.eq_zero_start] = V[self.slack_node_arr] - self.slack_value_arr

        if self.n_zero_constraint:
            F[self.eq_zero_start:self.eq_phi_start] = V[self.zero_con_i] - V[self.zero_con_j]

        if self.n_phi_fix:
            F[self.eq_phi_start:self.eq_dcdc_start] = phi[self.ref_phi_idx]

        if self.N_dcdc:
            p_from = terms["dcdc_pi"]
            p_to = terms["dcdc_pj"]
            vi = terms["dcdc_vi"]
            vj = terms["dcdc_vj"]
            ctrl_values = np.empty(self.N_dcdc, dtype=np.float64)
            ctrl_values[self.dcdc_i_ctrl_p_mask] = p_from[self.dcdc_i_ctrl_p_mask] - self.dcdc_p_set[self.dcdc_i_ctrl_p_mask]
            ctrl_values[self.dcdc_i_ctrl_v_mask] = vi[self.dcdc_i_ctrl_v_mask] - self.dcdc_v_set[self.dcdc_i_ctrl_v_mask]
            ctrl_values[self.dcdc_i_ctrl_i_mask] = (
                p_from[self.dcdc_i_ctrl_i_mask]
                - self.dcdc_i_set[self.dcdc_i_ctrl_i_mask] * vi[self.dcdc_i_ctrl_i_mask]
            )
            ctrl_values[self.dcdc_j_ctrl_p_mask] = p_to[self.dcdc_j_ctrl_p_mask] - self.dcdc_p_set[self.dcdc_j_ctrl_p_mask]
            ctrl_values[self.dcdc_j_ctrl_v_mask] = vj[self.dcdc_j_ctrl_v_mask] - self.dcdc_v_set[self.dcdc_j_ctrl_v_mask]
            ctrl_values[self.dcdc_j_ctrl_i_mask] = (
                p_to[self.dcdc_j_ctrl_i_mask]
                - self.dcdc_i_set[self.dcdc_j_ctrl_i_mask] * vj[self.dcdc_j_ctrl_i_mask]
            )
            F[self.dcdc_eq_ctrl] = ctrl_values

            if not getattr(self, "dcdc_eliminate_pj", False):
                vi2 = terms["dcdc_vi2"]
                vj2 = terms["dcdc_vj2"]
                # 第二条方程保证两端端口功率与 r1/r2 损耗模型一致。
                F[self.dcdc_eq_loss] = (
                    vi2 * vj2 * (p_from + p_to)
                    - self.dcdc_r1 * p_from * p_from * vj2
                    - self.dcdc_r2 * p_to * p_to * vi2
                )

        return F

    def get_f(self, x: np.ndarray) -> np.ndarray:
        """Public residual API for tests and external callers."""
        terms = self._eval_newton_terms(self.G, x)
        return self._get_f_from_terms(x, terms)

    def _write_back_ppc(self):
        """Write array-mode DC LF results to self.result without object topology."""
        x = self.x
        ppc = self.ppc
        V_final = x[:self.N]
        phi_final = x[self.N:self.N + self.N_phi] if self.N_phi > 0 else np.array([])
        if self.N_dcdc > 0:
            dcdc_width = self.N_dcdc if getattr(self, "dcdc_eliminate_pj", False) else self.N_dcdc * 2
            Pdc_final = x[self.N + self.N_phi:self.N + self.N_phi + dcdc_width]
        else:
            Pdc_final = np.array([])
        node_lookup = self._alive_node_lookup_array()

        bus = ppc["bus"].copy()
        branch = ppc["branch"].copy()
        load = ppc["load"].copy()
        gen = ppc["gen"].copy()
        zero_branch = ppc["zero_branch"].copy()
        switch = ppc["switch"].copy()
        breaker = ppc.get("break", np.zeros((0, len(DC_BREAK_COLS)), dtype=np.float64)).copy()
        dcdc = ppc["dcdc"].copy()

        if bus.size:
            bus_pos = self._map_nodes_with_lookup(bus[:, DC_BUS_COLS["idx"]], node_lookup)
            bus[:, DC_BUS_COLS["voltage"]] = 0.0
            active_bus = bus_pos >= 0
            bus[active_bus, DC_BUS_COLS["voltage"]] = V_final[bus_pos[active_bus]]

        P_inj = np.zeros(self.N, dtype=np.float64)

        if branch.size:
            branch[:, [DC_BRANCH_COLS["i_p"], DC_BRANCH_COLS["j_p"], DC_BRANCH_COLS["current"]]] = 0.0
        if self.branch_idx.size:
            vi = V_final[self.branch_i]
            vj = V_final[self.branch_j]
            current = (vi - vj) / self.branch_r
            i_p = vi * current
            j_p = -vj * current
            branch[self.branch_idx, DC_BRANCH_COLS["current"]] = current
            branch[self.branch_idx, DC_BRANCH_COLS["i_p"]] = i_p
            branch[self.branch_idx, DC_BRANCH_COLS["j_p"]] = j_p
            P_inj += np.bincount(self.branch_i, weights=i_p, minlength=P_inj.size)
            P_inj += np.bincount(self.branch_j, weights=j_p, minlength=P_inj.size)

        if zero_branch.size:
            zero_branch[:, [DC_ZERO_BRANCH_COLS["p"], DC_ZERO_BRANCH_COLS["current"]]] = 0.0
        if switch.size:
            switch[:, [DC_SWITCH_COLS["p"], DC_SWITCH_COLS["current"]]] = 0.0
        if breaker.size:
            breaker[:, [DC_BREAK_COLS["p"], DC_BREAK_COLS["current"]]] = 0.0
        for tp, dev_idx, i_node, j_node, phi_a, phi_b in self.zero_branch_info:
            current = phi_final[phi_a] - phi_final[phi_b]
            p_from = V_final[i_node] * current
            if tp == 'Z':
                zero_branch[int(dev_idx), DC_ZERO_BRANCH_COLS["p"]] = p_from
                zero_branch[int(dev_idx), DC_ZERO_BRANCH_COLS["current"]] = current
            elif tp == 'S':
                switch[int(dev_idx), DC_SWITCH_COLS["p"]] = p_from
                switch[int(dev_idx), DC_SWITCH_COLS["current"]] = current
            elif tp == 'B':
                breaker[int(dev_idx), DC_BREAK_COLS["p"]] = p_from
                breaker[int(dev_idx), DC_BREAK_COLS["current"]] = current
            P_inj[i_node] += p_from
            P_inj[j_node] -= V_final[j_node] * current

        if load.size:
            load[:, [DC_LOAD_COLS["p"], DC_LOAD_COLS["current"]]] = 0.0
            load_pos_all = self._map_nodes_with_lookup(load[:, DC_LOAD_COLS["node"]], node_lookup)
            load_mask = (load[:, DC_LOAD_COLS["run_stat"]] == 1) & (load_pos_all >= 0)
            if np.any(load_mask):
                load_pos = load_pos_all[load_mask]
                v = V_final[load_pos]
                p = load[load_mask, DC_LOAD_COLS["pbase"]] * (
                    load[load_mask, DC_LOAD_COLS["pv0"]]
                    + load[load_mask, DC_LOAD_COLS["pv1"]] * v
                    + load[load_mask, DC_LOAD_COLS["pv2"]] * v * v
                )
                current = np.divide(
                    p,
                    v,
                    out=np.zeros_like(p),
                    where=np.abs(v) > self.min_voltage,
                )
                load[load_mask, DC_LOAD_COLS["p"]] = p
                load[load_mask, DC_LOAD_COLS["current"]] = current
                P_inj += np.bincount(load_pos, weights=p, minlength=P_inj.size)

        if gen.size:
            gen[:, [DC_GEN_COLS["p"], DC_GEN_COLS["current"]]] = 0.0
            gen_pos_all = self._map_nodes_with_lookup(gen[:, DC_GEN_COLS["node"]], node_lookup)
            gen_mask = (gen[:, DC_GEN_COLS["run_stat"]] == 1) & (gen_pos_all >= 0)
            if np.any(gen_mask):
                gen_pos = gen_pos_all[gen_mask]
                ctrl = gen[gen_mask, DC_GEN_COLS["control_type"]].astype(np.int8, copy=False)
                v = V_final[gen_pos]
                p_mask = ctrl == DC_CTRL_P
                i_mask = ctrl == DC_CTRL_I
                p_values = np.zeros(gen_pos.size, dtype=np.float64)
                p_values[p_mask] = gen[gen_mask, DC_GEN_COLS["p_set"]][p_mask]
                p_values[i_mask] = gen[gen_mask, DC_GEN_COLS["i_set"]][i_mask] * v[i_mask]
                current = np.divide(
                    p_values,
                    v,
                    out=np.zeros_like(p_values),
                    where=np.abs(v) > self.min_voltage,
                )
                active_rows = np.nonzero(gen_mask)[0]
                non_slack = p_mask | i_mask
                gen[active_rows[non_slack], DC_GEN_COLS["p"]] = p_values[non_slack]
                gen[active_rows[non_slack], DC_GEN_COLS["current"]] = current[non_slack]
                P_inj += np.bincount(
                    gen_pos[non_slack], weights=-p_values[non_slack], minlength=P_inj.size
                )

        if dcdc.size:
            dcdc[:, [DC_DCDC_COLS["i_p"], DC_DCDC_COLS["j_p"], DC_DCDC_COLS["i_c"], DC_DCDC_COLS["j_c"]]] = 0.0
        if self.N_dcdc:
            dcdc_vi = V_final[self.dcdc_i]
            dcdc_vj = V_final[self.dcdc_j]
            if getattr(self, "dcdc_eliminate_pj", False):
                dcdc_i_p = Pdc_final
                dcdc_j_p = self._dcdc_j_power_from_loss(dcdc_i_p, dcdc_vi, dcdc_vj)[0]
            else:
                dcdc_i_p = Pdc_final[0::2]
                dcdc_j_p = Pdc_final[1::2]
            dcdc_i_c = np.divide(
                dcdc_i_p,
                dcdc_vi,
                out=np.zeros_like(dcdc_i_p),
                where=np.abs(dcdc_vi) > self.min_voltage,
            )
            dcdc_j_c = np.divide(
                dcdc_j_p,
                dcdc_vj,
                out=np.zeros_like(dcdc_j_p),
                where=np.abs(dcdc_vj) > self.min_voltage,
            )
            dcdc[self.dcdc_idx, DC_DCDC_COLS["i_p"]] = dcdc_i_p
            dcdc[self.dcdc_idx, DC_DCDC_COLS["j_p"]] = dcdc_j_p
            dcdc[self.dcdc_idx, DC_DCDC_COLS["i_c"]] = dcdc_i_c
            dcdc[self.dcdc_idx, DC_DCDC_COLS["j_c"]] = dcdc_j_c
            P_inj += np.bincount(self.dcdc_i, weights=dcdc_i_p, minlength=P_inj.size)
            P_inj += np.bincount(self.dcdc_j, weights=dcdc_j_p, minlength=P_inj.size)

        for node, gens in self.slack_gen_info.items():
            if not gens:
                continue
            share = P_inj[node] / len(gens)
            v = V_final[node]
            current = share / v if abs(v) > self.min_voltage else 0.0
            for gen_ref in gens:
                if isinstance(gen_ref, (int, np.integer)):
                    gen[int(gen_ref), DC_GEN_COLS["p"]] = share
                    gen[int(gen_ref), DC_GEN_COLS["current"]] = current

        self.result = {
            "bus": bus,
            "branch": branch,
            "load": load,
            "gen": gen,
            "zero_branch": zero_branch,
            "switch": switch,
            "break": breaker,
            "dcdc": dcdc,
        }
    def _write_ppc_result_to_network(self) -> None:
        """Copy ppc results back to an optional DCPowerNetwork object."""
        network = getattr(self, "_network_writeback", None)
        if network is None or not self.result:
            return

        def rows_by_idx(key, idx_col):
            rows = self.result.get(key)
            if rows is None or len(rows) == 0:
                return {}
            return {int(row[idx_col]): row for row in rows}

        bus_by_idx = rows_by_idx("bus", DC_BUS_COLS["idx"])
        slack_node_ids = {
            int(node_id)
            for node_id, pos in self.alive_node_dict.items()
            if int(pos) in self.slack_nodes
        }
        for node in getattr(network, "nodes", []):
            row = bus_by_idx.get(int(getattr(node, "idx", -1)))
            node.voltage = 0.0 if row is None else float(row[DC_BUS_COLS["voltage"]])
            node.is_alive = int(getattr(node, "idx", -1)) in self.alive_node_dict
            node.is_slack = int(getattr(node, "idx", -1)) in slack_node_ids

        def assign_devices(devices, key, idx_col, attrs):
            row_by_idx = rows_by_idx(key, idx_col)
            try:
                device_iter = iter(devices)
            except Exception:
                return
            for dev in device_iter:
                row = row_by_idx.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    for attr, _col in attrs:
                        setattr(dev, attr, 0.0)
                    dev.is_alive = False
                    continue
                for attr, col in attrs:
                    setattr(dev, attr, float(row[col]))
                run_stat = int(getattr(dev, "run_stat", 1)) == 1
                status_ok = int(getattr(dev, "status", 1)) == 1
                i_node = getattr(dev, "i_node", getattr(dev, "node", None))
                j_node = getattr(dev, "j_node", i_node)
                dev.is_alive = (
                    run_stat
                    and status_ok
                    and int(i_node) in self.alive_node_dict
                    and int(j_node) in self.alive_node_dict
                )

        assign_devices(
            getattr(network, "branches", []),
            "branch",
            DC_BRANCH_COLS["idx"],
            (("i_p", DC_BRANCH_COLS["i_p"]), ("j_p", DC_BRANCH_COLS["j_p"]), ("current", DC_BRANCH_COLS["current"])),
        )
        assign_devices(
            getattr(network, "zero_branches", []),
            "zero_branch",
            DC_ZERO_BRANCH_COLS["idx"],
            (("p", DC_ZERO_BRANCH_COLS["p"]), ("current", DC_ZERO_BRANCH_COLS["current"])),
        )
        assign_devices(
            getattr(network, "switches", []),
            "switch",
            DC_SWITCH_COLS["idx"],
            (("p", DC_SWITCH_COLS["p"]), ("current", DC_SWITCH_COLS["current"])),
        )
        assign_devices(
            getattr(network, "breakers", []),
            "break",
            DC_BREAK_COLS["idx"],
            (("p", DC_BREAK_COLS["p"]), ("current", DC_BREAK_COLS["current"])),
        )
        assign_devices(
            getattr(network, "loads", []),
            "load",
            DC_LOAD_COLS["idx"],
            (("p", DC_LOAD_COLS["p"]), ("current", DC_LOAD_COLS["current"])),
        )
        assign_devices(
            getattr(network, "generators", []),
            "gen",
            DC_GEN_COLS["idx"],
            (("p", DC_GEN_COLS["p"]), ("current", DC_GEN_COLS["current"])),
        )
        assign_devices(
            getattr(network, "dcdc_converters", []),
            "dcdc",
            DC_DCDC_COLS["idx"],
            (
                ("i_p", DC_DCDC_COLS["i_p"]),
                ("j_p", DC_DCDC_COLS["j_p"]),
                ("i_c", DC_DCDC_COLS["i_c"]),
                ("j_c", DC_DCDC_COLS["j_c"]),
            ),
        )

    def _build_lf_result_from_ppc(self) -> DCLFResult:
        # 与 ac_lf 同构：按列向量化抽取 + 单次 searchsorted 做节点电压查表，
        # 避免每行 Python float/int 转换与 dict 查询。
        result = DCLFResult()
        result.arrays = dict(self.result)
        bus_rows = self.result.get("bus")
        if bus_rows is None or len(bus_rows) == 0:
            return result

        bus_volt = bus_rows[:, DC_BUS_COLS["voltage"]]
        bus_idx_col = bus_rows[:, DC_BUS_COLS["idx"]].astype(np.int64)
        sort_order = np.argsort(bus_idx_col, kind="stable")
        sorted_idx = bus_idx_col[sort_order]

        def _lookup_voltage(col_array):
            if col_array.size == 0:
                return np.zeros(0, dtype=np.float64)
            nids = col_array.astype(np.int64)
            if sorted_idx.size == 0:
                return np.zeros(nids.shape, dtype=np.float64)
            pos = np.searchsorted(sorted_idx, nids)
            pos_clip = np.clip(pos, 0, sorted_idx.size - 1)
            hit = sorted_idx[pos_clip] == nids
            bus_row = np.where(hit, sort_order[pos_clip], 0)
            return np.where(hit, bus_volt[bus_row], 0.0)

        def _name_list(key, n):
            names = self.ppc.get(key)
            if names is None or len(names) != n:
                return np.arange(n).astype(str).tolist()
            if isinstance(names, np.ndarray):
                return names.astype(str).tolist()
            return [str(x) for x in names]

        n_bus = len(bus_rows)
        bus_names = _name_list("bus_name", n_bus)
        result.nodes = {
            name: SimpleNamespace(volt=v)
            for name, v in zip(bus_names, bus_volt.tolist())
        }

        def _build_two_port(rows, name_key, p_i, p_j, c_col, n_i, n_j, target, *, neg_j_current=False):
            if rows is None or len(rows) == 0:
                return
            names = _name_list(name_key, len(rows))
            i_p = rows[:, p_i].tolist()
            j_p = rows[:, p_j].tolist()
            currents = rows[:, c_col]
            i_c = currents.tolist()
            j_c = (-currents).tolist() if neg_j_current else currents.tolist()
            i_v = _lookup_voltage(rows[:, n_i]).tolist()
            j_v = _lookup_voltage(rows[:, n_j]).tolist()
            for name, ip, ic, iv, jp, jc, jv in zip(names, i_p, i_c, i_v, j_p, j_c, j_v):
                target[name] = SimpleNamespace(
                    i_p=ip, i_c=ic, i_v=iv,
                    j_p=jp, j_c=jc, j_v=jv,
                )

        def _build_two_port_separate_currents(rows, name_key, p_i, p_j, c_i, c_j, n_i, n_j, target):
            if rows is None or len(rows) == 0:
                return
            names = _name_list(name_key, len(rows))
            i_p = rows[:, p_i].tolist()
            j_p = rows[:, p_j].tolist()
            i_c = rows[:, c_i].tolist()
            j_c = rows[:, c_j].tolist()
            i_v = _lookup_voltage(rows[:, n_i]).tolist()
            j_v = _lookup_voltage(rows[:, n_j]).tolist()
            for name, ip, ic, iv, jp, jc, jv in zip(names, i_p, i_c, i_v, j_p, j_c, j_v):
                target[name] = SimpleNamespace(
                    i_p=ip, i_c=ic, i_v=iv,
                    j_p=jp, j_c=jc, j_v=jv,
                )

        def _build_single_port(rows, name_key, p_col, c_col, n_col, target):
            if rows is None or len(rows) == 0:
                return
            names = _name_list(name_key, len(rows))
            i_p = rows[:, p_col].tolist()
            i_c = rows[:, c_col].tolist()
            i_v = _lookup_voltage(rows[:, n_col]).tolist()
            for name, p, c, v in zip(names, i_p, i_c, i_v):
                target[name] = SimpleNamespace(i_p=p, i_c=c, i_v=v)

        _build_two_port(
            self.result.get("branch"), "branch_name",
            DC_BRANCH_COLS["i_p"], DC_BRANCH_COLS["j_p"], DC_BRANCH_COLS["current"],
            DC_BRANCH_COLS["i_node"], DC_BRANCH_COLS["j_node"], result.branches,
            neg_j_current=True,
        )
        # DC/DC has separate i_c / j_c columns and i_p / j_p.
        _build_two_port_separate_currents(
            self.result.get("dcdc"), "dcdc_name",
            DC_DCDC_COLS["i_p"], DC_DCDC_COLS["j_p"],
            DC_DCDC_COLS["i_c"], DC_DCDC_COLS["j_c"],
            DC_DCDC_COLS["i_node"], DC_DCDC_COLS["j_node"], result.dcdc_converters,
        )
        _build_single_port(
            self.result.get("zero_branch"), "zero_branch_name",
            DC_ZERO_BRANCH_COLS["p"], DC_ZERO_BRANCH_COLS["current"],
            DC_ZERO_BRANCH_COLS["i_node"], result.zero_branches,
        )
        _build_single_port(
            self.result.get("break"), "break_name",
            DC_BREAK_COLS["p"], DC_BREAK_COLS["current"],
            DC_BREAK_COLS["i_node"], result.breakers,
        )
        _build_single_port(
            self.result.get("gen"), "gen_name",
            DC_GEN_COLS["p"], DC_GEN_COLS["current"],
            DC_GEN_COLS["node"], result.generators,
        )
        _build_single_port(
            self.result.get("load"), "load_name",
            DC_LOAD_COLS["p"], DC_LOAD_COLS["current"],
            DC_LOAD_COLS["node"], result.loads,
        )
        return result

    def _summary_node_ids(self):
        node_ids = np.full(self.N, -1, dtype=np.int32)
        for node_id, pos in self.alive_node_dict.items():
            pos = int(pos)
            node_id = int(node_id)
            if 0 <= pos < self.N and (node_ids[pos] < 0 or node_id < node_ids[pos]):
                node_ids[pos] = node_id
        if np.any(node_ids < 0) and getattr(self, "alive_nodes", None):
            for pos, node in enumerate(self.alive_nodes[: self.N]):
                if node_ids[pos] < 0:
                    node_ids[pos] = int(getattr(node, "idx", pos))
        return node_ids

    def _write_summary_result(self):
        x = self.x
        voltage = x[:self.N].copy()
        self.result = {
            "node_id": self._summary_node_ids(),
            "voltage": voltage,
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
            },
        }
        self.lf_result = None

    def _write_back(self):
        """结果回填；数值计算批量完成，Python 循环只负责对象属性赋值。"""
        self._write_back_ppc()
        if self.result_mode == "none":
            self.result = {}
            self.lf_result = None
            return
        if self.result_mode == "summary":
            self._write_summary_result()
            return

        self._write_ppc_result_to_network()
        if self.result_mode != "array" and not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result()
        return

    def _build_lf_result(self) -> DCLFResult:
        return self._build_lf_result_from_ppc()

    def _build_newton_system(self, x: np.ndarray, *, return_jacobian=True, jacobian_format="csc"):
        """Compute residual and Jacobian together for one DC Newton iteration."""
        terms = self._eval_newton_terms(self.G, x)
        F = self._get_f_from_terms(x, terms)
        J = self._get_jacobi_from_terms(
            self.G,
            x,
            terms,
            matrix_format=jacobian_format,
            build_matrix=return_jacobian,
        )
        return F, J

    def run(self, result_mode=None) -> int:
        """执行直流 Newton 迭代并在收敛后回填结果。"""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
            self._cache_csr_jacobian_pattern = self.result_mode == "full"
            self.keep_node_objects = False
            self.alive_nodes = []
        if self.x.size == 0 or self.G is None:
            self.prepare()
        return self._run_newton_raphson()

    def _run_newton_raphson(self) -> int:
        """执行牛顿-拉夫逊迭代求解。"""
        self.converged = False
        self.iterations = 0
        x = self.x.copy()

        for it in range(self.max_iter):
            self.iterations += 1
            F, J = self._build_newton_system(x)
            self.normF = np.linalg.norm(F, np.inf)
            if self.verbose:
                print(f"Iter {it + 1}: |F| = {self.normF:.2e}")

            if self.normF < self.tol:
                if self.verbose:
                    print(f"收敛于第 {it + 1} 次迭代")
                self.converged = True
                self.x = x
                self._write_back()
                return 0

            try:
                factor = _factor_jacobian(J, self._linear_solver_resolved, self._linear_solver_fn)
                delta = factor.solve(F)
            except Exception:
                _OPTIONAL_SPARSE_SOLVERS.pop(self._linear_solver_resolved, None)
                _OPTIONAL_SPARSE_MISSING.add(self._linear_solver_resolved)
                self._linear_solver_resolved = "scipy"
                self._linear_solver_fn = spsolve
                delta = spsolve(J, F)

            # 与 AC 潮流一致：方程定义为 F(x)=0，使用 x_new = x - J^{-1}F。
            x -= delta

        if self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未收敛")
        self.x = x
        self._write_back()
        return -1

def print_dc_result(calc: DCPowerFlowCalc, rc: int) -> None:
    # 9. 输出详细结果
    print("\n===输出直流电网潮流计算结果===")

    if calc.result:
        def _names(key, count):
            values = calc.ppc.get(key)
            if values is None:
                return np.asarray([str(i) for i in range(count)], dtype=object)
            return values

        bus = calc.result["bus"]
        bus_names = _names("bus_name", bus.shape[0])
        slack_node_ids = {
            int(node_id)
            for node_id, pos in calc.alive_node_dict.items()
            if int(pos) in calc.slack_nodes
        }

        print("\n1. 节点电压 (pu):")
        for row, name in zip(bus, bus_names):
            node_idx = int(row[DC_BUS_COLS["idx"]])
            flag = " (松弛节点)" if node_idx in slack_node_ids else ""
            print(f"   节点 {node_idx} {name}: {row[DC_BUS_COLS['voltage']]:.6f}{flag}")

        branch = calc.result["branch"]
        branch_names = _names("branch_name", branch.shape[0])
        print("\n2. 普通电阻支路信息:")
        for row, name in zip(branch, branch_names):
            loss = row[DC_BRANCH_COLS["i_p"]] + row[DC_BRANCH_COLS["j_p"]]
            print(
                f"   支路 {int(row[DC_BRANCH_COLS['idx']])} {name} "
                f"({int(row[DC_BRANCH_COLS['i_node']])}->{int(row[DC_BRANCH_COLS['j_node']])}, "
                f"r={row[DC_BRANCH_COLS['r']]}pu):"
            )
            print(f"     电流: {row[DC_BRANCH_COLS['current']]:.6f} pu")
            print(f"     送端功率: {row[DC_BRANCH_COLS['i_p']]:.6f} pu, 受端功率: {row[DC_BRANCH_COLS['j_p']]:.6f} pu")
            print(f"     损耗功率: {loss:.6f} pu")

        zero_branch = calc.result["zero_branch"]
        zero_names = _names("zero_branch_name", zero_branch.shape[0])
        print("\n3. 零阻抗支路信息:")
        for row, name in zip(zero_branch, zero_names):
            print(
                f"   零阻抗支路 {int(row[DC_ZERO_BRANCH_COLS['idx']])} {name} "
                f"({int(row[DC_ZERO_BRANCH_COLS['i_node']])}->{int(row[DC_ZERO_BRANCH_COLS['j_node']])}):"
            )
            print(f"     电流: {row[DC_ZERO_BRANCH_COLS['current']]:.6f} pu, 功率: {row[DC_ZERO_BRANCH_COLS['p']]:.6f} pu")

        switch = calc.result["switch"]
        switch_names = _names("switch_name", switch.shape[0])
        print("\n4. 开关信息:")
        for row, name in zip(switch, switch_names):
            status = "闭合" if int(row[DC_SWITCH_COLS["status"]]) == 1 else "断开"
            i_node = int(row[DC_SWITCH_COLS["i_node"]])
            j_node = int(row[DC_SWITCH_COLS["j_node"]])
            print(
                f"   开关 {int(row[DC_SWITCH_COLS['idx']])} {name} "
                f"({i_node}->{j_node}, 状态:{status}):"
            )
            print(f"     电流: {row[DC_SWITCH_COLS['current']]:.6f} pu, 功率: {row[DC_SWITCH_COLS['p']]:.6f} pu")

        dcdc = calc.result["dcdc"]
        dcdc_names = _names("dcdc_name", dcdc.shape[0])
        print("\n5. DC-DC变流器信息:")
        for row, name in zip(dcdc, dcdc_names):
            print(
                f"   变流器 {int(row[DC_DCDC_COLS['idx']])} {name} "
                f"({int(row[DC_DCDC_COLS['i_node']])}->{int(row[DC_DCDC_COLS['j_node']])}):"
            )
            print(f"     送端功率: {row[DC_DCDC_COLS['i_p']]:.6f} pu, 送端电流: {row[DC_DCDC_COLS['i_c']]:.6f} pu")
            print(f"     受端功率: {row[DC_DCDC_COLS['j_p']]:.6f} pu, 受端电流: {row[DC_DCDC_COLS['j_c']]:.6f} pu")
            print(f"     损耗功率: {row[DC_DCDC_COLS['i_p']] + row[DC_DCDC_COLS['j_p']]:.6f} pu")

        load = calc.result["load"]
        load_names = _names("load_name", load.shape[0])
        print("\n6. 负荷信息:")
        for row, name in zip(load, load_names):
            print(f"   负荷 {int(row[DC_LOAD_COLS['idx']])} {name} (节点{int(row[DC_LOAD_COLS['node']])}):")
            print(f"     消耗功率: {row[DC_LOAD_COLS['p']]:.6f} pu, 电流: {row[DC_LOAD_COLS['current']]:.6f} pu")

        gen = calc.result["gen"]
        gen_names = _names("gen_name", gen.shape[0])
        print("\n7. 发电机信息:")
        for row, name in zip(gen, gen_names):
            print(f"   发电机 {int(row[DC_GEN_COLS['idx']])} {name} (节点{int(row[DC_GEN_COLS['node']])}):")
            print(f"     出力功率: {row[DC_GEN_COLS['p']]:.6f} pu, 电流: {row[DC_GEN_COLS['current']]:.6f} pu")

        print("\n8. 计算收敛信息:")
        print(
            f"   收敛状态: {'✓ 已收敛' if calc.converged else '✗ 未收敛'}, "
            f"返回码: {rc}, 迭代次数: {calc.iterations}, 最终残差: {calc.normF:.2e}"
        )

        total_gen_power = float(np.sum(gen[:, DC_GEN_COLS["p"]])) if gen.size else 0.0
        total_load_power = float(np.sum(load[:, DC_LOAD_COLS["p"]])) if load.size else 0.0
        print("\n9. 功率平衡校验:")
        print(f"   总发电功率: {total_gen_power:.6f} pu")
        print(f"   总负荷功率: {total_load_power:.6f} pu")
        print(f"   网损: {total_gen_power - total_load_power:.6f} pu")
        return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DC power flow")
    parser.add_argument("file", nargs="?", default=str(model_file("dc", "dc_net_30.e")), help="DC E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument("--linear-solver", default="pyklu")
    parser.add_argument("--result-mode", default="full", choices=("full", "array", "summary", "none"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ppc = load_dc_ppc_from_e_file(args.file)
    calc = DCPowerFlowCalc(
        ppc,
        parameter_file=args.para,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        linear_solver=args.linear_solver,
        result_mode=args.result_mode,
        verbose=not args.quiet,
    )
    rc = calc.run()
    if not args.quiet and calc.result_mode == "full":
        print_dc_result(calc, rc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
