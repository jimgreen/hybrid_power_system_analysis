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
import contextlib
import io
import importlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

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

from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from model import topology as network_topology
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BUS_COLS as DC_BUS_COLS,
    CTRL_I as DC_CTRL_I,
    CTRL_P as DC_CTRL_P,
    CTRL_V as DC_CTRL_V,
    DCDC_COLS as DC_DCDC_COLS,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    BREAK_COLS as DC_BREAK_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
    build_dc_network_from_ppc,
    build_dc_ppc_from_e_file as _build_dc_ppc_from_e_file,
)
from model.ppc_topology import build_dc_ppc_with_topology_from_e_file, ensure_dc_ppc_topology


@dataclass
class DCLFResult:
    branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    zero_branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    breakers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    dcdc_converters: Dict[str, SimpleNamespace] = field(default_factory=dict)
    generators: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)

def _device_key(device) -> str:
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def load_dc_ppc_from_e_file(file_name) -> Dict:
    """Read a DC E file into PPC with topology arrays attached."""
    return build_dc_ppc_with_topology_from_e_file(file_name)


def _dc_network_from_ppc(ppc):
    ensure_dc_ppc_topology(ppc)
    network = build_dc_network_from_ppc(ppc)
    network_topology.apply_dc_topology_arrays(network, ppc["_topology_arrays"])
    return network


_OPTIONAL_SPARSE_SOLVERS = {}
_OPTIONAL_SPARSE_MISSING = set()
_OPTIONAL_SOLVER_CANDIDATES = {
    "pypardiso": ("pypardiso", "spsolve"),
    "umfpack": ("scikits.umfpack", "spsolve"),
    "sksparse.klu.klu_solve": ("sksparse.klu", "klu_solve"),
    "klu_solve": ("sksparse.klu", "klu_solve"),
    "pyklu": ("PyKLU", "Klu"),
    "klu": ("sksparse.klu", "spsolve"),
    "klu_alt": ("klu", "solve"),
}


def _load_named_sparse_solver(solver_name):
    """Return a named optional sparse solver when installed."""
    solver_name = str(solver_name).strip().lower()
    if solver_name in _OPTIONAL_SPARSE_SOLVERS:
        return _OPTIONAL_SPARSE_SOLVERS[solver_name]
    if solver_name in _OPTIONAL_SPARSE_MISSING:
        return None

    candidate_names = ("pyklu", "sksparse.klu.klu_solve", "klu", "klu_alt", "pypardiso") if solver_name == "auto" else (solver_name,)
    for candidate_name in candidate_names:
        module_name, func_name = _OPTIONAL_SOLVER_CANDIDATES.get(candidate_name, (None, None))
        if module_name is None:
            continue
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
        except (ImportError, ValueError):
            continue
        try:
            module = importlib.import_module(module_name)
            solver = getattr(module, func_name, None)
        except Exception:
            continue
        if solver is not None:
            if candidate_name == "pyklu":
                klu_cls = solver

                def solver(matrix, rhs, _klu_cls=klu_cls):
                    return _klu_cls(matrix.tocsc()).solve(rhs)

            _OPTIONAL_SPARSE_SOLVERS[solver_name] = solver
            return solver

    _OPTIONAL_SPARSE_MISSING.add(solver_name)
    return None


def solve_sparse_system(matrix, rhs, solver_name="scipy"):
    """Solve a sparse linear system, preferring optional high-performance bindings."""
    solver_name = str(solver_name or "scipy").strip().lower()
    if solver_name in {"scipy", "superlu", "default"}:
        return spsolve(matrix, rhs)

    solver = _load_named_sparse_solver(solver_name)
    if solver is not None:
        try:
            return solver(matrix, rhs)
        except Exception:
            _OPTIONAL_SPARSE_SOLVERS.pop(solver_name, None)
            _OPTIONAL_SPARSE_MISSING.add(solver_name)
    return spsolve(matrix, rhs)


def _resolve_linear_solver(solver_name):
    """Return (resolved_name, callable) for a requested linear solver.

    "auto" picks the best available optional solver (pyklu, then KLU bindings,
    then pypardiso). Unknown / unavailable names silently fall back to scipy
    SuperLU.
    """
    name = str(solver_name or "scipy").strip().lower()
    if name in {"scipy", "superlu", "default"}:
        return "scipy", spsolve
    solver = _load_named_sparse_solver(name)
    if solver is None:
        return "scipy", spsolve
    return name, solver


class _CallableFactor:
    """Wraps a (matrix, rhs)->solution callable into a .solve(rhs) factor object."""

    __slots__ = ("_matrix", "_fn")

    def __init__(self, matrix, fn):
        self._matrix = matrix
        self._fn = fn

    def solve(self, rhs):
        return self._fn(self._matrix, rhs)


def _get_pyklu_cls():
    """Lazily import PyKLU.Klu and cache the class (or None when unavailable)."""
    cache = _OPTIONAL_SPARSE_SOLVERS
    if "__pyklu_cls__" in cache:
        return cache["__pyklu_cls__"]
    try:
        from PyKLU import Klu as klu_cls  # type: ignore
    except Exception:
        klu_cls = None
    cache["__pyklu_cls__"] = klu_cls
    return klu_cls


def _factor_jacobian(matrix, resolved_name, solver_fn):
    """Build a factored solver object that supports repeated .solve(b) calls."""
    if resolved_name in {"scipy", "superlu", "default"}:
        return splu(matrix.tocsc())
    if resolved_name in {"pyklu", "auto"}:
        klu_cls = _get_pyklu_cls()
        if klu_cls is not None:
            return klu_cls(matrix.tocsc())
    return _CallableFactor(matrix, solver_fn)


def find_spanning_tree_edges(edges, n_nodes):
    """查找生成树的边（Kruskal算法）"""
    parent = np.arange(n_nodes, dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False
    tree = []
    for idx, (u, v) in enumerate(edges):
        if union(u, v):
            tree.append(idx)
    return tree

class DCPowerFlowCalc:
    """直流潮流计算器，使用节点电压、零阻抗 phi 和 DCDC 端口功率统一求解。

    状态向量布局为 ``V``、``phi``、``Pdc_i/Pdc_j``。节点功率平衡、定压
    节点约束、零阻抗电压相等约束、phi 参考约束和 DC-DC 控制/损耗方程
    在同一个 Newton 系统中求解。
    """

    _DIRECT_PPC_STATIC_ATTRS = (
        "alive_nodes",
        "alive_node_dict",
        "alive_node_ids",
        "_alive_node_lookup",
        "_dc_jac_csr_indices",
        "_dc_jac_csr_indptr",
        "_dc_jac_csr_data",
        "_dc_jac_raw_data",
        "_dc_jac_raw_to_csr_pos",
        "_dc_jac_unknown_slice",
        "_dc_jac_unknown_row_nodes",
        "_dc_jac_unknown_col_nodes",
        "_dc_jac_unknown_g_data",
        "_dc_jac_unknown_diag_mask",
        "_dc_jac_zero_i_slice",
        "_dc_jac_zero_j_slice",
        "_dc_jac_dcdc_i_slice",
        "_dc_jac_dcdc_j_slice",
        "_dc_jac_known_slice",
        "_dc_jac_zero_con_slice",
        "_dc_jac_phi_fix_slice",
        "_dc_jac_dcdc_ctrl_p_slice",
        "_dc_jac_dcdc_ctrl_v_slice",
        "_dc_jac_dcdc_ctrl_i_slice",
        "_dc_jac_dcdc_loss_slice",
        "N",
        "P_const",
        "I_shunt",
        "slack_gen_info",
        "branch_idx",
        "branch_i",
        "branch_j",
        "branch_r",
        "alive_loads",
        "alive_generators",
        "G",
        "zero_edges",
        "comp_nodes",
        "comp_tree_edges",
        "N_phi",
        "ref_phi_idx",
        "zero_branch_info",
        "zero_type",
        "zero_dev_idx",
        "zero_i",
        "zero_j",
        "zero_phi_a",
        "zero_phi_b",
        "zero_con_i",
        "zero_con_j",
        "dcdc_idx",
        "dcdc_i",
        "dcdc_j",
        "dcdc_ctrl_code",
        "dcdc_p_set",
        "dcdc_i_set",
        "dcdc_v_set",
        "dcdc_r1",
        "dcdc_r2",
        "N_dcdc",
        "dcdc_ctrl",
        "alive_dcdc_tuples",
        "slack_nodes",
        "slack_node_arr",
        "slack_value_arr",
        "total_vars",
        "unknown_nodes",
        "n_unknown",
        "n_known",
        "node_eq",
        "n_zero_constraint",
        "n_phi_fix",
        "n_dcdc",
        "total_eq",
        "eq_unknown_start",
        "eq_known_start",
        "eq_zero_start",
        "eq_phi_start",
        "eq_dcdc_start",
        "unknown_map",
        "zero_con_rows",
        "phi_fix_rows",
        "dcdc_seq",
        "dcdc_p_col",
        "dcdc_q_col",
        "dcdc_eq_ctrl",
        "dcdc_eq_loss",
        "dcdc_ctrl_p_mask",
        "dcdc_ctrl_v_mask",
        "dcdc_ctrl_i_mask",
        "dcdc_ones",
        "zero_i_eq",
        "zero_j_eq",
        "zero_i_unknown_mask",
        "zero_j_unknown_mask",
        "zero_i_unknown_count",
        "zero_j_unknown_count",
        "zero_i_rows_jac",
        "zero_i_cols_jac",
        "zero_j_rows_jac",
        "zero_j_cols_jac",
        "known_rows_jac",
        "known_cols_jac",
        "known_data_jac",
        "zero_con_rows_jac",
        "zero_con_cols_jac",
        "zero_con_data_jac",
        "phi_fix_cols_jac",
        "phi_fix_data_jac",
        "dcdc_i_eq",
        "dcdc_j_eq",
        "dcdc_i_unknown_mask",
        "dcdc_j_unknown_mask",
        "dcdc_i_unknown_idx",
        "dcdc_j_unknown_idx",
        "dcdc_i_eq_rows_jac",
        "dcdc_j_eq_rows_jac",
        "dcdc_i_eq_cols_jac",
        "dcdc_j_eq_cols_jac",
        "dcdc_i_eq_data_jac",
        "dcdc_j_eq_data_jac",
        "dcdc_ctrl_p_count",
        "dcdc_ctrl_v_count",
        "dcdc_ctrl_i_count",
        "dcdc_ctrl_p_data_jac",
        "dcdc_ctrl_v_data_jac",
        "dcdc_ctrl_i_rows_jac",
        "dcdc_ctrl_i_cols_jac",
        "dcdc_ctrl_i_data_jac",
        "dcdc_loss_rows_jac",
        "dcdc_loss_cols_jac",
    )
    _DIRECT_PPC_SHAPE_KEYS = ("bus", "branch", "load", "gen", "zero_branch", "switch", "break", "dcdc")

    def __init__(
        self,
        model,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        island=None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        algorithm: str = "nr",
        keep_node_objects: bool = True,
        linear_solver: str = "auto",
        writeback_network=None,
        result_mode: str = "full",
        jacobian_refresh_period: int = 1,
    ):
        algorithm = str(algorithm).strip().lower()
        if algorithm not in {"nr"}:
            raise ValueError(f"Unsupported DC power-flow algorithm: {algorithm!r}")
        self._direct_ppc_mode = isinstance(model, dict) and model.get("format") == "dc_ppc_v1"
        self.model = writeback_network if self._direct_ppc_mode else model
        self.ppc = model if self._direct_ppc_mode else getattr(model, "ppc", None)
        self.array_mode = isinstance(self.ppc, dict) and self.ppc.get("format") == "dc_ppc_v1"
        self._network_writeback = writeback_network if self._direct_ppc_mode else (model if self.array_mode else None)
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self.runtime_params = self.params
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.min_voltage = self.params.min_voltage
        self.algorithm = algorithm
        self.used_algorithm = algorithm
        self.target_island = island
        # 用户传入的求解器名原样保留，便于上层日志/测试断言；实际 callable
        # 由 _resolve_linear_solver 决定，未安装时回退 SuperLU。
        self.linear_solver = str(linear_solver or "scipy").strip().lower()
        self._linear_solver_resolved, self._linear_solver_fn = _resolve_linear_solver(self.linear_solver)
        # jacobian_refresh_period>=2 触发 Honest-Newton：复用同一因子若干次。
        self.jacobian_refresh_period = max(1, int(jacobian_refresh_period or 1))
        self.result_mode = self._normalize_result_mode(result_mode)
        self.keep_node_objects = bool(keep_node_objects) and (
            not self.array_mode or self.result_mode == "full"
        )
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.verbose = False
        self.result: Dict = {}

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        mode = str(result_mode or "full").strip().lower()
        aliases = {
            "all": "full",
            "full": "full",
            "complete": "full",
            "array": "array",
            "arrays": "array",
            "ppc": "array",
            "summary": "summary",
            "minimal": "summary",
            "none": "none",
            "off": "none",
            "skip": "none",
        }
        if mode not in aliases:
            raise ValueError(f"Unsupported DC result_mode: {result_mode!r}")
        return aliases[mode]

    @staticmethod
    def _clone_static_value(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            return list(value)
        return value

    def _direct_ppc_shape_signature(self):
        return {
            key: tuple(self.ppc[key].shape)
            for key in self._DIRECT_PPC_SHAPE_KEYS
            if key in self.ppc
        }

    def _direct_ppc_array_id_signature(self):
        return {
            key: id(self.ppc[key])
            for key in self._DIRECT_PPC_SHAPE_KEYS
            if key in self.ppc
        }

    def _load_direct_ppc_static(self):
        static = self.ppc.get("_dc_pf_static")
        if not isinstance(static, dict):
            return None
        if static.get("format") != "dc_pf_static_v1":
            return None
        if static.get("keep_node_objects") != self.keep_node_objects:
            return None
        if static.get("shapes") != self._direct_ppc_shape_signature():
            return None
        if static.get("array_ids") != self._direct_ppc_array_id_signature():
            return None

        for name, value in static["attrs"].items():
            setattr(self, name, self._clone_static_value(value))
        self._lf_branch_devices = []
        self._lf_inactive_branch_devices = []
        return self.G, static["x"].copy()

    def _store_direct_ppc_static(self, x):
        attrs = {
            name: self._clone_static_value(getattr(self, name))
            for name in self._DIRECT_PPC_STATIC_ATTRS
            if hasattr(self, name)
        }
        self.ppc["_dc_pf_static"] = {
            "format": "dc_pf_static_v1",
            "keep_node_objects": self.keep_node_objects,
            "shapes": self._direct_ppc_shape_signature(),
            "array_ids": self._direct_ppc_array_id_signature(),
            "attrs": attrs,
            "x": x.copy(),
        }

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

    @staticmethod
    def _find_parent(parent, item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            item, parent[item] = parent[item], root
        return root

    @classmethod
    def _union_parent(cls, parent, left, right):
        left_root = cls._find_parent(parent, left)
        right_root = cls._find_parent(parent, right)
        if left_root != right_root:
            parent[right_root] = left_root

    def _live_ppc_terminal_pair(self, row, i_col, j_col, run_col, running_node_ids, status_col=None):
        if row[run_col] != 1:
            return None
        if status_col is not None and row[status_col] != 1:
            return None
        i_node = int(row[i_col])
        j_node = int(row[j_col])
        if i_node == j_node or i_node not in running_node_ids or j_node not in running_node_ids:
            return None
        return i_node, j_node

    @staticmethod
    def _direct_node_position_lookup(node_ids):
        if node_ids.size == 0:
            return None, {}
        node_ids = node_ids.astype(np.int64, copy=False)
        min_id = int(node_ids.min())
        max_id = int(node_ids.max())
        if min_id >= 0 and max_id <= max(1_000_000, int(node_ids.size) * 8):
            lookup = np.full(max_id + 1, -1, dtype=np.int32)
            lookup[node_ids.astype(np.intp)] = np.arange(node_ids.size, dtype=np.int32)
            return lookup, None
        return None, {int(node_id): int(pos) for pos, node_id in enumerate(node_ids)}

    @staticmethod
    def _map_direct_node_positions(node_values, dense_lookup, dict_lookup):
        node_values = np.asarray(node_values, dtype=np.int64)
        pos = np.full(node_values.shape, -1, dtype=np.int32)
        if dense_lookup is not None:
            valid = (node_values >= 0) & (node_values < dense_lookup.size)
            if np.any(valid):
                pos[valid] = dense_lookup[node_values[valid].astype(np.intp)]
            return pos
        return np.fromiter(
            (dict_lookup.get(int(node_id), -1) for node_id in node_values),
            dtype=np.int32,
            count=node_values.size,
        )

    def _direct_ppc_pair_positions(self, rows, cols, dense_lookup, dict_lookup, status_col=None):
        if rows is None or rows.size == 0:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        i_pos = self._map_direct_node_positions(rows[:, cols["i_node"]], dense_lookup, dict_lookup)
        j_pos = self._map_direct_node_positions(rows[:, cols["j_node"]], dense_lookup, dict_lookup)
        mask = (rows[:, cols["run_stat"]] == 1) & (i_pos >= 0) & (j_pos >= 0) & (i_pos != j_pos)
        if status_col is not None:
            mask &= rows[:, status_col] == 1
        return i_pos[mask].astype(np.int32, copy=False), j_pos[mask].astype(np.int32, copy=False)

    @staticmethod
    def _component_labels_from_edges(n_nodes, edge_parts):
        edge_parts = [(left, right) for left, right in edge_parts if left.size]
        if not edge_parts:
            return np.arange(n_nodes, dtype=np.int32)
        left = np.concatenate([part[0] for part in edge_parts])
        right = np.concatenate([part[1] for part in edge_parts])
        rows = np.concatenate((left, right))
        cols = np.concatenate((right, left))
        data = np.ones(rows.size, dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        _n_comp, labels = connected_components(graph, directed=False, return_labels=True)
        return labels.astype(np.int32, copy=False)

    def _prepare_direct_ppc_topology(self):
        """Build active DC solver-node mapping directly from dc_ppc_v1 arrays."""
        ppc = self.ppc
        ensure_dc_ppc_topology(ppc)
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
            if active_bus_pos.size:
                self.alive_node_dict = {}
                alive_ids = []
                alive_pos_values = []
                for solver_pos, bus_pos in enumerate(active_bus_pos):
                    start = int(topology.bus_node_offsets[bus_pos])
                    end = int(topology.bus_node_offsets[bus_pos + 1])
                    group_nodes = topology.node_ids[topology.bus_node_indices[start:end]].astype(np.int32, copy=False)
                    alive_ids.extend(int(node_id) for node_id in group_nodes)
                    alive_pos_values.extend([int(solver_pos)] * int(group_nodes.size))
                    for node_id in group_nodes:
                        self.alive_node_dict[int(node_id)] = int(solver_pos)
                self.alive_node_ids = np.asarray(alive_ids, dtype=np.int32)
                self.N = int(active_bus_pos.size)
                if self.alive_node_ids.size and np.all(self.alive_node_ids >= 0):
                    self._alive_node_lookup = np.full(int(self.alive_node_ids.max()) + 1, -1, dtype=np.int32)
                    self._alive_node_lookup[self.alive_node_ids.astype(np.intp)] = np.asarray(alive_pos_values, dtype=np.int32)
                else:
                    self._alive_node_lookup = np.array([], dtype=np.int32)
            else:
                self.alive_node_dict = {}
                self.alive_node_ids = np.array([], dtype=np.int32)
                self.N = 0
                self._alive_node_lookup = np.array([], dtype=np.int32)

        # Direct ppc result/writeback paths use alive_node_dict and result arrays; constructing
        # SimpleNamespace node facades here is pure cold-start overhead for large ppc cases.
        self.alive_nodes = []

    def prepare(self):
        """
        运行潮流计算（修正版，采用节点电位法处理零阻抗支路）
        变量：V (N个) + φ (N_phi个) + Pdc (N_dcdc个)
        方程：功率平衡（除松弛节点外各节点） + 松弛节点电压方程 + 零阻抗电压约束（树支） + φ参考固定 + DC-DC方程
        变量数与方程数严格相等。
        """
        if self.array_mode and self._direct_ppc_mode:
            cached = self._load_direct_ppc_static()
            if cached is not None:
                return cached
            self._prepare_direct_ppc_topology()
        else:
            bus_nodes = [] if self.keep_node_objects else [
                bus
                for bus in getattr(self.model, "buses", [])
                if getattr(bus, "isl_obj", None) is not None and bus.isl_obj.is_alive
            ]
            self.alive_nodes = bus_nodes or [
                node
                for node in self.model.nodes
                if node.isl_obj is not None and node.isl_obj.is_alive
            ]

            self.N = len(self.alive_nodes)
            self.alive_node_dict = {}
            for idx, node in enumerate(self.alive_nodes):
                for member in getattr(node, "nodes", ()):
                    self.alive_node_dict[int(member.idx)] = idx
                self.alive_node_dict.setdefault(int(node.idx), idx)
            self.alive_node_ids = np.asarray(list(self.alive_node_dict.keys()), dtype=np.int32)

        if self.N == 0:
            raise ValueError("电网中没有活节点")

        self.P_const = np.zeros(self.N, dtype=np.float64)   # 注入为正：P型发电机 - P型负荷
        self.I_shunt = np.zeros(self.N, dtype=np.float64)   # 消耗为正：负荷电流 - 发电电流
        self.slack_gen_info = {}
        node_lookup = self._alive_node_lookup_array() if self.array_mode else None

        # ---------- 1. 数据预处理 ----------
        if self.array_mode:
            ppc = self.ppc

            branch = ppc["branch"]
            if branch.size:
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
            if self.model is not None and hasattr(self.model, "branches"):
                self._lf_branch_devices = [self.model.branches[int(idx)] for idx in self.branch_idx]
                self._lf_inactive_branch_devices = [
                    self.model.branches[int(idx)] for idx in np.nonzero(~branch_mask)[0]
                ]
            else:
                self._lf_branch_devices = []
                self._lf_inactive_branch_devices = []

            load = ppc["load"]
            if load.size:
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
                gen_pos_all = self._map_nodes_with_lookup(gen[:, DC_GEN_COLS["node"]], node_lookup)
                gen_mask = (gen[:, DC_GEN_COLS["run_stat"]] == 1) & (gen_pos_all >= 0)
                gen_pos = gen_pos_all[gen_mask]
                gen_active = gen[gen_mask]
                gen_ctrl = gen_active[:, DC_GEN_COLS["control_type"]].astype(np.int8, copy=False)
                gen_rows = np.nonzero(gen_mask)[0]
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
                        slack_ref = int(row_idx) if self._direct_ppc_mode else self.model.generators[int(row_idx)]
                        self.slack_gen_info.setdefault(int(node), []).append(slack_ref)
                bad_ctrl = ~(p_mask | i_mask | v_mask)
                if np.any(bad_ctrl):
                    raise ValueError(f"未知发电机控制类型: {gen_ctrl[int(np.where(bad_ctrl)[0][0])]}")
            self.alive_loads = []
            self.alive_generators = []
        else:
            self.alive_branch_tuple = [
                (idx, self.alive_node_dict[br.i_node], self.alive_node_dict[br.j_node], float(br.r))
                for idx, br in enumerate(self.model.branches)
                if br.is_alive and br.i_node in self.alive_node_dict and br.j_node in self.alive_node_dict
            ]
            self.alive_loads = [
                (load, self.alive_node_dict[load.node])
                for load in self.model.loads
                if load.is_alive and load.node in self.alive_node_dict
            ]
            self.alive_generators = [
                (gen, self.alive_node_dict[gen.node])
                for gen in self.model.generators
                if gen.is_alive and gen.node in self.alive_node_dict
            ]

            # V型发电机提供电压参考；P/I型发电机进入节点功率方程。
            for gen, node in self.alive_generators:
                if gen.control_type == 'V':
                    self.slack_gen_info.setdefault(node, []).append(gen)
                elif gen.control_type == 'P':
                    self.P_const[node] += gen.p_set
                elif gen.control_type == 'I':
                    self.I_shunt[node] -= gen.i_set
                else:
                    raise ValueError(f"未知发电机控制类型: {gen.control_type}")

            load_nodes = []
            load_g = []
            for ld, node in self.alive_loads:
                pbase = float(getattr(ld, "pbase", 1.0))
                pv0 = pbase * ld.pv0
                pv1 = pbase * ld.pv1
                pv2 = pbase * ld.pv2
                self.P_const[node] -= pv0
                self.I_shunt[node] += pv1
                if pv2 != 0.0:
                    load_nodes.append(node)
                    load_g.append(pv2)
            load_nodes_arr = np.asarray(load_nodes, dtype=np.int32) if load_nodes else np.array([], dtype=np.int32)
            load_g_arr = np.asarray(load_g, dtype=np.float64) if load_nodes else np.array([], dtype=np.float64)

        # G 矩阵只包含线性电导；恒功率、恒电流和二次负荷项分开放入方程。
        rows_parts = []
        cols_parts = []
        data_parts = []
        if load_nodes_arr.size:
            rows_parts.append(load_nodes_arr)
            cols_parts.append(load_nodes_arr)
            data_parts.append(load_g_arr)

        if self.array_mode:
            if self.branch_idx.size:
                branch_g = 1.0 / self.branch_r
                rows_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_i, self.branch_j)))
                cols_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_j, self.branch_i)))
                data_parts.append(np.concatenate((branch_g, branch_g, -branch_g, -branch_g)))
        elif self.alive_branch_tuple:
            branch_arr = np.asarray(self.alive_branch_tuple, dtype=object)
            self.branch_idx = branch_arr[:, 0].astype(np.int32)
            self.branch_i = branch_arr[:, 1].astype(np.int32)
            self.branch_j = branch_arr[:, 2].astype(np.int32)
            self.branch_r = branch_arr[:, 3].astype(np.float64)
            if np.any(self.branch_r <= 0.0):
                bad = int(np.where(self.branch_r <= 0.0)[0][0])
                raise ValueError(f"支路电阻必须为正数: r={self.branch_r[bad]}")
            branch_g = 1.0 / self.branch_r
            rows_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_i, self.branch_j)))
            cols_parts.append(np.concatenate((self.branch_i, self.branch_j, self.branch_j, self.branch_i)))
            data_parts.append(np.concatenate((branch_g, branch_g, -branch_g, -branch_g)))
        else:
            self.branch_idx = self.branch_i = self.branch_j = np.array([], dtype=np.int32)
            self.branch_r = np.array([], dtype=np.float64)
        if not self.array_mode:
            alive_branch_idx = {int(idx) for idx in self.branch_idx}
            self._lf_branch_devices = [self.model.branches[int(idx)] for idx in self.branch_idx]
            self._lf_inactive_branch_devices = [
                br for idx, br in enumerate(self.model.branches) if idx not in alive_branch_idx
            ]

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
        if self.array_mode:
            zero_branch = self.ppc["zero_branch"]
            zero_parts = []
            if zero_branch.size:
                zero_i_all = self._map_nodes_with_lookup(zero_branch[:, DC_ZERO_BRANCH_COLS["i_node"]], node_lookup)
                zero_j_all = self._map_nodes_with_lookup(zero_branch[:, DC_ZERO_BRANCH_COLS["j_node"]], node_lookup)
                zero_mask = (
                    (zero_branch[:, DC_ZERO_BRANCH_COLS["run_stat"]] == 1)
                    & (zero_i_all >= 0)
                    & (zero_j_all >= 0)
                    & (zero_i_all != zero_j_all)
                )
                if np.any(zero_mask):
                    zero_parts.extend(
                        zip(
                            np.repeat("Z", int(np.count_nonzero(zero_mask))),
                            np.nonzero(zero_mask)[0].astype(np.int32),
                            zero_i_all[zero_mask],
                            zero_j_all[zero_mask],
                        )
                    )

            switch = self.ppc["switch"]
            if switch.size:
                switch_i_all = self._map_nodes_with_lookup(switch[:, DC_SWITCH_COLS["i_node"]], node_lookup)
                switch_j_all = self._map_nodes_with_lookup(switch[:, DC_SWITCH_COLS["j_node"]], node_lookup)
                switch_mask = (
                    (switch[:, DC_SWITCH_COLS["run_stat"]] == 1)
                    & (switch[:, DC_SWITCH_COLS["status"]] == 1)
                    & (switch_i_all >= 0)
                    & (switch_j_all >= 0)
                    & (switch_i_all != switch_j_all)
                )
                if np.any(switch_mask):
                    zero_parts.extend(
                        zip(
                            np.repeat("S", int(np.count_nonzero(switch_mask))),
                            np.nonzero(switch_mask)[0].astype(np.int32),
                            switch_i_all[switch_mask],
                            switch_j_all[switch_mask],
                        )
                    )
            self.zero_edges = [
                (str(tp), int(dev_idx), int(i_node), int(j_node))
                for tp, dev_idx, i_node, j_node in zero_parts
            ]
            breaker = self.ppc.get("break")
            if breaker is not None and breaker.size:
                break_i_all = self._map_nodes_with_lookup(breaker[:, DC_BREAK_COLS["i_node"]], node_lookup)
                break_j_all = self._map_nodes_with_lookup(breaker[:, DC_BREAK_COLS["j_node"]], node_lookup)
                break_mask = (
                    (breaker[:, DC_BREAK_COLS["run_stat"]] == 1)
                    & (breaker[:, DC_BREAK_COLS["status"]] == 1)
                    & (break_i_all >= 0)
                    & (break_j_all >= 0)
                    & (break_i_all != break_j_all)
                )
                if np.any(break_mask):
                    self.zero_edges.extend(
                        (str(tp), int(dev_idx), int(i_node), int(j_node))
                        for tp, dev_idx, i_node, j_node in zip(
                            np.repeat("B", int(np.count_nonzero(break_mask))),
                            np.nonzero(break_mask)[0].astype(np.int32),
                            break_i_all[break_mask],
                            break_j_all[break_mask],
                        )
                    )
        else:
            self.zero_edges = [
                ('Z', zb_idx, self.alive_node_dict[zb.i_node], self.alive_node_dict[zb.j_node])
                for zb_idx, zb in enumerate(self.model.zero_branches)
                if zb.is_alive and zb.i_node in self.alive_node_dict and zb.j_node in self.alive_node_dict
                and self.alive_node_dict[zb.i_node] != self.alive_node_dict[zb.j_node]
            ]
            for sw_idx, sw in enumerate(self.model.switches):
                if (
                    sw.is_alive
                    and sw.status == 1
                    and sw.run_stat == 1
                    and sw.i_node in self.alive_node_dict
                    and sw.j_node in self.alive_node_dict
                    and self.alive_node_dict[sw.i_node] != self.alive_node_dict[sw.j_node]
                ):
                    self.zero_edges.append(('S', sw_idx, self.alive_node_dict[sw.i_node], self.alive_node_dict[sw.j_node]))
            for brk_idx, brk in enumerate(getattr(self.model, "breakers", [])):
                if (
                    brk.is_alive
                    and brk.status == 1
                    and brk.run_stat == 1
                    and brk.i_node in self.alive_node_dict
                    and brk.j_node in self.alive_node_dict
                    and self.alive_node_dict[brk.i_node] != self.alive_node_dict[brk.j_node]
                ):
                    self.zero_edges.append(('B', brk_idx, self.alive_node_dict[brk.i_node], self.alive_node_dict[brk.j_node]))

        zero_adj = [[] for _ in range(self.N)]
        for edge_idx, (_, _, i_node, j_node) in enumerate(self.zero_edges):
            zero_adj[i_node].append((edge_idx, j_node))
            zero_adj[j_node].append((edge_idx, i_node))

        visited = np.zeros(self.N, dtype=bool)
        edge_used = np.zeros(len(self.zero_edges), dtype=bool)
        comp_nodes = []
        comp_edge_indices = []

        for start in range(self.N):
            if visited[start] or not zero_adj[start]:
                continue
            q = deque([start])
            visited[start] = True
            nodes = []
            edges_idx = []
            while q:
                u = q.popleft()
                nodes.append(u)
                for edge_idx, v in zero_adj[u]:
                    if not edge_used[edge_idx]:
                        edge_used[edge_idx] = True
                        edges_idx.append(edge_idx)
                    if not visited[v]:
                        visited[v] = True
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
        if self.array_mode:
            dcdc = self.ppc["dcdc"]
            if dcdc.size:
                dcdc_i_all = self._map_nodes_with_lookup(dcdc[:, DC_DCDC_COLS["i_node"]], node_lookup)
                dcdc_j_all = self._map_nodes_with_lookup(dcdc[:, DC_DCDC_COLS["j_node"]], node_lookup)
                dcdc_mask = (
                    (dcdc[:, DC_DCDC_COLS["run_stat"]] == 1)
                    & (dcdc_i_all >= 0)
                    & (dcdc_j_all >= 0)
                )
                dcdc_active = dcdc[dcdc_mask]
                self.dcdc_idx = np.nonzero(dcdc_mask)[0].astype(np.int32)
                self.dcdc_i = dcdc_i_all[dcdc_mask].astype(np.int32, copy=False)
                self.dcdc_j = dcdc_j_all[dcdc_mask].astype(np.int32, copy=False)
                self.dcdc_ctrl_code = dcdc_active[:, DC_DCDC_COLS["control_type"]].astype(np.int8, copy=False)
                self.dcdc_p_set = dcdc_active[:, DC_DCDC_COLS["p_set"]].astype(np.float64, copy=False)
                self.dcdc_i_set = dcdc_active[:, DC_DCDC_COLS["i_set"]].astype(np.float64, copy=False)
                self.dcdc_v_set = dcdc_active[:, DC_DCDC_COLS["v_set"]].astype(np.float64, copy=False)
                self.dcdc_r1 = dcdc_active[:, DC_DCDC_COLS["r1"]].astype(np.float64, copy=False)
                self.dcdc_r2 = dcdc_active[:, DC_DCDC_COLS["r2"]].astype(np.float64, copy=False)
            else:
                self.dcdc_idx = self.dcdc_i = self.dcdc_j = np.array([], dtype=np.int32)
                self.dcdc_ctrl_code = np.array([], dtype=np.int8)
                self.dcdc_p_set = self.dcdc_i_set = self.dcdc_v_set = np.array([], dtype=np.float64)
                self.dcdc_r1 = self.dcdc_r2 = np.array([], dtype=np.float64)
            self.N_dcdc = self.dcdc_idx.size
            self.dcdc_ctrl = self.dcdc_ctrl_code
            self.alive_dcdc_tuples = []
            bad_ctrl = ~np.isin(self.dcdc_ctrl_code, np.asarray([DC_CTRL_P, DC_CTRL_V, DC_CTRL_I], dtype=np.int8))
            if np.any(bad_ctrl):
                raise ValueError(f"未知DC-DC控制模式: {self.dcdc_ctrl_code[int(np.where(bad_ctrl)[0][0])]}")
        else:
            self.alive_dcdc_tuples = [
                (idx, self.alive_node_dict[dc.i_node], self.alive_node_dict[dc.j_node], dc.control_type,
                 dc.p_set, dc.i_set, dc.v_set, dc.r1, dc.r2)
                for idx, dc in enumerate(self.model.dcdc_converters)
                if dc.is_alive and dc.i_node in self.alive_node_dict and dc.j_node in self.alive_node_dict
            ]
            self.N_dcdc = len(self.alive_dcdc_tuples)
            if self.N_dcdc:
                # DCDC 采用 r1 + 理想变压 + r2 模型，因此两端功率都作为未知量。
                dcdc_arr = np.asarray(self.alive_dcdc_tuples, dtype=object)
                self.dcdc_idx = dcdc_arr[:, 0].astype(np.int32)
                self.dcdc_i = dcdc_arr[:, 1].astype(np.int32)
                self.dcdc_j = dcdc_arr[:, 2].astype(np.int32)
                self.dcdc_ctrl = dcdc_arr[:, 3]
                self.dcdc_p_set = dcdc_arr[:, 4].astype(np.float64)
                self.dcdc_i_set = dcdc_arr[:, 5].astype(np.float64)
                self.dcdc_v_set = dcdc_arr[:, 6].astype(np.float64)
                self.dcdc_r1 = dcdc_arr[:, 7].astype(np.float64)
                self.dcdc_r2 = dcdc_arr[:, 8].astype(np.float64)
                ctrl_map = {"P": 0, "V": 1, "I": 2}
                try:
                    self.dcdc_ctrl_code = np.asarray([ctrl_map[str(ctrl)] for ctrl in self.dcdc_ctrl], dtype=np.int8)
                except KeyError as exc:
                    raise ValueError(f"未知DC-DC控制模式: {exc.args[0]}") from exc
            else:
                self.dcdc_idx = self.dcdc_i = self.dcdc_j = np.array([], dtype=np.int32)
                self.dcdc_ctrl = np.array([], dtype=object)
                self.dcdc_ctrl_code = np.array([], dtype=np.int8)
                self.dcdc_p_set = self.dcdc_i_set = self.dcdc_v_set = np.array([], dtype=np.float64)
                self.dcdc_r1 = self.dcdc_r2 = np.array([], dtype=np.float64)

        # ---------- 4. 确定松弛节点 ----------
        if self.array_mode and self._direct_ppc_mode:
            gen = self.ppc["gen"]
            self.slack_nodes = {
                node: float(gen[int(gens[0]), DC_GEN_COLS["v_set"]])
                for node, gens in self.slack_gen_info.items()
            }
        else:
            self.slack_nodes = {node: gens[0].v_set for node, gens in self.slack_gen_info.items()}
        self.slack_node_arr = np.fromiter(self.slack_nodes.keys(), dtype=np.int32, count=len(self.slack_nodes))
        self.slack_value_arr = np.fromiter(self.slack_nodes.values(), dtype=np.float64, count=len(self.slack_nodes))

        if self.verbose:
            print("self.N = ", self.N)
            print("self.N_phi = ", self.N_phi)
            print("self.N_dcdc = ", self.N_dcdc)

        # ---------- 5. 变量定义 ----------
        # 前 N 个变量是节点电压；随后是零阻抗连通分量的 phi；最后每台
        # DCDC 有两端端口功率未知量，用于同时表达控制模式和损耗方程。
        self.total_vars = self.N + self.N_phi + self.N_dcdc * 2
        x = np.zeros(self.total_vars, dtype=np.float64)
        x[:self.N] = 1.0
        if self.slack_node_arr.size:
            x[self.slack_node_arr] = self.slack_value_arr

        if self.verbose:
            print(x)

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

        self.total_eq = self.n_unknown + self.n_known + self.n_zero_constraint + self.n_phi_fix + self.n_dcdc * 2
        if self.total_vars != self.total_eq:
            if self.verbose:
                print(f"警告：变量数({self.total_vars})与方程数({self.total_eq})不匹配，请检查零阻抗支路设置。")

        if self.verbose:
            print("total_vars", self.total_vars)
            print("total_eq", self.total_eq)

        self.eq_unknown_start = 0
        self.eq_known_start = self.eq_unknown_start + self.n_unknown
        self.eq_zero_start = self.eq_known_start + self.n_known
        self.eq_phi_start = self.eq_zero_start + self.n_zero_constraint
        self.eq_dcdc_start = self.eq_phi_start + self.n_phi_fix

        self.unknown_map = {int(node): int(i) for i, node in enumerate(self.unknown_nodes)}
        self.zero_con_rows = self.eq_zero_start + np.arange(self.n_zero_constraint, dtype=np.int32)
        self.phi_fix_rows = self.eq_phi_start + np.arange(self.n_phi_fix, dtype=np.int32)
        self.dcdc_seq = np.arange(self.N_dcdc, dtype=np.int32)
        self.dcdc_p_col = self.N + self.N_phi + 2 * self.dcdc_seq
        self.dcdc_q_col = self.dcdc_p_col + 1
        self.dcdc_eq_ctrl = self.eq_dcdc_start + 2 * self.dcdc_seq
        self.dcdc_eq_loss = self.dcdc_eq_ctrl + 1
        self.dcdc_ctrl_p_mask = self.dcdc_ctrl_code == 0
        self.dcdc_ctrl_v_mask = self.dcdc_ctrl_code == 1
        self.dcdc_ctrl_i_mask = self.dcdc_ctrl_code == 2
        self.dcdc_ones = np.ones(self.N_dcdc, dtype=np.float64)
        self._prepare_static_jacobian_indices()
        if self.array_mode and self._direct_ppc_mode:
            self._store_direct_ppc_static(x)

        return G, x

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
            self.dcdc_j_eq_rows_jac = self.dcdc_j_eq[self.dcdc_j_unknown_mask]
            self.dcdc_i_eq_cols_jac = self.dcdc_p_col[self.dcdc_i_unknown_idx]
            self.dcdc_j_eq_cols_jac = self.dcdc_q_col[self.dcdc_j_unknown_idx]
            self.dcdc_i_eq_data_jac = np.ones(self.dcdc_i_unknown_idx.size, dtype=np.float64)
            self.dcdc_j_eq_data_jac = np.ones(self.dcdc_j_unknown_idx.size, dtype=np.float64)

            self.dcdc_ctrl_p_count = int(np.count_nonzero(self.dcdc_ctrl_p_mask))
            self.dcdc_ctrl_v_count = int(np.count_nonzero(self.dcdc_ctrl_v_mask))
            self.dcdc_ctrl_i_count = int(np.count_nonzero(self.dcdc_ctrl_i_mask))
            self.dcdc_ctrl_p_data_jac = np.ones(self.dcdc_ctrl_p_count, dtype=np.float64)
            self.dcdc_ctrl_v_data_jac = np.ones(self.dcdc_ctrl_v_count, dtype=np.float64)
            self.dcdc_ctrl_i_rows_jac = np.repeat(self.dcdc_eq_ctrl[self.dcdc_ctrl_i_mask], 2)
            if self.dcdc_ctrl_i_count:
                self.dcdc_ctrl_i_cols_jac = np.empty(2 * self.dcdc_ctrl_i_count, dtype=np.int32)
                self.dcdc_ctrl_i_data_jac = np.empty(2 * self.dcdc_ctrl_i_count, dtype=np.float64)
                self.dcdc_ctrl_i_cols_jac[0::2] = self.dcdc_p_col[self.dcdc_ctrl_i_mask]
                self.dcdc_ctrl_i_cols_jac[1::2] = self.dcdc_i[self.dcdc_ctrl_i_mask]
                self.dcdc_ctrl_i_data_jac[0::2] = 1.0
                self.dcdc_ctrl_i_data_jac[1::2] = -self.dcdc_i_set[self.dcdc_ctrl_i_mask]
            else:
                self.dcdc_ctrl_i_cols_jac = np.array([], dtype=np.int32)
                self.dcdc_ctrl_i_data_jac = np.array([], dtype=np.float64)
            self.dcdc_loss_rows_jac = np.repeat(self.dcdc_eq_loss, 4)
            self.dcdc_loss_cols_jac = np.empty(self.N_dcdc * 4, dtype=np.int32)
            self.dcdc_loss_cols_jac[0::4] = self.dcdc_p_col
            self.dcdc_loss_cols_jac[1::4] = self.dcdc_q_col
            self.dcdc_loss_cols_jac[2::4] = self.dcdc_i
            self.dcdc_loss_cols_jac[3::4] = self.dcdc_j
        else:
            self.dcdc_i_unknown_idx = self.dcdc_j_unknown_idx = np.array([], dtype=np.int32)
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
            for eq_row, node in enumerate(self.unknown_nodes):
                node = int(node)
                start = int(G_csr.indptr[node])
                end = int(G_csr.indptr[node + 1])
                row_cols = G_csr.indices[start:end]
                row_data = G_csr.data[start:end]
                has_diag = False
                for col, value in zip(row_cols, row_data):
                    col = int(col)
                    unknown_rows.append(eq_row)
                    unknown_cols.append(col)
                    unknown_row_nodes.append(node)
                    unknown_col_nodes.append(col)
                    unknown_g_data.append(float(value))
                    if col == node:
                        has_diag = True
                if not has_diag:
                    unknown_rows.append(eq_row)
                    unknown_cols.append(node)
                    unknown_row_nodes.append(node)
                    unknown_col_nodes.append(node)
                    unknown_g_data.append(0.0)
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
            "dcdc_ctrl_p",
            self.dcdc_eq_ctrl[self.dcdc_ctrl_p_mask] if self.N_dcdc else [],
            self.dcdc_p_col[self.dcdc_ctrl_p_mask] if self.N_dcdc else [],
        )
        add_part(
            "dcdc_ctrl_v",
            self.dcdc_eq_ctrl[self.dcdc_ctrl_v_mask] if self.N_dcdc else [],
            self.dcdc_i[self.dcdc_ctrl_v_mask] if self.N_dcdc else [],
        )
        add_part("dcdc_ctrl_i", self.dcdc_ctrl_i_rows_jac if self.N_dcdc else [], self.dcdc_ctrl_i_cols_jac if self.N_dcdc else [])
        add_part("dcdc_loss", self.dcdc_loss_rows_jac if self.N_dcdc else [], self.dcdc_loss_cols_jac if self.N_dcdc else [])

        self._dc_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            self._dc_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
            self._dc_jac_csr_indices = np.array([], dtype=np.int32)
            self._dc_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
            self._dc_jac_csr_data = np.array([], dtype=np.float64)
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        pattern = coo_matrix(
            (np.ones(raw_count, dtype=np.float64), (raw_rows, raw_cols)),
            shape=(self.total_eq, self.total_vars),
        ).tocsr()
        pattern.sum_duplicates()
        self._dc_jac_csr_indices = pattern.indices.astype(np.int32, copy=True)
        self._dc_jac_csr_indptr = pattern.indptr.astype(np.int32, copy=True)
        self._dc_jac_csr_data = np.empty(self._dc_jac_csr_indices.size, dtype=np.float64)

        positions = {}
        for row in range(self.total_eq):
            start = int(self._dc_jac_csr_indptr[row])
            end = int(self._dc_jac_csr_indptr[row + 1])
            for pos in range(start, end):
                positions[(row, int(self._dc_jac_csr_indices[pos]))] = pos
        self._dc_jac_raw_to_csr_pos = np.fromiter(
            (positions[(int(row), int(col))] for row, col in zip(raw_rows, raw_cols)),
            dtype=np.intp,
            count=raw_count,
        )

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
            part_slice = self._dc_jac_dcdc_ctrl_p_slice
            if self._slice_len(part_slice):
                raw[part_slice] = self.dcdc_ctrl_p_data_jac
            part_slice = self._dc_jac_dcdc_ctrl_v_slice
            if self._slice_len(part_slice):
                raw[part_slice] = self.dcdc_ctrl_v_data_jac
            part_slice = self._dc_jac_dcdc_ctrl_i_slice
            if self._slice_len(part_slice):
                raw[part_slice] = self.dcdc_ctrl_i_data_jac

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

    def _get_jacobi_from_precomputed_pattern(self, terms):
        if not hasattr(self, "_dc_jac_raw_data"):
            return None
        if self._dc_jac_raw_data.size == 0:
            return csr_matrix((self.total_eq, self.total_vars), dtype=np.float64)
        self._fill_jacobian_raw_data(terms)
        # bincount 比 np.add.at 快很多，且每次迭代都要执行。
        self._dc_jac_csr_data[:] = np.bincount(
            self._dc_jac_raw_to_csr_pos,
            weights=self._dc_jac_raw_data,
            minlength=self._dc_jac_csr_data.size,
        )
        return csr_matrix(
            (self._dc_jac_csr_data, self._dc_jac_csr_indices, self._dc_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )

    def _eval_newton_terms(self, G, x):
        """Evaluate DC Newton quantities shared by residual and Jacobian."""
        V = x[:self.N]
        phi = x[self.N:self.N + self.N_phi]
        Pdc = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])
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
            pi = Pdc[0::2]
            pj = Pdc[1::2]
            terms.update(
                dcdc_vi=vi,
                dcdc_vj=vj,
                dcdc_pi=pi,
                dcdc_pj=pj,
                dcdc_vi2=vi * vi,
                dcdc_vj2=vj * vj,
                dcdc_pi2=pi * pi,
                dcdc_pj2=pj * pj,
            )
        return terms

    def _get_jacobi_from_terms(self, G, x, terms):
        """组装 DC Newton 方程的稀疏 Jacobian。"""
        precomputed = self._get_jacobi_from_precomputed_pattern(terms)
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
            if self.dcdc_ctrl_p_count:
                rows_parts.append(self.dcdc_eq_ctrl[self.dcdc_ctrl_p_mask])
                cols_parts.append(self.dcdc_p_col[self.dcdc_ctrl_p_mask])
                data_parts.append(self.dcdc_ctrl_p_data_jac)
            if self.dcdc_ctrl_v_count:
                rows_parts.append(self.dcdc_eq_ctrl[self.dcdc_ctrl_v_mask])
                cols_parts.append(self.dcdc_i[self.dcdc_ctrl_v_mask])
                data_parts.append(self.dcdc_ctrl_v_data_jac)
            if self.dcdc_ctrl_i_count:
                rows_parts.append(self.dcdc_ctrl_i_rows_jac)
                cols_parts.append(self.dcdc_ctrl_i_cols_jac)
                data_parts.append(self.dcdc_ctrl_i_data_jac)

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

        return coo_matrix((J_data, (J_rows, J_cols)), shape=(self.total_eq, self.total_vars)).tocsr()

    def get_jacobi(self, G, x, terms=None):
        """Public Jacobian API for tests and external callers."""
        terms = self._eval_newton_terms(G, x) if terms is None else terms
        return self._get_jacobi_from_terms(G, x, terms)

    def _get_f_from_terms(self, x, terms):
        """计算 DC 残差：节点功率平衡、参考电压、零阻抗约束和 DCDC 约束。"""
        V = terms["V"]
        phi = terms["phi"]
        Pdc = terms["Pdc"]

        P_inj = V * terms["GV"] + self.I_shunt * V - self.P_const

        if self.zero_i.size:
            current = terms["zero_current"]
            P_inj += np.bincount(self.zero_i, weights=V[self.zero_i] * current, minlength=P_inj.size)
            P_inj += np.bincount(self.zero_j, weights=-V[self.zero_j] * current, minlength=P_inj.size)

        if self.N_dcdc:
            P_inj += np.bincount(self.dcdc_i, weights=Pdc[0::2], minlength=P_inj.size)
            P_inj += np.bincount(self.dcdc_j, weights=Pdc[1::2], minlength=P_inj.size)

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
            p_from = Pdc[0::2]
            p_to = Pdc[1::2]
            vi = terms["dcdc_vi"]
            ctrl_values = np.empty(self.N_dcdc, dtype=np.float64)
            ctrl_values[self.dcdc_ctrl_p_mask] = p_from[self.dcdc_ctrl_p_mask] - self.dcdc_p_set[self.dcdc_ctrl_p_mask]
            ctrl_values[self.dcdc_ctrl_v_mask] = vi[self.dcdc_ctrl_v_mask] - self.dcdc_v_set[self.dcdc_ctrl_v_mask]
            ctrl_values[self.dcdc_ctrl_i_mask] = (
                p_from[self.dcdc_ctrl_i_mask]
                - self.dcdc_i_set[self.dcdc_ctrl_i_mask] * vi[self.dcdc_ctrl_i_mask]
            )
            F[self.dcdc_eq_ctrl] = ctrl_values

            vi2 = terms["dcdc_vi2"]
            vj2 = terms["dcdc_vj2"]
            # 第二条方程保证两端端口功率与 r1/r2 损耗模型一致。
            F[self.dcdc_eq_loss] = (
                vi2 * vj2 * (p_from + p_to)
                - self.dcdc_r1 * p_from * p_from * vj2
                - self.dcdc_r2 * p_to * p_to * vi2
            )

        return F

    def get_f(self, x, terms=None):
        """Public residual API for tests and external callers."""
        terms = self._eval_newton_terms(self.G, x) if terms is None else terms
        return self._get_f_from_terms(x, terms)

    def _write_back_ppc(self, x):
        """Write array-mode DC LF results to self.result without object topology."""
        ppc = self.ppc
        V_final = x[:self.N]
        phi_final = x[self.N:self.N + self.N_phi] if self.N_phi > 0 else np.array([])
        Pdc_final = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])
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
                    where=np.abs(v) > self.runtime_params.min_voltage,
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
                    where=np.abs(v) > self.runtime_params.min_voltage,
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
            dcdc_i_p = Pdc_final[0::2]
            dcdc_j_p = Pdc_final[1::2]
            dcdc_vi = V_final[self.dcdc_i]
            dcdc_vj = V_final[self.dcdc_j]
            dcdc_i_c = np.divide(
                dcdc_i_p,
                dcdc_vi,
                out=np.zeros_like(dcdc_i_p),
                where=np.abs(dcdc_vi) > self.runtime_params.min_voltage,
            )
            dcdc_j_c = np.divide(
                dcdc_j_p,
                dcdc_vj,
                out=np.zeros_like(dcdc_j_p),
                where=np.abs(dcdc_vj) > self.runtime_params.min_voltage,
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
            current = share / v if abs(v) > self.runtime_params.min_voltage else 0.0
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
        self._write_ppc_result_to_network()

    def _write_ppc_result_to_network(self) -> None:
        """Copy ppc results back to an optional DCPowerNetwork object facade."""
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
            for dev in devices:
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

    def _result_node_voltage(self, node_idx) -> float:
        rows = self.result.get("bus") if isinstance(getattr(self, "result", None), dict) else None
        if rows is None or len(rows) == 0:
            return 0.0
        node_idx = int(node_idx)
        for row in rows:
            if int(row[DC_BUS_COLS["idx"]]) == node_idx:
                return float(row[DC_BUS_COLS["voltage"]])
        return 0.0

    def _build_lf_result_from_ppc(self) -> DCLFResult:
        # 与 ac_lf 同构：按列向量化抽取 + 单次 searchsorted 做节点电压查表，
        # 避免每行 Python float/int 转换与 dict 查询。
        result = DCLFResult()
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

    def _write_summary_result(self, x):
        voltage = x[:self.N].copy()
        self.result = {
            "node_id": self._summary_node_ids(),
            "voltage": voltage,
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
                "node_count": int(self.N),
                "total_vars": int(self.total_vars),
                "total_eq": int(self.total_eq),
                "v_min": float(np.min(voltage)) if voltage.size else 0.0,
                "v_max": float(np.max(voltage)) if voltage.size else 0.0,
                "v_mean": float(np.mean(voltage)) if voltage.size else 0.0,
            },
        }

    def update_lf_info(self, x):
        """将求解后的电压、电流和功率写回 DC 模型对象。"""
        if self.result_mode == "none":
            self.result = {}
            return
        if self.result_mode == "summary":
            self._write_summary_result(x)
            return

        if self.array_mode and self._direct_ppc_mode:
            self._write_back_ppc(x)
            if self.result_mode != "array" and not getattr(self, "skip_lf_result", False):
                self.lf_result = self._build_lf_result()
            return

        # ---------- 9. 结果回填 ----------
        V_final = x[:self.N]
        phi_final = x[self.N:self.N + self.N_phi] if self.N_phi > 0 else np.array([])
        Pdc_final = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])

        # 节点电压
        for node in self.model.nodes:
            idx = self.alive_node_dict.get(node.idx, -1)
            node.voltage = 0.0 if idx < 0 else V_final[idx]
        for idx, bus in enumerate(self.alive_nodes):
            bus.voltage = float(V_final[idx])


        P_inj = np.zeros(self.N)  # 恒功率注入

        # 电阻支路数量大时，先数组化计算，再做对象字段回填，减少逐支路公式开销。
        for br in self._lf_inactive_branch_devices:
            br.current = br.i_p = br.j_p = 0.0
        if self.branch_idx.size:
            vi = V_final[self.branch_i]
            vj = V_final[self.branch_j]
            current = (vi - vj) / self.branch_r
            i_p = vi * current
            j_p = -vj * current
            for br, cur, p_from, p_to in zip(self._lf_branch_devices, current, i_p, j_p):
                br.current = float(cur)
                br.i_p = float(p_from)
                br.j_p = float(p_to)
            P_inj += np.bincount(self.branch_i, weights=i_p, minlength=P_inj.size)
            P_inj += np.bincount(self.branch_j, weights=j_p, minlength=P_inj.size)

        # 零阻抗支路
        for tp, zb_idx, i_node, j_node, phi_a, phi_b in self.zero_branch_info:
            current = phi_final[phi_a] - phi_final[phi_b]
            if tp == 'Z':
                zb = self.model.zero_branches[zb_idx]
                zb.current = current
                zb.p = V_final[i_node] * current
                P_inj[i_node] += zb.p
                P_inj[j_node] -= V_final[j_node] * current
            if tp == 'S':
                sw = self.model.switches[zb_idx]
                sw.current = current
                sw.p = V_final[i_node] * current
                P_inj[i_node] += sw.p
                P_inj[j_node] -= V_final[j_node] * current
            if tp == 'B':
                brk = self.model.breakers[zb_idx]
                brk.current = current
                brk.p = V_final[i_node] * current
                P_inj[i_node] += brk.p
                P_inj[j_node] -= V_final[j_node] * current

        # 负荷
        for ld in self.model.loads:
            idx = self.alive_node_dict.get(ld.node, -1)
            if idx < 0 or not ld.is_alive:
                ld.p = 0.0
                ld.current = 0.0
                continue
            v = V_final[idx]
            pbase = float(getattr(ld, "pbase", 1.0))
            ld.p = pbase * (ld.pv0 + ld.pv1 * v + ld.pv2 * v * v)
            ld.current = ld.p / v
            P_inj[idx] += ld.p

        for gen in self.model.generators:
            if gen.control_type == 'V':
                continue
            idx = self.alive_node_dict.get(gen.node, -1)
            if idx < 0 or not gen.is_alive:
                gen.p = 0.0
                gen.current = 0.0
                continue
            v = V_final[idx]
            if gen.control_type == 'P':
                gen.p = gen.p_set
            elif gen.control_type == 'I':
                gen.p = gen.i_set * v
            else:
                gen.p = None
            gen.current = gen.p / v if abs(v) > self.runtime_params.min_voltage else 0.0
            P_inj[idx] -= gen.p

        # DC-DC变流器
        if self.N_dcdc:
            dcdc_i_p = Pdc_final[0::2]
            dcdc_j_p = Pdc_final[1::2]
            dcdc_vi = V_final[self.dcdc_i]
            dcdc_vj = V_final[self.dcdc_j]
            dcdc_i_c = np.divide(
                dcdc_i_p,
                dcdc_vi,
                out=np.zeros_like(dcdc_i_p),
                where=np.abs(dcdc_vi) > self.runtime_params.min_voltage,
            )
            dcdc_j_c = np.divide(
                dcdc_j_p,
                dcdc_vj,
                out=np.zeros_like(dcdc_j_p),
                where=np.abs(dcdc_vj) > self.runtime_params.min_voltage,
            )
            P_inj += np.bincount(self.dcdc_i, weights=dcdc_i_p, minlength=P_inj.size)
            P_inj += np.bincount(self.dcdc_j, weights=dcdc_j_p, minlength=P_inj.size)
            for dc_idx, i_p, j_p, i_c, j_c in zip(self.dcdc_idx, dcdc_i_p, dcdc_j_p, dcdc_i_c, dcdc_j_c):
                dc = self.model.dcdc_converters[int(dc_idx)]
                dc.i_p = float(i_p)
                dc.j_p = float(j_p)
                dc.i_c = float(i_c)
                dc.j_c = float(j_c)

        # P_inj 是剩余的不平衡功率，由平衡机来承担。。。

        for node, gens in self.slack_gen_info.items():
            share = P_inj[node] / len(gens)
            for gen in gens:
                gen.p = share
                gen.current = gen.p / V_final[node] if abs(V_final[node]) > self.runtime_params.min_voltage else 0.0
        if not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result()

    def _node_voltage(self, node_idx) -> float:
        pos = self.alive_node_dict.get(int(node_idx), -1)
        if pos < 0 or not hasattr(self, "x"):
            return 0.0
        return float(self.x[int(pos)])

    def _build_lf_result(self) -> DCLFResult:
        if self.array_mode and isinstance(getattr(self, "result", None), dict) and self.result:
            return self._build_lf_result_from_ppc()
        result = DCLFResult()
        voltage_by_node = {
            int(getattr(node, "idx", -1)): float(getattr(node, "voltage", 0.0) or 0.0)
            for node in getattr(self.model, "nodes", [])
        }

        def node_voltage(node_idx) -> float:
            return voltage_by_node.get(int(node_idx), 0.0)

        for node in getattr(self.model, "nodes", []):
            result.nodes[_device_key(node)] = SimpleNamespace(
                volt=float(getattr(node, "voltage", 0.0) or 0.0),
            )
        branch_devices = getattr(self, "_lf_branch_devices", None)
        if branch_devices is not None:
            inactive_branch_devices = getattr(self, "_lf_inactive_branch_devices", [])
            branch_devices = list(branch_devices) + list(inactive_branch_devices)
        else:
            branch_devices = getattr(self.model, "branches", [])
        for br in branch_devices:
            result.branches[_device_key(br)] = SimpleNamespace(
                i_p=float(getattr(br, "i_p", 0.0) or 0.0),
                i_c=float(getattr(br, "current", 0.0) or 0.0),
                i_v=node_voltage(getattr(br, "i_node", -1)),
                j_p=float(getattr(br, "j_p", 0.0) or 0.0),
                j_c=-float(getattr(br, "current", 0.0) or 0.0),
                j_v=node_voltage(getattr(br, "j_node", -1)),
            )
        for zbr in getattr(self.model, "zero_branches", []):
            result.zero_branches[_device_key(zbr)] = SimpleNamespace(
                i_p=float(getattr(zbr, "p", 0.0) or 0.0),
                i_c=float(getattr(zbr, "current", 0.0) or 0.0),
                i_v=node_voltage(getattr(zbr, "i_node", -1)),
            )
        for brk in getattr(self.model, "breakers", []):
            result.breakers[_device_key(brk)] = SimpleNamespace(
                i_p=float(getattr(brk, "p", 0.0) or 0.0),
                i_c=float(getattr(brk, "current", 0.0) or 0.0),
                i_v=node_voltage(getattr(brk, "i_node", -1)),
            )
        for conv in getattr(self.model, "dcdc_converters", []):
            result.dcdc_converters[_device_key(conv)] = SimpleNamespace(
                i_p=float(getattr(conv, "i_p", 0.0) or 0.0),
                i_c=float(getattr(conv, "i_c", 0.0) or 0.0),
                i_v=node_voltage(getattr(conv, "i_node", -1)),
                j_p=float(getattr(conv, "j_p", 0.0) or 0.0),
                j_c=float(getattr(conv, "j_c", 0.0) or 0.0),
                j_v=node_voltage(getattr(conv, "j_node", -1)),
            )
        for gen in getattr(self.model, "generators", []):
            result.generators[_device_key(gen)] = SimpleNamespace(
                i_p=float(getattr(gen, "p", 0.0) or 0.0),
                i_c=float(getattr(gen, "current", 0.0) or 0.0),
                i_v=node_voltage(getattr(gen, "node", -1)),
            )
        for load in getattr(self.model, "loads", []):
            result.loads[_device_key(load)] = SimpleNamespace(
                i_p=float(getattr(load, "p", 0.0) or 0.0),
                i_c=float(getattr(load, "current", 0.0) or 0.0),
                i_v=node_voltage(getattr(load, "node", -1)),
            )
        return result

    def _build_newton_system(self, G, x):
        """Compute residual and Jacobian together for one DC Newton iteration."""
        terms = self._eval_newton_terms(G, x)
        F = self._get_f_from_terms(x, terms)
        J = self._get_jacobi_from_terms(G, x, terms)
        return F, J


    def run(
        self,
        tol=None,
        max_iter=None,
        min_voltage=None,
        divergence_threshold=None,
        verbose=False,
        result_mode=None,
    ):
        """执行直流 Newton 迭代并在收敛后回填结果。"""

        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
            if self.array_mode and self.result_mode != "full":
                self.keep_node_objects = False
                self.alive_nodes = []
        params = self.params.with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
            divergence_threshold=divergence_threshold,
        )
        self.runtime_params = params
        self.tol = params.tol
        self.max_iter = params.max_iter
        self.min_voltage = params.min_voltage
        self.verbose = verbose
        G, x = self.prepare()
        self.converged = False

        refresh_period = self.jacobian_refresh_period
        factor = None
        steps_since_refresh = refresh_period
        solver_name = self._linear_solver_resolved
        solver_fn = self._linear_solver_fn

        # ---------- 7. 牛顿-拉夫逊迭代 ----------
        for it in range(params.max_iter):
            F, J = self._build_newton_system(G, x)

            # 收敛检查
            self.normF = np.linalg.norm(F, np.inf)

            if self.verbose:
                print("it:", it, f"eps:{self.normF:.3e}")

            self.iterations = it + 1
            if self.normF < params.tol:
                if self.verbose:
                    print(f"\n收敛于第 {it+1} 次迭代，最大残差 = {self.normF:.2e}")
                self.converged = True
                self.x = x
                self.update_lf_info(x)
                return 0
            if self.normF > params.divergence_threshold:
                if self.verbose:
                    print(f"\n警告：残差过大 ({self.normF:.2e})，迭代发散")
                self.converged = False
                break

            # 因子复用：refresh_period=1 时等价于标准 NR。
            try:
                if factor is None or steps_since_refresh >= refresh_period:
                    factor = _factor_jacobian(J, solver_name, solver_fn)
                    steps_since_refresh = 0
                delta = factor.solve(-F)
                steps_since_refresh += 1
            except Exception as e:
                if self.verbose:
                    print(f"\n线性方程组求解失败: {e}")
                # 可选求解器失败时退回到 SuperLU；再不行就走最小二乘。
                _OPTIONAL_SPARSE_SOLVERS.pop(solver_name, None)
                _OPTIONAL_SPARSE_MISSING.add(solver_name)
                solver_name = "scipy"
                solver_fn = spsolve
                self._linear_solver_resolved = "scipy"
                self._linear_solver_fn = spsolve
                factor = None
                steps_since_refresh = refresh_period
                try:
                    delta = spsolve(J, -F)
                except Exception:
                    J_dense = J.toarray()
                    try:
                        delta = np.linalg.lstsq(J_dense, -F, rcond=None)[0]
                    except Exception:
                        if self.verbose:
                            print("最小二乘也失败，迭代终止")
                        break

            # 更新变量
            x += delta

        else:
            if self.verbose:
                print(f"\n警告：达到最大迭代次数 {params.max_iter}，未收敛")
            self.converged = False

        self.x = x
        return -1

def print_dc_result(calc: DCPowerFlowCalc) -> None:
    # 9. 输出详细结果
    print("\n===输出直流电网潮流计算结果===")

    if calc.array_mode and calc.result:
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
        print(f"   收敛状态: {'✓ 已收敛' if calc.converged else '✗ 未收敛'}")
        print(f"   迭代次数: {calc.iterations}")
        print(f"   最终残差: {calc.normF:.2e}")

        total_gen_power = float(np.sum(gen[:, DC_GEN_COLS["p"]])) if gen.size else 0.0
        total_load_power = float(np.sum(load[:, DC_LOAD_COLS["p"]])) if load.size else 0.0
        print("\n9. 功率平衡校验:")
        print(f"   总发电功率: {total_gen_power:.6f} pu")
        print(f"   总负荷功率: {total_load_power:.6f} pu")
        print(f"   网损: {total_gen_power - total_load_power:.6f} pu")
        return

    net = calc.model

    print("\n1. 节点电压 (pu):")
    for node in net.nodes:
        print(f"   节点 {node.idx}: {node.voltage:.6f} {'(松弛节点)' if node.is_slack else ''}")

    print("\n2. 普通电阻支路信息:")
    for br in net.branches:
        print(f"   支路 {br.idx} ({br.i_node}->{br.j_node}, r={br.r}pu):")
        print(f"     电流: {br.current:.6f} pu")
        print(f"     送端功率: {br.i_p:.6f} pu, 受端功率: {br.j_p:.6f} pu")
        print(f"     损耗功率: {br.j_p + br.i_p:.6f} pu")

    print("\n3. 零阻抗支路信息:")
    for zb in net.zero_branches:
        print(f"   零阻抗支路 {zb.idx} ({zb.i_node}->{zb.j_node}):")
        print(f"     电流: {zb.current:.6f} pu, 功率: {zb.p:.6f} pu")

    print("\n4. 开关信息:")
    for sw in net.switches:
        print(f"   开关 {sw.idx} ({sw.i_node}->{sw.j_node}, 状态:{'闭合' if sw.status == 1 else '断开'}):")
        print(f"     电流: {sw.current:.6f} pu, 功率: {sw.p:.6f} pu")

    print("\n5. DC-DC变流器信息:")
    for conv in net.dcdc_converters:
        print(f"   变流器 {conv.idx} ({conv.i_node}->{conv.j_node}, 控制:{conv.control_type}):")
        print(f"     设定值: {conv.p_set}, {conv.i_set},{conv.v_set}, 电阻: r1={conv.r1}, r2={conv.r2}")
        print(f"     送端功率: {conv.i_p:.6f} pu, 送端电流: {conv.i_c:.6f} pu")
        print(f"     受端功率: {conv.j_p:.6f} pu, 受端电流: {conv.j_c:.6f} pu")
        print(f"     损耗功率: {conv.j_p + conv.i_p:.6f} pu")

    print("\n6. 负荷信息:")
    for load in net.loads:
        print(f"   负荷 {load.idx} (节点{load.node}):")
        print(f"     消耗功率: {load.p:.6f} pu, 电流: {load.current:.6f} pu")

    print("\n7. 发电机信息:")
    for gen in net.generators:
        print(f"   发电机 {gen.idx} (节点{gen.node}, 类型{gen.control_type}):")
        print(f"     出力功率: {gen.p:.6f} pu, 电流: {gen.current:.6f} pu")

    print("\n8. 计算收敛信息:")
    print(f"   收敛状态: {'✓ 已收敛' if calc.converged else '✗ 未收敛'}")
    print(f"   迭代次数: {calc.iterations}")
    print(f"   最终残差: {calc.normF:.2e}")

    # 功率平衡校验
    total_gen_power = sum(gen.p for gen in net.generators)
    total_load_power = sum(load.p for load in net.loads)
    total_loss = total_gen_power - total_load_power
    print(f"\n9. 功率平衡校验:")
    print(f"   总发电功率: {total_gen_power:.6f} pu")
    print(f"   总负荷功率: {total_load_power:.6f} pu")
    print(f"   网损: {total_loss:.6f} pu")


def _run_with_optional_output(emit_output: bool, func, *args, **kwargs):
    if emit_output:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DC power flow")
    parser.add_argument("file", nargs="?", default=str(model_file("dc", "dc_net_30.e")), help="DC E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument("--linear-solver", default="scipy")
    parser.add_argument("--result-mode", default="full", choices=("full", "summary", "none"))
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
    )
    if not args.quiet:
        print("=== 开始直流电网潮流计算===")
    rc = _run_with_optional_output(not args.quiet, calc.run, verbose=not args.quiet)
    if not args.quiet and calc.result_mode == "full":
        print_dc_result(calc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
