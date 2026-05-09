import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

from collections import deque
from typing import List, Tuple, Dict, Optional
import warnings
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from ac_array_model import (
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PQ,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    SHUNT_B,
    SHUNT_COLS,
    SHUNT_Q,
    SHUNT_V,
    SHUNT_Z,
    SWITCH_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
    build_ac_ppc_from_network,
)


@dataclass
class ACLFResult:
    branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    transformers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    zero_branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    breakers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    generators: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)

def _device_key(device) -> str:
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def load_ac_ppc_from_e_file(file_name) -> Dict:
    """Read an AC E file via efile_read-backed array model loading."""
    from ac_model import ACPowerNetwork

    source = Path(file_name).resolve()
    network = ACPowerNetwork()
    network.source = str(source)
    network.read_from_file(source)
    network.source = str(source)
    ppc = build_ac_ppc_from_network(network)
    ppc["source"] = str(source)
    return ppc


_SPARSE_SOLVER = None
_SPARSE_SOLVER_NAME = None
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

    candidate_names = ("umfpack", "sksparse.klu.klu_solve", "pyklu", "klu", "klu_alt") if solver_name == "auto" else (solver_name,)
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


class _PPCNode:
    __slots__ = ("idx", "name", "vbase", "voltage", "angle")

    def __init__(self, idx, name, vbase, voltage, angle):
        self.idx = idx
        self.name = name
        self.vbase = vbase
        self.voltage = voltage
        self.angle = angle


# ==============================================================================
# 核心工具函数
# ==============================================================================
def find_spanning_tree_edges(edges: List[Tuple[int, int]], n_nodes: int) -> List[int]:
    """Kruskal算法：寻找生成树的边索引（向量化优化）"""
    parent = np.arange(n_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True
        return False

    tree_indices = []
    for idx, (u, v) in enumerate(edges):
        if union(u, v):
            tree_indices.append(idx)
    return tree_indices


def safe_division(a, b, default=0.0):
    """安全除法，避免除零错误"""
    try:
        return a / b if abs(b) > 1e-12 else default
    except:
        return default


def matpower_branch_stamp(r: float, x: float, b: float = 0.0, tap: float = 1.0, shift: float = 0.0):
    """Return MATPOWER-compatible branch admittance entries Yff, Yft, Ytf, Ytt."""
    y = safe_division(1.0, complex(r, x))
    tap_mag = tap if abs(tap) > 1e-12 else 1.0
    tap_complex = tap_mag * np.exp(1j * np.deg2rad(shift))
    y_sh = 1j * b / 2.0
    y_series_shunt = y + y_sh
    return (
        y_series_shunt / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y_series_shunt,
    )


def matpower_branch_stamp_vectorized(r, x, b=0.0, tap=1.0, shift=0.0):
    """Vectorized MATPOWER-compatible branch admittance stamp for branch batches."""
    r = np.asarray(r, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), r.shape)
    tap = np.broadcast_to(np.asarray(tap, dtype=np.float64), r.shape)
    shift = np.broadcast_to(np.asarray(shift, dtype=np.float64), r.shape)

    z = r + 1j * x
    y = np.divide(
        1.0,
        z,
        out=np.zeros(r.shape, dtype=np.complex128),
        where=np.abs(z) > 1e-12,
    )
    tap_mag = np.where(np.abs(tap) > 1e-12, tap, 1.0)
    tap_complex = tap_mag * np.exp(1j * np.deg2rad(shift))
    y_sh = 1j * b / 2.0
    y_series_shunt = y + y_sh
    return (
        y_series_shunt / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y_series_shunt,
    )


def build_jacobian_matrix(rows, cols, data, shape):
    """Build a sparse Jacobian from COO triplets."""
    J = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=shape).tocsr()
    J.sum_duplicates()
    return J


# ==============================================================================
# 核心潮流计算类（精简极速版）
# ==============================================================================
class ACPowerFlowCalc:
    """交流潮流计算类（极坐标牛顿-拉夫逊法）。

    求解变量按 ``theta_unknown``、``V_unknown``、零阻抗支路辅助电位
    ``phi_re/phi_im`` 排列。节点 P/Q 平衡、PV/Slack 电压约束、零阻抗
    支路电压相等约束和 phi 参考约束共同组成 Newton 方程组。
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
        algorithm: str = "nr",
        keep_node_objects: bool = True,
        linear_solver: str = "scipy",
        result_mode: str = "full",
        verbose: bool = False,
    ):
        # 基础配置
        algorithm = str(algorithm).strip().lower()
        if algorithm not in {"nr", "pq"}:
            raise ValueError(f"Unsupported AC power-flow algorithm: {algorithm!r}")
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self._network_writeback = None
        if isinstance(network, dict) and network.get("format") == "ac_ppc_v1":
            self.ppc = network
        elif island is None and hasattr(network, "nodes"):
            self._network_writeback = network
            self.ppc = build_ac_ppc_from_network(network)
            if hasattr(network, "source") and "source" not in self.ppc:
                self.ppc["source"] = str(getattr(network, "source"))
        else:
            self.ppc = None
        self.array_mode = self.ppc is not None
        self.keep_node_objects = bool(keep_node_objects)
        self.net = self._network_writeback if self._network_writeback is not None else (None if self.array_mode else network)
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.min_voltage = self.params.min_voltage
        self.algorithm = algorithm
        self.used_algorithm = algorithm
        self.linear_solver = str(linear_solver or "scipy").strip().lower()
        self.result_mode = self._normalize_result_mode(result_mode)
        self.verbose = bool(verbose)
        self.target_island = island
        self.skipped_islands: List = []
        self.calc_islands: List = []

        # 状态变量
        self.converged: bool = False
        self.iterations: int = 0
        self.normF: float = np.inf

        # 核心变量初始化（精简合并）
        self.N: int = 0
        self.node_list: List = []
        self.node_pos: Dict[int, int] = {}
        self.ppc_node_idx = np.array([], dtype=np.int64)
        self.ppc_node_name = np.array([], dtype=object)
        self.ppc_node_vbase = np.array([], dtype=np.float64)
        self.ppc_node_voltage = np.array([], dtype=np.float64)
        self.ppc_node_angle = np.array([], dtype=np.float64)
        self.isl = None
        self.Y: Optional[csr_matrix] = None

        # 节点参数（使用更简洁的初始化方式）
        self.node_type: np.ndarray = np.array([])
        self.V_spec: np.ndarray = np.array([])
        self.theta_spec: np.ndarray = np.array([])
        self.P_spec: np.ndarray = np.array([])
        self.Q_spec: np.ndarray = np.array([])
        self._slack_mask = np.array([], dtype=bool)
        self._fixed_voltage_mask = np.array([], dtype=bool)
        self.load_info: List = []
        self.slack_node: int = -1

        # 零阻抗支路相关。
        # 零阻抗支路不进入常规 Y 矩阵；这里把每个零阻抗连通分量映射为
        # 一组 phi 辅助变量，支路电流由两端 phi 差表示。
        self.zero_edges = []
        self.comp_nodes: List[List[int]] = []
        self.comp_tree_edges: List[List[int]] = []
        self.phi_node: List[int] = []
        self.phi_comp: List[int] = []
        self.ref_phi_idx: List[int] = []
        self.zero_branch_info: np.ndarray = np.array([])
        self.N_phi: int = 0

        # 变量/方程索引。
        # theta_idx/V_idx 指向 Newton 状态向量中的列号；后续构造残差和
        # 雅可比时只使用这些预计算索引，避免在迭代中重复查表。
        self.theta_unknown: np.ndarray = np.array([])
        self.V_unknown: np.ndarray = np.array([])
        self.theta_idx: Dict[int, int] = {}
        self.V_idx: Dict[int, int] = {}
        self.base_phi_re: int = 0
        self.base_phi_im: int = 0
        self.n_theta: int = 0
        self.n_V: int = 0
        self.total_vars: int = 0
        self.total_eq: int = 0

        # 缓存变量（精简键名）
        self._cache = dict(theta=None, V=None, Vc=None, cos_theta=None, sin_theta=None, P_load=None, Q_load=None)

        # 迭代变量
        self.x: np.ndarray = np.array([])
        self.load_pos = np.array([], dtype=np.int32)
        self.load_pv0 = np.array([], dtype=np.float64)
        self.load_pv1 = np.array([], dtype=np.float64)
        self.load_pv2 = np.array([], dtype=np.float64)
        self.load_qv0 = np.array([], dtype=np.float64)
        self.load_qv1 = np.array([], dtype=np.float64)
        self.load_qv2 = np.array([], dtype=np.float64)
        self.zero_idx = np.array([], dtype=np.int32)
        self.zero_type = np.array([], dtype=np.int32)
        self.zero_a = np.array([], dtype=np.int32)
        self.zero_b = np.array([], dtype=np.int32)
        self.zero_phi_a = np.array([], dtype=np.int32)
        self.zero_phi_b = np.array([], dtype=np.int32)
        self.zero_top_left_rows = np.array([], dtype=np.int32)
        self.zero_top_left_cols = np.array([], dtype=np.int32)
        self.zero_top_left_edge = np.array([], dtype=np.intp)
        self.zero_top_left_kind = np.array([], dtype=np.int8)
        self.zero_top_left_data = np.array([], dtype=np.float64)
        self.zero_top_right_rows = np.array([], dtype=np.int32)
        self.zero_top_right_cols = np.array([], dtype=np.int32)
        self.zero_top_right_edge = np.array([], dtype=np.intp)
        self.zero_top_right_kind = np.array([], dtype=np.int8)
        self.zero_top_right_data = np.array([], dtype=np.float64)
        self.zero_bottom_left_rows = np.array([], dtype=np.int32)
        self.zero_bottom_left_cols = np.array([], dtype=np.int32)
        self.zero_bottom_left_node = np.array([], dtype=np.int32)
        self.zero_bottom_left_kind = np.array([], dtype=np.int8)
        self.zero_bottom_left_data = np.array([], dtype=np.float64)
        self.zero_bottom_right_rows = np.array([], dtype=np.int32)
        self.zero_bottom_right_cols = np.array([], dtype=np.int32)
        self.zero_bottom_right_data = np.array([], dtype=np.float64)
        self.zero_top_left_pos_by_kind = ()
        self.zero_top_left_edge_by_kind = ()
        self.zero_top_right_pos_by_kind = ()
        self.zero_top_right_edge_by_kind = ()
        self.zero_bottom_left_pos_by_kind = ()
        self.zero_bottom_left_node_by_kind = ()
        self._zero_I_re = np.array([], dtype=np.float64)
        self._zero_I_im = np.array([], dtype=np.float64)
        self._zero_Va = np.array([], dtype=np.float64)
        self._zero_Vb = np.array([], dtype=np.float64)
        self._zero_cos_a = np.array([], dtype=np.float64)
        self._zero_sin_a = np.array([], dtype=np.float64)
        self._zero_cos_b = np.array([], dtype=np.float64)
        self._zero_sin_b = np.array([], dtype=np.float64)
        self._zero_a_pv = np.array([], dtype=np.float64)
        self._zero_a_pt = np.array([], dtype=np.float64)
        self._zero_b_pv = np.array([], dtype=np.float64)
        self._zero_b_pt = np.array([], dtype=np.float64)
        self._zero_va_cos = np.array([], dtype=np.float64)
        self._zero_va_sin = np.array([], dtype=np.float64)
        self._zero_vb_cos = np.array([], dtype=np.float64)
        self._zero_vb_sin = np.array([], dtype=np.float64)
        self._zero_tmp = np.array([], dtype=np.float64)
        self.pq_theta_rows = np.array([], dtype=np.int32)
        self.pq_v_cols = np.array([], dtype=np.int32)
        self.Y_diag = np.array([], dtype=np.complex128)
        self.Y_jac_rows = np.array([], dtype=np.int32)
        self.Y_jac_cols = np.array([], dtype=np.int32)
        self.Y_jac_data = np.array([], dtype=np.complex128)
        self.Y_jac_g = np.array([], dtype=np.float64)
        self.Y_jac_b = np.array([], dtype=np.float64)
        self.Y_jac_diag = np.array([], dtype=np.complex128)
        self.Y_offdiag_indices: List[np.ndarray] = []
        self.Y_offdiag_data: List[np.ndarray] = []
        self.theta_col_by_node = np.array([], dtype=np.int32)
        self.v_col_by_node = np.array([], dtype=np.int32)
        self.p_row_by_node = np.array([], dtype=np.int32)
        self.q_row_by_node = np.array([], dtype=np.int32)
        self.standard_jac_csr_indices = np.array([], dtype=np.int32)
        self.standard_jac_csr_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csr_order = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_raw_data = np.array([], dtype=np.float64)
        self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.full_jac_csr_indices = np.array([], dtype=np.int32)
        self.full_jac_csr_indptr = np.array([], dtype=np.int32)
        self.full_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_standard_slice = slice(0, 0)
        self.full_jac_zero_top_left_slice = slice(0, 0)
        self.full_jac_zero_top_right_slice = slice(0, 0)
        self.full_jac_zero_bottom_left_slice = slice(0, 0)
        self.full_jac_zero_bottom_right_slice = slice(0, 0)
        self.std_jac_load_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_extra_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_p_pos = np.array([], dtype=np.intp)
        self.std_jac_load_q_pos = np.array([], dtype=np.intp)
        self._jac_delta = np.array([], dtype=np.float64)
        self._jac_cos_delta = np.array([], dtype=np.float64)
        self._jac_sin_delta = np.array([], dtype=np.float64)
        self._jac_vivj = np.array([], dtype=np.float64)
        self._jac_common_p = np.array([], dtype=np.float64)
        self._jac_common_q = np.array([], dtype=np.float64)
        self._jac_tmp = np.array([], dtype=np.float64)
        self.live_gens: List = []
        self.live_loads: List = []
        self.live_shunts: List = []
        self.live_branches: List = []
        self.live_transformers: List = []
        self.live_zero_branches: List = []
        self.live_switches: List = []
        self.gen_pos = np.array([], dtype=np.int32)
        self.gen_share = np.array([], dtype=np.float64)
        self.load_obj_pos = np.array([], dtype=np.int32)
        self.shunt_pos = np.array([], dtype=np.int32)
        self.branch_i = np.array([], dtype=np.int32)
        self.branch_j = np.array([], dtype=np.int32)
        self.branch_yff = np.array([], dtype=np.complex128)
        self.branch_yft = np.array([], dtype=np.complex128)
        self.branch_ytf = np.array([], dtype=np.complex128)
        self.branch_ytt = np.array([], dtype=np.complex128)
        self.transformer_i = np.array([], dtype=np.int32)
        self.transformer_j = np.array([], dtype=np.int32)
        self.transformer_yff = np.array([], dtype=np.complex128)
        self.transformer_yft = np.array([], dtype=np.complex128)
        self.transformer_ytf = np.array([], dtype=np.complex128)
        self.transformer_ytt = np.array([], dtype=np.complex128)
        self.active_bus_rows = np.array([], dtype=np.int32)
        self.row_to_pos = np.array([], dtype=np.int32)
        self.ppc_gen_rows = np.array([], dtype=np.int32)
        self.ppc_gen_pos = np.array([], dtype=np.int32)
        self.ppc_gen_share = np.array([], dtype=np.float64)
        self.ppc_load_rows = np.array([], dtype=np.int32)
        self.ppc_load_pos = np.array([], dtype=np.int32)
        self.ppc_shunt_rows = np.array([], dtype=np.int32)
        self.ppc_shunt_pos = np.array([], dtype=np.int32)
        self.ppc_branch_rows = np.array([], dtype=np.int32)
        self.ppc_transformer_rows = np.array([], dtype=np.int32)
        self.pq_Bp = None
        self.pq_Bpp = None
        self.pq_Bp_factor = None
        self.pq_Bpp_factor = None
        self._state_theta = np.array([], dtype=np.float64)
        self._state_voltage = np.array([], dtype=np.float64)
        self._state_vc = np.array([], dtype=np.complex128)
        self._empty_phi = np.array([], dtype=np.float64)
        self._state_x_obj = None
        self._power_x_obj = None
        self._last_Ibus = None
        self._last_Sbus = None
        self._load_p_work = np.array([], dtype=np.float64)
        self._load_q_work = np.array([], dtype=np.float64)
        self._load_dp_work = np.array([], dtype=np.float64)
        self._load_dq_work = np.array([], dtype=np.float64)
        self._load_vm_work = np.array([], dtype=np.float64)
        self._load_value_work = np.array([], dtype=np.float64)
        self._load_aux_work = np.array([], dtype=np.float64)
        self._residual_work = np.array([], dtype=np.float64)
        self.result: Dict = {}

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        mode = str(result_mode or "full").strip().lower()
        aliases = {
            "all": "full",
            "full": "full",
            "complete": "full",
            "summary": "summary",
            "brief": "summary",
            "minimal": "summary",
            "none": "none",
            "skip": "none",
            "raw": "none",
        }
        if mode not in aliases:
            raise ValueError(f"Unsupported AC result_mode: {result_mode!r}")
        return aliases[mode]

    def _cache_node_type_masks(self):
        self._slack_mask = self.node_type == 'SLACK'
        self._fixed_voltage_mask = self._slack_mask | (self.node_type == 'PV')

    def _cache_static_arrays(self):
        """Convert object lists and sparse Y rows into arrays reused by each Newton step."""
        self._cache_node_type_masks()
        self.live_gens = [gen for gen in self.isl.gens if gen.is_alive]
        self.live_loads = [ld for ld in self.isl.loads if ld.is_alive]
        self.live_shunts = [sc for sc in self.isl.shunt_compensators if sc.is_alive]
        self.live_zero_branches = [zbr for zbr in self.isl.zero_branches if zbr.is_alive]
        self.live_switches = [sw for sw in self.isl.switches if sw.is_alive]

        if self.load_info:
            load_data = np.asarray(self.load_info, dtype=np.float64)
            self.load_pos = load_data[:, 0].astype(np.int32)
            self.load_pv0 = load_data[:, 1]
            self.load_pv1 = load_data[:, 2]
            self.load_pv2 = load_data[:, 3]
            self.load_qv0 = load_data[:, 4]
            self.load_qv1 = load_data[:, 5]
            self.load_qv2 = load_data[:, 6]
        else:
            self.load_pos = np.array([], dtype=np.int32)
            self.load_pv0 = self.load_pv1 = self.load_pv2 = np.array([], dtype=np.float64)
            self.load_qv0 = self.load_qv1 = self.load_qv2 = np.array([], dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)

        if self.N_phi > 0 and len(self.zero_branch_info) > 0:
            self.zero_idx = self.zero_branch_info[:, 0].astype(np.int32)
            self.zero_type = self.zero_branch_info[:, 1].astype(np.int32)
            self.zero_a = self.zero_branch_info[:, 2].astype(np.int32)
            self.zero_b = self.zero_branch_info[:, 3].astype(np.int32)
            self.zero_phi_a = self.zero_branch_info[:, 4].astype(np.int32)
            self.zero_phi_b = self.zero_branch_info[:, 5].astype(np.int32)
        else:
            self.zero_idx = self.zero_type = self.zero_a = self.zero_b = np.array([], dtype=np.int32)
            self.zero_phi_a = self.zero_phi_b = np.array([], dtype=np.int32)

        self.theta_unknown = np.asarray(self.theta_unknown, dtype=np.int32)
        self.V_unknown = np.asarray(self.V_unknown, dtype=np.int32)
        self.pq_theta_rows = np.fromiter(
            (self.theta_idx[int(pos)] for pos in self.V_unknown),
            dtype=np.int32,
            count=self.V_unknown.size,
        )
        self.pq_v_cols = np.arange(self.V_unknown.size, dtype=np.int32)
        self._cache_zero_jacobian_pattern()

        self.gen_pos = np.asarray([self.node_pos[gen.node] for gen in self.live_gens], dtype=np.int32)
        self.gen_share = np.ones(len(self.live_gens), dtype=np.float64)
        if self.live_gens:
            groups: Dict[int, List[int]] = {}
            for idx, pos in enumerate(self.gen_pos):
                groups.setdefault(int(pos), []).append(idx)
            for indices in groups.values():
                total_alpha = sum(
                    self.live_gens[idx].alpha
                    for idx in indices
                    if self.live_gens[idx].alpha is not None
                )
                if total_alpha > 0.0:
                    for idx in indices:
                        alpha = self.live_gens[idx].alpha
                        self.gen_share[idx] = (alpha if alpha is not None else 0.0) / total_alpha
                else:
                    self.gen_share[indices] = 1.0 / len(indices)

        self.load_obj_pos = np.asarray([self.node_pos[ld.node] for ld in self.live_loads], dtype=np.int32)
        self.shunt_pos = np.asarray([self.node_pos[sc.node] for sc in self.live_shunts], dtype=np.int32)

        # 稀疏矩阵批量 Jacobian 不使用逐行 Y 缓存，跳过可减少大算例 prepare 时间。
        self.Y_diag = np.array([], dtype=np.complex128)
        self.Y_offdiag_indices = []
        self.Y_offdiag_data = []
        return

    # --------------------------------------------------------------------------
    # 私有辅助函数（精简版）
    # --------------------------------------------------------------------------
    def _extract_state_vars(self, x: np.ndarray, update_cache: bool = True) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将扁平 Newton 向量还原为 theta、V 和零阻抗支路的 phi 变量。"""
        theta = self._state_theta
        V = self._state_voltage
        theta.fill(0.0)
        V.fill(1.0)

        # 填充未知变量
        theta[self.theta_unknown] = x[:self.n_theta]
        V[self.V_unknown] = x[self.n_theta:self.n_theta + self.n_V]

        # 填充已知变量（精简判断逻辑）
        theta[self._slack_mask] = self.theta_spec[self._slack_mask]
        V[self._fixed_voltage_mask] = self.V_spec[self._fixed_voltage_mask]

        # 提取phi变量（精简条件判断）
        phi_slice = slice(self.base_phi_re, self.base_phi_re + self.N_phi)
        phi_re = x[phi_slice] if self.N_phi > 0 else self._empty_phi
        phi_im = x[self.base_phi_im:self.base_phi_im + self.N_phi] if self.N_phi > 0 else self._empty_phi

        # 更新缓存
        if update_cache:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            Vc = self._state_vc
            np.multiply(V, cos_theta, out=Vc.real)
            np.multiply(V, sin_theta, out=Vc.imag)
            self._cache.update({
                'theta': theta,
                'V': V,
                'cos_theta': cos_theta,
                'sin_theta': sin_theta,
                'Vc': Vc
            })
            self._state_x_obj = x
            self._power_x_obj = None
            self._last_Ibus = None
            self._last_Sbus = None

        return theta, V, phi_re, phi_im

    def _calc_power_balance(self, theta: np.ndarray, V: np.ndarray, phi_re: np.ndarray, phi_im: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """计算每个节点的 P/Q 不平衡量，包含 Y 矩阵注入、负荷和零阻抗电流。"""
        # 从缓存获取预计算值
        cache = self._cache
        Vc = cache['Vc']

        # 导纳注入功率
        I_y = self.Y.dot(Vc)
        S_y = Vc * np.conj(I_y)
        self._power_x_obj = self._state_x_obj
        self._last_Ibus = I_y
        self._last_Sbus = S_y

        # 负荷功率（首次计算时缓存）
        P_load, Q_load = self._calc_load_power(V)
        cache.update(P_load=P_load, Q_load=Q_load)

        # 零阻抗支路功率贡献（精简向量化逻辑）
        P_zero, Q_zero = self._calc_zero_branch_power(Vc, phi_re, phi_im)

        # 功率不平衡量
        P_calc = S_y.real + P_zero
        Q_calc = S_y.imag + Q_zero

        return P_calc - (self.P_spec - P_load), Q_calc - (self.Q_spec - Q_load)

    def _calc_load_power(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """提取负荷功率计算逻辑（精简独立函数）"""
        P_load = self._load_p_work
        Q_load = self._load_q_work
        P_load.fill(0.0)
        Q_load.fill(0.0)

        if self.load_pos.size:
            vm_vals = self._load_vm_work
            values = self._load_value_work
            aux = self._load_aux_work
            np.take(V, self.load_pos, out=vm_vals)

            # 向量化计算负荷功率
            np.multiply(vm_vals, vm_vals, out=values)
            values *= self.load_pv2
            np.multiply(self.load_pv1, vm_vals, out=aux)
            values += aux
            values += self.load_pv0

            # 累加至对应节点
            np.add.at(P_load, self.load_pos, values)

            np.multiply(vm_vals, vm_vals, out=values)
            values *= self.load_qv2
            np.multiply(self.load_qv1, vm_vals, out=aux)
            values += aux
            values += self.load_qv0
            np.add.at(Q_load, self.load_pos, values)

        return P_load, Q_load

    def _calc_zero_branch_power(self, Vc: np.ndarray, phi_re: np.ndarray, phi_im: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """把零阻抗支路的显式电流变量转换为两端节点功率注入。"""
        P_zero = np.zeros(self.N, dtype=np.float64)
        Q_zero = np.zeros(self.N, dtype=np.float64)

        if self.N_phi > 0 and self.zero_a.size:
            I_ab_re = phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]
            I_ab_im = phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]

            # 向量化计算功率（精简计算公式）
            Vc_a, Vc_b = Vc[self.zero_a], Vc[self.zero_b]
            Sa = Vc_a * (I_ab_re - 1j * I_ab_im)
            Sb = Vc_b * (-I_ab_re + 1j * I_ab_im)

            # 累加功率
            np.add.at(P_zero, self.zero_a, Sa.real)
            np.add.at(Q_zero, self.zero_a, Sa.imag)
            np.add.at(P_zero, self.zero_b, Sb.real)
            np.add.at(Q_zero, self.zero_b, Sb.imag)

        return P_zero, Q_zero

    def get_current(self, v: float, p: float, q: float):
        return abs(np.conj(complex(p, q)) / v) if abs(v) > self.min_voltage else 0.0

    def _ppc_node_rows(self, node_ids) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if self._ppc_sequential_node_ids:
            return node_ids.astype(np.int32)
        return np.fromiter((self._ppc_node_row_by_id[int(node_id)] for node_id in node_ids), dtype=np.int32, count=node_ids.size)

    def _bind_ppc_nodes_to_network(self) -> None:
        """Use original ACNode objects for network-input array solves."""
        network = getattr(self, "_network_writeback", None)
        if network is None or not self.keep_node_objects:
            return
        node_by_idx = {int(node.idx): node for node in getattr(network, "nodes", [])}
        if not node_by_idx:
            return
        bound_nodes = []
        for pos, node_idx in enumerate(self.ppc_node_idx):
            node = node_by_idx.get(int(node_idx))
            if node is None:
                node = self.node_list[pos] if pos < len(self.node_list) else _PPCNode(
                    int(node_idx),
                    str(self.ppc_node_name[pos]) if pos < self.ppc_node_name.size else f"bus_{int(node_idx)}",
                    float(self.ppc_node_vbase[pos]),
                    float(self.ppc_node_voltage[pos]),
                    float(self.ppc_node_angle[pos]),
                )
            bound_nodes.append(node)
        self.node_list = bound_nodes
        self.node_pos = {int(node_idx): pos for pos, node_idx in enumerate(self.ppc_node_idx)}

    def _prepare_from_ppc(self):
        """Prepare a Newton system from an already arrayized AC ppc dictionary."""
        ppc = self.ppc
        static = ppc.get("_pf_static")
        if static is not None:
            self._load_ppc_static(static)
            if self.algorithm == "pq" and self.pq_Bp is None:
                self._cache_pq_decoupled_matrices()
            self._bind_ppc_nodes_to_network()
            if self.verbose:
                print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")
            return

        bus = np.asarray(ppc["bus"], dtype=np.float64)
        branch = np.asarray(ppc.get("branch", np.zeros((0, len(BRANCH_COLS)))), dtype=np.float64)
        transformer = np.asarray(ppc.get("transformer", np.zeros((0, len(TRANSFORMER_COLS)))), dtype=np.float64)
        gen = np.asarray(ppc.get("gen", np.zeros((0, len(GEN_COLS)))), dtype=np.float64)
        load = np.asarray(ppc.get("load", np.zeros((0, len(LOAD_COLS)))), dtype=np.float64)
        shunt = np.asarray(ppc.get("shunt", np.zeros((0, len(SHUNT_COLS)))), dtype=np.float64)
        zero_branch = np.asarray(ppc.get("zero_branch", np.zeros((0, len(ZERO_BRANCH_COLS)))), dtype=np.float64)
        switch = np.asarray(ppc.get("switch", np.zeros((0, len(SWITCH_COLS)))), dtype=np.float64)
        breaker = np.asarray(ppc.get("break", np.zeros((0, len(SWITCH_COLS)))), dtype=np.float64)

        n_bus_all = bus.shape[0]
        bus_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
        self._ppc_sequential_node_ids = bool(np.array_equal(bus_ids, np.arange(n_bus_all)))
        self._ppc_node_row_by_id = {} if self._ppc_sequential_node_ids else {int(node_id): pos for pos, node_id in enumerate(bus_ids)}
        running_bus = bus[:, BUS_COLS["run_stat"]] == 1

        edge_i_parts = []
        edge_j_parts = []

        def collect_device_edges(dev_array, cols, require_closed=False):
            if dev_array.size == 0:
                return
            i_rows = self._ppc_node_rows(dev_array[:, cols["i_node"]])
            j_rows = self._ppc_node_rows(dev_array[:, cols["j_node"]])
            mask = (dev_array[:, cols["run_stat"]] == 1) & running_bus[i_rows] & running_bus[j_rows] & (i_rows != j_rows)
            if require_closed:
                mask &= dev_array[:, cols["status"]] == 1
            if np.any(mask):
                edge_i_parts.append(i_rows[mask])
                edge_j_parts.append(j_rows[mask])

        collect_device_edges(branch, BRANCH_COLS)
        collect_device_edges(transformer, TRANSFORMER_COLS)
        collect_device_edges(zero_branch, ZERO_BRANCH_COLS)
        collect_device_edges(breaker, SWITCH_COLS, require_closed=True)
        collect_device_edges(switch, SWITCH_COLS, require_closed=True)

        if edge_i_parts:
            graph_rows = np.concatenate(edge_i_parts + edge_j_parts).astype(np.int32, copy=False)
            graph_cols = np.concatenate(edge_j_parts + edge_i_parts).astype(np.int32, copy=False)
            graph_data = np.ones(graph_rows.size, dtype=np.int8)
            graph = coo_matrix((graph_data, (graph_rows, graph_cols)), shape=(n_bus_all, n_bus_all)).tocsr()
            _, comp_labels = connected_components(graph, directed=False, return_labels=True)
        else:
            comp_labels = np.arange(n_bus_all, dtype=np.int32)

        slack_components = np.array([], dtype=comp_labels.dtype)
        if gen.size:
            gen_rows = self._ppc_node_rows(gen[:, GEN_COLS["node"]])
            gen_live = (gen[:, GEN_COLS["run_stat"]] == 1) & running_bus[gen_rows]
            slack_live = gen_live & (gen[:, GEN_COLS["control_type"]] == CTRL_SLACK)
            slack_components = np.unique(comp_labels[gen_rows[slack_live]])

        active_bus = running_bus & np.isin(comp_labels, slack_components)
        if not np.any(active_bus):
            raise RuntimeError("电网中无带平衡节点的存活拓扑岛，无法进行潮流计算")

        self.active_bus_rows = np.where(active_bus)[0].astype(np.int32)
        self.row_to_pos = np.full(n_bus_all, -1, dtype=np.int32)
        self.row_to_pos[self.active_bus_rows] = np.arange(self.active_bus_rows.size, dtype=np.int32)
        self.N = int(self.active_bus_rows.size)

        active_rows = self.active_bus_rows
        self.ppc_node_idx = bus[active_rows, BUS_COLS["idx"]].astype(np.int64, copy=True)
        self.ppc_node_vbase = bus[active_rows, BUS_COLS["vbase"]].astype(np.float64, copy=True)
        self.ppc_node_voltage = bus[active_rows, BUS_COLS["voltage"]].astype(np.float64, copy=True)
        self.ppc_node_angle = bus[active_rows, BUS_COLS["angle"]].astype(np.float64, copy=True)
        if self.keep_node_objects:
            bus_names = ppc.get("bus_name", np.asarray([f"bus_{int(idx)}" for idx in bus_ids], dtype=object))
            self.ppc_node_name = np.asarray(bus_names[active_rows], dtype=object)
            self.node_list = [
                _PPCNode(
                    idx=int(idx),
                    name=str(name),
                    vbase=float(vbase),
                    voltage=float(voltage),
                    angle=float(angle),
                )
                for idx, name, vbase, voltage, angle in zip(
                    self.ppc_node_idx,
                    self.ppc_node_name,
                    self.ppc_node_vbase,
                    self.ppc_node_voltage,
                    self.ppc_node_angle,
                )
            ]
            self.node_pos = {node.idx: pos for pos, node in enumerate(self.node_list)}
        else:
            self.ppc_node_name = np.empty(self.N, dtype=object)
            self.node_list = []
            self.node_pos = {}
        self._bind_ppc_nodes_to_network()

        self.node_type = np.full(self.N, 'PQ', dtype='U5')
        self.V_spec = np.full(self.N, np.nan, dtype=np.float64)
        self.theta_spec = np.zeros(self.N, dtype=np.float64)
        self.P_spec = np.zeros(self.N, dtype=np.float64)
        self.Q_spec = np.zeros(self.N, dtype=np.float64)
        self.load_info = []
        self.slack_node = -1

        self._prepare_ppc_devices(branch, transformer, gen, load, shunt, zero_branch, switch, active_bus)
        self._prepare_ppc_y_matrix(branch, transformer, shunt)
        self._prepare_ppc_zero_edges(zero_branch, switch, active_bus, breaker)
        self._finalize_prepared_arrays()
        self._store_ppc_static()
        if self.verbose:
            print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")

    def _store_ppc_static(self):
        """Cache immutable ppc preparation artifacts on the ppc for repeated solves."""
        self.ppc["_pf_static"] = {
            "N": self.N,
            "active_bus_rows": self.active_bus_rows,
            "row_to_pos": self.row_to_pos,
            "node_idx": self.ppc_node_idx,
            "node_name": self.ppc_node_name,
            "node_vbase": self.ppc_node_vbase,
            "node_voltage": self.ppc_node_voltage,
            "node_angle": self.ppc_node_angle,
            "node_pos": self.node_pos,
            "node_type": self.node_type,
            "V_spec": self.V_spec,
            "theta_spec": self.theta_spec,
            "P_spec": self.P_spec,
            "Q_spec": self.Q_spec,
            "_slack_mask": self._slack_mask,
            "_fixed_voltage_mask": self._fixed_voltage_mask,
            "Y": self.Y,
            "zero_edges": self.zero_edges,
            "comp_nodes": self.comp_nodes,
            "comp_tree_edges": self.comp_tree_edges,
            "phi_node": self.phi_node,
            "phi_comp": self.phi_comp,
            "ref_phi_idx": self.ref_phi_idx,
            "zero_branch_info": self.zero_branch_info,
            "N_phi": self.N_phi,
            "theta_unknown": self.theta_unknown,
            "V_unknown": self.V_unknown,
            "theta_idx": self.theta_idx,
            "V_idx": self.V_idx,
            "n_theta": self.n_theta,
            "n_V": self.n_V,
            "base_phi_re": self.base_phi_re,
            "base_phi_im": self.base_phi_im,
            "total_vars": self.total_vars,
            "total_eq": self.total_eq,
            "x0": self.x,
            "load_pos": self.load_pos,
            "load_pv0": self.load_pv0,
            "load_pv1": self.load_pv1,
            "load_pv2": self.load_pv2,
            "load_qv0": self.load_qv0,
            "load_qv1": self.load_qv1,
            "load_qv2": self.load_qv2,
            "zero_idx": self.zero_idx,
            "zero_type": self.zero_type,
            "zero_a": self.zero_a,
            "zero_b": self.zero_b,
            "zero_phi_a": self.zero_phi_a,
            "zero_phi_b": self.zero_phi_b,
            "zero_top_left_rows": self.zero_top_left_rows,
            "zero_top_left_cols": self.zero_top_left_cols,
            "zero_top_left_edge": self.zero_top_left_edge,
            "zero_top_left_kind": self.zero_top_left_kind,
            "zero_top_left_data": self.zero_top_left_data,
            "zero_top_right_rows": self.zero_top_right_rows,
            "zero_top_right_cols": self.zero_top_right_cols,
            "zero_top_right_edge": self.zero_top_right_edge,
            "zero_top_right_kind": self.zero_top_right_kind,
            "zero_top_right_data": self.zero_top_right_data,
            "zero_bottom_left_rows": self.zero_bottom_left_rows,
            "zero_bottom_left_cols": self.zero_bottom_left_cols,
            "zero_bottom_left_node": self.zero_bottom_left_node,
            "zero_bottom_left_kind": self.zero_bottom_left_kind,
            "zero_bottom_left_data": self.zero_bottom_left_data,
            "zero_bottom_right_rows": self.zero_bottom_right_rows,
            "zero_bottom_right_cols": self.zero_bottom_right_cols,
            "zero_bottom_right_data": self.zero_bottom_right_data,
            "pq_theta_rows": self.pq_theta_rows,
            "pq_v_cols": self.pq_v_cols,
            "theta_col_by_node": self.theta_col_by_node,
            "v_col_by_node": self.v_col_by_node,
            "p_row_by_node": self.p_row_by_node,
            "q_row_by_node": self.q_row_by_node,
            "Y_jac_rows": self.Y_jac_rows,
            "Y_jac_cols": self.Y_jac_cols,
            "Y_jac_data": self.Y_jac_data,
            "Y_jac_g": self.Y_jac_g,
            "Y_jac_b": self.Y_jac_b,
            "Y_jac_diag": self.Y_jac_diag,
            "Y_jac_diag_idx": self.Y_jac_diag_idx,
            "Y_jac_diag_nodes": self.Y_jac_diag_nodes,
            "Y_jac_diag_g": self.Y_jac_diag_g,
            "Y_jac_diag_b": self.Y_jac_diag_b,
            "standard_jac_rows": self.standard_jac_rows,
            "standard_jac_cols": self.standard_jac_cols,
            "standard_jac_data": self.standard_jac_data,
            "std_jac_p_theta_idx": self.std_jac_p_theta_idx,
            "std_jac_p_vm_idx": self.std_jac_p_vm_idx,
            "std_jac_q_theta_idx": self.std_jac_q_theta_idx,
            "std_jac_q_vm_idx": self.std_jac_q_vm_idx,
            "std_jac_p_theta_slice": self.std_jac_p_theta_slice,
            "std_jac_p_vm_slice": self.std_jac_p_vm_slice,
            "std_jac_q_theta_slice": self.std_jac_q_theta_slice,
            "std_jac_q_vm_slice": self.std_jac_q_vm_slice,
            "std_jac_load_p_slice": self.std_jac_load_p_slice,
            "std_jac_load_q_slice": self.std_jac_load_q_slice,
            "standard_jac_csr_indices": self.standard_jac_csr_indices,
            "standard_jac_csr_indptr": self.standard_jac_csr_indptr,
            "standard_jac_csr_order": self.standard_jac_csr_order,
            "standard_jac_csr_data": self.standard_jac_csr_data,
            "full_jac_raw_data": self.full_jac_raw_data,
            "full_jac_raw_to_csr_pos": self.full_jac_raw_to_csr_pos,
            "full_jac_csr_indices": self.full_jac_csr_indices,
            "full_jac_csr_indptr": self.full_jac_csr_indptr,
            "full_jac_csr_data": self.full_jac_csr_data,
            "full_jac_standard_slice": self.full_jac_standard_slice,
            "full_jac_zero_top_left_slice": self.full_jac_zero_top_left_slice,
            "full_jac_zero_top_right_slice": self.full_jac_zero_top_right_slice,
            "full_jac_zero_bottom_left_slice": self.full_jac_zero_bottom_left_slice,
            "full_jac_zero_bottom_right_slice": self.full_jac_zero_bottom_right_slice,
            "std_jac_load_nodes": self.std_jac_load_nodes,
            "std_jac_load_extra_nodes": self.std_jac_load_extra_nodes,
            "std_jac_load_p_pos": self.std_jac_load_p_pos,
            "std_jac_load_q_pos": self.std_jac_load_q_pos,
            "_jac_delta": self._jac_delta,
            "_jac_cos_delta": self._jac_cos_delta,
            "_jac_sin_delta": self._jac_sin_delta,
            "_jac_vivj": self._jac_vivj,
            "_jac_common_p": self._jac_common_p,
            "_jac_common_q": self._jac_common_q,
            "_jac_tmp": self._jac_tmp,
            "_load_p_work": self._load_p_work,
            "_load_q_work": self._load_q_work,
            "_load_dp_work": self._load_dp_work,
            "_load_dq_work": self._load_dq_work,
            "_load_vm_work": self._load_vm_work,
            "_load_value_work": self._load_value_work,
            "_load_aux_work": self._load_aux_work,
            "_residual_work": self._residual_work,
            "pq_Bp": self.pq_Bp,
            "pq_Bpp": self.pq_Bpp,
            "pq_Bp_factor": self.pq_Bp_factor,
            "pq_Bpp_factor": self.pq_Bpp_factor,
            "_state_theta": self._state_theta,
            "_state_voltage": self._state_voltage,
            "_state_vc": self._state_vc,
            "_empty_phi": self._empty_phi,
            "branch_i": self.branch_i,
            "branch_j": self.branch_j,
            "branch_yff": self.branch_yff,
            "branch_yft": self.branch_yft,
            "branch_ytf": self.branch_ytf,
            "branch_ytt": self.branch_ytt,
            "transformer_i": self.transformer_i,
            "transformer_j": self.transformer_j,
            "transformer_yff": self.transformer_yff,
            "transformer_yft": self.transformer_yft,
            "transformer_ytf": self.transformer_ytf,
            "transformer_ytt": self.transformer_ytt,
            "ppc_gen_rows": self.ppc_gen_rows,
            "ppc_gen_pos": self.ppc_gen_pos,
            "ppc_gen_share": self.ppc_gen_share,
            "ppc_load_rows": self.ppc_load_rows,
            "ppc_load_pos": self.ppc_load_pos,
            "ppc_shunt_rows": self.ppc_shunt_rows,
            "ppc_shunt_pos": self.ppc_shunt_pos,
            "ppc_branch_rows": self.ppc_branch_rows,
            "ppc_transformer_rows": self.ppc_transformer_rows,
        }

    def _load_ppc_static(self, static: Dict):
        """Load cached ppc preparation artifacts and reset dynamic iteration state."""
        self.N = static["N"]
        self.active_bus_rows = static["active_bus_rows"]
        self.row_to_pos = static["row_to_pos"]
        self.ppc_node_idx = static["node_idx"]
        self.ppc_node_name = static["node_name"]
        self.ppc_node_vbase = static["node_vbase"]
        self.ppc_node_voltage = static["node_voltage"]
        self.ppc_node_angle = static["node_angle"]
        if self.keep_node_objects:
            if self.ppc_node_name.size != self.N:
                bus = self.ppc["bus"]
                bus_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
                bus_names = self.ppc.get("bus_name", np.asarray([f"bus_{int(idx)}" for idx in bus_ids], dtype=object))
                self.ppc_node_name = np.asarray(bus_names[self.active_bus_rows], dtype=object)
            self.node_list = [
                _PPCNode(int(idx), str(name), float(vbase), float(voltage), float(angle))
                for idx, name, vbase, voltage, angle in zip(
                    self.ppc_node_idx,
                    self.ppc_node_name,
                    self.ppc_node_vbase,
                    self.ppc_node_voltage,
                    self.ppc_node_angle,
                )
            ]
        else:
            self.node_list = []
        self.node_pos = static["node_pos"]
        if self.keep_node_objects and not self.node_pos:
            self.node_pos = {int(idx): pos for pos, idx in enumerate(self.ppc_node_idx)}
        for name in (
            "node_type", "V_spec", "theta_spec", "P_spec", "Q_spec", "Y",
            "_slack_mask", "_fixed_voltage_mask", "zero_edges", "comp_nodes", "comp_tree_edges", "phi_node", "phi_comp",
            "ref_phi_idx", "zero_branch_info", "N_phi", "theta_unknown", "V_unknown",
            "theta_idx", "V_idx", "n_theta", "n_V", "base_phi_re", "base_phi_im",
            "total_vars", "total_eq", "load_pos", "load_pv0", "load_pv1", "load_pv2",
            "load_qv0", "load_qv1", "load_qv2", "zero_idx", "zero_type", "zero_a",
            "zero_b", "zero_phi_a", "zero_phi_b", "zero_top_left_rows",
            "zero_top_left_cols", "zero_top_left_edge", "zero_top_left_kind",
            "zero_top_left_data", "zero_top_right_rows", "zero_top_right_cols",
            "zero_top_right_edge", "zero_top_right_kind", "zero_top_right_data",
            "zero_bottom_left_rows", "zero_bottom_left_cols", "zero_bottom_left_node",
            "zero_bottom_left_kind", "zero_bottom_left_data", "zero_bottom_right_rows",
            "zero_bottom_right_cols", "zero_bottom_right_data", "pq_theta_rows", "pq_v_cols",
            "theta_col_by_node", "v_col_by_node", "p_row_by_node", "q_row_by_node",
            "Y_jac_rows", "Y_jac_cols", "Y_jac_data", "Y_jac_g", "Y_jac_b", "Y_jac_diag",
            "Y_jac_diag_idx", "Y_jac_diag_nodes", "Y_jac_diag_g", "Y_jac_diag_b",
            "standard_jac_rows", "standard_jac_cols", "standard_jac_data",
            "std_jac_p_theta_idx", "std_jac_p_vm_idx", "std_jac_q_theta_idx",
            "std_jac_q_vm_idx", "std_jac_p_theta_slice", "std_jac_p_vm_slice",
            "std_jac_q_theta_slice", "std_jac_q_vm_slice", "std_jac_load_p_slice",
            "std_jac_load_q_slice", "standard_jac_csr_indices", "standard_jac_csr_indptr",
            "standard_jac_csr_order", "standard_jac_csr_data", "full_jac_raw_data",
            "full_jac_raw_to_csr_pos", "full_jac_csr_indices", "full_jac_csr_indptr",
            "full_jac_csr_data", "full_jac_standard_slice", "full_jac_zero_top_left_slice",
            "full_jac_zero_top_right_slice", "full_jac_zero_bottom_left_slice",
            "full_jac_zero_bottom_right_slice", "std_jac_load_nodes",
            "std_jac_load_extra_nodes", "std_jac_load_p_pos", "std_jac_load_q_pos", "pq_Bp", "pq_Bpp", "pq_Bp_factor", "pq_Bpp_factor",
            "_jac_delta", "_jac_cos_delta", "_jac_sin_delta", "_jac_vivj", "_jac_common_p", "_jac_common_q", "_jac_tmp",
            "_load_p_work", "_load_q_work", "_load_dp_work", "_load_dq_work",
            "_load_vm_work", "_load_value_work", "_load_aux_work", "_residual_work",
            "_state_theta", "_state_voltage", "_state_vc", "_empty_phi",
            "branch_i", "branch_j", "branch_yff", "branch_yft", "branch_ytf",
            "branch_ytt", "transformer_i", "transformer_j", "transformer_yff",
            "transformer_yft", "transformer_ytf", "transformer_ytt", "ppc_gen_rows",
            "ppc_gen_pos", "ppc_gen_share", "ppc_load_rows", "ppc_load_pos",
            "ppc_shunt_rows", "ppc_shunt_pos", "ppc_branch_rows", "ppc_transformer_rows",
        ):
            setattr(self, name, static[name])
        self.x = static["x0"].copy()
        self.standard_jac_data = static["standard_jac_data"].copy()
        self.zero_top_left_data = static["zero_top_left_data"].copy()
        self.zero_top_right_data = static["zero_top_right_data"].copy()
        self.zero_bottom_left_data = static["zero_bottom_left_data"].copy()
        self.zero_bottom_right_data = static["zero_bottom_right_data"].copy()
        self.full_jac_raw_data = static["full_jac_raw_data"].copy()
        self.full_jac_csr_data = static["full_jac_csr_data"].copy()
        self._cache_zero_jacobian_runtime_arrays()
        self._state_x_obj = None
        self._power_x_obj = None
        self._last_Ibus = None
        self._last_Sbus = None

    def _prepare_ppc_devices(self, branch, transformer, gen, load, shunt, zero_branch, switch, active_bus):
        """Convert live ppc device rows into compact node positions and specification arrays."""
        if gen.size:
            gen_rows = self._ppc_node_rows(gen[:, GEN_COLS["node"]])
            gen_live = (gen[:, GEN_COLS["run_stat"]] == 1) & active_bus[gen_rows]
            self.ppc_gen_rows = np.where(gen_live)[0].astype(np.int32)
            self.ppc_gen_pos = self.row_to_pos[gen_rows[self.ppc_gen_rows]]
            self.ppc_gen_share = np.ones(self.ppc_gen_rows.size, dtype=np.float64)
            live_gen = gen[self.ppc_gen_rows]
            controls = live_gen[:, GEN_COLS["control_type"]].astype(np.int32, copy=False)

            slack_mask = controls == CTRL_SLACK
            if np.any(slack_mask):
                slack_pos = self.ppc_gen_pos[slack_mask]
                self.node_type[slack_pos] = 'SLACK'
                self.V_spec[slack_pos] = live_gen[slack_mask, GEN_COLS["v_set"]]
                self.theta_spec[slack_pos] = self.ppc["bus"][self.active_bus_rows[slack_pos], BUS_COLS["angle"]]
                self.slack_node = int(slack_pos[-1])

            pv_mask = (controls == CTRL_PV) & (self.node_type[self.ppc_gen_pos] != 'SLACK')
            if np.any(pv_mask):
                pv_pos = self.ppc_gen_pos[pv_mask]
                pv_v = live_gen[pv_mask, GEN_COLS["v_set"]]
                for pos, v_set in zip(pv_pos, pv_v):
                    if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - v_set) > 1e-6:
                        raise ValueError(f"节点{pos}多个PV发电机电压设定冲突")
                self.node_type[pv_pos] = 'PV'
                self.V_spec[pv_pos] = pv_v
                np.add.at(self.P_spec, pv_pos, live_gen[pv_mask, GEN_COLS["p_set"]])

            pq_mask = (controls == CTRL_PQ) | (controls == CTRL_P)
            if np.any(pq_mask):
                pq_pos = self.ppc_gen_pos[pq_mask]
                np.add.at(self.P_spec, pq_pos, live_gen[pq_mask, GEN_COLS["p_set"]])
                np.add.at(self.Q_spec, pq_pos, live_gen[pq_mask, GEN_COLS["q_set"]])

            alpha = live_gen[:, GEN_COLS["alpha"]]
            alpha_sum = np.bincount(self.ppc_gen_pos, weights=alpha, minlength=self.N)
            gen_count = np.bincount(self.ppc_gen_pos, minlength=self.N)
            positive_alpha = alpha_sum[self.ppc_gen_pos] > 0.0
            self.ppc_gen_share[positive_alpha] = alpha[positive_alpha] / alpha_sum[self.ppc_gen_pos[positive_alpha]]
            no_alpha = ~positive_alpha
            if np.any(no_alpha):
                self.ppc_gen_share[no_alpha] = 1.0 / gen_count[self.ppc_gen_pos[no_alpha]]

        if shunt.size:
            shunt_rows = self._ppc_node_rows(shunt[:, SHUNT_COLS["node"]])
            shunt_live = (shunt[:, SHUNT_COLS["run_stat"]] == 1) & active_bus[shunt_rows]
            self.ppc_shunt_rows = np.where(shunt_live)[0].astype(np.int32)
            self.ppc_shunt_pos = self.row_to_pos[shunt_rows[self.ppc_shunt_rows]]
            live_shunt = shunt[self.ppc_shunt_rows]
            controls = live_shunt[:, SHUNT_COLS["control_type"]].astype(np.int32, copy=False)
            q_mask = controls == SHUNT_Q
            if np.any(q_mask):
                np.add.at(self.Q_spec, self.ppc_shunt_pos[q_mask], live_shunt[q_mask, SHUNT_COLS["q_set"]])
            v_mask = (controls == SHUNT_V) & (self.node_type[self.ppc_shunt_pos] != 'SLACK')
            if np.any(v_mask):
                v_pos = self.ppc_shunt_pos[v_mask]
                v_set_values = live_shunt[v_mask, SHUNT_COLS["v_set"]]
                for pos, v_set in zip(v_pos, v_set_values):
                    if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - v_set) > 1e-6:
                        raise ValueError(f"节点{pos}电压设定冲突")
                self.node_type[v_pos] = 'PV'
                self.V_spec[v_pos] = v_set_values

        if load.size:
            load_rows = self._ppc_node_rows(load[:, LOAD_COLS["node"]])
            load_live = (load[:, LOAD_COLS["run_stat"]] == 1) & active_bus[load_rows]
            self.ppc_load_rows = np.where(load_live)[0].astype(np.int32)
            self.ppc_load_pos = self.row_to_pos[load_rows[self.ppc_load_rows]]
            self.load_info = np.column_stack(
                (
                    self.ppc_load_pos,
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv0"]],
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv1"]],
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv2"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv0"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv1"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv2"]],
                )
            )

        if self.slack_node == -1:
            pv_indices = np.where(self.node_type == 'PV')[0]
            if pv_indices.size:
                self.slack_node = int(pv_indices[0])
                self.node_type[self.slack_node] = 'SLACK'
                self.theta_spec[self.slack_node] = self.ppc["bus"][self.active_bus_rows[self.slack_node], BUS_COLS["angle"]]
        if self.slack_node == -1:
            raise RuntimeError("电网中无平衡节点，无法进行潮流计算")

    def _prepare_ppc_y_matrix(self, branch, transformer, shunt):
        row_parts, col_parts, data_parts = [], [], []
        if branch.size:
            i_rows = self._ppc_node_rows(branch[:, BRANCH_COLS["i_node"]])
            j_rows = self._ppc_node_rows(branch[:, BRANCH_COLS["j_node"]])
            live = (branch[:, BRANCH_COLS["run_stat"]] == 1) & (self.row_to_pos[i_rows] >= 0) & (self.row_to_pos[j_rows] >= 0)
            self.ppc_branch_rows = np.where(live)[0].astype(np.int32)
            self.branch_i = self.row_to_pos[i_rows[self.ppc_branch_rows]]
            self.branch_j = self.row_to_pos[j_rows[self.ppc_branch_rows]]
            n_branch = self.ppc_branch_rows.size
            self.branch_yff = np.empty(n_branch, dtype=np.complex128)
            self.branch_yft = np.empty(n_branch, dtype=np.complex128)
            self.branch_ytf = np.empty(n_branch, dtype=np.complex128)
            self.branch_ytt = np.empty(n_branch, dtype=np.complex128)
            self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt = matpower_branch_stamp_vectorized(
                branch[self.ppc_branch_rows, BRANCH_COLS["r"]],
                branch[self.ppc_branch_rows, BRANCH_COLS["x"]],
                branch[self.ppc_branch_rows, BRANCH_COLS["b"]],
            )
            row_parts.append(np.column_stack((self.branch_i, self.branch_i, self.branch_j, self.branch_j)).ravel())
            col_parts.append(np.column_stack((self.branch_i, self.branch_j, self.branch_i, self.branch_j)).ravel())
            data_parts.append(np.column_stack((self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt)).ravel())

        if transformer.size:
            i_rows = self._ppc_node_rows(transformer[:, TRANSFORMER_COLS["i_node"]])
            j_rows = self._ppc_node_rows(transformer[:, TRANSFORMER_COLS["j_node"]])
            live = (transformer[:, TRANSFORMER_COLS["run_stat"]] == 1) & (self.row_to_pos[i_rows] >= 0) & (self.row_to_pos[j_rows] >= 0)
            self.ppc_transformer_rows = np.where(live)[0].astype(np.int32)
            self.transformer_i = self.row_to_pos[i_rows[self.ppc_transformer_rows]]
            self.transformer_j = self.row_to_pos[j_rows[self.ppc_transformer_rows]]
            n_transformer = self.ppc_transformer_rows.size
            self.transformer_yff = np.empty(n_transformer, dtype=np.complex128)
            self.transformer_yft = np.empty(n_transformer, dtype=np.complex128)
            self.transformer_ytf = np.empty(n_transformer, dtype=np.complex128)
            self.transformer_ytt = np.empty(n_transformer, dtype=np.complex128)
            (
                self.transformer_yff,
                self.transformer_yft,
                self.transformer_ytf,
                self.transformer_ytt,
            ) = matpower_branch_stamp_vectorized(
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["r"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["x"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["b"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["tap"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["shift"]],
            )
            row_parts.append(
                np.column_stack((self.transformer_i, self.transformer_i, self.transformer_j, self.transformer_j)).ravel()
            )
            col_parts.append(
                np.column_stack((self.transformer_i, self.transformer_j, self.transformer_i, self.transformer_j)).ravel()
            )
            data_parts.append(
                np.column_stack((self.transformer_yff, self.transformer_yft, self.transformer_ytf, self.transformer_ytt)).ravel()
            )

        if shunt.size and self.ppc_shunt_rows.size:
            shunt_live = shunt[self.ppc_shunt_rows]
            control = shunt_live[:, SHUNT_COLS["control_type"]].astype(np.int32)
            y_sh = shunt_live[:, SHUNT_COLS["g_set"]] + 1j * shunt_live[:, SHUNT_COLS["b_set"]]
            mask = ((control == SHUNT_B) | (control == SHUNT_Z) | (shunt_live[:, SHUNT_COLS["g_set"]] != 0.0)) & (y_sh != 0.0)
            if np.any(mask):
                shunt_pos = self.ppc_shunt_pos[mask]
                row_parts.append(shunt_pos)
                col_parts.append(shunt_pos)
                data_parts.append(y_sh[mask])

        if row_parts:
            rows = np.concatenate(row_parts).astype(np.int32, copy=False)
            cols = np.concatenate(col_parts).astype(np.int32, copy=False)
            data = np.concatenate(data_parts).astype(np.complex128, copy=False)
        else:
            rows = cols = np.array([], dtype=np.int32)
            data = np.array([], dtype=np.complex128)
        self.Y = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        self.Y.sum_duplicates()

    def _prepare_ppc_zero_edges(self, zero_branch, switch, active_bus, breaker=None):
        self.zero_edges = []
        if zero_branch.size:
            i_rows = self._ppc_node_rows(zero_branch[:, ZERO_BRANCH_COLS["i_node"]])
            j_rows = self._ppc_node_rows(zero_branch[:, ZERO_BRANCH_COLS["j_node"]])
            live = (zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] == 1) & active_bus[i_rows] & active_bus[j_rows]
            for row_idx in np.where(live)[0]:
                a, b = int(self.row_to_pos[i_rows[row_idx]]), int(self.row_to_pos[j_rows[row_idx]])
                if not self._is_redundant_slack_zero_edge(a, b):
                    self.zero_edges.append((int(row_idx), 0, a, b))
        if breaker is not None and breaker.size:
            i_rows = self._ppc_node_rows(breaker[:, SWITCH_COLS["i_node"]])
            j_rows = self._ppc_node_rows(breaker[:, SWITCH_COLS["j_node"]])
            live = (
                (breaker[:, SWITCH_COLS["run_stat"]] == 1)
                & (breaker[:, SWITCH_COLS["status"]] == 1)
                & active_bus[i_rows]
                & active_bus[j_rows]
            )
            for row_idx in np.where(live)[0]:
                a, b = int(self.row_to_pos[i_rows[row_idx]]), int(self.row_to_pos[j_rows[row_idx]])
                if not self._is_redundant_slack_zero_edge(a, b):
                    self.zero_edges.append((int(row_idx), 2, a, b))
        if switch.size:
            i_rows = self._ppc_node_rows(switch[:, SWITCH_COLS["i_node"]])
            j_rows = self._ppc_node_rows(switch[:, SWITCH_COLS["j_node"]])
            live = (
                (switch[:, SWITCH_COLS["run_stat"]] == 1)
                & (switch[:, SWITCH_COLS["status"]] == 1)
                & active_bus[i_rows]
                & active_bus[j_rows]
            )
            for row_idx in np.where(live)[0]:
                a, b = int(self.row_to_pos[i_rows[row_idx]]), int(self.row_to_pos[j_rows[row_idx]])
                if not self._is_redundant_slack_zero_edge(a, b):
                    self.zero_edges.append((int(row_idx), 1, a, b))
        self._prepare_zero_branch_components()

    def _finalize_prepared_arrays(self):
        """Finalize variable/equation indices after node specs, Y and zero edges are ready."""
        self.theta_unknown = np.where(self.node_type != 'SLACK')[0].astype(np.int32)
        self.V_unknown = np.where(self.node_type == 'PQ')[0].astype(np.int32)
        self.theta_idx = {int(pos): i for i, pos in enumerate(self.theta_unknown)}
        self.V_idx = {int(pos): i for i, pos in enumerate(self.V_unknown)}
        self.n_theta = len(self.theta_unknown)
        self.n_V = len(self.V_unknown)
        self.base_phi_re = self.n_theta + self.n_V
        self.base_phi_im = self.base_phi_re + self.N_phi
        self.total_vars = self.base_phi_im + self.N_phi
        n_tree = sum(len(edges) for edges in self.comp_tree_edges)
        n_phi_fix = 2 * len(self.comp_nodes)
        self.total_eq = self.n_theta + self.n_V + 2 * n_tree + n_phi_fix
        self.x = np.zeros(self.total_vars, dtype=np.float64)
        self.x[self.n_theta:self.n_theta + self.n_V] = 1.0
        self._state_theta = np.empty(self.N, dtype=np.float64)
        self._state_voltage = np.empty(self.N, dtype=np.float64)
        self._state_vc = np.empty(self.N, dtype=np.complex128)
        self._empty_phi = np.array([], dtype=np.float64)
        self._load_p_work = np.zeros(self.N, dtype=np.float64)
        self._load_q_work = np.zeros(self.N, dtype=np.float64)
        self._load_dp_work = np.zeros(self.N, dtype=np.float64)
        self._load_dq_work = np.zeros(self.N, dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._residual_work = np.zeros(self.total_eq, dtype=np.float64)
        self._cache_static_numeric_arrays()
        self._cache_pq_decoupled_matrices()
        if self.total_vars != self.total_eq:
            warnings.warn(f"变量数({self.total_vars})与方程数({self.total_eq})不匹配！")

    def _cache_static_numeric_arrays(self):
        """Cache numeric arrays used by Newton iterations without object-model access."""
        self._cache_node_type_masks()
        if len(self.load_info):
            load_data = np.asarray(self.load_info, dtype=np.float64)
            self.load_pos = load_data[:, 0].astype(np.int32)
            self.load_pv0 = load_data[:, 1]
            self.load_pv1 = load_data[:, 2]
            self.load_pv2 = load_data[:, 3]
            self.load_qv0 = load_data[:, 4]
            self.load_qv1 = load_data[:, 5]
            self.load_qv2 = load_data[:, 6]
        else:
            self.load_pos = np.array([], dtype=np.int32)
            self.load_pv0 = self.load_pv1 = self.load_pv2 = np.array([], dtype=np.float64)
            self.load_qv0 = self.load_qv1 = self.load_qv2 = np.array([], dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)
        if self.N_phi > 0 and len(self.zero_branch_info) > 0:
            self.zero_idx = self.zero_branch_info[:, 0].astype(np.int32)
            self.zero_type = self.zero_branch_info[:, 1].astype(np.int32)
            self.zero_a = self.zero_branch_info[:, 2].astype(np.int32)
            self.zero_b = self.zero_branch_info[:, 3].astype(np.int32)
            self.zero_phi_a = self.zero_branch_info[:, 4].astype(np.int32)
            self.zero_phi_b = self.zero_branch_info[:, 5].astype(np.int32)
        else:
            self.zero_idx = self.zero_type = self.zero_a = self.zero_b = np.array([], dtype=np.int32)
            self.zero_phi_a = self.zero_phi_b = np.array([], dtype=np.int32)
        self.pq_theta_rows = np.fromiter(
            (self.theta_idx[int(pos)] for pos in self.V_unknown),
            dtype=np.int32,
            count=self.V_unknown.size,
        )
        self.pq_v_cols = np.arange(self.V_unknown.size, dtype=np.int32)
        self.theta_col_by_node = np.full(self.N, -1, dtype=np.int32)
        self.v_col_by_node = np.full(self.N, -1, dtype=np.int32)
        self.p_row_by_node = np.full(self.N, -1, dtype=np.int32)
        self.q_row_by_node = np.full(self.N, -1, dtype=np.int32)
        self.theta_col_by_node[self.theta_unknown] = np.arange(self.n_theta, dtype=np.int32)
        self.p_row_by_node[self.theta_unknown] = self.theta_col_by_node[self.theta_unknown]
        self.v_col_by_node[self.V_unknown] = self.n_theta + np.arange(self.n_V, dtype=np.int32)
        self.q_row_by_node[self.V_unknown] = self.n_theta + np.arange(self.n_V, dtype=np.int32)
        y_csr = self.Y.tocsr()
        self.Y_jac_rows = np.repeat(np.arange(self.N, dtype=np.int32), np.diff(y_csr.indptr))
        self.Y_jac_cols = y_csr.indices.astype(np.int32, copy=True)
        self.Y_jac_data = y_csr.data.astype(np.complex128, copy=True)
        self.Y_jac_g = self.Y_jac_data.real
        self.Y_jac_b = self.Y_jac_data.imag
        self.Y_jac_diag = y_csr.diagonal().astype(np.complex128, copy=False)
        self.Y_jac_diag_idx = np.flatnonzero(self.Y_jac_rows == self.Y_jac_cols)
        self.Y_jac_diag_nodes = self.Y_jac_rows[self.Y_jac_diag_idx]
        self.Y_jac_diag_g = self.Y_jac_diag[self.Y_jac_diag_nodes].real
        self.Y_jac_diag_b = self.Y_jac_diag[self.Y_jac_diag_nodes].imag
        self._cache_standard_jacobian_pattern()
        self._cache_zero_jacobian_pattern()
        self._cache_full_jacobian_pattern()
        self.Y_diag = np.array([], dtype=np.complex128)
        self.Y_offdiag_indices = []
        self.Y_offdiag_data = []

    def _cache_standard_jacobian_pattern(self):
        """Cache fixed COO coordinates for the standard P/Q Jacobian."""
        self.standard_jac_rows = np.array([], dtype=np.int32)
        self.standard_jac_cols = np.array([], dtype=np.int32)
        self.standard_jac_data = np.array([], dtype=np.float64)
        self.std_jac_p_theta_idx = self.std_jac_p_vm_idx = np.array([], dtype=np.intp)
        self.std_jac_q_theta_idx = self.std_jac_q_vm_idx = np.array([], dtype=np.intp)
        self.std_jac_p_theta_slice = self.std_jac_p_vm_slice = slice(0, 0)
        self.std_jac_q_theta_slice = self.std_jac_q_vm_slice = slice(0, 0)
        self.std_jac_load_p_slice = self.std_jac_load_q_slice = slice(0, 0)
        self.standard_jac_csr_indices = np.array([], dtype=np.int32)
        self.standard_jac_csr_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csr_order = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.std_jac_load_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_extra_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_p_pos = np.array([], dtype=np.intp)
        self.std_jac_load_q_pos = np.array([], dtype=np.intp)

        if not self.Y_jac_rows.size:
            return

        p_rows = self.p_row_by_node[self.Y_jac_rows]
        q_rows = self.q_row_by_node[self.Y_jac_rows]
        theta_cols = self.theta_col_by_node[self.Y_jac_cols]
        v_cols = self.v_col_by_node[self.Y_jac_cols]

        self.std_jac_p_theta_idx = np.flatnonzero((p_rows >= 0) & (theta_cols >= 0))
        self.std_jac_p_vm_idx = np.flatnonzero((p_rows >= 0) & (v_cols >= 0))
        self.std_jac_q_theta_idx = np.flatnonzero((q_rows >= 0) & (theta_cols >= 0))
        self.std_jac_q_vm_idx = np.flatnonzero((q_rows >= 0) & (v_cols >= 0))

        rows_parts = [
            p_rows[self.std_jac_p_theta_idx],
            p_rows[self.std_jac_p_vm_idx],
            q_rows[self.std_jac_q_theta_idx],
            q_rows[self.std_jac_q_vm_idx],
        ]
        cols_parts = [
            theta_cols[self.std_jac_p_theta_idx],
            v_cols[self.std_jac_p_vm_idx],
            theta_cols[self.std_jac_q_theta_idx],
            v_cols[self.std_jac_q_vm_idx],
        ]
        cursor = 0
        self.std_jac_p_theta_slice = slice(cursor, cursor + self.std_jac_p_theta_idx.size)
        cursor = self.std_jac_p_theta_slice.stop
        self.std_jac_p_vm_slice = slice(cursor, cursor + self.std_jac_p_vm_idx.size)
        cursor = self.std_jac_p_vm_slice.stop
        self.std_jac_q_theta_slice = slice(cursor, cursor + self.std_jac_q_theta_idx.size)
        cursor = self.std_jac_q_theta_slice.stop
        self.std_jac_q_vm_slice = slice(cursor, cursor + self.std_jac_q_vm_idx.size)
        cursor = self.std_jac_q_vm_slice.stop

        if self.V_unknown.size:
            diag_idx_by_node = np.full(self.N, -1, dtype=np.intp)
            diag_idx_by_node[self.Y_jac_diag_nodes] = self.Y_jac_diag_idx
            load_diag_idx = diag_idx_by_node[self.V_unknown]
            yidx_to_p_vm_pos = np.full(self.Y_jac_rows.size, -1, dtype=np.intp)
            yidx_to_q_vm_pos = np.full(self.Y_jac_rows.size, -1, dtype=np.intp)
            yidx_to_p_vm_pos[self.std_jac_p_vm_idx] = np.arange(self.std_jac_p_vm_idx.size, dtype=np.intp)
            yidx_to_q_vm_pos[self.std_jac_q_vm_idx] = np.arange(self.std_jac_q_vm_idx.size, dtype=np.intp)
            p_local_pos = np.where(load_diag_idx >= 0, yidx_to_p_vm_pos[load_diag_idx], -1)
            q_local_pos = np.where(load_diag_idx >= 0, yidx_to_q_vm_pos[load_diag_idx], -1)
            direct_mask = (p_local_pos >= 0) & (q_local_pos >= 0)
            self.std_jac_load_nodes = self.V_unknown[direct_mask]
            self.std_jac_load_p_pos = self.std_jac_p_vm_slice.start + p_local_pos[direct_mask]
            self.std_jac_load_q_pos = self.std_jac_q_vm_slice.start + q_local_pos[direct_mask]

            self.std_jac_load_extra_nodes = self.V_unknown[~direct_mask]
            if self.std_jac_load_extra_nodes.size:
                v_load_cols = self.v_col_by_node[self.std_jac_load_extra_nodes]
                rows_parts.extend((self.p_row_by_node[self.std_jac_load_extra_nodes], self.q_row_by_node[self.std_jac_load_extra_nodes]))
                cols_parts.extend((v_load_cols, v_load_cols))
                self.std_jac_load_p_slice = slice(cursor, cursor + self.std_jac_load_extra_nodes.size)
                cursor = self.std_jac_load_p_slice.stop
                self.std_jac_load_q_slice = slice(cursor, cursor + self.std_jac_load_extra_nodes.size)
                cursor = self.std_jac_load_q_slice.stop

        self.standard_jac_rows = np.concatenate(rows_parts).astype(np.int32, copy=False)
        self.standard_jac_cols = np.concatenate(cols_parts).astype(np.int32, copy=False)
        self.standard_jac_data = np.empty(cursor, dtype=np.float64)
        if cursor:
            marker = np.arange(cursor, dtype=np.float64)
            pattern = coo_matrix(
                (marker, (self.standard_jac_rows, self.standard_jac_cols)),
                shape=(self.n_theta + self.n_V, self.n_theta + self.n_V),
            ).tocsr()
            self.standard_jac_csr_indices = pattern.indices.astype(np.int32, copy=True)
            self.standard_jac_csr_indptr = pattern.indptr.astype(np.int32, copy=True)
            self.standard_jac_csr_order = pattern.data.astype(np.intp, copy=True)
            self.standard_jac_csr_data = np.empty_like(self.standard_jac_data)
            y_nnz = self.Y_jac_rows.size
            self._jac_delta = np.empty(y_nnz, dtype=np.float64)
            self._jac_cos_delta = np.empty(y_nnz, dtype=np.float64)
            self._jac_sin_delta = np.empty(y_nnz, dtype=np.float64)
            self._jac_vivj = np.empty(y_nnz, dtype=np.float64)
            self._jac_common_p = np.empty(y_nnz, dtype=np.float64)
            self._jac_common_q = np.empty(y_nnz, dtype=np.float64)
            self._jac_tmp = np.empty(y_nnz, dtype=np.float64)

    @staticmethod
    def _group_indices_by_kind(kind, source_index, n_kind):
        """Group data positions and source indices once, avoiding per-iteration masks."""
        empty_groups = tuple(np.array([], dtype=np.intp) for _ in range(n_kind))
        if kind.size == 0:
            return empty_groups, empty_groups
        pos_groups = []
        index_groups = []
        for code in range(n_kind):
            pos = np.flatnonzero(kind == code).astype(np.intp, copy=False)
            pos_groups.append(pos)
            if pos.size:
                index_groups.append(source_index[pos].astype(np.intp, copy=False))
            else:
                index_groups.append(np.array([], dtype=np.intp))
        return tuple(pos_groups), tuple(index_groups)

    def _cache_zero_jacobian_runtime_arrays(self):
        """Cache runtime grouping and work arrays for zero-impedance Jacobian refresh."""
        self.zero_top_left_pos_by_kind, self.zero_top_left_edge_by_kind = self._group_indices_by_kind(
            self.zero_top_left_kind,
            self.zero_top_left_edge,
            8,
        )
        self.zero_top_right_pos_by_kind, self.zero_top_right_edge_by_kind = self._group_indices_by_kind(
            self.zero_top_right_kind,
            self.zero_top_right_edge,
            16,
        )
        self.zero_bottom_left_pos_by_kind, self.zero_bottom_left_node_by_kind = self._group_indices_by_kind(
            self.zero_bottom_left_kind,
            self.zero_bottom_left_node,
            8,
        )

        n_edge = int(self.zero_a.size)
        if n_edge == 0:
            self._zero_I_re = np.array([], dtype=np.float64)
            self._zero_I_im = np.array([], dtype=np.float64)
            self._zero_Va = np.array([], dtype=np.float64)
            self._zero_Vb = np.array([], dtype=np.float64)
            self._zero_cos_a = np.array([], dtype=np.float64)
            self._zero_sin_a = np.array([], dtype=np.float64)
            self._zero_cos_b = np.array([], dtype=np.float64)
            self._zero_sin_b = np.array([], dtype=np.float64)
            self._zero_a_pv = np.array([], dtype=np.float64)
            self._zero_a_pt = np.array([], dtype=np.float64)
            self._zero_b_pv = np.array([], dtype=np.float64)
            self._zero_b_pt = np.array([], dtype=np.float64)
            self._zero_va_cos = np.array([], dtype=np.float64)
            self._zero_va_sin = np.array([], dtype=np.float64)
            self._zero_vb_cos = np.array([], dtype=np.float64)
            self._zero_vb_sin = np.array([], dtype=np.float64)
            self._zero_tmp = np.array([], dtype=np.float64)
            return

        self._zero_I_re = np.empty(n_edge, dtype=np.float64)
        self._zero_I_im = np.empty(n_edge, dtype=np.float64)
        self._zero_Va = np.empty(n_edge, dtype=np.float64)
        self._zero_Vb = np.empty(n_edge, dtype=np.float64)
        self._zero_cos_a = np.empty(n_edge, dtype=np.float64)
        self._zero_sin_a = np.empty(n_edge, dtype=np.float64)
        self._zero_cos_b = np.empty(n_edge, dtype=np.float64)
        self._zero_sin_b = np.empty(n_edge, dtype=np.float64)
        self._zero_a_pv = np.empty(n_edge, dtype=np.float64)
        self._zero_a_pt = np.empty(n_edge, dtype=np.float64)
        self._zero_b_pv = np.empty(n_edge, dtype=np.float64)
        self._zero_b_pt = np.empty(n_edge, dtype=np.float64)
        self._zero_va_cos = np.empty(n_edge, dtype=np.float64)
        self._zero_va_sin = np.empty(n_edge, dtype=np.float64)
        self._zero_vb_cos = np.empty(n_edge, dtype=np.float64)
        self._zero_vb_sin = np.empty(n_edge, dtype=np.float64)
        self._zero_tmp = np.empty(n_edge, dtype=np.float64)

    def _cache_zero_jacobian_pattern(self):
        """Cache fixed zero-impedance Jacobian coordinates for Newton iterations."""
        self.zero_top_left_rows = np.array([], dtype=np.int32)
        self.zero_top_left_cols = np.array([], dtype=np.int32)
        self.zero_top_left_edge = np.array([], dtype=np.intp)
        self.zero_top_left_kind = np.array([], dtype=np.int8)
        self.zero_top_left_data = np.array([], dtype=np.float64)
        self.zero_top_right_rows = np.array([], dtype=np.int32)
        self.zero_top_right_cols = np.array([], dtype=np.int32)
        self.zero_top_right_edge = np.array([], dtype=np.intp)
        self.zero_top_right_kind = np.array([], dtype=np.int8)
        self.zero_top_right_data = np.array([], dtype=np.float64)
        self.zero_bottom_left_rows = np.array([], dtype=np.int32)
        self.zero_bottom_left_cols = np.array([], dtype=np.int32)
        self.zero_bottom_left_node = np.array([], dtype=np.int32)
        self.zero_bottom_left_kind = np.array([], dtype=np.int8)
        self.zero_bottom_left_data = np.array([], dtype=np.float64)
        self.zero_bottom_right_rows = np.array([], dtype=np.int32)
        self.zero_bottom_right_cols = np.array([], dtype=np.int32)
        self.zero_bottom_right_data = np.array([], dtype=np.float64)
        self._cache_zero_jacobian_runtime_arrays()

        if self.N_phi <= 0:
            return

        top_left_rows, top_left_cols, top_left_edge, top_left_kind = [], [], [], []
        top_right_rows, top_right_cols, top_right_edge, top_right_kind = [], [], [], []
        for edge_pos, (a_raw, b_raw, phi_a_raw, phi_b_raw) in enumerate(
            zip(self.zero_a, self.zero_b, self.zero_phi_a, self.zero_phi_b)
        ):
            a = int(a_raw)
            b = int(b_raw)
            phi_a = int(phi_a_raw)
            phi_b = int(phi_b_raw)
            phi_cols = (phi_a, self.N_phi + phi_a, phi_b, self.N_phi + phi_b)

            if self.node_type[a] != 'SLACK':
                eq_a = self.theta_idx[a]
                top_left_rows.append(eq_a)
                top_left_cols.append(eq_a)
                top_left_edge.append(edge_pos)
                top_left_kind.append(0)
                if self.node_type[a] == 'PQ':
                    top_left_rows.append(eq_a)
                    top_left_cols.append(self.n_theta + self.V_idx[a])
                    top_left_edge.append(edge_pos)
                    top_left_kind.append(1)
                top_right_rows.extend([eq_a] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([0, 1, 2, 3])

            if self.node_type[a] == 'PQ':
                eq_a = self.n_theta + self.V_idx[a]
                top_left_rows.extend([eq_a, eq_a])
                top_left_cols.extend([self.theta_idx[a], self.n_theta + self.V_idx[a]])
                top_left_edge.extend([edge_pos, edge_pos])
                top_left_kind.extend([2, 3])
                top_right_rows.extend([eq_a] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([4, 5, 6, 7])

            if self.node_type[b] != 'SLACK':
                eq_b = self.theta_idx[b]
                top_left_rows.append(eq_b)
                top_left_cols.append(eq_b)
                top_left_edge.append(edge_pos)
                top_left_kind.append(4)
                if self.node_type[b] == 'PQ':
                    top_left_rows.append(eq_b)
                    top_left_cols.append(self.n_theta + self.V_idx[b])
                    top_left_edge.append(edge_pos)
                    top_left_kind.append(5)
                top_right_rows.extend([eq_b] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([8, 9, 10, 11])

            if self.node_type[b] == 'PQ':
                eq_b = self.n_theta + self.V_idx[b]
                top_left_rows.extend([eq_b, eq_b])
                top_left_cols.extend([self.theta_idx[b], self.n_theta + self.V_idx[b]])
                top_left_edge.extend([edge_pos, edge_pos])
                top_left_kind.extend([6, 7])
                top_right_rows.extend([eq_b] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([12, 13, 14, 15])

        self.zero_top_left_rows = np.asarray(top_left_rows, dtype=np.int32)
        self.zero_top_left_cols = np.asarray(top_left_cols, dtype=np.int32)
        self.zero_top_left_edge = np.asarray(top_left_edge, dtype=np.intp)
        self.zero_top_left_kind = np.asarray(top_left_kind, dtype=np.int8)
        self.zero_top_left_data = np.empty(self.zero_top_left_rows.size, dtype=np.float64)
        self.zero_top_right_rows = np.asarray(top_right_rows, dtype=np.int32)
        self.zero_top_right_cols = np.asarray(top_right_cols, dtype=np.int32)
        self.zero_top_right_edge = np.asarray(top_right_edge, dtype=np.intp)
        self.zero_top_right_kind = np.asarray(top_right_kind, dtype=np.int8)
        self.zero_top_right_data = np.empty(self.zero_top_right_rows.size, dtype=np.float64)

        bottom_left_rows, bottom_left_cols, bottom_left_node, bottom_left_kind = [], [], [], []
        eq_idx = 0
        for edges in self.comp_tree_edges:
            for edge_idx in edges:
                _, _, a, b = self.zero_edges[edge_idx]
                a = int(a)
                b = int(b)
                if self.node_type[a] != 'SLACK':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(0)
                if self.node_type[a] == 'PQ':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(1)
                if self.node_type[b] != 'SLACK':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(2)
                if self.node_type[b] == 'PQ':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(3)
                eq_idx += 1

                if self.node_type[a] != 'SLACK':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(4)
                if self.node_type[a] == 'PQ':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(5)
                if self.node_type[b] != 'SLACK':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(6)
                if self.node_type[b] == 'PQ':
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(7)
                eq_idx += 1

        self.zero_bottom_left_rows = np.asarray(bottom_left_rows, dtype=np.int32)
        self.zero_bottom_left_cols = np.asarray(bottom_left_cols, dtype=np.int32)
        self.zero_bottom_left_node = np.asarray(bottom_left_node, dtype=np.int32)
        self.zero_bottom_left_kind = np.asarray(bottom_left_kind, dtype=np.int8)
        self.zero_bottom_left_data = np.empty(self.zero_bottom_left_rows.size, dtype=np.float64)

        bottom_right_rows, bottom_right_cols = [], []
        for c, idx_phi in enumerate(self.ref_phi_idx):
            bottom_right_rows.append(eq_idx + 2 * c)
            bottom_right_cols.append(int(idx_phi))
            bottom_right_rows.append(eq_idx + 2 * c + 1)
            bottom_right_cols.append(self.N_phi + int(idx_phi))
        self.zero_bottom_right_rows = np.asarray(bottom_right_rows, dtype=np.int32)
        self.zero_bottom_right_cols = np.asarray(bottom_right_cols, dtype=np.int32)
        self.zero_bottom_right_data = np.ones(len(bottom_right_rows), dtype=np.float64)
        self._cache_zero_jacobian_runtime_arrays()

    def _cache_full_jacobian_pattern(self):
        """Precompute the full AC Jacobian CSR pattern for standard and zero-branch blocks."""
        self.full_jac_raw_data = np.array([], dtype=np.float64)
        self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.full_jac_csr_indices = np.array([], dtype=np.int32)
        self.full_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
        self.full_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_standard_slice = slice(0, 0)
        self.full_jac_zero_top_left_slice = slice(0, 0)
        self.full_jac_zero_top_right_slice = slice(0, 0)
        self.full_jac_zero_bottom_left_slice = slice(0, 0)
        self.full_jac_zero_bottom_right_slice = slice(0, 0)

        if self.total_eq == 0 or self.total_vars == 0 or self.standard_jac_rows.size == 0:
            return

        rows_parts = []
        cols_parts = []
        raw_count = 0

        def add_part(name, rows, cols):
            nonlocal raw_count
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            if rows.size != cols.size:
                raise ValueError(f"AC Jacobian pattern part {name!r} has mismatched row/column lengths")
            part_slice = slice(raw_count, raw_count + rows.size)
            setattr(self, f"full_jac_{name}_slice", part_slice)
            raw_count += rows.size
            if rows.size:
                rows_parts.append(rows)
                cols_parts.append(cols)

        std_eq = self.n_theta + self.n_V
        std_vars = std_eq
        add_part("standard", self.standard_jac_rows, self.standard_jac_cols)
        if self.N_phi:
            add_part("zero_top_left", self.zero_top_left_rows, self.zero_top_left_cols)
            add_part("zero_top_right", self.zero_top_right_rows, self.zero_top_right_cols + std_vars)
            add_part("zero_bottom_left", self.zero_bottom_left_rows + std_eq, self.zero_bottom_left_cols)
            add_part("zero_bottom_right", self.zero_bottom_right_rows + std_eq, self.zero_bottom_right_cols + std_vars)

        self.full_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        pattern = coo_matrix(
            (np.ones(raw_count, dtype=np.float64), (raw_rows, raw_cols)),
            shape=(self.total_eq, self.total_vars),
        ).tocsr()
        pattern.sum_duplicates()
        self.full_jac_csr_indices = pattern.indices.astype(np.int32, copy=True)
        self.full_jac_csr_indptr = pattern.indptr.astype(np.int32, copy=True)
        self.full_jac_csr_data = np.empty(self.full_jac_csr_indices.size, dtype=np.float64)

        positions = {}
        for row in range(self.total_eq):
            start = int(self.full_jac_csr_indptr[row])
            end = int(self.full_jac_csr_indptr[row + 1])
            for pos in range(start, end):
                positions[(row, int(self.full_jac_csr_indices[pos]))] = pos
        self.full_jac_raw_to_csr_pos = np.fromiter(
            (positions[(int(row), int(col))] for row, col in zip(raw_rows, raw_cols)),
            dtype=np.intp,
            count=raw_count,
        )

    def _cache_pq_decoupled_matrices(self):
        """Cache fixed susceptance matrices for the fast-decoupled PQ method."""
        self.pq_Bp = None
        self.pq_Bpp = None
        self.pq_Bp_factor = None
        self.pq_Bpp_factor = None
        if self.algorithm != "pq" or self.Y is None or self.N_phi > 0:
            return
        b_matrix = self._build_fast_decoupled_b_matrix()
        self.pq_Bp = b_matrix[self.theta_unknown, :][:, self.theta_unknown].tocsr()
        self.pq_Bpp = b_matrix[self.V_unknown, :][:, self.V_unknown].tocsr()
        if self.n_theta:
            self.pq_Bp_factor = splu(self.pq_Bp.tocsc())
        if self.n_V:
            self.pq_Bpp_factor = splu(self.pq_Bpp.tocsc())

    def _build_fast_decoupled_b_matrix(self):
        """Build the fixed B matrix used by fast-decoupled load flow.

        The fast-decoupled method relies on a lossless network approximation.
        Using ``-Y.imag`` from the full AC admittance matrix includes the impact
        of branch resistance and shunts, which seriously degrades convergence.
        """
        row_parts = []
        col_parts = []
        data_parts = []

        def append_series(i, j, x, tap=None):
            if i.size == 0:
                return
            b = np.divide(1.0, x, out=np.zeros_like(x, dtype=np.float64), where=np.abs(x) > 1e-12)
            if tap is None:
                yff = b
                yft = -b
                ytf = -b
                ytt = b
            else:
                tap_mag = tap.copy()
                tap_mag[np.abs(tap_mag) < 1e-12] = 1.0
                yff = b / (tap_mag * tap_mag)
                yft = -b / tap_mag
                ytf = -b / tap_mag
                ytt = b
            row_parts.append(np.column_stack((i, i, j, j)).ravel())
            col_parts.append(np.column_stack((i, j, i, j)).ravel())
            data_parts.append(np.column_stack((yff, yft, ytf, ytt)).ravel())

        if self.array_mode:
            branch = self.ppc["branch"]
            transformer = self.ppc["transformer"]
            append_series(
                self.branch_i,
                self.branch_j,
                branch[self.ppc_branch_rows, BRANCH_COLS["x"]] if self.ppc_branch_rows.size else np.array([], dtype=np.float64),
            )
            append_series(
                self.transformer_i,
                self.transformer_j,
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["x"]]
                if self.ppc_transformer_rows.size
                else np.array([], dtype=np.float64),
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["tap"]]
                if self.ppc_transformer_rows.size
                else np.array([], dtype=np.float64),
            )
        else:
            append_series(
                self.branch_i,
                self.branch_j,
                np.asarray([br.x for br in self.live_branches], dtype=np.float64),
            )
            append_series(
                self.transformer_i,
                self.transformer_j,
                np.asarray([tr.x for tr in self.live_transformers], dtype=np.float64),
                np.asarray([getattr(tr, "tap", 1.0) for tr in self.live_transformers], dtype=np.float64),
            )

        if not row_parts:
            return csr_matrix((self.N, self.N), dtype=np.float64)
        rows = np.concatenate(row_parts).astype(np.int32, copy=False)
        cols = np.concatenate(col_parts).astype(np.int32, copy=False)
        data = np.concatenate(data_parts).astype(np.float64, copy=False)
        matrix = coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        matrix.sum_duplicates()
        return matrix

    # --------------------------------------------------------------------------
    # 预处理阶段（精简版）
    # --------------------------------------------------------------------------
    def prepare(self):
        """预处理：合并带电拓扑岛，初始化参数并定义变量/方程索引。"""
        if self.array_mode:
            self._prepare_from_ppc()
            return

        if self.target_island is None:
            if not getattr(self.net, 'islands', None):
                self.net.topo()

            self.calc_islands = [isl for isl in self.net.islands if isl.is_alive]
            self.skipped_islands = [isl for isl in self.net.islands if not isl.is_alive]
            if not self.calc_islands:
                raise RuntimeError("无存活的拓扑岛，无法进行潮流计算")

            # 多岛场景下在一个 Newton 系统里统一求解，避免主程序逐岛循环调用。
            self.isl = SimpleNamespace(
                idx=0,
                is_alive=True,
                buses=[bus for isl in self.calc_islands for bus in isl.buses],
                gens=[gen for isl in self.calc_islands for gen in isl.gens],
                loads=[load for isl in self.calc_islands for load in isl.loads],
                branches=[br for isl in self.calc_islands for br in isl.branches],
                transformers=[tr for isl in self.calc_islands for tr in isl.transformers],
                zero_branches=[zbr for isl in self.calc_islands for zbr in isl.zero_branches],
                switches=[sw for isl in self.calc_islands for sw in isl.switches],
                breakers=[brk for isl in self.calc_islands for brk in getattr(isl, "breakers", [])],
                shunt_compensators=[sc for isl in self.calc_islands for sc in isl.shunt_compensators],
                slack_nodes=[node for isl in self.calc_islands for node in isl.slack_nodes],
                v_gens=[gen for isl in self.calc_islands for gen in isl.v_gens],
            )

        # 1. 拓扑校验
        if self.target_island is not None:
            self.isl = self.target_island
        if self.isl is None:
            raise RuntimeError("无存活的拓扑岛，无法进行潮流计算")

        # 2. 提取节点基础信息
        self.node_list = sorted(self.isl.buses, key=lambda n: n.idx)
        self.N = len(self.node_list)
        self.node_pos = {}
        for i, node in enumerate(self.node_list):
            for member in getattr(node, "nodes", ()):
                self.node_pos[int(member.idx)] = i
            self.node_pos.setdefault(int(node.idx), i)

        # 3. 核心预处理流程
        self._build_y_matrix()
        self._prepare_node_parameters()
        self._prepare_zero_branches()

        # 变量索引
        self.theta_unknown = np.where(self.node_type != 'SLACK')[0]
        self.V_unknown = np.where(self.node_type == 'PQ')[0]

        self.theta_idx = {pos: i for i, pos in enumerate(self.theta_unknown)}
        self.V_idx = {pos: i for i, pos in enumerate(self.V_unknown)}

        self.n_theta = len(self.theta_unknown)
        self.n_V = len(self.V_unknown)
        self.base_phi_re = self.n_theta + self.n_V
        self.base_phi_im = self.base_phi_re + self.N_phi
        self.total_vars = self.base_phi_im + self.N_phi

        # 方程索引（精简计算）
        n_tree = sum(len(edges) for edges in self.comp_tree_edges)
        n_phi_fix = 2 * len(self.comp_nodes)

        self.total_eq = self.n_theta + self.n_V + 2 * n_tree + n_phi_fix

        # 4. 初始化状态向量
        self.x = np.zeros(self.total_vars, dtype=np.float64)
        self.x[self.n_theta:self.n_theta + self.n_V] = 1.0
        self._state_theta = np.empty(self.N, dtype=np.float64)
        self._state_voltage = np.empty(self.N, dtype=np.float64)
        self._state_vc = np.empty(self.N, dtype=np.complex128)
        self._empty_phi = np.array([], dtype=np.float64)
        self._load_p_work = np.zeros(self.N, dtype=np.float64)
        self._load_q_work = np.zeros(self.N, dtype=np.float64)
        self._load_dp_work = np.zeros(self.N, dtype=np.float64)
        self._load_dq_work = np.zeros(self.N, dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._residual_work = np.zeros(self.total_eq, dtype=np.float64)
        self._cache_static_arrays()
        self._cache_pq_decoupled_matrices()

        # 维度校验
        if self.verbose:
            print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")
        if self.total_vars != self.total_eq:
            warnings.warn(f"变量数({self.total_vars})与方程数({self.total_eq})不匹配！")

    def _prepare_zero_branches(self):
        """为零阻抗支路和闭合开关建立电流 phi 变量及电压相等约束。"""
        # 收集零阻抗边（合并循环）
        self.zero_edges = []
        for idx, zb in [(idx, zb) for idx, zb in enumerate(self.isl.zero_branches) if zb.is_alive]:
            a = self.node_pos[zb.i_node]
            b = self.node_pos[zb.j_node]
            if a == b:
                continue
            if self._is_redundant_slack_zero_edge(a, b):
                continue
            self.zero_edges.append((idx, 0, a, b))
        for idx, brk in [(idx, brk) for idx, brk in enumerate(getattr(self.isl, "breakers", [])) if brk.is_alive]:
            a = self.node_pos[brk.i_node]
            b = self.node_pos[brk.j_node]
            if a == b:
                continue
            if self._is_redundant_slack_zero_edge(a, b):
                continue
            self.zero_edges.append((idx, 2, a, b))
        for idx, sw in [(idx,sw) for idx, sw in enumerate(self.isl.switches) if sw.is_alive and sw.status == 1]:
            a = self.node_pos[sw.i_node]
            b = self.node_pos[sw.j_node]
            if a == b:
                continue
            if self._is_redundant_slack_zero_edge(a, b):
                continue
            self.zero_edges.append((idx, 1, a, b))

        self._prepare_zero_branch_components()

    def _prepare_zero_branch_components(self):
        """根据 self.zero_edges 建立零阻抗连通块、生成树约束和 phi 变量映射。"""
        if not self.zero_edges:
            self.comp_nodes = []
            self.comp_tree_edges = []
            self.phi_node = []
            self.phi_comp = []
            self.ref_phi_idx = []
            self.zero_branch_info = np.zeros((0, 6))
            self.N_phi = 0
            return

        # 只围绕零阻抗边端点做连通分量分析，避免在大系统里扫描全部普通节点。
        edge_by_node: Dict[int, List[int]] = {}
        for edge_idx, (_, _, a, b) in enumerate(self.zero_edges):
            edge_by_node.setdefault(a, []).append(edge_idx)
            edge_by_node.setdefault(b, []).append(edge_idx)

        raw_components = []
        visited_nodes = set()
        visited_edges = set()
        for start in edge_by_node:
            if start in visited_nodes:
                continue
            q = deque([start])
            visited_nodes.add(start)
            nodes = []
            edges_idx = []
            while q:
                u = q.popleft()
                nodes.append(u)
                for edge_idx in edge_by_node[u]:
                    if edge_idx not in visited_edges:
                        visited_edges.add(edge_idx)
                        edges_idx.append(edge_idx)
                    _, _, a, b = self.zero_edges[edge_idx]
                    v = b if a == u else a
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        q.append(v)
            raw_components.append((nodes, edges_idx))

        # 只对生成树边施加电压相等约束，环路边由 phi 电流变量承担。
        self.comp_nodes = []
        self.comp_tree_edges = []
        for nodes, edges_idx in raw_components:
            if len(nodes) <= 1 or not edges_idx:
                continue
            comp_local_idx = {node: idx for idx, node in enumerate(nodes)}
            local_edges = []
            orig_indices = []
            for orig_idx in edges_idx:
                _, _, a, b = self.zero_edges[orig_idx]
                local_edges.append((comp_local_idx[a], comp_local_idx[b]))
                orig_indices.append(orig_idx)
            tree_local_idx = find_spanning_tree_edges(local_edges, len(nodes))
            tree_edges = [orig_indices[i] for i in tree_local_idx]
            if tree_edges:
                self.comp_nodes.append(nodes)
                self.comp_tree_edges.append(tree_edges)

        # 构建phi变量映射（精简）
        self.phi_node = []
        self.phi_comp = []
        phi_index_by_node = {}
        for c, nodes in enumerate(self.comp_nodes):
            for node in nodes:
                phi_index_by_node[node] = len(self.phi_node)
                self.phi_node.append(node)
                self.phi_comp.append(c)
        self.N_phi = len(self.phi_node)

        # 参考phi索引（精简）
        self.ref_phi_idx = [phi_index_by_node[nodes[0]] for nodes in self.comp_nodes]

        # 零阻抗边phi映射（精简）
        self.zero_branch_info = np.zeros((len(self.zero_edges), 6))

        for idx, (index, type, a, b) in enumerate(self.zero_edges):
            if a not in phi_index_by_node or b not in phi_index_by_node:
                raise RuntimeError(f"节点{a}/{b}不在φ变量映射中")

            self.zero_branch_info[idx] = [index, type, a, b, phi_index_by_node[a], phi_index_by_node[b]]

    def _is_redundant_slack_zero_edge(self, a: int, b: int) -> bool:
        """Skip ideal ties between equal fixed-voltage slack nodes in Newton equations.

        Such ties are still part of topology, but their voltage equality is already
        enforced by the fixed slack phasors. Adding zero-current phi variables for
        them creates singular rows at flat start without changing the solved state.
        """
        if self.node_type.size == 0:
            return False
        if self.node_type[a] != 'SLACK' or self.node_type[b] != 'SLACK':
            return False
        if abs(self.V_spec[a] - self.V_spec[b]) > 1e-10:
            return False
        if abs(self.theta_spec[a] - self.theta_spec[b]) > 1e-10:
            return False
        return True

    def _build_y_matrix(self):
        """构建稀疏导纳矩阵，支路/变压器采用 MATPOWER tap/shift stamp。"""
        N = self.N
        rows, cols, data = [], [], []
        self.live_branches = [br for br in self.isl.branches if br.is_alive]
        self.live_transformers = [tr for tr in self.isl.transformers if tr.is_alive]
        self.branch_i = np.empty(len(self.live_branches), dtype=np.int32)
        self.branch_j = np.empty(len(self.live_branches), dtype=np.int32)
        self.branch_yff = np.empty(len(self.live_branches), dtype=np.complex128)
        self.branch_yft = np.empty(len(self.live_branches), dtype=np.complex128)
        self.branch_ytf = np.empty(len(self.live_branches), dtype=np.complex128)
        self.branch_ytt = np.empty(len(self.live_branches), dtype=np.complex128)
        self.transformer_i = np.empty(len(self.live_transformers), dtype=np.int32)
        self.transformer_j = np.empty(len(self.live_transformers), dtype=np.int32)
        self.transformer_yff = np.empty(len(self.live_transformers), dtype=np.complex128)
        self.transformer_yft = np.empty(len(self.live_transformers), dtype=np.complex128)
        self.transformer_ytf = np.empty(len(self.live_transformers), dtype=np.complex128)
        self.transformer_ytt = np.empty(len(self.live_transformers), dtype=np.complex128)

        # 普通支路
        if self.live_branches:
            self.branch_i = np.asarray([self.node_pos[br.i_node] for br in self.live_branches], dtype=np.int32)
            self.branch_j = np.asarray([self.node_pos[br.j_node] for br in self.live_branches], dtype=np.int32)
            self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt = matpower_branch_stamp_vectorized(
                [br.r for br in self.live_branches],
                [br.x for br in self.live_branches],
                [br.b for br in self.live_branches],
            )
            rows.extend(np.column_stack((self.branch_i, self.branch_i, self.branch_j, self.branch_j)).ravel())
            cols.extend(np.column_stack((self.branch_i, self.branch_j, self.branch_i, self.branch_j)).ravel())
            data.extend(
                np.column_stack((self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt)).ravel()
            )

        # 变压器
        if self.live_transformers:
            self.transformer_i = np.asarray([self.node_pos[tr.i_node] for tr in self.live_transformers], dtype=np.int32)
            self.transformer_j = np.asarray([self.node_pos[tr.j_node] for tr in self.live_transformers], dtype=np.int32)
            (
                self.transformer_yff,
                self.transformer_yft,
                self.transformer_ytf,
                self.transformer_ytt,
            ) = matpower_branch_stamp_vectorized(
                [tr.r for tr in self.live_transformers],
                [tr.x for tr in self.live_transformers],
                [tr.b for tr in self.live_transformers],
                [tr.tap for tr in self.live_transformers],
                [tr.shift for tr in self.live_transformers],
            )
            rows.extend(
                np.column_stack((self.transformer_i, self.transformer_i, self.transformer_j, self.transformer_j)).ravel()
            )
            cols.extend(
                np.column_stack((self.transformer_i, self.transformer_j, self.transformer_i, self.transformer_j)).ravel()
            )
            data.extend(
                np.column_stack(
                    (self.transformer_yff, self.transformer_yft, self.transformer_ytf, self.transformer_ytt)
                ).ravel()
            )

        # 并联补偿器（精简）
        for sc in [sc for sc in self.isl.shunt_compensators if sc.is_alive]:
            i = self.node_pos[sc.node]
            if sc.control_type in ['B', 'Z'] or sc.g_set != 0.0:
                y_sh = sc.g_set + 1j * sc.b_set
                if y_sh != 0:
                    rows.append(i)
                    cols.append(i)
                    data.append(y_sh)

        # 构建稀疏矩阵
        self.Y = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N, N))
        self.Y.sum_duplicates()

    def _prepare_node_parameters(self):
        """汇总节点类型、发电机设定值、并联补偿和电压相关负荷参数。"""
        N = self.N
        # 初始化节点参数（精简）
        self.node_type = np.full(N, 'PQ', dtype='U5')
        self.V_spec = np.full(N, np.nan, dtype=np.float64)
        self.theta_spec = np.zeros(N, dtype=np.float64)
        self.P_spec = np.zeros(N, dtype=np.float64)
        self.Q_spec = np.zeros(N, dtype=np.float64)
        self.load_info = []
        self.slack_node = -1

        # 处理发电机（精简逻辑）
        for gen in [gen for gen in self.isl.gens if gen.is_alive]:
            pos = self.node_pos[gen.node]

            if gen.control_type in ['V', 'SLACK', 'PH']:
                self.node_type[pos] = 'SLACK'
                self.slack_node = pos
                self.V_spec[pos] = gen.v_set if gen.v_set is not None else 1.0
                self.theta_spec[pos] = getattr(self.node_list[pos], 'angle', 0.0)
            elif gen.control_type == 'PV' and self.node_type[pos] != 'SLACK':
                self.node_type[pos] = 'PV'
                if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - gen.v_set) > 1e-6:
                    raise ValueError(f"节点{gen.node}多个PV发电机电压设定冲突")
                self.V_spec[pos] = gen.v_set
                self.P_spec[pos] += gen.p_set
            elif gen.control_type in ['PQ', 'P']:
                self.P_spec[pos] += gen.p_set
                self.Q_spec[pos] += gen.q_set

        # 处理并联补偿器（精简）
        for sc in [sc for sc in self.isl.shunt_compensators if sc.is_alive]:
            pos = self.node_pos[sc.node]
            if sc.control_type == 'Q':
                self.Q_spec[pos] += sc.q_set
            elif sc.control_type == 'V' and self.node_type[pos] != 'SLACK':
                self.node_type[pos] = 'PV'
                if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - sc.v_set) > 1e-6:
                    raise ValueError(f"节点{sc.node}电压设定冲突")
                self.V_spec[pos] = sc.v_set

        # 处理负荷（精简）
        for ld in self.isl.loads:
            if ld.is_alive:
                pbase = float(getattr(ld, "pbase", 1.0))
                qbase = float(getattr(ld, "qbase", 1.0))
                self.load_info.append((
                    self.node_pos[ld.node],
                    pbase * ld.pv0,
                    pbase * ld.pv1,
                    pbase * ld.pv2,
                    qbase * ld.qv0,
                    qbase * ld.qv1,
                    qbase * ld.qv2,
                ))

        # 确保存在平衡节点（精简）
        if self.slack_node == -1:
            pv_indices = np.where(self.node_type == 'PV')[0]
            if pv_indices.size > 0:
                self.slack_node = pv_indices[0]
                self.node_type[self.slack_node] = 'SLACK'

        if self.slack_node == -1:
            raise RuntimeError("电网中无平衡节点，无法进行潮流计算")

    def get_f(self, x: np.ndarray) -> np.ndarray:
        """计算残差向量 F"""
        theta, V, phi_re, phi_im = self._extract_state_vars(x)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        return self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)

    def _fill_residual(self, theta, V, phi_re, phi_im, dP, dQ) -> np.ndarray:
        """Fill the preallocated Newton residual using already computed balances."""
        F = self._residual_work
        F.fill(0.0)
        eq_idx = 0
        # 方程顺序必须与 get_jacobi() 保持一致：P、Q、零阻抗电压约束、phi参考。

        # 有功方程
        F[eq_idx:eq_idx + self.n_theta] = dP[self.theta_unknown]
        eq_idx += self.n_theta

        # 无功方程
        F[eq_idx:eq_idx + self.n_V] = dQ[self.V_unknown]
        eq_idx += self.n_V

        # 零阻抗电压约束（精简循环）
        cos_theta, sin_theta = self._cache['cos_theta'], self._cache['sin_theta']
        for c in range(len(self.comp_nodes)):
            for edge_idx in self.comp_tree_edges[c]:
                _index, _type, a, b = self.zero_edges[edge_idx]
                F[eq_idx] = V[a] * cos_theta[a] - V[b] * cos_theta[b]
                F[eq_idx + 1] = V[a] * sin_theta[a] - V[b] * sin_theta[b]
                eq_idx += 2

        # phi参考固定（精简）
        if self.ref_phi_idx:
            ref_phi_arr = np.asarray(self.ref_phi_idx, dtype=np.int32)
            F[eq_idx:eq_idx + 2 * ref_phi_arr.size:2] = phi_re[ref_phi_arr]
            F[eq_idx + 1:eq_idx + 2 * ref_phi_arr.size:2] = phi_im[ref_phi_arr]

        return F

    def _get_jacobi_loop(self, x: np.ndarray) -> csr_matrix:
        """计算雅可比矩阵的逐行备用实现，供无 scipy 稀疏优化时使用。"""
        # 提取状态变量
        theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        cos_theta, sin_theta = self._cache['cos_theta'], self._cache['sin_theta']

        # 初始化雅可比矩阵数据
        rows, cols, data = [], [], []
        N = self.N

        # 标准雅可比计算（精简核心逻辑）
        for i in range(N):
            Vi_val, thetai_val = V[i], theta[i]

            # 计算Pi, Qi
            Yii = self.Y_diag[i]
            Pi = Vi_val ** 2 * Yii.real
            Qi = -Vi_val ** 2 * Yii.imag

            # 向量化计算功率项
            j_valid = self.Y_offdiag_indices[i]
            y_valid = self.Y_offdiag_data[i]
            if len(j_valid) > 0:

                Gij = y_valid.real
                Bij = y_valid.imag
                Vj = V[j_valid]
                delta = thetai_val - theta[j_valid]

                cos_delta = np.cos(delta)
                sin_delta = np.sin(delta)

                Pi += np.sum(Vi_val * Vj * (Gij * cos_delta + Bij * sin_delta))
                Qi += np.sum(Vi_val * Vj * (Gij * sin_delta - Bij * cos_delta))

            # 对角线元素
            Gii = Yii.real
            Bii = Yii.imag

            Hii = -Qi - Vi_val ** 2 * Bii
            Nii = Pi + Vi_val ** 2 * Gii
            Mii = Pi - Vi_val ** 2 * Gii
            Lii = Qi - Vi_val ** 2 * Bii

            # 有功方程偏导（精简判断）
            if self.node_type[i] != 'SLACK':
                eq_i = self.theta_idx[i]
                rows.append(eq_i)
                cols.append(self.theta_idx[i])
                data.append(Hii)

                if self.node_type[i] == 'PQ':
                    rows.append(eq_i)
                    cols.append(self.n_theta + self.V_idx[i])
                    data.append(Nii)

            # 无功方程偏导（精简）
            if self.node_type[i] == 'PQ':
                eq_i = self.n_theta + self.V_idx[i]
                rows.append(eq_i)
                cols.append(self.theta_idx[i])
                data.append(Mii)
                rows.append(eq_i)
                cols.append(self.n_theta + self.V_idx[i])
                data.append(Lii)

            # 非对角线元素（精简）
            if len(j_valid) > 0:
                Gij = y_valid.real
                Bij = y_valid.imag
                Vj = V[j_valid]
                delta = thetai_val - theta[j_valid]

                cos_delta = np.cos(delta)
                sin_delta = np.sin(delta)

                # 计算偏导数
                Hij = Vi_val * Vj * (Gij * sin_delta - Bij * cos_delta)
                Nij = Vi_val * Vj * (Gij * cos_delta + Bij * sin_delta)
                Mij = -Nij
                Lij = Hij

                # 批量处理
                for idx, j in enumerate(j_valid):
                    # 有功方程
                    if self.node_type[i] != 'SLACK':
                        eq_i = self.theta_idx[i]
                        if self.node_type[j] != 'SLACK':
                            rows.append(eq_i)
                            cols.append(self.theta_idx[j])
                            data.append(Hij[idx])
                        if self.node_type[j] == 'PQ':
                            rows.append(eq_i)
                            cols.append(self.n_theta + self.V_idx[j])
                            data.append(Nij[idx])

                    # 无功方程
                    if self.node_type[i] == 'PQ':
                        eq_i = self.n_theta + self.V_idx[i]
                        if self.node_type[j] != 'SLACK':
                            rows.append(eq_i)
                            cols.append(self.theta_idx[j])
                            data.append(Mij[idx])
                        if self.node_type[j] == 'PQ':
                            rows.append(eq_i)
                            cols.append(self.n_theta + self.V_idx[j])
                            data.append(Lij[idx])


        # 3. 计算负荷功率
        for idx, pos in enumerate(self.load_pos):
            vm = V[pos]
            if self.node_type[pos] == 'PQ':
                rows.append(self.theta_idx[pos])
                cols.append(self.n_theta + self.V_idx[pos])
                data.append(self.load_pv1[idx] + 2.0 * vm * self.load_pv2[idx])

                rows.append(self.n_theta + self.V_idx[pos])
                cols.append(self.n_theta + self.V_idx[pos])
                data.append(self.load_qv1[idx] + 2.0 * vm * self.load_qv2[idx])


        # 零阻抗支路雅可比（精简）
        if self.N_phi > 0 and self.zero_a.size:
            I_re = phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]
            I_im = phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]
            Va = V[self.zero_a]
            Vb = V[self.zero_b]
            cos_a = cos_theta[self.zero_a]
            sin_a = sin_theta[self.zero_a]
            cos_b = cos_theta[self.zero_b]
            sin_b = sin_theta[self.zero_b]

            # 批量处理
            for idx in range(len(self.zero_a)):
                a, b, phi_a, phi_b = self.zero_a[idx], self.zero_b[idx], self.zero_phi_a[idx], self.zero_phi_b[idx]

                # 节点a的偏导
                if self.node_type[a] != 'SLACK':
                    eq_a = self.theta_idx[a]
                    rows.append(eq_a)
                    cols.append(self.theta_idx[a])
                    data.append(Va[idx] * (-sin_a[idx] * I_re[idx] + cos_a[idx] * I_im[idx]))
                    if self.node_type[a] == 'PQ':
                        rows.append(eq_a)
                        cols.append(self.n_theta + self.V_idx[a])
                        data.append(cos_a[idx] * I_re[idx] + sin_a[idx] * I_im[idx])
                    rows.extend([eq_a] * 4)
                    cols.extend([self.base_phi_re + phi_a, self.base_phi_im + phi_a, self.base_phi_re + phi_b,
                                 self.base_phi_im + phi_b])
                    data.extend(
                        [Va[idx] * cos_a[idx], Va[idx] * sin_a[idx], -Va[idx] * cos_a[idx], -Va[idx] * sin_a[idx]])

                if self.node_type[a] == 'PQ':
                    eq_a = self.n_theta + self.V_idx[a]
                    rows.extend([eq_a, eq_a])
                    cols.extend([self.theta_idx[a], self.n_theta + self.V_idx[a]])
                    data.extend([
                        Va[idx] * (cos_a[idx] * I_re[idx] + sin_a[idx] * I_im[idx]),
                        sin_a[idx] * I_re[idx] - cos_a[idx] * I_im[idx],
                    ])
                    rows.extend([eq_a] * 4)
                    cols.extend([self.base_phi_re + phi_a, self.base_phi_im + phi_a, self.base_phi_re + phi_b,
                                 self.base_phi_im + phi_b])
                    data.extend(
                        [Va[idx] * sin_a[idx], -Va[idx] * cos_a[idx], -Va[idx] * sin_a[idx], Va[idx] * cos_a[idx]])

                # 节点b的偏导
                if self.node_type[b] != 'SLACK':
                    eq_b = self.theta_idx[b]
                    rows.append(eq_b)
                    cols.append(self.theta_idx[b])
                    data.append(Vb[idx] * (sin_b[idx] * I_re[idx] - cos_b[idx] * I_im[idx]))
                    if self.node_type[b] == 'PQ':
                        rows.append(eq_b)
                        cols.append(self.n_theta + self.V_idx[b])
                        data.append(-cos_b[idx] * I_re[idx] - sin_b[idx] * I_im[idx])
                    rows.extend([eq_b] * 4)
                    cols.extend([self.base_phi_re + phi_a, self.base_phi_im + phi_a, self.base_phi_re + phi_b,
                                 self.base_phi_im + phi_b])
                    data.extend(
                        [-Vb[idx] * cos_b[idx], -Vb[idx] * sin_b[idx], Vb[idx] * cos_b[idx], Vb[idx] * sin_b[idx]])

                if self.node_type[b] == 'PQ':
                    eq_b = self.n_theta + self.V_idx[b]
                    rows.extend([eq_b, eq_b])
                    cols.extend([self.theta_idx[b], self.n_theta + self.V_idx[b]])
                    data.extend([
                        -Vb[idx] * (cos_b[idx] * I_re[idx] + sin_b[idx] * I_im[idx]),
                        -sin_b[idx] * I_re[idx] + cos_b[idx] * I_im[idx],
                    ])
                    rows.extend([eq_b] * 4)
                    cols.extend([self.base_phi_re + phi_a, self.base_phi_im + phi_a, self.base_phi_re + phi_b,
                                 self.base_phi_im + phi_b])
                    data.extend(
                        [-Vb[idx] * sin_b[idx], Vb[idx] * cos_b[idx], Vb[idx] * sin_b[idx], -Vb[idx] * cos_b[idx]])

        # 零阻抗约束雅可比（精简）
        eq_idx = self.n_theta + self.n_V
        for c in range(len(self.comp_nodes)):
            for edge_idx in self.comp_tree_edges[c]:
                _index, _type, a, b = self.zero_edges[edge_idx]

                # 实部约束
                if self.node_type[a] != 'SLACK':
                    rows.append(eq_idx)
                    cols.append(self.theta_idx[a])
                    data.append(-V[a] * sin_theta[a])
                if self.node_type[a] == 'PQ':
                    rows.append(eq_idx)
                    cols.append(self.n_theta + self.V_idx[a])
                    data.append(cos_theta[a])
                if self.node_type[b] != 'SLACK':
                    rows.append(eq_idx)
                    cols.append(self.theta_idx[b])
                    data.append(V[b] * sin_theta[b])
                if self.node_type[b] == 'PQ':
                    rows.append(eq_idx)
                    cols.append(self.n_theta + self.V_idx[b])
                    data.append(-cos_theta[b])
                eq_idx += 1

                # 虚部约束
                if self.node_type[a] != 'SLACK':
                    rows.append(eq_idx)
                    cols.append(self.theta_idx[a])
                    data.append(V[a] * cos_theta[a])
                if self.node_type[a] == 'PQ':
                    rows.append(eq_idx)
                    cols.append(self.n_theta + self.V_idx[a])
                    data.append(sin_theta[a])
                if self.node_type[b] != 'SLACK':
                    rows.append(eq_idx)
                    cols.append(self.theta_idx[b])
                    data.append(-V[b] * cos_theta[b])
                if self.node_type[b] == 'PQ':
                    rows.append(eq_idx)
                    cols.append(self.n_theta + self.V_idx[b])
                    data.append(-sin_theta[b])
                eq_idx += 1

        # phi参考约束雅可比（精简）
        for c in range(len(self.comp_nodes)):
            idx_phi = self.ref_phi_idx[c]
            rows.append(eq_idx + 2 * c)
            cols.append(self.base_phi_re + idx_phi)
            data.append(1.0)
            rows.append(eq_idx + 2 * c + 1)
            cols.append(self.base_phi_im + idx_phi)
            data.append(1.0)

        return build_jacobian_matrix(rows, cols, data, (self.total_eq, self.total_vars))

    def _calc_load_power_derivatives(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dP = self._load_dp_work
        dQ = self._load_dq_work
        dP.fill(0.0)
        dQ.fill(0.0)
        if self.load_pos.size:
            vm = self._load_vm_work
            values = self._load_value_work
            np.take(V, self.load_pos, out=vm)
            np.multiply(vm, self.load_pv2, out=values)
            values *= 2.0
            values += self.load_pv1
            np.add.at(dP, self.load_pos, values)

            np.multiply(vm, self.load_qv2, out=values)
            values *= 2.0
            values += self.load_qv1
            np.add.at(dQ, self.load_pos, values)
        return dP, dQ

    def _get_standard_jacobi_sparse(self, V: np.ndarray, Sbus=None):
        """标准P/Q方程对theta/V变量的稀疏雅可比，采用MATPOWER矩阵化公式。"""
        if self.Y_jac_rows.size:
            return self._get_standard_jacobi_direct(V, Sbus=Sbus)

        Vc = self._cache['Vc']
        Ibus = self.Y.dot(Vc)
        Vnorm = np.divide(Vc, V, out=np.zeros_like(Vc), where=V != 0)

        # 等价于 MATPOWER dSbus_dV 公式，但直接缩放 Y 的列/行，避免构造
        # diag(V)、diag(V/|V|) 后再做稀疏矩阵乘法。
        dS_dVm = self.Y.multiply(Vnorm).conjugate().multiply(Vc[:, None]).tocsr()
        dS_dVm.setdiag(dS_dVm.diagonal() + np.conj(Ibus) * Vnorm)

        dS_dVa = self.Y.multiply(Vc).conjugate().multiply((-1j * Vc)[:, None]).tocsr()
        dS_dVa.setdiag(dS_dVa.diagonal() + 1j * Vc * np.conj(Ibus))

        dPload_dV, dQload_dV = self._calc_load_power_derivatives(V)
        pvpq = self.theta_unknown
        pq = self.V_unknown

        J11 = dS_dVa.real[pvpq, :][:, pvpq]
        J12 = dS_dVm.real[pvpq, :][:, pq]
        J21 = dS_dVa.imag[pq, :][:, pvpq]
        J22 = dS_dVm.imag[pq, :][:, pq]

        if pq.size:
            J12 = J12 + coo_matrix((dPload_dV[pq], (self.pq_theta_rows, self.pq_v_cols)), shape=(self.n_theta, self.n_V)).tocsr()
            J22 = J22 + diags(dQload_dV[pq], 0, shape=(pq.size, pq.size), format='csr')

        return vstack((hstack((J11, J12), format='csr'), hstack((J21, J22), format='csr')), format='csr')

    def _fill_standard_jacobian_data(self, V: np.ndarray, Sbus=None):
        """Refresh the standard P/Q Jacobian raw data in the cached coordinate order."""
        theta = self._cache['theta']
        Vc = self._cache['Vc']
        if Sbus is not None:
            pass
        elif self._power_x_obj is self._state_x_obj and self._last_Sbus is not None:
            Sbus = self._last_Sbus
        else:
            Ibus = self.Y.dot(Vc)
            Sbus = Vc * np.conj(Ibus)
            self._power_x_obj = self._state_x_obj
            self._last_Ibus = Ibus
            self._last_Sbus = Sbus
        P = Sbus.real
        Q = Sbus.imag

        i = self.Y_jac_rows
        j = self.Y_jac_cols
        G = self.Y_jac_g
        B = self.Y_jac_b
        Vi = V[i]
        Vj = V[j]
        delta = self._jac_delta
        cos_delta = self._jac_cos_delta
        sin_delta = self._jac_sin_delta
        vivj = self._jac_vivj
        common_p = self._jac_common_p
        common_q = self._jac_common_q
        tmp = self._jac_tmp

        np.subtract(theta[i], theta[j], out=delta)
        np.cos(delta, out=cos_delta)
        np.sin(delta, out=sin_delta)
        np.multiply(Vi, Vj, out=vivj)

        np.multiply(G, sin_delta, out=common_q)
        common_q -= B * cos_delta
        np.multiply(G, cos_delta, out=common_p)
        common_p += B * sin_delta

        diag_idx = self.Y_jac_diag_idx
        if diag_idx.size:
            diag_nodes = self.Y_jac_diag_nodes
            Gii = self.Y_jac_diag_g
            Bii = self.Y_jac_diag_b
            Vdiag = V[diag_nodes]

        data = self.standard_jac_data
        np.multiply(vivj, common_q, out=tmp)
        if diag_idx.size:
            tmp[diag_idx] = -Q[diag_nodes] - Bii * Vdiag * Vdiag
        data[self.std_jac_p_theta_slice] = tmp[self.std_jac_p_theta_idx]

        np.multiply(vivj, common_p, out=tmp)
        tmp *= -1.0
        if diag_idx.size:
            tmp[diag_idx] = P[diag_nodes] - Gii * Vdiag * Vdiag
        data[self.std_jac_q_theta_slice] = tmp[self.std_jac_q_theta_idx]

        np.multiply(Vi, common_p, out=tmp)
        if diag_idx.size:
            tmp[diag_idx] = np.divide(
                P[diag_nodes],
                Vdiag,
                out=np.zeros_like(Vdiag),
                where=np.abs(Vdiag) > 1e-12,
            ) + Gii * Vdiag
        data[self.std_jac_p_vm_slice] = tmp[self.std_jac_p_vm_idx]

        np.multiply(Vi, common_q, out=tmp)
        if diag_idx.size:
            tmp[diag_idx] = np.divide(
                Q[diag_nodes],
                Vdiag,
                out=np.zeros_like(Vdiag),
                where=np.abs(Vdiag) > 1e-12,
            ) - Bii * Vdiag
        data[self.std_jac_q_vm_slice] = tmp[self.std_jac_q_vm_idx]

        if self.V_unknown.size:
            dPload_dV, dQload_dV = self._calc_load_power_derivatives(V)
            if self.std_jac_load_nodes.size:
                data[self.std_jac_load_p_pos] += dPload_dV[self.std_jac_load_nodes]
                data[self.std_jac_load_q_pos] += dQload_dV[self.std_jac_load_nodes]
            if self.std_jac_load_p_slice.stop > self.std_jac_load_p_slice.start:
                data[self.std_jac_load_p_slice] = dPload_dV[self.std_jac_load_extra_nodes]
                data[self.std_jac_load_q_slice] = dQload_dV[self.std_jac_load_extra_nodes]

        return data

    def _get_standard_jacobi_direct(self, V: np.ndarray, Sbus=None):
        """Build the standard P/Q Jacobian directly from Y nonzeros."""
        data = self._fill_standard_jacobian_data(V, Sbus=Sbus)
        if self.standard_jac_csr_order.size:
            self.standard_jac_csr_data[:] = data[self.standard_jac_csr_order]
            return csr_matrix(
                (self.standard_jac_csr_data, self.standard_jac_csr_indices, self.standard_jac_csr_indptr),
                shape=(self.n_theta + self.n_V, self.n_theta + self.n_V),
                copy=False,
            )
        return coo_matrix(
            (data, (self.standard_jac_rows, self.standard_jac_cols)),
            shape=(self.n_theta + self.n_V, self.n_theta + self.n_V),
        ).tocsr()

    def _refresh_zero_edge_jacobian_work(self, V, phi_re, phi_im, cos_theta, sin_theta):
        """Refresh zero-edge workspace arrays shared by top-left and top-right blocks."""
        if not self.zero_a.size:
            return
        I_re = self._zero_I_re
        I_im = self._zero_I_im
        tmp = self._zero_tmp

        np.take(phi_re, self.zero_phi_a, out=I_re)
        np.take(phi_re, self.zero_phi_b, out=tmp)
        I_re -= tmp
        np.take(phi_im, self.zero_phi_a, out=I_im)
        np.take(phi_im, self.zero_phi_b, out=tmp)
        I_im -= tmp

        Va = self._zero_Va
        Vb = self._zero_Vb
        cos_a = self._zero_cos_a
        sin_a = self._zero_sin_a
        cos_b = self._zero_cos_b
        sin_b = self._zero_sin_b
        np.take(V, self.zero_a, out=Va)
        np.take(V, self.zero_b, out=Vb)
        np.take(cos_theta, self.zero_a, out=cos_a)
        np.take(sin_theta, self.zero_a, out=sin_a)
        np.take(cos_theta, self.zero_b, out=cos_b)
        np.take(sin_theta, self.zero_b, out=sin_b)

        a_pv = self._zero_a_pv
        a_pt = self._zero_a_pt
        b_pv = self._zero_b_pv
        b_pt = self._zero_b_pt
        np.multiply(cos_a, I_re, out=a_pv)
        np.multiply(sin_a, I_im, out=tmp)
        a_pv += tmp
        np.multiply(sin_a, I_re, out=a_pt)
        a_pt *= -1.0
        np.multiply(cos_a, I_im, out=tmp)
        a_pt += tmp
        np.multiply(cos_b, I_re, out=b_pv)
        np.multiply(sin_b, I_im, out=tmp)
        b_pv += tmp
        np.multiply(sin_b, I_re, out=b_pt)
        np.multiply(cos_b, I_im, out=tmp)
        b_pt -= tmp

        np.multiply(Va, cos_a, out=self._zero_va_cos)
        np.multiply(Va, sin_a, out=self._zero_va_sin)
        np.multiply(Vb, cos_b, out=self._zero_vb_cos)
        np.multiply(Vb, sin_b, out=self._zero_vb_sin)

    def _fill_zero_top_jacobian_data(self, V, phi_re, phi_im, cos_theta, sin_theta):
        if not self.zero_a.size:
            return
        self._refresh_zero_edge_jacobian_work(V, phi_re, phi_im, cos_theta, sin_theta)
        Va = self._zero_Va
        Vb = self._zero_Vb
        a_pv = self._zero_a_pv
        a_pt = self._zero_a_pt
        b_pv = self._zero_b_pv
        b_pt = self._zero_b_pt

        if self.zero_top_left_data.size:
            data = self.zero_top_left_data
            pos = self.zero_top_left_pos_by_kind
            edge = self.zero_top_left_edge_by_kind
            if pos[0].size:
                data[pos[0]] = Va[edge[0]] * a_pt[edge[0]]
            if pos[1].size:
                data[pos[1]] = a_pv[edge[1]]
            if pos[2].size:
                data[pos[2]] = Va[edge[2]] * a_pv[edge[2]]
            if pos[3].size:
                data[pos[3]] = -a_pt[edge[3]]
            if pos[4].size:
                data[pos[4]] = Vb[edge[4]] * b_pt[edge[4]]
            if pos[5].size:
                data[pos[5]] = -b_pv[edge[5]]
            if pos[6].size:
                data[pos[6]] = -Vb[edge[6]] * b_pv[edge[6]]
            if pos[7].size:
                data[pos[7]] = -b_pt[edge[7]]
        if self.zero_top_right_data.size:
            data = self.zero_top_right_data
            pos = self.zero_top_right_pos_by_kind
            edge = self.zero_top_right_edge_by_kind
            Va_cos = self._zero_va_cos
            Va_sin = self._zero_va_sin
            Vb_cos = self._zero_vb_cos
            Vb_sin = self._zero_vb_sin
            if pos[0].size:
                data[pos[0]] = Va_cos[edge[0]]
            if pos[1].size:
                data[pos[1]] = Va_sin[edge[1]]
            if pos[2].size:
                data[pos[2]] = -Va_cos[edge[2]]
            if pos[3].size:
                data[pos[3]] = -Va_sin[edge[3]]
            if pos[4].size:
                data[pos[4]] = Va_sin[edge[4]]
            if pos[5].size:
                data[pos[5]] = -Va_cos[edge[5]]
            if pos[6].size:
                data[pos[6]] = -Va_sin[edge[6]]
            if pos[7].size:
                data[pos[7]] = Va_cos[edge[7]]
            if pos[8].size:
                data[pos[8]] = -Vb_cos[edge[8]]
            if pos[9].size:
                data[pos[9]] = -Vb_sin[edge[9]]
            if pos[10].size:
                data[pos[10]] = Vb_cos[edge[10]]
            if pos[11].size:
                data[pos[11]] = Vb_sin[edge[11]]
            if pos[12].size:
                data[pos[12]] = -Vb_sin[edge[12]]
            if pos[13].size:
                data[pos[13]] = Vb_cos[edge[13]]
            if pos[14].size:
                data[pos[14]] = Vb_sin[edge[14]]
            if pos[15].size:
                data[pos[15]] = -Vb_cos[edge[15]]

    def _fill_zero_bottom_jacobian_data(self, V, cos_theta, sin_theta):
        if not self.zero_bottom_left_data.size:
            return
        data = self.zero_bottom_left_data
        pos = self.zero_bottom_left_pos_by_kind
        node = self.zero_bottom_left_node_by_kind
        if pos[0].size:
            data[pos[0]] = -V[node[0]] * sin_theta[node[0]]
        if pos[1].size:
            data[pos[1]] = cos_theta[node[1]]
        if pos[2].size:
            data[pos[2]] = V[node[2]] * sin_theta[node[2]]
        if pos[3].size:
            data[pos[3]] = -cos_theta[node[3]]
        if pos[4].size:
            data[pos[4]] = V[node[4]] * cos_theta[node[4]]
        if pos[5].size:
            data[pos[5]] = sin_theta[node[5]]
        if pos[6].size:
            data[pos[6]] = -V[node[6]] * cos_theta[node[6]]
        if pos[7].size:
            data[pos[7]] = -sin_theta[node[7]]

    def get_jacobi(self, x: np.ndarray) -> csr_matrix:
        """计算雅可比矩阵。标准AC部分使用稀疏矩阵批量公式，零阻抗扩展按块追加。"""
        if self._state_x_obj is x:
            theta = self._cache['theta']
            V = self._cache['V']
            phi_re = x[self.base_phi_re:self.base_phi_re + self.N_phi] if self.N_phi > 0 else self._empty_phi
            phi_im = x[self.base_phi_im:self.base_phi_im + self.N_phi] if self.N_phi > 0 else self._empty_phi
        else:
            theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        return self._get_jacobi_from_cached_state(theta, V, phi_re, phi_im)

    def _get_jacobi_from_precomputed_pattern(self, V, phi_re, phi_im, cos_theta, sin_theta, Sbus=None):
        if self.full_jac_raw_data.size == 0:
            return None

        raw = self.full_jac_raw_data
        raw[self.full_jac_standard_slice] = self._fill_standard_jacobian_data(V, Sbus=Sbus)
        if self.N_phi:
            self._fill_zero_top_jacobian_data(V, phi_re, phi_im, cos_theta, sin_theta)
            self._fill_zero_bottom_jacobian_data(V, cos_theta, sin_theta)
            raw[self.full_jac_zero_top_left_slice] = self.zero_top_left_data
            raw[self.full_jac_zero_top_right_slice] = self.zero_top_right_data
            raw[self.full_jac_zero_bottom_left_slice] = self.zero_bottom_left_data
            raw[self.full_jac_zero_bottom_right_slice] = self.zero_bottom_right_data

        self.full_jac_csr_data.fill(0.0)
        np.add.at(self.full_jac_csr_data, self.full_jac_raw_to_csr_pos, raw)
        return csr_matrix(
            (self.full_jac_csr_data, self.full_jac_csr_indices, self.full_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )

    def _get_jacobi_from_cached_state(self, theta, V, phi_re, phi_im, Sbus=None) -> csr_matrix:
        """Build Jacobian using state arrays already extracted for the current Newton step."""
        cos_theta, sin_theta = self._cache['cos_theta'], self._cache['sin_theta']

        precomputed = self._get_jacobi_from_precomputed_pattern(V, phi_re, phi_im, cos_theta, sin_theta, Sbus=Sbus)
        if precomputed is not None:
            return precomputed

        J_standard = self._get_standard_jacobi_sparse(V, Sbus=Sbus)
        if self.N_phi == 0:
            return J_standard

        # 标准 P/Q 子块保持 CSR，不再转成 Python list 后重建整矩阵。
        # 零阻抗支路只生成三个小稀疏块，再与标准块拼接。
        std_eq = self.n_theta + self.n_V
        std_vars = std_eq
        zero_eq = self.total_eq - std_eq
        phi_vars = 2 * self.N_phi

        self._fill_zero_top_jacobian_data(V, phi_re, phi_im, cos_theta, sin_theta)
        if self.zero_top_left_rows.size:
            J_standard = J_standard + coo_matrix(
                (self.zero_top_left_data, (self.zero_top_left_rows, self.zero_top_left_cols)),
                shape=(std_eq, std_vars),
            ).tocsr()
        J_phi_top = coo_matrix(
            (self.zero_top_right_data, (self.zero_top_right_rows, self.zero_top_right_cols)),
            shape=(std_eq, phi_vars),
        ).tocsr()

        self._fill_zero_bottom_jacobian_data(V, cos_theta, sin_theta)
        J_zero_left = coo_matrix(
            (self.zero_bottom_left_data, (self.zero_bottom_left_rows, self.zero_bottom_left_cols)),
            shape=(zero_eq, std_vars),
        ).tocsr()
        J_zero_right = coo_matrix(
            (self.zero_bottom_right_data, (self.zero_bottom_right_rows, self.zero_bottom_right_cols)),
            shape=(zero_eq, phi_vars),
        ).tocsr()
        return vstack(
            (
                hstack((J_standard, J_phi_top), format='csr'),
                hstack((J_zero_left, J_zero_right), format='csr'),
            ),
            format='csr',
        )

    def _build_newton_system(self, x: np.ndarray):
        """Compute residual and Jacobian together for one Newton iteration."""
        theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        F = self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)
        J = self._get_jacobi_from_cached_state(theta, V, phi_re, phi_im, Sbus=self._last_Sbus)
        return F, J

    # --------------------------------------------------------------------------
    # 迭代求解
    # --------------------------------------------------------------------------
    def run(self, result_mode=None) -> int:
        """执行所选潮流算法。"""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
        if self.algorithm == "pq" and self.N_phi == 0 and self.pq_Bp is not None and self.pq_Bpp is not None:
            return self._run_pq_decoupled()
        if self.algorithm == "pq":
            if self.verbose:
                print("PQ分解法不支持当前零阻抗扩展模型，自动回退到N-R法")
            return self._run_newton_raphson(used_label="pq->nr")
        return self._run_newton_raphson()

    def _run_newton_raphson(self, used_label: str = "nr") -> int:
        """执行牛顿-拉夫逊迭代求解"""
        self.used_algorithm = used_label
        self.converged = False
        self.iterations = 0
        x = self.x.copy()

        for it in range(self.max_iter):
            self.iterations += 1

            # 单轮内合并残差和 Jacobian 数值块计算，复用 YV/Sbus 等中间量。
            F, J = self._build_newton_system(x)
            self.normF = np.linalg.norm(F, np.inf)
            if self.verbose:
                print(f"Iter {it + 1}: |F| = {self.normF:.2e}")

            # 收敛判断
            if self.normF < self.tol:
                if self.verbose:
                    print(f"收敛于第 {it + 1} 次迭代")
                self.converged = True
                self.x = x
                self._write_back()
                return 0

            delta = solve_sparse_system(J, F, self.linear_solver)
            # 方程定义为 F(x)=0，这里使用 x_new = x - J^{-1}F。
            x -= delta

        # 未收敛
        if self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未收敛")
        self.x = x
        self._write_back()
        return -1

    def _run_pq_decoupled(self) -> int:
        """执行快速PQ分解潮流。"""
        self.used_algorithm = "pq"
        self.converged = False
        self.iterations = 0
        x0 = self.x.copy()
        x = x0.copy()
        v_slice = slice(self.n_theta, self.n_theta + self.n_V)

        for it in range(self.max_iter):
            self.iterations += 1

            theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
            dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
            p_mis = dP[self.theta_unknown]
            q_mis = dQ[self.V_unknown]
            self.normF = max(
                float(np.linalg.norm(p_mis, np.inf)) if p_mis.size else 0.0,
                float(np.linalg.norm(q_mis, np.inf)) if q_mis.size else 0.0,
            )
            if self.verbose:
                print(f"Iter {it + 1}: |F| = {self.normF:.2e}")

            if self.normF < self.tol:
                if self.verbose:
                    print(f"PQ分解法收敛于第 {it + 1} 次迭代")
                self.converged = True
                self.x = x
                self._write_back()
                return 0

            if self.n_theta:
                rhs_p = np.divide(
                    p_mis,
                    V[self.theta_unknown],
                    out=np.zeros_like(p_mis),
                    where=V[self.theta_unknown] > self.min_voltage,
                )
                dtheta = self.pq_Bp_factor.solve(rhs_p)
                x[:self.n_theta] -= dtheta

            theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
            _, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
            q_mis = dQ[self.V_unknown]
            if self.n_V:
                rhs_q = np.divide(
                    q_mis,
                    V[self.V_unknown],
                    out=np.zeros_like(q_mis),
                    where=V[self.V_unknown] > self.min_voltage,
                )
                dV = self.pq_Bpp_factor.solve(rhs_q)
                x[v_slice] -= dV
                np.maximum(x[v_slice], self.min_voltage, out=x[v_slice])

        if self.verbose:
            print(f"PQ分解法达到最大迭代次数 {self.max_iter}，未收敛，改用N-R法")
        self.x = x0
        return self._run_newton_raphson(used_label="pq->nr")

    def _summary_node_ids(self):
        if self.ppc_node_idx.size == self.N:
            return self.ppc_node_idx.copy()
        if self.node_list:
            return np.asarray([int(getattr(node, "idx", pos)) for pos, node in enumerate(self.node_list)], dtype=np.int64)
        return np.arange(self.N, dtype=np.int64)

    def _write_summary_result(self):
        theta, V, _phi_re, _phi_im = self._extract_state_vars(self.x)
        self.result = {
            "node_id": self._summary_node_ids(),
            "voltage": V.copy(),
            "angle": theta.copy(),
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
                "used_algorithm": self.used_algorithm,
            },
        }
        self.lf_result = None

    def _write_back(self):
        """结果回填；数值计算批量完成，Python 循环只负责对象属性赋值。"""
        if self.result_mode == "none":
            self.result = {}
            self.lf_result = None
            return
        if self.result_mode == "summary":
            self._write_summary_result()
            return

        if self.array_mode:
            self._write_back_ppc()
            return

        theta, V, phi_re, phi_im = self._extract_state_vars(self.x)
        Vc = self._cache['Vc']

        P_load, Q_load = self._cache['P_load'], self._cache['Q_load']
        if P_load is None or Q_load is None:
            P_load, Q_load = self._calc_load_power(V)

        # 计算节点注入功率
        I_y = self.Y.dot(Vc)
        S_y = Vc * np.conj(I_y)

        P_zero = np.zeros(self.N, dtype=np.float64)
        Q_zero = np.zeros(self.N, dtype=np.float64)
        if self.N_phi > 0 and self.zero_a.size:
            I_ab = (phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]) + 1j * (
                phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]
            )
            Sa = Vc[self.zero_a] * np.conj(I_ab)
            Sb = Vc[self.zero_b] * np.conj(-I_ab)
            np.add.at(P_zero, self.zero_a, Sa.real)
            np.add.at(Q_zero, self.zero_a, Sa.imag)
            np.add.at(P_zero, self.zero_b, Sb.real)
            np.add.at(Q_zero, self.zero_b, Sb.imag)
        else:
            I_ab = np.array([], dtype=np.complex128)

        P_gen = S_y.real + P_zero + P_load
        Q_gen = S_y.imag + Q_zero + Q_load

        for node in self.isl.buses:
            pos = self.node_pos[node.idx]
            node.voltage = V[pos]
            node.angle = theta[pos]
            for member in getattr(node, "nodes", ()):
                member.voltage = node.voltage
                member.angle = node.angle

        if self.live_gens:
            gen_p = self.gen_share * P_gen[self.gen_pos]
            gen_q = self.gen_share * Q_gen[self.gen_pos]
            gen_v = V[self.gen_pos]
            gen_current = np.divide(
                np.hypot(gen_p, gen_q),
                gen_v,
                out=np.zeros_like(gen_p),
                where=gen_v > self.min_voltage,
            )
            for gen, p, q, current in zip(self.live_gens, gen_p, gen_q, gen_current):
                gen.p = float(p)
                gen.q = float(q)
                gen.current = float(current)

        if self.live_loads:
            vm = V[self.load_obj_pos]
            load_p = self.load_pv0 + self.load_pv1 * vm + self.load_pv2 * vm ** 2
            load_q = self.load_qv0 + self.load_qv1 * vm + self.load_qv2 * vm ** 2
            load_current = np.divide(
                np.hypot(load_p, load_q),
                vm,
                out=np.zeros_like(load_p),
                where=vm > self.min_voltage,
            )
            for ld, p, q, current in zip(self.live_loads, load_p, load_q, load_current):
                ld.p = float(p)
                ld.q = float(q)
                ld.current = float(current)

        for sc, pos in zip(self.live_shunts, self.shunt_pos):
            if sc.control_type in ['B', 'Z'] or sc.g_set != 0.0:
                sc.p = V[pos] ** 2 * sc.g_set
                sc.q = -V[pos] ** 2 * sc.b_set
            else:
                sc.p = 0.0
                sc.q = sc.q_set if sc.control_type == 'Q' else 0.0
            sc.current = self.get_current(Vc[pos], sc.p, sc.q)

        if self.live_branches:
            Vi = Vc[self.branch_i]
            Vj = Vc[self.branch_j]
            I_ij = self.branch_yff * Vi + self.branch_yft * Vj
            I_ji = self.branch_ytf * Vi + self.branch_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            for idx, br in enumerate(self.live_branches):
                br.i_p = float(S_ij.real[idx])
                br.i_q = float(S_ij.imag[idx])
                br.i_c = float(abs(I_ij[idx]))
                br.j_p = float(S_ji.real[idx])
                br.j_q = float(S_ji.imag[idx])
                br.j_c = float(abs(I_ji[idx]))

        if self.live_transformers:
            Vi = Vc[self.transformer_i]
            Vj = Vc[self.transformer_j]
            I_ij = self.transformer_yff * Vi + self.transformer_yft * Vj
            I_ji = self.transformer_ytf * Vi + self.transformer_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            for idx, tr in enumerate(self.live_transformers):
                tr.i_p = float(S_ij.real[idx])
                tr.i_q = float(S_ij.imag[idx])
                tr.i_c = float(abs(I_ij[idx]))
                tr.j_p = float(S_ji.real[idx])
                tr.j_q = float(S_ji.imag[idx])
                tr.j_c = float(abs(I_ji[idx]))

        for sw in self.isl.switches:
            sw.current = sw.p = sw.q = 0.0

        for zb in self.isl.zero_branches:
            zb.current = zb.p = zb.q = 0.0
        for brk in getattr(self.isl, "breakers", []):
            brk.current = brk.p = brk.q = 0.0

        for dev_idx, dev_type, a, current in zip(
            self.zero_idx,
            self.zero_type,
            self.zero_a,
            I_ab,
        ):
            if dev_type == 0:
                dev = self.isl.zero_branches[int(dev_idx)]
            elif dev_type == 2:
                dev = self.isl.breakers[int(dev_idx)]
            else:
                dev = self.isl.switches[int(dev_idx)]
            s_from = Vc[a] * np.conj(current)
            dev.current = float(abs(current))
            dev.p = float(s_from.real)
            dev.q = float(s_from.imag)
        if not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result()

    def _device_voltage(self, node_idx) -> float:
        if int(node_idx) in getattr(self, "node_pos", {}):
            pos = self.node_pos[int(node_idx)]
            if hasattr(self, "x"):
                _theta, voltage, _phi_re, _phi_im = self._extract_state_vars(self.x)
                return float(voltage[pos])
        return 0.0

    def _build_lf_result(self) -> ACLFResult:
        if self.array_mode and isinstance(getattr(self, "result", None), dict):
            return self._build_lf_result_from_ppc()
        result = ACLFResult()
        model = self.net
        for node in getattr(model, "nodes", []):
            result.nodes[_device_key(node)] = SimpleNamespace(
                volt=float(getattr(node, "voltage", 0.0) or 0.0),
                angle=float(getattr(node, "angle", 0.0) or 0.0),
            )
        for br in getattr(model, "branches", []):
            result.branches[_device_key(br)] = SimpleNamespace(
                i_p=float(getattr(br, "i_p", 0.0) or 0.0),
                i_q=float(getattr(br, "i_q", 0.0) or 0.0),
                i_c=float(getattr(br, "i_c", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(br, "i_node", -1)),
                j_p=float(getattr(br, "j_p", 0.0) or 0.0),
                j_q=float(getattr(br, "j_q", 0.0) or 0.0),
                j_c=float(getattr(br, "j_c", 0.0) or 0.0),
                j_v=self._device_voltage(getattr(br, "j_node", -1)),
            )
        for tr in getattr(model, "transformers", []):
            result.transformers[_device_key(tr)] = SimpleNamespace(
                i_p=float(getattr(tr, "i_p", 0.0) or 0.0),
                i_q=float(getattr(tr, "i_q", 0.0) or 0.0),
                i_c=float(getattr(tr, "i_c", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(tr, "i_node", -1)),
                j_p=float(getattr(tr, "j_p", 0.0) or 0.0),
                j_q=float(getattr(tr, "j_q", 0.0) or 0.0),
                j_c=float(getattr(tr, "j_c", 0.0) or 0.0),
                j_v=self._device_voltage(getattr(tr, "j_node", -1)),
            )
        for zbr in getattr(model, "zero_branches", []):
            result.zero_branches[_device_key(zbr)] = SimpleNamespace(
                i_p=float(getattr(zbr, "p", 0.0) or 0.0),
                i_q=float(getattr(zbr, "q", 0.0) or 0.0),
                i_c=float(getattr(zbr, "current", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(zbr, "i_node", -1)),
            )
        for brk in getattr(model, "breakers", []):
            result.breakers[_device_key(brk)] = SimpleNamespace(
                i_p=float(getattr(brk, "p", 0.0) or 0.0),
                i_q=float(getattr(brk, "q", 0.0) or 0.0),
                i_c=float(getattr(brk, "current", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(brk, "i_node", -1)),
            )
        for gen in getattr(model, "generators", []):
            result.generators[_device_key(gen)] = SimpleNamespace(
                i_p=float(getattr(gen, "p", 0.0) or 0.0),
                i_q=float(getattr(gen, "q", 0.0) or 0.0),
                i_c=float(getattr(gen, "current", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(gen, "node", -1)),
            )
        for load in getattr(model, "loads", []):
            result.loads[_device_key(load)] = SimpleNamespace(
                i_p=float(getattr(load, "p", 0.0) or 0.0),
                i_q=float(getattr(load, "q", 0.0) or 0.0),
                i_c=float(getattr(load, "current", 0.0) or 0.0),
                i_v=self._device_voltage(getattr(load, "node", -1)),
            )
        return result

    def _build_lf_result_from_ppc(self) -> ACLFResult:
        result = ACLFResult()
        names = lambda key, n: self.ppc.get(key, np.asarray([str(i) for i in range(n)], dtype=object))
        for row, name in zip(self.result.get("bus", []), names("bus_name", len(self.result.get("bus", [])))):
            result.nodes[str(name)] = SimpleNamespace(
                volt=float(row[BUS_COLS["voltage"]]),
                angle=float(row[BUS_COLS["angle"]]),
            )
        for row, name in zip(self.result.get("branch", []), names("branch_name", len(self.result.get("branch", [])))):
            result.branches[str(name)] = SimpleNamespace(
                i_p=float(row[BRANCH_COLS["i_p"]]),
                i_q=float(row[BRANCH_COLS["i_q"]]),
                i_c=float(row[BRANCH_COLS["i_c"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[BRANCH_COLS["i_node"]])], BUS_COLS["voltage"]]) if int(row[BRANCH_COLS["i_node"]]) in self.node_pos else 0.0,
                j_p=float(row[BRANCH_COLS["j_p"]]),
                j_q=float(row[BRANCH_COLS["j_q"]]),
                j_c=float(row[BRANCH_COLS["j_c"]]),
                j_v=float(self.result["bus"][self.node_pos[int(row[BRANCH_COLS["j_node"]])], BUS_COLS["voltage"]]) if int(row[BRANCH_COLS["j_node"]]) in self.node_pos else 0.0,
            )
        for row, name in zip(self.result.get("transformer", []), names("transformer_name", len(self.result.get("transformer", [])))):
            result.transformers[str(name)] = SimpleNamespace(
                i_p=float(row[TRANSFORMER_COLS["i_p"]]),
                i_q=float(row[TRANSFORMER_COLS["i_q"]]),
                i_c=float(row[TRANSFORMER_COLS["i_c"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[TRANSFORMER_COLS["i_node"]])], BUS_COLS["voltage"]]) if int(row[TRANSFORMER_COLS["i_node"]]) in self.node_pos else 0.0,
                j_p=float(row[TRANSFORMER_COLS["j_p"]]),
                j_q=float(row[TRANSFORMER_COLS["j_q"]]),
                j_c=float(row[TRANSFORMER_COLS["j_c"]]),
                j_v=float(self.result["bus"][self.node_pos[int(row[TRANSFORMER_COLS["j_node"]])], BUS_COLS["voltage"]]) if int(row[TRANSFORMER_COLS["j_node"]]) in self.node_pos else 0.0,
            )
        for row, name in zip(self.result.get("zero_branch", []), names("zero_branch_name", len(self.result.get("zero_branch", [])))):
            result.zero_branches[str(name)] = SimpleNamespace(
                i_p=float(row[ZERO_BRANCH_COLS["p"]]),
                i_q=float(row[ZERO_BRANCH_COLS["q"]]),
                i_c=float(row[ZERO_BRANCH_COLS["current"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[ZERO_BRANCH_COLS["i_node"]])], BUS_COLS["voltage"]]) if int(row[ZERO_BRANCH_COLS["i_node"]]) in self.node_pos else 0.0,
            )
        for row, name in zip(self.result.get("break", []), names("break_name", len(self.result.get("break", [])))):
            result.breakers[str(name)] = SimpleNamespace(
                i_p=float(row[SWITCH_COLS["p"]]),
                i_q=float(row[SWITCH_COLS["q"]]),
                i_c=float(row[SWITCH_COLS["current"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[SWITCH_COLS["i_node"]])], BUS_COLS["voltage"]]) if int(row[SWITCH_COLS["i_node"]]) in self.node_pos else 0.0,
            )
        for row, name in zip(self.result.get("gen", []), names("gen_name", len(self.result.get("gen", [])))):
            result.generators[str(name)] = SimpleNamespace(
                i_p=float(row[GEN_COLS["p"]]),
                i_q=float(row[GEN_COLS["q"]]),
                i_c=float(row[GEN_COLS["current"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[GEN_COLS["node"]])], BUS_COLS["voltage"]]) if int(row[GEN_COLS["node"]]) in self.node_pos else 0.0,
            )
        for row, name in zip(self.result.get("load", []), names("load_name", len(self.result.get("load", [])))):
            result.loads[str(name)] = SimpleNamespace(
                i_p=float(row[LOAD_COLS["p"]]),
                i_q=float(row[LOAD_COLS["q"]]),
                i_c=float(row[LOAD_COLS["current"]]),
                i_v=float(self.result["bus"][self.node_pos[int(row[LOAD_COLS["node"]])], BUS_COLS["voltage"]]) if int(row[LOAD_COLS["node"]]) in self.node_pos else 0.0,
            )
        return result

    def _write_ppc_result_to_network(self) -> None:
        """Copy array-mode results back to the ACPowerNetwork passed by callers."""
        network = getattr(self, "_network_writeback", None)
        if network is None or not self.result:
            return

        def rows_by_idx(key, idx_col):
            rows = self.result.get(key)
            if rows is None or len(rows) == 0:
                return {}
            return {int(row[idx_col]): row for row in rows}

        bus_by_idx = rows_by_idx("bus", BUS_COLS["idx"])
        for node in getattr(network, "nodes", []):
            row = bus_by_idx.get(int(getattr(node, "idx", -1)))
            if row is None:
                continue
            node.voltage = float(row[BUS_COLS["voltage"]])
            node.angle = float(row[BUS_COLS["angle"]])

        for bus in getattr(network, "buses", []):
            members = [node for node in getattr(bus, "nodes", []) if getattr(node, "voltage", None) is not None]
            if not members:
                row = bus_by_idx.get(int(getattr(bus, "idx", -1)))
                if row is None:
                    continue
                bus.voltage = float(row[BUS_COLS["voltage"]])
                bus.angle = float(row[BUS_COLS["angle"]])
                continue
            bus.voltage = float(getattr(members[0], "voltage", 0.0) or 0.0)
            bus.angle = float(getattr(members[0], "angle", 0.0) or 0.0)

        def copy_fields(devices, row_map, fields):
            if not row_map:
                return
            for dev in devices:
                row = row_map.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    continue
                for attr, col in fields:
                    setattr(dev, attr, float(row[col]))

        copy_fields(
            getattr(network, "generators", []),
            rows_by_idx("gen", GEN_COLS["idx"]),
            (("p", GEN_COLS["p"]), ("q", GEN_COLS["q"]), ("current", GEN_COLS["current"])),
        )
        copy_fields(
            getattr(network, "loads", []),
            rows_by_idx("load", LOAD_COLS["idx"]),
            (("p", LOAD_COLS["p"]), ("q", LOAD_COLS["q"]), ("current", LOAD_COLS["current"])),
        )
        copy_fields(
            getattr(network, "shunt_compensators", []),
            rows_by_idx("shunt", SHUNT_COLS["idx"]),
            (("p", SHUNT_COLS["p"]), ("q", SHUNT_COLS["q"]), ("current", SHUNT_COLS["current"])),
        )
        copy_fields(
            getattr(network, "branches", []),
            rows_by_idx("branch", BRANCH_COLS["idx"]),
            (
                ("i_p", BRANCH_COLS["i_p"]),
                ("i_q", BRANCH_COLS["i_q"]),
                ("i_c", BRANCH_COLS["i_c"]),
                ("j_p", BRANCH_COLS["j_p"]),
                ("j_q", BRANCH_COLS["j_q"]),
                ("j_c", BRANCH_COLS["j_c"]),
            ),
        )
        copy_fields(
            getattr(network, "transformers", []),
            rows_by_idx("transformer", TRANSFORMER_COLS["idx"]),
            (
                ("i_p", TRANSFORMER_COLS["i_p"]),
                ("i_q", TRANSFORMER_COLS["i_q"]),
                ("i_c", TRANSFORMER_COLS["i_c"]),
                ("j_p", TRANSFORMER_COLS["j_p"]),
                ("j_q", TRANSFORMER_COLS["j_q"]),
                ("j_c", TRANSFORMER_COLS["j_c"]),
            ),
        )
        copy_fields(
            getattr(network, "zero_branches", []),
            rows_by_idx("zero_branch", ZERO_BRANCH_COLS["idx"]),
            (("p", ZERO_BRANCH_COLS["p"]), ("q", ZERO_BRANCH_COLS["q"]), ("current", ZERO_BRANCH_COLS["current"])),
        )
        copy_fields(
            getattr(network, "switches", []),
            rows_by_idx("switch", SWITCH_COLS["idx"]),
            (("p", SWITCH_COLS["p"]), ("q", SWITCH_COLS["q"]), ("current", SWITCH_COLS["current"])),
        )
        copy_fields(
            getattr(network, "breakers", []),
            rows_by_idx("break", SWITCH_COLS["idx"]),
            (("p", SWITCH_COLS["p"]), ("q", SWITCH_COLS["q"]), ("current", SWITCH_COLS["current"])),
        )

    def _write_back_ppc(self):
        """Write array-mode results to self.result without mutating the input ppc."""
        theta, V, phi_re, phi_im = self._extract_state_vars(self.x)
        Vc = self._cache['Vc']
        P_load, Q_load = self._cache['P_load'], self._cache['Q_load']
        if P_load is None or Q_load is None:
            P_load, Q_load = self._calc_load_power(V)

        I_y = self.Y.dot(Vc)
        S_y = Vc * np.conj(I_y)
        P_zero = np.zeros(self.N, dtype=np.float64)
        Q_zero = np.zeros(self.N, dtype=np.float64)
        if self.N_phi > 0 and self.zero_a.size:
            I_ab = (phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]) + 1j * (
                phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]
            )
            Sa = Vc[self.zero_a] * np.conj(I_ab)
            Sb = Vc[self.zero_b] * np.conj(-I_ab)
            np.add.at(P_zero, self.zero_a, Sa.real)
            np.add.at(Q_zero, self.zero_a, Sa.imag)
            np.add.at(P_zero, self.zero_b, Sb.real)
            np.add.at(Q_zero, self.zero_b, Sb.imag)
        else:
            I_ab = np.array([], dtype=np.complex128)

        P_gen = S_y.real + P_zero + P_load
        Q_gen = S_y.imag + Q_zero + Q_load

        bus = self.ppc["bus"].copy()
        bus[self.active_bus_rows, BUS_COLS["voltage"]] = V
        bus[self.active_bus_rows, BUS_COLS["angle"]] = theta
        for pos, node in enumerate(self.node_list):
            node.voltage = float(V[pos])
            node.angle = float(theta[pos])

        gen = self.ppc["gen"].copy()
        if self.ppc_gen_rows.size:
            gen_p = self.ppc_gen_share * P_gen[self.ppc_gen_pos]
            gen_q = self.ppc_gen_share * Q_gen[self.ppc_gen_pos]
            gen_v = V[self.ppc_gen_pos]
            gen_current = np.divide(
                np.hypot(gen_p, gen_q),
                gen_v,
                out=np.zeros_like(gen_p),
                where=gen_v > self.min_voltage,
            )
            gen[self.ppc_gen_rows, GEN_COLS["p"]] = gen_p
            gen[self.ppc_gen_rows, GEN_COLS["q"]] = gen_q
            gen[self.ppc_gen_rows, GEN_COLS["current"]] = gen_current

        load = self.ppc["load"].copy()
        if self.ppc_load_rows.size:
            vm = V[self.ppc_load_pos]
            load_p = self.load_pv0 + self.load_pv1 * vm + self.load_pv2 * vm ** 2
            load_q = self.load_qv0 + self.load_qv1 * vm + self.load_qv2 * vm ** 2
            load_current = np.divide(
                np.hypot(load_p, load_q),
                vm,
                out=np.zeros_like(load_p),
                where=vm > self.min_voltage,
            )
            load[self.ppc_load_rows, LOAD_COLS["p"]] = load_p
            load[self.ppc_load_rows, LOAD_COLS["q"]] = load_q
            load[self.ppc_load_rows, LOAD_COLS["current"]] = load_current

        shunt = self.ppc["shunt"].copy()
        if self.ppc_shunt_rows.size:
            for row_idx, pos in zip(self.ppc_shunt_rows, self.ppc_shunt_pos):
                control = int(shunt[row_idx, SHUNT_COLS["control_type"]])
                if control in (SHUNT_B, SHUNT_Z) or shunt[row_idx, SHUNT_COLS["g_set"]] != 0.0:
                    p = V[pos] ** 2 * shunt[row_idx, SHUNT_COLS["g_set"]]
                    q = -V[pos] ** 2 * shunt[row_idx, SHUNT_COLS["b_set"]]
                else:
                    p = 0.0
                    q = shunt[row_idx, SHUNT_COLS["q_set"]] if control == SHUNT_Q else 0.0
                shunt[row_idx, SHUNT_COLS["p"]] = p
                shunt[row_idx, SHUNT_COLS["q"]] = q
                shunt[row_idx, SHUNT_COLS["current"]] = self.get_current(Vc[pos], p, q)

        branch = self.ppc["branch"].copy()
        if self.ppc_branch_rows.size:
            Vi = Vc[self.branch_i]
            Vj = Vc[self.branch_j]
            I_ij = self.branch_yff * Vi + self.branch_yft * Vj
            I_ji = self.branch_ytf * Vi + self.branch_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            rows = self.ppc_branch_rows
            branch[rows, BRANCH_COLS["i_p"]] = S_ij.real
            branch[rows, BRANCH_COLS["i_q"]] = S_ij.imag
            branch[rows, BRANCH_COLS["i_c"]] = np.abs(I_ij)
            branch[rows, BRANCH_COLS["j_p"]] = S_ji.real
            branch[rows, BRANCH_COLS["j_q"]] = S_ji.imag
            branch[rows, BRANCH_COLS["j_c"]] = np.abs(I_ji)

        transformer = self.ppc["transformer"].copy()
        if self.ppc_transformer_rows.size:
            Vi = Vc[self.transformer_i]
            Vj = Vc[self.transformer_j]
            I_ij = self.transformer_yff * Vi + self.transformer_yft * Vj
            I_ji = self.transformer_ytf * Vi + self.transformer_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            rows = self.ppc_transformer_rows
            transformer[rows, TRANSFORMER_COLS["i_p"]] = S_ij.real
            transformer[rows, TRANSFORMER_COLS["i_q"]] = S_ij.imag
            transformer[rows, TRANSFORMER_COLS["i_c"]] = np.abs(I_ij)
            transformer[rows, TRANSFORMER_COLS["j_p"]] = S_ji.real
            transformer[rows, TRANSFORMER_COLS["j_q"]] = S_ji.imag
            transformer[rows, TRANSFORMER_COLS["j_c"]] = np.abs(I_ji)

        zero_branch = self.ppc["zero_branch"].copy()
        switch = self.ppc["switch"].copy()
        breaker = self.ppc.get("break", np.zeros((0, len(SWITCH_COLS)))).copy()
        for dev_idx, dev_type, a, current in zip(self.zero_idx, self.zero_type, self.zero_a, I_ab):
            s_from = Vc[a] * np.conj(current)
            if dev_type == 0:
                zero_branch[dev_idx, ZERO_BRANCH_COLS["p"]] = s_from.real
                zero_branch[dev_idx, ZERO_BRANCH_COLS["q"]] = s_from.imag
                zero_branch[dev_idx, ZERO_BRANCH_COLS["current"]] = abs(current)
            elif dev_type == 2:
                breaker[dev_idx, SWITCH_COLS["p"]] = s_from.real
                breaker[dev_idx, SWITCH_COLS["q"]] = s_from.imag
                breaker[dev_idx, SWITCH_COLS["current"]] = abs(current)
            else:
                switch[dev_idx, SWITCH_COLS["p"]] = s_from.real
                switch[dev_idx, SWITCH_COLS["q"]] = s_from.imag
                switch[dev_idx, SWITCH_COLS["current"]] = abs(current)

        self.result = {
            "bus": bus,
            "gen": gen,
            "load": load,
            "shunt": shunt,
            "branch": branch,
            "transformer": transformer,
            "zero_branch": zero_branch,
            "switch": switch,
            "break": breaker,
        }
        self._write_ppc_result_to_network()
        if not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result()

def print_ac_result(calc: ACPowerFlowCalc, rc: int) -> None:
    for isl in calc.skipped_islands:
        print(f"跳过岛屿 {isl.idx}: 无可用平衡节点或定电压源")

    def _names(key, count):
        values = calc.ppc.get(key)
        if values is None:
            return np.asarray([str(i) for i in range(count)], dtype=object)
        return values

    # 结果输出（精简）
    print("\n=== 潮流计算结果 ===")

    # 节点电压
    print("\n1. 节点电压 (pu):")
    bus = calc.result["bus"]
    bus_names = _names("bus_name", bus.shape[0])
    slack_node_ids = set(calc.ppc_node_idx[calc.node_type == "SLACK"].tolist())
    active_node_ids = set(calc.ppc_node_idx.tolist())
    for row, name in zip(bus, bus_names):
        node_idx = int(row[BUS_COLS["idx"]])
        flags = []
        if node_idx in slack_node_ids:
            flags.append("松弛节点")
        if node_idx not in active_node_ids:
            flags.append("未参与计算")
        suffix = f" ({', '.join(flags)})" if flags else ""
        print(
            f"   节点 {node_idx} {name}: "
            f"{row[BUS_COLS['voltage']]:.6f}  {row[BUS_COLS['angle']]:.6f}{suffix}"
        )

    # 支路信息
    branch = calc.result["branch"]
    branch_names = _names("branch_name", branch.shape[0])
    p_loss_br = float(np.sum(branch[:, BRANCH_COLS["i_p"]] + branch[:, BRANCH_COLS["j_p"]])) if branch.size else 0.0
    print("\n2. 支路信息:")
    for row, name in zip(branch, branch_names):
        loss = row[BRANCH_COLS["i_p"]] + row[BRANCH_COLS["j_p"]]
        print(f"   支路 {int(row[BRANCH_COLS['idx']])} {name} ({int(row[BRANCH_COLS['i_node']])}->{int(row[BRANCH_COLS['j_node']])}):")
        print(
            f"     送端功率: {row[BRANCH_COLS['i_p']]:.6f} + j {row[BRANCH_COLS['i_q']]:.6f} pu, "
            f"受端功率: {row[BRANCH_COLS['j_p']]:.6f} + j{row[BRANCH_COLS['j_q']]:.6f} pu"
        )
        print(f"     有功损耗: {loss:.6f} pu")

    # 变压器信息
    transformer = calc.result["transformer"]
    transformer_names = _names("transformer_name", transformer.shape[0])
    p_loss_tr = (
        float(np.sum(transformer[:, TRANSFORMER_COLS["i_p"]] + transformer[:, TRANSFORMER_COLS["j_p"]]))
        if transformer.size
        else 0.0
    )
    print("\n3. 变压器信息:")
    for row, name in zip(transformer, transformer_names):
        loss = row[TRANSFORMER_COLS["i_p"]] + row[TRANSFORMER_COLS["j_p"]]
        print(
            f"   变压器 {int(row[TRANSFORMER_COLS['idx']])} {name} "
            f"({int(row[TRANSFORMER_COLS['i_node']])}->{int(row[TRANSFORMER_COLS['j_node']])}):"
        )
        print(
            f"     送端功率: {row[TRANSFORMER_COLS['i_p']]:.6f} + j {row[TRANSFORMER_COLS['i_q']]:.6f} pu, "
            f"受端功率: {row[TRANSFORMER_COLS['j_p']]:.6f} + j{row[TRANSFORMER_COLS['j_q']]:.6f} pu"
        )
        print(f"     有功损耗: {loss:.6f} pu")

    # 其他设备信息
    zero_branch = calc.result["zero_branch"]
    zero_names = _names("zero_branch_name", zero_branch.shape[0])
    print("\n4. 零阻抗支路信息:")
    for row, name in zip(zero_branch, zero_names):
        print(
            f"   零阻抗支路 {int(row[ZERO_BRANCH_COLS['idx']])} {name}: "
            f"电流={row[ZERO_BRANCH_COLS['current']]:.6f} pu, "
            f"功率={row[ZERO_BRANCH_COLS['p']]:.6f} + j {row[ZERO_BRANCH_COLS['q']]:.6f} pu"
        )

    switch = calc.result["switch"]
    switch_names = _names("switch_name", switch.shape[0])
    print("\n5. 开关信息:")
    for row, name in zip(switch, switch_names):
        status = "闭合" if int(row[SWITCH_COLS["status"]]) == 1 else "断开"
        print(
            f"   开关 {int(row[SWITCH_COLS['idx']])} {name} "
            f"(状态:{status}): 电流={row[SWITCH_COLS['current']]:.6f} pu"
        )

    breaker = calc.result["break"]
    breaker_names = _names("break_name", breaker.shape[0])
    if breaker.shape[0]:
        print("\n5.1 刀闸信息:")
        for row, name in zip(breaker, breaker_names):
            status = "闭合" if int(row[SWITCH_COLS["status"]]) == 1 else "断开"
            print(
                f"   刀闸 {int(row[SWITCH_COLS['idx']])} {name} "
                f"(状态:{status}): 电流={row[SWITCH_COLS['current']]:.6f} pu"
            )

    load = calc.result["load"]
    load_names = _names("load_name", load.shape[0])
    print("\n6. 负荷信息:")
    for row, name in zip(load, load_names):
        print(
            f"   负荷 {int(row[LOAD_COLS['idx']])} {name}: "
            f"消耗功率={row[LOAD_COLS['p']]:.6f} + j {row[LOAD_COLS['q']]:.6f} pu"
        )

    gen = calc.result["gen"]
    gen_names = _names("gen_name", gen.shape[0])
    print("\n7. 发电机信息:")
    for row, name in zip(gen, gen_names):
        print(
            f"   发电机 {int(row[GEN_COLS['idx']])} {name}: "
            f"送出功率={row[GEN_COLS['p']]:.6f} + j {row[GEN_COLS['q']]:.6f} pu"
        )

    # 收敛信息和功率平衡
    print("\n8. 计算收敛信息:")
    calc_scope = f"全网 (参与节点 {calc.N}/{bus.shape[0]})"
    print(
        f"   {calc_scope}: {'✓ 已收敛' if calc.converged else '✗ 未收敛'}, "
        f"返回码: {rc}, 迭代次数: {calc.iterations}, 最终残差: {calc.normF:.2e}"
    )

    total_gen_power = float(np.sum(gen[:, GEN_COLS["p"]])) if gen.size else 0.0
    total_load_power = float(np.sum(load[:, LOAD_COLS["p"]])) if load.size else 0.0
    total_loss = total_gen_power - total_load_power
    shunt = calc.result["shunt"]
    p_loss_gs = float(np.sum(shunt[:, SHUNT_COLS["p"]])) if shunt.size else 0.0
    print("\n9. 功率平衡校验:")
    print(f"   总发电功率: {total_gen_power:.6f} pu")
    print(f"   总负荷功率: {total_load_power:.6f} pu")
    print(f"   总网损: {total_loss:.6f} pu (支路: {p_loss_br:.6f} pu, 变压器: {p_loss_tr:.6f} pu, 并联电导: {p_loss_gs:.6f} pu)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AC power flow")
    parser.add_argument("file", nargs="?", default=str(model_file("ac", "ieee300.e")), help="AC E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument("--algorithm", choices=("nr", "pq"), default="nr")
    parser.add_argument("--linear-solver", default="scipy")
    parser.add_argument("--result-mode", choices=("full", "summary", "none"), default="full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ppc = load_ac_ppc_from_e_file(args.file)
    calc = ACPowerFlowCalc(
        ppc,
        parameter_file=args.para,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        algorithm=args.algorithm,
        linear_solver=args.linear_solver,
        result_mode=args.result_mode,
        verbose=not args.quiet,
    )
    calc.prepare()
    rc = calc.run()
    if not args.quiet and calc.result_mode == "full":
        print_ac_result(calc, rc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
