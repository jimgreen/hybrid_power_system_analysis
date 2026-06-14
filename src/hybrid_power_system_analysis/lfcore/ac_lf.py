import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve

try:
    from scipy.sparse.linalg import use_solver as _scipy_use_solver
    _scipy_use_solver(useUmfpack=False)
except Exception:
    pass

from collections import deque
from typing import List, Tuple, Dict, Optional
import warnings
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, MODEL_DIR):
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
from ac_array_model import (
    ACAC_COLS,
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
    build_ac_ppc_from_e_file,
    build_ac_ppc_from_mat_file,
    build_ac_ppc_from_network,
)
from model.ppc_topology import build_ac_ppc_with_topology_from_e_file, ensure_ac_ppc_topology
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


AC_NODE_TYPE_PQ = np.int8(1)
AC_NODE_TYPE_PV = np.int8(2)
AC_NODE_TYPE_SLACK = np.int8(3)
AC_NODE_TYPE_LABELS = {
    int(AC_NODE_TYPE_PQ): "PQ",
    int(AC_NODE_TYPE_PV): "PV",
    int(AC_NODE_TYPE_SLACK): "SLACK",
}


def ac_node_type_label(node_type) -> str:
    return AC_NODE_TYPE_LABELS.get(int(node_type), str(int(node_type)))


@dataclass
class ACLFResult:
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    acac_converters: Dict[str, SimpleNamespace] = field(default_factory=dict)
    branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    transformers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    zero_branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    breakers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    generators: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)


def load_ac_ppc_from_e_file(file_name) -> Dict:
    """Read an AC E/MATPOWER file into PPC with topology arrays attached."""
    path = Path(file_name)
    suffix = path.suffix.lower()
    if suffix in {".m", ".mat"}:
        ppc = build_ac_ppc_from_mat_file(path)
        ppc["source"] = str(path.resolve())
        return ensure_ac_ppc_topology(ppc)
    return build_ac_ppc_with_topology_from_e_file(path)


def _build_csr_pattern_from_raw_coords(raw_rows, raw_cols, n_rows: int):
    """Build a CSR sparsity pattern and raw-entry-to-CSR-position map.

    ``raw_rows``/``raw_cols`` may contain duplicate coordinates.  The returned
    ``raw_to_csr`` array maps each raw coordinate to the corresponding entry in
    the unique CSR pattern so runtime values can be accumulated with
    a precomputed copy/reduce plan.
    """
    return build_compressed_pattern_from_raw_coords(raw_rows, raw_cols, n_rows)


class _PPCNode:
    __slots__ = ("idx", "name", "vbase", "voltage", "angle")

    def __init__(self, idx, name, vbase, voltage, angle):
        self.idx = idx
        self.name = name
        self.vbase = vbase
        self.voltage = voltage
        self.angle = angle


def safe_division(a, b, default=0.0):
    """安全除法，避免除零错误"""
    try:
        return a / b if abs(b) > 1e-12 else default
    except Exception:
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
    """Vectorized MATPOWER-compatible branch admittance stamp for branch batches.

    ``b`` remains a branch parameter, not a transformer parameter. It is the
    total shunt susceptance and is applied as ``j*b/2`` at each end.
    """
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


def matpower_transformer_stamp_vectorized(r, x, gt=0.0, bt=0.0, tap=1.0, shift=0.0):
    """Vectorized T-type transformer stamp with one grounding branch on the i side.

    ``gt + j*bt`` is a single shunt admittance on the i side before the ideal
    tap/phase-shift transformer. It is not MATPOWER's symmetric ``BR_B`` line
    charging and must not be split across both terminals.
    """
    r = np.asarray(r, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    gt = np.broadcast_to(np.asarray(gt, dtype=np.float64), r.shape)
    bt = np.broadcast_to(np.asarray(bt, dtype=np.float64), r.shape)
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
    # Transformer shunt is single-ended: add it only to the i-side
    # self-admittance, then refer it through the complex tap.
    y_ground = gt + 1j * bt
    return (
        (y + y_ground) / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y,
    )


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
        keep_node_objects: bool = True,
        linear_solver: str = "pyklu",
        result_mode: str = "full",
        verbose: bool = False,
    ):
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
            existing_ppc = getattr(network, "ppc", None)
            if isinstance(existing_ppc, dict) and existing_ppc.get("format") == "ac_ppc_v1":
                self.ppc = existing_ppc
            else:
                self.ppc = build_ac_ppc_from_network(network)
            if hasattr(network, "source") and "source" not in self.ppc:
                self.ppc["source"] = str(getattr(network, "source"))
        else:
            raise ValueError("ACPowerFlowCalc requires ac_ppc_v1 or ACPowerNetwork input")
        self.result_mode = self._normalize_result_mode(result_mode)
        self.keep_node_objects = False
        self._cache_csr_jacobian_pattern = self.result_mode == "full"
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.min_voltage = self.params.min_voltage
        # 用户请求的求解器名原样保留，便于上层日志/测试断言；实际 callable
        # 由 _resolve_linear_solver 决定，未安装时回退 SuperLU。
        self.linear_solver = str(linear_solver or "pyklu").strip().lower()
        self._linear_solver_resolved, self._linear_solver_fn = _resolve_linear_solver(self.linear_solver)
        # 实例级可选求解器黑名单: 失败时只在本实例回退 scipy, 不污染模块缓存。
        self._instance_solver_blacklist: set = set()
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
        self._ppc_node_row_lookup = np.array([], dtype=np.int32)
        self._active_node_solver_lookup = np.array([], dtype=np.int32)
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
        self.theta_idx = np.array([], dtype=np.int32)
        self.V_idx = np.array([], dtype=np.int32)
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
        self.standard_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.standard_jac_csr_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csr_pos, 0)
        self.standard_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.standard_jac_csc_indices = np.array([], dtype=np.int32)
        self.standard_jac_csc_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csc_data = np.array([], dtype=np.float64)
        self.standard_jac_csc_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csc_pos, 0)
        self.full_jac_raw_data = np.array([], dtype=np.float64)
        self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.full_jac_csr_indices = np.array([], dtype=np.int32)
        self.full_jac_csr_indptr = np.array([], dtype=np.int32)
        self.full_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, 0)
        self.full_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.full_jac_csc_indices = np.array([], dtype=np.int32)
        self.full_jac_csc_indptr = np.array([], dtype=np.int32)
        self.full_jac_csc_data = np.array([], dtype=np.float64)
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, 0)
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
        self._external_ac_p_injection = None
        self._external_ac_q_injection = None
        self.ppc_load_rows = np.array([], dtype=np.int32)
        self.ppc_load_pos = np.array([], dtype=np.int32)
        self.ppc_shunt_rows = np.array([], dtype=np.int32)
        self.ppc_shunt_pos = np.array([], dtype=np.int32)
        self.ppc_branch_rows = np.array([], dtype=np.int32)
        self.ppc_transformer_rows = np.array([], dtype=np.int32)
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

    @classmethod
    def from_file_fast(
        cls,
        file_name,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        linear_solver: Optional[str] = None,
        result_mode: str = "array",
        verbose: bool = False,
    ) -> "ACPowerFlowCalc":
        """Build an AC PPC-backed solver directly from file.

        Accepted inputs:
        - `.e` named-unit/network files
        - `.m` / `.mat` MATPOWER cases
        """
        path = Path(file_name)
        suffix = path.suffix.lower()
        if suffix in {".m", ".mat"}:
            ppc = build_ac_ppc_from_mat_file(path)
        elif suffix == ".e":
            ppc = build_ac_ppc_from_e_file(path)
        else:
            raise ValueError(f"ACPowerFlowCalc.from_file_fast() only supports .e/.m/.mat files, got: {path}")
        return cls(
            ppc,
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
            parameter_file=parameter_file,
            parameters=parameters,
            linear_solver=linear_solver,
            result_mode=result_mode,
            verbose=verbose,
        )

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        return _normalize_lf_result_mode(result_mode, "AC")

    def _cache_node_type_masks(self):
        self._slack_mask = self.node_type == AC_NODE_TYPE_SLACK
        self._fixed_voltage_mask = self._slack_mask | (self.node_type == AC_NODE_TYPE_PV)

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

            # 累加至对应节点（bincount 比 np.add.at 快 5-10×）。
            P_load[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

            np.multiply(vm_vals, vm_vals, out=values)
            values *= self.load_qv2
            np.multiply(self.load_qv1, vm_vals, out=aux)
            values += aux
            values += self.load_qv0
            Q_load[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

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

            # bincount 比 np.add.at 快得多；分两端调用避免引入 np.concatenate。
            P_zero = np.bincount(self.zero_a, weights=Sa.real, minlength=self.N)
            P_zero += np.bincount(self.zero_b, weights=Sb.real, minlength=self.N)
            Q_zero = np.bincount(self.zero_a, weights=Sa.imag, minlength=self.N)
            Q_zero += np.bincount(self.zero_b, weights=Sb.imag, minlength=self.N)

        return P_zero, Q_zero

    def get_current(self, v: float, p: float, q: float):
        return abs(np.conj(complex(p, q)) / v) if abs(v) > self.min_voltage else 0.0

    def _ppc_node_rows(self, node_ids) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if self._ppc_sequential_node_ids:
            return node_ids.astype(np.int32)
        lookup = getattr(self, "_ppc_node_row_lookup", None)
        if isinstance(lookup, np.ndarray) and lookup.size:
            rows = np.full(node_ids.shape, -1, dtype=np.int32)
            valid = (node_ids >= 0) & (node_ids < lookup.size)
            if np.any(valid):
                rows[valid] = lookup[node_ids[valid].astype(np.intp, copy=False)]
            if np.any(rows < 0):
                missing = int(node_ids[np.flatnonzero(rows < 0)[0]])
                raise KeyError(missing)
            return rows
        return np.fromiter(
            (self._ppc_node_row_by_id[int(node_id)] for node_id in node_ids),
            dtype=np.int32,
            count=node_ids.size,
        )

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
        self._ppc_node_row_lookup = np.array([], dtype=np.int32)
        self._ppc_node_row_by_id = {}
        if not self._ppc_sequential_node_ids:
            if bus_ids.size and np.all(bus_ids >= 0):
                self._ppc_node_row_lookup = np.full(int(np.max(bus_ids)) + 1, -1, dtype=np.int32)
                self._ppc_node_row_lookup[bus_ids.astype(np.intp, copy=False)] = np.arange(
                    n_bus_all,
                    dtype=np.int32,
                )
            else:
                self._ppc_node_row_by_id = {int(node_id): pos for pos, node_id in enumerate(bus_ids)}
        ensure_ac_ppc_topology(ppc)
        topology = ppc["_topology_arrays"]
        self._ppc_topology = topology
        active_bus = np.asarray(topology.node_alive_mask, dtype=bool)
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
        if self.ppc_node_idx.size and np.all(self.ppc_node_idx >= 0):
            self._active_node_solver_lookup = np.full(int(np.max(self.ppc_node_idx)) + 1, -1, dtype=np.int32)
            self._active_node_solver_lookup[self.ppc_node_idx.astype(np.intp, copy=False)] = np.arange(
                self.N,
                dtype=np.int32,
            )
        else:
            self._active_node_solver_lookup = np.array([], dtype=np.int32)
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

        self.node_type = np.full(self.N, AC_NODE_TYPE_PQ, dtype=np.int8)
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
        if self.verbose:
            print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")

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
                self.node_type[slack_pos] = AC_NODE_TYPE_SLACK
                self.V_spec[slack_pos] = live_gen[slack_mask, GEN_COLS["v_set"]]
                self.theta_spec[slack_pos] = self.ppc["bus"][self.active_bus_rows[slack_pos], BUS_COLS["angle"]]
                self.slack_node = int(slack_pos[-1])

            pv_mask = (controls == CTRL_PV) & (self.node_type[self.ppc_gen_pos] != AC_NODE_TYPE_SLACK)
            if np.any(pv_mask):
                pv_pos = self.ppc_gen_pos[pv_mask]
                pv_v = live_gen[pv_mask, GEN_COLS["v_set"]]
                for pos, v_set in zip(pv_pos, pv_v):
                    if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - v_set) > 1e-6:
                        raise ValueError(f"节点{pos}多个PV发电机电压设定冲突")
                self.node_type[pv_pos] = AC_NODE_TYPE_PV
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
            v_mask = (controls == SHUNT_V) & (self.node_type[self.ppc_shunt_pos] != AC_NODE_TYPE_SLACK)
            if np.any(v_mask):
                v_pos = self.ppc_shunt_pos[v_mask]
                v_set_values = live_shunt[v_mask, SHUNT_COLS["v_set"]]
                for pos, v_set in zip(v_pos, v_set_values):
                    if not np.isnan(self.V_spec[pos]) and abs(self.V_spec[pos] - v_set) > 1e-6:
                        raise ValueError(f"节点{pos}电压设定冲突")
                self.node_type[v_pos] = AC_NODE_TYPE_PV
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
            pv_indices = np.where(self.node_type == AC_NODE_TYPE_PV)[0]
            if pv_indices.size:
                self.slack_node = int(pv_indices[0])
                self.node_type[self.slack_node] = AC_NODE_TYPE_SLACK
                self.theta_spec[self.slack_node] = self.ppc["bus"][self.active_bus_rows[self.slack_node], BUS_COLS["angle"]]
        if self.slack_node == -1:
            external_refs = np.asarray(self.ppc.get("_external_angle_reference_node_ids", []), dtype=np.int64)
            if external_refs.size == 0:
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
            ) = matpower_transformer_stamp_vectorized(
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["r"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["x"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["gt"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["bt"]],
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
        self.theta_unknown = np.where(self.node_type != AC_NODE_TYPE_SLACK)[0].astype(np.int32)
        self.V_unknown = np.where(self.node_type == AC_NODE_TYPE_PQ)[0].astype(np.int32)
        self.n_theta = len(self.theta_unknown)
        self.n_V = len(self.V_unknown)
        self.theta_idx = np.full(self.N, -1, dtype=np.int32)
        self.V_idx = np.full(self.N, -1, dtype=np.int32)
        self.theta_idx[self.theta_unknown] = np.arange(self.n_theta, dtype=np.int32)
        self.V_idx[self.V_unknown] = np.arange(self.n_V, dtype=np.int32)
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
        # 残差里零阻抗树边按 comp_tree_edges 顺序展平成 (a, b) 数组。
        if self.comp_tree_edges:
            flat_edges = [
                self.zero_edges[edge_idx]
                for tree in self.comp_tree_edges
                for edge_idx in tree
            ]
            self._zero_residual_a = np.asarray(
                [edge[2] for edge in flat_edges], dtype=np.int32
            )
            self._zero_residual_b = np.asarray(
                [edge[3] for edge in flat_edges], dtype=np.int32
            )
        else:
            self._zero_residual_a = np.empty(0, dtype=np.int32)
            self._zero_residual_b = np.empty(0, dtype=np.int32)
        self._ref_phi_idx_arr = (
            np.asarray(self.ref_phi_idx, dtype=np.int32)
            if self.ref_phi_idx
            else np.empty(0, dtype=np.int32)
        )
        self._cache_static_numeric_arrays()
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
        self.pq_theta_rows = self.theta_idx[self.V_unknown].astype(np.int32, copy=True)
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
        self.standard_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.standard_jac_csr_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csr_pos, 0)
        self.standard_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.standard_jac_csc_indices = np.array([], dtype=np.int32)
        self.standard_jac_csc_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csc_data = np.array([], dtype=np.float64)
        self.standard_jac_csc_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csc_pos, 0)
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
            n_dim = self.n_theta + self.n_V
            cache_standard_csc = self.N_phi == 0
            if self._cache_csr_jacobian_pattern:
                (
                    self.standard_jac_csr_indices,
                    self.standard_jac_csr_indptr,
                    self.standard_jac_raw_to_csr_pos,
                ) = build_compressed_pattern_from_raw_coords(self.standard_jac_rows, self.standard_jac_cols, n_dim)
                self.standard_jac_csr_data = np.empty(self.standard_jac_csr_indices.size, dtype=np.float64)
            if cache_standard_csc:
                (
                    self.standard_jac_csc_indices,
                    self.standard_jac_csc_indptr,
                    self.standard_jac_raw_to_csc_pos,
                ) = build_compressed_pattern_from_raw_coords(self.standard_jac_cols, self.standard_jac_rows, n_dim)
                self.standard_jac_csc_data = np.empty(self.standard_jac_csc_indices.size, dtype=np.float64)
            if self._cache_csr_jacobian_pattern:
                self.standard_jac_csr_sum_plan = build_raw_sum_plan(
                    self.standard_jac_raw_to_csr_pos,
                    self.standard_jac_csr_data.size,
                )
            if cache_standard_csc:
                self.standard_jac_csc_sum_plan = build_raw_sum_plan(
                    self.standard_jac_raw_to_csc_pos,
                    self.standard_jac_csc_data.size,
                )
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

            if self.node_type[a] != AC_NODE_TYPE_SLACK:
                eq_a = self.theta_idx[a]
                top_left_rows.append(eq_a)
                top_left_cols.append(eq_a)
                top_left_edge.append(edge_pos)
                top_left_kind.append(0)
                if self.node_type[a] == AC_NODE_TYPE_PQ:
                    top_left_rows.append(eq_a)
                    top_left_cols.append(self.n_theta + self.V_idx[a])
                    top_left_edge.append(edge_pos)
                    top_left_kind.append(1)
                top_right_rows.extend([eq_a] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([0, 1, 2, 3])

            if self.node_type[a] == AC_NODE_TYPE_PQ:
                eq_a = self.n_theta + self.V_idx[a]
                top_left_rows.extend([eq_a, eq_a])
                top_left_cols.extend([self.theta_idx[a], self.n_theta + self.V_idx[a]])
                top_left_edge.extend([edge_pos, edge_pos])
                top_left_kind.extend([2, 3])
                top_right_rows.extend([eq_a] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([4, 5, 6, 7])

            if self.node_type[b] != AC_NODE_TYPE_SLACK:
                eq_b = self.theta_idx[b]
                top_left_rows.append(eq_b)
                top_left_cols.append(eq_b)
                top_left_edge.append(edge_pos)
                top_left_kind.append(4)
                if self.node_type[b] == AC_NODE_TYPE_PQ:
                    top_left_rows.append(eq_b)
                    top_left_cols.append(self.n_theta + self.V_idx[b])
                    top_left_edge.append(edge_pos)
                    top_left_kind.append(5)
                top_right_rows.extend([eq_b] * 4)
                top_right_cols.extend(phi_cols)
                top_right_edge.extend([edge_pos] * 4)
                top_right_kind.extend([8, 9, 10, 11])

            if self.node_type[b] == AC_NODE_TYPE_PQ:
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
                if self.node_type[a] != AC_NODE_TYPE_SLACK:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(0)
                if self.node_type[a] == AC_NODE_TYPE_PQ:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(1)
                if self.node_type[b] != AC_NODE_TYPE_SLACK:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(2)
                if self.node_type[b] == AC_NODE_TYPE_PQ:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(3)
                eq_idx += 1

                if self.node_type[a] != AC_NODE_TYPE_SLACK:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(4)
                if self.node_type[a] == AC_NODE_TYPE_PQ:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.n_theta + self.V_idx[a])
                    bottom_left_node.append(a)
                    bottom_left_kind.append(5)
                if self.node_type[b] != AC_NODE_TYPE_SLACK:
                    bottom_left_rows.append(eq_idx)
                    bottom_left_cols.append(self.theta_idx[b])
                    bottom_left_node.append(b)
                    bottom_left_kind.append(6)
                if self.node_type[b] == AC_NODE_TYPE_PQ:
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
        self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, 0)
        self.full_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.full_jac_csc_indices = np.array([], dtype=np.int32)
        self.full_jac_csc_indptr = np.zeros(self.total_vars + 1, dtype=np.int32)
        self.full_jac_csc_data = np.array([], dtype=np.float64)
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, 0)
        self.full_jac_standard_slice = slice(0, 0)
        self.full_jac_zero_top_left_slice = slice(0, 0)
        self.full_jac_zero_top_right_slice = slice(0, 0)
        self.full_jac_zero_bottom_left_slice = slice(0, 0)
        self.full_jac_zero_bottom_right_slice = slice(0, 0)

        if self.N_phi == 0:
            return
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
        if self._cache_csr_jacobian_pattern:
            (
                self.full_jac_csr_indices,
                self.full_jac_csr_indptr,
                self.full_jac_raw_to_csr_pos,
            ) = _build_csr_pattern_from_raw_coords(raw_rows, raw_cols, self.total_eq)
        (
            self.full_jac_csc_indices,
            self.full_jac_csc_indptr,
            self.full_jac_raw_to_csc_pos,
        ) = build_compressed_pattern_from_raw_coords(raw_cols, raw_rows, self.total_vars)
        self.full_jac_csr_data = np.empty(self.full_jac_csr_indices.size, dtype=np.float64)
        self.full_jac_csc_data = np.empty(self.full_jac_csc_indices.size, dtype=np.float64)
        if self._cache_csr_jacobian_pattern:
            self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, self.full_jac_csr_data.size)
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, self.full_jac_csc_data.size)

    # --------------------------------------------------------------------------
    # 预处理阶段（精简版）
    # --------------------------------------------------------------------------
    def prepare(self):
        """预处理并确保拓扑/PPC 就绪。

        调用方无需先对 ACPowerNetwork 显式执行 `topo()`；prepare() 会基于
        当前 PPC 数据自动补齐所需的拓扑信息。
        """
        self._prepare_from_ppc()

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
        if self.node_type[a] != AC_NODE_TYPE_SLACK or self.node_type[b] != AC_NODE_TYPE_SLACK:
            return False
        if abs(self.V_spec[a] - self.V_spec[b]) > 1e-10:
            return False
        if abs(self.theta_spec[a] - self.theta_spec[b]) > 1e-10:
            return False
        return True

    def get_f(self, x: np.ndarray) -> np.ndarray:
        """计算残差向量 F"""
        theta, V, phi_re, phi_im = self._extract_state_vars(x)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        return self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)

    def _fill_residual(self, theta, V, phi_re, phi_im, dP, dQ) -> np.ndarray:
        """Fill the preallocated Newton residual using already computed balances."""
        F = self._residual_work
        F.fill(0.0)
        # 方程顺序必须与 get_jacobi() 保持一致：P、Q、零阻抗电压约束、phi参考。
        n_theta = self.n_theta
        n_V = self.n_V

        # 有功 / 无功不平衡量
        F[:n_theta] = dP[self.theta_unknown]
        F[n_theta:n_theta + n_V] = dQ[self.V_unknown]

        # 零阻抗树边的电压约束：cos/sin 块按 stride=2 一次写入，避免 Python 循环。
        a = self._zero_residual_a
        eq_idx = n_theta + n_V
        if a.size:
            cos_theta = self._cache['cos_theta']
            sin_theta = self._cache['sin_theta']
            b = self._zero_residual_b
            n = a.size
            F[eq_idx:eq_idx + 2 * n:2] = V[a] * cos_theta[a] - V[b] * cos_theta[b]
            F[eq_idx + 1:eq_idx + 2 * n:2] = V[a] * sin_theta[a] - V[b] * sin_theta[b]
            eq_idx += 2 * n

        # phi 参考固定
        ref_phi_arr = self._ref_phi_idx_arr
        if ref_phi_arr.size:
            span = 2 * ref_phi_arr.size
            F[eq_idx:eq_idx + span:2] = phi_re[ref_phi_arr]
            F[eq_idx + 1:eq_idx + span:2] = phi_im[ref_phi_arr]

        return F

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
            dP[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

            np.multiply(vm, self.load_qv2, out=values)
            values *= 2.0
            values += self.load_qv1
            dQ[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)
        return dP, dQ

    def _get_standard_jacobi_sparse(self, V: np.ndarray, Sbus=None, *, matrix_format="csr", build_matrix=True):
        """标准P/Q方程对theta/V变量的稀疏雅可比，采用MATPOWER矩阵化公式。"""
        if self.Y_jac_rows.size:
            return self._get_standard_jacobi_direct(
                V,
                Sbus=Sbus,
                matrix_format=matrix_format,
                build_matrix=build_matrix,
            )

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

        jac = vstack((hstack((J11, J12), format='csr'), hstack((J21, J22), format='csr')), format='csr')
        if matrix_format == "csc":
            jac = jac.tocsc()
        return jac if build_matrix else None

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

    def _get_standard_jacobi_direct(self, V: np.ndarray, Sbus=None, *, matrix_format="csr", build_matrix=True):
        """Build the standard P/Q Jacobian directly from Y nonzeros."""
        data = self._fill_standard_jacobian_data(V, Sbus=Sbus)
        shape = (self.n_theta + self.n_V, self.n_theta + self.n_V)
        if matrix_format == "csc":
            if self.standard_jac_raw_to_csc_pos.size:
                apply_raw_sum_plan(self.standard_jac_csc_data, data, self.standard_jac_csc_sum_plan)
            if not build_matrix:
                return self.standard_jac_csc_data
            if not self.standard_jac_csc_indptr.size:
                return csc_matrix(shape, dtype=np.float64)
            return csc_matrix(
                (self.standard_jac_csc_data, self.standard_jac_csc_indices, self.standard_jac_csc_indptr),
                shape=shape,
                copy=False,
            )

        if self.standard_jac_raw_to_csr_pos.size:
            apply_raw_sum_plan(self.standard_jac_csr_data, data, self.standard_jac_csr_sum_plan)
            if not build_matrix:
                return self.standard_jac_csr_data
            return csr_matrix(
                (self.standard_jac_csr_data, self.standard_jac_csr_indices, self.standard_jac_csr_indptr),
                shape=shape,
                copy=False,
            )
        if not build_matrix:
            return self.standard_jac_csr_data
        return coo_matrix(
            (data, (self.standard_jac_rows, self.standard_jac_cols)),
            shape=shape,
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

    def _get_jacobi_from_precomputed_pattern(
        self,
        V,
        phi_re,
        phi_im,
        cos_theta,
        sin_theta,
        Sbus=None,
        *,
        matrix_format="csr",
        build_matrix=True,
    ):
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

        if matrix_format == "csc":
            apply_raw_sum_plan(self.full_jac_csc_data, raw, self.full_jac_csc_sum_plan)
            if not build_matrix:
                return self.full_jac_csc_data
            return csc_matrix(
                (self.full_jac_csc_data, self.full_jac_csc_indices, self.full_jac_csc_indptr),
                shape=(self.total_eq, self.total_vars),
                copy=False,
            )
        if not self.full_jac_raw_to_csr_pos.size:
            return None

        apply_raw_sum_plan(self.full_jac_csr_data, raw, self.full_jac_csr_sum_plan)
        if not build_matrix:
            return self.full_jac_csr_data
        return csr_matrix(
            (self.full_jac_csr_data, self.full_jac_csr_indices, self.full_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )

    def _get_jacobi_from_cached_state(
        self,
        theta,
        V,
        phi_re,
        phi_im,
        Sbus=None,
        *,
        matrix_format="csr",
        build_matrix=True,
    ) -> csr_matrix:
        """Build Jacobian using state arrays already extracted for the current Newton step."""
        cos_theta, sin_theta = self._cache['cos_theta'], self._cache['sin_theta']

        precomputed = self._get_jacobi_from_precomputed_pattern(
            V,
            phi_re,
            phi_im,
            cos_theta,
            sin_theta,
            Sbus=Sbus,
            matrix_format=matrix_format,
            build_matrix=build_matrix,
        )
        if precomputed is not None:
            return precomputed

        J_standard = self._get_standard_jacobi_sparse(
            V,
            Sbus=Sbus,
            matrix_format=matrix_format,
            build_matrix=build_matrix,
        )
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
        jac = vstack(
            (
                hstack((J_standard, J_phi_top), format='csr'),
                hstack((J_zero_left, J_zero_right), format='csr'),
            ),
            format='csr',
        )
        if matrix_format == "csc":
            jac = jac.tocsc()
        return jac if build_matrix else None

    def _build_newton_system(self, x: np.ndarray, *, return_jacobian=True, jacobian_format="csc"):
        """Compute residual and Jacobian together for one Newton iteration."""
        theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        F = self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)
        J = self._get_jacobi_from_cached_state(
            theta,
            V,
            phi_re,
            phi_im,
            Sbus=self._last_Sbus,
            matrix_format=jacobian_format,
            build_matrix=return_jacobian,
        )
        return F, J

    def _should_delegate_acac_to_hybrid(self) -> bool:
        acac = self.ppc.get("acac")
        return acac is not None and getattr(acac, "size", 0) > 0 and int(acac.shape[0]) > 0

    def _run_acac_converter_power_flow(self) -> int:
        from hybrid_lf import (
            DCAC_COLS,
            HybridPowerFlowCalc,
            _LightweightHybridNetwork,
            _lightweight_ac_network,
            _lightweight_dc_network,
        )

        base = self.ppc["base"]
        ac_network = _lightweight_ac_network(self.ppc)
        dc_network = _lightweight_dc_network()
        network = _LightweightHybridNetwork(
            _lf_lightweight=True,
            ac=ac_network,
            dc=dc_network,
            dcac_converters=[],
            acac_converters=ac_network.acac_converters,
            hybrid_islands=[],
        )
        network.ppc = {
            "format": "hybrid_ppc_v1",
            "source": self.ppc.get("source", "<ac_ppc>"),
            "base": base,
            "ac": self.ppc,
            "dc": None,
            "dcac": np.zeros((0, len(DCAC_COLS)), dtype=np.float64),
            "dcac_name": np.asarray([], dtype=object),
            "acac": self.ppc.get("acac", np.zeros((0, len(ACAC_COLS)), dtype=np.float64)),
            "acac_name": self.ppc.get("acac_name", np.asarray([], dtype=object)),
        }
        network._ac_ppc = self.ppc
        network.p_base = float(base["p_base"])
        network.u_scale = float(base["u_scale"])
        network.p_scale = float(base["p_scale"])
        network.i_scale = float(base["i_scale"])
        network.p_base_kW = float(base["p_base_kW"])

        calc = HybridPowerFlowCalc(
            network,
            parameters=self.params,
            keep_node_objects=False,
            linear_solver=self.linear_solver,
            result_mode=self.result_mode,
            verbose=self.verbose,
        )
        rc = calc.run()
        self._delegated_hybrid_calc = calc
        self.converged = bool(calc.converged)
        self.iterations = int(calc.iterations)
        self.normF = float(calc.normF)
        if calc.ac_calc is not None:
            for attr in (
                "N",
                "node_type",
                "ppc_node_idx",
                "ppc_node_name",
                "theta_unknown",
                "V_unknown",
                "slack_node",
                "skipped_islands",
            ):
                if hasattr(calc.ac_calc, attr):
                    setattr(self, attr, getattr(calc.ac_calc, attr))
        ac_result = calc.result.get("ac", {}) if isinstance(calc.result, dict) else {}
        self.result = dict(ac_result)
        acac_table = self.ppc.get("acac", np.zeros((0, len(ACAC_COLS)), dtype=np.float64)).copy()
        acac_result = calc.result.get("acac") if isinstance(calc.result, dict) else None
        if acac_result is not None and getattr(calc, "acac_row_pos", np.array([], dtype=np.int32)).size:
            rows = calc.acac_row_pos
            acac_table[rows, ACAC_COLS["i_p"]] = acac_result[:, 0]
            acac_table[rows, ACAC_COLS["i_q"]] = acac_result[:, 1]
            acac_table[rows, ACAC_COLS["j_p"]] = acac_result[:, 2]
            acac_table[rows, ACAC_COLS["j_q"]] = acac_result[:, 3]
            acac_table[rows, ACAC_COLS["i_i"]] = acac_result[:, 4]
            acac_table[rows, ACAC_COLS["j_i"]] = acac_result[:, 5]
        if self.result_mode != "none":
            self.result["acac"] = acac_table
        if self._network_writeback is not None and self.result_mode not in ("none", "summary"):
            self._write_ppc_result_to_network()
        if self.result_mode not in ("none", "summary", "array") and not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result_from_ppc()
        else:
            self.lf_result = None
        return rc

    # --------------------------------------------------------------------------
    # 迭代求解
    # --------------------------------------------------------------------------
    def run(self, result_mode=None) -> int:
        """执行所选潮流算法。"""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
            self._cache_csr_jacobian_pattern = self.result_mode == "full"
            self.keep_node_objects = False
            self.node_list = []
            self.node_pos = {}
        if self._should_delegate_acac_to_hybrid():
            return self._run_acac_converter_power_flow()
        if self.x.size == 0:
            self.prepare()
        return self._run_newton_raphson()

    def _run_newton_raphson(self) -> int:
        """执行牛顿-拉夫逊迭代求解"""
        self.converged = False
        self.iterations = 0
        x = self.x.copy()

        # 若本实例已把 KLU/UMFPACK 加入黑名单, 直接走 scipy, 避免重复触发失败路径。
        if self._instance_solver_blacklist and self._linear_solver_resolved in self._instance_solver_blacklist:
            self._linear_solver_resolved, self._linear_solver_fn = "scipy", spsolve

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

            try:
                factor = _factor_jacobian(J, self._linear_solver_resolved, self._linear_solver_fn)
                delta = factor.solve(F)
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                # 因子分解或解算失败时, 仅本实例回退到 SuperLU spsolve 单步路径,
                # 避免污染模块级缓存 (同进程内其他 calc 仍可继续尝试 KLU)。
                if self._linear_solver_resolved not in {"scipy", "superlu", "default"}:
                    self._instance_solver_blacklist.add(self._linear_solver_resolved)
                    if self.verbose:
                        print(f"[ac_lf] 可选稀疏求解器 {self._linear_solver_resolved!r} 失败，回退到 scipy: {exc}")
                self._linear_solver_resolved, self._linear_solver_fn = "scipy", spsolve
                delta = spsolve(J, F)
            # 方程定义为 F(x)=0，这里使用 x_new = x - J^{-1}F。
            x -= delta

        # 未收敛
        if self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未收敛")
        self.x = x
        self._write_back()
        return -1

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

    def _build_lf_result(self) -> ACLFResult:
        return self._build_lf_result_from_ppc()

    def _build_lf_result_from_ppc(self) -> ACLFResult:
        # 后处理：把 ppc 风格的 numpy 结果数组转成 dict[name]->SimpleNamespace。
        # 之前是逐行 Python 循环（30k 节点时占总耗时 ~33%），改成
        # 按列向量化抽取 + 单次 searchsorted 做节点电压查表。
        result = ACLFResult()
        result.arrays = dict(self.result)
        bus = self.result.get("bus")
        if bus is None or len(bus) == 0:
            return result

        bus_volt = bus[:, BUS_COLS["voltage"]]
        bus_angle = bus[:, BUS_COLS["angle"]]
        bus_idx_col = bus[:, BUS_COLS["idx"]].astype(np.int64)
        sort_order = np.argsort(bus_idx_col, kind="stable")
        sorted_idx = bus_idx_col[sort_order]

        def _lookup_voltage(col_array):
            """Return bus voltage for an array of node ids (0.0 when missing)."""
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
                # np.arange(...).astype(str).tolist() 比 [str(i) for i in range(n)] 显著更快。
                return np.arange(n).astype(str).tolist()
            if isinstance(names, np.ndarray):
                return names.astype(str).tolist()
            return [str(x) for x in names]

        n_bus = len(bus)
        bus_names = _name_list("bus_name", n_bus)
        result.nodes = {
            name: SimpleNamespace(volt=v, angle=a)
            for name, v, a in zip(bus_names, bus_volt.tolist(), bus_angle.tolist())
        }

        def _build_two_port(rows, name_key, cols, target):
            if rows is None or len(rows) == 0:
                return
            names = _name_list(name_key, len(rows))
            i_p = rows[:, cols["i_p"]].tolist()
            i_q = rows[:, cols["i_q"]].tolist()
            i_c = rows[:, cols["i_c"]].tolist()
            j_p = rows[:, cols["j_p"]].tolist()
            j_q = rows[:, cols["j_q"]].tolist()
            j_c = rows[:, cols["j_c"]].tolist()
            i_v = _lookup_voltage(rows[:, cols["i_node"]]).tolist()
            j_v = _lookup_voltage(rows[:, cols["j_node"]]).tolist()
            for name, ip, iq, ic, iv, jp, jq, jc, jv in zip(
                names, i_p, i_q, i_c, i_v, j_p, j_q, j_c, j_v
            ):
                target[name] = SimpleNamespace(
                    i_p=ip, i_q=iq, i_c=ic, i_v=iv,
                    j_p=jp, j_q=jq, j_c=jc, j_v=jv,
                )

        _build_two_port(self.result.get("branch"), "branch_name", BRANCH_COLS, result.branches)
        _build_two_port(self.result.get("transformer"), "transformer_name", TRANSFORMER_COLS, result.transformers)
        acac_rows = self.result.get("acac")
        if acac_rows is not None and len(acac_rows):
            names = _name_list("acac_name", len(acac_rows))
            i_v = _lookup_voltage(acac_rows[:, ACAC_COLS["i_node"]]).tolist()
            j_v = _lookup_voltage(acac_rows[:, ACAC_COLS["j_node"]]).tolist()
            for name, row, iv, jv in zip(names, acac_rows, i_v, j_v):
                result.acac_converters[name] = SimpleNamespace(
                    i_p=float(row[ACAC_COLS["i_p"]]),
                    i_q=float(row[ACAC_COLS["i_q"]]),
                    i_c=float(row[ACAC_COLS["i_i"]]),
                    i_v=iv,
                    j_p=float(row[ACAC_COLS["j_p"]]),
                    j_q=float(row[ACAC_COLS["j_q"]]),
                    j_c=float(row[ACAC_COLS["j_i"]]),
                    j_v=jv,
                )

        def _build_single_port(rows, name_key, p_col, q_col, c_col, node_col, target):
            if rows is None or len(rows) == 0:
                return
            names = _name_list(name_key, len(rows))
            i_p = rows[:, p_col].tolist()
            i_q = rows[:, q_col].tolist()
            i_c = rows[:, c_col].tolist()
            i_v = _lookup_voltage(rows[:, node_col]).tolist()
            for name, p, q, c, v in zip(names, i_p, i_q, i_c, i_v):
                target[name] = SimpleNamespace(i_p=p, i_q=q, i_c=c, i_v=v)

        _build_single_port(
            self.result.get("zero_branch"), "zero_branch_name",
            ZERO_BRANCH_COLS["p"], ZERO_BRANCH_COLS["q"],
            ZERO_BRANCH_COLS["current"], ZERO_BRANCH_COLS["i_node"],
            result.zero_branches,
        )
        _build_single_port(
            self.result.get("break"), "break_name",
            SWITCH_COLS["p"], SWITCH_COLS["q"],
            SWITCH_COLS["current"], SWITCH_COLS["i_node"],
            result.breakers,
        )
        _build_single_port(
            self.result.get("gen"), "gen_name",
            GEN_COLS["p"], GEN_COLS["q"],
            GEN_COLS["current"], GEN_COLS["node"],
            result.generators,
        )
        _build_single_port(
            self.result.get("load"), "load_name",
            LOAD_COLS["p"], LOAD_COLS["q"],
            LOAD_COLS["current"], LOAD_COLS["node"],
            result.loads,
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
        copy_fields(
            getattr(network, "acac_converters", []),
            rows_by_idx("acac", ACAC_COLS["idx"]),
            (
                ("i_p", ACAC_COLS["i_p"]),
                ("i_q", ACAC_COLS["i_q"]),
                ("i_i", ACAC_COLS["i_i"]),
                ("j_p", ACAC_COLS["j_p"]),
                ("j_q", ACAC_COLS["j_q"]),
                ("j_i", ACAC_COLS["j_i"]),
            ),
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
        if self.N_phi > 0 and self.zero_a.size:
            I_ab = (phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]) + 1j * (
                phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]
            )
            Sa = Vc[self.zero_a] * np.conj(I_ab)
            Sb = Vc[self.zero_b] * np.conj(-I_ab)
            P_zero = np.bincount(self.zero_a, weights=Sa.real, minlength=self.N)
            P_zero += np.bincount(self.zero_b, weights=Sb.real, minlength=self.N)
            Q_zero = np.bincount(self.zero_a, weights=Sa.imag, minlength=self.N)
            Q_zero += np.bincount(self.zero_b, weights=Sb.imag, minlength=self.N)
        else:
            I_ab = np.array([], dtype=np.complex128)
            P_zero = np.zeros(self.N, dtype=np.float64)
            Q_zero = np.zeros(self.N, dtype=np.float64)

        P_gen = S_y.real + P_zero + P_load
        Q_gen = S_y.imag + Q_zero + Q_load
        external_p = self._external_ac_p_injection
        external_q = self._external_ac_q_injection
        if external_p is not None:
            external_p = np.asarray(external_p, dtype=np.float64)
            if external_p.shape == P_gen.shape:
                P_gen = P_gen + external_p
        if external_q is not None:
            external_q = np.asarray(external_q, dtype=np.float64)
            if external_q.shape == Q_gen.shape:
                Q_gen = Q_gen + external_q

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
            rows = self.ppc_shunt_rows
            vm = V[self.ppc_shunt_pos]
            vm2 = vm * vm
            control = shunt[rows, SHUNT_COLS["control_type"]].astype(np.int32, copy=False)
            g_set = shunt[rows, SHUNT_COLS["g_set"]]
            b_set = shunt[rows, SHUNT_COLS["b_set"]]
            y_mask = (control == SHUNT_B) | (control == SHUNT_Z) | (g_set != 0.0)
            p = np.zeros(rows.size, dtype=np.float64)
            q = np.zeros(rows.size, dtype=np.float64)
            p[y_mask] = vm2[y_mask] * g_set[y_mask]
            q[y_mask] = -vm2[y_mask] * b_set[y_mask]
            q_mask = (~y_mask) & (control == SHUNT_Q)
            q[q_mask] = shunt[rows[q_mask], SHUNT_COLS["q_set"]]
            current = np.divide(
                np.hypot(p, q),
                vm,
                out=np.zeros_like(p),
                where=vm > self.min_voltage,
            )
            shunt[rows, SHUNT_COLS["p"]] = p
            shunt[rows, SHUNT_COLS["q"]] = q
            shunt[rows, SHUNT_COLS["current"]] = current

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
            "acac": self.ppc.get("acac", np.zeros((0, len(ACAC_COLS)), dtype=np.float64)).copy(),
        }

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
    slack_node_ids = set(calc.ppc_node_idx[calc.node_type == AC_NODE_TYPE_SLACK].tolist())
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
    parser.add_argument("--linear-solver", default="pyklu")
    parser.add_argument("--result-mode", choices=("full", "array", "summary", "none"), default="full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ppc = load_ac_ppc_from_e_file(args.file)
    calc = ACPowerFlowCalc(
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
        print_ac_result(calc, rc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
