import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
LFCORE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, LFCORE_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ac_lf import ACLFResult, ACPowerFlowCalc, _device_key as _lf_device_key, coo_matrix, csr_matrix, solve_sparse_system
from dc_lf import DCLFResult, DCPowerFlowCalc
from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from hybrid_model import ACAC_CONTROL_TYPES, HybridPowerNetwork
from ac_array_model import build_ac_network_from_ppc
from dc_array_model import build_dc_network_from_ppc
from hybrid_array_model import (
    ACAC_COLS,
    ACAC_CONTROL_LABEL,
    DCAC_COLS,
    DCAC_CONTROL_LABEL,
    build_hybrid_ppc_from_e_file,
)


DEFAULT_HYBRID_EFILE = model_file("hybrid", "hybrid_net_40.e")
LINEAR_SOLVER_CHOICES = (
    "scipy",
    "superlu",
    "default",
    "auto",
    "pypardiso",
    "umfpack",
    "sksparse.klu.klu_solve",
    "klu_solve",
    "pyklu",
    "klu",
    "klu_alt",
)
LINEAR_SOLVER_HELP = (
    "Sparse linear solver for the unified Newton system and AC/DC sub-solvers. "
    "scipy/superlu/default use SciPy SuperLU; auto tries optional UMFPACK/KLU bindings first."
)


def _array_device(idx, name=None, **values):
    return SimpleNamespace(idx=int(idx), name=str(name if name is not None else idx), **values)


def _build_lf_ac_network(ac_ppc):
    return build_ac_network_from_ppc(ac_ppc)


def _build_lf_dc_network(dc_ppc):
    return build_dc_network_from_ppc(dc_ppc)


def _build_lf_converters(ppc):
    dcac = [
        _array_device(
            row[DCAC_COLS["idx"]],
            ppc["dcac_name"][pos],
            ac_node=int(row[DCAC_COLS["ac_node"]]),
            dc_node=int(row[DCAC_COLS["dc_node"]]),
            r1=float(row[DCAC_COLS["r1"]]),
            r2=float(row[DCAC_COLS["r2"]]),
            control_type=DCAC_CONTROL_LABEL.get(int(row[DCAC_COLS["control_type"]]), "DCV"),
            p_ac_set=float(row[DCAC_COLS["p_ac_set"]]),
            q_ac_set=float(row[DCAC_COLS["q_ac_set"]]),
            v_ac_set=float(row[DCAC_COLS["v_ac_set"]]),
            v_dc_set=float(row[DCAC_COLS["v_dc_set"]]),
            run_stat=int(row[DCAC_COLS["run_stat"]]),
            dc_p=float(row[DCAC_COLS["dc_p"]]),
            ac_p=float(row[DCAC_COLS["ac_p"]]),
            ac_q=float(row[DCAC_COLS["ac_q"]]),
            dc_i=float(row[DCAC_COLS["dc_i"]]),
            ac_i=float(row[DCAC_COLS["ac_i"]]),
            ac_node_obj=None,
            dc_node_obj=None,
        )
        for pos, row in enumerate(ppc["dcac"])
    ]
    acac = [
        _array_device(
            row[ACAC_COLS["idx"]],
            ppc["acac_name"][pos],
            i_node=int(row[ACAC_COLS["i_node"]]),
            j_node=int(row[ACAC_COLS["j_node"]]),
            r1=float(row[ACAC_COLS["r1"]]),
            r2=float(row[ACAC_COLS["r2"]]),
            control_type=ACAC_CONTROL_LABEL.get(int(row[ACAC_COLS["control_type"]]), "PQQ"),
            p_set=float(row[ACAC_COLS["p_set"]]),
            i_q_set=float(row[ACAC_COLS["i_q_set"]]),
            j_q_set=float(row[ACAC_COLS["j_q_set"]]),
            i_v_set=float(row[ACAC_COLS["i_v_set"]]),
            j_v_set=float(row[ACAC_COLS["j_v_set"]]),
            run_stat=int(row[ACAC_COLS["run_stat"]]),
            i_p=float(row[ACAC_COLS["i_p"]]),
            i_q=float(row[ACAC_COLS["i_q"]]),
            j_p=float(row[ACAC_COLS["j_p"]]),
            j_q=float(row[ACAC_COLS["j_q"]]),
            i_i=float(row[ACAC_COLS["i_i"]]),
            j_i=float(row[ACAC_COLS["j_i"]]),
            i_node_obj=None,
            j_node_obj=None,
        )
        for pos, row in enumerate(ppc["acac"])
    ]
    return dcac, acac


def _read_lf_network_from_file(file_name) -> HybridPowerNetwork:
    network, ppc = build_hybrid_ppc_from_e_file(file_name)
    network.ppc = ppc
    network._ac_ppc = ppc["ac"]
    network._dc_ppc = ppc["dc"]
    network.ac.ppc = ppc["ac"]
    network.dc.ppc = ppc["dc"]
    base = ppc["base"]
    network.p_base = float(base[0])
    network.u_scale = float(base[1])
    network.p_scale = float(base[2])
    network.i_scale = float(base[3])
    network.p_base_kW = float(base[4])
    return network


@dataclass
class DCACLFResult:
    dcac_converters: dict = field(default_factory=dict)


@dataclass
class ACACLFResult:
    acac_converters: dict = field(default_factory=dict)


@dataclass
class HybridLFResult:
    network: Optional[HybridPowerNetwork] = None
    ac_network: Any = None
    dc_network: Any = None
    calc: Optional["HybridPowerFlowCalc"] = None
    ac_calc: Optional[ACPowerFlowCalc] = None
    dc_calc: Optional[DCPowerFlowCalc] = None
    rc: int = -1
    ac_warnings: List[str] = field(default_factory=list)
    ac_errors: List[str] = field(default_factory=list)
    dc_warnings: List[str] = field(default_factory=list)
    dc_errors: List[str] = field(default_factory=list)
    ac: Optional[ACLFResult] = None
    dc: Optional[DCLFResult] = None
    dcac: DCACLFResult = field(default_factory=DCACLFResult)
    acac: ACACLFResult = field(default_factory=ACACLFResult)

    @property
    def lf_result(self) -> "HybridLFResult":
        return self

    @property
    def total_nodes(self) -> int:
        return 0 if self.network is None else self.network.total_nodes

    @property
    def converged(self) -> bool:
        return (
            self.rc == 0
            and self.calc is not None
            and self.calc.converged
            and not self.ac_errors
            and not self.dc_errors
        )

    @property
    def global_jacobian_shape(self) -> Tuple[int, int]:
        return (0, 0) if self.calc is None else self.calc.last_jacobian_shape

    @property
    def has_ac(self) -> bool:
        return self.ac_network is not None and len(self.ac_network.nodes) > 0

    @property
    def has_dc(self) -> bool:
        return self.dc_network is not None and len(self.dc_network.nodes) > 0

    @property
    def has_dcac(self) -> bool:
        return self.network is not None and len(self.network.dcac_converters) > 0

    @property
    def has_acac(self) -> bool:
        return self.network is not None and len(self.network.acac_converters) > 0

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
        linear_solver: str = "scipy",
        result_mode: str = "full",
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
        self.linear_solver = str(linear_solver or "scipy").strip().lower()
        self.result_mode = self._normalize_result_mode(result_mode)
        self.has_ac = len(network.ac.nodes) > 0
        self.has_dc = len(network.dc.nodes) > 0
        self.ac_calc = (
            ACPowerFlowCalc(
                network._ac_ppc,
                parameters=self.params,
                linear_solver=self.linear_solver,
                result_mode=self.result_mode,
                verbose=self.verbose,
            )
            if self.has_ac and hasattr(network, "_ac_ppc")
            else ACPowerFlowCalc(
                network.ac,
                parameters=self.params,
                linear_solver=self.linear_solver,
                result_mode=self.result_mode,
                verbose=self.verbose,
            ) if self.has_ac else None
        )
        if self.has_dc and hasattr(network, "_dc_ppc"):
            self.dc_calc = DCPowerFlowCalc(
                network._dc_ppc,
                parameters=self.params,
                linear_solver=self.linear_solver,
                writeback_network=network.dc,
                result_mode=self.result_mode,
            )
        else:
            self.dc_calc = DCPowerFlowCalc(network.dc, parameters=self.params, linear_solver=self.linear_solver, result_mode=self.result_mode) if self.has_dc else None
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
        self._residual_work = np.array([], dtype=np.float64)
        self._ac_node_obj_by_idx = {int(node.idx): node for node in getattr(network.ac, "nodes", [])}
        self._dc_node_obj_by_idx = {int(node.idx): node for node in getattr(network.dc, "nodes", [])}
        self._clear_dcac_arrays()
        self._clear_acac_arrays()
        self._clear_converter_jacobian_structure()
        self._clear_global_jacobian_pattern()
        self.lf_result = None

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
            raise ValueError(f"Unsupported Hybrid result_mode: {result_mode!r}")
        return aliases[mode]

    def prepare(self):
        """Build the global hybrid state vector and block equation layout."""
        parts = []
        if self.ac_calc is not None:
            self.ac_calc.verbose = self.verbose
            self.ac_calc.prepare()
            self.ac_size = self.ac_calc.total_vars
            self.ac_eq = self.ac_calc.total_eq
            parts.append(self.ac_calc.x.copy())
        if self.dc_calc is not None:
            self.dc_G, dc_x = self.dc_calc.prepare()
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
        self._residual_work = np.empty(self.total_eq, dtype=np.float64)
        self._cache_converter_jacobian_structure()
        self._cache_global_jacobian_pattern()
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
            conv.is_alive = False
            if conv.run_stat != 1:
                continue
            if conv.ac_node not in self.ac_calc.node_pos:
                continue
            if conv.dc_node not in self.dc_calc.alive_node_dict:
                continue
            conv.ac_node_obj = self._ac_node_obj_by_idx.get(int(conv.ac_node))
            conv.dc_node_obj = self._dc_node_obj_by_idx.get(int(conv.dc_node))
            ac_pos = self.ac_calc.node_pos[conv.ac_node]
            if self.ac_calc.node_type[ac_pos] != "PQ":
                raise ValueError(f"DCACConverter[{conv.idx}] 的 AC 节点必须是 PQ 节点，当前为 {self.ac_calc.node_type[ac_pos]}")
            dc_pos = self.dc_calc.alive_node_dict[conv.dc_node]
            ctrl = str(conv.control_type).upper()
            if ctrl not in {"DCV", "ACV", "ACP"}:
                raise ValueError(f"未知 DCACConverter 控制模式: {conv.control_type}")
            conv.is_alive = True
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
            conv.is_alive = False
            if conv.run_stat != 1:
                continue
            if conv.i_node not in self.ac_calc.node_pos:
                continue
            if conv.j_node not in self.ac_calc.node_pos:
                continue
            conv.i_node_obj = self._ac_node_obj_by_idx.get(int(conv.i_node))
            conv.j_node_obj = self._ac_node_obj_by_idx.get(int(conv.j_node))
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
            conv.is_alive = True
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

    def _clear_global_jacobian_pattern(self):
        self.global_jac_raw_data = np.array([], dtype=np.float64)
        self.global_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.global_jac_csr_indices = np.array([], dtype=np.int32)
        self.global_jac_csr_indptr = np.array([], dtype=np.int32)
        self.global_jac_csr_data = np.array([], dtype=np.float64)
        self.global_jac_ac_slice = slice(0, 0)
        self.global_jac_dc_slice = slice(0, 0)
        self.global_jac_dcac_ac_p_slice = slice(0, 0)
        self.global_jac_dcac_ac_q_slice = slice(0, 0)
        self.global_jac_dcac_dc_eq_slice = slice(0, 0)
        self.global_jac_dcac_loss_slice = slice(0, 0)
        self.global_jac_dcac_ctrl_q_slice = slice(0, 0)
        self.global_jac_dcac_ctrl_dcv_slice = slice(0, 0)
        self.global_jac_dcac_ctrl_acv_slice = slice(0, 0)
        self.global_jac_dcac_ctrl_acp_slice = slice(0, 0)
        self.global_jac_acac_i_p_slice = slice(0, 0)
        self.global_jac_acac_i_q_slice = slice(0, 0)
        self.global_jac_acac_j_p_slice = slice(0, 0)
        self.global_jac_acac_j_q_slice = slice(0, 0)
        self.global_jac_acac_loss_slice = slice(0, 0)
        self.global_jac_acac_ctrl_p_slice = slice(0, 0)
        self.global_jac_acac_ctrl_q_i_slice = slice(0, 0)
        self.global_jac_acac_ctrl_v_i_slice = slice(0, 0)
        self.global_jac_acac_ctrl_q_j_slice = slice(0, 0)
        self.global_jac_acac_ctrl_v_j_slice = slice(0, 0)

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

    @staticmethod
    def _rows_from_csr_indptr(indptr):
        return np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))

    def _sub_jacobian_pattern(self, calc, is_dc=False):
        if calc is None or getattr(calc, "total_eq", 0) == 0 or getattr(calc, "total_vars", 0) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        if is_dc and hasattr(calc, "_dc_jac_csr_indices") and calc._dc_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc._dc_jac_csr_indptr),
                calc._dc_jac_csr_indices.astype(np.int32, copy=True),
            )
        if hasattr(calc, "full_jac_csr_indices") and calc.full_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc.full_jac_csr_indptr),
                calc.full_jac_csr_indices.astype(np.int32, copy=True),
            )
        if hasattr(calc, "standard_jac_csr_indices") and calc.standard_jac_csr_indices.size:
            return (
                self._rows_from_csr_indptr(calc.standard_jac_csr_indptr),
                calc.standard_jac_csr_indices.astype(np.int32, copy=True),
            )
        jac = calc.get_jacobi(calc.x) if not is_dc else calc.get_jacobi(self.dc_G, calc.x)
        jac = jac.tocsr()
        return self._rows_from_csr_indptr(jac.indptr), jac.indices.astype(np.int32, copy=True)

    def _cache_global_jacobian_pattern(self):
        """Precompute global hybrid Jacobian CSR pattern for repeated Newton iterations."""
        self._clear_global_jacobian_pattern()
        if self.total_eq == 0 or self.total_vars == 0:
            return

        rows_parts = []
        cols_parts = []
        raw_count = 0

        def add_part(name, rows, cols):
            nonlocal raw_count
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            if rows.size != cols.size:
                raise ValueError(f"Hybrid Jacobian pattern part {name!r} has mismatched row/column lengths")
            part_slice = slice(raw_count, raw_count + rows.size)
            setattr(self, f"global_jac_{name}_slice", part_slice)
            raw_count += rows.size
            if rows.size:
                rows_parts.append(rows)
                cols_parts.append(cols)

        if self.ac_calc is not None:
            ac_rows, ac_cols = self._sub_jacobian_pattern(self.ac_calc, is_dc=False)
            add_part("ac", ac_rows, ac_cols)
        if self.dc_calc is not None:
            dc_rows, dc_cols = self._sub_jacobian_pattern(self.dc_calc, is_dc=True)
            add_part("dc", dc_rows + self.ac_eq, dc_cols + self.ac_size)

        if self.N_dcac:
            add_part("dcac_ac_p", self.dcac_ac_p_row, self.dcac_ac_p_col)
            add_part("dcac_ac_q", self.dcac_ac_q_row, self.dcac_ac_q_col)
            add_part("dcac_dc_eq", self.dcac_dc_eq_rows, self.dcac_dc_eq_cols)
            add_part("dcac_loss", self.dcac_loss_rows, self.dcac_loss_cols)
            add_part("dcac_ctrl_q", self.dcac_eq_ctrl_2, self.dcac_ac_q_col)
            add_part("dcac_ctrl_dcv", self.dcac_eq_ctrl_1[self.dcac_ctrl_dc_v_mask], self.dcac_dc_v_col[self.dcac_ctrl_dc_v_mask])
            add_part("dcac_ctrl_acv", self.dcac_eq_ctrl_1[self.dcac_ctrl_ac_v_mask], self.dcac_ac_v_col[self.dcac_ctrl_ac_v_mask])
            add_part("dcac_ctrl_acp", self.dcac_eq_ctrl_1[self.dcac_ctrl_ac_p_mask], self.dcac_ac_p_col[self.dcac_ctrl_ac_p_mask])

        if self.N_acac:
            add_part("acac_i_p", self.acac_i_p_row, self.acac_i_p_col)
            add_part("acac_i_q", self.acac_i_q_row, self.acac_i_q_col)
            add_part("acac_j_p", self.acac_j_p_row, self.acac_j_p_col)
            add_part("acac_j_q", self.acac_j_q_row, self.acac_j_q_col)
            add_part("acac_loss", self.acac_loss_rows, self.acac_loss_cols)
            add_part("acac_ctrl_p", self.acac_eq_ctrl_1, self.acac_i_p_col)
            add_part("acac_ctrl_q_i", self.acac_eq_ctrl_2[self.acac_q_i_mask], self.acac_i_q_col[self.acac_q_i_mask])
            add_part("acac_ctrl_v_i", self.acac_eq_ctrl_2[self.acac_v_i_mask], self.acac_i_v_col[self.acac_v_i_mask])
            add_part("acac_ctrl_q_j", self.acac_eq_ctrl_3[self.acac_q_j_mask], self.acac_j_q_col[self.acac_q_j_mask])
            add_part("acac_ctrl_v_j", self.acac_eq_ctrl_3[self.acac_v_j_mask], self.acac_j_v_col[self.acac_v_j_mask])

        self.global_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            self.global_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        pattern = coo_matrix(
            (np.ones(raw_count, dtype=np.float64), (raw_rows, raw_cols)),
            shape=(self.total_eq, self.total_vars),
        ).tocsr()
        pattern.sum_duplicates()
        self.global_jac_csr_indices = pattern.indices.astype(np.int32, copy=True)
        self.global_jac_csr_indptr = pattern.indptr.astype(np.int32, copy=True)
        self.global_jac_csr_data = np.empty(self.global_jac_csr_indices.size, dtype=np.float64)

        positions = {}
        for row in range(self.total_eq):
            start = int(self.global_jac_csr_indptr[row])
            end = int(self.global_jac_csr_indptr[row + 1])
            for pos in range(start, end):
                positions[(row, int(self.global_jac_csr_indices[pos]))] = pos
        self.global_jac_raw_to_csr_pos = np.fromiter(
            (positions[(int(row), int(col))] for row, col in zip(raw_rows, raw_cols)),
            dtype=np.intp,
            count=raw_count,
        )

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

    def _append_converter_residuals(self, parts, ac_f, dc_f, dcac_x, acac_x, ac_V, dc_V):
        """Inject converter port powers and append converter equation residuals."""
        if self.N_dcac:
            parts.append(self._append_dcac_residuals(ac_f, dc_f, dcac_x, ac_V, dc_V))
        if self.N_acac:
            parts.append(self._append_acac_residuals(ac_f, acac_x, ac_V))

    def _append_dcac_residuals(self, ac_f, dc_f, dcac_x, ac_V, dc_V, out=None):
        """Mutate AC/DC nodal residuals and return DC/AC converter residual rows."""
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
        dcac_f = out if out is not None else np.empty(self.N_dcac * 3, dtype=np.float64)
        # r1+r2 converter loss equation in per-unit power/voltage variables.
        dcac_f[0::3] = (
            vd2 * va2 * (dc_p + ac_p)
            - self.dcac_r1 * dc_p * dc_p * va2
            - self.dcac_r2 * (ac_p * ac_p + ac_q * ac_q) * vd2
        )
        f_ctrl = dcac_f[1::3]
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
        return dcac_f

    def _append_acac_residuals(self, ac_f, acac_x, ac_V, out=None):
        """Mutate AC nodal residuals and return AC/AC converter residual rows."""
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
        acac_f = out if out is not None else np.empty(self.N_acac * 4, dtype=np.float64)
        acac_f[0::4] = (
            vi2 * vj2 * (i_p + j_p)
            - self.acac_r1 * (i_p * i_p + i_q * i_q) * vj2
            - self.acac_r2 * (j_p * j_p + j_q * j_q) * vi2
        )
        acac_f[1::4] = i_p - self.acac_p_set
        f2 = acac_f[2::4]
        f3 = acac_f[3::4]
        f2[self.acac_q_i_mask] = i_q[self.acac_q_i_mask] - self.acac_i_q_set[self.acac_q_i_mask]
        f2[self.acac_v_i_mask] = vi[self.acac_v_i_mask] - self.acac_i_v_set[self.acac_v_i_mask]
        f3[self.acac_q_j_mask] = j_q[self.acac_q_j_mask] - self.acac_j_q_set[self.acac_q_j_mask]
        f3[self.acac_v_j_mask] = vj[self.acac_v_j_mask] - self.acac_j_v_set[self.acac_v_j_mask]
        acac_f[2::4] = f2
        acac_f[3::4] = f3
        return acac_f

    def _fill_residual_work(self, ac_f, dc_f, dcac_x, acac_x, ac_V, dc_V):
        """Fill the preallocated global residual vector and return it."""
        if self._residual_work.size != self.total_eq:
            self._residual_work = np.empty(self.total_eq, dtype=np.float64)
        F = self._residual_work
        ac_view = None
        dc_view = None
        if self.ac_eq:
            ac_view = F[:self.ac_eq]
            ac_view[:] = ac_f
        if self.dc_eq:
            dc_view = F[self.ac_eq:self.ac_eq + self.dc_eq]
            dc_view[:] = dc_f
        if self.N_dcac:
            dcac_view = F[self.dcac_eq_start:self.acac_eq_start]
            self._append_dcac_residuals(ac_view, dc_view, dcac_x, ac_V, dc_V, out=dcac_view)
        if self.N_acac:
            acac_view = F[self.acac_eq_start:self.total_eq]
            self._append_acac_residuals(ac_view, acac_x, ac_V, out=acac_view)
        return F

    def get_f(self, x):
        """Assemble global residuals for AC, DC, DCAC and ACAC equations."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        ac_f = None
        dc_f = None
        if self.ac_calc is not None:
            ac_f = self.ac_calc.get_f(ac_x)
        if self.dc_calc is not None:
            dc_f = self.dc_calc.get_f(dc_x)
        ac_V = dc_V = None
        if self.N_dcac or self.N_acac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        return self._fill_residual_work(ac_f, dc_f, dcac_x, acac_x, ac_V, dc_V)

    def get_jacobi(self, x):
        """Build the global sparse Jacobian from sub-solver blocks plus converter couplings."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        ac_j = self.ac_calc.get_jacobi(ac_x) if self.ac_calc is not None else None
        dc_j = self.dc_calc.get_jacobi(self.dc_G, dc_x) if self.dc_calc is not None else None
        return self._assemble_jacobian(ac_x, dc_x, dcac_x, acac_x, ac_j, dc_j)

    @staticmethod
    def _slice_len(part_slice):
        return int(part_slice.stop - part_slice.start)

    def _fill_dcac_global_jacobian_data(self, raw, dcac_x, ac_V, dc_V):
        if not self.N_dcac:
            return
        dcac = dcac_x.reshape(self.N_dcac, 3)
        dc_p = dcac[:, 0]
        ac_p = dcac[:, 1]
        ac_q = dcac[:, 2]

        for part_slice in (
            self.global_jac_dcac_ac_p_slice,
            self.global_jac_dcac_ac_q_slice,
            self.global_jac_dcac_dc_eq_slice,
            self.global_jac_dcac_ctrl_q_slice,
            self.global_jac_dcac_ctrl_dcv_slice,
            self.global_jac_dcac_ctrl_acv_slice,
            self.global_jac_dcac_ctrl_acp_slice,
        ):
            if self._slice_len(part_slice):
                raw[part_slice] = 1.0

        va = ac_V[self.dcac_ac_pos]
        vd = dc_V[self.dcac_dc_pos]
        va2 = va * va
        vd2 = vd * vd
        dc_p2 = dc_p * dc_p
        ac_i2_num = ac_p * ac_p + ac_q * ac_q
        data = raw[self.global_jac_dcac_loss_slice]
        data[0::5] = vd2 * va2 - 2.0 * self.dcac_r1 * dc_p * va2
        data[1::5] = vd2 * va2 - 2.0 * self.dcac_r2 * ac_p * vd2
        data[2::5] = -2.0 * self.dcac_r2 * ac_q * vd2
        data[3::5] = 2.0 * va * vd2 * (dc_p + ac_p) - 2.0 * self.dcac_r1 * dc_p2 * va
        data[4::5] = 2.0 * vd * va2 * (dc_p + ac_p) - 2.0 * self.dcac_r2 * ac_i2_num * vd

    def _fill_acac_global_jacobian_data(self, raw, acac_x, ac_V):
        if not self.N_acac:
            return
        acac = acac_x.reshape(self.N_acac, 4)
        i_p = acac[:, 0]
        i_q = acac[:, 1]
        j_p = acac[:, 2]
        j_q = acac[:, 3]

        for part_slice in (
            self.global_jac_acac_i_p_slice,
            self.global_jac_acac_i_q_slice,
            self.global_jac_acac_j_p_slice,
            self.global_jac_acac_j_q_slice,
            self.global_jac_acac_ctrl_p_slice,
            self.global_jac_acac_ctrl_q_i_slice,
            self.global_jac_acac_ctrl_v_i_slice,
            self.global_jac_acac_ctrl_q_j_slice,
            self.global_jac_acac_ctrl_v_j_slice,
        ):
            if self._slice_len(part_slice):
                raw[part_slice] = 1.0

        vi = ac_V[self.acac_i_pos]
        vj = ac_V[self.acac_j_pos]
        vi2 = vi * vi
        vj2 = vj * vj
        i_s2 = i_p * i_p + i_q * i_q
        j_s2 = j_p * j_p + j_q * j_q
        data = raw[self.global_jac_acac_loss_slice]
        data[0::6] = vi2 * vj2 - 2.0 * self.acac_r1 * i_p * vj2
        data[1::6] = -2.0 * self.acac_r1 * i_q * vj2
        data[2::6] = vi2 * vj2 - 2.0 * self.acac_r2 * j_p * vi2
        data[3::6] = -2.0 * self.acac_r2 * j_q * vi2
        data[4::6] = 2.0 * vi * vj2 * (i_p + j_p) - 2.0 * self.acac_r2 * j_s2 * vi
        data[5::6] = 2.0 * vj * vi2 * (i_p + j_p) - 2.0 * self.acac_r1 * i_s2 * vj

    def _assemble_jacobian_from_precomputed_pattern(self, ac_x, dc_x, dcac_x, acac_x, ac_j=None, dc_j=None, ac_V=None, dc_V=None):
        if self.global_jac_raw_data.size == 0:
            return None
        raw = self.global_jac_raw_data
        if ac_j is not None and self._slice_len(self.global_jac_ac_slice):
            raw[self.global_jac_ac_slice] = ac_j.data
        if dc_j is not None and self._slice_len(self.global_jac_dc_slice):
            raw[self.global_jac_dc_slice] = dc_j.data

        if (self.N_dcac or self.N_acac) and (ac_V is None or (self.N_dcac and dc_V is None)):
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)
        self._fill_dcac_global_jacobian_data(raw, dcac_x, ac_V, dc_V)
        self._fill_acac_global_jacobian_data(raw, acac_x, ac_V)

        self.global_jac_csr_data.fill(0.0)
        np.add.at(self.global_jac_csr_data, self.global_jac_raw_to_csr_pos, raw)
        jac = csr_matrix(
            (self.global_jac_csr_data, self.global_jac_csr_indices, self.global_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )
        self.last_jacobian_shape = jac.shape
        return jac

    def _assemble_jacobian(self, ac_x, dc_x, dcac_x, acac_x, ac_j=None, dc_j=None, ac_V=None, dc_V=None):
        """Build the global sparse Jacobian from prepared sub-solver blocks."""
        precomputed = self._assemble_jacobian_from_precomputed_pattern(
            ac_x,
            dc_x,
            dcac_x,
            acac_x,
            ac_j=ac_j,
            dc_j=dc_j,
            ac_V=ac_V,
            dc_V=dc_V,
        )
        if precomputed is not None:
            return precomputed

        row_parts = []
        col_parts = []
        data_parts = []
        if ac_j is not None:
            ac_coo = ac_j.tocoo()
            row_parts.append(ac_coo.row)
            col_parts.append(ac_coo.col)
            data_parts.append(ac_coo.data)
        target_shape = (self.total_eq, self.total_vars)
        if dc_j is not None:
            dc_coo = dc_j.tocoo()
            row_parts.append(dc_coo.row + self.ac_eq)
            col_parts.append(dc_coo.col + self.ac_size)
            data_parts.append(dc_coo.data)

        if (self.N_dcac or self.N_acac) and (ac_V is None or (self.N_dcac and dc_V is None)):
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

    def _build_newton_system(self, x):
        """Build residual and Jacobian together, reusing AC/DC sub-solver caches."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        ac_f = ac_j = None
        dc_f = dc_j = None
        if self.ac_calc is not None:
            ac_f, ac_j = self.ac_calc._build_newton_system(ac_x)
        if self.dc_calc is not None:
            dc_f, dc_j = self.dc_calc._build_newton_system(self.dc_G, dc_x)

        ac_V = dc_V = None
        if self.N_dcac or self.N_acac:
            _, ac_V, dc_V = self._cached_state_values(ac_x, dc_x)

        F = self._fill_residual_work(ac_f, dc_f, dcac_x, acac_x, ac_V, dc_V)
        J = self._assemble_jacobian(ac_x, dc_x, dcac_x, acac_x, ac_j, dc_j, ac_V, dc_V)
        return F, J

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

    def run(self, result_mode=None):
        """Execute unified Newton iterations over the full hybrid state vector."""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
        if self.x.size == 0:
            self.prepare()

        self.converged = False
        x = self.x.copy()
        for it in range(self.max_iter):
            F, J = self._build_newton_system(x)
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
                self._finish_result(x)
                return 0

            delta = solve_sparse_system(J, -F, self.linear_solver)
            x += delta

        self.x = x
        self._finish_result(x)
        return -1

    def _finish_result(self, x):
        if self.result_mode == "full":
            self._write_back(x)
        elif self.result_mode == "summary":
            self._write_summary_result(x)
        else:
            self._write_none_result(x)

    def _write_none_result(self, x):
        ac_x, dc_x, _dcac_x, _acac_x = self._split_x(x)
        if self.ac_calc is not None:
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc.result = {}
            self.ac_calc.lf_result = None
        if self.dc_calc is not None:
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.result = {}
            self.dc_calc.lf_result = None
        self.lf_result = None

    def _write_summary_result(self, x):
        ac_x, dc_x, _dcac_x, _acac_x = self._split_x(x)
        ac_result = None
        dc_result = None
        if self.ac_calc is not None:
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            self.ac_calc.iterations = self.iterations
            self.ac_calc._write_summary_result()
            ac_result = self.ac_calc.result
        if self.dc_calc is not None:
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            self.dc_calc.iterations = self.iterations
            self.dc_calc._write_summary_result(dc_x)
            dc_result = self.dc_calc.result
        self.lf_result = {
            "ac": ac_result,
            "dc": dc_result,
            "hybrid": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
                "total_vars": int(self.total_vars),
                "total_eq": int(self.total_eq),
            },
        }

    def _write_ac_ppc_result_to_network(self) -> None:
        """Copy array-mode AC results back to the hybrid AC object facade."""
        from ac_array_model import BRANCH_COLS, BUS_COLS, GEN_COLS, LOAD_COLS, SHUNT_COLS, TRANSFORMER_COLS

        result = self.ac_calc.result
        if not result:
            return
        if getattr(self.network.ac, "_lf_lightweight", False):
            self.network.ac.result = result
            return

        def iter_aligned(devices, rows, idx_col):
            if len(devices) == len(rows) and all(int(dev.idx) == int(row[idx_col]) for dev, row in zip(devices, rows)):
                return zip(devices, rows)
            by_idx = {int(dev.idx): dev for dev in devices}
            return ((by_idx.get(int(row[idx_col])), row) for row in rows)

        for node, row in iter_aligned(self.network.ac.nodes, result["bus"], BUS_COLS["idx"]):
            if node is not None:
                node.voltage = float(row[BUS_COLS["voltage"]])
                node.angle = float(row[BUS_COLS["angle"]])
                node.isl = int(row[BUS_COLS["isl"]])
                node.is_alive = int(row[BUS_COLS["run_stat"]]) == 1

        for gen, row in iter_aligned(self.network.ac.generators, result["gen"], GEN_COLS["idx"]):
            if gen is not None:
                gen.p = float(row[GEN_COLS["p"]])
                gen.q = float(row[GEN_COLS["q"]])
                gen.current = float(row[GEN_COLS["current"]])
                gen.is_alive = int(row[GEN_COLS["run_stat"]]) == 1

        for load, row in iter_aligned(self.network.ac.loads, result["load"], LOAD_COLS["idx"]):
            if load is not None:
                load.p = float(row[LOAD_COLS["p"]])
                load.q = float(row[LOAD_COLS["q"]])
                load.current = float(row[LOAD_COLS["current"]])
                load.is_alive = int(row[LOAD_COLS["run_stat"]]) == 1

        for shunt, row in iter_aligned(self.network.ac.shunt_compensators, result["shunt"], SHUNT_COLS["idx"]):
            if shunt is not None:
                shunt.p = float(row[SHUNT_COLS["p"]])
                shunt.q = float(row[SHUNT_COLS["q"]])
                shunt.current = float(row[SHUNT_COLS["current"]])
                shunt.is_alive = int(row[SHUNT_COLS["run_stat"]]) == 1

        for branch, row in iter_aligned(self.network.ac.branches, result["branch"], BRANCH_COLS["idx"]):
            if branch is not None:
                branch.i_p = float(row[BRANCH_COLS["i_p"]])
                branch.i_q = float(row[BRANCH_COLS["i_q"]])
                branch.i_c = float(row[BRANCH_COLS["i_c"]])
                branch.j_p = float(row[BRANCH_COLS["j_p"]])
                branch.j_q = float(row[BRANCH_COLS["j_q"]])
                branch.j_c = float(row[BRANCH_COLS["j_c"]])
                branch.is_alive = int(row[BRANCH_COLS["run_stat"]]) == 1

        for transformer, row in iter_aligned(
            self.network.ac.transformers,
            result["transformer"],
            TRANSFORMER_COLS["idx"],
        ):
            if transformer is not None:
                transformer.i_p = float(row[TRANSFORMER_COLS["i_p"]])
                transformer.i_q = float(row[TRANSFORMER_COLS["i_q"]])
                transformer.i_c = float(row[TRANSFORMER_COLS["i_c"]])
                transformer.j_p = float(row[TRANSFORMER_COLS["j_p"]])
                transformer.j_q = float(row[TRANSFORMER_COLS["j_q"]])
                transformer.j_c = float(row[TRANSFORMER_COLS["j_c"]])
                transformer.is_alive = int(row[TRANSFORMER_COLS["run_stat"]]) == 1

    def _write_back(self, x):
        """Write final global state back into AC, DC and converter model objects."""
        ac_x, dc_x, dcac_x, acac_x = self._split_x(x)
        skip_lf_result = getattr(self, "skip_lf_result", False)
        if self.ac_calc is not None:
            self.ac_calc.x = ac_x
            self.ac_calc.converged = self.converged
            if skip_lf_result:
                self.ac_calc.skip_lf_result = True
            self.ac_calc._write_back()
            self._write_ac_ppc_result_to_network()
        if self.dc_calc is not None:
            self.dc_calc.x = dc_x
            self.dc_calc.converged = self.converged
            if skip_lf_result:
                self.dc_calc.skip_lf_result = True
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
                conv.i_c = float(cur_i)
                conv.j_c = float(cur_j)
        if skip_lf_result:
            self.lf_result = None
        else:
            self.lf_result = self._build_lf_result(ac_V, dc_V)

    def _build_lf_result(self, ac_V=None, dc_V=None) -> HybridLFResult:
        result = HybridLFResult(
            network=self.network,
            ac_network=self.network.ac,
            dc_network=self.network.dc,
            calc=self,
            ac_calc=self.ac_calc,
            dc_calc=self.dc_calc,
            rc=0 if self.converged else -1,
            ac=getattr(self.ac_calc, "lf_result", None),
            dc=getattr(self.dc_calc, "lf_result", None),
        )
        for conv in getattr(self.network, "dcac_converters", []):
            if getattr(conv, "run_stat", 1) != 1:
                continue
            dc_v = float(getattr(conv.dc_node_obj, "voltage", 0.0) or 0.0)
            ac_v = float(getattr(conv.ac_node_obj, "voltage", 0.0) or 0.0)
            result.dcac.dcac_converters[_lf_device_key(conv)] = SimpleNamespace(
                i_p=float(getattr(conv, "dc_p", 0.0) or 0.0),
                i_c=float(getattr(conv, "dc_i", 0.0) or 0.0),
                i_v=dc_v,
                j_p=float(getattr(conv, "ac_p", 0.0) or 0.0),
                j_q=float(getattr(conv, "ac_q", 0.0) or 0.0),
                j_c=float(getattr(conv, "ac_i", 0.0) or 0.0),
                j_v=ac_v,
            )
        for conv in getattr(self.network, "acac_converters", []):
            if getattr(conv, "run_stat", 1) != 1:
                continue
            i_v = float(getattr(conv.i_node_obj, "voltage", 0.0) or 0.0)
            j_v = float(getattr(conv.j_node_obj, "voltage", 0.0) or 0.0)
            result.acac.acac_converters[_lf_device_key(conv)] = SimpleNamespace(
                i_p=float(getattr(conv, "i_p", 0.0) or 0.0),
                i_q=float(getattr(conv, "i_q", 0.0) or 0.0),
                i_c=float(getattr(conv, "i_c", getattr(conv, "i_i", 0.0)) or 0.0),
                i_v=i_v,
                j_p=float(getattr(conv, "j_p", 0.0) or 0.0),
                j_q=float(getattr(conv, "j_q", 0.0) or 0.0),
                j_c=float(getattr(conv, "j_c", getattr(conv, "j_i", 0.0)) or 0.0),
                j_v=j_v,
            )
        return result


def _hybrid_result_from_calc(
    calc,
    rc: int,
    ac_warnings=None,
    ac_errors=None,
    dc_warnings=None,
    dc_errors=None,
) -> HybridLFResult:
    result = getattr(calc, "lf_result", None)
    if isinstance(result, dict):
        return result
    if result is None:
        if getattr(calc, "result_mode", "full") != "full":
            return HybridLFResult(
                network=calc.network,
                ac_network=calc.network.ac,
                dc_network=calc.network.dc,
                calc=calc,
                ac_calc=getattr(calc, "ac_calc", None),
                dc_calc=getattr(calc, "dc_calc", None),
                rc=rc,
                ac_warnings=list(ac_warnings or []),
                ac_errors=list(ac_errors or []),
                dc_warnings=list(dc_warnings or []),
                dc_errors=list(dc_errors or []),
            )
        result = calc._build_lf_result()
    network = getattr(calc, "network", None)
    if network is not None:
        result.network = network
        result.ac_network = network.ac
        result.dc_network = network.dc
    result.calc = calc
    result.ac_calc = getattr(calc, "ac_calc", None)
    result.dc_calc = getattr(calc, "dc_calc", None)
    result.rc = rc
    result.ac_warnings = list(ac_warnings or [])
    result.ac_errors = list(ac_errors or [])
    result.dc_warnings = list(dc_warnings or [])
    result.dc_errors = list(dc_errors or [])
    return result


def run_hybrid_power_flow(
    file_name=DEFAULT_HYBRID_EFILE,
    tol=None,
    max_iter=None,
    min_voltage=None,
    verbose=True,
    parameter_file=DEFAULT_LF_PARAMETER_FILE,
    parameters: Optional[PowerFlowParameters] = None,
    linear_solver: str = "scipy",
    result_mode: str = "full",
) -> HybridLFResult:
    # Main load-flow preparation is delegated to AC/DC sub-solvers.  Full hybrid
    # topology diagnostics remain available through HybridPowerNetwork.prepare()
    # and check_topology(), but the Newton path avoids that duplicate object scan.
    ac_warnings, ac_errors, dc_warnings, dc_errors = [], [], [], []

    network = _read_lf_network_from_file(file_name)
    calc = HybridPowerFlowCalc(
        network,
        tol=tol,
        max_iter=max_iter,
        min_voltage=min_voltage,
        verbose=verbose,
        parameter_file=parameter_file,
        parameters=parameters,
        linear_solver=linear_solver,
        result_mode=result_mode,
    )

    if ac_errors or dc_errors:
        return _hybrid_result_from_calc(calc, -1, ac_warnings, ac_errors, dc_warnings, dc_errors)

    calc.prepare()
    rc = calc.run()
    return _hybrid_result_from_calc(calc, rc, ac_warnings, ac_errors, dc_warnings, dc_errors)


def print_hybrid_result(result: HybridLFResult):
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
    if result.ac_calc is not None:
        print(f"   AC: {'已收敛' if result.ac_calc.converged else '未收敛'}, iter={result.ac_calc.iterations}, normF={result.ac_calc.normF:.3e}")
    else:
        print("   AC: 文件中无 AC 子网")
    if result.dc_calc is not None:
        print(f"   DC: {'已收敛' if result.dc_calc.converged else '未收敛'}, iter={result.dc_calc.iterations}, normF={result.dc_calc.normF:.3e}")
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid AC/DC power flow")
    parser.add_argument("file", nargs="?", default=str(DEFAULT_HYBRID_EFILE), help="hybrid E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument(
        "--linear-solver",
        choices=LINEAR_SOLVER_CHOICES,
        default="scipy",
        help=LINEAR_SOLVER_HELP,
    )
    parser.add_argument("--result-mode", choices=("full", "summary", "none"), default="full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    network = _read_lf_network_from_file(args.file)
    calc = HybridPowerFlowCalc(
        network,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        verbose=verbose,
        parameter_file=args.para,
        linear_solver=args.linear_solver,
        result_mode=args.result_mode,
    )
    calc.prepare()
    rc = calc.run()
    result = _hybrid_result_from_calc(calc, rc)
    if not args.quiet and isinstance(result, HybridLFResult):
        print_hybrid_result(result)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    converged = result.converged if isinstance(result, HybridLFResult) else calc.converged
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
