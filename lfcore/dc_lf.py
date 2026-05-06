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

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from collections import deque
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters


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
    """直流潮流计算器，使用节点电压、零阻抗 phi 和 DCDC 端口功率统一求解。"""

    def __init__(
        self,
        model,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: PowerFlowParameters = None,
    ):
        self.model = model
        self.params = parameters or load_lf_parameters(parameter_file)
        self.runtime_params = self.params
        self.converged = False
        self.iterations = 0
        self.normF = np.inf
        self.verbose = False

    def prepare(self):
        """
        运行潮流计算（修正版，采用节点电位法处理零阻抗支路）
        变量：V (N个) + φ (N_phi个) + Pdc (N_dcdc个)
        方程：功率平衡（除松弛节点外各节点） + 松弛节点电压方程 + 零阻抗电压约束（树支） + φ参考固定 + DC-DC方程
        变量数与方程数严格相等。
        """
        self.alive_nodes = [
            node
            for node in self.model.nodes
            if node.isl_obj is not None and node.isl_obj.is_alive
        ]

        self.N = len(self.alive_nodes)
        if self.N == 0:
            raise ValueError("电网中没有活节点")

        self.alive_node_dict = {node.idx: idx for idx, node in enumerate(self.alive_nodes)}
        self.alive_node_ids = np.asarray([node.idx for node in self.alive_nodes], dtype=np.int32)

        # ---------- 1. 数据预处理 ----------
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

        self.P_const = np.zeros(self.N, dtype=np.float64)   # 注入为正：P型发电机 - P型负荷
        self.I_shunt = np.zeros(self.N, dtype=np.float64)   # 消耗为正：负荷电流 - 发电电流
        self.slack_gen_info = {}

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
            self.P_const[node] -= ld.pv0
            self.I_shunt[node] += ld.pv1
            if ld.pv2 != 0.0:
                load_nodes.append(node)
                load_g.append(ld.pv2)

        # G 矩阵只包含线性电导；恒功率、恒电流和二次负荷项分开放入方程。
        rows_parts = []
        cols_parts = []
        data_parts = []
        if load_nodes:
            load_nodes_arr = np.asarray(load_nodes, dtype=np.int32)
            rows_parts.append(load_nodes_arr)
            cols_parts.append(load_nodes_arr)
            data_parts.append(np.asarray(load_g, dtype=np.float64))

        if self.alive_branch_tuple:
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
        self.zero_edges = [
            ('Z', zb_idx, self.alive_node_dict[zb.i_node], self.alive_node_dict[zb.j_node])
            for zb_idx, zb in enumerate(self.model.zero_branches)
            if zb.is_alive and zb.i_node in self.alive_node_dict and zb.j_node in self.alive_node_dict
        ]
        for sw_idx, sw in enumerate(self.model.switches):
            if (
                sw.is_alive
                and sw.status == 1
                and sw.run_stat == 1
                and sw.i_node in self.alive_node_dict
                and sw.j_node in self.alive_node_dict
            ):
                self.zero_edges.append(('S', sw_idx, self.alive_node_dict[sw.i_node], self.alive_node_dict[sw.j_node]))

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
        self.slack_nodes = {node: gens[0].v_set for node, gens in self.slack_gen_info.items()}
        self.slack_node_arr = np.fromiter(self.slack_nodes.keys(), dtype=np.int32, count=len(self.slack_nodes))
        self.slack_value_arr = np.fromiter(self.slack_nodes.values(), dtype=np.float64, count=len(self.slack_nodes))

        if self.verbose:
            print("self.N = ", self.N)
            print("self.N_phi = ", self.N_phi)
            print("self.N_dcdc = ", self.N_dcdc)

        # ---------- 5. 变量定义 ----------
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

        return G, x

    def get_jacobi(self, G, x):
        """组装 DC Newton 方程的稀疏 Jacobian。"""
        V = x[:self.N]
        phi = x[self.N:self.N + self.N_phi]
        Pdc = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])

        GV = G.dot(V)
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
            current = phi[self.zero_phi_a] - phi[self.zero_phi_b]

            eq_i = self.node_eq[self.zero_i]
            mask_i = eq_i >= 0
            if np.any(mask_i):
                n = int(np.count_nonzero(mask_i))
                cols = np.empty(3 * n, dtype=np.int32)
                data = np.empty(3 * n, dtype=np.float64)
                cols[0::3] = self.N + self.zero_phi_a[mask_i]
                cols[1::3] = self.N + self.zero_phi_b[mask_i]
                cols[2::3] = self.zero_i[mask_i]
                data[0::3] = V[self.zero_i[mask_i]]
                data[1::3] = -V[self.zero_i[mask_i]]
                data[2::3] = current[mask_i]
                rows_parts.append(np.repeat(eq_i[mask_i], 3))
                cols_parts.append(cols)
                data_parts.append(data)

            eq_j = self.node_eq[self.zero_j]
            mask_j = eq_j >= 0
            if np.any(mask_j):
                n = int(np.count_nonzero(mask_j))
                cols = np.empty(3 * n, dtype=np.int32)
                data = np.empty(3 * n, dtype=np.float64)
                cols[0::3] = self.N + self.zero_phi_a[mask_j]
                cols[1::3] = self.N + self.zero_phi_b[mask_j]
                cols[2::3] = self.zero_j[mask_j]
                data[0::3] = -V[self.zero_j[mask_j]]
                data[1::3] = V[self.zero_j[mask_j]]
                data[2::3] = -current[mask_j]
                rows_parts.append(np.repeat(eq_j[mask_j], 3))
                cols_parts.append(cols)
                data_parts.append(data)

        # 8.3 DC-DC功率变量对节点功率方程的偏导
        if self.N_dcdc:
            eq_i = self.node_eq[self.dcdc_i]
            mask_i = eq_i >= 0
            if np.any(mask_i):
                idx = np.where(mask_i)[0].astype(np.int32)
                rows_parts.append(eq_i[mask_i])
                cols_parts.append(self.N + self.N_phi + idx * 2)
                data_parts.append(np.ones(idx.size, dtype=np.float64))

            eq_j = self.node_eq[self.dcdc_j]
            mask_j = eq_j >= 0
            if np.any(mask_j):
                idx = np.where(mask_j)[0].astype(np.int32)
                rows_parts.append(eq_j[mask_j])
                cols_parts.append(self.N + self.N_phi + idx * 2 + 1)
                data_parts.append(np.ones(idx.size, dtype=np.float64))

        # 8.4 松弛节点电压方程行
        if self.n_known:
            rows_parts.append(self.eq_known_start + np.arange(self.n_known, dtype=np.int32))
            cols_parts.append(self.slack_node_arr)
            data_parts.append(np.ones(self.n_known, dtype=np.float64))

        # 8.5 零阻抗电压约束行
        if self.n_zero_constraint:
            cols = np.empty(2 * self.n_zero_constraint, dtype=np.int32)
            data = np.empty(2 * self.n_zero_constraint, dtype=np.float64)
            cols[0::2] = self.zero_con_i
            cols[1::2] = self.zero_con_j
            data[0::2] = 1.0
            data[1::2] = -1.0
            rows_parts.append(np.repeat(self.zero_con_rows, 2))
            cols_parts.append(cols)
            data_parts.append(data)

        # 8.6 φ参考固定行
        if self.n_phi_fix:
            rows_parts.append(self.phi_fix_rows)
            cols_parts.append(self.N + self.ref_phi_idx)
            data_parts.append(np.ones(self.n_phi_fix, dtype=np.float64))

        # 8.7 DC-DC方程行
        if self.N_dcdc:
            if np.any(self.dcdc_ctrl_p_mask):
                rows_parts.append(self.dcdc_eq_ctrl[self.dcdc_ctrl_p_mask])
                cols_parts.append(self.dcdc_p_col[self.dcdc_ctrl_p_mask])
                data_parts.append(np.ones(int(np.count_nonzero(self.dcdc_ctrl_p_mask)), dtype=np.float64))
            if np.any(self.dcdc_ctrl_v_mask):
                rows_parts.append(self.dcdc_eq_ctrl[self.dcdc_ctrl_v_mask])
                cols_parts.append(self.dcdc_i[self.dcdc_ctrl_v_mask])
                data_parts.append(np.ones(int(np.count_nonzero(self.dcdc_ctrl_v_mask)), dtype=np.float64))
            if np.any(self.dcdc_ctrl_i_mask):
                rows_parts.append(np.repeat(self.dcdc_eq_ctrl[self.dcdc_ctrl_i_mask], 2))
                cols_parts.append(
                    np.column_stack(
                        (
                            self.dcdc_p_col[self.dcdc_ctrl_i_mask],
                            self.dcdc_i[self.dcdc_ctrl_i_mask],
                        )
                    ).ravel()
                )
                data_parts.append(
                    np.column_stack(
                        (
                            np.ones(int(np.count_nonzero(self.dcdc_ctrl_i_mask)), dtype=np.float64),
                            -self.dcdc_i_set[self.dcdc_ctrl_i_mask],
                        )
                    ).ravel()
                )

            vi = V[self.dcdc_i]
            vj = V[self.dcdc_j]
            pi = Pdc[0::2]
            pj = Pdc[1::2]
            vi2 = vi * vi
            vj2 = vj * vj
            pi2 = pi * pi
            pj2 = pj * pj
            loss_data = np.empty(self.N_dcdc * 4, dtype=np.float64)
            loss_data[0::4] = vi2 * vj2 - 2.0 * self.dcdc_r1 * pi * vj2
            loss_data[1::4] = vi2 * vj2 - 2.0 * self.dcdc_r2 * pj * vi2
            loss_data[2::4] = 2.0 * vi * vj2 * (pi + pj) - 2.0 * self.dcdc_r2 * pj2 * vi
            loss_data[3::4] = 2.0 * vj * vi2 * (pi + pj) - 2.0 * self.dcdc_r1 * pi2 * vj
            rows_parts.append(np.repeat(self.dcdc_eq_loss, 4))
            cols_parts.append(np.column_stack((self.dcdc_p_col, self.dcdc_q_col, self.dcdc_i, self.dcdc_j)).ravel())
            data_parts.append(loss_data)

        if rows_parts:
            J_rows = np.concatenate(rows_parts)
            J_cols = np.concatenate(cols_parts)
            J_data = np.concatenate(data_parts)
        else:
            J_rows = J_cols = np.array([], dtype=np.int32)
            J_data = np.array([], dtype=np.float64)

        return coo_matrix((J_data, (J_rows, J_cols)), shape=(self.total_eq, self.total_vars)).tocsr()

    def get_f(self, x):
        """计算 DC 残差：节点功率平衡、参考电压、零阻抗约束和 DCDC 约束。"""
        V = x[:self.N]
        phi = x[self.N:self.N + self.N_phi]
        Pdc = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])

        P_inj = V * self.G.dot(V) + self.I_shunt * V - self.P_const

        if self.zero_i.size:
            current = phi[self.zero_phi_a] - phi[self.zero_phi_b]
            np.add.at(P_inj, self.zero_i, V[self.zero_i] * current)
            np.add.at(P_inj, self.zero_j, -V[self.zero_j] * current)

        if self.N_dcdc:
            np.add.at(P_inj, self.dcdc_i, Pdc[0::2])
            np.add.at(P_inj, self.dcdc_j, Pdc[1::2])

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
            vi = V[self.dcdc_i]
            vj = V[self.dcdc_j]
            ctrl_values = np.empty(self.N_dcdc, dtype=np.float64)
            ctrl_values[self.dcdc_ctrl_p_mask] = p_from[self.dcdc_ctrl_p_mask] - self.dcdc_p_set[self.dcdc_ctrl_p_mask]
            ctrl_values[self.dcdc_ctrl_v_mask] = vi[self.dcdc_ctrl_v_mask] - self.dcdc_v_set[self.dcdc_ctrl_v_mask]
            ctrl_values[self.dcdc_ctrl_i_mask] = (
                p_from[self.dcdc_ctrl_i_mask]
                - self.dcdc_i_set[self.dcdc_ctrl_i_mask] * vi[self.dcdc_ctrl_i_mask]
            )
            F[self.dcdc_eq_ctrl] = ctrl_values

            vi2 = vi * vi
            vj2 = vj * vj
            # 第二条方程保证两端端口功率与 r1/r2 损耗模型一致。
            F[self.dcdc_eq_loss] = (
                vi2 * vj2 * (p_from + p_to)
                - self.dcdc_r1 * p_from * p_from * vj2
                - self.dcdc_r2 * p_to * p_to * vi2
            )

        return F

    def update_lf_info(self, x):
        """将求解后的电压、电流和功率写回 DC 模型对象。"""
        # ---------- 9. 结果回填 ----------
        V_final = x[:self.N]
        phi_final = x[self.N:self.N + self.N_phi] if self.N_phi > 0 else np.array([])
        Pdc_final = x[self.N + self.N_phi:self.N + self.N_phi + self.N_dcdc * 2] if self.N_dcdc > 0 else np.array([])

        # 节点电压
        for node in self.model.nodes:
            idx = self.alive_node_dict.get(node.idx, -1)
            node.voltage = 0.0 if idx < 0 else V_final[idx]


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
            np.add.at(P_inj, self.branch_i, i_p)
            np.add.at(P_inj, self.branch_j, j_p)

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

        # 负荷
        for ld in self.model.loads:
            idx = self.alive_node_dict.get(ld.node, -1)
            if idx < 0 or not ld.is_alive:
                ld.p = 0.0
                ld.current = 0.0
                continue
            v = V_final[idx]
            ld.p = ld.pv0 + ld.pv1 * v + ld.pv2 * v * v
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
        for d_idx, (idx, i_node, j_node, ctrl, p_set, i_set, v_set, r1, r2) in enumerate(self.alive_dcdc_tuples):
            dc = self.model.dcdc_converters[idx]
            dc.i_p = Pdc_final[d_idx * 2]
            dc.j_p = Pdc_final[d_idx * 2 + 1]
            vi = V_final[i_node]
            vj = V_final[j_node]
            dc.i_c = dc.i_p / vi if abs(vi) > self.runtime_params.min_voltage else 0.0
            dc.j_c = dc.j_p / vj if abs(vj) > self.runtime_params.min_voltage else 0.0

            P_inj[i_node] += dc.i_p
            P_inj[j_node] += dc.j_p  # eta

        # P_inj 是剩余的不平衡功率，由平衡机来承担。。。

        for node, gens in self.slack_gen_info.items():
            share = P_inj[node] / len(gens)
            for gen in gens:
                gen.p = share
                gen.current = gen.p / V_final[node] if abs(V_final[node]) > self.runtime_params.min_voltage else 0.0


    def run(
        self,
        tol=None,
        max_iter=None,
        min_voltage=None,
        divergence_threshold=None,
        verbose=False,
    ):
        """执行直流 Newton 迭代并在收敛后回填结果。"""

        params = self.params.with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
            divergence_threshold=divergence_threshold,
        )
        self.runtime_params = params
        self.verbose = verbose
        G, x = self.prepare()
        self.converged = False


        # ---------- 7. 牛顿-拉夫逊迭代 ----------
        for it in range(params.max_iter):
            F = self.get_f(x)

            # print("F", F)
            # 收敛检查
            self.normF = np.linalg.norm(F, np.inf)

            if self.verbose:
                print("it:",it, f"eps:{self.normF:.3e}")

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

            J = self.get_jacobi(G,x)

            try:
                delta = spsolve(J, -F)
            except Exception as e:
                if self.verbose:
                    print(f"\n线性方程组求解失败: {e}")
                # 改用最小二乘作为备选
                J_dense = J.toarray()
                try:
                    delta = np.linalg.lstsq(J_dense, -F, rcond=None)[0]
                except:
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

if __name__ == "__main__":
    from dc_model import DCPowerNetwork

    net = DCPowerNetwork()
    net.read_from_file("../../data/dc/dc_net_30.e")

    net.topo()

    net.print_isl_info()

    # 8. 运行潮流计算
    print("=== 开始直流电网潮流计算===")
    calc = DCPowerFlowCalc(net)
    calc.run()

    # 9. 输出详细结果
    print("\n===输出直流电网潮流计算结果===")

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
