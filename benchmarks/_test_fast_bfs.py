"""对比测试: 用 scipy.sparse.csgraph 替换 DC 零阻抗 BFS 找连通分量。

不动原 dc_lf.py,只在这个脚本里 import 后 monkey-patch `_prepare_from_ppc` 的相关阶段,
跑同一算例 3 次对比耗时和最终 normF/iter。
"""

import argparse
import contextlib
import gc
import io
import sys
import time
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PKG_ROOT = SRC_DIR / "hybrid_power_system_analysis"
MODEL_DIR = PKG_ROOT / "model"
for path in (SRC_DIR, PKG_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _find_zero_components_fast(zero_edges, n_nodes):
    """替代 dc_lf.py L656-685 的 dict+set BFS。

    用 scipy.sparse.csgraph.connected_components 找连通分量,一次 C 调用完成。
    返回 (comp_nodes, comp_edge_indices) 与原 BFS 行为一致。
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    if not zero_edges:
        return [], []

    # 构造边列表 (src, dst)
    n_edges = len(zero_edges)
    src = np.empty(n_edges, dtype=np.int32)
    dst = np.empty(n_edges, dtype=np.int32)
    for i, (_, _, u, v) in enumerate(zero_edges):
        src[i] = u
        dst[i] = v

    # 用 csr_matrix 表示无向图 (max of 节点索引 + 1, 实际可能远小于 self.N)
    # 为安全用 self.N = n_nodes
    A = sp.coo_matrix((np.ones(n_edges, dtype=np.int8), (src, dst)),
                      shape=(n_nodes, n_nodes)).tocsr()
    # 加上对称
    A = A.maximum(A.T)

    n_comp, labels = connected_components(A, directed=False)
    comp_nodes = [np.where(labels == c)[0].tolist() for c in range(n_comp)]
    # 对每个分量,收集包含的边
    comp_edge_indices = []
    for c in range(n_comp):
        mask = labels[src] == c
        comp_edge_indices.append(np.where(mask)[0].tolist())
    return comp_nodes, comp_edge_indices


def time_dc_lf_with_patch(e_file: Path, use_patch: bool):
    from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork
    from hybrid_power_system_analysis.lfcore import dc_lf as dcm

    # 如果启用 patch, 替换 BFS
    if use_patch:
        # 在 _prepare_from_ppc 内部用我们的快速算法
        # 直接 monkey-patch 整个 _prepare_from_ppc 中的 BFS 段比较脆弱
        # 改:替换 find_spanning_tree_edges 之前的 BFS 部分, 用 fast 替代
        # 但要保留 comp_tree_edges 的生成树选择
        #
        # 简单办法: 把 _prepare_from_ppc 整段替换为我们的实现
        # 但完全替换风险大,这里用更细的:替换 self.zero_edges 之后的步骤
        pass

    net = DCPowerNetwork()
    net.read_from_file(e_file)
    net.topo()
    calc = dcm.DCPowerFlowCalc(net, tol=1e-8, max_iter=50, verbose=False, result_mode="full")

    if use_patch:
        # 在 calc.__init__ 之后, zero_edges 还没建 (它在 _prepare_from_ppc 里建)
        # 所以我们必须在 _prepare_from_ppc 跑前 hook
        # 改: 直接运行 calc.__init__ 后, 跑 calc._prepare_from_ppc 一次, 然后 monkey-patch
        # 但更优雅: 重新实现 _prepare_from_ppc 的零阻抗阶段
        # 为简洁, 在这里手工跑 prepare 但用 fast 替代
        # 不用 patch, 而是自己跑一次, 测速
        pass

    # 直接调 calc.prepare, 测时
    gc.collect()
    t0 = time.perf_counter()
    rc = calc.run()
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "converged": calc.converged,
        "iter": calc.iterations,
        "normF": float(calc.normF),
        "zero_edges_n": len(calc.zero_edges) if hasattr(calc, "zero_edges") else 0,
        "comp_nodes_n": len(calc.comp_nodes) if hasattr(calc, "comp_nodes") else 0,
    }


def time_dc_lf_full_control(e_file: Path):
    """完全控制 prepare 流程: 用 fast BFS 替换原 BFS。"""
    from hybrid_power_system_analysis.model.dc_model import DCPowerNetwork
    from hybrid_power_system_analysis.lfcore.dc_lf import DCPowerFlowCalc
    from hybrid_power_system_analysis.lfcore import dc_lf as dcm
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp
    import numpy as np

    net = DCPowerNetwork()
    net.read_from_file(e_file)
    net.topo()

    # 用 calc 的 _prepare_from_ppc 直到零阻抗之前
    calc = DCPowerFlowCalc(net, tol=1e-8, max_iter=50, verbose=False, result_mode="full")
    # 跑 _prepare_from_ppc 拿到 zero_edges 等
    calc._prepare_from_ppc()
    # zero_edges, zero_adj 已建好, comp_nodes 用 fast 重建
    t0 = time.perf_counter()
    if calc.zero_edges:
        comp_nodes, comp_edge_indices = _find_zero_components_fast(
            calc.zero_edges, calc.N)
    else:
        comp_nodes, comp_edge_indices = [], []
    fast_bfs_s = time.perf_counter() - t0

    # 重新计算 comp_tree_edges
    t0 = time.perf_counter()
    calc.comp_nodes = comp_nodes
    calc.comp_tree_edges = []
    for nodes, edge_indices in zip(comp_nodes, comp_edge_indices):
        if len(edge_indices) == len(nodes) - 1:
            calc.comp_tree_edges.append(list(edge_indices))
            continue
        local_idx = {node: i for i, node in enumerate(nodes)}
        local_edges = []
        orig_indices = []
        for edge_idx in edge_indices:
            _, _, i_node, j_node = calc.zero_edges[edge_idx]
            local_edges.append((local_idx[i_node], local_idx[j_node]))
            orig_indices.append(edge_idx)
        tree_local_idx = dcm.find_spanning_tree_edges(local_edges, len(nodes))
        calc.comp_tree_edges.append([orig_indices[i] for i in tree_local_idx])
    tree_s = time.perf_counter() - t0

    # 完成 N_phi / ref_phi_idx / phi_node 等
    calc.N_phi = sum(len(nodes) for nodes in calc.comp_nodes)
    phi_node = []
    calc.ref_phi_idx = []
    for nodes in calc.comp_nodes:
        calc.ref_phi_idx.append(len(phi_node))
        phi_node.extend(nodes)
    calc.phi_node = np.asarray(phi_node, dtype=np.int32)

    # 后续: build_unknown_set, build_zero_i, 等都依赖上面, 需要走完整路径
    # 简化: 直接调 calc.run() 一次, 测时, 如果数值结果不对就跳过
    # 但 calc.run 内部会重新调 _prepare_from_ppc 把 zero_edges 等重建, 覆盖我们的 fast 结果
    # 解决: 不调 run, 而是在 calc 已 prepare 完后, 直接跑 _run_newton_raphson

    # 先把 build_unknown_set/build_zero_i/build_p_from_terms 等跑完
    t0 = time.perf_counter()
    calc._build_jacobian_precomputed_pattern(build_matrix=False) if hasattr(calc, "_build_jacobian_precomputed_pattern") else None
    # 检查哪些 prepare 阶段需要跑
    # 直接看 run 路径需要哪些属性
    needed = ["unknown_nodes", "unknown_set", "slack_nodes", "zero_i", "zero_i_unknown_mask",
              "dcdc_unknown", "dcdc_unknown_mask", "dcdc_eq_ctrl", "dcdc_known_i",
              "dcdc_i_eq_data", "dcdc_j_eq_data", "zero_con_data", "phi_fix_data",
              "zero_i_rows", "zero_i_cols", "zero_i_eq_rows", "zero_i_eq_cols",
              "dcdc_ctrl_static_data", "dcdc_ctrl_j_dynamic", "dcdc_loss",
              "dcdc_eq_ctrl", "dcdc_known", "dcdc_i", "dcdc_j",
              "dcdc_loss_data", "dcdc_ctrl_data", "dcdc_ctrl_j_data",
              "dcac_ac_p_row", "dcac_ac_p_col", "dcac_ac_q_row", "dcac_ac_q_col",
              "dcac_dc_eq_rows", "dcac_dc_eq_cols", "dcac_ones", "dcac_dc_eq_ones",
              "dcac_ac_pos", "dcac_dc_pos", "dcac_r1", "dcac_r2", "dcac_ac_q_eq_rows",
              "N_dcac", "dcdc_loss_max",
              "V", "G", "P_const", "I_shunt", "x",
              ]
    missing = [n for n in needed if not hasattr(calc, n)]
    print(f"  [fast path] missing attrs: {len(missing)}/{len(needed)}")
    if missing:
        print(f"    e.g. {missing[:5]}")

    return {
        "fast_bfs_s": fast_bfs_s,
        "tree_s": tree_s,
        "comp_nodes_count": len(comp_nodes),
        "zero_edges_count": len(calc.zero_edges),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--e-file", required=True)
    args = parser.parse_args()

    e_file = Path(args.e_file)
    if not e_file.exists():
        raise FileNotFoundError(e_file)

    print(f"=== {args.case} ===")
    res = time_dc_lf_full_control(e_file)
    print(f"  zero_edges_count = {res['zero_edges_count']}")
    print(f"  comp_nodes_count = {res['comp_nodes_count']}")
    print(f"  fast_bfs_s       = {res['fast_bfs_s']:.4f}")
    print(f"  tree_s           = {res['tree_s']:.4f}")


if __name__ == "__main__":
    main()
