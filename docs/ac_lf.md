# ac_lf 技术文档

## 模块定位

`src/hybrid_power_system_analysis/lfcore/ac_lf.py` 实现交流电网 Newton-Raphson 潮流计算。核心类是 `ACPowerFlowCalc`。

该模块负责：

- 读取或接收交流网络拓扑与设备参数。
- 生成交流节点导纳矩阵 `Y`。
- 支持 MATPOWER 风格线路 stamp，包括线路充电电纳 `b`；变压器按 T 型单端对地支路建模，支持 `gt/bt/tap/shift`。
- 支持普通线路、主变、ZIP 负荷、并联补偿、发电机控制、开关和零阻抗支路。
- 生成残差向量 `F(x)` 和稀疏 Jacobian `J(x)`。
- 迭代求解后把节点电压、相角、设备潮流、电流和发电机出力回填到模型对象或 array-mode 结果。

## 入口与使用方式

主要 API：

```python
from lfcore.ac_lf import ACPowerFlowCalc

calc = ACPowerFlowCalc(network, tol=1e-8, max_iter=50, min_voltage=0.01)
calc.prepare()
rc = calc.run()
```

也支持数组化输入：

```python
calc = ACPowerFlowCalc.from_ppc(ppc, tol=1e-8, max_iter=50)
rc = calc.run()
result = calc.result
```

当前脚本尾部的 `__main__` 更偏向调试示例，默认读取 `data/model/ac/ieee300.e`，生产使用建议通过类接口或上层主程序调用。

## 支持的模型对象

交流潮流使用以下主要 E 文件对象：

| 对象 | 作用 |
| --- | --- |
| `ACNode` | 交流节点，含电压基准、初始电压和初始相角 |
| `ACBranch` | 普通交流线路，参数 `r/x/b` |
| `ACTransformer` | 主变，参数 `r/x/gt/bt/tap/shift`，其中 `gt/bt` 为 i 侧单端对地导纳 |
| `ACGenerator` | 发电机，支持 `V/SLACK/PH/PV/P/PQ` 等控制类型 |
| `ACLoad` | ZIP 负荷，`pv0/pv1/pv2/qv0/qv1/qv2` |
| `ACShuntCompensator` | 并联设备，支持 `g_set/b_set/q_set/v_set` |
| `ACZeroBranch` | 零阻抗支路 |
| `ACSwitch` | 交流开关，闭合时可作为零阻抗边 |

所有设备都通过 `run_stat` 控制是否参与计算。

## 状态变量

交流潮流状态由三类变量组成：

| 变量 | 含义 |
| --- | --- |
| `theta_unknown` | 非平衡节点相角 |
| `V_unknown` | PQ 节点电压幅值 |
| `phi_re/phi_im` | 零阻抗支路辅助势变量，用于表达零阻抗支路电流 |

平衡节点相角和电压由发电机或节点初值固定；PV 节点电压固定，相角参与求解；PQ 节点相角和电压都参与求解。

零阻抗支路不直接写成无限导纳，而是引入 `phi` 辅助变量，并将零阻抗支路两端电压相等作为约束方程。这样可以避免在 `Y` 阵中写入极大导纳导致病态。

## 导纳矩阵 stamp

普通支路和主变采用 MATPOWER 公式：

```text
yff, yft, ytf, ytt = matpower_transformer_stamp(r, x, gt, bt, tap, shift)
```

其中：

- `b` 保留为线路总充电电纳。
- `tap=0` 时按 `1.0` 处理。
- `shift` 为角度输入，内部转换为弧度。
- 主变按复变比 `tap * exp(j * shift)` stamp。

向量化函数 `matpower_branch_stamp_vectorized()` 用于批量生成大型算例的支路 stamp。

## 残差方程

`get_f(x)` 计算非线性方程残差，主要包括：

| 方程 | 说明 |
| --- | --- |
| AC 节点有功平衡 | 对非平衡节点或 PQ/PV 控制节点建立 P 平衡 |
| AC 节点无功平衡 | 对 PQ 节点建立 Q 平衡 |
| 零阻抗支路电压实部差 | 零阻抗边两端复电压实部相等 |
| 零阻抗支路电压虚部差 | 零阻抗边两端复电压虚部相等 |

节点网络注入通过 `S = V * conj(Y * V)` 计算。负荷按 ZIP 模型随电压变化，零阻抗支路注入通过 `phi` 差值对应的支路电流计算。

## Jacobian

`get_jacobi(x)` 返回 CSR 稀疏矩阵。实现分为两部分：

- 标准 AC 潮流块：相角/电压对 P/Q 平衡的解析导数。
- 零阻抗块：零阻抗电流注入对节点电压、相角和 `phi` 的解析导数，以及零阻抗电压约束对状态的导数。

大规模算例中主要使用稀疏 COO 拼接后转 CSR，避免构造完整稠密矩阵。

## 求解流程

`run()` 执行 Newton 迭代：

1. 调用 `get_f(x)` 计算残差。
2. 以无穷范数 `normF = ||F||_inf` 判断收敛。
3. 调用 `get_jacobi(x)` 生成稀疏 Jacobian。
4. 使用 `scipy.sparse.linalg.spsolve` 求解 `J * delta = F`。
5. 更新 `x = x - delta`。
6. 收敛后调用 `_write_back()`。

返回码：

| 返回码 | 含义 |
| --- | --- |
| `0` | 收敛 |
| `-1` | 达到最大迭代次数未收敛 |

## 结果回填

`_write_back()` 会更新：

- 节点 `voltage/angle`
- 发电机 `p/q/current`
- 负荷 `p/q/current`
- 并联补偿 `p/q/current`
- 线路和主变两端 `i_p/i_q/i_c/j_p/j_q/j_c`
- 开关和零阻抗支路 `p/q/current`

array-mode 下结果写入 `calc.result`，不会直接修改输入 `ppc`。

## 性能设计

当前实现的主要性能手段：

- 支路 stamp 向量化。
- `Y` 阵、支路数组、负荷数组、零阻抗数组在 `prepare()` 阶段缓存。
- P/Q 负荷、节点注入、零阻抗注入用 NumPy 批量运算。
- Jacobian 直接按稀疏 triplet 生成。
- array-mode 避免大量 Python 对象访问，适合 IEEE 大规模拼接算例。

## 注意事项

- 多个平衡节点只有在零阻抗等值相连且固定电压/相角一致时才可作为冗余参考处理。
- 零阻抗支路作为拓扑约束处理，不应通过极小阻抗普通支路替代。
- `run()` 会打印每次迭代残差，若需要静默运行可由上层重定向输出。
