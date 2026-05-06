# ac_se 技术文档

## 模块定位

`secore/ac_se.py` 实现交流电网加权最小二乘状态估计。核心类是 `ACStateEstimator`。

该模块负责：

- 读取 AC E 文件和 `.meas` 量测文件。
- 将有名值量测转换为内部标幺值和弧度。
- 自动过滤无效量测和离线设备量测。
- 自动添加低权重伪量测，提高无功率量测设备的可观测性。
- 自动添加零阻抗支路电压/相角等值约束。
- 构造量测函数 `h(x)` 和解析稀疏 Jacobian `H`。
- 完成可观测性分析、WLS 状态估计、坏数据辨识和可选坏数据剔除。

## 入口与使用方式

```python
from secore.ac_se import ACStateEstimator

estimator = ACStateEstimator(
    e_file="data/ac/ieee300.e",
    meas_file="data/ac/ieee300.meas",
    flat_start=True,
)
result = estimator.estimate(verbose=False)
```

命令行：

```powershell
python secore\ac_se.py --case data\ac\ieee300.e --meas data\ac\ieee300.meas --flat-start --quiet
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--case` | AC 网络 E 文件 |
| `--meas` | 量测 E 文件 |
| `--para` | 状态估计参数文件，默认 `se.para` |
| `--tol` | 状态修正量收敛阈值 |
| `--max-iter` | 最大 WLS 迭代次数 |
| `--bad-threshold` | 坏数据归一化残差阈值 |
| `--flat-start` | 使用平启动 |
| `--remove-bad-data` | 迭代剔除最大坏数据 |
| `--print-state` | 打印估计状态 |
| `--quiet` | 不打印迭代过程 |

## 量测格式

量测文件 `<Measurement>` 使用列：

```text
idx name dev_type dev_name meas_type weight valid value
```

`valid=0` 或 `weight<=0` 的量测不参与估计。

支持的主要 AC 量测：

| 设备 | 量测类型 |
| --- | --- |
| `ACNode` | `V`, `ANGLE`/`THETA` |
| `ACBranch` | `P_FROM`, `Q_FROM`, `V_FROM`, `I_FROM`, `P_TO`, `Q_TO`, `V_TO`, `I_TO` |
| `ACTransformer` | 同 `ACBranch` |
| `ACSwitch` | `P_FROM`, `Q_FROM`, `V_FROM`, `I_FROM`, `P_TO`, `Q_TO`, `V_TO`, `I_TO` |
| `ACZeroBranch` | `P_FROM`, `Q_FROM`, `V_FROM`, `I_FROM`, 以及 `V_DIFF`, `ANGLE_DIFF` |
| `ACGenerator` | `P_GEN`, `Q_GEN`, `V_GEN`, `I_GEN` |
| `ACLoad` | `P_LOAD`, `Q_LOAD`, `V_LOAD`, `I_LOAD` |
| `ACZeroBranchConstraint` | `V_DIFF`, `ANGLE_DIFF` |

## 状态变量

`ACStateEstimator` 状态向量：

```text
x = [theta_non_ref, V_all, I_zero_re, I_zero_im]
```

| 变量 | 含义 |
| --- | --- |
| `theta_non_ref` | 非参考节点相角 |
| `V_all` | 所有存活 AC 节点电压幅值 |
| `I_zero_re/I_zero_im` | ACZeroBranch/ACSwitch 显式电流状态的实部和虚部 |

每个存活拓扑岛选一个参考节点去除整体相角自由度。拼接 IEEE 大算例中，零阻抗互联还会自动加入约束量测保证等值节点可观测。

## 自动量测

### 伪量测

如果发电机或负荷没有任何有效量测，程序自动添加低权重伪量测：

- 发电机：`P_GEN`, `Q_GEN`, `V_GEN`
- 负荷：`P_LOAD`, `Q_LOAD`, `V_LOAD`

伪量测权重来自 `se.para` 的 `pseudo_measurement_weight`。

### 零阻抗约束

对每条存活 `ACZeroBranch` 自动添加：

```text
ACZeroBranchConstraint V_DIFF = 0
ACZeroBranchConstraint ANGLE_DIFF = 0
```

该约束反映理想零阻抗支路两端电压和相角相等。它解决了多个 IEEE300 拷贝通过零阻抗支路拼接后只选一个参考角导致的可观测性缺失问题。

## 量测函数

`evaluate(x)` 按设备类型计算估计量测 `h(x)`：

- 节点电压/相角直接取状态。
- 线路/主变潮流通过 MATPOWER stamp 计算端口电流和复功率。
- 开关/零阻抗支路通过显式电流状态计算 P/Q/I。
- 发电机功率由节点网络注入、负荷和零阻抗支路注入推断，并按同节点多发电机 `alpha` 分摊。
- 负荷按 ZIP 模型随电压计算。

## Jacobian

`jacobian_sparse(x)` 返回解析稀疏 Jacobian。主要优化：

- 线路和主变 P/Q/I 对相角、电压导数向量化。
- 开关和零阻抗支路 P/Q/I 对电压和显式电流导数批量生成。
- 负荷 P/Q/I 对电压导数批量生成。
- 发电机节点注入导数按节点/设备分组复用。
- 稀疏矩阵通过 `SparseJacobianBuilder` 直接累积 triplet。

稠密 `jacobian(x)` 保留用于测试和诊断。

## WLS 求解

目标函数：

```text
min 0.5 * (z - h(x)).T * W * (z - h(x))
```

迭代步骤：

1. 计算残差 `r = z - h(x)`。
2. 计算 `H = dh/dx`。
3. 构造正规方程 `G = H.T W H`, `rhs = H.T W r`。
4. 求解 `G * dx = rhs`。
5. 使用线搜索保证目标函数不增，同时限制电压不低于 `voltage_floor`。
6. 当 `max(abs(dx)) < tol` 时收敛。

## 可观测性分析

`observability_analysis()` 使用当前 Jacobian 或正规矩阵判断：

- `observable=True` 表示 `rank == state_count`。
- 若不可观测，会返回弱状态列表 `weak_states`。

大规模算例中会优先复用正规矩阵或 Cholesky 因子，避免退化到昂贵的 SVD。

## 坏数据辨识

`identify_bad_data()` 计算归一化残差：

```text
rN_i = abs(r_i) / sqrt(R_ii - h_i * G^-1 * h_i.T)
```

超过 `bad_threshold` 的量测列为坏数据。`estimate_with_bad_data_removal()` 可按最大归一化残差迭代剔除。

## 输出结果

`EstimateResult` 字段：

| 字段 | 含义 |
| --- | --- |
| `converged` | 是否收敛 |
| `iterations` | 迭代次数 |
| `objective` | WLS 目标函数 |
| `max_correction` | 最大状态修正量 |
| `residual_inf` | 残差无穷范数 |
| `x` | 最终状态 |
| `z_est` | 估计量测 |
| `residual` | 量测残差 |
| `H` | 最终 Jacobian |
| `gain` | 正规矩阵 |
| `observability` | 可观测性结果 |

## 注意事项

- `.meas` 中的相角是度，内部自动转弧度。
- E 文件和 `.meas` 的功率、电压、电流都应使用有名值，单位由 E 文件的 scale 字段解释。
- AC 大规模拼接算例必须保留 `ACZeroBranch`，不要用普通极小阻抗线路替代。

