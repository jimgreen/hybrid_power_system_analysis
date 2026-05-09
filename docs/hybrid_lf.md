# hybrid_lf 技术文档

## 模块定位

`src/hybrid_power_system_analysis/lfcore/hybrid_lf.py` 实现交直流联合潮流计算。核心类包括：

- `HybridACGrid`
- `HybridDCGrid`
- `HybridIsland`
- `HybridPowerNetwork`
- `HybridPowerFlowCalc`
- `HybridPowerFlowResult`

该模块不是简单顺序调用 AC 和 DC 潮流，而是将 AC、DC、DC/AC、AC/AC 等方程合并到一个全局 Newton 系统中统一求解。

## 支持的网络类型

`hybrid_lf.py` 同时支持：

| 类型 | 说明 |
| --- | --- |
| 纯交流 | E 文件只有 AC 设备，DC 子系统为空 |
| 纯直流 | E 文件只有 DC 设备，AC 子系统为空 |
| 交直流混联 | 同时含 AC、DC、DCAC、DCDC、ACAC 等设备 |

## 入口与使用方式

```python
from lfcore.hybrid_lf import run_hybrid_power_flow

result = run_hybrid_power_flow(
    "data/hybrid/qinling.e",
    tol=1e-8,
    max_iter=50,
    min_voltage=0.01,
    verbose=False,
)
```

命令行：

```powershell
python -m lfcore.hybrid_lf data\hybrid\qinling.e --quiet
```

参数：

| 参数 | 说明 |
| --- | --- |
| `file` | E 文件路径 |
| `--para` | 潮流参数文件，默认 `lf.para` |
| `--tol` | 覆盖收敛阈值 |
| `--max-iter` | 覆盖最大迭代次数 |
| `--min-voltage` | 覆盖最小电压 |
| `--quiet` | 抑制迭代输出 |

## 拓扑对象

### HybridACGrid

负责 AC 设备关联、拓扑岛划分、存活状态判断和 AC 拓扑校验。支持多平衡节点冗余校验：当多个平衡节点通过零阻抗等值相连且固定电压/相角一致时，作为冗余参考处理。

### HybridDCGrid

负责 DC 设备关联、拓扑岛划分、存活状态判断和 DC 拓扑校验。要求每个存活 DC 岛有电压控制源。

### HybridIsland

表示交直流联合拓扑岛。AC 岛、DC 岛可通过 DCAC、DCDC、ACAC 等跨网络设备合并为一个 HybridIsland。

### HybridPowerNetwork

统一持有 AC 子网、DC 子网、跨域变流器和 hybrid island。`prepare()` 负责：

1. AC 关联和拓扑。
2. DC 关联和拓扑。
3. 交直流联合拓扑合并。
4. AC/DC 拓扑检查。
5. DCAC/ACAC 跨网设备检查。

## 全局状态向量

`HybridPowerFlowCalc` 的状态向量由以下部分拼接：

```text
x = [ac_x, dc_x, dcac_x, acac_x]
```

| 子向量 | 内容 |
| --- | --- |
| `ac_x` | `ACPowerFlowCalc` 的 AC 状态 |
| `dc_x` | `DCPowerFlowCalc` 的 DC 状态 |
| `dcac_x` | 每台 DCAC 变流器 3 个变量：`P_DC, P_AC, Q_AC` |
| `acac_x` | 每台 ACAC 变流器 4 个变量：`P_i, Q_i, P_j, Q_j` |

## 全局残差方程

`get_f(x)` 组装全局残差：

1. 调用 AC 子求解器生成 AC 节点和零阻抗残差。
2. 调用 DC 子求解器生成 DC 节点和零阻抗残差。
3. 将 DCAC 变流器 AC 端功率注入 AC 节点平衡行。
4. 将 DCAC 变流器 DC 端功率注入 DC 节点平衡行。
5. 追加 DCAC 损耗方程、控制方程和无功控制方程。
6. 将 ACAC 两端功率注入对应 AC 节点平衡行。
7. 追加 ACAC 损耗方程和控制方程。

## DCAC 变流器模型

`DCACConverter` 使用 `r1 + 理想变换 + r2` 模型。每台 DCAC 变流器状态：

```text
[P_DC, P_AC, Q_AC]
```

损耗方程形式：

```text
Vdc^2 * Vac^2 * (Pdc + Pac)
- r1 * Pdc^2 * Vac^2
- r2 * (Pac^2 + Qac^2) * Vdc^2 = 0
```

控制模式：

| `control_type` | 控制方程 |
| --- | --- |
| 定 DC 电压 | `Vdc - v_dc_set = 0` |
| 定 AC 电压 | `Vac - v_ac_set = 0` |
| 定 AC 有功 | `P_AC - p_ac_set = 0` |

第三个方程固定 AC 无功：

```text
Q_AC - q_ac_set = 0
```

## ACAC 变流器模型

`ACACConverter` 用于交流柔性互联，状态：

```text
[P_i, Q_i, P_j, Q_j]
```

损耗方程：

```text
Vi^2 * Vj^2 * (P_i + P_j)
- r1 * (P_i^2 + Q_i^2) * Vj^2
- r2 * (P_j^2 + Q_j^2) * Vi^2 = 0
```

其余控制方程根据 `control_type` 在端口无功和端口电压之间选择。

## Jacobian

`get_jacobi(x)` 生成全局稀疏 Jacobian：

- AC 子块来自 `ACPowerFlowCalc.get_jacobi()`。
- DC 子块来自 `DCPowerFlowCalc.get_jacobi()`。
- DCAC 和 ACAC 的注入耦合、损耗方程、控制方程直接以 COO triplet 追加。
- 最后统一拼接为 CSR 矩阵。

## 求解流程

`run()` 执行统一 Newton 迭代：

1. 若状态为空，先调用 `prepare()`。
2. 计算全局残差 `F`。
3. 分别记录 AC、DC 残差范数，同时使用全局 `normF` 判收敛。
4. 构造全局 Jacobian。
5. 稀疏直接解 `J * delta = F`。
6. 更新 `x = x - delta`。
7. 收敛后 `_write_back(x)`。

## 结果对象

`HybridPowerFlowResult` 包含：

| 字段 | 含义 |
| --- | --- |
| `network` | HybridPowerNetwork |
| `ac_network` | AC 子网 |
| `dc_network` | DC 子网 |
| `calc` | HybridPowerFlowCalc |
| `ac` | AC 子求解器，可能为 `None` |
| `dc` | DC 子求解器，可能为 `None` |
| `rc` | 返回码 |
| `ac_warnings/ac_errors` | AC 拓扑信息 |
| `dc_warnings/dc_errors` | DC 拓扑信息 |

## 性能设计

- AC/DC 子系统只在全局残差/Jacobian 中调用一次。
- DCAC/ACAC 变流器数组化缓存节点位置、控制模式、方程行列索引。
- Jacobian 按稀疏 triplet 直接生成，避免构造全局稠密矩阵。
- 支持纯 AC/纯 DC 时自动跳过空子系统。

## 注意事项

- 交直流联合潮流不是交替迭代，不能把 AC 和 DC 结果分别求完后简单拼接。
- DCAC/ACAC 变流器两端节点必须属于存活拓扑。
- 变流器控制模式必须提供足够约束，否则全局 Jacobian 会奇异。
