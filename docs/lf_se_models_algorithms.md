# LF/SE 模型定义、算法原理及求解过程

本文档汇总当前工程中 6 个核心计算模块的建模口径、算法原理和主流程：

| 模块 | 核心类 | 源码 |
| --- | --- | --- |
| 交流潮流 | `ACPowerFlowCalc` | `src/hybrid_power_system_analysis/lfcore/ac_lf.py` |
| 直流潮流 | `DCPowerFlowCalc` | `src/hybrid_power_system_analysis/lfcore/dc_lf.py` |
| 交直流联合潮流 | `HybridPowerFlowCalc` | `src/hybrid_power_system_analysis/lfcore/hybrid_lf.py` |
| 交流状态估计 | `ACStateEstimator` | `src/hybrid_power_system_analysis/secore/ac_se.py` |
| 直流状态估计 | `DCStateEstimator` | `src/hybrid_power_system_analysis/secore/dc_se.py` |
| 交直流联合状态估计 | `HybridStateEstimator` | `src/hybrid_power_system_analysis/secore/hybrid_se.py` |

文档重点说明程序内部实际采用的数组化 PPC 口径。`full` 结果模式只是在最后把数组结果包装成对象视图或完整结果表，不改变中间计算模型。

## 1. 公共建模约定

### 1.1 E 文件到 PPC

E 文件提供设备拓扑、参数、控制设定和单位信息。读取后先转换成内部 PPC 字典：

| PPC 字段 | 含义 |
| --- | --- |
| `base` | 系统功率、电压、电流单位和标幺换算基准 |
| `bus` | AC 或 DC 节点数组 |
| `branch` | 普通线路或支路数组 |
| `transformer` | AC 主变数组 |
| `load` | ZIP 负荷数组 |
| `gen` | 发电机或电源数组 |
| `shunt` | AC 并联补偿数组 |
| `zero_branch` | 零阻抗支路数组 |
| `switch` / `break` | 开关和刀闸数组 |
| `dcdc` | DC/DC 变流器数组 |
| `dcac` | DC/AC 变流器数组 |
| `acac` | AC/AC 变流器数组 |
| `_topology_arrays` | 拓扑分析后的存活节点、母线、岛、设备端点索引 |

有名值输入通过 `p_unit`、`u_unit`、`i_unit` 自动转换为 `p_scale`、`u_scale`、`i_scale`，再进入标幺计算。相角文件值使用度，内部计算使用弧度。

### 1.2 拓扑和设备状态

公共状态字段：

| 字段 | 含义 |
| --- | --- |
| `idx` | 节点或设备编号 |
| `name` | 设备名称，用于量测和输出定位 |
| `run_stat` | 运行状态，`1` 表示参与计算 |
| `status` | 开关类设备闭合状态，`1` 表示闭合 |
| `i_node` / `j_node` | 两端节点 |
| `node` | 单端设备接入节点 |

拓扑处理完成后，只保留带有效参考源或电压源的存活岛。零阻抗支路、闭合开关和闭合刀闸不以极小阻抗代替，而是在 LF 中作为约束和显式电流状态处理，在 SE 中做状态压缩或显式电流状态处理。

### 1.3 结果模式

LF 和 SE 都支持 `result_mode`：

| 模式 | 计算中间过程 | 最终输出 |
| --- | --- | --- |
| `none` | 数组化 | 不构造结果对象，只保留收敛信息和最终状态 |
| `array` | 数组化 | 返回数组结果，不构造 full 表和对象视图 |
| `summary` | 数组化 | 构造摘要统计 |
| `full` | 数组化 | 最后构造完整结果对象、结果表或对象回填 |

当前实现的原则是：无论 `none/array/summary/full`，主计算过程都按 array-only 路径执行，避免同一份数据同时维护对象网络和数组网络。

## 2. 潮流模型定义

### 2.1 AC 潮流模型

AC 潮流采用极坐标 Newton-Raphson。节点复电压为：

```text
V_i_complex = V_i * exp(j * theta_i)
```

主要设备：

| 设备 | 模型 |
| --- | --- |
| `ACNode` | 电压幅值和相角载体，按发电机控制分成 PQ、PV、Slack |
| `ACBranch` | 串联阻抗 `r + jx`，线路充电电纳 `b` 两端平分 |
| `ACTransformer` | T 型单端对地模型，`gt/bt` 是 i 侧单端对地电导和电纳 |
| `ACLoad` | ZIP 负荷，`P = pbase*(pv0 + pv1*V + pv2*V^2)`，`Q` 同理 |
| `ACGenerator` | 支持 `PQ/P/PV/V/SLACK/PH` 等控制 |
| `ACShuntCompensator` | 按并联导纳、定无功或电压控制语义参与计算 |
| `ACZeroBranch` / `ACSwitch` / `ACBreak` | 闭合时按零阻抗约束和显式电流状态处理 |
| `ACACConverter` | 交流柔性互联设备，两端都是 AC 节点，状态为两端 P/Q |

状态向量：

```text
x_ac = [theta_unknown, V_unknown, phi_re, phi_im]
```

| 状态 | 含义 |
| --- | --- |
| `theta_unknown` | 非参考节点相角 |
| `V_unknown` | PQ 节点电压幅值 |
| `phi_re/phi_im` | 零阻抗连通分量辅助势变量，用于表达零阻抗支路电流 |

节点类型由发电机控制决定：

| 控制 | 节点类型 | 方程 |
| --- | --- | --- |
| `PQ` | PQ | P/Q 平衡 |
| `P` | PV 语义中的定 P | P 平衡，电压约束由其他电压源或拓扑决定 |
| `PV` | PV | P 平衡，V 固定 |
| `V/SLACK/PH` | Slack | V 和相角固定，承担平衡功率 |

主变 stamp：

```text
y = 1 / (r + j*x)
yt = gt + j*bt
tapc = tap * exp(j*shift)
Yff = (y + yt) / (tapc * conj(tapc))
Yft = -y / conj(tapc)
Ytf = -y / tapc
Ytt = y
```

该模型与 MATPOWER 标准 branch 的两端平分 `BR_B` 不完全等价。

### 2.2 DC 潮流模型

DC 潮流使用节点电压 Newton。普通 DC 支路按电阻模型：

```text
I_ij = (V_i - V_j) / r
P_i = V_i * I_ij
P_j = -V_j * I_ij
```

主要设备：

| 设备 | 模型 |
| --- | --- |
| `DCNode` | 直流节点电压 |
| `DCBranch` | 电阻支路 |
| `DCLoad` | 电压相关 ZIP 负荷 |
| `DCGenerator` | 支持 `P/V/I` 控制 |
| `DCZeroBranch` / `DCSwitch` / `DCBreak` | 闭合时按等电位约束或显式电流处理 |
| `DCDCConverter` | 两端 DC 变换设备，使用 `r1 + 理想变换 + r2` 损耗模型 |

状态向量：

```text
x_dc = [V, phi, P_dcdc]
```

| 状态 | 含义 |
| --- | --- |
| `V` | 存活 DC 节点电压 |
| `phi` | 零阻抗辅助势变量 |
| `P_dcdc` | DCDC 端口功率状态，array/summary/none 模式下对部分端口可闭式回算 |

DCDC 控制采用双端控制字段：

```text
i_control_type, j_control_type in {CTRL_P, CTRL_V, CTRL_I, SLACK}
```

约束规则是：一端必须为 `CTRL_P/CTRL_V/CTRL_I`，另一端必须为 `SLACK`。定功率、定电压、定电流端给出控制方程，Slack 端由损耗方程和平衡关系确定。

### 2.3 Hybrid 潮流模型

Hybrid 潮流不是顺序调用 AC 和 DC，而是拼成一个全局 Newton 系统：

```text
x_hybrid = [x_ac, x_dc, x_dcac, x_acac]
```

| 子向量 | 内容 |
| --- | --- |
| `x_ac` | AC 子系统状态 |
| `x_dc` | DC 子系统状态 |
| `x_dcac` | 每台 DCAC 的 `[P_DC, P_AC, Q_AC]` |
| `x_acac` | 每台 ACAC 的 `[P_i, Q_i, P_j, Q_j]` |

DCAC 变流器模型：

```text
Vdc^2 * Vac^2 * (Pdc + Pac)
- r1 * Pdc^2 * Vac^2
- r2 * (Pac^2 + Qac^2) * Vdc^2 = 0
```

控制类型：

| `control_type` | 控制方程 |
| --- | --- |
| `DCV` | DC 端电压固定 |
| `ACV` | AC 端电压固定，旧输入别名 `PH` 也按 ACV 处理 |
| `ACP` | AC 端有功固定，旧输入别名 `PQ` 也按 ACP 处理 |

ACAC 变流器模型：

```text
Vi^2 * Vj^2 * (P_i + P_j)
- r1 * (P_i^2 + Q_i^2) * Vj^2
- r2 * (P_j^2 + Q_j^2) * Vi^2 = 0
```

控制类型：

| `control_type` | i 端 | j 端 |
| --- | --- | --- |
| `PQQ` | 定 P、定 Q | 定 Q |
| `PVQ` | 定 P、定 V | 定 Q |
| `PQV` | 定 P、定 Q | 定 V |
| `PVV` | 定 P、定 V | 定 V |

`ACACConverter` 当前属于 AC 模型设备。导出 MATPOWER/PYPOWER case 时，两端口会按端口功率方向投影为 PQ 负荷或 PQ/PV/PH 电源。

### 2.4 元件支路结构图与运行方程

本节给出 LF 和 SE 共用的设备运行方程。除特别说明外，端口功率正方向约定为从节点流入设备；发电机出力正方向为从设备注入节点。AC 复电压记为 `U_i = V_i * exp(j*theta_i)`，端口复功率为 `S_i = P_i + jQ_i = U_i * conj(I_i)`。

#### ACBranch

支路结构：

```text
          j*b/2                 j*b/2
           |                     |
AC i o-----+----[ r + j*x ]------+-----o AC j
           |                     |
         ground                ground
```

运行方程：

```text
z = r + j*x
y = 1 / z
I_i = (y + j*b/2) * U_i - y * U_j
I_j = -y * U_i + (y + j*b/2) * U_j
S_i = U_i * conj(I_i)
S_j = U_j * conj(I_j)
```

`i_p/i_q/i_c`、`j_p/j_q/j_c` 分别由 `S_i`、`S_j` 和端口电流幅值回填。

#### ACTransformer

支路结构：

```text
AC i o--[ tap∠shift ]--o a--[ r + j*x ]--o AC j
                         |
                       gt+j*bt
                         |
                       ground
```

运行方程：

```text
y = 1 / (r + j*x)
yt = gt + j*bt
tapc = tap * exp(j*shift)
Yff = (y + yt) / (tapc * conj(tapc))
Yft = -y / conj(tapc)
Ytf = -y / tapc
Ytt = y
I_i = Yff * U_i + Yft * U_j
I_j = Ytf * U_i + Ytt * U_j
S_i = U_i * conj(I_i)
S_j = U_j * conj(I_j)
```

`gt/bt` 是 i 侧单端对地导纳，不是线路两端平分充电电纳。

#### ACLoad

支路结构：

```text
AC i o----[ ZIP load ]----ground
```

运行方程：

```text
P_load(V_i) = pbase * (pv0 + pv1 * V_i + pv2 * V_i^2)
Q_load(V_i) = qbase * (qv0 + qv1 * V_i + qv2 * V_i^2)
S_load = P_load + j*Q_load
I_load = conj(S_load / U_i)
```

负荷正值表示从节点吸收功率，在节点功率平衡中作为负注入。

#### ACGenerator

支路结构：

```text
generator ----> AC i
```

运行方程和控制约束：

| 控制 | 方程或约束 | 结果回填 |
| --- | --- | --- |
| `PQ` | `P_gen = p_set`, `Q_gen = q_set` | 固定出力 |
| `P` | `P_gen = p_set` | `Q_gen` 由节点无功平衡或结果分摊得到 |
| `PV` | `P_gen = p_set`, `V_i = v_set` | `Q_gen` 由无功平衡得到 |
| `V/SLACK/PH` | `V_i = v_set`, `theta_i = theta_set` | `P_gen/Q_gen` 承担岛内平衡功率 |

多台发电机接在同一节点时，平衡功率按 `alpha` 或内部分摊规则分配。

#### ACShuntCompensator

支路结构：

```text
AC i o----[ g_set + j*b_set or Q/V control ]----ground
```

运行方程：

```text
Y_sh = g_set + j*b_set
I_sh = Y_sh * U_i
S_sh = U_i * conj(I_sh)
P_sh = V_i^2 * g_set
Q_sh = -V_i^2 * b_set
```

当设备采用定 `Q` 控制时，`Q_sh = q_set`。当采用电压控制时，SE 中可把 shunt 无功作为显式状态，用电压量测或控制约束约束该状态。

#### ACZeroBranch / ACSwitch / ACBreak

支路结构：

```text
AC i o----[ zero impedance / closed switch ]----o AC j
```

运行方程：

```text
U_i - U_j = 0
I_ij = phi_i - phi_j
I_i = I_ij
I_j = -I_ij
S_i = U_i * conj(I_i)
S_j = U_j * conj(I_j)
```

闭合开关和闭合刀闸按零阻抗支路处理；断开设备不参与拓扑和方程。为消除 `phi` 平移自由度，每个零阻抗连通分量固定一个 `phi = 0`。

#### ACACConverter

支路结构：

```text
AC i o----[ r1 + ideal AC/AC converter + r2 ]----o AC j
```

端口状态：

```text
x_acac = [P_i, Q_i, P_j, Q_j]
```

损耗方程：

```text
V_i^2 * V_j^2 * (P_i + P_j)
- r1 * (P_i^2 + Q_i^2) * V_j^2
- r2 * (P_j^2 + Q_j^2) * V_i^2 = 0
```

控制方程：

```text
P_i - p_set = 0
```

其余两条控制方程由 `control_type` 决定：

| 控制 | i 端方程 | j 端方程 |
| --- | --- | --- |
| `PQQ` | `Q_i - i_q_set = 0` | `Q_j - j_q_set = 0` |
| `PVQ` | `V_i - i_v_set = 0` | `Q_j - j_q_set = 0` |
| `PQV` | `Q_i - i_q_set = 0` | `V_j - j_v_set = 0` |
| `PVV` | `V_i - i_v_set = 0` | `V_j - j_v_set = 0` |

端口功率正值表示对应 AC 节点向变流器送入功率；负值表示变流器向节点注入功率。

#### DCBranch

支路结构：

```text
DC i o----[ r ]----o DC j
```

运行方程：

```text
I_ij = (V_i - V_j) / r
P_i = V_i * I_ij
P_j = -V_j * I_ij
```

`P_i + P_j = r * I_ij^2` 为支路损耗。

#### DCLoad

支路结构：

```text
DC i o----[ ZIP load ]----ground
```

运行方程：

```text
P_load(V_i) = pbase * (pv0 + pv1 * V_i + pv2 * V_i^2)
I_load = P_load / V_i
```

负荷正值表示从 DC 节点吸收功率。

#### DCGenerator

支路结构：

```text
DC source ----> DC i
```

运行方程和控制约束：

| 控制 | 方程或约束 | 结果回填 |
| --- | --- | --- |
| `CTRL_P` / `P` | `P_gen = p_set` | `I_gen = P_gen / V_i` |
| `CTRL_I` / `I` | `I_gen = i_set` | `P_gen = V_i * i_set` |
| `CTRL_V` / `V` | `V_i = v_set` | `P_gen` 承担 DC 岛平衡功率 |

#### DCZeroBranch / DCSwitch / DCBreak

支路结构：

```text
DC i o----[ zero resistance / closed switch ]----o DC j
```

运行方程：

```text
V_i - V_j = 0
I_ij = phi_i - phi_j
P_i = V_i * I_ij
P_j = -V_j * I_ij
```

闭合开关和闭合刀闸按等电位约束处理；断开设备不参与方程。SE 中只有当量测需要显式电流时，才为 DCZeroBranch 创建电流状态。

#### DCDCConverter

支路结构：

```text
DC i o----[ r1 + ideal DC/DC converter + r2 ]----o DC j
```

端口状态：

```text
x_dcdc = [P_i, P_j]
```

损耗方程：

```text
V_i^2 * V_j^2 * (P_i + P_j)
- r1 * P_i^2 * V_j^2
- r2 * P_j^2 * V_i^2 = 0
```

控制方程由 `i_control_type` 和 `j_control_type` 决定，且只能一端为控制端，另一端为 `SLACK`：

| 控制端类型 | 控制方程 |
| --- | --- |
| `CTRL_P` | `P_terminal - p_set = 0` |
| `CTRL_V` | `V_terminal - v_set = 0` |
| `CTRL_I` | `P_terminal - i_set * V_terminal = 0` |
| `SLACK` | 不直接给控制方程，由损耗方程和节点平衡确定 |

端口功率正值表示对应 DC 节点向 DCDC 送入功率。

#### DCACConverter

支路结构：

```text
DC d o----[ r1 + ideal DC/AC converter + r2 ]----o AC a
```

端口状态：

```text
x_dcac = [P_DC, P_AC, Q_AC]
```

损耗方程：

```text
V_dc^2 * V_ac^2 * (P_DC + P_AC)
- r1 * P_DC^2 * V_ac^2
- r2 * (P_AC^2 + Q_AC^2) * V_dc^2 = 0
```

控制方程：

| `control_type` | 第一控制方程 | 第二控制方程 |
| --- | --- | --- |
| `DCV` | `V_dc - v_dc_set = 0` | `Q_AC - q_ac_set = 0` |
| `ACV` | `V_ac - v_ac_set = 0` | `Q_AC - q_ac_set = 0` |
| `ACP` | `P_AC - p_ac_set = 0` | `Q_AC - q_ac_set = 0` |

端口功率正值表示对应 AC 或 DC 节点向 DCAC 送入功率。`ACV` 表示交流侧定电压，也就是旧模型语义里的交流侧 PH 模式。

## 3. 潮流算法原理

### 3.1 Newton-Raphson 方程

LF 的基本形式为：

```text
F(x) = 0
J(x) * dx = F(x)
x <- x - dx
```

收敛判据：

```text
normF = ||F(x)||_inf < tol
```

`tol/max_iter/min_voltage/divergence_threshold` 来自 `lf.para`，也可由命令行或构造函数覆盖。

### 3.2 AC LF 求解过程

`ACPowerFlowCalc.run()` 主流程：

1. `prepare()`，构建 AC PPC 拓扑、存活节点、节点类型、状态索引。
2. 生成 Y 矩阵、负荷数组、发电机控制数组、零阻抗约束数组。
3. 构造初始状态，平衡节点和 PV 节点电压按设定值固定。
4. 每轮计算 `F_ac(x)`，包括 P/Q 平衡和零阻抗等值约束。
5. 构造稀疏 Jacobian。
6. 使用线性求解器解 `J dx = F`，更新 `x = x - dx`。
7. 收敛后计算节点电压、支路端口功率、电流、发电机平衡功率和负荷实际功率。

默认线性求解器是 `pyklu`，不可用或失败时回退到 SciPy SuperLU。

### 3.3 DC LF 求解过程

`DCPowerFlowCalc.run()` 主流程：

1. `prepare()`，构建 DC PPC 拓扑、存活节点、参考电压源和 DCDC 索引。
2. 构造电导矩阵、负荷数组、零阻抗约束和 DCDC 控制数组。
3. 构造初始状态。
4. 每轮计算节点功率平衡、定电压约束、零阻抗等电位约束、DCDC 控制和损耗方程。
5. 构造稀疏 Jacobian 并解线性方程。
6. 收敛后回填节点电压、支路功率、负荷功率、电源功率、DCDC 两端功率和电流。

默认线性求解器是 `pyklu`，失败时回退到 SciPy SuperLU。含复杂 DC 耦合的 Hybrid 求解通常默认使用 `umfpack`，用于规避部分 KLU 失败场景。

### 3.4 Hybrid LF 求解过程

`HybridPowerFlowCalc.run()` 主流程：

1. 读取或接收 hybrid PPC，拆出 AC PPC、DC PPC、DCAC、ACAC。
2. `prepare()` 分别准备 AC 和 DC 子求解器，并缓存跨域设备的节点位置、控制码、方程行列索引。
3. 若只有纯 AC 或纯 DC 且无跨域设备，直接复用单域 Newton 块，避免全局包装。
4. 含耦合设备时构造全局残差：
   - AC 子残差。
   - DC 子残差。
   - DCAC 的 AC/DC 端功率注入。
   - ACAC 的两端 AC 功率注入。
   - DCAC/ACAC 损耗和控制方程。
5. 构造全局 Jacobian，AC/DC 子块和变流器耦合块统一拼接。
6. 解全局 `J dx = F` 并更新。
7. 收敛后按 `result_mode` 输出 AC、DC、DCAC、ACAC 数组或完整结果。

默认求解器选择：

| 场景 | 默认求解器 |
| --- | --- |
| AC-only hybrid | `pyklu` |
| 含 DC 或跨域耦合 | `umfpack` |
| 指定求解器不可用或失败 | 回退 SciPy SuperLU |

## 4. 状态估计模型定义

### 4.1 量测模型

SE 量测文件使用 `<Measurement>`：

```text
idx name dev_type dev_name meas_type weight valid value
```

读取后转成量测 PPC/量测数组，核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `value` | 标幺或弧度量测值 |
| `weight` | WLS 权重 |
| `valid` | 是否有效 |
| `status_code` | 正常、无效、伪量测等结构化状态 |
| `device_type_code` | 整数设备类型 |
| `device_pos` | 设备在 PPC 表中的行号或局部位置 |
| `meas_type_code` | 整数量测类型 |

主路径不依赖字符串查找设备。字符串字段主要用于文件解析初期、输出展示和诊断信息。

### 4.2 AC SE 状态

AC SE 状态向量：

```text
x_ac_se = [theta_non_ref, V_all, I_zero_re, I_zero_im, P/Q_gen, P/Q_load, Q_shunt, P/Q_acac]
```

实际状态布局会按可计算设备压缩：

| 状态 | 含义 |
| --- | --- |
| `theta_non_ref` | 每个 AC 岛去掉一个参考角后的相角状态 |
| `V_all` | 存活节点电压状态 |
| `I_zero_re/I_zero_im` | ACZeroBranch/ACSwitch/ACBreak 需要的显式电流状态 |
| `P/Q_gen` | 发电机出力显式状态 |
| `P/Q_load` | 负荷功率显式状态 |
| `Q_shunt` | 电压控制 shunt 无功显式状态 |
| `P/Q_acac` | ACAC 两端端口功率状态 |

零阻抗或开关两端节点的电压/相角状态按 zero tie 压缩到等值状态列，但设备自身量测仍可通过显式电流状态或端口函数表达。

### 4.3 DC SE 状态

DC SE 状态向量：

```text
x_dc_se = [V_all, I_switch_or_zero, P_dcdc_from, P_dcdc_to, P_v_generator]
```

| 状态 | 含义 |
| --- | --- |
| `V_all` | 存活 DC 节点电压 |
| `I_switch_or_zero` | DCSwitch 或实际需要的 DCZeroBranch 显式电流 |
| `P_dcdc_from/P_dcdc_to` | DCDC 两端功率 |
| `P_v_generator` | 定电压 DCGenerator 的平衡功率 |

对没有功率或电流量测需求的 DCZeroBranch，不强制创建电流状态，避免无意义自由度。

### 4.4 Hybrid SE 状态

Hybrid SE 使用紧凑统一状态：

```text
x_hybrid_se = [
  AC theta/V/zero current,
  DC V/zero current,
  DCDC powers,
  DC V-generator powers,
  DCAC powers,
  ACAC powers
]
```

其中 AC/DC 单域状态尽量复用对应 SE 的 PPC 和拓扑数组，跨域设备按端口功率状态进入统一 WLS。

## 5. 状态估计算法原理

### 5.1 WLS 目标函数

状态估计采用加权最小二乘：

```text
min 0.5 * r(x).T * W * r(x)
r(x) = z - h(x)
```

其中：

| 符号 | 含义 |
| --- | --- |
| `z` | 有效量测值 |
| `h(x)` | 由当前状态计算的量测函数 |
| `W` | 对角权重矩阵 |
| `r` | 残差 |

每轮线性化：

```text
H = dh/dx
G = H.T * W * H
rhs = H.T * W * r
G * dx = rhs
x <- x + step * dx
```

`step` 由线搜索确定，用于保证目标函数下降并保护电压下限。

### 5.2 可观测性

可观测性判据：

```text
rank(H) == state_count
```

实现上优先复用正规矩阵、Cholesky 或稀疏因子信息。只有在必要时才退化到更昂贵的秩分析。不可观测时会返回弱状态或自由度信息。

### 5.3 伪量测

伪量测用于补足可观测性，权重低于真实量测。主要来源：

| 类型 | 典型内容 |
| --- | --- |
| 设备缺失量测 | 发电机、负荷、shunt、变流器等缺少关键 P/Q/V 量测 |
| 拓扑约束 | 零阻抗等值、开关等值、电压差或角差约束 |
| 定向可观测性补强 | 针对可观测性弱方向生成少量 targeted pseudo |

生成伪量测时使用 `device_type_code/device_pos/meas_type_code` 直接定位，不在主路径通过名字反查设备。

### 5.4 坏数据辨识

坏数据使用归一化残差：

```text
rN_i = |r_i| / sqrt(R_ii - h_i * G^-1 * h_i.T)
```

当 `rN_i > bad_threshold` 时，量测被标记为坏数据。`remove_bad_data=True` 时可迭代剔除最大归一化残差量测，剔除次数受 `max_remove` 限制。

`result_mode="array"` 不跳过坏数据识别。坏数据识别如需构造展示项，会在最后阶段按已有数组行构造，不影响 WLS 主路径。

## 6. 状态估计求解过程

### 6.1 AC SE 流程

`ACStateEstimator.run()` 主流程：

1. 加载 AC E 文件，构建 AC PPC 和拓扑数组。
2. 加载 `.meas`，解析为量测 PPC/量测数组。
3. 根据设备类型码、设备位置、量测类型码生成量测计划。
4. 转换量测到标幺值和弧度，过滤无效量测和离线设备量测。
5. 根据 `flat_start` 选择平启动或 AC LF 种子。
6. 添加必要伪量测和 targeted observability pseudo。
7. 构建状态布局、量测函数计划、Jacobian 固定 pattern 或刷新计划。
8. WLS 迭代：
   - 计算 `h(x)` 和残差。
   - 刷新稀疏 Jacobian data。
   - 构造 lower normal CSC 或 solver-ready 正规方程。
   - 求解 `dx`，线搜索更新。
9. 可观测性分析和坏数据辨识。
10. 根据 `result_mode` 构造数组、摘要或完整 `SEResult`。

### 6.2 DC SE 流程

`DCStateEstimator.run()` 与 AC SE 保持同类结构：

1. 加载 DC E 文件，构建 DC PPC 和拓扑数组。
2. 加载量测 PPC，建立整数码量测计划。
3. 转换单位并过滤无效量测。
4. 平启动或运行 DC LF 生成种子。
5. 添加电源、负荷、DCDC、拓扑约束等伪量测。
6. 构建 DC 状态布局和按设备类别分组的量测计划。
7. WLS 迭代，刷新 `h/H/G/rhs` 并求解。
8. 可观测性分析、坏数据辨识和结果输出。

DC SE 与 AC SE 的差异主要来自物理模型：DC 没有相角和无功；DCSwitch/DCDC/定电压电源状态替代 AC 中的 Q、相角和零阻抗复电流模型。

### 6.3 Hybrid SE 流程

`HybridStateEstimator.run()` 主流程：

1. 读取 hybrid E 文件，拆成 AC PPC、DC PPC、DCAC、ACAC。
2. 构建 hybrid topology，确认 AC/DC 子岛和跨域设备连通关系。
3. 根据 `flat_start` 选择平启动或运行 `HybridPowerFlowCalc` 生成物理一致种子。
4. 加载 `.meas`，将 AC、DC、DCDC、DCAC、ACAC 量测统一成数组计划。
5. 构建紧凑状态布局，AC zero tie 状态压缩，DC zero/switch 状态按量测需求保留。
6. active measurement 走快速评估路径，只处理有效量测行。
7. `_assemble_jacobian()` 按 AC、DC、变流器分块刷新 Jacobian。
8. 构造正规方程并求解 WLS 修正量。
9. 可观测性、坏数据辨识、结果输出。

纯 AC 或纯 DC 文件进入 `hybrid_se` 时，会跳过不存在的子系统和跨域设备。但若只关心单域调试，`ac_se` 或 `dc_se` 更便于定位。

## 7. 六个模块的差异对照

| 维度 | `ac_lf` | `dc_lf` | `hybrid_lf` | `ac_se` | `dc_se` | `hybrid_se` |
| --- | --- | --- | --- | --- | --- | --- |
| 数学问题 | 非线性方程 | 非线性方程 | 全局非线性方程 | WLS | WLS | 全局 WLS |
| 主要状态 | `theta/V/phi` | `V/phi/DCDC P` | `ac_x/dc_x/DCAC/ACAC` | AC 紧凑估计状态 | DC 紧凑估计状态 | AC/DC/变流器紧凑状态 |
| 主要矩阵 | `J = dF/dx` | `J = dF/dx` | 全局 `J` | `H` 和 `G=H.TWH` | `H` 和 `G=H.TWH` | 全局 `H/G` |
| 参考自由度 | 每个 AC 岛一个角参考 | 每个 DC 岛一个电压源 | AC/DC 子岛分别处理 | 每个 AC 岛去掉一个角自由度 | 电压源和量测共同约束 | AC 角参考、DC 电压源和跨域量测共同约束 |
| 零阻抗处理 | 显式 `phi_re/phi_im` | 显式 `phi` | 复用 AC/DC 子模型 | zero tie 压缩加必要电流状态 | 按量测需求保留电流状态 | AC/DC 分别按紧凑策略处理 |
| 线性求解 | 稀疏直接解 | 稀疏直接解 | 稀疏直接解 | 正规方程求解 | 正规方程求解 | 正规方程求解 |
| 结果模式 | `none/array/summary/full` | 同左 | 同左 | 同左 | 同左 | 同左 |

## 8. 调用和调试建议

### 8.1 选择求解器

| 场景 | 推荐入口 |
| --- | --- |
| 纯 AC 潮流 | `ac_lf.py` |
| 纯 DC 潮流 | `dc_lf.py` |
| 含 DCAC、DCDC、ACAC 或混合拓扑潮流 | `hybrid_lf.py` |
| 纯 AC 状态估计调试 | `ac_se.py` |
| 纯 DC 状态估计调试 | `dc_se.py` |
| 混合系统状态估计 | `hybrid_se.py` |
| 统一入口批量验证 | `hybrid_lf.py` / `hybrid_se.py` |

### 8.2 性能测试口径

- 冷启动端到端：单独启动 Python 进程，包含 import、文件加载、PPC 构造、拓扑、计算和输出。
- 计算核心耗时：复用已加载进程，统计 `prepare/run/estimate/linear_solve/jacobian/normal_equations` 等内部阶段。
- 大规模对比建议使用 `result_mode="array"` 或 `result_mode="none"`，避免 full 结果对象构造掩盖计算核心耗时。

### 8.3 结果一致性检查

LF 与 SE 对比时，应优先检查：

| 对比项 | 说明 |
| --- | --- |
| 节点电压幅值 | AC/DC 都适用 |
| AC 相角 | 注意每个 AC 岛参考角选择 |
| 设备端口 P/Q/I | AC 设备和变流器 |
| DC 设备 P/I | DC 支路、电源、负荷和 DCDC |
| 残差和目标函数 | 判断收敛质量 |
| 伪量测和坏数据分布 | 判断量测质量和可观测性 |

与 MATPOWER/PYPOWER 对比时，需要注意本工程 `ACTransformer` 的 `gt/bt` 是单端对地 T 型模型，不能无损映射成 MATPOWER branch 的两端平分 `BR_B`。
