# hybrid_se 技术文档

## 模块定位

`src/hybrid_power_system_analysis/secore/hybrid_se.py` 实现交直流混联系统加权最小二乘状态估计。核心类是 `HybridStateEstimator`。

该模块用于同一套状态估计程序处理：

- 纯交流模型文件。
- 纯直流模型文件。
- 交直流联合模型文件。

与分别调用 `ac_se`、`dc_se` 不同，`hybrid_se` 基于 `HybridPowerFlowCalc` 的统一状态，构建一个覆盖 AC、DC、DCAC、DCDC、ACAC 的紧凑 WLS 状态向量，并统一生成量测函数、Jacobian、正规方程和坏数据辨识结果。

## 入口与使用方式

```python
from secore.hybrid_se import HybridStateEstimator

estimator = HybridStateEstimator(
    e_file="data/model/hybrid/qinling.e",
    meas_file="data/meas/hybrid/qinling.meas",
    flat_start=True,
)
result = estimator.estimate(verbose=False)
```

命令行：

```powershell
python -m secore.hybrid_se --case data\model\hybrid\qinling.e --meas data\meas\hybrid\qinling.meas --flat-start --quiet
```

参数与 `ac_se.py`、`dc_se.py` 基本一致。

## 初始化流程

`HybridStateEstimator.__init__()` 主要流程：

1. 读取 `se.para` 并处理覆盖参数。
2. 读取 E 文件为 `HybridPowerNetwork`。
3. 运行 `network.prepare()` 完成 AC/DC/Hybrid 拓扑检查。
4. 构造 `HybridPowerFlowCalc` 并运行潮流，得到物理一致种子 `power_flow_x`。
5. 读取 `.meas` 文件。
6. 构建设备名索引。
7. 过滤无效量测。
8. 将量测有名值转换到内部标幺值。
9. 自动添加无量测设备的低权重伪量测。
10. 构建紧凑估计状态布局。
11. 构建导数缓存、静态 Jacobian 索引和快速量测评估索引。

## 状态变量

`hybrid_se` 不直接使用完整潮流 Newton 向量，而是压缩为 WLS 紧凑状态：

| 状态类别 | 说明 |
| --- | --- |
| AC 相角 | 非参考、非零阻抗等值冗余节点相角 |
| AC 电压 | 非冗余 AC 电压状态 |
| AC 零阻抗电流 | 必要的 ACSwitch/ACZeroBranch 电流实部和虚部 |
| DC 电压 | DC 存活节点电压 |
| DC 零阻抗/开关电流 | DCSwitch 或实际被量测的 DCZeroBranch 电流 |
| DCDC 功率 | DCDC 两端功率 |
| DC 定电压源功率 | 定 V DCGenerator 的平衡功率 |
| DCAC 功率 | `P_DC, P_AC, Q_AC` |
| ACAC 功率 | `P_FROM, Q_FROM, P_TO, Q_TO` |

DCAC 状态和量测采用端口功率约定：每一端都以对应电网流入变流器为正。因此 `DC -> AC` 时 `P_DC > 0、P_AC < 0`，`AC -> DC` 时 `P_DC < 0、P_AC > 0`；`Q_AC > 0` 表示从 AC 电网吸收无功。`ACDCConverter` 与 `DCACConverter` 两种 `dev_type` 只表示设备类型，不改变状态、量测或残差符号。

平启动时，`ac_control_type=NONE、dc_control_type=P` 的变流器使用 `P_DC=p_dc_set、P_AC=-p_dc_set、Q_AC=q_ac_set` 作为耦合状态初值；SE 仍由量测和网络方程共同修正这些状态，不额外复制 LF 的硬控制方程。

零阻抗等值节点会映射到同一个状态列或通过虚拟电流状态表达，减少不可观测冗余。

## 支持量测

`hybrid_se` 支持 AC、DC 和变流器量测：

| 设备 | 量测 |
| --- | --- |
| `ACNode` | `V`, `ANGLE`/`THETA` |
| `DCNode` | `V` |
| `ACBranch`, `ACTransformer` | 两端 P/Q/V/I |
| `ACThreeWindingTransformer` | i/j/k 三端 P/Q/V/I，k 端使用 `*_THIRD`，并接受 `*_K` 别名 |
| `DCBranch` | 两端 P/V/I |
| `ACSwitch`, `ACZeroBranch` | 两端 P/Q/V/I |
| `DCSwitch`, `DCZeroBranch` | 两端 P/V/I |
| `ACGenerator`, `ACLoad` | P/Q/V/I |
| `DCGenerator`, `DCLoad` | P/V/I |
| `DCDCConverter` | 两端 P/V/I |
| `DCACConverter` | `P_DC`, `V_DC`, `I_DC`, `P_AC`, `Q_AC`, `V_AC`, `I_AC` |
| `ACACConverter` | 两端 P/Q/V/I |

## DCZeroBranch 处理

纯 DC 大规模算例中，量测文件通常包含 DCSwitch 量测，但不包含 DCZeroBranch 量测。为避免不可观测虚拟电流状态：

- DCSwitch 总是保留电流状态。
- DCZeroBranch 只有在存在有效 `P_FROM/P_TO/I_FROM/I_TO` 量测时才创建电流状态。
- 对没有电流状态的 DCZeroBranch，电压量测仍取节点电压；P/I 量测按零值和零导数处理。

该策略使 `hybrid_se` 在 `dc_net_3000` 等纯 DC 文件上保持满秩可观测。

## 量测函数

`evaluate(x)` 在 active measurement 场景下优先走 `_evaluate_active_measurements_fast()`：

- 只展开映射状态，不重建完整零阻抗 `phi`，降低开销。
- AC 网络量测通过数组化的电压、相角、支路 stamp 计算。
- DC 网络量测通过电压数组和显式功率/电流状态计算。
- 发电机量测通过节点功率平衡和本地负荷/零阻抗注入推断。
- 变流器量测直接读取紧凑功率状态和端口电压。

非 active measurement 或测试场景仍可走通用 `evaluate()` 路径。

## Jacobian

`jacobian_sparse(x)` 调用 `_assemble_jacobian()`，按以下层次构造：

1. 预索引静态 Jacobian 行，如节点电压、固定电压量测。
2. 批量追加 DCBranch、DCZero、DCDC、DCAC、ACAC 等动态导数。
3. 批量追加 ACBranch、ACTransformer、ACThreeWindingTransformer P/Q/I 导数。
4. 批量追加 ACGenerator 注入导数。
5. 对少量未覆盖场景走通用逐量测解析路径。

`_build_static_jacobian_index()`、`_build_fast_evaluation_index()` 和各类 `_append_fast_*` 函数共同避免每轮 WLS 大量字符串分派和 Python 循环。

## WLS 求解

`estimate()` 使用统一 WLS 流程：

```text
r = z - h(x)
H = dh/dx
G = H.T W H
rhs = H.T W r
G dx = rhs
x <- x + step * dx
```

线搜索会检查目标函数是否下降，并保证电压状态不低于 `voltage_floor`。若解向量出现非有限数，则提前停止。

## 可观测性和坏数据

- `observability_analysis()` 支持复用最终 `H`、正规矩阵和分解信息。
- `identify_bad_data()` 计算归一化残差。
- `estimate_with_bad_data_removal()` 支持迭代剔除最大坏数据。

大型系统中，坏数据辨识的杠杆计算可能成为主要耗时，应与 WLS 核心用时分开评估。

## 性能设计

`hybrid_se` 的关键性能措施：

- 紧凑状态布局，合并零阻抗等值节点，避免冗余状态。
- `full_col_for_state` 和 `full_to_state_col` 数组缓存状态映射。
- active measurement 的 `z` 和 `weight` 预先数组化。
- 快速量测评估不写回模型对象。
- Jacobian 静态行和动态行分开缓存。
- AC 支路 stamp、Y 行拓扑、负荷数组、发电机 share、变流器位置等全部缓存。
- 大多数 Jacobian 块直接稀疏生成。

## 适用建议

| 场景 | 建议 |
| --- | --- |
| 纯 AC 小算例 | `ac_se` 或 `hybrid_se` 均可 |
| 纯 DC 小算例 | `dc_se` 或 `hybrid_se` 均可 |
| 交直流混联 | 使用 `hybrid_se` |
| 大量零阻抗互联的 AC 拼接系统 | `hybrid_se` 的紧凑状态更稳 |
| 需要与单域程序逐项对比 | 使用 `ac_se` 或 `dc_se` 更直观 |

## 注意事项

- `hybrid_se` 初始化会先运行联合潮流，所以初始化时间通常高于单域 SE。
- 对纯 AC/纯 DC 文件，未出现的子系统会自动跳过。
- 变流器控制方程必须与量测和拓扑共同提供足够可观测性。
