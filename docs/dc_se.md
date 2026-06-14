# dc_se 技术文档

## 模块定位

`src/hybrid_power_system_analysis/secore/dc_se.py` 实现直流电网加权最小二乘状态估计。核心类是 `DCStateEstimator`。

该模块负责：

- 读取 DC E 文件和 `.meas` 文件。
- 状态估计前运行 DC 潮流，得到物理一致种子。
- 支持平启动。
- 自动过滤无效量测和离线设备量测。
- 自动为无量测电源/负荷/变流器添加低权重伪量测。
- 解析构造量测函数和稀疏 Jacobian。
- 进行可观测性分析、WLS 状态估计、坏数据辨识。

## 入口与使用方式

```python
from secore.dc_se import DCStateEstimator

estimator = DCStateEstimator(
    e_file="data/model/dc/dc_net_3000.e",
    meas_file="data/meas/dc/dc_net_3000.meas",
    flat_start=True,
)
result = estimator.estimate(verbose=False)
```

命令行：

```powershell
python -m secore.dc_se --case data\model\dc\dc_net_3000.e --meas data\meas\dc\dc_net_3000.meas --flat-start --quiet
```

## 支持的量测

| 设备 | 量测类型 |
| --- | --- |
| `DCNode` | `V` |
| `DCBranch` | `P_FROM`, `V_FROM`, `I_FROM`, `P_TO`, `V_TO`, `I_TO` |
| `DCSwitch` | `P_FROM`, `V_FROM`, `I_FROM`, `P_TO`, `V_TO`, `I_TO` |
| `DCGenerator` | `P_GEN`, `V_GEN`, `I_GEN` |
| `DCLoad` | `P_LOAD`, `V_LOAD`, `I_LOAD` |
| `DCDCConverter` | `P_FROM`, `V_FROM`, `I_FROM`, `P_TO`, `V_TO`, `I_TO` |

`DCZeroBranch` 可被拓扑和模型识别，但纯 DC 状态估计默认不为未量测零阻抗支路建立电流状态。开关电流会作为状态，因为量测文件通常包含 DCSwitch P/V/I 量测。

## 状态变量

`DCStateEstimator` 状态向量：

```text
x = [V_all, I_switch, P_DCDC_FROM, P_DCDC_TO, P_VGEN]
```

| 变量 | 含义 |
| --- | --- |
| `V_all` | 所有存活 DC 节点电压 |
| `I_switch` | 闭合 DCSwitch 的显式电流状态 |
| `P_DCDC_FROM/P_DCDC_TO` | DC/DC 两端功率状态 |
| `P_VGEN` | 定电压 DCGenerator 的平衡有功 |

平启动时节点电压取 `1.0`，开关电流和变流器功率按初始策略填充；非平启动时使用内部 DC 潮流结果作为种子。

## 量测函数

`evaluate(x)` 中主要设备模型：

### DCBranch

```text
I = (V_i - V_j) / r
P_FROM = V_i * I
P_TO = -V_j * I
```

### DCSwitch

开关电流为显式状态：

```text
P_FROM = V_i * I_sw
P_TO = -V_j * I_sw
```

### DCLoad

ZIP 模型：

```text
P = pv0 + pv1 * V + pv2 * V^2
I = P / V
```

### DCGenerator

| 控制 | 估计量 |
| --- | --- |
| `V` | 功率来自 `P_VGEN` 状态，电压取节点状态 |
| `P` | 功率取设定值，电流为 `P/V` |
| `I` | 电流取设定值，功率为 `I*V` |

### DCDCConverter

两端功率为状态，电压取端口节点状态，电流为 `P/V`。

## Jacobian

`jacobian_sparse(x)` 直接生成解析稀疏 Jacobian：

- `DCBranch` P/I 对两端电压导数解析化。
- `DCSwitch` P/I 对电压和开关电流导数解析化。
- `DCLoad` P/I 对电压导数解析化。
- `DCGenerator` 按控制模式生成导数。
- `DCDCConverter` 电流量测对功率和电压导数解析化。

`_measurement_plan()` 会提前把同类量测分组，`_fill_measurement_values_vectorized()` 和 `_fill_jacobian_vectorized()` 使用数组化路径减少逐量测 Python 循环。

## WLS 求解

求解流程与 AC SE 一致：

1. 计算 `r = z - h(x)`。
2. 构造 `H`。
3. 构造正规方程。
4. 解状态修正量。
5. 线搜索和电压下限保护。
6. 判断 `max(abs(dx)) < tol`。

`estimate()` 返回 `EstimateResult`。`apply_state()` 可将估计结果回填到 DC 模型对象。

## 可观测性和坏数据

- `observability_analysis()` 判断 `rank == state_count`。
- `identify_bad_data()` 计算归一化残差。
- `estimate_with_bad_data_removal()` 支持迭代剔除最大坏数据。

对于大型 DC 系统，坏数据杠杆项计算可能比 WLS 迭代本身更耗时，应在性能测试中单独统计。

## 性能设计

- DC 潮流种子由 `DCPowerFlowCalc` 提供。
- 量测计划缓存，避免每轮重新解析字符串和设备。
- 量测值和 Jacobian 批量填充。
- Jacobian 直接以稀疏矩阵生成。
- 正规方程构造和求解复用 `src/hybrid_power_system_analysis/secore/se_math.py`。

## 注意事项

- `.meas` 文件不再包含独立 `PowerBase`，单位跟随 E 文件。
- `valid=0` 的量测会被跳过。
- 未量测设备的伪量测权重较低，只用于提高可观测性，不应替代真实量测质量建模。
