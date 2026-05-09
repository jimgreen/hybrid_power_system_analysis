# 潮流计算与状态估计技术文档索引

本文档目录整理当前工程中 6 个核心计算模块：

| 模块 | 文档 | 源码 |
| --- | --- | --- |
| 交流潮流 | [ac_lf.md](ac_lf.md) | `src/hybrid_power_system_analysis/lfcore/ac_lf.py` |
| 交流状态估计 | [ac_se.md](ac_se.md) | `src/hybrid_power_system_analysis/secore/ac_se.py` |
| 直流潮流 | [dc_lf.md](dc_lf.md) | `src/hybrid_power_system_analysis/lfcore/dc_lf.py` |
| 直流状态估计 | [dc_se.md](dc_se.md) | `src/hybrid_power_system_analysis/secore/dc_se.py` |
| 交直流联合潮流 | [hybrid_lf.md](hybrid_lf.md) | `src/hybrid_power_system_analysis/lfcore/hybrid_lf.py` |
| 交直流联合状态估计 | [hybrid_se.md](hybrid_se.md) | `src/hybrid_power_system_analysis/secore/hybrid_se.py` |
| 潮流设备模型汇总 | [load_flow_device_models.md](load_flow_device_models.md) | `src/hybrid_power_system_analysis/lfcore/*.py` |

## 公共约定

### E 文件单位

E 文件采用有名值输入，并通过模型中的基准配置转换为内部标幺值：

| 字段 | 含义 |
| --- | --- |
| `p_base` | 系统功率基准，配合 `p_scale` 解释输入有功/无功单位 |
| `u_scale` | 电压单位缩放，`1.0` 表示 kV，`1000.0` 表示 V |
| `p_scale` | 功率单位缩放，`1.0` 表示 kW/kVar，`1000.0` 表示 W/Var，`0.001` 表示 MW/MVar |
| `i_scale` | 电流单位缩放，`1.0` 表示 kA，`1000.0` 表示 A |

内部计算统一使用标幺值；状态估计量测文件中的有功、无功、电压、电流、相角会在读取后转换为内部单位。相角文件值为度，内部为弧度。

### 参数文件

潮流参数来自 `lf.para`：

| 参数 | 含义 |
| --- | --- |
| `tol` | Newton 残差无穷范数收敛阈值 |
| `max_iter` | 最大 Newton 迭代次数 |
| `min_voltage` | 电流计算和数值保护使用的最小电压 |
| `divergence_threshold` | 残差超过该值时判定直流潮流发散 |

状态估计参数来自 `se.para`：

| 参数 | 含义 |
| --- | --- |
| `tol` | WLS 状态修正量最大值收敛阈值 |
| `max_iter` | 最大 WLS 迭代次数 |
| `diff_step` | 数值差分兼容参数，当前核心 Jacobian 已解析化 |
| `flat_start` | 默认是否平启动 |
| `bad_threshold` | 坏数据归一化残差阈值 |
| `max_remove` | 坏数据迭代剔除最大数量 |
| `pseudo_measurement_weight` | 自动伪量测权重 |
| `voltage_floor` | WLS 线搜索中电压下限 |
| `min_current_voltage` | 电流量测计算中除法保护阈值 |
| `power_flow_tol` | 状态估计内部潮流种子计算容差 |
| `power_flow_max_iter` | 状态估计内部潮流最大迭代次数 |
| `power_flow_min_voltage` | 状态估计内部潮流最小电压 |

### 公共数值实现

状态估计公共稀疏矩阵工具位于 `src/hybrid_power_system_analysis/secore/se_math.py`：

- `SparseJacobianBuilder`：以 COO triplet 方式批量构造稀疏 Jacobian。
- `build_normal_equations()`：生成正规方程 `G = H.T W H` 和右端 `H.T W r`。
- `solve_normal_equations_with_factor()`：优先使用 Cholesky，可退化到稀疏直接解。
- `observability_rank_details()`：可观测性秩分析，优先复用正规矩阵或分解信息。
- `identify_bad_data()` 在各 SE 模块内调用 `inverse_gain_for_bad_data()` 和 `measurement_leverage()` 计算归一化残差。
