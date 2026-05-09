# AC/Hybrid 潮流与 MATPOWER 对比说明

本文档说明 `ac_lf`、`hybrid_lf` 与 MATPOWER/PYPOWER 在 `ieee300`、`ieee3k`、`ieee3w` 算例上的对比口径、结果解释和已知建模差异。

## 1. 对比对象

| 对象 | 实现 | 说明 |
| --- | --- | --- |
| `ac_lf` | `src/hybrid_power_system_analysis/lfcore/ac_lf.py` 中的 `ACPowerFlowCalc` | 纯交流 Newton-Raphson 潮流 |
| `hybrid_lf` | `src/hybrid_power_system_analysis/lfcore/hybrid_lf.py` 中的 `run_hybrid_power_flow` | 统一交直流 Newton 潮流；纯 AC 算例中只包含 AC 子系统 |
| MATPOWER 参照 | `PYPOWER runpf` | 使用 MATPOWER 兼容的 branch/tap 模型 |

本文记录的是当前 T 型变压器建模调整后的基准结果。若后续需要把该对比固化为持续基准，应将 E 文件到 MATPOWER/PYPOWER ppc 的投影逻辑放入 `benchmarks/`，并在脚本输出中显式标注变压器 `gt/bt` 到 `BR_B` 的近似关系。

## 2. 重要建模口径

### 2.1 普通线路

普通 `ACBranch` 使用 MATPOWER 风格线路充电模型：

```text
y = 1 / (r + j*x)
y_sh = j*b/2
Yff = y + y_sh
Yft = -y
Ytf = -y
Ytt = y + y_sh
```

`b` 是整条线路的总对地电纳，在两端各分一半。

### 2.2 变压器

当前 `ACTransformer` 使用 T 型单端对地模型：

```text
y = 1 / (r + j*x)
yt = gt + j*bt
tapc = tap * exp(j*shift)
Yff = (y + yt) / (tapc * conj(tapc))
Yft = -y / conj(tapc)
Ytf = -y / tapc
Ytt = y
```

其中 `gt/bt` 是 i 侧单端对地导纳，不按两端平分。它不是 MATPOWER `BR_B` 的语义。

### 2.3 MATPOWER 投影误差

MATPOWER 标准 branch/tap 模型不能直接表达“变压器 i 侧单端对地 `gt/bt`”。为了做工程对比，脚本把变压器 `bt` 投影为 MATPOWER 的对称充电：

```text
BR_B = 2 * bt
```

这个投影只保持总充电量相近，不保持端口分布一致。因此 `ac_lf/hybrid_lf` 相对 MATPOWER 的端口无功、发电机无功和电压会出现系统性差异。该差异不代表 Newton 求解未收敛。

## 3. 模型规模

| 算例 | E 节点 | MATPOWER 节点 | 线路 | 变压器 | 发电机 | 负荷 | `ac_lf` 变量 | `hybrid_lf` 变量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee300` | 300 | 300 | 283 | 128 | 69 | 201 | 530 | 530 |
| `ieee3k` | 3000 | 2982 | 2830 | 1280 | 690 | 2010 | 5320 | 5320 |
| `ieee3w` | 30000 | 29802 | 28300 | 12800 | 6900 | 20100 | 53200 | 53200 |

`ieee3k` 和 `ieee3w` 在 MATPOWER 参照中节点数少于 E 文件节点数，是因为转换时合并了零阻抗支路、闭合开关和闭合断路器两端的理想等电位节点，避免在 MATPOWER Ybus 中写入零阻抗支路导致奇异。

## 4. 收敛与效率

下表为预热后重复 5 次的中位数。计时包括求解器准备和求解，不包含 Python 进程启动时间。`ac_lf` 和 `hybrid_lf` 使用 `linear_solver="pyklu"`；MATPOWER 参照为 PYPOWER `runpf`。

| 算例 | MATPOWER ms | `ac_lf` ms | `hybrid_lf` ms | `ac_lf` / MATPOWER | `hybrid_lf` / MATPOWER | `ac_lf` 迭代 | `hybrid_lf` 迭代 | `ac_lf` 残差 | `hybrid_lf` 残差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee300` | 13.5505 | 6.2873 | 17.2138 | 2.16x faster | 0.79x | 6 | 6 | 1.211e-12 | 1.211e-12 |
| `ieee3k` | 69.1570 | 50.8262 | 123.0459 | 1.36x faster | 0.56x | 6 | 6 | 1.238e-12 | 1.238e-12 |
| `ieee3w` | 726.0462 | 568.3710 | 1549.1388 | 1.28x faster | 0.47x | 6 | 6 | 2.049e-12 | 2.049e-12 |

解释：

- `ac_lf` 和 `hybrid_lf` 都在 6 次 Newton 迭代收敛，最终残差达到 `1e-12` 量级。
- 纯 AC 算例中，`hybrid_lf` 的结果与 `ac_lf` 相同；当前实现已在单 AC Newton 块时复用 AC 子求解器 residual/Jacobian，但完整 hybrid 结果对象仍有额外回填和包装成本。
- `ieee300` 中 `ac_lf` 快于 PYPOWER；`ieee3k` 中 PYPOWER 更快，说明当前实现仍有大规模稀疏组装和线性求解优化空间。

`hybrid_lf` 新增 `result_mode="array"` 后，可跳过 hybrid 对象门面回填，仅保留数组结果。纯 AC 大算例的中位数如下：

| 算例 | `hybrid_lf full` ms | `hybrid_lf array` ms | array/full |
| --- | ---: | ---: | ---: |
| `ieee300` | 16.1591 | 14.8579 | 0.92 |
| `ieee3k` | 134.6634 | 122.2889 | 0.91 |
| `ieee3w` | 1709.6250 | 1499.1047 | 0.88 |

## 5. 相对 MATPOWER 的最大误差

误差单位：电压幅值为 pu，相角为 deg，功率为 pu。

| 算例 | 方法 | Vm | Va | 端口 P | 端口 Q | Pg | Qg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee300` | `ac_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-03 | 6.023e-01 |
| `ieee300` | `hybrid_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-03 | 6.023e-01 |
| `ieee3k` | `ac_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-02 | 6.023e-01 |
| `ieee3k` | `hybrid_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-02 | 6.023e-01 |
| `ieee3w` | `ac_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-01 | 6.023e-01 |
| `ieee3w` | `hybrid_lf` | 6.049e-03 | 1.206e-01 | 1.214e-02 | 6.382e-01 | 5.311e-01 | 6.023e-01 |

最大无功误差主要来自变压器单端对地 `bt` 与 MATPOWER 对称 `BR_B` 的差异。对比 MATPOWER 时不能只看端口 Q 最大误差来判断本地求解器精度。

## 6. `ac_lf` 与 `hybrid_lf` 同模型对比

| 算例 | Vm | Va | 线路 P | 线路 Q | 变压器 P | 变压器 Q | Pg | Qg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee300` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ieee3k` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ieee3w` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

这个结果说明，在纯 AC E 文件上：

- `hybrid_lf` 的 AC 子系统与 `ac_lf` 使用同一套数组模型和同一套导纳 stamp。
- 两者的潮流结果逐项一致。
- 差异主要体现在执行路径和运行效率，而不是物理结果。

## 7. 潮流结果汇总

| 算例 | 总发电 P | 总发电 Q | 总负荷 P | 总负荷 Q | 总有功损耗 | 总无功损耗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee300` | 239.359076 | 81.614292 | 235.258500 | 77.879700 | 4.088469 | -2.276333 |
| `ieee3k` | 2393.590760 | 816.142918 | 2352.585000 | 778.797000 | 40.884693 | -22.763325 |
| `ieee3w` | 23935.907600 | 8161.429180 | 23525.850000 | 7787.970000 | 408.846927 | -227.633253 |

`ieee3k` 和 `ieee3w` 是 `ieee300` 的规模扩展，因此发电、负荷和损耗随规模近似线性放大。

## 8. 后续严格对标建议

如果目标是验证本地 Newton 数值实现，应使用同一物理导纳模型：

- 用当前 T 型单端 `gt/bt` 模型对比 `ac_lf` 与 `hybrid_lf`。
- 或在 MATPOWER 侧额外增加等效 bus shunt，精确模拟变压器 i 侧单端 `gt/bt`。

如果目标是对标 MATPOWER 原始算例结果，应使用 MATPOWER 的原生模型：

- 把本地变压器临时退化为 MATPOWER branch/tap 模型。
- 将变压器充电作为对称 `BR_B` 处理。
- 明确该口径不再是当前工程的 T 型单端变压器模型。
