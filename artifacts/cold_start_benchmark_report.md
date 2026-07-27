# LF cold-start benchmark report

## 测试口径

- `result_mode='array'`
- 平启动(flat start)
- 单进程(single process)
- 随机调用顺序(randomized call order)
- **不考虑进程间缓存和热启动**
- 端到端(end-to-end, 从文件读取到求解完成)
- 每个算例运行 5 次
- 统计: 最小值 / 最大值 / 平均值

## 随机调用顺序(5轮)

1. `ieee3w -> ieee300 -> dc_net_3000 -> hybrid_net_4w -> qinling_1000 -> ieee1w -> ieee3k -> hybrid_net_4k -> dc_net_1000 -> dc_net_3w -> qinling_100`
2. `hybrid_net_4k -> qinling_1000 -> ieee3w -> ieee300 -> hybrid_net_4w -> qinling_100 -> ieee1w -> dc_net_3w -> dc_net_3000 -> ieee3k -> dc_net_1000`
3. `ieee3w -> qinling_1000 -> ieee300 -> dc_net_3w -> qinling_100 -> dc_net_1000 -> ieee1w -> hybrid_net_4k -> hybrid_net_4w -> ieee3k -> dc_net_3000`
4. `dc_net_3w -> ieee3w -> dc_net_1000 -> hybrid_net_4w -> qinling_100 -> qinling_1000 -> ieee1w -> ieee300 -> hybrid_net_4k -> dc_net_3000 -> ieee3k`
5. `hybrid_net_4w -> dc_net_3000 -> qinling_100 -> ieee3k -> dc_net_1000 -> hybrid_net_4k -> ieee300 -> dc_net_3w -> qinling_1000 -> ieee1w -> ieee3w`

## 结果汇总

| case | kind | min (s) | max (s) | avg (s) | iter | normF | converged |
|------|------|--------:|--------:|--------:|-----:|------:|:---------:|
| ieee300 | ac | 0.0052 | 0.0085 | 0.0068 | 6 | 1.211e-12 | Y |
| ieee3k | ac | 0.0256 | 0.0565 | 0.0373 | 6 | 2.071e-12 | Y |
| ieee1w | ac | 0.0702 | 0.1484 | 0.1027 | 6 | 1.635e-12 | Y |
| ieee3w | ac | 0.2140 | 0.4423 | 0.2953 | 6 | 2.067e-12 | Y |
| dc_net_1000 | dc | 0.0053 | 0.0144 | 0.0086 | 7 | 2.567e-13 | Y |
| dc_net_3000 | dc | 0.0113 | 0.0291 | 0.0187 | 7 | 2.567e-13 | Y |
| dc_net_3w | dc | 0.0992 | 0.3493 | 0.1612 | 7 | 2.567e-13 | Y |
| qinling_100 | hybrid | 0.0616 | 0.0885 | 0.0739 | 4 | 2.603e-09 | Y |
| qinling_1000 | hybrid | 0.6430 | 0.8653 | 0.7820 | 4 | 2.603e-09 | Y |
| hybrid_net_4k | hybrid | 0.0604 | 0.1262 | 0.0934 | 8 | 1.306e-13 | Y |
| hybrid_net_4w | hybrid | 0.6356 | 1.0904 | 0.7796 | 8 | 1.634e-13 | Y |

## 结果解读

### AC 算例

- 300 节点到 3 万节点,平均耗时从 **0.0068s** 增长到 **0.2953s**
- AC 路径在 `array + from_file_fast` 下非常轻,3 万节点仍保持在 **0.3s** 量级

### DC 算例

- 1k / 3k / 3w 节点平均分别约:
  - **0.0086s**
  - **0.0187s**
  - **0.1612s**
- DC 3 万节点冷启动端到端仍在 **0.2s 以内** 的平均水平

### Hybrid 算例

- `qinling_1000`: **avg 0.7820s**
- `hybrid_net_4w`: **avg 0.7796s**
- 两个大规模 hybrid 算例已经稳定在 **0.7~0.8s** 级

### 稳定性

- **11/11 算例全部收敛**
- 迭代次数固定
- `normF` 稳定,没有随机顺序导致的不收敛或精度漂移

## 当前稳定优化总性能总结

### 已稳定落地的优化

1. **array 模式路径稳定化**
   - `result_mode='array'` 跳过不必要的 Python 对象回填
   - 对 AC / DC / Hybrid 端到端都有效

2. **`from_file_fast()` 统一接口**
   - `ACPowerFlowCalc.from_file_fast()`
   - `DCPowerFlowCalc.from_file_fast()`
   - `HybridPowerFlowCalc.from_file_fast()`
   - 统一了 file → PPC-backed solver 的快速入口

3. **Hybrid PPC-only / lightweight fast path**
   - 不再走完整 `HybridPowerNetwork` 重对象图路径
   - 是本轮最大的性能收益来源

4. **DC array 模式跳过 `_write_ppc_result_to_network()`**
   - 避免 array 模式下无意义的对象写回

5. **AC/DC reusable factorizer 接入**
   - 将 Hybrid 里已有的可复用 factorizer 思路扩展到 AC/DC
   - 对 `linear_solver='umfpack'` 的 AC/DC 有稳定小幅收益

6. **默认 solver 收敛到 `pyklu`**
   - AC/DC/Hybrid 默认 solver 已统一为 `pyklu`
   - 在当前 cold-start、array、单进程口径下表现稳定

### 大算例口径总结

#### 之前的 full 模式（对象路径）
- `qinling_1000`: **6.95s**
- `hybrid_net_4w`: **5.29s**

#### 当前 cold-start / array / fast path
- `qinling_1000`: **avg 0.7820s**
- `hybrid_net_4w`: **avg 0.7796s**

### 对比结论

- `qinling_1000`: 从 **6.95s → 0.78s**
  - 整体约 **8.9x 提升**
- `hybrid_net_4w`: 从 **5.29s → 0.78s**
  - 整体约 **6.8x 提升**

这已经不是“小幅优化”，而是一次完整的性能层级迁移：
- 从**重对象网络 + full 回填路径**
- 切到了**PPC-backed + array fast path**

### 当前剩余瓶颈判断

在最新版本下,继续抠小热点（如 `_read_efile_rows`、`_raw_vbase_maps`）收益已经很有限,甚至可能适得其反。

如果后续还要继续提速,更可能有价值的方向是：

1. **solver numeric factorization / reuse**
2. **topology prepare 的进一步结构优化**
3. **真正的多进程并行**（系统级提升）

## 结论

当前版本在您指定的严格口径下（cold-start、无热启动、随机顺序、单进程、端到端）已经达到一个非常不错的性能状态：

- AC 3 万节点: **0.30s avg**
- DC 3 万节点: **0.16s avg**
- Hybrid 4w / qinling_1000: **0.78s avg**

并且 11/11 算例全部稳定收敛。
