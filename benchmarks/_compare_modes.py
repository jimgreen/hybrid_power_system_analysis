"""对比 full vs array 模式端到端 LF 性能."""

import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load(p):
    text = (ART / p).read_text(encoding="utf-8")
    # Strip trailing comment lines starting with '#'
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return json.loads("\n".join(lines))


def fmt_s(x):
    return f"{x:.4f}"


def main():
    full = load("e2e_full.json")
    arr = load("e2e_array.json")

    print("# result_mode 对比: full vs array (单进程 / 平启动)\n")
    print("基线(full):结果回填到 PPC + Python 对象 + 写回网络")
    print("对比(array):仅返回 numpy 数组结果, 跳过网络写回与对象构建")
    print("计时:7 次测量,2 次预热;`min` 取最快一次,`med` 取中位数")
    print()

    print("## 总览表\n")
    print("| case | kind | nodes | full min | array min | Δ min | full med | array med | Δ med | iter | normF |")
    print("|------|------|-------|---------|----------|-------|----------|----------|-------|------|-------|")
    keys = list(full["cases"].keys())
    deltas_min, deltas_med = [], []
    for c in keys:
        f = full["cases"][c]
        a = arr["cases"][c]
        d_min = (a["min_s"] - f["min_s"]) / f["min_s"] * 100
        d_med = (a["median_s"] - f["median_s"]) / f["median_s"] * 100
        deltas_min.append(d_min)
        deltas_med.append(d_med)
        norm_match = "✅" if abs(f["normF"] - a["normF"]) < 1e-9 else "❌"
        print(f"| {c} | {f['kind']} | {f['nodes']} | "
              f"{fmt_s(f['min_s'])} | {fmt_s(a['min_s'])} | {d_min:+.1f}% | "
              f"{fmt_s(f['median_s'])} | {fmt_s(a['median_s'])} | {d_med:+.1f}% | "
              f"{a['iter']} | {a['normF']:.2e} {norm_match} |")

    print()
    print("## 数值一致性\n")
    print("| case | full normF | array normF | 一致 |")
    print("|------|-----------|------------|------|")
    for c in keys:
        f = full["cases"][c]
        a = arr["cases"][c]
        m = "✅" if abs(f["normF"] - a["normF"]) < 1e-9 else "❌"
        print(f"| {c} | {f['normF']:.3e} | {a['normF']:.3e} | {m} |")

    print()
    print("## 总结\n")
    n_accel = sum(1 for d in deltas_min if d < -5)
    n_slow = sum(1 for d in deltas_min if d > 5)
    print(f"- **min 加速**算例(< -5%): {n_accel}/11")
    print(f"- **min 退化**算例(> +5%): {n_slow}/11")
    print(f"- **min 平均**: {sum(deltas_min)/len(deltas_min):+.1f}%")
    print(f"- **med 平均**: {sum(deltas_med)/len(deltas_med):+.1f}%")
    print(f"- **AC 最大加速**:ieee3w(30k 节点) min -43% / med -39%")
    print(f"- **AC ieee1w(10k 节点) min -34% / med -34%**")
    print(f"- **DC 30k 节点(dc_net_3w)**:min -2% / med +4% (基本无差异,Newton 求解主导)")
    print(f"- **Hybrid 80k 节点(qinling_1000) min -4% / med -1%**")
    print()
    print("## 结论\n")
    print("**`result_mode='array'` 对 AC 算例有显著加速效果(尤其大规模),对 DC/Hybrid 收益甚微。**")
    print()
    print("**AC 加速来源**:")
    print("- `full` 模式:Newton 收敛后要把 `V/theta` 写回 `ppc[\"bus\"][:, 7/8]` 数组,")
    print("  并触发 `_write_ppc_result_to_network` 把数据写回 Python 对象网络 (Network/Bus/Branch/...)")
    print("- `array` 模式:**完全跳过** `network.writeback` 路径(`ACPowerFlowCalc._network_writeback=None`)")
    print("- AC 30k 节点时 Python 对象写回的开销远大于 Newton 迭代本身,故收益显著")
    print()
    print("**DC 收益接近 0**:`dc_net_3w` 30k 节点,full 与 array 几乎一样(4.28s vs 4.28s)。")
    print("原因是 DCPowerFlowCalc 求解后 `_write_back_ppc` 在 array 模式下被跳过,但**DC 本来就主要是数值计算**,")
    print("Newton 1 次迭代 7 个 SC SuperLU 求解就是 4s,Python 对象回填占比小。")
    print()
    print("**Hybrid 收益也小**:`qinling_1000` 80k 节点,加速仅 -4% min / -1% med。")
    print("原因是 `HybridPowerNetwork.read_from_file` + `prepare` 占了端到端 80%+ 时间,")
    print("Newton 迭代本身 0.6-0.9s,写回 0.2-0.3s,占比小。")
    print()
    print("**使用建议**:")
    print("- AC LF(尤其 ≥ 3k 节点):用 `array` 模式,加速 **-30% ~ -43%**")
    print("- DC LF / Hybrid LF:用 `array` 与 `full` 模式差异不大,按需选用")


if __name__ == "__main__":
    main()
