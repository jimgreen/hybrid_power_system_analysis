"""Generate the round-4 report from measured timings and actual pytest XML."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def suite(path):
    root = ET.parse(path)
    tests = list(root.iter('testcase'))
    failures = {e.attrib['classname'] + '::' + e.attrib['name']
                for e in tests if e.find('failure') is not None or e.find('error') is not None}
    skipped = sum(e.find('skipped') is not None for e in tests)
    return dict(total=len(tests), passed=len(tests)-len(failures)-skipped,
                failed=len(failures), skipped=skipped, failures=sorted(failures))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'output/se_optimization_20261116/round4'
    data = json.loads((out / 'paired/comparison.json').read_text(encoding='utf8'))
    for name, digest in data['source_sha256']['optimized'].items():
        path = root / f'src/hybrid_power_system_analysis/secore/{name}.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    before, after, targeted = [suite(out / p) for p in
                              ('regression_before.xml', 'regression_after.xml', 'targeted.xml')]
    new_failures = sorted(set(after['failures']) - set(before['failures']))
    assert not new_failures
    assert targeted['failed'] == 0
    verification = dict(before=before, after=after, targeted=targeted, new_failures=new_failures,
                        resolved=sorted(set(before['failures'])-set(after['failures'])))
    (out / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf8')
    rows, ranges = [], []
    for case in data['cases']:
        t = case['total']
        values = [case['kind'] + ' / ' + case['case'],
                  f"{case['states']:,} / {case['measurements']:,}"]
        for phase in ('prepare', 'run', 'total'):
            v = case[phase]
            values.append(f"{v['before_s'] * 1000:.2f} → {v['after_s'] * 1000:.2f}")
        values += [f"{t['speedup']:.3f}×", f"{(1-t['after_s']/t['before_s'])*100:.2f}%"]
        rows.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>')
        ranges.append('<tr>' + ''.join(f'<td>{v}</td>' for v in (
            case['case'],
            f"{t['before_range_s'][0]*1000:.2f}–{t['before_range_s'][1]*1000:.2f}",
            f"{t['after_range_s'][0]*1000:.2f}–{t['after_range_s'][1]*1000:.2f}",
            f"{t['paired_speedup']:.3f}×")) + '</tr>')
    report = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SE 第四轮优化与性能对比</title><style>
body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f2f5f8;color:#182b3a}}
main{{max-width:1180px;margin:28px auto;background:white;padding:32px;border-radius:12px}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:30px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d9e1e8;text-align:left;white-space:nowrap}}th{{background:#eaf0f6}}
.scroll{{overflow:auto}}.note{{padding:14px 18px;background:#eaf5ef;border-left:4px solid #248260}}.limit{{background:#fff6e6;padding:14px 18px}}
pre{{background:#142f43;color:#eef5fa;padding:18px;overflow:auto}}code{{font-size:.92em}}a{{color:#0969a8}}small{{color:#566879}}
</style></head><body><main>
<h1>SE 第四轮优化与性能对比</h1><p>2026-11-16 · AC / DC / Hybrid · 基线为本轮优化前源码快照</p>
<p class="note">每个样本新建估计器、重新读取网络和量测、重新构建拓扑。量测文件缓存及稀疏数学模块全局缓存均在计时前清空。没有网络缓存、跨实例拓扑复用、上次解或结果缓存。</p>
<h2>实测效率</h2><p>每例 {data['repeats']} 轮，基线与优化版交替先后执行；下表为耗时中位数，单位 ms。耗时降幅为负表示实测变慢。</p>
<div class="scroll"><table><thead><tr><th>算例</th><th>状态 / 量测</th><th>准备：前 → 后</th><th>求解：前 → 后</th><th>合计：前 → 后</th><th>加速比</th><th>耗时降幅</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>本轮收益集中于 Hybrid 中小算例的状态布局。AC/DC 独立算法本轮没有实质变化，约 0.5–0.7% 的差异不视为加速或明显退化；最大算例 qinling_1000 原先已使用惰性布局，本轮总耗时仅下降约 1.8%，且样本区间重叠，应视为小幅收益。</p>
<h2>本轮具体修改</h2><ol>
<li>统一 evaluate 的参数顺序与类别：<code>(x, measurement_plan_tables=None, *, out=None, measurements=None)</code>；保留两种关键字，重复指定时报相同 TypeError。</li>
<li>Hybrid 所有非空布局使用惰性状态元数据视图，在建立视图时取得子估计器数组；中小算例保留原有详细状态标签，修复负索引越界行为。已有大算例的标签格式保持一致。</li>
<li>AC/DC 参考电压覆盖由每个参考节点扫描一次目标数组，改为排序后的批量查找，参考值优先级、重复节点、零参考值及缺失节点回退保持一致。</li>
<li>AC 外部量测计划（包括 Hybrid 的 AC 子块）启用已有固定 CSR 数值刷新路径；同一个计划复用稀疏结构，新量测计划建立新 builder，不缓存网络。</li>
</ol>
<h2>接口与数值验收</h2><p>所有 50 次计时运行收敛。每个配对的迭代次数、状态数、量测数、目标函数及残差无穷范数相同；状态和残差数组逐元素精确相同，最大差为 0。</p>
<p>专项测试 {targeted['passed']} 项通过：统一 prepare/finalize_prepare/evaluate/estimate/jacobian_sparse/observability_analysis/run 签名；两种 evaluate 关键字与 out 缓冲区；延迟准备、输入替换、归一化量测、纯侧委托、四种输出模式、流体路径、独立网络读取、固定 Jacobian 数值刷新与等长行重排。</p>
<p class="limit">相关大范围回归共 {after['total']} 项：接续时工作树为 {before['passed']} 通过、{before['failed']} 失败；最终为 {after['passed']} 通过、{after['failed']} 失败，新增失败 0，修复 1 项状态标签回归。不能据此宣称全套测试通过。测试对照基线是本次接续时的工作树；性能基线则是 round4/baseline_src 中更早的优化前快照，两者用途不同。</p>
<h2>样本波动</h2><div class="scroll"><table><thead><tr><th>算例</th><th>基线总耗时范围 ms</th><th>优化总耗时范围 ms</th><th>配对加速比中位数</th></tr></thead><tbody>{''.join(ranges)}</tbody></table></div>
<h2>复现与测量边界</h2><pre>python benchmarks/benchmark_se_round4.py --repeats 5
python -m pytest tests/test_se_optimization_contract.py tests/test_state_metadata.py -q -p no:cacheprovider</pre>
<p>flat_start=True、max_iter=30、result_mode=array、skip_bad_data=True、final_diagnostics=False；BLAS 线程固定为 1。准备包含构造、网络/量测读取、拓扑与计划构建；求解为 run 调用。导入、源码切换、垃圾回收和写报告在计时外。只切换三个 SE 模块，LF 与模型依赖在两组中完全相同；系统文件缓存未人为清空。本表不代表坏数据剔除、最终诊断或完整对象导出模式的总耗时。</p>
<p>下一步可继续处理列式网络准备、可观性候选搜索；稀疏求解器已有符号复用路径，后续应依据重新剖析的数据决定是否更改，避免重复实现。</p>
<p>原始与验证材料：<a href="paired/comparison.json">性能 JSON（含源码/输入 SHA256）</a> · <a href="paired/comparison.npz">状态/残差 NPZ</a> · <a href="verification.json">回归对照 JSON</a> · <a href="regression_after.txt">回归日志</a>。</p>
<small>{html.escape(data['python'])}<br>{html.escape(data['platform'])} · NumPy {data['numpy']} · SciPy {data['scipy']}</small>
</main></body></html>'''
    path = out / 'report.html'
    path.write_text(report, encoding='utf8')
    print(path)
    print(json.dumps(dict(targeted=targeted['passed'], before_failures=before['failed'],
                          after_failures=after['failed'], new_failures=new_failures)))


if __name__ == '__main__':
    main()
