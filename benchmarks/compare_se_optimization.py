"""Validate paired numerical outputs and write an inspectable HTML/JSON report."""
import argparse
import html
import json
import statistics
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('prefix', type=Path, help='Prefix of the paired .json and .npz files')
    parser.add_argument('--regression-before', type=Path)
    parser.add_argument('--regression-after', type=Path)
    parser.add_argument('--targeted', type=Path)
    args = parser.parse_args()
    payload = json.loads(args.prefix.with_suffix('.json').read_text(encoding='utf-8'))
    if not payload.get('paired'):
        raise ValueError('Use benchmark_se_optimization.py --compare-ref to collect paired samples')
    baseline = payload['revision']
    summaries = []
    with np.load(args.prefix.with_suffix('.npz')) as arrays:
        for case in dict.fromkeys(row['case'] for row in payload['runs']):
            before = sorted((r for r in payload['runs'] if r['case'] == case and r['revision'] == baseline), key=lambda r: r['repeat'])
            after = sorted((r for r in payload['runs'] if r['case'] == case and r['revision'] == 'working-tree'), key=lambda r: r['repeat'])
            assert before and len(before) == len(after), case
            row = dict(case=case, samples=len(before), states=before[0]['states'], measurements=before[0]['measurements'],
                       iterations=before[0]['iterations'], max_abs_x_diff=0.0, max_abs_residual_diff=0.0)
            for b, a in zip(before, after):
                assert b['repeat'] == a['repeat'] and b['converged'] and a['converged'], case
                for key in ('iterations', 'states', 'measurements', 'objective', 'residual_inf'):
                    assert b[key] == a[key], (case, key, b[key], a[key])
                for field in ('x', 'residual'):
                    old = arrays[f'{case}_{baseline}_{b["repeat"]}_{field}']
                    new = arrays[f'{case}_working-tree_{a["repeat"]}_{field}']
                    assert np.isfinite(old).all() and np.isfinite(new).all(), case
                    np.testing.assert_allclose(new, old, rtol=1e-11, atol=1e-12)
                    row[f'max_abs_{field}_diff'] = max(row[f'max_abs_{field}_diff'], float(np.max(np.abs(new-old), initial=0)))
            for stage in ('prepare', 'run', 'total'):
                old = [r[stage + '_s'] for r in before]
                new = [r[stage + '_s'] for r in after]
                row[stage] = dict(before_s=statistics.median(old), after_s=statistics.median(new),
                                  before_range_s=[min(old), max(old)], after_range_s=[min(new), max(new)],
                                  speedup=statistics.median(old)/statistics.median(new),
                                  paired_speedup=statistics.median(b/a for b, a in zip(old, new)))
            summaries.append(row)
    verification = {}
    for name, path in (('before', args.regression_before), ('after', args.regression_after), ('targeted', args.targeted)):
        if path:
            lines = path.read_text(encoding='utf-8-sig').splitlines()
            verification[name] = dict(summary=lines[-1], failures=sorted(line.split(' - ')[0] for line in lines if line.startswith('FAILED ')))
    if 'before' in verification and 'after' in verification:
        verification['new_failures'] = sorted(set(verification['after']['failures']) - set(verification['before']['failures']))
        assert not verification['new_failures'], verification['new_failures']
    output = args.prefix.parent / 'comparison.json'
    output.write_text(json.dumps(dict(cases=summaries, verification=verification), indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    for r in summaries:
        t = r['total']
        rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in (
            r['case'], f'{r["states"]:,} / {r["measurements"]:,}',
            f'{r["prepare"]["before_s"]*1000:.1f} → {r["prepare"]["after_s"]*1000:.1f}',
            f'{r["run"]["before_s"]*1000:.1f} → {r["run"]["after_s"]*1000:.1f}',
            f'{t["before_s"]*1000:.1f} → {t["after_s"]*1000:.1f}',
            f'{t["speedup"]:.2f}×', f'{(1-t["after_s"]/t["before_s"])*100:.1f}%')) + '</tr>')
    timing_ranges = ''.join(f'<li>{r["case"]}：旧版 {r["total"]["before_range_s"][0]*1000:.1f}–{r["total"]["before_range_s"][1]*1000:.1f} ms；新版 {r["total"]["after_range_s"][0]*1000:.1f}–{r["total"]["after_range_s"][1]*1000:.1f} ms；配对加速比中位数 {r["total"]["paired_speedup"]:.2f}×。</li>' for r in summaries)
    checks = ''.join(f'<li>{html.escape(k)}：{html.escape(v["summary"])}</li>' for k,v in verification.items() if isinstance(v,dict))
    count_path = args.prefix.parent / 'operation_counts.json'
    counts_note = ''
    if count_path.exists():
        counts = json.loads(count_path.read_text(encoding='utf-8'))
        counts_note = '<p>独立调用计数（另行运行，不参与计时）：</p><ul>' + ''.join(
            f'<li>{html.escape(r["case"])} / {html.escape(r["revision"])}：兼容设备视图构造 {r["compatibility_views"]} 次，求解阶段 evaluate {r["solve_evaluate_calls"]} 次。</li>'
            for r in counts) + '</ul>'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SE 无网络缓存优化与性能对比</title><style>
body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#192b39;background:#f5f7fa;margin:0}}main{{max-width:1180px;margin:32px auto;padding:32px;background:white;border-radius:12px}}h1,h2{{line-height:1.3}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px 8px;border-bottom:1px solid #d8e1e8;text-align:left}}th{{background:#eaf0f6}}pre{{background:#102c40;color:#ecf4fa;padding:18px;overflow:auto;border-radius:8px}}.note{{background:#edf5f2;padding:14px 18px;border-left:4px solid #248269}}.table{{overflow:auto}}code{{font-size:.92em}}small{{color:#506477}}</style>
<main><h1>SE 无网络缓存优化与性能对比</h1><p>2026-11-16 · 基线提交 <code>{baseline}</code> · 三个 SE 模块与当前工作树对照</p>
<p class="note">每个样本均新建估计器、重新读取网络文件、重新构建拓扑；每次计时前清空既有测量文件缓存。未新增网络、PPC 或拓扑的跨实例缓存，未使用上一轮估计结果作为初值。</p>
<h2>实测结果</h2><p>每例 {summaries[0]['samples']} 轮，旧版和新版交替先后运行，表中为各阶段耗时中位数，单位 ms。加速比为旧版总耗时中位数 / 新版总耗时中位数；负的耗时降幅表示变慢。</p>
<div class="table"><table><thead><tr><th>算例</th><th>状态 / 量测</th><th>准备：前 → 后</th><th>求解：前 → 后</th><th>总耗时：前 → 后</th><th>加速比</th><th>耗时降幅</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>收益集中于 Hybrid。AC/DC 独立求解算法本轮未更改，实测小幅差异不作为算法加速结论。机器存在负载波动，以下给出完整范围，保留所有样本：</p><ul>{timing_ranges}</ul>
<h2>具体修改</h2><ul><li>Hybrid 准备阶段只在状态布局确定后构造一次兼容设备/节点对象，移除前两次重复构造。</li><li>设备字段按列转换为 Python 标量，减少逐设备、逐字段数组索引与类型转换。</li><li>WLS 接受线搜索步后，直接使用该步已经计算的 h(x)、残差和目标函数；回退步重新计算。修复达到迭代上限时回退步与返回残差不同步的问题。</li><li>补齐 Hybrid 的 prepare/finalize_prepare 参数和阶段语义；AC/DC 增加 evaluate(measurements=...) 公共关键字，并保留旧 measurement_plan_tables 关键字；Hybrid 委托路径传递 out 缓冲区。</li><li>AC 表输入保留设备数字索引；Hybrid 支持纯 AC/DC 的 PPC 输入，显式替换输入后重建相关布局。</li></ul>
{counts_note}<h2>正确性与回归</h2><p>所有配对运行均收敛，迭代次数、状态数、量测数、目标函数和无穷范数残差一致。逐元素比较每轮状态及残差数组：最大状态差 {max(r['max_abs_x_diff'] for r in summaries):.3g}，最大残差差 {max(r['max_abs_residual_diff'] for r in summaries):.3g}。</p><ul>{checks}</ul>
<p>广泛回归中的失败在基线逐项对照；新增失败数：{len(verification.get('new_failures', []))}。这不是全套测试通过的声明。专项测试覆盖延迟准备、输入替换、归一化输入、纯侧委托、多能流、输出模式、out 缓冲区及线搜索回退。</p>
<h2>统一调用</h2><pre>est = Estimator(e_file=case, meas_file=meas, flat_start=True,
                auto_prepare=False)
est.prepare()
result = est.run(result_mode="array", skip_bad_data=True,
                 final_diagnostics=False, verbose=False)
state = est.estimate_result.x
h = est.evaluate(state, measurements=est.active_measurements, out=buffer)</pre>
<p>Estimator 可替换为 ACStateEstimator、DCStateEstimator 或 HybridStateEstimator。prepare、finalize_prepare、estimate、jacobian_sparse、observability_analysis、run 的参数名称、默认值和参数类别经过一致性检查。构造器保留既有位置参数顺序；公共选项使用关键字。支持 full、array、summary、none 四种输出模式。输入量测表应为真实量测输入；运行时追加的约束和伪量测不作为新的文件量测再次输入。</p>
<h2>复现与边界</h2><pre>python benchmarks/benchmark_se_optimization.py --compare-ref {baseline} --repeats 5 --output output/se_optimization_20261116/paired
python benchmarks/compare_se_optimization.py output/se_optimization_20261116/paired --regression-before output/se_optimization_20261116/regression_before.txt --regression-after output/se_optimization_20261116/regression_after.txt --targeted output/se_optimization_20261116/targeted.txt</pre>
<p>采用 flat_start=True、max_iter=30、result_mode=array、skip_bad_data=True、final_diagnostics=False。准备耗时含构造、网络与量测读取及拓扑/计划构建；求解耗时为 run 调用。导入、版本切换、垃圾回收和结果文件写入在计时外。操作系统文件缓存未人为清空。此表不代表包含坏数据剔除、最终诊断或完整对象结果导出的总耗时。</p>
<p>基线仅替换三个 SE 模块，其余依赖及工作树已有 LF 修改在两组中相同。原始数据：同目录 paired.json、paired.npz；汇总：comparison.json。修改保留在本地，未自动提交或推送。</p>
<small>{html.escape(payload['python'])}<br>{html.escape(payload['platform'])} · NumPy {payload['numpy']} · SciPy {payload['scipy']}</small></main></html>'''
    report_path = args.prefix.parent / 'report.html'
    report_path.write_text(report, encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(report_path)


if __name__ == '__main__':
    main()
