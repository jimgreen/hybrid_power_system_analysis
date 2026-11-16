"""Evidence report for bounded columnar SE measurement preparation."""
import hashlib
import html
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/se_optimization_20261116/round5'


def suite(name):
    cases=list(ET.parse(OUT/name).iter('testcase'))
    failures={c.attrib['classname']+'::'+c.attrib['name'] for c in cases
              if c.find('failure') is not None or c.find('error') is not None}
    return dict(total=len(cases),passed=len(cases)-len(failures),failed=len(failures),failures=sorted(failures))


def cells(values):
    return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'


def main():
    timing=json.loads((OUT/'paired/comparison.json').read_text(encoding='utf8'))
    parsing=json.loads((OUT/'parser_comparison.json').read_text(encoding='utf8'))
    before,after,targeted=[suite(p) for p in ('regression_before.xml','regression_after.xml','targeted.xml')]
    new=sorted(set(after['failures'])-set(before['failures']))
    assert not new and not targeted['failed']
    assert all(c['exact_equal'] for c in timing['cases'])
    for name in ('ac_se','dc_se','hybrid_se'):
        assert timing['source_sha256']['baseline'][name] == timing['source_sha256']['optimized'][name]
    for name,digest in timing['source_sha256']['optimized'].items():
        path=ROOT/'src/hybrid_power_system_analysis'/('model' if name=='meas_array_model' else 'secore')/(name+'.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
    verification=dict(before=before,after=after,targeted=targeted,new_failures=new)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    totals=[]
    ranges=[]
    for c in timing['cases']:
        t=c['total']
        totals.append(cells([c['kind']+' / '+c['case'],f"{c['states']:,} / {c['measurements']:,}",
            *[f"{c[p]['before_s']*1000:.2f} → {c[p]['after_s']*1000:.2f}" for p in ('prepare','run','total')],
            f"{t['speedup']:.3f}×",f"{(1-t['after_s']/t['before_s'])*100:.2f}%"]))
        ranges.append(cells([c['case'],
            f"{t['before_range_s'][0]*1000:.1f}–{t['before_range_s'][1]*1000:.1f}",
            f"{t['after_range_s'][0]*1000:.1f}–{t['after_range_s'][1]*1000:.1f}",
            f"{t['paired_speedup']:.3f}×"]))
    parser_rows=[]
    for c in parsing:
        b=statistics.median(c['timings']['baseline']);a=statistics.median(c['timings']['optimized'])
        mem=c['memory']
        parser_rows.append(cells([c['case'],f"{mem['baseline']['rows']:,}",f"{b*1000:.2f} → {a*1000:.2f}",
            f"{b/a:.3f}×",f"{(1-a/b)*100:.2f}%",
            f"{mem['baseline']['peak_working_set_mb']:.2f} → {mem['optimized']['peak_working_set_mb']:.2f}"]))
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SE 第五轮：分块列式准备与性能对比</title><style>
body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f2f5f8;color:#182d3e;margin:0}}main{{max-width:1200px;margin:28px auto;padding:32px;background:white;border-radius:12px}}
h1{{font-size:28px}}h2{{font-size:21px;margin-top:28px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #d9e2ea;text-align:left;white-space:nowrap}}th{{background:#e9f0f6}}.scroll{{overflow:auto}}.note{{padding:14px 18px;background:#eaf6ef;border-left:4px solid #287f58}}.limit{{padding:14px 18px;background:#fff4df}}pre{{background:#153348;color:white;padding:18px;overflow:auto}}small{{color:#587080}}a{{color:#0969a8}}</style></head>
<body><main><h1>SE 第五轮：分块列式准备与性能对比</h1><p>2026-11-16 · 基线为第四轮完成后的工作树快照</p>
<p class="note">仅修改三类 SE 共用的量测数值解析路径。每个基准样本重新读取文件、重新建立网络与拓扑；清空既有量测缓存和稀疏数学模块全局缓存。未新增网络缓存、跨实例拓扑复用、初值复用或解缓存。</p>
<h2>改动与适用范围</h2><p>原实现逐行调用 Python int/float，再逐元素写入 NumPy 数组。现改为每 16,384 行暂存数值字段，由 NumPy 批量转换并直接填入预分配列数组。容量不足时扩容，最后返回有效区间；避免全文件临时列表和全量分块拼接。设备编码、量测顺序、状态码、精度、返回字段与 dtype 保持一致。</p>
<p>ACStateEstimator、DCStateEstimator、HybridStateEstimator 的源码与公共调用接口本轮均未改变。共享路径用于 include_strings=False 的量测准备；包含字符串的通用解析路径保持原有行为。</p>
<h2>端到端实测</h2><p>每例 {timing['repeats']} 轮，基线/优化版交替先后运行。单位 ms，表中为各阶段中位数；耗时降幅为负表示变慢。</p>
<div class="scroll"><table><thead><tr><th>算例</th><th>状态 / 量测</th><th>准备：前 → 后</th><th>求解：前 → 后</th><th>总耗时：前 → 后</th><th>加速比</th><th>耗时降幅</th></tr></thead><tbody>{''.join(totals)}</tbody></table></div>
<p class="limit">本轮存在明显机器负载波动。下列范围保留全部样本；总耗时小幅变化不能单独证明加速，求解核心未改变，其计时差异也不应归因于本轮算法。hybrid_net_400 的总耗时中位数比与配对加速比方向相反，因此不认定它已稳定提速。纯解析阶段的独立结果用于辨别共享准备路径的收益，不将它冒充端到端提升。</p>
<div class="scroll"><table><thead><tr><th>算例</th><th>基线总耗时范围 ms</th><th>优化总耗时范围 ms</th><th>配对加速比中位数</th></tr></thead><tbody>{''.join(ranges)}</tbody></table></div>
<h2>独立解析与内存</h2><p>每例 5 轮直接调用文件解析器，完全绕开量测文件缓存；所有数值数组精确一致。峰值工作集来自独立新进程，包含 Python/NumPy 运行时，是单次观察值，单位 MiB；它不代表完整 SE 的峰值内存。</p>
<div class="scroll"><table><thead><tr><th>量测文件</th><th>原始量测行数</th><th>解析 ms：前 → 后</th><th>加速比</th><th>耗时降幅</th><th>进程峰值 MiB：前 → 后</th></tr></thead><tbody>{''.join(parser_rows)}</tbody></table></div>
<h2>正确性与接口</h2><p>{len(timing['rows'])} 次端到端计时均收敛，所有配对的迭代次数、状态数、量测数、目标函数和残差范数相同，状态及残差数组逐元素精确相等。原始量测行数与有效量测行数不同，是筛选、约束和伪量测追加所致。</p>
<p>专项测试 {targeted['passed']} 项通过，覆盖跨批次、容量低估、空表、非有限值、符号零、极小/极大浮点数、状态码、非法数值报错，以及三类 SE 的统一签名、输入替换、延迟准备、输出模式、Jacobian 刷新和网络独立读取。</p>
<p class="limit">相关回归：优化前 {before['passed']} 通过 / {before['failed']} 失败；优化后 {after['passed']} 通过 / {after['failed']} 失败；新增失败 0。已有失败仍然存在，不能宣称全套通过。</p>
<h2>复现</h2><pre>python benchmarks/benchmark_se_round5.py --repeats 7
python benchmarks/benchmark_meas_round5.py
python -m pytest tests/test_se_round5_preparation.py tests/test_meas_array_model.py tests/test_se_optimization_contract.py tests/test_state_metadata.py -q -p no:cacheprovider</pre>
<p>flat_start=True、max_iter=30、result_mode=array、skip_bad_data=True、final_diagnostics=False；BLAS 线程为 1。导入、源码切换、垃圾回收和结果保存均在计时外；系统文件缓存不清空。对比时仅切换共享解析模块，三个 SE 模块在两个版本中保持相同源码，其余模型/LF 依赖也一致。不能用本轮绝对耗时直接除以上一轮不同负载时的数值，作为累计加速比。</p>
<p>证据：<a href="paired/comparison.json">端到端 JSON（含输入/源码 SHA256）</a> · <a href="paired/comparison.npz">状态与残差数组</a> · <a href="parser_comparison.json">解析与内存数据</a> · <a href="verification.json">回归对照</a> · <a href="regression_after.txt">回归日志</a>。</p>
<p>后续按原顺序继续定位网络列式准备及可观性候选搜索。本轮未更改数值求解器，固定稀疏结构与符号复用仍使用既有实现。</p>
<small>{html.escape(timing['python'])}<br>{html.escape(timing['platform'])} · NumPy {timing['numpy']} · SciPy {timing['scipy']}</small></main></body></html>'''
    (OUT/'report.html').write_text(report,encoding='utf8')
    print(OUT/'report.html')
    print(json.dumps(dict(targeted=targeted['passed'],new_failures=new,remaining_failures=after['failed'])))


if __name__=='__main__':
    main()
