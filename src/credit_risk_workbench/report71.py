"""Offline, escaped reading report for the exact bound 71-feature snapshot."""
import math
from datetime import datetime,timedelta,timezone
from .contracts import require
from .candidate71_contract import digest
from .reports import esc,number
from .presentation_zh import feature,fact_value as chinese_value
from .reading71 import render as reading

VERSION='history_risk_reading_report_v1'
USE={'train':'训练样本','validation':'验证样本','unlabeled_scoring':'无标签申请','new_input_prediction_only':'新录入资料'}
POP={'labeled':'历史申请','unlabeled':'无标签申请','new_application71':'新录入申请'}
STATE={'observed':'已有观测','no_eligible_history':'无符合条件历史','source_null_or_no_eligible_observation':'未提供或无可用记录'}

def render(snapshot):
    c=snapshot['result'];facts=c['facts'];e=c['explanation'];p=c['probability'];threshold=c['threshold']
    require(c['feature_count']==71 and len(facts)==71 and len({f['name'] for f in facts})==71,'VERSION_MISMATCH')
    require(digest({k:v for k,v in c.items() if k!='case_revision'})==c['case_revision'],'VERSION_MISMATCH')
    require(all(type(v) in [int,float] and math.isfinite(v) for v in [p,threshold,e['base_log_odds'],e['contribution_sum'],e['raw_score'],*[f['contribution_log_odds'] for f in facts]]) and 0<=p<=1 and 0<=threshold<=1 and type(c['risk_flag']) is bool and c['risk_flag']==(p>=threshold),'VERSION_MISMATCH')
    total=sum(f['contribution_log_odds'] for f in facts);raw=e['base_log_odds']+total;prob=1/(1+math.exp(-raw)) if raw>=0 else math.exp(raw)/(1+math.exp(raw))
    require(max(abs(total-e['contribution_sum']),abs(raw-e['raw_score']),abs(prob-p))<=1e-10,'VERSION_MISMATCH')
    ident=c['case_key']['application_id'];require(type(ident) is int and ident>0,'VERSION_MISMATCH')
    created=datetime.fromisoformat(snapshot['created_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    top=sorted(facts,key=lambda f:abs(f['contribution_log_odds']),reverse=True)[:7]
    def value(v):return '—（来源为空或无观测）' if v is None else number(v)
    def contribution(f):
        v=f['contribution_log_odds'];return f'<span class="{"up" if v>=0 else "down"}">{v:+.2f}</span>'
    top_rows=''.join(f'<tr><th>{esc(feature(f["name"])["label"])}</th><td>{esc(chinese_value(f["name"],f["value"],f["missing_state"]))}</td><td>{contribution(f)}</td></tr>' for f in top)
    fact_rows=''.join(f'<tr data-feature="{esc(f["name"])}"><th>{esc(feature(f["name"])["label"])}</th><td>{esc(chinese_value(f["name"],f["value"],f["missing_state"]))}</td><td>{esc(feature(f["name"])["unit"])}</td><td>{contribution(f)}</td><td>{esc(STATE.get(f["missing_state"],f["missing_state"]))}</td></tr>' for f in facts)
    source_rows=''.join(f'<tr data-source="{esc(f["name"])}"><th>{esc(feature(f["name"])["label"])}</th><td>{esc(feature(f["name"])["source"])}<small>{esc(feature(f["name"])["formula"])}</small></td><td>{esc(feature(f["name"])["time"])}</td></tr>' for f in facts)
    quality={k:sum(f['missing_state']==k for f in facts) for k in STATE}
    metadata='<div><dt>评分模型</dt><dd>申请分析模型（树模型，71项输入）</dd></div>'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>风险分析报告 · 申请{ident}</title><style>
@page{{size:A4;margin:14mm}}*{{box-sizing:border-box}}body{{margin:0;background:#eef3f9;color:#172b46;font:14px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1080px;margin:28px auto;background:#fff;padding:36px 42px;border-top:5px solid #2563eb}}h1{{font-size:30px;margin:8px 0}}h2{{font-size:19px;margin:30px 0 12px;padding-bottom:8px;border-bottom:1px solid #dce5f1}}p{{margin:9px 0}}.brand{{color:#2563eb;font-weight:700}}.muted,small{{color:#60738b;font-size:12px}}small{{display:block;margin-top:4px}}.meta{{display:flex;gap:10px 28px;flex-wrap:wrap}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:24px 0}}.metrics>div{{padding:18px;background:#f3f7fd;border-radius:8px}}.metrics span{{font-size:12px;color:#60738b}}.metrics strong{{display:block;font-size:30px;margin-top:8px}}.notice{{padding:12px 16px;border-left:3px solid #2563eb;background:#f5f8fd}}table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px}}th,td{{text-align:left;vertical-align:top;padding:10px 9px;border-bottom:1px solid #e0e8f2;overflow-wrap:anywhere;word-break:break-word}}thead th{{background:#f1f5fb}}th{{font-weight:600}}.facts th:first-child{{width:30%}}.sources th:first-child{{width:23%}}.sources th:nth-child(2){{width:43%}}.up{{color:#a34b18}}.down{{color:#087f78}}.technical{{margin-top:26px;font-size:12px}}summary{{cursor:pointer;color:#315f9b}}dl>div{{padding:9px 0;border-bottom:1px solid #e5ebf3}}dt{{font-weight:600}}dd{{margin:3px 0 0;overflow-wrap:anywhere}}footer{{margin-top:28px;border-top:1px solid #dce5f1;padding-top:14px}}p,small{{overflow-wrap:anywhere}}@media(max-width:650px){{main{{margin:0;padding:20px 14px}}h1{{font-size:24px}}.metrics{{grid-template-columns:1fr}}th,td{{padding:8px 5px;font-size:11px}}.metrics strong{{font-size:26px}}}}@media print{{body{{background:white;font-size:11px}}main{{max-width:none;margin:0;padding:0;border-top:0}}h1{{font-size:24px}}h2{{break-after:avoid}}.metrics{{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}.metrics strong{{font-size:22px}}thead{{display:table-header-group}}tr,.metrics,.notice{{break-inside:avoid}}th,td{{padding:6px;font-size:10px}}.screen-only{{display:none}}}}
</style></head><body><main data-model-sha="{c['model_sha256']}" data-report-version="{VERSION}" data-application-id="{ident}">
<div class="brand">信贷研究 · 风险分析工作台</div><h1>申请 {ident} · 风险分析报告</h1>
<div class="meta muted"><span>申请来源：{esc(POP.get(c['case_key']['population'],c['case_key']['population']))}</span><span>样本用途：{esc(USE.get(c['sample_use'],c['sample_use']))}</span><span>生成时间：{esc(created)}（北京时间）</span></div>
<p class="screen-only muted">这是可离线阅读的报告。浏览器中按 Ctrl+P 可打印，或在打印目标中选择“另存为PDF”。</p>
<section class="metrics"><div><span>目标事件预测概率</span><strong>{p*100:.2f}%</strong></div><div><span>分析阈值</span><strong>{threshold*100:.2f}%</strong></div><div><span>阈值分组</span><strong>{'阈值以上' if c['risk_flag'] else '阈值以下'}</strong></div></section>
<p class="notice">申请分析结合本次申请的基本资料与借贷历史，共71项输入，使用同一树模型。预测概率是模型估计，阈值用于分析分组，不构成批准或拒绝贷款的结论。</p>
{reading(c['model_sha256'],threshold)}
<h2>主要影响因素</h2><p class="muted">按贡献绝对值列出前7项。正值推高模型内部得分，负值降低；贡献单位为对数几率，不能当作概率百分点相加，也不表示因果关系。</p><table><thead><tr><th>特征</th><th>特征值</th><th>模型贡献</th></tr></thead><tbody>{top_rows}</tbody></table>
<h2>资料概况</h2><p>有观测特征 {quality['observed']} 项；无符合条件历史的资料 {quality['no_eligible_history']} 项；未提供或无可用记录的资料 {quality['source_null_or_no_eligible_observation']} 项。</p><p class="muted">来源缺失与有效0分开保留。居住类型不说明产权；无符合条件历史不等于没有借贷或信用良好；金额沿用来源单位，不能直接解释为人民币或月薪。</p>
<h2>完整特征与模型贡献（71项）</h2><table class="facts"><thead><tr><th>特征</th><th>特征值</th><th>单位</th><th>模型贡献</th><th>资料状态</th></tr></thead><tbody>{fact_rows}</tbody></table>
<h2>来源与计算口径</h2><p class="muted">以下为各特征的来源与聚合规则，不等同于全部原始月度或支付明细。</p><table class="sources"><thead><tr><th>特征</th><th>来源与公式</th><th>时间口径</th></tr></thead><tbody>{source_rows}</tbody></table>
<details class="technical"><summary>得分计算与版本信息</summary><p>参考值 {e['base_log_odds']:.2f} ＋ 特征贡献合计 {total:.2f} ＝ 内部得分 {raw:.2f}，经概率转换得到 {p*100:.2f}%。展示数字可能有四舍五入差异。</p><dl>{metadata}</dl></details>
<footer class="muted">基于公开匿名数据的模型分析。训练样本不证明独立效果，验证数据参与过模型比较，无标签和新录入资料只提供预测。报告保留本次结果和口径，不替代实际授信审核。</footer>
</main></body></html>'''
