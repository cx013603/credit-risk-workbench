"""Immutable local HTML/JSON snapshots, composed from the same case facts."""
from contextlib import closing
import hashlib,html,json,re,uuid
from pathlib import Path
from .contracts import require
from .catalog import quality,filters
from .reviews import now

STATUS_ZH={'unreviewed':'未复核','pending':'待复核','needs_information':'资料待补','reviewed':'已复核'}
USE_ZH={'train':'训练样本（不作独立效果证明）','validation':'验证样本','unlabeled_scoring':'无标签评分（无真实命中率）'}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def esc(v):return html.escape(str(v),quote=True)
def number(v):return f'{v:,.0f}' if isinstance(v,(int,float)) and v==int(v) else f'{v:,.2f}' if isinstance(v,(int,float)) else 'null（缺失）' if v is None else str(v)
def selection_scope(v,c):
    require(isinstance(v,dict) and set(v) in [{'kind'},{'kind','filters','collection_revision','total'}])
    if v['kind']=='single':require(set(v)=={'kind'});return {'kind':'single','case_keys':[c['case_key']]}
    require(v['kind']=='collection' and set(v)=={'kind','filters','collection_revision','total'})
    f=v['filters'];require(isinstance(f,dict) and len(f)==8 and all(isinstance(x,(str,int,float)) and type(x) is not bool for x in f.values()))
    clean=filters({k:str(x) for k,x in f.items()});require(clean['population']==c['case_key']['population']);require(type(v['total']) is int and v['total']>=0 and isinstance(v['collection_revision'],str) and re.fullmatch('[a-f0-9]{64}',v['collection_revision']))
    return {'kind':'collection','filters':clean,'collection_revision':v['collection_revision'],'total':v['total'],'case_keys':[c['case_key']],'scope_notice':'仅报告当前一笔；筛选条件是打开时的集合快照，不宣称当前集合数量或效果。'}

def render_html(d):
    from .presentation_zh import feature,value as chinese_value,prose
    c=d['case'];r=d['review'];q=d['quality'];k=c['case_key'];m=c['model_outputs'];n=d['narratives'];e=c['explanation'];rows={x['fact_id']:x for x in n['rows']} if n else {}
    top=sorted(e['terms'],key=lambda t:abs(t['contribution_log_odds']),reverse=True)[:5];maximum=max(.01,*[abs(t['contribution_log_odds']) for t in top]);topids={t['fact_id'] for t in top};rest=sum(t['contribution_log_odds'] for t in e['terms'] if t['fact_id'] not in topids)
    facts={f['fact_id']:f for f in c['profile']['facts']};bars=[]
    for t in top:
        value=t['contribution_log_odds'];label=feature(t['fact_id'])['label'];bars.append(f'<tr><th>{esc(label)}</th><td><div class="track"><i style="width:{abs(value)/maximum*100:.5f}%;background:{"#b45332" if value>=0 else "#087f78"}"></i></div></td><td>{value:+.2f}</td></tr>')
    factrows=[]
    state={'observed':'已观测','source_null':'输入缺失','no_eligible_history':'无合格历史'}
    for f in c['profile']['facts']:
        nr=rows.get(f['fact_id']);evidence=nr['evidence_text'] if nr else '未连接来源说明，仅保存输入；不能称已核对原始记录。';unit=nr['unit_zh'] if nr else f['unit']
        factrows.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [feature(f['fact_id'])['label'],chinese_value(f['fact_id'],f['raw_value']),state[f['missing_state']],feature(f['fact_id'])['unit'],feature(f['fact_id'])['source'],prose(evidence)])+'</tr>')
    versionrows=''.join(f'<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>' for key,value in c['versions'].items())
    if d['selection']['kind']=='collection':scope=f"打开时筛选快照：{d['selection']['total']}笔；本报告只包含申请{k['application_id']}。筛选条件以打开时的记录为准。"
    else:scope=f"单笔范围：申请{k['application_id']}；不代表总体效果。"
    source_notice='历史索引已连接；逐项来源证据见下表。' if d['history']['status']=='ready' else '历史索引未连接；报告未核验历史来源记录。'
    reference_notice=f"固定训练参考{d['reference']['rows']:,}笔；分桶定位不是精确百分位。" if d['reference']['status']=='ready' else '训练参考未连接；不提供分布定位。'
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模拟分析报告 · {k['application_id']}</title><style>
@page{{size:A4;margin:13mm}}*{{box-sizing:border-box}}body{{font:13px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#172b46;margin:0;background:#f4f7fc}}main{{max-width:1000px;margin:24px auto;padding:28px;background:white}}h1{{font-size:26px}}h2{{font-size:18px;border-bottom:1px solid #dbe5f3;padding-bottom:8px;margin-top:26px}}p,td,th,pre{{overflow-wrap:anywhere;word-break:break-word}}.banner{{padding:12px;background:#edf3fe}}.muted{{font-size:11px;color:#566e8b}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.metrics strong{{display:block;font-size:24px}}table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #e2e8f2;padding:8px}}th{{font-weight:600}}.track{{height:11px;background:#eef2f8}}.track i{{display:block;height:11px}}.note{{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;padding:12px;background:#f5f7fb;border:1px solid #e3eaf4}}.evidence{{font-size:10px}}.evidence td:nth-child(6){{width:30%}}.versions{{font-size:9px}}.versions th{{width:31%}}@media print{{body{{background:white}}main{{max-width:none;margin:0;padding:0}}.screen-only{{display:none}}h2{{break-after:avoid}}tr,.metrics,.banner{{break-inside:avoid}}thead{{display:table-header-group}}.note{{background:white;orphans:3;widows:3}}a{{color:inherit;text-decoration:none}}}}@media(max-width:600px){{main{{margin:0;padding:16px}}.metrics{{grid-template-columns:1fr}}.evidence{{font-size:8px}}}}
</style></head><body><main><p class="screen-only banner">可用浏览器菜单“打印”保存PDF；也可下载此HTML离线查看。此页是生成时的固定快照。</p><h1>模拟分析报告 · 申请 {k['application_id']}</h1><p>来源：{esc('带标签历史回放' if k['population']=='labeled' else '无标签申请')} · 用途：{esc(USE_ZH[c['sample_use']])}</p><p class="muted">报告 {esc(d['report_id'])} · 生成于 {esc(d['created_at'])}（UTC）</p><div class="banner">个人本机技能学习／模拟分析记录。预测不是真实银行审批结论或固定期限PD；备注不能改变模型事实。</div>
<h2>选取范围</h2><p>{esc(scope)}</p><h2>本笔保存结果</h2><div class="metrics"><div>树模型概率<strong>{m['tree']['probability']*100:.2f}%</strong>分析阈值 {m['tree']['threshold']*100:.2f}%</div><div>资料完整度<strong>{q['completeness']*100:.2f}%</strong>{q['observed']}／40项有观测值</div><div>模拟复核状态<strong>{esc(STATUS_ZH[r['review_status']])}</strong>收藏：{'是' if r['favorite'] else '否'}</div></div><p>逻辑回归 {m['logistic']['probability']*100:.2f}% · 评分卡 {m['scorecard']['probability']*100:.2f}%／{m['scorecard']['score']:.2f}分。评分卡分数不解释树概率。</p>
<h2>资料不足与来源范围</h2><p>输入缺失 {q['source_null']}项；无合格历史字段 {q['no_eligible_history']}项；征信 {q['bureau_count']}条、历史申请 {q['previous_count']}条。零值与缺失原因分别保留；资料完整度不是预测置信度。</p><p>{esc(source_notice)} {esc(reference_notice)} 16项申请字段仅保存模型输入，原主表行未独立接入。金额单位／周期、自然人、绝对申请日期及标签期限的未知继续保留。</p>
<h2>模拟备注</h2><p class="muted">已保存于 {esc(r['updated_at'] or '尚无保存记录')} · 标记版本 {r['marks_revision']}</p><div class="note">{esc(r['note']) if r['note'] else '（空备注）'}</div>
<h2>主要模型贡献</h2><p>橙色＋为上推，绿色−为拉低；所有数值单位为对数几率。图条按绝对值绘制，符号与数值另列，不是概率百分点或因果。</p><table><thead><tr><th>因素</th><th>贡献幅度</th><th>原路径贡献</th></tr></thead><tbody>{''.join(bars)}<tr><th>其余35项合计</th><td>完整40项补足</td><td>{rest:+.2f}</td></tr><tr><th>训练路径参考值／全部贡献／内部得分</th><td colspan="2">{e['base_log_odds']:.2f} ＋ {e['contribution_sum']:.2f} ＝ {e['raw_score']:.2f}；按完整精度还原原概率，显示合计可能有舍入差异。</td></tr></tbody></table>
<h2>40项保存输入与来源证据</h2><table class="evidence"><thead><tr><th>字段</th><th>保存值</th><th>资料状态</th><th>单位</th><th>来源</th><th>实际证据范围</th></tr></thead><tbody>{''.join(factrows)}</tbody></table>
<h2>版本与适用限制</h2><p>769笔早期测试暴露、40列覆盖有限与许可未核验等限制保留。无标签不提供实际命中率；训练案例不作独立效果证明。模拟复核“已复核”只表示本机分析记录。</p></main></body></html>'''

class ReportStore:
    def __init__(self,directory,reviews):self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True);self.reviews=reviews
    def create(self,c,selection,expected,narratives,reference,history,fields):
        from .contracts import integer
        integer(expected,0);scope=selection_scope(selection,c);rid=uuid.uuid4().hex;paths=[]
        with self.reviews.marks.lock,closing(self.reviews.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');review=self.reviews.read(c,db);require(review['marks_revision']==expected and review['binding_matches'],'VERSION_MISMATCH');require(not review['deleted'],'REVIEW_DELETED')
            values={f['fact_id']:f['raw_value'] for f in c['profile']['facts']};q=quality(values,fields);q.pop('missing_bits')
            d={'schema_version':'workbench_report_v1','report_id':rid,'created_at':now(),'scope':'local_simulated_analysis_only','case':c,'review':review,'selection':scope,'quality':q,'narratives':narratives,'reference':reference or {'status':'not_connected'},'history':history or {'status':'not_connected'}}
            raw=json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2).encode();htmlraw=render_html(d).encode('utf-8');require(max(len(raw),len(htmlraw))<=2*1024*1024,'REPORT_QUOTA')
            require(db.execute('SELECT COUNT(*) FROM reports').fetchone()[0]<1000 and sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())+len(raw)+len(htmlraw)<=100*1024*1024,'REPORT_QUOTA')
            try:
                for suffix,data in [('.json',raw),('.html',htmlraw)]:
                    p=self.directory/(rid+suffix)
                    with p.open('xb') as file:paths.append(p);file.write(data)
                db.execute('INSERT INTO reports VALUES(?,?,?,?,?,?,?,?)',(rid,c['versions']['release_manifest_sha256'],c['case_key']['population'],c['case_key']['application_id'],d['created_at'],expected,digest(raw),digest(htmlraw)))
                db.commit()
            except BaseException:
                for p in paths:p.unlink(missing_ok=True)
                raise
        return {'report_id':rid,'case_key':c['case_key'],'case_revision':c['case_revision'],'marks_revision':expected,'created_at':d['created_at'],'json_sha256':digest(raw),'html_sha256':digest(htmlraw),'view_url':f'/api/v2/reports/{rid}/view','download_url':f'/api/v2/reports/{rid}/download','data_url':f'/api/v2/reports/{rid}/data'}
    def file(self,rid,kind):
        require(isinstance(rid,str) and re.fullmatch('[a-f0-9]{32}',rid) and kind in ['html','json'])
        with closing(self.reviews.connect()) as db:row=db.execute('SELECT * FROM reports WHERE report_id=?',(rid,)).fetchone()
        require(row is not None,'REPORT_NOT_FOUND');p=self.directory/(rid+'.'+kind);require(p.is_file() and digest(p.read_bytes())==row[kind+'_sha256'],'REPORT_CORRUPT');return p
    def listing(self,seal,limit=20):
        from .contracts import integer
        integer(limit,1,50)
        with closing(self.reviews.connect()) as db:rows=db.execute('SELECT report_id,population,id application_id,created_at,marks_revision FROM reports WHERE release=? ORDER BY created_at DESC LIMIT ?',(seal,limit)).fetchall()
        return {'items':[dict(r) for r in rows],'limit':limit,'release_manifest_sha256':seal}
