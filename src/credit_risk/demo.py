"""E11 local, server-rendered existing-application page, sharing the E10 session.

Only HTML and static assets are served. No independent scoring API, model fits,
free-form feature collection, remote assets, labels, or new test evaluation.
"""
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit

import pyarrow.compute as pc

from .raw import DataError, sha256_file
from .scoring import load_scoring_bundle

VERSION = 'demo_v1'
LABELS = {
    'NAME_CONTRACT_TYPE':'申请合同类型','FLAG_OWN_CAR':'是否有车','FLAG_OWN_REALTY':'是否有房产',
    'AMT_INCOME_TOTAL':'申报收入（源数据金额单位）','NAME_INCOME_TYPE':'收入类型','NAME_HOUSING_TYPE':'住房情况',
    'OCCUPATION_TYPE':'职业','ORGANIZATION_TYPE':'工作单位类型','FLAG_MOBIL':'手机联系标记',
    'FLAG_EMP_PHONE':'工作电话标记','FLAG_WORK_PHONE':'其他工作电话标记','FLAG_CONT_MOBILE':'手机可联系标记',
    'FLAG_PHONE':'固定电话标记','FLAG_EMAIL':'邮箱标记','WEEKDAY_APPR_PROCESS_START':'申请星期',
    'HOUR_APPR_PROCESS_START':'申请小时',
    'BUREAU_ELIGIBLE_COUNT':'合格征信借款记录数','BUREAU_HAS_ELIGIBLE_HISTORY':'有合格征信历史',
    'BUREAU_DAYS_SINCE_MOST_RECENT':'最近征信借款距今（天）','BUREAU_C1_COUNT':'币种1记录数',
    'BUREAU_OTHER_CURRENCY_COUNT':'其他币种记录数','BUREAU_CURRENCY_MISSING_COUNT':'币种缺失记录数',
    'BUREAU_C1_AMOUNT_AVAILABLE_COUNT':'币种1金额可用记录数','BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT':'币种1金额不可用记录数',
    'BUREAU_C1_CREDIT_SUM':'币种1合格借款金额合计','BUREAU_C1_CREDIT_MEAN':'币种1合格借款金额均值',
    'BUREAU_C1_CREDIT_MAX':'币种1合格借款金额最大值','BUREAU_ACTIVE_COUNT':'Active状态记录数',
    'BUREAU_CLOSED_COUNT':'Closed状态记录数','BUREAU_SOLD_COUNT':'Sold状态记录数',
    'BUREAU_BAD_DEBT_COUNT':'Bad debt状态记录数','BUREAU_OTHER_STATUS_COUNT':'其他或缺失状态记录数',
    'BUREAU_ACTIVE_FRACTION':'Active状态比例','PREV_ELIGIBLE_COUNT':'合格历史申请记录数',
    'PREV_HAS_ELIGIBLE_HISTORY':'有合格历史申请','PREV_DAYS_SINCE_MOST_RECENT':'最近历史申请距今（天）',
    'PREV_CASH_COUNT':'历史Cash loans记录数','PREV_CONSUMER_COUNT':'历史Consumer loans记录数',
    'PREV_REVOLVING_COUNT':'历史Revolving loans记录数','PREV_OTHER_CONTRACT_COUNT':'历史其他或缺失合同记录数'
}
PURPOSES = {'train':'训练样本','validation':'验证样本','test':'申请级保留测试样本'}
ERRORS = {
    'demo_id':'请输入有效的申请编号，仅使用正整数。',
    'demo_query':'一次只查询一个申请编号，请使用页面表单。',
    'scoring_unknown_id':'未找到该申请。当前支持已有带标签申请回放；无标签 application_test 尚未接入。',
    'scoring_fields':'必需评分字段缺失或不符合冻结契约，无法评分。',
    'experiment_feature_type':'特征类型与冻结契约不符，无法评分。',
    'experiment_nonfinite':'特征包含非法数值，无法评分。',
    'scoring_history':'历史计数或存在标记不符合契约，无法评分。'
}


def load_demo_config(root, path):
    root = Path(root).resolve()
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (set(value) != {'version','scoring_report','host','port'} or value['version'] != VERSION
                or value['host'] != '127.0.0.1' or type(value['port']) is not int or not 0 <= value['port'] <= 65535):
            raise ValueError('local fixed config required')
        report = (root/value['scoring_report']).resolve()
        if not report.is_relative_to(root/'reports/runs'):
            raise ValueError('project E10 report required')
        return {**value,'report_path':report}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataError('demo_config','Invalid local demo config') from exc


def parse_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[0-9]{1,19}',value) or not 0 < int(value) <= 2**63-1:
        raise DataError('demo_id','Positive int64 decimal application ID required')
    return int(value)


def display(value):
    if value is None: return '缺失（null）'
    if isinstance(value,float): return format(value,'.12g')
    return str(value)


def safe(value):
    return escape(str(value),quote=True)


class DemoState:
    def __init__(self, root, report_path):
        self.root, self.report_path = Path(root).resolve(), Path(report_path).resolve()
        self.session = None
        self.error_code = None
        self.examples = []
        self.score_lock = threading.Lock()
        self.requests = 0

    def load(self):
        try:
            self.session = load_scoring_bundle(self.root,self.report_path)
            f = self.session.features
            def first(mask):
                rows=pc.indices_nonzero(mask)
                return f['SK_ID_CURR'][rows[0].as_py()].as_py() if len(rows) else None
            specs=[('验证示例',pc.equal(f['sample_use'],'validation')),
                   ('无征信历史',pc.and_(pc.equal(f['sample_use'],'validation'),pc.equal(f['BUREAU_ELIGIBLE_COUNT'],0))),
                   ('两类均无合格历史',pc.and_(pc.equal(f['BUREAU_ELIGIBLE_COUNT'],0),pc.equal(f['PREV_ELIGIBLE_COUNT'],0))),
                   ('保留测试示例',pc.equal(f['sample_use'],'test'))]
            self.examples=[(label,value) for label,mask in specs if (value:=first(mask)) is not None]
            if 100003 in self.session.index: self.examples.append(('训练加总案例',100003))
            self.report_sha256=sha256_file(self.report_path)
        except (Exception,) as exc:
            self.session=None
            self.error_code=exc.code if isinstance(exc,DataError) else type(exc).__name__
        return self

    def get_case(self, application_id):
        if self.session is None:
            raise DataError('demo_unavailable','Complete E10 success bundle unavailable')
        with self.score_lock:
            # The only page scoring operation: same fitted models and feature source.
            case=self.session.case(application_id)
            self.requests+=1
            return case


def shell(content):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light"><title>申请风险工作台 · 信贷学习项目</title>'
        '<link rel="stylesheet" href="/assets/demo.css"><script src="/assets/demo.js" defer></script></head>'
        '<body><a class="skip" href="#main">跳到主要内容</a><header class="topbar"><div class="brand">'
        '<span class="brand-icon" aria-hidden="true">C</span><span>申请风险工作台<small>CREDIT RISK · LEARNING LAB</small></span></div>'
        '<span class="local-tag">本机演示 · 已有申请回放</span></header><main id="main">'+content+
        '</main><footer>公开匿名信贷数据 · 本机分析与研究 · 数据使用条件见来源说明。</footer></body></html>')


def query_panel(state, current=''):
    links=''.join(f'<a class="example" href="/?id={value}#case">{safe(label)} <span>{value}</span></a>' for label,value in state.examples)
    return ('<section class="query-panel" aria-labelledby="query-title"><div><span class="eyebrow">01 / 选择申请</span>'
        '<h2 id="query-title">读取一笔真实申请</h2><p>按申请编号取得同一份40列特征，调用已冻结的评分流程。</p></div>'
        '<form action="/#case" method="get"><label for="application-id">已有申请编号</label><div class="query-fields">'
        f'<input id="application-id" name="id" type="text" inputmode="numeric" autocomplete="off" required value="{safe(current)}" '
        'placeholder="例如 100003" aria-describedby="id-help"><button type="submit">查看评分 <span aria-hidden="true">→</span></button></div>'
        '<small id="id-help">只接受已有申请的正整数编号。缺字段时不会补造资料。</small></form>'
        f'<nav class="examples" aria-label="真实示例申请">{links}</nav></section>')


def hero():
    return ('<section class="hero"><div><span class="eyebrow">从数据到结果 / E11</span>'
        '<h1>从一笔申请，<br>看懂风险评分。</h1><p>先看资料与历史，再看模型判断，最后核对每一项评分卡分值。</p></div>'
        '<ol class="flow" aria-label="评分流程"><li><b>01</b>真实资料与两类历史</li>'
        '<li><b>02</b>冻结40列与已拟合模型</li><li><b>03</b>风险概率与可复查分值</li></ol></section>')


def field_table(features, names, contract):
    meta={f['name']:f for f in contract['features']}
    rows=''.join('<tr><th scope="row">'+safe(LABELS.get(name,name))+f'<code>{safe(name)}</code></th>'
        f'<td>{safe(display(features[name]))}</td><td>{safe(meta[name]["unit"])}</td></tr>' for name in names)
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="全部特征取值，可横向滚动"><table><thead><tr><th scope="col">字段</th><th scope="col">实际取值</th><th scope="col">单位</th></tr></thead><tbody>'+rows+'</tbody></table></div>'


def render_case(state, case):
    r, f = case['result'],case['features'];contract=state.session.contract
    probability=r['risk_probability'];threshold=r['threshold'];flag=r['risk_flag']
    purpose=PURPOSES[r['sample_use']]
    sample_notice=('这笔申请参与过模型训练，仅用于流程演示，不能作为独立效果证据。' if r['sample_use']=='train' else
        '验证样本用于过方案比较与阈值选择，不能当作未接触的独立测试。' if r['sample_use']=='validation' else
        '申请级保留测试；存在已披露的早期环境试跑暴露。当前只回放既有模型，不重新评估或选择方案。')
    flag_text='达到课堂标记阈值' if flag else '未达到课堂标记阈值'
    term_sum=sum(term['points'] for term in case['scorecard_terms'])
    # Preserve machine-readable full precision in HTML attributes for independent QA.
    result=(f'<section id="case" class="case-section" aria-labelledby="case-title" data-id="{r["SK_ID_CURR"]}">'
        f'<div class="section-heading"><div><span class="eyebrow">02 / 查看结果</span><h2 id="case-title">申请 {r["SK_ID_CURR"]}</h2></div>'
        f'<span class="purpose" data-sample-use="{safe(r["sample_use"])}">{safe(purpose)}</span></div>'
        f'<p class="sample-notice">{safe(sample_notice)}</p><div class="result-grid">'
        '<article class="risk-card"><div class="card-label">推荐树模型 · 数据集目标事件</div>'
        f'<p id="risk-probability" class="big-number" data-value="{probability!r}">{probability:.4%}</p>'
        f'<p class="risk-status {"marked" if flag else "unmarked"}" id="risk-flag" data-value="{str(flag).lower()}">{flag_text}</p>'
        f'<label for="risk-meter">模型概率在0–100%中的位置</label><meter id="risk-meter" min="0" max="1" value="{probability!r}">{probability:.4%}</meter>'
        f'<p class="threshold">原验证阈值 <b id="threshold" data-value="{threshold!r}">{threshold:.6%}</b> · 按概率 ≥ 阈值标记</p>'
        '<p class="fine-print">目标事件的 X／Y 未披露，不是固定期限PD。标记用于课堂取舍，不是放贷批准或拒绝建议。</p></article>'
        '<article class="score-card"><div class="card-label">传统评分卡 · 自身模型的可加总解释</div>'
        f'<p id="scorecard-score" class="big-number" data-value="{r["scorecard_score"]!r}">{r["scorecard_score"]:.2f}<small>分</small></p>'
        '<div class="addition"><span>基础分<strong id="base-points" '
        f'data-value="{r["scorecard_base_points"]!r}">{r["scorecard_base_points"]:.6f}</strong></span><b>＋</b><span>40项分值合计<strong '
        f'id="term-sum" data-value="{term_sum!r}">{term_sum:+.6f}</strong></span></div>'
        f'<p>评分卡目标事件概率 <b id="scorecard-probability" data-value="{r["scorecard_probability"]!r}">{r["scorecard_probability"]:.4%}</b></p>'
        '<p class="fine-print">高分对应较低的评分卡模型风险。600分／50:1赔率／PDO20。分项解释评分卡自身，不能解释或换算树模型概率。</p></article></div>')
    basic='<section class="panel"><div class="section-heading"><div><span class="eyebrow">03 / 核对输入</span><h2>申请资料与历史摘要</h2></div><span class="count">16项申请资料 ＋ 24项历史</span></div>'
    overview=['NAME_CONTRACT_TYPE','AMT_INCOME_TOTAL','NAME_INCOME_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','FLAG_OWN_REALTY']
    basic+='<dl class="basic-grid">'+''.join(f'<div><dt>{safe(LABELS.get(name,name))}</dt><dd>{safe(display(f[name]))}</dd></div>' for name in overview)+'</dl><div class="history-grid">'
    for prefix,title in [('BUREAU','征信借款历史'),('PREV','历史贷款申请')]:
        count=f[prefix+'_ELIGIBLE_COUNT'];recent=f[prefix+'_DAYS_SINCE_MOST_RECENT']
        message='没有符合本项目时点规则的历史记录。计数为0，金额／比例／距今天数保留null；这不等于此人从未借贷。' if count==0 else '仅使用满足预测时点规则的历史摘要。记录存在时，个别字段仍可能真实缺失。'
        basic+=(f'<article class="history-card" data-history="{prefix}"><h3>{title}</h3><p class="history-count">{count} <small>条合格记录</small></p>'
            f'<p>最近距今：{safe(display(recent))}{" 天" if recent is not None else ""}</p><p class="fine-print">{message}</p></article>')
    basic+='</div><details><summary>查看全部40列实际取值、字段名与单位</summary>'+field_table(f,contract['feature_columns'],contract)+'</details></section>'
    terms='<section class="panel"><span class="eyebrow">04 / 解释分数</span><h2>评分卡的40项构成</h2><p>分箱把原值归到一组，WOE是这一组在评分卡内的编码。分值保留未舍入值核对，总分只在显示时四舍五入。</p>'
    ranked=sorted(case['scorecard_terms'],key=lambda term:abs(term['points']),reverse=True)[:6]
    terms+='<div class="drivers">'+''.join(f'<div><span>{safe(LABELS.get(t["feature"],t["feature"]))}</span><strong class="{"positive" if t["points"]>=0 else "negative"}">{t["points"]:+.3f}</strong></div>' for t in ranked)+'</div>'
    terms+='<details id="scorecard-detail"><summary>展开全部40项分箱、WOE和分值</summary><label class="search-label" for="term-filter">筛选分项</label><input id="term-filter" type="search" placeholder="输入中文名称或字段名"><span id="term-count" role="status">40项</span><div class="table-scroll" tabindex="0" role="region" aria-label="评分卡分项，可横向滚动"><table id="terms-table"><thead><tr><th scope="col">字段</th><th scope="col">原值</th><th scope="col">分组</th><th scope="col">WOE</th><th scope="col">分值</th></tr></thead><tbody>'
    for term in case['scorecard_terms']:
        terms+=(f'<tr data-feature="{safe(term["feature"])}" data-points="{term["points"]!r}"><th scope="row">{safe(LABELS.get(term["feature"],term["feature"]))}<code>{safe(term["feature"])}</code></th>'
            f'<td>{safe(display(term["value"]))}</td><td>{safe(term["group"])}{"（未观察组，中性0）" if term["unobserved_group"] else ""}</td>'
            f'<td>{term["woe"]:.8f}</td><td class="{"positive" if term["points"]>=0 else "negative"}">{term["points"]:+.8f}</td></tr>')
    terms+='</tbody></table></div><p class="fine-print">分组名m:表示缺失；u:表示未知类别。分项是条件关联，不是因果作用或真实审批理由。</p></details></section>'
    model=state.session.models['tree']
    explanation=json.loads((model['directory']/'explanation.json').read_text(encoding='utf-8'))
    important=sorted(explanation['features'],key=lambda v:sum(v['auc_drop'])/len(v['auc_drop']),reverse=True)[:5]
    tree='<section class="panel"><span class="eyebrow">05 / 理解模型</span><h2>树模型参考了哪些线索</h2><p>以下是既有验证样本置换诊断：打乱一个字段后，AUC平均下降越大，说明模型在这份诊断样本中越依赖它。</p><ul class="importance">'
    tree+=''.join(f'<li><span>{safe(LABELS.get(v["feature"],v["feature"]))}<code>{safe(v["feature"])}</code></span><b>{sum(v["auc_drop"])/len(v["auc_drop"]):.5f}</b></li>' for v in important)
    tree+='</ul><p class="fine-print">来自E08事先固定的验证ID最小5000笔、每列2次置换，未重新计算。它描述模型整体依赖，不是这位申请人的可加总贡献；相关变量可能互相掩盖，也不代表因果。</p></section>'
    versions=[('树模型',r['tree_version']+' / '+r['tree_run_id']),('评分卡',r['scorecard_version']+' / '+r['scorecard_run_id']),
        ('LR40对照概率',f'{r["logistic_probability"]:.6%}'),('统一评分包',state.session.report['run_id']),('冻结评估',r['assessment_run_id']),
        ('特征版本',r['feature_version']),('特征契约SHA256',r['feature_contract_sha256']),('数据版本',r['dataset_version']),
        ('名单SHA256',r['split_manifest_sha256']),('实验协议',r['protocol_fingerprint']),('选择SHA256',r['selection_sha256'])]
    trace='<section class="panel provenance"><h2>版本与适用边界</h2><p>页面与批量结果共用E10。数据集标签、样本用途、特征和模型版本保持可追溯。</p><details><summary>查看完整版本与指纹</summary><dl>'+''.join(f'<div><dt>{safe(k)}</dt><dd>{safe(v)}</dd></div>' for k,v in versions)+'</dl></details>'
    trace+='<ul class="limits"><li>当前支持307,511笔已有带标签申请回放。无标签 application_test 的同口径上游未齐，暂不支持。</li><li>申请级随机划分；缺少自然人标识和绝对申请日，不证明跨人或时间外泛化。</li><li>申请级保留测试；存在已披露的早期环境试跑暴露：626笔曾fit＋143笔探针，共769笔，约1.67%。正式模型未复用这些旧拟合对象。</li><li>当前40列对偿债负担和逐期还款覆盖有限；模型概率与课堂阈值未经真实银行审批或收益认证。</li></ul></section></section>'
    return result+basic+terms+tree+trace


def render_page(state, query=''):
    if state.session is None:
        return 503,shell(hero()+'<section class="error-panel" role="alert"><h2>评分包尚未准备好</h2><p>需要完整且匹配的E10成功报告、模型、特征快照和完成回执。请先核对本机交付说明，再重启页面。</p><p>状态代码：<code>'+safe(state.error_code or 'demo_unavailable')+'</code></p><p>当前未进行评分，也不会从缺失资料补造结果。</p></section>')
    current=''
    try:
        values=parse_qs(query,keep_blank_values=True,max_num_fields=4)
        if values:
            if set(values)!={'id'} or len(values['id'])!=1:
                raise DataError('demo_query','Exactly one ID parameter required')
            current=values['id'][0];application_id=parse_id(current)
        else: application_id=state.examples[0][1];current=str(application_id)
        case=state.get_case(application_id)
        return 200,shell(hero()+query_panel(state,current)+render_case(state,case))
    except (ValueError,DataError,KeyError,IndexError) as exc:
        code=exc.code if isinstance(exc,DataError) else 'demo_query'
        status=404 if code=='scoring_unknown_id' else 400
        message=ERRORS.get(code,'申请资料或评分契约不匹配，当前无法展示结果。请核对成功包。')
        return status,shell(hero()+query_panel(state,current)+'<section class="error-panel" role="alert"><h2>暂时无法展示这笔申请</h2><p>'+safe(message)+'</p><p>状态代码：<code>'+safe(code)+'</code></p></section>')


def make_server(state, host='127.0.0.1', port=8765):
    if host!='127.0.0.1':raise DataError('demo_host','Local loopback host required')
    assets=state.root/'app'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                parsed=urlsplit(self.path)
                if len(self.path)>2048:
                    self.send_error(414);return
                if parsed.path in ['/assets/demo.css','/assets/demo.js'] and not parsed.query:
                    path=assets/Path(parsed.path).name
                    data=path.read_bytes();status=200
                    content_type='text/css; charset=utf-8' if path.suffix=='.css' else 'text/javascript; charset=utf-8'
                elif parsed.path=='/' and not parsed.scheme and not parsed.netloc:
                    status,html=render_page(state,parsed.query);data=html.encode('utf-8');content_type='text/html; charset=utf-8'
                elif parsed.path=='/favicon.ico':
                    self.send_response(204);self.end_headers();return
                else:
                    self.send_error(404);return
                self.send_response(status);self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
                self.end_headers();self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:
                # No traceback or local absolute paths sent to the browser.
                self.send_error(500,'Local page unavailable')
        def log_message(self,*args):pass
    return ThreadingHTTPServer((host,port),Handler)
