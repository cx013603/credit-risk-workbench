"""Local explanation page using the same verified scoring/explanation session."""
from http.server import BaseHTTPRequestHandler
import json
import threading
from urllib.parse import parse_qs, urlsplit

from .demo import display, parse_id, safe
from .raw import DataError
from .scoring_api import BoundedServer


def shell(content):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>看懂这一笔 · 树模型解释</title><link rel="stylesheet" href="/assets/tree_explanation.css"></head>'
        '<body><header><span>信用风险学习项目 / 推荐树模型</span><h1>看懂这一笔预测</h1>'
        '<p>沿模型的实际判断路径，查看哪些字段把内部得分推高或拉低。</p></header><main>'+content+
        '</main><footer>公开匿名信贷数据 · 本机分析与研究 · 数据使用条件见来源说明。</footer></body></html>')


def form(population, application_id):
    options = ''.join(f'<option value="{p}" {"selected" if population == p else ""}>{label}</option>'
                      for p, label in [('labeled', '带标签历史回放'), ('unlabeled', '无标签申请')])
    return ('<section class="panel"><form action="/" method="get"><label>申请来源<select name="population">'+options+
            '</select></label><label>申请编号<input name="id" inputmode="numeric" required value="'+safe(application_id)+
            '"></label><button type="submit">查看预测与解释</button></form><nav>'
            '<a href="/?population=labeled&amp;id=100006">验证示例 100006</a>'
            '<a href="/?population=labeled&amp;id=100003">训练示例 100003</a>'
            '<a href="/?population=unlabeled&amp;id=100001">无标签示例 100001</a></nav></section>')


def render_case(case):
    r = case['result']
    purpose = {'train': '训练样本', 'validation': '验证样本', 'test': '申请级保留测试样本',
               'unlabeled_scoring': '无标签申请，仅预测'}.get(r['sample_use'], r['sample_use'])
    cards = (f'<section id="case" data-id="{r["SK_ID_CURR"]}" class="panel"><div class="section-top">'
        f'<h2>申请 {r["SK_ID_CURR"]}</h2><span class="tag">{safe(purpose)}</span></div>'
        '<div class="metrics"><div><span>树模型预测概率</span><strong>'+f'{r["risk_probability"]:.4%}'+
        '</strong></div><div><span>原课堂阈值</span><strong>'+f'{r["threshold"]:.4%}'+
        '</strong></div><div><span>原阈值标记</span><strong>'+('达到阈值' if r['risk_flag'] else '低于阈值')+
        '</strong></div></div><p>预测目标沿用数据集定义的还款困难事件；未补固定期限，也不是银行审批结论。</p></section>')
    equation = (f'<section class="panel"><h2>先核对：解释能否加回原预测</h2><div class="equation">'
        f'<span>参考值 <b>{case["base_log_odds"]:.8f}</b></span><span>＋ 40项贡献 <b>{case["contribution_sum"]:+.8f}</b></span>'
        f'<span>＝ 内部得分 <b>{case["raw_score"]:.8f}</b></span></div>'
        f'<p>将内部得分代入 sigmoid(z) = 1 / (1 + exp(−z))，得到上面的原树概率。加总最大误差 '
        f'<code>{case["checks"]["additivity_max_difference"]:.3g}</code>，概率最大误差 '
        f'<code>{case["checks"]["probability_max_difference"]:.3g}</code>。</p>'
        '<p class="note">单位是对数几率：正值推高内部风险得分，负值拉低。各项不能当作概率百分点相加。'
        '参考值来自训练记录在树叶中的分布，是训练内部得分均值；它不是训练坏样本率或平均预测概率。</p></section>')
    drivers = '<section class="panel"><h2>这一笔的主要影响因素</h2><div class="driver-columns">'
    for name, title, cls in [('top_increasing', '推高内部得分', 'increase'), ('top_decreasing', '拉低内部得分', 'decrease')]:
        drivers += '<div><h3>'+title+'</h3><ol>'
        for t in case[name]:
            drivers += (f'<li><div><span>{safe(t["label"])}</span><b class="{cls}">{t["contribution_log_odds"]:+.6f}</b></div>'
                        f'<small>{safe(t["feature"])} · 原值：{safe(display(t["value"]))}</small></li>')
        drivers += '</ol></div>'
    drivers += '</div><p>仅列绝对影响较大的正／负项；其余字段保留在下面的完整加总中。</p><details><summary>展开全部40项字段与贡献</summary>'
    drivers += '<div class="table-scroll"><table><thead><tr><th>字段</th><th>原值</th><th>贡献（对数几率）</th></tr></thead><tbody>'
    for t in case['terms']:
        drivers += (f'<tr data-feature="{safe(t["feature"])}" data-contribution="{t["contribution_log_odds"]!r}">'
            f'<th>{safe(t["label"])}<small>{safe(t["feature"])}</small></th><td>{safe(display(t["value"]))}</td>'
            f'<td>{t["contribution_log_odds"]:+.9f}</td></tr>')
    drivers += '</tbody></table></div></details></section>'
    limits = ('<section class="panel"><h2>怎样理解这份解释</h2><ul>'
        '<li>方法是训练记录数加权的树路径贡献。它与当前树逐笔对应，分配结果受路径顺序和相关字段影响。</li>'
        '<li>它不是SHAP、全局置换重要性或因果分析；不能当作真实拒贷理由或保证有效的改善建议。</li>'
        '<li>这些是推荐树模型的内部得分贡献；评分卡分值仍属于另一模型，两者不混用。</li>'
        '<li>自然人标识、绝对申请日期、标签X／Y未知；原测试早期769笔暴露及40列覆盖局限保留。</li>'
        '<li>无标签申请没有真实结果，不能据此验证准确率。训练样本用于流程演示，不能当独立效果。</li></ul>'
        f'<details><summary>查看模型与输入版本</summary><dl><dt>方法</dt><dd>{safe(case["method"])}</dd>'
        f'<dt>树模型</dt><dd>{safe(r["tree_version"])} / {safe(r["tree_run_id"])}</dd>'
        f'<dt>特征契约</dt><dd>{safe(r["feature_contract_sha256"])}</dd>'
        f'<dt>数据版本</dt><dd>{safe(r["dataset_version"])}</dd>'
        f'<dt>样本用途</dt><dd>{safe(r["sample_use"])}</dd></dl></details></section>')
    return cards+equation+drivers+limits


def render_page(session, query):
    population, value = 'labeled', '100006'
    try:
        q = parse_qs(query, keep_blank_values=True, max_num_fields=3)
        if q:
            if set(q) != {'population', 'id'} or any(len(v) != 1 for v in q.values()):
                raise DataError('tree_explanation_query', 'Exactly one population and ID required')
            population, value = q['population'][0], q['id'][0]
        case = session.explain_ids(population, [parse_id(value)])[0]
        return 200, shell(form(population, value)+render_case(case))
    except (DataError, ValueError) as exc:
        code = exc.code if isinstance(exc, DataError) else 'tree_explanation_query'
        status = 404 if code == 'tree_explanation_unknown_id' else 400
        return status, shell(form(population, value)+'<section class="panel error" role="alert"><h2>无法展示这笔申请</h2>'
            '<p>请核对申请来源和正整数编号；当前结果已清空。</p><code>'+safe(code)+'</code></section>')


def make_server(session, host='127.0.0.1', port=8768):
    if host != '127.0.0.1' or type(port) is not int or not 0 <= port <= 65535:
        raise DataError('tree_explanation_host', 'Loopback and valid port required')
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        server_version = 'CreditRiskTreeExplanation/1'
        sys_version = ''
        def log_message(self, *_): pass
        def reply(self, status, body, kind):
            self.send_response(status); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Connection', 'close')
            self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers(); self.close_connection = True
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
        def do_GET(self):
            try:
                p = urlsplit(self.path)
                if len(self.path) > 2048:
                    self.reply(414, b'Request too long', 'text/plain'); return
                if p.scheme or p.netloc:
                    self.reply(400, b'Local path required', 'text/plain'); return
                if p.path == '/health/ready' and not p.query:
                    body = json.dumps({'ready': True, 'reference_run_id': session.report['run_id'],
                        'populations': {k: v.features.num_rows for k, v in session.sources.items()}}).encode()
                    self.reply(200, body, 'application/json'); return
                if p.path == '/assets/tree_explanation.css' and not p.query:
                    self.reply(200, (session.root/'app/tree_explanation.css').read_bytes(), 'text/css; charset=utf-8'); return
                if p.path != '/':
                    self.reply(404, b'Not found', 'text/plain'); return
                if not lock.acquire(blocking=False):
                    self.reply(429, b'Explanation busy; retry', 'text/plain'); return
                try: status, page = render_page(session, p.query)
                finally: lock.release()
                self.reply(status, page.encode('utf-8'), 'text/html; charset=utf-8')
            except Exception:
                self.reply(500, b'Local explanation unavailable', 'text/plain')
    return BoundedServer((host, port), Handler, None)
