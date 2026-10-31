import {field,fieldLabel,sourceName,unitName,valueText,stateName,prose,binText} from './presentation';
import {metric} from './format';
import {useEffect, useRef, useState} from 'react';
import {Alert, Button, Skeleton} from 'antd';
import type {CaseData} from './api/types';
import {fetchHistory} from './api/historyClient';
import type {HistoryData, HistoryInfo, HistoryQuery, HistorySelection, HistoryTable} from './api/historyClient';

export function useHistoryIndex(): HistoryInfo {
 const [info, setInfo] = useState<HistoryInfo>({status: 'loading'});
 useEffect(() => {
  const controller = new AbortController();
  void fetch('/api/v2/history-index', {signal: controller.signal}).then(r => r.json()).then(body => {
   if (!controller.signal.aborted) setInfo(body?.status === 'ready' && body.payload?.status === 'ready' ? body.payload : {status: 'not_connected'});
  }).catch(() => {if (!controller.signal.aborted) setInfo({status: 'failed'})});
  return () => controller.abort();
 }, []);
 return info;
}

const value = (v: string | number | null | undefined) => valueText('',v);
const relative = (v: string | number | null | undefined, label: string) => typeof v !== 'number' ? `${label}缺失` : v < 0 ? `${label}：当前申请前 ${metric(-v)} 天` : v === 0 ? `${label}：与当前申请同日` : `${label}：当前申请后 ${metric(v)} 天`;
const amountLabel: Record<string, string> = {currency1_nonnegative: '金额满足原算式条件（含有效0）', currency_missing: '币种缺失，金额不合计', other_currency_not_summed: '其他币种，金额不合计', amount_missing: '匿名币种一 金额缺失', amount_negative: '匿名币种一 金额为负，不合计'};

export function HistoryEvidence({c, info, table, feature, pageSize = 5, initialSelection='eligible'}: {c: CaseData; info: HistoryInfo; table: HistoryTable; feature?: string; pageSize?: number;initialSelection?:HistorySelection}) {
 const [selection, setSelection] = useState<HistorySelection>(feature ? 'inputs' : initialSelection);
 const [offset, setOffset] = useState(0), [retry, setRetry] = useState(0);
 const [state, setState] = useState<{key: string; data?: HistoryData; error?: string}>({key: ''});
 const sequence = useRef(0);
 const key = `${c.case_key.population}:${c.case_key.application_id}:${c.case_revision}:${info.index_manifest_sha256}:${table}:${feature || ''}:${selection}:${offset}:${retry}`;
 useEffect(() => {
  const controller = new AbortController(); const current = ++sequence.current;
  setState({key});
  if (info.status === 'ready') {
   const query: HistoryQuery = {table, feature, selection, offset, limit: pageSize};
   void fetchHistory(c, info, query, `history-${current}`, controller.signal).then(data => {
    if (!controller.signal.aborted && current === sequence.current) setState({key, data});
   }).catch(e => {if (!controller.signal.aborted && current === sequence.current) setState({key, error: e instanceof Error ? e.message : '历史资料暂不可用。'})});
  }
  return () => controller.abort();
 }, [key, c, info, table, feature, selection, offset, pageSize]);
 const data = state.key === key ? state.data : undefined, error = state.key === key ? state.error : undefined;
 if (info.status !== 'ready') return <div className="pending-panel"><strong>{info.status === 'loading' ? '正在检查历史索引…' : '历史索引尚未载入'}</strong><p>当前摘要仍取自保存输入；请使用已配置历史索引的新实例查看来源记录。</p></div>;
 return <section className="history-evidence" data-testid="history-evidence" data-case-key={`${c.case_key.population}:${c.case_key.application_id}`} data-feature={feature || ''} data-index-revision={info.index_manifest_sha256}>
  <div className="history-filters" aria-label="历史记录范围">{([...(feature ? [['inputs', '进入该因素算式']] : []), ['eligible', '合格记录'], ['excluded', '被时间排除'], ['all', '全部来源']] as [HistorySelection, string][]).map(([id, label]) => <button type="button" key={id} className={selection === id ? 'active' : ''} onClick={() => {setSelection(id); setOffset(0)}}>{label}</button>)}</div>
  {error ? <div data-testid="history-error"><Alert showIcon type="error" title="未能载入历史记录" description={error}/><Button onClick={() => setRetry(retry + 1)}>重新读取</Button></div> : !data ? <div aria-busy="true" data-testid="history-loading"><Skeleton active paragraph={{rows: 4}}/></div> : <>
   <div className="history-audit" data-testid="history-audit"><span>来源 <b>{data.audit.raw_count}</b> 条</span><span>合格 <b>{data.audit.eligible_count}</b> 条</span><span>时间排除 <b>{data.audit.excluded_count}</b> 条</span>{feature && <span>进入本算式 <b>{data.audit.feature_inputs_count}</b> 条</span>}</div>
   <p className="history-note">{data.audit.history_state === 'no_source_history' ? '当前来源没有记录；计数为0，比例、金额和距今天数等保留缺失值。' : data.audit.history_state === 'only_time_excluded' ? '存在来源记录，但全部被时间规则排除；不进入模型汇总。' : '先按时间筛选，再按各因素口径计算。'} 无合格历史不代表没有逾期。</p>
   {feature && <p className="history-formula-value">保存的模型输入：<strong>{value(data.summary_values[feature])}</strong> <span>来源记录用于复核聚合，与树模型贡献分别展示。</span></p>}
   {data.items.length === 0 ? <div className="history-empty" data-testid="history-empty">此范围没有记录。{data.audit.excluded_count > 0 && selection !== 'excluded' ? '可切换“被时间排除”核对原因。' : ''}</div> : <div className="history-record-list">{data.items.map(record => <article className="history-record" key={record.source_key} data-testid="history-record" data-source-key={record.source_key} data-projection-sha256={record.normalized_projection_sha256}>
    <header><h4>{table === 'previous' ? '历史申请' : '征信借款'} <span>#{record.source_key}</span></h4><span className={`badge ${record.eligible ? 'teal' : 'gray'}`}>{record.eligible ? '时间合格' : '已排除'}</span></header>
    <p className="history-record-time">{relative(record.source_values[table === 'previous' ? 'DAYS_DECISION' : 'DAYS_CREDIT'], table === 'previous' ? '决策日' : '借款日')}</p>
    {table === 'previous' ? <p>申请合同类型：<b>{value(record.source_values.NAME_CONTRACT_TYPE)}</b></p> : <><p>{relative(record.source_values.DAYS_CREDIT_UPDATE, '更新日')}</p><p>状态：{value(record.source_values.CREDIT_ACTIVE)} · 币种：{value(record.source_values.CREDIT_CURRENCY)}</p><p>信用金额原值：{value(record.source_values.AMT_CREDIT_SUM)} <small>数据集金额尺度</small></p><p className="history-note">{record.eligible?amountLabel[record.amount_state]:'金额仅作来源核对；此记录已被时间规则排除。'}</p></>}
    <div className="history-record-reason">{prose(record.reason_text)}{feature && <span> · {record.feature_input_used ? '进入该因素算式' : '不进入该因素算式'}</span>}</div>
    <details className="technical"><summary>查看来源记录明细</summary><dl>{Object.entries(record.source_values).map(([name, raw]) => <div key={name}><dt>{fieldLabel(name)}</dt><dd>{value(raw)}</dd></div>)}</dl></details>
   </article>)}</div>}
   <div className="history-pagination"><span>本范围共 {data.total} 条 · 来源键升序{data.total > 0 && ` · ${Math.min(offset + 1, data.total)}–${Math.min(offset + pageSize, data.total)}`}</span><div><Button size="small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>上一页</Button><Button size="small" disabled={data.next_offset === null} onClick={() => setOffset(data.next_offset!)}>下一页</Button></div></div>
   
  </>}
 </section>;
}

export function HistoryBrowser({c, info,initialTable='previous',initialSelection='eligible'}: {c: CaseData; info: HistoryInfo;initialTable?:HistoryTable;initialSelection?:HistorySelection}) {
 const [table, setTable] = useState<HistoryTable>(initialTable);
 return <section className="panel"><div className="panel-heading"><h2>历史来源与准入记录</h2><span className="badge gray">只读证据索引</span></div><div className="history-table-tabs"><button className={table === 'previous' ? 'active' : ''} onClick={() => setTable('previous')}>历史申请来源</button><button className={table === 'bureau' ? 'active' : ''} onClick={() => setTable('bureau')}>征信借款来源</button></div><HistoryEvidence key={`${c.case_key.population}:${c.case_key.application_id}:${table}`} c={c} info={info} table={table} pageSize={10} initialSelection={initialSelection}/></section>;
}
