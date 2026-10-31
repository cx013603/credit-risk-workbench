import {ApplicantFacts} from './applicantFacts';
import {field,fieldLabel,sourceName,unitName,valueText,stateName,prose,binText} from './presentation';
import type {CaseData} from './api/types';
import type {HistoryInfo} from './api/historyClient';
import type {NarrativeState} from './narratives';
import type {ReferenceState} from './references';
import {buildSummary} from './summaryData';
import type {SummaryTarget} from './summaryData';
import {decimal} from './format';
export function CaseSummary({c,narratives,reference,history,onAction,onModels}:{c:CaseData;narratives:NarrativeState;reference:ReferenceState;history:HistoryInfo;onAction:(target:SummaryTarget)=>void;onModels:()=>void}){
 let summary;try{summary=buildSummary(c,narratives.data,reference.data||null,history)}catch{return <section className="panel summary-error" role="alert">摘要未通过案例或来源版本核对，请重新查询当前申请。已停止展示关注事项。</section>}
 const pending=narratives.status==='loading';
 return <section className="case-summary" data-testid="case-summary" data-case-key={`${summary.caseKey.population}:${summary.caseKey.application_id}`} data-case-revision={summary.revision}>
  <section className="summary-result" aria-label="本笔模型结果"><div className="summary-probability"><span>目标事件预测概率</span><strong data-testid="risk-probability">{decimal(summary.probability*100)}<small>%</small></strong><p>推荐树模型 · 保存模型回放</p></div><div className="summary-threshold"><span>分析阈值</span><b data-testid="threshold">{decimal(summary.threshold*100)}%</b><p>{summary.gapPoints===0?'与阈值相同':`${summary.gapPoints>0?'高于':'低于'}阈值 ${decimal(Math.abs(summary.gapPoints))} 个百分点`}</p></div><div className="summary-drivers"><p><span>主要上推</span>{prose(summary.upper)}</p><p><span>主要拉低</span>{prose(summary.lower)}</p><small>路径贡献单位为对数几率，不能按概率百分点相加。</small><button className="text-button" onClick={onModels}>展开模型解释 →</button></div></section>
  <ApplicantFacts facts={c.profile.facts.map(f=>({name:f.fact_id,value:f.raw_value,state:f.missing_state}))}/>
  <div className="summary-section-heading"><h2>本笔优先核对</h2><span>按资料边界与模型响应组织 · 最多3项</span></div>
  {narratives.status==='failed'&&<div className="summary-service-note" role="status">证据说明未能载入，相关来源标记为尚未核对。<button className="text-button" onClick={narratives.retry}>重新读取说明</button></div>}
  {pending?<div className="summary-pending" aria-busy="true">正在核对本笔事实、模型响应与证据范围…</div>:<div className="summary-priorities" data-testid="summary-priorities">{summary.items.length?summary.items.map((item,i)=><article className="summary-priority" key={item.id} data-testid="summary-priority" data-fact-id={item.factId} data-target={item.target.kind==='history'?`history:${item.target.table}`:`fact:${item.target.factId}:${item.target.section}`}><header><span className="summary-order">{i+1}</span><h3>{prose(item.title)}</h3></header><p className="summary-fact">{prose(item.fact)}</p><p className="summary-why">{prose(item.why)}</p><div className="summary-evidence">{prose(item.evidence)}</div><button className="summary-action" onClick={()=>onAction(item.target)}>{prose(item.action)}<span>↗</span></button></article>):<p className="summary-pending">当前数据未产生额外核对事项，可进入模型解释查看完整40项。</p>}</div>}
  <section className="summary-scope" aria-label="证据范围" data-testid="summary-scope"><h2>证据范围</h2><div><span>申请字段：16项保存输入，原主表行未独立接入</span>{summary.coverage.map(s=><button key={s.table} onClick={()=>onAction({kind:'history',table:s.table})}>{s.label}：合格{s.count}条 · {s.verified?'来源已核对':'仅保存汇总'}</button>)}</div></section>
  <div className="summary-boundary"><span>{c.sample_use==='train'?'训练案例属于训练人群，不能证明独立效果。':c.sample_use==='unlabeled_scoring'?'无标签案例仅有预测，不能判断实际命中。':'验证案例用于已有申请回放。'} 阈值用于模型分析与比较。</span><details><summary>查看用途、单位与资料限制</summary><p>贡献是模型响应，来源用于核对事实；不代表因果或真实拒贷理由。当前没有当笔额度／期限及完整偿付表现，不生成偿付能力、授信额度或审批建议。币种、收入周期及绝对日期等未知项保留。</p><p>页面小数保留两位，计算使用完整精度；训练参考{reference.status==='ready'?'已核对':'尚未就绪'}，不提供精确百分位。</p>{reference.status==='failed'&&<button className="text-button" onClick={reference.retry}>重新读取训练参考</button>}</details></div>
 </section>;
}
