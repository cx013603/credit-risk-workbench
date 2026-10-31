import type {CaseData,Fact,Contribution} from './api/types';
import type {Narratives} from './api/narrativeClient';
import type {ReferenceData} from './api/referenceClient';
import type {HistoryInfo} from './api/historyClient';
import {assertNarratives} from './api/narrativeClient.ts';
import {assertReference} from './api/referenceClient.ts';
import {decimal,metric} from './format.ts';
export type SummaryTarget={kind:'fact';factId:string;section:'description'|'source'}|{kind:'history';table:'bureau'|'previous'};
export type SummaryItem={id:string;factId:string;title:string;fact:string;why:string;evidence:string;action:string;target:SummaryTarget;priority:number;weight:number};
export function buildSummary(c:CaseData,n:Narratives|null,r:ReferenceData|null,h:HistoryInfo){
 if(n)assertNarratives(n,c,n.template_sha256);
 if(r)assertReference(r,c,r.reference.reference_sha256);
 if(n?.history_manifest_sha256&&h.status==='ready'&&n.history_manifest_sha256!==h.index_manifest_sha256)throw new Error('历史来源版本与当前摘要不一致。');
 const facts=new Map(c.profile.facts.map(f=>[f.fact_id,f])),terms=new Map(c.explanation.terms.map(t=>[t.fact_id,t]));
 const term=(id:string)=>terms.get(id)!.contribution_log_odds;
 const narrative=(id:string)=>n?.rows.find(x=>x.fact_id===id);
 const verified=(id:string)=>h.status==='ready'&&!!narrative(id)?.evidence.verified_source_records;
 const raw=(f:Fact)=>f.raw_value===null?'null（保存值为空）':typeof f.raw_value==='number'?metric(f.raw_value):f.raw_value;
 const direction=(v:number)=>v>0?'上推':v<0?'拉低':'本例贡献为零';
 const signed=(v:number)=>(v>0?'+':'')+decimal(v);
 const evidence=(f:Fact)=>f.value_kind==='source_field'?'仅保存模型输入，原始主表行尚未独立接入':verified(f.fact_id)?`来源已核对，${narrative(f.fact_id)!.evidence.matching_record_count}条进入本项算式`:'仅保存汇总，来源记录尚未核对';
 const ranked=[...c.explanation.terms].sort((a,b)=>Math.abs(b.contribution_log_odds)-Math.abs(a.contribution_log_odds));
 const upper=ranked.filter(t=>t.contribution_log_odds>0).slice(0,2),lower=ranked.filter(t=>t.contribution_log_odds<0).slice(0,1);
 const describe=(ts:Contribution[],empty:string)=>ts.length?ts.map(t=>`${facts.get(t.fact_id)!.label}（${signed(t.contribution_log_odds)}）`).join('、'):empty;
 const candidates:SummaryItem[]=[];
 for(const [prefix,table,label] of [['BUREAU','bureau','征信'],['PREV','previous','历史申请']] as const){
  const id=prefix+'_ELIGIBLE_COUNT',f=facts.get(id)!;
  if(f.raw_value===0)candidates.push({id:'gap-'+table,factId:id,title:`${label}没有合格记录`,fact:`保存汇总：合格${label}记录为0条。`,why:'相关比例、金额或距今天数可能为空；这与有效0不同，不能据此推断没有借贷或逾期。',evidence:verified(id)?'来源已核对，可继续查看全部来源及时间排除情况':'来源尚未核对，先区分保存汇总与原记录',action:`查看${label}准入范围`,target:{kind:'history',table},priority:0,weight:Math.max(...c.explanation.terms.filter(t=>t.fact_id.startsWith(prefix+'_')).map(t=>Math.abs(t.contribution_log_odds)))})
 }
 const nulls=c.profile.facts.filter(f=>f.value_kind==='source_field'&&f.raw_value===null).sort((a,b)=>Math.abs(term(b.fact_id))-Math.abs(term(a.fact_id)));
 if(nulls.length){const f=nulls[0];candidates.push({id:'application-null',factId:f.fact_id,title:`${nulls.length}项申请输入为空`,fact:`${f.label}的保存值为空${nulls.length>1?'，另有'+(nulls.length-1)+'项为空':''}。`,why:`此项模型路径贡献${signed(term(f.fact_id))}对数几率；模型有输出不代表原资料已补齐。`,evidence:evidence(f),action:`查看${f.label}及缺失说明`,target:{kind:'fact',factId:f.fact_id,section:'description'},priority:1,weight:Math.abs(term(f.fact_id))})}
 const reviewed=new Set<string>();
 for(const row of r?.rows||[]){if(!['unknown_category','rare_category','outside_train_range','unseen_nonnull'].includes(row.position_status))continue;const f=facts.get(row.fact_id)!;reviewed.add(f.fact_id);candidates.push({id:'reference-'+f.fact_id,factId:f.fact_id,title:`${f.label}需要参考范围核对`,fact:`保存值：${raw(f)}。`,why:row.description,evidence:evidence(f),action:`查看${f.label}的训练参考`,target:{kind:'fact',factId:f.fact_id,section:'description'},priority:2,weight:Math.abs(term(f.fact_id))})}
 for(const t of ranked){const f=facts.get(t.fact_id)!;if(!t.contribution_log_odds||f.raw_value===null||reviewed.has(f.fact_id))continue;
  const v=t.contribution_log_odds;const count=narrative(f.fact_id)?.evidence.matching_record_count;
  candidates.push({id:'factor-'+f.fact_id,factId:f.fact_id,title:`${f.label}是${direction(v)}项`,fact:`保存值：${raw(f)}。`,why:`路径贡献${signed(v)}对数几率，${direction(v)}内部得分；与其余因素共同形成原预测。`,evidence:evidence(f),action:verified(f.fact_id)?`核对${count}条来源记录`:`查看${f.label}与模型响应`,target:{kind:'fact',factId:f.fact_id,section:verified(f.fact_id)?'source':'description'},priority:3,weight:Math.abs(v)})
 }
 candidates.sort((a,b)=>a.priority-b.priority||b.weight-a.weight);
 const used=new Set<string>(),items:SummaryItem[]=[];
 for(const item of candidates){const group=item.factId.startsWith('BUREAU_')?'bureau':item.factId.startsWith('PREV_')?'previous':item.factId;
  if(used.has(group)||items.some(x=>x.factId===item.factId))continue;used.add(group);items.push(item);if(items.length===3)break;
 }
 const coverage=([['PREV_ELIGIBLE_COUNT','历史申请','previous'],['BUREAU_ELIGIBLE_COUNT','征信','bureau']] as const).map(([id,label,table])=>({label,count:Number(facts.get(id)!.raw_value),verified:verified(id),table}));
 return {version:'case_reading_summary_v1',caseKey:c.case_key,revision:c.case_revision,probability:c.model_outputs.tree.probability,threshold:c.model_outputs.tree.threshold,gapPoints:(c.model_outputs.tree.probability-c.model_outputs.tree.threshold)*100,upper:describe(upper,'本例没有上推项'),lower:describe(lower,'本例没有拉低项'),items,coverage};
}
