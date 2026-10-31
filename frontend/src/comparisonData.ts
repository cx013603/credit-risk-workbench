import type {Pair} from './api/comparisonClient';
import {identityKey,incompatibilities} from './api/comparisonClient.ts';
import {qualityOf} from './api/collectionClient.ts';
export function comparisonData(pairs:Pair[]){
 const reasons=incompatibilities(pairs);if(pairs.length<2||pairs.length>4||reasons.length)throw new Error('当前案例尚未形成2～4笔同口径比较。');
 const fields=pairs[0].data.profile.facts,stablePairs=[...pairs].sort((a,b)=>identityKey(a.data.case_key).localeCompare(identityKey(b.data.case_key))); 
 const ranked=fields.map(f=>({id:f.fact_id,label:f.label,total:stablePairs.reduce((sum,p)=>sum+Math.abs(p.data.explanation.terms.find(t=>t.fact_id===f.fact_id)!.contribution_log_odds),0)})).sort((a,b)=>b.total-a.total||a.id.localeCompare(b.id));
 const top=ranked.slice(0,6),ids=new Set(top.map(t=>t.id));
 const columns=pairs.map(p=>{
  const c=p.data,q=qualityOf(c),values=new Map(c.profile.facts.map(f=>[f.fact_id,f]));
  const terms=top.map(t=>({id:t.id,label:t.label,value:c.explanation.terms.find(x=>x.fact_id===t.id)!.contribution_log_odds}));
  const rest=c.explanation.terms.filter(t=>!ids.has(t.fact_id)).reduce((s,t)=>s+t.contribution_log_odds,0);
  const total=terms.reduce((s,t)=>s+t.value,rest),score=c.explanation.base_log_odds+total;
  if(Math.abs(score-c.explanation.raw_score)>1e-10||Math.abs(1/(1+Math.exp(-score))-c.model_outputs.tree.probability)>1e-10)throw new Error('比较贡献未还原原分数。');
  return {key:`${c.case_key.population}:${c.case_key.application_id}`,case:c,quality:q,bureau:values.get('BUREAU_ELIGIBLE_COUNT')!.raw_value as number,previous:values.get('PREV_ELIGIBLE_COUNT')!.raw_value as number,terms,rest,total,score};
 });
 const contributionMax=Math.max(.01,...columns.flatMap(c=>c.terms.map(t=>Math.abs(t.value)))),historyMax=Math.max(1,...columns.flatMap(c=>[c.bureau,c.previous]));
 return {columns,top,contributionMax,historyMax,restCount:40-top.length};
}
export function fieldComparability(pairs:Pair[],id:string):string{
 const facts=pairs.map(p=>p.data.profile.facts.find(f=>f.fact_id===id));
 if(facts.some(f=>!f))return '字段未接入，不可比较';
 if(new Set(facts.map(f=>f!.unit)).size>1)return '单位不同，不可比较';
 if(facts.some(f=>f!.missing_state!=='observed'||f!.raw_value===null))return '含缺失／无合格历史，不计算差值';
 if(facts[0]!.unit.includes('unverified')||facts[0]!.unit.includes('anonymous_currency'))return '金额单位未核验，仅并列原值';
 return '同一字段口径，仅并列原值';
}
