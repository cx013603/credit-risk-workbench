import type {CaseData} from './api/types';
export function contributionSteps(c:CaseData){
 const e=c.explanation,top=[...e.terms].sort((a,b)=>Math.abs(b.contribution_log_odds)-Math.abs(a.contribution_log_odds)).slice(0,5),ids=new Set(top.map(t=>t.fact_id));
 const rest=e.terms.filter(t=>!ids.has(t.fact_id)).reduce((s,t)=>s+t.contribution_log_odds,0);
 let current=e.base_log_odds;
 const steps=[{id:'base',label:'训练路径参考值',from:0,to:current,delta:current,kind:'base'}];
 for(const t of top){const next=current+t.contribution_log_odds;steps.push({id:t.fact_id,label:c.profile.facts.find(f=>f.fact_id===t.fact_id)!.label,from:current,to:next,delta:t.contribution_log_odds,kind:'factor'});current=next}
 steps.push({id:'rest',label:'其余35项合计',from:current,to:current+rest,delta:rest,kind:'rest'});current+=rest;
 steps.push({id:'score',label:'最终内部得分',from:0,to:current,delta:current,kind:'score'});
 const probability=current>=0?1/(1+Math.exp(-current)):Math.exp(current)/(1+Math.exp(current));
 if(Math.abs(current-e.raw_score)>1e-10||Math.abs(probability-c.model_outputs.tree.probability)>1e-10)throw new Error('贡献合计与原模型结果不一致。');
 return {steps,score:current,probability,rest};
}
