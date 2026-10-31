import type {CaseData,Identity} from './types';
import type {ReferenceData} from './referenceClient';
import {ApiError,assertCase,fetchCase,parseIdentity} from './client.ts';
import {assertReference,fetchReference} from './referenceClient.ts';
export type Pair={data:CaseData;reference:ReferenceData};
export type ComparisonEntry={key:string;identity:Identity;status:'loading'|'ready'|'failed';pair:Pair|null;error:string|null;attempt:number};
export type ComparisonState={entries:ComparisonEntry[];revision:number};
export const identityKey=(i:Identity)=>`${i.population}:${i.application_id}`;
export function validatePair(pair:Pair,identity:Identity):Pair{
 const data=assertCase(pair.data,identity);
 return {data,reference:assertReference(pair.reference,data,pair.reference.reference.reference_sha256)};
}
export function incompatibilities(pairs:Pair[]):string[]{
 if(pairs.length<2)return [];
 const first=pairs[0],reasons=new Set<string>();
 const facts=(c:CaseData)=>JSON.stringify(c.profile.facts.map(f=>[f.fact_id,f.label,f.unit,f.rule.formula,f.rule.time_rule,f.rule.version]));
 const tree=(c:CaseData)=>JSON.stringify([c.model_outputs.tree.version,c.model_outputs.tree.run_id,c.model_outputs.tree.threshold,c.model_outputs.tree.threshold_scope]);
 const versions=(c:CaseData)=>JSON.stringify(Object.entries(c.versions).sort(([a],[b])=>a.localeCompare(b)));
 for(const p of pairs.slice(1)){
  if(p.data.case_key.population!==first.data.case_key.population)reasons.add('申请来源不同；请分别按同来源选择案例。');
  if(versions(p.data)!==versions(first.data)||tree(p.data)!==tree(first.data))reasons.add('模型、特征、划分或运行版本不同。');
  if(facts(p.data)!==facts(first.data))reasons.add('字段、单位或时间口径不同。');
  if(p.data.explanation.method!==first.data.explanation.method||p.data.explanation.unit!==first.data.explanation.unit||p.data.explanation.base_log_odds!==first.data.explanation.base_log_odds)reasons.add('贡献方法、单位或参考值不同。');
  const r=p.reference.reference,f=first.reference.reference;
  if(r.reference_sha256!==f.reference_sha256||r.population!==f.population||r.rows!==f.rows||r.version!==f.version)reasons.add('训练参考人群或参考包版本不同。');
 }
 return [...reasons];
}
type RequestPair=(identity:Identity,token:string,signal:AbortSignal)=>Promise<Pair>;
async function fetchPair(identity:Identity,token:string,signal:AbortSignal):Promise<Pair>{
 const data=await fetchCase(identity,token,signal);const reference=await fetchReference(data,token+'-reference',signal);return {data,reference};
}
// A single queue fits the existing single-flight worker; removal/retry use
// independent attempt tokens, so late success/failure cannot fill another slot.
export class ComparisonCoordinator{
 private state:ComparisonState={entries:[],revision:0};private tail:Promise<void>=Promise.resolve();private attempts=0;private controllers=new Map<string,AbortController>();
 private notify:(state:ComparisonState)=>void;private request:RequestPair;
 constructor(notify:(state:ComparisonState)=>void,request:RequestPair=fetchPair){this.notify=notify;this.request=request}
 snapshot(){return {...this.state,entries:this.state.entries.map(e=>({...e,identity:{...e.identity}}))}}
 private emit(){this.state={...this.state,revision:this.state.revision+1};this.notify(this.snapshot())}
 add(identity:Identity){
  parseIdentity(identity.population,String(identity.application_id));const key=identityKey(identity);
  if(this.state.entries.some(e=>e.key===key))throw new ApiError('DUPLICATE_CASE','这笔申请已经在比较中。');
  if(this.state.entries.length>=4)throw new ApiError('COMPARE_LIMIT','最多比较4笔申请，请先移除一笔。');
  const entry:ComparisonEntry={key,identity:{...identity},status:'loading',pair:null,error:null,attempt:++this.attempts};this.state.entries=[...this.state.entries,entry];this.emit();this.enqueue(entry);
 }
 retry(key:string){const old=this.state.entries.find(e=>e.key===key);if(!old)return;this.controllers.get(key)?.abort();const entry={...old,status:'loading' as const,pair:null,error:null,attempt:++this.attempts};this.state.entries=this.state.entries.map(e=>e.key===key?entry:e);this.emit();this.enqueue(entry)}
 private enqueue(entry:ComparisonEntry){
  const controller=new AbortController();this.controllers.set(entry.key,controller);
  const current=()=>!controller.signal.aborted&&this.state.entries.some(e=>e.key===entry.key&&e.attempt===entry.attempt);
  this.tail=this.tail.then(async()=>{
   if(!current())return;
   try{const pair=validatePair(await this.request(entry.identity,`comparison-${entry.attempt}`,controller.signal),entry.identity);if(current()){this.state.entries=this.state.entries.map(e=>e.key===entry.key?{...e,status:'ready',pair,error:null}:e);this.emit()}}
   catch(error){if(current()){this.state.entries=this.state.entries.map(e=>e.key===entry.key?{...e,status:'failed',pair:null,error:error instanceof Error?error.message:'当前申请未能载入。'}:e);this.emit()}}
   finally{if(this.controllers.get(entry.key)===controller)this.controllers.delete(entry.key)}
  });
 }
 remove(key:string){this.controllers.get(key)?.abort();this.controllers.delete(key);this.state.entries=this.state.entries.filter(e=>e.key!==key);this.emit()}
 clear(){for(const c of this.controllers.values())c.abort();this.controllers.clear();this.state.entries=[];this.emit()}
 move(key:string,direction:-1|1){const rows=[...this.state.entries],i=rows.findIndex(e=>e.key===key),j=i+direction;if(i<0||j<0||j>=rows.length)return;[rows[i],rows[j]]=[rows[j],rows[i]];this.state.entries=rows;this.emit()}
 sort(order:'asc'|'desc'){
  const rows=this.state.entries,pairs=rows.flatMap(e=>e.pair?[e.pair]:[]);
  if(rows.length<2||rows.some(e=>e.status!=='ready')||incompatibilities(pairs).length)throw new ApiError('COMPARE_INCOMPATIBLE','仅在全部案例加载成功且口径一致时，才允许按预测排序。');
  this.state.entries=[...rows].sort((a,b)=>(order==='asc'?1:-1)*(a.pair!.data.model_outputs.tree.probability-b.pair!.data.model_outputs.tree.probability)||a.identity.application_id-b.identity.application_id);this.emit();
 }
 whenIdle(){return this.tail}
 dispose(){for(const c of this.controllers.values())c.abort();this.controllers.clear()}
}
