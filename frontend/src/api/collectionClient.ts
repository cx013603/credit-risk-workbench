import type {CaseData,Envelope,Identity,Population} from './types';
import {ApiError} from './client.ts';
export type Filters={population:Population;sample_use:string;probability_min:number;probability_max:number;history:string;missing:string;signal:string;mark:string};
export const defaults=(population:Population='labeled'):Filters=>({population,sample_use:'all',probability_min:0,probability_max:1,history:'all',missing:'all',signal:'all',mark:'all'});
export type Item={population:Population;application_id:number;case_revision:string;sample_use:string;probability:number;bureau_count:number;previous_count:number;history_coverage:string;observed:number;source_null:number;no_eligible_history:number;signal_count:number;completeness:number;favorite:boolean;pending:boolean};
export type Collection={version:string;filters:Filters;release_manifest_sha256:string;dictionary_sha256:string;marks_revision:number;collection_revision:string;total:number;offset:number;limit:number;order:string;items:Item[];next_offset:number|null;current_in_collection:boolean|null;summary:{total:number;probability_bins:{lower:number;upper:number;count:number}[];history_coverage:Record<string,number>;field_missing:{fact_id:string;label:string;source_null:number}[]};scope:string};
export type Query={filters:Filters;offset:number;limit:number;order:string;current:Identity|null};
const fail=():never=>{throw new ApiError('COLLECTION_INVALID','案例集合未通过范围、版本或数量核对。')};
const int=(n:unknown):n is number=>typeof n==='number'&&Number.isSafeInteger(n)&&n>=0;
const fraction=(n:unknown):n is number=>typeof n==='number'&&Number.isFinite(n)&&n>=0&&n<=1;
const hash=(v:string)=>/^[a-f0-9]{64}$/.test(v||'');
export function assertCollection(value:unknown,q:Query):Collection{
 try{
  const d=value as Collection;
  if(d.version!=='workbench_collection_v1'||!hash(d.release_manifest_sha256)||!hash(d.dictionary_sha256)||!hash(d.collection_revision)||!int(d.marks_revision)||!int(d.total)||d.offset!==q.offset||d.limit!==q.limit||d.order!==q.order||!Array.isArray(d.items)||d.items.length!==Math.min(q.limit,Math.max(0,d.total-q.offset)))fail();
  for(const k of Object.keys(q.filters) as (keyof Filters)[])if(d.filters[k]!==q.filters[k])fail();
  if(d.next_offset!==(q.offset+q.limit<d.total?q.offset+q.limit:null)||d.summary.total!==d.total)fail();
  if(q.current?.population===q.filters.population?typeof d.current_in_collection!=='boolean':d.current_in_collection!==null)fail();
  const bins=d.summary.probability_bins,coverage=d.summary.history_coverage,fields=d.summary.field_missing;
  if(bins.length!==10||bins.reduce((s,b,i)=>{if(b.lower!==i/10||b.upper!==(i+1)/10||!int(b.count))fail();return s+b.count},0)!==d.total||Object.keys(coverage).sort().join()!==['both','bureau_only','none','previous_only'].join()||Object.values(coverage).some(n=>!int(n))||Object.values(coverage).reduce((a,b)=>a+b,0)!==d.total||fields.length!==40||new Set(fields.map(f=>f.fact_id)).size!==40||fields.some(f=>!f.label||!int(f.source_null)||f.source_null>d.total))fail();
  if(new Set(d.items.map(x=>x.application_id)).size!==d.items.length)fail();
  d.items.forEach((x,i)=>{
   if(x.population!==q.filters.population||!int(x.application_id)||x.application_id===0||!['train','validation','unlabeled_scoring'].includes(x.sample_use)||(x.population==='unlabeled')!==(x.sample_use==='unlabeled_scoring')||q.filters.sample_use!=='all'&&x.sample_use!==q.filters.sample_use||!fraction(x.probability)||x.probability<q.filters.probability_min||x.probability>q.filters.probability_max||!int(x.bureau_count)||!int(x.previous_count)||!int(x.observed)||!int(x.source_null)||!int(x.no_eligible_history)||x.observed+x.source_null+x.no_eligible_history!==40||x.completeness!==x.observed/40||!int(x.signal_count)||typeof x.favorite!=='boolean'||typeof x.pending!=='boolean')fail();
   if(!hash(x.case_revision))fail();
   const history=x.bureau_count>0?(x.previous_count>0?'both':'bureau_only'):(x.previous_count>0?'previous_only':'none');
   if(history!==x.history_coverage||q.filters.history!=='all'&&history!==q.filters.history||q.filters.missing==='present'&&x.source_null===0||q.filters.missing==='none'&&x.source_null!==0||q.filters.signal==='present'&&x.signal_count===0||q.filters.signal==='none'&&x.signal_count!==0||q.filters.mark==='favorite'&&!x.favorite||q.filters.mark==='pending'&&!x.pending)fail();
   const prev=d.items[i-1];if(prev){if(q.order==='id_asc'&&prev.application_id>=x.application_id||q.order==='id_desc'&&prev.application_id<=x.application_id||q.order==='probability_asc'&&prev.probability>x.probability||q.order==='probability_desc'&&prev.probability<x.probability||q.order.startsWith('probability')&&prev.probability===x.probability&&prev.application_id>=x.application_id)fail()}
  });return d;
 }catch(e){if(e instanceof ApiError)throw e;return fail()}
}
async function request(path:string,token:string,signal:AbortSignal,body?:unknown):Promise<unknown>{
 let r:Response,e:Envelope;
 try{r=await fetch(path,{signal,method:body?'POST':'GET',headers:{'X-Selection-Token':token,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{})});e=await r.json() as Envelope}catch(error){if(signal.aborted)throw error;throw new ApiError('CONNECTION_FAILED','无法读取案例集合，请重试。')}
 if(e.schema_version!=='workbench_api_v2.0'||e.selection_token!==token)fail();
 if(!r.ok||e.error)throw new ApiError(e.error?.code||'COLLECTION_UNAVAILABLE',e.error?.code==='VERSION_MISMATCH'?'标记已被更新，请刷新列表后重试。':e.error?.message||'案例集合暂不可用。');
 if(e.status!=='ready')fail();return e.payload;
}
export async function fetchCollection(q:Query,token:string,signal:AbortSignal):Promise<Collection>{
 const params=new URLSearchParams({...Object.fromEntries(Object.entries(q.filters).map(([k,v])=>[k,String(v)])),offset:String(q.offset),limit:String(q.limit),order:q.order});
 if(q.current?.population===q.filters.population)params.set('current_id',String(q.current.application_id));
 return assertCollection(await request('/api/v2/cases?'+params,token,signal),q);
}
export async function saveMark(item:Item,d:Collection,favorite:boolean,pending:boolean,token:string,signal:AbortSignal,caseRevision:string){
 const v=await request(`/api/v2/cases/${item.population}/${item.application_id}/marks`,token,signal,{favorite,pending,expected_marks_revision:d.marks_revision,expected_case_revision:caseRevision}) as {marks_revision:number;case_key:Identity;favorite:boolean;pending:boolean};
 if(v.marks_revision!==d.marks_revision+1||v.case_key?.population!==item.population||v.case_key?.application_id!==item.application_id||v.favorite!==favorite||v.pending!==pending)fail();return v;
}
export function qualityOf(c:CaseData){
 const fields=c.profile.facts;return {observed:fields.filter(f=>f.missing_state==='observed').length,missing:fields.filter(f=>f.missing_state==='source_null').length,noHistory:fields.filter(f=>f.missing_state==='no_eligible_history').length};
}
