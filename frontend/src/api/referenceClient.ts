import type {CaseData,Envelope} from './types';
import {ApiError} from './client.ts';
export type ReferenceBin={index:number;label:string;count:number;share:number;current:boolean};
export type ReferenceRow={fact_id:string;raw_value:string|number|null;unit:string;missing_state:string;kind:string;value_type:string;rows:number;missing_count:number;bins:ReferenceBin[];current_bucket:number;position_status:string;current_bucket_count:number;current_bucket_share:number;minimum:number|null;maximum:number|null;percentile:null;description:string};
export type ReferenceData={case_key:CaseData['case_key'];case_revision:string;sample_use:string;reference_revision:string;reference:{status:string;version:string;reference_sha256:string;dictionary_sha256:string;release_manifest_sha256:string;rows:number;feature_count:number;population:string;percentile_status:string};rows:ReferenceRow[];reconciliation:{method:string;unit:string;base:number;top:CaseData['explanation']['terms'];rest_count:number;rest:number;total:number;raw_score:number;probability:number}};
const invalid=():never=>{throw new ApiError('TRAIN_REFERENCE_INVALID','训练参考未通过当前案例、计数或版本核对。')};
const hash=(v:unknown)=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const near=(a:number,b:number)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-10;
export function assertReference(value:unknown,c:CaseData,seal:string):ReferenceData{
 try{
  const d=value as ReferenceData,r=d.reference,e=c.explanation,z=d.reconciliation;
  if(d.case_key.population!==c.case_key.population||d.case_key.application_id!==c.case_key.application_id||d.case_revision!==c.case_revision||d.sample_use!==c.sample_use||!hash(d.reference_revision)||r.status!=='ready'||r.version!=='workbench_train_reference_v1'||r.reference_sha256!==seal||!hash(seal)||r.dictionary_sha256!==c.versions.display_dictionary_sha256||r.release_manifest_sha256!==c.versions.release_manifest_sha256||r.population!=='labeled_train_applications'||!Number.isSafeInteger(r.rows)||r.rows<=0||r.feature_count!==40||d.rows.length!==40||r.percentile_status!=='not_provided_binned_reference_only')invalid();
  d.rows.forEach((x,i)=>{const f=c.profile.facts[i];
   if(x.fact_id!==f.fact_id||x.raw_value!==f.raw_value||x.unit!==f.unit||x.missing_state!==f.missing_state||x.rows!==r.rows||x.percentile!==null||!x.description||x.bins.length<2||x.bins.length>512||!Number.isSafeInteger(x.current_bucket)||x.current_bucket<0||x.current_bucket>=x.bins.length)invalid();
   if(!['missing','unseen_nonnull','unknown_category','rare_category','outside_train_range','observed'].includes(x.position_status)||f.raw_value===null&&x.position_status!=='missing'||f.raw_value!==null&&x.position_status==='missing')invalid();
   x.bins.forEach((b,j)=>{if(b.index!==j||typeof b.label!=='string'||!Number.isSafeInteger(b.count)||b.count<0||!near(b.share,b.count/r.rows)||b.current!==(j===x.current_bucket))invalid()});
   if(x.bins.reduce((a,b)=>a+b.count,0)!==r.rows||x.missing_count!==x.bins[0].count||x.current_bucket_count!==x.bins[x.current_bucket].count||!near(x.current_bucket_share,x.bins[x.current_bucket].share))invalid();
  });
  const top=[...e.terms].sort((a,b)=>Math.abs(b.contribution_log_odds)-Math.abs(a.contribution_log_odds)).slice(0,5),ids=new Set(top.map(t=>t.fact_id)),rest=e.terms.filter(t=>!ids.has(t.fact_id)).reduce((a,t)=>a+t.contribution_log_odds,0);
  if(z.method!==e.method||z.unit!=='log_odds'||z.rest_count!==35||JSON.stringify(z.top)!==JSON.stringify(top)||!near(z.base,e.base_log_odds)||!near(z.rest,rest)||!near(z.total,e.contribution_sum)||!near(z.base+top.reduce((a,t)=>a+t.contribution_log_odds,0)+z.rest,e.raw_score)||!near(z.raw_score,e.raw_score)||!near(z.probability,c.model_outputs.tree.probability))invalid();return d;
 }catch(e){if(e instanceof ApiError)throw e;return invalid()}
}
export async function fetchReference(c:CaseData,token:string,signal:AbortSignal):Promise<ReferenceData>{
 async function request(path:string):Promise<Envelope>{
  for(let attempt=0;attempt<6;attempt++){
   let body:Envelope,response:Response;
   try{response=await fetch(path,{signal,headers:{'X-Selection-Token':token}});body=await response.json() as Envelope}catch(e){if(signal.aborted)throw e;throw new ApiError('CONNECTION_FAILED','无法连接训练参考服务。')}
   if(body.schema_version!=='workbench_api_v2.0'||body.selection_token!==token)invalid();
   if(body.error?.code==='BUSY'&&attempt<5){await new Promise<void>((resolve,reject)=>{if(signal.aborted){reject(new DOMException('Aborted','AbortError'));return}const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'))};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},200*(attempt+1));signal.addEventListener('abort',abort,{once:true})});continue}
   if(!response.ok||body.error)throw new ApiError(body.error?.code||'TRAIN_REFERENCE_UNAVAILABLE',body.error?.message||'训练参考暂不可用。');if(body.status!=='ready')invalid();return body;
  }return invalid();
 }
 const info=(await request('/api/v2/train-reference')).payload as ReferenceData['reference'];if(info.status!=='ready'||!hash(info.reference_sha256))invalid();
 return assertReference((await request(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/reference?expected_revision=${c.case_revision}`)).payload,c,info.reference_sha256);
}
