import type {CaseData,Envelope} from './types';
import {ApiError} from './client.ts';
export type NarrativeRow={fact_id:string;label:string;raw_value:string|number|null;unit:string;unit_zh:string;missing_state:string;missing_text:string;definition_zh:string;formula:string;time_zh:string;fact_text:string;model_response_text:string;review_text:string;contribution_log_odds:number;direction:'positive'|'negative'|'zero';evidence_text:string;evidence:{kind:'history_index'|'history_unavailable'|'saved_model_input_only';verified_source_records:boolean;source_table:string;record_key_kind:string;application_id:number;rule_version:string;saved_fact_id:string;index_manifest_sha256?:string;matching_record_count?:number};limits:string[]};
export type Narratives={case_key:CaseData['case_key'];case_revision:string;sample_use:string;version:string;method:string;template_sha256:string;dictionary_sha256:string;history_manifest_sha256:string|null;narrative_revision:string;feature_count:number;rows:NarrativeRow[]};
const hash=(v:unknown):v is string=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const invalid=():never=>{throw new ApiError('NARRATIVE_RESPONSE_INVALID','中文解释未通过案例、原值或版本核对，请重新读取。')};
export function assertNarratives(value:unknown,c:CaseData,templateHash:string):Narratives{
 try{
  const n=value as Narratives;
  if(!n||n.case_key.population!==c.case_key.population||n.case_key.application_id!==c.case_key.application_id||n.case_revision!==c.case_revision||n.sample_use!==c.sample_use||n.version!=='workbench_narratives_zh_v1'||n.method!=='deterministic_fact_model_review_v1'||n.template_sha256!==templateHash||!hash(n.template_sha256)||n.dictionary_sha256!==c.versions.display_dictionary_sha256||!hash(n.narrative_revision)||n.history_manifest_sha256!==null&&!hash(n.history_manifest_sha256)||n.feature_count!==40||n.rows.length!==40)invalid();
  n.rows.forEach((r,i)=>{const f=c.profile.facts[i],t=c.explanation.terms[i],e=r.evidence;
   if(r.fact_id!==f.fact_id||r.label!==f.label||r.raw_value!==f.raw_value||r.unit!==f.unit||r.missing_state!==f.missing_state||r.formula!==f.rule.formula||r.contribution_log_odds!==t.contribution_log_odds||r.direction!==(r.contribution_log_odds>0?'positive':r.contribution_log_odds<0?'negative':'zero'))invalid();
   if([r.fact_text,r.definition_zh,r.model_response_text,r.review_text,r.evidence_text,r.unit_zh,r.time_zh,r.missing_text].some(x=>typeof x!=='string'||!x.trim())||!r.model_response_text.includes('对数几率')||r.model_response_text.includes('百分点')||!Array.isArray(r.limits))invalid();
   if(e.application_id!==c.case_key.application_id||e.saved_fact_id!==f.fact_id||e.source_table!==f.source.table||e.record_key_kind!==f.source.record_key_kind||e.rule_version!==f.rule.version)invalid();
   if(f.value_kind==='source_field'){if(e.kind!=='saved_model_input_only'||e.verified_source_records)invalid()}
   else if(n.history_manifest_sha256){if(e.kind!=='history_index'||e.verified_source_records!==true||e.index_manifest_sha256!==n.history_manifest_sha256||!Number.isSafeInteger(e.matching_record_count)||e.matching_record_count!<0)invalid()}
   else if(e.kind!=='history_unavailable'||e.verified_source_records)invalid();
  });return n;
 }catch(e){if(e instanceof ApiError)throw e;return invalid()}
}
export async function fetchNarratives(c:CaseData,token:string,signal:AbortSignal):Promise<Narratives>{
 async function request(url:string):Promise<Envelope>{
  for(let attempt=0;attempt<5;attempt++){
   let response:Response;try{response=await fetch(url,{signal,headers:{'X-Selection-Token':token}})}catch(e){if(signal.aborted)throw e;throw new ApiError('CONNECTION_FAILED','无法连接中文解释服务。')}
   let body:Envelope;try{body=await response.json() as Envelope}catch{return invalid()}
   if(body?.schema_version!=='workbench_api_v2.0'||body.selection_token!==token)invalid();
   if(body.error?.code==='BUSY'&&attempt<4){await new Promise<void>((resolve,reject)=>{if(signal.aborted){reject(new DOMException('Aborted','AbortError'));return}const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'))};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},200*(attempt+1));signal.addEventListener('abort',abort,{once:true})});continue}
   if(!response.ok||body.error)throw new ApiError(body.error?.code||'NARRATIVE_UNAVAILABLE',body.error?.message||'中文解释暂不可用。');
   if(body.status!=='ready')invalid();return body;
  }return invalid();
 }
 const info=(await request('/api/v2/narrative-templates')).payload as {status:string;template_sha256:string;dictionary_sha256:string};
 if(info?.status!=='ready'||!hash(info.template_sha256)||info.dictionary_sha256!==c.versions.display_dictionary_sha256)invalid();
 const body=await request(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/narratives?expected_revision=${c.case_revision}`);
 return assertNarratives(body.payload,c,info.template_sha256);
}
