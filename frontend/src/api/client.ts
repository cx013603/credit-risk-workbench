import type {Identity,CaseData,CaseState,Envelope} from './types';
export class ApiError extends Error {code:string;constructor(code:string,message:string){super(message);this.code=code}}
const fail=()=>{throw new ApiError('RESPONSE_INVALID','返回资料未通过身份或格式核对，请重新查询。')};
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const fraction=(v:unknown):v is number=>finite(v)&&v>=0&&v<=1;
export function parseIdentity(population:string,id:string):Identity{
 if(!['labeled','unlabeled'].includes(population)||!/^\d{1,10}$/.test(id)||!Number.isSafeInteger(Number(id))||Number(id)<1)throw new ApiError('INVALID_INPUT','请输入有效的正整数申请编号。');
 return {population:population as Identity['population'],application_id:Number(id)};
}
export function assertCase(value:unknown,identity:Identity):CaseData{
 try{return validateCase(value,identity)}catch(e){if(e instanceof ApiError)throw e;return fail()}
}
function validateCase(value:unknown,identity:Identity):CaseData{
 const c=value as CaseData;
 if(!c||c.case_key?.population!==identity.population||c.case_key?.application_id!==identity.application_id||!/^([a-f0-9]{64})$/.test(c.case_revision||''))fail();
 if(!['train','validation','unlabeled_scoring'].includes(c.sample_use)||(identity.population==='unlabeled')!==(c.sample_use==='unlabeled_scoring'))fail();
 const f=c.profile?.facts,e=c.explanation,m=c.model_outputs;
 if(c.profile?.scope!=='saved_40_model_inputs_only'||!Array.isArray(f)||f.length!==40||new Set(f.map(x=>x.fact_id)).size!==40)fail();
 if(!e||e.method!=='training_count_weighted_path_log_odds'||e.unit!=='log_odds'||e.feature_count!==40||!Array.isArray(e.terms)||e.terms.length!==40)fail();
 if(f.some((x,i)=>x.fact_id!==e.terms[i].fact_id||typeof x.label!=='string'||!x.source?.table||!x.rule?.formula||!['observed','source_null','no_eligible_history'].includes(x.missing_state)||!(x.raw_value===null||typeof x.raw_value==='string'||finite(x.raw_value))))fail();
 if(![e.base_log_odds,e.raw_score,e.contribution_sum,...e.terms.map(t=>t.contribution_log_odds)].every(finite))fail();
 if(!m||m.recommended!=='tree'||m.tree?.version!=='tree_v1'||m.scorecard?.score_unit!=='scorecard_points'||![m.tree.probability,m.tree.threshold,m.logistic?.probability,m.scorecard?.probability].every(fraction)||!finite(m.scorecard.score)||typeof m.tree.risk_flag!=='boolean')fail();
 if(m.tree.risk_flag!==(m.tree.probability>=m.tree.threshold))fail();
 const sum=e.terms.reduce((a,t)=>a+t.contribution_log_odds,0);
 if(Math.abs(sum-e.contribution_sum)>1e-10||Math.abs(e.base_log_odds+sum-e.raw_score)>1e-10||Math.abs(1/(1+Math.exp(-e.raw_score))-m.tree.probability)>1e-10)fail();
 if(c.versions?.feature_version!=='application_contract_v1'||c.versions?.tree_version!=='tree_v1'||!/^([a-f0-9]{64})$/.test(c.versions.release_manifest_sha256||'')||!/^([a-f0-9]{64})$/.test(c.versions.display_dictionary_sha256||''))fail();
 return c;
}
function delay(ms:number,signal:AbortSignal){return new Promise<void>((resolve,reject)=>{if(signal.aborted){reject(new DOMException('Aborted','AbortError'));return}const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},ms);function abort(){clearTimeout(timer);reject(new DOMException('Aborted','AbortError'))}signal.addEventListener('abort',abort,{once:true})})}
export async function fetchCase(identity:Identity,token:string,signal:AbortSignal):Promise<CaseData>{
 for(let attempt=0;attempt<5;attempt++){
  let response:Response;try{response=await fetch(`/api/v2/cases/${identity.population}/${identity.application_id}`,{signal,headers:{'X-Selection-Token':token}})}catch(e){if(signal.aborted)throw e;throw new ApiError('CONNECTION_FAILED','无法连接本机分析服务，请检查服务后重新查询。')}
  let body:Envelope;try{body=await response.json() as Envelope}catch{throw new ApiError('RESPONSE_INVALID','服务未返回有效资料，请重新查询。')}
  if(!body||typeof body!=='object')fail();
  if(body.schema_version!=='workbench_api_v2.0'||body.selection_token!==token)fail();
  if(!response.ok||body.error){if(body.error?.code==='BUSY'&&attempt<4){await delay(200*(attempt+1),signal);continue}throw new ApiError(body.error?.code||'REQUEST_FAILED',body.error?.message||'当前申请暂不可用。')}
  if(body.status!=='ready'||typeof body.request_id!=='string')fail();
  return assertCase(body.payload,identity);
 }
 throw new ApiError('BUSY','评分进程仍在处理其他请求，请稍后重试。');
}
type RequestCase=(identity:Identity,token:string,signal:AbortSignal)=>Promise<CaseData>;
export class CaseCoordinator{
 private sequence=0;private active:AbortController|null=null;private notify:(state:CaseState)=>void;private request:RequestCase;
 constructor(notify:(state:CaseState)=>void,request:RequestCase=fetchCase){this.notify=notify;this.request=request}
 cancel(notify=true){this.sequence++;this.active?.abort();this.active=null;if(notify)this.notify({status:'cancelled',identity:null,data:null,error:null,token:String(this.sequence)})}
 reset(){this.cancel(false);this.notify({status:'idle',identity:null,data:null,error:null,token:String(this.sequence)})}
 async load(identity:Identity){this.cancel(false);const sequence=this.sequence,token=`selection-${sequence}`,controller=new AbortController();this.active=controller;this.notify({status:'loading',identity,data:null,error:null,token});
  try{const data=assertCase(await this.request(identity,token,controller.signal),identity);if(sequence===this.sequence&&!controller.signal.aborted)this.notify({status:'ready',identity,data,error:null,token})}
  catch(e){if(sequence!==this.sequence||controller.signal.aborted)return;this.notify({status:'failed',identity,data:null,error:e instanceof Error?e.message:'未能读取当前申请，请检查服务后重试。',errorCode:e instanceof ApiError?e.code:'CONNECTION_FAILED',token})}
 }
}
