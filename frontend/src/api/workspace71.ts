export type Bin={label:string;count:number;share:number;current:boolean};
export type Fact={name:string;label:string;unit:string;source:string;formula:string;time_rule:string;missing_policy:string;value:string|number|null;missing_state:string;fact:string;contribution_log_odds:number;reference:{rows:number;missing_count:number;bins:Bin[];current_bucket:number}};
export type Result={version:string;case_key:{population:string;application_id:number};case_revision:string;sample_use:string;model_version:string;model_sha256:string;deployment_sha256:string;input_contract_version:string;feature_count:number;probability:number;threshold:number;risk_flag:boolean;facts:Fact[];explanation:{method:string;unit:string;base_log_odds:number;contribution_sum:number;raw_score:number;checks:Record<string,number>};limits:string[]};
export type Info={version:string;model_sha256:string;deployment_sha256:string;feature_count:number;threshold:number;template:Record<string,unknown>;fields:{name:string;label:string}[];raw_history_columns:Record<string,string[]>;examples?:number[]};
export type Draft={draft_id:string;revision:number;title:string;payload:Record<string,unknown>;payload_sha256:string;validation:{ready_to_score:boolean;issues:{path:string;message:string}[];warnings?:{path:string;message:string}[];warning_count?:number;warnings_truncated?:boolean};score:Result|null;score_stale:boolean};
export type Item={draft_id:string;title:string;revision:number;has_score:boolean};
export const prefix='/api/v2/candidate71';



export async function request(path:string,signal:AbortSignal,body?:unknown){
 const token='candidate71-'+Date.now()+'-'+Math.random().toString(16).slice(2);
 for(let attempt=0;attempt<6;attempt++){
  const response=await fetch(prefix+path,{signal,method:body===undefined?'GET':'POST',headers:{'X-Selection-Token':token,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const d=await response.json();if(d.schema_version!=='workbench_api_v2.0'||d.selection_token!==token)throw Error('响应身份不一致，请重新读取。');
  if(d.error?.code==='BUSY'&&attempt<5){await new Promise<void>((resolve,reject)=>{const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'))};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},150*(attempt+1));signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort()});continue}
  if(!response.ok||d.error)throw Error(({INVALID_DRAFT_ID:'申请记录编号无效。',DRAFT_NOT_FOUND:'未找到这份申请资料，请返回申请列表。',DRAFT_QUOTA:'已保存申请数量达到本机上限，请整理记录后重试。',DRAFT_SCHEMA_INVALID:'申请资料存储格式不兼容，请检查本机数据。',CASE_NOT_FOUND:'此编号不在当前可查询范围，请核对申请来源与编号。',VERSION_MISMATCH:'资料或分析版本已更新，请重新读取申请资料。',INPUT_INCOMPLETE:'资料不完整，先校验并补齐；未提供历史不能当无历史。',INVALID_CANDIDATE71_HISTORY:'原始历史存在重复月键或冲突支付记录，请核对后再评分。',SAMPLE_USE_NOT_ALLOWED:'该编号不在当前查询范围，请选择其他申请。',INPUT_SIZE:'资料超过上限，不会截断后评分。'} as Record<string,string>)[d.error?.code]||d.error?.message||'本次请求未完成。');
  return d.payload;
 }
 throw Error('服务繁忙，请稍后重试。');
}

export function checked(d:Result,info:Info,expected?:{population:string;application_id:number}){
 if(expected&&(d.case_key.population!==expected.population||d.case_key.application_id!==expected.application_id))throw Error('结果对应其他申请，请重新查询。');
 if(d.version!=='candidate71_local_v1'||d.model_sha256!==info.model_sha256||d.deployment_sha256!==info.deployment_sha256||d.feature_count!==71||d.facts.length!==71||new Set(d.facts.map(f=>f.name)).size!==71||d.facts.some((f,i)=>f.name!==info.fields[i].name)||!Number.isFinite(d.probability)||d.probability<0||d.probability>1||d.threshold!==info.threshold||d.risk_flag!==(d.probability>=d.threshold))throw Error('71项结果、模型或字段绑定不一致。');
 const total=d.facts.reduce((s,f)=>s+f.contribution_log_odds,0),raw=d.explanation.base_log_odds+total,p=1/(1+Math.exp(-raw));
 if(!d.facts.every(f=>Number.isFinite(f.contribution_log_odds))||Math.abs(total-d.explanation.contribution_sum)>1e-10||Math.abs(raw-d.explanation.raw_score)>1e-10||Math.abs(p-d.probability)>1e-10)throw Error('贡献合计与71项模型概率不一致。');
 return d;
}

export async function case71(population:string,application_id:number,info:Info,signal:AbortSignal){if(!['labeled','unlabeled'].includes(population)||!Number.isSafeInteger(application_id)||application_id<=0)throw Error('申请来源或编号无效。');return checked(await request(`/cases/${population}/${application_id}`,signal),info,{population,application_id})}
export function subjectPath(c:Result,draftId?:string){if(c.case_key.population==='new_application71'){if(!draftId||!/^[a-f0-9]{32}$/.test(draftId))throw Error('请重新打开对应申请资料。');return '/drafts/'+draftId}if(!['labeled','unlabeled'].includes(c.case_key.population)||!Number.isSafeInteger(c.case_key.application_id)||c.case_key.application_id<=0)throw Error('分析来源无效。');return `/cases/${c.case_key.population}/${c.case_key.application_id}`}
export const useLabel=(u:string)=>({train:'训练样本',validation:'验证样本',unlabeled_scoring:'无标签，仅预测',new_input_prediction_only:'新资料，仅预测'}[u]||'来源待核对');
