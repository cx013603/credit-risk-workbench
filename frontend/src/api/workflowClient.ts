import type {CaseData,Envelope,Identity} from './types';
import {ApiError} from './client.ts';
export const statuses={unreviewed:'未复核',pending:'待复核',needs_information:'资料待补',reviewed:'已复核'};
export type ReviewStatus=keyof typeof statuses;
export type Draft={favorite:boolean;review_status:ReviewStatus;note:string};
export type ReviewRecord=Draft&{case_key:Identity;case_revision:string;marks_revision:number;exists:boolean;pending:boolean;deleted:boolean;updated_at:string|null;saved_case_revision:string;binding_matches:boolean;versions:Record<string,string>;sample_use:string;scope:string};
export type ReviewList={items:{population:Identity['population'];application_id:number;favorite:boolean;pending:boolean;review_status:ReviewStatus;updated_at:string}[];total:number;offset:number;limit:number;deleted:boolean;marks_revision:number;release_manifest_sha256:string};
export type ReportReceipt={report_id:string;case_key:Identity;case_revision:string;marks_revision:number;created_at:string;json_sha256:string;html_sha256:string;view_url:string;download_url:string;data_url:string};
export type ReportList={items:{report_id:string;population:Identity['population'];application_id:number;created_at:string;marks_revision:number}[];limit:number;release_manifest_sha256:string};
const int=(n:unknown):n is number=>typeof n==='number'&&Number.isSafeInteger(n)&&n>=0;
const hash=(s:unknown)=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);
const date=(s:unknown)=>typeof s==='string'&&Number.isFinite(Date.parse(s));
const fail=():never=>{throw new ApiError('WORKFLOW_INVALID','模拟分析记录未通过案例、版本或内容核对。')};
export function draftOf(r:ReviewRecord):Draft{return {favorite:r.favorite,review_status:r.review_status,note:r.note}}
export function dirty(d:Draft,r:ReviewRecord){return d.favorite!==r.favorite||d.review_status!==r.review_status||d.note!==r.note}
export function assertReview(value:unknown,c:CaseData):ReviewRecord{
 try{const r=value as ReviewRecord;
  if(r.case_key.population!==c.case_key.population||r.case_key.application_id!==c.case_key.application_id||r.case_revision!==c.case_revision||!int(r.marks_revision)||typeof r.exists!=='boolean'||typeof r.deleted!=='boolean'||typeof r.favorite!=='boolean'||typeof r.pending!=='boolean'||!Object.hasOwn(statuses,r.review_status)||typeof r.note!=='string'||[...r.note].length>3000||/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(r.note)||!hash(r.saved_case_revision)||r.binding_matches!==(r.saved_case_revision===c.case_revision)||r.scope!=='local_simulated_analysis_only'||r.updated_at!==null&&!date(r.updated_at))fail();
  if(r.pending!==(!r.deleted&&['pending','needs_information'].includes(r.review_status))||r.binding_matches&&(r.sample_use!==c.sample_use||JSON.stringify(Object.entries(r.versions).sort())!==JSON.stringify(Object.entries(c.versions).sort())))fail();return r;
 }catch(e){if(e instanceof ApiError)throw e;return fail()}
}
export async function workflowRequest(path:string,token:string,signal:AbortSignal,body?:unknown):Promise<unknown>{
 for(let attempt=0;attempt<5;attempt++){
  let response:Response,v:Envelope;
  try{response=await fetch(path,{signal,method:body===undefined?'GET':'POST',headers:{'X-Selection-Token':token,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});v=await response.json() as Envelope}catch(e){if(signal.aborted)throw e;throw new ApiError('CONNECTION_FAILED','请求未确认完成；编辑内容已保留。请重新读取记录核对后再操作。')}
  if(v.schema_version!=='workbench_api_v2.0'||v.selection_token!==token)fail();
  if(v.error?.code==='BUSY'&&attempt<4){await new Promise<void>((resolve,reject)=>{const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'))};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve()},200*(attempt+1));if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true})});continue}
  if(!response.ok||v.error){const code=v.error?.code||'WORKFLOW_FAILED',messages:Record<string,string>={VERSION_MISMATCH:'记录版本已更新。请重新读取，对照已保存备注后再保存；当前编辑内容保留。',REVIEW_DELETED:'记录在回收站中，请先恢复。',LOCAL_STORAGE_UNAVAILABLE:'本机存储暂不可用，未确认保存成功；当前编辑内容保留。',REPORT_QUOTA:'报告存储达到上限，请先按说明备份整理。',REPORT_CORRUPT:'报告文件校验失败，不能作为完整报告打开。'};throw new ApiError(code,messages[code]||v.error?.message||'模拟分析操作暂未完成。')}
  if(v.status!=='ready')fail();return v.payload;
 }return fail();
}
export async function fetchReview(c:CaseData,token:string,signal:AbortSignal){return assertReview(await workflowRequest(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/review?expected_revision=${c.case_revision}`,token,signal),c)}
export async function saveReview(c:CaseData,r:ReviewRecord,draft:Draft,token:string,signal:AbortSignal){return assertReview(await workflowRequest(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/review`,token,signal,{expected_case_revision:c.case_revision,expected_marks_revision:r.marks_revision,...draft}),c)}
export async function reviewAction(c:CaseData,r:ReviewRecord,action:'delete'|'restore',token:string,signal:AbortSignal){return assertReview(await workflowRequest(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/review/${action}`,token,signal,{expected_case_revision:c.case_revision,expected_marks_revision:r.marks_revision}),c)}
export async function fetchReviewList(deleted:boolean,offset:number,token:string,signal:AbortSignal):Promise<ReviewList>{
 const d=await workflowRequest(`/api/v2/reviews?deleted=${deleted}&offset=${offset}&limit=10`,token,signal) as ReviewList;
 if(d.deleted!==deleted||d.offset!==offset||d.limit!==10||!int(d.total)||!int(d.marks_revision)||!hash(d.release_manifest_sha256)||!Array.isArray(d.items)||d.items.length!==Math.min(10,Math.max(0,d.total-offset))||d.items.some(i=>!['labeled','unlabeled'].includes(i.population)||!int(i.application_id)||!date(i.updated_at)||!Object.hasOwn(statuses,i.review_status)||typeof i.favorite!=='boolean'||typeof i.pending!=='boolean'))fail();return d;
}
export function assertReceipt(value:unknown,c:CaseData,r:ReviewRecord):ReportReceipt{
 const d=value as ReportReceipt;if(!d||!/^[a-f0-9]{32}$/.test(d.report_id)||d.case_key?.population!==c.case_key.population||d.case_key?.application_id!==c.case_key.application_id||d.case_revision!==c.case_revision||d.marks_revision!==r.marks_revision||!date(d.created_at)||!hash(d.json_sha256)||!hash(d.html_sha256)||d.view_url!==`/api/v2/reports/${d.report_id}/view`||d.download_url!==`/api/v2/reports/${d.report_id}/download`||d.data_url!==`/api/v2/reports/${d.report_id}/data`)fail();return d;
}
export async function createReport(c:CaseData,r:ReviewRecord,token:string,signal:AbortSignal){return assertReceipt(await workflowRequest(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/report`,token,signal,{expected_case_revision:c.case_revision,expected_marks_revision:r.marks_revision,selection:{kind:'single'}}),c,r)}
export async function fetchReports(token:string,signal:AbortSignal):Promise<ReportList>{
 const d=await workflowRequest('/api/v2/reports?limit=10',token,signal) as ReportList;if(d.limit!==10||!hash(d.release_manifest_sha256)||!Array.isArray(d.items)||d.items.length>10||d.items.some(i=>!/^[a-f0-9]{32}$/.test(i.report_id)||!['labeled','unlabeled'].includes(i.population)||!int(i.application_id)||!int(i.marks_revision)||!date(i.created_at)))fail();return d;
}
