import {field,fieldLabel,sourceName,unitName,valueText,stateName,prose,binText} from './presentation';
import {decimalText} from './format';
import {useEffect,useRef,useState} from 'react';
import {Alert,Button,Skeleton} from 'antd';
import type {CaseData} from './api/types';
import type {Narratives,NarrativeRow} from './api/narrativeClient';
import {fetchNarratives} from './api/narrativeClient';

export type NarrativeState={status:'idle'|'loading'|'ready'|'failed';data:Narratives|null;error:string|null;retry:()=>void};
export function useNarratives(c:CaseData|null):NarrativeState{
 const [attempt,setAttempt]=useState(0),[state,setState]=useState<{key:string;data:Narratives|null;error:string|null}>({key:'',data:null,error:null});
 const sequence=useRef(0);const key=c?`${c.case_key.population}:${c.case_key.application_id}:${c.case_revision}:${attempt}`:'';
 useEffect(()=>{const controller=new AbortController(),current=++sequence.current;setState({key,data:null,error:null});
  if(c)void fetchNarratives(c,`narrative-${current}`,controller.signal).then(data=>{if(!controller.signal.aborted&&current===sequence.current)setState({key,data,error:null})}).catch(e=>{if(!controller.signal.aborted&&current===sequence.current)setState({key,data:null,error:e instanceof Error?e.message:'中文解释暂不可用。'})});
  return()=>controller.abort();
 },[key,c]);
 const active=state.key===key?state:{key,data:null,error:null};
 return {status:!c?'idle':active.error?'failed':active.data?'ready':'loading',data:active.data,error:active.error,retry:()=>setAttempt(x=>x+1)};
}

export function NarrativeExplanation({state,factId}:{state:NarrativeState;factId:string}){
 const row=state.data?.rows.find(r=>r.fact_id===factId);
 if(state.status==='failed')return <section className="controlled-narrative" data-testid="narrative-error"><Alert showIcon type="error" title="中文解释未能载入" description={state.error}/><Button onClick={state.retry}>重新读取解释</Button></section>;
 if(!row)return <section className="controlled-narrative" aria-busy="true"><p>正在核对当前案例的中文解释…</p><Skeleton active paragraph={{rows:3}}/></section>;
 return <section className="controlled-narrative" data-testid="controlled-narrative" data-fact-id={row.fact_id} data-case-key={`${state.data!.case_key.population}:${state.data!.case_key.application_id}`} data-narrative-revision={state.data!.narrative_revision}>
  <h3>这项信息怎样理解</h3><p className="narrative-definition">{decimalText(prose(row.definition_zh))}</p>
  <div className="narrative-block"><span>事实</span><p data-testid="narrative-fact">{decimalText(prose(row.fact_text))}</p><small>{decimalText(prose(row.missing_text))}</small></div>
  <div className="narrative-block"><span>模型响应</span><p data-testid="narrative-response">{decimalText(prose(row.model_response_text))}</p></div>
  <div className="narrative-block"><span>需要复查</span><p data-testid="narrative-review">{decimalText(prose(row.review_text))}</p></div>
  <div className="narrative-evidence"><strong>{row.evidence.kind==='history_index'?'来源证据已核对':row.evidence.kind==='saved_model_input_only'?'当前证据为保存输入':'历史来源尚未核对'}</strong><p data-testid="narrative-evidence">{decimalText(prose(row.evidence_text))}</p><p>单位：{prose(row.unit_zh)}<br/>时间：{prose(row.time_zh)}</p></div>
  
 </section>;
}

export function NarrativeSummary({state,factId}:{state:NarrativeState;factId:string}){
 const row=state.data?.rows.find(r=>r.fact_id===factId);
 return <p data-testid="narrative-summary">{row?decimalText(prose(row.fact_text+' '+row.model_response_text)):state.status==='failed'?'本例中文解释未能载入，可在因素详情重试。':'正在核对本例的中文证据说明…'}</p>;
}

export const rowFor=(state:NarrativeState,id:string):NarrativeRow|undefined=>state.data?.rows.find(r=>r.fact_id===id);
