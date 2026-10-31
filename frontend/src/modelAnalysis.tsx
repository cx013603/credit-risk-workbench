import {useState} from 'react';
import type {CaseData} from './api/types';
import type {NarrativeState} from './narratives';
import {AllFactors,FactorPanel,Models} from './panels';
import {ContributionWaterfall} from './references';
export function ModelAnalysis({c,narratives,onFact}:{c:CaseData;narratives:NarrativeState;onFact:(id:string)=>void}){
 const [view,setView]=useState('contributions');
 return <section data-testid="model-analysis"><div className="model-subtabs" aria-label="模型解释分区"><button className={view==='contributions'?'active':''} onClick={()=>setView('contributions')}>贡献拆解</button><button className={view==='models'?'active':''} onClick={()=>setView('models')}>三模型对照</button></div>{view==='models'?<Models c={c}/>:<><FactorPanel c={c} onFact={onFact} onAll={()=>document.querySelector('[data-testid="all-factors"]')?.scrollIntoView({block:'start'})}/><ContributionWaterfall c={c} onFact={onFact} onAll={()=>document.querySelector('[data-testid="all-factors"]')?.scrollIntoView({block:'start'})}/><AllFactors c={c} narrativeState={narratives} onFact={onFact}/></>}</section>;
}
