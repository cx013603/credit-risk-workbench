import evidence from '../../src/credit_risk_workbench/model_reading71.json' with {type:'json'};
type Datum={name:string;value:unknown;missing_state:string};
export function coverage(facts:Datum[]){
 const unavailable=facts.filter(f=>(f.value===null||f.value===undefined)&&f.missing_state!=='no_eligible_history');
 const noHistory=facts.filter(f=>f.missing_state==='no_eligible_history');
 return {observed:facts.length-unavailable.length-noHistory.length,unavailable,noHistory};
}
export function validationReading(modelSha:string,threshold:number){
 if(evidence.model_sha256!==modelSha||evidence.threshold!==threshold)return null;
 const {TP,FP,TN,FN}=evidence.confusion;
 return {...evidence,recall:TP/(TP+FN),falsePositive:FP/(FP+TN),precision:TP/(TP+FP)};
}
