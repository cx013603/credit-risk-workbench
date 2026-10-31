import {factValue,fieldLabel} from './presentation';
type Datum={name:string;value:unknown;state?:string};
const primary=['FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_CONTRACT_TYPE','NAME_INCOME_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE'];
const contacts=['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL'];
export function ApplicantFacts({facts}:{facts:Datum[]}){
 const byName=new Map(facts.map(f=>[f.name,f]));
 const show=(id:string)=>{const f=byName.get(id);return f?factValue(id,f.value,f.state):'资料未提供'};
 return <section className="panel applicant-facts" data-testid="applicant-facts"><h2>申请资料一览</h2><dl className="applicant-fact-grid">{primary.map(id=><div key={id}><dt>{id==='FLAG_OWN_CAR'?'车辆情况':id==='FLAG_OWN_REALTY'?'房产情况':fieldLabel(id)}</dt><dd data-fact-name={id}>{show(id)}</dd></div>)}</dl><details className="secondary-facts"><summary>申请时段与联系方式</summary><p>申请记录：{show('WEEKDAY_APPR_PROCESS_START')}，{show('HOUR_APPR_PROCESS_START')}。具体日历日期未提供。</p><div className="applicant-contacts">{contacts.map(id=><span key={id}>{fieldLabel(id)}：{show(id)}</span>)}</div><p className="fine">联系方式标记仅反映申请时保存的记录，不代表已验证当前可联系。</p></details></section>
}
