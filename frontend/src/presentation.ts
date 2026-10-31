import dictionary from '../../src/credit_risk_workbench/presentation_zh.json' with {type:'json'};
import {decimal,metric} from './format.ts';
const d=dictionary as {fields:Record<string,{label:string;source:string;formula:string;time:string;unit:string}>;categories:Record<string,string>;sources:Record<string,string>;units:Record<string,string>;raw:Record<string,string>};
export const field=(name:string)=>d.fields[name];
export const fieldLabel=(name:string)=>d.raw[name]||'其他资料';
export const sourceName=(name:string)=>d.sources[name]||'来源资料';
export const unitName=(name:string)=>d.units[name]||'来源单位';
export function category(value:string):string{return d.categories[value]||(/^[\x20-\x7e]+$/.test(value)?'其他来源类别（含义未提供）':value)}
export function valueText(name:string,v:unknown):string{
 if(v===null||v===undefined)return '未提供';
 if(typeof v==='number')return metric(v);
 return category(String(v));
}
export const stateName=(s:string)=>({observed:'已提供',source_null:'未提供',no_eligible_history:'无符合条件的历史记录',source_null_or_no_eligible_observation:'未提供或无有效记录'}[s]||'待核对');
export function prose(text:string):string{
 let out=text.replaceAll('原40项父资料','申请基本资料').replaceAll('冻结SQL','资料筛选规则').replaceAll('后续更大输入契约','符合容量要求的资料文件');
 const paths:Record<string,string>={contract_version:'资料文件格式',source_kind:'资料来源',monetary_unit:'金额单位',time_reference:'时间参照',snapshot_availability:'资料可用时间',source_reference:'来源说明',application:'申请基本资料',history:'历史资料',bureau:'征信借款资料',previous:'历史申请资料',cc:'信用卡资料',pos:'消费分期资料',bb:'征信月度资料',inst:'分期还款资料',rows:'记录'};
 for(const [a,b] of Object.entries(paths))out=out.replace(new RegExp('\\b'+a+'\\b','g'),b);
 const pairs={...d.categories,...d.sources,...d.raw,...d.units,log_odds:'对数几率',sigmoid:'概率转换',DPD:'逾期天数',null:'未提供',observed:'已提供',application_snapshot:'申请时保存的资料'};
 for(const [a,b] of Object.entries(pairs).sort((a,b)=>b[0].length-a[0].length))out=out.split(a).join(b);
 return out.replace(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g,'资料项').replace(/(?:no_eligible_history|source_null_or_no_eligible_observation)/g,'未提供或无符合条件的记录');
}
export function binText(text:string):string{
 if(text==='null'||/missing|缺失/i.test(text))return '未提供';
 if(d.categories[text])return d.categories[text];
 if(/^[\x20-\x7e]+$/.test(text)&&/[a-zA-Z]/.test(text)&&!/^[-+\d\s.,()[\]<>≤≥∞%]+$/.test(text))return prose(text).replace(/unknown|unseen/gi,'未知类别').replace(/other/gi,'其他类别');
 return prose(text);
}
export const contributionText=(v:number)=>(v>0?'+':'')+decimal(v);
// Read only: describe this application's saved values without inferring extra assets.
export function factValue(name:string,v:unknown,state?:string):string{
 if(v===null||v===undefined){
  if(state==='no_eligible_history')return '无符合条件的记录，无法计算';
  return name==='FLAG_OWN_CAR'?'车辆信息未提供':name==='FLAG_OWN_REALTY'?'房产信息未提供':state==='source_null_or_no_eligible_observation'?'未提供或无可用记录':'资料未提供';
 }
 if(name==='FLAG_OWN_CAR')return v==='Y'?'有车（数量未提供）':v==='N'?'无车':'车辆情况待核对';
 if(name==='FLAG_OWN_REALTY')return v==='Y'?'有房产（数量未提供）':v==='N'?'无房产':'房产情况待核对';
 if(name.startsWith('FLAG_'))return v===1?'有相关记录':v===0?'无相关记录':'标记待核对';
 if(name.endsWith('_HAS_ELIGIBLE_HISTORY'))return v===1?'有符合条件的历史记录':v===0?'无符合条件的历史记录':'历史记录待核对';
 if(typeof v!=='number')return valueText(name,v);
 if(name==='AMT_INCOME_TOTAL')return `${metric(v)}（来源金额；币种及收入周期未提供）`;
 if(name==='HOUR_APPR_PROCESS_START')return `${metric(v)}时`;
 if(name.includes('MONTHS_SINCE_RECENT'))return `申请前${metric(v)}个月`;
 if(name.includes('DAYS_SINCE_'))return `申请前${metric(v)}天`;
 const unit=d.fields[name]?.unit;
 if(unit==='比例')return `${decimal(v*100)}%`;
 if(unit==='天')return `${metric(v)}天`;
 if(unit&&['条','条记录','条借款月记录','条支付记录','笔借款','个分期组','组'].includes(unit))return `${metric(v)}${unit}${state==='no_eligible_history'?'（无符合条件记录）':''}`;
 if(name.startsWith('BUREAU_C1_CREDIT_'))return `${metric(v)}（匿名币种一的来源金额）`;
 return metric(v);
}
export const modelDirection=(n:number)=>n===0?'本项未改变模型评分':n>0?'推高模型评分':'降低模型评分';
export function categoryOptions(name:string):{value:string;label:string}[]{
 let keys:string[]=[];
 if(name==='NAME_CONTRACT_TYPE')keys=['Cash loans','Consumer loans','Revolving loans'];
 if(name==='CREDIT_ACTIVE')keys=['Active','Closed','Sold','Bad debt'];
 if(name==='CREDIT_CURRENCY')keys=['currency 1','currency 2','currency 3','currency 4'];
 if(name==='NAME_INCOME_TYPE')keys=['Working','Commercial associate','Pensioner','State servant','Unemployed','Student','Businessman','Maternity leave'];
 if(name==='NAME_HOUSING_TYPE')keys=['House / apartment','Rented apartment','With parents','Municipal apartment','Office apartment','Co-op apartment'];
 if(name==='OCCUPATION_TYPE')keys=Object.keys(d.categories).slice(0,18);
 if(name==='ORGANIZATION_TYPE')keys=Object.keys(d.categories).filter(k=>k.startsWith('Business Entity')||k==='Self-employed'||Object.keys(d.categories).indexOf(k)>=Object.keys(d.categories).indexOf('School')&&!k.startsWith('currency'));
 if(name==='WEEKDAY_APPR_PROCESS_START')keys=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY'];
 if(name.startsWith('FLAG_'))return (['FLAG_OWN_CAR','FLAG_OWN_REALTY'].includes(name)?['Y','N']:['1','0']).map(value=>({value,label:['Y','1'].includes(value)?'是':'否'}));
 if(name==='STATUS')return ['0','1','2','3','4','5','C','X'].map(value=>({value,label:value==='C'?'已关闭':value==='X'?'状态未知':value==='0'?'未逾期':'逾期等级'+value}));
 return keys.map(value=>({value,label:category(value)}));
}
