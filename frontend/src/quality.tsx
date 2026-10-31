import {field,fieldLabel,sourceName,unitName,valueText,stateName,prose,binText} from './presentation';
import {useState} from 'react';
import {Select} from 'antd';
import type {CaseData} from './api/types';
import type {ReferenceData} from './api/referenceClient';
import {qualityOf} from './api/collectionClient';
import {Panel,HistoryCoverage,display} from './panels';
import {decimal} from './format';
const units:Record<string,string>={category:'类别',binary_category_Y_N:'是／否类别',binary:'0／1标记',records:'条',days:'天',fraction:'比例',source_amount_currency_and_period_unverified:'来源金额单位；币种／周期待核对',anonymous_currency_1_source_scale:'匿名币种1来源单位',hour:'小时',hour_0_23:'小时（0～23）'};
const states:Record<string,string>={observed:'已观测',source_null:'输入缺失',no_eligible_history:'无合格历史'};
export function DataQuality({c,historyReady,reference,onFact,onHistory}:{c:CaseData;historyReady:boolean;reference:ReferenceData|null;onFact:(id:string)=>void;onHistory:()=>void}){
 const [filter,setFilter]=useState('all');const q=qualityOf(c);
 const validReference=reference?.case_revision===c.case_revision?reference:null;
 const counts=Object.fromEntries(c.profile.facts.map(f=>[f.fact_id,f.raw_value]));
 function signal(id:string){const v=counts[id];if(['BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT','BUREAU_CURRENCY_MISSING_COUNT','BUREAU_OTHER_STATUS_COUNT','PREV_OTHER_CONTRACT_COUNT'].includes(id)&&typeof v==='number'&&v>0)return '存在不可用、其他或缺失记录';return ''}
 function referenceHint(id:string){const r=validReference?.rows.find(x=>x.fact_id===id);return r&&['outside_train_range','unknown_category','unseen_nonnull','rare_category'].includes(r.position_status)?r.description:''}
 const fields=c.profile.facts.filter(f=>filter==='all'||filter==='signals'?(filter==='all'||!!signal(f.fact_id)||!!referenceHint(f.fact_id)):f.missing_state===filter);
 return <section data-testid="data-quality"><Panel><div className="panel-heading"><div><h2>这笔申请的资料质量</h2><p>只检查已接入的40项输入，未填补、不删值，也不修改保存预测。</p></div></div><div className="quality-metrics"><div><span>资料完整度</span><strong>{decimal(q.observed/40*100)}%</strong><small>{q.observed}／40项有观测值</small></div><div><span>输入缺失</span><strong>{q.missing}</strong><small>有资料来源，但字段为空</small></div><div><span>无合格历史字段</span><strong>{q.noHistory}</strong><small>含结构性0及对应缺失值</small></div></div><p className="fine">完整度仅描述资料，不能代表预测置信度。不可用金额、其他／缺失状态和未知类别保留原样，提示核对；训练范围外不自动判定错误。</p></Panel>
 <HistoryCoverage c={c} historyReady={historyReady} onHistory={onHistory}/>
 <Panel><h2>来源、单位与时间边界</h2><ul className="quality-boundaries"><li>申请输入来自申请基本资料的16项已保存投影；其他主表资料尚未接入。</li><li>征信历史来自征信借款记录；仅借款日与更新日均早于申请日的记录进入原聚合。</li><li>历史申请来自历史贷款申请；仅决策相对日小于0的记录进入原聚合。</li><li>金额沿用匿名来源单位，未核验为人民币；绝对申请日期、自然人身份及标签期限未知。</li><li>{historyReady?'历史来源索引已连接，可在字段详情核对来源记录。':'历史索引未连接；当前只有保存输入，不能声称已核对来源记录。'}{validReference?'训练参考已连接，范围／类别提示如下。':'训练参考尚未可用，未据此判定范围／类别异常。'}</li></ul></Panel>
 <Panel><div className="catalog-toolbar"><h2>逐项资料检查 · 40项</h2><label>显示状态<Select aria-label="资料字段状态" value={filter} onChange={setFilter} options={[{value:'all',label:'全部字段'},{value:'source_null',label:'输入缺失'},{value:'no_eligible_history',label:'无合格历史'},{value:'observed',label:'有观测值'},{value:'signals',label:'编码／参考待核对'}]}/></label></div><div className="table-scroll" tabIndex={0}><table data-testid="quality-table"><thead><tr><th>字段</th><th>保存值</th><th>资料状态</th><th>来源</th><th>单位</th><th>核对提示</th></tr></thead><tbody>{fields.map(f=><tr key={f.fact_id}><th><button className="text-button" onClick={()=>onFact(f.fact_id)}>{fieldLabel(f.fact_id)}</button></th><td>{display(f)}</td><td>{states[f.missing_state]}</td><td>{sourceName(f.source.table)}<small>{f.source.table.startsWith('application')?'申请时点快照；绝对日期未知':f.source.table==='征信借款记录'?'借款日、更新日均早于申请':'决策日早于申请'}</small></td><td>{units[f.unit]||'来源单位待核对'}</td><td>{prose(signal(f.fact_id)||referenceHint(f.fact_id)||'—')}</td></tr>)}</tbody></table>{fields.length===0&&<p className="empty-state">此状态下没有字段。</p>}</div></Panel></section>;
}
