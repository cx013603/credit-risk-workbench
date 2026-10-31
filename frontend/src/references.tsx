import {field,fieldLabel,sourceName,unitName,valueText,stateName,prose,binText} from './presentation';
import {decimal,metric,decimalText} from './format';
import {useEffect,useRef,useState,lazy,Suspense} from 'react';
import {Alert,Button,Skeleton} from 'antd';
import type {CaseData} from './api/types';
import {fetchReference} from './api/referenceClient';
import type {ReferenceData} from './api/referenceClient';
import {contributionSteps} from './contributionData';
const Chart=lazy(()=>import('./charts'));
export type ReferenceState={status:'idle'|'loading'|'ready'|'failed';data?:ReferenceData;error?:string;retry:()=>void};
export function useReference(c:CaseData|null):ReferenceState{
 const [attempt,setAttempt]=useState(0),[stored,setStored]=useState<{key:string;data?:ReferenceData;error?:string}>({key:''}),seq=useRef(0);
 const key=c?`${c.case_key.population}:${c.case_key.application_id}:${c.case_revision}:${attempt}`:'';
 useEffect(()=>{const ac=new AbortController(),current=++seq.current;setStored({key});if(c)void fetchReference(c,`reference-${current}`,ac.signal).then(data=>{if(!ac.signal.aborted&&current===seq.current)setStored({key,data})}).catch(e=>{if(!ac.signal.aborted&&current===seq.current)setStored({key,error:e instanceof Error?e.message:'参考暂不可用'})});return()=>ac.abort()},[c,key]);
 const matching=stored.key===key;return {status:!c?'idle':matching&&stored.data?'ready':matching&&stored.error?'failed':'loading',data:matching?stored.data:undefined,error:matching?stored.error:undefined,retry:()=>setAttempt(x=>x+1)};
}
export function TrainingReference({c,factId,state}:{c:CaseData;factId:string;state:ReferenceState}){
 if(state.status==='failed')return <div data-testid="reference-error"><Alert type="error" title="训练参考未能载入" description={state.error}/><Button onClick={state.retry}>重新读取参考</Button></div>;
 const data=state.data,row=data?.rows.find(r=>r.fact_id===factId),fact=c.profile.facts.find(f=>f.fact_id===factId)!;
 if(!data||!row)return <div aria-busy="true"><Skeleton active paragraph={{rows:3}}/></div>;
 const chartRows=row.bins.map(b=>({id:String(b.index),label:binText(decimalText(b.label)),value:b.count,selected:b.current}));
 return <section className="training-reference" data-testid="training-reference" data-fact-id={factId} data-case-key={`${c.case_key.population}:${c.case_key.application_id}`} data-reference-revision={data.reference_revision} data-bucket={row.current_bucket} data-position={row.position_status}>
  <div className="reference-meta"><b>训练申请 {row.rows.toLocaleString('en-US')} 笔</b><span>缺失 {row.missing_count.toLocaleString('en-US')} 笔</span></div>
  <p>{prose(row.description)}</p><div className="reference-position">当前保存值：<strong>{valueText(factId,row.raw_value)}</strong> <small>{unitName(fact.unit)}</small><br/>所在桶：<b>{binText(decimalText(row.bins[row.current_bucket].label))}</b> · {row.current_bucket_count.toLocaleString('en-US')}笔 / {decimal(row.current_bucket_share*100)}%</div>
  <p className="fine">蓝色为当前所在桶，灰色为其他桶；数值刻度是训练笔数。数值分箱左闭右开，首尾开放；区间标签作显示舍入，不据标签反算边界。类别低于1%单列提示。</p>
  <div className="reference-viewport" tabIndex={0} aria-label="训练分布图与分桶标签，可上下滚动"><div className="reference-bars" style={{height:Math.max(180,row.bins.length*27+36)}}><div className="reference-labels">{chartRows.map(r=><span key={r.id} title={r.label}>{r.selected?'● ':''}{r.label}</span>)}</div><Suspense fallback={<Skeleton active/>}><Chart kind="reference" rows={chartRows}/></Suspense></div></div>
  <details className="chart-data"><summary>训练参考等价数据与当前定位</summary><div className="table-scroll" tabIndex={0}><table data-testid="reference-table"><thead><tr><th>分桶 / {unitName(fact.unit)}</th><th>训练笔数</th><th>占全部训练申请</th><th>当前值</th></tr></thead><tbody>{row.bins.map(b=><tr key={b.index} className={b.current?'current-bin':''}><th>{binText(decimalText(b.label))}</th><td>{b.count}</td><td>{decimal(b.share*100)}%</td><td>{b.current?'● 所在桶':''}</td></tr>)}</tbody></table></div></details>
  <details className="technical"><summary>参考口径与版本</summary><p>仅固定训练申请；验证和无标签案例只作定位，均不计入参考。训练案例本身属于参考人群。缺失单列，常量按小于／等于／大于分桶，全空字段保留非空未知桶；类别词表与模型编码器词表分别判断。</p><p>分箱参考，不提供精确百分位；该桶占比不是预测概率。超出训练范围、未知与稀有类别不等于风险必然增加。</p></details>
 </section>;
}
const sign=(n:number)=>(n>0?'+':'')+decimal(n);
export function ContributionWaterfall({c,onFact,onAll}:{c:CaseData;onFact:(id:string)=>void;onAll:()=>void}){
 const data=contributionSteps(c),all=data.steps.flatMap(s=>[s.from,s.to,0]),min=Math.min(...all)-.15,max=Math.max(...all)+.15,x=(n:number)=>175+(n-min)/(max-min)*420;
 const activate=(id:string,kind:string)=>{if(kind==='factor')onFact(id);else if(kind==='rest')onAll()};
 return <section className="panel waterfall-panel" data-testid="waterfall" data-score={data.score} data-probability={data.probability}><div className="panel-heading"><h2>从参考值到当前分数</h2><small>前5项 ＋ 其余35项</small></div><p className="fine">每步沿用原训练路径贡献，横轴为对数几率；标记＋／−分别推高／拉低，蓝色表示参考或最终合计。点因素可联动原值、中文解释、训练参考与来源。</p>
  <div className="waterfall-scroll" tabIndex={0} aria-label="贡献瀑布图，可横向滚动"><svg viewBox="0 0 730 350" role="img" aria-label="参考值加5项贡献及其余35项合计还原原内部得分。等价数据见下表。"><line x1={x(0)} x2={x(0)} y1="12" y2="315" stroke="#abb9cd"/>{data.steps.map((s,i)=>{const interactive=s.kind==='factor'||s.kind==='rest',y=16+i*37;return <g key={s.id} role={interactive?'button':undefined} tabIndex={interactive?0:undefined} aria-label={`${prose(s.label)}，${sign(s.delta)}对数几率${interactive?'，打开详情':''}`} onClick={()=>activate(s.id,s.kind)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate(s.id,s.kind)}}} className={interactive?'waterfall-action':''}><title>{prose(s.label)}：从{sign(s.from)}到{sign(s.to)}，变化{sign(s.delta)}</title><text x="5" y={y+17} fontSize="12" fill="#38516d">{prose(s.label).length>13?prose(s.label).slice(0,13)+'…':prose(s.label)}</text><rect x={Math.min(x(s.from),x(s.to))} y={y} width={Math.max(1,Math.abs(x(s.to)-x(s.from)))} height="24" rx="3" fill={s.kind==='base'||s.kind==='score'?'#2563eb':s.delta>=0?'#b45332':'#087f78'}/><text x="615" y={y+17} fontSize="12" fill="#38516d">{sign(s.delta)}</text>{i<data.steps.length-1&&<line x1={x(s.to)} x2={x(s.to)} y1={y+24} y2={y+37} stroke="#a9b6c9" strokeDasharray="3 3"/>}</g>})}{[min,(min+max)/2,max].map(v=><text key={v} x={x(v)} y="337" textAnchor="middle" fontSize="11" fill="#586b84">{decimal(v)}</text>)}</svg></div>
  <div className="waterfall-equation" data-testid="waterfall-equation">{decimal(data.score)} 对数几率 → 概率转换 → <b>{decimal(data.probability*100)}%</b><small>计算与原树模型概率一致；显示值保留两位小数，合计可能有舍入差异。</small></div>
  <details className="chart-data"><summary>瀑布图等价数据与键盘操作</summary><div className="table-scroll" tabIndex={0}><table data-testid="waterfall-table"><thead><tr><th>步骤</th><th>起点</th><th>变化</th><th>终点 / 对数几率</th></tr></thead><tbody>{data.steps.map(s=><tr key={s.id}><th>{s.kind==='factor'||s.kind==='rest'?<button className="text-button" onClick={()=>activate(s.id,s.kind)}>{prose(s.label)}</button>:s.label}</th><td>{decimal(s.from)}</td><td>{decimal(s.delta)}</td><td>{decimal(s.to)}</td></tr>)}</tbody></table></div></details>
 </section>;
}
