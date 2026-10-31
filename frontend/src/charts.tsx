import {decimal,metric} from './format';
import {useEffect,useRef} from 'react';
import {init,use} from 'echarts/core';
import {BarChart} from 'echarts/charts';
import {AriaComponent,GridComponent,TooltipComponent} from 'echarts/components';
import {SVGRenderer} from 'echarts/renderers';
use([BarChart,GridComponent,TooltipComponent,AriaComponent,SVGRenderer]);
type Row={id:string;label:string;value:number;selected?:boolean};
export default function Chart({rows,kind,onSelect}:{rows:Row[];kind:'models'|'contributions'|'reference';onSelect?:(id:string)=>void}){
 const element=useRef<HTMLDivElement>(null),select=useRef(onSelect);select.current=onSelect;
 useEffect(()=>{if(!element.current)return;const model=kind==='models',reference=kind==='reference';const chart=init(element.current,undefined,{renderer:'svg'});const max=Math.max(.001,...rows.map(r=>Math.abs(r.value)))*1.15;
  chart.setOption({animation:false,aria:{enabled:true,label:{description:reference?'固定训练申请分桶笔数；蓝色所在桶有圆点标记，等价数据见表格。':model?'同一申请的三模型概率，刻度0至100%，评分卡分值单独展示。':'主要五项路径贡献，正值推高、负值拉低内部得分，单位对数几率。等价数据见下表。'}},grid:{left:5,right:28,top:8,bottom:30,containLabel:false},tooltip:{trigger:'axis',valueFormatter:(v:number)=>reference?metric(v):model?decimal(v)+'%':(v>0?'+':'')+decimal(v)},xAxis:{type:'value',min:model||reference?0:-max,max:model?100:max,interval:model?25:undefined,axisLabel:{color:'#65758d',fontSize:10,formatter:reference?(v:number)=>v>=1000?decimal(v/1000)+'k':metric(v):model?'{value}%':(v:number)=>v===0?'0':decimal(v)},splitNumber:4,splitLine:{lineStyle:{color:'#edf1f7'}},axisLine:{show:false}},yAxis:{type:'category',inverse:true,data:rows.map(r=>r.label),axisLabel:{show:false},axisTick:{show:false},axisLine:{show:false}},series:[{type:'bar',barWidth:model?10:14,data:rows.map(r=>({value:r.value,itemStyle:{color:reference?(r.selected?'#2563eb':'#c7d3e3'):model?'#2563eb':r.value>=0?'#b45332':'#087f78',borderRadius:3}})),markLine:model||reference?undefined:{silent:true,symbol:'none',lineStyle:{color:'#a6b5ca',type:'solid'},data:[{xAxis:0}]}}]});
  chart.on('click',p=>{if(typeof p.dataIndex==='number')select.current?.(rows[p.dataIndex].id)});
  const observer=new ResizeObserver(()=>chart.resize());observer.observe(element.current);return()=>{observer.disconnect();chart.dispose()};
 },[rows,kind]);
 return <div ref={element} className={`echart ${kind}`} role="img" aria-label={kind==='reference'?'固定训练申请分布，计数尺度，当前所在桶以圆点及表格标记':kind==='models'?'三模型概率，同轴0至100%':'主要因素正负贡献，单位对数几率'} data-testid={`${kind}-chart`} data-chart-values={JSON.stringify(rows.map(r=>r.value))}/>;
}
