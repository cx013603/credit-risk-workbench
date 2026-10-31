"""User-facing Chinese labels, independent of the frozen scoring contract."""
import json
import re
from pathlib import Path
from .reports import number
DATA=json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
def feature(name):return DATA['fields'][name]
def category(value):
    if value in DATA['categories']:return DATA['categories'][value]
    return '其他来源类别（含义未提供）' if value.isascii() else value
def value(name,v):
    if v is None:return '—（来源为空或无观测）'
    return category(v) if isinstance(v,str) else number(v)
def fact_value(name,v,state=None):
    if v is None:
        if state=='no_eligible_history':return '无符合条件的记录，无法计算'
        return '车辆信息未提供' if name=='FLAG_OWN_CAR' else '房产信息未提供' if name=='FLAG_OWN_REALTY' else '未提供或无可用记录' if state=='source_null_or_no_eligible_observation' else '资料未提供'
    if name=='FLAG_OWN_CAR':return '有车（数量未提供）' if v=='Y' else '无车' if v=='N' else '车辆情况待核对'
    if name=='FLAG_OWN_REALTY':return '有房产（数量未提供）' if v=='Y' else '无房产' if v=='N' else '房产情况待核对'
    if name.startswith('FLAG_'):return '有相关记录' if v==1 else '无相关记录' if v==0 else '标记待核对'
    if name.endswith('_HAS_ELIGIBLE_HISTORY'):return '有符合条件的历史记录' if v==1 else '无符合条件的历史记录' if v==0 else '历史记录待核对'
    if isinstance(v,str):return category(v)
    n=number(v)
    if name=='AMT_INCOME_TOTAL':return f'{n}（来源金额；币种及收入周期未提供）'
    if name=='HOUR_APPR_PROCESS_START':return f'{n}时'
    if 'MONTHS_SINCE_RECENT' in name:return f'申请前{n}个月'
    if 'DAYS_SINCE_' in name:return f'申请前{n}天'
    unit=DATA['fields'].get(name,{}).get('unit')
    if unit=='比例':return f'{v*100:.2f}%'
    if unit=='天':return n+'天'
    if unit in ['条','条记录','条借款月记录','条支付记录','笔借款','个分期组','组']:return n+unit+('（无符合条件记录）' if state=='no_eligible_history' else '')
    if name.startswith('BUREAU_C1_CREDIT_'):return f'{n}（匿名币种一的来源金额）'
    return n
def prose(text):
    pairs={**DATA['categories'],**DATA['sources'],**DATA['raw'],**DATA['units'],'null':'未提供','DPD':'逾期天数'}
    for a,b in sorted(pairs.items(),key=lambda x:-len(x[0])):text=text.replace(a,b)
    return re.sub(r'\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b','资料项',text)
