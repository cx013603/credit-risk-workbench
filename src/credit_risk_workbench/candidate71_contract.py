"""Raw71 input presence/type/relationship contract; never invent missing history."""
import math
from .contracts import require
from .input_contract import blank as blank40, check as check40, VERSION as V40, canonical, digest

VERSION = 'new_application_raw71_v1'
MAX_BYTES = 240000
TABLES = {
    'cc': ['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','SK_DPD'],
    'pos': ['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','SK_DPD'],
    'bb': ['SK_ID_BUREAU','MONTHS_BALANCE','STATUS'],
    'inst': ['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION','NUM_INSTALMENT_NUMBER','DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT','AMT_INSTALMENT','AMT_PAYMENT'],
}


def blank():
    p = blank40(); p['contract_version'] = VERSION
    p.update({k:{'declaration':'not_provided','rows':[]} for k in TABLES})
    return p


def raw40(p):
    return {**{k:v for k,v in p.items() if k not in TABLES}, 'contract_version': V40}


def check(p):
    def error(path, code, message):
        return {'path':path,'code':code,'message':message}
    if not isinstance(p,dict):
        return {'ready_to_score':False,'issues':[error('input','OBJECT','需要完整资料对象。')],'warnings':[]}
    try:
        if len(canonical(p)) > MAX_BYTES:
            return {'ready_to_score':False,'issues':[error('input','INPUT_SIZE','完整资料最多240000字节；不得截断记录后评分。')],'warnings':[]}
    except (TypeError,ValueError,RecursionError,OverflowError):
        return {'ready_to_score':False,'issues':[error('input','JSON','JSON中只允许有限数值。')],'warnings':[]}
    try:result = check40(raw40(p))
    except (TypeError,ValueError,KeyError):result={'issues':[error('application','TYPE','原40项父资料包含无效结构或主键类型。')],'warnings':[]}
    issues, warnings = result['issues'], result['warnings']
    if p.get('contract_version') != VERSION:
        issues.append(error('contract_version','VERSION','请选择71项原始资料契约。'))
    ident = p.get('application',{}).get('SK_ID_CURR') if isinstance(p.get('application'),dict) else None
    previous = p.get('previous',{}); bureau = p.get('bureau',{})
    prev = {r.get('SK_ID_PREV'):r for r in previous.get('rows',[]) if isinstance(r,dict) and type(r.get('SK_ID_PREV')) is int} if isinstance(previous,dict) and isinstance(previous.get('rows'),list) else {}
    bs = {r.get('SK_ID_BUREAU'):r for r in bureau.get('rows',[]) if isinstance(r,dict) and type(r.get('SK_ID_BUREAU')) is int} if isinstance(bureau,dict) and isinstance(bureau.get('rows'),list) else {}
    total = 0
    for family, columns in TABLES.items():
        obj = p.get(family)
        if not isinstance(obj,dict) or set(obj) != {'declaration','rows'}:
            issues.append(error(family,'MISSING_HISTORY','需要明确提供资料、无该类历史或尚未提供声明。')); continue
        rows, declaration = obj['rows'], obj['declaration']
        if not isinstance(rows,list) or len(rows)>1000:
            issues.append(error(family,'ROWS','每类最多1000条原始记录，不自动截断。')); continue
        total += len(rows)
        if declaration not in ['provided','no_history'] or (declaration=='provided' and not rows) or (declaration=='no_history' and rows):
            issues.append(error(family,'DECLARATION','未提供资料不能当作无历史；声明和记录须一致。'))
        for j,row in enumerate(rows):
            path=f'{family}.rows[{j}]'
            if not isinstance(row,dict) or set(row) != set(columns):
                issues.append(error(path,'FIELDS','原始记录须提供全部约定键，来源空值写null，不接受标签或预计算特征。')); continue
            for name,value in row.items():
                if name=='STATUS':
                    good=value is None or isinstance(value,str) and len(value)<=80 and not any(ord(c)<32 for c in value)
                elif name.startswith('SK_ID_'):
                    good=type(value) is int and 0<value<2**53
                else:
                    good=value is None or type(value) in [int,float] and math.isfinite(value) and abs(value)<=1e15
                    if good and value is not None and name in ['MONTHS_BALANCE','NUM_INSTALMENT_NUMBER']:
                        good=value==int(value)
                if not good: issues.append(error(path+'.'+name,'TYPE','原始数值、整数键或来源空值类型不符合约定。'))
            if family=='bb':
                parent=bs.get(row['SK_ID_BUREAU']) if type(row['SK_ID_BUREAU']) is int else None
                if parent is None and type(row['SK_ID_BUREAU']) is int:
                    warnings.append(error(path,'ORPHAN_EXCLUDED','父征信记录未提供；按冻结SQL隔离此月记录，不参与71项聚合。'))
                elif parent is None or parent.get('SK_ID_CURR') != ident:
                    issues.append(error(path,'FOREIGN_KEY','征信月记录必须关联本申请中明确提供的征信父记录。'))
            elif type(row['SK_ID_PREV']) is not int or row['SK_ID_CURR'] != ident or row['SK_ID_PREV'] in prev and prev[row['SK_ID_PREV']].get('SK_ID_CURR') != ident:
                issues.append(error(path,'FOREIGN_KEY','月度／支付记录必须与本申请及提供的历史申请父键成对关联。'))
            elif row['SK_ID_PREV'] not in prev:
                warnings.append(error(path,'ORPHAN_EXCLUDED','父历史申请未提供；按冻结SQL隔离此记录，不参与71项聚合。'))
            times=['DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT'] if family=='inst' else ['MONTHS_BALANCE']
            if any(row.get(n) is None or type(row.get(n)) not in [int,float] or not math.isfinite(row[n]) or row[n]>=0 for n in times):
                warnings.append(error(path,'TIME_EXCLUDED','未知时点、同日／未来记录按冻结SQL隔离，不进入特征。'))
    if total>1600: issues.append(error('history','ROWS','新增四类历史总计最多1600条，不截断后评分。'))
    return {'ready_to_score':not issues,'issues':issues[:150],'warnings':warnings[:150],'warning_count':len(warnings),'warnings_truncated':len(warnings)>150}


def complete(p):
    v=check(p);require(v['ready_to_score'],'INPUT_INCOMPLETE');return v
