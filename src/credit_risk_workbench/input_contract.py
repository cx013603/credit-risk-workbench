"""Label-free raw input contract. Presence, source null and no-history are distinct."""
import hashlib
import json
import math
import re
from .contracts import require

VERSION='new_application_raw40_v1'
SEAL='6e8100f48b575404d49d789a326fae2280f4e8bc73da888c908350d54e3864f0'
MAIN=['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_INCOME_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START']
NUMERIC={'AMT_INCOME_TOTAL','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','HOUR_APPR_PROCESS_START'}
BUREAU=['SK_ID_CURR','SK_ID_BUREAU','DAYS_CREDIT','DAYS_CREDIT_UPDATE','CREDIT_CURRENCY','CREDIT_ACTIVE','AMT_CREDIT_SUM']
PREVIOUS=['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION','NAME_CONTRACT_TYPE']
MAX_BYTES=15000

def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def opaque_id(value):require(isinstance(value,str) and re.fullmatch('[a-f0-9]{32}',value) is not None,'INVALID_DRAFT_ID');return value
def blank():
    return {'contract_version':VERSION,'source_kind':'not_declared','source_reference':'','monetary_unit':'not_declared','time_reference':'not_declared','snapshot_availability':'not_declared','application':{},'bureau':{'declaration':'not_provided','rows':[]},'previous':{'declaration':'not_provided','rows':[]}}

def check(value):
    """Return bounded actionable issues; incomplete/invalid payloads can remain drafts."""
    issues=[];warnings=[]
    def issue(path,code,message):
        if len(issues)<150:issues.append({'path':path,'code':code,'message':message})
    def keys(obj,expected,path):
        if not isinstance(obj,dict):issue(path,'OBJECT','需要资料对象。');return False
        for n in expected-set(obj):issue(f'{path}.{n}','MISSING_FIELD','未提供；如来源字段确实为空，请显式填写 null。')
        for n in set(obj)-expected:issue(f'{path}.{n}','EXTRA_FIELD','不接受标签、汇总特征或未约定字段。')
        return True
    def num(v,path,lo=-1e15,hi=1e15,integral=False,nullable=True):
        if v is None and nullable:return True
        try:ok=type(v) in (int,float) and (nullable or type(v) is int) and lo<=v<=hi and math.isfinite(v) and (not integral or float(v).is_integer())
        except (ValueError,OverflowError,TypeError):ok=False
        if not ok:issue(path,'NUMBER','数值类型、范围或整数要求不符；不可用字符串、布尔或非有限值。')
        return ok
    def text(v,path,nullable=True):
        if v is None and nullable:return True
        ok=isinstance(v,str) and 1<=len(v)<=120 and not any(ord(c)<32 for c in v)
        if not ok:issue(path,'TEXT','填写原始类别文字；未知保留原文，来源空值用 null。')
        return ok
    expected=set(blank())
    if not keys(value,expected,'input'):return {'ready_to_score':False,'issues':issues,'warnings':warnings}
    for name,allowed,msg in [
        ('contract_version',[VERSION],'请选择当前原40项输入契约。'),
        ('source_kind',['home_credit_source_compatible','synthetic_source_encoded'],'声明公开来源兼容资料或按来源编码的合成演示。'),
        ('monetary_unit',['home_credit_source_scale'],'只接收原匿名金额尺度；人民币、月薪及其他币种须先取得映射证据。'),
        ('time_reference',['current_application_day_zero'],'历史时间必须以本次申请为0日的相对天数。'),
        ('snapshot_availability',['source_snapshot_declared'],'须声明资料为来源提供的申请前快照；无法独立重建真实 as-of。')]:
        if value.get(name) not in allowed:issue(name,'DECLARATION',msg)
    ref=value.get('source_reference')
    if not isinstance(ref,str) or not 1<=len(ref.strip())<=256 or any(ord(c)<32 for c in ref):issue('source_reference','SOURCE','填写来源说明（1～256字），不填个人身份信息。')
    a=value.get('application');ident=None
    if keys(a,set(['SK_ID_CURR',*MAIN]),'application'):
        ident=a.get('SK_ID_CURR');num(ident,'application.SK_ID_CURR',1,2**53-1,True,False)
        for name in MAIN:
            if name not in a:continue
            v=a[name];p='application.'+name
            if name in NUMERIC:
                if name.startswith('FLAG_'):num(v,p,0,1,True)
                elif name=='HOUR_APPR_PROCESS_START':num(v,p,0,23,True)
                else:num(v,p,0,1e15)
            else:text(v,p)
    total=0
    for family,names,key in [('bureau',BUREAU,'SK_ID_BUREAU'),('previous',PREVIOUS,'SK_ID_PREV')]:
        obj=value.get(family)
        if not keys(obj,{'declaration','rows'},family):continue
        rows=obj.get('rows');declaration=obj.get('declaration')
        if not isinstance(rows,list) or len(rows)>100:issue(family+'.rows','ROWS','每类最多100条记录。');continue
        total+=len(rows)
        if declaration not in ['provided','no_history']:issue(family+'.declaration','HISTORY_NOT_PROVIDED','选择已完整提供本契约所需历史，或明确声明来源无该类历史。')
        elif (declaration=='no_history' and rows) or (declaration=='provided' and not rows):issue(family+'.declaration','HISTORY_CONFLICT','无历史声明必须为空数组；已提供声明必须包含记录。')
        seen=set()
        for j,row in enumerate(rows):
            path=f'{family}.rows[{j}]'
            if not keys(row,set(names),path):continue
            pk=row.get(key)
            if num(pk,path+'.'+key,1,2**53-1,True,False):
                if pk in seen:issue(path+'.'+key,'DUPLICATE_KEY','重复历史主键，拒绝静默去重。')
                seen.add(pk)
            if not num(row.get('SK_ID_CURR'),path+'.SK_ID_CURR',1,2**53-1,True,False) or row.get('SK_ID_CURR')!=ident:issue(path+'.SK_ID_CURR','FOREIGN_KEY','历史必须关联本次申请编号。')
            times=['DAYS_CREDIT','DAYS_CREDIT_UPDATE'] if family=='bureau' else ['DAYS_DECISION']
            eligible=True
            for n in times:
                v=row.get(n)
                if not num(v,path+'.'+n,-1000000,1000000,True) or v is None or v>=0:eligible=False
            if not eligible:warnings.append({'path':path,'code':'TIME_EXCLUDED','message':'缺少时点、同日或未来记录按原SQL隔离，不进入历史特征。'})
            if family=='bureau':
                currency=row.get('CREDIT_CURRENCY')
                if currency is not None and currency not in ['currency 1','currency 2','currency 3','currency 4']:issue(path+'.CREDIT_CURRENCY','CURRENCY','仅接受来源匿名 currency 1～4 或 null，不换算真实币种。')
                text(row.get('CREDIT_ACTIVE'),path+'.CREDIT_ACTIVE')
                if num(row.get('AMT_CREDIT_SUM'),path+'.AMT_CREDIT_SUM') and row.get('AMT_CREDIT_SUM') is not None and row['AMT_CREDIT_SUM']<0:warnings.append({'path':path+'.AMT_CREDIT_SUM','code':'AMOUNT_UNAVAILABLE','message':'负金额按来源规则保留，隔离出金额公式，不当作可用额度。'})
            else:text(row.get('NAME_CONTRACT_TYPE'),path+'.NAME_CONTRACT_TYPE')
    if total>100:issue('history','ROWS','两类历史合计最多100条；超出请使用后续更大输入契约。')
    try:
        if len(canonical(value))>MAX_BYTES:issue('input','INPUT_SIZE','本契约完整资料上限15000字节，不能截断后评分。')
    except (ValueError,TypeError,OverflowError,RecursionError):issue('input','JSON','必须为有限数值的JSON对象。')
    return {'ready_to_score':not issues,'issues':issues,'warnings':warnings}

def require_complete(value):
    result=check(value);require(result['ready_to_score'],'INPUT_INCOMPLETE');return result
