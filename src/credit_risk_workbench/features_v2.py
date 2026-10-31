"""Stateless V2 features and cooperative frozen-input gates; no model imports."""
from contextlib import closing
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
from credit_risk.raw import DataError,sha256_file

VERSION='history_extended_v2.0'
def require(ok,message):
    if not ok:raise DataError('v2_feature_contract',message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def literal(v):return "'"+str(v).replace("'","''")+"'"
def scalar(db,sql):return db.execute(sql).fetchone()[0]
def record(db,sql):
    c=db.execute(sql);return dict(zip([v[0] for v in c.description],c.fetchone()))

def feature_definitions():
    definitions=[]
    def add(name,label,kind,unit,formula,source):definitions.append({'name':name,'label':label,'type':'int64' if kind=='count' else 'float64','kind':kind,'unit':unit,'formula':formula,'source':source,'missing':'0 = count of eligible records/loans/groups, not absence of all history' if kind=='count' else 'null when no eligible observations/denominator; observed zero retained'})
    for prefix,source in [('CC','credit_card_balance.csv'),('POS','POS_CASH_balance.csv')]:
        for suffix,label,kind,unit,formula in [
            ('LOAN_COUNT','合格月记录借款数','count','loans','count loan groups'),
            ('MONTH_COUNT','合格借款月记录数','count','loan-month observations','sum per-loan record counts'),
            ('MONTHS_SINCE_RECENT','最近合格快照距申请月数','numeric','relative months','-max eligible MONTHS_BALANCE'),
            ('MAX_DPD','合格月最大源DPD','numeric','days','max nonnegative integral SK_DPD'),
            ('POSITIVE_DPD_FRACTION','源DPD大于0的借款月比例','ratio','fraction','positive DPD rows / known nonnegative integral DPD rows; weighted by records'),
            ('DPD_UNKNOWN_COUNT','源DPD缺失或无效记录数','count','observations','eligible records minus known nonnegative integral DPD rows'),
            ('LAST6_MONTH_COUNT','申请前6个月借款月记录数','count','loan-month observations','count MONTHS_BALANCE in [-6,-1]'),
            ('LAST12_MONTH_COUNT','申请前12个月借款月记录数','count','loan-month observations','count MONTHS_BALANCE in [-12,-1]')]:add(prefix+'_'+suffix,label,kind,unit,formula,source)
    for suffix,label,kind,unit,formula in [
        ('LOAN_COUNT','合格征信月记录借款数','count','loans','count loan groups'),('MONTH_COUNT','合格征信借款月记录数','count','loan-month observations','sum loan record counts'),
        ('MONTHS_SINCE_RECENT','最近征信快照距申请月数','numeric','relative months','-max eligible MONTHS_BALANCE'),
        ('DELINQUENT_STATUS_FRACTION','源状态1至5的活动月比例','ratio','fraction','status1..5 / status0..5; exclude C and X from denominator'),
        ('UNKNOWN_STATUS_FRACTION','未知征信月状态比例','ratio','fraction','X/null/unrecognized status / all eligible rows'),
        ('CLOSED_STATUS_FRACTION','源关闭月状态比例','ratio','fraction','C / all eligible rows'),
        ('LAST12_MONTH_COUNT','申请前12个月征信借款月记录数','count','loan-month observations','count MONTHS_BALANCE in [-12,-1]')]:add('BB_'+suffix,label,kind,unit,formula,'bureau_balance.csv')
    for suffix,label,kind,unit,formula in [
        ('LOAN_COUNT','合格支付事件借款数','count','loans','count loans with known prior due/payment day'),
        ('PAYMENT_RECORD_COUNT','合格支付事件记录数','count','payment observations','sum per-loan eligible event counts; not installments count'),
        ('DAYS_SINCE_RECENT_PAYMENT','最近已知支付距申请天数','numeric','relative days','-max eligible DAYS_ENTRY_PAYMENT'),
        ('EXCLUDED_CALENDAR_LOAN_COUNT','日历口径隔离借款数','count','loans','eligible-event loans without exactly one positive integral calendar version'),
        ('SINGLE_CALENDAR_GROUP_COUNT','单正版本日历分期组数','count','calendar-installment groups','sum per-loan distinct (version,number,due day) groups'),
        ('LATE_PAYMENT_GROUP_FRACTION','含晚于应付日支付事件的组比例','ratio','fraction','groups whose last observed payment day > due day / eligible groups; not default or fully-paid status'),
        ('MAX_PAYMENT_DELAY_DAYS','最迟已知支付时间差正部','numeric','days','max(max(last observed payment day-due day,0)) over eligible groups'),
        ('MULTI_PAYMENT_GROUP_FRACTION','含多条支付记录的组比例','ratio','fraction','groups with >1 eligible payment row / eligible groups; never sum prescribed amount per row')]:add('INST_'+suffix,label,kind,unit,formula,'installments_payments.csv')
    require(len(definitions)==31,'31 frozen new feature definitions required');return definitions

def run_family(db,root,family):
    root=Path(root);name={'CC':'cc_input','POS':'pos_input','BB':'bb_input','INST':'inst_input'}[family]
    if family in ['CC','POS']:sql=(root/'sql/workbench_v2/monthly.sql').read_text(encoding='utf-8').format(prefix=family,input=name)
    else:sql=(root/('sql/workbench_v2/bureau_monthly.sql' if family=='BB' else 'sql/workbench_v2/installments.sql')).read_text(encoding='utf-8')
    db.execute(sql)
    key='SK_ID_BUREAU' if family=='BB' else 'SK_ID_PREV'
    if family in ['CC','POS','BB']:
        require(scalar(db,f'SELECT COUNT(*) FROM (SELECT {key},MONTHS_BALANCE FROM {family}_eligible GROUP BY 1,2 HAVING COUNT(*)>1)')==0,'Duplicate eligible loan-month key')
    else:
        columns='SK_ID_PREV,SK_ID_CURR,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER,DAYS_INSTALMENT,DAYS_ENTRY_PAYMENT,AMT_INSTALMENT,AMT_PAYMENT'
        require(scalar(db,f'SELECT COUNT(*) FROM INST_eligible')==scalar(db,f'SELECT COUNT(*) FROM (SELECT DISTINCT {columns} FROM INST_eligible)'),'Duplicate exact eligible payment record')
        require(scalar(db,'SELECT COUNT(*) FROM (SELECT SK_ID_CURR,SK_ID_PREV,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER,DAYS_INSTALMENT FROM INST_eligible GROUP BY 1,2,3,4,5 HAVING COUNT(DISTINCT AMT_INSTALMENT)>1)')==0,'Conflicting prescribed installment amount')
    for col,kind,*_ in db.execute(f'DESCRIBE {family}_features').fetchall():
        if col!='SK_ID_CURR':require(scalar(db,f'SELECT COUNT(*) FROM {family}_features WHERE "{col}" IS NOT NULL AND NOT isfinite("{col}")')==0,'Nonfinite aggregate')
    counts=record(db,f'SELECT COUNT(*) rows,COUNT(DISTINCT SK_ID_CURR) applications,COUNT(DISTINCT {key}) loans FROM {family}_eligible')
    count_col={'CC':'CC_MONTH_COUNT','POS':'POS_MONTH_COUNT','BB':'BB_MONTH_COUNT','INST':'INST_PAYMENT_RECORD_COUNT'}[family]
    require(scalar(db,f'SELECT COALESCE(SUM({count_col}),0) FROM {family}_features')==counts['rows'],'Two-level count conservation failed')
    return counts

def join_new(db):
    projection=[]
    for f in feature_definitions():
        alias=f['name'].split('_',1)[0];name=f['name'];projection.append((f'COALESCE({alias}."{name}",0)::BIGINT' if f['kind']=='count' else f'{alias}."{name}"')+f' AS "{name}"')
    db.execute('CREATE TABLE new_features AS SELECT a.SK_ID_CURR,'+','.join(projection)+' FROM allowed a '+''.join(f'LEFT JOIN {n}_features {n} USING(SK_ID_CURR) ' for n in ['CC','POS','BB','INST']))
    require(scalar(db,'SELECT COUNT(*) FROM new_features')==scalar(db,'SELECT COUNT(DISTINCT SK_ID_CURR) FROM new_features')==scalar(db,'SELECT COUNT(*) FROM allowed'),'One application one row failed')
    return feature_definitions()

def compare_common(db,names):
    # Float sums may differ by aggregation order, null/string/count identity exact.
    problems={}
    for f in names:
        name=f['name'];eq=f'a."{name}" IS NOT DISTINCT FROM b."{name}"'
        if f['type']=='float64':eq=f'({eq} OR (a."{name}" IS NOT NULL AND b."{name}" IS NOT NULL AND abs(a."{name}"-b."{name}")<=1e-9+1e-12*abs(b."{name}")))'
        n=scalar(db,f'SELECT COUNT(*) FROM original_recomputed a FULL JOIN base b USING(SK_ID_CURR) WHERE a.SK_ID_CURR IS NULL OR b.SK_ID_CURR IS NULL OR NOT {eq}')
        if n:problems[name]=n
    require(not problems,'Original40 recomputation mismatch: '+str(problems));return {'columns':len(names),'rows':scalar(db,'SELECT COUNT(*) FROM base'),'mismatch_counts':problems,'numeric_absolute_tolerance':1e-9,'numeric_relative_tolerance':1e-12,'null_category_count_comparison':'exact'}

def verify_frozen(root,directory,seal):
    root=Path(root).resolve();directory=Path(directory).resolve();f=directory/'freeze.json'
    require(isinstance(seal,str) and len(seal)==64 and sha256_file(f)==seal,'Freeze seal mismatch')
    receipt=read(f);require(receipt['version']==VERSION and receipt['G3_passed'] and not receipt['training_started'],'Not a completed step12 freeze')
    require(sha256_file(directory/'manifest.json')==receipt['manifest_sha256'],'Manifest changed');m=read(directory/'manifest.json')
    for name,info in m['files'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and p.is_file() and sha256_file(p)==info['sha256'],'Package artifact changed')
    for name,h in m['bindings'].items():
        p=(root/name).resolve();require(p.is_relative_to(root) and sha256_file(p)==h,'Bound code/config/source changed: '+name)
    import platform,importlib.metadata
    for name,version in m['environment'].items():
        require((platform.python_version() if name=='python' else importlib.metadata.version(name))==version,'Bound environment changed')
    require(digest(read(directory/'protocol.json'))==receipt['protocol_fingerprint'],'Protocol changed')
    return m

def load_partition(root,directory,seal,partition,*,purpose='score',heldout_fold=None):
    require(partition in ['train','validation','unlabeled_scoring'],'Reserved/unknown partition denied')
    require(purpose in ['score','fit'],'Unknown input purpose')
    require(purpose!='fit' or partition=='train','Fit may only use training rows')
    require(heldout_fold is None or type(heldout_fold) is int and heldout_fold in range(1,6),'Invalid CV fold')
    require(heldout_fold is None or purpose=='fit','Fold complement only meaningful for fit')
    verify_frozen(root,directory,seal)
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    table=pq.read_table(Path(directory)/(partition+'.parquet'))
    if heldout_fold is not None:table=table.filter(pc.not_equal(table['CV_FOLD'],heldout_fold))
    require(table.num_rows>0 and set(table['sample_use'].to_pylist())=={partition},'Wrong partition contents')
    contract=read(Path(directory)/'contract.json')
    names=contract['feature_columns']
    require(contract['version']==VERSION and len(names)==len(set(names))==71 and not set(names)&{'SK_ID_CURR','sample_use','CV_FOLD','TARGET','partition'},'Invalid feature allowlist')
    require(table.column_names==['SK_ID_CURR','sample_use','CV_FOLD']+names,'Unexpected package schema/order')
    return {'X':table.select(contract['feature_columns']),
            'identity':table.select(['SK_ID_CURR','sample_use','CV_FOLD']),
            'feature_columns':contract['feature_columns'],
            'purpose':purpose,'heldout_fold':heldout_fold}
