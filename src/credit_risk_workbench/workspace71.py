"""Single-model catalog and optimistic, immutable 71-feature analysis records."""
import hashlib,json,math,sqlite3,threading,uuid
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
from .contracts import require,integer
from .candidate71_contract import digest

STATUSES=['unreviewed','pending','needs_information','reviewed']
COUNTS={'bureau':'BUREAU_ELIGIBLE_COUNT','previous':'PREV_ELIGIBLE_COUNT','cc':'CC_MONTH_COUNT','pos':'POS_MONTH_COUNT','bb':'BB_MONTH_COUNT','inst':'INST_PAYMENT_RECORD_COUNT'}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quality71(values,names):
    counts={k:values[n] for k,n in COUNTS.items()};missing=[];no_history=0
    for n in names:
        group={'BUREAU':'bureau','PREV':'previous','CC':'cc','POS':'pos','BB':'bb','INST':'inst'}.get(n.split('_',1)[0])
        if group and counts[group]==0:no_history+=1
        elif values[n] is None:missing.append(n)
    return {'observed':71-len(missing)-no_history,'missing':len(missing),'no_history':no_history,'history_state':'present' if any(v>0 for v in counts.values()) else 'none','missing_names':missing,'history_counts':counts}

def build_catalog71(candidate,path):
    """Predict only: explicit allowed partitions and columns, no targets or fitting."""
    import pyarrow.parquet as pq
    from credit_risk.logistic import matrix
    from threadpoolctl import threadpool_limits
    path=Path(path);require(not path.exists(),'VERSION_MISMATCH');counts={}
    with closing(sqlite3.connect(path)) as db,db:
        db.executescript('CREATE TABLE cases(population TEXT,id INTEGER,sample_use TEXT,probability REAL,observed INTEGER,missing INTEGER,no_history INTEGER,history_state TEXT,missing_json TEXT,counts_json TEXT,PRIMARY KEY(population,id)); CREATE TABLE metadata(value TEXT);')
        for part in ['train','validation','unlabeled_scoring']:
            pop='unlabeled' if part=='unlabeled_scoring' else 'labeled';count=0
            parquet=pq.ParquetFile(candidate.spec['paths'][candidate.d['partitions'][part]])
            for batch in parquet.iter_batches(batch_size=4096,columns=['SK_ID_CURR','sample_use',*candidate.names]):
                import pyarrow as pa
                table=pa.Table.from_batches([batch]);X=matrix(table.select(candidate.names),candidate.names,candidate.contract)
                with threadpool_limits(limits=1):prob=candidate.model.predict_proba(X)[:,1]
                records=[]
                for row,p in zip(table.to_pylist(),prob):
                    require(row['sample_use']==part and math.isfinite(float(p)) and 0<=p<=1,'VERSION_MISMATCH');q=quality71(row,candidate.names)
                    records.append((pop,row['SK_ID_CURR'],part,float(p),q['observed'],q['missing'],q['no_history'],q['history_state'],json.dumps(q['missing_names']),json.dumps(q['history_counts'])))
                db.executemany('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?)',records);count+=len(records)
            counts[part]=count
        require(counts=={'train':215257,'validation':46127,'unlabeled_scoring':48744},'VERSION_MISMATCH')
        meta={**candidate.adapter,'counts':counts,'feature_count':71,'threshold':candidate.bundle['threshold']['value'],'version':'workspace71_catalog_v1'}
        db.execute('INSERT INTO metadata VALUES(?)',[json.dumps(meta)]);db.executescript('CREATE INDEX by_probability ON cases(population,probability,id); CREATE INDEX by_use ON cases(population,sample_use,id);')
    return {**meta,'sha256':sha(path),'file':path.name}

class Workflow71:
    def __init__(self,path,binding):
        self.path=Path(path);self.binding=binding;self.lock=threading.RLock()
        with closing(self.connect()) as db,db:
            db.executescript('CREATE TABLE IF NOT EXISTS workspace71_meta(version INTEGER PRIMARY KEY); INSERT OR IGNORE INTO workspace71_meta VALUES(1); CREATE TABLE IF NOT EXISTS reviews71(subject TEXT PRIMARY KEY,model TEXT NOT NULL,case_key TEXT NOT NULL,case_revision TEXT NOT NULL,revision INTEGER NOT NULL,favorite INTEGER NOT NULL,status TEXT NOT NULL,note TEXT NOT NULL,deleted INTEGER NOT NULL,updated TEXT NOT NULL); CREATE TABLE IF NOT EXISTS reports71(id TEXT PRIMARY KEY,subject TEXT NOT NULL,model TEXT NOT NULL,created TEXT NOT NULL,html TEXT NOT NULL,data TEXT NOT NULL,html_sha TEXT NOT NULL,data_sha TEXT NOT NULL);')
            require([r[0] for r in db.execute('SELECT version FROM workspace71_meta')]==[1],'VERSION_MISMATCH')
    def connect(self):
        db=sqlite3.connect(self.path,timeout=3);db.row_factory=sqlite3.Row;return db
    def validate(self,c):
        require(c['feature_count']==71 and all(c.get(k)==v for k,v in self.binding.items()),'VERSION_MISMATCH')
        require(digest({k:v for k,v in c.items() if k!='case_revision'})==c['case_revision'],'VERSION_MISMATCH')
    def read(self,subject,c,db=None):
        self.validate(c)
        if db is None:
            with self.lock,closing(self.connect()) as con:return self.read(subject,c,con)
        r=db.execute('SELECT * FROM reviews71 WHERE subject=? AND model=?',[subject,self.binding['model_sha256']]).fetchone()
        return {'subject':subject,'case_key':c['case_key'],'case_revision':c['case_revision'],'revision':r['revision'] if r else 0,'favorite':bool(r['favorite']) if r else False,'status':r['status'] if r else 'unreviewed','note':r['note'] if r else '', 'deleted':bool(r['deleted']) if r else False,'binding_matches':not r or r['case_revision']==c['case_revision'],'updated_at':r['updated'] if r else None,'model_sha256':self.binding['model_sha256']}
    def write(self,subject,c,v,action='save'):
        self.validate(c);integer(v['expected_revision'],0);require(v['expected_case_revision']==c['case_revision'],'VERSION_MISMATCH');require(action in ['save','delete','restore'])
        with self.lock,closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');old=self.read(subject,c,db);require(old['revision']==v['expected_revision'],'VERSION_MISMATCH')
            if action=='save':
                require(not old['deleted'],'REVIEW_DELETED');require(type(v['favorite']) is bool and v['status'] in STATUSES)
                note=v['note'];require(isinstance(note,str) and len(note)<=3000 and all(ord(x)>=32 or x in '\n\r\t' for x in note))
                favorite=v['favorite'];status=v['status'];deleted=False
            else:
                require(old['revision']>0 and old['deleted']==(action=='restore'),'INVALID_REVIEW_TRANSITION');note=old['note'];favorite=old['favorite'] if action=='restore' else False;status=old['status'];deleted=action=='delete'
            db.execute('INSERT OR REPLACE INTO reviews71 VALUES(?,?,?,?,?,?,?,?,?,?)',[subject,self.binding['model_sha256'],json.dumps(c['case_key']),c['case_revision'],old['revision']+1,int(favorite),status,note.replace('\r\n','\n').replace('\r','\n'),int(deleted),now()])
            return self.read(subject,c,db)
    def listing(self,deleted=False):
        with closing(self.connect()) as db:
            return {'items':[{'subject':r['subject'],'case_key':json.loads(r['case_key']),'revision':r['revision'],'favorite':bool(r['favorite']),'status':r['status'],'updated_at':r['updated']} for r in db.execute('SELECT * FROM reviews71 WHERE model=? AND deleted=? ORDER BY updated DESC LIMIT 100',[self.binding['model_sha256'],int(deleted)])]}
    def create_report(self,subject,c,v):
        from .report71 import render
        from .reports import esc
        self.validate(c);integer(v['expected_revision'],0);require(v['expected_case_revision']==c['case_revision'],'VERSION_MISMATCH')
        with self.lock,closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');review=self.read(subject,c,db);require(review['revision']==v['expected_revision'] and review['binding_matches'] and not review['deleted'],'VERSION_MISMATCH')
            created=now();ident=uuid.uuid4().hex;snapshot={'report_version':'workspace71_fixed_report_v1','created_at':created,'result':c,'review':review}
            raw=json.dumps(snapshot,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);html=render(snapshot)
            status=dict(zip(STATUSES,['未复核','待复核','资料待补','已复核']))[review['status']]
            html=html.replace('<footer',f'<h2>本次复核记录</h2><p>状态：{esc(status)}；收藏：{"是" if review["favorite"] else "否"}</p><div style="white-space:pre-wrap;overflow-wrap:anywhere">{esc(review["note"] or "暂无备注")}</div><footer',1)
            require(max(len(raw.encode()),len(html.encode()))<=2*1024*1024,'REPORT_QUOTA')
            n,size=db.execute('SELECT count(*),COALESCE(sum(length(CAST(html AS BLOB))+length(CAST(data AS BLOB))),0) FROM reports71').fetchone();require(n<1000 and size+len(raw.encode())+len(html.encode())<=100*1024*1024,'REPORT_QUOTA')
            db.execute('INSERT INTO reports71 VALUES(?,?,?,?,?,?,?,?)',[ident,subject,self.binding['model_sha256'],created,html,raw,hashlib.sha256(html.encode()).hexdigest(),hashlib.sha256(raw.encode()).hexdigest()])
            return self.report_info(ident,subject,created)
    @staticmethod
    def report_info(ident,subject,created):return {'report_id':ident,'subject':subject,'created_at':created,'view_url':f'/api/v2/candidate71/reports/{ident}/view','download_url':f'/api/v2/candidate71/reports/{ident}/download','data_url':f'/api/v2/candidate71/reports/{ident}/data'}
    def reports(self):
        with closing(self.connect()) as db:return {'items':[self.report_info(*r) for r in db.execute('SELECT id,subject,created FROM reports71 WHERE model=? ORDER BY created DESC LIMIT 100',[self.binding['model_sha256']])]}
    def report(self,ident,section):
        require(isinstance(ident,str) and len(ident)==32 and all(x in '0123456789abcdef' for x in ident));require(section in ['view','download','data'])
        with closing(self.connect()) as db:r=db.execute('SELECT * FROM reports71 WHERE id=? AND model=?',[ident,self.binding['model_sha256']]).fetchone()
        require(r is not None,'REPORT_NOT_FOUND');name='data' if section=='data' else 'html';raw=r[name].encode();require(hashlib.sha256(raw).hexdigest()==r[name+'_sha'],'REPORT_CORRUPT');return raw

class Catalog71:
    def __init__(self,path,receipt,workflow):
        self.path=Path(path);self.workflow=workflow;self.receipt=receipt;require(sha(path)==receipt['sha256'],'VERSION_MISMATCH')
        with closing(self.connect()) as db:require(json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])=={k:v for k,v in receipt.items() if k not in ['sha256','file']},'VERSION_MISMATCH')
    def connect(self):
        db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('ATTACH DATABASE ? AS work',[self.workflow.path.as_uri()+'?mode=ro']);return db
    def query(self,q):
        pop=q.get('population','labeled');use=q.get('sample_use','all');require(pop in ['labeled','unlabeled']);require(use in (['all','train','validation'] if pop=='labeled' else ['all','unlabeled_scoring']))
        try:lo=float(q.get('minimum','0'));hi=float(q.get('maximum','1'));offset=int(q.get('offset','0'));limit=int(q.get('limit','20'))
        except (ValueError,TypeError):require(False)
        require(math.isfinite(lo) and math.isfinite(hi) and 0<=lo<=hi<=1 and 0<=offset<=10**7 and limit in [10,20,50])
        order=q.get('order','id_asc');require(order in ['id_asc','id_desc','probability_asc','probability_desc']);qual=q.get('quality','all');hist=q.get('history','all');mark=q.get('mark','all');require(qual in ['all','missing','complete'] and hist in ['all','present','none'] and mark in ['all','favorite','pending'])
        where=['c.population=?','c.probability>=?','c.probability<=?'];args=[pop,lo,hi]
        if use!='all':where+=['c.sample_use=?'];args+=[use]
        if qual!='all':where+=['c.missing>0' if qual=='missing' else 'c.missing=0']
        if hist!='all':where+=['c.history_state=?'];args+=[hist]
        if mark!='all':where+=['COALESCE(r.deleted,0)=0',"r.favorite=1" if mark=='favorite' else "r.status IN ('pending','needs_information')"]
        join=" FROM cases c LEFT JOIN work.reviews71 r ON r.subject='case:'||c.population||':'||c.id AND r.model=? WHERE "+' AND '.join(where);params=[self.receipt['model_sha256'],*args]
        sorting={'id_asc':'c.id ASC','id_desc':'c.id DESC','probability_asc':'c.probability ASC,c.id ASC','probability_desc':'c.probability DESC,c.id ASC'}[order]
        with closing(self.connect()) as db:
            db.execute('BEGIN');total=db.execute('SELECT count(*)'+join,params).fetchone()[0]
            rows=db.execute('SELECT c.*,COALESCE(r.favorite,0) favorite,COALESCE(r.deleted,0) deleted,COALESCE(r.status,\'unreviewed\') review_status'+join+' ORDER BY '+sorting+' LIMIT ? OFFSET ?',(*params,limit,offset)).fetchall()
            bins=[0]*10
            for bucket,count in db.execute('SELECT min(CAST(c.probability*10 AS INTEGER),9),count(*)'+join+' GROUP BY 1',params):bins[bucket]=count
            items=[{k:r[k] for k in ['population','id','sample_use','probability','observed','missing','no_history','history_state','review_status']}|{'favorite':bool(r['favorite'] and not r['deleted']),'history_counts':json.loads(r['counts_json'])} for r in rows]
        return {'version':'workspace71_catalog_v1','model_sha256':self.receipt['model_sha256'],'deployment_sha256':self.receipt['deployment_sha256'],'feature_count':71,'threshold':self.receipt['threshold'],'total':total,'offset':offset,'limit':limit,'next_offset':offset+limit if offset+limit<total else None,'items':items,'probability_bins':bins,'filters':dict(q)}
