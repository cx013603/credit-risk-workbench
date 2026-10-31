"""Saved-score catalog, deterministic quality semantics and local simulation marks.

No labels, fitting or performance metrics. SQLite catalog is built by the frozen
worker from its accepted saved inputs/scores, then queried by the API process.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timezone
from contextlib import closing, contextmanager
from collections import OrderedDict
from .contracts import require, integer, population, revision

VERSION = 'workbench_collection_v1'
HISTORY = ['both', 'bureau_only', 'previous_only', 'none']
ORDERS = ['id_asc', 'id_desc', 'probability_asc', 'probability_desc']
SIGNALS = {'BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT': '币种1金额不可用记录',
           'BUREAU_CURRENCY_MISSING_COUNT': '历史币种缺失记录',
           'BUREAU_OTHER_STATUS_COUNT': '其他或缺失征信状态',
           'PREV_OTHER_CONTRACT_COUNT': '其他或缺失历史合同类型'}

def canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def quality(values, fields):
    b=values['BUREAU_ELIGIBLE_COUNT'];p=values['PREV_ELIGIBLE_COUNT']
    missing=unavailable=bits=0;rows=[];signals=[]
    for i,f in enumerate(fields):
        name=f['feature'];v=values[name]
        state='no_eligible_history' if (f['group']=='bureau' and b==0 or f['group']=='previous' and p==0) else 'source_null' if v is None else 'observed'
        if state=='source_null':missing+=1;bits|=1<<i
        if state=='no_eligible_history':unavailable+=1
        signal=name in SIGNALS and v is not None and v>0
        if signal:signals.append(name)
        rows.append({'fact_id':name,'label':f['label'],'group':f['group'],'state':state,'signal':SIGNALS[name] if signal else None,'source':f['source'],'unit':f['unit'],'time_rule':f['time_rule']})
    observed=40-missing-unavailable
    return {'observed':observed,'source_null':missing,'no_eligible_history':unavailable,'missing_bits':bits,'signal_count':len(signals),'signals':signals,'bureau_count':b,'previous_count':p,'history_coverage':'both' if b>0 and p>0 else 'bureau_only' if b>0 else 'previous_only' if p>0 else 'none','completeness':observed/40,'fields':rows}

def build_catalog(loaded, dictionary, path, seal, digest):
    """Uses the accepted release's scores projection; never calls score_ids."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    path=Path(path);require(not path.exists())
    db=sqlite3.connect(path)
    counts={}
    try:
        db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE cases(population TEXT NOT NULL,id INTEGER NOT NULL,sample_use TEXT NOT NULL,probability REAL NOT NULL,bureau_count INTEGER NOT NULL,previous_count INTEGER NOT NULL,history_coverage TEXT NOT NULL,observed INTEGER NOT NULL,source_null INTEGER NOT NULL,no_eligible_history INTEGER NOT NULL,missing_bits INTEGER NOT NULL,signal_count INTEGER NOT NULL,PRIMARY KEY(population,id));')
        for pop,s in loaded.sources.items():
            scores=pq.read_table(s.directory/'scores.parquet',columns=['SK_ID_CURR','sample_use','risk_probability'])
            scores=scores.filter(pc.not_equal(scores['sample_use'],'test'))
            features=s.features.filter(pc.not_equal(s.features['sample_use'],'test'))
            require(features.num_rows==scores.num_rows,'VERSION_MISMATCH')
            scoremap={r['SK_ID_CURR']:r for r in scores.to_pylist()}
            def records():
                for batch in features.to_batches(max_chunksize=2048):
                    for row in batch.to_pylist():
                        ident=row['SK_ID_CURR'];score=scoremap.pop(ident);use=row['sample_use'];prob=score['risk_probability']
                        require(use in ['train','validation','unlabeled_scoring'] and score['sample_use']==use and type(prob) is float and math.isfinite(prob) and 0<=prob<=1,'VERSION_MISMATCH')
                        q=quality(row,dictionary['fields'])
                        yield (pop,ident,use,prob,*[q[k] for k in ['bureau_count','previous_count','history_coverage','observed','source_null','no_eligible_history','missing_bits','signal_count']])
            db.executemany('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',records())
            require(not scoremap,'VERSION_MISMATCH');counts[pop]=features.num_rows
        metadata={'version':VERSION,'release_manifest_sha256':seal,'dictionary_sha256':digest,'counts':counts,'fields':dictionary['fields'],'score_origin':'accepted_release_saved_scores_no_new_scoring'}
        db.execute('INSERT INTO meta VALUES(?,?)',('metadata',json.dumps(metadata,ensure_ascii=False)))
        db.executescript('CREATE INDEX case_probability ON cases(population,probability,id); CREATE INDEX case_use ON cases(population,sample_use,id);')
        db.commit()
    finally:db.close()
    return {'version':VERSION,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'counts':counts}

class MarkStore:
    """Version-bound simulation metadata. Optimistic revision prevents lost edits."""
    def __init__(self,path):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()
        with closing(sqlite3.connect(self.path)) as db,db:
            db.executescript('CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),revision INTEGER NOT NULL); INSERT OR IGNORE INTO state VALUES(1,0); CREATE TABLE IF NOT EXISTS marks(release TEXT NOT NULL,population TEXT NOT NULL,id INTEGER NOT NULL,favorite INTEGER NOT NULL CHECK(favorite IN(0,1)),pending INTEGER NOT NULL CHECK(pending IN(0,1)),updated_at TEXT NOT NULL,PRIMARY KEY(release,population,id));')
    def set(self,seal,pop,ident,favorite,pending,expected):
        require(type(favorite) is bool and type(pending) is bool);integer(expected,0)
        with self.lock,closing(sqlite3.connect(self.path,timeout=2)) as db,db:
            db.execute('BEGIN IMMEDIATE');current=db.execute('SELECT revision FROM state').fetchone()[0]
            require(current==expected,'VERSION_MISMATCH')
            stamp=datetime.now(timezone.utc).isoformat()
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_records'").fetchone():
                row=db.execute('SELECT deleted,review_status FROM review_records WHERE release=? AND population=? AND id=?',(seal,pop,ident)).fetchone()
                if row:
                    require(not row[0],'REVIEW_DELETED')
                    status=('needs_information' if row[1]=='needs_information' else 'pending') if pending else ('reviewed' if row[1]=='reviewed' else 'unreviewed')
                    db.execute('UPDATE review_records SET favorite=?,review_status=?,updated_at=? WHERE release=? AND population=? AND id=?',(int(favorite),status,stamp,seal,pop,ident))
            db.execute('INSERT OR REPLACE INTO marks VALUES(?,?,?,?,?,?)',(seal,pop,ident,int(favorite),int(pending),stamp))
            db.execute('UPDATE state SET revision=revision+1');return current+1

def filters(q):
    pop=population(q.get('population'));use=q.get('sample_use','all')
    require(use in (['all','train','validation'] if pop=='labeled' else ['all','unlabeled_scoring']),'SAMPLE_USE_NOT_ALLOWED' if use=='test' else 'INVALID_INPUT')
    def fraction(name,default):
        v=q.get(name,str(default));require(isinstance(v,str) and 1<=len(v)<=24)
        try:n=float(v)
        except ValueError:require(False)
        require(math.isfinite(n) and 0<=n<=1);return n
    lo=fraction('probability_min',0);hi=fraction('probability_max',1);require(lo<=hi)
    hist=q.get('history','all');missing=q.get('missing','all');signal=q.get('signal','all');mark=q.get('mark','all')
    require(hist in ['all',*HISTORY] and missing in ['all','present','none'] and signal in ['all','present','none'] and mark in ['all','favorite','pending'])
    return dict(population=pop,sample_use=use,probability_min=lo,probability_max=hi,history=hist,missing=missing,signal=signal,mark=mark)

class CatalogIndex:
    def __init__(self,path,receipt,seal,digest,marks):
        self.path=Path(path).resolve();self.seal=seal;self.digest=digest;self.marks=marks;self.cache=OrderedDict()
        require(hashlib.sha256(self.path.read_bytes()).hexdigest()==receipt['sha256'],'VERSION_MISMATCH')
        with self.connect() as db:self.meta=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('metadata',)).fetchone()[0])
        require(self.meta['version']==VERSION and self.meta['release_manifest_sha256']==seal and self.meta['dictionary_sha256']==digest and self.meta['counts']==receipt['counts'],'VERSION_MISMATCH')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=2);db.row_factory=sqlite3.Row
        try:yield db
        finally:db.close()
    def query(self,q,offset=0,limit=20,order='id_asc',current_id=None):
        f=filters(q);integer(offset,0,10**7);integer(limit,1,50);require(order in ORDERS)
        with self.marks.lock,self.connect() as db:
            db.execute('ATTACH DATABASE ? AS local',(str(self.marks.path),));db.execute('BEGIN')
            mr=db.execute('SELECT revision FROM local.state').fetchone()[0]
            args=[self.seal,f['population'],f['probability_min'],f['probability_max']]
            where='c.population=? AND c.probability>=? AND c.probability<=?'
            if f['sample_use']!='all':where+=' AND c.sample_use=?';args.append(f['sample_use'])
            if f['history']!='all':where+=' AND c.history_coverage=?';args.append(f['history'])
            for key,col in [('missing','source_null'),('signal','signal_count')]:
                if f[key]!='all':where+=f" AND c.{col}{'>0' if f[key]=='present' else '=0'}"
            if f['mark']!='all':where+=f" AND m.{f['mark']}=1"
            base=' FROM cases c LEFT JOIN local.marks m ON m.release=? AND m.population=c.population AND m.id=c.id WHERE '+where
            cache_key=canonical({'filters':f,'marks_revision':mr})
            summary=self.cache.get(cache_key)
            if summary is None:
                # Broad filters scan rows in physical order, avoiding scattered
                # probability-index lookups. All aggregates share one scan.
                statsbase=base.replace('FROM cases c LEFT JOIN','FROM cases c NOT INDEXED LEFT JOIN') if f['mark']!='all' else ' FROM cases c NOT INDEXED WHERE '+where
                statsargs=args if f['mark']!='all' else args[1:]
                expressions=['COUNT(*) AS total']
                expressions += [f"SUM(CASE WHEN min(9,CAST(c.probability*10 AS INTEGER))={i} THEN 1 ELSE 0 END) AS p{i}" for i in range(10)]
                expressions += [f"SUM(CASE WHEN c.history_coverage='{k}' THEN 1 ELSE 0 END) AS h{i}" for i,k in enumerate(HISTORY)]
                expressions += [f'SUM((c.missing_bits >> {i}) & 1) AS n{i}' for i in range(40)]
                stats=db.execute('SELECT '+','.join(expressions)+statsbase,statsargs).fetchone()
                summary={'total':stats['total'],'probability_bins':[dict(lower=i/10,upper=(i+1)/10,count=stats[f'p{i}'] or 0) for i in range(10)],'history_coverage':{k:stats[f'h{i}'] or 0 for i,k in enumerate(HISTORY)},'field_missing':[{'fact_id':field['feature'],'label':field['label'],'source_null':stats[f'n{i}'] or 0} for i,field in enumerate(self.meta['fields'])]}
                self.cache[cache_key]=summary
                while len(self.cache)>32:self.cache.popitem(last=False)
            else:self.cache.move_to_end(cache_key)
            count=summary['total']
            sort={'id_asc':'c.id ASC','id_desc':'c.id DESC','probability_asc':'c.probability ASC,c.id ASC','probability_desc':'c.probability DESC,c.id ASC'}[order]
            rowbase=base.replace('FROM cases c LEFT JOIN', 'FROM cases c INDEXED BY '+('sqlite_autoindex_cases_1' if order.startswith('id_') else 'case_probability')+' LEFT JOIN')
            rows=db.execute('SELECT c.*,COALESCE(m.favorite,0) favorite,COALESCE(m.pending,0) pending'+rowbase+' ORDER BY '+sort+' LIMIT ? OFFSET ?',[*args,limit,offset]).fetchall()
            items=[]
            for row in rows:
                item=dict(row);item['application_id']=item.pop('id');item.pop('missing_bits');item['favorite']=bool(item['favorite']);item['pending']=bool(item['pending']);item['completeness']=item['observed']/40;item['case_revision']=revision(self.seal,self.digest,item['population'],item['application_id']);items.append(item)
            member=None if current_id is None else bool(db.execute('SELECT 1'+base+' AND c.id=?',[*args,current_id]).fetchone())
        identity={'version':VERSION,'filters':f,'release_manifest_sha256':self.seal,'dictionary_sha256':self.digest,'marks_revision':mr}
        return {**identity,'collection_revision':canonical(identity),'total':count,'offset':offset,'limit':limit,'order':order,'items':items,'next_offset':offset+limit if offset+limit<count else None,'current_in_collection':member,'summary':summary,'scope':'saved_predictions_descriptive_only_no_labels_or_effect_metrics'}
    def mark(self,pop,ident,favorite,pending,expected):
        population(pop);integer(ident)
        with self.connect() as db:require(db.execute('SELECT 1 FROM cases WHERE population=? AND id=?',(pop,ident)).fetchone(),'CASE_NOT_FOUND')
        return self.marks.set(self.seal,pop,ident,favorite,pending,expected)
