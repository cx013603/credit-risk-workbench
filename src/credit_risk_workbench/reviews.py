"""Local simulation records; transactional, optimistic and recoverable."""
from contextlib import closing
from datetime import datetime,timezone
import json
from .contracts import require,integer

STATUSES=['unreviewed','pending','needs_information','reviewed']
SCHEMA=2
def now():return datetime.now(timezone.utc).isoformat()
def note_text(v):
    require(isinstance(v,str) and len(v)<=3000 and all(ord(c)>=32 or c in '\n\r\t' for c in v))
    return v.replace('\r\n','\n').replace('\r','\n')

class ReviewStore:
    def __init__(self,marks):
        self.marks=marks
        with marks.lock,closing(self.connect()) as db,db:
            version=db.execute('PRAGMA user_version').fetchone()[0];require(version in [0,SCHEMA],'APP_DATA_VERSION_MISMATCH')
            db.execute('CREATE TABLE IF NOT EXISTS review_records(release TEXT NOT NULL,population TEXT NOT NULL,id INTEGER NOT NULL,case_revision TEXT NOT NULL,versions_json TEXT NOT NULL,sample_use TEXT NOT NULL,note TEXT NOT NULL,review_status TEXT NOT NULL CHECK(review_status IN (\'unreviewed\',\'pending\',\'needs_information\',\'reviewed\')),favorite INTEGER NOT NULL CHECK(favorite IN(0,1)),deleted INTEGER NOT NULL CHECK(deleted IN(0,1)),updated_at TEXT NOT NULL,PRIMARY KEY(release,population,id))')
            db.execute('CREATE TABLE IF NOT EXISTS reports(report_id TEXT PRIMARY KEY,release TEXT NOT NULL,population TEXT NOT NULL,id INTEGER NOT NULL,created_at TEXT NOT NULL,marks_revision INTEGER NOT NULL,json_sha256 TEXT NOT NULL,html_sha256 TEXT NOT NULL)')
            db.execute(f'PRAGMA user_version={SCHEMA}')
    def connect(self):
        import sqlite3
        db=sqlite3.connect(self.marks.path,timeout=2);db.row_factory=sqlite3.Row;return db
    def read(self,c,db=None):
        if db is None:
            with self.marks.lock,closing(self.connect()) as con:
                con.execute('BEGIN');return self.read(c,con)
        k=c['case_key'];seal=c['versions']['release_manifest_sha256'];key=(seal,k['population'],k['application_id'])
        row=db.execute('SELECT * FROM review_records WHERE release=? AND population=? AND id=?',key).fetchone();m=db.execute('SELECT * FROM marks WHERE release=? AND population=? AND id=?',key).fetchone()
        rev=db.execute('SELECT revision FROM state').fetchone()[0]
        status=row['review_status'] if row else 'pending' if m and m['pending'] else 'unreviewed'
        return {'case_key':k,'case_revision':c['case_revision'],'marks_revision':rev,'exists':bool(row or m and (m['favorite'] or m['pending'])),'favorite':bool(row['favorite']) if row and row['deleted'] else bool(m and m['favorite']),'pending':bool(m and m['pending']),'review_status':status,'note':row['note'] if row else '', 'deleted':bool(row and row['deleted']),'updated_at':row['updated_at'] if row else m['updated_at'] if m else None,'saved_case_revision':row['case_revision'] if row else c['case_revision'],'binding_matches':not row or row['case_revision']==c['case_revision'],'versions':json.loads(row['versions_json']) if row else c['versions'],'sample_use':row['sample_use'] if row else c['sample_use'],'scope':'local_simulated_analysis_only'}
    def write(self,c,expected,action='save',favorite=False,status='unreviewed',note=''):
        integer(expected,0);require(action in ['save','delete','restore']);require(type(favorite) is bool and status in STATUSES);note=note_text(note)
        k=c['case_key'];seal=c['versions']['release_manifest_sha256'];key=(seal,k['population'],k['application_id']);stamp=now()
        with self.marks.lock,closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');current=self.read(c,db);require(current['marks_revision']==expected,'VERSION_MISMATCH');require(current['binding_matches'],'VERSION_MISMATCH')
            if action=='save':require(not current['deleted'],'REVIEW_DELETED')
            else:
                require(current['exists'],'REVIEW_NOT_FOUND');require(current['deleted']==(action=='restore'),'INVALID_REVIEW_TRANSITION')
                note=current['note'];status=current['review_status'];favorite=current['favorite']
            deleted=action=='delete'
            # Trash retains the note/status; active flags are cleared. Restore
            # restores status (including pending), favorite may be reselected.
            pending=not deleted and status in ['pending','needs_information']
            db.execute('INSERT OR REPLACE INTO review_records VALUES(?,?,?,?,?,?,?,?,?,?,?)',(*key,c['case_revision'],json.dumps(c['versions'],ensure_ascii=False),c['sample_use'],note,status,int(favorite),int(deleted),stamp))
            db.execute('INSERT OR REPLACE INTO marks VALUES(?,?,?,?,?,?)',(*key,int(favorite and not deleted),int(pending),stamp))
            db.execute('UPDATE state SET revision=revision+1');return self.read(c,db)
    def listing(self,seal,offset=0,limit=20,deleted=False):
        integer(offset,0,10**7);integer(limit,1,50);require(type(deleted) is bool)
        with self.marks.lock,closing(self.connect()) as db:
            db.execute('BEGIN');revision=db.execute('SELECT revision FROM state').fetchone()[0]
            where='m.release=? AND COALESCE(r.deleted,0)=? AND (r.id IS NOT NULL OR m.favorite=1 OR m.pending=1)'
            base=' FROM marks m LEFT JOIN review_records r ON r.release=m.release AND r.population=m.population AND r.id=m.id WHERE '+where
            args=(seal,int(deleted));total=db.execute('SELECT COUNT(*)'+base,args).fetchone()[0]
            rows=db.execute("SELECT m.population,m.id application_id,m.favorite,m.pending,COALESCE(r.review_status,CASE WHEN m.pending=1 THEN 'pending' ELSE 'unreviewed' END) review_status,COALESCE(r.updated_at,m.updated_at) updated_at"+base+' ORDER BY updated_at DESC,m.population,m.id LIMIT ? OFFSET ?',(*args,limit,offset)).fetchall()
            return {'items':[{**dict(r),'favorite':bool(r['favorite']),'pending':bool(r['pending'])} for r in rows],'total':total,'offset':offset,'limit':limit,'deleted':deleted,'marks_revision':revision,'release_manifest_sha256':seal}
