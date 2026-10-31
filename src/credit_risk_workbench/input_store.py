"""Separate SQLite draft store; optimistic revisions and immutable result bindings."""
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from .contracts import require
from .input_contract import canonical,check,digest,opaque_id,MAX_BYTES

class InputStore:
    def __init__(self,path,binding=None,validator=check,max_bytes=MAX_BYTES):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.binding=binding or {};self.validator=validator;self.max_bytes=max_bytes
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS drafts (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, title TEXT NOT NULL, payload TEXT NOT NULL, updated TEXT NOT NULL, result TEXT)')
            require(c.execute('PRAGMA user_version').fetchone()[0] in [0,1],'DRAFT_SCHEMA_INVALID')
            c.execute('PRAGMA user_version=1')
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=3)
        try:
            with c:yield c
        finally:c.close()
    def listing(self):
        with self.connect() as c:rows=c.execute('SELECT id,revision,title,updated,result IS NOT NULL FROM drafts ORDER BY updated DESC,id LIMIT 100').fetchall()
        return {'items':[dict(zip(['draft_id','revision','title','updated_at','has_score'],r)) for r in rows],'limit':100}
    def read(self,ident):
        opaque_id(ident)
        with self.connect() as c:r=c.execute('SELECT revision,title,payload,updated,result FROM drafts WHERE id=?',[ident]).fetchone()
        require(r is not None,'DRAFT_NOT_FOUND');payload=json.loads(r[2]);validation=self.validator(payload)
        score=json.loads(r[4]) if r[4] else None;stale=score is not None and any(score.get(k)!=v for k,v in self.binding.items())
        return {'draft_id':ident,'revision':r[0],'title':r[1],'payload':payload,'updated_at':r[3],'payload_sha256':digest(payload),'validation':validation,'score':None if stale else score,'score_stale':stale}
    def save(self,ident,expected,title,payload):
        require(type(expected) is int and expected>=0,'INVALID_INPUT')
        require(isinstance(title,str) and 1<=len(title.strip())<=80 and not any(ord(c)<32 for c in title),'INVALID_INPUT')
        require(isinstance(payload,dict),'INVALID_INPUT')
        try:raw=canonical(payload)
        except (ValueError,TypeError,OverflowError,RecursionError):require(False,'INVALID_INPUT')
        require(len(raw)<=self.max_bytes,'INPUT_SIZE');ident=opaque_id(ident) if ident is not None else uuid.uuid4().hex
        stamp=datetime.now(timezone.utc).isoformat()
        with self.lock,self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            r=c.execute('SELECT revision FROM drafts WHERE id=?',[ident]).fetchone()
            require((r is None and expected==0) or (r is not None and r[0]==expected),'VERSION_MISMATCH')
            require(r is not None or c.execute('SELECT count(*) FROM drafts').fetchone()[0]<500,'DRAFT_QUOTA')
            c.execute('INSERT INTO drafts VALUES (?,?,?,?,?,NULL) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,title=excluded.title,payload=excluded.payload,updated=excluded.updated,result=NULL',[ident,expected+1,title,raw.decode('utf-8'),stamp])
        return self.read(ident)
    def attach_score(self,ident,expected,payload_sha,result):
        with self.lock,self.connect() as c:
            c.execute('BEGIN IMMEDIATE');r=c.execute('SELECT revision,payload FROM drafts WHERE id=?',[opaque_id(ident)]).fetchone()
            require(r is not None and r[0]==expected and digest(json.loads(r[1]))==payload_sha,'VERSION_MISMATCH')
            require(result.get('input_sha256')==payload_sha,'VERSION_MISMATCH')
            require(all(result.get(k)==v for k,v in self.binding.items()),'VERSION_MISMATCH')
            c.execute('UPDATE drafts SET result=? WHERE id=?',[canonical(result).decode('utf-8'),ident])
        return self.read(ident)
