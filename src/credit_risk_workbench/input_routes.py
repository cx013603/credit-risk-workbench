"""Draft and complete raw scoring endpoints, added without changing old case identity."""
import hashlib
import json
import uuid
from pathlib import Path
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from .contracts import require
from .input_contract import VERSION,SEAL,MAIN,NUMERIC,BUREAU,PREVIOUS,blank,canonical,digest,check,require_complete,opaque_id
from .api import params,strict_json,envelope

def register(app,client,root,spool,store,dictionary):
    spool=Path(spool);spool.mkdir(exist_ok=False)
    def reply(request,value):return envelope(request.state.request_id,value,token=request.state.selection_token)
    async def body(request,keys):
        params(request,set());require(request.headers.get('content-type','').split(';')[0].strip()=='application/json')
        require(request.headers.get('origin') in [None,'http://'+request.headers['host']])
        v=strict_json(await request.body());require(isinstance(v,dict) and set(v)==keys);return v
    def score(payloads):
        require(isinstance(payloads,list) and 1<=len(payloads)<=5)
        for p in payloads:require_complete(p)
        data={'contract_version':VERSION,'payloads':payloads};raw=canonical(data);require(len(raw)<=16000,'INPUT_SIZE')
        ident=uuid.uuid4().hex;path=spool/(ident+'.json')
        with path.open('xb') as f:f.write(raw)
        message={'op':'new_inputs','ticket':ident,'sha256':hashlib.sha256(raw).hexdigest()}
        try:return client.exchange(message,message)['payload']
        finally:path.unlink(missing_ok=True)
    @app.get('/api/v2/new-inputs/contract')
    def contract(request:Request):
        params(request,set())
        return reply(request,{'version':VERSION,'release_manifest_sha256':SEAL,'model':'tree_v1','feature_count':40,'template':blank(),'fields':dictionary['fields'],'main_fields':[{**f,'nullable':True} for f in dictionary['fields'] if f['feature'] in MAIN],'numeric_fields':sorted(NUMERIC),'bureau_columns':BUREAU,'previous_columns':PREVIOUS,'max_history_rows':100,'max_payload_bytes':15000,'batch_limit':5,'history_declarations':['provided','no_history','not_provided'],'supported_source_kinds':['home_credit_source_compatible','synthetic_source_encoded'],'required_units':{'monetary_unit':'home_credit_source_scale','time_reference':'current_application_day_zero','snapshot_availability':'source_snapshot_declared'}})
    @app.get('/api/v2/new-inputs/drafts')
    def listing(request:Request):params(request,set());return reply(request,store.listing())
    @app.get('/api/v2/new-inputs/drafts/{ident}')
    def read(request:Request,ident:str):params(request,set());return reply(request,store.read(ident))
    @app.post('/api/v2/new-inputs/validate')
    async def validate(request:Request):v=await body(request,{'payload'});return reply(request,check(v['payload']))
    @app.post('/api/v2/new-inputs/drafts')
    async def save(request:Request):
        v=await body(request,{'draft_id','expected_revision','title','payload'});require(v['draft_id'] is None or isinstance(v['draft_id'],str))
        result=await run_in_threadpool(store.save,v['draft_id'],v['expected_revision'],v['title'],v['payload']);return reply(request,result)
    @app.post('/api/v2/new-inputs/score-batch')
    async def batch(request:Request):
        v=await body(request,{'payloads','expected_contract_version'});require(v['expected_contract_version']==VERSION,'VERSION_MISMATCH')
        return reply(request,await run_in_threadpool(score,v['payloads']))
    @app.post('/api/v2/new-inputs/drafts/{ident}/score')
    async def score_draft(request:Request,ident:str):
        v=await body(request,{'expected_revision','expected_payload_sha256'})
        def apply():
            d=store.read(ident);require(d['revision']==v['expected_revision'] and d['payload_sha256']==v['expected_payload_sha256'],'VERSION_MISMATCH')
            result=score([d['payload']])['results'][0]
            return store.attach_score(ident,d['revision'],d['payload_sha256'],result)
        return reply(request,await run_in_threadpool(apply))
