"""Dedicated 71 model namespace and independent raw-input draft storage."""
import hashlib
import uuid
import json
from datetime import datetime,timezone
from pathlib import Path
from fastapi import Request
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool
from .contracts import require
from .api import params, strict_json, envelope, number
from .candidate71_contract import blank, check, complete, MAX_BYTES, TABLES, canonical, digest
from .input_store import InputStore
from .report71 import render as render_report


def register(app,client,spool,database,history=None):
    spool=Path(spool);spool.mkdir(exist_ok=False);info=client.ready['candidate71']
    binding={k:info[k] for k in ['model_sha256','deployment_sha256','input_contract_version']}
    store=InputStore(database,binding,validator=check,max_bytes=MAX_BYTES)
    from .workspace71 import Workflow71,Catalog71
    workflow=Workflow71(database,binding)
    catalog_receipt=client.ready.get('candidate71_catalog')
    catalog=Catalog71(spool.parent/catalog_receipt['file'],catalog_receipt,workflow) if catalog_receipt else None
    def reply(request,value):return envelope(request.state.request_id,value,token=request.state.selection_token)
    def execute(v):return client.exchange(v,v)['payload']
    def current(kind,first,second=None):
        if kind=='case':
            require(first in ['labeled','unlabeled']);result=execute({'op':'candidate71','action':'cases','population':first,'ids':[number(second)]})['results'][0];return f'case:{first}:{number(second)}',result
        d=store.read(first);require(d['score'] is not None and not d['score_stale'],'VERSION_MISMATCH');return 'draft:'+first,d['score']
    def analysis(request,result):
        require(request.query_params.get('revision')==result['case_revision'],'VERSION_MISMATCH')
        fmt=request.query_params.get('format','json');download=request.query_params.get('download','1');require(fmt in ['html','json'] and download in ['0','1'])
        snapshot={'report_version':'candidate71_analysis_snapshot_v1','created_at':datetime.now(timezone.utc).isoformat(),'scope':'本机研究分析快照；非真实审批','result':result}
        raw=render_report(snapshot).encode('utf-8') if fmt=='html' else json.dumps(snapshot,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
        filename=f'credit-risk-{"report" if fmt=="html" else "data"}-{result["case_key"]["application_id"]}-{result["case_revision"][:12]}.{fmt}'
        return Response(raw,media_type='text/html' if fmt=='html' else 'application/json',headers={'Content-Disposition':f'{"attachment" if download=="1" else "inline"}; filename="{filename}"'})
    async def body(request,keys):
        params(request,set());require(request.headers.get('content-type','').split(';')[0]=='application/json')
        require(request.headers.get('origin') in [None,'http://'+request.headers['host']])
        v=strict_json(await request.body());require(isinstance(v,dict) and set(v)==keys);return v
    def score(payloads):
        require(isinstance(payloads,list) and 1<=len(payloads)<=5)
        for p in payloads:complete(p)
        raw=canonical({'payloads':payloads});require(len(raw)<=250000,'INPUT_SIZE')
        ident=uuid.uuid4().hex;path=spool/(ident+'.json')
        with path.open('xb') as f:f.write(raw)
        try:return execute({'op':'candidate71','action':'raw','ticket':ident,'sha256':hashlib.sha256(raw).hexdigest()})
        finally:path.unlink(missing_ok=True)
    @app.get('/api/v2/candidate71/info')
    def metadata(request:Request):params(request,set());return reply(request,{**info,'template':blank(),'raw_history_columns':TABLES})
    @app.get('/api/v2/candidate71/cases/{pop}/{ident}')
    def case(request:Request,pop:str,ident:str):
        params(request,set());return reply(request,execute({'op':'candidate71','action':'cases','population':pop,'ids':[number(ident)]})['results'][0])
    @app.get('/api/v2/candidate71/examples/{ident}')
    def example(request:Request,ident:str):params(request,set());return reply(request,execute({'op':'candidate71','action':'example','id':number(ident)})['payload'])
    @app.get('/api/v2/candidate71/cases/{pop}/{ident}/analysis')
    def case_analysis(request:Request,pop:str,ident:str):
        params(request,{'revision','format','download'});result=execute({'op':'candidate71','action':'cases','population':pop,'ids':[number(ident)]})['results'][0]
        return analysis(request,result)
    @app.get('/api/v2/candidate71/drafts/{ident}/analysis')
    def draft_analysis(request:Request,ident:str):
        params(request,{'revision','format','download'});d=store.read(ident);require(d['score'] is not None and not d['score_stale'],'VERSION_MISMATCH')
        return analysis(request,d['score'])
    @app.post('/api/v2/candidate71/score-batch')
    async def batch(request:Request):
        v=await body(request,{'payloads'});return reply(request,await run_in_threadpool(score,v['payloads']))
    @app.get('/api/v2/candidate71/drafts')
    def listing(request:Request):params(request,set());return reply(request,store.listing())
    @app.get('/api/v2/candidate71/drafts/{ident}')
    def draft(request:Request,ident:str):params(request,set());return reply(request,store.read(ident))
    @app.post('/api/v2/candidate71/validate')
    async def validate(request:Request):v=await body(request,{'payload'});return reply(request,check(v['payload']))
    @app.post('/api/v2/candidate71/drafts')
    async def save(request:Request):
        v=await body(request,{'draft_id','expected_revision','title','payload'})
        return reply(request,await run_in_threadpool(store.save,v['draft_id'],v['expected_revision'],v['title'],v['payload']))
    @app.post('/api/v2/candidate71/drafts/{ident}/score')
    async def score_draft(request:Request,ident:str):
        v=await body(request,{'expected_revision','expected_payload_sha256'})
        def apply():
            d=store.read(ident);require(d['revision']==v['expected_revision'] and d['payload_sha256']==v['expected_payload_sha256'],'VERSION_MISMATCH')
            result=score([d['payload']])['results'][0]
            return store.attach_score(ident,d['revision'],d['payload_sha256'],result)
        return reply(request,await run_in_threadpool(apply))
    @app.get('/api/v2/candidate71/catalog')
    def collection(request:Request):
        q=params(request,{'population','sample_use','minimum','maximum','offset','limit','order','quality','history','mark'});require(catalog is not None,'COLLECTION_NOT_CONNECTED');return reply(request,catalog.query(q))
    @app.get('/api/v2/candidate71/cases/{pop}/{ident}/history/{table}')
    def source_rows(request:Request,pop:str,ident:str,table:str):
        q=params(request,{'revision','selection','offset','limit'});require(history is not None,'HISTORY_INDEX_UNAVAILABLE');_,c=current('case',pop,ident);require(q.get('revision')==c['case_revision'],'VERSION_MISMATCH');require(table in ['bureau','previous'])
        h=history.query(pop,number(ident),table,selection=q.get('selection','all'),offset=number(q.get('offset','0'),0,10**7),limit=number(q.get('limit','10'),1,50))
        values={f['name']:f['value'] for f in c['facts']};require(all(values[n]==v for n,v in h['summary_values'].items()),'VERSION_MISMATCH')
        return reply(request,{'case_key':c['case_key'],'case_revision':c['case_revision'],'model_sha256':c['model_sha256'],'table':table,'audit':h['audit'],'items':h['items'],'total':h['total'],'offset':h['offset'],'next_offset':h['next_offset']})
    @app.get('/api/v2/candidate71/reviews')
    def reviews(request:Request):
        q=params(request,{'deleted'});require(q.get('deleted','false') in ['true','false']);return reply(request,workflow.listing(q.get('deleted')=='true'))
    @app.get('/api/v2/candidate71/reports')
    def reports(request:Request):params(request,set());return reply(request,workflow.reports())
    @app.get('/api/v2/candidate71/reports/{ident}/{section}')
    def fixed_report(request:Request,ident:str,section:str):
        params(request,set());raw=workflow.report(ident,section);return Response(raw,media_type='application/json' if section=='data' else 'text/html',headers={'Content-Disposition':f'{"attachment" if section in ["download","data"] else "inline"}; filename="risk-analysis-{ident}.{"json" if section=="data" else "html"}"'})
    def routes(kind,base):
        # Separate paths preserve draft identities even when application numbers coincide.
        def resolve(request):
            d=request.path_params;return current(kind,d['pop'],d['ident']) if kind=='case' else current(kind,d['ident'])
        def read_review(request:Request):
            q=params(request,{'revision'});subject,c=resolve(request);require(q.get('revision')==c['case_revision'],'VERSION_MISMATCH');return reply(request,workflow.read(subject,c))
        async def write_review(request:Request):
            v=await body(request,{'expected_case_revision','expected_revision','favorite','status','note'})
            def apply():
                subject,c=resolve(request);return workflow.write(subject,c,v)
            return reply(request,await run_in_threadpool(apply))
        async def action_review(request:Request):
            action=request.path_params['action']
            require(action in ['delete','restore']);v=await body(request,{'expected_case_revision','expected_revision'})
            def apply():
                subject,c=resolve(request);return workflow.write(subject,c,v,action)
            return reply(request,await run_in_threadpool(apply))
        async def save_report(request:Request):
            v=await body(request,{'expected_case_revision','expected_revision'})
            def apply():
                subject,c=resolve(request);return workflow.create_report(subject,c,v)
            return reply(request,await run_in_threadpool(apply))
        app.add_api_route(base+'/review',read_review,methods=['GET']);app.add_api_route(base+'/review',write_review,methods=['POST']);app.add_api_route(base+'/review/{action}',action_review,methods=['POST']);app.add_api_route(base+'/report',save_report,methods=['POST'])
    routes('case','/api/v2/candidate71/cases/{pop}/{ident}');routes('draft','/api/v2/candidate71/drafts/{ident}')
