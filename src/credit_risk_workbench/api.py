"""Versioned, bounded workbench API; no model/data imports in API environment."""
import asyncio
import json
import uuid
import sqlite3
from pathlib import Path
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from .contracts import SCHEMA,MAX_BATCH,MAX_PAGE,ContractError,require,identifiers,population
from .ipc import WorkerFailure
from .history_index import HistoryError
from .narratives import NarrativeError
from .reference import ReferenceError

CSP="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
MESSAGES={'INVALID_INPUT':'参数不符合接口约定。','INVALID_POPULATION':'请选择正确的申请来源。','INPUT_SIZE':'请求超出允许大小。','CASE_NOT_FOUND':'当前来源未找到申请。','SAMPLE_USE_NOT_ALLOWED':'原保留测试样本不进入当前新推理入口。','VERSION_MISMATCH':'请求版本与当前绑定版本不一致。','BUSY':'评分进程正在处理请求，请稍后重试。','WORKER_TIMEOUT':'评分进程超时，当前实例已停止接收评分。','WORKER_EXITED':'评分进程已退出。','WORKER_UNAVAILABLE':'评分进程不可用。','METHOD_NOT_ALLOWED':'该接口不支持此方法。','NOT_FOUND':'接口或静态资源不存在。'}
MESSAGES.update({'INPUT_INCOMPLETE':'资料尚未满足完整契约，请先补齐并核对校验问题。','INVALID_DRAFT_ID':'草稿编号无效。','DRAFT_NOT_FOUND':'未找到此草稿。','DRAFT_QUOTA':'草稿数量已达到本机上限。','INPUT_DUPLICATE_HISTORY_KEY':'批量资料出现重复历史主键。','DRAFT_SCHEMA_INVALID':'草稿存储版本不兼容。','LOCAL_STORAGE_UNAVAILABLE':'本机存储不可用，未确认操作成功。'})
def envelope(rid,payload=None,error=None,token=None):
    return {'schema_version':SCHEMA,'request_id':rid,'selection_token':token,'status':'failed' if error else 'ready','payload':payload,'error':error}
def failure(rid,code,status,token=None):
    return JSONResponse(envelope(rid,error={'code':code,'message':MESSAGES.get(code,'当前请求未能完成，请查看本轮运行记录。'),'retryable':code=='BUSY'},token=token),status_code=status)
def number(value,minimum=1,maximum=2**63-1):
    require(isinstance(value,str) and value.isascii() and value.isdigit() and len(value)<=19)
    n=int(value);require(minimum<=n<=maximum);return n
def params(request,allowed):
    pairs=list(request.query_params.multi_items());require(len(set(k for k,v in pairs))==len(pairs) and all(k in allowed for k,v in pairs));return dict(pairs)
def strict_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d);d[k]=v
        return d
    def constant(value):raise ContractError('INVALID_INPUT')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (ValueError,UnicodeError,RecursionError):raise ContractError('INVALID_INPUT')

class Boundary:
    def __init__(self,app,max_body=16384,candidate71=False,unified71=False):self.app=app;self.max_body=max_body;self.candidate71=candidate71;self.unified71=unified71
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        rid=uuid.uuid4().hex;scope.setdefault('state',{})['request_id']=rid
        headers=scope.get('headers',[]);hosts=[v.decode('latin1') for k,v in headers if k.lower()==b'host'];token=next((v.decode('latin1') for k,v in headers if k.lower()==b'x-selection-token'),None)
        scope['state']['selection_token']=token
        async def secured(message):
            if message['type']=='http.response.start':
                message['headers']=list(message.get('headers',[]))+[(b'content-security-policy',CSP.encode()),(b'x-content-type-options',b'nosniff'),(b'cache-control',b'no-store'),(b'x-request-id',rid.encode())]
            await send(message)
        async def reject(code,status):return await failure(rid,code,status,token)(scope,receive,secured)
        if len(hosts)!=1 or hosts[0].split(':')[0] not in ['127.0.0.1','localhost']:return await reject('INVALID_INPUT',400)
        if self.unified71 and scope['path'].startswith('/api/v2') and not (scope['path'].startswith('/api/v2/candidate71/') or scope['path'].startswith('/api/v2/health/') or scope['path'] in ['/api/v2','/api/v2/capabilities']):return await reject('LEGACY_ENTRY_ARCHIVED',410)
        if token is not None and (not 1<=len(token)<=80 or any(ord(c)<32 or ord(c)>126 for c in token)):return await reject('INVALID_INPUT',400)
        if len(scope.get('query_string',b''))>1024 or len(scope['path'])>1024:return await reject('INPUT_SIZE',413)
        maximum=262144 if self.candidate71 and scope['path'].startswith('/api/v2/candidate71/') else self.max_body
        lengths=[v for k,v in headers if k.lower()==b'content-length']
        if len(lengths)>1 or any(k.lower()==b'transfer-encoding' for k,v in headers):return await reject('INPUT_SIZE',413)
        try:length=number(lengths[0].decode(),0,maximum) if lengths else 0
        except ContractError:return await reject('INPUT_SIZE',413)
        if scope['method'] not in ['GET','HEAD','POST']:return await reject('METHOD_NOT_ALLOWED',405)
        if scope['method']!='POST' and length:return await reject('INPUT_SIZE',413)
        body=bytearray()
        try:
            while True:
                event=await asyncio.wait_for(receive(),timeout=5)
                if event['type']=='http.disconnect':return
                body.extend(event.get('body',b''))
                if len(body)>maximum:return await reject('INPUT_SIZE',413)
                if not event.get('more_body',False):break
        except asyncio.TimeoutError:return await reject('INVALID_INPUT',408)
        if len(body)!=length:return await reject('INVALID_INPUT',400)
        consumed=False
        async def buffered():
            nonlocal consumed
            if not consumed:consumed=True;return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()
        return await self.app(scope,buffered,secured)

def create_app(client,config,root,history=None,narratives=None,reference=None,catalog_index=None,review_store=None,report_store=None):
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    mode_file=Path(root)/'config/workbench/unified71.json';unified=mode_file.is_file()
    if unified:require(json.loads(mode_file.read_text(encoding='utf-8'))=={'version':'workspace71_only_v1','enabled':True} and bool(client.ready.get('candidate71')),'VERSION_MISMATCH')
    def rid(request):return request.state.request_id
    def token(request):return request.state.selection_token
    def response(request,payload):return JSONResponse(envelope(rid(request),payload,token=token(request)))
    def execute(message):return client.exchange(message,message)['payload']
    def get_case(pop,ident,expected=None):
        c=execute({'op':'cases','population':population(pop),'ids':[number(ident)]})['cases'][0]
        if expected is not None:require(c['case_revision']==expected,'VERSION_MISMATCH')
        return c
    @app.exception_handler(ContractError)
    async def contract_error(request,exc):return failure(rid(request),str(exc),409 if str(exc)=='VERSION_MISMATCH' else 403 if str(exc)=='SAMPLE_USE_NOT_ALLOWED' else 404 if str(exc) in ['CASE_NOT_FOUND','REVIEW_NOT_FOUND','REPORT_NOT_FOUND'] else 503 if str(exc) in ['REPORT_CORRUPT','REPORT_QUOTA'] else 400,token(request))
    @app.exception_handler(sqlite3.Error)
    async def storage_error(request,exc):return failure(rid(request),'LOCAL_STORAGE_UNAVAILABLE',503,token(request))
    @app.exception_handler(WorkerFailure)
    async def worker_error(request,exc):return failure(rid(request),exc.code,exc.status,token(request))
    @app.exception_handler(HistoryError)
    async def history_error(request,exc):
        code=str(exc)
        return failure(rid(request),code,403 if code=='SAMPLE_USE_NOT_ALLOWED' else 404 if code=='CASE_NOT_FOUND' else 503,token(request))
    @app.exception_handler(NarrativeError)
    async def narrative_error(request,exc):return failure(rid(request),str(exc),503,token(request))
    @app.exception_handler(ReferenceError)
    async def reference_error(request,exc):return failure(rid(request),str(exc),503,token(request))
    @app.exception_handler(HTTPException)
    async def http_error(request,exc):return failure(rid(request),'NOT_FOUND' if exc.status_code==404 else 'METHOD_NOT_ALLOWED' if exc.status_code==405 else 'INVALID_INPUT',exc.status_code,token(request))
    @app.exception_handler(Exception)
    async def unexpected(request,exc):return failure(rid(request),'INTERNAL_ERROR',500,token(request))
    @app.get('/')
    def index(request:Request):
        file=Path(root)/'frontend/dist/index.html'
        if file.is_file():return FileResponse(file,media_type='text/html')
        return failure(rid(request),'FRONTEND_NOT_BUILT',503,token(request))
    @app.get('/api/v2')
    def index_api(request:Request):return response(request,{'service':'workspace71','feature_count':71,'routes':['/api/v2/candidate71/info','/api/v2/candidate71/catalog','/api/v2/candidate71/reviews','/api/v2/candidate71/reports']} if unified else {'service':'workbench_v2_api','phase':'real_overview_with_frozen_model','routes':['/api/v2/health/ready','/api/v2/capabilities','/api/v2/cases?population=labeled','/api/v2/cases/labeled/100006'],'UI_scope':'40_saved_inputs_model_results_basic_charts_and_history_evidence' if history else '40_saved_inputs_model_results_basic_charts; historical_details_pending'})
    @app.get('/api/v2/health/live')
    def live(request:Request):params(request,set());return response(request,{'alive':True})
    @app.get('/api/v2/health/ready')
    def ready(request:Request):
        params(request,set())
        if client.failed or client.process.poll() is not None:raise WorkerFailure('WORKER_UNAVAILABLE')
        return response(request,{'ready':True,'release_manifest_sha256':config['release_manifest_sha256'],'dictionary_sha256':config['dictionary_sha256'],'worker_load_attempts':client.ready['load_attempts'],'eligible_counts':client.ready['counts']})
    @app.get('/api/v2/capabilities')
    def capabilities(request:Request):
        params(request,set())
        if unified:return response(request,{'feature_count':71,'model':client.ready['candidate71']['model_version'],'single_model':True,'catalog':True,'review':True,'report':True,'new_inputs':True})
        return response(request,{'populations':['labeled','unlabeled'],'sample_uses':['train','validation','unlabeled_scoring'],'max_batch':MAX_BATCH,'max_page':MAX_PAGE,'max_body_bytes':config['max_body_bytes'],'list_order':['id_asc','id_desc','probability_asc','probability_desc'] if catalog_index else ['id_asc','id_desc'],'filters':['population','sample_use','probability_min','probability_max','history','missing','signal','mark'] if catalog_index else ['population','sample_use'],'collection':'ready' if catalog_index else 'not_connected','feature_count':40,'release_manifest_sha256':config['release_manifest_sha256'],'dictionary_sha256':config['dictionary_sha256'],'history_index':'ready' if history else 'not_connected','controlled_narratives':'ready' if narratives else 'not_connected','train_reference':'ready' if reference else 'not_connected','pending':([] if history else ['historical_rows'])+([] if reference else ['train_reference'])+([] if narratives else ['controlled_narratives'])+([] if review_store and report_store else ['review','report'])})
    @app.get('/api/v2/train-reference')
    def reference_info(request:Request):
        params(request,set());return response(request,reference.info() if reference else {'status':'not_connected'})
    @app.get('/api/v2/cases/{pop}/{ident}/reference')
    def case_reference(request:Request,pop:str,ident:str):
        q=params(request,{'expected_revision'})
        if reference is None:raise ReferenceError('TRAIN_REFERENCE_NOT_CONNECTED')
        c=get_case(pop,ident,q.get('expected_revision'));return response(request,reference.render(c))
    @app.get('/api/v2/history-index')
    def history_info(request:Request):
        params(request,set())
        return response(request,history.info() if history else {'status':'not_connected'})
    @app.get('/api/v2/narrative-templates')
    def narrative_info(request:Request):
        params(request,set())
        return response(request,narratives.info() if narratives else {'status':'not_connected'})
    @app.get('/api/v2/cases/{pop}/{ident}/narratives')
    def narrative_case(request:Request,pop:str,ident:str):
        q=params(request,{'expected_revision'})
        if narratives is None:raise NarrativeError('NARRATIVE_NOT_CONNECTED')
        c=get_case(pop,ident,q.get('expected_revision'))
        return response(request,narratives.render(c,history))
    @app.get('/api/v2/cases/{pop}/{ident}/history/{table}')
    def history_rows(request:Request,pop:str,ident:str,table:str):
        q=params(request,{'feature','selection','offset','limit','expected_revision'})
        if history is None:raise HistoryError('HISTORY_INDEX_UNAVAILABLE')
        return response(request,history.query(pop,number(ident),table,feature=q.get('feature'),selection=q.get('selection','eligible'),offset=number(q.get('offset','0'),0,10**7),limit=number(q.get('limit','10'),1,50),expected=q.get('expected_revision')))
    @app.get('/api/v2/cases')
    def catalog(request:Request):
        if catalog_index:
            q=params(request,{'population','sample_use','offset','limit','order','probability_min','probability_max','history','missing','signal','mark','current_id'})
            return response(request,catalog_index.query({k:v for k,v in q.items() if k not in ['offset','limit','order','current_id']},number(q.get('offset','0'),0,10**7),number(q.get('limit','20'),1,MAX_PAGE),q.get('order','id_asc'),number(q['current_id']) if 'current_id' in q else None))
        q=params(request,{'population','sample_use','offset','limit','order'});pop=population(q.get('population'))
        return response(request,execute({'op':'catalog','population':pop,'sample_use':q.get('sample_use','all'),'offset':number(q.get('offset','0'),0,10**7),'limit':number(q.get('limit','20'),1,MAX_PAGE),'order':q.get('order','id_asc')}))
    @app.get('/api/v2/collection-info')
    def collection_info(request:Request):
        params(request,set())
        return response(request,{'status':'ready' if catalog_index else 'not_connected','version':'workbench_collection_v1','orders':['id_asc','id_desc','probability_asc','probability_desc'],'test_policy':'not_exposed','quality_scope':'saved_40_inputs_only'})
    @app.post('/api/v2/cases/{pop}/{ident}/marks')
    async def mark_case(request:Request,pop:str,ident:str):
        params(request,set());require(catalog_index is not None,'COLLECTION_NOT_CONNECTED')
        require(request.headers.get('content-type','').split(';')[0].strip()=='application/json')
        # Mutations are local only and reject cross-origin form/fetch submissions.
        origin=request.headers.get('origin');require(origin is None or origin==f"http://{request.headers['host']}")
        v=strict_json(await request.body());require(isinstance(v,dict) and set(v)=={'favorite','pending','expected_marks_revision','expected_case_revision'})
        from .contracts import revision
        pop=population(pop);ident=number(ident)
        require(v['expected_case_revision']==revision(config['release_manifest_sha256'],config['dictionary_sha256'],pop,ident),'VERSION_MISMATCH')
        r=catalog_index.mark(pop,ident,v['favorite'],v['pending'],v['expected_marks_revision'])
        return response(request,{'marks_revision':r,'case_key':{'population':pop,'application_id':ident},'favorite':v['favorite'],'pending':v['pending'],'scope':'local_simulated_analysis_only'})
    @app.post('/api/v2/cases/batch')
    async def batch(request:Request):
        params(request,set());require(request.headers.get('content-type','').split(';')[0].strip()=='application/json')
        v=strict_json(await request.body());require(isinstance(v,dict) and {'population','application_ids'}<=set(v) and set(v)<={'population','application_ids','expected_release_manifest_sha256'})
        if 'expected_release_manifest_sha256' in v:require(v['expected_release_manifest_sha256']==config['release_manifest_sha256'],'VERSION_MISMATCH')
        message={'op':'cases','population':population(v['population']),'ids':identifiers(v['application_ids'])}
        # Blocking IPC belongs in a thread; liveness/ready remain responsive.
        from starlette.concurrency import run_in_threadpool
        result=await run_in_threadpool(execute,message)
        return response(request,result)
    async def review_body(request,keys):
        params(request,set());require(review_store is not None and report_store is not None,'WORKFLOW_NOT_CONNECTED')
        require(request.headers.get('content-type','').split(';')[0].strip()=='application/json')
        origin=request.headers.get('origin');require(origin is None or origin==f"http://{request.headers['host']}")
        v=strict_json(await request.body());require(isinstance(v,dict) and set(v)==keys);return v
    @app.get('/api/v2/reviews')
    def review_list(request:Request):
        q=params(request,{'deleted','offset','limit'});require(review_store is not None,'WORKFLOW_NOT_CONNECTED');require(q.get('deleted','false') in ['true','false'])
        return response(request,review_store.listing(config['release_manifest_sha256'],number(q.get('offset','0'),0,10**7),number(q.get('limit','20'),1,50),q.get('deleted','false')=='true'))
    @app.get('/api/v2/cases/{pop}/{ident}/review')
    def review_get(request:Request,pop:str,ident:str):
        q=params(request,{'expected_revision'});require(review_store is not None,'WORKFLOW_NOT_CONNECTED');return response(request,review_store.read(get_case(pop,ident,q.get('expected_revision'))))
    @app.post('/api/v2/cases/{pop}/{ident}/review')
    async def review_save(request:Request,pop:str,ident:str):
        v=await review_body(request,{'expected_case_revision','expected_marks_revision','favorite','review_status','note'})
        from starlette.concurrency import run_in_threadpool
        def save():return review_store.write(get_case(pop,ident,v['expected_case_revision']),v['expected_marks_revision'],favorite=v['favorite'],status=v['review_status'],note=v['note'])
        return response(request,await run_in_threadpool(save))
    @app.post('/api/v2/cases/{pop}/{ident}/review/{action}')
    async def review_action(request:Request,pop:str,ident:str,action:str):
        require(action in ['delete','restore']);v=await review_body(request,{'expected_case_revision','expected_marks_revision'})
        from starlette.concurrency import run_in_threadpool
        def apply():return review_store.write(get_case(pop,ident,v['expected_case_revision']),v['expected_marks_revision'],action=action)
        return response(request,await run_in_threadpool(apply))
    @app.post('/api/v2/cases/{pop}/{ident}/report')
    async def report_create(request:Request,pop:str,ident:str):
        v=await review_body(request,{'expected_case_revision','expected_marks_revision','selection'})
        from starlette.concurrency import run_in_threadpool
        def create():
            c=get_case(pop,ident,v['expected_case_revision'])
            return report_store.create(c,v['selection'],v['expected_marks_revision'],narratives.render(c,history) if narratives else None,reference.info() if reference else None,history.info() if history else None,catalog_index.meta['fields'])
        return response(request,await run_in_threadpool(create))
    @app.get('/api/v2/reports')
    def report_list(request:Request):
        q=params(request,{'limit'});require(report_store is not None,'WORKFLOW_NOT_CONNECTED');return response(request,report_store.listing(config['release_manifest_sha256'],number(q.get('limit','20'),1,50)))
    @app.get('/api/v2/reports/{ident}/{section}')
    def report_file(request:Request,ident:str,section:str):
        params(request,set());require(report_store is not None,'WORKFLOW_NOT_CONNECTED');require(section in ['view','download','data'])
        p=report_store.file(ident,'json' if section=='data' else 'html')
        return FileResponse(p,media_type='application/json' if section=='data' else 'text/html',filename=f'simulated-analysis-{ident}.html' if section=='download' else None)
    @app.get('/api/v2/cases/{pop}/{ident}')
    def case(request:Request,pop:str,ident:str):q=params(request,{'expected_revision'});return response(request,get_case(pop,ident,q.get('expected_revision')))
    @app.get('/api/v2/cases/{pop}/{ident}/{section}')
    def section(request:Request,pop:str,ident:str,section:str):
        require(section in ['profile','models','contributions']);q=params(request,{'expected_revision'});c=get_case(pop,ident,q.get('expected_revision'));key={'profile':'profile','models':'model_outputs','contributions':'explanation'}[section]
        return response(request,{k:c[k] for k in ['case_key','case_revision','sample_use','versions',key]})
    assets=Path(root)/'frontend/dist/assets'
    if assets.is_dir():app.mount('/assets',StaticFiles(directory=assets),name='assets')
    app.add_middleware(Boundary,max_body=config['max_body_bytes'],candidate71=bool(client.ready.get('candidate71')),unified71=unified)
    return app
