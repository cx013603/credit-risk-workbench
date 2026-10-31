"""Formal V2 worker: original 03 runtime/loader + bounded case/catalog operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from credit_risk_workbench.contracts import (SCHEMA,METHOD,MAX_BATCH,MAX_PAGE,ContractError,require,integer,population,identifiers,project_case)
from credit_risk_workbench.frozen_guard import install
from credit_risk_workbench.catalog import build_catalog
from credit_risk_workbench.workspace71 import build_catalog71

def emit(value):
    line=json.dumps(value,ensure_ascii=False,allow_nan=False)
    if len(line.encode('utf-8'))>2*1024*1024:raise RuntimeError('Response limit')
    print(line,flush=True)

def dispatch(v,loaded,catalog,dictionary,seal,digest,input_engine=None,input_spool=None,candidate71=None):
    if v.get('op')=='candidate71':
        require(candidate71 is not None,'VERSION_MISMATCH')
        if v.get('action')=='example':
            require(set(v)=={'request_id','op','action','id'} and type(v['id']) is int,'INVALID_INPUT')
            require(str(v['id']) in candidate71.examples,'CASE_NOT_FOUND');return {'payload':candidate71.examples[str(v['id'])]}
        if v.get('action')=='cases':
            require(set(v)=={'request_id','op','action','population','ids'},'INVALID_INPUT')
            return candidate71.cases(population(v['population']),v['ids'])
        from credit_risk_workbench.input_contract import opaque_id
        require(v.get('action')=='raw' and set(v)=={'request_id','op','action','ticket','sha256'},'INVALID_INPUT')
        path=input_spool.parent/'candidate71-spool'/(opaque_id(v['ticket'])+'.json')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=250000,'INVALID_INPUT')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==v['sha256'],'VERSION_MISMATCH');p=json.loads(raw)
        require(set(p)=={'payloads'} and isinstance(p['payloads'],list) and 1<=len(p['payloads'])<=5,'INVALID_INPUT')
        for payload in p['payloads']:
            from credit_risk_workbench.candidate71_contract import complete
            complete(payload);ident=payload['application']['SK_ID_CURR'];s=loaded.sources['labeled']
            require(ident not in s.index or s.features['sample_use'][s.index[ident]].as_py()!='test','SAMPLE_USE_NOT_ALLOWED')
        return candidate71.score(p['payloads'])
    if v.get('op')=='new_inputs':
        from credit_risk_workbench.input_contract import opaque_id
        require(set(v)=={'request_id','op','ticket','sha256'} and input_engine is not None,'INVALID_INPUT')
        ident=opaque_id(v['ticket']);path=input_spool/(ident+'.json')
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=16000,'INVALID_INPUT')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==v['sha256'],'VERSION_MISMATCH')
        payload=json.loads(raw);require(set(payload)=={'contract_version','payloads'} and payload['contract_version']=='new_application_raw40_v1','VERSION_MISMATCH')
        return input_engine.score(loaded,payload['payloads'])
    op=v.get('op');pop=population(v.get('population'))
    require(set(v)==({'request_id','op','population','ids'} if op=='cases' else {'request_id','op','population','sample_use','offset','limit','order'}))
    if op=='catalog':
        use=v['sample_use'];require(use in (['train','validation','all'] if pop=='labeled' else ['unlabeled_scoring','all']))
        offset=integer(v['offset'],0,10**7);limit=integer(v['limit'],1,MAX_PAGE);require(v['order'] in ['id_asc','id_desc'])
        rows=catalog[pop];rows=rows if use=='all' else [x for x in rows if x['sample_use']==use]
        if v['order']=='id_desc':rows=list(reversed(rows))
        return {'items':rows[offset:offset+limit],'total':len(rows),'offset':offset,'limit':limit,'next_offset':offset+limit if offset+limit<len(rows) else None,'order':v['order'],'sample_uses_allowed':['train','validation'] if pop=='labeled' else ['unlabeled_scoring']}
    require(op=='cases');ids=identifiers(v['ids']);source=loaded.sources[pop]
    require(all(i in source.index for i in ids),'CASE_NOT_FOUND')
    require(all(source.features['sample_use'][source.index[i]].as_py()!='test' for i in ids),'SAMPLE_USE_NOT_ALLOWED')
    return {'cases':[project_case(c,dictionary,seal,digest) for c in loaded.explanation.explain_ids(pop,ids)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--seal',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--origin',type=Path,required=True);p.add_argument('--dictionary',type=Path,required=True);p.add_argument('--dictionary-sha256',required=True)
    p.add_argument('--catalog-index',action='store_true');p.add_argument('--new-inputs',action='store_true');p.add_argument('--candidate71-config',type=Path)
    a=p.parse_args();out=a.output.resolve();dictionary_bytes=a.dictionary.read_bytes();digest=hashlib.sha256(dictionary_bytes).hexdigest()
    if digest!=a.dictionary_sha256:emit({'type':'fatal','error':'VERSION_MISMATCH'});return 1
    dictionary=json.loads(dictionary_bytes);sys.path.insert(0,str(a.release/'project/src'))
    # Load extension code before the original-directory fallback audit is installed.
    if a.new_inputs:
        from credit_risk_workbench.input_engine import InputEngine
        import credit_risk_workbench.input_contract
    candidate_spec=None
    if a.candidate71_config:
        from credit_risk_workbench.candidate71 import prepare,Candidate71
        candidate_spec=prepare(a.origin,a.candidate71_config)
    counters=install(a.origin.resolve(),out,candidate_spec['paths'].values() if candidate_spec else ());loads=0
    def save_guard(stage):
        (out/'worker-guard.json').write_text(json.dumps({'pid':os.getpid(),'interpreter':sys.executable,'load_attempts':loads,'stage':stage,**counters},ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        from credit_risk.portable_runtime import load_release
        loads+=1;loaded=load_release(a.release,a.seal)
        require(dictionary['order']==loaded.sources['labeled'].contract['feature_columns'] and len(dictionary['fields'])==40,'VERSION_MISMATCH')
        input_engine=InputEngine(a.release,a.seal,loaded.sources['labeled'].contract) if a.new_inputs else None
        candidate=Candidate71(candidate_spec,input_engine) if candidate_spec else None
        catalog={}
        for pop,s in loaded.sources.items():
            catalog[pop]=sorted([{'application_id':i,'sample_use':u} for i,u in zip(s.features['SK_ID_CURR'].to_pylist(),s.features['sample_use'].to_pylist()) if u!='test'],key=lambda x:x['application_id'])
        unified=(a.origin/'config/workbench/unified71.json').is_file()
        index=build_catalog(loaded,dictionary,out/'catalog.sqlite',a.seal,digest) if a.catalog_index and not unified else None
        index71=build_catalog71(candidate,out/'catalog71.sqlite') if unified and candidate else None
        save_guard('ready')
        emit({'type':'ready','schema_version':SCHEMA,'pid':os.getpid(),'load_attempts':loads,'release_manifest_sha256':a.seal,'dictionary_sha256':digest,'counts':{k:len(v) for k,v in catalog.items()},'interpreter':sys.executable,'catalog_index':index,'candidate71_catalog':index71,'candidate71':candidate.info() if candidate else None})
        while True:
            line=sys.stdin.buffer.readline(4097)
            if not line:break
            if len(line)>4096:emit({'type':'fatal','error':'IPC_SIZE'});return 2
            rid=None;v=None
            try:
                v=json.loads(line);require(isinstance(v,dict));rid=v.get('request_id');require(isinstance(rid,str) and 1<=len(rid)<=80)
                payload=dispatch(v,loaded,catalog,dictionary,a.seal,digest,input_engine,out/'input-spool',candidate)
                save_guard('request_completed');emit({'type':'result','request_id':rid,'identity':{k:x for k,x in v.items() if k!='request_id'},'payload':payload})
            except Exception as exc:
                code=str(exc) if isinstance(exc,ContractError) else 'INVALID_CANDIDATE71_HISTORY' if isinstance(v,dict) and v.get('op')=='candidate71' and getattr(exc,'code',None)=='v2_feature_contract' else 'WORKER_ERROR'
                save_guard('request_rejected');emit({'type':'error','request_id':rid,'error':{'code':code,'message':'Worker operation failed'}})
    except Exception as exc:emit({'type':'fatal','error':str(exc) if isinstance(exc,ContractError) else getattr(exc,'code','WORKER_START_FAILED')});return 1
    finally:save_guard('stopped')
    return 0
if __name__=='__main__':raise SystemExit(main())
