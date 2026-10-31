"""E12 orchestration of accepted modules, without retraining or reevaluating test."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import build_opener,ProxyHandler

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .scoring import load_scoring_bundle
from .normalize import load_normalized_bundle,write_json
from .history import load_history_bundle
from .experiments import verify_experiment_bundle,load_training_partition
from .demo import DemoState,make_server
from .raw import DataError,sha256_file,RawDataSource
from .runtime import load_config,RunRecord,now,process_peak_memory

VERSION='reproduction_v1'
REPORTS=['normalization','history','experiment','assessment','scoring','demo']


def recipe(root,path):
    root=Path(root).resolve()
    try:
        v=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if set(v)!={'version','tolerance','child_timeout_seconds',*[n+'_report' for n in REPORTS]} or v['version']!=VERSION or v['tolerance']!=1e-10 or type(v['child_timeout_seconds']) is not int or not 60<=v['child_timeout_seconds']<=3600:
            raise ValueError('unsupported recipe')
        for name in REPORTS:
            p=(root/v[name+'_report']).resolve()
            if not p.is_relative_to(root/'reports/runs') or p.name!='run.json':raise ValueError('report outside runs')
        return v
    except (OSError,ValueError,KeyError,TypeError) as exc:raise DataError('reproduction_config','Exact project-local reproduction recipe required') from exc


def audit(root,v):
    root=Path(root).resolve();paths={n:(root/v[n+'_report']).resolve() for n in REPORTS}
    session=load_scoring_bundle(root,paths['scoring'])
    b=session.bundle
    if Path(b['experiment_report']).resolve()!=paths['experiment'] or Path(b['assessment_report']).resolve()!=paths['assessment'] or Path(b['history_report']).resolve()!=paths['history']:
        raise DataError('reproduction_lineage','Recipe must reference the accepted E10 lineage')
    normal=load_normalized_bundle(root,paths['normalization']);history=load_history_bundle(root,paths['history']);experiment=verify_experiment_bundle(root,paths['experiment'])
    if Path(history['bundle']['input_run_path']).resolve()!=paths['normalization']:
        raise DataError('reproduction_lineage','E04 and E03 lineage differ')
    demo=json.loads(paths['demo'].read_text(encoding='utf-8'))
    if demo['status']!='passed' or demo['scope']!='official_local_demo' or demo['metadata']['module']!='E11' or Path(demo['demo']['scoring_report']).resolve()!=paths['scoring']:
        raise DataError('reproduction_demo','Accepted E11 lifecycle and scoring source required')
    if (demo['demo']['scoring_report_sha256']!=sha256_file(paths['scoring']) or any(sha256_file(root/p)!=h for p,h in demo['metadata']['code_sha256'].items()) or demo['metadata']['demo_config_sha256']!=sha256_file(root/'config/demo.json')):
        raise DataError('reproduction_demo_changed','Accepted page source/config changed')
    access=Path(json.loads(paths['assessment'].read_text(encoding='utf-8'))['metadata']['selection_path']).parent/'test_access.jsonl'
    if len(access.read_text(encoding='utf-8').splitlines())!=1:raise DataError('reproduction_access','Original test access history differs')
    try:load_training_partition(root,paths['experiment'],'test')
    except DataError as exc:
        if exc.code!='reserved_test_locked':raise
    else:raise DataError('reproduction_test','Training API unexpectedly released test')
    runtime=load_config(root,root/'config/runtime.json');source=RawDataSource(runtime.raw_dir,runtime.manifest)
    input_hashes={}
    for name,expected in normal['bundle']['inputs'].items():
        actual=sha256_file(source.path(name))
        if actual!=expected:raise DataError('reproduction_raw_changed','Original normalization input changed',table=name)
        input_hashes[name]=actual
    summary={'version':VERSION,'environment':session.bundle['environment'],'duckdb':importlib.metadata.version('duckdb'),
        'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'virtualenv_isolated':sys.prefix!=sys.base_prefix,
        'reports':{n:{'path':str(p),'sha256':sha256_file(p)} for n,p in paths.items()},
        'protocol_fingerprint':b['output_versions']['protocol_fingerprint'],'dataset_version':b['output_versions']['dataset_version'],
        'feature_columns':session.contract['feature_columns'],'feature_count':40,'rows':session.features.num_rows,
        'plan_sha256':sha256_file(root/'config/experiment_plan.json'),'input_hashes':input_hashes,
        'recommendation':'tree','threshold':b['threshold'],'original_test_access_intents':1,
        'test_training_API_rejected':True,'original_test_feature_table_read':False,'fit_or_new_test_metrics':False}
    return summary,{'session':session,'normal':normal,'history':history,'experiment':experiment,'demo':demo}


def child(root,record,name,args,timeout):
    """Only caller-defined scripts/arguments; no shell interpolation or recursive job engine."""
    root=Path(root).resolve();record.data['stage']=name;record.save()
    command=[sys.executable,'-B','-X','utf8',*args]
    print(json.dumps({'stage':name,'command':command}),flush=True)
    p=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    for label,value in [('stdout',p.stdout),('stderr',p.stderr)]:
        (record.directory/(name+'_'+label+'.txt')).write_text(value,encoding='utf-8')
    if p.returncode!=0:raise DataError('reproduction_child_failed',f'{name} failed; see immutable run log')
    try:
        result=json.loads(p.stdout.strip().splitlines()[-1]);report=Path(result['report']).resolve()
        if result['passed'] is not True or not report.is_relative_to(root/'reports/runs'):raise ValueError('wrong result')
        payload=json.loads(report.read_text(encoding='utf-8'))
        if payload['status']!='passed':raise ValueError('not passed')
    except (ValueError,KeyError,IndexError) as exc:raise DataError('reproduction_child_result','Expected complete successful child run') from exc
    reference={'command':command,'report':str(report),'report_sha256':sha256_file(report),'run_id':payload['run_id'],'returncode':p.returncode}
    record.data.setdefault('children',{})[name]=reference;record.save()
    return payload,reference


def compare_tables(old,new,names,tolerance=1e-10):
    result={}
    for name in names:
        a,b=Path(old)/name,Path(new)/name
        equal=sha256_file(a)==sha256_file(b)
        if equal:result[name]={'byte_identical':True,'max_absolute_difference':0.};continue
        x,y=pq.read_table(a),pq.read_table(b)
        if x.schema!=y.schema or x.num_rows!=y.num_rows:raise DataError('reproduction_schema','Replayed schema/population differs')
        maximum=0.
        for field in x.schema:
            left,right=x[field.name],y[field.name]
            if left.equals(right):continue
            if not pa.types.is_floating(field.type) or not pc.is_null(left).equals(pc.is_null(right)):
                raise DataError('reproduction_values','Non-floating values, IDs, nulls or versions differ')
            mask=pc.invert(pc.is_null(left));av=left.filter(mask).to_numpy();bv=right.filter(mask).to_numpy()
            if not np.isfinite(av).all() or not np.isfinite(bv).all():raise DataError('reproduction_nonfinite','Nonfinite replay')
            delta=float(np.max(np.abs(av-bv))) if len(av) else 0.
            if delta>tolerance:raise DataError('reproduction_difference','Replay exceeds predeclared 1e-10 tolerance')
            maximum=max(maximum,delta)
        result[name]={'byte_identical':False,'max_absolute_difference':maximum}
    return result


def page_probe(root,scoring_report):
    state=DemoState(root,scoring_report).load()
    if state.session is None:raise DataError('reproduction_page','Real success bundle did not load')
    server=make_server(state,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=build_opener(ProxyHandler({}));cases={}
    try:
        for suffix,expected in [('',200),('?id=100003',200),('?id=100024',200),('?id=100011',200),('?id=1',404),('?id=',400)]:
            try:
                with opener.open(f'http://127.0.0.1:{server.server_address[1]}/'+suffix,timeout=30) as response:
                    status=response.status;html=response.read().decode('utf-8')
            except HTTPError as exc:
                try:status=exc.code;html=exc.read().decode('utf-8')
                finally:exc.close()
            if status!=expected or (status!=200 and 'id="risk-probability"' in html):raise DataError('reproduction_page_result','Live page/error result differs')
            cases[suffix or 'default']=status
        return {'real_E10_loaded':True,'loopback_HTTP_codes':cases,'new_process_environment':sys.prefix}
    finally:server.shutdown();server.server_close();thread.join(3)


def reproduce(root,config,record,rebuild_data=False):
    root=Path(root).resolve();start=time.monotonic()
    try:
        v=recipe(root,config);record.data['stage']='audit_accepted_lineage';record.save()
        summary,context=audit(root,v);write_json(record.directory/'lineage.json',summary)
        if sys.prefix==sys.base_prefix:raise DataError('reproduction_venv','An isolated virtual environment is required')
        original_plan=summary['plan_sha256'];checks={};timeout=v['child_timeout_seconds']
        if rebuild_data:
            normalized,_=child(root,record,'E03_raw_rebuild',['scripts/check_normalization_rebuild.py','--input-run',v['normalization_report']],timeout)
            checks['raw_normalized_parquet_rebuilt_identically']=normalized['original_output_unchanged'] and normalized['temporary_rebuild_removed']
            h,href=child(root,record,'E04_SQL_rebuild',['scripts/run_history_features.py','--input-run',v['normalization_report']],timeout)
            checks['history']=compare_tables(context['history']['directory'],h['history_features']['artifact_dir'],['bureau_features.parquet','previous_features.parquet','history_audit.parquet','joined_features.parquet'],0.)
            e,_=child(root,record,'E05_contract_rebuild',['scripts/run_experiment_contract.py','--input-run',href['report']],timeout)
            checks['experiment']=compare_tables(context['experiment']['directory'],e['experiment']['artifact_dir'],['split_manifest.parquet','train.parquet','validation.parquet','test.parquet'],0.)
            if e['experiment']['protocol_fingerprint']!=summary['protocol_fingerprint']:raise DataError('reproduction_protocol','Rebuilt protocol differs')
            checks['protocol_identical_no_reactivation']=True
        new_score,sref=child(root,record,'E10_full_replay',['scripts/run_scoring.py','build'],timeout)
        checks['scoring']=compare_tables(context['session'].directory,new_score['scoring']['artifact_dir'],['features.parquet','scores.parquet','components.parquet'])
        record.data['stage']='E11_live_new_environment';record.save();checks['page']=page_probe(root,root/v['scoring_report'])
        if sha256_file(root/'config/experiment_plan.json')!=original_plan:raise DataError('reproduction_plan_changed','Frozen plan changed')
        after,_=audit(root,v)
        if summary!=after:raise DataError('reproduction_lineage_changed','Accepted sources changed during reproduction')
        result={'audit':summary,'data_rebuilt':rebuild_data,'checks':checks,'source_environment_is_new':str(Path(sys.prefix).resolve())!=str((root/'.venv').resolve()),
            'same_machine_same_base_python':True,'models_retrained':False,'test_reevaluated':False,'frozen_anchor_changed':False,
            'replayed_E10_report':sref,'purpose':'reproduce existing frozen experiment and inference; no new statistical evidence'}
        write_json(record.directory/'result.json',result)
        record.finish(True,reproduction={'result_path':str(record.directory/'result.json'),'result_sha256':sha256_file(record.directory/'result.json')},elapsed_seconds=round(time.monotonic()-start,3),**process_peak_memory())
        return result
    except BaseException as exc:
        error=exc.as_dict() if isinstance(exc,DataError) else {'code':type(exc).__name__,'message':str(exc)}
        record.finish(False,error=error,elapsed_seconds=round(time.monotonic()-start,3),**process_peak_memory());raise


def plan(root,v):
    return {'version':VERSION,'current_mode':'frozen_experiment_reproduction',
        'reproduce':['audit accepted lineage and original raw fingerprints','optional E03 raw deterministic rebuild','optional E04 SQL and E05 split/contract rebuild without freeze','E10 whole-population inference replay','E11 real loopback page check','recheck immutable anchors and sources'],
        'training_commands_separate':{'logistic':['scripts/run_logistic_baseline.py','--input-run',v['experiment_report']],
            'scorecard':['scripts/run_scorecard.py','--input-run',v['experiment_report']],
            'tree':['scripts/run_tree.py','--input-run',v['experiment_report']]},
        'training_note':'Reference only; reproduce does not retrain. New fits are not new independent evidence and cannot replace locked model references.',
        'assessment':'E09 already complete; consume load_assessment_bundle, never rerun evaluate or delete the access log',
        'absolute_paths':'Existing accepted official packages require the original project root; snapshot restore tests are separately scoped sealed recovery probes.'}
