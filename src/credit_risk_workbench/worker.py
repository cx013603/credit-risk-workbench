"""JSONL worker run only by the restored runtime; formal loader is unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys

MAX_REQUEST=4096
def emit(value):
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--seal',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--origin',type=Path,required=True)
    a=p.parse_args();release=a.release.resolve();out=a.output.resolve();origin=a.origin.resolve()
    sys.path.insert(0,str(release/'project/src'))
    # Reuse F05's guard policy without changing any frozen file/loader.
    import pyarrow.parquet as pq
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from credit_risk.scorecard import WOETransformer
    from credit_risk.tree import EmptyNumericGuard
    counters={'fit_attempts':0,'reserved_test_table_read_attempts':0,'origin_open_attempts':0,'loaded_model_paths':[],'parquet_table_reads':[]}
    def deny_fit(*args,**kwargs):
        counters['fit_attempts']+=1;raise RuntimeError('Fitting forbidden')
    for cls in [Pipeline,LogisticRegression,HistGradientBoostingClassifier,WOETransformer,EmptyNumericGuard]:cls.fit=deny_fit
    original_read=pq.read_table
    def guarded_read(path,*args,**kwargs):
        value=Path(path).resolve();counters['parquet_table_reads'].append(str(value))
        if value.name=='test.parquet':counters['reserved_test_table_read_attempts']+=1;raise RuntimeError('Reserved test feature read forbidden')
        return original_read(path,*args,**kwargs)
    pq.read_table=guarded_read
    original_load=joblib.load
    def guarded_load(path,*args,**kwargs):
        counters['loaded_model_paths'].append(str(Path(path).resolve()));return original_load(path,*args,**kwargs)
    joblib.load=guarded_load
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            try:value=Path(os.fsdecode(args[0])).resolve()
            except (OSError,ValueError):return
            if value.is_relative_to(origin) and not value.is_relative_to(out):
                counters['origin_open_attempts']+=1;raise RuntimeError('Original project fallback forbidden')
    sys.addaudithook(audit)
    loads=0
    try:
        from credit_risk.portable_runtime import load_release
        loads+=1;loaded=load_release(release,a.seal)
        emit({'type':'ready','pid':os.getpid(),'load_attempts':loads,'seal':a.seal,'interpreter':sys.executable})
        while True:
            line=sys.stdin.buffer.readline(MAX_REQUEST+1)
            if not line:break
            if len(line)>MAX_REQUEST:emit({'type':'fatal','error':'IPC_SIZE'});return 2
            request_id=None
            try:
                v=json.loads(line);request_id=v.get('request_id')
                if set(v)!={'request_id','population','application_id'} or not isinstance(request_id,str) or not 1<=len(request_id)<=80:raise ValueError('INVALID_INPUT')
                population=v['population'];ident=v['application_id']
                if population not in ['labeled','unlabeled']:raise ValueError('INVALID_POPULATION')
                if type(ident) is not int or ident<1:raise ValueError('INVALID_INPUT')
                if (population,ident) not in {('labeled',100006),('labeled',100003),('unlabeled',100001)}:raise ValueError('CASE_NOT_FOUND')
                case=loaded.explanation.explain_ids(population,[ident])[0]
                emit({'type':'result','request_id':request_id,'case_key':{'population':population,'application_id':ident},'payload':case,'worker':{'pid':os.getpid(),'load_attempts':loads}})
            except Exception as exc:
                code=str(exc) if isinstance(exc,ValueError) else getattr(exc,'code','WORKER_ERROR')
                emit({'type':'error','request_id':request_id,'error':{'code':code,'message':'当前兼容示例不可用。'}})
    except Exception as exc:
        emit({'type':'fatal','error':getattr(exc,'code','WORKER_START_FAILED')});return 1
    finally:
        (out/'worker-guard.json').write_text(json.dumps({'pid':os.getpid(),'interpreter':sys.executable,'load_attempts':loads,**counters},ensure_ascii=False,indent=2),encoding='utf-8')
    return 0

if __name__=='__main__':raise SystemExit(main())
