"""F05-equivalent deny policy; never substitutes loader/model/session."""
import json
import os
from pathlib import Path
import sys

def install(origin,output,allowed_files=()):
    allowed_files=frozenset(Path(p).resolve() for p in allowed_files)
    import pyarrow.parquet as pq
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from credit_risk.scorecard import WOETransformer
    from credit_risk.tree import EmptyNumericGuard
    counters={'fit_attempts':0,'reserved_test_table_read_attempts':0,'origin_open_attempts':0,'loaded_model_paths':[],'parquet_table_reads':[]}
    def deny_fit(*args,**kwargs):counters['fit_attempts']+=1;raise RuntimeError('Fitting forbidden')
    for cls in [Pipeline,LogisticRegression,HistGradientBoostingClassifier,WOETransformer,EmptyNumericGuard]:cls.fit=deny_fit
    read=pq.read_table
    def guarded_read(path,*args,**kwargs):
        value=Path(path).resolve();counters['parquet_table_reads'].append(str(value))
        if value.name=='test.parquet':counters['reserved_test_table_read_attempts']+=1;raise RuntimeError('Reserved test feature read forbidden')
        return read(path,*args,**kwargs)
    pq.read_table=guarded_read;load=joblib.load
    def guarded_load(path,*args,**kwargs):counters['loaded_model_paths'].append(str(Path(path).resolve()));return load(path,*args,**kwargs)
    joblib.load=guarded_load
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            try:value=Path(os.fsdecode(args[0])).resolve()
            except (OSError,ValueError):return
            if value.is_relative_to(origin) and not value.is_relative_to(output) and value not in allowed_files:counters['origin_open_attempts']+=1;raise RuntimeError('Original project fallback forbidden')
    sys.addaudithook(audit)
    return counters
