"""E13 preparation: accepted SQL projection, resource gates and result comparison.

This revision never creates SparkSession. Live fixtures, materialization and
same-session performance orchestration remain explicitly unverified/deferred.
"""
import ctypes
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc

from .history import FEATURES,FEATURE_NAMES,COUNT_NAMES,SQL_FILES,load_history_bundle
from .normalize import load_normalized_bundle
from .raw import DataError,sha256_file

FIXED={'version':'spark_comparison_preparation_v1','execution_status':'deferred_by_user',
       'minimum_available_bytes':3*1024**3,'spark_master':'local[1]','spark_driver_memory':'1g',
       'shuffle_partitions':4,'spark_ui_enabled':False,'duckdb_threads':1,
       'duckdb_memory_limit':'512MB','absolute_tolerance':1e-9,'relative_tolerance':1e-12}


def config(root,path):
    root=Path(root).resolve()
    try:
        v=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if set(v)!=set(FIXED)|{'normalization_report','history_report'}:raise ValueError('unexpected fields')
        if any(type(v[k]) is not type(value) or v[k]!=value for k,value in FIXED.items()):raise ValueError('fixed preparation policy differs')
        for name in ['normalization_report','history_report']:
            p=Path(v[name])
            if p.is_absolute() or not (root/p).resolve().is_relative_to(root/'reports/runs') or p.name!='run.json':raise ValueError('unsafe source report')
        return v
    except (OSError,ValueError,TypeError,KeyError) as exc:raise DataError('spark_config','Exact local E13 preparation config required') from exc


def source_sql(root,hashes):
    """Only translate the exact already accepted E04 statements, not arbitrary SQL."""
    root=Path(root);texts={}
    for name in SQL_FILES:
        if name not in hashes or sha256_file(root/name)!=hashes[name]:raise DataError('spark_source_changed','Accepted E04 SQL fingerprint differs')
        texts[name]=(root/name).read_text(encoding='utf-8')
    converted={}
    for name in SQL_FILES[:2]:
        text=texts[name]
        if text.count('SELECT * EXCLUDE (exclusion_reason)')!=1 or text.count('CREATE TABLE ')!=1:raise DataError('spark_sql_shape','Review a changed source shape before translating')
        text=text.replace('CREATE TEMP VIEW ','CREATE OR REPLACE TEMP VIEW ')
        text=text.replace('CREATE TABLE ','CREATE OR REPLACE TEMP VIEW ')
        # Keep the reason column in a temporary eligible view; it is never
        # selected by the aggregate. Avoid engine-specific star exclusion.
        text=text.replace('SELECT * EXCLUDE (exclusion_reason)','SELECT *')
        text=text.replace('COUNT(*)::BIGINT','CAST(COUNT(*) AS BIGINT)').replace('1::BIGINT','CAST(1 AS BIGINT)')
        text=text.replace("(COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active'))::DOUBLE","CAST(COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS DOUBLE)")
        if '::' in text or 'EXCLUDE' in text:raise DataError('spark_sql_shape','Untranslated DuckDB syntax')
        converted[Path(name).stem]=text
    columns=[]
    for name in FEATURE_NAMES:
        alias='b' if name.startswith('BUREAU_') else 'p'
        expression=f'COALESCE({alias}.{name}, CAST(0 AS BIGINT))' if name in COUNT_NAMES else f'{alias}.{name}'
        columns.append(f'{expression} AS {name}')
    converted['04_join_history_projection']='SELECT a.SK_ID_CURR,\n    '+',\n    '.join(columns)+'''\nFROM application_input a
LEFT JOIN bureau_features b USING (SK_ID_CURR)
LEFT JOIN previous_features p USING (SK_ID_CURR)
ORDER BY a.SK_ID_CURR;\n'''
    return converted


def prepare(root,v):
    root=Path(root).resolve();normal=load_normalized_bundle(root,root/v['normalization_report'])
    history=load_history_bundle(root,root/v['history_report'])
    if Path(history['bundle']['input_run_path']).resolve()!=(root/v['normalization_report']).resolve():raise DataError('spark_lineage','E04 must reference selected E03')
    sql=source_sql(root,history['bundle']['sql_sha256'])
    return {'version':v['version'],'execution_status':v['execution_status'],'SparkSession_started':False,
        'source_reports':{name:{'path':v[name],'sha256':sha256_file(root/v[name])} for name in ['normalization_report','history_report']},
        'normalization_directory':str(normal['directory']),'reference_directory':str(history['directory']),
        'read_columns':{'application_train.csv':['SK_ID_CURR'],
            'bureau.csv':['SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT_UPDATE','CREDIT_CURRENCY','CREDIT_ACTIVE','AMT_CREDIT_SUM'],
            'previous_application.csv':['SK_ID_CURR','DAYS_DECISION','NAME_CONTRACT_TYPE']},
        'output_columns':['SK_ID_CURR',*FEATURE_NAMES],'reference_projection_only':True,
        'engine_settings':{k:v[k] for k in ['spark_master','spark_driver_memory','shuffle_partitions','spark_ui_enabled','duckdb_threads','duckdb_memory_limit']},
        'tolerances':{'absolute':v['absolute_tolerance'],'relative':v['relative_tolerance'],'integer_ID_null_exact':True},
        'generated_sql':sql,'source_SQL_sha256':history['bundle']['sql_sha256'],
        'live_validation_remaining':['SparkSession runtime compatibility','real Spark edge-case fixtures','result materialization and cleanup failure tests','full 3-table aggregation and ID/type/null/value checks','fresh matched-projection DuckDB/Spark timings and process resource measurement','independent official integration review'],
        'training_or_test_evaluation_in_scope':False,'target_is_read':False,'frozen_V1_outputs_replaced':False}


def available_memory():
    if sys.platform!='win32':return None
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
            *[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]]
    data=Memory();data.length=ctypes.sizeof(data)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):return None
    return data.available


def require_memory(available,minimum=3*1024**3):
    if type(available) is not int or available<minimum:raise DataError('spark_memory','At least 3 GiB observed available memory required; unknown is not unlimited')


def reject_deferred_execution(v):
    if v.get('execution_status')=='deferred_by_user':raise DataError('spark_user_deferred','User requested preparation only. Do not start Spark; resume explicitly in a later step.')
    raise DataError('spark_runtime_not_validated','This preparation revision does not implement a validated live executor')


def preflight(root,v):
    """Read installed metadata and Java evidence without starting a Java/Spark job."""
    versions={}
    for name in ['pyspark','py4j','duckdb','pyarrow']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    candidates=[]
    for p in (Path(root)/'reports/e13_validation').glob('environment_preparation*.json'):
        obj=json.loads(p.read_text(encoding='utf-8'))
        if obj.get('passed') is True:candidates.append((p,obj))
    java_verified=False;java_version=None
    if len(candidates)==1:
        _,obj=candidates[0];p=Path(obj['java_home'])/'bin/java.exe'
        java_verified=p.is_file() and sha256_file(p)==obj['java_executable_sha256'];java_version=obj['java_version']
    memory=available_memory()
    return {'versions':versions,'versions_match_preparation':versions=={'pyspark':'4.2.0','py4j':'0.10.9.9','duckdb':'2.0.0','pyarrow':'26.0.0'},
        'java_preparation_verified':java_verified,'java_version':java_version,'observed_available_bytes':memory,
        'minimum_available_bytes':v['minimum_available_bytes'],'memory_budget_met':type(memory) is int and memory>=v['minimum_available_bytes'],
        'execution_status':v['execution_status'],'run_allowed':False,'SparkSession_started':False,
        'note':'Installation/import/Java version checks do not prove runtime Spark compatibility.'}


def compare_projection(reference,actual):
    """ID-sorted comparison; fixed tolerances, strict IDs/types/nulls/nonfinites."""
    if reference.column_names!=actual.column_names or reference.num_rows!=actual.num_rows or reference.schema.types!=actual.schema.types:raise DataError('spark_compare_schema','Column order, types or population differs')
    if 'SK_ID_CURR' not in reference.column_names or reference['SK_ID_CURR'].type!=pa.int64():raise DataError('spark_compare_ID','int64 application IDs required')
    for table in [reference,actual]:
        if table['SK_ID_CURR'].null_count or pc.count_distinct(table['SK_ID_CURR']).as_py()!=table.num_rows:raise DataError('spark_compare_ID','Unique non-null IDs required')
    left=reference.take(pc.sort_indices(reference,sort_keys=[('SK_ID_CURR','ascending')]))
    right=actual.take(pc.sort_indices(actual,sort_keys=[('SK_ID_CURR','ascending')]))
    if not left['SK_ID_CURR'].equals(right['SK_ID_CURR']):raise DataError('spark_compare_ID','Application ID set differs')
    result={}
    for field in left.schema:
        a,b=left[field.name],right[field.name]
        if not pc.is_null(a).equals(pc.is_null(b)):raise DataError('spark_compare_null','Null mask differs')
        if not pa.types.is_floating(field.type):
            if not a.equals(b):raise DataError('spark_compare_value','Integer/category differs')
            result[field.name]={'exact':True};continue
        maximum=0.;relative=0.
        for x,y in zip(a.to_pylist(),b.to_pylist()):
            if x is None:continue
            if not math.isfinite(x) or not math.isfinite(y):raise DataError('spark_compare_nonfinite','Nonfinite values are not accepted')
            delta=abs(x-y)
            if delta>1e-9+1e-12*abs(x):raise DataError('spark_compare_value','Difference exceeds predeclared E04 tolerances')
            maximum=max(maximum,delta)
            if x:relative=max(relative,delta/abs(x))
        result[field.name]={'max_absolute_difference':maximum,'max_relative_nonzero_reference':relative}
    return {'passed':True,'rows':left.num_rows,'columns':len(left.column_names),'comparison':result}
