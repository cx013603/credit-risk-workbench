"""E13 real local engines, shared bounded Arrow sink and owned-process cleanup."""
import contextlib
import ctypes
from ctypes import wintypes
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .history import FEATURES,FEATURE_NAMES
from .raw import DataError,sha256_file
from .runtime import now
from .spark_compare import available_memory,require_memory

BATCH=8192


def process_memory(pid):
    if os.name!='nt':return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),*[(n,ctypes.c_size_t) for n in ['peak','current','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
    if not handle:return None
    try:
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):return None
        return {'current_bytes':counters.current,'peak_bytes':counters.peak}
    finally:kernel.CloseHandle(handle)


def process_alive(pid):
    if os.name!='nt':return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)


class Engine:
    def __init__(self,name,root,work):
        self.name=name;self.root=Path(root);self.work=Path(work);self.spark=None;self.con=None;self.jvm_pid=None
        self.gateway=None;self.process=None;self.jvm=None;self.sampled_peak=0;self.samples=0;self.finished=threading.Event()
        self.sampler=threading.Thread(target=self._sample,daemon=True);self.sampler.start();start=time.monotonic()
        try:
            if name=='duckdb':
                self.con=duckdb.connect();self.con.execute("SET threads=1");self.con.execute("SET memory_limit='512MB'")
                self.con.execute("SET explain_output='all'")
                self.con.execute("SET temp_directory='"+str(self.work/'spill').replace("'","''")+"'")
            elif name=='spark':
                require_memory(available_memory())
                p=self.root/'reports/e13_validation/environment_preparation_retry2.json';environment=json.loads(p.read_text(encoding='utf-8'))
                java=Path(environment['java_home'])
                if not environment['passed'] or sha256_file(java/'bin/java.exe')!=environment['java_executable_sha256']:raise DataError('spark_java_changed','Accepted portable Java required')
                alias_info=json.loads((self.root/'reports/e13_validation/windows_runtime_alias.json').read_text(encoding='utf-8'))
                alias=Path(alias_info['ASCII_bootstrap_root'])
                if not str(alias).isascii() or alias.resolve()!=self.root:raise DataError('spark_alias','Verified same-project ASCII bootstrap alias required')
                java=alias/java.relative_to(self.root);python=alias/Path(os.sys.executable).relative_to(self.root)
                os.environ.update(JAVA_HOME=str(java),SPARK_LOCAL_IP='127.0.0.1',PYSPARK_PYTHON=str(python),PYSPARK_DRIVER_PYTHON=str(python),
                    PYSPARK_SUBMIT_ARGS='--driver-memory 1g pyspark-shell')
                os.environ['PATH']=str(java/'bin')+os.pathsep+os.environ.get('PATH','')
                temporary=alias/'.local-tmp/e13-spark/runtime-temp';temporary.mkdir(exist_ok=True)
                os.environ.update(TEMP=str(temporary),TMP=str(temporary));tempfile.tempdir=str(temporary)
                import pyspark
                os.environ['SPARK_HOME']=str(alias/Path(pyspark.__file__).parent.relative_to(self.root))
                from pyspark.sql import SparkSession
                builder=SparkSession.builder.master('local[1]').appName('credit-risk-E13')
                settings={'spark.driver.host':'127.0.0.1','spark.driver.bindAddress':'127.0.0.1','spark.driver.memory':'1g',
                    'spark.sql.shuffle.partitions':'4','spark.default.parallelism':'1','spark.sql.adaptive.enabled':'false',
                    'spark.ui.enabled':'false','spark.ui.showConsoleProgress':'false','spark.local.dir':str(alias/self.work.relative_to(self.root)/'spill'),
                    'spark.sql.warehouse.dir':(alias/self.work.relative_to(self.root)/'warehouse').as_uri(),
                    'spark.driver.extraJavaOptions':'-Dfile.encoding=UTF-8 -XX:ActiveProcessorCount=1'}
                for k,v in settings.items():builder=builder.config(k,v)
                self.spark=builder.getOrCreate();self.spark.sparkContext.setLogLevel('ERROR')
                self.gateway=self.spark.sparkContext._gateway;self.process=self.gateway.proc
                self.jvm=self.spark.sparkContext._jvm
                self.jvm_pid=int(self.spark.sparkContext._jvm.java.lang.ProcessHandle.current().pid())
                if self.spark.sparkContext.master!='local[1]':raise DataError('spark_master','Only local[1] accepted')
                if int(self.spark.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory())>1100*1024**2:raise DataError('spark_heap','Driver heap exceeds fixed budget')
                self.settings=settings
            else:raise DataError('spark_engine','Unknown engine')
            self.initialization_seconds=time.monotonic()-start
        except BaseException:
            module=sys.modules.get('pyspark')
            if self.gateway is None and module is not None:
                self.gateway=getattr(module.SparkContext,'_gateway',None)
                if self.gateway is not None:
                    self.process=self.gateway.proc;self.jvm=self.gateway.jvm
                    try:self.jvm_pid=int(self.jvm.java.lang.ProcessHandle.current().pid())
                    except Exception:pass
            self.close();raise

    def _sample(self):
        while not self.finished.wait(.1):
            p=process_memory(os.getpid());j=process_memory(self.jvm_pid) if self.jvm_pid else None
            if p is not None and (self.jvm_pid is None or j is not None):
                self.sampled_peak=max(self.sampled_peak,p['current_bytes']+(j['current_bytes'] if j else 0));self.samples+=1

    def sql(self,text):
        if self.con is not None:return self.con.execute(text)
        return self.spark.sql(text)

    def view(self,name,path,columns=None):
        if self.con is not None:
            projection=','.join(columns) if columns else '*'
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT {projection} FROM read_parquet('"+str(path).replace("'","''")+"')")
        else:
            frame=self.spark.read.parquet(str(path))
            if columns:frame=frame.select(*columns)
            frame.createOrReplaceTempView(name)

    def statements(self,text):
        for statement in text.split(';'):
            if statement.strip():self.sql(statement)

    def batches(self,query,schema,plan):
        if self.con is not None:
            rows=self.con.execute('EXPLAIN '+query).fetchall();plan.write_text('\n'.join(str(x) for x in rows),encoding='utf-8')
            yield from self.con.sql(query).to_arrow_reader(batch_size=BATCH)
        else:
            frame=self.spark.sql(query)
            actual=[f.dataType.simpleString() for f in frame.schema.fields]
            expected=['bigint' if pa.types.is_int64(f.type) else 'double' for f in schema]
            if frame.columns!=schema.names or actual!=expected:raise DataError('spark_schema','Spark output types/order differ before Arrow conversion')
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):frame.explain(mode='extended')
            plan.write_text(stream.getvalue(),encoding='utf-8')
            rows=[]
            for row in frame.toLocalIterator(prefetchPartitions=False):
                rows.append(row.asDict())
                if len(rows)==BATCH:yield pa.RecordBatch.from_pylist(rows,schema=schema);rows=[]
            if rows:yield pa.RecordBatch.from_pylist(rows,schema=schema)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):frame.explain(mode='formatted')
            plan.with_suffix('.physical.txt').write_text(stream.getvalue(),encoding='utf-8')

    def resources(self):
        return {'python':process_memory(os.getpid()),'JVM':process_memory(self.jvm_pid) if self.jvm_pid else None,
            'sampled_simultaneous_working_set_peak_bytes':self.sampled_peak if self.samples else None,'samples':self.samples,
            'method':'Windows process working sets sampled every 100ms; JVM included after PID discovery; individual cumulative peaks are separate, never summed'}

    def close(self):
        start=time.monotonic();error=None
        if self.con is not None:self.con.close();self.con=None
        if self.spark is not None:
            try:self.spark.stop()
            except Exception as exc:error=type(exc).__name__
            self.spark=None
        if self.jvm is not None:
            # Only this worker's verified JVM. SparkContext.stop does not exit
            # the Windows gateway process; run normal JVM shutdown hooks too.
            try:self.jvm.java.lang.System.exit(0)
            except Exception:pass  # The gateway connection closes on normal exit.
        if self.process is not None:
            try:
                if self.process.stdin and not self.process.stdin.closed:self.process.stdin.close()
                self.process.wait(timeout=25)
            except Exception as exc:error=type(exc).__name__
        self.finished.set();self.sampler.join(2)
        stopped=not process_alive(self.jvm_pid) if self.jvm_pid else None
        self.shutdown={'seconds':time.monotonic()-start,'JVM_stopped':stopped,'error':error}
        if error or stopped is False:raise DataError('spark_shutdown','Owned engine did not stop cleanly')


def output_schema(name):
    fields=[('SK_ID_CURR',pa.int64())]
    for key,kind,*_ in FEATURES:
        if name=='joined_features' or (name=='bureau_features' and key.startswith('BUREAU_')) or (name=='previous_features' and key.startswith('PREV_')):
            fields.append((key,pa.int64() if kind=='int64' else pa.float64()))
    return pa.schema(fields)


def write_batches(target,schema,batches):
    target=Path(target);partial=target.with_suffix('.parquet.partial')
    if target.exists() or partial.exists():raise DataError('spark_output_exists','Never overwrite a prior output')
    rows=0
    try:
        with partial.open('xb') as output,pq.ParquetWriter(output,schema,compression='snappy') as writer:
            for batch in batches:
                if batch.schema.names!=schema.names or batch.schema.types!=schema.types:raise DataError('spark_sink_schema','Batch schema differs')
                writer.write_batch(batch);rows+=batch.num_rows
        if pq.read_metadata(partial).num_rows!=rows:raise DataError('spark_sink_rows','Parquet row receipt differs')
        partial.rename(target)
        return {'rows':rows,'bytes':target.stat().st_size,'sha256':sha256_file(target)}
    except BaseException:
        if partial.exists():partial.unlink()
        raise


def execute_job(name,root,inputs,sql,out,read_columns):
    root=Path(root).resolve();out=Path(out).resolve()
    if not out.is_relative_to(root/'reports') or out.exists():raise DataError('spark_output','A fresh report-local job directory is required')
    staging=out.with_name(out.name+'.partial');staging.mkdir(parents=True,exist_ok=False);engine=None
    try:
        start=time.monotonic();engine=Engine(name,root,staging);stages={};files={}
        aliases={'application_train.csv':'application_input','bureau.csv':'bureau_input','previous_application.csv':'previous_input'}
        tick=time.monotonic()
        for table,alias in aliases.items():engine.view(alias,inputs[table],read_columns[table])
        for key in ['02_bureau_features','03_previous_features']:engine.statements(sql[key])
        prepare_seconds=time.monotonic()-tick
        for table in ['bureau_features','previous_features','joined_features']:
            query=sql['04_join_history_projection'] if table=='joined_features' else f'SELECT * FROM {table} ORDER BY SK_ID_CURR'
            schema=output_schema(table);tick=time.monotonic()
            files[table+'.parquet']=write_batches(staging/(table+'.parquet'),schema,engine.batches(query,schema,staging/(table+'_plan.txt')))
            stages[table]=time.monotonic()-tick
            if table!='joined_features':engine.view(table,staging/(table+'.parquet'))
        resource=engine.resources();initial=engine.initialization_seconds;engine.close();shutdown=engine.shutdown;engine=None
        result={'engine':name,'created_at':now(),'initialization_seconds':initial,'source_views_and_planning_seconds':prepare_seconds,'query_and_materialize_seconds':stages,'job_elapsed_seconds':time.monotonic()-start,
            'resources':resource,'shutdown':shutdown,'files':files,'sink':'shared PyArrow Snappy Parquet, batches8192; Spark uses JVM-to-Python rows; no Hadoop writer or Python UDF',
            'outputs':{'bureau_features':len(output_schema('bureau_features')),'previous_features':len(output_schema('previous_features')),'joined_features':25}}
        (staging/'job.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        staging.rename(out);return result
    finally:
        try:
            if engine is not None:engine.close()
        finally:
            if staging.exists():
                if not staging.is_relative_to(root/'reports') or not staging.name.endswith('.partial'):raise DataError('spark_cleanup','Unsafe job cleanup')
                shutil.rmtree(staging)
