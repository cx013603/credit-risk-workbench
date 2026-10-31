"""Single-flight bounded JSONL transport; a timeout invalidates the worker."""
import json
from pathlib import Path
import queue
import subprocess
import threading
import uuid

class WorkerFailure(Exception):
    def __init__(self,code,status=503):self.code,self.status=code,status;super().__init__(code)

class WorkerClient:
    def __init__(self,command,output,startup_timeout=120,request_timeout=20,max_response_bytes=1024*1024):
        self.lock=threading.Lock();self.messages=queue.Queue(maxsize=2);self.failed=False;self.request_timeout=request_timeout
        self.stderr=(Path(output)/'worker.stderr.log').open('x',encoding='utf-8')
        try:self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',bufsize=1)
        except BaseException:self.stderr.close();raise
        def reader():
            try:
                while True:
                    line=self.process.stdout.readline(max_response_bytes+1)
                    if not line:break
                    if len(line.encode('utf-8'))>max_response_bytes:raise ValueError('IPC_SIZE')
                    value=json.loads(line);json.dumps(value,allow_nan=False)
                    self.messages.put_nowait(value)
            except Exception:pass
            finally:
                try:self.messages.put_nowait({'type':'fatal','error':'WORKER_EXITED'})
                except queue.Full:pass
        self.reader=threading.Thread(target=reader,daemon=True)
        self.reader.start()
        try:
            self.ready=self.messages.get(timeout=startup_timeout)
            if self.ready.get('type')!='ready':raise WorkerFailure(self.ready.get('error','WORKER_START_FAILED'))
        except Exception:
            self.close();raise

    def request(self,population,ident):
        return self.exchange({'population':population,'application_id':ident},
                             {'population':population,'application_id':ident},identity_key='case_key')

    def exchange(self,message,identity,identity_key='identity'):
        if not self.lock.acquire(blocking=False):raise WorkerFailure('BUSY',429)
        try:
            if self.failed or self.process.poll() is not None:raise WorkerFailure('WORKER_UNAVAILABLE')
            rid=uuid.uuid4().hex
            encoded=json.dumps({'request_id':rid,**message},allow_nan=False)+'\n'
            if len(encoded.encode('utf-8'))>4096:raise WorkerFailure('INPUT_SIZE',413)
            self.process.stdin.write(encoded);self.process.stdin.flush()
            try:reply=self.messages.get(timeout=self.request_timeout)
            except queue.Empty:
                self.failed=True;self.close();raise WorkerFailure('WORKER_TIMEOUT',504)
            if reply.get('type')=='fatal':self.failed=True;raise WorkerFailure('WORKER_EXITED')
            if reply.get('request_id')!=rid:self.failed=True;self.close();raise WorkerFailure('IPC_IDENTITY_MISMATCH')
            if reply.get('type')=='error':
                code=reply['error']['code'];status={'CASE_NOT_FOUND':404,'SAMPLE_USE_NOT_ALLOWED':403,'VERSION_MISMATCH':409}.get(code,400 if code.startswith('INVALID') else 503)
                raise WorkerFailure(code,status)
            if reply.get('type')!='result' or not isinstance(reply.get('payload'),dict):self.failed=True;self.close();raise WorkerFailure('IPC_PROTOCOL')
            if reply.get(identity_key)!=identity:self.failed=True;self.close();raise WorkerFailure('IPC_IDENTITY_MISMATCH')
            return reply
        except (OSError,ValueError):self.failed=True;raise WorkerFailure('WORKER_UNAVAILABLE')
        finally:self.lock.release()

    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.close();self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self.process.kill();self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        for stream in [self.process.stdin,self.process.stdout,self.stderr]:
            if stream and not stream.closed:stream.close()
