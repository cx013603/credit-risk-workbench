"""Owned local lifecycle. Stop signals only this controller's nonce-bound run."""
from contextlib import contextmanager
import ctypes,json,os,re,socket,subprocess,sys,time,urllib.request,uuid,webbrowser
from ctypes import wintypes
from pathlib import Path
from .delivery import need,no_links,read,save,verify,setup,env,DeliveryError,BASE_SEAL

def signature(pid):
    if os.name!='nt':raise DeliveryError('本交付仅验证Windows x64')
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE;k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x1000,False,pid)
    if not handle:return None
    try:
        created=wintypes.FILETIME();exit_time=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
        k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        if not k.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exit_time),ctypes.byref(kernel),ctypes.byref(user)):return None
        # GetProcessTimes can succeed on a terminated process whose handle remains open.
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];code=wintypes.DWORD()
        if not k.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=259:return None
        size=wintypes.DWORD(32768);name=ctypes.create_unicode_buffer(size.value);k.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
        if not k.QueryFullProcessImageNameW(handle,0,name,ctypes.byref(size)):return None
        return {'pid':pid,'created_100ns':(created.dwHighDateTime<<32)|created.dwLowDateTime,'image':name.value}
    finally:k.CloseHandle(handle)
def same(sig):return isinstance(sig,dict) and signature(sig.get('pid',0))==sig
def memory(pid):
    class PM(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),*[(n,ctypes.c_size_t) for n in ['peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]]
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE;k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(0x410,False,pid)
    if not h:return {'pid':pid,'available':False}
    try:
        value=PM();value.cb=ctypes.sizeof(value);p=ctypes.WinDLL('psapi');p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        ok=p.GetProcessMemoryInfo(h,ctypes.byref(value),value.cb)
        return {'pid':pid,'available':bool(ok),'working_set_bytes':value.ws,'peak_working_set_bytes':value.peak_ws}
    finally:k.CloseHandle(h)
def port_owner(port):
    lib=ctypes.WinDLL('iphlpapi');size=wintypes.DWORD(0);fn=lib.GetExtendedTcpTable;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),wintypes.BOOL,wintypes.ULONG,ctypes.c_int,wintypes.ULONG]
    code=fn(None,ctypes.byref(size),False,2,5,0);need(code in [0,122],'无法核对端口归属')
    data=ctypes.create_string_buffer(size.value);need(fn(data,ctypes.byref(size),False,2,5,0)==0,'无法核对端口归属')
    class Row(ctypes.Structure):_fields_=[(n,wintypes.DWORD) for n in ['state','local_addr','local_port','remote_addr','remote_port','pid']]
    count=wintypes.DWORD.from_buffer(data).value;owners=[]
    for i in range(count):
        row=Row.from_buffer(data,4+i*ctypes.sizeof(Row))
        if row.state==2 and socket.ntohs(row.local_port&0xffff)==port:owners.append(row.pid)
    return owners
def paths(directory,tag):
    r=no_links(directory);need(re.fullmatch('[a-f0-9]{32}',tag) is not None,'运行ID不符');return r,r/'var/runs'/tag
@contextmanager
def control_lock(r):
    p=r/'var/control.lock';r.joinpath('var').mkdir(exist_ok=True)
    if p.exists():
        v=read(p);need(not same(v.get('owner')),'已有启动或关闭正在执行');p.rename(p.with_name('stale-lock-'+uuid.uuid4().hex+'.json'))
    save(p,{'owner':signature(os.getpid())},True)
    try:yield
    finally:p.unlink(missing_ok=True)
def status(directory):
    r=no_links(directory);p=r/'var/current.json'
    if not p.exists():return {'running':False,'reason':'尚无本启动器实例'}
    c=read(p)
    if not same(c.get('supervisor')):return {'running':False,'reason':'旧运行记录已失效；未向其PID发停止信号','previous_run':c.get('run_id')}
    ready=read(r/'var/runs'/c['run_id']/'ready.json') if (r/'var/runs'/c['run_id']/'ready.json').exists() else None
    if not ready:return {'running':False,'starting':True,'run_id':c['run_id']}
    need(same(ready['api_signature']) and same(ready['worker_signature']),'实例进程身份已失效')
    need(ready['service']['api_pid'] in port_owner(int(ready['url'].rsplit(':',1)[1])),'端口不属于本实例')
    with urllib.request.urlopen(ready['url']+'/api/v2/health/ready',timeout=5) as response:health=json.load(response)
    need(health['payload']['ready'] and health['payload']['release_manifest_sha256']==BASE_SEAL and health['payload']['worker_load_attempts']==1,'健康或版本不符')
    return {'running':True,'run_id':c['run_id'],'url':ready['url'],'supervisor':c['supervisor'],'api':ready['api_signature'],'worker':ready['worker_signature'],'memory':[memory(ready['api_signature']['pid']),memory(ready['worker_signature']['pid'])],'logs':str(r/'var/runs'/c['run_id'])}
def start(directory,open_browser=True,port=0):
    started=time.monotonic();r=no_links(directory);need(type(port) is int and 0<=port<=65535,'端口不符')
    print('正在校验版本与文件…',flush=True);setup(r)
    with control_lock(r):
        current=status(r)
        if current.get('running'):
            if open_browser:webbrowser.open(current['url']+('/?model=71' if (r/'workbench/config/workbench/candidate71.json').is_file() else ''))
            return {**current,'reused':True}
        need(not current.get('starting'),'已有实例启动中')
        old=r/'var/current.json'
        if old.exists():old.rename(old.with_name('previous-'+uuid.uuid4().hex+'.json'))
        tag=uuid.uuid4().hex;out=r/'var/runs'/tag;out.mkdir()
        log=(out/'supervisor.log').open('xb');command=[str(r/'python/python.exe'),'-I','-B','-X','utf8',str(r/'workbench/scripts/workbench_control.py'),'supervise','--directory',str(r),'--run-id',tag,'--port',str(port)]
        p=subprocess.Popen(command,cwd=r,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);log.close()
        sig=signature(p.pid);need(sig is not None,'启动器进程没有存活');save(r/'var/current.json',{'run_id':tag,'supervisor':sig},True)
        print('正在加载一次保存模型；日志：'+str(out),flush=True)
        try:
            while time.monotonic()-started<240:
                if (out/'failed.json').exists():raise DeliveryError(read(out/'failed.json')['message'])
                if p.poll() is not None:raise DeliveryError('启动进程提前退出，请看专属日志')
                if (out/'ready.json').exists():
                    result=status(r);need(result['running'],'就绪核对失败');result['cold_start_seconds']=round(time.monotonic()-started,3);result['reused']=False;save(out/'start-receipt.json',result,True)
                    if open_browser:webbrowser.open(result['url']+('/?model=71' if (r/'workbench/config/workbench/candidate71.json').is_file() else ''))
                    return result
                time.sleep(.15)
            raise DeliveryError('启动超时；已请求停止本次实例')
        except BaseException:
            (out/'stop.request').touch(exist_ok=True);raise
def stop(directory):
    r=no_links(directory)
    with control_lock(r):
        current=r/'var/current.json'
        if not current.exists():return {'running':False,'reason':'尚无本启动器实例'}
        c=read(current)
        if not same(c.get('supervisor')):return {'running':False,'reason':'旧实例已失效，未发送停止信号'}
        tag=c['run_id'];out=r/'var/runs'/tag
        if not (out/'stop.request').exists():save(out/'stop.request',{'run_id':tag},True)
        deadline=time.monotonic()+55
        while time.monotonic()<deadline:
            if (out/'stopped.json').exists():return read(out/'stopped.json')
            time.sleep(.15)
        raise DeliveryError('关闭尚未确认；保留专属停止文件和日志，不向其他PID发送终止')
def supervise(directory,tag,port):
    r,out=paths(directory,tag);job=None;p=None;ready=None;forced=False
    try:
        from credit_risk.owned_process import OwnedJob
        job=OwnedJob();stdout=(out/'api.stdout.log').open('xb');stderr=(out/'api.stderr.log').open('xb')
        command=[str(r/'api-env/Scripts/python.exe'),'-I','-B','-X','utf8',str(r/'workbench/scripts/workbench_service.py'),'--run-id',tag,'--directory',str(r),'--port',str(port)]
        p=subprocess.Popen(command,cwd=r/'workbench',env=env(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW);stdout.close();stderr.close()
        try:job.assign(p)
        except BaseException:p.terminate();p.wait(timeout=5);raise
        save(out/'ownership-ticket.json',{'run_id':tag,'supervisor':signature(os.getpid())},True)
        limit=time.monotonic()+210
        while time.monotonic()<limit and not (out/'stop.request').exists():
            if (out/'service/failed.json').exists():raise DeliveryError('API启动失败：'+str(read(out/'service/failed.json')))
            if p.poll() is not None:raise DeliveryError('API进程提前退出')
            if (out/'service/ready.json').exists():
                service=read(out/'service/ready.json');ready={'service':service,'url':service['url'],'api_signature':signature(service['api_pid']),'worker_signature':signature(service['worker']['pid'])};need(ready['api_signature'] and ready['worker_signature'],'服务身份无法确认');save(out/'ready.json',ready,True);break
            time.sleep(.15)
        need(ready is not None,'服务未按时就绪')
        while not (out/'stop.request').exists():
            if p.poll() is not None:raise DeliveryError('API意外退出，实例已关闭')
            time.sleep(.2)
    except BaseException as exc:save(out/'failed.json',{'message':str(exc),'run_id':tag})
    finally:
        if p and p.poll() is None:
            (out/'api.stop').touch(exist_ok=True)
            try:p.wait(timeout=42)
            except subprocess.TimeoutExpired:forced=True
        if job:job.close()
        if p and p.poll() is None:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        released=ready is None or ready['service']['api_pid'] not in port_owner(int(ready['url'].rsplit(':',1)[1]))
        save(out/'stopped.json',{'passed':p is None or p.poll() is not None,'run_id':tag,'owned_job_closed':job is not None,'forced_owned_tree':forced,'API_returncode':p.returncode if p else None,'owned_port_released':released},True)
