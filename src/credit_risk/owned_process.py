"""Windows job object: an E13 worker and its descendants share owned teardown."""
import ctypes
from ctypes import wintypes
import os
import subprocess
from types import SimpleNamespace
from .raw import DataError


class OwnedJob:
    def __init__(self):
        if os.name!='nt':raise DataError('owned_job_platform','Validated Windows process-tree ownership required')
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),
                ('minimum_ws',ctypes.c_size_t),('maximum_ws',ctypes.c_size_t),('processes',wintypes.DWORD),
                ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_uint64) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),*[(n,ctypes.c_size_t) for n in ['process_memory','job_memory','peak_process','peak_job']]]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];self.kernel.CreateJobObjectW.restype=wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle:raise DataError('owned_job_create','Cannot create owned job object')
        info=Extended();info.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            self.close();raise DataError('owned_job_limit','Cannot enforce kill-on-close')

    def assign(self,process):
        if not self.kernel.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle))):raise DataError('owned_job_assign','Cannot own worker tree; refuse execution')

    def close(self):
        if self.handle:self.kernel.CloseHandle(self.handle);self.handle=None


def run_owned(command,*,cwd,timeout):
    job=OwnedJob();process=None;timed_out=False
    try:
        process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:job.assign(process)
        except BaseException:
            process.terminate();process.communicate(timeout=20);raise
        try:stdout,stderr=process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;job.close();stdout,stderr=process.communicate(timeout=20)
        def decode(value):
            try:return value.decode('utf-8'),True
            except UnicodeDecodeError:return value.decode('utf-8',errors='replace'),False
        text_out,out_valid=decode(stdout);text_err,err_valid=decode(stderr)
        return SimpleNamespace(returncode=process.returncode,stdout=text_out,stderr=text_err,stdout_bytes=stdout,stderr_bytes=stderr,
            stdout_utf8_valid=out_valid,stderr_utf8_valid=err_valid,timed_out=timed_out,owned_tree_kill_on_close=True)
    finally:job.close()
