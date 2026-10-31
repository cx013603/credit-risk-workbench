"""本机运行配置和逐次记录；不执行训练，不改变输入文件。"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import uuid

from .raw import DataError, TABLES, sha256_file


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    raw_dir: Path
    manifest: Path
    output_dir: Path
    tables: tuple[str, ...]
    path: Path
    sha256: str


def load_config(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        settings = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(settings) != {"schema_version", "raw_dir", "manifest", "output_dir", "tables"}:
            raise ValueError("unexpected configuration fields")
        if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
            raise ValueError("unsupported schema version")
        resolved = {}
        for field in ("raw_dir", "manifest", "output_dir"):
            value = settings[field]
            if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
                raise ValueError("paths must be nonempty project-relative strings")
            resolved[field] = (root / value).resolve()
            if not resolved[field].is_relative_to(root):
                raise ValueError("path leaves project root")
        if not resolved["output_dir"].is_relative_to(root / "reports"):
            raise ValueError("run output must stay in project reports")
        tables = settings["tables"]
        if not isinstance(tables, list) or not tables or any(not isinstance(name, str) or name not in TABLES for name in tables):
            raise ValueError("tables must be a nonempty list of registered names")
        if len(tables) != len(set(tables)):
            raise ValueError("duplicate configured tables")
        return RuntimeConfig(root, **resolved, tables=tuple(tables), path=path, sha256=sha256_file(path))
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise DataError("runtime_config_invalid", "Runtime config has invalid fields or paths") from exc


def process_peak_memory():
    """Windows 进程累计峰值工作集，包含当前解释器；不冒充整机／流水线峰值。"""
    if sys.platform != "win32":
        return {"peak_working_set_bytes": None, "method": "not_measured_on_this_platform"}
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                            "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error())
        return {"peak_working_set_bytes": counters.peak_working_set,
                "method": "Windows GetProcessMemoryInfo.PeakWorkingSetSize; cumulative process high-water mark"}
    except (OSError, AttributeError):
        return {"peak_working_set_bytes": None, "method": "Windows measurement unavailable"}


class RunRecord:
    def __init__(self, config, *, scope, metadata=None):
        self.run_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.directory = config.output_dir / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "run.json"
        self.data = {"run_id": self.run_id, "status": "running", "started_at": now(),
                     "finished_at": None, "scope": scope, "python_version": sys.version,
                     "config_path": str(config.path), "config_sha256": config.sha256,
                     "metadata": metadata or {}, "tables": {}}
        self.save()

    def save(self):
        temporary = self.path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def finish(self, passed, **fields):
        self.data.update(fields)
        self.data.update(status="passed" if passed else "failed", finished_at=now())
        self.save()
