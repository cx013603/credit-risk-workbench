"""E04：从成功全量 E03 报告读取规范表，分别聚合历史后关联申请；不训练。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from _project import ROOT
from credit_risk.raw import DataError, RawDataSource, require_data_scope, sha256_file
from credit_risk.runtime import load_config, RunRecord, process_peak_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True, help="passed、全量 E03 run.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/runtime.json")
    parser.add_argument("--history-config", type=Path, default=ROOT / "config/history_features.json")
    parser.add_argument("--application-limit", type=int, help="只处理主表原顺序前N笔及其全部历史；省略为全量")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    record, stage, started = None, "configuration", time.monotonic()
    try:
        runtime = load_config(ROOT, args.config)
        scope = ("synthetic_" if args.fixture else "official_") + ("full_" if args.application_limit is None else "prefix_") + "history_features"
        record = RunRecord(runtime, scope=scope, metadata={"module": "E04", "command": [sys.executable, *sys.argv]})
        stage = "history_configuration"
        from credit_risk.history import load_history_config, build_history_features, REQUIRED
        config = load_history_config(ROOT, args.history_config)
        if not set(REQUIRED).issubset(runtime.tables):
            raise DataError("history_tables_invalid", "History tables must be permitted by runtime")
        stage = "raw_scope"
        source = RawDataSource(args.data_dir or runtime.raw_dir, None if args.fixture else runtime.manifest)
        require_data_scope(source, ROOT / "data/raw", fixture=args.fixture)
        experiment = ROOT / "config/experiment_plan.json"
        experiment_before = sha256_file(experiment)
        code = ["scripts/run_history_features.py", "scripts/_project.py", "src/credit_risk/history.py",
                "src/credit_risk/normalize.py", "src/credit_risk/parquet.py", "src/credit_risk/raw.py", "src/credit_risk/runtime.py"]
        record.data["metadata"].update(
            input_run=str(args.input_run.resolve()), input_run_sha256=sha256_file(args.input_run),
            history_config_path=str(config.path), history_config_sha256=config.sha256,
            policy_version=config.values["policy_version"], sql_sha256={p:i["sha256"] for p,i in config.sql.items()},
            code_sha256={p:sha256_file(ROOT/p) for p in code},
            dependencies={p:importlib.metadata.version(p) for p in ("duckdb", "pyarrow")},
            dependency_lock_sha256=sha256_file(ROOT / "requirements-data.txt"),
            manifest_sha256=sha256_file(source.manifest_path) if source.manifest_path else None,
            data_root=str(source.root), read_only_inputs=True, experiment_sha256_before=experiment_before,
            application_limit=args.application_limit,
            resource_limits="DuckDB 1 thread/512MB; spill within unique staging directory; Arrow verification batches 8192")
        record.save()
        stage = "aggregate_join_verify_and_trace"
        def progress(name, info):
            record.data["tables"][name] = info
            record.save()
            print(f"{name}: verified rows={info['rows']}; seconds={info['elapsed_seconds']}", file=sys.stderr, flush=True)
        result = build_history_features(config, args.input_run, source, record.run_id, fixture=args.fixture,
                                        application_limit=args.application_limit, progress=progress)
        stage = "experiment_unchanged"
        if sha256_file(experiment) != experiment_before:
            raise DataError("experiment_changed", "Experiment configuration changed during E04")
        record.finish(True, history_features=result, experiment_unchanged=True,
                      elapsed_seconds=round(time.monotonic()-started, 3), **process_peak_memory())
        print(json.dumps({"passed":True, "run_id":record.run_id, "report":str(record.path),
                          "artifact_dir":result["artifact_dir"], "scope":result["scope"]}, ensure_ascii=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        error = exc.as_dict() if isinstance(exc, DataError) else {"code":type(exc).__name__, "message":str(exc)}
        if record is not None:
            record.finish(False, failure_stage=stage, error=error,
                          elapsed_seconds=round(time.monotonic()-started, 3), **process_peak_memory())
        print(json.dumps({"passed":False, "error":error, "report":str(record.path) if record else None}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
