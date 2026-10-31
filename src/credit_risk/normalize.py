"""E03：完整字段的确定性规范层；保留特殊值，不拟合、不去重、不筛样本。"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .parquet import make_schema, typed_batches, write_verified_parquet
from .quality import inspect_table
from .raw import DataError, TABLES, sha256_file
from .runtime import now, process_peak_memory

CORE_TABLES = ("application_train.csv", "bureau.csv", "previous_application.csv")


@dataclass(frozen=True)
class NormalizationConfig:
    artifact_dir: Path
    batch_size: int
    tables: dict
    path: Path
    sha256: str
    values: dict


def load_normalization_config(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(values) != {"schema_version", "artifact_dir", "batch_size", "policy", "tables", "dictionary_source"}:
            raise ValueError("unexpected config fields")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("unsupported schema")
        if type(values["batch_size"]) is not int or not 1 <= values["batch_size"] <= 10000:
            raise ValueError("invalid batch size")
        relative = values["artifact_dir"]
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("relative artifact path required")
        output = (root / relative).resolve()
        if not output.is_relative_to(root / "data/interim") or output.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe artifact root")
        if values["policy"] != "empty_to_null_other_values_preserved_no_row_filter_v1":
            raise ValueError("unsupported conversion policy")
        dictionary = values["dictionary_source"]
        if (not isinstance(dictionary, dict) or set(dictionary) != {"path", "sha256", "basis"}
                or dictionary["path"] != "data/raw/home_credit/HomeCredit_columns_description.csv"
                or not isinstance(dictionary["basis"], str) or not dictionary["basis"]
                or not isinstance(dictionary["sha256"], str) or len(dictionary["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in dictionary["sha256"])):
            raise ValueError("invalid dictionary provenance")
        tables = values["tables"]
        if not isinstance(tables, dict) or set(tables) != set(CORE_TABLES):
            raise ValueError("all three core tables required")
        for name, columns in tables.items():
            required_fields = {"name", "type", "role", "unit", "description", "dictionary_row", "dictionary_match",
                               "source_special", "time_meaning", "feature_availability"}
            if not isinstance(columns, list) or any(not isinstance(c, dict) or set(c) != required_fields for c in columns):
                raise ValueError("invalid column contract")
            schema = make_schema([{k: c[k] for k in ("name", "type")} for c in columns])
            for key in TABLES[name].required:
                if key not in schema.names or schema.field(key).type != pa.int64():
                    raise ValueError("required IDs and target must be int64")
            if any(not isinstance(c[k], str) or not c[k] for c in columns for k in required_fields):
                raise ValueError("column annotations must be nonempty strings")
        return NormalizationConfig(output, values["batch_size"], tables, path, sha256_file(path), values)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise DataError("normalization_config_invalid", "Invalid normalization schema, policy or path") from exc


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def reference_ids(source):
    """只使用主表键／必需标签检查；无标签表的非 ID 字段不参与检查或输出。"""
    result, reports = {}, {}
    for name, role in (("application_train.csv", "train"), ("application_test.csv", "unlabeled")):
        inspection = inspect_table(source, name, columns=["SK_ID_CURR"], hash_file=True)
        reports[name] = inspection.report
        if not inspection.report["passed"]:
            raise DataError("reference_quality_failed", "Application reference key quality failed", table=name)
        schema = make_schema([{"name": "SK_ID_CURR", "type": "int64"}])
        keys = set()
        for batch in typed_batches(source, name, schema, row_limit=None, batch_size=8192):
            keys.update(batch.column(0).to_pylist())
        if len(keys) != inspection.report["rows"] or sha256_file(source.path(name)) != inspection.report["sha256"]:
            raise DataError("reference_keys_changed", "Reference keys changed or collide after typing", table=name)
        result[role] = keys
    if result["train"] & result["unlabeled"]:
        raise DataError("reference_overlap", "Train and unlabeled IDs overlap")
    return result, reports


def profile_table(path, columns):
    """SQL 一次汇总数值范围与候选编码；它们仅作审计，不据此替换。"""
    expressions, layouts = [], []
    for column in columns:
        name, kind = column["name"], column["type"]
        c = quoted(name)
        metrics = {"nulls": f"count(*) - count({c})"}
        if kind != "string":
            metrics.update(minimum=f"min({c})", maximum=f"max({c})", negative=f"count(*) FILTER (WHERE {c}<0)",
                           zero=f"count(*) FILTER (WHERE {c}=0)", positive=f"count(*) FILTER (WHERE {c}>0)")
            for value in (-1, -999, -365243, 365243):
                metrics[f"candidate_code_{value}"] = f"count(*) FILTER (WHERE {c}={value})"
        else:
            metrics["distinct_nonnull"] = f"count(DISTINCT {c})"
            for value in ("XNA", "XAP", "Unknown", "NULL", "NA"):
                metrics[f"literal_{value}"] = f"count(*) FILTER (WHERE {c}='{value}')"
        for key, expression in metrics.items():
            expressions.append(expression)
            layouts.append((name, key))
    with duckdb.connect(config={"threads": "1", "memory_limit": "512MB"}) as con:
        row = con.execute("SELECT " + ",".join(expressions) + " FROM read_parquet(?)", [str(path)]).fetchone()
        result = {c["name"]: {"type": c["type"], "unit": c["unit"], "time_meaning": c["time_meaning"],
                             "feature_availability": c["feature_availability"]} for c in columns}
        for (name, metric), value in zip(layouts, row):
            result[name][metric] = value
        for column in columns:
            if column["type"] == "string":
                c = quoted(column["name"])
                values = con.execute(f"SELECT {c},count(*) n FROM read_parquet(?) WHERE {c} IS NOT NULL "
                                     f"GROUP BY {c} ORDER BY n DESC,{c} LIMIT 25", [str(path)]).fetchall()
                result[column["name"]]["top_values"] = [{"value": v, "rows": n} for v, n in values]
    return result


def check_relations(paths, refs):
    """只验证三张实际使用表；返回两组主表覆盖及真正的孤儿，绝不拼接两张历史原表。"""
    reports = {}
    for name, path in paths.items():
        key = TABLES[name].primary_key
        with duckdb.connect(config={"threads": "1", "memory_limit": "512MB"}) as con:
            rows, distinct, missing = con.execute(
                f"SELECT count(*),count(DISTINCT {quoted(key)}),count(*)-count({quoted(key)}) FROM read_parquet(?)",
                [str(path)]).fetchone()
        if rows != distinct or missing:
            raise DataError("normalized_key_invalid", "Typed primary key is missing or duplicated", table=name)
        seen, counts = set(), {"train": 0, "unlabeled": 0, "orphan": 0, "missing_parent": 0}
        with pq.ParquetFile(path) as stored:
            for batch in stored.iter_batches(batch_size=8192, columns=["SK_ID_CURR"]):
                for value in batch.column(0).to_pylist():
                    if value is None:
                        counts["missing_parent"] += 1
                    elif value in refs["train"]:
                        counts["train"] += 1
                        seen.add(value)
                    elif value in refs["unlabeled"]:
                        counts["unlabeled"] += 1
                        seen.add(value)
                    else:
                        counts["orphan"] += 1
        report = {"rows": rows, "primary_key": key, "primary_key_distinct": distinct,
                  "primary_key_missing": missing, "parent_row_counts": counts,
                  "reference_population": "application_train UNION application_test; disjoint checked",
                  "coverage": {role: {"reference_ids": len(ids), "matched_ids": len(seen & ids),
                                       "without_history_ids": len(ids - seen),
                                       "fraction": len(seen & ids) / len(ids) if ids else None}
                               for role, ids in refs.items()}}
        if counts["orphan"] or counts["missing_parent"]:
            raise DataError("history_parent_invalid", "Historical parent missing or outside both application tables", table=name)
        if name == "application_train.csv" and counts["unlabeled"]:
            raise DataError("application_population_changed", "Labeled output contains unlabeled reference IDs", table=name)
        if name == "application_train.csv" and seen != refs["train"]:
            # 有界主表允许子集，完整守恒在 normalize_dataset 中核对。
            report["train_key_subset_only"] = True
        reports[name] = report
    return reports


def write_json(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalize_dataset(source, config, run_id, *, row_limit=None, progress=None):
    """不可变整批发布：三表及账本全部通过后才提升目录，失败不动旧成功目录。"""
    if row_limit is not None and (type(row_limit) is not int or not 1 <= row_limit <= 100000):
        raise DataError("normalization_limit_invalid", "Prefix limit must be 1..100000, or omit for full tables")
    if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise DataError("normalization_run_id_invalid", "Unsafe dataset identifier")
    output = config.artifact_dir.resolve()
    if output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise DataError("unsafe_output_path", "Output overlaps raw input")
    destination = output / ("full" if row_limit is None else "prefix") / run_id
    if destination.exists():
        raise DataError("dataset_exists", "Immutable dataset already exists; use a new run ID")
    staging = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        refs, references = reference_ids(source)
        # 配置要求完整表头，防止用投影样本冒充完整规范数据层。
        for name, columns in config.tables.items():
            with source.open_table(name) as (header, _):
                if header != [c["name"] for c in columns]:
                    raise DataError("normalization_header_mismatch", "Contract must match every source column in order", table=name)
        results, paths, ledger = {}, {}, {}
        for name, columns in config.tables.items():
            started = time.monotonic()
            path = staging / (Path(name).stem + ".parquet")
            info = write_verified_parquet(source, name, path,
                [{k: c[k] for k in ("name", "type")} for c in columns],
                output_root=output, row_limit=row_limit, batch_size=config.batch_size, columnar=True)
            profiles = profile_table(path, columns)
            if any(profiles[c]["nulls"] != info["quality"]["missing_counts"][c] for c in profiles):
                raise DataError("normalization_null_mismatch", "Null count changed beyond empty-to-null policy", table=name)
            info.update(output_path=str(destination / path.name), profile=profiles, passed=True,
                        elapsed_seconds=round(time.monotonic() - started, 3), **process_peak_memory())
            results[name], paths[name] = info, path
            ledger[name] = {"input_rows": info["quality"]["rows"], "output_rows": info["rows"],
                            "rows_removed": 0, "rows_added": 0, "rows_deduplicated": 0,
                            "row_order_preserved": True, "target_preserved": True,
                            "empty_cells_to_null_by_column": info["null_counts"],
                            "other_values_replaced": 0, "learned_transformations": [],
                            "numeric_representation": "int64 identifiers/label; other numeric fields float64; not decimal accounting",
                            "special_values": "candidate numeric codes and unknown categories counted, all preserved"}
            if progress:
                progress(name, info)
        relations = check_relations(paths, refs)
        if row_limit is None and (results["application_train.csv"]["rows"] != len(refs["train"])
                                 or relations["application_train.csv"].get("train_key_subset_only")):
            raise DataError("application_population_changed", "Full labeled sample population changed")
        for name, info in [*references.items(), *((n, i["quality"]) for n, i in results.items())]:
            if sha256_file(source.path(name)) != info["sha256"]:
                raise DataError("source_changed", "Input changed before dataset publication", table=name)
        quality = {"passed": True, "scope": "full_core_three_tables" if row_limit is None else "prefix_only",
                   "row_limit": row_limit, "references": references, "relations": relations,
                   "tables": results, "G2_core_normalization_checks_passed": row_limit is None,
                   "project_G2_join_checks": "pending E04 aggregation and join preservation",
                   "unused_history_child_tables_checked": False}
        write_json(staging / "quality.json", quality)
        write_json(staging / "transformations.json", ledger)
        write_json(staging / "contract.json", config.values)
        artifacts = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                     for p in staging.iterdir() if p.is_file()}
        bundle = {"schema_version": 1, "module": "E03", "status": "complete", "created_at": now(),
                  "run_id": run_id, "scope": quality["scope"], "row_limit": row_limit,
                  "contract_sha256": config.sha256, "files": artifacts,
                  "inputs": {n: i["sha256"] for n, i in references.items()} |
                            {n: i["quality"]["sha256"] for n, i in results.items()},
                  "limits": ["No feature fitting, joins or experiment freeze; G3-G6 unchanged.",
                             "Relative times and special codes preserved; feature eligibility requires E04/E05 review.",
                             "No reliable natural-person ID, absolute application date or undisclosed target X/Y inferred."]}
        write_json(staging / "bundle.json", bundle)
        staging.replace(destination)
        return {"artifact_dir": str(destination), "bundle_sha256": sha256_file(destination / "bundle.json"),
                "tables": results, "relations": relations, "scope": quality["scope"],
                "G2_core_normalization_checks_passed": row_limit is None}
    finally:
        if staging.exists():
            # 仅删除本函数刚创建且验证过的唯一临时目录。
            if not staging.resolve().is_relative_to(output) or not staging.name.endswith(".partial"):
                raise DataError("unsafe_cleanup", "Staging cleanup path escaped output root")
            shutil.rmtree(staging)


def load_normalized_bundle(root, report_path, *, require_full=True, require_official=True):
    """E04 下游入口：只接受成功运行、完整范围及指纹一致的规范产物。"""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "passed" or report["metadata"]["module"] != "E03":
            raise ValueError("not a successful E03 run")
        if require_official and report["scope"] != "official_full_normalization":
            raise ValueError("official full normalization required")
        result = report["normalization"]
        folder = Path(result["artifact_dir"]).resolve()
        if not folder.is_relative_to(root / "data/interim") or folder.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe bundle path")
        if sha256_file(folder / "bundle.json") != result["bundle_sha256"]:
            raise ValueError("bundle fingerprint mismatch")
        bundle = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        expected_files = {Path(n).stem + ".parquet" for n in CORE_TABLES} | {"quality.json", "transformations.json", "contract.json"}
        if (bundle["status"] != "complete" or bundle["module"] != "E03" or bundle["run_id"] != report["run_id"]
                or set(bundle["files"]) != expected_files or bundle["scope"] != result["scope"]
                or bundle["contract_sha256"] != report["metadata"]["normalization_config_sha256"]):
            raise ValueError("incomplete or inconsistent bundle")
        if require_full and (bundle["scope"] != "full_core_three_tables" or bundle["row_limit"] is not None):
            raise ValueError("prefix is not a full normalized dataset")
        for name, info in bundle["files"].items():
            p = (folder / name).resolve()
            if not p.is_relative_to(folder) or p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
                raise ValueError("artifact fingerprint mismatch")
        return {"directory": folder, "bundle": bundle,
                "tables": {n: folder / (Path(n).stem + ".parquet") for n in CORE_TABLES}}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError("normalized_bundle_invalid", "Failed, incomplete, bounded, unsafe or changed normalization bundle") from exc
