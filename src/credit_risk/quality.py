"""完整消费单表后返回基础质量报告；不清洗、不训练、不写入原文件。"""

from collections import Counter
from dataclasses import dataclass
from itertools import islice

from .raw import DataError, RawDataSource, TABLES, sha256_file


@dataclass
class TableInspection:
    report: dict
    keys: set[str]


def inspect_table(source: RawDataSource, name: str, *, collect_keys=False,
                  reference_ids=None, hash_file=False, row_limit=None, columns=None) -> TableInspection:
    """主键去重需 O(不同键数) 内存，其余逐行统计；空值严格指空字符串。

    row_limit 仅检查固定前缀，parse_complete 表示所选范围解析完成；不宣称全表质量通过。
    有 manifest 时比对文件指纹／大小／列数；仅全表扫描时比较总行数。
    reference_ids 仅检查所选表的 SK_ID_CURR 是否在明确给定的集合中。
    """
    if row_limit is not None and (type(row_limit) is not int or row_limit <= 0):
        raise DataError("row_limit_invalid", "row_limit must be a positive integer")
    spec = TABLES.get(name)
    info = {"rows": 0, "columns": 0, "header": [], "parse_complete": False,
            "empty_records_skipped": 0, "malformed_width_records": 0,
            "duplicate_column_names": None, "errors": [],
            "scan_scope": "full_table" if row_limit is None else "first_rows",
            "row_limit": row_limit, "full_file_quality_checked": row_limit is None}
    keys, missing, target = set(), Counter(), Counter()
    missing_keys = duplicate_keys = orphan_rows = invalid_target = 0
    def error(code, message, count=1):
        info["errors"].append({"code": code, "message": message, "count": count})

    try:
        selected = None if columns is None else tuple(dict.fromkeys((*TABLES[name].required, *columns)))
        with source.open_table(name, selected) as (header, records):
            info.update(header=header, columns=len(header), encoding=spec.encoding, duplicate_column_names=0)
            info["inspected_columns"] = list(selected) if selected is not None else header
            for row in islice(records, row_limit):
                info["rows"] += 1
                missing.update(column for column, value in row.items() if value == "")
                if spec.primary_key:
                    key = row[spec.primary_key]
                    if not key.strip():
                        missing_keys += 1
                    elif key in keys:
                        duplicate_keys += 1
                    else:
                        keys.add(key)
                if spec.target_role:
                    value = row["TARGET"]
                    target[value] += 1
                    if spec.target_role == "observed_label" and value not in {"0", "1"}:
                        invalid_target += 1
                if reference_ids is not None:
                    if "SK_ID_CURR" not in row:
                        raise DataError("reference_column_missing", "Reference check requires SK_ID_CURR", table=name)
                    if row["SK_ID_CURR"] not in reference_ids:
                        orphan_rows += 1
            info["parse_complete"] = True
    except DataError as exc:
        info["errors"].append(exc.as_dict())
        info["malformed_width_records"] = int(exc.code == "row_width")
    if spec and spec.primary_key:
        info.update(primary_key=spec.primary_key, primary_key_missing=missing_keys,
                    primary_key_duplicate_rows=duplicate_keys, primary_key_distinct=len(keys))
        if missing_keys:
            error("key_missing", "Primary key contains empty or whitespace-only values", missing_keys)
        if duplicate_keys:
            error("key_duplicate", "Primary key contains duplicate values", duplicate_keys)
    if spec and spec.target_role:
        info.update(target_counts=dict(target), target_role=spec.target_role)
        if invalid_target:
            error("target_invalid", "Observed TARGET must be exactly 0 or 1", invalid_target)
    if reference_ids is not None:
        info["rows_outside_train_test_application_ids"] = orphan_rows
        if orphan_rows:
            error("foreign_key_missing", "SK_ID_CURR is absent from supplied reference IDs", orphan_rows)
    info["missing_counts"] = {column: missing[column] for column in info.get("inspected_columns", info["header"])}
    if info["parse_complete"] and not info["rows"]:
        error("empty_dataset", "CSV has a header but no data rows")
    if source.manifest is not None or hash_file:
        try:
            path = source.path(name)
            info.update(bytes=path.stat().st_size, sha256=sha256_file(path))
            if source.manifest is not None:
                expected = source.entries.get(name)
                if expected is None:
                    error("manifest_entry_missing", "Table is absent from manifest")
                else:
                    fields = ("bytes", "sha256", "rows", "columns") if row_limit is None else ("bytes", "sha256", "columns")
                    mismatches = [field for field in fields
                                  if info.get(field) != expected[field]]
                    info["manifest_fields_compared"] = list(fields)
                    info["manifest_fingerprint_matches"] = not mismatches and info["parse_complete"]
                    info["manifest_matches"] = info["manifest_fingerprint_matches"] if row_limit is None else None
                    if mismatches:
                        error("manifest_mismatch", f"Manifest differs in: {', '.join(mismatches)}")
        except (DataError, OSError):
            error("fingerprint_unavailable", "Could not fingerprint the source file")
    info["passed"] = info["parse_complete"] and not info["errors"]
    return TableInspection(info, keys if collect_keys else set())
