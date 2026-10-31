"""E04：时点约束下分别聚合历史，再左关联申请；审计信息与候选特征分开。"""

from dataclasses import dataclass
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import load_normalized_bundle, quoted, write_json
from .raw import DataError, require_data_scope, sha256_file
from .runtime import now, process_peak_memory

POLICY = "history_before_day_zero_currency_separate_v1"
SQL_FILES = ("sql/02_bureau_features.sql", "sql/03_previous_features.sql", "sql/04_join_history.sql")
BUREAU_REASONS = ("missing_credit_time", "credit_not_before", "missing_update_time", "update_not_before")
PREV_REASONS = ("missing_decision_time", "decision_not_before")
REQUIRED = {
    "application_train.csv": {"SK_ID_CURR", "TARGET"},
    "bureau.csv": {"SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT", "DAYS_CREDIT_UPDATE", "CREDIT_CURRENCY", "CREDIT_ACTIVE", "AMT_CREDIT_SUM"},
    "previous_application.csv": {"SK_ID_CURR", "SK_ID_PREV", "DAYS_DECISION", "NAME_CONTRACT_TYPE"},
}

# 单一输出字段契约同时驱动 join 的结构性0和下游允许清单；公式实际执行在独立 SQL 文件。
FEATURES = [
    ("BUREAU_ELIGIBLE_COUNT", "int64", "count of eligible bureau records", "records"),
    ("BUREAU_HAS_ELIGIBLE_HISTORY", "int64", "1 if eligible bureau records exist else 0", "binary"),
    ("BUREAU_DAYS_SINCE_MOST_RECENT", "float64", "-max(DAYS_CREDIT)", "days"),
    ("BUREAU_C1_COUNT", "int64", "count(CREDIT_CURRENCY = currency 1)", "records"),
    ("BUREAU_OTHER_CURRENCY_COUNT", "int64", "count(nonnull CREDIT_CURRENCY <> currency 1)", "records"),
    ("BUREAU_CURRENCY_MISSING_COUNT", "int64", "count(null CREDIT_CURRENCY)", "records"),
    ("BUREAU_C1_AMOUNT_AVAILABLE_COUNT", "int64", "count(currency 1 and AMT_CREDIT_SUM >= 0)", "records"),
    ("BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT", "int64", "count(currency 1 and amount null or negative)", "records"),
    ("BUREAU_C1_CREDIT_SUM", "float64", "sum(AMT_CREDIT_SUM) over currency 1 and nonnegative amounts", "anonymous_currency_1_source_scale"),
    ("BUREAU_C1_CREDIT_MEAN", "float64", "avg(AMT_CREDIT_SUM) over currency 1 and nonnegative amounts", "anonymous_currency_1_source_scale"),
    ("BUREAU_C1_CREDIT_MAX", "float64", "max(AMT_CREDIT_SUM) over currency 1 and nonnegative amounts", "anonymous_currency_1_source_scale"),
    ("BUREAU_ACTIVE_COUNT", "int64", "count(CREDIT_ACTIVE = Active)", "records"),
    ("BUREAU_CLOSED_COUNT", "int64", "count(CREDIT_ACTIVE = Closed)", "records"),
    ("BUREAU_SOLD_COUNT", "int64", "count(CREDIT_ACTIVE = Sold)", "records"),
    ("BUREAU_BAD_DEBT_COUNT", "int64", "count(CREDIT_ACTIVE = Bad debt)", "records"),
    ("BUREAU_OTHER_STATUS_COUNT", "int64", "count(other or missing CREDIT_ACTIVE)", "records"),
    ("BUREAU_ACTIVE_FRACTION", "float64", "BUREAU_ACTIVE_COUNT / BUREAU_ELIGIBLE_COUNT", "fraction"),
    ("PREV_ELIGIBLE_COUNT", "int64", "count of previous applications with DAYS_DECISION < 0", "records"),
    ("PREV_HAS_ELIGIBLE_HISTORY", "int64", "1 if eligible previous applications exist else 0", "binary"),
    ("PREV_DAYS_SINCE_MOST_RECENT", "float64", "-max(DAYS_DECISION)", "days"),
    ("PREV_CASH_COUNT", "int64", "count(NAME_CONTRACT_TYPE = Cash loans)", "records"),
    ("PREV_CONSUMER_COUNT", "int64", "count(NAME_CONTRACT_TYPE = Consumer loans)", "records"),
    ("PREV_REVOLVING_COUNT", "int64", "count(NAME_CONTRACT_TYPE = Revolving loans)", "records"),
    ("PREV_OTHER_CONTRACT_COUNT", "int64", "count(other or missing NAME_CONTRACT_TYPE)", "records"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
COUNT_NAMES = [name for name, kind, *_ in FEATURES if kind == "int64"]


@dataclass(frozen=True)
class HistoryConfig:
    root: Path
    artifact_dir: Path
    path: Path
    sha256: str
    values: dict
    sql: dict


def load_history_config(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        v = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(v) != {"schema_version", "policy_version", "artifact_dir", "duckdb_memory_limit", "duckdb_threads",
                      "row_group_size", "case_ids", "manual_absolute_tolerance", "manual_relative_tolerance", "sql_files"}:
            raise ValueError("unexpected config fields")
        if type(v["schema_version"]) is not int or v["schema_version"] != 1 or v["policy_version"] != POLICY:
            raise ValueError("unsupported version")
        if v["duckdb_memory_limit"] != "512MB" or type(v["duckdb_threads"]) is not int or v["duckdb_threads"] != 1:
            raise ValueError("fixed resource budget required")
        if type(v["row_group_size"]) is not int or not 2048 <= v["row_group_size"] <= 65536 or v["row_group_size"] % 2048:
            raise ValueError("invalid row group size")
        if (not isinstance(v["case_ids"], list) or not 1 <= len(v["case_ids"]) <= 10
                or any(type(k) is not int or not 0 < k < 2**63 for k in v["case_ids"])
                or len(set(v["case_ids"])) != len(v["case_ids"])):
            raise ValueError("1..10 unique positive case IDs required")
        if v["manual_absolute_tolerance"] != 1e-9 or v["manual_relative_tolerance"] != 1e-12:
            raise ValueError("fixed manual comparison tolerances required")
        relative = v["artifact_dir"]
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("relative artifact path required")
        output = (root / relative).resolve()
        if not output.is_relative_to(root / "data/interim") or output.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe output root")
        if v["sql_files"] != list(SQL_FILES):
            raise ValueError("registered SQL files required")
        sql = {p: {"text": (root / p).read_text(encoding="utf-8"), "sha256": sha256_file(root / p)} for p in SQL_FILES}
        return HistoryConfig(root, output, path, sha256_file(path), v, sql)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise DataError("history_config_invalid", "Invalid history policy, configuration, resources or SQL paths") from exc


def feature_contract():
    return {"schema_version": 1, "policy_version": POLICY, "purpose": "deterministic_history_candidates_not_frozen",
            "join_key": "SK_ID_CURR", "label": "TARGET", "feature_columns": FEATURE_NAMES,
            "features": [{"name": name, "type": kind, "formula": formula, "unit": unit,
                "source": "bureau.csv" if name.startswith("BUREAU_") else "previous_application.csv",
                "time_rule": "DAYS_CREDIT < 0 AND DAYS_CREDIT_UPDATE < 0" if name.startswith("BUREAU_") else "DAYS_DECISION < 0",
                "no_eligible_history": "structural_zero" if kind == "int64" else "null",
                "all_amounts_unavailable": "amount sum/mean/max remain null, available count=0; no learned imputation",
                "fit_scope": "none; fixed SQL, no label, population statistics or learned preprocessing"}
                for name, kind, formula, unit in FEATURES],
            "audit_table_role": "explanation_only_never_model_input; can change when future records are added",
            "main_table_feature_status": "all original fields preserved; main feature eligibility pending E05",
            "held_fields": {"previous_application.csv": ["AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_STATUS",
                "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"],
                "bureau.csv": ["AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "DAYS_CREDIT_ENDDATE"]},
            "limits": ["Day zero is excluded because within-day order is unavailable.",
                       "Currency 1 is a recoded unit, not a real currency or an exchange-rate base.",
                       "Previous amounts lack a currency field; final-status update timing is not established; held for later review.",
                       "Negative credit amounts are unavailable only to credit-amount formulas, not removed from source or record counts.",
                       "Snapshot availability follows the supplied dictionary, not an independently reconstructed production as-of database."]}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def join_query(config):
    columns = []
    for name, kind, *_ in FEATURES:
        ref = ("b." if name.startswith("BUREAU_") else "p.") + quoted(name)
        expression = f"COALESCE({ref}, 0::BIGINT)" if kind == "int64" else ref
        columns.append(expression + " AS " + quoted(name))
    template = config.sql[SQL_FILES[2]]["text"]
    if template.count("{{history_columns}}") != 2:  # 注释与真正查询中各一处。
        raise DataError("history_sql_invalid", "Join SQL placeholder is missing or ambiguous")
    return template.replace("{{history_columns}}", ", ".join(columns)).rstrip().rstrip(";")


def create_audit(con):
    aggregates, projection = [], []
    for alias, table, reasons in (("b", "bureau", BUREAU_REASONS), ("p", "previous", PREV_REASONS)):
        metrics = [f"COUNT(*)::BIGINT AS {table}_raw_count", f"COUNT(*) FILTER (WHERE exclusion_reason='eligible') AS {table}_eligible_count"]
        for reason in reasons:
            metrics.append(f"COUNT(*) FILTER (WHERE exclusion_reason='{reason}') AS {table}_{reason}_count")
        names = [f"{table}_raw_count", f"{table}_eligible_count", *[f"{table}_{r}_count" for r in reasons]]
        aggregates.append(f"{alias} AS (SELECT SK_ID_CURR, {','.join(metrics)} FROM {table}_classified GROUP BY SK_ID_CURR)")
        projection.extend(f"COALESCE({alias}.{n},0::BIGINT) AS {n}" for n in names)
        projection.append(f"CASE WHEN {alias}.{table}_eligible_count>0 THEN 'eligible_present' "
                          f"WHEN {alias}.{table}_raw_count>0 THEN 'only_time_excluded' ELSE 'no_source_history' END AS {table}_history_state")
    con.execute("CREATE TABLE history_audit AS WITH keys AS (SELECT SK_ID_CURR FROM application_input UNION "
                "SELECT SK_ID_CURR FROM bureau_input UNION SELECT SK_ID_CURR FROM previous_input), " +
                ",".join(aggregates) + " SELECT k.SK_ID_CURR," + ",".join(projection) +
                " FROM keys k LEFT JOIN b USING(SK_ID_CURR) LEFT JOIN p USING(SK_ID_CURR)")


def verify_unique(con, relation):
    rows, distinct, nonnull = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR), COUNT(SK_ID_CURR) FROM {relation}").fetchone()
    if rows != distinct or rows != nonnull:
        raise DataError("history_duplicate_key", "Summary or joined key is missing or duplicated")
    return rows


def copy_verified(con, query, path, row_group_size):
    con.execute(f"COPY ({query}) TO {literal(path)} (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {row_group_size})")
    # SQL 输出流与实际 Parquet 逐批比较；不以文件存在冒充验证通过。
    with con.sql(query).to_arrow_reader(8192) as expected, pq.ParquetFile(path) as stored:
        # DuckDB COPY 会将可证明非空的列编码为 Parquet required；Arrow 查询输出仍标 nullable。
        # 比较真实列名／顺序／类型及下面的每个值和 null，允许这个更强的可空性元数据。
        if [(f.name, f.type) for f in expected.schema] != [(f.name, f.type) for f in stored.schema_arrow]:
            raise DataError("history_parquet_schema_mismatch", "Stored SQL schema differs")
        rows = 0
        for left, right in zip_longest(expected, stored.iter_batches(batch_size=8192)):
            if left is None or right is None or left.num_rows != right.num_rows or any(
                    not a.equals(b) for a,b in zip(left.columns,right.columns)):
                raise DataError("history_parquet_values_mismatch", "Stored SQL values, count or order differ")
            rows += right.num_rows
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": sha256_file(path), "sql_parquet_exact_match": True}


def verify_main_preservation(original, joined, names, row_limit):
    remaining, rows = row_limit, 0
    with pq.ParquetFile(original) as source, pq.ParquetFile(joined) as output:
        actual = iter(output.iter_batches(batch_size=8192, columns=names))
        for batch in source.iter_batches(batch_size=8192):
            if remaining is not None:
                if remaining == 0:
                    break
                batch = batch.slice(0, min(remaining, batch.num_rows))
                remaining -= batch.num_rows
            got = next(actual, None)
            if (got is None or batch.num_rows != got.num_rows or batch.schema.names != got.schema.names
                    or any(not a.equals(b) for a,b in zip(batch.columns,got.columns))):
                raise DataError("history_main_changed", "Original main columns, label, nulls or row order changed")
            rows += batch.num_rows
        if next(actual, None) is not None:
            raise DataError("history_join_expansion", "Joined table contains extra rows")
    return rows


def manual_features(bureau, previous):
    """小型原始记录的独立 Python 算式，逐项对照 SQL；只用于已指定案例。"""
    def number(value):
        return None if value == "" else float(value)
    b = [r for r in bureau if number(r["DAYS_CREDIT"]) is not None and number(r["DAYS_CREDIT"]) < 0
         and number(r["DAYS_CREDIT_UPDATE"]) is not None and number(r["DAYS_CREDIT_UPDATE"]) < 0]
    p = [r for r in previous if number(r["DAYS_DECISION"]) is not None and number(r["DAYS_DECISION"]) < 0]
    result = {n: 0 if kind == "int64" else None for n, kind, *_ in FEATURES}
    if b:
        primary = [r for r in b if r["CREDIT_CURRENCY"] == "currency 1"]
        amounts = [number(r["AMT_CREDIT_SUM"]) for r in primary if number(r["AMT_CREDIT_SUM"]) is not None and number(r["AMT_CREDIT_SUM"]) >= 0]
        result.update(BUREAU_ELIGIBLE_COUNT=len(b), BUREAU_HAS_ELIGIBLE_HISTORY=1,
            BUREAU_DAYS_SINCE_MOST_RECENT=-max(number(r["DAYS_CREDIT"]) for r in b),
            BUREAU_C1_COUNT=len(primary), BUREAU_OTHER_CURRENCY_COUNT=sum(r["CREDIT_CURRENCY"] not in ("", "currency 1") for r in b),
            BUREAU_CURRENCY_MISSING_COUNT=sum(r["CREDIT_CURRENCY"] == "" for r in b),
            BUREAU_C1_AMOUNT_AVAILABLE_COUNT=len(amounts), BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT=len(primary)-len(amounts),
            BUREAU_C1_CREDIT_SUM=math.fsum(amounts) if amounts else None,
            BUREAU_C1_CREDIT_MEAN=math.fsum(amounts)/len(amounts) if amounts else None,
            BUREAU_C1_CREDIT_MAX=max(amounts) if amounts else None)
        for text, name in (("Active", "BUREAU_ACTIVE_COUNT"), ("Closed", "BUREAU_CLOSED_COUNT"),
                           ("Sold", "BUREAU_SOLD_COUNT"), ("Bad debt", "BUREAU_BAD_DEBT_COUNT")):
            result[name] = sum(r["CREDIT_ACTIVE"] == text for r in b)
        result["BUREAU_OTHER_STATUS_COUNT"] = sum(r["CREDIT_ACTIVE"] not in ("Active", "Closed", "Sold", "Bad debt") for r in b)
        result["BUREAU_ACTIVE_FRACTION"] = result["BUREAU_ACTIVE_COUNT"] / len(b)
    if p:
        result.update(PREV_ELIGIBLE_COUNT=len(p), PREV_HAS_ELIGIBLE_HISTORY=1,
                      PREV_DAYS_SINCE_MOST_RECENT=-max(number(r["DAYS_DECISION"]) for r in p))
        for text, name in (("Cash loans", "PREV_CASH_COUNT"), ("Consumer loans", "PREV_CONSUMER_COUNT"), ("Revolving loans", "PREV_REVOLVING_COUNT")):
            result[name] = sum(r["NAME_CONTRACT_TYPE"] == text for r in p)
        result["PREV_OTHER_CONTRACT_COUNT"] = sum(r["NAME_CONTRACT_TYPE"] not in ("Cash loans", "Consumer loans", "Revolving loans") for r in p)
    return result


def case_trace(source, con, ids, config):
    records = {k: {"bureau.csv": [], "previous_application.csv": []} for k in ids}
    for name in ("bureau.csv", "previous_application.csv"):
        with source.open_table(name, sorted(REQUIRED[name])) as (_, rows):
            for row in rows:
                key = int(row["SK_ID_CURR"])
                if key in records:
                    records[key][name].append(row)
                    if sum(len(v[n]) for v in records.values() for n in v) > 20000:
                        raise DataError("history_case_too_large", "Manual case trace exceeds 20000 records")
    output = {}
    for key, rows in records.items():
        actual_row = con.execute("SELECT " + ",".join(quoted(n) for n in FEATURE_NAMES) + " FROM joined_features WHERE SK_ID_CURR=?", [key]).fetchone()
        if actual_row is None:
            raise DataError("history_case_missing", "Manual case is absent from selected application scope")
        expected = manual_features(rows["bureau.csv"], rows["previous_application.csv"])
        actual = dict(zip(FEATURE_NAMES, actual_row))
        for name, wanted in expected.items():
            got = actual[name]
            equal = got is None if wanted is None else (got == wanted if name in COUNT_NAMES else got is not None and
                math.isclose(got, wanted, abs_tol=config.values["manual_absolute_tolerance"], rel_tol=config.values["manual_relative_tolerance"]))
            if not equal:
                raise DataError("history_case_mismatch", f"Manual arithmetic differs in {name}")
        output[str(key)] = {"raw_records": rows, "raw_direct_join_rows": len(rows["bureau.csv"])*len(rows["previous_application.csv"]),
                            "python_manual_expected": expected, "sql_actual": actual, "all_24_features_match": True}
    return output


def build_history_features(config, report_path, source, run_id, *, fixture=False, application_limit=None, progress=None):
    if application_limit is not None and (type(application_limit) is not int or not 1 <= application_limit <= 100000):
        raise DataError("history_limit_invalid", "Application prefix must be 1..100000, or omit for full cohort")
    if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise DataError("history_run_id_invalid", "Unsafe run identifier")
    report_path = Path(report_path).resolve()
    require_data_scope(source, config.root / "data/raw", fixture=fixture)
    loaded = load_normalized_bundle(config.root, report_path, require_official=not fixture)
    upstream = json.loads(report_path.read_text(encoding="utf-8"))
    if upstream["scope"].startswith("synthetic_") != fixture:
        raise DataError("history_scope_mismatch", "Fixture flag and upstream scope disagree")
    output = config.artifact_dir.resolve()
    for input_root in (source.root, loaded["directory"]):
        if output.is_relative_to(input_root) or input_root.is_relative_to(output):
            raise DataError("unsafe_output_path", "History output overlaps input")
    destination = output / ("full" if application_limit is None else "prefix") / run_id
    if destination.exists():
        raise DataError("history_dataset_exists", "History output is immutable; use a new run ID")
    staging = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        def check_inputs():
            for name, digest in loaded["bundle"]["inputs"].items():
                if sha256_file(source.path(name)) != digest:
                    raise DataError("history_raw_changed", "Raw case inputs differ from normalized bundle", table=name)
        check_inputs()
        main_schema = pq.ParquetFile(loaded["tables"]["application_train.csv"]).schema_arrow
        if set(main_schema.names) & (set(FEATURE_NAMES) | {"file_row_number"}):
            raise DataError("history_column_collision", "Main input contains reserved history or order fields")
        for name, required in REQUIRED.items():
            schema = pq.ParquetFile(loaded["tables"][name]).schema_arrow
            if not required.issubset(schema.names):
                raise DataError("history_columns_missing", "Required history fields missing", table=name)
        with duckdb.connect(config={"threads": "1", "memory_limit": "512MB", "temp_directory": str(staging / "duckdb_spill")}) as con:
            limit = "" if application_limit is None else f" WHERE file_row_number < {application_limit}"
            con.execute("CREATE TEMP VIEW application_input AS SELECT * FROM read_parquet(" +
                        literal(loaded["tables"]["application_train.csv"]) + ", file_row_number=true)" + limit)
            for name, alias in (("bureau.csv", "bureau"), ("previous_application.csv", "previous")):
                restriction = "" if application_limit is None else " WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_input)"
                con.execute(f"CREATE TEMP VIEW {alias}_input AS SELECT * FROM read_parquet({literal(loaded['tables'][name])}){restriction}")
            main_rows = verify_unique(con, "application_input")
            if not main_rows:
                raise DataError("history_empty_main", "No applications selected")
            for path in SQL_FILES[:2]:
                con.execute(config.sql[path]["text"])
            for table in ("bureau_features", "previous_features"):
                verify_unique(con, table)
            create_audit(con)
            joined = join_query(config)
            con.execute("CREATE TABLE joined_features AS " + joined)
            joined_rows = verify_unique(con, "joined_features")
            if joined_rows != main_rows:
                raise DataError("history_join_expansion", "Application row count changed")
            difference = con.execute("SELECT COUNT(*) FROM ((SELECT SK_ID_CURR FROM application_input EXCEPT ALL SELECT SK_ID_CURR FROM joined_features) "
                                     "UNION ALL (SELECT SK_ID_CURR FROM joined_features EXCEPT ALL SELECT SK_ID_CURR FROM application_input))").fetchone()[0]
            if difference:
                raise DataError("history_key_set_changed", "Application ID population changed")
            stats = {}
            for alias in ("bureau", "previous"):
                counts = dict(con.execute(f"SELECT exclusion_reason,COUNT(*) FROM {alias}_classified GROUP BY 1 ORDER BY 1").fetchall())
                raw_rows = con.execute(f"SELECT COUNT(*) FROM {alias}_input").fetchone()[0]
                eligible_rows = con.execute(f"SELECT COUNT(*) FROM {alias}_eligible").fetchone()[0]
                if sum(counts.values()) != raw_rows or counts.get("eligible", 0) != eligible_rows:
                    raise DataError("history_audit_not_conserved", "Time-disposition counts do not conserve source records")
                stats[alias] = {"raw_rows_in_scope": raw_rows, "eligible_rows": eligible_rows,
                                "excluded_rows": raw_rows-eligible_rows, "mutually_exclusive_dispositions": counts}
            # 核对特征类型、非负计数、有限值和类别分母恒等式。
            for name, kind, *_ in FEATURES:
                dtype = con.execute(f"SELECT typeof({quoted(name)}) FROM joined_features LIMIT 1").fetchone()[0]
                if dtype != ("BIGINT" if kind == "int64" else "DOUBLE"):
                    raise DataError("history_feature_type", f"Unexpected type for {name}")
                invalid = f"{quoted(name)}<0 OR {quoted(name)} IS NULL" if kind == "int64" else f"{quoted(name)} IS NOT NULL AND NOT isfinite({quoted(name)})"
                if con.execute(f"SELECT COUNT(*) FROM joined_features WHERE {invalid}").fetchone()[0]:
                    raise DataError("history_feature_invalid", f"Invalid value in {name}")
            checks = ["BUREAU_ACTIVE_COUNT+BUREAU_CLOSED_COUNT+BUREAU_SOLD_COUNT+BUREAU_BAD_DEBT_COUNT+BUREAU_OTHER_STATUS_COUNT <> BUREAU_ELIGIBLE_COUNT",
                      "BUREAU_C1_COUNT+BUREAU_OTHER_CURRENCY_COUNT+BUREAU_CURRENCY_MISSING_COUNT <> BUREAU_ELIGIBLE_COUNT",
                      "BUREAU_C1_AMOUNT_AVAILABLE_COUNT+BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT <> BUREAU_C1_COUNT",
                      "PREV_CASH_COUNT+PREV_CONSUMER_COUNT+PREV_REVOLVING_COUNT+PREV_OTHER_CONTRACT_COUNT <> PREV_ELIGIBLE_COUNT"]
            if con.execute("SELECT COUNT(*) FROM joined_features WHERE " + " OR ".join(checks)).fetchone()[0]:
                raise DataError("history_feature_count_mismatch", "Category partitions do not sum to eligible record counts")
            artifacts = {}
            for table in ("bureau_features", "previous_features", "history_audit", "joined_features"):
                query = joined if table == "joined_features" else f"SELECT * FROM {table} ORDER BY SK_ID_CURR"
                started = time.monotonic()
                artifacts[table + ".parquet"] = copy_verified(con, query, staging / (table + ".parquet"), config.values["row_group_size"])
                if progress:
                    progress(table, {**artifacts[table + ".parquet"], "elapsed_seconds": round(time.monotonic()-started, 3)})
            preserved_rows = verify_main_preservation(loaded["tables"]["application_train.csv"], staging / "joined_features.parquet", main_schema.names, application_limit)
            if preserved_rows != main_rows:
                raise DataError("history_main_changed", "Original main columns were not fully preserved")
            traces = case_trace(source, con, config.values["case_ids"], config)
            no_history = con.execute("SELECT COUNT(*) FILTER(WHERE BUREAU_HAS_ELIGIBLE_HISTORY=0), COUNT(*) FILTER(WHERE PREV_HAS_ELIGIBLE_HISTORY=0) FROM joined_features").fetchone()
            states = {alias: dict(con.execute(f"SELECT {alias}_history_state,COUNT(*) FROM history_audit WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_input) GROUP BY 1").fetchall()) for alias in ("bureau", "previous")}
        check_inputs()
        load_normalized_bundle(config.root, report_path, require_official=not fixture)
        if sha256_file(config.path) != config.sha256 or any(sha256_file(config.root/p) != v["sha256"] for p,v in config.sql.items()):
            raise DataError("history_recipe_changed", "Configuration or SQL changed during execution")
        quality = {"passed": True, "scope": "full_history_and_labeled_applications" if application_limit is None else "first_labeled_applications_and_all_their_history",
                   "application_limit": application_limit, "input_application_rows": main_rows, "output_application_rows": joined_rows,
                   "id_multiset_difference": difference, "all_original_main_values_nulls_target_order_preserved": True,
                   "summary_primary_keys_unique": True, "time_audit": stats,
                   "applications_without_eligible_history": dict(zip(("bureau", "previous"), no_history)),
                   "labeled_application_history_states": states, "manual_cases": list(traces),
                   "history_feature_count": len(FEATURES), "G2_scope": "first three tables; unused historical child tables excluded",
                   "G3_status": "not_frozen; E05 main feature selection and split pending"}
        write_json(staging / "quality.json", quality)
        write_json(staging / "case_trace.json", traces)
        write_json(staging / "feature_contract.json", feature_contract())
        write_json(staging / "recipe.json", config.values)
        for path, info in config.sql.items():
            (staging / Path(path).name).write_text(info["text"], encoding="utf-8")
        spill = staging / "duckdb_spill"
        if spill.exists():
            assert spill.resolve().is_relative_to(staging.resolve())
            shutil.rmtree(spill)
        files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in staging.iterdir() if p.is_file()}
        bundle = {"schema_version": 1, "module": "E04", "status": "complete", "run_id": run_id, "created_at": now(),
                  "policy_version": POLICY, "scope": quality["scope"], "application_limit": application_limit,
                  "input_run_path": str(report_path), "input_run_sha256": sha256_file(report_path),
                  "input_bundle_sha256": upstream["normalization"]["bundle_sha256"],
                  "config_sha256": config.sha256, "sql_sha256": {p:v["sha256"] for p,v in config.sql.items()}, "files": files}
        write_json(staging / "bundle.json", bundle)
        staging.replace(destination)
        return {"artifact_dir": str(destination), "bundle_sha256": sha256_file(destination / "bundle.json"),
                "scope": quality["scope"], "quality": quality, "files": files}
    finally:
        if staging.exists():
            if not staging.resolve().is_relative_to(output) or not staging.name.endswith(".partial"):
                raise DataError("unsafe_cleanup", "History staging cleanup escaped output root")
            shutil.rmtree(staging)


def load_history_bundle(root, report_path, *, require_official=True):
    """E05 只接入成功全量 E04；返回明确的历史候选列，不将审计字段或标签当特征。"""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["history_features"]
        if report["status"] != "passed" or report["metadata"]["module"] != "E04":
            raise ValueError("successful E04 run required")
        if require_official and report["scope"] != "official_full_history_features":
            raise ValueError("official full run required")
        if report["scope"] not in ("official_full_history_features", "synthetic_full_history_features"):
            raise ValueError("full run scope required")
        folder = Path(result["artifact_dir"]).resolve()
        if not folder.is_relative_to(root / "data/interim") or folder.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe artifact path")
        if sha256_file(folder / "bundle.json") != result["bundle_sha256"]:
            raise ValueError("bundle fingerprint differs")
        bundle = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        expected = {"bureau_features.parquet", "previous_features.parquet", "history_audit.parquet", "joined_features.parquet",
                    "quality.json", "case_trace.json", "feature_contract.json", "recipe.json", *[Path(p).name for p in SQL_FILES]}
        if (bundle["status"] != "complete" or bundle["module"] != "E04" or bundle["policy_version"] != POLICY
                or bundle["run_id"] != report["run_id"] or set(bundle["files"]) != expected
                or bundle["scope"] != "full_history_and_labeled_applications" or bundle["application_limit"] is not None
                or bundle["scope"] != result["scope"]
                or bundle["config_sha256"] != report["metadata"]["history_config_sha256"]):
            raise ValueError("incomplete, bounded or incompatible bundle")
        for name, info in bundle["files"].items():
            p = (folder / name).resolve()
            if not p.is_relative_to(folder) or p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
                raise ValueError("output file changed")
        contract = json.loads((folder / "feature_contract.json").read_text(encoding="utf-8"))
        if contract != feature_contract():
            raise ValueError("candidate contract differs")
        upstream_path = Path(bundle["input_run_path"])
        if sha256_file(upstream_path) != bundle["input_run_sha256"]:
            raise ValueError("upstream report changed")
        upstream = load_normalized_bundle(root, upstream_path, require_official=require_official)
        if sha256_file(upstream["directory"] / "bundle.json") != bundle["input_bundle_sha256"]:
            raise ValueError("upstream bundle changed")
        return {"directory": folder, "joined_features": folder / "joined_features.parquet", "history_feature_columns": FEATURE_NAMES.copy(),
                "contract": contract, "bundle": bundle}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise DataError("history_bundle_invalid", "Failed, bounded, incomplete, unsafe, incompatible or changed E04 output") from exc
