"""有界、显式类型的 Parquet 往返；复用原始入口和质量口径，不做业务清洗。"""

from dataclasses import dataclass
import hashlib
from itertools import islice, zip_longest
import json
import math
from pathlib import Path
import re
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .quality import inspect_table
from .raw import DataError, TABLES, sha256_file

TYPES = {"int64": pa.int64(), "float64": pa.float64(), "string": pa.string()}


@dataclass(frozen=True)
class ParquetConfig:
    artifact_dir: Path
    row_limit: int
    batch_size: int
    tables: dict
    path: Path
    sha256: str


def load_parquet_config(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(values) != {"schema_version", "artifact_dir", "row_limit", "batch_size", "tables"}:
            raise ValueError("unexpected fields")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("unsupported schema")
        for key, maximum in (("row_limit", 100000), ("batch_size", 10000)):
            if type(values[key]) is not int or not 1 <= values[key] <= maximum:
                raise ValueError("smoke limits out of range")
        relative = values["artifact_dir"]
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("relative artifact path required")
        output = (root / relative).resolve()
        # 固定在中间层；即使 interim 本身被链接到 raw，也拒绝输出。
        if not output.is_relative_to(root / "data/interim") or output.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("artifact path must stay in data/interim")
        tables = values["tables"]
        if not isinstance(tables, dict) or not tables:
            raise ValueError("tables required")
        for name, columns in tables.items():
            if name not in TABLES:
                raise ValueError("unknown table")
            make_schema(columns)
        return ParquetConfig(output, values["row_limit"], values["batch_size"], tables, path, sha256_file(path))
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise DataError("parquet_config_invalid", "Parquet smoke config has invalid fields, types or paths") from exc


def make_schema(columns):
    if not isinstance(columns, list) or not columns:
        raise ValueError("nonempty column list required")
    names, fields = set(), []
    for column in columns:
        if not isinstance(column, dict) or set(column) != {"name", "type"}:
            raise ValueError("column requires name and type")
        name, kind = column["name"], column["type"]
        if not isinstance(name, str) or not name.strip() or name in names or kind not in TYPES:
            raise ValueError("invalid or duplicate column")
        names.add(name)
        fields.append(pa.field(name, TYPES[kind], nullable=True))
    return pa.schema(fields)


def typed_batches(source, name, schema, *, row_limit, batch_size):
    """空字符串映射 null，字符串原样保留；数值严格解析，不推断类别或填补。"""
    with source.open_table(name, schema.names) as (_, records):
        batch = []
        for index, row in enumerate(islice(records, row_limit), 1):
            converted = {}
            for field in schema:
                value = row[field.name]
                try:
                    if value == "":
                        result = None
                    elif pa.types.is_string(field.type):
                        result = value
                    elif pa.types.is_int64(field.type):
                        if not re.fullmatch(r"[+-]?[0-9]+", value):
                            raise ValueError("integer syntax")
                        result = int(value)
                        if not -(2 ** 63) <= result < 2 ** 63:
                            raise ValueError("integer overflow")
                    else:
                        if value != value.strip():
                            raise ValueError("numeric whitespace")
                        result = float(value)
                        if not math.isfinite(result):
                            raise ValueError("nonfinite number")
                except (ValueError, OverflowError) as exc:
                    raise DataError("type_conversion_failed", f"Invalid {field.type} in column {field.name}",
                                    table=name, line=index + 1) from exc
                converted[field.name] = result
            batch.append(converted)
            if len(batch) == batch_size:
                yield pa.RecordBatch.from_pylist(batch, schema=schema)
                batch = []
        if batch:
            yield pa.RecordBatch.from_pylist(batch, schema=schema)


def verify_roundtrip(source, name, path, schema, *, row_limit, batch_size):
    """Arrow 核对 schema／逐值／顺序；DuckDB 独立 SQL 读取同文件并逐值核对。"""
    nulls = dict.fromkeys(schema.names, 0)
    count, digest = 0, hashlib.sha256()
    with pq.ParquetFile(path) as stored:
        if not stored.schema_arrow.equals(schema, check_metadata=False):
            raise DataError("roundtrip_schema_mismatch", "Parquet schema differs", table=name)
        actual = stored.iter_batches(batch_size=batch_size)
        expected = typed_batches(source, name, schema, row_limit=row_limit, batch_size=batch_size)
        for left, right in zip_longest(expected, actual):
            if left is None or right is None or not left.equals(right):
                raise DataError("roundtrip_values_mismatch", "Parquet row count, order or values differ", table=name)
            count += left.num_rows
            for field, array in zip(schema, right.columns):
                nulls[field.name] += array.null_count
            for row in right.to_pylist():
                digest.update((json.dumps(list(row.values()), ensure_ascii=False, separators=(",", ":"),
                                          allow_nan=False) + "\n").encode("utf-8"))
    if not count:
        raise DataError("empty_dataset", "No rows were written", table=name)
    with duckdb.connect(config={"threads": "2", "memory_limit": "512MB"}) as connection:
        query = connection.execute(
            "SELECT * EXCLUDE (file_row_number) FROM read_parquet(?, file_row_number=true) ORDER BY file_row_number",
            [str(path)])
        expected_types = {"int64": "BIGINT", "double": "DOUBLE", "string": "VARCHAR"}
        description = [(item[0], str(item[1])) for item in query.description]
        if description != [(field.name, expected_types[str(field.type)]) for field in schema]:
            raise DataError("duckdb_schema_mismatch", "SQL column order or types differ", table=name)
        for batch in typed_batches(source, name, schema, row_limit=row_limit, batch_size=batch_size):
            expected_rows = [tuple(row.values()) for row in batch.to_pylist()]
            if query.fetchmany(batch.num_rows) != expected_rows:
                raise DataError("duckdb_values_mismatch", "SQL values or row order differ", table=name)
        if query.fetchone() is not None:
            raise DataError("duckdb_values_mismatch", "SQL has unexpected extra rows", table=name)
    return {"rows": count, "columns": schema.names, "types": [str(field.type) for field in schema],
            "null_counts": nulls, "ordered_values_sha256": digest.hexdigest(),
            "arrow_exact_match": True, "duckdb_exact_match": True, "duckdb_schema": description}


def verify_columnar_roundtrip(source, name, path, schema, *, row_limit, batch_size):
    """全量列式核对，复用相同严格转换；SQL 按物理行号核对，不把全表变成 Python 列表。"""
    nulls, count = dict.fromkeys(schema.names, 0), 0
    with pq.ParquetFile(path) as stored:
        if not stored.schema_arrow.equals(schema, check_metadata=False):
            raise DataError("roundtrip_schema_mismatch", "Parquet schema differs", table=name)
        for expected, actual in zip_longest(
                typed_batches(source, name, schema, row_limit=row_limit, batch_size=batch_size),
                stored.iter_batches(batch_size=batch_size)):
            if expected is None or actual is None or not expected.equals(actual):
                raise DataError("roundtrip_values_mismatch", "Parquet count, order or values differ", table=name)
            count += actual.num_rows
            for field, array in zip(schema, actual.columns):
                nulls[field.name] += array.null_count
    if not count:
        raise DataError("empty_dataset", "No rows were written", table=name)
    with duckdb.connect(config={"threads": "1", "memory_limit": "512MB"}) as con:
        relation = con.sql(
            "SELECT * EXCLUDE (file_row_number) FROM read_parquet(?, file_row_number=true) ORDER BY file_row_number",
            params=[str(path)])
        with relation.to_arrow_reader(batch_size) as reader, pq.ParquetFile(path) as stored:
            if not reader.schema.equals(schema, check_metadata=False):
                raise DataError("duckdb_schema_mismatch", "SQL schema differs", table=name)
            for expected, actual in zip_longest(stored.iter_batches(batch_size=batch_size), reader):
                if expected is None or actual is None or not expected.equals(actual):
                    raise DataError("duckdb_values_mismatch", "SQL count, order or values differ", table=name)
    return {"rows": count, "columns": schema.names, "types": [str(f.type) for f in schema],
            "null_counts": nulls, "arrow_exact_match": True, "duckdb_exact_match": True,
            "verification": "all_rows_all_values_columnar_in_source_order"}


def write_verified_parquet(source, name, destination, columns, *, output_root, row_limit, batch_size,
                           columnar=False):
    """验证临时文件后原子替换单文件。失败保留旧成功文件；不会追加或写原始目录。"""
    output_root, destination = Path(output_root).resolve(), Path(destination).resolve()
    if (not destination.is_relative_to(output_root) or destination.suffix != ".parquet"
            or output_root.is_relative_to(source.root) or source.root.is_relative_to(output_root)):
        raise DataError("unsafe_output_path", "Output must be separate from input and inside output root", table=name)
    if (row_limit is not None and (type(row_limit) is not int or row_limit <= 0)) or type(batch_size) is not int or batch_size <= 0:
        raise DataError("parquet_limits_invalid", "Positive row and batch limits required", table=name)
    schema = make_schema(columns)
    quality = inspect_table(source, name, hash_file=True, row_limit=row_limit).report
    if not quality["passed"]:
        raise DataError("source_quality_failed", "Selected CSV range failed shared quality checks", table=name)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
            for batch in typed_batches(source, name, schema, row_limit=row_limit, batch_size=batch_size):
                writer.write_batch(batch)
        verify = verify_columnar_roundtrip if columnar else verify_roundtrip
        result = verify(source, name, temporary, schema, row_limit=row_limit, batch_size=batch_size)
        if result["rows"] != quality["rows"]:
            raise DataError("roundtrip_values_mismatch", "Quality scan and conversion row counts differ", table=name)
        if sha256_file(source.path(name)) != quality["sha256"]:
            raise DataError("source_changed", "CSV changed during conversion", table=name)
        result.update(quality=quality, output_path=str(destination), bytes=temporary.stat().st_size,
                      sha256=sha256_file(temporary), compression="snappy")
        temporary.replace(destination)
        return result
    finally:
        if temporary.exists():
            temporary.unlink()
