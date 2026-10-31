"""只读、逐行 CSV 入口。保留原始字符串，不推断类型或填补缺失。"""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


class DataError(ValueError):
    """可供命令行和下游识别的错误；不包含原始记录值。"""

    def __init__(self, code, message, *, table=None, line=None):
        super().__init__(message)
        self.code, self.table, self.line = code, table, line

    def as_dict(self):
        return {"code": self.code, "message": str(self),
                "table": self.table, "line": self.line}


@dataclass(frozen=True)
class TableSpec:
    required: tuple[str, ...]
    primary_key: str | None = None
    target_role: str | None = None
    encoding: str = "utf-8-sig"
    allow_unnamed_column: bool = False
    forbidden: tuple[str, ...] = ()


# 仅登记已有证据支持的最小契约；历史子表组合键留待 G2。
TABLES = {
    "application_train.csv": TableSpec(("SK_ID_CURR", "TARGET"), "SK_ID_CURR", "observed_label"),
    "application_test.csv": TableSpec(("SK_ID_CURR",), "SK_ID_CURR", forbidden=("TARGET",)),
    "bureau.csv": TableSpec(("SK_ID_BUREAU", "SK_ID_CURR"), "SK_ID_BUREAU"),
    "previous_application.csv": TableSpec(("SK_ID_PREV", "SK_ID_CURR"), "SK_ID_PREV"),
    "bureau_balance.csv": TableSpec(("SK_ID_BUREAU", "MONTHS_BALANCE")),
    "POS_CASH_balance.csv": TableSpec(("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE")),
    "credit_card_balance.csv": TableSpec(("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE")),
    "installments_payments.csv": TableSpec(("SK_ID_PREV", "SK_ID_CURR")),
    "sample_submission.csv": TableSpec(("SK_ID_CURR", "TARGET"), "SK_ID_CURR", "submission_placeholder"),
    "HomeCredit_columns_description.csv": TableSpec(
        ("Table", "Row", "Description", "Special"), encoding="cp1252", allow_unnamed_column=True),
}


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_data_scope(source, official_root, *, fixture=False):
    """检查项目内部运行前提；G1 标志不代表平台许可已核验。"""
    official_root = Path(official_root).resolve()
    if fixture and (source.root.is_relative_to(official_root) or official_root.is_relative_to(source.root)):
        raise DataError("fixture_path_invalid", "Fixture root must be separate from data/raw")
    if not fixture and (source.manifest is None or source.manifest.get("G1_passed") is not True):
        raise DataError("g1_not_passed", "Official quality run requires the current project G1 gate to pass")


class RawDataSource:
    def __init__(self, root, manifest=None):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest).resolve() if manifest is not None else None
        self.manifest = None
        self.entries = {}
        if self.manifest_path is not None:
            try:
                self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
                files = self.manifest["files"]
                if not isinstance(files, list):
                    raise ValueError("files must be a list")
                for item in files:
                    name = item["name"]
                    if name not in TABLES or name in self.entries:
                        raise ValueError("unknown or duplicate table in manifest")
                    for field in ("bytes", "rows", "columns"):
                        if type(item[field]) is not int or item[field] < 0:
                            raise ValueError(f"invalid {field}")
                    fingerprint = item["sha256"]
                    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
                            c not in "0123456789abcdef" for c in fingerprint):
                        raise ValueError("invalid sha256")
                    if item["encoding"] != TABLES[name].encoding:
                        raise ValueError("encoding differs from table contract")
                    self.entries[name] = item
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                raise DataError("manifest_invalid", "Manifest is unreadable or has an invalid file entry") from exc

    def path(self, name):
        if name not in TABLES:
            raise DataError("unknown_table", "Table is not registered", table=name)
        try:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise DataError("path_outside_root", "Table resolves outside data root", table=name)
            if not path.is_file():
                raise DataError("path_missing", "CSV file does not exist", table=name)
        except OSError as exc:
            raise DataError("path_unreadable", "CSV path cannot be accessed", table=name) from exc
        return path

    @contextmanager
    def open_table(self, name, columns=None):
        """上下文结束时关闭文件。只验证结构；键／标签检查使用 inspect_table。"""
        path = self.path(name)
        spec = TABLES[name]
        try:
            stream = path.open(encoding=spec.encoding, newline="")
        except (OSError, LookupError) as exc:
            raise DataError("csv_read_error", "CSV could not be opened", table=name) from exc
        with stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader, None)
            except (OSError, UnicodeError, csv.Error) as exc:
                raise DataError("csv_read_error", "CSV header could not be read or decoded", table=name,
                                line=reader.line_num) from exc
            line = reader.line_num
            if not header:
                raise DataError("header_missing", "CSV header is empty", table=name, line=line)
            if len(header) != len(set(header)):
                raise DataError("duplicate_columns", "CSV has duplicate column names", table=name, line=line)
            if any(not h.strip() for h in header) and not spec.allow_unnamed_column:
                raise DataError("unnamed_column", "CSV has an unnamed column", table=name, line=line)
            selected = tuple(header) if columns is None else tuple(columns)
            if not selected or len(selected) != len(set(selected)):
                raise DataError("invalid_projection", "Select a nonempty set of distinct columns", table=name)
            missing = (set(spec.required) | set(selected)) - set(header)
            if missing:
                raise DataError("columns_missing", f"Required columns missing: {sorted(missing)}", table=name)
            if set(spec.forbidden) & set(header):
                raise DataError("forbidden_columns", "CSV contains a forbidden label column", table=name)

            def rows():
                try:
                    for values in reader:
                        if not values:
                            raise DataError("empty_record", "CSV contains a blank record", table=name, line=reader.line_num)
                        if len(values) != len(header):
                            raise DataError("row_width", "CSV row width differs from header", table=name, line=reader.line_num)
                        record = dict(zip(header, values))
                        yield {column: record[column] for column in selected}
                except (OSError, UnicodeError, csv.Error) as exc:
                    raise DataError("csv_read_error", "CSV row could not be read or decoded", table=name,
                                    line=reader.line_num) from exc

            yield header, rows()
