"""E02 模型接口检查；不提供正式训练、调参、效果评估或评分入口。"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import uuid
import warnings

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .raw import DataError, RawDataSource, require_data_scope, sha256_file

TABLE = "application_train.csv"
ATOL = 1e-10
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "cloudpickle", "narwhals", "threadpoolctl", "duckdb", "pyarrow")


def environment_versions():
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in PACKAGES}}


def require_smoke_fit_allowed(plan_path, *, fixture=False):
    """正式环境拟合仅限实验冻结前；人工夹具仍由独立的数据范围检查约束。"""
    if fixture:
        return
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8-sig"))
        if not isinstance(plan, dict) or not isinstance(plan.get("status"), str):
            raise ValueError("missing experiment status")
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise DataError("smoke_experiment_plan_invalid", "Cannot establish pre-freeze scope for official smoke fitting") from exc
    # 未知状态也拒绝；已冻结锚点不能仅改status就当成未冻结草稿。
    if (plan["status"] != "draft_not_frozen"
            or any(key in plan for key in ("active_run_id", "frozen_at", "plan_payload_sha256"))):
        raise DataError("smoke_official_fit_locked", "Official smoke fitting is disabled after experiment freeze; use isolated synthetic fixtures")


@dataclass(frozen=True)
class ModelSmokeConfig:
    artifact_dir: Path
    features: tuple
    max_rows: int
    probe_fraction: float
    random_seed: int
    max_iter: int
    path: Path
    sha256: str


def load_model_config(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if set(values) != {"schema_version", "purpose", "artifact_dir", "features", "max_rows",
                           "probe_fraction", "random_seed", "max_iter"}:
            raise ValueError("unexpected fields")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1 or values["purpose"] != "smoke_check":
            raise ValueError("only smoke_check is supported")
        relative = values["artifact_dir"]
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("relative artifact directory required")
        output = (root / relative).resolve()
        if not output.is_relative_to(root / "models/smoke"):
            raise ValueError("model artifacts must stay in models/smoke")
        features = values["features"]
        if (not isinstance(features, list) or not features or len(features) > 20
                or any(not isinstance(name, str) or not name.strip() or name.startswith("SK_ID_") or name == "TARGET" for name in features)
                or len(features) != len(set(features))):
            raise ValueError("distinct numeric input features required; no ID or TARGET")
        for key, lower, upper in (("max_rows", 10, 10000), ("max_iter", 10, 1000), ("random_seed", 0, 2**32 - 1)):
            if type(values[key]) is not int or not lower <= values[key] <= upper:
                raise ValueError("smoke integer setting out of range")
        if type(values["probe_fraction"]) not in (int, float) or not 0.1 <= values["probe_fraction"] <= 0.4:
            raise ValueError("probe fraction out of range")
        return ModelSmokeConfig(output, tuple(features), values["max_rows"], float(values["probe_fraction"]),
                                values["random_seed"], values["max_iter"], path, sha256_file(path))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise DataError("model_smoke_config_invalid", "Model smoke config has invalid fields, paths or limits") from exc


def read_smoke_input(runtime, model_config, report_path, *, fixture=False):
    """只消费成功存储运行的本机产物；G1 仍由公共检查执行，不重扫原始 CSV。"""
    report_path = Path(report_path).resolve()
    try:
        if not report_path.is_relative_to(runtime.root / "reports"):
            raise DataError("smoke_input_path_invalid", "Storage run report must be in project reports")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise DataError("smoke_input_not_passed", "Storage run must have status=passed")
        expected_scope = "synthetic_parquet_smoke" if fixture else "official_prefix_parquet_smoke"
        if report["scope"] != expected_scope:
            raise DataError("smoke_input_scope_invalid", "Storage scope does not match fixture/official mode")
        metadata = report["metadata"]
        source = RawDataSource(metadata["data_root"] if fixture else runtime.raw_dir,
                               None if fixture else runtime.manifest)
        require_data_scope(source, runtime.root / "data/raw", fixture=fixture)
        if TABLE not in runtime.tables:
            raise DataError("smoke_table_not_configured", "Runtime must include labeled application table")
        info = report["tables"][TABLE]
        if not all(info.get(key) is True for key in ("passed", "arrow_exact_match", "duckdb_exact_match")):
            raise DataError("smoke_table_not_passed", "Input table must have passed both storage readers")
        if not fixture:
            if (metadata["manifest_sha256"] != sha256_file(runtime.manifest)
                    or Path(metadata["data_root"]).resolve() != runtime.raw_dir
                    or info["quality"]["sha256"] != source.entries[TABLE]["sha256"]):
                raise DataError("smoke_source_version_mismatch", "Storage lineage differs from current manifest/config")
        path = Path(info["output_path"]).resolve()
        if not path.is_relative_to(runtime.root / "data/interim"):
            raise DataError("smoke_input_path_invalid", "Stored input must stay in project data/interim")
        if sha256_file(path) != info["sha256"]:
            raise DataError("smoke_input_hash_mismatch", "Stored input fingerprint differs")
        with pq.ParquetFile(path) as stored:
            if not 10 <= stored.metadata.num_rows <= model_config.max_rows or stored.metadata.num_rows != info["rows"]:
                raise DataError("smoke_input_rows_invalid", "Input size differs or exceeds the bounded smoke limit")
            names = ["SK_ID_CURR", "TARGET", *model_config.features]
            if not set(names).issubset(stored.schema_arrow.names):
                raise DataError("smoke_input_columns_missing", "Required model smoke columns missing")
            for name in names:
                kind = stored.schema_arrow.field(name).type
                if not (pa.types.is_integer(kind) or pa.types.is_floating(kind)):
                    raise DataError("smoke_input_type_invalid", "Smoke input columns must be numeric")
                if name in ("SK_ID_CURR", "TARGET") and not pa.types.is_integer(kind):
                    raise DataError("smoke_input_type_invalid", "ID and TARGET must have integer types")
            table = stored.read(columns=names)
        records = table.to_pylist()
        if any(row["SK_ID_CURR"] is None or row["TARGET"] not in (0, 1) for row in records):
            raise DataError("smoke_labels_or_ids_invalid", "Missing ID or nonbinary label")
        ids = np.asarray([row["SK_ID_CURR"] for row in records], dtype=np.int64)
        y = np.asarray([row["TARGET"] for row in records], dtype=np.int64)
        X = np.asarray([[np.nan if row[name] is None else row[name] for name in model_config.features] for row in records], dtype=np.float64)
        if len(np.unique(ids)) != len(ids) or np.isinf(X).any():
            raise DataError("smoke_values_invalid", "Duplicate IDs or nonfinite numeric values")
        return X, y, ids, {"source_run_id": report["run_id"], "source_report_path": str(report_path),
            "source_report_sha256": sha256_file(report_path), "parquet_path": str(path),
            "parquet_sha256": info["sha256"], "raw_csv_sha256": info["quality"]["sha256"],
            "selection": metadata["selection"], "rows": len(ids)}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError("smoke_input_invalid", "Storage input report or artifact is unreadable/invalid") from exc


def id_fingerprint(ids):
    return hashlib.sha256(json.dumps([int(value) for value in ids], separators=(",", ":")).encode()).hexdigest()


def fit_smoke(X, y, ids, config):
    X, y, ids = np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(ids)
    if (X.ndim != 2 or X.shape != (len(y), len(config.features)) or len(ids) != len(y)
            or not 10 <= len(y) <= config.max_rows or len(np.unique(ids)) != len(ids) or np.isinf(X).any()):
        raise DataError("smoke_arrays_invalid", "Invalid shapes, IDs, row count or feature values")
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [0, 1] or min(counts) < 2:
        raise DataError("smoke_classes_invalid", "Both binary classes need at least two rows")
    try:
        train, probe = train_test_split(np.arange(len(y)), test_size=config.probe_fraction,
                                        random_state=config.random_seed, stratify=y)
    except ValueError as exc:
        raise DataError("smoke_split_invalid", "Smoke split cannot retain both classes") from exc
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("model", LogisticRegression(solver="lbfgs", C=1.0, max_iter=config.max_iter,
                                                       random_state=config.random_seed))])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X[train], y[train])
        probabilities = pipeline.predict_proba(X[probe])
        single = np.vstack([pipeline.predict_proba(row.reshape(1, -1)) for row in X[probe]])
    match = check_predictions(probabilities, single)
    split = {"purpose": "smoke_check_not_formal_split", "method": "stratified_random_split",
             "random_seed": config.random_seed, "probe_fraction": config.probe_fraction,
             "train_ids": [int(value) for value in ids[train]], "probe_ids": [int(value) for value in ids[probe]],
             "train_ids_sha256": id_fingerprint(ids[train]), "probe_ids_sha256": id_fingerprint(ids[probe]),
             "id_overlap": int(len(set(ids[train]) & set(ids[probe]))),
             "train_class_counts": {str(value): int((y[train] == value).sum()) for value in (0, 1)},
             "probe_class_counts": {str(value): int((y[probe] == value).sum()) for value in (0, 1)}}
    return pipeline, X[probe], probabilities, split, match


def check_predictions(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    for value in (expected, actual):
        if (value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all()
                or (value < 0).any() or (value > 1).any() or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=ATOL)):
            raise DataError("smoke_probability_invalid", "Invalid probability shape, bounds or sums")
    if expected.shape != actual.shape:
        raise DataError("smoke_prediction_mismatch", "Prediction shapes differ")
    difference = float(np.max(np.abs(expected - actual)))
    if difference > ATOL:
        raise DataError("smoke_prediction_mismatch", "Prediction difference exceeds 1e-10")
    return {"passed": True, "max_absolute_difference": difference, "absolute_tolerance": ATOL,
            "rows": len(expected), "probabilities_finite_and_in_range": True}


def load_smoke_bundle(directory, *, expected_metadata_sha256):
    """仅用于本项目刚生成的可信本机包；哈希校验不代表可安全加载外部 pickle。"""
    directory = Path(directory)
    try:
        metadata_path = directory / "bundle.json"
        if sha256_file(metadata_path) != expected_metadata_sha256:
            raise DataError("smoke_bundle_hash_mismatch", "Model metadata fingerprint differs")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or metadata["purpose"] != "smoke_check":
            raise DataError("smoke_bundle_invalid", "Bundle is not a model smoke artifact")
        if metadata["environment"] != environment_versions():
            raise DataError("smoke_environment_mismatch", "Model environment versions differ")
        if sha256_file(directory / "model.joblib") != metadata["model_sha256"]:
            raise DataError("smoke_bundle_hash_mismatch", "Model fingerprint differs")
        if sha256_file(directory / "probe.npz") != metadata["probe_sha256"]:
            raise DataError("smoke_bundle_hash_mismatch", "Probe fingerprint differs")
        if sha256_file(directory / "split.json") != metadata["split_sha256"]:
            raise DataError("smoke_bundle_hash_mismatch", "Split fingerprint differs")
        return joblib.load(directory / "model.joblib"), metadata
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError("smoke_bundle_invalid", "Model smoke bundle is unreadable or invalid") from exc


def save_verified_smoke(pipeline, probe, probabilities, split, config, destination, *, lineage):
    """在唯一临时目录写全包并读回；成功才提升目录，不覆盖已有成功包。"""
    root, destination = config.artifact_dir.resolve(), Path(destination).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise DataError("smoke_output_path_invalid", "Model bundle must be below configured smoke directory")
    if destination.exists():
        raise DataError("smoke_output_exists", "Refuse to replace an existing model bundle")
    stage = root / (destination.name + ".partial-" + uuid.uuid4().hex)
    try:
        stage.mkdir(parents=True, exist_ok=False)
        joblib.dump(pipeline, stage / "model.joblib", compress=3)
        np.savez_compressed(stage / "probe.npz", X=probe, expected=probabilities)
        (stage / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {"schema_version": 1, "purpose": "smoke_check", "environment": environment_versions(),
                    "features": list(config.features), "model_sha256": sha256_file(stage / "model.joblib"),
                    "probe_sha256": sha256_file(stage / "probe.npz"), "split_sha256": sha256_file(stage / "split.json"),
                    "lineage": lineage, "model_config_sha256": config.sha256,
                    "recipe": {"imputer": "median; keep_empty_features=True", "scaler": "StandardScaler",
                               "model": "LogisticRegression", "solver": "lbfgs", "C": 1.0,
                               "max_iter": config.max_iter, "native_threads": 1}}
        metadata_path = stage / "bundle.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fingerprint = sha256_file(metadata_path)
        restored, _ = load_smoke_bundle(stage, expected_metadata_sha256=fingerprint)
        with threadpool_limits(limits=1):
            match = check_predictions(probabilities, restored.predict_proba(probe))
        stage.rename(destination)
        return {"directory": str(destination), "metadata_sha256": fingerprint,
                "model_sha256": metadata["model_sha256"], "probe_sha256": metadata["probe_sha256"],
                "split_sha256": metadata["split_sha256"], "same_process_reload": match}
    finally:
        if stage.exists():
            assert stage.resolve().is_relative_to(root)
            shutil.rmtree(stage)
