"""E05：固定字段契约、申请级分层划分和训练内CV；无训练或模型效果计算。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, train_test_split

from .history import FEATURE_NAMES, load_history_bundle
from .normalize import load_normalized_bundle, write_json
from .raw import DataError, sha256_file
from .runtime import now

VERSION = "application_contract_v1"
MAIN_FEATURES = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"]
FAMILIES = ["logistic_regression", "traditional_scorecard", "one_tree_model"]
PARTITIONS = ("train", "validation", "test")
LIMITS = {"natural_person_id": "not_available; person-disjointness not verified",
          "absolute_application_time": "not_available; not an out-of-time validation",
          "target_horizon": "X and Y undisclosed; no fixed-horizon PD claim",
          "population": "released labeled applications only; no rejected-applicant outcomes or current-bank representativeness",
          "prior_environment_smoke": "E02 used first 5000 applications for interface fitting/probes before this split; no performance selection; its fitted preprocessing and models must not be reused",
          "availability": "source-dictionary application snapshot and D018 history rules; not a reconstructed production as-of database"}


@dataclass(frozen=True)
class ExperimentRecipe:
    root: Path
    path: Path
    artifact_dir: Path
    sha256: str
    values: dict


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_experiment_recipe(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        v = json.loads(path.read_text(encoding="utf-8-sig"))
        expected = {"schema_version", "recipe_version", "artifact_dir", "random_seed", "split_ratios", "cv_folds", "main_features",
                    "history_features", "model_families", "max_candidate_settings_per_family", "calibration_method",
                    "threshold_selection", "model_selection", "learned_processing_scope", "test_policy", "case_ids"}
        if set(v) != expected or type(v["schema_version"]) is not int or v["schema_version"] != 1 or v["recipe_version"] != VERSION:
            raise ValueError("unsupported recipe")
        if v["main_features"] != MAIN_FEATURES or v["history_features"] != "all_24_verified_E04_columns":
            raise ValueError("fixed feature version required")
        if type(v["random_seed"]) is not int or not 0 <= v["random_seed"] < 2**32:
            raise ValueError("invalid random seed")
        if v["split_ratios"] != {"train": .7, "validation": .15, "test": .15} or type(v["cv_folds"]) is not int or v["cv_folds"] != 5:
            raise ValueError("unsupported split or folds")
        policies = {"model_families": FAMILIES, "max_candidate_settings_per_family": 10,
                    "calibration_method": "none_for_first_comparison", "threshold_selection": "max_validation_ks_tie_highest_finite_threshold",
                    "model_selection": "validation_auc_with_0.002_tolerance_then_brier_then_simplicity",
                    "learned_processing_scope": "training_partition_and_inside_each_training_cv_fold", "test_policy": "reserved_until_E09_selection_locked"}
        if any(v[k] != wanted for k,wanted in policies.items()):
            raise ValueError("unsupported experiment policy")
        if (not isinstance(v["case_ids"], list) or not 1 <= len(v["case_ids"]) <= 10
                or len(v["case_ids"]) != len(set(v["case_ids"])) or any(type(k) is not int or k <= 0 for k in v["case_ids"])):
            raise ValueError("invalid cases")
        relative = v["artifact_dir"]
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("relative output required")
        output = (root / relative).resolve()
        if not output.is_relative_to(root / "data/processed") or output.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe experiment output")
        return ExperimentRecipe(root, path, output, sha256_file(path), v)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise DataError("experiment_recipe_invalid", "Invalid feature, split, experiment policy or path") from exc


def make_split(ids, target, seed, folds=5):
    """输入按 ID 排序后划分；标签仅用于分层和计数，不用于选择字段或拟合。"""
    ids, target = np.asarray(ids), np.asarray(target)
    if (ids.ndim != 1 or target.shape != ids.shape or ids.dtype.kind not in "iu" or
            len(ids) == 0 or len(np.unique(ids)) != len(ids) or not np.isin(target, [0,1]).all()):
        raise DataError("experiment_population_invalid", "Unique integer IDs and binary labels required")
    order = np.argsort(ids, kind="stable")
    ids, target = ids[order], target[order].astype(np.int64)
    try:
        train, rest = train_test_split(np.arange(len(ids)), test_size=.3, stratify=target, random_state=seed)
        validation, test = train_test_split(rest, test_size=.5, stratify=target[rest], random_state=seed)
        parts = {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
        for name, indexes in parts.items():
            counts = np.bincount(target[indexes], minlength=2)
            if counts.min() < (folds if name == "train" else 2):
                raise ValueError("insufficient class counts")
        partition = np.empty(len(ids), dtype=object)
        cv = np.full(len(ids), -1, dtype=np.int64)
        for name, indexes in parts.items():
            partition[indexes] = name
        for fold, (_, heldout) in enumerate(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(train, target[train]), 1):
            cv[train[heldout]] = fold
        if not ((cv[parts["train"]] >= 1).all() and (cv[parts["validation"]] == -1).all() and (cv[parts["test"]] == -1).all()):
            raise ValueError("invalid CV membership")
        return ids, target, order, parts, partition, cv
    except (ValueError, IndexError) as exc:
        raise DataError("experiment_split_invalid", "Both labels need sufficient samples in all partitions and five training folds") from exc


def contract_for(loaded, columns):
    original = {c["name"]: c for c in columns}
    history = {c["name"]: c for c in loaded["contract"]["features"]}
    included, review = [], []
    for name in MAIN_FEATURES:
        if name not in original:
            raise DataError("experiment_feature_missing", f"Required main feature missing: {name}")
        c = original[name]
        included.append({"name": name, "type": c["type"], "source": "application_train.csv", "description": c["description"],
            "formula": "identity; no fitted or global transformation", "unit": c["unit"],
            "availability": "applicant-provided asset/income/work/contact information or application product/day/hour as described in supplied dictionary",
            "missing": "preserve null and literal categories; imputation and vocabulary learned only in training/folds",
            "special_values": "preserve; no income clipping, arbitrary negative replacement or category normalization"})
    included.extend(history[n] for n in FEATURE_NAMES)
    for name,c in original.items():
        if name in MAIN_FEATURES:
            reason, role = "included_by_fixed_source_based_contract", "feature"
        elif name == "SK_ID_CURR":
            reason, role = "join_and_split_key_never_feature", "identifier"
        elif name == "TARGET":
            reason, role = "outcome_only_never_feature", "label"
        elif name in ("AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
            reason, role = "initial_application_availability_of_recorded_current_loan_terms_not_established", "held"
        elif name == "DAYS_EMPLOYED":
            reason, role = "unexplained_365243_code_retained_upstream_no_automatic_missing_mapping", "held"
        elif name.startswith("EXT_SOURCE_"):
            reason, role = "external_score_generation_time_and_composition_undisclosed", "held"
        elif name in ("CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"):
            reason, role = "demographic_fields_not_selected_for_first_contract_no_fairness_guarantee", "held"
        elif "SOCIAL_CIRCLE" in name:
            reason, role = "social_observation_window_asof_time_not_disclosed", "held"
        else:
            reason, role = "not_in_first_fixed_allowlist; detailed_timing_units_or_special_values_not_promoted", "held"
        review.append({"name": name, "role": role, "reason": reason, "source_description": c["description"]})
    return {"version": VERSION, "unit": "one current application", "feature_columns": MAIN_FEATURES + FEATURE_NAMES,
            "main_features": MAIN_FEATURES, "history_features": FEATURE_NAMES, "features": included,
            "source_main_column_review": review, "forbidden": ["SK_ID_CURR", "TARGET", "CV_FOLD", "partition", "all_history_audit_fields"],
            "limitations": LIMITS, "label_definition": original["TARGET"]["description"],
            "preprocessing": {"fit_scope": "training_only_and_each_training_CV_fold", "numeric": "training-fitted imputation/scaling/binning as required by family; retain empty feature dimensions",
                              "categorical": "training-fitted vocabulary or WOE/binning; explicit missing and unknown handling",
                              "feature_filtering": "fixed source-based allowlist here; any learned filter training/folds only"}}


def verify_feature_values(table, contract):
    for c in contract["features"]:
        name, column = c["name"], table[c["name"]]
        expected = {"int64": pa.int64(), "float64": pa.float64(), "string": pa.string()}[c["type"]]
        if column.type != expected:
            raise DataError("experiment_feature_type", f"Unexpected feature type: {name}")
        if not pa.types.is_string(column.type):
            if pc.any(pc.invert(pc.is_finite(column))).as_py() is True:
                raise DataError("experiment_nonfinite", f"Nonfinite feature: {name}")
        if name.startswith("FLAG_") and name not in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
            if pc.any(pc.and_(pc.not_equal(column, 0), pc.not_equal(column, 1))).as_py() is True:
                raise DataError("experiment_flag_invalid", f"Nonbinary contact flag: {name}")
        if name == "HOUR_APPR_PROCESS_START" and pc.any(pc.or_(pc.less(column, 0), pc.greater(column, 23))).as_py() is True:
            raise DataError("experiment_hour_invalid", "Application hour outside 0..23")


def save_arrow(path, table):
    pq.write_table(table, path, compression="snappy", row_group_size=65536)
    restored = pq.read_table(path)
    if not table.equals(restored):
        raise DataError("experiment_roundtrip_failed", "Experiment Parquet differs after readback")
    return {"rows": table.num_rows, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_experiment(recipe, history_report, run_id, *, fixture=False, application_limit=None):
    if application_limit is not None and (type(application_limit) is not int or not 50 <= application_limit <= 100000):
        raise DataError("experiment_limit_invalid", "Bounded checks require 50..100000 applications")
    if not isinstance(run_id,str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise DataError("experiment_run_id_invalid", "Unsafe run identifier")
    history_report = Path(history_report).resolve()
    loaded = load_history_bundle(recipe.root, history_report, require_official=not fixture)
    upstream_report = json.loads(history_report.read_text(encoding="utf-8"))
    if upstream_report["scope"].startswith("synthetic_") != fixture:
        raise DataError("experiment_scope_mismatch", "Fixture flag and upstream scope disagree")
    normal = load_normalized_bundle(recipe.root, Path(loaded["bundle"]["input_run_path"]), require_official=not fixture)
    source_contract = json.loads((normal["directory"] / "contract.json").read_text(encoding="utf-8"))
    contract = contract_for(loaded, source_contract["tables"]["application_train.csv"])
    output = recipe.artifact_dir.resolve()
    if output.is_relative_to(loaded["directory"]) or loaded["directory"].is_relative_to(output):
        raise DataError("unsafe_output_path", "Experiment output overlaps input")
    destination = output / ("full" if application_limit is None else "prefix") / run_id
    if destination.exists():
        raise DataError("experiment_exists", "Immutable experiment exists; use a new run ID")
    staging = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        selected = pq.read_table(loaded["joined_features"], columns=["SK_ID_CURR", "TARGET", *contract["feature_columns"]])
        if application_limit is not None:
            selected = selected.slice(0, application_limit)
        if selected["SK_ID_CURR"].null_count or selected["TARGET"].null_count:
            raise DataError("experiment_population_invalid", "Missing ID or label")
        verify_feature_values(selected, contract)
        ids, y, order, parts, partition, cv = make_split(selected["SK_ID_CURR"].to_numpy(), selected["TARGET"].to_numpy(), recipe.values["random_seed"])
        selected = selected.take(pa.array(order))
        cv_array = pa.array([None if i < 0 else int(i) for i in cv], type=pa.int64())
        split_table = pa.table({"SK_ID_CURR": pa.array(ids, type=pa.int64()), "partition": pa.array(partition), "CV_FOLD": cv_array})
        files = {"split_manifest.parquet": save_arrow(staging / "split_manifest.parquet", split_table)}
        data = selected.append_column("CV_FOLD", cv_array)
        counts = {}
        for name, indexes in parts.items():
            subset = data.take(pa.array(indexes))
            if not np.array_equal(subset["SK_ID_CURR"].to_numpy(), ids[indexes]) or not np.array_equal(subset["TARGET"].to_numpy(), y[indexes]):
                raise DataError("experiment_labels_changed", "Partition IDs and labels differ from source")
            files[name + ".parquet"] = save_arrow(staging / (name + ".parquet"), subset)
            classes = np.bincount(y[indexes], minlength=2)
            counts[name] = {"rows": len(indexes), "TARGET_0": int(classes[0]), "TARGET_1": int(classes[1]),
                            "event_rate": float(classes[1]/len(indexes)), "id_sha256": digest_json(ids[indexes].tolist())}
        folded = []
        for fold in range(1,6):
            held = (cv == fold)
            fold_y = y[held]
            values = np.bincount(fold_y, minlength=2)
            if values.min() < 1:
                raise DataError("experiment_cv_invalid", "A training CV fold lacks a class")
            folded.append({"fold":fold,"rows":int(held.sum()),"TARGET_0":int(values[0]),"TARGET_1":int(values[1])})
        if set(np.concatenate(list(parts.values())).tolist()) != set(range(len(ids))) or sum(len(i) for i in parts.values()) != len(ids):
            raise DataError("experiment_partition_overlap", "Partition membership not disjoint and complete")
        cases = {}
        for key in recipe.values["case_ids"]:
            where = np.flatnonzero(ids == key)
            if len(where) != 1:
                raise DataError("experiment_case_missing", "Contract case is outside selected population")
            index = int(where[0])
            cases[str(key)] = {"partition": partition[index], "CV_FOLD": None if cv[index] < 0 else int(cv[index]),
                               "raw_projected_features": selected.select(contract["feature_columns"]).slice(index,1).to_pylist()[0],
                               "identity_projection_verified": True, "history_case_trace": "E04 case_trace.json if this ID was checked upstream"}
        quality = {"passed":True,"scope":"full_labeled_population" if application_limit is None else "prefix_only",
                   "rows":len(ids),"feature_count":len(contract["feature_columns"]),"partitions":counts,"training_cv_folds":folded,
                   "id_missing":0,"id_duplicates":0,"cross_partition_id_intersections":0,"rows_removed":0,
                   "labels_unchanged":True,"feature_values_and_nulls_identity_projected":True,"fitted_preprocessors":[],
                   "test_labels_used_for":"stratification and partition counts only; no metrics, tuning or feature selection",
                   "G3_ready_for_freeze":not fixture and application_limit is None,"limitations":LIMITS}
        write_json(staging / "feature_contract.json", contract)
        write_json(staging / "quality.json", quality)
        write_json(staging / "cases.json", cases)
        write_json(staging / "recipe.json", recipe.values)
        protocol = {"version":VERSION,"status":"validated_protocol_snapshot", "dataset_version":loaded["bundle"]["files"]["joined_features.parquet"]["sha256"],
                    "target_definition":contract["label_definition"],"target_X":None,"target_Y":None,"limitations":LIMITS,
                    "random_seed":recipe.values["random_seed"],"split_strategy":"ID_sorted_two_stage_stratified_shuffle",
                    "split_ratios":recipe.values["split_ratios"],"cv_folds":5,"cv_scope":"training_only; folds persisted in split manifest",
                    "model_families":FAMILIES,"max_candidate_settings_per_family":10,
                    "candidate_registration":"family-specific immutable candidate recipe must be registered before first fit; not selected by final test",
                    "calibration_method":recipe.values["calibration_method"],"threshold_selection_scope":"validation_only",
                    "threshold_selection_rule":recipe.values["threshold_selection"],"model_selection":recipe.values["model_selection"],
                    "model_selection_details":{"auc_tolerance":.002,"within_tolerance":"lower validation Brier; exact ties prefer logistic_regression, then traditional_scorecard, then tree",
                                               "tolerance_is_not_statistical_significance":True},
                    "metrics":["roc_auc","ks","brier_score","calibration_curve"],"simple_baseline":"training event rate only",
                    "comparisons":["all_40_features_across_three_families","main_16_vs_main_plus_history_with_same_ID_partitions"],
                    "fit_scope":contract["preprocessing"],"test_policy":recipe.values["test_policy"],"test_used_for_selection":False,
                    "feature_contract_sha256":sha256_file(staging / "feature_contract.json"),"recipe_sha256":recipe.sha256,
                    "split_manifest_sha256":files["split_manifest.parquet"]["sha256"],
                    "partition_files_sha256":{p:files[p+".parquet"]["sha256"] for p in PARTITIONS}}
        protocol_fingerprint = digest_json(protocol)
        write_json(staging / "protocol.json", protocol)
        load_history_bundle(recipe.root, history_report, require_official=not fixture)
        if sha256_file(recipe.path) != recipe.sha256:
            raise DataError("experiment_recipe_changed", "Recipe changed during experiment preparation")
        all_files = {p.name:{"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in staging.iterdir() if p.is_file()}
        bundle = {"schema_version":1,"module":"E05","status":"complete","run_id":run_id,"prepared_at":now(),
                  "scope":quality["scope"],"fixture":fixture,"application_limit":application_limit,"protocol_fingerprint":protocol_fingerprint,
                  "input_run_path":str(history_report),"input_run_sha256":sha256_file(history_report),
                  "input_bundle_sha256":sha256_file(loaded["directory"] / "bundle.json"),"files":all_files}
        write_json(staging / "bundle.json", bundle)
        staging.replace(destination)
        return {"artifact_dir":str(destination),"bundle_sha256":sha256_file(destination / "bundle.json"),"scope":quality["scope"],
                "protocol_fingerprint":protocol_fingerprint,"quality":quality,"files":all_files}
    finally:
        if staging.exists():
            if not staging.resolve().is_relative_to(output) or not staging.name.endswith(".partial"):
                raise DataError("unsafe_cleanup", "Experiment staging cleanup escaped output root")
            shutil.rmtree(staging)


def verify_experiment_bundle(root, report_path, *, require_official=True):
    """只读完整性检查；尚未激活的全量产物也能检查，不能据此进入训练。"""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = json.loads(report_path.read_text(encoding="utf-8"))
        result = r["experiment"]
        if r["status"] != "passed" or r["metadata"]["module"] != "E05":
            raise ValueError("successful E05 required")
        if r["scope"] not in ("official_full_experiment_contract", "synthetic_full_experiment_contract"):
            raise ValueError("full scope required")
        if require_official and r["scope"] != "official_full_experiment_contract":
            raise ValueError("official full scope required")
        folder = Path(result["artifact_dir"]).resolve()
        if not folder.is_relative_to(root / "data/processed") or folder.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("unsafe output path")
        if sha256_file(folder / "bundle.json") != result["bundle_sha256"]:
            raise ValueError("bundle hash differs")
        b = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        expected = {"train.parquet","validation.parquet","test.parquet","split_manifest.parquet","feature_contract.json","quality.json","cases.json","recipe.json","protocol.json"}
        if (b["status"] != "complete" or b["module"] != "E05" or b["run_id"] != r["run_id"] or b["scope"] != "full_labeled_population"
                or b["application_limit"] is not None or set(b["files"]) != expected or b["fixture"] != r["scope"].startswith("synthetic_")):
            raise ValueError("incomplete or wrong scope")
        for name,info in b["files"].items():
            p = (folder / name).resolve()
            if not p.is_relative_to(folder) or p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
                raise ValueError("artifact changed")
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        contract = json.loads((folder / "feature_contract.json").read_text(encoding="utf-8"))
        if (digest_json(protocol) != b["protocol_fingerprint"] or b["protocol_fingerprint"] != result["protocol_fingerprint"]
                or contract["feature_columns"] != MAIN_FEATURES + FEATURE_NAMES or protocol["version"] != VERSION):
            raise ValueError("protocol or feature allowlist differs")
        upstream = Path(b["input_run_path"])
        if sha256_file(upstream) != b["input_run_sha256"]:
            raise ValueError("upstream report changed")
        loaded = load_history_bundle(root, upstream, require_official=require_official)
        if sha256_file(loaded["directory"] / "bundle.json") != b["input_bundle_sha256"]:
            raise ValueError("upstream bundle differs")
        return {"directory":folder,"report":r,"bundle":b,"protocol":protocol,"contract":contract}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise DataError("experiment_bundle_invalid", "Failed, incomplete, bounded, incompatible or changed E05 output") from exc


def activate_experiment(root, plan_path, report_path, expected_plan_sha256):
    """最后一步原子激活；失败保留旧方案；同一协议重跑幂等，不静默更换冻结实验。"""
    root, plan_path, report_path = Path(root).resolve(), Path(plan_path).resolve(), Path(report_path).resolve()
    if plan_path != (root / "config/experiment_plan.json").resolve():
        raise DataError("experiment_plan_path_invalid", "Only the project's experiment plan may be activated")
    loaded = verify_experiment_bundle(root, report_path)
    if loaded["report"]["experiment"]["quality"]["G3_ready_for_freeze"] is not True:
        raise DataError("g3_not_ready", "E05 must pass full official G3 preparation")
    lock = plan_path.with_suffix(".json.lock")
    temporary = plan_path.with_name(plan_path.name + "." + uuid.uuid4().hex + ".partial")
    token, owns_lock = uuid.uuid4().hex, False
    try:
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write(token)
            owns_lock = True
        except FileExistsError as exc:
            raise DataError("experiment_plan_locked", "Another activation lock exists; inspect it before retrying") from exc
        current = json.loads(plan_path.read_text(encoding="utf-8"))
        fingerprint = loaded["bundle"]["protocol_fingerprint"]
        if current.get("status") == "frozen":
            if current.get("plan_payload_sha256") != digest_json({k:v for k,v in current.items() if k != "plan_payload_sha256"}):
                raise DataError("experiment_plan_changed", "Frozen plan integrity check failed")
            if current.get("protocol_fingerprint") == fingerprint:
                return {"status":"already_frozen_same_protocol","active_run_id":current["active_run_id"],"plan_sha256":sha256_file(plan_path)}
            raise DataError("experiment_frozen_conflict", "A different protocol is frozen; a reviewed new version is required")
        if current.get("status") != "draft_not_frozen" or sha256_file(plan_path) != expected_plan_sha256:
            raise DataError("experiment_plan_changed", "Draft plan changed since preparation")
        p = loaded["protocol"]
        updated = {**current,"status":"frozen","frozen_at":now(),"active_run_id":loaded["report"]["run_id"],
            "active_report_path":str(report_path),"active_report_sha256":sha256_file(report_path),
            "active_bundle_sha256":sha256_file(loaded["directory"] / "bundle.json"),"protocol_fingerprint":fingerprint,
            "dataset_version":p["dataset_version"],"target_definition_verified":True,
            "target_verification_scope":"published binary definition verified; X/Y undisclosed",
            "entity_structure_verified":False,"application_time_verified":False,"verification_limitations":LIMITS,
            "random_seed":p["random_seed"],"split_strategy":p["split_strategy"],"split_ratios":p["split_ratios"],
            "split_manifest_sha256":p["split_manifest_sha256"],"feature_contract_sha256":p["feature_contract_sha256"],
            "feature_count":len(MAIN_FEATURES+FEATURE_NAMES),"cv_folds":5,
            "calibration_method":p["calibration_method"],"threshold_selection_rule":p["threshold_selection_rule"],
            "model_selection":p["model_selection"],"G3_passed_for_selected_contract":True,
            "training_started":False,"final_test_evaluated":False}
        updated["plan_payload_sha256"] = digest_json(updated)
        write_json(temporary, updated)
        if sha256_file(plan_path) != expected_plan_sha256:
            raise DataError("experiment_plan_changed", "Draft changed before atomic publication")
        temporary.replace(plan_path)
        return {"status":"frozen","active_run_id":updated["active_run_id"],"plan_sha256":sha256_file(plan_path)}
    finally:
        if temporary.exists():
            temporary.unlink()
        if owns_lock and lock.exists() and lock.read_text(encoding="utf-8") == token:
            lock.unlink()


def load_training_partition(root, report_path, partition="train", *, require_official=True):
    """E06～E08 接口：只有激活协议的 train／validation；最终测试留 E09 锁定方案后开发读取。"""
    if partition not in ("train", "validation"):
        raise DataError("reserved_test_locked", "Current training API cannot load the reserved final test partition")
    loaded = verify_experiment_bundle(root, report_path, require_official=require_official)
    plan_path = Path(root).resolve() / "config/experiment_plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("plan_payload_sha256") != digest_json({k:v for k,v in plan.items() if k != "plan_payload_sha256"}):
            raise ValueError("frozen plan payload changed")
        if plan["status"] != "frozen" or plan["protocol_fingerprint"] != loaded["bundle"]["protocol_fingerprint"]:
            raise ValueError("active protocol differs")
        anchor = Path(plan["active_report_path"])
        if sha256_file(anchor) != plan["active_report_sha256"]:
            raise ValueError("active report changed")
        active = verify_experiment_bundle(root, anchor, require_official=require_official)
        if sha256_file(active["directory"] / "bundle.json") != plan["active_bundle_sha256"]:
            raise ValueError("active bundle changed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError("experiment_not_active", "A matching complete frozen protocol is required before training") from exc
    table = pq.read_table(loaded["directory"] / (partition + ".parquet"))
    return {"features":table.select(loaded["contract"]["feature_columns"]),"target":table["TARGET"],"ids":table["SK_ID_CURR"],
            "cv_fold":table["CV_FOLD"],"contract":loaded["contract"],"protocol":loaded["protocol"]}
