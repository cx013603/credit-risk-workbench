"""E08: one native categorical histogram boosting family; no final-test reader."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .evaluation import ranking_metrics, validation_report, prediction_agreement
from .experiments import digest_json, load_training_partition
from .logistic import environment, matrix
from .normalize import write_json
from .raw import DataError, sha256_file
from .runtime import now, process_peak_memory

VERSION = 'tree_v1'
CODE = ['src/credit_risk/tree.py', 'scripts/run_tree.py', 'scripts/check_tree_model.py',
        'src/credit_risk/logistic.py', 'src/credit_risk/evaluation.py', 'src/credit_risk/experiments.py',
        'src/credit_risk/runtime.py', 'src/credit_risk/raw.py', 'src/credit_risk/normalize.py',
        'scripts/_project.py']


def load_recipe(path):
    try:
        r = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        expected = {
            'version': VERSION, 'family': 'hist_gradient_boosting',
            'candidates': [{'id': 1, 'max_leaf_nodes': 15, 'l2_regularization': 1.},
                           {'id': 2, 'max_leaf_nodes': 31, 'l2_regularization': 1.},
                           {'id': 3, 'max_leaf_nodes': 31, 'l2_regularization': 10.}],
            'learning_rate': .08, 'max_iter': 120, 'min_samples_leaf': 40, 'max_bins': 255,
            'random_state': 42, 'early_stopping': False, 'class_weight': None, 'calibration': 'none',
            'native_threads': 2, 'cv_folds': 5, 'features': 'all_frozen_40',
            'numeric': 'native_nan_except_fit_all_empty_forced_zero_no_scaling_clipping_or_selection',
            'categorical': 'stateless_literal_prefix_explicit_missing_then_native_fit_only_ordinal_unknown_nan_max255',
            'degenerate': 'retain_constant_and_all_empty_no_split_unless_missingness_varies',
            'native_binning': 'fit_only_sklearn_200000_bootstrap_subsample_when_larger_seeded_by_estimator',
            'selection': 'mean_cv_auc_then_lower_mean_brier_then_candidate_id',
            'explanation': {'method': 'validation_permutation_auc_and_brier', 'rows': 5000,
                            'sampling': 'lowest_application_ID', 'repeats': 2, 'seed': 31415},
            'stopping': {'max_total_seconds': 3600, 'max_fit_seconds': 300,
                         'max_parent_peak_bytes': 2147483648,
                         'check': 'before_and_after_each_fit_and_diagnostic_no_midfit_interrupt'}}
        if ({k: v for k, v in r.items() if k != 'protocol_fingerprint'} != expected
                or len(r['protocol_fingerprint']) != 64):
            raise ValueError('unsupported immutable recipe')
        return r
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('tree_recipe_invalid', 'Invalid fixed tree recipe') from exc


def register(root, recipe, protocol):
    root = Path(root).resolve()
    if recipe['protocol_fingerprint'] != digest_json(protocol):
        raise DataError('tree_protocol_mismatch', 'Tree recipe and active protocol differ')
    payload = {'purpose': 'formal_family_recipe', 'recipe': recipe,
               'code_sha256': {p: sha256_file(root/p) for p in CODE}, 'environment': environment()}
    path = root/'reports/recipes/official'/VERSION/'registration.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'registered_at': now(), 'payload': payload, 'fingerprint': digest_json(payload)}
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    except FileExistsError:
        old = json.loads(path.read_text(encoding='utf-8'))
        if old.get('payload') != payload or old.get('fingerprint') != digest_json(payload):
            raise DataError('tree_recipe_conflict', 'Existing tree recipe differs; reviewed new version required')
    return path


class EmptyNumericGuard(TransformerMixin, BaseEstimator):
    """sklearn 1.9.1 all-NaN binning workaround, learned on this fit only.

    Keep every dimension. A numeric dimension all empty in fit is always mapped
    to zero, including later nonempty inputs: the fitted tree cannot learn from it.
    Other NaN values are left for native missing-value routing.
    """
    def __init__(self, categorical):
        self.categorical = categorical

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.empty_indices_ = [j for j, cat in enumerate(self.categorical)
                               if not cat and np.isnan(np.asarray(X[:, j], dtype=float)).all()]
        return self

    def transform(self, X):
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise DataError('tree_input_shape', 'Exact fitted feature width required')
        out = X.copy()
        for j in self.empty_indices_:
            out[:, j] = 0.
        return out


def fit_tree(X, y, contract, candidate, recipe):
    columns = contract['feature_columns']
    if X.shape != (len(y), len(columns)) or set(np.unique(y)) != {0, 1}:
        raise DataError('tree_fit_input', 'Exact feature shape and both binary classes required')
    categorical = [c['type'] == 'string' for c in contract['features']]
    for j, cat in enumerate(categorical):
        if cat:
            if len(np.unique(X[:, j])) > recipe['max_bins']:
                raise DataError('tree_cardinality', 'Training category count exceeds native tree limit')
        elif np.isinf(np.asarray(X[:, j], dtype=float)).any():
            raise DataError('tree_nonfinite', 'Infinite numeric feature rejected')
    # The estimator's own ColumnTransformer/OrdinalEncoder and histogram binning fit
    # inside each fit call. No prefit vocabulary, imputer, scaler or smoke object.
    tree = HistGradientBoostingClassifier(
        loss='log_loss', learning_rate=recipe['learning_rate'], max_iter=recipe['max_iter'],
        max_leaf_nodes=candidate['max_leaf_nodes'], max_depth=None,
        min_samples_leaf=recipe['min_samples_leaf'], l2_regularization=candidate['l2_regularization'],
        max_features=1., max_bins=recipe['max_bins'], categorical_features=categorical,
        early_stopping=False, warm_start=False, random_state=recipe['random_state'], class_weight=None)
    model = Pipeline([('empty', EmptyNumericGuard(categorical)), ('tree', tree)])
    started = time.monotonic()
    with threadpool_limits(limits=recipe['native_threads']):
        model.fit(X, y)
    if time.monotonic()-started > recipe['stopping']['max_fit_seconds']:
        raise DataError('tree_fit_budget', 'Fit exceeded predeclared elapsed-time limit')
    if tree.n_iter_ != recipe['max_iter'] or tree.do_early_stopping_:
        raise DataError('tree_iterations', 'Fixed iterations without early stopping required')
    return model


def predict(model, X):
    # This helper is tree-specific because strings are encoded by the estimator.
    with threadpool_limits(limits=2):
        p = model.predict_proba(X)[:, 1]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataError('tree_probability', 'Invalid tree probabilities')
    return p


def fit_state(model, X, contract):
    """Version-pinned diagnostics of native fit state, not a second transformer."""
    columns = contract['feature_columns']
    empty_columns = [columns[j] for j in model.named_steps['empty'].empty_indices_]
    X = model.named_steps['empty'].transform(X)
    model = model.named_steps['tree']
    mask = np.asarray(model.is_categorical_)
    cats = [n for n, b in zip(columns, mask) if b]
    numeric = [n for n, b in zip(columns, mask) if not b]
    internal_order = cats+numeric
    encoder = model._preprocessor.named_transformers_['encoder']
    categories = {n: v.tolist() for n, v in zip(cats, encoder.categories_)}
    thresholds = {n: v.tolist() for n, v in zip(internal_order, model._bin_mapper.bin_thresholds_)}
    used = set()
    split_counts = dict.fromkeys(columns, 0)
    for iteration in model._predictors:
        for tree in iteration:
            nodes = tree.nodes
            for idx in nodes['feature_idx'][nodes['is_leaf'] == 0]:
                name = internal_order[int(idx)]; used.add(name); split_counts[name] += 1
    constant = []
    for j, cat in enumerate(mask):
        if cat:
            is_constant = len(np.unique(X[:, j])) <= 1
        else:
            a = np.asarray(X[:, j], dtype=float)
            is_constant = len(np.unique(a)) <= 1  # np.unique counts NaN as one value.
        if is_constant:
            constant.append(columns[j])
    if set(constant) & used:
        raise DataError('tree_constant_split', 'Constant/all-empty training feature used for a split')
    return {'categories': categories, 'internal_order': internal_order, 'bin_thresholds': thresholds,
            'fit_all_empty_forced_zero_columns': empty_columns,
            'binning_subsample': model._bin_mapper.subsample,
            'binning_random_state': int(model._bin_mapper.random_state),
            'fit_rows': len(X), 'iterations': int(model.n_iter_), 'early_stopping': bool(model.do_early_stopping_),
            'initial_log_odds': float(model._baseline_prediction[0, 0]),
            'constant_input_columns': constant, 'split_counts': split_counts,
            'interpretation': 'split frequency is structural only, not causal importance or additive points'}


def permutation_diagnostic(model, X, y, ids, columns, recipe, checkpoint):
    spec = recipe['explanation']; idx = np.argsort(ids)[:spec['rows']]
    X, y, ids = X[idx].copy(), y[idx], ids[idx]
    original = ranking_metrics(y, predict(model, X))
    rng = np.random.default_rng(spec['seed']); rows = []
    for j, name in enumerate(columns):
        checkpoint(); auc, brier = [], []
        for _ in range(spec['repeats']):
            permuted = X.copy(); permuted[:, j] = X[rng.permutation(len(y)), j]
            m = ranking_metrics(y, predict(model, permuted))
            auc.append(original['roc_auc']-m['roc_auc'])
            brier.append(m['brier_score']-original['brier_score'])
        rows.append({'feature': name, 'auc_drop': auc, 'mean_auc_drop': float(np.mean(auc)),
                     'brier_increase': brier, 'mean_brier_increase': float(np.mean(brier))})
    return {'scope': 'validation_diagnostic_only_no_refit_or_selection', 'rows': len(y),
            'ID_sha256': digest_json(ids.tolist()), 'baseline': original, 'spec': spec, 'features': rows,
            'limitations': 'correlated features can mask importance; 5000 ID-selected rows and two permutations are descriptive; no causal or per-person additive explanation'}


def load_package(root, directory, expected_sha256):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    try:
        if directory == root/'models/tree' or not directory.is_relative_to(root/'models/tree'):
            raise ValueError('formal tree output required')
        if sha256_file(directory/'bundle.json') != expected_sha256:
            raise ValueError('bundle changed')
        b = json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
        if (b['purpose'] != 'formal_tree' or b['version'] != VERSION or b['status'] != 'complete'
                or b['fit_origin'] != 'fresh_training_partition_and_each_fold' or b['environment'] != environment()):
            raise ValueError('invalid purpose, lineage or environment')
        expected = {'model.joblib', 'probe.parquet', 'results.json', 'validation.parquet',
                    'oof.parquet', 'recipe.json', 'registration.json', 'explanation.json'}
        if set(b['files']) != expected:
            raise ValueError('incomplete package')
        for name, info in b['files'].items():
            p = (directory/name).resolve()
            if not p.is_relative_to(directory) or sha256_file(p) != info['sha256'] or p.stat().st_size != info['bytes']:
                raise ValueError('changed file')
        return joblib.load(directory/'model.joblib'), b
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('tree_bundle_invalid', 'Reject smoke, changed, unsafe, incomplete or incompatible tree package') from exc


def check_package(root, directory, expected_sha256):
    model, b = load_package(root, directory, expected_sha256)
    probe = pq.read_table(Path(directory)/'probe.parquet')
    X = matrix(probe.select(b['contract']['feature_columns']), b['contract']['feature_columns'], b['contract'])
    return prediction_agreement(probe['expected_probability'].to_numpy(), predict(model, X))


def load_tree_bundle(root, report_path):
    """E09 consumer: only a full, passed, fingerprint-matching E08 package."""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        if not report_path.is_relative_to(root/'reports'):
            raise ValueError('project report required')
        r = json.loads(report_path.read_text(encoding='utf-8'))
        if r['status'] != 'passed' or r['metadata']['module'] != 'E08' or r['scope'] != 'official_full_tree':
            raise ValueError('passed full E08 required')
        info = r['tree']; model, b = load_package(root, info['artifact_dir'], info['bundle_sha256'])
        plan = json.loads((root/'config/experiment_plan.json').read_text(encoding='utf-8'))
        if (plan['status'] != 'frozen' or plan['plan_payload_sha256'] != digest_json({k: v for k, v in plan.items() if k != 'plan_payload_sha256'})
                or plan['protocol_fingerprint'] != b['recipe']['protocol_fingerprint']):
            raise ValueError('active frozen plan differs')
        if (sha256_file(b['registration_path']) != b['registration_sha256']
                or sha256_file(b['input_report']) != b['input_report_sha256']
                or r['run_id'] != b['run_id'] or info['results'] != json.loads((Path(info['artifact_dir'])/'results.json').read_text(encoding='utf-8'))):
            raise ValueError('training lineage or report changed')
        registration = json.loads(Path(b['registration_path']).read_text(encoding='utf-8'))
        if (registration['fingerprint'] != digest_json(registration['payload'])
                or registration['payload']['recipe'] != b['recipe']
                or any(sha256_file(root/p) != h for p, h in registration['payload']['code_sha256'].items())):
            raise ValueError('registered code or recipe differs')
        # Validate E05 activation and all upstream/output fingerprints, without test table reads.
        train = load_training_partition(root, b['input_report'], 'train')
        if train['contract'] != b['contract']:
            raise ValueError('feature contract differs')
        return {'model': model, 'bundle': b, 'directory': Path(info['artifact_dir']), 'report': r}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('tree_report_invalid', 'Complete matching formal E08 required') from exc


def run_tree(root, input_report, recipe_path, record, progress=None):
    root, input_report, recipe_path = Path(root).resolve(), Path(input_report).resolve(), Path(recipe_path).resolve()
    started = time.monotonic(); recipe = load_recipe(recipe_path)
    train = load_training_partition(root, input_report, 'train')
    contract = train['contract']; columns = contract['feature_columns']
    reg = register(root, recipe, train['protocol'])
    hashes = {p: sha256_file(root/p) for p in CODE}
    paths = [root/'config/experiment_plan.json', recipe_path, reg, input_report]
    path_hashes = {str(p): sha256_file(p) for p in paths}

    def checkpoint():
        if (any(sha256_file(root/p) != h for p, h in hashes.items())
                or any(sha256_file(p) != h for p, h in path_hashes.items())):
            raise DataError('tree_inputs_changed', 'Registered code, recipe or frozen lineage changed')
        peak = process_peak_memory()['peak_working_set_bytes']
        if (time.monotonic()-started > recipe['stopping']['max_total_seconds']
                or (peak is not None and peak > recipe['stopping']['max_parent_peak_bytes'])):
            raise DataError('tree_resource_budget', 'Predeclared time or process peak budget exceeded')

    record.data['metadata'].update(registration_path=str(reg), registration_sha256=sha256_file(reg),
        code_sha256=hashes, environment=environment(), input_fingerprints=path_hashes,
        protocol_fingerprint=recipe['protocol_fingerprint'], first_fit_after_registration=True)
    record.save()
    ids, y, folds = train['ids'].to_numpy(), train['target'].to_numpy(), train['cv_fold'].to_numpy()
    X = matrix(train['features'], columns, contract)
    if len(np.unique(ids)) != len(ids) or set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise DataError('tree_cv_invalid', 'Unique IDs and persisted five folds required')
    candidates, predictions, baseline = [], {}, np.zeros(len(y))
    for candidate in recipe['candidates']:
        oof, details = np.full(len(y), np.nan), []
        for fold in range(1, 6):
            checkpoint(); fit, held = folds != fold, folds == fold; t = time.monotonic()
            model = fit_tree(X[fit], y[fit], contract, candidate, recipe)
            p = predict(model, X[held]); oof[held] = p; baseline[held] = y[fit].mean()
            details.append({'fold': fold, 'fit_rows': int(fit.sum()), 'heldout_rows': int(held.sum()),
                'fit_id_sha256': digest_json(ids[fit].tolist()), 'heldout_id_sha256': digest_json(ids[held].tolist()),
                'state': fit_state(model, X[fit], contract), 'metrics': ranking_metrics(y[held], p),
                'elapsed_seconds': round(time.monotonic()-t, 3)})
            checkpoint()
            if progress: progress({'candidate': candidate['id'], 'fold': fold, 'roc_auc': details[-1]['metrics']['roc_auc']})
        candidates.append({'parameters': candidate, 'mean_cv_auc': float(np.mean([d['metrics']['roc_auc'] for d in details])),
            'mean_cv_brier': float(np.mean([d['metrics']['brier_score'] for d in details])),
            'folds': details, 'oof_metrics': ranking_metrics(y, oof)})
        predictions[candidate['id']] = oof
    chosen = sorted(candidates, key=lambda c: (-c['mean_cv_auc'], c['mean_cv_brier'], c['parameters']['id']))[0]['parameters']
    checkpoint(); model = fit_tree(X, y, contract, chosen, recipe)
    state = fit_state(model, X, contract); checkpoint()
    # Validation first loaded after all parameter selection and final training.
    val = load_training_partition(root, input_report, 'validation')
    vy, vids = val['target'].to_numpy(), val['ids'].to_numpy()
    if set(ids) & set(vids): raise DataError('tree_overlap', 'Train/validation IDs overlap')
    VX = matrix(val['features'], columns, contract); p = predict(model, VX)
    single = prediction_agreement(p[:16], np.array([predict(model, VX[i:i+1])[0] for i in range(min(16, len(vy)))]))
    explanation = permutation_diagnostic(model, VX, vy, vids, columns, recipe, checkpoint)
    results = {'family': recipe['family'], 'purpose': 'validation_not_final_test', 'selected_parameters': chosen,
        'candidates': candidates, 'selection_scope': 'training_cv_only', 'train_rows': len(y), 'validation_rows': len(vy),
        'train_ID_sha256': digest_json(ids.tolist()), 'state': state, 'validation': validation_report(vy, p),
        'baseline_validation': validation_report(vy, np.full(len(vy), y.mean())),
        'baseline_oof': ranking_metrics(y, baseline), 'single_batch': single,
        'limitations': train['protocol']['limitations'], 'calibration': 'none', 'test_evaluated': False}
    destination = root/'models/tree'/record.run_id
    if destination.exists(): raise DataError('tree_exists', 'Immutable tree output exists')
    stage = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial'); stage.mkdir(parents=True)
    try:
        joblib.dump(model, stage/'model.joblib', compress=3)
        results['same_process_reload'] = prediction_agreement(p, predict(joblib.load(stage/'model.joblib'), VX))
        for name, value in [('results.json', results), ('recipe.json', recipe), ('explanation.json', explanation)]:
            write_json(stage/name, value)
        (stage/'registration.json').write_bytes(reg.read_bytes())
        pq.write_table(pa.table({'SK_ID_CURR': vids, 'TARGET': vy, 'probability': p,
            'baseline_probability': np.full(len(vy), y.mean())}), stage/'validation.parquet')
        pq.write_table(pa.table({'SK_ID_CURR': ids, 'TARGET': y, 'CV_FOLD': folds,
            'probability': predictions[chosen['id']], 'baseline_probability': baseline}), stage/'oof.parquet')
        n = min(64, len(vy))
        pq.write_table(val['features'].slice(0, n).append_column('expected_probability', pa.array(p[:n])), stage/'probe.parquet')
        bundle = {'status': 'complete', 'purpose': 'formal_tree', 'fit_origin': 'fresh_training_partition_and_each_fold',
            'version': VERSION, 'run_id': record.run_id, 'environment': environment(), 'recipe': recipe, 'contract': contract,
            'registration_path': str(reg), 'registration_sha256': sha256_file(reg),
            'input_report': str(input_report), 'input_report_sha256': sha256_file(input_report),
            'files': {q.name: {'bytes': q.stat().st_size, 'sha256': sha256_file(q)} for q in stage.iterdir()}}
        write_json(stage/'bundle.json', bundle); digest = sha256_file(stage/'bundle.json'); check_package(root, stage, digest)
        cmd = [sys.executable, '-B', '-X', 'utf8', str(root/'scripts/check_tree_model.py'), '--bundle', str(stage), '--sha256', digest]
        child = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=60)
        if child.returncode: raise DataError('tree_fresh_process', 'Fresh-process check failed: '+child.stdout[-1000:])
        fresh = json.loads(child.stdout)
        if fresh.get('passed') is not True: raise DataError('tree_fresh_process', 'Fresh-process prediction differs')
        checkpoint(); load_training_partition(root, input_report, 'train'); stage.replace(destination)
        return {'artifact_dir': str(destination), 'bundle_sha256': digest, 'results': results, 'fresh_process': fresh}
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to((root/'models/tree').resolve()) or not stage.name.endswith('.partial'):
                raise DataError('unsafe_cleanup', 'Tree staging cleanup escaped root')
            shutil.rmtree(stage)
