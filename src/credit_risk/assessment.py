"""E09: validation-based immutable selection, audited final-test access and comparison.

No fitting, calibration training, feature search or test-based threshold selection.
The original E05 and registered E06-E08 source files remain unchanged.
"""
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from .evaluation import checked_probabilities, ranking_metrics, validation_report, prediction_agreement
from .experiments import digest_json, load_training_partition, verify_experiment_bundle
from .logistic import load_logistic_bundle, matrix, predict as linear_predict, environment
from .scorecard import load_scorecard_bundle
from .tree import load_tree_bundle, predict as tree_predict
from .raw import DataError, sha256_file
from .normalize import write_json
from .runtime import now, process_peak_memory

VERSION = 'assessment_v1'
MODELS = ['logistic', 'scorecard', 'tree', 'main16', 'baseline']
DISCLOSURE = '申请级保留测试；存在已披露的早期环境试跑暴露'
CODE = ['src/credit_risk/assessment.py', 'scripts/run_assessment.py',
        'src/credit_risk/logistic.py', 'src/credit_risk/scorecard.py', 'src/credit_risk/tree.py',
        'src/credit_risk/evaluation.py', 'src/credit_risk/experiments.py', 'src/credit_risk/runtime.py',
        'src/credit_risk/raw.py', 'src/credit_risk/normalize.py', 'scripts/_project.py']
FILES = {'test_predictions.parquet', 'test_probe.parquet', 'metrics.json', 'uncertainty.json', 'curves.json', 'selection.json'}


def load_recipe(path):
    try:
        r = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        expected = {'version': VERSION, 'accepted_versions': {'logistic': 'logistic_v1', 'scorecard': 'scorecard_v2', 'tree': 'tree_v1'},
            'selection': 'validation_auc_with_0.002_tolerance_then_brier_then_simplicity',
            'auc_tolerance': .002, 'simplicity_order': ['logistic', 'scorecard', 'tree'],
            'calibration': 'none_for_first_comparison', 'threshold': 'max_validation_ks_tie_highest_finite_threshold',
            'validation_threshold_grid': [.05, .1, .2],
            'uncertainty': {'method': 'paired_stratified_ID_bootstrap_percentile', 'replicates': 500,
                            'confidence': .95, 'validation_seed': 4242, 'test_seed': 4243},
            'test_scope': 'original_full_reserved_partition_no_subsets',
            'history': {'prior_fit_overlap': 626, 'prior_probe_overlap': 143, 'total_exposed': 769, 'expected_test_rows': 46127},
            'no_refit': True}
        if ({k: v for k, v in r.items() if k not in ['experiment_report', 'model_reports', 'protocol_fingerprint']} != expected
                or set(r['model_reports']) != {'logistic', 'scorecard', 'tree'} or len(r['protocol_fingerprint']) != 64):
            raise ValueError('unsupported assessment recipe')
        return r
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DataError('assessment_recipe_invalid', 'Fixed E09 recipe required') from exc


def choose_model(metrics, recipe):
    if set(metrics) != {'logistic', 'scorecard', 'tree'}:
        raise DataError('assessment_families', 'All three accepted families required')
    best = max(m['roc_auc'] for m in metrics.values())
    if any(not np.isfinite(m['roc_auc']) or not np.isfinite(m['brier_score']) for m in metrics.values()):
        raise DataError('assessment_metric_invalid', 'Finite validation metrics required')
    eligible = [n for n, m in metrics.items() if m['roc_auc'] >= best-recipe['auc_tolerance']]
    chosen = min(eligible, key=lambda n: (metrics[n]['brier_score'], recipe['simplicity_order'].index(n)))
    return {'recommended': chosen, 'eligible': eligible, 'best_validation_auc': best,
            'rule': recipe['selection'], 'auc_tolerance_is_not_significance': True}


def fixed_threshold_diagnostics(y, p, threshold):
    y, p = checked_probabilities(y, p)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DataError('assessment_threshold', 'Finite prelocked threshold in [0,1] required')
    positive = p >= threshold; tp = int(y[positive].sum()); fp = int(positive.sum())-tp
    fn = int(y.sum())-tp; tn = len(y)-tp-fp-fn
    return {'value': float(threshold), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn),
            'false_positive_rate': fp/(fp+tn), 'selected_fraction': float(positive.mean()),
            'scope': 'locked_validation_threshold_applied_without_reselection',
            'meaning': 'classroom risk flag only; no lending approval or economic optimality claim'}


def final_diagnostics(y, p, threshold):
    result = ranking_metrics(y, p)
    # Calibration values from the official API; no calibration model is fitted.
    true, predicted = calibration_curve(y, p, n_bins=10, strategy='uniform')
    bins = np.searchsorted(np.linspace(0, 1, 11)[1:-1], p)
    nonempty = iter(zip(true.tolist(), predicted.tolist())); rows = []
    for i in range(10):
        count = int((bins == i).sum())
        event, prob = next(nonempty) if count else (None, None)
        rows.append({'bin': i, 'rows': count, 'mean_prediction': prob, 'event_fraction': event})
    result.update(threshold=fixed_threshold_diagnostics(y, p, threshold), calibration_bins=rows)
    return result


def paired_bootstrap(y, probabilities, seed, replicates=500, progress=None):
    """Paired within-class ID resampling; conditional on fitted models/class counts.

    These are marginal descriptive intervals, not selection-adjusted or evidence
    of person-disjoint/time-out generalization. No p-value or test-based choice.
    """
    if type(replicates) is not int or replicates < 2: raise DataError('assessment_bootstrap', 'At least two replicates required')
    names = list(probabilities)
    for p in probabilities.values(): checked_probabilities(y, p)
    rng = np.random.default_rng(seed); groups = [np.flatnonzero(y == v) for v in [0, 1]]
    draws = {n: {'auc': [], 'brier': []} for n in names}
    for i in range(replicates):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]); sy = y[idx]
        for n in names:
            p = probabilities[n][idx]
            draws[n]['auc'].append(float(roc_auc_score(sy, p)))
            draws[n]['brier'].append(float(np.mean((p-sy)**2)))
        if progress and (i+1) % 100 == 0: progress({'bootstrap_seed': seed, 'replicate': i+1})
    intervals = {n: {k: np.quantile(v, [.025, .975]).tolist() for k, v in d.items()} for n, d in draws.items()}
    contrasts = {}
    for left, right in [('tree', 'logistic'), ('scorecard', 'logistic'), ('logistic', 'main16'), ('tree', 'baseline')]:
        if left in draws and right in draws:
            contrasts[left+'_minus_'+right] = {k: np.quantile(np.subtract(draws[left][k], draws[right][k]), [.025, .975]).tolist() for k in ['auc', 'brier']}
    return {'method': 'paired_stratified_ID_bootstrap_percentile', 'replicates': replicates, 'seed': seed,
            'confidence': .95, 'intervals': intervals, 'paired_difference_intervals': contrasts,
            'limitations': 'fixed event counts; conditional on fitted models; ID independence not verified at person level; not selection-adjusted, simultaneous or causal; 500-replicate endpoints have Monte Carlo variation'}


def _project_path(root, value):
    p = (root/value).resolve()
    if not p.is_relative_to(root): raise DataError('assessment_path', 'Input must stay within project')
    return p


def _active_experiment(root, report):
    loaded = verify_experiment_bundle(root, report)
    plan_path = root/'config/experiment_plan.json'; plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if (plan['status'] != 'frozen' or plan['plan_payload_sha256'] != digest_json({k: v for k, v in plan.items() if k != 'plan_payload_sha256'})
            or plan['protocol_fingerprint'] != loaded['bundle']['protocol_fingerprint']
            or Path(plan['active_report_path']).resolve() != Path(report).resolve()
            or plan['active_report_sha256'] != sha256_file(report)
            or plan['active_bundle_sha256'] != sha256_file(loaded['directory']/'bundle.json')):
        raise DataError('assessment_protocol', 'Exact active E05 report and frozen plan required')
    return loaded


def _predictions(models, features, contract, baseline):
    full = matrix(features, contract['feature_columns'], contract)
    return {'logistic': linear_predict(models['logistic']['models']['full40'], full),
            'scorecard': linear_predict(models['scorecard']['model'], full),
            'tree': tree_predict(models['tree']['model'], full),
            'main16': linear_predict(models['logistic']['models']['main16'], matrix(features, contract['main_features'], contract)),
            'baseline': np.full(features.num_rows, baseline)}


def selection_payload(root, recipe):
    """Only train/validation reads: creates a deterministic reviewable lock payload."""
    root = Path(root).resolve(); source = _project_path(root, recipe['experiment_report'])
    exp = _active_experiment(root, source)
    if digest_json(exp['protocol']) != recipe['protocol_fingerprint']:
        raise DataError('assessment_protocol', 'Recipe and active protocol differ')
    models = {n: loader(root, _project_path(root, recipe['model_reports'][n])) for n, loader in
              [('logistic', load_logistic_bundle), ('scorecard', load_scorecard_bundle), ('tree', load_tree_bundle)]}
    tr = load_training_partition(root, source, 'train'); va = load_training_partition(root, source, 'validation')
    contract = tr['contract']; baseline = float(tr['target'].to_numpy().mean()); refs = {}
    for name, obj in models.items():
        b = obj['bundle']; path = _project_path(root, recipe['model_reports'][name])
        if b['version'] != recipe['accepted_versions'][name] or b['contract'] != contract or Path(b['input_report']).resolve() != source:
            raise DataError('assessment_lineage', 'Accepted versions and identical E05 input/contract required')
        reg = json.loads(Path(b['registration_path']).read_text(encoding='utf-8'))
        fingerprint = reg.get('fingerprint', reg.get('payload_fingerprint'))
        if (fingerprint != digest_json(reg['payload']) or any(sha256_file(root/p) != h for p, h in reg['payload']['code_sha256'].items())):
            raise DataError('assessment_source_changed', 'Registered family source changed')
        disk_results = json.loads((obj['directory']/'results.json').read_text(encoding='utf-8'))
        report_result = obj['report'][{'logistic': 'logistic', 'scorecard': 'scorecard', 'tree': 'tree'}[name]]['results']
        if disk_results != report_result: raise DataError('assessment_report_changed', 'Model report and signed package results differ')
        refs[name] = {'report_path': str(path), 'report_sha256': sha256_file(path),
            'bundle_path': str(obj['directory']/'bundle.json'), 'bundle_sha256': sha256_file(obj['directory']/'bundle.json'),
            'registration_path': b['registration_path'], 'registration_sha256': sha256_file(b['registration_path']),
            'training_seconds': obj['report']['elapsed_seconds'], 'training_peak_bytes': obj['report']['peak_working_set_bytes'],
            'accepted_version': b['version']}
    ids, y = va['ids'].to_numpy(), va['target'].to_numpy(); probabilities = _predictions(models, va['features'], contract, baseline)
    for name in ['logistic', 'scorecard', 'tree', 'main16']:
        owner = models['logistic' if name == 'main16' else name]
        file = ('validation_'+('main16' if name == 'main16' else 'full40')+'.parquet') if name in ['main16', 'logistic'] else 'validation.parquet'
        saved = pq.read_table(owner['directory']/file)
        if not np.array_equal(saved['SK_ID_CURR'].to_numpy(), ids) or not np.array_equal(saved['TARGET'].to_numpy(), y):
            raise DataError('assessment_validation_rows', 'Same validation ID order and labels required')
        prediction_agreement(saved['probability'].to_numpy(), probabilities[name])
    metrics = {n: validation_report(y, p) for n, p in probabilities.items()}
    decision = choose_model({n: metrics[n] for n in ['logistic', 'scorecard', 'tree']}, recipe)
    payload = {'version': VERSION, 'purpose': 'formal_validation_selection_lock', 'recipe': recipe,
        'protocol_fingerprint': recipe['protocol_fingerprint'], 'plan_sha256': sha256_file(root/'config/experiment_plan.json'),
        'environment': environment(), 'code_sha256': {p: sha256_file(root/p) for p in CODE}, 'models': refs,
        'experiment_report': str(source), 'experiment_report_sha256': sha256_file(source),
        'validation_ID_sha256': digest_json(ids.tolist()), 'baseline_training_probability': baseline,
        'validation_metrics': metrics, 'decision': decision,
        'thresholds': {n: m['threshold']['value'] for n, m in metrics.items()},
        'validation_threshold_sensitivity': {n: [fixed_threshold_diagnostics(y, p, t) for t in recipe['validation_threshold_grid']] for n, p in probabilities.items()},
        'limitations': exp['protocol']['limitations'], 'test_disclosure': DISCLOSURE,
        'prior_environment_exposure': recipe['history'], 'test_used_for_selection': False,
        'post_test_model_or_threshold_changes_allowed': False}
    return payload, {'models': models, 'experiment': exp, 'validation': va, 'probabilities': probabilities}


def selection_directory(root, fingerprint):
    return Path(root).resolve()/'reports/selection/official'/fingerprint


def prepare_selection(root, recipe_path):
    root = Path(root).resolve(); recipe = load_recipe(recipe_path)
    payload, _ = selection_payload(root, recipe)
    folder = selection_directory(root, recipe['protocol_fingerprint']); folder.mkdir(parents=True, exist_ok=True)
    path = folder/'selection.json'; mutex = folder/'selection_write.lock'; token = uuid.uuid4().hex
    try:
        with mutex.open('x', encoding='utf-8') as f: f.write(token)
    except FileExistsError as exc: raise DataError('assessment_selection_busy', 'Selection writer lock exists; inspect interrupted/concurrent run') from exc
    temporary = folder/(token+'.partial')
    try:
        fingerprint = digest_json(payload)
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['payload'] != payload or old['fingerprint'] != fingerprint:
                raise DataError('assessment_selection_conflict', 'Protocol already has a different selection; cannot silently replace after exposure')
        else:
            write_json(temporary, {'locked_at': now(), 'fingerprint': fingerprint, 'payload': payload})
            temporary.replace(path)
        return {'selection_path': str(path), 'selection_sha256': sha256_file(path), 'recommended': payload['decision']['recommended'], 'thresholds': payload['thresholds']}
    finally:
        if temporary.exists(): temporary.unlink()
        if mutex.exists() and mutex.read_text(encoding='utf-8') == token: mutex.unlink()


def verify_selection(root, path, expected_sha256):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        lock = json.loads(path.read_text(encoding='utf-8')); p = lock['payload']
        if (sha256_file(path) != expected_sha256 or lock['fingerprint'] != digest_json(p)
                or path != selection_directory(root, p['protocol_fingerprint'])/'selection.json'
                or p['purpose'] != 'formal_validation_selection_lock' or p['version'] != VERSION
                or p['test_used_for_selection'] is not False or p['post_test_model_or_threshold_changes_allowed'] is not False
                or set(p['code_sha256']) != set(CODE)
                or p['recipe'] != load_recipe(root/'config/assessment_recipe.json')
                or p['environment'] != environment() or p['plan_sha256'] != sha256_file(root/'config/experiment_plan.json')
                or any(sha256_file(root/name) != h for name, h in p['code_sha256'].items())):
            raise ValueError('selection, plan, code, purpose or environment changed')
        # Reconstruct using validation alone: no user-supplied handpicked winner/threshold.
        current, context = selection_payload(root, p['recipe'])
        if current != p: raise ValueError('locked model lineage or validation decision differs')
        return lock, context
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DataError('assessment_selection_invalid', 'Complete immutable matching validation selection required') from exc


def load_reserved_test(root, path, expected_sha256, run_id):
    """Only E09 explicit locked entry; durable access intent before any test table read.

    The training API still refuses test. Hashes are local integrity checks, not
    permissions preventing a human or arbitrary Python from opening local files.
    """
    root = Path(root).resolve(); lock, context = verify_selection(root, path, expected_sha256)
    folder = Path(path).parent
    if (folder/'completed.json').exists() or (folder/'test_access.jsonl').exists():
        raise DataError('assessment_previous_test_access', 'Test was already accessed; consume success output or inspect interrupted attempt, never silently re-expose')
    if not isinstance(run_id, str) or not run_id or not (root/'reports/runs'/run_id/'run.json').is_file():
        raise DataError('assessment_access_run', 'Persistent E09 running record required')
    record = json.loads((root/'reports/runs'/run_id/'run.json').read_text(encoding='utf-8'))
    if record.get('status') != 'running' or record.get('scope') != 'official_final_assessment' or record.get('metadata', {}).get('module') != 'E09':
        raise DataError('assessment_access_run', 'Only a running formal E09 evaluation may read test')
    event = {'at': now(), 'run_id': run_id, 'event': 'test_table_read_intent', 'selection_sha256': expected_sha256,
             'selection_fingerprint': lock['fingerprint'], 'disclosure': DISCLOSURE}
    ledger = folder/'test_access.jsonl'
    with ledger.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
    table = pq.read_table(context['experiment']['directory']/'test.parquet')
    c = context['experiment']['contract']; quality = json.loads((context['experiment']['directory']/'quality.json').read_text(encoding='utf-8'))['partitions']['test']
    ids = table['SK_ID_CURR'].to_numpy(); y = table['TARGET'].to_numpy()
    if (len(ids) != quality['rows'] or len(np.unique(ids)) != len(ids) or int(y.sum()) != quality['TARGET_1']
            or digest_json(ids.tolist()) != quality['id_sha256'] or set(np.unique(y)) != {0, 1}):
        raise DataError('assessment_test_population', 'Full original reserved population required')
    manifest = pq.read_table(context['experiment']['directory']/'split_manifest.parquet')
    mid = manifest['SK_ID_CURR'].to_numpy(); partitions = np.asarray(manifest['partition'].to_pylist())
    if set(ids) != set(mid[partitions == 'test']): raise DataError('assessment_test_manifest', 'Reserved ID manifest differs')
    return {'features': table.select(c['feature_columns']), 'target': y, 'ids': ids, 'contract': c}, lock, context


def run_final_assessment(root, path, expected_sha256, record, progress=None):
    started = time.monotonic(); root = Path(root).resolve(); lock, context = verify_selection(root, path, expected_sha256)
    folder = Path(path).parent; mutex = folder/'evaluation.lock'; token = uuid.uuid4().hex
    try:
        with mutex.open('x', encoding='utf-8') as f: f.write(token)
    except FileExistsError as exc: raise DataError('assessment_busy', 'Concurrent/interrupted evaluation lock requires inspection') from exc
    destination = root/'reports/assessments/full'/record.run_id
    stage = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        if (folder/'completed.json').exists(): raise DataError('assessment_already_completed', 'Final evaluation already published; consume saved report rather than re-expose test')
        if destination.exists(): raise DataError('assessment_exists', 'Immutable output exists')
        previous = (folder/'test_access.jsonl').exists()
        if previous: raise DataError('assessment_previous_test_access', 'A prior access intent exists; reconcile saved outputs/failure before any further exposure')
        record.data['metadata'].update(selection_path=str(path), selection_sha256=expected_sha256,
            repeated_access_after_prior_attempt=previous, environment=environment(), decision=lock['payload']['decision'])
        record.save()
        # Finish validation diagnostics/uncertainty before opening final test.
        p = lock['payload']; va = context['validation']; spec = p['recipe']['uncertainty']
        val_uncertainty = paired_bootstrap(va['target'].to_numpy(), context['probabilities'], spec['validation_seed'], spec['replicates'], progress)
        reserved, lock, context = load_reserved_test(root, path, expected_sha256, record.run_id)
        if progress: progress({'event': 'locked_final_test_loaded', 'rows': len(reserved['ids'])})
        probabilities = _predictions(context['models'], reserved['features'], reserved['contract'], p['baseline_training_probability'])
        y = reserved['target']; metrics = {n: final_diagnostics(y, probabilities[n], p['thresholds'][n]) for n in MODELS}
        test_uncertainty = paired_bootstrap(y, probabilities, spec['test_seed'], spec['replicates'], progress)
        summary = {'purpose': 'locked_final_evaluation_not_model_selection', 'recommended': p['decision']['recommended'],
            'decision_fixed_before_test': p['decision'], 'thresholds_fixed_before_test': p['thresholds'],
            'validation': p['validation_metrics'], 'test': metrics,
            'history_comparison': {scope: {'auc_full40_minus_main16': m['logistic']['roc_auc']-m['main16']['roc_auc'],
                'brier_full40_minus_main16': m['logistic']['brier_score']-m['main16']['brier_score'],
                'meaning': 'same IDs, same CV budget, separately chosen C; descriptive feature-set comparison, not causal'}
                for scope, m in [('validation', p['validation_metrics']), ('test', metrics)]},
            'test_rows': len(y), 'test_events': int(y.sum()), 'test_ID_sha256': digest_json(reserved['ids'].tolist()),
            'test_disclosure': DISCLOSURE, 'prior_environment_exposure': p['prior_environment_exposure'],
            'limitations': p['limitations'], 'calibration_fit': False, 'refit': False,
            'post_test_selection_performed': False, 'repeated_access_after_prior_attempt': previous,
            'resource_sources': {n: {k: v for k, v in ref.items() if k.startswith('training_')} for n, ref in p['models'].items()}}
        curves = {}
        for scope, labels, probs in [('validation', va['target'].to_numpy(), context['probabilities']), ('test', y, probabilities)]:
            curves[scope] = {}
            for n, probability in probs.items():
                fpr, tpr, _ = roc_curve(labels, probability)
                curves[scope][n] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                    'calibration_bins': (p['validation_metrics'] if scope == 'validation' else metrics)[n]['calibration_bins']}
        stage.mkdir(parents=True)
        write_json(stage/'metrics.json', summary); write_json(stage/'uncertainty.json', {'validation': val_uncertainty, 'test': test_uncertainty})
        write_json(stage/'curves.json', curves); (stage/'selection.json').write_bytes(Path(path).read_bytes())
        pq.write_table(pa.table({'SK_ID_CURR': reserved['ids'], 'TARGET': y, **{n+'_probability': v for n, v in probabilities.items()}}), stage/'test_predictions.parquet')
        count = min(64, len(y)); probe = reserved['features'].slice(0, count).append_column('SK_ID_CURR', pa.array(reserved['ids'][:count]))
        for n, values in probabilities.items(): probe = probe.append_column(n+'_probability', pa.array(values[:count]))
        pq.write_table(probe, stage/'test_probe.parquet')
        bundle = {'status': 'complete', 'module': 'E09', 'version': VERSION, 'purpose': 'formal_locked_final_assessment',
            'run_id': record.run_id, 'environment': environment(), 'selection_path': str(Path(path).resolve()),
            'selection_sha256': expected_sha256, 'selection_fingerprint': lock['fingerprint'],
            'files': {q.name: {'bytes': q.stat().st_size, 'sha256': sha256_file(q)} for q in stage.iterdir()}}
        write_json(stage/'bundle.json', bundle); digest = sha256_file(stage/'bundle.json')
        # Validate complete written outputs; no new model fitting or alternative selection.
        saved = pq.read_table(stage/'test_predictions.parquet')
        for n in MODELS: prediction_agreement(probabilities[n], saved[n+'_probability'].to_numpy())
        verify_selection(root, path, expected_sha256); stage.replace(destination)
        result = {'artifact_dir': str(destination), 'bundle_sha256': digest, 'recommended': summary['recommended'],
                  'test_metrics': metrics, 'selection_sha256': expected_sha256}
        # Keep the mutex until the stable passed report and completion marker exist.
        # A crash leaves the durable access intent, which blocks silent re-exposure.
        record.finish(True, assessment=result, elapsed_seconds=round(time.monotonic()-started, 3), **process_peak_memory())
        mark_complete(root, record)
        return result
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to((root/'reports/assessments/full').resolve()) or not stage.name.endswith('.partial'):
                raise DataError('unsafe_cleanup', 'Assessment cleanup escaped output root')
            shutil.rmtree(stage)
        if mutex.exists() and mutex.read_text(encoding='utf-8') == token: mutex.unlink()


def mark_complete(root, record):
    r = record.data
    if r['status'] != 'passed': raise DataError('assessment_completion', 'Stable passed report required')
    path = Path(r['metadata']['selection_path']).parent/'completed.json'
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps({'completed_at': now(), 'report': str(record.path), 'report_sha256': sha256_file(record.path),
                            'bundle_sha256': r['assessment']['bundle_sha256']}, ensure_ascii=False, indent=2)+'\n')


def load_assessment_bundle(root, report_path):
    """E10 consumer of a successful immutable E09; never reads test feature table."""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = json.loads(report_path.read_text(encoding='utf-8')); info = r['assessment']
        if (not report_path.is_relative_to(root/'reports') or r['status'] != 'passed'
                or r['scope'] != 'official_final_assessment' or r['metadata']['module'] != 'E09'):
            raise ValueError('full passed E09 required')
        folder = Path(info['artifact_dir']).resolve()
        if not folder.is_relative_to(root/'reports/assessments/full') or sha256_file(folder/'bundle.json') != info['bundle_sha256']:
            raise ValueError('artifact path or hash differs')
        b = json.loads((folder/'bundle.json').read_text(encoding='utf-8'))
        if (b['status'] != 'complete' or b['purpose'] != 'formal_locked_final_assessment' or b['version'] != VERSION
                or b['run_id'] != r['run_id'] or b['environment'] != environment() or set(b['files']) != FILES):
            raise ValueError('incomplete, incompatible or wrong-purpose output')
        for name, meta in b['files'].items():
            q = (folder/name).resolve()
            if not q.is_relative_to(folder) or q.stat().st_size != meta['bytes'] or sha256_file(q) != meta['sha256']: raise ValueError('changed output')
        lock, _ = verify_selection(root, b['selection_path'], b['selection_sha256'])
        summary = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if (lock['fingerprint'] != b['selection_fingerprint'] or summary['recommended'] != lock['payload']['decision']['recommended']
                or info['recommended'] != summary['recommended'] or info['test_metrics'] != summary['test']
                or sha256_file(folder/'selection.json') != b['selection_sha256']): raise ValueError('decision or report differs')
        completion = json.loads((Path(b['selection_path']).parent/'completed.json').read_text(encoding='utf-8'))
        if Path(completion['report']).resolve() != report_path or completion['report_sha256'] != sha256_file(report_path): raise ValueError('completion differs')
        return {'report': r, 'bundle': b, 'directory': folder, 'summary': summary, 'selection': lock}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('assessment_bundle_invalid', 'Matching complete final-assessment success package required') from exc
