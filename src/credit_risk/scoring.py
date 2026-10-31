"""E10: one frozen-model core for existing-ID single, batch and future page use.

The local package is a verified reference bundle: fitted pipelines remain in their
immutable E06/E07/E08 packages. No fit, test evaluation, or free-form applicant API.
"""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .assessment import load_assessment_bundle, DISCLOSURE
from .experiments import verify_experiment_bundle, verify_feature_values, digest_json
from .history import load_history_bundle
from .logistic import load_logistic_bundle, matrix, predict as linear_predict, environment
from .scorecard import load_scorecard_bundle, points, point_table
from .tree import load_tree_bundle, predict as tree_predict
from .raw import DataError, sha256_file
from .normalize import write_json
from .runtime import now, process_peak_memory

VERSION = 'scoring_v1'
FILES = {'features.parquet', 'scores.parquet', 'components.parquet', 'case.json',
         'feature_contract.json', 'point_table.json', 'recipe.json', 'checks.json'}
CODE = ['src/credit_risk/scoring.py', 'scripts/run_scoring.py']


def load_recipe(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (set(value) != {'version','assessment_report','scope','source','batch_size','tolerance','case_id','repeat','unlabeled_application_test'}
                or value['version'] != VERSION or value['scope'] != 'existing_labeled_applications_replay'
                or value['source'] != 'E04_joined_projection_E05_frozen_40'
                or type(value['batch_size']) is not int or not 1 <= value['batch_size'] <= 50000
                or value['tolerance'] != 1e-10 or type(value['case_id']) is not int
                or value['repeat'] != 'new_immutable_run_same_input_same_values'
                or value['unlabeled_application_test'] != 'unsupported_pending_upstream_contract'):
            raise ValueError('unsupported recipe')
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('scoring_recipe_invalid', 'Explicit supported scoring recipe required') from exc


def validate_ids(ids):
    if not isinstance(ids, (list, tuple)) or not ids:
        raise DataError('scoring_ids', 'Nonempty list of integer application IDs required')
    if any(type(x) is not int or x <= 0 or x > 2**63-1 for x in ids):
        raise DataError('scoring_id_type', 'IDs must be positive int64 integers, not strings, floats or booleans')
    if len(set(ids)) != len(ids):
        raise DataError('scoring_duplicate_id', 'Duplicate application IDs rejected; no silent deduplication')
    return ids


def validate_features(features, contract):
    if not isinstance(features, pa.Table) or features.column_names != contract['feature_columns']:
        raise DataError('scoring_fields', 'Exact ordered frozen fields required; missing, extra, TARGET and audit fields rejected')
    if features.num_rows == 0:
        raise DataError('scoring_empty', 'Empty feature input rejected')
    verify_feature_values(features, contract)
    # Null is permitted by the upstream contract; NaN/inf are not alternative null encodings.
    # History structural zero/null semantics come from E04, never from filling absent fields.
    for prefix in ('BUREAU', 'PREV'):
        count, flag = prefix+'_ELIGIBLE_COUNT', prefix+'_HAS_ELIGIBLE_HISTORY'
        if count in features.column_names:
            c, f = features[count], features[flag]
            if c.null_count or f.null_count or pc.any(pc.less(c, 0)).as_py() or not pc.all(pc.equal(f, pc.cast(pc.greater(c, 0), pa.int64()))).as_py():
                raise DataError('scoring_history', 'History counts and presence flag must agree; absent history is not a missing field')


def _models(root, selection):
    payload = selection['payload']
    loaders = {'tree': load_tree_bundle, 'scorecard': load_scorecard_bundle, 'logistic': load_logistic_bundle}
    return {n: loader(root, payload['models'][n]['report_path']) for n, loader in loaders.items()}


def _verify_references(bundle):
    references = [(bundle['assessment_report'],bundle['assessment_report_sha256']),
                  (bundle['history_report'],bundle['history_report_sha256']),
                  (bundle['experiment_report'],bundle['experiment_report_sha256'])]
    for ref in bundle['model_references'].values():
        references.extend((ref[k+'_path'],ref[k+'_sha256']) for k in ['report','bundle','registration'])
    if any(sha256_file(path) != expected for path,expected in references):
        raise DataError('scoring_source_changed','Immutable upstream reference changed during scoring')


class ScoringSession:
    """A loaded immutable snapshot; UI calls score_ids()/case(), never fits models.

    _score_features is an internal checked adapter for validation fixtures. Public
    callers cannot submit guessed features for an existing ID.
    """
    def __init__(self, root, bundle, features, models, directory=None, report=None):
        self.root, self.bundle, self.features, self.models = Path(root), bundle, features, models
        self.directory, self.report = directory, report
        self.contract = bundle['contract']
        self.index = {x: i for i, x in enumerate(features['SK_ID_CURR'].to_pylist())}

    def _score_features(self, features):
        validate_features(features, self.contract)
        X = matrix(features, self.contract['feature_columns'], self.contract)
        tree = tree_predict(self.models['tree']['model'], X)
        card = linear_predict(self.models['scorecard']['model'], X)
        lr = linear_predict(self.models['logistic']['models']['full40'], X)
        base, components, score = points(self.models['scorecard']['model'], X, self.models['scorecard']['bundle']['recipe'])
        # This exact additive explanation belongs to the scorecard, not the tree.
        return {'risk_probability': tree, 'risk_flag': tree >= self.bundle['threshold'],
                'scorecard_probability': card, 'scorecard_score': score,
                'scorecard_base_points': np.full(len(tree), base), 'logistic_probability': lr}, components

    def _score_rows(self, rows):
        values, components = self._score_features(rows.select(self.contract['feature_columns']))
        count = rows.num_rows
        metadata = {k: [v]*count for k, v in self.bundle['output_versions'].items()}
        data = {'SK_ID_CURR': rows['SK_ID_CURR'], 'sample_use': rows['sample_use'], **values,
                'threshold': np.full(count, self.bundle['threshold']), **metadata}
        details = pa.table({'SK_ID_CURR': rows['SK_ID_CURR'],
                           **{name: components[:, j] for j, name in enumerate(self.contract['feature_columns'])}})
        return pa.table(data), details

    def score_ids(self, ids):
        ids = validate_ids(ids)
        if any(x not in self.index for x in ids):
            raise DataError('scoring_unknown_id', 'ID absent from supported existing labeled population; application_test is not yet supported')
        return self._score_rows(self.features.take(pa.array([self.index[x] for x in ids], type=pa.int64())))

    def case(self, application_id):
        scores, details = self.score_ids([application_id])
        row = self.features.slice(self.index[application_id], 1)
        values = row.select(self.contract['feature_columns']).to_pylist()[0]
        X = matrix(row.select(self.contract['feature_columns']), self.contract['feature_columns'], self.contract)
        woe = self.models['scorecard']['model'].named_steps['woe']
        terms = []
        for j, rule in enumerate(woe.rules_):
            group = str(woe._groups(X, j, rule)[0])
            terms.append({'feature': rule['name'], 'value': values[rule['name']], 'group': group,
                          'woe': rule['bins'].get(group, {'woe': 0.})['woe'],
                          'points': details[rule['name']][0].as_py(),
                          'unobserved_group': group not in rule['bins']})
        return {'result': scores.to_pylist()[0], 'features': values, 'scorecard_terms': terms,
                'history': {p: ('no_eligible_history' if values[p+'_ELIGIBLE_COUNT'] == 0 else 'eligible_history_present') for p in ['BUREAU','PREV']},
                'meaning': 'Tree probability and original validation risk flag; scorecard terms explain only its own score/probability. No personal additive tree contribution.',
                'sample_notice': 'Training example is a workflow demonstration, not independent performance evidence.' if row['sample_use'][0].as_py() == 'train' else DISCLOSURE,
                'limitations': self.bundle['limitations']}


def prepare_session(root, recipe):
    root = Path(root).resolve()
    assessment_path = (root/recipe['assessment_report']).resolve()
    # Mandatory complete E09 consumption; does not reread original test features.
    assessment = load_assessment_bundle(root, assessment_path)
    selection = assessment['selection']; p = selection['payload']
    if p['decision']['recommended'] != 'tree':
        raise DataError('scoring_recommendation', 'This version requires the accepted frozen tree recommendation')
    experiment = verify_experiment_bundle(root, p['experiment_report'])
    history_path = Path(experiment['bundle']['input_run_path'])
    history = load_history_bundle(root, history_path)
    contract = experiment['contract']; models = _models(root, selection)
    if any(m['bundle']['contract'] != contract for m in models.values()):
        raise DataError('scoring_model_contract', 'All fitted model contracts must match')
    # Projection pushes TARGET exclusion down to Parquet. No test.parquet read.
    projected = pq.read_table(history['joined_features'], columns=['SK_ID_CURR', *contract['feature_columns']])
    validate_features(projected.select(contract['feature_columns']), contract)
    ids = projected['SK_ID_CURR'].to_pylist()
    validate_ids(ids)
    split = pq.read_table(experiment['directory']/'split_manifest.parquet', columns=['SK_ID_CURR','partition'])
    if split.num_rows != projected.num_rows or len(set(split['SK_ID_CURR'].to_pylist())) != split.num_rows:
        raise DataError('scoring_population', 'Unique complete frozen ID population required')
    positions = pc.index_in(projected['SK_ID_CURR'], value_set=split['SK_ID_CURR'])
    if positions.null_count:
        raise DataError('scoring_population', 'E04 and E05 IDs differ')
    uses = split['partition'].take(positions)
    if set(uses.to_pylist()) != {'train','validation','test'}:
        raise DataError('scoring_partition', 'Frozen split purposes required')
    features = projected.append_column('sample_use', uses)
    protocol = experiment['protocol']
    versions = {'dataset_version': protocol['dataset_version'], 'feature_version': contract['version'],
        'feature_contract_sha256': protocol['feature_contract_sha256'], 'protocol_fingerprint': p['protocol_fingerprint'],
        'split_manifest_sha256': protocol['split_manifest_sha256'], 'recommended_model': 'tree',
        'tree_version': models['tree']['bundle']['version'], 'tree_run_id': models['tree']['report']['run_id'],
        'scorecard_version': models['scorecard']['bundle']['version'], 'scorecard_run_id': models['scorecard']['report']['run_id'],
        'logistic_version': models['logistic']['bundle']['version'], 'logistic_run_id': models['logistic']['report']['run_id'],
        'scoring_version': VERSION, 'assessment_run_id': assessment['report']['run_id'],
        'selection_sha256': assessment['bundle']['selection_sha256'], 'scoring_scope': recipe['scope']}
    bundle = {'version': VERSION, 'status': 'complete', 'purpose': 'formal_existing_application_scoring',
        'contract': contract, 'recipe': recipe, 'environment': environment(), 'output_versions': versions,
        'threshold': p['thresholds']['tree'], 'threshold_comparison': 'probability >= threshold',
        'model_references': p['models'], 'assessment_report': str(assessment_path),
        'assessment_report_sha256': sha256_file(assessment_path), 'assessment_bundle_sha256': sha256_file(assessment['directory']/'bundle.json'),
        'selection_sha256': assessment['bundle']['selection_sha256'],
        'history_report': str(history_path), 'history_report_sha256': sha256_file(history_path),
        'history_bundle_sha256': sha256_file(history['directory']/'bundle.json'),
        'feature_source_sha256': sha256_file(history['joined_features']),
        'experiment_report': p['experiment_report'], 'experiment_report_sha256': sha256_file(p['experiment_report']),
        'code_sha256': {name: sha256_file(root/name) for name in CODE}, 'rows': len(ids),
        'limitations': {**p['limitations'], 'test_disclosure': DISCLOSURE, 'prior_exposed_test_rows': 769,
            'scorecard_explanation': 'additive scorecard terms do not explain or convert tree probability',
            'use': 'local skills learning; license not verified; classroom flag is not an approval decision',
            'unlabeled_application_test': recipe['unlabeled_application_test']}}
    return ScoringSession(root, bundle, features, models)


def _publish(partial, final):
    if final.exists():
        raise DataError('scoring_exists', 'Never overwrite a successful output directory')
    partial.rename(final)


def run_scoring(root, recipe_path, record, progress=None):
    root = Path(root).resolve(); start = time.monotonic()
    partial = root/'reports/scoring/full'/('.'+record.run_id+'.partial')
    final = partial.parent/record.run_id
    promoted = False
    def stage(name):
        record.data['stage'] = name; record.save()
        if progress: progress({'stage': name})
    try:
        stage('verify_E09_and_project_E04')
        recipe = load_recipe(recipe_path); recipe_hash = sha256_file(recipe_path)
        session = prepare_session(root, recipe)
        b = session.bundle
        b.update(run_id=record.run_id, created_at=now(), recipe_sha256=recipe_hash,
                 input_fingerprint=digest_json({k: b[k] for k in ['assessment_report_sha256','selection_sha256','feature_source_sha256','history_bundle_sha256','output_versions','code_sha256']}))
        partial.mkdir(parents=True, exist_ok=False)
        stage('score_full_population')
        scores, terms = [], []
        for offset in range(0, session.features.num_rows, recipe['batch_size']):
            result, detail = session._score_rows(session.features.slice(offset, recipe['batch_size']))
            scores.append(result); terms.append(detail)
        scores, terms = pa.concat_tables(scores), pa.concat_tables(terms)
        for name, table in [('features.parquet',session.features),('scores.parquet',scores),('components.parquet',terms)]:
            pq.write_table(table, partial/name, compression='snappy', row_group_size=65536)
            if not table.equals(pq.read_table(partial/name)):
                raise DataError('scoring_roundtrip', 'Written table failed readback')
        stage('single_batch_consistency')
        # Representative real records: first 8 plus both no-history and null examples.
        examples = session.features['SK_ID_CURR'].to_pylist()[:8]
        for prefix in ['BUREAU','PREV']:
            idx = pc.indices_nonzero(pc.equal(session.features[prefix+'_ELIGIBLE_COUNT'], 0)).to_pylist()
            if idx: examples.append(session.features['SK_ID_CURR'][idx[0]].as_py())
        for c in session.contract['features']:
            if session.features[c['name']].null_count:
                idx = pc.indices_nonzero(pc.is_null(session.features[c['name']]))[0].as_py()
                examples.append(session.features['SK_ID_CURR'][idx].as_py())
        examples = list(dict.fromkeys(examples))
        batch, _ = session.score_ids(examples); delta = 0.
        for i, application_id in enumerate(examples):
            single, _ = session.score_ids([application_id])
            for field in ['risk_probability','scorecard_probability','logistic_probability','scorecard_score']:
                delta = max(delta, abs(single[field][0].as_py()-batch[field][i].as_py()))
        if delta > 1e-10:
            raise DataError('scoring_consistency', 'Single and batch scores differ beyond 1e-10')
        checks = {'single_batch_max_difference': delta, 'representative_ids': examples,
                  'rows': scores.num_rows, 'sample_uses': {x: uses for x, uses in zip(*np.unique(session.features['sample_use'].to_numpy(), return_counts=True))},
                  'fit_called': False, 'original_test_feature_table_read': False,
                  'publication_new_run_no_overwrite': True}
        checks['sample_uses'] = {k: int(v) for k, v in checks['sample_uses'].items()}
        for name, value in [('case.json',session.case(recipe['case_id'])),('feature_contract.json',session.contract),
                            ('point_table.json',point_table(session.models['scorecard']['model'],session.models['scorecard']['bundle']['recipe'])),
                            ('recipe.json',recipe),('checks.json',checks)]:
            write_json(partial/name, value)
        b['files'] = {name: {'bytes': (partial/name).stat().st_size, 'sha256': sha256_file(partial/name)} for name in sorted(FILES)}
        _verify_references(b)
        if sha256_file(recipe_path) != recipe_hash or any(sha256_file(root/p) != h for p, h in b['code_sha256'].items()):
            raise DataError('scoring_changed', 'Scoring recipe or code changed during run')
        write_json(partial/'bundle.json', b)
        stage('publish_complete_package')
        digest = sha256_file(partial/'bundle.json')
        _publish(partial, final); promoted = True
        record.finish(True, scoring={'artifact_dir': str(final), 'bundle_sha256': digest,
                      'input_fingerprint': b['input_fingerprint'], 'rows': scores.num_rows, 'checks': checks},
                      elapsed_seconds=round(time.monotonic()-start, 3), **process_peak_memory())
        # Stable passed report + receipt are required by the downstream loader.
        with (final/'completed.json').open('x', encoding='utf-8') as f:
            json.dump({'report': str(record.path), 'report_sha256': sha256_file(record.path), 'bundle_sha256': digest}, f)
        return record.data['scoring']
    except BaseException as exc:
        # Only this run's explicitly resolved, workspace-contained output is removable.
        for target in [partial, final] if promoted else [partial]:
            if target.exists() and target.resolve().is_relative_to((root/'reports/scoring/full').resolve()):
                shutil.rmtree(target)
        error = exc.as_dict() if isinstance(exc, DataError) else {'code': type(exc).__name__, 'message': str(exc)}
        record.finish(False, error=error, elapsed_seconds=round(time.monotonic()-start, 3), **process_peak_memory())
        raise


def load_scoring_bundle(root, report_path):
    """E11 success interface. Reject partial/failed/tampered/mismatched packages."""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = json.loads(report_path.read_text(encoding='utf-8')); info = r['scoring']
        if not report_path.is_relative_to(root/'reports/runs') or r['status'] != 'passed' or r['scope'] != 'official_full_scoring' or r['metadata']['module'] != 'E10':
            raise ValueError('passed full E10 required')
        folder = Path(info['artifact_dir']).resolve()
        if folder.parent != root/'reports/scoring/full' or folder.name != r['run_id'] or sha256_file(folder/'bundle.json') != info['bundle_sha256']:
            raise ValueError('path or bundle hash')
        b = json.loads((folder/'bundle.json').read_text(encoding='utf-8'))
        if (b['purpose'] != 'formal_existing_application_scoring' or b['version'] != VERSION or b['status'] != 'complete'
                or b['run_id'] != r['run_id'] or b['environment'] != environment() or set(b['files']) != FILES
                or set(b['code_sha256']) != set(CODE) or any(sha256_file(root/p) != h for p,h in b['code_sha256'].items())):
            raise ValueError('purpose/version/environment/code/files')
        if info['input_fingerprint'] != b['input_fingerprint'] or json.loads((folder/'recipe.json').read_text(encoding='utf-8')) != b['recipe']:
            raise ValueError('recipe or input fingerprint')
        _verify_references(b)
        for name, meta in b['files'].items():
            path = folder/name
            if path.stat().st_size != meta['bytes'] or sha256_file(path) != meta['sha256']:
                raise ValueError('changed output')
        receipt = json.loads((folder/'completed.json').read_text(encoding='utf-8'))
        if Path(receipt['report']).resolve() != report_path or receipt['report_sha256'] != sha256_file(report_path) or receipt['bundle_sha256'] != info['bundle_sha256']:
            raise ValueError('completion receipt')
        if sha256_file(b['assessment_report']) != b['assessment_report_sha256']:
            raise ValueError('assessment report changed')
        upstream = load_assessment_bundle(root, b['assessment_report'])
        p = upstream['selection']['payload']; contract = json.loads((folder/'feature_contract.json').read_text(encoding='utf-8'))
        if (b['selection_sha256'] != upstream['bundle']['selection_sha256'] or b['model_references'] != p['models']
                or b['threshold'] != p['thresholds']['tree'] or b['contract'] != contract
                or b['output_versions']['protocol_fingerprint'] != p['protocol_fingerprint']
                or sha256_file(upstream['directory']/'bundle.json') != b['assessment_bundle_sha256']
                or b['threshold_comparison'] != 'probability >= threshold'):
            raise ValueError('selection/threshold/contract mismatch')
        features = pq.read_table(folder/'features.parquet')
        if features.column_names != ['SK_ID_CURR',*contract['feature_columns'],'sample_use'] or features.num_rows != b['rows'] or info['rows'] != b['rows']:
            raise ValueError('feature population')
        validate_ids(features['SK_ID_CURR'].to_pylist()); validate_features(features.select(contract['feature_columns']),contract)
        models = _models(root, upstream['selection'])
        if any(x['bundle']['contract'] != contract for x in models.values()):
            raise ValueError('model contract')
        return ScoringSession(root, b, features, models, folder, r)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('scoring_bundle_invalid', 'Complete matching immutable E10 success package and E09 required') from exc
