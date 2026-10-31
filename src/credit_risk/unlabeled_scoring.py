"""Additive E03/E04/E10 path: unlabeled applications, fixed SQL and saved models.

Never trains, changes splits, reads reserved test features or evaluates a label.
The original E10 bundle remains a required, independently verified dependency.
"""
import copy
import json
from pathlib import Path
import shutil
import time
import traceback

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .history import (FEATURE_NAMES, SQL_FILES, join_query, literal,
                      load_history_bundle, load_history_config, verify_unique)
from .normalize import load_normalized_bundle, write_json
from .parquet import write_verified_parquet
from .raw import DataError, RawDataSource, require_data_scope, sha256_file
from .runtime import load_config, now, process_peak_memory
from .scoring import (ScoringSession, load_scoring_bundle, validate_features,
                      validate_ids)

VERSION = 'unlabeled_scoring_v1'
SCOPE = 'existing_unlabeled_applications_scoring'
USE = 'unlabeled_scoring'
FILES = {'application_test.parquet', 'features.parquet', 'scores.parquet',
         'components.parquet', 'checks.json', 'case.json', 'recipe.json',
         'feature_contract.json', *[Path(p).name for p in SQL_FILES]}
CODE = ['src/credit_risk/'+n+'.py' for n in (
    'unlabeled_scoring', 'scoring', 'history', 'normalize', 'parquet', 'raw',
    'runtime', 'quality', 'experiments', 'logistic', 'scorecard', 'tree',
    'assessment')] + ['scripts/run_unlabeled_scoring.py', 'scripts/_project.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_recipe(root, path):
    try:
        v = read(path)
        if (set(v) != {'version', 'scoring_report', 'scope', 'source', 'batch_size', 'tolerance'}
                or v['version'] != VERSION or v['scope'] != SCOPE
                or v['source'] != 'application_test.csv'
                or type(v['batch_size']) is not int or not 1 <= v['batch_size'] <= 10000
                or v['tolerance'] != 1e-10
                or not isinstance(v['scoring_report'], str)
                or Path(v['scoring_report']).is_absolute()
                or not (Path(root)/v['scoring_report']).resolve().is_relative_to(Path(root)/'reports/runs')):
            raise ValueError('unsupported recipe')
        return v
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('unlabeled_recipe_invalid', 'Fixed unlabeled scoring recipe required') from exc


class UnlabeledSession(ScoringSession):
    """Public ID lookup is confined to the verified unlabeled population."""
    def score_ids(self, ids):
        ids = validate_ids(ids)
        if any(x not in self.index for x in ids):
            raise DataError('unlabeled_unknown_id', 'ID absent from this unlabeled scoring population')
        return self._score_rows(self.features.take(pa.array([self.index[x] for x in ids], type=pa.int64())))

    def case(self, application_id):
        result = super().case(application_id)
        result['sample_notice'] = (
            'Unlabeled application: prediction only; no observed TARGET, AUC, '
            'hit rate or independent performance claim. Not a labeled test partition.')
        return result


def make_session(base, features, raw_sha, run_id):
    b = copy.deepcopy(base.bundle)
    b.update(version=VERSION, purpose='formal_unlabeled_application_scoring', rows=features.num_rows, run_id=run_id)
    b['output_versions'].update(
        training_dataset_version=base.bundle['output_versions']['dataset_version'],
        dataset_version='application_test_sha256:'+raw_sha,
        scoring_version=VERSION, scoring_scope=SCOPE, unlabeled_scoring_run_id=run_id)
    b['limitations']['unlabeled_application_test'] = 'supported_fixed_40_saved_models_no_labels'
    b['limitations']['sample_use'] = 'unlabeled_scoring is distinct from the frozen labeled test partition'
    b['limitations']['threshold_transport'] = 'Original validation threshold reused; no unlabeled quality or calibration validation'
    return UnlabeledSession(base.root, b, features, base.models)


def build_features(root, source, normalized, history, directory, contract):
    """All 121 main fields retain E03 types; all 24 history formulas reuse E04 SQL."""
    columns = [c for c in read(normalized['directory']/'contract.json')['tables']['application_train.csv']
               if c['name'] != 'TARGET']
    with source.open_table('application_test.csv') as (header, _):
        if header != [c['name'] for c in columns]:
            raise DataError('unlabeled_schema', 'Exact main schema minus TARGET required')
    conversion = write_verified_parquet(source, 'application_test.csv', directory/'application_test.parquet',
        [{k: c[k] for k in ('name', 'type')} for c in columns], output_root=directory,
        row_limit=None, batch_size=8192, columnar=True)
    cfg = load_history_config(root, root/'config/history_features.json')
    if (cfg.sha256 != history['bundle']['config_sha256']
            or {p: v['sha256'] for p, v in cfg.sql.items()} != history['bundle']['sql_sha256']):
        raise DataError('unlabeled_history_contract', 'History SQL/config must equal accepted E04')
    with duckdb.connect(config={'threads': '1', 'memory_limit': '512MB',
                               'temp_directory': str(directory/'spill')}) as con:
        con.execute('CREATE TEMP VIEW application_input AS SELECT * FROM read_parquet('
                    +literal(directory/'application_test.parquet')+', file_row_number=true)')
        n = verify_unique(con, 'application_input')
        con.fetchall()  # Complete E04 helper's fetchone result before relation API use.
        for name, alias in [('bureau.csv', 'bureau'), ('previous_application.csv', 'previous')]:
            con.execute(f'CREATE TEMP VIEW {alias}_input AS SELECT * FROM read_parquet('
                        +literal(normalized['tables'][name])+')')
        for sql in SQL_FILES[:2]:
            con.execute(cfg.sql[sql]['text'])
        for relation in ['bureau_features', 'previous_features']:
            verify_unique(con, relation)
            con.fetchall()
        joined = con.sql(join_query(cfg)).to_arrow_table()
        if joined.num_rows != n or n != conversion['rows']:
            raise DataError('unlabeled_join', 'Join changed main row count')
        original = pq.read_table(directory/'application_test.parquet')
        if not joined.select(original.column_names).equals(original):
            raise DataError('unlabeled_preservation', 'Join changed main fields, order or ID population')
        # Independent comparison against already accepted full-history summaries.
        # E04 full aggregation includes historical IDs for both main populations.
        for relation, filename in [('bureau_features', 'bureau_features.parquet'),
                                   ('previous_features', 'previous_features.parquet')]:
            old = con.sql('SELECT * FROM read_parquet('+literal(history['directory']/filename)+')').to_arrow_table()
            new = con.sql('SELECT * FROM '+relation).to_arrow_table()
            if not old.sort_by('SK_ID_CURR').equals(new.sort_by('SK_ID_CURR')):
                raise DataError('unlabeled_aggregation', 'Reused SQL differs from accepted full-history aggregation')
        features = joined.select(['SK_ID_CURR', *contract['feature_columns']])
        validate_ids(features['SK_ID_CURR'].to_pylist())
        validate_features(features.select(contract['feature_columns']), contract)
        features = features.append_column('sample_use', pa.array([USE]*n))
        coverage = {p: int(pc.sum(pc.cast(pc.greater(features[p+'_ELIGIBLE_COUNT'], 0), pa.int64())).as_py())
                    for p in ['BUREAU', 'PREV']}
        pq.write_table(features, directory/'features.parquet', compression='snappy')
    for sql in SQL_FILES:
        (directory/Path(sql).name).write_text(cfg.sql[sql]['text'], encoding='utf-8')
    return features, {'conversion': conversion, 'rows': n, 'history_present_rows': coverage,
        'main_all_fields_values_order_preserved': True, 'history_summaries_match_accepted_E04': True,
        'feature_count': len(contract['feature_columns']), 'label_column_present': False}


def validate_outputs(session, scores, components, tolerance):
    ids = session.features['SK_ID_CURR']
    value_fields = ['risk_probability', 'risk_flag', 'scorecard_probability', 'scorecard_score',
                    'scorecard_base_points', 'logistic_probability']
    if (scores.column_names != ['SK_ID_CURR', 'sample_use', *value_fields, 'threshold', *session.bundle['output_versions']]
            or components.column_names != ['SK_ID_CURR', *session.contract['feature_columns']]):
        raise DataError('unlabeled_output_schema', 'Exact score and component schemas required')
    if (not scores['SK_ID_CURR'].equals(ids) or not components['SK_ID_CURR'].equals(ids)
            or set(scores['sample_use'].to_pylist()) != {USE}):
        raise DataError('unlabeled_output_population', 'Output IDs/order/purpose differ')
    if any(scores[k].unique().to_pylist() != [v] for k, v in session.bundle['output_versions'].items()):
        raise DataError('unlabeled_output_versions', 'Stored output versions differ from the frozen session')
    for key in ['risk_probability', 'scorecard_probability', 'logistic_probability']:
        a = scores[key].to_numpy()
        if not np.all(np.isfinite(a) & (a >= 0) & (a <= 1)):
            raise DataError('unlabeled_probability', 'Invalid output probability')
    if not np.array_equal(scores['risk_flag'].to_numpy(), scores['risk_probability'].to_numpy() >= session.bundle['threshold']):
        raise DataError('unlabeled_threshold', 'Original validation threshold comparison differs')
    if not np.all(scores['threshold'].to_numpy() == session.bundle['threshold']):
        raise DataError('unlabeled_threshold', 'Output threshold differs')
    total = scores['scorecard_base_points'].to_numpy().copy()
    for key in session.contract['feature_columns']:
        if not np.all(np.isfinite(components[key].to_numpy())):
            raise DataError('unlabeled_points', 'Nonfinite scorecard component')
        total += components[key].to_numpy()
    delta = float(np.max(np.abs(total-scores['scorecard_score'].to_numpy())))
    if not np.isfinite(delta) or delta > tolerance:
        raise DataError('unlabeled_points', 'Scorecard components do not sum to total')
    return delta


def run_unlabeled(root, recipe_path, record, progress=None):
    root, recipe_path = Path(root).resolve(), Path(recipe_path).resolve()
    started = time.monotonic()
    partial = root/'reports/unlabeled_scoring'/('partial-'+record.run_id)
    final = root/'reports/unlabeled_scoring'/record.run_id
    promoted, created = False, False
    try:
        recipe = load_recipe(root, recipe_path)
        runtime = load_config(root, root/'config/runtime.json')
        source = RawDataSource(runtime.raw_dir, runtime.manifest)
        require_data_scope(source, root/'data/raw')
        refs = {}
        def bind(path):
            path = Path(path).resolve(); refs[str(path)] = sha256_file(path)
        for path in [recipe_path, runtime.path, source.manifest_path, *[root/p for p in CODE]]:
            bind(path)
        raw_path = source.path('application_test.csv'); bind(raw_path)
        entry = source.entries['application_test.csv']
        if refs[str(raw_path)] != entry['sha256'] or raw_path.stat().st_size != entry['bytes']:
            raise DataError('unlabeled_raw_changed', 'Raw unlabeled CSV differs from accepted manifest')
        if progress: progress('load_verified_upstream')
        report_path = root/recipe['scoring_report']; bind(report_path)
        base = load_scoring_bundle(root, report_path)
        history = load_history_bundle(root, base.bundle['history_report'])
        normalized = load_normalized_bundle(root, history['bundle']['input_run_path'])
        for path in [root/'config/history_features.json', normalized['directory']/'bundle.json',
                     normalized['directory']/'contract.json', history['directory']/'bundle.json',
                     *normalized['tables'].values(), *[root/p for p in SQL_FILES]]:
            bind(path)
        if partial.exists() or final.exists():
            raise DataError('unlabeled_exists', 'Never overwrite an existing run')
        partial.mkdir(parents=True, exist_ok=False)
        created = True
        if progress: progress('normalize_and_reuse_history_sql')
        features, checks = build_features(root, source, normalized, history, partial, base.contract)
        if features.num_rows != entry['rows']:
            raise DataError('unlabeled_rows', 'Full manifest population required')
        if set(features['SK_ID_CURR'].to_pylist()) & set(base.index):
            raise DataError('unlabeled_overlap', 'Unlabeled IDs overlap frozen labeled population')
        session = make_session(base, features, refs[str(raw_path)], record.run_id)
        del base
        if progress: progress('predict_saved_models_no_fit')
        score_batches, component_batches = [], []
        for batch in features.to_batches(max_chunksize=recipe['batch_size']):
            a, b = session._score_rows(pa.Table.from_batches([batch]))
            score_batches.append(a); component_batches.append(b)
        scores, components = pa.concat_tables(score_batches), pa.concat_tables(component_batches)
        points_delta = validate_outputs(session, scores, components, recipe['tolerance'])
        ids = [session.features['SK_ID_CURR'][i].as_py() for i in sorted({0, features.num_rows//2, features.num_rows-1})]
        consistency = 0.
        for application_id in ids:
            single, detail = session.score_ids([application_id]); pos = session.index[application_id]
            for key in ['risk_probability', 'scorecard_probability', 'logistic_probability', 'scorecard_score']:
                consistency = max(consistency, abs(single[key][0].as_py()-scores[key][pos].as_py()))
            for key in session.contract['feature_columns']:
                consistency = max(consistency, abs(detail[key][0].as_py()-components[key][pos].as_py()))
        if consistency > recipe['tolerance']:
            raise DataError('unlabeled_consistency', 'Single and full batch differ')
        pq.write_table(scores, partial/'scores.parquet', compression='snappy')
        pq.write_table(components, partial/'components.parquet', compression='snappy')
        checks.update(scorecard_sum_max_difference=points_delta, single_batch_max_difference=consistency,
            representative_ids=ids, labeled_id_overlap=0, fit_called=False, evaluation_called=False,
            reserved_test_features_read=False, target_created=False, sample_use=USE)
        checks['conversion']['output_path'] = str(final/'application_test.parquet')
        for name, value in [('checks.json', checks), ('recipe.json', recipe),
                            ('feature_contract.json', session.contract), ('case.json', session.case(ids[0]))]:
            write_json(partial/name, value)
        # Validate all source bindings again before publishing the new success package.
        if any(sha256_file(p) != h for p, h in refs.items()):
            raise DataError('unlabeled_source_changed', 'An input changed during scoring')
        spill = partial/'spill'
        if spill.exists():
            shutil.rmtree(spill)
        b = {'version': VERSION, 'status': 'complete', 'scope': SCOPE, 'run_id': record.run_id,
             'created_at': now(), 'rows': features.num_rows, 'contract': session.contract,
             'threshold': session.bundle['threshold'], 'output_versions': session.bundle['output_versions'],
             'references': refs, 'code': CODE, 'upstream_scoring_report': str(report_path),
             'files': {p: {'sha256': sha256_file(partial/p), 'bytes': (partial/p).stat().st_size} for p in sorted(FILES)}}
        write_json(partial/'bundle.json', b)
        digest = sha256_file(partial/'bundle.json')
        partial.rename(final); promoted = True
        result = {'artifact_dir': str(final), 'bundle_sha256': digest, 'rows': features.num_rows, 'checks': checks}
        record.finish(True, unlabeled_scoring=result, elapsed_seconds=round(time.monotonic()-started, 3), **process_peak_memory())
        with (final/'completed.json').open('x', encoding='utf-8') as f:
            json.dump({'report': str(record.path), 'report_sha256': sha256_file(record.path), 'bundle_sha256': digest}, f)
        return result
    except BaseException as exc:
        record.finish(False, error=exc.as_dict() if isinstance(exc, DataError) else {'code': type(exc).__name__, 'message': str(exc)}, traceback=traceback.format_exc())
        if created and not promoted and partial.exists() and partial.resolve().parent == (root/'reports/unlabeled_scoring').resolve():
            shutil.rmtree(partial)
        raise


def load_unlabeled_bundle(root, report_path):
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = read(report_path); info = r['unlabeled_scoring']
        folder = Path(info['artifact_dir']).resolve()
        if (not report_path.is_relative_to(root/'reports/runs') or r['status'] != 'passed'
                or r['scope'] != SCOPE or r['metadata']['module'] != 'E10_unlabeled'
                or folder.parent != root/'reports/unlabeled_scoring' or folder.name != r['run_id']
                or sha256_file(folder/'bundle.json') != info['bundle_sha256']):
            raise ValueError('success scope/path/hash')
        b = read(folder/'bundle.json'); receipt = read(folder/'completed.json')
        if (b['version'] != VERSION or b['status'] != 'complete' or b['scope'] != SCOPE
                or b['run_id'] != r['run_id'] or b['code'] != CODE or set(b['files']) != FILES
                or receipt != {'report': str(report_path), 'report_sha256': sha256_file(report_path), 'bundle_sha256': info['bundle_sha256']}):
            raise ValueError('bundle/receipt')
        if any(sha256_file(p) != h for p, h in b['references'].items()):
            raise ValueError('changed reference')
        if not all(str((root/p).resolve()) in b['references'] for p in CODE):
            raise ValueError('code binding missing')
        for p, m in b['files'].items():
            if (folder/p).stat().st_size != m['bytes'] or sha256_file(folder/p) != m['sha256']:
                raise ValueError('changed output')
        recipe = load_recipe(root, folder/'recipe.json')
        if str((root/recipe['scoring_report']).resolve()) != b['upstream_scoring_report']:
            raise ValueError('upstream mismatch')
        base = load_scoring_bundle(root, b['upstream_scoring_report'])
        features = pq.read_table(folder/'features.parquet')
        if (features.column_names != ['SK_ID_CURR', *base.contract['feature_columns'], 'sample_use']
                or features.num_rows != b['rows'] or b['rows'] != info['rows']
                or set(features['sample_use'].to_pylist()) != {USE}
                or b['contract'] != base.contract or b['contract'] != read(folder/'feature_contract.json')):
            raise ValueError('population/contract')
        validate_ids(features['SK_ID_CURR'].to_pylist())
        validate_features(features.select(base.contract['feature_columns']), base.contract)
        if set(features['SK_ID_CURR'].to_pylist()) & set(base.index):
            raise ValueError('labeled overlap')
        raw_sha = b['references'][str((root/'data/raw/home_credit/application_test.csv').resolve())]
        session = make_session(base, features, raw_sha, r['run_id'])
        if b['threshold'] != session.bundle['threshold'] or b['output_versions'] != session.bundle['output_versions']:
            raise ValueError('frozen model/threshold/version mismatch')
        validate_outputs(session, pq.read_table(folder/'scores.parquet'), pq.read_table(folder/'components.parquet'), 1e-10)
        session.directory, session.report = folder, r
        return session
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('unlabeled_bundle_invalid', 'Complete matching immutable unlabeled package required') from exc
