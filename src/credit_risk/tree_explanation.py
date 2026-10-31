"""Training-count weighted path contributions for the frozen binary HGB model.

For each tree, a node reference is its descendant leaf values weighted by
training record counts. Child-reference minus parent-reference is assigned to
the split feature along the applicant's path. The sum telescopes exactly to
the leaf value. Contributions add in raw log odds, never percentage points.
This path-order-dependent allocation is neither SHAP nor a causal explanation.
"""
import importlib
import inspect
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .demo import LABELS
from .logistic import matrix
from .normalize import write_json
from .raw import DataError, sha256_file
from .runtime import process_peak_memory
from .scoring import load_scoring_bundle, validate_features, validate_ids
from .unlabeled_scoring import load_unlabeled_bundle

VERSION = 'tree_path_explanation_v1'
METHOD = 'training_count_weighted_path_log_odds'
FILES = {'reference.json', 'feature_contract.json', 'recipe.json', 'cases.json', 'checks.json'}
CODE = ['src/credit_risk/tree_explanation.py', 'src/credit_risk/tree_explanation_page.py',
        'scripts/run_tree_explanation.py', 'app/tree_explanation.css',
        'src/credit_risk/demo.py', 'src/credit_risk/scoring_api.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def native_sources():
    names = ['gradient_boosting', 'binning', 'predictor', '_predictor', 'common']
    return {str(Path(inspect.getfile(importlib.import_module(
        'sklearn.ensemble._hist_gradient_boosting.'+n))).resolve()): None for n in names}


def load_recipe(root, path):
    try:
        v = read(path)
        if (set(v) != {'version', 'method', 'labeled_report', 'unlabeled_report',
                       'verification_rows_per_population', 'tolerance', 'max_batch'}
                or v['version'] != VERSION or v['method'] != METHOD
                or type(v['verification_rows_per_population']) is not int
                or v['verification_rows_per_population'] != 256
                or v['tolerance'] != 1e-10 or type(v['max_batch']) is not int or v['max_batch'] != 100):
            raise ValueError('fixed recipe required')
        for k in ['labeled_report', 'unlabeled_report']:
            if (not isinstance(v[k], str) or Path(v[k]).is_absolute()
                    or not (Path(root)/v[k]).resolve().is_relative_to(Path(root)/'reports/runs')):
                raise ValueError('project report required')
        return v
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('tree_explanation_recipe', 'Unsupported explanation method, budget or source') from exc


def node_references(nodes, width):
    """Compute references from final leaf values, not internal training updates."""
    required = {'value', 'count', 'feature_idx', 'left', 'right', 'is_leaf',
                'is_categorical', 'bitset_idx', 'bin_threshold', 'missing_go_to_left'}
    if (nodes.ndim != 1 or not len(nodes) or not required.issubset(nodes.dtype.names or ())
            or not np.all(nodes['count'] > 0) or not np.all(np.isfinite(nodes['value']))):
        raise DataError('tree_explanation_structure', 'Finite nonempty counted tree nodes required')
    seen, order, stack = set(), [], [0]
    while stack:
        i = stack.pop()
        if i in seen or not 0 <= i < len(nodes):
            raise DataError('tree_explanation_structure', 'Tree must have unique reachable acyclic nodes')
        seen.add(i); order.append(i); node = nodes[i]
        if int(node['is_leaf']) not in (0, 1):
            raise DataError('tree_explanation_structure', 'Invalid leaf flag')
        if not node['is_leaf']:
            left, right = int(node['left']), int(node['right'])
            if (not 0 <= int(node['feature_idx']) < width or not 0 <= left < len(nodes)
                    or not 0 <= right < len(nodes) or left == right
                    or int(nodes[left]['count'])+int(nodes[right]['count']) != int(node['count'])):
                raise DataError('tree_explanation_structure', 'Child count, index or feature mismatch')
            stack.extend([left, right])
    if len(seen) != len(nodes):
        raise DataError('tree_explanation_structure', 'Unreachable nodes rejected')
    refs = np.empty(len(nodes), dtype=float)
    for i in reversed(order):
        n = nodes[i]
        if n['is_leaf']: refs[i] = float(n['value'])
        else:
            left, right = int(n['left']), int(n['right'])
            refs[i] = (int(nodes[left]['count'])*refs[left]+int(nodes[right]['count'])*refs[right])/int(n['count'])
    return refs


def walk_tree(nodes, bitsets, refs, binned, missing_bin):
    """Independent vectorized traversal of the fitted model's binned inputs."""
    n, width = binned.shape
    contributions = np.zeros((n, width)); leaves = np.empty(n)
    stack = [(0, np.arange(n))]
    while stack:
        i, rows = stack.pop()
        if not len(rows): continue
        node = nodes[i]
        if node['is_leaf']:
            leaves[rows] = node['value']; continue
        feature = int(node['feature_idx']); values = binned[rows, feature]
        if node['is_categorical']:
            k = int(node['bitset_idx'])
            if bitsets.ndim != 2 or bitsets.shape[1] != 8 or not 0 <= k < len(bitsets):
                raise DataError('tree_explanation_structure', 'Categorical bitset layout differs')
            left = ((bitsets[k, values.astype(int)//32] >> (values.astype(np.uint32) % 32)) & 1).astype(bool)
        else: left = values <= node['bin_threshold']
        left = np.where(values == missing_bin, bool(node['missing_go_to_left']), left)
        for take_left, child in [(True, int(node['left'])), (False, int(node['right']))]:
            selected = rows[left == take_left]
            contributions[selected, feature] += refs[child]-refs[i]
            stack.append((child, selected))
    return contributions, leaves


class TreePathExplainer:
    def __init__(self, pipeline, contract):
        self.pipeline, self.contract = pipeline, contract
        self.columns = contract['feature_columns']
        self.tree = pipeline.named_steps['tree']
        t = self.tree
        if (sklearn.__version__ != '1.9.1' or set(pipeline.named_steps) != {'empty', 'tree'}
                or len(self.columns) != 40 or t.n_trees_per_iteration_ != 1
                or list(t.classes_) != [0, 1] or t.loss != 'log_loss'
                or t.early_stopping is not False or t.warm_start
                or t.n_iter_ != 120 or t._baseline_prediction.shape != (1, 1)):
            raise DataError('tree_explanation_model', 'Pinned frozen binary tree pipeline required')
        mask = np.asarray(t.is_categorical_, dtype=bool)
        if mask.shape != (40,) or mask.tolist() != [x['type'] == 'string' for x in contract['features']]:
            raise DataError('tree_explanation_order', 'Native categorical mask differs from contract')
        self.order = np.r_[np.flatnonzero(mask), np.flatnonzero(~mask)]
        self.predictors = [p[0] for p in t._predictors]
        self.refs = [node_references(p.nodes, 40) for p in self.predictors]
        self.intercept = float(t._baseline_prediction[0, 0])
        self.base = self.intercept+sum(float(r[0]) for r in self.refs)
        counts = {int(p.nodes[0]['count']) for p in self.predictors}
        if len(counts) != 1: raise DataError('tree_explanation_counts', 'Root training counts differ')
        self.training_rows = counts.pop()
        self.used = sorted({int(n['feature_idx']) for p in self.predictors for n in p.nodes if not n['is_leaf']})

    def reference(self):
        return {'version': VERSION, 'method': METHOD, 'unit': 'log_odds', 'tree_count': len(self.predictors),
                'training_rows': self.training_rows, 'intercept': self.intercept,
                'base_log_odds': self.base, 'sigmoid_of_base': float(expit(self.base)),
                'internal_feature_order': [self.columns[i] for i in self.order],
                'node_references': [r.tolist() for r in self.refs],
                'formula': 'base_log_odds + sum(feature_contribution_log_odds) = raw_tree_score; sigmoid(raw_tree_score) = probability',
                'reference_meaning': 'Count-weighted descendant final leaves; baseline is mean training raw score, not event rate or mean probability.',
                'limits': ['Path order and correlated fields affect attribution; not SHAP or permutation importance.',
                           'Describes this saved model, not causes, approval reasons or actions guaranteed to change outcomes.',
                           'Log-odds terms are not independent probability percentage points.']}

    def explain(self, features):
        validate_features(features, self.contract)
        X = matrix(features, self.columns, self.contract)
        with threadpool_limits(limits=2):
            guarded = self.pipeline.named_steps['empty'].transform(X)
            internal = self.tree._preprocess_X(guarded, reset=False)
            binned = self.tree._bin_mapper.transform(internal)
            if binned.shape != (features.num_rows, 40) or binned.dtype != np.uint8:
                raise DataError('tree_explanation_bins', 'Unexpected fitted bin mapper result')
            values = np.zeros((features.num_rows, 40)); raw_leaves = np.full(features.num_rows, self.intercept)
            leaf_delta = 0.
            for predictor, refs in zip(self.predictors, self.refs):
                terms, leaves = walk_tree(predictor.nodes, predictor.binned_left_cat_bitsets, refs, binned,
                                          self.tree._bin_mapper.missing_values_bin_idx_)
                oracle = predictor.predict_binned(binned, self.tree._bin_mapper.missing_values_bin_idx_, 2)
                leaf_delta = max(leaf_delta, float(np.max(np.abs(leaves-oracle))))
                values += terms; raw_leaves += leaves
            original_order = np.zeros_like(values); original_order[:, self.order] = values
            raw = self.pipeline.decision_function(X)
            probability = self.pipeline.predict_proba(X)[:, 1]
        reconstructed = self.base+original_order.sum(axis=1)
        errors = {'leaf_max_difference': leaf_delta,
                  'leaf_sum_raw_max_difference': float(np.max(np.abs(raw_leaves-raw))),
                  'additivity_max_difference': float(np.max(np.abs(reconstructed-raw))),
                  'probability_max_difference': float(np.max(np.abs(expit(reconstructed)-probability)))}
        if any(not np.isfinite(v) or v > 1e-10 for v in errors.values()):
            raise DataError('tree_explanation_consistency', 'Path, raw additivity or probability differs from saved tree')
        if not np.isfinite(original_order).all():
            raise DataError('tree_explanation_nonfinite', 'Nonfinite contribution')
        return {'contributions': original_order, 'raw_score': raw, 'probability': probability, 'checks': errors}


class ExplanationSession:
    def __init__(self, root, recipe):
        self.root, self.recipe = Path(root).resolve(), recipe
        labeled = load_scoring_bundle(self.root, self.root/recipe['labeled_report'])
        unlabeled = load_unlabeled_bundle(self.root, self.root/recipe['unlabeled_report'])
        if (labeled.contract != unlabeled.contract or labeled.bundle['threshold'] != unlabeled.bundle['threshold']
                or labeled.bundle['model_references'] != unlabeled.bundle['model_references']):
            raise DataError('tree_explanation_sources', 'Both scoring populations must share the frozen tree/contract')
        unlabeled.models = labeled.models
        self.sources = {'labeled': labeled, 'unlabeled': unlabeled}
        self.explainer = TreePathExplainer(labeled.models['tree']['model'], labeled.contract)
        self.directory, self.report = None, None

    def explain_ids(self, population, ids):
        if population not in self.sources:
            raise DataError('tree_explanation_population', 'Select labeled or unlabeled population')
        ids = validate_ids(ids)
        if len(ids) > self.recipe['max_batch']:
            raise DataError('tree_explanation_batch', 'At most 100 unique IDs per request')
        source = self.sources[population]
        if any(i not in source.index for i in ids):
            raise DataError('tree_explanation_unknown_id', 'ID absent from selected scoring population')
        rows = source.features.take(pa.array([source.index[i] for i in ids], type=pa.int64()))
        explanation = self.explainer.explain(rows.select(source.contract['feature_columns']))
        scores, _ = source.score_ids(ids)
        delta = float(np.max(np.abs(explanation['probability']-scores['risk_probability'].to_numpy())))
        if delta > 1e-10: raise DataError('tree_explanation_scoring', 'Explanation differs from shared scoring result')
        cases = []
        for j, (values, result) in enumerate(zip(rows.to_pylist(), scores.to_pylist())):
            terms = [{'feature': name, 'label': LABELS.get(name, name), 'value': values[name],
                      'contribution_log_odds': float(explanation['contributions'][j, k])}
                     for k, name in enumerate(source.contract['feature_columns'])]
            cases.append({'result': result, 'population': population, 'method': METHOD, 'unit': 'log_odds',
                'base_log_odds': self.explainer.base, 'raw_score': float(explanation['raw_score'][j]),
                'contribution_sum': float(explanation['contributions'][j].sum()), 'terms': terms,
                'top_increasing': sorted([t for t in terms if t['contribution_log_odds'] > 0], key=lambda t: -t['contribution_log_odds'])[:5],
                'top_decreasing': sorted([t for t in terms if t['contribution_log_odds'] < 0], key=lambda t: t['contribution_log_odds'])[:5],
                'checks': {**explanation['checks'], 'shared_scoring_max_difference': delta},
                'limits': self.explainer.reference()['limits'],
                'sample_notice': 'Prediction only; no observed label or measured accuracy.' if population == 'unlabeled'
                    else 'Existing sample replay, not new performance evidence; prior 769 test exposures remain disclosed.'})
        return cases


def build_reference(root, recipe_path, record, progress=None):
    root, recipe_path = Path(root).resolve(), Path(recipe_path).resolve()
    final = root/'reports/tree_explanations'/record.run_id
    partial = final.with_name('partial-'+record.run_id)
    created, promoted = False, False
    started = time.monotonic()
    try:
        recipe = load_recipe(root, recipe_path)
        refs = {str((root/p).resolve()): sha256_file(root/p) for p in CODE}
        refs.update({p: sha256_file(p) for p in native_sources()})
        refs[str(recipe_path)] = sha256_file(recipe_path)
        if progress: progress('load_formal_scoring_populations')
        session = ExplanationSession(root, recipe)
        for source in session.sources.values():
            report = root/recipe['labeled_report' if source is session.sources['labeled'] else 'unlabeled_report']
            refs[str(report.resolve())] = sha256_file(report)
            refs[str((source.directory/'bundle.json').resolve())] = sha256_file(source.directory/'bundle.json')
        if final.exists() or partial.exists(): raise DataError('tree_explanation_exists', 'Run directory cannot be overwritten')
        partial.mkdir(parents=True, exist_ok=False); created = True
        if progress: progress('verify_training_reference_without_labels')
        source = session.sources['labeled']
        saved = pq.read_table(source.directory/'scores.parquet', columns=['sample_use', 'risk_probability'])
        probabilities = saved.filter(pc.equal(saved['sample_use'], 'train'))['risk_probability'].to_numpy()
        if len(probabilities) != session.explainer.training_rows or not np.all((probabilities > 0) & (probabilities < 1)):
            raise DataError('tree_explanation_reference', 'Training counts or finite training raw reference differ')
        mean_raw = float(np.mean(np.log(probabilities)-np.log1p(-probabilities)))
        baseline_delta = abs(mean_raw-session.explainer.base)
        if baseline_delta > 1e-10: raise DataError('tree_explanation_reference', 'Node reference differs from mean saved training raw score')
        if progress: progress('verify_paths_and_reconstruct_scores')
        checks = {'training_reference_mean_raw_difference': baseline_delta, 'populations': {},
                  'fit_called': False, 'new_effect_evaluation': False, 'original_reserved_test_features_read': False}
        for name, purpose in [('labeled', 'train'), ('labeled', 'validation'), ('unlabeled', 'unlabeled_scoring')]:
            source = session.sources[name]
            rows = source.features.filter(pc.equal(source.features['sample_use'], purpose)).sort_by('SK_ID_CURR').slice(0, 256)
            result = session.explainer.explain(rows.select(source.contract['feature_columns']))
            checks['populations'][purpose] = {'rows': rows.num_rows, 'ids': rows['SK_ID_CURR'].to_pylist(), **result['checks']}
        case_specs = [('labeled', 100003), ('labeled', 100006), ('unlabeled', 100001)]
        cases = [session.explain_ids(p, [i])[0] for p, i in case_specs]
        for name, value in [('reference.json', session.explainer.reference()), ('recipe.json', recipe),
                            ('feature_contract.json', session.explainer.contract), ('cases.json', cases), ('checks.json', checks)]:
            write_json(partial/name, value)
        if any(sha256_file(p) != h for p, h in refs.items()):
            raise DataError('tree_explanation_changed', 'Code, native dependency or source changed during build')
        b = {'version': VERSION, 'method': METHOD, 'status': 'complete', 'run_id': record.run_id,
             'references': refs, 'code': CODE, 'sklearn_version': sklearn.__version__, 'recipe': recipe,
             'files': {n: {'bytes': (partial/n).stat().st_size, 'sha256': sha256_file(partial/n)} for n in sorted(FILES)}}
        write_json(partial/'bundle.json', b); digest = sha256_file(partial/'bundle.json')
        partial.rename(final); promoted = True
        result = {'artifact_dir': str(final), 'bundle_sha256': digest, 'checks': checks}
        record.finish(True, tree_explanation=result, elapsed_seconds=round(time.monotonic()-started, 3), **process_peak_memory())
        with (final/'completed.json').open('x', encoding='utf-8') as f:
            json.dump({'report': str(record.path), 'report_sha256': sha256_file(record.path), 'bundle_sha256': digest}, f)
        return result
    except BaseException as exc:
        record.finish(False, error=exc.as_dict() if isinstance(exc, DataError) else {'code': type(exc).__name__, 'message': str(exc)}, traceback=traceback.format_exc())
        if created and not promoted and partial.exists() and partial.resolve().parent == (root/'reports/tree_explanations').resolve():
            shutil.rmtree(partial)
        raise


def load_explanation_bundle(root, report_path):
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = read(report_path); info = r['tree_explanation']; folder = Path(info['artifact_dir']).resolve()
        if (not report_path.is_relative_to(root/'reports/runs') or r['status'] != 'passed'
                or r['scope'] != 'official_tree_path_explanation' or r['metadata']['module'] != 'E10_tree_explanation'
                or folder.parent != root/'reports/tree_explanations' or folder.name != r['run_id']
                or sha256_file(folder/'bundle.json') != info['bundle_sha256']): raise ValueError('success/path/hash')
        b = read(folder/'bundle.json')
        if (b['version'] != VERSION or b['method'] != METHOD or b['status'] != 'complete'
                or b['run_id'] != r['run_id'] or b['code'] != CODE or set(b['files']) != FILES
                or b['sklearn_version'] != sklearn.__version__ or any(sha256_file(p) != h for p, h in b['references'].items())):
            raise ValueError('bundle/reference mismatch')
        expected_refs = {str((root/p).resolve()) for p in CODE} | set(native_sources())
        if not expected_refs.issubset(b['references']): raise ValueError('missing code/native bindings')
        if read(folder/'completed.json') != {'report': str(report_path), 'report_sha256': sha256_file(report_path), 'bundle_sha256': info['bundle_sha256']}:
            raise ValueError('completion receipt')
        for name, m in b['files'].items():
            if (folder/name).stat().st_size != m['bytes'] or sha256_file(folder/name) != m['sha256']: raise ValueError('changed output')
        recipe = load_recipe(root, folder/'recipe.json')
        if recipe != b['recipe']: raise ValueError('recipe mismatch')
        session = ExplanationSession(root, recipe)
        if read(folder/'reference.json') != session.explainer.reference() or read(folder/'feature_contract.json') != session.explainer.contract:
            raise ValueError('recomputed reference or feature contract mismatch')
        session.directory, session.report = folder, r
        return session
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('tree_explanation_bundle', 'Complete unchanged explanation package and scoring sources required') from exc
