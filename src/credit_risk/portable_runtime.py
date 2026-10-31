"""Formal portable consumer and shared page/API/monitoring bindings for F05."""
from pathlib import Path, PureWindowsPath
import json
import threading

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .portable_release import verify_directory, read, require, SPECS, VERSION
from .scoring import ScoringSession, validate_ids, validate_features
from .unlabeled_scoring import make_session, validate_outputs
from .tree_explanation import ExplanationSession, TreePathExplainer
from .scoring_api import ApiState
from .demo import DemoState
from .raw import DataError
from .logistic import environment


class PortableExplanation(ExplanationSession):
    def __init__(self, root, recipe, sources, reference, report, directory):
        self.root, self.recipe, self.sources = root, recipe, sources
        self.explainer = TreePathExplainer(sources['labeled'].models['tree']['model'], sources['labeled'].contract)
        require(self.explainer.reference() == reference, 'portable_explanation', 'Original explanation reference differs')
        self.report, self.directory = report, directory


class Release:
    def __init__(self, origin, graph, digest):
        self.origin, self.root, self.manifest = origin, origin.root, origin.m
        self.manifest_sha256, self.graph_files = digest, graph
        self.sources = {}

    def package(self, role):
        report = self.origin.json(str(self.origin.old/self.manifest['reports'][role]))
        info = report[SPECS[role][0]]
        directory = info.get('artifact_dir', info.get('directory'))
        filename = 'reference.json' if role == 'monitor' else 'bundle.json'
        b = self.origin.json(str(PureWindowsPath(directory)/filename))
        return report, b, self.origin.path(str(PureWindowsPath(directory)/filename)).parent


def load_release(directory, digest):
    """Only public constructor: complete external seal + lineage before joblib."""
    origin, graph = verify_directory(directory, digest)
    release = Release(origin, graph, digest)
    r, b, folder = release.package('scoring')
    contract = b['contract']; acceptance = release.manifest['acceptance']
    require(b['status'] == 'complete' and b['purpose'] == 'formal_existing_application_scoring'
            and b['contract'] == acceptance['contract'] and b['threshold'] == acceptance['threshold']
            and b['output_versions'] == acceptance['output_versions'] and b['environment'] == environment(),
            'portable_scoring', 'Original frozen scoring contract differs')
    assessment = origin.json(b['assessment_report'])
    assessment_bundle = origin.json(str(PureWindowsPath(assessment['assessment']['artifact_dir'])/'bundle.json'))
    selection = origin.json(assessment_bundle['selection_path'])['payload']
    require(assessment['status'] == 'passed' and assessment['scope'] == 'official_final_assessment'
            and selection['decision']['recommended'] == 'tree' and b['model_references'] == selection['models']
            and selection['thresholds']['tree'] == b['threshold'], 'portable_selection', 'Locked original selection required')
    models = {}
    for name, module, scope in [('logistic', 'E06', 'official_full_logistic_baseline'),
                                 ('scorecard', 'E07', 'official_full_scorecard'), ('tree', 'E08', 'official_full_tree')]:
        ref = b['model_references'][name]; report = origin.json(ref['report_path'])
        mb = origin.json(ref['bundle_path']); modeldir = origin.path(ref['bundle_path']).parent
        require(report['status'] == 'passed' and report['scope'] == scope and report['metadata']['module'] == module
                and mb['status'] == 'complete' and mb['environment'] == environment() and mb['contract'] == contract,
                'portable_model', 'Accepted matching fitted model required')
        require(mb['version'] == {'logistic':'logistic_v1','scorecard':'scorecard_v2','tree':'tree_v1'}[name]
                and mb['purpose'] == {'logistic':'formal_logistic_baseline','scorecard':'formal_traditional_scorecard',
                                      'tree':'formal_tree'}[name]
                and mb['fit_origin'] == 'fresh_training_partition_and_each_fold',
                'portable_model', 'Original formal version and fit lineage required')
        # Every model byte and registration was already checked, before any load.
        item = {'report': report, 'bundle': mb, 'directory': modeldir}
        if name == 'logistic': item['models'] = {n:joblib.load(modeldir/('model_'+n+'.joblib')) for n in ['full40', 'main16']}
        else: item['model'] = joblib.load(modeldir/'model.joblib')
        models[name] = item
    features = pq.read_table(folder/'features.parquet')
    require(features.column_names == ['SK_ID_CURR', *contract['feature_columns'], 'sample_use']
            and features.num_rows == b['rows'] == r['scoring']['rows'] == acceptance['rows']['labeled']
            and set(features['sample_use'].to_pylist()) == {'train', 'validation', 'test'},
            'portable_population', 'Original complete labeled population required')
    validate_ids(features['SK_ID_CURR'].to_pylist()); validate_features(features.select(contract['feature_columns']), contract)
    labeled = ScoringSession(release.root, b, features, models, folder, r)
    release.sources['labeled'] = labeled
    ur, ub, uf = release.package('unlabeled'); ufeatures = pq.read_table(uf/'features.parquet')
    require(ub['contract'] == contract and ub['threshold'] == b['threshold']
            and ufeatures.column_names == features.column_names
            and ufeatures.num_rows == ub['rows'] == ur['unlabeled_scoring']['rows'] == acceptance['rows']['unlabeled']
            and set(ufeatures['sample_use'].to_pylist()) == {'unlabeled_scoring'},
            'portable_unlabeled', 'Original complete unlabeled population required')
    raw_path = str(origin.old/'data/raw/home_credit/application_test.csv')
    validate_ids(ufeatures['SK_ID_CURR'].to_pylist())
    validate_features(ufeatures.select(contract['feature_columns']), contract)
    unlabeled = make_session(labeled, ufeatures, ub['references'][raw_path], ur['run_id'])
    require(unlabeled.bundle['output_versions'] == ub['output_versions'] and not set(labeled.index) & set(unlabeled.index),
            'portable_unlabeled', 'Frozen versions or populations differ')
    validate_outputs(unlabeled, pq.read_table(uf/'scores.parquet'), pq.read_table(uf/'components.parquet'), 1e-10)
    unlabeled.directory, unlabeled.report = uf, ur
    release.sources['unlabeled'] = unlabeled
    er, eb, ef = release.package('explanation')
    require(eb['status'] == 'complete' and eb['method'] == 'training_count_weighted_path_log_odds'
            and read(ef/'feature_contract.json') == contract, 'portable_explanation', 'Matching original explanation required')
    release.explanation = PortableExplanation(release.root, eb['recipe'], release.sources,
                                              read(ef/'reference.json'), er, ef)
    require(release.explanation.explainer.reference() == acceptance['explanation_reference'],
            'portable_explanation', 'Accepted reference differs')
    mr, reference, mf = release.package('monitor')
    from .monitoring import POLICY, VERSION as MONITOR_VERSION
    require(reference['status'] == 'complete' and reference['version'] == MONITOR_VERSION
            and reference['purpose'] == 'train_only_descriptive_reference' and reference['policy'] == POLICY
            and reference['feature_columns'] == contract['feature_columns'] and reference['output_versions'] == b['output_versions']
            and reference['original_threshold'] == b['threshold'] and reference['rows'] == mr['monitoring_reference']['training_rows'],
            'portable_monitor', 'Matching original training-only monitor reference required')
    for spec in [*reference['features'].values(), reference['risk_probability']]:
        counts = spec['reference_counts']
        require(len(counts) == spec['buckets'] and all(type(x) is int and x >= 0 for x in counts)
                and sum(counts) == reference['rows'], 'portable_monitor', 'Reference counts differ')
    release.reference, release.reference_sha = reference, mr['monitoring_reference']['reference_sha256']
    return release


class PortableApiState(ApiState):
    """Production load path for the new release; no post-load session injection."""
    def __init__(self, directory, digest):
        self.directory, self.digest = Path(directory).resolve(), digest
        super().__init__(self.directory/'project', self.directory/'portable_manifest.json')

    def load(self):
        if self.phase != 'created': raise DataError('api_load_once', 'New immutable service instance required')
        self.phase = 'loading'; self.load_attempts += 1
        try:
            self.release = load_release(self.directory, self.digest)
            self.session = self.release.sources['labeled']
            # Keep the original scoring provenance in API/monitoring events;
            # the new release identity is an additional field in health.
            self.report_sha256 = self.release.manifest['acceptance']['reports']['scoring']['sha256']
            self.phase = 'ready'
        except Exception as exc:
            self.phase = 'failed'; self.error_code = exc.code if isinstance(exc, DataError) else type(exc).__name__
            raise
        return self

    def health(self):
        return {**super().health(), 'release_version': VERSION, 'release_sha256': self.digest}


class PortableDemoState(DemoState):
    def __init__(self, release):
        super().__init__(release.root, release.root/release.manifest['reports']['scoring'])
        self.session = release.sources['labeled']
        self.report_sha256 = release.manifest['acceptance']['reports']['scoring']['sha256']
        self.examples = [('验证示例', 100006), ('训练加总案例', 100003)]


def verify_replay(release):
    """Compare fixed train/validation and unlabeled examples; no test sampling."""
    results = {}
    for population, session in release.sources.items():
        features = session.features
        eligible = features if population == 'unlabeled' else features.filter(pc.not_equal(features['sample_use'], 'test'))
        ids = eligible['SK_ID_CURR'].to_pylist()[::max(1, eligible.num_rows//100)][:100]
        scores, terms = session.score_ids(ids)
        saved = pq.read_table(session.directory/'scores.parquet')
        components = pq.read_table(session.directory/'components.parquet')
        positions = pc.index_in(scores['SK_ID_CURR'], value_set=saved['SK_ID_CURR'])
        errors = {n:float(np.max(np.abs(scores[n].to_numpy()-saved[n].take(positions).to_numpy())))
                  for n in ['risk_probability', 'scorecard_probability', 'logistic_probability', 'scorecard_score', 'scorecard_base_points']}
        errors['components'] = max(float(np.max(np.abs(terms[n].to_numpy()-components[n].take(positions).to_numpy())))
                                   for n in session.contract['feature_columns'])
        require(max(errors.values()) <= 1e-10 and scores['risk_flag'].equals(saved['risk_flag'].take(positions))
                and scores['sample_use'].equals(saved['sample_use'].take(positions)), 'portable_replay', 'Saved scores differ')
        explanations = release.explanation.explain_ids(population, ids[:3])
        results[population] = {'ids': ids, 'rows': len(ids), 'max_absolute_errors': errors,
                               'explanation_cases': explanations}
    return {'passed': True, 'version': VERSION, 'release_sha256': release.manifest_sha256,
            'root': str(release.root), 'source_graph_JSON_files': release.graph_files,
            'populations': results, 'threshold': release.sources['labeled'].bundle['threshold'],
            'scope': 'saved_model_replay_no_training_or_new_effect_evaluation'}
