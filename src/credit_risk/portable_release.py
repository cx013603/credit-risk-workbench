"""F05 content-preserving release, with an externally pinned source namespace.

Legacy JSON is evidence, never rewritten. A new release authenticates the entire
accepted source closure before deserializing a model. Logical origin paths resolve
only through the manifest; they are never opened or used as fallback locations.
The producer runs the unchanged formal loaders. The consumer proves byte identity
to that accepted closure, checks the success/receipt/lineage graph, then uses the
shared scoring and explanation implementations. This is a new versioned loader,
not an assertion that the old absolute-path loaders have become relocatable.
"""
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
import sys
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

from .raw import DataError, sha256_file
from .recovery import safe_member
from .recovery_v11 import (no_links, metadata, encoded, save_new, unique_object,
                           copy_checked, require, MAX_BYTES, MAX_FILE, MAX_FILES)
from .runtime import now

VERSION = 'portable_release_v1'
MANIFEST = 'portable_manifest.json'
PURPOSE = 'accepted_frozen_content_relocation_no_training'
RUNTIME_FILES = ['src/credit_risk/portable_release.py', 'src/credit_risk/portable_runtime.py',
                 'scripts/run_platform.py', 'config/portable_release.json']
SUPPORT_FILES = ['scripts/guard_platform_entry.py', 'scripts/verify_portable_runtime.py',
                 'tests/test_portable_release.py', 'docs/44_portable_reproduction.md',
                 'deliverables/15_portable_reproduction.md']
SPECS = {
    'normalization': ('normalization', 'E03', 'official_full_normalization'),
    'history': ('history_features', 'E04', 'official_full_history_features'),
    'experiment': ('experiment', 'E05', 'official_full_experiment_contract'),
    'scoring': ('scoring', 'E10', 'official_full_scoring'),
    'unlabeled': ('unlabeled_scoring', 'E10_unlabeled', 'existing_unlabeled_applications_scoring'),
    'explanation': ('tree_explanation', 'E10_tree_explanation', 'official_tree_path_explanation'),
    'monitor': ('monitoring_reference', 'E15', 'official_monitoring_reference'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)


def member(name):
    p = safe_member(name)
    require(len(p.parts) >= 2 and p.parts[0] in {'project', 'wheelhouse', 'installers'},
            'portable_path', 'Supported release namespace required')
    require(not any(x.casefold() in {'.git', '.venv', '__pycache__', '.kaggle', 'credentials'}
                        or x.startswith('.env') or x == 'kaggle.json' for x in p.parts),
            'portable_path', 'Environment, credentials and links are not release payloads')
    # Only the F03 source CSV is needed by this frozen release. Full data rebuild
    # continues to require the separately retained original dataset.
    if p.parts[:3] == ('project', 'data', 'raw'):
        require(name == 'project/data/raw/home_credit/application_test.csv',
                'portable_raw', 'Only the explicitly bound unlabeled source CSV is included')
    return p


def _source_acceptance(root, reports):
    from .scoring import load_scoring_bundle
    from .unlabeled_scoring import load_unlabeled_bundle
    from .tree_explanation import load_explanation_bundle
    from .monitoring import load_reference
    from .experiments import verify_experiment_bundle
    from .history import load_history_bundle
    from .normalize import load_normalized_bundle
    # No alternate sessions and no substituted loaders at the export boundary.
    load_normalized_bundle(root, root/reports['normalization'])
    load_history_bundle(root, root/reports['history'])
    verify_experiment_bundle(root, root/reports['experiment'])
    s = load_scoring_bundle(root, root/reports['scoring'])
    u = load_unlabeled_bundle(root, root/reports['unlabeled'])
    e = load_explanation_bundle(root, root/reports['explanation'])
    reference, _ = load_reference(root, root/reports['monitor'], root/reports['scoring'])
    require(reference['output_versions'] == s.bundle['output_versions']
            and reference['feature_columns'] == s.contract['feature_columns']
            and reference['original_threshold'] == s.bundle['threshold'],
            'portable_reference', 'Original monitoring and scoring contract differ')
    return {'verified_at': now(), 'method': 'unchanged_complete_formal_source_loaders',
            'reports': {k: metadata(root/v) for k, v in reports.items()},
            'threshold': s.bundle['threshold'], 'contract': s.contract,
            'output_versions': s.bundle['output_versions'],
            'rows': {'labeled': s.features.num_rows, 'unlabeled': u.features.num_rows},
            'explanation_reference': e.explainer.reference()}


def make_plan(root):
    root = Path(root).resolve()
    config = read(root/'config/portable_release.json')
    require(config['version'] == VERSION and set(config['reports']) == set(SPECS),
            'portable_config', 'Explicit complete source reports required')
    acceptance = _source_acceptance(root, config['reports'])
    base_receipt = read(root/'docs/evidence/v11_recovery_official_integration_v2.json')['archive']
    receipts = [base_receipt, *[read(root/'reports'/n/'archive_receipt.json')
                                for n in ['unlabeled_validation', 'tree_explanation_validation']]]
    supplement = root/'.local-backups/credit-risk-recovery-v11-spark-checker-v2-20261026.zip'
    # Its hash is also pinned by the accepted v2 evidence, not an unverified file.
    v2 = read(root/'docs/evidence/v11_recovery_official_integration_v2.json')
    def find_hash(value):
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, dict) and v.get('sha256') and any(str(supplement.name) in str(x) for x in v.values()):
                    return v['sha256']
                result = find_hash(v)
                if result: return result
        elif isinstance(value, list):
            for v in value:
                result = find_hash(v)
                if result: return result
    expected = find_hash(v2)
    require(expected is not None and sha256_file(supplement) == expected,
            'portable_ancestor', 'Accepted Spark checker v2 supplement required')
    receipts.insert(1, {'archive': str(supplement), **metadata(supplement)})
    maintenance = read(root/'docs/evidence/local_services_recovery_engineering9_20261026.json')['maintenance_archive']
    receipts.insert(2, {'archive':maintenance['path'], 'sha256':maintenance['sha256']})
    files, ancestors = {}, []
    for i, receipt in enumerate(receipts):
        archive = no_links(receipt['archive'])
        require(sha256_file(archive) == receipt['sha256'], 'portable_ancestor', 'Accepted archive changed')
        ancestors.append({'archive': str(archive), **metadata(archive)})
        with ZipFile(archive) as z:
            require(z.testzip() is None, 'portable_ancestor', 'Ancestor CRC failure')
            for entry in z.infolist():
                if entry.is_dir(): continue
                safe_member(entry.filename)
                # Existing archive manifests retain their original identities as
                # provenance files; they are not treated as new release manifests.
                if '/' not in entry.filename and (entry.filename == 'manifest.json' or entry.filename.endswith('_manifest.json')):
                    name = 'project/reports/portable_ancestry/'+str(i)+'_'+entry.filename
                else:
                    name = entry.filename if entry.filename.split('/')[0] in {'project','wheelhouse','installers'} else 'project/'+entry.filename
                member(name)
                with z.open(entry) as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                files[name] = {'bytes': entry.file_size, 'sha256': digest,
                               'source': {'archive': i, 'member': entry.filename}}
    for relative in [*RUNTIME_FILES, *SUPPORT_FILES, 'data/raw/home_credit/application_test.csv']:
        path = no_links(root/relative)
        require(path.is_file(), 'portable_source', 'Explicit source missing')
        files['project/'+relative] = {**metadata(path), 'source': {'local': relative}}
    # The installed native implementation remains an external content binding;
    # an equivalent rebuilt environment is allowed only with identical bytes.
    from .tree_explanation import native_sources
    native = {}
    for p in native_sources():
        path = Path(p)
        name = path.name
        native[str(path)] = {'module': 'sklearn.ensemble._hist_gradient_boosting.'+name.split('.')[0],
                             'sha256': sha256_file(path)}
    m = {'version': VERSION, 'purpose': PURPOSE, 'created_at': now(), 'original_root': str(root),
         'environment': read(root/'reports/v11_recovery/plan_v1.json')['environment'],
         'ancestors': ancestors, 'reports': config['reports'], 'acceptance': acceptance,
         'native': native, 'files': dict(sorted(files.items())),
         'contract': 'original_bytes_complete_closure_then_shared_runtime_v1',
         'scope': 'frozen_replay_labeled_unlabeled_explanation_page_API_monitoring',
         'training_or_new_assessment': False}
    require(len(files) <= MAX_FILES and sum(v['bytes'] for v in files.values()) <= MAX_BYTES,
            'portable_quota', 'Release exceeds original archive quotas')
    # Ensure every needed origin hash is present before publishing the plan.
    Origin(root, m, producer=True).check_graph()
    return m


def inspect_archive(z):
    entries = z.infolist(); names = [e.filename for e in entries]
    require(0 < len(entries) <= MAX_FILES and len(set(n.casefold() for n in names)) == len(names)
            and sum(e.file_size for e in entries) <= MAX_BYTES, 'portable_quota', 'Duplicate or excessive archive')
    keys = {n.casefold() for n in names}
    for e in entries:
        p = safe_member(e.filename)
        require(not e.is_dir() and stat.S_IFMT(e.external_attr >> 16) in (0, stat.S_IFREG)
                and not e.flag_bits & 1, 'portable_type', 'Unencrypted regular files required')
        require(e.file_size <= MAX_FILE and (e.file_size < 1024**2 or e.file_size <= max(1, e.compress_size)*1000)
                and not any(str(x).casefold() in keys for x in p.parents if str(x) != '.'),
                'portable_quota', 'File quota or file/directory collision')
    require(MANIFEST in names and z.getinfo(MANIFEST).file_size <= 8*1024**2,
            'portable_manifest', 'Bounded manifest required')
    m = json.loads(z.read(MANIFEST), object_pairs_hook=unique_object)
    validate_manifest(m)
    require(set(names) == set(m['files']) | {MANIFEST}, 'portable_manifest', 'Missing or extra payload')
    for n, v in m['files'].items():
        require(z.getinfo(n).file_size == v['bytes'], 'portable_manifest', 'Member size differs')
    return m


def validate_manifest(m):
    require(m.get('version') == VERSION and m.get('purpose') == PURPOSE
            and m.get('contract') == 'original_bytes_complete_closure_then_shared_runtime_v1'
            and m.get('training_or_new_assessment') is False and set(m['reports']) == set(SPECS)
            and isinstance(m['files'], dict) and 0 < len(m['files']) <= MAX_FILES,
            'portable_manifest', 'Unsupported or incomplete source contract')
    require(len({n.casefold() for n in m['files']}) == len(m['files'])
            and sum(v['bytes'] for v in m['files'].values()) <= MAX_BYTES,
            'portable_quota', 'Duplicate or excessive files')
    for n, v in m['files'].items():
        member(n)
        require(type(v['bytes']) is int and 0 <= v['bytes'] <= MAX_FILE
                and isinstance(v['sha256'], str) and re.fullmatch('[0-9a-f]{64}', v['sha256']),
                'portable_manifest', 'Invalid metadata')
    require(all('project/'+n in m['files'] for n in RUNTIME_FILES),
            'portable_runtime', 'Complete versioned runtime required')


def build(root, plan, digest, output):
    root, output = Path(root).resolve(), no_links(output)
    require(sha256_file(plan) == digest, 'portable_plan', 'Externally pinned plan required')
    m = read(plan); validate_manifest(m)
    require(m['original_root'] == str(root), 'portable_source', 'Producer root differs')
    require(not output.exists() and output.parent.is_dir(), 'portable_exists', 'New archive required')
    partial = output.with_name(output.name+'.partial'); created = False
    try:
        with ExitStack() as stack:
            archives = []
            for a in m['ancestors']:
                require(sha256_file(no_links(a['archive'])) == a['sha256'], 'portable_ancestor', 'Ancestor changed')
                archives.append(stack.enter_context(ZipFile(a['archive'])))
            raw = stack.enter_context(partial.open('xb')); created = True
            with ZipFile(raw, 'w', compression=ZIP_DEFLATED, compresslevel=3) as z:
                for n, v in m['files'].items():
                    with ExitStack() as pair:
                        src = v['source']
                        source = pair.enter_context(archives[src['archive']].open(src['member']) if 'archive' in src
                                                    else no_links(root/src['local']).open('rb'))
                        out = pair.enter_context(z.open(n, 'w', force_zip64=True))
                        copy_checked(source, out, v)
                z.writestr(MANIFEST, encoded(m))
            raw.flush(); os.fsync(raw.fileno())
        verify_archive(partial, sha256_file(partial))
        if os.name == 'nt': os.rename(partial, output)
        else: os.link(partial, output); partial.unlink()
        created = False
        return {'archive': str(output), **metadata(output), 'manifest_sha256': hashlib.sha256(encoded(m)).hexdigest(),
                'files': len(m['files']), 'created_at': now()}
    finally:
        if created and partial.exists(): partial.unlink()


def verify_archive(archive, digest, destination=None):
    archive = no_links(archive)
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
            and sha256_file(archive) == digest, 'portable_hash', 'External trusted archive SHA256 required')
    dest = no_links(destination) if destination is not None else None
    if dest: require(not dest.exists() and dest.parent.is_dir(), 'portable_exists', 'Entirely new target required')
    created = False
    try:
        with ZipFile(archive) as z:
            m = inspect_archive(z)
            if dest: dest.mkdir(); created = True
            for n, v in m['files'].items():
                with ExitStack() as pair:
                    source = pair.enter_context(z.open(n))
                    if dest:
                        target = dest/n; target.parent.mkdir(parents=True, exist_ok=True)
                        out = pair.enter_context(target.open('xb'))
                    else: out = pair.enter_context(open(os.devnull, 'wb'))
                    copy_checked(source, out, v)
            if dest: (dest/MANIFEST).write_bytes(z.read(MANIFEST))
            return m
    except (BadZipFile, EOFError) as exc:
        raise DataError('portable_zip', 'Corrupt or truncated archive') from exc
    finally:
        if created and sys.exc_info()[0] is not None:
            require(no_links(dest).resolve() == dest, 'portable_cleanup', 'Cleanup path changed')
            shutil.rmtree(dest)


class Origin:
    """Total, allowlisted translation from historical origin to present content."""
    def __init__(self, directory, manifest, producer=False):
        self.directory, self.m = Path(directory).resolve(), manifest
        self.root = self.directory if producer else self.directory/'project'
        self.producer = producer
        self.old = PureWindowsPath(manifest['original_root'])

    def key(self, path):
        p = PureWindowsPath(path)
        require(p.is_absolute(), 'portable_origin', 'Historical absolute origin required')
        try: relative = p.relative_to(self.old).as_posix()
        except ValueError as exc: raise DataError('portable_origin', 'Origin outside declared project') from exc
        name = 'project/'+relative; member(name)
        require(name in self.m['files'], 'portable_missing', 'Origin absent from complete release: '+relative)
        return name

    def path(self, path):
        name = self.key(path)
        p = self.root/name.removeprefix('project/')
        no_links(p)
        require(p.resolve().is_relative_to(self.root), 'portable_origin', 'Resolved origin escaped target')
        return p

    def json(self, path): return read(self.path(path))

    def check_hash(self, path, digest):
        if path in self.m['native']:
            require(self.m['native'][path]['sha256'] == digest, 'portable_native', 'Native binding differs')
        else:
            name = self.key(path)
            require(self.m['files'][name]['sha256'] == digest, 'portable_lineage', 'Source hash is not closed: '+name)

    def check_graph(self):
        from .experiments import digest_json
        old = str(self.old)
        # Validate all active successful reports before model deserialization.
        for role, (key, module, scope) in SPECS.items():
            path = str(self.old/self.m['reports'][role]); r = self.json(path)
            require(r['status'] == 'passed' and r['metadata']['module'] == module and r['scope'] == scope,
                    'portable_success', 'Accepted success scope required: '+role)
            self.check_hash(path, self.m['acceptance']['reports'][role]['sha256'])
            info = r[key]; folder = info.get('artifact_dir', info.get('directory'))
            if role == 'monitor':
                filename, digest = 'reference.json', info['reference_sha256']
            else: filename, digest = 'bundle.json', info['bundle_sha256']
            self.check_hash(str(PureWindowsPath(folder)/filename), digest)
            if role in {'scoring', 'unlabeled', 'explanation', 'monitor'}:
                receipt = self.json(str(PureWindowsPath(folder)/'completed.json'))
                require(receipt['report'] == path and receipt['report_sha256'] == self.m['files'][self.key(path)]['sha256'],
                        'portable_receipt', 'Original stable completion receipt differs')
                self.check_hash(str(PureWindowsPath(folder)/filename), receipt.get('bundle_sha256', receipt.get('reference_sha256')))
        # Walk only JSON in the active source graph, not historical failure reports.
        active = set()
        def walk(path):
            name = self.key(path)
            if name in active: return
            active.add(name); value = self.json(path)
            folder = PureWindowsPath(path).parent
            if PureWindowsPath(path).name == 'run.json':
                require(value['status'] == 'passed' and not value['scope'].startswith('synthetic_'),
                        'portable_success', 'All active source runs must be successful official runs')
            if PureWindowsPath(path).name == 'bundle.json':
                require(value.get('status') == 'complete', 'portable_success', 'Complete source artifact required: '+path)
            def visit(v):
                if isinstance(v, dict):
                    if isinstance(v.get('files'), dict) and all(isinstance(x, dict) and 'sha256' in x and 'bytes' in x for x in v['files'].values()):
                        for n, meta in v['files'].items():
                            output = PureWindowsPath(v.get('artifact_dir', str(folder)))
                            q = str(output/n); self.check_hash(q, meta['sha256'])
                            require(self.m['files'][self.key(q)]['bytes'] == meta['bytes'], 'portable_lineage', 'Size differs')
                            if n.endswith('.json'): walk(q)
                    for key in ['references', 'source_sha256', 'protected_sha256', 'code_sha256']:
                        refs = v.get(key)
                        if isinstance(refs, dict):
                            for p, h in refs.items():
                                if isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h):
                                    q = p if PureWindowsPath(p).is_absolute() else str(self.old/p)
                                    self.check_hash(q, h)
                    for k, h in v.items():
                        if isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) and k.endswith('_sha256'):
                            prefix = k[:-7]
                            p = v.get(prefix+'_path', v.get(prefix))
                            if isinstance(p, str) and PureWindowsPath(p).is_absolute():
                                self.check_hash(p, h)
                                if p.endswith('.json') and p not in self.m['native']: walk(p)
                    if 'payload' in v and isinstance(v['payload'], dict):
                        fingerprint = v.get('fingerprint', v.get('payload_fingerprint'))
                        if fingerprint: require(digest_json(v['payload']) == fingerprint, 'portable_fingerprint', 'Payload fingerprint differs')
                    for x in v.values(): visit(x)
                elif isinstance(v, list):
                    for x in v: visit(x)
            visit(value)
        for role in SPECS:
            path = str(self.old/self.m['reports'][role]); walk(path)
            info = self.json(path)[SPECS[role][0]]
            folder = info.get('artifact_dir', info.get('directory'))
            walk(str(PureWindowsPath(folder)/('reference.json' if role == 'monitor' else 'bundle.json')))
        plan = self.json(str(self.old/'config/experiment_plan.json'))
        require(plan['status'] == 'frozen' and plan['plan_payload_sha256'] == digest_json({k:v for k,v in plan.items() if k != 'plan_payload_sha256'}),
                'portable_frozen', 'Original frozen plan required')
        self.check_hash(plan['active_report_path'], plan['active_report_sha256'])
        return len(active)


def verify_directory(directory, digest):
    directory = no_links(directory).resolve(); manifest = directory/MANIFEST
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
            and sha256_file(manifest) == digest, 'portable_hash', 'External trusted manifest SHA256 required')
    m = read(manifest); validate_manifest(m)
    for n, meta in m['files'].items():
        p = no_links(directory/n)
        require(p.is_file() and metadata(p) == {k:meta[k] for k in ['sha256', 'bytes']},
                'portable_file', 'Release content changed: '+n)
    # Pin actual imported application sources, not merely unused source copies.
    root = directory/'project'
    for name, module in list(sys.modules.items()):
        if name.startswith('credit_risk.') and getattr(module, '__file__', None):
            p = Path(module.__file__).resolve()
            require(p.is_relative_to(root/'src/credit_risk'), 'portable_import', 'Runtime imported outside restored project: '+name)
            key = 'project/'+p.relative_to(root).as_posix()
            require(key in m['files'] and metadata(p) == {k:m['files'][key][k] for k in ['bytes','sha256']},
                    'portable_import', 'Imported application source is not in the authenticated closure: '+name)
    from .logistic import environment
    for n, v in m['environment'].items():
        actual = sys.version.split()[0] if n == 'python' else importlib.metadata.version(n)
        require(actual == v, 'portable_environment', 'Rebuild exact locked environment: '+n)
    for old, v in m['native'].items():
        p = Path(inspect.getfile(importlib.import_module(v['module']))).resolve()
        require(sha256_file(p) == v['sha256'], 'portable_native', 'Native implementation changed')
    origin = Origin(directory, m); graph = origin.check_graph()
    return origin, graph
