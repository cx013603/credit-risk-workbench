"""Versioned, explicit V1.1 recovery; accepted V1 producers stay untouched.

Archives are local trusted snapshots, not a transport for untrusted pickle files.
The external SHA256 is the trust anchor. Historical absolute lineage is preserved.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

from .raw import DataError, sha256_file
from .recovery import safe_member
from .runtime import now

VERSION = 'recovery_v11_explicit_v1'
PURPOSE = 'trusted_local_V11_recovery'
MANIFEST = 'recovery_manifest.json'
MAX_BYTES = 2 * 1024**3
MAX_FILE = 600 * 1024**2
MAX_FILES = 10000
MAX_MANIFEST = 8 * 1024**2
CHUNK = 1024**2
STATIC = {'src', 'scripts', 'config', 'sql', 'app', 'tests', 'docs', 'deliverables'}
MUTABLE_DOCS = {'STATUS.md', 'HANDOFF.md', 'README.md', 'docs/13_task_board.md',
                'docs/14_acceptance.md', 'docs/03_learning_log.md', 'docs/05_architecture.md',
                'docs/07_tools.md', 'docs/17_engineering_plan.md', 'docs/decisions.md',
                'deliverables/README.md'}
ACTIVE_RUNS = {'20261026-112152-fd575e25', '20261026-152623-1930c5ec'}


def require(value, code, message):
    if not value:
        raise DataError(code, message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def save_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def no_links(path):
    """Check lexical ancestors before resolve, including Windows junctions."""
    path = Path(os.path.abspath(path))
    for p in (path, *path.parents):
        require(not p.is_symlink() and not p.is_junction(), 'v11_link', 'Links/junctions are not recovery files or destinations')
    return path


def local_file(root, relative):
    safe_member(relative)
    p = no_links(Path(root)/relative)
    require(p.resolve().is_relative_to(Path(root).resolve()) and p.is_file(), 'v11_source', 'Explicit regular project file required: '+relative)
    require(not any(x.casefold() in {'.git', '.venv', 'credentials', '.kaggle', '__pycache__'} for x in p.relative_to(root).parts)
            and not p.name.startswith('.env') and p.name != 'kaggle.json', 'v11_source', 'Excluded source')
    return p


def metadata(path):
    return {'sha256': sha256_file(path), 'bytes': Path(path).stat().st_size}


def member_allowed(name):
    p = safe_member(name)
    require(len(p.parts) >= 2 and p.parts[0] in {'project', 'wheelhouse', 'installers'}, 'v11_namespace', 'Unsupported recovery namespace')
    require(not set(p.parts) & ACTIVE_RUNS and not any(x.casefold() in {'.git', '.venv', 'credentials', '.kaggle', '__pycache__'} for x in p.parts)
            and not any(x.startswith('.env') or x == 'kaggle.json' for x in p.parts), 'v11_excluded', 'Excluded active run or sensitive content')
    sample_csv = p.parts[:3] == ('project', 'data', 'sample') or p.parts[:3] == ('project', 'tests', 'fixtures')
    require(not (p.parts[:3] == ('project', 'data', 'raw')) and not (p.parts[0] == 'project' and (p.suffix.lower() in {'.zip', '.whl', '.pyc'} or (p.suffix.lower() == '.csv' and not sample_csv))),
            'v11_excluded', 'Raw data and media must not be project payloads')
    return p


def make_plan(root):
    root = Path(root).resolve()
    proof = read(root/'docs/evidence/recovery_official_integration.json')
    archive = no_links(proof['archive'])
    require(sha256_file(archive) == proof['archive_sha256'], 'v11_base', 'Accepted V1 archive changed')
    with ZipFile(archive) as z:
        raw = z.read(MANIFEST)
        require(hashlib.sha256(raw).hexdigest() == proof['manifest_sha256'], 'v11_base', 'Accepted V1 manifest changed')
        base = json.loads(raw)
    files = {n: {**v, 'source': 'base', 'origin': 'accepted_V1_archive'} for n,v in base['files'].items()}

    def add(relative, origin, expected=None):
        p = local_file(root, relative)
        meta = metadata(p)
        require(expected is None or meta['sha256'] == expected, 'v11_seal', 'Accepted immutable file changed: '+relative)
        name = 'project/'+relative
        member_allowed(name)
        files[name] = {**meta, 'source': relative, 'origin': origin}

    # Static trees only: never recursively collect reports, environments or data.
    for folder in sorted(STATIC):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() not in {'.pyc', '.zip', '.whl'}:
                add(p.relative_to(root).as_posix(), 'current_static_project')
    for p in [*root.glob('*.md'), *root.glob('requirements*.txt'), root/'.gitignore']:
        add(p.relative_to(root).as_posix(), 'current_project_control')

    seals, deltas = [], []
    for evidence, key, hashkey in [
        ('docs/evidence/spark_official_integration.json', 'inventory', 'inventory_sha256'),
        ('docs/evidence/scoring_api_delivery_checks.json', 'inventory', 'inventory_sha256'),
        ('docs/evidence/monitoring_final_checks.json', 'final_inventory', 'final_inventory_sha256')]:
        v = read(root/evidence)
        p = Path(v[key])
        relative = p.relative_to(root).as_posix()
        add(relative, 'accepted_component_inventory', v[hashkey])
        inventory = read(p)
        listed = inventory.get('files', inventory)
        for n, item in listed.items():
            n = n.replace('\\', '/')
            expected = item if isinstance(item, str) else item['sha256']
            if n in MUTABLE_DOCS:
                current = sha256_file(root/n)
                if current != expected:
                    deltas.append({'path': n, 'historical_sha256': expected, 'current_sha256': current,
                                   'reason': 'authorized shared documentation; historical inventory is not resigned'})
                add(n, 'current_shared_document')
            else:
                add(n, 'accepted_component_inventory', expected)
        seals.append({'path': relative, 'sha256': v[hashkey], 'listed_files': len(listed)})

    # Bounded supplemental evidence: saved installation attempts, authorizations,
    # protections and reports directly referred to by component evidence.
    for relative in ['reports/e13_validation/resume_authorization.json', 'reports/e13_validation/windows_runtime_alias.json',
                     'reports/e13_validation/before.json', 'reports/e14_validation/before.json', 'reports/e15_validation/before.json',
                     'reports/v11_delivery_preparation/chat_registry_before_engineering9.json']:
        add(relative, 'accepted_history_or_protection')
    for n in ['environment_preparation.json', 'environment_preparation_retry1.json', 'environment_preparation_retry2.json',
              'new_environment_checks.json', 'environment_preparation_retry2_pip_install.json']:
        if (root/'reports/e13_validation'/n).is_file():
            add('reports/e13_validation/'+n, 'Spark_installation_history')
    # API run receipts are not part of its old flat inventory.
    api = read(root/'docs/evidence/scoring_api_official_integration_v3.json')
    for k in ['formal_report', 'restart_report']:
        if k in api:
            p = Path(api[k]); add(p.relative_to(root).as_posix(), 'accepted_API_run_report')
    fixture = read(root/'docs/evidence/spark_runtime_validation_v2.json')
    fixture_report = Path(fixture['report'])
    add(fixture_report.relative_to(root).as_posix(), 'accepted_Spark_fixture_report', fixture['report_sha256'])
    fixture_id = fixture_report.parent.name
    for folder in [fixture_report.parent, root/'reports/e13_validation/fixtures'/fixture_id,
                   root/'reports/e13_validation/fixtures_out'/fixture_id]:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): add(p.relative_to(root).as_posix(), 'accepted_Spark_fixture_evidence')
    for evidence in ['docs/evidence/scoring_api_delivery_checks.json','docs/evidence/monitoring_delivery_checks.json']:
        def add_failure_reports(obj):
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if k=='report' and isinstance(v,str):
                        p=Path(v)
                        if p.is_absolute() and p.is_relative_to(root/'reports/runs') and p.name=='run.json':
                            require(read(p)['status']!='running','v11_running','Do not collect live run reports')
                            add(p.relative_to(root).as_posix(),'historical_failure_report')
                    elif isinstance(v,(dict,list)): add_failure_reports(v)
            elif isinstance(obj,list):
                for v in obj:add_failure_reports(v)
        add_failure_reports(read(root/evidence))
    coverage = read(root/'docs/evidence/v11_recovery_coverage_audit.json')
    for group in coverage['groups'].values():
        for v in group['items']:
            add(v['path'], 'explicit_V11_required', v['current_sha256'])
    media = []
    preparation = read(root/'docs/evidence/spark_environment_preparation.json')
    for name, v in preparation['artifacts_verified'].items():
        p = local_file(root, v['file'])
        require(metadata(p) == {k:v[k] for k in ['sha256','bytes']}, 'v11_media', 'Accepted installer changed')
        member = 'installers/spark/'+p.name
        files[member] = {**metadata(p), 'source': v['file'], 'origin': 'accepted_Spark_installer'}
        media.append({'name': name, 'member': member, **v, 'version': preparation['versions'].get(name, preparation['java_version'])})
    for n in files:
        member_allowed(n)
    require(sum(v['bytes'] for v in files.values()) <= MAX_BYTES and len(files) < MAX_FILES, 'v11_quota', 'Recovery plan exceeds quota')
    return {'version': VERSION, 'purpose': PURPOSE, 'created_at': now(), 'original_root': str(root),
            'base_archive': str(archive), 'base_archive_sha256': proof['archive_sha256'], 'base_manifest_sha256': proof['manifest_sha256'],
            'environment': base['environment'], 'models': base['models'], 'scoring_directory': base['scoring_directory'],
            'component_seals': seals, 'shared_document_deltas': deltas, 'Spark_installers': media,
            'excluded_active_runs': sorted(ACTIVE_RUNS), 'external_alias_record_only': True,
            'raw_CSV_included': False, 'official_bundle_relocation_supported': False,
            'reports_are_historical_records_not_all_passed': True, 'files': dict(sorted(files.items()))}


def validate_manifest(m, infos):
    require(isinstance(m, dict) and m.get('version') == VERSION and m.get('purpose') == PURPOSE
            and isinstance(m.get('files'), dict) and bool(m['files']), 'v11_manifest', 'Complete V1.1 manifest required')
    require(set(infos) == set(m['files']) | {MANIFEST}, 'v11_manifest', 'Archive and manifest members must match exactly')
    for n, v in m['files'].items():
        member_allowed(n)
        require(isinstance(v, dict) and type(v.get('bytes')) is int and 0 <= v['bytes'] <= MAX_FILE
                and isinstance(v.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', v['sha256'])
                and infos[n].file_size == v['bytes'], 'v11_manifest', 'Invalid file metadata')


def inspect(z):
    entries = z.infolist()
    names = [e.filename for e in entries]
    require(0 < len(entries) <= MAX_FILES and len(set(n.casefold() for n in names)) == len(names)
            and sum(e.file_size for e in entries) <= MAX_BYTES, 'v11_quota', 'Duplicate members or archive quota exceeded')
    keys = {n.casefold() for n in names}
    for e in entries:
        p = safe_member(e.filename)
        require(not e.is_dir() and stat.S_IFMT(e.external_attr >> 16) in (0, stat.S_IFREG)
                and not e.flag_bits & 1, 'v11_link', 'Only unencrypted regular files accepted')
        require(e.file_size <= MAX_FILE and (e.file_size < 1024**2 or e.file_size <= max(1,e.compress_size)*1000),
                'v11_quota', 'File size/compression quota exceeded')
        require(not any(str(x).casefold() in keys for x in p.parents if str(x) != '.'), 'v11_path', 'File/directory collision')
    infos = {e.filename:e for e in entries}
    require(MANIFEST in infos and infos[MANIFEST].file_size <= MAX_MANIFEST, 'v11_manifest', 'Bounded manifest required')
    try:
        m = json.loads(z.read(MANIFEST), object_pairs_hook=unique_object)
        validate_manifest(m, infos)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise DataError('v11_manifest', 'Invalid recovery manifest') from exc
    return m


def unique_object(pairs):
    result = {}
    for k,v in pairs:
        if k in result:
            raise ValueError('duplicate JSON key')
        result[k] = v
    return result


def copy_checked(source, out, expected):
    total, digest = 0, hashlib.sha256()
    while block := source.read(CHUNK):
        total += len(block)
        require(total <= expected['bytes'] and total <= MAX_FILE, 'v11_size', 'Stream exceeds declared size')
        digest.update(block)
        out.write(block)
    require(total == expected['bytes'] and digest.hexdigest() == expected['sha256'], 'v11_file', 'Payload size/SHA256 mismatch')


def build(root, plan_path, expected_plan_sha256, output):
    root, output = Path(root).resolve(), no_links(output)
    require(sha256_file(plan_path) == expected_plan_sha256, 'v11_plan', 'Exact frozen plan required')
    m = read(plan_path)
    require(m['version'] == VERSION and m['purpose'] == PURPOSE and m['original_root'] == str(root), 'v11_plan', 'Matching project plan required')
    require(not output.exists() and output.parent.is_dir(), 'v11_exists', 'New archive destination required')
    base = no_links(m['base_archive'])
    require(sha256_file(base) == m['base_archive_sha256'], 'v11_base', 'Base archive changed')
    partial = output.with_name(output.name+'.partial')
    created = False
    try:
        with partial.open('xb') as raw:
            created = True
            with ZipFile(base) as old, ZipFile(raw, 'w', compression=ZIP_DEFLATED, compresslevel=3) as z:
                for n,v in m['files'].items():
                    member_allowed(n)
                    with ExitStack() as stack:
                        if v['source'] == 'base':
                            source = stack.enter_context(old.open(n))
                        else:
                            source = stack.enter_context(local_file(root,v['source']).open('rb'))
                        out = stack.enter_context(z.open(n,'w',force_zip64=True))
                        copy_checked(source, out, v)
                z.writestr(MANIFEST, encoded(m))
            raw.flush(); os.fsync(raw.fileno())
        # Independent read pass checks CRC and every content hash before publication.
        verify_archive(partial, sha256_file(partial))
        # Windows rename refuses an existing target; POSIX uses an exclusive
        # hard-link publication because its rename would replace the target.
        if os.name == 'nt':
            os.rename(partial, output)
        else:
            os.link(partial, output); partial.unlink()
        created = False
        return {'archive': str(output), **metadata(output), 'plan_sha256': expected_plan_sha256,
                'manifest_sha256': hashlib.sha256(encoded(m)).hexdigest(), 'files': len(m['files']),
                'uncompressed_bytes': sum(v['bytes'] for v in m['files'].values()), 'created_at': now()}
    finally:
        if created and partial.exists():
            partial.unlink()


def verify_archive(path, expected_sha256, destination=None):
    path = no_links(path)
    require(isinstance(expected_sha256,str) and re.fullmatch('[0-9a-f]{64}',expected_sha256)
            and sha256_file(path) == expected_sha256, 'v11_hash', 'External trusted archive SHA256 required')
    dest = no_links(destination) if destination is not None else None
    if dest:
        require(not dest.exists() and dest.parent.is_dir(), 'v11_exists', 'Restore only into a new directory with an existing parent')
    created = False
    try:
        with ZipFile(path) as z:
            m = inspect(z)
            if dest:
                dest.mkdir(); created = True
            for n,v in m['files'].items():
                with ExitStack() as stack:
                    source = stack.enter_context(z.open(n))
                    if dest:
                        target = dest/n
                        target.parent.mkdir(parents=True,exist_ok=True)
                        out = stack.enter_context(target.open('xb'))
                    else:
                        out = stack.enter_context(open(os.devnull,'wb'))
                    copy_checked(source,out,v)
            if dest:
                (dest/MANIFEST).write_bytes(z.read(MANIFEST))
            return m
    except (BadZipFile, EOFError) as exc:
        raise DataError('v11_zip', 'Corrupt or truncated archive') from exc
    finally:
        if created and sys_exception():
            require(dest.resolve() == dest and not dest.is_junction() and not dest.is_symlink(), 'v11_cleanup', 'Cleanup target changed')
            shutil.rmtree(dest)


def sys_exception():
    import sys
    return sys.exc_info()[0] is not None
