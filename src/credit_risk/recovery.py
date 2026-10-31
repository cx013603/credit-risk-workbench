"""E12 trusted local recovery archive and isolated recovered-source/model probe.

Legacy official bundles retain original absolute lineage. This does not rewrite
them into portable official packages or authorize foreign pickle deserialization.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import uuid
from zipfile import ZipFile,ZIP_DEFLATED

from .reproduction import audit
from .raw import DataError,sha256_file
from .runtime import RunRecord,load_config,now

VERSION='recovery_v1'
MAX_BYTES=2*1024**3


def safe_member(value):
    if not isinstance(value,str):raise DataError('recovery_path','Archive member must be text')
    p=PurePosixPath(value)
    if not value or not p.parts or p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value or str(p)!=value or any(
        part.endswith((' ','.')) or any(ord(c)<32 or c in '<>"|?*' for c in part) or
        re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part) for part in p.parts):
        raise DataError('recovery_path','Unsafe archive member')
    return p


def snapshot(root,v):
    root=Path(root).resolve();summary,context=audit(root,v);files=set()
    for name in ['src','scripts','config','sql','app','tests','docs','deliverables','data/sample','models','reports/recipes','reports/selection','reports/runs','reports/e12_validation']:
        folder=root/name
        if folder.exists():files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.whl','.zip'])
    files.update(root.glob('*.md'));files.update(root.glob('requirements*.txt'));files.add(root/'.gitignore')
    for obj in [context['normal'],context['history'],context['experiment'],context['session']]:
        directory=obj.directory if hasattr(obj,'directory') else obj['directory']
        files.update(p for p in directory.iterdir() if p.is_file())
    assessment=json.loads((root/v['assessment_report']).read_text(encoding='utf-8'))
    files.update(p for p in Path(assessment['assessment']['artifact_dir']).iterdir() if p.is_file())
    mapping={}
    for p in sorted(files):
        resolved=p.resolve()
        if not resolved.is_relative_to(root) or any(x in resolved.parts for x in ['credentials','.venv','.git']):raise DataError('recovery_source_path','Archive source must stay in project allowlist')
        if p.name.startswith('.env') or p.name=='kaggle.json':continue
        mapping['project/'+p.relative_to(root).as_posix()]=p
    wheelhouse=root/'.local-tmp/e12-repro/wheelhouse'
    wheel_expected={}
    for name,key in [('data_dependencies.json','installed'),('model_dependencies.json','newly_installed')]:
        for item in json.loads((root/'docs/evidence'/name).read_text(encoding='utf-8'))[key]:wheel_expected[Path(item['wheel_url']).name]=item['wheel_sha256']
    for name,h in wheel_expected.items():
        p=wheelhouse/name
        if sha256_file(p)!=h:raise DataError('recovery_wheel','Exact accepted offline wheels required')
        mapping['wheelhouse/'+name]=p
    session=context['session'];models={}
    for name,obj in session.models.items():models[name]=obj['directory'].relative_to(root).as_posix()
    manifest={'version':VERSION,'purpose':'trusted_local_project_recovery_snapshot','created_at':now(),'original_root':str(root),
        'reports_are_historical_records_not_all_passed':True,
        'environment':{**session.bundle['environment'],**{n:importlib.metadata.version(n) for n in ['duckdb','cloudpickle','narwhals']}},
        'scoring_directory':session.directory.relative_to(root).as_posix(),'models':models,
        'official_bundle_relocation_supported':False,'raw_CSV_included':False,'offline_wheels_included':True,
        'files':{name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for name,p in mapping.items()}}
    if sum(v['bytes'] for v in manifest['files'].values())>MAX_BYTES:raise DataError('recovery_size','Snapshot exceeds declared 2 GiB limit')
    output=root/'.local-backups'/('credit-risk-recovery-'+uuid.uuid4().hex+'.zip');output.parent.mkdir(exist_ok=True)
    partial=output.with_suffix('.zip.partial')
    try:
        with ZipFile(partial,'x',compression=ZIP_DEFLATED,compresslevel=3) as archive:
            for name,p in sorted(mapping.items()):
                if sha256_file(p)!=manifest['files'][name]['sha256']:raise DataError('recovery_changed','Source changed during snapshot')
                archive.write(p,name)
            archive.writestr('recovery_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        with ZipFile(partial) as archive:
            if archive.testzip() is not None:raise DataError('recovery_crc','Snapshot CRC failed')
        for name,p in mapping.items():
            if sha256_file(p)!=manifest['files'][name]['sha256']:raise DataError('recovery_changed','Source changed before snapshot publication')
        partial.rename(output)
        result={'archive':str(output),'archive_sha256':sha256_file(output),'manifest_sha256':hashlib.sha256((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')).hexdigest(),
            'files':len(mapping),'uncompressed_bytes':sum(v['bytes'] for v in manifest['files'].values()),'archive_bytes':output.stat().st_size,
            'original_root':str(root),'raw_CSV_included':False,'offline_wheels':len(wheel_expected),'legacy_absolute_paths_preserved':True}
        with output.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        return result
    except BaseException:
        if partial.exists():partial.unlink()
        raise


def extract_verified(archive_path,expected_sha256,destination):
    archive_path,destination=Path(archive_path),Path(destination).resolve()
    if not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256) or sha256_file(archive_path)!=expected_sha256:raise DataError('recovery_hash','Trusted exact archive SHA256 required')
    if destination.exists():raise DataError('recovery_exists','Never overwrite a restore directory')
    with ZipFile(archive_path) as archive:
        entries=archive.infolist();names=[x.filename for x in entries]
        if len(entries)>10000 or len(names)!=len(set(n.casefold() for n in names)) or sum(e.file_size for e in entries)>MAX_BYTES:raise DataError('recovery_size','Duplicate or excessive archive content')
        for entry in entries:
            safe_member(entry.filename)
            mode=stat.S_IFMT(entry.external_attr>>16)
            if mode not in (0,stat.S_IFREG) or entry.is_dir() or entry.flag_bits&1:raise DataError('recovery_link','Only explicit unencrypted regular files accepted')
            if any(str(parent).casefold() in {n.casefold() for n in names} for parent in PurePosixPath(entry.filename).parents if str(parent)!='.'):
                raise DataError('recovery_path','File/directory collision')
        try:
            if archive.getinfo('recovery_manifest.json').file_size>4*1024**2:raise ValueError('manifest too large')
            manifest=json.loads(archive.read('recovery_manifest.json'))
            if manifest['version']!=VERSION or manifest['purpose']!='trusted_local_project_recovery_snapshot' or not isinstance(manifest['files'],dict) or not manifest['files'] or set(names)!=set(manifest['files'])|{'recovery_manifest.json'}:raise ValueError('incomplete manifest')
            for name,meta in manifest['files'].items():
                member=safe_member(name)
                if member.parts[0] not in ('project','wheelhouse') or len(member.parts)<2 or type(meta['bytes']) is not int or meta['bytes']<0 or not isinstance(meta['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',meta['sha256']) or archive.getinfo(name).file_size!=meta['bytes']:raise ValueError('invalid file metadata')
        except (KeyError,ValueError,TypeError) as exc:raise DataError('recovery_manifest','Complete valid recovery manifest required') from exc
        destination.mkdir(parents=True,exist_ok=False)
        try:
            for entry in entries:
                target=(destination/entry.filename).resolve()
                if not target.is_relative_to(destination):raise DataError('recovery_path','Destination escape')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(entry) as source,target.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
                if entry.filename!='recovery_manifest.json' and sha256_file(target)!=manifest['files'][entry.filename]['sha256']:raise DataError('recovery_file','Restored file changed')
        except BaseException:
            # Only this call's new, resolved unique destination is removable.
            if destination.exists():shutil.rmtree(destination)
            raise
    return manifest


def restore_check(root,archive_path,expected_sha256):
    root,archive_path=Path(root).resolve(),Path(archive_path).resolve()
    if not archive_path.is_relative_to(root/'.local-backups'):raise DataError('recovery_archive','Only trusted local backup-directory archives accepted')
    record=RunRecord(load_config(root,root/'config/runtime.json'),scope='official_recovery_check',metadata={'module':'E12_recovery','archive':str(archive_path),'sha256':expected_sha256})
    destination=root/'.local-tmp/e12-restore'/record.run_id
    try:
        record.data['stage']='extract_verify_all_files';record.save()
        manifest=extract_verified(archive_path,expected_sha256,destination)
        record.data['stage']='recovered_source_and_models_in_new_process';record.save()
        output=record.directory/'recovered_probe.json'
        command=[sys.executable,'-B','-X','utf8',str(destination/'project/scripts/check_recovery.py'),'--manifest',str(destination/'recovery_manifest.json'),
            '--manifest-sha256',sha256_file(destination/'recovery_manifest.json'),'--output',str(output)]
        process=subprocess.run(command,cwd=destination/'project',capture_output=True,text=True,encoding='utf-8',timeout=300)
        (record.directory/'probe_stdout.txt').write_text(process.stdout,encoding='utf-8');(record.directory/'probe_stderr.txt').write_text(process.stderr,encoding='utf-8')
        if process.returncode!=0:raise DataError('recovery_probe','Recovered-source/model probe failed; see run logs')
        probe=json.loads(output.read_text(encoding='utf-8'))
        if probe['passed'] is not True:raise DataError('recovery_probe','Recovery probe not passed')
        result={'archive':str(archive_path),'archive_sha256':expected_sha256,'restored_directory':str(destination),'files_verified':len(manifest['files']),
            'recovered_probe':str(output),'recovered_probe_sha256':sha256_file(output),'same_machine_only':True,
            'official_bundle_root_rebinding':False,'original_files_overwritten':False,'original_test_reevaluated':False}
        record.finish(True,recovery=result)
        return {'report':str(record.path),'run_id':record.run_id,**result}
    except BaseException as exc:
        error=exc.as_dict() if isinstance(exc,DataError) else {'code':type(exc).__name__,'message':str(exc)}
        record.finish(False,error=error);raise
