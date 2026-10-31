"""F06: reviewable source versions and exact, movable local backup sets.

No invented Git identity, global configuration, network upload, model loading,
or primary workspace index mutation. A candidate tree is not a signed commit.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

from .raw import sha256_file, DataError
from .recovery import safe_member
from .recovery_v11 import no_links, metadata, encoded, save_new, require, unique_object
from .runtime import now

VERSION='release_management_v1'
STATIC=('src','scripts','config','sql','app','tests','docs','deliverables','data/sample')
SUFFIXES={'.py','.md','.json','.sql','.css','.js','.txt','.csv','.ps1','.toml','.yml','.yaml'}
DENY={'.git','.venv','.local-tmp','.local-backups','__pycache__','credentials','.kaggle'}
BACKUP_NAMES={'frozen_runtime.zip','frozen_runtime.receipt.json','delivery_sidecar.zip',
              'delivery_sidecar.receipt.json','original_dataset.zip','dataset_manifest.json','code_candidate.zip',
              'source_history.bundle','source_history.receipt.json'}
LIMIT=4*1024**3
SECRET=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique_object)


def source_allowed(relative):
    p=safe_member(relative)
    require(not any(x.casefold() in DENY or x.casefold().startswith('.env') or x.casefold()=='kaggle.json' for x in p.parts),
            'version_excluded','Sensitive/generated path excluded')
    allowed=(p.parts[0] in STATIC[:8] or p.parts[:2]==('data','sample') or
             relative=='reports/.gitkeep' or len(p.parts)==1 and (p.suffix=='.md' or p.name.startswith('requirements') and p.suffix=='.txt'
                                 or p.name in {'.gitignore','.gitattributes'}))
    require(allowed and (p.suffix in SUFFIXES or p.name in {'.gitignore','.gitattributes','.gitkeep'}),
            'version_excluded','Only explicit static source/document files may enter Git')
    if p.suffix=='.csv':
        require(p.parts[:2] in {('data','sample'),('tests','fixtures')},'version_excluded','Only artificial/sample CSV in code version')
    return p


def source_plan(root):
    root=Path(root).resolve(); names=set()
    for folder in STATIC:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: names.add(p.relative_to(root).as_posix())
    for p in [*root.glob('*.md'),*root.glob('requirements*.txt'),root/'.gitignore',root/'.gitattributes']:
        names.add(p.relative_to(root).as_posix())
    if (root/'reports/.gitkeep').is_file(): names.add('reports/.gitkeep')
    files={}
    for n in sorted(names):
        source_allowed(n); p=no_links(root/n)
        require(p.is_file() and p.stat().st_size<=8*1024**2,'version_source','Bounded regular source file required')
        data=p.read_bytes()
        require(not SECRET.search(data),'version_secret','Private key or access token detected; inspect file privately: '+n)
        files[root.name+'/'+n]={**metadata(p),'source':str(p),
            'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
    for name in ['AGENTS.md','.gitattributes','.gitignore']:
        if name=='.gitignore' and not (root.parent/name).exists(): continue
        guidance=no_links(root.parent/name); data=guidance.read_bytes()
        files[name]={**metadata(guidance),'source':str(guidance),
            'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
    require(sum(v['bytes'] for v in files.values())<=128*1024**2,'version_quota','Source snapshot exceeds code-only budget')
    return {'version':VERSION,'purpose':'code_only_reviewable_git_candidate','created_at':now(),
            'original_project':str(root),'files':files,'identity_selected':False,'commit_created':False}


def git(directory,*arguments):
    env={**os.environ,'GIT_CONFIG_GLOBAL':os.devnull,'GIT_CONFIG_NOSYSTEM':'1','LC_ALL':'C'}
    child=subprocess.run(['git','-c','core.autocrlf=true',*arguments],cwd=directory,env=env,
                         capture_output=True,timeout=120)
    require(child.returncode==0,'version_git',child.stderr.decode('utf-8',errors='replace')[-1000:])
    return child.stdout


def copy_exact(source,destination,meta):
    count=0; digest=hashlib.sha256()
    with no_links(source).open('rb') as src,Path(destination).open('xb') as out:
        while block:=src.read(1024**2):
            count+=len(block); require(count<=meta['bytes'] and count<=LIMIT,'backup_size','Artifact exceeds declared size')
            digest.update(block); out.write(block)
    require(count==meta['bytes'] and digest.hexdigest()==meta['sha256'],'backup_source_changed','Source changed while copying')


def prepare_code(root,destination,archive):
    root=Path(root).resolve(); destination=no_links(destination); archive=no_links(archive)
    require(not destination.exists() and destination.parent.is_dir() and not archive.exists() and archive.parent.is_dir(),
            'version_exists','New code candidate and archive destinations required')
    plan=source_plan(root); destination.mkdir()
    repository=destination/'repository'; repository.mkdir()
    try:
        for n,v in plan['files'].items():
            target=repository/n; target.parent.mkdir(parents=True,exist_ok=True); copy_exact(v['source'],target,v)
        git(repository,'init','--quiet','--template=','--initial-branch=codex/credit-risk-platform')
        git(repository,'add','--all')
        staged=git(repository,'ls-files','--stage','-z').decode('utf-8').split('\0')
        actual={}
        for line in staged:
            if not line: continue
            mode,blob,rest=line.split(' ',2); stage,name=rest.split('\t',1)
            require(mode=='100644' and stage=='0','version_index','Only regular unstaged-conflict files allowed')
            actual[name]=blob
        require(actual=={n:v['git_blob_sha1'] for n,v in plan['files'].items()},'version_bytes','Git conversion changed a frozen source byte')
        tree=git(repository,'write-tree').decode().strip()
        checkout=destination/'checkout'; checkout.mkdir()
        git(repository,'checkout-index','--all','--prefix='+str(checkout).replace('\\','/')+'/')
        require(all(metadata(checkout/n)=={k:v[k] for k in ['bytes','sha256']} for n,v in plan['files'].items()),
                'version_checkout','Fresh checkout differs from original source bytes')
        stat_text=git(repository,'diff','--cached','--stat').decode('utf-8',errors='replace')
        (destination/'review.txt').write_text(stat_text,encoding='utf-8')
        plan.update(git_tree=tree,git_version=git(repository,'--version').decode().strip(),
                    primary_workspace_index_modified=False,core_autocrlf_true_checkout_exact=True)
        save_new(destination/'source_manifest.json',plan)
        with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
            for n in sorted(plan['files']): z.write(repository/n,n)
            z.writestr('code_snapshot_manifest.json',encoded(plan))
        verify_code_archive(archive,sha256_file(archive))
        receipt={'passed':True,'archive':str(archive),**metadata(archive),'git_tree':tree,'files':len(actual),
                 'candidate_repository':str(repository),'review':str(destination/'review.txt'),
                 'commit_created':False,'identity_selected':False,'byte_exact_checkout':True}
        save_new(destination/'receipt.json',receipt)
        return receipt
    except BaseException as exc:
        save_new(destination/'failure.json',{'status':'failed','error':str(exc),'artifacts_retained_for_diagnosis':True})
        raise


def verify_code_archive(path,digest):
    require(sha256_file(no_links(path))==digest,'backup_hash','External trusted code archive SHA256 required')
    with ZipFile(path) as z:
        entries=z.infolist(); names=[x.filename for x in entries]
        require(len(names)<=3000 and len({n.casefold() for n in names})==len(names)
                and sum(x.file_size for x in entries)<=128*1024**2,'version_archive','Code archive quota/alias failure')
        for entry in entries:
            safe_member(entry.filename)
            require(not entry.is_dir() and stat.S_IFMT(entry.external_attr>>16) in {0,stat.S_IFREG}
                    and not entry.flag_bits&1 and entry.file_size<=8*1024**2,'version_archive','Regular bounded files required')
        m=json.loads(z.read('code_snapshot_manifest.json'),object_pairs_hook=unique_object)
        require(m['version']==VERSION and m['purpose']=='code_only_reviewable_git_candidate'
                and set(names)==set(m['files'])|{'code_snapshot_manifest.json'},'version_archive','Complete matching code manifest required')
        project=Path(m['original_project']).name
        for n,v in m['files'].items():
            if n not in {'AGENTS.md','.gitattributes','.gitignore'}:
                require(n.startswith(project+'/'),'version_archive','Code project namespace differs')
                source_allowed(n[len(project)+1:])
            data=z.read(n)
            require(len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
                    and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==v['git_blob_sha1'],
                    'version_archive','Code snapshot bytes differ')
        return m


def backup_sources(root,code_archive,code_digest,history_bundle,history_digest,history_receipt):
    root=Path(root).resolve(); code=verify_code_archive(code_archive,code_digest)
    runtime_receipt=root/'.local-backups/credit-risk-portable-v1-20261027-03.receipt.json'
    sidecar_receipt=root/'reports/portable_validation/closure_receipt_v2.json'
    data_manifest=root/'docs/evidence/data_manifest.json'
    runtime,sidecar,data=read(runtime_receipt),read(sidecar_receipt),read(data_manifest)
    require(sha256_file(no_links(history_bundle))==history_digest,'backup_history','Trusted Git history bundle required')
    history=read(history_receipt)
    require(history['passed'] is True and history['bundle_sha256']==history_digest
            and history['fresh_clone_and_checkout_exact'] is True,'backup_history','Verified Git bundle receipt required')
    return {
      'frozen_runtime.zip':{'source':runtime['archive'],'bytes':runtime['bytes'],'sha256':runtime['sha256']},
      'frozen_runtime.receipt.json':{'source':str(runtime_receipt),**metadata(runtime_receipt)},
      'delivery_sidecar.zip':{'source':sidecar['archive'],'bytes':sidecar['bytes'],'sha256':sidecar['sha256']},
      'delivery_sidecar.receipt.json':{'source':str(sidecar_receipt),**metadata(sidecar_receipt)},
      'original_dataset.zip':{'source':data['archive']['project_path'],'bytes':data['archive']['bytes'],'sha256':data['archive']['sha256']},
      'dataset_manifest.json':{'source':str(data_manifest),**metadata(data_manifest)},
      'code_candidate.zip':{'source':str(Path(code_archive).resolve()),**metadata(code_archive)},
      'source_history.bundle':{'source':str(Path(history_bundle).resolve()),**metadata(history_bundle)},
      'source_history.receipt.json':{'source':str(Path(history_receipt).resolve()),**metadata(history_receipt)}}


def validate_backup_manifest(m):
    require(m.get('version')==VERSION and m.get('purpose')=='private_movable_backup_set'
            and set(m['files'])==BACKUP_NAMES and sum(x['bytes'] for x in m['files'].values())<=LIMIT,
            'backup_manifest','Explicit bounded backup set required')
    for n,v in m['files'].items():
        safe_member(n)
        require(type(v['bytes']) is int and 0<=v['bytes']<=LIMIT and re.fullmatch('[0-9a-f]{64}',v['sha256']),
                'backup_manifest','Invalid artifact size or fingerprint')


def prepare_backup(root,code_archive,code_digest,destination,history_bundle,history_digest,history_receipt):
    destination=no_links(destination)
    require(not destination.exists() and destination.parent.is_dir(),'backup_exists','Entirely new backup destination required')
    files=backup_sources(root,code_archive,code_digest,history_bundle,history_digest,history_receipt)
    m={'version':VERSION,'purpose':'private_movable_backup_set','created_at':now(),'files':files,
       'license_verified':False,'use':'personal local skills learning; no external publication',
       'Git_commit_included':True,'offsite_disaster_recovery_verified':False,
       'runtime_use':'Restore frozen_runtime.zip using the F05 formal entry; sidecar is separate reading material',
       'raw_use':'Original unchanged dataset ZIP; no training or raw extraction performed here'}
    validate_backup_manifest(m); destination.mkdir()
    try:
        for n,v in files.items(): copy_exact(v['source'],destination/n,v)
        save_new(destination/'backup_manifest.json',m)
        result=verify_backup(destination,sha256_file(destination/'backup_manifest.json'))
        return {'passed':True,'directory':str(destination),'manifest_sha256':sha256_file(destination/'backup_manifest.json'),**result}
    except BaseException:
        require(no_links(destination).resolve()==destination,'backup_cleanup','New destination identity changed')
        shutil.rmtree(destination)
        raise


def verify_backup(directory,digest):
    directory=no_links(directory)
    require(sha256_file(directory/'backup_manifest.json')==digest,'backup_hash','External trusted backup manifest SHA256 required')
    m=read(directory/'backup_manifest.json'); validate_backup_manifest(m)
    require({p.name for p in directory.iterdir()}==BACKUP_NAMES|{'backup_manifest.json'},'backup_manifest','Missing/extra backup artifact')
    for n,v in m['files'].items():
        p=no_links(directory/n)
        require(p.is_file() and metadata(p)=={k:v[k] for k in ['bytes','sha256']},'backup_artifact','Backup artifact changed: '+n)
    runtime,sidecar=read(directory/'frozen_runtime.receipt.json'),read(directory/'delivery_sidecar.receipt.json')
    require(runtime['sha256']==m['files']['frozen_runtime.zip']['sha256']
            and sidecar['sha256']==m['files']['delivery_sidecar.zip']['sha256'],
            'backup_receipt','Original artifact receipts differ')
    for name in ['frozen_runtime.zip','delivery_sidecar.zip']:
        with ZipFile(directory/name) as z: require(z.testzip() is None,'backup_crc','Artifact CRC failure')
    code=verify_code_archive(directory/'code_candidate.zip',m['files']['code_candidate.zip']['sha256'])
    history=read(directory/'source_history.receipt.json')
    require(history['passed'] is True and history['fresh_clone_and_checkout_exact'] is True
            and history['bundle_sha256']==m['files']['source_history.bundle']['sha256'],
            'backup_history','Verified Git history receipt differs')
    heads=subprocess.run(['git','bundle','list-heads',str((directory/'source_history.bundle').resolve())],
                         capture_output=True,timeout=60)
    require(heads.returncode==0 and history['commit_sha256_or_sha1'].encode() in heads.stdout,
            'backup_history','Git bundle heads differ')
    data=read(directory/'dataset_manifest.json')
    require(data['archive']['sha256']==m['files']['original_dataset.zip']['sha256'],'backup_dataset','Original data archive differs')
    with ZipFile(directory/'original_dataset.zip') as z:
        expected={v['name']:v for v in data['files']}
        require(set(z.namelist())==set(expected),'backup_dataset','All ten original CSV members required')
        for n,v in expected.items():
            require(z.getinfo(n).file_size==v['bytes'],'backup_dataset','Original CSV size differs')
            with z.open(n) as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest()==v['sha256'],'backup_dataset','Original CSV content/CRC differs')
    return {'files':len(m['files']),'source_code_files':len(code['files']),'source_git_tree':code['git_tree'],
            'original_CSV_members_checked':len(data['files']),'all_artifact_SHA256_and_ZIP_CRC_verified':True,
            'offsite_disaster_recovery_verified':False}


def replicate_backup(source,digest,destination):
    source=no_links(source); verify_backup(source,digest); m=read(source/'backup_manifest.json')
    destination=no_links(destination)
    require(not destination.resolve().is_relative_to(source.resolve()),'backup_overlap','Replica must not be nested within its immutable source set')
    require(not destination.exists() and destination.parent.is_dir(),'backup_exists','New replica destination required')
    destination.mkdir()
    try:
        for n,v in m['files'].items(): copy_exact(source/n,destination/n,v)
        copy_exact(source/'backup_manifest.json',destination/'backup_manifest.json',metadata(source/'backup_manifest.json'))
        result=verify_backup(destination,digest)
        return {'passed':True,'destination':str(destination),'manifest_sha256':digest,**result,
                'storage_independence':'not_inferred_from_drive_letter_or_directory_name'}
    except BaseException:
        require(no_links(destination).resolve()==destination,'backup_cleanup','New replica identity changed')
        shutil.rmtree(destination)
        raise
