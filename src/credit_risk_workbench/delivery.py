"""Versioned local V2 media, offline environments and separate application snapshots."""
from contextlib import closing
import hashlib,json,os,re,shutil,sqlite3,stat,subprocess,sys,time,uuid,zipfile
from pathlib import Path,PurePosixPath

VERSION='workbench_delivery_v2.0'
MANIFEST='workbench_manifest.json'
BASE_SEAL='6e8100f48b575404d49d789a326fae2280f4e8bc73da888c908350d54e3864f0'
MAX_FILES=30000;MAX_TOTAL=12*1024**3;MAX_FILE=2*1024**3
class DeliveryError(ValueError):pass
def need(value,message):
    if not value:raise DeliveryError(message)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def encoded(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
def read(path):
    def unique(items):
        v={}
        for k,x in items:need(k not in v,'重复JSON键');v[k]=x
        return v
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)
def save(path,value,new=False):
    p=Path(path)
    if new:
        with p.open('xb') as f:f.write(encoded(value))
    else:
        tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
        with tmp.open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
def safe(value):
    need(isinstance(value,str) and value and '\\' not in value and ':' not in value and '\0' not in value,'非法介质路径')
    p=PurePosixPath(value);need(bool(p.parts) and not p.is_absolute() and p.as_posix()==value,'路径须为规范相对路径')
    reserved={'con','prn','aux','nul',*[f'com{i}' for i in range(1,10)],*[f'lpt{i}' for i in range(1,10)]}
    need(all(x not in ['.','..',''] and not x.endswith((' ','.')) and not any(ord(c)<32 or c in '<>"|?*' for c in x) and x.split('.')[0].casefold() not in reserved for x in p.parts),'路径含保留或危险名称')
    return p
def no_links(path):
    p=Path(path).absolute()
    for q in [p,*p.parents]:
        if q.exists() or q.is_symlink():
            info=q.lstat();need(not q.is_symlink() and not getattr(info,'st_file_attributes',0)&0x400,'不接收符号链接或重解析点')
    return p.resolve()
def meta(path):return {'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def copy_checked(source,target,info):
    h=hashlib.sha256();size=0
    while block:=source.read(1024*1024):
        size+=len(block);need(size<=info['bytes'],'文件超过登记大小');h.update(block);target.write(block)
    need(size==info['bytes'] and h.hexdigest()==info['sha256'],'文件大小或指纹不符')
def validate_manifest(m):
    need(m.get('version')==VERSION and m.get('base_manifest_sha256')==BASE_SEAL and m.get('active_model') in ['original40','original40_with_candidate71'],'不兼容介质版本／模型')
    if m.get('active_model')=='original40_with_candidate71':need(m.get('new71_activated') is True and 'workbench/config/workbench/candidate71.json' in m.get('files',{}),'缺少71项部署绑定')
    files=m.get('files');need(isinstance(files,dict) and 0<len(files)<=MAX_FILES,'无效文件清单')
    need(not {MANIFEST,'release-trust.json'}&set(files),'保留元数据不能登记为普通文件')
    names=set();total=0
    for name,info in files.items():
        safe(name);need(name.casefold() not in names,'大小写路径冲突');names.add(name.casefold())
        need(type(info.get('bytes')) is int and 0<=info['bytes']<=MAX_FILE and isinstance(info.get('sha256'),str) and re.fullmatch('[a-f0-9]{64}',info['sha256']),'无效文件登记')
        total+=info['bytes']
    need(total<=MAX_TOTAL and not any(str(parent).casefold() in names for name in files for parent in PurePosixPath(name).parents if str(parent)!='.'),'总配额或文件目录冲突')
    need({'启动工作台.cmd','关闭工作台.cmd','查看状态.cmd','python/python.exe','workbench/scripts/workbench_control.py','workbench/config/workbench/api-portable.json','app-snapshot/manifest.json'}<=set(files),'缺少必要启动／恢复组件')
    return m
def build(plan,output):
    m=validate_manifest(plan);output=no_links(output);need(not output.exists() and output.parent.is_dir(),'输出介质必须是新文件')
    partial=output.with_suffix(output.suffix+'.partial');need(not partial.exists(),'残留partial请先核对')
    with partial.open('xb') as raw,zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for name,info in m['files'].items():
            p=no_links(info['source']);need(p.is_file(),'来源文件缺失')
            zi=zipfile.ZipInfo(name);zi.compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.zip','.whl','.gz'] else zipfile.ZIP_DEFLATED;zi.external_attr=(stat.S_IFREG|0o644)<<16
            with p.open('rb') as src,z.open(zi,'w',force_zip64=True) as dst:copy_checked(src,dst,info)
        # No producer absolute paths are needed by a consumer.
        portable={**m,'files':{p:{k:v for k,v in info.items() if k!='source'} for p,info in m['files'].items()}}
        manifest_bytes=encoded(portable);z.writestr(MANIFEST,manifest_bytes)
        z.writestr('release-trust.json',encoded({'version':VERSION,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'archive_sha256':None,'trust_basis':'publisher_supplied_manifest_for_trusted_manual_extract; external_archive_receipt_available_separately'}))
    os.rename(partial,output)
    return {'archive':str(output),**meta(output),'manifest_sha256':hashlib.sha256(encoded(portable)).hexdigest(),'files':len(portable['files'])}
def extract(archive,expected,destination):
    archive=no_links(archive);destination=no_links(destination)
    need(re.fullmatch('[a-f0-9]{64}',expected or '') and sha(archive)==expected,'介质外部SHA256不符')
    need(not destination.exists() and destination.parent.is_dir(),'恢复目标必须为全新目录')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();names=[e.filename for e in entries]
        need(len(entries)<=MAX_FILES+2 and len({n.casefold() for n in names})==len(names),'重复或过多ZIP成员')
        for e in entries:
            safe(e.filename);need(not e.is_dir() and not e.flag_bits&1 and stat.S_IFMT(e.external_attr>>16) in [0,stat.S_IFREG] and e.file_size<=MAX_FILE,'不支持的ZIP成员')
        need(MANIFEST in names and z.getinfo(MANIFEST).file_size<=12*1024**2,'清单缺失或过大')
        # Strict duplicate-key decoder via read() also applies to archive metadata.
        text=z.read(MANIFEST);temporary=destination.parent/(destination.name+'.manifest-'+uuid.uuid4().hex)
        temporary.write_bytes(text)
        try:m=validate_manifest(read(temporary))
        finally:temporary.unlink()
        need(set(names)==set(m['files'])|{MANIFEST,'release-trust.json'} and z.getinfo('release-trust.json').file_size<=4096 and all(z.getinfo(n).file_size==v['bytes'] for n,v in m['files'].items()),'清单与ZIP不一致')
        if os.name=='nt':need(max(len(str(destination/n)) for n in m['files'])<258,'Windows恢复路径过长；请选较短目录（例如 C:/CreditWorkbench），目标尚未创建')
        destination.mkdir()
        # Failure stays in this new directory for diagnosis; never deletes another checkout.
        try:
            for n,info in m['files'].items():
                p=destination/n;p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(n) as src,p.open('xb') as dst:copy_checked(src,dst,info)
            (destination/MANIFEST).write_bytes(text)
            save(destination/'release-trust.json',{'version':VERSION,'manifest_sha256':hashlib.sha256(text).hexdigest(),'archive_sha256':expected},True)
        except BaseException:
            save(destination/'RESTORE_FAILED.json',{'passed':False,'archive_sha256':expected},True);raise
    return {'passed':True,'directory':str(destination),'manifest_sha256':sha(destination/MANIFEST),'files':len(m['files'])}
def verify(directory):
    r=no_links(directory);trust=read(r/'release-trust.json');need(trust['version']==VERSION and sha(r/MANIFEST)==trust['manifest_sha256'],'本地信任记录／清单不符')
    m=validate_manifest(read(r/MANIFEST))
    for n,info in m['files'].items():
        p=no_links(r/n);need(p.is_relative_to(r) and p.is_file() and meta(p)==info,'受保护文件改变：'+n)
    return m
def env():
    v={k:x for k,x in os.environ.items() if k.upper() not in ['PYTHONHOME','PYTHONPATH','VIRTUAL_ENV']}
    v.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_INDEX='1',PIP_CONFIG_FILE=os.devnull,NPM_CONFIG_UPDATE_NOTIFIER='false',NPM_CONFIG_AUDIT='false',NPM_CONFIG_FUND='false')
    return v
def run(command,cwd,timeout=600):
    command=list(command)
    if Path(command[0]).name.lower() in ['python.exe','pythonw.exe'] and 'utf8' not in command:command[1:1]=['-X','utf8']
    p=subprocess.run([str(x) for x in command],cwd=cwd,env=env(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    need(p.returncode==0,'子命令失败：'+p.stderr[-1800:]+p.stdout[-600:]);return {'command':[str(x) for x in command],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def check_database(db,name):
    version=2 if name=='simulation.sqlite' else 1
    need(db.execute('PRAGMA user_version').fetchone()[0]==version and db.execute('PRAGMA quick_check').fetchone()[0]=='ok','应用库版本或完整性错误')
    tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if name=='simulation.sqlite':need({'state','marks','review_records','reports'}<=tables,'备注数据库结构不符')
    else:need('drafts' in tables and {x[1] for x in db.execute('PRAGMA table_info(drafts)')}=={'id','revision','title','payload','updated','result'},'草稿数据库结构不符')
def generated_environments(r):
    return {p.relative_to(r).as_posix():meta(p) for folder in [r/'api-env',r/'base-release/runtime'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo']}
def app_backup(source,destination):
    source=no_links(source);dest=no_links(destination);need(not dest.exists() and dest.parent.is_dir(),'备份目录必须为新目录');dest.mkdir();(dest/'reports').mkdir()
    records={}
    # The two independent databases have separate consistent snapshot points, no cross-DB business transaction.
    databases=[('simulation.sqlite',2),('new_inputs.sqlite',1)]+([('candidate71.sqlite',1)] if (source/'candidate71.sqlite').exists() else [])
    for name,version in databases:
        path=no_links(source/name);need(path.is_file(),'应用库缺失')
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db,closing(sqlite3.connect(dest/name)) as target:
            db.execute('BEGIN');check_database(db,name)
            dump='\n'.join(db.iterdump());rows=db.execute('SELECT report_id,json_sha256,html_sha256 FROM reports ORDER BY report_id').fetchall() if name=='simulation.sqlite' else []
            db.backup(target);need('\n'.join(target.iterdump())==dump,'SQLite备份逻辑不一致')
            records[name]={'schema_version':version,'logical_sha256':hashlib.sha256(dump.encode()).hexdigest(),'at':time.time()}
            for ident,j,h in rows:
                need(re.fullmatch('[a-f0-9]{32}',ident) is not None,'报告ID非法')
                for ext,expected in [('json',j),('html',h)]:
                    p=no_links(source/'reports'/(ident+'.'+ext));need(p.is_relative_to(source) and sha(p)==expected,'报告内容改变')
                    shutil.copyfile(p,dest/'reports'/p.name)
    files={p.relative_to(dest).as_posix():meta(p) for p in dest.rglob('*') if p.is_file()}
    save(dest/'manifest.json',{'version':'workbench_app_backup_v2','files':files,'databases':records,'cross_database_atomic_snapshot':False},True)
    return {'passed':True,'manifest_sha256':sha(dest/'manifest.json'),'files':len(files),'directory':str(dest)}
def app_restore(source,dest,expected):
    source=no_links(source);dest=no_links(dest);need(not dest.exists() and dest.parent.is_dir() and sha(source/'manifest.json')==expected,'应用备份指纹／目标不符')
    m=read(source/'manifest.json');need(m['version']=='workbench_app_backup_v2' and {'simulation.sqlite','new_inputs.sqlite'}<=set(m['files']),'应用备份版本错误')
    need(len(m['files'])<=10002,'应用备份文件超额')
    for n,v in m['files'].items():
        safe(n);need(n in ['simulation.sqlite','new_inputs.sqlite','candidate71.sqlite'] or re.fullmatch(r'reports/[a-f0-9]{32}\.(json|html)',n),'应用备份路径非法')
        need(meta(no_links(source/n))==v,'应用备份内容改变')
    for name in ['simulation.sqlite','new_inputs.sqlite']+(['candidate71.sqlite'] if 'candidate71.sqlite' in m['files'] else []):
        with closing(sqlite3.connect((source/name).as_uri()+'?mode=ro',uri=True)) as db:
            check_database(db,name);dump='\n'.join(db.iterdump());need(hashlib.sha256(dump.encode()).hexdigest()==m['databases'][name]['logical_sha256'],'数据库逻辑指纹不符')
    dest.mkdir();(dest/'reports').mkdir()
    for n,v in m['files'].items():shutil.copyfile(source/n,dest/n)
    return {'passed':True,'directory':str(dest),'files':len(m['files'])}
def setup(directory):
    r=no_links(directory);m=verify(r);logs=[];receipt=r/'setup.json'
    if receipt.exists():
        trust=read(r/'release-trust.json');need(sha(receipt)==trust.get('environment_receipt_sha256'),'环境安装回执改变');result=read(receipt)
        need(result['passed'] and generated_environments(r)==result['environment_files'],'运行环境文件改变，请从新目录重建');return result
    need(not (r/'api-env').exists() and not (r/'base-release').exists(),'存在未完成环境；保留诊断并用新恢复目录')
    py=r/'python/python.exe';logs.append(run([py,'-I','-B','-c','import sys;print(sys.version.split()[0]);print(sys.base_prefix)'],r))
    need(logs[-1]['stdout'].splitlines()[0]=='3.14.5' and Path(logs[-1]['stdout'].splitlines()[1]).resolve()==(r/'python').resolve(),'便携Python版本或基址不符')
    base=m['base_archive'];need(base['sha256']=='bdc60f86f15da5267b9b29436f2aca69d917f7ca9cce12ce7ad655dc496f3504' and sha(r/base['path'])==base['sha256'],'原03介质不符')
    # Bootstrap uses only stdlib; the original loader runs after its exact wheels are installed.
    with zipfile.ZipFile(r/base['path']) as z:
        raw=z.read('portable_manifest.json');need(hashlib.sha256(raw).hexdigest()==BASE_SEAL,'原03清单不符');bm=json.loads(raw)
        need(set(z.namelist())==set(bm['files'])|{'portable_manifest.json'} and len(z.namelist())==903,'原03成员不符')
        dest=r/'base-release';dest.mkdir()
        for n,info in bm['files'].items():
            safe(n);p=dest/n;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(n) as src,p.open('xb') as dst:copy_checked(src,dst,info)
        (dest/'portable_manifest.json').write_bytes(raw)
    for dest,wheels,lock in [(r/'api-env',r/'api-wheels',r/'workbench/requirements-workbench.lock'),(r/'base-release/runtime',r/'base-release/wheelhouse',r/'base-release/project/requirements-repro.txt')]:
        logs.append(run([py,'-I','-B','-m','venv','--copies',dest],r))
        exe=dest/'Scripts/python.exe';logs.append(run([exe,'-I','-B','-m','pip','install','--no-index','--no-compile','--require-hashes','--find-links',wheels,'-r',lock],r));logs.append(run([exe,'-I','-B','-m','pip','check'],r))
        logs.append(run([exe,'-I','-B','-c','import sys;print(sys.prefix);print(sys.base_prefix);print(sys.path)'],r))
        need(Path(logs[-1]['stdout'].splitlines()[1]).resolve()==(r/'python').resolve(),'环境依赖了原Python路径')
    app_restore(r/'app-snapshot',r/'app-data',m['app_snapshot_sha256'])
    (r/'var').mkdir();(r/'var/runs').mkdir()
    result={'passed':True,'version':VERSION,'runtime_environments':'offline_new_API_and_original_core','python_base':str(r/'python'),'logs':logs,'training_started':False,'original_directory_used':False,'environment_files':generated_environments(r)}
    save(receipt,result,True);trust=read(r/'release-trust.json');trust['environment_receipt_sha256']=sha(receipt);save(r/'release-trust.json',trust);return result
def frontend_build(directory):
    r=no_links(directory);m=verify(r);tag=uuid.uuid4().hex;out=r/'builds'/tag;out.mkdir(parents=True)
    archive=r/m['node_archive']['path'];node=r/'node'
    if not node.exists():
        with zipfile.ZipFile(archive) as z:
            names=z.namelist();prefix='node-v24.15.0-win-x64/'
            for n in names:
                need(n.startswith(prefix),'Node介质前缀错误')
                rel=n[len(prefix):]
                if not rel or n.endswith('/'):continue
                safe(rel);p=node/rel;p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(n) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    source=r/'workbench/frontend';project=out/'frontend';project.mkdir()
    for p in source.rglob('*'):
        if p.is_file() and not {'dist','node_modules'}&set(p.relative_to(source).parts):
            dest=project/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    npm=node/'node_modules/npm/bin/npm-cli.js';environment=env();environment['PATH']=str(node)+os.pathsep+environment.get('PATH','')
    logs=[]
    for args in [['ci','--offline','--ignore-scripts','--no-audit','--no-fund','--cache',str(r/'npm-cache')],['run','build']]:
        p=subprocess.run([str(node/'node.exe'),str(npm),*args],cwd=project,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));logs.append({'args':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        save(out/'progress.json',{'logs':logs})
        need(p.returncode==0,'离线前端构建失败：'+p.stderr[-1600:])
    built={p.relative_to(project/'dist').as_posix():meta(p) for p in (project/'dist').rglob('*') if p.is_file()};shipped={p.relative_to(source/'dist').as_posix():meta(p) for p in (source/'dist').rglob('*') if p.is_file()}
    result={'passed':True,'offline':True,'output':str(out),'logs':logs,'built_files':built,'shipped_dist_exact':built==shipped,'original_dist_unchanged':True};save(out/'validation.json',result,True);return result
