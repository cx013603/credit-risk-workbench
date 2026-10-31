"""Fixed-budget history exploration. New runs only; no reserved-test reader."""
import gc,json,os,sys,time
from pathlib import Path
import joblib,numpy as np,pyarrow as pa,pyarrow.compute as pc,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from credit_risk.raw import sha256_file
from credit_risk.logistic import matrix,environment
from credit_risk.tree import fit_tree,fit_state
from credit_risk.evaluation import ranking_metrics,validation_report
from credit_risk.runtime import now,process_peak_memory
from .features_v2 import require,read,write,digest,load_partition,verify_frozen

VERSION='history_model_exploration_v2.0'
FEATURE_SEAL='d648b8c5ef64a114317bde2c38f40006f4518b241096cde49480b3364d74f6f2'
PACKAGE='data/processed/workbench_v2/20261029-02'
LABEL_BASE='data/processed/experiments/full/20261025-152621-d3528f85'
BASELINE='models/tree/20261025-201440-458db9f1/validation.parquet'
CODE=['src/credit_risk_workbench/experiment_v2.py','scripts/run_workbench_experiment_v2.py','scripts/workbench_experiment_worker.py','src/credit_risk/tree.py','src/credit_risk/logistic.py','src/credit_risk/evaluation.py','src/credit_risk/owned_process.py','src/credit_risk/runtime.py']

def contract_for(contract,feature_set):
    require(feature_set in ['original40','extended71'],'Unknown feature set')
    count=40 if feature_set=='original40' else 71
    return {**contract,'feature_columns':contract['feature_columns'][:count],'features':contract['features'][:count]}

def labels(root,partition,ids,audit):
    require(partition in ['train','validation'],'Reserved/unlabeled labels denied before access')
    root=Path(root);api=read(root/'config/workbench/api.json');release=(root/api['release_directory']).resolve()
    require(sha256_file(release/'portable_manifest.json')==api['release_manifest_sha256'],'Original03 seal changed')
    relative=LABEL_BASE+'/'+partition+'.parquet';path=root/relative
    require(sha256_file(path)==read(release/'portable_manifest.json')['files']['project/'+relative]['sha256'],'Original label partition changed')
    table=pq.read_table(path,columns=['SK_ID_CURR','TARGET']);allids=table['SK_ID_CURR'].to_numpy();y=table['TARGET'].to_numpy()
    require(len(np.unique(allids))==len(allids) and set(np.unique(y))=={0,1},'Invalid source label keys/classes')
    order=np.argsort(allids);positions=np.searchsorted(allids[order],ids)
    require(np.all(positions<len(order)) and np.array_equal(allids[order][positions],ids),'Feature/label identity mismatch')
    audit.append({'partition':partition,'path':relative,'columns':['SK_ID_CURR','TARGET'],'source_rows':len(allids),'selected_rows':len(ids),'sha256':sha256_file(path)})
    return y[order][positions].astype(int)

def checked_predict(model,X):
    with threadpool_limits(limits=1):p=model.predict_proba(X)[:,1]
    require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid probability')
    return p

def projected(view,contract):return matrix(view['X'].select(contract['feature_columns']),contract['feature_columns'],contract)

def identity_digest(view):return digest(view['identity']['SK_ID_CURR'].to_pylist())

def model_checks(model,contract):
    tree=model.named_steps['tree'];require(tree.n_iter_==120 and not tree.do_early_stopping_,'Nonfixed iterations')
    # Public native fitted categories are inspected through the same pinned
    # helper as V1; preprocessing is fitted only by fit_tree on its input.
    cats=[f['name'] for f in contract['features'] if f['type']=='string']
    encoder=tree._preprocessor.named_transformers_['encoder']
    return dict(zip(cats,[v.tolist() for v in encoder.categories_]))

def coverage_report(table,y,p):
    result={}
    for source,column in [('CC','CC_MONTH_COUNT'),('POS','POS_MONTH_COUNT'),('BB','BB_MONTH_COUNT'),('INST','INST_PAYMENT_RECORD_COUNT')]:
        observed=table[column].to_numpy()>0;groups={}
        for name,mask in [('eligible_history',observed),('no_eligible_history',~observed)]:
            group={'rows':int(mask.sum()),'events':int(y[mask].sum()),'probabilities_finite':bool(np.isfinite(p[mask]).all())}
            if len(np.unique(y[mask]))==2:group['metrics']=ranking_metrics(y[mask],p[mask])
            else:group['metrics']=None
            groups[name]=group
        require(sum(v['rows'] for v in groups.values())==len(y),'Coverage lost rows');result[source]=groups
    return result

def run_task(root,task,out):
    root=Path(root);out=Path(out);started=time.monotonic();package=root/PACKAGE
    registration=read(Path(task['registration_path']));require(digest(registration)==task['registration_fingerprint'],'Registration changed')
    require(all(sha256_file(root/p)==h for p,h in registration['code_sha256'].items()),'Registered code changed')
    require(task['purpose'] in ['CV','final'],'Unknown fit purpose')
    protocol=read(package/'protocol.json');verify_frozen(root,package,FEATURE_SEAL)
    require(registration['protocol_fingerprint']==digest(protocol),'Frozen experiment changed')
    feature_set=task['feature_set'];contract=contract_for(read(package/'contract.json'),feature_set)
    candidates={v['id']:v for v in protocol['candidates']};require(task['candidate_id'] in candidates,'Unregistered candidate')
    audit=[];fold=task.get('fold');require(task['purpose']!='CV' or type(fold) is int and fold in range(1,6),'Invalid held fold')
    view=load_partition(root,package,FEATURE_SEAL,'train',purpose='fit',heldout_fold=fold)
    fit_ids=view['identity']['SK_ID_CURR'].to_numpy();fit_hash=identity_digest(view);fit_rows=len(fit_ids)
    y=labels(root,'train',fit_ids,audit);X=projected(view,contract)
    fit_metadata={'feature_count':len(contract['feature_columns']),'fit_rows':fit_rows,'fit_ID_sha256':fit_hash,'fit_fold_counts':{str(i):view['identity']['CV_FOLD'].to_pylist().count(i) for i in range(1,6)},'columns':contract['feature_columns'],'label_partition':'train','heldout_fold':fold,'fit_classes':np.bincount(y,minlength=2).tolist()}
    del view;gc.collect();peak_before=process_peak_memory();require(peak_before['peak_working_set_bytes'] is not None and peak_before['peak_working_set_bytes']<=protocol['stopping']['max_process_peak_bytes'],'Pre-fit memory limit')
    write(out/'fit-start.json',{'started_at':now(),'pid':os.getpid(),**fit_metadata})
    recipe={**protocol['fixed_parameters'],'native_threads':1,'stopping':protocol['stopping']}
    tick=time.monotonic();model=fit_tree(X,y,contract,candidates[task['candidate_id']],recipe);fit_seconds=time.monotonic()-tick
    state=fit_state(model,X,contract);require(state['fit_rows']==fit_rows,'Fit state row mismatch')
    require(all(set(values)==set(X[:,contract['feature_columns'].index(name)]) for name,values in state['categories'].items()),'Category vocabulary not fit-only')
    joblib.dump(model,out/'model.joblib');write(out/'fit-state.json',state)
    probe=pa.Table.from_pydict({})
    # Use an independently reloadable raw input probe from the fitted partition.
    source=load_partition(root,package,FEATURE_SEAL,'train',purpose='score');probe=source['X'].select(contract['feature_columns']).slice(0,12)
    expected=checked_predict(model,matrix(probe,contract['feature_columns'],contract));pq.write_table(probe.append_column('expected_probability',pa.array(expected)),out/'probe.parquet')
    del X,y,source;gc.collect();outputs={}
    if task['purpose']=='CV':
        hold=load_partition(root,package,FEATURE_SEAL,'train',purpose='score');mask=pc.equal(hold['identity']['CV_FOLD'],fold)
        hold={'X':hold['X'].filter(mask),'identity':hold['identity'].filter(mask)};ids=hold['identity']['SK_ID_CURR'].to_numpy()
        require(not np.intersect1d(fit_ids,ids).size and fit_rows+len(ids)==215257,'Training/CV leakage or lost rows')
        p=checked_predict(model,projected(hold,contract));yh=labels(root,'train',ids,audit)
        outputs['CV']=ranking_metrics(yh,p);outputs['heldout_ID_sha256']=identity_digest(hold)
        pq.write_table(pa.table({'SK_ID_CURR':ids,'CV_FOLD':[fold]*len(ids),'probability':p}),out/'heldout_predictions.parquet');del hold;gc.collect()
    else:
        for partition in ['validation','unlabeled_scoring']:
            view=load_partition(root,package,FEATURE_SEAL,partition,purpose='score');ids=view['identity']['SK_ID_CURR'].to_numpy()
            p=checked_predict(model,projected(view,contract));pq.write_table(pa.table({'SK_ID_CURR':ids,'sample_use':[partition]*len(ids),'probability':p}),out/(partition+'_predictions.parquet'))
            outputs[partition]={'rows':len(ids),'predictions_finite_0_to_1':True,'ID_sha256':identity_digest(view)}
            if partition=='validation':
                yv=labels(root,'validation',ids,audit);outputs[partition]['report']=validation_report(yv,p);outputs[partition]['source_coverage']=coverage_report(view['X'],yv,p)
            del view;gc.collect()
    peak=process_peak_memory();require(peak['peak_working_set_bytes'] is not None and peak['peak_working_set_bytes']<=protocol['stopping']['max_process_peak_bytes'],'Post-fit process peak limit')
    require(fit_seconds<=protocol['stopping']['max_fit_seconds'],'Fit time limit')
    files={p.name:sha256_file(p) for p in out.iterdir() if p.is_file() and p.suffix!='.log'}
    result={'passed':True,'task':task,'fit_metadata':fit_metadata,'outputs':outputs,'fit_seconds':round(fit_seconds,4),'job_seconds':round(time.monotonic()-started,4),'process_peak':peak,'label_access':audit,'original_test_access':False,'files':files,'fit_called':1}
    write(out/'result.json',result);return result

def aggregate_cv(results):
    require(len(results)==30,'All 30 CV fits required')
    aggregates=[]
    for feature_set in ['original40','extended71']:
        for identifier in [1,2,3]:
            group=[v for v in results if v['task']['feature_set']==feature_set and v['task']['candidate_id']==identifier]
            require(len(group)==5 and {v['task']['fold'] for v in group}==set(range(1,6)),'Incomplete CV candidate')
            auc=[v['outputs']['CV']['roc_auc'] for v in group];brier=[v['outputs']['CV']['brier_score'] for v in group]
            aggregates.append({'feature_set':feature_set,'candidate_id':identifier,'mean_auc':float(np.mean(auc)),'std_auc':float(np.std(auc,ddof=0)),'mean_brier':float(np.mean(brier)),'fold_auc':auc,'fold_brier':brier})
    selected={feature_set:min([v for v in aggregates if v['feature_set']==feature_set],key=lambda v:(-v['mean_auc'],v['mean_brier'],v['candidate_id'])) for feature_set in ['original40','extended71']}
    return aggregates,selected

def decide(protocol,baseline,original,extended,selected):
    spec=protocol['acceptance'];checks={}
    for name,reference in [('saved_tree',baseline),('fresh_original40',original)]:
        checks['auc_gain_'+name]=extended['roc_auc']-reference['roc_auc']>=spec['validation_auc_gain_over_both_baselines_at_least']
        checks['brier_'+name]=extended['brier_score']-reference['brier_score']<=spec['validation_brier_above_either_baseline_at_most']
    checks['CV_auc_std']=selected['extended71']['std_auc']<=selected['original40']['std_auc']+spec['cv_auc_std_above_original40_at_most']
    checks['protocol_and_coverage']=True # set only after successful complete task/gate validation in the caller
    return {'accept_extended71':all(checks.values()),'checks':checks,'scope':'previously_used_validation_exploration_only_no_independent_final_test','deployed':False}

def owned_task(command,cwd,out,timeout):
    """Bound a whole one-fit job (including preparation) more strictly than fit."""
    import subprocess
    from credit_risk.owned_process import OwnedJob
    out=Path(out);job=OwnedJob();process=None;started=time.monotonic();timed_out=False
    try:
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            try:job.assign(process)
            except BaseException:
                process.terminate();process.wait(timeout=20);raise
            write(out/'ownership-ticket.json',{'parent_pid':os.getpid(),'launcher_pid':process.pid,'assigned_before_start':True})
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;job.close();process.wait(timeout=20)
        return {'returncode':process.returncode,'timed_out':timed_out,'owned_process_tree':True,'wall_seconds':round(time.monotonic()-started,4),'timeout_seconds':timeout}
    finally:job.close()
