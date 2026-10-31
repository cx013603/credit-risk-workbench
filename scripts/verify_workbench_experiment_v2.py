"""Recompute every prediction metric, fold identity, fit vocabulary and reload."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import joblib,numpy as np,pyarrow.compute as pc,pyarrow.parquet as pq
from credit_risk.raw import sha256_file
from credit_risk.logistic import matrix,environment
from credit_risk_workbench.features_v2 import read,write,require,digest,verify_frozen
from credit_risk_workbench.experiment_v2 import PACKAGE,FEATURE_SEAL,labels,contract_for,checked_predict,BASELINE

def independent_metrics(y,p):
    """Mann-Whitney tied ranks, grouped empirical CDFs, mean squared error."""
    y=np.asarray(y);p=np.asarray(p);require(set(np.unique(y))=={0,1} and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid metric input')
    order=np.argsort(p,kind='stable');sortedp=p[order];sy=y[order]
    ends=np.r_[np.flatnonzero(sortedp[1:]!=sortedp[:-1])+1,len(y)];starts=np.r_[0,ends[:-1]]
    n1=int(y.sum());n0=len(y)-n1;event_sums=np.add.reduceat(sy,starts)
    auc=float((np.sum(event_sums*(starts+1+ends)/2)-n1*(n1+1)/2)/(n1*n0))
    counts=ends-starts;cs1=np.cumsum(event_sums[::-1]);cs0=np.cumsum((counts-event_sums)[::-1]);ks=cs1/n1-cs0/n0
    thresholds=sortedp[ends-1][::-1];best=float(np.max(thresholds[ks==ks.max()]))
    return {'roc_auc':auc,'ks':float(ks.max()),'brier_score':float(np.mean((y-p)**2)),'threshold':best}

def check_metrics(y,p,expected,with_threshold=False):
    actual=independent_metrics(y,p)
    for name in ['roc_auc','ks','brier_score']:require(abs(actual[name]-expected[name])<=1e-12,'Independent metric differs '+name)
    if with_threshold:
        require(abs(actual['threshold']-expected['threshold']['value'])<=1e-12,'Threshold differs')
        for i,bin_info in enumerate(expected['calibration_bins']):
            mask=(p<=((i+1)/10)) & ((p>=0) if i==0 else p>(i/10))
            # linspace boundaries are checked with their exact source float values.
            edges=np.linspace(0,1,11);mask=(p<=edges[i+1]) & ((p>=0) if i==0 else p>edges[i])
            require(int(mask.sum())==bin_info['rows'],'Calibration rows differ')
            if mask.any():
                require(abs(float(p[mask].mean())-bin_info['mean_prediction'])<=1e-12 and abs(float(y[mask].mean())-bin_info['event_fraction'])<=1e-12,'Calibration averages differ')
    return actual

def verify(directory,output):
    directory=Path(directory).resolve();require(directory.is_relative_to(ROOT/'reports/workbench'),'Unsafe run root')
    result=read(directory/'results.json');registration=read(directory/'registration.json');protocol=read(ROOT/PACKAGE/'protocol.json');contract=read(ROOT/PACKAGE/'contract.json')
    verify_frozen(ROOT,ROOT/PACKAGE,FEATURE_SEAL)
    require(result['passed'] and result['formal_fit_attempts']==32 and result['CV_fits']==30 and result['final_fits']==2 and not result['original_test_access'],'Incomplete experiment')
    require(result['registration_fingerprint']==digest(registration) and registration['protocol_fingerprint']==digest(protocol),'Registration differs')
    require(registration['environment']==environment(),'Registered environment differs')
    require(all(sha256_file(ROOT/p)==h for p,h in registration['code_sha256'].items()),'Registered source changed')
    baseline=read(directory.parent/'before.json');require(all(sha256_file(ROOT/p)==h for p,h in baseline['files'].items()),'Original protected bytes changed')
    train=pq.read_table(ROOT/PACKAGE/'train.parquet');identities=train.select(['SK_ID_CURR','CV_FOLD']);allids=identities['SK_ID_CURR'].to_numpy();folds=identities['CV_FOLD'].to_numpy();audit=[]
    yt=labels(ROOT,'train',allids,audit);val=pq.read_table(ROOT/PACKAGE/'validation.parquet',columns=['SK_ID_CURR']);valids=val['SK_ID_CURR'].to_numpy();yv=labels(ROOT,'validation',valids,audit)
    ylookup={'train':dict(zip(allids,yt)),'validation':dict(zip(valids,yv))};receipts=[];cv={};newbundles={};probe_rows=0
    dirs=sorted(p for p in directory.iterdir() if p.is_dir() and (p/'result.json').is_file());require(len(dirs)==32,'Missing fit directories')
    for folder in dirs:
        r=read(folder/'result.json');task=r['task'];meta=r['fit_metadata'];execution=read(folder/'execution.json')
        require(r['passed'] and not execution['timed_out'] and execution['returncode']==0 and execution['owned_process_tree'],'Failed owned job')
        require(all(sha256_file(folder/p)==h for p,h in r['files'].items()),'Changed fit artifacts')
        require(r['fit_called']==1 and r['fit_seconds']<=300 and r['process_peak']['peak_working_set_bytes']<=2*1024**3,'Fit resource budget')
        require(not r['original_test_access'] and all(v['partition'] in (['train'] if task['purpose']=='CV' else ['train','validation']) for v in r['label_access']),'Label scope violation')
        mask=np.ones(len(allids),dtype=bool) if task['purpose']=='final' else folds!=task['fold'];fit_ids=allids[mask]
        require(meta['fit_ID_sha256']==digest(fit_ids.tolist()) and meta['fit_rows']==len(fit_ids),'Fit complement differs')
        selected_contract=contract_for(contract,task['feature_set']);require(meta['columns']==selected_contract['feature_columns'],'Model feature allowlist differs')
        state=read(folder/'fit-state.json');require(state['iterations']==120 and not state['early_stopping'] and state['fit_rows']==len(fit_ids),'Fit state differs')
        for name,values in state['categories'].items():
            source=train[name].filter(mask)
            vocabulary={'m:' if v is None else 'v:'+v for v in source.to_pylist()};require(set(values)==vocabulary,'Fit-only vocabulary differs')
        expected_empty=[f['name'] for f in selected_contract['features'] if f['type']!='string' and train[f['name']].filter(mask).null_count==len(fit_ids)]
        require(set(state['fit_all_empty_forced_zero_columns'])==set(expected_empty),'All-empty handling used nonfit rows')
        model=joblib.load(folder/'model.joblib');probe=pq.read_table(folder/'probe.parquet');actual=checked_predict(model,matrix(probe.select(selected_contract['feature_columns']),selected_contract['feature_columns'],selected_contract))
        require(np.max(np.abs(actual-probe['expected_probability'].to_numpy()))<=1e-10,'Reloaded probe differs');probe_rows+=len(actual);del model,probe;gc.collect()
        if task['purpose']=='CV':
            p=pq.read_table(folder/'heldout_predictions.parquet');ids=p['SK_ID_CURR'].to_numpy();require(np.array_equal(ids,allids[~mask]) and len(np.unique(ids))==len(ids),'Heldout IDs changed')
            actual=check_metrics(np.array([ylookup['train'][i] for i in ids]),p['probability'].to_numpy(),r['outputs']['CV']);cv.setdefault((task['feature_set'],task['candidate_id']),[]).append((task['fold'],actual))
        else:
            for partition in ['validation','unlabeled_scoring']:
                p=pq.read_table(folder/(partition+'_predictions.parquet'));ids=p['SK_ID_CURR'].to_numpy();expected_ids=pq.read_table(ROOT/PACKAGE/(partition+'.parquet'),columns=['SK_ID_CURR'])['SK_ID_CURR'].to_numpy()
                require(np.array_equal(ids,expected_ids) and np.isfinite(p['probability'].to_numpy()).all() and np.all((p['probability'].to_numpy()>=0)&(p['probability'].to_numpy()<=1)),'Final scoring identity/probability differs')
                if partition=='validation':check_metrics(yv,p['probability'].to_numpy(),r['outputs'][partition]['report'],True)
            newbundles[task['feature_set']]={'version':'tree_history_v2.0' if task['feature_set']=='extended71' else 'tree_original40_comparator_v2.0','purpose':'verified_exploratory_candidate_not_activated','model_path':str((folder/'model.joblib').relative_to(ROOT)).replace('\\','/'),'model_sha256':sha256_file(folder/'model.joblib'),'contract':selected_contract,'feature_package':PACKAGE,'feature_freeze_sha256':FEATURE_SEAL,'threshold':r['outputs']['validation']['report']['threshold'],'candidate_id':task['candidate_id'],'registration_fingerprint':digest(registration),'environment':environment(),'selected_in_training_CV':True,'validation_used_for_exploration':True,'independent_test_available':False,'deployed':False}
        receipts.append({'slot':task['slot'],'directory':folder.name,'result_sha256':sha256_file(folder/'result.json'),'model_sha256':sha256_file(folder/'model.joblib'),'probe_rows':12,'independent_metrics_passed':True,'fit_IDs_and_vocabulary_training_only':True})
        print('verified '+folder.name,flush=True)
    for entry in result['CV_aggregates']:
        group=sorted(cv[(entry['feature_set'],entry['candidate_id'])]);values=[v['roc_auc'] for _,v in group];brier=[v['brier_score'] for _,v in group]
        require(abs(np.mean(values)-entry['mean_auc'])<=1e-12 and abs(np.std(values)-entry['std_auc'])<=1e-12 and abs(np.mean(brier)-entry['mean_brier'])<=1e-12,'CV aggregate differs')
    chosen={s:min([v for v in result['CV_aggregates'] if v['feature_set']==s],key=lambda v:(-v['mean_auc'],v['mean_brier'],v['candidate_id'])) for s in ['original40','extended71']};require(chosen==result['selected'],'CV selection differs')
    p=pq.read_table(ROOT/BASELINE,columns=['SK_ID_CURR','probability']);ids=p['SK_ID_CURR'].to_numpy();check_metrics(np.array([ylookup['validation'][i] for i in ids]),p['probability'].to_numpy(),result['validation']['original_saved_tree'],True)
    spec=protocol['acceptance'];extended=result['validation']['extended71'];checks=[]
    for name in ['original_saved_tree','fresh_original40']:
        base=result['validation'][name];checks.extend([extended['roc_auc']-base['roc_auc']>=spec['validation_auc_gain_over_both_baselines_at_least'],extended['brier_score']<=base['brier_score']+spec['validation_brier_above_either_baseline_at_most']])
    checks.append(chosen['extended71']['std_auc']<=chosen['original40']['std_auc']+spec['cv_auc_std_above_original40_at_most']);require(all(checks)==result['decision']['accept_extended71'],'Acceptance differs')
    bundle_seals={}
    for name,bundle in newbundles.items():
        path=directory/(name+'-candidate-bundle.json')
        if path.exists():require(read(path)==bundle,'Existing candidate descriptor differs; never overwrite')
        else:write(path,bundle)
        bundle_seals[name]=sha256_file(path)
    release=ROOT.parent/'credit-risk-portable-validation/release-03';ledger=next(v for v in read(release/'portable_manifest.json')['files'] if v.endswith('/test_access.jsonl'))
    for path in [ROOT/ledger.removeprefix('project/'),release/ledger]:require(sha256_file(path)=='565d05fb3342c1bd855ff0bc31031956b48ce3687c0a6c6fdd385f5988c5640f' and len(path.read_text(encoding='utf-8').splitlines())==1,'Test ledger changed')
    record={'passed':True,'scope':'docs51_step13_exploratory_model_comparison_only','run':str(directory.relative_to(ROOT)).replace('\\','/'),'formal_fits':32,'CV_fits':30,'final_fits':2,'verified_jobs':receipts,'reloaded_models':32,'reloaded_probe_rows':probe_rows,'independent_metric_method':'tied Mann-Whitney ranks; grouped empirical CDF KS; direct mean squared error; explicit calibration bins','training_ID_fold_vocabulary_empty_policy_verified':True,'independent_acceptance_check':True,'accept_extended71':result['decision']['accept_extended71'],'protected_entries_unchanged':len(baseline['files']),'old_and_release_test_ledger_rows':1,'original_test_table_or_predictions_read':False,'original_model_and_threshold_unchanged':True,'deployed':False,'independent_final_test':False,'results_sha256':sha256_file(directory/'results.json'),'candidate_bundle_sha256':bundle_seals,'label_access':audit}
    write(output,record);print(json.dumps({'passed':True,'models_reloaded':32,'probe_rows':probe_rows,'accept_extended71':record['accept_extended71']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.directory,a.output)
