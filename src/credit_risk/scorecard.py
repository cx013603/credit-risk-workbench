"""E07传统评分卡：训练内分箱／平滑WOE／LR，精确可加总的分值。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import warnings
import time
import numpy as np
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .raw import DataError,sha256_file
from .runtime import now
from .normalize import write_json
from .experiments import digest_json,load_training_partition
from .evaluation import ranking_metrics,validation_report,prediction_agreement
from .logistic import matrix,predict,environment

VERSION='scorecard_v2'
CODE=['src/credit_risk/scorecard.py','src/credit_risk/evaluation.py','src/credit_risk/logistic.py',
      'src/credit_risk/experiments.py','scripts/run_scorecard.py','scripts/check_scorecard_model.py']


class WOETransformer(TransformerMixin,BaseEstimator):
    def __init__(self,columns,categorical,numeric_bins=5,rare_fraction=.01,smoothing=.5):
        self.columns=columns;self.categorical=categorical;self.numeric_bins=numeric_bins
        self.rare_fraction=rare_fraction;self.smoothing=smoothing

    def _groups(self,X,j,rule):
        if rule['kind']=='numeric':
            a=np.asarray(X[:,j],dtype=float)
            if np.isinf(a).any():raise DataError('scorecard_infinite','Infinite numeric input')
            return np.where(np.isnan(a),'m:',np.char.add('n:',np.searchsorted(rule['cuts'],a,side='right').astype(str)))
        return np.array([rule['category_groups'].get(x,'u:') for x in X[:,j]],dtype=object)

    def fit(self,X,y):
        X=np.asarray(X,dtype=object);y=np.asarray(y)
        if X.ndim!=2 or X.shape!=(len(y),len(self.columns)) or set(np.unique(y))!={0,1}:
            raise DataError('scorecard_fit_invalid','Fixed feature dimensions and both labels required')
        self.n_features_in_=X.shape[1];self.rules_=[]
        for j,name in enumerate(self.columns):
            if name in self.categorical:
                values,counts=np.unique(X[:,j].astype(str),return_counts=True)
                mapping={str(v):str(v) if c>=self.rare_fraction*len(y) or v=='m:' else 'r:' for v,c in zip(values,counts)}
                rule={'name':name,'kind':'category','category_groups':mapping}
            else:
                a=np.asarray(X[:,j],dtype=float)
                if np.isinf(a).any():raise DataError('scorecard_infinite','Infinite numeric input')
                observed=a[~np.isnan(a)]
                levels=np.unique(observed)
                if len(levels)<=self.numeric_bins:
                    cuts=levels[1:].tolist()
                else:
                    cuts=np.unique(np.quantile(observed,np.arange(1,self.numeric_bins)/self.numeric_bins)).tolist()
                    cuts=[v for v in cuts if observed.min()<v<=observed.max()]
                rule={'name':name,'kind':'numeric','cuts':cuts,'interval':'searchsorted right: [lower,upper); outside training range uses edge bin'}
            groups=self._groups(X,j,rule);keys=np.unique(groups);K=len(keys)
            good=int((y==0).sum());bad=int((y==1).sum());bins={}
            for key in keys:
                mask=groups==key;g=int(((y==0)&mask).sum());b=int(((y==1)&mask).sum())
                w=0. if K==1 else float(np.log(((g+self.smoothing)/(good+self.smoothing*K))/((b+self.smoothing)/(bad+self.smoothing*K))))
                bins[str(key)]={'rows':g+b,'nonevents':g,'events':b,'woe':w}
            # 未观察到的缺失、未见类别、中间空箱均为中性WOE，不虚构标签。
            rule.update(bins=bins,unknown_woe=0.,observed_group_count=K,fit_rows=len(y),fit_events=bad)
            self.rules_.append(rule)
        return self

    def transform(self,X):
        X=np.asarray(X,dtype=object)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_:
            raise DataError('scorecard_shape','Input feature count differs')
        out=np.zeros(X.shape,dtype=float)
        for j,rule in enumerate(self.rules_):
            out[:,j]=[rule['bins'].get(str(k),{'woe':0.})['woe'] for k in self._groups(X,j,rule)]
        return out


def fit_card(X,y,contract,C,recipe):
    columns=contract['feature_columns'];cats=[f['name'] for f in contract['features'] if f['type']=='string']
    p=Pipeline([('woe',WOETransformer(columns,cats,recipe['numeric_bins'],recipe['rare_fraction'],recipe['smoothing'])),
        ('model',LogisticRegression(C=C,l1_ratio=0.,solver='lbfgs',max_iter=recipe['max_iter'],tol=recipe['tol'],random_state=42))])
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);p.fit(X,y)
    return p


def points(model,X,recipe):
    factor=recipe['pdo']/np.log(2);offset=recipe['base_score']-factor*np.log(recipe['base_nonevent_event_odds'])
    woe=model.named_steps['woe'].transform(X)
    components=-factor*woe*model.named_steps['model'].coef_[0]
    base=offset-factor*model.named_steps['model'].intercept_[0]
    score=base+components.sum(axis=1)
    predicted=expit((offset-score)/factor)
    prediction_agreement(predict(model,X),predicted)
    return float(base),components,score


def point_table(model,recipe):
    factor=recipe['pdo']/np.log(2);rows=[]
    for j,rule in enumerate(model.named_steps['woe'].rules_):
        for group,b in rule['bins'].items():
            rows.append({'feature':rule['name'],'group':group,**b,
                'coefficient':float(model.named_steps['model'].coef_[0,j]),
                'points':float(-factor*model.named_steps['model'].coef_[0,j]*b['woe'])})
        rows.append({'feature':rule['name'],'group':'unobserved_or_unknown','woe':0.,'points':0.,'rows':0})
    return rows


def load_recipe(path):
    try:
        r=json.loads(Path(path).read_text(encoding='utf-8'))
        expected={'version':VERSION,'family':'traditional_scorecard','numeric_bins':5,'rare_fraction':.01,'smoothing':.5,
            'C_candidates':[.01,.1,1.],'max_iter':1000,'tol':1e-5,'base_score':600,'base_nonevent_event_odds':50,'pdo':20,
            'features':'all_frozen_40','selection':'mean_cv_auc_then_brier_then_smaller_C','unknown_woe':0,'calibration':'none','cv_folds':5}
        if {k:v for k,v in r.items() if k!='protocol_fingerprint'}!=expected or len(r['protocol_fingerprint'])!=64:raise ValueError('recipe')
        return r
    except (OSError,KeyError,TypeError,ValueError) as exc:raise DataError('scorecard_recipe_invalid','Unsupported scorecard recipe') from exc


def register(root,recipe,protocol):
    if recipe['protocol_fingerprint']!=digest_json(protocol):raise DataError('scorecard_protocol','Active protocol differs')
    payload={'recipe':recipe,'purpose':'formal_family_recipe','environment':environment(),
        'code_sha256':{p:sha256_file(root/p) for p in CODE}}
    path=root/'reports/recipes/official'/VERSION/'registration.json';path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x',encoding='utf-8') as f:f.write(json.dumps({'registered_at':now(),'payload':payload,'fingerprint':digest_json(payload)},ensure_ascii=False,indent=2)+'\n')
    except FileExistsError:
        current=json.loads(path.read_text(encoding='utf-8'))
        if current.get('payload')!=payload or current.get('fingerprint')!=digest_json(payload):raise DataError('scorecard_recipe_conflict','Immutable family version differs')
    return path


FILES={'model.joblib','results.json','bins_points.json','case.json','recipe.json','registration.json','validation.parquet','oof.parquet','probe.parquet'}
def load_package(root,directory,digest):
    root=Path(root).resolve();directory=Path(directory).resolve()
    try:
        if directory==root/'models/scorecard' or not directory.is_relative_to(root/'models/scorecard'):raise ValueError('path')
        if sha256_file(directory/'bundle.json')!=digest:raise ValueError('bundle')
        b=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
        if b['purpose']!='formal_traditional_scorecard' or b['fit_origin']!='fresh_training_partition_and_each_fold' or b['version']!=VERSION or b['status']!='complete' or b['environment']!=environment() or set(b['files'])!=FILES:raise ValueError('scope')
        for name,info in b['files'].items():
            p=(directory/name).resolve()
            if not p.is_relative_to(directory) or p.stat().st_size!=info['bytes'] or sha256_file(p)!=info['sha256']:raise ValueError('changed')
        return joblib.load(directory/'model.joblib'),b
    except (OSError,KeyError,ValueError,TypeError) as exc:raise DataError('scorecard_bundle_invalid','Reject unsafe, changed, smoke or incomplete scorecard package') from exc


def check_package(root,directory,digest):
    model,b=load_package(root,directory,digest);table=pq.read_table(Path(directory)/'probe.parquet')
    X=matrix(table.select(b['contract']['feature_columns']),b['contract']['feature_columns'],b['contract'])
    agreement=prediction_agreement(table['expected_probability'].to_numpy(),predict(model,X))
    _,_,scores=points(model,X,b['recipe'])
    delta=float(np.max(np.abs(scores-table['expected_score'].to_numpy())))
    if delta>1e-10:raise DataError('scorecard_score_mismatch','Score readback differs')
    return {'probability':agreement,'score_max_absolute_difference':delta}


def load_scorecard_bundle(root,report_path):
    root=Path(root).resolve();p=Path(report_path).resolve()
    try:
        if not p.is_relative_to(root/'reports'):raise ValueError('path')
        r=json.loads(p.read_text(encoding='utf-8'))
        if r['status']!='passed' or r['scope']!='official_full_scorecard' or r['metadata']['module']!='E07':raise ValueError('report')
        result=r['scorecard'];model,b=load_package(root,result['artifact_dir'],result['bundle_sha256'])
        plan=json.loads((root/'config/experiment_plan.json').read_text(encoding='utf-8'))
        if plan['status']!='frozen' or plan['plan_payload_sha256']!=digest_json({k:v for k,v in plan.items() if k!='plan_payload_sha256'}) or plan['protocol_fingerprint']!=b['recipe']['protocol_fingerprint']:raise ValueError('protocol')
        if sha256_file(b['registration_path'])!=b['registration_sha256'] or sha256_file(b['input_report'])!=b['input_report_sha256']:raise ValueError('lineage')
        return {'model':model,'bundle':b,'report':r,'directory':Path(result['artifact_dir'])}
    except (OSError,KeyError,ValueError,TypeError) as exc:raise DataError('scorecard_report_invalid','Matching passed scorecard report required') from exc


def run_scorecard(root,input_report,recipe_path,record,progress=None):
    root=Path(root).resolve();input_report=Path(input_report).resolve();recipe_path=Path(recipe_path).resolve()
    recipe=load_recipe(recipe_path);train=load_training_partition(root,input_report,'train');contract=train['contract']
    reg=register(root,recipe,train['protocol'])
    protected={str(root/p):sha256_file(root/p) for p in CODE+['config/experiment_plan.json']}
    protected.update({str(p):sha256_file(p) for p in [input_report,recipe_path,reg]})
    def checkpoint():
        if any(sha256_file(p)!=d for p,d in protected.items()):raise DataError('scorecard_changed','Code, recipe or frozen input changed')
    record.data['metadata'].update(registration_path=str(reg),registration_sha256=sha256_file(reg),protected_sha256=protected,environment=environment());record.save()
    columns=contract['feature_columns'];X=matrix(train['features'],columns,contract);y=train['target'].to_numpy();ids=train['ids'].to_numpy();folds=train['cv_fold'].to_numpy()
    if len(np.unique(ids))!=len(ids) or set(np.unique(folds))!={1,2,3,4,5}:raise DataError('scorecard_cv','Invalid persisted training folds')
    candidates=[];predictions={};baseline=np.zeros(len(y))
    for C in recipe['C_candidates']:
        oof=np.zeros(len(y));details=[]
        for fold in range(1,6):
            checkpoint();fit=folds!=fold;held=folds==fold;started=time.monotonic()
            model=fit_card(X[fit],y[fit],contract,C,recipe);p=predict(model,X[held]);oof[held]=p;baseline[held]=y[fit].mean()
            details.append({'fold':fold,'fit_id_sha256':digest_json(ids[fit].tolist()),'heldout_id_sha256':digest_json(ids[held].tolist()),
                'fit_rows':int(fit.sum()),'heldout_rows':int(held.sum()),'rules':model.named_steps['woe'].rules_,
                'metrics':ranking_metrics(y[held],p),'elapsed_seconds':round(time.monotonic()-started,3)})
            if progress:progress({'C':C,'fold':fold,'roc_auc':details[-1]['metrics']['roc_auc']})
        candidates.append({'C':C,'mean_cv_auc':float(np.mean([d['metrics']['roc_auc'] for d in details])),
            'mean_cv_brier':float(np.mean([d['metrics']['brier_score'] for d in details])),'folds':details,'oof_metrics':ranking_metrics(y,oof)})
        predictions[C]=oof
    chosen=sorted(candidates,key=lambda c:(-c['mean_cv_auc'],c['mean_cv_brier'],c['C']))[0]['C']
    val=load_training_partition(root,input_report,'validation');vy=val['target'].to_numpy();vids=val['ids'].to_numpy()
    if set(ids)&set(vids):raise DataError('scorecard_overlap','Train/validation IDs overlap')
    checkpoint();model=fit_card(X,y,contract,chosen,recipe);VX=matrix(val['features'],columns,contract);p=predict(model,VX)
    base,components,scores=points(model,VX,recipe)
    single=prediction_agreement(p[:16],np.array([predict(model,VX[i:i+1])[0] for i in range(min(16,len(vy)))]))
    factor=recipe['pdo']/np.log(2);offset=recipe['base_score']-factor*np.log(recipe['base_nonevent_event_odds'])
    score_error=float(np.max(np.abs(scores-(offset-factor*model.decision_function(VX)))))
    if score_error>1e-10:raise DataError('scorecard_addition','Points and logit score disagree')
    where=np.flatnonzero(ids==100003);case_index=int(where[0]) if len(where) else 0
    caseX=X[case_index:case_index+1];cb,cp,cs=points(model,caseX,recipe)
    case={'SK_ID_CURR':int(ids[case_index]),'partition':'train','base_points':cb,'feature_points':dict(zip(columns,cp[0].tolist())),
        'total_score':float(cs[0]),'probability':float(predict(model,caseX)[0]),'meaning':'training demonstration, not independent evaluation'}
    results={'family':'traditional_scorecard','purpose':'validation_not_final_test','selected_C':chosen,'candidates':candidates,
        'validation':validation_report(vy,p),'baseline_validation':validation_report(vy,np.full(len(vy),y.mean())),
        'baseline_oof':ranking_metrics(y,baseline),'single_batch':single,'score_addition_max_error':score_error,
        'limitations':train['protocol']['limitations'],'calibration':'none','test_evaluated':False}
    destination=root/'models/scorecard'/record.run_id
    if destination.exists():raise DataError('scorecard_exists','Immutable output exists')
    stage=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial');stage.mkdir(parents=True)
    try:
        joblib.dump(model,stage/'model.joblib',compress=3)
        readback=prediction_agreement(p,predict(joblib.load(stage/'model.joblib'),VX));results['same_process_reload']=readback
        for name,value in [('results.json',results),('bins_points.json',{'rules':model.named_steps['woe'].rules_,'points':point_table(model,recipe),'base_points':base}),('case.json',case),('recipe.json',recipe)]:write_json(stage/name,value)
        (stage/'registration.json').write_bytes(reg.read_bytes())
        pq.write_table(pa.table({'SK_ID_CURR':vids,'TARGET':vy,'probability':p,'score':scores}),stage/'validation.parquet')
        pq.write_table(pa.table({'SK_ID_CURR':ids,'TARGET':y,'CV_FOLD':folds,'probability':predictions[chosen],'baseline_probability':baseline}),stage/'oof.parquet')
        probe=val['features'].slice(0,min(64,len(vy))).append_column('expected_probability',pa.array(p[:64])).append_column('expected_score',pa.array(scores[:64]));pq.write_table(probe,stage/'probe.parquet')
        bundle={'status':'complete','purpose':'formal_traditional_scorecard','fit_origin':'fresh_training_partition_and_each_fold','version':VERSION,'environment':environment(),'recipe':recipe,'contract':contract,
            'registration_path':str(reg),'registration_sha256':sha256_file(reg),'input_report':str(input_report),'input_report_sha256':sha256_file(input_report),
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in stage.iterdir()}}
        write_json(stage/'bundle.json',bundle);digest=sha256_file(stage/'bundle.json');check_package(root,stage,digest)
        cmd=[sys.executable,'-B','-X','utf8',str(root/'scripts/check_scorecard_model.py'),'--bundle',str(stage),'--sha256',digest]
        child=subprocess.run(cmd,cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=60);fresh=json.loads(child.stdout)
        if child.returncode or not fresh['passed']:raise DataError('scorecard_fresh_process','Fresh process failed')
        checkpoint();load_training_partition(root,input_report,'train');stage.replace(destination)
        return {'artifact_dir':str(destination),'bundle_sha256':digest,'results':results,'case':case,'fresh_process':fresh}
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to((root/'models/scorecard').resolve()) or not stage.name.endswith('.partial'):raise DataError('unsafe_cleanup','Scorecard cleanup escaped root')
            shutil.rmtree(stage)
