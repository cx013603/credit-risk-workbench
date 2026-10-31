"""E06：不可变家族配方、训练内五折、混合特征LR及可信本机包；无最终测试入口。"""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import warnings

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import ranking_metrics, validation_report, prediction_agreement
from .experiments import digest_json, load_training_partition
from .normalize import write_json
from .raw import DataError, sha256_file
from .runtime import now

VERSION = 'logistic_v1'
CODE = ['src/credit_risk/logistic.py','src/credit_risk/evaluation.py','scripts/run_logistic_baseline.py',
        'scripts/check_logistic_model.py','src/credit_risk/experiments.py']


def environment():
    return {'python':sys.version.split()[0], **{p:importlib.metadata.version(p)
        for p in ('numpy','scipy','scikit-learn','joblib','pyarrow','threadpoolctl')}}


def load_recipe(path):
    try:
        v = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (v['schema_version'] != 1 or v['recipe_version'] != VERSION or v['family'] != 'logistic_regression'
                or v['variants'] != ['full40','main16'] or v['C_candidates'] != [.01,.1,1.]
                or v['solver'] != 'lbfgs' or v['l1_ratio'] != 0 or v['max_iter'] != 1000 or v['tol'] != 1e-5
                or v['class_weight'] is not None or v['cv_folds'] != 5 or v['native_threads'] != 1
                or v['numeric'] != 'median_keep_empty_add_indicator_then_standard_scale'
                or v['categorical'] != 'prefix_literal_values_explicit_missing_then_onehot_ignore_unknown'
                or v['selection'] != 'mean_training_cv_auc_then_lower_mean_brier_then_smaller_C'
                or v['calibration'] != 'none' or v['fit_initialization'] != 'fresh_pipeline_each_fit_no_smoke_reuse'
                or not isinstance(v['protocol_fingerprint'],str) or len(v['protocol_fingerprint']) != 64
                or len(v) != 18):
            raise ValueError('unsupported recipe')
        return v
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('logistic_recipe_invalid','Invalid fixed LR recipe') from exc


def register_recipe(root, recipe, protocol, *, fixture=False):
    """在任何fit前独占登记；同版本不同配方／代码／环境必须立新版本，不能静默重试调参。"""
    root = Path(root).resolve()
    if recipe['protocol_fingerprint'] != protocol:
        raise DataError('logistic_protocol_mismatch','Recipe and active protocol differ')
    payload = {'purpose':'synthetic_recipe' if fixture else 'formal_family_recipe', 'recipe':recipe,
        'protocol_fingerprint':protocol,'code_sha256':{p:sha256_file(root/p) for p in CODE},'environment':environment()}
    path = root/'reports/recipes'/('synthetic' if fixture else 'official')/VERSION/'registration.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    record = {'registered_at':now(),'payload':payload,'payload_fingerprint':digest_json(payload)}
    try:
        with path.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    except FileExistsError:
        current = json.loads(path.read_text(encoding='utf-8'))
        if current.get('payload') != payload or current.get('payload_fingerprint') != digest_json(payload):
            raise DataError('logistic_recipe_conflict','Existing family registration differs; reviewed new version required')
    return path


def matrix(table, columns, contract):
    if table.column_names != contract['feature_columns']:
        raise DataError('logistic_input_columns','Exact frozen feature input required; no ID, label, fold or audit columns')
    kinds = {c['name']:c['type'] for c in contract['features']}
    out = np.empty((table.num_rows,len(columns)),dtype=object)
    for j,name in enumerate(columns):
        values = table[name].to_pylist()
        if kinds[name] == 'string':
            out[:,j] = ['m:' if value is None else 'v:'+value for value in values]
        else:
            out[:,j] = [np.nan if value is None else float(value) for value in values]
            if np.isinf(np.asarray(out[:,j],dtype=float)).any():
                raise DataError('logistic_nonfinite','Infinite numeric feature')
    return out


def fresh_pipeline(columns, contract, C, recipe):
    kinds = {c['name']:c['type'] for c in contract['features']}
    num = [i for i,n in enumerate(columns) if kinds[n] != 'string']
    cat = [i for i,n in enumerate(columns) if kinds[n] == 'string']
    preprocess = ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),
                             ('scaler',StandardScaler())]),num),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cat)], sparse_threshold=1.)
    return Pipeline([('preprocess',preprocess),('model',LogisticRegression(C=C,solver=recipe['solver'],
        l1_ratio=0.,max_iter=recipe['max_iter'],tol=recipe['tol'],class_weight=None,random_state=42))])


def fit_fresh(X,y,columns,contract,C,recipe):
    pipeline = fresh_pipeline(columns,contract,C,recipe)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipeline.fit(X,y)
    return pipeline


def predict(pipeline,X):
    with threadpool_limits(limits=1):
        p = pipeline.predict_proba(X)[:,1]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise DataError('logistic_probability_invalid','Invalid logistic probabilities')
    return p


def preprocessing_state(pipeline,columns,contract):
    pre = pipeline.named_steps['preprocess']
    numeric = pre.named_transformers_['numeric']
    imputer, scaler = numeric.named_steps['imputer'],numeric.named_steps['scaler']
    kinds = {c['name']:c['type'] for c in contract['features']}
    names = [n for n in columns if kinds[n] != 'string']
    # get_feature_names_out保留数值补缺指示名；类别编码名称另由categories显式映射。
    numeric_names = imputer.get_feature_names_out(names).tolist()
    encoded_names = ['numeric__'+n for n in numeric_names]
    categories = {}
    for name,values in zip([n for n in columns if kinds[n]=='string'],pre.named_transformers_['category'].categories_):
        categories[name] = values.tolist()
        encoded_names.extend('category__'+name+'='+v for v in values.tolist())
    coefs = pipeline.named_steps['model'].coef_[0]
    if len(encoded_names) != len(coefs):
        raise DataError('logistic_coefficient_shape','Encoded names and coefficients differ')
    constant = [numeric_names[i] for i,v in enumerate(scaler.var_) if v==0]
    for i,var in enumerate(scaler.var_):
        if var==0 and abs(coefs[i])>1e-12:
            raise DataError('logistic_constant_contribution','Scaled constant numeric feature has nonzero contribution')
    return {'numeric_columns':names,'median':imputer.statistics_.tolist(),'scaled_columns':numeric_names,
        'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'variance':scaler.var_.tolist(),
        'constant_scaled_columns':constant,'categories':categories,'encoded_names':encoded_names,
        'coefficients':[{'name':n,'coefficient':float(c)} for n,c in zip(encoded_names,coefs)],
        'intercept':float(pipeline.named_steps['model'].intercept_[0]),
        'iterations':pipeline.named_steps['model'].n_iter_.tolist(),
        'interpretation':'conditional log-odds; numeric per training standard deviation after imputation; not causal or scorecard points'}


def cross_validate(X,y,ids,folds,columns,contract,recipe,checkpoint,progress=None):
    if (len(np.unique(ids))!=len(ids) or len(ids)!=len(y) or set(np.unique(folds))!={1,2,3,4,5}
            or X.shape!=(len(y),len(columns))):
        raise DataError('logistic_cv_invalid','Unique training IDs and persisted five-fold membership required')
    candidates, predictions = [], {}
    baseline_oof = np.zeros(len(y))
    for C in recipe['C_candidates']:
        oof, details = np.full(len(y),np.nan),[]
        for fold in range(1,6):
            checkpoint()
            fit, held = folds!=fold,folds==fold
            if set(np.unique(y[fit]))!={0,1} or set(np.unique(y[held]))!={0,1}:
                raise DataError('logistic_cv_classes','Both classes required in fit and heldout fold')
            started = time.monotonic()
            model = fit_fresh(X[fit],y[fit],columns,contract,C,recipe)
            p = predict(model,X[held]);oof[held] = p
            baseline_oof[held] = y[fit].mean()
            state = preprocessing_state(model,columns,contract)
            details.append({'fold':fold,'fit_rows':int(fit.sum()),'heldout_rows':int(held.sum()),
                'fit_id_sha256':digest_json(ids[fit].tolist()),'heldout_id_sha256':digest_json(ids[held].tolist()),
                'fit_heldout_intersection':int(len(set(ids[fit])&set(ids[held]))),
                'preprocessing':state,'metrics':ranking_metrics(y[held],p),
                'elapsed_seconds':round(time.monotonic()-started,3)})
            if progress: progress({'C':C,'fold':fold,'roc_auc':details[-1]['metrics']['roc_auc']})
        summary = {'C':C,'mean_cv_auc':float(np.mean([d['metrics']['roc_auc'] for d in details])),
            'mean_cv_brier':float(np.mean([d['metrics']['brier_score'] for d in details])),
            'oof_metrics':ranking_metrics(y,oof),'folds':details}
        candidates.append(summary);predictions[C]=oof
    chosen = sorted(candidates,key=lambda c:(-c['mean_cv_auc'],c['mean_cv_brier'],c['C']))[0]
    return {'selected_C':chosen['C'],'candidates':candidates,'selection_scope':'training_cv_only',
        'oof':predictions[chosen['C']],'baseline_oof':baseline_oof}


def load_local_package(root,directory,expected_sha256):
    """仅加载本项目刚生成的可信包；校验用途／范围在反序列化之前，拒绝smoke目录。"""
    root,directory = Path(root).resolve(),Path(directory).resolve()
    try:
        if directory == root/'models/logistic' or not directory.is_relative_to(root/'models/logistic'):
            raise ValueError('formal LR directory required')
        if sha256_file(directory/'bundle.json')!=expected_sha256:
            raise ValueError('bundle changed')
        b=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
        if (b['purpose']!='formal_logistic_baseline' or b['version']!=VERSION or b['status']!='complete'
                or b['fit_origin']!='fresh_training_partition_and_each_fold' or b['environment']!=environment()):
            raise ValueError('purpose, fit lineage or environment differs')
        expected = {'model_full40.joblib','model_main16.joblib','probe.parquet','results.json',
            'validation_full40.parquet','validation_main16.parquet','oof_full40.parquet','oof_main16.parquet','recipe.json','registration.json'}
        if set(b['files'])!=expected:
            raise ValueError('incomplete formal package')
        for name,info in b['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory) or sha256_file(path)!=info['sha256'] or path.stat().st_size!=info['bytes']:
                raise ValueError('file changed')
        return {v:joblib.load(directory/('model_'+v+'.joblib')) for v in ('full40','main16')},b
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataError('logistic_bundle_invalid','Reject incomplete, changed, smoke, incompatible or unsafe formal LR package') from exc


def check_saved_package(root,directory,expected_sha256):
    models,b=load_local_package(root,directory,expected_sha256)
    table=pq.read_table(Path(directory)/'probe.parquet')
    features=table.select(b['contract']['feature_columns'])
    checks={}
    for variant in ('full40','main16'):
        X=matrix(features,b['variant_columns'][variant],b['contract'])
        checks[variant]=prediction_agreement(table['expected_'+variant].to_numpy(),predict(models[variant],X))
    return checks


def load_logistic_bundle(root,report_path,*,require_official=True):
    """E07～E09只消费完整passed E06；没有正式测试释放或通用评分服务。"""
    root=Path(root).resolve();report_path=Path(report_path).resolve()
    try:
        if not report_path.is_relative_to(root/'reports'):
            raise ValueError('project report required')
        r=json.loads(report_path.read_text(encoding='utf-8'))
        if r['status']!='passed' or r['metadata']['module']!='E06' or r['scope']!='official_full_logistic_baseline':
            raise ValueError('official full passed E06 required')
        info=r['logistic']
        models,b=load_local_package(root,info['artifact_dir'],info['bundle_sha256'])
        plan=json.loads((root/'config/experiment_plan.json').read_text(encoding='utf-8'))
        if (plan['plan_payload_sha256']!=digest_json({k:v for k,v in plan.items() if k!='plan_payload_sha256'})
                or plan['status']!='frozen' or plan['protocol_fingerprint']!=b['protocol_fingerprint']):
            raise ValueError('active frozen protocol differs')
        if sha256_file(b['registration_path'])!=b['registration_sha256'] or sha256_file(b['input_report'])!=b['input_report_sha256']:
            raise ValueError('training lineage changed')
        return {'models':models,'bundle':b,'directory':Path(info['artifact_dir']),'report':r}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataError('logistic_report_invalid','Complete matching formal E06 report required') from exc


def run_baseline(root,input_report,recipe_path,record,progress=None):
    root=Path(root).resolve();recipe_path=Path(recipe_path).resolve();input_report=Path(input_report).resolve()
    recipe=load_recipe(recipe_path)
    train=load_training_partition(root,input_report,'train')
    protocol=train['protocol'];contract=train['contract']
    if recipe['protocol_fingerprint']!=digest_json(protocol):
        raise DataError('logistic_protocol_mismatch','Registered LR recipe does not match loaded protocol')
    registration=register_recipe(root,recipe,record.data['metadata']['protocol_fingerprint'])
    plan_hash=sha256_file(root/'config/experiment_plan.json')
    recipe_hash=sha256_file(recipe_path);registration_hash=sha256_file(registration)
    code_hashes={p:sha256_file(root/p) for p in CODE}
    input_hash=sha256_file(input_report)
    def checkpoint():
        if (sha256_file(root/'config/experiment_plan.json')!=plan_hash or sha256_file(recipe_path)!=recipe_hash
                or sha256_file(registration)!=registration_hash or sha256_file(input_report)!=input_hash
                or any(sha256_file(root/p)!=d for p,d in code_hashes.items())):
            raise DataError('logistic_inputs_changed','Frozen inputs, registered recipe or training code changed')
    record.data['metadata'].update(recipe_sha256=recipe_hash,registration_path=str(registration),
        registration_sha256=registration_hash,code_sha256=code_hashes,environment=environment(),
        input_report_sha256=input_hash,plan_sha256=plan_hash,first_fit_after_registration=True)
    record.save()
    ids=train['ids'].to_numpy();y=train['target'].to_numpy();folds=train['cv_fold'].to_numpy()
    variant_columns={'full40':contract['feature_columns'],'main16':contract['main_features']}
    cv={}
    for variant,columns in variant_columns.items():
        X=matrix(train['features'],columns,contract)
        cv[variant]=cross_validate(X,y,ids,folds,columns,contract,recipe,checkpoint,
            (lambda value,v=variant:progress({'variant':v,**value})) if progress else None)
    # 验证集在全部训练内选参结束后才读取；最终测试没有任何读取调用。
    validation=load_training_partition(root,input_report,'validation')
    vy=validation['target'].to_numpy();vids=validation['ids'].to_numpy()
    if set(ids)&set(vids):
        raise DataError('logistic_split_overlap','Training and validation IDs overlap')
    base_probability=float(y.mean())
    results={'purpose':'validation_not_final_test','family':'logistic_regression','recipe_version':VERSION,
        'protocol_fingerprint':recipe['protocol_fingerprint'],'baseline':validation_report(vy,np.full(len(vy),base_probability)),
        'train_rows':len(y),'validation_rows':len(vy),'test_evaluated':False,'variants':{},
        'limitations':protocol['limitations'],'fit_origin':'fresh_training_partition_and_each_fold'}
    destination=root/'models/logistic'/record.run_id
    if destination.exists():raise DataError('logistic_exists','Immutable model output already exists')
    staging=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    staging.mkdir(parents=True,exist_ok=False)
    try:
        probe=validation['features'].slice(0,min(64,len(vy)))
        for variant,columns in variant_columns.items():
            checkpoint();X=matrix(train['features'],columns,contract);VX=matrix(validation['features'],columns,contract)
            model=fit_fresh(X,y,columns,contract,cv[variant]['selected_C'],recipe)
            p=predict(model,VX)
            single=np.array([predict(model,VX[i:i+1])[0] for i in range(min(16,len(vy)))])
            single_check=prediction_agreement(p[:len(single)],single)
            joblib.dump(model,staging/('model_'+variant+'.joblib'),compress=3)
            restored=joblib.load(staging/('model_'+variant+'.joblib'))
            restore_check=prediction_agreement(p,predict(restored,VX))
            state=preprocessing_state(model,columns,contract)
            selected_oof=cv[variant].pop('oof');base_oof=cv[variant].pop('baseline_oof')
            results['variants'][variant]={**cv[variant],'validation':validation_report(vy,p),
                'preprocessing':state,'single_vs_batch':single_check,'same_process_reload':restore_check,
                'baseline_oof_metrics':ranking_metrics(y,base_oof)}
            pq.write_table(pa.table({'SK_ID_CURR':vids,'TARGET':vy,'probability':p,
                'baseline_probability':np.full(len(vy),base_probability)}),staging/('validation_'+variant+'.parquet'))
            pq.write_table(pa.table({'SK_ID_CURR':ids,'TARGET':y,'CV_FOLD':folds,'probability':selected_oof,
                'baseline_probability':base_oof}),staging/('oof_'+variant+'.parquet'))
            probe=probe.append_column('expected_'+variant,pa.array(p[:probe.num_rows]))
        results['history_gain_validation_auc']=(results['variants']['full40']['validation']['roc_auc']-
            results['variants']['main16']['validation']['roc_auc'])
        write_json(staging/'results.json',results);write_json(staging/'recipe.json',recipe)
        (staging/'registration.json').write_bytes(registration.read_bytes())
        pq.write_table(probe,staging/'probe.parquet')
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in staging.iterdir() if p.is_file()}
        bundle={'status':'complete','version':VERSION,'purpose':'formal_logistic_baseline',
            'fit_origin':'fresh_training_partition_and_each_fold','run_id':record.run_id,'environment':environment(),
            'protocol_fingerprint':recipe['protocol_fingerprint'],'contract':contract,'variant_columns':variant_columns,
            'registration_path':str(registration),'registration_sha256':registration_hash,
            'input_report':str(input_report),'input_report_sha256':input_hash,'files':files}
        write_json(staging/'bundle.json',bundle);digest=sha256_file(staging/'bundle.json')
        check_saved_package(root,staging,digest)
        cmd=[sys.executable,'-B','-X','utf8',str(root/'scripts/check_logistic_model.py'),
            '--bundle',str(staging),'--sha256',digest]
        child=subprocess.run(cmd,cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=60)
        fresh=json.loads(child.stdout)
        if child.returncode or fresh.get('passed') is not True:
            raise DataError('logistic_fresh_process_failed','Fresh process reload comparison failed')
        checkpoint()
        # 再校验正式输入包；最终测试仅以字节指纹确认未变，不读表或预测。
        load_training_partition(root,input_report,'train')
        staging.replace(destination)
        return {'artifact_dir':str(destination),'bundle_sha256':digest,'results':results,
                'fresh_process':{'command':cmd,'exit_code':child.returncode,**fresh}}
    finally:
        if staging.exists():
            if not staging.resolve().is_relative_to((root/'models/logistic').resolve()) or not staging.name.endswith('.partial'):
                raise DataError('unsafe_cleanup','LR staging cleanup escaped output root')
            shutil.rmtree(staging)
