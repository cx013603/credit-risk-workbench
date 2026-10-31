"""Pinned candidate inference, shared path arithmetic and raw-SQL71 parity."""
import json
import hashlib
import math
from pathlib import Path
import duckdb
import numpy as np
import pyarrow as pa
import joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
from credit_risk.logistic import matrix
from credit_risk.scoring import validate_features
from credit_risk.tree_explanation import node_references, walk_tree
from .contracts import require
from .features_v2 import run_family, join_new
from .reference import bucket, labels
from .candidate71_contract import complete, raw40, VERSION as INPUT_VERSION, digest

VERSION='candidate71_local_v1'
BUNDLE_SHA='5aa209abcf2dd7002f0581e2f4b9060345160acc967690c36093becb128fe314'
MODEL_SHA='448c91f5a12f4241ce5d9d1190d628073d05116c5cf039806ffc7e24e3084442'
FEATURE_SEAL='d648b8c5ef64a114317bde2c38f40006f4518b241096cde49480b3364d74f6f2'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(root, config):
    root=Path(root).resolve();cfg=json.loads(Path(config).read_text(encoding='utf-8'))
    require(set(cfg)=={'version','enabled','manifest','sha256'} and cfg['version']==VERSION and cfg['enabled'] is True,'VERSION_MISMATCH')
    path=(root/cfg['manifest']).resolve();require(path.is_relative_to(root) and sha(path)==cfg['sha256'],'VERSION_MISMATCH')
    d=json.loads(path.read_text(encoding='utf-8'));require(d['version']==VERSION and d['model_sha256']==MODEL_SHA and d['bundle_sha256']==BUNDLE_SHA and d['feature_freeze_sha256']==FEATURE_SEAL,'VERSION_MISMATCH')
    paths={}
    for n,h in d['files'].items():
        p=(root/n).resolve();require(p.is_relative_to(root) and p.is_file() and not p.is_symlink() and sha(p)==h,'VERSION_MISMATCH');paths[n]=p
    require(d['bundle'] in paths and d['model'] in paths and d['reference'] in paths and set(d['partitions'])=={'train','validation','unlabeled_scoring'},'VERSION_MISMATCH')
    bundle=json.loads(paths[d['bundle']].read_text(encoding='utf-8'))
    require(sha(paths[d['bundle']])==BUNDLE_SHA and sha(paths[d['model']])==MODEL_SHA and bundle['threshold']['value']==0.0764883811769358,'VERSION_MISMATCH')
    require(bundle['contract']['feature_columns']==d['feature_columns'] and len(d['feature_columns'])==71,'VERSION_MISMATCH')
    reference=json.loads(paths[d['reference']].read_text(encoding='utf-8'))
    require(reference['rows']==215257 and reference['feature_columns']==d['feature_columns'] and reference['train_only'] is True,'VERSION_MISMATCH')
    return {'root':root,'manifest':d,'manifest_sha256':cfg['sha256'],'paths':paths,'bundle':bundle,'reference':reference}


class Explainer71:
    def __init__(self, model, contract):
        self.model=model;self.contract=contract;self.names=contract['feature_columns'];self.tree=model.named_steps['tree'];t=self.tree
        require(set(model.named_steps)=={'empty','tree'} and t.n_iter_==120 and t.n_trees_per_iteration_==1 and list(t.classes_)==[0,1] and not t.do_early_stopping_,'VERSION_MISMATCH')
        mask=np.asarray(t.is_categorical_,dtype=bool)
        require(mask.shape==(71,) and mask.tolist()==[f['type']=='string' for f in contract['features']],'VERSION_MISMATCH')
        self.order=np.r_[np.flatnonzero(mask),np.flatnonzero(~mask)]
        self.predictors=[p[0] for p in t._predictors];self.refs=[node_references(p.nodes,71) for p in self.predictors]
        self.intercept=float(t._baseline_prediction[0,0]);self.base=self.intercept+sum(float(r[0]) for r in self.refs)
        require({int(p.nodes[0]['count']) for p in self.predictors}=={215257},'VERSION_MISMATCH')
    def explain(self, features):
        validate_features(features,self.contract);X=matrix(features,self.names,self.contract)
        with threadpool_limits(limits=1):
            t=self.tree;binned=t._bin_mapper.transform(t._preprocess_X(self.model.named_steps['empty'].transform(X),reset=False))
            require(binned.shape==(features.num_rows,71) and binned.dtype==np.uint8,'VERSION_MISMATCH')
            values=np.zeros((features.num_rows,71));raw_leaves=np.full(features.num_rows,self.intercept);delta=0.
            for p,refs in zip(self.predictors,self.refs):
                terms,leaves=walk_tree(p.nodes,p.binned_left_cat_bitsets,refs,binned,t._bin_mapper.missing_values_bin_idx_)
                oracle=p.predict_binned(binned,t._bin_mapper.missing_values_bin_idx_,1)
                delta=max(delta,float(np.max(np.abs(leaves-oracle))));values+=terms;raw_leaves+=leaves
            contributions=np.zeros_like(values);contributions[:,self.order]=values
            raw=self.model.decision_function(X);probability=self.model.predict_proba(X)[:,1]
        reconstructed=self.base+contributions.sum(axis=1)
        checks={'leaf_max_difference':delta,'leaf_sum_raw_max_difference':float(np.max(np.abs(raw_leaves-raw))),'additivity_max_difference':float(np.max(np.abs(reconstructed-raw))),'probability_max_difference':float(np.max(np.abs(expit(reconstructed)-probability)))}
        require(all(math.isfinite(x) and x<=1e-10 for x in checks.values()) and np.isfinite(contributions).all(),'VERSION_MISMATCH')
        return {'contributions':contributions,'probability':probability,'raw_score':raw,'checks':checks}


class Candidate71:
    def __init__(self,spec,raw_engine):
        self.spec=spec;self.root=spec['root'];self.d=spec['manifest'];self.bundle=spec['bundle'];self.reference=spec['reference'];self.contract=self.bundle['contract'];self.names=self.d['feature_columns'];self.raw_engine=raw_engine
        self.model=joblib.load(spec['paths'][self.d['model']]);self.model_loads=1;self.explainer=Explainer71(self.model,self.contract)
        self.fields=self.d['fields'];self.sql_root=Path(self.d['sql_root_absolute']) if 'sql_root_absolute' in self.d else self.root
        self.sql={n:spec['paths'][n].read_text(encoding='utf-8') for n in self.d['sql']}
        self.examples=json.loads(spec['paths'][self.d['examples']].read_text(encoding='utf-8')) if self.d.get('examples') else {}
        self.adapter={'model_sha256':MODEL_SHA,'deployment_sha256':spec['manifest_sha256'],'input_contract_version':INPUT_VERSION}
    def info(self):
        return {'version':VERSION,'model_version':'tree_history_v2.0','feature_version':'history_extended_v2.0','feature_count':71,'threshold':self.bundle['threshold']['value'],'model_loads':self.model_loads,**self.adapter,'training_rows':215257,'sample_uses':['train','validation','unlabeled_scoring'],'independent_final_test':False,'evaluation_scope':'previous_validation_exploration_only','fields':self.fields,'examples':sorted(int(k) for k in self.examples),'input_limits':{'bytes':240000,'new_history_rows':1600},'limits':['个人本机学习；许可未核验。','旧validation探索；没有新增独立最终测试。','原test不进入本模型新推理。','路径贡献非SHAP／因果／审批理由。']}
    def features(self,pop,identifiers):
        require(pop in ['labeled','unlabeled'] and isinstance(identifiers,list) and 1<=len(identifiers)<=5 and len(set(identifiers))==len(identifiers),'INVALID_INPUT')
        require(all(type(i) is int and 0<i<2**53 for i in identifiers),'INVALID_INPUT')
        parts=['train','validation'] if pop=='labeled' else ['unlabeled_scoring'];rows={}
        names=','.join('"'+n+'"' for n in ['SK_ID_CURR','sample_use',*self.names]);ids=','.join('?' for _ in identifiers)
        with duckdb.connect(':memory:') as db:
            db.execute('SET threads=1');db.execute("SET memory_limit='128MB'");db.execute("SET max_temp_directory_size='0B'")
            for part in parts:
                path=self.spec['paths'][self.d['partitions'][part]]
                cur=db.execute(f'SELECT {names} FROM read_parquet(?) WHERE SK_ID_CURR IN ({ids})',[str(path),*identifiers]);columns=[d[0] for d in cur.description]
                for record in cur.fetchall():
                    row=dict(zip(columns,record));rows[row['SK_ID_CURR']]=row
        require(all(i in rows for i in identifiers),'CASE_NOT_FOUND')
        use=[rows[i]['sample_use'] for i in identifiers];table=pa.Table.from_pylist([{n:rows[i][n] for n in self.names} for i in identifiers],schema=pa.schema([pa.field(f['name'],getattr(pa,f['type'])()) for f in self.contract['features']]))
        return table,use
    def render(self,table,identities,uses,raw_inputs=None):
        e=self.explainer.explain(table);rows=table.to_pylist();results=[]
        for j,(values,identity,use) in enumerate(zip(rows,identities,uses)):
            facts=[]
            for k,field in enumerate(self.fields):
                name=field['name'];value=values[name];s=self.reference['features'][name];index=bucket(s,value);bins=[{'label':label,'count':n,'share':n/215257,'current':a==index} for a,(label,n) in enumerate(zip(labels(s),s['reference_counts']))]
                state='observed' if value is not None else 'source_null_or_no_eligible_observation'
                prefix=name.split('_',1)[0]
                count=values.get(prefix+'_ELIGIBLE_COUNT') if prefix in ['BUREAU','PREV'] else values.get(prefix+'_PAYMENT_RECORD_COUNT' if prefix=='INST' else prefix+'_MONTH_COUNT')
                if count==0:state='no_eligible_history'
                facts.append({**field,'value':value,'missing_state':state,'contribution_log_odds':float(e['contributions'][j,k]),'reference':{'rows':215257,'missing_count':s['reference_counts'][0],'bins':bins,'current_bucket':index,'percentile':None},'fact':f"保存值：{value if value is not None else 'null'}；"+('无合格记录不等于没有借贷或信用良好。' if state=='no_eligible_history' else '来源空值与有效0分开保留。' if value is None else '按冻结口径保留。')})
            p=float(e['probability'][j]);result={'version':VERSION,'case_key':identity,'sample_use':use,'model_version':'tree_history_v2.0','feature_count':71,**self.adapter,'probability':p,'threshold':self.bundle['threshold']['value'],'risk_flag':p>=self.bundle['threshold']['value'],'facts':facts,'explanation':{'method':'training_count_weighted_path_log_odds','unit':'log_odds','base_log_odds':self.explainer.base,'contribution_sum':float(e['contributions'][j].sum()),'raw_score':float(e['raw_score'][j]),'checks':e['checks']},'reference_population':'train_only_215257','limits':self.info()['limits']}
            if raw_inputs:
                result['input_sha256']=digest(raw_inputs[j]);result['validation']=complete(raw_inputs[j])
                result['raw_history_counts']={k:len(raw_inputs[j][k]['rows']) for k in ['bureau','previous','cc','pos','bb','inst']}
            result['case_revision']=digest(result);results.append(result)
        return {'results':results,'count':len(results),'deployment_sha256':self.spec['manifest_sha256']}
    def cases(self,pop,ids):
        table,uses=self.features(pop,ids);return self.render(table,[{'population':pop,'application_id':i} for i in ids],uses)
    def build(self,payloads):
        require(isinstance(payloads,list) and 1<=len(payloads)<=5,'INVALID_INPUT')
        for p in payloads:complete(p)
        ids=[p['application']['SK_ID_CURR'] for p in payloads];require(len(set(ids))==len(ids),'INVALID_INPUT')
        common=self.raw_engine.build([raw40(p) for p in payloads])
        from .candidate71_contract import TABLES
        with duckdb.connect(':memory:') as db:
            db.execute('SET threads=1');db.execute("SET memory_limit='128MB'");db.execute("SET max_temp_directory_size='0B'")
            db.register('allowed',pa.table({'SK_ID_CURR':ids}));schema={
                'previous_raw':pa.schema([pa.field('SK_ID_CURR',pa.int64()),pa.field('SK_ID_PREV',pa.int64()),pa.field('DAYS_DECISION',pa.float64()),pa.field('NAME_CONTRACT_TYPE',pa.string())]),
                'bureau_raw':pa.schema([pa.field('SK_ID_CURR',pa.int64()),pa.field('SK_ID_BUREAU',pa.int64()),pa.field('DAYS_CREDIT',pa.float64()),pa.field('DAYS_CREDIT_UPDATE',pa.float64()),pa.field('CREDIT_CURRENCY',pa.string()),pa.field('CREDIT_ACTIVE',pa.string()),pa.field('AMT_CREDIT_SUM',pa.float64())])}
            for name,family in [('previous_raw','previous'),('bureau_raw','bureau')]:db.register(name,pa.Table.from_pylist([r for p in payloads for r in p[family]['rows']],schema=schema[name]))
            for family,names in TABLES.items():
                s=pa.schema([pa.field(n,pa.string() if n=='STATUS' else pa.int64() if n.startswith('SK_ID_') or n in ['MONTHS_BALANCE','NUM_INSTALMENT_NUMBER'] else pa.float64()) for n in names])
                db.register(family+'_input',pa.Table.from_pylist([r for p in payloads for r in p[family]['rows']],schema=s))
            for family in ['CC','POS','BB','INST']:
                # Execute the exact frozen SQL text, then call shared integrity/count checks.
                require(all((self.root/n).read_text(encoding='utf-8')==text for n,text in self.sql.items()),'VERSION_MISMATCH')
                run_family(db,self.root,family)
            join_new(db)
            cur=db.execute('SELECT * FROM new_features');columns=[d[0] for d in cur.description]
            indexed={record[0]:dict(zip(columns,record)) for record in cur.fetchall()}
        rows=common.to_pylist()
        for row,ident in zip(rows,ids):row.update({n:indexed[ident][n] for n in self.names[40:]})
        table=pa.Table.from_pylist(rows,schema=pa.schema([pa.field(f['name'],getattr(pa,f['type'])()) for f in self.contract['features']]))
        validate_features(table,self.contract);return table
    def score(self,payloads):
        table=self.build(payloads)
        # Reject reserved-test identities by the original session at dispatch before this call.
        return self.render(table,[{'population':'new_application71','application_id':p['application']['SK_ID_CURR']} for p in payloads],['new_input_prediction_only']*len(payloads),payloads)
